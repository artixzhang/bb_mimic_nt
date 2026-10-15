"""Hierarchical unified imitation reward.

The Isaac Lab reward manager contains exactly one active term. This class
computes and logs the complete hierarchy while the small functions below keep
every mathematical sub-reward independently testable.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch

from isaaclab.managers import ManagerTermBase
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_error_magnitude, quat_inv, quat_mul

from bb_mimic_nt.core import (
    gated_height_shortfall_cost,
    gated_top_level_reward,
    gaussian,
    normalized_weighted_sum,
    rational_kernel,
)

from .commands import MotionReferenceCommand
from .contacts import contact_graph
from .events import curriculum_strength, training_policy_steps


def root_position_reward(actual: torch.Tensor, reference: torch.Tensor, sigma: float) -> torch.Tensor:
    z_error_sq = (actual[:, 2] - reference[:, 2]) ** 2
    r_z = gaussian(z_error_sq, sigma)
    xy_error_sq = torch.sum((actual[:, :2] - reference[:, :2]) ** 2, dim=-1)
    r_xy = rational_kernel(xy_error_sq, 0.1)
    return 0.8 * r_z + 0.2 * r_xy


def root_rotation_reward(actual: torch.Tensor, reference: torch.Tensor, sigma: float) -> torch.Tensor:
    return gaussian(quat_error_magnitude(actual, reference) ** 2, sigma)


def root_velocity_reward(actual: torch.Tensor, reference: torch.Tensor, sigma: float) -> torch.Tensor:
    return rational_kernel(torch.sum((actual - reference) ** 2, dim=-1), sigma)


def joint_position_reward(
    actual: torch.Tensor, 
    reference: torch.Tensor, 
    sigma: float, 
    worst_weight: float = 0.0
) -> torch.Tensor:
    # 算全身 29 个关节的高斯跟踪分 (shape: [num_envs, 29])
    scores = gaussian((actual - reference) ** 2, sigma)
    mean_score = scores.mean(dim=-1)
    worst_score = scores.amin(dim=-1)
    # 混合返回: [全局 + 最差] 加权
    return (1.0 - worst_weight) * mean_score + worst_weight * worst_score


def joint_velocity_reward(actual: torch.Tensor, reference: torch.Tensor, sigma: float) -> torch.Tensor:
    return rational_kernel((actual - reference) ** 2, sigma).mean(dim=-1)


def _resolve_body_weights(
    body_names: tuple[str, ...], 
    overrides: dict[str, float] | None, 
    device: torch.device
) -> torch.Tensor | None:
    if not overrides:
        return None
    weights = torch.ones(len(body_names), device=device)
    for name, weight in overrides.items():
        if name in body_names:
            weights[body_names.index(name)] = weight
    return weights


def link_position_reward(
    actual: torch.Tensor, 
    reference: torch.Tensor, 
    sigma: float, 
    weights: torch.Tensor | None = None,
    worst_weight: float = 0.0
) -> torch.Tensor:
    scores = gaussian(torch.sum((actual - reference) ** 2, dim=-1), sigma)
    if weights is None:
        mean_score = scores.mean(dim=-1)
    else:
        mean_score = torch.sum(scores * weights, dim=-1) / torch.sum(weights)
    worst_score = scores.amin(dim=-1)
    return (1.0 - worst_weight) * mean_score + worst_weight * worst_score


def link_rotation_reward(
    actual: torch.Tensor, 
    reference: torch.Tensor, 
    sigma: float, 
    weights: torch.Tensor | None = None,
    worst_weight: float = 0.0
) -> torch.Tensor:
    shape = actual.shape
    error = quat_error_magnitude(actual.reshape(-1, 4), reference.reshape(-1, 4)).reshape(shape[:2])
    scores = gaussian(error**2, sigma)
    if weights is None:
        mean_score = scores.mean(dim=-1)
    else:
        mean_score = torch.sum(scores * weights, dim=-1) / torch.sum(weights)
    worst_score = scores.amin(dim=-1)
    return (1.0 - worst_weight) * mean_score + worst_weight * worst_score


def object_position_reward(actual: torch.Tensor, reference: torch.Tensor, sigma: float) -> torch.Tensor:
    return rational_kernel(torch.sum((actual - reference) ** 2, dim=-1), sigma)


def object_direction_reward(actual: torch.Tensor, reference: torch.Tensor, sigma: float) -> torch.Tensor:
    actual_norm = torch.linalg.vector_norm(actual, dim=-1)
    ref_norm = torch.linalg.vector_norm(reference, dim=-1)
    
    # 球处于静止阶段 (比如准备期 ref_speed < 0.2 m/s), 方向未定义, 直接给满分 1.0
    is_moving = ref_norm > 0.05
    
    # cos(theta) = (v1 · v2) / (|v1| * |v2|)
    denom = (actual_norm * ref_norm).clamp_min(1e-6)
    cos_theta = (torch.sum(actual * reference, dim=-1) / denom).clamp(-1.0, 1.0)

    # error^2 \sim 2*(1-cos(\theta))
    directional_error_sq = 2.0 * (1.0 - cos_theta)

    direction_score = rational_kernel(directional_error_sq, sigma)
    
    return torch.where(is_moving, direction_score, torch.ones_like(direction_score))


def object_speed_reward(actual: torch.Tensor, reference: torch.Tensor, sigma: float) -> torch.Tensor:
    actual_speed = torch.linalg.vector_norm(actual, dim=-1)
    ref_speed = torch.linalg.vector_norm(reference, dim=-1)
    speed_error_sq = (actual_speed - ref_speed) ** 2
    return rational_kernel(speed_error_sq, sigma)


def object_rotation_reward(actual: torch.Tensor, reference: torch.Tensor, sigma: float) -> torch.Tensor:
    return gaussian(quat_error_magnitude(actual, reference) ** 2, sigma)


def relative_position_reward(actual: torch.Tensor, reference: torch.Tensor, sigma: float) -> torch.Tensor:
    return rational_kernel(torch.sum((actual - reference) ** 2, dim=-1), sigma).mean(dim=-1)


def contact_graph_reward(
    actual: torch.Tensor, reference: torch.Tensor, hand_sensitivity: float, foot_sensitivity: float, hand_gate: torch.Tensor | None = None
) -> torch.Tensor:
    sensitivity = actual.new_tensor((hand_sensitivity, hand_sensitivity, foot_sensitivity, foot_sensitivity))
    error = sensitivity * torch.abs(actual - reference)
    if hand_gate is not None:
        error[:, :2] *= hand_gate[:, None]
    return torch.exp(-torch.sum(error, dim=-1))


@configclass
class ObjectRewardCurriculumCfg:
    """Smoothly tighten object tracking kernels over training progress."""

    enabled: bool = True
    start_fraction: float = 0.0
    end_fraction: float = 0.60
    full_strength: bool = False
    # position, direction, speed_magnitude, rotation
    initial_sigmas: tuple[float, float, float, float] = (8.0, 6.0, 0.5, 1.0)
    final_sigmas: tuple[float, float, float, float] = (20.0, 10.0, 5.0, 1.0)


def object_reward_sigmas(settings, env) -> tuple[tuple[float, ...], float]:
    """Interpolate to the configured final sigmas with ``3t² - 2t³``."""
    schedule = settings.object_curriculum
    if not schedule.enabled:
        return schedule.final_sigmas, 1.0
    if len(schedule.initial_sigmas) != 4 or len(schedule.final_sigmas) != 4:
        raise ValueError("Object reward curriculum requires four initial and final sigmas.")
    if any(value <= 0.0 for value in (*schedule.initial_sigmas, *schedule.final_sigmas)):
        raise ValueError("Object reward curriculum sigmas must be positive.")
    iteration = training_policy_steps(env) / env.cfg.dr.steps_per_iteration
    strength = curriculum_strength(schedule, iteration, env.cfg.dr.total_iterations)
    sigmas = tuple(
        initial + strength * (final - initial)
        for initial, final in zip(schedule.initial_sigmas, schedule.final_sigmas)
    )
    return sigmas, strength


@configclass
class ControlRegularizationCfg:
    """HumanX control costs, adapted to this task's unit-scale tracking reward.

    ``normalization`` is the selected HumanX positive reward weight sum, not
    a joint count or a time-step multiplier. Disabling the curriculum selects
    its final parameters; ``full_strength`` does the same for evaluation.
    """

    action_rate_weight: float = 0.5
    torque_weight: float = 1.0e-5
    position_limit_weight: float = 10.0
    velocity_limit_weight: float = 5.0
    torque_limit_weight: float = 5.0
    normalization: float = 16.05

    enabled: bool = True
    start_fraction: float = 0.10
    end_fraction: float = 0.60
    full_strength: bool = False
    initial_strength: float = 0.10
    final_strength: float = 1.0
    soft_position_limit_initial: float = 0.95
    soft_position_limit_final: float = 0.90
    soft_velocity_limit_initial: float = 0.96
    soft_velocity_limit_final: float = 0.95
    soft_torque_limit_initial: float = 0.95
    soft_torque_limit_final: float = 0.85

    def __post_init__(self) -> None:
        weights = (
            self.action_rate_weight, self.torque_weight, self.position_limit_weight,
            self.velocity_limit_weight, self.torque_limit_weight,
        )
        if any(not math.isfinite(value) or value < 0.0 for value in weights):
            raise ValueError("Control regularization weights must be finite and non-negative.")
        if not math.isfinite(self.normalization) or self.normalization <= 0.0:
            raise ValueError("Control regularization normalization must be positive and finite.")
        if not 0.0 <= self.start_fraction < self.end_fraction <= 1.0:
            raise ValueError("Control regularization schedule must satisfy 0 <= start < end <= 1.")
        if not 0.0 <= self.initial_strength <= self.final_strength <= 1.0:
            raise ValueError("Control regularization strengths must satisfy 0 <= initial <= final <= 1.")
        for initial, final in (
            (self.soft_position_limit_initial, self.soft_position_limit_final),
            (self.soft_velocity_limit_initial, self.soft_velocity_limit_final),
            (self.soft_torque_limit_initial, self.soft_torque_limit_final),
        ):
            if not 0.0 < final <= initial <= 1.0:
                raise ValueError("Control soft limits must satisfy 0 < final <= initial <= 1.")


def control_regularization_parameters(settings: ControlRegularizationCfg, env) -> tuple[float, ...]:
    """Return progress, penalty strength, and the three active soft-limit factors."""
    if not settings.enabled or settings.full_strength:
        progress = 1.0
    else:
        iteration = training_policy_steps(env) / env.cfg.dr.steps_per_iteration
        progress = curriculum_strength(settings, iteration, env.cfg.dr.total_iterations)
    pairs = (
        (settings.initial_strength, settings.final_strength),
        (settings.soft_position_limit_initial, settings.soft_position_limit_final),
        (settings.soft_velocity_limit_initial, settings.soft_velocity_limit_final),
        (settings.soft_torque_limit_initial, settings.soft_torque_limit_final),
    )
    return (progress, *(initial + progress * (final - initial) for initial, final in pairs))


@configclass
class UnifiedRewardCfg:
    # body, obj, relative, contact
    # original (0.45, 0.20, 0.20, 0.15)
    global_weights: tuple[float, float, float, float] = (0.35, 0.35, 0.15, 0.15)
    # root, joint, link, rotation_only
    body_weights: tuple[float, float, float, float] = (0.30, 0.30, 0.30, 0.10)
    # position, rotation, linear_velocity, angular_velocity
    root_weights: tuple[float, float, float, float] = (0.50, 0.30, 0.15, 0.05)
    # position, velocity
    joint_weights: tuple[float, float] = (0.70, 0.30)
    # position, rotation
    link_weights: tuple[float, float] = (0.50, 0.50)

    # position, direction, speed_magnitude, rotation
    object_weights: tuple[float, float, float, float] = (0.50, 0.30, 0.2, 0.0)
    object_curriculum: ObjectRewardCurriculumCfg = ObjectRewardCurriculumCfg()
    object_free_weight: float = 1.0
    relative_free_weight: float = 0.3

    link_pos_body_weights: dict[str, float] = {
        "left_hand": 2.0, 
        "right_hand": 2.0, 
        "left_ankle_roll_link": 2.0,
        "right_ankle_roll_link": 2.0,
    }
    link_rot_body_weights: dict[str, float] = {
        "left_hand": 2.0, 
        "right_hand": 2.0,
        "left_ankle_roll_link": 3.0,
        "right_ankle_roll_link": 3.0,
    }

    relative_weights: tuple[float, float] = (1.0, 0.0)
    root_sigmas: tuple[float, float, float, float] = (40.0, 10.0, 2.0, 2.0)
    joint_sigmas: tuple[float, float] = (4.0, 0.1)
    joint_worst_weight: float = 0.30
    link_sigmas: tuple[float, float] = (40.0, 5.0)
    link_pos_worst_weight: float = 0.30
    link_rot_worst_weight: float = 0.30
    rotation_only_sigma: float = 20.0
    relative_sigmas: tuple[float, float] = (40.0, 1.0)

    hand_contact_distance: float = 0.02
    hand_contact_sensitivity: float = 1.0
    foot_contact_sensitivity: float = 1.0
    hand_contact_force: float = 1.0
    foot_contact_force: float = 5.0
    foot_airborne_height_tolerance: float = 0.0
    foot_airborne_height_weight: float = 0.2
    root_airborne_height_tolerance: float = 0.0
    root_airborne_height_weight: float = 2.0
    
    regularization: ControlRegularizationCfg = ControlRegularizationCfg()
    termination_penalty: float = 100.0


def regularization_cost(term: MotionReferenceCommand, settings: UnifiedRewardCfg, env) -> dict[str, torch.Tensor]:
    """Compute uncapped HumanX costs in physical units and requested-action space.

    Actions are clipped, dimensionless requests before control delay. Limits
    are the action term's nominal snapshots, in the same joint order. Every
    term sums over joints; only velocity/torque excess is capped per joint as
    in HumanX's soft-limit curriculum. No independent ``dt`` scaling applies.
    """
    robot = term.robot
    joint_ids = term.joint_ids
    action_term = env.action_manager.get_term("joint_pos")
    cfg = settings.regularization
    progress, strength, position_factor, velocity_factor, torque_factor = control_regularization_parameters(cfg, env)
    action_rate = torch.sum((action_term.requested_actions - action_term.previous_requested_actions) ** 2, dim=-1)
    torque = robot.data.applied_torque[:, joint_ids]
    joint_pos = robot.data.joint_pos[:, joint_ids]
    limits = action_term.joint_position_limits
    midpoint = 0.5 * (limits[..., 0] + limits[..., 1])
    half_range = 0.5 * (limits[..., 1] - limits[..., 0])
    lower = midpoint - position_factor * half_range
    upper = midpoint + position_factor * half_range
    joint_vel = robot.data.joint_vel[:, joint_ids]
    costs = {
        "action_rate": action_rate,
        "torque": torch.sum(torque**2, dim=-1),
        "position_limit": torch.sum(torch.relu(lower - joint_pos) + torch.relu(joint_pos - upper), dim=-1),
        "velocity_limit": torch.sum(
            (joint_vel.abs() - velocity_factor * action_term.velocity_limits).clamp(0.0, 1.0), dim=-1
        ),
        "torque_limit": torch.sum(
            (torque.abs() - torque_factor * action_term.effort_limits).clamp(0.0, 1.0), dim=-1
        ),
    }
    weighted = {
        name: value * getattr(cfg, f"{name}_weight") * strength / cfg.normalization
        for name, value in costs.items()
    }
    total = sum(weighted.values())
    return {
        **{f"reg/{name}": value for name, value in costs.items()},
        **{f"reg/weighted/{name}": value for name, value in weighted.items()},
        "reg/total": total,
        "regularization_curriculum/progress": total.new_full(total.shape, progress),
        "regularization_curriculum/strength": total.new_full(total.shape, strength),
        "regularization_curriculum/soft_position_limit": total.new_full(total.shape, position_factor),
        "regularization_curriculum/soft_velocity_limit": total.new_full(total.shape, velocity_factor),
        "regularization_curriculum/soft_torque_limit": total.new_full(total.shape, torque_factor),
    }


class UnifiedMimicReward(ManagerTermBase):
    """Compute ``mimic - regularization - termination`` and log its hierarchy."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self._episode_sums: dict[str, torch.Tensor] = {}
        self._body_weight_cache: dict[str, tuple[tuple, torch.Tensor | None]] = {}

    def _body_weights(self, kind: str, body_names: tuple[str, ...], overrides) -> torch.Tensor | None:
        # Include values in the key so interactive edits of reward settings
        # still take effect, without rebuilding CUDA tensors every step.
        key = (tuple(body_names), tuple(overrides.items()) if overrides else ())
        cached = self._body_weight_cache.get(kind)
        if cached is None or cached[0] != key:
            with torch.inference_mode(False):
                weights = _resolve_body_weights(body_names, overrides, self.device)
            self._body_weight_cache[kind] = (key, weights)
        return self._body_weight_cache[kind][1]

    def _record(self, values: dict[str, torch.Tensor]) -> None:
        for name, value in values.items():
            if name not in self._episode_sums:
                # Interactive policy inference may call env.step from an
                # inference-mode context. Keep mutable episode state as a
                # normal tensor so a later manual reset can clear it safely.
                with torch.inference_mode(False):
                    self._episode_sums[name] = torch.zeros(self.num_envs, device=self.device)
            self._episode_sums[name] += value.detach() * self._env.step_dt

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        duration = (
            self._env.episode_length_buf[env_ids].float() * self._env.step_dt
        ).clamp_min(self._env.step_dt)
        for name, total in self._episode_sums.items():
            self._env.extras["log"][f"Reward/{name}"] = (total[env_ids] / duration).mean()
            total[env_ids] = 0.0

    def __call__(self, env, settings: UnifiedRewardCfg, command_name: str = "motion") -> torch.Tensor:
        term: MotionReferenceCommand = env.command_manager.get_term(command_name)
        term.update_metrics()
        reference = term.reference
        robot = term.robot
        origin = env.scene.env_origins

        root_parts = {
            "mimic/body/root/position": root_position_reward(
                robot.data.root_link_pos_w - origin, reference["root_pos"], settings.root_sigmas[0]
            ),
            "mimic/body/root/rotation": root_rotation_reward(
                robot.data.root_link_quat_w, reference["root_quat"], settings.root_sigmas[1]
            ),
            "mimic/body/root/linear_velocity": root_velocity_reward(
                robot.data.root_link_lin_vel_w, reference["root_lin_vel"], settings.root_sigmas[2]
            ),
            "mimic/body/root/angular_velocity": root_velocity_reward(
                robot.data.root_link_ang_vel_w, reference["root_ang_vel"], settings.root_sigmas[3]
            ),
        }
        root = normalized_weighted_sum(tuple(root_parts.values()), settings.root_weights)

        joint_parts = {
            "mimic/body/joint/position": joint_position_reward(
                robot.data.joint_pos[:, term.joint_ids],
                reference["dof_pos"],
                settings.joint_sigmas[0],
                settings.joint_worst_weight
            ),
            "mimic/body/joint/velocity": joint_velocity_reward(
                robot.data.joint_vel[:, term.joint_ids], reference["dof_vel"], settings.joint_sigmas[1]
            ),
        }
        joint = normalized_weighted_sum(tuple(joint_parts.values()), settings.joint_weights)

        actual_link_pos = term.actual_link_pos_b()
        root_quat = robot.data.root_link_quat_w[:, None].expand(-1, len(term.body_ids), -1)
        body_quat = robot.data.body_quat_w[:, term.body_ids]
        actual_link_quat = quat_mul(
            quat_inv(root_quat.reshape(-1, 4)), body_quat.reshape(-1, 4)
        ).reshape_as(body_quat)

        link_pos_weights = self._body_weights("position", term.tracked_body_names, settings.link_pos_body_weights)
        link_rot_weights = self._body_weights("rotation", term.tracked_body_names, settings.link_rot_body_weights)

        link_parts = {
            "mimic/body/link/position": link_position_reward(
                actual_link_pos,
                reference["link_pos_b"],
                settings.link_sigmas[0],
                weights=link_pos_weights,
                worst_weight=settings.link_pos_worst_weight,
            ),
            "mimic/body/link/rotation": link_rotation_reward(
                actual_link_quat,
                reference["link_quat_b"],
                settings.link_sigmas[1],
                weights=link_rot_weights,
                worst_weight=settings.link_rot_worst_weight,
            ),
        }
        
        link = normalized_weighted_sum(tuple(link_parts.values()), settings.link_weights)
        rotation_body_quat = robot.data.body_quat_w[:, term.rotation_body_ids]
        rotation_root_quat = robot.data.root_link_quat_w[:, None].expand_as(rotation_body_quat)
        rotation_quat_b = quat_mul(
            quat_inv(rotation_root_quat.reshape(-1, 4)), rotation_body_quat.reshape(-1, 4)
        ).reshape_as(rotation_body_quat)
        rotation_only = link_rotation_reward(
            rotation_quat_b, reference["rotation_quat_b"], settings.rotation_only_sigma,
            term.rotation_body_weights,
        )
        body = normalized_weighted_sum((root, joint, link, rotation_only), settings.body_weights)

        object_sigmas, object_curriculum_strength = object_reward_sigmas(settings, env)
        object_parts = {
            "mimic/object/position": object_position_reward(
                term.ball.data.root_pos_w - origin, reference["object_pos"], object_sigmas[0]
            ),
            "mimic/object/direction": object_direction_reward(
                term.ball.data.root_lin_vel_w, reference["object_lin_vel"], object_sigmas[1]
            ),
            "mimic/object/speed": object_speed_reward(
                term.ball.data.root_lin_vel_w, reference["object_lin_vel"], object_sigmas[2]
            ),
            "mimic/object/rotation": object_rotation_reward(
                term.ball.data.root_quat_w, reference["object_quat"], object_sigmas[3]
            ),
        }
        obj = normalized_weighted_sum(tuple(object_parts.values()), settings.object_weights)

        relative_parts = {
            "mimic/relative/position": relative_position_reward(
                term.actual_anchor_pos_w() - term.ball.data.root_pos_w[:, None],
                reference["anchor_object_rel_pos_w"],
                settings.relative_sigmas[0],
            ),
            "mimic/relative/rotation": torch.ones(self.num_envs, device=self.device),
        }
        relative = normalized_weighted_sum(tuple(relative_parts.values()), settings.relative_weights)

        actual_contact = contact_graph(
            env,
            settings.hand_contact_force,
            settings.foot_contact_force,
            settings.hand_contact_distance,
            command_name=command_name,
        )
        contact = contact_graph_reward(
            actual_contact,
            reference["contact"],
            settings.hand_contact_sensitivity,
            settings.foot_contact_sensitivity,
            hand_gate=term.object_reward_active.float(),
        )
        foot_link_indices = tuple(
            term.tracked_body_names.index(name)
            for name in ("left_ankle_roll_link", "right_ankle_roll_link")
        )
        foot_contact_indices = tuple(
            term.contact_names.index(name)
            for name in ("left_foot_ground", "right_foot_ground")
        )
        foot_body_ids = [term.body_ids[index] for index in foot_link_indices]
        actual_foot_height = robot.data.body_pos_w[:, foot_body_ids, 2]
        reference_foot_height = (
            reference["link_pos_w"][:, foot_link_indices] + origin[:, None]
        )[:, :, 2]
        reference_foot_contact = reference["contact"][:, foot_contact_indices]
        foot_airborne_height_shortfall = gated_height_shortfall_cost(
            actual_foot_height,
            reference_foot_height,
            reference_foot_contact,
            settings.foot_airborne_height_tolerance,
        )
        foot_airborne_height_penalty = settings.foot_airborne_height_weight * foot_airborne_height_shortfall
        # Treat the root as supported whenever either reference foot is on the
        # ground. The one-sided shortfall is therefore active only during the
        # reference flight interval and cannot be satisfied by merely tucking the
        # legs while leaving the pelvis low.
        reference_root_support = torch.amax(reference_foot_contact, dim=-1)
        actual_root_height = robot.data.root_link_pos_w[:, 2] - origin[:, 2]
        root_airborne_height_shortfall = gated_height_shortfall_cost(
            actual_root_height[:, None],
            reference["root_pos"][:, 2, None],
            reference_root_support[:, None],
            settings.root_airborne_height_tolerance,
        )
        root_airborne_height_penalty = settings.root_airborne_height_weight * root_airborne_height_shortfall
        object_control = term.object_control_active.float()
        reward_active = term.object_reward_active.float()
        object_gate = reward_active * (
            object_control + (1.0 - object_control) * settings.object_free_weight
        )
        relative_gate = reward_active * (
            object_control + (1.0 - object_control) * settings.relative_free_weight
        )
        mimic = gated_top_level_reward(
            body,
            obj,
            relative,
            contact,
            object_gate,
            settings.global_weights,
            relative_gate=relative_gate,
        )
        regularization = regularization_cost(term, settings, env)
        termination = env.termination_manager.terminated.float() * settings.termination_penalty
        total = torch.nan_to_num(
            mimic
            - regularization["reg/total"]
            - foot_airborne_height_penalty
            - root_airborne_height_penalty
            - termination,
            nan=-settings.termination_penalty,
            posinf=-settings.termination_penalty,
            neginf=-settings.termination_penalty,
        )

        self._record(
            {
                **root_parts,
                **joint_parts,
                **link_parts,
                **object_parts,
                "object_curriculum/strength": obj.new_full(obj.shape, object_curriculum_strength),
                **{
                    f"object_curriculum/sigma_{name}": obj.new_full(obj.shape, sigma)
                    for name, sigma in zip(("position", "direction", "speed", "rotation"), object_sigmas)
                },
                **relative_parts,
                "mimic/body/root": root,
                "mimic/body/joint": joint,
                "mimic/body/link": link,
                "mimic/body/rotation_only": rotation_only,
                "mimic/body": body,
                "mimic/object": obj,
                "mimic/object/gate": object_gate,
                "mimic/relative": relative,
                "mimic/relative/gate": relative_gate,
                "mimic/contact": contact,
                "mimic/total": mimic,
                "penalty/foot_airborne_height": foot_airborne_height_penalty,
                "penalty/root_airborne_height": root_airborne_height_penalty,
                **regularization,
                "termination": termination,
                "total": total,
            }
        )
        # RewardManager multiplies every term by dt. Divide here so the
        # externally observed per-step reward follows the requested equation.
        return total / env.step_dt
