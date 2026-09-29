"""Hierarchical unified imitation reward.

The Isaac Lab reward manager contains exactly one active term. This class
computes and logs the complete hierarchy while the small functions below keep
every mathematical sub-reward independently testable.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch

from isaaclab.managers import ManagerTermBase
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_error_magnitude, quat_inv, quat_mul

from bb_mimic_nt.core import (
    clipped_regularization,
    gated_height_shortfall_cost,
    gated_top_level_reward,
    gaussian,
    joint_jerk_cost,
    normalized_weighted_sum,
    rational_kernel,
)

from .commands import MotionReferenceCommand
from .contacts import contact_graph


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
    is_moving = ref_norm > 0.2
    
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
    actual: torch.Tensor, reference: torch.Tensor, hand_sensitivity: float, foot_sensitivity: float
) -> torch.Tensor:
    sensitivity = actual.new_tensor((hand_sensitivity, hand_sensitivity, foot_sensitivity, foot_sensitivity))
    return torch.exp(-torch.sum(sensitivity * torch.abs(actual - reference), dim=-1))


@configclass
class UnifiedRewardCfg:
    # body, obj, relative, contact
    global_weights: tuple[float, float, float, float] = (0.45, 0.25, 0.25, 0.15)
    # root, joint, link, rotation_only
    body_weights: tuple[float, float, float, float] = (0.30, 0.30, 0.30, 0.10)
    # position, rotation, linear_velocity, angular_velocity
    root_weights: tuple[float, float, float, float] = (0.50, 0.30, 0.15, 0.05)
    # position, velocity
    joint_weights: tuple[float, float] = (0.70, 0.30)
    # position, rotation
    link_weights: tuple[float, float] = (0.50, 0.50)

    # position, direction, speed_magnitude, rotation
    object_weights: tuple[float, float, float, float] = (0.45, 0.35, 0.2, 0.0)
    object_sigmas: tuple[float, float, float, float] = (10.0, 5.0, 0.5, 1.0)
    object_free_phase_weight: float = 0.15
    relative_free_phase_weight: float = 0.0

    link_pos_body_weights: dict[str, float] = {
        "left_hand": 2.0, 
        "right_hand": 2.0, 
        "left_ankle_roll_link": 2.0,
        "right_ankle_roll_link": 2.0,
    }
    link_rot_body_weights: dict[str, float] = {
        "left_hand": 2.0, 
        "right_hand": 2.0,
        "left_ankle_roll_link": 2.0,
        "right_ankle_roll_link": 2.0,
    }

    relative_weights: tuple[float, float] = (1.0, 0.0)
    root_sigmas: tuple[float, float, float, float] = (40.0, 10.0, 2.0, 0.5)
    joint_sigmas: tuple[float, float] = (4.0, 0.1)
    joint_worst_weight: float = 0.30
    link_sigmas: tuple[float, float] = (40.0, 5.0)
    link_pos_worst_weight: float = 0.30
    link_rot_worst_weight: float = 0.30
    rotation_only_sigma: float = 5.0
    relative_sigmas: tuple[float, float] = (40.0, 1.0)

    hand_contact_distance: float = 0.02
    hand_contact_sensitivity: float = 1.0
    foot_contact_sensitivity: float = 1.0
    hand_contact_force: float = 1.0
    foot_contact_force: float = 5.0
    foot_airborne_height_tolerance: float = 0.0
    foot_airborne_height_weight: float = 0.0
    root_airborne_height_tolerance: float = 0.0
    root_airborne_height_weight: float = 1.2
    
    action_magnitude_weight: float = 0.05
    action_rate_weight: float = 0.20
    target_residual_rate_weight: float = 0.05
    torque_weight: float = 1.0e-5
    limit_weight: float = 0.05
    joint_velocity_error_weight: float = 0.001
    joint_jerk_weight: float = 1.0e-11
    regularization_clip: float = 0.5
    termination_penalty: float = 100.0


def regularization_cost(
    term: MotionReferenceCommand, settings: UnifiedRewardCfg, env, joint_jerk: torch.Tensor
) -> dict[str, torch.Tensor]:
    robot = term.robot
    joint_ids = term.joint_ids
    action_term = env.action_manager.get_term("joint_pos")
    action_magnitude = torch.mean(action_term.raw_actions**2, dim=-1)
    action_rate = torch.mean((action_term.raw_actions - action_term.previous_raw_actions) ** 2, dim=-1)
    # Measure the target residual after position clipping; away from the limits this equals action_rate.
    current_residual = (action_term.desired_actions - action_term.reference_positions) / action_term.residual_scale
    previous_residual = (
        action_term.previous_desired_actions - action_term.previous_reference_positions
    ) / action_term.residual_scale
    target_residual_rate = torch.mean((current_residual - previous_residual) ** 2, dim=-1)

    torque = robot.data.applied_torque[:, joint_ids]
    effort_limit = robot.data.joint_effort_limits[:, joint_ids].clamp_min(1.0e-6)
    torque_cost = torch.mean((torque / effort_limit) ** 2, dim=-1)

    joint_pos = robot.data.joint_pos[:, joint_ids]
    limits = robot.data.joint_pos_limits[:, joint_ids]
    midpoint = 0.5 * (limits[..., 0] + limits[..., 1])
    half_range = (0.5 * (limits[..., 1] - limits[..., 0])).clamp_min(1.0e-6)
    normalized_pos = torch.abs((joint_pos - midpoint) / half_range)
    limit_cost = torch.mean((torch.relu(normalized_pos - 0.9) / 0.1) ** 2, dim=-1)

    joint_vel = robot.data.joint_vel[:, joint_ids]
    joint_velocity_error = torch.mean((joint_vel - term.reference["dof_vel"]) ** 2, dim=-1)

    weighted = clipped_regularization(
        (
            action_magnitude,
            action_rate,
            target_residual_rate,
            torque_cost,
            limit_cost,
            joint_velocity_error,
            joint_jerk,
        ),
        (
            settings.action_magnitude_weight,
            settings.action_rate_weight,
            settings.target_residual_rate_weight,
            settings.torque_weight,
            settings.limit_weight,
            settings.joint_velocity_error_weight,
            settings.joint_jerk_weight,
        ),
        settings.regularization_clip,
    )
    return {
        "reg/action_magnitude": action_magnitude,
        "reg/action_rate": action_rate,
        "reg/target_residual_rate": target_residual_rate,
        "reg/torque": torque_cost,
        "reg/limit": limit_cost,
        "reg/joint_velocity_error": joint_velocity_error,
        "reg/joint_jerk": joint_jerk,
        "reg/total": weighted,
    }


class UnifiedMimicReward(ManagerTermBase):
    """Compute ``mimic - regularization - termination`` and log its hierarchy."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self._episode_sums: dict[str, torch.Tensor] = {}
        self._previous_joint_velocity: torch.Tensor | None = None
        self._previous_joint_acceleration: torch.Tensor | None = None
        self._joint_history_length = torch.zeros(self.num_envs, dtype=torch.uint8, device=self.device)

    def _joint_jerk(self, velocity: torch.Tensor) -> torch.Tensor:
        if self._previous_joint_velocity is None:
            with torch.inference_mode(False):
                self._previous_joint_velocity = torch.zeros_like(velocity)
                self._previous_joint_acceleration = torch.zeros_like(velocity)
        cost, acceleration = joint_jerk_cost(
            velocity,
            self._previous_joint_velocity,
            self._previous_joint_acceleration,
            self._joint_history_length,
            self._env.step_dt,
        )
        self._previous_joint_velocity.copy_(velocity)
        self._previous_joint_acceleration.copy_(
            torch.where(self._joint_history_length[:, None] > 0, acceleration, 0.0)
        )
        self._joint_history_length.add_(1).clamp_(max=2)
        return cost

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
        self._joint_history_length[env_ids] = 0
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

        link_pos_weights = _resolve_body_weights(term.tracked_body_names, settings.link_pos_body_weights, self.device)
        link_rot_weights = _resolve_body_weights(term.tracked_body_names, settings.link_rot_body_weights, self.device)

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

        object_parts = {
            "mimic/object/position": object_position_reward(
                term.ball.data.root_pos_w - origin, reference["object_pos"], settings.object_sigmas[0]
            ),
            "mimic/object/direction": object_direction_reward(
                term.ball.data.root_lin_vel_w, reference["object_lin_vel"], settings.object_sigmas[1]
            ),
            "mimic/object/speed": object_speed_reward(
                term.ball.data.root_lin_vel_w, reference["object_lin_vel"], settings.object_sigmas[2]
            ),
            "mimic/object/rotation": object_rotation_reward(
                term.ball.data.root_quat_w, reference["object_quat"], settings.object_sigmas[3]
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
        # reference flight phase and cannot be satisfied by merely tucking the
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
        reward_active = term.motion_reward_active.float()
        object_gate = reward_active * (
            object_control + (1.0 - object_control) * settings.object_free_phase_weight
        )
        relative_gate = reward_active * (
            object_control + (1.0 - object_control) * settings.relative_free_phase_weight
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
        joint_jerk = self._joint_jerk(robot.data.joint_vel[:, term.joint_ids])
        regularization = regularization_cost(term, settings, env, joint_jerk)
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
