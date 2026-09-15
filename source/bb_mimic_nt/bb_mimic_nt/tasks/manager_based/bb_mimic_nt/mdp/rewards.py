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
    gated_top_level_reward,
    gaussian,
    normalized_weighted_sum,
    rational_kernel,
)

from .commands import MotionReferenceCommand
from .contacts import contact_graph


def root_position_reward(actual: torch.Tensor, reference: torch.Tensor, sigma: float) -> torch.Tensor:
    return gaussian(torch.sum((actual - reference) ** 2, dim=-1), sigma)


def root_rotation_reward(actual: torch.Tensor, reference: torch.Tensor, sigma: float) -> torch.Tensor:
    return gaussian(quat_error_magnitude(actual, reference) ** 2, sigma)


def root_velocity_reward(actual: torch.Tensor, reference: torch.Tensor, sigma: float) -> torch.Tensor:
    return rational_kernel(torch.sum((actual - reference) ** 2, dim=-1), sigma)


def joint_position_reward(actual: torch.Tensor, reference: torch.Tensor, sigma: float) -> torch.Tensor:
    return gaussian((actual - reference) ** 2, sigma).mean(dim=-1)


def joint_velocity_reward(actual: torch.Tensor, reference: torch.Tensor, sigma: float) -> torch.Tensor:
    return rational_kernel((actual - reference) ** 2, sigma).mean(dim=-1)


def link_position_reward(actual: torch.Tensor, reference: torch.Tensor, sigma: float) -> torch.Tensor:
    return gaussian(torch.sum((actual - reference) ** 2, dim=-1), sigma).mean(dim=-1)


def link_rotation_reward(actual: torch.Tensor, reference: torch.Tensor, sigma: float) -> torch.Tensor:
    shape = actual.shape
    error = quat_error_magnitude(actual.reshape(-1, 4), reference.reshape(-1, 4)).reshape(shape[:2])
    return gaussian(error**2, sigma).mean(dim=-1)


def object_position_reward(actual: torch.Tensor, reference: torch.Tensor, sigma: float) -> torch.Tensor:
    return rational_kernel(torch.sum((actual - reference) ** 2, dim=-1), sigma)


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
    global_weights: tuple[float, float, float, float] = (0.45, 0.25, 0.20, 0.10)
    body_weights: tuple[float, float, float] = (0.30, 0.40, 0.30)
    root_weights: tuple[float, float, float, float] = (0.35, 0.25, 0.20, 0.20)
    joint_weights: tuple[float, float] = (0.70, 0.30)
    link_weights: tuple[float, float] = (0.70, 0.30)
    object_weights: tuple[float, float] = (1.0, 0.0)
    relative_weights: tuple[float, float] = (1.0, 0.0)
    root_sigmas: tuple[float, float, float, float] = (20.0, 5.0, 2.0, 0.5)
    joint_sigmas: tuple[float, float] = (4.0, 0.1)
    link_sigmas: tuple[float, float] = (40.0, 5.0)
    object_sigmas: tuple[float, float] = (20.0, 1.0)
    relative_sigmas: tuple[float, float] = (40.0, 1.0)
    hand_contact_sensitivity: float = 2.0
    foot_contact_sensitivity: float = 1.0
    hand_contact_force: float = 1.0
    foot_contact_force: float = 5.0
    action_magnitude_weight: float = 0.005
    action_rate_weight: float = 0.05
    target_residual_rate_weight: float = 0.05
    torque_weight: float = 0.005
    limit_weight: float = 0.05
    joint_velocity_error_weight: float = 0.02
    regularization_clip: float = 1.0
    termination_penalty: float = 50.0


def regularization_cost(term: MotionReferenceCommand, settings: UnifiedRewardCfg, env) -> dict[str, torch.Tensor]:
    robot = term.robot
    joint_ids = term.joint_ids
    action_term = env.action_manager.get_term("joint_pos")
    action_magnitude = torch.mean(action_term.filtered_actions**2, dim=-1)
    action_rate = torch.mean((action_term.raw_actions - action_term.previous_raw_actions) ** 2, dim=-1)
    current_residual = (action_term.processed_actions - action_term.reference_positions) / action_term.residual_scale
    previous_residual = (
        action_term.previous_processed_actions - action_term.previous_reference_positions
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
    velocity_limit = robot.data.joint_vel_limits[:, joint_ids].clamp_min(1.0e-6)
    joint_velocity_error = torch.mean(
        ((joint_vel - term.reference["dof_vel"]) / velocity_limit) ** 2, dim=-1
    )
    weighted = clipped_regularization(
        (
            action_magnitude,
            action_rate,
            target_residual_rate,
            torque_cost,
            limit_cost,
            joint_velocity_error,
        ),
        (
            settings.action_magnitude_weight,
            settings.action_rate_weight,
            settings.target_residual_rate_weight,
            settings.torque_weight,
            settings.limit_weight,
            settings.joint_velocity_error_weight,
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
        "reg/total": weighted,
    }


class UnifiedMimicReward(ManagerTermBase):
    """Compute ``mimic - regularization - termination`` and log its hierarchy."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self._episode_sums: dict[str, torch.Tensor] = {}

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
                robot.data.joint_pos[:, term.joint_ids], reference["dof_pos"], settings.joint_sigmas[0]
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
        link_parts = {
            "mimic/body/link/position": link_position_reward(
                actual_link_pos, reference["link_pos_b"], settings.link_sigmas[0]
            ),
            "mimic/body/link/rotation": link_rotation_reward(
                actual_link_quat, reference["link_quat_b"], settings.link_sigmas[1]
            ),
        }
        link = normalized_weighted_sum(tuple(link_parts.values()), settings.link_weights)
        body = normalized_weighted_sum((root, joint, link), settings.body_weights)

        object_parts = {
            "mimic/object/position": object_position_reward(
                term.ball.data.root_pos_w - origin, reference["object_pos"], settings.object_sigmas[0]
            ),
            "mimic/object/rotation": object_rotation_reward(
                term.ball.data.root_quat_w, reference["object_quat"], settings.object_sigmas[1]
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

        actual_contact = contact_graph(env, settings.hand_contact_force, settings.foot_contact_force)
        contact = contact_graph_reward(
            actual_contact,
            reference["contact"],
            settings.hand_contact_sensitivity,
            settings.foot_contact_sensitivity,
        )
        mimic = gated_top_level_reward(
            body,
            obj,
            relative,
            contact,
            term.object_reward_active.float(),
            settings.global_weights,
        )
        regularization = regularization_cost(term, settings, env)
        termination = env.termination_manager.terminated.float() * settings.termination_penalty
        total = torch.nan_to_num(
            mimic - regularization["reg/total"] - termination,
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
                "mimic/body": body,
                "mimic/object": obj,
                "mimic/relative": relative,
                "mimic/contact": contact,
                "mimic/total": mimic,
                **regularization,
                "termination": termination,
                "total": total,
            }
        )
        # RewardManager multiplies every term by dt. Divide here so the
        # externally observed per-step reward follows the requested equation.
        return total / env.step_dt
