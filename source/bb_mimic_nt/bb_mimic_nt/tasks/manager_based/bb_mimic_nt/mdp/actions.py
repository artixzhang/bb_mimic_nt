# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Task-specific joint action terms."""

from __future__ import annotations

from collections.abc import Sequence

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import ActionTerm, ActionTermCfg
from isaaclab.utils import configclass

from bb_mimic_nt.core import reference_residual_target

from .events import get_dr_context


def delayed_target(history: torch.Tensor, cursor: int, delay_steps: torch.Tensor) -> torch.Tensor:
    """Select the final target delayed by each environment's policy-step count."""
    indices = (cursor - delay_steps) % history.shape[0]
    return history[indices, torch.arange(history.shape[1], device=history.device)]


class ReferenceResidualJointPositionAction(ActionTerm):
    """Apply a bounded residual around the current motion reference.

    A zero policy action requests the reference joint pose. The policy only
    learns the feedback correction required by physics and ball interaction,
    rather than reconstructing the entire feed-forward motion through a very
    large absolute action range.
    """

    cfg: "ReferenceResidualJointPositionActionCfg"
    _asset: Articulation

    def __init__(self, cfg: "ReferenceResidualJointPositionActionCfg", env):
        super().__init__(cfg, env)
        if not 0.0 < cfg.residual_scale_fraction <= 1.0:
            raise ValueError("residual_scale_fraction must be in (0, 1].")
        if cfg.minimum_residual_scale <= 0.0 or cfg.maximum_residual_scale < cfg.minimum_residual_scale:
            raise ValueError("Residual scale bounds must be positive and ordered.")
        if cfg.position_limit_margin < 0.0:
            raise ValueError("position_limit_margin must be non-negative.")
        self._joint_ids, self._joint_names = self._asset.find_joints(cfg.joint_names, preserve_order=True)
        if len(self._joint_ids) != len(cfg.joint_names):
            raise ValueError(
                f"Expected {len(cfg.joint_names)} controlled joints, resolved {len(self._joint_ids)}: {self._joint_names}"
            )
        self._command = env.command_manager.get_term(cfg.command_name)
        if tuple(self._joint_names) != tuple(self._command.dof_names):
            raise ValueError(
                f"Action/reference joint order mismatch: {self._joint_names} != {self._command.dof_names}"
            )

        self._raw_actions = torch.zeros(self.num_envs, len(self._joint_ids), device=self.device)
        self._previous_raw_actions = torch.zeros_like(self._raw_actions)
        self._desired_actions = torch.zeros_like(self._raw_actions)
        self._previous_desired_actions = torch.zeros_like(self._raw_actions)
        self._reference_positions = torch.zeros_like(self._raw_actions)
        self._previous_reference_positions = torch.zeros_like(self._raw_actions)
        history_length = env.cfg.dr.delay_nominal_steps + env.cfg.dr.delay_max_offset_steps + 1
        self._target_history = torch.zeros((history_length, self.num_envs, len(self._joint_ids)), device=self.device)
        self._history_cursor = -1
        self._nominal = self._asset.data.default_joint_pos[:, self._joint_ids].clone()
        limits = self._asset.data.joint_pos_limits[:, self._joint_ids]
        self._lower = (limits[..., 0] + cfg.position_limit_margin).clone()
        self._upper = (limits[..., 1] - cfg.position_limit_margin).clone()
        if torch.any(self._lower >= self._upper):
            raise ValueError("position_limit_margin leaves an empty target interval.")
        if torch.any(self._nominal < self._lower) or torch.any(self._nominal > self._upper):
            raise ValueError("The nominal joint pose must lie within every safe target limit.")
        reference_outside = (
            (self._command.minimum_reference_dof_pos < self._lower[0])
            | (self._command.maximum_reference_dof_pos > self._upper[0])
        )
        if torch.any(reference_outside):
            invalid_names = [
                name for name, invalid in zip(self._joint_names, reference_outside.tolist()) if invalid
            ]
            raise ValueError(
                "Reference trajectory exceeds the safe action target limits for joints: "
                f"{invalid_names}. Reduce position_limit_margin or repair the trajectory."
            )
        target_width = self._upper - self._lower
        self._residual_scale = (target_width * cfg.residual_scale_fraction).clamp(
            min=cfg.minimum_residual_scale,
            max=cfg.maximum_residual_scale,
        )
        self._dr_context = get_dr_context(env)
        self.synchronize_reference()

    @property
    def action_dim(self) -> int:
        return len(self._joint_ids)

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def previous_raw_actions(self) -> torch.Tensor:
        return self._previous_raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        # Required by Isaac Lab's ActionTerm; this is the target selected for execution.
        return self._desired_actions

    @property
    def desired_actions(self) -> torch.Tensor:
        return self._desired_actions

    @property
    def previous_desired_actions(self) -> torch.Tensor:
        return self._previous_desired_actions

    @property
    def reference_positions(self) -> torch.Tensor:
        return self._reference_positions

    @property
    def previous_reference_positions(self) -> torch.Tensor:
        return self._previous_reference_positions

    @property
    def residual_scale(self) -> torch.Tensor:
        return self._residual_scale

    def process_actions(self, actions: torch.Tensor) -> None:
        self._previous_raw_actions[:] = self._raw_actions
        self._previous_desired_actions[:] = self._desired_actions
        self._previous_reference_positions[:] = self._reference_positions
        # RSL-RL samples an unbounded Gaussian.  A smooth tanh transform avoids
        # the large dead zones created when many samples are hard-clipped at
        # +/-1 while preserving a bounded residual command.
        self._raw_actions[:] = torch.tanh(actions)
        self._reference_positions[:] = self._command.reference["dof_pos"]
        current_target = reference_residual_target(
            self._raw_actions,
            self._reference_positions,
            self._residual_scale,
            self._lower,
            self._upper,
        )
        self._history_cursor = (self._history_cursor + 1) % self._target_history.shape[0]
        self._target_history[self._history_cursor] = current_target
        self._desired_actions[:] = delayed_target(
            self._target_history, self._history_cursor, self._dr_context.delay_steps
        )

    def apply_actions(self) -> None:
        self._asset.set_joint_position_target(self._desired_actions, joint_ids=self._joint_ids)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        self.synchronize_reference(env_ids)

    def synchronize_reference(self, env_ids: Sequence[int] | torch.Tensor | None = None) -> None:
        """Synchronize action state after the motion command chooses a reset frame."""
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        elif not isinstance(env_ids, torch.Tensor):
            env_ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        reference = self._command.reference["dof_pos"][env_ids]
        reference = torch.maximum(torch.minimum(reference, self._upper[env_ids]), self._lower[env_ids])
        self._raw_actions[env_ids] = 0.0
        self._previous_raw_actions[env_ids] = 0.0
        self._reference_positions[env_ids] = reference
        self._previous_reference_positions[env_ids] = reference
        self._desired_actions[env_ids] = reference
        self._previous_desired_actions[env_ids] = reference
        self._target_history[:, env_ids] = reference


@configclass
class ReferenceResidualJointPositionActionCfg(ActionTermCfg):
    class_type: type[ActionTerm] = ReferenceResidualJointPositionAction
    joint_names: list[str] = []
    command_name: str = "motion"
    residual_scale_fraction: float = 0.50
    minimum_residual_scale: float = 0.10
    maximum_residual_scale: float = 0.80
    position_limit_margin: float = 0.020


class StudentNominalJointPositionAction(ActionTerm):
    """Apply an unfiltered joint-position offset from the robot nominal pose."""

    cfg: "StudentNominalJointPositionActionCfg"
    _asset: Articulation

    def __init__(self, cfg: "StudentNominalJointPositionActionCfg", env):
        super().__init__(cfg, env)
        if not 0.0 < cfg.residual_scale_fraction <= 1.0:
            raise ValueError("Teacher residual_scale_fraction must be in (0, 1].")
        if cfg.minimum_residual_scale <= 0.0 or cfg.maximum_residual_scale < cfg.minimum_residual_scale:
            raise ValueError("Teacher residual scale bounds must be positive and ordered.")
        self._joint_ids, self._joint_names = self._asset.find_joints(cfg.joint_names, preserve_order=True)
        self._command = env.command_manager.get_term(cfg.command_name)
        if tuple(self._joint_names) != tuple(self._command.dof_names):
            raise ValueError("Student action and Teacher reference joint orders differ.")

        shape = (self.num_envs, len(self._joint_ids))
        self._raw_actions = torch.zeros(shape, device=self.device)
        self._previous_raw_actions = torch.zeros_like(self._raw_actions)
        self._desired_actions = torch.zeros_like(self._raw_actions)
        self._previous_desired_actions = torch.zeros_like(self._raw_actions)
        self._reference_positions = torch.zeros_like(self._raw_actions)
        self._previous_reference_positions = torch.zeros_like(self._raw_actions)
        self._teacher_residual = torch.zeros_like(self._raw_actions)
        self._teacher_mask = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        self.nominal = self._asset.data.default_joint_pos[:, self._joint_ids].clone()
        limits = self._asset.data.joint_pos_limits[:, self._joint_ids]
        self.lower = (limits[..., 0] + cfg.position_limit_margin).clone()
        self.upper = (limits[..., 1] - cfg.position_limit_margin).clone()
        if torch.any(self.lower >= self.upper):
            raise ValueError("Student action safe interval is empty.")
        if torch.any(self.nominal < self.lower) or torch.any(self.nominal > self.upper):
            raise ValueError("Student nominal pose lies outside the safe interval.")
        self.residual_scale = ((self.upper - self.lower) * cfg.residual_scale_fraction).clamp(
            min=cfg.minimum_residual_scale,
            max=cfg.maximum_residual_scale,
        )
        history_length = env.cfg.dr.delay_nominal_steps + env.cfg.dr.delay_max_offset_steps + 1
        self._target_history = torch.zeros((history_length, *shape), device=self.device)
        self._history_cursor = -1
        self._dr_context = get_dr_context(env)
        self.synchronize_reference()

    @property
    def action_dim(self) -> int:
        return len(self._joint_ids)

    @property
    def raw_actions(self) -> torch.Tensor:
        """Teacher-compatible residual representation of the last requested target."""
        return self._raw_actions

    @property
    def previous_raw_actions(self) -> torch.Tensor:
        return self._previous_raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._desired_actions

    @property
    def desired_actions(self) -> torch.Tensor:
        return self._desired_actions

    @property
    def previous_desired_actions(self) -> torch.Tensor:
        return self._previous_desired_actions

    @property
    def reference_positions(self) -> torch.Tensor:
        return self._reference_positions

    @property
    def previous_reference_positions(self) -> torch.Tensor:
        return self._previous_reference_positions

    @property
    def actual_action_offset(self) -> torch.Tensor:
        """The delayed PD target expressed relative to the nominal pose."""
        return self._desired_actions - self.nominal

    def set_teacher_residual(self, raw: torch.Tensor, mask: torch.Tensor) -> None:
        """Preserve the Teacher action semantics for clean Teacher observations."""
        if raw.shape != self._teacher_residual.shape or mask.shape != self._teacher_mask.shape:
            raise ValueError("Teacher residual or execution mask has the wrong shape.")
        self._teacher_residual.copy_(torch.tanh(raw))
        self._teacher_mask.copy_(mask)

    def process_actions(self, actions: torch.Tensor) -> None:
        self._previous_raw_actions.copy_(self._raw_actions)
        self._previous_desired_actions.copy_(self._desired_actions)
        self._previous_reference_positions.copy_(self._reference_positions)
        self._reference_positions.copy_(self._command.reference["dof_pos"])

        target = torch.maximum(torch.minimum(self.nominal + actions, self.upper), self.lower)
        self._raw_actions.copy_(
            ((target - self._reference_positions) / self.residual_scale).clamp(-1.0, 1.0)
        )
        self._raw_actions[self._teacher_mask] = self._teacher_residual[self._teacher_mask]
        self._teacher_mask.zero_()

        self._history_cursor = (self._history_cursor + 1) % self._target_history.shape[0]
        self._target_history[self._history_cursor].copy_(target)
        self._desired_actions.copy_(
            delayed_target(self._target_history, self._history_cursor, self._dr_context.delay_steps)
        )

    def apply_actions(self) -> None:
        self._asset.set_joint_position_target(self._desired_actions, joint_ids=self._joint_ids)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        self.synchronize_reference(env_ids)

    def synchronize_reference(self, env_ids: Sequence[int] | torch.Tensor | None = None) -> None:
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        elif not isinstance(env_ids, torch.Tensor):
            env_ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        reference = self._command.reference["dof_pos"][env_ids]
        reference = torch.maximum(torch.minimum(reference, self.upper[env_ids]), self.lower[env_ids])
        self._raw_actions[env_ids] = 0.0
        self._previous_raw_actions[env_ids] = 0.0
        self._teacher_residual[env_ids] = 0.0
        self._teacher_mask[env_ids] = False
        self._reference_positions[env_ids] = reference
        self._previous_reference_positions[env_ids] = reference
        self._desired_actions[env_ids] = reference
        self._previous_desired_actions[env_ids] = reference
        self._target_history[:, env_ids] = reference


@configclass
class StudentNominalJointPositionActionCfg(ActionTermCfg):
    class_type: type[ActionTerm] = StudentNominalJointPositionAction
    joint_names: list[str] = []
    command_name: str = "motion"
    residual_scale_fraction: float = 0.20
    minimum_residual_scale: float = 0.10
    maximum_residual_scale: float = 0.50
    position_limit_margin: float = 0.020
