# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Direct joint-position requests shared by Teacher and Student policies."""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import ActionTerm, ActionTermCfg
from isaaclab.utils import configclass

from bb_mimic_nt.action_math import clip_joint_actions, joint_position_target, resolve_action_scale

from .events import get_dr_context


def delayed_target(history: torch.Tensor, cursor: int, delay_steps: torch.Tensor) -> torch.Tensor:
    """Select the final target delayed by each environment's policy-step count."""
    indices = (cursor - delay_steps) % history.shape[0]
    return history[indices, torch.arange(history.shape[1], device=history.device)]


class DirectJointPositionAction(ActionTerm):
    """Request ``nominal + scale * clip(policy_output)`` independently of reference motion.

    Observations and action regularization use the current clipped request.
    Only the applied PD target passes through the per-environment delay queue.
    """

    cfg: "DirectJointPositionActionCfg"
    _asset: Articulation

    def __init__(self, cfg: "DirectJointPositionActionCfg", env):
        super().__init__(cfg, env)
        if not math.isfinite(cfg.action_clip) or cfg.action_clip <= 0.0:
            raise ValueError("action_clip must be finite and positive.")
        self._joint_ids, self._joint_names = self._asset.find_joints(cfg.joint_names, preserve_order=True)
        if tuple(self._joint_names) != tuple(cfg.joint_names):
            raise ValueError(f"Action joint order differs from configured names: {self._joint_names}")
        command = env.command_manager.get_term("motion")
        if tuple(self._joint_names) != tuple(command.dof_names):
            raise ValueError("Action and reference joint orders differ.")

        # Snapshot nominal physics before resets can randomize live PD gains.
        data = self._asset.data
        self.nominal = data.default_joint_pos[:, self._joint_ids].clone()
        self.effort_limits = data.joint_effort_limits[:, self._joint_ids].clone()
        self.joint_position_limits = data.joint_pos_limits[:, self._joint_ids].clone()
        self.velocity_limits = data.joint_vel_limits[:, self._joint_ids].clone()
        self.action_scale = resolve_action_scale(
            self.effort_limits, data.default_joint_stiffness[:, self._joint_ids], cfg.action_scale_factor
        )
        # Deployment uses the exact archived coordinates instead of recalculating
        # an old policy's mapping from the currently installed robot parameters.
        for name, attribute in (("nominal_joint_position", "nominal"), ("action_scale_values", "action_scale")):
            saved = getattr(cfg, name)
            if saved is not None:
                values = torch.as_tensor(saved, dtype=self.nominal.dtype, device=self.device)
                if values.shape != (len(self._joint_ids),) or not torch.isfinite(values).all():
                    raise ValueError(f"Deployment {name} must contain one finite value per controlled joint.")
                if attribute == "action_scale" and not torch.all(values > 0.0):
                    raise ValueError("Deployment action_scale_values must be positive.")
                setattr(self, attribute, values.expand_as(self.nominal).clone())

        self._raw_actions = torch.zeros_like(self.nominal)
        self.requested_actions = torch.zeros_like(self.nominal)
        self.previous_requested_actions = torch.zeros_like(self.nominal)
        self.requested_targets = torch.zeros_like(self.nominal)
        self.applied_targets = torch.zeros_like(self.nominal)
        self.previous_applied_targets = torch.zeros_like(self.nominal)
        history_length = env.cfg.dr.delay_nominal_steps + env.cfg.dr.delay_max_offset_steps + 1
        self._target_history = torch.zeros((history_length, *self.nominal.shape), device=self.device)
        self._history_cursor = -1
        self._dr_context = get_dr_context(env)
        self.reset_to_current_state()

    @property
    def action_dim(self) -> int:
        return len(self._joint_ids)

    @property
    def raw_actions(self) -> torch.Tensor:
        """Unmodified policy output, before clipping, scaling or delay."""
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        """The delayed target submitted to Isaac Lab's position controller."""
        return self.applied_targets

    def process_actions(self, actions: torch.Tensor) -> None:
        self.previous_requested_actions.copy_(self.requested_actions)
        self.previous_applied_targets.copy_(self.applied_targets)
        self._raw_actions.copy_(actions)
        self.requested_actions.copy_(clip_joint_actions(actions, self.cfg.action_clip))
        self.requested_targets.copy_(
            joint_position_target(self.requested_actions, self.nominal, self.action_scale, self.cfg.action_clip)
        )
        self._history_cursor = (self._history_cursor + 1) % self._target_history.shape[0]
        self._target_history[self._history_cursor].copy_(self.requested_targets)
        self.applied_targets.copy_(delayed_target(self._target_history, self._history_cursor, self._dr_context.delay_steps))

    def apply_actions(self) -> None:
        self._asset.set_joint_position_target(self.applied_targets, joint_ids=self._joint_ids)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        self.reset_to_current_state(env_ids)

    def reset_to_current_state(self, env_ids: Sequence[int] | torch.Tensor | None = None) -> None:
        """Clear requests and hold actual reset q until delayed commands arrive.

        The motion command calls this again after writing its new RSI state,
        including reset noise. A partial reset never changes other histories.
        """
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        elif not isinstance(env_ids, torch.Tensor):
            env_ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        position = self._asset.data.joint_pos[env_ids[:, None], self._joint_ids]
        self._raw_actions[env_ids] = 0.0
        self.requested_actions[env_ids] = 0.0
        self.previous_requested_actions[env_ids] = 0.0
        self.requested_targets[env_ids] = self.nominal[env_ids]
        self.applied_targets[env_ids] = position
        self.previous_applied_targets[env_ids] = position
        self._target_history[:, env_ids] = position


@configclass
class DirectJointPositionActionCfg(ActionTermCfg):
    class_type: type[ActionTerm] = DirectJointPositionAction
    joint_names: list[str] = []
    action_scale_factor: float = 0.25
    action_clip: float = 100.0
    nominal_joint_position: list[float] | None = None
    action_scale_values: list[float] | None = None
