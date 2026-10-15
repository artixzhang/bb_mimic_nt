"""Shared direct joint-action math for simulation, distillation and deployment."""

from __future__ import annotations

import math

import torch


LINEAR_VELOCITY_OBSERVATION_SCALE = 2.0
ANGULAR_VELOCITY_OBSERVATION_SCALE = 0.25
JOINT_POSITION_OBSERVATION_SCALE = 1.0
JOINT_VELOCITY_OBSERVATION_SCALE = 0.05
ACTION_OBSERVATION_SCALE = 1.0


def clip_joint_actions(actions: torch.Tensor, clip: float = 100.0) -> torch.Tensor:
    """Bound dimensionless requests before scaling, without changing their input."""
    if not math.isfinite(clip) or clip <= 0.0:
        raise ValueError("Action clip must be finite and positive.")
    return actions.clamp(-clip, clip)


def joint_position_target(
    actions: torch.Tensor,
    nominal: torch.Tensor,
    action_scale: torch.Tensor,
    clip: float = 100.0,
) -> torch.Tensor:
    """Map dimensionless requests to PD angles; physical limits are not a target clamp."""
    return nominal + action_scale * clip_joint_actions(actions, clip)


def resolve_action_scale(
    effort: torch.Tensor, stiffness: torch.Tensor, factor: float = 0.25
) -> torch.Tensor:
    """Compute HumanX's action scale once from nominal actuator effort and stiffness."""
    if not math.isfinite(factor) or factor <= 0.0:
        raise ValueError("Action scale factor must be finite and positive.")
    if effort.shape != stiffness.shape:
        raise ValueError("Effort and stiffness must have matching shapes.")
    if not torch.isfinite(effort).all() or not torch.all(effort > 0.0):
        raise ValueError("Nominal effort limits must be finite and positive.")
    if not torch.isfinite(stiffness).all() or not torch.all(stiffness > 0.0):
        raise ValueError("Nominal joint stiffness must be finite and positive.")
    scale = factor * effort / stiffness
    if not torch.isfinite(scale).all() or not torch.all(scale > 0.0):
        raise ValueError("Resolved action scale must be finite and positive.")
    return scale
