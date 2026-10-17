"""Tensor-only contracts used by Student PPO+BC training and deployment."""

from __future__ import annotations

import math

import torch


STUDENT_HISTORY_LENGTH = 10
STUDENT_ACTION_DIM = 29


def student_observation_fields(history_length: int, command_dimension: int = 3) -> tuple[tuple[str, int], ...]:
    if not isinstance(history_length, int) or history_length < 1:
        raise ValueError("Student history length must be a positive integer.")
    if type(command_dimension) is not int or command_dimension < 1:
        raise ValueError("Command dimension must be a positive integer.")
    return (
        ("skill_command", command_dimension),
        ("projected_gravity_history", 3 * history_length),
        ("pelvis_angular_velocity_history", 3 * history_length),
        ("joint_position_offset_history", 29 * history_length),
        ("joint_velocity_history", 29 * history_length),
        ("previous_action_history", 29 * history_length),
        ("hoop_pos_cyl", 3),
    )


STUDENT_OBSERVATION_FIELDS = student_observation_fields(STUDENT_HISTORY_LENGTH)
STUDENT_OBSERVATION_DIM = sum(size for _, size in STUDENT_OBSERVATION_FIELDS)


def student_delay_bounds(nominal: int, minimum: int, maximum: int, strength: float = 1.0) -> tuple[int, int]:
    """Return reachable support bounds after interpolation and stochastic rounding.

    These bounds describe support only; sampling uniformly between them would
    reintroduce abrupt curriculum transitions. Sample from the full final range.
    """
    if any(not isinstance(value, int) for value in (nominal, minimum, maximum)):
        raise ValueError("Student observation delay steps must be integers.")
    if not 0 <= minimum <= nominal <= maximum:
        raise ValueError("Student observation delay must satisfy 0 <= minimum <= nominal <= maximum.")
    if not math.isfinite(strength) or not 0.0 <= strength <= 1.0:
        raise ValueError("Student observation curriculum strength must be in [0, 1].")
    low = nominal - math.ceil((nominal - minimum) * strength)
    high = nominal + math.ceil((maximum - nominal) * strength)
    return low, high


def student_history(
    history: torch.Tensor,
    cursor: torch.Tensor,
    delay: torch.Tensor,
    length: int = STUDENT_HISTORY_LENGTH,
) -> torch.Tensor:
    """Select per-environment history frames in oldest-to-newest order."""
    if length < 1 or history.ndim != 3:
        raise ValueError("History must be [time, environment, feature] with a positive output length.")
    env_ids = torch.arange(history.shape[1], device=history.device)
    return torch.stack(
        [history[(cursor - delay - age) % history.shape[0], env_ids] for age in range(length - 1, -1, -1)],
        dim=1,
    )
