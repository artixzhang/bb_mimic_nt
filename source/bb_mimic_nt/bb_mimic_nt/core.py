# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Pure PyTorch task math shared by Isaac terms and CPU unit tests."""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch


def reference_residual_target(
    action: torch.Tensor,
    reference: torch.Tensor,
    residual_scale: torch.Tensor,
    lower: torch.Tensor,
    upper: torch.Tensor,
) -> torch.Tensor:
    """Convert a normalized residual action into a safe reference-relative target."""
    desired = reference + action.clamp(-1.0, 1.0) * residual_scale
    return torch.maximum(torch.minimum(desired, upper), lower)


def rate_limit_target(
    desired: torch.Tensor,
    previous: torch.Tensor,
    maximum_velocity: torch.Tensor,
    dt: float,
) -> torch.Tensor:
    """Limit position-target motion to a physically meaningful velocity."""
    maximum_delta = maximum_velocity * float(dt)
    delta = torch.maximum(torch.minimum(desired - previous, maximum_delta), -maximum_delta)
    return previous + delta


def low_pass_filter(
    value: torch.Tensor,
    previous: torch.Tensor,
    dt: float,
    time_constant: float,
) -> torch.Tensor:
    """Apply a first-order low-pass filter with a time-step invariant pole."""
    if time_constant <= 0.0:
        return value
    alpha = 1.0 - math.exp(-float(dt) / float(time_constant))
    return previous + alpha * (value - previous)


def slerp_wxyz(first: torch.Tensor, second: torch.Tensor, blend: torch.Tensor) -> torch.Tensor:
    """Shortest-arc quaternion interpolation for batched WXYZ tensors."""
    dot = torch.sum(first * second, dim=-1, keepdim=True)
    second = torch.where(dot < 0.0, -second, second)
    dot = torch.sum(first * second, dim=-1, keepdim=True).clamp(-1.0, 1.0)
    blend = blend.view((-1,) + (1,) * (first.ndim - 1))
    theta = torch.acos(dot)
    sin_theta = torch.sin(theta)
    near = sin_theta.abs() < 1.0e-6
    denominator = torch.where(near, torch.ones_like(sin_theta), sin_theta)
    interpolated = (
        torch.sin((1.0 - blend) * theta) / denominator * first
        + torch.sin(blend * theta) / denominator * second
    )
    linear = first + blend * (second - first)
    return torch.nn.functional.normalize(torch.where(near, linear, interpolated), dim=-1)


def interpolate_motion(
    values: torch.Tensor,
    clip_ids: torch.Tensor,
    frame: torch.Tensor,
    lengths: torch.Tensor,
    *,
    quaternion: bool = False,
) -> torch.Tensor:
    """Interpolate padded motion tensors without ever indexing padding."""
    last = lengths[clip_ids] - 1
    lower = torch.minimum(torch.floor(frame).long().clamp_min(0), last)
    upper = torch.minimum(lower + 1, last)
    blend = frame - lower.float()
    first = values[clip_ids, lower]
    second = values[clip_ids, upper]
    if quaternion:
        return slerp_wxyz(first, second, blend)
    blend_shape = (-1,) + (1,) * (first.ndim - 1)
    return torch.lerp(first, second, blend.view(blend_shape))


def rsi_probability(step: int, decay_steps: int, initial_probability: float = 0.8) -> float:
    if decay_steps <= 0:
        return 0.0
    return initial_probability * max(0.0, 1.0 - float(step) / float(decay_steps))


def select_adaptive_speed(error: torch.Tensor, choices: tuple[float, ...]) -> torch.Tensor:
    """Select one of four speed levels from a normalized tracking error."""
    if len(choices) != 4 or tuple(choices) != tuple(sorted(choices)):
        raise ValueError("Adaptive speed requires four ascending choices.")
    values = torch.as_tensor(choices, device=error.device, dtype=error.dtype)
    return torch.where(
        error < 0.25,
        values[3],
        torch.where(error < 0.65, values[2], torch.where(error < 1.0, values[1], values[0])),
    )


def apply_ballistic_speed_lock(
    selected_speed: torch.Tensor, frame: torch.Tensor, last_hand_contact_frame: torch.Tensor
) -> torch.Tensor:
    """Lock reference time to 1x after both reference hands have released."""
    return torch.where(frame > last_hand_contact_frame, torch.ones_like(selected_speed), selected_speed)


def advance_reference_frame(
    frame: torch.Tensor,
    selected_speed: torch.Tensor,
    last_hand_contact_frame: torch.Tensor,
    frame_rate: float,
    dt: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Advance reference time without stretching the post-release flight.

    If a policy interval straddles the release instant, the interval portion
    before release uses the adaptive speed and the remaining portion uses 1x.
    """
    speed = apply_ballistic_speed_lock(selected_speed, frame, last_hand_contact_frame)
    proposed = frame + speed * frame_rate * dt
    crosses_release = (frame <= last_hand_contact_frame) & (proposed > last_hand_contact_frame)
    safe_rate = (speed * frame_rate).clamp_min(1.0e-8)
    time_to_release = ((last_hand_contact_frame - frame) / safe_rate).clamp(0.0, dt)
    post_release_frame = last_hand_contact_frame + (dt - time_to_release) * frame_rate
    next_frame = torch.where(crosses_release, post_release_frame, proposed)
    next_speed = torch.where(crosses_release, torch.ones_like(speed), speed)
    return next_frame, next_speed


def gaussian(error_squared: torch.Tensor, sigma: float) -> torch.Tensor:
    return torch.exp(-sigma * error_squared)


def rational_kernel(error_squared: torch.Tensor, sigma: float) -> torch.Tensor:
    """Bounded tracking kernel that retains resolution for large errors."""
    return 1.0 / (1.0 + sigma * error_squared)


def normalized_weighted_sum(values: Sequence[torch.Tensor], weights: Sequence[float]) -> torch.Tensor:
    positive = [(value, float(weight)) for value, weight in zip(values, weights) if weight > 0.0]
    if not positive:
        raise ValueError("At least one reward weight must be positive.")
    denominator = sum(weight for _, weight in positive)
    return sum(value * (weight / denominator) for value, weight in positive)


def gated_top_level_reward(
    body: torch.Tensor,
    obj: torch.Tensor,
    relative: torch.Tensor,
    contact: torch.Tensor,
    gate: torch.Tensor,
    weights: Sequence[float],
) -> torch.Tensor:
    values = torch.stack((body, obj, relative, contact), dim=-1)
    weight = values.new_tensor(weights).expand_as(values).clone()
    weight[:, 1] *= gate
    weight[:, 2] *= gate
    weight = torch.where(weight > 0.0, weight, torch.zeros_like(weight))
    return torch.sum(values * weight, dim=-1) / weight.sum(dim=-1).clamp_min(1.0e-8)


def clipped_regularization(
    costs: Sequence[torch.Tensor], weights: Sequence[float], maximum: float = 0.20
) -> torch.Tensor:
    return sum(cost * weight for cost, weight in zip(costs, weights)).clamp_max(maximum)


def downward_hoop_crossing(
    previous_height: torch.Tensor,
    ball_position: torch.Tensor,
    hoop_position: torch.Tensor,
    ball_linear_velocity: torch.Tensor,
    radius: float = 0.20,
) -> torch.Tensor:
    """Detect a downward crossing of the hoop plane inside the rim radius."""
    height = ball_position[:, 2] - hoop_position[:, 2]
    horizontal = torch.linalg.vector_norm(ball_position[:, :2] - hoop_position[:, :2], dim=-1)
    return (
        (previous_height > 0.0)
        & (height <= 0.0)
        & (ball_linear_velocity[:, 2] < 0.0)
        & (horizontal <= radius)
    )
