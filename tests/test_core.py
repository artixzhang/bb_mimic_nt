# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Pure CPU tests for action, reference, reward, and shot math."""

from __future__ import annotations

import math

import pytest
import torch

from bb_mimic_nt.core import (
    advance_reference_frame,
    apply_ballistic_speed_lock,
    downward_hoop_crossing,
    gated_height_shortfall_cost,
    gated_top_level_reward,
    interpolate_motion,
    normalized_weighted_sum,
    rational_kernel,
    rsi_probability,
    select_adaptive_speed,
    slerp_wxyz,
)
from bb_mimic_nt.training import (
    PPO_MAX_ITERATIONS, PPO_STEPS_PER_ENV, RSI_DECAY_STEPS, RSI_DECAY_FRACTION,
    RSI_INITIAL_PROBABILITY, TEACHER_OBSERVATION_DIM,
)


def test_rsi_schedule_uses_central_training_length() -> None:
    assert RSI_DECAY_STEPS == int(PPO_STEPS_PER_ENV * PPO_MAX_ITERATIONS * RSI_DECAY_FRACTION)
    assert rsi_probability(0, RSI_DECAY_STEPS) == pytest.approx(RSI_INITIAL_PROBABILITY)
    assert rsi_probability(RSI_DECAY_STEPS // 2, RSI_DECAY_STEPS) == pytest.approx(RSI_INITIAL_PROBABILITY / 2)
    assert rsi_probability(RSI_DECAY_STEPS, RSI_DECAY_STEPS) == 0.0
    assert rsi_probability(RSI_DECAY_STEPS * 2, RSI_DECAY_STEPS) == 0.0


# def test_teacher_observation_contract_is_479() -> None:
#     current = 3 + 1 + 4 + 3 + 3 + 29 + 29 + 12 + 3 + 3 + 3 + 29 + 29
#     reference = 3 + 4 + 29 + 12 + 3 + 29 + 12 + 3 + 3 + 3 + 29 + 3 + 4
#     history = 3 * (3 + 29 + 29)
#     assert (current, reference, history) == (151, 137, 183)
#     assert current + reference + history + 2 + 6 == TEACHER_OBSERVATION_DIM


def test_adaptive_speed_levels_are_monotonic() -> None:
    errors = torch.tensor([0.0, 0.3, 0.8, 1.2])
    speed = select_adaptive_speed(errors, (0.5, 0.75, 1.0, 1.25))
    assert torch.equal(speed, torch.tensor([1.25, 1.0, 0.75, 0.5]))
    locked = apply_ballistic_speed_lock(
        speed, torch.tensor([79.0, 80.0, 81.0, 120.0]), torch.full((4,), 80.0)
    )
    assert torch.equal(locked, torch.tensor([1.25, 1.0, 1.0, 1.0]))

    next_frame, next_speed = advance_reference_frame(
        torch.tensor([79.75]), torch.tensor([1.25]), torch.tensor([80.0]), 100.0, 0.01
    )
    # 2 ms reaches frame 80 at 1.25x; the remaining 8 ms advances at 1x.
    assert next_frame.item() == pytest.approx(80.8)
    assert next_speed.item() == 1.0


def test_slerp_shortest_arc_and_normalization() -> None:
    identity = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    same_with_opposite_sign = -identity
    middle = slerp_wxyz(identity, same_with_opposite_sign, torch.tensor([0.5]))
    assert torch.allclose(middle, identity)

    half_turn_z = torch.tensor([[0.0, 0.0, 0.0, 1.0]])
    quarter_turn_z = slerp_wxyz(identity, half_turn_z, torch.tensor([0.5]))
    expected = torch.tensor([[math.sqrt(0.5), 0.0, 0.0, math.sqrt(0.5)]])
    assert torch.allclose(quarter_turn_z, expected, atol=1.0e-6)
    assert torch.allclose(torch.linalg.vector_norm(quarter_turn_z, dim=-1), torch.ones(1))


def test_motion_interpolation_respects_each_clip_length() -> None:
    values = torch.tensor(
        [
            [[0.0], [2.0], [4.0], [99.0]],
            [[10.0], [20.0], [30.0], [40.0]],
        ]
    )
    result = interpolate_motion(
        values,
        clip_ids=torch.tensor([0, 1]),
        frame=torch.tensor([1.5, 2.25]),
        lengths=torch.tensor([3, 4]),
    )
    assert torch.equal(result, torch.tensor([[3.0], [32.5]]))

    clamped = interpolate_motion(
        values,
        clip_ids=torch.tensor([0]),
        frame=torch.tensor([2.75]),
        lengths=torch.tensor([3, 4]),
    )
    assert torch.equal(clamped, torch.tensor([[4.0]]))


def test_reward_normalization_gating() -> None:
    one = torch.ones(3)
    root = normalized_weighted_sum((one, one, one, one), (0.35, 0.25, 0.20, 0.20))
    joint = normalized_weighted_sum((one, one), (0.70, 0.30))
    body = normalized_weighted_sum((root, joint, one), (0.30, 0.40, 0.30))
    active = gated_top_level_reward(body, one, one, one, one, (0.45, 0.25, 0.20, 0.10))
    end_hold = gated_top_level_reward(
        body, torch.zeros(3), torch.zeros(3), one, torch.zeros(3), (0.45, 0.25, 0.20, 0.10)
    )
    separate_gates = gated_top_level_reward(
        body,
        torch.zeros(3),
        torch.zeros(3),
        one,
        torch.full((3,), 0.2),
        (0.45, 0.25, 0.20, 0.10),
        relative_gate=torch.zeros(3),
    )
    assert torch.equal(active, one)
    assert torch.equal(end_hold, one)
    assert torch.allclose(separate_gates, torch.full((3,), 0.55 / 0.60))



def test_gated_height_shortfall_cost_is_one_sided_and_per_foot() -> None:
    actual = torch.tensor([[0.20, 0.10], [0.10, 0.30], [0.10, 0.10]])
    reference = torch.tensor([[0.30, 0.30], [0.30, 0.20], [0.30, 0.30]])
    contact = torch.tensor([[0.0, 0.0], [0.0, 1.0], [1.0, 1.0]])

    assert torch.allclose(
        gated_height_shortfall_cost(actual, reference, contact, allowed_shortfall=0.05),
        torch.tensor([0.20, 0.15, 0.0]),
    )

    with pytest.raises(ValueError, match="matching shapes"):
        gated_height_shortfall_cost(actual, reference[:, :1], contact, allowed_shortfall=0.0)
    with pytest.raises(ValueError, match="non-negative"):
        gated_height_shortfall_cost(actual, reference, contact, allowed_shortfall=-0.01)


def test_root_height_shortfall_requires_both_reference_feet_airborne() -> None:
    actual_root_height = torch.tensor([0.70, 0.70, 0.70])
    reference_root_height = torch.tensor([0.90, 0.90, 0.90])
    reference_foot_contact = torch.tensor([[1.0, 1.0], [1.0, 0.0], [0.0, 0.0]])
    reference_root_support = torch.amax(reference_foot_contact, dim=-1)
    cost = gated_height_shortfall_cost(
        actual_root_height[:, None],
        reference_root_height[:, None],
        reference_root_support[:, None],
        allowed_shortfall=0.02,
    )
    assert cost.tolist() == pytest.approx([0.0, 0.0, 0.18])


def test_downward_hoop_crossing_rejects_reverse_and_offset() -> None:
    previous = torch.tensor([0.1, 0.1, 0.1])
    hoop = torch.zeros((3, 3))
    ball = torch.tensor([[0.1, 0.0, -0.01], [0.1, 0.0, -0.01], [0.25, 0.0, -0.01]])
    velocity = torch.tensor([[0.0, 0.0, -1.0], [0.0, 0.0, 1.0], [0.0, 0.0, -1.0]])
    assert torch.equal(
        downward_hoop_crossing(previous, ball, hoop, velocity),
        torch.tensor([True, False, False]),
    )
