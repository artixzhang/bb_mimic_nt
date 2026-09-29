"""CPU contracts for Student targets, observation history, DAgger mixing, and replay."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from bb_mimic_nt.core import reference_residual_target
from bb_mimic_nt.student import (
    STUDENT_OBSERVATION_DIM,
    STUDENT_OBSERVATION_FIELDS,
    StudentReplay,
    student_execution_probability,
    student_history,
    student_delay_bounds,
    student_mix_actions,
    student_teacher_target,
)


def test_student_contract_and_teacher_target_equivalence() -> None:
    assert sum(size for _, size in STUDENT_OBSERVATION_FIELDS) == STUDENT_OBSERVATION_DIM == 469
    raw = torch.tensor([[-3.0, 0.0, 3.0]])
    reference = torch.tensor([[0.3, 0.0, -0.4]])
    scale = torch.tensor([[0.5, 0.5, 0.5]])
    lower = torch.tensor([[0.0, -0.2, -0.5]])
    upper = torch.tensor([[0.6, 0.2, 0.0]])
    nominal = torch.tensor([[0.1, 0.0, -0.1]])
    label = student_teacher_target(raw, reference, scale, lower, upper, nominal)
    expected = reference_residual_target(torch.tanh(raw), reference, scale, lower, upper) - nominal
    torch.testing.assert_close(label, expected)


def test_student_history_is_per_environment_and_oldest_first() -> None:
    history = torch.arange(7 * 2, dtype=torch.float32).reshape(7, 2, 1)
    cursor = torch.tensor([5, 2])
    delay = torch.tensor([0, 2])
    selected = student_history(history, cursor, delay)
    assert selected[:, :, 0].tolist() == [[2.0, 4.0, 6.0, 8.0, 10.0], [7.0, 9.0, 11.0, 13.0, 1.0]]


@pytest.mark.parametrize("nominal, minimum, maximum", [(1, 0, 3), (0, 0, 3), (0, 0, 0)])
def test_student_delay_curriculum_has_valid_endpoints(nominal, minimum, maximum) -> None:
    assert student_delay_bounds(nominal, minimum, maximum, 0.0) == (nominal, nominal)
    assert student_delay_bounds(nominal, minimum, maximum, 1.0) == (minimum, maximum)
    for strength in (0.01, 0.25, 0.5, 0.75, 0.99):
        low, high = student_delay_bounds(nominal, minimum, maximum, strength)
        assert minimum <= low <= nominal <= high <= maximum


@pytest.mark.parametrize("bounds", [(-1, 0, 3), (1, -1, 3), (4, 0, 3), (1.5, 0, 3)])
def test_student_delay_rejects_invalid_configuration(bounds) -> None:
    with pytest.raises(ValueError, match="delay"):
        student_delay_bounds(*bounds)


def test_replay_reservoir_updates_original_storage_and_last_duplicate_wins(monkeypatch) -> None:
    replay = StudentReplay(8, observation_dim=1, action_dim=1)
    initial = torch.arange(4.).reshape(-1, 1)
    replay.add(initial, initial + 100)
    monkeypatch.setattr(torch, "rand", lambda count, **kwargs: torch.tensor([0., .1, .9, 0., .99, .99]))
    incoming = torch.arange(10., 16.).reshape(-1, 1)
    replay.add(incoming, incoming + 100)
    # Destinations are 0, 0, 6, 0, 8, 9: the last accepted write replaces slot 0.
    assert replay.observations[:, 0].tolist() == [13., 1., 2., 3.]
    assert replay.labels[:, 0].tolist() == [113., 101., 102., 103.]
    assert replay.recent_observations[:, 0].tolist() == [12., 13., 14., 15.]
    assert replay.seen == 10


def test_replay_history_is_not_frozen_after_initial_fill() -> None:
    replay = StudentReplay(8, observation_dim=1, action_dim=1)
    initial = torch.arange(4.).reshape(-1, 1)
    replay.add(initial, initial)
    torch.manual_seed(0)
    replay.add(torch.full((1000, 1), 99.), torch.full((1000, 1), 99.))
    assert torch.all(replay.observations == 99.)
    assert torch.all(replay.labels == 99.)


def test_student_execution_schedule_and_policy_mix() -> None:
    assert student_execution_probability(0, 100) == 0.0
    assert student_execution_probability(20, 100) == 0.0
    assert student_execution_probability(35, 100) == pytest.approx(0.5)
    assert student_execution_probability(50, 100) == 1.0
    teacher, student = torch.ones(4, 29), torch.zeros(4, 29)
    mixed, mask = student_mix_actions(teacher, student, 0.0)
    assert not mask.any() and torch.equal(mixed, teacher)
    mixed, mask = student_mix_actions(teacher, student, 1.0)
    assert mask.all() and torch.equal(mixed, student)


def test_student_uses_full_physical_dr_from_first_iteration() -> None:
    path = (
        Path(__file__).resolve().parents[1]
        / "source/bb_mimic_nt/bb_mimic_nt/tasks/manager_based/bb_mimic_nt/mdp/events.py"
    )
    spec = importlib.util.spec_from_file_location("student_dr_events", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    schedule = SimpleNamespace(enabled=True, start_fraction=0.2, end_fraction=0.6, full_strength=True)
    assert module.curriculum_strength(schedule, 0, 100) == 1.0
    schedule.full_strength = False
    assert [module.curriculum_strength(schedule, step, 100) for step in (0, 20, 40, 60)] == pytest.approx(
        [0.0, 0.0, 0.5, 1.0]
    )


def test_student_replay_is_bounded_and_restorable() -> None:
    torch.manual_seed(3)
    replay = StudentReplay(7, observation_dim=2, action_dim=1)
    observations = torch.arange(50.0).reshape(25, 2)
    labels = torch.arange(25.0).reshape(25, 1)
    replay.add(observations, labels)
    assert replay.size == 7 and replay.seen == 25
    assert replay.recent_size == replay.recent_capacity == 4
    sampled_observation, sampled_label = replay.sample(20)
    torch.testing.assert_close(sampled_observation[:, 0] / 2, sampled_label[:, 0])
    recent_observation, recent_label = replay.sample_recent(20)
    torch.testing.assert_close(recent_observation[:, 0] / 2, recent_label[:, 0])
    assert set(recent_label[:, 0].tolist()).issubset({21.0, 22.0, 23.0, 24.0})
    restored = StudentReplay(7, observation_dim=2, action_dim=1)
    restored.load_state_dict(replay.state_dict())
    torch.testing.assert_close(restored.observations, replay.observations)
    torch.testing.assert_close(restored.recent_observations, replay.recent_observations)
    assert restored.seen == 25
