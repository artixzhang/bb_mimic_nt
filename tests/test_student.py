"""CPU contracts for Student targets, observation history, and observation delays."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from bb_mimic_nt.action_math import clip_joint_actions
from bb_mimic_nt.student import (
    STUDENT_OBSERVATION_DIM,
    STUDENT_OBSERVATION_FIELDS,
    student_history,
    student_delay_bounds,
)


def test_student_contract_and_teacher_labels_use_dimensionless_requests() -> None:
    assert sum(size for _, size in STUDENT_OBSERVATION_FIELDS) == STUDENT_OBSERVATION_DIM == 936
    assert STUDENT_OBSERVATION_FIELDS[-2] == ('previous_action_history', 290)
    raw = torch.tensor([[-300.0, .3, 150.0]])
    label = clip_joint_actions(raw)
    torch.testing.assert_close(label, torch.tensor([[-100.0, .3, 100.0]]))
    torch.testing.assert_close(raw, torch.tensor([[-300.0, .3, 150.0]]))


def test_student_history_is_per_environment_and_oldest_first() -> None:
    history = torch.arange(7 * 2, dtype=torch.float32).reshape(7, 2, 1)
    cursor = torch.tensor([5, 2])
    delay = torch.tensor([0, 2])
    selected = student_history(history, cursor, delay, length=5)
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
