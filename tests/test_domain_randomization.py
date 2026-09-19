"""CPU checks for DR sampling, observations, and action-target delay."""

from __future__ import annotations

import ast
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


MDP_ROOT = (
    Path(__file__).resolve().parents[1]
    / "source/bb_mimic_nt/bb_mimic_nt/tasks/manager_based/bb_mimic_nt/mdp"
)


def _ranges():
    return dict(
        delay_nominal_steps=2,
        delay_max_offset_steps=2,
        ball_mass_fraction=0.05,
        pd_gain_fraction=0.10,
        hand_friction_nominal=0.8,
        hand_friction_delta=0.2,
        foot_friction_nominal=0.9,
        foot_friction_delta=0.3,
        link_mass_fraction=0.10,
        push_force_max_n=(200.0, 200.0, 50.0),
        push_torque_max_nm=3.0,
    )


spec = importlib.util.spec_from_file_location("task_dr_events", MDP_ROOT / "events.py")
events = importlib.util.module_from_spec(spec)
spec.loader.exec_module(events)


def _delay_selector():
    # The action module imports Isaac Sim; compile its small pure selector for CPU testing.
    source = (MDP_ROOT / "actions.py").read_text(encoding="utf-8")
    node = next(node for node in ast.parse(source).body if isinstance(node, ast.FunctionDef) and node.name == "delayed_target")
    module = ast.Module(body=[node], type_ignores=[])
    namespace = {"torch": torch}
    exec(compile(module, str(MDP_ROOT / "actions.py"), "exec"), namespace)
    return namespace["delayed_target"]


def test_linear_curriculum_and_switch() -> None:
    schedule = SimpleNamespace(enabled=True, start_fraction=0.2, end_fraction=0.6)
    assert [events.curriculum_strength(schedule, step, 100) for step in (0, 20, 40, 60, 100)] == pytest.approx(
        [0.0, 0.0, 0.5, 1.0, 1.0]
    )
    schedule.enabled = False
    assert events.curriculum_strength(schedule, 60, 100) == 0.0


def test_delay_selector_uses_latest_final_target_and_per_env_levels() -> None:
    select = _delay_selector()
    history = torch.arange(5 * 2 * 3, dtype=torch.float32).reshape(5, 2, 3)
    delays = torch.tensor([0, 4])
    result = select(history, 1, delays)
    assert torch.equal(result[0], history[1, 0])
    assert torch.equal(result[1], history[2, 1])


def test_reset_only_changes_selected_context_rows(monkeypatch) -> None:
    schedule_names = ("delay", "ball_mass", "pd_gains", "hand_friction", "foot_friction", "link_mass", "push")
    schedules = {name: SimpleNamespace(enabled=True, start_fraction=0.2, end_fraction=0.6) for name in schedule_names}
    cfg = SimpleNamespace(total_iterations=100, steps_per_iteration=10, **_ranges(), **schedules)
    env = SimpleNamespace(device="cpu", common_step_counter=1000, cfg=SimpleNamespace(dr=cfg))
    context = SimpleNamespace(
        iteration_offset=0,
        delay_steps=torch.full((3,), 2, dtype=torch.long),
        ball_mass_scale=torch.ones(3),
        pd_scale=torch.ones(3),
        hand_friction=torch.full((3,), 0.8),
        foot_friction=torch.full((3,), 0.9),
        link_mass_scale=torch.ones(3),
        push_force=torch.zeros(3, 1, 3),
        push_force_max=torch.tensor([200.0, 200.0, 50.0]),
        push_torque=torch.zeros(3, 1, 3),
        push_target_frame=torch.full((3,), -1, dtype=torch.long),
        push_steps_left=torch.zeros(3, dtype=torch.long),
        push_used=torch.zeros(3, dtype=torch.bool),
        ball=object(),
        robot=object(),
        link_mass_ids=(),
    )
    monkeypatch.setattr(events, "get_dr_context", lambda env: context)
    monkeypatch.setattr(events, "_set_masses_and_inertias", lambda *args: None)
    monkeypatch.setattr(events, "_set_pd_gains", lambda *args: None)
    monkeypatch.setattr(events, "_set_friction", lambda *args: None)
    torch.manual_seed(7)
    events.reset_domain_randomization(env, torch.tensor([0, 2]))
    assert context.delay_steps[1] == 2
    assert context.ball_mass_scale[1] == 1.0
    assert context.pd_scale[1] == 1.0
    assert context.hand_friction[1] == pytest.approx(0.8)
    assert context.foot_friction[1] == pytest.approx(0.9)
    assert context.link_mass_scale[1] == 1.0
    assert context.push_force[1].count_nonzero() == 0
    assert torch.all((context.delay_steps[[0, 2]] >= 0) & (context.delay_steps[[0, 2]] <= 4))
    assert torch.all((context.ball_mass_scale[[0, 2]] >= 0.95) & (context.ball_mass_scale[[0, 2]] <= 1.05))
    assert torch.all(context.push_force[[0, 2], 0].abs() <= context.push_force_max)
    assert torch.all(context.push_force[[0, 2], 0].abs().amax(dim=0) > context.push_force_max * 0.1)


def test_privileged_observation_order_and_nominal_values() -> None:
    context = object.__new__(events.DomainRandomizationContext)
    context.cfg = SimpleNamespace(**_ranges())
    context.delay_steps = torch.tensor([0, 2, 4])
    context.ball_mass_scale = torch.tensor([0.95, 1.0, 1.05])
    context.pd_scale = torch.tensor([0.9, 1.0, 1.1])
    context.hand_friction = torch.tensor([0.6, 0.8, 1.0])
    context.foot_friction = torch.tensor([0.6, 0.9, 1.2])
    context.link_mass_scale = torch.tensor([0.9, 1.0, 1.1])
    assert torch.allclose(context.privileged, torch.tensor([[-1.0] * 6, [0.0] * 6, [1.0] * 6]), atol=1e-6)


def test_push_pulse_last_twenty_steps_and_availability_wins(monkeypatch) -> None:
    class Robot:
        def __init__(self):
            self.force = None
            self.torque = None

        def set_external_force_and_torque(self, force, torque, **kwargs):
            assert kwargs == {"body_ids": [3], "is_global": True}
            self.force = force.clone()
            self.torque = torque.clone()

    robot = Robot()
    motion = SimpleNamespace(frame=torch.tensor([0.0]), reference={"push_available": torch.tensor([True])})
    context = SimpleNamespace(
        motion=motion,
        robot=robot,
        torso_id=3,
        push_force=torch.tensor([[[10.0, -5.0, 2.0]]]),
        push_torque=torch.tensor([[[1.0, -2.0, 3.0]]]),
        push_target_frame=torch.tensor([0]),
        push_steps_left=torch.tensor([0]),
        push_used=torch.tensor([False]),
    )
    env = SimpleNamespace(cfg=SimpleNamespace(dr=SimpleNamespace(push_duration_s=0.2)), step_dt=0.01)
    monkeypatch.setattr(events, "get_dr_context", lambda env: context)
    for remaining in range(19, -1, -1):
        events.advance_push_pulse(env, None)
        assert context.push_steps_left.item() == remaining
        assert torch.equal(robot.force, context.push_force)
        assert torch.equal(robot.torque, context.push_torque)
    events.advance_push_pulse(env, None)
    assert torch.count_nonzero(robot.force) == 0
    assert torch.count_nonzero(robot.torque) == 0
    context.push_used[:] = False
    context.push_steps_left[:] = 0
    events.advance_push_pulse(env, None)
    motion.reference["push_available"][:] = False
    events.advance_push_pulse(env, None)
    assert context.push_steps_left.item() == 0
    assert torch.count_nonzero(robot.force) == 0
