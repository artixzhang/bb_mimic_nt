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
        push_force_min_n=(40.0, 40.0, 10.0),
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
        push_force_min=torch.tensor([40.0, 40.0, 10.0]),
        push_force_max=torch.tensor([200.0, 200.0, 50.0]),
        push_torque=torch.zeros(3, 1, 3),
        push_target_frame=torch.full((3,), -1, dtype=torch.long),
        push_steps_left=torch.zeros(3, dtype=torch.long),
        push_used=torch.zeros(3, dtype=torch.bool),
        push_applied=torch.tensor([True, True, False]),
        ball=object(),
        robot=SimpleNamespace(has_external_wrench=True),
        ball_mass_cache=object(),
        robot_mass_cache=object(),
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
    assert torch.equal(context.push_applied, torch.tensor([False, True, False]))
    assert context.robot.has_external_wrench
    assert torch.all((context.delay_steps[[0, 2]] >= 0) & (context.delay_steps[[0, 2]] <= 4))
    assert torch.all((context.ball_mass_scale[[0, 2]] >= 0.95) & (context.ball_mass_scale[[0, 2]] <= 1.05))
    assert torch.all(context.push_force[[0, 2], 0].abs() <= context.push_force_max)
    assert torch.all(context.push_force[[0, 2], 0].abs() >= context.push_force_min)


def test_push_force_range_is_scaled_by_curriculum_strength() -> None:
    torch.manual_seed(11)
    lower = torch.tensor([10.0, 20.0, 2.0])
    upper = torch.tensor([50.0, 60.0, 10.0])
    force = events._uniform_signed(1024, lower, upper, 0.25, "cpu")
    assert torch.all(force.abs() >= lower * 0.25)
    assert torch.all(force.abs() <= upper * 0.25)
    assert torch.all(force.min(dim=0).values < 0.0)
    assert torch.all(force.max(dim=0).values > 0.0)


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


def test_mass_cache_reuses_physx_data_and_scales_selected_rows() -> None:
    class View:
        def __init__(self):
            self.masses = torch.tensor([[2.0, 4.0], [3.0, 5.0], [7.0, 11.0]])
            self.inertias = self.masses[:, :, None].expand(-1, -1, 9).clone()
            self.reads = 0

        def get_masses(self):
            self.reads += 1
            return self.masses.clone()

        def get_inertias(self):
            self.reads += 1
            return self.inertias.clone()

        def set_masses(self, values, indices):
            self.masses[indices] = values[indices]

        def set_inertias(self, values, indices):
            self.inertias[indices] = values[indices]

    view = View()
    asset = SimpleNamespace(
        root_physx_view=view,
        data=SimpleNamespace(default_mass=view.masses.clone(), default_inertia=view.inertias.clone()),
    )
    cache = events._MassCache(asset)
    events._set_masses_and_inertias(cache, torch.tensor([0, 2]), (1,), torch.tensor([0.9, 1.1]))
    assert torch.allclose(view.masses, torch.tensor([[2.0, 3.6], [3.0, 5.0], [7.0, 12.1]]))
    assert torch.allclose(view.inertias[:, :, 0], view.masses)
    events._set_masses_and_inertias(cache, torch.tensor([0]), (1,), torch.tensor([1.0]))
    assert view.masses[0, 1] == 4.0
    assert view.reads == 2


def test_friction_cache_updates_only_hand_and_foot_shapes() -> None:
    class View:
        def __init__(self):
            self.materials = torch.full((3, 4, 3), 0.4)
            self.reads = 0

        def get_material_properties(self):
            self.reads += 1
            return self.materials.clone()

        def set_material_properties(self, values, indices):
            self.materials[indices] = values[indices]

    view = View()
    context = SimpleNamespace(
        robot=SimpleNamespace(root_physx_view=view),
        materials=view.get_material_properties().clone(),
        hand_shapes=torch.tensor([1]),
        foot_shapes=torch.tensor([3]),
    )
    events._set_friction(context, torch.tensor([0, 2]), torch.tensor([[0.6, 0.8], [1.0, 1.2]]))
    assert torch.allclose(view.materials[[0, 2], 1, 0], torch.tensor([0.6, 1.0]))
    assert torch.allclose(view.materials[[0, 2], 3, 1], torch.tensor([0.8, 1.2]))
    assert torch.all(view.materials[1] == 0.4)
    assert torch.all(view.materials[:, [0, 2]] == 0.4)
    assert view.reads == 1


def test_pd_cache_keeps_simulation_and_actuator_gains_in_sync() -> None:
    class View:
        def __init__(self, stiffness, damping):
            self.stiffness = stiffness.clone()
            self.damping = damping.clone()
            self.stiffness_buffer = None

        def set_dof_stiffnesses(self, values, indices):
            self.stiffness_buffer = values
            self.stiffness[indices] = values[indices]

        def set_dof_dampings(self, values, indices):
            self.damping[indices] = values[indices]

    stiffness = torch.tensor([[100.0, 200.0]] * 3)
    damping = torch.tensor([[10.0, 20.0]] * 3)
    view = View(stiffness, damping)
    data = SimpleNamespace(
        default_joint_stiffness=stiffness.clone(),
        default_joint_damping=damping.clone(),
        joint_stiffness=stiffness.clone(),
        joint_damping=damping.clone(),
    )
    actuator = SimpleNamespace(joint_indices=torch.tensor([0, 1]), stiffness=stiffness.clone(), damping=damping.clone())
    robot = SimpleNamespace(data=data, root_physx_view=view, actuators={"all": actuator})
    context = SimpleNamespace(
        robot=robot,
        pd_scale=torch.tensor([0.9, 1.0, 1.1]),
        joint_ids=[0, 1],
        joint_ids_cpu=torch.tensor([0, 1]),
        joint_stiffness_cpu=stiffness.clone(),
        joint_damping_cpu=damping.clone(),
    )
    events._set_pd_gains(context, torch.tensor([0, 2]), torch.tensor([0, 2]))
    assert torch.allclose(view.stiffness, data.joint_stiffness)
    assert torch.allclose(view.damping, data.joint_damping)
    assert torch.allclose(actuator.stiffness, view.stiffness)
    assert torch.allclose(actuator.damping, view.damping)
    assert torch.allclose(view.stiffness[:, 0], torch.tensor([90.0, 100.0, 110.0]))
    assert torch.allclose(view.damping[:, 0], 10.0 * torch.sqrt(context.pd_scale))
    assert view.stiffness_buffer is context.joint_stiffness_cpu


def test_push_pulse_last_twenty_steps_and_availability_wins(monkeypatch) -> None:
    class Robot:
        def __init__(self):
            self.force = torch.zeros(1, 1, 3)
            self.torque = torch.zeros(1, 1, 3)
            self.has_external_wrench = False
            self.calls = 0

        def set_external_force_and_torque(self, force, torque, **kwargs):
            assert kwargs["body_ids"] == [3]
            assert kwargs["is_global"] is True
            indices = kwargs["env_ids"]
            self.force[indices] = force
            self.torque[indices] = torque
            self.calls += 1

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
        push_applied=torch.tensor([False]),
        iteration_offset=0,
    )
    env = SimpleNamespace(
        cfg=SimpleNamespace(dr=SimpleNamespace(
            push_duration_s=0.2,
            push=SimpleNamespace(enabled=True, start_fraction=0.2, end_fraction=0.6),
            steps_per_iteration=10,
            total_iterations=100,
        )),
        step_dt=0.01,
        common_step_counter=1000,
    )
    monkeypatch.setattr(events, "get_dr_context", lambda env: context)
    for remaining in range(19, -1, -1):
        events.advance_push_pulse(env, None)
        assert context.push_steps_left.item() == remaining
        assert torch.equal(robot.force, context.push_force)
        assert torch.equal(robot.torque, context.push_torque)
    assert robot.calls == 1
    events.advance_push_pulse(env, None)
    assert torch.count_nonzero(robot.force) == 0
    assert torch.count_nonzero(robot.torque) == 0
    assert robot.calls == 2
    context.push_used[:] = False
    context.push_steps_left[:] = 0
    events.advance_push_pulse(env, None)
    motion.reference["push_available"][:] = False
    events.advance_push_pulse(env, None)
    assert context.push_steps_left.item() == 0
    assert torch.count_nonzero(robot.force) == 0
    assert robot.calls == 4
