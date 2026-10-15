"""CPU checks for DR sampling, observations, and action-target delay."""

from __future__ import annotations

import ast
import importlib.util
import math
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


def test_smoothstep_curriculum_and_switch() -> None:
    schedule = SimpleNamespace(enabled=True, start_fraction=0.2, end_fraction=0.6)
    assert [
        events.curriculum_strength(schedule, step, 100)
        for step in (0, 20, 30, 40, 50, 60, 100)
    ] == pytest.approx(
        [0.0, 0.0, 0.15625, 0.5, 0.84375, 1.0, 1.0]
    )
    schedule.enabled = False
    assert events.curriculum_strength(schedule, 60, 100) == 0.0


@pytest.mark.parametrize("legacy", [False, True])
def test_curriculum_logging_accepts_new_and_legacy_randomization_settings(legacy) -> None:
    path = MDP_ROOT / "curriculums.py"
    node = next(node for node in ast.parse(path.read_text()).body
                if isinstance(node, ast.FunctionDef) and node.name == "dr_curriculum")
    namespace = dict(curriculum_strength=events.curriculum_strength,
                     get_dr_context=lambda env: SimpleNamespace(iteration_offset=20))
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), namespace)
    cfg = SimpleNamespace(total_iterations=100, steps_per_iteration=10)
    for name in ("delay", "ball_mass", "pd_gains", "hand_friction", "foot_friction", "link_mass", "push", "global_push", "object_push"):
        setattr(cfg, name, SimpleNamespace(enabled=True, start_fraction=.2, end_fraction=.6))
    components = ("dof_pos", "dof_vel", "root_pos", "root_rot", "root_vel", "root_ang_vel", "object_pos")
    if not legacy:
        cfg.velocity_push = SimpleNamespace(enabled=True, full_strength=True)
        cfg.init_state = SimpleNamespace(enabled=True, **{
            name: SimpleNamespace(enabled=name != "root_rot") for name in components
        })
    env = SimpleNamespace(cfg=SimpleNamespace(dr=cfg), common_step_counter=200)
    logged = namespace["dr_curriculum"](env, None)
    assert logged["push"] == pytest.approx(.5)  # Includes the restored iteration offset.
    assert logged["velocity_push"] == (0.0 if legacy else 1.0)
    for name in components:
        assert logged[f"init_state/{name}"] == float(not legacy and name != "root_rot")
    if not legacy:
        cfg.init_state.enabled = False
        assert all(namespace["dr_curriculum"](env, None)[f"init_state/{name}"] == 0.0 for name in components)


def test_delay_selector_uses_latest_final_target_and_per_env_levels() -> None:
    select = _delay_selector()
    history = torch.arange(5 * 2 * 3, dtype=torch.float32).reshape(5, 2, 3)
    delays = torch.tensor([0, 4])
    result = select(history, 1, delays)
    assert torch.equal(result[0], history[1, 0])
    assert torch.equal(result[1], history[2, 1])


def test_zero_delay_with_single_history_slot_uses_current_target() -> None:
    select = _delay_selector()
    history = torch.arange(6, dtype=torch.float32).reshape(1, 2, 3)
    assert torch.equal(select(history, 0, torch.zeros(2, dtype=torch.long)), history[0])


@pytest.mark.parametrize("nominal, offset", [(2, 2), (1, 1), (0, 0), (1, 0), (0, 2)])
@pytest.mark.parametrize("iteration, strength", [(0, 0.0), (40, 0.5), (100, 1.0)])
def test_reset_only_changes_selected_context_rows(monkeypatch, nominal, offset, iteration, strength) -> None:
    schedule_names = ("delay", "ball_mass", "pd_gains", "hand_friction", "foot_friction", "link_mass", "push")
    schedules = {name: SimpleNamespace(enabled=True, start_fraction=0.2, end_fraction=0.6) for name in schedule_names}
    for name in schedule_names:
        schedules[name].full_strength = name != "delay"
    cfg = SimpleNamespace(total_iterations=100, steps_per_iteration=10, **_ranges(), **schedules)
    cfg.delay_nominal_steps = nominal
    cfg.delay_max_offset_steps = offset
    env = SimpleNamespace(device="cpu", common_step_counter=iteration * 10, cfg=SimpleNamespace(dr=cfg))
    context = SimpleNamespace(
        iteration_offset=0,
        delay_steps=torch.full((3,), nominal, dtype=torch.long),
        ball_mass_scale=torch.ones(3),
        pd_scale=torch.ones(3),
        hand_friction=torch.full((3,), 0.8),
        foot_friction=torch.full((3,), 0.9),
        link_mass_scale=torch.ones(3),
        ball=object(),
        robot=SimpleNamespace(has_external_wrench=True),
        ball_mass_cache=object(),
        robot_mass_cache=object(),
        link_mass_ids=(),
        velocity_push=None,
    )
    env.num_envs = 3
    env.step_dt = 0.01
    context._body_ids = lambda names: (0,)
    context.robot_push_bodies = [0]
    context.robot.set_external_force_and_torque = lambda *args, **kwargs: None
    context.ball = SimpleNamespace(set_external_force_and_torque=lambda *args, **kwargs: None)
    for name in events.PUSH_TYPES:
        setattr(cfg, name, schedules["push"])
        for suffix, value in dict(force_min_n=cfg.push_force_min_n, force_max_n=cfg.push_force_max_n,
                                  torque_max_nm=3., duration_s=.2, probability=.8,
                                  link_weights={"torso_link": 1.}).items():
            setattr(cfg, f"{name}_{suffix}", value)
    context.pushes = {name: events.PushPulse(env, context, name) for name in events.PUSH_TYPES}
    deadlines = torch.tensor([0.5, 10., 20.], dtype=torch.float64)
    for pulse in context.pushes.values():
        pulse.applied[:] = torch.tensor([True, True, False])
        pulse.next_trigger_s[:] = deadlines
    monkeypatch.setattr(events, "get_dr_context", lambda env: context)
    monkeypatch.setattr(events, "_set_masses_and_inertias", lambda *args: None)
    monkeypatch.setattr(events, "_set_pd_gains", lambda *args: None)
    monkeypatch.setattr(events, "_set_friction", lambda *args: None)
    randint = torch.randint
    sampled_bounds = []

    def record_randint(low, high, size, **kwargs):
        sampled_bounds.append((low, high))
        return randint(low, high, size, **kwargs)

    monkeypatch.setattr(torch, "randint", record_randint)
    torch.manual_seed(7)
    events.reset_domain_randomization(env, torch.tensor([0, 2]))
    assert context.delay_steps[1] == nominal
    assert context.ball_mass_scale[1] == 1.0
    assert context.pd_scale[1] == 1.0
    assert context.hand_friction[1] == pytest.approx(0.8)
    assert context.foot_friction[1] == pytest.approx(0.9)
    assert context.link_mass_scale[1] == 1.0
    assert context.pushes["push"].force[1].count_nonzero() == 0
    assert torch.equal(context.pushes["push"].applied, torch.tensor([False, True, False]))
    assert context.robot.has_external_wrench
    lower = max(0, nominal - offset)
    low = nominal - math.ceil((nominal - lower) * strength)
    high = nominal + math.ceil(offset * strength)
    assert sampled_bounds == ([] if strength == 0 or offset == 0 else [(lower, nominal + offset + 1)])
    assert torch.all(
        (context.delay_steps[[0, 2]] >= low)
        & (context.delay_steps[[0, 2]] <= high)
    )
    assert torch.all((context.ball_mass_scale[[0, 2]] >= 0.95) & (context.ball_mass_scale[[0, 2]] <= 1.05))
    assert not context.pushes["push"].force[[0, 2]].any()
    for pulse in context.pushes.values():
        assert torch.equal(pulse.next_trigger_s, deadlines)


@pytest.mark.parametrize("nominal, minimum, maximum", [(2, 0, 4), (1, 0, 3), (0, 0, 3), (0, 0, 0)])
def test_delay_sampling_has_nominal_start_and_uniform_final_distribution(nominal, minimum, maximum):
    assert events.sample_delay_steps(100, nominal, minimum, maximum, 0., "cpu").eq(nominal).all()
    torch.manual_seed(12)
    sample = events.sample_delay_steps(100000, nominal, minimum, maximum, 1., "cpu")
    frequency = torch.bincount(sample, minlength=maximum + 1).float() / len(sample)
    expected = 1 / (maximum - minimum + 1)
    assert (frequency[minimum:] - expected).abs().max() < .01


@pytest.mark.parametrize("threshold", [.5, 1.])
def test_delay_probabilities_are_continuous_across_old_integer_radius_thresholds(threshold):
    # Coupling the random draws isolates the distribution change from Monte Carlo noise.
    torch.manual_seed(15)
    left = events.sample_delay_steps(100000, 2, 0, 4, threshold - .001, "cpu")
    torch.manual_seed(15)
    right = events.sample_delay_steps(100000, 2, 0, 4, min(1., threshold + .001), "cpu")
    assert (left != right).float().mean() < .005


def test_delay_grows_from_first_fraction_and_preserves_asymmetric_expected_value():
    torch.manual_seed(18)
    early = events.sample_delay_steps(100000, 1, 0, 3, .01, "cpu")
    assert 0 < (early != 1).float().mean() < .03
    sample = events.sample_delay_steps(100000, 1, 0, 3, .3, "cpu")
    # Full-range mean 1.5 contracts toward nominal 1, without biased rounding.
    assert sample.float().mean() == pytest.approx(1.15, abs=.01)
    assert sample.min() >= 0 and sample.max() <= 3


@pytest.mark.parametrize("args", [(1, -1, 3, .5), (2, 0, 1, .5), (1.5, 0, 3, .5),
                                  (1, 0, 3, -1.), (1, 0, 3, float('nan'))])
def test_delay_sampler_rejects_invalid_configuration(args):
    with pytest.raises(ValueError, match="Delay"):
        events.sample_delay_steps(2, *args, "cpu")


def test_push_force_range_is_scaled_by_curriculum_strength() -> None:
    torch.manual_seed(11)
    lower = torch.tensor([10.0, 20.0, 2.0])
    upper = torch.tensor([50.0, 60.0, 10.0])
    force = events._uniform_signed(1024, lower, upper, 0.25, "cpu")
    assert torch.all(force.abs() >= lower * 0.25)
    assert torch.all(force.abs() <= upper * 0.25)
    assert torch.all(force.min(dim=0).values < 0.0)
    assert torch.all(force.max(dim=0).values > 0.0)


@pytest.mark.parametrize("nominal, offset", [(2, 2), (1, 1), (0, 0), (1, 0), (0, 2)])
def test_privileged_observation_order_and_nominal_values(nominal, offset) -> None:
    context = object.__new__(events.DomainRandomizationContext)
    context.cfg = SimpleNamespace(**_ranges())
    context.cfg.delay_nominal_steps = nominal
    context.cfg.delay_max_offset_steps = offset
    context.delay_steps = torch.tensor([max(0, nominal - offset), nominal, nominal + offset])
    context.ball_mass_scale = torch.tensor([0.95, 1.0, 1.05])
    context.pd_scale = torch.tensor([0.9, 1.0, 1.1])
    context.hand_friction = torch.tensor([0.6, 0.8, 1.0])
    context.foot_friction = torch.tensor([0.6, 0.9, 1.2])
    context.link_mass_scale = torch.tensor([0.9, 1.0, 1.1])
    expected = torch.tensor([[-1.0] * 6, [0.0] * 6, [1.0] * 6])
    expected[0, 0] = -min(nominal, offset) / max(offset, 1)
    if offset == 0:
        expected[:, 0] = 0.0
    assert torch.isfinite(context.privileged).all()
    assert torch.allclose(context.privileged, expected, atol=1e-6)


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
