"""HumanX reset distributions and command/history integration without Isaac Sim."""

import ast
import copy
import importlib.util
import math
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from test_skills import make_command
from test_student_observations import make_sensor


MDP = Path(__file__).resolve().parents[1] / "source/bb_mimic_nt/bb_mimic_nt/tasks/manager_based/bb_mimic_nt/mdp"
spec = importlib.util.spec_from_file_location("reset_initialization", MDP / "initialization.py")
initialization = importlib.util.module_from_spec(spec)
spec.loader.exec_module(initialization)


def init_config(enabled=True):
    return SimpleNamespace(
        enabled=enabled,
        dof_pos=SimpleNamespace(enabled=True, std=.1),
        dof_vel=SimpleNamespace(enabled=True, std=.01),
        root_pos=SimpleNamespace(enabled=True, std=.02),
        root_rot=SimpleNamespace(enabled=True, max_angle_deg=10.),
        root_vel=SimpleNamespace(enabled=True, std=.05),
        root_ang_vel=SimpleNamespace(enabled=True, std=.02),
        object_pos=SimpleNamespace(enabled=True, half_range_m=.1),
    )


def reference_state(count):
    reference = {
        name: torch.zeros(count, 29 if name.startswith("dof_") else 3)
        for name in ("root_pos", "root_lin_vel", "root_ang_vel", "dof_pos", "dof_vel",
                     "object_pos", "object_lin_vel", "object_ang_vel")
    }
    for name in ("root_quat", "object_quat"):
        reference[name] = torch.tensor([1., 0., 0., 0.]).repeat(count, 1)
    reference["object_pos"][:, 2] = 1.2
    return reference


def test_humanx_gaussian_uniform_and_rotation_distributions():
    torch.manual_seed(81)
    count = 20000
    reference = reference_state(count)
    cfg = init_config()
    sampled = initialization.sample_initial_state(reference, torch.arange(count), cfg)
    for component, field in initialization._GAUSSIAN_FIELDS.items():
        scale = getattr(cfg, component).std
        noise = (sampled[field] - reference[field]).flatten()
        assert abs(noise.mean()) < .02 * scale
        assert abs(noise.std() / scale - 1.) < .03
        # Gaussian tails distinguish this from the old bounded uniform noise.
        assert .035 < (noise.abs() > 2. * scale).float().mean() < .055
    xy = sampled["object_pos"][:, :2]
    assert xy.min() >= -.1 and xy.max() <= .1
    assert abs(xy.mean()) < .002
    assert abs(xy.std() - .1 / math.sqrt(3.)) < .002
    for field in ("object_quat", "object_lin_vel", "object_ang_vel"):
        assert torch.equal(sampled[field], reference[field])
    assert torch.equal(sampled["object_pos"][:, 2], reference["object_pos"][:, 2])
    quat = sampled["root_quat"]
    torch.testing.assert_close(torch.linalg.vector_norm(quat, dim=-1), torch.ones(count))
    angle = 2 * torch.atan2(torch.linalg.vector_norm(quat[:, 1:], dim=-1), quat[:, 0])
    assert angle.min() >= 0. and angle.max() <= math.radians(10.)
    assert abs(angle.mean() - math.radians(5.)) < math.radians(.1)
    axis = torch.nn.functional.normalize(quat[:, 1:], dim=-1)
    assert (axis.mean(dim=0).abs() < .015).all()
    torch.testing.assert_close((axis ** 2).mean(dim=0), torch.full((3,), 1 / 3), atol=.01, rtol=0)


@pytest.mark.parametrize("component, field", [*initialization._GAUSSIAN_FIELDS.items(),
                                            ("root_rot", "root_quat"), ("object_pos", "object_pos")])
def test_each_component_has_an_independent_switch(component, field):
    cfg = init_config()
    for name in vars(cfg):
        if name != "enabled":
            getattr(cfg, name).enabled = name == component
    reference = reference_state(8)
    untouched = copy.deepcopy(reference)
    ids = torch.tensor([2, 6])
    sampled = initialization.sample_initial_state(reference, ids, cfg)
    assert not torch.equal(sampled[field], reference[field][ids])
    for name in sampled:
        if name != field:
            assert torch.equal(sampled[name], reference[name][ids])
        assert torch.equal(reference[name], untouched[name])


@pytest.mark.parametrize("cfg", [None, init_config(False)])
def test_missing_or_disabled_configuration_returns_independent_reference_copies(cfg):
    reference = reference_state(3)
    original = copy.deepcopy(reference)
    ids = torch.tensor([2, 0])
    state = initialization.sample_initial_state(reference, ids, cfg)
    for name, value in state.items():
        assert torch.equal(value, reference[name][ids])
        value.fill_(42.)
        assert torch.equal(reference[name], original[name])


def test_root_noise_left_multiplies_wxyz_reference(monkeypatch):
    cfg = init_config()
    for component in initialization._GAUSSIAN_FIELDS:
        getattr(cfg, component).enabled = False
    cfg.object_pos.enabled = False
    reference = reference_state(1)
    # A reference rotation about Y and noise about X distinguish multiplication order.
    a = math.sqrt(.5)
    reference["root_quat"][0] = torch.tensor([a, 0., a, 0.])
    monkeypatch.setattr(torch, "randn_like", lambda value: torch.tensor([[1., 0., 0.]]).expand_as(value))
    monkeypatch.setattr(torch, "rand_like", lambda value: torch.full_like(value, .5))
    result = initialization.sample_initial_state(reference, torch.tensor([0]), cfg)
    c, s = math.cos(math.radians(2.5)), math.sin(math.radians(2.5))
    torch.testing.assert_close(result["root_quat"], torch.tensor([[a*c, a*s, a*c, a*s]]))


@pytest.mark.parametrize("component, field", [(name, "std") for name in initialization._GAUSSIAN_FIELDS]
                                            + [("root_rot", "max_angle_deg"), ("object_pos", "half_range_m")])
@pytest.mark.parametrize("invalid", [-.1, float("nan"), float("inf")])
def test_invalid_noise_amplitudes_rejected_even_when_disabled(component, field, invalid):
    cfg = init_config(False)
    getattr(cfg, component).enabled = False
    setattr(getattr(cfg, component), field, invalid)
    with pytest.raises(ValueError, match=f"init_state.{component}.{field}"):
        initialization.sample_initial_state(reference_state(1), torch.tensor([0]), cfg)


def write_reference_method():
    cls = next(node for node in ast.parse((MDP / "commands.py").read_text()).body
               if isinstance(node, ast.ClassDef) and node.name == "MotionReferenceCommand")
    method = next(node for node in cls.body
                  if isinstance(node, ast.FunctionDef) and node.name == "_write_reference_state")
    scope = dict(torch=torch, sample_initial_state=initialization.sample_initial_state,
                 quat_apply=lambda quat, value: value.clone())
    exec(compile(ast.Module(body=[method], type_ignores=[]), "commands.py", "exec"), scope)
    return scope["_write_reference_state"]


def attach_simulated_state(term, cfg, data=None):
    count = term.num_envs
    term._env.cfg = SimpleNamespace(dr=SimpleNamespace(init_state=cfg))
    term.cfg.reset_dof_pos_noise = term.cfg.reset_dof_vel_noise = .05
    term.joint_ids = list(range(29))
    if data is None:
        data = SimpleNamespace(joint_pos=torch.full((count, 29), 99.), joint_vel=torch.full((count, 29), 99.))
    data.joint_pos_limits = torch.tensor([-.2, .2]).repeat(count, 29, 1)
    recorded = {name: torch.full((count, size), 99.)
                for name, size in (("robot_pose", 7), ("robot_velocity", 6), ("ball_pose", 7),
                                   ("ball_velocity", 6), ("hoop_pose", 7))}
    # Capture complete asset writes, including the target environment indices.
    def write(name, value, env_ids):
        recorded[name][env_ids] = value
    def joints(position, velocity, joint_ids, env_ids):
        data.joint_pos[env_ids[:, None], joint_ids] = position
        data.joint_vel[env_ids[:, None], joint_ids] = velocity
    term.robot = SimpleNamespace(data=data,
        write_root_pose_to_sim=lambda value, env_ids: write("robot_pose", value, env_ids),
        write_root_link_velocity_to_sim=lambda value, env_ids: write("robot_velocity", value, env_ids),
        write_joint_state_to_sim=joints)
    term.ball = SimpleNamespace(
        write_root_pose_to_sim=lambda value, env_ids: write("ball_pose", value, env_ids),
        write_root_velocity_to_sim=lambda value, env_ids: write("ball_velocity", value, env_ids))
    term.hoop = SimpleNamespace(write_root_pose_to_sim=lambda value, env_ids: write("hoop_pose", value, env_ids))
    term.hoop_root_quat = torch.tensor([1., 0., 0., 0.])
    term.hoop_center_offset = torch.zeros(3)
    term._hoop_pose_validated = True
    term.reference.update(reference_state(count))
    term.reference["hoop_pos_cyl"] = torch.zeros(count, 3)
    write_method = write_reference_method()
    term._write_reference_state = lambda ids: write_method(term, ids)
    return recorded


def test_command_partial_write_preserves_reference_and_clamps_only_joint_positions():
    torch.manual_seed(5)
    term = make_command(3)
    cfg = init_config()
    cfg.dof_pos.std = 20.
    recorded = attach_simulated_state(term, cfg)
    term._env.scene.env_origins[:] = torch.tensor([[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]])
    reference = copy.deepcopy(term.reference)
    term._write_reference_state(torch.tensor([0, 2]))
    assert (term.robot.data.joint_pos[[0, 2]].abs() <= .2).all()
    assert (term.robot.data.joint_pos[[0, 2]].abs() == .2).any()
    assert term.robot.data.joint_pos[1].eq(99.).all()
    for value in recorded.values():
        assert value[1].eq(99.).all()
    expected_z = reference["object_pos"][[0, 2], 2] + term._env.scene.env_origins[[0, 2], 2]
    torch.testing.assert_close(recorded["ball_pose"][[0, 2], 2], expected_z)
    for name, value in term.reference.items():
        assert torch.equal(value, reference[name])


@pytest.mark.parametrize("legacy", ["missing", "none", "disabled", "components_disabled"])
def test_legacy_noise_only_applies_when_new_configuration_is_absent(legacy):
    torch.manual_seed(21)
    term = make_command(2)
    cfg = init_config(False)
    if legacy == "components_disabled":
        cfg.enabled = True
        for name in vars(cfg):
            if name != "enabled":
                getattr(cfg, name).enabled = False
    attach_simulated_state(term, None if legacy == "none" else cfg)
    if legacy == "missing":
        del term._env.cfg.dr.init_state
    term._write_reference_state(torch.tensor([0, 1]))
    if legacy in ("missing", "none"):
        assert term.robot.data.joint_pos.abs().max() <= .05
        assert term.robot.data.joint_vel.abs().max() <= .05
        assert torch.count_nonzero(term.robot.data.joint_pos) > 0
        assert torch.count_nonzero(term.robot.data.joint_vel) > 0
    else:
        assert not torch.count_nonzero(term.robot.data.joint_pos)
        assert not torch.count_nonzero(term.robot.data.joint_vel)


def test_rsi_state_is_sampled_before_noise_and_histories_see_final_reset_state():
    torch.manual_seed(14)
    count = 3
    term = make_command(count)
    sensor, sensor_env, data, action = make_sensor(nominal=1, count=count)
    sensor_env.cfg.student_observation.randomize_delay = False
    recorded = attach_simulated_state(term, init_config(), data)
    term.fixed_clip_ids[:] = 1
    term.cfg.enable_rsi = True
    term._env.scene.env_origins.zero_()
    def refresh():
        term.reference["root_pos"][:, 0] = term.frame
        term.reference["dof_pos"][:] = term.frame[:, None] * .001
    term._refresh_reference = refresh
    action_history = torch.full((5, count, 29), 99.)
    def reset_action(ids):
        assert not recorded["hoop_pose"][ids].eq(99.).any()
        action_history[:, ids] = data.joint_pos[ids]
        action.requested_actions[ids] = 0.
    term._env.action_manager.get_term = lambda _: SimpleNamespace(reset_to_current_state=reset_action)
    sensor_env.command_manager.get_term = lambda _: term
    sensor.reset([0, 2])
    term.reset([0, 2])
    assert term.rsi_started[[0, 2]].all() and term.frame[[0, 2]].min() > 0.
    torch.testing.assert_close(recorded["robot_pose"][[0, 2], 0], term.frame[[0, 2]], atol=.1, rtol=0)
    assert not torch.equal(data.joint_pos[[0, 2]], term.reference["dof_pos"][[0, 2]])
    torch.testing.assert_close(action_history[:, [0, 2]], data.joint_pos[[0, 2]].expand(5, -1, -1))
    sensor(sensor_env)
    # The actual Student builder fills every delay/history slot from the new state.
    joint_start = 3 + 3  # Per-frame gravity and angular velocity precede the joints.
    torch.testing.assert_close(sensor._sensor[:, [0, 2], joint_start:joint_start + 29],
                               data.joint_pos[[0, 2]].expand(sensor._sensor.shape[0], -1, -1))
    assert action_history[:, 1].eq(99.).all()
