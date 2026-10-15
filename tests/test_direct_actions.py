"""Exercise the real shared action term on CPU without launching Isaac Sim."""

import ast
import math
from collections.abc import Sequence
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from bb_mimic_nt.action_math import clip_joint_actions, joint_position_target, resolve_action_scale


MDP = Path(__file__).resolve().parents[1] / "source/bb_mimic_nt/bb_mimic_nt/tasks/manager_based/bb_mimic_nt/mdp"


def make_direct_action(cfg=None, *, delays=(0, 2), joint_count=3, resolved_names=None):
    """Use production lifecycle methods with tensor-backed articulation state."""
    names = [f"joint_{i}" for i in range(joint_count)]
    if cfg is None:
        cfg = SimpleNamespace(joint_names=names, action_scale_factor=.25, action_clip=100.,
                              nominal_joint_position=None, action_scale_values=None)
    count = len(delays)
    data = SimpleNamespace(
        default_joint_pos=torch.linspace(-.1, .1, joint_count).repeat(count, 1),
        joint_pos=torch.full((count, joint_count), .2),
        joint_effort_limits=torch.linspace(10., 30., joint_count).repeat(count, 1),
        default_joint_stiffness=torch.linspace(40., 80., joint_count).repeat(count, 1),
        joint_pos_limits=torch.tensor([-.5, .5]).repeat(count, joint_count, 1),
        joint_vel_limits=torch.full((count, joint_count), 20.),
    )
    data.joint_stiffness = data.default_joint_stiffness.clone()
    asset = SimpleNamespace(data=data, find_joints=lambda _, preserve_order: (
        list(range(joint_count)), names if resolved_names is None else resolved_names))
    asset.set_joint_position_target = lambda target, joint_ids: setattr(asset, "last_target", target.clone())
    motion = SimpleNamespace(dof_names=names, reference={"dof_pos": torch.ones(count, joint_count)})
    env = SimpleNamespace(asset=asset, command_manager=SimpleNamespace(get_term=lambda _: motion),
                          num_envs=count, device="cpu",
                          cfg=SimpleNamespace(dr=SimpleNamespace(delay_nominal_steps=max(delays), delay_max_offset_steps=0)))

    class Base:
        def __init__(self, config, environment):
            self.cfg, self.device, self.num_envs, self._asset = config, "cpu", count, environment.asset

    tree = ast.parse((MDP / "actions.py").read_text())
    nodes = [node for node in tree.body if isinstance(node, (ast.ClassDef, ast.FunctionDef))
             and node.name in ("DirectJointPositionAction", "delayed_target")]
    namespace = dict(torch=torch, math=math, Sequence=Sequence, Articulation=object, ActionTerm=Base,
                     clip_joint_actions=clip_joint_actions, joint_position_target=joint_position_target,
                     resolve_action_scale=resolve_action_scale,
                     get_dr_context=lambda _: SimpleNamespace(delay_steps=torch.tensor(delays)))
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(MDP / "actions.py"), "exec"), namespace)
    return namespace["DirectJointPositionAction"](cfg, env), env, data


def test_direct_formula_preserves_inputs_and_has_no_target_angle_clamp():
    raw = torch.tensor([[-200., 0., 200.]])
    nominal = torch.tensor([[.2, -.4, .6]])
    scale = resolve_action_scale(torch.tensor([[20., 40., 60.]]), torch.tensor([[100., 100., 100.]]))
    torch.testing.assert_close(scale, torch.tensor([[.05, .1, .15]]))
    torch.testing.assert_close(clip_joint_actions(raw), torch.tensor([[-100., 0., 100.]]))
    torch.testing.assert_close(joint_position_target(raw, nominal, scale), torch.tensor([[-4.8, -.4, 15.6]]))
    torch.testing.assert_close(joint_position_target(torch.zeros_like(raw), nominal, scale), nominal)
    assert raw.tolist() == [[-200., 0., 200.]]


@pytest.mark.parametrize("value", [0., -1., float("nan"), float("inf")])
def test_action_math_rejects_invalid_mapping_parameters(value):
    with pytest.raises(ValueError):
        clip_joint_actions(torch.zeros(3), value)
    with pytest.raises(ValueError):
        resolve_action_scale(torch.ones(3), torch.ones(3), value)
    with pytest.raises(ValueError):
        resolve_action_scale(torch.full((3,), value), torch.ones(3))
    with pytest.raises(ValueError):
        resolve_action_scale(torch.ones(3), torch.full((3,), value))


def test_requested_observations_and_pd_error_do_not_follow_target_delay():
    action, env, data = make_direct_action()
    tree = ast.parse((MDP / "observations.py").read_text())
    names = {"dof_pos", "reference_dof_pos", "previous_action", "pd_error"}
    nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    motion = env.command_manager.get_term("motion")
    motion.robot, motion.joint_ids = env.asset, list(range(3))
    env.action_manager = SimpleNamespace(get_term=lambda _: action)
    scope = dict(torch=torch, motion_command=lambda _: motion)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), "observations.py", "exec"), scope)
    torch.testing.assert_close(scope["pd_error"](env), action.nominal - data.joint_pos)
    torch.testing.assert_close(scope["dof_pos"](env), data.joint_pos - action.nominal)
    torch.testing.assert_close(scope["reference_dof_pos"](env), motion.reference["dof_pos"] - action.nominal)
    raw = torch.tensor([[200., -2., 0.], [200., -2., 0.]])
    action.process_actions(raw)
    torch.testing.assert_close(action.raw_actions, raw)
    torch.testing.assert_close(scope["previous_action"](env), raw.clamp(-100, 100))
    torch.testing.assert_close(scope["pd_error"](env), action.requested_targets - data.joint_pos)
    torch.testing.assert_close(action.requested_targets[0], action.requested_targets[1])
    torch.testing.assert_close(action.applied_targets[0], action.requested_targets[0])
    torch.testing.assert_close(action.applied_targets[1], data.joint_pos[1])
    assert action.requested_targets.max() > data.joint_pos_limits[..., 1].max()
    motion.reference["dof_pos"].fill_(999.)
    previous_target = action.requested_targets.clone()
    action.process_actions(raw)
    torch.testing.assert_close(action.requested_targets, previous_target)
    action.apply_actions()
    torch.testing.assert_close(env.asset.last_target, action.applied_targets)


def test_delay_and_partial_reset_preserve_other_environment_history():
    action, _, data = make_direct_action()
    for value in (1., 2., 3.):
        action.process_actions(torch.full_like(action.nominal, value))
    torch.testing.assert_close(action.applied_targets[1], action.nominal[1] + action.action_scale[1])
    untouched = action._target_history[:, 0].clone()
    cursor = action._history_cursor
    data.joint_pos[1] = torch.tensor([.27, -.13, .38])  # RSI pose including reset noise.
    action.reset_to_current_state([1])
    assert action._history_cursor == cursor
    assert action.requested_actions[1].eq(0).all() and action.previous_requested_actions[1].eq(0).all()
    torch.testing.assert_close(action.requested_targets[1], action.nominal[1])
    torch.testing.assert_close(action.applied_targets[1], data.joint_pos[1])
    torch.testing.assert_close(action.previous_applied_targets[1], data.joint_pos[1])
    torch.testing.assert_close(action._target_history[:, 0], untouched)
    for tick in range(3):
        action.process_actions(torch.full_like(action.nominal, 4.))
        expected = data.joint_pos[1] if tick < 2 else action.requested_targets[1]
        torch.testing.assert_close(action.applied_targets[1], expected)


def test_scale_is_a_nominal_snapshot_and_deployment_can_override_mapping():
    action, _, data = make_direct_action()
    scale = action.action_scale.clone()
    data.joint_stiffness.mul_(1.1)
    data.joint_effort_limits.mul_(.8)
    action.process_actions(torch.ones_like(action.nominal))
    torch.testing.assert_close(action.action_scale, scale)
    torch.testing.assert_close(action.requested_targets, action.nominal + scale)
    cfg = SimpleNamespace(joint_names=[f"joint_{i}" for i in range(3)], action_scale_factor=.25,
                          action_clip=100., nominal_joint_position=[.1, .2, .3], action_scale_values=[.2, .4, .6])
    deployed, _, _ = make_direct_action(cfg)
    deployed.process_actions(torch.ones_like(deployed.nominal))
    torch.testing.assert_close(deployed.requested_targets, torch.tensor([[.3, .6, .9]]).repeat(2, 1))
    cfg.action_scale_values[1] = 0.
    with pytest.raises(ValueError, match="positive"):
        make_direct_action(cfg)


def test_joint_order_mismatch_is_rejected():
    with pytest.raises(ValueError, match="joint order"):
        make_direct_action(resolved_names=["joint_1", "joint_0", "joint_2"])


def test_teacher_velocity_scales_preserve_unscaled_reference_commands():
    """Execute the real observation config with lightweight Isaac term stand-ins."""
    path = MDP.parent / "bb_mimic_nt_env_cfg.py"
    tree = ast.parse(path.read_text())
    observation_class = next(node for node in tree.body
                             if isinstance(node, ast.ClassDef) and node.name == "ObservationsCfg")
    imports = [node for node in tree.body if isinstance(node, ast.ImportFrom) and (
        node.module == "bb_mimic_nt.action_math"
        or (node.module == "bb_mimic_nt.training"
            and any(alias.name == "TEACHER_HISTORY_LENGTH" for alias in node.names))
    )]
    functions = {node.attr: object() for node in ast.walk(observation_class)
                 if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name)
                 and node.value.id == "mdp"}

    def observation_term(*, func, scale=None, history_length=0):
        return SimpleNamespace(func=func, scale=scale, history_length=history_length)

    namespace = dict(configclass=lambda cls: cls, ObsGroup=object,
                     ObsTerm=observation_term, mdp=SimpleNamespace(**functions))
    exec(compile(ast.Module(body=[*imports, observation_class], type_ignores=[]), str(path), "exec"), namespace)
    policy = namespace["ObservationsCfg"]().policy
    assert policy.root_linear_velocity.scale == 2.0
    assert policy.root_angular_velocity.scale == .25
    assert policy.joint_velocity.scale == .05
    for name in ("reference_root_linear_velocity", "reference_root_angular_velocity",
                 "reference_joint_velocity", "reference_ball_linear_velocity"):
        assert getattr(policy, name).scale in (None, 1.0), name
