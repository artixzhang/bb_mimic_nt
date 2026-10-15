"""Same-shaped policies must not silently change control coordinates."""

import copy
from types import SimpleNamespace

import pytest
import torch

from bb_mimic_nt.policy_contract import (
    build_policy_contract, load_teacher_contract, save_policy_contract, validate_policy_contract,
)


def contract_environment():
    names = [f"joint_{index}" for index in range(29)]
    gains = torch.arange(1, 30, dtype=torch.float32)[None].repeat(2, 1)
    action = SimpleNamespace(
        cfg=SimpleNamespace(joint_names=names, action_clip=100.), nominal=torch.zeros(2, 29),
        action_scale=.25 * 30 / gains, effort_limits=torch.full((2, 29), 30.),
        velocity_limits=torch.full((2, 29), 20.),
        joint_position_limits=torch.tensor([-2., 3.]).expand(2, 29, 2),
    )
    robot = SimpleNamespace(
        find_joints=lambda joint_names, preserve_order: (list(range(29)), joint_names),
        data=SimpleNamespace(default_joint_stiffness=gains, default_joint_damping=torch.ones(2, 29),
                             joint_stiffness=gains.clone()),
    )
    group = SimpleNamespace(joint_velocity=SimpleNamespace(scale=torch.tensor(.05), history_length=0),
                            action_history=SimpleNamespace(scale=None, history_length=5))
    return SimpleNamespace(
        action_manager=SimpleNamespace(get_term=lambda _: action), scene={"robot": robot}, step_dt=.01,
        observation_manager=SimpleNamespace(active_terms={"policy": ["joint_velocity", "action_history"]},
                                            cfg=SimpleNamespace(policy=group)),
    )


def test_contract_uses_frozen_parameters_and_student_teacher_group(tmp_path):
    env = contract_environment()
    original = build_policy_contract(env)
    env.scene["robot"].data.joint_stiffness.mul_(1.1)
    assert build_policy_contract(env) == original
    env.observation_manager.active_terms = {"policy": ["student_observation"],
                                            "teacher": ["joint_velocity", "action_history"]}
    env.observation_manager.cfg.teacher = env.observation_manager.cfg.policy
    assert build_policy_contract(env) == original
    save_policy_contract(env, tmp_path / "params")
    assert load_teacher_contract(tmp_path / "model_1.pt", env) == original


def test_missing_legacy_and_same_shape_changed_scale_are_rejected(tmp_path):
    env = contract_environment()
    with pytest.raises(ValueError, match="retrain"):
        load_teacher_contract(tmp_path / "model_1.pt")
    with pytest.raises(ValueError, match="retrain"):
        validate_policy_contract({"action_dim": 29, "observation_dim": 676})
    save_policy_contract(env, tmp_path / "params")
    env.action_manager.get_term("joint_pos").action_scale[:, 0] *= 2
    with pytest.raises(ValueError, match="action_scale"):
        load_teacher_contract(tmp_path / "model_1.pt", env)


@pytest.mark.parametrize("field", ["nominal_joint_position", "pd_stiffness", "action_scale", "velocity_limits"])
def test_contract_rejects_nonfinite_parameters(field):
    contract = build_policy_contract(contract_environment())
    contract[field][0] = float("nan")
    with pytest.raises(ValueError, match=field):
        validate_policy_contract(contract)


def test_contract_rejects_target_clipping_and_observation_semantic_change():
    contract = build_policy_contract(contract_environment())
    changed = copy.deepcopy(contract)
    changed["target_clip"] = [-1., 1.]
    with pytest.raises(ValueError, match="angle-clipped"):
        validate_policy_contract(changed)
    changed = copy.deepcopy(contract)
    changed["teacher_observation_terms"][0]["scale"] = 1.
    with pytest.raises(ValueError, match="teacher_observation_terms"):
        validate_policy_contract(changed, contract)
