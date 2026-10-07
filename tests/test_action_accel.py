"""CPU regression checks for action history across independent environment resets."""

from __future__ import annotations

import ast
from collections.abc import Sequence
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from bb_mimic_nt.core import action_accel_cost


@pytest.mark.parametrize("dt", [0.01, 0.02])
def test_action_accel_history_survives_buffer_reuse_and_partial_reset(dt: float) -> None:
    # Load the actual stateful methods without starting Isaac Sim, as in the DR tests.
    path = (
        Path(__file__).resolve().parents[1]
        / "source/bb_mimic_nt/bb_mimic_nt/tasks/manager_based/bb_mimic_nt/mdp/rewards.py"
    )
    node = next(
        node for node in ast.parse(path.read_text()).body
        if isinstance(node, ast.ClassDef) and node.name == "UnifiedMimicReward"
    )
    node.body = [
        method for method in node.body
        if isinstance(method, ast.FunctionDef) and method.name in {"__init__", "_action_accel", "reset"}
    ]

    class ManagerTermBase:
        def __init__(self, cfg, env):
            self._env = env
            self.num_envs = env.num_envs
            self.device = env.device

    namespace = dict(
        ManagerTermBase=ManagerTermBase, torch=torch, Sequence=Sequence, action_accel_cost=action_accel_cost
    )
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), namespace)
    env = SimpleNamespace(num_envs=2, device="cpu", step_dt=dt, episode_length_buf=torch.ones(2))
    reward = namespace["UnifiedMimicReward"](None, env)
    action = torch.zeros(2, 1)

    def step(first, second):
        action[:, 0] = torch.tensor([first, second])
        return reward._action_accel(action)

    # Initialize under inference mode, then reuse the manager's mutable action buffer.
    with torch.inference_mode():
        assert step(0.0, 0.0).tolist() == [0.0, 0.0]
        assert step(1.0, 1.0).tolist() == [0.0, 0.0]
        assert step(0.0, 2.0).tolist() == pytest.approx([4.0, 0.0])
    reward.reset([0])
    assert step(100.0, 1.0).tolist() == pytest.approx([0.0, 4.0])
    assert step(101.0, 0.0).tolist() == [0.0, 0.0]
    assert step(102.0, -1.0).tolist() == [0.0, 0.0]
    assert step(101.0, -2.0).tolist() == pytest.approx([4.0, 0.0])
    reward.reset()
    assert step(-100.0, 100.0).tolist() == [0.0, 0.0]
    assert step(0.0, 0.0).tolist() == [0.0, 0.0]
    assert step(0.0, 0.0).tolist() == pytest.approx([1.0e4, 1.0e4])
