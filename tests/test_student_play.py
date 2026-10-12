"""Exercise Student playback selection and keyboard resets without Isaac Sim."""

import ast
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch


@pytest.fixture
def playback():
    path = Path(__file__).resolve().parents[1] / "scripts/rsl_rl/play_student.py"
    main = next(node for node in ast.parse(path.read_text()).body
                if isinstance(node, ast.FunctionDef) and node.name == "main")
    main.decorator_list = []
    # Unequal counts and command IDs different from skill indices expose mapping errors.
    skills = [dict(name=name, command_id=command, motion_file=name)
              for name, command in (("hold", 2), ("shoot", 0), ("stand_to_hold", 1))]
    counts = dict(hold=2, shoot=3, stand_to_hold=1)
    cfg = SimpleNamespace(
        commands=SimpleNamespace(motion=SimpleNamespace(skills_config=None, skills_path="skills.yaml")),
        scene=SimpleNamespace(num_envs=1), sim=SimpleNamespace(device="cpu"),
        configure_randomization=Mock(),
    )
    motion = SimpleNamespace(
        num_clips=6, clip_skill_ids=torch.tensor([0, 0, 1, 1, 1, 2]),
        set_clip_ids=Mock(), update_reference_markers=Mock(), set_debug_vis=Mock(),
    )
    base = SimpleNamespace(device="cpu", step_dt=0.01,
                           command_manager=SimpleNamespace(get_term=lambda _: motion))
    observations = {"policy": torch.zeros(1, 1)}
    env = SimpleNamespace(
        reset=Mock(return_value=(observations, {})),
        step=Mock(return_value=(observations, None, None, None)), close=Mock(),
    )
    keyboard = SimpleNamespace(clip_delta=0, reset_requested=False,
                               visibility_requested=False, pause_requested=False, close=Mock())

    def make_env(*_, **__):
        base.num_envs = cfg.scene.num_envs
        return SimpleNamespace(unwrapped=base)

    args = SimpleNamespace(
        student_weights=None, student_config=None, skill=None, clip_id=0,
        num_envs=4, all_clips=False, random_init=False, domain_randomization=False,
        device=None, ignore_failures=False, no_timeout=False,
        headless=True, steps=1, real_time=False,
    )
    namespace = dict(
        args=args, torch=torch, time=time, app=SimpleNamespace(is_running=lambda: True),
        gym=SimpleNamespace(make=Mock(side_effect=make_env)),
        joblib=SimpleNamespace(load=lambda name: [None] * counts[name]),
        load_skill_config=lambda _: {"skills": skills},
        student_read_artifact=lambda *_: ({}, {}), student_apply_deployment_config=Mock(),
        student_policy_from_state=lambda *_: lambda observation: observation,
        student_validate_deployment=Mock(), RslRlVecEnvWrapper=lambda _: env,
        StudentKeyboardController=lambda: keyboard,
    )
    exec(compile(ast.Module(body=[main], type_ignores=[]), str(path), "exec"), namespace)
    return SimpleNamespace(run=lambda: namespace["main"](cfg, None), args=args,
                           cfg=cfg, motion=motion, env=env, keyboard=keyboard, namespace=namespace)


@pytest.mark.parametrize("skill,clip_id,expected", [
    (None, 4, 4), (None, 7, 1), ("hold", -1, 1),
    ("shoot", 0, 2), ("shoot", 4, 3), ("stand_to_hold", 2, 5),
])
def test_single_clip_uses_skill_local_index(playback, skill, clip_id, expected):
    playback.args.skill, playback.args.clip_id = skill, clip_id
    playback.run()
    assert playback.cfg.scene.num_envs == 4
    assert playback.motion.set_clip_ids.call_args.args[0].tolist() == [expected] * 4


@pytest.mark.parametrize("skill,expected", [
    (None, [1, 2, 3, 4, 5, 0]), ("hold", [1, 0]),
    ("shoot", [3, 4, 2]), ("stand_to_hold", [5]),
])
def test_all_clips_creates_one_env_per_selected_clip(playback, skill, expected):
    playback.args.skill, playback.args.clip_id, playback.args.all_clips = skill, 1, True
    playback.run()
    assert playback.cfg.scene.num_envs == len(expected)
    assert playback.motion.set_clip_ids.call_args.args[0].tolist() == expected


@pytest.mark.parametrize("all_clips,delta,expected", [
    (False, -1, [4] * 4), (False, 3, [2] * 4),
    (True, -1, [4, 2, 3]), (True, 3, [2, 3, 4]),
])
def test_keyboard_wraps_and_restarts_within_skill(playback, all_clips, delta, expected):
    playback.args.skill, playback.args.all_clips, playback.args.headless = "shoot", all_clips, False
    playback.keyboard.clip_delta, playback.keyboard.reset_requested = delta, True
    playback.run()
    calls = playback.motion.set_clip_ids.call_args_list
    assert len(calls) == 3  # Initial assignment, keyboard switch, then restart.
    assert calls[1].args[0].tolist() == calls[2].args[0].tolist() == expected
    assert playback.env.reset.call_count == 3
    playback.keyboard.close.assert_called_once()
    playback.env.close.assert_called_once()


def test_unknown_skill_fails_before_creating_environment(playback):
    playback.args.skill = "missing"
    with pytest.raises(ValueError, match="Unknown skill 'missing'; available skills:.*hold.*shoot.*stand_to_hold"):
        playback.run()
    playback.namespace["gym"].make.assert_not_called()
