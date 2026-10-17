"""Closed-loop comparisons must retain first terminal episodes across auto resets."""

import ast
import copy
import hashlib
import json
import math
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from bb_mimic_nt.student_evaluation import ClosedLoopEpisodeCollector, summarize_closed_loop


def test_first_terminal_snapshot_survives_later_auto_resets():
    collector = ClosedLoopEpisodeCollector(2, ["reference_finished", "fall_or_tilt"], "cpu")
    snapshot = dict(
        elapsed_steps=torch.tensor([2, 0]), planned_steps=torch.tensor([10.5, 20.0]),
        survival_fraction=torch.tensor([2 / 10.5, 0.0]),
        completed=torch.tensor([False, False]), failed=torch.tensor([True, False]),
        reasons=torch.tensor([[False, True], [False, False]]),
    )
    collector.capture(torch.tensor([True, False]), **snapshot)
    with pytest.raises(RuntimeError, match="every selected episode"):
        collector.records(.01)
    snapshot["elapsed_steps"][:] = torch.tensor([99, 20])
    snapshot["survival_fraction"][:] = 1.
    snapshot["completed"][:] = True
    snapshot["failed"][:] = False
    snapshot["reasons"][:] = torch.tensor([True, False])
    collector.capture(torch.tensor([True, True]), **snapshot)
    records = collector.records(.01)
    assert records[0]["episode_steps"] == 2
    assert records[0]["planned_steps"] == 10.5
    assert records[0]["survival_fraction"] == pytest.approx(2 / 10.5)
    assert records[0]["termination"] == ["fall_or_tilt"]
    assert records[1]["completed"] and not records[1]["failed"]


def test_simultaneous_completion_and_failure_is_not_success():
    collector = ClosedLoopEpisodeCollector(1, ["reference_finished", "non_finite_state"], "cpu")
    collector.capture(torch.tensor([True]), elapsed_steps=torch.tensor([10]),
                      planned_steps=torch.tensor([10.]), survival_fraction=torch.tensor([1.]),
                      completed=torch.tensor([True]), failed=torch.tensor([True]),
                      reasons=torch.tensor([[True, True]]))
    record = collector.records(.01)[0]
    summary = summarize_closed_loop([{"teacher": record, "student": record}], collector.termination_names)
    assert summary["student"]["completion_fraction"] == 0
    assert summary["student"]["failure_fraction"] == 1
    assert summary["student"]["termination_counts"] == {"reference_finished": 1, "non_finite_state": 1}


@pytest.fixture
def comparison(tmp_path):
    """Run the real CLI main against an environment that auto resets asynchronously."""
    path = Path(__file__).resolve().parents[1] / "scripts/rsl_rl/evaluate_student.py"
    main = next(node for node in ast.parse(path.read_text()).body
                if isinstance(node, ast.FunctionDef) and node.name == "main")
    main.decorator_list = []
    teacher_file = tmp_path / "teacher.pt"
    teacher_file.write_bytes(b"teacher")
    skills = [dict(name=name, command_id=command, end={"trigger_probability": .1})
              for name, command in (("hold", 2), ("shoot", 0), ("stand_to_hold", 1))]
    cfg = SimpleNamespace(
        commands=SimpleNamespace(motion=SimpleNamespace(skills_config={"skills": skills})),
        scene=SimpleNamespace(num_envs=2), sim=SimpleNamespace(device="cpu"),
        dr=SimpleNamespace(), student_observation=SimpleNamespace(),
        student_termination=SimpleNamespace(enabled=True), configure_randomization=Mock(),
        terminations=SimpleNamespace(),
    )
    args = SimpleNamespace(
        teacher_checkpoint=teacher_file, student_weights=tmp_path / "student.pt", student_config=None,
        num_envs=2, episodes_per_clip=1, mode="closed-loop", skill="shoot", end_hold_s=0.,
        clip_id=0, num_clips=0, device=None, output=tmp_path / "result.json",
    )
    names = ["reference_finished", "fall_or_tilt"]
    controller = SimpleNamespace(last_episode={}, last_episode_causes={})
    motion = SimpleNamespace(num_clips=6, clip_skill_ids=torch.tensor([0, 0, 1, 1, 1, 2]),
                             lengths=torch.tensor([3, 3, 4, 5, 6, 3]), fps=100.)
    motion.set_clip_ids = lambda ids: setattr(motion, "clip_ids", ids.clone())
    base = SimpleNamespace(
        num_envs=2, device="cpu", step_dt=.01, max_episode_length=20,
        command_manager=SimpleNamespace(get_term=lambda _: motion),
        action_manager=SimpleNamespace(get_term=lambda _: SimpleNamespace(cfg=SimpleNamespace(action_clip=100))),
        reward_manager=SimpleNamespace(get_term_cfg=lambda _: SimpleNamespace(func=None)),
        termination_manager=SimpleNamespace(active_terms=names), student_termination_curriculum=controller,
    )
    observations = {"policy": torch.zeros(2, 1)}
    counters = torch.zeros(2, dtype=torch.long)
    assignments = []

    def reset():
        counters.zero_()
        assignments.append(motion.clip_ids.tolist())

    def step(actions):
        counters.add_(1)
        student = bool(actions[0, 0])
        planned = (motion.lengths[motion.clip_ids] - 1).float()
        done = counters >= (torch.ones_like(planned) if student else planned)
        # This imitates the command's pre-reset copy. Live state then resets.
        controller.last_episode = dict(
            elapsed_steps=counters.clone(), planned_steps=planned,
            survival_fraction=(counters / planned).clamp(max=1),
            completed=done & ~torch.full_like(done, student),
            failed=done & torch.full_like(done, student),
        )
        controller.last_episode_causes = dict(
            reference_finished=done & ~torch.full_like(done, student),
            fall_or_tilt=done & torch.full_like(done, student),
        )
        counters[done] = 0
        return observations, None, done, {}

    base.reset = reset
    env = SimpleNamespace(get_observations=lambda: observations, step=step, close=Mock())
    teacher = SimpleNamespace(evaluate=lambda _: torch.zeros(2, 1))
    namespace = dict(
        args=args, torch=torch, copy=copy, json=json, hashlib=hashlib, math=math,
        gym=SimpleNamespace(make=Mock(return_value=SimpleNamespace(unwrapped=base))),
        RslRlVecEnvWrapper=lambda _: env, TerminationsCfg=lambda: SimpleNamespace(original=True),
        student_read_artifact=lambda *_: ({}, {}), student_apply_deployment_config=Mock(),
        student_apply_teacher_config=Mock(), student_policy_from_state=lambda *_: lambda _: torch.ones(2, 1),
        student_validate_deployment=Mock(), student_load_teacher=lambda *_: (teacher, {}),
        clip_joint_actions=lambda value, _: value, ClosedLoopEpisodeCollector=ClosedLoopEpisodeCollector,
        summarize_closed_loop=summarize_closed_loop,
    )
    exec(compile(ast.Module(body=[main], type_ignores=[]), str(path), "exec"), namespace)
    return SimpleNamespace(run=lambda: namespace["main"](cfg, None), args=args, cfg=cfg, env=env,
                           assignments=assignments, namespace=namespace)


def test_closed_loop_matches_skill_clips_and_retains_terminal_state(comparison):
    comparison.run()
    report = json.loads(comparison.args.output.read_text())
    assert [record["clip_id"] for record in report["clips"]] == [2, 3, 4]
    assert all(record["command_id"] == 0 for record in report["clips"])
    assert comparison.assignments == [[2, 3], [2, 3], [4, 4], [4, 4]]
    assert report["summary"]["teacher"]["completion_fraction"] == 1
    assert report["summary"]["student"]["failure_fraction"] == 1
    assert [record["teacher"]["episode_steps"] for record in report["clips"]] == [3, 4, 5]
    assert all(record["student"]["episode_steps"] == 1 for record in report["clips"])
    assert comparison.cfg.terminations.original
    assert not comparison.cfg.student_termination.enabled
    assert comparison.cfg.dr.delay_nominal_steps == comparison.cfg.student_observation.delay_max_steps == 0
    assert all(skill["end"]["trigger_probability"] == 1
               for skill in comparison.cfg.commands.motion.skills_config["skills"])
    comparison.cfg.configure_randomization.assert_called_once_with(random_init=False, domain_randomization=False)
    comparison.env.close.assert_called_once()


@pytest.mark.parametrize("skill,match", [(None, "explicit --skill"), ("missing", "Unknown skill")])
def test_closed_loop_requires_known_explicit_skill_before_simulation(comparison, skill, match):
    comparison.args.skill = skill
    with pytest.raises(ValueError, match=match):
        comparison.run()
    comparison.namespace["gym"].make.assert_not_called()
