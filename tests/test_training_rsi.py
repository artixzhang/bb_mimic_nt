"""Shared RSI budgets, Student CLI overrides, and resumed live command clocks."""

import ast
import copy
import math
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

from bb_mimic_nt.core import rsi_probability
from bb_mimic_nt.training import (
    RSI_DECAY_FRACTION,
    RSI_INITIAL_PROBABILITY,
    configure_training_curricula,
)

ROOT = Path(__file__).resolve().parents[1]
MDP = ROOT / "source/bb_mimic_nt/bb_mimic_nt/tasks/manager_based/bb_mimic_nt/mdp"


def environment_config():
    return SimpleNamespace(
        dr=SimpleNamespace(total_iterations=7000, steps_per_iteration=48),
        commands=SimpleNamespace(
            motion=SimpleNamespace(
                enable_rsi=True, rsi_initial_probability=RSI_INITIAL_PROBABILITY, rsi_decay_steps=50400
            )
        ),
        scene=SimpleNamespace(num_envs=8192),
    )


def live_schedule():
    spec = importlib.util.spec_from_file_location("training_rsi_events", MDP / "events.py")
    events = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(events)
    node = next(
        n
        for n in ast.parse((MDP / "curriculums.py").read_text()).body
        if isinstance(n, ast.FunctionDef) and n.name == "rsi_schedule"
    )
    scope = dict(rsi_probability=rsi_probability, training_policy_steps=events.training_policy_steps)
    exec(compile(ast.Module(body=[node], type_ignores=[]), "curriculums.py", "exec"), scope)
    return scope["rsi_schedule"]


@pytest.mark.parametrize("iterations,rollout", [(7000, 48), (1000, 48), (2000, 24), (1000, 96)])
def test_teacher_and_student_budgets_exit_at_the_same_fraction(iterations, rollout):
    cfg = environment_config()
    configure_training_curricula(cfg, iterations, rollout)
    assert cfg.dr.total_iterations == iterations and cfg.dr.steps_per_iteration == rollout
    term = SimpleNamespace(cfg=copy.deepcopy(cfg.commands.motion))
    env = SimpleNamespace(
        cfg=cfg,
        common_step_counter=0,
        dr_context=SimpleNamespace(iteration_offset=0),
        command_manager=SimpleNamespace(get_term=lambda _: term),
    )
    schedule = live_schedule()
    assert schedule(env)["probability"] == pytest.approx(RSI_INITIAL_PROBABILITY)
    env.common_step_counter = round(iterations * rollout * RSI_DECAY_FRACTION / 2)
    assert schedule(env)["probability"] == pytest.approx(RSI_INITIAL_PROBABILITY / 2)
    env.common_step_counter = round(iterations * rollout * RSI_DECAY_FRACTION)
    assert schedule(env)["probability"] == 0.0
    term.cfg.enable_rsi = False
    env.common_step_counter = 0
    assert schedule(env)["probability"] == 0.0


@pytest.mark.parametrize("resume_iteration,expected", [(75, 0.15), (150, 0.0), (400, 0.0)])
def test_student_resume_updates_copied_command_without_restarting_rsi(resume_iteration, expected):
    cfg = environment_config()
    live_cfg = copy.deepcopy(cfg.commands.motion)
    configure_training_curricula(cfg, 1000, 48, motion_cfg=live_cfg)
    assert live_cfg.rsi_decay_steps == cfg.commands.motion.rsi_decay_steps == 7200
    env = SimpleNamespace(
        cfg=cfg,
        common_step_counter=0,
        dr_context=SimpleNamespace(iteration_offset=resume_iteration),
        command_manager=SimpleNamespace(get_term=lambda _: SimpleNamespace(cfg=live_cfg)),
    )
    assert live_schedule()(env)["probability"] == pytest.approx(expected)


def test_budget_preserves_explicit_probability_and_play_switches():
    cfg = environment_config()
    cfg.commands.motion.rsi_initial_probability = 0.2
    cfg.commands.motion.enable_rsi = False
    live_cfg = copy.deepcopy(cfg.commands.motion)
    configure_training_curricula(cfg, 2000, 24, motion_cfg=live_cfg)
    for motion in (cfg.commands.motion, live_cfg):
        assert motion.rsi_initial_probability == 0.2 and not motion.enable_rsi


@pytest.mark.parametrize("iterations,rollout", [(0, 48), (1000, 0), (-1, 24)])
def test_invalid_training_budget_is_rejected(iterations, rollout):
    with pytest.raises(ValueError, match="positive"):
        configure_training_curricula(environment_config(), iterations, rollout)


@pytest.mark.parametrize("iterations,rollout", [(1000, 48), (2000, 24), (1000, 96)])
def test_actual_dagger_startup_applies_cli_budget_before_environment_creation(iterations, rollout):
    source = ROOT / "scripts/rsl_rl/train_dagger.py"
    main = next(n for n in ast.parse(source.read_text()).body if isinstance(n, ast.FunctionDef) and n.name == "main")
    body = []
    for node in main.body:
        body.append(node)
        if (
            isinstance(node, ast.Expr)
            and isinstance(node.value, ast.Call)
            and isinstance(node.value.func, ast.Name)
            and node.value.func.id == "configure_training_curricula"
        ):
            break
    else:
        pytest.fail("DAgger must configure its RSI budget before constructing the environment.")
    node = ast.FunctionDef(
        name="startup",
        args=main.args,
        body=body + [ast.Return(ast.Name(id="env_cfg", ctx=ast.Load()))],
        decorator_list=[],
    )
    args = SimpleNamespace(
        teacher_checkpoint=Path("teacher.pt"),
        max_iterations=iterations,
        num_steps_per_env=rollout,
        batch_size=None,
        num_learning_epochs=None,
        bc_coef=None,
        num_envs=None,
        seed=None,
    )
    scope = dict(
        args=args,
        student_apply_teacher_config=lambda *args: ({}, {}),
        configure_training_curricula=configure_training_curricula,
        math=math,
    )
    exec(compile(ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])), str(source), "exec"), scope)
    agent = SimpleNamespace(
        max_iterations=1000,
        num_steps_per_env=48,
        algorithm=SimpleNamespace(batch_size=10, num_learning_epochs=5, bc_coef=1.0),
    )
    cfg = scope["startup"](environment_config(), agent)
    assert cfg.commands.motion.rsi_decay_steps == int(iterations * rollout * RSI_DECAY_FRACTION)
    assert cfg.dr.total_iterations == iterations and cfg.dr.steps_per_iteration == rollout
