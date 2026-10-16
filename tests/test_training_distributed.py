"""Exercise the training entrypoint without starting Isaac Sim or using GPUs."""

import argparse
import ast
import json
import os
import sys
import traceback
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import pytest
from torch.distributed.elastic.multiprocessing.errors import record


TRAIN = Path(__file__).resolve().parents[1] / "scripts/rsl_rl/train.py"


def execute(nodes, namespace):
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(TRAIN), "exec"), namespace)


@pytest.mark.parametrize(
    "flags,blocking,full_error",
    [([], None, None), (["--debug_cuda"], "1", "1"), (["--distributed"], None, "1")],
)
def test_diagnostic_environment_is_ready_when_simulator_starts(monkeypatch, flags, blocking, full_error):
    monkeypatch.delenv("CUDA_LAUNCH_BLOCKING", raising=False)
    monkeypatch.delenv("HYDRA_FULL_ERROR", raising=False)
    monkeypatch.delenv("WORLD_SIZE", raising=False)
    monkeypatch.setattr(sys, "argv", [str(TRAIN), *flags, "env.scene.num_envs=8"])
    seen = []

    class Launcher:
        @staticmethod
        def add_app_launcher_args(parser):
            parser.add_argument("--device", default="cuda:0")

        def __init__(self, args):
            seen.append((os.environ.get("CUDA_LAUNCH_BLOCKING"), os.environ.get("HYDRA_FULL_ERROR")))
            self.app = object()

    namespace = dict(
        argparse=argparse,
        os=os,
        sys=sys,
        AppLauncher=Launcher,
        cli_args=SimpleNamespace(add_rsl_rl_args=lambda parser: None),
    )
    startup = []
    for node in ast.parse(TRAIN.read_text()).body:
        if not isinstance(node, (ast.Import, ast.ImportFrom)):
            startup.append(node)
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "simulation_app" for target in node.targets
        ):
            break
    execute(startup, namespace)
    assert seen == [(blocking, full_error)]
    assert namespace["sys"].argv == [str(TRAIN), "env.scene.num_envs=8"]


def training_case(distributed, mode, runner_name="OnPolicyRunner", device=None):
    events = []

    class ManagerCfg(SimpleNamespace):
        pass

    class OtherCfg:
        pass

    cfg = ManagerCfg(scene=SimpleNamespace(num_envs=8), sim=SimpleNamespace(device="cuda:0"))
    agent = SimpleNamespace(
        seed=42, max_iterations=2, num_steps_per_env=48, device="cuda:0",
        experiment_name="test", run_name="", resume=False, algorithm=SimpleNamespace(class_name="PPO"),
        class_name=runner_name, clip_actions=None,
    )
    agent.to_dict = lambda: {}
    env = SimpleNamespace(cfg=cfg, close=lambda: events.append("env.close"))
    env.unwrapped = env

    def make_env(*args, **kwargs):
        events.append(("gym.make", cfg.sim.device))
        return env

    class Runner:
        def __init__(self, env, config, **kwargs):
            events.append(("runner", kwargs["device"]))
            self.alg = object()

        def add_git_repo_to_log(self, path):
            pass

        def learn(self, **kwargs):
            events.append("learn")

    def install(algorithm):
        events.append("install")
        return True

    namespace = dict(
        __file__=str(TRAIN), os=os, datetime=datetime, installed_version="3.0.1",
        args_cli=SimpleNamespace(
            distributed=distributed, ppo_minibatch_indices=mode, num_envs=None, max_iterations=None,
            device=device, task="test", video=False, export_io_descriptors=False,
        ),
        cli_args=SimpleNamespace(update_rsl_rl_cfg=lambda config, args: config),
        app_launcher=SimpleNamespace(local_rank=2),
        torch=SimpleNamespace(
            __version__="test", cuda=SimpleNamespace(set_device=lambda rank: events.append(("set_device", rank)))
        ),
        ManagerBasedRLEnvCfg=ManagerCfg, DirectRLEnvCfg=OtherCfg, DirectMARLEnvCfg=OtherCfg,
        DirectMARLEnv=OtherCfg, RslRlBaseRunnerCfg=SimpleNamespace,
        gym=SimpleNamespace(make=make_env),
        RslRlVecEnvWrapper=lambda env, **kwargs: env,
        OnPolicyRunner=Runner, DistillationRunner=Runner,
        install_cpu_minibatch_sampler=install,
        dump_yaml=lambda *args: None, save_policy_contract=lambda *args: None,
    )
    main = next(node for node in ast.parse(TRAIN.read_text()).body if isinstance(node, ast.FunctionDef) and node.name == "main")
    main.decorator_list = []
    execute([main], namespace)
    return namespace["main"], cfg, agent, events


@pytest.mark.parametrize(
    "distributed,mode,installed",
    [(False, "auto", False), (True, "auto", True), (False, "cpu", True),
     (True, "cpu", True), (False, "native", False), (True, "native", False)],
)
def test_sampler_mode_and_local_device_binding(distributed, mode, installed):
    main, cfg, agent, events = training_case(distributed, mode)
    main(cfg, agent)
    assert ("install" in events) == installed
    if installed:
        assert events.index("install") < events.index("learn")
    if distributed:
        assert events[:3] == [("set_device", 2), ("gym.make", "cuda:2"), ("runner", "cuda:2")]
        assert cfg.seed == agent.seed == 44
    else:
        assert events[:2] == [("gym.make", "cuda:0"), ("runner", "cuda:0")]
        assert cfg.seed == agent.seed == 42
    assert events[-2:] == ["learn", "env.close"]


def test_cpu_simulation_is_rejected_before_creating_distributed_environment():
    main, cfg, agent, events = training_case(True, "auto", device="cpu")
    with pytest.raises(ValueError, match="Distributed training is not supported"):
        main(cfg, agent)
    assert events == []


def test_distillation_runner_keeps_its_own_sampler():
    main, cfg, agent, events = training_case(True, "cpu", runner_name="DistillationRunner")
    main(cfg, agent)
    assert "install" not in events
    assert events[-2:] == ["learn", "env.close"]


def entrypoint():
    return next(
        node for node in ast.parse(TRAIN.read_text()).body
        if isinstance(node, ast.If) and isinstance(node.test, ast.Compare)
        and isinstance(node.test.left, ast.Name) and node.test.left.id == "__name__"
    )


class WorkerExit(BaseException):
    """Stand in for os._exit so a worker failure cannot terminate pytest."""


@pytest.mark.parametrize("distributed", [False, True])
def test_worker_failure_is_recorded_before_any_forced_exit(monkeypatch, tmp_path, distributed):
    error_file = tmp_path / "worker_error.json"
    monkeypatch.setenv("TORCHELASTIC_ERROR_FILE", str(error_file))
    events = []

    def fail():
        raise RuntimeError("original minibatch indexing failure")

    def hard_exit(code):
        error = json.loads(error_file.read_text())
        assert "original minibatch indexing failure" in error["message"]["message"]
        events.append(("exit", code))
        raise WorkerExit()

    namespace = dict(
        __name__="__main__", record=record, main=fail,
        args_cli=SimpleNamespace(distributed=distributed),
        os=SimpleNamespace(_exit=hard_exit), sys=sys, traceback=traceback,
        simulation_app=SimpleNamespace(close=lambda: events.append("app.close")),
    )
    with pytest.raises(WorkerExit if distributed else RuntimeError):
        execute([entrypoint()], namespace)
    assert error_file.is_file()
    assert events == ([("exit", 1)] if distributed else [])


@pytest.mark.parametrize("distributed", [False, True])
def test_successful_worker_closes_the_simulator(monkeypatch, tmp_path, distributed):
    error_file = tmp_path / "worker_error.json"
    monkeypatch.setenv("TORCHELASTIC_ERROR_FILE", str(error_file))
    events = []
    namespace = dict(
        __name__="__main__", record=record, main=lambda: events.append("main"),
        args_cli=SimpleNamespace(distributed=distributed),
        os=SimpleNamespace(_exit=lambda code: pytest.fail("successful worker must not hard-exit")),
        sys=sys, traceback=traceback,
        simulation_app=SimpleNamespace(close=lambda: events.append("app.close")),
    )
    execute([entrypoint()], namespace)
    assert events == ["main", "app.close"]
    assert not error_file.exists()
