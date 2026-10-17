#!/usr/bin/env python3
"""Train a deployable Student with on-policy PPO and frozen-Teacher action MSE."""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
teacher_argument = parser.add_argument("--teacher-checkpoint", type=Path)
parser.add_argument("--num-envs", type=int, default=None)
parser.add_argument("--max-iterations", type=int, default=None, help="Target total iterations, including completed iterations on resume.")
parser.add_argument("--num-steps-per-env", type=int, default=None)
parser.add_argument("--batch-size", type=int, default=None)
parser.add_argument("--num-learning-epochs", type=int, default=None)
parser.add_argument("--bc-coef", type=float, default=None)
parser.add_argument("--resume-checkpoint", type=Path, default=None, help="Resume a PPO+BC checkpoint; pure DAgger checkpoints are inference-only.")
parser.add_argument("--run-name", type=str, default="")
parser.add_argument("--seed", type=int, default=None)
AppLauncher.add_app_launcher_args(parser)
# AppLauncher probes the parser with help temporarily disabled; require the
# checkpoint only for the final parse so --help works without a dummy path.
teacher_argument.required = True
args, hydra_args = parser.parse_known_args()
# These used to be accepted by this script. Reject them before starting the simulator.
removed_options = (
    "--replay-capacity", "--replay-recent-sample-fraction", "--updates-per-iteration", "--save-replay",
)
for argument in hydra_args:
    if argument.split("=", 1)[0] in removed_options:
        parser.error(f"{argument.split('=', 1)[0]} was removed: Student training now uses five complete on-policy PPO+BC epochs.")
sys.argv = [sys.argv[0]] + hydra_args
app = AppLauncher(args).app

import gymnasium as gym
import torch

from isaaclab.utils.io import dump_yaml
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from isaaclab_tasks.utils.hydra import hydra_task_config

import bb_mimic_nt.tasks  # noqa: F401
from bb_mimic_nt.training import configure_training_curricula
from bb_mimic_nt.student_ppo_runner import StudentPPOBCRunner
from bb_mimic_nt.student_runtime import (
    StudentPolicy, student_apply_teacher_config, student_export, student_metadata,
)


torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


@hydra_task_config("BbMimicNT-G1-Shoot-Student-DAgger-v0", "rsl_rl_cfg_entry_point")
def main(env_cfg, agent_cfg):
    teacher_checkpoint = args.teacher_checkpoint.resolve()
    teacher_agent, _ = student_apply_teacher_config(env_cfg, teacher_checkpoint)
    if args.max_iterations is not None:
        agent_cfg.max_iterations = args.max_iterations
    if args.num_steps_per_env is not None:
        agent_cfg.num_steps_per_env = args.num_steps_per_env
    if args.batch_size is not None:
        agent_cfg.algorithm.batch_size = args.batch_size
    if args.num_learning_epochs is not None:
        agent_cfg.algorithm.num_learning_epochs = args.num_learning_epochs
    if args.bc_coef is not None:
        agent_cfg.algorithm.bc_coef = args.bc_coef
    if args.num_envs is not None:
        env_cfg.scene.num_envs = args.num_envs
    if args.seed is not None:
        agent_cfg.seed = args.seed
    if min(agent_cfg.max_iterations, agent_cfg.num_steps_per_env,
           agent_cfg.algorithm.batch_size, agent_cfg.algorithm.num_learning_epochs, env_cfg.scene.num_envs) < 1:
        raise ValueError("PPO+BC iteration, rollout, batch, epoch, and environment counts must be positive.")
    if not math.isfinite(agent_cfg.algorithm.bc_coef) or agent_cfg.algorithm.bc_coef < 0.0:
        raise ValueError("BC coefficient must be finite and non-negative.")
    configure_training_curricula(env_cfg, agent_cfg.max_iterations, agent_cfg.num_steps_per_env)
    # These are resolved before archiving the config, not silently replaced by the runner.
    teacher_policy = teacher_agent["policy"]
    agent_cfg.policy.critic_hidden_dims = list(teacher_policy["critic_hidden_dims"])
    agent_cfg.policy.critic_obs_normalization = bool(
        teacher_policy.get("critic_obs_normalization", teacher_agent.get("empirical_normalization", False))
    )
    agent_cfg.policy.activation = teacher_policy["activation"]
    # batch_size is the source of truth; the native PPO field is derived only for compatibility.
    agent_cfg.algorithm.num_mini_batches = math.ceil(
        env_cfg.scene.num_envs * agent_cfg.num_steps_per_env / agent_cfg.algorithm.batch_size
    )
    env_cfg.seed = agent_cfg.seed
    if args.device is not None:
        env_cfg.sim.device = args.device
        agent_cfg.device = args.device

    suffix = f"_{args.run_name}" if args.run_name else ""
    run = (Path("logs/rsl_rl") / agent_cfg.experiment_name
           / f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}{suffix}").resolve()
    params = run / "params"
    params.mkdir(parents=True, exist_ok=False)
    env_cfg.log_dir = str(run)
    dump_yaml(str(params / "env.yaml"), env_cfg)
    dump_yaml(str(params / "agent.yaml"), agent_cfg)

    gym_env = gym.make("BbMimicNT-G1-Shoot-Student-DAgger-v0", cfg=env_cfg)
    env = RslRlVecEnvWrapper(gym_env, clip_actions=None)
    runner = None
    try:
        base = gym_env.unwrapped
        torch.manual_seed(agent_cfg.seed)
        metadata = student_metadata(base, agent_cfg, teacher_checkpoint)
        (params / "student_config.json").write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        runner = StudentPPOBCRunner(
            env, agent_cfg.to_dict(), teacher_checkpoint, metadata, log_dir=str(run), device=base.device,
        )
        if args.resume_checkpoint:
            runner.load(args.resume_checkpoint)
        remaining = agent_cfg.max_iterations - runner.current_learning_iteration
        if remaining < 0:
            raise ValueError("Target max-iterations is less than the completed checkpoint iteration.")
        if remaining:
            runner.learn(num_learning_iterations=remaining, init_at_random_ep_len=False)
        completed = runner.current_learning_iteration
        # The runner exposes completed iterations consistently; the deployed policy is its mean only.
        policy = StudentPolicy(
            metadata["student_hidden_dims"], metadata["activation"], metadata["observation_dim"]
        ).to(base.device)
        policy.student.load_state_dict(runner.alg.policy.actor.state_dict())
        source = run / f"model_{completed}.pt"
        if not source.is_file():
            runner.save(str(source))
        student_export(policy.eval(), run / "exported", metadata,
                       source_path=source, source_iteration=completed)
        print(f"Student artifacts: {run / 'exported'}")
    finally:
        if runner is not None and runner.writer is not None:
            runner.writer.close()
        env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        app.close()
