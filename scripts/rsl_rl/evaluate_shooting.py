#!/usr/bin/env python3
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Evaluate a G1 jump-shot checkpoint uniformly over all motion clips."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

import cli_args  # isort: skip

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", default="BbMimicNT-G1-Shoot-Play-v0")
parser.add_argument("--agent", default="rsl_rl_cfg_entry_point")
parser.add_argument("--num-envs", type=int, default=100)
parser.add_argument("--episodes-per-clip", type=int, default=1)
parser.add_argument("--output", type=Path, default=Path("outputs/shooting_evaluation.json"))
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
if not args_cli.checkpoint:
    parser.error("--checkpoint is required")
if args_cli.num_envs < 1:
    parser.error("--num-envs must be at least 1")
if args_cli.episodes_per_clip < 1:
    parser.error("--episodes-per-clip must be at least 1")
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch
from rsl_rl.runners import OnPolicyRunner

from isaaclab.utils.assets import retrieve_file_path
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from isaaclab_tasks.utils.hydra import hydra_task_config

import bb_mimic_nt.tasks  # noqa: F401


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg, agent_cfg):
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = min(args_cli.num_envs, 100)
    env_cfg.commands.motion.enable_rsi = False
    env_cfg.commands.motion.enable_push = False
    env_cfg.observations.policy.enable_corruption = False
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    gym_env = gym.make(args_cli.task, cfg=env_cfg)
    base_env = gym_env.unwrapped
    motion = base_env.command_manager.get_term("motion")
    env = RslRlVecEnvWrapper(gym_env, clip_actions=agent_cfg.clip_actions)
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    checkpoint = retrieve_file_path(args_cli.checkpoint)
    print(f"[INFO] Loading checkpoint: {checkpoint}")
    runner.load(checkpoint)
    policy = runner.get_inference_policy(device=base_env.device)

    records: list[dict] = []
    try:
        for start_clip in range(0, motion.num_clips, base_env.num_envs):
            count = min(base_env.num_envs, motion.num_clips - start_clip)
            clip_ids = torch.arange(start_clip, start_clip + count, device=base_env.device)
            if count < base_env.num_envs:
                clip_ids = torch.cat((clip_ids, clip_ids[-1:].expand(base_env.num_envs - count)))
            motion.set_clip_ids(clip_ids)

            for repetition in range(args_cli.episodes_per_clip):
                base_env.reset()
                obs = env.get_observations()
                finished = torch.zeros(base_env.num_envs, dtype=torch.bool, device=base_env.device)
                results = {
                    name: torch.zeros(base_env.num_envs, device=base_env.device)
                    for name in motion.last_episode_metrics
                }
                termination_names = list(base_env.termination_manager.active_terms)
                termination_reasons = torch.zeros(
                    (base_env.num_envs, len(termination_names)), dtype=torch.bool, device=base_env.device
                )

                while not torch.all(finished[:count]):
                    with torch.inference_mode():
                        actions = policy(obs)
                    # Manager reset/reward buffers are mutable and must not be
                    # created or updated under PyTorch inference mode.
                    obs, _, dones, _ = env.step(actions)
                    newly_finished = dones.bool() & ~finished
                    for name, value in results.items():
                        value[newly_finished] = motion.last_episode_metrics[name][newly_finished]
                    current_reasons = torch.stack(
                        [base_env.termination_manager.get_term(name) for name in termination_names], dim=-1
                    )
                    termination_reasons[newly_finished] = current_reasons[newly_finished]
                    finished |= dones.bool()

                for env_index in range(count):
                    records.append(
                        {
                            "clip_id": start_clip + env_index,
                            "repetition": repetition,
                            "success": bool(results["shot_success"][env_index].item()),
                            **{
                                name: float(value[env_index].item())
                                for name, value in results.items()
                                if name != "shot_success"
                            },
                            "termination": [
                                name
                                for reason_index, name in enumerate(termination_names)
                                if termination_reasons[env_index, reason_index]
                            ],
                        }
                    )
    finally:
        env.close()

    summary = {
        "checkpoint": str(checkpoint),
        "episodes": len(records),
        "hit_rate": sum(item["success"] for item in records) / max(len(records), 1),
        "mean_root_rmse": sum(item["root_rmse"] for item in records) / max(len(records), 1),
        "mean_dof_rmse": sum(item["dof_rmse"] for item in records) / max(len(records), 1),
        "mean_link_rmse": sum(item["link_rmse"] for item in records) / max(len(records), 1),
        "mean_ball_rmse": sum(item["ball_rmse"] for item in records) / max(len(records), 1),
        "termination_counts": {name: sum(name in item["termination"] for item in records) for name in termination_names},
    }
    args_cli.output.parent.mkdir(parents=True, exist_ok=True)
    args_cli.output.write_text(json.dumps({"summary": summary, "clips": records}, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"[INFO] Wrote {args_cli.output.resolve()}")


if __name__ == "__main__":
    main()
    simulation_app.close()
