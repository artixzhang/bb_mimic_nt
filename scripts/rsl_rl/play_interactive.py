#!/usr/bin/env python3
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Interactively play a deterministic G1 jump-shot Teacher checkpoint."""

from __future__ import annotations

import argparse
import os
import sys
import time

from isaaclab.app import AppLauncher

import cli_args  # isort: skip

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", default="BbMimicNT-G1-Shoot-Play-v0")
parser.add_argument("--agent", default="rsl_rl_cfg_entry_point")
parser.add_argument("--clip-id", type=int, default=0)
parser.add_argument("--num-envs", type=int, default=1)
parser.add_argument("--all-clips", action="store_true")
parser.add_argument("--real-time", action="store_true")
parser.add_argument(
    "--ignore-failures",
    action="store_true",
    help="Disable early failure terminations so the complete reference can be inspected.",
)
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
if args_cli.headless:
    parser.error("Interactive playback requires a visible window (omit --headless).")
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import carb
import gymnasium as gym
import omni.appwindow
import torch
from rsl_rl.runners import OnPolicyRunner

from isaaclab.utils.assets import retrieve_file_path
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

import bb_mimic_nt.tasks  # noqa: F401


class KeyboardController:
    """Turn key events into requests consumed by the simulation loop."""

    def __init__(self):
        self.pending_clip_delta = 0
        self.pending_reset = False
        self.pending_visibility_toggle = False
        self._input = carb.input.acquire_input_interface()
        self._keyboard = omni.appwindow.get_default_app_window().get_keyboard()
        self._subscription = self._input.subscribe_to_keyboard_events(self._keyboard, self._on_event)

    def _on_event(self, event):
        if event.type != carb.input.KeyboardEventType.KEY_PRESS:
            return True
        name = event.input.name.upper()
        if name in {"COMMA", ","}:
            self.pending_clip_delta -= 1
        elif name in {"PERIOD", "."}:
            self.pending_clip_delta += 1
        elif name == "R":
            self.pending_reset = True
        elif name == "V":
            self.pending_visibility_toggle = True
        return True

    def close(self) -> None:
        self._input.unsubscribe_to_keyboard_events(self._keyboard, self._subscription)


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg, agent_cfg):
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = 100 if args_cli.all_clips else max(1, args_cli.num_envs)
    env_cfg.commands.motion.enable_rsi = False
    env_cfg.commands.motion.enable_push = False
    env_cfg.observations.policy.enable_corruption = False
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    if args_cli.ignore_failures:
        # Reference-inspection mode: a failed policy may remain fallen until
        # reference_finished resets the environment.
        env_cfg.terminations.fall_or_tilt = None
        env_cfg.terminations.mechanical_joint_limit = None
        env_cfg.terminations.non_finite_state = None
        env_cfg.terminations.root_tracking_error = None
        env_cfg.terminations.dof_tracking_error = None
        env_cfg.terminations.object_tracking_error = None
        env_cfg.terminations.interaction_tracking_error = None

    gym_env = gym.make(args_cli.task, cfg=env_cfg)
    base_env = gym_env.unwrapped
    motion = base_env.command_manager.get_term("motion")
    env = RslRlVecEnvWrapper(gym_env, clip_actions=agent_cfg.clip_actions)
    clip_offset = args_cli.clip_id % motion.num_clips

    def assign_clips():
        if args_cli.all_clips:
            ids = (torch.arange(base_env.num_envs, device=base_env.device) + clip_offset) % motion.num_clips
        else:
            ids = torch.full((base_env.num_envs,), clip_offset, device=base_env.device, dtype=torch.long)
        motion.set_clip_ids(ids)
        observations, _ = env.reset()
        motion.update_reference_markers()
        return observations

    obs = assign_clips()
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    if args_cli.checkpoint:
        checkpoint = retrieve_file_path(args_cli.checkpoint)
    else:
        log_root = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
        checkpoint = get_checkpoint_path(log_root, agent_cfg.load_run, agent_cfg.load_checkpoint)
        print(f"[INFO] --checkpoint omitted; selected latest checkpoint under {log_root}")
    print(f"[INFO] Loading checkpoint: {checkpoint}")
    runner.load(checkpoint)
    policy = runner.get_inference_policy(device=base_env.device)

    # ``set_debug_vis`` mutates marker state and does not return the resulting
    # visibility.  Keep the UI state explicitly so the first V press really
    # hides the markers instead of redundantly enabling them again.
    markers_visible = True
    motion.set_debug_vis(markers_visible)
    keyboard = KeyboardController()
    print("[INFO] Keys: V reference markers, , previous clip, . next clip, R reset")
    print(
        f"[INFO] clip offset={clip_offset}; all_clips={args_cli.all_clips}; "
        f"ignore_failures={args_cli.ignore_failures}"
    )
    if args_cli.ignore_failures:
        print("[WARN] Early failure resets are disabled; a failed robot may remain fallen while reference advances.")

    try:
        while simulation_app.is_running():
            start = time.time()
            if keyboard.pending_clip_delta:
                clip_offset = (clip_offset + keyboard.pending_clip_delta) % motion.num_clips
                keyboard.pending_clip_delta = 0
                obs = assign_clips()
                print(f"[INFO] clip offset={clip_offset}")
            if keyboard.pending_reset:
                keyboard.pending_reset = False
                obs = assign_clips()
                print(f"[INFO] reset clip offset={clip_offset}")
            if keyboard.pending_visibility_toggle:
                keyboard.pending_visibility_toggle = False
                markers_visible = not markers_visible
                motion.set_debug_vis(markers_visible)
                print(f"[INFO] reference markers={'on' if markers_visible else 'off'}")

            with torch.inference_mode():
                actions = policy(obs)
            # Keep simulator/reward/reset state out of inference mode: these
            # managers own mutable tensors and R may reset them at any time.
            obs, _, _, _ = env.step(actions)
            if markers_visible:
                motion.update_reference_markers()

            delay = base_env.step_dt - (time.time() - start)
            if args_cli.real_time and delay > 0.0:
                time.sleep(delay)
    finally:
        keyboard.close()
        env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
