#!/usr/bin/env python3
"""Interactively play a Student, optionally by skill, without artificial sensor or action delays."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--student-weights", type=Path, required=True, help="Exported weights or a model_*.pt checkpoint.")
parser.add_argument("--student-config", type=Path, default=None)
parser.add_argument(
    "--clip-id", type=int, default=0,
    help="Starting clip index; local to the selected skill when --skill is set.",
)
parser.add_argument("--skill", default=None, help="Play only this skill name; default plays all skills.")
parser.add_argument("--num-envs", type=int, default=1)
parser.add_argument(
    "--all-clips", action="store_true",
    help="Play every clip of the selected skill, or every clip of all skills when --skill is omitted.",
)
parser.add_argument("--random-init", action="store_true", help="Enable reset joint position/velocity noise.")
parser.add_argument(
    "--domain-randomization", action="store_true",
    help="Enable full-strength physical randomization and pushes.",
)
parser.add_argument("--real-time", action="store_true")
parser.add_argument("--ignore-failures", action="store_true")
parser.add_argument("--no-timeout", action="store_true")
parser.add_argument("--steps", type=int, default=0, help="Stop after this many steps; zero runs until closed.")
AppLauncher.add_app_launcher_args(parser)
args, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args
app = AppLauncher(args).app

import gymnasium as gym
import joblib
import torch

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from isaaclab_tasks.utils.hydra import hydra_task_config

import bb_mimic_nt.tasks  # noqa: F401
from bb_mimic_nt.student_runtime import (
    student_apply_deployment_config,
    student_policy_from_state,
    student_read_artifact,
    student_validate_deployment,
)
from bb_mimic_nt.skills import load_skill_config


class StudentKeyboardController:
    """Translate playback keys into requests handled by the simulation loop."""

    def __init__(self):
        import carb
        import omni.appwindow

        self._carb = carb
        self.clip_delta = 0
        self.reset_requested = False
        self.visibility_requested = False
        self.pause_requested = False
        self._input = carb.input.acquire_input_interface()
        self._keyboard = omni.appwindow.get_default_app_window().get_keyboard()
        self._subscription = self._input.subscribe_to_keyboard_events(self._keyboard, self._on_event)

    def _on_event(self, event):
        if event.type != self._carb.input.KeyboardEventType.KEY_PRESS:
            return True
        name = event.input.name.upper()
        if name in {"COMMA", ","}:
            self.clip_delta -= 1
        elif name in {"PERIOD", "."}:
            self.clip_delta += 1
        elif name == "R":
            self.reset_requested = True
        elif name == "V":
            self.visibility_requested = True
        elif name in {"P", "SPACE"}:
            self.pause_requested = True
        return True

    def close(self) -> None:
        self._input.unsubscribe_to_keyboard_events(self._keyboard, self._subscription)


@hydra_task_config("BbMimicNT-G1-Shoot-Student-Play-v0", "rsl_rl_cfg_entry_point")
def main(env_cfg, _agent_cfg):
    state, metadata = student_read_artifact(args.student_weights, args.student_config)
    student_apply_deployment_config(env_cfg, metadata)
    env_cfg.configure_randomization(
        random_init=args.random_init, domain_randomization=args.domain_randomization,
    )

    motion_cfg = env_cfg.commands.motion
    motion_cfg.skills_config = motion_cfg.skills_config or load_skill_config(motion_cfg.skills_path)
    skills = motion_cfg.skills_config["skills"]
    skill_names = [skill["name"] for skill in skills]
    if args.skill is not None and args.skill not in skill_names:
        raise ValueError(f"Unknown skill {args.skill!r}; available skills: {skill_names}.")
    selected_skill_index = skill_names.index(args.skill) if args.skill is not None else None
    clip_counts = [len(joblib.load(skill["motion_file"])) for skill in skills]
    selected_clip_count = sum(clip_counts) if selected_skill_index is None else clip_counts[selected_skill_index]
    env_cfg.scene.num_envs = selected_clip_count if args.all_clips else max(1, args.num_envs)
    if args.device is not None:
        env_cfg.sim.device = args.device
    if args.ignore_failures:
        env_cfg.terminations.fall_or_tilt = None
        env_cfg.terminations.mechanical_joint_limit = None
        env_cfg.terminations.non_finite_state = None
        env_cfg.terminations.root_tracking_error = None
        env_cfg.terminations.dof_tracking_error = None
        env_cfg.terminations.interaction_tracking_error = None
    if args.no_timeout:
        env_cfg.terminations.reference_finished = None
        env_cfg.episode_length_s = 1.0e9

    gym_env = gym.make("BbMimicNT-G1-Shoot-Student-Play-v0", cfg=env_cfg)
    base = gym_env.unwrapped
    env = RslRlVecEnvWrapper(gym_env)
    policy = student_policy_from_state(state, metadata, base.device)
    student_validate_deployment(base, metadata)
    motion = base.command_manager.get_term("motion")
    selected_clips = torch.arange(motion.num_clips, device=base.device)
    if selected_skill_index is not None:
        selected_clips = selected_clips[motion.clip_skill_ids == selected_skill_index]
    clip_offset = args.clip_id % len(selected_clips)

    def assign_clips():
        if args.all_clips:
            indices = (torch.arange(base.num_envs, device=base.device) + clip_offset) % len(selected_clips)
            clip_ids = selected_clips[indices]
        else:
            clip_ids = selected_clips[clip_offset].expand(base.num_envs)
        motion.set_clip_ids(clip_ids)
        observations, _ = env.reset()
        motion.update_reference_markers()
        return observations

    def describe_clip():
        global_clip_id = int(selected_clips[clip_offset])
        skill = skills[int(motion.clip_skill_ids[global_clip_id])]
        return (
            f"skill={skill['name']}; command_id={skill['command_id']}; "
            f"clip offset={clip_offset}; global clip id={global_clip_id}"
        )

    observations = assign_clips()
    keyboard = None if args.headless else StudentKeyboardController()
    markers_visible = not args.headless
    motion.set_debug_vis(markers_visible)
    paused = False
    if keyboard:
        print("[INFO] Keys: V markers, ,/. clip, P or Space pause/resume, R restart")
    print(
        f"[INFO] {describe_clip()}; selected clips={len(selected_clips)}; "
        f"all_clips={args.all_clips}; ignore_failures={args.ignore_failures}"
    )
    try:
        step = 0
        while app.is_running() and (args.steps == 0 or step < args.steps):
            start = time.time()
            if keyboard and keyboard.clip_delta:
                clip_offset = (clip_offset + keyboard.clip_delta) % len(selected_clips)
                keyboard.clip_delta = 0
                observations = assign_clips()
                print(f"[INFO] {describe_clip()}")
            if keyboard and keyboard.reset_requested:
                keyboard.reset_requested = False
                observations = assign_clips()
                print(f"[INFO] restarted {describe_clip()}")
            if keyboard and keyboard.visibility_requested:
                keyboard.visibility_requested = False
                markers_visible = not markers_visible
                motion.set_debug_vis(markers_visible)
            if keyboard and keyboard.pause_requested:
                keyboard.pause_requested = False
                paused = not paused
                print(f"[INFO] playback={'paused' if paused else 'running'}")
            if paused:
                app.update()
                time.sleep(0.01)
                continue
            with torch.inference_mode():
                output = policy(observations["policy"])
            observations, _, _, _ = env.step(output)
            if markers_visible:
                motion.update_reference_markers()
            step += 1
            remaining = base.step_dt - (time.time() - start)
            if args.real_time and remaining > 0.0:
                time.sleep(remaining)
    finally:
        if keyboard:
            keyboard.close()
        env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        app.close()
