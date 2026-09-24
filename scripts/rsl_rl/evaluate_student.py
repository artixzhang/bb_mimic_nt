#!/usr/bin/env python3
"""Compare Teacher and Student basketball landing positions and peak heights on matched clips."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--teacher-checkpoint", type=Path, required=True)
parser.add_argument("--student-weights", type=Path, required=True)
parser.add_argument("--student-config", type=Path, default=None)
parser.add_argument("--num-envs", type=int, default=100)
parser.add_argument("--episodes-per-clip", type=int, default=1)
parser.add_argument("--clip-id", type=int, default=0, help="First clip to evaluate.")
parser.add_argument("--num-clips", type=int, default=0, help="Number of clips; zero evaluates through the end.")
parser.add_argument("--output", type=Path, default=Path("outputs/student_comparison.json"))
AppLauncher.add_app_launcher_args(parser)
args, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args
app = AppLauncher(args).app

import gymnasium as gym
import torch

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from isaaclab_tasks.utils.hydra import hydra_task_config

import bb_mimic_nt.tasks  # noqa: F401
from bb_mimic_nt.student import student_teacher_target
from bb_mimic_nt.student_runtime import (
    student_apply_deployment_config,
    student_apply_teacher_config,
    student_load_teacher,
    student_policy_from_state,
    student_read_artifact,
    student_validate_deployment,
)


@hydra_task_config("BbMimicNT-G1-Shoot-Student-Play-v0", "rsl_rl_cfg_entry_point")
def main(env_cfg, agent_cfg):
    if args.num_envs < 1 or args.episodes_per_clip < 1:
        raise ValueError("Evaluation environment and episode counts must be positive.")
    student_state, exported = student_read_artifact(args.student_weights, args.student_config)
    with args.teacher_checkpoint.open("rb") as stream:
        teacher_sha256 = hashlib.file_digest(stream, "sha256").hexdigest()
    if exported.get("teacher_sha256") not in (None, teacher_sha256):
        raise ValueError("The Student export was distilled from a different Teacher checkpoint.")
    student_apply_deployment_config(env_cfg, exported)
    student_apply_teacher_config(env_cfg, args.teacher_checkpoint.resolve())
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.episode_length_s = 10.0
    env_cfg.terminations.reference_finished = None
    env_cfg.terminations.fall_or_tilt = None
    env_cfg.terminations.mechanical_joint_limit = None
    env_cfg.terminations.non_finite_state = None
    env_cfg.terminations.root_tracking_error = None
    env_cfg.terminations.dof_tracking_error = None
    env_cfg.terminations.interaction_tracking_error = None
    if args.device is not None:
        env_cfg.sim.device = args.device

    gym_env = gym.make("BbMimicNT-G1-Shoot-Student-Play-v0", cfg=env_cfg)
    base = gym_env.unwrapped
    env = RslRlVecEnvWrapper(gym_env)
    metadata = exported
    student = student_policy_from_state(student_state, metadata, base.device)
    student_validate_deployment(base, metadata)
    teacher, _ = student_load_teacher(args.teacher_checkpoint.resolve(), env.get_observations(), agent_cfg, base)
    motion = base.command_manager.get_term("motion")
    action = base.action_manager.get_term("joint_pos")
    metrics = base.reward_manager.get_term_cfg("ball_flight_metrics").func
    records = []
    try:
        first_clip = args.clip_id % motion.num_clips
        last_clip = motion.num_clips if args.num_clips == 0 else min(motion.num_clips, first_clip + args.num_clips)
        for clip_start in range(first_clip, last_clip, base.num_envs):
            count = min(base.num_envs, last_clip - clip_start)
            clip_ids = torch.arange(clip_start, clip_start + count, device=base.device)
            if count < base.num_envs:
                clip_ids = torch.cat((clip_ids, clip_ids[-1:].expand(base.num_envs - count)))
            motion.set_clip_ids(clip_ids)
            for repetition in range(args.episodes_per_clip):
                paired = {}
                for policy_name in ("teacher", "student"):
                    base.reset()
                    metrics.peak_height.copy_((motion.ball.data.root_pos_w - base.scene.env_origins)[:, 2])
                    observations = env.get_observations()
                    for _ in range(base.max_episode_length - 1):
                        with torch.inference_mode():
                            if policy_name == "student":
                                output = student(observations["policy"])
                            else:
                                teacher_raw = teacher.evaluate(observations)
                                output = student_teacher_target(
                                    teacher_raw,
                                    motion.reference["dof_pos"],
                                    action.residual_scale,
                                    action.lower,
                                    action.upper,
                                    action.nominal,
                                )
                                action.set_teacher_residual(
                                    teacher_raw,
                                    torch.ones(base.num_envs, dtype=torch.bool, device=base.device),
                                )
                        observations, _, _, _ = env.step(output)
                        if bool(torch.all(metrics.landed[:count])):
                            break
                    paired[policy_name] = (
                        metrics.peak_height.clone(),
                        metrics.landing_xy.clone(),
                        metrics.landed.clone(),
                    )
                for index in range(count):
                    record = {"clip_id": clip_start + index, "repetition": repetition}
                    for name, (peak, landing, landed) in paired.items():
                        record[name] = {
                            "peak_height_m": float(peak[index].item()) if torch.isfinite(peak[index]) else None,
                            "landing_xy_from_hoop_m": landing[index].tolist() if landed[index] else None,
                        }
                    records.append(record)
    finally:
        env.close()

    summary = {
        "episodes_per_policy": len(records),
        "teacher_missing_landing": sum(item["teacher"]["landing_xy_from_hoop_m"] is None for item in records),
        "student_missing_landing": sum(item["student"]["landing_xy_from_hoop_m"] is None for item in records),
    }
    for name in ("teacher", "student"):
        peaks = [item[name]["peak_height_m"] for item in records if item[name]["peak_height_m"] is not None]
        landings = [
            item[name]["landing_xy_from_hoop_m"]
            for item in records
            if item[name]["landing_xy_from_hoop_m"] is not None
        ]
        summary[f"{name}_mean_peak_height_m"] = sum(peaks) / len(peaks) if peaks else None
        summary[f"{name}_mean_landing_xy_from_hoop_m"] = (
            [sum(point[axis] for point in landings) / len(landings) for axis in range(2)] if landings else None
        )
        summary[f"{name}_mean_landing_radius_m"] = (
            sum(math.hypot(*point) for point in landings) / len(landings) if landings else None
        )
    paired_landings = [
        (item["teacher"]["landing_xy_from_hoop_m"], item["student"]["landing_xy_from_hoop_m"])
        for item in records
        if item["teacher"]["landing_xy_from_hoop_m"] is not None
        and item["student"]["landing_xy_from_hoop_m"] is not None
    ]
    summary["paired_landing_count"] = len(paired_landings)
    summary["mean_paired_landing_distance_m"] = (
        sum(math.dist(first, second) for first, second in paired_landings) / len(paired_landings)
        if paired_landings
        else None
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"summary": summary, "clips": records}, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    try:
        main()
    finally:
        app.close()
