#!/usr/bin/env python3
"""Compare matched Teacher/Student ball flights or complete closed-loop skill episodes."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import sys
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
teacher_argument = parser.add_argument("--teacher-checkpoint", type=Path)
student_argument = parser.add_argument("--student-weights", type=Path)
parser.add_argument("--student-config", type=Path, default=None)
parser.add_argument("--mode", choices=("flight", "closed-loop"), default="flight")
parser.add_argument("--skill", default=None, help="Skill to evaluate; required in closed-loop mode.")
parser.add_argument("--end-hold-s", type=float, default=5.0, help="Fixed end hold in closed-loop mode, in seconds.")
parser.add_argument("--num-envs", type=int, default=100)
parser.add_argument("--episodes-per-clip", type=int, default=1)
parser.add_argument("--clip-id", type=int, default=0, help="First clip; local to --skill when selected.")
parser.add_argument("--num-clips", type=int, default=0, help="Number of clips; zero evaluates through the end.")
parser.add_argument("--output", type=Path, default=Path("outputs/student_comparison.json"))
AppLauncher.add_app_launcher_args(parser)
# AppLauncher probes arguments before enabling help handling.
teacher_argument.required = True
student_argument.required = True
args, hydra_args = parser.parse_known_args()
if args.mode == "closed-loop" and args.skill is None:
    parser.error("--mode closed-loop requires an explicit --skill")
if not math.isfinite(args.end_hold_s) or args.end_hold_s < 0:
    parser.error("--end-hold-s must be finite and non-negative")
if args.num_clips < 0:
    parser.error("--num-clips must be non-negative")
sys.argv = [sys.argv[0]] + hydra_args
app = AppLauncher(args).app

import gymnasium as gym
import torch

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from isaaclab_tasks.utils.hydra import hydra_task_config

import bb_mimic_nt.tasks  # noqa: F401
from bb_mimic_nt.action_math import clip_joint_actions
from bb_mimic_nt.student_evaluation import ClosedLoopEpisodeCollector, summarize_closed_loop
from bb_mimic_nt.tasks.manager_based.bb_mimic_nt.bb_mimic_nt_env_cfg import TerminationsCfg
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
    if args.mode == "closed-loop" and args.skill is None:
        raise ValueError("Closed-loop evaluation requires an explicit --skill.")
    student_state, exported = student_read_artifact(args.student_weights, args.student_config)
    with args.teacher_checkpoint.open("rb") as stream:
        teacher_sha256 = hashlib.file_digest(stream, "sha256").hexdigest()
    expected_teacher = exported.get("teacher_sha256", exported.get("_provenance", {}).get("teacher_sha256"))
    if expected_teacher not in (None, teacher_sha256):
        raise ValueError("The Student export was distilled from a different Teacher checkpoint.")
    student_apply_deployment_config(env_cfg, exported)
    student_apply_teacher_config(env_cfg, args.teacher_checkpoint.resolve())
    env_cfg.scene.num_envs = args.num_envs
    skills = env_cfg.commands.motion.skills_config["skills"]
    skill_names = [skill["name"] for skill in skills]
    if args.skill is not None and args.skill not in skill_names:
        raise ValueError(f"Unknown skill {args.skill!r}; available skills: {skill_names}.")
    selected_skill = skill_names.index(args.skill) if args.skill is not None else None
    if args.mode == "flight":
        env_cfg.episode_length_s = 10.0
        env_cfg.terminations.reference_finished = None
        env_cfg.terminations.fall_or_tilt = None
        env_cfg.terminations.mechanical_joint_limit = None
        env_cfg.terminations.non_finite_state = None
        env_cfg.terminations.root_tracking_error = None
        env_cfg.terminations.dof_tracking_error = None
        env_cfg.terminations.interaction_tracking_error = None
    else:
        env_cfg.configure_randomization(random_init=False, domain_randomization=False)
        env_cfg.dr.delay_nominal_steps = 0
        env_cfg.dr.delay_max_offset_steps = 0
        env_cfg.student_observation.delay_nominal_steps = 0
        env_cfg.student_observation.delay_min_steps = 0
        env_cfg.student_observation.delay_max_steps = 0
        env_cfg.commands.motion.enable_rsi = False
        env_cfg.commands.motion.randomize_clip = False
        env_cfg.commands.motion.enable_adaptive_speed = False
        env_cfg.commands.motion.end_duration_s = args.end_hold_s
        # A fixed hold must also bypass the per-skill probabilistic end trigger.
        env_cfg.commands.motion.skills_config = copy.deepcopy(env_cfg.commands.motion.skills_config)
        for skill in env_cfg.commands.motion.skills_config["skills"]:
            skill["end"]["trigger_probability"] = 1.0
        env_cfg.terminations = TerminationsCfg()
        env_cfg.student_termination.enabled = False
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
    termination_names = list(base.termination_manager.active_terms)
    try:
        selected_clips = torch.arange(motion.num_clips, device=base.device)
        if selected_skill is not None:
            selected_clips = selected_clips[motion.clip_skill_ids == selected_skill]
        first_clip = args.clip_id % len(selected_clips)
        last_clip = len(selected_clips) if args.num_clips == 0 else min(len(selected_clips), first_clip + args.num_clips)
        selected_clips = selected_clips[first_clip:last_clip]
        for clip_start in range(0, len(selected_clips), base.num_envs):
            clip_ids = selected_clips[clip_start:clip_start + base.num_envs]
            count = len(clip_ids)
            evaluated_clip_ids = clip_ids.tolist()
            if count < base.num_envs:
                clip_ids = torch.cat((clip_ids, clip_ids[-1:].expand(base.num_envs - count)))
            motion.set_clip_ids(clip_ids)
            for repetition in range(args.episodes_per_clip):
                paired = {}
                for policy_name in ("teacher", "student"):
                    base.reset()
                    if args.mode == "flight":
                        metrics.peak_height.copy_((motion.ball.data.root_pos_w - base.scene.env_origins)[:, 2])
                    observations = env.get_observations()
                    if args.mode == "closed-loop":
                        collector = ClosedLoopEpisodeCollector(count, termination_names, base.device)
                        # No artificial timeout: clip lengths and the fixed hold determine the horizon.
                        max_steps = math.ceil(float((
                            (motion.lengths[clip_ids] - 1).float() / motion.fps + args.end_hold_s
                        ).max()) / base.step_dt) + 4
                    else:
                        max_steps = base.max_episode_length - 1
                    for _ in range(max_steps):
                        with torch.inference_mode():
                            if policy_name == "student":
                                output = student(observations["policy"])
                            else:
                                teacher_raw = teacher.evaluate(observations)
                                output = clip_joint_actions(teacher_raw, action.cfg.action_clip)
                        observations, _, done, _ = env.step(output)
                        if args.mode == "closed-loop":
                            if bool(torch.any(done[:count])):
                                controller = base.student_termination_curriculum
                                terminal = controller.last_episode
                                reasons = torch.stack(
                                    [controller.last_episode_causes[name][:count] for name in termination_names], dim=-1
                                )
                                collector.capture(
                                    done[:count],
                                    elapsed_steps=terminal["elapsed_steps"][:count],
                                    planned_steps=terminal["planned_steps"][:count],
                                    survival_fraction=terminal["survival_fraction"][:count],
                                    completed=terminal["completed"][:count],
                                    failed=terminal["failed"][:count],
                                    reasons=reasons,
                                )
                            if bool(torch.all(collector.finished)):
                                break
                        elif bool(torch.all(metrics.landed[:count])):
                            break
                    if args.mode == "closed-loop":
                        if not bool(torch.all(collector.finished)):
                            raise RuntimeError("Closed-loop episodes exceeded their reference plus hold horizon.")
                        paired[policy_name] = collector.records(base.step_dt)
                    else:
                        paired[policy_name] = (
                            metrics.peak_height.clone(),
                            metrics.landing_xy.clone(),
                            metrics.landed.clone(),
                        )
                for index in range(count):
                    record = {"clip_id": evaluated_clip_ids[index], "repetition": repetition}
                    if args.mode == "closed-loop":
                        record.update(skill=args.skill, command_id=skills[selected_skill]["command_id"])
                        record.update({name: episodes[index] for name, episodes in paired.items()})
                        records.append(record)
                        continue
                    for name, (peak, landing, landed) in paired.items():
                        record[name] = {
                            "peak_height_m": float(peak[index].item()) if torch.isfinite(peak[index]) else None,
                            "landing_xy_from_hoop_m": landing[index].tolist() if landed[index] else None,
                        }
                    records.append(record)
    finally:
        env.close()

    if args.mode == "closed-loop":
        summary = summarize_closed_loop(records, termination_names)
        report = {
            "mode": args.mode,
            "settings": {
                "skill": args.skill, "end_hold_s": args.end_hold_s,
                "domain_randomization": False, "rsi": False,
                "termination_curriculum": False, "deterministic_policies": True,
                "teacher_checkpoint": str(args.teacher_checkpoint.resolve()),
                "student_weights": str(args.student_weights.resolve()),
            },
            "summary": summary,
            "clips": records,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps(summary, indent=2))
        return

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
