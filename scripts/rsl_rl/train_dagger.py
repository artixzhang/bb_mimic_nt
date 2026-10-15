#!/usr/bin/env python3
"""Train a nominal-relative Student with online DAgger against a frozen Teacher."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--teacher-checkpoint", type=Path, required=True)
parser.add_argument("--num-envs", type=int, default=None)
parser.add_argument("--max-iterations", type=int, default=None)
parser.add_argument("--num-steps-per-env", type=int, default=None)
parser.add_argument("--replay-capacity", type=int, default=None)
parser.add_argument("--replay-recent-sample-fraction", type=float, default=None)
parser.add_argument("--batch-size", type=int, default=None)
parser.add_argument("--updates-per-iteration", type=int, default=None)
parser.add_argument("--resume-checkpoint", type=Path, default=None)
parser.add_argument("--save-replay", action="store_true", help="Include the device replay buffer in checkpoints.")
parser.add_argument("--run-name", type=str, default="")
parser.add_argument("--seed", type=int, default=None)
AppLauncher.add_app_launcher_args(parser)
args, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args
app = AppLauncher(args).app

import gymnasium as gym
import torch
import torch.nn.functional as functional
from torch.utils.tensorboard import SummaryWriter

from isaaclab.utils.io import dump_yaml
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from isaaclab_tasks.utils.hydra import hydra_task_config

import bb_mimic_nt.tasks  # noqa: F401
from bb_mimic_nt.student import StudentReplay, student_execution_probability, student_mix_actions
from bb_mimic_nt.action_math import clip_joint_actions, joint_position_target
from bb_mimic_nt.training import configure_training_curricula
from bb_mimic_nt.tasks.manager_based.bb_mimic_nt.mdp.curriculums import rsi_schedule
from bb_mimic_nt.student_runtime import (
    StudentPolicy,
    student_apply_teacher_config,
    student_export,
    student_load_teacher,
    student_metadata,
    student_validate_resume_observation,
)


torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


@hydra_task_config("BbMimicNT-G1-Shoot-Student-DAgger-v0", "rsl_rl_cfg_entry_point")
def main(env_cfg, agent_cfg):
    teacher_checkpoint = args.teacher_checkpoint.resolve()
    student_apply_teacher_config(env_cfg, teacher_checkpoint)
    if args.max_iterations is not None:
        agent_cfg.max_iterations = args.max_iterations
    if args.num_steps_per_env is not None:
        agent_cfg.num_steps_per_env = args.num_steps_per_env
    if args.replay_capacity is not None:
        agent_cfg.replay_capacity = args.replay_capacity
    if args.replay_recent_sample_fraction is not None:
        agent_cfg.replay_recent_sample_fraction = args.replay_recent_sample_fraction
    if args.batch_size is not None:
        agent_cfg.batch_size = args.batch_size
    if args.updates_per_iteration is not None:
        agent_cfg.updates_per_iteration = args.updates_per_iteration
    if args.num_envs is not None:
        env_cfg.scene.num_envs = args.num_envs
    if args.seed is not None:
        agent_cfg.seed = args.seed
    values = (
        agent_cfg.max_iterations,
        agent_cfg.num_steps_per_env,
        agent_cfg.replay_capacity,
        agent_cfg.batch_size,
        agent_cfg.updates_per_iteration,
    )
    if min(values) < 1:
        raise ValueError("DAgger iterations, rollout length, replay capacity, batch size, and updates must be positive.")
    if not 0.0 < agent_cfg.replay_recent_storage_fraction < 1.0:
        raise ValueError("Recent replay storage fraction must be in (0, 1).")
    if not 0.0 <= agent_cfg.replay_recent_sample_fraction <= 1.0:
        raise ValueError("Recent replay sample fraction must be in [0, 1].")
    configure_training_curricula(env_cfg, agent_cfg.max_iterations, agent_cfg.num_steps_per_env)
    env_cfg.seed = agent_cfg.seed
    if args.device is not None:
        env_cfg.sim.device = args.device
        agent_cfg.device = args.device

    suffix = f"_{args.run_name}" if args.run_name else ""
    run = (
        Path("logs/rsl_rl")
        / agent_cfg.experiment_name
        / f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}{suffix}"
    ).resolve()
    params = run / "params"
    params.mkdir(parents=True, exist_ok=False)
    env_cfg.log_dir = str(run)
    dump_yaml(str(params / "env.yaml"), env_cfg)
    dump_yaml(str(params / "agent.yaml"), agent_cfg)

    gym_env = gym.make("BbMimicNT-G1-Shoot-Student-DAgger-v0", cfg=env_cfg)
    base = gym_env.unwrapped
    env = RslRlVecEnvWrapper(gym_env, clip_actions=None)
    writer = SummaryWriter(str(run))
    try:
        torch.manual_seed(agent_cfg.seed)
        observations = env.get_observations()
        model, _ = student_load_teacher(teacher_checkpoint, observations, agent_cfg, base)
        action_term = base.action_manager.get_term("joint_pos")
        motion = base.command_manager.get_term("motion")
        replay = StudentReplay(
            agent_cfg.replay_capacity,
            observation_dim=observations["policy"].shape[-1],
            device=base.device,
            recent_storage_fraction=agent_cfg.replay_recent_storage_fraction,
            recent_sample_fraction=agent_cfg.replay_recent_sample_fraction,
        )
        optimizer = torch.optim.Adam(model.student.parameters(), lr=agent_cfg.algorithm.learning_rate)
        metadata = student_metadata(base, agent_cfg, teacher_checkpoint)
        (params / "student_config.json").write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        start_iteration = 0
        export_checkpoint = None
        export_iteration = None
        if args.resume_checkpoint:
            state = torch.load(args.resume_checkpoint, map_location=base.device, weights_only=False)
            if state.get("format") != "bb_mimic_nt_student_dagger_v1":
                raise ValueError("Resume checkpoint is not a compatible Student DAgger checkpoint.")
            if state["teacher_sha256"] != metadata["teacher_sha256"]:
                raise ValueError("Resume checkpoint was trained against a different Teacher.")
            student_validate_resume_observation(state["metadata"], metadata)
            model.student.load_state_dict(state["student"])
            optimizer.load_state_dict(state["optimizer"])
            if state.get("replay") is not None:
                replay.load_state_dict(state["replay"])
            start_iteration = int(state["iteration"])
            export_checkpoint = args.resume_checkpoint
            export_iteration = start_iteration
            configure_training_curricula(
                base.cfg, agent_cfg.max_iterations, agent_cfg.num_steps_per_env, motion_cfg=motion.cfg,
            )
            base.dr_context.iteration_offset = start_iteration
            base.reset()
            observations = env.get_observations()

        episode_steps = torch.zeros(base.num_envs, device=base.device)
        previous_student = previous_label = None
        previous_done = torch.ones(base.num_envs, dtype=torch.bool, device=base.device)
        for iteration in range(start_iteration, agent_cfg.max_iterations):
            student_probability = student_execution_probability(
                iteration,
                agent_cfg.max_iterations,
                agent_cfg.student_execution_start_fraction,
                agent_cfg.student_execution_end_fraction,
            )
            student_count = torch.zeros((), device=base.device)
            target_error = torch.zeros(action_term.action_dim, device=base.device)
            delta_error = torch.zeros((), device=base.device)
            student_delta_sq = torch.zeros((), device=base.device)
            teacher_delta_sq = torch.zeros((), device=base.device)
            delta_count = torch.zeros((), device=base.device)
            completed_count = torch.zeros((), device=base.device)
            failure_count = torch.zeros((), device=base.device)
            completed_steps = torch.zeros((), device=base.device)
            for _ in range(agent_cfg.num_steps_per_env):
                with torch.no_grad():
                    student_action = model.act_inference(observations)
                    teacher_raw = model.evaluate(observations)
                    label = clip_joint_actions(teacher_raw, action_term.cfg.action_clip)
                    executed, student_mask = student_mix_actions(label, student_action, student_probability)
                    student_target = joint_position_target(
                        student_action, action_term.nominal, action_term.action_scale, action_term.cfg.action_clip
                    )
                    teacher_target = joint_position_target(
                        label, action_term.nominal, action_term.action_scale, action_term.cfg.action_clip
                    )
                    target_error += (student_target - teacher_target).abs().mean(dim=0)
                    if previous_student is not None:
                        valid = ~previous_done
                        student_delta = student_target - previous_student
                        teacher_delta = teacher_target - previous_label
                        delta_error += ((student_delta - teacher_delta).abs().mean(dim=-1) * valid).sum()
                        student_delta_sq += (student_delta.square().mean(dim=-1) * valid).sum()
                        teacher_delta_sq += (teacher_delta.square().mean(dim=-1) * valid).sum()
                        delta_count += valid.sum()
                    previous_student, previous_label = student_target, teacher_target
                student_count += student_mask.sum()
                replay.add(observations["policy"], label)
                observations, _, done, info = env.step(executed)
                previous_done = done.bool()
                episode_steps += 1
                completed_count += previous_done.sum()
                time_outs = info.get("time_outs", torch.zeros_like(previous_done)).bool()
                failure_count += (previous_done & ~time_outs).sum()
                completed_steps += (episode_steps * previous_done).sum()
                episode_steps[previous_done] = 0

            model.student.train()
            losses = []
            for _ in range(agent_cfg.updates_per_iteration):
                inputs, targets = replay.sample(agent_cfg.batch_size)
                predictions = model.student(inputs)
                loss = functional.mse_loss(predictions, targets)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if agent_cfg.algorithm.max_grad_norm is not None:
                    torch.nn.utils.clip_grad_norm_(model.student.parameters(), agent_cfg.algorithm.max_grad_norm)
                optimizer.step()
                losses.append(float(loss.detach().item()))
            model.teacher.eval()
            mean_loss = sum(losses) / len(losses)
            with torch.no_grad():
                recent_inputs, recent_targets = replay.sample_recent(agent_cfg.batch_size)
                recent_loss = functional.mse_loss(model.student(recent_inputs), recent_targets).item()
            execution_rate = (student_count / (agent_cfg.num_steps_per_env * base.num_envs)).item()
            writer.add_scalar("DAgger/behavior_mse", mean_loss, iteration)
            writer.add_scalar("DAgger/recent_behavior_mse", recent_loss, iteration)
            writer.add_scalar("DAgger/student_probability", student_probability, iteration)
            writer.add_scalar("DAgger/student_execution_rate", execution_rate, iteration)
            joint_mae = target_error / agent_cfg.num_steps_per_env
            writer.add_scalar("DAgger/student_teacher_target_mae_rad", joint_mae.mean(), iteration)
            for name, error in zip(motion.dof_names, joint_mae):
                writer.add_scalar(f"DAgger/joint_target_mae_rad/{name}", error, iteration)
            writer.add_scalar("DAgger/target_delta_mae_rad", delta_error / delta_count.clamp_min(1), iteration)
            writer.add_scalar("DAgger/student_target_delta_rms_rad", (student_delta_sq / delta_count.clamp_min(1)).sqrt(), iteration)
            writer.add_scalar("DAgger/teacher_target_delta_rms_rad", (teacher_delta_sq / delta_count.clamp_min(1)).sqrt(), iteration)
            writer.add_scalar("DAgger/completed_episodes", completed_count, iteration)
            if completed_count.item() > 0:
                writer.add_scalar("DAgger/failure_fraction", failure_count / completed_count, iteration)
                writer.add_scalar("DAgger/mean_episode_steps", completed_steps / completed_count, iteration)
            writer.add_scalar("DAgger/sensor_curriculum_strength", base.student_observation.curriculum_strength, iteration)
            writer.add_scalar("Curriculum/rsi/probability", rsi_schedule(base)["probability"], iteration)
            writer.add_scalar("DAgger/replay_size", replay.size, iteration)
            writer.add_scalar("DAgger/replay_recent_size", replay.recent_size, iteration)
            writer.add_scalar("DAgger/replay_history_size", replay.history_size, iteration)
            writer.add_scalar("DAgger/observation_delay_encoded_mean", base.student_observation.delay_encoding.mean(), iteration)
            if iteration % 10 == 0 or iteration + 1 == agent_cfg.max_iterations:
                print(
                    f"iteration={iteration + 1} loss={mean_loss:.6f} recent_loss={recent_loss:.6f} "
                    f"replay={replay.size} "
                    f"student={execution_rate:.3f}",
                    flush=True,
                )
            if (iteration + 1) % agent_cfg.save_interval == 0 or iteration + 1 == agent_cfg.max_iterations:
                payload = {
                    "format": "bb_mimic_nt_student_dagger_v1",
                    "student": {key: value.detach().cpu() for key, value in model.student.state_dict().items()},
                    "optimizer": optimizer.state_dict(),
                    "replay": replay.state_dict() if args.save_replay else None,
                    "iteration": iteration + 1,
                    "teacher_sha256": metadata["teacher_sha256"],
                    "metadata": metadata,
                }
                export_checkpoint = run / f"model_{iteration + 1}.pt"
                torch.save(payload, export_checkpoint)
                export_iteration = iteration + 1

        policy = StudentPolicy(
            agent_cfg.policy.student_hidden_dims, agent_cfg.policy.activation, metadata["observation_dim"]
        ).to(base.device)
        policy.student.load_state_dict(model.student.state_dict())
        student_export(policy.eval(), run / "exported", metadata,
                       source_path=export_checkpoint, source_iteration=export_iteration)
        print(f"Student artifacts: {run / 'exported'}")
    finally:
        writer.close()
        env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        app.close()
