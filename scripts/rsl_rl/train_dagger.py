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
from bb_mimic_nt.student import StudentReplay, student_execution_probability, student_mix_actions, student_teacher_target
from bb_mimic_nt.student_runtime import (
    StudentPolicy,
    student_apply_teacher_config,
    student_export,
    student_load_teacher,
    student_metadata,
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
    env_cfg.dr.total_iterations = agent_cfg.max_iterations
    env_cfg.dr.steps_per_iteration = agent_cfg.num_steps_per_env
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
        if args.resume_checkpoint:
            state = torch.load(args.resume_checkpoint, map_location=base.device, weights_only=False)
            if state.get("format") != "bb_mimic_nt_student_dagger_v1":
                raise ValueError("Resume checkpoint is not a compatible Student DAgger checkpoint.")
            if state["teacher_sha256"] != metadata["teacher_sha256"]:
                raise ValueError("Resume checkpoint was trained against a different Teacher.")
            model.student.load_state_dict(state["student"])
            optimizer.load_state_dict(state["optimizer"])
            if state.get("replay") is not None:
                replay.load_state_dict(state["replay"])
            start_iteration = int(state["iteration"])
            base.dr_context.iteration_offset = start_iteration
            base.reset()
            observations = env.get_observations()

        for iteration in range(start_iteration, agent_cfg.max_iterations):
            student_probability = student_execution_probability(
                iteration,
                agent_cfg.max_iterations,
                agent_cfg.student_execution_start_fraction,
                agent_cfg.student_execution_end_fraction,
            )
            student_count = 0
            target_error = 0.0
            for _ in range(agent_cfg.num_steps_per_env):
                with torch.no_grad():
                    student_action = model.act_inference(observations)
                    teacher_raw = model.evaluate(observations)
                    label = student_teacher_target(
                        teacher_raw,
                        motion.reference["dof_pos"],
                        action_term.residual_scale,
                        action_term.lower,
                        action_term.upper,
                        action_term.nominal,
                    )
                    executed, student_mask = student_mix_actions(label, student_action, student_probability)
                    target_error += functional.l1_loss(student_action, label).item()
                student_count += int(student_mask.sum().item())
                replay.add(observations["policy"], label)
                action_term.set_teacher_residual(teacher_raw, ~student_mask)
                observations, _, _, _ = env.step(executed)

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
            execution_rate = student_count / (agent_cfg.num_steps_per_env * base.num_envs)
            writer.add_scalar("DAgger/behavior_mse", mean_loss, iteration)
            writer.add_scalar("DAgger/recent_behavior_mse", recent_loss, iteration)
            writer.add_scalar("DAgger/student_probability", student_probability, iteration)
            writer.add_scalar("DAgger/student_execution_rate", execution_rate, iteration)
            writer.add_scalar("DAgger/student_teacher_action_mae", target_error / agent_cfg.num_steps_per_env, iteration)
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
                torch.save(payload, run / f"model_{iteration + 1}.pt")

        policy = StudentPolicy(agent_cfg.policy.student_hidden_dims, agent_cfg.policy.activation).to(base.device)
        policy.student.load_state_dict(model.student.state_dict())
        student_export(policy.eval(), run / "exported", metadata)
        print(f"Student artifacts: {run / 'exported'}")
    finally:
        writer.close()
        env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        app.close()
