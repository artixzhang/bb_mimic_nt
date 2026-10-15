# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Shared defaults for Teacher PPO, Student DAgger, and environment curricula."""

NUM_ENVS = 8192  # Per process; shared by Teacher and Student training scenes.
PPO_STEPS_PER_ENV = 48
PPO_MAX_ITERATIONS = 8_000
RSI_INITIAL_PROBABILITY = 0.3
RSI_DECAY_FRACTION = 0.15
TOTAL_POLICY_STEPS = PPO_STEPS_PER_ENV * PPO_MAX_ITERATIONS
RSI_DECAY_STEPS = int(TOTAL_POLICY_STEPS * RSI_DECAY_FRACTION)
TEACHER_OBSERVATION_DIM = 676  # Default three-dimensional skill command.
TEACHER_HISTORY_LENGTH = 5

DAGGER_STEPS_PER_ENV = PPO_STEPS_PER_ENV
# Keep at least one complete rollout in the recent half of replay.
DAGGER_REPLAY_CAPACITY = 2 * NUM_ENVS * max(32, DAGGER_STEPS_PER_ENV)
DAGGER_BATCH_SIZE = NUM_ENVS


def configure_training_curricula(env_cfg, total_iterations: int, steps_per_iteration: int, *, motion_cfg=None) -> None:
    """Use the actual training budget for DR and the shared RSI decay fraction.

    Pass the live command configuration after environment creation: command
    managers copy their input configurations, including the RSI deadline.
    Explicit probability overrides and per-skill/play switches are preserved.
    """
    if total_iterations < 1 or steps_per_iteration < 1:
        raise ValueError("Training iterations and rollout length must be positive.")
    env_cfg.dr.total_iterations = total_iterations
    env_cfg.dr.steps_per_iteration = steps_per_iteration
    decay_steps = int(total_iterations * steps_per_iteration * RSI_DECAY_FRACTION)
    env_cfg.commands.motion.rsi_decay_steps = decay_steps
    if motion_cfg is not None:
        motion_cfg.rsi_decay_steps = decay_steps
