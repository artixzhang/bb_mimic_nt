# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Shared defaults for Teacher PPO, Student DAgger, and environment curricula."""

NUM_ENVS = 8192  # Per process; shared by Teacher and Student training scenes.
PPO_STEPS_PER_ENV = 48
PPO_MAX_ITERATIONS = 7_000
RSI_DECAY_FRACTION = 0.30
TOTAL_POLICY_STEPS = PPO_STEPS_PER_ENV * PPO_MAX_ITERATIONS
RSI_DECAY_STEPS = int(TOTAL_POLICY_STEPS * RSI_DECAY_FRACTION)
TEACHER_OBSERVATION_DIM = 674
TEACHER_HISTORY_LENGTH = 5

DAGGER_STEPS_PER_ENV = PPO_STEPS_PER_ENV
# Keep at least one complete rollout in the recent half of replay.
DAGGER_REPLAY_CAPACITY = 2 * NUM_ENVS * max(32, DAGGER_STEPS_PER_ENV)
DAGGER_BATCH_SIZE = NUM_ENVS
