# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Training constants shared by environment curricula and PPO."""

PPO_STEPS_PER_ENV = 24
PPO_MAX_ITERATIONS = 3_000
RSI_DECAY_FRACTION = 0.30
TOTAL_POLICY_STEPS = PPO_STEPS_PER_ENV * PPO_MAX_ITERATIONS
RSI_DECAY_STEPS = int(TOTAL_POLICY_STEPS * RSI_DECAY_FRACTION)
TEACHER_OBSERVATION_DIM = 473
