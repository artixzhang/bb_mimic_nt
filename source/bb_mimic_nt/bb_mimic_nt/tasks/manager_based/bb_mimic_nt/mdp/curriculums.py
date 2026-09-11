# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Curriculum logging hooks."""

from __future__ import annotations

from bb_mimic_nt.training import TEACHER_OBSERVATION_DIM

from .commands import MotionReferenceCommand, rsi_probability
from .contacts import CONTACT_SENSOR_NAMES


def rsi_curriculum(env, env_ids, command_name: str = "motion") -> dict[str, float]:
    """Expose the command's centralized RSI schedule to the curriculum log."""
    term: MotionReferenceCommand = env.command_manager.get_term(command_name)
    if not getattr(term, "_environment_contract_validated", False):
        if env.action_manager.total_action_dim != 29:
            raise ValueError(f"Expected a 29-dimensional action, got {env.action_manager.total_action_dim}.")
        observation_shape = env.observation_manager.group_obs_dim["policy"]
        if observation_shape != (TEACHER_OBSERVATION_DIM,):
            raise ValueError(
                f"Expected policy observation shape ({TEACHER_OBSERVATION_DIM},), got {observation_shape}."
            )
        missing_sensors = set(CONTACT_SENSOR_NAMES).difference(env.scene.keys())
        if missing_sensors:
            raise ValueError(f"Missing filtered contact sensors: {sorted(missing_sensors)}")
        if env.reward_manager.active_terms != ["unified_mimic"]:
            raise ValueError(f"Expected one unified reward term, got {env.reward_manager.active_terms}.")
        term._environment_contract_validated = True
    probability = 0.0
    if term.cfg.enable_rsi:
        probability = rsi_probability(
            env.common_step_counter,
            term.cfg.rsi_decay_steps,
            term.cfg.rsi_initial_probability,
        )
    return {"probability": probability}
