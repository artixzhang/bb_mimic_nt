# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Curriculum logging hooks."""

from __future__ import annotations

from .commands import MotionReferenceCommand, rsi_probability
from .contacts import CONTACT_SENSOR_NAMES
from .events import curriculum_strength, get_dr_context, training_policy_steps


def rsi_curriculum(env, env_ids, command_name: str = "motion") -> dict[str, float]:
    """Expose the command's centralized RSI schedule to the curriculum log."""
    term: MotionReferenceCommand = env.command_manager.get_term(command_name)
    if not getattr(term, "_environment_contract_validated", False):
        if env.action_manager.total_action_dim != 29:
            raise ValueError(f"Expected a 29-dimensional action, got {env.action_manager.total_action_dim}.")
        observation_shape = env.observation_manager.group_obs_dim["policy"]
        expected_dimension = 673 + term.command_dimension
        if observation_shape != (expected_dimension,):
            raise ValueError(
                f"Expected policy observation shape ({expected_dimension},), got {observation_shape}."
            )
        missing_sensors = set(CONTACT_SENSOR_NAMES).difference(env.scene.keys())
        if missing_sensors:
            raise ValueError(f"Missing filtered contact sensors: {sorted(missing_sensors)}")
        if env.reward_manager.active_terms != ["unified_mimic"]:
            raise ValueError(f"Expected one unified reward term, got {env.reward_manager.active_terms}.")
        term._environment_contract_validated = True
    return rsi_schedule(env, command_name)


def rsi_schedule(env, command_name: str = "motion") -> dict[str, float]:
    """Report live RSI probability for either Teacher or Student environments."""
    term: MotionReferenceCommand = env.command_manager.get_term(command_name)
    probability = 0.0
    if term.cfg.enable_rsi:
        probability = rsi_probability(
            training_policy_steps(env),
            term.cfg.rsi_decay_steps,
            term.cfg.rsi_initial_probability,
        )
    return {"probability": probability}


def dr_curriculum(env, env_ids) -> dict[str, float]:
    cfg = env.cfg.dr
    iteration = get_dr_context(env).iteration_offset + env.common_step_counter / cfg.steps_per_iteration
    strengths = {}
    for name in ("delay", "ball_mass", "pd_gains", "hand_friction", "foot_friction", "link_mass", "push", "global_push", "object_push", "velocity_push"):
        schedule = getattr(cfg, name, None)
        strengths[name] = 0.0 if schedule is None else curriculum_strength(schedule, iteration, cfg.total_iterations)
    initial = getattr(cfg, "init_state", None)
    for name in ("dof_pos", "dof_vel", "root_pos", "root_rot", "root_vel", "root_ang_vel", "object_pos"):
        strengths[f"init_state/{name}"] = float(
            initial is not None and initial.enabled and getattr(initial, name).enabled
        )
    return strengths
