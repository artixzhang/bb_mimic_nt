# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Failure and timeout conditions for the basketball task."""

from __future__ import annotations

import math

import torch

from .commands import MotionReferenceCommand


def _motion(env, command_name: str) -> MotionReferenceCommand:
    return env.command_manager.get_term(command_name)


def reference_finished(env, command_name: str = "motion") -> torch.Tensor:
    return _motion(env, command_name).motion_done


def fall_or_tilt(
    env, minimum_height: float = 0.45, maximum_tilt: float = math.radians(60.0), command_name: str = "motion"
) -> torch.Tensor:
    robot = _motion(env, command_name).robot
    tilt = torch.acos((-robot.data.projected_gravity_b[:, 2]).clamp(-1.0, 1.0))
    return (robot.data.root_link_pos_w[:, 2] < minimum_height) | (tilt > maximum_tilt)


def mechanical_joint_limit(
    env, tolerance: float = 0.01, command_name: str = "motion"
) -> torch.Tensor:
    term = _motion(env, command_name)
    joint_pos = term.robot.data.joint_pos[:, term.joint_ids]
    limits = term.robot.data.joint_pos_limits[:, term.joint_ids]
    return torch.any(
        (joint_pos < limits[..., 0] - tolerance) | (joint_pos > limits[..., 1] + tolerance),
        dim=-1,
    )


def non_finite_state(env, command_name: str = "motion") -> torch.Tensor:
    term = _motion(env, command_name)
    tensors = (
        term.robot.data.root_link_state_w,
        term.robot.data.joint_pos,
        term.robot.data.joint_vel,
        term.ball.data.root_state_w,
        env.action_manager.action,
        *term.reference.values(),
    )
    result = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    for tensor in tensors:
        result |= ~torch.isfinite(tensor).reshape(env.num_envs, -1).all(dim=-1)
    return result


def root_tracking_error(env, maximum_error: float = 1.5, command_name: str = "motion") -> torch.Tensor:
    term = _motion(env, command_name)
    root_local = term.robot.data.root_link_pos_w - env.scene.env_origins
    return torch.linalg.vector_norm(root_local - term.reference["root_pos"], dim=-1) > maximum_error


def reference_dof_error(env, maximum_rmse: float = 1.0, command_name: str = "motion") -> torch.Tensor:
    term = _motion(env, command_name)
    error = term.robot.data.joint_pos[:, term.joint_ids] - term.reference["dof_pos"]
    return torch.sqrt(torch.mean(error**2, dim=-1)) > maximum_rmse


def interaction_tracking_error(env, maximum_distance: float = 0.35, command_name: str = "motion") -> torch.Tensor:
    term = _motion(env, command_name)
    distance = torch.linalg.vector_norm(term.actual_anchor_pos_w() - term.ball.data.root_pos_w[:, None], dim=-1)
    required = term.reference["contact"][:, :2] > 0.5
    return torch.any(required & (distance > maximum_distance), dim=-1)
