# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Teacher-policy observations for basketball imitation."""

from __future__ import annotations

import torch

from isaaclab.utils.math import quat_apply_inverse, quat_inv, quat_mul

from .commands import MotionReferenceCommand


def motion_command(env, name: str = "motion") -> MotionReferenceCommand:
    return env.command_manager.get_term(name)


def gravity_vec_b(env) -> torch.Tensor:
    return motion_command(env).robot.data.projected_gravity_b


def root_pos_z_w(env) -> torch.Tensor:
    term = motion_command(env)
    return (term.robot.data.root_link_pos_w - env.scene.env_origins)[:, 2:3]


def root_quat_w(env) -> torch.Tensor:
    quat = motion_command(env).robot.data.root_link_quat_w
    return torch.where(quat[:, :1] < 0.0, -quat, quat)


def root_lin_vel_b(env) -> torch.Tensor:
    term = motion_command(env)
    return quat_apply_inverse(term.robot.data.root_link_quat_w, term.robot.data.root_link_lin_vel_w)


def root_ang_vel_b(env) -> torch.Tensor:
    term = motion_command(env)
    return quat_apply_inverse(term.robot.data.root_link_quat_w, term.robot.data.root_link_ang_vel_w)


def dof_pos(env) -> torch.Tensor:
    term = motion_command(env)
    return term.robot.data.joint_pos[:, term.joint_ids]


def dof_vel(env) -> torch.Tensor:
    term = motion_command(env)
    return term.robot.data.joint_vel[:, term.joint_ids]


def link_pos_b(env) -> torch.Tensor:
    return motion_command(env).actual_link_pos_b().flatten(1)


def _position_in_root_frame(position_w: torch.Tensor, term: MotionReferenceCommand) -> torch.Tensor:
    return quat_apply_inverse(term.robot.data.root_link_quat_w, position_w - term.robot.data.root_link_pos_w)


def object_pos_b(env) -> torch.Tensor:
    term = motion_command(env)
    return _position_in_root_frame(term.ball.data.root_pos_w, term)


def object_lin_vel_b(env) -> torch.Tensor:
    term = motion_command(env)
    return quat_apply_inverse(term.robot.data.root_link_quat_w, term.ball.data.root_lin_vel_w)


def hoop_pos_b(env) -> torch.Tensor:
    term = motion_command(env)
    return _position_in_root_frame(term.hoop_center_pos_w(), term)


def previous_action(env) -> torch.Tensor:
    # Expose the bounded residual used to form the PD target.
    return env.action_manager.get_term("joint_pos").raw_actions


def pd_error(env) -> torch.Tensor:
    term = motion_command(env)
    target = env.action_manager.get_term("joint_pos").desired_actions
    return target - term.robot.data.joint_pos[:, term.joint_ids]


def delta_root_pos_b(env) -> torch.Tensor:
    term = motion_command(env)
    reference_w = term.reference["root_pos"] + env.scene.env_origins
    return _position_in_root_frame(reference_w, term)


def delta_root_quat_b(env) -> torch.Tensor:
    term = motion_command(env)
    delta = quat_mul(quat_inv(term.robot.data.root_link_quat_w), term.reference["root_quat"])
    return torch.where(delta[:, :1] < 0.0, -delta, delta)


def delta_dof_pos(env) -> torch.Tensor:
    term = motion_command(env)
    return term.reference["dof_pos"] - term.robot.data.joint_pos[:, term.joint_ids]


def delta_link_pos_b(env) -> torch.Tensor:
    term = motion_command(env)
    return (term.reference["link_pos_b"] - term.actual_link_pos_b()).flatten(1)


def delta_object_pos_w(env) -> torch.Tensor:
    term = motion_command(env)
    ball_local_w = term.ball.data.root_pos_w - env.scene.env_origins
    return term.reference["object_pos"] - ball_local_w


def reference_dof_pos(env) -> torch.Tensor:
    return motion_command(env).reference["dof_pos"]


def reference_root_lin_vel_b(env) -> torch.Tensor:
    term = motion_command(env)
    return quat_apply_inverse(term.robot.data.root_link_quat_w, term.reference["root_lin_vel"])


def reference_root_ang_vel_b(env) -> torch.Tensor:
    term = motion_command(env)
    return quat_apply_inverse(term.robot.data.root_link_quat_w, term.reference["root_ang_vel"])


def reference_dof_vel(env) -> torch.Tensor:
    return motion_command(env).reference["dof_vel"]


def reference_link_pos_b(env) -> torch.Tensor:
    return motion_command(env).reference["link_pos_b"].flatten(1)


def reference_object_pos_w(env) -> torch.Tensor:
    return motion_command(env).reference["object_pos"]


def reference_object_lin_vel_b(env) -> torch.Tensor:
    term = motion_command(env)
    return quat_apply_inverse(term.robot.data.root_link_quat_w, term.reference["object_lin_vel"])


def reference_contact(env) -> torch.Tensor:
    return motion_command(env).reference["contact"]


def phase(env) -> torch.Tensor:
    return motion_command(env).normalized_phase.unsqueeze(-1)


def reference_speed(env) -> torch.Tensor:
    return motion_command(env).speed.unsqueeze(-1)
