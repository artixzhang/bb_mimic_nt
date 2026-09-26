# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Contact-graph helpers shared by rewards and terminations."""

from __future__ import annotations

import torch

CONTACT_SENSOR_NAMES = (
    "left_hand_ball_contact",
    "right_hand_ball_contact",
    "left_foot_ground_contact",
    "right_foot_ground_contact",
)


def sensor_contact(env, sensor_name: str, threshold: float) -> torch.Tensor:
    data = env.scene[sensor_name].data
    force = data.force_matrix_w
    if force is None:
        force = data.net_forces_w
    norm = torch.linalg.vector_norm(force, dim=-1)
    return norm.reshape(norm.shape[0], -1).amax(dim=-1) > threshold


def contact_graph(
    env,
    hand_threshold: float = 1.0,
    foot_threshold: float = 5.0,
    hand_distance_threshold: float = 0.08,
    command_name: str = "motion",
) -> torch.Tensor:
    thresholds = (hand_threshold, hand_threshold, foot_threshold, foot_threshold)
    # 1. 原始的 4 通道接触力判定 (bool: left_hand, right_hand, left_foot, right_foot)
    force_contact = torch.stack(
        [sensor_contact(env, name, threshold) for name, threshold in zip(CONTACT_SENSOR_NAMES, thresholds)], dim=-1
    )

    # 2. 计算左右手 target anchor 到球心的实际欧氏距离
    motion = env.command_manager.get_term(command_name)
    anchor_pos = motion.actual_anchor_pos_w()  # shape: (num_envs, 2, 3), 对应 left/right 手掌球心位
    ball_pos = motion.ball.data.root_pos_w     # shape: (num_envs, 3)
    hand_dist = torch.linalg.vector_norm(anchor_pos - ball_pos[:, None, :], dim=-1)  # (num_envs, 2)

    # 3. 手部采用与逻辑：有接触力 AND 距离在手窝容差范围内
    hand_in_range = hand_dist < hand_distance_threshold
    valid_hand_contact = force_contact[:, :2] & hand_in_range

    # 4. 双手与双脚拼接 (双脚保持原有的纯接触力判定)
    return torch.cat((valid_hand_contact, force_contact[:, 2:]), dim=-1).float()
