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


def contact_graph(env, hand_threshold: float = 1.0, foot_threshold: float = 5.0) -> torch.Tensor:
    thresholds = (hand_threshold, hand_threshold, foot_threshold, foot_threshold)
    return torch.stack(
        [sensor_contact(env, name, threshold) for name, threshold in zip(CONTACT_SENSOR_NAMES, thresholds)], dim=-1
    ).float()
