"""Student-only ball-flight statistics with an identically zero training reward."""

from __future__ import annotations

import torch

from isaaclab.managers import ManagerTermBase


class StudentBallFlightMetrics(ManagerTermBase):
    """Track ball peak height and first ground landing without shaping the policy."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        count = env.num_envs
        self.peak_height = torch.full((count,), float("-inf"), device=env.device)
        self.landing_xy = torch.full((count, 2), float("nan"), device=env.device)
        self.landed = torch.zeros(count, dtype=torch.bool, device=env.device)
        self.last_peak_height = self.peak_height.clone()
        self.last_landing_xy = self.landing_xy.clone()
        self.last_landed = self.landed.clone()

    def __call__(self, env) -> torch.Tensor:
        motion = env.command_manager.get_term("motion")
        ball_position = motion.ball.data.root_pos_w
        local_position = ball_position - env.scene.env_origins
        self.peak_height.copy_(torch.maximum(self.peak_height, local_position[:, 2]))
        forces = env.scene["ball_ground_contact"].data.force_matrix_w
        contact = torch.linalg.vector_norm(forces, dim=-1).reshape(env.num_envs, -1).amax(dim=-1) > 1.0
        first_contact = contact & ~self.landed
        self.landing_xy[first_contact] = (
            ball_position - motion.hoop_center_pos_w()
        )[first_contact, :2]
        self.landed |= first_contact
        return torch.zeros(env.num_envs, device=env.device)

    def reset(self, env_ids=None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        self.last_peak_height[env_ids] = self.peak_height[env_ids]
        self.last_landing_xy[env_ids] = self.landing_xy[env_ids]
        self.last_landed[env_ids] = self.landed[env_ids]
        self.peak_height[env_ids] = float("-inf")
        self.landing_xy[env_ids] = float("nan")
        self.landed[env_ids] = False
