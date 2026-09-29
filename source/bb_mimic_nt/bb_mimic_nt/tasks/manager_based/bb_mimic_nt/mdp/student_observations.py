"""Student-only sensor history, observation noise, and observation latency."""

from __future__ import annotations

import torch

from isaaclab.managers import ManagerTermBase
from isaaclab.utils.math import quat_apply_inverse

from bb_mimic_nt.student import student_delay_bounds, student_history, student_observation_fields

from .events import curriculum_strength


class StudentObservation(ManagerTermBase):
    """Build deployable Student history with per-episode latency and sensor noise."""

    _SENSOR_DIM = 64
    _ACTION_DIM = 29

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        settings = env.cfg.student_observation
        _, maximum_delay = student_delay_bounds(
            settings.delay_nominal_steps, settings.delay_min_steps, settings.delay_max_steps
        )
        self.history_length = settings.history_length
        self.observation_dim = sum(size for _, size in student_observation_fields(self.history_length))
        count = env.num_envs
        sensor_slots = maximum_delay + self.history_length
        self._sensor = torch.zeros((sensor_slots, count, self._SENSOR_DIM), device=env.device)
        self._action = torch.zeros((self.history_length, count, self._ACTION_DIM), device=env.device)
        self._initial_hoop = torch.zeros((count, 3), device=env.device)
        self._sensor_cursor = torch.full((count,), -1, dtype=torch.long, device=env.device)
        self._action_cursor = torch.full((count,), -1, dtype=torch.long, device=env.device)
        self._last_step = torch.full((count,), -1, dtype=torch.long, device=env.device)
        self._pending_reset = torch.ones(count, dtype=torch.bool, device=env.device)
        self.delay_steps = torch.full(
            (count,), settings.delay_nominal_steps, dtype=torch.long, device=env.device
        )
        self._env_ids = torch.arange(count, device=env.device)
        self._noise_std = torch.tensor(
            [settings.gravity_noise_std] * 3
            + [settings.angular_velocity_noise_std] * 3
            + [settings.joint_position_noise_std] * 29
            + [settings.joint_velocity_noise_std] * 29,
            device=env.device,
        )
        env.student_observation = self

    @property
    def delay_encoding(self) -> torch.Tensor:
        settings = self._env.cfg.student_observation
        return (
            self.delay_steps.float() - settings.delay_nominal_steps
        ) / max(
            settings.delay_nominal_steps - settings.delay_min_steps,
            settings.delay_max_steps - settings.delay_nominal_steps,
            1,
        )

    @property
    def curriculum_strength(self) -> float:
        env = self._env
        context = getattr(env, "dr_context", None)
        offset = 0 if context is None else context.iteration_offset
        iteration = offset + env.common_step_counter / env.cfg.dr.steps_per_iteration
        return curriculum_strength(env.cfg.student_observation, iteration, env.cfg.dr.total_iterations)

    def reset(self, env_ids=None) -> None:
        if env_ids is None:
            env_ids = self._env_ids
        elif not isinstance(env_ids, torch.Tensor):
            env_ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        self._pending_reset[env_ids] = True
        settings = self._env.cfg.student_observation
        if settings.randomize_delay:
            low, high = student_delay_bounds(
                settings.delay_nominal_steps, settings.delay_min_steps, settings.delay_max_steps,
                self.curriculum_strength,
            )
            self.delay_steps[env_ids] = torch.randint(low, high + 1, (len(env_ids),), device=self.device)
        else:
            self.delay_steps[env_ids] = settings.delay_nominal_steps

    def __call__(self, env) -> torch.Tensor:
        motion = env.command_manager.get_term("motion")
        action = env.action_manager.get_term("joint_pos")
        robot = motion.robot
        settings = env.cfg.student_observation
        step = env.common_step_counter
        fresh = self._pending_reset | (self._last_step != step)
        if torch.any(fresh):
            ids = self._env_ids[fresh]
            self._sensor_cursor[ids] = (self._sensor_cursor[ids] + 1) % self._sensor.shape[0]
            self._action_cursor[ids] = (self._action_cursor[ids] + 1) % self._action.shape[0]
            sensor = torch.cat(
                (
                    robot.data.projected_gravity_b,
                    quat_apply_inverse(robot.data.root_link_quat_w, robot.data.root_link_ang_vel_w),
                    robot.data.joint_pos[:, motion.joint_ids] - action.nominal,
                    robot.data.joint_vel[:, motion.joint_ids],
                ),
                dim=-1,
            )
            if settings.enable_noise and settings.noise_strength > 0.0:
                sensor = sensor + (
                    torch.randn_like(sensor) * self._noise_std * settings.noise_strength * self.curriculum_strength
                )
            applied_action = action.actual_action_offset
            reset_ids = ids[self._pending_reset[ids]]
            if len(reset_ids):
                hoop_delta = motion.hoop_center_pos_w()[reset_ids] - robot.data.root_link_pos_w[reset_ids]
                self._initial_hoop[reset_ids] = quat_apply_inverse(
                    robot.data.root_link_quat_w[reset_ids], hoop_delta
                )
                self._sensor[:, reset_ids] = sensor[reset_ids]
                self._action[:, reset_ids] = applied_action[reset_ids]
            self._sensor[self._sensor_cursor[ids], ids] = sensor[ids]
            self._action[self._action_cursor[ids], ids] = applied_action[ids]
            self._last_step[ids] = step
            self._pending_reset[ids] = False

        sensor_frames = student_history(self._sensor, self._sensor_cursor, self.delay_steps, self.history_length)
        action_frames = student_history(
            self._action,
            self._action_cursor,
            torch.zeros_like(self._action_cursor),
            self.history_length,
        )
        scales = settings.scales
        observation = torch.cat(
            (
                motion.normalized_phase[:, None] * scales.phase,
                sensor_frames[:, :, :3].flatten(1) * scales.gravity,
                sensor_frames[:, :, 3:6].flatten(1) * scales.angular_velocity,
                sensor_frames[:, :, 6:35].flatten(1) * scales.joint_position,
                sensor_frames[:, :, 35:64].flatten(1) * scales.joint_velocity,
                action_frames.flatten(1) * scales.applied_action,
                self._initial_hoop * scales.hoop_position,
            ),
            dim=-1,
        )
        if observation.shape[-1] != self.observation_dim:
            raise RuntimeError(f"Student observation contract changed to {observation.shape[-1]} values.")
        return observation
