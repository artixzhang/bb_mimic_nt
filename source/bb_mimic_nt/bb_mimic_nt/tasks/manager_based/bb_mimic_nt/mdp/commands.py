# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Motion-batch command used by the basketball Teacher policy."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import MISSING
from pathlib import Path

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_apply, quat_apply_inverse, quat_error_magnitude, quat_inv, quat_mul

from bb_mimic_nt.core import (
    advance_reference_frame,
    downward_hoop_crossing,
    interpolate_motion,
    rsi_probability,
    select_adaptive_speed,
)
from bb_mimic_nt.trajectory import (
    TRACKED_BODY_NAMES,
    cache_is_current,
    load_motion_batch,
    preprocess_motion_batch,
)

from .contacts import contact_graph
from .events import arm_push, training_policy_steps

STAGE_PRE_HOLD = 0
STAGE_ACTIVE = 1
STAGE_POST_HOLD = 2
STAGE_DONE = 3


class MotionReferenceCommand(CommandTerm):
    """Sample, interpolate, reset, and advance reference motion clips."""

    cfg: "MotionReferenceCommandCfg"

    def __init__(self, cfg: "MotionReferenceCommandCfg", env):
        source = Path(cfg.source_path)
        urdf = Path(cfg.urdf_path)
        cache = Path(cfg.cache_path)
        if cfg.auto_preprocess and not cache_is_current(cache, source, urdf, tuple(cfg.tracked_body_names)):
            preprocess_motion_batch(source, urdf, cache, tracked_body_names=tuple(cfg.tracked_body_names))
        if not cache.is_file():
            raise FileNotFoundError(
                f"Motion cache not found: {cache}. Run scripts/preprocess_trajectory.py or enable auto_preprocess."
            )
        self._batch = load_motion_batch(cache, device=env.device)
        super().__init__(cfg, env)

        self.robot: Articulation = env.scene[cfg.robot_name]
        self.ball: RigidObject = env.scene[cfg.ball_name]
        self.hoop: RigidObject = env.scene[cfg.hoop_name]
        self.hoop_center_offset = torch.tensor(cfg.hoop_center_offset, dtype=torch.float, device=self.device)
        self.hoop_root_quat = torch.tensor(cfg.hoop_root_quat, dtype=torch.float, device=self.device)
        self.hoop_root_quat /= torch.linalg.vector_norm(self.hoop_root_quat).clamp_min(1.0e-8)
        self._hoop_pose_validated = False
        metadata = self._batch["metadata"]
        self.dof_names = tuple(metadata["dof_names"])
        self.tracked_body_names = tuple(metadata["tracked_body_names"])
        self.rotation_body_names = tuple(metadata["rotation_body_names"])
        self.rotation_body_weights = torch.tensor(
            metadata["rotation_body_weights"], dtype=torch.float, device=self.device
        )
        if len(self.rotation_body_names) != len(self.rotation_body_weights) or torch.any(self.rotation_body_weights <= 0):
            raise ValueError("Rotation-only tracked bodies must have positive matching weights.")
        self.contact_names = tuple(metadata["contact_names"])
        self.anchor_offsets = torch.tensor(
            [metadata["anchor_offsets"][name] for name in metadata["anchor_body_names"]],
            dtype=torch.float,
            device=self.device,
        )
        self.joint_ids, resolved_joint_names = self.robot.find_joints(self.dof_names, preserve_order=True)
        self.body_ids, resolved_body_names = self.robot.find_bodies(self.tracked_body_names, preserve_order=True)
        self.rotation_body_ids, resolved_rotation_names = self.robot.find_bodies(
            self.rotation_body_names, preserve_order=True
        )
        if tuple(resolved_joint_names) != self.dof_names:
            raise ValueError(f"Robot/cache joint mismatch: {resolved_joint_names} != {self.dof_names}")
        if tuple(resolved_body_names) != self.tracked_body_names:
            raise ValueError(f"Robot/cache body mismatch: {resolved_body_names} != {self.tracked_body_names}")
        if tuple(resolved_rotation_names) != self.rotation_body_names:
            raise ValueError(
                f"Robot/cache rotation body mismatch: {resolved_rotation_names} != {self.rotation_body_names}"
            )
        if tuple(self.tracked_body_names[:2]) != ("left_hand", "right_hand"):
            raise ValueError("The first two tracked bodies must be left_hand and right_hand.")
        for anchor_path in (
            f"{self.robot.cfg.prim_path}/left_hand/left_hand/center_of_ball",
            f"{self.robot.cfg.prim_path}/right_hand/right_hand/center_of_ball",
        ):
            matches = sim_utils.find_matching_prim_paths(anchor_path)
            if len(matches) != self.num_envs:
                raise ValueError(
                    f"Expected one center_of_ball prim per environment for {anchor_path!r}, found {len(matches)}."
                )

        self.lengths = self._batch["lengths"]
        valid_dof = self._batch["dof_pos"][self._batch["valid"]]
        self.minimum_reference_dof_pos = valid_dof.amin(dim=0)
        self.maximum_reference_dof_pos = valid_dof.amax(dim=0)
        self.maximum_reference_dof_speed = self._batch["dof_vel"].abs().amax(dim=(0, 1))
        self.num_clips = int(metadata["num_clips"])
        self.fps = float(metadata["fps"])
        if not 0 <= cfg.default_clip_id < self.num_clips:
            raise ValueError(f"default_clip_id must be between 0 and {self.num_clips - 1}.")
        frame_index = torch.arange(self._batch["contact"].shape[1], device=self.device)[None]
        valid_hand_contact = (
            torch.any(self._batch["contact"][..., :2] > 0.5, dim=-1)
            & (frame_index < self.lengths[:, None])
        )
        self.release_frame = torch.where(valid_hand_contact, frame_index, -1).amax(dim=1).float()
        self.clip_ids = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.fixed_clip_ids = torch.full_like(self.clip_ids, -1)
        self.frame = torch.zeros(self.num_envs, device=self.device)
        self.stage = torch.full_like(self.clip_ids, STAGE_PRE_HOLD)
        self.hold_time = torch.zeros(self.num_envs, device=self.device)
        self.speed = torch.zeros(self.num_envs, device=self.device)
        self.tracking_error_ema = torch.full(
            (self.num_envs,), cfg.adaptive_initial_error, device=self.device
        )
        self.speed_update_elapsed = torch.zeros(self.num_envs, device=self.device)
        self._adaptive_error_alpha = 1.0 - math.exp(
            -env.step_dt / max(cfg.adaptive_error_time_constant_s, env.step_dt)
        )
        self.rsi_started = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.shot_success = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.shot_eligible = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        self.previous_ball_above_hoop = torch.zeros(self.num_envs, device=self.device)
        self.reference: dict[str, torch.Tensor] = {}
        self.metrics = {
            "shot_success": torch.zeros(self.num_envs, device=self.device),
            "mean_speed": torch.zeros(self.num_envs, device=self.device),
            "root_rmse": torch.zeros(self.num_envs, device=self.device),
            "dof_rmse": torch.zeros(self.num_envs, device=self.device),
            "link_rmse": torch.zeros(self.num_envs, device=self.device),
            "ball_rmse": torch.zeros(self.num_envs, device=self.device),
            "error/root_rotation_rad": torch.zeros(self.num_envs, device=self.device),
            "error/root_linear_velocity_mps": torch.zeros(self.num_envs, device=self.device),
            "error/root_angular_velocity_radps": torch.zeros(self.num_envs, device=self.device),
            "error/dof_velocity_rmse_radps": torch.zeros(self.num_envs, device=self.device),
            "error/link_rotation_rmse_rad": torch.zeros(self.num_envs, device=self.device),
            "error/ball_rotation_rad": torch.zeros(self.num_envs, device=self.device),
            "error/ball_linear_velocity_mps": torch.zeros(self.num_envs, device=self.device),
            "error/ball_angular_velocity_radps": torch.zeros(self.num_envs, device=self.device),
            "error/relative_position_rmse_m": torch.zeros(self.num_envs, device=self.device),
            "error/contact_mismatch_rate": torch.zeros(self.num_envs, device=self.device),
            "control/raw_action_rms": torch.zeros(self.num_envs, device=self.device),
            "control/raw_action_delta_rms": torch.zeros(self.num_envs, device=self.device),
            "control/action_saturation_rate": torch.zeros(self.num_envs, device=self.device),
            "control/joint_target_delta_rms_rad": torch.zeros(self.num_envs, device=self.device),
            "control/reference_joint_delta_rms_rad": torch.zeros(self.num_envs, device=self.device),
            "control/pd_error_rmse_rad": torch.zeros(self.num_envs, device=self.device),
            "control/joint_velocity_rms_radps": torch.zeros(self.num_envs, device=self.device),
            "control/torque_limit_ratio_rms": torch.zeros(self.num_envs, device=self.device),
        }
        self.metrics.update(
            {
                f"error/rotation/{name}_rad": torch.zeros(self.num_envs, device=self.device)
                for name in self.rotation_body_names
            }
        )
        self._time_averaged_metric_names = tuple(name for name in self.metrics if name != "shot_success")
        self.last_episode_metrics = {
            name: torch.zeros(self.num_envs, device=self.device) for name in self.metrics
        }
        self.last_episode_clip_ids = torch.full_like(self.clip_ids, -1)
        self._refresh_reference()
        self._previous_reference_dof_pos = self.reference["dof_pos"].clone()

    @property
    def command(self) -> torch.Tensor:
        return torch.stack((self.normalized_phase, self.speed), dim=-1)

    @property
    def normalized_phase(self) -> torch.Tensor:
        last = (self.lengths[self.clip_ids] - 1).clamp_min(1).float()
        return (self.frame / last).clamp(0.0, 1.0)

    @property
    def object_reward_active(self) -> torch.Tensor:
        return self.stage < STAGE_POST_HOLD

    @property
    def motion_done(self) -> torch.Tensor:
        return self.stage == STAGE_DONE

    @property
    def in_active_motion(self) -> torch.Tensor:
        return self.stage == STAGE_ACTIVE

    def _sample(self, field: str, quaternion: bool = False) -> torch.Tensor:
        return interpolate_motion(
            self._batch[field],
            self.clip_ids,
            self.frame,
            self.lengths,
            quaternion=quaternion,
        )

    def _refresh_reference(self) -> None:
        quaternion_fields = {"root_quat", "object_quat", "link_quat_w", "link_quat_b", "rotation_quat_b"}
        fields = (
            "root_pos",
            "root_quat",
            "root_lin_vel",
            "root_ang_vel",
            "dof_pos",
            "dof_vel",
            "object_pos",
            "object_quat",
            "object_lin_vel",
            "object_ang_vel",
            "contact",
            "hoop_pos",
            "link_pos_w",
            "link_quat_w",
            "link_pos_b",
            "link_quat_b",
            "rotation_quat_b",
            "anchor_pos_w",
            "anchor_object_rel_pos_w",
        )
        self.reference = {name: self._sample(name, name in quaternion_fields) for name in fields}
        self.reference["contact"] = (self.reference["contact"] >= 0.5).float()
        frame_ids = torch.floor(self.frame).long().clamp_min(0)
        frame_ids = torch.minimum(frame_ids, self.lengths[self.clip_ids] - 1)
        self.reference["push_available"] = self._batch["push_available"][self.clip_ids, frame_ids] > 0.5
        velocity_fields = (
            "root_lin_vel",
            "root_ang_vel",
            "dof_vel",
            "object_lin_vel",
            "object_ang_vel",
        )
        active = self.in_active_motion.float()
        for name in velocity_fields:
            shape = (-1,) + (1,) * (self.reference[name].ndim - 1)
            self.reference[name] = self.reference[name] * active.view(shape) * self.speed.view(shape)

    def _write_reference_state(self, env_ids: torch.Tensor) -> None:
        origin = self._env.scene.env_origins[env_ids]
        root_pose = torch.cat(
            (self.reference["root_pos"][env_ids] + origin, self.reference["root_quat"][env_ids]), dim=-1
        )
        root_velocity = torch.cat(
            (self.reference["root_lin_vel"][env_ids], self.reference["root_ang_vel"][env_ids]), dim=-1
        )
        self.robot.write_root_pose_to_sim(root_pose, env_ids=env_ids)
        self.robot.write_root_link_velocity_to_sim(root_velocity, env_ids=env_ids)
        self.robot.write_joint_state_to_sim(
            self.reference["dof_pos"][env_ids],
            self.reference["dof_vel"][env_ids],
            joint_ids=self.joint_ids,
            env_ids=env_ids,
        )
        ball_pose = torch.cat(
            (self.reference["object_pos"][env_ids] + origin, self.reference["object_quat"][env_ids]), dim=-1
        )
        ball_velocity = torch.cat(
            (self.reference["object_lin_vel"][env_ids], self.reference["object_ang_vel"][env_ids]), dim=-1
        )
        self.ball.write_root_pose_to_sim(ball_pose, env_ids=env_ids)
        self.ball.write_root_velocity_to_sim(ball_velocity, env_ids=env_ids)
        hoop_quat = self.hoop_root_quat.expand(len(env_ids), -1)
        hoop_offset = self.hoop_center_offset.expand(len(env_ids), -1)
        hoop_root_pos = (
            self.reference["hoop_pos"][env_ids]
            + origin
            - quat_apply(hoop_quat, hoop_offset)
        )
        hoop_pose = torch.cat((hoop_root_pos, hoop_quat), dim=-1)
        self.hoop.write_root_pose_to_sim(hoop_pose, env_ids=env_ids)
        if not self._hoop_pose_validated:
            expected_center = self.reference["hoop_pos"][env_ids] + origin
            center_error = torch.linalg.vector_norm(self.hoop_center_pos_w()[env_ids] - expected_center, dim=-1)
            if torch.any(center_error > 1.0e-4):
                raise RuntimeError(
                    "Floating hoop center transform is inconsistent with hoop_pos_w; "
                    f"maximum error={center_error.max().item():.6f} m."
                )
            self._hoop_pose_validated = True

    def reset(self, env_ids: Sequence[int] | None = None) -> dict[str, float]:
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        elif not isinstance(env_ids, torch.Tensor):
            env_ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        duration = (self._env.episode_length_buf[env_ids].float() * self._env.step_dt).clamp_min(
            self._env.step_dt
        )
        self.last_episode_metrics["shot_success"][env_ids] = self.metrics["shot_success"][env_ids]
        for name in self._time_averaged_metric_names:
            self.last_episode_metrics[name][env_ids] = self.metrics[name][env_ids] / duration
        self.last_episode_clip_ids[env_ids] = self.clip_ids[env_ids]
        extras = {
            name: value[env_ids].mean().item()
            for name, value in self.last_episode_metrics.items()
        }
        extras["clip_id_mean"] = self.last_episode_clip_ids[env_ids].float().mean().item()
        for metric in self.metrics.values():
            metric[env_ids] = 0.0

        if self.cfg.randomize_clip:
            sampled_clips = torch.randint(self.num_clips, (len(env_ids),), device=self.device)
        else:
            sampled_clips = torch.full(
                (len(env_ids),), self.cfg.default_clip_id, dtype=torch.long, device=self.device
            )
        fixed = self.fixed_clip_ids[env_ids]
        self.clip_ids[env_ids] = torch.where(fixed >= 0, fixed, sampled_clips)
        probability = rsi_probability(
            training_policy_steps(self._env), self.cfg.rsi_decay_steps, self.cfg.rsi_initial_probability
        ) if self.cfg.enable_rsi else 0.0
        use_rsi = torch.rand(len(env_ids), device=self.device) < probability
        valid_frame_count = self.lengths[self.clip_ids[env_ids]].clamp_min(1)
        random_frames = torch.floor(torch.rand(len(env_ids), device=self.device) * valid_frame_count).float()
        self.frame[env_ids] = torch.where(use_rsi, random_frames, torch.zeros_like(random_frames))
        self.stage[env_ids] = torch.where(
            use_rsi,
            torch.full_like(self.stage[env_ids], STAGE_ACTIVE),
            torch.full_like(self.stage[env_ids], STAGE_PRE_HOLD),
        )
        self.hold_time[env_ids] = 0.0
        self.speed[env_ids] = torch.where(use_rsi, torch.ones_like(random_frames), torch.zeros_like(random_frames))
        self.tracking_error_ema[env_ids] = self.cfg.adaptive_initial_error
        self.speed_update_elapsed[env_ids] = 0.0
        self.rsi_started[env_ids] = use_rsi
        self.shot_success[env_ids] = False
        self.shot_eligible[env_ids] = (
            self.frame[env_ids] <= self.release_frame[self.clip_ids[env_ids]]
        )
        self._refresh_reference()
        arm_push(self._env, env_ids)
        self._previous_reference_dof_pos[env_ids] = self.reference["dof_pos"][env_ids]
        self._write_reference_state(env_ids)
        # ManagerBasedRLEnv resets the action manager before the command
        # manager.  Synchronize again after selecting the new clip/RSI frame
        # so the first PD target is exactly the newly written reference pose.
        self._env.action_manager.get_term("joint_pos").synchronize_reference(env_ids)
        self.previous_ball_above_hoop[env_ids] = (
            self.reference["object_pos"][env_ids, 2] - self.reference["hoop_pos"][env_ids, 2]
        )
        self.time_left[env_ids] = 1.0e9
        self.command_counter[env_ids] = 1
        return extras

    def set_clip_ids(self, clip_ids: int | torch.Tensor, env_ids: Sequence[int] | None = None) -> None:
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        elif not isinstance(env_ids, torch.Tensor):
            env_ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        ids = torch.as_tensor(clip_ids, dtype=torch.long, device=self.device).flatten()
        if ids.numel() == 1:
            ids = ids.expand(len(env_ids))
        if ids.numel() != len(env_ids) or torch.any(ids < 0) or torch.any(ids >= self.num_clips):
            raise ValueError(f"clip_ids must contain one valid id per environment (0..{self.num_clips - 1}).")
        self.fixed_clip_ids[env_ids] = ids

    def clear_fixed_clip_ids(self, env_ids: Sequence[int] | None = None) -> None:
        self.fixed_clip_ids[slice(None) if env_ids is None else env_ids] = -1

    def actual_link_pos_b(self) -> torch.Tensor:
        root_pos = self.robot.data.root_link_pos_w
        root_quat = self.robot.data.root_link_quat_w
        link_pos = self.robot.data.body_pos_w[:, self.body_ids]
        count = len(self.body_ids)
        return quat_apply_inverse(
            root_quat[:, None, :].expand(-1, count, -1).reshape(-1, 4),
            (link_pos - root_pos[:, None, :]).reshape(-1, 3),
        ).reshape(self.num_envs, count, 3)

    def actual_anchor_pos_w(self) -> torch.Tensor:
        body_pos = self.robot.data.body_pos_w[:, self.body_ids[:2]]
        body_quat = self.robot.data.body_quat_w[:, self.body_ids[:2]]
        offsets = self.anchor_offsets[None].expand(self.num_envs, -1, -1)
        return body_pos + quat_apply(body_quat.reshape(-1, 4), offsets.reshape(-1, 3)).reshape_as(body_pos)

    def hoop_center_pos_w(self) -> torch.Tensor:
        """Return the physical rim center rather than the floating asset root."""
        offset = self.hoop_center_offset.expand(self.num_envs, -1)
        return self.hoop.data.root_pos_w + quat_apply(self.hoop.data.root_quat_w, offset)

    def _tracking_error(self) -> torch.Tensor:
        root_local = self.robot.data.root_link_pos_w - self._env.scene.env_origins
        root_error = torch.linalg.vector_norm(root_local - self.reference["root_pos"], dim=-1) / 0.25
        dof = self.robot.data.joint_pos[:, self.joint_ids]
        dof_error = torch.sqrt(torch.mean((dof - self.reference["dof_pos"]) ** 2, dim=-1)) / 0.5
        link_error = torch.sqrt(
            torch.mean((self.actual_link_pos_b() - self.reference["link_pos_b"]) ** 2, dim=(-1, -2))
        ) / 0.2
        return (root_error + dof_error + link_error) / 3.0

    def update_metrics(self) -> None:
        """Accumulate tracking and shot metrics before automatic resets run."""
        dt = self._env.step_dt
        ball_position = self.ball.data.root_pos_w
        hoop_position = self.hoop_center_pos_w()
        above = ball_position[:, 2] - hoop_position[:, 2]
        self.shot_success |= self.shot_eligible & downward_hoop_crossing(
            self.previous_ball_above_hoop,
            ball_position,
            hoop_position,
            self.ball.data.root_lin_vel_w,
            self.cfg.success_radius,
        )
        self.previous_ball_above_hoop[:] = above
        self.metrics["shot_success"][:] = self.shot_success.float()
        self.metrics["mean_speed"] += self.speed * dt
        root_local = self.robot.data.root_link_pos_w - self._env.scene.env_origins
        joint_pos = self.robot.data.joint_pos[:, self.joint_ids]
        joint_vel = self.robot.data.joint_vel[:, self.joint_ids]
        actual_link_pos = self.actual_link_pos_b()
        root_quat = self.robot.data.root_link_quat_w
        body_quat = self.robot.data.body_quat_w[:, self.body_ids]
        root_quat_expanded = root_quat[:, None].expand(-1, len(self.body_ids), -1)
        actual_link_quat = quat_mul(
            quat_inv(root_quat_expanded.reshape(-1, 4)), body_quat.reshape(-1, 4)
        ).reshape_as(body_quat)
        link_rotation_error = quat_error_magnitude(
            actual_link_quat.reshape(-1, 4), self.reference["link_quat_b"].reshape(-1, 4)
        ).reshape(self.num_envs, len(self.body_ids))
        rotation_body_quat = self.robot.data.body_quat_w[:, self.rotation_body_ids]
        rotation_root_quat = root_quat[:, None].expand_as(rotation_body_quat)
        rotation_quat_b = quat_mul(
            quat_inv(rotation_root_quat.reshape(-1, 4)), rotation_body_quat.reshape(-1, 4)
        )
        rotation_error = quat_error_magnitude(
            rotation_quat_b, self.reference["rotation_quat_b"].reshape(-1, 4)
        ).reshape(self.num_envs, len(self.rotation_body_ids))
        actual_relative_position = self.actual_anchor_pos_w() - ball_position[:, None]
        actual_contact = contact_graph(self._env)
        action_term = self._env.action_manager.get_term("joint_pos")
        raw_action = action_term.raw_actions
        effort_limit = self.robot.data.joint_effort_limits[:, self.joint_ids].clamp_min(1.0e-6)

        instantaneous = {
            # Keep the original four names stable for the evaluator.
            "root_rmse": torch.linalg.vector_norm(root_local - self.reference["root_pos"], dim=-1),
            "dof_rmse": torch.sqrt(torch.mean((joint_pos - self.reference["dof_pos"]) ** 2, dim=-1)),
            "link_rmse": torch.sqrt(torch.mean((actual_link_pos - self.reference["link_pos_b"]) ** 2, dim=(-1, -2))),
            "ball_rmse": torch.linalg.vector_norm(
                ball_position - self._env.scene.env_origins - self.reference["object_pos"], dim=-1
            ),
            "error/root_rotation_rad": quat_error_magnitude(root_quat, self.reference["root_quat"]),
            "error/root_linear_velocity_mps": torch.linalg.vector_norm(
                self.robot.data.root_link_lin_vel_w - self.reference["root_lin_vel"], dim=-1
            ),
            "error/root_angular_velocity_radps": torch.linalg.vector_norm(
                self.robot.data.root_link_ang_vel_w - self.reference["root_ang_vel"], dim=-1
            ),
            "error/dof_velocity_rmse_radps": torch.sqrt(
                torch.mean((joint_vel - self.reference["dof_vel"]) ** 2, dim=-1)
            ),
            "error/link_rotation_rmse_rad": torch.sqrt(torch.mean(link_rotation_error**2, dim=-1)),
            **{
                f"error/rotation/{name}_rad": rotation_error[:, index]
                for index, name in enumerate(self.rotation_body_names)
            },
            "error/ball_rotation_rad": quat_error_magnitude(
                self.ball.data.root_quat_w, self.reference["object_quat"]
            ),
            "error/ball_linear_velocity_mps": torch.linalg.vector_norm(
                self.ball.data.root_lin_vel_w - self.reference["object_lin_vel"], dim=-1
            ),
            "error/ball_angular_velocity_radps": torch.linalg.vector_norm(
                self.ball.data.root_ang_vel_w - self.reference["object_ang_vel"], dim=-1
            ),
            "error/relative_position_rmse_m": torch.sqrt(
                torch.mean(
                    (actual_relative_position - self.reference["anchor_object_rel_pos_w"]) ** 2,
                    dim=(-1, -2),
                )
            ),
            "error/contact_mismatch_rate": torch.mean(
                torch.abs(actual_contact - self.reference["contact"]), dim=-1
            ),
            "control/raw_action_rms": torch.sqrt(torch.mean(raw_action**2, dim=-1)),
            "control/raw_action_delta_rms": torch.sqrt(
                torch.mean((raw_action - action_term.previous_raw_actions) ** 2, dim=-1)
            ),
            "control/action_saturation_rate": torch.mean((torch.abs(raw_action) >= 0.999).float(), dim=-1),
            "control/joint_target_delta_rms_rad": torch.sqrt(
                torch.mean((action_term.desired_actions - action_term.previous_desired_actions) ** 2, dim=-1)
            ),
            "control/reference_joint_delta_rms_rad": torch.sqrt(
                torch.mean((self.reference["dof_pos"] - self._previous_reference_dof_pos) ** 2, dim=-1)
            ),
            "control/pd_error_rmse_rad": torch.sqrt(
                torch.mean((action_term.desired_actions - joint_pos) ** 2, dim=-1)
            ),
            "control/joint_velocity_rms_radps": torch.sqrt(torch.mean(joint_vel**2, dim=-1)),
            "control/torque_limit_ratio_rms": torch.sqrt(
                torch.mean((self.robot.data.applied_torque[:, self.joint_ids] / effort_limit) ** 2, dim=-1)
            ),
        }
        for name, value in instantaneous.items():
            self.metrics[name] += value * dt
        self._previous_reference_dof_pos[:] = self.reference["dof_pos"]

    def _update_metrics(self) -> None:
        # CommandManager.compute runs after automatic resets. The unified
        # reward calls update_metrics() earlier in the step so terminal frames
        # are included and reset states are not charged to the old episode.
        pass

    def _resample_command(self, env_ids: Sequence[int]) -> None:
        # A motion is sampled only on environment reset, never on the generic
        # command resampling timer.
        self.time_left[env_ids] = 1.0e9

    def _update_command(self) -> None:
        dt = self._env.step_dt
        pre = self.stage == STAGE_PRE_HOLD
        post = self.stage == STAGE_POST_HOLD
        self.hold_time[pre | post] += dt
        start_ids = (pre & (self.hold_time + 1.0e-6 >= self.cfg.pre_hold_s)).nonzero(as_tuple=False).flatten()
        self.stage[start_ids] = STAGE_ACTIVE
        self.speed[start_ids] = 1.0
        self.tracking_error_ema[start_ids] = self.cfg.adaptive_initial_error
        self.speed_update_elapsed[start_ids] = 0.0
        self.hold_time[start_ids] = 0.0

        active = self.stage == STAGE_ACTIVE
        if torch.any(active):
            tracking_error = self._tracking_error()
            self.tracking_error_ema[active] += self._adaptive_error_alpha * (
                tracking_error[active] - self.tracking_error_ema[active]
            )
            self.speed_update_elapsed[active] += dt
            speed_update = active & (
                self.speed_update_elapsed + 1.0e-6 >= self.cfg.speed_update_interval_s
            )
            if torch.any(speed_update):
                if self.cfg.enable_adaptive_speed:
                    chosen = select_adaptive_speed(
                        self.tracking_error_ema, tuple(self.cfg.speed_choices)
                    )
                    self.speed[speed_update] = chosen[speed_update]
                else:
                    self.speed[speed_update] = 1.0
                self.speed_update_elapsed[speed_update] = 0.0

            # advance_reference_frame locks speed to 1x at release, preserving
            # the physical timing of the free basketball flight.
            next_frame, next_speed = advance_reference_frame(
                self.frame,
                self.speed,
                self.release_frame[self.clip_ids],
                self.fps,
                dt,
            )
            self.speed[active] = next_speed[active]
            self.frame[active] = next_frame[active]
            last = (self.lengths[self.clip_ids] - 1).float()
            reached = active & (self.frame >= last)
            self.frame[:] = torch.minimum(self.frame, last)
            self.stage[reached] = STAGE_POST_HOLD
            self.speed[reached] = 0.0
            self.hold_time[reached] = 0.0

        # Terminations are evaluated before CommandManager.compute. Mark done
        # one policy interval early so the triggering physics frame completes
        # exactly post_hold_s seconds of last-frame tracking.
        done_threshold = max(self.cfg.post_hold_s - dt, 0.0)
        finished = (self.stage == STAGE_POST_HOLD) & (self.hold_time + 1.0e-6 >= done_threshold)
        self.stage[finished] = STAGE_DONE
        self.speed[finished] = 0.0
        self._refresh_reference()

    def _set_debug_vis_impl(self, debug_vis: bool) -> None:
        if not hasattr(self, "_reference_markers"):
            self._reference_markers = None
        if self._reference_markers is None and debug_vis:
            marker_cfg = VisualizationMarkersCfg(
                prim_path="/Visuals/G1ShootReference",
                markers={
                    "tracked_body": sim_utils.SphereCfg(
                        radius=0.045,
                        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.1, 1.0, 0.15)),
                    ),
                    "basketball": sim_utils.SphereCfg(
                        radius=0.075,
                        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.1, 0.35, 1.0)),
                    ),
                    "hoop_center": sim_utils.SphereCfg(
                        radius=0.065,
                        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.1, 0.1)),
                    ),
                },
            )
            self._reference_markers = VisualizationMarkers(marker_cfg)
        if self._reference_markers is not None:
            self._reference_markers.set_visibility(debug_vis)

    def _debug_vis_callback(self, event) -> None:
        self.update_reference_markers()

    def update_reference_markers(self) -> None:
        """Immediately refresh body, ball, and hoop reference markers."""
        if self._reference_markers is None or not hasattr(self, "reference") or not self.reference:
            return
        origins = self._env.scene.env_origins
        link_points = self.reference["link_pos_w"] + origins[:, None, :]
        ball_points = self.reference["object_pos"] + origins
        hoop_points = self.reference["hoop_pos"] + origins
        translations = torch.cat(
            (link_points, ball_points[:, None, :], hoop_points[:, None, :]), dim=1
        ).flatten(0, 1)
        marker_indices = torch.tensor((0, 0, 0, 0, 1, 2), device=self.device).repeat(self.num_envs)
        self._reference_markers.visualize(translations=translations, marker_indices=marker_indices)


@configclass
class MotionReferenceCommandCfg(CommandTermCfg):
    class_type: type[CommandTerm] = MotionReferenceCommand
    source_path: str = MISSING
    cache_path: str = MISSING
    urdf_path: str = MISSING
    robot_name: str = "robot"
    ball_name: str = "ball"
    hoop_name: str = "hoop"
    hoop_center_offset: tuple[float, float, float] = (2.17577, 0.0, 3.02)
    hoop_root_quat: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0)
    tracked_body_names: list[str] = list(TRACKED_BODY_NAMES)
    auto_preprocess: bool = True
    pre_hold_s: float = 0.5
    post_hold_s: float = 1.0
    randomize_clip: bool = True
    default_clip_id: int = 0
    enable_rsi: bool = True
    rsi_initial_probability: float = 0.8
    rsi_decay_steps: int = 144_000
    enable_adaptive_speed: bool = True
    speed_choices: tuple[float, float, float, float] = (0.5, 0.75, 1.0, 1.25)
    adaptive_error_time_constant_s: float = 0.10
    adaptive_initial_error: float = 0.50
    speed_update_interval_s: float = 0.10
    success_radius: float = 0.20
    resampling_time_range: tuple[float, float] = (1.0e9, 1.0e9)
