"""Simulator-independent deployment contract, phase clock, and observation builder.

Hardware adapters supply calibrated pelvis-frame IMU samples and ordered joint
states. This module does not enable motors or implement hardware fault handling.
"""
from __future__ import annotations

import math

import torch

from .student import student_observation_fields


def build_deployment_contract(metadata: dict, batch: dict, motion_cfg) -> dict:
    names = metadata['joint_names']
    if list(batch['metadata']['dof_names']) != names:
        raise ValueError('Motion cache joint order differs from exported policy.')
    fps = float(batch['metadata']['fps'])
    dt = 1. / float(metadata['policy_hz'])
    if fps <= 0 or dt <= 0:
        raise ValueError('Motion and policy frequencies must be positive.')
    nominal = torch.tensor(metadata['nominal_joint_position'])
    lower = torch.tensor(metadata['safe_lower_joint_position'])
    upper = torch.tensor(metadata['safe_upper_joint_position'])
    profiles = []
    for index, frames in enumerate(batch['lengths'].tolist()):
        if frames < 2:
            raise ValueError('Deployment motion must contain at least two frames.')
        first = batch['dof_pos'][index, 0].detach().cpu()
        target = first.clamp(lower, upper)
        profiles.append(dict(
            clip_id=index, frame_count=int(frames), reference_fps=fps,
            duration_s=(frames - 1) / fps, phase_increment_per_tick=fps * dt / (frames - 1),
            initial_joint_position_rad=first.tolist(), initial_pd_target_rad=target.tolist(),
            initial_applied_action_offset_rad=(target - nominal).tolist(),
            initial_root_position_m=batch['root_pos'][index, 0].detach().cpu().tolist(),
            initial_root_quaternion_wxyz=batch['root_quat'][index, 0].detach().cpu().tolist(),
            initial_ball_position_m=batch['object_pos'][index, 0].detach().cpu().tolist(),
            initial_ball_controlled=bool(batch['release'][index, 0] < .5),
        ))
    fields = []
    offset = 0
    units = ['dimensionless', 'unit vector', 'rad/s', 'rad', 'rad/s', 'rad', 'm']
    for (name, size), unit in zip(student_observation_fields(metadata['history_length']), units):
        fields.append(dict(name=name, start=offset, stop=offset + size, unit_before_scaling=unit))
        offset += size
    return dict(
        schema='bb_mimic_nt_deployment_contract_v1', policy_period_s=dt,
        observation=dict(
            fields=fields, dtype='float32', shape=['batch', metadata['observation_dim']],
            normalization='explicit scales only; no running normalization or input clipping',
            layout='field-major; each history field is flattened oldest frame first, then joint/axis order',
            projected_gravity='R_world_from_pelvis.T @ [0, 0, -1]; unit vector, not m/s^2',
            angular_velocity='pelvis/root_link frame, rad/s; calibrate IMU mounting rotation',
            joint_position='measured q minus nominal_joint_position, in joint_names order',
            applied_action='last PD target actually applied after clipping/actuator delay minus nominal; not measured q or raw network output',
            initial_hoop='R_world_from_pelvis_at_reset.T @ (hoop_center_world - pelvis_position_world_at_reset); freeze until episode reset',
            timing='sample before inference; phase and applied-action history are undelayed; sensor history ends at tick minus observation delay',
        ),
        phase=dict(
            initial=0., ready='hold phase=0 while continuing sensor/action histories',
            active='infer at current phase, apply target, then frame=min(frame+reference_fps*policy_period_s, frame_count-1); phase=frame/(frame_count-1)',
            speed=1., adaptive_speed=False, trigger='READY -> ACTIVE without clearing observation history or changing the frozen hoop vector',
            final='phase=1 during POST_HOLD; finish after post_hold_s; stopping policy does not imply disabling motor torque',
            trained_pre_hold_s=float(motion_cfg.pre_hold_s), trained_post_hold_s=float(motion_cfg.post_hold_s),
            training_hold_sampling_weights=getattr(motion_cfg, 'hold_sampling_weights', None),
            training_hold_sampling='whole hold segments sampled per reset; weights are expected hold time relative to nominal remaining active time, not phase increments; deployment uses full configured hold durations',
            ready_policy_validated=False,
            ready_validation='pre_hold_s=0 provides only a reset boundary, not training for indefinite phase-zero holding; validate or retrain both Teacher and Student with an extended PRE_HOLD',
            motion_profiles=profiles,
        ),
        initialization=dict(
            sequence=['select clip profile', 'bring robot and ball to the initial condition through the hardware startup controller',
                      'read real calibrated sensors and the actual applied PD target',
                      'capture initial hoop vector in the reset pelvis frame',
                      'repeat this first sensor sample into every sensor-history slot and the applied target into every action-history slot',
                      'start policy at phase=0; continue updating histories during READY; trigger the shot without another reset'],
            sensor_history_padding='repeat measured initial sample, including measured joint velocities; do not fill with zeros',
            action_history_padding='repeat actual initial PD target minus nominal; training initializes this from the clipped first reference pose',
            nominal_pose='coordinate origin for q offsets, not the shooting startup pose',
        ),
        actuation=dict(
            output='29 nominal-relative joint-position offsets in radians',
            target='clip(nominal_joint_position + network_output, safe_lower_joint_position, safe_upper_joint_position)',
            tanh=False, reference_residual_scale=False, smoothing=False,
            pd='tau = Kp * (target - measured_q) - Kd * measured_qdot; hardware actuator conventions must match',
            delay='default_action_delay_steps is the trained/play latency assumption; do not add a second delay if hardware already supplies it',
        ),
        provenance=dict(motion_source_sha256=batch['metadata'].get('source_sha256'),
                        robot_urdf_sha256=batch['metadata'].get('urdf_sha256')),
        hardware_integration_required=['joint index/sign/zero calibration', 'IMU-to-pelvis rotation and gravity convention',
                                       'hoop-center measurement', 'actual command acknowledgement/latency',
                                       'motor limits and gain conventions', 'startup, watchdog, fallback, and emergency-stop implementation'],
    )


class StudentPhaseClock:
    """Explicit shot clock. READY is opt-in; it has no learned-stability guarantee."""
    def __init__(self, profile: dict, policy_hz: float, post_hold_s: float):
        if (profile['frame_count'] < 2 or not math.isfinite(policy_hz) or policy_hz <= 0
            or not math.isfinite(profile['reference_fps']) or profile['reference_fps'] <= 0
            or not math.isfinite(post_hold_s) or post_hold_s < 0):
            raise ValueError('Invalid phase-clock motion, frequency, or hold duration.')
        self.last_frame = int(profile['frame_count']) - 1
        self.advance = float(profile['reference_fps']) / policy_hz
        self.post_ticks = max(1, math.ceil(post_hold_s * policy_hz - 1.e-6))
        self.reset()

    def reset(self):
        self.frame = 0.
        self.stage = 'READY'
        self.hold_ticks = 0

    @property
    def phase(self):
        return self.frame / self.last_frame

    def trigger(self):
        if self.stage != 'READY':
            raise ValueError('Shot trigger is valid only in READY.')
        self.stage = 'ACTIVE'

    def after_policy_tick(self):
        if self.stage == 'ACTIVE':
            self.frame = min(self.frame + self.advance, self.last_frame)
            if self.frame >= self.last_frame:
                self.stage = 'POST_HOLD'
        elif self.stage == 'POST_HOLD':
            self.hold_ticks += 1
            if self.hold_ticks >= self.post_ticks:
                self.stage = 'DONE'


class StudentObservationState:
    """One robot's raw histories; inputs are calibrated, delay-free sensor samples.

When hardware samples already have latency, avoid applying the same latency a
second time. Match the trained total latency through the hardware adapter.
"""
    def __init__(self, metadata: dict, sensor_delay_steps: int | None = None):
        self.metadata = metadata
        self.history = metadata['history_length']
        self.delay = metadata['default_observation_delay_steps'] if sensor_delay_steps is None else sensor_delay_steps
        if not isinstance(self.delay, int) or self.delay < 0:
            raise ValueError('Sensor delay must be a non-negative integer.')
        self.nominal = torch.tensor(metadata['nominal_joint_position'], dtype=torch.float32)
        self.lower = torch.tensor(metadata['safe_lower_joint_position'], dtype=torch.float32)
        self.upper = torch.tensor(metadata['safe_upper_joint_position'], dtype=torch.float32)
        self._sensor = None

    @staticmethod
    def _vector(value, size):
        result = torch.as_tensor(value, dtype=torch.float32, device='cpu').reshape(-1)
        if result.numel() != size or not torch.isfinite(result).all():
            raise ValueError(f'Expected {size} finite sensor/command values.')
        return result

    def _sample(self, gravity, gyro, q, qdot):
        return torch.cat((self._vector(gravity, 3), self._vector(gyro, 3),
                          self._vector(q, 29) - self.nominal, self._vector(qdot, 29)))

    def reset(self, gravity, gyro, q, qdot, applied_pd_target, initial_hoop_b):
        sensor = self._sample(gravity, gyro, q, qdot)
        action = self._vector(applied_pd_target, 29) - self.nominal
        self._sensor = sensor.repeat(self.history + self.delay, 1)
        self._action = action.repeat(self.history, 1)
        self._hoop = self._vector(initial_hoop_b, 3).clone()
        self._last_tick = -1
        self._last_phase = None

    def observe(self, tick: int, phase: float, gravity, gyro, q, qdot, applied_pd_target):
        if self._sensor is None:
            raise ValueError('Reset deployment histories before inference.')
        if not math.isfinite(phase) or not 0 <= phase <= 1:
            raise ValueError('Phase must be finite and in [0, 1].')
        if tick == self._last_tick:
            if phase != self._last_phase:
                raise ValueError('Phase changed within the same policy tick.')
            return self._observation.clone()
        if tick != self._last_tick + 1:
            raise ValueError('Policy tick skipped or reversed; hardware watchdog must handle missed deadlines.')
        sensor = self._sample(gravity, gyro, q, qdot)
        action = self._vector(applied_pd_target, 29) - self.nominal
        self._sensor = torch.cat((self._sensor[1:], sensor[None]))
        self._action = torch.cat((self._action[1:], action[None]))
        frames = self._sensor[:self.history]
        scales = self.metadata['observation_settings']['scales']
        self._observation = torch.cat((torch.tensor([phase * scales['phase']]),
            frames[:, :3].flatten() * scales['gravity'], frames[:, 3:6].flatten() * scales['angular_velocity'],
            frames[:, 6:35].flatten() * scales['joint_position'], frames[:, 35:64].flatten() * scales['joint_velocity'],
            self._action.flatten() * scales['applied_action'], self._hoop * scales['hoop_position']))[None]
        self._last_tick, self._last_phase = tick, phase
        return self._observation.clone()

    def action_target(self, network_output):
        offset = self._vector(network_output, 29)
        return (self.nominal + offset).clamp(self.lower, self.upper)
