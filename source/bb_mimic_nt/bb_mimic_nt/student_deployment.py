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
    profiles = []
    for index, frames in enumerate(batch['lengths'].tolist()):
        if frames < 2:
            raise ValueError('Deployment motion must contain at least two frames.')
        profiles.append(dict(
            clip_id=index, frame_count=int(frames), reference_fps=fps,
            initial_joint_position_rad=batch['dof_pos'][index, 0].detach().cpu().tolist(),
            initial_root_position_m=batch['root_pos'][index, 0].detach().cpu().tolist(),
            initial_root_quaternion_wxyz=batch['root_quat'][index, 0].detach().cpu().tolist(),
            initial_ball_position_m=batch['object_pos'][index, 0].detach().cpu().tolist(),
            initial_ball_controlled=bool(batch['release'][index, 0] < .5),
        ))
    # Checkpoints retain motion facts; the exporter builds the public interface.
    return dict(
        phase=dict(trained_pre_hold_s=float(motion_cfg.pre_hold_s),
                   trained_post_hold_s=float(motion_cfg.post_hold_s), motion_profiles=profiles),
        provenance=dict(motion_source_sha256=batch['metadata'].get('source_sha256'),
                        robot_urdf_sha256=batch['metadata'].get('urdf_sha256')),
    )


STUDENT_DEPLOYMENT_FORMAT = 'bb_mimic_nt_student_v2'
_SCALE_NAMES = ('phase', 'gravity', 'angular_velocity', 'joint_position',
                'joint_velocity', 'applied_action', 'hoop_position')
_ACTION_TARGET = 'clip(nominal_joint_position + network_output, safe_lower_joint_position, safe_upper_joint_position)'
_UNITS = ('dimensionless', 'dimensionless', 'rad/s', 'rad', 'rad/s', 'rad', 'm, rad, m')
_SOURCES = (
    'frame / (frame_count - 1)',
    'R_world_from_pelvis.T @ [0, 0, -1]; xyz direction, not acceleration',
    'pelvis-frame angular velocity; xyz',
    'measured joint position - nominal_joint_position; joint_names order',
    'measured joint velocity; joint_names order',
    'PD target actually applied after position clipping and action delay - nominal_joint_position; joint_names order',
    'trajectory hoop_pos_cyl; [r, theta, z], frozen until reset',
)


def student_observation_scales(metadata: dict) -> dict:
    if metadata.get('format') == STUDENT_DEPLOYMENT_FORMAT:
        return {key: field['scale'] for key, field in zip(_SCALE_NAMES, metadata['observation']['fields'])}
    scales = dict(metadata['observation_settings']['scales'])
    position_scale = scales['hoop_position']
    scales['hoop_position'] = [position_scale, scales.get('hoop_theta', position_scale), position_scale]
    return scales


def student_sensor_delay(metadata: dict) -> int:
    if metadata.get('format') == STUDENT_DEPLOYMENT_FORMAT:
        return metadata['observation']['sensor_delay_steps']
    return metadata['default_observation_delay_steps']


def compact_student_config(metadata: dict) -> dict:
    """Project checkpoint/legacy metadata onto the deployment-only v2 interface."""
    keys = ('observation_dim', 'action_dim', 'history_length', 'joint_names',
            'nominal_joint_position', 'safe_lower_joint_position', 'safe_upper_joint_position',
            'pd_stiffness', 'pd_damping', 'policy_hz', 'student_hidden_dims', 'activation')
    if metadata.get('format') == STUDENT_DEPLOYMENT_FORMAT:
        # Preserve the declared interface, rather than silently repairing corruption.
        result = {key: metadata[key] for key in (
            'format', *keys, 'observation', 'action', 'phase', 'initialization')}
        if 'export_source' in metadata:
            result['export_source'] = metadata['export_source']
        validate_student_config(result)
        return result
    contract = metadata['deployment_contract']
    phase = contract['phase']
    scales = student_observation_scales(metadata)
    fields, offset = [], 0
    for (name, size), key, unit, source in zip(
        student_observation_fields(metadata['history_length']), _SCALE_NAMES, _UNITS, _SOURCES
    ):
        fields.append(dict(name=name, start=offset, stop=offset + size,
                           unit=unit, scale=scales[key], source=source))
        offset += size
    result = dict(format=STUDENT_DEPLOYMENT_FORMAT, **{key: metadata[key] for key in keys})
    result.update(
        observation=dict(
            dtype='float32', layout='field_major', history_order='oldest_to_newest',
            preprocessing='multiply each field by its scale; no other normalization or input clipping',
            sensor_delay_steps=metadata['default_observation_delay_steps'], fields=fields,
            timing='sample before inference; only sensor histories have sensor_delay_steps; phase and applied-target history use the current tick',
        ),
        action=dict(
            unit='rad', output='joint_position_offset_from_nominal',
            target=_ACTION_TARGET,
            delay_steps=metadata['default_action_delay_steps'],
            timing='clip before action delay; pad delay queue with the actual initial PD target; match total latency without duplicating hardware delay',
            pd='tau = pd_stiffness * (target - measured_q) - pd_damping * measured_qdot',
        ),
        phase=dict(
            pre_hold_s=phase['trained_pre_hold_s'], post_hold_s=phase['trained_post_hold_s'],
            ready='phase=0; update histories continuously; trigger ACTIVE without resetting histories or frozen hoop position',
            active='infer, submit target, then frame=min(frame + reference_fps / policy_hz, frame_count - 1); phase=frame / (frame_count - 1)',
            final='phase=1 for post_hold_s, then hand control to the hardware controller',
            motion_profiles=[{key: profile[key] for key in (
                'clip_id', 'frame_count', 'reference_fps', 'initial_joint_position_rad',
                'initial_root_position_m', 'initial_root_quaternion_wxyz',
                'initial_ball_position_m', 'initial_ball_controlled',
            )} for profile in phase['motion_profiles']],
        ),
        initialization=dict(
            startup='use the hardware startup controller to reach the selected profile pose and ball condition; nominal is the offset origin, not the startup pose',
            sensor_history='repeat the first measured gravity, angular velocity, q-minus-nominal and qdot into all sensor slots, including delay padding',
            action_history='repeat the actual initial applied PD target minus nominal into all action slots; the training target is the clipped initial reference pose',
            hoop='supply trajectory hoop_pos_cyl in [r, theta, z] order (m, rad, m) once at reset',
        ),
    )
    validate_student_config(result)
    return result


def validate_student_config(metadata: dict) -> None:
    """Validate dimensions, temporal layout, units and numerical deployment inputs."""
    if metadata.get('format') != STUDENT_DEPLOYMENT_FORMAT:
        raise ValueError('Unsupported Student deployment configuration format.')
    history = metadata['history_length']
    expected = student_observation_fields(history)
    if metadata['observation_dim'] != sum(size for _, size in expected) or metadata['action_dim'] != 29:
        raise ValueError('Student deployment dimensions disagree with history length.')
    names = metadata['joint_names']
    if len(names) != 29 or any(not isinstance(name, str) or not name for name in names) or len(set(names)) != 29:
        raise ValueError('Deployment requires 29 distinct joint names.')
    if not math.isfinite(metadata['policy_hz']) or metadata['policy_hz'] <= 0:
        raise ValueError('Policy frequency must be positive and finite.')
    if (not metadata['student_hidden_dims'] or any(type(size) is not int or size <= 0 for size in metadata['student_hidden_dims'])
        or not isinstance(metadata['activation'], str)):
        raise ValueError('Invalid Student network architecture.')
    vectors = {}
    for key in ('nominal_joint_position', 'safe_lower_joint_position', 'safe_upper_joint_position', 'pd_stiffness', 'pd_damping'):
        value = torch.as_tensor(metadata[key])
        if value.shape != (29,) or not torch.isfinite(value).all():
            raise ValueError(f'Deployment {key} must contain 29 finite values.')
        vectors[key] = value
    lower, upper, nominal = (vectors[key] for key in ('safe_lower_joint_position', 'safe_upper_joint_position', 'nominal_joint_position'))
    if torch.any(lower >= upper) or torch.any(nominal < lower) or torch.any(nominal > upper):
        raise ValueError('Deployment joint limits/nominal position are invalid.')
    if torch.any(vectors['pd_stiffness'] <= 0) or torch.any(vectors['pd_damping'] < 0):
        raise ValueError('Deployment PD gains must have positive Kp and non-negative Kd.')
    observation = metadata['observation']
    if (observation['dtype'] != 'float32' or observation['layout'] != 'field_major'
        or observation['history_order'] != 'oldest_to_newest'):
        raise ValueError('Unsupported Student observation dtype/layout/history order.')
    fields = observation['fields']
    if len(fields) != len(expected):
        raise ValueError('Student observation fields disagree with its history length.')
    offset = 0
    for field, (name, size), unit in zip(fields, expected, _UNITS):
        if (field['name'], field['start'], field['stop'], field['unit']) != (name, offset, offset + size, unit):
            raise ValueError('Student observation fields/units disagree with its history length.')
        scale = torch.as_tensor(field['scale'])
        allowed_shapes = ((), (3,)) if name == 'hoop_pos_cyl' else ((),)
        if scale.shape not in allowed_shapes or not torch.isfinite(scale).all() or torch.any(scale <= 0):
            raise ValueError('Observation scales must be positive and finite, with three components only for hoop_pos_cyl.')
        offset += size
    for delay in (observation['sensor_delay_steps'], metadata['action']['delay_steps']):
        if type(delay) is not int or delay < 0:
            raise ValueError('Deployment delays must be non-negative integer policy steps.')
    if (metadata['action']['unit'] != 'rad' or metadata['action']['output'] != 'joint_position_offset_from_nominal'
        or metadata['action']['target'] != _ACTION_TARGET):
        raise ValueError('Unsupported Student action semantics.')
    phase = metadata['phase']
    for key in ('pre_hold_s', 'post_hold_s'):
        if not math.isfinite(phase[key]) or phase[key] < 0:
            raise ValueError('Phase hold durations must be finite and non-negative.')
    profiles = phase['motion_profiles']
    if not profiles:
        raise ValueError('Deployment needs at least one motion profile.')
    for index, profile in enumerate(profiles):
        if (profile['clip_id'] != index or type(profile['frame_count']) is not int or profile['frame_count'] < 2
            or not math.isfinite(profile['reference_fps']) or profile['reference_fps'] <= 0):
            raise ValueError('Invalid deployment motion profile timing/order.')
        for key, size in (('initial_joint_position_rad', 29), ('initial_root_position_m', 3),
                          ('initial_root_quaternion_wxyz', 4), ('initial_ball_position_m', 3)):
            value = torch.as_tensor(profile[key])
            if value.shape != (size,) or not torch.isfinite(value).all():
                raise ValueError(f'Invalid motion profile {key}.')
        quat = torch.tensor(profile['initial_root_quaternion_wxyz'])
        if abs(float(quat.norm()) - 1.) > 1.e-3 or type(profile['initial_ball_controlled']) is not bool:
            raise ValueError('Invalid motion profile orientation/ball condition.')


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
        if metadata.get('format') == STUDENT_DEPLOYMENT_FORMAT:
            validate_student_config(metadata)
        self.metadata = metadata
        self.history = metadata['history_length']
        self.delay = student_sensor_delay(metadata) if sensor_delay_steps is None else sensor_delay_steps
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

    def reset(self, gravity, gyro, q, qdot, applied_pd_target, hoop_pos_cyl):
        sensor = self._sample(gravity, gyro, q, qdot)
        action = self._vector(applied_pd_target, 29) - self.nominal
        self._sensor = sensor.repeat(self.history + self.delay, 1)
        self._action = action.repeat(self.history, 1)
        self._hoop = self._vector(hoop_pos_cyl, 3).clone()
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
        scales = student_observation_scales(self.metadata)
        self._observation = torch.cat((torch.tensor([phase * scales['phase']]),
            frames[:, :3].flatten() * scales['gravity'], frames[:, 3:6].flatten() * scales['angular_velocity'],
            frames[:, 6:35].flatten() * scales['joint_position'], frames[:, 35:64].flatten() * scales['joint_velocity'],
            self._action.flatten() * scales['applied_action'], self._hoop * torch.as_tensor(scales['hoop_position'], dtype=self._hoop.dtype)))[None]
        self._last_tick, self._last_phase = tick, phase
        return self._observation.clone()

    def action_target(self, network_output):
        offset = self._vector(network_output, 29)
        return (self.nominal + offset).clamp(self.lower, self.upper)
