"""Simulator-independent deployment contract and observation builder.

Hardware adapters supply calibrated pelvis-frame IMU samples and ordered joint
states. This module does not enable motors or implement hardware fault handling.
"""
from __future__ import annotations

import copy

import torch

from .student import student_observation_fields


def build_deployment_contract(batch: dict) -> dict:
    fps = float(batch['metadata']['fps'])
    config = batch['metadata']['skill_config']
    profiles = []
    clip_id = 0
    for skill, count in zip(config['skills'], batch['metadata']['skill_clip_counts']):
        for local_clip_id in range(count):
            frames = int(batch['lengths'][clip_id])
            profiles.append(dict(
                clip_id=clip_id, skill_clip_id=local_clip_id, command_id=skill['command_id'],
                frame_count=frames, reference_fps=fps, reference_duration_s=(frames - 1) / fps,
                initial_joint_position_rad=batch['dof_pos'][clip_id, 0].detach().cpu().tolist(),
                initial_root_position_m=batch['root_pos'][clip_id, 0].detach().cpu().tolist(),
                initial_root_quaternion_wxyz=batch['root_quat'][clip_id, 0].detach().cpu().tolist(),
                initial_ball_position_m=batch['object_pos'][clip_id, 0].detach().cpu().tolist(),
                initial_ball_controlled=bool(batch['release'][clip_id, 0] < .5),
                initial_hoop_pos_cyl=batch['hoop_pos_cyl'][clip_id, 0].detach().cpu().tolist(),
            ))
            clip_id += 1
    # Checkpoints retain motion facts; the exporter builds the public interface.
    return dict(
        skills=dict(config=copy.deepcopy(config), motion_profiles=profiles),
        provenance=dict(motion_source_sha256=[source['source_sha256'] for source in batch['metadata']['skill_sources']],
                        robot_urdf_sha256=batch['metadata']['urdf_sha256']),
    )


_SCALE_NAMES = ('skill_command', 'gravity', 'angular_velocity', 'joint_position',
                'joint_velocity', 'applied_action', 'hoop_position')
_ACTION_TARGET = 'clip(nominal_joint_position + network_output, safe_lower_joint_position, safe_upper_joint_position)'
_UNITS = ('dimensionless', 'dimensionless', 'rad/s', 'rad', 'rad/s', 'rad', 'm, rad, m')
_SOURCES = (
    'one_hot(command_id, command_dimension); fixed for the episode',
    'R_world_from_pelvis.T @ [0, 0, -1]; xyz direction, not acceleration',
    'pelvis-frame angular velocity; xyz',
    'measured joint position - nominal_joint_position; joint_names order',
    'measured joint velocity; joint_names order',
    'PD target actually applied after position clipping and action delay - nominal_joint_position; joint_names order',
    'trajectory hoop_pos_cyl; [r, theta, z], frozen until reset',
)
_SKILL_FIELDS = ('name', 'command_id', 'motion_file', 'initialization', 'object', 'end')
_PROFILE_FIELDS = (
    'clip_id', 'skill_clip_id', 'command_id', 'frame_count', 'reference_fps', 'reference_duration_s',
    'initial_joint_position_rad', 'initial_root_position_m', 'initial_root_quaternion_wxyz',
    'initial_ball_position_m', 'initial_ball_controlled', 'initial_hoop_pos_cyl',
)


def student_observation_scales(metadata: dict) -> dict:
    if 'observation' in metadata:
        return {key: field['scale'] for key, field in zip(_SCALE_NAMES, metadata['observation']['fields'])}
    scales = dict(metadata['observation_settings']['scales'])
    scales['skill_command'] = 1.
    position_scale = scales['hoop_position']
    scales['hoop_position'] = [position_scale, scales['hoop_theta'], position_scale]
    return scales


def student_sensor_delay(metadata: dict) -> int:
    if 'observation' in metadata:
        return metadata['observation']['sensor_delay_steps']
    return metadata['default_observation_delay_steps']


def deployment_skills(contract: dict) -> dict:
    """Describe command selection and reference initialization for each trained skill."""
    config = contract['config']
    profiles = [{key: profile[key] for key in _PROFILE_FIELDS} for profile in contract['motion_profiles']]
    definitions = []
    for skill in config['skills']:
        definition = {key: skill[key] for key in _SKILL_FIELDS}
        definition['motion_profile_ids'] = [
            profile['clip_id'] for profile in profiles if profile['command_id'] == skill['command_id']
        ]
        definitions.append(definition)
    return dict(
        command=dict(
            dimension=config['command']['dimension'], encoding='one_hot',
            selection='explicit_command_id', lifetime='episode', switching='reset_required',
        ),
        definitions=definitions, motion_profiles=profiles,
        execution=dict(
            reference_speed=1.0, end_trigger_sampling='bernoulli_per_episode',
            end_duration_sampling='uniform', end_hold='last_reference_frame',
            completion='reference_then_optional_end_hold', timing='external_controller',
            object_interaction='training reward and termination mask; does not actuate the ball',
        ),
    )


def student_simulation_skill_config(metadata: dict) -> dict:
    """Reconstruct simulator settings for explicit clip playback, without training sampling."""
    skills = metadata['skills']
    return dict(
        command=dict(dimension=skills['command']['dimension']),
        skills=[dict(**{key: copy.deepcopy(skill[key]) for key in _SKILL_FIELDS},
                     sampling_weight=1.0, rsi=dict(enabled=False))
                for skill in skills['definitions']],
    )


def compact_student_config(metadata: dict) -> dict:
    """Project checkpoint metadata onto the deployment-only skill-command interface."""
    keys = ('observation_dim', 'action_dim', 'history_length', 'command_dimension', 'joint_names',
            'nominal_joint_position', 'safe_lower_joint_position', 'safe_upper_joint_position',
            'pd_stiffness', 'pd_damping', 'policy_hz', 'student_hidden_dims', 'activation')
    if 'observation' in metadata:
        result = {key: metadata[key] for key in (
            *keys, 'model', 'observation', 'action', 'skills', 'initialization')}
        if 'export_source' in metadata:
            result['export_source'] = metadata['export_source']
        return copy.deepcopy(result)
    contract = metadata['deployment_contract']
    skills = deployment_skills(contract['skills'])
    scales = student_observation_scales(metadata)
    fields, offset = [], 0
    for (name, size), key, unit, source in zip(
        student_observation_fields(metadata['history_length'], metadata['command_dimension']), _SCALE_NAMES, _UNITS, _SOURCES
    ):
        fields.append(dict(name=name, start=offset, stop=offset + size,
                           unit=unit, scale=scales[key], source=source,
                           frame_size=size // metadata['history_length'] if name.endswith('_history') else size,
                           history_length=metadata['history_length'] if name.endswith('_history') else 1))
        offset += size
    result = {key: metadata[key] for key in keys}
    result.update(
        model=dict(
            file='student_policy.onnx',
            input=dict(name='student_observation', dtype='float32', shape=[None, metadata['observation_dim']]),
            output=dict(name='joint_position_offset', dtype='float32', shape=[None, metadata['action_dim']]),
        ),
        observation=dict(
            dtype='float32', layout='field_major', history_order='oldest_to_newest',
            preprocessing='multiply each field by its scale; no other normalization or input clipping',
            sensor_delay_steps=metadata['default_observation_delay_steps'], fields=fields,
            timing='sample before inference; only sensor histories have sensor_delay_steps; skill command and applied-target history use the current tick',
        ),
        action=dict(
            unit='rad', output='joint_position_offset_from_nominal',
            target=_ACTION_TARGET,
            delay_steps=metadata['default_action_delay_steps'],
            timing='clip before action delay; pad delay queue with the actual initial PD target; match total latency without duplicating hardware delay',
            pd='tau = pd_stiffness * (target - measured_q) - pd_damping * measured_qdot',
        ),
        skills=skills,
        initialization=dict(
            startup='use the hardware startup controller to reach the selected profile pose and ball condition; nominal is the offset origin, not the startup pose',
            sensor_history='repeat the first measured gravity, angular velocity, q-minus-nominal and qdot into all sensor slots, including delay padding',
            action_history='repeat the actual initial applied PD target minus nominal into all action slots; the training target is the clipped initial reference pose',
            hoop='supply selected profile initial_hoop_pos_cyl in [r, theta, z] order (m, rad, m) once at reset',
        ),
    )
    if 'export_source' in metadata:
        result['export_source'] = copy.deepcopy(metadata['export_source'])
    return copy.deepcopy(result)

class StudentObservationState:
    """One robot's raw histories; inputs are calibrated, delay-free sensor samples.

When hardware samples already have latency, avoid applying the same latency a
second time. Match the trained total latency through the hardware adapter.
"""
    def __init__(self, metadata: dict, sensor_delay_steps: int | None = None):
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
        self._last_command = None

    def observe(self, tick: int, command_id: int, gravity, gyro, q, qdot, applied_pd_target):
        if self._sensor is None:
            raise ValueError('Reset deployment histories before inference.')
        if type(command_id) is not int or not 0 <= command_id < self.metadata['command_dimension']:
            raise ValueError('Invalid skill command ID.')
        if self._last_command is not None and command_id != self._last_command:
            raise ValueError('Reset histories before changing the episode skill command.')
        if tick == self._last_tick:
            return self._observation.clone()
        if tick != self._last_tick + 1:
            raise ValueError('Policy tick skipped or reversed; hardware watchdog must handle missed deadlines.')
        sensor = self._sample(gravity, gyro, q, qdot)
        action = self._vector(applied_pd_target, 29) - self.nominal
        self._sensor = torch.cat((self._sensor[1:], sensor[None]))
        self._action = torch.cat((self._action[1:], action[None]))
        frames = self._sensor[:self.history]
        scales = student_observation_scales(self.metadata)
        self._observation = torch.cat((torch.nn.functional.one_hot(torch.tensor(command_id), self.metadata['command_dimension']).float(),
            frames[:, :3].flatten() * scales['gravity'], frames[:, 3:6].flatten() * scales['angular_velocity'],
            frames[:, 6:35].flatten() * scales['joint_position'], frames[:, 35:64].flatten() * scales['joint_velocity'],
            self._action.flatten() * scales['applied_action'], self._hoop * torch.as_tensor(scales['hoop_position'], dtype=self._hoop.dtype)))[None]
        self._last_tick, self._last_command = tick, command_id
        return self._observation.clone()

    def action_target(self, network_output):
        offset = self._vector(network_output, 29)
        return (self.nominal + offset).clamp(self.lower, self.upper)
