"""Simulator-independent deployment contract and observation builder.

Hardware adapters supply calibrated pelvis-frame IMU samples and ordered joint
states. This module does not enable motors or implement hardware fault handling.
"""
from __future__ import annotations

import copy
import math

import torch

from .student import student_observation_fields
from .action_math import clip_joint_actions, joint_position_target
from .policy_contract import validate_policy_contract


STUDENT_TRAINING_FORMAT = 'bb_mimic_nt_student_training_v4'
STUDENT_DEPLOYMENT_FORMAT = 'bb_mimic_nt_student_deployment_v4'
STUDENT_ACTION_SEMANTICS = 'direct_joint_position_v1'
_ROBOT_FIELDS = ('joint_names', 'nominal_joint_position', 'action_scale', 'action_clip',
                 'pd_stiffness', 'pd_damping', 'effort_limits', 'velocity_limits',
                 'joint_position_limits', 'policy_hz')


def validate_student_metadata(metadata: dict) -> None:
    """Reject incompatible policies before loading weights or changing an environment."""
    version = metadata.get('format')
    if version not in (STUDENT_TRAINING_FORMAT, STUDENT_DEPLOYMENT_FORMAT):
        raise ValueError('Legacy Student action/observation contract is incompatible; train and export a new direct-action Student.')
    if metadata.get('action_semantics') != STUDENT_ACTION_SEMANTICS:
        raise ValueError('Student action semantics must be direct_joint_position_v1.')
    names = metadata.get('joint_names', [])
    if len(names) != 29 or len(set(names)) != 29 or any(not isinstance(name, str) or not name for name in names):
        raise ValueError('Student joint_names must contain 29 unique names.')
    for key in ('nominal_joint_position', 'action_scale', 'pd_stiffness', 'pd_damping',
                'effort_limits', 'velocity_limits', 'joint_position_limits'):
        values = torch.as_tensor(metadata.get(key, []), dtype=torch.float64)
        shape = (29, 2) if key == 'joint_position_limits' else (29,)
        if values.shape != shape or not torch.isfinite(values).all():
            raise ValueError(f'Student {key} must contain finite values with shape {shape}.')
        if key in ('action_scale', 'pd_stiffness', 'effort_limits', 'velocity_limits') and torch.any(values <= 0):
            raise ValueError(f'Student {key} must be positive.')
        if key == 'pd_damping' and torch.any(values < 0):
            raise ValueError('Student pd_damping must be non-negative.')
        if key == 'joint_position_limits' and torch.any(values[:, 0] >= values[:, 1]):
            raise ValueError('Student joint position limits must be ordered.')
    for key in ('action_clip', 'policy_hz'):
        value = metadata.get(key)
        if not isinstance(value, (float, int)) or not math.isfinite(value) or value <= 0:
            raise ValueError(f'Student {key} must be finite and positive.')
    expected = student_observation_fields(metadata['history_length'], metadata['command_dimension'])
    if metadata.get('action_dim') != 29 or metadata.get('observation_dim') != sum(size for _, size in expected):
        raise ValueError('Student observation/action dimensions do not match the deployment contract.')
    scales = student_observation_scales(metadata)
    for name in _SCALE_NAMES:
        scale = torch.as_tensor(scales.get(name, []), dtype=torch.float64)
        shape = (3,) if name == 'hoop_position' else ()
        if scale.shape != shape or not torch.isfinite(scale).all() or torch.any(scale <= 0):
            raise ValueError(f'Student observation scale {name} must be finite, positive and have shape {shape}.')
    if version == STUDENT_TRAINING_FORMAT:
        contract = validate_policy_contract(metadata.get('policy_contract'))
        for key in _ROBOT_FIELDS:
            if metadata[key] != contract[key]:
                raise ValueError(f'Student {key} differs from the archived policy contract.')
        fields = [dict(name=name, size=size) for name, size in expected]
        if metadata.get('observation_fields') != fields:
            raise ValueError('Checkpoint observation fields differ from the direct-action interface.')
    else:
        action = metadata.get('action', {})
        if (action.get('output') != 'dimensionless_joint_action' or action.get('unit') != 'dimensionless'
                or action.get('target') != _ACTION_TARGET or action.get('target_clip') is not None):
            raise ValueError('Student deployment action transformation is incompatible.')
        model = metadata.get('model', {})
        for key, name, dimension in (('input', 'student_observation', metadata['observation_dim']),
                                      ('output', 'joint_action', metadata['action_dim'])):
            if model.get(key) != dict(name=name, dtype='float32', shape=[None, dimension]):
                raise ValueError(f'Student deployment model {key} interface is incompatible.')
        fields = metadata.get('observation', {}).get('fields', [])
        offset = 0
        if len(fields) != len(expected):
            raise ValueError('Student observation fields do not match the direct-action interface.')
        for field, (name, size) in zip(fields, expected):
            if (field.get('name'), field.get('start'), field.get('stop')) != (name, offset, offset + size):
                raise ValueError(f'Student observation layout mismatch at {name}.')
            offset += size


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
                'joint_velocity', 'previous_action', 'hoop_position')
_ACTION_TARGET = 'nominal_joint_position + action_scale * clip(network_output, -action_clip, action_clip)'
_UNITS = ('dimensionless', 'dimensionless', 'rad/s', 'rad', 'rad/s', 'dimensionless', 'm, rad, m')
_SOURCES = (
    'one_hot(command_id, command_dimension); runtime skill selection',
    'R_world_from_pelvis.T @ [0, 0, -1]; xyz direction, not acceleration',
    'pelvis-frame angular velocity; xyz',
    'measured joint position - nominal_joint_position; joint_names order',
    'measured joint velocity; joint_names order',
    'previous policy tick requested action after action clipping, before execution delay; joint_names order',
    'external hoop_pos_cyl parameter; [r, theta, z]; update between shots, hold during a shot',
)
_SKILL_FIELDS = ('name', 'command_id', 'object', 'end')


def student_observation_scales(metadata: dict) -> dict:
    if 'observation' in metadata:
        return {key: field['scale'] for key, field in zip(_SCALE_NAMES, metadata['observation']['fields'])}
    scales = dict(metadata['observation_settings']['scales'])
    scales['skill_command'] = 1.
    position_scale = scales['hoop_position']
    scales['hoop_position'] = [position_scale, scales['hoop_theta'], position_scale]
    return scales


def deployment_skills(contract: dict) -> dict:
    """Export command labels and skill settings, without reference motion data."""
    config = contract['config']
    return dict(
        command=dict(dimension=config['command']['dimension'], encoding='one_hot',
                     switching='runtime_preserve_history'),
        definitions=[{key: copy.deepcopy(skill[key]) for key in _SKILL_FIELDS}
                     for skill in config['skills']],
    )


def student_simulation_skill_config(metadata: dict, simulation_config: dict | None = None) -> dict:
    """Simulation supplies its own reference files; robot bundles do not contain them."""
    if simulation_config is None:
        from .skills import load_skill_config
        simulation_config = load_skill_config()
    result = copy.deepcopy(simulation_config)
    definitions = metadata['skills']['definitions']
    mapping = lambda skills: {(s['name'], s['command_id']) for s in skills}
    if (result['command']['dimension'] != metadata['command_dimension']
            or mapping(result['skills']) != mapping(definitions)):
        raise ValueError('Simulation skills must match the exported skill names and command IDs.')
    by_name = {s['name']: s for s in definitions}
    for skill in result['skills']:
        skill.update(copy.deepcopy(by_name[skill['name']]))
        skill.update(sampling_weight=1.0, rsi=dict(enabled=False))
    return result


def _startup_config(metadata, definitions, default_skill, initial_pose, initial_clip_index):
    names = [s['name'] for s in definitions]
    selected = default_skill or metadata.get('default_skill') or ('hold' if 'hold' in names else names[0])
    if selected not in names:
        raise ValueError(f'Unknown default_skill: {selected}')
    if initial_pose is not None and initial_clip_index is not None:
        raise ValueError('Specify initial_pose or initial_clip_index, not both.')
    if initial_pose is None:
        if (initial_clip_index is None and selected == metadata.get('default_skill')
                and 'initial_pose' in metadata):
            initial_pose = metadata['initial_pose']
        else:
            index = 0 if initial_clip_index is None else initial_clip_index
            if type(index) is not int or index < 0:
                raise ValueError('initial_clip_index must be a non-negative integer within the default skill.')
            contract = metadata.get('deployment_contract', {}).get('skills', metadata.get('skills', {}))
            command_id = next(s['command_id'] for s in definitions if s['name'] == selected)
            profiles = [p for p in contract.get('motion_profiles', []) if p['command_id'] == command_id]
            if index >= len(profiles):
                raise ValueError('Startup pose unavailable; export from a training checkpoint or supply initial_pose.')
            initial_pose = profiles[index]['initial_joint_position_rad']
    pose = torch.as_tensor(initial_pose, dtype=torch.float32)
    if pose.shape != (29,) or not torch.isfinite(pose).all():
        raise ValueError('initial_pose must contain 29 finite joint positions in radians, in joint_names order.')
    return selected, pose.tolist()


def compact_student_config(metadata: dict, *, default_skill: str | None = None,
                           initial_pose=None, initial_clip_index: int | None = None) -> dict:
    """Project checkpoint metadata onto the deployment-only skill-command interface."""
    validate_student_metadata(metadata)
    keys = ('observation_dim', 'action_dim', 'history_length', 'command_dimension',
            *_ROBOT_FIELDS, 'action_semantics', 'student_hidden_dims', 'activation')
    if 'observation' in metadata:
        result = {key: metadata[key] for key in (
            *keys, 'format', 'model', 'observation', 'action', 'skills', 'initialization')}
        if 'export_source' in metadata:
            result['export_source'] = metadata['export_source']
        result = copy.deepcopy(result)
        _configure_startup(result, metadata, default_skill, initial_pose, initial_clip_index)
        _finalize_command_contract(result)
        return result
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
        format=STUDENT_DEPLOYMENT_FORMAT,
        model=dict(
            file='student_policy.onnx',
            input=dict(name='student_observation', dtype='float32', shape=[None, metadata['observation_dim']]),
            output=dict(name='joint_action', dtype='float32', shape=[None, metadata['action_dim']]),
        ),
        observation=dict(
            dtype='float32', layout='field_major', history_order='oldest_to_newest',
            preprocessing='multiply each field by its scale; no other normalization or input clipping',
            fields=fields,
            timing='sample current sensors before inference; previous_action_history ends with the previous policy tick requested action',
        ),
        action=dict(
            unit='dimensionless', output='dimensionless_joint_action',
            target=_ACTION_TARGET,
            target_clip=None,
            timing='clip the requested action and apply its scaled target immediately after inference',
            pd='tau = pd_stiffness * (target - measured_q) - pd_damping * measured_qdot',
        ),
        skills=skills,
    )
    if 'export_source' in metadata:
        result['export_source'] = copy.deepcopy(metadata['export_source'])
    result = copy.deepcopy(result)
    _configure_startup(result, metadata, default_skill, initial_pose, initial_clip_index)
    _finalize_command_contract(result)
    return result


def _configure_startup(result, metadata, default_skill, initial_pose, initial_clip_index):
    result['observation'].pop('sensor_delay_steps', None)
    result['action'].pop('delay_steps', None)
    result['observation']['timing'] = 'sample current sensors before inference; previous_action_history ends with the previous policy tick requested action'
    result['action']['timing'] = 'clip the requested action and apply its scaled target immediately after inference'
    definitions = result['skills']['definitions']
    if not definitions:
        raise ValueError('At least one deployment skill is required.')
    if result['skills']['command']['dimension'] != result['command_dimension']:
        raise ValueError('Skill command dimension differs from the observation dimension.')
    result['default_skill'], result['initial_pose'] = _startup_config(
        metadata, definitions, default_skill, initial_pose, initial_clip_index)
    # Keep the runtime skill interface compact when re-exporting.
    result['skills'] = deployment_skills(dict(config=dict(
        command=dict(dimension=result['command_dimension']), skills=definitions)))
    for field in result['observation']['fields']:
        if field['name'] == 'skill_command':
            field['source'] = _SOURCES[0]
        elif field['name'] == 'previous_action_history':
            field['source'] = _SOURCES[5]
        elif field['name'] == 'hoop_pos_cyl':
            field['source'] = _SOURCES[-1]
    result['initialization'] = dict(
        startup='reach initial_pose (rad, joint_names order), then run default_skill',
        sensor_history='repeat first measured gravity, gyro, q-minus-nominal and qdot',
        action_history='all zeros at reset; append the previous policy tick clipped request; retain histories across skill and hoop updates',
    )


def _finalize_command_contract(config: dict) -> None:
    """Validate the actual network layout and make sparse command labels explicit."""
    validate_student_metadata(config)
    dimension = config['command_dimension']
    skills = config['skills']
    if skills['command']['dimension'] != dimension:
        raise ValueError('Skill command dimension differs from the observation dimension.')
    labels = [None] * dimension
    names = set()
    for skill in skills['definitions']:
        command_id = skill['command_id']
        if type(command_id) is not int or not 0 <= command_id < dimension or labels[command_id] is not None:
            raise ValueError('Skill command IDs must be unique valid one-hot indices.')
        if not skill['name'] or skill['name'] in names:
            raise ValueError('Skill names must be nonempty and unique.')
        names.add(skill['name'])
        labels[command_id] = skill['name']
        skill['command_one_hot'] = [float(i == command_id) for i in range(dimension)]
    if not names:
        raise ValueError('At least one deployment skill is required.')
    skills['command']['labels'] = labels
    skills['command']['valid_ids'] = [i for i, label in enumerate(labels) if label is not None]

class StudentObservationState:
    """One robot's histories, using current calibrated samples at each policy tick."""
    def __init__(self, metadata: dict):
        validate_student_metadata(metadata)
        self.metadata = metadata
        skills = metadata.get('skills')
        if skills is not None:
            self._valid_commands = {skill['command_id'] for skill in skills['definitions']}
        else:
            contract = metadata.get('deployment_contract', {}).get('skills')
            self._valid_commands = ({skill['command_id'] for skill in contract['config']['skills']}
                                    if contract else set(range(metadata['command_dimension'])))
        self.history = metadata['history_length']
        self.nominal = torch.tensor(metadata['nominal_joint_position'], dtype=torch.float32)
        self.action_scale = torch.tensor(metadata['action_scale'], dtype=torch.float32)
        self.action_clip = float(metadata['action_clip'])
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

    def reset(self, gravity, gyro, q, qdot, hoop_pos_cyl):
        sensor = self._sample(gravity, gyro, q, qdot)
        self._sensor = sensor.repeat(self.history, 1)
        self._action = torch.zeros(self.history, 29)
        self._hoop = self._vector(hoop_pos_cyl, 3).clone()
        self._last_tick = -1
        default = self.metadata.get('default_skill')
        self._last_command = next((s['command_id'] for s in self.metadata.get('skills', {}).get('definitions', [])
                                   if s['name'] == default), None)

    def observe(self, tick: int, command_id: int | None, gravity, gyro, q, qdot, *,
                previous_requested_action=None, hoop_pos_cyl=None):
        """Sample one tick; None retains the current command/hoop. Repeated ticks are cached."""
        if self._sensor is None:
            raise ValueError('Reset deployment histories before inference.')
        if command_id is None:
            command_id = self._last_command
        if type(command_id) is not int or command_id not in self._valid_commands:
            raise ValueError('Invalid skill command ID.')
        if tick == self._last_tick:
            return self._observation.clone()
        if tick != self._last_tick + 1:
            raise ValueError('Policy tick skipped or reversed; hardware watchdog must handle missed deadlines.')
        sensor = self._sample(gravity, gyro, q, qdot)
        if previous_requested_action is None:
            if tick != 0:
                raise ValueError('Provide the previous policy tick requested action after tick zero.')
            action = torch.zeros(29)
        else:
            action = self._vector(previous_requested_action, 29)
            if tick == 0 and torch.any(action != 0):
                raise ValueError('The previous requested action must be zero at tick zero.')
            if torch.any(action.abs() > self.action_clip):
                raise ValueError('The previous requested action must already be clipped; use process_action().')
        if hoop_pos_cyl is not None:
            self._hoop = self._vector(hoop_pos_cyl, 3).clone()
        self._sensor = torch.cat((self._sensor[1:], sensor[None]))
        self._action = torch.cat((self._action[1:], action[None]))
        frames = self._sensor
        scales = student_observation_scales(self.metadata)
        self._observation = torch.cat((torch.nn.functional.one_hot(torch.tensor(command_id), self.metadata['command_dimension']).float(),
            frames[:, :3].flatten() * scales['gravity'], frames[:, 3:6].flatten() * scales['angular_velocity'],
            frames[:, 6:35].flatten() * scales['joint_position'], frames[:, 35:64].flatten() * scales['joint_velocity'],
            self._action.flatten() * scales['previous_action'], self._hoop * torch.as_tensor(scales['hoop_position'], dtype=self._hoop.dtype)))[None]
        self._last_tick, self._last_command = tick, command_id
        return self._observation.clone()

    def process_action(self, network_output):
        """Return the clipped request for next tick's history and its PD target."""
        requested = clip_joint_actions(self._vector(network_output, 29), self.action_clip)
        target = joint_position_target(requested, self.nominal, self.action_scale, self.action_clip)
        return requested, target

    def action_target(self, network_output):
        """Convert a network output without changing observation histories."""
        return self.process_action(network_output)[1]
