"""The public deployment interface preserves sensor/action semantics and rejects corrupt layouts."""
import copy
from types import SimpleNamespace

import pytest
import torch

from bb_mimic_nt.student_deployment import (
    StudentObservationState, build_deployment_contract, compact_student_config, validate_student_config,
)
from test_student_observations import make_sensor


@pytest.fixture
def legacy():
    metadata = dict(format='bb_mimic_nt_student_v1', observation_dim=469, action_dim=29,
                    history_length=5, joint_names=[f'joint_{i}' for i in range(29)],
                    nominal_joint_position=[.1] * 29, safe_lower_joint_position=[-2.] * 29,
                    safe_upper_joint_position=[2.] * 29, pd_stiffness=[100.] * 29,
                    pd_damping=[3.] * 29, policy_hz=100., student_hidden_dims=[16, 8], activation='elu',
                    default_observation_delay_steps=1, default_action_delay_steps=0,
                    observation_settings=dict(scales=dict(phase=1., gravity=1., angular_velocity=.25,
                        joint_position=1., joint_velocity=.05, applied_action=1., hoop_position=.25, hoop_theta=2.)))
    batch = dict(metadata=dict(dof_names=metadata['joint_names'], fps=100.), lengths=torch.tensor([166]),
                 dof_pos=torch.zeros(1, 166, 29), root_pos=torch.zeros(1, 166, 3),
                 root_quat=torch.tensor([1., 0., 0., 0.]).repeat(1, 166, 1),
                 object_pos=torch.zeros(1, 166, 3), release=torch.zeros(1, 166))
    metadata['deployment_contract'] = build_deployment_contract(metadata, batch,
                                                               SimpleNamespace(pre_hold_s=5., post_hold_s=5.))
    return metadata


@pytest.mark.parametrize('delay', [0, 1, 3])
def test_v2_observations_match_training_with_actual_scales_and_nonzero_nominal(legacy, delay):
    legacy['default_observation_delay_steps'] = delay
    config = compact_student_config(legacy)
    sim, env, data, action = make_sensor(nominal=delay)
    env.cfg.student_observation.scales = SimpleNamespace(**legacy['observation_settings']['scales'])
    action.nominal[:] = .1
    data.projected_gravity_b[:] = torch.tensor([.1, -.2, -.97])
    data.root_link_ang_vel_w[:] = torch.tensor([.4, -.6, .8])
    data.joint_pos[:] = torch.linspace(-.3, .6, 29)
    data.joint_vel[:] = torch.linspace(-.5, .9, 29)
    action.actual_action_offset[:] = .7
    env.command_manager.get_term('motion').normalized_phase[:] = .4
    hoop = torch.tensor([4., -.7, 3.05])
    env.command_manager.get_term('motion').reference['hoop_pos_cyl'][:] = hoop
    old, new = StudentObservationState(legacy), StudentObservationState(config)
    for builder in (old, new):
        builder.reset(data.projected_gravity_b[0], data.root_link_ang_vel_w[0], data.joint_pos[0],
                      data.joint_vel[0], action.actual_action_offset[0] + .1, hoop)
    for tick in range(12):
        env.common_step_counter = tick
        if tick:
            data.joint_pos += .01
            data.joint_vel += .02
            data.root_link_ang_vel_w += .1
            action.actual_action_offset += .03
        expected = sim(env)[0:1]
        torch.testing.assert_close(expected[0, -3:], hoop * torch.tensor([.25, 2., .25]))
        for builder in (old, new):
            actual = builder.observe(tick, .4, data.projected_gravity_b[0], data.root_link_ang_vel_w[0],
                                     data.joint_pos[0], data.joint_vel[0], action.actual_action_offset[0] + .1)
            torch.testing.assert_close(actual, expected)
    # Large raw predictions are clipped outside the policy in both interfaces.
    prediction = torch.linspace(-10, 10, 29)
    torch.testing.assert_close(new.action_target(prediction), old.action_target(prediction))


def test_reexport_drops_training_metadata_from_v2(legacy):
    config = compact_student_config(legacy)
    polluted = dict(config, training_observation_settings={'enable_noise': True},
                    optimizer={'lr': .001}, teacher_checkpoint='/training/teacher.pt',
                    _provenance={'teacher_sha256': 'teacher-hash'})
    assert compact_student_config(polluted) == config


@pytest.mark.parametrize('corrupt', ['field_order', 'slice', 'unit', 'scale', 'delay', 'limit', 'quaternion'])
def test_v2_rejects_inconsistent_interfaces(legacy, corrupt):
    cfg = compact_student_config(legacy)
    if corrupt == 'field_order':
        cfg['observation']['fields'][1:3] = cfg['observation']['fields'][1:3][::-1]
    elif corrupt == 'slice':
        cfg['observation']['fields'][2]['start'] += 1
    elif corrupt == 'unit':
        cfg['observation']['fields'][3]['unit'] = 'degree'
    elif corrupt == 'scale':
        cfg['observation']['fields'][4]['scale'] = float('nan')
    elif corrupt == 'delay':
        cfg['action']['delay_steps'] = -1
    elif corrupt == 'limit':
        cfg['safe_lower_joint_position'][0] = 3.
    else:
        cfg['phase']['motion_profiles'][0]['initial_root_quaternion_wxyz'] = [0.] * 4
    with pytest.raises(ValueError):
        validate_student_config(cfg)
    with pytest.raises(ValueError):
        compact_student_config(copy.deepcopy(cfg))


def _simulation_action(cfg, delay=0):
    """Run the actual action term with CPU articulation buffers, without Isaac Sim."""
    import ast
    from collections.abc import Sequence
    from pathlib import Path
    from bb_mimic_nt.core import apply_joint_residual_caps
    from test_domain_randomization import _delay_selector

    path = Path(__file__).resolve().parents[1] / 'source/bb_mimic_nt/bb_mimic_nt/tasks/manager_based/bb_mimic_nt/mdp/actions.py'
    node = next(node for node in ast.parse(path.read_text()).body
                if isinstance(node, ast.ClassDef) and node.name == 'StudentNominalJointPositionAction')

    class Base:
        def __init__(self, config, env):
            self.cfg, self.device, self.num_envs, self._asset = config, 'cpu', 1, env.asset

    names = [f'joint_{i}' for i in range(29)]
    data = SimpleNamespace(default_joint_pos=torch.zeros(1, 29),
                           joint_pos_limits=torch.tensor([-3., 3.]).repeat(1, 29, 1))
    asset = SimpleNamespace(data=data, find_joints=lambda _, preserve_order: (list(range(29)), names),
                            set_joint_position_target=lambda targets, joint_ids: None)
    motion = SimpleNamespace(dof_names=names, reference=dict(dof_pos=torch.full((1, 29), .2)))
    env = SimpleNamespace(asset=asset, command_manager=SimpleNamespace(get_term=lambda _: motion),
                          cfg=SimpleNamespace(dr=SimpleNamespace(delay_nominal_steps=delay, delay_max_offset_steps=0)))
    namespace = dict(torch=torch, ActionTerm=Base, Articulation=object, Sequence=Sequence,
                     apply_joint_residual_caps=apply_joint_residual_caps,
                     delayed_target=_delay_selector(),
                     get_dr_context=lambda _: SimpleNamespace(delay_steps=torch.tensor([delay])))
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), 'exec'), namespace)
    return namespace['StudentNominalJointPositionAction'](cfg, env)


@pytest.mark.parametrize('delay', [0, 2])
def test_simulation_uses_exported_limits_and_offsets_without_teacher_scaling(legacy, delay):
    config = compact_student_config(legacy)
    config['safe_lower_joint_position'] = [-.5] * 29
    config['safe_upper_joint_position'] = [1.] * 29
    cfg = SimpleNamespace(joint_names=config['joint_names'], command_name='motion',
                          position_limit_margin=.02, residual_scale_fraction=.2,
                          minimum_residual_scale=.1, maximum_residual_scale=.5,
                          joint_maximum_residual_scales={},
                          nominal_joint_position=config['nominal_joint_position'],
                          safe_lower_joint_position=config['safe_lower_joint_position'],
                          safe_upper_joint_position=config['safe_upper_joint_position'])
    action = _simulation_action(cfg, delay)
    prediction = torch.linspace(-3., .6, 29)[None]
    expected = StudentObservationState(config).action_target(prediction)
    for step in range(delay + 1):
        action.process_actions(prediction)
        target = expected[None] if step >= delay else torch.full((1, 29), .2)
        torch.testing.assert_close(action.desired_actions, target)
    torch.testing.assert_close(action.actual_action_offset, expected[None] - .1)
    cfg.safe_lower_joint_position = [-4.] * 29
    with pytest.raises(ValueError, match='exceed simulator'):
        _simulation_action(cfg)


def test_legacy_cache_recovery_keeps_provenance_outside_public_config(legacy, tmp_path, monkeypatch):
    pytest.importorskip('rsl_rl')
    from bb_mimic_nt import student_runtime as runtime

    del legacy['deployment_contract']
    legacy['teacher_checkpoint'] = '/archived/teacher/model.pt'
    cache = tmp_path / 'motion.pt'
    torch.save(dict(metadata=dict(dof_names=legacy['joint_names'], fps=100., source_sha256='motion-hash',
                                  urdf_sha256='urdf-hash'), lengths=torch.tensor([166]),
                    dof_pos=torch.zeros(1, 166, 29), root_pos=torch.zeros(1, 166, 3),
                    root_quat=torch.tensor([1., 0., 0., 0.]).repeat(1, 166, 1),
                    object_pos=torch.zeros(1, 166, 3), release=torch.zeros(1, 166)), cache)
    monkeypatch.setattr(runtime, 'student_teacher_configs', lambda _: ({}, dict(commands=dict(motion=dict(
        cache_path=str(cache), pre_hold_s=5., post_hold_s=5.)))))
    provenance = {}
    config = runtime.student_prepare_export_metadata(legacy, provenance)
    assert config['phase']['pre_hold_s'] == 5.
    assert provenance == dict(motion_source_sha256='motion-hash', robot_urdf_sha256='urdf-hash',
                              recovered_from_archived_cache=True)
    assert 'provenance' not in config and 'teacher_checkpoint' not in config
