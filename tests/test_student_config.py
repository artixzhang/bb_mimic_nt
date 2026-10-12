"""Deployment startup/command mapping and sensor/action semantics."""
from types import SimpleNamespace

import pytest
import torch

from bb_mimic_nt.student_deployment import (
    StudentObservationState, build_deployment_contract, compact_student_config, student_simulation_skill_config,
)
from test_student_observations import make_sensor


@pytest.fixture
def training_metadata():
    metadata = dict(command_dimension=3, format='bb_mimic_nt_student_training_v3', observation_dim=471, action_dim=29,
                    history_length=5, joint_names=[f'joint_{i}' for i in range(29)],
                    nominal_joint_position=[.1] * 29, safe_lower_joint_position=[-2.] * 29,
                    safe_upper_joint_position=[2.] * 29, pd_stiffness=[100.] * 29,
                    pd_damping=[3.] * 29, policy_hz=100., student_hidden_dims=[16, 8], activation='elu',
                    default_observation_delay_steps=1, default_action_delay_steps=0,
                    observation_settings=dict(scales=dict(gravity=1., angular_velocity=.25,
                        joint_position=1., joint_velocity=.05, applied_action=1., hoop_position=.25, hoop_theta=2.)))
    batch = dict(metadata=dict(dof_names=metadata['joint_names'], fps=100.), lengths=torch.tensor([1, 166, 66, 100]),
                 dof_pos=torch.arange(4.).view(4, 1, 1).expand(4, 166, 29).clone(),
                 root_pos=torch.zeros(4, 166, 3),
                 root_quat=torch.tensor([1., 0., 0., 0.]).repeat(4, 166, 1),
                 object_pos=torch.zeros(4, 166, 3), release=torch.zeros(4, 166),
                 hoop_pos_cyl=torch.tensor([4., -.7, 3.05]).repeat(4, 166, 1))
    from bb_mimic_nt.skills import load_skill_config
    batch['metadata'].update(skill_config=load_skill_config(), skill_clip_counts=[1, 2, 1],
                             skill_sources=[dict(source_sha256=f'motion-{i}') for i in range(3)],
                             urdf_sha256='urdf-hash')
    metadata['deployment_contract'] = build_deployment_contract(batch)
    return metadata


@pytest.mark.parametrize('delay', [0, 1, 3])
def test_deployment_observations_match_training_with_actual_scales_and_nonzero_nominal(training_metadata, delay):
    training_metadata['default_observation_delay_steps'] = delay
    config = compact_student_config(training_metadata)
    sim, env, data, action = make_sensor(nominal=delay)
    env.cfg.student_observation.scales = SimpleNamespace(**training_metadata['observation_settings']['scales'])
    action.nominal[:] = .1
    data.projected_gravity_b[:] = torch.tensor([.1, -.2, -.97])
    data.root_link_ang_vel_w[:] = torch.tensor([.4, -.6, .8])
    data.joint_pos[:] = torch.linspace(-.3, .6, 29)
    data.joint_vel[:] = torch.linspace(-.5, .9, 29)
    action.actual_action_offset[:] = .7
    env.command_manager.get_term('motion').command[:] = torch.tensor([0., 1., 0.])
    hoop = torch.tensor([4., -.7, 3.05])
    env.command_manager.get_term('motion').reference['hoop_pos_cyl'][:] = hoop
    old, new = StudentObservationState(training_metadata), StudentObservationState(config)
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
            actual = builder.observe(tick, 1, data.projected_gravity_b[0], data.root_link_ang_vel_w[0],
                                     data.joint_pos[0], data.joint_vel[0], action.actual_action_offset[0] + .1)
            torch.testing.assert_close(actual, expected)
    # Large raw predictions are clipped outside the policy in both interfaces.
    prediction = torch.linspace(-10, 10, 29)
    torch.testing.assert_close(new.action_target(prediction), old.action_target(prediction))


def test_reexport_drops_training_metadata(training_metadata):
    config = compact_student_config(training_metadata)
    polluted = dict(config, training_observation_settings={'enable_noise': True},
                    optimizer={'lr': .001}, teacher_checkpoint='/training/teacher.pt',
                    _provenance={'teacher_sha256': 'teacher-hash'})
    assert compact_student_config(polluted) == config


def test_multi_skill_contract_is_self_contained(training_metadata):
    config = compact_student_config(training_metadata)
    assert 'format' not in config
    assert config['model'] == dict(
        file='student_policy.onnx',
        input=dict(name='student_observation', dtype='float32', shape=[None, 471]),
        output=dict(name='joint_position_offset', dtype='float32', shape=[None, 29]))
    skills = config['skills']
    assert skills['command'] == dict(dimension=3, encoding='one_hot',
                                     switching='runtime_preserve_history',
                                     labels=['hold', 'shoot', 'stand_to_hold'], valid_ids=[0, 1, 2])
    assert [(skill['name'], skill['command_id']) for skill in skills['definitions']] == [
        ('hold', 0), ('shoot', 1), ('stand_to_hold', 2)]
    assert all(not {'sampling_weight', 'rsi', 'motion_file', 'motion_profile_ids', 'initialization'} & skill.keys()
               for skill in skills['definitions'])
    assert 'motion_profiles' not in skills and 'execution' not in skills
    assert config['default_skill'] == 'hold'
    assert config['initial_pose'] == [0.] * 29
    fields = config['observation']['fields']
    assert [(f['start'], f['stop']) for f in fields] == [
        (0, 3), (3, 18), (18, 33), (33, 178), (178, 323), (323, 468), (468, 471)]
    assert [(f['frame_size'], f['history_length']) for f in fields] == [
        (3, 1), (3, 5), (3, 5), (29, 5), (29, 5), (29, 5), (3, 1)]
    simulation = student_simulation_skill_config(config)
    assert simulation['command'] == dict(dimension=3)
    assert all(s['sampling_weight'] == 1. and not s['rsi']['enabled'] for s in simulation['skills'])
    # An adapter must not change the archived checkpoint or another export.
    skills['definitions'][0]['end']['duration_s'][0] = 100.
    assert training_metadata['deployment_contract']['skills']['config']['skills'][0]['end']['duration_s'][0] == 2.


def test_sparse_command_ids_keep_their_one_hot_indices(training_metadata):
    training_metadata['command_dimension'] = 5
    training_metadata['observation_dim'] = 473
    skills = training_metadata['deployment_contract']['skills']
    skills['config']['command']['dimension'] = 5
    for skill, command_id in zip(skills['config']['skills'], [0, 2, 4]):
        skill['command_id'] = command_id
    for profile, command_id in zip(skills['motion_profiles'], [0, 2, 2, 4]):
        profile['command_id'] = command_id
    config = compact_student_config(training_metadata)
    assert config['skills']['command']['labels'] == ['hold', None, 'shoot', None, 'stand_to_hold']
    assert config['skills']['command']['valid_ids'] == [0, 2, 4]
    assert config['skills']['definitions'][1]['command_one_hot'] == [0., 0., 1., 0., 0.]
    state = StudentObservationState(config)
    state.reset([0., 0., -1.], [0.] * 3, [0.] * 29, [0.] * 29, [0.] * 29,
                [4., -.7, 3.05])
    with pytest.raises(ValueError, match='Invalid skill command'):
        state.observe(0, 1, None, None, None, None, None)
    obs = state.observe(0, 4, [0., 0., -1.], [0.] * 3, [0.] * 29, [0.] * 29, [0.] * 29)
    assert obs.shape == (1, 473)
    assert obs[0, :5].tolist() == [0., 0., 0., 0., 1.]


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
def test_simulation_uses_exported_limits_and_offsets_without_teacher_scaling(training_metadata, delay):
    config = compact_student_config(training_metadata)
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


def test_export_projects_saved_contract_without_reloading_training_sources(training_metadata):
    pytest.importorskip('rsl_rl')
    from bb_mimic_nt.student_runtime import student_prepare_export_metadata
    training_metadata['teacher_checkpoint'] = '/unavailable/teacher/model.pt'
    provenance = {}
    config = student_prepare_export_metadata(training_metadata, provenance)
    assert provenance == dict(motion_source_sha256=['motion-0', 'motion-1', 'motion-2'],
                              robot_urdf_sha256='urdf-hash')
    assert 'provenance' not in config and 'teacher_checkpoint' not in config
    assert compact_student_config(config) == config


def test_early_multiskill_checkpoint_needs_no_motion_files(training_metadata):
    import copy
    pytest.importorskip('rsl_rl')
    from bb_mimic_nt.student_runtime import student_prepare_export_metadata
    expected = compact_student_config(training_metadata)
    contract = training_metadata['deployment_contract']
    for skill in contract['skills']['config']['skills']:
        skill['motion_file'] = '/unavailable/motion.pkl'
    for profile in contract['skills']['motion_profiles']:
        for key in ('skill_clip_id', 'reference_duration_s', 'initial_hoop_pos_cyl'):
            del profile[key]
    before = copy.deepcopy(training_metadata)
    assert student_prepare_export_metadata(training_metadata) == expected
    assert training_metadata == before


def test_startup_overrides_and_reexport(training_metadata):
    config = compact_student_config(training_metadata, default_skill='shoot')
    assert config['initial_pose'] == [1.] * 29
    assert config['default_skill'] == 'shoot'
    assert compact_student_config(config) == config
    config = compact_student_config(training_metadata, default_skill='shoot', initial_clip_index=1)
    assert config['initial_pose'] == [2.] * 29
    assert compact_student_config(config) == config
    overridden = compact_student_config(config, default_skill='hold', initial_pose=[.5] * 29)
    assert overridden['default_skill'] == 'hold' and overridden['initial_pose'] == [.5] * 29
    with pytest.raises(ValueError, match='Startup pose unavailable'):
        compact_student_config(config, default_skill='hold')
    for kwargs in (dict(default_skill='missing'), dict(initial_pose=[0.] * 28),
                   dict(initial_pose=[float('nan')] * 29), dict(initial_clip_index=-1),
                   dict(initial_clip_index=99), dict(initial_clip_index=0, initial_pose=[0.] * 29)):
        with pytest.raises(ValueError):
            compact_student_config(training_metadata, **kwargs)


def test_old_export_compacts_to_single_startup_pose(training_metadata):
    config = compact_student_config(training_metadata)
    old = dict(config)
    del old['initial_pose'], old['default_skill']
    old['skills'] = dict(config['skills'], motion_profiles=training_metadata['deployment_contract']['skills']['motion_profiles'])
    assert compact_student_config(old) == config


@pytest.mark.parametrize('corruption', ['dimension', 'layout', 'duplicate', 'pose', 'command_dimension'])
def test_reexport_rejects_inconsistent_contract(training_metadata, corruption):
    config = compact_student_config(training_metadata)
    if corruption == 'dimension':
        config['observation_dim'] -= 1
    elif corruption == 'layout':
        config['observation']['fields'][0]['name'] = 'phase'
    elif corruption == 'duplicate':
        config['skills']['definitions'][1]['command_id'] = 0
    elif corruption == 'command_dimension':
        config['skills']['command']['dimension'] += 1
    else:
        config['initial_pose'] = [0.] * 28
    with pytest.raises(ValueError):
        compact_student_config(config)


def test_phase_checkpoint_cannot_be_relabelled_as_commands():
    with pytest.raises(ValueError, match='phase-based'):
        compact_student_config(dict(observation_dim=469))


def test_simulation_references_are_supplied_separately(training_metadata):
    import copy
    config = compact_student_config(training_metadata)
    local = copy.deepcopy(training_metadata['deployment_contract']['skills']['config'])
    local['skills'][0]['motion_file'] = '/local/simulation/hold.pkl'
    restored = student_simulation_skill_config(config, local)
    assert restored['skills'][0]['motion_file'] == '/local/simulation/hold.pkl'
    local['skills'][0]['command_id'] = 9
    with pytest.raises(ValueError, match='Simulation skills'):
        student_simulation_skill_config(config, local)
