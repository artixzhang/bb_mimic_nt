"""Student deployment export and inference round trips."""

from types import SimpleNamespace

import pytest
import torch

pytest.importorskip('rsl_rl')
from bb_mimic_nt.student import student_observation_fields
from student_fixtures import direct_student_metadata
from bb_mimic_nt.student_runtime import (
    StudentPolicy, student_apply_deployment_config,
    student_export, student_load_policy, student_policy_from_state,
)


@pytest.mark.parametrize('history', [3, 5, 10])
@pytest.mark.parametrize('stale_source', [False, True])
def test_student_deployment_and_torchscript_roundtrip(tmp_path, history, stale_source, monkeypatch):
    metadata = direct_student_metadata(history=history, nominal=0.)
    metadata['teacher_sha256'] = 'archived-teacher-hash'
    metadata['export_source'] = dict(checkpoint='model_42.pt', iteration=42)
    policy = StudentPolicy([16, 8], 'elu', metadata['observation_dim']).eval()
    if stale_source:
        import inspect, types
        forward = StudentPolicy.forward
        replacement = types.FunctionType(
            forward.__code__.replace(co_firstlineno=StudentPolicy.__init__.__code__.co_firstlineno),
            forward.__globals__, forward.__name__, forward.__defaults__, forward.__closure__)
        monkeypatch.setattr(StudentPolicy, 'forward', replacement)
        assert 'super().__init__()' in inspect.getsource(StudentPolicy.forward)
    (tmp_path / "deployment_config.json").write_text("obsolete generated duplicate")
    source = tmp_path / 'model_42.pt'
    torch.save(dict(student=policy.student.state_dict(), iteration=42), source)
    student_export(policy, tmp_path, metadata, source_path=source, source_iteration=42)
    loaded, saved = student_load_policy(tmp_path / 'student_weights.pt', None, 'cpu')
    scripted = torch.jit.load(str(tmp_path / 'student_policy.pt'))
    inputs = torch.randn(4, metadata['observation_dim'])
    torch.testing.assert_close(loaded(inputs), policy(inputs))
    torch.testing.assert_close(scripted(inputs), policy(inputs))
    assert saved['history_length'] == history
    import hashlib, json
    manifest = json.loads((tmp_path / 'manifest.json').read_text())
    for name, digest in manifest['artifacts_sha256'].items():
        assert hashlib.sha256((tmp_path / name).read_bytes()).hexdigest() == digest
    assert manifest['provenance']['teacher_sha256'] == 'archived-teacher-hash'
    assert saved['_provenance'] == manifest['provenance']
    deployment = json.loads((tmp_path / 'student_config.json').read_text())
    assert deployment['export_source'] == dict(
        path=str(source.resolve()), file=source.name, size_bytes=source.stat().st_size,
        sha256=hashlib.sha256(source.read_bytes()).hexdigest(), iteration=42)
    assert manifest['provenance']['export_source'] == deployment['export_source']
    assert not (tmp_path / 'deployment_config.json').exists()
    assert deployment['format'] == 'bb_mimic_nt_student_deployment_v4'
    assert deployment['model']['output']['name'] == 'joint_action'
    assert deployment['default_skill'] == 'hold'
    assert deployment['initial_pose'] == [0.] * 29
    assert 'motion_profiles' not in deployment['skills']
    assert not {'deployment_contract', 'observation_settings', 'training_observation_settings',
                'action_adapter', 'teacher_checkpoint'} & deployment.keys()
    cfg = SimpleNamespace(student_observation=SimpleNamespace(scales=SimpleNamespace()),
                          dr=SimpleNamespace(), actions=SimpleNamespace(joint_pos=SimpleNamespace()),
                          commands=SimpleNamespace(motion=SimpleNamespace()))
    student_apply_deployment_config(cfg, saved)
    assert cfg.student_observation.history_length == history
    assert cfg.student_observation.scales.joint_velocity == .05
    assert cfg.student_observation.scales.hoop_position == .25
    assert cfg.student_observation.scales.hoop_theta == 2.
    assert cfg.student_observation.delay_min_steps == cfg.student_observation.delay_max_steps == 0
    assert cfg.actions.joint_pos.action_scale_values == metadata['action_scale']
    assert cfg.actions.joint_pos.action_clip == 100.
    assert [skill['command_id'] for skill in cfg.commands.motion.skills_config['skills']] == [0, 1, 2]
    assert [skill['end'] for skill in cfg.commands.motion.skills_config['skills']] == [
        skill['end'] for skill in saved['skills']['definitions']]
    assert all(not skill['rsi']['enabled'] for skill in cfg.commands.motion.skills_config['skills'])
    import numpy as np
    vectors = np.load(tmp_path / 'verification_vectors.npz')['observations']
    commands = vectors[:, :3]
    assert set(commands.argmax(axis=1)) == {0, 1, 2}
    assert np.all((commands == 0) | (commands == 1)) and np.all(commands.sum(axis=1) == 1)
    # A deployment bundle can be exported again without the archived Teacher/cache.
    student_export(loaded, tmp_path / 'again', saved)
    assert json.loads((tmp_path / 'again/student_config.json').read_text()) == deployment
    assert json.loads((tmp_path / 'again/manifest.json').read_text())['provenance'] == manifest['provenance']

    # An explicit weight re-export records the actual input, not an inherited checkpoint.
    weights = tmp_path / 'student_weights.pt'
    student_export(loaded, tmp_path / 'from_weights', saved, source_path=weights)
    reexported = json.loads((tmp_path / 'from_weights/student_config.json').read_text())
    assert reexported['export_source'] == dict(
        path=str(weights.resolve()), file=weights.name, size_bytes=weights.stat().st_size,
        sha256=hashlib.sha256(weights.read_bytes()).hexdigest())

    import onnx
    onnx.checker.check_model(str(tmp_path / 'student_policy.onnx'))


def test_training_metadata_deploys_without_artificial_delay():
    metadata = direct_student_metadata(history=3)
    metadata['default_action_delay_steps'] = 2
    cfg = SimpleNamespace(student_observation=SimpleNamespace(scales=SimpleNamespace()),
                          dr=SimpleNamespace(), actions=SimpleNamespace(joint_pos=SimpleNamespace()),
                          commands=SimpleNamespace(motion=SimpleNamespace()))
    student_apply_deployment_config(cfg, metadata)
    assert cfg.student_observation.history_length == 3
    assert (cfg.student_observation.delay_min_steps, cfg.student_observation.delay_max_steps) == (0, 0)
    assert cfg.student_observation.delay_nominal_steps == 0
    assert cfg.dr.delay_nominal_steps == cfg.dr.delay_max_offset_steps == 0
    assert cfg.student_observation.randomize_delay is False



@pytest.mark.parametrize('enabled', [True, False])
@pytest.mark.parametrize('modern_archive', [True, False])
@pytest.mark.parametrize('teacher_init_enabled', [True, False])
def test_teacher_loading_preserves_student_curriculum_and_play_flags(
        monkeypatch, enabled, modern_archive, teacher_init_enabled):
    import copy
    import bb_mimic_nt.student_runtime as runtime

    names = ('delay', 'ball_mass', 'pd_gains', 'hand_friction', 'foot_friction', 'link_mass',
             'push', 'global_push', 'object_push', 'velocity_push')
    ranges = dict(delay_nominal_steps=0, delay_max_offset_steps=2, ball_mass_fraction=.05,
                  pd_gain_fraction=.1, hand_friction_nominal=.8, hand_friction_delta=.2,
                  foot_friction_nominal=.9, foot_friction_delta=.3, link_mass_fraction=.1,
                  push_torque_max_nm=5., push_duration_s=.8,
                  push_force_min_n=(8., 8., 3.), push_force_max_n=(20., 20., 8.))
    archived = dict(ranges)
    archived.update({name: dict(enabled=True, start_fraction=.15, end_fraction=.8)
                     for name in (names if modern_archive else names[:-3])})
    init_components = {
        name: dict(enabled=True, std=value) for name, value in
        [('dof_pos', .12), ('dof_vel', .02), ('root_pos', .03), ('root_vel', .06), ('root_ang_vel', .03)]
    }
    init_components.update(root_rot=dict(enabled=True, max_angle_deg=12.),
                           object_pos=dict(enabled=True, half_range_m=.2))
    init_components['dof_vel']['enabled'] = False
    if modern_archive:
        archived['init_state'] = dict(enabled=teacher_init_enabled, **init_components)
        for prefix in ('push', 'global_push', 'object_push', 'velocity_push'):
            archived[f'{prefix}_interval_range_s'] = [2., 6.]
            archived[f'{prefix}_probability'] = .7
        archived['velocity_push_linear_max_m_s'] = .8
        archived['velocity_push_angular_max_rad_s'] = .9
    action = dict(action_scale_factor=.25, action_clip=100.)
    from bb_mimic_nt.skills import load_skill_config
    motion = dict(skills_path='skills.yaml', skills_config=load_skill_config(), urdf_path='robot.urdf',
                  reset_dof_pos_noise=.05,
                  reset_dof_vel_noise=.05, default_clip_id=0)
    environment = dict(dr=archived, actions=dict(joint_pos=action), commands=dict(motion=motion))
    monkeypatch.setattr(runtime, 'student_teacher_configs', lambda checkpoint: ({}, environment))
    contract = direct_student_metadata()['policy_contract']
    monkeypatch.setattr(runtime, 'load_teacher_contract', lambda checkpoint: contract)
    cfg = SimpleNamespace(dr=SimpleNamespace(**{
        name: SimpleNamespace(enabled=enabled, start_fraction=.1, end_fraction=.35, full_strength=False)
        for name in names}), actions=SimpleNamespace(joint_pos=SimpleNamespace()),
        commands=SimpleNamespace(motion=SimpleNamespace(enable_rsi=enabled)))
    cfg.dr.init_state = SimpleNamespace(enabled=enabled, **{
        name: SimpleNamespace(**dict(component, enabled=True)) for name, component in init_components.items()})
    # A destination's independently disabled component must stay disabled, too.
    cfg.dr.init_state.root_vel.enabled = False
    cfg.dr.delay_nominal_steps = 1
    cfg.dr.delay_max_offset_steps = 4
    archived_before = copy.deepcopy(environment)
    runtime.student_apply_teacher_config(cfg, None)
    assert cfg.dr.push_force_max_n == (20., 20., 8.)
    assert cfg.dr.delay_nominal_steps == 1
    assert cfg.dr.delay_max_offset_steps == 4
    assert cfg.commands.motion.skills_config == motion['skills_config']
    assert cfg.commands.motion.enable_rsi == enabled
    for name in names:
        schedule = getattr(cfg.dr, name)
        assert schedule.enabled == (enabled and (modern_archive or name in names[:-3]))
        assert (schedule.start_fraction, schedule.end_fraction) == (.1, .35)
        assert schedule.full_strength is False
    assert environment == archived_before
    if modern_archive:
        assert cfg.dr.init_state.enabled is (enabled and teacher_init_enabled)
        for name, source in init_components.items():
            component = getattr(cfg.dr.init_state, name)
            for key, value in source.items():
                if key != 'enabled':
                    assert getattr(component, key) == value
            assert component.enabled is (name not in ('root_vel', 'dof_vel'))
        for prefix in ('push', 'global_push', 'object_push', 'velocity_push'):
            assert getattr(cfg.dr, f'{prefix}_interval_range_s') == (2., 6.)
            assert getattr(cfg.dr, f'{prefix}_probability') == .7
        assert cfg.dr.velocity_push_linear_max_m_s == .8
        assert cfg.dr.velocity_push_angular_max_rad_s == .9
    else:
        assert cfg.dr.init_state is None
    expected_legacy_noise = .05 if enabled and not modern_archive else 0.
    assert cfg.commands.motion.reset_dof_pos_noise == expected_legacy_noise
    assert cfg.commands.motion.reset_dof_vel_noise == expected_legacy_noise


def test_resume_allows_new_curriculum_but_not_new_sensor_contract():
    from bb_mimic_nt.student_runtime import student_validate_resume_observation
    previous = direct_student_metadata()
    current = dict(previous, training_observation_settings=dict(previous['training_observation_settings'], end_fraction=.2))
    student_validate_resume_observation(previous, current)
    current['training_observation_settings']['delay_nominal_steps'] = 0
    with pytest.raises(ValueError, match='noise/history/delay'):
        student_validate_resume_observation(previous, current)
    with pytest.raises(ValueError, match='dimensions'):
        student_validate_resume_observation(dict(previous, observation_dim=283), previous)


@pytest.mark.parametrize('legacy_format', [None, 'bb_mimic_nt_student_training_v3'])
def test_old_checkpoint_is_rejected_before_loading_student_weights(tmp_path, legacy_format):
    from bb_mimic_nt.student_runtime import student_read_artifact
    metadata = direct_student_metadata()
    metadata['format'] = legacy_format
    checkpoint = tmp_path / 'model_1.pt'
    torch.save(dict(format='bb_mimic_nt_student_dagger_v1', student={}, metadata=metadata), checkpoint)
    with pytest.raises(ValueError, match='Legacy Student'):
        student_read_artifact(checkpoint)
    with pytest.raises(ValueError, match='Legacy Student'):
        student_policy_from_state({}, metadata, 'cpu')


def test_resume_rejects_same_size_action_coordinate_change():
    import copy
    from bb_mimic_nt.student_runtime import student_validate_resume_observation
    previous = direct_student_metadata()
    current = copy.deepcopy(previous)
    current['action_scale'][0] *= 2
    current['policy_contract']['action_scale'][0] *= 2
    with pytest.raises(ValueError, match='action_scale'):
        student_validate_resume_observation(previous, current)


@pytest.mark.parametrize('checkpoint_format,history', [
    ('bb_mimic_nt_student_dagger_v1', 5), ('bb_mimic_nt_student_ppo_bc_v1', 10),
])
def test_training_artifacts_use_their_own_history_and_actor_weights(tmp_path, checkpoint_format, history):
    metadata = direct_student_metadata(history=history)
    policy = StudentPolicy(metadata['student_hidden_dims'], metadata['activation'], metadata['observation_dim'])
    path = tmp_path / 'model_50.pt'
    torch.save(dict(format=checkpoint_format, student=policy.student.state_dict(), metadata=metadata,
                    next_iteration=50), path)
    restored, saved = student_load_policy(path, None, 'cpu')
    inputs = torch.randn(2, metadata['observation_dim'])
    torch.testing.assert_close(restored(inputs), policy(inputs))
    assert saved['history_length'] == history
