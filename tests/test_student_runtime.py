"""Legacy deployment, five-frame network export, and dimension validation."""

from types import SimpleNamespace

import pytest
import torch

pytest.importorskip('rsl_rl')
from bb_mimic_nt.student import student_observation_fields
from bb_mimic_nt.student_runtime import (
    StudentPolicy, student_apply_deployment_config, student_artifact_history,
    student_export, student_load_policy, student_policy_from_state,
)


@pytest.mark.parametrize('history', [3, 5])
@pytest.mark.parametrize('stale_source', [False, True])
def test_student_deployment_and_torchscript_roundtrip(tmp_path, history, stale_source, monkeypatch):
    metadata = dict(format='bb_mimic_nt_student_v1', observation_dim=93 * history + 4,
                    action_dim=29, student_hidden_dims=[16, 8], activation='elu',
                    observation_fields=[dict(name=name, size=size) for name, size in student_observation_fields(history)])
    if history == 5:
        metadata['history_length'] = 5
    metadata.update(joint_names=[f'joint_{i}' for i in range(29)],
                    nominal_joint_position=[0.] * 29, safe_lower_joint_position=[-2.] * 29,
                    safe_upper_joint_position=[2.] * 29, pd_stiffness=[100.] * 29, pd_damping=[3.] * 29,
                    policy_hz=100., history_order='oldest_to_newest',
                    action_semantics='joint_position_radians_minus_nominal_pose',
                    default_action_delay_steps=0, default_observation_delay_steps=1,
                    observation_settings=dict(scales=dict(phase=1., gravity=1., angular_velocity=.25,
                        joint_position=1., joint_velocity=.05, applied_action=1., hoop_position=.25, hoop_theta=2.)))
    from bb_mimic_nt.student_deployment import build_deployment_contract
    batch = dict(metadata=dict(dof_names=metadata['joint_names'], fps=100.), lengths=torch.tensor([166]),
                 dof_pos=torch.zeros(1, 166, 29), root_pos=torch.zeros(1, 166, 3),
                 root_quat=torch.tensor([1., 0., 0., 0.]).repeat(1, 166, 1),
                 object_pos=torch.zeros(1, 166, 3), release=torch.zeros(1, 166))
    metadata['history_length'] = history
    metadata['deployment_contract'] = build_deployment_contract(
        metadata, batch, SimpleNamespace(pre_hold_s=0., post_hold_s=1.))
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
    assert student_artifact_history(saved) == history
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
    assert deployment['format'] == 'bb_mimic_nt_student_v2'
    profile = deployment['phase']['motion_profiles'][0]
    assert profile['reference_fps'] / deployment['policy_hz'] / (profile['frame_count'] - 1) == pytest.approx(1 / 165)
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
    assert cfg.student_observation.delay_min_steps == cfg.student_observation.delay_max_steps == 1
    assert cfg.actions.joint_pos.safe_lower_joint_position == [-2.] * 29
    assert cfg.commands.motion.post_hold_s == 1.
    # A v2 bundle can be exported again without the archived Teacher/cache.
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


def test_legacy_deployment_restores_original_history_and_delay():
    metadata = dict(observation_dim=283, default_action_delay_steps=2, action_adapter={},
                    observation_settings=dict(delay_nominal_steps=2, delay_max_offset_steps=2,
                                              gravity_noise_std=.015, angular_velocity_noise_std=.05,
                                              joint_position_noise_std=.01, joint_velocity_noise_std=.1, scales={}))
    cfg = SimpleNamespace(student_observation=SimpleNamespace(scales=SimpleNamespace()),
                          dr=SimpleNamespace(), actions=SimpleNamespace(joint_pos=SimpleNamespace()))
    student_apply_deployment_config(cfg, metadata)
    assert cfg.student_observation.history_length == 3
    assert (cfg.student_observation.delay_min_steps, cfg.student_observation.delay_max_steps) == (0, 4)
    assert cfg.student_observation.delay_nominal_steps == 2
    assert cfg.student_observation.randomize_delay is False


def test_export_rejects_inconsistent_history_metadata():
    with pytest.raises(ValueError, match='history length'):
        student_artifact_history(dict(observation_dim=469, history_length=3))


@pytest.mark.parametrize('enabled', [True, False])
@pytest.mark.parametrize('hold_weights', [None, (.2, .3)])
def test_teacher_loading_preserves_student_curriculum_and_play_flags(monkeypatch, enabled, hold_weights):
    import bb_mimic_nt.student_runtime as runtime

    names = ('delay', 'ball_mass', 'pd_gains', 'hand_friction', 'foot_friction', 'link_mass', 'push')
    ranges = dict(delay_nominal_steps=0, delay_max_offset_steps=2, ball_mass_fraction=.05,
                  pd_gain_fraction=.1, hand_friction_nominal=.8, hand_friction_delta=.2,
                  foot_friction_nominal=.9, foot_friction_delta=.3, link_mass_fraction=.1,
                  push_torque_max_nm=5., push_duration_s=.8,
                  push_force_min_n=(8., 8., 3.), push_force_max_n=(20., 20., 8.))
    archived = dict(ranges)
    archived.update({name: dict(enabled=True, start_fraction=.15, end_fraction=.8) for name in names})
    action = dict(residual_scale_fraction=.2, minimum_residual_scale=.05,
                  maximum_residual_scale=.5, position_limit_margin=.01)
    motion = dict(source_path='motion.pkl', cache_path='motion.pt', urdf_path='robot.urdf',
                  pre_hold_s=0., post_hold_s=1., reset_dof_pos_noise=.05,
                  reset_dof_vel_noise=.05, default_clip_id=0)
    if hold_weights is not None:
        motion['hold_sampling_weights'] = hold_weights
    environment = dict(dr=archived, actions=dict(joint_pos=action), commands=dict(motion=motion))
    monkeypatch.setattr(runtime, 'student_teacher_configs', lambda checkpoint: ({}, environment))
    cfg = SimpleNamespace(dr=SimpleNamespace(**{
        name: SimpleNamespace(enabled=enabled, start_fraction=.1, end_fraction=.35, full_strength=False)
        for name in names}), actions=SimpleNamespace(joint_pos=SimpleNamespace()),
        commands=SimpleNamespace(motion=SimpleNamespace(hold_sampling_weights=(.15, .15) if enabled else None)))
    runtime.student_apply_teacher_config(cfg, None)
    assert cfg.dr.push_force_max_n == (20., 20., 8.)
    assert cfg.dr.delay_nominal_steps == 0
    assert cfg.commands.motion.hold_sampling_weights == (hold_weights if enabled else None)
    for name in names:
        schedule = getattr(cfg.dr, name)
        assert schedule.enabled == enabled
        assert (schedule.start_fraction, schedule.end_fraction) == (.1, .35)
        assert schedule.full_strength is False


def test_resume_allows_new_curriculum_but_not_new_sensor_contract():
    from bb_mimic_nt.student_runtime import student_validate_resume_observation
    previous = dict(observation_dim=469, training_observation_settings=dict(
        history_length=5, delay_nominal_steps=1, delay_min_steps=0, delay_max_steps=3,
        start_fraction=0., end_fraction=.3, noise_strength=1.))
    current = dict(previous, training_observation_settings=dict(previous['training_observation_settings'], end_fraction=.2))
    student_validate_resume_observation(previous, current)
    current['training_observation_settings']['delay_nominal_steps'] = 0
    with pytest.raises(ValueError, match='noise/history/delay'):
        student_validate_resume_observation(previous, current)
    with pytest.raises(ValueError, match='history dimension'):
        student_validate_resume_observation(dict(previous, observation_dim=283), previous)
