"""Deployment tick timing and requested-action history parity with simulation."""

import pytest
import torch

from bb_mimic_nt.action_math import joint_position_target
from bb_mimic_nt.student_deployment import StudentObservationState, compact_student_config
from student_fixtures import direct_student_metadata
from test_student_observations import make_sensor


@pytest.mark.parametrize('training_delay', [0, 1, 3])
def test_robot_observation_builder_uses_current_sensors_and_previous_request(training_delay):
    sim, env, data, action = make_sensor(nominal=0, maximum=3)
    sim.delay_steps[:] = 0
    data.projected_gravity_b[:, 2] = -1
    metadata = direct_student_metadata(nominal=.1)
    metadata['default_observation_delay_steps'] = training_delay
    metadata['observation_settings']['scales'] = vars(env.cfg.student_observation.scales)
    action.nominal[:] = .1
    real = StudentObservationState(compact_student_config(metadata))
    real.reset([0, 0, -1], [0, 0, 0], [.1] * 29, [0.] * 29, [0, 0, 0])
    for tick in range(12):
        env.common_step_counter = tick
        data.joint_pos[:] = .1 + tick
        data.joint_vel[:] = tick * .2
        action.requested_actions[:] = tick * 2
        reference = sim(env)[0:1]
        actual = real.observe(tick, 0, [0, 0, -1], [0, 0, 0], data.joint_pos[0],
                              data.joint_vel[0], previous_requested_action=action.requested_actions[0])
        torch.testing.assert_close(actual, reference)
        torch.testing.assert_close(real.observe(tick, 0, None, None, None, None), actual)
    with pytest.raises(ValueError, match='skipped'):
        real.observe(14, 0, None, None, None, None)
    with pytest.raises(ValueError, match='finite'):
        real.action_target(torch.full((29,), float('nan')))
    requested, target = real.process_action(torch.full((29,), 200.))
    assert requested.eq(100).all()
    torch.testing.assert_close(target, real.nominal + real.action_scale * 100)
    assert target.gt(3.).all()  # Physical position limits are not a target clamp.


def test_reset_zeroes_action_history_without_replacing_initial_sensor_state():
    state = StudentObservationState(compact_student_config(direct_student_metadata()))
    state.reset([0, 0, -1], [1, 2, 3], [.7] * 29, [.4] * 29, [4., 0., 3.])
    observation = state.observe(0, None, [0, 0, -1], [1, 2, 3], [.7] * 29, [.4] * 29)
    assert observation[0, -148:-3].eq(0).all()
    torch.testing.assert_close(state._sensor[:, 6:35], torch.full((5, 29), .6))
    assert state._sensor[:, 35:64].eq(.4).all()
    snapshot = state._action.clone()
    requested, target = state.process_action(torch.linspace(-200, 200, 29))
    torch.testing.assert_close(target, joint_position_target(requested, state.nominal, state.action_scale))
    torch.testing.assert_close(state._action, snapshot)
    next_observation = state.observe(1, None, [0, 0, -1], [1, 2, 3], [.7] * 29, [.4] * 29,
                                     previous_requested_action=requested)
    torch.testing.assert_close(next_observation[0, -32:-3], requested)
    assert next_observation[0, -148:-32].eq(0).all()
    with pytest.raises(ValueError, match='previous policy tick'):
        state.observe(2, None, [0, 0, -1], [0]*3, [0]*29, [0]*29)
    with pytest.raises(ValueError, match='already be clipped'):
        state.observe(2, None, [0, 0, -1], [0]*3, [0]*29, [0]*29,
                      previous_requested_action=[101.] * 29)
    state.reset([0, 0, -1], [0]*3, [.7]*29, [0]*29, [4., 0., 3.])
    with pytest.raises(ValueError, match='zero at tick zero'):
        state.observe(0, None, [0, 0, -1], [0]*3, [.7]*29, [0]*29,
                      previous_requested_action=requested)


def test_runtime_command_and_hoop_changes_preserve_histories():
    metadata = compact_student_config(direct_student_metadata())
    state, baseline = StudentObservationState(metadata), StudentObservationState(metadata)
    for builder in (state, baseline):
        builder.reset([0, 0, -1], [0]*3, [0]*29, [0]*29, [4., 0., 3.])
    for tick in range(15):
        command = None if tick == 0 else (1 if tick % 4 == 1 else 0)
        args = ([0, 0, -1], [tick*.1]*3, [tick*.2]*29, [tick*.3]*29)
        requested = [tick*.4]*29
        hoop = [5., -.3, 3.05] if tick == 8 else None
        actual = state.observe(tick, command, *args, previous_requested_action=requested, hoop_pos_cyl=hoop)
        reference = baseline.observe(tick, 0, *args, previous_requested_action=requested)
        torch.testing.assert_close(actual[:, 3:-3], reference[:, 3:-3])
        assert actual[0, :3].tolist() == ([0., 1., 0.] if command == 1 else [1., 0., 0.])
        expected_hoop = torch.tensor([5., -.3, 3.05] if tick >= 8 else [4., 0., 3.])
        expected_hoop *= torch.tensor([.25, 2., .25])
        torch.testing.assert_close(actual[0, -3:], expected_hoop)
        torch.testing.assert_close(state.observe(tick, None, None, None, None, None), actual)
    before = state._sensor.clone()
    with pytest.raises(ValueError, match='finite'):
        state.observe(15, 0, *args, previous_requested_action=requested, hoop_pos_cyl=[float('nan'), 0., 3.])
    torch.testing.assert_close(state._sensor, before)


@pytest.mark.parametrize('corruption', ['missing_version', 'old_version', 'old_history', 'scale', 'target_clip'])
def test_deployment_rejects_old_or_inconsistent_action_contract(corruption):
    metadata = compact_student_config(direct_student_metadata())
    if corruption == 'missing_version':
        del metadata['format']
    elif corruption == 'old_version':
        metadata['format'] = 'bb_mimic_nt_student_deployment_v3'
    elif corruption == 'old_history':
        metadata['observation']['fields'][5]['name'] = 'applied_action_offset_history'
    elif corruption == 'scale':
        metadata['action_scale'][0] = 0.
    else:
        metadata['action']['target_clip'] = [-3., 3.]
    with pytest.raises(ValueError):
        StudentObservationState(metadata)
