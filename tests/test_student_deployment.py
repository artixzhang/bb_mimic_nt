"""Deployment sensor parity and shot-clock behavior against simulation semantics."""

import pytest
import torch

from bb_mimic_nt.student_deployment import StudentObservationState
from test_student_observations import make_sensor


@pytest.mark.parametrize('delay', [0, 1, 3])
def test_robot_observation_builder_ignores_legacy_delay(delay):
    sim, env, data, action = make_sensor(nominal=0, maximum=3)
    sim.delay_steps[:] = 0
    data.projected_gravity_b[:, 2] = -1
    meta = dict(command_dimension=3, history_length=5, default_observation_delay_steps=delay,
                nominal_joint_position=[0.] * 29, safe_lower_joint_position=[-100.] * 29,
                safe_upper_joint_position=[100.] * 29,
                observation_settings=dict(scales=vars(env.cfg.student_observation.scales)))
    real = StudentObservationState(meta)
    real.reset([0, 0, -1], [0, 0, 0], [0.] * 29, [0.] * 29, [0.] * 29, [0, 0, 0])
    for tick in range(12):
        env.common_step_counter = tick
        data.joint_pos[:] = tick
        data.joint_vel[:] = tick * .2
        action.actual_action_offset[:] = tick * 2
        reference = sim(env)[0:1]
        actual = real.observe(tick, 0, [0, 0, -1], [0, 0, 0], data.joint_pos[0],
                              data.joint_vel[0], action.actual_action_offset[0])
        torch.testing.assert_close(actual, reference)
        torch.testing.assert_close(real.observe(tick, 0, None, None, None, None, None), actual)
    with pytest.raises(ValueError, match='skipped'):
        real.observe(14, 0, None, None, None, None, None)
    with pytest.raises(ValueError, match='finite'):
        real.action_target(torch.full((29,), float('nan')))
    assert real.action_target(torch.full((29,), 200.)).eq(100).all()


@pytest.mark.parametrize('delay', [0, 1, 3])
def test_runtime_command_and_hoop_changes_preserve_histories(delay):
    sim, env, data, action = make_sensor()
    metadata = dict(command_dimension=3, history_length=5, default_observation_delay_steps=delay,
                    nominal_joint_position=[0.] * 29, safe_lower_joint_position=[-2.] * 29,
                    safe_upper_joint_position=[2.] * 29,
                    default_skill='hold', skills=dict(definitions=[dict(name='hold', command_id=0),
                                                                   dict(name='shoot', command_id=1)]),
                    observation_settings=dict(scales=vars(env.cfg.student_observation.scales)))
    state, baseline = StudentObservationState(metadata), StudentObservationState(metadata)
    for builder in (state, baseline):
        builder.reset([0, 0, -1], [0]*3, [0]*29, [0]*29, [0]*29, [4., 0., 3.])
    for tick in range(15):
        command = None if tick == 0 else (1 if tick % 4 == 1 else 0)
        args = ([0, 0, -1], [tick*.1]*3, [tick*.2]*29, [tick*.3]*29, [tick*.4]*29)
        hoop = [5., -.3, 3.05] if tick == 8 else None
        actual = state.observe(tick, command, *args, hoop_pos_cyl=hoop)
        reference = baseline.observe(tick, 0, *args)
        torch.testing.assert_close(actual[:, 3:-3], reference[:, 3:-3])
        assert actual[0, :3].tolist() == ([0., 1., 0.] if command == 1 else [1., 0., 0.])
        scales = metadata['observation_settings']['scales']
        expected_hoop = torch.tensor([5., -.3, 3.05] if tick >= 8 else [4., 0., 3.])
        expected_hoop *= torch.tensor([scales['hoop_position'], scales['hoop_theta'], scales['hoop_position']])
        torch.testing.assert_close(actual[0, -3:], expected_hoop)
        torch.testing.assert_close(state.observe(tick, None, None, None, None, None, None), actual)
    before = state._sensor.clone()
    with pytest.raises(ValueError, match='finite'):
        state.observe(15, 0, *args, hoop_pos_cyl=[float('nan'), 0., 3.])
    torch.testing.assert_close(state._sensor, before)
