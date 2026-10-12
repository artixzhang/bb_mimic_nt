"""Deployment sensor parity and shot-clock behavior against simulation semantics."""

import pytest
import torch

from bb_mimic_nt.student_deployment import StudentObservationState
from test_student_observations import make_sensor


@pytest.mark.parametrize('delay', [0, 1, 3])
def test_robot_observation_builder_matches_simulation(delay):
    sim, env, data, action = make_sensor(nominal=delay, maximum=3)
    sim.delay_steps[:] = delay
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


def test_skill_change_requires_episode_reset():
    sim, env, data, action = make_sensor()
    metadata = dict(command_dimension=3, history_length=5, default_observation_delay_steps=0,
                    nominal_joint_position=[0.] * 29, safe_lower_joint_position=[-2.] * 29,
                    safe_upper_joint_position=[2.] * 29,
                    observation_settings=dict(scales=vars(env.cfg.student_observation.scales)))
    state = StudentObservationState(metadata)
    state.reset([0, 0, -1], [0]*3, [0]*29, [0]*29, [0]*29, [0]*3)
    observation = state.observe(0, 2, [0, 0, -1], [0]*3, [0]*29, [0]*29, [0]*29)
    assert observation[0, :3].tolist() == [0., 0., 1.]
    with pytest.raises(ValueError, match='Reset'):
        state.observe(1, 1, None, None, None, None, None)
