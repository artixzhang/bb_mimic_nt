"""Deployment sensor parity and shot-clock behavior against simulation semantics."""

import pytest
import torch

from bb_mimic_nt.student_deployment import StudentObservationState, StudentPhaseClock
from test_student_observations import make_sensor


@pytest.mark.parametrize('delay', [0, 1, 3])
def test_robot_observation_builder_matches_simulation(delay):
    sim, env, data, action = make_sensor(nominal=delay, maximum=3)
    sim.delay_steps[:] = delay
    data.projected_gravity_b[:, 2] = -1
    meta = dict(history_length=5, default_observation_delay_steps=delay,
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
        actual = real.observe(tick, 0., [0, 0, -1], [0, 0, 0], data.joint_pos[0],
                              data.joint_vel[0], action.actual_action_offset[0])
        torch.testing.assert_close(actual, reference)
        torch.testing.assert_close(real.observe(tick, 0., None, None, None, None, None), actual)
    with pytest.raises(ValueError, match='skipped'):
        real.observe(14, 0., None, None, None, None, None)
    with pytest.raises(ValueError, match='finite'):
        real.action_target(torch.full((29,), float('nan')))
    assert real.action_target(torch.full((29,), 200.)).eq(100).all()


def test_ready_phase_zero_does_not_reset_history_and_shot_finishes_on_time():
    clock = StudentPhaseClock(dict(frame_count=166, reference_fps=100.), 100., 1.)
    for _ in range(500):
        clock.after_policy_tick()
    assert clock.phase == 0 and clock.stage == 'READY'
    clock.trigger()
    assert clock.phase == 0
    for step in range(165):
        assert clock.phase == pytest.approx(step / 165)
        clock.after_policy_tick()
    assert clock.phase == 1 and clock.stage == 'POST_HOLD'
    for _ in range(99):
        clock.after_policy_tick()
    assert clock.stage == 'POST_HOLD'
    clock.after_policy_tick()
    assert clock.stage == 'DONE'
