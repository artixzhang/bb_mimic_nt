"""CPU checks for recurrent independent force and root-link velocity pushes."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

MDP = Path(__file__).resolve().parents[1] / "source/bb_mimic_nt/bb_mimic_nt/tasks/manager_based/bb_mimic_nt/mdp"
spec = importlib.util.spec_from_file_location("push_test_events", MDP / "events.py")
events = importlib.util.module_from_spec(spec)
spec.loader.exec_module(events)


class Asset:
    def __init__(self, count, bodies):
        self.force = torch.zeros(count, bodies, 3)
        self.torque = torch.zeros_like(self.force)
        self.data = SimpleNamespace(root_link_vel_w=torch.ones(count, 6))
        self.velocity_writes = 0
        self.has_external_wrench = False

    def set_external_force_and_torque(self, force, torque, *, body_ids, env_ids, is_global):
        assert is_global
        self.force[env_ids[:, None], body_ids] = force
        self.torque[env_ids[:, None], body_ids] = torque
        self.has_external_wrench = bool(force.any() or torque.any())

    def write_root_link_velocity_to_sim(self, velocity, *, env_ids):
        self.data.root_link_vel_w[env_ids] = velocity
        self.velocity_writes += 1


def setup_pushes(count=3, interval=(.5, .5), *, arm=True):
    cfg = SimpleNamespace(total_iterations=100, steps_per_iteration=10)
    for name in (*events.PUSH_TYPES, "velocity_push"):
        setattr(cfg, name, SimpleNamespace(enabled=True, full_strength=True, start_fraction=.2, end_fraction=.6))
        setattr(cfg, f"{name}_interval_range_s", interval)
        setattr(cfg, f"{name}_probability", 1.)
        if name != "velocity_push":
            for suffix, value in dict(force_min_n=(1., 1., 1.), force_max_n=(1., 1., 1.),
                                      torque_max_nm=1., duration_s=.2,
                                      link_weights={"torso": 1., "pelvis": 3.}).items():
                setattr(cfg, f"{name}_{suffix}", value)
    cfg.velocity_push_linear_max_m_s = .5
    cfg.velocity_push_angular_max_rad_s = .52
    motion = SimpleNamespace(
        frame=torch.zeros(count), fps=10., stage=torch.zeros(count, dtype=torch.long),
        motion_done=torch.zeros(count, dtype=torch.bool), hold_time=torch.zeros(count),
        reference={"push_available": torch.ones(count, dtype=torch.bool), "obj_push": torch.ones(count, dtype=torch.bool)},
    )
    context = SimpleNamespace(robot=Asset(count, 2), ball=Asset(count, 1), motion=motion,
                              _body_ids=lambda names: tuple(range(len(names))), robot_push_bodies=[0, 1], iteration_offset=0)
    env = SimpleNamespace(num_envs=count, device="cpu", step_dt=.01, common_step_counter=0,
                          cfg=SimpleNamespace(dr=cfg), dr_context=context)
    context.pushes = {name: events.PushPulse(env, context, name) for name in events.PUSH_TYPES}
    context.velocity_push = events.VelocityPush(env)
    if arm:
        events.arm_push(env, torch.arange(count))
    return env, context


def advance(env, step, ids=None):
    env.common_step_counter = step
    events.advance_push_pulse(env, ids)


def constant_wrench(pulse, body=0, force=1., torque=2.):
    def sample(env_ids, strength):
        pulse.force[env_ids] = force * strength
        pulse.torque[env_ids] = torque * strength
        pulse.body[env_ids] = body
    pulse.sample_wrench = sample


def test_addition_gating_duration_and_repeated_triggers_in_one_episode():
    env, ctx = setup_pushes()
    for pulse in ctx.pushes.values():
        constant_wrench(pulse)
    advance(env, 49)
    assert not ctx.robot.force.any()
    advance(env, 50)
    assert torch.all(ctx.robot.force[:, 0] == 2.)
    assert torch.all(ctx.robot.torque[:, 0] == 4.)
    assert torch.all(ctx.ball.force == 1.)
    ctx.motion.reference['push_available'][:] = False
    ctx.motion.reference['obj_push'][:] = False
    for step in range(51, 70):
        advance(env, step)
        assert torch.all(ctx.robot.force[:, 0] == 1.)  # Global ignores the gates.
        assert not ctx.ball.force.any()
    advance(env, 70)
    assert not ctx.robot.force.any()
    ctx.motion.reference['push_available'][:] = True
    ctx.motion.reference['obj_push'][:] = True
    advance(env, 99)
    assert not ctx.robot.force.any()
    advance(env, 100)
    assert torch.all(ctx.robot.force[:, 0] == 2.) and torch.all(ctx.ball.force == 1.)
    assert ctx.robot.velocity_writes == 2


def test_gate_waits_for_one_opportunity_and_reschedules_from_actual_trigger():
    env, ctx = setup_pushes()
    ctx.motion.reference['push_available'][:] = False
    ctx.motion.reference['obj_push'][:] = False
    advance(env, 50)
    assert ctx.robot.force.any() and ctx.robot.velocity_writes == 1
    assert ctx.pushes['push'].next_trigger_s.eq(.5).all()
    advance(env, 150)
    assert ctx.pushes['push'].next_trigger_s.eq(.5).all()
    events.arm_push(env, torch.arange(env.num_envs))
    assert ctx.pushes['push'].next_trigger_s.eq(.5).all()
    assert ctx.pushes['object_push'].next_trigger_s.eq(.5).all()
    ctx.motion.reference['push_available'][:] = True
    advance(env, 151)
    assert ctx.pushes['push'].applied.all()
    torch.testing.assert_close(ctx.pushes['push'].next_trigger_s, torch.full((3,), 2.01, dtype=torch.float64))
    assert ctx.pushes['object_push'].next_trigger_s.eq(.5).all()


def test_partial_reset_clears_only_requested_pulses_and_preserves_all_four_timers():
    env, ctx = setup_pushes()
    for name, pulse in ctx.pushes.items():
        constant_wrench(pulse, body=int(name == 'global_push'))
    advance(env, 50)
    assert torch.all(ctx.robot.force == 1.)
    env.common_step_counter = 55
    ids = torch.tensor([0, 2])
    velocity_before = ctx.robot.data.root_link_vel_w.clone()
    events.arm_push(env, ids)
    assert not ctx.robot.force[ids].any() and not ctx.ball.force[ids].any()
    assert torch.all(ctx.robot.force[1] == 1.) and torch.all(ctx.ball.force[1] == 1.)
    assert ctx.robot.has_external_wrench and ctx.ball.has_external_wrench
    for perturbation in (*ctx.pushes.values(), ctx.velocity_push):
        torch.testing.assert_close(perturbation.next_trigger_s, torch.ones(3, dtype=torch.float64))
        assert not perturbation.selected[ids].any()
    # Reset cannot restart a cleared pulse before its existing deadline.
    advance(env, 55, ids)
    assert not ctx.robot.force[ids].any()
    torch.testing.assert_close(ctx.robot.data.root_link_vel_w, velocity_before)


def test_first_arm_samples_only_uninitialized_environments():
    env, ctx = setup_pushes(interval=(1., 5.), arm=False)
    schedules = (*ctx.pushes.values(), ctx.velocity_push)
    events.arm_push(env, torch.tensor([1]))
    initial = [schedule.next_trigger_s.clone() for schedule in schedules]
    for deadline in initial:
        assert torch.isinf(deadline[[0, 2]]).all()
        assert 1. <= deadline[1] <= 5.
    env.common_step_counter = 20
    events.arm_push(env, torch.arange(env.num_envs))
    for schedule, deadline in zip(schedules, initial):
        assert schedule.next_trigger_s[1] == deadline[1]
        assert (schedule.next_trigger_s[[0, 2]] >= 1.2).all()
        assert (schedule.next_trigger_s[[0, 2]] <= 5.2).all()
    before = [schedule.next_trigger_s.clone() for schedule in schedules]
    rng = torch.random.get_rng_state()
    events.arm_push(env, torch.arange(env.num_envs))
    assert torch.equal(torch.random.get_rng_state(), rng)
    for schedule, deadline in zip(schedules, before):
        assert torch.equal(schedule.next_trigger_s, deadline)


def test_short_episodes_do_not_postpone_pushes_and_reset_step_can_trigger():
    env, ctx = setup_pushes(interval=(1., 1.))
    ids = torch.arange(env.num_envs)
    for step in range(1, 221):
        env.common_step_counter = step
        if step % 20 == 0:
            events.arm_push(env, ids)  # Episodes are only 0.2 s long.
            assert not ctx.robot.force.any() and not ctx.ball.force.any()
        advance(env, step)
        assert ctx.robot.velocity_writes == step // 100
        for schedule in (*ctx.pushes.values(), ctx.velocity_push):
            assert schedule.next_trigger_s.eq(step // 100 + 1.).all()
        if step in (100, 200):
            assert ctx.robot.force.any() and ctx.ball.force.any()


def test_done_clears_pulses_and_overdue_deadlines_survive_reset():
    env, ctx = setup_pushes(interval=(1., 1.))
    ids = torch.arange(env.num_envs)
    advance(env, 100)
    ctx.motion.motion_done[:] = True
    advance(env, 101)
    assert not ctx.robot.force.any() and not ctx.ball.force.any()
    advance(env, 250)
    assert ctx.robot.velocity_writes == 1
    events.arm_push(env, ids)
    for schedule in (*ctx.pushes.values(), ctx.velocity_push):
        assert schedule.next_trigger_s.eq(2.).all()
    ctx.motion.motion_done[:] = False
    advance(env, 250)
    assert ctx.robot.force.any() and ctx.ball.force.any()
    assert ctx.robot.velocity_writes == 2
    for schedule in (*ctx.pushes.values(), ctx.velocity_push):
        assert schedule.next_trigger_s.eq(3.5).all()  # One opportunity, no catch-up burst.


def test_probability_links_and_continuous_intervals_are_independent_per_attempt():
    torch.manual_seed(42)
    env, ctx = setup_pushes(10000, interval=(1., 5.))
    perturbations = (*ctx.pushes.values(), ctx.velocity_push)
    for perturbation in perturbations:
        perturbation.probability = .8
        assert (perturbation.next_trigger_s >= 1).all() and (perturbation.next_trigger_s <= 5).all()
        assert 2.95 < perturbation.next_trigger_s.mean() < 3.05
        assert ((perturbation.next_trigger_s / env.step_dt) % 1 != 0).any()
        perturbation.next_trigger_s[:] = 0.
    advance(env, 0)
    first = ctx.pushes['push'].selected.clone()
    for perturbation in perturbations:
        assert .78 < perturbation.selected.float().mean() < .82
    pulse = ctx.pushes['push']
    assert .73 < pulse.body[pulse.selected].float().mean() < .77
    joint = pulse.selected & ctx.pushes['global_push'].selected
    assert .62 < joint.float().mean() < .66
    assert not ctx.pushes['object_push'].body.any()
    advance(env, 500)
    assert .62 < (first & pulse.selected).float().mean() < .66
    assert (pulse.next_trigger_s >= 6).all()
    assert (pulse.next_trigger_s <= 10).all()


def test_probability_miss_reschedules_and_can_hit_next_opportunity():
    env, ctx = setup_pushes()
    pulse = ctx.pushes['push']
    pulse.probability = 0.
    advance(env, 50)
    assert not pulse.applied.any() and pulse.next_trigger_s.eq(1.).all()
    pulse.probability = 1.
    advance(env, 100)
    assert pulse.applied.all() and pulse.next_trigger_s.eq(1.5).all()


def test_real_time_is_independent_of_rsi_reference_speed_hold_and_training_offset():
    env, ctx = setup_pushes()
    ctx.motion.frame[:] = 20000.
    ctx.motion.hold_time[:] = 1000.
    ctx.iteration_offset = 50000
    ctx.motion.stage[:] = 1
    advance(env, 49)
    assert not ctx.robot.force.any() and ctx.robot.velocity_writes == 0
    ctx.motion.frame[:] = -50.
    advance(env, 50)
    assert ctx.robot.force.any() and ctx.ball.force.any() and ctx.robot.velocity_writes == 1
    ctx.motion.motion_done[:] = True
    advance(env, 51)
    assert not ctx.robot.force.any() and not ctx.ball.force.any()
    advance(env, 100)
    assert ctx.robot.velocity_writes == 1


def test_velocity_push_adds_world_root_link_velocities_without_touching_forces():
    torch.manual_seed(9)
    env, ctx = setup_pushes(10000)
    for pulse in ctx.pushes.values():
        pulse.schedule.enabled = False
    ctx.motion.reference['push_available'][:] = False
    ctx.motion.reference['obj_push'][:] = False
    before = ctx.robot.data.root_link_vel_w.clone()
    advance(env, 50)
    delta = ctx.robot.data.root_link_vel_w - before
    limit = torch.tensor([.5] * 3 + [.52] * 3)
    assert (delta.abs() <= limit).all()
    assert (delta.min(0).values < -.95 * limit).all()
    assert (delta.max(0).values > .95 * limit).all()
    assert (delta.mean(0).abs() < .02).all()
    assert not ctx.robot.force.any() and not ctx.ball.force.any()
    before = ctx.robot.data.root_link_vel_w.clone()
    advance(env, 100, torch.tensor([1, 3]))
    assert torch.equal(before[0], ctx.robot.data.root_link_vel_w[0])
    assert not torch.equal(before[1], ctx.robot.data.root_link_vel_w[1])


def test_trigger_uses_current_curriculum_for_force_and_velocity():
    env, ctx = setup_pushes(1000)
    for perturbation in (*ctx.pushes.values(), ctx.velocity_push):
        perturbation.schedule.full_strength = False
    advance(env, 50)  # Before the curriculum starts: consumes an opportunity without a hit.
    assert not ctx.robot.force.any() and ctx.robot.velocity_writes == 0
    advance(env, 400)  # Current strength is 0.5, regardless of episode-reset strength.
    for pulse in ctx.pushes.values():
        torch.testing.assert_close(pulse.force.abs(), torch.full_like(pulse.force, .5))
    velocity_delta = ctx.robot.data.root_link_vel_w - 1
    assert (velocity_delta[:, :3].abs() <= .25).all()
    assert (velocity_delta[:, 3:].abs() <= .26).all()


def test_adjacent_pulses_rewrite_amplitude_and_clear_previous_target_link():
    env, ctx = setup_pushes(interval=(.2, .2))
    ctx.pushes['global_push'].schedule.enabled = False
    pulse = ctx.pushes['push']
    constant_wrench(pulse, body=0, force=1.)
    for step in range(20, 40):
        advance(env, step)
    assert ctx.robot.force[:, 0].eq(1.).all()
    constant_wrench(pulse, body=1, force=3.)
    advance(env, 40)
    assert not ctx.robot.force[:, 0].any()
    assert ctx.robot.force[:, 1].eq(3.).all()


def test_disable_clears_active_pulse_and_velocity_stops():
    env, ctx = setup_pushes()
    advance(env, 50)
    for perturbation in (*ctx.pushes.values(), ctx.velocity_push):
        perturbation.schedule.enabled = False
    advance(env, 51)
    assert not ctx.robot.force.any() and not ctx.ball.force.any()
    advance(env, 100)
    assert ctx.robot.velocity_writes == 1


@pytest.mark.parametrize('key,value', [
    ('push_probability', 1.1), ('push_link_weights', {'torso': 0.}),
    ('push_duration_s', .015), ('push_force_min_n', (-1., 0., 0.)),
    ('push_interval_range_s', (0., 1.)), ('push_interval_range_s', (.1, .2)),
    ('push_interval_range_s', (2., 1.)), ('push_interval_range_s', (1., float('inf'))),
])
def test_invalid_push_configuration(key, value):
    env, ctx = setup_pushes()
    setattr(env.cfg.dr, key, value)
    with pytest.raises(ValueError):
        events.PushPulse(env, ctx, 'push')


def test_integer_link_weights_and_velocity_limits_validation():
    env, ctx = setup_pushes()
    env.cfg.dr.push_link_weights = {"torso": 0, "pelvis": 1}
    pulse = events.PushPulse(env, ctx, 'push')
    pulse.sample_wrench(torch.arange(env.num_envs), 1.)
    assert (pulse.body == 1).all()
    env.cfg.dr.velocity_push_linear_max_m_s = -1.
    with pytest.raises(ValueError, match='velocity'):
        events.VelocityPush(env)
