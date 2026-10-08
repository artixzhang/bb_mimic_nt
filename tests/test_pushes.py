"""CPU regression tests for independent, additive motion-gated pushes."""
from types import SimpleNamespace

import pytest
import torch

from test_domain_randomization import events


class Asset:
    def __init__(self, count, bodies):
        self.force = torch.zeros(count, bodies, 3)
        self.torque = torch.zeros_like(self.force)
        self.has_external_wrench = False

    def set_external_force_and_torque(self, force, torque, *, body_ids, env_ids, is_global):
        assert is_global
        self.force[env_ids[:, None], body_ids] = force
        self.torque[env_ids[:, None], body_ids] = torque
        self.has_external_wrench = bool(force.any() or torque.any())


def setup_pushes(count=3):
    cfg = SimpleNamespace()
    for name in events.PUSH_TYPES:
        setattr(cfg, name, SimpleNamespace(enabled=True))
        for suffix, value in dict(force_min_n=(1., 1., 1.), force_max_n=(1., 1., 1.),
                                  torque_max_nm=1., duration_s=.2, probability=.8,
                                  link_weights={"torso": 1., "pelvis": 3.}).items():
            setattr(cfg, f"{name}_{suffix}", value)
    motion = SimpleNamespace(
        frame=torch.zeros(count), fps=10., stage=torch.ones(count, dtype=torch.long),
        hold_time=torch.zeros(count), pre_hold_duration=torch.ones(count), post_hold_duration=torch.ones(count),
        clip_ids=torch.zeros(count, dtype=torch.long), lengths=torch.tensor([11]),
        _batch={"push_available": torch.ones(1, 11), "obj_push": torch.ones(1, 11)},
        reference={"push_available": torch.ones(count, dtype=torch.bool), "obj_push": torch.ones(count, dtype=torch.bool)},
    )
    context = SimpleNamespace(robot=Asset(count, 2), ball=Asset(count, 1), motion=motion,
                              _body_ids=lambda names: tuple(range(len(names))), robot_push_bodies=[0, 1])
    env = SimpleNamespace(num_envs=count, device="cpu", step_dt=.01, cfg=SimpleNamespace(dr=cfg), dr_context=context)
    context.pushes = {name: events.PushPulse(env, context, name) for name in events.PUSH_TYPES}
    for pulse in context.pushes.values():
        pulse.reset(torch.arange(count), 1.)
        pulse.selected[:] = True
    return env, context


def test_addition_gating_duration_and_no_retrigger():
    env, ctx = setup_pushes()
    for pulse in ctx.pushes.values():
        pulse.target[:] = 0.
        pulse.force[:] = 1.
        pulse.torque[:] = 2.
        pulse.body[:] = 0
    events.advance_push_pulse(env, None)
    assert torch.all(ctx.robot.force[:, 0] == 2.)
    assert torch.all(ctx.robot.torque[:, 0] == 4.)
    assert torch.all(ctx.ball.force == 1.)
    ctx.motion.reference['push_available'][:] = False
    ctx.motion.reference['obj_push'][:] = False
    events.advance_push_pulse(env, None)
    assert torch.all(ctx.robot.force[:, 0] == 1.)  # Global ignores both gates.
    assert not ctx.ball.force.any()
    for _ in range(18):
        events.advance_push_pulse(env, None)
        assert torch.all(ctx.robot.force[:, 0] == 1.)
    events.advance_push_pulse(env, None)
    assert not ctx.robot.force.any()
    ctx.motion.reference['push_available'][:] = True
    ctx.motion.reference['obj_push'][:] = True
    events.advance_push_pulse(env, None)
    assert not ctx.robot.force.any() and not ctx.ball.force.any()


def test_different_links_and_partial_reset_preserve_other_envs():
    env, ctx = setup_pushes()
    for name, pulse in ctx.pushes.items():
        pulse.target[:] = 0.
        pulse.force[:] = 1.
        pulse.body[:] = int(name == 'global_push')
    events.advance_push_pulse(env, None)
    assert torch.all(ctx.robot.force == 1.)
    ids = torch.tensor([0, 2])
    for pulse in ctx.pushes.values():
        pulse.reset(ids, 1.)
    events._write_push_wrenches(ctx, ids)
    assert not ctx.robot.force[ids].any() and not ctx.ball.force[ids].any()
    assert torch.all(ctx.robot.force[1] == 1.) and torch.all(ctx.ball.force[1] == 1.)
    assert ctx.robot.has_external_wrench and ctx.ball.has_external_wrench


def test_probability_and_link_weights_are_independent_episode_draws():
    torch.manual_seed(42)
    env, ctx = setup_pushes(10000)
    for pulse in ctx.pushes.values():
        pulse.reset(torch.arange(env.num_envs), 1.)
        assert .78 < pulse.selected.float().mean() < .82
    assert .73 < ctx.pushes['push'].body.float().mean() < .77
    joint = ctx.pushes['push'].selected & ctx.pushes['global_push'].selected
    assert .62 < joint.float().mean() < .66
    assert not ctx.pushes['object_push'].body.any()
    pulse = ctx.pushes['global_push']
    pulse.probability = 0.
    pulse.reset(torch.arange(env.num_envs), 1.)
    events.arm_push(env, torch.arange(env.num_envs))
    assert torch.isinf(pulse.target).all()


def test_global_samples_all_stages_and_object_inherits_hold_endpoint_gates():
    torch.manual_seed(3)
    env, ctx = setup_pushes(2000)
    ctx.motion.stage[:] = 0
    ctx.motion._batch['push_available'][:] = 0
    ctx.motion._batch['obj_push'][:] = 0
    ctx.motion._batch['obj_push'][0, 0] = 1
    ctx.motion._batch['obj_push'][0, -1] = 1
    events.arm_push(env, torch.arange(env.num_envs))
    target = ctx.pushes['global_push'].target
    assert (target < 0).any() and ((target >= 0) & (target < 1)).any() and (target >= 1).any()
    assert torch.isinf(ctx.pushes['push'].target).all()
    target = ctx.pushes['object_push'].target
    assert ((target >= -1) & (target < 1.99)).all()
    assert not ((target >= .1) & (target < 1)).any()
    assert (target < 0).any() and (target >= 1).any()


def test_rsi_skips_past_and_padding_and_hold_pulse_stops_at_done():
    env, ctx = setup_pushes(500)
    ctx.motion.frame[:] = 7.
    ctx.motion.pre_hold_duration[:] = 0.
    ctx.motion._batch['obj_push'] = torch.ones(1, 30)  # Padding must not create targets.
    events.arm_push(env, torch.arange(env.num_envs))
    for pulse in ctx.pushes.values():
        assert (pulse.target >= .7).all() and (pulse.target < 1.99).all()
    ctx.motion.stage[:] = 2
    ctx.motion.frame[:] = 10.
    ctx.motion.hold_time[:] = .5
    pulse = ctx.pushes['object_push']
    pulse.target[:] = 1.4
    events.advance_push_pulse(env, None)
    assert ctx.ball.force.any()
    ctx.motion.stage[:] = 3
    events.advance_push_pulse(env, None)
    assert not ctx.ball.force.any()


@pytest.mark.parametrize('key,value', [('push_probability', 1.1), ('push_link_weights', {'torso': 0.}),
                                      ('push_duration_s', .015), ('push_force_min_n', (-1., 0., 0.))])
def test_invalid_push_configuration(key, value):
    env, ctx = setup_pushes()
    setattr(env.cfg.dr, key, value)
    with pytest.raises(ValueError):
        events.PushPulse(env, ctx, 'push')


def test_integer_weights_and_disabled_switch():
    env, ctx = setup_pushes()
    env.cfg.dr.push_link_weights = {"torso": 0, "pelvis": 1}
    pulse = events.PushPulse(env, ctx, 'push')
    ids = torch.arange(env.num_envs)
    pulse.reset(ids, 1.)
    assert (pulse.body == 1).all()
    ctx.pushes['push'] = pulse
    pulse.target[:] = 0.
    pulse.schedule.enabled = False
    events.advance_push_pulse(env, None)
    assert not ctx.robot.force.any()
    pulse.schedule.enabled = True
    events.advance_push_pulse(env, None)
    assert ctx.robot.force.any()
    pulse.schedule.enabled = False
    events.advance_push_pulse(env, None)
    assert not ctx.robot.force.any()
