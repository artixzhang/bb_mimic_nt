"""Hold exposure budgets and command transitions without launching Isaac Sim."""
import ast
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from bb_mimic_nt.core import hold_sampling_probability


def command_methods():
    path = Path(__file__).resolve().parents[1] / 'source/bb_mimic_nt/bb_mimic_nt/tasks/manager_based/bb_mimic_nt/mdp/commands.py'
    cls = next(n for n in ast.parse(path.read_text()).body if isinstance(n, ast.ClassDef) and n.name == 'MotionReferenceCommand')
    methods = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in ('_sample_hold_durations', '_update_command')]
    namespace = dict(torch=torch, hold_sampling_probability=hold_sampling_probability,
                     STAGE_PRE_HOLD=0, STAGE_ACTIVE=1, STAGE_POST_HOLD=2, STAGE_DONE=3)
    exec(compile(ast.Module(body=methods, type_ignores=[]), str(path), 'exec'), namespace)
    return namespace


@pytest.mark.parametrize('duration', [1., 5., 30.])
def test_long_holds_preserve_active_exposure(duration):
    active = torch.tensor([1.65, 2.4])
    probability = hold_sampling_probability(active, duration, .15)
    torch.testing.assert_close(probability * duration, .15 * active)
    assert torch.all(1 / (1 + 2 * probability * duration / active) >= .769)


@pytest.mark.parametrize('duration,weight', [(-1., .1), (float('nan'), .1), (1., -1.), (1., float('inf'))])
def test_invalid_sampling_parameters(duration, weight):
    with pytest.raises(ValueError):
        hold_sampling_probability(torch.ones(1), duration, weight)


def test_zero_and_short_holds():
    active = torch.ones(2)
    assert not hold_sampling_probability(active, 0., .2).any()
    assert not hold_sampling_probability(active, 2., 0.).any()
    torch.testing.assert_close(hold_sampling_probability(active, .1, .2), torch.ones(2))


def test_actual_reset_sampler_preserves_full_segments_and_rsi():
    torch.manual_seed(123)
    count = 10000
    cmd = SimpleNamespace(cfg=SimpleNamespace(pre_hold_s=10., post_hold_s=5., hold_sampling_weights=(.2, .1)),
        lengths=torch.tensor([201]), clip_ids=torch.zeros(count, dtype=torch.long),
        frame=torch.zeros(count), fps=100., device='cpu',
        pre_hold_duration=torch.zeros(count), post_hold_duration=torch.zeros(count))
    sample = command_methods()['_sample_hold_durations']
    ids = torch.arange(count)
    sample(cmd, ids, torch.zeros(count, dtype=torch.bool))
    assert set(cmd.pre_hold_duration.tolist()) == {0., 10.}
    assert set(cmd.post_hold_duration.tolist()) == {0., 5.}
    assert cmd.pre_hold_duration.mean().item() == pytest.approx(.4, abs=.07)
    assert cmd.post_hold_duration.mean().item() == pytest.approx(.2, abs=.04)
    cmd.cfg.hold_sampling_weights = None
    sample(cmd, ids, torch.ones(count, dtype=torch.bool))
    assert not cmd.pre_hold_duration.any()
    assert torch.all(cmd.post_hold_duration == 5.)
    sample(cmd, ids[:2], torch.zeros(2, dtype=torch.bool))
    assert torch.all(cmd.pre_hold_duration[:2] == 10.)


def test_independent_hold_durations_phase_transition_and_completion():
    cmd = SimpleNamespace(_env=SimpleNamespace(step_dt=.01),
        cfg=SimpleNamespace(adaptive_initial_error=.5, speed_update_interval_s=1., enable_adaptive_speed=False),
        stage=torch.tensor([0, 0, 2, 2]), stage_duration=torch.zeros(4, 3),
        hold_time=torch.tensor([.01, .01, 0., 0.]),
        pre_hold_duration=torch.tensor([.02, 1., 0., 0.]),
        post_hold_duration=torch.tensor([0., 0., 0., 1.]), speed=torch.zeros(4),
        tracking_error_ema=torch.zeros(4), speed_update_elapsed=torch.zeros(4),
        _adaptive_error_alpha=.1, _tracking_error=lambda: torch.zeros(4),
        object_control_active=torch.ones(4, dtype=torch.bool), frame=torch.zeros(4),
        lengths=torch.tensor([101]), clip_ids=torch.zeros(4, dtype=torch.long), fps=100.,
        _refresh_reference=lambda: None)
    command_methods()['_update_command'](cmd)
    assert cmd.stage.tolist() == [1, 0, 3, 2]
    assert cmd.frame.tolist() == [1., 0., 0., 0.]

