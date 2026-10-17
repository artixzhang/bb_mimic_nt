"""CPU checks for terminal accounting, skill-local thresholds, and resume."""

import ast
import math
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from bb_mimic_nt.student_curriculum import StudentTerminationCurriculum, get_student_termination_curriculum


def settings(**overrides):
    return SimpleNamespace(**dict(dict(enabled=True, initial_scale=1.5, minimum_scale=1., maximum_scale=2.,
        minimum_episodes=256, ema_episodes=2048., relax_below=.5, tighten_above=.8,
        relax_factor=1.01, tighten_factor=.99), **overrides))


def controller(count=512, **overrides):
    return StudentTerminationCurriculum(count, ['hold', 'shoot', 'stand_to_hold'], 'cpu', settings(**overrides))


def finish(control, ids, skills, survival, *, completed=False):
    control.begin_episode(ids, skills, torch.full((len(ids),), 100.))
    failed = torch.full((len(ids),), not completed)
    control.finish_episodes(ids, torch.full((len(ids),), survival * 100.), failed,
                            dict(reference_finished=~failed, interaction_tracking_error=failed))


def test_only_completed_episodes_count_and_skill_updates_have_hysteresis():
    control = controller()
    ids = torch.arange(256)
    # Initial reset has no episode and must not generate a failed example.
    control.finish_episodes(ids, torch.zeros(256), torch.ones(256, dtype=torch.bool), {})
    assert control.pending_count.sum() == 0
    finish(control, ids[:255], torch.zeros(255, dtype=torch.long), .2)
    control.update_iteration()
    assert control.scales.tolist() == [1.5] * 3
    finish(control, ids[255:], torch.zeros(1, dtype=torch.long), .2)
    logs = control.update_iteration()
    assert control.scales[0] == pytest.approx(1.515)
    assert control.scales[1] == 1.5
    assert logs['StudentTermination/hold/failure_fraction'] == 1.
    assert logs['StudentTermination/hold/threshold/interaction_tracking_error'] == pytest.approx(1.515 * .35)
    # A second call without episodes must not repeatedly change the curriculum.
    control.update_iteration()
    assert control.scales[0] == pytest.approx(1.515)
    finish(control, ids, torch.ones(256, dtype=torch.long), .999, completed=True)
    control.update_iteration()
    assert control.scales[1] == pytest.approx(1.485)
    assert control.last_episode['survival_fraction'][ids].eq(1).all()


def test_ema_uses_episode_count_and_existing_episode_thresholds_are_frozen():
    control = controller(minimum_episodes=2)
    ids = torch.arange(2)
    finish(control, ids, torch.zeros(2, dtype=torch.long), .2)
    control.update_iteration()
    control.begin_episode(torch.tensor([2]), torch.tensor([0]), torch.tensor([200.]))
    frozen = control.episode_scale[2].clone()
    finish(control, ids, torch.zeros(2, dtype=torch.long), 1., completed=True)
    control.update_iteration()
    expected = .2 + (1 - math.exp(-2 / 2048)) * .8
    assert control.survival_ema[0] == pytest.approx(expected)
    assert control.episode_scale[2] == frozen
    assert control.scales[0] > frozen


def test_disabled_curriculum_still_captures_original_threshold_episode_statistics():
    control = controller(enabled=False)
    ids = torch.tensor([1, 3])
    control.begin_episode(ids, torch.tensor([0, 2]), torch.tensor([800., 200.]))
    control.finish_episodes(ids, torch.tensor([400., 100.]), torch.ones(2, dtype=torch.bool),
                            dict(fall_or_tilt=torch.tensor([True, False])))
    assert control.last_episode['skill_id'][ids].tolist() == [0, 2]
    assert control.last_episode['survival_fraction'][ids].tolist() == [.5, .5]
    assert control.last_episode['scale'][ids].eq(1).all()
    assert control.last_episode_causes['fall_or_tilt'][ids].tolist() == [True, False]
    assert control.update_iteration()['StudentTermination/hold/scale'] == 1.
    assert control.pending_count.sum() == 0


def test_resume_preserves_pending_statistics_and_excludes_simulation_episode_state():
    original, restored = controller(), controller()
    ids = torch.arange(200)
    finish(original, ids, torch.zeros(200, dtype=torch.long), .25)
    restored.load_state_dict(original.state_dict())
    assert not restored.episode_active.any()
    for item in (original, restored):
        finish(item, ids[:56], torch.zeros(56, dtype=torch.long), .25)
    assert original.update_iteration() == restored.update_iteration()
    assert restored.last_cause_fractions['interaction_tracking_error'][0] == 1.
    wrong = restored.state_dict()
    wrong['skill_names'] = ['shoot', 'hold', 'stand_to_hold']
    with pytest.raises(ValueError, match='skill mapping'):
        restored.load_state_dict(wrong)


def test_scale_bounds_and_nonterminal_initial_episode_are_not_recorded_twice():
    control = controller(minimum_episodes=1)
    control.scales[0] = 2.
    finish(control, torch.tensor([0]), torch.tensor([0]), .1)
    control.update_iteration()
    assert control.scales[0] == 2.
    control.finish_episodes(torch.tensor([0]), torch.tensor([10.]), torch.tensor([True]), {})
    assert control.pending_count.sum() == 0
    control.scales[1] = 1.
    finish(control, torch.tensor([1]), torch.tensor([1]), 1., completed=True)
    control.update_iteration()
    assert control.scales[1] == 1.


def test_command_reset_snapshots_old_skill_and_normalizes_rsi_and_end_duration():
    # Execute the real command reset implementation with CPU scene buffers.
    from test_skills import make_command
    term = make_command(count=3)
    term.skill_config = dict(skills=[dict(name=name) for name in ['hold', 'shoot', 'stand_to_hold']])
    term._env.cfg = SimpleNamespace(student_termination=settings(), terminations=SimpleNamespace(
        root_tracking_error=SimpleNamespace(params=dict(maximum_error=1.)),
        dof_tracking_error=SimpleNamespace(params=dict(maximum_rmse=1.)),
        interaction_tracking_error=SimpleNamespace(params=dict(maximum_distance=.35))))
    term._env.num_envs, term._env.device = 3, 'cpu'
    term._env.episode_length_buf = torch.zeros(3, dtype=torch.long)
    causes = dict(reference_finished=torch.zeros(3, dtype=torch.bool),
                  interaction_tracking_error=torch.zeros(3, dtype=torch.bool))
    term._env.termination_manager = SimpleNamespace(dones=torch.zeros(3, dtype=torch.bool),
        terminated=torch.zeros(3, dtype=torch.bool), active_terms=list(causes), get_term=lambda name: causes[name])
    term.fixed_clip_ids[:] = torch.tensor([0, 1, 5])
    term.skill_end_bounds[:] = torch.tensor([[2., 2.], [3., 3.], [4., 4.]])
    torch.manual_seed(11)
    term.reset()
    control = term._env.student_termination_curriculum
    expected = ((term.lengths[term.clip_ids] - 1 - term.frame) / term.fps + term.end_hold_duration) / .01
    torch.testing.assert_close(control.episode_planned_steps, expected)
    assert control.episode_planned_steps[0] == 200.
    assert control.episode_planned_steps[1] < 400.  # RSI shortens the shoot's remaining horizon.
    assert control.pending_count.sum() == 0
    # The same reset now selects a different skill, but records the old one.
    term._env.episode_length_buf[0] = 100
    term._env.termination_manager.dones[0] = True
    term._env.termination_manager.terminated[0] = True
    causes['interaction_tracking_error'][0] = True
    term.fixed_clip_ids[0] = 5
    term.reset([0])
    assert control.last_episode['skill_id'][0] == 0
    assert control.last_episode['survival_fraction'][0] == .5
    assert control.episode_skill[0] == 2
    assert control.pending_count.tolist() == [1, 0, 0]
    # A manual reset of an active, nonterminal episode does not count a failure.
    term._env.episode_length_buf[1] = 75
    term.reset([1])
    assert control.pending_count.tolist() == [1, 0, 0]


def test_student_threshold_wrappers_scale_only_tracking_and_handle_per_hand_broadcast():
    mdp = Path(__file__).resolve().parents[1] / 'source/bb_mimic_nt/bb_mimic_nt/tasks/manager_based/bb_mimic_nt/mdp'
    namespace = dict(torch=torch, math=math, MotionReferenceCommand=object,
                     get_student_termination_curriculum=get_student_termination_curriculum)
    for file in ('terminations.py', 'student_terminations.py'):
        nodes = [node for node in ast.parse((mdp / file).read_text()).body if isinstance(node, ast.FunctionDef)]
        exec(compile(ast.Module(body=nodes, type_ignores=[]), file, 'exec'), namespace)
    q = torch.tensor([1.2, 1.6, .9])[:, None].expand(-1, 29)
    root = torch.tensor([[1.2, 0., 1.], [1.6, 0., 1.], [.9, 0., 1.]])
    motion = SimpleNamespace(robot=SimpleNamespace(data=SimpleNamespace(joint_pos=q, root_link_pos_w=root)),
        joint_ids=list(range(29)), reference=dict(root_pos=torch.tensor([[0., 0., 1.]]).repeat(3, 1),
            dof_pos=torch.zeros_like(q), contact=torch.ones(3, 4)),
        actual_anchor_pos_w=lambda: torch.tensor([[[.4, 0., 0.], [.4, 0., 0.]],
                                                  [[.6, 0., 0.], [.6, 0., 0.]],
                                                  [[.3, 0., 0.], [.3, 0., 0.]]]),
        ball=SimpleNamespace(data=SimpleNamespace(root_pos_w=torch.zeros(3, 3))),
        object_reward_active=torch.ones(3, dtype=torch.bool))
    control = controller(count=3)
    control.begin_episode(torch.arange(3), torch.zeros(3, dtype=torch.long), torch.ones(3))
    env = SimpleNamespace(cfg=SimpleNamespace(student_termination=settings()), student_termination_curriculum=control,
        command_manager=SimpleNamespace(get_term=lambda _: motion), scene=SimpleNamespace(env_origins=torch.zeros(3, 3)))
    for name in ('student_root_tracking_error', 'student_reference_dof_error', 'student_interaction_tracking_error'):
        assert namespace[name](env).tolist() == [False, True, False]
    assert namespace['root_tracking_error'](env, 1.).tolist() == [True, True, False]


def test_controller_can_be_created_during_inference_and_used_afterward():
    with torch.inference_mode():
        control = controller()
        finish(control, torch.arange(256), torch.zeros(256, dtype=torch.long), .2)
    control.update_iteration()
    assert control.scales[0] > 1.5
