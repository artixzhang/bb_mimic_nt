"""Exercise the actual sensor builder on CPU without starting Isaac Sim."""

import ast
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from bb_mimic_nt.student import student_delay_bounds, student_history, student_observation_fields

MDP = Path(__file__).resolve().parents[1] / 'source/bb_mimic_nt/bb_mimic_nt/tasks/manager_based/bb_mimic_nt/mdp'
spec = importlib.util.spec_from_file_location('sensor_test_events', MDP / 'events.py')
events = importlib.util.module_from_spec(spec)
spec.loader.exec_module(events)


class TermBase:
    def __init__(self, cfg, env):
        self._env = env
        self.device = env.device
        self.num_envs = env.num_envs


def make_sensor(nominal=1, maximum=3, count=2, history=5):
    source = ast.parse((MDP / 'student_observations.py').read_text())
    node = next(node for node in source.body if isinstance(node, ast.ClassDef))
    namespace = dict(torch=torch, ManagerTermBase=TermBase,
                     student_delay_bounds=student_delay_bounds, student_history=student_history,
                     student_observation_fields=student_observation_fields,
                     curriculum_strength=events.curriculum_strength,
                     quat_apply_inverse=lambda quat, vector: vector.clone())
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(MDP / 'student_observations.py'), 'exec'), namespace)
    settings = SimpleNamespace(history_length=history, delay_nominal_steps=nominal,
                               delay_min_steps=0, delay_max_steps=maximum,
                               randomize_delay=True, enable_noise=False, noise_strength=1.,
                               gravity_noise_std=.015, angular_velocity_noise_std=.05,
                               joint_position_noise_std=.01, joint_velocity_noise_std=.1,
                               enabled=True, start_fraction=0., end_fraction=.3, full_strength=True,
                               scales=SimpleNamespace(phase=1., gravity=1., angular_velocity=1.,
                                                      joint_position=1., joint_velocity=1.,
                                                      applied_action=1., hoop_position=1.))
    data = SimpleNamespace(projected_gravity_b=torch.zeros(count, 3),
                           root_link_quat_w=torch.tensor([1., 0., 0., 0.]).repeat(count, 1),
                           root_link_ang_vel_w=torch.zeros(count, 3), root_link_pos_w=torch.zeros(count, 3),
                           joint_pos=torch.zeros(count, 29), joint_vel=torch.zeros(count, 29))
    motion = SimpleNamespace(robot=SimpleNamespace(data=data), joint_ids=list(range(29)),
                             normalized_phase=torch.zeros(count), hoop_center_pos_w=lambda: torch.zeros(count, 3))
    action = SimpleNamespace(nominal=torch.zeros(count, 29), actual_action_offset=torch.zeros(count, 29))
    env = SimpleNamespace(num_envs=count, device='cpu', common_step_counter=0,
                          cfg=SimpleNamespace(student_observation=settings,
                                              dr=SimpleNamespace(steps_per_iteration=1, total_iterations=100)),
                          command_manager=SimpleNamespace(get_term=lambda name: motion),
                          action_manager=SimpleNamespace(get_term=lambda name: action))
    return namespace['StudentObservation'](None, env), env, data, action


def test_five_sensor_frames_and_actions_are_aligned_and_reset_is_isolated():
    sensor, env, data, action = make_sensor(nominal=0)
    sensor.delay_steps[:] = torch.tensor([0, 3])
    for step in range(11):
        env.common_step_counter = step
        data.joint_pos[:] = step
        action.actual_action_offset[:] = step * 2
        output = sensor(env)
    fields = dict(student_observation_fields(5))
    assert output.shape == (2, 469)
    joint_start = 1 + fields['projected_gravity_history'] + fields['pelvis_angular_velocity_history']
    joints = output[:, joint_start:joint_start + 145].reshape(2, 5, 29)
    assert joints[:, :, 0].tolist() == [[6., 7., 8., 9., 10.], [3., 4., 5., 6., 7.]]
    actions = output[:, -148:-3].reshape(2, 5, 29)
    assert actions[:, :, 0].tolist() == [[12., 14., 16., 18., 20.]] * 2
    # Reading twice within a policy step must not advance the history.
    torch.testing.assert_close(sensor(env), output)
    sensor.reset([0])
    data.joint_pos[0] = 99
    after = sensor(env)
    assert after[0, joint_start:joint_start + 145].eq(99).all()
    torch.testing.assert_close(after[1], output[1])


@pytest.mark.parametrize('nominal, maximum', [(1, 3), (0, 3), (0, 0)])
def test_delay_sampling_includes_valid_bounds_and_zero_encoding_is_finite(nominal, maximum):
    sensor, env, _, _ = make_sensor(nominal=nominal, maximum=maximum, count=1024)
    torch.manual_seed(4)
    sensor.reset()
    assert sensor.delay_steps.min() == 0
    assert sensor.delay_steps.max() == maximum
    assert torch.isfinite(sensor.delay_encoding).all()
    assert sensor(env).shape == (1024, 469)


def test_sensor_curriculum_starts_clean_at_nominal_delay_and_reaches_full_range():
    sensor, env, _, _ = make_sensor(count=1024)
    settings = env.cfg.student_observation
    settings.full_strength = False
    settings.enable_noise = True
    sensor.reset()
    assert sensor.delay_steps.eq(1).all()
    assert sensor(env)[:, 1:31].eq(0).all()
    env.common_step_counter = 30
    sensor.reset()
    assert sensor.delay_steps.min() == 0 and sensor.delay_steps.max() == 3
    assert sensor(env)[:, 1:31].abs().sum() > 0


def test_legacy_three_frame_builder_dimension():
    sensor, env, _, _ = make_sensor(nominal=2, maximum=4, history=3)
    assert sensor(env).shape == (2, 283)
