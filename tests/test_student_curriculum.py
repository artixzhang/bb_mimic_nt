"""Check effective Student curriculum defaults without launching the simulator."""

import ast
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1] / 'source/bb_mimic_nt/bb_mimic_nt/tasks/manager_based/bb_mimic_nt'


@pytest.fixture
def configuration_classes():
    # Use Isaac Lab's real deep-copy semantics, without importing the simulator task.
    configclass = pytest.importorskip('isaaclab.utils').configclass
    teacher_names = {'DRPhysicsScheduleCfg', 'DRLatencyScheduleCfg', 'DRPushScheduleCfg', 'DRPushVelocityScheduleCfg',
                     'DRGaussianNoiseCfg', 'DRRotationNoiseCfg', 'DRObjectPositionNoiseCfg',
                     'DRInitialStateCfg', 'DomainRandomizationCfg'}
    student_names = {'StudentPhysicsScheduleCfg', 'StudentLatencyScheduleCfg',
                     'StudentPushScheduleCfg', 'StudentDomainRandomizationCfg'}
    namespace = dict(configclass=configclass, PPO_MAX_ITERATIONS=5000, PPO_STEPS_PER_ENV=48,
                     __name__=__name__)
    for file, names in [('bb_mimic_nt_env_cfg.py', teacher_names),
                        ('bb_mimic_nt_student_env_cfg.py', student_names)]:
        nodes = [node for node in ast.parse((ROOT / file).read_text()).body
                 if isinstance(node, ast.ClassDef) and node.name in names]
        exec(compile(ast.Module(body=nodes, type_ignores=[]), str(ROOT / file), 'exec'), namespace)
    return namespace


@pytest.mark.parametrize('random_init,domain_randomization', [(False, False), (True, False), (False, True), (True, True)])
def test_student_play_randomization_is_off_by_default_and_independently_enabled(
        random_init, domain_randomization, configuration_classes):
    # Exercise the real Play configuration methods with simulator-free config buffers.
    names = ('delay', 'ball_mass', 'pd_gains', 'hand_friction', 'foot_friction',
             'link_mass', 'push', 'global_push', 'object_push', 'velocity_push')

    class TrainingCfg:
        def __init__(self):
            self.scene = SimpleNamespace(num_envs=32)
            self.commands = SimpleNamespace(motion=SimpleNamespace(
                enable_rsi=True, randomize_clip=True,
                reset_dof_pos_noise=0., reset_dof_vel_noise=0.))
            self.student_observation = SimpleNamespace(enable_noise=True, randomize_delay=True)
            self.student_termination = SimpleNamespace(enabled=True)
            self.dr = configuration_classes['StudentDomainRandomizationCfg']()

        def __post_init__(self):
            pass

    path = ROOT / 'bb_mimic_nt_student_env_cfg.py'
    node = next(node for node in ast.parse(path.read_text()).body
                if isinstance(node, ast.ClassDef) and node.name == 'G1ShootStudentPlayEnvCfg')
    namespace = dict(configclass=lambda cls: cls, G1ShootStudentEnvCfg=TrainingCfg)
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), 'exec'), namespace)
    cfg = namespace['G1ShootStudentPlayEnvCfg']()
    cfg.__post_init__()
    assert cfg.commands.motion.reset_dof_pos_noise == cfg.commands.motion.reset_dof_vel_noise == 0.
    assert not cfg.dr.init_state.enabled
    assert all(not getattr(cfg.dr, name).enabled for name in names)
    cfg.configure_randomization(random_init=random_init, domain_randomization=domain_randomization)
    assert cfg.commands.motion.reset_dof_pos_noise == cfg.commands.motion.reset_dof_vel_noise == 0.
    assert cfg.dr.init_state.enabled is random_init
    assert all(getattr(cfg.dr.init_state, name).enabled for name in (
        'dof_pos', 'dof_vel', 'root_pos', 'root_rot', 'root_vel', 'root_ang_vel', 'object_pos'))
    for name in names:
        assert getattr(cfg.dr, name).enabled is (domain_randomization and name != 'delay')
        assert getattr(cfg.dr, name).full_strength is domain_randomization
    assert not cfg.commands.motion.enable_rsi
    assert not cfg.commands.motion.randomize_clip
    assert not cfg.student_observation.enable_noise
    assert not cfg.student_observation.randomize_delay
    assert not cfg.student_termination.enabled


def test_student_curricula_reach_full_strength_early_without_changing_teacher(configuration_classes):
    teacher = configuration_classes['DomainRandomizationCfg']()
    student = configuration_classes['StudentDomainRandomizationCfg']()
    spec = importlib.util.spec_from_file_location('student_curriculum_events', ROOT / 'mdp/events.py')
    events = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(events)
    assert (teacher.push.start_fraction, teacher.push.end_fraction) == (.15, .8)
    assert (teacher.velocity_push.start_fraction, teacher.velocity_push.end_fraction) == (0., .5)
    assert (teacher.delay.start_fraction, teacher.delay.end_fraction) == (.3, .7)
    for name, start, end in [('ball_mass', 50, 250), ('pd_gains', 50, 250),
                             ('hand_friction', 50, 250), ('foot_friction', 50, 250),
                             ('link_mass', 50, 250), ('push', 100, 350),
                             ('global_push', 100, 350), ('object_push', 100, 350),
                             ('velocity_push', 100, 350)]:
        schedule = getattr(student, name)
        assert events.curriculum_strength(schedule, start, 1000) == 0.
        assert events.curriculum_strength(schedule, (start + end) / 2, 1000) == pytest.approx(.5)
        assert events.curriculum_strength(schedule, end, 1000) == 1.
        assert events.curriculum_strength(schedule, 999, 1000) == 1.
    assert not student.delay.enabled
    assert student.delay_nominal_steps == 0
    assert student.delay_max_offset_steps == 2
    assert events.curriculum_strength(student.delay, 999, 1000) == 0.


def test_humanx_initialization_defaults_and_independent_switches(configuration_classes):
    teacher = configuration_classes['DomainRandomizationCfg']()
    student = configuration_classes['StudentDomainRandomizationCfg']()
    another_student = configuration_classes['StudentDomainRandomizationCfg']()
    for cfg in (teacher, student):
        init = cfg.init_state
        assert init.enabled
        assert not hasattr(init, 'start_fraction')
        for name, expected in [('dof_pos', .1), ('dof_vel', .01), ('root_pos', .02),
                               ('root_vel', .05), ('root_ang_vel', .02)]:
            assert getattr(init, name).enabled
            assert getattr(init, name).std == expected
        assert init.root_rot.enabled and init.root_rot.max_angle_deg == 10.
        assert init.object_pos.enabled and init.object_pos.half_range_m == .1
        assert cfg.velocity_push_linear_max_m_s == .5
        assert cfg.velocity_push_angular_max_rad_s == .52
        assert cfg.velocity_push_probability == 1.
        for prefix in ('push', 'global_push', 'object_push', 'velocity_push'):
            assert getattr(cfg, f'{prefix}_interval_range_s') == (1., 5.)
    student.init_state.dof_pos.enabled = False
    student.init_state.dof_pos.std = .3
    assert student.init_state.dof_vel.enabled
    assert teacher.init_state.dof_pos.enabled and teacher.init_state.dof_pos.std == .1
    assert another_student.init_state.dof_pos.enabled and another_student.init_state.dof_pos.std == .1
    student.init_state.enabled = False
    student.velocity_push.enabled = False
    assert teacher.init_state.enabled and another_student.init_state.enabled
    assert teacher.velocity_push.enabled and another_student.velocity_push.enabled
    assert student.push.enabled
