"""Check effective Student curriculum defaults without launching the simulator."""

import ast
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1] / 'source/bb_mimic_nt/bb_mimic_nt/tasks/manager_based/bb_mimic_nt'


@pytest.mark.parametrize('random_init,domain_randomization', [(False, False), (True, False), (False, True), (True, True)])
def test_student_play_randomization_is_off_by_default_and_independently_enabled(random_init, domain_randomization):
    # Exercise the real Play configuration methods with simulator-free config buffers.
    names = ('delay', 'ball_mass', 'pd_gains', 'hand_friction', 'foot_friction',
             'link_mass', 'push', 'global_push', 'object_push')

    class TrainingCfg:
        def __init__(self):
            self.scene = SimpleNamespace(num_envs=32)
            self.commands = SimpleNamespace(motion=SimpleNamespace(
                enable_rsi=True, randomize_clip=True,
                reset_dof_pos_noise=.05, reset_dof_vel_noise=.05))
            self.student_observation = SimpleNamespace(enable_noise=True, randomize_delay=True)
            self.dr = SimpleNamespace(**{name: SimpleNamespace(enabled=True, full_strength=False) for name in names})

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
    assert all(not getattr(cfg.dr, name).enabled for name in names)
    cfg.configure_randomization(random_init=random_init, domain_randomization=domain_randomization)
    expected_noise = .05 if random_init else 0.
    assert cfg.commands.motion.reset_dof_pos_noise == cfg.commands.motion.reset_dof_vel_noise == expected_noise
    for name in names:
        assert getattr(cfg.dr, name).enabled is domain_randomization
        assert getattr(cfg.dr, name).full_strength is domain_randomization
    assert not cfg.commands.motion.enable_rsi
    assert not cfg.commands.motion.randomize_clip
    assert not cfg.student_observation.enable_noise
    assert not cfg.student_observation.randomize_delay


def test_student_curricula_reach_full_strength_early_without_changing_teacher():
    # Compile the actual configuration classes, omitting simulator-dependent imports.
    teacher_names = {'DRPhysicsScheduleCfg', 'DRLatencyScheduleCfg', 'DRPushScheduleCfg',
                     'DomainRandomizationCfg'}
    student_names = {'StudentPhysicsScheduleCfg', 'StudentLatencyScheduleCfg',
                     'StudentPushScheduleCfg', 'StudentDomainRandomizationCfg'}
    namespace = dict(configclass=lambda cls: cls, PPO_MAX_ITERATIONS=5000, PPO_STEPS_PER_ENV=48)
    for file, names in [('bb_mimic_nt_env_cfg.py', teacher_names),
                        ('bb_mimic_nt_student_env_cfg.py', student_names)]:
        nodes = [node for node in ast.parse((ROOT / file).read_text()).body
                 if isinstance(node, ast.ClassDef) and node.name in names]
        exec(compile(ast.Module(body=nodes, type_ignores=[]), str(ROOT / file), 'exec'), namespace)
    teacher = namespace['DomainRandomizationCfg']()
    student = namespace['StudentDomainRandomizationCfg']()
    spec = importlib.util.spec_from_file_location('student_curriculum_events', ROOT / 'mdp/events.py')
    events = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(events)
    assert (teacher.push.start_fraction, teacher.push.end_fraction) == (.15, .8)
    assert (teacher.delay.start_fraction, teacher.delay.end_fraction) == (.3, .7)
    for name, start, end in [('ball_mass', 50, 250), ('pd_gains', 50, 250),
                             ('hand_friction', 50, 250), ('foot_friction', 50, 250),
                             ('link_mass', 50, 250), ('delay', 100, 300), ('push', 100, 350)]:
        schedule = getattr(student, name)
        assert events.curriculum_strength(schedule, start, 1000) == 0.
        assert events.curriculum_strength(schedule, (start + end) / 2, 1000) == pytest.approx(.5)
        assert events.curriculum_strength(schedule, end, 1000) == 1.
        assert events.curriculum_strength(schedule, 999, 1000) == 1.
