"""Check effective Student curriculum defaults without launching the simulator."""

import ast
import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1] / 'source/bb_mimic_nt/bb_mimic_nt/tasks/manager_based/bb_mimic_nt'


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
