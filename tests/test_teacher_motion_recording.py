"""Check raw training compatibility without starting Isaac Sim."""
import ast
import copy
from pathlib import Path
from types import SimpleNamespace

import joblib
import numpy as np
import torch

from bb_mimic_nt.trajectory.processing import validate_motion_batch

ROOT = Path(__file__).resolve().parents[1]


def test_recorded_motion_matches_source_schema(tmp_path):
    path = ROOT / 'scripts/rsl_rl/record_teacher_motion.py'
    functions = [node for node in ast.parse(path.read_text()).body
                 if isinstance(node, ast.FunctionDef) and node.name in ('record_frame', 'assemble_clip')]
    namespace = dict(np=np, copy=copy, contact_graph=lambda env: torch.tensor([[1., 0., 0., 1.]]))
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(path), 'exec'), namespace)
    template = joblib.load(ROOT / 'source/bb_mimic_nt/assets/trajectory/shoot_batch_0928.pkl')[0]
    origin = torch.tensor([[7., 14., 0.]])
    robot = SimpleNamespace(root_link_pos_w=origin + torch.tensor([[1., 2., 3.]]),
                            root_link_quat_w=torch.tensor([[1., 0., 0., 0.]]),
                            joint_pos=torch.arange(29, dtype=torch.float32)[None])
    ball = SimpleNamespace(root_pos_w=origin + torch.tensor([[4., 5., 6.]]),
                           root_quat_w=torch.tensor([[0., 1., 0., 0.]]))
    env = SimpleNamespace(scene=SimpleNamespace(env_origins=origin))
    motion = SimpleNamespace(robot=SimpleNamespace(data=robot), ball=SimpleNamespace(data=ball),
                             reference={'obj_push': torch.tensor([False]), 'push_available': torch.tensor([True]), 'release': torch.tensor([0.])})
    frame = namespace['record_frame'](env, motion, list(reversed(range(29))), [2, 3, 0, 1])
    np.testing.assert_array_equal(frame['root_pos'], [1, 2, 3])
    np.testing.assert_array_equal(frame['obj_pos'], [4, 5, 6])
    np.testing.assert_array_equal(frame['root_rot'], [1, 0, 0, 0])
    np.testing.assert_array_equal(frame['dof'], np.arange(29)[::-1])
    np.testing.assert_array_equal(frame['contact'], [0, 1, 1, 0])
    robot.root_link_pos_w.zero_()  # Recorder snapshots must survive subsequent simulation writes.
    clip = namespace['assemble_clip'](template, [frame] * 3, np.array([1., 2., 3.]))
    assert clip.keys() == template.keys()
    for key, value in template.items():
        assert type(clip[key]) is type(value)
        if isinstance(value, np.ndarray):
            assert clip[key].dtype == value.dtype
        else:
            assert clip[key] == value
    assert clip['root_pos'].shape == (3, 3)
    np.testing.assert_array_equal(clip['root_pos'][0], [1, 2, 3])
    output = tmp_path / 'motion.pkl'
    joblib.dump([clip], output)
    validate_motion_batch(joblib.load(output))
