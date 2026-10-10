"""Check raw training compatibility without starting Isaac Sim."""
import ast
import copy
from pathlib import Path
from types import SimpleNamespace

import joblib
import numpy as np
import pytest
import torch

from bb_mimic_nt.trajectory.processing import validate_motion_batch

ROOT = Path(__file__).resolve().parents[1]


def test_recorded_motion_matches_source_schema(tmp_path):
    path = ROOT / 'scripts/rsl_rl/record_teacher_motion.py'
    functions = [node for node in ast.parse(path.read_text()).body
                 if isinstance(node, ast.FunctionDef) and node.name in ('record_frame', 'assemble_clip')]
    namespace = dict(np=np, copy=copy, contact_graph=lambda env: torch.tensor([[1., 0., 0., 1.]]))
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(path), 'exec'), namespace)
    template = joblib.load(ROOT / 'source/bb_mimic_nt/assets/trajectory/batch_setshot_1009.pkl')[0]
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


def recording_helpers():
    path = ROOT / 'scripts/rsl_rl/record_teacher_motion.py'
    functions = [node for node in ast.parse(path.read_text()).body
                 if isinstance(node, ast.FunctionDef) and node.name != 'main']
    namespace = dict(np=np, copy=copy, torch=torch, contact_graph=lambda env: env.contacts)
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(path), 'exec'), namespace)
    return namespace


class FakeRecordingEnv:
    """Auto-reset immediately overwrites state, as in Isaac Lab."""

    def __init__(self, source, num_envs, failure_slot=None):
        self.unwrapped = self
        self.num_envs = num_envs
        self.device = 'cpu'
        self.source = source
        self.failure_slot = failure_slot
        self.capture_terminal = None
        self.scene = SimpleNamespace(env_origins=torch.arange(num_envs * 3).reshape(num_envs, 3).float() * 7)
        robot = SimpleNamespace(root_link_pos_w=torch.zeros(num_envs, 3),
                                root_link_quat_w=torch.tensor([[1., 0., 0., 0.]]).repeat(num_envs, 1),
                                joint_pos=torch.zeros(num_envs, 29))
        ball = SimpleNamespace(root_pos_w=torch.zeros(num_envs, 3),
                               root_quat_w=torch.tensor([[1., 0., 0., 0.]]).repeat(num_envs, 1))
        self.motion = SimpleNamespace(
            robot=SimpleNamespace(data=robot, find_joints=lambda names, preserve_order: (
                [source[0]['dof_names'].index(name) for name in names], names)),
            ball=SimpleNamespace(data=ball), contact_names=list(source[0]['contact_names']),
            reference={name: torch.zeros(num_envs) for name in ('release', 'push_available', 'obj_push')},
            set_clip_ids=self.set_clip_ids,
            hoop_center_pos_w=lambda: self.scene.env_origins + torch.tensor([1., 2., 3.]),
        )
        self.contacts = torch.zeros(num_envs, 4)

    def set_clip_ids(self, ids):
        self.clip_ids = ids.tolist()

    def write_state(self):
        value = self.age.float() + torch.tensor(self.clip_ids).float() * 100
        self.motion.robot.data.root_link_pos_w[:] = self.scene.env_origins + value[:, None]
        self.motion.ball.data.root_pos_w[:] = self.scene.env_origins + value[:, None] + 10
        self.motion.robot.data.joint_pos[:] = value[:, None] + torch.arange(29)
        self.contacts[:] = (value[:, None] + torch.arange(4)) % 2
        for field in self.motion.reference.values():
            field[:] = self.age % 2

    def reset(self):
        self.age = torch.zeros(self.num_envs, dtype=torch.long)
        self.reset_terminated = torch.zeros(self.num_envs, dtype=torch.bool)
        self.write_state()

    def get_observations(self):
        return self.age.clone()

    def step(self, actions):
        self.age += 1
        self.write_state()
        dones = self.age >= torch.tensor([clip_id + 2 for clip_id in self.clip_ids])
        self.reset_terminated.zero_()
        if self.failure_slot is not None:
            dones[self.failure_slot] = True
            self.reset_terminated[self.failure_slot] = True
        reset_ids = dones.nonzero().flatten()
        if len(reset_ids):
            if self.capture_terminal is not None:
                self.capture_terminal(reset_ids)
            self.age[reset_ids] = 0
            self.write_state()
        return self.get_observations(), None, dones, None


def recording_source():
    template = joblib.load(ROOT / 'source/bb_mimic_nt/assets/trajectory/batch_setshot_1009.pkl')[0]
    source = [copy.deepcopy(template) for _ in range(3)]
    source[1]['dof_names'] = list(reversed(source[1]['dof_names']))
    source[1]['contact_names'] = list(reversed(source[1]['contact_names']))
    return source


def test_parallel_recording_equals_serial_with_different_lengths_and_order(tmp_path):
    helpers = recording_helpers()
    source = recording_source()
    # Distinct labels in each template's own order must override reset forces.
    for clip, labels in zip(source, ([1, 1, 1, 1], [0, 1, 1, 0], [1, 0, 0, 1])):
        clip['contact'][0] = labels
    ids = [2, 0, 1, 2]  # Out of order, duplicate, and a partially filled final wave.
    app = SimpleNamespace(is_running=lambda: True)
    parallel = FakeRecordingEnv(source, 3)
    serial = FakeRecordingEnv(source, 1)
    actual, expected = [], []
    for start in range(0, len(ids), 3):
        actual.extend(helpers['record_batch'](parallel, lambda obs: obs, parallel.motion,
                                              ids[start:start + 3], source, app))
    for clip_id in ids:
        expected.extend(helpers['record_batch'](serial, lambda obs: obs, serial.motion, [clip_id], source, app))
    assert [len(clip['root_pos']) for clip in actual] == [5, 3, 4, 5]
    for clip, baseline in zip(actual, expected):
        assert clip.keys() == baseline.keys()
        for name, value in baseline.items():
            if isinstance(value, np.ndarray):
                assert clip[name].dtype == value.dtype
                np.testing.assert_array_equal(clip[name], value)
            else:
                assert clip[name] == value
    # Initial and terminal physics frames survive reset, without a reset-frame tail.
    for clip_id, clip in zip(ids, actual):
        np.testing.assert_array_equal(clip['contact'][0], source[clip_id]['contact'][0])
        channel_ids = [source[0]['contact_names'].index(name) for name in source[clip_id]['contact_names']]
        for age in range(1, len(clip['contact'])):
            measured = (clip_id * 100 + age + np.arange(4)) % 2
            np.testing.assert_array_equal(clip['contact'][age], measured[channel_ids])
    np.testing.assert_array_equal(actual[0]['root_pos'][:, 0], [200, 201, 202, 203, 204])
    np.testing.assert_array_equal(actual[2]['dof'][0], np.arange(29)[::-1] + 100)
    np.testing.assert_array_equal(actual[2]['contact'][0], source[1]['contact'][0])
    assert parallel.capture_terminal is None
    output = tmp_path / 'teacher_motion.pkl'
    joblib.dump(actual, output)
    validate_motion_batch(joblib.load(output))


def test_parallel_snapshot_matches_single_frame_and_survives_writes():
    helpers = recording_helpers()
    env = FakeRecordingEnv(recording_source(), 1)
    env.set_clip_ids(torch.tensor([1]))
    env.reset()
    joint_ids = list(reversed(range(29)))
    contact_ids = [2, 3, 0, 1]
    baseline = helpers['record_frame'](env, env.motion, joint_ids, contact_ids)
    frame = helpers['record_frames'](env, env.motion, torch.tensor([joint_ids]),
                                    torch.tensor([contact_ids]), [0])[0]
    env.age += 1
    env.write_state()
    for name in baseline:
        np.testing.assert_array_equal(frame[name], baseline[name])


@pytest.mark.parametrize('failure_slot,raises', [(0, True), (1, True), (2, False)])
def test_parallel_recording_failure_and_unused_slots(failure_slot, raises):
    helpers = recording_helpers()
    source = recording_source()
    env = FakeRecordingEnv(source, 3, failure_slot=failure_slot)

    def record():
        return helpers['record_batch'](env, lambda obs: obs, env.motion, [0, 1], source,
                                       SimpleNamespace(is_running=lambda: True))

    if raises:
        with pytest.raises(RuntimeError, match=f'Clip {failure_slot} failed before completion'):
            record()
    else:
        assert len(record()) == 2
    assert env.capture_terminal is None


def test_parallel_recording_interruption_clears_capture():
    helpers = recording_helpers()
    source = recording_source()
    env = FakeRecordingEnv(source, 2)
    with pytest.raises(RuntimeError, match='Simulator closed'):
        helpers['record_batch'](env, lambda obs: obs, env.motion, [0, 1], source,
                               SimpleNamespace(is_running=lambda: False))
    assert env.capture_terminal is None
