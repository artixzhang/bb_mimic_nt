# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Record teacher rollouts in the same raw joblib/pkl format used for training."""

import argparse
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--checkpoint', required=True, help='Teacher model checkpoint.')
parser.add_argument('--export_dir', required=True, help='Directory for teacher_motion.pkl.')
parser.add_argument('--end-duration-s', type=float, default=None)
parser.add_argument('--clip_ids', type=int, nargs='+', help='Source clip IDs; default: all clips.')
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--num_envs', type=int, default=32, help='Parallel recording environments (default: 32; 1 for serial).')
AppLauncher.add_app_launcher_args(parser)
args, hydra_args = parser.parse_known_args()
if args.end_duration_s is not None and args.end_duration_s < 0:
    parser.error('Hold durations must be non-negative.')
if args.num_envs < 1:
    parser.error('--num_envs must be positive.')
sys.argv = [sys.argv[0]] + hydra_args
app = AppLauncher(args).app

import copy
from pathlib import Path

import joblib
import numpy as np
import torch
from rsl_rl.runners import OnPolicyRunner

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.utils.assets import retrieve_file_path
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from isaaclab_tasks.utils.hydra import hydra_task_config

import bb_mimic_nt.tasks  # noqa: F401
from bb_mimic_nt.tasks.manager_based.bb_mimic_nt.mdp.contacts import contact_graph
from bb_mimic_nt.trajectory.processing import validate_motion_batch


class RecordingEnv(ManagerBasedRLEnv):
    """Keep the terminal physics state before Isaac Lab automatically resets it."""

    capture_terminal = None

    def _reset_idx(self, env_ids):
        if self.capture_terminal is not None:
            self.capture_terminal(env_ids)
        super()._reset_idx(env_ids)


def record_frame(env, motion, joint_ids, contact_ids):
    """Positions are environment-local; both quaternion fields use WXYZ."""
    origin = env.scene.env_origins[0]
    robot, ball = motion.robot.data, motion.ball.data
    values = {
        'root_pos': robot.root_link_pos_w[0] - origin,
        'root_rot': robot.root_link_quat_w[0],
        'dof': robot.joint_pos[0, joint_ids],
        'obj_pos': ball.root_pos_w[0] - origin,
        'obj_rot': ball.root_quat_w[0],
        'contact': contact_graph(env)[0, contact_ids],
        # These are semantic reference annotations, not contact-derived labels.
        'push_available': motion.reference['push_available'][0],
        'obj_push': motion.reference['obj_push'][0],
        'release': motion.reference['release'][0],
    }
    return {name: value.detach().cpu().numpy().copy() for name, value in values.items()}


def assemble_clip(template, frames, hoop_pos):
    """Preserve every source field, channel order, and dtype."""
    clip = copy.deepcopy(template)
    for name in frames[0]:
        if name in clip:
            clip[name] = np.stack([frame[name] for frame in frames]).astype(np.asarray(template[name]).dtype)
    clip['hoop_pos_w'] = np.asarray(hoop_pos, dtype=np.asarray(template['hoop_pos_w']).dtype)
    return clip


def record_frames(env, motion, joint_ids, contact_ids, env_ids):
    """Snapshot selected environments with one device-to-host copy per field."""
    ids = torch.as_tensor(env_ids, dtype=torch.long, device=env.device)
    origin = env.scene.env_origins[ids]
    robot, ball = motion.robot.data, motion.ball.data
    values = {
        'root_pos': robot.root_link_pos_w[ids] - origin,
        'root_rot': robot.root_link_quat_w[ids],
        'dof': robot.joint_pos[ids].gather(1, joint_ids[ids]),
        'obj_pos': ball.root_pos_w[ids] - origin,
        'obj_rot': ball.root_quat_w[ids],
        'contact': contact_graph(env)[ids].gather(1, contact_ids[ids]),
        'push_available': motion.reference['push_available'][ids],
        'obj_push': motion.reference['obj_push'][ids],
        'release': motion.reference['release'][ids],
    }
    snapshots = {name: value.detach().cpu().numpy().copy() for name, value in values.items()}
    return [{name: value[index] for name, value in snapshots.items()} for index in range(len(env_ids))]


def record_batch(env, policy, motion, clip_ids, source, app):
    """Record one wave; finished slots are ignored until the next wave resets all slots."""
    raw_env = env.unwrapped
    templates = [source[clip_id] for clip_id in clip_ids]
    # The final wave may use fewer slots. Unused environments run the last clip,
    # but their frames and termination flags are never included in the export.
    slot_ids = list(clip_ids) + [clip_ids[-1]] * (raw_env.num_envs - len(clip_ids))
    joint_ids = torch.tensor([
        motion.robot.find_joints(source[clip_id]['dof_names'], preserve_order=True)[0]
        for clip_id in slot_ids
    ], dtype=torch.long, device=raw_env.device)
    contact_ids = torch.tensor([
        [motion.contact_names.index(name) for name in source[clip_id]['contact_names']]
        for clip_id in slot_ids
    ], dtype=torch.long, device=raw_env.device)
    motion.set_clip_ids(torch.tensor(slot_ids, device=raw_env.device))
    env.reset()
    obs = env.get_observations()
    hoop_pos = (motion.hoop_center_pos_w() - raw_env.scene.env_origins).cpu().numpy().copy()
    active = list(range(len(clip_ids)))
    frames = [[frame] for frame in record_frames(raw_env, motion, joint_ids, contact_ids, active)]
    # Reset restores the source pose before a new physics step has produced
    # contact forces. Use source labels only for this initial frame, preserving
    # each clip's contact channel order; subsequent frames use measured contacts.
    for index, template in enumerate(templates):
        frames[index][0]['contact'] = np.asarray(template['contact'][0]).copy()

    def capture_terminal(reset_ids):
        reset_slots = set(reset_ids.cpu().tolist())
        terminal = [index for index in active if index in reset_slots]
        if terminal:
            for index, frame in zip(terminal, record_frames(raw_env, motion, joint_ids, contact_ids, terminal)):
                frames[index].append(frame)

    raw_env.capture_terminal = capture_terminal
    try:
        while active:
            if not app.is_running():
                raise RuntimeError('Simulator closed before recording completed.')
            obs, _, dones, _ = env.step(policy(obs))
            done = dones.cpu().tolist()
            terminated = raw_env.reset_terminated.cpu().tolist()
            for index in active:
                if done[index] and terminated[index]:
                    raise RuntimeError(f'Clip {clip_ids[index]} failed before completion; no batch exported.')
            active = [index for index in active if not done[index]]
            if active:
                for index, frame in zip(active, record_frames(raw_env, motion, joint_ids, contact_ids, active)):
                    frames[index].append(frame)
    finally:
        raw_env.capture_terminal = None
    clips = []
    for index, template in enumerate(templates):
        clips.append(assemble_clip(template, frames[index], hoop_pos[index]))
        print(f'[INFO] Clip {clip_ids[index]}: {len(frames[index])} frames, {(len(frames[index]) - 1) / 100:.2f}s')
    return clips


@hydra_task_config('BbMimicNT-G1-Shoot-Play-v0', 'rsl_rl_cfg_entry_point')
def main(env_cfg, agent_cfg):
    env_cfg.seed = agent_cfg.seed = args.seed
    if args.device is not None:
        env_cfg.sim.device = agent_cfg.device = args.device
    cfg = env_cfg.commands.motion
    cfg.end_duration_s = args.end_duration_s
    cfg.enable_rsi = cfg.randomize_clip = cfg.enable_adaptive_speed = False
    from bb_mimic_nt.skills import load_skill_config
    cfg.skills_config = cfg.skills_config or load_skill_config(cfg.skills_path)
    source = []
    for skill in cfg.skills_config['skills']:
        clips, _ = validate_motion_batch(joblib.load(skill['motion_file']))
        source.extend(clips)
    clip_ids = args.clip_ids if args.clip_ids is not None else list(range(len(source)))
    if not clip_ids:
        raise ValueError('No source clips to record.')
    for clip_id in clip_ids:
        if not 0 <= clip_id < len(source):
            raise ValueError(f'Invalid clip ID: {clip_id}; source has {len(source)} clips.')
    env_cfg.scene.num_envs = min(args.num_envs, len(clip_ids))
    # Training's raw motion contract requires 100 Hz; do not silently relabel samples.
    if not np.isclose(env_cfg.sim.dt * env_cfg.decimation, 0.01):
        raise ValueError('Recording requires the training control rate of 100 Hz.')

    env = RslRlVecEnvWrapper(RecordingEnv(cfg=env_cfg), clip_actions=agent_cfg.clip_actions)
    raw_env = env.unwrapped
    try:
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        runner.load(retrieve_file_path(args.checkpoint))
        policy = runner.get_inference_policy(device=raw_env.device)
        motion = raw_env.command_manager.get_term('motion')
        clips = []
        with torch.inference_mode():
            for start in range(0, len(clip_ids), raw_env.num_envs):
                clips.extend(record_batch(
                    env, policy, motion, clip_ids[start:start + raw_env.num_envs], source, app
                ))
        validate_motion_batch(clips)
        export_dir = Path(args.export_dir).expanduser().resolve()
        export_dir.mkdir(parents=True, exist_ok=True)
        output = export_dir / 'teacher_motion.pkl'
        joblib.dump(clips, output)
        print(f'[INFO] Exported {len(clips)} clips to {output}')
    finally:
        env.close()


if __name__ == '__main__':
    try:
        main()
    finally:
        app.close()
