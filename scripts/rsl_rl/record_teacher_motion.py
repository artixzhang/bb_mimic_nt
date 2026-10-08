# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Record teacher rollouts in the same raw joblib/pkl format used for training."""

import argparse
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--checkpoint', required=True, help='Teacher model checkpoint.')
parser.add_argument('--export_dir', required=True, help='Directory for teacher_motion.pkl.')
parser.add_argument('--pre_hold', '--pre_hold_s', dest='pre_hold', type=float, default=0.0)
parser.add_argument('--post_hold', '--post_hold_s', dest='post_hold', type=float, default=5.0)
parser.add_argument('--clip_ids', type=int, nargs='+', help='Source clip IDs; default: all clips.')
parser.add_argument('--seed', type=int, default=42)
AppLauncher.add_app_launcher_args(parser)
args, hydra_args = parser.parse_known_args()
if args.pre_hold < 0 or args.post_hold < 0:
    parser.error('Hold durations must be non-negative.')
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
            self.capture_terminal()
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


@hydra_task_config('BbMimicNT-G1-Shoot-Play-v0', 'rsl_rl_cfg_entry_point')
def main(env_cfg, agent_cfg):
    env_cfg.scene.num_envs = 1
    env_cfg.seed = agent_cfg.seed = args.seed
    if args.device is not None:
        env_cfg.sim.device = agent_cfg.device = args.device
    cfg = env_cfg.commands.motion
    cfg.pre_hold_s, cfg.post_hold_s = args.pre_hold, args.post_hold
    cfg.enable_rsi = cfg.randomize_clip = cfg.enable_adaptive_speed = False
    cfg.hold_sampling_weights = None
    source, _ = validate_motion_batch(joblib.load(cfg.source_path))
    clip_ids = args.clip_ids if args.clip_ids is not None else range(len(source))
    for clip_id in clip_ids:
        if not 0 <= clip_id < len(source):
            raise ValueError(f'Invalid clip ID: {clip_id}; source has {len(source)} clips.')
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
            for clip_id in clip_ids:
                template = source[clip_id]
                joint_ids, _ = motion.robot.find_joints(template['dof_names'], preserve_order=True)
                contact_ids = [motion.contact_names.index(name) for name in template['contact_names']]
                motion.set_clip_ids(clip_id)
                env.reset()
                obs = env.get_observations()
                hoop_pos = (motion.hoop_center_pos_w()[0] - raw_env.scene.env_origins[0]).cpu().numpy().copy()
                frames = [record_frame(raw_env, motion, joint_ids, contact_ids)]
                raw_env.capture_terminal = lambda: frames.append(
                    record_frame(raw_env, motion, joint_ids, contact_ids)
                )
                try:
                    while True:
                        if not app.is_running():
                            raise RuntimeError('Simulator closed before recording completed.')
                        obs, _, dones, _ = env.step(policy(obs))
                        if dones[0].item():
                            if raw_env.reset_terminated[0].item():
                                raise RuntimeError(f'Clip {clip_id} failed before completion; no batch exported.')
                            break
                        frames.append(record_frame(raw_env, motion, joint_ids, contact_ids))
                finally:
                    raw_env.capture_terminal = None
                clips.append(assemble_clip(template, frames, hoop_pos))
                print(f'[INFO] Clip {clip_id}: {len(frames)} frames, {(len(frames) - 1) / 100:.2f}s')
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
