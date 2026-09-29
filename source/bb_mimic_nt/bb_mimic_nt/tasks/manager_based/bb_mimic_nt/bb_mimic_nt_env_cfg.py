# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Manager-based Unitree G1 basketball imitation environment."""

from __future__ import annotations

import math
from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils import configclass

from bb_mimic_nt.training import TEACHER_HISTORY_LENGTH

from bb_mimic_nt.objects import (
    BB_BALL_CFG,
    BB_BLANK_PLANE_CFG,
    BB_HOOP_CENTER_OFFSET,
    BB_HOOP_FLOATING_CFG,
    BB_HOOP_ROOT_ROTATION,
)
from bb_mimic_nt.robots.unitree_g1_bb import G1_BB_CFG, G1Constants
from bb_mimic_nt.training import PPO_MAX_ITERATIONS, PPO_STEPS_PER_ENV, RSI_DECAY_STEPS

from . import mdp


EXTENSION_ROOT = Path(__file__).resolve().parents[4]
ASSET_ROOT = EXTENSION_ROOT / "assets"
MOTION_SOURCE = ASSET_ROOT / "trajectory" / "shoot_batch_0928.pkl"
MOTION_CACHE = ASSET_ROOT / "trajectory" / "shoot_batch_0928_processed.pt"
ROBOT_URDF = ASSET_ROOT / "robots" / "g1" / "urdf" / "g1_29dof_mode16_bb.urdf"


@configclass
class DRPhysicsScheduleCfg:
    enabled: bool = True
    start_fraction: float = 0.10
    end_fraction: float = 0.60
    full_strength: bool = False


@configclass
class DRLatencyScheduleCfg:
    enabled: bool = True
    start_fraction: float = 0.30
    end_fraction: float = 0.70
    full_strength: bool = False


@configclass
class DRPushScheduleCfg:
    enabled: bool = True
    start_fraction: float = 0.15
    end_fraction: float = 0.80
    full_strength: bool = False

@configclass
class DRDisabledCfg:
    enabled: bool = False


@configclass
class DomainRandomizationCfg:
    total_iterations: int = PPO_MAX_ITERATIONS
    steps_per_iteration: int = PPO_STEPS_PER_ENV
    # Policy steps: fixed nominal at play; train uniformly over
    # max(0, nominal - radius)..nominal + radius as the curriculum expands.
    delay_nominal_steps: int = 0
    delay_max_offset_steps: int = 2
    ball_mass_fraction: float = 0.05
    pd_gain_fraction: float = 0.10
    hand_friction_nominal: float = 0.8
    hand_friction_delta: float = 0.2
    foot_friction_nominal: float = 0.9
    foot_friction_delta: float = 0.3
    link_mass_fraction: float = 0.10
    push_force_min_n: tuple[float, float, float] = (8.0, 8.0, 3.0)
    push_force_max_n: tuple[float, float, float] = (20.0, 20.0, 8.0)
    push_torque_max_nm: float = 5.0
    push_duration_s: float = 0.8

    delay: DRLatencyScheduleCfg = DRLatencyScheduleCfg()
    ball_mass: DRPhysicsScheduleCfg = DRPhysicsScheduleCfg()
    pd_gains: DRPhysicsScheduleCfg = DRPhysicsScheduleCfg()
    hand_friction: DRPhysicsScheduleCfg = DRPhysicsScheduleCfg()
    foot_friction: DRPhysicsScheduleCfg = DRPhysicsScheduleCfg()
    link_mass: DRPhysicsScheduleCfg = DRPhysicsScheduleCfg()
    push: DRPushScheduleCfg = DRPushScheduleCfg()

    # delay: DRScheduleCfg = DRDisabledCfg()
    # ball_mass: DRScheduleCfg = DRDisabledCfg()
    # pd_gains: DRScheduleCfg = DRDisabledCfg()
    # hand_friction: DRScheduleCfg = DRDisabledCfg()
    # foot_friction: DRScheduleCfg = DRDisabledCfg()
    # link_mass: DRScheduleCfg = DRDisabledCfg()
    # push: DRScheduleCfg = DRDisabledCfg()


@configclass
class G1ShootSceneCfg(InteractiveSceneCfg):
    """G1, dynamic basketball, kinematic hoop, and filtered contacts."""

    ground = BB_BLANK_PLANE_CFG
    robot: ArticulationCfg = G1_BB_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    ball: RigidObjectCfg = BB_BALL_CFG
    hoop: RigidObjectCfg = BB_HOOP_FLOATING_CFG

    left_hand_ball_contact = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/left_hand/left_hand",
        update_period=0.0,
        filter_prim_paths_expr=["{ENV_REGEX_NS}/bb_ball"],
    )
    right_hand_ball_contact = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/right_hand/right_hand",
        update_period=0.0,
        filter_prim_paths_expr=["{ENV_REGEX_NS}/bb_ball"],
    )
    left_foot_ground_contact = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/left_ankle_roll_link",
        update_period=0.0,
        filter_prim_paths_expr=["/World/ground/terrain/GroundPlane/CollisionPlane"],
    )
    right_foot_ground_contact = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/right_ankle_roll_link",
        update_period=0.0,
        filter_prim_paths_expr=["/World/ground/terrain/GroundPlane/CollisionPlane"],
    )

    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(color=(1.0, 1.0, 1.0), intensity=2000.0),
    )


@configclass
class CommandsCfg:
    motion = mdp.MotionReferenceCommandCfg(
        source_path=str(MOTION_SOURCE),
        cache_path=str(MOTION_CACHE),
        urdf_path=str(ROBOT_URDF),
        rsi_decay_steps=RSI_DECAY_STEPS,
        hoop_center_offset=BB_HOOP_CENTER_OFFSET,
        hoop_root_quat=BB_HOOP_ROOT_ROTATION,
    )


@configclass
class ActionsCfg:
    joint_pos = mdp.ReferenceResidualJointPositionActionCfg(
        asset_name="robot",
        joint_names=list(G1Constants.G1_ACTUATED_JOINT_NAMES),
        command_name="motion",
    )


@configclass
class ObservationsCfg:
    """Teacher observations, including the explicit object-control phase."""

    @configclass
    class PolicyCfg(ObsGroup):
        # Current state: 175.
        gravity = ObsTerm(func=mdp.gravity_vec_b)
        root_height = ObsTerm(func=mdp.root_pos_z_w)
        root_quaternion = ObsTerm(func=mdp.root_quat_w)
        root_linear_velocity = ObsTerm(func=mdp.root_lin_vel_b)
        root_angular_velocity = ObsTerm(func=mdp.root_ang_vel_b)
        joint_position = ObsTerm(func=mdp.dof_pos)
        joint_velocity = ObsTerm(func=mdp.dof_vel)
        tracked_link_position = ObsTerm(func=mdp.link_pos_b)
        ball_position = ObsTerm(func=mdp.object_pos_b)
        ball_linear_velocity = ObsTerm(func=mdp.object_lin_vel_b)
        hoop_position = ObsTerm(func=mdp.hoop_pos_b)
        previous_action = ObsTerm(func=mdp.previous_action)
        pd_error = ObsTerm(func=mdp.pd_error)

        # Reference deltas, pose targets, velocities, contacts, and release: 186.
        root_position_error = ObsTerm(func=mdp.delta_root_pos_b)
        root_rotation_error = ObsTerm(func=mdp.delta_root_quat_b)
        joint_position_error = ObsTerm(func=mdp.delta_dof_pos)
        tracked_link_position_error = ObsTerm(func=mdp.delta_link_pos_b)
        ball_position_error = ObsTerm(func=mdp.delta_object_pos_w)
        reference_joint_position = ObsTerm(func=mdp.reference_dof_pos)
        reference_tracked_link_position = ObsTerm(func=mdp.reference_link_pos_b)
        reference_ball_position = ObsTerm(func=mdp.reference_object_pos_w)
        reference_root_linear_velocity = ObsTerm(func=mdp.reference_root_lin_vel_b)
        reference_root_angular_velocity = ObsTerm(func=mdp.reference_root_ang_vel_b)
        reference_joint_velocity = ObsTerm(func=mdp.reference_dof_vel)
        reference_ball_linear_velocity = ObsTerm(func=mdp.reference_object_lin_vel_b)
        reference_contact = ObsTerm(func=mdp.reference_contact)
        reference_release = ObsTerm(func=mdp.reference_release)

        gravity_history = ObsTerm(func=mdp.gravity_vec_b, history_length=TEACHER_HISTORY_LENGTH)
        joint_position_history = ObsTerm(func=mdp.dof_pos, history_length=TEACHER_HISTORY_LENGTH)
        action_history = ObsTerm(func=mdp.previous_action, history_length=TEACHER_HISTORY_LENGTH)
        phase = ObsTerm(func=mdp.phase)
        reference_speed = ObsTerm(func=mdp.reference_speed)
        domain_randomization = ObsTerm(func=mdp.dr_privileged_observation)

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class RewardsCfg:
    # Keep exactly one RewardManager term. It owns and logs the full hierarchy.
    unified_mimic = RewTerm(
        func=mdp.UnifiedMimicReward,
        weight=1.0,
        params={"settings": mdp.UnifiedRewardCfg(), "command_name": "motion"},
    )


@configclass
class TerminationsCfg:
    reference_finished = DoneTerm(func=mdp.reference_finished, time_out=True)
    fall_or_tilt = DoneTerm(
        func=mdp.fall_or_tilt,
        params={"minimum_height": 0.45, "maximum_tilt": math.radians(60.0)},
    )
    mechanical_joint_limit = DoneTerm(
        func=mdp.mechanical_joint_limit,
        params={"tolerance": 0.01},
    )
    non_finite_state = DoneTerm(func=mdp.non_finite_state)
    root_tracking_error = DoneTerm(func=mdp.root_tracking_error, params={"maximum_error": 0.75})
    dof_tracking_error = DoneTerm(func=mdp.reference_dof_error, params={"maximum_rmse": 1.0})
    interaction_tracking_error = DoneTerm(
        func=mdp.interaction_tracking_error,
        params={"maximum_distance": 0.08},
    )


@configclass
class CurriculumCfg:
    rsi = CurrTerm(func=mdp.rsi_curriculum, params={"command_name": "motion"})
    domain_randomization = CurrTerm(func=mdp.dr_curriculum)


@configclass
class EventsCfg:
    randomize = EventTerm(func=mdp.reset_domain_randomization, mode="reset")
    push = EventTerm(
        func=mdp.advance_push_pulse,
        mode="interval",
        interval_range_s=(0.01, 0.01),
        is_global_time=True,
    )


@configclass
class G1ShootEnvCfg(ManagerBasedRLEnvCfg):
    scene: G1ShootSceneCfg = G1ShootSceneCfg(num_envs=8192, env_spacing=7.0)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    curriculum: CurriculumCfg = CurriculumCfg()
    events: EventsCfg = EventsCfg()
    dr: DomainRandomizationCfg = DomainRandomizationCfg()

    def __post_init__(self) -> None:
        self.decimation = 2
        # Long enough for the longest clip at 0.5x plus both holds. The motion
        # command remains the authoritative timeout condition.
        self.episode_length_s = 7.0
        self.is_finite_horizon = False
        self.viewer.eye = (7.0, 6.0, 4.0)
        self.viewer.lookat = (1.0, 0.0, 1.5)
        self.sim.dt = 1.0 / 200.0
        self.sim.render_interval = self.decimation
        self.events.push.interval_range_s = (self.decimation * self.sim.dt,) * 2
        # Isaac Sim exposes CCD at scene level; this protects the fast ball.
        self.sim.physx.enable_ccd = True
        self.sim.physx.bounce_threshold_velocity = 0.2
        # Leave headroom for contact peaks during synchronized resets and
        # Student take-over with 8192 environments. These fixed buffers do not
        # automatically grow when more device memory is available.
        self.sim.physx.gpu_max_rigid_patch_count = 2**19
        self.sim.physx.gpu_collision_stack_size = 2**29


@configclass
class G1ShootPlayEnvCfg(G1ShootEnvCfg):
    """Deterministic play configuration."""

    def __post_init__(self) -> None:
        super().__post_init__()
        self.scene.num_envs = 1
        self.observations.policy.enable_corruption = False
        self.commands.motion.enable_rsi = False
        self.commands.motion.randomize_clip = False
        self.commands.motion.enable_adaptive_speed = False
        self.commands.motion.reset_dof_pos_noise = 0.0
        self.commands.motion.reset_dof_vel_noise = 0.0
        for name in ("delay", "ball_mass", "pd_gains", "hand_friction", "foot_friction", "link_mass", "push"):
            getattr(self.dr, name).enabled = False
