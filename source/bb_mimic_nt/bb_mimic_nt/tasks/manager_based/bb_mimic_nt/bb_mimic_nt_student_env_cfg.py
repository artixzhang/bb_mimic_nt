"""Independent Student DAgger environment and deterministic playback configuration."""

from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils import configclass

from bb_mimic_nt.action_math import (
    ACTION_OBSERVATION_SCALE, ANGULAR_VELOCITY_OBSERVATION_SCALE,
    JOINT_POSITION_OBSERVATION_SCALE, JOINT_VELOCITY_OBSERVATION_SCALE,
)
from bb_mimic_nt.student import STUDENT_HISTORY_LENGTH
from bb_mimic_nt.training import NUM_ENVS

from . import mdp
from .bb_mimic_nt_env_cfg import (
    ActionsCfg, DomainRandomizationCfg, DRPhysicsScheduleCfg, G1ShootEnvCfg, G1ShootSceneCfg, ObservationsCfg,
)


@configclass
class StudentPhysicsScheduleCfg(DRPhysicsScheduleCfg):
    start_fraction: float = 0.05
    end_fraction: float = 0.25


@configclass
class StudentLatencyScheduleCfg(DRPhysicsScheduleCfg):
    start_fraction: float = 0.10
    end_fraction: float = 0.30


@configclass
class StudentPushScheduleCfg(DRPhysicsScheduleCfg):
    start_fraction: float = 0.10
    end_fraction: float = 0.35


@configclass
class StudentDomainRandomizationCfg(DomainRandomizationCfg):
    """Student-owned timing; physical ranges come from the selected Teacher."""

    delay: StudentLatencyScheduleCfg = StudentLatencyScheduleCfg()
    ball_mass: StudentPhysicsScheduleCfg = StudentPhysicsScheduleCfg()
    pd_gains: StudentPhysicsScheduleCfg = StudentPhysicsScheduleCfg()
    hand_friction: StudentPhysicsScheduleCfg = StudentPhysicsScheduleCfg()
    foot_friction: StudentPhysicsScheduleCfg = StudentPhysicsScheduleCfg()
    link_mass: StudentPhysicsScheduleCfg = StudentPhysicsScheduleCfg()
    push: StudentPushScheduleCfg = StudentPushScheduleCfg()
    global_push: StudentPushScheduleCfg = StudentPushScheduleCfg()
    object_push: StudentPushScheduleCfg = StudentPushScheduleCfg()
    velocity_push: StudentPushScheduleCfg = StudentPushScheduleCfg()


@configclass
class StudentObservationScalesCfg:
    gravity: float = 1.0
    angular_velocity: float = ANGULAR_VELOCITY_OBSERVATION_SCALE
    joint_position: float = JOINT_POSITION_OBSERVATION_SCALE
    joint_velocity: float = JOINT_VELOCITY_OBSERVATION_SCALE
    previous_action: float = ACTION_OBSERVATION_SCALE
    hoop_position: float = 0.25  # Radius and height.
    hoop_theta: float = 2.0  # Absolute scale for theta in radians.


@configclass
class StudentObservationSettingsCfg:
    history_length: int = STUDENT_HISTORY_LENGTH
    delay_nominal_steps: int = 1
    delay_min_steps: int = 0
    delay_max_steps: int = 3
    # Ramp sensor noise and delay spread before Student takes full control.
    enabled: bool = True
    start_fraction: float = 0.0
    end_fraction: float = 0.20
    full_strength: bool = False
    randomize_delay: bool = True
    enable_noise: bool = True
    noise_strength: float = 1.0
    gravity_noise_std: float = 0.015
    angular_velocity_noise_std: float = 0.05
    joint_position_noise_std: float = 0.01
    joint_velocity_noise_std: float = 0.10
    scales: StudentObservationScalesCfg = StudentObservationScalesCfg()


@configclass
class StudentSceneCfg(G1ShootSceneCfg):
    ball_ground_contact = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/bb_ball",
        update_period=0.0,
        filter_prim_paths_expr=["/World/ground/terrain/GroundPlane/CollisionPlane"],
    )


@configclass
class StudentObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        student_observation = ObsTerm(func=mdp.StudentObservation)

        def __post_init__(self) -> None:
            self.concatenate_terms = True
            self.enable_corruption = False

    policy: PolicyCfg = PolicyCfg()
    teacher: ObservationsCfg.PolicyCfg = ObservationsCfg.PolicyCfg()


@configclass
class StudentRewardsCfg:
    ball_flight_metrics = RewTerm(func=mdp.StudentBallFlightMetrics, weight=1.0)


@configclass
class StudentCurriculumCfg:
    domain_randomization = CurrTerm(func=mdp.dr_curriculum)


@configclass
class G1ShootStudentEnvCfg(G1ShootEnvCfg):
    scene: StudentSceneCfg = StudentSceneCfg(num_envs=NUM_ENVS, env_spacing=7.0)
    observations: StudentObservationsCfg = StudentObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    rewards: StudentRewardsCfg = StudentRewardsCfg()
    curriculum: StudentCurriculumCfg = StudentCurriculumCfg()
    student_observation: StudentObservationSettingsCfg = StudentObservationSettingsCfg()
    dr: StudentDomainRandomizationCfg = StudentDomainRandomizationCfg()

    def __post_init__(self) -> None:
        super().__post_init__()
        self.commands.motion.enable_adaptive_speed = False


@configclass
class G1ShootStudentPlayEnvCfg(G1ShootStudentEnvCfg):
    """Fixed-latency, noise-free Student inference environment."""

    def __post_init__(self) -> None:
        super().__post_init__()
        self.scene.num_envs = 1
        self.commands.motion.enable_rsi = False
        self.commands.motion.randomize_clip = False
        self.commands.motion.reset_dof_pos_noise = 0.0
        self.commands.motion.reset_dof_vel_noise = 0.0
        if self.dr.init_state is not None:
            self.dr.init_state.enabled = False
        self.student_observation.enable_noise = False
        self.student_observation.randomize_delay = False
        for name in ("delay", "ball_mass", "pd_gains", "hand_friction", "foot_friction", "link_mass",
                     "push", "global_push", "object_push", "velocity_push"):
            getattr(self.dr, name).enabled = False

    def configure_randomization(self, *, random_init: bool = False, domain_randomization: bool = False) -> None:
        """Opt into HumanX initial-state noise and full-strength physical DR independently."""
        training_cfg = G1ShootStudentEnvCfg()
        if self.dr.init_state is not None:
            self.dr.init_state.enabled = random_init and training_cfg.dr.init_state.enabled
        for name in ("reset_dof_pos_noise", "reset_dof_vel_noise"):
            legacy_noise = getattr(training_cfg.commands.motion, name) if random_init and self.dr.init_state is None else 0.0
            setattr(self.commands.motion, name, legacy_noise)
        for name in ("delay", "ball_mass", "pd_gains", "hand_friction", "foot_friction", "link_mass",
                     "push", "global_push", "object_push", "velocity_push"):
            schedule = getattr(self.dr, name)
            schedule.enabled = domain_randomization and getattr(training_cfg.dr, name).enabled
            schedule.full_strength = domain_randomization
