"""Independent Student DAgger environment and deterministic playback configuration."""

from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils import configclass

from bb_mimic_nt.robots.unitree_g1_bb import G1Constants

from . import mdp
from .bb_mimic_nt_env_cfg import G1ShootEnvCfg, G1ShootSceneCfg, ObservationsCfg


@configclass
class StudentObservationScalesCfg:
    phase: float = 1.0
    gravity: float = 1.0
    angular_velocity: float = 0.25
    joint_position: float = 1.0
    joint_velocity: float = 0.05
    applied_action: float = 1.0
    hoop_position: float = 0.25


@configclass
class StudentObservationSettingsCfg:
    delay_nominal_steps: int = 2
    delay_max_offset_steps: int = 2
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
class StudentActionsCfg:
    joint_pos = mdp.StudentNominalJointPositionActionCfg(
        asset_name="robot",
        joint_names=list(G1Constants.G1_ACTUATED_JOINT_NAMES),
        command_name="motion",
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
    scene: StudentSceneCfg = StudentSceneCfg(num_envs=4096, env_spacing=5.0)
    observations: StudentObservationsCfg = StudentObservationsCfg()
    actions: StudentActionsCfg = StudentActionsCfg()
    rewards: StudentRewardsCfg = StudentRewardsCfg()
    curriculum: StudentCurriculumCfg = StudentCurriculumCfg()
    student_observation: StudentObservationSettingsCfg = StudentObservationSettingsCfg()

    def __post_init__(self) -> None:
        super().__post_init__()
        self.commands.motion.enable_rsi = False
        self.commands.motion.enable_adaptive_speed = False
        for name in ("delay", "ball_mass", "pd_gains", "hand_friction", "foot_friction", "link_mass", "push"):
            getattr(self.dr, name).full_strength = True


@configclass
class G1ShootStudentPlayEnvCfg(G1ShootStudentEnvCfg):
    """Fixed-latency, noise-free Student inference environment."""

    def __post_init__(self) -> None:
        super().__post_init__()
        self.scene.num_envs = 1
        self.commands.motion.enable_rsi = False
        self.commands.motion.randomize_clip = False
        self.student_observation.enable_noise = False
        self.student_observation.randomize_delay = False
        for name in ("delay", "ball_mass", "pd_gains", "hand_friction", "foot_friction", "link_mass", "push"):
            getattr(self.dr, name).enabled = False
