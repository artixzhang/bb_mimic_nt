"""Student PPO with frozen-Teacher action supervision (legacy task entry point)."""

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg

from bb_mimic_nt.training import STUDENT_BATCH_SIZE, STUDENT_STEPS_PER_ENV


@configclass
class StudentPPOBCAlgorithmCfg(RslRlPpoAlgorithmCfg):
    class_name: str = "PPOWithBC"
    batch_size: int = STUDENT_BATCH_SIZE
    bc_coef: float = 1.0
    # None preserves existing learning_rate/max_grad_norm Hydra overrides for both sides.
    actor_learning_rate: float | None = None
    critic_learning_rate: float | None = None
    actor_max_grad_norm: float | None = None
    critic_max_grad_norm: float | None = None


@configclass
class G1ShootStudentDAggerCfg(RslRlOnPolicyRunnerCfg):
    """Keep the registered task/config name while replacing its training algorithm."""

    class_name: str = "StudentPPOBCRunner"
    num_steps_per_env: int = STUDENT_STEPS_PER_ENV
    max_iterations: int = 1000
    save_interval: int = 50
    experiment_name: str = "g1_shoot_student_ppo_bc_v1"
    obs_groups: dict[str, list[str]] = {"policy": ["policy"], "critic": ["teacher"]}
    clip_actions: float | None = None
    policy: RslRlPpoActorCriticCfg = RslRlPpoActorCriticCfg(
        init_noise_std=0.8,
        noise_std_type="log",
        actor_obs_normalization=False,
        critic_obs_normalization=True,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[1024, 512, 256],
        activation="elu",
    )
    algorithm: StudentPPOBCAlgorithmCfg = StudentPPOBCAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        learning_rate=3.0e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
