"""RSL-RL PPO configuration for the G1 jump-shot Teacher."""

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg

from bb_mimic_nt.training import PPO_MAX_ITERATIONS, PPO_STEPS_PER_ENV


@configclass
class G1ShootPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = PPO_STEPS_PER_ENV
    max_iterations = PPO_MAX_ITERATIONS
    save_interval = 100
    experiment_name = "g1_shoot_teacher_v2"
    seed = 42
    empirical_normalization = False
    clip_actions = None
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=0.25,
        actor_obs_normalization=True,
        critic_obs_normalization=True,
        actor_hidden_dims=[1024, 512, 256],
        critic_hidden_dims=[1024, 512, 256],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.001,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=3.0e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
