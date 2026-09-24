"""Official RSL-RL network/algorithm configuration plus DAgger loop settings."""

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import (
    RslRlDistillationAlgorithmCfg,
    RslRlDistillationRunnerCfg,
    RslRlDistillationStudentTeacherCfg,
)


@configclass
class G1ShootStudentDAggerCfg(RslRlDistillationRunnerCfg):
    num_steps_per_env: int = 24
    max_iterations: int = 500
    save_interval: int = 50
    experiment_name: str = "g1_shoot_student_dagger_v1"
    obs_groups: dict[str, list[str]] = {"policy": ["policy"], "teacher": ["teacher"]}
    clip_actions: float | None = None
    policy: RslRlDistillationStudentTeacherCfg = RslRlDistillationStudentTeacherCfg(
        init_noise_std=0.0,
        student_obs_normalization=False,
        teacher_obs_normalization=True,
        student_hidden_dims=[512, 256, 128],
        teacher_hidden_dims=[1024, 512, 256],
        activation="elu",
    )
    algorithm: RslRlDistillationAlgorithmCfg = RslRlDistillationAlgorithmCfg(
        num_learning_epochs=1,
        learning_rate=3.0e-4,
        gradient_length=1,
        max_grad_norm=1.0,
        loss_type="mse",
    )
    replay_capacity: int = 262_144
    replay_recent_storage_fraction: float = 0.5
    replay_recent_sample_fraction: float = 0.75
    updates_per_iteration: int = 32
    batch_size: int = 8192
    student_execution_start_fraction: float = 0.20
    student_execution_end_fraction: float = 0.50
