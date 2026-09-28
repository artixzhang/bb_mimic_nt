"""Student checkpoint compatibility, construction, validation, and deployment export."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch
import yaml
from rsl_rl.modules import StudentTeacher
from rsl_rl.networks import MLP

from bb_mimic_nt.student import STUDENT_ACTION_DIM, STUDENT_OBSERVATION_DIM, STUDENT_OBSERVATION_FIELDS
from bb_mimic_nt.training import TEACHER_OBSERVATION_DIM


class StudentYamlLoader(yaml.SafeLoader):
    """Safe loader extended only for tuples emitted by Isaac Lab configuration dumps."""


StudentYamlLoader.add_constructor(
    "tag:yaml.org,2002:python/tuple",
    lambda loader, node: tuple(loader.construct_sequence(node)),
)


class StudentPolicy(torch.nn.Module):
    """Deployment wrapper containing only the Student MLP."""

    is_recurrent = False

    def __init__(self, hidden_dims: list[int], activation: str):
        super().__init__()
        self.student = MLP(STUDENT_OBSERVATION_DIM, STUDENT_ACTION_DIM, hidden_dims, activation)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.student(observations)


def student_teacher_configs(checkpoint: Path) -> tuple[dict, dict]:
    """Load the archived Teacher network and environment configurations."""
    checkpoint = checkpoint.resolve()
    params = checkpoint.parent / "params"
    agent_path = params / "agent.yaml"
    environment_path = params / "env.yaml"
    if not checkpoint.is_file() or not agent_path.is_file() or not environment_path.is_file():
        raise FileNotFoundError("Teacher checkpoint and params/agent.yaml plus params/env.yaml are required.")
    with agent_path.open(encoding="utf-8") as stream:
        agent = yaml.load(stream, Loader=StudentYamlLoader)
    with environment_path.open(encoding="utf-8") as stream:
        environment = yaml.load(stream, Loader=StudentYamlLoader)
    return agent, environment


def student_apply_teacher_config(env_cfg, checkpoint: Path) -> tuple[dict, dict]:
    """Make the Student action adapter and physical DR match the selected Teacher run."""
    agent, environment = student_teacher_configs(checkpoint)
    teacher_action = environment["actions"]["joint_pos"]
    for name in (
        "residual_scale_fraction",
        "minimum_residual_scale",
        "maximum_residual_scale",
        "position_limit_margin",
    ):
        setattr(env_cfg.actions.joint_pos, name, float(teacher_action[name]))
    env_cfg.actions.joint_pos.joint_maximum_residual_scales = {
        str(name): float(value)
        for name, value in teacher_action.get("joint_maximum_residual_scales", {}).items()
    }
    teacher_dr = environment["dr"]
    scalar_names = (
        "delay_nominal_steps",
        "delay_max_offset_steps",
        "ball_mass_fraction",
        "pd_gain_fraction",
        "hand_friction_nominal",
        "hand_friction_delta",
        "foot_friction_nominal",
        "foot_friction_delta",
        "link_mass_fraction",
        "push_torque_max_nm",
        "push_duration_s",
    )
    for name in scalar_names:
        setattr(env_cfg.dr, name, teacher_dr[name])
    env_cfg.dr.push_force_max_n = tuple(teacher_dr["push_force_max_n"])
    env_cfg.dr.push_force_min_n = tuple(teacher_dr.get("push_force_min_n", (0.0, 0.0, 0.0)))
    return agent, environment


def student_validate_teacher(environment: dict, env) -> None:
    action = env.action_manager.get_term("joint_pos")
    archived_action = environment["actions"]["joint_pos"]
    if archived_action["joint_names"] != list(action.cfg.joint_names):
        raise ValueError("Teacher checkpoint joint order differs from the Student environment.")
    for name in (
        "residual_scale_fraction",
        "minimum_residual_scale",
        "maximum_residual_scale",
        "position_limit_margin",
    ):
        if abs(float(archived_action[name]) - float(getattr(action.cfg, name))) > 1.0e-8:
            raise ValueError(f"Teacher action setting {name} differs from the Student adapter.")
    archived_caps = {
        str(name): float(value)
        for name, value in archived_action.get("joint_maximum_residual_scales", {}).items()
    }
    current_caps = {
        str(name): float(value) for name, value in action.cfg.joint_maximum_residual_scales.items()
    }
    if archived_caps != current_caps:
        raise ValueError("Teacher per-joint residual caps differ from the Student adapter.")
    dimensions = env.observation_manager.group_obs_dim
    expected = {"policy": (STUDENT_OBSERVATION_DIM,), "teacher": (TEACHER_OBSERVATION_DIM,)}
    if any(dimensions.get(name) != shape for name, shape in expected.items()):
        raise ValueError(f"Unexpected Student/Teacher observation dimensions: {dimensions}.")
    if action.action_dim != STUDENT_ACTION_DIM:
        raise ValueError(f"Expected {STUDENT_ACTION_DIM} controlled joints, got {action.action_dim}.")


def student_load_teacher(checkpoint: Path, observations, student_cfg, env) -> tuple[StudentTeacher, dict]:
    """Construct the official RSL-RL StudentTeacher module and load only the Teacher actor."""
    agent, environment = student_teacher_configs(checkpoint)
    student_validate_teacher(environment, env)
    policy_cfg = agent["policy"]
    if policy_cfg["class_name"] != "ActorCritic":
        raise ValueError("The Teacher checkpoint must contain a feed-forward ActorCritic.")
    teacher_normalization = bool(
        policy_cfg.get("actor_obs_normalization", agent.get("empirical_normalization", False))
    )
    model = StudentTeacher(
        observations,
        {"policy": ["policy"], "teacher": ["teacher"]},
        STUDENT_ACTION_DIM,
        student_obs_normalization=False,
        teacher_obs_normalization=teacher_normalization,
        student_hidden_dims=list(student_cfg.policy.student_hidden_dims),
        teacher_hidden_dims=list(policy_cfg["actor_hidden_dims"]),
        activation=policy_cfg["activation"],
        init_noise_std=0.0,
    ).to(env.device)
    state = torch.load(checkpoint, map_location=env.device, weights_only=True)
    model.load_state_dict(state["model_state_dict"])
    if not model.loaded_teacher:
        raise RuntimeError("Teacher actor weights were not loaded.")
    model.teacher.eval()
    model.teacher_obs_normalizer.eval()
    return model, agent


def student_metadata(env, student_cfg, teacher_checkpoint: Path | None = None) -> dict:
    action = env.action_manager.get_term("joint_pos")
    robot = env.scene["robot"]
    joint_ids = action._joint_ids
    training_observation = env.cfg.student_observation.to_dict()
    inference_observation = json.loads(json.dumps(training_observation))
    inference_observation["enable_noise"] = False
    inference_observation["randomize_delay"] = False
    metadata = {
        "format": "bb_mimic_nt_student_v1",
        "observation_dim": STUDENT_OBSERVATION_DIM,
        "observation_fields": [{"name": name, "size": size} for name, size in STUDENT_OBSERVATION_FIELDS],
        "history_order": "oldest_to_newest",
        "action_dim": STUDENT_ACTION_DIM,
        "action_semantics": "joint_position_radians_minus_nominal_pose",
        "joint_names": list(action.cfg.joint_names),
        "nominal_joint_position": action.nominal[0].detach().cpu().tolist(),
        "safe_lower_joint_position": action.lower[0].detach().cpu().tolist(),
        "safe_upper_joint_position": action.upper[0].detach().cpu().tolist(),
        "pd_stiffness": robot.data.default_joint_stiffness[0, joint_ids].detach().cpu().tolist(),
        "pd_damping": robot.data.default_joint_damping[0, joint_ids].detach().cpu().tolist(),
        "policy_hz": 1.0 / env.step_dt,
        "student_hidden_dims": list(student_cfg.policy.student_hidden_dims),
        "activation": student_cfg.policy.activation,
        "action_adapter": {
            **{
                name: float(getattr(action.cfg, name))
                for name in (
                    "residual_scale_fraction",
                    "minimum_residual_scale",
                    "maximum_residual_scale",
                    "position_limit_margin",
                )
            },
            "joint_maximum_residual_scales": {
                str(name): float(value)
                for name, value in action.cfg.joint_maximum_residual_scales.items()
            },
        },
        "observation_settings": inference_observation,
        "training_observation_settings": training_observation,
        "default_action_delay_steps": env.cfg.dr.delay_nominal_steps,
        "default_observation_delay_steps": env.cfg.student_observation.delay_nominal_steps,
        "observation_delay_encoding": "(delay_steps - 2) / 2; logged only, not part of the 283 inputs",
        "teacher_checkpoint": str(teacher_checkpoint.resolve()) if teacher_checkpoint else None,
    }
    if teacher_checkpoint:
        with teacher_checkpoint.open("rb") as stream:
            metadata["teacher_sha256"] = hashlib.file_digest(stream, "sha256").hexdigest()
    return metadata


def student_read_artifact(weights: Path, config: Path | None = None) -> tuple[dict, dict]:
    """Read either a training checkpoint or exported Student weights once on CPU."""
    state = torch.load(weights, map_location="cpu", weights_only=False)
    if isinstance(state, dict) and state.get("format") == "bb_mimic_nt_student_dagger_v1":
        if "student" not in state or "metadata" not in state:
            raise ValueError("Student DAgger checkpoint is missing weights or deployment metadata.")
        return state["student"], state["metadata"]
    config = config or weights.with_name("student_config.json")
    if not config.is_file():
        raise FileNotFoundError(
            f"Exported Student weights require their deployment config: {config}. "
            "Pass --student-config or use a self-contained model_*.pt checkpoint."
        )
    metadata = json.loads(config.read_text(encoding="utf-8"))
    return state, metadata


def student_policy_from_state(state: dict, metadata: dict, device: str) -> StudentPolicy:
    """Construct a deployable Student policy from an already-loaded state dictionary."""
    if metadata.get("format") != "bb_mimic_nt_student_v1":
        raise ValueError("Unsupported Student deployment configuration format.")
    if metadata["observation_dim"] != STUDENT_OBSERVATION_DIM or metadata["action_dim"] != STUDENT_ACTION_DIM:
        raise ValueError("Student deployment configuration has incompatible dimensions.")
    policy = StudentPolicy(metadata["student_hidden_dims"], metadata["activation"]).to(device)
    policy.student.load_state_dict(state)
    return policy.eval()


def student_load_policy(weights: Path, config: Path | None, device: str) -> tuple[StudentPolicy, dict]:
    state, metadata = student_read_artifact(weights, config)
    return student_policy_from_state(state, metadata, device), metadata


def student_apply_deployment_config(env_cfg, metadata: dict) -> None:
    settings = metadata["observation_settings"]
    for name, value in settings["scales"].items():
        setattr(env_cfg.student_observation.scales, name, float(value))
    for name in (
        "delay_nominal_steps",
        "delay_max_offset_steps",
        "gravity_noise_std",
        "angular_velocity_noise_std",
        "joint_position_noise_std",
        "joint_velocity_noise_std",
    ):
        setattr(env_cfg.student_observation, name, settings[name])
    env_cfg.student_observation.enable_noise = False
    env_cfg.student_observation.randomize_delay = False
    env_cfg.dr.delay_nominal_steps = int(metadata["default_action_delay_steps"])
    for name, value in metadata["action_adapter"].items():
        setattr(env_cfg.actions.joint_pos, name, value)


def student_validate_deployment(env, metadata: dict) -> None:
    action = env.action_manager.get_term("joint_pos")
    if metadata["joint_names"] != list(action.cfg.joint_names):
        raise ValueError("Student export joint order does not match the environment.")
    robot = env.scene["robot"]
    joint_ids = action._joint_ids
    tensors = {
        "nominal_joint_position": action.nominal[0],
        "safe_lower_joint_position": action.lower[0],
        "safe_upper_joint_position": action.upper[0],
        "pd_stiffness": robot.data.default_joint_stiffness[0, joint_ids],
        "pd_damping": robot.data.default_joint_damping[0, joint_ids],
    }
    for name, current in tensors.items():
        saved = torch.as_tensor(metadata[name], device=current.device)
        if not torch.allclose(current, saved, rtol=0.0, atol=1.0e-5):
            raise ValueError(f"Student export {name} differs from the environment.")


def student_export(policy: StudentPolicy, destination: Path, metadata: dict) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    policy.eval()
    state = {key: value.detach().cpu() for key, value in policy.student.state_dict().items()}
    torch.save(state, destination / "student_weights.pt")
    (destination / "student_config.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    export_policy = StudentPolicy(metadata["student_hidden_dims"], metadata["activation"])
    export_policy.student.load_state_dict(state)
    export_policy.eval()
    scripted = torch.jit.script(export_policy)
    scripted.save(str(destination / "student_policy.pt"))
    example = torch.zeros((1, STUDENT_OBSERVATION_DIM))
    torch.onnx.export(
        export_policy,
        example,
        destination / "student_policy.onnx",
        input_names=["student_observation"],
        output_names=["joint_position_offset"],
        dynamic_axes={"student_observation": {0: "batch"}, "joint_position_offset": {0: "batch"}},
        opset_version=17,
    )
