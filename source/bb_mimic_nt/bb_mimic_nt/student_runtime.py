"""Student checkpoint compatibility, construction, validation, and deployment export."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

import torch
import yaml
from rsl_rl.modules import StudentTeacher
from rsl_rl.networks import MLP

from bb_mimic_nt.student import (
    STUDENT_ACTION_DIM, STUDENT_OBSERVATION_DIM, student_delay_bounds, student_observation_fields,
)
from bb_mimic_nt.student_deployment import (
    STUDENT_DEPLOYMENT_FORMAT, build_deployment_contract, compact_student_config,
    student_observation_scales, student_sensor_delay, validate_student_config,
)
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

    def __init__(self, hidden_dims: list[int], activation: str, observation_dim: int = STUDENT_OBSERVATION_DIM):
        super().__init__()
        self.student = MLP(observation_dim, STUDENT_ACTION_DIM, hidden_dims, activation)

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
    """Copy Teacher action/range settings while preserving Student curricula and play flags."""
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
    teacher_motion = environment["commands"]["motion"]
    for name in (
        "source_path", "cache_path", "urdf_path", "pre_hold_s", "post_hold_s",
        "reset_dof_pos_noise", "reset_dof_vel_noise", "default_clip_id",
    ):
        setattr(env_cfg.commands.motion, name, teacher_motion[name])
    # Older checkpoints used full holds; preserve that behavior. Play always
    # executes full holds, while distillation inherits the trained sampling.
    if getattr(env_cfg.commands.motion, "hold_sampling_weights", (0.15, 0.15)) is not None:
        weights = teacher_motion.get("hold_sampling_weights")
        env_cfg.commands.motion.hold_sampling_weights = tuple(weights) if weights is not None else None
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
    observation_dim = sum(size for _, size in student_observation_fields(env.cfg.student_observation.history_length))
    expected = {"policy": (observation_dim,), "teacher": (TEACHER_OBSERVATION_DIM,)}
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
    if student_cfg.policy.activation != policy_cfg["activation"]:
        raise ValueError("StudentTeacher uses a shared activation; Student must match the archived Teacher activation.")
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
    fields = student_observation_fields(env.cfg.student_observation.history_length)
    delay_settings = env.cfg.student_observation
    delay_denominator = max(
        delay_settings.delay_nominal_steps - delay_settings.delay_min_steps,
        delay_settings.delay_max_steps - delay_settings.delay_nominal_steps, 1,
    )
    metadata = {
        "format": "bb_mimic_nt_student_v1",
        "observation_dim": sum(size for _, size in fields),
        "observation_fields": [{"name": name, "size": size} for name, size in fields],
        "history_length": env.cfg.student_observation.history_length,
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
        "observation_delay_encoding": (
            f"(delay_steps - {delay_settings.delay_nominal_steps}) / {delay_denominator}; logged only"
        ),
        "teacher_checkpoint": str(teacher_checkpoint.resolve()) if teacher_checkpoint else None,
    }
    if teacher_checkpoint:
        with teacher_checkpoint.open("rb") as stream:
            metadata["teacher_sha256"] = hashlib.file_digest(stream, "sha256").hexdigest()
    motion = env.command_manager.get_term("motion")
    metadata["deployment_contract"] = build_deployment_contract(metadata, motion._batch, motion.cfg)
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
    manifest_path = config.with_name("manifest.json")
    if metadata.get("format") == STUDENT_DEPLOYMENT_FORMAT and manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        metadata["_provenance"] = manifest.get("provenance", {})
    return state, metadata


def student_validate_resume_observation(previous: dict, current: dict) -> None:
    """Allow curriculum timing changes while rejecting incompatible replay inputs."""
    if previous["observation_dim"] != current["observation_dim"]:
        raise ValueError("Resume history dimension differs; five-frame Student requires a new training run.")
    schedule_fields = {"enabled", "start_fraction", "end_fraction", "full_strength"}
    previous_settings = {
        name: value for name, value in previous.get("training_observation_settings", {}).items()
        if name not in schedule_fields
    }
    current_settings = {
        name: value for name, value in current.get("training_observation_settings", {}).items()
        if name not in schedule_fields
    }
    if previous_settings != current_settings:
        raise ValueError("Resume observation settings differ from the current noise/history/delay configuration.")


def student_artifact_history(metadata: dict) -> int:
    """Recover legacy three-frame exports and validate current history metadata."""
    if metadata.get("format") == STUDENT_DEPLOYMENT_FORMAT:
        validate_student_config(metadata)
        return metadata["history_length"]
    dimension = metadata["observation_dim"]
    if not isinstance(dimension, int) or dimension <= 4 or (dimension - 4) % 93:
        raise ValueError("Student deployment configuration has incompatible observation dimensions.")
    history = (dimension - 4) // 93
    if metadata.get("history_length", history) != history:
        raise ValueError("Student deployment history length and observation dimension disagree.")
    fields = [{"name": name, "size": size} for name, size in student_observation_fields(history)]
    if metadata.get("observation_fields", fields) != fields:
        raise ValueError("Student deployment observation fields disagree with its history length.")
    return history


def student_policy_from_state(state: dict, metadata: dict, device: str) -> StudentPolicy:
    """Construct a deployable Student policy from an already-loaded state dictionary."""
    if metadata.get("format") not in ("bb_mimic_nt_student_v1", STUDENT_DEPLOYMENT_FORMAT):
        raise ValueError("Unsupported Student deployment configuration format.")
    student_artifact_history(metadata)
    if metadata["action_dim"] != STUDENT_ACTION_DIM:
        raise ValueError("Student deployment configuration has incompatible dimensions.")
    policy = StudentPolicy(
        metadata["student_hidden_dims"], metadata["activation"], metadata["observation_dim"]
    ).to(device)
    policy.student.load_state_dict(state)
    return policy.eval()


def student_load_policy(weights: Path, config: Path | None, device: str) -> tuple[StudentPolicy, dict]:
    state, metadata = student_read_artifact(weights, config)
    return student_policy_from_state(state, metadata, device), metadata


def student_apply_deployment_config(env_cfg, metadata: dict) -> None:
    if metadata.get("format") == STUDENT_DEPLOYMENT_FORMAT:
        validate_student_config(metadata)
        phase = metadata["phase"]
        env_cfg.commands.motion.pre_hold_s = phase["pre_hold_s"]
        env_cfg.commands.motion.post_hold_s = phase["post_hold_s"]
        delay = student_sensor_delay(metadata)
        env_cfg.student_observation.history_length = metadata["history_length"]
        env_cfg.student_observation.delay_nominal_steps = delay
        env_cfg.student_observation.delay_min_steps = delay
        env_cfg.student_observation.delay_max_steps = delay
        for name, value in student_observation_scales(metadata).items():
            if name == "hoop_position":
                radius, theta, height = value if isinstance(value, (list, tuple)) else (value,) * 3
                if radius != height:
                    raise ValueError("Simulation requires matching hoop radius and height scales.")
                env_cfg.student_observation.scales.hoop_theta = float(theta)
                value = radius
            setattr(env_cfg.student_observation.scales, name, float(value))
        env_cfg.student_observation.enable_noise = False
        env_cfg.student_observation.randomize_delay = False
        env_cfg.dr.delay_nominal_steps = metadata["action"]["delay_steps"]
        # The simulation consumes the exact exported coordinates/limits; it
        # does not reconstruct them from Teacher residual adapter parameters.
        for name in ("nominal_joint_position", "safe_lower_joint_position", "safe_upper_joint_position"):
            setattr(env_cfg.actions.joint_pos, name, list(metadata[name]))
        return
    contract = metadata.get("deployment_contract")
    if contract is not None:
        env_cfg.commands.motion.pre_hold_s = contract["phase"]["trained_pre_hold_s"]
        env_cfg.commands.motion.post_hold_s = contract["phase"]["trained_post_hold_s"]
    settings = metadata["observation_settings"]
    env_cfg.student_observation.history_length = student_artifact_history(metadata)
    nominal = settings["delay_nominal_steps"]
    offset = settings.get("delay_max_offset_steps", 0)
    minimum = settings.get("delay_min_steps", max(0, nominal - offset))
    maximum = settings.get("delay_max_steps", nominal + offset)
    student_delay_bounds(nominal, minimum, maximum)
    env_cfg.student_observation.delay_min_steps = minimum
    env_cfg.student_observation.delay_max_steps = maximum
    env_cfg.student_observation.scales.hoop_theta = float(
        settings["scales"].get("hoop_theta", settings["scales"].get("hoop_position", 0.25))
    )
    for name, value in settings["scales"].items():
        setattr(env_cfg.student_observation.scales, name, float(value))
    for name in ("delay_nominal_steps", "gravity_noise_std", "angular_velocity_noise_std",
                 "joint_position_noise_std", "joint_velocity_noise_std"):
        setattr(env_cfg.student_observation, name, settings[name])
    env_cfg.student_observation.enable_noise = False
    env_cfg.student_observation.randomize_delay = False
    env_cfg.dr.delay_nominal_steps = int(metadata["default_action_delay_steps"])
    for name, value in metadata["action_adapter"].items():
        setattr(env_cfg.actions.joint_pos, name, value)


def student_validate_deployment(env, metadata: dict) -> None:
    """Check the policy interface while allowing different simulation settings."""
    if env.observation_manager.group_obs_dim["policy"] != (metadata["observation_dim"],):
        raise ValueError("Student export observation dimension does not match the environment.")
    compact = metadata.get("format") == STUDENT_DEPLOYMENT_FORMAT
    if compact:
        validate_student_config(metadata)
    action = env.action_manager.get_term("joint_pos")
    if metadata["joint_names"] != list(action.cfg.joint_names):
        raise ValueError("Student export joint order does not match the environment.")


def student_prepare_export_metadata(metadata: dict, provenance: dict | None = None) -> dict:
    """Require a complete contract, recovering legacy metadata from archived motion configuration."""
    metadata = json.loads(json.dumps(metadata, allow_nan=False))
    metadata["history_length"] = student_artifact_history(metadata)
    if metadata.get("format") == STUDENT_DEPLOYMENT_FORMAT:
        return compact_student_config(metadata)
    if "deployment_contract" not in metadata:
        if not metadata.get("teacher_checkpoint"):
            raise ValueError("Export needs deployment motion metadata or an accessible archived Teacher checkpoint.")
        _, environment = student_teacher_configs(Path(metadata["teacher_checkpoint"]))
        motion = environment["commands"]["motion"]
        cache = Path(motion["cache_path"])
        if not cache.is_file():
            raise FileNotFoundError(f"Legacy export needs the archived motion cache: {cache}")
        batch = torch.load(cache, map_location="cpu", weights_only=False)
        metadata["deployment_contract"] = build_deployment_contract(metadata, batch, SimpleNamespace(**motion))
        metadata["deployment_contract"]["provenance"]["recovered_from_archived_cache"] = True
    if provenance is not None:
        provenance.update(metadata["deployment_contract"].get("provenance", {}))
    return compact_student_config(metadata)


def student_export(
    policy: StudentPolicy, destination: Path, metadata: dict, *,
    source_path: Path | None = None, source_iteration: int | None = None,
) -> None:
    # Read the source before writing artifacts, including in-place re-exports.
    source = metadata.get("export_source", metadata.get("_provenance", {}).get("export_source"))
    if source_path is not None:
        source_path = source_path.resolve()
        with source_path.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        source = dict(path=str(source_path), file=source_path.name,
                      size_bytes=source_path.stat().st_size, sha256=digest)
        if source_iteration is not None:
            source["iteration"] = source_iteration
    # Other provenance stays beside the runtime interface, including on v2 re-export.
    provenance = dict(metadata.get("_provenance", {}))
    provenance.update(metadata.get("deployment_contract", {}).get("provenance", {}))
    for key in ("teacher_checkpoint", "teacher_sha256", "export_source"):
        if metadata.get(key) is not None:
            provenance[key] = metadata[key]
    metadata = student_prepare_export_metadata(metadata, provenance)
    if source is not None:
        metadata["export_source"] = source
        provenance["export_source"] = source
    destination.mkdir(parents=True, exist_ok=True)
    policy.eval()
    state = {key: value.detach().cpu() for key, value in policy.student.state_dict().items()}
    torch.save(state, destination / "student_weights.pt")
    (destination / "student_config.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    export_policy = StudentPolicy(
        metadata["student_hidden_dims"], metadata["activation"], metadata["observation_dim"]
    )
    export_policy.student.load_state_dict(state)
    export_policy.eval()
    # Compile native layers, not our Python wrapper. A long-running training
    # process may hold old source line numbers after this file is edited;
    # TorchScript otherwise reads a different function from the current file.
    # Sequential iteration preserves repeated activation instances; children()
    # deduplicates them and would silently drop intermediate activations.
    inference_network = torch.nn.Sequential(*export_policy.student).eval()
    scripted = torch.jit.script(inference_network)
    scripted.save(str(destination / "student_policy.pt"))
    example = torch.zeros((1, metadata["observation_dim"]))
    torch.onnx.export(
        export_policy,
        example,
        destination / "student_policy.onnx",
        input_names=["student_observation"],
        output_names=["joint_position_offset"],
        dynamic_axes={"student_observation": {0: "batch"}, "joint_position_offset": {0: "batch"}},
        opset_version=17,
    )

    import onnx
    onnx.checker.check_model(str(destination / "student_policy.onnx"))
    inputs = torch.randn((16, metadata["observation_dim"]), generator=torch.Generator().manual_seed(42)) * .1
    inputs[:, 0] = torch.linspace(0., 1., 16)
    with torch.inference_mode():
        expected = export_policy(inputs)
        actual = torch.jit.load(str(destination / "student_policy.pt"))(inputs)
    torch.testing.assert_close(actual, expected)
    if not torch.isfinite(expected).all():
        raise ValueError("Exported policy produced non-finite outputs.")
    from onnx.reference import ReferenceEvaluator
    reference_output = ReferenceEvaluator(str(destination / "student_policy.onnx")).run(
        None, {"student_observation": inputs.numpy()}
    )[0]
    np.testing.assert_allclose(reference_output, expected.numpy(), rtol=1.e-4, atol=1.e-5)
    np.savez(destination / "verification_vectors.npz", observations=inputs.numpy(), action_offsets=expected.numpy())
    # One canonical configuration serves simulation and robot deployment.
    # Remove the obsolete generated duplicate when re-exporting an old bundle.
    obsolete = destination / "deployment_config.json"
    if obsolete.is_file():
        obsolete.unlink()
    artifacts = ["student_weights.pt", "student_policy.pt", "student_policy.onnx", "student_config.json",
                 "verification_vectors.npz"]
    manifest = {
        "schema": "bb_mimic_nt_student_bundle_v1",
        "provenance": provenance,
        "artifacts_sha256": {name: hashlib.sha256((destination / name).read_bytes()).hexdigest() for name in artifacts},
        "verification": {"torchscript_matches_pytorch": True, "onnx_structure_checked": True,
                         "onnx_reference_matches_pytorch": True, "onnx_runtime_matches_pytorch": False, "robot_validated": False},
    }
    try:
        import onnxruntime
    except ImportError:
        pass
    else:
        session = onnxruntime.InferenceSession(str(destination / "student_policy.onnx"), providers=["CPUExecutionProvider"])
        onnx_output = session.run(None, {"student_observation": inputs.numpy()})[0]
        np.testing.assert_allclose(onnx_output, expected.numpy(), rtol=1.e-4, atol=1.e-5)
        manifest["verification"]["onnx_runtime_matches_pytorch"] = True
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
