"""RSL-RL runner integration and self-contained Student PPO+BC checkpoints."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import torch
from rsl_rl.modules import ActorCritic
from rsl_rl.runners import OnPolicyRunner

from bb_mimic_nt.policy_contract import load_teacher_contract
from bb_mimic_nt.student_ppo import PPOWithBC
from bb_mimic_nt.student_runtime import (
    student_teacher_configs, student_validate_resume_observation, student_validate_teacher,
)


CHECKPOINT_FORMAT = "bb_mimic_nt_student_ppo_bc_v1"


class FrozenTeacher(torch.nn.Module):
    """Only the archived Teacher actor and its frozen observation normalizer."""

    def __init__(self, actor, normalizer):
        super().__init__()
        self.actor = actor
        self.normalizer = normalizer
        self.eval().requires_grad_(False)

    def forward(self, observations):
        return self.actor(self.normalizer(observations["teacher"]))


class StudentPPOBCRunner(OnPolicyRunner):
    """Retain the upstream training loop, with completed-iteration checkpoints."""

    def __init__(self, env, train_cfg, teacher_checkpoint, metadata, log_dir=None, device="cpu"):
        self.teacher_checkpoint = Path(teacher_checkpoint).resolve()
        self.metadata = deepcopy(metadata)
        self.training_config = deepcopy(train_cfg)
        self.base = env.unwrapped
        self._inside_learn = False
        self._target_completed = None
        self._last_saved_completed = None
        # The parent calls our construction hook before setting its logging fields.
        super().__init__(env, deepcopy(train_cfg), log_dir=log_dir, device=device)
        if log_dir is None:
            # Upstream otherwise tries to archive git state into a None directory.
            self.disable_logs = True

    def _construct_algorithm(self, obs):
        groups = self.cfg["obs_groups"]
        if groups.get("policy") != ["policy"] or groups.get("critic") != ["teacher"]:
            raise ValueError("Student PPO+BC requires policy=['policy'] and critic=['teacher'].")
        archived_agent, archived_env = student_teacher_configs(self.teacher_checkpoint)
        load_teacher_contract(self.teacher_checkpoint, self.base)
        student_validate_teacher(archived_env, self.base)
        teacher_cfg = deepcopy(archived_agent["policy"])
        if teacher_cfg.pop("class_name") != "ActorCritic":
            raise ValueError("Student PPO+BC requires a feed-forward ActorCritic Teacher.")
        legacy_norm = bool(archived_agent.get("empirical_normalization", False))
        for name in ("actor_obs_normalization", "critic_obs_normalization"):
            if teacher_cfg.get(name) is None:
                teacher_cfg[name] = legacy_norm
        archived_model = ActorCritic(
            obs, {"policy": ["teacher"], "critic": ["teacher"]}, self.env.num_actions, **teacher_cfg,
        ).to(self.device)
        archived = torch.load(self.teacher_checkpoint, weights_only=True, map_location=self.device)
        archived_model.load_state_dict(archived["model_state_dict"])

        policy_cfg = deepcopy(self.policy_cfg)
        if policy_cfg.pop("class_name") != "ActorCritic":
            raise ValueError("Student PPO+BC uses the official feed-forward ActorCritic.")
        if policy_cfg.get("actor_obs_normalization", False):
            raise ValueError("The deployable Student actor must not use empirical observation normalization.")
        if policy_cfg.get("activation", "elu") != teacher_cfg.get("activation", "elu"):
            raise ValueError("Student and archived critic activation must match.")
        policy_cfg["critic_hidden_dims"] = teacher_cfg["critic_hidden_dims"]
        policy_cfg["critic_obs_normalization"] = teacher_cfg["critic_obs_normalization"]
        policy = ActorCritic(obs, groups, self.env.num_actions, **policy_cfg).to(self.device)
        policy.critic.load_state_dict(archived_model.critic.state_dict())
        policy.critic_obs_normalizer.load_state_dict(archived_model.critic_obs_normalizer.state_dict())
        teacher = FrozenTeacher(archived_model.actor, archived_model.actor_obs_normalizer)

        algorithm_cfg = deepcopy(self.alg_cfg)
        if algorithm_cfg.pop("class_name") != "PPOWithBC":
            raise ValueError("StudentPPOBCRunner requires PPOWithBC.")
        algorithm = PPOWithBC(
            policy, teacher, device=self.device, multi_gpu_cfg=self.multi_gpu_cfg,
            action_clip=self.metadata["action_clip"], action_scale=self.metadata["action_scale"],
            iteration_callback=self._iteration_completed, **algorithm_cfg,
        )
        algorithm.init_storage("rl", self.env.num_envs, self.num_steps_per_env, obs, [self.env.num_actions])
        return algorithm

    def _iteration_completed(self, completed):
        controller = getattr(self.base, "student_termination_curriculum", None)
        metrics = controller.update_iteration() if controller is not None else {}
        if self.writer is not None and not self.disable_logs:
            for name, value in metrics.items():
                self.writer.add_scalar(name, value, completed)
            for name, value in self.alg.last_update_statistics.items():
                self.writer.add_scalar(f"StudentPPO/{name}", value, completed)
            for role in ("actor", "critic"):
                self.writer.add_scalar(
                    f"Loss/{role}_learning_rate", getattr(self.alg, f"{role}_learning_rate"), completed,
                )
        if self.log_dir is not None and not self.disable_logs and completed % self.save_interval == 0:
            self._write_checkpoint(Path(self.log_dir) / f"model_{completed}.pt")

    def learn(self, num_learning_iterations, init_at_random_ep_len=False):
        if init_at_random_ep_len:
            raise ValueError("Student survival curricula require real episode lengths, not randomized counters.")
        if num_learning_iterations < 0:
            raise ValueError("The remaining iteration count cannot be negative.")
        self.current_learning_iteration = self.alg.completed_iterations
        if num_learning_iterations == 0:
            return
        self._inside_learn = True
        self._target_completed = self.alg.completed_iterations + num_learning_iterations
        try:
            return super().learn(num_learning_iterations, init_at_random_ep_len=False)
        finally:
            self.current_learning_iteration = self.alg.completed_iterations
            self._inside_learn = False
            self._target_completed = None

    def save(self, path, infos=None):
        if self._inside_learn:
            # Parent save triggers use zero-based indices. Periodic saves already
            # happen in the iteration callback; retain only its final-save call.
            completed = self.alg.completed_iterations
            if completed != self._target_completed or completed == self._last_saved_completed:
                return
            path = Path(path).with_name(f"model_{completed}.pt")
        self._write_checkpoint(Path(path), infos)

    def log(self, locs, width=80, pad=35):
        # Keep losses, environment metrics and curriculum metrics on the same
        # completed-iteration axis, while translating the parent's ETA formula.
        completed_locs = dict(locs)
        completed_locs["it"] = self.alg.completed_iterations
        completed_locs["start_iter"] = locs["start_iter"] + 1
        completed_locs["num_learning_iterations"] = locs["num_learning_iterations"] - 1
        return super().log(completed_locs, width=width, pad=pad)

    def _write_checkpoint(self, path, infos=None):
        completed = self.alg.completed_iterations
        controller = getattr(self.base, "student_termination_curriculum", None)
        payload = {
            "format": CHECKPOINT_FORMAT,
            "model_state_dict": self.alg.policy.state_dict(),
            "student": {name: value.detach().cpu() for name, value in self.alg.policy.actor.state_dict().items()},
            "optimizer_format": "separate_actor_critic_v1",
            "actor_optimizer_state_dict": self.alg.actor_optimizer.state_dict(),
            "critic_optimizer_state_dict": self.alg.critic_optimizer.state_dict(),
            "actor_learning_rate": self.alg.actor_learning_rate,
            "critic_learning_rate": self.alg.critic_learning_rate,
            # Keep the upstream log/metadata convention: learning_rate is the actor rate.
            "learning_rate": self.alg.learning_rate,
            "next_iteration": completed,
            "teacher_sha256": self.metadata["teacher_sha256"],
            "metadata": deepcopy(self.metadata),
            "training_config": deepcopy(self.training_config),
            "curriculum_state": controller.state_dict() if controller is not None else None,
            "infos": infos,
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(payload, path)
        self._last_saved_completed = completed
        if getattr(self, "logger_type", None) in ("wandb", "neptune") and not self.disable_logs:
            self.writer.save_model(str(path), completed)

    def load(self, path, load_optimizer=True, map_location=None):
        state = torch.load(path, weights_only=False, map_location=map_location or self.device)
        if state.get("format") != CHECKPOINT_FORMAT:
            raise ValueError("Resume requires a Student PPO+BC checkpoint; DAgger warm starts are unsupported.")
        if state["teacher_sha256"] != self.metadata["teacher_sha256"]:
            raise ValueError("Resume Teacher SHA256 differs from the selected Teacher.")
        student_validate_resume_observation(state["metadata"], self.metadata)
        completed = state["next_iteration"]
        if isinstance(completed, bool) or not isinstance(completed, int) or completed < 0:
            raise ValueError("Checkpoint next_iteration must be a non-negative completed-iteration count.")
        controller = getattr(self.base, "student_termination_curriculum", None)
        if state.get("curriculum_state") is not None and controller is None:
            raise ValueError("Checkpoint contains a termination curriculum but the environment has none.")
        # RSL's normalizer may replace buffers while collecting in inference
        # mode. Loading into an already-used runner must also allow their copy.
        with torch.inference_mode():
            self.alg.policy.load_state_dict(state["model_state_dict"])
        if load_optimizer:
            self._load_optimizers(state)
        self.alg.completed_iterations = self.current_learning_iteration = completed
        if controller is not None and state.get("curriculum_state") is not None:
            controller.load_state_dict(state["curriculum_state"])
        if hasattr(self.base, "dr_context"):
            # Also works if load is called on an environment already stepped.
            common_steps = getattr(self.base, "common_step_counter", 0)
            self.base.dr_context.iteration_offset = completed - common_steps / self.num_steps_per_env
        # A new physical episode snapshots the restored per-skill termination scale.
        self.base.reset()
        return state.get("infos")

    def _load_optimizers(self, checkpoint):
        """Restore separate Adam states or split the original single-group Adam.

        Legacy Adam serialized ``policy.parameters()`` in registration order;
        log_std precedes the actor in that order. Map parameter identities rather
        than assuming the actor and critic occupy contiguous state slices.
        """
        layout = checkpoint.get("optimizer_format")
        if layout not in (None, "separate_actor_critic_v1"):
            raise ValueError(f"Unsupported Student optimizer format: {layout!r}.")
        separate = layout is not None or any(
            f"{role}_optimizer_state_dict" in checkpoint for role in ("actor", "critic")
        )
        if separate:
            if not all(f"{role}_optimizer_state_dict" in checkpoint for role in ("actor", "critic")):
                raise ValueError("Student checkpoint must contain both Actor and Critic optimizer states.")
            states = {role: checkpoint[f"{role}_optimizer_state_dict"] for role in ("actor", "critic")}
            rates = {role: float(checkpoint[f"{role}_learning_rate"]) for role in ("actor", "critic")}
        else:
            saved = checkpoint["optimizer_state_dict"]
            if len(saved["param_groups"]) != 1:
                raise ValueError("Legacy Student PPO+BC checkpoints require one shared Adam parameter group.")
            group = saved["param_groups"][0]
            parameters = list(self.alg.policy.parameters())
            if len(group["params"]) != len(parameters):
                raise ValueError("Legacy Adam parameter count differs from the Student policy.")
            source_ids = {id(parameter): index for parameter, index in zip(parameters, group["params"])}
            source_names = dict(zip(group["params"], group["param_names"])) if "param_names" in group else None
            states = {}
            for role in ("actor", "critic"):
                selected = [source_ids[id(parameter)] for parameter in getattr(self.alg, f"{role}_parameters")]
                options = {key: deepcopy(value) for key, value in group.items() if key not in ("params", "param_names")}
                options["params"] = list(range(len(selected)))
                if source_names is not None:
                    options["param_names"] = [source_names[index] for index in selected]
                states[role] = {
                    "param_groups": [options],
                    "state": {index: deepcopy(saved["state"][source])
                              for index, source in enumerate(selected) if source in saved["state"]},
                }
            rates = dict.fromkeys(("actor", "critic"), float(checkpoint.get("learning_rate", group["lr"])))

        for role in ("actor", "critic"):
            optimizer = getattr(self.alg, f"{role}_optimizer")
            optimizer.load_state_dict(states[role])
            setattr(self.alg, f"{role}_learning_rate", rates[role])
            for group in optimizer.param_groups:
                group["lr"] = rates[role]
        self.alg.learning_rate = self.alg.actor_learning_rate
        if not separate:
            print("Restored legacy shared Adam state into separate Actor/Critic optimizers.")
