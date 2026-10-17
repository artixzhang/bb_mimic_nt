"""Tensor-only, per-skill Student termination curriculum and episode snapshots."""

from __future__ import annotations

import math

import torch


class StudentTerminationCurriculum:
    """Collect terminal episodes, then adjust each skill at most once per update.

    Episode scales are frozen by ``begin_episode``. Checkpoints contain the
    curriculum statistics, not episode state: reset the environment after load.
    """

    def __init__(self, num_envs, skill_names, device, settings, base_thresholds=None):
        self.skill_names = tuple(skill_names)
        self.settings = settings
        self.enabled = bool(settings.enabled)
        self.base_thresholds = dict(base_thresholds or {
            "root_tracking_error": 1.0, "dof_tracking_error": 1.0, "interaction_tracking_error": .35,
        })
        if not self.skill_names or len(set(self.skill_names)) != len(self.skill_names):
            raise ValueError("Termination curriculum requires distinct skill names.")
        if not 0 < settings.minimum_scale <= settings.initial_scale <= settings.maximum_scale:
            raise ValueError("Termination curriculum scales must satisfy 0 < min <= initial <= max.")
        if settings.minimum_episodes < 1 or settings.ema_episodes <= 0:
            raise ValueError("Termination curriculum episode counts must be positive.")
        if not 0 <= settings.relax_below < settings.tighten_above <= 1:
            raise ValueError("Termination curriculum survival thresholds must be ordered in [0, 1].")
        if not settings.relax_factor > 1 or not 0 < settings.tighten_factor < 1:
            raise ValueError("Termination curriculum factors must relax above one and tighten below one.")
        count = len(self.skill_names)
        # These tensors may be first allocated from an inference-mode env.step.
        with torch.inference_mode(False):
            self.scales = torch.full((count,), float(settings.initial_scale), device=device)
            self.survival_ema = torch.zeros(count, device=device)
            self.ema_initialized = torch.zeros(count, dtype=torch.bool, device=device)
            self.pending_count = torch.zeros(count, dtype=torch.long, device=device)
            self.pending_survival = torch.zeros(count, device=device)
            self.pending_failures = torch.zeros(count, device=device)
            self.pending_completions = torch.zeros(count, device=device)
            self.total_episodes = torch.zeros(count, dtype=torch.long, device=device)
            self.last_failure_fraction = torch.zeros(count, device=device)
            self.last_completion_fraction = torch.zeros(count, device=device)
            self.episode_skill = torch.full((num_envs,), -1, dtype=torch.long, device=device)
            self.episode_planned_steps = torch.ones(num_envs, device=device)
            self.episode_scale = torch.ones(num_envs, device=device)
            self.episode_active = torch.zeros(num_envs, dtype=torch.bool, device=device)
            self.last_episode = {
                "skill_id": self.episode_skill.clone(),
                "elapsed_steps": torch.zeros(num_envs, device=device),
                "planned_steps": torch.ones(num_envs, device=device),
                "survival_fraction": torch.zeros(num_envs, device=device),
                "completed": torch.zeros(num_envs, dtype=torch.bool, device=device),
                "failed": torch.zeros(num_envs, dtype=torch.bool, device=device),
                "scale": torch.ones(num_envs, device=device),
            }
        self.last_episode_causes = {}
        self.pending_causes = {}
        self.last_cause_fractions = {}

    def begin_episode(self, env_ids, skill_ids, planned_steps):
        self.episode_skill[env_ids] = skill_ids
        self.episode_planned_steps[env_ids] = planned_steps.clamp_min(1)
        self.episode_scale[env_ids] = self.scales[skill_ids] if self.enabled else 1.0
        self.episode_active[env_ids] = True

    def finish_episodes(self, env_ids, elapsed_steps, failed, causes):
        """Capture only caller-confirmed terminal resets, before sampling new skills."""
        valid = self.episode_active[env_ids] & (elapsed_steps > 0)
        env_ids, elapsed_steps, failed = env_ids[valid], elapsed_steps[valid], failed[valid]
        if not len(env_ids):
            return
        selected_causes = {name: value[valid].bool() for name, value in causes.items()}
        completed = selected_causes.get("reference_finished", torch.zeros_like(failed)) & ~failed
        skills = self.episode_skill[env_ids]
        planned = self.episode_planned_steps[env_ids]
        survival = (elapsed_steps.float() / planned).clamp(0, 1)
        survival = torch.where(completed, torch.ones_like(survival), survival)
        values = dict(skill_id=skills, elapsed_steps=elapsed_steps, planned_steps=planned,
                      survival_fraction=survival, completed=completed, failed=failed,
                      scale=self.episode_scale[env_ids])
        for name, value in values.items():
            self.last_episode[name][env_ids] = value.to(self.last_episode[name].dtype)
        for name, value in selected_causes.items():
            if name not in self.last_episode_causes:
                with torch.inference_mode(False):
                    self.last_episode_causes[name] = torch.zeros_like(self.episode_active)
                    if name not in self.pending_causes:
                        self.pending_causes[name] = torch.zeros_like(self.scales)
                        self.last_cause_fractions[name] = torch.zeros_like(self.scales)
            self.last_episode_causes[name][env_ids] = value
            if self.enabled:
                self.pending_causes[name].index_add_(0, skills, value.float())
        if self.enabled:
            self.pending_count.index_add_(0, skills, torch.ones_like(skills))
            self.pending_survival.index_add_(0, skills, survival)
            self.pending_failures.index_add_(0, skills, failed.float())
            self.pending_completions.index_add_(0, skills, completed.float())
            self.total_episodes.index_add_(0, skills, torch.ones_like(skills))
        self.episode_active[env_ids] = False

    def update_iteration(self):
        """Apply bounded hysteresis once per skill after enough completed episodes."""
        cfg = self.settings
        ready = (self.pending_count >= cfg.minimum_episodes) & self.enabled
        for index in ready.nonzero(as_tuple=False).flatten().tolist():
            count = int(self.pending_count[index].item())
            mean = self.pending_survival[index] / count
            alpha = -math.expm1(-count / cfg.ema_episodes)
            if self.ema_initialized[index]:
                self.survival_ema[index].lerp_(mean, alpha)
            else:
                self.survival_ema[index] = mean
                self.ema_initialized[index] = True
            if self.survival_ema[index] < cfg.relax_below:
                self.scales[index] *= cfg.relax_factor
            elif self.survival_ema[index] > cfg.tighten_above:
                self.scales[index] *= cfg.tighten_factor
            self.scales[index].clamp_(cfg.minimum_scale, cfg.maximum_scale)
            self.last_failure_fraction[index] = self.pending_failures[index] / count
            self.last_completion_fraction[index] = self.pending_completions[index] / count
            for name, sums in self.pending_causes.items():
                self.last_cause_fractions[name][index] = sums[index] / count
                sums[index] = 0
            for name in ("pending_count", "pending_survival", "pending_failures", "pending_completions"):
                getattr(self, name)[index] = 0
        logs = {}
        for index, name in enumerate(self.skill_names):
            prefix = f"StudentTermination/{name}"
            scale = float(self.scales[index].item()) if self.enabled else 1.0
            logs.update({f"{prefix}/scale": scale,
                         f"{prefix}/survival_ema": float(self.survival_ema[index].item()),
                         f"{prefix}/failure_fraction": float(self.last_failure_fraction[index].item()),
                         f"{prefix}/completion_fraction": float(self.last_completion_fraction[index].item()),
                         f"{prefix}/episodes": int(self.total_episodes[index].item()),
                         f"{prefix}/pending_episodes": int(self.pending_count[index].item())})
            for term, threshold in self.base_thresholds.items():
                logs[f"{prefix}/threshold/{term}"] = threshold * scale
            for term, fraction in self.last_cause_fractions.items():
                logs[f"{prefix}/termination/{term}"] = float(fraction[index].item())
        return logs

    _STATE_FIELDS = ("scales", "survival_ema", "ema_initialized", "pending_count", "pending_survival",
                     "pending_failures", "pending_completions", "total_episodes", "last_failure_fraction",
                     "last_completion_fraction")

    def state_dict(self):
        return {"version": 1, "skill_names": self.skill_names,
                **{name: getattr(self, name).detach().cpu().clone() for name in self._STATE_FIELDS},
                "pending_causes": {k: v.detach().cpu().clone() for k, v in self.pending_causes.items()},
                "last_cause_fractions": {k: v.detach().cpu().clone() for k, v in self.last_cause_fractions.items()}}

    def load_state_dict(self, state):
        if state.get("version") != 1 or tuple(state["skill_names"]) != self.skill_names:
            raise ValueError("Student termination curriculum skill mapping or checkpoint version differs.")
        for name in self._STATE_FIELDS:
            target = getattr(self, name)
            value = state[name].to(device=target.device, dtype=target.dtype)
            if value.shape != target.shape or not torch.isfinite(value).all():
                raise ValueError(f"Invalid Student termination curriculum state: {name}.")
            target.copy_(value)
        for name in ("pending_causes", "last_cause_fractions"):
            setattr(self, name, {k: v.to(self.scales).clone() for k, v in state[name].items()})
        self.episode_active.zero_()


def get_student_termination_curriculum(env, motion):
    """Lazily create Student state; Teacher environments never get a controller."""
    settings = getattr(getattr(env, "cfg", None), "student_termination", None)
    if settings is None:
        return None
    controller = getattr(env, "student_termination_curriculum", None)
    if controller is None:
        thresholds = {}
        for name, field in (("root_tracking_error", "maximum_error"), ("dof_tracking_error", "maximum_rmse"),
                            ("interaction_tracking_error", "maximum_distance")):
            term_cfg = getattr(env.cfg.terminations, name, None)
            if term_cfg is not None:
                thresholds[name] = term_cfg.params[field]
        controller = StudentTerminationCurriculum(env.num_envs,
            [skill["name"] for skill in motion.skill_config["skills"]], env.device, settings, thresholds)
        env.student_termination_curriculum = controller
    return controller
