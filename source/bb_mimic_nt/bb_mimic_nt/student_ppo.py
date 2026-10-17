"""On-policy Student PPO with frozen Teacher supervision (RSL-RL 3.0.1)."""

from __future__ import annotations

import math
from collections.abc import Callable

import torch
from rsl_rl.algorithms import PPO
from rsl_rl.storage import RolloutStorage

from bb_mimic_nt.action_math import clip_joint_actions


BC_TARGET = "bc_target"


class FullEpochRolloutStorage(RolloutStorage):
    """Shuffle every rollout row once per epoch, including a short final batch."""

    def full_epoch_generator(self, batch_size: int, num_epochs: int):
        if self.training_type != "rl" or self.step != self.num_transitions_per_env:
            raise RuntimeError("PPO+BC requires a complete reinforcement-learning rollout.")
        if batch_size < 1 or num_epochs < 1:
            raise ValueError("Batch size and epoch count must be positive.")
        count = self.num_envs * self.num_transitions_per_env
        observations = self.observations.flatten(0, 1)
        fields = [getattr(self, name).flatten(0, 1) for name in (
            "actions", "values", "advantages", "returns", "actions_log_prob", "mu", "sigma",
        )]
        for _ in range(num_epochs):
            # Keep the project's CPU permutation workaround, without its dropped tail.
            indices = torch.randperm(count, device="cpu").to(self.device)
            if indices.dtype != torch.long or indices.shape != (count,):
                raise RuntimeError("Invalid PPO+BC minibatch permutation.")
            minimum, maximum = torch.aminmax(indices)
            if minimum.item() < 0 or maximum.item() >= count:
                raise RuntimeError("PPO+BC minibatch permutation is outside the rollout.")
            for start in range(0, count, batch_size):
                selected = indices[start:start + batch_size]
                yield observations[selected], *(field[selected] for field in fields), (None, None), None


class PPOWithBC(PPO):
    """Joint PPO/BC updates; environment actions always come from the Student.

    The inherited process_env_step/compute_returns deliberately preserve RSL-RL
    3.0.1's timeout bootstrap convention (gamma * the pre-step value).
    """

    def __init__(
        self, policy, teacher, *, batch_size: int = 8192, bc_coef: float = 1.0,
        action_clip: float = 100.0, action_scale=None,
        iteration_callback: Callable[[int], None] | None = None,
        num_learning_epochs: int = 5, learning_rate: float = 3.0e-4, max_grad_norm: float = 1.0,
        actor_learning_rate: float | None = None, critic_learning_rate: float | None = None,
        actor_max_grad_norm: float | None = None, critic_max_grad_norm: float | None = None,
        **kwargs,
    ):
        if policy.is_recurrent:
            raise ValueError("Student PPO+BC currently requires a feed-forward policy.")
        if kwargs.get("rnd_cfg") is not None or kwargs.get("symmetry_cfg") is not None:
            raise ValueError("Student PPO+BC does not support RND or symmetry augmentation.")
        if isinstance(batch_size, bool) or int(batch_size) != batch_size or batch_size < 1:
            raise ValueError("PPO+BC batch_size must be a positive integer.")
        if isinstance(num_learning_epochs, bool) or int(num_learning_epochs) != num_learning_epochs or num_learning_epochs < 1:
            raise ValueError("PPO+BC num_learning_epochs must be a positive integer.")
        if not math.isfinite(bc_coef) or bc_coef < 0:
            raise ValueError("PPO+BC bc_coef must be finite and non-negative.")
        if not math.isfinite(action_clip) or action_clip <= 0:
            raise ValueError("PPO+BC action_clip must be finite and positive.")
        rates = (learning_rate if actor_learning_rate is None else actor_learning_rate,
                 learning_rate if critic_learning_rate is None else critic_learning_rate)
        limits = (max_grad_norm if actor_max_grad_norm is None else actor_max_grad_norm,
                  max_grad_norm if critic_max_grad_norm is None else critic_max_grad_norm)
        if any(not math.isfinite(rate) or rate <= 0 for rate in rates):
            raise ValueError("Actor and critic learning rates must be finite and positive.")
        # Positive infinity explicitly disables clipping, as in torch.clip_grad_norm_.
        if any(math.isnan(limit) or limit <= 0 for limit in limits):
            raise ValueError("Actor and critic gradient norm limits must be positive.")
        super().__init__(policy, num_learning_epochs=num_learning_epochs,
                         learning_rate=learning_rate, max_grad_norm=max_grad_norm, **kwargs)
        self.actor_parameters = [parameter for parameter in policy.actor.parameters() if parameter.requires_grad]
        self.actor_parameters.extend(
            parameter for name in ("std", "log_std")
            if isinstance(parameter := getattr(policy, name, None), torch.nn.Parameter) and parameter.requires_grad
        )
        self.critic_parameters = [parameter for parameter in policy.critic.parameters() if parameter.requires_grad]
        actor_ids = {id(parameter) for parameter in self.actor_parameters}
        critic_ids = {id(parameter) for parameter in self.critic_parameters}
        policy_ids = {id(parameter) for parameter in policy.parameters() if parameter.requires_grad}
        if actor_ids & critic_ids or actor_ids | critic_ids != policy_ids or not actor_ids or not critic_ids:
            raise ValueError("Student actor/std and critic must form disjoint, complete parameter groups.")
        self.actor_learning_rate, self.critic_learning_rate = map(float, rates)
        self.actor_max_grad_norm, self.critic_max_grad_norm = map(float, limits)
        self.actor_optimizer = torch.optim.Adam(self.actor_parameters, lr=self.actor_learning_rate)
        self.critic_optimizer = torch.optim.Adam(self.critic_parameters, lr=self.critic_learning_rate)
        # A shared optimizer would silently reintroduce the old checkpoint/update path.
        del self.optimizer
        self.learning_rate = self.actor_learning_rate
        self.teacher = teacher.to(self.device).eval().requires_grad_(False)
        self.batch_size = int(batch_size)
        self.bc_coef = float(bc_coef)
        self.action_clip = float(action_clip)
        self.action_scale = None if action_scale is None else torch.as_tensor(action_scale, device=self.device)
        self.iteration_callback = iteration_callback
        self.completed_iterations = 0
        self.last_update_statistics = {}

    @property
    def learning_rate(self):
        """Keep the upstream runner's single-rate log tied to the actor rate."""
        return self.actor_learning_rate

    @learning_rate.setter
    def learning_rate(self, value):
        self.actor_learning_rate = value

    def init_storage(self, training_type, num_envs, num_transitions_per_env, obs, actions_shape):
        if training_type != "rl" or BC_TARGET in obs.keys():
            raise ValueError("PPO+BC requires RL storage and a reserved, unused bc_target observation key.")
        stored_obs = obs.clone(recurse=False)
        stored_obs[BC_TARGET] = torch.zeros(num_envs, *actions_shape, device=self.device)
        self.storage = FullEpochRolloutStorage(
            "rl", num_envs, num_transitions_per_env, stored_obs, actions_shape, self.device,
        )

    def act(self, obs):
        actions = super().act(obs)
        with torch.no_grad():
            labels = clip_joint_actions(self.teacher(obs), self.action_clip).detach()
        if labels.shape != actions.shape:
            raise ValueError("Teacher labels must match the Student action shape.")
        # Adding a key to this shallow TensorDict copy must not mutate env observations.
        stored_obs = obs.clone(recurse=False)
        stored_obs[BC_TARGET] = labels
        self.transition.observations = stored_obs
        return actions

    def _adapt_learning_rate(self, mean, std, old_mean, old_std):
        with torch.inference_mode():
            local_kl = torch.sum(
                torch.log(std / old_std + 1.0e-5)
                + (old_std.square() + (old_mean - mean).square()) / (2.0 * std.square()) - 0.5,
                dim=-1,
            ).mean()
            if self.desired_kl is None or self.schedule != "adaptive":
                return local_kl
            kl = local_kl.clone()
            if self.is_multi_gpu:
                torch.distributed.all_reduce(kl)
                kl /= self.gpu_world_size
            if self.gpu_global_rank == 0:
                if kl > self.desired_kl * 2.0:
                    self.actor_learning_rate = max(1.0e-5, self.actor_learning_rate / 1.5)
                    self.critic_learning_rate = max(1.0e-5, self.critic_learning_rate / 1.5)
                elif 0.0 < kl < self.desired_kl / 2.0:
                    self.actor_learning_rate = min(1.0e-2, self.actor_learning_rate * 1.5)
                    self.critic_learning_rate = min(1.0e-2, self.critic_learning_rate * 1.5)
            if self.is_multi_gpu:
                rates = torch.tensor([self.actor_learning_rate, self.critic_learning_rate], device=self.device)
                torch.distributed.broadcast(rates, src=0)
                self.actor_learning_rate, self.critic_learning_rate = rates.tolist()
            self.learning_rate = self.actor_learning_rate
            for optimizer, rate in ((self.actor_optimizer, self.actor_learning_rate),
                                    (self.critic_optimizer, self.critic_learning_rate)):
                for group in optimizer.param_groups:
                    group["lr"] = rate
            return local_kl

    def update(self):
        sums = {name: 0.0 for name in ("value_function", "surrogate", "entropy", "behavior_cloning", "kl")}
        if self.action_scale is not None:
            sums["target_mae_rad"] = 0.0
        consumed = updates = 0
        actor_norm_sum = critic_norm_sum = 0.0
        actor_clipped = critic_clipped = 0
        for obs, actions, old_value, advantage, returns, old_log_prob, old_mean, old_std, _, _ in (
            self.storage.full_epoch_generator(self.batch_size, self.num_learning_epochs)
        ):
            if self.normalize_advantage_per_mini_batch:
                with torch.no_grad():
                    # A one-row tail still has a well-defined normalization.
                    correction = 1 if advantage.numel() > 1 else 0
                    advantage = (advantage - advantage.mean()) / (advantage.std(correction=correction) + 1.0e-8)
            self.policy.act(obs)
            log_prob = self.policy.get_actions_log_prob(actions)
            value = self.policy.evaluate(obs)
            mean, std, entropy = self.policy.action_mean, self.policy.action_std, self.policy.entropy.mean()
            kl = self._adapt_learning_rate(mean, std, old_mean, old_std)
            ratio = torch.exp(log_prob - old_log_prob.squeeze(-1))
            surrogate = torch.maximum(
                -advantage.squeeze(-1) * ratio,
                -advantage.squeeze(-1) * ratio.clamp(1.0 - self.clip_param, 1.0 + self.clip_param),
            ).mean()
            if self.use_clipped_value_loss:
                clipped = old_value + (value - old_value).clamp(-self.clip_param, self.clip_param)
                value_loss = torch.maximum((value - returns).square(), (clipped - returns).square()).mean()
            else:
                value_loss = (returns - value).square().mean()
            bc_loss = torch.nn.functional.mse_loss(mean, obs[BC_TARGET])
            loss = surrogate + self.value_loss_coef * value_loss - self.entropy_coef * entropy + self.bc_coef * bc_loss
            self.actor_optimizer.zero_grad(set_to_none=True)
            self.critic_optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if self.is_multi_gpu:
                self.reduce_parameters()
            actor_norm = torch.nn.utils.clip_grad_norm_(self.actor_parameters, self.actor_max_grad_norm).item()
            critic_norm = torch.nn.utils.clip_grad_norm_(self.critic_parameters, self.critic_max_grad_norm).item()
            self.actor_optimizer.step()
            self.critic_optimizer.step()
            size = actions.shape[0]
            actor_norm_sum += actor_norm * size
            critic_norm_sum += critic_norm * size
            actor_clipped += actor_norm > self.actor_max_grad_norm
            critic_clipped += critic_norm > self.critic_max_grad_norm
            for name, metric in (("value_function", value_loss), ("surrogate", surrogate),
                                 ("entropy", entropy), ("behavior_cloning", bc_loss), ("kl", kl)):
                sums[name] += metric.detach().item() * size
            if self.action_scale is not None:
                with torch.no_grad():
                    delta = (clip_joint_actions(mean, self.action_clip) - obs[BC_TARGET]) * self.action_scale
                    sums["target_mae_rad"] += delta.abs().mean().item() * size
            consumed += size
            updates += 1
        count = self.storage.num_envs * self.storage.num_transitions_per_env
        self.last_update_statistics = {
            "rollout_samples": count, "samples_consumed": consumed,
            "epochs_completed": self.num_learning_epochs, "optimizer_updates": updates,
            "actor_grad_norm": actor_norm_sum / consumed,
            "critic_grad_norm": critic_norm_sum / consumed,
            "actor_clip_fraction": actor_clipped / updates,
            "critic_clip_fraction": critic_clipped / updates,
            "actor_learning_rate": self.actor_learning_rate,
            "critic_learning_rate": self.critic_learning_rate,
        }
        self.storage.clear()
        self.completed_iterations += 1
        if self.iteration_callback is not None:
            self.iteration_callback(self.completed_iterations)
        return {name: total / consumed for name, total in sums.items()}
