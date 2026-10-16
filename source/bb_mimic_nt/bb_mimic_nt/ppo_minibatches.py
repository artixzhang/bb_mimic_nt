"""CPU permutation workaround for RSL-RL 3.0 feedforward PPO minibatches.

The observed CUDA assertion occurs while PPO consumes its first minibatch. A
bad CUDA permutation is a suspected cause, not a confirmed upstream defect.
This instance-local workaround avoids CUDA permutation generation and checks
the transferred index bounds before any gather can poison the CUDA context.

The batch layout, truncated sample count, and permutation reuse across epochs
match ``rsl_rl.storage.RolloutStorage.mini_batch_generator`` in RSL-RL 3.0.1.
No simulator or RSL-RL imports are needed to install or test this workaround.
"""

from __future__ import annotations

import os
from contextlib import nullcontext
from operator import index
from types import MethodType

import torch


_RL_FIELDS = (
    "actions", "rewards", "dones", "values", "actions_log_prob", "mu",
    "sigma", "returns", "advantages",
)


def _location(storage) -> str:
    return f"rank={os.environ.get('RANK', '0')}, device={storage.device}"


def _positive_int(value, name: str) -> int:
    try:
        result = index(value)
    except TypeError as exc:
        raise ValueError(f"{name} must be a positive integer, got {value!r}.") from exc
    if isinstance(value, bool) or result < 1:
        raise ValueError(f"{name} must be a positive integer, got {value!r}.")
    return result


def _check_tensor_shape(storage, value, name: str, expected: tuple[int, int]) -> None:
    if not isinstance(value, torch.Tensor) or tuple(value.shape[:2]) != expected:
        shape = getattr(value, "shape", None)
        raise RuntimeError(
            f"Invalid PPO rollout field {name!r} ({_location(storage)}): "
            f"expected a tensor with leading dimensions {expected}, got {shape}."
        )


def _check_observation_leaves(storage, observations, expected: tuple[int, int], prefix="observations") -> None:
    for key, value in observations.items():
        name = f"{prefix}.{key}"
        if isinstance(value, torch.Tensor):
            _check_tensor_shape(storage, value, name, expected)
        elif hasattr(value, "items"):
            _check_observation_leaves(storage, value, expected, name)
        else:
            raise RuntimeError(
                f"Invalid PPO rollout field {name!r} ({_location(storage)}): "
                "expected a tensor or nested observation mapping."
            )


def _check_rollout(storage) -> int:
    if getattr(storage, "training_type", None) != "rl":
        raise RuntimeError(
            f"CPU PPO minibatches require RSL-RL 3.0 reinforcement-learning storage "
            f"({_location(storage)}); got training_type={getattr(storage, 'training_type', None)!r}."
        )
    num_steps = _positive_int(storage.num_transitions_per_env, "num_transitions_per_env")
    num_envs = _positive_int(storage.num_envs, "num_envs")
    expected = (num_steps, num_envs)
    if getattr(storage, "step", None) != num_steps:
        raise RuntimeError(
            f"Incomplete PPO rollout ({_location(storage)}): "
            f"expected {num_steps} stored steps, got {getattr(storage, 'step', None)}."
        )
    observations = getattr(storage, "observations", None)
    batch_size = getattr(observations, "batch_size", None)
    if batch_size is None or tuple(batch_size) != expected or not callable(getattr(observations, "flatten", None)):
        raise RuntimeError(
            f"Invalid PPO observations ({_location(storage)}): expected RSL-RL 3.0 "
            f"TensorDict batch dimensions {expected}, got {batch_size}."
        )
    _check_observation_leaves(storage, observations, expected)
    for name in _RL_FIELDS:
        _check_tensor_shape(storage, getattr(storage, name, None), name, expected)
    return num_steps * num_envs


def _cpu_mini_batch_generator(storage, num_mini_batches, num_epochs=8):
    num_mini_batches = _positive_int(num_mini_batches, "num_mini_batches")
    num_epochs = _positive_int(num_epochs, "num_epochs")
    batch_size = _check_rollout(storage)
    mini_batch_size = batch_size // num_mini_batches
    if mini_batch_size == 0:
        raise ValueError(
            f"num_mini_batches={num_mini_batches} exceeds PPO rollout size {batch_size} "
            f"({_location(storage)})."
        )
    sample_count = num_mini_batches * mini_batch_size
    cpu_indices = torch.randperm(sample_count, requires_grad=False, device="cpu")
    device = torch.device(storage.device)
    device_context = torch.cuda.device(device) if device.type == "cuda" else nullcontext()
    with device_context:
        indices = cpu_indices.to(device=device, non_blocking=False)
        if indices.dtype != torch.long or tuple(indices.shape) != (sample_count,):
            raise RuntimeError(
                f"Invalid PPO permutation ({_location(storage)}): expected int64 shape "
                f"({sample_count},), got dtype={indices.dtype}, shape={tuple(indices.shape)}."
            )
        minimum, maximum = torch.aminmax(indices)
        low, high = int(minimum.item()), int(maximum.item())
        if low < 0 or high >= sample_count:
            raise RuntimeError(
                f"Invalid PPO permutation ({_location(storage)}): index range [{low}, {high}] "
                f"is outside [0, {sample_count - 1}] for rollout size {batch_size}. "
                "Stopped before minibatch indexing; indices were not clamped or retried."
            )

    observations = storage.observations.flatten(0, 1)
    actions = storage.actions.flatten(0, 1)
    values = storage.values.flatten(0, 1)
    returns = storage.returns.flatten(0, 1)
    old_actions_log_prob = storage.actions_log_prob.flatten(0, 1)
    advantages = storage.advantages.flatten(0, 1)
    old_mu = storage.mu.flatten(0, 1)
    old_sigma = storage.sigma.flatten(0, 1)

    for _ in range(num_epochs):
        for batch in range(num_mini_batches):
            start = batch * mini_batch_size
            batch_idx = indices[start : start + mini_batch_size]
            yield (
                observations[batch_idx],
                actions[batch_idx],
                values[batch_idx],
                advantages[batch_idx],
                returns[batch_idx],
                old_actions_log_prob[batch_idx],
                old_mu[batch_idx],
                old_sigma[batch_idx],
                (None, None),
                None,
            )


def install_cpu_minibatch_sampler(algorithm) -> bool:
    """Patch only this algorithm's feedforward storage; return whether installed.

    Recurrent policies keep their trajectory-aware sampler. Other runners and
    the installed RSL-RL package are unchanged. The rollout is validated when
    the generator runs, after collection has filled the storage.
    """
    if algorithm.policy.is_recurrent:
        return False
    storage = algorithm.storage
    if storage is None or getattr(storage, "training_type", None) != "rl":
        raise RuntimeError("CPU PPO minibatches require initialized RSL-RL 3.0 reinforcement-learning storage.")
    storage.mini_batch_generator = MethodType(_cpu_mini_batch_generator, storage)
    return True
