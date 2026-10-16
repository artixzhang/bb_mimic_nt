"""Regression coverage for feedforward PPO minibatch indexing without Isaac Sim."""

from types import SimpleNamespace

import pytest
import torch

TensorDict = pytest.importorskip("tensordict").TensorDict
RolloutStorage = pytest.importorskip("rsl_rl.storage.rollout_storage").RolloutStorage

from bb_mimic_nt.ppo_minibatches import install_cpu_minibatch_sampler


STORAGE_FIELDS = (
    "actions", "rewards", "dones", "values", "actions_log_prob", "mu", "sigma", "returns", "advantages",
)
BATCH_FIELDS = ("actions", "values", "advantages", "returns", "actions_log_prob", "mu", "sigma")


def _storage(*, steps=3, envs=4, device="cpu"):
    observations = TensorDict(
        {"policy": torch.zeros(envs, 3, device=device), "critic": torch.zeros(envs, 5, device=device)},
        batch_size=[envs],
        device=device,
    )
    storage = RolloutStorage("rl", envs, steps, observations, [2], device=device)
    row = torch.arange(steps * envs, device=device).reshape(steps, envs, 1)
    for index, field in enumerate(STORAGE_FIELDS):
        value = getattr(storage, field)
        value.copy_((row + index * 20).expand_as(value))
    storage.observations["policy"].copy_(row.expand(steps, envs, 3))
    storage.observations["critic"].copy_((row + 200).expand(steps, envs, 5))
    storage.step = steps
    return storage


def _install(storage, *, recurrent=False):
    return install_cpu_minibatch_sampler(
        SimpleNamespace(storage=storage, policy=SimpleNamespace(is_recurrent=recurrent))
    )


@pytest.mark.parametrize("steps, envs, batches", [(3, 4, 3), (3, 5, 4), (1, 1, 1)])
def test_batches_match_rsl_rl_and_preserve_field_alignment(steps, envs, batches):
    storage = _storage(steps=steps, envs=envs)
    torch.manual_seed(83)
    expected = list(storage.mini_batch_generator(batches, num_epochs=2))
    assert _install(storage)
    torch.manual_seed(83)
    actual = list(storage.mini_batch_generator(batches, num_epochs=2))

    assert len(actual) == len(expected) == 2 * batches
    for result, reference in zip(actual, expected):
        assert len(result) == 10
        assert isinstance(result[0], TensorDict)
        for key in ("policy", "critic"):
            assert torch.equal(result[0][key], reference[0][key])
        ids = result[0]["policy"][:, 0].long()
        assert torch.equal(result[0]["critic"][:, 0], ids + 200)
        for index, field in enumerate(BATCH_FIELDS, start=1):
            assert torch.equal(result[index], reference[index])
            assert torch.equal(result[index], getattr(storage, field).flatten(0, 1)[ids])
        assert result[8] == (None, None)
        assert result[9] is None


def test_cpu_permutation_is_created_once_and_reused_for_all_epochs(monkeypatch):
    storage = _storage()
    original_randperm = torch.randperm
    calls = []

    def record_randperm(count, **kwargs):
        calls.append((count, kwargs.get("device")))
        return original_randperm(count, **kwargs)

    monkeypatch.setattr(torch, "randperm", record_randperm)
    assert _install(storage)
    batches = list(storage.mini_batch_generator(3, num_epochs=4))
    assert len(calls) == 1
    assert calls[0][0] == 12
    assert torch.device(calls[0][1]).type == "cpu"
    first_epoch = torch.cat([batch[0]["policy"][:, 0] for batch in batches[:3]])
    assert sorted(first_epoch.tolist()) == list(range(12))
    for epoch in range(1, 4):
        values = torch.cat([batch[0]["policy"][:, 0] for batch in batches[epoch * 3:(epoch + 1) * 3]])
        assert torch.equal(values, first_epoch)


def test_nondivisible_tail_and_first_last_permitted_indices(monkeypatch):
    storage = _storage(steps=3, envs=5)
    monkeypatch.setattr(torch, "randperm", lambda count, **kwargs: torch.arange(count - 1, -1, -1))
    assert _install(storage)
    batches = list(storage.mini_batch_generator(4, num_epochs=1))
    ids = torch.cat([batch[0]["policy"][:, 0] for batch in batches])
    assert ids.tolist() == list(range(11, -1, -1))
    assert all(batch[0].batch_size == torch.Size([3]) for batch in batches)
    # Match RSL-RL: truncate before shuffling, leaving rows 12, 13 and 14 unused.
    assert ids.max().item() == 11
    assert ids.min().item() == 0


@pytest.mark.parametrize("field", STORAGE_FIELDS)
def test_every_storage_field_is_checked_before_sampling(monkeypatch, field):
    storage = _storage()
    setattr(storage, field, getattr(storage, field)[:-1])
    assert _install(storage)
    monkeypatch.setattr(torch, "randperm", lambda *args, **kwargs: pytest.fail("sampled invalid storage"))
    with pytest.raises((ValueError, RuntimeError), match=field):
        next(storage.mini_batch_generator(3, num_epochs=1))


@pytest.mark.parametrize("kind", ["batch", "leaf"])
def test_observation_shape_mismatch_fails_before_sampling(monkeypatch, kind):
    storage = _storage()
    if kind == "batch":
        storage.observations = storage.observations[:-1]
    else:
        # TensorDict metadata can become stale if a contained tensor is resized.
        storage.observations["critic"].resize_(3, 3, 5)
    assert _install(storage)
    monkeypatch.setattr(torch, "randperm", lambda *args, **kwargs: pytest.fail("sampled invalid observations"))
    with pytest.raises((ValueError, RuntimeError), match="observation"):
        next(storage.mini_batch_generator(3, num_epochs=1))


@pytest.mark.parametrize("filled", [0, 2, 4])
def test_incomplete_or_overfull_rollout_fails_before_sampling(monkeypatch, filled):
    storage = _storage()
    storage.step = filled
    assert _install(storage)
    monkeypatch.setattr(torch, "randperm", lambda *args, **kwargs: pytest.fail("sampled incomplete rollout"))
    with pytest.raises((ValueError, RuntimeError), match="rollout|step|filled"):
        next(storage.mini_batch_generator(3, num_epochs=1))


@pytest.mark.parametrize("bad_index", [-1, 12])
def test_out_of_bounds_permutation_is_rejected_before_gather(monkeypatch, bad_index):
    storage = _storage()

    def invalid_randperm(count, **kwargs):
        indices = torch.arange(count)
        indices[0] = bad_index
        return indices

    monkeypatch.setattr(torch, "randperm", invalid_randperm)
    assert _install(storage)
    with pytest.raises((ValueError, RuntimeError), match="index|indices|permutation"):
        next(storage.mini_batch_generator(3, num_epochs=1))


@pytest.mark.parametrize("kind", ["dtype", "shape"])
def test_malformed_permutation_is_rejected_before_gather(monkeypatch, kind):
    storage = _storage()

    def invalid_randperm(count, **kwargs):
        indices = torch.arange(count)
        return indices.float() if kind == "dtype" else indices[:, None]

    monkeypatch.setattr(torch, "randperm", invalid_randperm)
    assert _install(storage)
    with pytest.raises(RuntimeError, match="Invalid PPO permutation.*int64 shape"):
        next(storage.mini_batch_generator(3, num_epochs=1))


@pytest.mark.parametrize(
    "batches, epochs", [(0, 1), (-1, 1), (13, 1), (3, 0), (3, -1), (1.5, 1), (3, True)]
)
def test_invalid_batch_configuration_fails_clearly(batches, epochs):
    storage = _storage()
    assert _install(storage)
    with pytest.raises((ValueError, RuntimeError), match="batch|epoch"):
        next(storage.mini_batch_generator(batches, num_epochs=epochs))


def test_recurrent_policy_keeps_existing_sampler():
    storage = _storage()
    original = storage.mini_batch_generator
    assert not _install(storage, recurrent=True)
    assert storage.mini_batch_generator == original


def test_installation_is_scoped_to_one_storage_instance():
    storage = _storage()
    other = _storage()
    original_method = RolloutStorage.mini_batch_generator
    original_randperm = torch.randperm
    assert _install(storage)
    assert storage.mini_batch_generator.__func__ is not original_method
    assert other.mini_batch_generator.__func__ is original_method
    assert RolloutStorage.mini_batch_generator is original_method
    assert torch.randperm is original_randperm


def test_real_ppo_collects_updates_and_reuses_cleared_storage(monkeypatch):
    ActorCritic = pytest.importorskip("rsl_rl.modules").ActorCritic
    PPO = pytest.importorskip("rsl_rl.algorithms").PPO
    # ActorCritic changes this process-wide PyTorch setting; restore it after the test.
    distribution = torch.distributions.Distribution
    monkeypatch.setattr(distribution, "_validate_args", distribution._validate_args)
    torch.manual_seed(91)
    envs, steps = 4, 3

    def observation():
        return TensorDict(
            {"policy": torch.randn(envs, 3), "critic": torch.randn(envs, 5)}, batch_size=[envs]
        )

    obs = observation()
    policy = ActorCritic(
        obs, {"policy": ["policy"], "critic": ["critic"]}, num_actions=2,
        actor_hidden_dims=[8], critic_hidden_dims=[8],
        actor_obs_normalization=True, critic_obs_normalization=True,
    )
    algorithm = PPO(policy, num_learning_epochs=2, num_mini_batches=3, device="cpu")
    algorithm.init_storage("rl", envs, steps, obs, [2])
    assert install_cpu_minibatch_sampler(algorithm)
    for _ in range(2):
        before = [parameter.detach().clone() for parameter in policy.parameters()]
        with torch.inference_mode():
            for step in range(steps):
                actions = algorithm.act(obs)
                assert actions.shape == (envs, 2)
                obs = observation()
                rewards = 1.0 - actions.square().mean(dim=-1)
                dones = torch.tensor([step == steps - 1, False, False, False])
                algorithm.process_env_step(obs, rewards, dones, {})
            algorithm.compute_returns(obs)
        assert algorithm.storage.step == steps
        losses = algorithm.update()
        assert {"value_function", "surrogate", "entropy"} <= losses.keys()
        assert all(torch.isfinite(torch.tensor(value)) for value in losses.values())
        assert algorithm.storage.step == 0
        assert any(not torch.equal(old, new) for old, new in zip(before, policy.parameters()))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_cuda_storage_uses_cpu_permutation_and_preserves_device(monkeypatch):
    device = f"cuda:{torch.cuda.device_count() - 1}"
    storage = _storage(device=device)
    original_randperm = torch.randperm
    devices = []

    def record_randperm(count, **kwargs):
        devices.append(torch.device(kwargs["device"]).type)
        return original_randperm(count, **kwargs)

    monkeypatch.setattr(torch, "randperm", record_randperm)
    assert _install(storage)
    with torch.cuda.device(0):
        batches = list(storage.mini_batch_generator(3, num_epochs=2))
    assert devices == ["cpu"]
    assert all(batch[0]["policy"].device == torch.device(device) for batch in batches)
    ids = torch.cat([batch[0]["policy"][:, 0].cpu() for batch in batches[:3]])
    assert sorted(ids.tolist()) == list(range(12))
