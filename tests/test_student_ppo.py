"""CPU invariants for Student PPO+BC and complete rollout epochs."""

from copy import deepcopy

import pytest
import torch

TensorDict = pytest.importorskip("tensordict").TensorDict
ActorCritic = pytest.importorskip("rsl_rl.modules").ActorCritic
PPO = pytest.importorskip("rsl_rl.algorithms").PPO

from bb_mimic_nt.student_ppo import BC_TARGET, FullEpochRolloutStorage, PPOWithBC


class Teacher(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = torch.nn.Linear(3, 2)

    def forward(self, obs):
        return self.layer(obs["teacher"])


def make_algorithm(*, envs=3, steps=4, **kwargs):
    obs = TensorDict({"policy": torch.randn(envs, 2), "teacher": torch.randn(envs, 3)}, [envs])
    policy = ActorCritic(
        obs, {"policy": ["policy"], "critic": ["teacher"]}, 2,
        actor_hidden_dims=[8], critic_hidden_dims=[8], init_noise_std=.8, noise_std_type="log",
    )
    algorithm = PPOWithBC(policy, Teacher(), device="cpu", schedule=kwargs.pop("schedule", "fixed"), **kwargs)
    algorithm.init_storage("rl", envs, steps, obs, [2])
    return algorithm, obs


def collect(algorithm, obs):
    for _ in range(algorithm.storage.num_transitions_per_env):
        with torch.inference_mode():
            algorithm.act(obs)
            algorithm.process_env_step(obs, torch.ones(obs.shape[0]), torch.zeros(obs.shape[0]), {})
    with torch.inference_mode():
        algorithm.compute_returns(obs)


def test_five_epochs_cover_every_sample_and_keep_labels_aligned(monkeypatch):
    algorithm, obs = make_algorithm(batch_size=5, num_learning_epochs=5)
    collect(algorithm, obs)
    storage = algorithm.storage
    ids = torch.arange(12).reshape(4, 3, 1)
    storage.observations["policy"][..., :1] = ids
    storage.observations[BC_TARGET][..., :1] = ids + 100
    paired_fields = ("actions", "values", "advantages", "returns", "actions_log_prob", "mu", "sigma")
    with torch.inference_mode():
        for index, name in enumerate(paired_fields, start=1):
            value = getattr(storage, name)
            value.copy_((ids + 1000 * index).expand_as(value))
    calls = []
    original = torch.randperm

    def randperm(count, **kwargs):
        calls.append(kwargs["device"])
        return original(count, **kwargs)

    monkeypatch.setattr(torch, "randperm", randperm)
    batches = list(storage.full_epoch_generator(5, 5))
    assert len(batches) == 15 and calls == ["cpu"] * 5
    permutations = []
    for epoch in range(5):
        selected = batches[3 * epoch:3 * epoch + 3]
        assert [len(batch[1]) for batch in selected] == [5, 5, 2]
        order = torch.cat([batch[0]["policy"][:, 0] for batch in selected])
        assert sorted(order.tolist()) == list(range(12))
        permutations.append(order)
        for batch in selected:
            torch.testing.assert_close(batch[0][BC_TARGET][:, 0], batch[0]["policy"][:, 0] + 100)
            row_ids = batch[0]["policy"][:, :1]
            for index, name in enumerate(paired_fields, start=1):
                expected = (row_ids + 1000 * index).expand_as(batch[index])
                torch.testing.assert_close(batch[index], expected, msg=f"Misaligned rollout field: {name}")
    assert any(not torch.equal(permutations[0], order) for order in permutations[1:])


def test_teacher_labels_are_cached_pre_step_and_teacher_never_trains():
    algorithm, obs = make_algorithm(batch_size=5, num_learning_epochs=5)
    teacher_before = deepcopy(algorithm.teacher.state_dict())
    expected = algorithm.teacher(obs).detach().clone()
    with torch.inference_mode():
        action = algorithm.act(obs)
        assert BC_TARGET not in obs.keys()
        torch.testing.assert_close(algorithm.transition.observations[BC_TARGET], expected)
        assert action.data_ptr() == algorithm.transition.actions.data_ptr()
        algorithm.process_env_step(obs, torch.ones(3), torch.zeros(3), {})
        for _ in range(3):
            algorithm.act(obs)
            algorithm.process_env_step(obs, torch.ones(3), torch.zeros(3), {})
        algorithm.compute_returns(obs)
    algorithm.policy.train()
    losses = algorithm.update()
    assert all(torch.isfinite(torch.tensor(value)) for value in losses.values())
    assert algorithm.completed_iterations == 1
    for name, value in dict(rollout_samples=12, samples_consumed=60, epochs_completed=5, optimizer_updates=15).items():
        assert algorithm.last_update_statistics[name] == value
    for group in ("actor", "critic"):
        assert algorithm.last_update_statistics[f"{group}_grad_norm"] >= 0
        assert 0 <= algorithm.last_update_statistics[f"{group}_clip_fraction"] <= 1
    assert not algorithm.teacher.training
    assert all(not param.requires_grad and param.grad is None for param in algorithm.teacher.parameters())
    for name, value in algorithm.teacher.state_dict().items():
        torch.testing.assert_close(value, teacher_before[name])


def test_zero_bc_matches_native_ppo_update():
    algorithm, obs = make_algorithm(
        batch_size=12, num_learning_epochs=1, bc_coef=0., num_mini_batches=1, max_grad_norm=float("inf"),
    )
    collect(algorithm, obs)
    baseline = PPO(deepcopy(algorithm.policy), num_learning_epochs=1, num_mini_batches=1, schedule="fixed",
                   learning_rate=algorithm.learning_rate, max_grad_norm=float("inf"))
    baseline.storage = deepcopy(algorithm.storage)
    torch.manual_seed(19)
    expected = baseline.update()
    torch.manual_seed(19)
    actual = algorithm.update()
    for key in expected:
        assert actual[key] == pytest.approx(expected[key], rel=1e-6, abs=1e-7)
    for name, value in baseline.policy.state_dict().items():
        torch.testing.assert_close(value, algorithm.policy.state_dict()[name])


def test_native_timeout_bootstrap_and_terminal_gae_do_not_cross_reset():
    algorithm, obs = make_algorithm(envs=2, steps=1)
    for parameter in algorithm.policy.critic.parameters():
        parameter.data.zero_()
    algorithm.policy.critic[-1].bias.data.fill_(2.)
    with torch.inference_mode():
        algorithm.act(obs)
        algorithm.process_env_step(obs, torch.ones(2), torch.ones(2), {"time_outs": torch.tensor([1., 0.])})
        algorithm.compute_returns(obs)
    expected = torch.tensor([1. + algorithm.gamma * 2., 1.])
    torch.testing.assert_close(algorithm.storage.rewards[0, :, 0], expected)
    torch.testing.assert_close(algorithm.storage.returns[0, :, 0], expected)


def test_ordinary_gae_and_rollout_boundary_use_the_final_future_value(monkeypatch):
    algorithm, obs = make_algorithm(
        envs=2, steps=3, gamma=.9, lam=.8, normalize_advantage_per_mini_batch=True,
    )
    storage = algorithm.storage
    storage.values[..., 0] = torch.tensor([[1., 10.], [2., 20.], [3., 30.]])
    storage.rewards[..., 0] = torch.tensor([[.5, 1.], [1., 2.], [1.5, 3.]])
    storage.dones.zero_()
    storage.step = 3
    monkeypatch.setattr(algorithm.policy, "evaluate", lambda observations: torch.tensor([[4.], [40.]]))
    algorithm.compute_returns(obs)
    # gamma * lambda = .72. Last-step deltas are 2.1 and 9; preceding
    # advantages recursively include those deltas instead of zeroing at rollout end.
    expected_advantages = torch.tensor([[3.61264, 20.1456], [3.212, 15.48], [2.1, 9.]])
    expected_returns = torch.tensor([[4.61264, 30.1456], [5.212, 35.48], [5.1, 39.]])
    torch.testing.assert_close(storage.advantages[..., 0], expected_advantages)
    torch.testing.assert_close(storage.returns[..., 0], expected_returns)


def test_privileged_observations_do_not_change_actor_and_single_row_tail_is_finite():
    algorithm, obs = make_algorithm(envs=3, steps=3, batch_size=4, normalize_advantage_per_mini_batch=True)
    changed = obs.clone()
    changed["teacher"] += 1000
    torch.testing.assert_close(algorithm.policy.act_inference(obs), algorithm.policy.act_inference(changed))
    collect(algorithm, obs)
    assert all(torch.isfinite(torch.tensor(value)) for value in algorithm.update().values())


def test_incomplete_rollout_is_rejected():
    algorithm, _ = make_algorithm()
    with pytest.raises(RuntimeError, match="complete"):
        next(algorithm.storage.full_epoch_generator(4, 5))


def test_kl_is_sample_weighted_even_with_fixed_schedule_and_short_tail(monkeypatch):
    algorithm, obs = make_algorithm(batch_size=5, num_learning_epochs=1)
    collect(algorithm, obs)
    # Distinct per-batch means make a wrong unweighted mean detectable.
    batch_kls = iter((1., 2., 9.))
    monkeypatch.setattr(algorithm, "_adapt_learning_rate", lambda *args: torch.tensor(next(batch_kls)))
    losses = algorithm.update()
    assert losses["kl"] == pytest.approx((5 * 1. + 5 * 2. + 2 * 9.) / 12.)
    mean = torch.zeros(2, 2)
    std = torch.ones(2, 2)
    actual = PPOWithBC._adapt_learning_rate(algorithm, mean, std, mean + 1., std)
    assert actual.item() == pytest.approx(1., abs=3e-5)


def test_critic_loss_scale_does_not_change_actor_clipping_or_update():
    torch.manual_seed(42)
    algorithm, obs = make_algorithm(
        batch_size=12, num_learning_epochs=1, use_clipped_value_loss=False,
        actor_max_grad_norm=.01, critic_max_grad_norm=.02,
    )
    collect(algorithm, obs)
    stronger_critic = deepcopy(algorithm)
    stronger_critic.value_loss_coef *= 1e6
    torch.manual_seed(15)
    algorithm.update()
    torch.manual_seed(15)
    stronger_critic.update()
    for normal, stronger in zip(algorithm.actor_parameters, stronger_critic.actor_parameters):
        torch.testing.assert_close(normal.grad, stronger.grad, rtol=0, atol=0)
        torch.testing.assert_close(normal, stronger, rtol=0, atol=0)
    normal_stats = algorithm.last_update_statistics
    stronger_stats = stronger_critic.last_update_statistics
    assert normal_stats["actor_clip_fraction"] == stronger_stats["actor_clip_fraction"] == 1.
    assert normal_stats["actor_grad_norm"] == stronger_stats["actor_grad_norm"]
    assert stronger_stats["critic_grad_norm"] / normal_stats["critic_grad_norm"] == pytest.approx(1e6, rel=1e-6)
    for parameters, limit in ((algorithm.actor_parameters, .01), (algorithm.critic_parameters, .02)):
        norm = torch.linalg.vector_norm(torch.cat([parameter.grad.flatten() for parameter in parameters]))
        assert norm.item() <= limit + 1e-7


def test_optimizer_groups_are_disjoint_exclude_teacher_and_update_learned_std():
    algorithm, obs = make_algorithm(batch_size=12, num_learning_epochs=1, bc_coef=0., entropy_coef=.1)
    actor_ids = {id(parameter) for parameter in algorithm.actor_parameters}
    critic_ids = {id(parameter) for parameter in algorithm.critic_parameters}
    teacher_ids = {id(parameter) for parameter in algorithm.teacher.parameters()}
    assert not hasattr(algorithm, "optimizer")
    assert not actor_ids & critic_ids and not (actor_ids | critic_ids) & teacher_ids
    assert actor_ids | critic_ids == {id(parameter) for parameter in algorithm.policy.parameters()}
    assert id(algorithm.policy.log_std) in actor_ids
    for optimizer, expected in ((algorithm.actor_optimizer, actor_ids), (algorithm.critic_optimizer, critic_ids)):
        assert {id(parameter) for group in optimizer.param_groups for parameter in group["params"]} == expected
    collect(algorithm, obs)
    # Isolate entropy's update of std from sampled-action surrogate gradients.
    with torch.inference_mode():
        algorithm.storage.advantages.zero_()
    previous_std = algorithm.policy.log_std.detach().clone()
    algorithm.update()
    assert torch.all(algorithm.policy.log_std > previous_std)
    assert algorithm.policy.log_std in algorithm.actor_optimizer.state
    assert all(parameter.grad is None for parameter in algorithm.teacher.parameters())


def test_kl_adjusts_both_independent_learning_rates_and_respects_limits():
    algorithm, _ = make_algorithm(
        schedule="adaptive", actor_learning_rate=3e-4, critic_learning_rate=6e-4,
    )
    mean, std = torch.zeros(2, 2), torch.ones(2, 2)
    algorithm._adapt_learning_rate(mean, std, mean + 1., std)
    assert algorithm.actor_learning_rate == pytest.approx(2e-4)
    assert algorithm.critic_learning_rate == pytest.approx(4e-4)
    algorithm._adapt_learning_rate(mean, std, mean, std)
    assert algorithm.actor_learning_rate == pytest.approx(3e-4)
    assert algorithm.critic_learning_rate == pytest.approx(6e-4)
    assert algorithm.learning_rate == algorithm.actor_learning_rate
    for optimizer, rate in ((algorithm.actor_optimizer, 3e-4), (algorithm.critic_optimizer, 6e-4)):
        assert optimizer.param_groups[0]["lr"] == pytest.approx(rate)
    algorithm.actor_learning_rate, algorithm.critic_learning_rate = 1e-5, 1.2e-5
    algorithm._adapt_learning_rate(mean, std, mean + 1., std)
    assert algorithm.actor_learning_rate == algorithm.critic_learning_rate == 1e-5
    algorithm.actor_learning_rate, algorithm.critic_learning_rate = 9e-3, 8e-3
    algorithm._adapt_learning_rate(mean, std, mean, std)
    assert algorithm.actor_learning_rate == algorithm.critic_learning_rate == 1e-2


def test_separate_optimizer_defaults_inherit_shared_configuration():
    algorithm, _ = make_algorithm(learning_rate=7e-4, max_grad_norm=.3)
    assert algorithm.actor_learning_rate == algorithm.critic_learning_rate == algorithm.learning_rate == 7e-4
    assert algorithm.actor_max_grad_norm == algorithm.critic_max_grad_norm == .3


@pytest.mark.parametrize("settings", [
    {"actor_learning_rate": 0.}, {"critic_learning_rate": float("nan")},
    {"actor_max_grad_norm": -1.}, {"critic_max_grad_norm": float("nan")},
])
def test_invalid_separate_optimizer_settings_are_rejected(settings):
    with pytest.raises(ValueError, match="must be"):
        make_algorithm(**settings)
