"""Exercise the inherited runner loop, archived critic, saves and resume on CPU."""

from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch
import yaml

TensorDict = pytest.importorskip("tensordict").TensorDict
ActorCritic = pytest.importorskip("rsl_rl.modules").ActorCritic

from bb_mimic_nt import student_ppo_runner as runner_module


class Controller:
    def __init__(self):
        self.updates = 0

    def update_iteration(self):
        self.updates += 1
        return {"StudentTermination/hold/scale": 1.5}

    def state_dict(self):
        return {"updates": self.updates}

    def load_state_dict(self, state):
        self.updates = state["updates"]


class Environment:
    num_envs = 2
    num_actions = 2
    device = "cpu"

    def __init__(self):
        self.unwrapped = self
        self.common_step_counter = 0
        self.dr_context = SimpleNamespace(iteration_offset=0)
        self.student_termination_curriculum = Controller()
        self.resets = 0
        self.obs = TensorDict({"policy": torch.zeros(2, 2), "teacher": torch.ones(2, 3)}, [2])

    def get_observations(self):
        return self.obs.clone()

    def step(self, action):
        self.common_step_counter += 1
        return self.get_observations(), torch.ones(2), torch.zeros(2), {}

    def reset(self):
        self.resets += 1


@pytest.fixture
def setup_runner(tmp_path, monkeypatch):
    env = Environment()
    policy_cfg = dict(
        class_name="ActorCritic", actor_hidden_dims=[8], critic_hidden_dims=[8], activation="elu",
        actor_obs_normalization=True, critic_obs_normalization=True, init_noise_std=.8, noise_std_type="log",
    )
    teacher = ActorCritic(
        env.obs, {"policy": ["teacher"], "critic": ["teacher"]}, 2,
        **{key: value for key, value in policy_cfg.items() if key != "class_name"},
    )
    teacher.update_normalization(env.obs)
    # Normalizer groups refer to teacher, so their archived statistics are nontrivial.
    params = tmp_path / "teacher" / "params"
    params.mkdir(parents=True)
    (params / "agent.yaml").write_text(yaml.safe_dump({"policy": policy_cfg}))
    (params / "env.yaml").write_text("{}")
    checkpoint = params.parent / "model_0.pt"
    torch.save({"model_state_dict": teacher.state_dict()}, checkpoint)
    monkeypatch.setattr(runner_module, "load_teacher_contract", lambda *args: None)
    monkeypatch.setattr(runner_module, "student_validate_teacher", lambda *args: None)
    monkeypatch.setattr(runner_module, "student_validate_resume_observation", lambda *args: None)
    monkeypatch.setattr("rsl_rl.runners.on_policy_runner.store_code_state", lambda *args: [])
    cfg = dict(
        num_steps_per_env=2, max_iterations=1000, save_interval=2, logger="tensorboard",
        obs_groups={"policy": ["policy"], "critic": ["teacher"]},
        policy={**policy_cfg, "actor_obs_normalization": False},
        algorithm=dict(class_name="PPOWithBC", batch_size=3, num_learning_epochs=5,
                       bc_coef=1., learning_rate=3e-4, schedule="fixed"),
    )
    metadata = dict(action_clip=100., action_scale=[.1, .2], teacher_sha256="test-teacher-sha")

    def create(*, logs=True):
        return runner_module.StudentPPOBCRunner(
            Environment(), cfg, checkpoint, metadata, str(tmp_path / "run") if logs else None, "cpu",
        )

    return create, teacher, cfg, metadata, tmp_path


def test_archived_critic_and_normalizers_are_loaded_but_teacher_is_external(setup_runner):
    create, teacher, cfg, _, _ = setup_runner
    original = deepcopy(cfg)
    runner = create(logs=False)
    assert cfg == original
    for name, value in teacher.critic.state_dict().items():
        torch.testing.assert_close(runner.alg.policy.critic.state_dict()[name], value)
    for name, value in teacher.critic_obs_normalizer.state_dict().items():
        torch.testing.assert_close(runner.alg.policy.critic_obs_normalizer.state_dict()[name], value)
    assert not any("teacher" in name for name in runner.alg.policy.state_dict())
    runner.train_mode()
    assert not runner.alg.teacher.training and runner.alg.policy.critic.training
    assert isinstance(runner.alg.policy.actor_obs_normalizer, torch.nn.Identity)


def test_completed_iteration_checkpoint_names_resume_and_curriculum(setup_runner):
    create, teacher, _, _, tmp_path = setup_runner
    runner = create()
    teacher_state = deepcopy(runner.alg.teacher.state_dict())
    runner.learn(5)
    assert runner.current_learning_iteration == runner.alg.completed_iterations == 5
    assert runner.base.student_termination_curriculum.updates == 5
    assert runner.alg.policy.critic_obs_normalizer.count.item() == teacher.critic_obs_normalizer.count.item() + 20
    assert sorted(path.name for path in (tmp_path / "run").glob("model_*.pt")) == [
        "model_2.pt", "model_4.pt", "model_5.pt",
    ]
    saved = torch.load(tmp_path / "run/model_5.pt", weights_only=False)
    assert saved["next_iteration"] == 5
    assert saved["format"] == runner_module.CHECKPOINT_FORMAT
    for name, value in saved["student"].items():
        torch.testing.assert_close(value, runner.alg.policy.actor.state_dict()[name])
    for name, value in teacher_state.items():
        torch.testing.assert_close(value, runner.alg.teacher.state_dict()[name])
    runner.writer.flush()
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    events = EventAccumulator(str(tmp_path / "run")).Reload()
    assert [event.step for event in events.Scalars("Loss/kl")] == list(range(1, 6))
    assert [event.step for event in events.Scalars("StudentPPO/samples_consumed")] == list(range(1, 6))
    resumed = create(logs=False)
    resumed.load(tmp_path / "run/model_5.pt")
    assert resumed.current_learning_iteration == 5
    assert resumed.base.student_termination_curriculum.updates == 5
    assert resumed.base.dr_context.iteration_offset == 5
    assert resumed.base.resets == 1
    assert resumed.alg.learning_rate == runner.alg.learning_rate
    resumed.learn(1)
    assert resumed.current_learning_iteration == 6
    assert resumed.base.student_termination_curriculum.updates == 6
    runner.writer.close()


def test_resume_rejects_old_format_and_teacher_change(setup_runner):
    create, _, _, _, tmp_path = setup_runner
    runner = create(logs=False)
    checkpoint = tmp_path / "checkpoint.pt"
    torch.save({"format": "bb_mimic_nt_student_dagger_v1"}, checkpoint)
    with pytest.raises(ValueError, match="DAgger"):
        runner.load(checkpoint)
    runner.save(checkpoint)
    state = torch.load(checkpoint, weights_only=False)
    state["teacher_sha256"] = "different"
    torch.save(state, checkpoint)
    with pytest.raises(ValueError, match="SHA256"):
        runner.load(checkpoint)
    with pytest.raises(ValueError, match="real episode lengths"):
        runner.learn(1, init_at_random_ep_len=True)


def assert_nested_equal(actual, expected):
    """Compare optimizer moments and metadata without relying on parameter IDs."""
    if isinstance(expected, torch.Tensor):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    elif isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for name in expected:
            assert_nested_equal(actual[name], expected[name])
    elif isinstance(expected, (list, tuple)):
        assert len(actual) == len(expected)
        for left, right in zip(actual, expected):
            assert_nested_equal(left, right)
    else:
        assert actual == expected


def optimizer_by_name(optimizer, policy):
    names = {id(parameter): name for name, parameter in policy.named_parameters()}
    return {
        "param_groups": [{**{key: deepcopy(value) for key, value in group.items() if key != "params"},
                          "params": [names[id(parameter)] for parameter in group["params"]]}
                         for group in optimizer.param_groups],
        "state": {names[id(parameter)]: deepcopy(value) for parameter, value in optimizer.state.items()},
    }


def set_learning_rates(algorithm, actor, critic):
    algorithm.actor_learning_rate = actor
    algorithm.critic_learning_rate = critic
    for optimizer, rate in ((algorithm.actor_optimizer, actor), (algorithm.critic_optimizer, critic)):
        for group in optimizer.param_groups:
            group["lr"] = rate


def make_legacy_checkpoint(runner, path):
    """Create the old single Adam layout in actual policy parameter order.

    log_std is the first policy parameter, whereas the new actor optimizer may
    list network parameters first. Distinct gradients expose positional remaps.
    """
    parameters = list(runner.alg.policy.parameters())
    assert next(iter(runner.alg.policy.named_parameters()))[0] == "log_std"
    learning_rate = 8e-5
    groups = [dict(params=parameters, betas=(.83, .97), eps=4e-6, weight_decay=.015, amsgrad=True)]
    legacy = torch.optim.Adam(groups, lr=learning_rate)
    for step in range(3):
        legacy.zero_grad(set_to_none=True)
        for index, parameter in enumerate(parameters):
            parameter.grad = torch.full_like(parameter, .2 + index * .07 + step * .03)
        legacy.step()
    legacy.zero_grad(set_to_none=True)
    runner.alg.completed_iterations = 7
    runner.save(path)
    state = torch.load(path, weights_only=False)
    for name in ("optimizer_format", "actor_optimizer_state_dict", "critic_optimizer_state_dict",
                 "actor_learning_rate", "critic_learning_rate"):
        state.pop(name, None)
    state["optimizer_state_dict"] = legacy.state_dict()
    state["learning_rate"] = learning_rate
    torch.save(state, path)
    return legacy, learning_rate


def test_separate_optimizers_roundtrip_preserves_moments_rates_and_next_update(setup_runner):
    create, _, _, _, tmp_path = setup_runner
    original = create(logs=False)
    original.learn(2)
    set_learning_rates(original.alg, actor=7e-5, critic=4e-4)
    checkpoint = tmp_path / "separate.pt"
    original.save(checkpoint)
    saved = torch.load(checkpoint, weights_only=False)
    assert saved["format"] == runner_module.CHECKPOINT_FORMAT
    assert saved["next_iteration"] == 2
    assert saved["optimizer_format"] == "separate_actor_critic_v1"
    assert "optimizer_state_dict" not in saved
    assert saved["actor_learning_rate"] == 7e-5
    assert saved["critic_learning_rate"] == 4e-4
    for prefix in ("actor", "critic"):
        optimizer = getattr(original.alg, f"{prefix}_optimizer")
        parameters = getattr(original.alg, f"{prefix}_parameters")
        assert len(optimizer.state) == len(parameters)
        assert all(value["step"] > 0 for value in optimizer.state.values())
        assert_nested_equal(saved[f"{prefix}_optimizer_state_dict"], optimizer.state_dict())

    restored = create(logs=False)
    restored.load(checkpoint)
    assert restored.alg.learning_rate == restored.alg.actor_learning_rate == 7e-5
    assert restored.alg.critic_learning_rate == 4e-4
    assert restored.current_learning_iteration == restored.alg.completed_iterations == 2
    for prefix in ("actor", "critic"):
        assert_nested_equal(optimizer_by_name(getattr(restored.alg, f"{prefix}_optimizer"), restored.alg.policy),
                            optimizer_by_name(getattr(original.alg, f"{prefix}_optimizer"), original.alg.policy))

    # The fake env is deterministic. Resetting the RNG gives identical rollout
    # actions and minibatch permutations, so resumed Adam must take the same step.
    torch.manual_seed(924)
    original.learn(1)
    torch.manual_seed(924)
    restored.learn(1)
    assert_nested_equal(restored.alg.policy.state_dict(), original.alg.policy.state_dict())
    for prefix in ("actor", "critic"):
        assert_nested_equal(optimizer_by_name(getattr(restored.alg, f"{prefix}_optimizer"), restored.alg.policy),
                            optimizer_by_name(getattr(original.alg, f"{prefix}_optimizer"), original.alg.policy))


def test_legacy_adam_is_split_by_parameter_identity_with_all_state_preserved(setup_runner):
    create, _, _, _, tmp_path = setup_runner
    original = create(logs=False)
    checkpoint = tmp_path / "legacy.pt"
    legacy, learning_rate = make_legacy_checkpoint(original, checkpoint)
    expected = optimizer_by_name(legacy, original.alg.policy)
    restored = create(logs=False)
    restored.load(checkpoint)
    assert restored.current_learning_iteration == 7
    assert restored.alg.actor_learning_rate == restored.alg.critic_learning_rate == learning_rate
    assert restored.alg.learning_rate == learning_rate
    seen = set()
    expected_groups = {name: {key: value for key, value in group.items() if key != "params"}
                       for group in expected["param_groups"] for name in group["params"]}
    for prefix in ("actor", "critic"):
        actual = optimizer_by_name(getattr(restored.alg, f"{prefix}_optimizer"), restored.alg.policy)
        for name, moments in actual["state"].items():
            assert name not in seen
            seen.add(name)
            assert_nested_equal(moments, expected["state"][name])
        for group in actual["param_groups"]:
            options = {key: value for key, value in group.items() if key != "params"}
            for name in group["params"]:
                assert_nested_equal(options, expected_groups[name])
    assert seen == set(expected["state"])
    assert "log_std" in optimizer_by_name(restored.alg.actor_optimizer, restored.alg.policy)["state"]
    assert "log_std" not in optimizer_by_name(restored.alg.critic_optimizer, restored.alg.policy)["state"]
    restored.learn(1)
    assert restored.current_learning_iteration == 8
    assert all(torch.isfinite(parameter).all() for parameter in restored.alg.policy.parameters())


@pytest.mark.parametrize("legacy", [False, True])
def test_load_without_optimizers_preserves_existing_moments_and_independent_rates(setup_runner, legacy):
    create, _, _, _, tmp_path = setup_runner
    source = create(logs=False)
    source.learn(1)
    checkpoint = tmp_path / "weights_only.pt"
    if legacy:
        make_legacy_checkpoint(source, checkpoint)
    else:
        set_learning_rates(source.alg, actor=8e-5, critic=6e-4)
        source.save(checkpoint)
    restored = create(logs=False)
    restored.learn(2)
    set_learning_rates(restored.alg, actor=2e-4, critic=9e-4)
    before = {prefix: optimizer_by_name(getattr(restored.alg, f"{prefix}_optimizer"), restored.alg.policy)
              for prefix in ("actor", "critic")}
    restored.load(checkpoint, load_optimizer=False)
    assert restored.alg.learning_rate == restored.alg.actor_learning_rate == 2e-4
    assert restored.alg.critic_learning_rate == 9e-4
    for prefix in ("actor", "critic"):
        assert_nested_equal(optimizer_by_name(getattr(restored.alg, f"{prefix}_optimizer"), restored.alg.policy),
                            before[prefix])
    assert_nested_equal(restored.alg.policy.state_dict(), source.alg.policy.state_dict())
