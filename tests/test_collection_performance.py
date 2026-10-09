"""Numerical regression checks for collection optimizations, without Isaac Sim."""
import ast
import copy
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from bb_mimic_nt.core import interpolate_motion, motion_interpolation_indices, slerp_wxyz

MDP = Path(__file__).resolve().parents[1] / "source/bb_mimic_nt/bb_mimic_nt/tasks/manager_based/bb_mimic_nt/mdp"
DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


def load_methods(file, class_name, names, **namespace):
    tree = ast.parse((MDP / file).read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == class_name)
    methods = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in names]
    scope = dict(torch=torch, **namespace)
    exec(compile(ast.Module(body=methods, type_ignores=[]), file, "exec"), scope)
    return scope


# Frozen pre-optimization implementation: preserve reduction order and float API.
def legacy_episode_metrics(self, env_ids):
    duration = (self._env.episode_length_buf[env_ids].float() * self._env.step_dt).clamp_min(self._env.step_dt)
    self.last_episode_metrics['shot_success'][env_ids] = self.metrics['shot_success'][env_ids]
    for name in self._time_averaged_metric_names:
        self.last_episode_metrics[name][env_ids] = self.metrics[name][env_ids] / duration
    self.last_episode_clip_ids[env_ids] = self.clip_ids[env_ids]
    extras = {name: value[env_ids].mean().item() for name, value in self.last_episode_metrics.items()}
    extras['clip_id_mean'] = self.last_episode_clip_ids[env_ids].float().mean().item()
    extras['interaction/pickup_count'] = self._pickup_event_count[env_ids].mean().item()
    extras['interaction/release_count'] = self._release_event_count[env_ids].mean().item()
    self._pickup_event_count[env_ids] = 0.0
    self._release_event_count[env_ids] = 0.0
    for phase in self._phase_names:
        phase_duration = self._phase_durations[phase][env_ids]
        extras[f'phase/{phase}/fraction'] = (phase_duration / duration).mean().item()
        present = phase_duration > 0.0
        for metric in self._phase_metric_names:
            key = f'phase/{phase}/{metric}'
            conditional_mean = self._phase_metric_sums[key][env_ids] / phase_duration.clamp_min(self._env.step_dt)
            extras[key] = conditional_mean[present].mean().item() if torch.any(present) else 0.0
            self._phase_metric_sums[key][env_ids] = 0.0
        self._phase_durations[phase][env_ids] = 0.0
    action_term = self._env.action_manager.get_term('joint_pos')
    for index, name in enumerate(self.dof_names):
        extras[f'control/joint/{name}/residual_scale_rad'] = action_term.residual_scale[env_ids, index].mean().item()
    for metric in self.metrics.values():
        metric[env_ids] = 0.0
    stage_time = self.stage_duration[env_ids]
    total_stage_time = stage_time.sum().clamp_min(self._env.step_dt)
    for index, name in enumerate(('pre_hold', 'active', 'post_hold')):
        extras[f'sampling/{name}/fraction'] = (stage_time[:, index].sum() / total_stage_time).item()
    self.stage_duration[env_ids] = 0.0
    return extras


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("ids", [[0, 2], [1], [0, 1, 2]])
def test_episode_logging_matches_legacy_and_preserves_other_envs(device, ids):
    tensor = lambda x: torch.tensor(x, device=device, dtype=torch.float32)
    metrics = {"shot_success": tensor([1, 0, 1]), "error": tensor([.2, .6, .8])}
    term = SimpleNamespace(
        _env=SimpleNamespace(episode_length_buf=tensor([0, 4, 9]), step_dt=.01,
                            action_manager=SimpleNamespace(get_term=lambda _: SimpleNamespace(
                                residual_scale=tensor([[.2, .3], [.4, .5], [.6, .7]])))),
        metrics=metrics, last_episode_metrics={k: torch.zeros_like(v) for k, v in metrics.items()},
        _time_averaged_metric_names=("error",),
        last_episode_clip_ids=torch.zeros(3, device=device, dtype=torch.long),
        clip_ids=torch.tensor([2, 1, 0], device=device),
        _pickup_event_count=tensor([1, 2, 3]), _release_event_count=tensor([0, 1, 2]),
        _phase_names=("controlled", "free"), _phase_metric_names=("error", "other"),
        _phase_durations={"controlled": tensor([.01, 0, .03]), "free": tensor([0, 0, 0])},
        _phase_metric_sums={f"phase/{p}/{m}": tensor([.002, 0, .006])
                            for p in ("controlled", "free") for m in ("error", "other")},
        dof_names=("a", "b"), stage_duration=tensor([[.01, 0, 0], [0, .04, 0], [.01, .03, .05]]),
    )
    expected_term = copy.deepcopy(term)
    env_ids = torch.tensor(ids, device=device)
    expected = legacy_episode_metrics(expected_term, env_ids)
    collect = load_methods("commands.py", "MotionReferenceCommand", {"_collect_episode_metrics"})[
        "_collect_episode_metrics"]
    with torch.inference_mode():
        actual = collect(term, env_ids)
    assert actual == expected
    assert all(isinstance(v, float) for v in actual.values())
    for key in ("metrics", "last_episode_metrics", "_phase_durations", "_phase_metric_sums"):
        for name, value in getattr(term, key).items():
            assert torch.equal(value, getattr(expected_term, key)[name])
    for key in ("last_episode_clip_ids", "_pickup_event_count", "_release_event_count", "stage_duration"):
        assert torch.equal(getattr(term, key), getattr(expected_term, key))


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("quaternion", [False, True])
@pytest.mark.parametrize("shape", [(3,), (5, 3)])
def test_shared_interpolation_is_bitwise_equal_to_original(device, quaternion, shape):
    if quaternion:
        shape = (*shape[:-1], 4)
    generator = torch.Generator(device=device).manual_seed(42)
    values = torch.randn((3, 8, *shape), generator=generator, device=device)
    if quaternion:
        values = torch.nn.functional.normalize(values, dim=-1)
    lengths = torch.tensor([1, 5, 8], device=device)
    for frame, clips in [([0., 1.25, 7., 4.75], [0, 1, 2, 1]),
                         ([.3, 0., 3.8, 2.], [2, 0, 1, 2])]:
        frame = torch.tensor(frame, device=device)
        clips = torch.tensor(clips, device=device)
        last = lengths[clips] - 1
        lower = torch.minimum(torch.floor(frame).long().clamp_min(0), last)
        upper = torch.minimum(lower + 1, last)
        blend = frame - lower.float()
        first, second = values[clips, lower], values[clips, upper]
        expected = (slerp_wxyz(first, second, blend) if quaternion else
                    torch.lerp(first, second, blend.view((-1,) + (1,) * (first.ndim - 1))))
        indices = motion_interpolation_indices(clips, frame, lengths)
        actual = interpolate_motion(values, clips, frame, lengths, quaternion=quaternion, indices=indices)
        assert torch.equal(actual, expected)
        assert torch.equal(interpolate_motion(values, clips, frame, lengths, quaternion=quaternion), expected)


def test_weight_cache_reuses_tensors_and_observes_config_edits():
    tree = ast.parse((MDP / "rewards.py").read_text())
    resolver = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_resolve_body_weights")
    scope = dict(torch=torch)
    exec(compile(ast.Module(body=[resolver], type_ignores=[]), "rewards.py", "exec"), scope)
    resolve = scope["_resolve_body_weights"]
    method = load_methods("rewards.py", "UnifiedMimicReward", {"_body_weights"},
                          _resolve_body_weights=resolve)["_body_weights"]
    term = SimpleNamespace(_body_weight_cache={}, device="cpu")
    names, overrides = ("hand", "foot"), {"hand": 2.}
    with torch.inference_mode():
        first = method(term, "position", names, overrides)
    assert not first.is_inference()
    assert method(term, "position", names, overrides) is first
    overrides["hand"] = 3.
    changed = method(term, "position", names, overrides)
    assert changed is not first
    assert torch.equal(changed, resolve(names, overrides, "cpu"))
    assert method(term, "position", names, {}) is None
    assert method(term, "rotation", names, overrides) is not changed


@pytest.mark.parametrize("device", DEVICES)
def test_reference_refresh_preserves_all_fields_after_partial_reset(device):
    methods = load_methods("commands.py", "MotionReferenceCommand", {"_sample", "_refresh_reference"},
                           interpolate_motion=interpolate_motion,
                           motion_interpolation_indices=motion_interpolation_indices)
    quaternion_fields = {"root_quat", "object_quat", "link_quat_w", "link_quat_b", "rotation_quat_b"}
    velocity_fields = {"root_lin_vel", "root_ang_vel", "dof_vel", "object_lin_vel", "object_ang_vel"}
    fields = ("root_pos", "root_quat", "root_lin_vel", "root_ang_vel", "dof_pos", "dof_vel",
              "object_pos", "object_quat", "object_lin_vel", "object_ang_vel", "contact", "hoop_pos",
              "hoop_pos_cyl", "link_pos_w", "link_quat_w", "link_pos_b", "link_quat_b",
              "rotation_quat_b", "anchor_pos_w", "anchor_object_rel_pos_w")
    generator = torch.Generator(device=device).manual_seed(7)
    batch = {}
    for name in fields:
        width = 4 if name in quaternion_fields else 3
        shape = (2, width) if name.startswith(("link_", "anchor_", "rotation_")) else (width,)
        batch[name] = torch.randn((3, 8, *shape), device=device, generator=generator)
        if name in quaternion_fields:
            batch[name] = torch.nn.functional.normalize(batch[name], dim=-1)
    discrete = ("release", "release_rising", "release_falling", "push_available", "obj_push")
    for name in discrete:
        batch[name] = torch.rand((3, 8), device=device, generator=generator)
    term = SimpleNamespace(_batch=batch, clip_ids=torch.tensor([0, 1, 2], device=device),
                           frame=torch.tensor([0., 1.5, 7.], device=device),
                           lengths=torch.tensor([1, 5, 8], device=device),
                           in_active_motion=torch.tensor([False, True, True], device=device),
                           speed=torch.tensor([0., .5, 1.], device=device))
    term._sample = lambda *args, **kwargs: methods["_sample"](term, *args, **kwargs)
    for reset in (False, True):
        if reset:
            term.clip_ids[1] = 2
            term.frame[1] = 3.25
        methods["_refresh_reference"](term)
        expected = {name: interpolate_motion(batch[name], term.clip_ids, term.frame, term.lengths,
                                             quaternion=name in quaternion_fields) for name in fields}
        expected["contact"] = (expected["contact"] >= .5).float()
        lower = torch.minimum(torch.floor(term.frame).long().clamp_min(0), term.lengths[term.clip_ids] - 1)
        for name in discrete:
            expected[name] = batch[name][term.clip_ids, lower]
            if name in ("push_available", "obj_push"):
                expected[name] = expected[name] > .5
        for name in velocity_fields:
            shape = (-1,) + (1,) * (expected[name].ndim - 1)
            expected[name] = expected[name] * term.in_active_motion.float().view(shape) * term.speed.view(shape)
        assert term.reference.keys() == expected.keys()
        for name, value in expected.items():
            assert torch.equal(term.reference[name], value), name
