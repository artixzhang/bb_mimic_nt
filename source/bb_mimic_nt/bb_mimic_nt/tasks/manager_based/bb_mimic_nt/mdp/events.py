# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Domain randomization state, reset sampling, and physical perturbations."""

from __future__ import annotations

import math

import torch


LINK_MASS_BODIES = (
    "pelvis",
    "left_hip_yaw_link",
    "left_hip_roll_link",
    "left_hip_pitch_link",
    "left_knee_link",
    "right_hip_yaw_link",
    "right_hip_roll_link",
    "right_hip_pitch_link",
    "right_knee_link",
)
HAND_BODIES = ("left_hand", "right_hand")
FOOT_BODIES = ("left_ankle_roll_link", "right_ankle_roll_link")


def curriculum_strength(schedule, iteration: float, total_iterations: int) -> float:
    """Return the active DR range fraction using a cubic smoothstep curriculum."""
    if not schedule.enabled:
        return 0.0
    if getattr(schedule, "full_strength", False):
        return 1.0
    if not 0.0 <= schedule.start_fraction < schedule.end_fraction <= 1.0:
        raise ValueError("DR schedule fractions must satisfy 0 <= start < end <= 1.")
    progress = iteration / max(total_iterations, 1)
    span = schedule.end_fraction - schedule.start_fraction
    normalized = min(1.0, max(0.0, (progress - schedule.start_fraction) / span))
    return normalized * normalized * (3.0 - 2.0 * normalized)


class _MassCache:
    def __init__(self, asset):
        self.asset = asset
        self.masses = asset.root_physx_view.get_masses().clone()
        self.inertias = asset.root_physx_view.get_inertias().clone()
        self.default_masses = asset.data.default_mass.cpu().clone()
        self.default_inertias = asset.data.default_inertia.cpu().clone()


PUSH_TYPES = ("push", "global_push", "object_push")


class PushPulse:
    """Independent episode draw, target, and pulse lifetime for one perturbation."""

    def __init__(self, env, context, name):
        cfg = env.cfg.dr
        self.name = name
        self.schedule = getattr(cfg, name)
        self.probability = getattr(cfg, f"{name}_probability")
        lower = getattr(cfg, f"{name}_force_min_n")
        upper = getattr(cfg, f"{name}_force_max_n")
        self.torque_max = getattr(cfg, f"{name}_torque_max_nm")
        duration = getattr(cfg, f"{name}_duration_s")
        if len(lower) != 3 or len(upper) != 3 or any(
            not math.isfinite(lo) or not math.isfinite(hi) or not 0 <= lo <= hi
            for lo, hi in zip(lower, upper)
        ):
            raise ValueError(f"{name}: expected finite 0 <= min <= max force limits for x/y/z.")
        if not 0 <= self.probability <= 1 or not math.isfinite(self.torque_max) or self.torque_max < 0:
            raise ValueError(f"{name}: invalid probability or torque limit.")
        if not math.isfinite(duration) or duration <= 0:
            raise ValueError(f"{name}: duration must be positive and finite.")
        self.pulse_steps = round(duration / env.step_dt)
        if self.pulse_steps < 1 or not math.isclose(self.pulse_steps * env.step_dt, duration, abs_tol=1e-8):
            raise ValueError(f"{name}: duration must be a positive multiple of the policy step time.")
        self.asset = context.ball if name == "object_push" else context.robot
        weights = {"object": 1.0} if name == "object_push" else getattr(cfg, f"{name}_link_weights")
        if not weights or any(not math.isfinite(w) or w < 0 for w in weights.values()) or sum(weights.values()) <= 0:
            raise ValueError(f"{name}: link weights must be finite, non-negative, and have a positive sum.")
        ids = (0,) if name == "object_push" else context._body_ids(tuple(weights))
        self.body_ids = torch.tensor(ids, device=env.device, dtype=torch.long)
        self.weights = torch.tensor(list(weights.values()), device=env.device, dtype=torch.float32)
        self.force_min = torch.tensor(lower, device=env.device)
        self.force_max = torch.tensor(upper, device=env.device)
        self.force = torch.zeros((env.num_envs, 3), device=env.device)
        self.torque = torch.zeros_like(self.force)
        self.body = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)
        self.target = torch.full((env.num_envs,), float("inf"), device=env.device)
        self.steps_left = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)
        self.used = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
        self.applied = torch.zeros_like(self.used)
        self.selected = torch.zeros_like(self.used)

    def reset(self, env_ids, strength):
        count = len(env_ids)
        device = self.force.device
        self.force[env_ids] = _uniform_signed(count, self.force_min, self.force_max, strength, device)
        self.torque[env_ids] = (2 * torch.rand((count, 3), device=device) - 1) * self.torque_max * strength
        self.body[env_ids] = self.body_ids[torch.multinomial(self.weights, count, replacement=True)]
        self.selected[env_ids] = (torch.rand(count, device=device) < self.probability) & (strength > 0)
        self.target[env_ids] = float("inf")
        self.steps_left[env_ids] = 0
        self.used[env_ids] = False
        self.applied[env_ids] = False


class DomainRandomizationContext:
    """Per-environment parameters and pulse state shared by events, actions, and observations."""

    def __init__(self, env):
        cfg = env.cfg.dr
        if cfg.total_iterations < 1 or cfg.steps_per_iteration < 1:
            raise ValueError("DR training iterations and steps per iteration must be positive.")
        if cfg.delay_max_offset_steps < 0 or cfg.delay_nominal_steps < 0:
            raise ValueError("DR delay nominal and maximum offset must be non-negative.")
        if min(
            cfg.ball_mass_fraction,
            cfg.pd_gain_fraction,
            cfg.hand_friction_delta,
            cfg.foot_friction_delta,
            cfg.link_mass_fraction,
        ) <= 0.0:
            raise ValueError("DR randomization ranges must be positive.")
        if cfg.ball_mass_fraction >= 1.0 or cfg.pd_gain_fraction >= 1.0 or cfg.link_mass_fraction >= 1.0:
            raise ValueError("DR mass and PD fractions must be less than one.")
        if cfg.hand_friction_nominal < cfg.hand_friction_delta or cfg.foot_friction_nominal < cfg.foot_friction_delta:
            raise ValueError("DR friction ranges must be non-negative.")
        self.cfg = cfg
        self.robot = env.scene["robot"]
        self.ball = env.scene["ball"]
        self.motion = env.command_manager.get_term("motion")
        count = env.num_envs
        device = env.device
        self.delay_steps = torch.full((count,), cfg.delay_nominal_steps, device=device, dtype=torch.long)
        self.ball_mass_scale = torch.ones(count, device=device)
        self.pd_scale = torch.ones(count, device=device)
        self.hand_friction = torch.full((count,), cfg.hand_friction_nominal, device=device)
        self.foot_friction = torch.full((count,), cfg.foot_friction_nominal, device=device)
        self.link_mass_scale = torch.ones(count, device=device)
        self.pushes = {name: PushPulse(env, self, name) for name in PUSH_TYPES}
        self.robot_push_bodies = sorted(set(
            self.pushes["push"].body_ids.tolist() + self.pushes["global_push"].body_ids.tolist()
        ))
        self.iteration_offset = 0

        self.joint_ids = self.motion.joint_ids
        self.link_mass_ids = self._body_ids(LINK_MASS_BODIES)
        self.hand_ids = self._body_ids(HAND_BODIES)
        self.foot_ids = self._body_ids(FOOT_BODIES)

        shape_counts = []
        for link_path in self.robot.root_physx_view.link_paths[0]:
            view = self.robot._physics_sim_view.create_rigid_body_view(link_path)
            shape_counts.append(view.max_shapes)
        if sum(shape_counts) != self.robot.root_physx_view.max_shapes:
            raise ValueError("Could not map robot collision shapes to bodies for friction DR.")
        self.hand_shapes = self._shape_ids(shape_counts, self.hand_ids)
        self.foot_shapes = self._shape_ids(shape_counts, self.foot_ids)
        self.ball_mass_cache = _MassCache(self.ball)
        self.robot_mass_cache = _MassCache(self.robot)
        self.materials = self.robot.root_physx_view.get_material_properties().clone()
        self.joint_stiffness_cpu = self.robot.data.joint_stiffness.cpu().clone()
        self.joint_damping_cpu = self.robot.data.joint_damping.cpu().clone()
        self.joint_ids_cpu = torch.as_tensor(self.joint_ids, dtype=torch.long, device="cpu")

    def _body_ids(self, names: tuple[str, ...]) -> tuple[int, ...]:
        ids, resolved = self.robot.find_bodies(list(names), preserve_order=True)
        if tuple(resolved) != names:
            raise ValueError(f"DR body mismatch: expected {names}, got {resolved}.")
        return tuple(ids)

    @staticmethod
    def _shape_ids(counts: list[int], body_ids: tuple[int, ...]) -> torch.Tensor:
        ids = [index for body in body_ids for index in range(sum(counts[:body]), sum(counts[:body + 1]))]
        if not ids:
            raise ValueError(f"No collision shapes found for DR bodies {body_ids}.")
        return torch.tensor(ids, dtype=torch.long)

    @property
    def privileged(self) -> torch.Tensor:
        cfg = self.cfg
        return torch.stack(
            (
                (self.delay_steps.float() - cfg.delay_nominal_steps) / max(cfg.delay_max_offset_steps, 1),
                (self.ball_mass_scale - 1.0) / cfg.ball_mass_fraction,
                (self.pd_scale - 1.0) / cfg.pd_gain_fraction,
                (self.hand_friction - cfg.hand_friction_nominal) / cfg.hand_friction_delta,
                (self.foot_friction - cfg.foot_friction_nominal) / cfg.foot_friction_delta,
                (self.link_mass_scale - 1.0) / cfg.link_mass_fraction,
            ),
            dim=-1,
        )


def get_dr_context(env) -> DomainRandomizationContext:
    context = getattr(env, "dr_context", None)
    if context is None:
        context = DomainRandomizationContext(env)
        env.dr_context = context
    return context


def training_policy_steps(env) -> float:
    context = getattr(env, "dr_context", None)
    offset = 0 if context is None else context.iteration_offset * env.cfg.dr.steps_per_iteration
    return offset + env.common_step_counter


def _uniform_scale(count: int, half_range: float, strength: float, device: str) -> torch.Tensor:
    return 1.0 + (2.0 * torch.rand(count, device=device) - 1.0) * half_range * strength


def _uniform_signed(
    count: int, lower: torch.Tensor, upper: torch.Tensor, strength: float, device: str
) -> torch.Tensor:
    """Sample per-axis magnitudes in [lower, upper], randomize signs, then apply curriculum strength."""
    magnitude = lower + torch.rand((count, 3), device=device) * (upper - lower)
    sign = torch.where(torch.rand((count, 3), device=device) < 0.5, -1.0, 1.0)
    return sign * magnitude * strength


def _set_masses_and_inertias(
    cache: _MassCache, indices: torch.Tensor, body_ids: tuple[int, ...] | None, factors: torch.Tensor
):
    if body_ids is None:
        mass_scale = factors.reshape((-1,) + (1,) * (cache.default_masses.ndim - 1))
        inertia_scale = factors.reshape((-1,) + (1,) * (cache.default_inertias.ndim - 1))
        cache.masses[indices] = cache.default_masses[indices] * mass_scale
        cache.inertias[indices] = cache.default_inertias[indices] * inertia_scale
    else:
        bodies = torch.tensor(body_ids, dtype=torch.long)
        cache.masses[indices[:, None], bodies] = cache.default_masses[indices[:, None], bodies] * factors[:, None]
        cache.inertias[indices[:, None], bodies] = (
            cache.default_inertias[indices[:, None], bodies] * factors[:, None, None]
        )
    cache.asset.root_physx_view.set_masses(cache.masses, indices)
    cache.asset.root_physx_view.set_inertias(cache.inertias, indices)


def _set_pd_gains(context: DomainRandomizationContext, env_ids: torch.Tensor, indices: torch.Tensor):
    robot = context.robot
    alpha = context.pd_scale[env_ids, None]
    stiffness = robot.data.default_joint_stiffness[env_ids[:, None], context.joint_ids] * alpha
    damping = robot.data.default_joint_damping[env_ids[:, None], context.joint_ids] * torch.sqrt(alpha)
    robot.data.joint_stiffness[env_ids[:, None], context.joint_ids] = stiffness
    robot.data.joint_damping[env_ids[:, None], context.joint_ids] = damping
    joint_ids = context.joint_ids_cpu
    gains_cpu = torch.stack((stiffness, damping)).cpu()
    context.joint_stiffness_cpu[indices[:, None], joint_ids] = gains_cpu[0]
    context.joint_damping_cpu[indices[:, None], joint_ids] = gains_cpu[1]
    robot.root_physx_view.set_dof_stiffnesses(context.joint_stiffness_cpu, indices)
    robot.root_physx_view.set_dof_dampings(context.joint_damping_cpu, indices)
    for actuator in robot.actuators.values():
        joint_ids = actuator.joint_indices
        if isinstance(joint_ids, slice):
            joint_ids = torch.arange(robot.num_joints, device=env_ids.device)[joint_ids]
        actuator.stiffness[env_ids] = robot.data.joint_stiffness[env_ids[:, None], joint_ids]
        actuator.damping[env_ids] = robot.data.joint_damping[env_ids[:, None], joint_ids]


def _set_friction(context: DomainRandomizationContext, indices: torch.Tensor, frictions: torch.Tensor):
    materials = context.materials
    for shapes, values in (
        (context.hand_shapes, frictions[:, 0]),
        (context.foot_shapes, frictions[:, 1]),
    ):
        materials[indices[:, None], shapes[None, :], :2] = values[:, None, None]
    context.robot.root_physx_view.set_material_properties(materials, indices)


def reset_domain_randomization(env, env_ids: torch.Tensor):
    """Sample and apply all episode-constant DR parameters once per reset."""
    if len(env_ids) == 0:
        return
    context = get_dr_context(env)
    cfg = env.cfg.dr
    iteration = context.iteration_offset + env.common_step_counter / cfg.steps_per_iteration
    strength = {
        name: curriculum_strength(getattr(cfg, name), iteration, cfg.total_iterations)
        for name in ("delay", "ball_mass", "pd_gains", "hand_friction", "foot_friction", "link_mass")
    }
    count = len(env_ids)
    device = env.device
    delay_radius = math.floor(cfg.delay_max_offset_steps * strength["delay"] + 1.0e-6)
    delay_low = max(0, cfg.delay_nominal_steps - delay_radius)
    delay_high = cfg.delay_nominal_steps + delay_radius
    context.delay_steps[env_ids] = torch.randint(
        delay_low, delay_high + 1, (count,), device=device
    )
    context.ball_mass_scale[env_ids] = _uniform_scale(count, cfg.ball_mass_fraction, strength["ball_mass"], device)
    context.pd_scale[env_ids] = _uniform_scale(count, cfg.pd_gain_fraction, strength["pd_gains"], device)
    context.hand_friction[env_ids] = cfg.hand_friction_nominal + (
        2.0 * torch.rand(count, device=device) - 1.0
    ) * cfg.hand_friction_delta * strength["hand_friction"]
    context.foot_friction[env_ids] = cfg.foot_friction_nominal + (
        2.0 * torch.rand(count, device=device) - 1.0
    ) * cfg.foot_friction_delta * strength["foot_friction"]
    context.link_mass_scale[env_ids] = _uniform_scale(count, cfg.link_mass_fraction, strength["link_mass"], device)
    for pulse in context.pushes.values():
        pulse.reset(env_ids, curriculum_strength(pulse.schedule, iteration, cfg.total_iterations))
    _write_push_wrenches(context, env_ids)
    indices = env_ids.cpu()
    sampled = torch.stack(
        (
            context.ball_mass_scale[env_ids],
            context.link_mass_scale[env_ids],
            context.hand_friction[env_ids],
            context.foot_friction[env_ids],
        ),
        dim=-1,
    ).cpu()
    _set_masses_and_inertias(context.ball_mass_cache, indices, None, sampled[:, 0])
    _set_masses_and_inertias(context.robot_mass_cache, indices, context.link_mass_ids, sampled[:, 1])
    _set_pd_gains(context, env_ids, indices)
    _set_friction(context, indices, sampled[:, 2:])


def _push_clock(motion):
    """Monotonic reference seconds: negative pre-hold, frames/fps, then post-hold."""
    clock = motion.frame / motion.fps
    clock = torch.where(motion.stage == 0, motion.hold_time - motion.pre_hold_duration, clock)
    return torch.where(motion.stage >= 2, clock + motion.hold_time, clock)


def arm_push(env, env_ids: torch.Tensor):
    """Draw one target per type over the remaining reference timeline, including holds.

    Holds inherit the endpoint gate. Sampling uses reference time because active
    motion speed is adaptive and its future physical duration is not yet known.
    """
    if len(env_ids) == 0:
        return
    context = get_dr_context(env)
    motion = context.motion
    clips = motion.clip_ids[env_ids]
    start = _push_clock(motion)[env_ids]
    last = (motion.lengths[clips] - 1) / motion.fps
    # The command marks DONE one policy interval before the end of post-hold.
    post = (motion.post_hold_duration[env_ids] - env.step_dt).clamp_min(0)
    end = last + post
    for name, pulse in context.pushes.items():
        if name == "global_push":
            target = start + torch.rand_like(start) * (end - start).clamp_min(0)
            available = end > start
        else:
            field = "obj_push" if name == "object_push" else "push_available"
            allowed = motion._batch[field][clips] > 0.5
            frames = torch.arange(allowed.shape[1], device=env.device)[None, :]
            left = torch.maximum(frames / motion.fps, start[:, None].clamp_min(0))
            right = torch.minimum((frames + 1) / motion.fps, last[:, None])
            weights = (right - left).clamp_min(0) * allowed
            pre_weight = (-start).clamp_min(0) * allowed[:, 0]
            post_weight = post * allowed.gather(1, (motion.lengths[clips] - 1)[:, None]).squeeze(1)
            weights = torch.cat((pre_weight[:, None], weights, post_weight[:, None]), dim=1)
            total = weights.sum(1)
            rank = torch.rand_like(total) * total
            cumulative = weights.cumsum(1)
            index = (cumulative <= rank[:, None]).sum(1).clamp_max(weights.shape[1] - 1)
            previous = (cumulative - weights).gather(1, index[:, None]).squeeze(1)
            starts = torch.cat((start[:, None], left, last[:, None]), dim=1)
            target = starts.gather(1, index[:, None]).squeeze(1) + rank - previous
            available = total > 0
        pulse.target[env_ids] = torch.where(pulse.selected[env_ids] & available, target, float("inf"))


def _write_push_wrenches(context, env_ids):
    """Compose both robot pulses before writing; preserve other environments."""
    for asset, names, bodies in (
        (context.robot, ("push", "global_push"), context.robot_push_bodies),
        (context.ball, ("object_push",), [0]),
    ):
        sample = context.pushes[names[0]]
        body_ids = torch.tensor(bodies, device=sample.force.device)
        force = torch.zeros((len(env_ids), len(bodies), 3), device=sample.force.device)
        torque = torch.zeros_like(force)
        for name in names:
            pulse = context.pushes[name]
            mask = pulse.applied[env_ids, None] & (pulse.body[env_ids, None] == body_ids[None, :])
            force += pulse.force[env_ids, None, :] * mask[:, :, None]
            torque += pulse.torque[env_ids, None, :] * mask[:, :, None]
        asset.set_external_force_and_torque(force, torque, body_ids=bodies, env_ids=env_ids, is_global=True)
        # Isaac Lab's setter may disable the whole asset when a subset is zero.
        asset.has_external_wrench = any(bool(context.pushes[name].applied.any()) for name in names)


def advance_push_pulse(env, env_ids: torch.Tensor | None):
    """Advance all three independent pulses and atomically apply their sum."""
    context = get_dr_context(env)
    motion = context.motion
    ids = torch.arange(env.num_envs, device=env.device) if env_ids is None else env_ids
    clock = _push_clock(motion)[ids]
    changed = torch.zeros(len(ids), device=env.device, dtype=torch.bool)
    for name, pulse in context.pushes.items():
        allowed = motion.stage[ids] != 3
        if name != "global_push":
            field = "obj_push" if name == "object_push" else "push_available"
            allowed &= motion.reference[field][ids]
        allowed &= pulse.schedule.enabled
        remaining = pulse.steps_left[ids]
        remaining[~allowed] = 0
        due = allowed & ~pulse.used[ids] & (clock >= pulse.target[ids])
        remaining[due] = pulse.pulse_steps
        pulse.used[ids[due]] = True
        active = remaining > 0
        changed |= active != pulse.applied[ids]
        pulse.applied[ids] = active
        pulse.steps_left[ids] = (remaining - active.long()).clamp_min(0)
    changed_ids = ids[changed]
    if len(changed_ids):
        _write_push_wrenches(context, changed_ids)
