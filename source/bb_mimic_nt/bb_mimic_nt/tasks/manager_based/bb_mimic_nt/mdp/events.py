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


class DomainRandomizationContext:
    """Per-environment parameters and pulse state shared by events, actions, and observations."""

    def __init__(self, env):
        cfg = env.cfg.dr
        if cfg.total_iterations < 1 or cfg.steps_per_iteration < 1:
            raise ValueError("DR training iterations and steps per iteration must be positive.")
        if cfg.delay_max_offset_steps < 0 or cfg.delay_nominal_steps < 0:
            raise ValueError("DR delay nominal and maximum offset must be non-negative.")
        if len(cfg.push_force_min_n) != 3 or len(cfg.push_force_max_n) != 3:
            raise ValueError("DR push force ranges must contain x, y, and z limits.")
        if min(
            cfg.ball_mass_fraction,
            cfg.pd_gain_fraction,
            cfg.hand_friction_delta,
            cfg.foot_friction_delta,
            cfg.link_mass_fraction,
            cfg.push_torque_max_nm,
            *cfg.push_force_max_n,
        ) <= 0.0:
            raise ValueError("DR randomization ranges must be positive.")
        if min(cfg.push_force_min_n) < 0.0:
            raise ValueError("DR push force minimum magnitudes must be non-negative.")
        if any(lower > upper for lower, upper in zip(cfg.push_force_min_n, cfg.push_force_max_n)):
            raise ValueError("DR push force minimums must not exceed their maximums.")
        if cfg.ball_mass_fraction >= 1.0 or cfg.pd_gain_fraction >= 1.0 or cfg.link_mass_fraction >= 1.0:
            raise ValueError("DR mass and PD fractions must be less than one.")
        if cfg.hand_friction_nominal < cfg.hand_friction_delta or cfg.foot_friction_nominal < cfg.foot_friction_delta:
            raise ValueError("DR friction ranges must be non-negative.")
        pulse_steps = round(cfg.push_duration_s / env.step_dt)
        if pulse_steps < 1 or not math.isclose(pulse_steps * env.step_dt, cfg.push_duration_s, abs_tol=1e-8):
            raise ValueError("DR push duration must be a positive multiple of the policy step time.")
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
        self.push_force = torch.zeros((count, 1, 3), device=device)
        self.push_force_min = torch.tensor(cfg.push_force_min_n, device=device)
        self.push_force_max = torch.tensor(cfg.push_force_max_n, device=device)
        self.push_torque = torch.zeros_like(self.push_force)
        self.push_target_frame = torch.full((count,), -1, device=device, dtype=torch.long)
        self.push_steps_left = torch.zeros(count, device=device, dtype=torch.long)
        self.push_used = torch.zeros(count, device=device, dtype=torch.bool)
        self.push_applied = torch.zeros(count, device=device, dtype=torch.bool)
        self.iteration_offset = 0

        self.joint_ids = self.motion.joint_ids
        self.link_mass_ids = self._body_ids(LINK_MASS_BODIES)
        self.hand_ids = self._body_ids(HAND_BODIES)
        self.foot_ids = self._body_ids(FOOT_BODIES)
        self.torso_id = self._body_ids(("torso_link",))[0]

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
        for name in ("delay", "ball_mass", "pd_gains", "hand_friction", "foot_friction", "link_mass", "push")
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
    context.push_force[env_ids, 0] = _uniform_signed(
        count, context.push_force_min, context.push_force_max, strength["push"], device
    )
    context.push_torque[env_ids, 0] = (
        2.0 * torch.rand((count, 3), device=device) - 1.0
    ) * cfg.push_torque_max_nm * strength["push"]
    context.push_target_frame[env_ids] = -1
    context.push_steps_left[env_ids] = 0
    context.push_used[env_ids] = False
    context.push_applied[env_ids] = False
    if context.robot.has_external_wrench:
        context.robot.has_external_wrench = bool(context.push_applied.any())
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


def arm_push(env, env_ids: torch.Tensor):
    """Choose one available reference frame after the motion reset selects clip and RSI."""
    context = get_dr_context(env)
    motion = context.motion
    allowed = motion._batch["push_available"][motion.clip_ids[env_ids]] > 0.5
    frames = torch.arange(allowed.shape[1], device=env.device)[None, :]
    valid = allowed & (frames < motion.lengths[motion.clip_ids[env_ids], None])
    valid &= frames >= torch.floor(motion.frame[env_ids, None]).long()
    counts = valid.sum(dim=1)
    random_rank = torch.floor(torch.rand(len(env_ids), device=env.device) * counts.clamp_min(1)).long()
    selected = torch.argmax(((valid.cumsum(dim=1) == random_rank[:, None] + 1) & valid).long(), dim=1)
    context.push_target_frame[env_ids] = torch.where(counts > 0, selected, -1)


def advance_push_pulse(env, env_ids: torch.Tensor | None):
    """Update the torso wrench once per policy step and obey the reference gate."""
    context = get_dr_context(env)
    cfg = env.cfg.dr
    iteration = context.iteration_offset + env.common_step_counter / cfg.steps_per_iteration
    if curriculum_strength(cfg.push, iteration, cfg.total_iterations) == 0.0 and not context.robot.has_external_wrench:
        return
    motion = context.motion
    allowed = motion.reference["push_available"]
    context.push_steps_left[~allowed] = 0
    due = (
        allowed
        & ~context.push_used
        & (context.push_target_frame >= 0)
        & (motion.frame >= context.push_target_frame)
    )
    pulse_steps = round(cfg.push_duration_s / env.step_dt)
    context.push_steps_left[due] = pulse_steps
    context.push_used[due] = True
    active = context.push_steps_left > 0
    changed = torch.nonzero(active != context.push_applied).flatten()
    if len(changed) > 0:
        enabled = active[changed, None, None]
        forces = torch.where(enabled, context.push_force[changed], 0.0)
        torques = torch.where(enabled, context.push_torque[changed], 0.0)
        context.robot.set_external_force_and_torque(
            forces, torques, body_ids=[context.torso_id], env_ids=changed, is_global=True
        )
        context.push_applied[changed] = active[changed]
        context.robot.has_external_wrench = bool(context.push_applied.any())
    context.push_steps_left[active] -= 1
