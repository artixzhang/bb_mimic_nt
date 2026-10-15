"""Pure Torch reset-state sampling with the HumanX noise distributions."""

from __future__ import annotations

import math

import torch


_GAUSSIAN_FIELDS = {
    "dof_pos": "dof_pos",
    "dof_vel": "dof_vel",
    "root_pos": "root_pos",
    "root_vel": "root_lin_vel",
    "root_ang_vel": "root_ang_vel",
}
_STATE_FIELDS = (
    "root_pos", "root_quat", "root_lin_vel", "root_ang_vel", "dof_pos", "dof_vel",
    "object_pos", "object_quat", "object_lin_vel", "object_ang_vel",
)


def validate_initial_state_cfg(cfg) -> None:
    """Reject malformed amplitudes even when a component is currently disabled."""
    if cfg is None:
        return
    amplitudes = [(name, "std") for name in _GAUSSIAN_FIELDS]
    amplitudes += [("root_rot", "max_angle_deg"), ("object_pos", "half_range_m")]
    for component, field in amplitudes:
        value = float(getattr(getattr(cfg, component), field))
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(f"dr.init_state.{component}.{field} must be finite and non-negative.")


def sample_initial_state(
    reference: dict[str, torch.Tensor], env_ids: torch.Tensor, cfg,
) -> dict[str, torch.Tensor]:
    """Copy selected reference states and perturb only enabled components.

    The reference is always unchanged. Quaternions use Isaac Lab's wxyz order;
    the sampled world-frame rotation left-multiplies the reference orientation.
    A missing configuration produces an unperturbed copy for the caller's
    legacy reset-noise path. A disabled configuration never invokes that path.
    """
    validate_initial_state_cfg(cfg)
    state = {name: reference[name].index_select(0, env_ids) for name in _STATE_FIELDS}
    if cfg is None or not cfg.enabled:
        return state

    for component, field in _GAUSSIAN_FIELDS.items():
        settings = getattr(cfg, component)
        if settings.enabled and settings.std > 0.0:
            state[field].add_(torch.randn_like(state[field]) * settings.std)

    if cfg.root_rot.enabled and cfg.root_rot.max_angle_deg > 0.0:
        quat = state["root_quat"]
        axis = torch.randn_like(quat[:, 1:])
        norm = torch.linalg.vector_norm(axis, dim=-1, keepdim=True)
        fallback = torch.zeros_like(axis)
        fallback[:, 0] = 1.0
        axis = torch.where(norm > 0.0, axis / norm.clamp_min(torch.finfo(axis.dtype).tiny), fallback)
        half_angle = torch.rand_like(quat[:, :1]) * (math.radians(cfg.root_rot.max_angle_deg) / 2.0)
        scalar, vector = torch.cos(half_angle), torch.sin(half_angle) * axis
        # Hamilton product delta * reference, expressed directly to stay CPU-testable.
        state["root_quat"] = torch.cat(
            (scalar * quat[:, :1] - (vector * quat[:, 1:]).sum(dim=-1, keepdim=True),
             scalar * quat[:, 1:] + quat[:, :1] * vector + torch.cross(vector, quat[:, 1:], dim=-1)),
            dim=-1,
        )

    if cfg.object_pos.enabled and cfg.object_pos.half_range_m > 0.0:
        xy = state["object_pos"][:, :2]
        xy.add_((torch.rand_like(xy) * 2.0 - 1.0) * cfg.object_pos.half_range_m)
    return state
