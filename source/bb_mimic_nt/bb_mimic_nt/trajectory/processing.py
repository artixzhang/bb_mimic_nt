# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Offline preprocessing for the basketball motion batch.

This module deliberately has no Isaac Lab imports.  It can therefore validate
and preprocess a dataset with a normal Python process in the Isaac Lab conda
environment instead of booting Kit or running FK inside every RL environment.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Iterable

import joblib
import numpy as np
import torch

CACHE_SCHEMA_VERSION = 3

TRACKED_BODY_NAMES = (
    "left_hand",
    "right_hand",
    "left_ankle_roll_link",
    "right_ankle_roll_link",
    "left_shoulder_yaw_link",
    "right_shoulder_yaw_link",
    "left_elbow_link",
    "right_elbow_link",
    "left_hip_yaw_link",
    "right_hip_yaw_link",
    "left_knee_link",
    "right_knee_link",
)
# Root-local rotation tracking only. Add (body name, relative weight) here;
# these bodies do not enter link-position tracking or policy observations.
ROTATION_ONLY_TRACKING_BODIES = (
    ("torso_link", 1.0),
)
CONTACT_NAMES = (
    "left_hand_ball",
    "right_hand_ball",
    "left_foot_ground",
    "right_foot_ground",
)
ANCHOR_BODY_NAMES = ("left_hand", "right_hand")
DEFAULT_ANCHOR_OFFSETS = {
    "left_hand": (0.082, -0.115, 0.0),
    "right_hand": (0.082, 0.115, 0.0),
}
EXPECTED_DOF_NAMES = (
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
)

_REQUIRED_KEYS = {
    "root_pos",
    "root_rot",
    "dof",
    "obj_pos",
    "obj_rot",
    "contact",
    "push_available",
    "contact_names",
    "fps",
    "dof_names",
    "hoop_pos_w",
}


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _normalize_quaternion_sequence(quat_wxyz: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat_wxyz, dtype=np.float64).copy()
    norms = np.linalg.norm(quat, axis=-1, keepdims=True)
    if not np.all(np.isfinite(norms)) or np.any(norms < 1.0e-8):
        raise ValueError("Quaternion sequence contains non-finite or zero-length values.")
    quat /= norms
    for index in range(1, len(quat)):
        dot = np.sum(quat[index - 1] * quat[index], axis=-1)
        quat[index] = np.where(np.asarray(dot < 0.0)[..., None], -quat[index], quat[index])
    return quat.astype(np.float32)


def _quat_conjugate(quat: np.ndarray) -> np.ndarray:
    result = quat.copy()
    result[..., 1:] *= -1.0
    return result


def _quat_multiply(lhs: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    lw, lx, ly, lz = np.moveaxis(lhs, -1, 0)
    rw, rx, ry, rz = np.moveaxis(rhs, -1, 0)
    return np.stack(
        (
            lw * rw - lx * rx - ly * ry - lz * rz,
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
        ),
        axis=-1,
    )


def _quat_to_matrix(quat_wxyz: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat_wxyz, dtype=np.float64)
    w, x, y, z = np.moveaxis(quat, -1, 0)
    return np.stack(
        (
            1 - 2 * (y * y + z * z),
            2 * (x * y - z * w),
            2 * (x * z + y * w),
            2 * (x * y + z * w),
            1 - 2 * (x * x + z * z),
            2 * (y * z - x * w),
            2 * (x * z - y * w),
            2 * (y * z + x * w),
            1 - 2 * (x * x + y * y),
        ),
        axis=-1,
    ).reshape(quat.shape[:-1] + (3, 3))


def _matrix_to_quat_wxyz(matrix: np.ndarray) -> np.ndarray:
    # Pinocchio is already a required preprocessing dependency and provides a
    # robust matrix-to-quaternion implementation.
    import pinocchio as pin

    xyzw = pin.Quaternion(matrix).coeffs()
    return np.asarray((xyzw[3], xyzw[0], xyzw[1], xyzw[2]), dtype=np.float32)


def finite_difference(values: np.ndarray, dt: float) -> np.ndarray:
    """Differentiate a sequence with central and one-sided finite differences."""
    values = np.asarray(values, dtype=np.float64)
    if values.shape[0] < 2:
        return np.zeros_like(values, dtype=np.float32)
    result = np.empty_like(values)
    result[0] = (values[1] - values[0]) / dt
    result[-1] = (values[-1] - values[-2]) / dt
    if values.shape[0] > 2:
        result[1:-1] = (values[2:] - values[:-2]) / (2.0 * dt)
    return result.astype(np.float32)


def quaternion_angular_velocity(quat_wxyz: np.ndarray, dt: float) -> np.ndarray:
    """Compute world-frame angular velocity from a WXYZ quaternion sequence."""
    quat = _normalize_quaternion_sequence(quat_wxyz).astype(np.float64)
    result = np.zeros((len(quat), 3), dtype=np.float64)
    if len(quat) < 2:
        return result.astype(np.float32)

    def interval_velocity(first: np.ndarray, second: np.ndarray, duration: float) -> np.ndarray:
        delta = _quat_multiply(second, _quat_conjugate(first))
        if delta[0] < 0.0:
            delta = -delta
        vector = delta[1:]
        vector_norm = np.linalg.norm(vector)
        if vector_norm < 1.0e-10:
            return np.zeros(3, dtype=np.float64)
        angle = 2.0 * np.arctan2(vector_norm, np.clip(delta[0], -1.0, 1.0))
        return vector / vector_norm * angle / duration

    result[0] = interval_velocity(quat[0], quat[1], dt)
    result[-1] = interval_velocity(quat[-2], quat[-1], dt)
    for index in range(1, len(quat) - 1):
        result[index] = interval_velocity(quat[index - 1], quat[index + 1], 2.0 * dt)
    return result.astype(np.float32)


def _validate_clip(clip: dict[str, Any], clip_index: int) -> None:
    missing = _REQUIRED_KEYS.difference(clip)
    if missing:
        raise ValueError(f"Clip {clip_index} is missing fields: {sorted(missing)}")
    fps = float(clip["fps"])
    if not np.isclose(fps, 100.0):
        raise ValueError(f"Clip {clip_index} has fps={fps}; expected 100 Hz.")

    frame_count = len(clip["root_pos"])
    expected_shapes = {
        "root_pos": (frame_count, 3),
        "root_rot": (frame_count, 4),
        "dof": (frame_count, 29),
        "obj_pos": (frame_count, 3),
        "obj_rot": (frame_count, 4),
        "contact": (frame_count, 4),
        "push_available": (frame_count,),
        "hoop_pos_w": (3,),
    }
    if frame_count < 2:
        raise ValueError(f"Clip {clip_index} must contain at least two frames.")
    for key, shape in expected_shapes.items():
        value = np.asarray(clip[key])
        if value.shape != shape:
            raise ValueError(f"Clip {clip_index} field {key!r} has shape {value.shape}; expected {shape}.")
        if not np.all(np.isfinite(value)):
            raise ValueError(f"Clip {clip_index} field {key!r} contains NaN or Inf.")

    dof_names = tuple(str(name) for name in clip["dof_names"])
    if len(dof_names) != 29 or len(set(dof_names)) != 29:
        raise ValueError(f"Clip {clip_index} must contain 29 unique DoF names.")
    if set(dof_names) != set(EXPECTED_DOF_NAMES):
        missing = sorted(set(EXPECTED_DOF_NAMES).difference(dof_names))
        unexpected = sorted(set(dof_names).difference(EXPECTED_DOF_NAMES))
        raise ValueError(
            f"Clip {clip_index} does not contain the expected Unitree G1 DoFs; "
            f"missing={missing}, unexpected={unexpected}."
        )

    contact_names = tuple(str(name) for name in clip["contact_names"])
    if set(contact_names) != set(CONTACT_NAMES):
        raise ValueError(
            f"Clip {clip_index} contact names are {contact_names}; expected exactly {CONTACT_NAMES}."
        )
    contacts = np.asarray(clip["contact"])
    if not np.all(np.logical_or(np.isclose(contacts, 0.0), np.isclose(contacts, 1.0))):
        raise ValueError(f"Clip {clip_index} contact labels must be binary.")
    push_available = np.asarray(clip["push_available"])
    if not np.all(np.logical_or(push_available == 0.0, push_available == 1.0)):
        raise ValueError(f"Clip {clip_index} push_available labels must be binary.")

    for field in ("root_rot", "obj_rot"):
        norms = np.linalg.norm(np.asarray(clip[field], dtype=np.float64), axis=-1)
        if not np.allclose(norms, 1.0, atol=1.0e-3, rtol=0.0):
            raise ValueError(f"Clip {clip_index} field {field!r} contains non-normalized quaternions.")
        _normalize_quaternion_sequence(clip[field])


def reorder_clip_channels(clip: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    """Return DoF and contact arrays in the canonical cache order."""
    dof_names = tuple(str(name) for name in clip["dof_names"])
    contact_names = tuple(str(name) for name in clip["contact_names"])
    dof_indices = [dof_names.index(name) for name in EXPECTED_DOF_NAMES]
    contact_indices = [contact_names.index(name) for name in CONTACT_NAMES]
    return (
        np.asarray(clip["dof"], dtype=np.float32)[:, dof_indices],
        np.asarray(clip["contact"], dtype=np.float32)[:, contact_indices],
    )


def validate_motion_batch(raw_batch: Any) -> tuple[list[dict[str, Any]], tuple[str, ...]]:
    if not isinstance(raw_batch, list) or not raw_batch:
        raise ValueError("The trajectory file must contain a non-empty list of clips.")
    for clip_index, clip in enumerate(raw_batch):
        if not isinstance(clip, dict):
            raise ValueError(f"Clip {clip_index} is not a dictionary.")
        _validate_clip(clip, clip_index)
    return raw_batch, EXPECTED_DOF_NAMES


def _compute_fk(
    root_pos: np.ndarray,
    root_quat_wxyz: np.ndarray,
    dof_pos: np.ndarray,
    dof_names: tuple[str, ...],
    urdf_path: str | Path,
    tracked_body_names: tuple[str, ...],
    anchor_offsets: dict[str, tuple[float, float, float]],
    rotation_body_names: tuple[str, ...] = tuple(name for name, _ in ROTATION_ONLY_TRACKING_BODIES),
) -> dict[str, np.ndarray]:
    import pinocchio as pin

    model = pin.buildModelFromUrdf(str(urdf_path), pin.JointModelFreeFlyer())
    data = model.createData()
    joint_q_indices: list[int] = []
    for name in dof_names:
        joint_id = model.getJointId(name)
        if joint_id == 0 or model.joints[joint_id].nq != 1:
            raise ValueError(f"URDF does not contain a scalar joint named {name!r}.")
        joint_q_indices.append(model.joints[joint_id].idx_q)

    frame_ids: list[int] = []
    for name in tracked_body_names:
        frame_id = model.getFrameId(name)
        if frame_id >= len(model.frames):
            raise ValueError(f"URDF does not contain a frame named {name!r}.")
        frame_ids.append(frame_id)
    rotation_frame_ids: list[int] = []
    for name in rotation_body_names:
        frame_id = model.getFrameId(name)
        if frame_id >= len(model.frames):
            raise ValueError(f"URDF does not contain a frame named {name!r}.")
        rotation_frame_ids.append(frame_id)

    frame_count = len(root_pos)
    link_pos_w = np.empty((frame_count, len(frame_ids), 3), dtype=np.float32)
    link_quat_w = np.empty((frame_count, len(frame_ids), 4), dtype=np.float32)
    rotation_quat_b = np.empty((frame_count, len(rotation_frame_ids), 4), dtype=np.float32)
    for frame_index in range(frame_count):
        q = pin.neutral(model)
        q[:3] = root_pos[frame_index]
        # Pinocchio free-flyer quaternions use XYZW.
        q[3:7] = root_quat_wxyz[frame_index, (1, 2, 3, 0)]
        q[joint_q_indices] = dof_pos[frame_index]
        pin.forwardKinematics(model, data, q)
        pin.updateFramePlacements(model, data)
        for body_index, frame_id in enumerate(frame_ids):
            placement = data.oMf[frame_id]
            link_pos_w[frame_index, body_index] = placement.translation
            link_quat_w[frame_index, body_index] = _matrix_to_quat_wxyz(placement.rotation)
        root_rotation = _quat_to_matrix(root_quat_wxyz[frame_index])
        for body_index, frame_id in enumerate(rotation_frame_ids):
            rotation_quat_b[frame_index, body_index] = _matrix_to_quat_wxyz(
                root_rotation.T @ data.oMf[frame_id].rotation
            )

    root_rotation = _quat_to_matrix(root_quat_wxyz)
    link_pos_b = np.einsum(
        "tij,tkj->tki", np.swapaxes(root_rotation, -1, -2), link_pos_w - root_pos[:, None, :]
    ).astype(np.float32)
    link_quat_b = np.empty_like(link_quat_w)
    for frame_index in range(frame_count):
        for body_index in range(len(frame_ids)):
            link_matrix = _quat_to_matrix(link_quat_w[frame_index, body_index])
            link_quat_b[frame_index, body_index] = _matrix_to_quat_wxyz(
                root_rotation[frame_index].T @ link_matrix
            )
    link_quat_b = _normalize_quaternion_sequence(link_quat_b.reshape(frame_count, -1, 4)).reshape(
        link_quat_b.shape
    )
    rotation_quat_b = _normalize_quaternion_sequence(rotation_quat_b)

    anchor_pos_w = np.empty((frame_count, len(ANCHOR_BODY_NAMES), 3), dtype=np.float32)
    for anchor_index, body_name in enumerate(ANCHOR_BODY_NAMES):
        body_index = tracked_body_names.index(body_name)
        body_rotation = _quat_to_matrix(link_quat_w[:, body_index])
        offset = np.asarray(anchor_offsets[body_name], dtype=np.float32)
        anchor_pos_w[:, anchor_index] = link_pos_w[:, body_index] + np.einsum("tij,j->ti", body_rotation, offset)
    anchor_pos_b = np.einsum(
        "tij,tkj->tki", np.swapaxes(root_rotation, -1, -2), anchor_pos_w - root_pos[:, None]
    ).astype(np.float32)

    return {
        "link_pos_w": link_pos_w,
        "link_quat_w": link_quat_w,
        "link_pos_b": link_pos_b,
        "link_quat_b": link_quat_b,
        "rotation_quat_b": rotation_quat_b,
        "anchor_pos_w": anchor_pos_w,
        "anchor_pos_b": anchor_pos_b,
    }


def _make_padded_tensor(
    values: Iterable[np.ndarray], max_frames: int, *, repeat_last: bool = True
) -> torch.Tensor:
    arrays = [np.asarray(value, dtype=np.float32) for value in values]
    output = np.zeros((len(arrays), max_frames, *arrays[0].shape[1:]), dtype=np.float32)
    for index, value in enumerate(arrays):
        output[index, : len(value)] = value
        if repeat_last and len(value) < max_frames:
            output[index, len(value) :] = value[-1]
    return torch.from_numpy(output)


def cache_is_current(
    cache_path: str | Path,
    source_path: str | Path,
    urdf_path: str | Path,
    tracked_body_names: tuple[str, ...] = TRACKED_BODY_NAMES,
) -> bool:
    cache_path = Path(cache_path)
    if not cache_path.is_file():
        return False
    try:
        cache = torch.load(cache_path, map_location="cpu", weights_only=True)
        metadata = cache["metadata"]
        return bool(
            metadata["schema_version"] == CACHE_SCHEMA_VERSION
            and "push_available" in cache
            and cache["push_available"].shape == cache["valid"].shape
            and metadata["source_sha256"] == _sha256(source_path)
            and metadata["urdf_sha256"] == _sha256(urdf_path)
            and tuple(metadata["tracked_body_names"]) == tuple(tracked_body_names)
            and tuple(zip(metadata["rotation_body_names"], metadata["rotation_body_weights"]))
            == ROTATION_ONLY_TRACKING_BODIES
            and tuple(metadata["contact_names"]) == CONTACT_NAMES
        )
    except (KeyError, OSError, RuntimeError, TypeError, ValueError):
        return False


def preprocess_motion_batch(
    source_path: str | Path,
    urdf_path: str | Path,
    output_path: str | Path,
    *,
    force: bool = False,
    tracked_body_names: tuple[str, ...] = TRACKED_BODY_NAMES,
    anchor_offsets: dict[str, tuple[float, float, float]] | None = None,
) -> dict[str, Any]:
    """Validate, preprocess, and save a padded torch trajectory cache."""
    source_path = Path(source_path).resolve()
    urdf_path = Path(urdf_path).resolve()
    output_path = Path(output_path).resolve()
    anchor_offsets = dict(DEFAULT_ANCHOR_OFFSETS if anchor_offsets is None else anchor_offsets)
    rotation_body_names = tuple(name for name, _ in ROTATION_ONLY_TRACKING_BODIES)
    if not rotation_body_names or len(set(rotation_body_names)) != len(rotation_body_names):
        raise ValueError("Rotation-only tracking must contain unique body names.")
    if any(weight <= 0.0 for _, weight in ROTATION_ONLY_TRACKING_BODIES):
        raise ValueError("Rotation-only tracking weights must be positive.")
    if not source_path.is_file():
        raise FileNotFoundError(f"Trajectory file does not exist: {source_path}")
    if not urdf_path.is_file():
        raise FileNotFoundError(f"URDF file does not exist: {urdf_path}")
    if not force and cache_is_current(output_path, source_path, urdf_path, tracked_body_names):
        return torch.load(output_path, map_location="cpu", weights_only=True)

    raw_batch, dof_names = validate_motion_batch(joblib.load(source_path))
    processed: list[dict[str, np.ndarray]] = []
    lengths: list[int] = []
    for clip in raw_batch:
        root_pos = np.asarray(clip["root_pos"], dtype=np.float32)
        root_quat = _normalize_quaternion_sequence(clip["root_rot"])
        dof_pos, contact = reorder_clip_channels(clip)
        object_pos = np.asarray(clip["obj_pos"], dtype=np.float32)
        object_quat = _normalize_quaternion_sequence(clip["obj_rot"])
        dt = 1.0 / float(clip["fps"])
        fk = _compute_fk(
            root_pos,
            root_quat,
            dof_pos,
            dof_names,
            urdf_path,
            tuple(tracked_body_names),
            anchor_offsets,
            rotation_body_names,
        )
        link_ang_vel = np.stack(
            [
                quaternion_angular_velocity(fk["link_quat_w"][:, body_index], dt)
                for body_index in range(len(tracked_body_names))
            ],
            axis=1,
        )
        processed.append(
            {
                "root_pos": root_pos,
                "root_quat": root_quat,
                "root_lin_vel": finite_difference(root_pos, dt),
                "root_ang_vel": quaternion_angular_velocity(root_quat, dt),
                "dof_pos": dof_pos,
                "dof_vel": finite_difference(dof_pos, dt),
                "object_pos": object_pos,
                "object_quat": object_quat,
                "object_lin_vel": finite_difference(object_pos, dt),
                "object_ang_vel": quaternion_angular_velocity(object_quat, dt),
                "contact": contact,
                "push_available": np.asarray(clip["push_available"], dtype=np.float32),
                "hoop_pos": np.repeat(np.asarray(clip["hoop_pos_w"], dtype=np.float32)[None], len(root_pos), axis=0),
                "link_lin_vel_w": finite_difference(fk["link_pos_w"], dt),
                "link_ang_vel_w": link_ang_vel,
                "anchor_object_rel_pos_w": fk["anchor_pos_w"] - object_pos[:, None, :],
                **fk,
            }
        )
        lengths.append(len(root_pos))

    max_frames = max(lengths)
    cache: dict[str, Any] = {
        "metadata": {
            "schema_version": CACHE_SCHEMA_VERSION,
            "source_path": str(source_path),
            "source_sha256": _sha256(source_path),
            "urdf_path": str(urdf_path),
            "urdf_sha256": _sha256(urdf_path),
            "fps": 100.0,
            "dof_names": list(dof_names),
            "tracked_body_names": list(tracked_body_names),
            "rotation_body_names": [name for name, _ in ROTATION_ONLY_TRACKING_BODIES],
            "rotation_body_weights": [weight for _, weight in ROTATION_ONLY_TRACKING_BODIES],
            "anchor_body_names": list(ANCHOR_BODY_NAMES),
            "anchor_offsets": {key: list(value) for key, value in anchor_offsets.items()},
            "contact_names": list(CONTACT_NAMES),
            "num_clips": len(processed),
            "max_frames": max_frames,
        },
        "lengths": torch.tensor(lengths, dtype=torch.long),
        "valid": torch.arange(max_frames)[None, :] < torch.tensor(lengths)[:, None],
    }
    zero_padded_fields = {
        "root_lin_vel",
        "root_ang_vel",
        "dof_vel",
        "object_lin_vel",
        "object_ang_vel",
        "link_lin_vel_w",
        "link_ang_vel_w",
        "push_available",
    }
    for field in processed[0]:
        cache[field] = _make_padded_tensor(
            (clip[field] for clip in processed), max_frames, repeat_last=field not in zero_padded_fields
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(cache, output_path)
    return cache


def load_motion_batch(
    cache_path: str | Path,
    *,
    device: str | torch.device = "cpu",
) -> dict[str, Any]:
    cache = torch.load(Path(cache_path), map_location=device, weights_only=True)
    if cache.get("metadata", {}).get("schema_version") != CACHE_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported motion cache schema: {cache.get('metadata', {}).get('schema_version')}; "
            f"expected {CACHE_SCHEMA_VERSION}."
        )
    if "push_available" not in cache or cache["push_available"].shape != cache["valid"].shape:
        raise ValueError("Motion cache has no valid push_available field.")
    return cache
