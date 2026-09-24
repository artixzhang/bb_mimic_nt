# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Motion validation, kinematics, interpolation inputs, and cache tests."""

from __future__ import annotations

import copy
from pathlib import Path

import joblib
import numpy as np
import pytest
import torch

from bb_mimic_nt.trajectory import (
    CACHE_SCHEMA_VERSION,
    CONTACT_NAMES,
    EXPECTED_DOF_NAMES,
    ROTATION_ONLY_TRACKING_BODIES,
    TRACKED_BODY_NAMES,
    cache_is_current,
    load_motion_batch,
    preprocess_motion_batch,
    reorder_clip_channels,
)
from bb_mimic_nt.trajectory.processing import (
    _compute_fk,
    _quat_multiply,
    _quat_to_matrix,
    finite_difference,
    quaternion_angular_velocity,
    validate_motion_batch,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ASSET_ROOT = PROJECT_ROOT / "source/bb_mimic_nt/assets"
SOURCE = ASSET_ROOT / "trajectory/shoot_batch_0922.pkl"
URDF = ASSET_ROOT / "robots/g1/urdf/g1_29dof_mode16_bb.urdf"


def test_real_batch_validation_contact_reordering_padding_and_hashes(tmp_path: Path) -> None:
    raw, dof_names = validate_motion_batch(joblib.load(SOURCE))
    cache_path = tmp_path / "motion.pt"
    preprocess_motion_batch(SOURCE, URDF, cache_path)
    cache = load_motion_batch(cache_path)
    assert len(raw) == 100
    assert len(dof_names) == 29
    assert dof_names == EXPECTED_DOF_NAMES
    assert cache["metadata"]["schema_version"] == CACHE_SCHEMA_VERSION
    assert tuple(cache["metadata"]["rotation_body_names"]) == tuple(
        name for name, _ in ROTATION_ONLY_TRACKING_BODIES
    )
    assert cache["rotation_quat_b"].shape[-2:] == (len(ROTATION_ONLY_TRACKING_BODIES), 4)
    assert tuple(cache["metadata"]["contact_names"]) == CONTACT_NAMES
    assert cache_is_current(cache_path, SOURCE, URDF)
    assert cache["anchor_pos_b"].shape[-2:] == (2, 3)
    assert cache["push_available"].shape == cache["valid"].shape

    length = int(cache["lengths"][0])
    source_indices = [tuple(raw[0]["contact_names"]).index(name) for name in CONTACT_NAMES]
    assert np.array_equal(cache["contact"][0, :length].numpy(), np.asarray(raw[0]["contact"])[:, source_indices])
    assert np.array_equal(cache["push_available"][0, :length].numpy(), np.asarray(raw[0]["push_available"]))
    assert torch.all(cache["valid"][0, :length])
    assert not torch.any(cache["valid"][0, length:])
    if length < cache["valid"].shape[1]:
        expected_padding = cache["root_pos"][0, length - 1].expand_as(cache["root_pos"][0, length:])
        assert torch.equal(cache["root_pos"][0, length:], expected_padding)
        assert torch.count_nonzero(cache["root_lin_vel"][0, length:]) == 0
        assert torch.count_nonzero(cache["push_available"][0, length:]) == 0


def test_invalid_fps_and_quaternion_are_rejected() -> None:
    clip = copy.deepcopy(joblib.load(SOURCE)[0])
    clip["fps"] = 60
    with pytest.raises(ValueError, match="100 Hz"):
        validate_motion_batch([clip])
    clip["fps"] = 100
    clip["root_rot"][0] = 0.0
    with pytest.raises(ValueError, match="non-normalized"):
        validate_motion_batch([clip])


def test_push_available_requires_one_binary_value_per_frame() -> None:
    clip = copy.deepcopy(joblib.load(SOURCE)[0])
    clip["push_available"] = clip["push_available"][:-1]
    with pytest.raises(ValueError, match="push_available.*shape"):
        validate_motion_batch([clip])
    clip["push_available"] = np.full(len(clip["root_pos"]), 0.5, dtype=np.float32)
    with pytest.raises(ValueError, match="push_available labels must be binary"):
        validate_motion_batch([clip])


def test_dof_and_contact_channels_are_reordered_by_name() -> None:
    clip = copy.deepcopy(joblib.load(SOURCE)[0])
    expected_dof = np.asarray(clip["dof"], dtype=np.float32).copy()
    expected_contact = np.asarray(clip["contact"], dtype=np.float32).copy()
    expected_contact_names = tuple(clip["contact_names"])

    dof_permutation = np.roll(np.arange(29), 7)
    contact_permutation = np.asarray((2, 0, 3, 1))
    clip["dof"] = expected_dof[:, dof_permutation]
    clip["dof_names"] = [clip["dof_names"][index] for index in dof_permutation]
    clip["contact"] = expected_contact[:, contact_permutation]
    clip["contact_names"] = [clip["contact_names"][index] for index in contact_permutation]

    _, dof_names = validate_motion_batch([clip])
    dof, contact = reorder_clip_channels(clip)
    source_contact_indices = [expected_contact_names.index(name) for name in CONTACT_NAMES]
    assert dof_names == EXPECTED_DOF_NAMES
    assert np.array_equal(dof, expected_dof)
    assert np.array_equal(contact, expected_contact[:, source_contact_indices])


def test_central_difference_and_quaternion_angular_velocity() -> None:
    time = np.arange(5, dtype=np.float32) * 0.1
    values = (time**2)[:, None]
    derivative = finite_difference(values, 0.1)[:, 0]
    assert np.allclose(derivative[1:-1], 2.0 * time[1:-1], atol=1.0e-6)

    angle = time * 2.0
    quat = np.stack((np.cos(angle / 2), np.zeros_like(angle), np.zeros_like(angle), np.sin(angle / 2)), axis=-1)
    angular_velocity = quaternion_angular_velocity(quat, 0.1)
    assert np.allclose(angular_velocity[:, :2], 0.0, atol=1.0e-5)
    assert np.allclose(angular_velocity[:, 2], 2.0, atol=1.0e-4)


def test_pinocchio_fk_world_and_root_local_are_consistent() -> None:
    clip = joblib.load(SOURCE)[0]
    frame_count = 3
    root_pos = np.asarray(clip["root_pos"][:frame_count], dtype=np.float32)
    root_quat = np.asarray(clip["root_rot"][:frame_count], dtype=np.float32)
    result = _compute_fk(
        root_pos,
        root_quat,
        np.asarray(clip["dof"][:frame_count], dtype=np.float32),
        tuple(clip["dof_names"]),
        URDF,
        TRACKED_BODY_NAMES,
        {"left_hand": (0.082, -0.115, 0.0), "right_hand": (0.082, 0.115, 0.0)},
    )
    assert result["link_pos_w"].shape == (frame_count, 4, 3)
    assert np.all(np.isfinite(result["link_quat_w"]))
    rotation = _quat_to_matrix(root_quat)
    reconstructed = root_pos[:, None] + np.einsum("tij,tkj->tki", rotation, result["link_pos_b"])
    assert np.allclose(reconstructed, result["link_pos_w"], atol=1.0e-5)
    assert np.allclose(np.linalg.norm(result["link_quat_w"], axis=-1), 1.0, atol=1.0e-5)
    assert result["rotation_quat_b"].shape == (frame_count, len(ROTATION_ONLY_TRACKING_BODIES), 4)
    world_yaw = np.broadcast_to(np.array((0.7071068, 0.0, 0.0, 0.7071068)), root_quat.shape)
    rotated_root = _quat_multiply(world_yaw, root_quat)
    rotated = _compute_fk(
        root_pos, rotated_root, np.asarray(clip["dof"][:frame_count], dtype=np.float32),
        tuple(clip["dof_names"]), URDF, TRACKED_BODY_NAMES,
        {"left_hand": (0.082, -0.115, 0.0), "right_hand": (0.082, 0.115, 0.0)},
        rotation_body_names=tuple(name for name, _ in ROTATION_ONLY_TRACKING_BODIES) + ("left_elbow_link",),
    )
    assert rotated["rotation_quat_b"].shape == (frame_count, len(ROTATION_ONLY_TRACKING_BODIES) + 1, 4)
    alignment = np.abs(
        np.sum(
            result["rotation_quat_b"] * rotated["rotation_quat_b"][:, : len(ROTATION_ONLY_TRACKING_BODIES)],
            axis=-1,
        )
    )
    assert np.allclose(alignment, 1.0, atol=1.0e-5)


def test_cache_hash_invalidation(tmp_path: Path) -> None:
    source = tmp_path / "source.pkl"
    urdf = tmp_path / "robot.urdf"
    cache_path = tmp_path / "cache.pt"
    source.write_bytes(b"motion-a")
    urdf.write_bytes(b"robot-a")

    import hashlib

    sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    torch.save(
        {
            "metadata": {
                "schema_version": CACHE_SCHEMA_VERSION,
                "source_sha256": sha(source),
                "urdf_sha256": sha(urdf),
                "tracked_body_names": list(TRACKED_BODY_NAMES),
                "rotation_body_names": [name for name, _ in ROTATION_ONLY_TRACKING_BODIES],
                "rotation_body_weights": [weight for _, weight in ROTATION_ONLY_TRACKING_BODIES],
                "contact_names": list(CONTACT_NAMES),
            },
            "push_available": torch.zeros(1, 1),
            "valid": torch.ones(1, 1, dtype=torch.bool),
        },
        cache_path,
    )
    assert cache_is_current(cache_path, source, urdf)
    source.write_bytes(b"motion-b")
    assert not cache_is_current(cache_path, source, urdf)

    payload = torch.load(cache_path, weights_only=True)
    payload["metadata"]["source_sha256"] = sha(source)
    payload["metadata"]["schema_version"] = CACHE_SCHEMA_VERSION + 1
    torch.save(payload, cache_path)
    assert not cache_is_current(cache_path, source, urdf)
