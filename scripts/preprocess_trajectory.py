#!/usr/bin/env python3
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Validate and preprocess a basketball motion batch without launching Isaac Sim."""

from __future__ import annotations

import argparse
from pathlib import Path

from bb_mimic_nt.trajectory import preprocess_motion_batch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = PROJECT_ROOT / "source/bb_mimic_nt/assets/trajectory/shoot_batch_0910.pkl"
DEFAULT_OUTPUT = PROJECT_ROOT / "source/bb_mimic_nt/assets/trajectory/shoot_batch_0910_processed.pt"
DEFAULT_URDF = PROJECT_ROOT / "source/bb_mimic_nt/assets/robots/g1/urdf/unitree_g1_bb.urdf"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Raw joblib motion batch.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Output MotionBatchV1 torch cache.")
    parser.add_argument("--urdf", type=Path, default=DEFAULT_URDF, help="URDF used for forward kinematics.")
    parser.add_argument("--force", action="store_true", help="Rebuild even when the cache hash is current.")
    args = parser.parse_args()

    cache = preprocess_motion_batch(args.input, args.urdf, args.output, force=args.force)
    metadata = cache["metadata"]
    lengths = cache["lengths"]
    print(f"MotionBatchV{metadata['schema_version']} ready at {args.output.resolve()}")
    print(
        f"clips={metadata['num_clips']} fps={metadata['fps']:.0f} "
        f"frames={int(lengths.min())}..{int(lengths.max())} padded={metadata['max_frames']}"
    )
    print(f"dofs={len(metadata['dof_names'])} bodies={metadata['tracked_body_names']}")
    print(f"contacts={metadata['contact_names']}")
    valid_frames = int(cache["valid"].sum())
    padded_frames = int(cache["valid"].numel() - valid_frames)
    print(f"valid_frames={valid_frames} padded_frames={padded_frames}")
    print(f"source_sha256={metadata['source_sha256']}")
    print(f"urdf_sha256={metadata['urdf_sha256']}")


if __name__ == "__main__":
    main()
