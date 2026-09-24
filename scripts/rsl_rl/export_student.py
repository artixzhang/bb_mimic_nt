#!/usr/bin/env python3
"""Export Student weights, TorchScript, ONNX, and deployment parameters."""

from __future__ import annotations

import argparse
from pathlib import Path


parser = argparse.ArgumentParser(description=__doc__)
source = parser.add_mutually_exclusive_group(required=True)
source.add_argument("--checkpoint", type=Path, help="A model_N.pt DAgger training checkpoint.")
source.add_argument("--student-weights", type=Path, help="An exported student_weights.pt file.")
parser.add_argument("--student-config", type=Path, default=None)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()

import torch

from bb_mimic_nt.student_runtime import StudentPolicy, student_export, student_load_policy


if __name__ == "__main__":
    if args.checkpoint:
        payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        if payload.get("format") != "bb_mimic_nt_student_dagger_v1" or "metadata" not in payload:
            raise ValueError("Checkpoint is not an exportable Student DAgger checkpoint.")
        metadata = payload["metadata"]
        policy = StudentPolicy(metadata["student_hidden_dims"], metadata["activation"])
        policy.student.load_state_dict(payload["student"])
        policy.eval()
    else:
        config = args.student_config or args.student_weights.with_name("student_config.json")
        policy, metadata = student_load_policy(args.student_weights, config, "cpu")
    student_export(policy, args.output, metadata)
    print(f"Student artifacts: {args.output.resolve()}")
