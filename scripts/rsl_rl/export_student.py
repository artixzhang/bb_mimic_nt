#!/usr/bin/env python3
"""Export the latest Student by default, or a selected checkpoint/weight artifact."""

from __future__ import annotations

import argparse
from pathlib import Path
import re

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LOGS = PROJECT_ROOT / 'logs/rsl_rl/g1_shoot_student_dagger_v1'


def latest_student_checkpoint(logs_dir: Path) -> Path:
    """Newest timestamped run with checkpoints; greatest numeric model index."""
    if not logs_dir.is_dir():
        raise FileNotFoundError(f'Student log directory does not exist: {logs_dir}')
    runs = [path for path in logs_dir.iterdir() if path.is_dir()
            and re.match(r'^\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}(?:_|$)', path.name)]
    for run in sorted(runs, key=lambda path: path.name, reverse=True):
        checkpoints = []
        for path in run.glob('model_*.pt'):
            match = re.fullmatch(r'model_(\d+)\.pt', path.name)
            if match and path.is_file() and path.stat().st_size > 0:
                checkpoints.append((int(match.group(1)), path))
        if checkpoints:
            return max(checkpoints, key=lambda item: item[0])[1]
    raise FileNotFoundError(f'No model_N.pt checkpoint found under {logs_dir}')


def export_destination(source: Path, output: Path | None, output_root: Path = PROJECT_ROOT / 'outputs') -> Path:
    if output is not None:
        return output
    run = source.parent.parent if source.parent.name == 'exported' else source.parent
    return output_root / f'student_{run.name}'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    source.add_argument('--checkpoint', type=Path, help='A model_N.pt checkpoint; default: latest Student run/model.')
    source.add_argument('--student-weights', type=Path, help='Re-export a student_weights.pt artifact.')
    parser.add_argument('--student-config', type=Path, default=None)
    parser.add_argument('--output', type=Path, default=None, help='Default: outputs/student_<run-time>.')
    parser.add_argument('--logs-dir', type=Path, default=DEFAULT_LOGS, help='Student run directory used for automatic selection.')
    args = parser.parse_args()
    if args.student_config and not args.student_weights:
        parser.error('--student-config applies only to --student-weights')
    import torch
    from bb_mimic_nt.student_runtime import student_export, student_load_policy, student_policy_from_state

    if args.student_weights:
        selected = args.student_weights
        policy, metadata = student_load_policy(selected, args.student_config, 'cpu')
    else:
        selected = args.checkpoint or latest_student_checkpoint(args.logs_dir)
        print(f'Selected Student checkpoint: {selected.resolve()}', flush=True)
        payload = torch.load(selected, map_location='cpu', weights_only=False)
        if payload.get('format') != 'bb_mimic_nt_student_dagger_v1' or 'metadata' not in payload:
            raise ValueError(f'Checkpoint is not an exportable Student DAgger checkpoint: {selected}')
        metadata = dict(payload['metadata'])
        metadata['export_source'] = dict(run=selected.parent.name, checkpoint=selected.name,
                                        iteration=payload['iteration'])
        policy = student_policy_from_state(payload['student'], metadata, 'cpu')
    destination = export_destination(selected, args.output)
    student_export(policy, destination, metadata)
    print(f'Student artifacts: {destination.resolve()}')


if __name__ == '__main__':
    main()
