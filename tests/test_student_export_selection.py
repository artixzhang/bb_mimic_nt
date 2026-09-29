"""Automatic run/checkpoint selection and default export directory."""
import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location('student_export_cli', Path(__file__).resolve().parents[1] / 'scripts/rsl_rl/export_student.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_latest_run_numeric_checkpoint_and_empty_run_skip(tmp_path):
    old = tmp_path / '2026-09-28_23-50-26'
    newest = tmp_path / '2026-09-29_17-26-30'
    empty = tmp_path / '2026-09-29_19-00-00'
    for path in (old, newest, empty):
        path.mkdir()
    (old / 'model_9999.pt').write_bytes(b'old')
    for index in (9, 50, 650):
        (newest / f'model_{index}.pt').write_bytes(b'new')
    (empty / 'model_700.pt').touch()
    (newest / 'model_latest.pt').write_bytes(b'not numbered')
    assert module.latest_student_checkpoint(tmp_path) == newest / 'model_650.pt'
    assert module.export_destination(newest / 'model_650.pt', None, tmp_path / 'outputs') == tmp_path / 'outputs/student_2026-09-29_17-26-30'


def test_explicit_output_and_weight_artifact_run_name(tmp_path):
    source = tmp_path / '2026-09-29_17-26-30/exported/student_weights.pt'
    assert module.export_destination(source, None, tmp_path) == tmp_path / 'student_2026-09-29_17-26-30'
    assert module.export_destination(source, tmp_path / 'custom') == tmp_path / 'custom'
    with pytest.raises(FileNotFoundError, match='No model_N'):
        module.latest_student_checkpoint(tmp_path)
