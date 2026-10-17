"""CLI migration errors must be reported before a simulator is started."""

from pathlib import Path
import importlib.util
import subprocess
import sys

import pytest

pytestmark = pytest.mark.skipif(importlib.util.find_spec('isaaclab') is None, reason='Isaac Lab is not installed')
ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / 'scripts/rsl_rl/train_dagger.py'


def test_help_exposes_ppo_controls_without_requiring_teacher():
    result = subprocess.run([sys.executable, str(SCRIPT), '--help'], cwd=ROOT, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert '--num-learning-epochs' in result.stdout and '--bc-coef' in result.stdout
    assert '--replay-capacity' not in result.stdout


@pytest.mark.parametrize('option', ['--replay-capacity=20', '--updates-per-iteration=32', '--save-replay'])
def test_removed_dagger_options_fail_before_launch(option):
    result = subprocess.run([sys.executable, str(SCRIPT), '--teacher-checkpoint', 'unused.pt', option],
                            cwd=ROOT, capture_output=True, text=True)
    assert result.returncode == 2
    assert 'was removed' in result.stderr
