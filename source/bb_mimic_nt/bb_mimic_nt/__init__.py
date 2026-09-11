# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unitree G1 basketball imitation learning package.

Task registration is intentionally kept in :mod:`bb_mimic_nt.tasks`.  Keeping
the top-level package light makes the trajectory tools usable without starting
Isaac Sim first.
"""

__version__ = "0.5.0"


def register_tasks():
    """Import task registrations after Isaac Sim has initialized."""
    from . import tasks

    return tasks


# Omniverse imports the extension module after Kit initialization. Keep normal
# Python imports lightweight so offline trajectory tooling remains CPU-only.
try:
    import omni  # noqa: F401
except ModuleNotFoundError:
    pass
else:
    register_tasks()
