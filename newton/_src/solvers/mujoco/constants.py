# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Shared constants for Newton's MuJoCo solver integration."""

from __future__ import annotations

DEFAULT_LIMIT_SOLREF = (0.02, 1.0)
"""MuJoCo's implicit default joint-limit solref pair."""

DEFAULT_LIMIT_SOLREF_DAMPRATIO = DEFAULT_LIMIT_SOLREF[1]
"""MuJoCo's implicit default joint-limit solref damping ratio component."""

DEFAULT_LIMIT_SOLREF_TIMECONST = DEFAULT_LIMIT_SOLREF[0]
"""MuJoCo's implicit default joint-limit solref time-constant component."""

HINGE_CONNECT_HALF_SPAN = 0.05
"""Distance [m] along the hinge axis from a revolute loop joint's anchor to each of its two CONNECT points."""

KINEMATIC_WELD_TIMECONST_STEPS = 2.0
"""Kinematic weld time constant in timesteps; the stiffest MuJoCo's REFSAFE allows."""

KINEMATIC_WELD_TIMECONST_STEPS_RK4 = 8.0
"""Kinematic weld time constant in timesteps under RK4, whose intermediate stages destabilize stiffer welds."""

KINEMATIC_WELD_DAMPRATIO = 1.0
"""Kinematic weld damping ratio."""

KINEMATIC_WELD_SOLIMP = (0.99, 0.99, 0.001, 0.5, 2.0)
"""Constant-impedance weld solimp, so the target look-ahead cancels the weld damping exactly."""

MJ_MINMU = 1.0e-5
"""MuJoCo's minimum friction coefficient clamp."""

MJ_MINVAL = 2.220446049250313e-16
"""MuJoCo's minimum positive scalar guard value."""

SOLREF_MODE_FORCE_SPACE = 0
"""Interpret joint-limit gains as Newton force-space ``joint_limit_ke``/``joint_limit_kd``."""

SOLREF_MODE_RAW = 1
"""Interpret ``mujoco.solreflimit`` as a raw MuJoCo-authored solref value."""

SOLREF_MODE_MJCF_DEFAULT = 2
"""Preserve MuJoCo's implicit joint-limit solref until Newton's generic gains are edited."""
