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

KINEMATIC_INVWEIGHT_SCALE = 1.0e-4
"""Factor on MuJoCo ``body_invweight0`` of kinematic bodies.

Regularizes their welds and contacts as nearly infinite mass; exactly zero makes
those rows too stiff for MuJoCo Warp's float32 CG solver."""

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
