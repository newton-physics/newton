# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Shared constants for Newton's MuJoCo solver integration."""

from __future__ import annotations

DEFAULT_LIMIT_GAIN_RTOL = 1.0e-5
"""Relative tolerance for detecting imported MuJoCo default joint-limit gains.

Used to recognise the ``joint_limit_ke`` / ``joint_limit_kd`` values
that result from importing MuJoCo's implicit default ``solreflimit``
(``(0.02, 1.0)``), so that ``SOLREF_MODE_MJCF_DEFAULT`` joints stay in
the "preserve compile-time default" state until the user actually
edits the gains. A relative tolerance lets the threshold scale with the
default magnitudes ``DEFAULT_LIMIT_KE = 2500`` and
``DEFAULT_LIMIT_KD = 100`` instead of being invisible at ``ke≈2500``.
"""

DEFAULT_LIMIT_KD = 100.0
"""Newton damping gain equivalent to MuJoCo's implicit default joint-limit solref."""

DEFAULT_LIMIT_KE = 2500.0
"""Newton stiffness gain equivalent to MuJoCo's implicit default joint-limit solref."""

DEFAULT_LIMIT_SOLREF = (0.02, 1.0)
"""MuJoCo's implicit default joint-limit solref pair."""

DEFAULT_LIMIT_SOLREF_DAMPRATIO = DEFAULT_LIMIT_SOLREF[1]
"""MuJoCo's implicit default joint-limit solref damping ratio component."""

DEFAULT_LIMIT_SOLREF_TIMECONST = DEFAULT_LIMIT_SOLREF[0]
"""MuJoCo's implicit default joint-limit solref time-constant component."""

HINGE_CONNECT_AXIS_OFFSET = 0.1
"""Distance [m] along the hinge axis for the second CONNECT constraint point of a revolute loop joint."""

KINEMATIC_ARMATURE = 1.0e10
"""Large MuJoCo armature value used to make kinematic-body DOFs effectively immovable."""

MJ_MINMU = 1.0e-5
"""MuJoCo's minimum friction coefficient clamp."""

MJ_MINVAL = 2.220446049250313e-16
"""MuJoCo's minimum positive scalar guard value."""

SOLREF_MODE_FORCE_SPACE = 0
"""Interpret joint-limit gains as Newton force-space ``joint_limit_ke``/``joint_limit_kd``."""

SOLREF_MODE_RAW = 1
"""Interpret ``mujoco.solreflimit`` as a raw MuJoCo-authored solref value."""

SOLREF_MODE_MJCF_DEFAULT = 2
"""Preserve MuJoCo's implicit joint-limit solref until imported default gains are edited."""
