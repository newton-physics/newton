# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Newton's canonical :class:`ControlSignal` constants.

This module publishes the joint-level signal vocabulary Newton itself ships;
controllers and tests / examples import these constants by identity. PID
gains, IK targets, and other application-specific signals are **not**
shipped here — they're shown in the tests / examples / docs as user-defined
signals so users can see the pattern for adding their own.
"""

from __future__ import annotations

import warp as wp

from .utils import ControlSignal

JOINT_Q = ControlSignal(
    dtype=wp.float32,
    ndim=1,
    description="joint positions [m or rad]",
)
"""Joint coordinate vector. Read from a simulation state."""

JOINT_QD = ControlSignal(
    dtype=wp.float32,
    ndim=1,
    description="joint velocities [m/s or rad/s]",
)
"""Joint velocity vector. Read from a simulation state."""

JOINT_TARGET_Q = ControlSignal(
    dtype=wp.float32,
    ndim=1,
    description="commanded joint positions [m or rad]",
)
"""Joint-position target. May appear as either an input or an output
depending on whether the controller emits targets or consumes them."""

JOINT_TARGET_QD = ControlSignal(
    dtype=wp.float32,
    ndim=1,
    description="commanded joint velocities [m/s or rad/s]",
)
"""Joint-velocity target. May appear as either an input or an output."""

JOINT_F = ControlSignal(
    dtype=wp.float32,
    ndim=1,
    description="joint efforts [N or N·m]",
)
"""Joint-effort output."""
