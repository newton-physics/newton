# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""MuJoCo solver enums."""

from enum import IntEnum


class EqType(IntEnum):
    """MuJoCo equality constraint type."""

    CONNECT = 0
    """Constrains two bodies at a point (like a ball joint)."""

    WELD = 1
    """Welds two bodies together (like a fixed joint)."""

    JOINT = 2
    """Constrains one scalar joint coordinate to a quartic polynomial of another."""


# Mirrors of MuJoCo's mjtBias/mjtDyn/mjtGain, duplicated so MJCF and USD import do not
# need MuJoCo. Types Newton does not support are listed too, so Newton's ordinals track
# MuJoCo's. TestActuatorTypes pins these against mujoco.mjt*.
class _ActuatorBiasType(IntEnum):
    NONE = 0
    AFFINE = 1
    MUSCLE = 2
    DCMOTOR = 3
    SO3 = 4
    USER = 5


class _ActuatorDynamicsType(IntEnum):
    NONE = 0
    INTEGRATOR = 1
    FILTER = 2
    FILTER_EXACT = 3
    MUSCLE = 4
    DCMOTOR = 5
    PID = 6
    USER = 7


class _ActuatorGainType(IntEnum):
    FIXED = 0
    AFFINE = 1
    MUSCLE = 2
    DCMOTOR = 3
    SO3 = 4
    PID = 5
    USER = 6


__all__ = ["EqType"]
