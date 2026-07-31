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


class _ActuatorBiasType(IntEnum):
    NONE = 0
    AFFINE = 1
    MUSCLE = 2
    USER = 3


class _ActuatorDynamicsType(IntEnum):
    NONE = 0
    INTEGRATOR = 1
    FILTER = 2
    FILTER_EXACT = 3
    MUSCLE = 4
    USER = 5


class _ActuatorGainType(IntEnum):
    FIXED = 0
    AFFINE = 1
    MUSCLE = 2
    USER = 3


__all__ = ["EqType"]
