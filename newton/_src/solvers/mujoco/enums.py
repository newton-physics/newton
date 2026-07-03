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


__all__ = ["EqType"]
