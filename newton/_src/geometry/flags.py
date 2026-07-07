# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

from enum import IntEnum


# Particle flags
class ParticleFlags(IntEnum):
    """
    Flags for particle properties.
    """

    ACTIVE = 1 << 0
    """Indicates that the particle is active."""

    PROXY = 1 << 1
    """Indicates that the particle is a solver-coupling proxy.

    .. experimental::

        This flag is part of the experimental coupled-solver contract and may
        change without prior notice.
    """


# Shape flags
class ShapeFlags(IntEnum):
    """
    Flags for shape properties.
    """

    VISIBLE = 1 << 0
    """Indicates that the shape is visible."""

    COLLIDE_SHAPES = 1 << 1
    """Indicates that the shape collides with other shapes."""

    COLLIDE_PARTICLES = 1 << 2
    """Indicates that the shape collides with particles."""

    SITE = 1 << 3
    """Indicates that the shape is a site (non-colliding reference point)."""

    HYDROELASTIC = 1 << 4
    """Indicates that the shape uses hydroelastic collision."""

    MESH_SIGN_NORMAL = 1 << 5
    """Use the closest face normal to determine the sign of mesh point queries."""

    MESH_SIGN_PARITY = 1 << 6
    """Use ray intersection parity to determine the sign of mesh point queries."""

    MESH_SIGN_METHOD_MASK = 0b111 << 5
    """Bit mask for the encoded mesh sign method."""


__all__ = [
    "ParticleFlags",
    "ShapeFlags",
]
