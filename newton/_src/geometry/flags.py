# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

from enum import IntEnum, IntFlag


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


class MeshSignMethod(IntEnum):
    """Method used to determine the inside/outside sign of a mesh point query."""

    NORMAL = 0
    """Angle-weighted closest-face pseudo-normal; robust for open surfaces."""
    PARITY = 1
    """Ray-crossing parity; correct and cheap for watertight (closed) meshes."""


class MeshProperties(IntFlag):
    """Per-shape mesh properties consumed by the collision kernels."""

    WATERTIGHT = 1 << 0
    """The source mesh is closed (every edge shared by exactly two triangles)."""


__all__ = [
    "MeshProperties",
    "MeshSignMethod",
    "ParticleFlags",
    "ShapeFlags",
]
