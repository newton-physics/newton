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
    """Force the closest-face pseudo-normal sign for mesh point queries,
    overriding the automatic watertight-based selection.

    Honored by the runtime paths that query a triangle mesh directly: soft and
    particle contacts, and the mesh-mesh SDF narrow phase driven by
    :class:`~newton.CollisionPipeline`. It has no effect on
    :meth:`~newton.Mesh.build_sdf`, which chooses its own sign method through
    its ``sign_method`` argument. Prefer this for open (non-watertight)
    geometry such as planes or sheets, where parity has no consistent inside.
    """

    MESH_SIGN_PARITY = 1 << 6
    """Force the ray-crossing parity sign for mesh point queries, overriding the
    automatic watertight-based selection.

    Honored by the same runtime paths as :attr:`MESH_SIGN_NORMAL`. Parity is
    correct and cheap for watertight (closed) meshes but unreliable on open
    ones; set it to recover parity when a closed mesh is conservatively
    misdetected as open by :attr:`~newton.Mesh.is_watertight`. Leaving both
    mesh-sign bits unset selects automatically (parity for watertight meshes,
    normal otherwise).
    """

    MESH_SIGN_METHOD_MASK = MESH_SIGN_NORMAL | MESH_SIGN_PARITY
    """Mask over the two mesh sign-method bits. Setting both at once is an
    ambiguous encoding and is rejected at model finalization."""


__all__ = [
    "ParticleFlags",
    "ShapeFlags",
]
