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

    # The mesh sign method occupies bits 5-7 as a 3-bit field holding a
    # continuous value (matching MeshSignMethod), not one-hot flag bits.
    MESH_SIGN_AUTO = 0
    """Select the mesh sign method automatically from the mesh topology
    (default). Watertight meshes use the ray-crossing parity sign; open
    meshes use each path's non-watertight method: the closest-face
    pseudo-normal for runtime mesh queries (soft and particle contacts, the
    mesh-mesh SDF narrow phase of :class:`~newton.CollisionPipeline`) and
    winding numbers for SDFs baked during model finalization.

    This is a field value of zero, not a testable bit: compare
    ``flags & ShapeFlags.MESH_SIGN_METHOD_MASK`` against it.
    """

    MESH_SIGN_NORMAL = 1 << 5
    """Treat the mesh as non-watertight regardless of topology.

    Runtime mesh queries use the closest-face pseudo-normal sign; an SDF
    baked during model finalization uses winding numbers (the bake pipeline
    has no pseudo-normal sampler). Prefer this for open geometry such as
    planes or sheets, where parity has no consistent inside. Pre-built SDFs
    (:meth:`~newton.Mesh.build_sdf` called before the shape is added) are
    already baked and remain unaffected.
    """

    MESH_SIGN_PARITY = 2 << 5
    """Treat the mesh as watertight regardless of topology.

    Both runtime mesh queries and SDFs baked during model finalization use
    the ray-crossing parity sign, which is correct and cheap for closed
    meshes but unreliable on open ones. Set it to recover parity when a
    closed mesh is conservatively misdetected as open by
    :attr:`~newton.Mesh.is_watertight`. Pre-built SDFs remain unaffected.
    """

    MESH_SIGN_METHOD_MASK = 0b111 << 5
    """Mask over the 3-bit mesh sign-method field. Field values beyond
    :attr:`MESH_SIGN_PARITY` are reserved for future methods (e.g. winding
    numbers at runtime) and are rejected at model finalization."""


__all__ = [
    "ParticleFlags",
    "ShapeFlags",
]
