# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""KAMINO: Shape Utilities"""

from __future__ import annotations

import warp as wp

from .....geometry.types import GeoType

###
# Module interface
###

__all__ = [
    "GeoType",
    "max_contacts_for_shape_pair",
]


###
# Module configs
###

wp.set_module_options({"enable_backward": False})


###
# Utilities
###

# Contact counts for mesh/heightfield pairs are dynamic (bounded by the
# pipeline's max_contacts_per_pair setting).  The values below are
# conservative upper-bound estimates used for capacity allocation.
_MESH_CONVEX_MAX = 32
_MESH_MESH_MAX = 64


@wp.func
def max_contacts_for_shape_pair(type_a: int, type_b: int) -> tuple[int, int]:
    """
    Count the number of potential contact points for a collision pair in both
    directions of the collision pair (collisions from A to B and from B to A).

    Shape types are canonicalized such that the type of shape A is less than or equal to the type of shape B.

    Args:
        type_a: First shape type as :class:`GeoType` integer value.
        type_b: Second shape type as :class:`GeoType` integer value.

    Returns:
        Number of contact points for collisions between A->B and B->A.
        Each component independently bounds the contacts generated in its
        direction. The reverse capacity is zero only when the pair's
        narrow-phase implementation emits contacts in canonical order alone;
        otherwise it reserves capacity for the reverse pass.
    """
    # Ensure the shape types are ordered canonically
    if type_a > type_b:
        type_a, type_b = type_b, type_a

    return _max_contacts_for_shape_pair_impl(type_a, type_b)


@wp.func
def _max_contacts_for_shape_pair_impl(type_a: int, type_b: int) -> tuple[int, int]:
    """
    Return the contact capacity for a canonical shape pair without reordering.

    This is used for testing purposes, asserting that type_a > type_b always returns (0, 0).
    This enforces that the implementation doesn't accidentally specify an unreachable pair.
    """
    if type_a == GeoType.PLANE:
        if type_b == GeoType.HFIELD:
            return _MESH_CONVEX_MAX, 0
        elif type_b == GeoType.SPHERE:
            return 1, 0
        elif type_b == GeoType.CAPSULE:
            return 2, 0
        elif type_b == GeoType.ELLIPSOID:
            return 1, 0
        elif type_b == GeoType.CYLINDER:
            return 4, 0
        elif type_b == GeoType.BOX:
            return 4, 0
        elif type_b == GeoType.MESH or type_b == GeoType.CONVEX_MESH:
            return _MESH_CONVEX_MAX, 0
        elif type_b == GeoType.CONE:
            return 5, 0

    elif type_a == GeoType.HFIELD:
        if type_b == GeoType.MESH:
            return _MESH_MESH_MAX, 0
        elif type_b >= GeoType.HFIELD:
            return _MESH_CONVEX_MAX, 0

    elif type_a == GeoType.SPHERE:
        if type_b == GeoType.MESH or type_b == GeoType.CONVEX_MESH:
            return _MESH_CONVEX_MAX, 0
        elif type_b >= GeoType.SPHERE:
            return 1, 0

    elif type_a == GeoType.CAPSULE:
        if type_b == GeoType.CAPSULE:
            return 2, 0
        elif type_b == GeoType.ELLIPSOID:
            return 1, 0
        elif type_b == GeoType.CYLINDER:
            return 5, 0
        elif type_b == GeoType.BOX:
            return 5, 0
        elif type_b == GeoType.MESH or type_b == GeoType.CONVEX_MESH:
            return _MESH_CONVEX_MAX, 0
        elif type_b == GeoType.CONE:
            return 5, 0

    elif type_a == GeoType.ELLIPSOID:
        if type_b == GeoType.ELLIPSOID:
            return 1, 0
        elif type_b == GeoType.CYLINDER:
            return 1, 0
        elif type_b == GeoType.BOX:
            return 1, 0
        elif type_b == GeoType.MESH or type_b == GeoType.CONVEX_MESH:
            return _MESH_CONVEX_MAX, 0
        elif type_b == GeoType.CONE:
            return 1, 0

    elif type_a == GeoType.CYLINDER:
        if type_b == GeoType.CYLINDER:
            return 5, 0
        elif type_b == GeoType.BOX:
            return 5, 0
        elif type_b == GeoType.MESH or type_b == GeoType.CONVEX_MESH:
            return _MESH_CONVEX_MAX, 0
        elif type_b == GeoType.CONE:
            return 5, 0

    elif type_a == GeoType.BOX:
        if type_b == GeoType.BOX:
            return 8, 0
        elif type_b == GeoType.MESH or type_b == GeoType.CONVEX_MESH:
            return _MESH_CONVEX_MAX, 0
        elif type_b == GeoType.CONE:
            return 5, 0

    elif type_a == GeoType.MESH:
        if type_b == GeoType.MESH:
            return _MESH_MESH_MAX, 0
        elif type_b == GeoType.CONE:
            return _MESH_CONVEX_MAX, 0
        elif type_b == GeoType.CONVEX_MESH:
            return _MESH_MESH_MAX, 0

    elif type_a == GeoType.CONE:
        if type_b == GeoType.CONE:
            return 5, 0
        elif type_b == GeoType.CONVEX_MESH:
            return _MESH_CONVEX_MAX, 0

    elif type_a == GeoType.CONVEX_MESH:
        if type_b == GeoType.CONVEX_MESH:
            return _MESH_MESH_MAX, 0

    # unsupported type combination
    return 0, 0
