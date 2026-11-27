# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Directional recovery contacts against finite heightfield triangles."""

from typing import Any

import warp as wp

from ..utils.heightfield import HeightfieldData, heightfield_point_below
from .contact_data import CONTACT_SORT_CONVEX_SUB_KEY_BITS, ContactData
from .simplex_solver import create_solve_closest_distance
from .support_function import GeoTypeEx
from .types import GeoType

HEIGHTFIELD_RECOVERY_DEPTH = wp.constant(2.0)
"""Downward extension [m] of the heightfield's collision broad-phase bounds."""

HEIGHTFIELD_RECOVERY_SPEED = wp.constant(1.0)
"""XPBD target speed limit [m/s] for heightfield push-out."""

_CAST_TOLERANCE = wp.constant(1.0e-5)
_FOOTPRINT_TOLERANCE = wp.constant(1.0e-5)
_cast_func = wp.func(inline=False) if tuple(int(v) for v in wp.__version__.split(".")[:2]) >= (1, 18) else wp.func


def create_heightfield_recovery(support_func: Any, writer_func: Any):
    """Create a full-shape cast along local -Z against a heightfield top triangle."""
    solve_gjk = create_solve_closest_distance(support_func)

    @_cast_func
    def recover(
        terrain: Any,
        shape: Any,
        field: HeightfieldData,
        elevations: wp.array[float],
        local_origin: wp.vec3,
        rotation: wp.quat,
        origin: wp.vec3,
        relative_rotation: wp.quat,
        relative_position: wp.vec3,
        rim: bool,
        provider: Any,
        writer: Any,
        contact: ContactData,
    ) -> bool:
        up = wp.vec3(0.0, 0.0, 1.0)
        # Use the physical triangle, with no bottom or skirt in the support map.
        triangle = terrain
        triangle.shape_type = int(GeoTypeEx.TRIANGLE)
        if rim:
            # The exterior has no solid side wall. Require intersection with
            # the real top feature before recovering outside the footprint.
            separated, _a, _b, _n, gap = wp.static(solve_gjk.core)(
                triangle,
                shape,
                relative_rotation,
                relative_position,
                0.0,
                provider,
                max_dist=_CAST_TOLERANCE,
            )
            if separated and gap > _CAST_TOLERANCE:
                return False

        edge_b = triangle.scale
        edge_c = triangle.auxiliary
        normal = wp.cross(edge_b, edge_c)
        if normal[2] <= 0.0:
            return False
        direction = wp.quat_rotate_inv(relative_rotation, -wp.normalize(normal))
        support = wp.quat_rotate(relative_rotation, support_func(shape, direction, provider)) + relative_position
        lift = -wp.dot(support, normal) / normal[2]
        if lift <= 0.0:
            return False

        # Test the footprint in XY. Its conditioning is independent of the
        # slope, unlike a 3D closest-point test on a very steep triangle.
        b = (support[0] * edge_c[1] - support[1] * edge_c[0]) / normal[2]
        c = (edge_b[0] * support[1] - edge_b[1] * support[0]) / normal[2]
        terrain_point = support + lift * up

        # The support plane gives the exact hit when the witness lands
        # inside the face. Otherwise advance against the finite triangle.
        if b < -_FOOTPRINT_TOLERANCE or c < -_FOOTPRINT_TOLERANCE or b + c > 1.0 + _FOOTPRINT_TOLERANCE:
            # For a sphere, the owning triangle is sufficient when its surface
            # reaches this support point. Other shapes benefit from a manifold.
            if shape.shape_type == GeoType.SPHERE and heightfield_point_below(
                field, elevations, local_origin + terrain_point - _CAST_TOLERANCE * up
            ):
                return False
            direction = wp.quat_rotate_inv(relative_rotation, -up)
            lowest = wp.quat_rotate(relative_rotation, support_func(shape, direction, provider)) + relative_position
            lift = wp.max(0.0, wp.max(triangle.scale[2], triangle.auxiliary[2])) - lowest[2] + _CAST_TOLERANCE
            hit = bool(False)
            for _iteration in range(24):
                separated, terrain_point, _b, axis, gap = wp.static(solve_gjk.core)(
                    triangle,
                    shape,
                    relative_rotation,
                    relative_position + lift * up,
                    0.0,
                    provider,
                )
                if not separated or gap <= _CAST_TOLERANCE:
                    hit = True
                    break
                if axis[2] <= 0.0:
                    break
                # The separating support plane gives a lower bound on the
                # advance even when the GJK witness has not fully converged.
                direction = wp.quat_rotate_inv(relative_rotation, -axis)
                support = wp.quat_rotate(relative_rotation, support_func(shape, direction, provider))
                support += relative_position + lift * up
                bound = wp.dot(support, axis) - wp.max(
                    0.0, wp.max(wp.dot(triangle.scale, axis), wp.dot(triangle.auxiliary, axis))
                )
                if bound <= 0.0:
                    break
                lift -= bound / axis[2]
                if lift <= 0.0:
                    break
            if not hit:
                return False
        else:
            # Adjacent grid cells can round their shared edge differently.
            b = wp.max(0.0, b)
            c = wp.max(0.0, c)
            terrain_point = (b * edge_b + c * edge_c) / wp.max(1.0, b + c)
            lift = terrain_point[2] - support[2]

        contact.contact_normal_a_to_b = wp.quat_rotate(rotation, up)
        contact.contact_point_center = wp.quat_rotate(rotation, terrain_point - (0.5 * lift) * up) + origin
        contact.contact_distance = -lift
        writer_func(contact, writer, -1)
        return True

    @wp.func
    def try_recovery(
        terrain: Any,
        shape: Any,
        field: HeightfieldData,
        elevations: wp.array[float],
        field_transform: wp.transform,
        origin: wp.vec3,
        position: wp.vec3,
        orientation: wp.quat,
        pair: wp.vec3i,
        margin_a: float,
        margin_b: float,
        gap: float,
        provider: Any,
        writer: Any,
    ) -> bool:
        if shape.shape_type == GeoType.PLANE:
            return False
        field_rotation = wp.transform_get_rotation(field_transform)
        local_position = wp.quat_rotate_inv(field_rotation, position - wp.transform_get_translation(field_transform))
        center = local_position
        if shape.shape_type == GeoType.CONVEX_MESH:
            center += wp.quat_rotate_inv(field_rotation, wp.quat_rotate(orientation, shape.center))
        rim = wp.abs(center[0]) > field.hx or wp.abs(center[1]) > field.hy
        if not rim:
            if not heightfield_point_below(field, elevations, center):
                return False
            world_up = wp.quat_rotate(field_rotation, wp.vec3(0.0, 0.0, 1.0))
            local_up = wp.quat_rotate_inv(orientation, world_up)
            highest = support_func(shape, local_up, provider)
            center[2] = local_position[2] + wp.dot(highest, local_up)
            if not heightfield_point_below(field, elevations, center):
                return False
        relative_position = wp.quat_rotate_inv(field_rotation, position - origin)
        triangle_top = wp.max(0.0, wp.max(terrain.scale[2], terrain.auxiliary[2]))
        if rim and relative_position[2] >= triangle_top:
            return False

        contact = ContactData()
        contact.shape_a = pair[0]
        contact.shape_b = pair[1]
        contact.margin_a = margin_a
        contact.margin_b = margin_b
        contact.gap_sum = gap
        contact.sort_sub_key = ((pair[2] << 1) | 1) << CONTACT_SORT_CONVEX_SUB_KEY_BITS
        recovered = recover(
            terrain,
            shape,
            field,
            elevations,
            local_position - relative_position,
            field_rotation,
            origin,
            wp.quat_inverse(field_rotation) * orientation,
            relative_position,
            rim,
            provider,
            writer,
            contact,
        )
        # For a buried object only the top triangles supply contacts.
        return not rim or recovered

    return try_recovery
