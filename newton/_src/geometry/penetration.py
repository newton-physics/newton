# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Find box penetration directions with constant-storage separating axes."""

from typing import Any

import warp as wp

from .support_function import _has_verified_hull, unpack_mesh_ptr
from .types import GeoType

# Tangent offset [unitless] of the tilted certificate normals, about 2 degrees.
_CERTIFICATE_TILT = wp.constant(0.035)


@wp.func
def box_polyhedron_pair(a: Any, b: Any) -> bool:
    """Select box pairs whose portal normal needs a minimum-depth certificate."""
    return (
        a.shape_type == int(GeoType.BOX)
        and (b.shape_type == int(GeoType.BOX) or b.shape_type == int(GeoType.CONVEX_MESH))
    ) or (a.shape_type == int(GeoType.CONVEX_MESH) and b.shape_type == int(GeoType.BOX))


@wp.func
def _axis(index: int) -> wp.vec3:
    """Return a box-frame coordinate axis."""
    axis = wp.vec3(0.0)
    axis[index] = 1.0
    return axis


@wp.func
def _box_triangle_axis(first: wp.vec3, second: wp.vec3, third: wp.vec3, index: int) -> wp.vec3:
    """Return one of the 13 facet-normal candidates of a box and a triangle."""
    if index < 3:
        return _axis(index)
    if index == 3:
        return wp.cross(second - first, third - first)
    edge = second - first
    if index >= 10:
        edge = first - third
    elif index >= 7:
        edge = third - second
    return wp.cross(_axis(index % 3), edge)


@wp.func
def _box_triangle_depth(half: wp.vec3, first: wp.vec3, second: wp.vec3, third: wp.vec3) -> tuple[float, wp.vec3]:
    """Return the penetration depth of a box and a triangle of partner points.

    The triangle may degenerate to a segment or a point. The returned unit
    direction moves the triangle out of the box by the returned depth.
    """
    depth = float(1.0e30)
    direction = wp.vec3(1.0, 0.0, 0.0)
    for index in range(13):
        axis = _box_triangle_axis(first, second, third, index)
        length = wp.length(axis)
        if length > 0.0:
            axis /= length
            radius = wp.dot(wp.abs(axis), half)
            d0 = wp.dot(axis, first)
            d1 = wp.dot(axis, second)
            d2 = wp.dot(axis, third)
            positive = radius - wp.min(d0, wp.min(d1, d2))
            negative = radius + wp.max(d0, wp.max(d1, d2))
            if positive < depth:
                depth = positive
                direction = axis
            if negative < depth:
                depth = negative
                direction = -axis
    return depth, direction


@wp.func
def _box_box_depth(half_a: wp.vec3, half_b: wp.vec3, rotation: wp.quat, position: wp.vec3) -> tuple[wp.vec3, float]:
    """Return the minimum-depth normal and depth of two boxes in A's frame.

    The face normals and edge cross products include every Minkowski facet
    normal. Each axis's closed-form support-plane overlap bounds the minimum
    from above, so degenerate cross products cannot undercut it. The normal
    points from A toward B, matching the Minkowski support convention.
    """
    axes_b = wp.quat_to_matrix(rotation)
    normal = wp.vec3(1.0, 0.0, 0.0)
    depth = float(1.0e30)
    for index in range(15):
        axis = wp.vec3(0.0)
        if index < 3:
            axis = _axis(index)
        elif index < 6:
            axis = axes_b * _axis(index - 3)
        else:
            axis = wp.cross(_axis((index - 6) // 3), axes_b * _axis((index - 6) % 3))
        length = wp.length(axis)
        if length > 0.0:
            axis /= length
            radius = wp.dot(wp.abs(axis), half_a) + wp.dot(wp.abs(wp.transpose(axes_b) * axis), half_b)
            offset = wp.dot(axis, position)
            overlap = radius - wp.abs(offset)
            if overlap < depth:
                depth = overlap
                normal = axis if offset >= 0.0 else -axis
    return normal, depth


def create_solve_box_penetration(support: Any):
    """Find minimum depth for a box against a box or convex mesh.

    Box pairs use the closed-form separating-axis test. For convex meshes,
    cheap certificates come first: a box plus a triangle of genuine partner
    points is an inner Minkowski body, so its depth bounds the minimum from
    below, and a support plane bounds it from above. Only pairs neither
    certificate resolves stream the complete separating-axis basis: face
    normals and edge cross products include every Minkowski facet normal of
    a hull triangulation. Storage is constant and no expanding polytope is
    constructed.
    """

    @wp.func
    def test_direction(
        a: Any,
        b: Any,
        rotation: wp.quat,
        position: wp.vec3,
        provider: Any,
        normal: wp.vec3,
        best_normal: wp.vec3,
        best_depth: float,
        best_vertex: wp.vec3,
        recent_vertex: wp.vec3,
    ) -> tuple[wp.vec3, float, wp.vec3, wp.vec3]:
        # Genuine support points bound this direction from below.
        # Repeated or inferior axes need no further support query.
        lower = wp.max(wp.dot(best_vertex, normal), wp.dot(recent_vertex, normal))
        if lower < best_depth:
            vertex = support(a, b, normal, rotation, position, 0.0, provider)
            recent_vertex = vertex.BtoA
            depth = wp.dot(vertex.BtoA, normal)
            if depth < best_depth:
                best_depth = depth
                best_normal = normal
                best_vertex = vertex.BtoA
        return best_normal, best_depth, best_vertex, recent_vertex

    @wp.func
    def test_axis(
        a: Any,
        b: Any,
        rotation: wp.quat,
        position: wp.vec3,
        provider: Any,
        axis: wp.vec3,
        best_normal: wp.vec3,
        best_depth: float,
        best_vertex: wp.vec3,
        recent_vertex: wp.vec3,
    ) -> tuple[wp.vec3, float, wp.vec3, wp.vec3]:
        length = wp.length(axis)
        if length > 0.0:
            direction = axis / length
            best_normal, best_depth, best_vertex, recent_vertex = test_direction(
                a, b, rotation, position, provider, direction, best_normal, best_depth, best_vertex, recent_vertex
            )
            best_normal, best_depth, best_vertex, recent_vertex = test_direction(
                a, b, rotation, position, provider, -direction, best_normal, best_depth, best_vertex, recent_vertex
            )
        return best_normal, best_depth, best_vertex, recent_vertex

    @wp.func
    def solve_box_mesh(
        a: Any,
        b: Any,
        rotation: wp.quat,
        position: wp.vec3,
        provider: Any,
        normal: wp.vec3,
        depth: float,
        best_vertex: wp.vec3,
        recent_vertex: wp.vec3,
    ) -> tuple[wp.vec3, float]:
        """Stream mesh axes, bounding depths with the box's closed-form support.

        Work in the mesh frame. For a mesh direction ``m``, the depth is
        ``h_mesh(m) + h_box(-m)``. A triangle vertex bounds ``h_mesh`` from
        below, exactly when its feature supports ``m``, so an axis needs a
        support query only when its bound still beats the current minimum.
        Triangles need not be hull faces; only exact queries set the depth.
        """
        mesh_a = a.shape_type == int(GeoType.CONVEX_MESH)
        mesh_scale = b.scale
        mesh_ptr = b.auxiliary
        half = a.scale
        mesh_rotation = rotation
        to_box = rotation
        # Mesh origin minus box origin in the mesh frame.
        offset = wp.quat_rotate_inv(rotation, position)
        # The common normal points from B to A; flip mesh directions for B.
        side = float(-1.0)
        if mesh_a:
            mesh_scale = a.scale
            mesh_ptr = a.auxiliary
            half = b.scale
            mesh_rotation = wp.quat_identity()
            to_box = wp.quat_inverse(rotation)
            offset = -position
            side = 1.0
        # Rows are the box axes in the mesh frame.
        box_axes = wp.quat_to_matrix(to_box)
        mesh = wp.mesh_get(unpack_mesh_ptr(mesh_ptr))
        # Any vertex average is interior and orients the triangles.
        center = wp.vec3(0.0)
        extent = float(0.0)
        for i in range(mesh.points.shape[0]):
            point = wp.cw_mul(mesh.points[i], mesh_scale)
            center += point
            extent = wp.max(extent, wp.max(wp.abs(point[0]), wp.max(wp.abs(point[1]), wp.abs(point[2]))))
        center /= float(mesh.points.shape[0])
        tolerance = 1.0e-6 * extent
        for triangle in range(mesh.indices.shape[0] / 3):
            p0 = wp.cw_mul(mesh.points[mesh.indices[3 * triangle]], mesh_scale)
            p1 = wp.cw_mul(mesh.points[mesh.indices[3 * triangle + 1]], mesh_scale)
            p2 = wp.cw_mul(mesh.points[mesh.indices[3 * triangle + 2]], mesh_scale)
            face = wp.cross(p1 - p0, p2 - p0)
            length = wp.length(face)
            if length > 0.0:
                face /= length
                for orientation in range(2):
                    m = face if orientation == 0 else -face
                    # Skip inward normals; a flat hull keeps both.
                    if wp.dot(m, p0 - center) >= -tolerance:
                        lower = wp.dot(m, p0 + offset) + wp.dot(wp.abs(box_axes * m), half)
                        if lower < depth:
                            normal, depth, best_vertex, recent_vertex = test_direction(
                                a,
                                b,
                                rotation,
                                position,
                                provider,
                                side * wp.quat_rotate(mesh_rotation, m),
                                normal,
                                depth,
                                best_vertex,
                                recent_vertex,
                            )
            for corner in range(3):
                start = p0
                end = p1
                opposite = p2
                if corner == 1:
                    start = p1
                    end = p2
                    opposite = p0
                elif corner == 2:
                    start = p2
                    end = p0
                    opposite = p1
                for axis in range(3):
                    m = wp.cross(box_axes[axis], end - start)
                    length = wp.length(m)
                    if length > 0.0:
                        m /= length
                        reach = wp.dot(m, opposite - start)
                        for _orientation in range(2):
                            # A supporting edge has its triangle behind it.
                            if reach <= tolerance:
                                lower = wp.dot(m, start + offset) + wp.dot(wp.abs(box_axes * m), half)
                                if lower < depth:
                                    normal, depth, best_vertex, recent_vertex = test_direction(
                                        a,
                                        b,
                                        rotation,
                                        position,
                                        provider,
                                        side * wp.quat_rotate(mesh_rotation, m),
                                        normal,
                                        depth,
                                        best_vertex,
                                        recent_vertex,
                                    )
                            m = -m
                            reach = -reach
        # Box faces are the remaining Minkowski facet normals.
        for axis in range(3):
            normal, depth, best_vertex, recent_vertex = test_axis(
                a,
                b,
                rotation,
                position,
                provider,
                side * wp.quat_rotate(mesh_rotation, box_axes[axis]),
                normal,
                depth,
                best_vertex,
                recent_vertex,
            )
        return normal, depth

    @wp.func
    def partner_point(box_a: bool, rotation: wp.quat, position: wp.vec3, vertex: Any) -> wp.vec3:
        """Return the non-box support point in the box frame."""
        if box_a:
            return vertex.B
        return wp.quat_rotate_inv(rotation, vertex.B + vertex.BtoA - position)

    @wp.func
    def certify(
        a: Any,
        b: Any,
        rotation: wp.quat,
        position: wp.vec3,
        provider: Any,
        box_a: bool,
        half: wp.vec3,
        first: wp.vec3,
        second: wp.vec3,
        third: wp.vec3,
    ) -> tuple[wp.vec3, float, float, wp.vec3]:
        """Bound the minimum depth from both sides with partner points.

        The box plus a triangle of genuine partner points is an inner
        Minkowski body, so its depth bounds the minimum from below. The
        support plane along its minimizing axis bounds it from above.
        """
        lower, direction = _box_triangle_depth(half, first, second, third)
        # The box-frame direction translates the partner out of the box.
        normal = direction if box_a else -wp.quat_rotate(rotation, direction)
        vertex = support(a, b, normal, rotation, position, 0.0, provider)
        return normal, wp.dot(vertex.BtoA, normal), lower, vertex.BtoA

    @wp.func
    def solve(
        a: Any, b: Any, rotation: wp.quat, position: wp.vec3, provider: Any, hint: wp.vec3
    ) -> tuple[wp.vec3, float, bool]:
        """Return a depth candidate and whether its minimum is certified."""
        if a.shape_type == int(GeoType.BOX) and b.shape_type == int(GeoType.BOX):
            normal, depth = _box_box_depth(a.scale, b.scale, rotation, position)
            return normal, depth, True
        normal = wp.vec3(1.0, 0.0, 0.0)
        depth = float(1.0e30)
        box_a = a.shape_type == int(GeoType.BOX)
        half = a.scale if box_a else b.scale
        offset = position if box_a else wp.quat_rotate_inv(rotation, -position)
        distances = half - wp.abs(offset)
        # Visit the nearby box face first. This only orders the search;
        # acceptance still requires the inner-body certificate below.
        if distances[0] <= distances[1] and distances[0] <= distances[2]:
            normal = wp.vec3(1.0, 0.0, 0.0)
        elif distances[1] <= distances[2]:
            normal = wp.vec3(0.0, 1.0, 0.0)
        else:
            normal = wp.vec3(0.0, 0.0, 1.0)
        if not box_a:
            normal = wp.quat_rotate(rotation, normal)
        first = support(a, b, normal, rotation, position, 0.0, provider)
        opposite = support(a, b, -normal, rotation, position, 0.0, provider)
        best_vertex = first.BtoA
        recent_vertex = opposite.BtoA
        depth = wp.dot(best_vertex, normal)
        opposite_depth = wp.dot(opposite.BtoA, -normal)
        if opposite_depth < depth:
            depth = opposite_depth
            normal = -normal
            best_vertex = opposite.BtoA
            recent_vertex = first.BtoA
        if depth < 0.0:
            return normal, depth, False
        # A box plus a segment of genuine partner points is an inner
        # Minkowski body. Matching its depth to a support-plane upper bound
        # certifies the global minimum without visiting the mesh's edges.
        point = partner_point(box_a, rotation, position, first)
        far = partner_point(box_a, rotation, position, opposite)
        lower, _direction = _box_triangle_depth(half, point, point, far)
        if lower > 0.0 and depth - lower <= 1.0e-6:
            return normal, depth, True
        # Partner points supporting normals tilted around the caller's
        # candidate (e.g. a portal normal) span the partner's contact feature.
        hint_length = wp.length(hint)
        if hint_length > 0.0:
            hint = hint / hint_length
            tangent = wp.cross(hint, wp.vec3(1.0, 0.0, 0.0))
            if wp.length_sq(tangent) < 0.25:
                tangent = wp.cross(hint, wp.vec3(0.0, 1.0, 0.0))
            tangent = wp.normalize(tangent)
            bitangent = wp.cross(hint, tangent)
            points = wp.mat33()
            for corner in range(3):
                angle = float(corner) * (2.0 * wp.pi / 3.0)
                tilt = wp.cos(angle) * tangent + wp.sin(angle) * bitangent
                vertex = support(a, b, wp.normalize(hint + _CERTIFICATE_TILT * tilt), rotation, position, 0.0, provider)
                points[corner] = partner_point(box_a, rotation, position, vertex)
            hint, upper, lower, hint_vertex = certify(
                a, b, rotation, position, provider, box_a, half, points[0], points[1], points[2]
            )
            if lower > 0.0 and upper - lower <= 1.0e-6:
                return hint, upper, True
            if upper < depth:
                depth = upper
                normal = hint
                recent_vertex = best_vertex
                best_vertex = hint_vertex
        mesh = b if box_a else a
        verified = _has_verified_hull(mesh, provider)
        # Incomplete source triangles cannot certify an interior ball or a
        # minimum depth. Keep the support-plane candidate as an upper bound.
        if not verified:
            return normal, depth, False
        normal, depth = solve_box_mesh(a, b, rotation, position, provider, normal, depth, best_vertex, recent_vertex)
        return normal, depth, True

    return solve
