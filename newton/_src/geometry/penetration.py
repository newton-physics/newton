# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Find box penetration directions with constant-storage separating axes."""

from typing import Any

import warp as wp

from .support_function import (
    _CONVEX_SUPPORT_RESOLUTION_MASK,
    GeoTypeEx,
    _hull_edge_block,
    unpack_mesh_ptr,
)
from .types import GeoType

# Tangent offset [unitless] of the tilted certificate normals, about 0.2 degrees.
# On a curved partner the tilted support points lie below the apex by about
# radius * tilt^2 / 2, which must stay within the certificate tolerance for
# the inner-body bound to meet the support plane; this covers radii up to
# roughly 0.4 m. Flat faces certify at any tilt, and pairs the certificate
# cannot pin down fall back to the exact scan.
_CERTIFICATE_TILT = wp.constant(0.003)


@wp.func
def box_polyhedron_pair(a: Any, b: Any) -> bool:
    """Select box pairs whose portal normal needs a minimum-depth certificate."""
    return (
        a.shape_type == int(GeoType.BOX)
        and (
            b.shape_type == int(GeoType.BOX)
            or b.shape_type == int(GeoType.CONVEX_MESH)
            or b.shape_type == int(GeoTypeEx.TRIANGLE)
        )
    ) or (
        (a.shape_type == int(GeoType.CONVEX_MESH) or a.shape_type == int(GeoTypeEx.TRIANGLE))
        and b.shape_type == int(GeoType.BOX)
    )


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


@wp.func
def _box_hull_frame(
    a: Any, b: Any, rotation: wp.quat, position: wp.vec3
) -> tuple[wp.vec3, wp.uint64, wp.vec3, wp.quat, wp.mat33, wp.vec3, float]:
    """Return the hull scale and mesh, box half extents, and the box axes and offset in the hull frame."""
    mesh_scale = b.scale
    mesh_ptr = b.auxiliary
    half = a.scale
    mesh_rotation = rotation
    to_box = rotation
    # Mesh origin minus box origin in the mesh frame.
    offset = wp.quat_rotate_inv(rotation, position)
    # The common normal points from B to A; flip mesh directions for B.
    side = float(-1.0)
    if a.shape_type == int(GeoType.CONVEX_MESH):
        mesh_scale = a.scale
        mesh_ptr = a.auxiliary
        half = b.scale
        mesh_rotation = wp.quat_identity()
        to_box = wp.quat_inverse(rotation)
        offset = -position
        side = 1.0
    # Rows are the box axes in the mesh frame.
    return mesh_scale, unpack_mesh_ptr(mesh_ptr), half, mesh_rotation, wp.quat_to_matrix(to_box), offset, side


@wp.func
def _box_hull_partial(
    a: Any, b: Any, rotation: wp.quat, position: wp.vec3, provider: Any, edge_block: int, lane: int, lanes: int
) -> tuple[wp.vec3, wp.vec3, wp.vec3, float]:
    """Scan every ``lanes``-th hull vertex and edge for :func:`_box_hull_depth`.

    Returns the extreme vertex projections onto the box axes and the
    shallowest hull-face or edge-pair axis, in the hull frame. Lanes combine
    by componentwise max, min, and the minimum depth.
    """
    mesh_scale, mesh_id, half, _mesh_rotation, box_axes, offset, _side = _box_hull_frame(a, b, rotation, position)
    mesh = wp.mesh_get(mesh_id)
    upper = wp.vec3(-1.0e30)
    lower = wp.vec3(1.0e30)
    for i in range(lane, mesh.points.shape[0], lanes):
        projection = box_axes * wp.cw_mul(mesh.points[i], mesh_scale)
        upper = wp.max(upper, projection)
        lower = wp.min(lower, projection)

    # Mirroring scales reverse the stored face orientation.
    orientation = float(1.0)
    if mesh_scale[0] * mesh_scale[1] * mesh_scale[2] < 0.0:
        orientation = -1.0
    depth = float(1.0e30)
    normal = wp.vec3(1.0, 0.0, 0.0)
    edges = provider.support_neighbors
    for edge in range(lane, edges[edge_block], lanes):
        record = edge_block + 1 + 4 * edge
        p0 = wp.cw_mul(mesh.points[edges[record]], mesh_scale)
        direction = wp.cw_mul(mesh.points[edges[record + 1]], mesh_scale) - p0
        face_0 = orientation * wp.cross(direction, wp.cw_mul(mesh.points[edges[record + 2]], mesh_scale) - p0)
        face_1 = orientation * wp.cross(wp.cw_mul(mesh.points[edges[record + 3]], mesh_scale) - p0, direction)
        for candidate in range(5):
            m = face_0
            if candidate == 1:
                m = face_1
            elif candidate >= 2:
                # A box edge along this axis supports exactly the directions
                # perpendicular to it. The hull edge supports the arc between
                # its face normals, so they share a Minkowski face only if the
                # arc crosses that great circle.
                box_axis = box_axes[candidate - 2]
                d0 = wp.dot(face_0, box_axis)
                d1 = wp.dot(face_1, box_axis)
                m = wp.vec3(0.0)
                if d0 * d1 < 0.0:
                    m = (d1 * face_0 - d0 * face_1) / (d1 - d0)
            length = wp.length(m)
            if length > 0.0:
                m /= length
                candidate_depth = wp.dot(m, p0 + offset) + wp.dot(wp.abs(box_axes * m), half)
                if candidate_depth < depth:
                    depth = candidate_depth
                    normal = m
    return upper, lower, normal, depth


@wp.func
def _box_hull_finish(
    a: Any, b: Any, rotation: wp.quat, position: wp.vec3, upper: wp.vec3, lower: wp.vec3, normal: wp.vec3, depth: float
) -> tuple[wp.vec3, float]:
    """Add the box-face axes to combined :func:`_box_hull_partial` results."""
    _mesh_scale, _mesh_id, half, mesh_rotation, box_axes, offset, side = _box_hull_frame(a, b, rotation, position)
    base = box_axes * offset
    for axis in range(3):
        positive = upper[axis] + base[axis] + half[axis]
        negative = half[axis] - lower[axis] - base[axis]
        if positive < depth:
            depth = positive
            normal = box_axes[axis]
        if negative < depth:
            depth = negative
            normal = -box_axes[axis]
    return side * wp.quat_rotate(mesh_rotation, normal), depth


@wp.func
def _box_hull_depth(
    a: Any, b: Any, rotation: wp.quat, position: wp.vec3, provider: Any, edge_block: int
) -> tuple[wp.vec3, float]:
    """Return the exact minimum-depth normal and depth of a box and a verified hull.

    Separating axes are the box faces, the hull faces, and the box-edge/hull-edge
    cross products whose Gauss-map arcs intersect. Supports along hull faces and
    intersecting edge arcs are their own vertices, so no axis needs a support
    query: the cost is one pass over the vertices plus one over the edges.
    """
    upper, lower, normal, depth = _box_hull_partial(a, b, rotation, position, provider, edge_block, 0, 1)
    return _box_hull_finish(a, b, rotation, position, upper, lower, normal, depth)


def create_solve_box_penetration(support: Any):
    """Find minimum depth for a box against a box, triangle, or convex mesh.

    Box and triangle pairs use the closed-form separating-axis test. Convex
    meshes first try cheap certificates: a box plus a triangle of genuine
    partner points is an inner Minkowski body, so its depth bounds the
    minimum from below, and a support plane bounds it from above; matching
    bounds certify the minimum with a few support queries. Verified hulls
    that remain fall back to the exact Gauss-map test of
    :func:`_box_hull_depth`.
    """

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
    ) -> tuple[wp.vec3, float, float]:
        """Bound the minimum depth from both sides with partner points.

        The box plus a triangle of genuine partner points is an inner
        Minkowski body, so its depth bounds the minimum from below. The
        support plane along its minimizing axis bounds it from above.
        """
        lower, direction = _box_triangle_depth(half, first, second, third)
        # The box-frame direction translates the partner out of the box.
        normal = direction if box_a else -wp.quat_rotate(rotation, direction)
        vertex = support(a, b, normal, rotation, position, 0.0, provider)
        return normal, wp.dot(vertex.BtoA, normal), lower

    @wp.func
    def cheap_certificate(
        a: Any, b: Any, rotation: wp.quat, position: wp.vec3, provider: Any, hint: wp.vec3
    ) -> tuple[bool, wp.vec3, float]:
        """Bound a box-mesh minimum depth with a few support queries.

        Returns whether the bounds met and so certify the minimum, with the
        best support-plane depth candidate and its normal either way.
        """
        box_a = a.shape_type == int(GeoType.BOX)
        normal = wp.vec3(1.0, 0.0, 0.0)
        depth = float(1.0e30)
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
        depth = wp.dot(first.BtoA, normal)
        opposite_depth = wp.dot(opposite.BtoA, -normal)
        if opposite_depth < depth:
            depth = opposite_depth
            normal = -normal
        if depth < 0.0:
            # A negative support depth already proves separation.
            return False, normal, depth
        # A box plus a segment of genuine partner points is an inner
        # Minkowski body. Matching its depth to a support-plane upper bound
        # certifies the global minimum without visiting the mesh's edges.
        point = partner_point(box_a, rotation, position, first)
        far = partner_point(box_a, rotation, position, opposite)
        lower, _direction = _box_triangle_depth(half, point, point, far)
        if lower > 0.0 and depth - lower <= 1.0e-6:
            return True, normal, depth
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
            hint, upper, lower = certify(
                a, b, rotation, position, provider, box_a, half, points[0], points[1], points[2]
            )
            if lower > 0.0 and upper - lower <= 1.0e-6:
                return True, hint, upper
            if upper < depth:
                depth = upper
                normal = hint
        return False, normal, depth

    @wp.func
    def solve(
        a: Any, b: Any, rotation: wp.quat, position: wp.vec3, provider: Any, hint: wp.vec3
    ) -> tuple[wp.vec3, float, bool]:
        """Return a depth candidate and whether its minimum is certified."""
        if a.shape_type == int(GeoTypeEx.TRIANGLE) and b.shape_type == int(GeoType.BOX):
            # Triangle support vertices are (0, scale, auxiliary) in A's frame.
            # Compute the SAT depth in the box frame, then orient A toward B.
            p0 = wp.quat_rotate_inv(rotation, -position)
            p1 = wp.quat_rotate_inv(rotation, a.scale - position)
            p2 = wp.quat_rotate_inv(rotation, a.auxiliary - position)
            depth, normal = _box_triangle_depth(b.scale, p0, p1, p2)
            return -wp.quat_rotate(rotation, normal), depth, True
        if b.shape_type == int(GeoTypeEx.TRIANGLE) and a.shape_type == int(GeoType.BOX):
            p0 = position
            p1 = position + wp.quat_rotate(rotation, b.scale)
            p2 = position + wp.quat_rotate(rotation, b.auxiliary)
            depth, normal = _box_triangle_depth(a.scale, p0, p1, p2)
            return normal, depth, True
        if a.shape_type == int(GeoType.BOX) and b.shape_type == int(GeoType.BOX):
            normal, depth = _box_box_depth(a.scale, b.scale, rotation, position)
            return normal, depth, True
        box_a = a.shape_type == int(GeoType.BOX)
        edge_block = _hull_edge_block(b if box_a else a, provider)
        certified, normal, depth = cheap_certificate(a, b, rotation, position, provider, hint)
        if certified:
            return normal, depth, True
        if edge_block >= 0 and depth >= 0.0:
            # Partner points did not pin the minimum; scan every axis exactly.
            normal, depth = _box_hull_depth(a, b, rotation, position, provider, edge_block)
            return normal, depth, True
        # Without verified hull topology, triangles cannot certify an interior
        # ball or a minimum depth. Keep the support-plane candidate as an upper bound.
        return normal, depth, False

    @wp.func
    def cooperative_certificate(
        a: Any, b: Any, rotation: wp.quat, position: wp.vec3, provider: Any, hint: wp.vec3, lane: int, lanes: int
    ) -> tuple[bool, wp.vec3, float]:
        """Compute a box-hull minimum depth cooperatively across a thread block.

        Every thread of the block must call this with the same pair. The first
        lane tries the partner-point certificate; only pairs it cannot pin down
        scan the hull with all lanes. Returns False for pairs other than a box
        and a verified hull.
        """
        hull = b
        if a.shape_type == int(GeoType.CONVEX_MESH):
            hull = a
        edge_block = int(-1)
        if (a.shape_type == int(GeoType.BOX) and b.shape_type == int(GeoType.CONVEX_MESH)) or (
            a.shape_type == int(GeoType.CONVEX_MESH) and b.shape_type == int(GeoType.BOX)
        ):
            edge_block = _hull_edge_block(hull, provider)
        if edge_block < 0:
            return False, wp.vec3(0.0), 0.0
        certified = int(0)
        normal = wp.vec3(0.0)
        depth = float(0.0)
        # The certificate's few support queries are short walks on accelerated
        # hulls. Without acceleration they scan every vertex serially, which
        # costs more than the cooperative scan below.
        packed = provider.shape_support_data[hull.shape_index][3]
        if lane == 0 and (packed & _CONVEX_SUPPORT_RESOLUTION_MASK) > 0:
            ok, normal, depth = cheap_certificate(a, b, rotation, position, provider, hint)
            if ok or depth < 0.0:
                certified = 1
        certified = wp.tile_max(wp.tile(certified))[0]
        if certified != 0:
            normal = wp.tile(normal, preserve_type=True)[0]
            depth = wp.tile(depth)[0]
            return True, normal, depth
        upper, lower, normal, depth = _box_hull_partial(a, b, rotation, position, provider, edge_block, lane, lanes)
        upper = wp.tile_reduce(wp.max, wp.tile(upper, preserve_type=True))[0]
        lower = wp.tile_reduce(wp.min, wp.tile(lower, preserve_type=True))[0]
        best = wp.tile_argmin(wp.tile(depth))[0]
        normal = wp.tile(normal, preserve_type=True)[best]
        depth = wp.tile_min(wp.tile(depth))[0]
        normal, depth = _box_hull_finish(a, b, rotation, position, upper, lower, normal, depth)
        return True, normal, depth

    solve.applies = box_polyhedron_pair
    solve.cooperative = cooperative_certificate
    return solve
