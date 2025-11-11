# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# This code is based on the multi-contact manifold generation from Jitter Physics 2
# Original: https://github.com/notgiven688/jitterphysics2
# Copyright (c) Thorben Linneweber (MIT License)
# The code has been translated from C# to Python and modified for use in Newton.

"""
Multi-contact manifold generation for collision detection.

This module implements contact manifold generation algorithms for computing
multiple contact points between colliding shapes. It includes polygon clipping,
feature tracking, and contact point selection algorithms.
"""

from typing import Any

import warp as wp

from .kernels import build_orthonormal_basis

# Constants
EPS = 0.00001
# The tilt angle defines how much the search direction gets tilted while searching for
# points on the contact manifold.
TILT_ANGLE_RAD = wp.static(2.0 * wp.pi / 180.0)
SIN_TILT_ANGLE = wp.static(wp.sin(TILT_ANGLE_RAD))
COS_TILT_ANGLE = wp.static(wp.cos(TILT_ANGLE_RAD))

COS_DEEPEST_CONTACT_THRESHOLD_ANGLE = wp.static(wp.cos(0.1 * wp.pi / 180.0))


@wp.func
def should_include_deepest_contact(normal_dot: float) -> bool:
    return normal_dot < COS_DEEPEST_CONTACT_THRESHOLD_ANGLE


@wp.func
def excess_normal_deviation(dir_a: wp.vec3, dir_b: wp.vec3) -> bool:
    """
    Check if the angle between two direction vectors exceeds the tilt angle threshold.

    This is used to detect when contact polygon normals deviate too much from the
    collision normal, indicating that the contact manifold may be unreliable.

    Args:
        dir_a: First direction vector.
        dir_b: Second direction vector.

    Returns:
        True if the angle between the vectors exceeds TILT_ANGLE_RAD (2 degrees).
    """
    dot = wp.abs(wp.dot(dir_a, dir_b))
    return dot < COS_TILT_ANGLE


@wp.func
def signed_area(a: wp.vec2, b: wp.vec2, query_point: wp.vec2) -> float:
    """
    Calculates twice the signed area for the triangle (a, b, query_point).

    The result's sign indicates the triangle's orientation and is a robust way
    to check which side of a line a point is on.

    Args:
        a: The first vertex of the triangle and the start of the line segment.
        b: The second vertex of the triangle and the end of the line segment.
        query_point: The third vertex of the triangle, the point to test against the line a-b.

    Returns:
        The result's sign determines the orientation of the points:
        - Positive (> 0): The points are in a counter-clockwise (CCW) order.
          This means query_point is to the "left" of the directed line from a to b.
        - Negative (< 0): The points are in a clockwise (CW) order.
          This means query_point is to the "right" of the directed line from a to b.
        - Zero (== 0): The points are collinear; query_point lies on the infinite line defined by a and b.
    """
    # It returns twice the signed area of the triangle
    return (b[0] - a[0]) * (query_point[1] - a[1]) - (b[1] - a[1]) * (query_point[0] - a[0])


@wp.func
def ray_plane_intersection(
    ray_origin: wp.vec3, ray_direction: wp.vec3, plane_d: float, plane_normal: wp.vec3
) -> wp.vec3:
    """
    Compute intersection of a ray with a plane.

    The plane is defined by the equation: dot(point, plane_normal) + plane_d = 0
    where plane_d = -dot(point_on_plane, plane_normal).

    Args:
        ray_origin: Starting point of the ray.
        ray_direction: Direction vector of the ray.
        plane_d: Plane distance parameter (negative dot product of any point on plane with normal).
        plane_normal: Normal vector of the plane.

    Returns:
        Intersection point of the ray with the plane.
    """
    denom = wp.dot(ray_direction, plane_normal)
    # Avoid division by zero; if denom is near zero, return origin unchanged
    if wp.abs(denom) < 1.0e-12:
        return ray_origin
    # Plane equation: dot(point, normal) + d = 0
    # Solve for t: dot(ray_origin + t*ray_direction, normal) + d = 0
    # t = -(dot(ray_origin, normal) + d) / dot(ray_direction, normal)
    t = -(wp.dot(ray_origin, plane_normal) + plane_d) / denom
    return ray_origin + ray_direction * t


@wp.struct
class BodyProjector:
    """
    Plane projector for back-projecting contact points onto shape surfaces.

    The plane is defined by the equation: dot(point, normal) + plane_d = 0
    where plane_d = -dot(point_on_plane, normal) for any point on the plane.

    This representation uses a single float instead of storing a full point_on_plane vector,
    saving 8 bytes per projector (2 floats on typical architectures with alignment).
    """

    plane_d: float
    normal: wp.vec3


@wp.func
def make_body_projector_from_polygon(
    poly: wp.array(dtype=wp.vec3), poly_count: int, anchor_point: wp.vec3
) -> BodyProjector:
    """
    Create a body projector (plane definition) from a polygon.

    This function computes a best-fit plane for back-projecting contact points
    onto the original shape surfaces. It uses the triangle with the largest area
    to compute a robust normal vector, avoiding numerical issues with collinear points.

    Args:
        poly: Array of polygon vertices in world space.
        poly_count: Number of vertices in the polygon.
        anchor_point: Reference point on the plane (typically the contact anchor point).

    Returns:
        BodyProjector with the plane normal and plane_d parameter.
    """
    proj = BodyProjector()
    # Find the triangle with the largest area for numerical stability
    # This avoids issues with nearly collinear points
    best_normal = wp.vec3(0.0, 0.0, 0.0)
    max_area_sq = float(0.0)

    for i in range(1, poly_count - 1):
        # Compute cross product for triangle (poly[0], poly[i], poly[i+1])
        edge1 = poly[i] - poly[0]
        edge2 = poly[i + 1] - poly[0]
        cross = wp.cross(edge1, edge2)
        area_sq = wp.dot(cross, cross)

        if area_sq > max_area_sq:
            max_area_sq = area_sq
            best_normal = cross

    # Normalize, avoid zero
    len_n = wp.sqrt(wp.max(1.0e-12, max_area_sq))
    proj.normal = best_normal / len_n
    # Compute plane_d from the plane equation: dot(point, normal) + plane_d = 0
    proj.plane_d = -wp.dot(anchor_point, proj.normal)
    return proj


@wp.func
def compute_line_segment_projector_normal(
    segment_dir: wp.vec3,
    reference_normal: wp.vec3,
) -> wp.vec3:
    """
    Compute a normal for a line segment projector that is perpendicular to the segment
    and lies in the plane defined by the segment and the reference normal.

    Args:
        segment_dir: Direction vector of the line segment.
        reference_normal: Normal from the other body to use as reference.

    Returns:
        Normalized normal vector for the line segment projector.
    """
    right = wp.cross(segment_dir, reference_normal)
    normal = wp.cross(right, segment_dir)
    length = wp.length(normal)
    return normal / length if length > 1.0e-12 else reference_normal


@wp.func
def create_body_projectors(
    poly_a: wp.array(dtype=wp.vec3),
    poly_count_a: int,
    anchor_point_a: wp.vec3,
    poly_b: wp.array(dtype=wp.vec3),
    poly_count_b: int,
    anchor_point_b: wp.vec3,
    contact_normal: wp.vec3,
) -> tuple[BodyProjector, BodyProjector]:
    projector_a = BodyProjector()
    projector_b = BodyProjector()

    if poly_count_a < 3 and poly_count_b < 3:
        # Both are line segments - compute normals using contact_normal as reference
        dir_a = poly_a[1] - poly_a[0]
        dir_b = poly_b[1] - poly_b[0]

        point_on_plane_a = 0.5 * (poly_a[0] + poly_a[1])
        projector_a.normal = compute_line_segment_projector_normal(dir_a, contact_normal)
        projector_a.plane_d = -wp.dot(point_on_plane_a, projector_a.normal)

        point_on_plane_b = 0.5 * (poly_b[0] + poly_b[1])
        projector_b.normal = compute_line_segment_projector_normal(dir_b, contact_normal)
        projector_b.plane_d = -wp.dot(point_on_plane_b, projector_b.normal)

        return projector_a, projector_b

    if poly_count_a >= 3:
        projector_a = make_body_projector_from_polygon(poly_a, poly_count_a, anchor_point_a)
    if poly_count_b >= 3:
        projector_b = make_body_projector_from_polygon(poly_b, poly_count_b, anchor_point_b)

    if poly_count_a < 3:
        dir = poly_a[1] - poly_a[0]
        point_on_plane_a = 0.5 * (poly_a[0] + poly_a[1])
        projector_a.normal = compute_line_segment_projector_normal(dir, projector_b.normal)
        projector_a.plane_d = -wp.dot(point_on_plane_a, projector_a.normal)

    if poly_count_b < 3:
        dir = poly_b[1] - poly_b[0]
        point_on_plane_b = 0.5 * (poly_b[0] + poly_b[1])
        projector_b.normal = compute_line_segment_projector_normal(dir, projector_a.normal)
        projector_b.plane_d = -wp.dot(point_on_plane_b, projector_b.normal)

    return projector_a, projector_b


@wp.func
def body_projector_project(
    proj: BodyProjector,
    input: wp.vec3,
    contact_normal: wp.vec3,
) -> wp.vec3:
    """
    Project a point back onto the original shape surface using a plane projector.

    This function casts a ray from the input point along the contact normal and
    finds where it intersects the projector's plane.

    Args:
        proj: Body projector defining the projection plane.
        input: Point to project (typically in contact plane space).
        contact_normal: Direction to cast the ray (typically the collision normal).

    Returns:
        Projected point on the shape's surface in world space.
    """
    # Only plane projection is supported
    return ray_plane_intersection(input, contact_normal, proj.plane_d, proj.normal)


@wp.func
def intersection_point(trim_seg_start: wp.vec3, trim_seg_end: wp.vec3, a: wp.vec3, b: wp.vec3) -> wp.vec3:
    """
    Calculate the intersection point between a line segment and a polygon edge.

    It is known that a and b lie on different sides of the trim segment.

    Args:
        trim_seg_start: Start point of the trimming segment.
        trim_seg_end: End point of the trimming segment.
        a: First point of the polygon edge.
        b: Second point of the polygon edge.

    Returns:
        The intersection point as a vec3.
    """
    # Get 2D projections
    trim_start_xy = wp.vec2(trim_seg_start[0], trim_seg_start[1])
    trim_end_xy = wp.vec2(trim_seg_end[0], trim_seg_end[1])
    a_xy = wp.vec2(a[0], a[1])
    b_xy = wp.vec2(b[0], b[1])

    dist_a = wp.abs(signed_area(trim_start_xy, trim_end_xy, a_xy))
    dist_b = wp.abs(signed_area(trim_start_xy, trim_end_xy, b_xy))
    interp_ab = dist_a / (dist_a + dist_b)

    # Interpolate between a and b
    interpolated_ab = (1.0 - interp_ab) * a + interp_ab * b

    # Calculate projection along trim segment (used to interpolate trim-segment Z into W)
    delta = trim_end_xy - trim_start_xy
    delta = wp.normalize(delta)
    # interpolated point computed above; offset not required in vec3 version
    # Note: projection parameter not needed in vec3 version
    return interpolated_ab


@wp.func
def insert_vec3(arr: wp.array(dtype=wp.vec3), arr_count: int, index: int, element: wp.vec3):
    """
    Insert an element into an array at the specified index, shifting elements to the right.

    Args:
        arr: Array to insert into.
        arr_count: Current number of elements in the array.
        index: Index at which to insert the element.
        element: Element to insert.
    """
    i = arr_count
    while i > index:
        arr[i] = arr[i - 1]
        i -= 1
    arr[index] = element


@wp.func
def insert_byte(arr: wp.array(dtype=wp.uint8), arr_count: int, index: int, element: wp.uint8):
    """
    Insert a byte element into an array at the specified index, shifting elements to the right.

    Args:
        arr: Array to insert into.
        arr_count: Current number of elements in the array.
        index: Index at which to insert the element.
        element: Element to insert.
    """
    i = arr_count
    while i > index:
        arr[i] = arr[i - 1]
        i -= 1
    arr[index] = element


@wp.func
def trim_in_place(
    trim_seg_start: wp.vec3,
    trim_seg_end: wp.vec3,
    trim_seg_id: wp.uint8,
    loop: wp.array(dtype=wp.vec3),
    loop_seg_ids: wp.array(dtype=wp.uint8),
    loop_count: int,
) -> int:
    """
    Trim a polygon in place using a line segment.

    The vec3 format is as follows:
    - X, Y: 2D coordinates projected onto the contact plane
    - Z: The offset out of the plane for the polygon called loop

    The trim segment format is as follows:
    - X, Y: 2D coordinates projected onto the contact plane
    - Z: The offset out of the plane for the trim segment

    loopSegIds[0] refers to the segment from loop[0] to loop[1], etc.

    Args:
        trim_seg_start: Start point of the trimming segment.
        trim_seg_end: End point of the trimming segment.
        trim_seg_id: ID of the trimming segment.
        loop: Array of loop vertices.
        loop_seg_ids: Array of segment IDs for the loop.
        loop_count: Number of vertices in the loop.

    Returns:
        New number of vertices in the trimmed loop.
    """
    if loop_count < 3:
        return loop_count

    intersection_a = wp.vec3(0.0, 0.0, 0.0)
    change_a = int(-1)
    change_a_seg_id = wp.uint8(255)
    intersection_b = wp.vec3(0.0, 0.0, 0.0)
    change_b = int(-1)
    change_b_seg_id = wp.uint8(255)

    keep = bool(False)

    # Get 2D projections for the trim segment
    trim_start_xy = wp.vec2(trim_seg_start[0], trim_seg_start[1])
    trim_end_xy = wp.vec2(trim_seg_end[0], trim_seg_end[1])

    # Check first vertex
    loop0_xy = wp.vec2(loop[0][0], loop[0][1])
    prev_outside = bool(signed_area(trim_start_xy, trim_end_xy, loop0_xy) <= 0.0)

    for i in range(loop_count):
        next_idx = (i + 1) % loop_count
        loop_next_xy = wp.vec2(loop[next_idx][0], loop[next_idx][1])
        outside = signed_area(trim_start_xy, trim_end_xy, loop_next_xy) <= 0.0

        if outside != prev_outside:
            intersection = intersection_point(trim_seg_start, trim_seg_end, loop[i], loop[next_idx])
            if change_a < 0:
                change_a = i
                change_a_seg_id = loop_seg_ids[i]
                keep = not prev_outside
                intersection_a = intersection
            else:
                change_b = i
                change_b_seg_id = loop_seg_ids[i]
                intersection_b = intersection

        prev_outside = outside

    if change_a >= 0 and change_b >= 0:
        loop_indexer = int(-1)
        new_loop_count = int(loop_count)

        i = int(0)
        while i < loop_count:
            # If the current vertex is on the side to be kept, copy it and its segment ID.
            if keep:
                loop_indexer += 1
                loop[loop_indexer] = loop[i]
                loop_seg_ids[loop_indexer] = loop_seg_ids[i]

            # If the current edge is one of the two that intersects the trim line,
            # add the intersection point to the new polygon.
            if i == change_a or i == change_b:
                pt = intersection_a if i == change_a else intersection_b
                original_seg_id = change_a_seg_id if i == change_a else change_b_seg_id

                # Determine the correct ID for the segment starting at the new intersection point.
                # If we are currently keeping vertices (`keep` is true), it means we're transitioning
                # to a discarded section. The new segment connects the two intersection points,
                # so its ID is `trim_seg_id`.
                # If we are currently discarding vertices (`keep` is false), it means we're
                # transitioning to a kept section. The new segment is a continuation of the
                # original edge that was cut, so it keeps its `original_seg_id`.
                new_seg_id = trim_seg_id if keep else original_seg_id

                # This block handles a special case for inserting the new point.
                if loop_indexer == i and not keep:
                    loop_indexer += 1
                    insert_vec3(loop, new_loop_count, loop_indexer, pt)
                    insert_byte(loop_seg_ids, new_loop_count, loop_indexer, new_seg_id)

                    new_loop_count += 1
                    # Advance i and adjust change_b to account for insertion
                    i += 1
                    change_b += 1
                    # Keep iteration bound consistent with source mutation
                    loop_count += 1
                else:
                    loop_indexer += 1
                    loop[loop_indexer] = pt
                    loop_seg_ids[loop_indexer] = new_seg_id

                # Flip the keep flag after processing an intersection.
                keep = not keep

            i += 1

        new_loop_count = loop_indexer + 1
    elif prev_outside:
        # If there was no intersection, all points are on the same side.
        # If all are outside, clear the loop.
        new_loop_count = 0
    else:
        new_loop_count = loop_count

    return new_loop_count


@wp.func
def trim_all_in_place(
    trim_poly: wp.array(dtype=wp.vec3),
    trim_poly_count: int,
    loop: wp.array(dtype=wp.vec3),
    loop_segments: wp.array(dtype=wp.uint8),
    loop_count: int,
) -> int:
    """
    Trim a polygon using all edges of another polygon.

    Both polygons (trim_poly and loop) are in the contact frame space and they are both convex.

    Args:
        trim_poly: Array of vertices defining the trimming polygon.
        trim_poly_count: Number of vertices in the trimming polygon.
        loop: Array of vertices in the loop to be trimmed.
        loop_segments: Array of segment IDs for the loop.
        loop_count: Number of vertices in the loop.

    Returns:
        New number of vertices in the trimmed loop.
    """

    if trim_poly_count <= 1:
        return wp.min(1, loop_count)  # There is no trim polygon

    move_distance = float(1e-5)

    if trim_poly_count == 2:
        # Convert line segment to thin rectangle
        # Line segment: trim_poly[0] to trim_poly[1]
        p0 = trim_poly[0]
        p1 = trim_poly[1]

        # Direction vector (only x, y matter for 2D operations)
        dir_x = p1[0] - p0[0]
        dir_y = p1[1] - p0[1]
        dir_len = wp.sqrt(dir_x * dir_x + dir_y * dir_y)

        if dir_len > 1e-10:
            # Perpendicular vector (rotate 90 degrees: (x,y) -> (-y,x))
            perp_x = -dir_y / dir_len
            perp_y = dir_x / dir_len

            # Create 4 corners of rectangle (counterclockwise order)
            # Start from p0, go along one side, then back along the other
            offset_x = perp_x * move_distance
            offset_y = perp_y * move_distance

            trim_poly[0] = wp.vec3(p0[0] - offset_x, p0[1] - offset_y, p0[2])
            trim_poly[1] = wp.vec3(p1[0] - offset_x, p1[1] - offset_y, p1[2])
            trim_poly[2] = wp.vec3(p1[0] + offset_x, p1[1] + offset_y, p1[2])
            trim_poly[3] = wp.vec3(p0[0] + offset_x, p0[1] + offset_y, p0[2])
            trim_poly_count = 4
        else:
            return wp.min(1, loop_count)

    if loop_count == 2:
        # Convert line segment to thin rectangle
        p0 = loop[0]
        p1 = loop[1]
        seg0 = loop_segments[0]
        seg1 = loop_segments[1]

        # Direction vector (only x, y matter for 2D operations)
        dir_x = p1[0] - p0[0]
        dir_y = p1[1] - p0[1]
        dir_len = wp.sqrt(dir_x * dir_x + dir_y * dir_y)

        if dir_len > 1e-10:
            # Perpendicular vector (rotate 90 degrees: (x,y) -> (-y,x))
            perp_x = -dir_y / dir_len
            perp_y = dir_x / dir_len

            # Create 4 corners of rectangle (counterclockwise order)
            offset_x = perp_x * move_distance
            offset_y = perp_y * move_distance

            loop[0] = wp.vec3(p0[0] - offset_x, p0[1] - offset_y, p0[2])
            loop[1] = wp.vec3(p1[0] - offset_x, p1[1] - offset_y, p1[2])
            loop[2] = wp.vec3(p1[0] + offset_x, p1[1] + offset_y, p1[2])
            loop[3] = wp.vec3(p0[0] + offset_x, p0[1] + offset_y, p0[2])

            # Segment IDs: edges 0-1 and 1-2 inherit from original edge 0-1
            # edges 2-3 and 3-0 form the "caps"
            loop_segments[0] = seg0
            loop_segments[1] = seg1
            loop_segments[2] = seg1
            loop_segments[3] = seg0

            loop_count = 4
        else:
            return wp.min(1, loop_count)

    current_loop_count = loop_count

    for i in range(trim_poly_count):
        # For each trim segment, we will call the efficient trim function.
        trim_seg_start = trim_poly[i]
        trim_seg_end = trim_poly[(i + 1) % trim_poly_count]
        # Perform the in-place trimming for this segment.
        current_loop_count = trim_in_place(
            trim_seg_start, trim_seg_end, wp.uint8(i), loop, loop_segments, current_loop_count
        )

    return current_loop_count


@wp.func
def approx_max_quadrilateral_area_with_calipers(hull: wp.array(dtype=wp.vec3), hull_count: int) -> wp.vec4i:
    """
    Finds an approximate maximum area quadrilateral inside a convex hull in O(n) time
    using the Rotating Calipers algorithm to find the hull's diameter.

    Args:
        hull: Array of hull vertices.
        hull_count: Number of vertices in the hull.

    Returns:
        vec4i containing (p1, p2, p3, p4) where p1, p2, p3, p4 are the indices
        of the quadrilateral vertices that form the maximum area quadrilateral.
    """
    n = hull_count

    # --- Step 1: Find the hull's diameter using Rotating Calipers in O(n) ---
    p1 = int(0)
    p3 = int(1)
    hp1 = hull[p1]
    hp3 = hull[p3]
    diff = wp.vec3(hp1[0] - hp3[0], hp1[1] - hp3[1], hp1[2] - hp3[2])
    max_dist_sq = diff[0] * diff[0] + diff[1] * diff[1] + diff[2] * diff[2]

    # Relative epsilon for tie-breaking: only update if new value is at least (1 + epsilon) times better
    # This is scale-invariant and avoids catastrophic cancellation in floating-point comparisons
    # Important for objects with circular geometry to ensure consistent point selection
    tie_epsilon_rel = 1.0e-3

    # Start with point j opposite point i=0
    j = int(1)
    for i in range(n):
        # For the current point i, find its antipodal point j by advancing j
        # while the area of the triangle formed by the edge (i, i+1) and point j increases.
        # This is equivalent to finding the point j furthest from the edge (i, i+1).
        hull_i_xy = wp.vec2(hull[i][0], hull[i][1])
        hull_i_plus_1_xy = wp.vec2(hull[(i + 1) % n][0], hull[(i + 1) % n][1])

        while True:
            hull_j_xy = wp.vec2(hull[j][0], hull[j][1])
            hull_j_plus_1_xy = wp.vec2(hull[(j + 1) % n][0], hull[(j + 1) % n][1])

            area_j_plus_1 = signed_area(hull_i_xy, hull_i_plus_1_xy, hull_j_plus_1_xy)
            area_j = signed_area(hull_i_xy, hull_i_plus_1_xy, hull_j_xy)

            if area_j_plus_1 > area_j:
                j = (j + 1) % n
            else:
                break

        # Now, (i, j) is an antipodal pair. Check its distance (XYZ)
        hi = hull[i]
        hj = hull[j]
        d1 = wp.vec3(hi[0] - hj[0], hi[1] - hj[1], hi[2] - hj[2])
        dist_sq_1 = d1[0] * d1[0] + d1[1] * d1[1] + d1[2] * d1[2]
        # Use relative tie-breaking: only update if new distance is meaningfully larger
        if dist_sq_1 > max_dist_sq * (1.0 + tie_epsilon_rel):
            max_dist_sq = dist_sq_1
            p1 = i
            p3 = j

        # The next point, (i+1, j), is also an antipodal pair. Check its distance too (XYZ)
        hip1 = hull[(i + 1) % n]
        d2 = wp.vec3(hip1[0] - hj[0], hip1[1] - hj[1], hip1[2] - hj[2])
        dist_sq_2 = d2[0] * d2[0] + d2[1] * d2[1] + d2[2] * d2[2]
        # Use relative tie-breaking: only update if new distance is meaningfully larger
        if dist_sq_2 > max_dist_sq * (1.0 + tie_epsilon_rel):
            max_dist_sq = dist_sq_2
            p1 = (i + 1) % n
            p3 = j

    # --- Step 2: Find points p2 and p4 furthest from the diameter (p1, p3) ---
    p2 = int(0)
    p4 = int(0)
    max_area_1 = float(0.0)
    max_area_2 = float(0.0)

    hull_p1_xy = wp.vec2(hull[p1][0], hull[p1][1])
    hull_p3_xy = wp.vec2(hull[p3][0], hull[p3][1])

    for i in range(n):
        # Use the signed area to determine which side of the line the point is on.
        hull_i_xy = wp.vec2(hull[i][0], hull[i][1])
        area = signed_area(hull_p1_xy, hull_p3_xy, hull_i_xy)

        # Use relative tie-breaking: only update if new area is meaningfully larger
        if area > max_area_1 * (1.0 + tie_epsilon_rel):
            max_area_1 = area
            p2 = i
        elif -area > max_area_2 * (1.0 + tie_epsilon_rel):  # Check the other side
            max_area_2 = -area
            p4 = i

    return wp.vec4i(p1, p2, p3, p4)


@wp.func
def remove_zero_length_edges(
    loop: wp.array(dtype=wp.vec3), loop_seg_ids: wp.array(dtype=wp.uint8), loop_count: int, eps: float
) -> int:
    """
    Remove zero-length edges from a polygon loop.

    Args:
        loop: Array of loop vertices.
        loop_seg_ids: Array of segment IDs for the loop.
        loop_count: Number of vertices in the loop.
        eps: Epsilon threshold for considering edges as zero-length.

    Returns:
        New number of vertices in the cleaned loop.
    """
    # A loop must have at least 2 points to be valid per your requirement.
    if loop_count < 2:
        return 0

    # 'write_idx' is the index for the new, compacted loop.
    # It always points to the last valid point found so far.
    write_idx = int(0)

    # Iterate through the original loop, starting from the second point.
    # 'read_idx' is the index of the point we are currently considering.
    for read_idx in range(1, loop_count):
        # Check if the current point is distinct from the last point we kept.
        loop_read_xy = wp.vec2(loop[read_idx][0], loop[read_idx][1])
        loop_write_xy = wp.vec2(loop[write_idx][0], loop[write_idx][1])
        diff = loop_read_xy - loop_write_xy

        if wp.length_sq(diff) > eps:
            # It's a distinct point, so we advance the write index and keep it.
            write_idx += 1
            loop[write_idx] = loop[read_idx]
            loop_seg_ids[write_idx - 1] = loop_seg_ids[read_idx - 1]

    loop_seg_ids[write_idx] = loop_seg_ids[loop_count - 1]

    # At this point, the loop is clean but might not be closed properly.
    # The number of points in our cleaned chain is 'write_idx + 1'.

    # Handle the loop closure by checking if the last point is the same as the first.
    if write_idx > 0:
        loop_write_xy = wp.vec2(loop[write_idx][0], loop[write_idx][1])
        loop_0_xy = wp.vec2(loop[0][0], loop[0][1])
        diff = loop_write_xy - loop_0_xy

        if wp.length_sq(diff) < eps:
            # The last point is a duplicate of the first; we need to remove it.
            new_loop_count = write_idx
        else:
            # The last point is not a duplicate, so we keep all 'write_idx + 1' points.
            new_loop_count = write_idx + 1
    else:
        new_loop_count = write_idx + 1

    # Final check based on your requirement.
    # If simplification resulted in fewer than 2 points, it's a degenerate point.
    if new_loop_count < 2:
        new_loop_count = 0

    return new_loop_count


@wp.func
def edge_key(per_vertex_features: wp.types.vector(6, wp.uint8), count: int, edge_id: int) -> wp.uint32:
    """
    Creates a packed edge key from two consecutive feature IDs.
    Used to create compact identifiers for edges defined by vertex pairs.

    Args:
        per_vertex_features: Array of feature IDs.
        count: Number of features in the array.
        edge_id: Index of the first vertex of the edge.

    Returns:
        16-bit packed edge key: (first_feature << 8) | second_feature.
    """
    # An edge always goes from one vertex to the next, wrapping around at the end.
    first = per_vertex_features[edge_id]
    second = per_vertex_features[(edge_id + 1) % count]
    return wp.uint32(wp.uint32(first) << wp.uint32(8)) | wp.uint32(second)


@wp.func
def feature_id(
    loop_seg_ids: wp.array(dtype=wp.uint8),
    loop_id: int,
    loop_count: int,
    features_a: wp.types.vector(6, wp.uint8),
    features_b: wp.types.vector(6, wp.uint8),
    m_a_count: int,
    m_b_count: int,
) -> wp.uint32:
    """
    Determines the feature identifier for a vertex in the clipped contact polygon.
    This function assigns feature IDs that encode which geometric features from the original
    collision shapes (vertices, edges, or edge-edge intersections) each contact point represents.

    ENCODING SCHEME:
    - Original trim poly vertex: 8-bit feature ID from features_a
    - Original loop poly vertex: 16-bit (features_b[vertex] << 8)
    - Edge intersections: 32-bit ((edge1_key << 16) | edge2_key)
    - Shape intersections: 32-bit ((shapeA_edge << 16) | shapeB_edge)

    SEGMENT ID CONVENTION:
    - IDs 0-5: segments from trim polygon (shape A)
    - IDs 6+: segments from loop polygon (shape B, with offset)

    Args:
        loop_seg_ids: Array of segment IDs for the current clipped polygon.
        loop_id: Index of the vertex to compute feature ID for.
        loop_count: Total number of vertices in the polygon.
        features_a: Original feature IDs from trim polygon (shape A).
        features_b: Original feature IDs from loop polygon (shape B).
        m_a_count: Number of vertices in original trim polygon.
        m_b_count: Number of vertices in original loop polygon.

    Returns:
        A feature ID encoding the geometric origin of this contact point.
    """
    feature = wp.uint32(0)

    incoming = loop_seg_ids[(loop_id - 1 + loop_count) % loop_count]
    outgoing = loop_seg_ids[loop_id]
    incoming_belongs_to_trim_poly = incoming < 6
    outgoing_belongs_to_trim_poly = outgoing < 6

    if incoming_belongs_to_trim_poly != outgoing_belongs_to_trim_poly:
        # This must be an intersection point
        if incoming_belongs_to_trim_poly:
            x = edge_key(features_a, m_a_count, int(incoming))
        else:
            x = edge_key(features_b, m_b_count, int(incoming) - 6)

        if outgoing_belongs_to_trim_poly:
            y = edge_key(features_a, m_a_count, int(outgoing))
        else:
            y = edge_key(features_b, m_b_count, int(outgoing) - 6)

        feature = (x << wp.uint32(16)) | y
    else:
        if incoming_belongs_to_trim_poly:
            next_seg = (int(incoming) + 1) % m_a_count
            is_original_poly_point = next_seg == int(outgoing)
            if is_original_poly_point:
                feature = wp.uint32(features_a[int(outgoing)])
            else:
                # Should not happen because input poly A would have self intersections
                x = edge_key(features_a, m_a_count, int(incoming))
                y = edge_key(features_a, m_a_count, int(outgoing))
                feature = (x << wp.uint32(16)) | y
        else:
            next_seg = (int(incoming) - 6 + 1) % m_b_count + 6
            is_original_poly_point = next_seg == int(outgoing)
            if is_original_poly_point:
                # Shifted such that not the same id can get generated as produced by features_a
                feature = wp.uint32(features_b[int(outgoing) - 6]) << wp.uint32(8)
            else:
                # Should not happen because input poly B would have self intersections
                x = edge_key(features_b, m_b_count, int(incoming) - 6)
                y = edge_key(features_b, m_b_count, int(outgoing) - 6)
                feature = (x << wp.uint32(16)) | y

    return feature


@wp.func
def add_avoid_duplicates_vec3(
    arr: wp.array(dtype=wp.vec3), arr_count: int, vec: wp.vec3, eps: float
) -> tuple[int, bool]:
    """
    Add a vector to an array, avoiding duplicates.

    Args:
        arr: Array to add to.
        arr_count: Current number of elements in the array.
        vec: Vector to add.
        eps: Epsilon threshold for duplicate detection.

    Returns:
        Tuple of (new_count, was_added) where was_added is True if point was added
    """
    # Check for duplicates. If the new vertex 'vec' is too close to the first or last existing vertex, ignore it.
    # This is a simple reduction step to avoid redundant points.
    if arr_count > 0:
        if wp.length_sq(arr[0] - vec) < eps:
            return arr_count, False

    if arr_count > 1:
        if wp.length_sq(arr[arr_count - 1] - vec) < eps:
            return arr_count, False

    arr[arr_count] = vec
    return arr_count + 1, True


vec6_uint8 = wp.types.vector(6, wp.uint8)


@wp.func
def extract_4_point_contact_manifolds(
    m_a: wp.array(dtype=wp.vec3),
    features_a: wp.types.vector(6, wp.uint8),
    m_a_count: int,
    m_b: wp.array(dtype=wp.vec3),
    features_b: wp.types.vector(6, wp.uint8),
    m_b_count: int,
    normal: wp.vec3,
    cross_vector_1: wp.vec3,
    cross_vector_2: wp.vec3,
    anchor_point_a: wp.vec3,
    anchor_point_b: wp.vec3,
    result_features: wp.array(dtype=wp.uint32),
) -> tuple[int, float]:
    """
    Extract up to 4 contact points from two convex contact polygons using polygon clipping (before optional deepest point addition).

        This function performs the core manifold generation algorithm:
        1. Validates input polygons and checks for normal deviation from collision normal
        2. Projects both polygons into 2D contact plane space (XY = tangent plane, Z = depth)
        3. Clips polygon B against all edges of polygon A (Sutherland-Hodgman style clipping)
        4. Removes zero-length edges from the clipped result
        5. If more than 4 points remain, selects the best 4 using rotating calipers algorithm
        6. Projects all contact points back onto the original shape surfaces in world space
        7. Computes and tracks feature IDs for contact persistence
        Note: Returns up to 4 contacts; the 5th can be added later via should_include_deepest_contact check

    Args:
        m_a: Contact polygon vertices for shape A (input: world space, up to 6 points).
             Modified in place and used as output buffer.
        features_a: Feature IDs for each vertex of polygon A.
        m_a_count: Number of vertices in polygon A.
        m_b: Contact polygon vertices for shape B (input: world space, up to 6 points).
             Modified in place and used as output buffer. Must have space for 12 points.
        features_b: Feature IDs for each vertex of polygon B.
        m_b_count: Number of vertices in polygon B.
        normal: Collision normal vector pointing from A to B.
        cross_vector_1: First tangent vector (forms contact plane basis with cross_vector_2).
        cross_vector_2: Second tangent vector (forms contact plane basis with cross_vector_1).
        anchor_point_a: Anchor contact point on shape A (from GJK/MPR).
        anchor_point_b: Anchor contact point on shape B (from GJK/MPR).
        result_features: Output array for feature IDs of final contact points.

    Returns:
        Number of valid contact points generated (0-4). Contact points are stored in
        m_a (shape A side) and m_b (shape B side) arrays, with feature IDs in result_features.
    """
    # Early-out for simple cases: if both have <=2 or either is empty, return single anchor pair
    # if True or m_a_count < 3 or m_b_count < 3:
    if m_a_count < 2 or m_b_count < 2:  # or (m_a_count < 3 and m_b_count < 3):
        m_a[0] = anchor_point_a
        m_b[0] = anchor_point_b
        result_features[0] = wp.uint32(0)
        return 1, 1.0

    # Projectors for back-projection onto the shape surfaces
    projector_a, projector_b = create_body_projectors(
        m_a, m_a_count, anchor_point_a, m_b, m_b_count, anchor_point_b, normal
    )

    if excess_normal_deviation(normal, projector_a.normal) or excess_normal_deviation(normal, projector_b.normal):
        m_a[0] = anchor_point_a
        m_b[0] = anchor_point_b
        result_features[0] = wp.uint32(0)
        return 1, 1.0

    normal_dot = wp.abs(wp.dot(projector_a.normal, projector_b.normal))

    # The trim poly (poly A) should be the polygon with the most points
    # This should ensure that zero area loops with only two points get trimmed correctly (they are considered valid)
    center = 0.5 * (anchor_point_a + anchor_point_b)

    # Transform into contact plane space
    for i in range(m_a_count):
        projected = m_a[i] - center
        m_a[i] = wp.vec3(
            wp.dot(cross_vector_1, projected),
            wp.dot(cross_vector_2, projected),
            wp.dot(normal, projected),
        )

    loop_seg_ids = wp.zeros(shape=(12,), dtype=wp.uint8)  # stackalloc byte[maxPoints];

    for i in range(m_b_count):
        projected = m_b[i] - center
        m_b[i] = wp.vec3(
            wp.dot(cross_vector_1, projected),
            wp.dot(cross_vector_2, projected),
            wp.dot(normal, projected),
        )
        loop_seg_ids[i] = wp.uint8(i + 6)

    loop_count = trim_all_in_place(m_a, m_a_count, m_b, loop_seg_ids, m_b_count)

    loop_count = remove_zero_length_edges(m_b, loop_seg_ids, loop_count, EPS)

    if loop_count > 4:
        result = approx_max_quadrilateral_area_with_calipers(m_b, loop_count)
        ia = int(result[0])
        ib = int(result[1])
        ic = int(result[2])
        id = int(result[3])

        result_features[0] = feature_id(loop_seg_ids, ia, loop_count, features_a, features_b, m_a_count, m_b_count)
        result_features[1] = feature_id(loop_seg_ids, ib, loop_count, features_a, features_b, m_a_count, m_b_count)
        result_features[2] = feature_id(loop_seg_ids, ic, loop_count, features_a, features_b, m_a_count, m_b_count)
        result_features[3] = feature_id(loop_seg_ids, id, loop_count, features_a, features_b, m_a_count, m_b_count)

        # Transform back to world space using projectors
        a = m_b[ia]
        a_world = a[0] * cross_vector_1 + a[1] * cross_vector_2 + center
        b = m_b[ib]
        b_world = b[0] * cross_vector_1 + b[1] * cross_vector_2 + center
        c = m_b[ic]
        c_world = c[0] * cross_vector_1 + c[1] * cross_vector_2 + center
        d = m_b[id]
        d_world = d[0] * cross_vector_1 + d[1] * cross_vector_2 + center

        # normal vector points from A to B
        m_a[0] = body_projector_project(projector_a, a_world, normal)
        m_a[1] = body_projector_project(projector_a, b_world, normal)
        m_a[2] = body_projector_project(projector_a, c_world, normal)
        m_a[3] = body_projector_project(projector_a, d_world, normal)

        m_b[0] = body_projector_project(projector_b, a_world, normal)
        m_b[1] = body_projector_project(projector_b, b_world, normal)
        m_b[2] = body_projector_project(projector_b, c_world, normal)
        m_b[3] = body_projector_project(projector_b, d_world, normal)

        loop_count = 4
    else:
        if loop_count <= 1:
            # Degenerate; return single anchor pair
            m_a[0] = anchor_point_a
            m_b[0] = anchor_point_b
            if result_features.shape[0] > 0:
                result_features[0] = wp.uint32(0)
            loop_count = 1
        else:
            # Transform back to world space using projectors
            for i in range(loop_count):
                l = m_b[i]
                feat = feature_id(loop_seg_ids, i, loop_count, features_a, features_b, m_a_count, m_b_count)
                world = l[0] * cross_vector_1 + l[1] * cross_vector_2 + center
                m_a[i] = body_projector_project(projector_a, world, normal)
                m_b[i] = body_projector_project(projector_b, world, normal)
                result_features[i] = feat

    return loop_count, normal_dot


def create_build_manifold(support_func: Any):
    """
    Factory function to create manifold generation functions with a specific support mapping function.

    This factory creates two related functions for multi-contact manifold generation:
    - build_manifold_core: The core implementation that uses preallocated buffers
    - build_manifold: The main entry point that handles buffer allocation and result extraction

    Args:
        support_func: Support mapping function for shapes that takes
                     (geometry, direction, data_provider) and returns (point, feature_id)

    Returns:
        build_manifold function that generates up to 5 contact points between two shapes
        using perturbed support mapping and polygon clipping.
    """

    # Main contact manifold generation function
    @wp.func
    def build_manifold_core(
        geom_a: Any,
        geom_b: Any,
        quaternion_a: wp.quat,
        quaternion_b: wp.quat,
        position_a: wp.vec3,
        position_b: wp.vec3,
        p_a: wp.vec3,
        p_b: wp.vec3,
        normal: wp.vec3,
        a_buffer: wp.array(dtype=wp.vec3),
        b_buffer: wp.array(dtype=wp.vec3),
        result_features: wp.array(dtype=wp.uint32),
        feature_anchor_a: wp.int32,
        feature_anchor_b: wp.int32,
        data_provider: Any,
    ) -> tuple[int, float]:
        """
        Core implementation of multi-contact manifold generation using perturbed support mapping.

        This function discovers contact polygons on both shapes by querying the support function
        in 6 perturbed directions around the collision normal. The perturbed directions form
        a hexagonal pattern tilted 2 degrees from the contact plane. The resulting contact
        polygons are then clipped and reduced to up to 4 contact points.

        The result is stored in a_buffer and b_buffer, which also serve as scratch memory
        during the calculation. a_buffer must have space for 6 elements, b_buffer for 12 elements.
        Both buffers will have the same number of valid entries on return.

        The two shapes must always be queried in the same order to get stable feature IDs
        for contact tracking across frames.

        Args:
            geom_a: Geometry data for shape A.
            geom_b: Geometry data for shape B.
            quaternion_a: Orientation quaternion of shape A.
            quaternion_b: Orientation quaternion of shape B.
            position_a: World position of shape A.
            position_b: World position of shape B.
            p_a: Anchor contact point on shape A (from GJK/MPR).
            p_b: Anchor contact point on shape B (from GJK/MPR).
            normal: Collision normal pointing from A to B.
            a_buffer: Output buffer for shape A contact points (preallocated, size 6).
            b_buffer: Output buffer for shape B contact points (preallocated, size 12).
            result_features: Output buffer for feature IDs (preallocated, size 6).
            feature_anchor_a: Feature ID of anchor point on shape A.
            feature_anchor_b: Feature ID of anchor point on shape B.
            data_provider: Support mapping data provider.

        Returns:
            Tuple of (num_contacts, normal_dot) where num_contacts is the number of valid contact
            points generated (0-4) and normal_dot is the absolute dot product of polygon normals.
        """

        ROT_DELTA_ANGLE = wp.static(2.0 * wp.pi / float(6))

        # Reset all counters for a new calculation.
        a_count = int(0)
        b_count = int(0)

        # Create an orthonormal basis from the collision normal.
        tangent_a, tangent_b = build_orthonormal_basis(normal)

        features_a = vec6_uint8(wp.uint8(0))
        features_b = vec6_uint8(wp.uint8(0))

        # --- Step 1: Find Contact Polygons using Perturbed Support Mapping ---
        # Loop 6 times to find up to 6 vertices for each shape's contact polygon.
        for e in range(6):
            # Create a perturbed normal direction. This is the main collision normal slightly
            # altered by a vector on the contact plane, defined by the hexagonal vertices.
            angle = float(e) * ROT_DELTA_ANGLE
            s = wp.sin(angle)
            c = wp.cos(angle)
            offset_normal = (
                normal * COS_TILT_ANGLE + (c * SIN_TILT_ANGLE) * tangent_a + (s * SIN_TILT_ANGLE) * tangent_b
            )

            # Find the support point on shape A in the perturbed direction.
            # 1. Transform the world-space direction into shape A's local space.
            tmp = wp.quat_rotate_inv(quaternion_a, offset_normal)
            # 2. Find the furthest point on shape A in that local direction.
            (pt_a, feature_a) = support_func(geom_a, tmp, data_provider)
            # 3. Transform the local-space support point back to world space.
            pt_a = wp.quat_rotate(quaternion_a, pt_a) + position_a
            # 4. Add the world-space point to the 'left' polygon, checking for duplicates.
            a_count, was_added_a = add_avoid_duplicates_vec3(a_buffer, a_count, pt_a, EPS)
            # Only store feature ID if the point was actually added (not a duplicate)
            if was_added_a:
                features_a[a_count - 1] = wp.uint8(int(feature_a) + 1)

            # Invert the direction for the other shape.
            offset_normal = -offset_normal

            # Find the support point on shape B in the opposite perturbed direction.
            # (Process is identical to the one for shape A).
            tmp = wp.quat_rotate_inv(quaternion_b, offset_normal)
            (pt_b, feature_b) = support_func(geom_b, tmp, data_provider)
            pt_b = wp.quat_rotate(quaternion_b, pt_b) + position_b
            b_count, was_added_b = add_avoid_duplicates_vec3(b_buffer, b_count, pt_b, EPS)
            # Only store feature ID if the point was actually added (not a duplicate)
            if was_added_b:
                features_b[b_count - 1] = wp.uint8(int(feature_b) + 1)

        # wp.printf("a_count: %d, b_count: %d\n", a_count, b_count)

        # All feature ids are one based such that it is clearly visible in a uint which of the 4 slots (8 bits each) are in use
        num_contacts, normal_dot = extract_4_point_contact_manifolds(
            a_buffer,
            features_a,
            a_count,
            b_buffer,
            features_b,
            b_count,
            normal,
            tangent_a,
            tangent_b,
            p_a,
            p_b,
            result_features,
        )
        return num_contacts, normal_dot

    Mat53f = wp.types.matrix(shape=(5, 3), dtype=wp.float32)
    vec5 = wp.types.vector(5, wp.float32)
    vec5u = wp.types.vector(5, wp.uint32)

    @wp.func
    def build_manifold(
        geom_a: Any,
        geom_b: Any,
        quaternion_a: wp.quat,
        quaternion_b: wp.quat,
        position_a: wp.vec3,
        position_b: wp.vec3,
        p_a: wp.vec3,
        p_b: wp.vec3,
        normal: wp.vec3,
        feature_anchor_a: wp.int32,
        feature_anchor_b: wp.int32,
        data_provider: Any,
    ) -> tuple[
        int,
        vec5,
        Mat53f,
        vec5u,
    ]:
        """
        Build a contact manifold between two convex shapes using perturbed support mapping and polygon clipping.

        This function generates up to 5 contact points between two colliding convex shapes by:
        1. Finding contact polygons using perturbed support mapping in 6 directions
        2. Clipping the polygons against each other in contact plane space
        3. Selecting the best 4 points using rotating calipers algorithm if more than 4 exist
        4. Transforming results back to world space with feature tracking
        5. Optionally appending the deepest contact point if should_include_deepest_contact returns true

        The contact normal is the same for all contact points in the manifold. The two shapes
        must always be queried in the same order to get stable feature IDs for contact tracking.

        Args:
            geom_a: Geometry data for the first shape.
            geom_b: Geometry data for the second shape.
            quaternion_a: Orientation quaternion of the first shape.
            quaternion_b: Orientation quaternion of the second shape.
            position_a: World position of the first shape.
            position_b: World position of the second shape.
            p_a: Anchor contact point on the first shape (from GJK/EPA).
            p_b: Anchor contact point on the second shape (from GJK/EPA).
            normal: Contact normal vector pointing from shape A to shape B.
            feature_anchor_a: Feature ID of the anchor point on shape A. Can pass in 0 if anchor tracking is not needed.
            feature_anchor_b: Feature ID of the anchor point on shape B. Can pass in 0 if anchor tracking is not needed.
            data_provider: Support mapping data provider for shape queries.
        Returns:
            A tuple containing:
            - int: Number of valid contact points in the manifold (0-5).
            - vec5: Signed distances for each contact point (negative when shapes overlap).
            - Mat53f: Contact points at the center of the manifold contact
              (midpoint between points on shape A and shape B) in world space.
            - vec5u: Feature IDs for each contact point, enabling contact tracking across
              multiple frames for warm starting and contact persistence.

        Note:
            The feature IDs encode geometric information about which features (vertices, edges,
            or edge-edge intersections) each contact point represents, allowing the physics
            solver to maintain contact consistency over time.
        """
        left = wp.zeros(shape=(6,), dtype=wp.vec3)  # Array for shape A contact points
        right = wp.zeros(
            shape=(12,), dtype=wp.vec3
        )  # Array for shape B contact points - also provides storage for intermediate results
        result_features = wp.zeros(shape=(6,), dtype=wp.uint32)

        num_manifold_points, normal_dot = build_manifold_core(
            geom_a,
            geom_b,
            quaternion_a,
            quaternion_b,
            position_a,
            position_b,
            p_a,
            p_b,
            normal,
            left,
            right,
            result_features,
            feature_anchor_a,
            feature_anchor_b,
            data_provider,
        )

        # Extract results into fixed-size matrices
        contact_points = Mat53f()
        feature_ids = vec5u(wp.uint32(0), wp.uint32(0), wp.uint32(0), wp.uint32(0), wp.uint32(0))
        signed_distances = vec5(0.0, 0.0, 0.0, 0.0, 0.0)

        # Copy contact points and extract feature IDs
        count_out = wp.min(num_manifold_points, 4)
        for i in range(count_out):
            contact_point_a = left[i]
            contact_point_b = right[i]

            contact_points[i] = 0.5 * (contact_point_a + contact_point_b)

            feature_ids[i] = result_features[i]
            signed_distances[i] = wp.dot(contact_point_b - contact_point_a, normal)

        # Check if we should include the deepest contact point
        if count_out < 5 and count_out > 1:
            # Check if we should include the deepest contact point using the normal_dot
            # computed from the polygon normals in extract_4_point_contact_manifolds
            if should_include_deepest_contact(normal_dot):
                deepest_contact_center = 0.5 * (p_a + p_b)
                contact_points[count_out] = deepest_contact_center
                signed_distances[count_out] = wp.dot(p_b - p_a, normal)
                feature_ids[count_out] = wp.uint32(0)  # Use 0 for the deepest contact feature ID
                count_out += 1

        return count_out, signed_distances, contact_points, feature_ids

    return build_manifold
