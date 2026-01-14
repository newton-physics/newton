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
multiple contact points between colliding shapes. It includes polygon clipping
and contact point selection algorithms.
"""

from typing import Any

import warp as wp

from .contact_data import ContactData
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


@wp.struct
class IncrementalPlaneTracker:
    reference_point: wp.vec3
    previous_point: wp.vec3
    normal: wp.vec3
    largest_area_sq: float


@wp.func
def update_incremental_plane_tracker(
    tracker: IncrementalPlaneTracker,
    current_point: wp.vec3,
    current_point_id: int,
) -> IncrementalPlaneTracker:
    """
    Update the incremental plane tracker with a new point.
    """
    if current_point_id == 0:
        tracker.reference_point = current_point
        tracker.largest_area_sq = 0.0
    elif current_point_id == 1:
        tracker.previous_point = current_point
    else:
        edge1 = tracker.previous_point - tracker.reference_point
        edge2 = current_point - tracker.reference_point
        cross = wp.cross(edge1, edge2)
        area_sq = wp.dot(cross, cross)
        if area_sq > tracker.largest_area_sq:
            tracker.largest_area_sq = area_sq
            tracker.normal = cross
        tracker.previous_point = current_point
    return tracker


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
    plane_tracker_a: IncrementalPlaneTracker,
    anchor_point_a: wp.vec3,
    plane_tracker_b: IncrementalPlaneTracker,
    anchor_point_b: wp.vec3,
    contact_normal: wp.vec3,
) -> tuple[BodyProjector, BodyProjector]:
    projector_a = BodyProjector()
    projector_b = BodyProjector()

    if plane_tracker_a.largest_area_sq == 0.0 and plane_tracker_b.largest_area_sq == 0.0:
        # Both are line segments - compute normals using contact_normal as reference
        dir_a = plane_tracker_a.previous_point - plane_tracker_a.reference_point
        dir_b = plane_tracker_b.previous_point - plane_tracker_b.reference_point

        point_on_plane_a = 0.5 * (plane_tracker_a.reference_point + plane_tracker_a.previous_point)
        projector_a.normal = compute_line_segment_projector_normal(dir_a, contact_normal)
        projector_a.plane_d = -wp.dot(point_on_plane_a, projector_a.normal)

        point_on_plane_b = 0.5 * (plane_tracker_b.reference_point + plane_tracker_b.previous_point)
        projector_b.normal = compute_line_segment_projector_normal(dir_b, contact_normal)
        projector_b.plane_d = -wp.dot(point_on_plane_b, projector_b.normal)

        return projector_a, projector_b

    if plane_tracker_a.largest_area_sq > 0.0:
        len_n = wp.sqrt(wp.max(1.0e-12, plane_tracker_a.largest_area_sq))
        projector_a.normal = plane_tracker_a.normal / len_n
        projector_a.plane_d = -wp.dot(anchor_point_a, projector_a.normal)
    if plane_tracker_b.largest_area_sq > 0.0:
        len_n = wp.sqrt(wp.max(1.0e-12, plane_tracker_b.largest_area_sq))
        projector_b.normal = plane_tracker_b.normal / len_n
        projector_b.plane_d = -wp.dot(anchor_point_b, projector_b.normal)

    if plane_tracker_a.largest_area_sq == 0.0:
        dir = plane_tracker_a.previous_point - plane_tracker_a.reference_point
        point_on_plane_a = 0.5 * (plane_tracker_a.reference_point + plane_tracker_a.previous_point)
        projector_a.normal = compute_line_segment_projector_normal(dir, projector_b.normal)
        projector_a.plane_d = -wp.dot(point_on_plane_a, projector_a.normal)

    if plane_tracker_b.largest_area_sq == 0.0:
        dir = plane_tracker_b.previous_point - plane_tracker_b.reference_point
        point_on_plane_b = 0.5 * (plane_tracker_b.reference_point + plane_tracker_b.previous_point)
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
def intersection_point(trim_seg_start: wp.vec2, trim_seg_end: wp.vec2, a: wp.vec2, b: wp.vec2) -> wp.vec2:
    """
    Calculate the intersection point between a line segment and a polygon edge.

    It is known that a and b lie on different sides of the trim segment.

    Args:
        trim_seg_start: Start point of the trimming segment.
        trim_seg_end: End point of the trimming segment.
        a: First point of the polygon edge.
        b: Second point of the polygon edge.

    Returns:
        The intersection point as a vec2.
    """
    # Since a and b are on opposite sides, their signed areas have opposite signs
    # We can optimize: abs(signed_a) + abs(signed_b) = abs(signed_a - signed_b)
    signed_a = signed_area(trim_seg_start, trim_seg_end, a)
    signed_b = signed_area(trim_seg_start, trim_seg_end, b)
    interp_ab = wp.abs(signed_a) / wp.abs(signed_a - signed_b)

    # Interpolate between a and b
    return (1.0 - interp_ab) * a + interp_ab * b


@wp.func
def insert_vec2(arr: wp.array(dtype=wp.vec2), arr_count: int, index: int, element: wp.vec2):
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
    trim_seg_start: wp.vec2,
    trim_seg_end: wp.vec2,
    trim_seg_id: wp.uint8,
    loop: wp.array(dtype=wp.vec2),
    loop_seg_ids: wp.array(dtype=wp.uint8),
    loop_count: int,
) -> int:
    """
    Trim a polygon in place using a line segment.

    All points are in 2D contact plane space.

    loopSegIds[0] refers to the segment from loop[0] to loop[1], etc.

    Args:
        trim_seg_start: Start point of the trimming segment.
        trim_seg_end: End point of the trimming segment.
        trim_seg_id: ID of the trimming segment.
        loop: Array of loop vertices (2D).
        loop_seg_ids: Array of segment IDs for the loop.
        loop_count: Number of vertices in the loop.

    Returns:
        New number of vertices in the trimmed loop.
    """
    if loop_count < 3:
        return loop_count

    intersection_a = wp.vec2(0.0, 0.0)
    change_a = int(-1)
    change_a_seg_id = wp.uint8(255)
    intersection_b = wp.vec2(0.0, 0.0)
    change_b = int(-1)
    change_b_seg_id = wp.uint8(255)

    keep = bool(False)

    # Check first vertex
    prev_outside = bool(signed_area(trim_seg_start, trim_seg_end, loop[0]) <= 0.0)

    for i in range(loop_count):
        next_idx = (i + 1) % loop_count
        outside = signed_area(trim_seg_start, trim_seg_end, loop[next_idx]) <= 0.0

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
                    insert_vec2(loop, new_loop_count, loop_indexer, pt)
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
    trim_poly: wp.array(dtype=wp.vec2),
    trim_poly_count: int,
    loop: wp.array(dtype=wp.vec2),
    loop_segments: wp.array(dtype=wp.uint8),
    loop_count: int,
) -> int:
    """
    Trim a polygon using all edges of another polygon.

    Both polygons (trim_poly and loop) are in 2D contact plane space and they are both convex.

    Args:
        trim_poly: Array of vertices defining the trimming polygon (2D).
        trim_poly_count: Number of vertices in the trimming polygon.
        loop: Array of vertices in the loop to be trimmed (2D).
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

        # Direction vector
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

            trim_poly[0] = wp.vec2(p0[0] - offset_x, p0[1] - offset_y)
            trim_poly[1] = wp.vec2(p1[0] - offset_x, p1[1] - offset_y)
            trim_poly[2] = wp.vec2(p1[0] + offset_x, p1[1] + offset_y)
            trim_poly[3] = wp.vec2(p0[0] + offset_x, p0[1] + offset_y)
            trim_poly_count = 4
        else:
            return wp.min(1, loop_count)

    if loop_count == 2:
        # Convert line segment to thin rectangle
        p0 = loop[0]
        p1 = loop[1]
        seg0 = loop_segments[0]
        seg1 = loop_segments[1]

        # Direction vector
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

            loop[0] = wp.vec2(p0[0] - offset_x, p0[1] - offset_y)
            loop[1] = wp.vec2(p1[0] - offset_x, p1[1] - offset_y)
            loop[2] = wp.vec2(p1[0] + offset_x, p1[1] + offset_y)
            loop[3] = wp.vec2(p0[0] + offset_x, p0[1] + offset_y)

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

    trim_poly_0 = trim_poly[0]  # This allows to do more memory aliasing
    for i in range(trim_poly_count):
        # For each trim segment, we will call the efficient trim function.
        trim_seg_start = trim_poly[i]
        # trim_seg_end = trim_poly[(i + 1) % trim_poly_count]
        trim_seg_end = trim_poly_0 if i == trim_poly_count - 1 else trim_poly[i + 1]
        # Perform the in-place trimming for this segment.
        current_loop_count = trim_in_place(
            trim_seg_start, trim_seg_end, wp.uint8(i), loop, loop_segments, current_loop_count
        )

    return current_loop_count


@wp.func
def approx_max_quadrilateral_area_with_calipers(hull: wp.array(dtype=wp.vec2), hull_count: int) -> wp.vec4i:
    """
    Finds an approximate maximum area quadrilateral inside a convex hull in O(n) time
    using the Rotating Calipers algorithm to find the hull's diameter.

    Args:
        hull: Array of hull vertices (2D).
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
    diff = wp.vec2(hp1[0] - hp3[0], hp1[1] - hp3[1])
    max_dist_sq = diff[0] * diff[0] + diff[1] * diff[1]

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
        hull_i = hull[i]
        hull_i_plus_1 = hull[(i + 1) % n]

        while True:
            hull_j = hull[j]
            hull_j_plus_1 = hull[(j + 1) % n]

            area_j_plus_1 = signed_area(hull_i, hull_i_plus_1, hull_j_plus_1)
            area_j = signed_area(hull_i, hull_i_plus_1, hull_j)

            if area_j_plus_1 > area_j:
                j = (j + 1) % n
            else:
                break

        # Now, (i, j) is an antipodal pair. Check its distance (2D)
        hi = hull[i]
        hj = hull[j]
        d1 = wp.vec2(hi[0] - hj[0], hi[1] - hj[1])
        dist_sq_1 = d1[0] * d1[0] + d1[1] * d1[1]
        # Use relative tie-breaking: only update if new distance is meaningfully larger
        if dist_sq_1 > max_dist_sq * (1.0 + tie_epsilon_rel):
            max_dist_sq = dist_sq_1
            p1 = i
            p3 = j

        # The next point, (i+1, j), is also an antipodal pair. Check its distance too (2D)
        hip1 = hull[(i + 1) % n]
        d2 = wp.vec2(hip1[0] - hj[0], hip1[1] - hj[1])
        dist_sq_2 = d2[0] * d2[0] + d2[1] * d2[1]
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

    hull_p1 = hull[p1]
    hull_p3 = hull[p3]

    for i in range(n):
        # Use the signed area to determine which side of the line the point is on.
        hull_i = hull[i]
        area = signed_area(hull_p1, hull_p3, hull_i)

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
    loop: wp.array(dtype=wp.vec2), loop_seg_ids: wp.array(dtype=wp.uint8), loop_count: int, eps: float
) -> int:
    """
    Remove zero-length edges from a polygon loop.

    Args:
        loop: Array of loop vertices (2D).
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
        diff = loop[read_idx] - loop[write_idx]

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
        diff = loop[write_idx] - loop[0]

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
def add_avoid_duplicates_vec2(
    arr: wp.array(dtype=wp.vec2), arr_count: int, vec: wp.vec2, eps: float
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


@wp.func_native("""
    return (uint64_t)a.data;
""")
def get_ptr(a: wp.array(dtype=wp.vec2)) -> wp.uint64: ...


def create_build_manifold(support_func: Any, writer_func: Any, post_process_contact: Any):
    """
    Factory function to create manifold generation functions with a specific support mapping function.

    This factory creates two related functions for multi-contact manifold generation:
    - build_manifold_core: The core implementation that uses preallocated buffers
    - build_manifold: The main entry point that handles buffer allocation and result extraction

    Args:
        support_func: Support mapping function for shapes that takes
                     (geometry, direction, data_provider) and returns a support point
        writer_func: Function to write contact data (signature: (ContactData, writer_data) -> None)
        post_process_contact: Function to post-process contact data

    Returns:
        build_manifold function that generates up to 5 contact points between two shapes
        using perturbed support mapping and polygon clipping.
    """

    @wp.func
    def extract_4_point_contact_manifolds(
        m_a: wp.array(dtype=wp.vec2),
        m_a_count: int,
        m_b: wp.array(dtype=wp.vec2),
        m_b_count: int,
        normal: wp.vec3,
        cross_vector_1: wp.vec3,
        cross_vector_2: wp.vec3,
        center: wp.vec3,
        projector_a: BodyProjector,
        projector_b: BodyProjector,
        writer_data: Any,
        contact_template: Any,
        geom_a: Any,
        geom_b: Any,
        position_a: wp.vec3,
        position_b: wp.vec3,
        quaternion_a: wp.quat,
        quaternion_b: wp.quat,
    ) -> tuple[int, float]:
        """
        Extract up to 4 contact points from two convex contact polygons and write them immediately.

        This function performs the core manifold generation algorithm:
        1. Validates input polygons (already in 2D contact plane space)
        2. Clips polygon B against all edges of polygon A (Sutherland-Hodgman style clipping)
        3. Removes zero-length edges from the clipped result
        4. If more than 4 points remain, selects the best 4 using rotating calipers algorithm
        5. Projects all contact points back onto the original shape surfaces in world space
        6. Post-processes and writes each contact immediately

        Uses writer_func and post_process_contact from the factory closure.

        Args:
            m_a: Contact polygon vertices for shape A (2D contact plane space, up to 6 points).
            m_a_count: Number of vertices in polygon A.
            m_b: Contact polygon vertices for shape B (2D contact plane space, up to 6 points, space for 12).
            m_b_count: Number of vertices in polygon B.
            normal: Collision normal vector pointing from A to B.
            cross_vector_1: First tangent vector (forms contact plane basis).
            cross_vector_2: Second tangent vector (forms contact plane basis).
            center: Center point for back-projection to world space.
            projector_a: Body projector for shape A.
            projector_b: Body projector for shape B.
            writer_data: Data structure for contact writer.
            contact_template: Pre-packed ContactData with static fields.
            geom_a: Geometry data for shape A.
            geom_b: Geometry data for shape B.
            position_a: World position of shape A.
            position_b: World position of shape B.
            quaternion_a: Orientation of shape A.
            quaternion_b: Orientation of shape B.

        Returns:
            Tuple of (loop_count, normal_dot) where:
            - loop_count: Number of valid contact points written (0-4)
            - normal_dot: Absolute dot product of polygon normals
        """

        normal_dot = wp.abs(wp.dot(projector_a.normal, projector_b.normal))

        # Initialize loop segment IDs for polygon B
        loop_seg_ids = wp.zeros(shape=(12,), dtype=wp.uint8)
        for i in range(m_b_count):
            loop_seg_ids[i] = wp.uint8(i + 6)

        loop_count = trim_all_in_place(m_a, m_a_count, m_b, loop_seg_ids, m_b_count)

        loop_count = remove_zero_length_edges(m_b, loop_seg_ids, loop_count, EPS)

        if loop_count > 1:
            result = wp.vec4i()
            if loop_count > 4:
                result = approx_max_quadrilateral_area_with_calipers(m_b, loop_count)
                loop_count = 4
            else:
                result = wp.vec4i(0, 1, 2, 3)

            for i in range(loop_count):
                ia = int(result[i])

                # Transform back to world space using projectors
                p_world = m_b[ia].x * cross_vector_1 + m_b[ia].y * cross_vector_2 + center

                # normal vector points from A to B
                a = body_projector_project(projector_a, p_world, normal)
                b = body_projector_project(projector_b, p_world, normal)
                contact_point = 0.5 * (a + b)
                signed_distance = wp.dot(b - a, normal)

                # Write contact immediately
                contact_data = contact_template
                contact_data.contact_point_center = contact_point
                contact_data.contact_normal_a_to_b = normal
                contact_data.contact_distance = signed_distance

                contact_data = post_process_contact(
                    contact_data, geom_a, position_a, quaternion_a, geom_b, position_b, quaternion_b
                )
                writer_func(contact_data, writer_data, -1)
        else:
            normal_dot = 0.0
            loop_count = 0

        return loop_count, normal_dot

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
        data_provider: Any,
        writer_data: Any,
        contact_template: ContactData,
    ) -> int:
        """
        Build a contact manifold between two convex shapes and write contacts directly.

        This function generates up to 4 contact points between two colliding convex shapes by:
        1. Finding contact polygons using perturbed support mapping in 6 directions
        2. Clipping the polygons against each other in contact plane space
        3. Selecting the best 4 points using rotating calipers algorithm if more than 4 exist
        4. Transforming results back to world space
        5. Post-processing each contact and writing it via the writer function

        The contact normal is the same for all contact points in the manifold.

        Args:
            geom_a: Geometry data for the first shape.
            geom_b: Geometry data for the second shape.
            quaternion_a: Orientation quaternion of the first shape.
            quaternion_b: Orientation quaternion of the second shape.
            position_a: World position of the first shape.
            position_b: World position of the second shape.
            p_a: Anchor contact point on the first shape (from GJK/MPR).
            p_b: Anchor contact point on the second shape (from GJK/MPR).
            normal: Contact normal vector pointing from shape A to shape B.
            data_provider: Support mapping data provider for shape queries.
            writer_data: Data structure for contact writer.
            contact_template: Pre-packed ContactData with static fields.

        Returns:
            Number of valid contact points written (0-5).
        """

        ROT_DELTA_ANGLE = wp.static(2.0 * wp.pi / float(6))

        # Reset all counters for a new calculation.
        a_count = int(0)
        b_count = int(0)

        # Create an orthonormal basis from the collision normal.
        tangent_a, tangent_b = build_orthonormal_basis(normal)

        plane_tracker_a = IncrementalPlaneTracker()
        plane_tracker_b = IncrementalPlaneTracker()

        # Compute center for 2D projection
        center = 0.5 * (p_a + p_b)

        # Allocate buffers for contact polygons (2D projected)
        b_buffer = wp.zeros(shape=(12,), dtype=wp.vec2f)
        # a_buffer = wp.array(ptr=b_buffer.ptr + wp.uint64(6 * 8), shape=(6,), dtype=wp.vec2f)
        a_buffer = wp.array(ptr=get_ptr(b_buffer) + wp.uint64(6 * 8), shape=(6,), dtype=wp.vec2f)

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
            pt_a = support_func(geom_a, tmp, data_provider)
            # 3. Transform the local-space support point back to world space.
            pt_a_3d = wp.quat_rotate(quaternion_a, pt_a) + position_a
            # 4. Project to 2D contact plane space
            projected_a = pt_a_3d - center
            pt_a_2d = wp.vec2(wp.dot(tangent_a, projected_a), wp.dot(tangent_b, projected_a))
            # 5. Add the 2D projected point, checking for duplicates.
            a_count, was_added_a = add_avoid_duplicates_vec2(a_buffer, a_count, pt_a_2d, EPS)
            if was_added_a:
                plane_tracker_a = update_incremental_plane_tracker(plane_tracker_a, pt_a_3d, a_count - 1)

            # Invert the direction for the other shape.
            offset_normal = -offset_normal

            # Find the support point on shape B in the opposite perturbed direction.
            # (Process is identical to the one for shape A).
            tmp = wp.quat_rotate_inv(quaternion_b, offset_normal)
            pt_b = support_func(geom_b, tmp, data_provider)
            pt_b_3d = wp.quat_rotate(quaternion_b, pt_b) + position_b
            # Project to 2D contact plane space
            projected_b = pt_b_3d - center
            pt_b_2d = wp.vec2(wp.dot(tangent_a, projected_b), wp.dot(tangent_b, projected_b))
            b_count, was_added_b = add_avoid_duplicates_vec2(b_buffer, b_count, pt_b_2d, EPS)
            if was_added_b:
                plane_tracker_b = update_incremental_plane_tracker(plane_tracker_b, pt_b_3d, b_count - 1)

        # Early-out for simple cases: if both have <=2 or either is empty
        if a_count < 2 or b_count < 2:
            count_out = 0
            normal_dot = 0.0
        else:
            # Projectors for back-projection onto the shape surfaces
            projector_a, projector_b = create_body_projectors(plane_tracker_a, p_a, plane_tracker_b, p_b, normal)

            if excess_normal_deviation(normal, projector_a.normal) or excess_normal_deviation(
                normal, projector_b.normal
            ):
                count_out = 0
                normal_dot = 0.0
            else:
                # Extract and write up to 4 contact points
                num_manifold_points, normal_dot = extract_4_point_contact_manifolds(
                    a_buffer,
                    a_count,
                    b_buffer,
                    b_count,
                    normal,
                    tangent_a,
                    tangent_b,
                    0.5 * (p_a + p_b),
                    projector_a,
                    projector_b,
                    writer_data,
                    contact_template,
                    geom_a,
                    geom_b,
                    position_a,
                    position_b,
                    quaternion_a,
                    quaternion_b,
                )
                count_out = wp.min(num_manifold_points, 4)

        # Check if we should include the deepest contact point using the normal_dot
        # computed from the polygon normals in extract_4_point_contact_manifolds
        if should_include_deepest_contact(normal_dot) or count_out == 0:
            # Write the deepest contact immediately
            deepest_contact_center = 0.5 * (p_a + p_b)
            deepest_signed_distance = wp.dot(p_b - p_a, normal)

            contact_data = contact_template
            contact_data.contact_point_center = deepest_contact_center
            contact_data.contact_normal_a_to_b = normal
            contact_data.contact_distance = deepest_signed_distance

            contact_data = post_process_contact(
                contact_data, geom_a, position_a, quaternion_a, geom_b, position_b, quaternion_b
            )
            writer_func(contact_data, writer_data, -1)

            count_out += 1

        return count_out

    return build_manifold
