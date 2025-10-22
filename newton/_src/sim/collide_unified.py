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


from __future__ import annotations

from enum import IntEnum

import warp as wp

from ..core.types import Devicelike
from ..geometry.broad_phase_nxn import BroadPhaseAllPairs, BroadPhaseExplicit
from ..geometry.broad_phase_sap import BroadPhaseSAP
from ..geometry.collision_convex import create_solve_convex_multi_contact
from ..geometry.support_function import (
    GenericShapeData,
    GeoTypeEx,
    SupportMapDataProvider,
    pack_mesh_ptr,
    support_map,
)
from ..geometry.support_function import (
    support_map as support_map_func,
)
from ..geometry.types import GeoType
from .contacts import Contacts
from .model import Model
from .state import State


class BroadPhaseMode(IntEnum):
    """Broad phase collision detection mode.

    Attributes:
        NXN: All-pairs broad phase with AABB checks (simple, O(N²) but good for small scenes)
        SAP: Sweep and Prune broad phase with AABB sorting (faster for larger scenes, O(N log N))
        EXPLICIT: Use precomputed shape pairs (most efficient when pairs are known ahead of time)
    """

    NXN = 0
    SAP = 1
    EXPLICIT = 2


# Pre-create the convex multi-contact solver (usable inside kernels)
solve_convex_multi_contact = create_solve_convex_multi_contact(support_map_func)

# Type definitions for multi-contact manifolds
_mat53f = wp.types.matrix((5, 3), wp.float32)
_vec5 = wp.types.vector(5, wp.float32)


@wp.func
def write_contact(
    contact_point_center: wp.vec3,
    contact_normal_a_to_b: wp.vec3,
    contact_distance: float,
    radius_eff_a: float,
    radius_eff_b: float,
    thickness_a: float,
    thickness_b: float,
    shape_a: int,
    shape_b: int,
    X_bw_a: wp.transform,
    X_bw_b: wp.transform,
    tid: int,
    rigid_contact_margin: float,
    contact_max: int,
    # outputs
    contact_count: wp.array(dtype=int),
    out_shape0: wp.array(dtype=int),
    out_shape1: wp.array(dtype=int),
    out_point0: wp.array(dtype=wp.vec3),
    out_point1: wp.array(dtype=wp.vec3),
    out_offset0: wp.array(dtype=wp.vec3),
    out_offset1: wp.array(dtype=wp.vec3),
    out_normal: wp.array(dtype=wp.vec3),
    out_thickness0: wp.array(dtype=float),
    out_thickness1: wp.array(dtype=float),
    out_tids: wp.array(dtype=int),
):
    """
    Write a contact to the output arrays.

    Args:
        contact_point_center: Center point of contact in world space
        contact_normal_a_to_b: Contact normal pointing from shape A to B
        contact_distance: Distance between contact points
        radius_eff_a: Effective radius of shape A (only use nonzero values for shapes that are a minkowski sum of a sphere and another object, eg sphere or capsule)
        radius_eff_b: Effective radius of shape B (only use nonzero values for shapes that are a minkowski sum of a sphere and another object, eg sphere or capsule)
        thickness_a: Contact thickness for shape A (similar to contact offset)
        thickness_b: Contact thickness for shape B (similar to contact offset)
        shape_a: Shape A index
        shape_b: Shape B index
        X_bw_a: Transform from world to body A
        X_bw_b: Transform from world to body B
        tid: Thread ID
        rigid_contact_margin: Contact margin for rigid bodies
        contact_max: Maximum number of contacts
        contact_count: Array to track contact count
        out_shape0: Output array for shape A indices
        out_shape1: Output array for shape B indices
        out_point0: Output array for contact points on shape A
        out_point1: Output array for contact points on shape B
        out_offset0: Output array for offsets on shape A
        out_offset1: Output array for offsets on shape B
        out_normal: Output array for contact normals
        out_thickness0: Output array for thickness values for shape A
        out_thickness1: Output array for thickness values for shape B
        out_tids: Output array for thread IDs
    """

    total_separation_needed = radius_eff_a + radius_eff_b + thickness_a + thickness_b

    offset_mag_a = radius_eff_a + thickness_a
    offset_mag_b = radius_eff_b + thickness_b

    # Distance calculation matching box_plane_collision
    contact_normal_a_to_b = wp.normalize(contact_normal_a_to_b)

    a_contact_world = contact_point_center - contact_normal_a_to_b * (0.5 * contact_distance + radius_eff_a)
    b_contact_world = contact_point_center + contact_normal_a_to_b * (0.5 * contact_distance + radius_eff_b)

    diff = b_contact_world - a_contact_world
    distance = wp.dot(diff, contact_normal_a_to_b)
    d = distance - total_separation_needed
    if d < rigid_contact_margin:
        index = wp.atomic_add(contact_count, 0, 1)
        if index >= contact_max:
            # Reached buffer limit
            return

        out_shape0[index] = shape_a
        out_shape1[index] = shape_b

        # Contact points are stored in body frames
        out_point0[index] = wp.transform_point(X_bw_a, a_contact_world)
        out_point1[index] = wp.transform_point(X_bw_b, b_contact_world)

        # Match kernels.py convention: normal should point from box to plane (downward)
        contact_normal = -contact_normal_a_to_b

        # Offsets in body frames
        out_offset0[index] = wp.transform_vector(X_bw_a, -offset_mag_a * contact_normal)
        out_offset1[index] = wp.transform_vector(X_bw_b, offset_mag_b * contact_normal)

        out_normal[index] = contact_normal
        out_thickness0[index] = offset_mag_a
        out_thickness1[index] = offset_mag_b
        out_tids[index] = tid


@wp.func
def convert_infinite_plane_to_cube(
    shape_data: GenericShapeData,
    plane_rotation: wp.quat,
    plane_position: wp.vec3,
    other_position: wp.vec3,
    other_radius: float,
) -> tuple[GenericShapeData, wp.vec3]:
    """
    Convert an infinite plane into a cube proxy for GJK/MPR collision detection.

    Since GJK/MPR cannot handle infinite planes, we create a finite cube where:
    - The cube is positioned at the plane surface, directly under/over the other object
    - The cube's lateral dimensions are sized based on the other object's bounding sphere
    - The cube extends 'downward' from the plane (in -Z direction in plane's local frame)

    Args:
        shape_data: The plane's shape data (should have shape_type == GeoType.PLANE)
        plane_rotation: The plane's orientation (plane normal is along local +Z)
        plane_position: The plane's position in world space
        other_position: The other object's position in world space
        other_radius: Bounding sphere radius of the colliding object

    Returns:
        Tuple of (modified_shape_data, adjusted_position):
        - modified_shape_data: GenericShapeData configured as a BOX
        - adjusted_position: The cube's center position (centered on other object projected to plane)
    """
    result = GenericShapeData()
    result.shape_type = int(GeoType.BOX)

    # Size the cube based on the other object's bounding sphere radius
    # Make it large enough to always contain potential contact points
    # The lateral dimensions (x, y) should be at least 2x the radius to ensure coverage
    lateral_size = other_radius * 10.0

    # The depth (z) should be large enough to encompass the potential collision region
    # Make it extend from above the plane surface to well below
    depth = other_radius * 10.0

    # Set the box half-extents
    # x, y: lateral coverage (parallel to plane)
    # z: depth perpendicular to plane
    result.scale = wp.vec3(lateral_size, lateral_size, depth)

    # Position the cube center at the plane surface, directly under/over the other object
    # Project the other object's position onto the plane
    plane_normal = wp.quat_rotate(plane_rotation, wp.vec3(0.0, 0.0, 1.0))
    to_other = other_position - plane_position
    distance_along_normal = wp.dot(to_other, plane_normal)

    # Point on plane surface closest to the other object
    plane_surface_point = other_position - plane_normal * distance_along_normal

    # Position cube center slightly below the plane surface so the top face is at the surface
    # Since the cube has half-extent 'depth', its top face is at center + depth*normal
    # We want: center + depth*normal = plane_surface, so center = plane_surface - depth*normal
    adjusted_position = plane_surface_point - plane_normal * depth

    return result, adjusted_position


@wp.func
def extract_shape_data(
    shape_idx: int,
    body_q: wp.array(dtype=wp.transform),
    shape_transform: wp.array(dtype=wp.transform),
    shape_body: wp.array(dtype=int),
    shape_type: wp.array(dtype=int),
    shape_scale: wp.array(dtype=wp.vec3),
    shape_collision_radius: wp.array(dtype=float),
    shape_source_ptr: wp.array(dtype=wp.uint64),
):
    """
    Extract shape data for any primitive shape type.

    Args:
        shape_idx: Index of the shape
        body_q: Body transforms
        shape_transform: Shape local transforms
        shape_body: Shape to body mapping
        shape_type: Shape types
        shape_scale: Shape scales
        shape_collision_radius: Precomputed collision radius
        shape_source_ptr: Array of mesh/SDF source pointers

    Returns:
        tuple: (position, orientation, shape_data, bounding_sphere_center, bounding_sphere_radius)
    """
    # Get shape's world transform
    body_idx = shape_body[shape_idx]
    X_ws = shape_transform[shape_idx]
    if body_idx >= 0:
        X_ws = wp.transform_multiply(body_q[body_idx], shape_transform[shape_idx])

    position = wp.transform_get_translation(X_ws)
    orientation = wp.transform_get_rotation(X_ws)

    # Create generic shape data
    result = GenericShapeData()
    result.shape_type = shape_type[shape_idx]
    result.scale = shape_scale[shape_idx]
    result.auxiliary = wp.vec3(0.0, 0.0, 0.0)

    # For CONVEX_MESH, pack the mesh pointer into auxiliary
    if shape_type[shape_idx] == int(GeoType.CONVEX_MESH):
        result.auxiliary = pack_mesh_ptr(shape_source_ptr[shape_idx])

    # For primitive shapes, bounding sphere center is the shape center
    bounding_sphere_center = position
    bounding_sphere_radius = shape_collision_radius[shape_idx]

    return position, orientation, result, bounding_sphere_center, bounding_sphere_radius


@wp.func
def is_discrete_shape(shape_type: int) -> bool:
    """A discrete shape can be represented with a finite amount of flat polygon faces."""
    return (
        shape_type == int(GeoType.BOX)
        or shape_type == int(GeoType.CONVEX_MESH)
        or shape_type == int(GeoTypeEx.TRIANGLE)
        or shape_type == int(GeoType.PLANE)
    )


@wp.func
def project_point_onto_plane(point: wp.vec3, plane_point: wp.vec3, plane_normal: wp.vec3) -> wp.vec3:
    """
    Project a point onto a plane defined by a point and normal.

    Args:
        point: The point to project
        plane_point: A point on the plane
        plane_normal: Normal vector of the plane (should be normalized)

    Returns:
        The projected point on the plane
    """
    to_point = point - plane_point
    distance_to_plane = wp.dot(to_point, plane_normal)
    projected_point = point - plane_normal * distance_to_plane
    return projected_point


@wp.func
def compute_plane_normal_from_contacts(
    points: _mat53f,
    normal: wp.vec3,
    signed_distances: _vec5,
    count: int,
) -> wp.vec3:
    """
    Compute plane normal from reconstructed plane points.

    Reconstructs the plane points from contact data and computes the plane normal
    using fan triangulation to find the largest area triangle for numerical stability.

    Args:
        points: Contact points matrix (5x3)
        normal: Initial contact normal (used for reconstruction)
        signed_distances: Signed distances vector (5 elements)
        count: Number of contact points

    Returns:
        Normalized plane normal from the contact points
    """
    if count < 3:
        # Not enough points to form a triangle, return original normal
        return normal

    # Reconstruct plane points from contact data
    # Use first point as anchor for fan triangulation
    # Contact points are at midpoint, move to discrete surface (plane)
    p0 = points[0] + normal * (signed_distances[0] * 0.5)

    # Find the triangle with the largest area for numerical stability
    # This avoids issues with nearly collinear points
    best_normal = wp.vec3(0.0, 0.0, 0.0)
    max_area_sq = float(0.0)

    for i in range(1, count - 1):
        # Reconstruct plane points for this triangle
        pi = points[i] + normal * (signed_distances[i] * 0.5)
        pi_next = points[i + 1] + normal * (signed_distances[i + 1] * 0.5)

        # Compute cross product for triangle (p0, pi, pi_next)
        edge1 = pi - p0
        edge2 = pi_next - p0
        cross = wp.cross(edge1, edge2)
        area_sq = wp.dot(cross, cross)

        if area_sq > max_area_sq:
            max_area_sq = area_sq
            best_normal = cross

    # Normalize, avoid zero
    len_n = wp.sqrt(wp.max(1.0e-12, max_area_sq))
    plane_normal = best_normal / len_n

    # Ensure normal points in same direction as original normal
    if wp.dot(plane_normal, normal) < 0.0:
        plane_normal = -plane_normal

    return plane_normal


@wp.func
def postprocess_axial_shape_discrete_contacts(
    points: _mat53f,
    normal: wp.vec3,
    signed_distances: _vec5,
    count: int,
    shape_rot: wp.quat,
    shape_radius: float,
    shape_half_height: float,
    shape_pos: wp.vec3,
    is_cone: bool,
) -> tuple[int, _vec5, _mat53f]:
    """
    Post-process contact points for axial shape (cylinder/cone) vs discrete surface collisions.

    When an axial shape is rolling on a discrete surface (plane, box, convex hull),
    we project contact points onto a plane perpendicular to both the shape axis and
    contact normal to stabilize rolling contacts.

    Works for:
    - Cylinders: Axis perpendicular to surface normal when rolling
    - Cones: Axis at an angle = cone half-angle when rolling on base

    Args:
        points: Contact points matrix (5x3)
        normal: Contact normal (from discrete to shape)
        signed_distances: Signed distances vector (5 elements)
        count: Number of input contact points
        shape_rot: Shape orientation
        shape_radius: Shape radius (constant for cylinder, base radius for cone)
        shape_half_height: Shape half height
        shape_pos: Shape position
        is_cone: True if shape is a cone, False if cylinder

    Returns:
        Tuple of (new_count, new_signed_distances, new_points)
    """
    # Get shape axis in world space (Z-axis for both cylinders and cones)
    shape_axis = wp.quat_rotate(shape_rot, wp.vec3(0.0, 0.0, 1.0))

    # Check if shape is in rolling configuration
    axis_normal_dot = wp.abs(wp.dot(shape_axis, normal))

    # Compute threshold based on shape type
    if is_cone:
        # For a cone rolling on its base, the axis makes an angle with the normal
        # equal to the cone's half-angle: angle = atan(radius / (2 * half_height))
        # When rolling: dot(axis, normal) = cos(90 - angle) = sin(angle)
        # Add tolerance of +/-2 degrees
        cone_half_angle = wp.atan2(shape_radius, 2.0 * shape_half_height)
        tolerance_angle = wp.static(2.0 * wp.pi / 180.0)  # 2 degrees
        lower_threshold = wp.sin(cone_half_angle - tolerance_angle)
        upper_threshold = wp.sin(cone_half_angle + tolerance_angle)

        # Check if axis_normal_dot is in the expected range for rolling
        if axis_normal_dot < lower_threshold or axis_normal_dot > upper_threshold:
            # Not in rolling configuration
            return count, signed_distances, points
    else:
        # For cylinder: axis should be perpendicular to normal (dot product ≈ 0)
        perpendicular_threshold = wp.static(wp.sin(2.0 * wp.pi / 180.0))
        if axis_normal_dot > perpendicular_threshold:
            # Not rolling, return original contacts
            return count, signed_distances, points

    # Estimate plane from contact points using the contact normal
    # Use first contact point as plane reference
    if count == 0:
        return 0, signed_distances, points

    # Compute plane normal from the largest area triangle formed by contact points
    # shape_plane_normal = compute_plane_normal_from_contacts(points, normal, signed_distances, count)
    # projection_plane_normal = wp.normalize(wp.cross(shape_axis, shape_plane_normal))

    projection_plane_normal = wp.normalize(wp.cross(shape_axis, normal))
    point_on_projection_plane = shape_pos

    # Project points onto the projection plane and remove duplicates in one pass
    # This avoids creating intermediate arrays and saves registers
    tolerance = shape_radius * 0.01  # 1% of radius for duplicate detection
    output_count = int(0)
    first_point = wp.vec3(0.0, 0.0, 0.0)

    for i in range(count):
        # Project contact point onto projection plane
        projected_point = project_point_onto_plane(points[i], point_on_projection_plane, projection_plane_normal)
        is_duplicate = False

        if output_count > 0:
            # Check against previous output point
            if wp.length(projected_point - points[output_count - 1]) < tolerance:
                is_duplicate = True

        if not is_duplicate and i > 0 and i == count - 1 and output_count > 0:
            # Last point: check against first point (cyclic)
            if wp.length(projected_point - first_point) < tolerance:
                is_duplicate = True

        if not is_duplicate:
            points[output_count] = projected_point
            signed_distances[output_count] = signed_distances[i]
            if output_count == 0:
                first_point = projected_point
            output_count += 1

    return output_count, signed_distances, points


@wp.kernel(enable_backward=False)
def build_contacts_kernel_gjk_mpr(
    body_q: wp.array(dtype=wp.transform),
    body_qd: wp.array(dtype=wp.spatial_vector),
    shape_transform: wp.array(dtype=wp.transform),
    shape_body: wp.array(dtype=int),
    shape_type: wp.array(dtype=int),
    shape_scale: wp.array(dtype=wp.vec3),
    shape_thickness: wp.array(dtype=float),
    shape_collision_radius: wp.array(dtype=float),
    shape_pairs: wp.array(dtype=wp.vec2i),
    sum_of_contact_offsets: float,
    rigid_contact_margin: float,
    contact_max: int,
    sap_shape_pair_count: wp.array(dtype=int),
    shape_source_ptr: wp.array(dtype=wp.uint64),
    total_num_threads: int,
    # outputs
    contact_count: wp.array(dtype=int),
    out_shape0: wp.array(dtype=int),
    out_shape1: wp.array(dtype=int),
    out_point0: wp.array(dtype=wp.vec3),
    out_point1: wp.array(dtype=wp.vec3),
    out_offset0: wp.array(dtype=wp.vec3),
    out_offset1: wp.array(dtype=wp.vec3),
    out_normal: wp.array(dtype=wp.vec3),
    out_thickness0: wp.array(dtype=float),
    out_thickness1: wp.array(dtype=float),
    out_tids: wp.array(dtype=int),
):
    tid = wp.tid()

    num_work_items = wp.min(shape_pairs.shape[0], sap_shape_pair_count[0])

    for t in range(tid, num_work_items, total_num_threads):
        # Get shape pair
        pair = shape_pairs[t]
        shape_a = pair[0]
        shape_b = pair[1]

        # Safety: ignore self-collision pairs
        if shape_a == shape_b:
            continue

        # Validate shape indices
        if shape_a < 0 or shape_b < 0:
            continue

        # Get shape types
        type_a = shape_type[shape_a]
        type_b = shape_type[shape_b]

        # Skip mesh collisions (not supported in this simplified version)
        if type_a == int(GeoType.MESH) or type_b == int(GeoType.MESH):
            continue

        # Sort shapes by type to ensure consistent collision handling order
        if type_a > type_b:
            # Swap shapes to maintain consistent ordering
            shape_a, shape_b = shape_b, shape_a
            type_a, type_b = type_b, type_a
            tmp = pair[0]
            pair[0] = pair[1]
            pair[1] = tmp

        # Extract shape data for both shapes
        pos_a, quat_a, shape_data_a, bsphere_center_a, bsphere_radius_a = extract_shape_data(
            shape_a,
            body_q,
            shape_transform,
            shape_body,
            shape_type,
            shape_scale,
            shape_collision_radius,
            shape_source_ptr,
        )

        pos_b, quat_b, shape_data_b, bsphere_center_b, bsphere_radius_b = extract_shape_data(
            shape_b,
            body_q,
            shape_transform,
            shape_body,
            shape_type,
            shape_scale,
            shape_collision_radius,
            shape_source_ptr,
        )

        # Get body indices
        rigid_a = shape_body[shape_a]
        rigid_b = shape_body[shape_b]

        # Get shape types from the extracted data
        type_a = shape_data_a.shape_type
        type_b = shape_data_b.shape_type

        # Check if shapes are infinite planes (scale.x == 0 and scale.y == 0)
        scale_a = shape_data_a.scale
        scale_b = shape_data_b.scale
        is_infinite_plane_a = (type_a == int(GeoType.PLANE)) and (scale_a[0] == 0.0 and scale_a[1] == 0.0)
        is_infinite_plane_b = (type_b == int(GeoType.PLANE)) and (scale_b[0] == 0.0 and scale_b[1] == 0.0)

        # Early bounding sphere check using extracted bounding spheres
        # Inflate by rigid_contact_margin only (no speculative expansion)
        radius_a = bsphere_radius_a + rigid_contact_margin
        radius_b = bsphere_radius_b + rigid_contact_margin
        center_distance = wp.length(bsphere_center_b - bsphere_center_a)

        # Early out if bounding spheres don't overlap
        # Skip this check if either shape is an infinite plane
        if not (is_infinite_plane_a or is_infinite_plane_b):
            if center_distance > (radius_a + radius_b):
                continue
        elif is_infinite_plane_a and is_infinite_plane_b:
            # Plane-plane collisions are not supported
            continue
        elif is_infinite_plane_a:
            # Check if shape B is close enough to the infinite plane A
            plane_normal = wp.quat_rotate(quat_a, wp.vec3(0.0, 0.0, 1.0))
            # Distance from shape B center to plane A
            dist_to_plane = wp.abs(wp.dot(pos_b - pos_a, plane_normal))
            if dist_to_plane > radius_b:
                continue
        elif is_infinite_plane_b:
            # Check if shape A is close enough to the infinite plane B
            plane_normal = wp.quat_rotate(quat_b, wp.vec3(0.0, 0.0, 1.0))
            # Distance from shape A center to plane B
            dist_to_plane = wp.abs(wp.dot(pos_a - pos_b, plane_normal))
            if dist_to_plane > radius_a:
                continue

        # Use the extracted orientations
        rot_a = quat_a
        rot_b = quat_b

        # World->body transforms for writing contact points
        X_bw_a = wp.transform_identity() if rigid_a == -1 else wp.transform_inverse(body_q[rigid_a])
        X_bw_b = wp.transform_identity() if rigid_b == -1 else wp.transform_inverse(body_q[rigid_b])

        # Use the extracted geometry data structures for convex collision detection
        geom_a = shape_data_a
        geom_b = shape_data_b

        # Convert infinite planes to cube proxies for GJK/MPR compatibility
        # Use the OTHER object's radius to properly size the cube
        # Only convert if it's an infinite plane (finite planes can be handled normally)
        pos_a_adjusted = pos_a
        if is_infinite_plane_a:
            # Position the cube based on the OTHER object's position (pos_b)
            geom_a, pos_a_adjusted = convert_infinite_plane_to_cube(geom_a, rot_a, pos_a, pos_b, radius_b)
            type_a = int(GeoType.BOX)

        pos_b_adjusted = pos_b
        if is_infinite_plane_b:
            # Position the cube based on the OTHER object's position (pos_a)
            geom_b, pos_b_adjusted = convert_infinite_plane_to_cube(geom_b, rot_b, pos_b, pos_a, radius_a)
            type_b = int(GeoType.BOX)

        data_provider = SupportMapDataProvider()

        radius_eff_a = float(0.0)
        radius_eff_b = float(0.0)

        small_radius = 0.0001

        # Special treatment for minkowski objects
        if type_a == int(GeoType.SPHERE) or type_a == int(GeoType.CAPSULE):
            radius_eff_a = geom_a.scale[0]
            geom_a.scale[0] = small_radius

        if type_b == int(GeoType.SPHERE) or type_b == int(GeoType.CAPSULE):
            radius_eff_b = geom_b.scale[0]
            geom_b.scale[0] = small_radius

        count, normal, signed_distances, points, _features = wp.static(solve_convex_multi_contact)(
            geom_a,
            geom_b,
            rot_a,
            rot_b,
            pos_a_adjusted,
            pos_b_adjusted,
            0.0,  # sum_of_contact_offsets - gap
            data_provider,
            rigid_contact_margin + radius_eff_a + radius_eff_b,
            type_a == int(GeoType.SPHERE) or type_b == int(GeoType.SPHERE),
        )

        # Special post processing for minkowski objects
        if type_a == int(GeoType.SPHERE) or type_a == int(GeoType.CAPSULE):
            for i in range(count):
                points[i] = points[i] + normal * (radius_eff_a * 0.5)
                signed_distances[i] -= radius_eff_a - small_radius
        if type_b == int(GeoType.SPHERE) or type_b == int(GeoType.CAPSULE):
            for i in range(count):
                points[i] = points[i] - normal * (radius_eff_b * 0.5)
                signed_distances[i] -= radius_eff_b - small_radius

        # Post-process for axial shapes (cylinder/cone) rolling on discrete surfaces
        is_discrete_a = is_discrete_shape(geom_a.shape_type)
        is_discrete_b = is_discrete_shape(geom_b.shape_type)
        is_axial_a = type_a == int(GeoType.CYLINDER) or type_a == int(GeoType.CONE)
        is_axial_b = type_b == int(GeoType.CYLINDER) or type_b == int(GeoType.CONE)

        if is_discrete_a and is_axial_b and count >= 3:
            # Post-process axial shape (B) rolling on discrete surface (A)
            shape_radius = geom_b.scale[0]  # radius for cylinder, base radius for cone
            shape_half_height = geom_b.scale[1]
            is_cone_b = type_b == int(GeoType.CONE)
            count, signed_distances, points = postprocess_axial_shape_discrete_contacts(
                points,
                normal,
                signed_distances,
                count,
                rot_b,
                shape_radius,
                shape_half_height,
                pos_b_adjusted,
                is_cone_b,
            )

        if is_discrete_b and is_axial_a and count >= 3:
            # Post-process axial shape (A) rolling on discrete surface (B)
            # Note: normal points from A to B, so we need to negate it for the shape processing
            shape_radius = geom_a.scale[0]  # radius for cylinder, base radius for cone
            shape_half_height = geom_a.scale[1]
            is_cone_a = type_a == int(GeoType.CONE)
            count, signed_distances, points = postprocess_axial_shape_discrete_contacts(
                points,
                -normal,
                signed_distances,
                count,
                rot_a,
                shape_radius,
                shape_half_height,
                pos_a_adjusted,
                is_cone_a,
            )

        for id in range(count):
            write_contact(
                points[id],
                normal,
                signed_distances[id],
                radius_eff_a,
                radius_eff_b,
                shape_thickness[shape_a],
                shape_thickness[shape_b],
                shape_a,
                shape_b,
                X_bw_a,
                X_bw_b,
                t,
                rigid_contact_margin,
                contact_max,
                contact_count,
                out_shape0,
                out_shape1,
                out_point0,
                out_point1,
                out_offset0,
                out_offset1,
                out_normal,
                out_thickness0,
                out_thickness1,
                out_tids,
            )


@wp.kernel
def compute_shape_aabbs(
    body_q: wp.array(dtype=wp.transform),
    shape_transform: wp.array(dtype=wp.transform),
    shape_body: wp.array(dtype=int),
    shape_type: wp.array(dtype=int),
    shape_scale: wp.array(dtype=wp.vec3),
    shape_collision_radius: wp.array(dtype=float),
    shape_source_ptr: wp.array(dtype=wp.uint64),
    rigid_contact_margin: float,
    # outputs
    aabb_lower: wp.array(dtype=wp.vec3),
    aabb_upper: wp.array(dtype=wp.vec3),
):
    """Compute axis-aligned bounding boxes for each shape in world space.

    Uses support function for most shapes. Infinite planes and meshes use bounding sphere fallback.
    AABBs are enlarged by rigid_contact_margin for contact detection margin.
    """
    shape_id = wp.tid()

    rigid_id = shape_body[shape_id]
    geo_type = shape_type[shape_id]

    # Compute world transform
    if rigid_id == -1:
        X_ws = shape_transform[shape_id]
    else:
        X_ws = wp.transform_multiply(body_q[rigid_id], shape_transform[shape_id])

    pos = wp.transform_get_translation(X_ws)
    orientation = wp.transform_get_rotation(X_ws)

    # Enlarge AABB by rigid_contact_margin for contact detection
    margin_vec = wp.vec3(rigid_contact_margin, rigid_contact_margin, rigid_contact_margin)

    # Check if this is an infinite plane or mesh - use bounding sphere fallback
    scale = shape_scale[shape_id]
    is_infinite_plane = (geo_type == int(GeoType.PLANE)) and (scale[0] == 0.0 and scale[1] == 0.0)
    is_mesh = geo_type == int(GeoType.MESH)

    if is_infinite_plane or is_mesh:
        # Use conservative bounding sphere approach for infinite planes and meshes
        radius = shape_collision_radius[shape_id]
        half_extents = wp.vec3(radius, radius, radius)
        aabb_lower[shape_id] = pos - half_extents - margin_vec
        aabb_upper[shape_id] = pos + half_extents + margin_vec
    else:
        # Use support function to compute tight AABB
        # Create generic shape data
        shape_data = GenericShapeData()
        shape_data.shape_type = geo_type
        shape_data.scale = scale
        shape_data.auxiliary = wp.vec3(0.0, 0.0, 0.0)

        # For CONVEX_MESH, pack the mesh pointer
        if geo_type == int(GeoType.CONVEX_MESH):
            shape_data.auxiliary = pack_mesh_ptr(shape_source_ptr[shape_id])

        data_provider = SupportMapDataProvider()

        # Transpose orientation matrix to transform world axes to local space
        # Convert quaternion to 3x3 rotation matrix and transpose (inverse rotation)
        rot_mat = wp.quat_to_matrix(orientation)
        rot_mat_t = wp.transpose(rot_mat)

        # Transform world axes to local space (multiply by transposed rotation = inverse rotation)
        local_x = wp.vec3(rot_mat_t[0, 0], rot_mat_t[1, 0], rot_mat_t[2, 0])
        local_y = wp.vec3(rot_mat_t[0, 1], rot_mat_t[1, 1], rot_mat_t[2, 1])
        local_z = wp.vec3(rot_mat_t[0, 2], rot_mat_t[1, 2], rot_mat_t[2, 2])

        # Compute AABB extents by evaluating support function in local space
        # Dot products are done in local space to avoid expensive rotations
        support_point = wp.vec3()

        # Max X: support along +local_x, dot in local space
        support_point, _feature_id = support_map(shape_data, local_x, data_provider)
        max_x = wp.dot(local_x, support_point)

        # Max Y: support along +local_y, dot in local space
        support_point, _feature_id = support_map(shape_data, local_y, data_provider)
        max_y = wp.dot(local_y, support_point)

        # Max Z: support along +local_z, dot in local space
        support_point, _feature_id = support_map(shape_data, local_z, data_provider)
        max_z = wp.dot(local_z, support_point)

        # Min X: support along -local_x, dot in local space
        support_point, _feature_id = support_map(shape_data, -local_x, data_provider)
        min_x = wp.dot(local_x, support_point)

        # Min Y: support along -local_y, dot in local space
        support_point, _feature_id = support_map(shape_data, -local_y, data_provider)
        min_y = wp.dot(local_y, support_point)

        # Min Z: support along -local_z, dot in local space
        support_point, _feature_id = support_map(shape_data, -local_z, data_provider)
        min_z = wp.dot(local_z, support_point)

        # AABB in world space (add world position to extents)
        aabb_min_world = wp.vec3(min_x, min_y, min_z) + pos
        aabb_max_world = wp.vec3(max_x, max_y, max_z) + pos

        aabb_lower[shape_id] = aabb_min_world - margin_vec
        aabb_upper[shape_id] = aabb_max_world + margin_vec


class CollisionPipelineUnified:
    """
    CollisionPipelineUnified manages collision detection and contact generation for a simulation.

    This class is responsible for allocating and managing buffers for collision detection,
    generating rigid and soft contacts between shapes and particles, and providing an interface
    for running the collision pipeline on a given simulation state.
    """

    def __init__(
        self,
        shape_count: int,
        particle_count: int,
        shape_pairs_filtered: wp.array(dtype=wp.vec2i) | None = None,
        rigid_contact_max: int | None = None,
        rigid_contact_max_per_pair: int = 10,
        rigid_contact_margin: float = 0.01,
        soft_contact_max: int | None = None,
        soft_contact_margin: float = 0.01,
        edge_sdf_iter: int = 10,
        iterate_mesh_vertices: bool = True,
        requires_grad: bool = False,
        device: Devicelike = None,
        broad_phase_mode: BroadPhaseMode = BroadPhaseMode.NXN,
    ):
        """
        Initialize the CollisionPipeline.

        Args:
            shape_count (int): Number of shapes in the simulation.
            particle_count (int): Number of particles in the simulation.
            shape_pairs_filtered (wp.array | None, optional): Precomputed shape pairs for EXPLICIT broad phase mode.
                Required when broad_phase_mode is BroadPhaseMode.EXPLICIT, ignored otherwise.
            rigid_contact_max (int | None, optional): Maximum number of rigid contacts to allocate.
                If None, computed as shape_pairs_max * rigid_contact_max_per_pair.
            rigid_contact_max_per_pair (int, optional): Maximum number of contact points per shape pair. Defaults to 10.
            rigid_contact_margin (float, optional): Margin for rigid contact generation. Defaults to 0.01.
            soft_contact_max (int | None, optional): Maximum number of soft contacts to allocate.
                If None, computed as shape_count * particle_count.
            soft_contact_margin (float, optional): Margin for soft contact generation. Defaults to 0.01.
            edge_sdf_iter (int, optional): Number of iterations for edge SDF collision. Defaults to 10.
            iterate_mesh_vertices (bool, optional): Whether to iterate mesh vertices for collision. Defaults to True.
            requires_grad (bool, optional): Whether to enable gradient computation. Defaults to False.
            device (Devicelike, optional): The device on which to allocate arrays and perform computation.
            broad_phase_mode (BroadPhaseMode, optional): Broad phase mode for collision detection.
                - BroadPhaseMode.NXN: Use all-pairs AABB broad phase (O(N²), good for small scenes)
                - BroadPhaseMode.SAP: Use sweep-and-prune AABB broad phase (O(N log N), better for larger scenes)
                - BroadPhaseMode.EXPLICIT: Use precomputed shape pairs (most efficient when pairs known)
                Defaults to BroadPhaseMode.NXN.
        """
        # will be allocated during collide
        self.contacts = None  # type: Contacts | None

        self.shape_count = shape_count
        self.broad_phase_mode = broad_phase_mode

        # Estimate based on worst case (all pairs)
        self.shape_pairs_max = (shape_count * (shape_count - 1)) // 2

        self.rigid_contact_margin = rigid_contact_margin
        if rigid_contact_max is not None:
            self.rigid_contact_max = rigid_contact_max
        else:
            self.rigid_contact_max = self.shape_pairs_max * rigid_contact_max_per_pair

        # Initialize broad phase based on mode
        if self.broad_phase_mode == BroadPhaseMode.NXN:
            raise NotImplementedError(
                "NxN broad phase mode is currently not supported due to open collision filtering issues"
            )
            self.nxn_broadphase = BroadPhaseAllPairs()
            self.sap_broadphase = None
            self.explicit_broadphase = None
            self.shape_pairs_filtered = None
        elif self.broad_phase_mode == BroadPhaseMode.SAP:
            raise NotImplementedError(
                "SAP broad phase mode is currently not supported due to open collision filtering issues"
            )
            # Estimate max groups for SAP - use reasonable defaults
            max_num_negative_group_members = max(int(shape_count**0.5), 10)
            max_num_distinct_positive_groups = max(int(shape_count**0.5), 10)
            self.sap_broadphase = BroadPhaseSAP(
                max_broad_phase_elements=shape_count,
                max_num_distinct_positive_groups=max_num_distinct_positive_groups,
                max_num_negative_group_members=max_num_negative_group_members,
            )
            self.nxn_broadphase = None
            self.explicit_broadphase = None
            self.shape_pairs_filtered = None
        else:  # BroadPhaseMode.EXPLICIT
            if shape_pairs_filtered is None:
                raise ValueError("shape_pairs_filtered must be provided when using BroadPhaseMode.EXPLICIT")
            self.explicit_broadphase = BroadPhaseExplicit()
            self.nxn_broadphase = None
            self.sap_broadphase = None
            self.shape_pairs_filtered = shape_pairs_filtered
            # Update shape_pairs_max to reflect actual precomputed pairs
            self.shape_pairs_max = len(shape_pairs_filtered)

        # Allocate buffers for broadphase collision handling
        with wp.ScopedDevice(device):
            self.rigid_pair_shape0 = wp.empty(self.rigid_contact_max, dtype=wp.int32)
            self.rigid_pair_shape1 = wp.empty(self.rigid_contact_max, dtype=wp.int32)
            self.rigid_pair_point_limit = None  # wp.empty(self.shape_count ** 2, dtype=wp.int32)
            self.rigid_pair_point_count = None  # wp.empty(self.shape_count ** 2, dtype=wp.int32)
            self.rigid_pair_point_id = wp.empty(self.rigid_contact_max, dtype=wp.int32)

            # Allocate buffers for dynamically computed pairs and AABBs
            self.broad_phase_pair_count = wp.zeros(1, dtype=wp.int32, device=device)
            self.broad_phase_shape_pairs = wp.zeros(self.shape_pairs_max, dtype=wp.vec2i, device=device)
            self.shape_aabb_lower = wp.zeros(shape_count, dtype=wp.vec3, device=device)
            self.shape_aabb_upper = wp.zeros(shape_count, dtype=wp.vec3, device=device)
            # Allocate dummy collision/shape group arrays once (reused every frame)
            # Collision group: 1 (positive) = all shapes in same group, collide with each other
            # Environment group: -1 (global) = collides with all environments
            # self.dummy_collision_group = wp.full(shape_count, 1, dtype=wp.int32, device=device)
            # self.dummy_shape_group = wp.full(shape_count, -1, dtype=wp.int32, device=device)

        if soft_contact_max is None:
            soft_contact_max = shape_count * particle_count
        self.soft_contact_margin = soft_contact_margin
        self.soft_contact_max = soft_contact_max

        self.iterate_mesh_vertices = iterate_mesh_vertices
        self.requires_grad = requires_grad
        self.edge_sdf_iter = edge_sdf_iter

    @classmethod
    def from_model(
        cls,
        model: Model,
        rigid_contact_max_per_pair: int | None = None,
        rigid_contact_margin: float = 0.01,
        soft_contact_max: int | None = None,
        soft_contact_margin: float = 0.01,
        edge_sdf_iter: int = 10,
        iterate_mesh_vertices: bool = True,
        requires_grad: bool | None = None,
        broad_phase_mode: BroadPhaseMode = BroadPhaseMode.NXN,
        shape_pairs_filtered: wp.array(dtype=wp.vec2i) | None = None,
    ) -> CollisionPipelineUnified:
        """
        Create a CollisionPipelineUnified instance from a Model.

        Args:
            model (Model): The simulation model.
            rigid_contact_max_per_pair (int | None, optional): Maximum number of contact points per shape pair.
                If None, uses model.rigid_contact_max and sets per-pair to 0.
            rigid_contact_margin (float, optional): Margin for rigid contact generation. Defaults to 0.01.
            soft_contact_max (int | None, optional): Maximum number of soft contacts to allocate.
            soft_contact_margin (float, optional): Margin for soft contact generation. Defaults to 0.01.
            edge_sdf_iter (int, optional): Number of iterations for edge SDF collision. Defaults to 10.
            iterate_mesh_vertices (bool, optional): Whether to iterate mesh vertices for collision. Defaults to True.
            requires_grad (bool | None, optional): Whether to enable gradient computation. If None, uses model.requires_grad.
            broad_phase_mode (BroadPhaseMode, optional): Broad phase collision detection mode. Defaults to BroadPhaseMode.NXN.
            shape_pairs_filtered (wp.array | None, optional): Precomputed shape pairs for EXPLICIT mode.
                Required when broad_phase_mode is BroadPhaseMode.EXPLICIT. For NXN/SAP modes, can use model.shape_contact_pairs if available.

        Returns:
            CollisionPipeline: The constructed collision pipeline.
        """
        rigid_contact_max = None
        if rigid_contact_max_per_pair is None:
            rigid_contact_max = model.rigid_contact_max
            rigid_contact_max_per_pair = 0
        if requires_grad is None:
            requires_grad = model.requires_grad

        # For EXPLICIT mode, use provided shape_pairs_filtered or fall back to model pairs
        # For NXN/SAP modes, shape_pairs_filtered is not used (but can be provided for EXPLICIT)
        if shape_pairs_filtered is None and broad_phase_mode == BroadPhaseMode.EXPLICIT:
            # Try to use model.shape_contact_pairs if available
            if hasattr(model, "shape_contact_pairs") and model.shape_contact_pairs is not None:
                shape_pairs_filtered = model.shape_contact_pairs
            else:
                # Will raise error in __init__ if EXPLICIT mode requires it
                shape_pairs_filtered = None

        return CollisionPipelineUnified(
            model.shape_count,
            model.particle_count,
            shape_pairs_filtered,
            rigid_contact_max,
            rigid_contact_max_per_pair,
            rigid_contact_margin,
            soft_contact_max,
            soft_contact_margin,
            edge_sdf_iter,
            iterate_mesh_vertices,
            requires_grad,
            model.device,
            broad_phase_mode,
        )

    def collide(self, model: Model, state: State) -> Contacts:
        """
        Run the collision pipeline for the given model and state, generating contacts.

        This method allocates or clears the contact buffer as needed, then generates
        soft and rigid contacts using the current simulation state. If using SAP broad phase,
        potentially colliding pairs are computed dynamically based on bounding sphere overlaps.

        Args:
            model (Model): The simulation model.
            state (State): The current simulation state.

        Returns:
            Contacts: The generated contacts for the current state.
        """
        # Allocate new contact memory for contacts if needed (e.g., for gradients)
        if self.contacts is None or self.requires_grad:
            self.contacts = Contacts(
                self.rigid_contact_max,
                self.soft_contact_max,
                requires_grad=self.requires_grad,
                device=model.device,
            )
        else:
            self.contacts.clear()

        # output contacts buffer
        contacts = self.contacts

        # Clear counters at start of frame
        self.broad_phase_pair_count.zero_()

        # Compute AABBs for all shapes in world space (needed for both NXN and SAP)
        # AABBs are computed using support function for most shapes
        wp.launch(
            kernel=compute_shape_aabbs,
            dim=model.shape_count,
            inputs=[
                state.body_q,
                model.shape_transform,
                model.shape_body,
                model.shape_type,
                model.shape_scale,
                model.shape_collision_radius,
                model.shape_source_ptr,
                self.rigid_contact_margin,
            ],
            outputs=[
                self.shape_aabb_lower,
                self.shape_aabb_upper,
            ],
            device=model.device,
        )

        # debug_a = model.shape_collision_group.numpy()
        # debug_b = model.shape_world.numpy()

        # Run appropriate broad phase
        if self.broad_phase_mode == BroadPhaseMode.NXN:
            # Use NxN all-pairs broad phase with AABB overlaps
            self.nxn_broadphase.launch(
                self.shape_aabb_lower,
                self.shape_aabb_upper,
                model.shape_thickness,  # Use thickness as cutoff
                model.shape_collision_group,  # Collision groups for filtering
                model.shape_world,  # Environment groups for filtering
                model.shape_count,
                self.broad_phase_shape_pairs,
                self.broad_phase_pair_count,
                device=model.device,
            )
        elif self.broad_phase_mode == BroadPhaseMode.SAP:
            # Use sweep-and-prune broad phase with AABB sorting
            self.sap_broadphase.launch(
                self.shape_aabb_lower,
                self.shape_aabb_upper,
                model.shape_thickness,  # Use thickness as cutoff
                model.shape_collision_group,  # Collision groups for filtering
                model.shape_world,  # Environment groups for filtering
                model.shape_count,
                self.broad_phase_shape_pairs,
                self.broad_phase_pair_count,
                device=model.device,
            )
        else:  # BroadPhaseMode.EXPLICIT
            # Use explicit precomputed pairs broad phase with AABB filtering
            self.explicit_broadphase.launch(
                self.shape_aabb_lower,
                self.shape_aabb_upper,
                model.shape_thickness,  # Use thickness as cutoff
                self.shape_pairs_filtered,
                len(self.shape_pairs_filtered),
                self.broad_phase_shape_pairs,
                self.broad_phase_pair_count,
                device=model.device,
            )

        # debug_c = self.broad_phase_shape_pairs.numpy()
        # debug_d = self.broad_phase_pair_count.numpy()

        # # Get the number of pairs
        # num_pairs = debug_d[0]

        # # Create array to store pairs containing last shape
        # last_shape_idx = model.shape_count - 1
        # last_shape_pairs = []

        # # Go through all pairs looking for last shape index
        # for i in range(num_pairs):
        #     pair = debug_c[i]
        #     if pair[0] == last_shape_idx or pair[1] == last_shape_idx:
        #         last_shape_pairs.append(pair)

        # Launch kernel across all shape pairs
        # Use fixed dimension as we don't know pair count ahead of time
        block_dim = 128
        total_num_threads = block_dim * 32
        wp.launch(
            kernel=build_contacts_kernel_gjk_mpr,
            dim=total_num_threads,
            inputs=[
                state.body_q,
                state.body_qd,
                model.shape_transform,
                model.shape_body,
                model.shape_type,
                model.shape_scale,
                model.shape_thickness,
                model.shape_collision_radius,
                self.broad_phase_shape_pairs,
                0.0,  # sum_of_contact_offsets
                self.rigid_contact_margin,
                contacts.rigid_contact_max,
                self.broad_phase_pair_count,
                model.shape_source_ptr,
                total_num_threads,
            ],
            outputs=[
                contacts.rigid_contact_count,
                contacts.rigid_contact_shape0,
                contacts.rigid_contact_shape1,
                contacts.rigid_contact_point0,
                contacts.rigid_contact_point1,
                contacts.rigid_contact_offset0,
                contacts.rigid_contact_offset1,
                contacts.rigid_contact_normal,
                contacts.rigid_contact_thickness0,
                contacts.rigid_contact_thickness1,
                contacts.rigid_contact_tids,
            ],
            device=contacts.device,
            block_dim=block_dim,
        )

        return contacts
