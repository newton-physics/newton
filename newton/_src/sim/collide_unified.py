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
from ..geometry.broad_phase_common import binary_search
from ..geometry.broad_phase_nxn import BroadPhaseAllPairs, BroadPhaseExplicit
from ..geometry.broad_phase_sap import BroadPhaseSAP
from ..geometry.collision_convex import create_solve_convex_multi_contact, create_solve_convex_single_contact
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

ENABLE_MULTI_CONTACT = True
ENABLE_TILE_BVH_QUERY = False


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
solve_convex_single_contact = create_solve_convex_single_contact(support_map_func)

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
    - The cube is positioned with its top face at the plane surface
    - The cube's lateral dimensions are sized based on the other object's bounding sphere
    - The cube extends only 'downward' from the plane (half-space in -Z direction in plane's local frame)

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
    # Half-space behavior: cube extends only below the plane surface (negative Z)
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
        shape_source_ptr: Array of mesh/SDF source pointers

    Returns:
        tuple: (position, orientation, shape_data)
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

    return position, orientation, result


@wp.func
def compute_gjk_mpr_contacts(
    geom_a: GenericShapeData,
    geom_b: GenericShapeData,
    type_a: int,
    type_b: int,
    rot_a: wp.quat,
    rot_b: wp.quat,
    pos_a_adjusted: wp.vec3,
    pos_b_adjusted: wp.vec3,
    rigid_contact_margin: float,
):
    """
    Compute contacts between two shapes using GJK/MPR algorithm.

    Returns:
        Tuple of (count, normal, signed_distances, points, radius_eff_a, radius_eff_b)
    """
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

    if wp.static(ENABLE_MULTI_CONTACT):
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
    else:
        count, normal, signed_distances, points, _features = wp.static(solve_convex_single_contact)(
            geom_a,
            geom_b,
            rot_a,
            rot_b,
            pos_a_adjusted,
            pos_b_adjusted,
            0.0,  # sum_of_contact_offsets - gap
            data_provider,
            rigid_contact_margin + radius_eff_a + radius_eff_b,
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

    if wp.static(ENABLE_MULTI_CONTACT):
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

    return count, normal, signed_distances, points, radius_eff_a, radius_eff_b


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
def compute_tight_aabb_from_support(
    shape_data: GenericShapeData,
    orientation: wp.quat,
    center_pos: wp.vec3,
    data_provider: SupportMapDataProvider,
) -> tuple[wp.vec3, wp.vec3]:
    """
    Compute tight AABB for a shape using support function.

    Args:
        shape_data: Generic shape data
        orientation: Shape orientation (quaternion)
        center_pos: Center position of the shape
        data_provider: Support map data provider

    Returns:
        Tuple of (aabb_min, aabb_max) in world space
    """
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
    aabb_min = wp.vec3(min_x, min_y, min_z) + center_pos
    aabb_max = wp.vec3(max_x, max_y, max_z) + center_pos

    return aabb_min, aabb_max


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


@wp.func
def compute_bounding_sphere_from_aabb(aabb_lower: wp.vec3, aabb_upper: wp.vec3) -> tuple[wp.vec3, float]:
    """
    Compute a bounding sphere from an AABB.

    Returns:
        Tuple of (center, radius) where center is the AABB center and radius is half the diagonal.
    """
    center = 0.5 * (aabb_lower + aabb_upper)
    half_extents = 0.5 * (aabb_upper - aabb_lower)
    radius = wp.length(half_extents)
    return center, radius


@wp.func
def check_infinite_plane_bsphere_overlap(
    shape_data_a: GenericShapeData,
    shape_data_b: GenericShapeData,
    pos_a: wp.vec3,
    pos_b: wp.vec3,
    quat_a: wp.quat,
    quat_b: wp.quat,
    bsphere_center_a: wp.vec3,
    bsphere_center_b: wp.vec3,
    bsphere_radius_a: float,
    bsphere_radius_b: float,
) -> bool:
    """
    Check if an infinite plane overlaps with another shape's bounding sphere.
    Treats the plane as a half-space: objects on or below the plane (negative side of the normal)
    are considered to overlap and will generate contacts.
    Returns True if they overlap, False otherwise.
    Uses data already extracted by extract_shape_data.
    """
    type_a = shape_data_a.shape_type
    type_b = shape_data_b.shape_type
    scale_a = shape_data_a.scale
    scale_b = shape_data_b.scale

    # Check if either shape is an infinite plane
    is_infinite_plane_a = (type_a == int(GeoType.PLANE)) and (scale_a[0] == 0.0 and scale_a[1] == 0.0)
    is_infinite_plane_b = (type_b == int(GeoType.PLANE)) and (scale_b[0] == 0.0 and scale_b[1] == 0.0)

    # If neither is an infinite plane, return True (no culling)
    if not (is_infinite_plane_a or is_infinite_plane_b):
        return True

    # Determine which is the plane and which is the other shape
    if is_infinite_plane_a:
        plane_pos = pos_a
        plane_quat = quat_a
        other_center = bsphere_center_b
        other_radius = bsphere_radius_b
    else:
        plane_pos = pos_b
        plane_quat = quat_b
        other_center = bsphere_center_a
        other_radius = bsphere_radius_a

    # Compute plane normal (plane's local +Z axis in world space)
    plane_normal = wp.quat_rotate(plane_quat, wp.vec3(0.0, 0.0, 1.0))

    # Distance from sphere center to plane (positive = above plane, negative = below plane)
    center_dist = wp.dot(other_center - plane_pos, plane_normal)

    # Treat plane as a half-space: objects on or below the plane (negative side) generate contacts
    # Remove absolute value to only check penetration side
    return center_dist <= other_radius


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
    shape_aabb_lower: wp.array(dtype=wp.vec3),
    shape_aabb_upper: wp.array(dtype=wp.vec3),
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
    # mesh collision outputs
    shape_pairs_mesh: wp.array(dtype=wp.vec2i),
    shape_pairs_mesh_count: wp.array(dtype=int),
    # mesh-plane collision outputs
    shape_pairs_mesh_plane: wp.array(dtype=wp.vec2i),
    shape_pairs_mesh_plane_cumsum: wp.array(dtype=int),
    shape_pairs_mesh_plane_count: wp.array(dtype=int),
    mesh_plane_vertex_total_count: wp.array(dtype=int),
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

        # Load AABBs once for both shapes
        aabb_a_lower = shape_aabb_lower[shape_a]
        aabb_a_upper = shape_aabb_upper[shape_a]
        aabb_b_lower = shape_aabb_lower[shape_b]
        aabb_b_upper = shape_aabb_upper[shape_b]

        # Inline AABB overlap check - skip pairs that don't overlap
        # Especially useful for the EXPLICIT broad phase mode
        if aabb_a_upper[0] < aabb_b_lower[0] or aabb_b_upper[0] < aabb_a_lower[0]:
            continue
        if aabb_a_upper[1] < aabb_b_lower[1] or aabb_b_upper[1] < aabb_a_lower[1]:
            continue
        if aabb_a_upper[2] < aabb_b_lower[2] or aabb_b_upper[2] < aabb_a_lower[2]:
            continue

        # Get shape types
        type_a = shape_type[shape_a]
        type_b = shape_type[shape_b]

        # Sort shapes by type to ensure consistent collision handling order
        if type_a > type_b:
            # Swap shapes to maintain consistent ordering
            shape_a, shape_b = shape_b, shape_a
            type_a, type_b = type_b, type_a
            # Swap AABBs as well
            aabb_a_lower, aabb_b_lower = aabb_b_lower, aabb_a_lower
            aabb_a_upper, aabb_b_upper = aabb_b_upper, aabb_a_upper
            tmp = pair[0]
            pair[0] = pair[1]
            pair[1] = tmp

        # Extract shape data for both shapes
        pos_a, quat_a, shape_data_a = extract_shape_data(
            shape_a,
            body_q,
            shape_transform,
            shape_body,
            shape_type,
            shape_scale,
            shape_source_ptr,
        )

        pos_b, quat_b, shape_data_b = extract_shape_data(
            shape_b,
            body_q,
            shape_transform,
            shape_body,
            shape_type,
            shape_scale,
            shape_source_ptr,
        )

        # Check if shapes are infinite planes (scale.x == 0 and scale.y == 0)
        is_infinite_plane_a = (type_a == int(GeoType.PLANE)) and (
            shape_data_a.scale[0] == 0.0 and shape_data_a.scale[1] == 0.0
        )
        is_infinite_plane_b = (type_b == int(GeoType.PLANE)) and (
            shape_data_b.scale[0] == 0.0 and shape_data_b.scale[1] == 0.0
        )

        if is_infinite_plane_a and is_infinite_plane_b:
            continue

        # Compute bounding spheres from AABBs instead of using mesh bounding spheres
        bsphere_center_a, bsphere_radius_a = compute_bounding_sphere_from_aabb(aabb_a_lower, aabb_a_upper)
        bsphere_center_b, bsphere_radius_b = compute_bounding_sphere_from_aabb(aabb_b_lower, aabb_b_upper)

        # Check if infinite plane vs bounding sphere overlap - early rejection
        if not check_infinite_plane_bsphere_overlap(
            shape_data_a,
            shape_data_b,
            pos_a,
            pos_b,
            quat_a,
            quat_b,
            bsphere_center_a,
            bsphere_center_b,
            bsphere_radius_a,
            bsphere_radius_b,
        ):
            continue

        # Check for mesh vs infinite plane collision - special handling
        # After sorting, type_a <= type_b, so we only need to check one direction
        if type_a == int(GeoType.PLANE) and type_b == int(GeoType.MESH):
            # Check if plane is infinite (scale x and y are zero)
            scale_a = shape_scale[shape_a]
            if scale_a[0] == 0.0 and scale_a[1] == 0.0:
                # Get mesh vertex count
                mesh_id = shape_source_ptr[shape_b]
                if mesh_id != wp.uint64(0):
                    mesh_obj = wp.mesh_get(mesh_id)
                    vertex_count = mesh_obj.points.shape[0]

                    # Add to mesh-plane collision buffer with cumulative vertex count
                    mesh_plane_idx = wp.atomic_add(shape_pairs_mesh_plane_count, 0, 1)
                    if mesh_plane_idx < shape_pairs_mesh_plane.shape[0]:
                        # Store shape indices (mesh, plane)
                        shape_pairs_mesh_plane[mesh_plane_idx] = wp.vec2i(shape_b, shape_a)
                        # Store inclusive cumulative vertex count in separate array for better cache locality
                        cumulative_count_before = wp.atomic_add(mesh_plane_vertex_total_count, 0, vertex_count)
                        cumulative_count_inclusive = cumulative_count_before + vertex_count
                        shape_pairs_mesh_plane_cumsum[mesh_plane_idx] = cumulative_count_inclusive
                continue

        # Check for other mesh collisions - add to separate buffer for specialized handling
        if type_a == int(GeoType.MESH) or type_b == int(GeoType.MESH):
            # Add to mesh collision buffer using atomic counter
            mesh_pair_idx = wp.atomic_add(shape_pairs_mesh_count, 0, 1)
            if mesh_pair_idx < shape_pairs_mesh.shape[0]:
                shape_pairs_mesh[mesh_pair_idx] = pair
            continue

        # Get body indices
        rigid_a = shape_body[shape_a]
        rigid_b = shape_body[shape_b]

        # World->body transforms for writing contact points
        X_bw_a = wp.transform_identity() if rigid_a == -1 else wp.transform_inverse(body_q[rigid_a])
        X_bw_b = wp.transform_identity() if rigid_b == -1 else wp.transform_inverse(body_q[rigid_b])

        # Convert infinite planes to cube proxies for GJK/MPR compatibility
        # Use the OTHER object's radius to properly size the cube
        # Only convert if it's an infinite plane (finite planes can be handled normally)
        pos_a_adjusted = pos_a
        if is_infinite_plane_a:
            # Position the cube based on the OTHER object's position (pos_b)
            shape_data_a, pos_a_adjusted = convert_infinite_plane_to_cube(
                shape_data_a, quat_a, pos_a, pos_b, bsphere_radius_b + rigid_contact_margin
            )
            type_a = int(GeoType.BOX)

        pos_b_adjusted = pos_b
        if is_infinite_plane_b:
            # Position the cube based on the OTHER object's position (pos_a)
            shape_data_b, pos_b_adjusted = convert_infinite_plane_to_cube(
                shape_data_b, quat_b, pos_b, pos_a, bsphere_radius_a + rigid_contact_margin
            )
            type_b = int(GeoType.BOX)

        # Compute contacts using GJK/MPR
        count, normal, signed_distances, points, radius_eff_a, radius_eff_b = compute_gjk_mpr_contacts(
            shape_data_a,
            shape_data_b,
            type_a,
            type_b,
            quat_a,
            quat_b,
            pos_a_adjusted,
            pos_b_adjusted,
            rigid_contact_margin,
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


@wp.kernel(enable_backward=False)
def find_mesh_triangle_overlaps_kernel(
    body_q: wp.array(dtype=wp.transform),
    shape_transform: wp.array(dtype=wp.transform),
    shape_body: wp.array(dtype=int),
    shape_type: wp.array(dtype=int),
    shape_scale: wp.array(dtype=wp.vec3),
    shape_collision_radius: wp.array(dtype=float),
    shape_source_ptr: wp.array(dtype=wp.uint64),
    shape_pairs_mesh: wp.array(dtype=wp.vec2i),
    shape_pairs_mesh_count: wp.array(dtype=int),
    rigid_contact_margin: float,
    total_num_threads: int,
    # outputs
    triangle_pairs: wp.array(dtype=wp.vec3i),  # (shape_a, shape_b, triangle_idx)
    triangle_pairs_count: wp.array(dtype=int),
):
    """
    For each mesh collision pair, find all triangles that overlap with the non-mesh shape's AABB.
    Outputs triples of (shape_a, shape_b, triangle_idx) for further processing.
    Uses tiled mesh query for improved performance.
    """
    tid, _j = wp.tid()

    num_mesh_pairs = shape_pairs_mesh_count[0]

    # Strided loop over mesh pairs
    for i in range(tid, num_mesh_pairs, total_num_threads):
        pair = shape_pairs_mesh[i]
        shape_a = pair[0]
        shape_b = pair[1]

        # Determine which shape is the mesh
        type_a = shape_type[shape_a]
        type_b = shape_type[shape_b]

        mesh_shape = -1
        non_mesh_shape = -1

        if type_a == int(GeoType.MESH) and type_b != int(GeoType.MESH):
            mesh_shape = shape_a
            non_mesh_shape = shape_b
        elif type_b == int(GeoType.MESH) and type_a != int(GeoType.MESH):
            mesh_shape = shape_b
            non_mesh_shape = shape_a
        else:
            # Mesh-mesh collision not supported yet
            return

        # Get mesh BVH ID and mesh transform
        mesh_id = shape_source_ptr[mesh_shape]
        if mesh_id == wp.uint64(0):
            return

        # Get mesh world transform
        mesh_body_idx = shape_body[mesh_shape]
        X_mesh_ws = shape_transform[mesh_shape]
        if mesh_body_idx >= 0:
            X_mesh_ws = wp.transform_multiply(body_q[mesh_body_idx], shape_transform[mesh_shape])

        # Get inverse mesh transform (world to mesh local space)
        X_mesh_sw = wp.transform_inverse(X_mesh_ws)

        # Get non-mesh shape world transform
        body_idx = shape_body[non_mesh_shape]
        X_ws = shape_transform[non_mesh_shape]
        if body_idx >= 0:
            X_ws = wp.transform_multiply(body_q[body_idx], shape_transform[non_mesh_shape])

        # Compute transform from non-mesh shape local space to mesh local space
        # X_mesh_shape = X_mesh_sw * X_ws
        X_mesh_shape = wp.transform_multiply(X_mesh_sw, X_ws)
        pos_in_mesh = wp.transform_get_translation(X_mesh_shape)
        orientation_in_mesh = wp.transform_get_rotation(X_mesh_shape)

        # Create generic shape data for non-mesh shape
        geo_type = shape_type[non_mesh_shape]
        scale = shape_scale[non_mesh_shape]

        shape_data = GenericShapeData()
        shape_data.shape_type = geo_type
        shape_data.scale = scale
        shape_data.auxiliary = wp.vec3(0.0, 0.0, 0.0)

        # For CONVEX_MESH, pack the mesh pointer
        if geo_type == int(GeoType.CONVEX_MESH):
            shape_data.auxiliary = pack_mesh_ptr(shape_source_ptr[non_mesh_shape])

        data_provider = SupportMapDataProvider()

        # Compute tight AABB directly in mesh local space for optimal fit
        aabb_lower, aabb_upper = compute_tight_aabb_from_support(
            shape_data, orientation_in_mesh, pos_in_mesh, data_provider
        )

        # Add small margin for contact detection
        margin_vec = wp.vec3(rigid_contact_margin, rigid_contact_margin, rigid_contact_margin)
        aabb_lower = aabb_lower - margin_vec
        aabb_upper = aabb_upper + margin_vec

        if wp.static(ENABLE_TILE_BVH_QUERY):
            # Query mesh BVH for overlapping triangles in mesh local space using tiled version
            query = wp.tile_mesh_query_aabb(mesh_id, aabb_lower, aabb_upper)

            result_tile = wp.tile_mesh_query_aabb_next(query)

            # Continue querying while we have results
            # Each iteration, each thread in the block gets one result (or -1)
            while wp.tile_max(result_tile)[0] >= 0:
                # Each thread processes its result from the tile
                tri_index = wp.untile(result_tile)

                # Add this triangle pair to the output buffer if valid
                # Store (mesh_shape, non_mesh_shape, tri_index) to guarantee mesh is always first
                if tri_index >= 0:
                    out_idx = wp.atomic_add(triangle_pairs_count, 0, 1)
                    if out_idx < triangle_pairs.shape[0]:
                        triangle_pairs[out_idx] = wp.vec3i(mesh_shape, non_mesh_shape, tri_index)

                result_tile = wp.tile_mesh_query_aabb_next(query)
        else:
            query = wp.mesh_query_aabb(mesh_id, aabb_lower, aabb_upper)
            tri_index = wp.int32(0)
            while wp.mesh_query_aabb_next(query, tri_index):
                # Add this triangle pair to the output buffer if valid
                # Store (mesh_shape, non_mesh_shape, tri_index) to guarantee mesh is always first
                if tri_index >= 0:
                    out_idx = wp.atomic_add(triangle_pairs_count, 0, 1)
                    if out_idx < triangle_pairs.shape[0]:
                        triangle_pairs[out_idx] = wp.vec3i(mesh_shape, non_mesh_shape, tri_index)


@wp.func
def find_pair_from_cumulative_index(
    global_idx: int,
    cumulative_sums: wp.array(dtype=int),
    num_pairs: int,
) -> tuple[int, int]:
    """
    Binary search to find which pair a global index belongs to.

    Args:
        global_idx: Global index to search for
        cumulative_sums: Array of inclusive cumulative sums (end indices)
        num_pairs: Number of pairs

    Returns:
        Tuple of (pair_index, local_index_within_pair)
    """
    # Use binary_search to find first index where cumulative_sums[i] > global_idx
    # This gives us the bucket that contains global_idx
    pair_idx = binary_search(cumulative_sums, global_idx, 0, num_pairs)

    # Get cumulative start for this pair to calculate local index
    cumulative_start = int(0)
    if pair_idx > 0:
        cumulative_start = int(cumulative_sums[pair_idx - 1])

    local_idx = global_idx - cumulative_start

    return pair_idx, local_idx


@wp.kernel(enable_backward=False)
def process_mesh_triangle_contacts_kernel(
    body_q: wp.array(dtype=wp.transform),
    shape_transform: wp.array(dtype=wp.transform),
    shape_body: wp.array(dtype=int),
    shape_type: wp.array(dtype=int),
    shape_scale: wp.array(dtype=wp.vec3),
    shape_collision_radius: wp.array(dtype=float),
    shape_thickness: wp.array(dtype=float),
    shape_source_ptr: wp.array(dtype=wp.uint64),
    triangle_pairs: wp.array(dtype=wp.vec3i),
    triangle_pairs_count: wp.array(dtype=int),
    rigid_contact_margin: float,
    contact_max: int,
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
    """
    Process triangle pairs to generate contacts using GJK/MPR.
    """
    tid = wp.tid()

    num_triangle_pairs = triangle_pairs_count[0]

    for i in range(tid, num_triangle_pairs, total_num_threads):
        if i >= triangle_pairs.shape[0]:
            break

        triple = triangle_pairs[i]
        shape_a = triple[0]
        shape_b = triple[1]
        tri_idx = triple[2]

        # Get mesh data for shape A
        mesh_id_a = shape_source_ptr[shape_a]
        if mesh_id_a == wp.uint64(0):
            continue

        mesh_a = wp.mesh_get(mesh_id_a)
        mesh_scale_a = shape_scale[shape_a]

        # Get mesh world transform for shape A
        mesh_body_idx_a = shape_body[shape_a]
        X_mesh_ws_a = shape_transform[shape_a]
        if mesh_body_idx_a >= 0:
            X_mesh_ws_a = wp.transform_multiply(body_q[mesh_body_idx_a], shape_transform[shape_a])

        # Extract triangle vertices from mesh (indices are stored as flat array: i0, i1, i2, i0, i1, i2, ...)
        idx0 = mesh_a.indices[tri_idx * 3 + 0]
        idx1 = mesh_a.indices[tri_idx * 3 + 1]
        idx2 = mesh_a.indices[tri_idx * 3 + 2]

        # Get vertex positions in mesh local space (with scale applied)
        v0_local = wp.cw_mul(mesh_a.points[idx0], mesh_scale_a)
        v1_local = wp.cw_mul(mesh_a.points[idx1], mesh_scale_a)
        v2_local = wp.cw_mul(mesh_a.points[idx2], mesh_scale_a)

        # Transform vertices to world space
        v0_world = wp.transform_point(X_mesh_ws_a, v0_local)
        v1_world = wp.transform_point(X_mesh_ws_a, v1_local)
        v2_world = wp.transform_point(X_mesh_ws_a, v2_local)

        # Compute triangle centroid as position for shape A
        pos_a = (v0_world + v1_world + v2_world) / 3.0
        quat_a = wp.quat_identity()  # Triangle has no orientation, use identity

        # Create triangle shape data: vertex A at origin, B-A in scale, C-A in auxiliary
        shape_data_a = GenericShapeData()
        shape_data_a.shape_type = int(GeoTypeEx.TRIANGLE)
        shape_data_a.scale = v1_world - v0_world  # B - A
        shape_data_a.auxiliary = v2_world - v0_world  # C - A

        # Override pos_a to be vertex A (origin of triangle in local frame)
        pos_a = v0_world

        # Extract shape B data
        pos_b, quat_b, shape_data_b = extract_shape_data(
            shape_b,
            body_q,
            shape_transform,
            shape_body,
            shape_type,
            shape_scale,
            shape_source_ptr,
        )

        # Get body inverse transforms for contact point conversion
        mesh_body_idx_a = shape_body[shape_a]
        X_bw_a = wp.transform_identity()
        if mesh_body_idx_a >= 0:
            X_bw_a = wp.transform_inverse(body_q[mesh_body_idx_a])

        body_idx_b = shape_body[shape_b]
        X_bw_b = wp.transform_identity()
        if body_idx_b >= 0:
            X_bw_b = wp.transform_inverse(body_q[body_idx_b])

        # Compute contacts using GJK/MPR
        type_a = int(GeoTypeEx.TRIANGLE)
        type_b = shape_type[shape_b]

        count, normal, signed_distances, points, radius_eff_a, radius_eff_b = compute_gjk_mpr_contacts(
            shape_data_a,
            shape_data_b,
            type_a,
            type_b,
            quat_a,
            quat_b,
            pos_a,
            pos_b,
            rigid_contact_margin,
        )

        # Write contacts
        for contact_id in range(count):
            write_contact(
                points[contact_id],
                normal,
                signed_distances[contact_id],
                radius_eff_a,
                radius_eff_b,
                shape_thickness[shape_a],
                shape_thickness[shape_b],
                shape_a,
                shape_b,
                X_bw_a,
                X_bw_b,
                tid,
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


@wp.kernel(enable_backward=False)
def process_mesh_plane_contacts_kernel(
    body_q: wp.array(dtype=wp.transform),
    shape_transform: wp.array(dtype=wp.transform),
    shape_body: wp.array(dtype=int),
    shape_type: wp.array(dtype=int),
    shape_scale: wp.array(dtype=wp.vec3),
    shape_thickness: wp.array(dtype=float),
    shape_source_ptr: wp.array(dtype=wp.uint64),
    shape_pairs_mesh_plane: wp.array(dtype=wp.vec2i),
    shape_pairs_mesh_plane_cumsum: wp.array(dtype=int),
    shape_pairs_mesh_plane_count: wp.array(dtype=int),
    mesh_plane_vertex_total_count: wp.array(dtype=int),
    rigid_contact_margin: float,
    contact_max: int,
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
    """
    Process mesh-plane collisions by checking each mesh vertex against the infinite plane.
    Uses binary search to map thread index to (mesh-plane pair, vertex index).
    Fixed thread count with strided loop over vertices.
    """
    tid = wp.tid()

    total_vertices = mesh_plane_vertex_total_count[0]
    num_pairs = shape_pairs_mesh_plane_count[0]

    if num_pairs == 0:
        return

    # Process vertices in a strided loop
    for task_id in range(tid, total_vertices, total_num_threads):
        if task_id >= total_vertices:
            break

        # Use binary search helper to find which mesh-plane pair this vertex belongs to
        pair_idx, vertex_idx = find_pair_from_cumulative_index(task_id, shape_pairs_mesh_plane_cumsum, num_pairs)

        # Get the mesh-plane pair
        pair = shape_pairs_mesh_plane[pair_idx]
        mesh_shape = pair[0]
        plane_shape = pair[1]

        # Get mesh
        mesh_id = shape_source_ptr[mesh_shape]
        if mesh_id == wp.uint64(0):
            continue

        mesh_obj = wp.mesh_get(mesh_id)
        if vertex_idx >= mesh_obj.points.shape[0]:
            continue

        # Get mesh world transform
        mesh_body_idx = shape_body[mesh_shape]
        X_mesh_ws = shape_transform[mesh_shape]
        if mesh_body_idx >= 0:
            X_mesh_ws = wp.transform_multiply(body_q[mesh_body_idx], shape_transform[mesh_shape])

        # Get plane world transform
        plane_body_idx = shape_body[plane_shape]
        X_plane_ws = shape_transform[plane_shape]
        if plane_body_idx >= 0:
            X_plane_ws = wp.transform_multiply(body_q[plane_body_idx], shape_transform[plane_shape])

        # Get vertex position in mesh local space and transform to world space
        mesh_scale = shape_scale[mesh_shape]
        vertex_local = wp.cw_mul(mesh_obj.points[vertex_idx], mesh_scale)
        vertex_world = wp.transform_point(X_mesh_ws, vertex_local)

        # Get plane normal in world space (plane normal is along local +Z, pointing upward)
        plane_normal = wp.transform_vector(X_plane_ws, wp.vec3(0.0, 0.0, 1.0))

        # Project vertex onto plane to get closest point
        X_plane_sw = wp.transform_inverse(X_plane_ws)
        vertex_in_plane_space = wp.transform_point(X_plane_sw, vertex_world)
        point_on_plane_local = wp.vec3(vertex_in_plane_space[0], vertex_in_plane_space[1], 0.0)
        point_on_plane = wp.transform_point(X_plane_ws, point_on_plane_local)

        # Compute distance and normal
        diff = vertex_world - point_on_plane
        distance = wp.dot(diff, plane_normal)

        # Check if vertex is within collision margin
        thickness_mesh = shape_thickness[mesh_shape]
        thickness_plane = shape_thickness[plane_shape]
        total_thickness = thickness_mesh + thickness_plane

        # Treat plane as a half-space: generate contact for all vertices on or below the plane
        # (distance < margin means vertex is close to or penetrating the plane)
        if distance < rigid_contact_margin + total_thickness:
            # Get inverse transforms for body-local contact points
            X_mesh_bw = wp.transform_identity() if mesh_body_idx == -1 else wp.transform_inverse(body_q[mesh_body_idx])
            X_plane_bw = (
                wp.transform_identity() if plane_body_idx == -1 else wp.transform_inverse(body_q[plane_body_idx])
            )

            # Write contact
            # Note: write_contact expects contact_normal_a_to_b pointing FROM mesh TO plane (downward)
            # plane_normal points upward, so we need to negate it
            write_contact(
                (vertex_world + point_on_plane) * 0.5,  # contact_point_center
                -plane_normal,  # contact_normal_a_to_b (from mesh to plane, pointing downward)
                distance,  # contact_distance
                0.0,  # radius_eff_a (mesh has no effective radius)
                0.0,  # radius_eff_b (plane has no effective radius)
                thickness_mesh,  # thickness_a
                thickness_plane,  # thickness_b
                mesh_shape,  # shape_a
                plane_shape,  # shape_b
                X_mesh_bw,  # X_bw_a
                X_plane_bw,  # X_bw_b
                task_id,  # tid
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

        # Compute tight AABB using helper function
        aabb_min_world, aabb_max_world = compute_tight_aabb_from_support(shape_data, orientation, pos, data_provider)

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
        shape_collision_group: wp.array(dtype=int) | None = None,
        shape_world: wp.array(dtype=int) | None = None,
        shape_flags: wp.array(dtype=int) | None = None,
        sap_sort_type=None,
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
            shape_collision_group (wp.array | None, optional): Array of collision group IDs for each shape.
                Used during broad phase kernel execution to filter pairs based on collision group rules.
            shape_world (wp.array | None, optional): Array of world indices for each shape.
                Required by NXN and SAP broad phases to organize geometries by world. If None, will be set during collide().
            shape_flags (wp.array | None, optional): Array of shape flags (ShapeFlags) for each shape.
                Used by NXN and SAP broad phases to filter out non-colliding shapes (e.g., visual-only).
                If provided, only shapes with COLLIDE_SHAPES flag will participate in broad phase.
            sap_sort_type (SAPSortType | None, optional): Sorting algorithm for SAP broad phase.
                Only used when broad_phase_mode is BroadPhaseMode.SAP. Options: SEGMENTED or TILE.
                If None, uses default (SEGMENTED).
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
            if shape_world is None:
                raise ValueError("shape_world must be provided when using BroadPhaseMode.NXN")
            self.nxn_broadphase = BroadPhaseAllPairs(shape_world, geom_flags=shape_flags, device=device)
            self.sap_broadphase = None
            self.explicit_broadphase = None
            self.shape_pairs_filtered = None
        elif self.broad_phase_mode == BroadPhaseMode.SAP:
            if shape_world is None:
                raise ValueError("shape_world must be provided when using BroadPhaseMode.SAP")
            # Pass sort_type if provided, otherwise use default
            sap_kwargs = {"geom_flags": shape_flags, "device": device}
            if sap_sort_type is not None:
                sap_kwargs["sort_type"] = sap_sort_type
            self.sap_broadphase = BroadPhaseSAP(shape_world, **sap_kwargs)
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

            # Allocate buffers for mesh collision handling
            self.shape_pairs_mesh = wp.zeros(self.shape_pairs_max, dtype=wp.vec2i, device=device)
            self.shape_pairs_mesh_count = wp.zeros(1, dtype=wp.int32, device=device)
            # Conservative estimate: each mesh pair could have many triangle overlaps
            # Use a generous multiplier for the triangle pairs buffer
            max_triangle_pairs = 1000000  # Conservative estimate
            self.triangle_pairs = wp.zeros(max_triangle_pairs, dtype=wp.vec3i, device=device)
            self.triangle_pairs_count = wp.zeros(1, dtype=wp.int32, device=device)

            # Allocate buffers for mesh-plane collision handling
            self.shape_pairs_mesh_plane = wp.zeros(self.shape_pairs_max, dtype=wp.vec2i, device=device)
            self.shape_pairs_mesh_plane_count = wp.zeros(1, dtype=wp.int32, device=device)
            self.shape_pairs_mesh_plane_cumsum = wp.zeros(self.shape_pairs_max, dtype=wp.int32, device=device)
            self.mesh_plane_vertex_total_count = wp.zeros(1, dtype=wp.int32, device=device)

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
        sap_sort_type=None,
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
            sap_sort_type (SAPSortType | None, optional): Sorting algorithm for SAP broad phase.
                Only used when broad_phase_mode is BroadPhaseMode.SAP. If None, uses default (SEGMENTED).

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
            shape_collision_group=model.shape_collision_group if hasattr(model, "shape_collision_group") else None,
            shape_world=model.shape_world if hasattr(model, "shape_world") else None,
            shape_flags=model.shape_flags if hasattr(model, "shape_flags") else None,
            sap_sort_type=sap_sort_type,
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
        self.shape_pairs_mesh_count.zero_()
        self.triangle_pairs_count.zero_()
        self.shape_pairs_mesh_plane_count.zero_()
        self.mesh_plane_vertex_total_count.zero_()

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
        total_num_threads = block_dim * 1024
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
                self.shape_aabb_lower,
                self.shape_aabb_upper,
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
                self.shape_pairs_mesh,
                self.shape_pairs_mesh_count,
                self.shape_pairs_mesh_plane,
                self.shape_pairs_mesh_plane_cumsum,
                self.shape_pairs_mesh_plane_count,
                self.mesh_plane_vertex_total_count,
            ],
            device=contacts.device,
            block_dim=block_dim,
        )

        # Launch mesh-plane contact processing kernel with fixed thread count
        wp.launch(
            kernel=process_mesh_plane_contacts_kernel,
            dim=total_num_threads,
            inputs=[
                state.body_q,
                model.shape_transform,
                model.shape_body,
                model.shape_type,
                model.shape_scale,
                model.shape_thickness,
                model.shape_source_ptr,
                self.shape_pairs_mesh_plane,
                self.shape_pairs_mesh_plane_cumsum,
                self.shape_pairs_mesh_plane_count,
                self.mesh_plane_vertex_total_count,
                self.rigid_contact_margin,
                contacts.rigid_contact_max,
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

        # Launch mesh triangle overlap detection kernel
        num_tile_blocks = 1024
        tile_size = 128
        second_dim = tile_size if ENABLE_TILE_BVH_QUERY else 1
        wp.launch(
            kernel=find_mesh_triangle_overlaps_kernel,
            dim=[num_tile_blocks, second_dim],
            inputs=[
                state.body_q,
                model.shape_transform,
                model.shape_body,
                model.shape_type,
                model.shape_scale,
                model.shape_collision_radius,
                model.shape_source_ptr,
                self.shape_pairs_mesh,
                self.shape_pairs_mesh_count,
                self.rigid_contact_margin,
                num_tile_blocks,
            ],
            outputs=[
                self.triangle_pairs,
                self.triangle_pairs_count,
            ],
            device=model.device,
            block_dim=tile_size,
        )

        # Launch mesh triangle contact processing kernel
        wp.launch(
            kernel=process_mesh_triangle_contacts_kernel,
            dim=total_num_threads,
            inputs=[
                state.body_q,
                model.shape_transform,
                model.shape_body,
                model.shape_type,
                model.shape_scale,
                model.shape_collision_radius,
                model.shape_thickness,
                model.shape_source_ptr,
                self.triangle_pairs,
                self.triangle_pairs_count,
                self.rigid_contact_margin,
                contacts.rigid_contact_max,
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
            device=model.device,
            block_dim=block_dim,
        )

        return contacts
