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

import warp as wp

from .broad_phase_common import binary_search
from .collision_convex import create_solve_convex_multi_contact, create_solve_convex_single_contact
from .support_function import GenericShapeData, GeoTypeEx, SupportMapDataProvider, pack_mesh_ptr, support_map
from .types import GeoType

# Configuration flag for multi-contact generation
ENABLE_MULTI_CONTACT = True

# Configuration flag for tiled BVH queries (experimental)
ENABLE_TILE_BVH_QUERY = False

# Pre-create the convex contact solvers (usable inside kernels)
solve_convex_multi_contact = create_solve_convex_multi_contact(support_map)
solve_convex_single_contact = create_solve_convex_single_contact(support_map)

# Type definitions for multi-contact manifolds
_mat53f = wp.types.matrix((5, 3), wp.float32)
_vec5 = wp.types.vector(5, wp.float32)


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


@wp.func
def compute_gjk_mpr_contacts(
    geom_a: GenericShapeData,
    geom_b: GenericShapeData,
    rot_a: wp.quat,
    rot_b: wp.quat,
    pos_a_adjusted: wp.vec3,
    pos_b_adjusted: wp.vec3,
    rigid_contact_margin: float,
):
    """
    Compute contacts between two shapes using GJK/MPR algorithm.

    Args:
        geom_a: Generic shape data for shape A (contains shape_type)
        geom_b: Generic shape data for shape B (contains shape_type)
        rot_a: Orientation of shape A
        rot_b: Orientation of shape B
        pos_a_adjusted: Adjusted position of shape A
        pos_b_adjusted: Adjusted position of shape B
        rigid_contact_margin: Contact margin for rigid bodies

    Returns:
        Tuple of (count, normal, signed_distances, points, radius_eff_a, radius_eff_b)
    """
    data_provider = SupportMapDataProvider()

    radius_eff_a = float(0.0)
    radius_eff_b = float(0.0)

    small_radius = 0.0001

    # Get shape types from shape data
    type_a = geom_a.shape_type
    type_b = geom_b.shape_type

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


@wp.func
def find_contacts(
    pos_a: wp.vec3,
    pos_b: wp.vec3,
    quat_a: wp.quat,
    quat_b: wp.quat,
    shape_data_a: GenericShapeData,
    shape_data_b: GenericShapeData,
    is_infinite_plane_a: bool,
    is_infinite_plane_b: bool,
    bsphere_radius_a: float,
    bsphere_radius_b: float,
    rigid_contact_margin: float,
):
    """
    Find contacts between two shapes using GJK/MPR algorithm.

    Args:
        pos_a: Position of shape A in world space
        pos_b: Position of shape B in world space
        quat_a: Orientation of shape A
        quat_b: Orientation of shape B
        shape_data_a: Generic shape data for shape A (contains shape_type)
        shape_data_b: Generic shape data for shape B (contains shape_type)
        is_infinite_plane_a: Whether shape A is an infinite plane
        is_infinite_plane_b: Whether shape B is an infinite plane
        bsphere_radius_a: Bounding sphere radius of shape A
        bsphere_radius_b: Bounding sphere radius of shape B
        rigid_contact_margin: Contact margin for rigid bodies

    Returns:
        Tuple of (count, normal, signed_distances, points, radius_eff_a, radius_eff_b)
    """
    # Convert infinite planes to cube proxies for GJK/MPR compatibility
    # Use the OTHER object's radius to properly size the cube
    # Only convert if it's an infinite plane (finite planes can be handled normally)
    pos_a_adjusted = pos_a
    if is_infinite_plane_a:
        # Position the cube based on the OTHER object's position (pos_b)
        # Note: convert_infinite_plane_to_cube modifies shape_data_a.shape_type to BOX
        shape_data_a, pos_a_adjusted = convert_infinite_plane_to_cube(
            shape_data_a, quat_a, pos_a, pos_b, bsphere_radius_b + rigid_contact_margin
        )

    pos_b_adjusted = pos_b
    if is_infinite_plane_b:
        # Position the cube based on the OTHER object's position (pos_a)
        # Note: convert_infinite_plane_to_cube modifies shape_data_b.shape_type to BOX
        shape_data_b, pos_b_adjusted = convert_infinite_plane_to_cube(
            shape_data_b, quat_b, pos_b, pos_a, bsphere_radius_a + rigid_contact_margin
        )

    # Compute contacts using GJK/MPR
    count, normal, signed_distances, points, radius_eff_a, radius_eff_b = compute_gjk_mpr_contacts(
        shape_data_a,
        shape_data_b,
        quat_a,
        quat_b,
        pos_a_adjusted,
        pos_b_adjusted,
        rigid_contact_margin,
    )

    return count, normal, signed_distances, points, radius_eff_a, radius_eff_b


@wp.func
def pre_contact_check(
    shape_a: int,
    shape_b: int,
    pos_a: wp.vec3,
    pos_b: wp.vec3,
    quat_a: wp.quat,
    quat_b: wp.quat,
    shape_data_a: GenericShapeData,
    shape_data_b: GenericShapeData,
    aabb_a_lower: wp.vec3,
    aabb_a_upper: wp.vec3,
    aabb_b_lower: wp.vec3,
    aabb_b_upper: wp.vec3,
    pair: wp.vec2i,
    mesh_id_a: wp.uint64,
    mesh_id_b: wp.uint64,
    shape_pairs_mesh: wp.array(dtype=wp.vec2i),
    shape_pairs_mesh_count: wp.array(dtype=int),
    shape_pairs_mesh_plane: wp.array(dtype=wp.vec2i),
    shape_pairs_mesh_plane_cumsum: wp.array(dtype=int),
    shape_pairs_mesh_plane_count: wp.array(dtype=int),
    mesh_plane_vertex_total_count: wp.array(dtype=int),
):
    """
    Perform pre-contact checks for early rejection and special case handling.

    Args:
        shape_a: Index of shape A
        shape_b: Index of shape B
        pos_a: Position of shape A in world space
        pos_b: Position of shape B in world space
        quat_a: Orientation of shape A
        quat_b: Orientation of shape B
        shape_data_a: Generic shape data for shape A (contains shape_type and scale)
        shape_data_b: Generic shape data for shape B (contains shape_type and scale)
        aabb_a_lower: Lower bound of AABB for shape A
        aabb_a_upper: Upper bound of AABB for shape A
        aabb_b_lower: Lower bound of AABB for shape B
        aabb_b_upper: Upper bound of AABB for shape B
        pair: Shape pair indices
        mesh_id_a: Mesh ID pointer for shape A (wp.uint64(0) if not a mesh)
        mesh_id_b: Mesh ID pointer for shape B (wp.uint64(0) if not a mesh)
        shape_pairs_mesh: Output array for mesh collision pairs
        shape_pairs_mesh_count: Counter for mesh collision pairs
        shape_pairs_mesh_plane: Output array for mesh-plane collision pairs
        shape_pairs_mesh_plane_cumsum: Cumulative sum array for mesh-plane vertices
        shape_pairs_mesh_plane_count: Counter for mesh-plane collision pairs
        mesh_plane_vertex_total_count: Total vertex count for mesh-plane collisions

    Returns:
        Tuple of (skip_pair, is_infinite_plane_a, is_infinite_plane_b, bsphere_radius_a, bsphere_radius_b)
    """
    # Get shape types from shape data
    type_a = shape_data_a.shape_type
    type_b = shape_data_b.shape_type

    # Check if shapes are infinite planes (scale.x == 0 and scale.y == 0)
    # Scale is already in shape_data, no need for array lookup
    is_infinite_plane_a = (type_a == int(GeoType.PLANE)) and (
        shape_data_a.scale[0] == 0.0 and shape_data_a.scale[1] == 0.0
    )
    is_infinite_plane_b = (type_b == int(GeoType.PLANE)) and (
        shape_data_b.scale[0] == 0.0 and shape_data_b.scale[1] == 0.0
    )

    # Early return: both shapes are infinite planes
    if is_infinite_plane_a and is_infinite_plane_b:
        return True, is_infinite_plane_a, is_infinite_plane_b, float(0.0), float(0.0)

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
        return True, is_infinite_plane_a, is_infinite_plane_b, bsphere_radius_a, bsphere_radius_b

    # Check for mesh vs infinite plane collision - special handling
    # After sorting, type_a <= type_b, so we only need to check one direction
    if type_a == int(GeoType.PLANE) and type_b == int(GeoType.MESH):
        # Check if plane is infinite (scale x and y are zero) - use scale from shape_data
        if shape_data_a.scale[0] == 0.0 and shape_data_a.scale[1] == 0.0:
            # Get mesh vertex count using the provided mesh_id
            if mesh_id_b != wp.uint64(0):
                mesh_obj = wp.mesh_get(mesh_id_b)
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
            return True, is_infinite_plane_a, is_infinite_plane_b, bsphere_radius_a, bsphere_radius_b

    # Check for other mesh collisions - add to separate buffer for specialized handling
    if type_a == int(GeoType.MESH) or type_b == int(GeoType.MESH):
        # Add to mesh collision buffer using atomic counter
        mesh_pair_idx = wp.atomic_add(shape_pairs_mesh_count, 0, 1)
        if mesh_pair_idx < shape_pairs_mesh.shape[0]:
            shape_pairs_mesh[mesh_pair_idx] = pair
        return True, is_infinite_plane_a, is_infinite_plane_b, bsphere_radius_a, bsphere_radius_b

    return False, is_infinite_plane_a, is_infinite_plane_b, bsphere_radius_a, bsphere_radius_b


@wp.func
def mesh_vs_convex_midphase(
    mesh_shape: int,
    non_mesh_shape: int,
    X_mesh_ws: wp.transform,
    X_ws: wp.transform,
    mesh_id: wp.uint64,
    shape_type: wp.array(dtype=int),
    shape_data: wp.array(dtype=wp.vec4),
    shape_source_ptr: wp.array(dtype=wp.uint64),
    rigid_contact_margin: float,
    triangle_pairs: wp.array(dtype=wp.vec3i),
    triangle_pairs_count: wp.array(dtype=int),
):
    """
    Perform mesh vs convex shape midphase collision detection.

    This function finds all mesh triangles that overlap with the convex shape's AABB
    by querying the mesh BVH. The results are output as triangle pairs for further
    narrow-phase collision detection.

    Args:
        mesh_shape: Index of the mesh shape
        non_mesh_shape: Index of the non-mesh (convex) shape
        X_mesh_ws: Mesh world-space transform
        X_ws: Non-mesh shape world-space transform
        mesh_id: Mesh BVH ID
        shape_type: Array of shape types
        shape_data: Array of shape data (vec4: scale.xyz, thickness.w)
        shape_source_ptr: Array of mesh/SDF source pointers
        rigid_contact_margin: Contact margin for rigid bodies
        triangle_pairs: Output array for triangle pairs (mesh_shape, non_mesh_shape, tri_index)
        triangle_pairs_count: Counter for triangle pairs
    """
    # Get inverse mesh transform (world to mesh local space)
    X_mesh_sw = wp.transform_inverse(X_mesh_ws)

    # Compute transform from non-mesh shape local space to mesh local space
    # X_mesh_shape = X_mesh_sw * X_ws
    X_mesh_shape = wp.transform_multiply(X_mesh_sw, X_ws)
    pos_in_mesh = wp.transform_get_translation(X_mesh_shape)
    orientation_in_mesh = wp.transform_get_rotation(X_mesh_shape)

    # Create generic shape data for non-mesh shape
    geo_type = shape_type[non_mesh_shape]
    data_vec4 = shape_data[non_mesh_shape]
    scale = wp.vec3(data_vec4[0], data_vec4[1], data_vec4[2])

    generic_shape_data = GenericShapeData()
    generic_shape_data.shape_type = geo_type
    generic_shape_data.scale = scale
    generic_shape_data.auxiliary = wp.vec3(0.0, 0.0, 0.0)

    # For CONVEX_MESH, pack the mesh pointer
    if geo_type == int(GeoType.CONVEX_MESH):
        generic_shape_data.auxiliary = pack_mesh_ptr(shape_source_ptr[non_mesh_shape])

    data_provider = SupportMapDataProvider()

    # Compute tight AABB directly in mesh local space for optimal fit
    aabb_lower, aabb_upper = compute_tight_aabb_from_support(
        generic_shape_data, orientation_in_mesh, pos_in_mesh, data_provider
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

    This function is useful for mapping a flat global index to a (pair_index, local_index)
    tuple when work is distributed across multiple pairs with varying sizes.

    Args:
        global_idx: Global index to search for
        cumulative_sums: Array of inclusive cumulative sums (end indices for each pair)
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


@wp.func
def get_triangle_shape_from_mesh(
    mesh_id: wp.uint64,
    mesh_scale: wp.vec3,
    X_mesh_ws: wp.transform,
    tri_idx: int,
) -> tuple[GenericShapeData, wp.vec3]:
    """
    Extract triangle shape data from a mesh.

    This function retrieves a specific triangle from a mesh and creates a GenericShapeData
    structure for collision detection. The triangle is represented in world space with
    vertex A as the origin.

    Args:
        mesh_id: The mesh ID (use wp.mesh_get to retrieve the mesh object)
        mesh_scale: Scale to apply to mesh vertices
        X_mesh_ws: Mesh world-space transform
        tri_idx: Triangle index in the mesh

    Returns:
        Tuple of (shape_data, v0_world) where:
        - shape_data: GenericShapeData with triangle geometry (type=TRIANGLE, scale=B-A, auxiliary=C-A)
        - v0_world: First vertex position in world space (used as triangle origin)
    """
    # Get the mesh object from the ID
    mesh = wp.mesh_get(mesh_id)

    # Extract triangle vertices from mesh (indices are stored as flat array: i0, i1, i2, i0, i1, i2, ...)
    idx0 = mesh.indices[tri_idx * 3 + 0]
    idx1 = mesh.indices[tri_idx * 3 + 1]
    idx2 = mesh.indices[tri_idx * 3 + 2]

    # Get vertex positions in mesh local space (with scale applied)
    v0_local = wp.cw_mul(mesh.points[idx0], mesh_scale)
    v1_local = wp.cw_mul(mesh.points[idx1], mesh_scale)
    v2_local = wp.cw_mul(mesh.points[idx2], mesh_scale)

    # Transform vertices to world space
    v0_world = wp.transform_point(X_mesh_ws, v0_local)
    v1_world = wp.transform_point(X_mesh_ws, v1_local)
    v2_world = wp.transform_point(X_mesh_ws, v2_local)

    # Create triangle shape data: vertex A at origin, B-A in scale, C-A in auxiliary
    shape_data = GenericShapeData()
    shape_data.shape_type = int(GeoTypeEx.TRIANGLE)
    shape_data.scale = v1_world - v0_world  # B - A
    shape_data.auxiliary = v2_world - v0_world  # C - A

    return shape_data, v0_world


@wp.func
def postprocess_triangle_contacts(
    triangle_shape_data: GenericShapeData,
    triangle_pos: wp.vec3,
    normal: wp.vec3,
    signed_distances: _vec5,
    count: int,
) -> tuple[_vec5, wp.vec3]:
    """
    Post-process contacts for triangle vs convex shape collisions.

    This function checks if the contact normal is pushing an object into the triangle
    (opposite to the triangle's face normal) and zeros the penetration depth if needed.
    This prevents incorrect contact forces that would push objects through triangles.

    The correction is only applied when the contact normal is nearly parallel to the
    triangle normal (within 10 degrees), indicating a face-to-face contact scenario.

    Args:
        triangle_shape_data: Triangle shape data (type=TRIANGLE, scale=B-A, auxiliary=C-A)
        triangle_pos: Position of triangle vertex A in world space
        normal: Contact normal from GJK/MPR (points from shape A to shape B)
        signed_distances: Signed distances for each contact point
        count: Number of contact points

    Returns:
        Tuple of (corrected_signed_distances, corrected_normal)
    """
    # Reconstruct triangle vertices from shape data
    # Triangle is stored as: vertex A at origin (triangle_pos), B-A in scale, C-A in auxiliary
    v0_world = triangle_pos
    v1_world = triangle_pos + triangle_shape_data.scale  # A + (B - A) = B
    v2_world = triangle_pos + triangle_shape_data.auxiliary  # A + (C - A) = C

    # Compute triangle normal (cross product of edges)
    edge1 = v1_world - v0_world  # B - A
    edge2 = v2_world - v0_world  # C - A
    triangle_normal = wp.normalize(wp.cross(edge1, edge2))

    # Post-process contacts: check if contact normal is pushing object into the triangle
    # Only apply correction if the contact normal is nearly parallel to the triangle normal
    # (within 10 degrees, meaning cos(angle) > cos(10°) ≈ 0.985)
    cos_threshold = wp.static(wp.cos(wp.radians(10.0)))
    dot_product = wp.dot(normal, triangle_normal)
    abs_dot = wp.abs(dot_product)

    # Check if nearly parallel (within 10 degrees of 0° or 180°)
    if abs_dot > cos_threshold:
        # If dot product is negative, contact normal is pointing opposite to triangle normal
        # (pushing object into the triangle), so we zero the penetration depth
        # to prevent the object from being pushed through the triangle
        if dot_product < 0.0:
            normal = -normal
            for i in range(count):
                signed_distances[i] = 0.0  # This prevents energetic reactions

    return signed_distances, normal
