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

from typing import Any

import warp as wp

from .broad_phase_common import binary_search
from .collision_convex import create_solve_convex_multi_contact, create_solve_convex_single_contact
from .contact_data import ContactData
from .support_function import GenericShapeData, GeoTypeEx, SupportMapDataProvider, pack_mesh_ptr, support_map
from .types import GeoType

# Configuration flag for multi-contact generation
ENABLE_MULTI_CONTACT = True

# Configuration flag for tiled BVH queries (experimental)
ENABLE_TILE_BVH_QUERY = True

# Type definitions for multi-contact manifolds
_mat53f = wp.types.matrix((5, 3), wp.float32)
_vec5 = wp.types.vector(5, wp.float32)

# Type definitions for single-contact mode
_vec1 = wp.types.vector(1, wp.float32)


@wp.func
def build_pair_key2(shape_a: wp.uint32, shape_b: wp.uint32) -> wp.uint64:
    """
    Build a 64-bit key from two shape indices.
    Upper 32 bits: shape_a
    Lower 32 bits: shape_b
    """
    key = wp.uint64(shape_a)
    key = key << wp.uint64(32)
    key = key | wp.uint64(shape_b)
    return key


@wp.func
def build_pair_key3(shape_a: wp.uint32, shape_b: wp.uint32, triangle_idx: wp.uint32) -> wp.uint64:
    """
    Build a 63-bit key from two shape indices and a triangle index (MSB is 0 for signed int64 compatibility).
    Bit 63: 0 (reserved for sign bit)
    Bits 62-43: shape_a (20 bits)
    Bits 42-23: shape_b (20 bits)
    Bits 22-0: triangle_idx (23 bits)

    Max values: shape_a < 2^20 (1,048,576), shape_b < 2^20 (1,048,576), triangle_idx < 2^23 (8,388,608)
    """
    assert shape_a < wp.uint32(1048576), "shape_a must be < 2^20 (1,048,576)"
    assert shape_b < wp.uint32(1048576), "shape_b must be < 2^20 (1,048,576)"
    assert triangle_idx < wp.uint32(8388608), "triangle_idx must be < 2^23 (8,388,608)"

    key = wp.uint64(shape_a & wp.uint32(0xFFFFF))  # Mask to 20 bits
    key = key << wp.uint64(20)
    key = key | wp.uint64(shape_b & wp.uint32(0xFFFFF))  # Mask to 20 bits
    key = key << wp.uint64(23)
    key = key | wp.uint64(triangle_idx & wp.uint32(0x7FFFFF))  # Mask to 23 bits
    return key


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
def no_post_process_contact(
    contact_data: ContactData,
    shape_a: GenericShapeData,
    pos_a_adjusted: wp.vec3,
    rot_a: wp.quat,
    shape_b: GenericShapeData,
    pos_b_adjusted: wp.vec3,
    rot_b: wp.quat,
) -> ContactData:
    return contact_data


@wp.func
def post_process_axial_on_discrete_contact(
    contact_data: ContactData,
    shape_a: GenericShapeData,
    pos_a_adjusted: wp.vec3,
    rot_a: wp.quat,
    shape_b: GenericShapeData,
    pos_b_adjusted: wp.vec3,
    rot_b: wp.quat,
) -> ContactData:
    """
    Post-process a single contact for minkowski objects and axial shape rolling.

    This function handles:
    1. Minkowski objects (spheres/capsules): Adjusts contact point and distance for rounded geometry
    2. Axial shapes on discrete surfaces: Projects contact point for rolling stabilization

    Args:
        contact_data: Contact data to post-process
        shape_a: Shape data for shape A
        pos_a_adjusted: Position of shape A
        rot_a: Orientation of shape A
        shape_b: Shape data for shape B
        pos_b_adjusted: Position of shape B
        rot_b: Orientation of shape B

    Returns:
        Post-processed contact data
    """
    type_a = shape_a.shape_type
    type_b = shape_b.shape_type
    normal = contact_data.contact_normal_a_to_b
    radius_eff_a = contact_data.radius_eff_a
    radius_eff_b = contact_data.radius_eff_b

    # 1. Minkowski object processing for spheres and capsules
    # Adjust contact point and distance for sphere/capsule A
    if type_a == int(GeoType.SPHERE) or type_a == int(GeoType.CAPSULE):
        contact_data.contact_point_center = contact_data.contact_point_center + normal * (radius_eff_a * 0.5)
        contact_data.contact_distance = contact_data.contact_distance - radius_eff_a

    # Adjust contact point and distance for sphere/capsule B
    if type_b == int(GeoType.SPHERE) or type_b == int(GeoType.CAPSULE):
        contact_data.contact_point_center = contact_data.contact_point_center - normal * (radius_eff_b * 0.5)
        contact_data.contact_distance = contact_data.contact_distance - radius_eff_b

    # 2. Axial shape rolling stabilization (cylinders and cones on discrete surfaces)
    is_discrete_a = is_discrete_shape(type_a)
    is_discrete_b = is_discrete_shape(type_b)
    is_axial_a = type_a == int(GeoType.CYLINDER) or type_a == int(GeoType.CONE)
    is_axial_b = type_b == int(GeoType.CYLINDER) or type_b == int(GeoType.CONE)

    # Only process if we have discrete vs axial configuration
    if (is_discrete_a and is_axial_b) or (is_discrete_b and is_axial_a):
        # Extract the axial shape parameters
        if is_discrete_a and is_axial_b:
            shape_axis = wp.quat_rotate(rot_b, wp.vec3(0.0, 0.0, 1.0))
            shape_radius = shape_b.scale[0]
            shape_half_height = shape_b.scale[1]
            is_cone = type_b == int(GeoType.CONE)
            shape_pos = pos_b_adjusted
            axial_normal = normal
        else:  # is_discrete_b and is_axial_a
            shape_axis = wp.quat_rotate(rot_a, wp.vec3(0.0, 0.0, 1.0))
            shape_radius = shape_a.scale[0]
            shape_half_height = shape_a.scale[1]
            is_cone = type_a == int(GeoType.CONE)
            shape_pos = pos_a_adjusted
            axial_normal = -normal  # Flip normal for shape A

        # Check if shape is in rolling configuration
        axis_normal_dot = wp.abs(wp.dot(shape_axis, axial_normal))

        # Compute threshold based on shape type
        is_rolling = False
        if is_cone:
            # For a cone rolling on its base, the axis makes an angle with the normal
            cone_half_angle = wp.atan2(shape_radius, 2.0 * shape_half_height)
            tolerance_angle = wp.static(2.0 * wp.pi / 180.0)  # 2 degrees
            lower_threshold = wp.sin(cone_half_angle - tolerance_angle)
            upper_threshold = wp.sin(cone_half_angle + tolerance_angle)

            if axis_normal_dot >= lower_threshold and axis_normal_dot <= upper_threshold:
                is_rolling = True
        else:
            # For cylinder: axis should be perpendicular to normal (dot product ≈ 0)
            perpendicular_threshold = wp.static(wp.sin(2.0 * wp.pi / 180.0))
            if axis_normal_dot <= perpendicular_threshold:
                is_rolling = True

        # If rolling, project contact point onto the projection plane
        if is_rolling:
            projection_plane_normal = wp.normalize(wp.cross(shape_axis, axial_normal))
            point_on_projection_plane = shape_pos

            # Project the contact point
            projected_point = project_point_onto_plane(
                contact_data.contact_point_center, point_on_projection_plane, projection_plane_normal
            )

            # Update the contact with the projected point
            contact_data.contact_point_center = projected_point

    return contact_data


def create_compute_gjk_mpr_contacts(
    writer_func: Any, post_process_contact: Any = post_process_axial_on_discrete_contact
):
    """
    Factory function to create a compute_gjk_mpr_contacts function with a specific writer function.

    Args:
        writer_func: Function to write contact data (signature: (ContactData, writer_data) -> None)

    Returns:
        A compute_gjk_mpr_contacts function with the writer function baked in
    """

    @wp.func
    def compute_gjk_mpr_contacts(
        shape_a_data: GenericShapeData,
        shape_b_data: GenericShapeData,
        rot_a: wp.quat,
        rot_b: wp.quat,
        pos_a_adjusted: wp.vec3,
        pos_b_adjusted: wp.vec3,
        rigid_contact_margin: float,
        shape_a: int,
        shape_b: int,
        thickness_a: float,
        thickness_b: float,
        writer_data: Any,
        pair_key: wp.uint64,
    ):
        """
        Compute contacts between two shapes using GJK/MPR algorithm and write them.

        Args:
            shape_a_data: Generic shape data for shape A (contains shape_type)
            shape_b_data: Generic shape data for shape B (contains shape_type)
            rot_a: Orientation of shape A
            rot_b: Orientation of shape B
            pos_a_adjusted: Adjusted position of shape A
            pos_b_adjusted: Adjusted position of shape B
            rigid_contact_margin: Contact margin for rigid bodies
            shape_a: Index of shape A
            shape_b: Index of shape B
            thickness_a: Thickness of shape A
            thickness_b: Thickness of shape B
            writer_data: Data structure for contact writer
        """
        data_provider = SupportMapDataProvider()

        radius_eff_a = float(0.0)
        radius_eff_b = float(0.0)

        small_radius = 0.0001

        # Get shape types from shape data
        type_a = shape_a_data.shape_type
        type_b = shape_b_data.shape_type

        # Special treatment for minkowski objects
        if type_a == int(GeoType.SPHERE) or type_a == int(GeoType.CAPSULE):
            radius_eff_a = shape_a_data.scale[0]
            shape_a_data.scale[0] = small_radius

        if type_b == int(GeoType.SPHERE) or type_b == int(GeoType.CAPSULE):
            radius_eff_b = shape_b_data.scale[0]
            shape_b_data.scale[0] = small_radius

        # Pre-pack ContactData template with static information
        contact_template = ContactData()
        contact_template.radius_eff_a = radius_eff_a
        contact_template.radius_eff_b = radius_eff_b
        contact_template.thickness_a = thickness_a
        contact_template.thickness_b = thickness_b
        contact_template.shape_a = shape_a
        contact_template.shape_b = shape_b
        contact_template.margin = rigid_contact_margin
        contact_template.feature_pair_key = pair_key

        if wp.static(ENABLE_MULTI_CONTACT):
            wp.static(create_solve_convex_multi_contact(support_map, writer_func, post_process_contact))(
                shape_a_data,
                shape_b_data,
                rot_a,
                rot_b,
                pos_a_adjusted,
                pos_b_adjusted,
                0.0,  # sum_of_contact_offsets - gap
                data_provider,
                rigid_contact_margin + radius_eff_a + radius_eff_b,
                type_a == int(GeoType.SPHERE)
                or type_b == int(GeoType.SPHERE)
                or type_a == int(GeoType.ELLIPSOID)
                or type_b == int(GeoType.ELLIPSOID),
                writer_data,
                contact_template,
            )
        else:
            wp.static(create_solve_convex_single_contact(support_map, writer_func, post_process_contact))(
                shape_a_data,
                shape_b_data,
                rot_a,
                rot_b,
                pos_a_adjusted,
                pos_b_adjusted,
                0.0,  # sum_of_contact_offsets - gap
                data_provider,
                rigid_contact_margin + radius_eff_a + radius_eff_b,
                writer_data,
                contact_template,
            )

    return compute_gjk_mpr_contacts


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


def create_find_contacts(writer_func: Any):
    """
    Factory function to create a find_contacts function with a specific writer function.

    Args:
        writer_func: Function to write contact data (signature: (ContactData, writer_data) -> None)

    Returns:
        A find_contacts function with the writer function baked in
    """

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
        shape_a: int,
        shape_b: int,
        thickness_a: float,
        thickness_b: float,
        writer_data: Any,
    ):
        """
        Find contacts between two shapes using GJK/MPR algorithm and write them using the writer function.

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
            shape_a: Index of shape A
            shape_b: Index of shape B
            thickness_a: Thickness of shape A
            thickness_b: Thickness of shape B
            writer_data: Data structure for contact writer
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

        # Build pair key for contact matching
        pair_key = build_pair_key2(wp.uint32(shape_a), wp.uint32(shape_b))

        # Compute and write contacts using GJK/MPR
        wp.static(create_compute_gjk_mpr_contacts(writer_func))(
            shape_data_a,
            shape_data_b,
            quat_a,
            quat_b,
            pos_a_adjusted,
            pos_b_adjusted,
            rigid_contact_margin,
            shape_a,
            shape_b,
            thickness_a,
            thickness_b,
            writer_data,
            pair_key,
        )

    return find_contacts


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
