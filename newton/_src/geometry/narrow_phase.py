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

from ..geometry.collision_core import (
    ENABLE_TILE_BVH_QUERY,
    compute_gjk_mpr_contacts,
    compute_tight_aabb_from_support,
    find_contacts,
    find_pair_from_cumulative_index,
    get_triangle_shape_from_mesh,
    mesh_vs_convex_midphase,
)
from ..geometry.support_function import (
    GenericShapeData,
    SupportMapDataProvider,
    pack_mesh_ptr,
)
from ..geometry.types import GeoType


@wp.kernel
def _extract_scale_kernel(
    geom_data: wp.array(dtype=wp.vec4),
    scale_array: wp.array(dtype=wp.vec3),
):
    """Extract scale (xyz) from geom_data (vec4) into scale_array (vec3)."""
    idx = wp.tid()
    data = geom_data[idx]
    scale_array[idx] = wp.vec3(data[0], data[1], data[2])


@wp.func
def write_contact_simple(
    contact_point_center: wp.vec3,
    contact_normal_a_to_b: wp.vec3,
    contact_distance: float,
    radius_eff_a: float,
    radius_eff_b: float,
    thickness_a: float,
    thickness_b: float,
    shape_a: int,
    shape_b: int,
    tid: int,
    rigid_contact_margin: float,
    contact_max: int,
    # outputs
    contact_count: wp.array(dtype=int),
    contact_pair: wp.array(dtype=wp.vec2i),
    contact_position: wp.array(dtype=wp.vec3),
    contact_normal: wp.array(dtype=wp.vec3),
    contact_penetration: wp.array(dtype=float),
    contact_tangent: wp.array(dtype=wp.vec3),
):
    """
    Write a contact to the output arrays using the simplified API format.

    Args:
        contact_point_center: Center point of contact in world space
        contact_normal_a_to_b: Contact normal pointing from shape A to B
        contact_distance: Distance between contact points
        radius_eff_a: Effective radius of shape A
        radius_eff_b: Effective radius of shape B
        thickness_a: Contact thickness for shape A
        thickness_b: Contact thickness for shape B
        shape_a: Shape A index
        shape_b: Shape B index
        tid: Thread ID
        rigid_contact_margin: Contact margin for rigid bodies
        contact_max: Maximum number of contacts
        contact_count: Array to track contact count
        contact_pair: Output array for shape pairs
        contact_position: Output array for contact positions (center point)
        contact_normal: Output array for contact normals
        contact_penetration: Output array for penetration depths
        contact_tangent: Output array for contact tangents
    """
    total_separation_needed = radius_eff_a + radius_eff_b + thickness_a + thickness_b

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

        contact_pair[index] = wp.vec2i(shape_a, shape_b)

        # Contact position is the center point
        contact_position[index] = contact_point_center

        # Normal pointing from shape A to shape B (negated to match convention)
        contact_normal[index] = -contact_normal_a_to_b

        # Penetration depth (negative if penetrating)
        contact_penetration[index] = d

        # Compute tangent vector (x-axis of local contact frame)
        # Use perpendicular to normal, defaulting to world x-axis if normal is parallel
        world_x = wp.vec3(1.0, 0.0, 0.0)
        normal = contact_normal[index]
        if wp.abs(wp.dot(normal, world_x)) > 0.99:
            world_x = wp.vec3(0.0, 1.0, 0.0)
        contact_tangent[index] = wp.normalize(world_x - wp.dot(world_x, normal) * normal)


@wp.func
def extract_shape_data_from_api(
    shape_idx: int,
    geom_transform: wp.array(dtype=wp.transform),
    geom_types: wp.array(dtype=int),
    geom_data: wp.array(dtype=wp.vec4),  # scale (xyz), thickness (w) or other data
    geom_source: wp.array(dtype=wp.uint64),
):
    """
    Extract shape data from the narrow phase API arrays.

    Args:
        shape_idx: Index of the shape
        geom_transform: World space transforms (already computed)
        geom_types: Shape types
        geom_data: Shape data (vec4 - scale xyz, thickness w)
        geom_source: Source pointers (mesh IDs etc.)

    Returns:
        tuple: (position, orientation, shape_data, scale, thickness)
    """
    # Get shape's world transform (already in world space)
    X_ws = geom_transform[shape_idx]

    position = wp.transform_get_translation(X_ws)
    orientation = wp.transform_get_rotation(X_ws)

    # Extract scale and thickness from geom_data
    # Assuming geom_data stores scale in xyz and thickness in w
    data = geom_data[shape_idx]
    scale = wp.vec3(data[0], data[1], data[2])
    thickness = data[3]

    # Create generic shape data
    result = GenericShapeData()
    result.shape_type = geom_types[shape_idx]
    result.scale = scale
    result.auxiliary = wp.vec3(0.0, 0.0, 0.0)

    # For CONVEX_MESH, pack the mesh pointer into auxiliary
    if geom_types[shape_idx] == int(GeoType.CONVEX_MESH):
        result.auxiliary = pack_mesh_ptr(geom_source[shape_idx])

    return position, orientation, result, scale, thickness


@wp.kernel(enable_backward=False)
def narrow_phase_kernel_gjk_mpr(
    candidate_pair: wp.array(dtype=wp.vec2i),
    num_candidate_pair: wp.array(dtype=int),
    geom_types: wp.array(dtype=int),
    geom_data: wp.array(dtype=wp.vec4),
    geom_transform: wp.array(dtype=wp.transform),
    geom_source: wp.array(dtype=wp.uint64),
    geom_cutoff: wp.array(dtype=float),
    rigid_contact_margin: float,
    contact_max: int,
    total_num_threads: int,
    # outputs
    contact_count: wp.array(dtype=int),
    contact_pair: wp.array(dtype=wp.vec2i),
    contact_position: wp.array(dtype=wp.vec3),
    contact_normal: wp.array(dtype=wp.vec3),
    contact_penetration: wp.array(dtype=float),
    contact_tangent: wp.array(dtype=wp.vec3),
    # mesh collision outputs (for mesh processing)
    shape_pairs_mesh: wp.array(dtype=wp.vec2i),
    shape_pairs_mesh_count: wp.array(dtype=int),
    # mesh-plane collision outputs
    shape_pairs_mesh_plane: wp.array(dtype=wp.vec2i),
    shape_pairs_mesh_plane_cumsum: wp.array(dtype=int),
    shape_pairs_mesh_plane_count: wp.array(dtype=int),
    mesh_plane_vertex_total_count: wp.array(dtype=int),
):
    """
    Narrow phase collision detection kernel using GJK/MPR.
    Processes candidate pairs from broad phase and generates contacts.
    """
    tid = wp.tid()

    num_work_items = wp.min(candidate_pair.shape[0], num_candidate_pair[0])

    for t in range(tid, num_work_items, total_num_threads):
        # Get shape pair
        pair = candidate_pair[t]
        shape_a = pair[0]
        shape_b = pair[1]

        # Safety: ignore self-collision pairs
        if shape_a == shape_b:
            continue

        # Validate shape indices
        if shape_a < 0 or shape_b < 0:
            continue

        # Get shape types
        type_a = geom_types[shape_a]
        type_b = geom_types[shape_b]

        # Sort shapes by type to ensure consistent collision handling order
        if type_a > type_b:
            # Swap shapes to maintain consistent ordering
            shape_a, shape_b = shape_b, shape_a
            type_a, type_b = type_b, type_a

        # Extract shape data for both shapes
        pos_a, quat_a, shape_data_a, scale_a, thickness_a = extract_shape_data_from_api(
            shape_a,
            geom_transform,
            geom_types,
            geom_data,
            geom_source,
        )

        pos_b, quat_b, shape_data_b, scale_b, thickness_b = extract_shape_data_from_api(
            shape_b,
            geom_transform,
            geom_types,
            geom_data,
            geom_source,
        )

        # Compute AABBs using support function for tight bounds
        # Since we don't have precomputed AABBs, we compute them on the fly
        data_provider = SupportMapDataProvider()
        aabb_a_lower, aabb_a_upper = compute_tight_aabb_from_support(shape_data_a, quat_a, pos_a, data_provider)
        aabb_b_lower, aabb_b_upper = compute_tight_aabb_from_support(shape_data_b, quat_b, pos_b, data_provider)

        # Add margin to AABBs using per-geometry cutoff
        cutoff_a = geom_cutoff[shape_a]
        cutoff_b = geom_cutoff[shape_b]
        margin_vec_a = wp.vec3(cutoff_a, cutoff_a, cutoff_a)
        margin_vec_b = wp.vec3(cutoff_b, cutoff_b, cutoff_b)
        aabb_a_lower = aabb_a_lower - margin_vec_a
        aabb_a_upper = aabb_a_upper + margin_vec_a
        aabb_b_lower = aabb_b_lower - margin_vec_b
        aabb_b_upper = aabb_b_upper + margin_vec_b

        # Pre-contact check handles mesh and plane special cases
        # Note: We pass geom_data/geom_source directly - pre_contact_check will extract scale as needed
        # For now, we'll skip the mesh handling in pre_contact_check and handle it separately
        # since populating scale arrays inside the kernel is inefficient
        # Instead, we'll detect mesh pairs here and handle them in separate kernels
        type_a = shape_data_a.shape_type
        type_b = shape_data_b.shape_type

        # Check for mesh collisions - skip regular contact generation for now
        # Mesh collisions will be handled in separate kernels
        if type_a == int(GeoType.MESH) or type_b == int(GeoType.MESH):
            # Check if this is a mesh-plane collision
            if type_a == int(GeoType.PLANE) and type_b == int(GeoType.MESH) and scale_a[0] == 0.0 and scale_a[1] == 0.0:
                # Mesh-plane collision - handle in separate kernel
                mesh_id = geom_source[shape_b]
                if mesh_id != wp.uint64(0):
                    mesh_obj = wp.mesh_get(mesh_id)
                    vertex_count = mesh_obj.points.shape[0]
                    mesh_plane_idx = wp.atomic_add(shape_pairs_mesh_plane_count, 0, 1)
                    if mesh_plane_idx < shape_pairs_mesh_plane.shape[0]:
                        shape_pairs_mesh_plane[mesh_plane_idx] = wp.vec2i(shape_b, shape_a)
                        cumulative_count_before = wp.atomic_add(mesh_plane_vertex_total_count, 0, vertex_count)
                        cumulative_count_inclusive = cumulative_count_before + vertex_count
                        shape_pairs_mesh_plane_cumsum[mesh_plane_idx] = cumulative_count_inclusive
            else:
                # Regular mesh collision - add to mesh collision buffer
                mesh_pair_idx = wp.atomic_add(shape_pairs_mesh_count, 0, 1)
                if mesh_pair_idx < shape_pairs_mesh.shape[0]:
                    shape_pairs_mesh[mesh_pair_idx] = pair
            continue

        # Check for infinite planes
        is_infinite_plane_a = (type_a == int(GeoType.PLANE)) and (scale_a[0] == 0.0 and scale_a[1] == 0.0)
        is_infinite_plane_b = (type_b == int(GeoType.PLANE)) and (scale_b[0] == 0.0 and scale_b[1] == 0.0)

        # Early return: both shapes are infinite planes
        if is_infinite_plane_a and is_infinite_plane_b:
            continue

        # Compute bounding spheres from AABBs
        bsphere_center_a = 0.5 * (aabb_a_lower + aabb_a_upper)
        bsphere_center_b = 0.5 * (aabb_b_lower + aabb_b_upper)
        bsphere_half_extents_a = 0.5 * (aabb_a_upper - aabb_a_lower)
        bsphere_half_extents_b = 0.5 * (aabb_b_upper - aabb_b_lower)
        bsphere_radius_a = wp.length(bsphere_half_extents_a)
        bsphere_radius_b = wp.length(bsphere_half_extents_b)

        # Check if infinite plane vs bounding sphere overlap - early rejection
        if is_infinite_plane_a or is_infinite_plane_b:
            # Check plane vs sphere overlap
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

            plane_normal = wp.quat_rotate(plane_quat, wp.vec3(0.0, 0.0, 1.0))
            center_dist = wp.dot(other_center - plane_pos, plane_normal)
            if center_dist > other_radius:
                continue

        # Use per-geometry cutoff for contact detection
        # find_contacts expects a scalar margin, so we use max of the two cutoffs
        cutoff_a = geom_cutoff[shape_a]
        cutoff_b = geom_cutoff[shape_b]
        margin = wp.max(cutoff_a, cutoff_b)

        # Compute contacts using GJK/MPR
        count, normal, signed_distances, points, radius_eff_a, radius_eff_b = find_contacts(
            pos_a,
            pos_b,
            quat_a,
            quat_b,
            shape_data_a,
            shape_data_b,
            is_infinite_plane_a,
            is_infinite_plane_b,
            bsphere_radius_a,
            bsphere_radius_b,
            margin,  # Use per-geometry cutoff instead of rigid_contact_margin
        )

        # Write contacts
        for id in range(count):
            write_contact_simple(
                points[id],
                normal,
                signed_distances[id],
                radius_eff_a,
                radius_eff_b,
                thickness_a,
                thickness_b,
                shape_a,
                shape_b,
                t,
                margin,  # Use per-geometry cutoff instead of rigid_contact_margin
                contact_max,
                contact_count,
                contact_pair,
                contact_position,
                contact_normal,
                contact_penetration,
                contact_tangent,
            )


@wp.kernel(enable_backward=False)
def narrow_phase_find_mesh_triangle_overlaps_kernel(
    geom_types: wp.array(dtype=int),
    geom_transform: wp.array(dtype=wp.transform),
    geom_source: wp.array(dtype=wp.uint64),
    geom_cutoff: wp.array(dtype=float),  # Per-geometry cutoff distances
    scale_array: wp.array(dtype=wp.vec3),  # Scale array extracted from geom_data
    shape_pairs_mesh: wp.array(dtype=wp.vec2i),
    shape_pairs_mesh_count: wp.array(dtype=int),
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
        type_a = geom_types[shape_a]
        type_b = geom_types[shape_b]

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
        mesh_id = geom_source[mesh_shape]
        if mesh_id == wp.uint64(0):
            return

        # Get mesh world transform
        X_mesh_ws = geom_transform[mesh_shape]

        # Get non-mesh shape world transform
        X_ws = geom_transform[non_mesh_shape]

        # Use per-geometry cutoff for the non-mesh shape
        # mesh_vs_convex_midphase uses rigid_contact_margin for the margin, but we'll use geom_cutoff
        # Note: mesh_vs_convex_midphase still expects a scalar margin, so we use max of the two cutoffs
        cutoff_non_mesh = geom_cutoff[non_mesh_shape]
        cutoff_mesh = geom_cutoff[mesh_shape]
        margin = wp.max(cutoff_non_mesh, cutoff_mesh)

        # Call mesh_vs_convex_midphase with the scale array and cutoff
        mesh_vs_convex_midphase(
            mesh_shape,
            non_mesh_shape,
            X_mesh_ws,
            X_ws,
            mesh_id,
            geom_types,
            scale_array,
            geom_source,
            margin,
            triangle_pairs,
            triangle_pairs_count,
        )


@wp.kernel(enable_backward=False)
def narrow_phase_process_mesh_triangle_contacts_kernel(
    geom_types: wp.array(dtype=int),
    geom_data: wp.array(dtype=wp.vec4),
    geom_transform: wp.array(dtype=wp.transform),
    geom_source: wp.array(dtype=wp.uint64),
    geom_cutoff: wp.array(dtype=float),  # Per-geometry cutoff distances
    triangle_pairs: wp.array(dtype=wp.vec3i),
    triangle_pairs_count: wp.array(dtype=int),
    contact_max: int,
    total_num_threads: int,
    # outputs
    contact_count: wp.array(dtype=int),
    contact_pair: wp.array(dtype=wp.vec2i),
    contact_position: wp.array(dtype=wp.vec3),
    contact_normal: wp.array(dtype=wp.vec3),
    contact_penetration: wp.array(dtype=float),
    contact_tangent: wp.array(dtype=wp.vec3),
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
        mesh_id_a = geom_source[shape_a]
        if mesh_id_a == wp.uint64(0):
            continue

        scale_data_a = geom_data[shape_a]
        mesh_scale_a = wp.vec3(scale_data_a[0], scale_data_a[1], scale_data_a[2])

        # Get mesh world transform for shape A
        X_mesh_ws_a = geom_transform[shape_a]

        # Extract triangle shape data from mesh
        shape_data_a, v0_world = get_triangle_shape_from_mesh(mesh_id_a, mesh_scale_a, X_mesh_ws_a, tri_idx)

        # Extract shape B data
        pos_b, quat_b, shape_data_b, scale_b, thickness_b = extract_shape_data_from_api(
            shape_b,
            geom_transform,
            geom_types,
            geom_data,
            geom_source,
        )

        # Set pos_a to be vertex A (origin of triangle in local frame)
        pos_a = v0_world
        quat_a = wp.quat_identity()  # Triangle has no orientation, use identity

        # Extract thickness for shape A
        thickness_a = geom_data[shape_a][3]

        # Use per-geometry cutoff for contact detection
        cutoff_a = geom_cutoff[shape_a]
        cutoff_b = geom_cutoff[shape_b]
        margin = wp.max(cutoff_a, cutoff_b)

        # Compute contacts using GJK/MPR
        count, normal, signed_distances, points, radius_eff_a, radius_eff_b = compute_gjk_mpr_contacts(
            shape_data_a,
            shape_data_b,
            quat_a,
            quat_b,
            pos_a,
            pos_b,
            margin,  # Use per-geometry cutoff instead of rigid_contact_margin
        )

        # Write contacts
        for contact_id in range(count):
            write_contact_simple(
                points[contact_id],
                normal,
                signed_distances[contact_id],
                radius_eff_a,
                radius_eff_b,
                thickness_a,
                thickness_b,
                shape_a,
                shape_b,
                tid,
                margin,  # Use per-geometry cutoff instead of rigid_contact_margin
                contact_max,
                contact_count,
                contact_pair,
                contact_position,
                contact_normal,
                contact_penetration,
                contact_tangent,
            )


@wp.kernel(enable_backward=False)
def narrow_phase_process_mesh_plane_contacts_kernel(
    geom_types: wp.array(dtype=int),
    geom_data: wp.array(dtype=wp.vec4),
    geom_transform: wp.array(dtype=wp.transform),
    geom_source: wp.array(dtype=wp.uint64),
    geom_cutoff: wp.array(dtype=float),  # Per-geometry cutoff distances
    shape_pairs_mesh_plane: wp.array(dtype=wp.vec2i),
    shape_pairs_mesh_plane_cumsum: wp.array(dtype=int),
    shape_pairs_mesh_plane_count: wp.array(dtype=int),
    mesh_plane_vertex_total_count: wp.array(dtype=int),
    contact_max: int,
    total_num_threads: int,
    # outputs
    contact_count: wp.array(dtype=int),
    contact_pair: wp.array(dtype=wp.vec2i),
    contact_position: wp.array(dtype=wp.vec3),
    contact_normal: wp.array(dtype=wp.vec3),
    contact_penetration: wp.array(dtype=float),
    contact_tangent: wp.array(dtype=wp.vec3),
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
        mesh_id = geom_source[mesh_shape]
        if mesh_id == wp.uint64(0):
            continue

        mesh_obj = wp.mesh_get(mesh_id)
        if vertex_idx >= mesh_obj.points.shape[0]:
            continue

        # Get mesh world transform
        X_mesh_ws = geom_transform[mesh_shape]

        # Get plane world transform
        X_plane_ws = geom_transform[plane_shape]

        # Get vertex position in mesh local space and transform to world space
        scale_data = geom_data[mesh_shape]
        mesh_scale = wp.vec3(scale_data[0], scale_data[1], scale_data[2])
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

        # Extract thickness values
        thickness_mesh = geom_data[mesh_shape][3]
        thickness_plane = geom_data[plane_shape][3]
        total_thickness = thickness_mesh + thickness_plane

        # Use per-geometry cutoff instead of rigid_contact_margin
        cutoff_mesh = geom_cutoff[mesh_shape]
        cutoff_plane = geom_cutoff[plane_shape]
        margin = wp.max(cutoff_mesh, cutoff_plane)

        # Treat plane as a half-space: generate contact for all vertices on or below the plane
        # (distance < margin means vertex is close to or penetrating the plane)
        if distance < margin + total_thickness:
            # Write contact
            # Note: write_contact_simple expects contact_normal_a_to_b pointing FROM mesh TO plane (downward)
            # plane_normal points upward, so we need to negate it
            write_contact_simple(
                (vertex_world + point_on_plane) * 0.5,  # contact_point_center
                -plane_normal,  # contact_normal_a_to_b (from mesh to plane, pointing downward)
                distance,  # contact_distance
                0.0,  # radius_eff_a (mesh has no effective radius)
                0.0,  # radius_eff_b (plane has no effective radius)
                thickness_mesh,  # thickness_a
                thickness_plane,  # thickness_b
                mesh_shape,  # shape_a
                plane_shape,  # shape_b
                task_id,  # tid
                margin,  # Use per-geometry cutoff instead of rigid_contact_margin
                contact_max,
                contact_count,
                contact_pair,
                contact_position,
                contact_normal,
                contact_penetration,
                contact_tangent,
            )


class NarrowPhase:
    def launch(
        self,
        candidate_pair: wp.array(dtype=wp.vec2i, ndim=1),  # Maybe colliding pairs
        num_candidate_pair: wp.array(dtype=wp.int32, ndim=1),  # Size one array
        geom_types: wp.array(dtype=wp.int32, ndim=1),  # All geom types, pairs index into it
        geom_data: wp.array(dtype=wp.vec4, ndim=1),  # Geom data (scale xyz, thickness w)
        geom_transform: wp.array(dtype=wp.transform, ndim=1),  # In world space
        geom_source: wp.array(dtype=wp.uint64, ndim=1),  # The index into the source array, type define by geom_types
        geom_cutoff: wp.array(dtype=wp.float32, ndim=1),  # per-geom (take the max)
    # Outputs
        contact_pair: wp.array(dtype=wp.vec2i),
    contact_position: wp.array(dtype=wp.vec3),
        contact_normal: wp.array(
            dtype=wp.vec3
        ),  # Pointing from pairId.x to pairId.y, represents z axis of local contact frame
        contact_penetration: wp.array(dtype=float),  # negative if bodies overlap
        contact_tangent: wp.array(dtype=wp.vec3),  # Represents x axis of local contact frame
        contact_count: wp.array(dtype=int),  # Number of active contacts after narrow
    device=None,  # Device to launch on
        rigid_contact_margin: float = 0.01,  # Contact margin for rigid bodies
    ):
        """
        Launch narrow phase collision detection on candidate pairs from broad phase.

        Args:
            candidate_pair: Array of potentially colliding shape pairs from broad phase
            num_candidate_pair: Single-element array containing the number of candidate pairs
            geom_types: Array of geometry types for all shapes
            geom_data: Array of vec4 containing scale (xyz) and thickness (w) for each shape
            geom_transform: Array of world-space transforms for each shape
            geom_source: Array of source pointers (mesh IDs, etc.) for each shape
            geom_cutoff: Array of cutoff distances for each shape
            contact_pair: Output array for contact shape pairs
            contact_position: Output array for contact positions (center point)
            contact_normal: Output array for contact normals
            contact_penetration: Output array for penetration depths
            contact_tangent: Output array for contact tangents
            contact_count: Output array (single element) for contact count
            device: Device to launch on
            rigid_contact_margin: Contact margin for rigid bodies (default 0.01)
        """
        if device is None:
            device = candidate_pair.device

        contact_max = contact_pair.shape[0]

        # Clear contact count
        contact_count.zero_()

        # Extract scale array from geom_data for mesh processing
        # mesh_vs_convex_midphase requires a vec3 scale array
        num_shapes = geom_types.shape[0]
        scale_array = wp.zeros(num_shapes, dtype=wp.vec3, device=device)
        # Extract scale from geom_data (xyz) into scale_array
        wp.launch(
            kernel=_extract_scale_kernel,
            dim=num_shapes,
            inputs=[geom_data],
            outputs=[scale_array],
            device=device,
        )

        # Allocate temporary buffers for mesh processing
        # These would ideally be pre-allocated, but for now we allocate them here
        max_pairs = candidate_pair.shape[0]

        with wp.ScopedDevice(device):
            # Buffers for mesh collision handling
            shape_pairs_mesh = wp.zeros(max_pairs, dtype=wp.vec2i, device=device)
            shape_pairs_mesh_count = wp.zeros(1, dtype=wp.int32, device=device)

            # Conservative estimate for triangle pairs
            max_triangle_pairs = 1000000
            triangle_pairs = wp.zeros(max_triangle_pairs, dtype=wp.vec3i, device=device)
            triangle_pairs_count = wp.zeros(1, dtype=wp.int32, device=device)

            # Buffers for mesh-plane collision handling
            shape_pairs_mesh_plane = wp.zeros(max_pairs, dtype=wp.vec2i, device=device)
            shape_pairs_mesh_plane_count = wp.zeros(1, dtype=wp.int32, device=device)
            shape_pairs_mesh_plane_cumsum = wp.zeros(max_pairs, dtype=wp.int32, device=device)
            mesh_plane_vertex_total_count = wp.zeros(1, dtype=wp.int32, device=device)

        # Use fixed thread count for kernel launches
        block_dim = 128
        total_num_threads = block_dim * 1024

        # Launch main narrow phase kernel
        wp.launch(
            kernel=narrow_phase_kernel_gjk_mpr,
            dim=total_num_threads,
            inputs=[
                candidate_pair,
                num_candidate_pair,
                geom_types,
                geom_data,
                geom_transform,
                geom_source,
                geom_cutoff,
                rigid_contact_margin,
                contact_max,
                total_num_threads,
            ],
            outputs=[
                contact_count,
                contact_pair,
                contact_position,
                contact_normal,
                contact_penetration,
                contact_tangent,
                shape_pairs_mesh,
                shape_pairs_mesh_count,
                shape_pairs_mesh_plane,
                shape_pairs_mesh_plane_cumsum,
                shape_pairs_mesh_plane_count,
                mesh_plane_vertex_total_count,
            ],
            device=device,
            block_dim=block_dim,
        )

        # Launch mesh-plane contact processing kernel
        wp.launch(
            kernel=narrow_phase_process_mesh_plane_contacts_kernel,
            dim=total_num_threads,
            inputs=[
                geom_types,
                geom_data,
                geom_transform,
                geom_source,
                geom_cutoff,
                shape_pairs_mesh_plane,
                shape_pairs_mesh_plane_cumsum,
                shape_pairs_mesh_plane_count,
                mesh_plane_vertex_total_count,
                contact_max,
                total_num_threads,
            ],
            outputs=[
                contact_count,
                contact_pair,
                contact_position,
                contact_normal,
                contact_penetration,
                contact_tangent,
            ],
            device=device,
            block_dim=block_dim,
        )

        # Launch mesh triangle overlap detection kernel
        num_tile_blocks = 1024
        tile_size = 128
        second_dim = tile_size if ENABLE_TILE_BVH_QUERY else 1
        wp.launch(
            kernel=narrow_phase_find_mesh_triangle_overlaps_kernel,
            dim=[num_tile_blocks, second_dim],
            inputs=[
                geom_types,
                geom_transform,
                geom_source,
                geom_cutoff,
                scale_array,
                shape_pairs_mesh,
                shape_pairs_mesh_count,
                num_tile_blocks,  # Use num_tile_blocks as total_num_threads for tiled kernel
            ],
            outputs=[
                triangle_pairs,
                triangle_pairs_count,
            ],
            device=device,
            block_dim=tile_size,
        )

        # Launch mesh triangle contact processing kernel
        wp.launch(
            kernel=narrow_phase_process_mesh_triangle_contacts_kernel,
            dim=total_num_threads,
            inputs=[
                geom_types,
                geom_data,
                geom_transform,
                geom_source,
                geom_cutoff,
                triangle_pairs,
                triangle_pairs_count,
                contact_max,
                total_num_threads,
            ],
            outputs=[
                contact_count,
                contact_pair,
                contact_position,
                contact_normal,
                contact_penetration,
                contact_tangent,
            ],
            device=device,
            block_dim=block_dim,
        )
