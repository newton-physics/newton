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

from functools import cache
from typing import Any

import warp as wp

from ..geometry.collision_core import (
    ENABLE_TILE_BVH_QUERY,
    build_pair_key2,
    build_pair_key3,
    compute_tight_aabb_from_support,
    create_compute_gjk_mpr_contacts,
    create_find_contacts,
    find_pair_from_cumulative_index,
    get_triangle_shape_from_mesh,
    mesh_vs_convex_midphase,
    pre_contact_check,
)
from ..geometry.contact_data import ContactData
from ..geometry.support_function import (
    GenericShapeData,
    SupportMapDataProvider,
    pack_mesh_ptr,
)
from ..geometry.types import GeoType


@wp.struct
class ContactWriterData:
    contact_max: int
    contact_count: wp.array(dtype=int)
    contact_pair: wp.array(dtype=wp.vec2i)
    contact_position: wp.array(dtype=wp.vec3)
    contact_normal: wp.array(dtype=wp.vec3)
    contact_penetration: wp.array(dtype=float)
    contact_tangent: wp.array(dtype=wp.vec3)
    # Contact matching arrays (optional)
    contact_pair_key: wp.array(dtype=wp.uint64)
    contact_key: wp.array(dtype=wp.uint32)


@wp.func
def write_contact_simple(
    contact_data: ContactData,
    writer_data: ContactWriterData,
):
    """
    Write a contact to the output arrays using the simplified API format.

    Args:
        contact_data: ContactData struct containing contact information (includes feature and feature_pair_key)
        writer_data: ContactWriterData struct containing output arrays (includes contact_pair_key and contact_key)
    """
    total_separation_needed = (
        contact_data.radius_eff_a + contact_data.radius_eff_b + contact_data.thickness_a + contact_data.thickness_b
    )

    # Distance calculation matching box_plane_collision
    contact_normal_a_to_b = wp.normalize(contact_data.contact_normal_a_to_b)

    a_contact_world = contact_data.contact_point_center - contact_normal_a_to_b * (
        0.5 * contact_data.contact_distance + contact_data.radius_eff_a
    )
    b_contact_world = contact_data.contact_point_center + contact_normal_a_to_b * (
        0.5 * contact_data.contact_distance + contact_data.radius_eff_b
    )

    diff = b_contact_world - a_contact_world
    distance = wp.dot(diff, contact_normal_a_to_b)
    d = distance - total_separation_needed

    if d < contact_data.margin:
        index = wp.atomic_add(writer_data.contact_count, 0, 1)
        if index >= writer_data.contact_max:
            # Reached buffer limit
            wp.atomic_add(writer_data.contact_count, 0, -1)
            return

        writer_data.contact_pair[index] = wp.vec2i(contact_data.shape_a, contact_data.shape_b)

        # Contact position is the center point
        writer_data.contact_position[index] = contact_data.contact_point_center

        # Normal pointing from shape A to shape B
        writer_data.contact_normal[index] = contact_normal_a_to_b

        # Penetration depth (negative if penetrating)
        writer_data.contact_penetration[index] = d

        # Compute tangent vector only if tangent array is non-empty
        if writer_data.contact_tangent.shape[0] > 0:
            # Compute tangent vector (x-axis of local contact frame)
            # Use perpendicular to normal, defaulting to world x-axis if normal is parallel
            world_x = wp.vec3(1.0, 0.0, 0.0)
            normal = contact_normal_a_to_b
            if wp.abs(wp.dot(normal, world_x)) > 0.99:
                world_x = wp.vec3(0.0, 1.0, 0.0)
            writer_data.contact_tangent[index] = wp.normalize(world_x - wp.dot(world_x, normal) * normal)

        # Write contact key only if contact_key array is non-empty
        if writer_data.contact_key.shape[0] > 0 and writer_data.contact_pair_key.shape[0] > 0:
            writer_data.contact_key[index] = contact_data.feature
            writer_data.contact_pair_key[index] = contact_data.feature_pair_key


@wp.func
def extract_shape_data(
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


@cache
def create_narrow_phase_kernel_gjk_mpr(external_aabb: bool, writer_func: Any):
    @wp.kernel(enable_backward=False)
    def narrow_phase_kernel_gjk_mpr(
        candidate_pair: wp.array(dtype=wp.vec2i),
        num_candidate_pair: wp.array(dtype=int),
        geom_types: wp.array(dtype=int),
        geom_data: wp.array(dtype=wp.vec4),
        geom_transform: wp.array(dtype=wp.transform),
        geom_source: wp.array(dtype=wp.uint64),
        geom_cutoff: wp.array(dtype=float),
        geom_collision_radius: wp.array(dtype=float),
        geom_aabb_lower: wp.array(dtype=wp.vec3),
        geom_aabb_upper: wp.array(dtype=wp.vec3),
        writer_data: Any,
        total_num_threads: int,
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
            pos_a, quat_a, shape_data_a, scale_a, thickness_a = extract_shape_data(
                shape_a,
                geom_transform,
                geom_types,
                geom_data,
                geom_source,
            )

            pos_b, quat_b, shape_data_b, scale_b, thickness_b = extract_shape_data(
                shape_b,
                geom_transform,
                geom_types,
                geom_data,
                geom_source,
            )

            if wp.static(external_aabb):
                aabb_a_lower = geom_aabb_lower[shape_a]
                aabb_a_upper = geom_aabb_upper[shape_a]
                aabb_b_lower = geom_aabb_lower[shape_b]
                aabb_b_upper = geom_aabb_upper[shape_b]
            if wp.static(not external_aabb):
                # Compute AABBs - use special handling for infinite planes and meshes
                # This matches the approach in collide_unified.py compute_shape_aabbs
                cutoff_a = geom_cutoff[shape_a]
                cutoff_b = geom_cutoff[shape_b]
                margin_vec_a = wp.vec3(cutoff_a, cutoff_a, cutoff_a)
                margin_vec_b = wp.vec3(cutoff_b, cutoff_b, cutoff_b)

                # Check if shape A is an infinite plane, mesh, or SDF
                is_infinite_plane_a = (type_a == int(GeoType.PLANE)) and (scale_a[0] == 0.0 and scale_a[1] == 0.0)
                is_mesh_a = type_a == int(GeoType.MESH)
                is_sdf_a = type_a == int(GeoType.SDF)

                if is_infinite_plane_a or is_mesh_a or is_sdf_a:
                    # Use conservative bounding sphere approach for infinite planes, meshes, and SDFs
                    radius_a = geom_collision_radius[shape_a]
                    half_extents_a = wp.vec3(radius_a, radius_a, radius_a)
                    aabb_a_lower = pos_a - half_extents_a - margin_vec_a
                    aabb_a_upper = pos_a + half_extents_a + margin_vec_a
                else:
                    # Use support function to compute tight AABB
                    data_provider = SupportMapDataProvider()
                    aabb_a_lower, aabb_a_upper = compute_tight_aabb_from_support(
                        shape_data_a, quat_a, pos_a, data_provider
                    )
                    aabb_a_lower = aabb_a_lower - margin_vec_a
                    aabb_a_upper = aabb_a_upper + margin_vec_a

                # Check if shape B is an infinite plane, mesh, or SDF
                is_infinite_plane_b = (type_b == int(GeoType.PLANE)) and (scale_b[0] == 0.0 and scale_b[1] == 0.0)
                is_mesh_b = type_b == int(GeoType.MESH)
                is_sdf_b = type_b == int(GeoType.SDF)

                if is_infinite_plane_b or is_mesh_b or is_sdf_b:
                    # Use conservative bounding sphere approach for infinite planes, meshes, and SDFs
                    radius_b = geom_collision_radius[shape_b]
                    half_extents_b = wp.vec3(radius_b, radius_b, radius_b)
                    aabb_b_lower = pos_b - half_extents_b - margin_vec_b
                    aabb_b_upper = pos_b + half_extents_b + margin_vec_b
                else:
                    # Use support function to compute tight AABB
                    data_provider = SupportMapDataProvider()
                    aabb_b_lower, aabb_b_upper = compute_tight_aabb_from_support(
                        shape_data_b, quat_b, pos_b, data_provider
                    )
                    aabb_b_lower = aabb_b_lower - margin_vec_b
                    aabb_b_upper = aabb_b_upper + margin_vec_b

            # Use pre_contact_check to handle mesh and plane special cases
            # This avoids code duplication with collide_unified.py
            skip_pair, is_infinite_plane_a, is_infinite_plane_b, bsphere_radius_a, bsphere_radius_b = pre_contact_check(
                shape_a,
                shape_b,
                pos_a,
                pos_b,
                quat_a,
                quat_b,
                shape_data_a,
                shape_data_b,
                aabb_a_lower,
                aabb_a_upper,
                aabb_b_lower,
                aabb_b_upper,
                pair,
                geom_source[shape_a],
                geom_source[shape_b],
                shape_pairs_mesh,
                shape_pairs_mesh_count,
                shape_pairs_mesh_plane,
                shape_pairs_mesh_plane_cumsum,
                shape_pairs_mesh_plane_count,
                mesh_plane_vertex_total_count,
            )
            if skip_pair:
                continue

            # Use per-geometry cutoff for contact detection
            # find_contacts expects a scalar margin, so we use max of the two cutoffs
            cutoff_a = geom_cutoff[shape_a]
            cutoff_b = geom_cutoff[shape_b]
            margin = wp.max(cutoff_a, cutoff_b)

            # Find and write contacts using GJK/MPR
            wp.static(create_find_contacts(writer_func))(
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
                margin,
                shape_a,
                shape_b,
                thickness_a,
                thickness_b,
                writer_data,
            )

    return narrow_phase_kernel_gjk_mpr


@wp.kernel(enable_backward=False)
def narrow_phase_find_mesh_triangle_overlaps_kernel(
    geom_types: wp.array(dtype=int),
    geom_transform: wp.array(dtype=wp.transform),
    geom_source: wp.array(dtype=wp.uint64),
    geom_cutoff: wp.array(dtype=float),  # Per-geometry cutoff distances
    geom_data: wp.array(dtype=wp.vec4),  # Geom data (scale xyz, thickness w)
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
        # Note: mesh_vs_convex_midphase expects a scalar margin, so we use max of the two cutoffs
        cutoff_non_mesh = geom_cutoff[non_mesh_shape]
        cutoff_mesh = geom_cutoff[mesh_shape]
        margin = wp.max(cutoff_non_mesh, cutoff_mesh)

        # Call mesh_vs_convex_midphase with the geom_data and cutoff
        mesh_vs_convex_midphase(
            mesh_shape,
            non_mesh_shape,
            X_mesh_ws,
            X_ws,
            mesh_id,
            geom_types,
            geom_data,
            geom_source,
            margin,
            triangle_pairs,
            triangle_pairs_count,
        )


@cache
def create_narrow_phase_process_mesh_triangle_contacts_kernel(writer_func: Any):
    @wp.kernel(enable_backward=False)
    def narrow_phase_process_mesh_triangle_contacts_kernel(
        geom_types: wp.array(dtype=int),
        geom_data: wp.array(dtype=wp.vec4),
        geom_transform: wp.array(dtype=wp.transform),
        geom_source: wp.array(dtype=wp.uint64),
        geom_cutoff: wp.array(dtype=float),  # Per-geometry cutoff distances
        triangle_pairs: wp.array(dtype=wp.vec3i),
        triangle_pairs_count: wp.array(dtype=int),
        writer_data: Any,
        total_num_threads: int,
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
            pos_b, quat_b, shape_data_b, _scale_b, thickness_b = extract_shape_data(
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

            # Build pair key including triangle index for unique contact tracking
            pair_key = build_pair_key3(wp.uint32(shape_a), wp.uint32(shape_b), wp.uint32(tri_idx))

            # Compute and write contacts using GJK/MPR with standard post-processing
            wp.static(create_compute_gjk_mpr_contacts(writer_func))(
                shape_data_a,
                shape_data_b,
                quat_a,
                quat_b,
                pos_a,
                pos_b,
                margin,
                shape_a,
                shape_b,
                thickness_a,
                thickness_b,
                writer_data,
                pair_key,
            )

    return narrow_phase_process_mesh_triangle_contacts_kernel


@cache
def create_narrow_phase_process_mesh_plane_contacts_kernel(writer_func: Any):
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
        writer_data: Any,
        total_num_threads: int,
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

            # Use per-geometry cutoff for contact detection
            cutoff_mesh = geom_cutoff[mesh_shape]
            cutoff_plane = geom_cutoff[plane_shape]
            margin = wp.max(cutoff_mesh, cutoff_plane)

            # Treat plane as a half-space: generate contact for all vertices on or below the plane
            # (distance < margin means vertex is close to or penetrating the plane)
            if distance < margin + total_thickness:
                # Write contact
                # Note: write_contact_simple expects contact_normal_a_to_b pointing FROM mesh TO plane (downward)
                # plane_normal points upward, so we need to negate it
                pair_key = build_pair_key2(wp.uint32(mesh_shape), wp.uint32(plane_shape))

                contact_data = ContactData()
                contact_data.contact_point_center = (vertex_world + point_on_plane) * 0.5
                contact_data.contact_normal_a_to_b = -plane_normal
                contact_data.contact_distance = distance
                contact_data.radius_eff_a = 0.0  # mesh has no effective radius
                contact_data.radius_eff_b = 0.0  # plane has no effective radius
                contact_data.thickness_a = thickness_mesh
                contact_data.thickness_b = thickness_plane
                contact_data.shape_a = mesh_shape
                contact_data.shape_b = plane_shape
                contact_data.margin = margin
                contact_data.feature = wp.uint32(vertex_idx + 1)
                contact_data.feature_pair_key = pair_key

                writer_func(contact_data, writer_data)

    return narrow_phase_process_mesh_plane_contacts_kernel


class NarrowPhase:
    def __init__(
        self,
        max_candidate_pairs: int,
        max_triangle_pairs: int = 1000000,
        device=None,
        geom_aabb_lower: wp.array(dtype=wp.vec3) | None = None,
        geom_aabb_upper: wp.array(dtype=wp.vec3) | None = None,
        contact_writer_warp_func: Any | None = None,
    ):
        """
        Initialize NarrowPhase with pre-allocated buffers.

        Args:
            max_candidate_pairs: Maximum number of candidate pairs from broad phase
            max_triangle_pairs: Maximum number of mesh triangle pairs (conservative estimate)
            device: Device to allocate buffers on
            geom_aabb_lower: Optional external AABB lower bounds array (if provided, AABBs won't be computed internally)
            geom_aabb_upper: Optional external AABB upper bounds array (if provided, AABBs won't be computed internally)
            contact_writer_warp_func: Optional custom contact writer function (first arg: ContactData, second arg: custom struct type)
        """
        self.max_candidate_pairs = max_candidate_pairs
        self.max_triangle_pairs = max_triangle_pairs
        self.device = device

        # Determine if we're using external AABBs
        self.external_aabb = geom_aabb_lower is not None and geom_aabb_upper is not None

        if self.external_aabb:
            # Use provided AABB arrays
            self.geom_aabb_lower = geom_aabb_lower
            self.geom_aabb_upper = geom_aabb_upper
        else:
            # Create empty AABB arrays (won't be used)
            with wp.ScopedDevice(device):
                self.geom_aabb_lower = wp.zeros(0, dtype=wp.vec3, device=device)
                self.geom_aabb_upper = wp.zeros(0, dtype=wp.vec3, device=device)

        # Determine the writer function
        if contact_writer_warp_func is None:
            writer_func = write_contact_simple
        else:
            writer_func = contact_writer_warp_func

        # Create the appropriate kernel variants
        self.narrow_phase_kernel = create_narrow_phase_kernel_gjk_mpr(self.external_aabb, writer_func)
        self.mesh_triangle_contacts_kernel = create_narrow_phase_process_mesh_triangle_contacts_kernel(writer_func)
        self.mesh_plane_contacts_kernel = create_narrow_phase_process_mesh_plane_contacts_kernel(writer_func)

        # Pre-allocate all intermediate buffers
        with wp.ScopedDevice(device):
            # Buffers for mesh collision handling
            self.shape_pairs_mesh = wp.zeros(max_candidate_pairs, dtype=wp.vec2i, device=device)
            self.shape_pairs_mesh_count = wp.zeros(1, dtype=wp.int32, device=device)

            # Buffers for triangle pairs
            self.triangle_pairs = wp.zeros(max_triangle_pairs, dtype=wp.vec3i, device=device)
            self.triangle_pairs_count = wp.zeros(1, dtype=wp.int32, device=device)

            # Buffers for mesh-plane collision handling
            self.shape_pairs_mesh_plane = wp.zeros(max_candidate_pairs, dtype=wp.vec2i, device=device)
            self.shape_pairs_mesh_plane_count = wp.zeros(1, dtype=wp.int32, device=device)
            self.shape_pairs_mesh_plane_cumsum = wp.zeros(max_candidate_pairs, dtype=wp.int32, device=device)
            self.mesh_plane_vertex_total_count = wp.zeros(1, dtype=wp.int32, device=device)

            # Empty tangent array for when tangent computation is disabled
            self.empty_tangent = wp.zeros(0, dtype=wp.vec3, device=device)

            # Empty contact_pair_key array for when contact pair key collection is disabled
            self.empty_contact_pair_key = wp.zeros(0, dtype=wp.uint64, device=device)

            # Empty contact_key array for when contact key collection is disabled
            self.empty_contact_key = wp.zeros(0, dtype=wp.uint32, device=device)

        # Fixed thread count for kernel launches
        self.block_dim = 128
        gpu_thread_limit = 1024 * 1024 * 4
        max_blocks_limit = gpu_thread_limit // self.block_dim
        candidate_blocks = (max_candidate_pairs + self.block_dim - 1) // self.block_dim
        num_blocks = max(1024, min(candidate_blocks, max_blocks_limit))
        self.total_num_threads = self.block_dim * num_blocks
        self.num_tile_blocks = num_blocks
        self.tile_size = 128

    def launch_custom_write(
        self,
        candidate_pair: wp.array(dtype=wp.vec2i, ndim=1),  # Maybe colliding pairs
        num_candidate_pair: wp.array(dtype=wp.int32, ndim=1),  # Size one array
        geom_types: wp.array(dtype=wp.int32, ndim=1),  # All geom types, pairs index into it
        geom_data: wp.array(dtype=wp.vec4, ndim=1),  # Geom data (scale xyz, thickness w)
        geom_transform: wp.array(dtype=wp.transform, ndim=1),  # In world space
        geom_source: wp.array(dtype=wp.uint64, ndim=1),  # The index into the source array, type define by geom_types
        geom_cutoff: wp.array(dtype=wp.float32, ndim=1),  # per-geom (take the max)
        geom_collision_radius: wp.array(dtype=wp.float32, ndim=1),  # per-geom collision radius for AABB fallback
        writer_data: Any,
        device=None,  # Device to launch on
    ):
        """
        Launch narrow phase collision detection with a custom contact writer struct.

        Args:
            candidate_pair: Array of potentially colliding shape pairs from broad phase
            num_candidate_pair: Single-element array containing the number of candidate pairs
            geom_types: Array of geometry types for all shapes
            geom_data: Array of vec4 containing scale (xyz) and thickness (w) for each shape
            geom_transform: Array of world-space transforms for each shape
            geom_source: Array of source pointers (mesh IDs, etc.) for each shape
            geom_cutoff: Array of cutoff distances for each shape
            geom_collision_radius: Array of collision radii for each shape (for AABB fallback for planes/meshes)
            writer_data: Custom struct instance for contact writing (type must match the custom writer function)
            device: Device to launch on
        """
        if device is None:
            device = self.device if self.device is not None else candidate_pair.device

        # Clear all counters
        self.shape_pairs_mesh_count.zero_()
        self.triangle_pairs_count.zero_()
        self.shape_pairs_mesh_plane_count.zero_()
        self.mesh_plane_vertex_total_count.zero_()

        # Launch main narrow phase kernel (using the appropriate kernel variant)
        wp.launch(
            kernel=self.narrow_phase_kernel,
            dim=self.total_num_threads,
            inputs=[
                candidate_pair,
                num_candidate_pair,
                geom_types,
                geom_data,
                geom_transform,
                geom_source,
                geom_cutoff,
                geom_collision_radius,
                self.geom_aabb_lower,
                self.geom_aabb_upper,
                writer_data,
                self.total_num_threads,
            ],
            outputs=[
                self.shape_pairs_mesh,
                self.shape_pairs_mesh_count,
                self.shape_pairs_mesh_plane,
                self.shape_pairs_mesh_plane_cumsum,
                self.shape_pairs_mesh_plane_count,
                self.mesh_plane_vertex_total_count,
            ],
            device=device,
            block_dim=self.block_dim,
        )

        # Launch mesh-plane contact processing kernel
        wp.launch(
            kernel=self.mesh_plane_contacts_kernel,
            dim=self.total_num_threads,
            inputs=[
                geom_types,
                geom_data,
                geom_transform,
                geom_source,
                geom_cutoff,
                self.shape_pairs_mesh_plane,
                self.shape_pairs_mesh_plane_cumsum,
                self.shape_pairs_mesh_plane_count,
                self.mesh_plane_vertex_total_count,
                writer_data,
                self.total_num_threads,
            ],
            device=device,
            block_dim=self.block_dim,
        )

        # Launch mesh triangle overlap detection kernel
        second_dim = self.tile_size if ENABLE_TILE_BVH_QUERY else 1
        wp.launch(
            kernel=narrow_phase_find_mesh_triangle_overlaps_kernel,
            dim=[self.num_tile_blocks, second_dim],
            inputs=[
                geom_types,
                geom_transform,
                geom_source,
                geom_cutoff,
                geom_data,
                self.shape_pairs_mesh,
                self.shape_pairs_mesh_count,
                self.num_tile_blocks,  # Use num_tile_blocks as total_num_threads for tiled kernel
            ],
            outputs=[
                self.triangle_pairs,
                self.triangle_pairs_count,
            ],
            device=device,
            block_dim=self.tile_size,
        )

        # Launch mesh triangle contact processing kernel
        wp.launch(
            kernel=self.mesh_triangle_contacts_kernel,
            dim=self.total_num_threads,
            inputs=[
                geom_types,
                geom_data,
                geom_transform,
                geom_source,
                geom_cutoff,
                self.triangle_pairs,
                self.triangle_pairs_count,
                writer_data,
                self.total_num_threads,
            ],
            device=device,
            block_dim=self.block_dim,
        )

    def launch(
        self,
        candidate_pair: wp.array(dtype=wp.vec2i, ndim=1),  # Maybe colliding pairs
        num_candidate_pair: wp.array(dtype=wp.int32, ndim=1),  # Size one array
        geom_types: wp.array(dtype=wp.int32, ndim=1),  # All geom types, pairs index into it
        geom_data: wp.array(dtype=wp.vec4, ndim=1),  # Geom data (scale xyz, thickness w)
        geom_transform: wp.array(dtype=wp.transform, ndim=1),  # In world space
        geom_source: wp.array(dtype=wp.uint64, ndim=1),  # The index into the source array, type define by geom_types
        geom_cutoff: wp.array(dtype=wp.float32, ndim=1),  # per-geom (take the max)
        geom_collision_radius: wp.array(dtype=wp.float32, ndim=1),  # per-geom collision radius for AABB fallback
        # Outputs
        contact_pair: wp.array(dtype=wp.vec2i),
        contact_position: wp.array(dtype=wp.vec3),
        contact_normal: wp.array(
            dtype=wp.vec3
        ),  # Pointing from pairId.x to pairId.y, represents z axis of local contact frame
        contact_penetration: wp.array(dtype=float),  # negative if bodies overlap
        contact_count: wp.array(dtype=int),  # Number of active contacts after narrow
        contact_tangent: wp.array(dtype=wp.vec3)
        | None = None,  # Represents x axis of local contact frame (None to disable)
        contact_pair_key: wp.array(dtype=wp.uint64) | None = None,  # Contact pair keys (None to disable)
        contact_key: wp.array(dtype=wp.uint32) | None = None,  # Contact feature keys (None to disable)
        device=None,  # Device to launch on
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
            geom_collision_radius: Array of collision radii for each shape (for AABB fallback for planes/meshes)
            contact_pair: Output array for contact shape pairs
            contact_position: Output array for contact positions (center point)
            contact_normal: Output array for contact normals
            contact_penetration: Output array for penetration depths
            contact_tangent: Output array for contact tangents, or None to disable tangent computation
            contact_key: Output array for contact feature keys, or None to disable key collection
            contact_count: Output array (single element) for contact count
            device: Device to launch on
        """
        if device is None:
            device = self.device if self.device is not None else candidate_pair.device

        contact_max = contact_pair.shape[0]

        # Handle optional tangent array - use empty array if None
        if contact_tangent is None:
            contact_tangent = self.empty_tangent

        # Handle optional contact_pair_key array - use empty array if None
        if contact_pair_key is None:
            contact_pair_key = self.empty_contact_pair_key

        # Handle optional contact_key array - use empty array if None
        if contact_key is None:
            contact_key = self.empty_contact_key

        # Clear all counters and contact count
        contact_count.zero_()
        self.shape_pairs_mesh_count.zero_()
        self.triangle_pairs_count.zero_()
        self.shape_pairs_mesh_plane_count.zero_()
        self.mesh_plane_vertex_total_count.zero_()

        # Create ContactWriterData struct
        writer_data = ContactWriterData()
        writer_data.contact_max = contact_max
        writer_data.contact_count = contact_count
        writer_data.contact_pair = contact_pair
        writer_data.contact_position = contact_position
        writer_data.contact_normal = contact_normal
        writer_data.contact_penetration = contact_penetration
        writer_data.contact_tangent = contact_tangent
        writer_data.contact_pair_key = contact_pair_key
        writer_data.contact_key = contact_key

        # Delegate to launch_custom_write
        self.launch_custom_write(
            candidate_pair,
            num_candidate_pair,
            geom_types,
            geom_data,
            geom_transform,
            geom_source,
            geom_cutoff,
            geom_collision_radius,
            writer_data,
            device,
        )
