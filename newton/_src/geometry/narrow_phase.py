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

import warnings
from typing import Any

import warp as wp

from newton._src.core.types import MAXVAL

from ..geometry.collision_core import (
    ENABLE_TILE_BVH_QUERY,
    check_infinite_plane_bsphere_overlap,
    compute_bounding_sphere_from_aabb,
    compute_tight_aabb_from_support,
    create_compute_gjk_mpr_contacts,
    create_find_contacts,
    get_triangle_shape_from_mesh,
    mesh_vs_convex_midphase,
)
from ..geometry.collision_primitive import (
    collide_capsule_capsule,
    collide_plane_box,
    collide_plane_capsule,
    collide_plane_cylinder,
    collide_plane_ellipsoid,
    collide_plane_sphere,
    collide_sphere_box,
    collide_sphere_capsule,
    collide_sphere_cylinder,
    collide_sphere_sphere,
)
from ..geometry.contact_data import ContactData, contact_passes_margin_check
from ..geometry.contact_reduction import (
    ContactReductionFunctions,
    ContactStruct,
    compute_voxel_index,
    synchronize,
)
from ..geometry.contact_reduction_global import (
    GlobalContactReducer,
    create_export_reduced_contacts_kernel,
    mesh_triangle_contacts_to_reducer_kernel,
    reduce_buffered_contacts_kernel,
)
from ..geometry.flags import ShapeFlags
from ..geometry.sdf_contact import create_narrow_phase_process_mesh_mesh_contacts_kernel
from ..geometry.sdf_hydroelastic import HydroelasticSDF as _HydroelasticSDF
from ..geometry.sdf_utils import SDFData
from ..geometry.support_function import (
    SupportMapDataProvider,
    extract_shape_data,
)
from ..geometry.types import GeoType
from ..utils.heightfield import HeightfieldData, get_triangle_from_heightfield_cell


@wp.struct
class ContactWriterData:
    contact_max: int
    contact_count: wp.array(dtype=int)
    contact_pair: wp.array(dtype=wp.vec2i)
    contact_position: wp.array(dtype=wp.vec3)
    contact_normal: wp.array(dtype=wp.vec3)
    contact_penetration: wp.array(dtype=float)
    contact_tangent: wp.array(dtype=wp.vec3)


@wp.func
def write_contact_simple(
    contact_data: ContactData,
    writer_data: ContactWriterData,
    output_index: int,
):
    """
    Write a contact to the output arrays using the simplified API format.

    Args:
        contact_data: ContactData struct containing contact information
        writer_data: ContactWriterData struct containing output arrays
        output_index: If -1, use atomic_add to get the next available index if contact distance is less than margin. If >= 0, use this index directly and skip margin check.
    """
    total_separation_needed = (
        contact_data.radius_eff_a + contact_data.radius_eff_b + contact_data.thickness_a + contact_data.thickness_b
    )

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

    if output_index < 0:
        if d >= contact_data.margin:
            return
        index = wp.atomic_add(writer_data.contact_count, 0, 1)
        if index >= writer_data.contact_max:
            wp.atomic_add(writer_data.contact_count, 0, -1)
            return
    else:
        index = output_index

    writer_data.contact_pair[index] = wp.vec2i(contact_data.shape_a, contact_data.shape_b)
    writer_data.contact_position[index] = contact_data.contact_point_center
    writer_data.contact_normal[index] = contact_normal_a_to_b
    writer_data.contact_penetration[index] = d

    if writer_data.contact_tangent.shape[0] > 0:
        world_x = wp.vec3(1.0, 0.0, 0.0)
        normal = contact_normal_a_to_b
        if wp.abs(wp.dot(normal, world_x)) > 0.99:
            world_x = wp.vec3(0.0, 1.0, 0.0)
        writer_data.contact_tangent[index] = wp.normalize(world_x - wp.dot(world_x, normal) * normal)


def create_narrow_phase_primitive_kernel(writer_func: Any):
    """
    Create a kernel for fast analytical collision detection of primitive shapes.

    This kernel handles lightweight primitive pairs (sphere-sphere, sphere-capsule,
    capsule-capsule, plane-sphere, plane-capsule) using direct analytical formulas
    instead of iterative GJK/MPR. Remaining pairs are routed to specialized buffers
    for mesh handling or to the GJK/MPR kernel for complex convex pairs.

    Args:
        writer_func: Contact writer function (e.g., write_contact_simple)

    Returns:
        A warp kernel for primitive collision detection
    """

    @wp.kernel(enable_backward=False, module="unique")
    def narrow_phase_primitive_kernel(
        candidate_pair: wp.array(dtype=wp.vec2i),
        candidate_pair_count: wp.array(dtype=int),
        shape_types: wp.array(dtype=int),
        shape_data: wp.array(dtype=wp.vec4),
        shape_transform: wp.array(dtype=wp.transform),
        shape_source: wp.array(dtype=wp.uint64),
        shape_contact_margin: wp.array(dtype=float),
        shape_flags: wp.array(dtype=wp.int32),
        writer_data: Any,
        total_num_threads: int,
        # Output: pairs that need GJK/MPR processing
        gjk_candidate_pairs: wp.array(dtype=wp.vec2i),
        gjk_candidate_pairs_count: wp.array(dtype=int),
        # Output: mesh collision pairs (for mesh processing)
        shape_pairs_mesh: wp.array(dtype=wp.vec2i),
        shape_pairs_mesh_count: wp.array(dtype=int),
        # Output: mesh-plane collision pairs
        shape_pairs_mesh_plane: wp.array(dtype=wp.vec2i),
        shape_pairs_mesh_plane_cumsum: wp.array(dtype=int),
        shape_pairs_mesh_plane_count: wp.array(dtype=int),
        mesh_plane_vertex_total_count: wp.array(dtype=int),
        # Output: mesh-mesh collision pairs
        shape_pairs_mesh_mesh: wp.array(dtype=wp.vec2i),
        shape_pairs_mesh_mesh_count: wp.array(dtype=int),
        # Output: sdf-sdf hydroelastic collision pairs
        shape_pairs_sdf_sdf: wp.array(dtype=wp.vec2i),
        shape_pairs_sdf_sdf_count: wp.array(dtype=int),
        # Output: heightfield collision pairs
        shape_pairs_heightfield: wp.array(dtype=wp.vec2i),
        shape_pairs_heightfield_count: wp.array(dtype=int),
    ):
        """
        Fast narrow phase kernel for primitive shape collisions.

        Handles sphere-sphere, sphere-capsule, capsule-capsule, plane-sphere, and
        plane-capsule collisions analytically. Routes mesh pairs and complex convex
        pairs to specialized processing pipelines.
        """
        tid = wp.tid()

        num_work_items = wp.min(candidate_pair.shape[0], candidate_pair_count[0])

        # Early exit if no work
        if num_work_items == 0:
            return

        for t in range(tid, num_work_items, total_num_threads):
            # Get shape pair
            pair = candidate_pair[t]
            shape_a = pair[0]
            shape_b = pair[1]

            # Safety: ignore self-collision and invalid pairs
            if shape_a == shape_b or shape_a < 0 or shape_b < 0:
                continue

            # Get shape types
            type_a = shape_types[shape_a]
            type_b = shape_types[shape_b]

            # Sort shapes by type to ensure consistent collision handling order
            if type_a > type_b:
                shape_a, shape_b = shape_b, shape_a
                type_a, type_b = type_b, type_a

            # Check if both shapes are hydroelastic - route to SDF-SDF pipeline
            is_hydro_a = (shape_flags[shape_a] & ShapeFlags.HYDROELASTIC) != 0
            is_hydro_b = (shape_flags[shape_b] & ShapeFlags.HYDROELASTIC) != 0
            if is_hydro_a and is_hydro_b and shape_pairs_sdf_sdf:
                idx = wp.atomic_add(shape_pairs_sdf_sdf_count, 0, 1)
                if idx < shape_pairs_sdf_sdf.shape[0]:
                    shape_pairs_sdf_sdf[idx] = wp.vec2i(shape_a, shape_b)
                continue

            # Get shape data
            data_a = shape_data[shape_a]
            data_b = shape_data[shape_b]
            scale_a = wp.vec3(data_a[0], data_a[1], data_a[2])
            scale_b = wp.vec3(data_b[0], data_b[1], data_b[2])
            thickness_a = data_a[3]
            thickness_b = data_b[3]

            # Get transforms
            X_a = shape_transform[shape_a]
            X_b = shape_transform[shape_b]
            pos_a = wp.transform_get_translation(X_a)
            pos_b = wp.transform_get_translation(X_b)
            quat_a = wp.transform_get_rotation(X_a)
            quat_b = wp.transform_get_rotation(X_b)

            # Calculate contact margin
            margin_a = shape_contact_margin[shape_a]
            margin_b = shape_contact_margin[shape_b]
            margin = margin_a + margin_b

            # =====================================================================
            # Route heightfield pairs to specialized buffer
            # =====================================================================
            is_hfield_a = type_a == GeoType.HFIELD
            is_hfield_b = type_b == GeoType.HFIELD

            # Skip unsupported heightfield combinations
            if is_hfield_a and is_hfield_b:
                continue
            if type_a == GeoType.PLANE and is_hfield_b:
                continue
            if is_hfield_a and (type_b == GeoType.MESH or type_b == GeoType.SDF):
                continue

            # Route heightfield-convex to dedicated buffer
            if is_hfield_a:
                idx = wp.atomic_add(shape_pairs_heightfield_count, 0, 1)
                if idx < shape_pairs_heightfield.shape[0]:
                    shape_pairs_heightfield[idx] = wp.vec2i(shape_a, shape_b)
                continue

            # =====================================================================
            # Route mesh pairs to specialized buffers
            # =====================================================================
            is_mesh_a = type_a == GeoType.MESH
            is_mesh_b = type_b == GeoType.MESH
            is_plane_a = type_a == GeoType.PLANE
            is_infinite_plane_a = is_plane_a and (scale_a[0] == 0.0 and scale_a[1] == 0.0)

            # Mesh-mesh collision
            if is_mesh_a and is_mesh_b:
                idx = wp.atomic_add(shape_pairs_mesh_mesh_count, 0, 1)
                if idx < shape_pairs_mesh_mesh.shape[0]:
                    shape_pairs_mesh_mesh[idx] = wp.vec2i(shape_a, shape_b)
                continue

            # Mesh-plane collision (infinite plane only)
            if is_infinite_plane_a and is_mesh_b:
                mesh_id = shape_source[shape_b]
                if mesh_id != wp.uint64(0):
                    mesh_obj = wp.mesh_get(mesh_id)
                    vertex_count = mesh_obj.points.shape[0]
                    mesh_plane_idx = wp.atomic_add(shape_pairs_mesh_plane_count, 0, 1)
                    if mesh_plane_idx < shape_pairs_mesh_plane.shape[0]:
                        # Store (mesh, plane)
                        shape_pairs_mesh_plane[mesh_plane_idx] = wp.vec2i(shape_b, shape_a)
                        cumulative_count_before = wp.atomic_add(mesh_plane_vertex_total_count, 0, vertex_count)
                        shape_pairs_mesh_plane_cumsum[mesh_plane_idx] = cumulative_count_before + vertex_count
                continue

            # Mesh-convex collision
            if is_mesh_a or is_mesh_b:
                idx = wp.atomic_add(shape_pairs_mesh_count, 0, 1)
                if idx < shape_pairs_mesh.shape[0]:
                    shape_pairs_mesh[idx] = wp.vec2i(shape_a, shape_b)
                continue

            # =====================================================================
            # Handle lightweight primitives analytically
            # =====================================================================
            is_sphere_a = type_a == GeoType.SPHERE
            is_sphere_b = type_b == GeoType.SPHERE
            is_capsule_a = type_a == GeoType.CAPSULE
            is_capsule_b = type_b == GeoType.CAPSULE
            is_ellipsoid_b = type_b == GeoType.ELLIPSOID
            is_cylinder_b = type_b == GeoType.CYLINDER
            is_box_b = type_b == GeoType.BOX

            # Compute effective radii for spheres and capsules
            # (radius that can be represented as Minkowski sum with a sphere)
            radius_eff_a = float(0.0)
            radius_eff_b = float(0.0)
            if is_sphere_a or is_capsule_a:
                radius_eff_a = scale_a[0]
            if is_sphere_b or is_capsule_b:
                radius_eff_b = scale_b[0]

            # Initialize contact result storage (supports up to 2 contacts)
            num_contacts = 0
            contact_dist_0 = float(0.0)
            contact_dist_1 = float(0.0)
            contact_dist_2 = float(0.0)
            contact_dist_3 = float(0.0)
            contact_pos_0 = wp.vec3()
            contact_pos_1 = wp.vec3()
            contact_pos_2 = wp.vec3()
            contact_pos_3 = wp.vec3()
            contact_normal = wp.vec3()

            # -----------------------------------------------------------------
            # Plane-Sphere collision (type_a=PLANE=0, type_b=SPHERE=2)
            # -----------------------------------------------------------------
            if is_plane_a and is_sphere_b:
                plane_normal = wp.quat_rotate(quat_a, wp.vec3(0.0, 0.0, 1.0))
                sphere_radius = scale_b[0]
                contact_dist_0, contact_pos_0 = collide_plane_sphere(plane_normal, pos_a, pos_b, sphere_radius)
                contact_normal = plane_normal
                num_contacts = 1

            # -----------------------------------------------------------------
            # Plane-Ellipsoid collision (type_a=PLANE=0, type_b=ELLIPSOID=4)
            # Produces 1 contact
            # -----------------------------------------------------------------
            elif is_plane_a and is_ellipsoid_b:
                plane_normal = wp.quat_rotate(quat_a, wp.vec3(0.0, 0.0, 1.0))
                ellipsoid_rot = wp.quat_to_matrix(quat_b)
                ellipsoid_size = scale_b
                contact_dist_0, contact_pos_0, contact_normal = collide_plane_ellipsoid(
                    plane_normal, pos_a, pos_b, ellipsoid_rot, ellipsoid_size
                )
                num_contacts = 1

            # -----------------------------------------------------------------
            # Plane-Box collision (type_a=PLANE=0, type_b=BOX=6)
            # Produces up to 4 contacts
            # -----------------------------------------------------------------
            elif is_plane_a and is_box_b:
                plane_normal = wp.quat_rotate(quat_a, wp.vec3(0.0, 0.0, 1.0))
                box_rot = wp.quat_to_matrix(quat_b)
                box_size = scale_b

                dists4_box, positions4_box, contact_normal = collide_plane_box(
                    plane_normal, pos_a, pos_b, box_rot, box_size, margin
                )

                contact_dist_0 = dists4_box[0]
                contact_dist_1 = dists4_box[1]
                contact_dist_2 = dists4_box[2]
                contact_dist_3 = dists4_box[3]
                contact_pos_0 = wp.vec3(positions4_box[0, 0], positions4_box[0, 1], positions4_box[0, 2])
                contact_pos_1 = wp.vec3(positions4_box[1, 0], positions4_box[1, 1], positions4_box[1, 2])
                contact_pos_2 = wp.vec3(positions4_box[2, 0], positions4_box[2, 1], positions4_box[2, 2])
                contact_pos_3 = wp.vec3(positions4_box[3, 0], positions4_box[3, 1], positions4_box[3, 2])
                num_contacts = 4

            # -----------------------------------------------------------------
            # Sphere-Sphere collision (type_a=SPHERE=2, type_b=SPHERE=2)
            # -----------------------------------------------------------------
            elif is_sphere_a and is_sphere_b:
                radius_a = scale_a[0]
                radius_b = scale_b[0]
                contact_dist_0, contact_pos_0, contact_normal = collide_sphere_sphere(pos_a, radius_a, pos_b, radius_b)
                num_contacts = 1

            # -----------------------------------------------------------------
            # Plane-Capsule collision (type_a=PLANE=0, type_b=CAPSULE=3)
            # Produces 2 contacts (both share same normal)
            # -----------------------------------------------------------------
            elif is_plane_a and is_capsule_b:
                plane_normal = wp.quat_rotate(quat_a, wp.vec3(0.0, 0.0, 1.0))
                capsule_axis = wp.quat_rotate(quat_b, wp.vec3(0.0, 0.0, 1.0))
                capsule_radius = scale_b[0]
                capsule_half_length = scale_b[1]

                dists, positions, _frame = collide_plane_capsule(
                    plane_normal, pos_a, pos_b, capsule_axis, capsule_radius, capsule_half_length
                )

                contact_dist_0 = dists[0]
                contact_dist_1 = dists[1]
                contact_pos_0 = wp.vec3(positions[0, 0], positions[0, 1], positions[0, 2])
                contact_pos_1 = wp.vec3(positions[1, 0], positions[1, 1], positions[1, 2])
                contact_normal = plane_normal
                num_contacts = 2

            # -----------------------------------------------------------------
            # Plane-Cylinder collision (type_a=PLANE=0, type_b=CYLINDER=5)
            # Produces up to 4 contacts
            # -----------------------------------------------------------------
            elif is_plane_a and is_cylinder_b:
                plane_normal = wp.quat_rotate(quat_a, wp.vec3(0.0, 0.0, 1.0))
                cylinder_axis = wp.quat_rotate(quat_b, wp.vec3(0.0, 0.0, 1.0))
                cylinder_radius = scale_b[0]
                cylinder_half_height = scale_b[1]

                dists4, positions4, contact_normal = collide_plane_cylinder(
                    plane_normal, pos_a, pos_b, cylinder_axis, cylinder_radius, cylinder_half_height
                )

                contact_dist_0 = dists4[0]
                contact_dist_1 = dists4[1]
                contact_dist_2 = dists4[2]
                contact_dist_3 = dists4[3]
                contact_pos_0 = wp.vec3(positions4[0, 0], positions4[0, 1], positions4[0, 2])
                contact_pos_1 = wp.vec3(positions4[1, 0], positions4[1, 1], positions4[1, 2])
                contact_pos_2 = wp.vec3(positions4[2, 0], positions4[2, 1], positions4[2, 2])
                contact_pos_3 = wp.vec3(positions4[3, 0], positions4[3, 1], positions4[3, 2])
                num_contacts = 4

            # -----------------------------------------------------------------
            # Sphere-Capsule collision (type_a=SPHERE=2, type_b=CAPSULE=3)
            # -----------------------------------------------------------------
            elif is_sphere_a and is_capsule_b:
                sphere_radius = scale_a[0]
                capsule_axis = wp.quat_rotate(quat_b, wp.vec3(0.0, 0.0, 1.0))
                capsule_radius = scale_b[0]
                capsule_half_length = scale_b[1]
                contact_dist_0, contact_pos_0, contact_normal = collide_sphere_capsule(
                    pos_a, sphere_radius, pos_b, capsule_axis, capsule_radius, capsule_half_length
                )
                num_contacts = 1

            # -----------------------------------------------------------------
            # Capsule-Capsule collision (type_a=CAPSULE=3, type_b=CAPSULE=3)
            # Produces 1 contact (non-parallel) or 2 contacts (parallel axes)
            # -----------------------------------------------------------------
            elif is_capsule_a and is_capsule_b:
                axis_a = wp.quat_rotate(quat_a, wp.vec3(0.0, 0.0, 1.0))
                axis_b = wp.quat_rotate(quat_b, wp.vec3(0.0, 0.0, 1.0))
                radius_a = scale_a[0]
                half_length_a = scale_a[1]
                radius_b = scale_b[0]
                half_length_b = scale_b[1]

                dists, positions, contact_normal = collide_capsule_capsule(
                    pos_a, axis_a, radius_a, half_length_a, pos_b, axis_b, radius_b, half_length_b
                )

                contact_dist_0 = dists[0]
                contact_pos_0 = wp.vec3(positions[0, 0], positions[0, 1], positions[0, 2])

                # Check if second contact is valid (parallel axes case)
                if dists[1] < MAXVAL:
                    contact_dist_1 = dists[1]
                    contact_pos_1 = wp.vec3(positions[1, 0], positions[1, 1], positions[1, 2])
                    num_contacts = 2
                else:
                    num_contacts = 1

            # -----------------------------------------------------------------
            # Sphere-Cylinder collision (type_a=SPHERE=2, type_b=CYLINDER=5)
            # -----------------------------------------------------------------
            elif is_sphere_a and is_cylinder_b:
                sphere_radius = scale_a[0]
                cylinder_axis = wp.quat_rotate(quat_b, wp.vec3(0.0, 0.0, 1.0))
                cylinder_radius = scale_b[0]
                cylinder_half_height = scale_b[1]
                contact_dist_0, contact_pos_0, contact_normal = collide_sphere_cylinder(
                    pos_a, sphere_radius, pos_b, cylinder_axis, cylinder_radius, cylinder_half_height
                )
                num_contacts = 1

            # -----------------------------------------------------------------
            # Sphere-Box collision (type_a=SPHERE=2, type_b=BOX=6)
            # -----------------------------------------------------------------
            elif is_sphere_a and is_box_b:
                sphere_radius = scale_a[0]
                box_rot = wp.quat_to_matrix(quat_b)
                box_size = scale_b
                contact_dist_0, contact_pos_0, contact_normal = collide_sphere_box(
                    pos_a, sphere_radius, pos_b, box_rot, box_size
                )
                num_contacts = 1

            # =====================================================================
            # Write all contacts (single write block for 0 to 4 contacts)
            # =====================================================================
            if num_contacts > 0:
                # Prepare contact data (shared fields for both contacts)
                contact_data = ContactData()
                contact_data.contact_normal_a_to_b = contact_normal
                contact_data.radius_eff_a = radius_eff_a
                contact_data.radius_eff_b = radius_eff_b
                contact_data.thickness_a = thickness_a
                contact_data.thickness_b = thickness_b
                contact_data.shape_a = shape_a
                contact_data.shape_b = shape_b
                contact_data.margin = margin

                # Check margin for all possible contacts
                contact_0_valid = False
                if contact_dist_0 < MAXVAL:
                    contact_data.contact_point_center = contact_pos_0
                    contact_data.contact_distance = contact_dist_0
                    contact_0_valid = contact_passes_margin_check(contact_data)

                contact_1_valid = False
                if num_contacts > 1 and contact_dist_1 < MAXVAL:
                    contact_data.contact_point_center = contact_pos_1
                    contact_data.contact_distance = contact_dist_1
                    contact_1_valid = contact_passes_margin_check(contact_data)

                contact_2_valid = False
                if num_contacts > 2 and contact_dist_2 < MAXVAL:
                    contact_data.contact_point_center = contact_pos_2
                    contact_data.contact_distance = contact_dist_2
                    contact_2_valid = contact_passes_margin_check(contact_data)

                contact_3_valid = False
                if num_contacts > 3 and contact_dist_3 < MAXVAL:
                    contact_data.contact_point_center = contact_pos_3
                    contact_data.contact_distance = contact_dist_3
                    contact_3_valid = contact_passes_margin_check(contact_data)

                # Count valid contacts and allocate consecutive indices
                num_valid = int(contact_0_valid) + int(contact_1_valid) + int(contact_2_valid) + int(contact_3_valid)
                if num_valid > 0:
                    base_index = wp.atomic_add(writer_data.contact_count, 0, num_valid)

                    # Bounds check: ensure we don't overflow the contact buffer
                    if base_index + num_valid > writer_data.contact_max:
                        # Rollback the allocation
                        wp.atomic_add(writer_data.contact_count, 0, -num_valid)
                        continue

                    # Write first contact if valid
                    if contact_0_valid:
                        contact_data.contact_point_center = contact_pos_0
                        contact_data.contact_distance = contact_dist_0
                        writer_func(contact_data, writer_data, base_index)
                        base_index += 1

                    # Write second contact if valid
                    if contact_1_valid:
                        contact_data.contact_point_center = contact_pos_1
                        contact_data.contact_distance = contact_dist_1
                        writer_func(contact_data, writer_data, base_index)
                        base_index += 1

                    # Write third contact if valid
                    if contact_2_valid:
                        contact_data.contact_point_center = contact_pos_2
                        contact_data.contact_distance = contact_dist_2
                        writer_func(contact_data, writer_data, base_index)
                        base_index += 1

                    # Write fourth contact if valid
                    if contact_3_valid:
                        contact_data.contact_point_center = contact_pos_3
                        contact_data.contact_distance = contact_dist_3
                        writer_func(contact_data, writer_data, base_index)

                continue

            # =====================================================================
            # Route remaining pairs to GJK/MPR kernel
            # =====================================================================
            idx = wp.atomic_add(gjk_candidate_pairs_count, 0, 1)
            if idx < gjk_candidate_pairs.shape[0]:
                gjk_candidate_pairs[idx] = wp.vec2i(shape_a, shape_b)

    return narrow_phase_primitive_kernel


def create_narrow_phase_kernel_gjk_mpr(external_aabb: bool, writer_func: Any):
    """
    Create a GJK/MPR narrow phase kernel for complex convex shape collisions.

    This kernel is called AFTER the primitive kernel has already:
    - Sorted pairs by type (type_a <= type_b)
    - Routed mesh pairs to specialized buffers
    - Routed hydroelastic pairs to SDF-SDF buffer
    - Handled primitive collisions analytically

    The remaining pairs are complex convex-convex (plane-box, plane-cylinder,
    plane-cone, box-box, cylinder-cylinder, etc.) that need GJK/MPR.
    """

    @wp.kernel(enable_backward=False, module="unique")
    def narrow_phase_kernel_gjk_mpr(
        candidate_pair: wp.array(dtype=wp.vec2i),
        candidate_pair_count: wp.array(dtype=int),
        shape_types: wp.array(dtype=int),
        shape_data: wp.array(dtype=wp.vec4),
        shape_transform: wp.array(dtype=wp.transform),
        shape_source: wp.array(dtype=wp.uint64),
        shape_contact_margin: wp.array(dtype=float),
        shape_collision_radius: wp.array(dtype=float),
        shape_aabb_lower: wp.array(dtype=wp.vec3),
        shape_aabb_upper: wp.array(dtype=wp.vec3),
        writer_data: Any,
        total_num_threads: int,
    ):
        """
        GJK/MPR collision detection for complex convex pairs.

        Pairs arrive pre-sorted (type_a <= type_b) and pre-filtered
        (no meshes, no hydroelastic, no simple primitives).
        """
        tid = wp.tid()

        num_work_items = wp.min(candidate_pair.shape[0], candidate_pair_count[0])

        # Early exit if no work (fast path for primitive-only scenes)
        if num_work_items == 0:
            return

        for t in range(tid, num_work_items, total_num_threads):
            # Get shape pair (already sorted by primitive kernel)
            pair = candidate_pair[t]
            shape_a = pair[0]
            shape_b = pair[1]

            # Safety checks
            if shape_a == shape_b or shape_a < 0 or shape_b < 0:
                continue

            # Get shape types (already sorted: type_a <= type_b)
            type_a = shape_types[shape_a]
            type_b = shape_types[shape_b]

            # Extract shape data
            pos_a, quat_a, shape_data_a, scale_a, thickness_a = extract_shape_data(
                shape_a, shape_transform, shape_types, shape_data, shape_source
            )
            pos_b, quat_b, shape_data_b, scale_b, thickness_b = extract_shape_data(
                shape_b, shape_transform, shape_types, shape_data, shape_source
            )

            # Check for infinite planes
            is_infinite_plane_a = (type_a == GeoType.PLANE) and (scale_a[0] == 0.0 and scale_a[1] == 0.0)
            is_infinite_plane_b = (type_b == GeoType.PLANE) and (scale_b[0] == 0.0 and scale_b[1] == 0.0)

            # Early exit: both infinite planes can't collide
            if is_infinite_plane_a and is_infinite_plane_b:
                continue

            # Compute or fetch AABBs for bounding sphere overlap check
            if wp.static(external_aabb):
                aabb_a_lower = shape_aabb_lower[shape_a]
                aabb_a_upper = shape_aabb_upper[shape_a]
                aabb_b_lower = shape_aabb_lower[shape_b]
                aabb_b_upper = shape_aabb_upper[shape_b]
            if wp.static(not external_aabb):
                margin_a = shape_contact_margin[shape_a]
                margin_b = shape_contact_margin[shape_b]
                margin_vec_a = wp.vec3(margin_a, margin_a, margin_a)
                margin_vec_b = wp.vec3(margin_b, margin_b, margin_b)

                # Shape A AABB
                is_sdf_a = type_a == GeoType.SDF
                if is_infinite_plane_a or is_sdf_a:
                    radius_a = shape_collision_radius[shape_a]
                    half_extents_a = wp.vec3(radius_a, radius_a, radius_a)
                    aabb_a_lower = pos_a - half_extents_a - margin_vec_a
                    aabb_a_upper = pos_a + half_extents_a + margin_vec_a
                else:
                    data_provider = SupportMapDataProvider()
                    aabb_a_lower, aabb_a_upper = compute_tight_aabb_from_support(
                        shape_data_a, quat_a, pos_a, data_provider
                    )
                    aabb_a_lower = aabb_a_lower - margin_vec_a
                    aabb_a_upper = aabb_a_upper + margin_vec_a

                # Shape B AABB
                is_sdf_b = type_b == GeoType.SDF
                if is_infinite_plane_b or is_sdf_b:
                    radius_b = shape_collision_radius[shape_b]
                    half_extents_b = wp.vec3(radius_b, radius_b, radius_b)
                    aabb_b_lower = pos_b - half_extents_b - margin_vec_b
                    aabb_b_upper = pos_b + half_extents_b + margin_vec_b
                else:
                    data_provider = SupportMapDataProvider()
                    aabb_b_lower, aabb_b_upper = compute_tight_aabb_from_support(
                        shape_data_b, quat_b, pos_b, data_provider
                    )
                    aabb_b_lower = aabb_b_lower - margin_vec_b
                    aabb_b_upper = aabb_b_upper + margin_vec_b

            # Compute bounding spheres and check for overlap (early rejection)
            bsphere_center_a, bsphere_radius_a = compute_bounding_sphere_from_aabb(aabb_a_lower, aabb_a_upper)
            bsphere_center_b, bsphere_radius_b = compute_bounding_sphere_from_aabb(aabb_b_lower, aabb_b_upper)

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

            # Compute contact margin
            margin_a = shape_contact_margin[shape_a]
            margin_b = shape_contact_margin[shape_b]
            margin = margin_a + margin_b

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
    shape_types: wp.array(dtype=int),
    shape_transform: wp.array(dtype=wp.transform),
    shape_source: wp.array(dtype=wp.uint64),
    shape_contact_margin: wp.array(dtype=float),  # Per-shape contact margins
    shape_data: wp.array(dtype=wp.vec4),  # Shape data (scale xyz, thickness w)
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
    tid, j = wp.tid()

    num_mesh_pairs = shape_pairs_mesh_count[0]

    # Strided loop over mesh pairs
    for i in range(tid, num_mesh_pairs, total_num_threads):
        pair = shape_pairs_mesh[i]
        shape_a = pair[0]
        shape_b = pair[1]

        # Determine which shape is the mesh
        type_a = shape_types[shape_a]
        type_b = shape_types[shape_b]

        mesh_shape = -1
        non_mesh_shape = -1

        if type_a == GeoType.MESH and type_b != GeoType.MESH:
            mesh_shape = shape_a
            non_mesh_shape = shape_b
        elif type_b == GeoType.MESH and type_a != GeoType.MESH:
            mesh_shape = shape_b
            non_mesh_shape = shape_a
        else:
            # Mesh-mesh collision not supported yet
            return

        # Get mesh BVH ID and mesh transform
        mesh_id = shape_source[mesh_shape]
        if mesh_id == wp.uint64(0):
            return

        # Get mesh world transform
        X_mesh_ws = shape_transform[mesh_shape]

        # Get non-mesh shape world transform
        X_ws = shape_transform[non_mesh_shape]

        # Use per-shape contact margin for the non-mesh shape
        # Sum margins for consistency with thickness summing
        margin_non_mesh = shape_contact_margin[non_mesh_shape]
        margin_mesh = shape_contact_margin[mesh_shape]
        margin = margin_non_mesh + margin_mesh

        # Call mesh_vs_convex_midphase with the shape_data and margin
        mesh_vs_convex_midphase(
            j,
            mesh_shape,
            non_mesh_shape,
            X_mesh_ws,
            X_ws,
            mesh_id,
            shape_types,
            shape_data,
            shape_source,
            margin,
            triangle_pairs,
            triangle_pairs_count,
        )


def create_narrow_phase_process_mesh_triangle_contacts_kernel(writer_func: Any):
    @wp.kernel(enable_backward=False, module="unique")
    def narrow_phase_process_mesh_triangle_contacts_kernel(
        shape_types: wp.array(dtype=int),
        shape_data: wp.array(dtype=wp.vec4),
        shape_transform: wp.array(dtype=wp.transform),
        shape_source: wp.array(dtype=wp.uint64),
        shape_contact_margin: wp.array(dtype=float),  # Per-shape contact margins
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
            mesh_id_a = shape_source[shape_a]
            if mesh_id_a == wp.uint64(0):
                continue

            scale_data_a = shape_data[shape_a]
            mesh_scale_a = wp.vec3(scale_data_a[0], scale_data_a[1], scale_data_a[2])

            # Get mesh world transform for shape A
            X_mesh_ws_a = shape_transform[shape_a]

            # Extract triangle shape data from mesh
            shape_data_a, v0_world = get_triangle_shape_from_mesh(mesh_id_a, mesh_scale_a, X_mesh_ws_a, tri_idx)

            # Extract shape B data
            pos_b, quat_b, shape_data_b, _scale_b, thickness_b = extract_shape_data(
                shape_b,
                shape_transform,
                shape_types,
                shape_data,
                shape_source,
            )

            # Set pos_a to be vertex A (origin of triangle in local frame)
            pos_a = v0_world
            quat_a = wp.quat_identity()  # Triangle has no orientation, use identity

            # Extract thickness for shape A
            thickness_a = shape_data[shape_a][3]

            # Use per-shape contact margin for contact detection
            # Sum margins for consistency with thickness summing
            margin_a = shape_contact_margin[shape_a]
            margin_b = shape_contact_margin[shape_b]
            margin = margin_a + margin_b

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
            )

    return narrow_phase_process_mesh_triangle_contacts_kernel


def create_narrow_phase_process_mesh_plane_contacts_kernel(
    writer_func: Any,
    contact_reduction_funcs: ContactReductionFunctions | None = None,
):
    """
    Create a mesh-plane collision kernel.

    Args:
        writer_func: Contact writer function (e.g., write_contact_simple)
        contact_reduction_funcs: ContactReductionFunctions instance. If None, no contact reduction is used.

    Returns:
        A warp kernel that processes mesh-plane collisions
    """

    @wp.kernel(enable_backward=False, module="unique")
    def narrow_phase_process_mesh_plane_contacts_kernel(
        shape_data: wp.array(dtype=wp.vec4),
        shape_transform: wp.array(dtype=wp.transform),
        shape_source: wp.array(dtype=wp.uint64),
        shape_contact_margin: wp.array(dtype=float),
        _shape_collision_aabb_lower: wp.array(dtype=wp.vec3),  # Unused but kept for API compatibility
        _shape_collision_aabb_upper: wp.array(dtype=wp.vec3),  # Unused but kept for API compatibility
        _shape_voxel_resolution: wp.array(dtype=wp.vec3i),  # Unused but kept for API compatibility
        shape_pairs_mesh_plane: wp.array(dtype=wp.vec2i),
        shape_pairs_mesh_plane_count: wp.array(dtype=int),
        writer_data: Any,
        total_num_blocks: int,
    ):
        """
        Process mesh-plane collisions without contact reduction.

        Each thread processes vertices in a strided manner and writes contacts directly.
        """
        tid = wp.tid()

        pair_count = shape_pairs_mesh_plane_count[0]

        # Iterate over all mesh-plane pairs
        for pair_idx in range(pair_count):
            pair = shape_pairs_mesh_plane[pair_idx]
            mesh_shape = pair[0]
            plane_shape = pair[1]

            # Get mesh
            mesh_id = shape_source[mesh_shape]
            if mesh_id == wp.uint64(0):
                continue

            mesh_obj = wp.mesh_get(mesh_id)
            num_vertices = mesh_obj.points.shape[0]

            # Get mesh world transform
            X_mesh_ws = shape_transform[mesh_shape]

            # Get plane world transform
            X_plane_ws = shape_transform[plane_shape]
            X_plane_sw = wp.transform_inverse(X_plane_ws)

            # Get plane normal in world space (plane normal is along local +Z, pointing upward)
            plane_normal = wp.transform_vector(X_plane_ws, wp.vec3(0.0, 0.0, 1.0))

            # Get mesh scale
            scale_data = shape_data[mesh_shape]
            mesh_scale = wp.vec3(scale_data[0], scale_data[1], scale_data[2])

            # Extract thickness values
            thickness_mesh = shape_data[mesh_shape][3]
            thickness_plane = shape_data[plane_shape][3]
            total_thickness = thickness_mesh + thickness_plane

            # Use per-shape contact margin for contact detection
            # Sum margins for consistency with thickness summing
            margin_mesh = shape_contact_margin[mesh_shape]
            margin_plane = shape_contact_margin[plane_shape]
            margin = margin_mesh + margin_plane

            # Strided loop over vertices across all threads in the launch
            total_num_threads = total_num_blocks * wp.block_dim()
            for vertex_idx in range(tid, num_vertices, total_num_threads):
                # Get vertex position in mesh local space and transform to world space
                vertex_local = wp.cw_mul(mesh_obj.points[vertex_idx], mesh_scale)
                vertex_world = wp.transform_point(X_mesh_ws, vertex_local)

                # Project vertex onto plane to get closest point
                vertex_in_plane_space = wp.transform_point(X_plane_sw, vertex_world)
                point_on_plane_local = wp.vec3(vertex_in_plane_space[0], vertex_in_plane_space[1], 0.0)
                point_on_plane = wp.transform_point(X_plane_ws, point_on_plane_local)

                # Compute distance
                diff = vertex_world - point_on_plane
                distance = wp.dot(diff, plane_normal)

                # Check if this vertex generates a contact
                if distance < margin + total_thickness:
                    # Contact position is the midpoint
                    contact_pos = (vertex_world + point_on_plane) * 0.5

                    # Normal points from mesh to plane (negate plane normal since plane normal points up/away from plane)
                    contact_normal = -plane_normal

                    # Create contact data - contacts are already in world space
                    contact_data = ContactData()
                    contact_data.contact_point_center = contact_pos
                    contact_data.contact_normal_a_to_b = contact_normal
                    contact_data.contact_distance = distance
                    contact_data.radius_eff_a = 0.0
                    contact_data.radius_eff_b = 0.0
                    contact_data.thickness_a = thickness_mesh
                    contact_data.thickness_b = thickness_plane
                    contact_data.shape_a = mesh_shape
                    contact_data.shape_b = plane_shape
                    contact_data.margin = margin

                    writer_func(contact_data, writer_data, -1)

    # Return early if contact reduction is disabled
    if contact_reduction_funcs is None:
        return narrow_phase_process_mesh_plane_contacts_kernel

    # Extract functions and constants from the contact reduction configuration

    reduction_slot_count = contact_reduction_funcs.reduction_slot_count
    store_reduced_contact_func = contact_reduction_funcs.store_reduced_contact
    get_smem_slots_plus_1 = contact_reduction_funcs.get_smem_slots_plus_1
    get_smem_slots_contacts = contact_reduction_funcs.get_smem_slots_contacts

    @wp.kernel(enable_backward=False, module="unique")
    def narrow_phase_process_mesh_plane_contacts_reduce_kernel(
        shape_data: wp.array(dtype=wp.vec4),
        shape_transform: wp.array(dtype=wp.transform),
        shape_source: wp.array(dtype=wp.uint64),
        shape_contact_margin: wp.array(dtype=float),
        shape_collision_aabb_lower: wp.array(dtype=wp.vec3),
        shape_collision_aabb_upper: wp.array(dtype=wp.vec3),
        shape_voxel_resolution: wp.array(dtype=wp.vec3i),
        shape_pairs_mesh_plane: wp.array(dtype=wp.vec2i),
        shape_pairs_mesh_plane_count: wp.array(dtype=int),
        writer_data: Any,
        total_num_blocks: int,
    ):
        """
        Process mesh-plane collisions with contact reduction.

        Each thread block handles one mesh-plane pair. Threads cooperatively iterate
        over all vertices of the mesh, generate contacts, and reduce them using
        shared memory contact reduction. Uses grid stride loop to handle more pairs
        than available blocks.
        """
        block_id, t = wp.tid()

        pair_count = shape_pairs_mesh_plane_count[0]

        # Initialize shared memory buffers for contact reduction (reused across pairs)
        empty_marker = -1000000000.0
        active_contacts_shared_mem = wp.array(
            ptr=wp.static(get_smem_slots_plus_1)(),
            shape=(wp.static(reduction_slot_count) + 1,),
            dtype=wp.int32,
        )
        contacts_shared_mem = wp.array(
            ptr=wp.static(get_smem_slots_contacts)(),
            shape=(wp.static(reduction_slot_count),),
            dtype=ContactStruct,
        )

        # Grid stride loop over mesh-plane pairs
        for pair_idx in range(block_id, pair_count, total_num_blocks):
            # Get the mesh-plane pair
            pair = shape_pairs_mesh_plane[pair_idx]
            mesh_shape = pair[0]
            plane_shape = pair[1]

            # Get mesh
            mesh_id = shape_source[mesh_shape]
            if mesh_id == wp.uint64(0):
                continue

            mesh_obj = wp.mesh_get(mesh_id)
            num_vertices = mesh_obj.points.shape[0]

            # Get mesh world transform
            X_mesh_ws = shape_transform[mesh_shape]
            X_ws_mesh = wp.transform_inverse(X_mesh_ws)  # World to mesh local

            # Load voxel binning data for mesh
            aabb_lower_mesh = shape_collision_aabb_lower[mesh_shape]
            aabb_upper_mesh = shape_collision_aabb_upper[mesh_shape]
            voxel_res_mesh = shape_voxel_resolution[mesh_shape]

            # Get plane world transform
            X_plane_ws = shape_transform[plane_shape]
            X_plane_sw = wp.transform_inverse(X_plane_ws)

            # Get plane normal in world space (plane normal is along local +Z, pointing upward)
            plane_normal = wp.transform_vector(X_plane_ws, wp.vec3(0.0, 0.0, 1.0))

            # Get mesh scale
            scale_data = shape_data[mesh_shape]
            mesh_scale = wp.vec3(scale_data[0], scale_data[1], scale_data[2])

            # Extract thickness values
            thickness_mesh = shape_data[mesh_shape][3]
            thickness_plane = shape_data[plane_shape][3]
            total_thickness = thickness_mesh + thickness_plane

            # Use per-shape contact margin for contact detection
            # Sum margins for consistency with thickness summing
            margin_mesh = shape_contact_margin[mesh_shape]
            margin_plane = shape_contact_margin[plane_shape]
            margin = margin_mesh + margin_plane

            # Reset contact buffer for this pair
            for i in range(t, wp.static(reduction_slot_count), wp.block_dim()):
                contacts_shared_mem[i].projection = empty_marker

            if t == 0:
                active_contacts_shared_mem[wp.static(reduction_slot_count)] = 0

            synchronize()

            # Process vertices in batches using strided loop

            num_iterations = (num_vertices + wp.block_dim() - 1) // wp.block_dim()
            for i in range(num_iterations):
                vertex_idx = i * wp.block_dim() + t
                has_contact = wp.bool(False)
                c = ContactStruct()

                if vertex_idx < num_vertices:
                    # Get vertex position in mesh local space and transform to world space
                    vertex_local = wp.cw_mul(mesh_obj.points[vertex_idx], mesh_scale)
                    vertex_world = wp.transform_point(X_mesh_ws, vertex_local)

                    # Project vertex onto plane to get closest point
                    vertex_in_plane_space = wp.transform_point(X_plane_sw, vertex_world)
                    point_on_plane_local = wp.vec3(vertex_in_plane_space[0], vertex_in_plane_space[1], 0.0)
                    point_on_plane = wp.transform_point(X_plane_ws, point_on_plane_local)

                    # Compute distance
                    diff = vertex_world - point_on_plane
                    distance = wp.dot(diff, plane_normal)

                    # Check if this vertex generates a contact
                    if distance < margin + total_thickness:
                        has_contact = True

                        # Contact position is the midpoint
                        contact_pos = (vertex_world + point_on_plane) * 0.5

                        # Normal points from mesh to plane (negate plane normal since plane normal points up/away from plane)
                        contact_normal = -plane_normal

                        c.position = contact_pos
                        c.normal = contact_normal
                        c.depth = distance
                        c.feature = vertex_idx
                        c.projection = empty_marker

                # Compute voxel index for contact position in mesh's local space
                voxel_idx = int(0)
                if has_contact:
                    point_mesh_local = wp.transform_point(X_ws_mesh, contact_pos)
                    voxel_idx = compute_voxel_index(point_mesh_local, aabb_lower_mesh, aabb_upper_mesh, voxel_res_mesh)

                # Apply contact reduction
                store_reduced_contact_func(
                    t, has_contact, c, contacts_shared_mem, active_contacts_shared_mem, empty_marker, voxel_idx
                )

            # Write reduced contacts to output (store_reduced_contact ends with sync)
            num_contacts_to_keep = wp.min(
                active_contacts_shared_mem[wp.static(reduction_slot_count)], wp.static(reduction_slot_count)
            )

            for i in range(t, num_contacts_to_keep, wp.block_dim()):
                contact_id = active_contacts_shared_mem[i]
                contact = contacts_shared_mem[contact_id]

                # Create contact data - contacts are already in world space
                contact_data = ContactData()
                contact_data.contact_point_center = contact.position
                contact_data.contact_normal_a_to_b = contact.normal
                contact_data.contact_distance = contact.depth
                contact_data.radius_eff_a = 0.0
                contact_data.radius_eff_b = 0.0
                contact_data.thickness_a = thickness_mesh
                contact_data.thickness_b = thickness_plane
                contact_data.shape_a = mesh_shape
                contact_data.shape_b = plane_shape
                contact_data.margin = margin

                writer_func(contact_data, writer_data, -1)

            # Ensure all threads complete before processing next pair
            synchronize()

    return narrow_phase_process_mesh_plane_contacts_reduce_kernel


@wp.kernel(enable_backward=False)
def heightfield_midphase_kernel(
    shape_types: wp.array(dtype=int),
    shape_transform: wp.array(dtype=wp.transform),
    shape_collision_radius: wp.array(dtype=float),
    shape_contact_margin: wp.array(dtype=float),
    shape_heightfield_data: wp.array(dtype=HeightfieldData),
    shape_pairs_heightfield: wp.array(dtype=wp.vec2i),
    shape_pairs_heightfield_count: wp.array(dtype=int),
    total_num_threads: int,
    # outputs
    heightfield_cell_pairs: wp.array(dtype=wp.vec4i),
    heightfield_cell_pairs_count: wp.array(dtype=int),
):
    """Find heightfield grid cells that overlap with another shape.

    For each (heightfield, other) pair, projects the other shape's bounding
    sphere onto the heightfield grid and emits (hfield_shape, other_shape, row, col)
    for each overlapping cell.
    """
    tid = wp.tid()

    num_pairs = shape_pairs_heightfield_count[0]

    for i in range(tid, num_pairs, total_num_threads):
        if i >= shape_pairs_heightfield.shape[0]:
            break

        pair = shape_pairs_heightfield[i]
        shape_a = pair[0]
        shape_b = pair[1]

        # Determine which is the heightfield
        hfield_shape = shape_a
        other_shape = shape_b
        if shape_types[shape_b] == GeoType.HFIELD:
            hfield_shape = shape_b
            other_shape = shape_a

        hfd = shape_heightfield_data[hfield_shape]
        if hfd.nrow <= 1 or hfd.ncol <= 1:
            continue

        # Transform other shape's position to heightfield local space
        X_hfield_ws = shape_transform[hfield_shape]
        X_hfield_inv = wp.transform_inverse(X_hfield_ws)
        X_other_ws = shape_transform[other_shape]
        pos_in_hfield = wp.transform_point(X_hfield_inv, wp.transform_get_translation(X_other_ws))

        # Use bounding sphere radius for conservative AABB in heightfield-local space
        radius = shape_collision_radius[other_shape]
        margin = shape_contact_margin[hfield_shape] + shape_contact_margin[other_shape]
        extent = radius + margin

        aabb_lower = pos_in_hfield - wp.vec3(extent, extent, extent)
        aabb_upper = pos_in_hfield + wp.vec3(extent, extent, extent)

        # Map AABB to grid cell indices
        dx = 2.0 * hfd.hx / wp.float32(hfd.ncol - 1)
        dy = 2.0 * hfd.hy / wp.float32(hfd.nrow - 1)

        col_min_f = (aabb_lower[0] + hfd.hx) / dx
        col_max_f = (aabb_upper[0] + hfd.hx) / dx
        row_min_f = (aabb_lower[1] + hfd.hy) / dy
        row_max_f = (aabb_upper[1] + hfd.hy) / dy

        col_min = wp.max(wp.int32(col_min_f), 0)
        col_max = wp.min(wp.int32(col_max_f), hfd.ncol - 2)
        row_min = wp.max(wp.int32(row_min_f), 0)
        row_max = wp.min(wp.int32(row_max_f), hfd.nrow - 2)

        for r in range(row_min, row_max + 1):
            for c in range(col_min, col_max + 1):
                out_idx = wp.atomic_add(heightfield_cell_pairs_count, 0, 1)
                if out_idx < heightfield_cell_pairs.shape[0]:
                    heightfield_cell_pairs[out_idx] = wp.vec4i(hfield_shape, other_shape, r, c)


def create_heightfield_triangle_contacts_kernel(writer_func: Any):
    """Create a kernel that processes heightfield cell pairs into triangle contacts."""

    @wp.kernel(enable_backward=False)
    def heightfield_triangle_contacts_kernel(
        shape_types: wp.array(dtype=int),
        shape_data: wp.array(dtype=wp.vec4),
        shape_transform: wp.array(dtype=wp.transform),
        shape_source: wp.array(dtype=wp.uint64),
        shape_contact_margin: wp.array(dtype=float),
        shape_heightfield_data: wp.array(dtype=HeightfieldData),
        heightfield_elevation_data: wp.array(dtype=wp.float32),
        heightfield_cell_pairs: wp.array(dtype=wp.vec4i),
        heightfield_cell_pairs_count: wp.array(dtype=int),
        writer_data: Any,
        total_num_threads: int,
    ):
        """Process heightfield cell pairs to generate contacts via GJK/MPR.

        Each cell produces 2 triangles. For each triangle, runs GJK/MPR
        against the convex shape using the same infrastructure as mesh collisions.
        """
        tid = wp.tid()

        num_cell_pairs = heightfield_cell_pairs_count[0]

        for i in range(tid, num_cell_pairs, total_num_threads):
            if i >= heightfield_cell_pairs.shape[0]:
                break

            quad = heightfield_cell_pairs[i]
            hfield_shape = quad[0]
            convex_shape = quad[1]
            row = quad[2]
            col = quad[3]

            hfd = shape_heightfield_data[hfield_shape]
            X_hfield_ws = shape_transform[hfield_shape]

            # Extract convex shape data
            pos_b, quat_b, shape_data_b, _scale_b, thickness_b = extract_shape_data(
                convex_shape, shape_transform, shape_types, shape_data, shape_source
            )

            thickness_a = shape_data[hfield_shape][3]
            margin_a = shape_contact_margin[hfield_shape]
            margin_b = shape_contact_margin[convex_shape]
            margin = margin_a + margin_b

            # Process 2 triangles per cell
            for tri_sub in range(2):
                shape_data_tri, v0_world = get_triangle_from_heightfield_cell(
                    hfd, heightfield_elevation_data, X_hfield_ws, row, col, tri_sub
                )

                wp.static(create_compute_gjk_mpr_contacts(writer_func))(
                    shape_data_tri,
                    shape_data_b,
                    wp.quat_identity(),
                    quat_b,
                    v0_world,
                    pos_b,
                    margin,
                    hfield_shape,
                    convex_shape,
                    thickness_a,
                    thickness_b,
                    writer_data,
                )

    return heightfield_triangle_contacts_kernel


# =============================================================================
# Verification kernel
# =============================================================================


@wp.kernel(enable_backward=False)
def verify_narrow_phase_buffers(
    broad_phase_count: wp.array(dtype=int),
    max_broad_phase: int,
    gjk_count: wp.array(dtype=int),
    max_gjk: int,
    mesh_count: wp.array(dtype=int),
    max_mesh: int,
    triangle_count: wp.array(dtype=int),
    max_triangle: int,
    mesh_plane_count: wp.array(dtype=int),
    max_mesh_plane: int,
    mesh_mesh_count: wp.array(dtype=int),
    max_mesh_mesh: int,
    hf_pairs_count: wp.array(dtype=int),
    max_hf_pairs: int,
    hf_cells_count: wp.array(dtype=int),
    max_hf_cells: int,
    sdf_sdf_count: wp.array(dtype=int),
    max_sdf_sdf: int,
    contact_count: wp.array(dtype=int),
    max_contacts: int,
):
    """Check for buffer overflows in the collision pipeline."""
    if broad_phase_count[0] > max_broad_phase:
        wp.printf(
            "Warning: Broad phase pair buffer overflowed %d > %d.\n",
            broad_phase_count[0],
            max_broad_phase,
        )
    if gjk_count[0] > max_gjk:
        wp.printf(
            "Warning: GJK candidate pair buffer overflowed %d > %d.\n",
            gjk_count[0],
            max_gjk,
        )
    if mesh_count:
        if mesh_count[0] > max_mesh:
            wp.printf(
                "Warning: Mesh-convex shape pair buffer overflowed %d > %d.\n",
                mesh_count[0],
                max_mesh,
            )
    if triangle_count:
        if triangle_count[0] > max_triangle:
            wp.printf(
                "Warning: Triangle pair buffer overflowed %d > %d.\n",
                triangle_count[0],
                max_triangle,
            )
    if mesh_plane_count:
        if mesh_plane_count[0] > max_mesh_plane:
            wp.printf(
                "Warning: Mesh-plane shape pair buffer overflowed %d > %d.\n",
                mesh_plane_count[0],
                max_mesh_plane,
            )
    if mesh_mesh_count:
        if mesh_mesh_count[0] > max_mesh_mesh:
            wp.printf(
                "Warning: Mesh-mesh shape pair buffer overflowed %d > %d.\n",
                mesh_mesh_count[0],
                max_mesh_mesh,
            )
    if hf_pairs_count[0] > max_hf_pairs:
        wp.printf(
            "Warning: Heightfield shape pair buffer overflowed %d > %d.\n",
            hf_pairs_count[0],
            max_hf_pairs,
        )
    if hf_cells_count[0] > max_hf_cells:
        wp.printf(
            "Warning: Heightfield cell pair buffer overflowed %d > %d.\n",
            hf_cells_count[0],
            max_hf_cells,
        )
    if sdf_sdf_count:
        if sdf_sdf_count[0] > max_sdf_sdf:
            wp.printf(
                "Warning: SDF-SDF shape pair buffer overflowed %d > %d.\n",
                sdf_sdf_count[0],
                max_sdf_sdf,
            )
    if contact_count[0] > max_contacts:
        wp.printf(
            "Warning: Contact buffer overflowed %d > %d.\n",
            contact_count[0],
            max_contacts,
        )


class NarrowPhase:
    class HydroelasticSDF(_HydroelasticSDF):
        """NarrowPhase-scoped hydroelastic API wrapper.

        This wrapper keeps the discoverable API path ``NarrowPhase.HydroelasticSDF``
        without aliasing the exact same class object that is already documented in
        ``newton.geometry``. Using a distinct subclass avoids duplicate Sphinx object
        descriptions under ``-W`` docs builds.
        """

        class Config(_HydroelasticSDF.Config):
            """Hydroelastic configuration namespace for NarrowPhase usage."""

    def __init__(
        self,
        *,
        max_candidate_pairs: int,
        max_triangle_pairs: int = 1000000,
        reduce_contacts: bool = True,
        device=None,
        shape_aabb_lower: wp.array(dtype=wp.vec3) | None = None,
        shape_aabb_upper: wp.array(dtype=wp.vec3) | None = None,
        shape_voxel_resolution: wp.array(dtype=wp.vec3i) | None = None,
        contact_writer_warp_func: Any | None = None,
        hydroelastic_sdf: _HydroelasticSDF | None = None,
        has_meshes: bool = True,
        has_heightfields: bool = False,
    ):
        """
        Initialize NarrowPhase with pre-allocated buffers.

        Args:
            max_candidate_pairs: Maximum number of candidate pairs from broad phase
            max_triangle_pairs: Maximum number of mesh triangle pairs (conservative estimate)
            reduce_contacts: Whether to reduce contacts for mesh-mesh and mesh-plane collisions.
                When True, uses shared memory contact reduction to select representative contacts.
                This improves performance and stability for meshes with many vertices. Defaults to True.
            device: Device to allocate buffers on
            shape_aabb_lower: Optional external AABB lower bounds array (if provided, AABBs won't be computed internally)
            shape_aabb_upper: Optional external AABB upper bounds array (if provided, AABBs won't be computed internally)
            shape_voxel_resolution: Optional per-shape voxel resolution array used for mesh/SDF and
                hydroelastic contact processing.
            contact_writer_warp_func: Optional custom contact writer function (first arg: ContactData, second arg: custom struct type)
            hydroelastic_sdf: Optional SDF hydroelastic instance. Set is_hydroelastic=True on shapes to enable hydroelastic collisions.
            has_meshes: Whether the scene contains any mesh shapes (GeoType.MESH). When False, mesh-related
                kernel launches are skipped, improving performance for scenes with only primitive shapes.
                Defaults to True for safety. Set to False when constructing from a model with no meshes.
            has_heightfields: Whether the scene contains any heightfield shapes (GeoType.HFIELD). When True,
                heightfield collision buffers and kernels are allocated. Defaults to False.
        """
        self.max_candidate_pairs = max_candidate_pairs
        self.max_triangle_pairs = max_triangle_pairs
        self.device = device
        self.reduce_contacts = reduce_contacts
        self.has_meshes = has_meshes
        self.has_heightfields = has_heightfields

        # Warn when running on CPU with meshes: mesh-mesh SDF contacts require CUDA
        is_gpu_device = wp.get_device(device).is_cuda
        if has_meshes and not is_gpu_device:
            warnings.warn(
                "NarrowPhase running on CPU: mesh-mesh contacts will be skipped "
                "(SDF-based mesh-mesh collision requires CUDA). "
                "Use a CUDA device for full mesh-mesh contact support.",
                stacklevel=2,
            )

        # Create contact reduction functions only when reduce_contacts is enabled, running on GPU, and has meshes
        # Contact reduction requires GPU for shared memory operations and is only used for mesh contacts
        if reduce_contacts and is_gpu_device and has_meshes:
            self.contact_reduction_funcs = ContactReductionFunctions()
            self.reduction_slot_count = self.contact_reduction_funcs.reduction_slot_count
        else:
            self.contact_reduction_funcs = None
            self.reduction_slot_count = 0
            self.reduce_contacts = False

        # Determine if we're using external AABBs
        self.external_aabb = shape_aabb_lower is not None and shape_aabb_upper is not None

        if self.external_aabb:
            # Use provided AABB arrays
            self.shape_aabb_lower = shape_aabb_lower
            self.shape_aabb_upper = shape_aabb_upper
        else:
            # Create empty AABB arrays (won't be used)
            with wp.ScopedDevice(device):
                self.shape_aabb_lower = wp.zeros(0, dtype=wp.vec3, device=device)
                self.shape_aabb_upper = wp.zeros(0, dtype=wp.vec3, device=device)
        self.shape_voxel_resolution = shape_voxel_resolution

        # Determine the writer function
        if contact_writer_warp_func is None:
            writer_func = write_contact_simple
        else:
            writer_func = contact_writer_warp_func

        self.tile_size_mesh_convex = 128
        self.tile_size_mesh_mesh = 256
        self.tile_size_mesh_plane = 512
        self.block_dim = 128

        # Create the appropriate kernel variants
        # Primitive kernel handles lightweight primitives and routes remaining pairs
        self.primitive_kernel = create_narrow_phase_primitive_kernel(writer_func)
        # GJK/MPR kernel handles remaining convex-convex pairs
        self.narrow_phase_kernel = create_narrow_phase_kernel_gjk_mpr(self.external_aabb, writer_func)

        # Create mesh kernels only when has_meshes=True
        if has_meshes:
            self.mesh_triangle_contacts_kernel = create_narrow_phase_process_mesh_triangle_contacts_kernel(writer_func)

            # Create mesh-plane and mesh-mesh kernels (contact_reduction_funcs=None disables reduction)
            self.mesh_plane_contacts_kernel = create_narrow_phase_process_mesh_plane_contacts_kernel(
                writer_func,
                contact_reduction_funcs=self.contact_reduction_funcs,
            )
            # Only create mesh-mesh SDF kernel on CUDA (uses __shared__ memory via func_native)
            if is_gpu_device:
                self.mesh_mesh_contacts_kernel = create_narrow_phase_process_mesh_mesh_contacts_kernel(
                    writer_func,
                    contact_reduction_funcs=self.contact_reduction_funcs,
                )
            else:
                self.mesh_mesh_contacts_kernel = None
        else:
            self.mesh_triangle_contacts_kernel = None
            self.mesh_plane_contacts_kernel = None
            self.mesh_mesh_contacts_kernel = None

        # Create heightfield kernels
        if has_heightfields:
            self.heightfield_triangle_contacts_kernel = create_heightfield_triangle_contacts_kernel(writer_func)
        else:
            self.heightfield_triangle_contacts_kernel = None

        # Create global contact reduction kernels for mesh-triangle contacts (only if has_meshes and reduce_contacts)
        if self.reduce_contacts and has_meshes:
            # Global contact reducer uses hardcoded BETA_THRESHOLD (0.1mm) same as shared-memory reduction
            # Slot layout: 6 spatial direction slots + 1 max-depth slot = 7 slots per key (VALUES_PER_KEY)
            self.export_reduced_contacts_kernel = create_export_reduced_contacts_kernel(writer_func)
            # Global contact reducer for mesh-triangle contacts
            # Capacity is based on max_triangle_pairs since that's the max contacts we might generate
            self.global_contact_reducer = GlobalContactReducer(max_triangle_pairs, device=device)
        else:
            self.export_reduced_contacts_kernel = None
            self.global_contact_reducer = None

        self.hydroelastic_sdf = hydroelastic_sdf

        # Pre-allocate all intermediate buffers.
        # Counters live in one consolidated array for efficient zeroing.
        with wp.ScopedDevice(device):
            n = 0  # counter index
            gjk_idx = n
            n += 1
            sdf_sdf_idx = n
            n += 1
            mesh_idx = n if has_meshes else None
            n += 5 if has_meshes else 0  # mesh, triangle, mesh_plane, mesh_plane_vtx, mesh_mesh
            hf_pairs_idx = n
            n += 1
            hf_cells_idx = n
            n += 1
            c = wp.zeros(n, dtype=wp.int32, device=device)
            self._counter_array = c

            self.gjk_candidate_pairs_count = c[gjk_idx : gjk_idx + 1]
            self.shape_pairs_sdf_sdf_count = c[sdf_sdf_idx : sdf_sdf_idx + 1]
            self.shape_pairs_heightfield_count = c[hf_pairs_idx : hf_pairs_idx + 1]
            self.heightfield_cell_pairs_count = c[hf_cells_idx : hf_cells_idx + 1]

            self.shape_pairs_mesh_count = c[mesh_idx : mesh_idx + 1] if has_meshes else None
            self.triangle_pairs_count = c[mesh_idx + 1 : mesh_idx + 2] if has_meshes else None
            self.shape_pairs_mesh_plane_count = c[mesh_idx + 2 : mesh_idx + 3] if has_meshes else None
            self.mesh_plane_vertex_total_count = c[mesh_idx + 3 : mesh_idx + 4] if has_meshes else None
            self.shape_pairs_mesh_mesh_count = c[mesh_idx + 4 : mesh_idx + 5] if has_meshes else None

            # Pair and work buffers
            self.gjk_candidate_pairs = wp.zeros(max_candidate_pairs, dtype=wp.vec2i, device=device)

            self.shape_pairs_mesh = wp.zeros(max_candidate_pairs, dtype=wp.vec2i, device=device) if has_meshes else None
            self.triangle_pairs = wp.zeros(max_triangle_pairs, dtype=wp.vec3i, device=device) if has_meshes else None
            self.shape_pairs_mesh_plane = (
                wp.zeros(max_candidate_pairs, dtype=wp.vec2i, device=device) if has_meshes else None
            )
            self.shape_pairs_mesh_plane_cumsum = (
                wp.zeros(max_candidate_pairs, dtype=wp.int32, device=device) if has_meshes else None
            )
            self.shape_pairs_mesh_mesh = (
                wp.zeros(max_candidate_pairs, dtype=wp.vec2i, device=device) if has_meshes else None
            )

            hf_n = max_candidate_pairs if has_heightfields else 1
            self.shape_pairs_heightfield = wp.zeros(hf_n, dtype=wp.vec2i, device=device)
            self.heightfield_cell_pairs = wp.zeros(hf_n * 20 if has_heightfields else 1, dtype=wp.vec4i, device=device)

            self.empty_tangent = None

            if hydroelastic_sdf is not None:
                self.shape_pairs_sdf_sdf = wp.zeros(hydroelastic_sdf.max_num_shape_pairs, dtype=wp.vec2i, device=device)
            else:
                # Empty arrays for when hydroelastic is disabled
                self.shape_pairs_sdf_sdf = None

        # Fixed thread count for kernel launches
        # Use a reasonable minimum for GPU occupancy (256 blocks = 32K threads)
        # but scale with expected workload to avoid massive overprovisioning.
        # 256 blocks provides good occupancy on most GPUs (2-4 blocks per SM).

        # Query GPU properties to compute appropriate thread limits
        device_obj = wp.get_device(device)
        if device_obj.is_cuda:
            # Use 4 blocks per SM as a reasonable upper bound for occupancy
            # This balances parallelism with resource utilization
            max_blocks_limit = device_obj.sm_count * 4
        else:
            # CPU fallback: use a conservative limit
            max_blocks_limit = 256

        candidate_blocks = (max_candidate_pairs + self.block_dim - 1) // self.block_dim
        min_blocks = 256  # 32K threads minimum for reasonable GPU occupancy on CUDA
        num_blocks = max(min_blocks, min(candidate_blocks, max_blocks_limit))
        self.total_num_threads = self.block_dim * num_blocks
        self.num_tile_blocks = num_blocks

    def launch_custom_write(
        self,
        *,
        candidate_pair: wp.array(dtype=wp.vec2i, ndim=1),  # Maybe colliding pairs
        candidate_pair_count: wp.array(dtype=wp.int32, ndim=1),  # Size one array
        shape_types: wp.array(dtype=wp.int32, ndim=1),  # All shape types, pairs index into it
        shape_data: wp.array(dtype=wp.vec4, ndim=1),  # Shape data (scale xyz, thickness w)
        shape_transform: wp.array(dtype=wp.transform, ndim=1),  # In world space
        shape_source: wp.array(dtype=wp.uint64, ndim=1),  # The index into the source array, type define by shape_types
        shape_sdf_data: wp.array(dtype=SDFData, ndim=1),  # SDF data structs for mesh shapes
        shape_contact_margin: wp.array(dtype=wp.float32, ndim=1),  # per-shape contact margin
        shape_collision_radius: wp.array(dtype=wp.float32, ndim=1),  # per-shape collision radius for AABB fallback
        shape_flags: wp.array(dtype=wp.int32, ndim=1),  # per-shape flags (includes ShapeFlags.HYDROELASTIC)
        shape_collision_aabb_lower: wp.array(dtype=wp.vec3, ndim=1),  # Local-space AABB lower bounds
        shape_collision_aabb_upper: wp.array(dtype=wp.vec3, ndim=1),  # Local-space AABB upper bounds
        shape_voxel_resolution: wp.array(dtype=wp.vec3i, ndim=1),  # Voxel grid resolution per shape
        shape_heightfield_data: wp.array(dtype=HeightfieldData, ndim=1) | None = None,
        heightfield_elevation_data: wp.array(dtype=wp.float32, ndim=1) | None = None,
        writer_data: Any,
        device=None,  # Device to launch on
    ):
        """
        Launch narrow phase collision detection with a custom contact writer struct.

        Args:
            candidate_pair: Array of potentially colliding shape pairs from broad phase
            candidate_pair_count: Single-element array containing the number of candidate pairs
            shape_types: Array of geometry types for all shapes
            shape_data: Array of vec4 containing scale (xyz) and thickness (w) for each shape
            shape_transform: Array of world-space transforms for each shape
            shape_source: Array of source pointers (mesh IDs, etc.) for each shape
            shape_sdf_data: Array of SDFData structs for mesh shapes
            shape_contact_margin: Array of contact margins for each shape
            shape_collision_radius: Array of collision radii for each shape (for AABB fallback for planes/meshes)
            shape_flags: Array of shape flags for each shape (includes ShapeFlags.HYDROELASTIC)
            shape_collision_aabb_lower: Local-space AABB lower bounds for each shape (for voxel binning)
            shape_collision_aabb_upper: Local-space AABB upper bounds for each shape (for voxel binning)
            shape_voxel_resolution: Voxel grid resolution for each shape (for voxel binning)
            writer_data: Custom struct instance for contact writing (type must match the custom writer function)
            device: Device to launch on
        """
        if device is None:
            device = self.device if self.device is not None else candidate_pair.device

        # Clear all counters with a single kernel launch (consolidated counter array)
        self._counter_array.zero_()

        # Stage 1: Launch primitive kernel for fast analytical collisions
        # This handles sphere-sphere, sphere-capsule, capsule-capsule, plane-sphere, plane-capsule
        # and routes remaining pairs to gjk_candidate_pairs and mesh buffers
        wp.launch(
            kernel=self.primitive_kernel,
            dim=self.total_num_threads,
            inputs=[
                candidate_pair,
                candidate_pair_count,
                shape_types,
                shape_data,
                shape_transform,
                shape_source,
                shape_contact_margin,
                shape_flags,
                writer_data,
                self.total_num_threads,
            ],
            outputs=[
                self.gjk_candidate_pairs,
                self.gjk_candidate_pairs_count,
                self.shape_pairs_mesh,
                self.shape_pairs_mesh_count,
                self.shape_pairs_mesh_plane,
                self.shape_pairs_mesh_plane_cumsum,
                self.shape_pairs_mesh_plane_count,
                self.mesh_plane_vertex_total_count,
                self.shape_pairs_mesh_mesh,
                self.shape_pairs_mesh_mesh_count,
                self.shape_pairs_sdf_sdf,
                self.shape_pairs_sdf_sdf_count,
                self.shape_pairs_heightfield,
                self.shape_pairs_heightfield_count,
            ],
            device=device,
            block_dim=self.block_dim,
        )

        # Stage 2: Launch GJK/MPR kernel for remaining convex pairs
        # These are pairs that couldn't be handled analytically (box, cylinder, cone, convex hull, etc.)
        # All routing has been done by the primitive kernel, so this kernel just does GJK/MPR.
        wp.launch(
            kernel=self.narrow_phase_kernel,
            dim=self.total_num_threads,
            inputs=[
                self.gjk_candidate_pairs,
                self.gjk_candidate_pairs_count,
                shape_types,
                shape_data,
                shape_transform,
                shape_source,
                shape_contact_margin,
                shape_collision_radius,
                self.shape_aabb_lower,
                self.shape_aabb_upper,
                writer_data,
                self.total_num_threads,
            ],
            device=device,
            block_dim=self.block_dim,
        )

        # Skip mesh-related kernels when no meshes are present in the scene
        # This avoids kernel launch overhead for primitive-only scenes
        if self.has_meshes:
            # Launch mesh-plane contact processing kernel
            packaged_mesh_plane_inputs = [
                shape_data,
                shape_transform,
                shape_source,
                shape_contact_margin,
                shape_collision_aabb_lower,
                shape_collision_aabb_upper,
                shape_voxel_resolution,
                self.shape_pairs_mesh_plane,
                self.shape_pairs_mesh_plane_count,
                writer_data,
                self.num_tile_blocks,
            ]
            if self.reduce_contacts:
                # With contact reduction - use tiled launch
                wp.launch_tiled(
                    kernel=self.mesh_plane_contacts_kernel,
                    dim=(self.num_tile_blocks,),
                    inputs=packaged_mesh_plane_inputs,
                    device=device,
                    block_dim=self.tile_size_mesh_plane,
                )
            else:
                # Without contact reduction - use regular launch
                wp.launch(
                    kernel=self.mesh_plane_contacts_kernel,
                    dim=self.total_num_threads,
                    inputs=packaged_mesh_plane_inputs,
                    device=device,
                    block_dim=self.block_dim,
                )

            # Launch mesh triangle overlap detection kernel
            second_dim = self.tile_size_mesh_convex if ENABLE_TILE_BVH_QUERY else 1
            wp.launch(
                kernel=narrow_phase_find_mesh_triangle_overlaps_kernel,
                dim=[self.num_tile_blocks, second_dim],
                inputs=[
                    shape_types,
                    shape_transform,
                    shape_source,
                    shape_contact_margin,
                    shape_data,
                    self.shape_pairs_mesh,
                    self.shape_pairs_mesh_count,
                    self.num_tile_blocks,  # Use num_tile_blocks as total_num_threads for tiled kernel
                ],
                outputs=[
                    self.triangle_pairs,
                    self.triangle_pairs_count,
                ],
                device=device,
                block_dim=self.tile_size_mesh_convex,
            )

            # Launch mesh triangle contact processing kernel
            if self.reduce_contacts:
                assert self.global_contact_reducer is not None
                # Use global contact reduction for mesh-triangle contacts
                # First, clear the reducer
                self.global_contact_reducer.clear_active()

                # Collect contacts into the reducer
                reducer_data = self.global_contact_reducer.get_data_struct()
                wp.launch(
                    kernel=mesh_triangle_contacts_to_reducer_kernel,
                    dim=self.total_num_threads,
                    inputs=[
                        shape_types,
                        shape_data,
                        shape_transform,
                        shape_source,
                        shape_contact_margin,
                        self.triangle_pairs,
                        self.triangle_pairs_count,
                        reducer_data,
                        self.total_num_threads,
                    ],
                    device=device,
                    block_dim=self.block_dim,
                )

                # Register buffered contacts to hashtable (uses hardcoded BETA_THRESHOLD)
                # This is a separate pass to reduce register pressure on the contact generation kernel
                wp.launch(
                    kernel=reduce_buffered_contacts_kernel,
                    dim=self.total_num_threads,
                    inputs=[
                        reducer_data,
                        shape_transform,
                        shape_collision_aabb_lower,
                        shape_collision_aabb_upper,
                        shape_voxel_resolution,
                        self.total_num_threads,
                    ],
                    device=device,
                    block_dim=self.block_dim,
                )

                # Export reduced contacts to writer
                wp.launch(
                    kernel=self.export_reduced_contacts_kernel,
                    dim=self.total_num_threads,
                    inputs=[
                        self.global_contact_reducer.hashtable.keys,
                        self.global_contact_reducer.ht_values,
                        self.global_contact_reducer.hashtable.active_slots,
                        self.global_contact_reducer.position_depth,
                        self.global_contact_reducer.normal,
                        self.global_contact_reducer.shape_pairs,
                        shape_types,
                        shape_data,
                        shape_contact_margin,
                        writer_data,
                        self.total_num_threads,
                    ],
                    device=device,
                    block_dim=self.block_dim,
                )
            else:
                # Direct contact processing without reduction
                wp.launch(
                    kernel=self.mesh_triangle_contacts_kernel,
                    dim=self.total_num_threads,
                    inputs=[
                        shape_types,
                        shape_data,
                        shape_transform,
                        shape_source,
                        shape_contact_margin,
                        self.triangle_pairs,
                        self.triangle_pairs_count,
                        writer_data,
                        self.total_num_threads,
                    ],
                    device=device,
                    block_dim=self.block_dim,
                )

            # Launch mesh-mesh contact processing kernel (only on CUDA with SDF data)
            # SDF-based mesh-mesh collision requires GPU: uses shared memory (launch_tiled)
            # and wp.Volume which only supports CUDA
            if shape_sdf_data.shape[0] > 0 and wp.get_device(device).is_cuda:
                wp.launch_tiled(
                    kernel=self.mesh_mesh_contacts_kernel,
                    dim=(self.num_tile_blocks,),
                    inputs=[
                        shape_data,
                        shape_transform,
                        shape_source,
                        shape_sdf_data,
                        shape_contact_margin,
                        shape_collision_aabb_lower,
                        shape_collision_aabb_upper,
                        shape_voxel_resolution,
                        self.shape_pairs_mesh_mesh,
                        self.shape_pairs_mesh_mesh_count,
                        writer_data,
                        self.num_tile_blocks,
                    ],
                    device=device,
                    block_dim=self.tile_size_mesh_mesh,
                )

        # Stage: Heightfield collision processing
        if self.has_heightfields and shape_heightfield_data is not None and heightfield_elevation_data is not None:
            # Midphase: find overlapping grid cells for each heightfield-convex pair
            wp.launch(
                kernel=heightfield_midphase_kernel,
                dim=self.total_num_threads,
                inputs=[
                    shape_types,
                    shape_transform,
                    shape_collision_radius,
                    shape_contact_margin,
                    shape_heightfield_data,
                    self.shape_pairs_heightfield,
                    self.shape_pairs_heightfield_count,
                    self.total_num_threads,
                ],
                outputs=[
                    self.heightfield_cell_pairs,
                    self.heightfield_cell_pairs_count,
                ],
                device=device,
                block_dim=self.block_dim,
            )

            # Process heightfield cell pairs into triangle contacts via GJK/MPR
            wp.launch(
                kernel=self.heightfield_triangle_contacts_kernel,
                dim=self.total_num_threads,
                inputs=[
                    shape_types,
                    shape_data,
                    shape_transform,
                    shape_source,
                    shape_contact_margin,
                    shape_heightfield_data,
                    heightfield_elevation_data,
                    self.heightfield_cell_pairs,
                    self.heightfield_cell_pairs_count,
                    writer_data,
                    self.total_num_threads,
                ],
                device=device,
                block_dim=self.block_dim,
            )

        if self.hydroelastic_sdf is not None:
            self.hydroelastic_sdf.launch(
                shape_sdf_data,
                shape_transform,
                shape_contact_margin,
                shape_collision_aabb_lower,
                shape_collision_aabb_upper,
                shape_voxel_resolution,
                self.shape_pairs_sdf_sdf,
                self.shape_pairs_sdf_sdf_count,
                writer_data,
            )

        # Verify no collision pipeline buffers overflowed
        wp.launch(
            kernel=verify_narrow_phase_buffers,
            dim=[1],
            inputs=[
                candidate_pair_count,
                candidate_pair.shape[0],
                self.gjk_candidate_pairs_count,
                self.gjk_candidate_pairs.shape[0],
                self.shape_pairs_mesh_count,
                self.shape_pairs_mesh.shape[0] if self.shape_pairs_mesh is not None else 0,
                self.triangle_pairs_count,
                self.triangle_pairs.shape[0] if self.triangle_pairs is not None else 0,
                self.shape_pairs_mesh_plane_count,
                self.shape_pairs_mesh_plane.shape[0] if self.shape_pairs_mesh_plane is not None else 0,
                self.shape_pairs_mesh_mesh_count,
                self.shape_pairs_mesh_mesh.shape[0] if self.shape_pairs_mesh_mesh is not None else 0,
                self.shape_pairs_heightfield_count,
                self.shape_pairs_heightfield.shape[0],
                self.heightfield_cell_pairs_count,
                self.heightfield_cell_pairs.shape[0],
                self.shape_pairs_sdf_sdf_count,
                self.shape_pairs_sdf_sdf.shape[0] if self.shape_pairs_sdf_sdf is not None else 0,
                writer_data.contact_count,
                writer_data.contact_max,
            ],
            device=device,
        )

    def launch(
        self,
        *,
        candidate_pair: wp.array(dtype=wp.vec2i, ndim=1),  # Maybe colliding pairs
        candidate_pair_count: wp.array(dtype=wp.int32, ndim=1),  # Size one array
        shape_types: wp.array(dtype=wp.int32, ndim=1),  # All shape types, pairs index into it
        shape_data: wp.array(dtype=wp.vec4, ndim=1),  # Shape data (scale xyz, thickness w)
        shape_transform: wp.array(dtype=wp.transform, ndim=1),  # In world space
        shape_source: wp.array(dtype=wp.uint64, ndim=1),  # The index into the source array, type define by shape_types
        shape_sdf_data: wp.array(dtype=SDFData, ndim=1),  # SDF data structs for mesh shapes
        shape_contact_margin: wp.array(dtype=wp.float32, ndim=1),  # per-shape contact margin
        shape_collision_radius: wp.array(dtype=wp.float32, ndim=1),  # per-shape collision radius for AABB fallback
        shape_flags: wp.array(dtype=wp.int32, ndim=1),  # per-shape flags (includes ShapeFlags.HYDROELASTIC)
        shape_collision_aabb_lower: wp.array(dtype=wp.vec3, ndim=1) | None = None,  # Local-space AABB lower bounds
        shape_collision_aabb_upper: wp.array(dtype=wp.vec3, ndim=1) | None = None,  # Local-space AABB upper bounds
        shape_voxel_resolution: wp.array(dtype=wp.vec3i, ndim=1),  # Voxel grid resolution per shape
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
        device=None,  # Device to launch on
        **kwargs,
    ):
        """
        Launch narrow phase collision detection on candidate pairs from broad phase.

        Args:
            candidate_pair: Array of potentially colliding shape pairs from broad phase
            candidate_pair_count: Single-element array containing the number of candidate pairs
            shape_types: Array of geometry types for all shapes
            shape_data: Array of vec4 containing scale (xyz) and thickness (w) for each shape
            shape_transform: Array of world-space transforms for each shape
            shape_source: Array of source pointers (mesh IDs, etc.) for each shape
            shape_sdf_data: Array of SDFData structs for mesh shapes
            shape_contact_margin: Array of contact margins for each shape
            shape_collision_radius: Array of collision radii for each shape (for AABB fallback for planes/meshes)
            shape_collision_aabb_lower: Local-space AABB lower bounds for each shape (for voxel binning)
            shape_collision_aabb_upper: Local-space AABB upper bounds for each shape (for voxel binning)
            shape_voxel_resolution: Voxel grid resolution for each shape (for voxel binning)
            contact_pair: Output array for contact shape pairs
            contact_position: Output array for contact positions (center point)
            contact_normal: Output array for contact normals
            contact_penetration: Output array for penetration depths
            contact_tangent: Output array for contact tangents, or None to disable tangent computation
            contact_count: Output array (single element) for contact count
            device: Device to launch on
        """
        if device is None:
            device = self.device if self.device is not None else candidate_pair.device

        # Backward compatibility for older call sites/tests that still pass
        # shape_local_aabb_lower/upper.
        shape_local_aabb_lower = kwargs.pop("shape_local_aabb_lower", None)
        shape_local_aabb_upper = kwargs.pop("shape_local_aabb_upper", None)
        if kwargs:
            unknown_keys = sorted(kwargs.keys())
            if len(unknown_keys) == 1:
                raise TypeError(f"NarrowPhase.launch() got an unexpected keyword argument '{unknown_keys[0]}'")
            unknown = ", ".join(unknown_keys)
            raise TypeError(f"NarrowPhase.launch() got unexpected keyword arguments: {unknown}")

        if shape_collision_aabb_lower is None:
            shape_collision_aabb_lower = shape_local_aabb_lower
        if shape_collision_aabb_upper is None:
            shape_collision_aabb_upper = shape_local_aabb_upper
        if shape_collision_aabb_lower is None or shape_collision_aabb_upper is None:
            raise TypeError(
                "NarrowPhase.launch() missing required AABB bounds: provide either "
                "shape_collision_aabb_lower/shape_collision_aabb_upper or "
                "shape_local_aabb_lower/shape_local_aabb_upper"
            )

        contact_max = contact_pair.shape[0]

        # Handle optional tangent array - use empty array if None
        if contact_tangent is None:
            contact_tangent = self.empty_tangent

        # Clear external contact count (internal counters are cleared in launch_custom_write)
        contact_count.zero_()

        # Create ContactWriterData struct
        writer_data = ContactWriterData()
        writer_data.contact_max = contact_max
        writer_data.contact_count = contact_count
        writer_data.contact_pair = contact_pair
        writer_data.contact_position = contact_position
        writer_data.contact_normal = contact_normal
        writer_data.contact_penetration = contact_penetration
        writer_data.contact_tangent = contact_tangent

        # Delegate to launch_custom_write
        self.launch_custom_write(
            candidate_pair=candidate_pair,
            candidate_pair_count=candidate_pair_count,
            shape_types=shape_types,
            shape_data=shape_data,
            shape_transform=shape_transform,
            shape_source=shape_source,
            shape_sdf_data=shape_sdf_data,
            shape_contact_margin=shape_contact_margin,
            shape_collision_radius=shape_collision_radius,
            shape_flags=shape_flags,
            shape_collision_aabb_lower=shape_collision_aabb_lower,
            shape_collision_aabb_upper=shape_collision_aabb_upper,
            shape_voxel_resolution=shape_voxel_resolution,
            writer_data=writer_data,
            device=device,
        )
