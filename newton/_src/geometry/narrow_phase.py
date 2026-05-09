# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any

import warp as wp

from ..core.types import MAXVAL, Devicelike
from ..geometry.collision_core import (
    ENABLE_TILE_BVH_QUERY,
    check_infinite_plane_bsphere_overlap,
    compute_bounding_sphere_from_aabb,
    compute_tight_aabb_from_support,
    create_compute_gjk_mpr_contacts,
    create_find_contacts,
    get_triangle_shape_from_mesh,
    mesh_vs_convex_midphase,
    post_process_minkowski_only,
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
from ..geometry.contact_data import SHAPE_PAIR_HFIELD_BIT, ContactData, contact_passes_gap_check, make_contact_sort_key
from ..geometry.contact_reduction_global import (
    GlobalContactReducer,
    create_export_reduced_contacts_kernel,
    mesh_triangle_contacts_to_reducer_kernel,
    reduce_buffered_contacts_kernel,
    write_contact_to_reducer,
)
from ..geometry.contact_sort import ContactSorter
from ..geometry.flags import ShapeFlags
from ..geometry.sdf_contact import (
    MESH_SDF_BLOCK_DIM,
    compute_block_counts_from_weights,
    compute_mesh_mesh_block_offsets_scan,
    create_narrow_phase_process_mesh_mesh_contacts_kernel,
)
from ..geometry.sdf_hydroelastic import HydroelasticSDF
from ..geometry.sdf_texture import TextureSDFData
from ..geometry.support_function import (
    GeoTypeEx,
    SupportMapDataProvider,
    extract_shape_data,
    support_map_lean,
)
from ..geometry.types import GeoType
from ..utils.heightfield import (
    HeightfieldData,
    get_triangle_shape_from_heightfield,
    heightfield_vs_convex_midphase,
)


@wp.struct
class ContactWriterData:
    contact_max: int
    contact_count: wp.array[int]
    contact_pair: wp.array[wp.vec2i]
    contact_position: wp.array[wp.vec3]
    contact_normal: wp.array[wp.vec3]
    contact_penetration: wp.array[float]
    contact_tangent: wp.array[wp.vec3]
    contact_sort_key: wp.array[wp.int64]


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
        output_index: If -1, use atomic_add to get the next available index if contact distance is less than gap_sum. If >= 0, use this index directly and skip gap check.
    """
    total_separation_needed = (
        contact_data.radius_eff_a + contact_data.radius_eff_b + contact_data.margin_a + contact_data.margin_b
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
        if d >= contact_data.gap_sum:
            return
        index = wp.atomic_add(writer_data.contact_count, 0, 1)
    else:
        index = output_index
    if index >= writer_data.contact_max:
        return

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

    if writer_data.contact_sort_key.shape[0] > 0:
        writer_data.contact_sort_key[index] = make_contact_sort_key(
            contact_data.shape_a, contact_data.shape_b, contact_data.sort_sub_key
        )


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
    _module = f"narrow_phase_primitive_{writer_func.__name__}"

    @wp.kernel(enable_backward=False, module=_module)
    def narrow_phase_primitive_kernel(
        candidate_pair: wp.array[wp.vec2i],
        candidate_pair_count: wp.array[int],
        shape_types: wp.array[int],
        shape_data: wp.array[wp.vec4],
        shape_transform: wp.array[wp.transform],
        shape_source: wp.array[wp.uint64],
        shape_gap: wp.array[float],
        shape_flags: wp.array[wp.int32],
        writer_data: Any,
        total_num_threads: int,
        # Output: pairs that need GJK/MPR processing
        gjk_candidate_pairs: wp.array[wp.vec2i],
        gjk_candidate_pairs_count: wp.array[int],
        # Output: mesh collision pairs (for mesh processing)
        shape_pairs_mesh: wp.array[wp.vec2i],
        shape_pairs_mesh_count: wp.array[int],
        # Output: mesh-plane collision pairs
        shape_pairs_mesh_plane: wp.array[wp.vec2i],
        shape_pairs_mesh_plane_cumsum: wp.array[int],
        shape_pairs_mesh_plane_count: wp.array[int],
        mesh_plane_vertex_total_count: wp.array[int],
        # Output: mesh-mesh collision pairs
        shape_pairs_mesh_mesh: wp.array[wp.vec2i],
        shape_pairs_mesh_mesh_count: wp.array[int],
        # Output: sdf-sdf hydroelastic collision pairs
        shape_pairs_sdf_sdf: wp.array[wp.vec2i],
        shape_pairs_sdf_sdf_count: wp.array[int],
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
            margin_offset_a = data_a[3]
            margin_offset_b = data_b[3]

            # Get transforms
            X_a = shape_transform[shape_a]
            X_b = shape_transform[shape_b]
            pos_a = wp.transform_get_translation(X_a)
            pos_b = wp.transform_get_translation(X_b)
            quat_a = wp.transform_get_rotation(X_a)
            quat_b = wp.transform_get_rotation(X_b)
            gap_a = shape_gap[shape_a]
            gap_b = shape_gap[shape_b]
            gap_sum = gap_a + gap_b

            # =====================================================================
            # Route heightfield pairs.
            # Heightfield-vs-mesh and heightfield-vs-heightfield go through the
            # mesh-mesh SDF kernel (on-the-fly triangle + SDF evaluation).
            # Other heightfield combinations (convex, plane) use the dedicated
            # heightfield midphase with GJK/MPR per cell.
            # =====================================================================
            is_hfield_a = type_a == GeoType.HFIELD
            is_hfield_b = type_b == GeoType.HFIELD

            if is_hfield_a or is_hfield_b:
                is_mesh_like_a = type_a == GeoType.MESH or is_hfield_a
                is_mesh_like_b = type_b == GeoType.MESH or is_hfield_b

                if is_mesh_like_a and is_mesh_like_b:
                    # Heightfield-vs-heightfield is unsupported in this path.
                    if is_hfield_a and is_hfield_b:
                        continue
                    # Normalize order so heightfield (if present) is always pair[0],
                    # and mark pair[0] with a high-bit flag consumed by the SDF kernel.
                    if is_hfield_b:
                        encoded_a = shape_b | SHAPE_PAIR_HFIELD_BIT
                        encoded_b = shape_a
                    elif is_hfield_a:
                        encoded_a = shape_a | SHAPE_PAIR_HFIELD_BIT
                        encoded_b = shape_b
                    else:
                        encoded_a = shape_a
                        encoded_b = shape_b
                    idx = wp.atomic_add(shape_pairs_mesh_mesh_count, 0, 1)
                    if idx < shape_pairs_mesh_mesh.shape[0]:
                        shape_pairs_mesh_mesh[idx] = wp.vec2i(encoded_a, encoded_b)
                    continue

                # All other heightfield pairs: route through mesh midphase + GJK/MPR.
                # Normalize so the heightfield is always pair[0].
                if is_hfield_a:
                    hf_pair = wp.vec2i(shape_a, shape_b)
                else:
                    hf_pair = wp.vec2i(shape_b, shape_a)
                idx = wp.atomic_add(shape_pairs_mesh_count, 0, 1)
                if idx < shape_pairs_mesh.shape[0]:
                    shape_pairs_mesh[idx] = hf_pair
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

            # Initialize contact result storage (up to 4 contacts).
            # Distances default to MAXVAL so unused slots are automatically
            # excluded by the unified num_contacts count after the if/elif chain.
            contact_dist_0 = float(MAXVAL)
            contact_dist_1 = float(MAXVAL)
            contact_dist_2 = float(MAXVAL)
            contact_dist_3 = float(MAXVAL)
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

            # -----------------------------------------------------------------
            # Plane-Box collision (type_a=PLANE=0, type_b=BOX=6)
            # Produces up to 4 contacts
            # -----------------------------------------------------------------
            elif is_plane_a and is_box_b:
                plane_normal = wp.quat_rotate(quat_a, wp.vec3(0.0, 0.0, 1.0))
                box_rot = wp.quat_to_matrix(quat_b)
                box_size = scale_b

                dists4_box, positions4_box, contact_normal = collide_plane_box(
                    plane_normal, pos_a, pos_b, box_rot, box_size, gap_sum
                )

                contact_dist_0 = dists4_box[0]
                contact_dist_1 = dists4_box[1]
                contact_dist_2 = dists4_box[2]
                contact_dist_3 = dists4_box[3]
                contact_pos_0 = wp.vec3(positions4_box[0, 0], positions4_box[0, 1], positions4_box[0, 2])
                contact_pos_1 = wp.vec3(positions4_box[1, 0], positions4_box[1, 1], positions4_box[1, 2])
                contact_pos_2 = wp.vec3(positions4_box[2, 0], positions4_box[2, 1], positions4_box[2, 2])
                contact_pos_3 = wp.vec3(positions4_box[3, 0], positions4_box[3, 1], positions4_box[3, 2])

            # -----------------------------------------------------------------
            # Sphere-Sphere collision (type_a=SPHERE=2, type_b=SPHERE=2)
            # -----------------------------------------------------------------
            elif is_sphere_a and is_sphere_b:
                radius_a = scale_a[0]
                radius_b = scale_b[0]
                contact_dist_0, contact_pos_0, contact_normal = collide_sphere_sphere(pos_a, radius_a, pos_b, radius_b)

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
                contact_dist_1 = dists[1]
                contact_pos_1 = wp.vec3(positions[1, 0], positions[1, 1], positions[1, 2])

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

            # =====================================================================
            # Write all contacts (single write block for 0 to 4 contacts)
            # =====================================================================
            num_contacts = (
                int(contact_dist_0 < MAXVAL)
                + int(contact_dist_1 < MAXVAL)
                + int(contact_dist_2 < MAXVAL)
                + int(contact_dist_3 < MAXVAL)
            )
            if num_contacts > 0:
                # Prepare contact data (shared fields for both contacts)
                contact_data = ContactData()
                contact_data.contact_normal_a_to_b = contact_normal
                contact_data.radius_eff_a = radius_eff_a
                contact_data.radius_eff_b = radius_eff_b
                contact_data.margin_a = margin_offset_a
                contact_data.margin_b = margin_offset_b
                contact_data.shape_a = shape_a
                contact_data.shape_b = shape_b
                contact_data.gap_sum = gap_sum

                # Check margin for all possible contacts
                contact_0_valid = False
                if contact_dist_0 < MAXVAL:
                    contact_data.contact_point_center = contact_pos_0
                    contact_data.contact_distance = contact_dist_0
                    contact_0_valid = contact_passes_gap_check(contact_data)

                contact_1_valid = False
                if contact_dist_1 < MAXVAL:
                    contact_data.contact_point_center = contact_pos_1
                    contact_data.contact_distance = contact_dist_1
                    contact_1_valid = contact_passes_gap_check(contact_data)

                contact_2_valid = False
                if contact_dist_2 < MAXVAL:
                    contact_data.contact_point_center = contact_pos_2
                    contact_data.contact_distance = contact_dist_2
                    contact_2_valid = contact_passes_gap_check(contact_data)

                contact_3_valid = False
                if contact_dist_3 < MAXVAL:
                    contact_data.contact_point_center = contact_pos_3
                    contact_data.contact_distance = contact_dist_3
                    contact_3_valid = contact_passes_gap_check(contact_data)

                # Count valid contacts and allocate consecutive indices
                num_valid = int(contact_0_valid) + int(contact_1_valid) + int(contact_2_valid) + int(contact_3_valid)
                if num_valid > 0:
                    base_index = wp.atomic_add(writer_data.contact_count, 0, num_valid)
                    # Do not invoke the writer callback for overflowing batches.
                    # This keeps user-provided writers safe while still preserving
                    # overflow visibility via contact_count > contact_max.
                    if base_index + num_valid > writer_data.contact_max:
                        continue

                    # Write first contact if valid
                    if contact_0_valid:
                        contact_data.contact_point_center = contact_pos_0
                        contact_data.contact_distance = contact_dist_0
                        contact_data.sort_sub_key = 0
                        writer_func(contact_data, writer_data, base_index)
                        base_index += 1

                    # Write second contact if valid
                    if contact_1_valid:
                        contact_data.contact_point_center = contact_pos_1
                        contact_data.contact_distance = contact_dist_1
                        contact_data.sort_sub_key = 1
                        writer_func(contact_data, writer_data, base_index)
                        base_index += 1

                    # Write third contact if valid
                    if contact_2_valid:
                        contact_data.contact_point_center = contact_pos_2
                        contact_data.contact_distance = contact_dist_2
                        contact_data.sort_sub_key = 2
                        writer_func(contact_data, writer_data, base_index)
                        base_index += 1

                    # Write fourth contact if valid
                    if contact_3_valid:
                        contact_data.contact_point_center = contact_pos_3
                        contact_data.contact_distance = contact_dist_3
                        contact_data.sort_sub_key = 3
                        writer_func(contact_data, writer_data, base_index)

                continue

            # =====================================================================
            # Route remaining pairs to GJK/MPR kernel
            # =====================================================================
            idx = wp.atomic_add(gjk_candidate_pairs_count, 0, 1)
            if idx < gjk_candidate_pairs.shape[0]:
                gjk_candidate_pairs[idx] = wp.vec2i(shape_a, shape_b)

    return narrow_phase_primitive_kernel


def create_narrow_phase_kernel_gjk_mpr(
    external_aabb: bool, writer_func: Any, support_func: Any = None, post_process_contact: Any = None
):
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
    _sf = support_func.__name__ if support_func is not None else "default"
    _ppc = post_process_contact.__name__ if post_process_contact is not None else "default"
    _module = f"narrow_phase_gjk_mpr_{external_aabb}_{writer_func.__name__}_{_sf}_{_ppc}"

    @wp.kernel(enable_backward=False, module=_module)
    def narrow_phase_kernel_gjk_mpr(
        candidate_pair: wp.array[wp.vec2i],
        candidate_pair_count: wp.array[int],
        shape_types: wp.array[int],
        shape_data: wp.array[wp.vec4],
        shape_transform: wp.array[wp.transform],
        shape_source: wp.array[wp.uint64],
        shape_gap: wp.array[float],
        shape_collision_radius: wp.array[float],
        shape_aabb_lower: wp.array[wp.vec3],
        shape_aabb_upper: wp.array[wp.vec3],
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
            pos_a, quat_a, shape_data_a, scale_a, margin_offset_a = extract_shape_data(
                shape_a, shape_transform, shape_types, shape_data, shape_source
            )
            pos_b, quat_b, shape_data_b, scale_b, margin_offset_b = extract_shape_data(
                shape_b, shape_transform, shape_types, shape_data, shape_source
            )

            # Check for infinite planes
            is_infinite_plane_a = (type_a == GeoType.PLANE) and (scale_a[0] == 0.0 and scale_a[1] == 0.0)
            is_infinite_plane_b = (type_b == GeoType.PLANE) and (scale_b[0] == 0.0 and scale_b[1] == 0.0)

            # Early exit: both infinite planes can't collide
            if is_infinite_plane_a and is_infinite_plane_b:
                continue

            # Bounding sphere check is only needed for infinite plane pairs.
            # For non-plane pairs with external AABBs, SAP already verified AABB overlap.
            bsphere_radius_a = float(0.0)
            bsphere_radius_b = float(0.0)
            has_infinite_plane = is_infinite_plane_a or is_infinite_plane_b

            if has_infinite_plane:
                # Compute or fetch AABBs for bounding sphere overlap check
                if wp.static(external_aabb):
                    aabb_a_lower = shape_aabb_lower[shape_a]
                    aabb_a_upper = shape_aabb_upper[shape_a]
                    aabb_b_lower = shape_aabb_lower[shape_b]
                    aabb_b_upper = shape_aabb_upper[shape_b]
                if wp.static(not external_aabb):
                    gap_a = shape_gap[shape_a]
                    gap_b = shape_gap[shape_b]
                    gap_vec_a = wp.vec3(gap_a, gap_a, gap_a)
                    gap_vec_b = wp.vec3(gap_b, gap_b, gap_b)

                    # Shape A AABB
                    if is_infinite_plane_a:
                        radius_a = shape_collision_radius[shape_a]
                        half_extents_a = wp.vec3(radius_a, radius_a, radius_a)
                        aabb_a_lower = pos_a - half_extents_a - gap_vec_a
                        aabb_a_upper = pos_a + half_extents_a + gap_vec_a
                    else:
                        data_provider = SupportMapDataProvider()
                        aabb_a_lower, aabb_a_upper = compute_tight_aabb_from_support(
                            shape_data_a, quat_a, pos_a, data_provider
                        )
                        aabb_a_lower = aabb_a_lower - gap_vec_a
                        aabb_a_upper = aabb_a_upper + gap_vec_a

                    # Shape B AABB
                    if is_infinite_plane_b:
                        radius_b = shape_collision_radius[shape_b]
                        half_extents_b = wp.vec3(radius_b, radius_b, radius_b)
                        aabb_b_lower = pos_b - half_extents_b - gap_vec_b
                        aabb_b_upper = pos_b + half_extents_b + gap_vec_b
                    else:
                        data_provider = SupportMapDataProvider()
                        aabb_b_lower, aabb_b_upper = compute_tight_aabb_from_support(
                            shape_data_b, quat_b, pos_b, data_provider
                        )
                        aabb_b_lower = aabb_b_lower - gap_vec_b
                        aabb_b_upper = aabb_b_upper + gap_vec_b

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

            # Compute pairwise gap sum for contact detection
            gap_a = shape_gap[shape_a]
            gap_b = shape_gap[shape_b]
            gap_sum = gap_a + gap_b

            # Find and write contacts using GJK/MPR
            wp.static(
                create_find_contacts(writer_func, support_func=support_func, post_process_contact=post_process_contact)
            )(
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
                gap_sum,
                shape_a,
                shape_b,
                margin_offset_a,
                margin_offset_b,
                writer_data,
            )

    return narrow_phase_kernel_gjk_mpr


@wp.kernel(enable_backward=False)
def narrow_phase_find_mesh_triangle_overlaps_kernel(
    shape_types: wp.array[int],
    shape_transform: wp.array[wp.transform],
    shape_source: wp.array[wp.uint64],
    shape_gap: wp.array[float],  # Per-shape contact gaps
    shape_data: wp.array[wp.vec4],  # Shape data (scale xyz, margin w)
    shape_collision_radius: wp.array[float],
    shape_collision_aabb_lower: wp.array[wp.vec3],  # Local-space AABB lower bounds
    shape_collision_aabb_upper: wp.array[wp.vec3],  # Local-space AABB upper bounds
    shape_heightfield_index: wp.array[wp.int32],
    heightfield_data: wp.array[HeightfieldData],
    shape_pairs_mesh: wp.array[wp.vec2i],
    shape_pairs_mesh_count: wp.array[int],
    total_num_threads: int,
    # outputs
    triangle_pairs: wp.array[wp.vec3i],  # (shape_a, shape_b, triangle_idx)
    triangle_pairs_count: wp.array[int],
):
    """Find triangles that overlap with a convex shape for mesh and heightfield pairs.

    For mesh pairs, uses a tiled BVH query. For heightfield pairs, projects the
    convex shape's bounding sphere onto the heightfield grid and emits triangle
    pairs for each overlapping cell.

    Outputs triples of ``(mesh_or_hfield_shape, other_shape, triangle_idx)``.
    """
    tid, j = wp.tid()

    num_mesh_pairs = shape_pairs_mesh_count[0]

    # Strided loop over mesh pairs
    for i in range(tid, num_mesh_pairs, total_num_threads):
        pair = shape_pairs_mesh[i]
        shape_a = pair[0]
        shape_b = pair[1]

        type_a = shape_types[shape_a]
        type_b = shape_types[shape_b]

        # -----------------------------------------------------------------
        # Heightfield-vs-convex midphase (grid cell lookup)
        # Pairs are normalized so the heightfield is always shape_a.
        # -----------------------------------------------------------------
        if type_a == GeoType.HFIELD:
            # Only run on j==0; the j dimension is for tiled BVH queries (mesh only).
            if j != 0:
                continue
            hfd = heightfield_data[shape_heightfield_index[shape_a]]
            heightfield_vs_convex_midphase(
                shape_a,
                shape_b,
                hfd,
                shape_transform,
                shape_collision_aabb_lower,
                shape_collision_aabb_upper,
                shape_gap,
                triangle_pairs,
                triangle_pairs_count,
            )
            continue

        # -----------------------------------------------------------------
        # Mesh-vs-convex midphase (BVH query)
        # -----------------------------------------------------------------
        mesh_shape = -1
        non_mesh_shape = -1

        if type_a == GeoType.MESH and type_b != GeoType.MESH:
            mesh_shape = shape_a
            non_mesh_shape = shape_b
        elif type_b == GeoType.MESH and type_a != GeoType.MESH:
            mesh_shape = shape_b
            non_mesh_shape = shape_a
        else:
            # Mesh-mesh collision not supported in this path
            continue

        # Get mesh BVH ID and mesh transform
        mesh_id = shape_source[mesh_shape]
        if mesh_id == wp.uint64(0):
            continue

        # Get mesh world transform
        X_mesh_ws = shape_transform[mesh_shape]

        # Get non-mesh shape world transform
        X_ws = shape_transform[non_mesh_shape]

        # Use per-shape contact gaps for consistent pairwise thresholding.
        gap_non_mesh = shape_gap[non_mesh_shape]
        gap_mesh = shape_gap[mesh_shape]
        gap_sum = gap_non_mesh + gap_mesh

        # Call mesh_vs_convex_midphase with the shape_data and pair gap sum.
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
            gap_sum,
            triangle_pairs,
            triangle_pairs_count,
        )


def create_narrow_phase_process_mesh_triangle_contacts_kernel(writer_func: Any):
    _module = f"narrow_phase_mesh_tri_{writer_func.__name__}"

    @wp.kernel(enable_backward=False, module=_module)
    def narrow_phase_process_mesh_triangle_contacts_kernel(
        shape_types: wp.array[int],
        shape_data: wp.array[wp.vec4],
        shape_transform: wp.array[wp.transform],
        shape_source: wp.array[wp.uint64],
        shape_gap: wp.array[float],  # Per-shape contact gaps
        shape_heightfield_index: wp.array[wp.int32],
        heightfield_data: wp.array[HeightfieldData],
        heightfield_elevations: wp.array[wp.float32],
        triangle_pairs: wp.array[wp.vec3i],
        triangle_pairs_count: wp.array[int],
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

            type_a = shape_types[shape_a]

            if type_a == GeoType.HFIELD:
                # Heightfield triangle
                hfd = heightfield_data[shape_heightfield_index[shape_a]]
                X_ws_a = shape_transform[shape_a]
                shape_data_a, v0_world = get_triangle_shape_from_heightfield(
                    hfd, heightfield_elevations, X_ws_a, tri_idx
                )
            else:
                # Mesh triangle
                mesh_id_a = shape_source[shape_a]
                scale_data_a = shape_data[shape_a]
                mesh_scale_a = wp.vec3(scale_data_a[0], scale_data_a[1], scale_data_a[2])
                X_ws_a = shape_transform[shape_a]
                shape_data_a, v0_world = get_triangle_shape_from_mesh(mesh_id_a, mesh_scale_a, X_ws_a, tri_idx)

            # Extract shape B data
            pos_b, quat_b, shape_data_b, _scale_b, margin_offset_b = extract_shape_data(
                shape_b,
                shape_transform,
                shape_types,
                shape_data,
                shape_source,
            )

            # Triangle position is vertex A in world space.
            # For heightfield prisms, edges are in heightfield-local space
            # so we pass the heightfield rotation to let MPR/GJK work in
            # that frame (where -Z is always the down axis).
            pos_a = v0_world
            if type_a == GeoType.HFIELD:
                quat_a = wp.transform_get_rotation(X_ws_a)
            else:
                quat_a = wp.quat_identity()

            # Back-face culling: skip when the entire convex shape is behind
            # the triangle face.  TRIANGLE_PRISM (heightfields) handles
            # this via its extruded support function.
            if shape_data_a.shape_type == int(GeoTypeEx.TRIANGLE):
                face_normal = wp.cross(shape_data_a.scale, shape_data_a.auxiliary)
                # Signed distance of shape B's center from triangle plane
                center_dist = wp.dot(face_normal, pos_b - pos_a)
                if center_dist < 0.0:
                    continue

            # Extract margin offset for shape A (signed distance padding)
            margin_offset_a = shape_data[shape_a][3]

            # Sum per-shape contact gaps for consistent pairwise thresholding
            gap_a = shape_gap[shape_a]
            gap_b = shape_gap[shape_b]
            gap_sum = gap_a + gap_b

            # Compute and write contacts using GJK/MPR with standard post-processing
            wp.static(create_compute_gjk_mpr_contacts(writer_func))(
                shape_data_a,
                shape_data_b,
                quat_a,
                quat_b,
                pos_a,
                pos_b,
                gap_sum,
                shape_a,
                shape_b,
                margin_offset_a,
                margin_offset_b,
                writer_data,
                (tri_idx << 1) | 1,
            )

    return narrow_phase_process_mesh_triangle_contacts_kernel


@wp.kernel(enable_backward=False)
def compute_mesh_plane_vert_counts(
    shape_pairs_mesh_plane: wp.array[wp.vec2i],
    shape_pairs_mesh_plane_count: wp.array[int],
    shape_source: wp.array[wp.uint64],
    vert_counts: wp.array[wp.int32],
):
    """Compute per-pair vertex counts in parallel for mesh-plane pairs.

    Slots beyond ``pair_count`` are zeroed for correct ``array_scan`` results.
    """
    i = wp.tid()
    pair_count = wp.min(shape_pairs_mesh_plane_count[0], shape_pairs_mesh_plane.shape[0])
    if i >= pair_count:
        vert_counts[i] = 0
        return

    pair = shape_pairs_mesh_plane[i]
    mesh_shape = pair[0]
    mesh_id = shape_source[mesh_shape]
    pair_verts = int(0)
    if mesh_id != wp.uint64(0):
        pair_verts = wp.mesh_get(mesh_id).points.shape[0]
    vert_counts[i] = wp.int32(pair_verts)


def compute_mesh_plane_block_offsets_scan(
    shape_pairs_mesh_plane: wp.array,
    shape_pairs_mesh_plane_count: wp.array,
    shape_source: wp.array,
    target_blocks: int,
    block_offsets: wp.array,
    block_counts: wp.array,
    weight_prefix_sums: wp.array,
    device: str | None = None,
    record_tape: bool = True,
):
    """Compute mesh-plane block offsets using parallel kernels and array_scan."""
    n = block_counts.shape[0]
    # Step 1: compute per-pair vertex counts in parallel
    wp.launch(
        kernel=compute_mesh_plane_vert_counts,
        dim=n,
        inputs=[
            shape_pairs_mesh_plane,
            shape_pairs_mesh_plane_count,
            shape_source,
            block_counts,  # reuse as temp storage for vert counts
        ],
        device=device,
        record_tape=record_tape,
    )
    # Step 2: inclusive scan to get total
    wp.utils.array_scan(block_counts, weight_prefix_sums, inclusive=True)
    # Step 3: compute per-pair block counts using adaptive threshold
    wp.launch(
        kernel=compute_block_counts_from_weights,
        dim=n,
        inputs=[
            weight_prefix_sums,
            block_counts,  # still holds vert counts
            shape_pairs_mesh_plane_count,
            shape_pairs_mesh_plane.shape[0],
            target_blocks,
            block_offsets,  # reuse as temp for block counts
        ],
        device=device,
        record_tape=record_tape,
    )
    # Step 4: exclusive scan of block counts → block_offsets
    wp.utils.array_scan(block_offsets, block_offsets, inclusive=False)


def create_narrow_phase_process_mesh_plane_contacts_kernel(
    writer_func: Any,
    reduce_contacts: bool = False,
):
    """
    Create a mesh-plane collision kernel.

    Args:
        writer_func: Contact writer function (e.g., write_contact_simple)
        reduce_contacts: If True, return multi-block load-balanced variant for global reduction.

    Returns:
        A warp kernel that processes mesh-plane collisions
    """
    _module = f"narrow_phase_mesh_plane_{writer_func.__name__}_{reduce_contacts}"

    @wp.kernel(enable_backward=False, module=_module)
    def narrow_phase_process_mesh_plane_contacts_kernel(
        shape_data: wp.array[wp.vec4],
        shape_transform: wp.array[wp.transform],
        shape_source: wp.array[wp.uint64],
        shape_gap: wp.array[float],
        _shape_collision_aabb_lower: wp.array[wp.vec3],  # Unused but kept for API compatibility
        _shape_collision_aabb_upper: wp.array[wp.vec3],  # Unused but kept for API compatibility
        _shape_voxel_resolution: wp.array[wp.vec3i],  # Unused but kept for API compatibility
        shape_pairs_mesh_plane: wp.array[wp.vec2i],
        shape_pairs_mesh_plane_count: wp.array[int],
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

            # Extract per-shape margin offsets (stored in shape_data.w)
            margin_offset_mesh = shape_data[mesh_shape][3]
            margin_offset_plane = shape_data[plane_shape][3]
            total_margin_offset = margin_offset_mesh + margin_offset_plane

            # Use per-shape contact gap for contact detection threshold
            gap_mesh = shape_gap[mesh_shape]
            gap_plane = shape_gap[plane_shape]
            gap_sum = gap_mesh + gap_plane

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
                if distance < gap_sum + total_margin_offset:
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
                    contact_data.margin_a = margin_offset_mesh
                    contact_data.margin_b = margin_offset_plane
                    contact_data.shape_a = mesh_shape
                    contact_data.shape_b = plane_shape
                    contact_data.gap_sum = gap_sum
                    contact_data.sort_sub_key = vertex_idx

                    if writer_data.contact_count[0] < writer_data.contact_max:
                        writer_func(contact_data, writer_data, -1)

    # Return early if contact reduction is disabled
    if not reduce_contacts:
        return narrow_phase_process_mesh_plane_contacts_kernel

    @wp.kernel(enable_backward=False, module=_module)
    def narrow_phase_process_mesh_plane_contacts_reduce_kernel(
        shape_data: wp.array[wp.vec4],
        shape_transform: wp.array[wp.transform],
        shape_source: wp.array[wp.uint64],
        shape_gap: wp.array[float],
        _shape_collision_aabb_lower: wp.array[wp.vec3],
        _shape_collision_aabb_upper: wp.array[wp.vec3],
        _shape_voxel_resolution: wp.array[wp.vec3i],
        shape_pairs_mesh_plane: wp.array[wp.vec2i],
        shape_pairs_mesh_plane_count: wp.array[int],
        block_offsets: wp.array[wp.int32],
        writer_data: Any,
        total_num_blocks: int,
    ):
        """Process mesh-plane collisions with dynamic load balancing.

        Multiple GPU blocks can be assigned to the same mesh-plane pair
        based on vertex count.  Contacts are written directly to the
        global contact reducer buffer via ``writer_func``; reduction into
        the hashtable happens in a separate pass.  This avoids per-block
        shared-memory reduction and unifies the contact reduction path
        with the one used for mesh-mesh contacts.
        """
        block_id, t = wp.tid()

        pair_count = wp.min(shape_pairs_mesh_plane_count[0], shape_pairs_mesh_plane.shape[0])
        total_combos = block_offsets[pair_count]

        # Grid stride loop over (pair, sub-block) combos for multi-block load balancing.
        for combo_idx in range(block_id, total_combos, total_num_blocks):
            # Binary search block_offsets to find the pair for this block
            lo = int(0)
            hi = int(pair_count)
            while lo < hi:
                mid = (lo + hi) // 2
                if block_offsets[mid + 1] <= combo_idx:
                    lo = mid + 1
                else:
                    hi = mid
            pair_idx = int(lo)
            pair_block_start = block_offsets[pair_idx]
            block_in_pair = combo_idx - pair_block_start
            blocks_for_pair = block_offsets[pair_idx + 1] - pair_block_start

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

            # Compute vertex range for this sub-block
            chunk_size = (num_vertices + blocks_for_pair - 1) // blocks_for_pair
            vert_start = block_in_pair * chunk_size
            vert_end = wp.min(vert_start + chunk_size, num_vertices)

            # Get mesh world transform
            X_mesh_ws = shape_transform[mesh_shape]

            # Get plane world transform
            X_plane_ws = shape_transform[plane_shape]
            X_plane_sw = wp.transform_inverse(X_plane_ws)

            # Get plane normal in world space (plane normal is along local +Z)
            plane_normal = wp.transform_vector(X_plane_ws, wp.vec3(0.0, 0.0, 1.0))

            # Get mesh scale
            scale_data = shape_data[mesh_shape]
            mesh_scale = wp.vec3(scale_data[0], scale_data[1], scale_data[2])

            # Extract per-shape margin offsets (stored in shape_data.w)
            margin_offset_mesh = shape_data[mesh_shape][3]
            margin_offset_plane = shape_data[plane_shape][3]
            total_margin_offset = margin_offset_mesh + margin_offset_plane

            # Use per-shape contact gap for contact detection threshold
            gap_mesh = shape_gap[mesh_shape]
            gap_plane = shape_gap[plane_shape]
            gap_sum = gap_mesh + gap_plane

            # Process this block's chunk of vertices — write contacts directly
            # to the global reducer buffer (no per-block shared memory reduction).
            chunk_len = vert_end - vert_start
            num_iterations = (chunk_len + wp.block_dim() - 1) // wp.block_dim()
            for i in range(num_iterations):
                vertex_idx = vert_start + i * wp.block_dim() + t

                if vertex_idx < vert_end:
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
                    if distance < gap_sum + total_margin_offset:
                        # Contact position is the midpoint
                        contact_pos = (vertex_world + point_on_plane) * 0.5

                        # Normal points from mesh to plane
                        contact_normal = -plane_normal

                        contact_data = ContactData()
                        contact_data.contact_point_center = contact_pos
                        contact_data.contact_normal_a_to_b = contact_normal
                        contact_data.contact_distance = distance
                        contact_data.radius_eff_a = 0.0
                        contact_data.radius_eff_b = 0.0
                        contact_data.margin_a = margin_offset_mesh
                        contact_data.margin_b = margin_offset_plane
                        contact_data.shape_a = mesh_shape
                        contact_data.shape_b = plane_shape
                        contact_data.gap_sum = gap_sum
                        contact_data.sort_sub_key = vertex_idx

                        writer_func(contact_data, writer_data, -1)

    return narrow_phase_process_mesh_plane_contacts_reduce_kernel


# =============================================================================
# Verification kernel
# =============================================================================


@wp.kernel(enable_backward=False)
def verify_narrow_phase_buffers(
    broad_phase_count: wp.array[int],
    max_broad_phase: int,
    gjk_count: wp.array[int],
    max_gjk: int,
    mesh_count: wp.array[int],
    max_mesh: int,
    triangle_count: wp.array[int],
    max_triangle: int,
    mesh_plane_count: wp.array[int],
    max_mesh_plane: int,
    mesh_mesh_count: wp.array[int],
    max_mesh_mesh: int,
    sdf_sdf_count: wp.array[int],
    max_sdf_sdf: int,
    contact_count: wp.array[int],
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
    """Resolve broad-phase shape pairs into simulation contacts.

    This class orchestrates the narrow-phase collision pipeline by launching the
    appropriate Warp kernels for primitive, mesh, heightfield, and SDF shape
    pairs. It owns the intermediate counters and pair buffers used while
    processing candidate pairs, then writes final contacts through a configurable
    contact writer function.
    """

    def __init__(
        self,
        *,
        max_candidate_pairs: int,
        max_triangle_pairs: int = 1000000,
        reduce_contacts: bool = True,
        device: Devicelike | None = None,
        shape_aabb_lower: wp.array[wp.vec3] | None = None,
        shape_aabb_upper: wp.array[wp.vec3] | None = None,
        shape_voxel_resolution: wp.array[wp.vec3i] | None = None,
        contact_writer_warp_func: Any | None = None,
        hydroelastic_sdf: HydroelasticSDF | None = None,
        has_meshes: bool = True,
        has_heightfields: bool = False,
        use_lean_gjk_mpr: bool = False,
        deterministic: bool = False,
        contact_max: int | None = None,
        verify_buffers: bool = True,
    ) -> None:
        """
        Initialize NarrowPhase with pre-allocated buffers.

        Args:
            max_candidate_pairs: Maximum number of candidate pairs from broad phase
            max_triangle_pairs: Maximum number of triangle pairs for mesh and
                heightfield collisions (conservative estimate).
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
            deterministic: Sort contacts after the narrow phase so that results are
                independent of GPU thread scheduling.  Adds a radix sort + gather
                pass.  Hydroelastic contacts are not yet covered.
            contact_max: Maximum number of contacts for the deterministic sort buffer.
                Must match the ``contact_pair`` array size passed to :meth:`launch`.
                Defaults to ``max_candidate_pairs``.  Set this to a larger value when
                a single candidate pair can emit multiple contacts (e.g. up to 4 for
                primitive multi-contact paths).
            verify_buffers: When True (the default), launch a ``dim=[1]``
                diagnostic kernel (:func:`verify_narrow_phase_buffers`) at the
                end of :meth:`launch` that compares each public counter on this
                class (``gjk_candidate_pairs_count``, ``shape_pairs_mesh_count``,
                ``triangle_pairs_count``, ``shape_pairs_mesh_plane_count``,
                ``shape_pairs_mesh_mesh_count``, ``shape_pairs_sdf_sdf_count``)
                and the output ``contact_count`` against the capacity of its
                backing array, printing ``wp.printf`` warnings on overflow.
                Users who want a programmatic overflow hook can disable this and
                read those counters themselves.  Overhead is one extra kernel
                launch per collision pass (roughly a few µs of launch latency on
                CUDA; the kernel body is a handful of scalar comparisons on one
                thread).  Disable in hot loops or CUDA graph capture once buffer
                sizes are known to be adequate.
        """
        self.max_candidate_pairs = max_candidate_pairs
        self.max_triangle_pairs = max_triangle_pairs
        self.device = device
        self.reduce_contacts = reduce_contacts
        self.has_meshes = has_meshes
        self.has_heightfields = has_heightfields
        self.deterministic = deterministic
        self.verify_buffers = verify_buffers
        device_obj = wp.get_device(device)

        # Contact reduction requires either meshes or heightfields (the
        # mesh/heightfield-triangle path feeds the global reducer, so
        # heightfield-only scenes still benefit from reduction).
        if reduce_contacts and not (has_meshes or has_heightfields):
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

        # CPU kernels currently observe ``wp.block_dim() == 1`` regardless
        # of the plain ``wp.launch(..., block_dim=N)`` parameter (Warp
        # GH-1413). Keep the mesh-convex midphase launch grid, tile shape,
        # and kernel-side ``wp.block_dim()`` in sync on CPU.
        self.tile_size_mesh_convex = 1 if device_obj.is_cpu else 128
        # Must match ``MESH_SDF_BLOCK_DIM`` in sdf_contact.py: the mesh-SDF
        # kernels assume ``wp.block_dim()`` equals that constant so the
        # tile-stack overflow margin (``STACK_CAPACITY = 2 *
        # MESH_SDF_BLOCK_DIM``) is correctly sized. Re-use the constant
        # rather than duplicating the value so the two can't drift.
        self.tile_size_mesh_mesh = MESH_SDF_BLOCK_DIM
        assert self.tile_size_mesh_mesh == MESH_SDF_BLOCK_DIM, (
            "mesh-SDF tile launches must use block_dim == MESH_SDF_BLOCK_DIM"
        )
        self.tile_size_mesh_plane = 512
        # Generic block dim for non-tile-stack kernels (primitive /
        # GJK-MPR / export). Not used for the mesh-SDF tile launches,
        # which use ``self.tile_size_mesh_mesh`` above.
        #
        # Plain ``wp.launch`` does not auto-clamp ``block_dim`` on CPU like
        # ``wp.launch_tiled`` does. Match the kernel-observed value so
        # strided-loop and tile-index calculations cannot run past the CPU
        # launch geometry.
        self.block_dim = 1 if device_obj.is_cpu else 128

        # Create the appropriate kernel variants
        # Primitive kernel handles lightweight primitives and routes remaining pairs
        self.primitive_kernel = create_narrow_phase_primitive_kernel(writer_func)
        # GJK/MPR kernel handles remaining convex-convex pairs
        if use_lean_gjk_mpr:
            # Use lean support function (CONVEX_MESH, BOX, SPHERE only) and lean post-processing
            # (skip axial shape rolling stabilization) to reduce GPU i-cache pressure
            self.narrow_phase_kernel = create_narrow_phase_kernel_gjk_mpr(
                self.external_aabb,
                writer_func,
                support_func=support_map_lean,
                post_process_contact=post_process_minkowski_only,
            )
        else:
            self.narrow_phase_kernel = create_narrow_phase_kernel_gjk_mpr(self.external_aabb, writer_func)
        # Create triangle contacts kernel when meshes or heightfields are present
        if has_meshes or has_heightfields:
            self.mesh_triangle_contacts_kernel = create_narrow_phase_process_mesh_triangle_contacts_kernel(writer_func)
        else:
            self.mesh_triangle_contacts_kernel = None

        # Create mesh-specific kernels only when has_meshes=True
        if has_meshes:
            # Create mesh-plane kernel.
            # When reducing, use multi-block load balancing and write contacts to the
            # global reducer buffer (same path as mesh-mesh and mesh-triangle).
            if self.reduce_contacts:
                self.mesh_plane_contacts_kernel = create_narrow_phase_process_mesh_plane_contacts_kernel(
                    write_contact_to_reducer,
                    reduce_contacts=True,
                )
            else:
                self.mesh_plane_contacts_kernel = create_narrow_phase_process_mesh_plane_contacts_kernel(
                    writer_func,
                )
            if self.reduce_contacts:
                self.mesh_mesh_contacts_kernel = create_narrow_phase_process_mesh_mesh_contacts_kernel(
                    write_contact_to_reducer,
                    enable_heightfields=has_heightfields,
                    reduce_contacts=True,
                )
            else:
                self.mesh_mesh_contacts_kernel = create_narrow_phase_process_mesh_mesh_contacts_kernel(
                    writer_func,
                    enable_heightfields=has_heightfields,
                )
        else:
            self.mesh_plane_contacts_kernel = None
            self.mesh_mesh_contacts_kernel = None

        # Create global contact reduction kernels for mesh/heightfield-triangle
        # contacts (mirror the predicate used to gate ``self.reduce_contacts``
        # above so heightfield-only scenes also get the reducer allocated).
        if self.reduce_contacts and (has_meshes or has_heightfields):
            # Global contact reducer uses hardcoded BETA_THRESHOLD (0.1mm) same as shared-memory reduction
            # Slot layout: NUM_SPATIAL_DIRECTIONS spatial + 1 max-depth = VALUES_PER_KEY slots per key
            self.export_reduced_contacts_kernel = create_export_reduced_contacts_kernel(writer_func)
            # Global contact reducer for all mesh contact types
            self.global_contact_reducer = GlobalContactReducer(
                max_triangle_pairs, device=device, deterministic=deterministic
            )
        else:
            self.export_reduced_contacts_kernel = None
            self.global_contact_reducer = None

        self.hydroelastic_sdf = hydroelastic_sdf

        # Pre-allocate all intermediate buffers.
        # Counters live in one consolidated array for efficient zeroing.
        with wp.ScopedDevice(device):
            has_mesh_like = has_meshes or has_heightfields
            n = 0  # counter index
            gjk_idx = n
            n += 1
            sdf_sdf_idx = n
            n += 1
            mesh_like_idx = n if has_mesh_like else None
            n += 2 if has_mesh_like else 0  # mesh_like pairs, triangle pairs
            mesh_only_idx = n if has_meshes else None
            n += 3 if has_meshes else 0  # mesh_plane, mesh_plane_vtx, mesh_mesh
            c = wp.zeros(n, dtype=wp.int32, device=device)
            self._counter_array = c

            self.gjk_candidate_pairs_count = c[gjk_idx : gjk_idx + 1]
            self.shape_pairs_sdf_sdf_count = c[sdf_sdf_idx : sdf_sdf_idx + 1]
            self.shape_pairs_mesh_count = c[mesh_like_idx : mesh_like_idx + 1] if has_mesh_like else None
            self.triangle_pairs_count = c[mesh_like_idx + 1 : mesh_like_idx + 2] if has_mesh_like else None
            self.shape_pairs_mesh_plane_count = c[mesh_only_idx : mesh_only_idx + 1] if has_meshes else None
            self.mesh_plane_vertex_total_count = c[mesh_only_idx + 1 : mesh_only_idx + 2] if has_meshes else None
            self.shape_pairs_mesh_mesh_count = c[mesh_only_idx + 2 : mesh_only_idx + 3] if has_meshes else None

            # Pair and work buffers
            self.gjk_candidate_pairs = wp.zeros(max_candidate_pairs, dtype=wp.vec2i, device=device)

            self.shape_pairs_mesh = (
                wp.zeros(max_candidate_pairs, dtype=wp.vec2i, device=device) if has_mesh_like else None
            )
            self.triangle_pairs = (
                wp.zeros(max_triangle_pairs, dtype=wp.vec3i, device=device) if has_meshes or has_heightfields else None
            )
            self.shape_pairs_mesh_plane = (
                wp.zeros(max_candidate_pairs, dtype=wp.vec2i, device=device) if has_meshes else None
            )
            self.shape_pairs_mesh_plane_cumsum = (
                wp.zeros(max_candidate_pairs, dtype=wp.int32, device=device) if has_meshes else None
            )
            self.shape_pairs_mesh_mesh = (
                wp.zeros(max_candidate_pairs, dtype=wp.vec2i, device=device) if has_meshes else None
            )

            self.empty_tangent = None
            self._empty_sort_key = wp.zeros(0, dtype=wp.int64, device=device)
            det_capacity = contact_max if contact_max is not None else max_candidate_pairs
            if deterministic:
                self._sort_key_array = wp.zeros(det_capacity, dtype=wp.int64, device=device)
                self._contact_sorter = ContactSorter(det_capacity, device=device)
            else:
                self._sort_key_array = wp.zeros(0, dtype=wp.int64, device=device)
                self._contact_sorter = None
            # Sentinel edge buffers used when no edge data is provided.
            # _empty_edge_range is indexed by shape id, so it must have one
            # slot per shape (not per candidate pair).
            num_shapes = shape_aabb_lower.shape[0] if shape_aabb_lower is not None else max_candidate_pairs
            self._empty_edge_indices = wp.zeros(1, dtype=wp.vec2i, device=device)
            self._empty_edge_range = wp.full(max(num_shapes, 1), (-1, 0), dtype=wp.vec2i, device=device)

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

        # Dynamic block allocation for mesh-mesh and mesh-plane contacts.
        # On CUDA we target ~4 blocks per SM for good occupancy; on CPU
        # there is no SM notion so we pick 64 as a modest parallelism
        # target that splits pair work across OpenMP threads without
        # over-subscribing on small scenes.
        if self.reduce_contacts:
            target_blocks = device_obj.sm_count * 4 if device_obj.is_cuda else 64
            n = max_candidate_pairs + 1
            # Mesh-mesh
            self.num_mesh_mesh_blocks = target_blocks
            self.mesh_mesh_target_blocks = target_blocks
            self.mesh_mesh_block_offsets = wp.zeros(n, dtype=wp.int32, device=device)
            self.mesh_mesh_block_counts = wp.zeros(n, dtype=wp.int32, device=device)
            self.mesh_mesh_weight_prefix_sums = wp.zeros(n, dtype=wp.int32, device=device)
            # Mesh-plane
            self.num_mesh_plane_blocks = target_blocks
            self.mesh_plane_target_blocks = target_blocks
            self.mesh_plane_block_offsets = wp.zeros(n, dtype=wp.int32, device=device)
            self.mesh_plane_block_counts = wp.zeros(n, dtype=wp.int32, device=device)
            self.mesh_plane_weight_prefix_sums = wp.zeros(n, dtype=wp.int32, device=device)
        else:
            self.num_mesh_mesh_blocks = self.num_tile_blocks
            self.mesh_mesh_target_blocks = self.num_tile_blocks
            self.mesh_mesh_block_offsets = None
            self.mesh_mesh_block_counts = None
            self.mesh_mesh_weight_prefix_sums = None
            self.num_mesh_plane_blocks = self.num_tile_blocks
            self.mesh_plane_target_blocks = self.num_tile_blocks
            self.mesh_plane_block_offsets = None
            self.mesh_plane_block_counts = None
            self.mesh_plane_weight_prefix_sums = None

    def launch_custom_write(
        self,
        *,
        candidate_pair: wp.array[wp.vec2i],  # Maybe colliding pairs
        candidate_pair_count: wp.array[wp.int32],  # Size one array
        shape_types: wp.array[wp.int32],  # All shape types, pairs index into it
        shape_data: wp.array[wp.vec4],  # Shape data (scale xyz, margin w)
        shape_transform: wp.array[wp.transform],  # In world space
        shape_source: wp.array[wp.uint64],  # The index into the source array, type define by shape_types
        shape_sdf_index: wp.array[wp.int32],  # Per-shape index into texture_sdf_data (-1 for none)
        shape_gap: wp.array[wp.float32],  # per-shape contact gap (detection threshold)
        shape_collision_radius: wp.array[wp.float32],  # per-shape collision radius for AABB fallback
        shape_flags: wp.array[wp.int32],  # per-shape flags (includes ShapeFlags.HYDROELASTIC)
        shape_collision_aabb_lower: wp.array[wp.vec3],  # Local-space AABB lower bounds
        shape_collision_aabb_upper: wp.array[wp.vec3],  # Local-space AABB upper bounds
        shape_voxel_resolution: wp.array[wp.vec3i],  # Voxel grid resolution per shape
        texture_sdf_data: wp.array[TextureSDFData] | None = None,  # Compact texture SDF data table
        shape_heightfield_index: wp.array[wp.int32] | None = None,
        heightfield_data: wp.array[HeightfieldData] | None = None,
        heightfield_elevations: wp.array[wp.float32] | None = None,
        mesh_edge_indices: wp.array[wp.vec2i] | None = None,
        shape_edge_range: wp.array[wp.vec2i] | None = None,
        writer_data: Any,
        device: Devicelike | None = None,  # Device to launch on
    ) -> None:
        """
        Launch narrow phase collision detection with a custom contact writer struct.

        All internal kernel launches use ``record_tape=False`` so that calls
        are safe inside a :class:`warp.Tape` context.

        Args:
            candidate_pair: Array of potentially colliding shape pairs from broad phase
            candidate_pair_count: Single-element array containing the number of candidate pairs
            shape_types: Array of geometry types for all shapes
            shape_data: Array of vec4 containing scale (xyz) and margin (w) for each shape
            shape_transform: Array of world-space transforms for each shape
            shape_source: Array of source pointers (mesh IDs, etc.) for each shape
            shape_sdf_index: Per-shape SDF table index (-1 for shapes without SDF)
            texture_sdf_data: Compact array of TextureSDFData structs
            shape_gap: Array of per-shape contact gaps (detection threshold) for each shape
            shape_collision_radius: Array of collision radii for each shape (for AABB fallback for planes/meshes)
            shape_flags: Array of shape flags for each shape (includes ShapeFlags.HYDROELASTIC)
            shape_collision_aabb_lower: Local-space AABB lower bounds for each shape (for voxel binning)
            shape_collision_aabb_upper: Local-space AABB upper bounds for each shape (for voxel binning)
            shape_voxel_resolution: Voxel grid resolution for each shape (for voxel binning)
            mesh_edge_indices: Packed array of mesh edge vertex pairs for all shapes.
            shape_edge_range: Per-shape (start, count) into mesh_edge_indices.
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
                shape_gap,
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
            ],
            device=device,
            block_dim=self.block_dim,
            record_tape=False,
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
                shape_gap,
                shape_collision_radius,
                self.shape_aabb_lower,
                self.shape_aabb_upper,
                writer_data,
                self.total_num_threads,
            ],
            device=device,
            block_dim=self.block_dim,
            record_tape=False,
        )

        # Skip mesh/heightfield kernels when no meshes or heightfields are present
        if self.has_meshes or self.has_heightfields:
            # Launch mesh-plane contact processing kernel (meshes only)
            if self.has_meshes and not self.reduce_contacts:
                wp.launch(
                    kernel=self.mesh_plane_contacts_kernel,
                    dim=self.total_num_threads,
                    inputs=[
                        shape_data,
                        shape_transform,
                        shape_source,
                        shape_gap,
                        shape_collision_aabb_lower,
                        shape_collision_aabb_upper,
                        shape_voxel_resolution,
                        self.shape_pairs_mesh_plane,
                        self.shape_pairs_mesh_plane_count,
                        writer_data,
                        self.num_tile_blocks,
                    ],
                    device=device,
                    block_dim=self.block_dim,
                    record_tape=False,
                )

            # Launch midphase: finds overlapping triangles for both mesh and heightfield pairs
            second_dim = self.tile_size_mesh_convex if ENABLE_TILE_BVH_QUERY else 1
            wp.launch(
                kernel=narrow_phase_find_mesh_triangle_overlaps_kernel,
                dim=[self.num_tile_blocks, second_dim],
                inputs=[
                    shape_types,
                    shape_transform,
                    shape_source,
                    shape_gap,
                    shape_data,
                    shape_collision_radius,
                    shape_collision_aabb_lower,
                    shape_collision_aabb_upper,
                    shape_heightfield_index,
                    heightfield_data,
                    self.shape_pairs_mesh,
                    self.shape_pairs_mesh_count,
                    self.num_tile_blocks,
                ],
                outputs=[
                    self.triangle_pairs,
                    self.triangle_pairs_count,
                ],
                device=device,
                block_dim=self.tile_size_mesh_convex,
                record_tape=False,
            )

            # Launch contact processing for triangle pairs
            if self.reduce_contacts:
                # Unified global reduction for all mesh contact types.
                assert self.global_contact_reducer is not None
                self.global_contact_reducer.clear_active()
                reducer_data = self.global_contact_reducer.get_data_struct()

                # Mesh-plane contacts → global reducer (meshes only)
                if self.has_meshes:
                    compute_mesh_plane_block_offsets_scan(
                        shape_pairs_mesh_plane=self.shape_pairs_mesh_plane,
                        shape_pairs_mesh_plane_count=self.shape_pairs_mesh_plane_count,
                        shape_source=shape_source,
                        target_blocks=self.mesh_plane_target_blocks,
                        block_offsets=self.mesh_plane_block_offsets,
                        block_counts=self.mesh_plane_block_counts,
                        weight_prefix_sums=self.mesh_plane_weight_prefix_sums,
                        device=device,
                        record_tape=False,
                    )
                    wp.launch_tiled(
                        kernel=self.mesh_plane_contacts_kernel,
                        dim=(self.num_mesh_plane_blocks,),
                        inputs=[
                            shape_data,
                            shape_transform,
                            shape_source,
                            shape_gap,
                            shape_collision_aabb_lower,
                            shape_collision_aabb_upper,
                            shape_voxel_resolution,
                            self.shape_pairs_mesh_plane,
                            self.shape_pairs_mesh_plane_count,
                            self.mesh_plane_block_offsets,
                            reducer_data,
                            self.num_mesh_plane_blocks,
                        ],
                        device=device,
                        block_dim=self.tile_size_mesh_plane,
                        record_tape=False,
                    )

                # Mesh/heightfield-triangle contacts → same global reducer
                wp.launch(
                    kernel=mesh_triangle_contacts_to_reducer_kernel,
                    dim=self.total_num_threads,
                    inputs=[
                        shape_types,
                        shape_data,
                        shape_transform,
                        shape_source,
                        shape_gap,
                        shape_heightfield_index,
                        heightfield_data,
                        heightfield_elevations,
                        self.triangle_pairs,
                        self.triangle_pairs_count,
                        reducer_data,
                        self.total_num_threads,
                    ],
                    device=device,
                    block_dim=self.block_dim,
                    record_tape=False,
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
                        shape_gap,
                        shape_heightfield_index,
                        heightfield_data,
                        heightfield_elevations,
                        self.triangle_pairs,
                        self.triangle_pairs_count,
                        writer_data,
                        self.total_num_threads,
                    ],
                    device=device,
                    block_dim=self.block_dim,
                    record_tape=False,
                )

            # Register mesh-plane/mesh-triangle contacts in hashtable BEFORE mesh-mesh.
            # Mesh-mesh does inline hashtable registration in its kernel.
            if self.reduce_contacts:
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
                    record_tape=False,
                )

            # Launch mesh-mesh contact processing kernel.
            # The kernel uses texture SDF for fast sampling, with BVH fallback via shape_sdf_index,
            # as well as on-the-fly heightfield evaluation via heightfield_data.
            if texture_sdf_data is None:
                texture_sdf_data = wp.zeros(0, dtype=TextureSDFData, device=device)
            if mesh_edge_indices is None:
                mesh_edge_indices = self._empty_edge_indices
            if shape_edge_range is None:
                shape_edge_range = self._empty_edge_range

            if self.mesh_mesh_contacts_kernel is not None:
                if self.reduce_contacts and self.mesh_mesh_block_offsets is not None:
                    # Mesh-mesh contacts → buffer + inline hashtable registration
                    compute_mesh_mesh_block_offsets_scan(
                        shape_pairs_mesh_mesh=self.shape_pairs_mesh_mesh,
                        shape_pairs_mesh_mesh_count=self.shape_pairs_mesh_mesh_count,
                        shape_edge_range=shape_edge_range,
                        shape_heightfield_index=shape_heightfield_index,
                        heightfield_data=heightfield_data,
                        target_blocks=self.mesh_mesh_target_blocks,
                        block_offsets=self.mesh_mesh_block_offsets,
                        block_counts=self.mesh_mesh_block_counts,
                        weight_prefix_sums=self.mesh_mesh_weight_prefix_sums,
                        device=device,
                        record_tape=False,
                    )

                    wp.launch_tiled(
                        kernel=self.mesh_mesh_contacts_kernel,
                        dim=(self.num_mesh_mesh_blocks,),
                        inputs=[
                            shape_data,
                            shape_transform,
                            shape_source,
                            texture_sdf_data,
                            shape_sdf_index,
                            shape_gap,
                            shape_collision_aabb_lower,
                            shape_collision_aabb_upper,
                            shape_voxel_resolution,
                            self.shape_pairs_mesh_mesh,
                            self.shape_pairs_mesh_mesh_count,
                            shape_heightfield_index,
                            heightfield_data,
                            heightfield_elevations,
                            mesh_edge_indices,
                            shape_edge_range,
                            self.mesh_mesh_block_offsets,
                            reducer_data,
                            self.num_mesh_mesh_blocks,
                        ],
                        device=device,
                        block_dim=self.tile_size_mesh_mesh,
                        record_tape=False,
                    )
                else:
                    # Non-reduce fallback: direct contact write, no dynamic allocation
                    wp.launch_tiled(
                        kernel=self.mesh_mesh_contacts_kernel,
                        dim=(self.num_tile_blocks,),
                        inputs=[
                            shape_data,
                            shape_transform,
                            shape_source,
                            texture_sdf_data,
                            shape_sdf_index,
                            shape_gap,
                            shape_collision_aabb_lower,
                            shape_collision_aabb_upper,
                            shape_voxel_resolution,
                            self.shape_pairs_mesh_mesh,
                            self.shape_pairs_mesh_mesh_count,
                            shape_heightfield_index,
                            heightfield_data,
                            heightfield_elevations,
                            mesh_edge_indices,
                            shape_edge_range,
                            writer_data,
                            self.num_tile_blocks,
                        ],
                        device=device,
                        block_dim=self.tile_size_mesh_mesh,
                        record_tape=False,
                    )

            # Export reduced contacts from hashtable
            if self.reduce_contacts:
                # Zero exported_flags for cross-entry deduplication
                self.global_contact_reducer.exported_flags.zero_()
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
                        self.global_contact_reducer.contact_fingerprints,
                        self.global_contact_reducer.exported_flags,
                        shape_types,
                        shape_data,
                        shape_gap,
                        writer_data,
                        self.total_num_threads,
                        int(self.global_contact_reducer.deterministic),
                    ],
                    device=device,
                    block_dim=self.block_dim,
                    record_tape=False,
                )
        if self.hydroelastic_sdf is not None:
            self.hydroelastic_sdf.launch(
                texture_sdf_data,
                shape_sdf_index,
                shape_transform,
                shape_gap,
                shape_collision_aabb_lower,
                shape_collision_aabb_upper,
                shape_voxel_resolution,
                self.shape_pairs_sdf_sdf,
                self.shape_pairs_sdf_sdf_count,
                writer_data,
            )

        # Verify no collision pipeline buffers overflowed
        if self.verify_buffers:
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
                    self.shape_pairs_sdf_sdf_count,
                    self.shape_pairs_sdf_sdf.shape[0] if self.shape_pairs_sdf_sdf is not None else 0,
                    writer_data.contact_count,
                    writer_data.contact_max,
                ],
                device=device,
                record_tape=False,
            )

    def launch(
        self,
        *,
        candidate_pair: wp.array[wp.vec2i],  # Maybe colliding pairs
        candidate_pair_count: wp.array[wp.int32],  # Size one array
        shape_types: wp.array[wp.int32],  # All shape types, pairs index into it
        shape_data: wp.array[wp.vec4],  # Shape data (scale xyz, margin w)
        shape_transform: wp.array[wp.transform],  # In world space
        shape_source: wp.array[wp.uint64],  # The index into the source array, type define by shape_types
        shape_sdf_index: wp.array[wp.int32] | None = None,  # Per-shape index into texture_sdf_data (-1 for none)
        texture_sdf_data: wp.array[TextureSDFData] | None = None,  # Compact texture SDF data table
        shape_gap: wp.array[wp.float32],  # per-shape contact gap (detection threshold)
        shape_collision_radius: wp.array[wp.float32],  # per-shape collision radius for AABB fallback
        shape_flags: wp.array[wp.int32],  # per-shape flags (includes ShapeFlags.HYDROELASTIC)
        shape_collision_aabb_lower: wp.array[wp.vec3] | None = None,  # Local-space AABB lower bounds
        shape_collision_aabb_upper: wp.array[wp.vec3] | None = None,  # Local-space AABB upper bounds
        shape_voxel_resolution: wp.array[wp.vec3i],  # Voxel grid resolution per shape
        # Outputs
        contact_pair: wp.array[wp.vec2i],
        contact_position: wp.array[wp.vec3],
        contact_normal: wp.array[
            wp.vec3
        ],  # Pointing from pairId.x to pairId.y, represents z axis of local contact frame
        contact_penetration: wp.array[float],  # negative if bodies overlap
        contact_count: wp.array[int],  # Number of active contacts after narrow
        contact_tangent: wp.array[wp.vec3] | None = None,  # Represents x axis of local contact frame (None to disable)
        device: Devicelike | None = None,  # Device to launch on
        **kwargs: Any,
    ) -> None:
        """
        Launch narrow phase collision detection on candidate pairs from broad phase.

        Args:
            candidate_pair: Array of potentially colliding shape pairs from broad phase
            candidate_pair_count: Single-element array containing the number of candidate pairs
            shape_types: Array of geometry types for all shapes
            shape_data: Array of vec4 containing scale (xyz) and margin (w) for each shape
            shape_transform: Array of world-space transforms for each shape
            shape_source: Array of source pointers (mesh IDs, etc.) for each shape
            shape_sdf_index: Per-shape SDF table index (-1 for shapes without SDF)
            texture_sdf_data: Compact array of TextureSDFData structs
            shape_gap: Array of per-shape contact gaps (detection threshold) for each shape
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
        mesh_edge_indices = kwargs.pop("mesh_edge_indices", None)
        shape_edge_range = kwargs.pop("shape_edge_range", None)
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
        if shape_sdf_index is None:
            shape_sdf_index = wp.full(shape_types.shape[0], -1, dtype=wp.int32, device=device)

        contact_max = contact_pair.shape[0]

        # Handle optional tangent array - use empty array if None
        if contact_tangent is None:
            contact_tangent = self.empty_tangent

        # Clear external contact count (internal counters are cleared in launch_custom_write)
        contact_count.zero_()

        # Verify sort-key buffer and sorter match the contact output capacity.
        # Raising instead of silently reallocating keeps this path
        # CUDA-graph-capturable and consistent with CollisionPipeline.collide().
        if self.deterministic and self._sort_key_array.shape[0] != contact_max:
            raise ValueError(
                f"Contact output capacity ({contact_max}) does not match the "
                f"deterministic sort buffer size ({self._sort_key_array.shape[0]}). "
                f"The sorter operates over fixed-capacity buffers for CUDA graph capture "
                f"compatibility, so the sizes must match exactly."
            )

        # Create ContactWriterData struct
        sort_key_arr = self._sort_key_array if self.deterministic else self._empty_sort_key

        writer_data = ContactWriterData()
        writer_data.contact_max = contact_max
        writer_data.contact_count = contact_count
        writer_data.contact_pair = contact_pair
        writer_data.contact_position = contact_position
        writer_data.contact_normal = contact_normal
        writer_data.contact_penetration = contact_penetration
        writer_data.contact_tangent = contact_tangent
        writer_data.contact_sort_key = sort_key_arr

        # Delegate to launch_custom_write
        self.launch_custom_write(
            candidate_pair=candidate_pair,
            candidate_pair_count=candidate_pair_count,
            shape_types=shape_types,
            shape_data=shape_data,
            shape_transform=shape_transform,
            shape_source=shape_source,
            shape_sdf_index=shape_sdf_index,
            texture_sdf_data=texture_sdf_data,
            shape_gap=shape_gap,
            shape_collision_radius=shape_collision_radius,
            shape_flags=shape_flags,
            shape_collision_aabb_lower=shape_collision_aabb_lower,
            shape_collision_aabb_upper=shape_collision_aabb_upper,
            shape_voxel_resolution=shape_voxel_resolution,
            mesh_edge_indices=mesh_edge_indices,
            shape_edge_range=shape_edge_range,
            writer_data=writer_data,
            device=device,
        )

        if self.deterministic:
            self._contact_sorter.sort_simple(
                sort_key_arr,
                contact_count,
                contact_pair=contact_pair,
                contact_position=contact_position,
                contact_normal=contact_normal,
                contact_penetration=contact_penetration,
                contact_tangent=contact_tangent,
                device=device,
            )
