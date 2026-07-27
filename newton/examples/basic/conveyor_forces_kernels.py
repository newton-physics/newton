# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Prepare Warp buffers for conveyor belt contact-force kernels."""

import warp as wp

from newton.examples.basic import conveyor_forces_utils

# not needed for the purpose of this sample
wp.config.enable_backward = False


@wp.kernel
def prepare_buffers(
    parallel_body_processing_count: int,
    parallel_conveyor_belt_processing_count: int,
    body_count: int,
    conveyor_belt_count: int,
    dt: wp.float32,
    conveyor_belt_speed_startup_duration: wp.float32,
    body_positions: wp.indexedarray2d(dtype=wp.float32),
    body_orientations: wp.array2d[wp.float32],
    body_com_positions: wp.indexedarray2d(dtype=wp.float32),
    body_com_orientations: wp.array2d[wp.float32],
    body_inverse_inertias: wp.indexedarray2d(dtype=wp.float32),
    body_material_index_buffer: wp.array[wp.uint32],
    conveyor_belt_to_indices_map: wp.array2d[wp.uint32],
    material_pair_friction_table: wp.array2d[wp.float32],
    pair_contacts_count: wp.indexedarray2d(dtype=wp.uint32),
    pair_contacts_start_indices: wp.indexedarray2d(dtype=wp.uint32),
    # output
    body_to_world_transform_buffer: wp.array[wp.transform],
    body_inverse_inertia_buffer: wp.array[wp.mat33],
    point_to_indices_map: wp.array2d[wp.uint32],
    friction_coefficient_buffer: wp.array[wp.float32],
    total_contact_count: wp.array[wp.uint32],
    total_elapsed_time: wp.array[wp.float32],
    global_conveyor_belt_speed_scale: wp.array[wp.float32],
):
    """Fill buffers for subsequent conveyor belt force computation stages.

    For each body, computes the center-of-mass-to-world transform by composing the body
    transform with the center-of-mass local transform, rotates the body-frame inverse inertia
    tensor into world space, and fills ``point_to_indices_map`` with the body index and velocity
    field type and ID for every contact point between that body and each conveyor belt. Also
    looks up and stores the friction coefficient for each contact point and atomically accumulates
    the total contact count.

    Args:
        parallel_body_processing_count: Stride used to distribute bodies across threads.
        parallel_conveyor_belt_processing_count: Stride used to distribute belts across threads.
        body_count: Total number of tracked bodies (N).
        conveyor_belt_count: Total number of conveyor belt objects (M).
        dt: Simulation time-step in seconds.
        conveyor_belt_speed_startup_duration: Duration (in seconds) until the conveyor belts reach
            full speed.
        body_positions: (N, 3) world-space body positions (x, y, z).
        body_orientations: (N, 4) world-space body orientation quaternions (w, x, y, z).
        body_com_positions: (N, 3) body-local center-of-mass positions.
        body_com_orientations: (N, 4) body-local center-of-mass orientation quaternions (w, x, y, z).
        body_inverse_inertias: (N, 9) body-frame inverse inertia 3x3 matrices.
        body_material_index_buffer: (N,) material index for each body.
        conveyor_belt_to_indices_map: (M, 3) per-belt [velocity_field_type, velocity_field_id, material_index].
        material_pair_friction_table: 2-D friction lookup table indexed by [body_material, belt_material].
        pair_contacts_count: (N, M) number of contacts per body-belt pair.
        pair_contacts_start_indices: (N, M) start index into the contact buffers (positions, normals, forces, etc.)
            per body-belt pair.
        body_to_world_transform_buffer: Output - center-of-mass-to-world transform for each body.
        body_inverse_inertia_buffer: Output - world-space inverse inertia tensor for each body.
        point_to_indices_map: Output - (C, 3) per-contact [body_index, velocity_field_type, velocity_field_id].
        friction_coefficient_buffer: Output - per-contact friction coefficient.
        total_contact_count: Output - single-element array accumulating the total contact count.
        total_elapsed_time: Output - single-element array to track the accumulated simulation time.
        global_conveyor_belt_speed_scale: Output - single-element array holding a global conveyor belt speed
            scale.
    """
    body_index, initial_conveyor_belt_index = wp.tid()

    contact_count_sum = wp.uint32(0)

    if (body_index == 0) and (
        initial_conveyor_belt_index == 0
    ):  # this ensures only one thread writes the following properties
        total_elapsed_time[0] += dt

        if total_elapsed_time[0] > conveyor_belt_speed_startup_duration:
            global_conveyor_belt_speed_scale[0] = 1.0
        else:
            global_conveyor_belt_speed_scale[0] = total_elapsed_time[0] / conveyor_belt_speed_startup_duration

    while body_index < body_count:
        conveyor_belt_index = initial_conveyor_belt_index

        body_material_index = body_material_index_buffer[body_index]

        while conveyor_belt_index < conveyor_belt_count:
            contact_count = pair_contacts_count[body_index, conveyor_belt_index]
            contact_count_sum += contact_count

            start_index = pair_contacts_start_indices[body_index, conveyor_belt_index]
            end_index_plus_one = start_index + contact_count

            conveyor_belt_material_index = conveyor_belt_to_indices_map[conveyor_belt_index, 2]

            friction_coefficient = material_pair_friction_table[body_material_index, conveyor_belt_material_index]

            if conveyor_belt_index == 0:  # this ensures only one thread writes the body properties
                body_transform = wp.transform(
                    body_positions[body_index, 0],
                    body_positions[body_index, 1],
                    body_positions[body_index, 2],
                    body_orientations[body_index, 1],
                    body_orientations[body_index, 2],
                    body_orientations[body_index, 3],
                    body_orientations[body_index, 0],
                )

                com_transform = wp.transform(
                    body_com_positions[body_index, 0],
                    body_com_positions[body_index, 1],
                    body_com_positions[body_index, 2],
                    body_com_orientations[body_index, 1],
                    body_com_orientations[body_index, 2],
                    body_com_orientations[body_index, 3],
                    body_com_orientations[body_index, 0],
                )

                body_to_world = body_transform * com_transform

                body_to_world_transform_buffer[body_index] = body_to_world

                inertia_tensor_body_frame = wp.mat33(
                    body_inverse_inertias[body_index, 0],
                    body_inverse_inertias[body_index, 1],
                    body_inverse_inertias[body_index, 2],
                    body_inverse_inertias[body_index, 3],
                    body_inverse_inertias[body_index, 4],
                    body_inverse_inertias[body_index, 5],
                    body_inverse_inertias[body_index, 6],
                    body_inverse_inertias[body_index, 7],
                    body_inverse_inertias[body_index, 8],
                )

                body_to_world_rot_mat = wp.quat_to_matrix(body_to_world.q)

                inertia_tensor_world_frame = (
                    body_to_world_rot_mat * inertia_tensor_body_frame * wp.transpose(body_to_world_rot_mat)
                )

                body_inverse_inertia_buffer[body_index] = inertia_tensor_world_frame

            i = start_index
            while i < end_index_plus_one:
                point_to_indices_map[i, 0] = wp.uint32(body_index)
                point_to_indices_map[i, 1] = conveyor_belt_to_indices_map[conveyor_belt_index, 0]
                point_to_indices_map[i, 2] = conveyor_belt_to_indices_map[conveyor_belt_index, 1]

                friction_coefficient_buffer[i] = friction_coefficient

                i += wp.uint32(1)

            conveyor_belt_index += parallel_conveyor_belt_processing_count

        body_index += parallel_body_processing_count

    wp.atomic_add(total_contact_count, 0, contact_count_sum)


@wp.kernel
def clear_buffers(
    # output
    total_contact_count: wp.array[wp.uint32],
):
    """Warp kernel: reset the total contact count to zero at the start of each step."""
    total_contact_count[0] = wp.uint32(0)

    # note: it is not necessary to zero the per-body force buffers or per-contact force buffers since
    #       those get initialized or set for all registered rigid bodies and all contact points even
    #       if a rigid body had no contacts (see sum_up_force()) or if a contact is ignored see
    #       (velocity_field_compute_force()).


@wp.struct
class Patch:
    """Structure to represent contact patches of a rigid body vs. conveyor belt objects.

    A contact patch is a set of contact points that have similar contact normals. The patch
    tracks all points in a patch via a linked list. The patch also links to the next (uncorrelated)
    patch of the rigid body if there is any.
    """

    # the normal of the contact patch. Has unit length.
    normal: wp.vec3

    # the number of points assigned to the patch
    point_count: wp.uint32

    # the index of the next contact point that belongs to the patch.
    # INVALID_INDEX if there is no other point
    next_point: wp.uint32

    # the index of the last point belonging to this patch
    last_point: wp.uint32

    # the index of the next patch that has a different enough normal
    # compared to this patch. INVALID_INDEX if there is no other patch
    # for this group of patches.
    next_uncorrelated_patch: wp.uint32

    # the index of the conveyor belt object the patch belongs to or INVALID_INDEX
    # if the patch covers more than one conveyor belt object
    conveyor_belt_index: wp.uint32


SAME_NORMAL_THRESHOLD = 0.999  # around 2 degrees

INVALID_INDEX = wp.uint32(0xFFFFFFFF)


@wp.kernel
def correlate_and_filter_contact_points(
    parallel_body_processing_count: int,
    body_count: int,
    conveyor_belt_count: int,
    conveyor_belt_surface_normal_buffer: wp.array[wp.vec3],
    conveyor_belt_contact_processing_threshold_buffer: wp.array[wp.float32],
    pair_contacts_count: wp.indexedarray2d(dtype=wp.uint32),
    pair_contacts_start_indices: wp.indexedarray2d(dtype=wp.uint32),
    contact_normal_buffer: wp.array2d[wp.float32],
    contact_force_buffer: wp.array2d[wp.float32],
    # output
    contact_patch_buffer: wp.array[Patch],
    body_to_patch_buffer: wp.array[wp.uint32],
    mass_splitting_scale_buffer: wp.array[wp.float32],
):
    """Group contact points into patches by similar normals and filter out invalid contacts.

    Iterates over every contact point for each body and conveyor belt pair. Contact points
    are discarded (mass splitting scale set to zero) when their normal force is zero or when
    the dot product of the contact normal with the belt surface normal falls below the belt's
    processing threshold. Accepted points are linked into singly-linked patch lists stored in
    ``contact_patch_buffer``; patches whose normals differ by more than ``SAME_NORMAL_THRESHOLD``
    are kept as separate patch nodes connected via ``next_uncorrelated_patch``. Only the head
    entry of each patch has all ``Patch`` struct fields set; non-head entries only use
    ``next_point``.

    Args:
        parallel_body_processing_count: Stride used to distribute bodies across threads.
        body_count: Total number of tracked bodies.
        conveyor_belt_count: Total number of conveyor belt objects.
        conveyor_belt_surface_normal_buffer: Per-belt expected contact normal (world space).
        conveyor_belt_contact_processing_threshold_buffer: Per-belt minimum acceptable
            dot-product between a contact normal and the belt surface normal.
        pair_contacts_count: (N, M) number of contacts per body-belt pair.
        pair_contacts_start_indices: (N, M) start index into the contact buffers (normals, forces)
            per pair.
        contact_normal_buffer: (C, 3) contact normals for all contact points.
        contact_force_buffer: (C, 1) contact normal forces for all contact points.
        contact_patch_buffer: Output - per-contact ``Patch`` linked-list nodes.
        body_to_patch_buffer: Output - index of the first patch head for each body.
        mass_splitting_scale_buffer: Output - set to 0 for filtered-out contacts.
    """
    body_index = wp.tid()

    while body_index < body_count:
        body_to_patch_buffer[body_index] = INVALID_INDEX

        patch_head_index = INVALID_INDEX

        cached_patch_index = INVALID_INDEX

        j = wp.int32(0)
        while j < conveyor_belt_count:
            conveyor_belt_surface_normal = conveyor_belt_surface_normal_buffer[j]
            conveyor_belt_contact_processing_threshold = conveyor_belt_contact_processing_threshold_buffer[j]

            start_index = pair_contacts_start_indices[body_index, j]
            end_index_plus_one = start_index + pair_contacts_count[body_index, j]

            i = start_index
            while i < end_index_plus_one:
                # since the conveyor belt sections are all defined as static geometry in this sample,
                # the expectation is that all contact point normals point from the conveyor belt
                # geometry towards the rigid body, thus no need to test and potentially flip the
                # normal direction.

                normal = wp.vec3(
                    contact_normal_buffer[i, 0],
                    contact_normal_buffer[i, 1],
                    contact_normal_buffer[i, 2],
                )

                force = contact_force_buffer[i, 0]

                acceptance_value = wp.dot(normal, conveyor_belt_surface_normal)

                if (force != 0.0) and (acceptance_value >= conveyor_belt_contact_processing_threshold):
                    if patch_head_index != INVALID_INDEX:
                        # try first to compare with the last chosen patch. Hope is that contact reports send points
                        # ordered by patch already

                        cos_angle = wp.dot(contact_patch_buffer[cached_patch_index].normal, normal)

                        current_patch_index = cached_patch_index

                        if cos_angle <= SAME_NORMAL_THRESHOLD:
                            # the previously used patch has a normal that is different enough
                            # => check if another patch has a matching normal

                            if cached_patch_index != patch_head_index:
                                next_uncorrelated_patch = patch_head_index
                            else:
                                next_uncorrelated_patch = contact_patch_buffer[patch_head_index].next_uncorrelated_patch

                                # note: even if this one is INVALID_INDEX, the subsequent code will work since the
                                #       following will hold:
                                #       current_patch_index = cached_patch_index = patch_head_index

                            correlation_found = wp.bool(False)

                            while next_uncorrelated_patch != INVALID_INDEX:
                                current_patch_index = next_uncorrelated_patch

                                cos_angle = wp.dot(contact_patch_buffer[current_patch_index].normal, normal)

                                if cos_angle > SAME_NORMAL_THRESHOLD:
                                    correlation_found = True
                                    break

                                next_uncorrelated_patch = contact_patch_buffer[
                                    next_uncorrelated_patch
                                ].next_uncorrelated_patch

                            if not correlation_found:
                                # no patch matching this normal has been found
                                # => create a new patch

                                contact_patch_buffer[current_patch_index].next_uncorrelated_patch = i

                                contact_patch_buffer[i].normal = normal
                                contact_patch_buffer[i].point_count = wp.uint32(1)
                                contact_patch_buffer[i].next_point = INVALID_INDEX
                                contact_patch_buffer[i].last_point = i
                                contact_patch_buffer[i].next_uncorrelated_patch = INVALID_INDEX
                                contact_patch_buffer[i].conveyor_belt_index = wp.uint32(j)

                                cached_patch_index = i

                                i += wp.uint32(1)
                                continue

                            cached_patch_index = current_patch_index

                        # found a patch with a very similar normal
                        # => link the point to the patch

                        last_point_index = contact_patch_buffer[current_patch_index].last_point

                        contact_patch_buffer[last_point_index].next_point = i

                        contact_patch_buffer[current_patch_index].last_point = i
                        contact_patch_buffer[current_patch_index].point_count += wp.uint32(1)

                        if contact_patch_buffer[current_patch_index].conveyor_belt_index != wp.uint32(j):
                            contact_patch_buffer[current_patch_index].conveyor_belt_index = INVALID_INDEX

                        contact_patch_buffer[i].next_point = INVALID_INDEX
                    else:
                        body_to_patch_buffer[body_index] = i

                        patch_head_index = i

                        contact_patch_buffer[i].normal = normal
                        contact_patch_buffer[i].point_count = wp.uint32(1)
                        contact_patch_buffer[i].next_point = INVALID_INDEX
                        contact_patch_buffer[i].last_point = i
                        contact_patch_buffer[i].next_uncorrelated_patch = INVALID_INDEX
                        contact_patch_buffer[i].conveyor_belt_index = wp.uint32(j)

                        cached_patch_index = i

                else:
                    mass_splitting_scale_buffer[i] = 0.0

                i += wp.uint32(1)

            j += 1

        body_index += parallel_body_processing_count


ZERO_DISTANCE_DENSITY_VALUE = 1.0


@wp.kernel
def redistribute_contact_force(
    parallel_body_processing_count: int,
    parallel_patch_processing_count: int,
    body_count: int,
    body_to_patch_buffer: wp.array[wp.uint32],
    contact_patch_buffer: wp.array[Patch],
    contact_point_buffer: wp.array2d[wp.float32],
    contact_force_buffer: wp.array2d[wp.float32],
    body_to_world_transform_buffer: wp.array[wp.transform],
    # output
    adjusted_contact_normal_force_buffer: wp.array2d[wp.float32],
    mass_splitting_scale_buffer: wp.array[wp.float32],
):
    """Redistribute contact normal forces within each patch using a point-density heuristic.

    For each patch that spans more than one conveyor belt and contains more than one contact
    point, estimates a 2-D point density in the patch plane using a Gaussian kernel and
    redistributes the total patch normal force so that sparser points receive proportionally
    more force. Patches with a single point, or patches belonging to a single conveyor belt,
    are left unchanged (original per-point normal forces are kept as-is). Also sets the mass
    splitting scale for every contact point in the patch to ``1 / point_count``.

    Args:
        parallel_body_processing_count: Stride used to distribute bodies across threads.
        parallel_patch_processing_count: Stride used to distribute patches across threads.
        body_count: Total number of tracked bodies.
        body_to_patch_buffer: Index of the first patch head for each body.
        contact_patch_buffer: Linked-list patch structure from ``correlate_and_filter_contact_points``.
        contact_point_buffer: (C, 3) world-space contact point positions.
        contact_force_buffer: (C, 1) original contact normal forces.
        body_to_world_transform_buffer: Per-body center-of-mass-to-world transforms (used as reference point).
        adjusted_contact_normal_force_buffer: Output - redistributed normal force for each contact.
        mass_splitting_scale_buffer: Output - mass splitting scale for each contact point.
    """
    body_index, patch_start_offset = wp.tid()

    while body_index < body_count:
        patch_index = body_to_patch_buffer[body_index]

        i = wp.uint32(0)

        next_patch_offset = wp.uint32(patch_start_offset)

        reference_point = body_to_world_transform_buffer[body_index].p

        while True:
            while (i < next_patch_offset) and (patch_index != INVALID_INDEX):
                patch_index = contact_patch_buffer[patch_index].next_uncorrelated_patch

                i += wp.uint32(1)

            if patch_index == INVALID_INDEX:
                break

            point_count = contact_patch_buffer[patch_index].point_count

            mass_splitting_scale = 1.0 / wp.float32(point_count)

            if (point_count == 1) or (contact_patch_buffer[patch_index].conveyor_belt_index != INVALID_INDEX):
                #
                # if a patch has a single contact point only or if a patch spans a single conveyor belt only,
                # then the contact force is not re-distributed and the original normal forces are used as is.
                #

                point_index = patch_index

                while True:
                    mass_splitting_scale_buffer[point_index] = mass_splitting_scale

                    adjusted_contact_normal_force_buffer[point_index, 0] = contact_force_buffer[point_index, 0]

                    point_index = contact_patch_buffer[point_index].next_point

                    if point_index == INVALID_INDEX:
                        break

            else:
                point_index = patch_index

                patch_force_sum = wp.float32(0.0)

                #
                # - estimate the extents of the patch
                # - sum the patch point normal forces
                # - initialize the point density estimates
                #

                patch_normal = contact_patch_buffer[patch_index].normal

                basis_vectors = conveyor_forces_utils.compute_basis_vectors(patch_normal)

                contact_position = wp.vec3(
                    contact_point_buffer[point_index, 0],
                    contact_point_buffer[point_index, 1],
                    contact_point_buffer[point_index, 2],
                )

                delta = contact_position - reference_point

                proj0 = wp.dot(basis_vectors.v0, delta)
                proj1 = wp.dot(basis_vectors.v1, delta)

                min0 = proj0
                max0 = proj0

                min1 = proj1
                max1 = proj1

                # initialize the density at a point with the zero distance density value. The force buffer
                # is used as a temporary cache to store the density estimates.
                adjusted_contact_normal_force_buffer[point_index, 0] = ZERO_DISTANCE_DENSITY_VALUE

                mass_splitting_scale_buffer[point_index] = mass_splitting_scale

                patch_force_sum += contact_force_buffer[point_index, 0]

                while True:
                    point_index = contact_patch_buffer[point_index].next_point

                    if point_index == INVALID_INDEX:
                        break

                    # initialize the density at a point with the zero distance density value. The force buffer
                    # is used as a temporary cache to store the density estimates.
                    adjusted_contact_normal_force_buffer[point_index, 0] = ZERO_DISTANCE_DENSITY_VALUE

                    mass_splitting_scale_buffer[point_index] = mass_splitting_scale

                    patch_force_sum += contact_force_buffer[point_index, 0]

                    contact_position = wp.vec3(
                        contact_point_buffer[point_index, 0],
                        contact_point_buffer[point_index, 1],
                        contact_point_buffer[point_index, 2],
                    )

                    delta = contact_position - reference_point

                    proj0 = wp.dot(basis_vectors.v0, delta)
                    proj1 = wp.dot(basis_vectors.v1, delta)

                    if proj0 < min0:
                        min0 = proj0
                    elif proj0 > max0:
                        max0 = proj0

                    if proj1 < min1:
                        min1 = proj1
                    elif proj1 > max1:
                        max1 = proj1

                #
                # estimate the density distribution of the contact points using a kernel function
                #

                # using half of the average of the patch extents along the two basis vectors
                # as kernel radius
                #
                # kernel: e^-(x^2 / 2) with x^2 = (dist^2 / kernelRadius^2)
                #
                # => a distance that matches the average of the patch extents will give a weight
                #    of around 0.15 (with a weight of 1 for a distance of zero)
                #
                kernel_radius = 0.5 * (0.5 * ((max0 - min0) + (max1 - min1)))
                kernel_radius_sqr = kernel_radius * kernel_radius

                point_index = patch_index

                if kernel_radius > 0.0:
                    point_force_weight_sum = wp.float32(0.0)

                    n = wp.uint32(0)

                    while n < point_count:
                        contact_n_position = wp.vec3(
                            contact_point_buffer[point_index, 0],
                            contact_point_buffer[point_index, 1],
                            contact_point_buffer[point_index, 2],
                        )

                        m = n + wp.uint32(1)

                        point_index_other = contact_patch_buffer[point_index].next_point

                        while m < point_count:
                            # evaluate the kernel function for each point pair and increase the density estimate
                            # of each point accordingly

                            contact_m_position = wp.vec3(
                                contact_point_buffer[point_index_other, 0],
                                contact_point_buffer[point_index_other, 1],
                                contact_point_buffer[point_index_other, 2],
                            )

                            delta = contact_n_position - contact_m_position
                            delta_proj = delta - (wp.dot(delta, patch_normal) * patch_normal)
                            dist_sqr = wp.length_sq(delta_proj)

                            kernel_input = -0.5 * (dist_sqr / kernel_radius_sqr)

                            density_value = wp.exp(kernel_input)

                            # the force buffer is used as a temporary cache to store the density estimate
                            adjusted_contact_normal_force_buffer[point_index, 0] += density_value
                            adjusted_contact_normal_force_buffer[point_index_other, 0] += density_value

                            point_index_other = contact_patch_buffer[point_index_other].next_point

                            m += wp.uint32(1)

                        # the weight for the force re-distribution is set to the inverse of the density
                        # estimate. Note that it will be normalized later. The force buffer
                        # is used as a temporary cache to store the weight.
                        weight = 1.0 / adjusted_contact_normal_force_buffer[point_index, 0]

                        point_force_weight_sum += weight

                        adjusted_contact_normal_force_buffer[point_index, 0] = weight

                        point_index = contact_patch_buffer[point_index].next_point

                        n += wp.uint32(1)

                    #
                    # redistribute the total patch normal force based on the weights of each contact point
                    #

                    if point_force_weight_sum > 0.0:
                        point_force_base_value = patch_force_sum / point_force_weight_sum
                    else:
                        point_force_base_value = 0.0

                    point_index = patch_index

                    while True:
                        # weight was cached temporarily here (see further above)
                        weight = adjusted_contact_normal_force_buffer[point_index, 0]

                        adjusted_force = weight * point_force_base_value

                        adjusted_contact_normal_force_buffer[point_index, 0] = adjusted_force

                        point_index = contact_patch_buffer[point_index].next_point

                        if point_index == INVALID_INDEX:
                            break

                else:
                    adjusted_force = patch_force_sum / wp.float32(point_count)

                    while True:
                        adjusted_contact_normal_force_buffer[point_index, 0] = adjusted_force

                        point_index = contact_patch_buffer[point_index].next_point

                        if point_index == INVALID_INDEX:
                            break

            next_patch_offset += wp.uint32(parallel_patch_processing_count)

        body_index += parallel_body_processing_count


@wp.kernel
def sum_up_force(
    parallel_body_processing_count: int,
    body_count: int,
    body_to_patch_buffer: wp.array[wp.uint32],
    contact_patch_buffer: wp.array[Patch],
    per_point_force_torque_buffer: wp.array[wp.spatial_vector],
    # output
    body_force_buffer: wp.array2d[wp.float32],
    body_torque_buffer: wp.array2d[wp.float32],
):
    """Accumulate per-contact-point force/torque spatial vectors into per-body totals.

    For each body, sums the spatial force/torque contributions from all of its contact points
    across all contact patches and writes the resulting world-space linear force and torque
    into ``body_force_buffer`` and ``body_torque_buffer``.

    Args:
        parallel_body_processing_count: Stride used to distribute bodies across threads.
        body_count: Total number of tracked bodies.
        body_to_patch_buffer: Index of the first patch head for each body.
        contact_patch_buffer: Per-contact ``Patch`` linked-list nodes.
        per_point_force_torque_buffer: (C,) spatial force/torque for each contact point.
        body_force_buffer: Output - (N, 3) accumulated linear force per body.
        body_torque_buffer: Output - (N, 3) accumulated torque per body.
    """
    body_index = wp.tid()

    while body_index < body_count:
        force_torque = wp.spatial_vector(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

        patch_index = body_to_patch_buffer[body_index]

        while patch_index != INVALID_INDEX:
            point_index = patch_index

            while True:
                force_torque += per_point_force_torque_buffer[point_index]

                point_index = contact_patch_buffer[point_index].next_point

                if point_index == INVALID_INDEX:
                    break

            patch_index = contact_patch_buffer[patch_index].next_uncorrelated_patch

        body_force_buffer[body_index, 0] = force_torque[0]
        body_force_buffer[body_index, 1] = force_torque[1]
        body_force_buffer[body_index, 2] = force_torque[2]

        body_torque_buffer[body_index, 0] = force_torque[3]
        body_torque_buffer[body_index, 1] = force_torque[4]
        body_torque_buffer[body_index, 2] = force_torque[5]

        body_index += parallel_body_processing_count
