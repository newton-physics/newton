# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Deterministic matrix-free sparse kernels for Kamino contact APGD."""

from __future__ import annotations

import warp as wp

from ...core.types import vec6f

wp.set_module_options({"enable_backward": False})

float32 = wp.float32
int32 = wp.int32
mat33f = wp.mat33f
vec3f = wp.vec3f
vec2i = wp.vec2i


@wp.kernel
def build_sparse_body_adjacency(
    problem_dim: wp.array[int32],
    problem_nc: wp.array[int32],
    problem_cio: wp.array[int32],
    problem_ccgo: wp.array[int32],
    contact_indices: wp.array[int32],
    contact_nzb_offsets: wp.array[int32],
    jacobian_num_nzb: wp.array[int32],
    jacobian_nzb_start: wp.array[int32],
    jacobian_nzb_coords: wp.array2d[int32],
    body_offset: wp.array[int32],
    max_contacts_per_world: int32,
    max_bodies_per_world: int32,
    body_contact_count: wp.array[int32],
    body_block_count: wp.array[int32],
    body_contact_offsets: wp.array[int32],
    body_contact_slots: wp.array[vec2i],
    body_block_offsets: wp.array[int32],
    body_block_slots: wp.array[int32],
    body_block_scratch: wp.array[int32],
):
    """Build stable per-body CSR traversals for the current sparse topology."""
    wid = wp.tid()
    body_count = body_offset[wid + int32(1)] - body_offset[wid]
    count_base = wid * max_bodies_per_world
    offset_base = wid * (max_bodies_per_world + int32(1))
    contact_slot_base = int32(2) * wid * max_contacts_per_world
    matrix_begin = jacobian_nzb_start[wid]
    matrix_end = matrix_begin + jacobian_num_nzb[wid]

    for local_body in range(body_count):
        body_contact_count[count_base + local_body] = int32(0)
        body_block_count[count_base + local_body] = int32(0)

    contact_count = problem_nc[wid]
    contact_index_offset = problem_cio[wid]
    contact_group_offset = problem_ccgo[wid]
    dimension = problem_dim[wid]
    contact_end = contact_group_offset + int32(3) * contact_count
    for cid in range(contact_count):
        contact_id = contact_indices[contact_index_offset + cid]
        if contact_id < int32(0):
            continue
        block_offset = contact_nzb_offsets[contact_id]
        expected_row = contact_group_offset + int32(3) * cid
        for side in range(2):
            side_offset = block_offset + int32(3) * side
            if (
                side_offset >= matrix_begin
                and side_offset < matrix_end
                and jacobian_nzb_coords[side_offset, 0] == expected_row
            ):
                local_body = jacobian_nzb_coords[side_offset, 1] / int32(6)
                if local_body >= int32(0) and local_body < body_count:
                    body_contact_count[count_base + local_body] += int32(1)

    for local_block in range(jacobian_num_nzb[wid]):
        block_id = matrix_begin + local_block
        row = jacobian_nzb_coords[block_id, 0]
        local_body = jacobian_nzb_coords[block_id, 1] / int32(6)
        if (
            row < dimension
            and (row < contact_group_offset or row >= contact_end)
            and local_body >= int32(0)
            and local_body < body_count
        ):
            body_block_count[count_base + local_body] += int32(1)

    contact_cursor = contact_slot_base
    block_cursor = matrix_begin
    for local_body in range(body_count):
        count_index = count_base + local_body
        body_contact_offsets[offset_base + local_body] = contact_cursor
        body_block_offsets[offset_base + local_body] = block_cursor
        contact_cursor += body_contact_count[count_index]
        block_cursor += body_block_count[count_index]
        body_contact_count[count_index] = body_contact_offsets[offset_base + local_body]
        body_block_count[count_index] = body_block_offsets[offset_base + local_body]
    body_contact_offsets[offset_base + body_count] = contact_cursor
    body_block_offsets[offset_base + body_count] = block_cursor

    # Repeat the logical contact traversal so each body's entries are stable
    # even though the sparse contact blocks were allocated atomically.
    for cid in range(contact_count):
        contact_id = contact_indices[contact_index_offset + cid]
        if contact_id < int32(0):
            continue
        block_offset = contact_nzb_offsets[contact_id]
        expected_row = contact_group_offset + int32(3) * cid
        for side in range(2):
            side_offset = block_offset + int32(3) * side
            if (
                side_offset >= matrix_begin
                and side_offset < matrix_end
                and jacobian_nzb_coords[side_offset, 0] == expected_row
            ):
                local_body = jacobian_nzb_coords[side_offset, 1] / int32(6)
                if local_body >= int32(0) and local_body < body_count:
                    count_index = count_base + local_body
                    slot = body_contact_count[count_index]
                    body_contact_slots[slot] = vec2i(cid, side_offset)
                    body_contact_count[count_index] = slot + int32(1)

    # First group sparse blocks by body; logical row ordering is restored below.
    for local_block in range(jacobian_num_nzb[wid]):
        block_id = matrix_begin + local_block
        row = jacobian_nzb_coords[block_id, 0]
        local_body = jacobian_nzb_coords[block_id, 1] / int32(6)
        if (
            row < dimension
            and (row < contact_group_offset or row >= contact_end)
            and local_body >= int32(0)
            and local_body < body_count
        ):
            count_index = count_base + local_body
            slot = body_block_count[count_index]
            body_block_slots[slot] = block_id
            body_block_count[count_index] = slot + int32(1)

    # Dynamic limit/contact blocks receive storage through atomic allocation.
    # Restore logical row order before reducing the non-contact contribution so
    # floating-point summation does not inherit that allocation order.
    for local_body in range(body_count):
        slot_begin = body_block_offsets[offset_base + local_body]
        slot_end = body_block_offsets[offset_base + local_body + int32(1)]
        width = int32(1)
        while width < slot_end - slot_begin:
            merge_begin = slot_begin
            while merge_begin < slot_end:
                middle = wp.min(merge_begin + width, slot_end)
                merge_end = wp.min(middle + width, slot_end)
                left = merge_begin
                right = middle
                output = merge_begin
                while left < middle and right < merge_end:
                    left_block = body_block_slots[left]
                    right_block = body_block_slots[right]
                    left_row = jacobian_nzb_coords[left_block, 0]
                    right_row = jacobian_nzb_coords[right_block, 0]
                    take_left = left_row < right_row or (left_row == right_row and left_block <= right_block)
                    if take_left:
                        body_block_scratch[output] = left_block
                        left += int32(1)
                    else:
                        body_block_scratch[output] = right_block
                        right += int32(1)
                    output += int32(1)
                while left < middle:
                    body_block_scratch[output] = body_block_slots[left]
                    left += int32(1)
                    output += int32(1)
                while right < merge_end:
                    body_block_scratch[output] = body_block_slots[right]
                    right += int32(1)
                    output += int32(1)
                for slot in range(merge_begin, merge_end):
                    body_block_slots[slot] = body_block_scratch[slot]
                merge_begin += int32(2) * width
            width *= int32(2)


@wp.kernel
def sparse_contact_transpose_mass(
    problem_ccgo: wp.array[int32],
    problem_vio: wp.array[int32],
    contact_row_offset: wp.array[int32],
    body_contact_offsets: wp.array[int32],
    body_contact_slots: wp.array[vec2i],
    jacobian_nzb_coords: wp.array2d[int32],
    jacobian_nzb_values: wp.array[vec6f],
    jacobian_col_start: wp.array[int32],
    body_offset: wp.array[int32],
    body_inv_mass: wp.array[float32],
    body_inv_inertia: wp.array[mat33f],
    preconditioner: wp.array[float32],
    max_bodies_per_world: int32,
    x: wp.array[float32],
    world_mask: wp.array[bool],
    body_space: wp.array[float32],
):
    """Form ``M^-1 J_C^T P_C x`` with a fixed contact traversal."""
    wid, local_body = wp.tid()
    body_begin = body_offset[wid]
    body_count = body_offset[wid + 1] - body_begin
    if local_body >= body_count:
        return

    linear = vec3f(0.0)
    angular = vec3f(0.0)
    if world_mask[wid]:
        contact_group_offset = problem_ccgo[wid]
        vector_offset = problem_vio[wid]
        compact_offset = contact_row_offset[wid]
        body_offset_index = wid * (max_bodies_per_world + int32(1)) + local_body
        slot_begin = body_contact_offsets[body_offset_index]
        slot_end = body_contact_offsets[body_offset_index + int32(1)]

        for slot in range(slot_begin, slot_end):
            contact_slot = body_contact_slots[slot]
            cid = contact_slot[0]
            block_offset = contact_slot[1]
            row_begin = contact_group_offset + int32(3) * cid
            compact_begin = compact_offset + int32(3) * cid
            for component in range(3):
                row = row_begin + component
                scale = preconditioner[vector_offset + row] * x[compact_begin + component]
                block_id = block_offset + component
                if jacobian_nzb_coords[block_id, 0] == row:
                    block = jacobian_nzb_values[block_id]
                    linear += scale * vec3f(block[0], block[1], block[2])
                    angular += scale * vec3f(block[3], block[4], block[5])

    global_body = body_begin + local_body
    linear = body_inv_mass[global_body] * linear
    angular = body_inv_inertia[global_body] @ angular
    body_dof = jacobian_col_start[wid] + int32(6) * local_body
    for component in range(3):
        body_space[body_dof + component] = linear[component]
        body_space[body_dof + int32(3) + component] = angular[component]


@wp.kernel
def sparse_noncontact_transpose_mass(
    problem_dim: wp.array[int32],
    problem_nc: wp.array[int32],
    problem_ccgo: wp.array[int32],
    problem_vio: wp.array[int32],
    body_block_offsets: wp.array[int32],
    body_block_slots: wp.array[int32],
    jacobian_nzb_coords: wp.array2d[int32],
    jacobian_nzb_values: wp.array[vec6f],
    jacobian_col_start: wp.array[int32],
    body_offset: wp.array[int32],
    body_inv_mass: wp.array[float32],
    body_inv_inertia: wp.array[mat33f],
    preconditioner: wp.array[float32],
    max_bodies_per_world: int32,
    full_solution: wp.array[float32],
    world_mask: wp.array[bool],
    body_space: wp.array[float32],
):
    """Form ``M^-1 J_notC^T P_notC lambda_notC`` in row order."""
    wid, local_body = wp.tid()
    body_begin = body_offset[wid]
    body_count = body_offset[wid + 1] - body_begin
    if local_body >= body_count:
        return

    linear = vec3f(0.0)
    angular = vec3f(0.0)
    if world_mask[wid]:
        dimension = problem_dim[wid]
        contact_begin = problem_ccgo[wid]
        contact_end = contact_begin + int32(3) * problem_nc[wid]
        vector_offset = problem_vio[wid]
        body_offset_index = wid * (max_bodies_per_world + int32(1)) + local_body
        slot_begin = body_block_offsets[body_offset_index]
        slot_end = body_block_offsets[body_offset_index + int32(1)]

        for slot in range(slot_begin, slot_end):
            block_id = body_block_slots[slot]
            row = jacobian_nzb_coords[block_id, 0]
            if row < dimension and (row < contact_begin or row >= contact_end):
                scale = preconditioner[vector_offset + row] * full_solution[vector_offset + row]
                block = jacobian_nzb_values[block_id]
                linear += scale * vec3f(block[0], block[1], block[2])
                angular += scale * vec3f(block[3], block[4], block[5])

    global_body = body_begin + local_body
    linear = body_inv_mass[global_body] * linear
    angular = body_inv_inertia[global_body] @ angular
    body_dof = jacobian_col_start[wid] + int32(6) * local_body
    for component in range(3):
        body_space[body_dof + component] = linear[component]
        body_space[body_dof + int32(3) + component] = angular[component]


@wp.kernel
def sparse_contact_matvec(
    problem_nc: wp.array[int32],
    problem_cio: wp.array[int32],
    problem_ccgo: wp.array[int32],
    problem_vio: wp.array[int32],
    contact_row_offset: wp.array[int32],
    contact_indices: wp.array[int32],
    contact_nzb_offsets: wp.array[int32],
    jacobian_num_nzb: wp.array[int32],
    jacobian_nzb_start: wp.array[int32],
    jacobian_nzb_coords: wp.array2d[int32],
    jacobian_nzb_values: wp.array[vec6f],
    jacobian_row_start: wp.array[int32],
    jacobian_col_start: wp.array[int32],
    preconditioner: wp.array[float32],
    regularization: wp.array[float32],
    represented_compliance: wp.array[float32],
    body_space: wp.array[float32],
    x: wp.array[float32],
    world_mask: wp.array[bool],
    y: wp.array[float32],
):
    """Finish ``(P J M^-1 J^T P + R + E_hat)_CC x``."""
    wid, local_row = wp.tid()
    contact_row_count = int32(3) * problem_nc[wid]
    if not world_mask[wid] or local_row >= contact_row_count:
        return

    cid = local_row / int32(3)
    component = local_row - int32(3) * cid
    contact_id = contact_indices[problem_cio[wid] + cid]
    full_row = problem_ccgo[wid] + local_row
    vector_index = problem_vio[wid] + full_row
    compact_index = contact_row_offset[wid] + local_row
    value = float32(0.0)

    if contact_id >= int32(0):
        block_offset = contact_nzb_offsets[contact_id]
        matrix_begin = jacobian_nzb_start[wid]
        matrix_end = matrix_begin + jacobian_num_nzb[wid]
        block_id = block_offset + component
        if block_id >= matrix_begin and block_id < matrix_end and jacobian_nzb_coords[block_id, 0] == full_row:
            block = jacobian_nzb_values[block_id]
            body_dof = jacobian_col_start[wid] + jacobian_nzb_coords[block_id, 1]
            for dof in range(6):
                value += block[dof] * body_space[body_dof + dof]

        block_id = block_offset + int32(3) + component
        if block_id >= matrix_begin and block_id < matrix_end and jacobian_nzb_coords[block_id, 0] == full_row:
            block = jacobian_nzb_values[block_id]
            body_dof = jacobian_col_start[wid] + jacobian_nzb_coords[block_id, 1]
            for dof in range(6):
                value += block[dof] * body_space[body_dof + dof]

    value *= preconditioner[vector_index]
    diagonal = regularization[jacobian_row_start[wid] + full_row] + represented_compliance[vector_index]
    y[compact_index] = value + diagonal * x[compact_index]


@wp.kernel
def build_sparse_contact_rhs(
    problem_nc: wp.array[int32],
    problem_cio: wp.array[int32],
    problem_ccgo: wp.array[int32],
    problem_vio: wp.array[int32],
    contact_row_offset: wp.array[int32],
    contact_indices: wp.array[int32],
    contact_nzb_offsets: wp.array[int32],
    jacobian_num_nzb: wp.array[int32],
    jacobian_nzb_start: wp.array[int32],
    jacobian_nzb_coords: wp.array2d[int32],
    jacobian_nzb_values: wp.array[vec6f],
    jacobian_col_start: wp.array[int32],
    preconditioner: wp.array[float32],
    free_velocity: wp.array[float32],
    body_space: wp.array[float32],
    world_mask: wp.array[bool],
    rhs: wp.array[float32],
):
    """Finish ``b_C = -(v_f,C + P_C J_C body_notC)``."""
    wid, local_row = wp.tid()
    contact_row_count = int32(3) * problem_nc[wid]
    if not world_mask[wid] or local_row >= contact_row_count:
        return

    cid = local_row / int32(3)
    component = local_row - int32(3) * cid
    contact_id = contact_indices[problem_cio[wid] + cid]
    full_row = problem_ccgo[wid] + local_row
    vector_index = problem_vio[wid] + full_row
    value = float32(0.0)

    if contact_id >= int32(0):
        block_offset = contact_nzb_offsets[contact_id]
        matrix_begin = jacobian_nzb_start[wid]
        matrix_end = matrix_begin + jacobian_num_nzb[wid]
        block_id = block_offset + component
        if block_id >= matrix_begin and block_id < matrix_end and jacobian_nzb_coords[block_id, 0] == full_row:
            block = jacobian_nzb_values[block_id]
            body_dof = jacobian_col_start[wid] + jacobian_nzb_coords[block_id, 1]
            for dof in range(6):
                value += block[dof] * body_space[body_dof + dof]

        block_id = block_offset + int32(3) + component
        if block_id >= matrix_begin and block_id < matrix_end and jacobian_nzb_coords[block_id, 0] == full_row:
            block = jacobian_nzb_values[block_id]
            body_dof = jacobian_col_start[wid] + jacobian_nzb_coords[block_id, 1]
            for dof in range(6):
                value += block[dof] * body_space[body_dof + dof]

    rhs[contact_row_offset[wid] + local_row] = -(free_velocity[vector_index] + preconditioner[vector_index] * value)
