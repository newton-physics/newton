# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""CUDA mass assembly using compacted Jacobian columns or atomic link contributions."""

import warp as wp

Vector6 = wp.types.vector(length=6, dtype=wp.float32)


@wp.kernel(enable_backward=False, module="unique")
def _mass_matrix_compact(
    starts: wp.array[wp.int32],
    ends: wp.array[wp.int32],
    count: int,
    mask: wp.array[wp.bool],
    children: wp.array[wp.int32],
    qstarts: wp.array[wp.int32],
    inertia: wp.array[wp.spatial_matrix],
    jacobian: wp.array3d[wp.float32],
    armature: wp.array[wp.float32],
    accumulator: wp.array3d[wp.float32],
    links: int,
):
    block, lane = wp.tid()
    articulation = block // links
    link = block % links
    if articulation >= count:
        return
    if mask:
        if not mask[articulation]:
            return
    joint_start = starts[articulation]
    joint_end = ends[articulation]
    if link >= joint_end - joint_start:
        return
    active_dofs = qstarts[joint_end] - qstarts[joint_start]
    if active_dofs == 0:
        return
    # Add rotor inertia once per articulation, including structurally zero columns.
    if armature and link == 0 and lane < active_dofs:
        wp.atomic_add(accumulator, articulation, lane, lane, armature[qstarts[joint_start] + lane])
    column_values = Vector6(0.0)
    if lane < active_dofs:
        for spatial_index in range(6):
            column_values[spatial_index] = jacobian[articulation, link * 6 + spatial_index, lane]
    nonzero = (
        column_values[0] != 0.0
        or column_values[1] != 0.0
        or column_values[2] != 0.0
        or column_values[3] != 0.0
        or column_values[4] != 0.0
        or column_values[5] != 0.0
    )
    prefixes = wp.tile_scan_inclusive(wp.tile(int(nonzero)))
    position = wp.untile(prefixes) - int(nonzero)
    active_count = wp.tile_extract(prefixes, 127)
    if active_count == 0:
        return
    active_indices = wp.tile_zeros(shape=(128,), dtype=int, storage="shared")
    wp.tile_scatter_masked(active_indices, position, lane, nonzero)
    values = wp.tile(column_values)
    spatial_inertia = inertia[children[joint_start + link]]
    pair_count = active_count * active_count
    for pair_base in range(0, pair_count, 128):
        pair = pair_base + lane
        safe_pair = wp.min(pair, pair_count - 1)
        row = wp.tile_extract(active_indices, safe_pair // active_count)
        column = wp.tile_extract(active_indices, safe_pair % active_count)
        left_values = Vector6(0.0)
        right_values = Vector6(0.0)
        for spatial_index in range(6):
            left_values[spatial_index] = wp.tile_extract(values, spatial_index, row)
            right_values[spatial_index] = wp.tile_extract(values, spatial_index, column)
        if pair < pair_count:
            subtotal = float(0.0)
            for spatial_row in range(6):
                for spatial_column in range(6):
                    subtotal += (
                        left_values[spatial_row]
                        * spatial_inertia[spatial_row, spatial_column]
                        * right_values[spatial_column]
                    )
            wp.atomic_add(accumulator, articulation, row, column, subtotal)


@wp.kernel(enable_backward=False, module="unique")
def _mass_matrix_atomic(
    starts: wp.array[wp.int32],
    ends: wp.array[wp.int32],
    count: int,
    mask: wp.array[wp.bool],
    children: wp.array[wp.int32],
    qstarts: wp.array[wp.int32],
    inertia: wp.array[wp.spatial_matrix],
    jacobian: wp.array3d[wp.float32],
    armature: wp.array[wp.float32],
    accumulator: wp.array3d[wp.float32],
    dofs: int,
):
    articulation, link, element = wp.tid()
    if articulation >= count:
        return
    if mask:
        if not mask[articulation]:
            return
    joint_start = starts[articulation]
    joint_end = ends[articulation]
    if link >= joint_end - joint_start:
        return
    active_dofs = qstarts[joint_end] - qstarts[joint_start]
    row = element // dofs
    column = element % dofs
    if row >= active_dofs or column >= active_dofs:
        return
    spatial_inertia = inertia[children[joint_start + link]]
    subtotal = float(0.0)
    for spatial_row in range(6):
        for spatial_column in range(6):
            left = jacobian[articulation, link * 6 + spatial_row, row]
            right = jacobian[articulation, link * 6 + spatial_column, column]
            subtotal += left * spatial_inertia[spatial_row, spatial_column] * right
    if armature and link == 0 and row == column:
        subtotal += armature[qstarts[joint_start] + row]
    wp.atomic_add(accumulator, articulation, row, column, subtotal)


def _assemble_mass_matrix(inputs, H):
    """Accumulate link contributions without allocating scratch or changing streams."""
    count = inputs[2]
    J = inputs[7]
    dofs = H.shape[1]
    links = J.shape[1] // 6
    if count == 0 or dofs == 0 or links == 0:
        return
    # Scan/compaction is worthwhile only when enough link blocks are available.
    if tuple(int(part) for part in wp.__version__.split(".")[:2]) >= (1, 18) and dofs <= 128 and count * links >= 128:
        wp.launch_tiled(
            _mass_matrix_compact,
            dim=count * links,
            inputs=[*inputs, H, links],
            block_dim=128,
            device=H.device,
        )
    else:
        wp.launch(
            _mass_matrix_atomic,
            dim=(count, links, dofs * dofs),
            inputs=[*inputs, H, dofs],
            block_dim=128,
            device=H.device,
        )
