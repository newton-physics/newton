# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import warp as wp

BLOCK_DIM = wp.constant(64)
OUTPUT_COLUMNS = wp.constant(4)

wp.set_module_options({"block_dim": 64, "enable_backward": False})


@wp.kernel
def fused_mass_matrix(
    articulation_start: wp.array[wp.int32],
    articulation_end: wp.array[wp.int32],
    articulation_count: int,
    articulation_mask: wp.array[wp.bool],
    joint_child: wp.array[wp.int32],
    joint_qd_start: wp.array[wp.int32],
    body_I_s: wp.array[wp.spatial_matrix],
    J: wp.array3d[wp.float32],
    H: wp.array3d[wp.float32],
):
    articulation, dof_row, column_tile, lane = wp.tid()
    column_start = column_tile * OUTPUT_COLUMNS
    joint_start = int(0)
    joint_count = int(0)
    dof_count = int(0)
    enabled = articulation < articulation_count

    if articulation_mask:
        if not articulation_mask[articulation]:
            enabled = False

    if enabled:
        joint_start = articulation_start[articulation]
        joint_end = articulation_end[articulation]
        joint_count = joint_end - joint_start
        dof_count = joint_qd_start[joint_end] - joint_qd_start[joint_start]

    partial = wp.vec4(0.0)

    if enabled and dof_row < dof_count:
        for link in range(lane, joint_count, BLOCK_DIM):
            inertia = body_I_s[joint_child[joint_start + link]]
            row_start = link * 6
            left = wp.spatial_vector(
                J[articulation, row_start, dof_row],
                J[articulation, row_start + 1, dof_row],
                J[articulation, row_start + 2, dof_row],
                J[articulation, row_start + 3, dof_row],
                J[articulation, row_start + 4, dof_row],
                J[articulation, row_start + 5, dof_row],
            )
            weighted = wp.spatial_vector(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

            for spatial_column in range(6):
                component = float(0.0)
                for spatial_row in range(6):
                    component += left[spatial_row] * inertia[spatial_row, spatial_column]
                weighted[spatial_column] = component

            for output_column in range(OUTPUT_COLUMNS):
                dof_column = column_start + output_column
                if dof_column < dof_count and dof_column < J.shape[2]:
                    contribution = float(0.0)
                    for spatial_column in range(6):
                        contribution += (
                            weighted[spatial_column] * J[articulation, row_start + spatial_column, dof_column]
                        )
                    partial[output_column] += contribution

    partial_tile = wp.tile(partial)
    reduced = wp.tile_sum(partial_tile, axis=1)
    output_tile = wp.tile_reshape(reduced, shape=(1, 1, 4))
    wp.tile_store(H, output_tile, offset=(articulation, dof_row, column_start))
