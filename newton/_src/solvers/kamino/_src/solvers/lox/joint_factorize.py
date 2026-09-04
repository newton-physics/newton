# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Tiled multi-right-hand-side kernels for structural Delassus assembly."""

from functools import cache

import warp as wp

from ...linalg.factorize._tile_builtins import (
    HAS_NATIVE_TILE_MATMUL_LEFT_TRANSPOSE_UPDATE,
    HAS_TILE_MATMUL_LEFT_TRANSPOSE_UPDATE,
    make_tile_matmul_left_transpose_update_func,
)
from ...linalg.factorize.llt_blocked import get_float32_array_offset_ptr

wp.set_module_options({"enable_backward": False})


@cache
def make_batched_body_solve_kernel(block_size: int, right_hand_side_tile_size: int):
    """Build a tiled Cholesky solve kernel for a packed matrix of right-hand sides."""

    @wp.kernel(enable_backward=False)
    def solve_joint_basis_body_right_hand_sides(
        body_dimensions: wp.array[wp.int32],
        body_matrix_offsets: wp.array[wp.int32],
        joint_dimensions: wp.array[wp.int32],
        response_offsets: wp.array[wp.int32],
        response_leading_dimensions: wp.array[wp.int32],
        factor: wp.array[wp.float32],
        value: wp.array[wp.float32],
        intermediate: wp.array[wp.float32],
    ):
        world, right_hand_side_block, thread = wp.tid()
        thread_count = wp.block_dim()
        column = right_hand_side_block * right_hand_side_tile_size
        right_hand_side_count = joint_dimensions[world]
        if column >= right_hand_side_count:
            return

        dimension = body_dimensions[world]
        matrix_offset = body_matrix_offsets[world]
        response_offset = response_offsets[world]
        leading_dimension = response_leading_dimensions[world]
        factor_pointer = get_float32_array_offset_ptr(factor, matrix_offset)
        value_pointer = get_float32_array_offset_ptr(value, response_offset)
        intermediate_pointer = get_float32_array_offset_ptr(intermediate, response_offset)
        factor_matrix = wp.array(ptr=factor_pointer, shape=(dimension, dimension), dtype=wp.float32)
        value_matrix = wp.array(
            ptr=value_pointer,
            shape=(dimension, leading_dimension),
            dtype=wp.float32,
        )
        intermediate_matrix = wp.array(
            ptr=intermediate_pointer,
            shape=(dimension, leading_dimension),
            dtype=wp.float32,
        )
        padded_dimension = ((dimension + block_size - 1) // block_size) * block_size

        for row in range(0, padded_dimension, block_size):
            right_hand_side = wp.tile_load(
                value_matrix,
                shape=(block_size, right_hand_side_tile_size),
                offset=(row, column),
            )
            diagonal = wp.tile_load(factor_matrix, shape=(block_size, block_size), offset=(row, row))
            if row > 0:
                for inner in range(0, row, block_size):
                    factor_block = wp.tile_load(
                        factor_matrix,
                        shape=(block_size, block_size),
                        offset=(row, inner),
                    )
                    solved_block = wp.tile_load(
                        intermediate_matrix,
                        shape=(block_size, right_hand_side_tile_size),
                        offset=(inner, column),
                    )
                    wp.tile_matmul(factor_block, solved_block, right_hand_side, alpha=-1.0)
            wp.tile_lower_solve_inplace(diagonal, right_hand_side)
            wp.tile_store(intermediate_matrix, right_hand_side, offset=(row, column))

        for row in range(padded_dimension - block_size, -1, -block_size):
            row_end = row + block_size
            right_hand_side = wp.tile_load(
                intermediate_matrix,
                shape=(block_size, right_hand_side_tile_size),
                offset=(row, column),
            )
            diagonal = wp.tile_load(factor_matrix, shape=(block_size, block_size), offset=(row, row))
            if row + block_size > dimension:
                tile_element_count = block_size * block_size
                iteration_count = (tile_element_count + thread_count - 1) // thread_count
                for iteration in range(iteration_count):
                    index = (thread + iteration * thread_count) % tile_element_count
                    local_row = index // block_size
                    local_column = index % block_size
                    diagonal_value = diagonal[local_row, local_column]
                    if row + local_row >= dimension:
                        diagonal_value = wp.where(
                            local_row == local_column,
                            wp.float32(1.0),
                            wp.float32(0.0),
                        )
                    diagonal[local_row, local_column] = diagonal_value
            if row_end < padded_dimension:
                for inner in range(row_end, padded_dimension, block_size):
                    factor_block = wp.tile_load(
                        factor_matrix,
                        shape=(block_size, block_size),
                        offset=(inner, row),
                    )
                    solved_block = wp.tile_load(
                        value_matrix,
                        shape=(block_size, right_hand_side_tile_size),
                        offset=(inner, column),
                    )
                    if wp.static(HAS_TILE_MATMUL_LEFT_TRANSPOSE_UPDATE):
                        wp.tile_matmul_left_transpose_update(
                            right_hand_side,
                            factor_block,
                            solved_block,
                            alpha=-1.0,
                        )
                    elif wp.static(HAS_NATIVE_TILE_MATMUL_LEFT_TRANSPOSE_UPDATE and right_hand_side_tile_size == 1):
                        wp.static(make_tile_matmul_left_transpose_update_func(block_size))(
                            right_hand_side, factor_block, solved_block, -1.0
                        )
                    else:
                        wp.tile_matmul(
                            wp.tile_transpose(factor_block),
                            solved_block,
                            right_hand_side,
                            alpha=-1.0,
                        )
            wp.tile_upper_solve_inplace(wp.tile_transpose(diagonal), right_hand_side)
            wp.tile_store(value_matrix, right_hand_side, offset=(row, column))

    return solve_joint_basis_body_right_hand_sides
