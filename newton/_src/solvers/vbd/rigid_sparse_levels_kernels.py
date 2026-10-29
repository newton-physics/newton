# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Cooperative CUDA block Cholesky using independent elimination levels."""

import warp as wp

from .rigid_sparse_articulation_kernels import (
    _block_sync,
    _cholesky66_serial,
    _scalar_block_get,
    _scalar_block_set,
    _scalar_vec_load,
    _scalar_vec_set,
    _solve_lower66,
    _solve_upper66_from_lower,
    mat66f,
    vec6f,
)

wp.set_module_options({"enable_backward": False})


@wp.func
def _load_block66_scalar(values: wp.array[float], slot: int) -> mat66f:
    block = mat66f(0.0)
    for i in range(6):
        for j in range(6):
            block[i, j] = _scalar_block_get(values, slot, i, j)
    return block


@wp.func
def _cholesky66_scalar(values: wp.array[float], slot: int):
    # Keep the dependent scalar operations in thread-local storage. Publishing
    # intermediate pivots to global memory adds latency without sharing work.
    factor = _cholesky66_serial(_load_block66_scalar(values, slot))
    for i in range(6):
        for j in range(6):
            _scalar_block_set(values, slot, i, j, factor[i, j])


@wp.func
def _right_solve_lower_transpose66_scalar_row(values: wp.array[float], slot: int, diag_slot: int, row: int):
    factor = _load_block66_scalar(values, diag_slot)
    result = vec6f(0.0)
    for col in range(6):
        value = _scalar_block_get(values, slot, row, col)
        for k in range(6):
            if k < col:
                value = value - result[k] * factor[col, k]
        result[col] = value / factor[col, col]
    for col in range(6):
        _scalar_block_set(values, slot, row, col, result[col])


@wp.kernel
def solve_articulation_sparse_levels(
    articulation_body_offsets: wp.array[wp.int32],
    block_row_offsets: wp.array[wp.int32],
    block_cols: wp.array[wp.int32],
    block_col_offsets: wp.array[wp.int32],
    block_col_rows: wp.array[wp.int32],
    block_col_slots: wp.array[wp.int32],
    diag_slots: wp.array[wp.int32],
    articulation_level_offsets: wp.array[wp.int32],
    level_body_offsets: wp.array[wp.int32],
    level_column_offsets: wp.array[wp.int32],
    level_column_slots: wp.array[wp.int32],
    level_column_diag_slots: wp.array[wp.int32],
    level_update_offsets: wp.array[wp.int32],
    level_update_dst_slots: wp.array[wp.int32],
    level_update_term_offsets: wp.array[wp.int32],
    level_update_left_slots: wp.array[wp.int32],
    level_update_right_slots: wp.array[wp.int32],
    values: wp.array[float],
    rhs: wp.array[float],
    delta: wp.array[float],
):
    """Cooperative factor and triangular solves over independent pivot levels.

    One CTA owns an articulation. Every destination block entry has a single
    writer per level, including shared Schur complements at loop closures.
    """
    tid = wp.tid()
    threads = wp.block_dim()
    articulation = tid // threads
    lane = tid - articulation * threads
    body_start = articulation_body_offsets[articulation]
    first = articulation_level_offsets[articulation]
    last = articulation_level_offsets[articulation + 1]

    for level in range(first, last):
        begin = level_body_offsets[level]
        end = level_body_offsets[level + 1]
        for row in range(begin + lane, end, threads):
            _cholesky66_scalar(values, diag_slots[row])
        _block_sync()

        column_begin = level_column_offsets[level]
        column_end = level_column_offsets[level + 1]
        for task in range(column_begin * 6 + lane, column_end * 6, threads):
            entry = task // 6
            component = task - entry * 6
            _right_solve_lower_transpose66_scalar_row(
                values, level_column_slots[entry], level_column_diag_slots[entry], component
            )
        _block_sync()

        update_begin = level_update_offsets[level]
        update_end = level_update_offsets[level + 1]
        for task in range(update_begin * 36 + lane, update_end * 36, threads):
            update = task // 36
            component = task - update * 36
            i = component // 6
            j = component - i * 6
            dst = level_update_dst_slots[update]
            value = _scalar_block_get(values, dst, i, j)
            for term in range(level_update_term_offsets[update], level_update_term_offsets[update + 1]):
                left = level_update_left_slots[term]
                right = level_update_right_slots[term]
                accum = float(0.0)
                for k in range(6):
                    accum = accum + _scalar_block_get(values, left, i, k) * _scalar_block_get(values, right, j, k)
                value = value - accum
            _scalar_block_set(values, dst, i, j, value)
        _block_sync()

    for level in range(first, last):
        begin = level_body_offsets[level]
        end = level_body_offsets[level + 1]
        for row in range(begin + lane, end, threads):
            vector_value = _scalar_vec_load(rhs, row)
            for slot in range(block_row_offsets[row], block_row_offsets[row + 1]):
                col = body_start + block_cols[slot]
                if col < row:
                    vector_value = vector_value - _load_block66_scalar(values, slot) * _scalar_vec_load(delta, col)
            result = _solve_lower66(_load_block66_scalar(values, diag_slots[row]), vector_value)
            for i in range(6):
                _scalar_vec_set(delta, row, i, result[i])
        _block_sync()

    for reverse_level in range(last - first):
        level = last - reverse_level - 1
        begin = level_body_offsets[level]
        end = level_body_offsets[level + 1]
        for row in range(begin + lane, end, threads):
            vector_value = _scalar_vec_load(delta, row)
            for entry in range(block_col_offsets[row], block_col_offsets[row + 1]):
                col = body_start + block_col_rows[entry]
                slot = block_col_slots[entry]
                vector_value = vector_value - wp.transpose(_load_block66_scalar(values, slot)) * _scalar_vec_load(
                    delta, col
                )
            result = _solve_upper66_from_lower(_load_block66_scalar(values, diag_slots[row]), vector_value)
            for i in range(6):
                _scalar_vec_set(delta, row, i, result[i])
        _block_sync()
