# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Implicit complete-pattern LDL storage for changing dense active dimensions."""

from functools import cache

import warp as wp

from ..core import dense_matrix_layout, validate_dense_matrix_shape
from ._array_ptr import get_float32_array_offset_ptr

###
# Module interface
###

__all__ = ["DenseLDLStorage", "rhs_array_type"]


###
# Module configs
###

wp.set_module_options({"enable_backward": False})


###
# Functions
###


def rhs_array_type(per_world_rows: bool):
    """Return the Warp array type of a flat RHS buffer, or of one RHS row per world."""
    return wp.array2d[wp.float32] if per_world_rows else wp.array[wp.float32]


@wp.func
def _row(index: int, padded: int, reverse: bool):
    """Map a scalar row through the optional reversal of padded six-row groups."""
    return (padded // 6 - 1 - index // 6) * 6 + index % 6 if reverse else index


###
# Kernels
###


@wp.kernel
def _factor(
    dims: wp.array[wp.int32],
    a_storage: wp.array2d[wp.float32],
    lower_storage: wp.array2d[wp.float32],
    diagonal_storage: wp.array2d[wp.float32],
    failed: wp.array[wp.int32],
    reverse: bool,
    retry: bool,
    shift: float,
):
    """Factor each active world into unit-lower L and diagonal D, with optional shifted retry."""
    world = wp.tid()
    if retry and failed[world] == 0:
        return
    n = dims[world]
    padded = (n + 5) // 6 * 6
    av = wp.array(ptr=get_float32_array_offset_ptr(a_storage[world], 0), shape=(n, n), dtype=wp.float32)
    lv = wp.array(ptr=get_float32_array_offset_ptr(lower_storage[world], 0), shape=(padded, padded), dtype=wp.float32)
    dv = wp.array(ptr=get_float32_array_offset_ptr(diagonal_storage[world], 0), shape=(1, padded), dtype=wp.float32)
    for j in range(0, padded, 6):
        original_j = _row(j, padded, reverse)
        schur = wp.tile_load(av, shape=(6, 6), offset=(original_j, original_j))
        regularization = wp.tile_zeros(shape=(6, 6), dtype=wp.float32)
        # Identity padding stays within the original final six-row group,
        # which becomes the first group under complete-graph block RCM.
        for r in range(6):
            for c in range(6):
                if original_j + r >= n or original_j + c >= n:
                    schur[r, c] = wp.float32(1.0) if r == c else wp.float32(0.0)
            if retry and original_j + r < n:
                regularization[r, r] = shift
        schur += regularization
        for k in range(0, j, 6):
            previous = wp.tile_load(lv, shape=(6, 6), offset=(j, k))
            previous_d = wp.tile_load(dv, shape=(1, 6), offset=(0, k))
            weighted = wp.tile_map(wp.mul, previous, wp.tile_broadcast(previous_d, shape=(6, 6)))
            wp.tile_matmul(weighted, wp.tile_transpose(previous), schur, alpha=-1.0)
        block = wp.tile_zeros(shape=(6, 6), dtype=wp.float32)
        d = wp.tile_zeros(shape=(1, 6), dtype=wp.float32)
        for c in range(6):
            pivot = schur[c, c]
            for k in range(c):
                pivot -= block[c, k] * d[0, k] * block[c, k]
            d[0, c] = pivot
            if not wp.isfinite(pivot) or pivot <= 0.0:
                failed[world] = 1
            block[c, c] = wp.float32(1.0)
            for r in range(c + 1, 6):
                value = schur[r, c]
                for k in range(c):
                    value -= block[r, k] * d[0, k] * block[c, k]
                block[r, c] = value / pivot
        wp.tile_store(lv, block, offset=(j, j))
        wp.tile_store(dv, d, offset=(0, j))
        for i in range(j + 6, padded, 6):
            original_i = _row(i, padded, reverse)
            off = wp.tile_load(av, shape=(6, 6), offset=(original_i, original_j))
            for k in range(0, j, 6):
                left = wp.tile_load(lv, shape=(6, 6), offset=(i, k))
                right = wp.tile_load(lv, shape=(6, 6), offset=(j, k))
                previous_d = wp.tile_load(dv, shape=(1, 6), offset=(0, k))
                weighted = wp.tile_map(wp.mul, left, wp.tile_broadcast(previous_d, shape=(6, 6)))
                wp.tile_matmul(weighted, wp.tile_transpose(right), off, alpha=-1.0)
            transposed = wp.tile_transpose(off)
            wp.tile_lower_solve_inplace(block, transposed)
            result = wp.tile_map(wp.div, wp.tile_transpose(transposed), wp.tile_broadcast(d, shape=(6, 6)))
            wp.tile_store(lv, result, offset=(i, j))


@cache
def _make_triangular_kernel(per_world_rows: bool):
    """Specialize triangular substitution for flat or per-world-row RHS storage."""
    rhs = rhs_array_type(per_world_rows)

    @wp.kernel(module="unique")
    def triangular(
        dims: wp.array[wp.int32],
        active: wp.array[wp.int32],
        offsets: wp.array[wp.int32],
        strides: wp.array[wp.int32],
        columns: wp.array[wp.int32],
        lower_storage: wp.array2d[wp.float32],
        b: rhs,
        x: rhs,
        reverse: bool,
        forward: bool,
    ):
        """Solve L or its transpose for active columns stored in each world's strided RHS layout."""
        world, column = wp.tid()
        if column >= columns[world]:
            return
        padded = (dims[world] + 5) // 6 * 6
        n = active[world]
        if wp.static(per_world_rows):
            b_ptr = get_float32_array_offset_ptr(b[world], offsets[world])
            x_ptr = get_float32_array_offset_ptr(x[world], offsets[world])
        else:
            b_ptr = get_float32_array_offset_ptr(b, offsets[world])
            x_ptr = get_float32_array_offset_ptr(x, offsets[world])
        bv = wp.array(ptr=b_ptr, shape=(n, strides[world]), dtype=wp.float32)
        xv = wp.array(ptr=x_ptr, shape=(n, strides[world]), dtype=wp.float32)
        lv = wp.array(
            ptr=get_float32_array_offset_ptr(lower_storage[world], 0), shape=(padded, padded), dtype=wp.float32
        )
        for step in range(padded // 6):
            i = step * 6 if forward else padded - 6 - step * 6
            original_i = _row(i, padded, reverse)
            if original_i >= n:
                continue
            value = wp.tile_load(bv, shape=(6, 1), offset=(original_i, column))
            for previous in range(step):
                j = previous * 6 if forward else padded - 6 - previous * 6
                original_j = _row(j, padded, reverse)
                solved = wp.tile_load(xv, shape=(6, 1), offset=(original_j, column))
                if forward:
                    block = wp.tile_load(lv, shape=(6, 6), offset=(i, j))
                    wp.tile_matmul(block, solved, value, alpha=-1.0)
                else:
                    block = wp.tile_load(lv, shape=(6, 6), offset=(j, i))
                    wp.tile_matmul(wp.tile_transpose(block), solved, value, alpha=-1.0)
            block = wp.tile_load(lv, shape=(6, 6), offset=(i, i))
            if original_i + 6 > n:
                for r in range(6):
                    for c in range(6):
                        if original_i + r >= n or original_i + c >= n:
                            block[r, c] = wp.float32(1.0) if r == c else wp.float32(0.0)
            if forward:
                wp.tile_lower_solve_inplace(block, value)
            else:
                wp.tile_upper_solve_inplace(wp.tile_transpose(block), value)
            wp.tile_store(xv, value, offset=(original_i, column))

    return triangular


@cache
def _make_diagonal_kernel(per_world_rows: bool):
    """Specialize diagonal scaling for flat or per-world-row RHS storage."""
    rhs = rhs_array_type(per_world_rows)

    @wp.kernel(module="unique")
    def diagonal_scale(
        dims: wp.array[wp.int32],
        active: wp.array[wp.int32],
        offsets: wp.array[wp.int32],
        strides: wp.array[wp.int32],
        columns: wp.array[wp.int32],
        diagonal_storage: wp.array2d[wp.float32],
        b: rhs,
        x: rhs,
        reverse: bool,
        inverse_sqrt: bool,
    ):
        """Scale active RHS entries by inverse D or inverse square-root D in original row order."""
        world, row, column = wp.tid()
        if row >= active[world] or column >= columns[world]:
            return
        padded = (dims[world] + 5) // 6 * 6
        d = diagonal_storage[world, _row(row, padded, reverse)]
        if inverse_sqrt:
            d = wp.sqrt(d)
        target = offsets[world] + row * strides[world] + column
        if wp.static(per_world_rows):
            x[world, target] = b[world, target] / d
        else:
            x[target] = b[target] / d

    return diagonal_scale


@wp.kernel
def _reconstruct(
    dims: wp.array[wp.int32],
    lower_storage: wp.array2d[wp.float32],
    diagonal_storage: wp.array2d[wp.float32],
    a_storage: wp.array2d[wp.float32],
    reverse: bool,
):
    """Write L D L transpose into each world's matrix row, excluding padding."""
    world, row, column = wp.tid()
    n = dims[world]
    if row >= n or column >= n:
        return
    padded = (n + 5) // 6 * 6
    i, j = _row(row, padded, reverse), _row(column, padded, reverse)
    lower = lower_storage[world]
    diagonal = diagonal_storage[world]
    value = wp.float32(0.0)
    for k in range(wp.min(i, j) + 1):
        value += lower[i * padded + k] * diagonal[k] * lower[j * padded + k]
    a_storage[world, row * n + column] = value


###
# Interfaces
###


class DenseLDLStorage:
    """Represent a complete block graph implicitly, with quadratic storage only."""

    def __init__(self, operator, ordering, device):
        """Allocate one padded factor row per world at its maximum capacity and cache its block ordering."""
        self.info = operator.info
        self.device = device
        self.reverse = ordering == "rcm"
        capacities = [(int(n) + 5) // 6 * 6 for n in self.info.maxdim.numpy()]
        self.worlds = len(capacities)
        self.maxdim = max(capacities, default=0)
        lower_shape = dense_matrix_layout([n * n for n in capacities])
        validate_dense_matrix_shape(lower_shape, wp.float32)
        self.lower = wp.zeros(lower_shape, dtype=wp.float32, device=device)
        self.diagonal = wp.zeros(dense_matrix_layout(capacities), dtype=wp.float32, device=device)
        self.failed = wp.zeros(self.worlds, dtype=wp.int32, device=device)

    ###
    # Operations
    ###

    def factorize(self, a, shift):
        """Factor active matrices, retrying only failed worlds with the requested diagonal shift."""
        self.failed.zero_()
        for retry in (False, True) if shift > 0.0 else (False,):
            wp.launch_tiled(
                _factor,
                dim=self.worlds,
                inputs=[
                    self.info.dim,
                    a,
                    self.lower,
                    self.diagonal,
                    self.failed,
                    self.reverse,
                    retry,
                    shift,
                ],
                block_dim=32,
                device=self.device,
            )

    def triangular(self, b, x, *, offsets, strides, columns, max_columns, forward, active_dim):
        """Apply forward or backward substitution to packed RHS columns; b and x may alias."""
        wp.launch_tiled(
            _make_triangular_kernel(b.ndim == 2),
            dim=(self.worlds, max_columns),
            inputs=[
                self.info.dim,
                self.info.dim if active_dim is None else active_dim,
                offsets,
                strides,
                columns,
                self.lower,
                b,
                x,
                self.reverse,
                forward,
            ],
            block_dim=32,
            device=self.device,
        )

    def scale(self, b, x, *, offsets, strides, columns, max_columns, inverse_sqrt, active_dim):
        """Apply inverse D or inverse square-root D to active packed RHS columns; b and x may alias."""
        wp.launch(
            _make_diagonal_kernel(b.ndim == 2),
            dim=(self.worlds, self.maxdim, max_columns),
            inputs=[
                self.info.dim,
                self.info.dim if active_dim is None else active_dim,
                offsets,
                strides,
                columns,
                self.diagonal,
                b,
                x,
                self.reverse,
                inverse_sqrt,
            ],
            device=self.device,
        )

    def reconstruct(self, a):
        """Reconstruct active matrices in the operator's original row order."""
        wp.launch(
            _reconstruct,
            dim=(self.worlds, self.maxdim, self.maxdim),
            inputs=[
                self.info.dim,
                self.lower,
                self.diagonal,
                a,
                self.reverse,
            ],
            device=self.device,
        )
