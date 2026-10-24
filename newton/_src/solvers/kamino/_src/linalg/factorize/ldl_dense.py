# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Implicit complete-pattern LDL storage for changing dense active dimensions."""

import numpy as np
import warp as wp

from ._array_ptr import get_float32_array_offset_ptr

wp.set_module_options({"enable_backward": False})


@wp.func
def _row(index: int, padded: int, reverse: bool):
    return (padded // 6 - 1 - index // 6) * 6 + index % 6 if reverse else index


@wp.kernel
def _factor(
    dims: wp.array[wp.int32],
    mio: wp.array[wp.int32],
    lio: wp.array[wp.int32],
    dio: wp.array[wp.int32],
    a: wp.array[wp.float32],
    lower: wp.array[wp.float32],
    diagonal: wp.array[wp.float32],
    failed: wp.array[wp.int32],
    reverse: bool,
    retry: bool,
    shift: float,
):
    world = wp.tid()
    if retry and failed[world] == 0:
        return
    n = dims[world]
    padded = (n + 5) // 6 * 6
    av = wp.array(ptr=get_float32_array_offset_ptr(a, mio[world]), shape=(n, n), dtype=wp.float32)
    lv = wp.array(ptr=get_float32_array_offset_ptr(lower, lio[world]), shape=(padded, padded), dtype=wp.float32)
    dv = wp.array(ptr=get_float32_array_offset_ptr(diagonal, dio[world]), shape=(1, padded), dtype=wp.float32)
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


@wp.kernel
def _triangular(
    dims: wp.array[wp.int32],
    active: wp.array[wp.int32],
    lio: wp.array[wp.int32],
    offsets: wp.array[wp.int32],
    strides: wp.array[wp.int32],
    columns: wp.array[wp.int32],
    lower: wp.array[wp.float32],
    b: wp.array[wp.float32],
    x: wp.array[wp.float32],
    reverse: bool,
    forward: bool,
):
    world, column = wp.tid()
    if column >= columns[world]:
        return
    padded = (dims[world] + 5) // 6 * 6
    n = active[world]
    bv = wp.array(ptr=get_float32_array_offset_ptr(b, offsets[world]), shape=(n, strides[world]), dtype=wp.float32)
    xv = wp.array(ptr=get_float32_array_offset_ptr(x, offsets[world]), shape=(n, strides[world]), dtype=wp.float32)
    lv = wp.array(ptr=get_float32_array_offset_ptr(lower, lio[world]), shape=(padded, padded), dtype=wp.float32)
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


@wp.kernel
def _diagonal(
    dims: wp.array[wp.int32],
    active: wp.array[wp.int32],
    dio: wp.array[wp.int32],
    offsets: wp.array[wp.int32],
    strides: wp.array[wp.int32],
    columns: wp.array[wp.int32],
    diagonal: wp.array[wp.float32],
    b: wp.array[wp.float32],
    x: wp.array[wp.float32],
    reverse: bool,
    inverse_sqrt: bool,
):
    world, row, column = wp.tid()
    if row >= active[world] or column >= columns[world]:
        return
    padded = (dims[world] + 5) // 6 * 6
    d = diagonal[dio[world] + _row(row, padded, reverse)]
    if inverse_sqrt:
        d = wp.sqrt(d)
    target = offsets[world] + row * strides[world] + column
    x[target] = b[target] / d


@wp.kernel
def _reconstruct(
    dims: wp.array[wp.int32],
    mio: wp.array[wp.int32],
    lio: wp.array[wp.int32],
    dio: wp.array[wp.int32],
    lower: wp.array[wp.float32],
    diagonal: wp.array[wp.float32],
    a: wp.array[wp.float32],
    reverse: bool,
):
    world, row, column = wp.tid()
    n = dims[world]
    if row >= n or column >= n:
        return
    padded = (n + 5) // 6 * 6
    i, j = _row(row, padded, reverse), _row(column, padded, reverse)
    value = wp.float32(0.0)
    for k in range(wp.min(i, j) + 1):
        value += lower[lio[world] + i * padded + k] * diagonal[dio[world] + k] * lower[lio[world] + j * padded + k]
    a[mio[world] + row * n + column] = value


class DenseLDLStorage:
    """Represent a complete block graph implicitly, with quadratic storage only."""

    def __init__(self, operator, ordering, device):
        self.info = operator.info
        self.device = device
        self.reverse = ordering == "rcm"
        capacities = [(int(n) + 5) // 6 * 6 for n in self.info.maxdim.numpy()]
        self.worlds = len(capacities)
        self.maxdim = max(capacities, default=0)
        self.lio = wp.array(np.cumsum([0, *[n * n for n in capacities[:-1]]]), dtype=wp.int32, device=device)
        self.dio = wp.array(np.cumsum([0, *capacities[:-1]]), dtype=wp.int32, device=device)
        self.lower = wp.zeros(sum(n * n for n in capacities), dtype=wp.float32, device=device)
        self.diagonal = wp.zeros(sum(capacities), dtype=wp.float32, device=device)
        self.failed = wp.zeros(self.worlds, dtype=wp.int32, device=device)

    def factorize(self, a, shift):
        self.failed.zero_()
        for retry in (False, True) if shift > 0.0 else (False,):
            wp.launch_tiled(
                _factor,
                dim=self.worlds,
                inputs=[
                    self.info.dim,
                    self.info.mio,
                    self.lio,
                    self.dio,
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
        wp.launch_tiled(
            _triangular,
            dim=(self.worlds, max_columns),
            inputs=[
                self.info.dim,
                self.info.dim if active_dim is None else active_dim,
                self.lio,
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
        wp.launch(
            _diagonal,
            dim=(self.worlds, self.maxdim, max_columns),
            inputs=[
                self.info.dim,
                self.info.dim if active_dim is None else active_dim,
                self.dio,
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
        wp.launch(
            _reconstruct,
            dim=(self.worlds, self.maxdim, self.maxdim),
            inputs=[
                self.info.dim,
                self.info.mio,
                self.lio,
                self.dio,
                self.lower,
                self.diagonal,
                a,
                self.reverse,
            ],
            device=self.device,
        )
