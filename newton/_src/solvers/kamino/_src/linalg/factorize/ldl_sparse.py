# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Level-scheduled block-sparse LDL transpose factorization for SPD systems.

The reordered matrix is L D L^T: L is unit lower triangular and D is stored
separately as six scalar pivots per identity-padded block. Numeric elimination
uses no square roots or numerical pivoting. Symbolic elimination and allocation
happen at finalize time; numeric operations are capturable on CUDA and also
run on CPU. No physics-specific types are used here.
"""

from __future__ import annotations

from collections.abc import Sequence
from functools import cache

import numpy as np
import warp as wp

from ..core import DenseLinearOperatorData, DenseSquareMultiLinearInfo
from ..linear import DirectSolver

wp.set_module_options({"enable_backward": False})

_Vector = wp.types.vector(length=6, dtype=wp.float32)
_BlockRows = Sequence[Sequence[Sequence[int]]]
_BlockEdges = Sequence[Sequence[tuple[int, int]]]


def _symbolic(groups, edges):
    """Order blocks by minimum degree and compute exact symbolic fill."""
    graph = [set() for _ in groups]
    for i, j in edges:
        if not (0 <= i < len(groups) and 0 <= j < len(groups)):
            raise ValueError("Block edge index outside the row partition.")
        if i != j:
            graph[i].add(j)
            graph[j].add(i)
    remaining = set(range(len(groups)))
    order, columns = [], []
    while remaining:
        node = min(remaining, key=lambda i: (len(graph[i] & remaining), i))
        adjacent = graph[node] & remaining
        order.append(node)
        columns.append(sorted(adjacent))
        for other in adjacent:
            graph[other].update(adjacent - {other})
        remaining.remove(node)
    inverse = {node: i for i, node in enumerate(order)}
    return [groups[i] for i in order], [sorted(inverse[j] for j in col) for col in columns]


@wp.kernel
def _gather_matrix(
    dim: wp.array[wp.int32],
    mio: wp.array[wp.int32],
    worlds: wp.array[wp.int32],
    rows: wp.array2d[wp.int32],
    block_i: wp.array[wp.int32],
    block_j: wp.array[wp.int32],
    source: wp.array[wp.float32],
    values: wp.array3d[wp.float32],
):
    block, r, c = wp.tid()
    i, j = block_i[block], block_j[block]
    world = worlds[i]
    row, col = rows[i, r], rows[j, c]
    value = wp.float32(0.0)
    if row >= 0 and col >= 0 and row < dim[world] and col < dim[world]:
        value = source[mio[world] + row * dim[world] + col]
    elif i == j and r == c:
        value = wp.float32(1.0)
    values[block, r, c] = value


@wp.kernel
def _factor_level(
    pivots: wp.array[wp.int32],
    offset: int,
    worlds: wp.array[wp.int32],
    rows: wp.array2d[wp.int32],
    dim: wp.array[wp.int32],
    col_ptr: wp.array[wp.int32],
    row_idx: wp.array[wp.int32],
    row_ptr: wp.array[wp.int32],
    row_slots: wp.array[wp.int32],
    block_j: wp.array[wp.int32],
    pred_ptr: wp.array[wp.int32],
    pred_left: wp.array[wp.int32],
    pred_right: wp.array[wp.int32],
    values: wp.array3d[wp.float32],
    factors: wp.array3d[wp.float32],
    diagonal: wp.array2d[wp.float32],
    pivot_count: int,
    retry: bool,
    failed: wp.array[wp.int32],
    shift: float,
):
    """Apply left-looking L D L^T updates, then factor each six-row pivot block."""
    j = pivots[offset + wp.tid()]
    world = worlds[j]
    if retry and failed[world] == 0:
        return
    schur = wp.tile_reshape(wp.tile_load(values, shape=(1, 6, 6), offset=(j, 0, 0)), (6, 6))
    if retry:
        identity = wp.tile_zeros(shape=(6, 6), dtype=wp.float32)
        for r in range(6):
            if rows[j, r] >= 0 and rows[j, r] < dim[world]:
                identity[r, r] = shift
        schur += identity
    for p in range(row_ptr[j], row_ptr[j + 1]):
        slot = row_slots[p]
        block = wp.tile_reshape(wp.tile_load(factors, shape=(1, 6, 6), offset=(slot, 0, 0)), (6, 6))
        d = wp.tile_load(diagonal, shape=(1, 6), offset=(block_j[slot], 0))
        weighted = wp.tile_map(wp.mul, block, wp.tile_broadcast(d, shape=(6, 6)))
        wp.tile_matmul(weighted, wp.tile_transpose(block), schur, alpha=-1.0)
    lower = wp.tile_zeros(shape=(6, 6), dtype=wp.float32)
    d = wp.tile_zeros(shape=(1, 6), dtype=wp.float32)
    for c in range(6):
        pivot = schur[c, c]
        for k in range(c):
            pivot -= lower[c, k] * d[0, k] * lower[c, k]
        d[0, c] = pivot
        lower[c, c] = wp.float32(1.0)
        for r in range(c + 1, 6):
            value = schur[r, c]
            for k in range(c):
                value -= lower[r, k] * d[0, k] * lower[c, k]
            lower[r, c] = value / pivot
    wp.tile_store(diagonal, d, offset=(j, 0))
    wp.tile_store(factors, wp.tile_reshape(lower, (1, 6, 6)), offset=(j, 0, 0))
    for entry in range(col_ptr[j], col_ptr[j + 1]):
        slot = pivot_count + entry
        block = wp.tile_reshape(wp.tile_load(values, shape=(1, 6, 6), offset=(slot, 0, 0)), (6, 6))
        for p in range(pred_ptr[entry], pred_ptr[entry + 1]):
            left = wp.tile_reshape(wp.tile_load(factors, shape=(1, 6, 6), offset=(pred_left[p], 0, 0)), (6, 6))
            right = wp.tile_reshape(wp.tile_load(factors, shape=(1, 6, 6), offset=(pred_right[p], 0, 0)), (6, 6))
            previous_d = wp.tile_load(diagonal, shape=(1, 6), offset=(block_j[pred_left[p]], 0))
            weighted = wp.tile_map(wp.mul, left, wp.tile_broadcast(previous_d, shape=(6, 6)))
            wp.tile_matmul(weighted, wp.tile_transpose(right), block, alpha=-1.0)
        transposed = wp.tile_transpose(block)
        wp.tile_lower_solve_inplace(lower, transposed)
        off_diagonal = wp.tile_map(wp.div, wp.tile_transpose(transposed), wp.tile_broadcast(d, shape=(6, 6)))
        wp.tile_store(factors, wp.tile_reshape(off_diagonal, (1, 6, 6)), offset=(slot, 0, 0))


@wp.kernel
def _check_factors(worlds: wp.array[wp.int32], diagonal: wp.array2d[wp.float32], failed: wp.array[wp.int32]):
    pivot, row = wp.tid()
    value = diagonal[pivot, row]
    if not wp.isfinite(value) or value <= 0.0:
        wp.atomic_max(failed, worlds[pivot], 1)


@cache
def _make_solve_kernel(forward: bool, serial: bool):
    @wp.kernel(module="unique")
    def solve_kernel(
        pivots: wp.array[wp.int32],
        offset: int,
        world_ptr: wp.array[wp.int32],
        worlds: wp.array[wp.int32],
        rows: wp.array2d[wp.int32],
        starts: wp.array[wp.int32],
        neighbors: wp.array[wp.int32],
        slots: wp.array[wp.int32],
        factors: wp.array3d[wp.float32],
        dim: wp.array[wp.int32],
        offsets: wp.array[wp.int32],
        strides: wp.array[wp.int32],
        columns: wp.array[wp.int32],
        source: wp.array[wp.float32],
        result: wp.array[wp.float32],
    ):
        local, column = wp.tid()
        first, last = offset + local, offset + local + 1
        if wp.static(serial):
            first = int(world_ptr[local])
            last = int(world_ptr[local + 1])
        for index in range(first, last):
            i = wp.int32(0)
            if wp.static(serial):
                if wp.static(forward):
                    i = index
                else:
                    i = last - 1 - (index - first)
            else:
                i = pivots[index]
            world = worlds[i]
            n = dim[world]
            if n == 0 or column >= columns[world]:
                return
            start, stride = offsets[world], strides[world]
            v = _Vector(0.0)
            for r in range(6):
                row = rows[i, r]
                if row >= 0 and row < n:
                    v[r] = source[start + row * stride + column]
            for entry in range(starts[i], starts[i + 1]):
                j, slot = neighbors[entry], slots[entry]
                for c in range(6):
                    col = rows[j, c]
                    if col >= 0 and col < n:
                        x = result[start + col * stride + column]
                        for r in range(6):
                            if wp.static(forward):
                                v[r] -= factors[slot, r, c] * x
                            else:
                                v[r] -= factors[slot, c, r] * x
            for step in range(6):
                if wp.static(forward):
                    r = step
                    for c in range(r):
                        v[r] -= factors[i, r, c] * v[c]
                else:
                    r = 5 - step
                    for c in range(r + 1, 6):
                        v[r] -= factors[i, c, r] * v[c]
            for r in range(6):
                row = rows[i, r]
                if row >= 0 and row < n:
                    result[start + row * stride + column] = v[r]

    return solve_kernel


@wp.kernel
def _solve_diagonal(
    worlds: wp.array[wp.int32],
    rows: wp.array2d[wp.int32],
    diagonal: wp.array2d[wp.float32],
    dim: wp.array[wp.int32],
    offsets: wp.array[wp.int32],
    strides: wp.array[wp.int32],
    columns: wp.array[wp.int32],
    inverse_sqrt: bool,
    source: wp.array[wp.float32],
    result: wp.array[wp.float32],
):
    pivot, r, column = wp.tid()
    world, row = worlds[pivot], rows[pivot, r]
    if row < 0 or row >= dim[world] or column >= columns[world]:
        return
    index = offsets[world] + row * strides[world] + column
    d = diagonal[pivot, r]
    if inverse_sqrt:
        d = wp.sqrt(d)
    result[index] = source[index] / d


@wp.func
def _factor_entry(
    i: int,
    r: int,
    j: int,
    c: int,
    row_ptr: wp.array[wp.int32],
    col_idx: wp.array[wp.int32],
    slots: wp.array[wp.int32],
    factors: wp.array3d[wp.float32],
):
    value = wp.float32(0.0)
    if i == j and r >= c:
        value = factors[i, r, c]
    elif i > j:
        for entry in range(row_ptr[i], row_ptr[i + 1]):
            if col_idx[entry] == j:
                value = factors[slots[entry], r, c]
    return value


@wp.kernel
def _reconstruct(
    dim: wp.array[wp.int32],
    mio: wp.array[wp.int32],
    vio: wp.array[wp.int32],
    inverse: wp.array[wp.int32],
    row_ptr: wp.array[wp.int32],
    col_idx: wp.array[wp.int32],
    slots: wp.array[wp.int32],
    factors: wp.array3d[wp.float32],
    diagonal: wp.array2d[wp.float32],
    result: wp.array[wp.float32],
):
    world, row, col = wp.tid()
    n = dim[world]
    if row >= n or col >= n:
        return
    ir, jc = inverse[vio[world] + row], inverse[vio[world] + col]
    value = wp.float32(0.0)
    for k in range(n):
        kc = inverse[vio[world] + k]
        value += (
            _factor_entry(ir // 6, ir % 6, kc // 6, kc % 6, row_ptr, col_idx, slots, factors)
            * diagonal[kc // 6, kc % 6]
            * _factor_entry(jc // 6, jc % 6, kc // 6, kc % 6, row_ptr, col_idx, slots, factors)
        )
    result[mio[world] + row * n + col] = value


class LDLBlockSparseSolver(DirectSolver):
    """Factor independent SPD matrices with a fixed block sparsity pattern.

    ``block_rows[w]`` partitions world ``w`` into groups of one to six scalar
    rows. ``block_edges[w]`` contains its undirected block edges. Omitting both
    uses contiguous six-row blocks and a full pattern, suitable for changing
    active dimensions in dense operators. Numeric zeros never define topology.

    ``factors`` stores the unit-lower L blocks, including unit-diagonal blocks.
    ``diagonal`` stores the scalar entries of D in block order, with ones for
    padded rows. The permutation is described by ``rows``. No numerical
    pivoting is performed; this backend targets positive-definite systems.
    """

    def __init__(
        self,
        operator: DenseLinearOperatorData | None = None,
        *,
        block_rows: _BlockRows | None = None,
        block_edges: _BlockEdges | None = None,
        failed_pivot_shift: float = 0.0,
        dtype: type = wp.float32,
        device: wp.DeviceLike | None = None,
    ) -> None:
        if not np.isfinite(failed_pivot_shift) or failed_pivot_shift < 0:
            raise ValueError("failed_pivot_shift must be finite and nonnegative.")
        self._failed_pivot_shift = failed_pivot_shift
        self._block_rows = block_rows
        self._block_edges = block_edges
        super().__init__(operator=operator, dtype=dtype, device=device)

    def _allocate_impl(self, operator: DenseLinearOperatorData) -> None:
        if not isinstance(operator.info, DenseSquareMultiLinearInfo):
            raise ValueError("LDL factorization requires a square matrix.")
        if operator.info.dtype != wp.float32:
            raise ValueError("LDLBlockSparseSolver supports float32 matrices.")
        if operator.info.itype != wp.int32:
            raise ValueError("LDLBlockSparseSolver supports int32 indices.")
        info = operator.info
        self._device = info.dim.device
        capacities = info.maxdim.numpy().tolist()
        self._maxdim = max(capacities, default=0)
        self._world_count = len(capacities)
        block_rows, block_edges = self._block_rows, self._block_edges
        if block_rows is None:
            block_rows = [[list(range(i, min(i + 6, n))) for i in range(0, n, 6)] for n in capacities]
        if len(block_rows) != len(capacities):
            raise ValueError("Provide one row partition per matrix.")
        if block_edges is None:
            block_edges = [[(i, j) for i in range(len(groups)) for j in range(i)] for groups in block_rows]
        if len(block_edges) != len(capacities):
            raise ValueError("Provide one block graph per matrix.")
        rows, worlds, columns, world_ptr = [], [], [], [0]
        cache_symbolic = {}
        self._row_maps = []
        for world, (n, row_groups, edges) in enumerate(zip(capacities, block_rows, block_edges, strict=True)):
            groups = tuple(tuple(int(r) for r in group) for group in row_groups)
            if any(not 1 <= len(group) <= 6 for group in groups) or sorted(r for g in groups for r in g) != list(
                range(n)
            ):
                raise ValueError("Block rows must partition each matrix into groups of one to six rows.")
            key = (groups, tuple(sorted({(min(i, j), max(i, j)) for i, j in edges})))
            if key not in cache_symbolic:
                cache_symbolic[key] = _symbolic(*key)
            ordered, local_columns = cache_symbolic[key]
            base = len(rows)
            mapping = {}
            for i, group in enumerate(ordered):
                mapping.update({r: (base + i, c) for c, r in enumerate(group)})
                rows.append([*group, *([-1] * (6 - len(group)))])
                worlds.append(world)
                columns.append([base + j for j in local_columns[i]])
            self._row_maps.append(mapping)
            world_ptr.append(len(rows))
        self.pivot_count = len(rows)
        col_ptr, row_idx, block_i, block_j = [0], [], list(range(len(rows))), list(range(len(rows)))
        predecessors = [{} for _ in rows]
        self._block_lookup = {(i, i): i for i in range(len(rows))}
        for j, column in enumerate(columns):
            for i in column:
                slot = len(block_i)
                self._block_lookup[i, j] = slot
                predecessors[i][j] = slot
                row_idx.append(i)
                block_i.append(i)
                block_j.append(j)
            col_ptr.append(len(row_idx))
        row_ptr, col_idx, row_slots = [0], [], []
        levels = []
        for previous in predecessors:
            col_idx.extend(previous)
            row_slots.extend(previous.values())
            row_ptr.append(len(col_idx))
            levels.append(1 + max((levels[j] for j in previous), default=-1))
        pred_ptr, pred_left, pred_right = [0], [], []
        for i, j in zip(block_i[self.pivot_count :], block_j[self.pivot_count :], strict=True):
            for k, slot in predecessors[j].items():
                if k in predecessors[i]:
                    pred_left.append(predecessors[i][k])
                    pred_right.append(slot)
            pred_ptr.append(len(pred_left))
        vio = info.vio.numpy()
        inverse = np.zeros(info.total_vec_size, dtype=np.int32)
        for w, mapping in enumerate(self._row_maps):
            for row, (i, r) in mapping.items():
                inverse[vio[w] + row] = 6 * i + r
        level_indices = sorted(range(len(rows)), key=lambda i: (levels[i], i))
        counts = np.bincount(levels) if levels else []
        self._levels = list(zip(np.cumsum([0, *counts[:-1]]).tolist(), counts, strict=True)) if len(counts) else []

        def ints(values):
            return wp.array(values, dtype=wp.int32, device=self.device)

        self.rows = ints(np.asarray(rows, dtype=np.int32).reshape(-1, 6))
        self.worlds = ints(worlds)
        self._world_ptr = ints(world_ptr)
        self.block_i, self.block_j = ints(block_i), ints(block_j)
        self._col_ptr, self._row_idx = ints(col_ptr), ints(row_idx)
        self._row_ptr, self._col_idx, self._row_slots = ints(row_ptr), ints(col_idx), ints(row_slots)
        self._col_slots = ints(list(range(self.pivot_count, len(block_i))))
        self._pred_ptr, self._pred_left, self._pred_right = ints(pred_ptr), ints(pred_left), ints(pred_right)
        self._pivots, self._inverse = ints(level_indices), ints(inverse)
        self.values = wp.zeros((len(block_i), 6, 6), dtype=wp.float32, device=self.device)
        self.factors = wp.zeros_like(self.values)
        self.diagonal = wp.zeros((self.pivot_count, 6), dtype=wp.float32, device=self.device)
        self.failed = wp.zeros(len(capacities), dtype=wp.int32, device=self.device)
        self._ones = wp.ones(len(capacities), dtype=wp.int32, device=self.device)
        self._has_factors = False

    def entry_offsets(self, worlds: Sequence[int], rows: Sequence[int], cols: Sequence[int]) -> np.ndarray:
        """Map structural scalar entries to flat block storage before capture."""
        result = []
        for world, row, col in zip(worlds, rows, cols, strict=True):
            i, r = self._row_maps[int(world)][int(row)]
            j, c = self._row_maps[int(world)][int(col)]
            if i < j:
                i, j, r, c = j, i, c, r
            result.append(self._block_lookup[i, j] * 36 + r * 6 + c)
        return np.asarray(result, dtype=np.int32)

    def _reset_impl(self) -> None:
        self._has_factors = False

    def _factorize_impl(self, A: wp.array[wp.float32]) -> None:
        wp.launch(
            _gather_matrix,
            dim=self.values.shape,
            inputs=[
                self.operator.info.dim,
                self.operator.info.mio,
                self.worlds,
                self.rows,
                self.block_i,
                self.block_j,
                A,
                self.values,
            ],
            device=self.device,
        )
        self.factorize_values()

    def factorize_values(self) -> None:
        """Factor the assembled sparse values, retaining them for a masked retry."""
        self.failed.zero_()
        self._factor_levels(False)
        if self._failed_pivot_shift > 0.0:
            wp.launch(
                _check_factors,
                dim=(self.pivot_count, 6),
                inputs=[self.worlds, self.diagonal, self.failed],
                device=self.device,
            )
            self._factor_levels(True)
        self._has_factors = True

    def _factor_levels(self, retry: bool) -> None:
        for offset, count in self._levels:
            wp.launch_tiled(
                _factor_level,
                dim=int(count),
                inputs=[
                    self._pivots,
                    int(offset),
                    self.worlds,
                    self.rows,
                    self.operator.info.dim,
                    self._col_ptr,
                    self._row_idx,
                    self._row_ptr,
                    self._row_slots,
                    self.block_j,
                    self._pred_ptr,
                    self._pred_left,
                    self._pred_right,
                    self.values,
                    self.factors,
                    self.diagonal,
                    self.pivot_count,
                    retry,
                    self.failed,
                    self._failed_pivot_shift,
                ],
                block_dim=32,
                device=self.device,
            )

    def solve_matrix(
        self,
        b: wp.array[wp.float32],
        x: wp.array[wp.float32],
        *,
        offsets: wp.array[wp.int32],
        strides: wp.array[wp.int32],
        columns: wp.array[wp.int32],
        max_columns: int,
        forward_only: bool = False,
        active_dim: wp.array[wp.int32] | None = None,
    ) -> None:
        """Solve L D L^T for multiple RHS, or only L when forward_only is set."""
        self._check_has_factorization()
        for forward in (True,) if forward_only else (True, False):
            if not forward:
                self.solve_diagonal(
                    x,
                    x,
                    offsets=offsets,
                    strides=strides,
                    columns=columns,
                    max_columns=max_columns,
                    active_dim=active_dim,
                )
            self.solve_triangular(
                b if forward else x,
                x,
                forward=forward,
                offsets=offsets,
                strides=strides,
                columns=columns,
                max_columns=max_columns,
                active_dim=active_dim,
            )

    def solve_triangular(
        self,
        b: wp.array[wp.float32],
        x: wp.array[wp.float32],
        *,
        forward: bool,
        offsets: wp.array[wp.int32],
        strides: wp.array[wp.int32],
        columns: wp.array[wp.int32],
        max_columns: int,
        active_dim: wp.array[wp.int32] | None = None,
    ) -> None:
        """Solve with unit-lower L or L^T in original row order, without D scaling."""
        self._check_has_factorization()
        starts, neighbors, slots = (
            (self._row_ptr, self._col_idx, self._row_slots)
            if forward
            else (self._col_ptr, self._row_idx, self._col_slots)
        )
        # Small single-RHS batches benefit from eliminating level-launch
        # overhead, especially inside APGD's repeated Schur products. Each
        # thread owns a complete world, so no inter-thread synchronization is
        # required while traversing its pivots in elimination order.
        serial = self.device.is_cuda and self._world_count <= 16 and max_columns == 1
        schedule = [(0, self._world_count)] if serial else (self._levels if forward else reversed(self._levels))
        for offset, count in schedule:
            wp.launch(
                _make_solve_kernel(forward, serial),
                dim=(int(count), max_columns),
                inputs=[
                    self._pivots,
                    int(offset),
                    self._world_ptr,
                    self.worlds,
                    self.rows,
                    starts,
                    neighbors,
                    slots,
                    self.factors,
                    self.operator.info.dim if active_dim is None else active_dim,
                    offsets,
                    strides,
                    columns,
                    b,
                    x,
                ],
                device=self.device,
            )

    def solve_diagonal(
        self,
        b: wp.array[wp.float32],
        x: wp.array[wp.float32],
        *,
        offsets: wp.array[wp.int32],
        strides: wp.array[wp.int32],
        columns: wp.array[wp.int32],
        max_columns: int,
        inverse_sqrt: bool = False,
        active_dim: wp.array[wp.int32] | None = None,
    ) -> None:
        """Apply D^-1, or D^-1/2 for a whitened response, optionally in place."""
        self._check_has_factorization()
        wp.launch(
            _solve_diagonal,
            dim=(self.pivot_count, 6, max_columns),
            inputs=[
                self.worlds,
                self.rows,
                self.diagonal,
                self.operator.info.dim if active_dim is None else active_dim,
                offsets,
                strides,
                columns,
                inverse_sqrt,
                b,
                x,
            ],
            device=self.device,
        )

    def _solve_impl(self, b: wp.array[wp.float32], x: wp.array[wp.float32]) -> None:
        self.solve_matrix(b, x, offsets=self.operator.info.vio, strides=self._ones, columns=self._ones, max_columns=1)

    def _solve_inplace_impl(self, x: wp.array[wp.float32]) -> None:
        self._solve_impl(x, x)

    def _reconstruct_impl(self, A: wp.array[wp.float32]) -> None:
        info = self.operator.info
        wp.launch(
            _reconstruct,
            dim=(self._world_count, self._maxdim, self._maxdim),
            inputs=[
                info.dim,
                info.mio,
                info.vio,
                self._inverse,
                self._row_ptr,
                self._col_idx,
                self._row_slots,
                self.factors,
                self.diagonal,
                A,
            ],
            device=self.device,
        )
