# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""RCM-reordered, semi-sparse blocked LLT solver with a fixed symbolic structure.

Kamino's :class:`~....linalg.factorize.llt_blocked_rcm_solver.LLTBlockedRCMSolver`
orders each block and builds its tile pattern from the numeric matrix at every
factorization. LOX knows the structure of its smooth systems from the joint
graph, so this solver computes the Reverse Cuthill-McKee ordering, the block
symbolic Cholesky fill, and the tiles visited by the triangular solves once on
the host, and reuses them for every numeric factorization. The numeric
factorization uses Kamino's tile-skipping kernels.

The caller-visible API is that of :class:`LLTBlockedSolver`:

.. code-block:: python

    solver = LLTBlockedRCMSolver(operator=operator, symbolic_adjacency=adjacency, block_size=32)
    solver.compute(A)  # permutes into L and factorizes in place with the fixed structure
    solver.solve(b, x)
"""

from __future__ import annotations

from collections import deque
from collections.abc import Sequence
from functools import lru_cache
from typing import Any

import numpy as np
import warp as wp

from .......core.types import override
from ....core.types import FloatType, to_warp_int32_array
from ....linalg.core import DenseLinearOperatorData, DenseSquareMultiLinearInfo
from ....linalg.factorize.llt_blocked_rcm import (
    llt_blocked_rcm_factorize,
    llt_blocked_rcm_factorize_parallel,
    make_llt_blocked_rcm_factorize_kernel,
    make_llt_blocked_rcm_parallel_factorize_kernels,
)
from ....linalg.linear import DirectSolver
from .llt_blocked_rcm import (
    llt_blocked_rcm_fixed_solve,
    llt_blocked_rcm_permute_matrix,
    make_llt_blocked_rcm_fixed_solve_kernel,
    make_llt_blocked_rcm_permute_matrix_kernel,
)

###
# Module interface
###

__all__ = ["LLTBlockedRCMSolver", "fixed_symbolic_data"]


###
# Module configs
###

wp.set_module_options({"enable_backward": False})


###
# Functions
###


def _reverse_cuthill_mckee(adjacency: Sequence[set[int]]) -> list[int]:
    """Compute a deterministic RCM permutation for a host-side graph."""
    degrees = [len(neighbors) for neighbors in adjacency]
    remaining = set(range(len(adjacency)))
    order = []
    while remaining:
        root = min(remaining, key=lambda vertex: (degrees[vertex], vertex))
        queue = deque([root])
        remaining.remove(root)
        while queue:
            vertex = queue.popleft()
            order.append(vertex)
            neighbors = sorted(
                (neighbor for neighbor in adjacency[vertex] if neighbor in remaining),
                key=lambda neighbor: (degrees[neighbor], neighbor),
            )
            for neighbor in neighbors:
                remaining.remove(neighbor)
                queue.append(neighbor)
    order.reverse()
    return order


@lru_cache(maxsize=32)
def _fixed_symbolic_block_data(
    dimension: int,
    adjacency_rows: tuple[tuple[int, ...], ...],
    block_size: int,
) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
    """Build the permutation, filled tile pattern, and solve tile traversal of one matrix block."""
    if len(adjacency_rows) != dimension:
        raise ValueError("Each symbolic adjacency graph must match its matrix dimension.")
    adjacency = []
    for row, neighbors in enumerate(adjacency_rows):
        neighbor_set = {int(neighbor) for neighbor in neighbors}
        if any(neighbor < 0 or neighbor >= dimension for neighbor in neighbor_set):
            raise ValueError("Symbolic adjacency indices must reference their matrix block.")
        neighbor_set.discard(row)
        adjacency.append(neighbor_set)
    if any(row not in adjacency[neighbor] for row, neighbors in enumerate(adjacency) for neighbor in neighbors):
        raise ValueError("symbolic_adjacency must be symmetric.")

    permutation = _reverse_cuthill_mckee(adjacency)
    inverse = [0] * dimension
    for reordered, original in enumerate(permutation):
        inverse[original] = reordered

    tile_count = (dimension + block_size - 1) // block_size
    tile_pattern = [0] * (tile_count * tile_count)
    for tile in range(tile_count):
        tile_pattern[tile * tile_count + tile] = 1
    for original_row, neighbors in enumerate(adjacency):
        tile_row = inverse[original_row] // block_size
        for original_col in neighbors:
            tile_col = inverse[original_col] // block_size
            high = max(tile_row, tile_col)
            low = min(tile_row, tile_col)
            tile_pattern[high * tile_count + low] = 1

    for column in range(tile_count):
        for row in range(column + 1, tile_count):
            if tile_pattern[row * tile_count + column] != 0:
                continue
            for previous in range(column):
                if tile_pattern[row * tile_count + previous] != 0 and tile_pattern[column * tile_count + previous] != 0:
                    tile_pattern[row * tile_count + column] = 1
                    break

    tile_traversal = [-1] * (2 * tile_count * tile_count)
    backward_offset = tile_count * tile_count
    for row in range(tile_count):
        forward = [column for column in range(row) if tile_pattern[row * tile_count + column] != 0]
        backward = [
            following for following in range(row + 1, tile_count) if tile_pattern[following * tile_count + row] != 0
        ]
        row_offset = row * tile_count
        tile_traversal[row_offset : row_offset + len(forward)] = forward
        tile_traversal[backward_offset + row_offset : backward_offset + row_offset + len(backward)] = backward
    return tuple(permutation), tuple(tile_pattern), tuple(tile_traversal)


def fixed_symbolic_data(
    dimensions: Sequence[int],
    adjacency_blocks: Sequence[Sequence[Sequence[int]]],
    block_size: int,
) -> tuple[list[int], list[int], list[int]]:
    """Build the packed permutations, filled tile patterns, and ordered solve work of the blocks."""
    if len(adjacency_blocks) != len(dimensions):
        raise ValueError("symbolic_adjacency must contain one graph per matrix block.")

    packed_permutation = []
    packed_tile_pattern = []
    packed_tile_traversal = []
    block_cache = {}
    for dimension, adjacency_rows in zip(dimensions, adjacency_blocks, strict=True):
        cache_key = (dimension, id(adjacency_rows))
        block_data = block_cache.get(cache_key)
        if block_data is None:
            try:
                hash(adjacency_rows)
                normalized_adjacency = adjacency_rows
            except TypeError:
                normalized_adjacency = tuple(
                    tuple(int(neighbor) for neighbor in neighbors) for neighbors in adjacency_rows
                )
            block_data = _fixed_symbolic_block_data(dimension, normalized_adjacency, block_size)
            block_cache[cache_key] = block_data
        permutation, tile_pattern, tile_traversal = block_data
        packed_permutation.extend(permutation)
        packed_tile_pattern.extend(tile_pattern)
        packed_tile_traversal.extend(tile_traversal)

    return packed_permutation, packed_tile_pattern, packed_tile_traversal


###
# Interfaces
###


class LLTBlockedRCMSolver(DirectSolver[wp.float32, wp.int32]):
    """RCM-reordered, semi-sparse blocked LLT (Cholesky) solver with a fixed symbolic structure.

    Same public API as :class:`LLTBlockedSolver`. At allocation, the host computes
    for each block, from its structural adjacency, the RCM permutation ``P``, the
    tile pattern of ``L`` inflated by the block symbolic Cholesky fill, and the
    nonzero tiles visited by each tile row of the triangular solves.

    1. ``compute(A)`` writes ``P A P^T`` into ``L`` and factorizes it in place,
       ``P A P^T = L L^T``, with Kamino's tile-skipping kernels. They are left-looking:
       each tile of a column is read before its factor tile is written.
    2. ``solve(b, x)`` permutes ``b``, runs the forward and backward substitutions
       over the precomputed tiles, and un-permutes the result.

    All launches have fixed dimensions, so ``compute`` and ``solve`` can be
    captured in a CUDA graph.
    """

    def __init__(
        self,
        operator: DenseLinearOperatorData[wp.float32, wp.int32] | None = None,
        symbolic_adjacency: Sequence[Sequence[Sequence[int]]] = (),
        block_size: int = 32,
        solve_block_dim: int = 256,
        factorize_block_dim: int = 128,
        parallel_factorization: bool = False,
        atol: float | None = None,
        rtol: float | None = None,
        ftol: float | None = None,
        dtype: FloatType = wp.float32,
        device: wp.DeviceLike | None = None,
        **kwargs: dict[str, Any],
    ):
        """
        Args:
            operator: Optional operator; if provided, :meth:`finalize` is called during init.
            symbolic_adjacency: Structural adjacency of every matrix block: for each
                row, the columns of its possibly nonzero off-diagonal entries.
            block_size: Tile size of the factorization and solve kernels.
            solve_block_dim: Thread-block size of the solve kernels.
            factorize_block_dim: Thread-block size of the factorization kernels.
            parallel_factorization: Whether to factorize the off-diagonal tiles of each
                Cholesky panel in parallel.
        """
        # The kernels are hard-coded to wp.float32.
        if dtype != wp.float32:
            raise NotImplementedError("LLTBlockedRCMSolver currently supports only wp.float32.")

        self._L: wp.array[dtype] | None = None
        self._y: wp.array[dtype] | None = None
        self._x_hat: wp.array[dtype] | None = None
        self._P: wp.array[wp.int32] | None = None
        self._tile_pattern: wp.array[wp.int32] | None = None
        self._tile_traversal: wp.array[wp.int32] | None = None
        self._tpo: wp.array[wp.int32] | None = None
        self._max_dim: int = 0

        self._block_size: int = block_size
        self._solve_block_dim: int = solve_block_dim
        self._factorize_block_dim: int = factorize_block_dim
        self._parallel_factorization = parallel_factorization
        self._symbolic_adjacency = symbolic_adjacency

        self._factorize_kernel = make_llt_blocked_rcm_factorize_kernel(block_size)
        self._parallel_factorize_kernels = make_llt_blocked_rcm_parallel_factorize_kernels(block_size)
        self._solve_kernel = make_llt_blocked_rcm_fixed_solve_kernel(block_size)
        self._permute_matrix_kernel = make_llt_blocked_rcm_permute_matrix_kernel()

        super().__init__(
            operator=operator,
            atol=atol,
            rtol=rtol,
            ftol=ftol,
            dtype=dtype,
            device=device,
            **kwargs,
        )

    ###
    # Properties
    ###

    @property
    def L(self) -> wp.array:
        if self._L is None:
            raise ValueError("The factorization array has not been allocated!")
        return self._L

    @property
    def y(self) -> wp.array:
        if self._y is None:
            raise ValueError("The intermediate result array has not been allocated!")
        return self._y

    @property
    def P(self) -> wp.array:
        """Concatenated per-block RCM permutation (wp.int32[total_vec_size])."""
        if self._P is None:
            raise ValueError("Permutation array has not been allocated!")
        return self._P

    @property
    def tile_pattern(self) -> wp.array:
        """Concatenated per-block tile-sparsity mask (wp.int32, lower-tri inflated by fill-in)."""
        if self._tile_pattern is None:
            raise ValueError("Tile pattern array has not been allocated!")
        return self._tile_pattern

    @property
    def block_size(self) -> int:
        """Return the tile size used by factorization and solve kernels."""
        return self._block_size

    ###
    # Implementation
    ###

    @override
    def _allocate_impl(self, A: DenseLinearOperatorData[wp.float32, wp.int32], **kwargs: dict[str, Any]) -> None:
        if A.info is None:
            raise ValueError("The provided operator does not have any associated info!")
        if not isinstance(A.info, DenseSquareMultiLinearInfo):
            raise ValueError("LLT factorization requires a square matrix.")

        info = self._operator.info
        self._max_dim = int(info.max_dimension)

        # Per-block tile-pattern layout: n_tiles_i^2 entries per block.
        dims = list(info.dimensions)
        tile_counts = (np.asarray(dims, dtype=np.int64) + self._block_size - 1) // self._block_size
        tp_offsets = np.empty(len(dims) + 1, dtype=np.int64)
        tp_offsets[0] = 0
        np.cumsum(tile_counts * tile_counts, out=tp_offsets[1:])
        permutation, tile_pattern, tile_traversal = fixed_symbolic_data(
            dims, self._symbolic_adjacency, self._block_size
        )

        with wp.ScopedDevice(self._device):
            self._L = wp.zeros(shape=(info.total_mat_size,), dtype=self._dtype)
            self._y = wp.zeros(shape=(info.total_vec_size,), dtype=self._dtype)
            self._x_hat = wp.zeros(shape=(info.total_vec_size,), dtype=self._dtype)
            self._P = wp.array(permutation, dtype=wp.int32)
            self._tile_pattern = wp.array(tile_pattern, dtype=wp.int32)
            self._tile_traversal = wp.array(tile_traversal if tile_traversal else [-1], dtype=wp.int32)
            self._tpo = to_warp_int32_array(tp_offsets[:-1])
        self._has_factors = False

    @override
    def _reset_impl(self) -> None:
        # The fixed symbolic data survives resets.
        self._L.zero_()
        self._y.zero_()
        self._x_hat.zero_()
        self._has_factors = False

    @override
    def _factorize_impl(self, A: wp.array[Any]) -> None:
        info = self._operator.info
        num_blocks = info.num_blocks
        llt_blocked_rcm_permute_matrix(
            kernel=self._permute_matrix_kernel,
            dim=info.dim,
            mio=info.mio,
            vio=info.vio,
            P=self._P,
            A=A,
            A_hat=self._L,
            num_blocks=num_blocks,
            max_dim=self._max_dim,
            device=self._device,
        )
        if self._parallel_factorization:
            llt_blocked_rcm_factorize_parallel(
                kernels=self._parallel_factorize_kernels,
                dim=info.dim,
                mio=info.mio,
                tpo=self._tpo,
                A=self._L,
                tile_pattern=self._tile_pattern,
                L=self._L,
                num_blocks=num_blocks,
                max_tiles=(self._max_dim + self._block_size - 1) // self._block_size,
                block_dim=self._factorize_block_dim,
                device=self._device,
            )
        else:
            llt_blocked_rcm_factorize(
                kernel=self._factorize_kernel,
                dim=info.dim,
                mio=info.mio,
                tpo=self._tpo,
                A=self._L,
                tile_pattern=self._tile_pattern,
                L=self._L,
                num_blocks=num_blocks,
                block_dim=self._factorize_block_dim,
                device=self._device,
            )

    @override
    def _reconstruct_impl(self, A: wp.array[Any]) -> None:
        raise NotImplementedError("LLT matrix reconstruction is not yet implemented.")

    @override
    def _solve_impl(self, b: wp.array[Any], x: wp.array[Any]) -> None:
        info = self._operator.info
        # Solve L L^T x_hat = P b and scatter x_hat -> x.
        llt_blocked_rcm_fixed_solve(
            kernel=self._solve_kernel,
            dim=info.dim,
            mio=info.mio,
            vio=info.vio,
            tpo=self._tpo,
            P=self._P,
            L=self._L,
            tile_traversal=self._tile_traversal,
            b=b,
            y=self._y,
            x_hat=self._x_hat,
            x=x,
            num_blocks=info.num_blocks,
            block_dim=self._solve_block_dim,
            device=self._device,
        )

    @override
    def _solve_inplace_impl(self, x: wp.array[Any]) -> None:
        raise NotImplementedError("In-place LLT solves are not implemented; use solve().")
