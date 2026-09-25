# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""KAMINO: Linear Algebra: RCM-reordered semi-sparse Blocked LLT solver.

Mirrors :class:`LLTBlockedSolver` from :mod:`linalg.linear` but transparently
computes and applies a per-block fill-reducing Reverse Cuthill-McKee (RCM)
permutation, and uses a per-block tile-granularity zero-block mask
("semi-sparse") to skip work on guaranteed-zero tiles during factorize and
solve.

The caller-visible API is identical to :class:`LLTBlockedSolver`:

.. code-block:: python

    solver = LLTBlockedRCMSolver(operator=operator, block_size=32, dtype=wp.float32)
    solver.compute(A)  # factorizes; reordering is internal
    solver.solve(b, x)  # or: solver.solve_inplace(x)

The reordering ``P`` and its inverse ``inv_P`` are stored on the solver next
to the factorization buffer ``L``. They are exposed as read-only properties
for debugging/introspection.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Sequence
from functools import lru_cache
from typing import Any

import numpy as np
import warp as wp

from ......core.types import override
from ...core.types import FloatType, to_warp_int32_array
from ..core import DenseLinearOperatorData, DenseSquareMultiLinearInfo
from ..linear import DirectSolver
from . import rcm_batch as _rcm_batch
from .llt_blocked_rcm import (
    llt_blocked_rcm_factorize,
    llt_blocked_rcm_factorize_parallel,
    llt_blocked_rcm_fused_permute_and_tp,
    llt_blocked_rcm_permute_matrix,
    llt_blocked_rcm_permute_vector,
    llt_blocked_rcm_solve,
    llt_blocked_rcm_solve_inplace,
    llt_blocked_rcm_symbolic_fill_in,
    make_llt_blocked_rcm_factorize_kernel,
    make_llt_blocked_rcm_fused_permute_and_tp_kernel,
    make_llt_blocked_rcm_parallel_factorize_kernels,
    make_llt_blocked_rcm_permute_matrix_kernel,
    make_llt_blocked_rcm_permute_vector_kernel,
    make_llt_blocked_rcm_solve_inplace_kernel,
    make_llt_blocked_rcm_solve_kernel,
    make_llt_blocked_rcm_symbolic_fill_in_kernel,
)

###
# Module interface
###

__all__ = ["LLTBlockedRCMSolver"]


###
# Module configs
###

wp.set_module_options({"enable_backward": False})


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
) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
    """Build reusable symbolic data for one matrix block."""
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
    return tuple(permutation), tuple(inverse), tuple(tile_pattern), tuple(tile_traversal)


def _fixed_symbolic_data(
    dimensions: Sequence[int],
    adjacency_blocks: Sequence[Sequence[Sequence[int]]],
    block_size: int,
) -> tuple[list[int], list[int], list[int], list[int]]:
    """Build packed permutations, filled tile patterns, and ordered solve work."""
    if len(adjacency_blocks) != len(dimensions):
        raise ValueError("symbolic_adjacency must contain one graph per matrix block.")

    packed_permutation = []
    packed_inverse = []
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
        permutation, inverse, tile_pattern, tile_traversal = block_data
        packed_permutation.extend(permutation)
        packed_inverse.extend(inverse)
        packed_tile_pattern.extend(tile_pattern)
        packed_tile_traversal.extend(tile_traversal)

    return packed_permutation, packed_inverse, packed_tile_pattern, packed_tile_traversal


class LLTBlockedRCMSolver(DirectSolver[wp.float32, wp.int32]):
    """RCM-reordered, semi-sparse Blocked LLT (Cholesky) solver.

    Same public API as :class:`LLTBlockedSolver`.
    Internally:

    1. ``compute(A)`` / ``_factorize_impl``:

       a. Runs batched GPU RCM to compute per-block permutations
          ``P_i`` (concatenated in ``self._P``) in a single set of launches.
       b. Builds ``inv_P`` from ``P``.
       c. Permutes ``A -> A_hat`` (``A_hat_i = P_i A_i P_i^T``).
       d. Builds a tile-level sparsity mask for each block by thresholding
          ``|A_hat_i|`` and then inflates it by a classical block symbolic
          Cholesky fill-in step. Both steps run on the GPU with fixed launch
          shapes and are CUDA-graph capturable.
       e. Numerically factorizes ``A_hat = L L^T`` with the tile-pattern-aware
          kernel that skips guaranteed-zero tiles.

    2. ``solve(b, x)`` / ``_solve_impl``:

       a. Permutes ``b -> b_hat``.
       b. Runs the tile-pattern-aware blocked forward/backward solve.
       c. Un-permutes ``x_hat -> x``.

    All launches use fixed dimensions known at ``finalize()`` time, so the
    entire ``compute`` and ``solve`` flow can be captured into a CUDA graph
    by the caller (same pattern as :class:`LLTBlockedSolver`).
    """

    def __init__(
        self,
        operator: DenseLinearOperatorData[wp.float32, wp.int32] | None = None,
        block_size: int = 32,
        solve_block_dim: int = 256,
        factorize_block_dim: int = 128,
        atol: float | None = None,
        rtol: float | None = None,
        ftol: float | None = None,
        # Reordering options
        # Threshold below which ``|A[i,j]|`` is treated as a non-edge by the
        # RCM adjacency scan and by the tile-pattern builder.
        reorder_tol: float = 0.0,
        # Optional approximate traversal cap. None completes every component.
        rcm_max_bfs_iters: int | None = None,
        reuse_permutation: bool = True,
        parallel_factorization: bool = False,
        symbolic_adjacency: Sequence[Sequence[Sequence[int]]] | None = None,
        dtype: FloatType = wp.float32,
        device: wp.DeviceLike | None = None,
        **kwargs: dict[str, Any],
    ):
        """
        Args:
            operator: optional operator; if provided, :meth:`finalize` is called during init.
            block_size: tile block size passed to the kernel factories.
            solve_block_dim: thread-block size for solve kernels.
            factorize_block_dim: thread-block size for factorize kernel.
            reorder_tol: threshold below which an off-diagonal entry is
                treated as a non-edge by the RCM adjacency scan and by the
                tile-pattern builder.
            rcm_max_bfs_iters: optional BFS step cap. By default every connected
                component is traversed completely.
            reuse_permutation: whether to compute RCM once and reuse that
                permutation for later numeric factorizations. The numeric tile
                pattern is still rebuilt each time. Defaults to ``True``.
            parallel_factorization: whether to solve off-diagonal tiles of
                each Cholesky panel in parallel. Defaults to ``False``.
            symbolic_adjacency: optional fixed structural adjacency for every
                matrix block. When provided, ordering and symbolic Cholesky
                fill are computed once during allocation and reused by every
                numeric factorization.
        """
        # The underlying kernels (factorize / solve / permute / tile-pattern)
        # are hard-coded to wp.float32, so reject any other dtype up front
        # instead of failing later at kernel launch with a shape/type error.
        if dtype != wp.float32:
            raise NotImplementedError("LLTBlockedRCMSolver currently supports only wp.float32.")

        # LLT-specific internal data
        self._L: wp.array[dtype] | None = None
        self._y: wp.array[dtype] | None = None
        # Reordering + semi-sparse state
        self._A_hat: wp.array[dtype] | None = None
        self._x_hat: wp.array[dtype] | None = None
        self._P: wp.array[wp.int32] | None = None
        self._inv_P: wp.array[wp.int32] | None = None
        self._tile_pattern: wp.array[wp.int32] | None = None
        self._tile_traversal: wp.array[wp.int32] | None = None
        self._tpo: wp.array[wp.int32] | None = None
        # Batched-RCM scratch (owned here so the recorded launches in
        # ``_reorder_callback`` never reference buffers that outlive our
        # dict). Allocated in ``_allocate_impl`` alongside the other solver
        # buffers, and reset in ``_reset_impl``.
        self._rcm_scratch: dict | None = None
        self._max_dim: int = 0
        # Batched-RCM launch callback: closure that replays a recorded launch
        # set over all blocks. Rebound whenever the caller-owned ``A`` buffer
        # pointer changes.
        self._reorder_callback = None
        self._reorder_attached_to: wp.array[dtype] | None = None
        self._permutation_initialized = False
        self._permutation_initialized_during_capture = False
        self._permutation_capture_id: int | None = None

        # Cache the fixed block/tile dimensions
        self._block_size: int = block_size
        self._solve_block_dim: int = solve_block_dim
        self._factorize_block_dim: int = factorize_block_dim

        # Reordering options
        self._reorder_tol: float = reorder_tol
        self._rcm_max_bfs_iters = rcm_max_bfs_iters
        self._reuse_permutation = reuse_permutation
        self._parallel_factorization = parallel_factorization
        self._symbolic_adjacency = symbolic_adjacency

        # Build kernels (cached by block_size / max_dim at allocate time).
        self._factorize_kernel = make_llt_blocked_rcm_factorize_kernel(block_size)
        self._parallel_factorize_kernels = make_llt_blocked_rcm_parallel_factorize_kernels(block_size)
        compact_traversal = symbolic_adjacency is not None
        self._solve_kernel = make_llt_blocked_rcm_solve_kernel(block_size, compact_traversal)
        self._solve_inplace_kernel = make_llt_blocked_rcm_solve_inplace_kernel(block_size, compact_traversal)
        # Auxiliary kernels resolved in _allocate_impl once we know max_dim.
        self._permute_vector_kernel = None
        self._fused_permute_and_tp_kernel = None
        self._permute_matrix_kernel = None
        self._symbolic_fill_in_kernel = None

        # Initialize base class members
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
    def inv_P(self) -> wp.array:
        """Concatenated per-block inverse RCM permutation (wp.int32[total_vec_size])."""
        if self._inv_P is None:
            raise ValueError("Inverse permutation array has not been allocated!")
        return self._inv_P

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
        self._permutation_initialized = False
        self._permutation_initialized_during_capture = False
        self._permutation_capture_id = None
        self._reorder_callback = None
        self._reorder_attached_to = None

        # Resolve auxiliary kernels now that max_dim is known.
        self._permute_vector_kernel = make_llt_blocked_rcm_permute_vector_kernel(self._max_dim)
        self._fused_permute_and_tp_kernel = make_llt_blocked_rcm_fused_permute_and_tp_kernel(
            self._block_size, self._max_dim
        )
        self._permute_matrix_kernel = make_llt_blocked_rcm_permute_matrix_kernel(self._max_dim)
        max_n_tiles = (self._max_dim + self._block_size - 1) // self._block_size
        self._symbolic_fill_in_kernel = make_llt_blocked_rcm_symbolic_fill_in_kernel(max_n_tiles)

        # Per-block tile-pattern layout: n_tiles_i^2 entries per block.
        # Compute all offsets at once from the cached host dimensions.
        dims = list(info.dimensions)
        bs = self._block_size
        dims_np = np.asarray(dims, dtype=np.int64)
        tile_counts = (dims_np + bs - 1) // bs
        tp_offsets = np.empty(len(dims) + 1, dtype=np.int64)
        tp_offsets[0] = 0
        np.cumsum(tile_counts * tile_counts, out=tp_offsets[1:])
        total_tp_size = int(tp_offsets[-1])

        fixed_symbolic_data = None
        if self._symbolic_adjacency is not None:
            fixed_symbolic_data = _fixed_symbolic_data(dims, self._symbolic_adjacency, self._block_size)

        with wp.ScopedDevice(self._device):
            # Factorization + intermediate buffers.
            self._L = wp.zeros(shape=(info.total_mat_size,), dtype=self._dtype)
            self._y = wp.zeros(shape=(info.total_vec_size,), dtype=self._dtype)

            # Reordering scratch.
            self._A_hat = wp.zeros(shape=(info.total_mat_size,), dtype=self._dtype)
            self._x_hat = wp.zeros(shape=(info.total_vec_size,), dtype=self._dtype)

            # Permutations (indexed by vio, length dim per block).
            self._P = wp.zeros(shape=(info.total_vec_size,), dtype=wp.int32)
            self._inv_P = wp.zeros(shape=(info.total_vec_size,), dtype=wp.int32)

            # Tile-pattern flat storage + offsets.
            self._tile_pattern = wp.zeros(shape=(total_tp_size,), dtype=wp.int32)
            traversal_size = len(fixed_symbolic_data[3]) if fixed_symbolic_data is not None else 0
            self._tile_traversal = wp.empty(shape=(max(1, traversal_size),), dtype=wp.int32)
            self._tpo = to_warp_int32_array(tp_offsets[:-1])

            # Batched-RCM scratch. Owning these here matches how the other
            # linalg solvers hold their buffers and keeps the recorded-launch
            # inputs alive for as long as the solver is alive.
            self._rcm_scratch = _rcm_batch.allocate_rcm_batch_scratch(
                total_vec=info.total_vec_size,
                num_blocks=info.num_blocks,
                device=self._device,
            )

            if fixed_symbolic_data is not None:
                permutation, inverse, tile_pattern, tile_traversal = fixed_symbolic_data
                self._P.assign(permutation)
                self._inv_P.assign(inverse)
                self._tile_pattern.assign(tile_pattern)
                if tile_traversal:
                    self._tile_traversal.assign(tile_traversal)

        # The batched-RCM launch callback (``self._reorder_callback``) is
        # (re)built lazily in ``_ensure_reorder_launches_bound`` the first
        # time a concrete A buffer arrives, and rebound only if its device
        # pointer changes.

        self._has_factors = False

    @override
    def _reset_impl(self) -> None:
        self._L.zero_()
        self._y.zero_()
        self._A_hat.zero_()
        self._x_hat.zero_()
        if self._symbolic_adjacency is None:
            self._P.zero_()
            self._rcm_scratch["permutation_valid"].zero_()
            self._rcm_scratch["permutation_dim"].zero_()
            self._inv_P.zero_()
            self._permutation_initialized = False
            self._permutation_initialized_during_capture = False
            self._permutation_capture_id = None
            self._tile_pattern.zero_()
        self._has_factors = False

    def _ensure_reorder_launches_bound(self, A: wp.array[Any]) -> None:
        """(Re)build the batched-RCM launch callback bound to the current A buffer.

        The callback captures ``wp.array`` views of ``A`` that must stay
        alive for as long as the recorded launches are reused, so we only
        rebind when the caller hands us a different ``A``.
        """
        if self._reorder_attached_to is A:
            return

        info = self._operator.info
        with wp.ScopedDevice(self._device):
            self._reorder_callback = _rcm_batch.create_rcm_batch_launch(
                A_flat=A,
                perm_flat=self._P,
                dims=info.dim,
                mio=info.mio,
                vio=info.vio,
                scratch=self._rcm_scratch,
                num_blocks=info.num_blocks,
                max_dim=int(info.max_dimension),
                tol=self._reorder_tol,
                max_bfs_iters=self._rcm_max_bfs_iters,
                use_cuda_graph=False,
                reuse_permutation=self._reuse_permutation,
                device=self._device,
            )
        self._reorder_attached_to = A

    @override
    def _factorize_impl(self, A: wp.array[Any]) -> None:
        info = self._operator.info
        num_blocks = info.num_blocks

        if self._symbolic_adjacency is not None:
            llt_blocked_rcm_permute_matrix(
                kernel=self._permute_matrix_kernel,
                dim=info.dim,
                mio=info.mio,
                vio=info.vio,
                P=self._P,
                A=A,
                A_hat=self._A_hat,
                num_blocks=num_blocks,
                max_dim=self._max_dim,
                device=self._device,
            )
        else:
            self._factorize_symbolic_and_permute(A)

        # Numeric factorization with tile-pattern skips.
        if self._parallel_factorization:
            llt_blocked_rcm_factorize_parallel(
                kernels=self._parallel_factorize_kernels,
                dim=info.dim,
                mio=info.mio,
                tpo=self._tpo,
                A=self._A_hat,
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
                A=self._A_hat,
                tile_pattern=self._tile_pattern,
                L=self._L,
                num_blocks=num_blocks,
                block_dim=self._factorize_block_dim,
                device=self._device,
            )

    def _factorize_symbolic_and_permute(self, A: wp.array[Any]) -> None:
        """Refresh numeric-derived symbolic data for the compatibility path."""
        info = self._operator.info
        num_blocks = info.num_blocks
        is_capturing = bool(self._device.is_capturing)
        capture_id = None
        if is_capturing and self._device.is_cuda:
            capture = self._device.captures.get(self._device.stream)
            if capture is not None:
                capture_id = capture.capture_id

        # Recording RCM launches mutates host cache state before a captured
        # graph executes. Validate that one-time transition against the device
        # flags after capture, so an unlaunched or aborted graph cannot leave
        # the replacement permutation buffers marked as initialized.
        if self._reuse_permutation and self._permutation_initialized_during_capture:
            if is_capturing and capture_id != self._permutation_capture_id:
                self._permutation_initialized = False
                self._permutation_initialized_during_capture = False
                self._permutation_capture_id = None
            elif not is_capturing:
                valid = self._rcm_scratch["permutation_valid"].numpy()
                valid_dims = self._rcm_scratch["permutation_dim"].numpy()
                self._permutation_initialized = all(
                    int(is_valid) != 0 and int(valid_dim) == expected_dim
                    for is_valid, valid_dim, expected_dim in zip(valid, valid_dims, info.dimensions, strict=True)
                )
                self._permutation_initialized_during_capture = False
                self._permutation_capture_id = None

        # Bind / rebind views to the current A buffer.
        self._ensure_reorder_launches_bound(A)

        # Compute per-block P via the batched RCM callback. The callback is a
        # set of recorded Warp launches and is safe to replay under CUDA graph
        # capture initiated by the caller.
        if not self._reuse_permutation or not self._permutation_initialized:
            self._reorder_callback()
            # Later calls are ordered behind these launches on the same stream,
            # including calls recorded later in the same CUDA graph capture.
            self._permutation_initialized = True
            self._permutation_initialized_during_capture = self._reuse_permutation and is_capturing
            self._permutation_capture_id = capture_id if self._permutation_initialized_during_capture else None

        # 2. Fused: build inv_P, permute A -> A_hat, and reduce |A_hat| into the
        #    raw tile pattern in a single launch. Each thread writes only its
        #    own (r, c) entry, so there is no data race on A_hat; tile-pattern
        #    bits are OR'd via atomic_max.
        self._tile_pattern.zero_()
        llt_blocked_rcm_fused_permute_and_tp(
            kernel=self._fused_permute_and_tp_kernel,
            dim=info.dim,
            mio=info.mio,
            vio=info.vio,
            tpo=self._tpo,
            tol=self._reorder_tol,
            P=self._P,
            A=A,
            A_hat=self._A_hat,
            inv_P=self._inv_P,
            tile_pattern=self._tile_pattern,
            num_blocks=num_blocks,
            max_dim=self._max_dim,
            device=self._device,
        )

        # 3. Inflate the tile pattern by block symbolic Cholesky fill-in.
        llt_blocked_rcm_symbolic_fill_in(
            kernel=self._symbolic_fill_in_kernel,
            dim=info.dim,
            tpo=self._tpo,
            block_size=self._block_size,
            tile_pattern=self._tile_pattern,
            num_blocks=num_blocks,
            device=self._device,
        )

    @override
    def _reconstruct_impl(self, A: wp.array[Any]) -> None:
        raise NotImplementedError("LLT matrix reconstruction is not yet implemented.")

    @override
    def _solve_impl(self, b: wp.array[Any], x: wp.array[Any]) -> None:
        info = self._operator.info
        num_blocks = info.num_blocks

        # Solve L L^T x_hat = P b and scatter x_hat -> x.
        llt_blocked_rcm_solve(
            kernel=self._solve_kernel,
            dim=info.dim,
            mio=info.mio,
            vio=info.vio,
            tpo=self._tpo,
            P=self._P,
            L=self._L,
            tile_pattern=self._tile_pattern,
            tile_traversal=self._tile_traversal,
            b=b,
            y=self._y,
            x_hat=self._x_hat,
            x=x,
            num_blocks=num_blocks,
            block_dim=self._solve_block_dim,
            device=self._device,
        )

    @override
    def _solve_inplace_impl(self, x: wp.array[Any]) -> None:
        info = self._operator.info
        num_blocks = info.num_blocks

        # Permute x -> x_hat (x is the RHS here).
        llt_blocked_rcm_permute_vector(
            kernel=self._permute_vector_kernel,
            dim=info.dim,
            vio=info.vio,
            P=self._P,
            src=x,
            dst=self._x_hat,
            num_blocks=num_blocks,
            max_dim=self._max_dim,
            device=self._device,
        )

        # In-place solve on x_hat (x_hat starts as the permuted RHS; y is scratch).
        llt_blocked_rcm_solve_inplace(
            kernel=self._solve_inplace_kernel,
            dim=info.dim,
            mio=info.mio,
            vio=info.vio,
            tpo=self._tpo,
            L=self._L,
            tile_pattern=self._tile_pattern,
            tile_traversal=self._tile_traversal,
            y=self._y,
            x=self._x_hat,
            num_blocks=num_blocks,
            block_dim=self._solve_block_dim,
            device=self._device,
        )

        # Un-permute x_hat -> x.
        llt_blocked_rcm_permute_vector(
            kernel=self._permute_vector_kernel,
            dim=info.dim,
            vio=info.vio,
            P=self._inv_P,
            src=self._x_hat,
            dst=x,
            num_blocks=num_blocks,
            max_dim=self._max_dim,
            device=self._device,
        )
