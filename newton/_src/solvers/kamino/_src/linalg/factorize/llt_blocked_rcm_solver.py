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

    solver = LLTBlockedRCMSolver(operator=operator, block_size=32, dtype=float32)
    solver.compute(A)  # factorizes; reordering is internal
    solver.solve(b, x)  # or: solver.solve_inplace(x)

The reordering ``P`` and its inverse ``inv_P`` are stored on the solver next
to the factorization buffer ``L``. They are exposed as read-only properties
for debugging/introspection.
"""

from typing import Any

import warp as wp

from ...core.types import FloatType, float32, int32, override
from ..core import DenseLinearOperatorData, DenseSquareMultiLinearInfo
from ..linear import DirectSolver
from . import rcm_batch as _rcm_batch
from .llt_blocked_rcm import (
    llt_blocked_rcm_factorize,
    llt_blocked_rcm_fused_permute_and_tp,
    llt_blocked_rcm_permute_vector,
    llt_blocked_rcm_solve,
    llt_blocked_rcm_solve_inplace,
    llt_blocked_rcm_symbolic_fill_in,
    make_llt_blocked_rcm_factorize_kernel,
    make_llt_blocked_rcm_fused_permute_and_tp_kernel,
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


class LLTBlockedRCMSolver(DirectSolver):
    """RCM-reordered, semi-sparse Blocked LLT (Cholesky) solver.

    Same public API as :class:`LLTBlockedSolver`.
    Internally:

    1. ``compute(A)`` / ``_factorize_impl``:

       a. Runs GPU-native batched RCM to compute per-block permutations
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
        operator: DenseLinearOperatorData | None = None,
        block_size: int = 32,
        solve_block_dim: int = 128,
        factorize_block_dim: int = 128,
        atol: float | None = None,
        rtol: float | None = None,
        ftol: float | None = None,
        # Reordering options
        # Threshold below which ``|A[i,j]|`` is treated as a non-edge by the
        # RCM adjacency scan and by the tile-pattern builder.
        reorder_tol: float = 0.0,
        # Cap on BFS steps per block. None => auto (``2*ceil(sqrt(n)) + 4``).
        rcm_max_bfs_iters: int | None = None,
        dtype: FloatType = float32,
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
            rcm_max_bfs_iters: BFS depth cap for the batched RCM pass.
        """
        # The underlying kernels (factorize / solve / permute / tile-pattern)
        # are hard-coded to float32, so reject any other dtype up front
        # instead of failing later at kernel launch with a shape/type error.
        if dtype != float32:
            raise NotImplementedError("LLTBlockedRCMSolver currently supports only float32.")

        # LLT-specific internal data
        self._L: wp.array | None = None
        self._y: wp.array | None = None
        # Reordering + semi-sparse state
        self._A_hat: wp.array | None = None
        self._b_hat: wp.array | None = None
        self._x_hat: wp.array | None = None
        self._P: wp.array | None = None
        self._inv_P: wp.array | None = None
        self._tile_pattern: wp.array | None = None
        self._tpo: wp.array | None = None
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
        self._reorder_attached_to: wp.array | None = None

        # Cache the fixed block/tile dimensions
        self._block_size: int = block_size
        self._solve_block_dim: int = solve_block_dim
        self._factorize_block_dim: int = factorize_block_dim

        # Reordering options
        self._reorder_tol: float = reorder_tol
        self._rcm_max_bfs_iters = rcm_max_bfs_iters

        # Build kernels (cached by block_size / max_dim at allocate time).
        self._factorize_kernel = make_llt_blocked_rcm_factorize_kernel(block_size)
        self._solve_kernel = make_llt_blocked_rcm_solve_kernel(block_size)
        self._solve_inplace_kernel = make_llt_blocked_rcm_solve_inplace_kernel(block_size)
        # Auxiliary kernels resolved in _allocate_impl once we know max_dim.
        self._permute_vector_kernel = None
        self._fused_permute_and_tp_kernel = None
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
        """Concatenated per-block RCM permutation (int32[total_vec_size])."""
        if self._P is None:
            raise ValueError("Permutation array has not been allocated!")
        return self._P

    @property
    def inv_P(self) -> wp.array:
        """Concatenated per-block inverse RCM permutation (int32[total_vec_size])."""
        if self._inv_P is None:
            raise ValueError("Inverse permutation array has not been allocated!")
        return self._inv_P

    @property
    def tile_pattern(self) -> wp.array:
        """Concatenated per-block tile-sparsity mask (int32, lower-tri inflated by fill-in)."""
        if self._tile_pattern is None:
            raise ValueError("Tile pattern array has not been allocated!")
        return self._tile_pattern

    ###
    # Implementation
    ###

    @override
    def _allocate_impl(self, A: DenseLinearOperatorData, **kwargs: dict[str, Any]) -> None:
        if A.info is None:
            raise ValueError("The provided operator does not have any associated info!")
        if not isinstance(A.info, DenseSquareMultiLinearInfo):
            raise ValueError("LLT factorization requires a square matrix.")

        info = self._operator.info
        self._max_dim = int(info.max_dimension)

        # Resolve auxiliary kernels now that max_dim is known.
        self._permute_vector_kernel = make_llt_blocked_rcm_permute_vector_kernel(self._max_dim)
        self._fused_permute_and_tp_kernel = make_llt_blocked_rcm_fused_permute_and_tp_kernel(
            self._block_size, self._max_dim
        )
        max_n_tiles = (self._max_dim + self._block_size - 1) // self._block_size
        self._symbolic_fill_in_kernel = make_llt_blocked_rcm_symbolic_fill_in_kernel(max_n_tiles)

        # Per-block tile-pattern layout: n_tiles_i^2 entries per block.
        # Computed on host once from info.dimensions (a cheap list).
        dims = list(info.dimensions)
        bs = self._block_size
        tp_sizes = [((d + bs - 1) // bs) ** 2 for d in dims]
        tp_offsets = [0]
        for s in tp_sizes:
            tp_offsets.append(tp_offsets[-1] + s)
        total_tp_size = tp_offsets[-1]

        with wp.ScopedDevice(self._device):
            # Factorization + intermediate buffers.
            self._L = wp.zeros(shape=(info.total_mat_size,), dtype=self._dtype)
            self._y = wp.zeros(shape=(info.total_vec_size,), dtype=self._dtype)

            # Reordering scratch.
            self._A_hat = wp.zeros(shape=(info.total_mat_size,), dtype=self._dtype)
            self._b_hat = wp.zeros(shape=(info.total_vec_size,), dtype=self._dtype)
            self._x_hat = wp.zeros(shape=(info.total_vec_size,), dtype=self._dtype)

            # Permutations (indexed by vio, length dim per block).
            self._P = wp.zeros(shape=(info.total_vec_size,), dtype=int32)
            self._inv_P = wp.zeros(shape=(info.total_vec_size,), dtype=int32)

            # Tile-pattern flat storage + offsets.
            self._tile_pattern = wp.zeros(shape=(total_tp_size,), dtype=int32)
            self._tpo = wp.array(tp_offsets[:-1], dtype=int32)

            # Batched-RCM scratch. Owning these here matches how the other
            # linalg solvers hold their buffers and keeps the recorded-launch
            # inputs alive for as long as the solver is alive.
            self._rcm_scratch = _rcm_batch.allocate_rcm_batch_scratch(
                total_vec=info.total_vec_size,
                num_blocks=info.num_blocks,
                device=self._device,
            )

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
        self._b_hat.zero_()
        self._x_hat.zero_()
        self._P.zero_()
        self._inv_P.zero_()
        self._tile_pattern.zero_()
        self._has_factors = False

    def _ensure_reorder_launches_bound(self, A: wp.array) -> None:
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
                device=self._device,
            )
        self._reorder_attached_to = A

    @override
    def _factorize_impl(self, A: wp.array) -> None:
        info = self._operator.info
        num_blocks = info.num_blocks

        # Bind / rebind views to the current A buffer.
        self._ensure_reorder_launches_bound(A)

        # 1. Compute per-block P via the batched RCM callback. The callback
        #    is a set of recorded Warp launches and is safe to replay under
        #    CUDA graph capture initiated by the caller.
        self._reorder_callback()

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

        # 4. Numeric factorization with tile-pattern skips.
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

    @override
    def _reconstruct_impl(self, A: wp.array) -> None:
        raise NotImplementedError("LLT matrix reconstruction is not yet implemented.")

    @override
    def _solve_impl(self, b: wp.array, x: wp.array) -> None:
        info = self._operator.info
        num_blocks = info.num_blocks

        # Permute b -> b_hat.
        llt_blocked_rcm_permute_vector(
            kernel=self._permute_vector_kernel,
            dim=info.dim,
            vio=info.vio,
            P=self._P,
            src=b,
            dst=self._b_hat,
            num_blocks=num_blocks,
            max_dim=self._max_dim,
            device=self._device,
        )

        # Solve L L^T x_hat = b_hat.
        llt_blocked_rcm_solve(
            kernel=self._solve_kernel,
            dim=info.dim,
            mio=info.mio,
            vio=info.vio,
            tpo=self._tpo,
            L=self._L,
            tile_pattern=self._tile_pattern,
            b=self._b_hat,
            y=self._y,
            x=self._x_hat,
            num_blocks=num_blocks,
            block_dim=self._solve_block_dim,
            device=self._device,
        )

        # Un-permute: x[P[r]] = x_hat[r]  =>  x[r] = x_hat[inv_P[r]].
        # Our permute_vector kernel computes ``dst[r] = src[P[r]]``, so using
        # inv_P as P and (x_hat, x) as (src, dst) yields the desired inverse
        # permutation on x.
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

    @override
    def _solve_inplace_impl(self, x: wp.array) -> None:
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
