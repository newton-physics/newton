# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""
KAMINO: Conjugate gradient and conjugate residual solvers
"""

from __future__ import annotations

import functools
import math
from collections.abc import Callable
from typing import Any

import warp as wp

from . import blas
from .core import DenseLinearOperatorData
from .sparse_matrix import BlockSparseMatrices
from .sparse_operator import BlockSparseLinearOperators

# No need to auto-generate adjoint code for linear solvers
wp.set_module_options({"enable_backward": False})

# based on the warp.optim.linear implementation


__all__ = [
    "BatchedLinearOperator",
    "CGSolver",
    "CRSolver",
    "make_jacobi_preconditioner",
]


class BatchedLinearOperator:
    """Linear operator for batched matrix-vector products.

    Supports dense, diagonal, and block-sparse matrices.
    Use class methods to create instances.
    """

    def __init__(
        self,
        gemv_fn: Callable,
        n_worlds: int,
        max_dim: int,
        active_dims: wp.array,
        device: wp.Device,
        dtype: type,
        matvec_fn: Callable | None = None,
    ):
        self._gemv_fn = gemv_fn
        self.n_worlds = n_worlds
        self.max_dim = max_dim
        self.active_dims = active_dims
        self.device = device
        self.dtype = dtype
        self._matvec_fn = matvec_fn

    @classmethod
    def from_dense(cls, operator: DenseLinearOperatorData) -> BatchedLinearOperator:
        """Create operator from dense matrix data."""
        info = operator.info
        n_worlds = info.num_blocks
        max_dim = info.max_dimension
        A_mat = operator.mat.reshape((n_worlds, max_dim * max_dim))
        active_dims = info.dim

        def gemv_fn(x, y, world_active, alpha, beta):
            blas.dense_gemv(A_mat, x, y, active_dims, world_active, alpha, beta, max_dim)

        return cls(gemv_fn, n_worlds, max_dim, active_dims, info.device, info.dtype)

    @classmethod
    def from_diagonal(cls, D: wp.array2d, active_dims: wp.array) -> BatchedLinearOperator:
        """Create operator from diagonal matrix."""
        n_worlds, max_dim = D.shape

        def gemv_fn(x, y, world_active, alpha, beta):
            blas.diag_gemv(D, x, y, active_dims, world_active, alpha, beta)

        return cls(gemv_fn, n_worlds, max_dim, active_dims, D.device, D.dtype)

    @classmethod
    def from_block_sparse(cls, A: BlockSparseMatrices, active_dims: wp.array) -> BatchedLinearOperator:
        """Create operator from block-sparse matrix.

        Requires all matrices to have the same max dimensions so that 2D arrays
        can be reshaped to 1D for the sparse gemv kernel.

        Args:
            A: Block-sparse matrices container.
            active_dims: 1D int array with active row dimension per matrix.
        """
        max_rows, max_cols = A.max_of_max_dims
        n_worlds = A.num_matrices

        def gemv_fn(x, y, world_active, alpha, beta):
            # Reshape 2D arrays to 1D for sparse gemv, then back
            x_flat = x.reshape((n_worlds * max_cols,))
            y_flat = y.reshape((n_worlds * max_rows,))
            blas.block_sparse_gemv(A, x_flat, y_flat, alpha, beta, world_active)

        def matvec_fn(x, y, world_active):
            # Reshape 2D arrays to 1D for sparse matvec, then back
            x_flat = x.reshape((n_worlds * max_cols,))
            y_flat = y.reshape((n_worlds * max_rows,))
            blas.block_sparse_matvec(A, x_flat, y_flat, world_active)

        dtype = A.nzb_dtype.dtype if A.nzb_dtype is not None else None
        return cls(gemv_fn, n_worlds, max_rows, active_dims, A.device, dtype, matvec_fn=matvec_fn)

    @classmethod
    def from_block_sparse_operator(cls, A: BlockSparseLinearOperators) -> BatchedLinearOperator:
        """Create operator from block-sparse operator.

        Requires all matrices to have the same max dimensions so that 2D arrays
        can be reshaped to 1D for the sparse gemv kernel.

        Args:
            A: Block-sparse matrices operator.
            active_dims: 1D int array with active row dimension per matrix.
        """
        max_rows, max_cols = A.max_of_max_dims
        n_worlds = A.num_matrices

        def gemv_fn(x, y, world_active, alpha, beta):
            x_flat = x.reshape((n_worlds * max_cols,))
            y_flat = y.reshape((n_worlds * max_rows,))
            A.gemv(x_flat, y_flat, world_active, alpha, beta)

        def matvec_fn(x, y, world_active):
            x_flat = x.reshape((n_worlds * max_cols,))
            y_flat = y.reshape((n_worlds * max_rows,))
            A.matvec(x_flat, y_flat, world_active)

        return cls(gemv_fn, n_worlds, max_rows, A.active_cols, A.device, A.dtype, matvec_fn=matvec_fn)

    def gemv(self, x: wp.array2d, y: wp.array2d, world_active: wp.array, alpha: float, beta: float):
        """Compute y = alpha * A @ x + beta * y."""
        self._gemv_fn(x, y, world_active, alpha, beta)

    def matvec(self, x: wp.array2d, y: wp.array2d, world_active: wp.array):
        if self._matvec_fn is not None:
            return self._matvec_fn(x, y, world_active)
        return self._gemv_fn(x, y, world_active, 1.0, 0.0)


# Implementations
# ---------------


@functools.cache
def make_termination_kernel(n_worlds):
    @wp.kernel
    def check_termination(
        maxiter: wp.array(dtype=int),
        cycle_size: int,
        r_norm_sq: wp.array(dtype=Any),
        atol_sq: wp.array(dtype=Any),
        world_active: wp.array(dtype=wp.int32),
        cur_iter: wp.array(dtype=int),
        world_condition: wp.array(dtype=wp.int32),
        batch_condition: wp.array(dtype=wp.int32),
    ):
        thread = wp.tid()
        active = wp.tile_load(world_active, (n_worlds,))
        condition = wp.tile_load(world_condition, (n_worlds,))
        world_stepped = wp.tile_map(wp.mul, active, condition)
        iter = world_stepped * cycle_size + wp.tile_load(cur_iter, (n_worlds,))

        wp.tile_store(cur_iter, iter)
        cont_norm = wp.tile_astype(
            wp.tile_map(lt_mask, wp.tile_load(atol_sq, (n_worlds,)), wp.tile_load(r_norm_sq, (n_worlds,))), wp.int32
        )
        cont_iter = wp.tile_map(lt_mask, iter, wp.tile_load(maxiter, (n_worlds,)))
        cont = wp.tile_map(wp.mul, wp.tile_map(wp.mul, cont_iter, cont_norm), world_stepped)
        wp.tile_store(world_condition, cont)
        batch_cont = wp.where(wp.tile_sum(cont)[0] > 0, 1, 0)
        if thread == 0:
            batch_condition[0] = batch_cont

    return check_termination


@wp.kernel
def _cg_kernel_1(
    tol: wp.array(dtype=Any),
    resid: wp.array(dtype=Any),
    rz_old: wp.array(dtype=Any),
    p_Ap: wp.array(dtype=Any),
    p: wp.array2d(dtype=Any),
    Ap: wp.array2d(dtype=Any),
    x: wp.array2d(dtype=Any),
    r: wp.array2d(dtype=Any),
):
    e, i = wp.tid()

    alpha = wp.where(resid[e] > tol[e] and p_Ap[e] > 0.0, rz_old[e] / p_Ap[e], rz_old.dtype(0.0))

    x[e, i] = x[e, i] + alpha * p[e, i]
    r[e, i] = r[e, i] - alpha * Ap[e, i]


@wp.kernel
def _cg_kernel_2(
    tol: wp.array(dtype=Any),
    resid_new: wp.array(dtype=Any),
    rz_old: wp.array(dtype=Any),
    rz_new: wp.array(dtype=Any),
    z: wp.array2d(dtype=Any),
    p: wp.array2d(dtype=Any),
):
    #    p = r + (rz_new / rz_old) * p;
    e, i = wp.tid()

    cond = resid_new[e] > tol[e]
    beta = wp.where(cond and rz_old[e] > 0.0, rz_new[e] / rz_old[e], rz_old.dtype(0.0))

    p[e, i] = z[e, i] + beta * p[e, i]


@wp.kernel
def _cr_kernel_1(
    tol: wp.array(dtype=Any),
    resid: wp.array(dtype=Any),
    zAz_old: wp.array(dtype=Any),
    y_Ap: wp.array(dtype=Any),
    p: wp.array2d(dtype=Any),
    Ap: wp.array2d(dtype=Any),
    y: wp.array2d(dtype=Any),
    x: wp.array2d(dtype=Any),
    r: wp.array2d(dtype=Any),
    z: wp.array2d(dtype=Any),
):
    e, i = wp.tid()

    alpha = wp.where(resid[e] > tol[e] and y_Ap[e] > 0.0, zAz_old[e] / y_Ap[e], zAz_old.dtype(0.0))

    x[e, i] = x[e, i] + alpha * p[e, i]
    r[e, i] = r[e, i] - alpha * Ap[e, i]
    z[e, i] = z[e, i] - alpha * y[e, i]


@wp.kernel
def _cr_kernel_2(
    tol: wp.array(dtype=Any),
    resid: wp.array(dtype=Any),
    zAz_old: wp.array(dtype=Any),
    zAz_new: wp.array(dtype=Any),
    z: wp.array2d(dtype=Any),
    Az: wp.array2d(dtype=Any),
    p: wp.array2d(dtype=Any),
    Ap: wp.array2d(dtype=Any),
):
    #    p = r + (rz_new / rz_old) * p;
    e, i = wp.tid()

    beta = wp.where(resid[e] > tol[e] and zAz_old[e] > 0.0, zAz_new[e] / zAz_old[e], zAz_old.dtype(0.0))

    p[e, i] = z[e, i] + beta * p[e, i]
    Ap[e, i] = Az[e, i] + beta * Ap[e, i]


def _run_capturable_loop(
    do_cycle: Callable,
    r_norm_sq: wp.array,
    world_active: wp.array(dtype=wp.int32),
    cur_iter: wp.array(dtype=wp.int32),
    conditions: wp.array(dtype=wp.int32),
    maxiter: wp.array(dtype=int),
    atol_sq: wp.array,
    callback: Callable | None,
    use_cuda_graph: bool,
    use_graph_conditionals: bool = True,
    maxiter_host: int | None = None,
    cycle_size: int = 1,
    termination_kernel=None,
):
    device = atol_sq.device

    n_worlds = maxiter.shape[0]
    cur_iter.fill_(-1)
    conditions.fill_(1)

    world_condition, global_condition = conditions[:n_worlds], conditions[n_worlds:]

    update_condition_launch = wp.launch(
        termination_kernel,
        dim=(1, n_worlds),
        block_dim=n_worlds,
        device=device,
        inputs=[maxiter, cycle_size, r_norm_sq, atol_sq, world_active, cur_iter],
        outputs=[world_condition, global_condition],
        record_cmd=True,
    )

    if isinstance(callback, wp.Kernel):
        callback_launch = wp.launch(
            callback, dim=n_worlds, device=device, inputs=[cur_iter, r_norm_sq, atol_sq], record_cmd=True
        )
    else:
        callback_launch = None

    # TODO: consider using a spinlock for fusing kernels
    # update_world_condition_launch.launch()
    # update_global_condition_launch.launch()
    update_condition_launch.launch()

    if callback_launch is not None:
        callback_launch.launch()

    def do_cycle_with_condition():
        # print("Global cond:", global_condition.numpy())
        do_cycle()
        update_condition_launch.launch()
        if callback_launch is not None:
            callback_launch.launch()

    if use_cuda_graph and device.is_cuda and device.is_capturing:
        if use_graph_conditionals:
            wp.capture_while(global_condition, do_cycle_with_condition)
        else:
            for _ in range(0, int(maxiter_host), cycle_size):
                do_cycle_with_condition()
    else:
        for _ in range(0, int(maxiter.numpy().max()), cycle_size):
            do_cycle_with_condition()
            if not global_condition.numpy()[0]:
                # print("Exiting")
                break

    return cur_iter, r_norm_sq, atol_sq


@wp.func
def lt_mask(a: Any, b: Any):
    """Return 1 if a < b, else 0"""
    return wp.where(a < b, type(a)(1), type(a)(0))


@wp.func
def mul_mask(mask: Any, value: Any):
    """Return value if mask is positive, else 0"""
    return wp.where(mask > type(mask)(0), value, type(value)(0))


@wp.func
def less_than_op(i: wp.int32, threshold: wp.int32) -> wp.float32:
    return 1.0 if i < threshold else 0.0


@functools.cache
def make_dot_kernel(tile_size: int, maxdim: int):
    num_tiles = (maxdim + tile_size - 1) // tile_size

    @wp.kernel(enable_backward=False)
    def dot(
        a: wp.array3d(dtype=Any),
        b: wp.array3d(dtype=Any),
        world_size: wp.array(dtype=wp.int32),
        world_active: wp.array(dtype=wp.int32),
        result: wp.array2d(dtype=Any),
    ):
        """Compute the dot products between the trailing-dim arrays in a and b using tiles and pairwise summation."""
        col, world, tid = wp.tid()
        if not world_active[world]:
            return
        n = world_size[world]

        if wp.static(num_tiles > 1):
            ts = wp.tile_zeros((num_tiles,), dtype=a.dtype, storage="shared")

        for tile_id in range(num_tiles):
            o_src = tile_id * tile_size
            if o_src >= n:
                break
            ta = wp.tile_load(a[col, world], shape=tile_size, offset=o_src)
            tb = wp.tile_load(b[col, world], shape=tile_size, offset=o_src)
            prod = wp.tile_map(wp.mul, ta, tb)
            if o_src > n - tile_size:
                mask = wp.tile_map(less_than_op, wp.tile_arange(tile_size, dtype=wp.int32), n - o_src)
                prod = wp.tile_map(mul_mask, mask, prod)
            if wp.static(num_tiles > 1):
                ts[tile_id] = wp.tile_sum(prod)[0]
            else:
                s = wp.tile_sum(prod)[0]
        if wp.static(num_tiles > 1):
            s = wp.tile_sum(ts)[0]
        if tid == 0:
            result[col, world] = s

    return dot


@wp.kernel
def dot_sequential(
    a: wp.array3d(dtype=Any),
    b: wp.array3d(dtype=Any),
    world_size: wp.array(dtype=wp.int32),
    world_active: wp.array(dtype=wp.int32),
    partial_sum: wp.array3d(dtype=Any),
):
    col, world = wp.tid()

    if not world_active[world]:
        return
    n = wp.int32(world_size[world])

    for i in range((n + 1) // 2):
        s = a[col, world, 2 * i] * b[col, world, 2 * i]
        if 2 * i + 1 < n:
            s += a[col, world, 2 * i + 1] * b[col, world, 2 * i + 1]
        partial_sum[col, world, i] = s

    n = (n + 1) // 2

    while n > 1:
        s = a.dtype(0)
        if n & 1:
            s += partial_sum[col, world, n - 1]
        for i in range(n // 2):
            s += partial_sum[col, world, 2 * i] + partial_sum[col, world, 2 * i + 1]
            partial_sum[col, world, i] = s
            s = a.dtype(0)
        n = n // 2


@wp.kernel
def _initialize_tolerance_kernel(
    rtol: wp.array(dtype=Any), atol: wp.array(dtype=Any), b_norm_sq: wp.array(dtype=Any), atol_sq: wp.array(dtype=Any)
):
    world = wp.tid()
    a, r = atol[world], rtol[world]
    atol_sq[world] = wp.max(r * r * b_norm_sq[world], a * a)


@wp.kernel
def make_jacobi_preconditioner(
    A: wp.array2d(dtype=Any), world_dims: wp.array(dtype=wp.int32), diag: wp.array2d(dtype=Any)
):
    world, row = wp.tid()
    world_dim = world_dims[world]
    if row >= world_dim:
        diag[world, row] = 0.0
        return
    el = A[world, row * world_dim + row]
    el_inv = 1.0 / (el + 1e-9)
    diag[world, row] = el_inv


class ConjugateSolver:
    """Base class for conjugate iterative solvers (CG, CR).

    Solves batched linear systems Ax = b for multiple independent worlds in parallel.
    Supports dense, diagonal, and block-sparse matrix operators with optional
    preconditioning.

    Note:
        Temporary arrays are zero-initialized to avoid NaN propagation.

    Args:
        A: Linear operator representing the system matrix.
        active_dims: Active dimension per world. If None, uses A.active_dims.
        world_active: Int32 mask indicating which worlds are active (1) or inactive (0).
        atol: Absolute tolerance for convergence. Scalar or per-world array.
        rtol: Relative tolerance for convergence. Scalar or per-world array.
        maxiter: Maximum iterations per world. If None, defaults to 1.5 * maxdims.
        Mi: Operator applying the inverse preconditioner M^-1, such that Mi @ A has a smaller condition number than A.
        callback: Optional callback kernel invoked each iteration.
        use_cuda_graph: Whether to use CUDA graph capture for the solve loop.
    """

    def __init__(
        self,
        A: BatchedLinearOperator,
        active_dims: wp.array(dtype=Any) | None = None,
        world_active: wp.array(dtype=wp.int32) | None = None,
        atol: float | wp.array(dtype=Any) | None = None,
        rtol: float | wp.array(dtype=Any) | None = None,
        maxiter: wp.array = None,
        Mi: BatchedLinearOperator | None = None,
        callback: Callable | None = None,
        use_cuda_graph: bool = True,
        use_graph_conditionals: bool = True,
    ):
        if not isinstance(A, BatchedLinearOperator):
            raise ValueError("A must be a BatchedLinearOperator")
        if Mi is not None and not isinstance(Mi, BatchedLinearOperator):
            raise ValueError("Mi must be a BatchedLinearOperator or None")

        self.scalar_type = wp.types.type_scalar_type(A.dtype)
        self.n_worlds = A.n_worlds
        self.maxdims = A.max_dim
        self.A = A
        self.Mi = Mi
        self.device = A.device
        self.active_dims = active_dims if active_dims is not None else A.active_dims
        self.use_graph_conditionals = use_graph_conditionals

        self.world_active = world_active
        self.atol = atol
        self.rtol = rtol
        self.maxiter = maxiter

        self.callback = callback
        self.use_cuda_graph = use_cuda_graph

        self.dot_tile_size = min(2048, 2 ** math.ceil(math.log(self.maxdims, 2)))
        self.tiled_dot_kernel = make_dot_kernel(self.dot_tile_size, self.maxdims)
        self._allocate()

    def _allocate(self):
        self.residual = wp.empty((self.n_worlds), dtype=self.scalar_type, device=self.device)

        if self.maxiter is None:
            maxiter = int(1.5 * self.maxdims)
            self.maxiter = wp.full(self.n_worlds, maxiter, dtype=int, device=self.device)
            self.maxiter_host = maxiter
        else:
            self.maxiter_host = int(max(self.maxiter.numpy()))

        # TODO: non-tiled variant for CPU
        if self.tiled_dot_product:
            self.dot_product = wp.zeros((2, self.n_worlds), dtype=self.scalar_type, device=self.device)
        else:
            self.dot_partial_sums = wp.zeros((2, self.n_worlds, (self.maxdims + 1) // 2), device=self.device)
            self.dot_product = self.dot_partial_sums[:, :, 0]

        atol_val = self.atol if isinstance(self.atol, float) else 1e-8
        rtol_val = self.rtol if isinstance(self.rtol, float) else 1e-8

        if self.atol is None or isinstance(self.atol, float):
            self.atol = wp.full(self.n_worlds, atol_val, dtype=self.scalar_type, device=self.device)

        if self.rtol is None or isinstance(self.rtol, float):
            self.rtol = wp.full(self.n_worlds, rtol_val, dtype=self.scalar_type, device=self.device)

        self.atol_sq = wp.empty(self.n_worlds, dtype=self.scalar_type, device=self.device)
        self.cur_iter = wp.empty(self.n_worlds, dtype=wp.int32, device=self.device)
        self.conditions = wp.empty(self.n_worlds + 1, dtype=wp.int32, device=self.device)
        self.termination_kernel = make_termination_kernel(self.n_worlds)

    @property
    def tiled_dot_product(self):
        return wp.get_device(self.device).is_cuda

    def compute_dot(self, a, b, active_dims, world_active, col_offset=0):
        if a.ndim == 2:
            a = a.reshape((1, *a.shape))
            b = b.reshape((1, *b.shape))
        if self.tiled_dot_product:
            result = self.dot_product[col_offset:]

            wp.launch_tiled(
                self.tiled_dot_kernel,
                dim=(a.shape[0], self.n_worlds),
                block_dim=min(256, self.dot_tile_size // 8),
                inputs=[a, b, active_dims, world_active],
                outputs=[result],
                device=self.device,
            )
        else:
            partial_sums = self.dot_partial_sums[col_offset:]
            wp.launch(
                dot_sequential,
                dim=(a.shape[0], self.n_worlds),
                inputs=[a, b, active_dims, world_active],
                outputs=[partial_sums],
                device=self.device,
            )


class CGSolver(ConjugateSolver):
    """Conjugate Gradient solver for symmetric positive definite systems.

    The solver terminates when ||r||^2 < max(rtol^2 * ||b||^2, atol^2) or
    when maxiter iterations are reached.
    """

    def _allocate(self):
        super()._allocate()

        # Temp storage
        self.r_and_z = wp.zeros((2, self.n_worlds, self.maxdims), dtype=self.scalar_type, device=self.device)
        self.p_and_Ap = wp.zeros_like(self.r_and_z)

        # (r, r) -- so we can compute r.z and r.r at once
        self.r_repeated = _repeat_first(self.r_and_z)
        if self.Mi is None:
            # without preconditioner r == z
            self.r_and_z = self.r_repeated
            self.rz_new = self.dot_product[0]
        else:
            self.rz_new = self.dot_product[1]

    def update_rr_rz(self, r, z, r_repeated, active_dims, world_active):
        # z = M r
        if self.Mi is None:
            self.compute_dot(r, r, active_dims, world_active)
        else:
            self.Mi.matvec(r, z, world_active)
            self.compute_dot(r_repeated, self.r_and_z, active_dims, world_active)

    def solve(
        self,
        b: wp.array,
        x: wp.array,
        active_dims: wp.array(dtype=Any) | None = None,
        world_active: wp.array(dtype=wp.int32) | None = None,
    ):
        if active_dims is None:
            if self.active_dims is None:
                raise ValueError("Error, active_dims must be provided either to constructor or to solve()")
            active_dims = self.active_dims
        if world_active is None:
            if self.world_active is None:
                raise ValueError("Error, world_active must be provided either to constructor or to solve()")
            world_active = self.world_active

        r, z = self.r_and_z[0], self.r_and_z[1]
        r_norm_sq = self.dot_product[0]
        p, Ap = self.p_and_Ap[0], self.p_and_Ap[1]

        self.compute_dot(b, b, active_dims, world_active)
        wp.launch(
            kernel=_initialize_tolerance_kernel,
            dim=self.n_worlds,
            device=self.device,
            inputs=[self.rtol, self.atol, self.dot_product[0]],
            outputs=[self.atol_sq],
        )
        r.assign(b)
        self.A.gemv(x, r, world_active, alpha=-1.0, beta=1.0)
        self.update_rr_rz(r, z, self.r_repeated, active_dims, world_active)
        p.assign(z)

        do_iteration = functools.partial(
            self.do_iteration,
            p=p,
            Ap=Ap,
            rz_old=self.residual,
            rz_new=self.rz_new,
            z=z,
            x=x,
            r=r,
            r_norm_sq=r_norm_sq,
            active_dims=active_dims,
            world_active=world_active,
        )

        return _run_capturable_loop(
            do_iteration,
            r_norm_sq,
            world_active,
            self.cur_iter,
            self.conditions,
            self.maxiter,
            self.atol_sq,
            self.callback,
            self.use_cuda_graph,
            termination_kernel=self.termination_kernel,
            use_graph_conditionals=self.use_graph_conditionals,
            maxiter_host=self.maxiter_host,
        )

    def do_iteration(self, p, Ap, rz_old, rz_new, z, x, r, r_norm_sq, active_dims, world_active):
        rz_old.assign(rz_new)

        # Ap = A * p
        self.A.matvec(p, Ap, world_active)
        self.compute_dot(p, Ap, active_dims, world_active, col_offset=1)
        p_Ap = self.dot_product[1]

        wp.launch(
            kernel=_cg_kernel_1,
            dim=(self.n_worlds, self.maxdims),
            inputs=[self.atol_sq, r_norm_sq, rz_old, p_Ap, p, Ap],
            outputs=[x, r],
            device=self.device,
        )

        self.update_rr_rz(r, z, self.r_repeated, active_dims, world_active)

        wp.launch(
            kernel=_cg_kernel_2,
            dim=(self.n_worlds, self.maxdims),
            inputs=[self.atol_sq, r_norm_sq, rz_old, rz_new, z],
            outputs=[p],
            device=self.device,
        )


class CRSolver(ConjugateSolver):
    """Conjugate Residual solver for symmetric (possibly indefinite) systems.

    The solver terminates when ||r||^2 < max(rtol^2 * ||b||^2, atol^2) or
    when maxiter iterations are reached.
    """

    def _allocate(self):
        super()._allocate()

        # Temp storage
        self.r_and_z = wp.zeros((2, self.n_worlds, self.maxdims), dtype=self.scalar_type, device=self.device)
        self.r_and_Az = wp.zeros_like(self.r_and_z)
        self.y_and_Ap = wp.zeros_like(self.r_and_z)
        self.p = wp.zeros((self.n_worlds, self.maxdims), dtype=self.scalar_type, device=self.device)
        # (r, r) -- so we can compute r.z and r.r at once

        if self.Mi is None:
            # For the unpreconditioned case, z == r and y == Ap
            self.r_and_z = _repeat_first(self.r_and_z)
            self.y_and_Ap = _repeat_first(self.y_and_Ap)

    def update_rr_zAz(self, z, Az, r, r_copy, active_dims, world_active):
        self.A.matvec(z, Az, world_active)
        r_copy.assign(r)
        self.compute_dot(self.r_and_z, self.r_and_Az, active_dims, world_active)

    def solve(
        self,
        b: wp.array,
        x: wp.array,
        active_dims: wp.array(dtype=Any) | None = None,
        world_active: wp.array(dtype=wp.int32) | None = None,
    ):
        if active_dims is None:
            if self.active_dims is None:
                raise ValueError("Error, active_dims must be provided either to constructor or to solve()")
            active_dims = self.active_dims
        if world_active is None:
            if self.world_active is None:
                raise ValueError("Error, world_active must be provided either to constructor or to solve()")
            world_active = self.world_active

        # named views
        r, z = self.r_and_z[0], self.r_and_z[1]
        r_copy, Az = self.r_and_Az[0], self.r_and_Az[1]
        y, Ap = self.y_and_Ap[0], self.y_and_Ap[1]

        r_norm_sq = self.dot_product[0]

        # Initialize tolerance from right-hand-side norm
        self.compute_dot(b, b, active_dims, world_active)
        wp.launch(
            kernel=_initialize_tolerance_kernel,
            dim=self.n_worlds,
            device=self.device,
            inputs=[self.rtol, self.atol, self.dot_product[0]],
            outputs=[self.atol_sq],
        )
        r.assign(b)
        self.A.gemv(x, r, world_active, alpha=-1.0, beta=1.0)

        # z = M r
        if self.Mi is not None:
            self.Mi.matvec(r, z, world_active)

        self.update_rr_zAz(z, Az, r, r_copy, active_dims, world_active)

        self.p.assign(z)
        Ap.assign(Az)

        do_iteration = functools.partial(
            self.do_iteration,
            p=self.p,
            Ap=Ap,
            Az=Az,
            zAz_old=self.residual,
            zAz_new=self.dot_product[1],
            z=z,
            y=y,
            x=x,
            r=r,
            r_copy=r_copy,
            r_norm_sq=r_norm_sq,
            active_dims=active_dims,
            world_active=world_active,
        )

        return _run_capturable_loop(
            do_iteration,
            r_norm_sq,
            world_active,
            self.cur_iter,
            self.conditions,
            self.maxiter,
            self.atol_sq,
            self.callback,
            self.use_cuda_graph,
            termination_kernel=self.termination_kernel,
            use_graph_conditionals=self.use_graph_conditionals,
            maxiter_host=self.maxiter_host,
        )

    def do_iteration(self, p, Ap, Az, zAz_old, zAz_new, z, y, x, r, r_copy, r_norm_sq, active_dims, world_active):
        zAz_old.assign(zAz_new)

        if self.Mi is not None:
            self.Mi.matvec(Ap, y, world_active)
        self.compute_dot(Ap, y, active_dims, world_active, col_offset=1)
        y_Ap = self.dot_product[1]

        if self.Mi is None:
            # In non-preconditioned case, first kernel is same as CG
            wp.launch(
                kernel=_cg_kernel_1,
                dim=(self.n_worlds, self.maxdims),
                inputs=[self.atol_sq, r_norm_sq, zAz_old, y_Ap, p, Ap],
                outputs=[x, r],
                device=self.device,
            )
        else:
            # In preconditioned case, we have one more vector to update
            wp.launch(
                kernel=_cr_kernel_1,
                dim=(self.n_worlds, self.maxdims),
                inputs=[self.atol_sq, r_norm_sq, zAz_old, y_Ap, p, Ap, y],
                outputs=[x, r, z],
                device=self.device,
            )

        self.update_rr_zAz(z, Az, r, r_copy, active_dims, world_active)

        wp.launch(
            kernel=_cr_kernel_2,
            dim=(self.n_worlds, self.maxdims),
            inputs=[self.atol_sq, r_norm_sq, zAz_old, zAz_new, z, Az],
            outputs=[p, Ap],
            device=self.device,
        )


def _repeat_first(arr: wp.array):
    # returns a view of the first element repeated arr.shape[0] times
    view = wp.array(
        ptr=arr.ptr,
        shape=arr.shape,
        dtype=arr.dtype,
        strides=(0, *arr.strides[1:]),
        device=arr.device,
    )
    view._ref = arr
    return view
