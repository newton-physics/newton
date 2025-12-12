# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
KAMINO: Conjugate gradient and conjugate residual solvers
"""

from __future__ import annotations

import functools
import math
from collections.abc import Callable
from typing import Any

import warp as wp

# No need to auto-generate adjoint code for linear solvers
wp.set_module_options({"enable_backward": False})

# based on the warp.optim.linear implementation


__all__ = [
    "BatchedLinearOperator",
    "CGSolver",
    "CRSolver",
    "make_dense_square_matrix_operator",
    "make_diag_matrix_operator",
    "make_jacobi_preconditioner",
]


class BatchedLinearOperator:
    """
    Linear operator to be used as left-hand-side of linear iterative solvers for a batch of independent problems.
    The rhs and x vectors are stored as 2d arrays with the first dimension corresponding to the world.

    Args:
        shape: Tuple of (n_worlds, max_rows, max_cols)
        dtype: Type of the operator elements
        device: Device on which computations involving the operator should be performed
        matvec: Matrix-vector multiplication routine
        ndim: int,

    The matrix-vector multiplication routine should have the following signature:

    .. code-block:: python

        def matvec(
            x: wp.array,
            y: wp.array,
            z: wp.array,
            dims: wp.array,
            alpha: Scalar,
            beta: Scalar,
        ):
            '''Perform a generalized matrix-vector product.

            This function computes the operation z = alpha * (A @ x) + beta * y, where 'A'
            is the linear operator represented by this class.

            The additional arguments enable batched solving across multiple world:
              - 'ndim' is the number of independent world processed in parallel
            '''
            ...

    For performance reasons, by default the iterative linear solvers in this module will try to capture the calls
    for one or more iterations in CUDA graphs. If the `matvec` routine of a custom :class:`BatchedLinearOperator`
    cannot be graph-captured, the ``use_cuda_graph=False`` parameter should be passed to the solver function.

    For performance reasons, no assumptions may be made about elements outside the active dims.

    """

    def __init__(self, shape: tuple[int, int, int], dtype: type, device: wp.context.Device, matvec: Callable):
        # TODO: active shapes per world, mostly for correctness
        self._shape = shape
        self._dtype = dtype
        self._device = device
        self._matvec: Callable[int, int] = matvec
        self._active_array = wp.full(shape[0], True, dtype=wp.bool, device=self._device)

    @property
    def shape(self) -> tuple[int, int, int]:
        return self._shape

    @property
    def dtype(self) -> type:
        return self._dtype

    @property
    def device(self) -> wp.context.Device:
        return self._device

    @property
    def matvec(self) -> Callable:
        return self._matvec

    @property
    def scalar_type(self):
        return wp.types.type_scalar_type(self.dtype)


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
        world_active: wp.array(dtype=wp.bool),
        cur_iter: wp.array(dtype=int),
        world_condition: wp.array(dtype=wp.int32),
        batch_condition: wp.array(dtype=wp.int32),
    ):
        thread = wp.tid()
        active = wp.tile_astype(wp.tile_load(world_active, (n_worlds,)), wp.int32)
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


# Diagonal matrices
@wp.kernel
def diag_matvec_kernel(
    x: wp.array2d(dtype=Any),
    y: wp.array2d(dtype=Any),
    D: wp.array2d(dtype=Any),
    active_dims: wp.array(dtype=Any),
    world_active: wp.array(dtype=wp.bool),
    alpha: Any,
    beta: Any,
    z: wp.array2d(dtype=Any),
):
    world, row = wp.tid()
    assert world < len(active_dims)
    if not world_active[world] or row >= active_dims[world]:
        return

    zero = type(alpha)(0)
    s = z.dtype(0)

    if alpha != zero:
        s += alpha * D[world, row] * x[world, row]
    if beta != zero:
        s += beta * y[world, row]
    z[world, row] = s


def make_diag_matrix_operator(
    diag: wp.array2d(dtype=Any),
    max_dims: int,
    active_dims: wp.array(dtype=Any),
) -> BatchedLinearOperator:
    # int_type = active_dims.dtype
    n = len(active_dims)
    device = diag.device
    assert active_dims.device == device

    dtype = diag.dtype
    batch_size = n

    def matvec(
        x: wp.array2d(dtype=Any),
        y: wp.array2d(dtype=Any),
        z: wp.array2d(dtype=Any),
        world_active: wp.array(dtype=wp.bool),
        alpha: Any,
        beta: Any,
    ):
        wp.launch(
            diag_matvec_kernel,
            dim=(batch_size, max_dims),
            inputs=[x, y, diag, active_dims, world_active, dtype(alpha), dtype(beta)],
            outputs=[z],
            device=device,
        )

    shape = (n, max_dims, max_dims)
    return BatchedLinearOperator(shape=shape, dtype=dtype, device=device, matvec=matvec)


# Dense matrices
@wp.kernel
def _dense_matvec_kernel(
    x: wp.array2d(dtype=Any),
    y: wp.array2d(dtype=Any),
    A: wp.array2d(dtype=Any),
    active_dims: wp.array(dtype=Any),
    world_active: wp.array(dtype=wp.bool),
    alpha: Any,
    beta: Any,
    matrix_stride: int,
    tile_size: int,
    z: wp.array2d(dtype=Any),
):
    """Computes z[i] = alpha * (A[i] @ x[i]) + beta * y[i]"""
    world, row, lane = wp.tid()
    assert world < len(active_dims)
    dim = active_dims[world]
    if not world_active[world] or row >= dim:
        return

    row_stride = active_dims[world]  # Active elements are contiguous in memory
    zero = type(alpha)(0)
    s = zero
    if alpha != zero:
        for col in range(lane, dim, tile_size):
            s += A[world, row * row_stride + col] * x[world, col]
    row_tile = wp.tile_sum(wp.tile(s * alpha))
    if beta != zero:
        row_tile += beta * wp.tile_load(y[world], shape=1, offset=row)
    wp.tile_store(z[world], row_tile, offset=row)


def make_dense_square_matrix_operator(
    A: wp.array2d(dtype=Any),
    active_dims: wp.array(dtype=Any),
    max_dims: int,
    matrix_stride: int,
    block_dim: int = 64,
) -> BatchedLinearOperator:
    """Create a BatchedLinearOperator computing `z = \\alpha (A x) + \\beta y`
    for multiple, differently sized square matrices."""
    dtype = A.dtype
    device = A.device

    tile_size = block_dim if device.is_cuda else 1
    n = len(active_dims)
    assert A.shape[0] == n

    def matvec(
        x: wp.array(dtype=Any),
        y: wp.array(dtype=Any),
        z: wp.array(dtype=Any),
        world_active: wp.array(dtype=wp.bool),
        alpha: Any,
        beta: Any,
    ):
        wp.launch(
            _dense_matvec_kernel,
            dim=(len(active_dims), max_dims, block_dim),
            inputs=[x, y, A, active_dims, world_active, dtype(alpha), dtype(beta), matrix_stride, tile_size],
            outputs=[z],
            device=device,
            block_dim=tile_size if tile_size > 1 else 256,
        )

    shape = (n, max_dims, max_dims)
    return BatchedLinearOperator(shape=shape, dtype=dtype, device=device, matvec=matvec)


def _run_capturable_loop(
    do_cycle: Callable,
    r_norm_sq: wp.array,
    world_active: wp.array(dtype=wp.bool),
    cur_iter: wp.array(dtype=wp.int32),
    conditions: wp.array(dtype=wp.int32),
    maxiter: wp.array(dtype=int),
    atol_sq: wp.array,
    callback: Callable | None,
    use_cuda_graph: bool,
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
        wp.capture_while(global_condition, do_cycle_with_condition)
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


@functools.cache
def make_dot_kernel(tile_size: int, maxdim: int):
    second_tile_size = (maxdim + tile_size - 1) // tile_size

    @wp.kernel(enable_backward=False)
    def dot(
        a: wp.array3d(dtype=Any),
        b: wp.array3d(dtype=Any),
        world_size: wp.array(dtype=wp.int32),
        world_active: wp.array(dtype=wp.bool),
        result: wp.array2d(dtype=Any),
    ):
        """Compute the dot products between the trailing-dim arrays in a and b using tiles and pairwise summation."""
        col, world, _ = wp.tid()
        if not world_active[world]:
            return
        n = world_size[world]

        ts = wp.tile_zeros((second_tile_size,), dtype=a.dtype, storage="shared")
        o_src = wp.int32(0)

        for block in range(second_tile_size):
            if o_src >= n:
                break
            ta = wp.tile_load(a[col, world], shape=tile_size, offset=o_src)
            tb = wp.tile_load(b[col, world], shape=tile_size, offset=o_src)
            # TODO: consider using ts[block] twice, look into += data race in wp
            prod = wp.tile_map(wp.mul, ta, tb)
            if o_src > n - tile_size:
                thresh = wp.tile_full((tile_size,), a.dtype(n - o_src), dtype=a.dtype)
                mask = wp.tile_map(lt_mask, wp.tile_arange(tile_size, dtype=a.dtype), thresh)
                prod = wp.tile_map(mul_mask, mask, prod)
            s = wp.tile_sum(prod)
            ts[block] = s[0]
            o_src += tile_size
        wp.tile_store(result[col], wp.tile_sum(ts), offset=world)

    return dot


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
    def __init__(
        self,
        A: BatchedLinearOperator,
        active_dims: wp.array(dtype=Any),
        world_active: wp.array(dtype=wp.bool),
        atol: float | wp.array(dtype=Any) | None = None,
        rtol: float | wp.array(dtype=Any) | None = None,
        maxiter: wp.array = None,
        M: BatchedLinearOperator | None = None,
        callback: Callable | None = None,
        use_cuda_graph=True,
    ):
        if not isinstance(A, BatchedLinearOperator):
            raise ValueError("A must be a BatchedLinearOperator")
        if not isinstance(M, BatchedLinearOperator | None):
            raise ValueError("M must be a BatchedLinearOperator or None")

        self.scalar_type = A.scalar_type

        self.n_worlds, self.maxdims, _ = A.shape
        if self.maxdims != A.shape[2]:
            raise ValueError("A must be a square BatchedLinearOperator")
        if M is not None and A.shape != M.shape:
            raise ValueError("M and A must have the same dimensions")

        self.A = A
        self.M = M
        self.device = A.device

        self.active_dims = active_dims
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
            self.maxiter = wp.full(self.n_worlds, int(1.5 * self.maxdims), dtype=int, device=self.device)

        # TODO: non-tiled variant for CPU
        self.dot_product = wp.empty((2, self.n_worlds), dtype=self.scalar_type, device=self.device)

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

    def compute_dot(self, a, b, col_offset=0):
        block_dim = 256
        if a.ndim == 2:
            a = a.reshape((1, *a.shape))
            b = b.reshape((1, *b.shape))

        result = self.dot_product[col_offset:]

        wp.launch(
            self.tiled_dot_kernel,
            dim=(a.shape[0], self.n_worlds, block_dim),
            block_dim=min(256, self.dot_tile_size // 8),
            inputs=[a, b, self.active_dims, self.world_active],
            outputs=[result],
            device=self.device,
        )


class CGSolver(ConjugateSolver):
    def _allocate(self):
        super()._allocate()

        # Temp storage
        self.r_and_z = wp.empty((2, *self.A.shape[:2]), dtype=self.scalar_type, device=self.device)
        self.p_and_Ap = wp.empty_like(self.r_and_z)

        # (r, r) -- so we can compute r.z and r.r at once
        self.r_repeated = _repeat_first(self.r_and_z)
        if self.M is None:
            # without preconditioner r == z
            # TODO: allocate r_and_z here
            self.r_and_z = self.r_repeated
            self.rz_new = self.dot_product[0]
        else:
            self.rz_new = self.dot_product[1]

    def update_rr_rz(self, r, z, r_repeated):
        # z = M r
        if self.M is None:
            self.compute_dot(r, r)
        else:
            self.M.matvec(r, z, z, self.world_active, alpha=1.0, beta=0.0)
            self.compute_dot(r_repeated, self.r_and_z)

    def solve(self, b: wp.array, x: wp.array):
        r, z = self.r_and_z[0], self.r_and_z[1]
        r_norm_sq = self.dot_product[0]
        p, Ap = self.p_and_Ap[0], self.p_and_Ap[1]

        self.compute_dot(b, b)
        wp.launch(
            kernel=_initialize_tolerance_kernel,
            dim=self.n_worlds,
            device=self.device,
            inputs=[self.rtol, self.atol, self.dot_product[0]],
            outputs=[self.atol_sq],
        )
        # Not strictly necessary, but makes it more robust to user-provided BatchedLinearOperators

        # Ap.zero_()
        # z.zero_()

        # Initialize residual
        self.A.matvec(x, b, r, self.world_active, alpha=-1.0, beta=1.0)
        self.update_rr_rz(r, z, self.r_repeated)
        p.assign(z)

        do_iteration = functools.partial(
            self.do_iteration, p=p, Ap=Ap, rz_old=self.residual, rz_new=self.rz_new, z=z, x=x, r=r, r_norm_sq=r_norm_sq
        )

        return _run_capturable_loop(
            do_iteration,
            r_norm_sq,
            self.world_active,
            self.cur_iter,
            self.conditions,
            self.maxiter,
            self.atol_sq,
            self.callback,
            self.use_cuda_graph,
            termination_kernel=self.termination_kernel,
        )

    def do_iteration(self, p, Ap, rz_old, rz_new, z, x, r, r_norm_sq):
        rz_old.assign(rz_new)

        # Ap = A * p;
        self.A.matvec(p, Ap, Ap, self.world_active, alpha=1, beta=0)
        self.compute_dot(p, Ap, col_offset=1)
        p_Ap = self.dot_product[1]

        wp.launch(
            kernel=_cg_kernel_1,
            dim=(self.n_worlds, self.maxdims),
            inputs=[self.atol_sq, r_norm_sq, rz_old, p_Ap, p, Ap],
            outputs=[x, r],
            device=self.device,
        )

        self.update_rr_rz(r, z, self.r_repeated)

        wp.launch(
            kernel=_cg_kernel_2,
            dim=(self.n_worlds, self.maxdims),
            inputs=[self.atol_sq, r_norm_sq, rz_old, rz_new, z],
            outputs=[p],
            device=self.device,
        )


class CRSolver(ConjugateSolver):
    # Notation roughly follow spseudo-code from https://en.wikipedia.org/wiki/Conjugate_residual_method
    # with z := M^-1 r and y := M^-1 Ap

    def _allocate(self):
        super()._allocate()

        # Temp storage
        self.r_and_z = wp.empty((2, *self.A.shape[:2]), dtype=self.scalar_type, device=self.device)
        self.r_and_Az = wp.empty_like(self.r_and_z)
        self.y_and_Ap = wp.empty_like(self.r_and_z)
        self.p = wp.empty(self.A.shape[:2], dtype=self.scalar_type, device=self.device)
        # (r, r) -- so we can compute r.z and r.r at once

        if self.M is None:
            # For the unpreconditioned case, z == r and y == Ap
            self.r_and_z = _repeat_first(self.r_and_z)
            self.y_and_Ap = _repeat_first(self.y_and_Ap)

    def update_rr_zAz(self, z, Az, r, r_copy):
        self.A.matvec(z, Az, Az, self.world_active, alpha=1, beta=0)
        r_copy.assign(r)
        self.compute_dot(self.r_and_z, self.r_and_Az)

    def solve(self, b: wp.array, x: wp.array):
        # named views
        r, z = self.r_and_z[0], self.r_and_z[1]
        r_copy, Az = self.r_and_Az[0], self.r_and_Az[1]
        y, Ap = self.y_and_Ap[0], self.y_and_Ap[1]

        r_norm_sq = self.dot_product[0]

        # Initialize tolerance from right-hand-side norm
        self.compute_dot(b, b)
        wp.launch(
            kernel=_initialize_tolerance_kernel,
            dim=self.n_worlds,
            device=self.device,
            inputs=[self.rtol, self.atol, self.dot_product[0]],
            outputs=[self.atol_sq],
        )
        self.A.matvec(x, b, r, self.world_active, alpha=-1.0, beta=1.0)

        # Not strictly necessary, but makes it more robust to user-provided LinearOperators

        # Ap.zero_()
        # y.zero_()

        # z = M r
        if self.M is not None:
            z.zero_()
            self.M.matvec(r, z, z, self.world_active, alpha=1.0, beta=0.0)

        self.update_rr_zAz(z, Az, r, r_copy)

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
        )

        return _run_capturable_loop(
            do_iteration,
            r_norm_sq,
            self.world_active,
            self.cur_iter,
            self.conditions,
            self.maxiter,
            self.atol_sq,
            self.callback,
            self.use_cuda_graph,
            termination_kernel=self.termination_kernel,
        )

    def do_iteration(self, p, Ap, Az, zAz_old, zAz_new, z, y, x, r, r_copy, r_norm_sq):
        zAz_old.assign(zAz_new)

        if self.M is not None:
            self.M.matvec(Ap, y, y, self.world_active, alpha=1.0, beta=0.0)
        self.compute_dot(Ap, y, col_offset=1)
        y_Ap = self.dot_product[1]

        if self.M is None:
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

        self.update_rr_zAz(z, Az, r, r_copy)

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
