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

from collections.abc import Callable
from typing import Any

import warp as wp


@wp.struct
class NonZeroEntry:
    """Represents a non-zero entry in a sparse matrix.
    This structure stores the column index and corresponding value in a packed format, which provides
    better cache locality for sequential access patterns.
    """

    column_index: int
    value: float


@wp.struct
class SparseMatrixELL:
    """Represents a sparse matrix in ELLPACK (ELL) format."""

    num_nz: wp.array(dtype=int)  # Non-zeros count per column
    nz_ell: wp.array2d(dtype=NonZeroEntry)  # Padded ELL storage [row-major, fixed-height]


@wp.func
def ell_mat_vec_mul(
    num_nz: wp.array(dtype=int),
    nz_ell: wp.array2d(dtype=NonZeroEntry),
    x: wp.array(dtype=wp.vec3),
    tid: int,
):
    Mx = wp.vec3(0.0)
    for k in range(num_nz[tid]):
        nz_entry = nz_ell[k, tid]
        Mx += x[nz_entry.column_index] * nz_entry.value
    return Mx


@wp.kernel
def eval_residual_kernel(
    A_non_diag: SparseMatrixELL,
    A_diag: wp.array(dtype=Any),
    x: wp.array(dtype=wp.vec3),
    b: wp.array(dtype=wp.vec3),
    # outputs
    r: wp.array(dtype=wp.vec3),
):
    tid = wp.tid()
    Ax = A_diag[tid] * x[tid]
    Ax += ell_mat_vec_mul(A_non_diag.num_nz, A_non_diag.nz_ell, x, tid)
    r[tid] = b[tid] - Ax


# Forward-declare instances of the generic kernel to support graph capture on CUDA <12.3 drivers
wp.overload(eval_residual_kernel, {"A_diag": wp.array(dtype=wp.float32)})
wp.overload(eval_residual_kernel, {"A_diag": wp.array(dtype=wp.mat33)})


@wp.kernel
def eval_residual_kernel_with_additional_Ax(
    A_non_diag: SparseMatrixELL,
    A_diag: wp.array(dtype=Any),
    x: wp.array(dtype=wp.vec3),
    b: wp.array(dtype=wp.vec3),
    additional_Ax: wp.array(dtype=wp.vec3),
    # outputs
    r: wp.array(dtype=wp.vec3),
):
    tid = wp.tid()
    Ax = A_diag[tid] * x[tid] + additional_Ax[tid]
    Ax += ell_mat_vec_mul(A_non_diag.num_nz, A_non_diag.nz_ell, x, tid)
    r[tid] = b[tid] - Ax


# Forward-declare instances of the generic kernel to support graph capture on CUDA <12.3 drivers
wp.overload(eval_residual_kernel_with_additional_Ax, {"A_diag": wp.array(dtype=wp.float32)})
wp.overload(eval_residual_kernel_with_additional_Ax, {"A_diag": wp.array(dtype=wp.mat33)})


@wp.kernel
def array_mul_kernel(
    a: wp.array(dtype=Any),
    b: wp.array(dtype=wp.vec3),
    # outputs
    out: wp.array(dtype=wp.vec3),
):
    tid = wp.tid()
    out[tid] = a[tid] * b[tid]


# Forward-declare instances of the generic kernel to support graph capture on CUDA <12.3 drivers
wp.overload(array_mul_kernel, {"a": wp.array(dtype=wp.float32)})
wp.overload(array_mul_kernel, {"a": wp.array(dtype=wp.mat33)})


@wp.kernel
def ell_mat_vec_mul_kernel(
    M_non_diag: SparseMatrixELL,
    M_diag: wp.array(dtype=Any),
    x: wp.array(dtype=wp.vec3),
    # outputs
    Mx: wp.array(dtype=wp.vec3),
):
    tid = wp.tid()
    Mx[tid] = (M_diag[tid] * x[tid]) + ell_mat_vec_mul(M_non_diag.num_nz, M_non_diag.nz_ell, x, tid)


# Forward-declare instances of the generic kernel to support graph capture on CUDA <12.3 drivers
wp.overload(ell_mat_vec_mul_kernel, {"M_diag": wp.array(dtype=wp.float32)})
wp.overload(ell_mat_vec_mul_kernel, {"M_diag": wp.array(dtype=wp.mat33)})


@wp.kernel
def ell_mat_vec_mul_add_kernel(
    M_non_diag: SparseMatrixELL,
    M_diag: wp.array(dtype=Any),
    x: wp.array(dtype=wp.vec3),
    additional_Mx: wp.array(dtype=wp.vec3),
    # outputs
    Mx: wp.array(dtype=wp.vec3),
):
    tid = wp.tid()
    result = (M_diag[tid] * x[tid]) + additional_Mx[tid]
    result += ell_mat_vec_mul(M_non_diag.num_nz, M_non_diag.nz_ell, x, tid)
    Mx[tid] = result


# Forward-declare instances of the generic kernel to support graph capture on CUDA <12.3 drivers
wp.overload(ell_mat_vec_mul_add_kernel, {"M_diag": wp.array(dtype=wp.float32)})
wp.overload(ell_mat_vec_mul_add_kernel, {"M_diag": wp.array(dtype=wp.mat33)})


@wp.kernel
def update_cg_direction_kernel(
    iter: int,
    z: wp.array(dtype=wp.vec3),
    rTz: wp.array(dtype=float),
    p_prev: wp.array(dtype=wp.vec3),
    # outputs
    p: wp.array(dtype=wp.vec3),
):
    # p = r + (rz_new / rz_old) * p;
    i = wp.tid()
    new_p = z[i]
    if iter > 0:
        num = rTz[iter]
        denom = rTz[iter - 1]
        beta = wp.float32(0.0)
        if (wp.abs(denom) > 1.0e-30) and (not wp.isnan(denom)) and (not wp.isnan(num)):
            beta = num / denom
        new_p += beta * p_prev[i]
    p[i] = new_p


@wp.kernel
def step_cg_kernel(
    iter: int,
    rTz: wp.array(dtype=float),
    pTAp: wp.array(dtype=float),
    p: wp.array(dtype=wp.vec3),
    Ap: wp.array(dtype=wp.vec3),
    # outputs
    x: wp.array(dtype=wp.vec3),
    r: wp.array(dtype=wp.vec3),
):
    i = wp.tid()
    num = rTz[iter]
    denom = pTAp[iter]
    alpha = wp.float32(0.0)
    if (wp.abs(denom) > 1.0e-30) and (not wp.isnan(denom)) and (not wp.isnan(num)):
        alpha = num / denom
    r[i] = r[i] - alpha * Ap[i]
    x[i] = x[i] + alpha * p[i]


@wp.kernel
def generate_test_data_kernel(
    dim: int,
    diag_term: float,
    A_non_diag: SparseMatrixELL,
    A_diag: wp.array(dtype=Any),
    b: wp.array(dtype=wp.vec3),
    x0: wp.array(dtype=wp.vec3),
):
    tid = wp.tid()

    t = wp.float32(tid)
    b[tid] = wp.vec3(wp.sin(t * 0.123), wp.cos(t * 0.456), wp.sin(t * 0.789))
    x0[tid] = wp.vec3(wp.cos(t * 0.123), wp.tan(t * 0.456), wp.cos(t * 0.789))

    A_diag[tid] = diag_term

    if tid == 0:
        A_non_diag.num_nz[tid] = 1
        A_non_diag.nz_ell[0, tid].value = -1.0
        A_non_diag.nz_ell[0, tid].column_index = 1
    elif tid == dim - 1:
        A_non_diag.num_nz[tid] = 1
        A_non_diag.nz_ell[0, tid].value = -1.0
        A_non_diag.nz_ell[0, tid].column_index = dim - 2
    else:
        A_non_diag.num_nz[tid] = 2
        A_non_diag.nz_ell[0, tid].value = -1.0
        A_non_diag.nz_ell[0, tid].column_index = tid + 1
        A_non_diag.nz_ell[1, tid].value = -1.0
        A_non_diag.nz_ell[1, tid].column_index = tid - 1


def array_inner(
    a: wp.array(dtype=wp.vec3),
    b: wp.array(dtype=wp.vec3),
    out_ptr: wp.uint64,
):
    from warp._src.context import runtime  # noqa: PLC0415

    if a.device.is_cpu:
        func = runtime.core.wp_array_inner_float_host
    else:
        func = runtime.core.wp_array_inner_float_device

    func(
        a.ptr,
        b.ptr,
        out_ptr,
        len(a),
        wp.types.type_size_in_bytes(a.dtype),
        wp.types.type_size_in_bytes(b.dtype),
        wp.types.type_size(a.dtype),
    )


class PcgSolver:
    """A Customized PCG implementation for efficient cloth simulation

    Ref: https://en.wikipedia.org/wiki/Conjugate_gradient_method

    Sparse Matrix Storages:
        Part-1: (static)
            1. Non-diagonals: SparseMatrixELL
            2. Diagonals: wp.array(dtype = mat3x3)
        Part-2: (dynamic)
            1. Preconditioner: wp.array(wp.mat3x3)
            2. Matrix-free Ax: wp.array(dtype = wp.vec3)
            3. Matrix-free diagonals: wp.array(wp.mat3x3)
    """

    def __init__(self, dim: int, device, maxIter: int = 999):
        self.dim = dim  # pre-allocation
        self.device = device
        self.maxIter = maxIter
        self.r = wp.array(shape=dim, dtype=wp.vec3, device=device)
        self.z = wp.array(shape=dim, dtype=wp.vec3, device=device)
        self.p = wp.array(shape=dim, dtype=wp.vec3, device=device)
        self.Ap = wp.array(shape=dim, dtype=wp.vec3, device=device)
        self.pTAp = wp.array(shape=maxIter, dtype=float, device=device)
        self.rTz = wp.array(shape=maxIter, dtype=float, device=device)

    def step1_update_r(
        self,
        A_non_diag: SparseMatrixELL,
        A_diag: wp.array(dtype=Any),
        b: wp.array(dtype=wp.vec3),
        x: wp.array(dtype=wp.vec3) = None,  # Pass `None` if x[:] == 0.0
        additional_Ax: wp.array(dtype=wp.vec3) = None,  # Pass `None` if additional_Ax[:] == 0.0
    ):
        """Update residual: r = b - A * x"""
        if x is None:
            self.r.assign(b)
        elif additional_Ax is None:
            wp.launch(
                eval_residual_kernel,
                dim=self.dim,
                inputs=[A_non_diag, A_diag, x, b],
                outputs=[self.r],
                device=self.device,
            )
        else:
            wp.launch(
                eval_residual_kernel_with_additional_Ax,
                dim=self.dim,
                inputs=[A_non_diag, A_diag, x, b, additional_Ax],
                outputs=[self.r],
                device=self.device,
            )

    def step2_update_z(self, inv_M: wp.array(dtype=Any)):
        wp.launch(array_mul_kernel, dim=self.dim, inputs=[inv_M, self.r], outputs=[self.z], device=self.device)

    def step3_update_rTz(self, iter: int):
        array_inner(self.r, self.z, self.rTz.ptr + iter * self.rTz.strides[0])

    def step4_update_p(self, iter: int):
        wp.launch(
            update_cg_direction_kernel,
            dim=self.dim,
            inputs=[iter, self.z, self.rTz, self.p],
            outputs=[self.p],
            device=self.device,
        )

    def step5_update_Ap(
        self,
        A_non_diag: SparseMatrixELL,
        A_diag: wp.array(dtype=Any),
        additional_Ap: wp.array(dtype=wp.vec3) = None,
    ):
        if additional_Ap is None:
            wp.launch(
                ell_mat_vec_mul_kernel,
                dim=self.dim,
                inputs=[A_non_diag, A_diag, self.p],
                outputs=[self.Ap],
                device=self.device,
            )
        else:
            wp.launch(
                ell_mat_vec_mul_add_kernel,
                dim=self.dim,
                inputs=[A_non_diag, A_diag, self.p, additional_Ap],
                outputs=[self.Ap],
                device=self.device,
            )

    def step6_update_pTAp(self, iter: int):
        array_inner(self.p, self.Ap, self.pTAp.ptr + iter * self.pTAp.strides[0])

    def step7_update_x_r(self, x: wp.array(dtype=wp.vec3), iter: int):
        wp.launch(
            step_cg_kernel,
            dim=self.dim,
            inputs=[iter, self.rTz, self.pTAp, self.p, self.Ap],
            outputs=[x, self.r],
            device=self.device,
        )

    def solve(
        self,
        A_non_diag: SparseMatrixELL,
        A_diag: wp.array(dtype=Any),
        x0: wp.array(dtype=wp.vec3),  # Pass `None` means x0[:] == 0.0
        b: wp.array(dtype=wp.vec3),
        inv_M: wp.array(dtype=Any),
        x1: wp.array(dtype=wp.vec3),
        iterations: int,
        additional_multiplier: Callable | None = None,
    ):
        # Prevent out-of-bounds in rTz/pTAp when iterations > maxIter.
        iterations = wp.min(iterations, self.maxIter)

        if x0 is None:
            x1.zero_()
        else:
            x1.assign(x0)

        if additional_multiplier is None:
            self.step1_update_r(A_non_diag, A_diag, b, x0)
        else:
            additional_Ax = additional_multiplier(x0) if x0 is not None else None
            self.step1_update_r(A_non_diag, A_diag, b, x0, additional_Ax)

        for iter in range(iterations):
            self.step2_update_z(inv_M)
            self.step3_update_rTz(iter)
            self.step4_update_p(iter)

            if additional_multiplier is None:
                self.step5_update_Ap(A_non_diag, A_diag)
            else:
                additional_Ap = additional_multiplier(self.p)
                self.step5_update_Ap(A_non_diag, A_diag, additional_Ap)

            self.step6_update_pTAp(iter)
            self.step7_update_x_r(x1, iter)


if __name__ == "__main__":
    wp.init()
    dim = 100000
    diag_term = 5.0

    A_non_diag = SparseMatrixELL()
    A_diag = wp.zeros(dim, dtype=wp.float32)
    A_non_diag.num_nz = wp.zeros(dim, dtype=wp.int32)
    A_non_diag.nz_ell = wp.zeros(shape=(2, dim), dtype=NonZeroEntry)
    b = wp.zeros(dim, dtype=wp.vec3)
    x0 = wp.zeros(dim, dtype=wp.vec3)
    x1 = wp.zeros(dim, dtype=wp.vec3)
    wp.launch(generate_test_data_kernel, dim=dim, inputs=[dim, diag_term], outputs=[A_non_diag, A_diag, b, x0])

    inv_M = wp.array([1.0 / diag_term] * dim, dtype=float)

    solver = PcgSolver(dim, device="cuda:0")
    solver.solve(A_non_diag, A_diag, x0, b, inv_M, x1, iterations=30)

    rTr = wp.zeros(1, dtype=float)
    array_inner(solver.r, solver.r, rTr.ptr)
    print(rTr.numpy()[0])
