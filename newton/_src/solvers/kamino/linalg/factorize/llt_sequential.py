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

"""KAMINO: Linear Algebra: Sequential LLT (i.e. Cholesky) factorization w/o intra-parallelism"""

from __future__ import annotations

import warp as wp
from warp.context import Devicelike

from ...core.math import FLOAT32_EPS
from ...core.types import float32, int32

###
# Module interface
###

__all__ = [
    "llt_sequential_factorize",
    "llt_sequential_solve",
    "llt_sequential_solve_inplace",
]


###
# Module configs
###

wp.set_module_options({"enable_backward": False})


###
# Kernels
###


@wp.kernel
def _llt_sequential_factorize(
    # Inputs:
    dim_in: wp.array(dtype=int32),
    mio_in: wp.array(dtype=int32),
    A_in: wp.array(dtype=float32),
    # Outputs:
    L_out: wp.array(dtype=float32),
):
    # Retrieve the thread index
    tid = wp.tid()

    # Retrieve the matrix start offset and dimension
    mio = mio_in[tid]
    n = dim_in[tid]

    # Compute the Cholesky factorization sequentially
    for i in range(n):
        m_i = mio + n * i
        m_ii = m_i + i
        A_ii = A_in[m_ii]
        for j in range(i + 1):
            m_j = mio + n * j
            m_jj = m_j + j
            m_ij = m_i + j
            A_ij = A_in[m_ij]
            L_jj = L_out[m_jj]
            sum = float32(0.0)
            for k in range(j):
                m_ik = m_i + k
                m_jk = m_j + k
                sum += L_out[m_ik] * L_out[m_jk]
            if i == j:
                L_out[m_ij] = wp.sqrt(wp.max(A_ii - sum, FLOAT32_EPS))
            else:
                L_out[m_ij] = (A_ij - sum) / L_jj


@wp.kernel
def _llt_sequential_solve(
    # Inputs:
    dim_in: wp.array(dtype=int32),
    mio_in: wp.array(dtype=int32),
    vio_in: wp.array(dtype=int32),
    L_in: wp.array(dtype=float32),
    b_in: wp.array(dtype=float32),
    # Outputs:
    y_out: wp.array(dtype=float32),
    x_out: wp.array(dtype=float32),
):
    # Retrieve the thread index
    tid = wp.tid()

    # Retrieve the start offsets and problem dimension
    mio = mio_in[tid]
    vio = vio_in[tid]
    n = dim_in[tid]

    # Forward substitution to solve L * y = b
    for i in range(n):
        m_i = mio + n * i
        m_ii = m_i + i
        L_ii = L_in[m_ii]
        sum_i = b_in[vio + i]
        for j in range(i):
            m_ij = m_i + j
            sum_i -= L_in[m_ij] * y_out[vio + j]
        y_out[vio + i] = sum_i / L_ii

    # Backward substitution to solve L^T * x = y
    for i in range(n - 1, -1, -1):
        m_i = mio + n * i
        m_ii = m_i + i
        LT_ii = L_in[m_ii]
        sum_i = y_out[vio + i]
        for j in range(i + 1, n):
            m_ji = mio + n * j + i
            sum_i -= L_in[m_ji] * x_out[vio + j]
        x_out[vio + i] = sum_i / LT_ii


@wp.kernel
def _llt_sequential_solve_inplace(
    # Inputs:
    dim_in: wp.array(dtype=int32),
    mio_in: wp.array(dtype=int32),
    vio_in: wp.array(dtype=int32),
    L_in: wp.array(dtype=float32),
    x_inout: wp.array(dtype=float32),
):
    # Retrieve the thread index
    tid = wp.tid()

    # Retrieve the start offsets and problem dimension
    mio = mio_in[tid]
    vio = vio_in[tid]
    n = dim_in[tid]

    # Forward substitution to solve L * y = b
    for i in range(n):
        m_i = mio + n * i
        m_ii = m_i + i
        L_ii = L_in[m_ii]
        sum_i = x_inout[vio + i]
        for j in range(i):
            m_ij = m_i + j
            sum_i -= L_in[m_ij] * x_inout[vio + j]
        x_inout[vio + i] = sum_i / L_ii

    # Backward substitution to solve L^T * x = y
    for i in range(n - 1, -1, -1):
        m_i = mio + n * i
        m_ii = m_i + i
        LT_ii = L_in[m_ii]
        sum_i = x_inout[vio + i]
        for j in range(i + 1, n):
            m_ji = mio + n * j + i
            sum_i -= L_in[m_ji] * x_inout[vio + j]
        x_inout[vio + i] = sum_i / LT_ii


###
# Launchers
###


def llt_sequential_factorize(
    num_blocks: int,
    dim: wp.array(dtype=int32),
    mio: wp.array(dtype=int32),
    A: wp.array(dtype=float32),
    L: wp.array(dtype=float32),
    device: Devicelike = None,
):
    """
    Launches the sequential Cholesky factorization kernel for a block partitioned matrix.

    Args:
        num_blocks (int): The number of matrix blocks to process.
        dim (wp.array): An array of shape `(num_blocks,)` containing the dimensions of each matrix block.
        mio (wp.array): An array of shape `(num_blocks,)` containing the start indices of each matrix block.
        A (wp.array): The flat input array containing the input matrix blocks to be factorized.
        L (wp.array): The flat output array containing the Cholesky factorization of each matrix block.
    """
    wp.launch(
        kernel=_llt_sequential_factorize,
        dim=num_blocks,
        inputs=[dim, mio, A, L],
        device=device,
    )


def llt_sequential_solve(
    num_blocks: int,
    dim: wp.array(dtype=int32),
    mio: wp.array(dtype=int32),
    vio: wp.array(dtype=int32),
    L: wp.array(dtype=float32),
    b: wp.array(dtype=float32),
    y: wp.array(dtype=float32),
    x: wp.array(dtype=float32),
    device: Devicelike = None,
):
    """
    Launches the sequential solve kernel using the Cholesky factorization of a block partitioned matrix.

    Args:
        num_blocks (int): The number of matrix blocks to process.
        dim (wp.array): An array of shape `(num_blocks,)` containing the dimensions of each matrix block.
        mio (wp.array): An array of shape `(num_blocks,)` containing the start indices of each matrix block.
        vio (wp.array): An array of shape `(num_blocks,)` containing the start indices of each vector block.
        L (wp.array): The flat input array containing the Cholesky factorization of each matrix block.
        b (wp.array): The flat input array containing the stacked right-hand side vectors.
        y (wp.array): The output array where the intermediate result will be stored.
        x (wp.array): The output array where the solution to the linear system `A @ x = b` will be stored.
    """
    wp.launch(
        kernel=_llt_sequential_solve,
        dim=num_blocks,
        inputs=[dim, mio, vio, L, b, y, x],
        device=device,
    )


def llt_sequential_solve_inplace(
    num_blocks: int,
    dim: wp.array(dtype=int32),
    mio: wp.array(dtype=int32),
    vio: wp.array(dtype=int32),
    L: wp.array(dtype=float32),
    x: wp.array(dtype=float32),
    device: Devicelike = None,
):
    """
    Launches the sequential in-place solve kernel using the Cholesky factorization of a block partitioned matrix.

    Args:
        num_blocks (int): The number of matrix blocks to process.
        dim (wp.array): An array of shape `(num_blocks,)` containing the dimensions of each matrix block.
        mio (wp.array): An array of shape `(num_blocks,)` containing the start indices of each matrix block.
        vio (wp.array): An array of shape `(num_blocks,)` containing the start indices of each vector block.
        L (wp.array): The flat input array containing the Cholesky factorization of each matrix block.
        x (wp.array): The input/output array where the solution to the linear system `A @ x = b` will be stored in-place.
    """
    wp.launch(
        kernel=_llt_sequential_solve_inplace,
        dim=num_blocks,
        inputs=[dim, mio, vio, L, x],
        device=device,
    )
