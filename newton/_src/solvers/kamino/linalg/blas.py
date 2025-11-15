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

"""BLAS-like operations for multi-linear systems"""

import warp as wp

from ..core.types import float32, int32

###
# Module configs
###

wp.set_module_options({"enable_backward": False})


##
# Kernels
##


@wp.kernel
def _mult_left_right_diag_matrix_with_matrix(
    # Inputs:
    dim: wp.array(dtype=int32),
    mio: wp.array(dtype=int32),
    vio: wp.array(dtype=int32),
    D: wp.array(dtype=float32),
    X: wp.array(dtype=float32),
    # Outputs:
    Y: wp.array(dtype=float32),
):
    # Retrieve the thread indices
    wid, tid = wp.tid()

    # Retrieve the number of active dimensions in the world
    n = dim[wid]

    # Compute i (row) and j (col) indices from the tid
    i = tid // n
    j = tid % n

    # Skip if indices exceed the problem size
    if i >= n or j >= n:
        return

    # Retrieve the matrix index offset of the world
    m_0 = mio[wid]

    # Retrieve the vector index offset of the world
    v_0 = vio[wid]

    # Compute the global index of the matrix entry
    m_ij = m_0 + n * i + j

    # Retrieve the ij entry of the input matrix
    X_ij = X[m_ij]

    # Retrieve the i,j entries of the diagonal matrix
    D_i = D[v_0 + i]
    D_j = D[v_0 + j]

    # Compute the i,j entry of the output matrix
    Y[m_ij] = D_i * D_j * X_ij


@wp.kernel
def _mult_left_diag_matrix_with_vector(
    # Inputs:
    dim: wp.array(dtype=int32),
    vio: wp.array(dtype=int32),
    D: wp.array(dtype=float32),
    x: wp.array(dtype=float32),
    # Outputs:
    y: wp.array(dtype=float32),
):
    # Retrieve the thread index
    wid, tid = wp.tid()

    # Retrieve the number of active constraints in the world
    n = dim[wid]

    # Skip if row index exceed the problem size
    if tid >= n:
        return

    # Retrieve the vector index offset of the world
    v_0 = vio[wid]

    # Compute the global index of the vector entry
    v_i = v_0 + tid

    # Retrieve the i-th entry of the input vector
    x_i = x[v_i]

    # Retrieve the i-th entry of the diagonal matrix
    D_i = D[v_i]

    # Compute the i-th entry of the output vector
    y[v_i] = D_i * x_i
