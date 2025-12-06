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

"""Utilities for extracting data from Kamino data structures"""

import numpy as np

from ...dynamics.delassus import DelassusOperator
from ...kinematics.jacobians import DenseSystemJacobians

###
# Helper functions
###


def get_matrix_block(index: int, flatmat: np.ndarray, dims: list[int], maxdims: list[int] | None = None) -> np.ndarray:
    """Extract a specific matrix block from a flattened array of matrices."""
    if maxdims is None:
        maxdims = dims
    mat_shape = (dims[index], dims[index])
    mat_start = sum(n * n for n in maxdims[:index])
    mat_end = mat_start + dims[index] ** 2
    return flatmat[mat_start:mat_end].reshape(mat_shape)


def get_vector_block(index: int, flatvec: np.ndarray, dims: list[int], maxdims: list[int] | None = None) -> np.ndarray:
    """Extract a specific matrix block from a flattened array of matrices."""
    if maxdims is None:
        maxdims = dims
    vec_start = sum(maxdims[:index])
    vec_end = vec_start + dims[index]
    return flatvec[vec_start:vec_end]


###
# Helper functions
###


def extract_active_constraint_dims(delassus: DelassusOperator) -> list[int]:
    # Extract the active constraint dimensions
    active_dim_np = delassus.info.dim.numpy()
    active_dims = [int(active_dim_np[i]) for i in range(len(active_dim_np))]
    return active_dims


def extract_cts_jacobians(
    jacobians: DenseSystemJacobians,
    num_bodies: list[int],
    active_dims: list[int] | None = None,
) -> list[np.ndarray]:
    # Reshape the flat Jacobian as a set of matrices
    num_body_dofs = [6 * num_bodies[i] for i in range(len(num_bodies))]
    cjmio = jacobians.data.J_cts_offsets.numpy()
    num_jacobians = int(cjmio.size)
    J_cts_flat = jacobians.data.J_cts_data.numpy()
    J_cts_flat_total_size = J_cts_flat.size
    J_cts_flat_offsets = [int(cjmio[i]) for i in range(num_jacobians)]
    J_cts_flat_sizes = [0] * num_jacobians
    J_cts_flat_offsets_ext = [*J_cts_flat_offsets, J_cts_flat_total_size]
    J_cts_flat_shapes = [(0, 0)] * num_jacobians
    for i in range(num_jacobians - 1, -1, -1):
        J_cts_flat_sizes[i] = J_cts_flat_offsets_ext[i + 1] - J_cts_flat_offsets_ext[i]
        nbd_i = num_body_dofs[i]
        J_cts_flat_shapes[i] = (J_cts_flat_sizes[i] // nbd_i, nbd_i)

    # Extract each Jacobian as a matrix
    J_cts_mat: list[np.ndarray] = []
    for i in range(num_jacobians):
        if active_dims is not None and len(active_dims) > 0:
            J_rows = active_dims[i]
        else:
            J_rows = J_cts_flat_shapes[i][0]
        J_cols = J_cts_flat_shapes[i][1]
        J_cts_mat.append(
            J_cts_flat[J_cts_flat_offsets[i] : J_cts_flat_offsets[i] + J_cts_flat_sizes[i]].reshape(
                J_cts_flat_shapes[i]
            )[:J_rows, :J_cols]
        )

    # Return the list of Jacobian matrices
    return J_cts_mat


def extract_dofs_jacobians(
    jacobians: DenseSystemJacobians,
    num_body_dofs: list[int],
    active_dims: list[int] | None = None,
) -> list[np.ndarray]:
    # Reshape the flat Jacobian as a set of matrices
    ajmio = jacobians.data.J_dofs_offsets.numpy()
    num_jacobians = int(ajmio.size)
    J_dofs_flat = jacobians.data.J_dofs_data.numpy()
    J_dofs_flat_total_size = J_dofs_flat.size
    J_dofs_flat_offsets = [int(ajmio[i]) for i in range(num_jacobians)]
    J_dofs_flat_sizes = [0] * num_jacobians
    J_dofs_flat_offsets_ext = [*J_dofs_flat_offsets, J_dofs_flat_total_size]
    J_dofs_flat_shapes = [(0, 0)] * num_jacobians
    for i in range(num_jacobians - 1, -1, -1):
        J_dofs_flat_sizes[i] = J_dofs_flat_offsets_ext[i + 1] - J_dofs_flat_offsets_ext[i]
        nbd_i = num_body_dofs[i]
        J_dofs_flat_shapes[i] = (J_dofs_flat_sizes[i] // nbd_i, nbd_i)

    # Extract each Jacobian as a matrix
    J_cts_mat: list[np.ndarray] = []
    for i in range(num_jacobians):
        if active_dims is not None and len(active_dims) > 0:
            J_rows = active_dims[i]
        else:
            J_rows = J_dofs_flat_shapes[i][0]
        J_cols = J_dofs_flat_shapes[i][1]
        J_cts_mat.append(
            J_dofs_flat[J_dofs_flat_offsets[i] : J_dofs_flat_offsets[i] + J_dofs_flat_sizes[i]].reshape(
                J_dofs_flat_shapes[i]
            )[:J_rows, :J_cols]
        )

    # Return the list of Jacobian matrices
    return J_cts_mat


def extract_delassus(delassus: DelassusOperator, only_active_dims: bool = False) -> list[np.ndarray]:
    maxdim_wp_np = delassus.info.maxdim.numpy()
    dim_wp_np = delassus.info.dim.numpy()
    mio_wp_np = delassus.info.mio.numpy()
    D_wp_np = delassus.D.numpy()

    # Extract each Delassus matrix for each world
    D_mat: list[np.ndarray] = []
    for i in range(delassus.num_worlds):
        D_maxdim = maxdim_wp_np[i]
        D_start = mio_wp_np[i]
        if only_active_dims:
            D_dim = dim_wp_np[i]
        else:
            D_dim = D_maxdim
        D_end = D_start + D_dim * D_dim
        D_mat.append(D_wp_np[D_start:D_end].reshape((D_dim, D_dim)))

    # Return the list of Delassus matrices
    return D_mat


def extract_problem_vector(
    delassus: DelassusOperator, vector: np.ndarray, only_active_dims: bool = False
) -> list[np.ndarray]:
    maxdim_wp_np = delassus.info.maxdim.numpy()
    dim_wp_np = delassus.info.dim.numpy()
    vio_wp_np = delassus.info.vio.numpy()

    # Extract each vector for each world
    vectors_np: list[np.ndarray] = []
    for i in range(delassus.num_worlds):
        vec_maxdim = maxdim_wp_np[i]
        vec_start = vio_wp_np[i]
        vec_end = vec_start + vec_maxdim
        if only_active_dims:
            vec_end = vec_start + dim_wp_np[i]
        else:
            vec_end = vec_start + vec_maxdim
        vectors_np.append(vector[vec_start:vec_end])

    # Return the list of Delassus matrices
    return vectors_np


def extract_info_vectors(offsets: np.ndarray, vectors: np.ndarray, dims: list[int] | None = None) -> list[np.ndarray]:
    # Determine vector sizes
    nv = offsets.size
    maxn = vectors.size // nv
    n = dims if dims is not None and len(dims) == nv else [maxn] * nv

    # Extract each vector for each world
    vectors_list: list[np.ndarray] = []
    for i in range(nv):
        vec_start = offsets[i]
        vec_end = vec_start + n[i]
        vectors_list.append(vectors[vec_start:vec_end])

    # Return the list of Delassus matrices
    return vectors_list
