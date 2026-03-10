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
import warp as wp

from ..._src.core.data import DataKamino
from ..._src.core.model import ModelKamino
from ..._src.dynamics.delassus import BlockSparseMatrixFreeDelassusOperator, DelassusOperator
from ..._src.geometry.contacts import ContactsKamino
from ..._src.kinematics.jacobians import DenseSystemJacobians, SparseSystemJacobians
from ..._src.kinematics.limits import LimitsKamino

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


def extract_active_constraint_dims(data: DataKamino) -> list[int]:
    active_dim_np = data.info.num_total_cts.numpy()
    return [int(active_dim_np[i]) for i in range(len(active_dim_np))]


def extract_active_constraint_vectors(model: ModelKamino, data: DataKamino, x: wp.array) -> list[np.ndarray]:
    cts_start_np = model.info.total_cts_offset.numpy()
    num_active_cts_np = extract_active_constraint_dims(data)
    x_np = x.numpy()
    return [x_np[cts_start_np[n] : cts_start_np[n] + num_active_cts_np[n]] for n in range(len(cts_start_np))]


def extract_actuation_forces(model: ModelKamino, data: DataKamino) -> list[np.ndarray]:
    dofs_start_np = model.info.joint_dofs_offset.numpy()
    num_dofs_np = model.info.num_joint_dofs.numpy()
    tau_j_np = data.joints.tau_j.numpy()
    return [tau_j_np[dofs_start_np[n] : dofs_start_np[n] + num_dofs_np[n]] for n in range(len(dofs_start_np))]


def extract_cts_jacobians(
    model: ModelKamino,
    limits: LimitsKamino | None,
    contacts: ContactsKamino | None,
    jacobians: DenseSystemJacobians | SparseSystemJacobians,
    only_active_cts: bool = False,
    verbose: bool = False,
) -> list[np.ndarray]:
    if isinstance(jacobians, SparseSystemJacobians):
        return jacobians._J_cts.bsm.numpy()

    # Retrieve the number of worlds in the model
    num_worlds = model.info.num_worlds

    # Reshape the flat Jacobian as a set of matrices
    J_cts_flat_offsets = jacobians.data.J_cts_offsets.numpy().astype(int).tolist()
    J_cts_flat = jacobians.data.J_cts_data.numpy()
    J_cts_flat_total_size = J_cts_flat.size
    J_cts_flat_sizes = [0] * num_worlds
    J_cts_flat_offsets_ext = [*J_cts_flat_offsets, J_cts_flat_total_size]
    for w in range(num_worlds - 1, -1, -1):
        J_cts_flat_sizes[w] = J_cts_flat_offsets_ext[w + 1] - J_cts_flat_offsets_ext[w]

    # Retrieve the Jacobian dimensions in each world
    has_limits = limits is not None and limits.model_max_limits_host > 0
    has_contacts = contacts is not None and contacts.model_max_contacts_host > 0
    num_bdofs = model.info.num_body_dofs.numpy().tolist()
    num_jcts = model.info.num_joint_cts.numpy().tolist()
    maxnl = limits.world_max_limits_host if has_limits else [0] * num_worlds
    maxnc = contacts.world_max_contacts_host if has_contacts else [0] * num_worlds
    nlact = limits.world_active_limits.numpy().tolist() if has_limits else [0] * num_worlds
    ncact = contacts.world_active_contacts.numpy().tolist() if has_contacts else [0] * num_worlds
    nl = nlact if only_active_cts else maxnl
    nc = ncact if only_active_cts else maxnc

    # Extract each Jacobian as a matrix
    J_cts_mat: list[np.ndarray] = []
    for w in range(num_worlds):
        ncts = num_jcts[w] + nl[w] + 3 * nc[w]
        J_cts_size = ncts * num_bdofs[w]
        if J_cts_size > J_cts_flat_sizes[w]:
            raise ValueError(f"Jacobian size {J_cts_size} exceeds flat size {J_cts_flat_sizes[w]} for world {w}")
        start = J_cts_flat_offsets[w]
        end = J_cts_flat_offsets[w] + J_cts_size
        J_cts_mat.append(J_cts_flat[start:end].reshape((ncts, num_bdofs[w])))

    # Optional verbose output
    if verbose:
        print(f"J_cts_flat_total_size: {J_cts_flat_total_size}")
        print(f"sum(J_cts_flat_sizes): {sum(J_cts_flat_sizes)}")
        print(f"J_cts_flat_sizes: {J_cts_flat_sizes}")
        print(f"J_cts_flat_offsets: {J_cts_flat_offsets}")
        print("")  # Add a newline for better readability
        for w in range(num_worlds):
            print(f"{w}: start={J_cts_flat_offsets[w]}, end={J_cts_flat_offsets[w] + J_cts_flat_sizes[w]}")
            print(f"J_cts_mat[{w}] ({J_cts_mat[w].shape}):\n{J_cts_mat[w]}\n")

    # Return the extracted Jacobians
    return J_cts_mat


def extract_dofs_jacobians(
    model: ModelKamino,
    jacobians: DenseSystemJacobians | SparseSystemJacobians,
    verbose: bool = False,
) -> list[np.ndarray]:
    if isinstance(jacobians, SparseSystemJacobians):
        return jacobians._J_cts.bsm.numpy()

    # Retrieve the number of worlds in the model
    num_worlds = model.info.num_worlds

    # Reshape the flat Jacobian as a set of matrices
    ajmio = jacobians.data.J_dofs_offsets.numpy()
    J_dofs_flat = jacobians.data.J_dofs_data.numpy()
    J_dofs_flat_total_size = J_dofs_flat.size
    J_dofs_flat_offsets = [int(ajmio[i]) for i in range(num_worlds)]
    J_dofs_flat_sizes = [0] * num_worlds
    J_dofs_flat_offsets_ext = [*J_dofs_flat_offsets, J_dofs_flat_total_size]
    for i in range(num_worlds - 1, -1, -1):
        J_dofs_flat_sizes[i] = J_dofs_flat_offsets_ext[i + 1] - J_dofs_flat_offsets_ext[i]

    # Extract each Jacobian as a matrix
    num_bdofs = model.info.num_body_dofs.numpy().tolist()
    num_jdofs = model.info.num_joint_dofs.numpy().tolist()
    J_dofs_mat: list[np.ndarray] = []
    for i in range(num_worlds):
        start = J_dofs_flat_offsets[i]
        end = J_dofs_flat_offsets[i] + J_dofs_flat_sizes[i]
        J_dofs_mat.append(J_dofs_flat[start:end].reshape((num_jdofs[i], num_bdofs[i])))

    # Optional verbose output
    if verbose:
        print(f"J_dofs_flat_total_size: {J_dofs_flat_total_size}")
        print(f"sum(J_dofs_flat_sizes): {sum(J_dofs_flat_sizes)}")
        print(f"J_dofs_flat_sizes: {J_dofs_flat_sizes}")
        print(f"J_dofs_flat_offsets: {J_dofs_flat_offsets}")
        print("")  # Add a newline for better readability
        for i in range(num_worlds):
            print(f"{i}: start={J_dofs_flat_offsets[i]}, end={J_dofs_flat_offsets[i] + J_dofs_flat_sizes[i]}")
            print(f"J_dofs_mat[{i}] ({J_dofs_mat[i].shape}):\n{J_dofs_mat[i]}\n")

    # Return the extracted Jacobians
    return J_dofs_mat


def extract_delassus(
    delassus: DelassusOperator | BlockSparseMatrixFreeDelassusOperator,
    only_active_dims: bool = False,
) -> list[np.ndarray]:
    if isinstance(delassus, BlockSparseMatrixFreeDelassusOperator):
        return extract_delassus_sparse(delassus=delassus, only_active_dims=only_active_dims)

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


def extract_delassus_sparse(
    delassus: BlockSparseMatrixFreeDelassusOperator, only_active_dims: bool = False
) -> list[np.ndarray]:
    """
    Extracts the (dense) Delassus matrix from the sparse matrix-free Delassus operator by querying
    individual matrix columns.
    """
    num_worlds = delassus._model.size.num_worlds
    sum_max_cts = delassus._model.size.sum_of_max_total_cts
    max_cts_np = delassus._model.info.max_total_cts.numpy()

    num_cts = delassus._data.info.num_total_cts
    num_cts_np = num_cts.numpy()
    max_dim = np.max(num_cts_np) if only_active_dims else np.max(max_cts_np)

    D_mat: list[np.ndarray] = []
    for world_id in range(num_worlds):
        if only_active_dims:
            D_mat.append(np.zeros((num_cts_np[world_id], num_cts_np[world_id]), dtype=np.float32))
        else:
            D_mat.append(np.zeros((max_cts_np[world_id], max_cts_np[world_id]), dtype=np.float32))

    vec_query = wp.empty((sum_max_cts,), dtype=wp.float32, device=delassus._device)
    vec_response = wp.empty((sum_max_cts,), dtype=wp.float32, device=delassus._device)

    @wp.kernel
    def _set_unit_entry(
        # Inputs:
        index: int,
        world_dim: wp.array(dtype=wp.int32),
        entry_start: wp.array(dtype=wp.int32),
        # Output:
        x: wp.array(dtype=wp.float32),
    ):
        world_id = wp.tid()

        if index >= world_dim[world_id]:
            return

        x[entry_start[world_id] + index] = 1.0

    entry_start_np = delassus.bsm.row_start.numpy()

    world_mask = wp.ones((num_worlds,), dtype=wp.int32, device=delassus._device)

    for dim in range(max_dim):
        # Query the operator by computing the product with a vector where only entry `dim` is set to 1.
        vec_query.zero_()
        wp.launch(
            kernel=_set_unit_entry,
            dim=num_worlds,
            inputs=[
                # Inputs:
                dim,
                num_cts,
                delassus.bsm.row_start,
                # Outputs:
                vec_query,
            ],
        )
        delassus.matvec(vec_query, vec_response, world_mask)
        vec_response_np = vec_response.numpy()

        # Set the response as the corresponding column of each matrix
        for world_id in range(num_worlds):
            D_mat_dim = D_mat[world_id].shape[0]
            if dim >= D_mat_dim:
                continue
            start_idx = entry_start_np[world_id]
            D_mat[world_id][:, dim] = vec_response_np[start_idx : start_idx + D_mat_dim]

    return D_mat


def extract_problem_vector(
    delassus: DelassusOperator | BlockSparseMatrixFreeDelassusOperator,
    vector: np.ndarray,
    only_active_dims: bool = False,
) -> list[np.ndarray]:
    maxdim_wp_np = delassus.info.maxdim.numpy()
    dim_wp_np = delassus.info.dim.numpy()
    vio_wp_np = delassus.info.vio.numpy()

    num_worlds = delassus.num_worlds if isinstance(delassus, DelassusOperator) else delassus.num_matrices

    # Extract each vector for each world
    vectors_np: list[np.ndarray] = []
    for i in range(num_worlds):
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
