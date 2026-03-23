# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""
Provides containers to represent and operate Delassus operators.

A Delassus operator is a symmetric semi-positive-definite matrix that
represents the apparent inertia within the space defined by the set of
active constraints imposed on a constrained rigid multi-body system.

This module thus provides building-blocks to realize Delassus operators across multiple
worlds contained in a :class:`ModelKamino`. The :class:`DelassusOperator` class provides a
high-level interface to encapsulate both the data representation as well as the
relevant operations. It provides methods to allocate the necessary data arrays, build
the Delassus matrix given the current state of the model and the active constraints,
add diagonal regularization, and solve linear systems of the form `D @ x = v` given
arrays holding the right-hand-side (rhs) vectors v. Moreover, it supports the use of
different linear solvers as a back-end for performing the aforementioned linear system
solve. Construction of the Delassus operator is realized using a set of Warp kernels
that parallelize the computation using various strategies.

Typical usage example:
    # Create a model builder and add bodies, joints, geoms, etc.
    builder = ModelBuilder()
    ...

    # Create a model from the builder and construct additional
    # containers to hold joint-limits, contacts, Jacobians
    model = builder.finalize()
    data = model.data()
    limits = LimitsKamino(model)
    contacts = ContactsKamino(builder)
    jacobians = DenseSystemJacobians(model, limits, contacts)

    # Define a linear solver type to use as a back-end for the
    # Delassus operator computations such as factorization and
    # solving the linear system when a rhs vector is provided
    linear_solver = LLTBlockedSolver
    ...

    # Build the Jacobians for the model and active limits and contacts
    jacobians.build(model, data, limits, contacts)
    ...

    # Create a Delassus operator and build it using the current model data
    # and active unilateral constraints (i.e. for limits and contacts).
    delassus = DelassusOperator(model, limits, contacts, linear_solver)
    delassus.build(model, data, jacobians)

    # Add diagonal regularization the Delassus matrix
    eta = ...
    delassus.regularize(eta=eta)

    # Factorize the Delassus matrix using the Cholesky factorization
    delassus.compute()

    # Solve a linear system using the Delassus operator
    rhs = ...
    solution = ...
    delassus.solve(b=rhs, x=solution)
"""

import copy
import functools
from typing import Any

import numpy as np
import warp as wp

from ..core.data import DataKamino
from ..core.model import ModelKamino
from ..core.size import SizeKamino
from ..core.types import FloatType, float32, int32, mat33f, vec3f, vec6f
from ..geometry.contacts import ContactsKamino
from ..kinematics.constraints import get_max_constraints_per_world
from ..kinematics.jacobians import ColMajorSparseConstraintJacobians, DenseSystemJacobians, SparseSystemJacobians
from ..kinematics.limits import LimitsKamino
from ..linalg import DenseLinearOperatorData, DenseSquareMultiLinearInfo, LinearSolverType
from ..linalg.linear import IterativeSolver
from ..linalg.sparse_matrix import BlockDType, BlockSparseMatrices
from ..linalg.sparse_operator import BlockSparseLinearOperators

###
# Module interface
###

__all__ = [
    "BlockSparseMatrixFreeDelassusOperator",
    "DelassusOperator",
]


###
# Module configs
###

wp.set_module_options({"enable_backward": False})


###
# Kernels
###


@wp.kernel
def _build_delassus_elementwise_dense(
    # Inputs:
    model_info_num_bodies: wp.array(dtype=int32),
    model_info_bodies_offset: wp.array(dtype=int32),
    model_bodies_inv_m_i: wp.array(dtype=float32),
    data_bodies_inv_I_i: wp.array(dtype=mat33f),
    jacobians_cts_offset: wp.array(dtype=int32),
    jacobians_cts_data: wp.array(dtype=float32),
    delassus_dim: wp.array(dtype=int32),
    delassus_mio: wp.array(dtype=int32),
    # Outputs:
    delassus_D: wp.array(dtype=float32),
):
    # Retrieve the thread index as the world index and Delassus element index
    wid, tid = wp.tid()

    # Retrieve the world dimensions
    nb = model_info_num_bodies[wid]
    bio = model_info_bodies_offset[wid]

    # Retrieve the problem dimensions
    ncts = delassus_dim[wid]

    # Skip if world has no constraints
    if ncts == 0:
        return

    # Compute i (row) and j (col) indices from the tid
    i = tid // ncts
    j = tid % ncts

    # Skip if indices exceed the problem size
    if i >= ncts or j >= ncts:
        return

    # Retrieve the world's matrix offsets
    dmio = delassus_mio[wid]
    cjmio = jacobians_cts_offset[wid]

    # Compute the number of body DoFs of the world
    nbd = 6 * nb

    # Buffers
    # tmp = vec3f(0.0)
    Jv_i = vec3f(0.0)
    Jv_j = vec3f(0.0)
    Jw_i = vec3f(0.0)
    Jw_j = vec3f(0.0)
    D_ij = float32(0.0)
    D_ji = float32(0.0)

    # Loop over rigid body blocks
    # NOTE: k is the body index w.r.t the world
    for k in range(nb):
        # Body index (bid) of body k w.r.t the model
        bid_k = bio + k
        # DoF index offset (dio) of body k in the flattened Jacobian matrix
        # NOTE: Equivalent to the column index in the matrix-form of the Jacobian matrix
        dio_k = 6 * k
        # Jacobian index offsets
        jio_ik = cjmio + nbd * i + dio_k
        jio_jk = cjmio + nbd * j + dio_k

        # Load the Jacobian blocks of body k
        for d in range(3):
            # Load the i-th row block
            Jv_i[d] = jacobians_cts_data[jio_ik + d]
            Jw_i[d] = jacobians_cts_data[jio_ik + d + 3]
            # Load the j-th row block
            Jv_j[d] = jacobians_cts_data[jio_jk + d]
            Jw_j[d] = jacobians_cts_data[jio_jk + d + 3]

        # Linear term: inv_m_k * dot(Jv_i, Jv_j)
        inv_m_k = model_bodies_inv_m_i[bid_k]
        lin_ij = inv_m_k * wp.dot(Jv_i, Jv_j)
        lin_ji = inv_m_k * wp.dot(Jv_j, Jv_i)

        # Angular term: dot(Jw_i.T * I_k, Jw_j)
        inv_I_k = data_bodies_inv_I_i[bid_k]
        ang_ij = wp.dot(Jw_i, inv_I_k @ Jw_j)
        ang_ji = wp.dot(Jw_j, inv_I_k @ Jw_i)

        # Accumulate
        D_ij += lin_ij + ang_ij
        D_ji += lin_ji + ang_ji

    # Store the result in the Delassus matrix
    delassus_D[dmio + ncts * i + j] = 0.5 * (D_ij + D_ji)


@wp.kernel
def _build_delassus_elementwise_sparse(
    # Inputs:
    model_info_bodies_offset: wp.array(dtype=int32),
    model_bodies_inv_m_i: wp.array(dtype=float32),
    data_bodies_inv_I_i: wp.array(dtype=mat33f),
    jacobian_cts_num_nzb: wp.array(dtype=int32),
    jacobian_cts_nzb_start: wp.array(dtype=int32),
    jacobian_cts_nzb_coords: wp.array2d(dtype=int32),
    jacobian_cts_nzb_values: wp.array(dtype=vec6f),
    delassus_dim: wp.array(dtype=int32),
    delassus_mio: wp.array(dtype=int32),
    # Outputs:
    delassus_D: wp.array(dtype=float32),
):
    # Retrieve the thread index as the world index and Jacobian block index pair
    wid, tid = wp.tid()

    # Retrieve the problem dimensions
    ncts = delassus_dim[wid]

    # Skip if world has no constraints
    if ncts == 0:
        return

    # Retrieve the world dimensions
    bio = model_info_bodies_offset[wid]

    # Retrieve the number of non-zero blocks
    num_nzb = jacobian_cts_num_nzb[wid]

    # Compute Jacobian block indices from the tid
    block_id_i = tid // num_nzb
    block_id_j = tid % num_nzb

    # Skip if index exceeds problem size
    if block_id_i >= num_nzb:
        return

    nzb_start = jacobian_cts_nzb_start[wid]
    global_block_id_i = nzb_start + block_id_i
    global_block_id_j = nzb_start + block_id_j

    # Get block coordinates
    block_coords_i = jacobian_cts_nzb_coords[global_block_id_i]
    block_coords_j = jacobian_cts_nzb_coords[global_block_id_j]

    # Skip if blocks don't affect the same body
    if block_coords_i[1] != block_coords_j[1]:
        return

    # Body index (bid) of body k w.r.t the model, from Jacobian block coords
    bid_k = bio + block_coords_i[1] // 6

    # Get block values
    block_i = jacobian_cts_nzb_values[global_block_id_i]
    block_j = jacobian_cts_nzb_values[global_block_id_j]

    # Retrieve the world's matrix offsets
    dmio = delassus_mio[wid]

    # Load the Jacobian blocks components for body
    Jv_i = vec3f(block_i[0], block_i[1], block_i[2])
    Jv_j = vec3f(block_j[0], block_j[1], block_j[2])
    Jw_i = vec3f(block_i[3], block_i[4], block_i[5])
    Jw_j = vec3f(block_j[3], block_j[4], block_j[5])

    # Linear term: inv_m_k * dot(Jv_i, Jv_j)
    inv_m_k = model_bodies_inv_m_i[bid_k]
    lin_ij = inv_m_k * wp.dot(Jv_i, Jv_j)
    lin_ji = inv_m_k * wp.dot(Jv_j, Jv_i)

    # Angular term: dot(Jw_i.T * I_k, Jw_j)
    inv_I_k = data_bodies_inv_I_i[bid_k]
    ang_ij = wp.dot(Jw_i, inv_I_k @ Jw_j)
    ang_ji = wp.dot(Jw_j, inv_I_k @ Jw_i)

    # Compute sum
    D_ij = lin_ij + ang_ij
    D_ji = lin_ji + ang_ji

    # Store the result in the Delassus matrix
    wp.atomic_add(delassus_D, dmio + ncts * block_coords_i[0] + block_coords_j[0], 0.5 * (D_ij + D_ji))


@wp.kernel
def _add_joint_armature_diagonal_regularization_dense(
    # Inputs:
    model_info_num_joint_dynamic_cts: wp.array(dtype=int32),
    model_info_joint_dynamic_cts_offset: wp.array(dtype=int32),
    model_joint_inv_m_j: wp.array(dtype=float32),
    delassus_dim: wp.array(dtype=int32),
    delassus_mio: wp.array(dtype=int32),
    # Outputs:
    delassus_D: wp.array(dtype=float32),
):
    # Retrieve the thread index as the world index and Delassus element index
    wid, tid = wp.tid()

    # Retrieve the world dimensions
    num_joint_dyn_cts = model_info_num_joint_dynamic_cts[wid]

    # Skip if world has no dynamic joint constraints or indices exceed the problem size
    if num_joint_dyn_cts == 0 or tid >= num_joint_dyn_cts:
        return

    # Retrieve the world's Delassus matrix dimension and offset
    ncts = delassus_dim[wid]
    dmio = delassus_mio[wid]

    # Retrieve the dynamic constraint index offset of the world
    world_joint_dynamic_cts_offset = model_info_joint_dynamic_cts_offset[wid]

    # Retrieve the joint's inverse mass for armature regularization
    inv_m_j = model_joint_inv_m_j[world_joint_dynamic_cts_offset + tid]

    # Add the armature regularization to the diagonal element of the Delassus matrix
    delassus_D[dmio + ncts * tid + tid] += inv_m_j


@wp.kernel
def _regularize_delassus_diagonal_dense(
    # Inputs:
    delassus_dim: wp.array(dtype=int32),
    delassus_vio: wp.array(dtype=int32),
    delassus_mio: wp.array(dtype=int32),
    eta: wp.array(dtype=float32),
    # Outputs:
    delassus_D: wp.array(dtype=float32),
):
    # Retrieve the thread index
    wid, tid = wp.tid()

    # Retrieve the problem dimensions and matrix block index offset
    dim = delassus_dim[wid]
    vio = delassus_vio[wid]
    mio = delassus_mio[wid]

    # Skip if row index exceed the problem size
    if tid >= dim:
        return

    # Regularize the diagonal element
    delassus_D[mio + dim * tid + tid] += eta[vio + tid]


@wp.kernel
def _merge_inv_mass_matrix_kernel(
    model_info_bodies_offset: wp.array(dtype=int32),
    model_bodies_inv_m_i: wp.array(dtype=float32),
    data_bodies_inv_I_i: wp.array(dtype=mat33f),
    num_nzb: wp.array(dtype=int32),
    nzb_start: wp.array(dtype=int32),
    nzb_coords: wp.array2d(dtype=int32),
    nzb_values: wp.array(dtype=vec6f),
):
    """
    Kernel to merge the inverse mass matrix into an existing sparse matrix, so that the resulting
    matrix is given as `A <- A @ M^-1`.
    """
    mat_id, block_idx = wp.tid()

    # Check if block index is valid for this matrix.
    if block_idx >= num_nzb[mat_id]:
        return

    global_block_idx = nzb_start[mat_id] + block_idx
    block_coord = nzb_coords[global_block_idx]
    block = nzb_values[global_block_idx]

    body_id = block_coord[1] // 6

    # Index of body w.r.t the model
    global_body_id = model_info_bodies_offset[mat_id] + body_id

    # Load the inverse mass and inverse inertia for this body
    inv_m = model_bodies_inv_m_i[global_body_id]
    inv_I = data_bodies_inv_I_i[global_body_id]

    # Apply inverse mass matrices to Jacobian block
    v = inv_m * vec3f(block[0], block[1], block[2])
    w = inv_I @ vec3f(block[3], block[4], block[5])

    # Write back values
    block[0] = v[0]
    block[1] = v[1]
    block[2] = v[2]
    block[3] = w[0]
    block[4] = w[1]
    block[5] = w[2]
    nzb_values[global_block_idx] = block


@functools.cache
def _make_merge_preconditioner_kernel(block_type: BlockDType):
    """
    Generates a kernel to merge the (diagonal) preconditioning into a sparse matrix.
    This effectively applies the preconditioning to the left (row-space) of the Jacobian.
    """
    # Determine (static) block size for kernel.
    block_shape = block_type.shape
    if isinstance(block_type.shape, int):
        block_shape = (block_shape, block_shape)
    elif len(block_shape) == 0:
        block_shape = (1, 1)
    elif len(block_shape) == 1:
        block_shape = (1, block_shape[0])

    @wp.kernel
    def merge_preconditioner_kernel(
        # Inputs:
        num_nzb: wp.array(dtype=int32),
        nzb_start: wp.array(dtype=int32),
        nzb_coords: wp.array2d(dtype=int32),
        row_start: wp.array(dtype=int32),
        preconditioner: wp.array(dtype=float32),
        # Outputs:
        nzb_values: wp.array(dtype=block_type.warp_type),
    ):
        mat_id, block_idx = wp.tid()

        # Check if block index is valid for this matrix.
        if block_idx >= num_nzb[mat_id]:
            return

        n_block_rows = wp.static(block_shape[0])
        n_block_cols = wp.static(block_shape[1])

        global_block_idx = nzb_start[mat_id] + block_idx
        block_coord = nzb_coords[global_block_idx]
        block = nzb_values[global_block_idx]

        if wp.static(n_block_rows == 1):
            vec_coord = block_coord[0] + row_start[mat_id]
            p_value = preconditioner[vec_coord]
            block = block * p_value

        else:
            vec_coord_start = block_coord[0] + row_start[mat_id]
            for i in range(n_block_rows):
                p_value = preconditioner[vec_coord_start + i]
                for j in range(n_block_cols):
                    block[i, j] = block[i, j] * p_value

        nzb_values[global_block_idx] = block

    return merge_preconditioner_kernel


@wp.kernel
def _add_armature_regularization_sparse(
    # Inputs:
    model_info_num_joint_dynamic_cts: wp.array(dtype=int32),
    model_info_joint_dynamic_cts_offset: wp.array(dtype=int32),
    row_start: wp.array(dtype=int32),
    model_joint_inv_m_j: wp.array(dtype=float32),
    # Outputs:
    combined_regularization: wp.array(dtype=float32),
):
    # Retrieve the thread index as the world index and joint dynamics index
    wid, tid = wp.tid()

    # Retrieve the world dimensions
    num_joint_dyn_cts = model_info_num_joint_dynamic_cts[wid]

    # Skip if world has no dynamic joint constraints or indices exceed the problem size
    if num_joint_dyn_cts == 0 or tid >= num_joint_dyn_cts:
        return

    # Retrieve the dynamic constraint index offset of the world
    world_joint_dynamic_cts_offset = model_info_joint_dynamic_cts_offset[wid]

    # Retrieve the joint's inverse mass for armature regularization
    inv_m_j = model_joint_inv_m_j[world_joint_dynamic_cts_offset + tid]

    # Get the index into the regularization
    vec_id = row_start[wid] + tid

    # Add the armature regularization
    combined_regularization[vec_id] += inv_m_j


@wp.kernel
def _add_armature_regularization_preconditioned_sparse(
    # Inputs:
    model_info_num_joint_dynamic_cts: wp.array(dtype=int32),
    model_info_joint_dynamic_cts_offset: wp.array(dtype=int32),
    model_joint_inv_m_j: wp.array(dtype=float32),
    row_start: wp.array(dtype=int32),
    preconditioner: wp.array(dtype=float32),
    # Outputs:
    combined_regularization: wp.array(dtype=float32),
):
    # Retrieve the thread index as the world index and joint dynamics index
    wid, tid = wp.tid()

    # Retrieve the world dimensions
    num_joint_dyn_cts = model_info_num_joint_dynamic_cts[wid]

    # Skip if world has no dynamic joint constraints or indices exceed the problem size
    if num_joint_dyn_cts == 0 or tid >= num_joint_dyn_cts:
        return

    # Retrieve the dynamic constraint index offset of the world
    world_joint_dynamic_cts_offset = model_info_joint_dynamic_cts_offset[wid]

    # Retrieve the joint's inverse mass for armature regularization
    inv_m_j = model_joint_inv_m_j[world_joint_dynamic_cts_offset + tid]

    # Get the index into the preconditioner and regularization
    vec_id = row_start[wid] + tid

    # Retrieve preconditioner value
    p = preconditioner[vec_id]

    # Add the armature regularization
    combined_regularization[vec_id] += p * p * inv_m_j


@wp.kernel
def _compute_block_sparse_delassus_diagonal(
    # Inputs:
    model_info_bodies_offset: wp.array(dtype=int32),
    model_bodies_inv_m_i: wp.array(dtype=float32),
    data_bodies_inv_I_i: wp.array(dtype=mat33f),
    bsm_nzb_start: wp.array(dtype=int32),
    bsm_num_nzb: wp.array(dtype=int32),
    bsm_nzb_coords: wp.array2d(dtype=int32),
    bsm_nzb_values: wp.array(dtype=vec6f),
    vec_start: wp.array(dtype=int32),
    # Outputs:
    diag: wp.array(dtype=float32),
):
    """
    Computes the diagonal entries of the Delassus matrix by summing up the contributions of each
    non-zero block of the Jacobian: D_ii = sum_k J_ik @ M_kk^-1 @ (J_ik)^T

    This kernel processes one non-zero block per thread and accumulates all contributions.
    """
    # Retrieve the thread index as the world index and block index
    world_id, block_idx_local = wp.tid()

    # Skip if block index exceeds the number of non-zero blocks
    if block_idx_local >= bsm_num_nzb[world_id]:
        return

    # Compute the global block index
    block_idx = bsm_nzb_start[world_id] + block_idx_local

    # Get the row and column for this block
    row = bsm_nzb_coords[block_idx, 0]
    col = bsm_nzb_coords[block_idx, 1]

    # Get the body index offset for this world
    body_index_offset = model_info_bodies_offset[world_id]

    # Get the Jacobian block and extract linear and angular components
    J_block = bsm_nzb_values[block_idx]
    Jv = J_block[0:3]
    Jw = J_block[3:6]

    # Get the body index from the column
    body_idx = col // 6
    body_idx_global = body_index_offset + body_idx

    # Load the inverse mass and inverse inertia for this body
    inv_m = model_bodies_inv_m_i[body_idx_global]
    inv_I = data_bodies_inv_I_i[body_idx_global]

    # Compute linear contribution: Jv^T @ inv_m @ Jv
    diag_kk = inv_m * wp.dot(Jv, Jv)

    # Compute angular contribution: Jw^T @ inv_I @ Jw
    diag_kk += wp.dot(Jw, inv_I @ Jw)

    # Atomically add contribution to the diagonal element
    wp.atomic_add(diag, vec_start[world_id] + row, diag_kk)


@wp.kernel
def _add_matrix_diag_product(
    model_data_num_total_cts: wp.array(dtype=int32),
    row_start: wp.array(dtype=int32),
    d: wp.array(dtype=float32),
    x: wp.array(dtype=float32),
    y: wp.array(dtype=float32),
    alpha: float,
    world_mask: wp.array(dtype=int32),
):
    """
    Adds the product of a vector with a diagonal matrix to another vector: y += alpha * diag(d) @ x
    This is used to apply a regularization to the Delassus matrix-vector product.
    """
    # Retrieve the thread index as the world index and constraint index
    world_id, ct_id = wp.tid()

    # Terminate early if world or constraint is inactive
    if world_mask[world_id] == 0 or ct_id >= model_data_num_total_cts[world_id]:
        return

    idx = row_start[world_id] + ct_id
    y[idx] += alpha * d[idx] * x[idx]


@wp.kernel
def _scale_row_vector_kernel(
    # Matrix data:
    matrix_dims: wp.array2d(dtype=int32),
    # Vector block offsets:
    row_start: wp.array(dtype=int32),
    # Inputs:
    x: wp.array(dtype=Any),
    beta: Any,
    # Mask:
    matrix_mask: wp.array(dtype=int32),
):
    """
    Computes a vector scaling for all active entries: y = beta * y
    """
    mat_id, entry_id = wp.tid()

    # Early exit if the matrix is flagged as inactive.
    if matrix_mask[mat_id] == 0 or entry_id >= matrix_dims[mat_id, 0]:
        return

    idx = row_start[mat_id] + entry_id
    x[idx] = beta * x[idx]


@functools.cache
def _make_block_sparse_gemv_regularization_kernel(alpha: float32):
    # Note: this kernel factory allows to optimize for the common case alpha = 1.0. In use cases where
    # alpha changes over time, this would need to be revisited (to avoid multiple recompilations)
    @wp.kernel
    def _block_sparse_gemv_regularization_kernel(
        # Matrix data:
        dims: wp.array2d(dtype=int32),
        num_nzb: wp.array(dtype=int32),
        nzb_start: wp.array(dtype=int32),
        nzb_coords: wp.array2d(dtype=int32),
        nzb_values: wp.array(dtype=vec6f),
        # Vector block offsets:
        row_start: wp.array(dtype=int32),
        col_start: wp.array(dtype=int32),
        # Regularization:
        eta: wp.array(dtype=float32),
        # Vector:
        x: wp.array(dtype=float32),
        y: wp.array(dtype=float32),
        z: wp.array(dtype=float32),
        # Mask:
        matrix_mask: wp.array(dtype=int32),
    ):
        """
        Computes a generalized matrix-vector product with an added diagonal regularization component:
            y <- y + alpha * (M @ x) + alpha * (diag(eta) @ z)
        """

        mat_id, block_idx = wp.tid()

        # Early exit if the matrix is flagged as inactive.
        if matrix_mask[mat_id] == 0:
            return

        # Check if block index is valid for this matrix.
        if block_idx >= num_nzb[mat_id]:
            return

        global_block_idx = nzb_start[mat_id] + block_idx
        block_coord = nzb_coords[global_block_idx]
        block = nzb_values[global_block_idx]

        # Perform block matrix-vector multiplication: z += alpha * A_block @ x_block
        x_idx_base = col_start[mat_id] + block_coord[1]
        acc = float32(0.0)

        for j in range(6):
            acc += block[j] * x[x_idx_base + j]
        if wp.static(alpha != 1.0):
            acc *= alpha

        wp.atomic_add(y, row_start[mat_id] + block_coord[0], acc)

        if block_idx < dims[mat_id][0]:
            vec_idx = row_start[mat_id] + block_idx
            if wp.static(alpha != 1.0):
                vec_val = z[vec_idx] * alpha * eta[vec_idx]
            else:
                vec_val = z[vec_idx] * eta[vec_idx]
            wp.atomic_add(y, vec_idx, vec_val)

    return _block_sparse_gemv_regularization_kernel


###
# Interfaces
###


class DelassusOperator:
    """
    A container to represent the Delassus matrix operator.
    """

    def __init__(
        self,
        model: ModelKamino | None = None,
        data: DataKamino | None = None,
        limits: LimitsKamino | None = None,
        contacts: ContactsKamino | None = None,
        solver: LinearSolverType = None,
        solver_kwargs: dict[str, Any] | None = None,
        device: wp.DeviceLike = None,
    ):
        """
        Creates a Delassus operator for the given model, limits and contacts containers.

        This class also supports deferred allocation, i.e. it can be initialized without
        a model, limits, or contacts, and the allocation can be performed later using the
        `allocate` method. This is useful for scenarios where the model or constraints are
        not known at the time of Delassus operator creation, but will be available later.

        The dimension of a Delassus matrix is defined as the sum over active
        joint, limit, and contact constraints, and the maximum dimension is
        the maximum number of constraints that can be active in each world.

        Args:
            model (ModelKamino): The model container for which the Delassus operator is built.
            data (DataKamino, optional): The model data container holding the state info and data.
            limits (LimitsKamino, optional): The container holding the allocated joint-limit data.
            contacts (ContactsKamino, optional): The container holding the allocated contacts data.
            device (wp.DeviceLike, optional): The device identifier for the Delassus operator. Defaults to None.
            factorizer (CholeskyFactorizer, optional): An optional Cholesky factorization object. Defaults to None.
        """
        # Declare and initialize the host-side cache of the necessary memory allocations
        self._num_worlds: int = 0
        self._model_maxdims: int = 0
        self._model_maxsize: int = 0
        self._world_maxdims: list[int] = []
        self._world_maxsize: list[int] = []
        self._max_of_max_total_D_size: int = 0

        # Cache the requested device
        self._device: wp.DeviceLike = device

        # Declare the model size cache
        self._size: SizeKamino | None = None

        # Initialize the Delassus data container
        self._operator: DenseLinearOperatorData | None = None

        # Declare the optional Cholesky factorization
        self._solver: LinearSolverType | None = None

        # Allocate the Delassus operator data if at least the model is provided
        if model is not None:
            self.finalize(
                model=model,
                data=data,
                limits=limits,
                contacts=contacts,
                solver=solver,
                solver_kwargs=solver_kwargs,
                device=device,
            )

    @property
    def num_worlds(self) -> int:
        """
        Returns the number of worlds represented by the Delassus operator.
        This is equal to the number of matrix blocks contained in the flat array.
        """
        return self._num_worlds

    @property
    def num_maxdims(self) -> int:
        """
        Returns the maximum dimension of the Delassus matrix across all worlds.
        This is the sum of per matrix block maximum dimensions.
        """
        return self._model_maxdims

    @property
    def num_maxsize(self) -> int:
        """
        Returns the maximum size of the Delassus matrix across all worlds.
        This is the sum over the sizes of all matrix blocks.
        """
        return self._model_maxsize

    @property
    def operator(self) -> DenseLinearOperatorData:
        """
        Returns a reference to the flat Delassus matrix array.
        """
        return self._operator

    @property
    def solver(self) -> LinearSolverType:
        """
        The linear solver object for the Delassus operator.
        This is used to perform the factorization of the Delassus matrix.
        """
        return self._solver

    @property
    def info(self) -> DenseSquareMultiLinearInfo:
        """
        Returns a reference to the flat Delassus matrix array.
        """
        return self._operator.info

    @property
    def D(self) -> wp.array:
        """
        Returns a reference to the flat Delassus matrix array.
        """
        return self._operator.mat

    def finalize(
        self,
        model: ModelKamino,
        data: DataKamino,
        limits: LimitsKamino | None = None,
        contacts: ContactsKamino | None = None,
        solver: LinearSolverType = None,
        device: wp.DeviceLike = None,
        solver_kwargs: dict[str, Any] | None = None,
    ):
        """
        Allocates the Delassus operator with the specified dimensions and device.

        Args
        ----
            dims (List[int]): The dimensions of the Delassus matrix for each world.
            device (wp.DeviceLike, optional): The device identifier for the Delassus operator. Defaults to None.
            factorizer (CholeskyFactorizer, optional): An optional Cholesky factorization object. Defaults to None.
        """

        # Ensure the model container is valid
        if model is None:
            raise ValueError(
                "A model container of type `ModelKamino` must be provided to allocate the Delassus operator."
            )
        elif not isinstance(model, ModelKamino):
            raise ValueError("Invalid model provided. Must be an instance of `ModelKamino`.")

        # Ensure the data container is valid if provided
        if data is None:
            raise ValueError(
                "A data container of type `DataKamino` must be provided to allocate the Delassus operator."
            )
        elif not isinstance(data, DataKamino):
            raise ValueError("Invalid data container provided. Must be an instance of `DataKamino`.")

        # Ensure the limits container is valid if provided
        if limits is not None:
            if not isinstance(limits, LimitsKamino):
                raise ValueError("Invalid limits container provided. Must be an instance of `LimitsKamino`.")

        # Ensure the contacts container is valid if provided
        if contacts is not None:
            if not isinstance(contacts, ContactsKamino):
                raise ValueError("Invalid contacts container provided. Must be an instance of `ContactsKamino`.")

        # Capture reference to the model size
        self._size = model.size

        # Extract required maximum number of constraints for each world
        maxdims = get_max_constraints_per_world(model, limits, contacts)

        # Update the allocation meta-data the specified constraint dimensions
        self._num_worlds = model.size.num_worlds
        self._world_dims = maxdims
        self._world_size = [maxdims[i] * maxdims[i] for i in range(self._num_worlds)]
        self._model_maxdims = sum(self._world_dims)
        self._model_maxsize = sum(self._world_size)
        self._max_of_max_total_D_size = max(self._world_size)

        # Override the device identifier if specified, otherwise use the current device
        if device is not None:
            self._device = device

        # Construct the Delassus operator data structure
        self._operator = DenseLinearOperatorData()
        self._operator.info = DenseSquareMultiLinearInfo()
        self._operator.mat = wp.zeros(shape=(self._model_maxsize,), dtype=float32, device=self._device)
        if (model.info is not None) and (data.info is not None):
            mat_offsets = [0] + [sum(self._world_size[:i]) for i in range(1, self._num_worlds + 1)]
            self._operator.info.assign(
                maxdim=model.info.max_total_cts,
                dim=data.info.num_total_cts,
                vio=model.info.total_cts_offset,
                mio=wp.array(mat_offsets[: self._num_worlds], dtype=int32, device=self._device),
                dtype=float32,
                device=self._device,
            )
        else:
            self._operator.info.finalize(dimensions=maxdims, dtype=float32, itype=int32, device=self._device)

        # Optionally initialize the linear system solver if one is specified
        if solver is not None:
            if not issubclass(solver, LinearSolverType):
                raise ValueError("Invalid solver provided. Must be a subclass of `LinearSolverType`.")
            solver_kwargs = solver_kwargs or {}
            self._solver = solver(operator=self._operator, device=self._device, **solver_kwargs)

    def zero(self):
        """
        Sets all values of the Delassus matrix to zero.
        This is useful for resetting the operator before recomputing it.
        """
        self._operator.mat.zero_()

    def build(
        self,
        model: ModelKamino,
        data: DataKamino,
        jacobians: DenseSystemJacobians | SparseSystemJacobians,
        reset_to_zero: bool = True,
    ):
        """
        Builds the Delassus matrix using the provided ModelKamino, DataKamino, and constraint Jacobians.

        Args:
            model (ModelKamino): The model for which the Delassus operator is built.
            data (DataKamino): The current data of the model.
            jacobians (DenseSystemJacobians | SparseSystemJacobians): The current Jacobians of the model.
            reset_to_zero (bool, optional): If True (default), resets the Delassus matrix to zero before building.

        Raises:
            ValueError: If the model, data, or Jacobians are not valid.
            ValueError: If the Delassus matrix is not allocated.
        """
        # Ensure the model is valid
        if model is None or not isinstance(model, ModelKamino):
            raise ValueError("A valid model of type `ModelKamino` must be provided to build the Delassus operator.")

        # Ensure the data is valid
        if data is None or not isinstance(data, DataKamino):
            raise ValueError("A valid model data of type `DataKamino` must be provided to build the Delassus operator.")

        # Ensure the Jacobians are valid
        if jacobians is None or not (
            isinstance(jacobians, DenseSystemJacobians) or isinstance(jacobians, SparseSystemJacobians)
        ):
            raise ValueError(
                "A valid Jacobians data container of type `DenseSystemJacobians` or "
                "`SparseSystemJacobians` must be provided to build the Delassus operator."
            )

        # Ensure the Delassus matrix is allocated
        if self._operator.mat is None:
            raise ValueError("Delassus matrix is not allocated. Call finalize() first.")

        # Initialize the Delassus matrix to zero
        if reset_to_zero:
            self.zero()

        # Build the Delassus matrix parallelized element-wise
        if isinstance(jacobians, DenseSystemJacobians):
            wp.launch(
                kernel=_build_delassus_elementwise_dense,
                dim=(self._size.num_worlds, self._max_of_max_total_D_size),
                inputs=[
                    # Inputs:
                    model.info.num_bodies,
                    model.info.bodies_offset,
                    model.bodies.inv_m_i,
                    data.bodies.inv_I_i,
                    jacobians.data.J_cts_offsets,
                    jacobians.data.J_cts_data,
                    self._operator.info.dim,
                    self._operator.info.mio,
                    # Outputs:
                    self._operator.mat,
                ],
            )
        else:
            jacobian_cts = jacobians._J_cts.bsm
            wp.launch(
                kernel=_build_delassus_elementwise_sparse,
                dim=(self._size.num_worlds, jacobian_cts.max_of_num_nzb * jacobian_cts.max_of_num_nzb),
                inputs=[
                    # Inputs:
                    model.info.bodies_offset,
                    model.bodies.inv_m_i,
                    data.bodies.inv_I_i,
                    jacobian_cts.num_nzb,
                    jacobian_cts.nzb_start,
                    jacobian_cts.nzb_coords,
                    jacobian_cts.nzb_values,
                    self._operator.info.dim,
                    self._operator.info.mio,
                    # Outputs:
                    self._operator.mat,
                ],
            )

        # Add armature regularization to the upper diagonal if dynamic joint constraints are present
        if model.size.sum_of_num_dynamic_joints > 0:
            wp.launch(
                kernel=_add_joint_armature_diagonal_regularization_dense,
                dim=(self._size.num_worlds, model.size.max_of_num_dynamic_joint_cts),
                inputs=[
                    # Inputs:
                    model.info.num_joint_dynamic_cts,
                    model.info.joint_dynamic_cts_offset,
                    data.joints.inv_m_j,
                    self._operator.info.dim,
                    self._operator.info.mio,
                    # Outputs:
                    self._operator.mat,
                ],
            )

    def regularize(self, eta: wp.array):
        """
        Adds diagonal regularization to each matrix block of the Delassus operator.

        Args:
            eta (wp.array): The regularization values to add to the diagonal of each matrix block.
            This should be an array of shape `(maxdims,)` and type :class:`float32`.
            Each value in `eta` corresponds to the regularization along each constraint.
        """
        wp.launch(
            kernel=_regularize_delassus_diagonal_dense,
            dim=(self._size.num_worlds, self._size.max_of_max_total_cts),
            inputs=[self._operator.info.dim, self._operator.info.vio, self._operator.info.mio, eta, self._operator.mat],
        )

    def compute(self, reset_to_zero: bool = True):
        """
        Runs Delassus pre-computation operations in preparation for linear systems solves.

        Depending on the configured solver type, this may perform different
        pre-computation, e.g. Cholesky factorization for direct solvers.

        Args:
            reset_to_zero (bool):
                If True, resets the Delassus matrix to zero.\n
                This is useful for ensuring that the matrix is in a clean state before pre-computation.
        """
        # Ensure the Delassus matrix is allocated
        if self._operator.mat is None:
            raise ValueError("Delassus matrix is not allocated. Call finalize() first.")

        # Ensure the solver is available if pre-computation is requested
        if self._solver is None:
            raise ValueError("A linear system solver is not available. Allocate with solver=LINEAR_SOLVER_TYPE.")

        # Optionally initialize the factorization matrix before factorizing
        if reset_to_zero:
            self._solver.reset()

        # Perform the Cholesky factorization
        self._solver.compute(A=self._operator.mat)

    def solve(self, v: wp.array, x: wp.array):
        """
        Solves the linear system D * x = v using the Cholesky factorization.

        Args:
            v (wp.array): The right-hand side vector of the linear system.
            x (wp.array): The array to hold the solution.

        Raises:
            ValueError: If the Delassus matrix is not allocated or the factorizer is not available.
            ValueError: If a factorizer has not been configured set.
        """
        # Ensure the Delassus matrix is allocated
        if self._operator.mat is None:
            raise ValueError("Delassus matrix is not allocated. Call finalize() first.")

        # Ensure the solver is available if solving is requested
        if self._solver is None:
            raise ValueError("A linear system solver is not available. Allocate with solver=LINEAR_SOLVER_TYPE.")

        # Solve the linear system using the factorized matrix
        return self._solver.solve(b=v, x=x)

    def solve_inplace(self, x: wp.array):
        """
        Solves the linear system D * x = v in-place.\n
        This modifies the input array x to contain the solution assuming it is initialized as x=v.

        Args:
            x (wp.array): The array to hold the solution. It should be initialized with the right-hand side vector v.

        Raises:
            ValueError: If the Delassus matrix is not allocated or the factorizer is not available.
            ValueError: If a factorizer has not been configured set.
        """
        # Ensure the Delassus matrix is allocated
        if self._operator.mat is None:
            raise ValueError("Delassus matrix is not allocated. Call finalize() first.")

        # Ensure the solvers is available if solving in-place is requested
        if self._solver is None:
            raise ValueError("A linear system solver is not available. Allocate with solver=LINEAR_SOLVER_TYPE.")

        # Solve the linear system in-place
        return self._solver.solve_inplace(x=x)


class BlockSparseMatrixFreeDelassusOperator(BlockSparseLinearOperators):
    """
    A matrix-free Delassus operator for representing and operating on multiple independent sparse
    linear systems.

    In contrast to the dense :class:`DelassusOperator`, this operator only provides functions to
    compute matrix-vector products with the Delassus matrix, not solve linear systems.

    The Delassus operator D is implicitly defined as D = J @ M^-1 @ J^T, where J is the constraint
    Jacobian and M is the mass matrix. It supports diagonal regularization and diagonal
    preconditioning.

    For a given diagonal regularization matrix R and a diagonal preconditioning matrix P, the
    final operator is defined by the matrix P @ D @ P + R.

    Typical usage example:

    .. code-block:: python

        # Create a model builder and add bodies, joints, geoms, etc.
        builder = ModelBuilder()
        ...

        # Create a model from the builder and construct additional
        # containers to hold joint-limits, contacts, Jacobians
        model = builder.finalize()
        data = model.data()
        limits = LimitsKamino(model)
        contacts = ContactsKamino(builder)
        jacobians = SparseSystemJacobians(model, limits, contacts)

        # Build the Jacobians for the model and active limits and contacts
        jacobians.build(model, data, limits, contacts)
        ...

        # Create a Delassus operator from the model data and Jacobians
        delassus = BlockSparseMatrixFreeDelassusOperator(model, data, jacobians)

        # Add diagonal regularization to the Delassus operator
        eta = ...
        delassus.set_regularization(eta=eta)

        # Add preconditioning to the Delassus operator
        P = ...
        delassus.set_preconditioner(preconditioner=P)

        # Compute the matrix-vector product `y = D @ x` using the Delassus operator
        x = ...
        y = ...
        world_mask = ...
        delassus.matvec(x=x, y=y, world_mask=world_mask)
    """

    def __init__(
        self,
        model: ModelKamino | None = None,
        data: DataKamino | None = None,
        limits: LimitsKamino | None = None,
        contacts: ContactsKamino | None = None,
        jacobians: SparseSystemJacobians | None = None,
        solver: LinearSolverType = None,
        solver_kwargs: dict[str, Any] | None = None,
        device: wp.DeviceLike = None,
    ):
        """
        Creates a Delassus operator for the given model.

        This class also supports deferred allocation, i.e. it can be initialized without a model,
        and the allocation can be performed later using the `finalize` method. This is useful for
        scenarios where the model or constraints are not known at the time of Delassus operator
        creation, but will be available later.

        The dimension of a Delassus matrix is defined as the sum over active joint, limit, and
        contact constraints, and the maximum dimension is the maximum number of constraints that can
        be active in each world.

        Args:
            model (ModelKamino):
                The model container for which the Delassus operator is built.
            data (DataKamino, optional):
                The model data container holding the state info and data.
            limits (LimitsKamino, optional):
                Limits data container for joint limit constraints.
            contacts (ContactsKamino, optional):
                Contacts data container for contact constraints.
            jacobians (SparseSystemJacobians, optional):
                The sparse Jacobians container.
                Can be assigned later via the `assign` method.
            solver (LinearSolverType, optional):
                The linear solver class to use for solving linear systems.
                Must be a subclass of `IterativeSolver`.
            solver_kwargs (dict, optional):
                Additional keyword arguments to pass to the solver constructor.
            device (wp.DeviceLike, optional):
                The device identifier for the Delassus operator. Defaults to None.
        """
        super().__init__()

        # self.bsm represents the constraint Jacobian
        self._model: ModelKamino | None = None
        self._data: DataKamino | None = None
        self._limits: LimitsKamino | None = None
        self._contacts: ContactsKamino | None = None
        self._preconditioner: wp.array | None = None
        self._eta: wp.array | None = None

        self._jacobians: SparseSystemJacobians | None = None

        # Problem info object
        # TODO: Create more general info object independent of dense matrix representation
        self._info: DenseSquareMultiLinearInfo | None = None

        # Cache the requested device
        self._device: wp.DeviceLike = device

        # Declare the optional (iterative) solver
        self._solver: LinearSolverType | None = None

        # Flag to indicate that the operator needs an update to its data structure
        self._needs_update: bool = False

        # Temporary vector to store results, sized to the number of body dofs in a model.
        self._vec_temp_body_space: wp.array | None = None

        self._col_major_jacobian: ColMajorSparseConstraintJacobians | None = None
        self._transpose_op_matrix: BlockSparseMatrices | None = None

        # Combined regularization vector for implicit joint dynamics
        self._combined_regularization: wp.array | None = None

        # Allocate the Delassus operator data if at least the model is provided
        if model is not None:
            self.finalize(
                model=model,
                data=data,
                limits=limits,
                contacts=contacts,
                jacobians=jacobians,
                solver=solver,
                device=device,
                solver_kwargs=solver_kwargs,
            )

    def finalize(
        self,
        model: ModelKamino,
        data: DataKamino,
        limits: LimitsKamino | None,
        contacts: ContactsKamino | None,
        jacobians: SparseSystemJacobians | None = None,
        solver: LinearSolverType = None,
        device: wp.DeviceLike = None,
        solver_kwargs: dict[str, Any] | None = None,
    ):
        """
        Allocates the Delassus operator with the specified dimensions and device.

        Args:
            model (ModelKamino):
                The model container for which the Delassus operator is built.
            data (DataKamino):
                The model data container holding the state info and data.
            limits (LimitsKamino, optional):
                Limits data container for joint limit constraints.
            contacts (ContactsKamino, optional):
                Contacts data container for contact constraints.
            jacobians (SparseSystemJacobians, optional):
                The sparse Jacobians container.
                Can be assigned later via the `assign` method.
            solver (LinearSolverType, optional):
                The linear solver class to use for solving linear systems.
                Must be a subclass of `IterativeSolver`.
            device (wp.DeviceLike, optional):
                The device identifier for the Delassus operator.
                Defaults to None.
            solver_kwargs (dict, optional):
                Additional keyword arguments to pass to the solver constructor.
        """
        # Ensure the model container is valid
        if model is None:
            raise ValueError(
                "A model container of type `ModelKamino` must be provided to allocate the Delassus operator."
            )
        elif not isinstance(model, ModelKamino):
            raise ValueError("Invalid model provided. Must be an instance of `ModelKamino`.")

        # Ensure the data container is valid if provided
        if data is None:
            raise ValueError(
                "A data container of type `DataKamino` must be provided to allocate the Delassus operator."
            )
        elif not isinstance(data, DataKamino):
            raise ValueError("Invalid data container provided. Must be an instance of `DataKamino`.")

        # Ensure the solver is iterative if provided
        if solver is not None and not issubclass(solver, IterativeSolver):
            raise ValueError("Invalid solver provided. Must be a subclass of `IterativeSolver`.")

        self._model = model
        self._data = data
        self._limits = limits
        self._contacts = contacts

        # Override the device identifier if specified, otherwise use the current device
        if device is not None:
            self._device = device

        self._info = DenseSquareMultiLinearInfo()
        if model.info is not None and data.info is not None:
            self._info.assign(
                maxdim=model.info.max_total_cts,
                dim=data.info.num_total_cts,
                vio=model.info.total_cts_offset,
                mio=wp.empty((self.num_matrices,), dtype=int32, device=self._device),
                dtype=float32,
                device=self._device,
            )
        else:
            self._info.finalize(
                dimensions=model.info.max_total_cts.numpy(),
                dtype=float32,
                itype=int32,
                device=self._device,
            )

        self._active_rows = wp.array(
            dtype=wp.int32,
            shape=(self._model.size.num_worlds,),
            ptr=self._data.info.num_total_cts.ptr,
            copy=False,
        )
        self._active_cols = wp.array(
            dtype=wp.int32,
            shape=(self._model.size.num_worlds,),
            ptr=self._data.info.num_total_cts.ptr,
            copy=False,
        )

        # Initialize temporary memory
        self._vec_temp_body_space = wp.empty(
            (self._model.size.sum_of_num_body_dofs,), dtype=float32, device=self._device
        )

        # Initialize memory for combined regularization, if necessary
        if self._model.size.max_of_num_dynamic_joint_cts > 0:
            self._combined_regularization = wp.empty(
                (self._model.size.sum_of_max_total_cts,), dtype=float32, device=self._device
            )

        # Check whether any of the maximum row dimensions of the Jacobians is smaller than six.
        # If so, we avoid building the column-major Jacobian due to potential memory access issues.
        min_of_max_rows = np.min(self._model.info.max_total_cts.numpy())

        if min_of_max_rows >= 6:
            self._col_major_jacobian = ColMajorSparseConstraintJacobians(
                model=self._model,
                limits=self._limits,
                contacts=self._contacts,
                jacobians=self._jacobians,
                device=self.device,
            )
            self._transpose_op_matrix = self._col_major_jacobian.bsm
        else:
            self._col_major_jacobian = None

        # Assign Jacobian if specified
        if jacobians is not None:
            self.assign(jacobians)

        # Optionally initialize the iterative linear system solver if one is specified
        if solver is not None:
            solver_kwargs = solver_kwargs or {}
            self._solver = solver(operator=self, device=self._device, **solver_kwargs)

        self.set_needs_update()

    def assign(self, jacobians: SparseSystemJacobians):
        """
        Assigns the constraint Jacobian to the Delassus operator.

        This method links the Delassus operator to the block sparse Jacobian matrix,
        which is used to compute the implicit Delassus matrix-vector products.

        Args:
            jacobians (SparseSystemJacobians):
                The sparse system Jacobians container holding the
                constraint Jacobian matrix in block sparse format.
        """
        self._jacobians = jacobians

        if self._col_major_jacobian is None and self._transpose_op_matrix is None:
            # Create copy of constraint Jacobian with separate non-zero block values, so we can apply
            # preconditioning directly to the Jacobian.
            self._transpose_op_matrix = copy.copy(jacobians._J_cts.bsm)
            self._transpose_op_matrix.nzb_values = wp.empty_like(self.constraint_jacobian.nzb_values)

        # Create a shallow copy of the constraint Jacobian, but with a separate array for non-zero block values.
        # The resulting sparse matrix will reference the structure of the original Jacobian, but we can apply
        # preconditioning and the inverse mass matrix to the non-zero blocks without affecting the original Jacobian.
        if self.bsm is None:
            self.bsm = copy.copy(jacobians._J_cts.bsm)
            self.bsm.nzb_values = wp.empty_like(self.constraint_jacobian.nzb_values)

        self.update()

    def set_needs_update(self):
        """
        Flags the operator as needing to update its data structure.
        """
        self._needs_update = True

    def update(self):
        """
        Updates any internal data structures that depend on the model, limits, contacts, or system Jacobians.
        """
        if self._jacobians is None:
            return

        # Update column-major constraint Jacobian based on current system Jacobian
        if self._col_major_jacobian is None:
            wp.copy(self._transpose_op_matrix.nzb_values, self.constraint_jacobian.nzb_values)
        else:
            self._col_major_jacobian.update(self._model, self._jacobians, self._limits, self._contacts)

        # Copy current Jacobian values to local constraint Jacobian
        wp.copy(self.bsm.nzb_values, self.constraint_jacobian.nzb_values)

        # Apply inverse mass matrix to (copy of) constraint Jacobian
        wp.launch(
            kernel=_merge_inv_mass_matrix_kernel,
            dim=(self.bsm.num_matrices, self.bsm.max_of_num_nzb),
            inputs=[
                # Inputs:
                self._model.info.bodies_offset,
                self._model.bodies.inv_m_i,
                self._data.bodies.inv_I_i,
                self.bsm.num_nzb,
                self.bsm.nzb_start,
                self.bsm.nzb_coords,
                # Outputs:
                self.bsm.nzb_values,
            ],
            device=self.bsm.device,
        )

        if self._preconditioner is not None:
            # Apply preconditioner to (copy of) constraint Jacobian
            wp.launch(
                kernel=_make_merge_preconditioner_kernel(self.bsm.nzb_dtype),
                dim=(self.bsm.num_matrices, self.bsm.max_of_num_nzb),
                inputs=[
                    # Inputs:
                    self.bsm.num_nzb,
                    self.bsm.nzb_start,
                    self.bsm.nzb_coords,
                    self.bsm.row_start,
                    self._preconditioner,
                    # Outputs:
                    self.bsm.nzb_values,
                ],
                device=self.bsm.device,
            )

            # Apply preconditioner to column-major constraint Jacobian
            wp.launch(
                kernel=_make_merge_preconditioner_kernel(self._transpose_op_matrix.nzb_dtype),
                dim=(self._transpose_op_matrix.num_matrices, self._transpose_op_matrix.max_of_num_nzb),
                inputs=[
                    # Inputs:
                    self._transpose_op_matrix.num_nzb,
                    self._transpose_op_matrix.nzb_start,
                    self._transpose_op_matrix.nzb_coords,
                    self._transpose_op_matrix.row_start,
                    self._preconditioner,
                    # Outputs:
                    self._transpose_op_matrix.nzb_values,
                ],
                device=self._transpose_op_matrix.device,
            )

        # Update combined regularization term, which includes the regular regularization (eta) as
        # well as the terms of the armature regularization. Since the armature regularization is
        # applied to the original Delassus matrix, preconditioning has to be applied to it if
        # present.
        if self._combined_regularization is not None:
            # Set the combined regularization to the regular regularization if present, otherwise
            # initialize to zero.
            if self._eta is not None:
                wp.copy(self._combined_regularization, self._eta)
            else:
                self._combined_regularization.zero_()

            if self._preconditioner is None:
                # If there is no preconditioner, we add the armature regularization directly to the
                # combined regularization term.
                wp.launch(
                    kernel=_add_armature_regularization_sparse,
                    dim=(self.num_matrices, self._model.size.max_of_num_dynamic_joint_cts),
                    inputs=[
                        # Inputs:
                        self._model.info.num_joint_dynamic_cts,
                        self._model.info.joint_dynamic_cts_offset,
                        self.bsm.row_start,
                        self._data.joints.inv_m_j,
                        # Outputs:
                        self._combined_regularization,
                    ],
                    device=self._device,
                )
            else:
                # If there is a preconditioner, we need to scale the armature regularization with
                # the preconditioner terms (the square of the preconditioner, to be exact) before
                # adding it to the combined regularization term.
                wp.launch(
                    kernel=_add_armature_regularization_preconditioned_sparse,
                    dim=(self.num_matrices, self._model.size.max_of_num_dynamic_joint_cts),
                    inputs=[
                        # Inputs:
                        self._model.info.num_joint_dynamic_cts,
                        self._model.info.joint_dynamic_cts_offset,
                        self._data.joints.inv_m_j,
                        self.bsm.row_start,
                        self._preconditioner,
                        # Outputs:
                        self._combined_regularization,
                    ],
                    device=self._device,
                )

        self._needs_update = False

    def set_regularization(self, eta: wp.array | None):
        """
        Adds diagonal regularization to each matrix block of the Delassus operator, replacing any
        previously set regularization.

        The regularized Delassus matrix is defined as D = J @ M^-1 @ J^T + diag(eta)

        Args:
            eta (wp.array): The regularization values to add to the diagonal of each matrix block,
                with each value corresponding to the regularization along a constraint.
                This should be an array of shape `(sum_of_max_total_cts,)` and type :class:`float32`,
                or `None` if no regularization should be applied.
        """
        self._eta = eta
        self.set_needs_update()

    def set_preconditioner(self, preconditioner: wp.array | None):
        """
        Sets the diagonal preconditioner for the Delassus operator, replacing any previously set
        preconditioner.

        With preconditioning, the effective operator becomes P @ D @ P, where P = diag(preconditioner).

        Args:
            preconditioner (wp.array): The diagonal preconditioner values to apply to the Delassus
                operator, with each value corresponding to a constraint. This should be an array of
                shape `(sum_of_max_total_cts,)` and type :class:`float32`, or `None` to disable
                preconditioning.
        """
        self._preconditioner = preconditioner
        self.set_needs_update()

    def diagonal(self, diag: wp.array):
        """Stores the diagonal of the Delassus matrix in the given array.

        Note:
            This uses the diagonal of the pure Delassus matrix, without any regularization or
            preconditioning.

        Args:
            diag (wp.array): Output vector for the Delassus matrix diagonal entries.
                Shape `(sum_of_max_total_cts,)` and type :class:`float32`.
        """
        if self._model is None or self._data is None:
            raise RuntimeError("ModelKamino and data must be assigned before computing diagonal.")
        if self.bsm is None:
            raise RuntimeError("Jacobian must be assigned before computing diagonal.")

        diag.zero_()

        # Launch kernel over all non-zero blocks
        wp.launch(
            kernel=_compute_block_sparse_delassus_diagonal,
            dim=(self._model.size.num_worlds, self.bsm.max_of_num_nzb),
            inputs=[
                self._model.info.bodies_offset,
                self._model.bodies.inv_m_i,
                self._data.bodies.inv_I_i,
                self.constraint_jacobian.nzb_start,
                self.constraint_jacobian.num_nzb,
                self.constraint_jacobian.nzb_coords,
                self.constraint_jacobian.nzb_values,
                self.constraint_jacobian.row_start,
                diag,
            ],
            device=self._device,
        )

        # Add armature regularization
        wp.launch(
            kernel=_add_armature_regularization_sparse,
            dim=(self.num_matrices, self._model.size.max_of_num_dynamic_joint_cts),
            inputs=[
                # Inputs:
                self._model.info.num_joint_dynamic_cts,
                self._model.info.joint_dynamic_cts_offset,
                self.bsm.row_start,
                self._data.joints.inv_m_j,
                # Outputs:
                diag,
            ],
            device=self._device,
        )

    def compute(self, reset_to_zero: bool = True):
        """
        Runs Delassus pre-computation operations in preparation for linear systems solves.

        Depending on the configured solver type, this may perform different pre-computation.

        Args:
            reset_to_zero (bool):
                If True, resets the Delassus matrix to zero.\n
                This is useful for ensuring that the matrix is in a clean state before pre-computation.
        """
        # Ensure that `finalize()` was called
        if self._info is None:
            raise ValueError("Data structure is not allocated. Call finalize() first.")

        # Ensure the Jacobian is set
        if self.bsm is None:
            raise ValueError("Jacobian matrix is not set. Call assign() first.")

        # Ensure the solver is available if pre-computation is requested
        if self._solver is None:
            raise ValueError("A linear system solver is not available. Allocate with solver=LINEAR_SOLVER_TYPE.")

        # Update if data has changed
        if self._needs_update:
            self.update()

        # Optionally initialize the solver
        if reset_to_zero:
            self._solver.reset()

        # Perform the pre-computation
        self._solver.compute()

    def solve(self, v: wp.array, x: wp.array):
        """
        Solves the linear system D * x = v using the assigned solver.

        Args:
            v (wp.array): The right-hand side vector of the linear system.
            x (wp.array): The array to hold the solution.

        Raises:
            ValueError: If the Delassus matrix is not allocated or the solver is not available.
        """
        # Ensure that `finalize()` was called
        if self._info is None:
            raise ValueError("Data structure is not allocated. Call finalize() first.")

        # Ensure the Jacobian is set
        if self.bsm is None:
            raise ValueError("Jacobian matrix is not set. Call assign() first.")

        # Ensure the solver is available
        if self._solver is None:
            raise ValueError("A linear system solver is not available. Allocate with solver=LINEAR_SOLVER_TYPE.")

        # Update if data has changed
        if self._needs_update:
            self.update()

        # Solve the linear system
        return self._solver.solve(b=v, x=x)

    def solve_inplace(self, x: wp.array):
        """
        Solves the linear system D * x = v in-place.\n
        This modifies the input array x to contain the solution assuming it is initialized as x=v.

        Args:
            x (wp.array): The array to hold the solution. It should be initialized with the right-hand side vector v.

        Raises:
            ValueError: If the Delassus matrix is not allocated or the solver is not available.
        """
        # Ensure that `finalize()` was called
        if self._info is None:
            raise ValueError("Data structure is not allocated. Call finalize() first.")

        # Ensure the Jacobian is set
        if self.bsm is None:
            raise ValueError("Jacobian matrix is not set. Call assign() first.")

        # Ensure the solver is available if pre-computation is requested
        if self._solver is None:
            raise ValueError("A linear system solver is not available. Allocate with solver=LINEAR_SOLVER_TYPE.")

        # Update if data has changed
        if self._needs_update:
            self.update()

        # Solve the linear system in-place
        return self._solver.solve_inplace(x=x)

    ###
    # Properties
    ###

    @property
    def info(self) -> DenseSquareMultiLinearInfo | None:
        """
        Returns the info object for the Delassus problem dimensions and sizes.
        """
        return self._info

    @property
    def num_matrices(self) -> int:
        """
        Returns the number of matrices represented by the Delassus operator.
        """
        return self._model.size.num_worlds

    @property
    def max_of_max_dims(self) -> tuple[int, int]:
        """
        Returns the maximum dimension of any Delassus matrix across all worlds.
        """
        max_jac_rows = self._model.size.max_of_max_total_cts
        return (max_jac_rows, max_jac_rows)

    @property
    def sum_of_max_dims(self) -> int:
        """
        Returns the sum of maximum dimensions of the Delassus matrix across all worlds.
        """
        return self._model.size.sum_of_max_total_cts

    @property
    def dtype(self) -> FloatType:
        return self._info.dtype

    @property
    def device(self) -> wp.DeviceLike:
        return self._model.device

    @property
    def constraint_jacobian(self) -> BlockSparseMatrices:
        return self._jacobians._J_cts.bsm

    ###
    # Operations
    ###

    def matvec(self, x: wp.array, y: wp.array, world_mask: wp.array):
        """
        Performs the sparse matrix-vector product `y = D @ x`, applying regularization and
        preconditioning if configured.
        """
        if self.Ax_op is None:
            raise RuntimeError("No `A@x` operator has been assigned.")
        if self.ATy_op is None:
            raise RuntimeError("No `A^T@y` operator has been assigned.")

        # Update if data has changed
        if self._needs_update:
            self.update()

        v = self._vec_temp_body_space
        v.zero_()

        # Compute first Jacobian matrix-vector product: v <- (P @ J)^T @ x
        self.ATy_op(self._transpose_op_matrix, x, v, world_mask)

        if self._eta is None and self._combined_regularization is None:
            # Compute second Jacobian matrix-vector product: y <- (P @ J @ M^-1) @ v
            self.Ax_op(self.bsm, v, y, world_mask)
        else:
            y.zero_()
            # Compute y <- (P @ J @ M^-1) @ v + diag(eta) @ x
            wp.launch(
                kernel=_make_block_sparse_gemv_regularization_kernel(1.0),
                dim=(self.bsm.num_matrices, self.bsm.max_of_num_nzb),
                inputs=[
                    self.bsm.dims,
                    self.bsm.num_nzb,
                    self.bsm.nzb_start,
                    self.bsm.nzb_coords,
                    self.bsm.nzb_values,
                    self.bsm.row_start,
                    self.bsm.col_start,
                    self._eta if self._combined_regularization is None else self._combined_regularization,
                    v,
                    y,
                    x,
                    world_mask,
                ],
                device=self.device,
            )

    def matvec_transpose(self, y: wp.array, x: wp.array, world_mask: wp.array):
        """
        Performs the sparse matrix-transpose-vector product `x = D^T @ y`.

        Note:
            Since the Delassus matrix is symmetric, this is equivalent to `matvec`.
        """
        # Update if data has changed
        if self._needs_update:
            self.update()

        self.matvec(x, y, world_mask)

    def gemv(self, x: wp.array, y: wp.array, world_mask: wp.array, alpha: float = 1.0, beta: float = 0.0):
        """
        Performs a BLAS-like generalized sparse matrix-vector product `y = alpha * D @ x + beta * y`,
        applying regularization and preconditioning if configured.
        """
        if self.gemv_op is None:
            raise RuntimeError("No BLAS-like `GEMV` operator has been assigned.")
        if self.ATy_op is None:
            raise RuntimeError("No `A^T@y` operator has been assigned.")

        # Update if data has changed
        if self._needs_update:
            self.update()

        v = self._vec_temp_body_space
        v.zero_()

        # Compute first Jacobian matrix-vector product: v <- (P @ J)^T @ x
        self.ATy_op(self._transpose_op_matrix, x, v, world_mask)

        if self._eta is None and self._combined_regularization is None:
            # Compute second Jacobian matrix-vector product as general matrix-vector product:
            #   y <- alpha * (P @ J @ M^-1) @ v + beta * y
            self.gemv_op(self.bsm, v, y, alpha, beta, world_mask)
        else:
            # Scale y <- beta * y
            wp.launch(
                kernel=_scale_row_vector_kernel,
                dim=(self.bsm.num_matrices, self.bsm.max_of_max_dims[0]),
                inputs=[self.bsm.dims, self.bsm.row_start, y, beta, world_mask],
                device=self.device,
            )

            # Compute y <- alpha * (P @ J @ M^-1) @ v + y + alpha * diag(eta) @ x
            wp.launch(
                kernel=_make_block_sparse_gemv_regularization_kernel(alpha),
                dim=(self.bsm.num_matrices, self.bsm.max_of_num_nzb),
                inputs=[
                    self.bsm.dims,
                    self.bsm.num_nzb,
                    self.bsm.nzb_start,
                    self.bsm.nzb_coords,
                    self.bsm.nzb_values,
                    self.bsm.row_start,
                    self.bsm.col_start,
                    self._eta if self._combined_regularization is None else self._combined_regularization,
                    v,
                    y,
                    x,
                    world_mask,
                ],
                device=self.device,
            )

    def gemv_transpose(self, y: wp.array, x: wp.array, world_mask: wp.array, alpha: float = 1.0, beta: float = 0.0):
        """
        Performs a BLAS-like generalized sparse matrix-transpose-vector product
        `x = alpha * D^T @ y + beta * x`.

        Note:
            Since the Delassus matrix is symmetric, this is equivalent to `gemv` with swapped arguments.
        """
        # Update if data has changed
        if self._needs_update:
            self.update()

        self.gemv(y, x, world_mask, alpha, beta)
