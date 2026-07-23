# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Sparse Warp kernels for the Kamino DVI solver."""

from __future__ import annotations

import warp as wp

from ...core.math import FLOAT32_EPS
from ...core.types import vec6f
from .projections import (
    contact_trace_preconditioner as _contact_trace_preconditioner,
)
from .projections import (
    project_contact_block_update as _project_contact_block_update,
)
from .projections import (
    project_contact_diagonal_update as _project_contact_diagonal_update,
)
from .types import DVIConfigStruct, DVIStatus

wp.set_module_options({"enable_backward": False})

float32 = wp.float32
int32 = wp.int32
mat33f = wp.mat33f
vec3f = wp.vec3f


@wp.kernel
def _zero_bilateral_lambdas(
    # Inputs:
    problem_njc: wp.array[int32],
    problem_vio: wp.array[int32],
    # Outputs:
    solution_lambdas: wp.array[float32],
):
    wid, row = wp.tid()

    njc = problem_njc[wid]
    if row >= njc:
        return

    solution_lambdas[problem_vio[wid] + row] = 0.0


@wp.kernel
def _build_sparse_bilateral_rhs(
    # Inputs:
    problem_vio: wp.array[int32],
    problem_njc: wp.array[int32],
    problem_v_f: wp.array[float32],
    state_v_aug: wp.array[float32],
    bilateral_vio: wp.array[int32],
    bilateral_P: wp.array[float32],
    # Outputs:
    bilateral_rhs: wp.array[float32],
):
    wid, row = wp.tid()

    njc = problem_njc[wid]
    if row >= njc:
        return

    pvio = problem_vio[wid]
    bvio = bilateral_vio[wid]
    rhs = -(state_v_aug[pvio + row] + problem_v_f[pvio + row])
    bilateral_rhs[bvio + row] = bilateral_P[bvio + row] * rhs


@wp.kernel
def _sparse_delassus_gemv_rows(
    # Matrix data:
    dims: wp.array2d[int32],
    num_nzb: wp.array[int32],
    nzb_start: wp.array[int32],
    nzb_coords: wp.array2d[int32],
    nzb_values: wp.array[vec6f],
    row_start: wp.array[int32],
    col_start: wp.array[int32],
    # Row ranges:
    problem_dim: wp.array[int32],
    problem_njc: wp.array[int32],
    row_kind: int32,
    # Regularization:
    eta: wp.array[float32],
    # Vectors:
    body_space: wp.array[float32],
    y: wp.array[float32],
    lambdas: wp.array[float32],
    # Mask:
    world_mask: wp.array[bool],
):
    wid, block_idx = wp.tid()

    if not world_mask[wid]:
        return

    dim = problem_dim[wid]
    njc = problem_njc[wid]

    if block_idx < dim:
        row = block_idx
        row_active = row < njc
        if row_kind == int32(1):
            row_active = row >= njc
        if row_active:
            vec_idx = row_start[wid] + row
            wp.atomic_add(y, vec_idx, eta[vec_idx] * lambdas[vec_idx])

    if block_idx >= num_nzb[wid]:
        return

    global_block_idx = nzb_start[wid] + block_idx
    block_coord = nzb_coords[global_block_idx]
    row = block_coord[0]
    if row < 0 or row >= dim:
        return

    row_active = row < njc
    if row_kind == int32(1):
        row_active = row >= njc
    if not row_active:
        return

    # The body-space input already contains M^-1 * J^T * lambda. Accumulate
    # selected rows of J times that vector; eta * lambda supplies R * lambda.
    block = nzb_values[global_block_idx]
    x_idx_base = col_start[wid] + block_coord[1]
    acc = float32(0.0)
    for j in range(6):
        acc += block[j] * body_space[x_idx_base + j]

    wp.atomic_add(y, row_start[wid] + row, acc)


@wp.func
def _apply_limit_offset_update(
    limit_id: int32,
    # Matrix data:
    num_nzb: wp.array[int32],
    nzb_start: wp.array[int32],
    nzb_coords: wp.array2d[int32],
    nzb_values: wp.array[vec6f],
    row_start: wp.array[int32],
    col_start: wp.array[int32],
    # Active limits:
    limits_model_active: wp.array[int32],
    limits_wid: wp.array[int32],
    limits_lid: wp.array[int32],
    limits_nzb_offsets: wp.array[int32],
    # Problem data:
    problem_vio: wp.array[int32],
    problem_nl: wp.array[int32],
    problem_lcgo: wp.array[int32],
    problem_diag: wp.array[float32],
    problem_P: wp.array[float32],
    problem_v_f: wp.array[float32],
    eta: wp.array[float32],
    body_space: wp.array[float32],
    block_iteration: int32,
    contact_iteration: int32,
    solver_config: wp.array[DVIConfigStruct],
    # Outputs:
    solution_lambdas: wp.array[float32],
):
    if limit_id >= limits_model_active[0]:
        return

    wid = limits_wid[limit_id]
    cfg = solver_config[wid]
    if block_iteration >= cfg.block_iterations or contact_iteration >= cfg.contact_iterations:
        return

    lid = limits_lid[limit_id]
    if lid >= problem_nl[wid]:
        return

    row = problem_lcgo[wid] + lid
    vec_idx = problem_vio[wid] + row
    eta_idx = row_start[wid] + row
    value = eta[eta_idx] * solution_lambdas[vec_idx]
    matrix_end = nzb_start[wid] + num_nzb[wid]
    nzb_offset = limits_nzb_offsets[limit_id]
    for k in range(2):
        nzb_idx = nzb_offset + k
        if nzb_idx < matrix_end:
            block_coord = nzb_coords[nzb_idx]
            if block_coord[0] == row:
                block = nzb_values[nzb_idx]
                x_idx_base = col_start[wid] + block_coord[1]
                for j in range(6):
                    value += block[j] * body_space[x_idx_base + j]

    v_i = value + problem_v_f[vec_idx]
    P_i = problem_P[vec_idx]
    D_ii_raw = wp.abs(problem_diag[vec_idx]) * P_i * P_i
    D_ii = D_ii_raw + cfg.regularization + FLOAT32_EPS

    # Each active limit row references at most two body blocks. The resulting
    # matrix-free velocity is followed by projection onto the nonnegative ray.
    lambda_limit_old = solution_lambdas[vec_idx]
    lambda_limit_new = lambda_limit_old
    if D_ii_raw > FLOAT32_EPS:
        lambda_limit_new = wp.max(0.0, lambda_limit_old - cfg.omega * v_i / D_ii)
    solution_lambdas[vec_idx] = lambda_limit_new


@wp.kernel
def _solve_dvi_sparse_limits_offset_update(
    # Matrix data:
    num_nzb: wp.array[int32],
    nzb_start: wp.array[int32],
    nzb_coords: wp.array2d[int32],
    nzb_values: wp.array[vec6f],
    row_start: wp.array[int32],
    col_start: wp.array[int32],
    # Active limits:
    limits_model_active: wp.array[int32],
    limits_wid: wp.array[int32],
    limits_lid: wp.array[int32],
    limits_nzb_offsets: wp.array[int32],
    # Problem data:
    problem_vio: wp.array[int32],
    problem_nl: wp.array[int32],
    problem_lcgo: wp.array[int32],
    problem_diag: wp.array[float32],
    problem_P: wp.array[float32],
    problem_v_f: wp.array[float32],
    eta: wp.array[float32],
    body_space: wp.array[float32],
    block_iteration: int32,
    contact_iteration: int32,
    solver_config: wp.array[DVIConfigStruct],
    # Outputs:
    solution_lambdas: wp.array[float32],
):
    _apply_limit_offset_update(
        wp.tid(),
        num_nzb,
        nzb_start,
        nzb_coords,
        nzb_values,
        row_start,
        col_start,
        limits_model_active,
        limits_wid,
        limits_lid,
        limits_nzb_offsets,
        problem_vio,
        problem_nl,
        problem_lcgo,
        problem_diag,
        problem_P,
        problem_v_f,
        eta,
        body_space,
        block_iteration,
        contact_iteration,
        solver_config,
        solution_lambdas,
    )


@wp.func
def _apply_contact_offset_update(
    contact_id: int32,
    # Matrix data:
    num_nzb: wp.array[int32],
    nzb_start: wp.array[int32],
    nzb_coords: wp.array2d[int32],
    nzb_values: wp.array[vec6f],
    row_start: wp.array[int32],
    col_start: wp.array[int32],
    # Active contacts:
    contacts_model_active: wp.array[int32],
    contacts_wid: wp.array[int32],
    contacts_cid: wp.array[int32],
    contacts_nzb_offsets: wp.array[int32],
    # Problem data:
    problem_vio: wp.array[int32],
    problem_nc: wp.array[int32],
    problem_ccgo: wp.array[int32],
    problem_cio: wp.array[int32],
    problem_mu: wp.array[float32],
    problem_diag: wp.array[float32],
    problem_P: wp.array[float32],
    problem_v_f: wp.array[float32],
    eta: wp.array[float32],
    body_space: wp.array[float32],
    contact_block_inv: wp.array[mat33f],
    block_iteration: int32,
    contact_iteration: int32,
    solver_config: wp.array[DVIConfigStruct],
    # Outputs:
    solution_lambdas: wp.array[float32],
):
    if contact_id >= contacts_model_active[0]:
        return

    wid = contacts_wid[contact_id]
    cfg = solver_config[wid]
    if block_iteration >= cfg.block_iterations or contact_iteration >= cfg.contact_iterations:
        return

    cid = contacts_cid[contact_id]
    nc = problem_nc[wid]
    if cid >= nc:
        return

    ccio = problem_ccgo[wid] + int32(3) * cid
    row_0 = ccio + int32(0)
    row_1 = ccio + int32(1)
    row_2 = ccio + int32(2)
    vio = problem_vio[wid]
    ccio_v = vio + ccio
    row_offset = row_start[wid]

    value_0 = eta[row_offset + row_0] * solution_lambdas[ccio_v + 0]
    value_1 = eta[row_offset + row_1] * solution_lambdas[ccio_v + 1]
    value_2 = eta[row_offset + row_2] * solution_lambdas[ccio_v + 2]

    matrix_end = nzb_start[wid] + num_nzb[wid]
    nzb_offset = contacts_nzb_offsets[contact_id]
    for k in range(3):
        nzb_idx = nzb_offset + k
        block = nzb_values[nzb_idx]
        x_idx_base = col_start[wid] + nzb_coords[nzb_idx, 1]
        acc = float32(0.0)
        for j in range(6):
            acc += block[j] * body_space[x_idx_base + j]
        if k == 0:
            value_0 += acc
        elif k == 1:
            value_1 += acc
        else:
            value_2 += acc

    second_body_offset = nzb_offset + 3
    if second_body_offset < matrix_end and nzb_coords[second_body_offset, 0] == row_0:
        for k in range(3):
            nzb_idx = second_body_offset + k
            block = nzb_values[nzb_idx]
            x_idx_base = col_start[wid] + nzb_coords[nzb_idx, 1]
            acc = float32(0.0)
            for j in range(6):
                acc += block[j] * body_space[x_idx_base + j]
            if k == 0:
                value_0 += acc
            elif k == 1:
                value_1 += acc
            else:
                value_2 += acc

    mu_c = problem_mu[problem_cio[wid] + cid]
    v_t0 = value_0 + problem_v_f[ccio_v + 0]
    v_t1 = value_1 + problem_v_f[ccio_v + 1]
    v_n = value_2 + problem_v_f[ccio_v + 2] + mu_c * wp.sqrt(v_t0 * v_t0 + v_t1 * v_t1)
    v_c = vec3f(v_t0, v_t1, v_n)

    P_0 = problem_P[ccio_v + 0]
    P_1 = problem_P[ccio_v + 1]
    P_2 = problem_P[ccio_v + 2]
    D_00 = wp.abs(problem_diag[ccio_v + 0]) * P_0 * P_0
    D_11 = wp.abs(problem_diag[ccio_v + 1]) * P_1 * P_1
    D_22 = wp.abs(problem_diag[ccio_v + 2]) * P_2 * P_2

    # Contact topology offsets select the one or two body blocks contributing
    # to this [t0, t1, n] row triplet before its Coulomb-cone projection.
    lambda_contact_old = vec3f(
        solution_lambdas[ccio_v + 0],
        solution_lambdas[ccio_v + 1],
        solution_lambdas[ccio_v + 2],
    )
    D_diag = _contact_trace_preconditioner(vec3f(D_00, D_11, D_22))
    if cfg.contact_block_preconditioner:
        lambda_projected = _project_contact_block_update(
            lambda_contact_old,
            v_c,
            D_diag,
            contact_block_inv[problem_cio[wid] + cid],
            cfg.regularization,
            cfg.contact_jacobi_omega,
            mu_c,
        )
    else:
        lambda_projected = _project_contact_diagonal_update(
            lambda_contact_old,
            v_c,
            D_diag,
            cfg.regularization,
            cfg.contact_jacobi_omega,
            mu_c,
        )
    lambda_contact_new = lambda_contact_old + cfg.contact_jacobi_relaxation * (lambda_projected - lambda_contact_old)

    solution_lambdas[ccio_v + 0] = lambda_contact_new.x
    solution_lambdas[ccio_v + 1] = lambda_contact_new.y
    solution_lambdas[ccio_v + 2] = lambda_contact_new.z


@wp.kernel
def _solve_dvi_sparse_contacts_offset_update(
    # Matrix data:
    num_nzb: wp.array[int32],
    nzb_start: wp.array[int32],
    nzb_coords: wp.array2d[int32],
    nzb_values: wp.array[vec6f],
    row_start: wp.array[int32],
    col_start: wp.array[int32],
    # Active contacts:
    contacts_model_active: wp.array[int32],
    contacts_wid: wp.array[int32],
    contacts_cid: wp.array[int32],
    contacts_nzb_offsets: wp.array[int32],
    # Problem data:
    problem_vio: wp.array[int32],
    problem_nc: wp.array[int32],
    problem_ccgo: wp.array[int32],
    problem_cio: wp.array[int32],
    problem_mu: wp.array[float32],
    problem_diag: wp.array[float32],
    problem_P: wp.array[float32],
    problem_v_f: wp.array[float32],
    eta: wp.array[float32],
    body_space: wp.array[float32],
    contact_block_inv: wp.array[mat33f],
    block_iteration: int32,
    contact_iteration: int32,
    solver_config: wp.array[DVIConfigStruct],
    # Outputs:
    solution_lambdas: wp.array[float32],
):
    _apply_contact_offset_update(
        wp.tid(),
        num_nzb,
        nzb_start,
        nzb_coords,
        nzb_values,
        row_start,
        col_start,
        contacts_model_active,
        contacts_wid,
        contacts_cid,
        contacts_nzb_offsets,
        problem_vio,
        problem_nc,
        problem_ccgo,
        problem_cio,
        problem_mu,
        problem_diag,
        problem_P,
        problem_v_f,
        eta,
        body_space,
        contact_block_inv,
        block_iteration,
        contact_iteration,
        solver_config,
        solution_lambdas,
    )


@wp.kernel
def _solve_dvi_sparse_unilateral_offset_update(
    limits_capacity: int32,
    # Shared matrix data:
    num_nzb: wp.array[int32],
    nzb_start: wp.array[int32],
    nzb_coords: wp.array2d[int32],
    nzb_values: wp.array[vec6f],
    row_start: wp.array[int32],
    col_start: wp.array[int32],
    # Active limits:
    limits_model_active: wp.array[int32],
    limits_wid: wp.array[int32],
    limits_lid: wp.array[int32],
    limits_nzb_offsets: wp.array[int32],
    problem_nl: wp.array[int32],
    problem_lcgo: wp.array[int32],
    # Active contacts:
    contacts_model_active: wp.array[int32],
    contacts_wid: wp.array[int32],
    contacts_cid: wp.array[int32],
    contacts_nzb_offsets: wp.array[int32],
    problem_nc: wp.array[int32],
    problem_ccgo: wp.array[int32],
    problem_cio: wp.array[int32],
    problem_mu: wp.array[float32],
    contact_block_inv: wp.array[mat33f],
    # Shared problem data:
    problem_vio: wp.array[int32],
    problem_diag: wp.array[float32],
    problem_P: wp.array[float32],
    problem_v_f: wp.array[float32],
    eta: wp.array[float32],
    body_space: wp.array[float32],
    block_iteration: int32,
    contact_iteration: int32,
    solver_config: wp.array[DVIConfigStruct],
    # Outputs:
    solution_lambdas: wp.array[float32],
):
    """Fused limit + contact projection sweep.

    Threads ``[0, limits_capacity)`` update joint limits; the remainder update
    contacts. Both read the same ``body_space`` and write disjoint regions of
    ``solution_lambdas``, so merging them into one launch is equivalent to the
    two separate sweeps while removing a per-iteration kernel launch.
    """
    tid = wp.tid()
    if tid < limits_capacity:
        _apply_limit_offset_update(
            tid,
            num_nzb,
            nzb_start,
            nzb_coords,
            nzb_values,
            row_start,
            col_start,
            limits_model_active,
            limits_wid,
            limits_lid,
            limits_nzb_offsets,
            problem_vio,
            problem_nl,
            problem_lcgo,
            problem_diag,
            problem_P,
            problem_v_f,
            eta,
            body_space,
            block_iteration,
            contact_iteration,
            solver_config,
            solution_lambdas,
        )
    else:
        _apply_contact_offset_update(
            tid - limits_capacity,
            num_nzb,
            nzb_start,
            nzb_coords,
            nzb_values,
            row_start,
            col_start,
            contacts_model_active,
            contacts_wid,
            contacts_cid,
            contacts_nzb_offsets,
            problem_vio,
            problem_nc,
            problem_ccgo,
            problem_cio,
            problem_mu,
            problem_diag,
            problem_P,
            problem_v_f,
            eta,
            body_space,
            contact_block_inv,
            block_iteration,
            contact_iteration,
            solver_config,
            solution_lambdas,
        )


@wp.kernel
def _build_sparse_bilateral_block(
    # Inputs:
    model_bodies_inv_m_i: wp.array[float32],
    data_bodies_inv_I_i: wp.array[mat33f],
    pair_wid: wp.array[int32],
    pair_row: wp.array[int32],
    pair_col: wp.array[int32],
    pair_bid: wp.array[int32],
    pair_i: wp.array[int32],
    pair_j: wp.array[int32],
    jacobian_cts_nzb_values: wp.array[vec6f],
    problem_njc: wp.array[int32],
    bilateral_mio: wp.array[int32],
    bilateral_vio: wp.array[int32],
    bilateral_P: wp.array[float32],
    # Output:
    bilateral_D: wp.array[float32],
):
    pair_id = wp.tid()
    wid = pair_wid[pair_id]
    njc = problem_njc[wid]
    row = pair_row[pair_id]
    col = pair_col[pair_id]
    block_i = jacobian_cts_nzb_values[pair_i[pair_id]]
    block_j = jacobian_cts_nzb_values[pair_j[pair_id]]
    Jv_i = vec3f(block_i[0], block_i[1], block_i[2])
    Jv_j = vec3f(block_j[0], block_j[1], block_j[2])
    Jw_i = vec3f(block_i[3], block_i[4], block_i[5])
    Jw_j = vec3f(block_j[3], block_j[4], block_j[5])

    bid_k = pair_bid[pair_id]
    inv_m_k = model_bodies_inv_m_i[bid_k]
    inv_I_k = data_bodies_inv_I_i[bid_k]
    D_ij = inv_m_k * wp.dot(Jv_i, Jv_j) + wp.dot(Jw_i, inv_I_k @ Jw_j)

    bvio = bilateral_vio[wid]
    p_row = bilateral_P[bvio + row]
    p_col = bilateral_P[bvio + col]
    val = p_row * D_ij * p_col

    bmio = bilateral_mio[wid]
    wp.atomic_add(bilateral_D, bmio + njc * row + col, val)
    wp.atomic_add(bilateral_D, bmio + njc * col + row, val)


@wp.kernel
def _set_sparse_bilateral_diagonal(
    # Inputs:
    problem_njc: wp.array[int32],
    problem_vio: wp.array[int32],
    bilateral_mio: wp.array[int32],
    bilateral_vio: wp.array[int32],
    problem_diag: wp.array[float32],
    # Outputs:
    bilateral_D: wp.array[float32],
    bilateral_P: wp.array[float32],
):
    wid, row = wp.tid()

    njc = problem_njc[wid]
    if row >= njc:
        return

    pvio = problem_vio[wid]
    bvio = bilateral_vio[wid]
    bmio = bilateral_mio[wid]
    diag = wp.abs(problem_diag[pvio + row])
    p = wp.sqrt(1.0 / (diag + FLOAT32_EPS))
    bilateral_P[bvio + row] = p
    bilateral_D[bmio + njc * row + row] = p * diag * p + float32(7.0e-7)


@wp.kernel
def _compute_sparse_contact_block_inverse(
    # Inputs:
    model_info_bodies_offset: wp.array[int32],
    model_bodies_inv_m_i: wp.array[float32],
    data_bodies_inv_I_i: wp.array[mat33f],
    jacobian_cts_nzb_start: wp.array[int32],
    jacobian_cts_num_nzb: wp.array[int32],
    jacobian_cts_nzb_coords: wp.array2d[int32],
    jacobian_cts_nzb_values: wp.array[vec6f],
    problem_nc: wp.array[int32],
    problem_ccgo: wp.array[int32],
    problem_cio: wp.array[int32],
    problem_vio: wp.array[int32],
    problem_P: wp.array[float32],
    solver_config: wp.array[DVIConfigStruct],
    max_num_nzb: int32,
    # Outputs:
    contact_block_inv: wp.array[mat33f],
):
    wid, cid = wp.tid()

    nc = problem_nc[wid]
    cio = problem_cio[wid]
    if cid >= nc:
        return

    cfg = solver_config[wid]
    if not cfg.contact_block_preconditioner:
        contact_block_inv[cio + cid] = mat33f(0.0)
        return

    ccgo = problem_ccgo[wid]
    ccio = ccgo + int32(3) * cid
    nzb_start = jacobian_cts_nzb_start[wid]
    num_nzb = jacobian_cts_num_nzb[wid]
    pvio = problem_vio[wid]

    D = mat33f(0.0)

    for block_id_i in range(max_num_nzb):
        if block_id_i >= num_nzb:
            continue

        global_block_id_i = nzb_start + block_id_i
        block_coords_i = jacobian_cts_nzb_coords[global_block_id_i]
        local_i = block_coords_i[0] - ccio
        if local_i < 0 or local_i >= int32(3):
            continue

        block_i = jacobian_cts_nzb_values[global_block_id_i]
        Jv_i = vec3f(block_i[0], block_i[1], block_i[2])
        Jw_i = vec3f(block_i[3], block_i[4], block_i[5])
        p_i = problem_P[pvio + ccio + local_i]

        for block_id_j in range(max_num_nzb):
            if block_id_j >= num_nzb:
                continue

            global_block_id_j = nzb_start + block_id_j
            block_coords_j = jacobian_cts_nzb_coords[global_block_id_j]
            local_j = block_coords_j[0] - ccio
            if local_j < 0 or local_j >= int32(3) or block_coords_i[1] != block_coords_j[1]:
                continue

            block_j = jacobian_cts_nzb_values[global_block_id_j]
            Jv_j = vec3f(block_j[0], block_j[1], block_j[2])
            Jw_j = vec3f(block_j[3], block_j[4], block_j[5])
            p_j = problem_P[pvio + ccio + local_j]

            bid = model_info_bodies_offset[wid] + block_coords_i[1] // int32(6)
            inv_m = model_bodies_inv_m_i[bid]
            inv_I = data_bodies_inv_I_i[bid]
            D[local_i, local_j] += p_i * (inv_m * wp.dot(Jv_i, Jv_j) + wp.dot(Jw_i, inv_I @ Jw_j)) * p_j

    D[0, 0] += cfg.regularization
    D[1, 1] += cfg.regularization
    D[2, 2] += cfg.regularization

    diag_max = wp.max(wp.max(wp.abs(D[0, 0]), wp.abs(D[1, 1])), wp.abs(D[2, 2]))
    if diag_max > FLOAT32_EPS:
        det = wp.determinant(D)
        det_min = FLOAT32_EPS * diag_max * diag_max * diag_max
        if det > det_min:
            contact_block_inv[cio + cid] = wp.inverse(D)
            return

    contact_block_inv[cio + cid] = mat33f(0.0)


@wp.kernel
def _solve_dvi_sparse_jacobi_update(
    # Inputs:
    problem_dim: wp.array[int32],
    problem_vio: wp.array[int32],
    problem_njc: wp.array[int32],
    problem_nl: wp.array[int32],
    problem_nc: wp.array[int32],
    problem_lcgo: wp.array[int32],
    problem_ccgo: wp.array[int32],
    problem_cio: wp.array[int32],
    problem_mu: wp.array[float32],
    problem_diag: wp.array[float32],
    problem_P: wp.array[float32],
    problem_v_f: wp.array[float32],
    state_v_aug: wp.array[float32],
    contact_block_inv: wp.array[mat33f],
    iteration: int32,
    solver_config: wp.array[DVIConfigStruct],
    # Outputs:
    solution_lambdas: wp.array[float32],
):
    wid, tid = wp.tid()

    ncts = problem_dim[wid]
    if tid >= ncts:
        return

    cfg = solver_config[wid]
    if iteration >= cfg.max_iterations:
        return

    # Every row uses the same matrix-free D * lambda snapshot. Only the first
    # row of each contact updates its three-row block, preserving Jacobi semantics.
    vio = problem_vio[wid]
    v_i = state_v_aug[vio + tid] + problem_v_f[vio + tid]
    P_i = problem_P[vio + tid]
    D_ii_raw = wp.abs(problem_diag[vio + tid]) * P_i * P_i
    D_ii = D_ii_raw + cfg.regularization + FLOAT32_EPS

    njc = problem_njc[wid]
    if tid < njc:
        solution_lambdas[vio + tid] += -cfg.omega * v_i / D_ii
        return

    nl = problem_nl[wid]
    lcgo = problem_lcgo[wid]
    lid = tid - lcgo
    if lid >= 0 and lid < nl:
        lambda_limit_old = solution_lambdas[vio + tid]
        lambda_limit_new = lambda_limit_old
        if D_ii_raw > FLOAT32_EPS:
            lambda_limit_new = wp.max(0.0, lambda_limit_old - cfg.omega * v_i / D_ii)
        solution_lambdas[vio + tid] = lambda_limit_new
        return

    nc = problem_nc[wid]
    ccgo = problem_ccgo[wid]
    contact_row = tid - ccgo
    if contact_row < 0 or contact_row >= int32(3) * nc or contact_row % int32(3) != int32(0):
        return

    cid = contact_row // int32(3)
    ccio = ccgo + int32(3) * cid
    ccio_v = vio + ccio
    mu_c = problem_mu[problem_cio[wid] + cid]

    v_t0 = state_v_aug[ccio_v + 0] + problem_v_f[ccio_v + 0]
    v_t1 = state_v_aug[ccio_v + 1] + problem_v_f[ccio_v + 1]
    v_n = state_v_aug[ccio_v + 2] + problem_v_f[ccio_v + 2] + mu_c * wp.sqrt(v_t0 * v_t0 + v_t1 * v_t1)
    v_c = vec3f(v_t0, v_t1, v_n)

    P_0 = problem_P[ccio_v + 0]
    P_1 = problem_P[ccio_v + 1]
    P_2 = problem_P[ccio_v + 2]
    D_00 = wp.abs(problem_diag[ccio_v + 0]) * P_0 * P_0
    D_11 = wp.abs(problem_diag[ccio_v + 1]) * P_1 * P_1
    D_22 = wp.abs(problem_diag[ccio_v + 2]) * P_2 * P_2

    lambda_contact_old = vec3f(
        solution_lambdas[ccio_v + 0],
        solution_lambdas[ccio_v + 1],
        solution_lambdas[ccio_v + 2],
    )
    D_diag = _contact_trace_preconditioner(vec3f(D_00, D_11, D_22))
    if cfg.contact_block_preconditioner:
        lambda_projected = _project_contact_block_update(
            lambda_contact_old,
            v_c,
            D_diag,
            contact_block_inv[problem_cio[wid] + cid],
            cfg.regularization,
            cfg.contact_jacobi_omega,
            mu_c,
        )
    else:
        lambda_projected = _project_contact_diagonal_update(
            lambda_contact_old,
            v_c,
            D_diag,
            cfg.regularization,
            cfg.contact_jacobi_omega,
            mu_c,
        )
    lambda_contact_new = lambda_contact_old + cfg.contact_jacobi_relaxation * (lambda_projected - lambda_contact_old)

    solution_lambdas[ccio_v + 0] = lambda_contact_new.x
    solution_lambdas[ccio_v + 1] = lambda_contact_new.y
    solution_lambdas[ccio_v + 2] = lambda_contact_new.z


@wp.kernel
def _solve_dvi_sparse_unilateral_jacobi_update(
    # Inputs:
    problem_dim: wp.array[int32],
    problem_vio: wp.array[int32],
    problem_njc: wp.array[int32],
    problem_nl: wp.array[int32],
    problem_nc: wp.array[int32],
    problem_lcgo: wp.array[int32],
    problem_ccgo: wp.array[int32],
    problem_cio: wp.array[int32],
    problem_mu: wp.array[float32],
    problem_diag: wp.array[float32],
    problem_P: wp.array[float32],
    problem_v_f: wp.array[float32],
    state_v_aug: wp.array[float32],
    contact_block_inv: wp.array[mat33f],
    block_iteration: int32,
    contact_iteration: int32,
    solver_config: wp.array[DVIConfigStruct],
    # Outputs:
    solution_lambdas: wp.array[float32],
):
    wid, tid = wp.tid()

    ncts = problem_dim[wid]
    if tid >= ncts:
        return

    cfg = solver_config[wid]
    if block_iteration >= cfg.block_iterations or contact_iteration >= cfg.contact_iterations:
        return

    njc = problem_njc[wid]
    if tid < njc:
        return

    vio = problem_vio[wid]
    v_i = state_v_aug[vio + tid] + problem_v_f[vio + tid]
    P_i = problem_P[vio + tid]
    D_ii_raw = wp.abs(problem_diag[vio + tid]) * P_i * P_i
    D_ii = D_ii_raw + cfg.regularization + FLOAT32_EPS

    nl = problem_nl[wid]
    lcgo = problem_lcgo[wid]
    lid = tid - lcgo
    if lid >= 0 and lid < nl:
        lambda_limit_old = solution_lambdas[vio + tid]
        lambda_limit_new = lambda_limit_old
        if D_ii_raw > FLOAT32_EPS:
            lambda_limit_new = wp.max(0.0, lambda_limit_old - cfg.omega * v_i / D_ii)
        solution_lambdas[vio + tid] = lambda_limit_new
        return

    nc = problem_nc[wid]
    ccgo = problem_ccgo[wid]
    contact_row = tid - ccgo
    if contact_row < 0 or contact_row >= int32(3) * nc or contact_row % int32(3) != int32(0):
        return

    cid = contact_row // int32(3)
    ccio = ccgo + int32(3) * cid
    ccio_v = vio + ccio
    mu_c = problem_mu[problem_cio[wid] + cid]

    v_t0 = state_v_aug[ccio_v + 0] + problem_v_f[ccio_v + 0]
    v_t1 = state_v_aug[ccio_v + 1] + problem_v_f[ccio_v + 1]
    v_n = state_v_aug[ccio_v + 2] + problem_v_f[ccio_v + 2] + mu_c * wp.sqrt(v_t0 * v_t0 + v_t1 * v_t1)
    v_c = vec3f(v_t0, v_t1, v_n)

    P_0 = problem_P[ccio_v + 0]
    P_1 = problem_P[ccio_v + 1]
    P_2 = problem_P[ccio_v + 2]
    D_00 = wp.abs(problem_diag[ccio_v + 0]) * P_0 * P_0
    D_11 = wp.abs(problem_diag[ccio_v + 1]) * P_1 * P_1
    D_22 = wp.abs(problem_diag[ccio_v + 2]) * P_2 * P_2

    lambda_contact_old = vec3f(
        solution_lambdas[ccio_v + 0],
        solution_lambdas[ccio_v + 1],
        solution_lambdas[ccio_v + 2],
    )
    D_diag = _contact_trace_preconditioner(vec3f(D_00, D_11, D_22))
    if cfg.contact_block_preconditioner:
        lambda_projected = _project_contact_block_update(
            lambda_contact_old,
            v_c,
            D_diag,
            contact_block_inv[problem_cio[wid] + cid],
            cfg.regularization,
            cfg.contact_jacobi_omega,
            mu_c,
        )
    else:
        lambda_projected = _project_contact_diagonal_update(
            lambda_contact_old,
            v_c,
            D_diag,
            cfg.regularization,
            cfg.contact_jacobi_omega,
            mu_c,
        )
    lambda_contact_new = lambda_contact_old + cfg.contact_jacobi_relaxation * (lambda_projected - lambda_contact_old)

    solution_lambdas[ccio_v + 0] = lambda_contact_new.x
    solution_lambdas[ccio_v + 1] = lambda_contact_new.y
    solution_lambdas[ccio_v + 2] = lambda_contact_new.z


@wp.kernel
def _compute_dvi_sparse_solution_vectors(
    # Inputs:
    problem_dim: wp.array[int32],
    problem_vio: wp.array[int32],
    problem_v_f: wp.array[float32],
    # Outputs:
    state_s: wp.array[float32],
    state_v_aug: wp.array[float32],
    solution_v_plus: wp.array[float32],
):
    wid, tid = wp.tid()

    ncts = problem_dim[wid]
    if tid >= ncts:
        return

    v_i = problem_vio[wid] + tid
    v_plus = state_v_aug[v_i] + problem_v_f[v_i]
    solution_v_plus[v_i] = v_plus
    state_v_aug[v_i] = v_plus
    state_s[v_i] = 0.0


@wp.kernel
def _set_dvi_sparse_status_iterations(
    # Inputs:
    problem_dim: wp.array[int32],
    solver_config: wp.array[DVIConfigStruct],
    # Outputs:
    solver_status: wp.array[DVIStatus],
):
    wid = wp.tid()
    status = solver_status[wid]
    if problem_dim[wid] == int32(0):
        status.iterations = int32(0)
    else:
        # Sparse Jacobi currently runs a fixed iteration count; terminal DVI
        # residuals are computed by the shared dense/sparse status kernel.
        status.iterations = solver_config[wid].max_iterations
    solver_status[wid] = status
