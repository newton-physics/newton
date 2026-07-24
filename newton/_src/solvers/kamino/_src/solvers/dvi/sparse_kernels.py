# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Sparse Warp kernels for the Kamino DVI solver."""

from __future__ import annotations

import warp as wp

from ...core.math import FLOAT32_EPS
from ...core.types import vec6f
from .kernels import _sync_threads
from .projections import (
    contact_normal_preconditioner as _contact_normal_preconditioner,
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


@wp.kernel
def _map_active_limits(
    limits_model_active: wp.array[int32],
    limits_wid: wp.array[int32],
    limits_lid: wp.array[int32],
    limits_bids: wp.array[wp.vec2i],
    problem_lio: wp.array[int32],
    problem_uio: wp.array[int32],
    limit_indices: wp.array[int32],
    inequality_bodies: wp.array[wp.vec2i],
):
    """Map active limits into the unified inequality topology."""
    limit_id = wp.tid()
    if limit_id < limits_model_active[0]:
        wid = limits_wid[limit_id]
        lid = limits_lid[limit_id]
        limit_indices[problem_lio[wid] + lid] = limit_id
        inequality_bodies[problem_uio[wid] + lid] = limits_bids[limit_id]


@wp.kernel
def _map_active_contacts(
    contacts_model_active: wp.array[int32],
    contacts_wid: wp.array[int32],
    contacts_cid: wp.array[int32],
    contacts_bid_AB: wp.array[wp.vec2i],
    problem_nl: wp.array[int32],
    problem_cio: wp.array[int32],
    problem_uio: wp.array[int32],
    contact_indices: wp.array[int32],
    inequality_bodies: wp.array[wp.vec2i],
):
    """Map active contacts into the unified inequality topology."""
    contact_id = wp.tid()
    if contact_id < contacts_model_active[0]:
        wid = contacts_wid[contact_id]
        cid = contacts_cid[contact_id]
        contact_indices[problem_cio[wid] + cid] = contact_id
        inequality_bodies[problem_uio[wid] + problem_nl[wid] + cid] = contacts_bid_AB[contact_id]


@wp.func
def _inequalities_share_dynamic_body(a: wp.vec2i, b: wp.vec2i) -> bool:
    shares_body = bool(False)
    if a[0] >= int32(0) and (a[0] == b[0] or a[0] == b[1]):
        shares_body = bool(True)
    if a[1] >= int32(0) and (a[1] == b[0] or a[1] == b[1]):
        shares_body = bool(True)
    return shares_body


@wp.kernel
def _color_mapped_dvi_inequalities(
    problem_nl: wp.array[int32],
    problem_nc: wp.array[int32],
    problem_uio: wp.array[int32],
    inequality_bodies: wp.array[wp.vec2i],
    inequality_colors: wp.array[int32],
    inequality_num_colors: wp.array[int32],
):
    """Color limits and contacts together using their dynamic body endpoints."""
    wid = wp.tid()
    nu = problem_nl[wid] + problem_nc[wid]
    uio = problem_uio[wid]
    num_colors = int32(0)
    for uid in range(nu):
        pair = inequality_bodies[uio + uid]
        color = int32(0)
        found = int32(0)
        while found == int32(0) and color < nu:
            conflict = int32(0)
            for previous_uid in range(uid):
                if inequality_colors[uio + previous_uid] == color:
                    previous_pair = inequality_bodies[uio + previous_uid]
                    if _inequalities_share_dynamic_body(pair, previous_pair):
                        conflict = int32(1)
            if conflict == int32(0):
                found = int32(1)
            else:
                color += int32(1)
        inequality_colors[uio + uid] = color
        num_colors = wp.max(num_colors, color + int32(1))
    inequality_num_colors[wid] = num_colors


@wp.kernel
def _solve_dvi_sparse_inequalities_pgs(
    bsm_num_nzb: wp.array[int32],
    bsm_nzb_start: wp.array[int32],
    bsm_nzb_coords: wp.array2d[int32],
    bsm_nzb_values: wp.array[vec6f],
    jacobian_nzb_values: wp.array[vec6f],
    bsm_row_start: wp.array[int32],
    bsm_col_start: wp.array[int32],
    limit_nzb_offsets: wp.array[int32],
    contact_nzb_offsets: wp.array[int32],
    limit_indices: wp.array[int32],
    contact_indices: wp.array[int32],
    problem_nl: wp.array[int32],
    problem_nc: wp.array[int32],
    problem_lio: wp.array[int32],
    problem_cio: wp.array[int32],
    problem_uio: wp.array[int32],
    problem_lcgo: wp.array[int32],
    problem_ccgo: wp.array[int32],
    problem_vio: wp.array[int32],
    problem_mu: wp.array[float32],
    problem_P: wp.array[float32],
    problem_v_f: wp.array[float32],
    problem_diag: wp.array[float32],
    eta: wp.array[float32],
    inequality_colors: wp.array[int32],
    inequality_num_colors: wp.array[int32],
    block_iteration: int32,
    solver_config: wp.array[DVIConfigStruct],
    body_space: wp.array[float32],
    solution_lambdas: wp.array[float32],
):
    """Apply one conflict-free sparse PGS schedule to every inequality."""
    tid = wp.tid()
    threads_per_world = int32(wp.block_dim())
    lane = tid % threads_per_world
    wid = tid / threads_per_world
    cfg = solver_config[wid]
    if block_iteration >= int32(0) and block_iteration >= cfg.block_iterations:
        return
    nl = problem_nl[wid]
    nc = problem_nc[wid]
    nu = nl + nc
    if nu == 0:
        return
    lio = problem_lio[wid]
    cio = problem_cio[wid]
    uio = problem_uio[wid]
    lcgo = problem_lcgo[wid]
    ccgo = problem_ccgo[wid]
    vio = problem_vio[wid]
    row_start = bsm_row_start[wid]
    col_start = bsm_col_start[wid]
    matrix_end = bsm_nzb_start[wid] + bsm_num_nzb[wid]
    sweep_budget = cfg.contact_iterations
    if block_iteration < int32(0):
        sweep_budget = cfg.max_iterations

    for _sweep in range(sweep_budget):
        for color in range(inequality_num_colors[wid]):
            uid = lane
            while uid < nu:
                if inequality_colors[uio + uid] == color:
                    if uid < nl:
                        limit_id = limit_indices[lio + uid]
                        row = lcgo + uid
                        vec_idx = vio + row
                        nzb_offset = limit_nzb_offsets[limit_id]
                        limit_value = eta[row_start + row] * solution_lambdas[vec_idx]
                        for k in range(2):
                            nzb_idx = nzb_offset + k
                            if nzb_idx < matrix_end and bsm_nzb_coords[nzb_idx, 0] == row:
                                block = bsm_nzb_values[nzb_idx]
                                x_idx_base = col_start + bsm_nzb_coords[nzb_idx, 1]
                                for j in range(6):
                                    limit_value += block[j] * body_space[x_idx_base + j]
                        limit_value += problem_v_f[vec_idx]
                        P_i = problem_P[vec_idx]
                        diagonal_raw = wp.abs(problem_diag[vec_idx]) * P_i * P_i
                        lambda_limit_old = solution_lambdas[vec_idx]
                        lambda_limit_new = lambda_limit_old
                        if diagonal_raw > FLOAT32_EPS:
                            lambda_limit_new = wp.max(
                                float32(0.0),
                                lambda_limit_old - limit_value / (diagonal_raw + cfg.regularization + FLOAT32_EPS),
                            )
                        limit_delta_body = P_i * (lambda_limit_new - lambda_limit_old)
                        solution_lambdas[vec_idx] = lambda_limit_new
                        for k in range(2):
                            nzb_idx = nzb_offset + k
                            if nzb_idx < matrix_end and bsm_nzb_coords[nzb_idx, 0] == row:
                                x_idx_base = col_start + bsm_nzb_coords[nzb_idx, 1]
                                jacobian_row = jacobian_nzb_values[nzb_idx]
                                for j in range(6):
                                    body_space[x_idx_base + j] += jacobian_row[j] * limit_delta_body
                    else:
                        cid = uid - nl
                        row = ccgo + int32(3) * cid
                        vec_idx = vio + row
                        contact_id = contact_indices[cio + cid]
                        nzb_offset = contact_nzb_offsets[contact_id]
                        contact_value = vec3f(0.0)
                        for component in range(3):
                            contact_value[component] = (
                                eta[row_start + row + component] * solution_lambdas[vec_idx + component]
                            )
                        block_count = int32(3)
                        second_body_offset = nzb_offset + int32(3)
                        if second_body_offset < matrix_end and bsm_nzb_coords[second_body_offset, 0] == row:
                            block_count = int32(6)
                        for local_block in range(block_count):
                            nzb_idx = nzb_offset + local_block
                            component = local_block % int32(3)
                            block = bsm_nzb_values[nzb_idx]
                            x_idx_base = col_start + bsm_nzb_coords[nzb_idx, 1]
                            for j in range(6):
                                contact_value[component] += block[j] * body_space[x_idx_base + j]
                        contact_value += vec3f(problem_v_f[vec_idx], problem_v_f[vec_idx + 1], problem_v_f[vec_idx + 2])
                        mu = problem_mu[cio + cid]
                        contact_value.z += mu * wp.sqrt(
                            contact_value.x * contact_value.x + contact_value.y * contact_value.y
                        )
                        lambda_contact_old = vec3f(
                            solution_lambdas[vec_idx],
                            solution_lambdas[vec_idx + 1],
                            solution_lambdas[vec_idx + 2],
                        )
                        contact_diagonal = vec3f(
                            wp.abs(problem_diag[vec_idx]) * problem_P[vec_idx] * problem_P[vec_idx],
                            wp.abs(problem_diag[vec_idx + 1]) * problem_P[vec_idx + 1] * problem_P[vec_idx + 1],
                            wp.abs(problem_diag[vec_idx + 2]) * problem_P[vec_idx + 2] * problem_P[vec_idx + 2],
                        )
                        lambda_contact_new = _project_contact_diagonal_update(
                            lambda_contact_old,
                            contact_value,
                            _contact_normal_preconditioner(contact_diagonal),
                            cfg.regularization,
                            float32(1.0),
                            mu,
                        )
                        contact_delta = lambda_contact_new - lambda_contact_old
                        contact_delta_body = vec3f(
                            problem_P[vec_idx] * contact_delta.x,
                            problem_P[vec_idx + 1] * contact_delta.y,
                            problem_P[vec_idx + 2] * contact_delta.z,
                        )
                        solution_lambdas[vec_idx] = lambda_contact_new.x
                        solution_lambdas[vec_idx + 1] = lambda_contact_new.y
                        solution_lambdas[vec_idx + 2] = lambda_contact_new.z
                        body_group = int32(0)
                        while body_group < block_count:
                            nzb_idx = nzb_offset + body_group
                            x_idx_base = col_start + bsm_nzb_coords[nzb_idx, 1]
                            row_0 = jacobian_nzb_values[nzb_idx]
                            row_1 = jacobian_nzb_values[nzb_idx + 1]
                            row_2 = jacobian_nzb_values[nzb_idx + 2]
                            for j in range(6):
                                body_space[x_idx_base + j] += (
                                    row_0[j] * contact_delta_body.x
                                    + row_1[j] * contact_delta_body.y
                                    + row_2[j] * contact_delta_body.z
                                )
                            body_group += int32(3)
                uid += threads_per_world
            _sync_threads()


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
    if njc == 0:
        if row == 0:
            bilateral_D[bilateral_mio[wid]] = float32(1.0)
            bilateral_P[bilateral_vio[wid]] = float32(1.0)
        return
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
        # Sparse inequality PGS runs a fixed iteration count; terminal DVI
        # residuals are computed by the shared dense/sparse status kernel.
        status.iterations = solver_config[wid].max_iterations
    solver_status[wid] = status
