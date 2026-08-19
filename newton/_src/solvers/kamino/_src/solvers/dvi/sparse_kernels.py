# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Sparse Warp kernels for the Kamino DVI solver."""

from __future__ import annotations

import warp as wp

from ...core.math import FLOAT32_EPS
from ...core.types import vec6f
from .kernels import _FUSED_INEQUALITY_BLOCK, _sync_threads
from .projections import (
    project_contact_normal_update as _project_contact_normal_update,
)
from .projections import (
    project_contact_tangent_update as _project_contact_tangent_update,
)
from .types import DVIConfigStruct

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


@wp.func_native("""
#if defined(__CUDA_ARCH__)
return ((int)__ffsll((long long)mask)) - 1;
#else
if (mask == 0) return -1;
int position = 0;
while ((mask & 1LL) == 0LL) { mask >>= 1; position++; }
return position;
#endif
""")
def _lowest_set_color(mask: wp.int64) -> wp.int32:
    """Return the lowest set bit, or -1 when no bit is set."""
    ...


@wp.kernel
def _color_mapped_dvi_inequalities(
    problem_nl: wp.array[int32],
    problem_nc: wp.array[int32],
    problem_uio: wp.array[int32],
    inequality_bodies: wp.array[wp.vec2i],
    body_color_masks: wp.array[wp.uint64],
    inequality_colors: wp.array[int32],
    inequality_num_colors: wp.array[int32],
    inequality_ids_by_color: wp.array[int32],
    inequality_color_starts: wp.array[int32],
):
    """Greedily color one world per thread using per-body 64-bit masks.

    This favors the many-small-world workload. Unusually high-degree graphs
    that exhaust 64 colors assign fresh colors without a color cap. The same
    pass emits compact color ranges shared by dense and sparse PGS.
    """
    wid = wp.tid()
    nu = problem_nl[wid] + problem_nc[wid]
    uio = problem_uio[wid]
    num_colors = int32(0)
    for uid in range(nu):
        pair = inequality_bodies[uio + uid]
        forbidden = wp.uint64(0)
        if pair[0] >= int32(0):
            forbidden |= body_color_masks[pair[0]]
        if pair[1] >= int32(0):
            forbidden |= body_color_masks[pair[1]]

        color = _lowest_set_color(wp.int64(forbidden) ^ wp.int64(-1))
        if color < int32(0):
            # A fresh color is always conflict-free and avoids a superlinear
            # search in dense manifolds that share a body.
            color = num_colors
        inequality_colors[uio + uid] = color
        num_colors = wp.max(num_colors, color + int32(1))
        if color < int32(64):
            color_bit = wp.uint64(1) << wp.uint64(color)
            if pair[0] >= int32(0):
                body_color_masks[pair[0]] |= color_bit
            if pair[1] >= int32(0):
                body_color_masks[pair[1]] |= color_bit

    inequality_num_colors[wid] = num_colors

    schedule_offset = uio + wid
    for color in range(num_colors + int32(1)):
        inequality_color_starts[schedule_offset + color] = int32(0)
    for uid in range(nu):
        color = inequality_colors[uio + uid]
        inequality_color_starts[schedule_offset + color + int32(1)] += int32(1)
    for color in range(num_colors):
        inequality_color_starts[schedule_offset + color + int32(1)] += inequality_color_starts[schedule_offset + color]
    for uid in range(nu):
        color = inequality_colors[uio + uid]
        slot = inequality_color_starts[schedule_offset + color]
        inequality_ids_by_color[uio + slot] = uid
        inequality_color_starts[schedule_offset + color] = slot + int32(1)
    previous = int32(0)
    for color in range(num_colors + int32(1)):
        cursor = inequality_color_starts[schedule_offset + color]
        inequality_color_starts[schedule_offset + color] = previous
        previous = cursor


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
    inequality_num_colors: wp.array[int32],
    inequality_ids_by_color: wp.array[int32],
    inequality_color_starts: wp.array[int32],
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
    if block_iteration >= int32(0) and block_iteration >= cfg.max_alternating_iterations:
        return
    nl = problem_nl[wid]
    nc = problem_nc[wid]
    nu = nl + nc
    if nu == 0:
        return
    lio = problem_lio[wid]
    cio = problem_cio[wid]
    uio = problem_uio[wid]
    schedule_offset = uio + wid
    lcgo = problem_lcgo[wid]
    ccgo = problem_ccgo[wid]
    vio = problem_vio[wid]
    row_start = bsm_row_start[wid]
    col_start = bsm_col_start[wid]
    matrix_end = bsm_nzb_start[wid] + bsm_num_nzb[wid]
    sweep_count = cfg.inequality_sweeps_per_iteration
    if block_iteration == int32(_FUSED_INEQUALITY_BLOCK):
        sweep_count *= cfg.max_alternating_iterations
    for _sweep in range(sweep_count):
        phase_count = int32(2)
        if block_iteration == int32(_FUSED_INEQUALITY_BLOCK) and _sweep < sweep_count / int32(2):
            # Match the dense path's inequality-only normal-load warmup.
            phase_count = int32(1)
        for phase in range(phase_count):
            # Symmetric tangent ordering reduces load bias in redundant sticking patches.
            reverse_colors = phase == int32(1) and _sweep % int32(2) != int32(0)
            num_colors = inequality_num_colors[wid]
            for color_index in range(num_colors):
                color = color_index
                if reverse_colors:
                    color = num_colors - int32(1) - color_index
                color_start = inequality_color_starts[schedule_offset + color]
                color_end = inequality_color_starts[schedule_offset + color + int32(1)]
                color_slot = color_start + lane
                while color_slot < color_end:
                    uid = inequality_ids_by_color[uio + color_slot]
                    # An inequality without mapped topology has no Jacobian offsets
                    # to read, so it is skipped rather than dereferenced.
                    mapped_id = int32(-1)
                    if uid < nl:
                        mapped_id = limit_indices[lio + uid]
                    else:
                        mapped_id = contact_indices[cio + uid - nl]
                    if mapped_id >= int32(0):
                        if uid < nl:
                            if phase == int32(0):
                                limit_id = mapped_id
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
                                        lambda_limit_old
                                        - cfg.omega * limit_value / (diagonal_raw + cfg.regularization + FLOAT32_EPS),
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
                            contact_id = mapped_id
                            nzb_offset = contact_nzb_offsets[contact_id]
                            block_count = int32(3)
                            second_body_offset = nzb_offset + int32(3)
                            if second_body_offset < matrix_end and bsm_nzb_coords[second_body_offset, 0] == row:
                                block_count = int32(6)

                            contact_value = vec3f(0.0)
                            for component in range(3):
                                if (phase == int32(0) and component == int32(2)) or (
                                    phase == int32(1) and component < int32(2)
                                ):
                                    contact_value[component] = (
                                        eta[row_start + row + component] * solution_lambdas[vec_idx + component]
                                    )
                            for local_block in range(block_count):
                                component = local_block % int32(3)
                                if (phase == int32(0) and component == int32(2)) or (
                                    phase == int32(1) and component < int32(2)
                                ):
                                    nzb_idx = nzb_offset + local_block
                                    block = bsm_nzb_values[nzb_idx]
                                    x_idx_base = col_start + bsm_nzb_coords[nzb_idx, 1]
                                    for j in range(6):
                                        contact_value[component] += block[j] * body_space[x_idx_base + j]

                            contact_delta_body = vec3f(0.0)
                            if phase == int32(0):
                                contact_value.z += problem_v_f[vec_idx + int32(2)]
                                lambda_n_old = solution_lambdas[vec_idx + int32(2)]
                                P_n = problem_P[vec_idx + int32(2)]
                                diagonal_n = wp.abs(problem_diag[vec_idx + int32(2)]) * P_n * P_n
                                lambda_n_new = _project_contact_normal_update(
                                    lambda_n_old,
                                    contact_value.z,
                                    diagonal_n,
                                    cfg.regularization,
                                    cfg.omega,
                                )
                                solution_lambdas[vec_idx + int32(2)] = lambda_n_new
                                contact_delta_body.z = P_n * (lambda_n_new - lambda_n_old)
                            else:
                                contact_value.x += problem_v_f[vec_idx]
                                contact_value.y += problem_v_f[vec_idx + int32(1)]
                                lambda_t0_old = solution_lambdas[vec_idx]
                                lambda_t1_old = solution_lambdas[vec_idx + int32(1)]
                                P_t0 = problem_P[vec_idx]
                                P_t1 = problem_P[vec_idx + int32(1)]
                                diagonal_t0 = wp.abs(problem_diag[vec_idx]) * P_t0 * P_t0
                                diagonal_t1 = wp.abs(problem_diag[vec_idx + int32(1)]) * P_t1 * P_t1
                                lambda_t_old = wp.vec2f(lambda_t0_old, lambda_t1_old)
                                off_diagonal = float32(0.0)
                                body_group = int32(0)
                                while body_group < block_count:
                                    nzb_idx = nzb_offset + body_group
                                    mass_weighted_t0 = bsm_nzb_values[nzb_idx]
                                    jacobian_t1 = jacobian_nzb_values[nzb_idx + int32(1)]
                                    for j in range(6):
                                        off_diagonal += mass_weighted_t0[j] * jacobian_t1[j]
                                    body_group += int32(3)
                                off_diagonal *= P_t1
                                lambda_t_new = _project_contact_tangent_update(
                                    lambda_t_old,
                                    wp.vec2f(contact_value.x, contact_value.y),
                                    wp.vec2f(diagonal_t0, diagonal_t1),
                                    off_diagonal,
                                    cfg.regularization,
                                    cfg.omega,
                                    problem_mu[cio + cid] * solution_lambdas[vec_idx + int32(2)],
                                )
                                solution_lambdas[vec_idx] = lambda_t_new.x
                                solution_lambdas[vec_idx + int32(1)] = lambda_t_new.y
                                contact_delta_body.x = P_t0 * (lambda_t_new.x - lambda_t_old.x)
                                contact_delta_body.y = P_t1 * (lambda_t_new.y - lambda_t_old.y)

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
                    color_slot += threads_per_world
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
