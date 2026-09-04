# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Warp kernels for the Kamino DVI solver."""

from __future__ import annotations

import warp as wp

from ...core.math import FLOAT32_EPS
from ..padmm.math import (
    compute_box_complementarity_residual,
    project_to_coulomb_cone,
    project_to_coulomb_dual_cone,
)
from .projections import (
    contact_friction_normal_load as _contact_friction_normal_load,
)
from .projections import (
    project_box_update as _project_box_update,
)
from .projections import (
    project_contact_normal_update as _project_contact_normal_update,
)
from .projections import (
    project_contact_tangent_update as _project_contact_tangent_update,
)
from .types import DVIConfigStruct, DVIStatus

wp.set_module_options({"enable_backward": False})

float32 = wp.float32
int32 = wp.int32
vec3f = wp.vec3f

_FUSED_INEQUALITY_BLOCK = -2
_FUSED_BILATERAL_BLOCK = -3


@wp.func
def _compute_row_velocity(
    ncts: int32,
    mio: int32,
    vio: int32,
    row: int32,
    D: wp.array[float32],
    v_f: wp.array[float32],
    lambdas: wp.array[float32],
) -> float32:
    # Full constraint-space velocity of one row: v_f[row] + sum_j D[row, j] * lambda[j].
    # The sum spans all columns, so a unilateral row picks up the D_ub * lambda_b
    # contribution from joint impulses (and a joint row picks up D_bu * lambda_u).
    v = v_f[vio + row]
    m_i = mio + ncts * row
    for j in range(ncts):
        v += D[m_i + j] * lambdas[vio + j]
    return v


@wp.func
def _contact_velocity_aug(
    ncts: int32,
    mio: int32,
    vio: int32,
    ccgo: int32,
    cio: int32,
    cid: int32,
    D: wp.array[float32],
    v_f: wp.array[float32],
    lambdas: wp.array[float32],
    mu: wp.array[float32],
) -> vec3f:
    # Contact rows are [t0, t1, n]. De Saxce augments the normal velocity by
    # mu * ||v_t|| before enforcing Coulomb-cone complementarity.
    ccio = ccgo + 3 * cid
    v_t0 = _compute_row_velocity(ncts, mio, vio, ccio + 0, D, v_f, lambdas)
    v_t1 = _compute_row_velocity(ncts, mio, vio, ccio + 1, D, v_f, lambdas)
    v_n = _compute_row_velocity(ncts, mio, vio, ccio + 2, D, v_f, lambdas)
    vt_norm = wp.sqrt(v_t0 * v_t0 + v_t1 * v_t1)
    return vec3f(v_t0, v_t1, v_n + mu[cio + cid] * vt_norm)


@wp.kernel
def _reset_dvi_solver_data(
    # Inputs:
    world_mask: wp.array[wp.bool],
    problem_vio: wp.array[int32],
    problem_maxdim: wp.array[int32],
    # Outputs:
    solution_lambdas: wp.array[float32],
    solution_v_plus: wp.array[float32],
):
    wid, tid = wp.tid()
    if not world_mask[wid] or tid >= problem_maxdim[wid]:
        return
    v_i = problem_vio[wid] + tid
    solution_lambdas[v_i] = 0.0
    solution_v_plus[v_i] = 0.0


@wp.kernel
def _scale_dvi_tangential_warmstart(
    model_info_total_cts_offset: wp.array[int32],
    data_info_contact_cts_group_offset: wp.array[int32],
    contact_model_num_contacts: wp.array[int32],
    contact_wid: wp.array[int32],
    contact_cid: wp.array[int32],
    solver_config: wp.array[DVIConfigStruct],
    solution_lambdas: wp.array[float32],
):
    """Decay copied tangential warmstarts while retaining normal warmstarts."""
    cid = wp.tid()
    if cid >= contact_model_num_contacts[0]:
        return

    wid = contact_wid[cid]
    vio_k = model_info_total_cts_offset[wid] + data_info_contact_cts_group_offset[wid] + 3 * contact_cid[cid]
    cfg = solver_config[wid]
    scale = cfg.tangential_warmstart_scale
    solution_lambdas[vio_k] *= scale
    solution_lambdas[vio_k + 1] *= scale


@wp.kernel
def _reset_dvi_status(
    # Outputs:
    solver_status: wp.array[DVIStatus],
):
    wid = wp.tid()
    solver_status[wid] = DVIStatus()


@wp.kernel
def _copy_bilateral_block(
    # Inputs:
    problem_dim: wp.array[int32],
    problem_mio: wp.array[int32],
    problem_njc: wp.array[int32],
    problem_D: wp.array[float32],
    bilateral_mio: wp.array[int32],
    bilateral_vio: wp.array[int32],
    # Outputs:
    bilateral_D: wp.array[float32],
    bilateral_P: wp.array[float32],
):
    wid, tid = wp.tid()

    njc = problem_njc[wid]
    if njc == 0:
        if tid == 0:
            bilateral_D[bilateral_mio[wid]] = float32(1.0)
            bilateral_P[bilateral_vio[wid]] = float32(1.0)
        return
    if tid >= njc * njc:
        return

    ncts = problem_dim[wid]
    pmio = problem_mio[wid]
    bmio = bilateral_mio[wid]
    bvio = bilateral_vio[wid]
    row = tid // njc
    col = tid - row * njc

    D_rr = problem_D[pmio + ncts * row + row]
    D_cc = problem_D[pmio + ncts * col + col]
    p_row = wp.sqrt(1.0 / (wp.abs(D_rr) + FLOAT32_EPS))
    p_col = wp.sqrt(1.0 / (wp.abs(D_cc) + FLOAT32_EPS))

    val = p_row * problem_D[pmio + ncts * row + col] * p_col
    if row == col:
        # Smaller floors reduce equality residual, but closed-loop robots lose contact below this.
        val += float32(7.0e-7)
        bilateral_P[bvio + row] = p_row
    bilateral_D[bmio + njc * row + col] = val


@wp.kernel
def _build_bilateral_rhs(
    # Inputs:
    problem_dim: wp.array[int32],
    problem_mio: wp.array[int32],
    problem_vio: wp.array[int32],
    problem_njc: wp.array[int32],
    problem_D: wp.array[float32],
    problem_v_f: wp.array[float32],
    bilateral_vio: wp.array[int32],
    bilateral_P: wp.array[float32],
    solution_lambdas: wp.array[float32],
    # Outputs:
    bilateral_rhs: wp.array[float32],
):
    wid, row = wp.tid()

    njc = problem_njc[wid]
    if row >= njc:
        return

    ncts = problem_dim[wid]
    pmio = problem_mio[wid]
    pvio = problem_vio[wid]
    bvio = bilateral_vio[wid]

    # Columns njc..ncts are the unilateral rows, so this loop subtracts the
    # D_bu * lambda_u coupling: the current limit and contact impulses enter the
    # joint solve, yielding rhs = -(v_f,b + D_bu * lambda_u).
    rhs = -problem_v_f[pvio + row]
    for col in range(njc, ncts):
        rhs -= problem_D[pmio + ncts * row + col] * solution_lambdas[pvio + col]
    bilateral_rhs[bvio + row] = bilateral_P[bvio + row] * rhs


@wp.kernel
def _scatter_bilateral_solution(
    # Inputs:
    problem_vio: wp.array[int32],
    problem_njc: wp.array[int32],
    bilateral_vio: wp.array[int32],
    bilateral_P: wp.array[float32],
    bilateral_solution: wp.array[float32],
    # Outputs:
    solution_lambdas: wp.array[float32],
):
    wid, row = wp.tid()

    njc = problem_njc[wid]
    if row >= njc:
        return

    bvio = bilateral_vio[wid]
    solution_lambdas[problem_vio[wid] + row] = bilateral_P[bvio + row] * bilateral_solution[bvio + row]


@wp.kernel
def _compute_dvi_status_residuals(
    # Inputs:
    problem_dim: wp.array[int32],
    problem_vio: wp.array[int32],
    problem_njc: wp.array[int32],
    problem_nbc: wp.array[int32],
    problem_nl: wp.array[int32],
    problem_nc: wp.array[int32],
    problem_bcgo: wp.array[int32],
    problem_lcgo: wp.array[int32],
    problem_ccgo: wp.array[int32],
    problem_bcio: wp.array[int32],
    problem_cio: wp.array[int32],
    problem_mu: wp.array[float32],
    problem_bound_lower: wp.array[float32],
    problem_bound_upper: wp.array[float32],
    solver_config: wp.array[DVIConfigStruct],
    state_v_aug: wp.array[float32],
    solution_lambdas: wp.array[float32],
    # Outputs:
    solver_status: wp.array[DVIStatus],
):
    wid = wp.tid()

    ncts = problem_dim[wid]
    vio = problem_vio[wid]
    njc = problem_njc[wid]
    nbc = problem_nbc[wid]
    nl = problem_nl[wid]
    nc = problem_nc[wid]
    bcgo = problem_bcgo[wid]
    lcgo = problem_lcgo[wid]
    ccgo = problem_ccgo[wid]
    bcio = problem_bcio[wid]
    cio = problem_cio[wid]
    cfg = solver_config[wid]

    status = solver_status[wid]
    if status.iterations == 0:
        status.iterations = int32(1)

    # These terminal diagnostics are distinct from the dense fallback's
    # iterate-change stopping test. Each value is a maximum over the world.
    r_b = float32(0.0)
    r_p = float32(0.0)
    r_d = float32(0.0)
    r_c = float32(0.0)

    # Bilateral rows require v_aug = 0.
    for jid in range(njc):
        v_j = state_v_aug[vio + jid]
        r_b = wp.max(r_b, wp.abs(v_j))

    # Bounded-multiplier rows require lambda in the box `[lower, upper]` and directional
    # complementarity with the face selected by the sign of v_aug. There is no dual
    # condition, since v_aug is free to take either sign on a box row.
    for bid in range(nbc):
        bcio_v = vio + bcgo + bid
        lambda_b = solution_lambdas[bcio_v]
        v_b = state_v_aug[bcio_v]
        lower = problem_bound_lower[bcio + bid]
        upper = problem_bound_upper[bcio + bid]
        r_p = wp.max(r_p, wp.abs(lambda_b - wp.clamp(lambda_b, lower, upper)))
        r_c = wp.max(r_c, wp.abs(compute_box_complementarity_residual(lambda_b, v_b, lower, upper)))

    # Limits require lambda and v_aug in R+ with lambda * v_aug = 0.
    for lid in range(nl):
        lcio = vio + lcgo + lid
        lambda_l = solution_lambdas[lcio]
        v_l = state_v_aug[lcio]
        r_p = wp.max(r_p, wp.abs(lambda_l - wp.max(0.0, lambda_l)))
        r_d = wp.max(r_d, wp.abs(v_l - wp.max(0.0, v_l)))
        r_c = wp.max(r_c, wp.abs(lambda_l * v_l))

    # Contacts require lambda in K_mu, v_aug in its dual cone, and orthogonality.
    for cid in range(nc):
        ccio = vio + ccgo + 3 * cid
        mu_c = problem_mu[cio + cid]
        lambda_c = vec3f(solution_lambdas[ccio], solution_lambdas[ccio + 1], solution_lambdas[ccio + 2])
        v_c = vec3f(state_v_aug[ccio], state_v_aug[ccio + 1], state_v_aug[ccio + 2])
        lambda_proj = project_to_coulomb_cone(lambda_c, mu_c)
        v_proj = project_to_coulomb_dual_cone(v_c, mu_c)
        r_p = wp.max(r_p, wp.max(wp.abs(lambda_c - lambda_proj)))
        r_d = wp.max(r_d, wp.max(wp.abs(v_c - v_proj)))
        r_c = wp.max(r_c, wp.abs(wp.dot(lambda_c, v_c)))

    # Thus r_p and r_d are infinity-norm box- and cone-projection distances, while r_c
    # is the maximum absolute impulse-velocity product.
    status.r_b = r_b
    status.r_p = r_p
    status.r_d = wp.max(r_d, r_b)
    status.r_c = r_c
    status.converged = int32(0)
    if ncts == 0 or (r_b <= cfg.tolerance and r_p <= cfg.tolerance and r_d <= cfg.tolerance and r_c <= cfg.tolerance):
        status.converged = int32(1)
    solver_status[wid] = status


@wp.kernel
def _initialize_dvi_status(
    # Inputs:
    solver_config: wp.array[DVIConfigStruct],
    # Outputs:
    solver_status: wp.array[DVIStatus],
):
    wid = wp.tid()
    cfg = solver_config[wid]
    status = DVIStatus()
    status.converged = int32(0)
    status.iterations = cfg.inequality_sweeps_per_iteration
    status.r_p = float32(0.0)
    status.r_d = float32(0.0)
    status.r_c = float32(0.0)
    status.r_b = float32(0.0)
    solver_status[wid] = status


@wp.kernel
def _set_dvi_direct_status_iterations(
    # Inputs:
    problem_nbc: wp.array[int32],
    problem_nl: wp.array[int32],
    problem_nc: wp.array[int32],
    solver_config: wp.array[DVIConfigStruct],
    preserve_reported: wp.bool,
    # Outputs:
    solver_status: wp.array[DVIStatus],
):
    wid = wp.tid()
    cfg = solver_config[wid]
    status = solver_status[wid]
    if not preserve_reported or status.iterations <= int32(0):
        if problem_nbc[wid] == int32(0) and problem_nl[wid] == int32(0) and problem_nc[wid] == int32(0):
            status.iterations = int32(1)
        else:
            status.iterations = cfg.max_alternating_iterations * cfg.inequality_sweeps_per_iteration
    solver_status[wid] = status


@wp.kernel
def _set_dvi_bilateral_active_dim(
    # Inputs:
    problem_njc: wp.array[int32],
    problem_nbc: wp.array[int32],
    problem_nl: wp.array[int32],
    problem_nc: wp.array[int32],
    block_iteration: int32,
    solver_config: wp.array[DVIConfigStruct],
    # Outputs:
    bilateral_active_dim: wp.array[int32],
):
    wid = wp.tid()
    active_dim = int32(0)
    if problem_nbc[wid] > int32(0) or problem_nl[wid] > int32(0) or problem_nc[wid] > int32(0):
        if block_iteration < int32(0):
            active_dim = problem_njc[wid]
        else:
            next_block = block_iteration + int32(1)
            cfg = solver_config[wid]
            if next_block < cfg.max_alternating_iterations and next_block % cfg.bilateral_solve_interval == int32(0):
                active_dim = problem_njc[wid]
    bilateral_active_dim[wid] = active_dim


@wp.func_native("""
#if defined(__CUDA_ARCH__)
__syncthreads();
#endif
""")
def _sync_threads(): ...


@wp.func_native("""
#if defined(__CUDA_ARCH__)
__syncwarp(0xffffffffu);
#endif
""")
def _sync_warp(): ...


@wp.kernel
def _solve_bilateral_contact_response(
    problem_dim: wp.array[int32],
    problem_mio: wp.array[int32],
    problem_njc: wp.array[int32],
    bilateral_mio: wp.array[int32],
    bilateral_vio: wp.array[int32],
    bilateral_P: wp.array[float32],
    projected_mio: wp.array[int32],
    problem_D: wp.array[float32],
    bilateral_L: wp.array[float32],
    bilateral_permutation: wp.array[int32],
    use_permutation: bool,
    unilateral_begin: int32,
    projected_D: wp.array[float32],
):
    wid, unilateral_local = wp.tid()
    ncts = problem_dim[wid]
    njc = problem_njc[wid]
    unilateral = njc + unilateral_begin + unilateral_local
    if njc == int32(0) or unilateral >= ncts:
        return

    source = problem_mio[wid]
    factor = bilateral_mio[wid]
    bvio = bilateral_vio[wid]
    target = projected_mio[wid]

    for row in range(njc):
        original_row = row
        if use_permutation:
            original_row = bilateral_permutation[bvio + row]
        value = bilateral_P[bvio + original_row] * problem_D[source + ncts * original_row + unilateral]
        for k in range(row):
            value -= bilateral_L[factor + njc * row + k] * projected_D[target + ncts * k + unilateral]
        projected_D[target + ncts * row + unilateral] = value / bilateral_L[factor + njc * row + row]

    for reverse_row in range(njc):
        row = njc - int32(1) - reverse_row
        value = projected_D[target + ncts * row + unilateral]
        for k in range(row + int32(1), njc):
            value -= bilateral_L[factor + njc * k + row] * projected_D[target + ncts * k + unilateral]
        projected_D[target + ncts * row + unilateral] = value / bilateral_L[factor + njc * row + row]


@wp.kernel
def _assemble_bilateral_contact_response(
    problem_dim: wp.array[int32],
    problem_mio: wp.array[int32],
    problem_njc: wp.array[int32],
    bilateral_vio: wp.array[int32],
    bilateral_P: wp.array[float32],
    projected_mio: wp.array[int32],
    problem_D: wp.array[float32],
    bilateral_permutation: wp.array[int32],
    use_permutation: bool,
    projected_D: wp.array[float32],
):
    wid, unilateral_row_local, unilateral_column_local = wp.tid()
    ncts = problem_dim[wid]
    njc = problem_njc[wid]
    unilateral_row = njc + unilateral_row_local
    unilateral_column = njc + unilateral_column_local
    if unilateral_row >= ncts or unilateral_column >= ncts:
        return

    source = problem_mio[wid]
    bvio = bilateral_vio[wid]
    target = projected_mio[wid]
    value = problem_D[source + ncts * unilateral_row + unilateral_column]
    for row in range(njc):
        original_row = row
        if use_permutation:
            original_row = bilateral_permutation[bvio + row]
        response = bilateral_P[bvio + original_row] * projected_D[target + ncts * row + unilateral_column]
        value -= problem_D[source + ncts * unilateral_row + original_row] * response
    projected_D[target + ncts * unilateral_row + unilateral_column] = value


@wp.func_native(
    """
#if defined(__CUDA_ARCH__)
    float r = value;
    #pragma unroll
    for (int offset = 8; offset > 0; offset >>= 1)
        r += __shfl_xor_sync(0xffffffffu, r, offset, 16);
    return r;
#else
    return value;
#endif
    """
)
def _subgroup_sum_16(value: float32) -> float32: ...


@wp.kernel
def _solve_bilateral_unilateral_response_cooperative(
    problem_dim: wp.array[int32],
    problem_njc: wp.array[int32],
    bilateral_mio: wp.array[int32],
    bilateral_vio: wp.array[int32],
    bilateral_P: wp.array[float32],
    bilateral_L: wp.array[float32],
    bilateral_permutation: wp.array[int32],
    use_permutation: bool,
    response_mio: wp.array[int32],
    response_stride: wp.array[int32],
    coupling: wp.array[float32],
    response_factor: wp.array[float32],
    response: wp.array[float32],
    first_unilateral: int32,
    tasks_per_world: int32,
):
    """Solve response columns cooperatively with persistent warp workers."""
    # response_factor is unilateral-major here; response always uses
    # original_row * unilateral_stride + unilateral.
    tid = wp.tid()
    lane = tid % int32(32)
    task = tid / int32(32)
    wid = task / tasks_per_world
    task_in_world = task - wid * tasks_per_world
    local_lane = lane % int32(16)
    njc = problem_njc[wid]
    nu = problem_dim[wid] - njc
    factor = bilateral_mio[wid]
    bvio = bilateral_vio[wid]
    offset = response_mio[wid]
    unilateral_stride = response_stride[wid]
    first_pair = (first_unilateral + int32(1)) / int32(2)
    pair_count = (nu + int32(1)) / int32(2)
    for unilateral_pair in range(first_pair + task_in_world, pair_count, tasks_per_world):
        unilateral = int32(2) * unilateral_pair + lane / int32(16)
        active = unilateral < nu
        for row in range(njc):
            partial = float32(0.0)
            if active:
                for k in range(local_lane, row, int32(16)):
                    partial += bilateral_L[factor + njc * row + k] * response_factor[offset + unilateral * njc + k]
            total = _subgroup_sum_16(partial)
            if local_lane == int32(0) and active:
                original_row = row
                if use_permutation:
                    original_row = bilateral_permutation[bvio + row]
                value = (
                    bilateral_P[bvio + original_row] * coupling[offset + original_row * unilateral_stride + unilateral]
                )
                response_factor[offset + unilateral * njc + row] = (value - total) / bilateral_L[
                    factor + njc * row + row
                ]
            _sync_warp()
        for reverse_row in range(njc):
            row = njc - int32(1) - reverse_row
            partial = float32(0.0)
            if active:
                for k in range(row + int32(1) + local_lane, njc, int32(16)):
                    partial += bilateral_L[factor + njc * k + row] * response_factor[offset + unilateral * njc + k]
            total = _subgroup_sum_16(partial)
            if local_lane == int32(0) and active:
                value = response_factor[offset + unilateral * njc + row]
                response_factor[offset + unilateral * njc + row] = (value - total) / bilateral_L[
                    factor + njc * row + row
                ]
            _sync_warp()
        if active:
            for row in range(local_lane, njc, int32(16)):
                original_row = row
                if use_permutation:
                    original_row = bilateral_permutation[bvio + row]
                response[offset + original_row * unilateral_stride + unilateral] = (
                    bilateral_P[bvio + original_row] * response_factor[offset + unilateral * njc + row]
                )


@wp.kernel
def _solve_bilateral_unilateral_response(
    problem_dim: wp.array[int32],
    problem_njc: wp.array[int32],
    bilateral_mio: wp.array[int32],
    bilateral_vio: wp.array[int32],
    bilateral_P: wp.array[float32],
    bilateral_L: wp.array[float32],
    bilateral_permutation: wp.array[int32],
    use_permutation: bool,
    response_mio: wp.array[int32],
    response_stride: wp.array[int32],
    coupling: wp.array[float32],
    response_factor: wp.array[float32],
    response: wp.array[float32],
):
    # response_factor is row-major here; response always uses
    # original_row * unilateral_stride + unilateral.
    tid = wp.tid()
    threads_per_world = int32(wp.block_dim())
    lane = tid % threads_per_world
    wid = tid / threads_per_world
    njc = problem_njc[wid]
    nu = problem_dim[wid] - njc
    factor = bilateral_mio[wid]
    bvio = bilateral_vio[wid]
    offset = response_mio[wid]
    unilateral_stride = response_stride[wid]
    for unilateral in range(lane, nu, threads_per_world):
        for row in range(njc):
            original_row = row
            if use_permutation:
                original_row = bilateral_permutation[bvio + row]
            value = bilateral_P[bvio + original_row] * coupling[offset + original_row * unilateral_stride + unilateral]
            for k in range(row):
                value -= (
                    bilateral_L[factor + njc * row + k] * response_factor[offset + k * unilateral_stride + unilateral]
                )
            response_factor[offset + row * unilateral_stride + unilateral] = (
                value / bilateral_L[factor + njc * row + row]
            )

        for reverse_row in range(njc):
            row = njc - int32(1) - reverse_row
            value = response_factor[offset + row * unilateral_stride + unilateral]
            for k in range(row + int32(1), njc):
                value -= (
                    bilateral_L[factor + njc * k + row] * response_factor[offset + k * unilateral_stride + unilateral]
                )
            response_factor[offset + row * unilateral_stride + unilateral] = (
                value / bilateral_L[factor + njc * row + row]
            )

        for row in range(njc):
            original_row = row
            if use_permutation:
                original_row = bilateral_permutation[bvio + row]
            response[offset + original_row * unilateral_stride + unilateral] = (
                bilateral_P[bvio + original_row] * response_factor[offset + row * unilateral_stride + unilateral]
            )


@wp.kernel
def _compute_dvi_unilateral_velocities(
    problem_dim: wp.array[int32],
    problem_mio: wp.array[int32],
    problem_vio: wp.array[int32],
    problem_nbc: wp.array[int32],
    problem_nl: wp.array[int32],
    problem_nc: wp.array[int32],
    problem_bcgo: wp.array[int32],
    problem_D: wp.array[float32],
    problem_v_f: wp.array[float32],
    solution_lambdas: wp.array[float32],
    state_v_aug: wp.array[float32],
):
    """Evaluate every bounded, limit, and contact row at the current dual iterate."""
    wid, local_row = wp.tid()
    nbc = problem_nbc[wid]
    nl = problem_nl[wid]
    nc = problem_nc[wid]
    unilateral_rows = nbc + nl + int32(3) * nc
    if local_row >= unilateral_rows:
        return
    ncts = problem_dim[wid]
    mio = problem_mio[wid]
    vio = problem_vio[wid]
    row = problem_bcgo[wid] + local_row
    state_v_aug[vio + row] = _compute_row_velocity(ncts, mio, vio, row, problem_D, problem_v_f, solution_lambdas)


@wp.func
def _dense_unilateral_terminal_residuals(
    nl: int32,
    nc: int32,
    lcgo: int32,
    ccgo: int32,
    cio: int32,
    vio: int32,
    lane: int32,
    threads_per_world: int32,
    problem_mu: wp.array[float32],
    problem_P: wp.array[float32],
    state_v_aug: wp.array[float32],
    solution_lambdas: wp.array[float32],
) -> vec3f:
    """Return physical unilateral cone/complementarity residuals owned by one lane."""
    r_p = float32(0.0)
    r_d = float32(0.0)
    r_c = float32(0.0)
    for lid in range(lane, nl, threads_per_world):
        row = vio + lcgo + lid
        scale = problem_P[row]
        lambda_l = scale * solution_lambdas[row]
        velocity_l = state_v_aug[row] / scale
        r_p = wp.max(r_p, wp.abs(lambda_l - wp.max(float32(0.0), lambda_l)))
        r_d = wp.max(r_d, wp.abs(velocity_l - wp.max(float32(0.0), velocity_l)))
        r_c = wp.max(r_c, wp.abs(lambda_l * velocity_l))
    for cid in range(lane, nc, threads_per_world):
        row = vio + ccgo + int32(3) * cid
        lambda_c = vec3f(0.0)
        velocity_c = vec3f(0.0)
        for component in range(3):
            scale = problem_P[row + component]
            lambda_c[component] = scale * solution_lambdas[row + component]
            velocity_c[component] = state_v_aug[row + component] / scale
        mu_c = problem_mu[cio + cid]
        velocity_c[2] += mu_c * wp.sqrt(velocity_c[0] * velocity_c[0] + velocity_c[1] * velocity_c[1])
        lambda_projected = project_to_coulomb_cone(lambda_c, mu_c)
        velocity_projected = project_to_coulomb_dual_cone(velocity_c, mu_c)
        r_p = wp.max(r_p, wp.max(wp.abs(lambda_c - lambda_projected)))
        r_d = wp.max(r_d, wp.max(wp.abs(velocity_c - velocity_projected)))
        r_c = wp.max(r_c, wp.abs(wp.dot(lambda_c, velocity_c)))
    return vec3f(r_p, r_d, r_c)


@wp.kernel
def _solve_dvi_inequalities_colored_pgs(
    problem_dim: wp.array[int32],
    problem_mio: wp.array[int32],
    problem_vio: wp.array[int32],
    problem_nbc: wp.array[int32],
    problem_nl: wp.array[int32],
    problem_nc: wp.array[int32],
    problem_bcgo: wp.array[int32],
    problem_lcgo: wp.array[int32],
    problem_ccgo: wp.array[int32],
    problem_bcio: wp.array[int32],
    problem_cio: wp.array[int32],
    problem_uio: wp.array[int32],
    problem_mu: wp.array[float32],
    problem_bound_lower: wp.array[float32],
    problem_bound_upper: wp.array[float32],
    problem_D: wp.array[float32],
    problem_P: wp.array[float32],
    problem_v_b: wp.array[float32],
    block_iteration: int32,
    inequality_num_colors: wp.array[int32],
    inequality_ids_by_color: wp.array[int32],
    inequality_color_starts: wp.array[int32],
    solver_config: wp.array[DVIConfigStruct],
    enable_adaptive: wp.bool,
    solver_status: wp.array[DVIStatus],
    state_delta: wp.array[float32],
    state_v_aug: wp.array[float32],
    solution_lambdas: wp.array[float32],
):
    """Apply one graph-colored PGS schedule to all DVI inequalities."""
    tid = wp.tid()
    threads_per_world = int32(wp.block_dim())
    lane = tid % threads_per_world
    wid = tid / threads_per_world
    cfg = solver_config[wid]
    if block_iteration >= int32(0) and block_iteration >= cfg.max_alternating_iterations:
        return

    nbc = problem_nbc[wid]
    nl = problem_nl[wid]
    nc = problem_nc[wid]
    nu = nbc + nl + nc
    if nu == 0:
        if enable_adaptive and lane == int32(0):
            status = solver_status[wid]
            status.iterations = int32(1)
            solver_status[wid] = status
        return
    ncts = problem_dim[wid]
    mio = problem_mio[wid]
    vio = problem_vio[wid]
    bcgo = problem_bcgo[wid]
    lcgo = problem_lcgo[wid]
    ccgo = problem_ccgo[wid]
    bcio = problem_bcio[wid]
    cio = problem_cio[wid]
    uio = problem_uio[wid]
    schedule_offset = uio + wid
    contact_end = ccgo + int32(3) * nc
    sweep_count = cfg.inequality_sweeps_per_iteration
    first_tangent_sweep = int32(0)
    if block_iteration == int32(_FUSED_BILATERAL_BLOCK):
        sweep_count *= cfg.max_alternating_iterations
    elif block_iteration == int32(_FUSED_INEQUALITY_BLOCK):
        tangent_sweep_count = sweep_count * cfg.max_alternating_iterations / int32(2)
        sweep_count = (sweep_count + int32(1)) * cfg.max_alternating_iterations
        first_tangent_sweep = sweep_count - tangent_sweep_count
    adaptive = enable_adaptive and cfg.tolerance > float32(0.0)
    completed_sweeps = sweep_count
    pair_update = float32(0.0)
    for _sweep in range(sweep_count):
        sweep_update = float32(0.0)
        phase_count = int32(2)
        if block_iteration == int32(_FUSED_INEQUALITY_BLOCK) and _sweep < first_tangent_sweep:
            # Establish the support load before friction in inequality-only solves.
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
                    delta_0 = float32(0.0)
                    delta_1 = float32(0.0)
                    column = bcgo + uid
                    column_count = int32(1)
                    active = int32(1)
                    if uid < nbc:
                        if phase == int32(1):
                            active = int32(0)
                        else:
                            vec_idx = vio + column
                            bio = bcio + uid
                            lambda_bound_old = solution_lambdas[vec_idx]
                            diagonal = wp.abs(problem_D[mio + ncts * column + column])
                            lambda_bound_new = _project_box_update(
                                lambda_bound_old,
                                state_v_aug[vec_idx],
                                diagonal,
                                cfg.regularization,
                                cfg.omega,
                                problem_bound_lower[bio],
                                problem_bound_upper[bio],
                            )
                            solution_lambdas[vec_idx] = lambda_bound_new
                            delta_0 = lambda_bound_new - lambda_bound_old
                    elif uid < nbc + nl:
                        column = lcgo + (uid - nbc)
                        if phase == int32(1):
                            active = int32(0)
                        else:
                            vec_idx = vio + column
                            lambda_limit_old = solution_lambdas[vec_idx]
                            diagonal = wp.abs(problem_D[mio + ncts * column + column])
                            lambda_limit_new = lambda_limit_old
                            if diagonal > FLOAT32_EPS:
                                lambda_limit_new = wp.max(
                                    float32(0.0),
                                    lambda_limit_old
                                    - cfg.omega * state_v_aug[vec_idx] / (diagonal + cfg.regularization),
                                )
                            solution_lambdas[vec_idx] = lambda_limit_new
                            delta_0 = lambda_limit_new - lambda_limit_old
                    else:
                        cid = uid - nbc - nl
                        column = ccgo + int32(3) * cid
                        if phase == int32(0):
                            column += int32(2)
                            vec_idx = vio + column
                            lambda_n_old = solution_lambdas[vec_idx]
                            diagonal_n = wp.abs(problem_D[mio + ncts * column + column])
                            lambda_n_new = _project_contact_normal_update(
                                lambda_n_old,
                                state_v_aug[vec_idx],
                                diagonal_n,
                                cfg.regularization,
                                cfg.omega,
                            )
                            solution_lambdas[vec_idx] = lambda_n_new
                            delta_0 = lambda_n_new - lambda_n_old
                        else:
                            column_count = int32(2)
                            vec_idx = vio + column
                            lambda_t0_old = solution_lambdas[vec_idx]
                            lambda_t1_old = solution_lambdas[vec_idx + int32(1)]
                            diagonal_t0 = wp.abs(problem_D[mio + ncts * column + column])
                            diagonal_t1 = wp.abs(problem_D[mio + ncts * (column + int32(1)) + column + int32(1)])
                            lambda_t_old = wp.vec2f(lambda_t0_old, lambda_t1_old)
                            lambda_t_new = _project_contact_tangent_update(
                                lambda_t_old,
                                wp.vec2f(state_v_aug[vec_idx], state_v_aug[vec_idx + int32(1)]),
                                wp.vec2f(diagonal_t0, diagonal_t1),
                                problem_D[mio + ncts * column + column + int32(1)],
                                cfg.regularization,
                                cfg.omega,
                                problem_mu[cio + cid]
                                * _contact_friction_normal_load(
                                    solution_lambdas[vec_idx + int32(2)],
                                    problem_v_b[vec_idx + int32(2)],
                                    problem_P[vec_idx + int32(2)],
                                    wp.abs(problem_D[mio + ncts * (column + int32(2)) + column + int32(2)]),
                                    cfg.regularization,
                                    cfg.omega,
                                ),
                            )
                            solution_lambdas[vec_idx] = lambda_t_new.x
                            solution_lambdas[vec_idx + int32(1)] = lambda_t_new.y
                            delta_0 = lambda_t_new.x - lambda_t_old.x
                            delta_1 = lambda_t_new.y - lambda_t_old.y

                    if active != int32(0):
                        if adaptive:
                            sweep_update = wp.max(sweep_update, wp.max(wp.abs(delta_0), wp.abs(delta_1)))
                        state_delta[vio + column] = delta_0
                        if column_count == int32(2):
                            state_delta[vio + column + int32(1)] = delta_1
                    color_slot += threads_per_world

                # The projected updates above have one owner per inequality.
                # Spread their dense velocity updates across the whole block;
                # small contact colors would otherwise leave most lanes idle.
                _sync_threads()
                row_count = contact_end - bcgo
                color_size = color_end - color_start
                update_task = lane
                while update_task < color_size * row_count:
                    local_color_slot = update_task / row_count
                    row = bcgo + update_task - local_color_slot * row_count
                    uid = inequality_ids_by_color[uio + color_start + local_color_slot]
                    active = int32(0)
                    if uid >= nbc + nl or phase == int32(0):
                        active = int32(1)
                    if active != int32(0):
                        column = bcgo + uid
                        column_count = int32(1)
                        if uid >= nbc + nl:
                            cid = uid - nbc - nl
                            column = ccgo + int32(3) * cid
                            if phase == int32(0):
                                column += int32(2)
                            else:
                                column_count = int32(2)
                        elif uid >= nbc:
                            column = lcgo + uid - nbc
                        row_mio = mio + ncts * row
                        dv = problem_D[row_mio + column] * state_delta[vio + column]
                        if column_count == int32(2):
                            dv += problem_D[row_mio + column + int32(1)] * state_delta[vio + column + int32(1)]
                        wp.atomic_add(state_v_aug, vio + row, dv)
                    update_task += threads_per_world
                _sync_threads()

        if adaptive:
            sweep_update_max = wp.tile_max(wp.tile(sweep_update))[0]
            pair_update = wp.max(pair_update, sweep_update_max)
            if (_sweep + int32(1)) % int32(2) == int32(0):
                if pair_update <= cfg.tolerance:
                    local_residuals = _dense_unilateral_terminal_residuals(
                        nl,
                        nc,
                        lcgo,
                        ccgo,
                        cio,
                        vio,
                        lane,
                        threads_per_world,
                        problem_mu,
                        problem_P,
                        state_v_aug,
                        solution_lambdas,
                    )
                    r_p = wp.tile_max(wp.tile(local_residuals[0]))[0]
                    r_d = wp.tile_max(wp.tile(local_residuals[1]))[0]
                    r_c = wp.tile_max(wp.tile(local_residuals[2]))[0]
                    if r_p <= cfg.tolerance and r_d <= cfg.tolerance and r_c <= cfg.tolerance:
                        completed_sweeps = _sweep + int32(1)
                        break
                pair_update = float32(0.0)

    if enable_adaptive and lane == int32(0):
        status = solver_status[wid]
        status.iterations = completed_sweeps
        solver_status[wid] = status


@wp.kernel
def _compute_dvi_solution_vectors(
    # Inputs:
    problem_dim: wp.array[int32],
    problem_mio: wp.array[int32],
    problem_vio: wp.array[int32],
    problem_D: wp.array[float32],
    problem_v_f: wp.array[float32],
    # Outputs:
    state_s: wp.array[float32],
    state_v_aug: wp.array[float32],
    solution_lambdas: wp.array[float32],
    solution_v_plus: wp.array[float32],
):
    wid, tid = wp.tid()

    ncts = problem_dim[wid]
    if tid >= ncts:
        return

    mio = problem_mio[wid]
    vio = problem_vio[wid]
    # Recover the physical post-event velocity v_plus = D * lambda + v_f.
    # De Saxce augmentation is stored separately for cone residual evaluation.
    v_i = _compute_row_velocity(ncts, mio, vio, tid, problem_D, problem_v_f, solution_lambdas)
    solution_v_plus[vio + tid] = v_i
    state_v_aug[vio + tid] = v_i
    state_s[vio + tid] = 0.0


@wp.kernel
def _compute_dvi_desaxce_corrections(
    # Inputs:
    problem_nc: wp.array[int32],
    problem_ccgo: wp.array[int32],
    problem_cio: wp.array[int32],
    problem_vio: wp.array[int32],
    problem_mu: wp.array[float32],
    # Outputs:
    state_s: wp.array[float32],
    state_v_aug: wp.array[float32],
    solution_v_plus: wp.array[float32],
):
    wid, cid = wp.tid()

    nc = problem_nc[wid]
    if cid >= nc:
        return

    vio = problem_vio[wid]
    ccgo = problem_ccgo[wid]
    ccio = ccgo + 3 * cid
    vt0 = solution_v_plus[vio + ccio]
    vt1 = solution_v_plus[vio + ccio + 1]
    # s = [0, 0, mu * ||v_t||] maps physical contact velocity to the dual-cone
    # variable v_aug = v_plus + s used by the DVI contact conditions.
    s_n = problem_mu[problem_cio[wid] + cid] * wp.sqrt(vt0 * vt0 + vt1 * vt1)
    state_s[vio + ccio + 2] = s_n
    state_v_aug[vio + ccio + 2] = solution_v_plus[vio + ccio + 2] + s_n


@wp.kernel
def _unprecondition_dvi_solution(
    # Inputs:
    problem_dim: wp.array[int32],
    problem_vio: wp.array[int32],
    problem_P: wp.array[float32],
    # Outputs:
    state_s: wp.array[float32],
    state_v_aug: wp.array[float32],
    solution_lambdas: wp.array[float32],
    solution_v_plus: wp.array[float32],
):
    wid, tid = wp.tid()

    ncts = problem_dim[wid]
    if tid >= ncts:
        return

    vio = problem_vio[wid]
    v_i = vio + tid
    P_i = problem_P[v_i]
    # The solver uses D_hat = P * D * P: impulses map with P, while
    # constraint-space velocities and De Saxce terms map with P^-1.
    solution_lambdas[v_i] = P_i * solution_lambdas[v_i]
    solution_v_plus[v_i] = solution_v_plus[v_i] / P_i
    state_v_aug[v_i] = state_v_aug[v_i] / P_i
    state_s[v_i] = state_s[v_i] / P_i
