# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Warp kernels for the Kamino DVI solver."""

from __future__ import annotations

import warp as wp

from ...core.math import FLOAT32_EPS
from ..padmm.math import project_to_coulomb_cone, project_to_coulomb_dual_cone
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
def _compute_dvi_contact_block_inverse(
    # Inputs:
    problem_dim: wp.array[int32],
    problem_mio: wp.array[int32],
    problem_nc: wp.array[int32],
    problem_ccgo: wp.array[int32],
    problem_cio: wp.array[int32],
    problem_D: wp.array[float32],
    solver_config: wp.array[DVIConfigStruct],
    # Outputs:
    contact_block_inv: wp.array[mat33f],
):
    wid, cid = wp.tid()

    nc = problem_nc[wid]
    if cid >= nc:
        return

    ncts = problem_dim[wid]
    mio = problem_mio[wid]
    ccgo = problem_ccgo[wid]
    cio = problem_cio[wid]
    ccio = ccgo + int32(3) * cid
    cfg = solver_config[wid]
    D_inv = mat33f(0.0)

    if not cfg.contact_block_preconditioner:
        contact_block_inv[cio + cid] = D_inv
        return

    r0 = mio + ncts * (ccio + 0)
    r1 = mio + ncts * (ccio + 1)
    r2 = mio + ncts * (ccio + 2)

    d00 = problem_D[r0 + ccio + 0]
    d01 = float32(0.5) * (problem_D[r0 + ccio + 1] + problem_D[r1 + ccio + 0])
    d02 = float32(0.5) * (problem_D[r0 + ccio + 2] + problem_D[r2 + ccio + 0])
    d11 = problem_D[r1 + ccio + 1]
    d12 = float32(0.5) * (problem_D[r1 + ccio + 2] + problem_D[r2 + ccio + 1])
    d22 = problem_D[r2 + ccio + 2]

    diag_max = wp.max(wp.max(wp.abs(d00), wp.abs(d11)), wp.abs(d22))
    if diag_max > FLOAT32_EPS:
        D_reg = mat33f(
            d00 + cfg.regularization,
            d01,
            d02,
            d01,
            d11 + cfg.regularization,
            d12,
            d02,
            d12,
            d22 + cfg.regularization,
        )
        det = wp.determinant(D_reg)
        det_min = FLOAT32_EPS * diag_max * diag_max * diag_max
        if det > det_min:
            D_inv = wp.inverse(D_reg)

    contact_block_inv[cio + cid] = D_inv


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
    problem_nl: wp.array[int32],
    problem_nc: wp.array[int32],
    problem_lcgo: wp.array[int32],
    problem_ccgo: wp.array[int32],
    problem_cio: wp.array[int32],
    problem_mu: wp.array[float32],
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
    nl = problem_nl[wid]
    nc = problem_nc[wid]
    lcgo = problem_lcgo[wid]
    ccgo = problem_ccgo[wid]
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

    # Thus r_p and r_d are infinity-norm cone-projection distances, while r_c
    # is the maximum absolute impulse-velocity inner product.
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
    status.iterations = cfg.contact_iterations
    status.r_p = float32(0.0)
    status.r_d = float32(0.0)
    status.r_c = float32(0.0)
    status.r_b = float32(0.0)
    solver_status[wid] = status


@wp.kernel
def _set_dvi_direct_status_iterations(
    # Inputs:
    problem_nl: wp.array[int32],
    problem_nc: wp.array[int32],
    solver_config: wp.array[DVIConfigStruct],
    # Outputs:
    solver_status: wp.array[DVIStatus],
):
    wid = wp.tid()
    cfg = solver_config[wid]
    status = solver_status[wid]
    if problem_nl[wid] == int32(0) and problem_nc[wid] == int32(0):
        status.iterations = int32(1)
    else:
        status.iterations = cfg.block_iterations * cfg.contact_iterations
    solver_status[wid] = status


@wp.kernel
def _set_dvi_bilateral_active_dim(
    # Inputs:
    problem_njc: wp.array[int32],
    problem_nl: wp.array[int32],
    problem_nc: wp.array[int32],
    # Outputs:
    bilateral_active_dim: wp.array[int32],
):
    wid = wp.tid()
    active_dim = int32(0)
    if problem_nl[wid] > int32(0) or problem_nc[wid] > int32(0):
        active_dim = problem_njc[wid]
    bilateral_active_dim[wid] = active_dim


@wp.func_native("""
#if defined(__CUDA_ARCH__)
__syncthreads();
#endif
""")
def _sync_threads(): ...


@wp.kernel
def _compute_dvi_unilateral_velocities(
    problem_dim: wp.array[int32],
    problem_mio: wp.array[int32],
    problem_vio: wp.array[int32],
    problem_nl: wp.array[int32],
    problem_nc: wp.array[int32],
    problem_lcgo: wp.array[int32],
    problem_ccgo: wp.array[int32],
    problem_D: wp.array[float32],
    problem_v_f: wp.array[float32],
    solution_lambdas: wp.array[float32],
    state_v_aug: wp.array[float32],
):
    """Evaluate every limit and contact row at the current dual iterate."""
    wid, local_row = wp.tid()
    nl = problem_nl[wid]
    nc = problem_nc[wid]
    unilateral_rows = nl + int32(3) * nc
    if local_row >= unilateral_rows:
        return
    ncts = problem_dim[wid]
    mio = problem_mio[wid]
    vio = problem_vio[wid]
    row = problem_lcgo[wid] + local_row
    state_v_aug[vio + row] = _compute_row_velocity(ncts, mio, vio, row, problem_D, problem_v_f, solution_lambdas)


@wp.kernel
def _solve_dvi_inequalities_colored_pgs(
    problem_dim: wp.array[int32],
    problem_mio: wp.array[int32],
    problem_vio: wp.array[int32],
    problem_nl: wp.array[int32],
    problem_nc: wp.array[int32],
    problem_lcgo: wp.array[int32],
    problem_ccgo: wp.array[int32],
    problem_cio: wp.array[int32],
    problem_uio: wp.array[int32],
    problem_mu: wp.array[float32],
    problem_D: wp.array[float32],
    block_iteration: int32,
    inequality_colors: wp.array[int32],
    inequality_num_colors: wp.array[int32],
    solver_config: wp.array[DVIConfigStruct],
    state_v_aug: wp.array[float32],
    solution_lambdas: wp.array[float32],
):
    """Apply one graph-colored PGS schedule to all DVI inequalities."""
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
    ncts = problem_dim[wid]
    mio = problem_mio[wid]
    vio = problem_vio[wid]
    lcgo = problem_lcgo[wid]
    ccgo = problem_ccgo[wid]
    cio = problem_cio[wid]
    uio = problem_uio[wid]
    contact_end = ccgo + int32(3) * nc
    sweep_budget = cfg.contact_iterations
    if block_iteration < int32(0):
        sweep_budget = cfg.max_iterations

    for _sweep in range(sweep_budget):
        for color in range(inequality_num_colors[wid]):
            uid = lane
            while uid < nu:
                if inequality_colors[uio + uid] == color:
                    delta_0 = float32(0.0)
                    delta_1 = float32(0.0)
                    delta_2 = float32(0.0)
                    column = lcgo + uid
                    column_count = int32(1)
                    if uid < nl:
                        vec_idx = vio + column
                        lambda_limit_old = solution_lambdas[vec_idx]
                        diagonal = wp.abs(problem_D[mio + ncts * column + column])
                        lambda_limit_new = lambda_limit_old
                        if diagonal > FLOAT32_EPS:
                            lambda_limit_new = wp.max(
                                float32(0.0), lambda_limit_old - state_v_aug[vec_idx] / (diagonal + cfg.regularization)
                            )
                        solution_lambdas[vec_idx] = lambda_limit_new
                        delta_0 = lambda_limit_new - lambda_limit_old
                    else:
                        cid = uid - nl
                        column = ccgo + int32(3) * cid
                        column_count = int32(3)
                        vec_idx = vio + column
                        mu = problem_mu[cio + cid]
                        v_t0 = state_v_aug[vec_idx]
                        v_t1 = state_v_aug[vec_idx + int32(1)]
                        velocity = vec3f(
                            v_t0,
                            v_t1,
                            state_v_aug[vec_idx + int32(2)] + mu * wp.sqrt(v_t0 * v_t0 + v_t1 * v_t1),
                        )
                        lambda_contact_old = vec3f(
                            solution_lambdas[vec_idx],
                            solution_lambdas[vec_idx + int32(1)],
                            solution_lambdas[vec_idx + int32(2)],
                        )
                        contact_diagonal = vec3f(
                            wp.abs(problem_D[mio + ncts * column + column]),
                            wp.abs(problem_D[mio + ncts * (column + int32(1)) + column + int32(1)]),
                            wp.abs(problem_D[mio + ncts * (column + int32(2)) + column + int32(2)]),
                        )
                        lambda_contact_new = _project_contact_diagonal_update(
                            lambda_contact_old,
                            velocity,
                            _contact_normal_preconditioner(contact_diagonal),
                            cfg.regularization,
                            float32(1.0),
                            mu,
                        )
                        contact_delta = lambda_contact_new - lambda_contact_old
                        solution_lambdas[vec_idx] = lambda_contact_new.x
                        solution_lambdas[vec_idx + int32(1)] = lambda_contact_new.y
                        solution_lambdas[vec_idx + int32(2)] = lambda_contact_new.z
                        delta_0 = contact_delta.x
                        delta_1 = contact_delta.y
                        delta_2 = contact_delta.z

                    row = lcgo
                    while row < contact_end:
                        row_mio = mio + ncts * row
                        dv = problem_D[row_mio + column] * delta_0
                        if column_count == int32(3):
                            dv += (
                                problem_D[row_mio + column + int32(1)] * delta_1
                                + problem_D[row_mio + column + int32(2)] * delta_2
                            )
                        wp.atomic_add(state_v_aug, vio + row, dv)
                        row += int32(1)
                uid += threads_per_world
            _sync_threads()


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
