# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Warp kernels for the Kamino DVI solver."""

from __future__ import annotations

import warp as wp

from ...core.math import FLOAT32_EPS
from ..padmm.math import project_to_coulomb_cone, project_to_coulomb_dual_cone
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
    if njc == 0 or tid >= njc * njc:
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
def _solve_dvi_pgs(
    # Inputs:
    problem_dim: wp.array[int32],
    problem_mio: wp.array[int32],
    problem_vio: wp.array[int32],
    problem_njc: wp.array[int32],
    problem_nl: wp.array[int32],
    problem_nc: wp.array[int32],
    problem_lcgo: wp.array[int32],
    problem_ccgo: wp.array[int32],
    problem_cio: wp.array[int32],
    problem_mu: wp.array[float32],
    problem_D: wp.array[float32],
    problem_v_f: wp.array[float32],
    contact_block_inv: wp.array[mat33f],
    solver_config: wp.array[DVIConfigStruct],
    # Outputs:
    solver_status: wp.array[DVIStatus],
    solution_lambdas: wp.array[float32],
):
    wid = wp.tid()

    ncts = problem_dim[wid]
    vio = problem_vio[wid]
    mio = problem_mio[wid]
    njc = problem_njc[wid]
    nl = problem_nl[wid]
    nc = problem_nc[wid]
    lcgo = problem_lcgo[wid]
    ccgo = problem_ccgo[wid]
    cio = problem_cio[wid]
    cfg = solver_config[wid]

    status = DVIStatus()
    status.converged = int32(0)
    status.iterations = int32(0)
    status.r_p = float32(0.0)
    status.r_d = float32(0.0)
    status.r_c = float32(0.0)
    status.r_b = float32(0.0)

    if ncts == 0:
        status.converged = int32(1)
        solver_status[wid] = status
        return

    # This fallback stops on the unnormalized infinity norm of the impulse
    # update. The residual fields below are provisional; solve() later replaces
    # them with terminal cone-feasibility and complementarity diagnostics.
    done = int32(0)
    for iteration in range(cfg.max_iterations):
        if done == 0:
            max_step = float32(0.0)
            max_velocity = float32(0.0)
            max_complementarity = float32(0.0)

            # Equality constraints use scalar Gauss-Seidel updates. This keeps the
            # solve on Kamino's Delassus system while avoiding a separate subsystem.
            for i in range(njc):
                v_i = _compute_row_velocity(ncts, mio, vio, i, problem_D, problem_v_f, solution_lambdas)
                D_ii = wp.abs(problem_D[mio + ncts * i + i]) + cfg.regularization + FLOAT32_EPS
                # Bilateral projection is the identity: lambda += -omega * B * v.
                delta = -cfg.omega * v_i / D_ii
                solution_lambdas[vio + i] += delta
                max_step = wp.max(max_step, wp.abs(delta))
                max_velocity = wp.max(max_velocity, wp.abs(v_i))

            for li in range(nl):
                i = lcgo + li
                v_i = _compute_row_velocity(ncts, mio, vio, i, problem_D, problem_v_f, solution_lambdas)
                D_ii_raw = wp.abs(problem_D[mio + ncts * i + i])
                lambda_limit_old = solution_lambdas[vio + i]
                lambda_limit_new = lambda_limit_old
                if D_ii_raw > FLOAT32_EPS:
                    # Project lambda - omega * B * v onto the nonnegative ray.
                    lambda_limit_new = wp.max(0.0, lambda_limit_old - cfg.omega * v_i / (D_ii_raw + cfg.regularization))
                solution_lambdas[vio + i] = lambda_limit_new
                max_step = wp.max(max_step, wp.abs(lambda_limit_new - lambda_limit_old))
                max_velocity = wp.max(max_velocity, wp.abs(wp.min(v_i, 0.0)))
                max_complementarity = wp.max(max_complementarity, wp.abs(lambda_limit_new * v_i))

            for cid in range(nc):
                ccio = ccgo + 3 * cid
                v_c = _contact_velocity_aug(
                    ncts,
                    mio,
                    vio,
                    ccgo,
                    cio,
                    cid,
                    problem_D,
                    problem_v_f,
                    solution_lambdas,
                    problem_mu,
                )
                D_00 = wp.abs(problem_D[mio + ncts * (ccio + 0) + (ccio + 0)])
                D_11 = wp.abs(problem_D[mio + ncts * (ccio + 1) + (ccio + 1)])
                D_22 = wp.abs(problem_D[mio + ncts * (ccio + 2) + (ccio + 2)])
                lambda_contact_old = vec3f(
                    solution_lambdas[vio + ccio + 0],
                    solution_lambdas[vio + ccio + 1],
                    solution_lambdas[vio + ccio + 2],
                )
                # Project the three-row impulse update onto K_mu. The block
                # preconditioner retains normal-tangential coupling in D_cc.
                if cfg.contact_block_preconditioner:
                    lambda_contact_projected = _project_contact_block_update(
                        lambda_contact_old,
                        v_c,
                        vec3f(D_00, D_11, D_22),
                        contact_block_inv[cio + cid],
                        cfg.regularization,
                        cfg.contact_jacobi_omega,
                        problem_mu[cio + cid],
                    )
                    lambda_contact_new = lambda_contact_old + cfg.contact_jacobi_relaxation * (
                        lambda_contact_projected - lambda_contact_old
                    )
                else:
                    lambda_contact_new = _project_contact_diagonal_update(
                        lambda_contact_old,
                        v_c,
                        _contact_trace_preconditioner(vec3f(D_00, D_11, D_22)),
                        cfg.regularization,
                        cfg.omega,
                        problem_mu[cio + cid],
                    )
                solution_lambdas[vio + ccio + 0] = lambda_contact_new.x
                solution_lambdas[vio + ccio + 1] = lambda_contact_new.y
                solution_lambdas[vio + ccio + 2] = lambda_contact_new.z
                lambda_delta = lambda_contact_new - lambda_contact_old
                max_step = wp.max(max_step, wp.max(wp.abs(lambda_delta)))
                max_velocity = wp.max(max_velocity, wp.max(wp.abs(v_c)))
                max_complementarity = wp.max(max_complementarity, wp.abs(wp.dot(lambda_contact_new, v_c)))

            status.iterations = iteration + int32(1)
            status.r_p = max_step
            status.r_d = max_velocity
            status.r_c = max_complementarity
            if max_step <= cfg.tolerance:
                status.converged = int32(1)
                done = int32(1)

    if done == 0:
        status.converged = int32(0)

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


@wp.kernel
def _solve_dvi_limits_pgs(
    # Inputs:
    problem_dim: wp.array[int32],
    problem_mio: wp.array[int32],
    problem_vio: wp.array[int32],
    problem_nl: wp.array[int32],
    problem_lcgo: wp.array[int32],
    problem_D: wp.array[float32],
    problem_v_f: wp.array[float32],
    block_iteration: int32,
    solver_config: wp.array[DVIConfigStruct],
    # Outputs:
    solver_status: wp.array[DVIStatus],
    solution_lambdas: wp.array[float32],
):
    wid = wp.tid()

    ncts = problem_dim[wid]
    vio = problem_vio[wid]
    mio = problem_mio[wid]
    nl = problem_nl[wid]
    lcgo = problem_lcgo[wid]
    cfg = solver_config[wid]

    if block_iteration >= cfg.block_iterations:
        return

    status = DVIStatus()
    status.converged = int32(0)
    status.iterations = cfg.contact_iterations
    status.r_p = float32(0.0)
    status.r_d = float32(0.0)
    status.r_c = float32(0.0)
    status.r_b = float32(0.0)

    if ncts == 0 or nl == 0:
        status.converged = int32(1)
        solver_status[wid] = status
        return

    # Limit sweeps use the same iterate-change stopping measure as the full
    # dense fallback. Terminal DVI residuals are evaluated after all blocks.
    done = int32(0)
    for iteration in range(cfg.contact_iterations):
        if done == 0:
            max_step = float32(0.0)
            max_velocity = float32(0.0)
            max_complementarity = float32(0.0)

            for li in range(nl):
                i = lcgo + li
                v_i = _compute_row_velocity(ncts, mio, vio, i, problem_D, problem_v_f, solution_lambdas)
                D_ii_raw = wp.abs(problem_D[mio + ncts * i + i])
                lambda_limit_old = solution_lambdas[vio + i]
                lambda_limit_new = lambda_limit_old
                if D_ii_raw > FLOAT32_EPS:
                    lambda_limit_new = wp.max(0.0, lambda_limit_old - cfg.omega * v_i / (D_ii_raw + cfg.regularization))
                solution_lambdas[vio + i] = lambda_limit_new
                max_step = wp.max(max_step, wp.abs(lambda_limit_new - lambda_limit_old))
                max_velocity = wp.max(max_velocity, wp.abs(wp.min(v_i, 0.0)))
                max_complementarity = wp.max(max_complementarity, wp.abs(lambda_limit_new * v_i))

            status.iterations = iteration + int32(1)
            status.r_p = max_step
            status.r_d = max_velocity
            status.r_c = max_complementarity
            if max_step <= cfg.tolerance:
                status.converged = int32(1)
                done = int32(1)

    if done == 0:
        status.converged = int32(0)

    solver_status[wid] = status


@wp.func_native("""
#if defined(__CUDA_ARCH__)
__syncthreads();
#endif
""")
def _sync_threads(): ...


@wp.func
def _contacts_share_dynamic_body(a: wp.vec2i, b: wp.vec2i) -> bool:
    a0 = a[0]
    a1 = a[1]
    b0 = b[0]
    b1 = b[1]
    share = bool(False)
    if a0 >= int32(0):
        if a0 == b0 or a0 == b1:
            share = bool(True)
    if a1 >= int32(0):
        if a1 == b0 or a1 == b1:
            share = bool(True)
    return share


@wp.kernel
def _color_dvi_contacts(
    # Inputs:
    problem_nc: wp.array[int32],
    problem_cio: wp.array[int32],
    contact_bid_AB: wp.array[wp.vec2i],
    # Outputs:
    contact_colors: wp.array[int32],
    contact_num_colors: wp.array[int32],
):
    wid = wp.tid()

    nc = problem_nc[wid]
    cio = problem_cio[wid]
    if nc == 0:
        contact_num_colors[wid] = int32(0)
        return

    # Contacts in one color share no dynamic body, so their Delassus
    # cross-blocks vanish and their Gauss-Seidel updates may run concurrently.
    num_colors = int32(0)
    for cid in range(nc):
        pair = contact_bid_AB[cio + cid]
        color = int32(0)
        found = int32(0)
        while found == int32(0) and color < nc:
            conflict = int32(0)
            prev = int32(0)
            while prev < cid:
                if contact_colors[cio + prev] == color:
                    prev_pair = contact_bid_AB[cio + prev]
                    if _contacts_share_dynamic_body(pair, prev_pair):
                        conflict = int32(1)
                prev = prev + int32(1)

            if conflict == int32(0):
                found = int32(1)
            else:
                color = color + int32(1)

        contact_colors[cio + cid] = color
        num_colors = wp.max(num_colors, color + int32(1))

    contact_num_colors[wid] = num_colors


@wp.kernel
def _solve_dvi_contacts_colored_gs(
    # Inputs:
    problem_dim: wp.array[int32],
    problem_mio: wp.array[int32],
    problem_vio: wp.array[int32],
    problem_nc: wp.array[int32],
    problem_ccgo: wp.array[int32],
    problem_cio: wp.array[int32],
    problem_mu: wp.array[float32],
    problem_D: wp.array[float32],
    block_iteration: int32,
    contact_block_inv: wp.array[mat33f],
    contact_colors: wp.array[int32],
    contact_num_colors: wp.array[int32],
    solver_config: wp.array[DVIConfigStruct],
    # Outputs:
    state_v_aug: wp.array[float32],
    solution_lambdas: wp.array[float32],
):
    tid = wp.tid()
    threads_per_world = int32(wp.block_dim())
    lane = tid % threads_per_world
    wid = tid / threads_per_world

    nc = problem_nc[wid]
    cio = problem_cio[wid]
    if nc == 0:
        return

    ncts = problem_dim[wid]
    vio = problem_vio[wid]
    mio = problem_mio[wid]
    ccgo = problem_ccgo[wid]
    cfg = solver_config[wid]
    if block_iteration >= cfg.block_iterations:
        return

    num_colors = contact_num_colors[wid]
    # Colored contact updates execute a fixed number of sweeps. Convergence is
    # evaluated only after the complete direct-bilateral block schedule.
    iteration = int32(0)
    while iteration < cfg.contact_iterations:
        color = int32(0)
        while color < num_colors:
            cid = lane
            while cid < nc:
                if contact_colors[cio + cid] == color:
                    ccio = ccgo + int32(3) * cid
                    ccio_v = vio + ccio
                    mu_c = problem_mu[cio + cid]

                    v_t0 = state_v_aug[ccio_v + 0]
                    v_t1 = state_v_aug[ccio_v + 1]
                    v_n = state_v_aug[ccio_v + 2] + mu_c * wp.sqrt(v_t0 * v_t0 + v_t1 * v_t1)
                    v_c = vec3f(v_t0, v_t1, v_n)

                    D_00 = wp.abs(problem_D[mio + ncts * (ccio + 0) + (ccio + 0)])
                    D_11 = wp.abs(problem_D[mio + ncts * (ccio + 1) + (ccio + 1)])
                    D_22 = wp.abs(problem_D[mio + ncts * (ccio + 2) + (ccio + 2)])

                    lambda_old = vec3f(
                        solution_lambdas[ccio_v + 0],
                        solution_lambdas[ccio_v + 1],
                        solution_lambdas[ccio_v + 2],
                    )
                    if cfg.contact_block_preconditioner:
                        lambda_projected = _project_contact_block_update(
                            lambda_old,
                            v_c,
                            vec3f(D_00, D_11, D_22),
                            contact_block_inv[cio + cid],
                            cfg.regularization,
                            cfg.contact_jacobi_omega,
                            mu_c,
                        )
                        lambda_new = lambda_old + cfg.contact_jacobi_relaxation * (lambda_projected - lambda_old)
                    else:
                        lambda_new = _project_contact_diagonal_update(
                            lambda_old,
                            v_c,
                            _contact_trace_preconditioner(vec3f(D_00, D_11, D_22)),
                            cfg.regularization,
                            cfg.omega,
                            mu_c,
                        )

                    delta = lambda_new - lambda_old
                    solution_lambdas[ccio_v + 0] = lambda_new.x
                    solution_lambdas[ccio_v + 1] = lambda_new.y
                    solution_lambdas[ccio_v + 2] = lambda_new.z

                    # Only contact velocities are read before all velocities are rebuilt.
                    row = ccgo
                    contact_end = ccgo + int32(3) * nc
                    while row < contact_end:
                        row_mio = mio + ncts * row
                        dv = (
                            problem_D[row_mio + ccio + 0] * delta.x
                            + problem_D[row_mio + ccio + 1] * delta.y
                            + problem_D[row_mio + ccio + 2] * delta.z
                        )
                        wp.atomic_add(state_v_aug, vio + row, dv)
                        row = row + int32(1)

                cid = cid + threads_per_world

            _sync_threads()
            color = color + int32(1)
        iteration = iteration + int32(1)


@wp.kernel
def _compute_dvi_contact_jacobi_delta(
    # Inputs:
    problem_dim: wp.array[int32],
    problem_mio: wp.array[int32],
    problem_vio: wp.array[int32],
    problem_nc: wp.array[int32],
    problem_ccgo: wp.array[int32],
    problem_cio: wp.array[int32],
    problem_mu: wp.array[float32],
    problem_D: wp.array[float32],
    block_iteration: int32,
    contact_iteration: int32,
    solver_config: wp.array[DVIConfigStruct],
    contact_block_inv: wp.array[mat33f],
    state_v_aug: wp.array[float32],
    # Outputs:
    solution_lambdas: wp.array[float32],
    state_scratch: wp.array[float32],
):
    wid, cid = wp.tid()

    nc = problem_nc[wid]
    if cid >= nc:
        return

    ncts = problem_dim[wid]
    vio = problem_vio[wid]
    mio = problem_mio[wid]
    ccgo = problem_ccgo[wid]
    ccio = ccgo + int32(3) * cid
    ccio_v = vio + ccio
    mu_c = problem_mu[problem_cio[wid] + cid]
    cfg = solver_config[wid]
    if block_iteration >= cfg.block_iterations or contact_iteration >= cfg.contact_iterations:
        return

    # All contacts read the same velocity snapshot. Store impulse deltas so the
    # companion kernel can apply their coupled Delassus effect simultaneously.
    v_t0 = state_v_aug[ccio_v + 0]
    v_t1 = state_v_aug[ccio_v + 1]
    v_n = state_v_aug[ccio_v + 2] + mu_c * wp.sqrt(v_t0 * v_t0 + v_t1 * v_t1)
    v_c = vec3f(v_t0, v_t1, v_n)
    D_00 = wp.abs(problem_D[mio + ncts * (ccio + 0) + (ccio + 0)])
    D_11 = wp.abs(problem_D[mio + ncts * (ccio + 1) + (ccio + 1)])
    D_22 = wp.abs(problem_D[mio + ncts * (ccio + 2) + (ccio + 2)])

    lambda_old = vec3f(
        solution_lambdas[ccio_v + 0],
        solution_lambdas[ccio_v + 1],
        solution_lambdas[ccio_v + 2],
    )
    if cfg.contact_block_preconditioner:
        lambda_projected = _project_contact_block_update(
            lambda_old,
            v_c,
            vec3f(D_00, D_11, D_22),
            contact_block_inv[problem_cio[wid] + cid],
            cfg.regularization,
            cfg.contact_jacobi_omega,
            mu_c,
        )
    else:
        lambda_projected = _project_contact_diagonal_update(
            lambda_old,
            v_c,
            _contact_trace_preconditioner(vec3f(D_00, D_11, D_22)),
            cfg.regularization,
            cfg.contact_jacobi_omega,
            mu_c,
        )
    lambda_new = lambda_old + cfg.contact_jacobi_relaxation * (lambda_projected - lambda_old)

    delta = lambda_new - lambda_old
    solution_lambdas[ccio_v + 0] = lambda_new.x
    solution_lambdas[ccio_v + 1] = lambda_new.y
    solution_lambdas[ccio_v + 2] = lambda_new.z
    state_scratch[ccio_v + 0] = delta.x
    state_scratch[ccio_v + 1] = delta.y
    state_scratch[ccio_v + 2] = delta.z


@wp.kernel
def _apply_dvi_contact_jacobi_delta(
    # Inputs:
    problem_dim: wp.array[int32],
    problem_mio: wp.array[int32],
    problem_vio: wp.array[int32],
    problem_nc: wp.array[int32],
    problem_ccgo: wp.array[int32],
    problem_D: wp.array[float32],
    block_iteration: int32,
    contact_iteration: int32,
    solver_config: wp.array[DVIConfigStruct],
    state_scratch: wp.array[float32],
    # Outputs:
    state_v_aug: wp.array[float32],
):
    wid, tid = wp.tid()

    cfg = solver_config[wid]
    if block_iteration >= cfg.block_iterations or contact_iteration >= cfg.contact_iterations:
        return

    nc = problem_nc[wid]
    if tid >= int32(3) * nc:
        return

    ncts = problem_dim[wid]
    vio = problem_vio[wid]
    mio = problem_mio[wid]
    ccgo = problem_ccgo[wid]
    row = ccgo + tid
    row_mio = mio + ncts * row

    # Accumulate D_cc * delta_lambda for the Jacobi sweep; no contact observes
    # another contact's new impulse until every delta has been formed.
    dv = float32(0.0)
    for cid in range(nc):
        ccio = ccgo + int32(3) * cid
        dv += problem_D[row_mio + ccio + 0] * state_scratch[vio + ccio + 0]
        dv += problem_D[row_mio + ccio + 1] * state_scratch[vio + ccio + 1]
        dv += problem_D[row_mio + ccio + 2] * state_scratch[vio + ccio + 2]

    state_v_aug[vio + row] += dv


@wp.kernel
def _compute_dvi_contact_velocities(
    # Inputs:
    problem_dim: wp.array[int32],
    problem_mio: wp.array[int32],
    problem_vio: wp.array[int32],
    problem_nc: wp.array[int32],
    problem_ccgo: wp.array[int32],
    problem_D: wp.array[float32],
    problem_v_f: wp.array[float32],
    solution_lambdas: wp.array[float32],
    # Outputs:
    state_v_aug: wp.array[float32],
):
    wid, tid = wp.tid()

    if tid >= int32(3) * problem_nc[wid]:
        return

    ncts = problem_dim[wid]
    mio = problem_mio[wid]
    vio = problem_vio[wid]
    row = problem_ccgo[wid] + tid
    state_v_aug[vio + row] = _compute_row_velocity(ncts, mio, vio, row, problem_D, problem_v_f, solution_lambdas)


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
