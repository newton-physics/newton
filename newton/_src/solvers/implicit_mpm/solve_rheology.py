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

import gc
import math
from dataclasses import dataclass
from typing import Any

import warp as wp
import warp.fem as fem
import warp.sparse as sp
from warp.fem.linalg import symmetric_eigenvalues_qr
from warp.optim.linear import LinearOperator, cg

_DELASSUS_PROXIMAL_REG = wp.constant(1.0e-6)
"""Cutoff for the trace of the diagonal block of the Delassus operator to disable constraints"""

__SLIDING_NEWTON_TOL = wp.constant(1.0e-7)
"""Tolerance for the Newton method to solve for the sliding velocity"""

_INCLUDE_LEFTOVER_STRAIN = wp.constant(False)
"Whether to include leftover strain (due to not fully-converged implicit solve) in the elastic strain. More accurate, but less stable for stiff materials"

_USE_CAM_CLAY = wp.constant(False)
"""Use Modified Cam-Clay flow rule instead of our custom one"""

vec6 = wp.types.vector(length=6, dtype=wp.float32)

mat66 = wp.types.matrix(shape=(6, 6), dtype=wp.float32)
mat55 = wp.types.matrix(shape=(5, 5), dtype=wp.float32)

mat13 = wp.vec3
mat31 = wp.vec3

wp.set_module_options({"enable_backward": False})


class YieldParamVec(wp.types.vector(length=6, dtype=wp.float32)):
    """Compact yield surface definition in an interpolation-friendly format.

    Layout::

        [0] p_max * sqrt(3/2)       -- scaled compressive yield pressure
        [1] p_min * sqrt(3/2)       -- scaled tensile yield pressure
        [2] s_max                   -- deviatoric yield stress
        [3] mu * p_max              -- frictional shear limit
        [4] dilatancy               -- dilatancy factor
        [5] viscosity               -- viscosity

    The scaling by sqrt(3/2) is related to the orthogonal mapping from spherical/deviatoric
    tensors to vectors in R^6.
    """

    @wp.func
    def from_values(
        friction_coeff: float,
        yield_pressure: float,
        tensile_yield_ratio: float,
        yield_stress: float,
        dilatancy: float,
        viscosity: float,
    ):
        pressure_scale = wp.sqrt(3.0 / 2.0)
        return YieldParamVec(
            yield_pressure * pressure_scale,
            tensile_yield_ratio * yield_pressure * pressure_scale,
            yield_stress,
            friction_coeff * yield_pressure,
            dilatancy,
            viscosity,
        )


@wp.func
def normal_yield_bounds(yield_params: YieldParamVec):
    """Extract bounds for the normal stress from the yield surface definition."""
    return -wp.max(0.0, yield_params[1]), yield_params[0]


@wp.func
def shear_yield_stress_camclay(yield_params: YieldParamVec, r_N: float):
    r_N_min, r_N_max = normal_yield_bounds(yield_params)

    mu = wp.where(r_N_max > 0.0, wp.max(0.0, yield_params[3] / r_N_max), 0.0)

    r_N = wp.clamp(r_N, r_N_min, r_N_max)

    beta_sq = mu * mu / (1.0 - 2.0 * (r_N_min / r_N_max))
    y_sq = beta_sq * (r_N - r_N_min) * (r_N_max - r_N)

    return wp.sqrt(y_sq), 0.0, r_N_min, r_N_max


@wp.func
def shear_yield_stress(yield_params: YieldParamVec, r_N: float):
    """Maximum deviatoric stress for a given value of the normal stress."""
    p_min, p_max = normal_yield_bounds(yield_params)

    mu = wp.where(p_max > 0.0, wp.max(0.0, yield_params[3] / p_max), 0.0)
    s = wp.max(yield_params[2], 0.0)

    r_N = wp.clamp(r_N, p_min, p_max)
    p1 = p_min + 0.5 * p_max
    p2 = 0.5 * p_max
    if r_N < p1:
        return s + mu * (r_N - p_min), mu, p_min, p1
    elif r_N > p2:
        return s + mu * (p_max - r_N), -mu, p2, p_max
    else:
        return s + mu * p2, 0.0, p1, p2


@wp.func
def _B_op(b: wp.vec3, u: wp.vec3):
    return fem.SymmetricTensorMapper.value_to_dof_3d(wp.outer(u, b)) * 2.0


@wp.func
def _B_transposed_op(b: wp.vec3, sig: vec6):
    return wp.vec3(
        wp.dot(_B_op(b, wp.vec3(1.0, 0.0, 0.0)), sig),
        wp.dot(_B_op(b, wp.vec3(0.0, 1.0, 0.0)), sig),
        wp.dot(_B_op(b, wp.vec3(0.0, 0.0, 1.0)), sig),
    )


@wp.kernel
def compute_delassus_diagonal(
    split_mass: wp.bool,
    strain_mat_offsets: wp.array(dtype=int),
    strain_mat_columns: wp.array(dtype=int),
    strain_mat_values: wp.array(dtype=mat13),
    inv_volume: wp.array(dtype=float),
    compliance_mat_offsets: wp.array(dtype=int),
    compliance_mat_columns: wp.array(dtype=int),
    compliance_mat_values: wp.array(dtype=mat66),
    transposed_strain_mat_offsets: wp.array(dtype=int),
    delassus_rotation: wp.array(dtype=mat55),
    delassus_diagonal: wp.array(dtype=vec6),
):
    """Computes the diagonal blocks of the Delassus operator and performs
    an eigendecomposition to decouple stress components.

    For each constraint (tau_i) this kernel:

    1. Assembles the diagonal block of the Delassus operator by summing
       contributions from connected velocity nodes.
    2. If mass splitting is enabled, scales contributions by the number
       of constraints each velocity node participates in.
    3. Zeros the shear-divergence coupling so the normal and deviatoric
       components are decoupled.
    4. Performs an eigendecomposition (``symmetric_eigenvalues_qr``) of
       the deviatoric sub-block, falling back to the diagonal when the
       decomposition is numerically unreliable.
    5. Stores the eigenvalues (``delassus_diagonal``) and the transpose
       of the deviatoric eigenvectors (``delassus_rotation``).
    """
    tau_i = wp.tid()
    block_beg = strain_mat_offsets[tau_i]
    block_end = strain_mat_offsets[tau_i + 1]

    compliance_diag_index = sp.bsr_block_index(tau_i, tau_i, compliance_mat_offsets, compliance_mat_columns)
    if compliance_diag_index == -1:
        diag_block = mat66(0.0)
    else:
        diag_block = compliance_mat_values[compliance_diag_index]

    mass_ratio = float(1.0)
    for b in range(block_beg, block_end):
        u_i = strain_mat_columns[b]

        if split_mass:
            mass_ratio = float(transposed_strain_mat_offsets[u_i + 1] - transposed_strain_mat_offsets[u_i])

        b_val = strain_mat_values[b]
        inv_frac = inv_volume[u_i] * mass_ratio

        b_v0 = _B_op(b_val, wp.vec3(1.0, 0.0, 0.0))
        diag_block += inv_frac * wp.outer(b_v0, b_v0)
        b_v1 = _B_op(b_val, wp.vec3(0.0, 1.0, 0.0))
        diag_block += inv_frac * wp.outer(b_v1, b_v1)
        b_v2 = _B_op(b_val, wp.vec3(0.0, 0.0, 1.0))
        diag_block += inv_frac * wp.outer(b_v2, b_v2)

    diag_block += _DELASSUS_PROXIMAL_REG * wp.identity(n=6, dtype=float)

    # Remove shear-divergence coupling
    # (current implementation of solve_coulomb_aniso normal and tangential responses are independent)
    # Ensures that only the tangential part is rotated
    for k in range(1, 6):
        diag_block[0, k] = 0.0
        diag_block[k, 0] = 0.0

    diag, ev = symmetric_eigenvalues_qr(diag_block, _DELASSUS_PROXIMAL_REG * 0.1)

    # symmetric_eigenvalues_qr may return nans for small coefficients
    if not (wp.ddot(ev, ev) < 1.0e16 and wp.length_sq(diag) < 1.0e16):
        diag = wp.get_diag(diag_block)
        ev = wp.identity(n=6, dtype=float)

    delassus_diagonal[tau_i] = diag
    delassus_rotation[tau_i] = wp.transpose(ev[1:6, 1:6])


@wp.kernel
def preprocess_stress_and_strain(
    unilateral_strain_offset: wp.array(dtype=float),
    strain_rhs: wp.array(dtype=vec6),
    stress: wp.array(dtype=vec6),
    yield_stress: wp.array(dtype=YieldParamVec),
):
    """Prepare stress and strain for the rheology solve.

    Adds the unilateral strain offset to ``strain_rhs`` (removed in
    :func:`postprocess_stress_and_strain`), disables cohesion for nodes
    with a positive offset, and projects the initial stress guess onto
    the yield surface.
    """

    tau_i = wp.tid()

    yield_params = yield_stress[tau_i]
    offset = unilateral_strain_offset[tau_i]

    if offset > 0.0:
        # add unilateral strain offset to strain rhs
        # will be removed in postprocess_stress_and_strain
        b = strain_rhs[tau_i]
        b += fem.SymmetricTensorMapper.value_to_dof_3d(offset / 3.0 * wp.identity(n=3, dtype=float))
        strain_rhs[tau_i] = b

        yield_params[1] = 0.0  # disable cohesion if offset > 0 (not compact)
        yield_stress[tau_i] = yield_params

    sig = stress[tau_i]
    stress[tau_i] = project_stress(sig, yield_params)


@wp.kernel
def postprocess_stress_and_strain(
    compliance_mat_offsets: wp.array(dtype=int),
    compliance_mat_columns: wp.array(dtype=int),
    compliance_mat_values: wp.array(dtype=mat66),
    strain_mat_offsets: wp.array(dtype=int),
    strain_mat_columns: wp.array(dtype=int),
    strain_mat_values: wp.array(dtype=mat13),
    delassus_diagonal: wp.array(dtype=vec6),
    delassus_rotation: wp.array(dtype=mat55),
    unilateral_strain_offset: wp.array(dtype=float),
    yield_params: wp.array(dtype=YieldParamVec),
    strain_node_volume: wp.array(dtype=float),
    strain_rhs: wp.array(dtype=vec6),
    stress: wp.array(dtype=vec6),
    velocity: wp.array(dtype=wp.vec3),
    elastic_strain: wp.array(dtype=vec6),
    plastic_strain: wp.array(dtype=vec6),
):
    """Computes elastic and plastic strain deltas after the solver iterations.

    Removes the unilateral strain offset added by
    :func:`preprocess_stress_and_strain`, accumulates compliance and
    velocity contributions, re-projects the plastic strain through the
    flow rule in the local eigenbasis, and writes the final
    ``elastic_strain`` and ``plastic_strain`` arrays.
    """
    tau_i = wp.tid()

    minus_elastic_strain = strain_rhs[tau_i]
    minus_elastic_strain -= fem.SymmetricTensorMapper.value_to_dof_3d(
        unilateral_strain_offset[tau_i] / 3.0 * wp.identity(n=3, dtype=float)
    )
    comp_block_beg = compliance_mat_offsets[tau_i]
    comp_block_end = compliance_mat_offsets[tau_i + 1]
    for b in range(comp_block_beg, comp_block_end):
        sig_i = compliance_mat_columns[b]
        minus_elastic_strain += compliance_mat_values[b] * stress[sig_i]

    world_plastic_strain = minus_elastic_strain
    block_beg = strain_mat_offsets[tau_i]
    block_end = strain_mat_offsets[tau_i + 1]
    for b in range(block_beg, block_end):
        u_i = strain_mat_columns[b]
        world_plastic_strain += _B_op(strain_mat_values[b], velocity[u_i])

    # Make sure that all strain that does not comply with the yield surface is moved
    # to the elastic strain. That way, if the solver has not fully converged, we track
    # the correct elastic strain.
    rot = delassus_rotation[tau_i]
    diag = delassus_diagonal[tau_i]

    loc_plastic_strain = _world_to_local(world_plastic_strain, rot)
    loc_stress = _world_to_local(stress[tau_i], rot)

    yp = yield_params[tau_i]
    loc_plastic_strain_new = solve_flow_rule_aniso(
        diag, loc_plastic_strain - wp.cw_mul(loc_stress, diag), loc_stress, yp, strain_node_volume[tau_i]
    )
    loc_stress += wp.cw_div(loc_plastic_strain_new - loc_plastic_strain, diag)

    world_plastic_strain_new = _local_to_world(loc_plastic_strain_new, rot)

    if _INCLUDE_LEFTOVER_STRAIN:
        minus_elastic_strain -= world_plastic_strain - world_plastic_strain_new

    elastic_strain[tau_i] = -minus_elastic_strain
    plastic_strain[tau_i] = world_plastic_strain_new


@wp.func
def eval_sliding_residual(alpha: float, D: Any, b_T: Any, gamma: float, mu_rn: float):
    """Evaluates the sliding residual and its derivative w.r.t. ``alpha``.

    The residual is ``f = |r(alpha)| * (1 - gamma * alpha) - mu_rn``
    where ``r(alpha) = b_T / (D + alpha I)``.
    """
    d_alpha = D + type(D)(alpha)

    r_alpha = wp.cw_div(b_T, d_alpha)
    r_alpha_norm = wp.length(r_alpha)
    dr_dalpha = -wp.cw_div(r_alpha, d_alpha * r_alpha_norm)

    g = 1.0 - gamma * alpha

    f = r_alpha_norm * g - mu_rn
    df_dalpha = wp.dot(r_alpha, dr_dalpha) * g - r_alpha_norm * gamma

    return f, df_dalpha


@wp.func
def solve_sliding_aniso(
    D: Any,
    b: Any,
    yield_stress: float,
    yield_stress_deriv: float,
    theta: float,
):
    """Solves the anisotropic sliding sub-problem with dilatancy coupling.

    Finds the velocity ``u`` such that the tangential stress satisfies
    the yield condition, accounting for the normal-tangential coupling
    through ``yield_stress_deriv`` and the dilatancy parameter ``theta``.

    Returns:
        Full velocity vector ``u`` (tangential *and* normal components).
        The normal component ``u[0]`` is set to
        ``theta * yield_stress_deriv * |u_T|``.
    """

    # yield_stress = f_yield( r_N0 )
    # r_N0 = ( u_N0 - b_N )/ D[0]
    # |r_T| = yield_stress + yield_stress_deriv * (r_N - r_N0)
    # |r_T| = yield_stress + yield_stress_deriv * (u_N - u_N0) / D[0]
    # |r_T| = yield_stress_0 + yield_stress_deriv^2 * theta * |u_T| / D[0]
    # |r_T| = yield_stress_0 + yield_stress_deriv^2 * theta / D[0] * alpha * |r_T|
    # (1.0 - yield_stress_deriv^2 * theta / D[0] * alpha) |r_T| = yield_stress

    yield_stress -= yield_stress_deriv * b[0] / D[0]

    b_T = b
    b_T[0] = 0.0
    alpha_0 = wp.length(b_T)

    gamma = theta * yield_stress_deriv * yield_stress_deriv / D[0]
    ref_stress = yield_stress + gamma * alpha_0

    if ref_stress <= 0.0:
        return b_T

    # (1.0 - gamma * alpha) |r_T| = yield_stress
    # (1.0 - gamma * alpha) |(D + alpha I)^{-1} b_t| = yield_stress
    # (1.0 - gamma * alpha) |(D ys + alpha ys I)^{-1} b_t| = 1

    # change of var: alpha -> alpha /yield_stress
    # (1.0 - gamma * alpha) |(D ys + alpha I)^{-1} b_t| = yield_stress/ref_stress
    Dmu_rn = D * ref_stress
    gamma = gamma / ref_stress
    target = yield_stress / ref_stress

    # Viscous shear opposite to tangential stress, zero divergence
    # find alpha, r_t,  mu_rn, (D + alpha/(mu r_n) I) r_t + b_t = 0, |r_t| = mu r_n
    # find alpha,  |(D mu r_n + alpha I)^{-1} b_t|^2 = 1.0

    # |b_T| = tg * (Dz + alpha) / (1 - gamma * alpha)
    # |b_T| (1 - gamma alpha) = tg * (Dz + alpha)
    # |b_T| = (Dz tg + alpha (tg + gamma |b_T|)
    # |b_T| = (Dz tg + alpha) as tg + gamma |b_T| = 1 for def of ref_stress

    alpha_Dmin = alpha_0 - wp.max(Dmu_rn) * target
    alpha_Dmax = alpha_0 - wp.min(Dmu_rn) * target
    alpha_root = 1.0 / gamma

    if target > 0.0:
        alpha_min = wp.max(0.0, alpha_Dmin)
        alpha_max = wp.min(alpha_Dmax, alpha_root)
    elif target < 0.0:
        alpha_min = wp.max(alpha_Dmax, alpha_root)
        alpha_max = alpha_Dmin
    else:
        alpha_max = alpha_root
        alpha_min = alpha_root

    # We're looking for the root of an hyperbola, approach using Newton from the left
    alpha_cur = alpha_min

    for _k in range(24):
        f_cur, df_dalpha = eval_sliding_residual(alpha_cur, Dmu_rn, b_T, gamma, target)

        delta_alpha = wp.min(-f_cur / df_dalpha, alpha_max - alpha_cur)

        if delta_alpha < __SLIDING_NEWTON_TOL * alpha_max:
            break

        alpha_cur += delta_alpha

    u = wp.cw_div(b_T * alpha_cur, Dmu_rn + type(D)(alpha_cur))
    u[0] = theta * yield_stress_deriv * wp.length(u)

    return u


@wp.func
def get_dilatancy(yield_params: YieldParamVec):
    return wp.clamp(yield_params[4], 0.0, 1.0)


@wp.func
def get_viscosity(yield_params: YieldParamVec):
    return wp.max(0.0, yield_params[5])


@wp.func
def solve_flow_rule_camclay(
    D: vec6,
    b: vec6,
    r: vec6,
    yield_params: YieldParamVec,
):
    use_nacc = get_dilatancy(yield_params) == 0.0

    if use_nacc:
        r_0 = -wp.cw_div(b, D)
    else:
        u = wp.cw_mul(r, D) + b
        r_0 = r - u / wp.max(D)

    r_N0 = r_0[0]
    r_T = r_0
    r_T[0] = 0.0

    ys, dys, r_N_min, r_N_max = shear_yield_stress_camclay(yield_params, r_N0)

    if r_N_max <= 0.0:
        return b

    if wp.length_sq(r_T) < ys * ys:
        return vec6(0.0)

    if use_nacc:
        # Non-Associated Cam Clay
        b_T = b
        b_T[0] = 0.0
        u = solve_sliding_aniso(D, b_T, ys, 0.0, 0.0)
        r_N = wp.clamp(r_N0, r_N_min, r_N_max)
        u[0] = D[0] * r_N + b[0]
        return u

    # Associated yield surface: project on 2d ellipse

    mu = wp.where(r_N_max > 0.0, wp.max(0.0, yield_params[3] / r_N_max), 0.0)
    beta_sq = mu * mu / (1.0 - 2.0 * (r_N_min / r_N_max))

    # z = y^2 = beta_sq (r_N_max - r_N) (r_N - r_N_min) = - beta_sq (r_N - r_N_mid)^2 + c^2
    # with c2 = beta_sq * (r_N_mid^2 - r_N_min * r_N_max)
    r_mid = 0.5 * (r_N_min + r_N_max)
    beta = wp.sqrt(beta_sq)
    c_sq = beta_sq * (r_mid * r_mid - r_N_min * r_N_max)
    c = wp.sqrt(c_sq)

    # x = r_N - r_mid
    # y^2 + beta_sq x^2 = c^2

    y = wp.length(r_T)
    x = r_N0 - r_mid

    # Add a dummy normal component so we can reuse the sliding solver
    W = wp.vec3(1.0, beta, 1.0)
    W_sq = wp.vec3(1.0, beta_sq, 1.0)
    W_sq_inv = wp.vec3(1.0, 1.0 / beta_sq, 1.0)

    X0 = wp.vec3(0.0, x, y)
    WinvX0 = wp.cw_div(X0, W)

    # |Y| = c = |W X|
    # W_inv Y + alpha W Y = X0
    # W^-2 Y - W_inv X0 = - alpha Y = Z

    Z = solve_sliding_aniso(W_sq_inv, -WinvX0, c, 0.0, 0.0)
    Y = wp.cw_mul(W_sq, Z + WinvX0)

    X = wp.cw_div(Y, W)

    r_N = r_mid + X[1]
    murn = wp.abs(X[2])

    r = wp.normalize(r_T) * murn
    r[0] = r_N
    u = wp.cw_mul(r, D) + b
    return u


@wp.func
def solve_flow_rule_aniso_impl(
    D: vec6,
    b: vec6,
    r: vec6,
    yield_params: YieldParamVec,
):
    dilatancy = get_dilatancy(yield_params)

    r_0 = -wp.cw_div(b, D)
    r_N0 = r_0[0]

    ys, dys, pmin, pmax = shear_yield_stress(yield_params, r_N0)

    u_N0 = D[0] * (wp.clamp(r_N0, pmin, pmax) - r_N0)

    # u_T = 0 ok
    r_T = r_0
    r_T[0] = 0.0
    r_T_n = wp.length(r_T)
    if r_T_n <= ys:
        u = vec6(0.0)
        u[0] = u_N0
        return u

    # sliding
    u = b
    u[0] = u_N0
    u = solve_sliding_aniso(D, u, ys, dys, dilatancy)

    # check for change of linear region
    r_N_new = (u[0] - b[0]) / D[0]
    r_N_clamp = wp.clamp(r_N_new, pmin, pmax)
    if r_N_clamp == r_N_new:
        return u

    # moved from conic part to constant part. clamp and resolve tangent part
    ys, dys, pmin, pmax = shear_yield_stress(yield_params, r_N_clamp)
    u = solve_sliding_aniso(D, b, ys, 0.0, dilatancy)
    u[0] = D[0] * (r_N_clamp - r_N0)

    return u


@wp.func
def solve_flow_rule_aniso(
    D: vec6,
    b: vec6,
    r_guess: vec6,
    yield_params: YieldParamVec,
    strain_node_volume: float,
):
    """Solves the local non-associated flow-rule problem.

    Applies viscosity scaling to the Delassus diagonal ``D`` and
    right-hand side ``b``, then dispatches to
    :func:`solve_flow_rule_aniso_impl` (or the cam-clay variant).

    The returned velocity ``u`` satisfies ``u = D r + b`` subject to
    the normal stress being clamped to the yield pressure bounds and
    the deviatoric stress satisfying the yield condition with
    dilatancy.
    """

    D_visc = vec6(1.0) + get_viscosity(yield_params) / strain_node_volume * D
    D = wp.cw_div(D, D_visc)
    b = wp.cw_div(b, D_visc)

    if wp.static(_USE_CAM_CLAY):
        return solve_flow_rule_camclay(D, b, r_guess, yield_params)
    else:
        return solve_flow_rule_aniso_impl(D, b, r_guess, yield_params)


@wp.func
def project_stress(
    r: vec6,
    yield_params: YieldParamVec,
):
    """Projects a stress vector onto the yield surface (non-orthogonally)."""

    r_N = r[0]
    r_T = r
    r_T[0] = 0.0

    if wp.static(_USE_CAM_CLAY):
        ys, dys, pmin, pmax = shear_yield_stress_camclay(yield_params, r_N)
    else:
        ys, dys, pmin, pmax = shear_yield_stress(yield_params, r_N)

    r_T_n2 = wp.length_sq(r_T)
    if r_T_n2 > ys * ys:
        r_T *= ys / wp.sqrt(r_T_n2)

    r = r_T
    r[0] = wp.clamp(r_N, pmin, pmax)
    return r


@wp.func
def compute_local_strain(
    tau_i: int,
    compliance_mat_offsets: wp.array(dtype=int),
    compliance_mat_columns: wp.array(dtype=int),
    compliance_mat_values: wp.array(dtype=mat66),
    strain_mat_offsets: wp.array(dtype=int),
    strain_mat_columns: wp.array(dtype=int),
    strain_mat_values: wp.array(dtype=mat13),
    local_strain_rhs: wp.array(dtype=vec6),
    velocities: wp.array(dtype=wp.vec3),
    local_stress: wp.array(dtype=vec6),
):
    """Computes the local strain based on the current stress and velocities."""
    tau = local_strain_rhs[tau_i]

    # tau += B v
    block_beg = strain_mat_offsets[tau_i]
    block_end = strain_mat_offsets[tau_i + 1]
    for b in range(block_beg, block_end):
        u_i = strain_mat_columns[b]
        tau += _B_op(strain_mat_values[b], velocities[u_i])

    # tau += C sigma
    comp_block_beg = compliance_mat_offsets[tau_i]
    comp_block_end = compliance_mat_offsets[tau_i + 1]
    for b in range(comp_block_beg, comp_block_end):
        sig_i = compliance_mat_columns[b]
        tau += compliance_mat_values[b] @ local_stress[sig_i]

    return tau


@wp.func
def _world_to_local(
    world_vec: vec6,
    rotation: mat55,
):
    local_vec = vec6(world_vec[0])
    local_vec[1:6] = world_vec[1:6] @ rotation
    return local_vec


@wp.func
def _local_to_world(
    local_vec: vec6,
    rotation: mat55,
):
    world_vec = vec6(local_vec[0])
    world_vec[1:6] = rotation @ local_vec[1:6]
    return world_vec


@wp.func
def solve_local_stress(
    tau_i: int,
    strain_rhs: vec6,
    yield_params: wp.array(dtype=YieldParamVec),
    strain_node_volume: wp.array(dtype=float),
    delassus_diagonal: wp.array(dtype=vec6),
    delassus_rotation: wp.array(dtype=mat55),
    cur_stress: wp.array(dtype=vec6),
):
    """Computes the stress delta required to satisfy the local flow rule."""

    rot = delassus_rotation[tau_i]
    local_strain = _world_to_local(strain_rhs, rot)

    D = delassus_diagonal[tau_i]
    local_stress = _world_to_local(cur_stress[tau_i], rot)

    tau_new = solve_flow_rule_aniso(
        D,
        local_strain - wp.cw_mul(local_stress, D),
        local_stress,
        yield_params[tau_i],
        strain_node_volume[tau_i],
    )

    return _local_to_world(wp.cw_div(tau_new - local_strain, D), rot)


@wp.kernel
def solve_local_stress_jacobi(
    yield_params: wp.array(dtype=YieldParamVec),
    strain_node_volume: wp.array(dtype=float),
    compliance_mat_offsets: wp.array(dtype=int),
    compliance_mat_columns: wp.array(dtype=int),
    local_compliance_mat_values: wp.array(dtype=mat66),
    strain_mat_offsets: wp.array(dtype=int),
    strain_mat_columns: wp.array(dtype=int),
    strain_mat_values: wp.array(dtype=mat13),
    delassus_diagonal: wp.array(dtype=vec6),
    delassus_rotation: wp.array(dtype=mat55),
    local_strain_rhs: wp.array(dtype=vec6),
    velocities: wp.array(dtype=wp.vec3),
    local_stress: wp.array(dtype=vec6),
    delta_correction: wp.array(dtype=vec6),
):
    """
    Solves the local stress problem for each constraint in a Jacobi-like manner.
    """
    tau_i = wp.tid()

    local_strain = compute_local_strain(
        tau_i,
        compliance_mat_offsets,
        compliance_mat_columns,
        local_compliance_mat_values,
        strain_mat_offsets,
        strain_mat_columns,
        strain_mat_values,
        local_strain_rhs,
        velocities,
        local_stress,
    )

    delta_correction[tau_i] = solve_local_stress(
        tau_i,
        local_strain,
        yield_params,
        strain_node_volume,
        delassus_diagonal,
        delassus_rotation,
        local_stress,
    )


@wp.func
def apply_stress_delta(
    tau_i: int,
    delta_stress: vec6,
    strain_mat_offsets: wp.array(dtype=int),
    strain_mat_columns: wp.array(dtype=int),
    strain_mat_values: wp.array(dtype=mat13),
    inv_mass_matrix: wp.array(dtype=float),
    velocities: wp.array(dtype=wp.vec3),
):
    """Updates particle velocities from a local stress delta."""

    block_beg = strain_mat_offsets[tau_i]
    block_end = strain_mat_offsets[tau_i + 1]

    for b in range(block_beg, block_end):
        u_i = strain_mat_columns[b]
        strain_val = strain_mat_values[b]
        delta_u = _B_transposed_op(strain_val, delta_stress)
        velocities[u_i] += inv_mass_matrix[u_i] * delta_u


@wp.kernel
def apply_stress_delta_jacobi(
    transposed_strain_mat_offsets: wp.array(dtype=int),
    transposed_strain_mat_columns: wp.array(dtype=int),
    transposed_strain_mat_values: wp.array(dtype=mat13),
    inv_mass_matrix: wp.array(dtype=float),
    stress: wp.array(dtype=vec6),
    velocities: wp.array(dtype=wp.vec3),
):
    """Updates particle velocities from a local stress delta."""

    u_i = wp.tid()

    inv_mass = inv_mass_matrix[u_i]

    block_beg = transposed_strain_mat_offsets[u_i]
    block_end = transposed_strain_mat_offsets[u_i + 1]

    delta_u = wp.vec3(0.0)
    for b in range(block_beg, block_end):
        tau_i = transposed_strain_mat_columns[b]
        delta_stress = stress[tau_i]
        delta_u += _B_transposed_op(transposed_strain_mat_values[b], delta_stress)

    velocities[u_i] += inv_mass * delta_u


@wp.kernel
def apply_velocity_delta(
    alpha: float,
    beta: float,
    strain_mat_offsets: wp.array(dtype=int),
    strain_mat_columns: wp.array(dtype=int),
    strain_mat_values: wp.array(dtype=mat13),
    velocity_delta: wp.array(dtype=wp.vec3),
    strain_prev: wp.array(dtype=vec6),
    strain: wp.array(dtype=vec6),
):
    """Computes strain from a velocity delta: ``strain = alpha * B @ velocity_delta + beta * strain_prev``."""

    tau_i = wp.tid()

    block_beg = strain_mat_offsets[tau_i]
    block_end = strain_mat_offsets[tau_i + 1]

    delta_stress = vec6(0.0)
    for b in range(block_beg, block_end):
        u_i = strain_mat_columns[b]
        delta_stress += _B_op(strain_mat_values[b], velocity_delta[u_i])

    delta_stress *= alpha
    if beta != 0.0:
        delta_stress += beta * strain_prev[tau_i]

    strain[tau_i] = delta_stress


@wp.kernel
def apply_stress_gs(
    color: int,
    launch_dim: int,
    color_offsets: wp.array(dtype=int),
    color_blocks: wp.array2d(dtype=int),
    strain_mat_offsets: wp.array(dtype=int),
    strain_mat_columns: wp.array(dtype=int),
    strain_mat_values: wp.array(dtype=mat13),
    inv_mass_matrix: wp.array(dtype=float),  # Note: Likely inv_volume in context
    stress: wp.array(dtype=vec6),
    velocities: wp.array(dtype=wp.vec3),
):
    """
    Update particle velocities from the current stress. Uses a coloring approach to
    avoid avoid race conditions. Used for Gauss-Seidel solver where the transposed
    strain matrix is not assembled
    """

    i = wp.tid()
    color_beg = color_offsets[color] + i
    color_end = color_offsets[color + 1]

    for color_offset in range(color_beg, color_end, launch_dim):
        beg, end = color_blocks[0, color_offset], color_blocks[1, color_offset]
        for tau_i in range(beg, end):
            cur_stress = stress[tau_i]

            apply_stress_delta(
                tau_i,
                cur_stress,
                strain_mat_offsets,
                strain_mat_columns,
                strain_mat_values,
                inv_mass_matrix,
                velocities,
            )


@wp.kernel
def solve_local_stress_gs(
    color: int,
    launch_dim: int,
    color_offsets: wp.array(dtype=int),
    color_blocks: wp.array2d(dtype=int),
    yield_params: wp.array(dtype=YieldParamVec),
    strain_node_volume: wp.array(dtype=float),
    compliance_mat_offsets: wp.array(dtype=int),
    compliance_mat_columns: wp.array(dtype=int),
    compliance_mat_values: wp.array(dtype=mat66),
    strain_mat_offsets: wp.array(dtype=int),
    strain_mat_columns: wp.array(dtype=int),
    strain_mat_values: wp.array(dtype=mat13),
    delassus_diagonal: wp.array(dtype=vec6),
    delassus_rotation: wp.array(dtype=mat55),
    inv_mass_matrix: wp.array(dtype=float),  # Note: Likely inv_volume in context
    local_strain_rhs: wp.array(dtype=vec6),
    velocities: wp.array(dtype=wp.vec3),
    local_stress: wp.array(dtype=vec6),
    delta_correction: wp.array(dtype=vec6),
):
    """
    Solves the local flow rule and immediately applies the resulting stress
    delta to particle velocities, using a coloring approach
    to avoid avoid race conditions.
    """

    i = wp.tid()
    color_beg = color_offsets[color] + i
    color_end = color_offsets[color + 1]

    for color_offset in range(color_beg, color_end, launch_dim):
        beg, end = color_blocks[0, color_offset], color_blocks[1, color_offset]
        for tau_i in range(beg, end):
            local_strain = compute_local_strain(
                tau_i,
                compliance_mat_offsets,
                compliance_mat_columns,
                compliance_mat_values,
                strain_mat_offsets,
                strain_mat_columns,
                strain_mat_values,
                local_strain_rhs,
                velocities,
                local_stress,
            )

            delta_stress = solve_local_stress(
                tau_i,
                local_strain,
                yield_params,
                strain_node_volume,
                delassus_diagonal,
                delassus_rotation,
                local_stress,
            )

            local_stress[tau_i] += delta_stress
            delta_correction[tau_i] = delta_stress  # for residual evaluation

            apply_stress_delta(
                tau_i,
                delta_stress,
                strain_mat_offsets,
                strain_mat_columns,
                strain_mat_values,
                inv_mass_matrix,
                velocities,
            )


@wp.kernel
def jacobi_preconditioner(
    delassus_diagonal: wp.array(dtype=vec6),
    delassus_rotation: wp.array(dtype=mat55),
    x: wp.array(dtype=vec6),
    y: wp.array(dtype=vec6),
    z: wp.array(dtype=vec6),
    alpha: float,
    beta: float,
):
    tau_i = wp.tid()
    rot = delassus_rotation[tau_i]
    diag = delassus_diagonal[tau_i]

    Wx = _local_to_world(wp.cw_div(_world_to_local(x[tau_i], rot), diag), rot)
    z[tau_i] = alpha * Wx + beta * y[tau_i]


@wp.kernel
def compute_collider_inv_mass(
    J_mat_offsets: wp.array(dtype=int),
    J_mat_columns: wp.array(dtype=int),
    J_mat_values: wp.array(dtype=wp.mat33),
    IJtm_mat_offsets: wp.array(dtype=int),
    IJtm_mat_columns: wp.array(dtype=int),
    IJtm_mat_values: wp.array(dtype=wp.mat33),
    collider_inv_mass: wp.array(dtype=float),
):
    i = wp.tid()

    block_beg = J_mat_offsets[i]
    block_end = J_mat_offsets[i + 1]

    w_mat = wp.mat33(0.0)

    for b in range(block_beg, block_end):
        col = J_mat_columns[b]
        transposed_block = sp.bsr_block_index(col, i, IJtm_mat_offsets, IJtm_mat_columns)
        if transposed_block == -1:
            continue

        # Mass-splitting: divide by number of nodes overlapping with this body
        multiplicity = float(IJtm_mat_offsets[col + 1] - IJtm_mat_offsets[col])

        w_mat += (J_mat_values[b] @ IJtm_mat_values[transposed_block]) * multiplicity

    _eigvecs, eigvals = wp.eig3(w_mat)
    collider_inv_mass[i] = wp.max(0.0, wp.max(eigvals))


@wp.func
def project_on_friction_cone(
    mu: float,
    nor: wp.vec3,
    r: wp.vec3,
):
    """Projects a stress vector ``r`` onto the Coulomb friction cone (non-orthogonally)."""

    r_n = wp.dot(r, nor)
    r_t = r - r_n * nor

    r_n = wp.max(0.0, r_n)
    mu_rn = mu * r_n

    r_t_n2 = wp.length_sq(r_t)
    if r_t_n2 > mu_rn * mu_rn:
        r_t *= mu_rn / wp.sqrt(r_t_n2)

    return r_n * nor + r_t


@wp.func
def solve_coulomb_isotropic(
    mu: float,
    nor: wp.vec3,
    u: wp.vec3,
):
    """Solves for the relative velocity in the Coulomb friction model,
    assuming an isotropic velocity-impulse relationship, u = r + b
    """

    u_n = wp.dot(u, nor)
    if u_n < 0.0:
        u -= u_n * nor
        tau = wp.length_sq(u)
        alpha = mu * u_n
        if tau <= alpha * alpha:
            u = wp.vec3(0.0)
        else:
            u *= 1.0 + mu * u_n / wp.sqrt(tau)

    return u


@wp.func
def filter_collider_impulse_warmstart(
    friction: float,
    nor: wp.vec3,
    adhesion: float,
    impulse: wp.vec3,
):
    """Filters the collider impulse to be within the friction cone"""

    if friction < 0.0:
        return wp.vec3(0.0)

    return project_on_friction_cone(friction, nor, impulse + adhesion * nor) - adhesion * nor


@wp.kernel
def apply_nodal_impulse_warmstart(
    collider_impulse: wp.array(dtype=wp.vec3),
    collider_friction: wp.array(dtype=float),
    collider_normals: wp.array(dtype=wp.vec3),
    collider_adhesion: wp.array(dtype=float),
    inv_mass: wp.array(dtype=float),
    velocities: wp.array(dtype=wp.vec3),
    delta_impulse: wp.array(dtype=wp.vec3),
):
    """
    Applies pre-computed impulses to particles and colliders.
    """
    i = wp.tid()

    impulse = filter_collider_impulse_warmstart(
        collider_friction[i], collider_normals[i], collider_adhesion[i], collider_impulse[i]
    )

    collider_impulse[i] = impulse
    delta_impulse[i] = impulse
    velocities[i] += inv_mass[i] * impulse


@wp.kernel
def solve_nodal_friction(
    inv_mass: wp.array(dtype=float),
    collider_friction: wp.array(dtype=float),
    collider_adhesion: wp.array(dtype=float),
    collider_normals: wp.array(dtype=wp.vec3),
    collider_inv_mass: wp.array(dtype=float),
    velocities: wp.array(dtype=wp.vec3),
    collider_velocities: wp.array(dtype=wp.vec3),
    impulse: wp.array(dtype=wp.vec3),
    delta_impulse: wp.array(dtype=wp.vec3),
):
    """
    Solves for frictional impulses at nodes interacting with colliders.

    For each node (i) potentially in contact:
    1. Skips if friction coefficient is negative (no friction).
    2. Calculates the relative velocity `u0` between the particle and collider,
       accounting for the existing impulse and adhesion.
    3. Computes the effective inverse mass `w` for the interaction.
    4. Calls `solve_coulomb_isotropic` to determine the change in relative
       velocity `u` due to friction.
    5. Calculates the change in impulse `delta_impulse` required to achieve this
       change in relative velocity.
    6. Updates the total impulse, particle velocity, and collider velocity.
    """
    i = wp.tid()

    friction_coeff = collider_friction[i]
    if friction_coeff < 0.0:
        return

    n = collider_normals[i]
    u0 = velocities[i] - collider_velocities[i]

    w = inv_mass[i] + collider_inv_mass[i]

    u = solve_coulomb_isotropic(friction_coeff, n, u0 - (impulse[i] + collider_adhesion[i] * n) * w)

    delta_u = u - u0
    delta_lambda = delta_u / w

    delta_impulse[i] = delta_lambda
    impulse[i] += delta_lambda
    velocities[i] += inv_mass[i] * delta_lambda


@wp.kernel
def apply_subgrid_impulse(
    tr_collider_mat_offsets: wp.array(dtype=int),
    tr_collider_mat_columns: wp.array(dtype=int),
    tr_collider_mat_values: wp.array(dtype=float),
    inv_mass: wp.array(dtype=float),
    impulses: wp.array(dtype=wp.vec3),
    velocities: wp.array(dtype=wp.vec3),
):
    """
    Applies pre-computed impulses to particles and colliders.
    """

    u_i = wp.tid()
    block_beg = tr_collider_mat_offsets[u_i]
    block_end = tr_collider_mat_offsets[u_i + 1]

    delta_f = wp.vec3(0.0)
    for b in range(block_beg, block_end):
        delta_f += tr_collider_mat_values[b] * impulses[tr_collider_mat_columns[b]]

    velocities[u_i] += inv_mass[u_i] * delta_f


@wp.kernel
def apply_subgrid_impulse_warmstart(
    collider_friction: wp.array(dtype=float),
    collider_normals: wp.array(dtype=wp.vec3),
    collider_adhesion: wp.array(dtype=float),
    collider_impulse: wp.array(dtype=wp.vec3),
    delta_impulse: wp.array(dtype=wp.vec3),
):
    i = wp.tid()

    impulse = filter_collider_impulse_warmstart(
        collider_friction[i], collider_normals[i], collider_adhesion[i], collider_impulse[i]
    )

    collider_impulse[i] = impulse
    delta_impulse[i] = impulse


@wp.kernel
def compute_collider_delassus_diagonal(
    collider_mat_offsets: wp.array(dtype=int),
    collider_mat_columns: wp.array(dtype=int),
    collider_mat_values: wp.array(dtype=float),
    collider_inv_mass: wp.array(dtype=float),
    transposed_collider_mat_offsets: wp.array(dtype=int),
    inv_volume: wp.array(dtype=float),
    delassus_diagonal: wp.array(dtype=float),
):
    i = wp.tid()

    block_beg = collider_mat_offsets[i]
    block_end = collider_mat_offsets[i + 1]

    inv_mass = collider_inv_mass[i]
    w = inv_mass

    for b in range(block_beg, block_end):
        u_i = collider_mat_columns[b]
        weight = collider_mat_values[b]

        multiplicity = transposed_collider_mat_offsets[u_i + 1] - transposed_collider_mat_offsets[u_i]

        w += weight * weight * inv_volume[u_i] * float(multiplicity)

    delassus_diagonal[i] = w


@wp.kernel
def solve_subgrid_friction(
    velocity: wp.array(dtype=wp.vec3),
    collider_mat_offsets: wp.array(dtype=int),
    collider_mat_columns: wp.array(dtype=int),
    collider_mat_values: wp.array(dtype=float),
    collider_friction: wp.array(dtype=float),
    collider_adhesion: wp.array(dtype=float),
    collider_normals: wp.array(dtype=wp.vec3),
    collider_delassus_diagonal: wp.array(dtype=float),
    collider_velocities: wp.array(dtype=wp.vec3),
    impulse: wp.array(dtype=wp.vec3),
    delta_impulse: wp.array(dtype=wp.vec3),
):
    i = wp.tid()

    w = collider_delassus_diagonal[i]
    friction_coeff = collider_friction[i]
    if w <= 0.0 or friction_coeff < 0.0:
        return

    beg = collider_mat_offsets[i]
    end = collider_mat_offsets[i + 1]

    u0 = -collider_velocities[i]
    for b in range(beg, end):
        u_i = collider_mat_columns[b]
        u0 += collider_mat_values[b] * velocity[u_i]

    n = collider_normals[i]

    u = solve_coulomb_isotropic(friction_coeff, n, u0 - (impulse[i] + collider_adhesion[i] * n) * w)

    delta_u = u - u0
    delta_lambda = delta_u / w

    impulse[i] += delta_lambda
    delta_impulse[i] = delta_lambda


@wp.kernel
def evaluate_strain_residual(
    delta_stress: wp.array(dtype=vec6),
    delassus_diagonal: wp.array(dtype=vec6),
    delassus_rotation: wp.array(dtype=mat55),
    residual: wp.array(dtype=float),
):
    tau_i = wp.tid()
    local_strain_delta = wp.cw_mul(
        _world_to_local(delta_stress[tau_i], delassus_rotation[tau_i]), delassus_diagonal[tau_i]
    )
    r = wp.length_sq(local_strain_delta)
    if not (r < 1.0e16):
        r = 0.0

    residual[tau_i] = r


_TILED_SUM_BLOCK_DIM = 512


@wp.kernel(module="unique")
def _tiled_sum_kernel(
    data: wp.array2d(dtype=float),
    partial_sums: wp.array2d(dtype=float),
):
    block_id = wp.tid()

    tile = wp.tile_load(data[0], shape=_TILED_SUM_BLOCK_DIM, offset=block_id * _TILED_SUM_BLOCK_DIM)
    wp.tile_store(partial_sums[0], wp.tile_sum(tile), offset=block_id)
    tile = wp.tile_load(data[1], shape=_TILED_SUM_BLOCK_DIM, offset=block_id * _TILED_SUM_BLOCK_DIM)
    wp.tile_store(partial_sums[1], wp.tile_max(tile), offset=block_id)


class ArraySquaredNorm:
    """Utility to compute squared L2 norm of a large array via tiled reductions."""

    def __init__(self, max_length: int, device=None, temporary_store=None):
        self.tile_size = _TILED_SUM_BLOCK_DIM
        self.device = device

        num_blocks = (max_length + self.tile_size - 1) // self.tile_size
        self.partial_sums_a = fem.borrow_temporary(
            temporary_store, shape=(2, num_blocks), dtype=float, device=self.device
        )
        self.partial_sums_b = fem.borrow_temporary(
            temporary_store, shape=(2, num_blocks), dtype=float, device=self.device
        )
        self.partial_sums_a.zero_()
        self.partial_sums_b.zero_()

        self.sum_launch: wp.Launch = wp.launch(
            _tiled_sum_kernel,
            dim=(num_blocks, self.tile_size),
            inputs=(self.partial_sums_a,),
            outputs=(self.partial_sums_b,),
            block_dim=self.tile_size,
            record_cmd=True,
        )

    # Result contains a single value, the sum of the array (will get updated by this function)
    def compute_squared_norm(self, data: wp.array(dtype=Any)):
        # cast vector types to float
        if data.ndim != 2:
            data = wp.array(
                ptr=data.ptr,
                shape=(2, data.shape[0]),
                dtype=data.dtype,
                strides=(0, data.strides[0]),
                device=data.device,
            )

        array_length = data.shape[1]

        flip_flop = False
        while True:
            num_blocks = (array_length + self.tile_size - 1) // self.tile_size
            partial_sums = (self.partial_sums_a if flip_flop else self.partial_sums_b)[:, :num_blocks]

            self.sum_launch.set_param_at_index(0, data[:, :array_length])
            self.sum_launch.set_param_at_index(1, partial_sums)
            self.sum_launch.set_dim((num_blocks, self.tile_size))
            self.sum_launch.launch()

            array_length = num_blocks
            data = partial_sums

            flip_flop = not flip_flop

            if num_blocks == 1:
                break

        return data[:, :1]

    def release(self):
        """Return borrowed temporaries to their pool."""
        for attr in ("partial_sums_a", "partial_sums_b"):
            temporary = getattr(self, attr, None)
            if temporary is not None:
                temporary.release()
                setattr(self, attr, None)

    def __del__(self):
        self.release()


@wp.kernel
def update_condition(
    residual_threshold: float,
    l2_scale: float,
    min_iterations: int,
    max_iterations: int,
    residual: wp.array2d(dtype=float),
    iteration: wp.array(dtype=int),
    condition: wp.array(dtype=int),
):
    cur_it = iteration[0] + 1
    stop = (
        residual[0, 0] < residual_threshold * l2_scale
        and residual[1, 0] < residual_threshold
        and cur_it > min_iterations
    ) or cur_it > max_iterations

    iteration[0] = cur_it
    condition[0] = wp.where(stop, 0, 1)


def apply_rigidity_operator(rigidity_operator, delta_collider_impulse, collider_velocity, delta_body_qd):
    """Apply collider rigidity feedback to the current collider velocities.

    Computes and applies a velocity correction induced by the rigid coupling
    operator according to the relation::

        delta_body_qd = -IJtm @ delta_collider_impulse
        collider_velocity += J @ delta_body_qd

    where ``(J, IJtm) = rigidity_operator`` are the block-sparse matrices
    returned by ``build_rigidity_operator``.

    Args:
        rigidity_operator: Pair ``(J, IJtm)`` of block-sparse matrices returned
            by ``build_rigidity_operator``.
        delta_collider_impulse: Change in collider impulse to be applied.
        collider_velocity: Current collider velocity vector to be corrected in place.
        delta_body_qd: Change in body velocity to be applied.
    """

    J, IJtm = rigidity_operator
    sp.bsr_mv(IJtm, x=delta_collider_impulse, y=delta_body_qd, alpha=-1.0, beta=0.0)
    sp.bsr_mv(J, x=delta_body_qd, y=collider_velocity, alpha=1.0, beta=1.0)


class _ScopedDisableGC:
    """Context manager to disable automatic garbage collection during graph capture.
    Avoids capturing deallocations of arrays exterior to the capture scope.
    """

    def __enter__(self):
        self.was_enabled = gc.isenabled()
        gc.disable()

    def __exit__(self, exc_type, exc_value, traceback):
        if self.was_enabled:
            gc.enable()


@dataclass
class MomentumData:
    """Per-node momentum quantities used by the rheology solver.

    Attributes:
        inv_volume: Inverse volume (or inverse mass scaling) per velocity
            node, shape ``[node_count]``.
        velocity: Grid velocity DOFs to be updated in place [m/s],
            shape ``[node_count, 3]``.
    """

    inv_volume: wp.array
    velocity: wp.array(dtype=wp.vec3)


@dataclass
class RheologyData:
    """Strain, compliance, yield, and coloring data for the rheology solve.

    Attributes:
        strain_mat: Strain-to-velocity block-sparse matrix (B).
        transposed_strain_mat: BSR container for B^T, used by the Jacobi
            solver path.
        compliance_mat: Compliance (inverse stiffness) block-sparse matrix.
        strain_node_volume: Volume associated with each strain node [m^3],
            shape ``[strain_count]``.
        yield_params: Yield-surface parameters per strain node,
            shape ``[strain_count]``.
        unilateral_strain_offset: Per-node offset enforcing unilateral
            incompressibility (void/critical fraction),
            shape ``[strain_count]``.
        color_offsets: Coloring offsets for Gauss-Seidel iteration,
            shape ``[num_colors + 1]``.
        color_blocks: Per-color strain-node indices for Gauss-Seidel,
            shape ``[num_colors, max_block_size]``.
        elastic_strain_delta: Output elastic strain increment per strain
            node, shape ``[strain_count, 6]``.
        plastic_strain_delta: Output plastic strain increment per strain
            node, shape ``[strain_count, 6]``.
        stress: In/out stress per strain node (rotated internally),
            shape ``[strain_count, 6]``.
    """

    strain_mat: sp.BsrMatrix
    transposed_strain_mat: sp.BsrMatrix
    compliance_mat: sp.BsrMatrix
    strain_node_volume: wp.array(dtype=float)
    yield_params: wp.array(dtype=YieldParamVec)
    unilateral_strain_offset: wp.array(dtype=float)

    color_offsets: wp.array(dtype=int)
    color_blocks: wp.array2d(dtype=int)

    elastic_strain_delta: wp.array(dtype=vec6)
    plastic_strain_delta: wp.array(dtype=vec6)
    stress: wp.array(dtype=vec6)


@dataclass
class CollisionData:
    """Collider contact data consumed by the rheology solver.

    Attributes:
        collider_mat: Block-sparse matrix mapping velocity nodes to
            collider DOFs.
        transposed_collider_mat: Transpose of ``collider_mat``.
        collider_friction: Per-node friction coefficients; negative values
            disable contact at that node, shape ``[node_count]``.
        collider_adhesion: Per-node adhesion coefficients [N s / V0],
            shape ``[node_count]``.
        collider_normals: Per-node contact normals,
            shape ``[node_count, 3]``.
        collider_velocities: Per-node collider rigid-body velocities [m/s],
            shape ``[node_count, 3]``.
        rigidity_operator: Optional pair of BSR matrices coupling velocity
            nodes to collider DOFs. ``None`` when unused.
        collider_impulse: In/out stored collider impulses for warm-starting
            [N s / V0], shape ``[node_count, 3]``.
    """

    collider_mat: sp.BsrMatrix
    transposed_collider_mat: sp.BsrMatrix
    collider_friction: wp.array(dtype=float)
    collider_adhesion: wp.array(dtype=float)
    collider_normals: wp.array(dtype=wp.vec3)
    collider_velocities: wp.array(dtype=wp.vec3)
    rigidity_operator: tuple[sp.BsrMatrix, sp.BsrMatrix] | None
    collider_impulse: wp.array(dtype=wp.vec3)


class _DelassusOperator:
    def __init__(
        self,
        rheology: RheologyData,
        momentum: MomentumData,
        temporary_store: fem.TemporaryStore | None = None,
    ):
        self.rheology = rheology
        self.momentum = momentum

        self.delassus_rotation = fem.borrow_temporary(temporary_store, shape=self.size, dtype=mat55)
        self.delassus_diagonal = fem.borrow_temporary(temporary_store, shape=self.size, dtype=vec6)

        self._computed = False
        self._split_mass = False

        self._has_strain_mat_transpose = False

        self.preprocess_stress_and_strain()

    def compute_diagonal_factorization(self, split_mass: bool):
        if self._computed and self._split_mass == split_mass:
            return

        if split_mass:
            self.require_strain_mat_transpose()

        strain_mat_values = self.rheology.strain_mat.values.view(dtype=mat13)
        wp.launch(
            kernel=compute_delassus_diagonal,
            dim=self.size,
            inputs=[
                split_mass,
                self.rheology.strain_mat.offsets,
                self.rheology.strain_mat.columns,
                strain_mat_values,
                self.momentum.inv_volume,
                self.rheology.compliance_mat.offsets,
                self.rheology.compliance_mat.columns,
                self.rheology.compliance_mat.values,
                self.rheology.transposed_strain_mat.offsets,
            ],
            outputs=[
                self.delassus_rotation,
                self.delassus_diagonal,
            ],
        )

        self._computed = True
        self._split_mass = split_mass

    def require_strain_mat_transpose(self):
        if not self._has_strain_mat_transpose:
            sp.bsr_set_transpose(dest=self.rheology.transposed_strain_mat, src=self.rheology.strain_mat)
            self._has_strain_mat_transpose = True

    def preprocess_stress_and_strain(self):
        # Project initial stress on yield surface
        wp.launch(
            kernel=preprocess_stress_and_strain,
            dim=self.size,
            inputs=[
                self.rheology.unilateral_strain_offset,
                self.rheology.elastic_strain_delta,
                self.rheology.stress,
                self.rheology.yield_params,
            ],
        )

    @property
    def size(self):
        return self.rheology.stress.shape[0]

    def release(self):
        self.delassus_rotation.release()
        self.delassus_diagonal.release()

    def apply_stress_delta(
        self, stress_delta: wp.array(dtype=vec6), velocity: wp.array(dtype=wp.vec3), record_cmd: bool = False
    ):
        return wp.launch(
            kernel=apply_stress_delta_jacobi,
            dim=self.momentum.velocity.shape[0],
            inputs=[
                self.rheology.transposed_strain_mat.offsets,
                self.rheology.transposed_strain_mat.columns,
                self.rheology.transposed_strain_mat.values.view(dtype=mat13),
                self.momentum.inv_volume,
                stress_delta,
            ],
            outputs=[velocity],
            record_cmd=record_cmd,
        )

    def apply_velocity_delta(
        self,
        velocity_delta: wp.array(dtype=wp.vec3),
        strain_prev: wp.array(dtype=vec6),
        strain: wp.array(dtype=vec6),
        alpha: float = 1.0,
        beta: float = 1.0,
        record_cmd: bool = False,
    ):
        return wp.launch(
            kernel=apply_velocity_delta,
            dim=self.size,
            inputs=[
                alpha,
                beta,
                self.rheology.strain_mat.offsets,
                self.rheology.strain_mat.columns,
                self.rheology.strain_mat.values.view(dtype=mat13),
                velocity_delta,
                strain_prev,
            ],
            outputs=[
                strain,
            ],
            record_cmd=record_cmd,
        )

    def postprocess_stress_and_strain(self):
        # Convert stress back to world space,
        # and compute final elastic strain
        wp.launch(
            kernel=postprocess_stress_and_strain,
            dim=self.size,
            inputs=[
                self.rheology.compliance_mat.offsets,
                self.rheology.compliance_mat.columns,
                self.rheology.compliance_mat.values,
                self.rheology.strain_mat.offsets,
                self.rheology.strain_mat.columns,
                self.rheology.strain_mat.values.view(dtype=mat13),
                self.delassus_diagonal,
                self.delassus_rotation,
                self.rheology.unilateral_strain_offset,
                self.rheology.yield_params,
                self.rheology.strain_node_volume,
                self.rheology.elastic_strain_delta,
                self.rheology.stress,
                self.momentum.velocity,
            ],
            outputs=[
                self.rheology.elastic_strain_delta,
                self.rheology.plastic_strain_delta,
            ],
        )


class _RheologySolver:
    def __init__(
        self,
        delassus_operator: _DelassusOperator,
        split_mass: bool,
        temporary_store: fem.TemporaryStore | None = None,
    ):
        self.delassus_operator = delassus_operator
        self.momentum = delassus_operator.momentum
        self.rheology = delassus_operator.rheology
        self.device = self.momentum.velocity.device

        self.delta_stress = fem.borrow_temporary_like(self.rheology.stress, temporary_store)
        self.strain_residual = fem.borrow_temporary(
            temporary_store, shape=(self.size,), dtype=float, device=self.device
        )
        self.strain_residual.zero_()

        self.delassus_operator.compute_diagonal_factorization(split_mass)

        self._evaluate_strain_residual_launch = wp.launch(
            kernel=evaluate_strain_residual,
            dim=self.size,
            inputs=[
                self.delta_stress,
                self.delassus_operator.delassus_diagonal,
                self.delassus_operator.delassus_rotation,
            ],
            outputs=[
                self.strain_residual,
            ],
            record_cmd=True,
        )

        # Utility to compute the squared norm of the residual
        self._residual_squared_norm_computer = ArraySquaredNorm(
            max_length=self.size,
            device=self.device,
            temporary_store=temporary_store,
        )

    @property
    def size(self):
        return self.rheology.stress.shape[0]

    def eval_residual(self):
        self._evaluate_strain_residual_launch.launch()
        return self._residual_squared_norm_computer.compute_squared_norm(self.strain_residual)

    def release(self):
        self.delta_stress.release()
        self.strain_residual.release()
        self._residual_squared_norm_computer.release()


class _GaussSeidelSolver(_RheologySolver):
    def __init__(
        self,
        delassus_operator: _DelassusOperator,
        temporary_store: fem.TemporaryStore | None = None,
    ) -> None:
        super().__init__(delassus_operator, split_mass=False, temporary_store=temporary_store)

        self.color_count = self.rheology.color_offsets.shape[0] - 1

        if self.device.is_cuda:
            color_block_count = self.device.sm_count * 2
        else:
            color_block_count = 1
        color_block_dim = 64
        color_launch_dim = color_block_count * color_block_dim

        self.apply_stress_launch = wp.launch(
            kernel=apply_stress_gs,
            dim=color_launch_dim,
            inputs=[
                0,  # color
                color_launch_dim,
                self.rheology.color_offsets,
                self.rheology.color_blocks,
                self.rheology.strain_mat.offsets,
                self.rheology.strain_mat.columns,
                self.rheology.strain_mat.values.view(dtype=mat13),
                self.momentum.inv_volume,
                self.rheology.stress,
            ],
            outputs=[
                self.momentum.velocity,
            ],
            block_dim=color_block_dim,
            max_blocks=color_block_count,
            record_cmd=True,
        )

        # Solve kernel
        self.solve_local_launch = wp.launch(
            kernel=solve_local_stress_gs,
            dim=color_launch_dim,
            inputs=[
                0,  # color
                color_launch_dim,
                self.rheology.color_offsets,
                self.rheology.color_blocks,
                self.rheology.yield_params,
                self.rheology.strain_node_volume,
                self.rheology.compliance_mat.offsets,
                self.rheology.compliance_mat.columns,
                self.rheology.compliance_mat.values,
                self.rheology.strain_mat.offsets,
                self.rheology.strain_mat.columns,
                self.rheology.strain_mat.values.view(dtype=mat13),
                self.delassus_operator.delassus_diagonal,
                self.delassus_operator.delassus_rotation,
                self.momentum.inv_volume,
                self.rheology.elastic_strain_delta,
            ],
            outputs=[
                self.momentum.velocity,
                self.rheology.stress,
                self.delta_stress,
            ],
            block_dim=color_block_dim,
            max_blocks=color_block_count,
            record_cmd=True,
        )

    @property
    def name(self):
        return "Gauss-Seidel"

    @property
    def solve_granularity(self):
        return 25

    def apply_initial_guess(self):
        # Apply initial guess
        for color in range(self.color_count):
            self.apply_stress_launch.set_param_at_index(0, color)
            self.apply_stress_launch.launch()

    def solve(self):
        # solve stress
        for color in range(self.color_count):
            self.solve_local_launch.set_param_at_index(0, color)
            self.solve_local_launch.launch()


class _JacobiSolver(_RheologySolver):
    def __init__(
        self,
        delassus_operator: _DelassusOperator,
        temporary_store: fem.TemporaryStore | None = None,
    ) -> None:
        super().__init__(delassus_operator, split_mass=True, temporary_store=temporary_store)

        self.apply_stress_launch = self.delassus_operator.apply_stress_delta(
            self.delta_stress,
            self.momentum.velocity,
            record_cmd=True,
        )

        # Solve kernel
        self.solve_local_launch = wp.launch(
            kernel=solve_local_stress_jacobi,
            dim=self.size,
            inputs=[
                self.rheology.yield_params,
                self.rheology.strain_node_volume,
                self.rheology.compliance_mat.offsets,
                self.rheology.compliance_mat.columns,
                self.rheology.compliance_mat.values,
                self.rheology.strain_mat.offsets,
                self.rheology.strain_mat.columns,
                self.rheology.strain_mat.values.view(dtype=mat13),
                self.delassus_operator.delassus_diagonal,
                self.delassus_operator.delassus_rotation,
                self.rheology.elastic_strain_delta,
                self.momentum.velocity,
                self.rheology.stress,
            ],
            outputs=[
                self.delta_stress,
            ],
            record_cmd=True,
        )

    @property
    def name(self):
        return "Jacobi"

    @property
    def solve_granularity(self):
        return 50

    def apply_initial_guess(self):
        # Apply initial guess
        self.delta_stress.assign(self.rheology.stress)
        self.apply_stress_launch.launch()

    def solve(self):
        self.solve_local_launch.launch()
        # Add jacobi delta
        self.apply_stress_launch.launch()
        fem.utils.array_axpy(x=self.delta_stress, y=self.rheology.stress, alpha=1.0, beta=1.0)


class _CGSolver:
    def __init__(
        self,
        delassus_operator: _DelassusOperator,
        temporary_store: fem.TemporaryStore | None = None,
    ) -> None:
        self.momentum = delassus_operator.momentum
        self.rheology = delassus_operator.rheology
        self.delassus_operator = delassus_operator

        self.delassus_operator.require_strain_mat_transpose()
        self.delassus_operator.compute_diagonal_factorization(split_mass=False)

        self.delta_velocity = fem.borrow_temporary_like(self.momentum.velocity, temporary_store)

        shape = self.rheology.compliance_mat.shape
        dtype = self.rheology.compliance_mat.dtype
        device = self.rheology.compliance_mat.device

        self.linear_operator = LinearOperator(shape=shape, dtype=dtype, device=device, matvec=self._delassus_matvec)
        self.preconditioner = LinearOperator(
            shape=shape, dtype=dtype, device=device, matvec=self._preconditioner_matvec
        )

    def _delassus_matvec(
        self, x: wp.array(dtype=vec6), y: wp.array(dtype=vec6), z: wp.array(dtype=vec6), alpha: float, beta: float
    ):
        # dv = B^T x
        self.delta_velocity.zero_()
        self.delassus_operator.apply_stress_delta(x, self.delta_velocity)
        # z = alpha B dv + beta * y
        self.delassus_operator.apply_velocity_delta(self.delta_velocity, y, z, alpha, beta)

        # z += C x
        sp.bsr_mv(self.rheology.compliance_mat, x, z, alpha=alpha, beta=1.0)

    def _preconditioner_matvec(self, x, y, z, alpha, beta):
        wp.launch(
            kernel=jacobi_preconditioner,
            dim=self.delassus_operator.size,
            inputs=[
                self.delassus_operator.delassus_diagonal,
                self.delassus_operator.delassus_rotation,
                x,
                y,
                z,
                alpha,
                beta,
            ],
        )

    def solve(self, tol: float, tolerance_scale: float, max_iterations: int, use_graph: bool, verbose: bool):
        self.delassus_operator.apply_velocity_delta(
            self.momentum.velocity,
            self.rheology.elastic_strain_delta,
            self.rheology.plastic_strain_delta,
            alpha=-1.0,
            beta=-1.0,
        )

        with _ScopedDisableGC():
            end_iter, residual, atol = cg(
                A=self.linear_operator,
                M=self.preconditioner,
                b=self.rheology.plastic_strain_delta,
                x=self.rheology.stress,
                atol=tol * tolerance_scale,
                tol=tol,
                maxiter=max_iterations,
                check_every=0 if use_graph else 10,
                use_cuda_graph=use_graph,
            )

        if use_graph:
            end_iter = end_iter.numpy()[0]
            residual = residual.numpy()[0]
            atol = atol.numpy()[0]

        if verbose:
            res = math.sqrt(residual) / tolerance_scale
            print(f"{self.name} terminated after {end_iter} iterations with residual {res}")

    @property
    def name(self):
        return "Conjugate Gradient"

    def release(self):
        self.delta_velocity.release()


class _ContactSolver:
    def __init__(
        self,
        momentum: MomentumData,
        collision: CollisionData,
        temporary_store: fem.TemporaryStore | None = None,
    ) -> None:
        self.momentum = momentum
        self.collision = collision

        self.delta_impulse = fem.borrow_temporary_like(self.collision.collider_impulse, temporary_store)
        self.collider_inv_mass = fem.borrow_temporary_like(self.collision.collider_friction, temporary_store)

        # Setup rigidity correction
        if self.collision.rigidity_operator is not None:
            J, IJtm = self.collision.rigidity_operator
            self.delta_body_qd = fem.borrow_temporary(temporary_store, shape=J.shape[1], dtype=float)

            wp.launch(
                compute_collider_inv_mass,
                dim=self.collision.collider_impulse.shape[0],
                inputs=[
                    J.offsets,
                    J.columns,
                    J.values,
                    IJtm.offsets,
                    IJtm.columns,
                    IJtm.values,
                ],
                outputs=[
                    self.collider_inv_mass,
                ],
            )

        else:
            self.collider_inv_mass.zero_()

    def release(self):
        self.delta_impulse.release()
        self.collider_inv_mass.release()
        if self.collision.rigidity_operator is not None:
            self.delta_body_qd.release()

    def apply_rigidity_operator(self):
        if self.collision.rigidity_operator is not None:
            apply_rigidity_operator(
                self.collision.rigidity_operator,
                self.delta_impulse,
                self.collision.collider_velocities,
                self.delta_body_qd,
            )


class _NodalContactSolver(_ContactSolver):
    def __init__(
        self,
        momentum: MomentumData,
        collision: CollisionData,
        temporary_store: fem.TemporaryStore | None = None,
    ) -> None:
        super().__init__(momentum, collision, temporary_store)

        # define solve operation
        self.solve_collider_launch = wp.launch(
            kernel=solve_nodal_friction,
            dim=self.collision.collider_impulse.shape[0],
            inputs=[
                self.momentum.inv_volume,
                self.collision.collider_friction,
                self.collision.collider_adhesion,
                self.collision.collider_normals,
                self.collider_inv_mass,
                self.momentum.velocity,
                self.collision.collider_velocities,
                self.collision.collider_impulse,
                self.delta_impulse,
            ],
            record_cmd=True,
        )

    def apply_initial_guess(self):
        # Apply initial impulse guess
        wp.launch(
            kernel=apply_nodal_impulse_warmstart,
            dim=self.collision.collider_impulse.shape[0],
            inputs=[
                self.collision.collider_impulse,
                self.collision.collider_friction,
                self.collision.collider_normals,
                self.collision.collider_adhesion,
                self.momentum.inv_volume,
                self.momentum.velocity,
                self.delta_impulse,
            ],
        )
        self.apply_rigidity_operator()

    def solve(self):
        self.solve_collider_launch.launch()
        self.apply_rigidity_operator()


class _SubgridContactSolver(_ContactSolver):
    def __init__(
        self,
        momentum: MomentumData,
        collision: CollisionData,
        temporary_store: fem.TemporaryStore | None = None,
    ) -> None:
        super().__init__(momentum, collision, temporary_store)

        self.collider_delassus_diagonal = fem.borrow_temporary_like(self.collider_inv_mass, temporary_store)

        sp.bsr_set_transpose(dest=self.collision.transposed_collider_mat, src=self.collision.collider_mat)

        wp.launch(
            compute_collider_delassus_diagonal,
            dim=self.collision.collider_impulse.shape[0],
            inputs=[
                self.collision.collider_mat.offsets,
                self.collision.collider_mat.columns,
                self.collision.collider_mat.values,
                self.collider_inv_mass,
                self.collision.transposed_collider_mat.offsets,
                self.momentum.inv_volume,
            ],
            outputs=[
                self.collider_delassus_diagonal,
            ],
        )

        # define solve operation
        self.apply_collider_impulse_launch = wp.launch(
            apply_subgrid_impulse,
            dim=self.momentum.velocity.shape[0],
            inputs=[
                self.collision.transposed_collider_mat.offsets,
                self.collision.transposed_collider_mat.columns,
                self.collision.transposed_collider_mat.values,
                self.momentum.inv_volume,
                self.delta_impulse,
                self.momentum.velocity,
            ],
            record_cmd=True,
        )

        self.solve_collider_launch = wp.launch(
            kernel=solve_subgrid_friction,
            dim=self.collision.collider_impulse.shape[0],
            inputs=[
                self.momentum.velocity,
                self.collision.collider_mat.offsets,
                self.collision.collider_mat.columns,
                self.collision.collider_mat.values,
                self.collision.collider_friction,
                self.collision.collider_adhesion,
                self.collision.collider_normals,
                self.collider_delassus_diagonal,
                self.collision.collider_velocities,
                self.collision.collider_impulse,
                self.delta_impulse,
            ],
            record_cmd=True,
        )

    def apply_initial_guess(self):
        wp.launch(
            apply_subgrid_impulse_warmstart,
            dim=self.delta_impulse.shape[0],
            inputs=[
                self.collision.collider_friction,
                self.collision.collider_normals,
                self.collision.collider_adhesion,
                self.collision.collider_impulse,
                self.delta_impulse,
            ],
        )
        self.apply_collider_impulse_launch.launch()
        self.apply_rigidity_operator()

    def solve(self):
        self.solve_collider_launch.launch()
        self.apply_collider_impulse_launch.launch()
        self.apply_rigidity_operator()

    def release(self):
        self.collider_delassus_diagonal.release()
        super().release()


def _run_solver_loop(
    rheology_solver: _RheologySolver,
    contact_solver: _ContactSolver,
    max_iterations: int,
    tolerance: float,
    l2_tolerance_scale: float,
    use_graph: bool,
    verbose: bool,
    temporary_store: fem.TemporaryStore,
):
    solve_graph = None
    if use_graph:
        min_iterations = 5
        iteration_and_condition = fem.borrow_temporary(temporary_store, shape=(2,), dtype=int)
        iteration_and_condition.fill_(1)

        iteration = iteration_and_condition[:1]
        condition = iteration_and_condition[1:]

        def do_iteration_with_condition():
            contact_solver.solve()
            rheology_solver.solve()
            residual = rheology_solver.eval_residual()
            wp.launch(
                update_condition,
                dim=1,
                inputs=[
                    tolerance * tolerance,
                    l2_tolerance_scale * l2_tolerance_scale,
                    min_iterations,
                    max_iterations,
                    residual,
                    iteration,
                    condition,
                ],
            )

        device = rheology_solver.device
        if device.is_capturing:
            with _ScopedDisableGC():
                wp.capture_while(condition, do_iteration_with_condition)
        else:
            with _ScopedDisableGC():
                with wp.ScopedCapture(force_module_load=False) as capture:
                    wp.capture_while(condition, do_iteration_with_condition)
            solve_graph = capture.graph
            wp.capture_launch(solve_graph)

            if verbose:
                residual = rheology_solver.eval_residual().numpy()
                res_l2, res_linf = math.sqrt(residual[0, 0]) / l2_tolerance_scale, math.sqrt(residual[1, 0])
                print(
                    f"{rheology_solver.name} terminated after {iteration_and_condition.numpy()[0]} iterations with residuals {res_l2}, {res_linf}"
                )

        iteration_and_condition.release()
    else:
        solve_granularity = rheology_solver.solve_granularity

        for batch in range(max_iterations // solve_granularity):
            for _k in range(solve_granularity):
                contact_solver.solve()
                rheology_solver.solve()

            residual = rheology_solver.eval_residual().numpy()
            res_l2, res_linf = math.sqrt(residual[0, 0]) / l2_tolerance_scale, math.sqrt(residual[1, 0])

            if verbose:
                print(
                    f"{rheology_solver.name} iteration #{(batch + 1) * solve_granularity} \t res(l2)={res_l2}, res(linf)={res_linf}"
                )
            if res_l2 < tolerance and res_linf < tolerance:
                break

    return solve_graph


def solve_rheology(
    solver: str,
    max_iterations: int,
    tolerance: float,
    momentum: MomentumData,
    rheology: RheologyData,
    collision: CollisionData,
    jacobi_warmstart_smoother_iterations: int = 0,
    temporary_store: fem.TemporaryStore | None = None,
    use_graph: bool = True,
    verbose: bool = True,
):
    """Solve coupled plasticity and collider contact to compute grid velocities.

    This function executes the implicit rheology loop that couples plastic
    stress update and nodal frictional contact with colliders:

    - Builds the Delassus operator diagonal blocks and rotates all local
      quantities into the decoupled eigenbasis (normal vs tangential).
    - Runs either Gauss-Seidel (with coloring) or Jacobi iterations to solve
      the local stress projection problem per strain node.
    - Applies collider impulses and, when provided, a rigidity coupling step on
      collider velocities each iteration.
    - Iterates until the residual on the stress update falls below
      ``tolerance`` or ``max_iterations`` is reached. Optionally records and
      executes CUDA graphs to reduce CPU overhead.

    On exit, the stress field is rotated back to world space and the elastic
    strain increment and plastic strain delta fields are produced.

    Args:
        solver: Solver type string. ``"gauss-seidel"``, ``"jacobi"``,
            ``"cg"``, or ``"cg+<solver>"`` (CG as initial guess then
            ``<solver>`` for the main solve).
        max_iterations: Maximum number of nonlinear iterations.
        tolerance: Solver tolerance for the stress residual (L2 norm).
        momentum: :class:`MomentumData` containing per-node inverse volume
            and velocity DOFs.
        rheology: :class:`RheologyData` containing strain/compliance matrices,
            yield parameters, coloring data, and output stress/strain arrays.
        collision: :class:`CollisionData` containing collider matrices, friction,
            adhesion, normals, velocities, rigidity operator, and impulse arrays.
        jacobi_warmstart_smoother_iterations: Number of Jacobi smoother
            iterations to run before the main Gauss-Seidel solve (ignored
            for Jacobi solver).
        temporary_store: Temporary storage arena for intermediate arrays.
        use_graph: If True, uses conditional CUDA graphs for the iteration loop.
        verbose: If True, prints residuals/iteration counts.

    Returns:
        A captured execution graph handle when ``use_graph`` is True and the
        device supports it; otherwise ``None``.
    """

    subgrid_collisions = collision.collider_mat.nnz > 0
    if subgrid_collisions:
        contact_solver = _SubgridContactSolver(momentum, collision, temporary_store)
    else:
        contact_solver = _NodalContactSolver(momentum, collision, temporary_store)

    contact_solver.apply_initial_guess()

    delassus_operator = _DelassusOperator(rheology, momentum, temporary_store)
    tolerance_scale = math.sqrt(1 + delassus_operator.size)

    if solver[:2] == "cg":  # matches "cg" or "cg+xxx"
        rheology_solver = _CGSolver(delassus_operator, temporary_store)
        rheology_solver.solve(tolerance, tolerance_scale, max_iterations, use_graph, verbose)
        rheology_solver.release()

        if solver == "cg":
            delassus_operator.apply_stress_delta(rheology.stress, momentum.velocity)
            delassus_operator.postprocess_stress_and_strain()
            delassus_operator.release()
            return None

        # use only as initial guess for the next solver
        solver = solver[3:]

    if solver == "gauss-seidel" and jacobi_warmstart_smoother_iterations > 0:
        # jacobi warmstart  smoother
        old_v = wp.clone(momentum.velocity)
        warmstart_solver = _JacobiSolver(delassus_operator, temporary_store)
        warmstart_solver.apply_initial_guess()
        for _ in range(jacobi_warmstart_smoother_iterations):
            warmstart_solver.solve()
        warmstart_solver.release()
        momentum.velocity.assign(old_v)

    if solver == "gauss-seidel":
        rheology_solver = _GaussSeidelSolver(delassus_operator, temporary_store)
    elif solver == "jacobi":
        rheology_solver = _JacobiSolver(delassus_operator, temporary_store)
    else:
        raise ValueError(f"Invalid solver: {solver}")

    rheology_solver.apply_initial_guess()

    solve_graph = _run_solver_loop(
        rheology_solver, contact_solver, max_iterations, tolerance, tolerance_scale, use_graph, verbose, temporary_store
    )

    # release temporary storage
    rheology_solver.release()
    contact_solver.release()

    delassus_operator.postprocess_stress_and_strain()
    delassus_operator.release()

    return solve_graph
