# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

from typing import Any

import warp as wp
import warp.fem as fem
import warp.sparse as sp
from warp.fem.linalg import symmetric_eigenvalues_qr

_DELASSUS_PROXIMAL_REG = wp.constant(1.0e-6)
"""Cutoff for the trace of the diagonal block of the Delassus operator to disable constraints"""

_SLIDING_NEWTON_TOL = wp.constant(1.0e-7)
"""Tolerance for the Newton method to solve for the sliding velocity"""

_INCLUDE_LEFTOVER_STRAIN = wp.constant(False)
"""Whether to include leftover strain (due to not fully-converged implicit solve) in the elastic strain.

More accurate, but less stable for stiff materials. Development toggle for experimentation."""

_USE_CAM_CLAY = wp.constant(False)
"""Use Modified Cam-Clay flow rule instead of the piecewise-linear anisotropic one. Development toggle for experimentation."""

_ISOTROPIC_LOCAL_LHS = wp.constant(False)
"""Use isotropic local left-hand side instead of anisotropic. Cheaper local solver but slower convergence. Development toggle for experimentation."""


vec6 = wp.types.vector(length=6, dtype=wp.float32)

mat66 = wp.types.matrix(shape=(6, 6), dtype=wp.float32)
mat55 = wp.types.matrix(shape=(5, 5), dtype=wp.float32)

mat13 = wp.vec3

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
def get_dilatancy(yield_params: YieldParamVec):
    return wp.clamp(yield_params[4], 0.0, 1.0)


@wp.func
def get_viscosity(yield_params: YieldParamVec):
    return wp.max(0.0, yield_params[5])


@wp.func
def normal_yield_bounds(yield_params: YieldParamVec):
    """Extract bounds for the normal stress from the yield surface definition."""
    return -wp.max(0.0, yield_params[1]), yield_params[0]


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
def shear_yield_stress_camclay(yield_params: YieldParamVec, r_N: float):
    r_N_min, r_N_max = normal_yield_bounds(yield_params)

    mu = wp.where(r_N_max > 0.0, wp.max(0.0, yield_params[3] / r_N_max), 0.0)

    r_N = wp.clamp(r_N, r_N_min, r_N_max)

    beta_sq = mu * mu / (1.0 - 2.0 * (r_N_min / r_N_max))
    y_sq = beta_sq * (r_N - r_N_min) * (r_N_max - r_N)

    return wp.sqrt(y_sq), 0.0, r_N_min, r_N_max


@wp.func
def _symmetric_part_op(b: wp.vec3, u: wp.vec3):
    return fem.SymmetricTensorMapper.value_to_dof_3d(wp.outer(u, b * 2.0))


@wp.func
def _symmetric_part_transposed_op(b: wp.vec3, sig: vec6):
    return fem.SymmetricTensorMapper.dof_to_value_3d(sig) @ (b * 0.5)


@wp.kernel
def compute_delassus_diagonal(
    split_mass: wp.bool,
    strain_mat_offsets: wp.array[int],
    strain_mat_columns: wp.array[int],
    strain_mat_values: wp.array[mat13],
    inv_volume: wp.array[float],
    compliance_mat_offsets: wp.array[int],
    compliance_mat_columns: wp.array[int],
    compliance_mat_values: wp.array[mat66],
    transposed_strain_mat_offsets: wp.array[int],
    delassus_rotation: wp.array[mat55],
    delassus_diagonal: wp.array[vec6],
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

        b_v0 = _symmetric_part_op(b_val, wp.vec3(1.0, 0.0, 0.0))
        diag_block += inv_frac * wp.outer(b_v0, b_v0)
        b_v1 = _symmetric_part_op(b_val, wp.vec3(0.0, 1.0, 0.0))
        diag_block += inv_frac * wp.outer(b_v1, b_v1)
        b_v2 = _symmetric_part_op(b_val, wp.vec3(0.0, 0.0, 1.0))
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

    if wp.static(_ISOTROPIC_LOCAL_LHS):
        diag = vec6(wp.max(diag))
        ev = wp.identity(n=6, dtype=float)

    delassus_diagonal[tau_i] = diag
    delassus_rotation[tau_i] = wp.transpose(ev[1:6, 1:6])


@wp.func
def unilateral_offset_to_strain_rhs(offset: float):
    return fem.SymmetricTensorMapper.value_to_dof_3d(offset * (2.0 / 3.0) * wp.identity(n=3, dtype=float))


@wp.kernel
def preprocess_stress_and_strain(
    unilateral_strain_offset: wp.array[float],
    strain_rhs: wp.array[vec6],
    stress: wp.array[vec6],
    yield_stress: wp.array[YieldParamVec],
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
        b += unilateral_offset_to_strain_rhs(offset)
        strain_rhs[tau_i] = b

        yield_params[1] = 0.0  # disable cohesion if offset > 0 (not compact)
        yield_stress[tau_i] = yield_params

    sig = stress[tau_i]
    stress[tau_i] = project_stress(sig, yield_params)


@wp.kernel
def postprocess_stress_and_strain(
    compliance_mat_offsets: wp.array[int],
    compliance_mat_columns: wp.array[int],
    compliance_mat_values: wp.array[mat66],
    strain_mat_offsets: wp.array[int],
    strain_mat_columns: wp.array[int],
    strain_mat_values: wp.array[mat13],
    delassus_diagonal: wp.array[vec6],
    delassus_rotation: wp.array[mat55],
    unilateral_strain_offset: wp.array[float],
    yield_params: wp.array[YieldParamVec],
    strain_node_volume: wp.array[float],
    strain_rhs: wp.array[vec6],
    stress: wp.array[vec6],
    velocity: wp.array[wp.vec3],
    elastic_strain: wp.array[vec6],
    plastic_strain: wp.array[vec6],
):
    """Computes elastic and plastic strain deltas after the solver iterations.

    Uses the generic (non-specialized) flow-rule solver to ensure correct
    results regardless of which compile-time flags were active during the
    iterative solve.
    """
    tau_i = wp.tid()

    minus_elastic_strain = strain_rhs[tau_i]
    minus_elastic_strain -= unilateral_offset_to_strain_rhs(unilateral_strain_offset[tau_i])
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
        world_plastic_strain += _symmetric_part_op(strain_mat_values[b], velocity[u_i])

    rot = delassus_rotation[tau_i]
    diag = delassus_diagonal[tau_i]

    loc_plastic_strain = _world_to_local(world_plastic_strain, rot)
    loc_stress = _world_to_local(stress[tau_i], rot)

    yp = yield_params[tau_i]
    loc_plastic_strain_new = wp.static(make_solve_flow_rule())(
        diag, loc_plastic_strain - wp.cw_mul(loc_stress, diag), loc_stress, yp, strain_node_volume[tau_i]
    )
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
def solve_sliding_no_dilatancy(
    D: Any,
    b: Any,
    yield_stress: float,
):
    """Simplified sliding solver when dilatancy=0 (theta=0).

    When there is no dilatancy coupling, gamma=0 and the residual
    simplifies to ``|b_T / (D + alpha I)| - yield_stress``.
    """

    b_T = b
    b_T[0] = 0.0

    if yield_stress <= 0.0:
        return b_T

    Dys = D * yield_stress

    alpha_min = wp.max(0.0, wp.length(b_T) - wp.max(Dys))
    alpha_max = wp.length(b_T) - wp.min(Dys)

    alpha_cur = alpha_min

    for _k in range(24):
        d_alpha = Dys + type(D)(alpha_cur)

        r_alpha = wp.cw_div(b_T, d_alpha)
        r_alpha_norm = wp.length(r_alpha)

        f = r_alpha_norm - 1.0
        df = wp.dot(r_alpha, -wp.cw_div(r_alpha, d_alpha * r_alpha_norm))

        delta = wp.min(-f / df, alpha_max - alpha_cur)
        if delta < _SLIDING_NEWTON_TOL * alpha_max:
            break
        alpha_cur += delta

    u = wp.cw_div(b_T * alpha_cur, Dys + type(D)(alpha_cur))
    u[0] = 0.0
    return u


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

        if delta_alpha < _SLIDING_NEWTON_TOL * alpha_max:
            break

        alpha_cur += delta_alpha

    u = wp.cw_div(b_T * alpha_cur, Dmu_rn + type(D)(alpha_cur))
    u[0] = theta * yield_stress_deriv * wp.length(u)

    return u


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

    ys, _dys, r_N_min, r_N_max = shear_yield_stress_camclay(yield_params, r_N0)

    if r_N_max <= 0.0:
        return b

    if wp.length_sq(r_T) < ys * ys:
        return vec6(0.0)

    if use_nacc:
        # Non-Associated Cam Clay
        b_T = b
        b_T[0] = 0.0
        u = solve_sliding_no_dilatancy(D, b_T, ys)
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

    Z = solve_sliding_no_dilatancy(W_sq_inv, -WinvX0, c)
    Y = wp.cw_mul(W_sq, Z + WinvX0)

    X = wp.cw_div(Y, W)

    r_N = r_mid + X[1]
    murn = wp.abs(X[2])

    r = wp.normalize(r_T) * murn
    r[0] = r_N
    u = wp.cw_mul(r, D) + b
    return u


def make_solve_flow_rule(has_viscosity: bool = True, has_dilatancy: bool = True):
    key = (has_viscosity, has_dilatancy)

    @fem.cache.dynamic_func(suffix=key)
    def solve_flow_rule_aniso_impl(
        D: vec6,
        b: vec6,
        r_guess: vec6,
        yield_params: YieldParamVec,
        strain_node_volume: float,
    ):
        if wp.static(has_dilatancy):
            dilatancy = get_dilatancy(yield_params)
        else:
            dilatancy = 0.0

        if wp.static(has_viscosity):
            D_visc = vec6(1.0) + get_viscosity(yield_params) / strain_node_volume * D
            D = wp.cw_div(D, D_visc)
            b = wp.cw_div(b, D_visc)

        if wp.static(_USE_CAM_CLAY):
            return solve_flow_rule_camclay(D, b, r_guess, yield_params)

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
        if wp.static(has_dilatancy):
            u = solve_sliding_aniso(D, u, ys, dys, dilatancy)
        else:
            u = solve_sliding_no_dilatancy(D, u, ys)

        # check for change of linear region
        r_N_new = (u[0] - b[0]) / D[0]
        r_N_clamp = wp.clamp(r_N_new, pmin, pmax)
        if r_N_clamp == r_N_new:
            return u

        # moved from conic part to constant part. clamp and resolve tangent part
        ys, dys, pmin, pmax = shear_yield_stress(yield_params, r_N_clamp)
        if wp.static(has_dilatancy):
            u = solve_sliding_aniso(D, b, ys, 0.0, dilatancy)
        else:
            u = solve_sliding_no_dilatancy(D, b, ys)
        u[0] = D[0] * (r_N_clamp - r_N0)

        return u

    return solve_flow_rule_aniso_impl


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
        ys, _dys, pmin, pmax = shear_yield_stress_camclay(yield_params, r_N)
    else:
        ys, _dys, pmin, pmax = shear_yield_stress(yield_params, r_N)

    r_T_n2 = wp.length_sq(r_T)
    if r_T_n2 > ys * ys:
        r_T *= ys / wp.sqrt(r_T_n2)

    r = r_T
    r[0] = wp.clamp(r_N, pmin, pmax)
    return r


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


def make_apply_stress_delta(strain_velocity_node_count: int = -1):
    @fem.cache.dynamic_func(suffix=strain_velocity_node_count)
    def apply_stress_delta_impl(
        tau_i: int,
        delta_stress: vec6,
        strain_mat_offsets: wp.array[int],
        strain_mat_columns: wp.array[int],
        strain_mat_values: wp.array[mat13],
        inv_mass_matrix: wp.array[float],
        velocities: wp.array[wp.vec3],
    ):
        """Updates particle velocities from a local stress delta."""

        block_beg = strain_mat_offsets[tau_i]

        if wp.static(strain_velocity_node_count > 0):
            for bk in range(strain_velocity_node_count):
                b = block_beg + bk
                u_i = strain_mat_columns[b]
                delta_u = inv_mass_matrix[u_i] * _symmetric_part_transposed_op(strain_mat_values[b], delta_stress)
                velocities[u_i] += delta_u
        else:
            block_end = strain_mat_offsets[tau_i + 1]
            for b in range(block_beg, block_end):
                u_i = strain_mat_columns[b]
                delta_u = inv_mass_matrix[u_i] * _symmetric_part_transposed_op(strain_mat_values[b], delta_stress)
                velocities[u_i] += delta_u

    return apply_stress_delta_impl


@wp.kernel
def apply_stress_delta_jacobi(
    transposed_strain_mat_offsets: wp.array[int],
    transposed_strain_mat_columns: wp.array[int],
    transposed_strain_mat_values: wp.array[mat13],
    inv_mass_matrix: wp.array[float],
    stress: wp.array[vec6],
    velocities: wp.array[wp.vec3],
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
        delta_u += _symmetric_part_transposed_op(transposed_strain_mat_values[b], delta_stress)

    velocities[u_i] += inv_mass * delta_u


@wp.kernel
def apply_velocity_delta(
    alpha: float,
    beta: float,
    strain_mat_offsets: wp.array[int],
    strain_mat_columns: wp.array[int],
    strain_mat_values: wp.array[mat13],
    velocity_delta: wp.array[wp.vec3],
    strain_prev: wp.array[vec6],
    strain: wp.array[vec6],
):
    """Computes strain from a velocity delta: ``strain = alpha * B @ velocity_delta + beta * strain_prev``."""

    tau_i = wp.tid()

    block_beg = strain_mat_offsets[tau_i]
    block_end = strain_mat_offsets[tau_i + 1]

    delta_stress = vec6(0.0)
    for b in range(block_beg, block_end):
        u_i = strain_mat_columns[b]
        delta_stress += _symmetric_part_op(strain_mat_values[b], velocity_delta[u_i])

    delta_stress *= alpha
    if beta != 0.0:
        delta_stress += beta * strain_prev[tau_i]

    strain[tau_i] = delta_stress


@wp.kernel
def apply_stress_gs(
    color: int,
    launch_dim: int,
    color_offsets: wp.array[int],
    color_blocks: wp.array2d[int],
    strain_mat_offsets: wp.array[int],
    strain_mat_columns: wp.array[int],
    strain_mat_values: wp.array[mat13],
    inv_mass_matrix: wp.array[float],  # Note: Likely inv_volume in context
    stress: wp.array[vec6],
    velocities: wp.array[wp.vec3],
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

            wp.static(make_apply_stress_delta())(
                tau_i,
                cur_stress,
                strain_mat_offsets,
                strain_mat_columns,
                strain_mat_values,
                inv_mass_matrix,
                velocities,
            )


def make_compute_local_strain(has_compliance_mat: bool = True, strain_velocity_node_count: int = -1):
    @fem.cache.dynamic_func(suffix=(has_compliance_mat, strain_velocity_node_count))
    def compute_local_strain_impl(
        tau_i: int,
        compliance_mat_offsets: wp.array[int],
        compliance_mat_columns: wp.array[int],
        compliance_mat_values: wp.array[mat66],
        strain_mat_offsets: wp.array[int],
        strain_mat_columns: wp.array[int],
        strain_mat_values: wp.array[mat13],
        local_strain_rhs: wp.array[vec6],
        velocities: wp.array[wp.vec3],
        local_stress: wp.array[vec6],
    ):
        """Computes the local strain based on the current stress and velocities."""
        tau = local_strain_rhs[tau_i]

        # tau += B v
        block_beg = strain_mat_offsets[tau_i]
        if wp.static(strain_velocity_node_count > 0):
            for bk in range(strain_velocity_node_count):
                b = block_beg + bk
                u_i = strain_mat_columns[b]
                tau += _symmetric_part_op(strain_mat_values[b], velocities[u_i])
        else:
            block_end = strain_mat_offsets[tau_i + 1]
            for b in range(block_beg, block_end):
                u_i = strain_mat_columns[b]
                tau += _symmetric_part_op(strain_mat_values[b], velocities[u_i])

        # tau += C sigma
        if wp.static(has_compliance_mat):
            comp_block_beg = compliance_mat_offsets[tau_i]
            comp_block_end = compliance_mat_offsets[tau_i + 1]
            for b in range(comp_block_beg, comp_block_end):
                sig_i = compliance_mat_columns[b]
                tau += compliance_mat_values[b] @ local_stress[sig_i]

        return tau

    return compute_local_strain_impl


def make_solve_local_stress(has_viscosity: bool, has_dilatancy: bool, has_rotation: bool = not _ISOTROPIC_LOCAL_LHS):
    """Return a specialized Warp func that applies the local stress projection for one strain node.

    Optionally rotates strain and stress into the Delassus eigenbasis before solving
    and back to world space on return. Each unique ``(has_viscosity, has_dilatancy,
    has_rotation)`` combination is compiled once and cached.
    """
    key = (has_viscosity, has_dilatancy, has_rotation)

    @fem.cache.dynamic_func(suffix=key)
    def solve_local_stress_impl(
        tau_i: int,
        strain_rhs: vec6,
        yield_params: wp.array[YieldParamVec],
        strain_node_volume: wp.array[float],
        delassus_diagonal: wp.array[vec6],
        delassus_rotation: wp.array[mat55],
        cur_stress: wp.array[vec6],
    ):
        D = delassus_diagonal[tau_i]
        if wp.static(has_rotation):
            rot = delassus_rotation[tau_i]
            local_strain = _world_to_local(strain_rhs, rot)
            local_stress = _world_to_local(cur_stress[tau_i], rot)
        else:
            local_strain = strain_rhs
            local_stress = cur_stress[tau_i]

        tau_new = wp.static(make_solve_flow_rule(has_viscosity, has_dilatancy))(
            D,
            local_strain - wp.cw_mul(local_stress, D),
            local_stress,
            yield_params[tau_i],
            strain_node_volume[tau_i],
        )

        delta_stress_loc = wp.cw_div(tau_new - local_strain, D)

        if wp.static(has_rotation):
            return _local_to_world(delta_stress_loc, rot)
        else:
            return delta_stress_loc

    return solve_local_stress_impl


def make_jacobi_solve_kernel(
    has_viscosity: bool,
    has_dilatancy: bool,
    has_compliance_mat: bool,
    strain_velocity_node_count: int = -1,
    has_rotation: bool = not _ISOTROPIC_LOCAL_LHS,
):
    """Return a Jacobi-style per-node stress solve kernel specialized for the given feature flags.

    Each unique combination of flags produces a separate Warp kernel compiled and cached
    via :func:`fem.cache.dynamic_kernel`.
    """

    key = (has_viscosity, has_dilatancy, has_compliance_mat, has_rotation, strain_velocity_node_count)

    @fem.cache.dynamic_kernel(suffix=key, kernel_options={"fast_math": True})
    def jacobi_solve_kernel_impl(
        yield_params: wp.array[YieldParamVec],
        strain_node_volume: wp.array[float],
        compliance_mat_offsets: wp.array[int],
        compliance_mat_columns: wp.array[int],
        local_compliance_mat_values: wp.array[mat66],
        strain_mat_offsets: wp.array[int],
        strain_mat_columns: wp.array[int],
        strain_mat_values: wp.array[mat13],
        delassus_diagonal: wp.array[vec6],
        delassus_rotation: wp.array[mat55],
        local_strain_rhs: wp.array[vec6],
        velocities: wp.array[wp.vec3],
        local_stress: wp.array[vec6],
        delta_correction: wp.array[vec6],
    ):
        tau_i = wp.tid()

        local_strain = wp.static(make_compute_local_strain(has_compliance_mat, strain_velocity_node_count))(
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

        delta_correction[tau_i] = wp.static(make_solve_local_stress(has_viscosity, has_dilatancy, has_rotation))(
            tau_i,
            local_strain,
            yield_params,
            strain_node_volume,
            delassus_diagonal,
            delassus_rotation,
            local_stress,
        )

    return jacobi_solve_kernel_impl


def make_gs_solve_kernel(
    has_viscosity: bool,
    has_dilatancy: bool,
    has_compliance_mat: bool,
    strain_velocity_node_count: int = -1,
    has_rotation: bool = not _ISOTROPIC_LOCAL_LHS,
):
    """Return a Gauss-Seidel colored-block stress solve kernel specialized for the given feature flags.

    The returned kernel processes strain nodes in color order to avoid write conflicts,
    immediately propagating velocity updates within each color. Each unique combination
    of flags produces a separate Warp kernel compiled and cached via
    :func:`fem.cache.dynamic_kernel`.
    """
    key = (has_viscosity, has_dilatancy, has_compliance_mat, has_rotation, strain_velocity_node_count)

    @fem.cache.dynamic_kernel(suffix=key, kernel_options={"fast_math": True})
    def gs_solve_kernel_impl(
        color: int,
        launch_dim: int,
        color_offsets: wp.array[int],
        color_blocks: wp.array2d[int],
        yield_params: wp.array[YieldParamVec],
        strain_node_volume: wp.array[float],
        compliance_mat_offsets: wp.array[int],
        compliance_mat_columns: wp.array[int],
        compliance_mat_values: wp.array[mat66],
        strain_mat_offsets: wp.array[int],
        strain_mat_columns: wp.array[int],
        strain_mat_values: wp.array[mat13],
        delassus_diagonal: wp.array[vec6],
        delassus_rotation: wp.array[mat55],
        inv_mass_matrix: wp.array[float],
        local_strain_rhs: wp.array[vec6],
        velocities: wp.array[wp.vec3],
        local_stress: wp.array[vec6],
        delta_correction: wp.array[vec6],
    ):
        i = wp.tid()
        color_beg = color_offsets[color] + i
        color_end = color_offsets[color + 1]

        for color_offset in range(color_beg, color_end, launch_dim):
            beg, end = color_blocks[0, color_offset], color_blocks[1, color_offset]
            for tau_i in range(beg, end):
                local_strain = wp.static(make_compute_local_strain(has_compliance_mat, strain_velocity_node_count))(
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

                delta_stress = wp.static(make_solve_local_stress(has_viscosity, has_dilatancy, has_rotation))(
                    tau_i,
                    local_strain,
                    yield_params,
                    strain_node_volume,
                    delassus_diagonal,
                    delassus_rotation,
                    local_stress,
                )

                local_stress[tau_i] += delta_stress
                delta_correction[tau_i] = delta_stress

                wp.static(make_apply_stress_delta(strain_velocity_node_count))(
                    tau_i,
                    delta_stress,
                    strain_mat_offsets,
                    strain_mat_columns,
                    strain_mat_values,
                    inv_mass_matrix,
                    velocities,
                )

    return gs_solve_kernel_impl


@wp.kernel
def jacobi_preconditioner(
    delassus_diagonal: wp.array[vec6],
    delassus_rotation: wp.array[mat55],
    x: wp.array[vec6],
    y: wp.array[vec6],
    z: wp.array[vec6],
    alpha: float,
    beta: float,
):
    tau_i = wp.tid()
    rot = delassus_rotation[tau_i]
    diag = delassus_diagonal[tau_i]

    Wx = _local_to_world(wp.cw_div(_world_to_local(x[tau_i], rot), diag), rot)
    z[tau_i] = alpha * Wx + beta * y[tau_i]


@wp.kernel
def evaluate_strain_residual(
    delta_stress: wp.array[vec6],
    delassus_diagonal: wp.array[vec6],
    delassus_rotation: wp.array[mat55],
    residual: wp.array[float],
):
    tau_i = wp.tid()
    local_strain_delta = wp.cw_mul(
        _world_to_local(delta_stress[tau_i], delassus_rotation[tau_i]), delassus_diagonal[tau_i]
    )
    r = wp.length_sq(local_strain_delta)

    residual[tau_i] = r
