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
from typing import Any

import warp as wp
import warp.fem as fem
import warp.sparse as sp
from warp.fem.linalg import array_axpy, symmetric_eigenvalues_qr

_DELASSUS_PROXIMAL_REG = wp.constant(1.0e-9)
"""Cutoff for the trace of the diagonal block of the Delassus operator to disable constraints"""

__SLIDING_NEWTON_TOL = wp.constant(1.0e-12)
"""Tolerance for the Newton method to solve for the sliding velocity"""

vec6 = wp.types.vector(length=6, dtype=wp.float32)
mat66 = wp.types.matrix(shape=(6, 6), dtype=wp.float32)
mat63 = wp.types.matrix(shape=(6, 3), dtype=wp.float32)
mat36 = wp.types.matrix(shape=(3, 6), dtype=wp.float32)

wp.set_module_options({"enable_backward": False})


class YieldParamVec(wp.vec4):
    """Compact yield surface definition in an interpolation-friendly format:
    [p_max sqrt_3_2, -p_min sqrt_3_2, s_max, mu p_max] in scaled units.

    The scaling by sqrt(3/2) is related to the orthogonal mapping from spherical/deviatoric
    tensors to vectors in R^6.
    """

    @wp.func
    def from_values(friction_coeff: float, yield_pressure: float, tensile_yield_ratio: float, yield_stress: float):
        pressure_scale = wp.sqrt(3.0 / 2.0)
        return YieldParamVec(
            yield_pressure * pressure_scale,
            tensile_yield_ratio * yield_pressure * pressure_scale,
            yield_stress,
            friction_coeff * yield_pressure,
        )


@wp.func
def normal_yield_bounds(yield_params: YieldParamVec):
    """Extract bounds for the normal stress from the yield surface definition."""
    return -yield_params[1], yield_params[0]


@wp.func
def shear_yield_stress(yield_params: YieldParamVec, r_N: float):
    """Maximum deviatoric stress for a given value of the normal stress."""
    p_min, p_max = normal_yield_bounds(yield_params)

    mu = wp.where(p_max > 0.0, yield_params[3] / p_max, 0.0)
    s = yield_params[2]
    return s + wp.min(0.5 * yield_params[3], mu * wp.min(r_N - p_min, p_max - r_N))


@wp.kernel
def compute_delassus_diagonal(
    split_mass: wp.bool,
    strain_mat_offsets: wp.array(dtype=int),
    strain_mat_columns: wp.array(dtype=int),
    strain_mat_values: wp.array(dtype=mat63),
    inv_volume: wp.array(dtype=float),
    compliance_mat_diagonal: wp.array(dtype=mat66),
    transposed_strain_mat_offsets: wp.array(dtype=int),
    strain_rhs: wp.array(dtype=vec6),
    stress: wp.array(dtype=vec6),
    delassus_rotation: wp.array(dtype=mat66),
    delassus_diagonal: wp.array(dtype=vec6),
    delassus_normal: wp.array(dtype=vec6),
    local_strain_mat_values: wp.array(dtype=mat63),
    local_strain_rhs: wp.array(dtype=vec6),
    local_stress: wp.array(dtype=vec6),
):
    """
    Computes the diagonal blocks of the Delassus operator and performs
    an eigendecomposition to decouple stress components.

    This kernel iterates over each constraint (tau_i) and:
    1. Assembles the diagonal block of the Delassus operator by summing contributions
       from connected particles/nodes (u_i).
    2. If mass splitting is enabled, it scales contributions by the (inverse) number of
       constraints a particle is involved in.
    3. Performs an eigendecomposition (symmetric_eigenvalues_qr) of the
       assembled diagonal block.
    4. Handles potential numerical issues by falling back to the diagonal if
       eigendecomposition fails or if modes are null.
    5. Stores the eigenvalues (delassus_diagonal) and the transpose of eigenvectors
       (forming a rotation matrix, delassus_rotation).
    6. Transforms the strain_rhs, stress, strain_mat_values, and stress_strain_matrices
       into the eigenbasis.
    7. Computes the normal vector in the rotated frame (delassus_normal).
    8. If the trace of the diagonal block is too small, it disables the constraint.
    """
    tau_i = wp.tid()
    block_beg = strain_mat_offsets[tau_i]
    block_end = strain_mat_offsets[tau_i + 1]

    diag_block = mat66(0.0)
    if compliance_mat_diagonal:
        diag_block = compliance_mat_diagonal[tau_i]

    mass_ratio = float(1.0)
    for b in range(block_beg, block_end):
        u_i = strain_mat_columns[b]

        if split_mass:
            mass_ratio = float(transposed_strain_mat_offsets[u_i + 1] - transposed_strain_mat_offsets[u_i])

        b_val = strain_mat_values[b]
        inv_frac = inv_volume[u_i] * mass_ratio

        diag_block += (b_val * inv_frac) @ wp.transpose(b_val)

    diag_block += _DELASSUS_PROXIMAL_REG * wp.identity(n=6, dtype=float)

    # Remove shear-divergence coupling
    # (current implementation of solve_coulomb_aniso normal and tangential responses are independent)
    for k in range(1, 6):
        diag_block[0, k] = 0.0
        diag_block[k, 0] = 0.0

    diag, ev = symmetric_eigenvalues_qr(diag_block, _DELASSUS_PROXIMAL_REG * 0.1)

    # symmetric_eigenvalues_qr may return nans for small coefficients
    if not (wp.ddot(ev, ev) < 1.0e16 and wp.length_sq(diag) < 1.0e16):
        # wp.print(diag_block)
        # wp.print(diag)
        diag = wp.get_diag(diag_block)
        ev = wp.identity(n=6, dtype=float)

    delassus_diagonal[tau_i] = diag
    delassus_rotation[tau_i] = wp.transpose(ev)

    # Apply rotation to contact data
    nor = ev * vec6(1.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    delassus_normal[tau_i] = nor

    local_strain_rhs[tau_i] = ev * strain_rhs[tau_i]
    local_stress[tau_i] = wp.cw_mul(ev * stress[tau_i], diag)

    for b in range(block_beg, block_end):
        local_strain_mat_values[b] = ev * strain_mat_values[b]


@wp.kernel
def project_initial_stress(
    stress: wp.array(dtype=vec6),
    yield_stress: wp.array(dtype=YieldParamVec),
):
    """Project the initial stress guess onto the yield surface."""

    tau_i = wp.tid()

    yield_params = yield_stress[tau_i]

    sig = stress[tau_i]

    # Only accept initial stress guesses when there's no adhesion
    # Otherwise there's risk of amplifying instabilities in the stress space
    # (e.g. checkerboard patterns)
    # TODO find a more focused way to do this
    p_min, _p_max = normal_yield_bounds(yield_params)
    if p_min < 0.0:
        sig = vec6(0.0)

    nor = vec6(1.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    stress[tau_i] = project_stress(sig, nor, yield_params)


@wp.kernel
def rotate_and_scale_compliance_mat(
    compliance_mat_offsets: wp.array(dtype=int),
    compliance_mat_columns: wp.array(dtype=int),
    compliance_mat_values: wp.array(dtype=mat66),
    delassus_rotation: wp.array(dtype=mat66),
    delassus_diagonal: wp.array(dtype=vec6),
):
    """Rotate and scale the compliance matrix according to the local basis."""

    sig_i = wp.tid()
    block_beg = compliance_mat_offsets[sig_i]
    block_end = compliance_mat_offsets[sig_i + 1]

    for b in range(block_beg, block_end):
        tau_i = compliance_mat_columns[b]

        compliance_mat_values[b] = (
            wp.transpose(delassus_rotation[sig_i])
            @ compliance_mat_values[b]
            @ delassus_rotation[tau_i]
            @ wp.diag(1.0 / delassus_diagonal[tau_i])
        )


@wp.kernel
def rotate_and_scale_transposed_strain_mat(
    tr_strain_mat_offsets: wp.array(dtype=int),
    tr_strain_mat_columns: wp.array(dtype=int),
    tr_strain_mat_values: wp.array(dtype=mat36),
    inv_volume: wp.array(dtype=float),
    delassus_rotation: wp.array(dtype=mat66),
    delassus_diagonal: wp.array(dtype=vec6),
):
    """
    Rotate and scale the transposed strain matrix according to the local basis.
    (Jacobi solver only)
    """

    u_i = wp.tid()
    block_beg = tr_strain_mat_offsets[u_i]
    block_end = tr_strain_mat_offsets[u_i + 1]

    for b in range(block_beg, block_end):
        tau_i = tr_strain_mat_columns[b]

        tr_strain_mat_values[b] = (
            inv_volume[u_i]
            * tr_strain_mat_values[b]
            @ delassus_rotation[tau_i]
            @ wp.diag(1.0 / delassus_diagonal[tau_i])
        )


@wp.kernel
def postprocess_stress_and_strain(
    delassus_rotation: wp.array(dtype=mat66),
    delassus_diagonal: wp.array(dtype=vec6),
    compliance_mat_offsets: wp.array(dtype=int),
    compliance_mat_columns: wp.array(dtype=int),
    local_compliance_mat_values: wp.array(dtype=mat66),
    strain_mat_offsets: wp.array(dtype=int),
    strain_mat_columns: wp.array(dtype=int),
    local_strain_mat_values: wp.array(dtype=mat63),
    node_volume: wp.array(dtype=float),
    local_stress: wp.array(dtype=vec6),
    velocity: wp.array(dtype=wp.vec3),
    stress: wp.array(dtype=vec6),
    elastic_strain: wp.array(dtype=vec6),
    plastic_strain: wp.array(dtype=vec6),
):
    """
    Transforms stress back to the original basis and computes final
    elastic and plastic strain deltas after the solver iterations.
    """
    tau_i = wp.tid()
    rot = delassus_rotation[tau_i]
    diag = delassus_diagonal[tau_i]

    loc_stress = local_stress[tau_i]

    loc_strain = elastic_strain[tau_i]
    comp_block_beg = compliance_mat_offsets[tau_i]
    comp_block_end = compliance_mat_offsets[tau_i + 1]
    for b in range(comp_block_beg, comp_block_end):
        sig_i = compliance_mat_columns[b]
        loc_strain += local_compliance_mat_values[b] * local_stress[sig_i]

    loc_plastic_strain = loc_strain
    block_beg = strain_mat_offsets[tau_i]
    block_end = strain_mat_offsets[tau_i + 1]
    for b in range(block_beg, block_end):
        u_i = strain_mat_columns[b]
        loc_plastic_strain += local_strain_mat_values[b] * velocity[u_i]

    stress[tau_i] = rot * wp.cw_div(loc_stress, diag)
    elastic_strain[tau_i] = -rot * loc_strain

    # The 2 factor is due to the SymTensorMapping being othonormal with (tau:sig)/2
    plastic_strain[tau_i] = (rot * loc_plastic_strain) / wp.max(1.0e-4, 2.0 * node_volume[tau_i])


@wp.func
def eval_sliding_residual(alpha: float, D: vec6, b_T: vec6):
    """Evaluates the value and gradient of the residual of the
    sliding velocity-to-force ratio
    """
    d_alpha = D + vec6(alpha)

    r_alpha = wp.cw_div(b_T, d_alpha)
    dr_dalpha = -wp.cw_div(r_alpha, d_alpha)

    f = wp.dot(r_alpha, r_alpha) - 1.0
    df_dalpha = 2.0 * wp.dot(r_alpha, dr_dalpha)
    return f, df_dalpha


@wp.func
def solve_sliding_aniso(D: vec6, b_T: vec6, yield_stress: float):
    """Solves for the tangential component of the relative velocity in the 'sliding' case
    of the frictional contact model.

    Returns:
        Tangential component of the relative velocity u_T such that
        u_T = alpha r_T = D r_T + b_T and |r_T| = yield_stress
    """

    if yield_stress <= 0.0:
        return b_T

    # Viscous shear opposite to tangential stress, zero divergence
    # find alpha, r_t,  mu_rn, (D + alpha/(mu r_n) I) r_t + b_t = 0, |r_t| = mu r_n
    # find alpha,  |(D mu r_n + alpha I)^{-1} b_t|^2 = 1.0

    Dmu_rn = D * yield_stress

    alpha_0 = wp.length(b_T)
    alpha_max = alpha_0 - wp.min(Dmu_rn)
    alpha_min = wp.max(0.0, alpha_0 - wp.max(Dmu_rn))

    # We're looking for the root of an hyperbola, approach using Newton from the left
    alpha_cur = alpha_min

    for _k in range(24):
        f_cur, df_dalpha = eval_sliding_residual(alpha_cur, Dmu_rn, b_T)
        delta_alpha = -f_cur / df_dalpha

        # delta_alpha should always be positive, no need to take abs
        if delta_alpha < __SLIDING_NEWTON_TOL:
            break

        alpha_cur = wp.clamp(alpha_cur + delta_alpha, alpha_min, alpha_max)

    u_T = wp.cw_div(b_T * alpha_cur, Dmu_rn + vec6(alpha_cur))

    return u_T


@wp.func
def solve_flow_rule_aniso(
    D: vec6,
    b: vec6,
    nor: vec6,
    off: float,
    yield_params: YieldParamVec,
):
    """Solves the local non-associated flow-rule problem.
    u = D r + b

    r_N = r_N- + r_N+
    u_n           in NC(p_min <= r_N- <= 0)
    u_n + off nor in NC(0 <= r_N+ <= p_max)

    u_T in NC(|r_T| <= alpha s(r_N))
    """

    # Note: this assumes that D.nor = lambda nor
    # i.e. nor should be along one canonical axis
    # (solve_sliding aniso would get a lot more complex otherwise as normal and tangential
    # responses become interlinked)

    b_N = wp.dot(b, nor)

    r_0 = -wp.cw_div(b, D)
    r_N0 = wp.dot(r_0, nor)
    r_N_min, r_N_max = normal_yield_bounds(yield_params)

    r_N = wp.clamp(r_N0, r_N_min, 0.0) + wp.clamp(r_N0 - off / D[0], 0.0, r_N_max)

    u_N = r_N * wp.cw_mul(nor, D) + b_N * nor

    # Static friction, zero shear
    mu_rn = shear_yield_stress(yield_params, r_N)
    r_T = r_0 - r_N0 * nor
    if wp.length_sq(r_T) <= mu_rn * mu_rn:
        return u_N

    # Sliding case
    b_T = b - b_N * nor
    return u_N + solve_sliding_aniso(D, b_T, mu_rn)


@wp.func
def project_stress(
    r: vec6,
    nor: vec6,
    yield_params: YieldParamVec,
):
    """Projects a stress vector onto the yield surface."""

    r_N = wp.dot(r, nor)
    r_T = r - r_N * nor

    r_N_min, r_N_max = normal_yield_bounds(yield_params)
    r_N = wp.clamp(r_N, r_N_min, r_N_max)
    mu_rn = shear_yield_stress(yield_params, r_N)

    r_T_n2 = wp.length_sq(r_T)
    if r_T_n2 > mu_rn * mu_rn:
        r_T *= mu_rn / wp.sqrt(r_T_n2)

    return r_N * nor + r_T


@wp.func
def compute_local_strain(
    tau_i: int,
    compliance_mat_offsets: wp.array(dtype=int),
    compliance_mat_columns: wp.array(dtype=int),
    local_compliance_mat_values: wp.array(dtype=mat66),
    strain_mat_offsets: wp.array(dtype=int),
    strain_mat_columns: wp.array(dtype=int),
    local_strain_mat_values: wp.array(dtype=mat63),
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
        tau += local_strain_mat_values[b] * velocities[u_i]

    # tau += C sigma
    comp_block_beg = compliance_mat_offsets[tau_i]
    comp_block_end = compliance_mat_offsets[tau_i + 1]
    for b in range(comp_block_beg, comp_block_end):
        sig_i = compliance_mat_columns[b]
        tau += local_compliance_mat_values[b] * local_stress[sig_i]

    return tau


@wp.func
def solve_local_stress(
    tau_i: int,
    D: vec6,
    local_strain: vec6,
    yield_params: wp.array(dtype=YieldParamVec),
    delassus_normal: wp.array(dtype=vec6),
    unilateral_strain_offset: wp.array(dtype=float),
    local_stress: wp.array(dtype=vec6),
):
    """Computes the stress delta required to satisfy the local flow rule."""

    nor = delassus_normal[tau_i]
    cur_stress = local_stress[tau_i]

    tau_new = solve_flow_rule_aniso(
        D,
        local_strain - cur_stress,
        nor,
        unilateral_strain_offset[tau_i],
        yield_params[tau_i],
    )

    return tau_new - local_strain


@wp.kernel
def solve_local_stress_jacobi(
    yield_params: wp.array(dtype=YieldParamVec),
    compliance_mat_offsets: wp.array(dtype=int),
    compliance_mat_columns: wp.array(dtype=int),
    local_compliance_mat_values: wp.array(dtype=mat66),
    strain_mat_offsets: wp.array(dtype=int),
    strain_mat_columns: wp.array(dtype=int),
    local_strain_mat_values: wp.array(dtype=mat63),
    delassus_diagonal: wp.array(dtype=vec6),
    delassus_normal: wp.array(dtype=vec6),
    local_strain_rhs: wp.array(dtype=vec6),
    unilateral_strain_offset: wp.array(dtype=float),
    velocities: wp.array(dtype=wp.vec3),
    local_stress: wp.array(dtype=vec6),
    delta_correction: wp.array(dtype=vec6),
):
    """
    Solves the local stress problem for each constraint in a Jacobi-like manner.
    """
    tau_i = wp.tid()
    D = delassus_diagonal[tau_i]

    local_strain = compute_local_strain(
        tau_i,
        compliance_mat_offsets,
        compliance_mat_columns,
        local_compliance_mat_values,
        strain_mat_offsets,
        strain_mat_columns,
        local_strain_mat_values,
        local_strain_rhs,
        velocities,
        local_stress,
    )

    delta_correction[tau_i] = solve_local_stress(
        tau_i,
        D,
        local_strain,
        yield_params,
        delassus_normal,
        unilateral_strain_offset,
        local_stress,
    )


@wp.func
def apply_stress_delta(
    tau_i: int,
    delta_stress: vec6,
    strain_mat_offsets: wp.array(dtype=int),
    strain_mat_columns: wp.array(dtype=int),
    local_strain_mat_values: wp.array(dtype=mat63),
    inv_mass_matrix: wp.array(dtype=float),
    velocities: wp.array(dtype=wp.vec3),
):
    """Updates particle velocities from a local stress delta."""

    block_beg = strain_mat_offsets[tau_i]
    block_end = strain_mat_offsets[tau_i + 1]

    for b in range(block_beg, block_end):
        u_i = strain_mat_columns[b]
        delta_impulse = delta_stress @ local_strain_mat_values[b]
        velocities[u_i] += inv_mass_matrix[u_i] * delta_impulse


@wp.kernel
def apply_stress_gs(
    color: int,
    launch_dim: int,
    color_nodes_per_element: int,
    color_offsets: wp.array(dtype=int),
    color_indices: wp.array(dtype=int),
    strain_mat_offsets: wp.array(dtype=int),
    strain_mat_columns: wp.array(dtype=int),
    local_strain_mat_values: wp.array(dtype=mat63),
    delassus_diagonal: wp.array(dtype=vec6),
    inv_mass_matrix: wp.array(dtype=float),  # Note: Likely inv_volume in context
    local_stress: wp.array(dtype=vec6),
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
        beg = color_indices[color_offset]
        end = beg + color_nodes_per_element
        for tau_i in range(beg, end):
            D = delassus_diagonal[tau_i]
            cur_stress = local_stress[tau_i]

            apply_stress_delta(
                tau_i,
                wp.cw_div(cur_stress, D),
                strain_mat_offsets,
                strain_mat_columns,
                local_strain_mat_values,
                inv_mass_matrix,
                velocities,
            )


@wp.kernel
def solve_local_stress_gs(
    color: int,
    launch_dim: int,
    color_nodes_per_element: int,
    color_offsets: wp.array(dtype=int),
    color_indices: wp.array(dtype=int),
    yield_params: wp.array(dtype=YieldParamVec),
    compliance_mat_offsets: wp.array(dtype=int),
    compliance_mat_columns: wp.array(dtype=int),
    local_compliance_mat_values: wp.array(dtype=mat66),
    strain_mat_offsets: wp.array(dtype=int),
    strain_mat_columns: wp.array(dtype=int),
    local_strain_mat_values: wp.array(dtype=mat63),
    delassus_diagonal: wp.array(dtype=vec6),
    delassus_normal: wp.array(dtype=vec6),
    inv_mass_matrix: wp.array(dtype=float),  # Note: Likely inv_volume in context
    local_strain_rhs: wp.array(dtype=vec6),
    unilateral_strain_offset: wp.array(dtype=float),
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
        beg = color_indices[color_offset]
        end = beg + color_nodes_per_element
        for tau_i in range(beg, end):
            local_strain = compute_local_strain(
                tau_i,
                compliance_mat_offsets,
                compliance_mat_columns,
                local_compliance_mat_values,
                strain_mat_offsets,
                strain_mat_columns,
                local_strain_mat_values,
                local_strain_rhs,
                velocities,
                local_stress,
            )

            D = delassus_diagonal[tau_i]
            delta_stress = solve_local_stress(
                tau_i,
                D,
                local_strain,
                yield_params,
                delassus_normal,
                unilateral_strain_offset,
                local_stress,
            )

            local_stress[tau_i] += delta_stress
            delta_correction[tau_i] = delta_stress  # for residual evaluation

            apply_stress_delta(
                tau_i,
                wp.cw_div(delta_stress, D),
                strain_mat_offsets,
                strain_mat_columns,
                local_strain_mat_values,
                inv_mass_matrix,
                velocities,
            )


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


@wp.kernel
def apply_nodal_impulse(
    collider_impulse: wp.array(dtype=wp.vec3),
    collider_friction: wp.array(dtype=float),
    inv_mass: wp.array(dtype=float),
    collider_inv_mass: wp.array(dtype=float),
    velocities: wp.array(dtype=wp.vec3),
    collider_velocities: wp.array(dtype=wp.vec3),
):
    """
    Applies pre-computed impulses to particles and colliders.
    """
    i = wp.tid()

    friction_coeff = collider_friction[i]
    if friction_coeff < 0.0:
        collider_impulse[i] = wp.vec3(0.0)
    else:
        velocities[i] += inv_mass[i] * collider_impulse[i]
        collider_velocities[i] -= collider_inv_mass[i] * collider_impulse[i]


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
):
    """
    Solves for frictional impulses at nodes interacting with colliders.

    For each node (i) potentially in contact:
    1. Skips if friction coefficient is negative (no friction).
    2. Calculates the relative velocity `u0` between the particle and collider,
       accounting for any existing normal impulse.
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
    delta_impulse = delta_u / w

    impulse[i] += delta_impulse
    velocities[i] += inv_mass[i] * delta_impulse
    collider_velocities[i] -= collider_inv_mass[i] * delta_impulse


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
    collider_inv_mass: wp.array(dtype=float),
    collider_friction: wp.array(dtype=float),
    impulse: wp.array(dtype=wp.vec3),
    delta_impulse: wp.array(dtype=wp.vec3),
    collider_velocities: wp.array(dtype=wp.vec3),
):
    i = wp.tid()

    friction_coeff = collider_friction[i]
    if friction_coeff < 0.0:
        impulse[i] = wp.vec3(0.0)

    delta_lambda = impulse[i]
    delta_impulse[i] = delta_lambda
    collider_velocities[i] -= collider_inv_mass[i] * delta_lambda


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
    collider_inv_mass: wp.array(dtype=float),
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
    collider_velocities[i] -= collider_inv_mass[i] * delta_lambda


_TILED_SUM_BLOCK_DIM = 512


@wp.kernel
def _tiled_sum_kernel(
    square_input: bool,
    data: wp.array(dtype=float),
    partial_sums: wp.array(dtype=float),
):
    block_id = wp.tid()

    tile = wp.tile_load(data, shape=_TILED_SUM_BLOCK_DIM, offset=block_id * _TILED_SUM_BLOCK_DIM)

    if square_input:
        tile = wp.tile_map(wp.mul, tile, tile)

    wp.tile_store(partial_sums, wp.tile_sum(tile), offset=block_id)


class ArraySquaredNorm:
    """Utility to compute squared L2 norm of a large array via tiled reductions."""

    def __init__(self, max_length: int, device=None, temporary_store=None):
        self.tile_size = _TILED_SUM_BLOCK_DIM
        self.device = device

        num_blocks = (max_length + self.tile_size - 1) // self.tile_size
        self.partial_sums_a = fem.borrow_temporary(
            temporary_store, shape=(num_blocks,), dtype=float, device=self.device
        )
        self.partial_sums_b = fem.borrow_temporary(
            temporary_store, shape=(num_blocks,), dtype=float, device=self.device
        )

        square_input = True
        self.sum_launch: wp.Launch = wp.launch(
            _tiled_sum_kernel,
            dim=(num_blocks, self.tile_size),
            inputs=(
                square_input,
                self.partial_sums_a,
            ),
            outputs=(self.partial_sums_b,),
            block_dim=self.tile_size,
            record_cmd=True,
        )

    # Result contains a single value, the sum of the array (will get updated by this function)
    def compute_squared_norm(self, data: wp.array(dtype=Any)):
        # cast vector types to float
        if data.dtype != float:
            data = wp.array(data, dtype=float).flatten()

        array_length = data.shape[0]
        square_input = True

        flip_flop = False
        while True:
            num_blocks = (array_length + self.tile_size - 1) // self.tile_size
            partial_sums = (self.partial_sums_a if flip_flop else self.partial_sums_b)[:num_blocks]

            self.sum_launch.set_param_at_index(0, square_input)
            self.sum_launch.set_param_at_index(1, data)
            self.sum_launch.set_param_at_index(2, partial_sums)
            self.sum_launch.set_dim((num_blocks, self.tile_size))
            self.sum_launch.launch()

            array_length = num_blocks
            data = partial_sums
            square_input = False

            flip_flop = not flip_flop

            if num_blocks == 1:
                break

        return data[:1]

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
    min_iterations: int,
    max_iterations: int,
    residual: wp.array(dtype=float),
    iteration: wp.array(dtype=int),
    condition: wp.array(dtype=int),
):
    cur_it = iteration[0] + 1
    stop = (wp.sqrt(residual[0]) < residual_threshold and cur_it > min_iterations) or cur_it > max_iterations

    iteration[0] = cur_it
    condition[0] = wp.where(stop, 0, 1)


def apply_rigidity_operator(rigidity_operator, prev_collider_velocity, collider_velocity, delta_body_qd):
    """Apply collider rigidity feedback to the current collider velocities.

    Computes and applies a velocity correction induced by the rigid coupling
    matrix according to the relation:

        collider_velocity += rigidity_mat @ (collider_velocity - prev_collider_velocity)

    Then saves the updated collider velocity into ``prev_collider_velocity``
    for the next iteration.

    Args:
        rigidity_mat: Block-sparse rigidity operator (D + J @ IJtm) returned by
            ``build_rigidity_operator``.
        prev_collider_velocity: Velocity vector from the previous iteration
            (updated in place at the end of the call).
        collider_velocity: Current collider velocity vector to be corrected in place.
    """

    D, J, IJtm = rigidity_operator

    # compute velocity delta, store in prev_collider_velocity
    array_axpy(
        y=prev_collider_velocity,
        x=collider_velocity,
        alpha=1.0,
        beta=-1.0,
    )
    delta_velocity = prev_collider_velocity

    sp.bsr_mv(D, x=delta_velocity, y=collider_velocity, alpha=1.0, beta=1.0)
    sp.bsr_mv(IJtm, x=delta_velocity, y=delta_body_qd, alpha=1.0, beta=0.0)
    sp.bsr_mv(J, x=delta_body_qd, y=collider_velocity, alpha=1.0, beta=1.0)

    # save for next iterations
    wp.copy(dest=prev_collider_velocity, src=collider_velocity)


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


def solve_rheology(
    max_iterations: int,
    tolerance: float,
    strain_mat: sp.BsrMatrix,
    transposed_strain_mat: sp.BsrMatrix,
    compliance_mat: sp.BsrMatrix,
    inv_volume,
    node_volume,
    yield_params,
    unilateral_strain_offset,
    strain_rhs,
    plastic_strain_delta,
    stress,
    velocity,
    collider_mat: sp.BsrMatrix,
    transposed_collider_mat: sp.BsrMatrix,
    collider_friction,
    collider_adhesion,
    collider_normals,
    collider_velocities,
    collider_inv_mass,
    collider_impulse,
    color_offsets,
    color_indices: wp.array | None = None,
    color_nodes_per_element: int = 1,
    rigidity_operator: tuple[sp.BsrMatrix, sp.BsrMatrix, sp.BsrMatrix] | None = None,
    temporary_store: fem.TemporaryStore | None = None,
    use_graph=True,
    verbose=wp.config.verbose,
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
        max_iterations: Maximum number of nonlinear iterations.
        tolerance: Solver tolerance for the stress residual (L2 norm).
        strain_mat: Strain-to-velocity block-sparse matrix (B).
        transposed_strain_mat: BSR matrix container for assembling B^T. Used for Jacobi only.
        compliance_mat: Compliance matrix for elastic materials.
        inv_volume: Per-velocity-node inverse mass (or volume scaling) used in
            the solver updates.
        node_volume: Per-strain-node particle volume measure.
        yield_params: Yield parameters per strain node.
        unilateral_strain_offset: Per-node offset enforcing unilateral
            incompressibility (void fraction/critical fraction).
        strain_rhs: Initial strain right-hand side; becomes elastic strain at
            the end of the solve.
        plastic_strain_delta: Output plastic strain increment per node.
        stress: In/out stress variable per node (rotated internally).
        velocity: Grid velocity degrees of freedom to be updated in place.
        collider_friction: Per-velocity-node collider friction coefficients; <0
            disables contact at that node.
        collider_adhesion: Per-velocity-node adhesion coefficients, same unit as impulse (N.s / V0) .
        collider_normals: Per-velocity-node contact normals.
        collider_velocities: Per-velocity-node collider rigid velocities.
        collider_inv_mass: Per-velocity-node collider inverse masses.
        collider_impulse: In/out stored collider impulses (warm start, N.s/V0).
        color_offsets: Optional coloring offsets for Gauss-Seidel.
        color_indices: Optional coloring indices for Gauss-Seidel.
        color_nodes_per_element: Number of nodes per colored element.
        rigidity_operator: Optional rigidity operator coupling nodes to collider DOFs.
        temporary_store: Temporary storage arena for intermediate arrays.
        use_graph: If True, uses conditional CUDA graphs for the iteration loop.
        verbose: If True, prints residuals/iteration counts.

    Returns:
        A captured execution graph handle when ``use_graph`` is True and the
        device supports it; otherwise ``None``.
    """
    borrowed_temporaries: list[Any] = []

    def _register_temp(temp):
        borrowed_temporaries.append(temp)
        return temp

    delta_stress = _register_temp(fem.borrow_temporary_like(stress, temporary_store))

    delassus_rotation = _register_temp(fem.borrow_temporary(temporary_store, shape=stress.shape, dtype=mat66))
    delassus_diagonal = _register_temp(fem.borrow_temporary(temporary_store, shape=stress.shape, dtype=vec6))
    delassus_normal = _register_temp(fem.borrow_temporary(temporary_store, shape=stress.shape, dtype=vec6))

    # If coloring is provided, use Gauss-Seidel, otherwise Jacobi with mass splitting
    color_count = 0 if color_offsets is None else len(color_offsets) - 1
    gs = color_count > 0
    split_mass = not gs

    # Build transposed matrix
    # Do it now as we need offsets to build the Delassus operator
    if not gs:
        sp.bsr_set_transpose(dest=transposed_strain_mat, src=strain_mat)

    if compliance_mat.nnz == 0:
        compliance_mat_diagonal = None
    else:
        compliance_mat_diagonal = _register_temp(fem.borrow_temporary(temporary_store, shape=stress.shape, dtype=mat66))
        # TODO: Remove .zero_() workaround once Newton uses warp_lang-1.12.0.dev20260113 or newer.
        compliance_mat_diagonal.zero_()
        sp.bsr_get_diag(compliance_mat, out=compliance_mat_diagonal)

    # Project initial stress on yield surface
    wp.launch(
        kernel=project_initial_stress,
        dim=stress.shape[0],
        inputs=[
            stress,
            yield_params,
        ],
    )

    # Compute and factorize diagonal blacks, rotate strain matrix to diagonal basis
    # NOTE: we reuse the same memory for local versions of the variables
    local_strain_mat_values = strain_mat.values
    local_compliance_mat_values = compliance_mat.values
    local_strain_rhs = strain_rhs
    local_stress = stress

    wp.launch(
        kernel=compute_delassus_diagonal,
        dim=stress.shape[0],
        inputs=[
            split_mass,
            strain_mat.offsets,
            strain_mat.columns,
            strain_mat.values,
            inv_volume,
            compliance_mat_diagonal,
            transposed_strain_mat.offsets,
            strain_rhs,
            stress,
        ],
        outputs=[
            delassus_rotation,
            delassus_diagonal,
            delassus_normal,
            local_strain_mat_values,
            local_strain_rhs,
            local_stress,
        ],
    )

    if compliance_mat is not None:
        wp.launch(
            kernel=rotate_and_scale_compliance_mat,
            dim=stress.shape[0],
            inputs=[
                compliance_mat.offsets,
                compliance_mat.columns,
                compliance_mat.values,
                delassus_rotation,
                delassus_diagonal,
            ],
        )

    if gs:
        device = velocity.device
        if device.is_cuda:
            color_block_count = device.sm_count * 2
        else:
            color_block_count = 1
        color_block_dim = 64
        color_launch_dim = color_block_count * color_block_dim

        apply_stress_launch = wp.launch(
            kernel=apply_stress_gs,
            dim=color_launch_dim,
            inputs=[
                0,  # color
                color_launch_dim,
                color_nodes_per_element,
                color_offsets,
                color_indices,
                strain_mat.offsets,
                strain_mat.columns,
                local_strain_mat_values,
                delassus_diagonal,
                inv_volume,
                local_stress,
            ],
            outputs=[
                velocity,
            ],
            block_dim=color_block_dim,
            max_blocks=color_block_count,
            record_cmd=True,
        )

        # Apply initial guess
        for color in range(color_count):
            apply_stress_launch.set_param_at_index(0, color)
            apply_stress_launch.launch()

        # Solve kernel
        solve_local_launch = wp.launch(
            kernel=solve_local_stress_gs,
            dim=color_launch_dim,
            inputs=[
                0,  # color
                color_launch_dim,
                color_nodes_per_element,
                color_offsets,
                color_indices,
                yield_params,
                compliance_mat.offsets,
                compliance_mat.columns,
                local_compliance_mat_values,
                strain_mat.offsets,
                strain_mat.columns,
                strain_mat.values,
                delassus_diagonal,
                delassus_normal,
                inv_volume,
                strain_rhs,
                unilateral_strain_offset,
            ],
            outputs=[
                velocity,
                local_stress,
                delta_stress,
            ],
            block_dim=color_block_dim,
            max_blocks=color_block_count,
            record_cmd=True,
        )

    else:
        # Apply local scaling and rotations to transposed strain matrix
        wp.launch(
            kernel=rotate_and_scale_transposed_strain_mat,
            dim=inv_volume.shape[0],
            inputs=[
                transposed_strain_mat.offsets,
                transposed_strain_mat.columns,
                transposed_strain_mat.values,
                inv_volume,
                delassus_rotation,
                delassus_diagonal,
            ],
        )

        # Apply initial guess
        sp.bsr_mv(A=transposed_strain_mat, x=stress, y=velocity, alpha=1.0, beta=1.0)

        # Solve kernel
        solve_local_launch = wp.launch(
            kernel=solve_local_stress_jacobi,
            dim=stress.shape[0],
            inputs=[
                yield_params,
                compliance_mat.offsets,
                compliance_mat.columns,
                local_compliance_mat_values,
                strain_mat.offsets,
                strain_mat.columns,
                local_strain_mat_values,
                delassus_diagonal,
                delassus_normal,
                local_strain_rhs,
                unilateral_strain_offset,
                velocity,
                local_stress,
            ],
            outputs=[
                delta_stress,
            ],
            record_cmd=True,
        )

    # Setup rigidity correction
    if rigidity_operator is not None:
        prev_collider_velocity = _register_temp(fem.borrow_temporary_like(collider_velocities, temporary_store))
        prev_collider_velocity.assign(collider_velocities)

        _D, J, _IJtm = rigidity_operator
        delta_body_qd = _register_temp(fem.borrow_temporary(temporary_store, shape=J.shape[1], dtype=float))

    # Collider contacts
    subgrid_collisions = collider_mat.nnz > 0
    if subgrid_collisions:
        sp.bsr_set_transpose(dest=transposed_collider_mat, src=collider_mat)

        collider_delassus_diagonal = _register_temp(fem.borrow_temporary_like(collider_inv_mass, temporary_store))
        wp.launch(
            compute_collider_delassus_diagonal,
            dim=collider_impulse.shape[0],
            inputs=[
                collider_mat.offsets,
                collider_mat.columns,
                collider_mat.values,
                collider_inv_mass,
                transposed_collider_mat.offsets,
                inv_volume,
            ],
            outputs=[
                collider_delassus_diagonal,
            ],
        )

        # define solve operation
        delta_impulse = _register_temp(fem.borrow_temporary_like(collider_impulse, temporary_store))
        apply_collider_impulse_launch = wp.launch(
            apply_subgrid_impulse,
            dim=velocity.shape[0],
            inputs=[
                transposed_collider_mat.offsets,
                transposed_collider_mat.columns,
                transposed_collider_mat.values,
                inv_volume,
                delta_impulse,
                velocity,
            ],
            record_cmd=True,
        )

        solve_collider_launch = wp.launch(
            kernel=solve_subgrid_friction,
            dim=collider_impulse.shape[0],
            inputs=[
                velocity,
                collider_mat.offsets,
                collider_mat.columns,
                collider_mat.values,
                collider_friction,
                collider_adhesion,
                collider_normals,
                collider_inv_mass,
                collider_delassus_diagonal,
                collider_velocities,
                collider_impulse,
                delta_impulse,
            ],
            record_cmd=True,
        )

        def solve_collider():
            solve_collider_launch.launch()
            apply_collider_impulse_launch.launch()

        # Apply initial impulse guess
        wp.launch(
            apply_subgrid_impulse_warmstart,
            dim=delta_impulse.shape[0],
            inputs=[
                collider_inv_mass,
                collider_friction,
                collider_impulse,
                delta_impulse,
                collider_velocities,
            ],
        )
        apply_collider_impulse_launch.launch()

    else:
        # define solve operation
        solve_collider_launch = wp.launch(
            kernel=solve_nodal_friction,
            dim=collider_impulse.shape[0],
            inputs=[
                inv_volume,
                collider_friction,
                collider_adhesion,
                collider_normals,
                collider_inv_mass,
                velocity,
                collider_velocities,
                collider_impulse,
            ],
            record_cmd=True,
        )

        def solve_collider():
            solve_collider_launch.launch()

        # Apply initial impulse guess
        wp.launch(
            kernel=apply_nodal_impulse,
            dim=collider_impulse.shape[0],
            inputs=[
                collider_impulse,
                collider_friction,
                inv_volume,
                collider_inv_mass,
                velocity,
                collider_velocities,
            ],
        )

    # Apply rigidity correction
    if rigidity_operator is not None:
        apply_rigidity_operator(rigidity_operator, prev_collider_velocity, collider_velocities, delta_body_qd)

    def do_iteration():
        # solve contacts
        solve_collider()
        if rigidity_operator is not None:
            apply_rigidity_operator(rigidity_operator, prev_collider_velocity, collider_velocities, delta_body_qd)

        # solve stress
        if gs:
            for color in range(color_count):
                solve_local_launch.set_param_at_index(0, color)
                solve_local_launch.launch()
        else:
            solve_local_launch.launch()
            # Add jacobi delta
            sp.bsr_mv(
                A=transposed_strain_mat,
                x=delta_stress,
                y=velocity,
                alpha=1.0,
                beta=1.0,
            )
            array_axpy(x=delta_stress, y=local_stress, alpha=1.0, beta=1.0)

    # Run solver loop

    residual_scale = 1 + stress.shape[0]

    # Utility to compute the squared norm of the residual
    residual_squared_norm_computer = ArraySquaredNorm(
        max_length=delta_stress.shape[0] * 6,
        device=delta_stress.device,
        temporary_store=temporary_store,
    )

    solve_graph = None
    if use_graph:
        min_iterations = 5
        iteration_and_condition = _register_temp(fem.borrow_temporary(temporary_store, shape=(2,), dtype=int))
        iteration_and_condition.fill_(1)

        iteration = iteration_and_condition[:1]
        condition = iteration_and_condition[1:]

        def do_iteration_with_condition():
            do_iteration()
            residual = residual_squared_norm_computer.compute_squared_norm(delta_stress)
            wp.launch(
                update_condition,
                dim=1,
                inputs=[
                    tolerance * residual_scale,
                    min_iterations,
                    max_iterations,
                    residual,
                    iteration,
                    condition,
                ],
            )

        device = delta_stress.device
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
                residual = residual_squared_norm_computer.compute_squared_norm(delta_stress)
                res = math.sqrt(residual.numpy()[0]) / residual_scale
                print(
                    f"{'Gauss-Seidel' if gs else 'Jacobi'} terminated after "
                    f"{iteration_and_condition.numpy()[0]} iterations with residual {res}"
                )
    else:
        solve_granularity = 25 if gs else 50

        for batch in range(max_iterations // solve_granularity):
            for _k in range(solve_granularity):
                do_iteration()

            residual = residual_squared_norm_computer.compute_squared_norm(delta_stress)
            res = math.sqrt(residual.numpy()[0]) / residual_scale

            if verbose:
                print(
                    f"{'Gauss-Seidel' if gs else 'Jacobi'} iterations #{(batch + 1) * solve_granularity} \t res(l2)={res}"
                )
            if res < tolerance:
                break

    # Convert stress back to world space,
    # and compute final elastic strain

    delta_stress.assign(local_stress)
    wp.launch(
        kernel=postprocess_stress_and_strain,
        dim=stress.shape[0],
        inputs=[
            delassus_rotation,
            delassus_diagonal,
            compliance_mat.offsets,
            compliance_mat.columns,
            local_compliance_mat_values,
            strain_mat.offsets,
            strain_mat.columns,
            local_strain_mat_values,
            node_volume,
            delta_stress,
            velocity,
        ],
        outputs=[
            stress,
            strain_rhs,
            plastic_strain_delta,
        ],
    )

    residual_squared_norm_computer.release()

    for temp in borrowed_temporaries:
        temp.release()

    return solve_graph
