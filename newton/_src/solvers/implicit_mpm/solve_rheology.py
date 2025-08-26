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

"""Common definitions for types and constants."""

import gc
import math
from typing import Any

import warp as wp
import warp.fem as fem
import warp.sparse as sp
from warp.fem.utils import symmetric_eigenvalues_qr

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
    @wp.func
    def from_values(friction_coeff: float, yield_pressure: float, tensile_yield_ratio: float, yield_stress: float):
        tangential_scale = wp.sqrt(3.0 / 2.0)
        return YieldParamVec(
            yield_pressure,
            tensile_yield_ratio * yield_pressure,
            yield_stress * tangential_scale,
            friction_coeff * yield_pressure * tangential_scale,
        )


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
    else:
        diag_block = mat66(0.0)

    mass_ratio = float(1.0)
    for b in range(block_beg, block_end):
        u_i = strain_mat_columns[b]

        if split_mass:
            mass_ratio = float(transposed_strain_mat_offsets[u_i + 1] - transposed_strain_mat_offsets[u_i])

        b_val = strain_mat_values[b]
        inv_frac = inv_volume[u_i] * mass_ratio

        diag_block += (b_val * inv_frac) @ wp.transpose(b_val)

    diag_block += _DELASSUS_PROXIMAL_REG * wp.identity(n=6, dtype=float)
    # diag_block += wp.clamp(1.0e-1 * wp.trace(diag_block), 1.0e-7, 1.0e-4) * wp.identity(n=6, dtype=float)

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
    tau_i = wp.tid()

    yield_params = yield_stress[tau_i]

    sig = stress[tau_i]

    # FIXME find a more focused way to do this
    if yield_params[1] > 0.0:
        sig = vec6(0.0)

    nor = vec6(1.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    stress[tau_i] = project_stress(sig, nor, yield_params)


@wp.kernel
def rotate_compliance_mat(
    compliance_mat_offsets: wp.array(dtype=int),
    compliance_mat_columns: wp.array(dtype=int),
    compliance_mat_values: wp.array(dtype=mat66),
    delassus_rotation: wp.array(dtype=mat66),
    delassus_diagonal: wp.array(dtype=vec6),
):
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
def scale_transposed_strain_mat(
    tr_strain_mat_offsets: wp.array(dtype=int),
    tr_strain_mat_columns: wp.array(dtype=int),
    tr_strain_mat_values: wp.array(dtype=mat36),
    inv_volume: wp.array(dtype=float),
    delassus_rotation: wp.array(dtype=mat66),
):
    """
    Scales the values of the transposed strain matrix (B^T).

    For each particle (u_i), this kernel iterates through its contributions
    to constraints (tau_i) in the transposed strain matrix.
    It scales the matrix entries (tr_strain_mat_values) by the particle's
    inverse volume and applies the Delassus rotation associated with the
    constraint. This prepares B^T for matrix-vector products in the
    solver iterations.
    """
    u_i = wp.tid()
    block_beg = tr_strain_mat_offsets[u_i]
    block_end = tr_strain_mat_offsets[u_i + 1]

    for b in range(block_beg, block_end):
        tau_i = tr_strain_mat_columns[b]

        tr_strain_mat_values[b] = inv_volume[u_i] * tr_strain_mat_values[b] @ delassus_rotation[tau_i]


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
    Transforms stress and computes elastic strain back to the original basis
    after the solver iterations.

    For each constraint (i):
    1. Retrieves the Delassus rotation and diagonal values.
    2. Computes the local elastic strain based on the current stress and the
       stress-strain matrix in the rotated frame.
    3. Rotates the final stress back to the original coordinate system.
    4. Rotates and scales the local elastic strain back to the original system
       and stores it in `elastic_strain` (which was `strain_rhs` before solver).
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
    of the frictional contact model."""

    if yield_stress <= 0.0:
        return b_T

    # Viscous shear opposite to tangential stress, zero divergence
    # find alpha, r_t,  mu_rn, (D + alpha/(mu r_n) I) r_t + b_t = 0, |r_t| = mu r_n
    # find alpha,  |(D mu r_n + alpha I)^{-1} b_t|^2 = 1.0

    mu_rn = yield_stress
    Dmu_rn = D * mu_rn

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
def normal_yield_bounds(yield_params: YieldParamVec):
    return -yield_params[1], yield_params[0]


@wp.func
def shear_yield_stress(yield_params: YieldParamVec, r_N: float):
    p_min, p_max = normal_yield_bounds(yield_params)

    mu = wp.where(p_max > 0.0, yield_params[3] / p_max, 0.0)
    s = yield_params[2]
    return s + wp.min(0.5 * yield_params[3], mu * wp.min(r_N - p_min, p_max - r_N))


@wp.func
def solve_coulomb_aniso(
    D: vec6,
    b: vec6,
    nor: vec6,
    off: float,
    yield_params: YieldParamVec,
):
    # Note: this assumes that D.nor = lambda nor
    # i.e. nor should be along one canonical axis
    # (solve_sliding aniso would get a lot more complex otherwise as normal and tangential
    # responses become interlinked)

    b_N = wp.dot(b, nor)

    r_0 = -wp.cw_div(b, D)
    r_N0 = wp.dot(r_0, nor)
    r_N_min, r_N_max = normal_yield_bounds(yield_params)

    r_N = wp.where(b_N > 0.0, wp.max(r_N0, r_N_min), wp.clamp(r_N0 - off / D[0], 0.0, r_N_max))

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
    nor = delassus_normal[tau_i]
    cur_stress = local_stress[tau_i]

    tau_new = solve_coulomb_aniso(
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

    This kernel iterates over each constraint (tau_i) and calls the
    `solve_local_stress` function. It uses the `delassus_diagonal`
    as the D matrix for the local solve. The result (delta_correction)
    represents the change in stress required to satisfy the local
    constitutive model. In a Jacobi scheme, these corrections are typically
    accumulated and then applied globally.
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
def apply_stress_delta_gs(
    tau_i: int,
    D: vec6,
    delta_stress: vec6,
    strain_mat_offsets: wp.array(dtype=int),
    strain_mat_columns: wp.array(dtype=int),
    local_strain_mat_values: wp.array(dtype=mat63),
    inv_mass_matrix: wp.array(dtype=float),
    velocities: wp.array(dtype=wp.vec3),
):
    block_beg = strain_mat_offsets[tau_i]
    block_end = strain_mat_offsets[tau_i + 1]

    for b in range(block_beg, block_end):
        u_i = strain_mat_columns[b]
        delta_vel = inv_mass_matrix[u_i] * wp.cw_div(delta_stress, D) @ local_strain_mat_values[b]
        velocities[u_i] += delta_vel


@wp.kernel
def apply_stress_gs(
    color_offset: int,
    color_nodes_per_element: int,
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
    Applies the current stress to update particle velocities in a Gauss-Seidel manner,
    typically used for an initial guess or applying accumulated stress.

    This kernel processes a batch of constraints defined by `color_indices`
    and `color_offset`. For each constraint `tau_i` in the batch:
    1. Retrieves the Delassus diagonal `D`.
    2. Calls `apply_stress_delta_gs` with the current `stress[tau_i]` as the
       delta_stress. This effectively applies the full current stress to update
       velocities of particles connected to this constraint.
    """

    base_index = color_indices[wp.tid() + color_offset]
    for k in range(color_nodes_per_element):
        tau_i = base_index * color_nodes_per_element + k

        D = delassus_diagonal[tau_i]
        cur_stress = local_stress[tau_i]

        apply_stress_delta_gs(
            tau_i,
            D,
            cur_stress,
            strain_mat_offsets,
            strain_mat_columns,
            local_strain_mat_values,
            inv_mass_matrix,
            velocities,
        )


@wp.kernel
def solve_local_stress_gs(
    color_offset: int,
    color_nodes_per_element: int,
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
    Solves the local stress problem and immediately applies the resulting stress
    delta to particle velocities in a Gauss-Seidel fashion.

    This kernel processes a batch of constraints defined by `color_indices`
    and `color_offset`. For each constraint `tau_i` in the batch:
    1. Retrieves the Delassus diagonal `D`.
    2. Calls `solve_local_stress` to compute the `delta_correction` for `stress[tau_i]`.
       This function updates `stress[tau_i]` and `delta_correction[tau_i]`.
    3. Calls `apply_stress_delta_gs` to immediately propagate the effect of
       `delta_correction[tau_i]` to the velocities of connected particles.
    """
    base_index = color_indices[wp.tid() + color_offset]
    for k in range(color_nodes_per_element):
        tau_i = base_index * color_nodes_per_element + k

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

        delta_stress = solve_local_stress(
            tau_i,
            D,
            local_strain,
            yield_params,
            delassus_normal,
            unilateral_strain_offset,
            local_stress,
        )

        apply_stress_delta_gs(
            tau_i,
            D,
            delta_stress,
            strain_mat_offsets,
            strain_mat_columns,
            local_strain_mat_values,
            inv_mass_matrix,
            velocities,
        )

        local_stress[tau_i] += delta_stress
        delta_correction[tau_i] = delta_stress  # for residual evaluation


@wp.kernel
def apply_collider_impulse(
    collider_impulse: wp.array(dtype=wp.vec3),
    collider_friction: wp.array(dtype=float),
    inv_mass: wp.array(dtype=float),
    collider_inv_mass: wp.array(dtype=float),
    velocities: wp.array(dtype=wp.vec3),
    collider_velocities: wp.array(dtype=wp.vec3),
):
    """
    Applies pre-computed impulses to particles and colliders.

    For each particle/collider pair (i):
    1. Updates the particle's velocity based on its inverse mass and the impulse.
    2. Updates the collider's velocity based on its inverse mass and the negative
       of the impulse (action-reaction).
    This is typically used to apply an initial guess for contact impulses or
    to apply accumulated impulses from a solver.
    """
    i = wp.tid()

    friction_coeff = collider_friction[i]
    if friction_coeff < 0.0:
        collider_impulse[i] = wp.vec3(0.0)
    else:
        velocities[i] += inv_mass[i] * collider_impulse[i]
        collider_velocities[i] -= collider_inv_mass[i] * collider_impulse[i]


@wp.func
def solve_coulomb_isotropic(
    mu: float,
    nor: wp.vec3,
    u: wp.vec3,
):
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


class ArraySquaredNorm:
    def __init__(self, max_length: int, tile_size=512, device=None, temporary_store=None):
        self.tile_size = tile_size
        self.device = device

        num_blocks = (max_length + self.tile_size - 1) // self.tile_size
        self.partial_sums_a = fem.borrow_temporary(
            temporary_store, shape=(num_blocks,), dtype=float, device=self.device
        )
        self.partial_sums_b = fem.borrow_temporary(
            temporary_store, shape=(num_blocks,), dtype=float, device=self.device
        )

        self.sum_squared_kernel = self._create_block_sum_kernel(square_input=True)
        self.sum_kernel = self._create_block_sum_kernel(square_input=False)

    # Result contains a single value, the sum of the array (will get updated by this function)
    def compute_squared_norm(self, data: wp.array(dtype=Any)):
        # cast vector types to float
        if data.dtype != float:
            data = wp.array(data, dtype=float).flatten()

        kernel = self.sum_squared_kernel
        array_length = data.shape[0]

        flip_flop = False
        while True:
            num_blocks = (array_length + self.tile_size - 1) // self.tile_size

            partial_sums = (self.partial_sums_a if flip_flop else self.partial_sums_b).array[:num_blocks]

            wp.launch_tiled(
                kernel,
                dim=num_blocks,
                inputs=(data,),
                outputs=(partial_sums,),
                block_dim=self.tile_size,
            )

            array_length = num_blocks
            data = partial_sums
            kernel = self.sum_kernel

            flip_flop = not flip_flop

            if num_blocks == 1:
                break

        return data[:1]

    def _create_block_sum_kernel(self, square_input):
        tile_size = self.tile_size

        @fem.cache.dynamic_kernel(suffix=f"{tile_size}{square_input}")
        def block_sum_kernel(
            data: wp.array(dtype=float, ndim=1),
            partial_sums: wp.array(dtype=float),
        ):
            block_id, tid_block = wp.tid()
            start = block_id * tile_size

            t = wp.tile_load(data, shape=tile_size, offset=start)

            if wp.static(square_input):
                t = wp.tile_map(wp.mul, t, t)

            tile_sum = wp.tile_sum(t)
            if tid_block == 0:
                partial_sums[block_id] = tile_sum[0]

        return block_sum_kernel


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
    stop = (wp.sqrt(residual[0]) < residual_threshold and cur_it >= min_iterations) or cur_it >= max_iterations

    iteration[0] = cur_it
    condition[0] = wp.where(stop, 0, 1)


def apply_rigidity_matrix(rigidity_mat, prev_collider_velocity, collider_velocity):
    """Apply rigidity matrix to the collider velocity delta

    collider_velocity += rigidity_mat * (collider_velocity - prev_collider_velocity)
    """
    # velocity delta
    fem.utils.array_axpy(
        y=prev_collider_velocity,
        x=collider_velocity,
        alpha=1.0,
        beta=-1.0,
    )
    # rigidity contribution to new velocity
    sp.bsr_mv(
        A=rigidity_mat,
        x=prev_collider_velocity,
        y=collider_velocity,
        alpha=1.0,
        beta=1.0,
    )
    # save for next iterations
    wp.copy(dest=prev_collider_velocity, src=collider_velocity)


def solve_rheology(
    max_iterations: int,
    tolerance: float,
    strain_mat: sp.BsrMatrix,
    transposed_strain_mat: sp.BsrMatrix | None,
    compliance_mat: sp.BsrMatrix | None,
    inv_volume,
    node_volume,
    yield_params,
    unilateral_strain_offset,
    strain_rhs,
    plastic_strain_delta,
    stress,
    velocity,
    collider_friction,
    collider_adhesion,
    collider_normals,
    collider_velocities,
    collider_inv_mass,
    collider_impulse,
    color_offsets,
    color_indices: wp.array | None = None,
    color_nodes_per_element: int = 1,
    rigidity_mat: sp.BsrMatrix | None = None,
    temporary_store: fem.TemporaryStore | None = None,
    use_graph=True,
    verbose=wp.config.verbose,
):
    delta_stress = fem.borrow_temporary_like(stress, temporary_store)

    delassus_rotation = fem.borrow_temporary(temporary_store, shape=stress.shape, dtype=mat66)
    delassus_diagonal = fem.borrow_temporary(temporary_store, shape=stress.shape, dtype=vec6)
    delassus_normal = fem.borrow_temporary(temporary_store, shape=stress.shape, dtype=vec6)

    # If coloring is provided, use Gauss-Seidel, otherwise Jacobi with mass splitting
    color_count = 0 if color_offsets is None else len(color_offsets) - 1
    gs = color_count > 0
    split_mass = not gs

    # Build transposed matrix
    # Do it now as we need offsets to build the Delassus operator
    if not gs:
        sp.bsr_set_transpose(dest=transposed_strain_mat, src=strain_mat)

    compliance_mat_diagonal = sp.bsr_get_diag(compliance_mat) if compliance_mat else None
    if compliance_mat is None:
        compliance_mat = sp.bsr_zeros(strain_mat.nrow, strain_mat.nrow, mat66)

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
            delassus_rotation.array,
            delassus_diagonal.array,
            delassus_normal.array,
            local_strain_mat_values,
            local_strain_rhs,
            local_stress,
        ],
    )

    if compliance_mat is not None:
        wp.launch(
            kernel=rotate_compliance_mat,
            dim=compliance_mat.nrow,
            inputs=[
                compliance_mat.offsets,
                compliance_mat.columns,
                compliance_mat.values,
                delassus_rotation.array,
                delassus_diagonal.array,
            ],
        )

    if gs:
        apply_stress_launch = wp.launch(
            kernel=apply_stress_gs,
            dim=1,
            inputs=[
                0,  # color offset
                color_nodes_per_element,
                color_indices,
                strain_mat.offsets,
                strain_mat.columns,
                local_strain_mat_values,
                delassus_diagonal.array,
                inv_volume,
                local_stress,
            ],
            outputs=[
                velocity,
            ],
            block_dim=64,
            record_cmd=True,
        )

        # Apply initial guess
        for k in range(color_count):
            apply_stress_launch.set_param_at_index(0, color_offsets[k])
            apply_stress_launch.set_dim((int(color_offsets[k + 1] - color_offsets[k]),))
            apply_stress_launch.launch()

        # Solve kernel
        solve_local_launch = wp.launch(
            kernel=solve_local_stress_gs,
            dim=1,
            inputs=[
                0,  # color offset
                color_nodes_per_element,
                color_indices,
                yield_params,
                compliance_mat.offsets,
                compliance_mat.columns,
                local_compliance_mat_values,
                strain_mat.offsets,
                strain_mat.columns,
                strain_mat.values,
                delassus_diagonal.array,
                delassus_normal.array,
                inv_volume,
                strain_rhs,
                unilateral_strain_offset,
            ],
            outputs=[
                velocity,
                local_stress,
                delta_stress.array,
            ],
            block_dim=64,
            record_cmd=True,
        )

    else:
        # Apply local scaling and rotations to transposed strain matrix
        wp.launch(
            kernel=scale_transposed_strain_mat,
            dim=inv_volume.shape[0],
            inputs=[
                transposed_strain_mat.offsets,
                transposed_strain_mat.columns,
                transposed_strain_mat.values,
                inv_volume,
                delassus_rotation.array,
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
                delassus_diagonal.array,
                delassus_normal.array,
                local_strain_rhs,
                unilateral_strain_offset,
                velocity,
                local_stress,
            ],
            outputs=[
                delta_stress.array,
            ],
            record_cmd=True,
        )

    # Collider contacts

    if rigidity_mat is None:
        prev_collider_velocity = fem.borrow_temporary_like(collider_velocities, temporary_store)
        wp.copy(dest=prev_collider_velocity.array, src=collider_velocities)

    # Apply initial impulse guess
    wp.launch(
        kernel=apply_collider_impulse,
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
    if rigidity_mat is not None:
        apply_rigidity_matrix(rigidity_mat, prev_collider_velocity.array, collider_velocities)

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

    def do_iteration():
        # solve contacts
        solve_collider_launch.launch()
        if rigidity_mat is not None:
            apply_rigidity_matrix(rigidity_mat, prev_collider_velocity.array, collider_velocities)

        # solve stress
        if gs:
            for k in range(color_count):
                solve_local_launch.set_param_at_index(0, color_offsets[k])
                solve_local_launch.set_dim((int(color_offsets[k + 1] - color_offsets[k]),))
                solve_local_launch.launch()
        else:
            solve_local_launch.launch()
            # Add jacobi delta
            sp.bsr_mv(
                A=transposed_strain_mat,
                x=delta_stress.array,
                y=velocity,
                alpha=1.0,
                beta=1.0,
            )
            fem.utils.array_axpy(x=delta_stress.array, y=local_stress, alpha=1.0, beta=1.0)

    # Run solver loop

    residual_scale = 1 + stress.shape[0]

    # Utility to compute the squared norm of the residual
    residual_squared_norm_computer = ArraySquaredNorm(
        max_length=delta_stress.array.shape[0] * 6,
        device=delta_stress.array.device,
        temporary_store=temporary_store,
    )

    if use_graph:
        min_iterations = 5
        iteration_and_condition = fem.borrow_temporary(temporary_store, shape=(2,), dtype=int)

        gc.disable()
        with wp.ScopedCapture(force_module_load=False) as iteration_capture:
            do_iteration()
            residual = residual_squared_norm_computer.compute_squared_norm(delta_stress.array)
            wp.launch(
                update_condition,
                dim=1,
                inputs=[
                    tolerance * residual_scale,
                    min_iterations,
                    max_iterations,
                    residual,
                    iteration_and_condition.array[:1],
                    iteration_and_condition.array[1:],
                ],
            )
        iteration_graph = iteration_capture.graph

        with wp.ScopedCapture(force_module_load=False) as capture:
            wp.capture_while(
                condition=iteration_and_condition.array[1:],
                while_body=iteration_graph,
            )
        solve_graph = capture.graph
        gc.enable()

        iteration_and_condition.array.assign([0, 1])
        wp.capture_launch(solve_graph)

        if verbose:
            res = math.sqrt(residual.numpy()[0]) / residual_scale
            print(
                f"{'Gauss-Seidel' if gs else 'Jacobi'} terminated after {iteration_and_condition.array.numpy()[0]} iterations with residual {res}"
            )
    else:
        solve_graph = None
        solve_granularity = 25 if gs else 50

        for batch in range(max_iterations // solve_granularity):
            for _k in range(solve_granularity):
                do_iteration()

            residual = residual_squared_norm_computer.compute_squared_norm(delta_stress.array)
            res = math.sqrt(residual.numpy()[0]) / residual_scale

            if verbose:
                print(
                    f"{'Gauss-Seidel' if gs else 'Jacobi'} iterations #{(batch + 1) * solve_granularity} \t res(l2)={res}"
                )
            if res < tolerance:
                break

    # Convert stress back to world space,
    # and compute final elastic strain

    delta_stress.array.assign(local_stress)
    wp.launch(
        kernel=postprocess_stress_and_strain,
        dim=stress.shape[0],
        inputs=[
            delassus_rotation.array,
            delassus_diagonal.array,
            compliance_mat.offsets,
            compliance_mat.columns,
            local_compliance_mat_values,
            strain_mat.offsets,
            strain_mat.columns,
            local_strain_mat_values,
            node_volume,
            delta_stress.array,
            velocity,
        ],
        outputs=[
            stress,
            strain_rhs,
            plastic_strain_delta,
        ],
    )

    return solve_graph
