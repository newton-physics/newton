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
from typing import Any, Optional

import warp as wp
import warp.fem as fem
import warp.sparse as sp
from warp.fem.utils import symmetric_eigenvalues_qr

_DELASSUS_DIAG_CUTOFF = wp.constant(1.0e-6)
"""Cutoff for the trace of the diagonal block of the Delassus operator to disable constraints"""

vec6 = wp.types.vector(length=6, dtype=wp.float32)
mat66 = wp.types.matrix(shape=(6, 6), dtype=wp.float32)
mat63 = wp.types.matrix(shape=(6, 3), dtype=wp.float32)
mat36 = wp.types.matrix(shape=(3, 6), dtype=wp.float32)

wp.set_module_options({"enable_backward": False})


@wp.kernel
def compute_delassus_diagonal(
    split_mass: wp.bool,
    strain_mat_offsets: wp.array(dtype=int),
    strain_mat_columns: wp.array(dtype=int),
    strain_mat_values: wp.array(dtype=mat63),
    inv_volume: wp.array(dtype=float),
    stress_strain_matrices: wp.array(dtype=mat66),
    transposed_strain_mat_offsets: wp.array(dtype=int),
    strain_rhs: wp.array(dtype=vec6),
    stress: wp.array(dtype=vec6),
    delassus_rotation: wp.array(dtype=mat66),
    delassus_diagonal: wp.array(dtype=vec6),
    delassus_normal: wp.array(dtype=vec6),
    local_strain_mat_values: wp.array(dtype=mat63),
    local_stress_strain_matrices: wp.array(dtype=mat66),
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

    if stress_strain_matrices:
        diag_block = stress_strain_matrices[tau_i]
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

    if wp.trace(diag_block) > _DELASSUS_DIAG_CUTOFF:
        for k in range(1, 6):
            # Remove shear-divergence coupling
            # (current implementation of solve_coulomb_aniso normal and tangential responses are independent)
            diag_block[0, k] = 0.0
            diag_block[k, 0] = 0.0

        diag, ev = symmetric_eigenvalues_qr(diag_block, 1.0e-12)
        if not (wp.ddot(ev, ev) < 1.0e16 and wp.length_sq(diag) < 1.0e16):
            # wp.print(diag_block)
            # wp.print(diag)
            diag = wp.get_diag(diag_block)
            ev = wp.identity(n=6, dtype=float)

        # Disable null modes -- e.g. from velocity boundary conditions
        for k in range(0, 6):
            if diag[k] < _DELASSUS_DIAG_CUTOFF:
                diag[k] = 1.0
                ev[k] = vec6(0.0)

        delassus_diagonal[tau_i] = diag
        delassus_rotation[tau_i] = wp.transpose(ev) @ wp.diag(wp.cw_div(vec6(1.0), diag))

        # Apply rotation to contact data
        nor = ev * vec6(1.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        delassus_normal[tau_i] = nor

        local_strain_rhs[tau_i] = ev * strain_rhs[tau_i]
        local_stress[tau_i] = wp.cw_mul(ev * stress[tau_i], diag)

        for b in range(block_beg, block_end):
            local_strain_mat_values[b] = ev * strain_mat_values[b]

        if local_stress_strain_matrices:
            local_stress_strain_matrices[tau_i] = ev * stress_strain_matrices[tau_i] * delassus_rotation[tau_i]
    else:
        # Not a valid constraint, disable
        delassus_diagonal[tau_i] = vec6(1.0)
        delassus_normal[tau_i] = vec6(1.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        delassus_rotation[tau_i] = wp.identity(n=6, dtype=float)
        local_stress[tau_i] = vec6(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        local_strain_rhs[tau_i] = vec6(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        if local_stress_strain_matrices:
            local_stress_strain_matrices[tau_i] = mat66(0.0)


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
def postprocess_stress(
    delassus_rotation: wp.array(dtype=mat66),
    delassus_diagonal: wp.array(dtype=vec6),
    local_stress_strain_matrices: wp.array(dtype=mat66),
    local_stress: wp.array(dtype=vec6),
    stress: wp.array(dtype=vec6),
    elastic_strain: wp.array(dtype=vec6),
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
    i = wp.tid()
    rot = delassus_rotation[i]
    diag = delassus_diagonal[i]

    loc_stress = local_stress[i]
    if local_stress_strain_matrices:
        loc_strain = -(local_stress_strain_matrices[i] * loc_stress)
    else:
        loc_strain = vec6(0.0)

    stress[i] = rot * loc_stress
    elastic_strain[i] = rot * wp.cw_mul(loc_strain, diag)


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

    if alpha_max - alpha_min > _DELASSUS_DIAG_CUTOFF:
        for _k in range(24):
            f_cur, df_dalpha = eval_sliding_residual(alpha_cur, Dmu_rn, b_T)

            alpha_next = wp.clamp(alpha_cur - f_cur / df_dalpha, alpha_min, alpha_max)
            alpha_cur = alpha_next

    u_T = wp.cw_div(b_T * alpha_cur, Dmu_rn + vec6(alpha_cur))

    # Sanity checkI
    # r_T_sol = -u_T * mu_rn / alpha_cur
    # err = wp.length(r_T_sol) - mu_rn
    # if err > 1.4:
    #     f_cur, df_dalpha = eval_sliding_residual(alpha_cur, Dmu_rn, b_T)
    #     wp.printf("%d %f %f %f %f %f \n", k, wp.length(r_T_sol), f_cur, mu_rn, alpha_cur / alpha_min, alpha_cur / alpha_max)
    #     #wp.print(D)
    #     #wp.print(b)
    #     #wp.print(mu_dyn)

    return u_T


@wp.func
def solve_coulomb_aniso(
    D: vec6,
    b: vec6,
    nor: vec6,
    unilateral: wp.bool,
    mu_st: float,
    mu_dyn: float,
    yield_stress: wp.vec3,
):
    # Note: this assumes that D.nor = lambda nor
    # i.e. nor should be along one canonical axis
    # (solve_sliding aniso would get a lot more complex otherwise as normal and tangential
    # responses become interlinked)

    # Positive divergence, zero stress
    b_N = wp.dot(b, nor)
    if unilateral and b_N >= 0.0:
        return b

    # Static friction, zero shear
    r_0 = -wp.cw_div(b, D)
    r_N0 = wp.dot(r_0, nor)
    r_T = r_0 - r_N0 * nor

    r_N = wp.clamp(r_N0, yield_stress[1], yield_stress[2])
    u_N = (r_N - r_N0) * wp.cw_mul(nor, D)

    mu_rn = wp.max(mu_st * r_N, yield_stress[0])
    mu_rn_sq = mu_rn * mu_rn
    if mu_rn >= 0 and wp.dot(r_T, r_T) <= mu_rn_sq:
        return u_N

    mu_rn = wp.max(mu_dyn * r_N, yield_stress[0])
    b_T = b - b_N * nor
    return u_N + solve_sliding_aniso(D, b_T, mu_rn)


@wp.func
def solve_local_stress(
    tau_i: int,
    unilateral: wp.bool,
    friction_coeff: float,
    D: vec6,
    yield_stress: wp.array(dtype=wp.vec3),
    local_stress_strain_matrices: wp.array(dtype=mat66),
    strain_mat_offsets: wp.array(dtype=int),
    strain_mat_columns: wp.array(dtype=int),
    local_strain_mat_values: wp.array(dtype=mat63),
    delassus_normal: wp.array(dtype=vec6),
    local_strain_rhs: wp.array(dtype=vec6),
    velocities: wp.array(dtype=wp.vec3),
    local_stress: wp.array(dtype=vec6),
    delta_correction: wp.array(dtype=vec6),
):
    block_beg = strain_mat_offsets[tau_i]
    block_end = strain_mat_offsets[tau_i + 1]

    tau = local_strain_rhs[tau_i]

    for b in range(block_beg, block_end):
        u_i = strain_mat_columns[b]
        tau += local_strain_mat_values[b] * velocities[u_i]

    nor = delassus_normal[tau_i]
    cur_stress = local_stress[tau_i]

    # subtract elastic strain
    # this is the one thing that separates elasticity from simple modification
    # of the Delassus operator
    if local_stress_strain_matrices:
        tau += local_stress_strain_matrices[tau_i] * cur_stress

    tau_new = solve_coulomb_aniso(
        D,
        tau - cur_stress,
        nor,
        unilateral,
        friction_coeff,
        friction_coeff,
        yield_stress[tau_i],
    )

    delta_stress = tau_new - tau

    delta_correction[tau_i] = delta_stress
    local_stress[tau_i] = cur_stress + delta_stress


@wp.kernel
def solve_local_stress_jacobi(
    unilateral: wp.bool,
    friction_coeff: float,
    yield_stress: wp.array(dtype=wp.vec3),
    local_stress_strain_matrices: wp.array(dtype=mat66),
    strain_mat_offsets: wp.array(dtype=int),
    strain_mat_columns: wp.array(dtype=int),
    local_strain_mat_values: wp.array(dtype=mat63),
    delassus_diagonal: wp.array(dtype=vec6),
    delassus_normal: wp.array(dtype=vec6),
    local_strain_rhs: wp.array(dtype=vec6),
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

    solve_local_stress(
        tau_i,
        unilateral,
        friction_coeff,
        D,
        yield_stress,
        local_stress_strain_matrices,
        strain_mat_offsets,
        strain_mat_columns,
        local_strain_mat_values,
        delassus_normal,
        local_strain_rhs,
        velocities,
        local_stress,
        delta_correction,
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
    tau_i = color_indices[wp.tid() + color_offset]

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
    unilateral: wp.bool,
    friction_coeff: float,
    color_indices: wp.array(dtype=int),
    yield_stress: wp.array(dtype=wp.vec3),
    local_stress_strain_matrices: wp.array(dtype=mat66),
    strain_mat_offsets: wp.array(dtype=int),
    strain_mat_columns: wp.array(dtype=int),
    local_strain_mat_values: wp.array(dtype=mat63),
    delassus_diagonal: wp.array(dtype=vec6),
    delassus_normal: wp.array(dtype=vec6),
    inv_mass_matrix: wp.array(dtype=float),  # Note: Likely inv_volume in context
    local_strain_rhs: wp.array(dtype=vec6),
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
    tau_i = color_indices[wp.tid() + color_offset]

    D = delassus_diagonal[tau_i]
    solve_local_stress(
        tau_i,
        unilateral,
        friction_coeff,
        D,
        yield_stress,
        local_stress_strain_matrices,
        strain_mat_offsets,
        strain_mat_columns,
        local_strain_mat_values,
        delassus_normal,
        local_strain_rhs,
        velocities,
        local_stress,
        delta_correction,
    )

    apply_stress_delta_gs(
        tau_i,
        D,
        delta_correction[tau_i],
        strain_mat_offsets,
        strain_mat_columns,
        local_strain_mat_values,
        inv_mass_matrix,
        velocities,
    )


@wp.kernel
def apply_collider_impulse(
    collider_impulse: wp.array(dtype=wp.vec3),
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

    u = solve_coulomb_isotropic(friction_coeff, n, u0 - impulse[i] * w)

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
    unilateral: bool,
    friction_coeff: float,
    max_iterations: int,
    tolerance: float,
    strain_mat: sp.BsrMatrix,
    transposed_strain_mat: Optional[sp.BsrMatrix],
    inv_volume,
    yield_stress,
    stress_strain_matrices,
    strain_rhs,
    stress,
    velocity,
    collider_friction,
    collider_normals,
    collider_velocities,
    collider_inv_mass,
    collider_impulse,
    color_offsets,
    color_indices: Optional[wp.array] = None,
    rigidity_mat: Optional[sp.BsrMatrix] = None,
    temporary_store: Optional[fem.TemporaryStore] = None,
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

    # Compute and factorize diagonal blacks, rotate strain matrix to diagonal basis
    # NOTE: we reuse the same memory for local versions of the variables
    local_strain_mat_values = strain_mat.values
    local_stress_strain_matrices = stress_strain_matrices
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
            stress_strain_matrices,
            transposed_strain_mat.offsets,
            strain_rhs,
            stress,
        ],
        outputs=[
            delassus_rotation.array,
            delassus_diagonal.array,
            delassus_normal.array,
            local_strain_mat_values,
            local_stress_strain_matrices,
            local_strain_rhs,
            local_stress,
        ],
    )

    if gs:
        apply_stress_launch = wp.launch(
            kernel=apply_stress_gs,
            dim=1,
            inputs=[
                0,
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
                0,
                unilateral,
                friction_coeff * math.sqrt(3.0 / 2.0),
                color_indices,
                yield_stress,
                stress_strain_matrices,
                strain_mat.offsets,
                strain_mat.columns,
                strain_mat.values,
                delassus_diagonal.array,
                delassus_normal.array,
                inv_volume,
                strain_rhs,
            ],
            outputs=[
                velocity,
                stress,
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
                unilateral,
                friction_coeff * math.sqrt(3.0 / 2.0),
                yield_stress,
                local_stress_strain_matrices,
                strain_mat.offsets,
                strain_mat.columns,
                local_strain_mat_values,
                delassus_diagonal.array,
                delassus_normal.array,
                local_strain_rhs,
                velocity,
            ],
            outputs=[
                local_stress,
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

        res = math.sqrt(residual.numpy()[0]) / residual_scale
        if verbose:
            print(
                f"{'Gauss-Seidel' if gs else 'Jacobi'} terminated after {iteration_and_condition.array.numpy()[0]} iterations with residual {res}"
            )
    else:
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

    wp.launch(
        kernel=postprocess_stress,
        dim=stress.shape[0],
        inputs=[
            delassus_rotation.array,
            delassus_diagonal.array,
            local_stress_strain_matrices,
            local_stress,
        ],
        outputs=[
            stress,
            strain_rhs,
        ],
    )
