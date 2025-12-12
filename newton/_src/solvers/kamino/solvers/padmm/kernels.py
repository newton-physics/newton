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

"""Defines the Warp kernels used by the Proximal-ADMM solver."""

import warp as wp

from ...core.math import FLOAT32_EPS, FLOAT32_MAX
from ...core.types import float32, int32, vec2f, vec3f, vec6f
from .math import (
    compute_cwise_vec_mul,
    compute_desaxce_corrections,
    compute_dot_product,
    compute_double_dot_product,
    compute_gemv,
    compute_inverse_preconditioned_iterate_residual,
    compute_l2_norm,
    compute_ncp_complementarity_residual,
    compute_ncp_dual_residual,
    compute_ncp_natural_map_residual,
    compute_ncp_primal_residual,
    compute_preconditioned_iterate_residual,
    compute_vector_sum,
    project_to_coulomb_cone,
)
from .types import PADMMConfig, PADMMPenalty, PADMMPenaltyUpdate, PADMMStatus

###
# Module interface
###

__all__ = [
    "_apply_dual_preconditioner_to_solution",
    "_apply_dual_preconditioner_to_state",
    "_compute_complementarity_residuals",
    "_compute_desaxce_correction",
    "_compute_final_desaxce_correction",
    "_compute_infnorm_residuals_serially",
    "_compute_infnorm_residuals_serially_accel",
    "_compute_projection_argument",
    "_compute_solution_vectors",
    "_compute_velocity_bias",
    "_project_to_feasible_cone",
    "_update_delassus_proximal_regularization",
    "_update_state_with_acceleration",
    "_warmstart_contact_constraints",
    "_warmstart_desaxce_correction",
    "_warmstart_joint_constraints",
    "_warmstart_limit_constraints",
    "make_collect_solver_info_kernel",
    "make_initialize_solver_kernel",
    "make_update_dual_variables_and_compute_primal_dual_residuals",
    "make_update_proximal_regularization_kernel",
]


###
# Module configs
###

wp.set_module_options({"enable_backward": False})


###
# Kernels
###


@wp.kernel
def _warmstart_desaxce_correction(
    problem_nc: wp.array(dtype=int32),
    problem_cio: wp.array(dtype=int32),
    problem_ccgo: wp.array(dtype=int32),
    problem_vio: wp.array(dtype=int32),
    problem_mu: wp.array(dtype=float32),
    # Outputs:
    solver_z: wp.array(dtype=float32),
):
    # Retrieve the thread index
    wid, cid = wp.tid()

    # Retrieve the number of contact active in the world
    nc = problem_nc[wid]

    # Retrieve the limit constraint group offset of the world
    ccgo = problem_ccgo[wid]

    # Skip if row index exceed the problem size or if the solver has already converged
    if cid >= nc:
        return

    # Retrieve the index offset of the vector block of the world
    cio = problem_cio[wid]

    # Retrieve the index offset of the vector block of the world
    vio = problem_vio[wid]

    # Retrieve the friction coefficient for this contact
    mu = problem_mu[cio + cid]

    # Compute the vector index offset of the corresponding contact constraint
    ccio_k = vio + ccgo + 3 * cid

    # Compute the norm of the tangential components, where:
    #   s = G(v_plus)
    #   v_plus = z - s  =>  z = v_plus + s
    vtx = solver_z[ccio_k]
    vty = solver_z[ccio_k + 1]
    vn = solver_z[ccio_k + 2]
    vt_norm = wp.sqrt(vtx * vtx + vty * vty)

    # Store De Saxce correction for this block
    solver_z[ccio_k + 2] = vn + mu * vt_norm


@wp.kernel
def _warmstart_joint_constraints(
    # Inputs:
    model_time_dt: wp.array(dtype=float32),
    model_info_total_cts_offset: wp.array(dtype=int32),
    model_info_joint_cts_offset: wp.array(dtype=int32),
    joint_wid: wp.array(dtype=int32),
    joint_num_cts: wp.array(dtype=int32),
    joint_cts_offset: wp.array(dtype=int32),
    joint_lambda_j: wp.array(dtype=float32),
    problem_P: wp.array(dtype=float32),
    # Outputs:
    x_0: wp.array(dtype=float32),
    y_0: wp.array(dtype=float32),
    z_0: wp.array(dtype=float32),
):
    # Retrieve the thread index as the joint index
    jid = wp.tid()

    # Retrieve the joint-specific model info
    wid = joint_wid[jid]
    num_cts = joint_num_cts[jid]
    cts_offset = joint_cts_offset[jid]

    # Retrieve the world-specific info
    dt = model_time_dt[wid]
    world_joints_cts_offset = model_info_joint_cts_offset[wid]
    world_total_cts_offset = model_info_total_cts_offset[wid]

    # Compute block offsets of the joint's constraints within
    # the joint-only constraints and total constraints arrays
    jio_j = world_joints_cts_offset + cts_offset
    vio_j = world_total_cts_offset + cts_offset

    # Load the diagonal preconditioner for the joint constraints
    # NOTE: We pre-emptively allocate a 6-element vector to
    # account for the worst-case scenario of a 0-DOF joint
    P = vec6f(0.0)
    for j in range(num_cts):
        P[j] = problem_P[vio_j + j]

    # Load the joint constraint reaction forces and velocities
    lambda_j = vec6f(0.0)
    for j in range(num_cts):
        lambda_j[j] = (dt / P[j]) * joint_lambda_j[jio_j + j]

    # Store the joint-constraint reaction forces
    for j in range(num_cts):
        x_0[vio_j + j] = lambda_j[j]
    for j in range(num_cts):
        y_0[vio_j + j] = lambda_j[j]
    for j in range(num_cts):
        z_0[vio_j + j] = 0.0


@wp.kernel
def _warmstart_limit_constraints(
    # Inputs:
    model_time_dt: wp.array(dtype=float32),
    model_info_total_cts_offset: wp.array(dtype=int32),
    data_info_limit_cts_group_offset: wp.array(dtype=int32),
    limit_model_num_active: wp.array(dtype=int32),
    limit_wid: wp.array(dtype=int32),
    limit_lid: wp.array(dtype=int32),
    limit_reaction: wp.array(dtype=float32),
    limit_velocity: wp.array(dtype=float32),
    problem_P: wp.array(dtype=float32),
    # Outputs:
    x_0: wp.array(dtype=float32),
    y_0: wp.array(dtype=float32),
    z_0: wp.array(dtype=float32),
):
    # Retrieve the thread index as the limit index
    lid = wp.tid()

    # Retrieve the number of limits active in the model
    model_nl = limit_model_num_active[0]

    # Skip if lid is greater than the number of limits active in the model
    if lid >= model_nl:
        return

    # Retrieve the limit-specific data
    wid = limit_wid[lid]
    lid_l = limit_lid[lid]
    lambda_l = limit_reaction[lid]
    v_plus_l = limit_velocity[lid]

    # Retrieve the world-specific info
    dt = model_time_dt[wid]
    total_cts_offset = model_info_total_cts_offset[wid]
    limit_cts_offset = data_info_limit_cts_group_offset[wid]

    # Compute block offsets of the limit constraints within
    # the limit-only constraints and total constraints arrays
    vio_l = total_cts_offset + limit_cts_offset + lid_l

    # Load the diagonal preconditioner for the limit constraints
    # NOTE: We only need to load the first element since by necessity
    # the preconditioner is constant across the 3 constraint dimensions
    P_l = problem_P[vio_l]

    # Scale the limit force by the time-step to
    # render an impulse and by the preconditioner
    lambda_l *= dt / P_l

    # Scale the limit velocity by the preconditioner
    v_plus_l *= P_l

    # Compute and store the limit-constraint reaction forces
    x_0[vio_l] = lambda_l
    y_0[vio_l] = lambda_l
    z_0[vio_l] = v_plus_l


@wp.kernel
def _warmstart_contact_constraints(
    # Inputs:
    model_time_dt: wp.array(dtype=float32),
    model_info_total_cts_offset: wp.array(dtype=int32),
    data_info_contact_cts_group_offset: wp.array(dtype=int32),
    contact_model_num_contacts: wp.array(dtype=int32),
    contact_wid: wp.array(dtype=int32),
    contact_cid: wp.array(dtype=int32),
    contact_material: wp.array(dtype=vec2f),
    contact_reaction: wp.array(dtype=vec3f),
    contact_velocity: wp.array(dtype=vec3f),
    problem_P: wp.array(dtype=float32),
    # Outputs:
    x_0: wp.array(dtype=float32),
    y_0: wp.array(dtype=float32),
    z_0: wp.array(dtype=float32),
):
    # Retrieve the thread index as the contact index
    cid = wp.tid()

    # Retrieve the number of contacts active in the model
    model_nc = contact_model_num_contacts[0]

    # Skip if cid is greater than the number of contacts active in the model
    if cid >= model_nc:
        return

    # Retrieve the contact-specific data
    wid = contact_wid[cid]
    cid_k = contact_cid[cid]
    material_k = contact_material[cid]
    lambda_k = contact_reaction[cid]
    v_plus_k = contact_velocity[cid]

    # Retrieve the world-specific info
    dt = model_time_dt[wid]
    total_cts_offset = model_info_total_cts_offset[wid]
    contact_cts_offset = data_info_contact_cts_group_offset[wid]

    # Compute block offsets of the contact constraints within
    # the contact-only constraints and total constraints arrays
    vio_k = total_cts_offset + contact_cts_offset + 3 * cid_k

    # Load the diagonal preconditioner for the contact constraints
    # NOTE: We only need to load the first element since by necessity
    # the preconditioner is constant across the 3 constraint dimensions
    P_k = problem_P[vio_k]

    # Scale the contact force by the time-step to
    # render an impulse and by the preconditioner
    lambda_k *= dt / P_k

    # Scale the contact velocity by the preconditioner
    # and apply the De Saxce correction to the post-event
    # contact velocity to render solver dual variables
    v_plus_k *= P_k
    mu_k = material_k[0]
    vt_norm = wp.sqrt(v_plus_k.x * v_plus_k.x + v_plus_k.y * v_plus_k.y)
    v_plus_k.z += mu_k * vt_norm

    # Compute and store the contact-constraint reaction forces
    for k in range(3):
        x_0[vio_k + k] = lambda_k[k]
    for k in range(3):
        y_0[vio_k + k] = lambda_k[k]
    for k in range(3):
        z_0[vio_k + k] = v_plus_k[k]


def make_initialize_solver_kernel(use_acceleration: bool = False):
    """
    Creates a kernel to initialize the PADMM solver state, status, and penalty parameters.

    Specialized for whether acceleration is enabled to reduce unnecessary overhead and branching.

    Args:
        use_acceleration (bool, optional): Flag indicating whether acceleration is enabled. Defaults to False.

    Returns:
        wp.kernel: The kernel function to initialize the PADMM solver.
    """

    @wp.kernel
    def _initialize_solver(
        # Inputs:
        solver_config: wp.array(dtype=PADMMConfig),
        # Outputs:
        solver_status: wp.array(dtype=PADMMStatus),
        solver_penalty: wp.array(dtype=PADMMPenalty),
        solver_state_sigma: wp.array(dtype=vec2f),
        solver_state_a_p: wp.array(dtype=float32),
    ):
        # Retrieve the world index as thread index
        wid = wp.tid()

        # Retrieve the per-world solver data
        config = solver_config[wid]
        status = solver_status[wid]
        penalty = solver_penalty[wid]
        sigma = solver_state_sigma[wid]

        # Initialize solver status
        status.iterations = int32(0)
        status.converged = int32(0)
        status.r_p = float32(0.0)
        status.r_d = float32(0.0)
        status.r_c = float32(0.0)
        # NOTE: We initialize acceleration-related
        # entries only if acceleration is enabled
        if wp.static(use_acceleration):
            status.r_dx = float32(0.0)
            status.r_dy = float32(0.0)
            status.r_dz = float32(0.0)
            status.r_a = FLOAT32_MAX
            status.r_a_p = FLOAT32_MAX
            status.r_a_pp = FLOAT32_MAX
            status.restart = int32(0)
            status.num_restarts = int32(0)

        # Initialize ALM penalty parameter and relevant meta-data
        # NOTE: Currently only fixed penalty is used
        penalty.rho = config.rho_0
        penalty.rho_p = float32(0.0)
        penalty.num_updates = int32(0)

        # Initialize the total proximal regularization
        sigma[0] = config.eta + config.rho_0
        sigma[1] = float32(0.0)

        # Store the initialized per-world solver data
        solver_status[wid] = status
        solver_penalty[wid] = penalty
        solver_state_sigma[wid] = sigma

        # Initialize the previous acclereration
        # variables only if acceleration is used
        if wp.static(use_acceleration):
            solver_state_a_p[wid] = config.a_0

    # Return the initialization kernel
    return _initialize_solver


def make_update_proximal_regularization_kernel(method: PADMMPenaltyUpdate):
    @wp.kernel
    def _update_proximal_regularization(
        # Inputs:
        solver_config: wp.array(dtype=PADMMConfig),
        solver_penalty: wp.array(dtype=PADMMPenalty),
        solver_status: wp.array(dtype=PADMMStatus),
        # Outputs:
        solver_state_sigma: wp.array(dtype=vec2f),
    ):
        # Retrieve the world index from the thread index
        wid = wp.tid()

        # Retrieve the solver status
        status = solver_status[wid]

        # Skip if row index exceed the problem size or if the solver has already converged
        if status.converged > 0:
            return

        # Retrieve the solver parameters
        cfg = solver_config[wid]
        pen = solver_penalty[wid]

        # Retrieve the (current, previous) proximal regularization pair
        sigma = solver_state_sigma[wid]

        # Extract the regularization parameters
        rho = pen.rho
        eta = cfg.eta

        # TODO: Add penalty update methods here

        # Update the diagonal proximal regularization
        sigma[1] = sigma[0]
        sigma[0] = eta + rho

    # Return the proximal regularization update kernel
    return _update_proximal_regularization


@wp.kernel
def _update_delassus_proximal_regularization(
    # Inputs:
    problem_dim: wp.array(dtype=int32),
    problem_mio: wp.array(dtype=int32),
    solver_status: wp.array(dtype=PADMMStatus),
    solver_state_sigma: wp.array(dtype=vec2f),
    # Outputs:
    D: wp.array(dtype=float32),
):
    # Retrieve the thread index
    wid, tid = wp.tid()

    # Retrieve the number of active constraints in the world
    ncts = problem_dim[wid]

    # Retrieve the solver status
    status = solver_status[wid]

    # Skip if row index exceed the problem size or if the solver has already converged
    if tid >= ncts or status.converged > 0:
        return

    # Retrieve the matrix index offset of the world
    mio = problem_mio[wid]

    # Retrieve the (current, previous) proximal regularization pair
    sigma = solver_state_sigma[wid]

    # Add the proximal regularization to the diagonal of the Delassus matrix
    D[mio + ncts * tid + tid] += sigma[0] - sigma[1]


@wp.kernel
def _compute_desaxce_correction(
    # Inputs:
    problem_nc: wp.array(dtype=int32),
    problem_cio: wp.array(dtype=int32),
    problem_ccgo: wp.array(dtype=int32),
    problem_vio: wp.array(dtype=int32),
    problem_mu: wp.array(dtype=float32),
    solver_status: wp.array(dtype=PADMMStatus),
    solver_z_p: wp.array(dtype=float32),
    # Outputs:
    solver_s: wp.array(dtype=float32),
):
    # Retrieve the thread index as the contact index
    wid, cid = wp.tid()

    # Retrieve the number of contact active in the world
    nc = problem_nc[wid]

    # Retrieve the solver status
    status = solver_status[wid]

    # Skip if row index exceed the problem size or if the solver has already converged
    if cid >= nc or status.converged > 0:
        return

    # Retrieve the contacts index offset of the world
    cio = problem_cio[wid]

    # Retrieve the index offset of the vector block of the world
    vio = problem_vio[wid]

    # Retrieve the contact constraints group offset of the world
    ccgo = problem_ccgo[wid]

    # Compute the index offset of the corresponding contact constraint
    ccio_k = vio + ccgo + 3 * cid

    # Retrieve the contact index w.r.t the model
    cio_k = cio + cid

    # Compute the norm of the tangential components
    vtx = solver_z_p[ccio_k]
    vty = solver_z_p[ccio_k + 1]
    vt_norm = wp.sqrt(vtx * vtx + vty * vty)

    # Store De Saxce correction for this block
    solver_s[ccio_k] = 0.0
    solver_s[ccio_k + 1] = 0.0
    solver_s[ccio_k + 2] = problem_mu[cio_k] * vt_norm


@wp.kernel
def _compute_velocity_bias(
    # Inputs:
    problem_dim: wp.array(dtype=int32),
    problem_vio: wp.array(dtype=int32),
    problem_v_f: wp.array(dtype=float32),
    solver_config: wp.array(dtype=PADMMConfig),
    solver_penalty: wp.array(dtype=PADMMPenalty),
    solver_status: wp.array(dtype=PADMMStatus),
    solver_s: wp.array(dtype=float32),
    solver_x_p: wp.array(dtype=float32),
    solver_y_p: wp.array(dtype=float32),
    solver_z_p: wp.array(dtype=float32),
    # Outputs:
    solver_v: wp.array(dtype=float32),
):
    # Retrieve the thread indices as the world and constraint index
    wid, tid = wp.tid()

    # Retrieve the total number of active constraints in the world
    ncts = problem_dim[wid]

    # Retrieve the solver status
    status = solver_status[wid]

    # Skip if row index exceed the problem size or if the solver has already converged
    if tid >= ncts or status.converged > 0:
        return

    # Retrieve the index offset of the vector block of the world
    vio = problem_vio[wid]

    # Retrieve solver parameters
    eta = solver_config[wid].eta
    rho = solver_penalty[wid].rho

    # Compute the index offset of the vector block of the world
    thread_offset = vio + tid

    # Retrieve the solver state
    v_f = problem_v_f[thread_offset]
    s = solver_s[thread_offset]
    x_p = solver_x_p[thread_offset]
    y_p = solver_y_p[thread_offset]
    z_p = solver_z_p[thread_offset]

    # Compute the total velocity bias for the thread_offset-th constraint
    solver_v[thread_offset] = -v_f - s + eta * x_p + rho * y_p + z_p


@wp.kernel
def _compute_projection_argument(
    # Inputs
    problem_dim: wp.array(dtype=int32),
    problem_vio: wp.array(dtype=int32),
    solver_penalty: wp.array(dtype=PADMMPenalty),
    solver_status: wp.array(dtype=PADMMStatus),
    solver_z_hat: wp.array(dtype=float32),
    solver_x: wp.array(dtype=float32),
    # Outputs
    solver_y: wp.array(dtype=float32),
):
    # Retrieve the thread indices as the world and constraint index
    wid, tid = wp.tid()

    # Retrieve the total number of active constraints in the world
    ncts = problem_dim[wid]

    # Retrieve the solver status
    status = solver_status[wid]

    # Skip if row index exceed the problem size or if the solver has already converged
    if tid >= ncts or status.converged > 0:
        return

    # Retrieve the index offset of the vector block of the world
    vio = problem_vio[wid]

    # Capture the ALM penalty
    rho = solver_penalty[wid].rho

    # Compute the index offset of the vector block of the world
    thread_offset = vio + tid

    # Retrieve the solver state variables
    z_hat = solver_z_hat[thread_offset]
    x = solver_x[thread_offset]

    # Compute and store the updated values back to the solver state
    solver_y[thread_offset] = x - (1.0 / rho) * z_hat


@wp.kernel
def _project_to_feasible_cone(
    # Inputs:
    problem_nl: wp.array(dtype=int32),
    problem_nc: wp.array(dtype=int32),
    problem_cio: wp.array(dtype=int32),
    problem_lcgo: wp.array(dtype=int32),
    problem_ccgo: wp.array(dtype=int32),
    problem_vio: wp.array(dtype=int32),
    problem_mu: wp.array(dtype=float32),
    solver_status: wp.array(dtype=PADMMStatus),
    # Outputs:
    solver_y: wp.array(dtype=float32),
):
    # Retrieve the thread index as the unilateral entity index
    wid, uid = wp.tid()

    # Retrieve the solver status
    status = solver_status[wid]

    # Retrieve the number of active limits and contacts in the world
    nl = problem_nl[wid]
    nc = problem_nc[wid]

    # Skip if row index exceed the problem size or if the solver has already converged
    if uid >= (nl + nc) or status.converged > 0:
        return

    # Retrieve the index offset of the vector block of the world
    vio = problem_vio[wid]

    # Check if the thread should handle a limit
    if nl > 0 and uid < nl:
        # Retrieve the limit constraint group offset of the world
        lcgo = problem_lcgo[wid]
        # Compute the constraint index offset of the limit element
        lcio_j = vio + lcgo + uid
        # Project to the non-negative orthant
        solver_y[lcio_j] = wp.max(solver_y[lcio_j], 0.0)

    # Check if the thread should handle a contact
    elif nc > 0 and uid >= nl:
        # Retrieve the contact index offset of the world
        cio = problem_cio[wid]
        # Retrieve the limit constraint group offset of the world
        ccgo = problem_ccgo[wid]
        # Compute the index of the contact element in the unilaterals array
        # NOTE: We need to subtract the number of active limits
        cid = uid - nl
        # Compute the index offset of the contact constraint
        ccio_j = vio + ccgo + 3 * cid
        # Capture a 3D vector
        x = vec3f(solver_y[ccio_j], solver_y[ccio_j + 1], solver_y[ccio_j + 2])
        # Project to the coulomb friction cone
        y_proj = project_to_coulomb_cone(x, problem_mu[cio + cid])
        # Copy vec3 projection into the slack variable array
        solver_y[ccio_j] = y_proj[0]
        solver_y[ccio_j + 1] = y_proj[1]
        solver_y[ccio_j + 2] = y_proj[2]


def make_update_dual_variables_and_compute_primal_dual_residuals(use_acceleration: bool = False):
    """
    Creates a kernel to update the dual variables and compute the primal and dual residuals.

    Specialized for whether acceleration is enabled to reduce unnecessary overhead and branching.

    Args:
        use_acceleration (bool, optional): Flag indicating whether acceleration is enabled. Defaults to False.

    Returns:
        wp.kernel: The kernel function to update dual variables and compute residuals.
    """

    @wp.kernel
    def _update_dual_variables_and_compute_primal_dual_residuals(
        # Inputs:
        problem_dim: wp.array(dtype=int32),
        problem_vio: wp.array(dtype=int32),
        problem_P: wp.array(dtype=float32),
        solver_config: wp.array(dtype=PADMMConfig),
        solver_penalty: wp.array(dtype=PADMMPenalty),
        solver_status: wp.array(dtype=PADMMStatus),
        solver_x: wp.array(dtype=float32),
        solver_y: wp.array(dtype=float32),
        solver_x_p: wp.array(dtype=float32),
        solver_y_p: wp.array(dtype=float32),
        solver_z_p: wp.array(dtype=float32),
        # Outputs:
        solver_z: wp.array(dtype=float32),
        solver_r_prim: wp.array(dtype=float32),
        solver_r_dual: wp.array(dtype=float32),
        solver_r_dx: wp.array(dtype=float32),
        solver_r_dy: wp.array(dtype=float32),
        solver_r_dz: wp.array(dtype=float32),
    ):
        # Retrieve the thread indices as the world and constraint index
        wid, tid = wp.tid()

        # Retrieve the total number of active constraints in the world
        ncts = problem_dim[wid]

        # Retrieve the solver status
        status = solver_status[wid]

        # Skip if row index exceed the problem size or if the solver has already converged
        if tid >= ncts or status.converged > 0:
            return

        # Retrieve the index offset of the vector block of the world
        vio = problem_vio[wid]

        # Capture proximal parameter and the ALM penalty
        eta = solver_config[wid].eta
        rho = solver_penalty[wid].rho

        # Compute the index offset of the vector block of the world
        thread_offset = vio + tid

        # Retrieve
        P_i = problem_P[thread_offset]

        # Retrieve the solver state inputs
        x = solver_x[thread_offset]
        y = solver_y[thread_offset]
        x_p = solver_x_p[thread_offset]
        y_p = solver_y_p[thread_offset]
        z_p = solver_z_p[thread_offset]

        # Compute and store the dual variable update
        z = z_p + rho * (y - x)
        solver_z[thread_offset] = z

        # Compute the primal residual as the consensus of the primal and slack variable
        solver_r_prim[thread_offset] = P_i * (x - y)

        # Compute the dual residual using the ADMM-specific shortcut
        solver_r_dual[thread_offset] = (1.0 / P_i) * (eta * (x - x_p) + rho * (y - y_p))

        # Compute the individual iterate residuals only if acceleration is enabled
        # NOTE: These are used to compute the combined residual
        # for checking the acceleration restart criteria
        if wp.static(use_acceleration):
            solver_r_dx[thread_offset] = P_i * (x - x_p)
            solver_r_dy[thread_offset] = P_i * (y - y_p)
            solver_r_dz[thread_offset] = (1.0 / P_i) * (z - z_p)

    # Return the dual update and residual computation kernel
    return _update_dual_variables_and_compute_primal_dual_residuals


@wp.kernel
def _compute_complementarity_residuals(
    # Inputs:
    problem_nl: wp.array(dtype=int32),
    problem_nc: wp.array(dtype=int32),
    problem_vio: wp.array(dtype=int32),
    problem_uio: wp.array(dtype=int32),
    problem_lcgo: wp.array(dtype=int32),
    problem_ccgo: wp.array(dtype=int32),
    solver_status: wp.array(dtype=PADMMStatus),
    solver_x: wp.array(dtype=float32),
    solver_z: wp.array(dtype=float32),
    # Outputs:
    solver_r_c: wp.array(dtype=float32),
):
    # Retrieve the thread index as the unilateral entity index
    wid, uid = wp.tid()

    # Retrieve the solver status
    status = solver_status[wid]

    # Retrieve the number of active limits and contacts in the world
    nl = problem_nl[wid]
    nc = problem_nc[wid]

    # Skip if row index exceed the problem size or if the solver has already converged
    if uid >= (nl + nc) or status.converged > 0:
        return

    # Retrieve the index offsets of the unilateral elements
    uio = problem_uio[wid]

    # Retrieve the index offset of the vector block of the world
    vio = problem_vio[wid]

    # Compute the index offset of the vector block of the world
    uio_u = uio + uid

    # Check if the thread should handle a limit
    if nl > 0 and uid < nl:
        # Retrieve the limit constraint group offset of the world
        lcgo = problem_lcgo[wid]
        # Compute the constraint index offset of the limit element
        lcio_j = vio + lcgo + uid
        # Compute the scalar product of the primal and dual variables
        solver_r_c[uio_u] = solver_x[lcio_j] * solver_z[lcio_j]

    # Check if the thread should handle a contact
    elif nc > 0 and uid >= nl:
        # Retrieve the limit constraint group offset of the world
        ccgo = problem_ccgo[wid]
        # Compute the index of the contact element in the unilaterals array
        # NOTE: We need to subtract the number of active limits
        cid = uid - nl
        # Compute the index offset of the contact constraint
        ccio_j = vio + ccgo + 3 * cid
        # Capture 3D vectors
        x_c = vec3f(solver_x[ccio_j], solver_x[ccio_j + 1], solver_x[ccio_j + 2])
        z_c = vec3f(solver_z[ccio_j], solver_z[ccio_j + 1], solver_z[ccio_j + 2])
        # Compute the inner product of the primal and dual variables
        solver_r_c[uio_u] = wp.dot(x_c, z_c)


@wp.kernel
def _compute_infnorm_residuals_serially(
    # Inputs:
    problem_nl: wp.array(dtype=int32),
    problem_nc: wp.array(dtype=int32),
    problem_uio: wp.array(dtype=int32),
    problem_dim: wp.array(dtype=int32),
    problem_vio: wp.array(dtype=int32),
    solver_config: wp.array(dtype=PADMMConfig),
    solver_r_p: wp.array(dtype=float32),
    solver_r_d: wp.array(dtype=float32),
    solver_r_c: wp.array(dtype=float32),
    # Outputs:
    solver_state_done: wp.array(dtype=int32),
    solver_status: wp.array(dtype=PADMMStatus),
):
    # Retrieve the thread index as the world index
    wid = wp.tid()

    # Retrieve the solver status
    status = solver_status[wid]

    # Skip this step if already converged
    if status.converged:
        return

    # Update iteration counter
    status.iterations += 1

    # Capture the size of the residuals arrays
    nl = problem_nl[wid]
    nc = problem_nc[wid]
    ncts = problem_dim[wid]

    # Retrieve the solver configurations
    config = solver_config[wid]

    # Retrieve the index offsets of the vector block and unilateral elements
    vio = problem_vio[wid]
    uio = problem_uio[wid]

    # Extract the solver tolerances
    eps_p = config.primal_tolerance
    eps_d = config.dual_tolerance
    eps_c = config.compl_tolerance

    # Extract the maximum number of iterations
    maxiters = config.max_iterations

    # Compute element-wise max over each residual vector to compute the infinity-norm
    r_p_max = float(0.0)
    r_d_max = float(0.0)
    for j in range(ncts):
        rio_j = vio + j
        r_p_max = wp.max(r_p_max, wp.abs(solver_r_p[rio_j]))
        r_d_max = wp.max(r_d_max, wp.abs(solver_r_d[rio_j]))

    # Compute the infinity-norm of the complementarity residuals
    nu = nl + nc
    r_c_max = float(0.0)
    for j in range(nu):
        r_c_max = wp.max(r_c_max, wp.abs(solver_r_c[uio + j]))

    # Store the scalar metric residuals in the solver status
    status.r_p = r_p_max
    status.r_d = r_d_max
    status.r_c = r_c_max

    # Check and store convergence state
    if status.iterations > 1 and r_p_max <= eps_p and r_d_max <= eps_d and r_c_max <= eps_c:
        status.converged = 1

    # If converged or reached max iterations, decrement the number of active worlds
    if status.converged or status.iterations >= maxiters:
        solver_state_done[0] -= 1

    # Store the updated status
    solver_status[wid] = status


@wp.kernel
def _compute_infnorm_residuals_serially_accel(
    # Inputs:
    problem_nl: wp.array(dtype=int32),
    problem_nc: wp.array(dtype=int32),
    problem_uio: wp.array(dtype=int32),
    problem_dim: wp.array(dtype=int32),
    problem_vio: wp.array(dtype=int32),
    solver_config: wp.array(dtype=PADMMConfig),
    solver_params: wp.array(dtype=PADMMPenalty),
    solver_r_p: wp.array(dtype=float32),
    solver_r_d: wp.array(dtype=float32),
    solver_r_c: wp.array(dtype=float32),
    solver_r_dx: wp.array(dtype=float32),
    solver_r_dy: wp.array(dtype=float32),
    solver_r_dz: wp.array(dtype=float32),
    solver_state_a_p: wp.array(dtype=float32),
    # Outputs:
    solver_state_done: wp.array(dtype=int32),
    solver_state_a: wp.array(dtype=float32),
    solver_status: wp.array(dtype=PADMMStatus),
):
    # Retrieve the thread index as the world index
    wid = wp.tid()

    # Retrieve the solver status
    status = solver_status[wid]

    # Skip this step if already converged
    if status.converged:
        return

    # Update iteration counter
    status.iterations += 1

    # Capture the size of the residuals arrays
    nl = problem_nl[wid]
    nc = problem_nc[wid]
    ncts = problem_dim[wid]

    # Retrieve the solver configurations
    config = solver_config[wid]

    # Retrieve the index offsets of the vector block and unilateral elements
    vio = problem_vio[wid]
    uio = problem_uio[wid]

    # Extract the solver tolerances
    eps_p = config.primal_tolerance
    eps_d = config.dual_tolerance
    eps_c = config.compl_tolerance

    # Extract the maximum number of iterations
    maxiters = config.max_iterations

    # Extract the penalty parameters
    params = solver_params[wid]
    rho = params.rho

    # Compute element-wise max over each residual vector to compute the infinity-norm
    r_p_max = float(0.0)
    r_d_max = float(0.0)
    r_dx_l2_sum = float(0.0)
    r_dy_l2_sum = float(0.0)
    r_dz_l2_sum = float(0.0)
    for j in range(ncts):
        rio_j = vio + j
        r_p_max = wp.max(r_p_max, wp.abs(solver_r_p[rio_j]))
        r_d_max = wp.max(r_d_max, wp.abs(solver_r_d[rio_j]))
        r_dx = solver_r_dx[rio_j]
        r_dy = solver_r_dy[rio_j]
        r_dz = solver_r_dz[rio_j]
        r_dx_l2_sum += r_dx * r_dx
        r_dy_l2_sum += r_dy * r_dy
        r_dz_l2_sum += r_dz * r_dz

    # Compute the infinity-norm of the complementarity residuals
    nu = nl + nc
    r_c_max = float(0.0)
    for j in range(nu):
        r_c_max = wp.max(r_c_max, wp.abs(solver_r_c[uio + j]))

    # Store the scalar metric residuals in the solver status
    status.r_p = r_p_max
    status.r_d = r_d_max
    status.r_c = r_c_max
    status.r_dx = wp.sqrt(r_dx_l2_sum)
    status.r_dy = wp.sqrt(r_dy_l2_sum)
    status.r_dz = wp.sqrt(r_dz_l2_sum)
    status.r_a = rho * status.r_dy + (1.0 / rho) * status.r_dz

    # Check and store convergence state
    if status.iterations > 1 and r_p_max <= eps_p and r_d_max <= eps_d and r_c_max <= eps_c:
        status.converged = 1

    # If converged or reached max iterations, decrement the number of active worlds
    if status.converged or status.iterations >= maxiters:
        solver_state_done[0] -= 1

    # Restart acceleration if the residuals are not decreasing sufficiently
    # TODO: Use a warp function that is wrapped with wp.static for conditionally compiling this
    if status.r_a < config.restart_tolerance * status.r_a_p:
        status.restart = 0
        a_p = solver_state_a_p[wid]
        solver_state_a[wid] = (1.0 + wp.sqrt(1.0 + 4.0 * a_p * a_p)) / 2.0
    else:
        status.restart = 1
        status.num_restarts += 1
        status.r_a = status.r_a_p / config.restart_tolerance
        solver_state_a[wid] = float(config.a_0)
    status.r_a_pp = status.r_a_p
    status.r_a_p = status.r_a

    # Store the updated status
    solver_status[wid] = status


@wp.kernel
def _update_state_with_acceleration(
    # Inputs:
    problem_dim: wp.array(dtype=int32),
    problem_vio: wp.array(dtype=int32),
    solver_status: wp.array(dtype=PADMMStatus),
    solver_state_a: wp.array(dtype=float32),
    solver_state_y: wp.array(dtype=float32),
    solver_state_z: wp.array(dtype=float32),
    solver_state_a_p: wp.array(dtype=float32),
    solver_state_y_p: wp.array(dtype=float32),
    solver_state_z_p: wp.array(dtype=float32),
    # Outputs:
    solver_state_y_hat: wp.array(dtype=float32),
    solver_state_z_hat: wp.array(dtype=float32),
):
    # Retrieve the thread indices as the world and constraint index
    wid, tid = wp.tid()

    # Retrieve the total number of active constraints in the world
    ncts = problem_dim[wid]

    # Retrieve the solver status
    status = solver_status[wid]

    # Skip if row index exceed the problem size or if the solver has already converged
    if tid >= ncts or status.converged > 0:
        return

    # Retrieve the index offset of the vector block of the world
    vio = problem_vio[wid]

    # Compute the index offset of the vector block of the world
    vid = vio + tid

    # Only apply acceleration if not restarting
    if status.restart == 0:
        # Retrieve the previous and current states
        a = solver_state_a[wid]
        y = solver_state_y[vid]
        z = solver_state_z[vid]
        a_p = solver_state_a_p[wid]
        y_p = solver_state_y_p[vid]
        z_p = solver_state_z_p[vid]

        # Compute the current acceleration factor
        factor = (a_p - 1.0) / a

        # Update the primal and dual variables with Nesterov acceleration
        y_hat_new = y + factor * (y - y_p)
        z_hat_new = z + factor * (z - z_p)

        # Store the updated states
        solver_state_y_hat[vid] = y_hat_new
        solver_state_z_hat[vid] = z_hat_new
    else:
        # If restarting, reset the auxiliary primal-dual state to the previous-step values
        solver_state_y_hat[vid] = solver_state_y_p[vid]
        solver_state_z_hat[vid] = solver_state_z_p[vid]


def make_collect_solver_info_kernel(use_acceleration: bool):
    """
    Creates a kernel to collect solver convergence information after each iteration.

    Specializes the kernel based on whether acceleration is enabled to reduce unnecessary overhead.

    Args:
        use_acceleration (bool): Whether acceleration is enabled in the solver.

    Returns:
        wp.kernel: The kernel function to collect solver convergence information.
    """

    @wp.kernel
    def _collect_solver_convergence_info(
        # Inputs:
        problem_nl: wp.array(dtype=int32),
        problem_nc: wp.array(dtype=int32),
        problem_cio: wp.array(dtype=int32),
        problem_lcgo: wp.array(dtype=int32),
        problem_ccgo: wp.array(dtype=int32),
        problem_dim: wp.array(dtype=int32),
        problem_vio: wp.array(dtype=int32),
        problem_mio: wp.array(dtype=int32),
        problem_mu: wp.array(dtype=float32),
        problem_v_f: wp.array(dtype=float32),
        problem_D: wp.array(dtype=float32),
        problem_P: wp.array(dtype=float32),
        solver_state_sigma: wp.array(dtype=vec2f),
        solver_state_s: wp.array(dtype=float32),
        solver_state_x: wp.array(dtype=float32),
        solver_state_x_p: wp.array(dtype=float32),
        solver_state_y: wp.array(dtype=float32),
        solver_state_y_p: wp.array(dtype=float32),
        solver_state_z: wp.array(dtype=float32),
        solver_state_z_p: wp.array(dtype=float32),
        solver_state_a: wp.array(dtype=float32),
        solver_penalty: wp.array(dtype=PADMMPenalty),
        solver_status: wp.array(dtype=PADMMStatus),
        # Outputs:
        solver_info_lambdas: wp.array(dtype=float32),
        solver_info_v_plus: wp.array(dtype=float32),
        solver_info_v_aug: wp.array(dtype=float32),
        solver_info_s: wp.array(dtype=float32),
        solver_info_offset: wp.array(dtype=int32),
        solver_info_num_restarts: wp.array(dtype=int32),
        solver_info_num_rho_updates: wp.array(dtype=int32),
        solver_info_a: wp.array(dtype=float32),
        solver_info_norm_s: wp.array(dtype=float32),
        solver_info_norm_x: wp.array(dtype=float32),
        solver_info_norm_y: wp.array(dtype=float32),
        solver_info_norm_z: wp.array(dtype=float32),
        solver_info_f_ccp: wp.array(dtype=float32),
        solver_info_f_ncp: wp.array(dtype=float32),
        solver_info_r_dx: wp.array(dtype=float32),
        solver_info_r_dy: wp.array(dtype=float32),
        solver_info_r_dz: wp.array(dtype=float32),
        solver_info_r_primal: wp.array(dtype=float32),
        solver_info_r_dual: wp.array(dtype=float32),
        solver_info_r_compl: wp.array(dtype=float32),
        solver_info_r_pd: wp.array(dtype=float32),
        solver_info_r_dp: wp.array(dtype=float32),
        solver_info_r_comb: wp.array(dtype=float32),
        solver_info_r_comb_ratio: wp.array(dtype=float32),
        solver_info_r_ncp_primal: wp.array(dtype=float32),
        solver_info_r_ncp_dual: wp.array(dtype=float32),
        solver_info_r_ncp_compl: wp.array(dtype=float32),
        solver_info_r_ncp_natmap: wp.array(dtype=float32),
    ):
        # Retrieve the thread index as the world index
        wid = wp.tid()

        # Retrieve the world-specific data
        nl = problem_nl[wid]
        nc = problem_nc[wid]
        ncts = problem_dim[wid]
        cio = problem_cio[wid]
        lcgo = problem_lcgo[wid]
        ccgo = problem_ccgo[wid]
        vio = problem_vio[wid]
        mio = problem_mio[wid]
        rio = solver_info_offset[wid]
        penalty = solver_penalty[wid]
        status = solver_status[wid]
        sigma = solver_state_sigma[wid]

        # Retrieve parameters
        iter = status.iterations - 1

        # Compute additional info
        njc = ncts - (nl + 3 * nc)

        # Compute and store the norms of the current solution state
        norm_s = compute_l2_norm(ncts, vio, solver_state_s)
        norm_x = compute_l2_norm(ncts, vio, solver_state_x)
        norm_y = compute_l2_norm(ncts, vio, solver_state_y)
        norm_z = compute_l2_norm(ncts, vio, solver_state_z)

        # Compute (division safe) residual ratios
        r_pd = status.r_p / (status.r_d + FLOAT32_EPS)
        r_dp = status.r_d / (status.r_p + FLOAT32_EPS)

        # Compute the post-event constraint-space velocity from the current solution: v_plus = v_f + D @ lambda
        compute_cwise_vec_mul(ncts, vio, problem_P, solver_state_y, solver_info_lambdas)

        # Compute the post-event constraint-space velocity from the current solution: v_plus = v_f + D @ lambda
        compute_gemv(ncts, vio, mio, sigma[0], problem_P, problem_D, solver_state_y, problem_v_f, solver_info_v_plus)

        # Compute the De Saxce correction for each contact as: s = G(v_plus)
        compute_desaxce_corrections(nc, cio, vio, ccgo, problem_mu, solver_info_v_plus, solver_info_s)

        # Compute the CCP optimization objective as: f_ccp = 0.5 * lambda.dot(v_plus + v_f)
        f_ccp = 0.5 * compute_double_dot_product(ncts, vio, solver_info_lambdas, solver_info_v_plus, problem_v_f)

        # Compute the NCP optimization objective as:  f_ncp = f_ccp + lambda.dot(s)
        f_ncp = compute_dot_product(ncts, vio, solver_info_lambdas, solver_info_s)
        f_ncp += f_ccp

        # Compute the augmented post-event constraint-space velocity as: v_aug = v_plus + s
        compute_vector_sum(ncts, vio, solver_info_v_plus, solver_info_s, solver_info_v_aug)

        # Compute the NCP primal residual as: r_p := || lambda - proj_K(lambda) ||_inf
        r_ncp_p, _ = compute_ncp_primal_residual(nl, nc, vio, lcgo, ccgo, cio, problem_mu, solver_info_lambdas)

        # Compute the NCP dual residual as: r_d := || v_plus + s - proj_dual_K(v_plus + s)  ||_inf
        r_ncp_d, _ = compute_ncp_dual_residual(njc, nl, nc, vio, lcgo, ccgo, cio, problem_mu, solver_info_v_aug)

        # Compute the NCP complementarity (lambda _|_ (v_plus + s)) residual as r_c := || lambda.dot(v_plus + s) ||_inf
        r_ncp_c, _ = compute_ncp_complementarity_residual(
            nl, nc, vio, lcgo, ccgo, solver_info_v_aug, solver_info_lambdas
        )

        # Compute the natural-map residuals as: r_natmap = || lambda - proj_K(lambda - (v + s)) ||_inf
        r_ncp_natmap, _ = compute_ncp_natural_map_residual(
            nl, nc, vio, lcgo, ccgo, cio, problem_mu, solver_info_v_aug, solver_info_lambdas
        )

        # Compute the iterate residual as: r_iter := || y - y_p ||_inf
        r_dx = compute_preconditioned_iterate_residual(ncts, vio, problem_P, solver_state_x, solver_state_x_p)
        r_dy = compute_preconditioned_iterate_residual(ncts, vio, problem_P, solver_state_y, solver_state_y_p)
        r_dz = compute_inverse_preconditioned_iterate_residual(ncts, vio, problem_P, solver_state_z, solver_state_z_p)

        # Compute index offset for the info of the current iteration
        iio = rio + iter

        # Store the convergence information in the solver info arrays
        solver_info_num_rho_updates[iio] = penalty.num_updates
        solver_info_norm_s[iio] = norm_s
        solver_info_norm_x[iio] = norm_x
        solver_info_norm_y[iio] = norm_y
        solver_info_norm_z[iio] = norm_z
        solver_info_r_dx[iio] = r_dx
        solver_info_r_dy[iio] = r_dy
        solver_info_r_dz[iio] = r_dz
        solver_info_r_primal[iio] = status.r_p
        solver_info_r_dual[iio] = status.r_d
        solver_info_r_compl[iio] = status.r_c
        solver_info_r_pd[iio] = r_pd
        solver_info_r_dp[iio] = r_dp
        solver_info_r_ncp_primal[iio] = r_ncp_p
        solver_info_r_ncp_dual[iio] = r_ncp_d
        solver_info_r_ncp_compl[iio] = r_ncp_c
        solver_info_r_ncp_natmap[iio] = r_ncp_natmap
        solver_info_f_ccp[iio] = f_ccp
        solver_info_f_ncp[iio] = f_ncp

        # Optionally store acceleration-relevant info if acceleration is enabled
        # NOTE: This is statically evaluated to avoid unnecessary overhead when acceleration is disabled
        if wp.static(use_acceleration):
            solver_info_a[iio] = solver_state_a[wid]
            solver_info_r_comb[iio] = status.r_a
            solver_info_r_comb_ratio[iio] = status.r_a / (status.r_a_pp)
            solver_info_num_restarts[iio] = status.num_restarts

    # Return the generated kernel
    return _collect_solver_convergence_info


@wp.kernel
def _apply_dual_preconditioner_to_state(
    # Inputs:
    problem_dim: wp.array(dtype=int32),
    problem_vio: wp.array(dtype=int32),
    problem_P: wp.array(dtype=float32),
    # Outputs:
    solver_x: wp.array(dtype=float32),
    solver_y: wp.array(dtype=float32),
    solver_z: wp.array(dtype=float32),
):
    # Retrieve the thread index
    wid, tid = wp.tid()

    # Retrieve the number of active constraints in the world
    ncts = problem_dim[wid]

    # Skip if row index exceed the problem size
    if tid >= ncts:
        return

    # Retrieve the vector index offset of the world
    vio = problem_vio[wid]

    # Compute the global index of the vector entry
    v_i = vio + tid

    # Retrieve the i-th entries of the target vectors
    x_i = solver_x[v_i]
    y_i = solver_y[v_i]
    z_i = solver_z[v_i]

    # Retrieve the i-th entry of the diagonal preconditioner
    P_i = problem_P[v_i]

    # Store the preconditioned i-th entry of the vector
    solver_x[v_i] = P_i * x_i
    solver_y[v_i] = P_i * y_i
    solver_z[v_i] = (1.0 / P_i) * z_i


@wp.kernel
def _apply_dual_preconditioner_to_solution(
    # Inputs:
    problem_dim: wp.array(dtype=int32),
    problem_vio: wp.array(dtype=int32),
    problem_P: wp.array(dtype=float32),
    # Outputs:
    solution_lambdas: wp.array(dtype=float32),
    solution_v_plus: wp.array(dtype=float32),
):
    # Retrieve the thread index
    wid, tid = wp.tid()

    # Retrieve the number of active constraints in the world
    ncts = problem_dim[wid]

    # Skip if row index exceed the problem size
    if tid >= ncts:
        return

    # Retrieve the vector index offset of the world
    vio = problem_vio[wid]

    # Compute the global index of the vector entry
    v_i = vio + tid

    # Retrieve the i-th entries of the target vectors
    lambdas_i = solution_lambdas[v_i]
    v_plus_i = solution_v_plus[v_i]

    # Retrieve the i-th entry of the diagonal preconditioner
    P_i = problem_P[v_i]

    # Store the preconditioned i-th entry of the vector
    solution_lambdas[v_i] = (1.0 / P_i) * lambdas_i
    solution_v_plus[v_i] = P_i * v_plus_i


@wp.kernel
def _compute_final_desaxce_correction(
    problem_nc: wp.array(dtype=int32),
    problem_cio: wp.array(dtype=int32),
    problem_ccgo: wp.array(dtype=int32),
    problem_vio: wp.array(dtype=int32),
    problem_mu: wp.array(dtype=float32),
    solver_z: wp.array(dtype=float32),
    # Outputs:
    solver_s: wp.array(dtype=float32),
):
    # Retrieve the thread index
    wid, cid = wp.tid()

    # Retrieve the number of contact active in the world
    nc = problem_nc[wid]

    # Retrieve the limit constraint group offset of the world
    ccgo = problem_ccgo[wid]

    # Skip if row index exceed the problem size or if the solver has already converged
    if cid >= nc:
        return

    # Retrieve the index offset of the vector block of the world
    cio = problem_cio[wid]

    # Retrieve the index offset of the vector block of the world
    vio = problem_vio[wid]

    # Compute the vector index offset of the corresponding contact constraint
    ccio_k = vio + ccgo + 3 * cid

    # Compute the norm of the tangential components
    vtx = solver_z[ccio_k]
    vty = solver_z[ccio_k + 1]
    vt_norm = wp.sqrt(vtx * vtx + vty * vty)

    # Store De Saxce correction for this block
    solver_s[ccio_k] = 0.0
    solver_s[ccio_k + 1] = 0.0
    solver_s[ccio_k + 2] = problem_mu[cio + cid] * vt_norm


@wp.kernel
def _compute_solution_vectors(
    # Inputs:
    problem_dim: wp.array(dtype=int32),
    problem_vio: wp.array(dtype=int32),
    solver_s: wp.array(dtype=float32),
    solver_y: wp.array(dtype=float32),
    solver_z: wp.array(dtype=float32),
    # Outputs:
    solver_v_plus: wp.array(dtype=float32),
    solver_lambdas: wp.array(dtype=float32),
):
    # Retrieve the thread index
    wid, tid = wp.tid()

    # Retrieve the total number of active constraints in the world
    ncts = problem_dim[wid]

    # Skip if row index exceed the problem size or if the solver has already converged
    if tid >= ncts:
        return

    # Retrieve the index offset of the vector block of the world
    vector_offset = problem_vio[wid]

    # Compute the index offset of the vector block of the world
    thread_offset = vector_offset + tid

    # Retrieve the solver state
    z = solver_z[thread_offset]
    s = solver_s[thread_offset]
    y = solver_y[thread_offset]

    # Update constraint velocity: v_plus = z - s;
    solver_v_plus[thread_offset] = z - s

    # Update constraint reactions: lambda = y
    solver_lambdas[thread_offset] = y
