# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""
Provides a data container and relevant operations to
represent and construct a dual forward dynamics problem.

The dual forward dynamics problem arises from the formulation of
the equations of motion in terms of constraint reactions.

`lambdas = argmin_{x} 1/2 * x^T D x + lambda^T (v_f + Gamma(v_plus(x)))`


This module thus provides building-blocks to realize Delassus operators across multiple
worlds contained in a :class:`ModelKamino`. The :class:`DelassusOperator` class provides a
high-level interface to encapsulate both the data representation as well as the
relevant operations. It provides methods to allocate the necessary data arrays, build
the Delassus matrix given the current state of the model and the active constraints,
add diagonal regularization, and solve linear systems of the form `D @ x = v` given
arrays holding the right-hand-side (rhs) vectors v. Moreover, it supports the use of
different linear solvers as a back-end for performing the aforementioned linear system
solve. Construction of the Delassus operator is realized using a set of Warp kernels
that parallelize the computation using various strategies.

Typical usage example:
    # Create a model builder and add bodies, joints, geoms, etc.
    builder = ModelBuilder()
    ...

    # Create a model from the builder and construct additional
    # containers to hold joint-limits, contacts, Jacobians
    model = builder.finalize()
    data = model.data()
    limits = LimitsKamino(model)
    contacts = ContactsKamino(builder)
    jacobians = DenseSystemJacobians(model, limits, contacts)

    # Define a linear solver type to use as a back-end for the
    # Delassus operator computations such as factorization and
    # solving the linear system when a rhs vector is provided
    linear_solver = LLTBlockedSolver
    ...

    # Build the Jacobians for the model and active limits and contacts
    jacobians.build(model, data, limits, contacts)
    ...

    # Create a dual forward dynamics problem and build it using the current model
    # data and active unilateral constraints (i.e. for limits and contacts).
    dual = DualProblem(model, limits, contacts, jacobians, linear_solver)
    dual.build(model, data, jacobians)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import warp as wp

from .....core.types import override
from ...config import ConfigBase, ConstrainedDynamicsConfig, ConstraintStabilizationConfig
from ..core.data import DataKamino
from ..core.math import FLOAT32_EPS, UNIT_Z, screw, screw_angular, screw_linear
from ..core.model import ModelKamino
from ..core.size import SizeKamino
from ..core.types import float32, int32, mat33f, vec2f, vec3f, vec4f, vec6f
from ..dynamics.delassus import BlockSparseMatrixFreeDelassusOperator, DelassusOperator
from ..geometry.contacts import ContactsKamino
from ..kinematics.jacobians import DenseSystemJacobians, SparseSystemJacobians
from ..kinematics.limits import LimitsKamino
from ..linalg import LinearSolverType

###
# Module interface
###

__all__ = [
    "DualProblem",
    "DualProblemData",
]


###
# Module configs
###

wp.set_module_options({"enable_backward": False})


###
# Types
###


@wp.struct
class DualProblemConfigStruct:
    """
    A Warp struct to hold on-device configuration parameters of a dual problem.
    """

    alpha: float32
    """Baumgarte stabilization parameter for bilateral joint constraints."""
    beta: float32
    """Baumgarte stabilization parameter for unilateral joint limit constraints."""
    gamma: float32
    """Baumgarte stabilization parameter for unilateral contact constraints."""
    delta: float32
    """Contact penetration margin used for unilateral contact constraints"""
    preconditioning: wp.bool
    """Flag to enable preconditioning of the dual problem."""


@dataclass
class DualProblemData:
    """
    A container to hold the the dual forward dynamics problem data over multiple worlds.
    """

    num_worlds: int = 0
    """The number of worlds represented in the dual problem."""

    max_of_maxdims: int = 0
    """The largest maximum number of dual problem dimensions (i.e. constraints) across all worlds."""

    ###
    # Problem configurations
    ###

    config: wp.array | None = None
    """
    Problem configuration parameters for each world.\n
    Shape of `(num_worlds,)` and type :class:`DualProblemConfigStruct`.
    """

    ###
    # Constraints info
    ###

    njc: wp.array | None = None
    """
    The number of active joint constraints in each world.\n
    Shape of `(num_worlds,)` and type :class:`int32`.
    """

    nl: wp.array | None = None
    """
    The number of active limit constraints in each world.\n
    Shape of `(num_worlds,)` and type :class:`int32`.
    """

    nc: wp.array | None = None
    """
    The number of active contact constraints in each world.\n
    Shape of `(num_worlds,)` and type :class:`int32`.
    """

    lio: wp.array | None = None
    """
    The limit index offset of each world.\n
    Shape of `(num_worlds,)` and type :class:`int32`.\n
    """

    cio: wp.array | None = None
    """
    The contact index offset of each world.\n
    Shape of `(num_worlds,)` and type :class:`int32`.\n
    """

    uio: wp.array | None = None
    """
    The unilateral index offset of each world.\n
    Shape of `(num_worlds,)` and type :class:`int32`.\n
    """

    lcgo: wp.array | None = None
    """
    The limit constraint group offset of each world.\n
    Shape of `(num_worlds,)` and type :class:`int32`.\n
    """

    ccgo: wp.array | None = None
    """
    The contact constraint group offset of each world.\n
    Shape of `(num_worlds,)` and type :class:`int32`.\n
    """

    ###
    # Delassus operator
    ###

    maxdim: wp.array | None = None
    """
    The maximum number of dual problem dimensions of each world.\n
    Shape of `(num_worlds,)` and type :class:`int32`.
    """

    dim: wp.array | None = None
    """
    The active number of dual problem dimensions of each world.\n
    Shape of `(num_worlds,)` and type :class:`int32`.
    """

    mio: wp.array | None = None
    """
    The matrix index offset of each Delassus matrix block.\n
    This is applicable to `D` as well as to its (optional) factorizations.\n
    Shape of `(num_worlds,)` and type :class:`int32`.
    """

    vio: wp.array | None = None
    """
    The vector index offset of each constraint dimension vector block.\n
    This is applicable to `v_b`, `v_i` and `v_f`.\n
    Shape of `(num_worlds,)` and type :class:`int32`.
    """

    D: wp.array | None = None
    """
    The flat array of Delassus matrix blocks (constraint-space apparent inertia).\n
    Shape of `(sum_of_max_total_delassus_size,)` and type :class:`float32`.
    """

    P: wp.array | None = None
    """
    The flat array of Delassus diagonal preconditioner blocks.\n
    Shape of `(sum_of_max_total_cts,)` and type :class:`float32`.
    """

    ###
    # Problem vectors
    ###

    h: wp.array | None = None
    """
    Stack of non-linear generalized forces vectors of each world.\n

    Computed as:
    `h = dt * (w_e + w_gc + w_a)`

    where:
    - `dt` is the simulation time step
    - `w_e` is the stack of per-body purely external wrenches
    - `w_gc` is the stack of per-body gravitational + Coriolis wrenches
    - `w_a` is the stack of per-body jointactuation wrenches

    Construction of this term is optional, as it's contributions are already
    incorporated in the computation of the generalized free-velocity `u_f`.
    It is can be optionally built for analysis or debugging purposes.

    Shape of `(sum_of_num_body_dofs,)` and type :class:`vec6f`.
    """

    u_f: wp.array | None = None
    """
    Stack of unconstrained generalized velocity vectors.\n

    Computed as:
    `u_f = u_minus + dt * M^{-1} @ h`

    where:
    - `u_minus` is the stack of per-body generalized velocities at the beginning of the time step
    - `M^{-1}` is the block-diagonal inverse generalized mass matrix
    - `h` is the stack of non-linear generalized forces vectors

    Shape of `(sum_of_num_body_dofs,)` and type :class:`vec6f`.
    """

    v_b: wp.array | None = None
    """
    Stack of free-velocity constraint bias vectors (in constraint-space).\n

    Computed as:
    `v_b = [ v_b_dynamics;
             alpha * inv_dt * r_joints;
             beta * inv_dt * r_limits;
             gamma * inv_dt * r_contacts ]`

    where:
    - `v_b_dynamics` is the joint dynamics velocity bias.
    - `dt` and `inv_dt` is the simulation time step and it inverse
    - `r_joints` is the stack of joint constraint residuals
    - `r_limits` is the stack of limit constraint residuals
    - `r_contacts` is the stack of contact constraint residuals
    - `alpha`, `beta`, `gamma` are the Baumgarte stabilization
        parameters for joints, limits and contacts, respectively

    Shape of `(sum_of_max_total_cts,)` and type :class:`float32`.
    """

    v_i: wp.array | None = None
    """
    The stack of free-velocity impact biases vector (in constraint-space).\n

    Computed as:
    `v_i = epsilon @ (J_cts @ u_minus)`

    where:
    - `epsilon` is the stack of per-contact restitution coefficients
    - `J_cts` is the constraint Jacobian matrix
    - `u_minus` is the stack of per-body generalized velocities at the beginning of the time step

    Shape of `(sum_of_max_total_cts,)` and type :class:`float32`.
    """

    v_f: wp.array | None = None
    """
    Stack of free-velocity vectors (constraint-space unconstrained velocity).\n

    Computed as:
    `v_f = J_cts @ u_f + v_b + v_i`

    where:
    - `J_cts` is the constraint Jacobian matrix
    - `u_f` is the stack of unconstrained generalized velocity vectors
    - `v_b` is the stack of free-velocity stabilization biases vectors
    - `v_i` is the stack of free-velocity impact biases vectors

    Shape of `(sum_of_max_total_cts,)` and type :class:`float32`.
    """

    mu: wp.array | None = None
    """
    Stack of per-contact constraint friction coefficient vectors.\n
    Shape of `(sum_of_max_contacts,)` and type :class:`float32`.
    """


###
# Functions
###


@wp.func
def gravity_plus_coriolis_wrench(
    g: vec3f,
    m_i: float32,
    I_i: mat33f,
    omega_i: vec3f,
) -> vec6f:
    """
    Compute the gravitational + Coriolis wrench acting on a body.
    """
    f_gi_i = m_i * g
    tau_gi_i = -wp.skew(omega_i) @ (I_i @ omega_i)
    return screw(f_gi_i, tau_gi_i)


@wp.func
def gravity_plus_coriolis_wrench_split(
    g: vec3f,
    m_i: float32,
    I_i: mat33f,
    omega_i: vec3f,
):
    """
    Compute the gravitational+inertial wrench on a body.
    """
    f_gi_i = m_i * g
    tau_gi_i = -wp.skew(omega_i) @ (I_i @ omega_i)
    return f_gi_i, tau_gi_i


###
# Kernels
###


@wp.kernel
def _build_nonlinear_generalized_force(
    # Inputs:
    model_time_dt: wp.array[float32],
    model_gravity_vector: wp.array[vec4f],
    model_bodies_wid: wp.array[int32],
    model_bodies_m_i: wp.array[float32],
    state_bodies_u_i: wp.array[vec6f],
    state_bodies_I_i: wp.array[mat33f],
    state_bodies_w_e_i: wp.array[vec6f],
    state_bodies_w_a_i: wp.array[vec6f],
    # Outputs:
    problem_h: wp.array[vec6f],
):
    # Retrieve the body index as the thread index
    bid = wp.tid()

    # Retrieve the body model and data
    wid = model_bodies_wid[bid]
    m_i = model_bodies_m_i[bid]
    I_i = state_bodies_I_i[bid]
    u_i = state_bodies_u_i[bid]
    w_e_i = state_bodies_w_e_i[bid]
    w_a_i = state_bodies_w_a_i[bid]

    # Get world data
    dt = model_time_dt[wid]
    gv = model_gravity_vector[wid]

    # Extract the effective gravity vector
    g = gv.w * vec3f(gv.x, gv.y, gv.z)

    # Extract the linear and angular components of the generalized velocity
    omega_i = screw_angular(u_i)

    # Compute the net external wrench on the body
    h_i = w_e_i + w_a_i + gravity_plus_coriolis_wrench(g, m_i, I_i, omega_i)

    # Store the generalized free-velocity vector
    problem_h[bid] = dt * h_i


@wp.kernel
def _build_generalized_free_velocity(
    # Inputs:
    model_time_dt: wp.array[float32],
    model_gravity_vector: wp.array[vec4f],
    model_bodies_wid: wp.array[int32],
    model_bodies_m_i: wp.array[float32],
    model_bodies_inv_m_i: wp.array[float32],
    state_bodies_u_i: wp.array[vec6f],
    state_bodies_I_i: wp.array[mat33f],
    state_bodies_inv_I_i: wp.array[mat33f],
    state_bodies_w_e_i: wp.array[vec6f],
    state_bodies_w_a_i: wp.array[vec6f],
    # Outputs:
    problem_u_f: wp.array[vec6f],
):
    # Retrieve the body index as the thread index
    bid = wp.tid()

    # Retrieve the body model and data
    wid = model_bodies_wid[bid]
    m_i = model_bodies_m_i[bid]
    I_i = state_bodies_I_i[bid]
    inv_m_i = model_bodies_inv_m_i[bid]
    inv_I_i = state_bodies_inv_I_i[bid]
    u_i = state_bodies_u_i[bid]
    w_e_i = state_bodies_w_e_i[bid]
    w_a_i = state_bodies_w_a_i[bid]

    # Get world data
    dt = model_time_dt[wid]
    gv = model_gravity_vector[wid]

    # Extract the effective gravity vector
    g = gv.w * vec3f(gv.x, gv.y, gv.z)

    # Extract the linear and angular components of the generalized velocity
    v_i = screw_linear(u_i)
    omega_i = screw_angular(u_i)

    # Compute the net external wrench on the body
    h_i = w_e_i + w_a_i + gravity_plus_coriolis_wrench(g, m_i, I_i, omega_i)
    f_h_i = screw_linear(h_i)
    tau_h_i = screw_angular(h_i)

    # Compute the generalized free-velocity vector components
    v_f_i = v_i + dt * (inv_m_i * f_h_i)
    omega_f_i = omega_i + dt * (inv_I_i @ tau_h_i)

    # Store the generalized free-velocity vector
    problem_u_f[bid] = screw(v_f_i, omega_f_i)


@wp.kernel
def _build_free_velocity_bias_joint_dynamics(
    # Inputs:
    model_joints_wid: wp.array[int32],
    model_joints_dynamic_cts_offset: wp.array[int32],
    model_joints_dynamic_cts_offset_total_cts: wp.array[int32],
    data_joints_dq_b_j: wp.array[float32],
    # Outputs:
    problem_v_b: wp.array[float32],
):
    # Retrieve the joint index as the thread index
    jid = wp.tid()

    # Retrieve the joint constraints size + index offset into the dynamic-only constraints array
    bias_row_start_j = model_joints_dynamic_cts_offset[jid]
    num_dyn_cts_j = model_joints_dynamic_cts_offset[jid + 1] - bias_row_start_j

    # Skip operation if the joint has no dynamic constraints
    if num_dyn_cts_j == 0:
        return

    # Retrieve the joint constraints index offset into the full constraints array
    cts_row_start_j = model_joints_dynamic_cts_offset_total_cts[jid]

    # Compute the free-velocity bias for the joint
    for j in range(num_dyn_cts_j):
        problem_v_b[cts_row_start_j + j] = -data_joints_dq_b_j[bias_row_start_j + j]


@wp.kernel
def _build_free_velocity_bias_joint_kinematics(
    # Inputs:
    model_time_inv_dt: wp.array[float32],
    model_joints_wid: wp.array[int32],
    model_joints_kinematic_cts_offset: wp.array[int32],
    model_joints_kinematic_cts_offset_total_cts: wp.array[int32],
    data_joints_r_j: wp.array[float32],
    problem_config: wp.array[DualProblemConfigStruct],
    # Outputs:
    problem_v_b: wp.array[float32],
):
    # Retrieve the joint index as the thread index
    jid = wp.tid()

    # Retrieve the joint constraints size + index offset into the kinematic-only constraints array
    res_row_start_j = model_joints_kinematic_cts_offset[jid]
    num_kin_cts_j = model_joints_kinematic_cts_offset[jid + 1] - res_row_start_j

    # Retrieve the joint constraints index offset into the full constraints array
    cts_row_start_j = model_joints_kinematic_cts_offset_total_cts[jid]

    # Retrieve the world index
    wid = model_joints_wid[jid]

    # Retrieve the model time step
    inv_dt = model_time_inv_dt[wid]

    # Retrieve the dual problem config
    config = problem_config[wid]

    # Compute baumgarte constraint stabilization coefficient
    c_b = config.alpha * inv_dt

    # Compute the free-velocity bias for the joint
    for j in range(num_kin_cts_j):
        problem_v_b[cts_row_start_j + j] = c_b * data_joints_r_j[res_row_start_j + j]


@wp.kernel
def _build_free_velocity_bias_limits(
    # Inputs:
    model_time_inv_dt: wp.array[float32],
    data_info_limit_cts_group_offset: wp.array[int32],
    limits_model_max: int32,
    limits_model_num: wp.array[int32],
    limits_wid: wp.array[int32],
    limits_lid: wp.array[int32],
    limits_r_q: wp.array[float32],
    problem_config: wp.array[DualProblemConfigStruct],
    problem_vio: wp.array[int32],
    # Outputs:
    problem_v_b: wp.array[float32],
):
    # Retrieve the limit index as the thread index
    tid = wp.tid()

    # Retrieve the number of contacts active in the model
    model_nl = wp.min(limits_model_num[0], limits_model_max)

    # Skip if cid is greater than the number of contacts active in the world
    if tid >= model_nl:
        return

    # Retrieve the limit entity data
    wid = limits_wid[tid]
    lid = limits_lid[tid]
    r_q = limits_r_q[tid]

    # Retrieve the world-specific data
    inv_dt = model_time_inv_dt[wid]
    config = problem_config[wid]
    vio = problem_vio[wid]
    lcio = data_info_limit_cts_group_offset[wid]

    # Compute the total constraint index offset of the current contact
    lcio_l = vio + lcio + lid

    # Compute the contact constraint stabilization bias
    problem_v_b[lcio_l] = config.beta * inv_dt * wp.min(0.0, r_q)


@wp.kernel
def _build_free_velocity_bias_contacts(
    # Inputs:
    model_time_inv_dt: wp.array[float32],
    model_info_contacts_offset: wp.array[int32],
    data_info_contact_cts_group_offset: wp.array[int32],
    contacts_model_max: int32,
    contacts_model_num: wp.array[int32],
    contacts_wid: wp.array[int32],
    contacts_cid: wp.array[int32],
    contacts_gapfunc: wp.array[vec4f],
    contacts_material: wp.array[vec2f],
    problem_config: wp.array[DualProblemConfigStruct],
    problem_vio: wp.array[int32],
    # Outputs:
    problem_v_b: wp.array[float32],
    problem_v_i: wp.array[float32],
    problem_mu: wp.array[float32],
):
    # Retrieve the contact index as the thread index
    tid = wp.tid()

    # Retrieve the number of contacts active in the model
    model_nc = wp.min(contacts_model_num[0], contacts_model_max)

    # Skip if cid is greater than the number of contacts active in the world
    if tid >= model_nc:
        return

    # Retrieve the contact entity data
    wid_k = contacts_wid[tid]
    cid_k = contacts_cid[tid]
    material_k = contacts_material[tid]
    penetration_k = contacts_gapfunc[tid][3]

    # Retrieve the world-specific data
    inv_dt = model_time_inv_dt[wid_k]
    cio = model_info_contacts_offset[wid_k]
    ccio = data_info_contact_cts_group_offset[wid_k]
    vio = problem_vio[wid_k]
    config = problem_config[wid_k]

    # Compute the total constraint index offset of the current contact
    ccio_k = vio + ccio + 3 * cid_k

    # Compute the total contact index offset of the current contact
    cio_k = cio + cid_k

    # Retrieve the contact material properties
    mu_k = material_k.x  # Friction coefficient
    epsilon_k = material_k.y  # Penetration reduction coefficient

    # The gap-function value (penetration_k) is the margin-shifted signed
    # distance: negative means penetration past the resting separation,
    # zero means at rest, positive means within the detection gap.
    # Pass the full value through so that one-sided Baumgarte stabilization
    # (xi_relaxed) can generate a positive bias for gap contacts, guiding
    # objects toward d = 0.  A dead-zone of config.delta filters out
    # floating-point noise on nearly-touching contacts that would otherwise
    # destabilize accelerated solvers (e.g. Nesterov momentum in PADMM).
    distance_k = wp.where(wp.abs(penetration_k) < config.delta, 0.0, penetration_k)

    # Compute the per-contact penetration error reduction term
    # NOTE#1: Penetrations are represented as xi < 0 (hence the sign inversion)
    # NOTE#2: xi_p_relaxed corresponds to one-sided Baumgarte-like stabilization
    xi = inv_dt * distance_k
    xi_relaxed = config.gamma * wp.min(0.0, xi) + wp.max(0.0, xi)
    if epsilon_k == 1.0:
        alpha = 0.0
    else:
        alpha = 1.0

    # Compute the contact constraint stabilization bias
    v_b_k = alpha * xi_relaxed * UNIT_Z

    # Store the contact constraint stabilization bias in the output vector
    for i in range(3):
        problem_v_b[ccio_k + i] = v_b_k[i]

    # Initialize the restitutive Newton-type impact model term
    problem_v_i[ccio_k] = 0.0
    problem_v_i[ccio_k + 1] = 0.0
    problem_v_i[ccio_k + 2] = epsilon_k

    # Store the contact friction coefficient in the output vector
    problem_mu[cio_k] = mu_k


@wp.kernel
def _build_free_velocity(
    # Inputs:
    model_info_bodies_offset: wp.array[int32],
    data_bodies_u_i: wp.array[vec6f],
    jacobians_J_cts_offsets: wp.array[int32],
    jacobians_J_cts_data: wp.array[float32],
    problem_dim: wp.array[int32],
    problem_vio: wp.array[int32],
    problem_u_f: wp.array[vec6f],
    problem_v_b: wp.array[float32],
    problem_v_i: wp.array[float32],
    # Outputs:
    problem_v_f: wp.array[float32],
):
    # Retrieve the thread index
    wid, tid = wp.tid()

    # Retrieve the problem dimensions and matrix block index offset
    ncts = problem_dim[wid]

    # Skip if row index exceed the problem size
    if tid >= ncts:
        return

    # Retrieve the world-specific data
    bio = model_info_bodies_offset[wid]
    nb = model_info_bodies_offset[wid + 1] - bio
    cjmio = jacobians_J_cts_offsets[wid]
    vio = problem_vio[wid]

    # Compute the number of Jacobian rows, i.e. the number of body DoFs
    nbd = 6 * nb

    # Compute the thread-specific index offset
    cts_offset = vio + tid

    # Append the column offset to the Jacobian index
    cjmio += nbd * tid

    # Extract the cached impact bias scaling (i.e. restitution coefficient)
    # NOTE: This is a quick hack to avoid multiple kernels. The
    # proper way would be to perform this op only for contacts
    epsilon_j = problem_v_i[cts_offset]

    # Retrieve the cached velocity bias term for the constraint
    v_b_j = problem_v_b[cts_offset]

    # Buffers
    J_i = vec6f(0.0)
    v_f_j = float32(0.0)

    # Iterate over each body to accumulate velocity contributions
    for i in range(nb):
        # Compute the Jacobian block index
        m_ji = cjmio + 6 * i

        # Extract the twist and unconstrained velocity of the body
        u_i = data_bodies_u_i[bio + i]
        u_f_i = problem_u_f[bio + i]

        # Extract the Jacobian block J_ji
        # TODO: use slicing operation when available
        for d in range(6):
            J_i[d] = jacobians_J_cts_data[m_ji + d]

        # Accumulate J_i @ u_i
        v_f_j += wp.dot(J_i, u_f_i)

        # Accumulate the impact bias term
        v_f_j += epsilon_j * wp.dot(J_i, u_i)

    # Store sum of velocity bias terms
    problem_v_f[cts_offset] = v_f_j + v_b_j


@wp.kernel
def _build_free_velocity_sparse(
    # Inputs:
    model_info_bodies_offset: wp.array[int32],
    state_bodies_u_i: wp.array[vec6f],
    jac_num_nzb: wp.array[int32],
    jac_nzb_start: wp.array[int32],
    jac_nzb_coords: wp.array2d[int32],
    jac_nzb_values: wp.array[vec6f],
    problem_vio: wp.array[int32],
    problem_u_f: wp.array[vec6f],
    problem_v_i: wp.array[float32],
    # Outputs:
    problem_v_f: wp.array[float32],
):
    # Retrieve the thread index
    wid, nzb_id = wp.tid()

    # Skip if block index exceed the number of blocks
    if nzb_id >= jac_num_nzb[wid]:
        return

    # Retrieve block data
    global_block_idx = jac_nzb_start[wid] + nzb_id
    jac_block_coord = jac_nzb_coords[global_block_idx]
    jac_block = jac_nzb_values[global_block_idx]

    # Retrieve the world-specific data
    bio = model_info_bodies_offset[wid]
    vio = problem_vio[wid]

    # Compute the thread-specific index offset
    thread_offset = vio + jac_block_coord[0]

    # Extract the cached impact bias scaling (i.e. restitution coefficient)
    # NOTE: This is a quick hack to avoid multiple kernels. The
    # proper way would be to perform this op only for contacts
    epsilon_j = problem_v_i[thread_offset]

    # Buffers
    v_f_j = float32(0.0)

    # Iterate over each body to accumulate velocity contributions
    bid = jac_block_coord[1] // 6

    # Extract the twist and unconstrained velocity of the body
    u_i = state_bodies_u_i[bio + bid]
    u_f_i = problem_u_f[bio + bid]

    # Accumulate J_i @ u_i
    v_f_j += wp.dot(jac_block, u_f_i)

    # Accumulate the impact bias term
    v_f_j += epsilon_j * wp.dot(jac_block, u_i)

    # Store sum of velocity bias terms
    wp.atomic_add(problem_v_f, thread_offset, v_f_j)


@wp.kernel
def _build_dual_preconditioner_all_constraints(
    # Inputs:
    problem_config: wp.array[DualProblemConfigStruct],
    problem_dim: wp.array[int32],
    problem_mio: wp.array[int32],
    problem_vio: wp.array[int32],
    problem_njc: wp.array[int32],
    problem_nl: wp.array[int32],
    problem_D: wp.array[float32],
    # Outputs:
    problem_P: wp.array[float32],
):
    # Retrieve the thread index
    wid, tid = wp.tid()

    # Retrieve the world-specific problem config
    config = problem_config[wid]

    # Retrieve the number of active constraints in the world
    ncts = problem_dim[wid]

    # Skip if row index exceed the problem size
    if tid >= ncts or not config.preconditioning:
        return

    # Retrieve the matrix index offset of the world
    mio = problem_mio[wid]

    # Retrieve the vector index offset of the world
    vio = problem_vio[wid]

    # Retrieve the number of active joint and limit constraints of the world
    njc = problem_njc[wid]
    nl = problem_nl[wid]
    njlc = njc + nl

    # Compute the preconditioner entry for the current constraint
    # First handle joint and limit constraints, then contact constraints
    if tid < njlc:
        # Retrieve the diagonal entry of the Delassus matrix
        D_ii = problem_D[mio + ncts * tid + tid]
        # Compute the corresponding Jacobi preconditioner entry
        problem_P[vio + tid] = wp.sqrt(1.0 / (wp.abs(D_ii) + FLOAT32_EPS))
    else:
        # Compute the contact constraint index
        ccid = tid - njlc
        # Only the thread of the first contact constraint dimension computes the preconditioner
        if ccid % 3 == 0:
            # Retrieve the diagonal entries of the Delassus matrix for the contact constraint set
            D_kk_0 = problem_D[mio + ncts * (tid + 0) + (tid + 0)]
            D_kk_1 = problem_D[mio + ncts * (tid + 1) + (tid + 1)]
            D_kk_2 = problem_D[mio + ncts * (tid + 2) + (tid + 2)]
            # Compute the effective diagonal entry
            # D_kk = (D_kk_0 + D_kk_1 + D_kk_2) / 3.0
            # D_kk = wp.min(vec3f(D_kk_0, D_kk_1, D_kk_2))
            D_kk = wp.max(vec3f(D_kk_0, D_kk_1, D_kk_2))
            # Compute the corresponding Jacobi preconditioner entry
            P_k = wp.sqrt(1.0 / (wp.abs(D_kk) + FLOAT32_EPS))
            problem_P[vio + tid] = P_k
            problem_P[vio + tid + 1] = P_k
            problem_P[vio + tid + 2] = P_k


@wp.kernel
def _build_dual_preconditioner_all_constraints_sparse(
    # Inputs:
    problem_config: wp.array[DualProblemConfigStruct],
    problem_dim: wp.array[int32],
    problem_vio: wp.array[int32],
    problem_njc: wp.array[int32],
    problem_nl: wp.array[int32],
    # Outputs:
    problem_P: wp.array[float32],
):
    # Retrieve the thread index
    wid, tid = wp.tid()

    # Retrieve the world-specific problem config
    config = problem_config[wid]

    # Retrieve the number of active constraints in the world
    ncts = problem_dim[wid]

    # Skip if row index exceed the problem size
    if tid >= ncts or not config.preconditioning:
        return

    # Retrieve the vector index offset of the world
    vio = problem_vio[wid]

    # Retrieve the number of active joint and limit constraints of the world
    njc = problem_njc[wid]
    nl = problem_nl[wid]
    njlc = njc + nl

    # Compute the preconditioner entry for the current constraint
    # First handle joint and limit constraints, then contact constraints
    if tid < njlc:
        # Retrieve the diagonal entry of the Delassus matrix
        D_ii = problem_P[vio + tid]
        # Compute the corresponding Jacobi preconditioner entry
        problem_P[vio + tid] = wp.sqrt(1.0 / (wp.abs(D_ii) + FLOAT32_EPS))
    else:
        # Compute the contact constraint index
        ccid = tid - njlc
        # Only the thread of the first contact constraint dimension computes the preconditioner
        if ccid % 3 == 0:
            # Retrieve the diagonal entries of the Delassus matrix for the contact constraint set
            D_kk_0 = problem_P[vio + tid]
            D_kk_1 = problem_P[vio + tid + 1]
            D_kk_2 = problem_P[vio + tid + 2]
            # Compute the effective diagonal entry
            # D_kk = (D_kk_0 + D_kk_1 + D_kk_2) / 3.0
            # D_kk = wp.min(vec3f(D_kk_0, D_kk_1, D_kk_2))
            D_kk = wp.max(vec3f(D_kk_0, D_kk_1, D_kk_2))
            # Compute the corresponding Jacobi preconditioner entry
            P_k = wp.sqrt(1.0 / (wp.abs(D_kk) + FLOAT32_EPS))
            problem_P[vio + tid] = P_k
            problem_P[vio + tid + 1] = P_k
            problem_P[vio + tid + 2] = P_k


@wp.kernel
def _apply_dual_preconditioner_to_matrix(
    # Inputs:
    problem_dim: wp.array[int32],
    problem_mio: wp.array[int32],
    problem_vio: wp.array[int32],
    problem_P: wp.array[float32],
    # Outputs:
    X: wp.array[float32],
):
    # Retrieve the thread index
    wid, tid = wp.tid()

    # Retrieve the number of active constraints in the world
    ncts = problem_dim[wid]

    # Skip if there no constraints ar active
    if ncts == 0:
        return

    # Compute i (row) and j (col) indices from the tid
    i = tid // ncts
    j = tid % ncts

    # Skip if indices exceed the problem size
    if i >= ncts or j >= ncts:
        return

    # Retrieve the matrix index offset of the world
    mio = problem_mio[wid]

    # Retrieve the vector index offset of the world
    vio = problem_vio[wid]

    # Compute the global index of the matrix entry
    m_ij = mio + ncts * i + j

    # Retrieve the i,j-th entry of the target matrix
    X_ij = X[m_ij]

    # Retrieve the i,j-th entries of the diagonal preconditioner
    P_i = problem_P[vio + i]
    P_j = problem_P[vio + j]

    # Store the preconditioned i,j-th entry of the matrix
    X[m_ij] = P_i * (P_j * X_ij)


@wp.kernel
def _apply_dual_preconditioner_to_vector(
    # Inputs:
    problem_dim: wp.array[int32],
    problem_vio: wp.array[int32],
    problem_P: wp.array[float32],
    # Outputs:
    x: wp.array[float32],
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

    # Retrieve the i-th entry of the target vector
    x_i = x[v_i]

    # Retrieve the i-th entry of the diagonal preconditioner
    P_i = problem_P[v_i]

    # Store the preconditioned i-th entry of the vector
    x[v_i] = P_i * x_i


###
# Interfaces
###


class DualProblem:
    """
    A container to hold, manage and operate a dynamics dual problem.
    """

    @dataclass
    class Config(ConfigBase):
        """
        Configuration class for :class:`DualProblem`.
        """

        constraints: ConstraintStabilizationConfig = field(default_factory=ConstraintStabilizationConfig)
        """Constraint stabilization global defaults/override configurations."""

        dynamics: ConstrainedDynamicsConfig = field(default_factory=ConstrainedDynamicsConfig)
        """Constrained dynamics problem construction configurations."""

        def to_struct(self) -> DualProblemConfigStruct:
            """
            Converts the config to a DualProblemConfigStruct struct.
            """
            config_struct = DualProblemConfigStruct()
            config_struct.alpha = wp.float32(self.constraints.alpha)
            config_struct.beta = wp.float32(self.constraints.beta)
            config_struct.gamma = wp.float32(self.constraints.gamma)
            config_struct.delta = wp.float32(self.constraints.delta)
            config_struct.preconditioning = wp.bool(self.dynamics.preconditioning)
            return config_struct

        @override
        def validate(self) -> None:
            """
            Validates the current values held by the :class:`DualProblem.Config` instance.
            """
            self.constraints.validate()
            self.dynamics.validate()

        @override
        def __post_init__(self):
            """Post-initialization to validate config."""
            self.validate()

    def __init__(
        self,
        model: ModelKamino | None = None,
        data: DataKamino | None = None,
        limits: LimitsKamino | None = None,
        contacts: ContactsKamino | None = None,
        jacobians: SparseSystemJacobians | DenseSystemJacobians | None = None,
        solver: LinearSolverType | None = None,
        solver_kwargs: dict[str, Any] | None = None,
        config: list[DualProblem.Config] | DualProblem.Config | None = None,
        compute_h: bool = False,
        sparse: bool = True,
    ):
        """
        Constructs a dual problem interface container.

        If `model`, `limits` and/or `contacts` containers are provided, it allocates the dual problem data members.
        Only the `model` is strictly required for the allocation, but the resulting dual problem will only represent
        bilateral (i.e. equality) joint constraints and possibly some unilateral (i.e. inequality) joint limits, but
        not contact constraints. The `contacts` container is required if the dual problem is to also incorporate
        contact constraints. If no `model` is provided at construction time, then deferred allocation is possible
        by calling the `finalize()` method at a later point.

        Args:
            model (ModelKamino, optional):
                The model to build the dual problem for.
            contacts (ContactsKamino, optional):
                The contacts container to use for the dual problem.
            jacobians (SparseSystemJacobians | DenseSystemJacobians, optional):
                The constraints Jacobians for this model. Must be provided if model is provided.
            solver (LinearSolverType, optional):
                The linear solver to use for the Delassus operator. Defaults to None.
            config (List[DualProblem.Config] | DualProblem.Config, optional):
                The config for the dual problem.\n
                If a single `DualProblem.Config` object is provided, it will be replicated for all worlds.
                Defaults to `None`, indicating that default config will be used for all worlds.
            compute_h (bool, optional):
                Set to `True` to enable the computation of the nonlinear
                generalized forces vectors in construction of the dual problem.\n
                Defaults to `False`.
        """
        # Ensure Jacobians are given if model is provided.
        if model is not None and jacobians is None:
            raise ValueError("`jacobians` parameter must be provided if `model` parameter is specified.")

        # Declare the device cache
        self._device: wp.DeviceLike = None

        # Declare the model size cache
        self._size: SizeKamino | None = None

        self._config: list[DualProblem.Config] = []
        """Host-side cache of the list of per world dual problem config."""

        self._delassus: DelassusOperator | BlockSparseMatrixFreeDelassusOperator | None = None
        """The Delassus operator interface container."""

        self._data: DualProblemData | None = None
        """The dual problem data container bundling are relevant memory allocations."""

        self._sparse: bool = sparse
        """Flag to indicate whether the dual uses a sparse data representation."""

        # Finalize the dual problem data if a model is provided
        if model is not None:
            self.finalize(
                model=model,
                data=data,
                limits=limits,
                contacts=contacts,
                jacobians=jacobians,
                solver=solver,
                solver_kwargs=solver_kwargs,
                config=config,
                compute_h=compute_h,
            )

    ###
    # Properties
    ###

    @property
    def device(self) -> wp.DeviceLike:
        """
        Returns the device the dual problem is allocated on.
        """
        return self._device

    @property
    def size(self) -> SizeKamino:
        """
        Returns the model size of the dual problem.
        This is the size of the model that the dual problem is built for.
        """
        if self._size is None:
            raise ValueError("ModelKamino size is not allocated. Call `finalize()` first.")
        return self._size

    @property
    def config(self) -> list[DualProblem.Config]:
        """
        Returns the list of per world dual problem config.
        """
        return self._config

    @config.setter
    def config(self, value: list[DualProblem.Config] | DualProblem.Config):
        """
        Sets the list of per world dual problem config.
        If a single `DualProblem.Config` object is provided, it will be replicated for all worlds.
        """
        self._config = self._check_config(value, self._data.num_worlds)

    @property
    def delassus(self) -> DelassusOperator | BlockSparseMatrixFreeDelassusOperator:
        """
        Returns the Delassus operator interface.
        """
        if self._delassus is None:
            raise ValueError("Delassus operator is not allocated. Call `finalize()` first.")
        return self._delassus

    @property
    def data(self) -> DualProblemData:
        """
        Returns the dual problem data container.
        """
        return self._data

    @property
    def sparse(self) -> bool:
        """
        Returns whether the dual problem is using sparse operators.
        """
        return self._sparse

    ###
    # Operations
    ###

    def finalize(
        self,
        model: ModelKamino,
        jacobians: SparseSystemJacobians | DenseSystemJacobians,
        data: DataKamino | None = None,
        limits: LimitsKamino | None = None,
        contacts: ContactsKamino | None = None,
        solver: LinearSolverType | None = None,
        solver_kwargs: dict[str, Any] | None = None,
        config: list[DualProblem.Config] | DualProblem.Config | None = None,
        compute_h: bool = False,
    ):
        """
        Finalizes all memory allocations of the dual problem data
        for the given model, limits, contacts and Jacobians.

        Args:
            model (ModelKamino):
                The model to build the dual problem for.
            jacobians (SparseSystemJacobians | DenseSystemJacobians):
                The constraints Jacobians for this model.
            contacts (ContactsKamino, optional):
                The contacts container to use for the dual problem.
            solver (LinearSolverType, optional):
                The linear solver to use for the Delassus operator.\n
                Defaults to `None`.
            config (List[DualProblem.Config] | DualProblem.Config, optional):
                The config for the dual problem.\n
                If a single `DualProblem.Config` object is provided, it will be replicated for all worlds.
                Defaults to `None`, indicating that default config will be used for all worlds.
            compute_nonlinear_forces (bool, optional):
                Set to `True` to enable the computation of the nonlinear
                generalized forces vectors in construction of the dual problem.\n
                Defaults to `False`.
        """
        # Ensure the model is valid
        if model is None:
            raise ValueError("A model of type `ModelKamino` must be provided to allocate the Delassus operator.")
        elif not isinstance(model, ModelKamino):
            raise ValueError("Invalid model provided. Must be an instance of `ModelKamino`.")

        # Ensure the data container is valid if provided
        if data is not None:
            if not isinstance(data, DataKamino):
                raise ValueError("Invalid data container provided. Must be an instance of `DataKamino`.")

        # Ensure the limits container is valid if provided
        if limits is not None:
            if not isinstance(limits, LimitsKamino):
                raise ValueError("Invalid limits container provided. Must be an instance of `LimitsKamino`.")

        # Ensure the contacts container is valid if provided
        if contacts is not None:
            if not isinstance(contacts, ContactsKamino):
                raise ValueError("Invalid contacts container provided. Must be an instance of `ContactsKamino`.")

        # Use the model's device
        self._device = model.device

        # Capture reference to the model size
        self._size = model.size

        # Check config validity and update cache
        self._config = self._check_config(config, model.info.num_worlds)
        self._compute_h = compute_h

        # Determine the maximum number of contacts supported by the model
        # in order to allocate corresponding per-friction-cone parameters
        model_max_contacts_host = contacts.model_max_contacts_host if contacts is not None else 0

        # Construct the Delassus operator first since it will already process the necessary
        # model and contacts allocation sizes and will create some of the necessary arrays
        if self._sparse:
            self._delassus = BlockSparseMatrixFreeDelassusOperator(
                model=model,
                data=data,
                limits=limits,
                contacts=contacts,
                jacobians=jacobians,
                solver=solver,
                solver_kwargs=solver_kwargs,
            )
            # Assign identity regularization, to be modified by solver
            self._delassus.set_regularization(
                wp.zeros(
                    (model.size.sum_of_max_total_cts,),
                    dtype=float32,
                    device=self._device,
                )
            )
        else:
            self._delassus = DelassusOperator(
                model=model,
                data=data,
                limits=limits,
                contacts=contacts,
                solver=solver,
                solver_kwargs=solver_kwargs,
            )

        # Construct the dual problem data container
        with wp.ScopedDevice(self._device):
            if self._sparse:
                self._data = DualProblemData(
                    # Set the host-side caches of the maximal problem dimensions
                    num_worlds=self._delassus.num_matrices,
                    max_of_maxdims=self._delassus.max_of_max_dims,
                    # Capture references to the mode and data info arrays
                    njc=model.info.num_joint_cts,
                    nl=data.info.num_limits,
                    nc=data.info.num_contacts,
                    lio=model.info.limits_offset,
                    cio=model.info.contacts_offset,
                    uio=model.info.unilaterals_offset,
                    lcgo=data.info.limit_cts_group_offset,
                    ccgo=data.info.contact_cts_group_offset,
                    # Capture references to arrays already create by the Delassus operator
                    maxdim=self._delassus.info.maxdim,
                    dim=self._delassus.info.dim,
                    mio=None,
                    vio=self._delassus.info.vio,
                    D=None,
                    # Allocate new memory for the remaining dual problem quantities
                    config=wp.array([c.to_struct() for c in self.config], dtype=DualProblemConfigStruct),
                    h=wp.zeros(shape=(model.size.sum_of_num_bodies,), dtype=vec6f) if self._compute_h else None,
                    u_f=wp.zeros(shape=(model.size.sum_of_num_bodies,), dtype=vec6f),
                    v_b=wp.zeros(shape=(self._delassus.sum_of_max_dims,), dtype=float32),
                    v_i=wp.zeros(shape=(self._delassus.sum_of_max_dims,), dtype=float32),
                    v_f=wp.zeros(shape=(self._delassus.sum_of_max_dims,), dtype=float32),
                    mu=wp.zeros(shape=(model_max_contacts_host,), dtype=float32),
                    P=wp.ones(shape=(self._delassus.sum_of_max_dims,), dtype=float32),
                )
                # Connect Delassus preconditioner to data array
                self._delassus.set_preconditioner(self._data.P)
            else:
                self._data = DualProblemData(
                    # Set the host-side caches of the maximal problem dimensions
                    num_worlds=self._delassus.num_worlds,
                    max_of_maxdims=self._delassus.num_maxdims,
                    # Capture references to the mode and data info arrays
                    njc=model.info.num_joint_cts,
                    nl=data.info.num_limits,
                    nc=data.info.num_contacts,
                    lio=model.info.limits_offset,
                    cio=model.info.contacts_offset,
                    uio=model.info.unilaterals_offset,
                    lcgo=data.info.limit_cts_group_offset,
                    ccgo=data.info.contact_cts_group_offset,
                    # Capture references to arrays already create by the Delassus operator
                    maxdim=self._delassus.info.maxdim,
                    dim=self._delassus.info.dim,
                    mio=self._delassus.info.mio,
                    vio=self._delassus.info.vio,
                    D=self._delassus.D,
                    # Allocate new memory for the remaining dual problem quantities
                    config=wp.array([c.to_struct() for c in self.config], dtype=DualProblemConfigStruct),
                    h=wp.zeros(shape=(model.size.sum_of_num_bodies,), dtype=vec6f) if self._compute_h else None,
                    u_f=wp.zeros(shape=(model.size.sum_of_num_bodies,), dtype=vec6f),
                    v_b=wp.zeros(shape=(self._delassus.num_maxdims,), dtype=float32),
                    v_i=wp.zeros(shape=(self._delassus.num_maxdims,), dtype=float32),
                    v_f=wp.zeros(shape=(self._delassus.num_maxdims,), dtype=float32),
                    mu=wp.zeros(shape=(model_max_contacts_host,), dtype=float32),
                    P=wp.ones(shape=(self._delassus.num_maxdims,), dtype=float32),
                )

    def zero(self):
        if self._compute_h:
            self._data.h.zero_()
        self._data.u_f.zero_()
        self._data.v_b.zero_()
        self._data.v_i.zero_()
        self._data.v_f.zero_()
        self._data.mu.zero_()
        self._data.P.fill_(1.0)
        if self._sparse:
            self._delassus.set_needs_update()

    def build(
        self,
        model: ModelKamino,
        data: DataKamino,
        jacobians: DenseSystemJacobians | SparseSystemJacobians,
        limits: LimitsKamino | None = None,
        contacts: ContactsKamino | None = None,
        reset_to_zero: bool = True,
    ):
        """
        Builds the dual problem for the given model, data, limits and contacts data.
        """
        if self._sparse and not isinstance(jacobians, SparseSystemJacobians):
            raise TypeError("Dual problem in sparse configuration requires sparse jacobians.")

        # Initialize problem data
        if reset_to_zero:
            self.zero()

        # Build the dense Delassus operator if applicable
        # NOTE: We build this first since it will update the arrays of active constraints
        if not self._sparse:
            self._delassus.build(
                model=model,
                data=data,
                jacobians=jacobians,
                reset_to_zero=reset_to_zero,
            )

        # Optionally also build the non-linear generalized force vector
        if self._compute_h:
            self._build_nonlinear_generalized_force(model, data)

        # Build the generalized free-velocity vector
        self._build_generalized_free_velocity(model, data)

        # Build the free-velocity bias terms
        self._build_free_velocity_bias(model, data, limits, contacts)

        # Build the free-velocity vector
        if isinstance(jacobians, SparseSystemJacobians):
            wp.copy(self._data.v_f, self._data.v_b)
            J_cts = jacobians._J_cts.bsm
            wp.launch(
                _build_free_velocity_sparse,
                dim=(self._size.num_worlds, J_cts.max_of_num_nzb),
                inputs=[
                    # Inputs:
                    model.info.bodies_offset,
                    data.bodies.u_i,
                    J_cts.num_nzb,
                    J_cts.nzb_start,
                    J_cts.nzb_coords,
                    J_cts.nzb_values,
                    self._data.vio,
                    self._data.u_f,
                    self._data.v_i,
                    # Outputs:
                    self._data.v_f,
                ],
                device=self.device,
            )
        else:
            wp.launch(
                _build_free_velocity,
                dim=(self._size.num_worlds, self._size.max_of_max_total_cts),
                inputs=[
                    # Inputs:
                    model.info.bodies_offset,
                    data.bodies.u_i,
                    jacobians.data.J_cts_offsets,
                    jacobians.data.J_cts_data,
                    self._data.dim,
                    self._data.vio,
                    self._data.u_f,
                    self._data.v_b,
                    self._data.v_i,
                    # Outputs:
                    self._data.v_f,
                ],
                device=self.device,
            )

        # Optionally build and apply the Delassus diagonal preconditioner
        if any(s.dynamics.preconditioning for s in self._config):
            self._build_dual_preconditioner()
            self._apply_dual_preconditioner_to_dual()

    ###
    # Internals
    ###

    @staticmethod
    def _check_config(
        config: list[DualProblem.Config] | DualProblem.Config | None, num_worlds: int
    ) -> list[DualProblem.Config]:
        """
        Checks and prepares the config for the dual problem.

        If a single `DualProblemConfig` object is provided, it will be replicated for all worlds.
        If a list of configs is provided, it will ensure that the number of configs matches the number of worlds.
        """
        if config is None:
            # If no config is provided, use default config
            return [DualProblem.Config()] * num_worlds
        elif isinstance(config, DualProblem.Config):
            # If a single config object is provided, replicate it for all worlds
            return [config] * num_worlds
        elif isinstance(config, list):
            # Ensure the configs are of the correct type and length
            if len(config) != num_worlds:
                raise ValueError(f"Expected {num_worlds} configs, got {len(config)}")
            for c in config:
                if not isinstance(c, DualProblem.Config):
                    raise TypeError(f"Expected DualProblem.Config, got {type(c)}")
            return config
        else:
            raise TypeError(f"Expected List[DualProblem.Config] or DualProblem.Config, got {type(config)}")

    def _build_nonlinear_generalized_force(model: ModelKamino, data: DataKamino, problem: DualProblemData):
        """
        Builds the nonlinear generalized force vector `h`.
        """
        wp.launch(
            _build_nonlinear_generalized_force,
            dim=model.size.sum_of_num_bodies,
            inputs=[
                # Inputs:
                model.time.dt,
                model.gravity.vector,
                model.bodies.wid,
                model.bodies.m_i,
                data.bodies.u_i,
                data.bodies.I_i,
                data.bodies.w_e_i,
                data.bodies.w_a_i,
                # Outputs:
                problem.h,
            ],
            device=model.device,
        )

    def _build_generalized_free_velocity(self, model: ModelKamino, data: DataKamino):
        """
        Builds the generalized free-velocity vector (i.e. unconstrained) `u_f`.
        """
        wp.launch(
            _build_generalized_free_velocity,
            dim=model.size.sum_of_num_bodies,
            inputs=[
                # Inputs:
                model.time.dt,
                model.gravity.vector,
                model.bodies.wid,
                model.bodies.m_i,
                model.bodies.inv_m_i,
                data.bodies.u_i,
                data.bodies.I_i,
                data.bodies.inv_I_i,
                data.bodies.w_e_i,
                data.bodies.w_a_i,
                # Outputs:
                self._data.u_f,
            ],
            device=self.device,
        )

    def _build_free_velocity_bias(
        self,
        model: ModelKamino,
        data: DataKamino,
        limits: LimitsKamino | None = None,
        contacts: ContactsKamino | None = None,
    ):
        """
        Builds the free-velocity bias vector `v_b`.
        """

        if model.size.sum_of_num_joints > 0:
            if model.size.sum_of_num_dynamic_joints > 0:
                wp.launch(
                    _build_free_velocity_bias_joint_dynamics,
                    dim=model.size.sum_of_num_joints,
                    inputs=[
                        # Inputs:
                        model.joints.wid,
                        model.joints.dynamic_cts_offset,
                        model.joints.dynamic_cts_offset_total_cts,
                        data.joints.dq_b_j,
                        # Outputs:
                        self._data.v_b,
                    ],
                    device=self.device,
                )
            wp.launch(
                _build_free_velocity_bias_joint_kinematics,
                dim=model.size.sum_of_num_joints,
                inputs=[
                    # Inputs:
                    model.time.inv_dt,
                    model.joints.wid,
                    model.joints.kinematic_cts_offset,
                    model.joints.kinematic_cts_offset_total_cts,
                    data.joints.r_j,
                    self._data.config,
                    # Outputs:
                    self._data.v_b,
                ],
                device=self.device,
            )

        if limits is not None and limits.model_max_limits_host > 0:
            wp.launch(
                _build_free_velocity_bias_limits,
                dim=limits.model_max_limits_host,
                inputs=[
                    # Inputs:
                    model.time.inv_dt,
                    data.info.limit_cts_group_offset,
                    limits.model_max_limits_host,
                    limits.model_active_limits,
                    limits.wid,
                    limits.lid,
                    limits.r_q,
                    self._data.config,
                    self._data.vio,
                    # Outputs:
                    self._data.v_b,
                ],
                device=self.device,
            )

        if contacts is not None and contacts.model_max_contacts_host > 0:
            wp.launch(
                _build_free_velocity_bias_contacts,
                dim=contacts.model_max_contacts_host,
                inputs=[
                    # Inputs:
                    model.time.inv_dt,
                    model.info.contacts_offset,
                    data.info.contact_cts_group_offset,
                    contacts.model_max_contacts_host,
                    contacts.model_active_contacts,
                    contacts.wid,
                    contacts.cid,
                    contacts.gapfunc,
                    contacts.material,
                    self._data.config,
                    self._data.vio,
                    # Outputs:
                    self._data.v_b,
                    self._data.v_i,
                    self._data.mu,
                ],
                device=self.device,
            )

    def _build_free_velocity(self, model: ModelKamino, data: DataKamino, jacobians: DenseSystemJacobians):
        """
        Builds the free-velocity vector `v_f`.
        """
        wp.launch(
            _build_free_velocity,
            dim=(self._size.num_worlds, self._size.max_of_max_total_cts),
            inputs=[
                # Inputs:
                model.info.bodies_offset,
                data.bodies.u_i,
                jacobians.data.J_cts_offsets,
                jacobians.data.J_cts_data,
                self._data.dim,
                self._data.vio,
                self._data.u_f,
                self._data.v_b,
                self._data.v_i,
                # Outputs:
                self._data.v_f,
            ],
            device=self.device,
        )

    def _build_dual_preconditioner(self):
        """
        Builds the diagonal preconditioner 'P' according to the current Delassus operator.
        """
        if self._sparse:
            self._delassus.diagonal(self._data.P)
            wp.launch(
                _build_dual_preconditioner_all_constraints_sparse,
                dim=(self._size.num_worlds, self._size.max_of_max_total_cts),
                inputs=[
                    # Inputs:
                    self._data.config,
                    self._data.dim,
                    self._data.vio,
                    self._data.njc,
                    self._data.nl,
                    # Outputs:
                    self._data.P,
                ],
                device=self.device,
            )
        else:
            wp.launch(
                _build_dual_preconditioner_all_constraints,
                dim=(self._size.num_worlds, self._size.max_of_max_total_cts),
                inputs=[
                    # Inputs:
                    self._data.config,
                    self._data.dim,
                    self._data.mio,
                    self._data.vio,
                    self._data.njc,
                    self._data.nl,
                    self._data.D,
                    # Outputs:
                    self._data.P,
                ],
                device=self.device,
            )

    def _apply_dual_preconditioner_to_dual(self):
        """
        Applies the diagonal preconditioner 'P' to the
        Delassus operator 'D' and free-velocity vector `v_f`.
        """
        if self._sparse:
            # Preconditioner has already been connected to appropriate array
            pass
        else:
            wp.launch(
                _apply_dual_preconditioner_to_matrix,
                dim=(self._size.num_worlds, self.delassus._max_of_max_total_D_size),
                inputs=[
                    # Inputs:
                    self._data.dim,
                    self._data.mio,
                    self._data.vio,
                    self._data.P,
                    # Outputs:
                    self._data.D,
                ],
                device=self.device,
            )

        wp.launch(
            _apply_dual_preconditioner_to_vector,
            dim=(self._size.num_worlds, self._size.max_of_max_total_cts),
            inputs=[
                # Inputs:
                self._data.dim,
                self._data.vio,
                self._data.P,
                # Outputs:
                self._data.v_f,
            ],
            device=self.device,
        )

    def _apply_dual_preconditioner_to_matrix(self, X: wp.array):
        """
        Applies the diagonal preconditioner 'P' to a given matrix.
        """
        wp.launch(
            _apply_dual_preconditioner_to_matrix,
            dim=(self._size.num_worlds, self._size.max_of_max_total_cts),
            inputs=[
                # Inputs:
                self._data.dim,
                self._data.mio,
                self._data.vio,
                self._data.P,
                # Outputs:
                X,
            ],
            device=self.device,
        )

    def _apply_dual_preconditioner_to_vector(self, x: wp.array):
        """
        Applies the diagonal preconditioner 'P' to a given vector.
        """
        wp.launch(
            _apply_dual_preconditioner_to_vector,
            dim=(self._size.num_worlds, self._size.max_of_max_total_cts),
            inputs=[
                # Inputs:
                self._data.dim,
                self._data.vio,
                self._data.P,
                # Outputs:
                x,
            ],
            device=self.device,
        )
