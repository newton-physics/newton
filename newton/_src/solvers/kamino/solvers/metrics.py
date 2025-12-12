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

"""
Provides a utilities to compute performance metrics that
assess physical plausibility and accuracy of simulations.


The :class:`SolutionMetricsData` class defines the set of simulation solver performance metrics that
provide quantitative measures of constraint satisfaction and physical plausibility of the computed
solution to the dual forward-dynamics Nonlinear Complementarity Problem (NCP).

The :class:`SolutionMetrics` class provides a high-level interface to manage all relevant data
allocations as well as provide the operations to compute the various performance metrics.

Usage
----
A typical example for using this module is:

    # Import all relevant types from Kamino
    from newton._src.solvers.kamino.core import ModelBuilder
    from newton._src.solvers.kamino.geometry import Contacts
    from newton._src.solvers.kamino.kinematics import Limits
    from newton._src.solvers.kamino.kinematics import DenseSystemJacobians
    from newton._src.solvers.kamino.dynamics import DualProblem
    from newton._src.solvers.kamino.solvers import PADMMSolver

    # Create a model builder and add bodies, joints, geoms, etc.
    builder = ModelBuilder()
    ...

    # Create a model from the builder and construct additional
    # containers to hold joint-limits, contacts, Jacobians
    model = builder.finalize()
    state_p = model.state()
    data = model.data()
    limits = Limits(builder)
    contacts = Contacts(builder)
    jacobians = DenseSystemJacobians(model, limits, contacts)

    # Build the Jacobians for the model and active limits and contacts
    jacobians.build(model, data, limits, contacts)
    ...

    # Create a forward-dynamics DualProblem to be solved
    dual = DualProblem(model, limits, contacts)
    dual.build(model, data, limits, contacts, jacobians)

    # Create a forward-dynamics PADMM solver
    solver = PADMMSolver(model, limits, contacts)

    # Solve the dual forward dynamics problem
    solver.coldstart()
    solver.solve(problem=dual)

    # Create a SolutionMetrics container
    metrics = SolutionMetrics(model)

    # Compute the solution metrics after solving the dual problem
    metrics.reset()
    metrics.evaluate(
        solver.state.sigma,
        solver.solution.lambdas,
        solver.solution.v_plus,
        model,
        data,
        state_p,
        dual,
        jacobians,
        limits,
        contacts,
    )
"""

from dataclasses import dataclass

import warp as wp
from warp.context import Devicelike

from ..core.math import screw, screw_angular, screw_linear
from ..core.model import Model, ModelData
from ..core.state import State
from ..core.types import float32, int32, int64, mat33f, uint32, vec2f, vec3f, vec4f, vec6f
from ..dynamics.dual import DualProblem
from ..geometry.contacts import Contacts
from ..geometry.keying import build_pair_key2
from ..kinematics.jacobians import DenseSystemJacobians
from ..kinematics.limits import Limits
from ..solvers.padmm.math import (
    compute_desaxce_corrections,
    compute_dot_product,
    compute_double_dot_product,
    compute_ncp_complementarity_residual,
    compute_ncp_dual_residual,
    compute_ncp_natural_map_residual,
    compute_ncp_primal_residual,
    compute_vector_sum,
)

###
# Module interface
###

__all__ = [
    "SolutionMetrics",
    "SolutionMetricsData",
]


###
# Module configs
###

wp.set_module_options({"enable_backward": False})


###
# Types
###


@dataclass
class SolutionMetricsData:
    """
    Defines a container to hold performance metrics that assess physical plausibility and accuracy
    of the computed solution to the dual forward-dynamics Nonlinear Complementarity Problem (NCP).

    Attributes:
        r_eom (wp.array):
            The largest residual across all DoF dimensions of the Equations-of-Motion (EoM).
        r_kinematics (wp.array):
            The largest residual across all kinematic bilateral (i.e. equality) constraints.
        r_cts_joints (wp.array):
            The largest constraint violation residual across all bilateral kinematic joint constraints.
        r_cts_limits (wp.array):
            The largest constraint violation residual across all unilateral joint-limit constraints.
        r_cts_contacts (wp.array):
            The largest constraint violation residual across all contact constraints.
        r_ncp_primal (wp.array):
            The NCP primal residual representing the violation of set-valued constraint reactions.
        r_ncp_dual (wp.array):
            The NCP dual residual representing the violation of set-valued augmented constraint velocities.
        r_ncp_compl (wp.array):
            The NCP complementarity residual representing the violation of complementarity conditions.
        r_vi_natmap (wp.array):
            The Variational Inequality (VI) natural-map residual representing the proximity
            of the constraint reactions to a true solution of the VI defined by the NCP.
    """

    r_eom: wp.array | None = None
    """
    The largest residual across all DoF dimensions of the Equations-of-Motion (EoM).

    Measures how well the computed post-event generalized velocity `u^+` and constraint reactions
    `lambdas` (i.e. Lagrange multipliers) satisfy the impulse-velocity form of the equations-of-motion.

    Computed as the maximum absolute value (i.e. infinity-norm) over the residual of:

    `r_eom :=  || M @ (u^+ - u^-) - dt * (h + J_dofs.T @ tau) - J_cts.T @ lambdas||_inf`,

    where:
    - `M` is the generalized mass matrix
    - `u^+` and `u^-` are the post- and pre-event generalized velocities
    - `dt` is the time-step
    - `h` is the generalized impulse vector
    - `J_cts` is the constraint Jacobian matrix
    - `lambdas` is the vector of all constraint reactions (i.e. Lagrange multipliers)
    - `J_dofs` is the actuation Jacobian matrix
    - `tau` is the vector of generalized actuation forces

    Shape of ``(num_worlds,)`` and type :class:`float32`.
    """

    r_eom_argmax: wp.array | None = None
    """
    The index pair key computed from the body index and
    degree-of-freedom (DoF) with the largest EoM residual.

    Shape of ``(num_worlds,)`` and type :class:`int64`.
    """

    r_kinematics: wp.array | None = None
    """
    The largest residual across all kinematic bilateral (i.e. equality) constraints.

    Measures how well the computed post-event generalized velocity
    `u^+` satisfies the velocity-level kinematic equality constraints.

    Computed as the maximum absolute value (i.e. infinity-norm) over velocity-level kinematics constraints:
    `r_kinematics := || J_cts_joints @ u^+ ||_inf`,

    where:
    - `J_cts_joints` is the constraint Jacobian matrix for bilateral joint constraints
    - `u^+` is the post-event generalized velocity

    Shape of ``(num_worlds,)`` and type :class:`float32`.
    """

    r_kinematics_argmax: wp.array | None = None
    """
    The index pair key computed from the joint index and
    bilateral kinematic constraint with the largest residual.

    Shape of ``(num_worlds,)`` and type :class:`int64`.
    """

    r_cts_joints: wp.array | None = None
    """
    The largest constraint violation residual across all bilateral joint constraints.

    Computed as the maximum absolute value (i.e. infinity-norm) over joint constraint residuals.

    Equivalent to `r_cts_joints := || r_j ||_inf`, where `r_j` is the
    array of joint constraint residuals defined in :class:`JointsData`.

    Shape of ``(num_worlds,)`` and type :class:`float32`.
    """

    r_cts_joints_argmax: wp.array | None = None
    """
    The index pair key computed from the joint index and bilateral
    kinematic joint constraint with the largest residual.

    Shape of ``(num_worlds,)`` and type :class:`int64`.
    """

    r_cts_limits: wp.array | None = None
    """
    The largest constraint violation residual across all unilateral joint-limit constraints.

    Computed as the maximum absolute value (i.e. infinity-norm) over joint-limit constraint residuals.

    Equivalent to `r_cts_limits := || r_l ||_inf`, where `r_l` would be an array of joint-limit
    constraint residuals constructed from the collection of `r_q` elements defined in :class:`LimitsData`.

    Shape of ``(num_worlds,)`` and type :class:`float32`.
    """

    r_cts_limits_argmax: wp.array | None = None
    """
    The index pair key computed from the joint index and degree-of-freedom
    (DoF) with the largest unilateral joint-limit constraint residual.

    Shape of ``(num_worlds,)`` and type :class:`int64`.
    """

    r_cts_contacts: wp.array | None = None
    """
    The largest constraint violation residual across all contact constraints.

    Computed as the maximum absolute value (i.e. infinity-norm) over contact constraint residuals.

    Equivalent to `r_cts_contacts := || d_k ||_inf`, where `d_k` would be an array of
    contact penetrations extracted from the `gapfunc` elements of :class:`ContactsData`.

    Shape of ``(num_worlds,)`` and type :class:`float32`.
    """

    r_cts_contacts_argmax: wp.array | None = None
    """
    The indexir of the contact  with the largest unilateral contact constraint residual.

    Shape of ``(num_worlds,)`` and type :class:`int32`.
    """

    r_v_plus: wp.array | None = None
    """
    The largest error in the estimation of the post-event constraint-space velocity.

    Measures how well the computed post-event constraint-space velocity `v^+` was estimated.

    Computed as the maximum absolute value (i.e. infinity-norm) over velocity-level kinematics constraints:
    `r_v_plus := || v_plus_est - v_plus_true ||_inf`,

    where:
    - `v_plus_est` is the estimated post-event constraint-space velocity
    - v_plus_true` is the true post-event constraint-space velocity computed as
      `v_plus_true = v_f + D @ lambdas`, where `v_f` is the unconstrained constraint-space velocity,
      `D` is the Delassus operator, and `lambdas` is the vector of all constraint reactions (i.e. Lagrange multipliers).

    Shape of ``(num_worlds,)`` and type :class:`float32`.
    """

    r_v_plus_argmax: wp.array | None = None
    """
    The index of the constraint with the largest post-event constraint-space velocity estimation error.

    Shape of ``(num_worlds,)`` and type :class:`int32`.
    """

    r_ncp_primal: wp.array | None = None
    """
    The NCP primal residual representing the violation of set-valued constraint reactions.

    Measures the feasibility of constraint reactions w.r.t the feasible-set cone `K`
    defined as the Cartesian product over all positive-orthants for joint-limits and
    Coulomb friction cones for contacts:
    `K = R^{n_l}_{+} x Π_{k=1}^{n_c} K_{mu_k}`,

    Computed as the maximum absolute value (i.e. infinity-norm) over the residual:
    `r_ncp_primal(lambda) = || lambda - P_K(lambda) ||_inf`, where `P_K()` is the
    Euclidean projection, i.e. proximal operator, onto K, and `lambda` is the
    vector of all constraint reactions (i.e. Lagrange multipliers).

    Shape of ``(num_worlds,)`` and type :class:`float32`.
    """

    r_ncp_primal_argmax: wp.array | None = None
    """
    The index of the constraint with the largest NCP primal residual.\n
    Shape of ``(num_worlds,)`` and type :class:`int32`.
    """

    r_ncp_dual: wp.array | None = None
    """
    The NCP dual residual representing the violation of set-valued augmented constraint velocities.

    Measures the feasibility of augmented constraint-space velocities w.r.t
    the dual cone `K*`, the Lagrange dual of the feasible-set cone `K`.

    Computed as the maximum absolute value (i.e. infinity-norm) over the residual:
    `r_ncp_dual(v_hat^+) = || v_hat^+ - P_K*(v_hat^+) ||_inf`, where `P_K*()` is
    the Euclidean projection, i.e. proximal operator, onto K*, and `v_hat^+` is
    the so-called augmented constraint-space velocity. The latter is defined as
    `v_hat^+ = v^+ + Gamma(v^+)`, where `v^+ := v_f D @ lambda` is the post-event
    constraint-space velocity, and `Gamma(v^+)` is the De Saxce correction term.

    Shape of ``(num_worlds,)`` and type :class:`float32`.
    """

    r_ncp_dual_argmax: wp.array | None = None
    """
    The index of the constraint with the largest NCP dual residual.

    Shape of ``(num_worlds,)`` and type :class:`int32`.
    """

    r_ncp_compl: wp.array | None = None
    """
    The NCP complementarity residual representing the violation of complementarity conditions.

    Measures the complementarity between constraint reactions and the augmented constraint-space
    velocities, as defined by the velocity-level Signorini (i.e. complementarity) conditions
    and positive orthants for joint-limits and Coulomb friction cones for contacts.

    Computed as the maximum absolute value (i.e. infinity-norm) over the residual:
    `r_ncp_compl(lambda) = || lambda.T @ v_hat^+ ||_inf`,
    where `lambda` is the vector of all constraint reactions (i.e. Lagrange multipliers),
    and `v_hat^+` is the augmented constraint-space velocity defined above.

    Shape of ``(num_worlds,)`` and type :class:`float32`.
    """

    r_ncp_compl_argmax: wp.array | None = None
    """
    The index of the constraint with the largest NCP complementarity residual.

    Shape of ``(num_worlds,)`` and type :class:`int32`.
    """

    r_vi_natmap: wp.array | None = None
    """
    The Variational Inequality (VI) natural-map residual representing the proximity
    of the constraint reactions to a true solution of the VI defined by the NCP.

    Measures the how well the given constraint reactions solve the `NCP(D, v_f, K)`,
    by simultaneously combining the effects of the primal, dual and complementarity
    residuals into a single value, providing a convenient short-hand for both solver
    convergence and solution validity.

    Computed as the maximum absolute value (i.e. infinity-norm) over the residual:
    `r_vi_natmapv_hat(lambda) = || lambda - P_K*(lambda - v_hat^+(lambda)) ||_inf`,
    where `P_K*()` is the Euclidean projection, i.e. proximal operator, onto K*,
    `lambda` is the vector of all constraint reactions (i.e. Lagrange multipliers),
    and `v_hat^+(lambda)` is the augmented constraint-space velocity defined above.

    Shape of ``(num_worlds,)`` and type :class:`float32`.
    """

    r_vi_natmap_argmax: wp.array | None = None
    """
    The index of the constraint with the largest VI natural-map residual.

    Shape of ``(num_worlds,)`` and type :class:`int32`.
    """

    f_ncp: wp.array | None = None
    """
    Evaluation of the NCP energy dissipation objective function.

    Represents only the energy dissipated through friction.

    Computed as as:
    `f_ncp(lambda) := 0.5 * lambda.T @ D @ lambda + lambda.T @ v_f + lambda.T @ s`,
    where `D` is the Delassus operator, `v_f` is the unconstrained constraint-space
    velocity and `s := Gamma(v^+)` is the De Saxce correction term.

    It is also equivalently computed as:
    `f_ncp(lambda) = f_ccp(lambda) + lambda.T @ s`,
    where `f_ccp` is the CCP energy dissipation objective.

    Shape of ``(num_worlds,)`` and type :class:`float32`.
    """

    f_ccp: wp.array | None = None
    """
    Evaluation of the CCP energy dissipation objective function.

    Represents only the energy dissipated through friction.

    Computed as as:
    `f_ccp(lambda) := 0.5 * lambda.T @ D @ lambda + v_f.T @ lambda`,
    where `lambda` is the vector of all constraint reactions (i.e. Lagrange multipliers),
    `D` is the Delassus operator and `v_f` is the unconstrained constraint-space velocity.

    It is also equivalently computed as:
    `f_ccp(lambda) := 0.5 * lambda.T @ (v+ + v_f)`,
    where `v+ = v_f + D @ lambda` is the post-event constraint-space velocity.

    Shape of ``(num_worlds,)`` and type :class:`float32`.
    """

    def clear(self):
        """
        Clears all metric-argmax indices to -1.
        """
        self.r_eom_argmax.fill_(-1)
        self.r_kinematics_argmax.fill_(-1)
        self.r_cts_joints_argmax.fill_(-1)
        self.r_cts_limits_argmax.fill_(-1)
        self.r_cts_contacts_argmax.fill_(-1)
        self.r_v_plus_argmax.fill_(-1)
        self.r_ncp_primal_argmax.fill_(-1)
        self.r_ncp_dual_argmax.fill_(-1)
        self.r_ncp_compl_argmax.fill_(-1)
        self.r_vi_natmap_argmax.fill_(-1)

    def zero(self):
        """
        Resets all metrics to zeros.
        """
        self.r_eom.zero_()
        self.r_kinematics.zero_()
        self.r_cts_joints.zero_()
        self.r_cts_limits.zero_()
        self.r_cts_contacts.zero_()
        self.r_v_plus.zero_()
        self.r_ncp_primal.zero_()
        self.r_ncp_dual.zero_()
        self.r_ncp_compl.zero_()
        self.r_vi_natmap.zero_()
        self.f_ncp.zero_()
        self.f_ccp.zero_()

    def reset(self):
        """
        Resets all metrics and argmax indices.
        """
        self.clear()
        self.zero()


###
# Functions
###


@wp.func
def compute_v_plus(
    dim: int32,
    vio: int32,
    mio: int32,
    sigma: float32,
    P: wp.array(dtype=float32),
    D_p: wp.array(dtype=float32),
    v_f_p: wp.array(dtype=float32),
    lambdas: wp.array(dtype=float32),
    v_plus: wp.array(dtype=float32),
):
    """
    Computes the post-event constraint-space velocity as:

    `v_plus = P^-1 @ (v_f + (D_p - sigma * I_n) @ (P^-1 @ lambdas))`.

    Where `P` is the diagonal preconditioning matrix, `D_p` is the
    preconditioned Delassus matrix, `v_f_p` is the preconditioned
    unconstrained constraint-space velocity, `lambdas` is the vector
    of constraint reactions (i.e. Lagrange multipliers), and `I_n`
    is the identity matrix of size `n x n`.

    The preconditioned Delassus matrix `D_p` is stored using row-major order in a
    flat array with allocation size `maxdim x maxdim`, starting from the matrix
    index offset `mio`. The active dimensions of the `D_p` are `dim x dim`, where
    `dim` is the number of active rows and columns.

    The vectors `v_f_p, lambdas, v_plus` are stored in flat arrays
    with dimensions `dim`, starting from the vector index offset `vio`.

    Args:
        maxdim (int32): The maximum dimension of the matrix `A`.
        dim (int32): The active dimension of the matrix `A` and the vectors `x, b, c`.
        vio (int32): The vector index offset (i.e. start index) for the vectors `x, b, c`.
        mio (int32): The matrix index offset (i.e. start index) for the matrix `A`.
        D_p (wp.array(dtype=float32)):
            Input preconditioned Delassus matrix stored in row-major order.
        v_f_p (wp.array(dtype=float32)):
            Input preconditioned unconstrained constraint-space velocity vector.
        lambdas (wp.array(dtype=float32)):
            Input constraint reactions (i.e. Lagrange multipliers) vector.
        v_plus (wp.array(dtype=float32)):
            Output array to store the post-event constraint-space velocity vector.
    """
    v_f_p_i = float(0.0)
    lambdas_j = float(0.0)
    for i in range(dim):
        v_i = vio + i
        m_i = mio + dim * i
        v_f_p_i = v_f_p[v_i]
        inv_P_i = 1.0 / P[v_i]
        lambdas_i = lambdas[v_i]
        v_f_p_i -= sigma * inv_P_i * lambdas_i
        for j in range(dim):
            v_j = vio + j
            inv_P_j = 1.0 / P[v_j]
            lambdas_j = lambdas[v_j]
            v_f_p_i += D_p[m_i + j] * inv_P_j * lambdas_j
        v_plus[v_i] = inv_P_i * v_f_p_i


@wp.func
def compute_vector_difference_infnorm(
    dim: int32,
    vio: int32,
    x: wp.array(dtype=float32),
    y: wp.array(dtype=float32),
) -> tuple[float32, int32]:
    """
    Computes the sum of two vectors `x` and `y` and stores the result in vector `z`.\n
    All vectors are stored in flat arrays, with dimension `dim` and starting from the vector index offset `vio`.

    Args:
        dim (int32): The dimension (i.e. size) of the vectors.
        vio (int32): The vector index offset (i.e. start index).
        x (wp.array(dtype=float32)): The first vector.
        y (wp.array(dtype=float32)): The second vector.
        z (wp.array(dtype=float32)): The output vector where the sum is stored.

    Returns:
        None: The result is stored in the output vector `z`.
    """
    max = float(0.0)
    argmax = int32(-1)
    for i in range(dim):
        v_i = vio + i
        err = wp.abs(x[v_i] - y[v_i])
        max = wp.max(max, err)
        if err == max:
            argmax = i
    return max, argmax


###
# Kernels
###


@wp.kernel
def _compute_eom_residual(
    # Inputs
    model_time_dt: wp.array(dtype=float32),
    model_gravity: wp.array(dtype=vec4f),
    model_bodies_wid: wp.array(dtype=int32),
    model_bodies_m_i: wp.array(dtype=float32),
    state_bodies_I_i: wp.array(dtype=mat33f),
    state_bodies_w_i: wp.array(dtype=vec6f),
    state_bodies_u_i: wp.array(dtype=vec6f),
    state_bodies_u_i_p: wp.array(dtype=vec6f),
    # Outputs
    metric_r_eom: wp.array(dtype=float32),
    metric_r_eom_argmax: wp.array(dtype=int64),
):
    # Retrieve the thread index as the body index
    bid = wp.tid()

    # Retrieve the body data
    wid = model_bodies_wid[bid]
    m_i = model_bodies_m_i[bid]
    I_i = state_bodies_I_i[bid]
    w_i = state_bodies_w_i[bid]
    u_i = state_bodies_u_i[bid]
    u_i_p = state_bodies_u_i_p[bid]

    # Retrieve the time step
    dt = model_time_dt[wid]
    gravity = model_gravity[wid]
    g = gravity.w * vec3f(gravity.x, gravity.y, gravity.z)

    # Decompose into linear and angular parts
    f_i = screw_linear(w_i)
    v_i = screw_linear(u_i)
    v_i_p = screw_linear(u_i_p)
    tau_i = screw_angular(w_i)
    omega_i = screw_angular(u_i)
    omega_i_p = screw_angular(u_i_p)
    S_i = wp.skew(omega_i_p)

    # Compute the per-body EoM residual over linear and angular parts
    r_linear_i = wp.abs(m_i * (v_i - v_i_p) - dt * (m_i * g + f_i))
    r_angular_i = wp.abs(I_i @ (omega_i - omega_i_p) - dt * (tau_i - S_i @ (I_i @ omega_i_p)))
    r_i = screw(r_linear_i, r_angular_i)

    # Compute the per-body maximum residual and argmax index
    r_eom_i = wp.max(r_i)
    r_eom_argmax_i = int32(wp.argmax(r_i))

    # Update the per-world maximum residual and argmax index
    previous_max = wp.atomic_max(metric_r_eom, wid, r_eom_i)
    if r_eom_i >= previous_max:
        argmax_key = int64(build_pair_key2(uint32(bid), uint32(r_eom_argmax_i)))
        wp.atomic_exch(metric_r_eom_argmax, int32(wid), argmax_key)


@wp.kernel
def _compute_joint_kinematics_residual(
    # Inputs:
    model_info_num_body_dofs: wp.array(dtype=int32),
    model_info_bodies_offset: wp.array(dtype=int32),
    model_joints_wid: wp.array(dtype=int32),
    model_joints_num_cts: wp.array(dtype=int32),
    model_joints_cts_offset: wp.array(dtype=int32),
    model_joints_bid_B: wp.array(dtype=int32),
    model_joints_bid_F: wp.array(dtype=int32),
    data_bodies_u_i: wp.array(dtype=vec6f),
    jacobian_cts_offset: wp.array(dtype=int32),
    jacobian_cts_data: wp.array(dtype=float32),
    # Outputs:
    metric_r_kinematics: wp.array(dtype=float32),
    metric_r_kinematics_argmax: wp.array(dtype=int64),
):
    # Retrieve the joint index from the thread index
    jid = wp.tid()

    # Retrieve the world index of the joint
    wid = model_joints_wid[jid]

    # Retrieve the body indices of the joint
    # NOTE: these indices are w.r.t the model
    bid_F_j = model_joints_bid_F[jid]
    bid_B_j = model_joints_bid_B[jid]

    # Retrieve the size and index offset of the joint constraint
    num_cts_j = model_joints_num_cts[jid]
    cts_offset_j = model_joints_cts_offset[jid]

    # Retrieve the number of body DoFs in the world
    nbd = model_info_num_body_dofs[wid]

    # Retrieve the element index offset of the bodies of the world
    bio = model_info_bodies_offset[wid]

    # Retrieve the constraint block index offsets of the
    # Jacobian matrix and multipliers vector of the world
    mio = jacobian_cts_offset[wid]

    # Compute the per-joint constraint Jacobian matrix-vector product
    j_v_j = vec6f(0.0)
    u_i_F = data_bodies_u_i[bid_F_j]
    dio_F = 6 * (bid_F_j - bio)
    for j in range(num_cts_j):
        mio_j = mio + nbd * (cts_offset_j + j) + dio_F
        for i in range(6):
            j_v_j[j] += jacobian_cts_data[mio_j + i] * u_i_F[i]
    if bid_B_j >= 0:
        u_i_B = data_bodies_u_i[bid_B_j]
        dio_B = 6 * (bid_B_j - bio)
        for j in range(num_cts_j):
            mio_j = mio + nbd * (cts_offset_j + j) + dio_B
            for i in range(6):
                j_v_j[j] += jacobian_cts_data[mio_j + i] * u_i_B[i]

    # Compute the per-joint kinematics residual
    r_kinematics_j = wp.max(wp.abs(j_v_j))

    # Update the per-world maximum residual and argmax index
    previous_max = wp.atomic_max(metric_r_kinematics, wid, r_kinematics_j)
    if r_kinematics_j >= previous_max:
        argmax_key = int64(build_pair_key2(uint32(jid), uint32(cts_offset_j)))
        wp.atomic_exch(metric_r_kinematics_argmax, wid, argmax_key)


@wp.kernel
def _compute_cts_joints_residual(
    # Inputs:
    data_info_joints_cts_offset: wp.array(dtype=int32),
    model_joints_wid: wp.array(dtype=int32),
    model_joints_num_cts: wp.array(dtype=int32),
    model_joints_cts_offset: wp.array(dtype=int32),
    data_joints_r_j: wp.array(dtype=float32),
    # Outputs:
    metric_r_cts_joints: wp.array(dtype=float32),
    metric_r_cts_joints_argmax: wp.array(dtype=int64),
):
    # Retrieve the joint index from the thread index
    jid = wp.tid()

    # Retrieve the joint-specific model data
    wid = model_joints_wid[jid]
    num_cts_j = model_joints_num_cts[jid]
    cts_offset_j = model_joints_cts_offset[jid]

    # Retrieve the joint-block constraint index offset of the world
    world_joints_cts_offset = data_info_joints_cts_offset[wid]

    # Compute the global constraint index offset for the specific joint
    cio_j = world_joints_cts_offset + cts_offset_j

    # Compute the per-joint constraint residual (infinity-norm)
    r_cts_joints_j = float32(0.0)
    for j in range(num_cts_j):
        r_cts_joints_j = wp.max(r_cts_joints_j, wp.abs(data_joints_r_j[cio_j + j]))

    # Update the per-world maximum residual and argmax index
    previous_max = wp.atomic_max(metric_r_cts_joints, wid, r_cts_joints_j)
    if r_cts_joints_j >= previous_max:
        argmax_key = int64(build_pair_key2(uint32(jid), uint32(cts_offset_j)))
        wp.atomic_exch(metric_r_cts_joints_argmax, wid, argmax_key)


@wp.kernel
def _compute_cts_limits_residual(
    # Inputs:
    limit_model_num_limits: wp.array(dtype=int32),
    limit_wid: wp.array(dtype=int32),
    limit_lid: wp.array(dtype=int32),
    limit_dof: wp.array(dtype=int32),
    limit_r_q: wp.array(dtype=float32),
    # Outputs:
    metric_r_cts_limits: wp.array(dtype=float32),
    metric_r_cts_limits_argmax: wp.array(dtype=int64),
):
    # Retrieve the thread index as the limit index
    lid = wp.tid()

    # Retrieve the number of limits active in the model
    model_nl = limit_model_num_limits[0]

    # Skip if lid is greater than the number of limits active in the model
    if lid >= model_nl:
        return

    # Retrieve the world index and the world-relative limit index for this limit
    wid = limit_wid[lid]
    wlid = limit_lid[lid]
    dof = limit_dof[lid]

    # Compute the per-limit constraint residual (infinity-norm)
    r_cts_limits_l = wp.abs(limit_r_q[lid])

    # Update the per-world maximum residual
    previous_max = wp.atomic_max(metric_r_cts_limits, wid, r_cts_limits_l)
    if r_cts_limits_l >= previous_max:
        argmax_key = int64(build_pair_key2(uint32(wlid), uint32(dof)))
        wp.atomic_exch(metric_r_cts_limits_argmax, wid, argmax_key)


@wp.kernel
def _compute_cts_contacts_residual(
    # Inputs:
    contact_model_num_contacts: wp.array(dtype=int32),
    contact_wid: wp.array(dtype=int32),
    contact_cid: wp.array(dtype=int32),
    contact_gapfunc: wp.array(dtype=vec4f),
    # Outputs:
    metric_r_cts_contacts: wp.array(dtype=float32),
    metric_r_cts_contacts_argmax: wp.array(dtype=int32),
):
    # Retrieve the thread index as the contact index
    cid = wp.tid()

    # Retrieve the number of contacts active in the model
    model_nc = contact_model_num_contacts[0]

    # Skip if cid is greater than the number of contacts active in the model
    if cid >= model_nc:
        return

    # Retrieve the per-contact data
    wid = contact_wid[cid]
    wcid = contact_cid[cid]
    gapfunc = contact_gapfunc[cid]

    # Compute the per-contact constraint residual (infinity-norm)
    r_cts_contacts_k = wp.abs(gapfunc[3])

    # Update the per-world maximum residual and argmax index
    previous_max = wp.atomic_max(metric_r_cts_contacts, wid, r_cts_contacts_k)
    if r_cts_contacts_k >= previous_max:
        wp.atomic_exch(metric_r_cts_contacts_argmax, wid, wcid)


@wp.kernel
def _compute_dual_problem_metrics(
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
    solution_sigma: wp.array(dtype=vec2f),
    solution_lambdas: wp.array(dtype=float32),
    solution_v_plus: wp.array(dtype=float32),
    # Buffers:
    buffer_s: wp.array(dtype=float32),
    buffer_v: wp.array(dtype=float32),
    # Outputs:
    metric_r_v_plus: wp.array(dtype=float32),
    metric_r_v_plus_argmax: wp.array(dtype=int32),
    metric_r_ncp_primal: wp.array(dtype=float32),
    metric_r_ncp_primal_argmax: wp.array(dtype=int32),
    metric_r_ncp_dual: wp.array(dtype=float32),
    metric_r_ncp_dual_argmax: wp.array(dtype=int32),
    metric_r_ncp_compl: wp.array(dtype=float32),
    metric_r_ncp_compl_argmax: wp.array(dtype=int32),
    metric_r_vi_natmap: wp.array(dtype=float32),
    metric_r_vi_natmap_argmax: wp.array(dtype=int32),
    metric_f_ncp: wp.array(dtype=float32),
    metric_f_ccp: wp.array(dtype=float32),
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
    sigma = solution_sigma[wid]

    # Compute additional info
    njc = ncts - (nl + 3 * nc)

    # Compute the post-event constraint-space velocity from the current solution: v_plus = v_f + D @ lambda
    # NOTE: We assume the dual problem linear terms `D` and `v_f` have already been preconditioned in-place using `P`
    compute_v_plus(ncts, vio, mio, sigma[0], problem_P, problem_D, problem_v_f, solution_lambdas, buffer_v)

    # Compute the post-event constraint-space velocity error as: r_v_plus = || v_plus_est - v_plus_true ||_inf
    r_v_plus, r_v_plus_argmax = compute_vector_difference_infnorm(ncts, vio, solution_v_plus, buffer_v)

    # Compute the De Saxce correction for each contact as: s = G(v_plus)
    compute_desaxce_corrections(nc, cio, vio, ccgo, problem_mu, buffer_v, buffer_s)

    # Compute the CCP optimization objective as: f_ccp = 0.5 * lambda.dot(v_plus + v_f)
    f_ccp = 0.5 * compute_double_dot_product(ncts, vio, solution_lambdas, buffer_v, problem_v_f)

    # Compute the NCP optimization objective as:  f_ncp = f_ccp + lambda.dot(s)
    f_ncp = compute_dot_product(ncts, vio, solution_lambdas, buffer_s)
    f_ncp += f_ccp

    # Compute the augmented post-event constraint-space velocity as: v_aug = v_plus + s
    compute_vector_sum(ncts, vio, buffer_v, buffer_s, buffer_v)

    # Compute the NCP primal residual as: r_p := || lambda - proj_K(lambda) ||_inf
    r_ncp_p, r_ncp_p_argmax = compute_ncp_primal_residual(nl, nc, vio, lcgo, ccgo, cio, problem_mu, solution_lambdas)

    # Compute the NCP dual residual as: r_d := || v_plus + s - proj_dual_K(v_plus + s)  ||_inf
    r_ncp_d, r_ncp_d_argmax = compute_ncp_dual_residual(njc, nl, nc, vio, lcgo, ccgo, cio, problem_mu, buffer_v)

    # Compute the NCP complementarity (lambda _|_ (v_plus + s)) residual as r_c := || lambda.dot(v_plus + s) ||_inf
    r_ncp_c, r_ncp_c_argmax = compute_ncp_complementarity_residual(nl, nc, vio, lcgo, ccgo, buffer_v, solution_lambdas)

    # Compute the natural-map residuals as: r_natmap = || lambda - proj_K(lambda - (v + s)) ||_inf
    r_ncp_natmap, r_ncp_natmap_argmax = compute_ncp_natural_map_residual(
        nl, nc, vio, lcgo, ccgo, cio, problem_mu, buffer_v, solution_lambdas
    )

    # Store the computed metrics in the output arrays
    metric_r_v_plus[wid] = r_v_plus
    metric_r_v_plus_argmax[wid] = r_v_plus_argmax
    metric_r_ncp_primal[wid] = r_ncp_p
    metric_r_ncp_primal_argmax[wid] = r_ncp_p_argmax
    metric_r_ncp_dual[wid] = r_ncp_d
    metric_r_ncp_dual_argmax[wid] = r_ncp_d_argmax
    metric_r_ncp_compl[wid] = r_ncp_c
    metric_r_ncp_compl_argmax[wid] = r_ncp_c_argmax
    metric_r_vi_natmap[wid] = r_ncp_natmap
    metric_r_vi_natmap_argmax[wid] = r_ncp_natmap_argmax
    metric_f_ncp[wid] = f_ncp
    metric_f_ccp[wid] = f_ccp


###
# Interfaces
###


class SolutionMetrics:
    """
    Provides a high-level interface to compute a set of simulation solver performance metrics
    that provide quantitative measures of constraint satisfaction and physical plausibility of
    the computed solution to the dual forward-dynamics Nonlinear Complementarity Problem (NCP).

    This class is therefore responsible for managing the lifetime of all metrics data memory
    allocations, as well as provide an easy-to-use public API of the relevant operations.

    Internally, it holds an instance of the :class:`SolutionMetricsData` class. For more details
    about the specific metrics computed, please refer to the documentation of that class.
    """

    def __init__(self, model: Model | None = None, device: Devicelike = None):
        """
        Initializes the solution metrics evaluator.

        Args:
            model (Model):
                The model containing the time-invariant data of the simulation.
            device (Devicelike, optional):
                The device where the metrics data should be allocated.\n
                If not specified, the model's device will be used by default.
        """
        # Declare and initialize the target device
        # NOTE: This can be overridden during a
        # later call to `finalize()` if needed
        self._device: Devicelike = device

        # Declare the metrics data container
        self._data: SolutionMetricsData | None = None

        # Declare data buffers for metrics computations
        self._buffer_s: wp.array | None = None
        self._buffer_v: wp.array | None = None

        # If a model is provided, finalize the metrics data allocations
        if model is not None:
            self.finalize(model, device)

    def finalize(self, model: Model, device: Devicelike = None):
        """
        Finalizes the metrics data allocations on the specified device.

        Args:
            model (Model):
                The model containing the time-invariant data of the simulation.
            device (Devicelike, optional):
                The device where the metrics data should be allocated.\n
                If not specified, the model's device will be used by default.
        """
        # Ensure the model is valid
        if not isinstance(model, Model):
            raise TypeError("Expected 'model' to be of type Model.")

        # Set the target device for metrics data allocation and execution
        # If no device is specified, use the model's device by default
        self._device = device
        if self._device is None:
            self._device = model.device

        # Allocate metrics data on the target device
        with wp.ScopedDevice(self._device):
            # Allocate reusable buffers for metrics computations
            self._buffer_v = wp.zeros(model.size.sum_of_max_total_cts, dtype=float32)
            self._buffer_s = wp.zeros(model.size.sum_of_max_total_cts, dtype=float32)

            # Allocate the metrics container data arrays
            self._data = SolutionMetricsData(
                r_eom=wp.zeros(model.size.num_worlds, dtype=float32),
                r_eom_argmax=wp.full(model.size.num_worlds, value=-1, dtype=int64),
                r_kinematics=wp.zeros(model.size.num_worlds, dtype=float32),
                r_kinematics_argmax=wp.full(model.size.num_worlds, value=-1, dtype=int64),
                r_cts_joints=wp.zeros(model.size.num_worlds, dtype=float32),
                r_cts_joints_argmax=wp.full(model.size.num_worlds, value=-1, dtype=int64),
                r_cts_limits=wp.zeros(model.size.num_worlds, dtype=float32),
                r_cts_limits_argmax=wp.full(model.size.num_worlds, value=-1, dtype=int64),
                r_cts_contacts=wp.zeros(model.size.num_worlds, dtype=float32),
                r_cts_contacts_argmax=wp.full(model.size.num_worlds, value=-1, dtype=int32),
                r_v_plus=wp.zeros(model.size.num_worlds, dtype=float32),
                r_v_plus_argmax=wp.full(model.size.num_worlds, value=-1, dtype=int32),
                r_ncp_primal=wp.zeros(model.size.num_worlds, dtype=float32),
                r_ncp_primal_argmax=wp.full(model.size.num_worlds, value=-1, dtype=int32),
                r_ncp_dual=wp.zeros(model.size.num_worlds, dtype=float32),
                r_ncp_dual_argmax=wp.full(model.size.num_worlds, value=-1, dtype=int32),
                r_ncp_compl=wp.zeros(model.size.num_worlds, dtype=float32),
                r_ncp_compl_argmax=wp.full(model.size.num_worlds, value=-1, dtype=int32),
                r_vi_natmap=wp.zeros(model.size.num_worlds, dtype=float32),
                r_vi_natmap_argmax=wp.full(model.size.num_worlds, value=-1, dtype=int32),
                f_ncp=wp.zeros(model.size.num_worlds, dtype=float32),
                f_ccp=wp.zeros(model.size.num_worlds, dtype=float32),
            )

    ###
    # Properties
    ###

    @property
    def device(self) -> Devicelike:
        """
        Returns the device where the metrics data is allocated.
        """
        return self._device

    @property
    def data(self) -> SolutionMetricsData:
        """
        Returns the metrics data container.
        """
        self._assert_has_data()
        return self._data

    ###
    # Public Operations
    ###

    def reset(self):
        """
        Resets all metrics to zeros.
        """
        self._data.reset()

    def evaluate(
        self,
        sigma: wp.array,
        lambdas: wp.array,
        v_plus: wp.array,
        model: Model,
        data: ModelData,
        state_p: State,
        problem: DualProblem,
        jacobians: DenseSystemJacobians,
        limits: Limits | None = None,
        contacts: Contacts | None = None,
    ):
        """
        Evaluates all solution performance metrics.

        Args:
            model (Model):
                The model containing the time-invariant data of the simulation.
            data (ModelData):
                The model data containing the time-variant data of the simulation.
            state_p (State):
                The previous state of the simulation.
            limits (Limits):
                The joint-limits data describing active limit constraints.
            contacts (Contacts):
                The contact data describing active contact constraints.
            problem (DualProblem):
                The dual forward dynamics problem of the current time-step.
            jacobians (DenseSystemJacobians):
                The dense system Jacobians of the current time-step.
            sigma (wp.array):
                The array diagonal regularization applied to the Delassus matrix of the current dual problem.
            lambdas (wp.array):
                The array of constraint reactions (i.e. Lagrange multipliers) of the current dual problem solution.
            v_plus (wp.array):
                The array of post-event constraint-space velocities of the current dual problem solution.
        """
        self._assert_has_data()
        self._evaluate_constraint_violations_perf(model, data, limits, contacts)
        self._evaluate_primal_problem_perf(model, data, state_p, jacobians)
        self._evaluate_dual_problem_perf(sigma, lambdas, v_plus, problem)

    ###
    # Internals
    ###

    def _assert_has_data(self):
        """
        Asserts that the metrics data has been finalized and is available.

        Raises:
            RuntimeError: If the data is not available.
        """
        if self._data is None:
            raise RuntimeError("SolutionMetrics data has not been finalized. Call finalize() first.")

    def _evaluate_constraint_violations_perf(
        self,
        model: Model,
        data: ModelData,
        limits: Limits | None = None,
        contacts: Contacts | None = None,
    ):
        """
        Evaluates the constraint-violation performance metrics.

        Args:
            model (Model):
                The model containing the time-invariant data of the simulation.
            data (ModelData):
                The model data containing the time-variant data of the simulation.
            limits (Limits):
                The joint-limits data describing active limit constraints.
            contacts (Contacts):
                The contact data describing active contact constraints.
        """
        # Ensure metrics data is available
        self._assert_has_data()

        # Compute the largest configuration-level joint constraint residuals (i.e. violations)
        if model.size.sum_of_num_joints > 0:
            wp.launch(
                kernel=_compute_cts_joints_residual,
                dim=model.size.sum_of_num_joints,
                inputs=[
                    # Inputs:
                    model.info.joint_cts_offset,
                    model.joints.wid,
                    model.joints.num_cts,
                    model.joints.cts_offset,
                    data.joints.r_j,
                    # Outputs:
                    self._data.r_cts_joints,
                    self._data.r_cts_joints_argmax,
                ],
                device=model.device,
            )

        # Compute the largest joint-limit constraint residuals (i.e. penetrations)
        if limits is not None and limits.model_max_limits_host > 0:
            wp.launch(
                kernel=_compute_cts_limits_residual,
                dim=limits.data.model_max_limits_host,
                inputs=[
                    # Inputs:
                    limits.data.model_active_limits,
                    limits.data.wid,
                    limits.data.lid,
                    limits.data.dof,
                    limits.data.r_q,
                    # Outputs:
                    self._data.r_cts_limits,
                    self._data.r_cts_limits_argmax,
                ],
                device=model.device,
            )

        # Compute the largest contact constraint residuals (i.e. penetrations)
        if contacts is not None and contacts.model_max_contacts_host > 0:
            wp.launch(
                kernel=_compute_cts_contacts_residual,
                dim=contacts.data.model_max_contacts_host,
                inputs=[
                    # Inputs:
                    contacts.data.model_active_contacts,
                    contacts.data.wid,
                    contacts.data.cid,
                    contacts.data.gapfunc,
                    # Outputs:
                    self._data.r_cts_contacts,
                    self._data.r_cts_contacts_argmax,
                ],
                device=model.device,
            )

    def _evaluate_primal_problem_perf(
        self,
        model: Model,
        data: ModelData,
        state_p: State,
        jacobians: DenseSystemJacobians,
    ):
        """
        Evaluates the primal problem performance metrics.

        Args:
            model (Model):
                The model containing the time-invariant data of the simulation.
            data (ModelData):
                The model data containing the time-variant data of the simulation.
            state_p (State):
                The previous state of the simulation.
            jacobians (DenseSystemJacobians):
                The dense system Jacobians of the current time-step.
        """
        # Ensure metrics data is available
        self._assert_has_data()

        # Compute the equations-of-motion residuals
        wp.launch(
            kernel=_compute_eom_residual,
            dim=model.size.sum_of_num_bodies,
            inputs=[
                # Inputs:
                model.time.dt,
                model.gravity.vector,
                model.bodies.wid,
                model.bodies.m_i,
                data.bodies.I_i,
                data.bodies.w_i,
                data.bodies.u_i,
                state_p.u_i,
                # Outputs:
                self._data.r_eom,
                self._data.r_eom_argmax,
            ],
            device=model.device,
        )

        # Compute the kinematics constraint residuals,
        # i.e. velocity-level joint constraint equations
        if model.size.sum_of_num_joints > 0:
            wp.launch(
                kernel=_compute_joint_kinematics_residual,
                dim=model.size.sum_of_num_joints,
                inputs=[
                    # Inputs:
                    model.info.num_body_dofs,
                    model.info.bodies_offset,
                    model.joints.wid,
                    model.joints.num_cts,
                    model.joints.cts_offset,
                    model.joints.bid_B,
                    model.joints.bid_F,
                    data.bodies.u_i,
                    jacobians.data.J_cts_offsets,
                    jacobians.data.J_cts_data,
                    # Outputs:
                    self._data.r_kinematics,
                    self._data.r_kinematics_argmax,
                ],
                device=model.device,
            )

    def _evaluate_dual_problem_perf(
        self,
        sigma: wp.array,
        lambdas: wp.array,
        v_plus: wp.array,
        problem: DualProblem,
    ):
        """
        Evaluates the dual problem performance metrics.

        Args:
            problem (DualProblem):
                The dual problem containing the time-invariant and time-variant data of the simulation.
            sigma (wp.array):
                The array of sigma values for the dual problem.
            lambdas (wp.array):
                The array of lambda values for the dual problem.
            v_plus (wp.array):
                The array of v_plus values for the dual problem.
        """
        # Ensure metrics data is available
        self._assert_has_data()

        # Compute the dual problem NCP/VI performance metrics
        wp.launch(
            kernel=_compute_dual_problem_metrics,
            dim=problem.size.num_worlds,
            inputs=[
                # Inputs:
                problem.data.nl,
                problem.data.nc,
                problem.data.cio,
                problem.data.lcgo,
                problem.data.ccgo,
                problem.data.dim,
                problem.data.vio,
                problem.data.mio,
                problem.data.mu,
                problem.data.v_f,
                problem.data.D,
                problem.data.P,
                sigma,
                lambdas,
                v_plus,
                # Buffers:
                self._buffer_s,
                self._buffer_v,
                # Outputs:
                self._data.r_v_plus,
                self._data.r_v_plus_argmax,
                self._data.r_ncp_primal,
                self._data.r_ncp_primal_argmax,
                self._data.r_ncp_dual,
                self._data.r_ncp_dual_argmax,
                self._data.r_ncp_compl,
                self._data.r_ncp_compl_argmax,
                self._data.r_vi_natmap,
                self._data.r_vi_natmap_argmax,
                self._data.f_ncp,
                self._data.f_ccp,
            ],
            device=problem.device,
        )
