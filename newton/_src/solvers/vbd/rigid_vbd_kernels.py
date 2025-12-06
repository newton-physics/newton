# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Rigid body VBD solver kernels and utilities.

This module contains all rigid body-specific kernels, device functions, data structures,
and constants for the VBD solver's rigid body domain (AVBD algorithm).

Organization:
- Constants: Solver parameters and thresholds
- Data structures: RigidForceElementAdjacencyInfo and related structs
- Device functions: Helper functions for rigid body dynamics
- Utility kernels: Adjacency building
- Pre-iteration kernels: Forward integration, warmstarting, Dahl parameter computation
- Iteration kernels: Contact accumulation, rigid body solve, dual updates
- Post-iteration kernels: Velocity updates, Dahl state updates
"""

import warp as wp
from warp.types import float32, vector

from newton._src.core.spatial import quat_velocity
from newton._src.sim import JointType
from newton._src.solvers.solver import integrate_rigid_body

wp.set_module_options({"enable_backward": False})

# ---------------------------------
# Constants
# ---------------------------------

_SMALL_ANGLE_EPS = wp.constant(1.0e-7)
"""Small-angle threshold [rad] for guards and series expansions"""

_SMALL_ANGLE_EPS2 = _SMALL_ANGLE_EPS * _SMALL_ANGLE_EPS
"""Square of _SMALL_ANGLE_EPS"""

_SMALL_LENGTH_EPS = wp.constant(1.0e-9)
"""Small length tolerance (e.g., segment length checks)"""

_USE_SMALL_ANGLE_APPROX = wp.constant(True)
"""If True use first-order small-angle rotation approximation; if False use closed-form rotation update"""

_DAHL_KAPPADOT_DEADBAND = wp.constant(1.0e-6)
"""Deadband threshold for hysteresis direction selection"""

_NUM_CONTACT_THREADS_PER_BODY = 16
"""Threads per body for contact accumulation using strided iteration"""


# ---------------------------------
# Helper classes and device functions
# ---------------------------------


class vec6(vector(length=6, dtype=float32)):
    """Packed lower-triangular 3x3 matrix storage: [L00, L10, L11, L20, L21, L22]."""

    pass


@wp.func
def chol33(A: wp.mat33) -> vec6:
    """
    Compute Cholesky factorization A = L*L^T for 3x3 SPD matrix.

    Uses packed storage for lower-triangular L to save memory and improve cache efficiency.
    Packed format: [L00, L10, L11, L20, L21, L22] stores only the 6 non-zero elements.

    Algorithm: Standard column-by-column Cholesky decomposition
      Column 0: L00 = sqrt(a00), L10 = a10/L00, L20 = a20/L00
      Column 1: L11 = sqrt(a11 - L10^2), L21 = (a21 - L20*L10)/L11
      Column 2: L22 = sqrt(a22 - L20^2 - L21^2)

    Args:
        A: Symmetric positive-definite 3x3 matrix (only lower triangle is accessed)

    Returns:
        vec6: Packed lower-triangular Cholesky factor L
              Layout: [L00, L10, L11, L20, L21, L22]
              Represents: L = [[L00,   0,   0],
                               [L10, L11,   0],
                               [L20, L21, L22]]

    Note: Assumes A is SPD. No checking for negative square roots.
    """
    # Extract lower triangle (A is symmetric, only lower half needed)
    a00 = A[0, 0]
    a10 = A[1, 0]
    a11 = A[1, 1]
    a20 = A[2, 0]
    a21 = A[2, 1]
    a22 = A[2, 2]

    # Column 0: Compute first column of L
    L00 = wp.sqrt(a00)
    L10 = a10 / L00
    L20 = a20 / L00

    # Column 1: Compute second column of L
    L11 = wp.sqrt(a11 - L10 * L10)
    L21 = (a21 - L20 * L10) / L11

    # Column 2: Compute third column of L
    L22 = wp.sqrt(a22 - L20 * L20 - L21 * L21)

    # Pack into vec6: [L00, L10, L11, L20, L21, L22]
    return vec6(L00, L10, L11, L20, L21, L22)


@wp.func
def chol33_solve(Lp: vec6, b: wp.vec3) -> wp.vec3:
    """
    Solve A*x = b given packed Cholesky factorization A = L*L^T.

    Uses two-stage triangular solve:
      1. Forward substitution:  L*y = b   (solve for y)
      2. Backward substitution: L^T*x = y (solve for x)

    This is more efficient than computing A^-1 explicitly and avoids
    numerical issues from matrix inversion.

    Args:
        Lp: Packed lower-triangular Cholesky factor from chol33()
            Layout: [L00, L10, L11, L20, L21, L22]
        b: Right-hand side vector

    Returns:
        vec3: Solution x to A*x = b

    Complexity: 6 multiplies, 6 divides (optimal for 3x3)
    """
    # Unpack Cholesky factor for readability
    L00 = Lp[0]
    L10 = Lp[1]
    L11 = Lp[2]
    L20 = Lp[3]
    L21 = Lp[4]
    L22 = Lp[5]

    # Forward substitution: L*y = b
    y0 = b[0] / L00
    y1 = (b[1] - L10 * y0) / L11
    y2 = (b[2] - L20 * y0 - L21 * y1) / L22

    # Backward substitution: L^T*x = y
    x2 = y2 / L22
    x1 = (y1 - L21 * x2) / L11
    x0 = (y0 - L10 * x1 - L20 * x2) / L00

    return wp.vec3(x0, x1, x2)


@wp.func
def cable_get_kappa(q_wp: wp.quat, q_wc: wp.quat, q_wp_rest: wp.quat, q_wc_rest: wp.quat) -> wp.vec3:
    """Compute cable bending curvature vector kappa in the parent frame.

    Kappa is the rotation vector (theta*axis) from the rest-aligned relative rotation.

    Args:
        q_wp: Parent orientation (world).
        q_wc: Child orientation (world).
        q_wp_rest: Parent rest orientation (world).
        q_wc_rest: Child rest orientation (world).

    Returns:
        wp.vec3: Curvature vector kappa in parent frame (rotation vector form).
    """
    # Build R_align = R_rel * R_rel_rest^T using quaternions
    q_rel = wp.mul(wp.quat_inverse(q_wp), q_wc)
    q_rel_rest = wp.mul(wp.quat_inverse(q_wp_rest), q_wc_rest)
    q_align = wp.mul(q_rel, wp.quat_inverse(q_rel_rest))

    # Enforce shortest path (w > 0) to avoid double-cover ambiguity
    if q_align[3] < 0.0:
        q_align = wp.quat(-q_align[0], -q_align[1], -q_align[2], -q_align[3])

    # Log map to rotation vector
    axis, angle = wp.quat_to_axis_angle(q_align)
    return axis * angle


@wp.func
def compute_right_jacobian_inverse(kappa: wp.vec3) -> wp.mat33:
    """Inverse right Jacobian Jr^{-1}(kappa) for SO(3) rotation vectors.

    Args:
        kappa: Rotation vector theta*axis (any frame).

    Returns:
        wp.mat33: Jr^{-1}(kappa) in the same frame as kappa.
    """
    theta = wp.length(kappa)
    kappa_skew = wp.skew(kappa)

    if (theta < _SMALL_ANGLE_EPS) or (_USE_SMALL_ANGLE_APPROX):
        return wp.identity(3, float) + 0.5 * kappa_skew + (1.0 / 12.0) * (kappa_skew * kappa_skew)

    sin_theta = wp.sin(theta)
    cos_theta = wp.cos(theta)
    b = (1.0 / (theta * theta)) - (1.0 + cos_theta) / (2.0 * theta * sin_theta)
    return wp.identity(3, float) + 0.5 * kappa_skew + b * (kappa_skew * kappa_skew)


@wp.func
def compute_kappa_dot_analytic(
    q_wp: wp.quat,
    q_wc: wp.quat,
    q_wp_rest: wp.quat,
    q_wc_rest: wp.quat,
    omega_p_world: wp.vec3,
    omega_c_world: wp.vec3,
    kappa_now: wp.vec3,
) -> wp.vec3:
    """Analytical time derivative of curvature vector d(kappa)/dt in parent frame.

    R_align = R_rel * R_rel_rest^T represents the rotation from rest to current configuration,
    which is the same deformation measure used in cable_get_kappa. This removes the rest offset
    so bending is measured relative to the undeformed state.

    Args:
        q_wp: Parent orientation (world).
        q_wc: Child orientation (world).
        q_wp_rest: Parent rest orientation (world).
        q_wc_rest: Child rest orientation (world).
        omega_p_world: Parent angular velocity (world) [rad/s].
        omega_c_world: Child angular velocity (world) [rad/s].
        kappa_now: Current curvature vector in parent frame.

    Returns:
        wp.vec3: Curvature rate kappa_dot in parent frame [rad/s].
    """
    R_wp = wp.quat_to_matrix(q_wp)
    omega_rel_parent = wp.transpose(R_wp) * (omega_c_world - omega_p_world)

    R_wc = wp.quat_to_matrix(q_wc)
    R_wp_r = wp.quat_to_matrix(q_wp_rest)
    R_wc_r = wp.quat_to_matrix(q_wc_rest)
    R_rel = wp.transpose(R_wp) * R_wc
    R_rel_rest = wp.transpose(R_wp_r) * R_wc_r
    R_align = R_rel * wp.transpose(R_rel_rest)

    Jr_inv = compute_right_jacobian_inverse(kappa_now)
    omega_right = wp.transpose(R_align) * omega_rel_parent
    return Jr_inv * omega_right


@wp.func
def evaluate_cable_bend_force_hessian_avbd(
    q_wp: wp.quat,
    q_wc: wp.quat,
    q_wp_rest: wp.quat,
    q_wc_rest: wp.quat,
    q_wp_prev: wp.quat,
    q_wc_prev: wp.quat,
    is_parent: bool,
    k_eff: float,
    sigma0: wp.vec3,  # Dahl friction initial stress
    C_fric: wp.vec3,  # Dahl friction stiffness d(sigma)/d(kappa)
    damping: float,  # Rayleigh damping coefficient
    dt: float,
):
    """
    AVBD cable bend (component-wise Dahl): generalized torque and angular Hessian.

    Summary:
      - Curvature kappa from rest-aligned relative rotation
      - Generalized local force: f_kappa = k_eff * kappa + sigma0 + f_damp
      - Local Hessian in kappa-space: H_local = diag(k_eff + C_fric + k_damp),
        where C_fric is the Dahl stiffness d(sigma)/d(kappa)
      - Map once: J = R_wp * (R_align * Jr^{-T}), tau = J * f_kappa, H_aa = J * H_local * J^T
      - Rayleigh damping in kappa-space contributes k_damp = (damping/dt) * k_eff on the diagonal

    Frames:
      - All outputs (tau_world, H_aa) are in world frame.
    """
    inv_dt = 1.0 / dt

    kappa_now_vec = cable_get_kappa(q_wp, q_wc, q_wp_rest, q_wc_rest)

    # Right-trivialized mechanics: use Jr^{-T} for virtual-work pullback
    Jr_inv = compute_right_jacobian_inverse(kappa_now_vec)
    R_wp = wp.quat_to_matrix(q_wp)

    # Build the same R_align used in cable_get_kappa
    R_wc = wp.quat_to_matrix(q_wc)
    R_wp_r = wp.quat_to_matrix(q_wp_rest)
    R_wc_r = wp.quat_to_matrix(q_wc_rest)
    R_rel = wp.transpose(R_wp) * R_wc
    R_rel_rest = wp.transpose(R_wp_r) * R_wc_r
    R_align = R_rel * wp.transpose(R_rel_rest)

    # Virtual-work pullback consistent with kappa_dot = Jr_inv * (R_align^T * (R_wp^T * w_rel_world))
    # tau_world = R_wp * (A^T * f_kappa), where A = R_align * Jr_inv
    # Therefore J_world = R_wp * A^T = R_wp * (R_align * Jr_inv^T)
    J_world = R_wp * (R_align * wp.transpose(Jr_inv))

    # Elastic: f_elastic = k_eff * kappa  [N*m]
    f_elastic_local = k_eff * kappa_now_vec

    # Component-wise Dahl friction: sigma_i = sigma0_i + C_fric_i * kappa_dot_i
    f_local = f_elastic_local + sigma0

    # Defer mapping to world torque until all local forces are accumulated, then map once
    # Base Hessian in kappa-space (component-wise Dahl)
    k_fric_0 = C_fric[0]
    k_fric_1 = C_fric[1]
    k_fric_2 = C_fric[2]
    k0 = k_eff + k_fric_0
    k1 = k_eff + k_fric_1
    k2 = k_eff + k_fric_2

    # Optional Rayleigh damping (independent of Dahl)
    if damping > 0.0:
        # World angular velocities
        omega_p_world = quat_velocity(q_wp, q_wp_prev, dt)
        omega_c_world = quat_velocity(q_wc, q_wc_prev, dt)

        # Consistent right-trivialized, aligned rate:
        # kappa_dot = Jr_inv(kappa_now) * (R_align^T * (R_wp^T * (omega_c - omega_p)))
        dkappa_dt_vec = compute_kappa_dot_analytic(
            q_wp, q_wc, q_wp_rest, q_wc_rest, omega_p_world, omega_c_world, kappa_now_vec
        )
        f_damp_local = (damping * k_eff) * dkappa_dt_vec

        # Accumulate damping in local kappa-space
        f_local = f_local + f_damp_local

        # Damping Hessian in kappa-space
        k_damp = (damping * inv_dt) * k_eff
        k0 = k0 + k_damp
        k1 = k1 + k_damp
        k2 = k2 + k_damp

    # Build final H_local (including damping if any) and map once to world
    H_local = wp.mat33(
        k0,
        0.0,
        0.0,
        0.0,
        k1,
        0.0,
        0.0,
        0.0,
        k2,
    )
    H_aa = J_world * (H_local * wp.transpose(J_world))

    # Map total local force once to world-space torque
    tau_world = J_world * f_local
    if not is_parent:
        tau_world = -tau_world

    return tau_world, H_aa


@wp.func
def evaluate_cable_stretch_force_hessian_avbd(
    X_wp: wp.transform,
    X_wc: wp.transform,
    X_wp_prev: wp.transform,
    X_wc_prev: wp.transform,
    parent_pose: wp.transform,
    child_pose: wp.transform,
    parent_com: wp.vec3,
    child_com: wp.vec3,
    is_parent: bool,
    penalty_k: float,
    damping: float,
    dt: float,
):
    """
    AVBD cable stretch: point-to-point pinning (C = x_c - x_p).

    Summary:
      - Force at attachment point: f_attachment = k*C + (damping*k)*dC/dt
      - Map to body: force = +/-f_attachment (sign depends on parent/child); torque = r x force
      - Gauss-Newton Hessian: K = k * I, J = [I, -[r]_x]

    Frames:
      - All outputs (force, torque, Hessian blocks) are in world frame.

    Args:
      - X_wp, X_wc: Parent/child joint frames (world)
      - X_wp_prev, X_wc_prev: Previous-step joint frames (for damping)
      - parent_pose, child_pose: Body poses (for COM mapping)
      - parent_com, child_com: COM offsets in body frames
      - is_parent: True if contributions are for the parent body
      - penalty_k: Spring stiffness [N/m]
      - damping: Rayleigh damping coefficient (dimensionless)
      - dt: Timestep [s]

    Returns:
      - force (wp.vec3): Linear force (world)
      - torque (wp.vec3): Angular torque (world)
      - H_ll (wp.mat33): Linear-linear block
      - H_al (wp.mat33): Angular-linear block
      - H_aa (wp.mat33): Angular-angular block
    """
    # Extract attachment points in world space
    x_p = wp.transform_get_translation(X_wp)
    x_c = wp.transform_get_translation(X_wc)

    # Compute moment arm from body COM to attachment point
    if is_parent:
        com_w = wp.transform_point(parent_pose, parent_com)
        r = x_p - com_w
    else:
        com_w = wp.transform_point(child_pose, child_com)
        r = x_c - com_w

    # Constraint violation: C = x_c - x_p (child attachment - parent attachment)
    C_vec = x_c - x_p

    # Force at attachment point (elastic + damping)
    f_attachment = penalty_k * C_vec

    # Gauss-Newton Hessian: H = J^T * K * J
    # where J = [I, -[r]_x] is the contact-point Jacobian (spatial vector format: [v, w])
    # and K = k * I is the point stiffness
    rx = wp.skew(r)
    K_point = penalty_k * wp.identity(3, float)

    H_ll = K_point
    H_al = rx * K_point
    H_aa = wp.transpose(rx) * K_point * rx

    # Rayleigh damping
    if damping > 0.0:
        # Constraint velocity: dC/dt = (C_now - C_prev) / dt
        x_p_prev = wp.transform_get_translation(X_wp_prev)
        x_c_prev = wp.transform_get_translation(X_wc_prev)
        C_vec_prev = x_c_prev - x_p_prev
        inv_dt = 1.0 / dt
        dC_dt = (C_vec - C_vec_prev) * inv_dt

        # Damping coefficient: D = damping * k
        damping_coeff = damping * penalty_k

        # Damping force at attachment point
        f_damping = damping_coeff * dC_dt
        f_attachment = f_attachment + f_damping

        # Damping Hessian: H_damp = (D/dt) * H_elastic = (damping/dt) * H_elastic
        damp_scale = damping * inv_dt
        H_ll_damp = damp_scale * H_ll
        H_al_damp = damp_scale * H_al
        H_aa_damp = damp_scale * H_aa

        # Accumulate damping contributions
        H_ll = H_ll + H_ll_damp
        H_al = H_al + H_al_damp
        H_aa = H_aa + H_aa_damp

    force = f_attachment if is_parent else -f_attachment
    torque = wp.cross(r, force)

    return force, torque, H_ll, H_al, H_aa


# ---------------------------------
# Data structures
# ---------------------------------


@wp.struct
class RigidForceElementAdjacencyInfo:
    r"""
    Stores adjacency information for rigid bodies and their connected joints using CSR (Compressed Sparse Row) format.

    - body_adj_joints: Flattened array of joint IDs. Size is sum over all bodies of N_i, where N_i is the
      number of joints connected to body i.

    - body_adj_joints_offsets: Offset array indicating where each body's joint list starts.
      Size is |B|+1 (number of bodies + 1).
      The number of joints adjacent to body i is: body_adj_joints_offsets[i+1] - body_adj_joints_offsets[i]
    """

    # Rigid body joint adjacency
    body_adj_joints: wp.array(dtype=wp.int32)
    body_adj_joints_offsets: wp.array(dtype=wp.int32)

    def to(self, device):
        if device == self.body_adj_joints.device:
            return self
        else:
            adjacency_gpu = RigidForceElementAdjacencyInfo()
            adjacency_gpu.body_adj_joints = self.body_adj_joints.to(device)
            adjacency_gpu.body_adj_joints_offsets = self.body_adj_joints_offsets.to(device)

            return adjacency_gpu


@wp.func
def get_body_num_adjacent_joints(adjacency: RigidForceElementAdjacencyInfo, body: wp.int32):
    """Number of joints adjacent to the given body from CSR offsets."""
    return adjacency.body_adj_joints_offsets[body + 1] - adjacency.body_adj_joints_offsets[body]


@wp.func
def get_body_adjacent_joint_id(adjacency: RigidForceElementAdjacencyInfo, body: wp.int32, joint: wp.int32):
    """Joint id at local index `joint` within the body's CSR-adjacent joint list."""
    offset = adjacency.body_adj_joints_offsets[body]
    return adjacency.body_adj_joints[offset + joint]


@wp.func
def evaluate_rigid_contact_from_collision(
    body_a_index: int,
    body_b_index: int,
    body_q: wp.array(dtype=wp.transform),
    body_q_anchor: wp.array(dtype=wp.transform),
    body_com: wp.array(dtype=wp.vec3),
    contact_point_a_local: wp.vec3,  # Local contact point on body A
    contact_point_b_local: wp.vec3,  # Local contact point on body B
    contact_normal: wp.vec3,  # Contact normal (A to B)
    penetration_depth: float,  # Penetration depth (> 0 when penetrating)
    contact_ke: float,  # Contact normal stiffness
    contact_kd: float,  # Contact damping coefficient
    friction_mu: float,  # Coulomb friction coefficient
    friction_epsilon: float,  # Friction regularization parameter
    dt: float,
):
    """Compute contact forces and 3x3 Hessian blocks for a rigid contact pair.

    All returned forces, torques, and Hessians are in world frame.

    Args:
        body_a_index: Body A index (-1 for static/kinematic).
        body_b_index: Body B index (-1 for static/kinematic).
        body_q: Current body transforms (world frame).
        body_q_anchor: Anchor body transforms (world frame) used as the
            "previous" pose for finite-difference velocity over the damping
            window.
        body_com: Body COM offsets (local body frame).
        contact_point_a_local: Contact point on A (local body frame).
        contact_point_b_local: Contact point on B (local body frame).
        contact_normal: Unit normal from A to B (world frame).
        penetration_depth: Penetration depth (> 0 when penetrating).
        contact_ke: Normal stiffness.
        contact_kd: Damping coefficient.
        friction_mu: Coulomb friction coefficient.
        friction_epsilon: Friction regularization length.
        dt: Effective damping window [s].

    Returns:
        tuple: (force_a, torque_a, h_ll_a, h_al_a, h_aa_a,
                force_b, torque_b, h_ll_b, h_al_b, h_aa_b)
            - h_ll: Linear-linear block
            - h_al: Angular-linear block
            - h_aa: Angular-angular block
    """

    # Early exit: no penetration or zero stiffness
    if penetration_depth <= 0.0 or contact_ke <= 0.0:
        zero_vec = wp.vec3(0.0)
        zero_mat = wp.mat33(0.0)
        return (zero_vec, zero_vec, zero_mat, zero_mat, zero_mat, zero_vec, zero_vec, zero_mat, zero_mat, zero_mat)

    if body_a_index < 0:
        X_wa = wp.transform_identity()
        X_wa_prev = wp.transform_identity()
        body_a_com_local = wp.vec3(0.0)
    else:
        X_wa = body_q[body_a_index]
        X_wa_prev = body_q_anchor[body_a_index]
        body_a_com_local = body_com[body_a_index]

    if body_b_index < 0:
        X_wb = wp.transform_identity()
        X_wb_prev = wp.transform_identity()
        body_b_com_local = wp.vec3(0.0)
    else:
        X_wb = body_q[body_b_index]
        X_wb_prev = body_q_anchor[body_b_index]
        body_b_com_local = body_com[body_b_index]

    # Centers of mass in world coordinates
    x_com_a_now = wp.transform_point(X_wa, body_a_com_local)
    x_com_b_now = wp.transform_point(X_wb, body_b_com_local)

    # Contact points in world coordinates (current and previous)
    x_c_a_now = wp.transform_point(X_wa, contact_point_a_local)
    x_c_b_now = wp.transform_point(X_wb, contact_point_b_local)
    x_c_a_prev = wp.transform_point(X_wa_prev, contact_point_a_local)
    x_c_b_prev = wp.transform_point(X_wb_prev, contact_point_b_local)

    # Contact motion for damping and friction (finite difference velocity estimation)
    dx_a = x_c_a_now - x_c_a_prev  # Motion of contact point on A over timestep dt
    dx_b = x_c_b_now - x_c_b_prev  # Motion of contact point on B over timestep dt
    dx_rel = dx_b - dx_a  # Relative contact motion (B relative to A)

    # Contact geometry - assume contact_normal is already unit length from collision detection

    # Normal force: f = contact_ke * penetration
    n_outer = wp.outer(contact_normal, contact_normal)
    f_total = contact_normal * (contact_ke * penetration_depth)
    K_total = contact_ke * n_outer

    # Compute relative velocity for damping and friction
    v_rel = dx_rel / dt
    v_dot_n = wp.dot(contact_normal, v_rel)

    # Damping only when compressing (v_n < 0, bodies approaching)
    if contact_kd > 0.0 and v_dot_n < 0.0:
        damping_coeff = contact_kd * contact_ke
        damping_force = -damping_coeff * v_dot_n * contact_normal
        damping_hessian = (damping_coeff / dt) * n_outer
        f_total = f_total + damping_force
        K_total = K_total + damping_hessian

    # Normal load for friction (elastic normal only)
    normal_load = contact_ke * penetration_depth

    # Friction forces (isotropic, no explicit tangent basis)
    if friction_mu > 0.0 and normal_load > 0.0:
        # Tangential slip (world space)
        v_n = contact_normal * v_dot_n
        v_t = v_rel - v_n
        u = v_t * dt
        eps_u = friction_epsilon * dt

        # Projected isotropic friction (no explicit tangent basis)
        f_friction, K_friction = compute_projected_isotropic_friction(
            friction_mu, normal_load, contact_normal, u, eps_u
        )
        f_total = f_total + f_friction
        K_total = K_total + K_friction

    # Split total contact force to both bodies (Newton's 3rd law)
    force_a = -f_total  # Force on A (opposite to normal, pushes A away from B)
    force_b = f_total  # Force on B (along normal, pushes B away from A)

    # Torque arms and resulting torques
    r_a = x_c_a_now - x_com_a_now  # Moment arm from A's COM to contact point
    r_b = x_c_b_now - x_com_b_now  # Moment arm from B's COM to contact point

    # Angular/linear coupling using contact-point Jacobian J = [-[r]x, I]
    r_a_skew = wp.skew(r_a)
    r_a_skew_T_K = wp.transpose(r_a_skew) * K_total
    torque_a = wp.cross(r_a, force_a)
    h_aa_a = r_a_skew_T_K * r_a_skew
    h_al_a = -r_a_skew_T_K

    h_ll_a = K_total  # Linear-linear

    r_b_skew = wp.skew(r_b)
    r_b_skew_T_K = wp.transpose(r_b_skew) * K_total
    torque_b = wp.cross(r_b, force_b)
    h_aa_b = r_b_skew_T_K * r_b_skew
    h_al_b = -r_b_skew_T_K

    h_ll_b = K_total  # Linear-linear

    return (force_a, torque_a, h_ll_a, h_al_a, h_aa_a, force_b, torque_b, h_ll_b, h_al_b, h_aa_b)


@wp.func
def evaluate_body_particle_contact(
    particle_index: int,
    particle_pos: wp.vec3,
    particle_prev_pos: wp.vec3,
    contact_index: int,
    body_particle_contact_ke: float,
    body_particle_contact_kd: float,
    friction_mu: float,
    friction_epsilon: float,
    particle_radius: wp.array(dtype=float),
    shape_material_mu: wp.array(dtype=float),
    shape_body: wp.array(dtype=int),
    body_q: wp.array(dtype=wp.transform),
    body_q_anchor: wp.array(dtype=wp.transform),
    body_qd: wp.array(dtype=wp.spatial_vector),
    body_com: wp.array(dtype=wp.vec3),
    contact_shape: wp.array(dtype=int),
    contact_body_pos: wp.array(dtype=wp.vec3),
    contact_body_vel: wp.array(dtype=wp.vec3),
    contact_normal: wp.array(dtype=wp.vec3),
    dt: float,
):
    """
    Evaluate particle-rigid body contact force and Hessian (on particle side).

    Computes contact forces and Hessians for a particle interacting with a rigid body shape.
    The function is agnostic to whether the rigid body is static, kinematic, or dynamic.

    Contact model:
    - Normal: Linear spring-damper (stiffness: body_particle_contact_ke, damping: body_particle_contact_kd)
    - Friction: 3D projector-based Coulomb friction with IPC regularization
    - Normal direction: Points from rigid surface towards particle (into particle)

    Args:
        particle_index: Index of the particle
        particle_pos: Current particle position (world frame)
        particle_prev_pos: Anchor particle position (world frame), used as the
            "previous" position for finite-difference velocity over the
            damping window.
        contact_index: Index in the body-particle contact arrays
        body_particle_contact_ke: Contact stiffness (model-level or AVBD adaptive)
        body_particle_contact_kd: Contact damping (model-level or AVBD averaged)
        friction_mu: Friction coefficient (model-level or AVBD averaged)
        friction_epsilon: Friction regularization distance
        particle_radius: Array of particle radii
        shape_material_mu: Array of shape friction coefficients
        shape_body: Array mapping shape index to body index
        body_q: Current body transforms
        body_q_anchor: Anchor body transforms (for finite-difference body
            velocity when available)
        body_qd: Body spatial velocities (fallback when no anchor is used)
        body_com: Body centers of mass (local frame)
        contact_shape: Array of shape indices for each soft contact
        contact_body_pos: Array of contact points (local to shape)
        contact_body_vel: Array of contact velocities (local frame)
        contact_normal: Array of contact normals (world frame, from rigid to particle)
        dt: Effective damping window [s].

    Returns:
        tuple[wp.vec3, wp.mat33]: (force, Hessian) on the particle (world frame)
    """
    shape_index = contact_shape[contact_index]
    body_index = shape_body[shape_index]

    X_wb = wp.transform_identity()
    X_com = wp.vec3()
    if body_index >= 0:
        X_wb = body_q[body_index]
        X_com = body_com[body_index]

    # body position in world space
    bx = wp.transform_point(X_wb, contact_body_pos[contact_index])

    n = contact_normal[contact_index]

    penetration_depth = -(wp.dot(n, particle_pos - bx) - particle_radius[particle_index])
    if penetration_depth > 0.0:
        body_contact_force_norm = penetration_depth * body_particle_contact_ke
        body_contact_force = n * body_contact_force_norm
        body_contact_hessian = body_particle_contact_ke * wp.outer(n, n)

        # Use the larger of body-particle friction and shape material friction
        mu = wp.max(friction_mu, shape_material_mu[shape_index])

        dx = particle_pos - particle_prev_pos

        if wp.dot(n, dx) < 0.0:
            # Damping coefficient is scaled by contact stiffness (consistent with rigid-rigid)
            damping_coeff = body_particle_contact_kd * body_particle_contact_ke
            damping_hessian = (damping_coeff / dt) * wp.outer(n, n)
            body_contact_hessian = body_contact_hessian + damping_hessian
            body_contact_force = body_contact_force - damping_hessian * dx

        # body velocity
        if body_q_anchor:
            # if body_q_anchor is available, compute velocity using finite difference method
            # this is more accurate for simulating static friction
            X_wb_prev = wp.transform_identity()
            if body_index >= 0:
                X_wb_prev = body_q_anchor[body_index]
            bx_prev = wp.transform_point(X_wb_prev, contact_body_pos[contact_index])
            bv = (bx - bx_prev) / dt + wp.transform_vector(X_wb, contact_body_vel[contact_index])

        else:
            # otherwise use the instantaneous velocity
            r = bx - wp.transform_point(X_wb, X_com)
            body_v_s = wp.spatial_vector()
            if body_index >= 0:
                body_v_s = body_qd[body_index]

            body_w = wp.spatial_bottom(body_v_s)
            body_v = wp.spatial_top(body_v_s)

            # compute the body velocity at the particle position
            bv = body_v + wp.cross(body_w, r) + wp.transform_vector(X_wb, contact_body_vel[contact_index])

        relative_translation = dx - bv * dt

        # Friction using 3D projector approach (consistent with rigid-rigid contacts)
        eps_u = friction_epsilon * dt
        friction_force, friction_hessian = compute_projected_isotropic_friction(
            mu, body_contact_force_norm, n, relative_translation, eps_u
        )
        body_contact_force = body_contact_force + friction_force
        body_contact_hessian = body_contact_hessian + friction_hessian
    else:
        body_contact_force = wp.vec3(0.0, 0.0, 0.0)
        body_contact_hessian = wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    return body_contact_force, body_contact_hessian


@wp.func
def compute_projected_isotropic_friction(
    friction_mu: float,
    normal_load: float,
    n_hat: wp.vec3,
    slip_u: wp.vec3,
    eps_u: float,
) -> tuple[wp.vec3, wp.mat33]:
    """Isotropic Coulomb friction in world frame using projector P = I - n n^T.

    Regularization: if ||u_t|| <= eps_u, uses a linear ramp; otherwise 1/||u_t||.

    Args:
        friction_mu: Coulomb friction coefficient (>= 0).
        normal_load: Normal load magnitude (>= 0).
        n_hat: Unit contact normal (world frame).
        slip_u: Tangential slip displacement over dt (world frame).
        eps_u: Smoothing distance (same units as slip_u, > 0).

    Returns:
        tuple[wp.vec3, wp.mat33]: (force, Hessian) in world frame.
    """
    # Tangential slip in the contact tangent plane without forming P: u_t = u - n * (n dot u)
    dot_nu = wp.dot(n_hat, slip_u)
    u_t = slip_u - n_hat * dot_nu
    u_norm = wp.length(u_t)

    if u_norm > 0.0:
        # IPC-style regularization
        if u_norm > eps_u:
            f1_SF_over_x = 1.0 / u_norm
        else:
            f1_SF_over_x = (-u_norm / eps_u + 2.0) / eps_u

        # Factor common scalar; force aligned with u_t, Hessian proportional to projector
        scale = friction_mu * normal_load * f1_SF_over_x
        f = -(scale * u_t)
        K = scale * (wp.identity(3, float) - wp.outer(n_hat, n_hat))
    else:
        f = wp.vec3(0.0)
        K = wp.mat33(0.0)

    return f, K


@wp.func
def evaluate_joint_force_hessian(
    body_index: int,
    joint_index: int,
    body_q: wp.array(dtype=wp.transform),
    body_q_anchor: wp.array(dtype=wp.transform),
    body_q_rest: wp.array(dtype=wp.transform),
    body_com: wp.array(dtype=wp.vec3),
    joint_type: wp.array(dtype=int),
    joint_parent: wp.array(dtype=int),
    joint_child: wp.array(dtype=int),
    joint_X_p: wp.array(dtype=wp.transform),
    joint_X_c: wp.array(dtype=wp.transform),
    joint_qd_start: wp.array(dtype=int),
    joint_target_kd: wp.array(dtype=float),
    joint_penalty_k: wp.array(dtype=float),
    joint_sigma_start: wp.array(dtype=wp.vec3),
    joint_C_fric: wp.array(dtype=wp.vec3),
    dt: float,
):
    """Compute AVBD joint force and Hessian contributions for a specific body (cable joints).

    Uses ``body_q_anchor`` as the previous pose when estimating damping and
    friction over the long-range window ``dt``, while ``body_q`` and
    ``body_q_rest`` provide current and rest configurations. Returns
    (force, torque, H_ll, H_al, H_aa) in world frame.
    """
    if joint_type[joint_index] != JointType.CABLE:
        return wp.vec3(0.0), wp.vec3(0.0), wp.mat33(0.0), wp.mat33(0.0), wp.mat33(0.0)

    parent_index = joint_parent[joint_index]
    child_index = joint_child[joint_index]
    if parent_index < 0 or (body_index != parent_index and body_index != child_index):
        return wp.vec3(0.0), wp.vec3(0.0), wp.mat33(0.0), wp.mat33(0.0), wp.mat33(0.0)

    is_parent_body = body_index == parent_index

    X_pj = joint_X_p[joint_index]
    X_cj = joint_X_c[joint_index]

    parent_pose = body_q[parent_index]
    child_pose = body_q[child_index]
    parent_pose_prev = body_q_anchor[parent_index]
    child_pose_prev = body_q_anchor[child_index]
    parent_pose_rest = body_q_rest[parent_index]
    child_pose_rest = body_q_rest[child_index]
    parent_com = body_com[parent_index]
    child_com = body_com[child_index]

    X_wp = parent_pose * X_pj
    X_wc = child_pose * X_cj
    X_wp_prev = parent_pose_prev * X_pj
    X_wc_prev = child_pose_prev * X_cj
    X_wp_rest = parent_pose_rest * X_pj
    X_wc_rest = child_pose_rest * X_cj

    dof_start = joint_qd_start[joint_index]
    stretch_penalty_k = joint_penalty_k[dof_start]
    bend_penalty_k = joint_penalty_k[dof_start + 1]
    stretch_damping = joint_target_kd[dof_start]
    bend_damping = joint_target_kd[dof_start + 1]

    # Cable segment convention: parent_com[2] stores half-length in local frame
    rest_length = 2.0 * parent_com[2]
    if rest_length < _SMALL_LENGTH_EPS:
        return wp.vec3(0.0), wp.vec3(0.0), wp.mat33(0.0), wp.mat33(0.0), wp.mat33(0.0)

    total_force = wp.vec3(0.0)
    total_torque = wp.vec3(0.0)
    total_H_ll = wp.mat33(0.0)
    total_H_al = wp.mat33(0.0)
    total_H_aa = wp.mat33(0.0)

    if bend_penalty_k > 0.0:
        q_wp = wp.transform_get_rotation(X_wp)
        q_wc = wp.transform_get_rotation(X_wc)
        q_wp_rest = wp.transform_get_rotation(X_wp_rest)
        q_wc_rest = wp.transform_get_rotation(X_wc_rest)
        q_wp_prev = wp.transform_get_rotation(X_wp_prev)
        q_wc_prev = wp.transform_get_rotation(X_wc_prev)
        bend_k_eff = bend_penalty_k / rest_length
        sigma0 = joint_sigma_start[joint_index]
        C_fric = joint_C_fric[joint_index]
        bend_torque, bend_H_aa = evaluate_cable_bend_force_hessian_avbd(
            q_wp,
            q_wc,
            q_wp_rest,
            q_wc_rest,
            q_wp_prev,
            q_wc_prev,
            is_parent_body,
            bend_k_eff,
            sigma0,
            C_fric,
            bend_damping,
            dt,
        )
        total_torque = total_torque + bend_torque
        total_H_aa = total_H_aa + bend_H_aa

    if stretch_penalty_k > 0.0:
        f_s, t_s, Hll_s, Hal_s, Haa_s = evaluate_cable_stretch_force_hessian_avbd(
            X_wp,
            X_wc,
            X_wp_prev,
            X_wc_prev,
            parent_pose,
            child_pose,
            parent_com,
            child_com,
            is_parent_body,
            stretch_penalty_k,
            stretch_damping,
            dt,
        )
        total_force = total_force + f_s
        total_torque = total_torque + t_s
        total_H_ll = total_H_ll + Hll_s
        total_H_al = total_H_al + Hal_s
        total_H_aa = total_H_aa + Haa_s

    return total_force, total_torque, total_H_ll, total_H_al, total_H_aa


# -----------------------------
# Utility kernels
# -----------------------------
@wp.kernel
def count_num_adjacent_joints(
    joint_parent: wp.array(dtype=wp.int32),
    joint_child: wp.array(dtype=wp.int32),
    num_body_adjacent_joints: wp.array(dtype=wp.int32),
):
    joint_count = joint_parent.shape[0]
    for joint_id in range(joint_count):
        parent_id = joint_parent[joint_id]
        child_id = joint_child[joint_id]

        # Skip world joints (parent/child == -1)
        if parent_id >= 0:
            num_body_adjacent_joints[parent_id] = num_body_adjacent_joints[parent_id] + 1
        if child_id >= 0:
            num_body_adjacent_joints[child_id] = num_body_adjacent_joints[child_id] + 1


@wp.kernel
def fill_adjacent_joints(
    joint_parent: wp.array(dtype=wp.int32),
    joint_child: wp.array(dtype=wp.int32),
    body_adjacent_joints_offsets: wp.array(dtype=wp.int32),
    body_adjacent_joints_fill_count: wp.array(dtype=wp.int32),
    body_adjacent_joints: wp.array(dtype=wp.int32),
):
    joint_count = joint_parent.shape[0]
    for joint_id in range(joint_count):
        parent_id = joint_parent[joint_id]
        child_id = joint_child[joint_id]

        # Add joint to parent body's adjacency list
        if parent_id >= 0:
            fill_count_parent = body_adjacent_joints_fill_count[parent_id]
            buffer_offset_parent = body_adjacent_joints_offsets[parent_id]
            body_adjacent_joints[buffer_offset_parent + fill_count_parent] = joint_id
            body_adjacent_joints_fill_count[parent_id] = fill_count_parent + 1

        # Add joint to child body's adjacency list
        if child_id >= 0:
            fill_count_child = body_adjacent_joints_fill_count[child_id]
            buffer_offset_child = body_adjacent_joints_offsets[child_id]
            body_adjacent_joints[buffer_offset_child + fill_count_child] = joint_id
            body_adjacent_joints_fill_count[child_id] = fill_count_child + 1


# -----------------------------
# Pre-iteration kernels (once per step)
# -----------------------------
@wp.kernel
def forward_step_rigid_bodies(
    # Inputs
    dt: float,
    update_step_history: bool,
    gravity: wp.array(dtype=wp.vec3),
    body_f: wp.array(dtype=wp.spatial_vector),
    body_com: wp.array(dtype=wp.vec3),
    body_inertia: wp.array(dtype=wp.mat33),
    body_inv_mass: wp.array(dtype=float),
    body_inv_inertia: wp.array(dtype=wp.mat33),
    body_q: wp.array(dtype=wp.transform),
    body_qd: wp.array(dtype=wp.spatial_vector),
    body_q_prev: wp.array(dtype=wp.transform),
    body_q_anchor: wp.array(dtype=wp.transform),
    body_inertia_q: wp.array(dtype=wp.transform),
):
    """
    Forward integration step for rigid bodies in the AVBD/VBD solver.

    Args:
        dt: Time step [s].
        update_step_history: If True, updates ``body_q_anchor`` with the current pose
            at the start of the macro-step (damping/friction anchor).
        gravity: Gravity vector array (world frame).
        body_f: External forces on bodies (spatial wrenches, world frame).
        body_com: Centers of mass (local body frame).
        body_inertia: Inertia tensors (local body frame).
        body_inv_mass: Inverse masses (0 for kinematic bodies).
        body_inv_inertia: Inverse inertia tensors (local body frame).
        body_q: Body transforms (input: start-of-step pose, output: integrated pose).
        body_qd: Body velocities (input: start-of-step velocity, output: integrated velocity).
        body_q_prev: Previous body transforms (output, snapshot of ``body_q`` before integration,
            used for BDF1 velocity update).
        body_q_anchor: Anchor body transforms (output, updated only when
            ``update_step_history`` is True, used as a long-range damping/friction anchor).
        body_inertia_q: Inertial target body transforms for the AVBD solve (output).
    """
    tid = wp.tid()

    # Read current transform
    q_current = body_q[tid]

    # Store previous transform for velocity computation
    body_q_prev[tid] = q_current

    # Update history buffer if this is the start of a frame
    if update_step_history:
        body_q_anchor[tid] = q_current

    # Early exit for kinematic bodies (inv_mass == 0)
    inv_m = body_inv_mass[tid]
    if inv_m == 0.0:
        body_inertia_q[tid] = q_current
        return

    # Read body state (only for dynamic bodies)
    qd_current = body_qd[tid]
    f_current = body_f[tid]
    com_local = body_com[tid]
    I_local = body_inertia[tid]
    inv_I = body_inv_inertia[tid]

    # Integrate rigid body motion (semi-implicit Euler, no angular damping)
    q_new, qd_new = integrate_rigid_body(
        q_current,
        qd_current,
        f_current,
        com_local,
        I_local,
        inv_m,
        inv_I,
        gravity,
        0.0,  # angular_damping = 0 (consistent with particle VBD)
        dt,
    )

    # Update current transform, velocity, and set inertial target
    body_q[tid] = q_new
    body_qd[tid] = qd_new
    body_inertia_q[tid] = q_new


@wp.kernel
def build_body_body_contact_lists(
    rigid_contact_count: wp.array(dtype=int),
    rigid_contact_shape0: wp.array(dtype=int),
    rigid_contact_shape1: wp.array(dtype=int),
    shape_body: wp.array(dtype=wp.int32),
    body_contact_buffer_pre_alloc: int,
    body_contact_counts: wp.array(dtype=wp.int32),
    body_contact_indices: wp.array(dtype=wp.int32),
):
    """
    Build per-body contact lists for body-centric per-color contact evaluation.

    Each contact index t_id is appended to both bodies' lists (shape0's body and
    shape1's body), enabling efficient lookup of all contacts involving a given body
    during per-body solve iterations.

    Notes:
      - body_contact_counts[b] is reset to 0 on the host before launch and
        atomically incremented here; consumers must only read the first
        body_contact_counts[b] indices for each body.
      - If a body has more than body_contact_buffer_pre_alloc contacts, extra indices
        are dropped (overflow is safely ignored here).
      - body_contact_indices is not cleared each step; only the prefix defined
        by body_contact_counts is considered valid.
    """
    t_id = wp.tid()
    if t_id >= rigid_contact_count[0]:
        return

    s0 = rigid_contact_shape0[t_id]
    s1 = rigid_contact_shape1[t_id]
    b0 = shape_body[s0] if s0 >= 0 else -1
    b1 = shape_body[s1] if s1 >= 0 else -1

    # Add contact to body0's list
    if b0 >= 0:
        idx = wp.atomic_add(body_contact_counts, b0, 1)
        if idx < body_contact_buffer_pre_alloc:
            body_contact_indices[b0 * body_contact_buffer_pre_alloc + idx] = t_id

    # Add contact to body1's list
    if b1 >= 0:
        idx = wp.atomic_add(body_contact_counts, b1, 1)
        if idx < body_contact_buffer_pre_alloc:
            body_contact_indices[b1 * body_contact_buffer_pre_alloc + idx] = t_id


@wp.kernel
def build_body_particle_contact_lists(
    body_particle_contact_count: wp.array(dtype=int),
    body_particle_contact_shape: wp.array(dtype=int),
    shape_body: wp.array(dtype=wp.int32),
    body_particle_contact_buffer_pre_alloc: int,
    body_particle_contact_counts: wp.array(dtype=wp.int32),
    body_particle_contact_indices: wp.array(dtype=wp.int32),
):
    """
    Build per-body contact lists for body-particle (particle-rigid) contacts.

    Each body-particle contact index tid (from the external contacts.soft_contact_*
    arrays) is appended to the list of the rigid body associated with
    body_particle_contact_shape[tid] via shape_body. This enables efficient
    per-body processing of particle-rigid contacts during the rigid-body solve
    (e.g., Gauss-Seidel color sweeps).

    Notes:
      - body_particle_contact_counts[b] must be zeroed on the host before launch.
      - Only the first body_particle_contact_counts[b] entries in the flat
        body_particle_contact_indices array are considered valid for body b.
      - If a body has more than body_particle_contact_buffer_pre_alloc body-particle
        contacts, extra indices are dropped (overflow is ignored safely).
    """
    tid = wp.tid()
    if tid >= body_particle_contact_count[0]:
        return

    shape = body_particle_contact_shape[tid]
    body = shape_body[shape] if shape >= 0 else -1

    if body < 0:
        return

    idx = wp.atomic_add(body_particle_contact_counts, body, 1)
    if idx < body_particle_contact_buffer_pre_alloc:
        body_particle_contact_indices[body * body_particle_contact_buffer_pre_alloc + idx] = tid


@wp.kernel
def warmstart_joints(
    # Inputs
    joint_target_ke: wp.array(dtype=float),
    joint_penalty_k_min: wp.array(dtype=float),
    gamma: float,
    # Input/output
    joint_penalty_k: wp.array(dtype=float),
):
    """
    Warm-start per-DOF penalty stiffness for joint constraints (runs once per step).

    Algorithm (per DOF):
      - Decay previous penalty: k <- gamma * k
      - Clamp to [joint_penalty_k_min[i], joint_target_ke[i]]
    """
    i = wp.tid()

    k_prev = joint_penalty_k[i]
    k_new = wp.clamp(gamma * k_prev, joint_penalty_k_min[i], joint_target_ke[i])
    joint_penalty_k[i] = k_new


@wp.kernel
def warmstart_body_body_contacts(
    rigid_contact_count: wp.array(dtype=int),
    rigid_contact_shape0: wp.array(dtype=int),
    rigid_contact_shape1: wp.array(dtype=int),
    shape_material_ke: wp.array(dtype=float),
    shape_material_kd: wp.array(dtype=float),
    shape_material_mu: wp.array(dtype=float),
    k_start_body_contact: float,
    # Outputs
    contact_penalty_k: wp.array(dtype=float),
    contact_material_ke: wp.array(dtype=float),
    contact_material_kd: wp.array(dtype=float),
    contact_material_mu: wp.array(dtype=float),
):
    """
    Warm-start contact penalties and cache material properties (once per macro-step).

    Computes averaged material properties for each rigid contact and resets the
    AVBD penalty to a bounded initial value based on `k_start_body_contact` and the
    material stiffness.
    """
    i = wp.tid()
    if i >= rigid_contact_count[0]:
        return

    # Read shape indices
    shape_id_0 = rigid_contact_shape0[i]
    shape_id_1 = rigid_contact_shape1[i]

    # Cache averaged material properties (arithmetic mean for stiffness/damping, geometric for friction)
    avg_ke = 0.5 * (shape_material_ke[shape_id_0] + shape_material_ke[shape_id_1])
    avg_kd = 0.5 * (shape_material_kd[shape_id_0] + shape_material_kd[shape_id_1])
    avg_mu = wp.sqrt(shape_material_mu[shape_id_0] * shape_material_mu[shape_id_1])

    contact_material_ke[i] = avg_ke
    contact_material_kd[i] = avg_kd
    contact_material_mu[i] = avg_mu

    # Reset contact penalty to k_start every frame because contact indices are not persistent across frames.
    k_new = wp.min(k_start_body_contact, avg_ke)
    contact_penalty_k[i] = k_new


@wp.kernel
def warmstart_body_particle_contacts(
    body_particle_contact_count: wp.array(dtype=int),
    body_particle_contact_shape: wp.array(dtype=int),
    soft_contact_ke: float,
    soft_contact_kd: float,
    soft_contact_mu: float,
    shape_material_ke: wp.array(dtype=float),
    shape_material_kd: wp.array(dtype=float),
    shape_material_mu: wp.array(dtype=float),
    k_start_body_contact: float,
    # Outputs
    body_particle_contact_penalty_k: wp.array(dtype=float),
    body_particle_contact_material_ke: wp.array(dtype=float),
    body_particle_contact_material_kd: wp.array(dtype=float),
    body_particle_contact_material_mu: wp.array(dtype=float),
):
    """
    Warm-start body-particle (particle-rigid) contact penalties and cache material
    properties (once per macro-step).

    The scalar inputs `soft_contact_ke/kd/mu` are the particle-side soft-contact
    material parameters (from `model.soft_contact_*`). For each body-particle
    contact, this kernel averages those particle-side values with the rigid
    shape's material parameters and resets the AVBD penalty to a bounded
    initial value based on `k_start_body_contact` and the averaged stiffness.
    """
    i = wp.tid()
    if i >= body_particle_contact_count[0]:
        return

    # Read shape index for the rigid body side
    shape_idx = body_particle_contact_shape[i]

    # Cache averaged material properties (arithmetic mean for stiffness/damping, geometric for friction)
    avg_ke = 0.5 * (soft_contact_ke + shape_material_ke[shape_idx])
    avg_kd = 0.5 * (soft_contact_kd + shape_material_kd[shape_idx])
    avg_mu = wp.sqrt(soft_contact_mu * shape_material_mu[shape_idx])

    body_particle_contact_material_ke[i] = avg_ke
    body_particle_contact_material_kd[i] = avg_kd
    body_particle_contact_material_mu[i] = avg_mu

    # Reset contact penalty to k_start every frame because contact indices are not persistent across frames.
    k_new = wp.min(k_start_body_contact, avg_ke)
    body_particle_contact_penalty_k[i] = k_new


@wp.kernel
def compute_cable_dahl_parameters(
    # Inputs
    joint_type: wp.array(dtype=int),
    joint_parent: wp.array(dtype=int),
    joint_child: wp.array(dtype=int),
    joint_X_p: wp.array(dtype=wp.transform),
    joint_X_c: wp.array(dtype=wp.transform),
    joint_qd_start: wp.array(dtype=int),
    joint_target_ke: wp.array(dtype=float),
    body_q: wp.array(dtype=wp.transform),
    body_q_rest: wp.array(dtype=wp.transform),
    body_com: wp.array(dtype=wp.vec3),
    joint_sigma_prev: wp.array(dtype=wp.vec3),
    joint_kappa_prev: wp.array(dtype=wp.vec3),
    joint_dkappa_prev: wp.array(dtype=wp.vec3),
    joint_eps_max: wp.array(dtype=float),
    joint_tau: wp.array(dtype=float),
    # Outputs
    joint_sigma_start: wp.array(dtype=wp.vec3),
    joint_C_fric: wp.array(dtype=wp.vec3),
):
    """
    Compute Dahl hysteresis parameters (sigma0, C_fric) for cable bending,
    given the current curvature state and the stored previous Dahl state.

    The outputs are:
      - sigma0: linearized friction stress at the start of the step (per component)
      - C_fric: tangent stiffness d(sigma)/d(kappa) (per component)
    """
    j = wp.tid()

    # Only process cable joints
    if joint_type[j] != JointType.CABLE:
        joint_sigma_start[j] = wp.vec3(0.0)
        joint_C_fric[j] = wp.vec3(0.0)
        return

    parent = joint_parent[j]
    child = joint_child[j]

    # Skip invalid joints
    if parent < 0 or child < 0:
        joint_sigma_start[j] = wp.vec3(0.0)
        joint_C_fric[j] = wp.vec3(0.0)
        return

    # Compute joint frames in world space (current and rest only)
    X_wp = body_q[parent] * joint_X_p[j]
    X_wc = body_q[child] * joint_X_c[j]
    X_wp_rest = body_q_rest[parent] * joint_X_p[j]
    X_wc_rest = body_q_rest[child] * joint_X_c[j]

    # Extract quaternions (current and rest configurations)
    q_wp = wp.transform_get_rotation(X_wp)
    q_wc = wp.transform_get_rotation(X_wc)
    q_wp_rest = wp.transform_get_rotation(X_wp_rest)
    q_wc_rest = wp.transform_get_rotation(X_wc_rest)

    # Compute current curvature vector at beginning-of-step (predicted state)
    kappa_now = cable_get_kappa(q_wp, q_wc, q_wp_rest, q_wc_rest)

    # Read previous state (from last converged timestep)
    kappa_prev = joint_kappa_prev[j]
    d_kappa_prev = joint_dkappa_prev[j]
    sigma_prev = joint_sigma_prev[j]

    # Read per-joint Dahl parameters (isotropic)
    eps_max = joint_eps_max[j]
    tau = joint_tau[j]

    # Get bend stiffness and segment length for this joint
    dof_start = joint_qd_start[j]
    bend_dof_index = dof_start + 1
    k_bend_material = joint_target_ke[bend_dof_index]

    # Compute segment length (Cosserat rod discretization)
    parent_com = body_com[parent]
    rest_length = 2.0 * parent_com[2]

    # Skip degenerate segments
    if rest_length < _SMALL_LENGTH_EPS:
        joint_sigma_start[j] = wp.vec3(0.0)
        joint_C_fric[j] = wp.vec3(0.0)
        return

    # Effective stiffness and friction envelope
    k_bend = k_bend_material / rest_length
    sigma_max = k_bend * eps_max
    if sigma_max <= 0.0 or tau <= 0.0:
        joint_sigma_start[j] = wp.vec3(0.0)
        joint_C_fric[j] = wp.vec3(0.0)
        return

    sigma_out = wp.vec3(0.0)
    C_fric_out = wp.vec3(0.0)

    for axis in range(3):
        kappa_i = kappa_now[axis]
        kappa_i_prev = kappa_prev[axis]
        sigma_i_prev = sigma_prev[axis]

        # Geometric curvature change
        d_kappa_i = kappa_i - kappa_i_prev

        # Direction flag based primarily on geometric change, with stored Delta-kappa fallback
        s_i = 1.0
        if d_kappa_i > _DAHL_KAPPADOT_DEADBAND:
            s_i = 1.0
        elif d_kappa_i < -_DAHL_KAPPADOT_DEADBAND:
            s_i = -1.0
        else:
            # Within deadband: maintain previous direction from stored Delta kappa
            s_i = 1.0 if d_kappa_prev[axis] >= 0.0 else -1.0
        exp_term = wp.exp(-s_i * d_kappa_i / tau)
        sigma0_i = s_i * sigma_max * (1.0 - exp_term) + sigma_i_prev * exp_term
        sigma0_i = wp.clamp(sigma0_i, -sigma_max, sigma_max)

        numerator = sigma_max - s_i * sigma0_i
        # Use geometric curvature change for the length scale
        denominator = tau + wp.abs(d_kappa_i)

        # Store pure stiffness K = numerator / (tau + |d_kappa|)
        C_fric_i = wp.max(numerator / denominator, 0.0)
        sigma_out[axis] = sigma0_i
        C_fric_out[axis] = C_fric_i

    joint_sigma_start[j] = sigma_out
    joint_C_fric[j] = C_fric_out


# -----------------------------
# Iteration kernels (per color per iteration)
# -----------------------------
@wp.kernel
def accumulate_body_body_contacts_per_body(
    dt_damping: float,
    color_group: wp.array(dtype=wp.int32),
    body_q_anchor: wp.array(dtype=wp.transform),
    body_q: wp.array(dtype=wp.transform),
    body_com: wp.array(dtype=wp.vec3),
    body_inv_mass: wp.array(dtype=float),
    friction_epsilon: float,
    contact_penalty_k: wp.array(dtype=float),
    contact_material_kd: wp.array(dtype=float),
    contact_material_mu: wp.array(dtype=float),
    rigid_contact_count: wp.array(dtype=int),
    rigid_contact_shape0: wp.array(dtype=int),
    rigid_contact_shape1: wp.array(dtype=int),
    rigid_contact_point0: wp.array(dtype=wp.vec3),
    rigid_contact_point1: wp.array(dtype=wp.vec3),
    rigid_contact_normal: wp.array(dtype=wp.vec3),
    rigid_contact_thickness0: wp.array(dtype=float),
    rigid_contact_thickness1: wp.array(dtype=float),
    shape_body: wp.array(dtype=wp.int32),
    body_contact_buffer_pre_alloc: int,
    body_contact_counts: wp.array(dtype=wp.int32),
    body_contact_indices: wp.array(dtype=wp.int32),
    body_forces: wp.array(dtype=wp.vec3),
    body_torques: wp.array(dtype=wp.vec3),
    body_hessian_ll: wp.array(dtype=wp.mat33),
    body_hessian_al: wp.array(dtype=wp.mat33),
    body_hessian_aa: wp.array(dtype=wp.mat33),
):
    """
    Per-body contact accumulation with _NUM_CONTACT_THREADS_PER_BODY strided threads.
    """
    tid = wp.tid()
    body_idx_in_group = tid // _NUM_CONTACT_THREADS_PER_BODY
    thread_id_within_body = tid % _NUM_CONTACT_THREADS_PER_BODY

    if body_idx_in_group >= color_group.shape[0]:
        return

    body_id = color_group[body_idx_in_group]
    if body_inv_mass[body_id] <= 0.0:
        return

    num_contacts = body_contact_counts[body_id]
    if num_contacts > body_contact_buffer_pre_alloc:
        num_contacts = body_contact_buffer_pre_alloc

    force_acc = wp.vec3(0.0)
    torque_acc = wp.vec3(0.0)
    h_ll_acc = wp.mat33(0.0)
    h_al_acc = wp.mat33(0.0)
    h_aa_acc = wp.mat33(0.0)

    i = thread_id_within_body
    while i < num_contacts:
        contact_idx = body_contact_indices[body_id * body_contact_buffer_pre_alloc + i]
        if contact_idx >= rigid_contact_count[0]:
            i += _NUM_CONTACT_THREADS_PER_BODY
            continue

        s0 = rigid_contact_shape0[contact_idx]
        s1 = rigid_contact_shape1[contact_idx]
        b0 = shape_body[s0] if s0 >= 0 else -1
        b1 = shape_body[s1] if s1 >= 0 else -1

        if b0 != body_id and b1 != body_id:
            i += _NUM_CONTACT_THREADS_PER_BODY
            continue

        cp0_local = rigid_contact_point0[contact_idx]
        cp1_local = rigid_contact_point1[contact_idx]
        contact_normal = -rigid_contact_normal[contact_idx]
        cp0_world = wp.transform_point(body_q[b0], cp0_local) if b0 >= 0 else cp0_local
        cp1_world = wp.transform_point(body_q[b1], cp1_local) if b1 >= 0 else cp1_local
        thickness = rigid_contact_thickness0[contact_idx] + rigid_contact_thickness1[contact_idx]
        dist = wp.dot(contact_normal, cp1_world - cp0_world)
        penetration = thickness - dist

        if penetration <= _SMALL_LENGTH_EPS:
            i += _NUM_CONTACT_THREADS_PER_BODY
            continue

        contact_ke = contact_penalty_k[contact_idx]
        contact_kd = contact_material_kd[contact_idx]
        contact_mu = contact_material_mu[contact_idx]
        (
            force_0,
            torque_0,
            h_ll_0,
            h_al_0,
            h_aa_0,
            force_1,
            torque_1,
            h_ll_1,
            h_al_1,
            h_aa_1,
        ) = evaluate_rigid_contact_from_collision(
            b0,
            b1,
            body_q,
            body_q_anchor,
            body_com,
            cp0_local,
            cp1_local,
            contact_normal,
            penetration,
            contact_ke,
            contact_kd,
            contact_mu,
            friction_epsilon,
            dt_damping,
        )

        if body_id == b0:
            force_acc += force_0
            torque_acc += torque_0
            h_ll_acc += h_ll_0
            h_al_acc += h_al_0
            h_aa_acc += h_aa_0
        else:
            force_acc += force_1
            torque_acc += torque_1
            h_ll_acc += h_ll_1
            h_al_acc += h_al_1
            h_aa_acc += h_aa_1

        i += _NUM_CONTACT_THREADS_PER_BODY

    wp.atomic_add(body_forces, body_id, force_acc)
    wp.atomic_add(body_torques, body_id, torque_acc)
    wp.atomic_add(body_hessian_ll, body_id, h_ll_acc)
    wp.atomic_add(body_hessian_al, body_id, h_al_acc)
    wp.atomic_add(body_hessian_aa, body_id, h_aa_acc)


@wp.kernel
def accumulate_body_particle_contacts_per_body(
    dt_damping: float,
    color_group: wp.array(dtype=wp.int32),
    # Particle state
    particle_q: wp.array(dtype=wp.vec3),
    particle_q_anchor: wp.array(dtype=wp.vec3),
    particle_radius: wp.array(dtype=float),
    # Rigid body state
    body_q_anchor: wp.array(dtype=wp.transform),
    body_q: wp.array(dtype=wp.transform),
    body_qd: wp.array(dtype=wp.spatial_vector),
    body_com: wp.array(dtype=wp.vec3),
    body_inv_mass: wp.array(dtype=float),
    # AVBD body-particle soft contact penalties and material properties
    friction_epsilon: float,
    body_particle_contact_penalty_k: wp.array(dtype=float),
    body_particle_contact_material_kd: wp.array(dtype=float),
    body_particle_contact_material_mu: wp.array(dtype=float),
    # Soft contact data (body-particle)
    body_particle_contact_count: wp.array(dtype=int),
    body_particle_contact_particle: wp.array(dtype=int),
    body_particle_contact_shape: wp.array(dtype=int),
    body_particle_contact_body_pos: wp.array(dtype=wp.vec3),
    body_particle_contact_body_vel: wp.array(dtype=wp.vec3),
    body_particle_contact_normal: wp.array(dtype=wp.vec3),
    # Shape/material data
    shape_material_mu: wp.array(dtype=float),
    shape_body: wp.array(dtype=wp.int32),
    # Per-body soft-contact adjacency (body-particle)
    body_particle_contact_buffer_pre_alloc: int,
    body_particle_contact_counts: wp.array(dtype=wp.int32),
    body_particle_contact_indices: wp.array(dtype=wp.int32),
    # Outputs
    body_forces: wp.array(dtype=wp.vec3),
    body_torques: wp.array(dtype=wp.vec3),
    body_hessian_ll: wp.array(dtype=wp.mat33),
    body_hessian_al: wp.array(dtype=wp.mat33),
    body_hessian_aa: wp.array(dtype=wp.mat33),
):
    """
    Per-body accumulation of body-particle (particle-rigid) soft contact forces and
    Hessians on rigid bodies.

    This kernel mirrors the Gauss-Seidel per-body pattern used for body-body contacts,
    but iterates over body-particle contacts associated with each body via the
    precomputed body_particle_contact_* adjacency.

    For each body-particle contact, we:
      1. Reuse the particle-side contact model via evaluate_body_particle_contact to
         compute the force and Hessian on the particle using adaptive AVBD penalties.
      2. Apply the equal-and-opposite reaction force, torque, and Hessian contributions
         to the rigid body.

    Notes:
      - Only dynamic bodies (inv_mass > 0) are updated.
      - Hessian contributions are accumulated into body_hessian_ll/al/aa.
      - Uses AVBD adaptive penalty parameters (warmstarted and updated per iteration).
    """
    tid = wp.tid()
    body_idx_in_group = tid // _NUM_CONTACT_THREADS_PER_BODY
    thread_id_within_body = tid % _NUM_CONTACT_THREADS_PER_BODY

    if body_idx_in_group >= color_group.shape[0]:
        return

    body_id = color_group[body_idx_in_group]
    if body_inv_mass[body_id] <= 0.0:
        return

    num_contacts = body_particle_contact_counts[body_id]
    if num_contacts > body_particle_contact_buffer_pre_alloc:
        num_contacts = body_particle_contact_buffer_pre_alloc

    i = thread_id_within_body
    max_contacts = body_particle_contact_count[0]

    while i < num_contacts:
        contact_idx = body_particle_contact_indices[body_id * body_particle_contact_buffer_pre_alloc + i]
        if contact_idx >= max_contacts:
            i += _NUM_CONTACT_THREADS_PER_BODY
            continue

        particle_idx = body_particle_contact_particle[contact_idx]
        if particle_idx < 0:
            i += _NUM_CONTACT_THREADS_PER_BODY
            continue

        # Early penetration check to avoid unnecessary function call
        particle_pos = particle_q[particle_idx]
        X_wb = body_q[body_id]
        cp_local = body_particle_contact_body_pos[contact_idx]
        cp_world = wp.transform_point(X_wb, cp_local)
        n = body_particle_contact_normal[contact_idx]
        radius = particle_radius[particle_idx]
        penetration_depth = -(wp.dot(n, particle_pos - cp_world) - radius)

        if penetration_depth <= 0.0:
            i += _NUM_CONTACT_THREADS_PER_BODY
            continue

        # Compute contact force and Hessian on particle using AVBD adaptive penalty
        particle_prev_pos = particle_q_anchor[particle_idx]

        # Read per-contact AVBD penalty and material properties
        contact_ke = body_particle_contact_penalty_k[contact_idx]
        contact_kd = body_particle_contact_material_kd[contact_idx]
        contact_mu = body_particle_contact_material_mu[contact_idx]

        force_on_particle, hessian_particle = evaluate_body_particle_contact(
            particle_idx,
            particle_pos,
            particle_prev_pos,
            contact_idx,
            contact_ke,
            contact_kd,
            contact_mu,
            friction_epsilon,
            particle_radius,
            shape_material_mu,
            shape_body,
            body_q,
            body_q_anchor,
            body_qd,
            body_com,
            body_particle_contact_shape,
            body_particle_contact_body_pos,
            body_particle_contact_body_vel,
            body_particle_contact_normal,
            dt_damping,
        )

        # Reaction on the body (Newton's 3rd law)
        f_body = -force_on_particle

        # Compute torque and Hessian contributions: tau = r x f, where r is from COM to contact point
        com_world = wp.transform_point(X_wb, body_com[body_id])
        r = cp_world - com_world
        tau_body = wp.cross(r, f_body)

        # Hessian contributions for the rigid body
        # The particle Hessian K_particle contributes to the body's linear-linear block
        # and we need to add angular-linear and angular-angular blocks from the moment arm
        K_total = hessian_particle
        r_skew = wp.skew(r)
        r_skew_T_K = wp.transpose(r_skew) * K_total

        h_ll_body = K_total  # Linear-linear block
        h_al_body = -r_skew_T_K  # Angular-linear block
        h_aa_body = r_skew_T_K * r_skew  # Angular-angular block

        wp.atomic_add(body_forces, body_id, f_body)
        wp.atomic_add(body_torques, body_id, tau_body)
        wp.atomic_add(body_hessian_ll, body_id, h_ll_body)
        wp.atomic_add(body_hessian_al, body_id, h_al_body)
        wp.atomic_add(body_hessian_aa, body_id, h_aa_body)

        i += _NUM_CONTACT_THREADS_PER_BODY


@wp.kernel
def solve_rigid_body(
    dt: float,
    dt_damping: float,
    body_ids_in_color: wp.array(dtype=wp.int32),
    body_q: wp.array(dtype=wp.transform),
    body_q_anchor: wp.array(dtype=wp.transform),
    body_q_rest: wp.array(dtype=wp.transform),
    body_mass: wp.array(dtype=float),
    body_inv_mass: wp.array(dtype=float),
    body_inertia: wp.array(dtype=wp.mat33),
    body_inertia_q: wp.array(dtype=wp.transform),
    body_com: wp.array(dtype=wp.vec3),
    adjacency: RigidForceElementAdjacencyInfo,
    # Joint data
    joint_type: wp.array(dtype=int),
    joint_parent: wp.array(dtype=int),
    joint_child: wp.array(dtype=int),
    joint_X_p: wp.array(dtype=wp.transform),
    joint_X_c: wp.array(dtype=wp.transform),
    joint_qd_start: wp.array(dtype=int),
    joint_target_kd: wp.array(dtype=float),
    # AVBD per-DOF penalty state
    joint_penalty_k: wp.array(dtype=float),
    # Dahl hysteresis parameters (frozen for this timestep, component-wise vec3 per joint)
    joint_sigma_start: wp.array(dtype=wp.vec3),
    joint_C_fric: wp.array(dtype=wp.vec3),
    external_forces: wp.array(dtype=wp.vec3),
    external_torques: wp.array(dtype=wp.vec3),
    external_hessian_ll: wp.array(dtype=wp.mat33),  # Linear-linear block from rigid contacts
    external_hessian_al: wp.array(dtype=wp.mat33),  # Angular-linear coupling block from rigid contacts
    external_hessian_aa: wp.array(dtype=wp.mat33),  # Angular-angular block from rigid contacts
    # Output
    body_q_new: wp.array(dtype=wp.transform),
):
    """
    AVBD solve step for rigid bodies using block Cholesky decomposition.

    Solves the 6-DOF rigid body system by assembling inertial, joint, and collision
    contributions into a 6x6 block system [M, C; C^T, A] and solving via Schur complement.
    Consistent with VBD particle solve pattern: inertia + external + constraint forces.

    Algorithm:
      1. Compute inertial forces/Hessians
      2. Accumulate external forces/Hessians from rigid contacts
      3. Accumulate joint forces/Hessians from adjacent joints
      4. Solve 6x6 block system via Schur complement: S = A - C*M^-1*C^T
      5. Update pose: rotation from angular increment, position from linear increment

    Args:
        dt: Time step.
        dt_damping: Long-range damping/history time window used for joint damping.
        body_ids_in_color: Body indices in current color group (for parallel coloring).
        body_q_anchor: Previous body transforms (damping anchor).
        body_q_rest: Rest transforms (for joint targets).
        body_mass: Body masses.
        body_inv_mass: Inverse masses (0 for kinematic bodies).
        body_inertia: Inertia tensors (local body frame).
        body_inertia_q: Inertial target transforms (from forward integration).
        body_com: Center of mass offsets (local body frame).
        adjacency: Body-joint adjacency (CSR format).
        joint_*: Joint configuration arrays.
        joint_penalty_k: AVBD per-DOF penalty stiffness.
        joint_sigma_start: Dahl hysteresis state at start of step.
        joint_C_fric: Dahl friction configuration per joint.
        external_forces: External linear forces from rigid contacts.
        external_torques: External angular torques from rigid contacts.
        external_hessian_ll: Linear-linear Hessian block (3x3) from rigid contacts.
        external_hessian_al: Angular-linear coupling Hessian block (3x3) from rigid contacts.
        external_hessian_aa: Angular-angular Hessian block (3x3) from rigid contacts.
        body_q: Current body transforms (input/output, updated in-place).

    Note:
      - All forces, torques, and Hessian blocks are expressed in the world frame.
    """
    tid = wp.tid()
    body_index = body_ids_in_color[tid]

    q_current = body_q[body_index]

    # Early exit for kinematic bodies
    if body_inv_mass[body_index] == 0.0:
        body_q_new[body_index] = q_current
        return

    # Inertial force and Hessian
    dt_sqr_reciprocal = 1.0 / (dt * dt)

    # Read body properties
    q_inertial = body_inertia_q[body_index]
    body_com_local = body_com[body_index]
    m = body_mass[body_index]
    I_body = body_inertia[body_index]

    # Extract poses
    pos_current = wp.transform_get_translation(q_current)
    rot_current = wp.transform_get_rotation(q_current)
    pos_star = wp.transform_get_translation(q_inertial)
    rot_star = wp.transform_get_rotation(q_inertial)

    # Compute COM positions
    com_current = pos_current + wp.quat_rotate(rot_current, body_com_local)
    com_star = pos_star + wp.quat_rotate(rot_star, body_com_local)

    # Linear inertial force and Hessian
    inertial_coeff = m * dt_sqr_reciprocal
    f_lin = (com_star - com_current) * inertial_coeff

    # Compute relative rotation via quaternion difference
    # dq = q_current^-1 * q_star
    q_delta = wp.mul(wp.quat_inverse(rot_current), rot_star)

    # Enforce shortest path (w > 0) to avoid double-cover ambiguity
    if q_delta[3] < 0.0:
        q_delta = wp.quat(-q_delta[0], -q_delta[1], -q_delta[2], -q_delta[3])

    # Rotation vector
    axis_body, angle_body = wp.quat_to_axis_angle(q_delta)
    theta_body = axis_body * angle_body

    # Angular inertial torque
    tau_body = I_body * (theta_body * dt_sqr_reciprocal)
    tau_world = wp.quat_rotate(rot_current, tau_body)

    # Angular Hessian in world frame: use full inertia (supports off-diagonal products of inertia)
    R_cur = wp.quat_to_matrix(rot_current)
    I_world = R_cur * I_body * wp.transpose(R_cur)
    angular_hessian = dt_sqr_reciprocal * I_world

    # Accumulate external forces (rigid contacts)
    # Read external contributions
    ext_torque = external_torques[body_index]
    ext_force = external_forces[body_index]
    ext_h_aa = external_hessian_aa[body_index]
    ext_h_al = external_hessian_al[body_index]
    ext_h_ll = external_hessian_ll[body_index]

    f_torque = tau_world + ext_torque
    f_force = f_lin + ext_force

    h_aa = angular_hessian + ext_h_aa
    h_al = ext_h_al
    h_ll = wp.mat33(
        ext_h_ll[0, 0] + inertial_coeff,
        ext_h_ll[0, 1],
        ext_h_ll[0, 2],
        ext_h_ll[1, 0],
        ext_h_ll[1, 1] + inertial_coeff,
        ext_h_ll[1, 2],
        ext_h_ll[2, 0],
        ext_h_ll[2, 1],
        ext_h_ll[2, 2] + inertial_coeff,
    )

    # Accumulate joint forces (constraints)
    num_adj_joints = get_body_num_adjacent_joints(adjacency, body_index)
    for joint_counter in range(num_adj_joints):
        joint_idx = get_body_adjacent_joint_id(adjacency, body_index, joint_counter)

        joint_force, joint_torque, joint_H_ll, joint_H_al, joint_H_aa = evaluate_joint_force_hessian(
            body_index,
            joint_idx,
            body_q,
            body_q_anchor,
            body_q_rest,
            body_com,
            joint_type,
            joint_parent,
            joint_child,
            joint_X_p,
            joint_X_c,
            joint_qd_start,
            joint_target_kd,
            joint_penalty_k,
            joint_sigma_start,
            joint_C_fric,
            dt_damping,
        )

        f_force = f_force + joint_force
        f_torque = f_torque + joint_torque

        h_ll = h_ll + joint_H_ll
        h_al = h_al + joint_H_al
        h_aa = h_aa + joint_H_aa

    # Solve 6x6 block system via Schur complement
    # Regularize angular Hessian (in-place)
    trA = wp.trace(h_aa) / 3.0
    epsA = 1.0e-9 * (trA + 1.0)
    h_aa[0, 0] = h_aa[0, 0] + epsA
    h_aa[1, 1] = h_aa[1, 1] + epsA
    h_aa[2, 2] = h_aa[2, 2] + epsA

    # Factorize linear Hessian
    Lm_p = chol33(h_ll)

    # Compute M^-1 * f_force
    MinvF = chol33_solve(Lm_p, f_force)

    # Compute M^-1 * C^T
    C_r0 = wp.vec3(h_al[0, 0], h_al[0, 1], h_al[0, 2])
    C_r1 = wp.vec3(h_al[1, 0], h_al[1, 1], h_al[1, 2])
    C_r2 = wp.vec3(h_al[2, 0], h_al[2, 1], h_al[2, 2])

    X0 = chol33_solve(Lm_p, C_r0)
    X1 = chol33_solve(Lm_p, C_r1)
    X2 = chol33_solve(Lm_p, C_r2)

    MinvCt = wp.mat33(X0[0], X1[0], X2[0], X0[1], X1[1], X2[1], X0[2], X1[2], X2[2])

    # Compute Schur complement
    S = h_aa - (h_al * MinvCt)

    # Factorize Schur complement
    Ls_p = chol33(S)

    # Solve for angular increment
    rhs_w = f_torque - (h_al * MinvF)
    w_world = chol33_solve(Ls_p, rhs_w)

    # Solve for linear increment
    Ct_w = wp.transpose(h_al) * w_world
    x_inc = chol33_solve(Lm_p, f_force - Ct_w)

    # Update pose from increments
    # Convert angular increment to quaternion
    if _USE_SMALL_ANGLE_APPROX:
        half_w = w_world * 0.5
        dq_world = wp.quat(half_w[0], half_w[1], half_w[2], 1.0)
        dq_world = wp.normalize(dq_world)
    else:
        ang_mag = wp.length(w_world)
        if ang_mag > _SMALL_ANGLE_EPS:
            dq_world = wp.quat_from_axis_angle(w_world / ang_mag, ang_mag)
        else:
            half_w = w_world * 0.5
            dq_world = wp.quat(half_w[0], half_w[1], half_w[2], 1.0)
            dq_world = wp.normalize(dq_world)

    # Apply rotation
    rot_new = wp.mul(dq_world, rot_current)
    rot_new = wp.normalize(rot_new)

    # Update position
    com_new = com_current + x_inc
    pos_new = com_new - wp.quat_rotate(rot_new, body_com_local)

    body_q_new[body_index] = wp.transform(pos_new, rot_new)


@wp.kernel
def copy_rigid_body_transforms_back(
    body_ids_in_color: wp.array(dtype=wp.int32),
    body_q_in: wp.array(dtype=wp.transform),
    body_q_out: wp.array(dtype=wp.transform),
):
    """
    Copy body transforms for the current color group from source to destination.

    Used after the per-color solve to write updated world transforms from a temporary
    buffer back to the main array.

    Args:
        body_ids_in_color: Body indices in the current color group
        body_q_in: Source body transforms (input)
        body_q_out: Destination body transforms (output)
    """
    tid = wp.tid()
    body_index = body_ids_in_color[tid]
    body_q_out[body_index] = body_q_in[body_index]


@wp.kernel
def update_duals_joint(
    joint_type: wp.array(dtype=int),
    joint_parent: wp.array(dtype=int),
    joint_child: wp.array(dtype=int),
    joint_X_p: wp.array(dtype=wp.transform),
    joint_X_c: wp.array(dtype=wp.transform),
    joint_qd_start: wp.array(dtype=int),
    joint_dof_dim: wp.array(dtype=int, ndim=2),
    joint_target_ke: wp.array(dtype=float),
    body_q: wp.array(dtype=wp.transform),
    body_q_rest: wp.array(dtype=wp.transform),
    body_com: wp.array(dtype=wp.vec3),
    beta: float,
    joint_penalty_k: wp.array(dtype=float),  # input/output
):
    """
    Update AVBD penalty parameters for joint constraints (per-iteration).

    Increases per-DOF penalties based on current constraint violation magnitudes,
    clamped by the per-DOF material target stiffness.

    Args:
        joint_type: Joint types
        joint_parent: Parent body indices
        joint_child: Child body indices
        joint_X_p: Parent joint frames (local)
        joint_X_c: Child joint frames (local)
        joint_qd_start: Start DOF index per joint
        joint_dof_dim: [num_lin, num_ang] DOF per joint
        joint_target_ke: Target stiffness per DOF
        body_q: Current body transforms (world)
        body_q_rest: Rest body transforms (world)
        body_com: Body COM offsets (local) used to infer segment length
        beta: Adaptation rate
        joint_penalty_k: In/out per-DOF adaptive penalties
    """
    j = wp.tid()
    parent = joint_parent[j]
    child = joint_child[j]

    # Early exit for invalid joints
    if parent < 0 or child < 0:
        return

    # Read DOF configuration
    dof_start = joint_qd_start[j]
    lin_axes = joint_dof_dim[j, 0]

    # Compute joint frames in world space
    X_wp = body_q[parent] * joint_X_p[j]
    X_wc = body_q[child] * joint_X_c[j]
    X_wp_rest = body_q_rest[parent] * joint_X_p[j]
    X_wc_rest = body_q_rest[child] * joint_X_c[j]

    # Cable joint: adaptive penalty for stretch and bend constraints
    if joint_type[j] == JointType.CABLE:
        # Read joint frame rotations
        q_wp = wp.transform_get_rotation(X_wp)
        q_wc = wp.transform_get_rotation(X_wc)
        q_wp_rest = wp.transform_get_rotation(X_wp_rest)
        q_wc_rest = wp.transform_get_rotation(X_wc_rest)

        # Compute segment rest length (capsule length) - shared by both modes
        parent_com_local = body_com[parent]
        rest_length = 2.0 * parent_com_local[2]

        # Skip degenerate segments
        if rest_length < _SMALL_LENGTH_EPS:
            return

        # Compute stretch constraint violation (simple pinning)
        # C = ||x_c - x_p|| (distance between attachment points)
        x_p = wp.transform_get_translation(X_wp)
        x_c = wp.transform_get_translation(X_wc)
        C_stretch = wp.length(x_c - x_p)

        # Compute bend constraint violation (rotation vector magnitude)
        kappa = cable_get_kappa(q_wp, q_wc, q_wp_rest, q_wc_rest)
        C_bend = wp.length(kappa)

        # Update stretch penalty (DOF 0)
        stretch_idx = dof_start
        stiffness_stretch = joint_target_ke[stretch_idx]
        k_stretch = joint_penalty_k[stretch_idx]
        k_stretch_new = wp.min(k_stretch + beta * C_stretch, stiffness_stretch)
        joint_penalty_k[stretch_idx] = k_stretch_new

        # Update bend penalty (DOF 1)
        bend_idx = dof_start + lin_axes
        stiffness_bend = joint_target_ke[bend_idx]
        k_bend = joint_penalty_k[bend_idx]
        k_bend_new = wp.min(k_bend + beta * C_bend, stiffness_bend)
        joint_penalty_k[bend_idx] = k_bend_new


@wp.kernel
def update_duals_body_body_contacts(
    rigid_contact_count: wp.array(dtype=int),
    rigid_contact_shape0: wp.array(dtype=int),
    rigid_contact_shape1: wp.array(dtype=int),
    rigid_contact_point0: wp.array(dtype=wp.vec3),
    rigid_contact_point1: wp.array(dtype=wp.vec3),
    rigid_contact_normal: wp.array(dtype=wp.vec3),
    rigid_contact_thickness0: wp.array(dtype=float),
    rigid_contact_thickness1: wp.array(dtype=float),
    shape_body: wp.array(dtype=int),
    body_q: wp.array(dtype=wp.transform),
    contact_material_ke: wp.array(dtype=float),
    beta: float,
    contact_penalty_k: wp.array(dtype=float),  # input/output
):
    """
    Update AVBD penalty parameters for contact constraints (per-iteration).

    Increases each contact's penalty by beta * penetration and clamps to the
    per-contact material stiffness.

    Args:
        rigid_contact_count: Number of active contacts
        rigid_contact_shape0/1: Shape ids for each contact pair
        rigid_contact_point0/1: Contact points in local shape frames
        rigid_contact_normal: Contact normals (pointing from shape0 to shape1)
        rigid_contact_thickness0/1: Per-shape thickness (for SDF/capsule padding)
        shape_body: Map from shape id to body id (-1 if kinematic/ground)
        body_q: Current body transforms
        contact_material_ke: Per-contact target stiffness
        beta: Adaptation rate
        contact_penalty_k: In/out per-contact adaptive penalties
    """
    idx = wp.tid()
    if idx >= rigid_contact_count[0]:
        return

    # Read contact geometry
    shape_id_0 = rigid_contact_shape0[idx]
    shape_id_1 = rigid_contact_shape1[idx]
    body_id_0 = shape_body[shape_id_0]
    body_id_1 = shape_body[shape_id_1]

    # Skip invalid contacts (both bodies kinematic/ground)
    if body_id_0 < 0 and body_id_1 < 0:
        return

    # Read cached material stiffness
    stiffness = contact_material_ke[idx]

    # Transform contact points to world space
    if body_id_0 >= 0:
        p0_world = wp.transform_point(body_q[body_id_0], rigid_contact_point0[idx])
    else:
        p0_world = rigid_contact_point0[idx]

    if body_id_1 >= 0:
        p1_world = wp.transform_point(body_q[body_id_1], rigid_contact_point1[idx])
    else:
        p1_world = rigid_contact_point1[idx]

    # Compute penetration depth (constraint violation)
    # Distance along the stored normal (normal points shape0 -> shape1)
    # dist = dot(n, p0 - p1); positive implies separation along normal
    d = p0_world - p1_world
    dist = wp.dot(rigid_contact_normal[idx], d)
    thickness_total = rigid_contact_thickness0[idx] + rigid_contact_thickness1[idx]
    penetration = wp.max(0.0, thickness_total - dist)

    # Update penalty: k_new = min(k + beta * |C|, stiffness)
    k = contact_penalty_k[idx]
    k_new = wp.min(k + beta * penetration, stiffness)
    contact_penalty_k[idx] = k_new


@wp.kernel
def update_duals_body_particle_contacts(
    body_particle_contact_count: wp.array(dtype=int),
    body_particle_contact_particle: wp.array(dtype=int),
    body_particle_contact_shape: wp.array(dtype=int),
    body_particle_contact_body_pos: wp.array(dtype=wp.vec3),
    body_particle_contact_normal: wp.array(dtype=wp.vec3),
    particle_q: wp.array(dtype=wp.vec3),
    particle_radius: wp.array(dtype=float),
    shape_body: wp.array(dtype=int),
    body_q: wp.array(dtype=wp.transform),
    body_particle_contact_material_ke: wp.array(dtype=float),
    beta: float,
    body_particle_contact_penalty_k: wp.array(dtype=float),  # input/output
):
    """
    Update AVBD penalty parameters for soft contact constraints (per-iteration).

    Increases each soft contact's penalty by beta * penetration and clamps to the
    per-contact material stiffness.

    Args:
        body_particle_contact_count: Number of active body-particle soft contacts
        body_particle_contact_particle: Particle index for each body-particle soft contact
        body_particle_contact_shape: Shape index for each body-particle soft contact
        body_particle_contact_body_pos: Contact points in local shape frames
        body_particle_contact_normal: Contact normals (pointing from rigid to particle)
        particle_q: Current particle positions
        particle_radius: Particle radii
        shape_body: Map from shape id to body id (-1 if kinematic/ground)
        body_q: Current body transforms
        body_particle_contact_material_ke: Per-contact target stiffness (averaged)
        beta: Adaptation rate
        body_particle_contact_penalty_k: In/out per-contact adaptive penalties
    """
    idx = wp.tid()
    if idx >= body_particle_contact_count[0]:
        return

    # Read contact data
    particle_idx = body_particle_contact_particle[idx]
    shape_idx = body_particle_contact_shape[idx]
    body_idx = shape_body[shape_idx] if shape_idx >= 0 else -1

    # Read cached material stiffness
    stiffness = body_particle_contact_material_ke[idx]

    # Transform contact point to world space
    # For rigid bodies (body_idx >= 0), transform from body-local to world space
    X_wb = wp.transform_identity()
    if body_idx >= 0:
        X_wb = body_q[body_idx]

    cp_local = body_particle_contact_body_pos[idx]
    cp_world = wp.transform_point(X_wb, cp_local)

    # Get particle data
    particle_pos = particle_q[particle_idx]
    radius = particle_radius[particle_idx]
    n = body_particle_contact_normal[idx]

    # Compute penetration depth (constraint violation)
    penetration = -(wp.dot(n, particle_pos - cp_world) - radius)
    penetration = wp.max(0.0, penetration)

    # Update penalty: k_new = min(k + beta * |C|, stiffness)
    k = body_particle_contact_penalty_k[idx]
    k_new = wp.min(k + beta * penetration, stiffness)
    body_particle_contact_penalty_k[idx] = k_new


# -----------------------------
# Post-iteration kernels (after all iterations)
# -----------------------------
@wp.kernel
def update_body_velocity(
    dt: float,
    body_q: wp.array(dtype=wp.transform),
    body_q_prev: wp.array(dtype=wp.transform),
    body_com: wp.array(dtype=wp.vec3),
    body_qd: wp.array(dtype=wp.spatial_vector),
):
    """
    Update body velocities from position changes (world frame).

    Computes linear and angular velocities using finite differences.

    Linear: v = (com_current - com_prev) / dt
    Angular: omega from quaternion difference dq = q * q_prev^-1

    Args:
        dt: Time step
        body_q: Current body transforms (world)
        body_q_prev: Previous body transforms (world)
        body_com: Center of mass offsets (local frame)
        body_qd: Output body velocities (spatial vectors, world frame)
    """
    tid = wp.tid()

    # Read transforms
    pose = body_q[tid]
    pose_prev = body_q_prev[tid]

    x = wp.transform_get_translation(pose)
    x_prev = wp.transform_get_translation(pose_prev)
    q = wp.transform_get_rotation(pose)
    q_prev = wp.transform_get_rotation(pose_prev)

    # Compute COM positions
    com_local = body_com[tid]
    x_com = x + wp.quat_rotate(q, com_local)
    x_com_prev = x_prev + wp.quat_rotate(q_prev, com_local)

    # Linear velocity
    v = (x_com - x_com_prev) / dt

    # Angular velocity
    omega = quat_velocity(q, q_prev, dt)

    body_qd[tid] = wp.spatial_vector(v, omega)


@wp.kernel
def update_cable_dahl_state(
    # Joint geometry
    joint_type: wp.array(dtype=int),
    joint_parent: wp.array(dtype=int),
    joint_child: wp.array(dtype=int),
    joint_X_p: wp.array(dtype=wp.transform),
    joint_X_c: wp.array(dtype=wp.transform),
    joint_qd_start: wp.array(dtype=int),
    joint_target_ke: wp.array(dtype=float),
    # Body states (final, after solver convergence)
    body_q: wp.array(dtype=wp.transform),
    body_q_rest: wp.array(dtype=wp.transform),
    body_com: wp.array(dtype=wp.vec3),
    # Dahl model parameters (PER-JOINT arrays, isotropic)
    joint_eps_max: wp.array(dtype=float),
    joint_tau: wp.array(dtype=float),
    # Dahl state (inputs - from previous timestep, outputs - to next timestep) - component-wise (vec3)
    joint_sigma_prev: wp.array(dtype=wp.vec3),  # input/output
    joint_kappa_prev: wp.array(dtype=wp.vec3),  # input/output
    joint_dkappa_prev: wp.array(dtype=wp.vec3),  # input/output (stores Delta kappa)
):
    """
    Post-iteration kernel: update Dahl hysteresis state after solver convergence (component-wise).

    Stores final curvature, friction stress, and curvature Delta kappa for the next step. Each
    curvature component (x, y, z) is updated independently to preserve path-dependent memory.

    Args:
        joint_type: Joint type (only updates for cable joints)
        joint_parent, joint_child: Parent/child body indices
        joint_X_p, joint_X_c: Joint frames in parent/child
        joint_qd_start: DOF start index per joint
        joint_target_ke: Target stiffness per DOF [N*m^2] (material EI for bending)
        body_q: Final body transforms (after convergence)
        body_q_rest: Rest body transforms
        body_com: Body COM in local frame (used to compute segment length for Cosserat discretization)
        joint_sigma_prev: Friction stress state (read old, write new), wp.vec3 per joint
        joint_kappa_prev: Curvature state (read old, write new), wp.vec3 per joint
        joint_dkappa_prev: Delta-kappa state (write new), wp.vec3 per joint
        joint_eps_max: Maximum persistent strain [rad] (scalar per joint)
        joint_tau: Memory decay length [rad] (scalar per joint)
    """
    j = wp.tid()

    # Only update cable joints
    if joint_type[j] != JointType.CABLE:
        return

    # Get parent and child body indices
    parent = joint_parent[j]
    child = joint_child[j]

    # Skip invalid joints
    if parent < 0 or child < 0:
        return

    # Compute joint frames in world space (final state)
    X_wp = body_q[parent] * joint_X_p[j]
    X_wc = body_q[child] * joint_X_c[j]
    X_wp_rest = body_q_rest[parent] * joint_X_p[j]
    X_wc_rest = body_q_rest[child] * joint_X_c[j]

    q_wp = wp.transform_get_rotation(X_wp)
    q_wc = wp.transform_get_rotation(X_wc)
    q_wp_rest = wp.transform_get_rotation(X_wp_rest)
    q_wc_rest = wp.transform_get_rotation(X_wc_rest)

    # Compute final curvature vector at end of timestep
    kappa_final = cable_get_kappa(q_wp, q_wc, q_wp_rest, q_wc_rest)

    # Read stored Dahl state (component-wise vectors)
    kappa_old = joint_kappa_prev[j]  # stored curvature
    d_kappa_old = joint_dkappa_prev[j]  # stored Delta kappa
    sigma_old = joint_sigma_prev[j]  # stored friction stress

    # Read per-joint Dahl parameters (isotropic)
    eps_max = joint_eps_max[j]  # Maximum persistent strain [rad]
    tau = joint_tau[j]  # Memory decay length [rad]

    # Get bend stiffness and segment length for this joint
    # Cable joint DOFs: DOF 0 = stretch (linear), DOF 1 = bend (angular)
    dof_start = joint_qd_start[j]
    bend_dof_index = dof_start + 1
    k_bend_material = joint_target_ke[bend_dof_index]  # Material stiffness EI [N*m^2]

    # Compute segment length (Cosserat rod discretization)
    parent_com = body_com[parent]
    rest_length = 2.0 * parent_com[2]  # Capsule segment length [m]

    # Skip degenerate segments
    if rest_length < _SMALL_LENGTH_EPS:
        return

    # Effective stiffness (same scaling as elastic force): k_eff = EI / L
    k_bend = k_bend_material / rest_length  # [N*m^2] / [m] = [N*m]

    # Compute maximum friction stress from strain (paper's approach: sigma_max = k_eff * eps_max)
    # Note: Must use k_eff (not k_material) to match elastic force scaling.
    # For isotropic case: same sigma_max for all components
    sigma_max = k_bend * eps_max  # [N*m] * [rad] = [N*m]

    # Early-out: disable friction if envelope is zero/invalid
    if sigma_max <= 0.0 or tau <= 0.0:
        joint_sigma_prev[j] = wp.vec3(0.0)
        joint_kappa_prev[j] = kappa_final
        joint_dkappa_prev[j] = kappa_final - kappa_old  # store Delta kappa
        return

    # Update each component independently (3 separate hysteresis loops)
    sigma_final_out = wp.vec3(0.0)
    d_kappa_out = wp.vec3(0.0)

    for axis in range(3):
        # Get component values
        kappa_i_final = kappa_final[axis]
        kappa_i_prev = kappa_old[axis]
        d_kappa_i_prev = d_kappa_old[axis]
        sigma_i_prev = sigma_old[axis]

        # Curvature change for this component
        d_kappa_i = kappa_i_final - kappa_i_prev

        # Direction flag (same logic as pre-iteration kernel), in kappa-space
        s_i = 1.0
        if d_kappa_i > _DAHL_KAPPADOT_DEADBAND:
            s_i = 1.0
        elif d_kappa_i < -_DAHL_KAPPADOT_DEADBAND:
            s_i = -1.0
        else:
            # Within deadband: maintain previous direction
            s_i = 1.0 if d_kappa_i_prev >= 0.0 else -1.0

        # sigma_i_next = s_i*sigma_max * [1 - exp(-s_i*d_kappa_i/tau)] + sigma_i_prev * exp(-s_i*d_kappa_i/tau)
        exp_term = wp.exp(-s_i * d_kappa_i / tau)
        sigma_i_next = s_i * sigma_max * (1.0 - exp_term) + sigma_i_prev * exp_term

        # Store component results
        sigma_final_out[axis] = sigma_i_next
        d_kappa_out[axis] = d_kappa_i

    # Store final vector state for next timestep
    joint_sigma_prev[j] = sigma_final_out
    joint_kappa_prev[j] = kappa_final
    joint_dkappa_prev[j] = d_kappa_out
