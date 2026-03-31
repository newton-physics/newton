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

from newton._src.core.types import MAXVAL
from newton._src.geometry.hashtable import HASHTABLE_EMPTY_KEY, hashtable_find, hashtable_find_or_insert
from newton._src.math import quat_velocity
from newton._src.sim import JointType
from newton._src.solvers.solver import integrate_rigid_body

wp.set_module_options({"enable_backward": False})

# ---------------------------------
# Constants
# ---------------------------------

_SMALL_ANGLE_EPS = wp.constant(1.0e-7)
"""Small-angle threshold [rad] for guards and series expansions"""

_DRIVE_LIMIT_MODE_NONE = wp.constant(0)
_DRIVE_LIMIT_MODE_LIMIT_LOWER = wp.constant(1)
_DRIVE_LIMIT_MODE_LIMIT_UPPER = wp.constant(2)
_DRIVE_LIMIT_MODE_DRIVE = wp.constant(3)

_SMALL_LENGTH_EPS = wp.constant(1.0e-9)
"""Small length tolerance (e.g., segment length checks)"""

_USE_SMALL_ANGLE_APPROX = wp.constant(True)
"""If True use first-order small-angle rotation approximation; if False use closed-form rotation update"""

_DAHL_KAPPADOT_DEADBAND = wp.constant(1.0e-6)
"""Deadband threshold for hysteresis direction selection"""

_NUM_CONTACT_THREADS_PER_BODY = wp.constant(16)
"""Threads per body for contact accumulation using strided iteration"""

_CONTACT_HASH_GRID_DIM = wp.constant(128)
"""Grid dimension per axis for contact spatial hashing (128^3 = ~2M cells, fits in 21 bits)"""

# ---------------------------------
# Contact history hash helpers
# ---------------------------------


@wp.func
def _pack_shape_pair_bits(s0: int, s1: int) -> wp.uint64:
    """Canonical shape pair packed into upper 42 bits (21 bits each, min/max ordered)."""
    lo = wp.min(s0, s1)
    hi = wp.max(s0, s1)
    return (wp.uint64(lo) << wp.uint64(21) | wp.uint64(hi)) << wp.uint64(21)


@wp.func
def _compute_spatial_cell(point: wp.vec3, cell_size_inv: float) -> int:
    """Quantize body-local point to 1D cell ID on a 128^3 grid."""
    gd = _CONTACT_HASH_GRID_DIM
    ix = int(wp.floor(point[0] * cell_size_inv)) % gd
    iy = int(wp.floor(point[1] * cell_size_inv)) % gd
    iz = int(wp.floor(point[2] * cell_size_inv)) % gd
    if ix < 0:
        ix += gd
    if iy < 0:
        iy += gd
    if iz < 0:
        iz += gd
    return ix * gd * gd + iy * gd + iz


@wp.func
def _offset_cell(base_cell: int, dx: int, dy: int, dz: int) -> int:
    """Compute neighbor cell ID from a base cell and 3D offset."""
    gd = _CONTACT_HASH_GRID_DIM
    ix = (base_cell // (gd * gd) + dx) % gd
    iy = ((base_cell // gd) % gd + dy) % gd
    iz = (base_cell % gd + dz) % gd
    if ix < 0:
        ix += gd
    if iy < 0:
        iy += gd
    if iz < 0:
        iz += gd
    return ix * gd * gd + iy * gd + iz


@wp.func
def _build_contact_history_key(pair_bits: wp.uint64, cell: int) -> wp.uint64:
    """Compose a 64-bit hash key from shape-pair bits (upper 42) and cell ID (lower 21)."""
    return pair_bits | wp.uint64(cell)


# ---------------------------------
# Helper classes and device functions
# ---------------------------------


class vec6(wp.types.vector(length=6, dtype=wp.float32)):
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

    q_rel = wp.quat_inverse(q_wp) * q_wc
    q_rel_rest = wp.quat_inverse(q_wp_rest) * q_wc_rest
    R_align = wp.quat_to_matrix(q_rel * wp.quat_inverse(q_rel_rest))

    Jr_inv = compute_right_jacobian_inverse(kappa_now)
    omega_right = wp.transpose(R_align) * omega_rel_parent
    return Jr_inv * omega_right


@wp.func
def build_joint_projectors(
    jt: int,
    joint_axis: wp.array(dtype=wp.vec3),
    qd_start: int,
    lin_count: int,
    ang_count: int,
    q_wp_rot: wp.quat,
):
    """Build orthogonal-complement projectors P_lin and P_ang for a joint.

    P = I - sum(ai * ai^T) removes free DOF directions.
    Invariant: free axes must be orthonormal for P to be a valid projector (P^2 = P).

    Args:
        jt: Joint type (JointType enum).
        joint_axis: Per-DOF axis directions.
        qd_start: Start index into joint_axis for this joint's DOFs.
        lin_count: Number of free linear DOFs (0 for most joints; 1 for PRISMATIC; 0-3 for D6).
        ang_count: Number of free angular DOFs (0 for most joints; 1 for REVOLUTE; 0-3 for D6).
        q_wp_rot: Parent joint frame rotation (world). Used to rotate linear axes to world space.

    Returns:
        (P_lin, P_ang): Orthogonal-complement projectors for linear and angular constraints.
    """
    P_lin = wp.identity(3, float)
    P_ang = wp.identity(3, float)

    if jt == JointType.PRISMATIC:
        a_w = wp.normalize(wp.quat_rotate(q_wp_rot, joint_axis[qd_start]))
        P_lin = P_lin - wp.outer(a_w, a_w)
    elif jt == JointType.D6:
        if lin_count > 0:
            a0_w = wp.normalize(wp.quat_rotate(q_wp_rot, joint_axis[qd_start]))
            P_lin = P_lin - wp.outer(a0_w, a0_w)
        if lin_count > 1:
            a1_w = wp.normalize(wp.quat_rotate(q_wp_rot, joint_axis[qd_start + 1]))
            P_lin = P_lin - wp.outer(a1_w, a1_w)
        if lin_count > 2:
            a2_w = wp.normalize(wp.quat_rotate(q_wp_rot, joint_axis[qd_start + 2]))
            P_lin = P_lin - wp.outer(a2_w, a2_w)

    if jt == JointType.REVOLUTE:
        a = wp.normalize(joint_axis[qd_start])
        P_ang = P_ang - wp.outer(a, a)
    elif jt == JointType.D6:
        if ang_count > 0:
            a0 = wp.normalize(joint_axis[qd_start + lin_count])
            P_ang = P_ang - wp.outer(a0, a0)
        if ang_count > 1:
            a1 = wp.normalize(joint_axis[qd_start + lin_count + 1])
            P_ang = P_ang - wp.outer(a1, a1)
        if ang_count > 2:
            a2 = wp.normalize(joint_axis[qd_start + lin_count + 2])
            P_ang = P_ang - wp.outer(a2, a2)

    return P_lin, P_ang


@wp.func
def _average_contact_material(
    ke0: float,
    kd0: float,
    mu0: float,
    ke1: float,
    kd1: float,
    mu1: float,
):
    """Average material properties for a contact pair.

    ke, kd: arithmetic mean (additive stiffness/damping).
    mu: geometric mean (standard friction convention).
    """
    avg_ke = 0.5 * (ke0 + ke1)
    avg_kd = 0.5 * (kd0 + kd1)
    avg_mu = wp.sqrt(mu0 * mu1)
    return avg_ke, avg_kd, avg_mu


@wp.func
def _update_dual_vec3(
    C_vec: wp.vec3,
    C0: wp.vec3,
    alpha: float,
    k: float,
    lam: wp.vec3,
    is_hard: int,
):
    """Shared AVBD dual update for a vec3 constraint slot.

    Hard mode: stabilized constraint + lambda accumulation.
    Soft mode: lambda unchanged.

    Args:
        C_vec: Current constraint violation vector.
        C0: Initial constraint violation snapshot for stabilization.
        alpha: C0 stabilization factor.
        k: Fixed penalty stiffness.
        lam: Current Lagrange multiplier.
        is_hard: 1 for hard (AL), 0 for soft (penalty-only).

    Returns:
        wp.vec3: Updated Lagrange multiplier.
    """
    if is_hard == 1:
        C_stab = C_vec - alpha * C0
        lam_new = k * C_stab + lam
    else:
        lam_new = lam
    return lam_new


@wp.func
def evaluate_angular_constraint_force_hessian(
    q_wp: wp.quat,
    q_wc: wp.quat,
    q_wp_rest: wp.quat,
    q_wc_rest: wp.quat,
    q_wp_prev: wp.quat,
    q_wc_prev: wp.quat,
    is_parent: bool,
    penalty_k: float,
    P: wp.mat33,
    sigma0: wp.vec3,
    C_fric: wp.vec3,
    lambda_ang: wp.vec3,
    C0_ang: wp.vec3,
    alpha: float,
    damping: float,
    dt: float,
):
    """Projected angular constraint force/Hessian using rotation-vector error (kappa).

    Unified evaluator for all joint types. Computes constraint force and Hessian
    in the constrained subspace defined by the orthogonal-complement projector P.

    C0 stabilization: when alpha > 0 and C0_ang is nonzero, the effective
    kappa is kappa - alpha*C0_ang (initial violation snapshot).

    Special cases by projector:
      - P = I: isotropic (CABLE bend, FIXED angular)
      - P = I - a*a^T: revolute (1 free angular axis)
      - arbitrary P: D6 (0-3 free angular axes)

    Dahl friction (sigma0, C_fric) is only valid when P = I (isotropic).
    Pass vec3(0) for both when P != I.

    Returns:
        (tau_world, H_aa, kappa, J_world) -- constraint torque and Hessian in world
        frame, plus the curvature vector and world-frame Jacobian for reuse by the
        drive/limit block.
    """
    inv_dt = 1.0 / dt

    kappa_now_vec = cable_get_kappa(q_wp, q_wc, q_wp_rest, q_wc_rest)
    kappa_stab = kappa_now_vec - alpha * C0_ang
    kappa_perp = P * kappa_stab

    Jr_inv = compute_right_jacobian_inverse(kappa_now_vec)
    R_wp = wp.quat_to_matrix(q_wp)

    q_rel = wp.quat_inverse(q_wp) * q_wc
    q_rel_rest = wp.quat_inverse(q_wp_rest) * q_wc_rest
    R_align = wp.quat_to_matrix(q_rel * wp.quat_inverse(q_rel_rest))

    J_world = R_wp * (R_align * wp.transpose(Jr_inv))

    f_local = penalty_k * kappa_perp + sigma0 + lambda_ang

    H_local = penalty_k * P + wp.mat33(
        C_fric[0],
        0.0,
        0.0,
        0.0,
        C_fric[1],
        0.0,
        0.0,
        0.0,
        C_fric[2],
    )

    if damping > 0.0:
        omega_p_world = quat_velocity(q_wp, q_wp_prev, dt)
        omega_c_world = quat_velocity(q_wc, q_wc_prev, dt)

        dkappa_dt_vec = compute_kappa_dot_analytic(
            q_wp, q_wc, q_wp_rest, q_wc_rest, omega_p_world, omega_c_world, kappa_now_vec
        )
        dkappa_perp = P * dkappa_dt_vec
        f_damp_local = (damping * penalty_k) * dkappa_perp
        f_local = f_local + f_damp_local

        k_damp = (damping * inv_dt) * penalty_k
        H_local = H_local + k_damp * P

    H_aa = J_world * (H_local * wp.transpose(J_world))

    tau_world = J_world * f_local
    if not is_parent:
        tau_world = -tau_world

    return tau_world, H_aa, kappa_now_vec, J_world


@wp.func
def evaluate_linear_constraint_force_hessian(
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
    P: wp.mat33,
    lambda_lin: wp.vec3,
    C0_lin: wp.vec3,
    alpha: float,
    damping: float,
    dt: float,
):
    """Projected linear constraint force/Hessian for anchor coincidence.

    Unified evaluator for all joint types. Computes C = x_c - x_p, projects
    with P, and returns force/Hessian in world frame.

    C0 stabilization: when alpha > 0 and C0_lin is nonzero, the effective
    constraint violation is C - alpha*C0 (initial violation snapshot).

    Special cases by projector:
      - P = I: isotropic (BALL, CABLE stretch, FIXED linear, REVOLUTE linear)
      - P = I - a*a^T: prismatic (1 free linear axis)
      - arbitrary P: D6 (0-3 free linear axes)

    Returns:
      - force (wp.vec3): Linear force (world)
      - torque (wp.vec3): Angular torque (world)
      - H_ll (wp.mat33): Linear-linear block
      - H_al (wp.mat33): Angular-linear block
      - H_aa (wp.mat33): Angular-angular block
    """
    x_p = wp.transform_get_translation(X_wp)
    x_c = wp.transform_get_translation(X_wc)

    if is_parent:
        com_w = wp.transform_point(parent_pose, parent_com)
        r = x_p - com_w
    else:
        com_w = wp.transform_point(child_pose, child_com)
        r = x_c - com_w

    C_vec = x_c - x_p
    C_stab = C_vec - alpha * C0_lin
    C_perp = P * C_stab

    f_attachment = penalty_k * C_perp + lambda_lin

    rx = wp.skew(r)
    K_point = penalty_k * P

    H_ll = K_point
    H_al = rx * K_point
    H_aa = wp.transpose(rx) * K_point * rx

    if damping > 0.0:
        x_p_prev = wp.transform_get_translation(X_wp_prev)
        x_c_prev = wp.transform_get_translation(X_wc_prev)
        C_vec_prev = x_c_prev - x_p_prev
        inv_dt = 1.0 / dt
        dC_dt = (C_vec - C_vec_prev) * inv_dt
        dC_dt_perp = P * dC_dt

        damping_coeff = damping * penalty_k
        f_damping = damping_coeff * dC_dt_perp
        f_attachment = f_attachment + f_damping

        damp_scale = damping * inv_dt
        H_ll_damp = damp_scale * H_ll
        H_al_damp = damp_scale * H_al
        H_aa_damp = damp_scale * H_aa

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
    body_q_prev: wp.array(dtype=wp.transform),
    body_com: wp.array(dtype=wp.vec3),
    contact_point_a_local: wp.vec3,
    contact_point_b_local: wp.vec3,
    contact_normal: wp.vec3,
    penetration_depth: float,
    tangential_constraint: wp.vec3,
    contact_ke: float,
    contact_ke_t: float,
    contact_kd: float,
    contact_lam: wp.vec3,
    friction_mu: float,
    friction_epsilon: float,
    hard_contact: int,
    dt: float,
):
    """Compute augmented-Lagrangian contact forces and 3x3 Hessian blocks for a rigid contact pair.

    Hard contacts: AL friction with vec3 lambda, cone clamping, tangential penalty.
    Soft contacts: velocity-based IPC friction with scalar penalty.
    """
    lam_n = wp.dot(contact_lam, contact_normal)

    if penetration_depth <= _SMALL_LENGTH_EPS and lam_n <= 0.0:
        zero_vec = wp.vec3(0.0)
        zero_mat = wp.mat33(0.0)
        return (zero_vec, zero_vec, zero_mat, zero_mat, zero_mat, zero_vec, zero_vec, zero_mat, zero_mat, zero_mat)

    f_n = contact_ke * penetration_depth + lam_n
    if f_n <= 0.0 or contact_ke <= 0.0:
        if lam_n > 0.0 and hard_contact == 1:
            f_n = 0.0
            k_rescaled = lam_n / wp.max(wp.abs(penetration_depth), 1.0e-8)
            k_rescaled = wp.min(k_rescaled, contact_ke)
            contact_ke = wp.max(k_rescaled, 1.0)
            contact_ke_t = 0.0
        else:
            zero_vec = wp.vec3(0.0)
            zero_mat = wp.mat33(0.0)
            return (zero_vec, zero_vec, zero_mat, zero_mat, zero_mat, zero_vec, zero_vec, zero_mat, zero_mat, zero_mat)

    if body_a_index < 0:
        X_wa = wp.transform_identity()
        X_wa_prev = wp.transform_identity()
        body_a_com_local = wp.vec3(0.0)
    else:
        X_wa = body_q[body_a_index]
        X_wa_prev = body_q_prev[body_a_index]
        body_a_com_local = body_com[body_a_index]

    if body_b_index < 0:
        X_wb = wp.transform_identity()
        X_wb_prev = wp.transform_identity()
        body_b_com_local = wp.vec3(0.0)
    else:
        X_wb = body_q[body_b_index]
        X_wb_prev = body_q_prev[body_b_index]
        body_b_com_local = body_com[body_b_index]

    x_com_a_now = wp.transform_point(X_wa, body_a_com_local)
    x_com_b_now = wp.transform_point(X_wb, body_b_com_local)

    x_c_a_now = wp.transform_point(X_wa, contact_point_a_local)
    x_c_b_now = wp.transform_point(X_wb, contact_point_b_local)
    x_c_a_prev = wp.transform_point(X_wa_prev, contact_point_a_local)
    x_c_b_prev = wp.transform_point(X_wb_prev, contact_point_b_local)

    n_outer = wp.outer(contact_normal, contact_normal)

    v_rel = (x_c_b_now - x_c_b_prev - x_c_a_now + x_c_a_prev) / dt
    v_dot_n = wp.dot(contact_normal, v_rel)

    if hard_contact == 1:
        lam_t = contact_lam - contact_normal * lam_n
        f_t_vec = contact_ke_t * tangential_constraint + lam_t
        f_t_len = wp.length(f_t_vec)
        cone_limit = friction_mu * f_n
        if f_t_len > cone_limit and f_t_len > 0.0:
            cone_ratio = cone_limit / f_t_len
            f_t_vec = f_t_vec * cone_ratio
            contact_ke_t = contact_ke_t * cone_ratio

        f_total = contact_normal * f_n + f_t_vec
        I3 = wp.identity(n=3, dtype=float)
        K_total = contact_ke * n_outer + contact_ke_t * (I3 - n_outer)

        if contact_kd > 0.0 and v_dot_n < 0.0 and f_n > 0.0:
            damping_coeff = contact_kd * contact_ke
            f_total = f_total - damping_coeff * v_dot_n * contact_normal
            K_total = K_total + (damping_coeff / dt) * n_outer
    else:
        f_total = contact_normal * f_n
        K_total = contact_ke * n_outer

        if contact_kd > 0.0 and v_dot_n < 0.0 and f_n > 0.0:
            damping_coeff = contact_kd * contact_ke
            f_total = f_total - damping_coeff * v_dot_n * contact_normal
            K_total = K_total + (damping_coeff / dt) * n_outer

        if friction_mu > 0.0 and f_n > 0.0:
            v_t = v_rel - contact_normal * v_dot_n
            f_friction, K_friction = compute_projected_isotropic_friction(
                friction_mu, f_n, contact_normal, v_t * dt, friction_epsilon * dt
            )
            f_total = f_total + f_friction
            K_total = K_total + K_friction

    r_a = x_c_a_now - x_com_a_now
    r_b = x_c_b_now - x_com_b_now

    r_a_skew = wp.skew(r_a)
    r_a_skew_T_K = wp.transpose(r_a_skew) * K_total
    h_aa_a = r_a_skew_T_K * r_a_skew
    h_al_a = -r_a_skew_T_K

    r_b_skew = wp.skew(r_b)
    r_b_skew_T_K = wp.transpose(r_b_skew) * K_total
    h_aa_b = r_b_skew_T_K * r_b_skew
    h_al_b = -r_b_skew_T_K

    return (
        -f_total,
        wp.cross(r_a, -f_total),
        K_total,
        h_al_a,
        h_aa_a,
        f_total,
        wp.cross(r_b, f_total),
        K_total,
        h_al_b,
        h_aa_b,
    )


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
    body_q_prev: wp.array(dtype=wp.transform),
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
        particle_prev_pos: Previous particle position (world frame) used as the
            "previous" position for finite-difference contact-relative velocity.
        contact_index: Index in the body-particle contact arrays
        body_particle_contact_ke: Effective body-particle contact stiffness
        body_particle_contact_kd: Effective body-particle contact damping
        friction_mu: Effective body-particle friction coefficient
        friction_epsilon: Friction regularization distance
        particle_radius: Array of particle radii
        shape_material_mu: Array of shape friction coefficients
        shape_body: Array mapping shape index to body index
        body_q: Current body transforms
        body_q_prev: Previous body transforms (for finite-difference body
            velocity when available)
        body_qd: Body spatial velocities (fallback when no previous pose is provided)
        body_com: Body centers of mass (local frame)
        contact_shape: Array of shape indices for each soft contact
        contact_body_pos: Array of contact points (local to shape)
        contact_body_vel: Array of contact velocities (local frame)
        contact_normal: Array of contact normals (world frame, from rigid to particle)
        dt: Time window [s] used for finite-difference damping/friction.

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

        # Combine body-particle friction and shape material friction using geometric mean.
        mu = wp.sqrt(friction_mu * shape_material_mu[shape_index])

        dx = particle_pos - particle_prev_pos

        if wp.dot(n, dx) < 0.0:
            # Damping coefficient is scaled by contact stiffness (consistent with rigid-rigid)
            damping_coeff = body_particle_contact_kd * body_particle_contact_ke
            damping_hessian = (damping_coeff / dt) * wp.outer(n, n)
            body_contact_hessian = body_contact_hessian + damping_hessian
            body_contact_force = body_contact_force - damping_hessian * dx

        # body velocity
        if body_q_prev:
            # if body_q_prev is available, compute velocity using finite difference method
            # this is more accurate for simulating static friction
            X_wb_prev = wp.transform_identity()
            if body_index >= 0:
                X_wb_prev = body_q_prev[body_index]
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
def resolve_drive_limit_mode(
    q: float,
    target_pos: float,
    lim_lower: float,
    lim_upper: float,
    has_drive: bool,
    has_limits: bool,
):
    mode = _DRIVE_LIMIT_MODE_NONE
    err_pos = float(0.0)
    drive_target = target_pos
    if has_limits:
        drive_target = wp.clamp(target_pos, lim_lower, lim_upper)
        if q < lim_lower:
            mode = _DRIVE_LIMIT_MODE_LIMIT_LOWER
            err_pos = q - lim_lower
        elif q > lim_upper:
            mode = _DRIVE_LIMIT_MODE_LIMIT_UPPER
            err_pos = q - lim_upper
    if mode == _DRIVE_LIMIT_MODE_NONE and has_drive:
        mode = _DRIVE_LIMIT_MODE_DRIVE
        err_pos = q - drive_target
    return mode, err_pos


@wp.func
def compute_kappa_and_jacobian(
    q_wp: wp.quat,
    q_wc: wp.quat,
    q_wp_rest: wp.quat,
    q_wc_rest: wp.quat,
):
    kappa = cable_get_kappa(q_wp, q_wc, q_wp_rest, q_wc_rest)
    Jr_inv = compute_right_jacobian_inverse(kappa)
    R_wp = wp.quat_to_matrix(q_wp)
    q_rel = wp.quat_inverse(q_wp) * q_wc
    q_rel_rest = wp.quat_inverse(q_wp_rest) * q_wc_rest
    R_align = wp.quat_to_matrix(q_rel * wp.quat_inverse(q_rel_rest))
    J_world = R_wp * (R_align * wp.transpose(Jr_inv))
    return kappa, J_world


@wp.func
def apply_angular_drive_limit_torque(
    a: wp.vec3,
    J_world: wp.mat33,
    is_parent: bool,
    f_scalar: float,
    H_scalar: float,
):
    f_local = f_scalar * a
    H_local = H_scalar * wp.outer(a, a)
    tau = J_world * f_local
    Haa = J_world * (H_local * wp.transpose(J_world))
    if not is_parent:
        tau = -tau
    return tau, Haa


@wp.func
def apply_linear_drive_limit_force(
    axis_w: wp.vec3,
    r: wp.vec3,
    is_parent: bool,
    f_scalar: float,
    H_scalar: float,
):
    f_attachment = f_scalar * axis_w
    aa = wp.outer(axis_w, axis_w)
    K_point = H_scalar * aa
    rx = wp.skew(r)
    Hll = K_point
    Hal = rx * K_point
    Haa = wp.transpose(rx) * K_point * rx
    force = f_attachment if is_parent else -f_attachment
    torque = wp.cross(r, force)
    return force, torque, Hll, Hal, Haa


@wp.func
def evaluate_joint_force_hessian(
    body_index: int,
    joint_index: int,
    body_q: wp.array(dtype=wp.transform),
    body_q_prev: wp.array(dtype=wp.transform),
    body_q_rest: wp.array(dtype=wp.transform),
    body_com: wp.array(dtype=wp.vec3),
    joint_type: wp.array(dtype=int),
    joint_enabled: wp.array(dtype=bool),
    joint_parent: wp.array(dtype=int),
    joint_child: wp.array(dtype=int),
    joint_X_p: wp.array(dtype=wp.transform),
    joint_X_c: wp.array(dtype=wp.transform),
    joint_axis: wp.array(dtype=wp.vec3),
    joint_qd_start: wp.array(dtype=int),
    joint_constraint_start: wp.array(dtype=int),
    joint_penalty_k: wp.array(dtype=float),
    joint_penalty_kd: wp.array(dtype=float),
    joint_sigma_start: wp.array(dtype=wp.vec3),
    joint_C_fric: wp.array(dtype=wp.vec3),
    # Drive parameters (DOF-indexed via joint_qd_start)
    joint_target_ke: wp.array(dtype=float),
    joint_target_kd: wp.array(dtype=float),
    joint_target_pos: wp.array(dtype=float),
    joint_target_vel: wp.array(dtype=float),
    # Limit parameters (DOF-indexed via joint_qd_start)
    joint_limit_lower: wp.array(dtype=float),
    joint_limit_upper: wp.array(dtype=float),
    joint_limit_ke: wp.array(dtype=float),
    joint_limit_kd: wp.array(dtype=float),
    joint_lambda_lin: wp.array(dtype=wp.vec3),
    joint_lambda_ang: wp.array(dtype=wp.vec3),
    joint_C0_lin: wp.array(dtype=wp.vec3),
    joint_C0_ang: wp.array(dtype=wp.vec3),
    joint_is_hard: wp.array(dtype=wp.int32),
    avbd_alpha: float,
    joint_dof_dim: wp.array(dtype=int, ndim=2),
    joint_rest_angle: wp.array(dtype=float),
    dt: float,
):
    """Compute AVBD joint force and Hessian contributions for one body.

    Supported joint types: CABLE, BALL, FIXED, REVOLUTE, PRISMATIC, D6.
    Uses unified projector-based constraint evaluators for all joint types.

    Indexing:
        joint_constraint_start[j] is a solver-owned start offset into the per-constraint
        arrays (joint_penalty_k, joint_penalty_kd). Only structural slots exist:
          - CABLE: 2 scalars -> [stretch, bend]
          - BALL:  1 scalar  -> [linear]
          - FIXED/REVOLUTE/PRISMATIC/D6: 2 scalars -> [linear, angular]
        Drive/limit forces read model stiffness directly and do not use penalty slots.
    """
    jt = joint_type[joint_index]
    if (
        jt != JointType.CABLE
        and jt != JointType.BALL
        and jt != JointType.FIXED
        and jt != JointType.REVOLUTE
        and jt != JointType.PRISMATIC
        and jt != JointType.D6
    ):
        return wp.vec3(0.0), wp.vec3(0.0), wp.mat33(0.0), wp.mat33(0.0), wp.mat33(0.0)

    if not joint_enabled[joint_index]:
        return wp.vec3(0.0), wp.vec3(0.0), wp.mat33(0.0), wp.mat33(0.0), wp.mat33(0.0)

    parent_index = joint_parent[joint_index]
    child_index = joint_child[joint_index]
    if body_index != child_index and (parent_index < 0 or body_index != parent_index):
        return wp.vec3(0.0), wp.vec3(0.0), wp.mat33(0.0), wp.mat33(0.0), wp.mat33(0.0)

    is_parent_body = parent_index >= 0 and body_index == parent_index

    X_pj = joint_X_p[joint_index]
    X_cj = joint_X_c[joint_index]

    if parent_index >= 0:
        parent_pose = body_q[parent_index]
        parent_pose_prev = body_q_prev[parent_index]
        parent_pose_rest = body_q_rest[parent_index]
        parent_com = body_com[parent_index]
    else:
        parent_pose = wp.transform(wp.vec3(0.0), wp.quat_identity())
        parent_pose_prev = parent_pose
        parent_pose_rest = parent_pose
        parent_com = wp.vec3(0.0)

    child_pose = body_q[child_index]
    child_pose_prev = body_q_prev[child_index]
    child_pose_rest = body_q_rest[child_index]
    child_com = body_com[child_index]

    X_wp = parent_pose * X_pj
    X_wc = child_pose * X_cj
    X_wp_prev = parent_pose_prev * X_pj
    X_wc_prev = child_pose_prev * X_cj
    X_wp_rest = parent_pose_rest * X_pj
    X_wc_rest = child_pose_rest * X_cj

    c_start = joint_constraint_start[joint_index]

    # Hoist quaternion extraction (shared by all angular constraints and drive/limits)
    q_wp = wp.transform_get_rotation(X_wp)
    q_wc = wp.transform_get_rotation(X_wc)
    q_wp_rest = wp.transform_get_rotation(X_wp_rest)
    q_wc_rest = wp.transform_get_rotation(X_wc_rest)
    q_wp_prev = wp.transform_get_rotation(X_wp_prev)
    q_wc_prev = wp.transform_get_rotation(X_wc_prev)

    P_I = wp.identity(3, float)

    # Hard/soft AL gating for the linear structural slot (slot 0)
    lin_lambda = wp.vec3(0.0)
    lin_C0 = wp.vec3(0.0)
    lin_alpha = float(0.0)
    if joint_is_hard[c_start] == 1:
        lin_lambda = joint_lambda_lin[joint_index]
        lin_C0 = joint_C0_lin[joint_index]
        lin_alpha = avbd_alpha

    # Hard/soft AL gating for the angular structural slot (slot 1)
    ang_lambda = wp.vec3(0.0)
    ang_C0 = wp.vec3(0.0)
    ang_alpha = float(0.0)
    ang_hard = 0
    if jt != JointType.BALL:
        ang_hard = joint_is_hard[c_start + 1]
    if ang_hard == 1:
        ang_lambda = joint_lambda_ang[joint_index]
        ang_C0 = joint_C0_ang[joint_index]
        ang_alpha = avbd_alpha

    if jt == JointType.CABLE:
        k_stretch = joint_penalty_k[c_start]
        k_bend = joint_penalty_k[c_start + 1]
        kd_stretch = joint_penalty_kd[c_start]
        kd_bend = joint_penalty_kd[c_start + 1]

        total_force = wp.vec3(0.0)
        total_torque = wp.vec3(0.0)
        total_H_ll = wp.mat33(0.0)
        total_H_al = wp.mat33(0.0)
        total_H_aa = wp.mat33(0.0)

        if k_bend > 0.0:
            if ang_hard == 1:
                sigma0 = wp.vec3(0.0)
                C_fric = wp.vec3(0.0)
            else:
                sigma0 = joint_sigma_start[joint_index]
                C_fric = joint_C_fric[joint_index]
            bend_torque, bend_H_aa, _bend_kappa, _bend_J = evaluate_angular_constraint_force_hessian(
                q_wp,
                q_wc,
                q_wp_rest,
                q_wc_rest,
                q_wp_prev,
                q_wc_prev,
                is_parent_body,
                k_bend,
                P_I,
                sigma0,
                C_fric,
                ang_lambda,
                ang_C0,
                ang_alpha,
                kd_bend,
                dt,
            )
            total_torque = total_torque + bend_torque
            total_H_aa = total_H_aa + bend_H_aa

        if k_stretch > 0.0:
            f_s, t_s, Hll_s, Hal_s, Haa_s = evaluate_linear_constraint_force_hessian(
                X_wp,
                X_wc,
                X_wp_prev,
                X_wc_prev,
                parent_pose,
                child_pose,
                parent_com,
                child_com,
                is_parent_body,
                k_stretch,
                P_I,
                lin_lambda,
                lin_C0,
                lin_alpha,
                kd_stretch,
                dt,
            )
            total_force = total_force + f_s
            total_torque = total_torque + t_s
            total_H_ll = total_H_ll + Hll_s
            total_H_al = total_H_al + Hal_s
            total_H_aa = total_H_aa + Haa_s

        return total_force, total_torque, total_H_ll, total_H_al, total_H_aa

    elif jt == JointType.BALL:
        k = joint_penalty_k[c_start]
        damping = joint_penalty_kd[c_start]
        if k > 0.0:
            return evaluate_linear_constraint_force_hessian(
                X_wp,
                X_wc,
                X_wp_prev,
                X_wc_prev,
                parent_pose,
                child_pose,
                parent_com,
                child_com,
                is_parent_body,
                k,
                P_I,
                lin_lambda,
                lin_C0,
                lin_alpha,
                damping,
                dt,
            )
        return wp.vec3(0.0), wp.vec3(0.0), wp.mat33(0.0), wp.mat33(0.0), wp.mat33(0.0)

    elif jt == JointType.FIXED:
        k_lin = joint_penalty_k[c_start + 0]
        kd_lin = joint_penalty_kd[c_start + 0]
        if k_lin > 0.0:
            f_lin, t_lin, Hll_lin, Hal_lin, Haa_lin = evaluate_linear_constraint_force_hessian(
                X_wp,
                X_wc,
                X_wp_prev,
                X_wc_prev,
                parent_pose,
                child_pose,
                parent_com,
                child_com,
                is_parent_body,
                k_lin,
                P_I,
                lin_lambda,
                lin_C0,
                lin_alpha,
                kd_lin,
                dt,
            )
        else:
            f_lin = wp.vec3(0.0)
            t_lin = wp.vec3(0.0)
            Hll_lin = wp.mat33(0.0)
            Hal_lin = wp.mat33(0.0)
            Haa_lin = wp.mat33(0.0)

        k_ang = joint_penalty_k[c_start + 1]
        kd_ang = joint_penalty_kd[c_start + 1]
        if k_ang > 0.0:
            t_ang, Haa_ang, _ang_kappa, _ang_J = evaluate_angular_constraint_force_hessian(
                q_wp,
                q_wc,
                q_wp_rest,
                q_wc_rest,
                q_wp_prev,
                q_wc_prev,
                is_parent_body,
                k_ang,
                P_I,
                wp.vec3(0.0),
                wp.vec3(0.0),
                ang_lambda,
                ang_C0,
                ang_alpha,
                kd_ang,
                dt,
            )
        else:
            t_ang = wp.vec3(0.0)
            Haa_ang = wp.mat33(0.0)

        return f_lin, t_lin + t_ang, Hll_lin, Hal_lin, Haa_lin + Haa_ang

    elif jt == JointType.REVOLUTE:
        qd_start = joint_qd_start[joint_index]
        P_lin, P_ang = build_joint_projectors(jt, joint_axis, qd_start, 0, 1, q_wp)
        a = wp.normalize(joint_axis[qd_start])

        k_lin = joint_penalty_k[c_start + 0]
        kd_lin = joint_penalty_kd[c_start + 0]
        if k_lin > 0.0:
            f_lin, t_lin, Hll_lin, Hal_lin, Haa_lin = evaluate_linear_constraint_force_hessian(
                X_wp,
                X_wc,
                X_wp_prev,
                X_wc_prev,
                parent_pose,
                child_pose,
                parent_com,
                child_com,
                is_parent_body,
                k_lin,
                P_lin,
                lin_lambda,
                lin_C0,
                lin_alpha,
                kd_lin,
                dt,
            )
        else:
            f_lin = wp.vec3(0.0)
            t_lin = wp.vec3(0.0)
            Hll_lin = wp.mat33(0.0)
            Hal_lin = wp.mat33(0.0)
            Haa_lin = wp.mat33(0.0)

        k_ang = joint_penalty_k[c_start + 1]
        kd_ang = joint_penalty_kd[c_start + 1]

        kappa_cached = wp.vec3(0.0)
        J_world_cached = wp.mat33(0.0)
        has_cached = False

        if k_ang > 0.0:
            t_ang, Haa_ang, kappa_cached, J_world_cached = evaluate_angular_constraint_force_hessian(
                q_wp,
                q_wc,
                q_wp_rest,
                q_wc_rest,
                q_wp_prev,
                q_wc_prev,
                is_parent_body,
                k_ang,
                P_ang,
                wp.vec3(0.0),
                wp.vec3(0.0),
                ang_lambda,
                ang_C0,
                ang_alpha,
                kd_ang,
                dt,
            )
            has_cached = True
        else:
            t_ang = wp.vec3(0.0)
            Haa_ang = wp.mat33(0.0)

        # Drive + limits on free angular DOF
        dof_idx = qd_start
        model_drive_ke = joint_target_ke[dof_idx]
        drive_kd = joint_target_kd[dof_idx]
        target_pos = joint_target_pos[dof_idx]
        target_vel = joint_target_vel[dof_idx]
        lim_lower = joint_limit_lower[dof_idx]
        lim_upper = joint_limit_upper[dof_idx]
        model_limit_ke = joint_limit_ke[dof_idx]
        lim_kd = joint_limit_kd[dof_idx]

        has_drive = model_drive_ke > 0.0 or drive_kd > 0.0
        has_limits = model_limit_ke > 0.0 and (lim_lower > -MAXVAL or lim_upper < MAXVAL)

        if has_drive or has_limits:
            inv_dt = 1.0 / dt

            if has_cached:
                kappa = kappa_cached
                J_world = J_world_cached
            else:
                kappa, J_world = compute_kappa_and_jacobian(q_wp, q_wc, q_wp_rest, q_wc_rest)

            theta = wp.dot(kappa, a)
            theta_abs = theta + joint_rest_angle[dof_idx]
            omega_p = quat_velocity(q_wp, q_wp_prev, dt)
            omega_c = quat_velocity(q_wc, q_wc_prev, dt)
            dkappa_dt = compute_kappa_dot_analytic(q_wp, q_wc, q_wp_rest, q_wc_rest, omega_p, omega_c, kappa)
            dtheta_dt = wp.dot(dkappa_dt, a)

            mode, err_pos = resolve_drive_limit_mode(theta_abs, target_pos, lim_lower, lim_upper, has_drive, has_limits)
            f_scalar = float(0.0)
            H_scalar = float(0.0)
            if mode == _DRIVE_LIMIT_MODE_LIMIT_LOWER or mode == _DRIVE_LIMIT_MODE_LIMIT_UPPER:
                lim_d = lim_kd * model_limit_ke
                f_scalar = model_limit_ke * err_pos + lim_d * dtheta_dt
                H_scalar = model_limit_ke + lim_d * inv_dt
            elif mode == _DRIVE_LIMIT_MODE_DRIVE:
                drive_d = drive_kd * model_drive_ke
                vel_err = dtheta_dt - target_vel
                f_scalar = model_drive_ke * err_pos + drive_d * vel_err
                H_scalar = model_drive_ke + drive_d * inv_dt

            if H_scalar > 0.0:
                tau_drive, Haa_drive = apply_angular_drive_limit_torque(a, J_world, is_parent_body, f_scalar, H_scalar)
                t_ang = t_ang + tau_drive
                Haa_ang = Haa_ang + Haa_drive

        return f_lin, t_lin + t_ang, Hll_lin, Hal_lin, Haa_lin + Haa_ang

    elif jt == JointType.PRISMATIC:
        qd_start = joint_qd_start[joint_index]
        axis_local = joint_axis[qd_start]
        P_lin, P_ang = build_joint_projectors(jt, joint_axis, qd_start, 1, 0, q_wp)

        k_lin = joint_penalty_k[c_start + 0]
        kd_lin = joint_penalty_kd[c_start + 0]
        if k_lin > 0.0:
            f_lin, t_lin, Hll_lin, Hal_lin, Haa_lin = evaluate_linear_constraint_force_hessian(
                X_wp,
                X_wc,
                X_wp_prev,
                X_wc_prev,
                parent_pose,
                child_pose,
                parent_com,
                child_com,
                is_parent_body,
                k_lin,
                P_lin,
                lin_lambda,
                lin_C0,
                lin_alpha,
                kd_lin,
                dt,
            )
        else:
            f_lin = wp.vec3(0.0)
            t_lin = wp.vec3(0.0)
            Hll_lin = wp.mat33(0.0)
            Hal_lin = wp.mat33(0.0)
            Haa_lin = wp.mat33(0.0)

        k_ang = joint_penalty_k[c_start + 1]
        kd_ang = joint_penalty_kd[c_start + 1]
        if k_ang > 0.0:
            t_ang, Haa_ang, _ang_kappa, _ang_J = evaluate_angular_constraint_force_hessian(
                q_wp,
                q_wc,
                q_wp_rest,
                q_wc_rest,
                q_wp_prev,
                q_wc_prev,
                is_parent_body,
                k_ang,
                P_ang,
                wp.vec3(0.0),
                wp.vec3(0.0),
                ang_lambda,
                ang_C0,
                ang_alpha,
                kd_ang,
                dt,
            )
        else:
            t_ang = wp.vec3(0.0)
            Haa_ang = wp.mat33(0.0)

        # Drive + limits on free linear DOF
        dof_idx = qd_start
        model_drive_ke = joint_target_ke[dof_idx]
        drive_kd = joint_target_kd[dof_idx]
        target_pos = joint_target_pos[dof_idx]
        target_vel = joint_target_vel[dof_idx]
        lim_lower = joint_limit_lower[dof_idx]
        lim_upper = joint_limit_upper[dof_idx]
        model_limit_ke = joint_limit_ke[dof_idx]
        lim_kd = joint_limit_kd[dof_idx]

        has_drive = model_drive_ke > 0.0 or drive_kd > 0.0
        has_limits = model_limit_ke > 0.0 and (lim_lower > -MAXVAL or lim_upper < MAXVAL)

        if has_drive or has_limits:
            inv_dt = 1.0 / dt

            x_p = wp.transform_get_translation(X_wp)
            x_c = wp.transform_get_translation(X_wc)
            C_vec = x_c - x_p
            axis_w = wp.normalize(wp.quat_rotate(q_wp, axis_local))

            d_along = wp.dot(C_vec, axis_w)
            x_p_prev = wp.transform_get_translation(X_wp_prev)
            x_c_prev = wp.transform_get_translation(X_wc_prev)
            C_vec_prev = x_c_prev - x_p_prev
            dC_dt = (C_vec - C_vec_prev) * inv_dt
            dd_dt = wp.dot(dC_dt, axis_w)

            mode, err_pos = resolve_drive_limit_mode(d_along, target_pos, lim_lower, lim_upper, has_drive, has_limits)
            f_scalar = float(0.0)
            H_scalar = float(0.0)
            if mode == _DRIVE_LIMIT_MODE_LIMIT_LOWER or mode == _DRIVE_LIMIT_MODE_LIMIT_UPPER:
                lim_d = lim_kd * model_limit_ke
                f_scalar = model_limit_ke * err_pos + lim_d * dd_dt
                H_scalar = model_limit_ke + lim_d * inv_dt
            elif mode == _DRIVE_LIMIT_MODE_DRIVE:
                drive_d = drive_kd * model_drive_ke
                vel_err = dd_dt - target_vel
                f_scalar = model_drive_ke * err_pos + drive_d * vel_err
                H_scalar = model_drive_ke + drive_d * inv_dt

            if H_scalar > 0.0:
                if is_parent_body:
                    com_w = wp.transform_point(parent_pose, parent_com)
                    r = x_p - com_w
                else:
                    com_w = wp.transform_point(child_pose, child_com)
                    r = x_c - com_w

                force_drive, torque_drive, Hll_drive, Hal_drive, Haa_drive = apply_linear_drive_limit_force(
                    axis_w, r, is_parent_body, f_scalar, H_scalar
                )

                f_lin = f_lin + force_drive
                t_lin = t_lin + torque_drive
                Hll_lin = Hll_lin + Hll_drive
                Hal_lin = Hal_lin + Hal_drive
                Haa_lin = Haa_lin + Haa_drive

        return f_lin, t_lin + t_ang, Hll_lin, Hal_lin, Haa_lin + Haa_ang

    elif jt == JointType.D6:
        lin_count = joint_dof_dim[joint_index, 0]
        ang_count = joint_dof_dim[joint_index, 1]
        qd_start = joint_qd_start[joint_index]

        P_lin, P_ang = build_joint_projectors(
            jt,
            joint_axis,
            qd_start,
            lin_count,
            ang_count,
            q_wp,
        )

        total_force = wp.vec3(0.0)
        total_torque = wp.vec3(0.0)
        total_H_ll = wp.mat33(0.0)
        total_H_al = wp.mat33(0.0)
        total_H_aa = wp.mat33(0.0)

        # Linear constraint (constrained when lin_count < 3)
        k_lin = joint_penalty_k[c_start + 0]
        kd_lin = joint_penalty_kd[c_start + 0]

        if lin_count < 3 and k_lin > 0.0:
            f_l, t_l, Hll_l, Hal_l, Haa_l = evaluate_linear_constraint_force_hessian(
                X_wp,
                X_wc,
                X_wp_prev,
                X_wc_prev,
                parent_pose,
                child_pose,
                parent_com,
                child_com,
                is_parent_body,
                k_lin,
                P_lin,
                lin_lambda,
                lin_C0,
                lin_alpha,
                kd_lin,
                dt,
            )
            total_force = total_force + f_l
            total_torque = total_torque + t_l
            total_H_ll = total_H_ll + Hll_l
            total_H_al = total_H_al + Hal_l
            total_H_aa = total_H_aa + Haa_l

        # Angular constraint (constrained when ang_count < 3)
        k_ang = joint_penalty_k[c_start + 1]
        kd_ang = joint_penalty_kd[c_start + 1]

        kappa_cached = wp.vec3(0.0)
        J_world_cached = wp.mat33(0.0)
        has_cached = False

        if ang_count < 3 and k_ang > 0.0:
            t_ang, Haa_ang, kappa_cached, J_world_cached = evaluate_angular_constraint_force_hessian(
                q_wp,
                q_wc,
                q_wp_rest,
                q_wc_rest,
                q_wp_prev,
                q_wc_prev,
                is_parent_body,
                k_ang,
                P_ang,
                wp.vec3(0.0),
                wp.vec3(0.0),
                ang_lambda,
                ang_C0,
                ang_alpha,
                kd_ang,
                dt,
            )
            has_cached = True

            total_torque = total_torque + t_ang
            total_H_aa = total_H_aa + Haa_ang

        # Linear drives/limits (per free linear DOF)
        if lin_count > 0:
            x_p = wp.transform_get_translation(X_wp)
            x_c = wp.transform_get_translation(X_wc)
            C_vec = x_c - x_p
            q_wp_rot = q_wp
            x_p_prev = wp.transform_get_translation(X_wp_prev)
            x_c_prev = wp.transform_get_translation(X_wc_prev)
            C_vec_prev = x_c_prev - x_p_prev
            inv_dt = 1.0 / dt
            dC_dt = (C_vec - C_vec_prev) * inv_dt

            if is_parent_body:
                com_w = wp.transform_point(parent_pose, parent_com)
                r_drive = x_p - com_w
            else:
                com_w = wp.transform_point(child_pose, child_com)
                r_drive = x_c - com_w

            for li in range(3):
                if li < lin_count:
                    dof_idx = qd_start + li
                    model_drive_ke = joint_target_ke[dof_idx]
                    drive_kd = joint_target_kd[dof_idx]
                    target_pos = joint_target_pos[dof_idx]
                    target_vel = joint_target_vel[dof_idx]
                    lim_lower = joint_limit_lower[dof_idx]
                    lim_upper = joint_limit_upper[dof_idx]
                    model_limit_ke = joint_limit_ke[dof_idx]
                    lim_kd = joint_limit_kd[dof_idx]

                    has_drive = model_drive_ke > 0.0 or drive_kd > 0.0
                    has_limits = model_limit_ke > 0.0 and (lim_lower > -MAXVAL or lim_upper < MAXVAL)

                    if has_drive or has_limits:
                        axis_w = wp.normalize(wp.quat_rotate(q_wp_rot, joint_axis[dof_idx]))
                        d_along = wp.dot(C_vec, axis_w)
                        dd_dt = wp.dot(dC_dt, axis_w)

                        mode, err_pos = resolve_drive_limit_mode(
                            d_along, target_pos, lim_lower, lim_upper, has_drive, has_limits
                        )
                        f_scalar = float(0.0)
                        H_scalar = float(0.0)
                        if mode == _DRIVE_LIMIT_MODE_LIMIT_LOWER or mode == _DRIVE_LIMIT_MODE_LIMIT_UPPER:
                            lim_d = lim_kd * model_limit_ke
                            f_scalar = model_limit_ke * err_pos + lim_d * dd_dt
                            H_scalar = model_limit_ke + lim_d * inv_dt
                        elif mode == _DRIVE_LIMIT_MODE_DRIVE:
                            drive_d = drive_kd * model_drive_ke
                            vel_err = dd_dt - target_vel
                            f_scalar = model_drive_ke * err_pos + drive_d * vel_err
                            H_scalar = model_drive_ke + drive_d * inv_dt

                        if H_scalar > 0.0:
                            force_drive, torque_drive, Hll_drive, Hal_drive, Haa_drive = apply_linear_drive_limit_force(
                                axis_w, r_drive, is_parent_body, f_scalar, H_scalar
                            )

                            total_force = total_force + force_drive
                            total_torque = total_torque + torque_drive
                            total_H_ll = total_H_ll + Hll_drive
                            total_H_al = total_H_al + Hal_drive
                            total_H_aa = total_H_aa + Haa_drive

        # Angular drives/limits (per free angular DOF)
        if ang_count > 0:
            inv_dt = 1.0 / dt

            if has_cached:
                kappa = kappa_cached
                J_world = J_world_cached
            else:
                kappa, J_world = compute_kappa_and_jacobian(q_wp, q_wc, q_wp_rest, q_wc_rest)

            omega_p = quat_velocity(q_wp, q_wp_prev, dt)
            omega_c = quat_velocity(q_wc, q_wc_prev, dt)
            dkappa_dt = compute_kappa_dot_analytic(q_wp, q_wc, q_wp_rest, q_wc_rest, omega_p, omega_c, kappa)

            for ai in range(3):
                if ai < ang_count:
                    dof_idx = qd_start + lin_count + ai
                    model_drive_ke = joint_target_ke[dof_idx]
                    drive_kd = joint_target_kd[dof_idx]
                    target_pos = joint_target_pos[dof_idx]
                    target_vel = joint_target_vel[dof_idx]
                    lim_lower = joint_limit_lower[dof_idx]
                    lim_upper = joint_limit_upper[dof_idx]
                    model_limit_ke = joint_limit_ke[dof_idx]
                    lim_kd = joint_limit_kd[dof_idx]

                    has_drive = model_drive_ke > 0.0 or drive_kd > 0.0
                    has_limits = model_limit_ke > 0.0 and (lim_lower > -MAXVAL or lim_upper < MAXVAL)

                    if has_drive or has_limits:
                        a = wp.normalize(joint_axis[dof_idx])
                        theta = wp.dot(kappa, a)
                        theta_abs = theta + joint_rest_angle[dof_idx]
                        dtheta_dt = wp.dot(dkappa_dt, a)

                        mode, err_pos = resolve_drive_limit_mode(
                            theta_abs, target_pos, lim_lower, lim_upper, has_drive, has_limits
                        )
                        f_scalar = float(0.0)
                        H_scalar = float(0.0)
                        if mode == _DRIVE_LIMIT_MODE_LIMIT_LOWER or mode == _DRIVE_LIMIT_MODE_LIMIT_UPPER:
                            lim_d = lim_kd * model_limit_ke
                            f_scalar = model_limit_ke * err_pos + lim_d * dtheta_dt
                            H_scalar = model_limit_ke + lim_d * inv_dt
                        elif mode == _DRIVE_LIMIT_MODE_DRIVE:
                            drive_d = drive_kd * model_drive_ke
                            vel_err = dtheta_dt - target_vel
                            f_scalar = model_drive_ke * err_pos + drive_d * vel_err
                            H_scalar = model_drive_ke + drive_d * inv_dt

                        if H_scalar > 0.0:
                            tau_drive, Haa_drive = apply_angular_drive_limit_torque(
                                a, J_world, is_parent_body, f_scalar, H_scalar
                            )
                            total_torque = total_torque + tau_drive
                            total_H_aa = total_H_aa + Haa_drive

        return total_force, total_torque, total_H_ll, total_H_al, total_H_aa

    return wp.vec3(0.0), wp.vec3(0.0), wp.mat33(0.0), wp.mat33(0.0), wp.mat33(0.0)


# -----------------------------
# Utility kernels
# -----------------------------
@wp.kernel
def _count_num_adjacent_joints(
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
def _fill_adjacent_joints(
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
    gravity: wp.array(dtype=wp.vec3),
    body_world: wp.array(dtype=wp.int32),
    body_f: wp.array(dtype=wp.spatial_vector),
    body_com: wp.array(dtype=wp.vec3),
    body_inertia: wp.array(dtype=wp.mat33),
    body_inv_mass: wp.array(dtype=float),
    body_inv_inertia: wp.array(dtype=wp.mat33),
    body_q: wp.array(dtype=wp.transform),
    body_qd: wp.array(dtype=wp.spatial_vector),
    body_inertia_q: wp.array(dtype=wp.transform),
    body_q_prev: wp.array(dtype=wp.transform),
):
    """
    Forward integration step for rigid bodies in the AVBD/VBD solver.

    Snapshots ``body_q_prev`` for dynamic bodies only. Kinematic bodies keep
    the previous step's pose so contact friction sees correct velocity.

    Args:
        dt: Time step [s].
        gravity: Gravity vector array (world frame).
        body_world: World index for each body.
        body_f: External forces on bodies (spatial wrenches, world frame).
        body_com: Centers of mass (local body frame).
        body_inertia: Inertia tensors (local body frame).
        body_inv_mass: Inverse masses (0 for kinematic bodies).
        body_inv_inertia: Inverse inertia tensors (local body frame).
        body_q: Body transforms (input: start-of-step pose, output: integrated pose).
        body_qd: Body velocities (input: start-of-step velocity, output: integrated velocity).
        body_inertia_q: Inertial target body transforms for the AVBD solve (output).
        body_q_prev: Previous body transforms (output, dynamic bodies only).
    """
    tid = wp.tid()

    q_current = body_q[tid]

    # Early exit for kinematic bodies (inv_mass == 0).
    # Do not snapshot body_q_prev here: kinematic bodies need body_q_prev from previous step.
    inv_m = body_inv_mass[tid]
    if inv_m == 0.0:
        body_inertia_q[tid] = q_current
        return

    # Snapshot current pose as previous before integration (dynamic bodies only).
    body_q_prev[tid] = q_current

    # Read body state (only for dynamic bodies)
    qd_current = body_qd[tid]
    f_current = body_f[tid]
    com_local = body_com[tid]
    I_local = body_inertia[tid]
    inv_I = body_inv_inertia[tid]
    world_idx = body_world[tid]
    world_g = gravity[wp.max(world_idx, 0)]

    # Integrate rigid body motion (semi-implicit Euler, no angular damping)
    q_new, qd_new = integrate_rigid_body(
        q_current,
        qd_current,
        f_current,
        com_local,
        I_local,
        inv_m,
        inv_I,
        world_g,
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
    body_contact_overflow_max: wp.array(dtype=wp.int32),
):
    """
    Build per-body contact lists for body-centric per-color contact evaluation.
    Tracks overflow into body_contact_overflow_max for diagnostics.
    """
    t_id = wp.tid()
    if t_id >= rigid_contact_count[0]:
        return

    s0 = rigid_contact_shape0[t_id]
    s1 = rigid_contact_shape1[t_id]
    b0 = shape_body[s0] if s0 >= 0 else -1
    b1 = shape_body[s1] if s1 >= 0 else -1

    if b0 >= 0:
        idx = wp.atomic_add(body_contact_counts, b0, 1)
        if idx < body_contact_buffer_pre_alloc:
            body_contact_indices[b0 * body_contact_buffer_pre_alloc + idx] = t_id
        else:
            wp.atomic_max(body_contact_overflow_max, 0, idx + 1)

    if b1 >= 0:
        idx = wp.atomic_add(body_contact_counts, b1, 1)
        if idx < body_contact_buffer_pre_alloc:
            body_contact_indices[b1 * body_contact_buffer_pre_alloc + idx] = t_id
        else:
            wp.atomic_max(body_contact_overflow_max, 0, idx + 1)


@wp.kernel
def build_body_particle_contact_lists(
    body_particle_contact_count: wp.array(dtype=int),
    body_particle_contact_shape: wp.array(dtype=int),
    shape_body: wp.array(dtype=wp.int32),
    body_particle_contact_buffer_pre_alloc: int,
    body_particle_contact_counts: wp.array(dtype=wp.int32),
    body_particle_contact_indices: wp.array(dtype=wp.int32),
    body_particle_contact_overflow_max: wp.array(dtype=wp.int32),
):
    """
    Build per-body contact lists for body-particle contacts.
    Tracks overflow into body_particle_contact_overflow_max for diagnostics.
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
    else:
        wp.atomic_max(body_particle_contact_overflow_max, 0, idx + 1)


@wp.kernel
def check_contact_overflow(
    overflow_max: wp.array(dtype=wp.int32),
    buffer_size: int,
    contact_type: int,
):
    """Print a warning if per-body contact buffer overflowed. Launched with dim=1."""
    omax = overflow_max[0]
    if omax > buffer_size:
        if contact_type == 0:
            wp.printf(
                "Warning: Per-body rigid contact buffer overflowed %d > %d.\n",
                omax,
                buffer_size,
            )
        else:
            wp.printf(
                "Warning: Per-body particle contact buffer overflowed %d > %d.\n",
                omax,
                buffer_size,
            )


@wp.kernel
def init_joint_avbd(
    joint_enabled: wp.array(dtype=bool),
    joint_parent: wp.array(dtype=int),
    joint_child: wp.array(dtype=int),
    joint_X_p: wp.array(dtype=wp.transform),
    joint_X_c: wp.array(dtype=wp.transform),
    body_q: wp.array(dtype=wp.transform),
    body_q_rest: wp.array(dtype=wp.transform),
    joint_constraint_start: wp.array(dtype=wp.int32),
    joint_constraint_dim: wp.array(dtype=wp.int32),
    joint_is_hard: wp.array(dtype=wp.int32),
    gamma: float,
    joint_C0_lin: wp.array(dtype=wp.vec3),
    joint_C0_ang: wp.array(dtype=wp.vec3),
    joint_lambda_lin: wp.array(dtype=wp.vec3),
    joint_lambda_ang: wp.array(dtype=wp.vec3),
):
    """Per-step joint AVBD maintenance: C0 snapshot + lambda decay.

    Sole owner of all joint decay. Runs every step regardless of refresh.
    """
    j = wp.tid()
    c_start = int(joint_constraint_start[j])
    c_dim = int(joint_constraint_dim[j])

    child = joint_child[j]
    if not joint_enabled[j] or c_dim == 0 or child < 0:
        joint_C0_lin[j] = wp.vec3(0.0)
        joint_C0_ang[j] = wp.vec3(0.0)
        joint_lambda_lin[j] = wp.vec3(0.0)
        joint_lambda_ang[j] = wp.vec3(0.0)
        return

    lin_hard = joint_is_hard[c_start]
    ang_hard = 0
    if c_dim > 1:
        ang_hard = joint_is_hard[c_start + 1]

    if lin_hard == 1 or ang_hard == 1:
        parent = joint_parent[j]
        if parent >= 0:
            X_wp = body_q[parent] * joint_X_p[j]
        else:
            X_wp = joint_X_p[j]
        X_wc = body_q[child] * joint_X_c[j]

        if lin_hard == 1:
            x_p = wp.transform_get_translation(X_wp)
            x_c = wp.transform_get_translation(X_wc)
            joint_C0_lin[j] = x_c - x_p
            joint_lambda_lin[j] = joint_lambda_lin[j] * gamma
        else:
            joint_C0_lin[j] = wp.vec3(0.0)
            joint_lambda_lin[j] = wp.vec3(0.0)

        if ang_hard == 1:
            if parent >= 0:
                X_wp_rest = body_q_rest[parent] * joint_X_p[j]
            else:
                X_wp_rest = joint_X_p[j]
            X_wc_rest = body_q_rest[child] * joint_X_c[j]
            q_wp = wp.transform_get_rotation(X_wp)
            q_wc = wp.transform_get_rotation(X_wc)
            q_wp_rest = wp.transform_get_rotation(X_wp_rest)
            q_wc_rest = wp.transform_get_rotation(X_wc_rest)
            joint_C0_ang[j] = cable_get_kappa(q_wp, q_wc, q_wp_rest, q_wc_rest)
            joint_lambda_ang[j] = joint_lambda_ang[j] * gamma
        else:
            joint_C0_ang[j] = wp.vec3(0.0)
            joint_lambda_ang[j] = wp.vec3(0.0)
    else:
        joint_C0_lin[j] = wp.vec3(0.0)
        joint_C0_ang[j] = wp.vec3(0.0)
        joint_lambda_lin[j] = wp.vec3(0.0)
        joint_lambda_ang[j] = wp.vec3(0.0)


@wp.kernel
def init_body_body_contacts(
    rigid_contact_count: wp.array(dtype=int),
    rigid_contact_shape0: wp.array(dtype=int),
    rigid_contact_shape1: wp.array(dtype=int),
    shape_material_ke: wp.array(dtype=float),
    shape_material_kd: wp.array(dtype=float),
    shape_material_mu: wp.array(dtype=float),
    # Outputs
    contact_penalty_k: wp.array(dtype=float),
    contact_material_kd: wp.array(dtype=float),
    contact_material_mu: wp.array(dtype=float),
):
    """
    Cold-start contact penalties and cache material properties.

    Computes averaged material properties for each rigid contact and sets the
    contact penalty stiffness to the averaged material stiffness.
    """
    i = wp.tid()
    if i >= rigid_contact_count[0]:
        return

    shape_id_0 = rigid_contact_shape0[i]
    shape_id_1 = rigid_contact_shape1[i]

    avg_ke, avg_kd, avg_mu = _average_contact_material(
        shape_material_ke[shape_id_0],
        shape_material_kd[shape_id_0],
        shape_material_mu[shape_id_0],
        shape_material_ke[shape_id_1],
        shape_material_kd[shape_id_1],
        shape_material_mu[shape_id_1],
    )

    contact_material_kd[i] = avg_kd
    contact_material_mu[i] = avg_mu

    contact_penalty_k[i] = avg_ke


@wp.kernel
def init_contact_avbd(
    # Dimensioning
    rigid_contact_count: wp.array(dtype=int),
    # Constraint data
    rigid_contact_shape0: wp.array(dtype=int),
    rigid_contact_shape1: wp.array(dtype=int),
    rigid_contact_point0: wp.array(dtype=wp.vec3),
    rigid_contact_point1: wp.array(dtype=wp.vec3),
    rigid_contact_normal: wp.array(dtype=wp.vec3),
    rigid_contact_margin0: wp.array(dtype=float),
    rigid_contact_margin1: wp.array(dtype=float),
    # Material
    shape_material_ke: wp.array(dtype=float),
    shape_material_kd: wp.array(dtype=float),
    shape_material_mu: wp.array(dtype=float),
    # Geometry
    shape_body: wp.array(dtype=wp.int32),
    body_q_prev: wp.array(dtype=wp.transform),
    hard_contacts: int,
    # Cross-step state
    ht_keys: wp.array(dtype=wp.uint64),
    history_point0: wp.array(dtype=wp.vec3),
    history_point1: wp.array(dtype=wp.vec3),
    history_lambda: wp.array(dtype=wp.vec3),
    history_normal: wp.array(dtype=wp.vec3),
    history_stick_flag: wp.array(dtype=wp.int32),
    # Scalar parameters
    cell_size_inv: float,
    match_tolerance_sq: float,
    # Outputs
    contact_penalty_k: wp.array(dtype=float),
    contact_lambda: wp.array(dtype=wp.vec3),
    contact_C0: wp.array(dtype=wp.vec3),
    contact_material_kd: wp.array(dtype=float),
    contact_material_mu: wp.array(dtype=float),
):
    """Warmstart body-body contacts using hash table-based contact history.

    Matched contacts copy lambda from history (no decay — decay is handled by
    step_contact_C0_lambda). For hard contacts, lambda is rotated from the old
    contact frame to the new one. C0 is computed as a vec3 constraint from
    body_q_prev.
    """
    i = wp.tid()
    if i >= rigid_contact_count[0]:
        return

    s0 = rigid_contact_shape0[i]
    s1 = rigid_contact_shape1[i]

    avg_ke, avg_kd, avg_mu = _average_contact_material(
        shape_material_ke[s0],
        shape_material_kd[s0],
        shape_material_mu[s0],
        shape_material_ke[s1],
        shape_material_kd[s1],
        shape_material_mu[s1],
    )
    contact_material_kd[i] = avg_kd
    contact_material_mu[i] = avg_mu

    p0 = rigid_contact_point0[i]
    pair_bits = _pack_shape_pair_bits(s0, s1)
    base_cell = _compute_spatial_cell(p0, cell_size_inv)

    best_dist_sq = match_tolerance_sq
    best_slot = int(-1)

    center_key = _build_contact_history_key(pair_bits, base_cell)
    center_slot = hashtable_find(center_key, ht_keys)

    if center_slot >= 0:
        d = p0 - history_point0[center_slot]
        dist_sq = wp.dot(d, d)
        if dist_sq < best_dist_sq:
            best_dist_sq = dist_sq
            best_slot = center_slot

    if best_slot < 0:
        for dx in range(-1, 2):
            for dy in range(-1, 2):
                for dz in range(-1, 2):
                    if dx == 0 and dy == 0 and dz == 0:
                        continue
                    cell = _offset_cell(base_cell, dx, dy, dz)
                    key = _build_contact_history_key(pair_bits, cell)
                    slot = hashtable_find(key, ht_keys)
                    if slot >= 0:
                        d = p0 - history_point0[slot]
                        dist_sq = wp.dot(d, d)
                        if dist_sq < best_dist_sq:
                            best_dist_sq = dist_sq
                            best_slot = slot

    contact_penalty_k[i] = avg_ke

    if best_slot >= 0:
        lam_hist = history_lambda[best_slot]
        if hard_contacts == 1:
            n_old = history_normal[best_slot]
            n_new = rigid_contact_normal[i]
            lam_n = wp.dot(lam_hist, n_old)
            lam_t_old = lam_hist - n_old * lam_n
            lam_t_new = lam_t_old - n_new * wp.dot(lam_t_old, n_new)
            contact_lambda[i] = n_new * lam_n + lam_t_new
        else:
            contact_lambda[i] = lam_hist

        if hard_contacts == 1 and history_stick_flag[best_slot] == 1:
            rigid_contact_point0[i] = history_point0[best_slot]
            rigid_contact_point1[i] = history_point1[best_slot]
    else:
        contact_lambda[i] = wp.vec3(0.0)

    if hard_contacts == 1:
        b0 = shape_body[s0] if s0 >= 0 else -1
        b1 = shape_body[s1] if s1 >= 0 else -1
        n = rigid_contact_normal[i]
        cp0 = wp.transform_point(body_q_prev[b0], rigid_contact_point0[i]) if b0 >= 0 else rigid_contact_point0[i]
        cp1 = wp.transform_point(body_q_prev[b1], rigid_contact_point1[i]) if b1 >= 0 else rigid_contact_point1[i]
        thickness = rigid_contact_margin0[i] + rigid_contact_margin1[i]
        d_vec = cp1 - cp0
        contact_C0[i] = n * thickness - d_vec
    else:
        contact_C0[i] = wp.vec3(0.0)


@wp.kernel
def snapshot_contact_history(
    rigid_contact_count: wp.array(dtype=int),
    rigid_contact_shape0: wp.array(dtype=int),
    rigid_contact_shape1: wp.array(dtype=int),
    rigid_contact_point0: wp.array(dtype=wp.vec3),
    rigid_contact_point1: wp.array(dtype=wp.vec3),
    rigid_contact_normal: wp.array(dtype=wp.vec3),
    contact_lambda: wp.array(dtype=wp.vec3),
    contact_stick_flag: wp.array(dtype=wp.int32),
    # Hash table
    ht_keys: wp.array(dtype=wp.uint64),
    ht_active_slots: wp.array(dtype=wp.int32),
    cell_size_inv: float,
    # Outputs
    history_point0: wp.array(dtype=wp.vec3),
    history_point1: wp.array(dtype=wp.vec3),
    history_lambda: wp.array(dtype=wp.vec3),
    history_normal: wp.array(dtype=wp.vec3),
    history_stick_flag: wp.array(dtype=wp.int32),
    history_age: wp.array(dtype=int),
    cached_slots: wp.array(dtype=int),
):
    """Snapshot converged contact state into the history hash table (post-solve).

    Stores lambda, normal, stick state, and contact point for warmstarting.
    Also caches the resolved slot index per contact so that subsequent
    non-rebuild steps can update history without re-hashing.
    """
    i = wp.tid()
    if i >= rigid_contact_count[0]:
        return

    s0 = rigid_contact_shape0[i]
    s1 = rigid_contact_shape1[i]
    p0 = rigid_contact_point0[i]

    pair_bits = _pack_shape_pair_bits(s0, s1)
    cell = _compute_spatial_cell(p0, cell_size_inv)
    key = _build_contact_history_key(pair_bits, cell)

    slot = hashtable_find_or_insert(key, ht_keys, ht_active_slots)
    cached_slots[i] = slot
    if slot >= 0:
        history_point0[slot] = p0
        history_point1[slot] = rigid_contact_point1[i]
        history_lambda[slot] = contact_lambda[i]
        history_normal[slot] = rigid_contact_normal[i]
        history_stick_flag[slot] = contact_stick_flag[i]
        history_age[slot] = 0


@wp.kernel
def evict_contact_history(
    ht_keys: wp.array(dtype=wp.uint64),
    history_age: wp.array(dtype=int),
    max_age: int,
):
    """Increment age for all occupied history slots and evict stale entries.

    Scans the full hash table capacity (not active_slots) to avoid unbounded
    active-list growth and ensure all persisted entries are properly aged.

    Note: snapshot_contact_history sets age = 0, and this kernel increments
    before comparing, so effective retention is max_age + 1 steps.
    """
    idx = wp.tid()
    if ht_keys[idx] == HASHTABLE_EMPTY_KEY:
        return
    age = history_age[idx] + 1
    if age > max_age:
        ht_keys[idx] = HASHTABLE_EMPTY_KEY
    else:
        history_age[idx] = age


@wp.kernel
def snapshot_contact_history_light(
    rigid_contact_count: wp.array(dtype=int),
    cached_slots: wp.array(dtype=int),
    contact_lambda: wp.array(dtype=wp.vec3),
    contact_stick_flag: wp.array(dtype=wp.int32),
    # Outputs (hash table history arrays, indexed by cached slot)
    history_lambda: wp.array(dtype=wp.vec3),
    history_stick_flag: wp.array(dtype=wp.int32),
):
    """Write latest lambda/stick state to the hash table at pre-cached slot positions.

    Runs on non-rebuild steps to keep the hash table fresh without hashing
    or probing.  Slots were cached by the previous full snapshot.
    """
    i = wp.tid()
    if i >= rigid_contact_count[0]:
        return
    slot = cached_slots[i]
    if slot >= 0:
        history_lambda[slot] = contact_lambda[i]
        history_stick_flag[slot] = contact_stick_flag[i]


@wp.kernel
def step_contact_C0_lambda(
    rigid_contact_count: wp.array(dtype=int),
    rigid_contact_shape0: wp.array(dtype=int),
    rigid_contact_shape1: wp.array(dtype=int),
    rigid_contact_point0: wp.array(dtype=wp.vec3),
    rigid_contact_point1: wp.array(dtype=wp.vec3),
    rigid_contact_normal: wp.array(dtype=wp.vec3),
    rigid_contact_margin0: wp.array(dtype=float),
    rigid_contact_margin1: wp.array(dtype=float),
    shape_body: wp.array(dtype=int),
    body_q_prev: wp.array(dtype=wp.transform),
    hard_contacts: int,
    gamma: float,
    recompute_C0: int,
    # In/out
    contact_C0: wp.array(dtype=wp.vec3),
    contact_lambda: wp.array(dtype=wp.vec3),
):
    """Per-step lambda decay, plus optional C0 recompute.

    Runs every step. On refresh steps (recompute_C0=0), only applies lambda decay
    (C0 was already set by init_contact_avbd). On non-refresh steps
    (recompute_C0=1), also recomputes vec3 C0 from body_q_prev.
    """
    i = wp.tid()
    if i >= rigid_contact_count[0]:
        return

    contact_lambda[i] = contact_lambda[i] * gamma

    if recompute_C0 == 1 and hard_contacts == 1:
        s0 = rigid_contact_shape0[i]
        s1 = rigid_contact_shape1[i]
        b0 = shape_body[s0] if s0 >= 0 else -1
        b1 = shape_body[s1] if s1 >= 0 else -1
        p0 = rigid_contact_point0[i]
        p1 = rigid_contact_point1[i]
        n = rigid_contact_normal[i]
        cp0 = wp.transform_point(body_q_prev[b0], p0) if b0 >= 0 else p0
        cp1 = wp.transform_point(body_q_prev[b1], p1) if b1 >= 0 else p1
        thickness = rigid_contact_margin0[i] + rigid_contact_margin1[i]
        d = cp1 - cp0
        contact_C0[i] = n * thickness - d


@wp.kernel
def init_body_particle_contacts(
    body_particle_contact_count: wp.array(dtype=int),
    body_particle_contact_shape: wp.array(dtype=int),
    soft_contact_ke: float,
    soft_contact_kd: float,
    soft_contact_mu: float,
    shape_material_ke: wp.array(dtype=float),
    shape_material_kd: wp.array(dtype=float),
    shape_material_mu: wp.array(dtype=float),
    # Outputs
    body_particle_contact_penalty_k: wp.array(dtype=float),
    body_particle_contact_material_kd: wp.array(dtype=float),
    body_particle_contact_material_mu: wp.array(dtype=float),
):
    """
    Initialize body-particle (particle-rigid) contact penalties and cache material
    properties.

    The scalar inputs `soft_contact_ke/kd/mu` are the particle-side soft-contact
    material parameters (from `model.soft_contact_*`). For each body-particle
    contact, this kernel averages those particle-side values with the rigid
    shape's material parameters and sets the penalty to the averaged stiffness.
    """
    i = wp.tid()
    if i >= body_particle_contact_count[0]:
        return

    shape_idx = body_particle_contact_shape[i]

    avg_ke, avg_kd, avg_mu = _average_contact_material(
        soft_contact_ke,
        soft_contact_kd,
        soft_contact_mu,
        shape_material_ke[shape_idx],
        shape_material_kd[shape_idx],
        shape_material_mu[shape_idx],
    )

    body_particle_contact_material_kd[i] = avg_kd
    body_particle_contact_material_mu[i] = avg_mu

    body_particle_contact_penalty_k[i] = avg_ke


@wp.kernel
def compute_cable_dahl_parameters(
    # Inputs
    joint_type: wp.array(dtype=int),
    joint_enabled: wp.array(dtype=bool),
    joint_parent: wp.array(dtype=int),
    joint_child: wp.array(dtype=int),
    joint_X_p: wp.array(dtype=wp.transform),
    joint_X_c: wp.array(dtype=wp.transform),
    joint_constraint_start: wp.array(dtype=int),
    joint_penalty_k: wp.array(dtype=float),
    body_q: wp.array(dtype=wp.transform),
    body_q_rest: wp.array(dtype=wp.transform),
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

    if not joint_enabled[j]:
        joint_sigma_start[j] = wp.vec3(0.0)
        joint_C_fric[j] = wp.vec3(0.0)
        return

    # Only process cable joints
    if joint_type[j] != JointType.CABLE:
        joint_sigma_start[j] = wp.vec3(0.0)
        joint_C_fric[j] = wp.vec3(0.0)
        return

    parent = joint_parent[j]
    child = joint_child[j]

    # World-parent joints are valid; child body must exist.
    if child < 0:
        joint_sigma_start[j] = wp.vec3(0.0)
        joint_C_fric[j] = wp.vec3(0.0)
        return

    # Compute joint frames in world space (current and rest only)
    if parent >= 0:
        X_wp = body_q[parent] * joint_X_p[j]
        X_wp_rest = body_q_rest[parent] * joint_X_p[j]
    else:
        X_wp = joint_X_p[j]
        X_wp_rest = joint_X_p[j]

    X_wc = body_q[child] * joint_X_c[j]
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

    # Use the per-joint bend stiffness from the solver constraint array (constraint slot 1 for cables).
    c_start = joint_constraint_start[j]
    k_bend_target = joint_penalty_k[c_start + 1]

    # Friction envelope: sigma_max = k_bend_target * eps_max.

    sigma_max = k_bend_target * eps_max
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
    dt: float,
    color_group: wp.array(dtype=wp.int32),
    body_q_prev: wp.array(dtype=wp.transform),
    body_q: wp.array(dtype=wp.transform),
    body_com: wp.array(dtype=wp.vec3),
    body_inv_mass: wp.array(dtype=float),
    friction_epsilon: float,
    contact_penalty_k: wp.array(dtype=float),
    contact_material_kd: wp.array(dtype=float),
    contact_material_mu: wp.array(dtype=float),
    contact_lambda: wp.array(dtype=wp.vec3),
    contact_C0: wp.array(dtype=wp.vec3),
    avbd_alpha: float,
    hard_contacts: int,
    rigid_contact_count: wp.array(dtype=int),
    rigid_contact_shape0: wp.array(dtype=int),
    rigid_contact_shape1: wp.array(dtype=int),
    rigid_contact_point0: wp.array(dtype=wp.vec3),
    rigid_contact_point1: wp.array(dtype=wp.vec3),
    rigid_contact_normal: wp.array(dtype=wp.vec3),
    rigid_contact_margin0: wp.array(dtype=float),
    rigid_contact_margin1: wp.array(dtype=float),
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
    Per-body augmented-Lagrangian contact accumulation with _NUM_CONTACT_THREADS_PER_BODY strided threads.
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

    contact_count = rigid_contact_count[0]

    force_acc = wp.vec3(0.0)
    torque_acc = wp.vec3(0.0)
    h_ll_acc = wp.mat33(0.0)
    h_al_acc = wp.mat33(0.0)
    h_aa_acc = wp.mat33(0.0)

    i = thread_id_within_body
    while i < num_contacts:
        contact_idx = body_contact_indices[body_id * body_contact_buffer_pre_alloc + i]
        if contact_idx >= contact_count:
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
        contact_normal = rigid_contact_normal[contact_idx]
        cp0_world = wp.transform_point(body_q[b0], cp0_local) if b0 >= 0 else cp0_local
        cp1_world = wp.transform_point(body_q[b1], cp1_local) if b1 >= 0 else cp1_local
        thickness = rigid_contact_margin0[contact_idx] + rigid_contact_margin1[contact_idx]
        d = cp1_world - cp0_world
        C_n = thickness - wp.dot(contact_normal, d)

        lam_n = float(0.0)
        C_eff = C_n
        C_stab_t = wp.vec3(0.0)
        lam_vec = wp.vec3(0.0)
        k = contact_penalty_k[contact_idx]

        if hard_contacts == 1:
            lam_vec = contact_lambda[contact_idx]
            lam_n = wp.dot(lam_vec, contact_normal)
            C0_vec = contact_C0[contact_idx]
            C_vec = contact_normal * thickness - d
            C_stab = C_vec - avbd_alpha * C0_vec
            C_stab_n = wp.dot(C_stab, contact_normal)
            C_stab_t = C_stab - contact_normal * C_stab_n
            C_eff = C_stab_n

        if C_n <= _SMALL_LENGTH_EPS and lam_n <= 0.0:
            i += _NUM_CONTACT_THREADS_PER_BODY
            continue

        f_n_check = k * C_eff + lam_n
        if f_n_check <= 0.0 and lam_n <= 0.0:
            i += _NUM_CONTACT_THREADS_PER_BODY
            continue

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
            body_q_prev,
            body_com,
            cp0_local,
            cp1_local,
            contact_normal,
            C_eff,
            C_stab_t,
            k,
            k,
            contact_kd,
            lam_vec,
            contact_mu,
            friction_epsilon,
            hard_contacts,
            dt,
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
def compute_rigid_contact_forces(
    dt: float,
    # Contact data
    rigid_contact_count: wp.array(dtype=int),
    rigid_contact_shape0: wp.array(dtype=int),
    rigid_contact_shape1: wp.array(dtype=int),
    rigid_contact_point0: wp.array(dtype=wp.vec3),
    rigid_contact_point1: wp.array(dtype=wp.vec3),
    rigid_contact_normal: wp.array(dtype=wp.vec3),
    rigid_contact_margin0: wp.array(dtype=float),
    rigid_contact_margin1: wp.array(dtype=float),
    # Model/state
    shape_body: wp.array(dtype=wp.int32),
    body_q: wp.array(dtype=wp.transform),
    body_q_prev: wp.array(dtype=wp.transform),
    body_com: wp.array(dtype=wp.vec3),
    # Contact material properties (per-contact)
    contact_penalty_k: wp.array(dtype=float),
    contact_material_kd: wp.array(dtype=float),
    contact_material_mu: wp.array(dtype=float),
    contact_lambda: wp.array(dtype=wp.vec3),
    contact_C0: wp.array(dtype=wp.vec3),
    avbd_alpha: float,
    hard_contacts: int,
    friction_epsilon: float,
    # Outputs (length = rigid_contact_max)
    out_body0: wp.array(dtype=wp.int32),
    out_body1: wp.array(dtype=wp.int32),
    out_point0_world: wp.array(dtype=wp.vec3),
    out_point1_world: wp.array(dtype=wp.vec3),
    out_force_on_body1: wp.array(dtype=wp.vec3),
):
    """Compute per-contact augmented-Lagrangian forces in world space."""
    contact_idx = wp.tid()

    rc = rigid_contact_count[0]
    if contact_idx >= rc:
        # Fill sentinel values for inactive entries (useful when launching with rigid_contact_max)
        out_body0[contact_idx] = wp.int32(-1)
        out_body1[contact_idx] = wp.int32(-1)
        out_point0_world[contact_idx] = wp.vec3(0.0)
        out_point1_world[contact_idx] = wp.vec3(0.0)
        out_force_on_body1[contact_idx] = wp.vec3(0.0)
        return

    s0 = rigid_contact_shape0[contact_idx]
    s1 = rigid_contact_shape1[contact_idx]
    if s0 < 0 or s1 < 0:
        out_body0[contact_idx] = wp.int32(-1)
        out_body1[contact_idx] = wp.int32(-1)
        out_point0_world[contact_idx] = wp.vec3(0.0)
        out_point1_world[contact_idx] = wp.vec3(0.0)
        out_force_on_body1[contact_idx] = wp.vec3(0.0)
        return

    b0 = shape_body[s0]
    b1 = shape_body[s1]
    out_body0[contact_idx] = b0
    out_body1[contact_idx] = b1

    cp0_local = rigid_contact_point0[contact_idx]
    cp1_local = rigid_contact_point1[contact_idx]
    contact_normal = rigid_contact_normal[contact_idx]

    cp0_world = wp.transform_point(body_q[b0], cp0_local) if b0 >= 0 else cp0_local
    cp1_world = wp.transform_point(body_q[b1], cp1_local) if b1 >= 0 else cp1_local
    out_point0_world[contact_idx] = cp0_world
    out_point1_world[contact_idx] = cp1_world

    thickness = rigid_contact_margin0[contact_idx] + rigid_contact_margin1[contact_idx]
    d = cp1_world - cp0_world
    C_n = thickness - wp.dot(contact_normal, d)

    lam_n = float(0.0)
    C_eff = C_n
    C_stab_t = wp.vec3(0.0)
    lam_vec = wp.vec3(0.0)
    k = contact_penalty_k[contact_idx]

    if hard_contacts == 1:
        lam_vec = contact_lambda[contact_idx]
        lam_n = wp.dot(lam_vec, contact_normal)
        C0_vec = contact_C0[contact_idx]
        C_vec = contact_normal * thickness - d
        C_stab = C_vec - avbd_alpha * C0_vec
        C_stab_n = wp.dot(C_stab, contact_normal)
        C_stab_t = C_stab - contact_normal * C_stab_n
        C_eff = C_stab_n

    if C_n <= _SMALL_LENGTH_EPS and lam_n <= 0.0:
        out_force_on_body1[contact_idx] = wp.vec3(0.0)
        return

    contact_kd = contact_material_kd[contact_idx]
    contact_mu = contact_material_mu[contact_idx]

    (
        _force_0,
        _torque_0,
        _h_ll_0,
        _h_al_0,
        _h_aa_0,
        force_1,
        _torque_1,
        _h_ll_1,
        _h_al_1,
        _h_aa_1,
    ) = evaluate_rigid_contact_from_collision(
        int(b0),
        int(b1),
        body_q,
        body_q_prev,
        body_com,
        cp0_local,
        cp1_local,
        contact_normal,
        C_eff,
        C_stab_t,
        k,
        k,
        contact_kd,
        lam_vec,
        contact_mu,
        friction_epsilon,
        hard_contacts,
        dt,
    )

    out_force_on_body1[contact_idx] = force_1


@wp.kernel
def accumulate_body_particle_contacts_per_body(
    dt: float,
    color_group: wp.array(dtype=wp.int32),
    # Particle state
    particle_q: wp.array(dtype=wp.vec3),
    particle_q_prev: wp.array(dtype=wp.vec3),
    particle_radius: wp.array(dtype=float),
    # Rigid body state
    body_q_prev: wp.array(dtype=wp.transform),
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
         compute the force and Hessian on the particle using the effective body-particle penalty stiffness.
      2. Apply the equal-and-opposite reaction force, torque, and Hessian contributions
         to the rigid body.

    Notes:
      - Only dynamic bodies (inv_mass > 0) are updated.
      - Hessian contributions are accumulated into body_hessian_ll/al/aa.
      - Uses per-contact effective penalty/material parameters initialized once per step.
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

    max_contacts = body_particle_contact_count[0]

    X_wb = body_q[body_id]
    com_world = wp.transform_point(X_wb, body_com[body_id])

    force_acc = wp.vec3(0.0)
    torque_acc = wp.vec3(0.0)
    h_ll_acc = wp.mat33(0.0)
    h_al_acc = wp.mat33(0.0)
    h_aa_acc = wp.mat33(0.0)

    i = thread_id_within_body
    while i < num_contacts:
        contact_idx = body_particle_contact_indices[body_id * body_particle_contact_buffer_pre_alloc + i]
        if contact_idx >= max_contacts:
            i += _NUM_CONTACT_THREADS_PER_BODY
            continue

        particle_idx = body_particle_contact_particle[contact_idx]
        if particle_idx < 0:
            i += _NUM_CONTACT_THREADS_PER_BODY
            continue

        particle_pos = particle_q[particle_idx]
        cp_local = body_particle_contact_body_pos[contact_idx]
        cp_world = wp.transform_point(X_wb, cp_local)
        n = body_particle_contact_normal[contact_idx]
        radius = particle_radius[particle_idx]
        penetration_depth = -(wp.dot(n, particle_pos - cp_world) - radius)

        if penetration_depth <= 0.0:
            i += _NUM_CONTACT_THREADS_PER_BODY
            continue

        particle_prev_pos = particle_q_prev[particle_idx]

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
            body_q_prev,
            body_qd,
            body_com,
            body_particle_contact_shape,
            body_particle_contact_body_pos,
            body_particle_contact_body_vel,
            body_particle_contact_normal,
            dt,
        )

        f_body = -force_on_particle

        r = cp_world - com_world
        tau_body = wp.cross(r, f_body)

        K_total = hessian_particle
        r_skew = wp.skew(r)
        r_skew_T_K = wp.transpose(r_skew) * K_total

        force_acc += f_body
        torque_acc += tau_body
        h_ll_acc += K_total
        h_al_acc += -r_skew_T_K
        h_aa_acc += r_skew_T_K * r_skew

        i += _NUM_CONTACT_THREADS_PER_BODY

    wp.atomic_add(body_forces, body_id, force_acc)
    wp.atomic_add(body_torques, body_id, torque_acc)
    wp.atomic_add(body_hessian_ll, body_id, h_ll_acc)
    wp.atomic_add(body_hessian_al, body_id, h_al_acc)
    wp.atomic_add(body_hessian_aa, body_id, h_aa_acc)


@wp.kernel
def solve_rigid_body(
    dt: float,
    body_ids_in_color: wp.array(dtype=wp.int32),
    body_q: wp.array(dtype=wp.transform),
    body_q_prev: wp.array(dtype=wp.transform),
    body_q_rest: wp.array(dtype=wp.transform),
    body_mass: wp.array(dtype=float),
    body_inv_mass: wp.array(dtype=float),
    body_inertia: wp.array(dtype=wp.mat33),
    body_inertia_q: wp.array(dtype=wp.transform),
    body_com: wp.array(dtype=wp.vec3),
    adjacency: RigidForceElementAdjacencyInfo,
    # Joint data
    joint_type: wp.array(dtype=int),
    joint_enabled: wp.array(dtype=bool),
    joint_parent: wp.array(dtype=int),
    joint_child: wp.array(dtype=int),
    joint_X_p: wp.array(dtype=wp.transform),
    joint_X_c: wp.array(dtype=wp.transform),
    joint_axis: wp.array(dtype=wp.vec3),
    joint_qd_start: wp.array(dtype=int),
    joint_constraint_start: wp.array(dtype=int),
    # AVBD per-constraint penalty state (scalar constraints indexed via joint_constraint_start)
    joint_penalty_k: wp.array(dtype=float),
    joint_penalty_kd: wp.array(dtype=float),
    # Dahl hysteresis parameters (frozen for this timestep, component-wise vec3 per joint)
    joint_sigma_start: wp.array(dtype=wp.vec3),
    joint_C_fric: wp.array(dtype=wp.vec3),
    # Drive parameters (DOF-indexed via joint_qd_start)
    joint_target_ke: wp.array(dtype=float),
    joint_target_kd: wp.array(dtype=float),
    joint_target_pos: wp.array(dtype=float),
    joint_target_vel: wp.array(dtype=float),
    # Limit parameters (DOF-indexed via joint_qd_start)
    joint_limit_lower: wp.array(dtype=float),
    joint_limit_upper: wp.array(dtype=float),
    joint_limit_ke: wp.array(dtype=float),
    joint_limit_kd: wp.array(dtype=float),
    joint_lambda_lin: wp.array(dtype=wp.vec3),
    joint_lambda_ang: wp.array(dtype=wp.vec3),
    joint_C0_lin: wp.array(dtype=wp.vec3),
    joint_C0_ang: wp.array(dtype=wp.vec3),
    joint_is_hard: wp.array(dtype=wp.int32),
    avbd_alpha: float,
    joint_dof_dim: wp.array(dtype=int, ndim=2),
    joint_rest_angle: wp.array(dtype=float),
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
    contributions into a 6x6 block system:

        [ H_ll   H_al^T ]
        [ H_al   H_aa   ]

    and solving via Schur complement.
    Consistent with VBD particle solve pattern: inertia + external + constraint forces.

    Algorithm:
      1. Compute inertial forces/Hessians
      2. Accumulate external forces/Hessians from rigid contacts
      3. Accumulate joint forces/Hessians from adjacent joints
      4. Solve 6x6 block system via Schur complement: S = A - C*M^-1*C^T
      5. Update pose: rotation from angular increment, position from linear increment

    Args:
        dt: Time step.
        body_ids_in_color: Body indices in current color group (for parallel coloring).
        body_q_prev: Previous body transforms (for damping and friction).
        body_q_rest: Rest transforms (for joint targets).
        body_mass: Body masses.
        body_inv_mass: Inverse masses (0 for kinematic bodies).
        body_inertia: Inertia tensors (local body frame).
        body_inertia_q: Inertial target transforms (from forward integration).
        body_com: Center of mass offsets (local body frame).
        adjacency: Body-joint adjacency (CSR format).
        joint_*: Joint configuration arrays.
        joint_penalty_k: AVBD per-constraint penalty stiffness (one scalar per solver constraint component).
        joint_sigma_start: Dahl hysteresis state at start of step.
        joint_C_fric: Dahl friction configuration per joint.
        external_forces: External linear forces from rigid contacts.
        external_torques: External angular torques from rigid contacts.
        external_hessian_ll: Linear-linear Hessian block (3x3) from rigid contacts.
        external_hessian_al: Angular-linear coupling Hessian block (3x3) from rigid contacts.
        external_hessian_aa: Angular-angular Hessian block (3x3) from rigid contacts.
        body_q: Current body transforms (input).
        body_q_new: Updated body transforms (output) for the current solve sweep.

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
            body_q_prev,
            body_q_rest,
            body_com,
            joint_type,
            joint_enabled,
            joint_parent,
            joint_child,
            joint_X_p,
            joint_X_c,
            joint_axis,
            joint_qd_start,
            joint_constraint_start,
            joint_penalty_k,
            joint_penalty_kd,
            joint_sigma_start,
            joint_C_fric,
            joint_target_ke,
            joint_target_kd,
            joint_target_pos,
            joint_target_vel,
            joint_limit_lower,
            joint_limit_upper,
            joint_limit_ke,
            joint_limit_kd,
            joint_lambda_lin,
            joint_lambda_ang,
            joint_C0_lin,
            joint_C0_ang,
            joint_is_hard,
            avbd_alpha,
            joint_dof_dim,
            joint_rest_angle,
            dt,
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

    # Compute H_ll^{-1} * (H_al^T)
    C_r0 = wp.vec3(h_al[0, 0], h_al[0, 1], h_al[0, 2])
    C_r1 = wp.vec3(h_al[1, 0], h_al[1, 1], h_al[1, 2])
    C_r2 = wp.vec3(h_al[2, 0], h_al[2, 1], h_al[2, 2])

    X0 = chol33_solve(Lm_p, C_r0)
    X1 = chol33_solve(Lm_p, C_r1)
    X2 = chol33_solve(Lm_p, C_r2)

    # Columns are the solved vectors X0, X1, X2
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
def update_duals_joint(
    joint_type: wp.array(dtype=int),
    joint_enabled: wp.array(dtype=bool),
    joint_parent: wp.array(dtype=int),
    joint_child: wp.array(dtype=int),
    joint_X_p: wp.array(dtype=wp.transform),
    joint_X_c: wp.array(dtype=wp.transform),
    joint_axis: wp.array(dtype=wp.vec3),
    joint_qd_start: wp.array(dtype=int),
    joint_constraint_start: wp.array(dtype=int),
    body_q: wp.array(dtype=wp.transform),
    body_q_rest: wp.array(dtype=wp.transform),
    joint_penalty_k: wp.array(dtype=float),  # input
    joint_dof_dim: wp.array(dtype=int, ndim=2),
    joint_rest_angle: wp.array(dtype=float),
    joint_lambda_lin: wp.array(dtype=wp.vec3),
    joint_lambda_ang: wp.array(dtype=wp.vec3),
    joint_C0_lin: wp.array(dtype=wp.vec3),
    joint_C0_ang: wp.array(dtype=wp.vec3),
    joint_is_hard: wp.array(dtype=wp.int32),
    avbd_alpha: float,
):
    """
    Update augmented-Lagrangian duals for structural joint constraints (per-iteration).

    Only structural slots (linear, angular) exist in the penalty arrays; drive/limit
    forces read model stiffness directly and have no dual update.

    Args:
        joint_type: Joint types.
        joint_enabled: Per-joint enable flag (disabled joints are skipped).
        joint_parent: Parent body indices.
        joint_child: Child body indices.
        joint_X_p: Parent joint frames (local).
        joint_X_c: Child joint frames (local).
        joint_axis: Joint axis directions (per-DOF, used by REVOLUTE, PRISMATIC, D6).
        joint_qd_start: Joint DOF start indices (index into joint_axis).
        joint_constraint_start: Start index per joint in the solver constraint layout.
        body_q: Current body transforms (world).
        body_q_rest: Rest body transforms (world).
        joint_penalty_k: Fixed per-constraint penalties (read only).
        joint_dof_dim: Per-joint [lin_count, ang_count] for D6 joints.
        joint_rest_angle: Rest angles for revolute/D6 drive error computation.
        joint_lambda_lin: In/out bilateral lambda for linear hard constraints.
        joint_lambda_ang: In/out bilateral lambda for angular hard constraints.
        joint_C0_lin: Linear C0 snapshot (from init_joint_avbd).
        joint_C0_ang: Angular C0 snapshot (from init_joint_avbd).
        joint_is_hard: Per-slot hard/soft flag (1 = AL with lambda + C0, 0 = penalty only).
        avbd_alpha: Stabilization factor for C_stab = C - alpha * C0.
    """
    j = wp.tid()

    if not joint_enabled[j]:
        return

    parent = joint_parent[j]
    child = joint_child[j]

    # Early exit for invalid joints
    if child < 0:
        return

    jt = joint_type[j]
    if (
        jt != JointType.CABLE
        and jt != JointType.BALL
        and jt != JointType.FIXED
        and jt != JointType.REVOLUTE
        and jt != JointType.PRISMATIC
        and jt != JointType.D6
    ):
        return

    # Read solver constraint start index
    c_start = joint_constraint_start[j]

    # Compute joint frames in world space
    if parent >= 0:
        X_wp = body_q[parent] * joint_X_p[j]
        X_wp_rest = body_q_rest[parent] * joint_X_p[j]
    else:
        X_wp = joint_X_p[j]
        X_wp_rest = joint_X_p[j]
    X_wc = body_q[child] * joint_X_c[j]
    X_wc_rest = body_q_rest[child] * joint_X_c[j]

    # Cable joint: adaptive penalty for stretch and bend constraints
    if jt == JointType.CABLE:
        q_wp = wp.transform_get_rotation(X_wp)
        q_wc = wp.transform_get_rotation(X_wc)
        q_wp_rest = wp.transform_get_rotation(X_wp_rest)
        q_wc_rest = wp.transform_get_rotation(X_wc_rest)

        x_p = wp.transform_get_translation(X_wp)
        x_c = wp.transform_get_translation(X_wc)
        C_vec_stretch = x_c - x_p

        kappa = cable_get_kappa(q_wp, q_wc, q_wp_rest, q_wc_rest)

        # Stretch penalty update (constraint slot 0)
        stretch_idx = c_start
        lam_new = _update_dual_vec3(
            C_vec_stretch,
            joint_C0_lin[j],
            avbd_alpha,
            joint_penalty_k[stretch_idx],
            joint_lambda_lin[j],
            joint_is_hard[stretch_idx],
        )
        joint_lambda_lin[j] = lam_new

        # Bend penalty update (constraint slot 1)
        bend_idx = c_start + 1
        lam_new = _update_dual_vec3(
            kappa,
            joint_C0_ang[j],
            avbd_alpha,
            joint_penalty_k[bend_idx],
            joint_lambda_ang[j],
            joint_is_hard[bend_idx],
        )
        joint_lambda_ang[j] = lam_new
        return

    # BALL joint: update isotropic linear anchor-coincidence penalty (single scalar).
    if jt == JointType.BALL:
        x_p = wp.transform_get_translation(X_wp)
        x_c = wp.transform_get_translation(X_wc)
        C_vec = x_c - x_p

        i0 = c_start
        lam_new = _update_dual_vec3(
            C_vec,
            joint_C0_lin[j],
            avbd_alpha,
            joint_penalty_k[i0],
            joint_lambda_lin[j],
            joint_is_hard[i0],
        )
        joint_lambda_lin[j] = lam_new
        return

    # FIXED joint: update isotropic linear + isotropic angular penalties (2 scalars).
    if jt == JointType.FIXED:
        i_lin = c_start + 0
        i_ang = c_start + 1

        x_p = wp.transform_get_translation(X_wp)
        x_c = wp.transform_get_translation(X_wc)
        C_vec_lin = x_c - x_p
        lam_new = _update_dual_vec3(
            C_vec_lin,
            joint_C0_lin[j],
            avbd_alpha,
            joint_penalty_k[i_lin],
            joint_lambda_lin[j],
            joint_is_hard[i_lin],
        )
        joint_lambda_lin[j] = lam_new

        q_wp = wp.transform_get_rotation(X_wp)
        q_wc = wp.transform_get_rotation(X_wc)
        q_wp_rest = wp.transform_get_rotation(X_wp_rest)
        q_wc_rest = wp.transform_get_rotation(X_wc_rest)
        kappa = cable_get_kappa(q_wp, q_wc, q_wp_rest, q_wc_rest)
        lam_new = _update_dual_vec3(
            kappa,
            joint_C0_ang[j],
            avbd_alpha,
            joint_penalty_k[i_ang],
            joint_lambda_ang[j],
            joint_is_hard[i_ang],
        )
        joint_lambda_ang[j] = lam_new
        return

    # REVOLUTE joint: isotropic linear + perpendicular angular penalties (2 scalars).
    if jt == JointType.REVOLUTE:
        i_lin = c_start + 0
        i_ang = c_start + 1
        qd_start = joint_qd_start[j]
        q_wp = wp.transform_get_rotation(X_wp)
        P_lin, P_ang = build_joint_projectors(jt, joint_axis, qd_start, 0, 1, q_wp)

        x_p = wp.transform_get_translation(X_wp)
        x_c = wp.transform_get_translation(X_wc)
        C_vec_lin = P_lin * (x_c - x_p)
        lam_new = _update_dual_vec3(
            C_vec_lin,
            P_lin * joint_C0_lin[j],
            avbd_alpha,
            joint_penalty_k[i_lin],
            joint_lambda_lin[j],
            joint_is_hard[i_lin],
        )
        joint_lambda_lin[j] = lam_new

        q_wc = wp.transform_get_rotation(X_wc)
        q_wp_rest = wp.transform_get_rotation(X_wp_rest)
        q_wc_rest = wp.transform_get_rotation(X_wc_rest)
        kappa = cable_get_kappa(q_wp, q_wc, q_wp_rest, q_wc_rest)
        kappa_perp = P_ang * kappa
        lam_new = _update_dual_vec3(
            kappa_perp,
            P_ang * joint_C0_ang[j],
            avbd_alpha,
            joint_penalty_k[i_ang],
            joint_lambda_ang[j],
            joint_is_hard[i_ang],
        )
        joint_lambda_ang[j] = lam_new
        return

    # PRISMATIC joint: perpendicular linear + isotropic angular penalties (2 scalars).
    if jt == JointType.PRISMATIC:
        i_lin = c_start + 0
        i_ang = c_start + 1
        qd_start = joint_qd_start[j]
        q_wp = wp.transform_get_rotation(X_wp)
        P_lin, P_ang = build_joint_projectors(jt, joint_axis, qd_start, 1, 0, q_wp)

        x_p = wp.transform_get_translation(X_wp)
        x_c = wp.transform_get_translation(X_wc)
        C_vec = x_c - x_p
        C_vec_perp = P_lin * C_vec
        lam_new = _update_dual_vec3(
            C_vec_perp,
            P_lin * joint_C0_lin[j],
            avbd_alpha,
            joint_penalty_k[i_lin],
            joint_lambda_lin[j],
            joint_is_hard[i_lin],
        )
        joint_lambda_lin[j] = lam_new

        q_wc = wp.transform_get_rotation(X_wc)
        q_wp_rest = wp.transform_get_rotation(X_wp_rest)
        q_wc_rest = wp.transform_get_rotation(X_wc_rest)
        kappa = cable_get_kappa(q_wp, q_wc, q_wp_rest, q_wc_rest)
        kappa_perp = P_ang * kappa
        lam_new = _update_dual_vec3(
            kappa_perp,
            P_ang * joint_C0_ang[j],
            avbd_alpha,
            joint_penalty_k[i_ang],
            joint_lambda_ang[j],
            joint_is_hard[i_ang],
        )
        joint_lambda_ang[j] = lam_new
        return

    # D6 joint: projected linear + projected angular penalties (2 scalars).
    if jt == JointType.D6:
        i_lin = c_start + 0
        i_ang = c_start + 1
        lin_count = joint_dof_dim[j, 0]
        ang_count = joint_dof_dim[j, 1]
        qd_start = joint_qd_start[j]
        q_wp_rot = wp.transform_get_rotation(X_wp)
        P_lin, P_ang = build_joint_projectors(jt, joint_axis, qd_start, lin_count, ang_count, q_wp_rot)

        x_p = wp.transform_get_translation(X_wp)
        x_c = wp.transform_get_translation(X_wc)
        C_vec = x_c - x_p
        if lin_count < 3:
            C_vec_perp = P_lin * C_vec
            lam_new = _update_dual_vec3(
                C_vec_perp,
                P_lin * joint_C0_lin[j],
                avbd_alpha,
                joint_penalty_k[i_lin],
                joint_lambda_lin[j],
                joint_is_hard[i_lin],
            )
            joint_lambda_lin[j] = lam_new

        q_wc = wp.transform_get_rotation(X_wc)
        q_wp_rest = wp.transform_get_rotation(X_wp_rest)
        q_wc_rest = wp.transform_get_rotation(X_wc_rest)
        kappa = cable_get_kappa(q_wp_rot, q_wc, q_wp_rest, q_wc_rest)
        if ang_count < 3:
            kappa_perp = P_ang * kappa
            lam_new = _update_dual_vec3(
                kappa_perp,
                P_ang * joint_C0_ang[j],
                avbd_alpha,
                joint_penalty_k[i_ang],
                joint_lambda_ang[j],
                joint_is_hard[i_ang],
            )
            joint_lambda_ang[j] = lam_new
        return


@wp.kernel
def update_duals_body_body_contacts(
    rigid_contact_count: wp.array(dtype=int),
    rigid_contact_shape0: wp.array(dtype=int),
    rigid_contact_shape1: wp.array(dtype=int),
    rigid_contact_point0: wp.array(dtype=wp.vec3),
    rigid_contact_point1: wp.array(dtype=wp.vec3),
    rigid_contact_normal: wp.array(dtype=wp.vec3),
    rigid_contact_margin0: wp.array(dtype=float),
    rigid_contact_margin1: wp.array(dtype=float),
    shape_body: wp.array(dtype=int),
    body_q: wp.array(dtype=wp.transform),
    contact_material_mu: wp.array(dtype=float),
    contact_penalty_k: wp.array(dtype=float),  # input
    contact_lambda: wp.array(dtype=wp.vec3),  # input/output
    contact_C0: wp.array(dtype=wp.vec3),
    avbd_alpha: float,
    hard_contacts: int,
    stick_tangential_eps: float,
    contact_stick_flag: wp.array(dtype=wp.int32),  # output
):
    """
    Update AVBD augmented-Lagrangian duals for contact constraints (per-iteration).
    Hard mode: scalar isotropic k with vec3 lambda and cone clamping.
    Soft mode: no lambda update.
    """
    idx = wp.tid()
    if idx >= rigid_contact_count[0]:
        return

    shape_id_0 = rigid_contact_shape0[idx]
    shape_id_1 = rigid_contact_shape1[idx]
    body_id_0 = shape_body[shape_id_0]
    body_id_1 = shape_body[shape_id_1]

    if body_id_0 < 0 and body_id_1 < 0:
        return

    if body_id_0 >= 0:
        p0_world = wp.transform_point(body_q[body_id_0], rigid_contact_point0[idx])
    else:
        p0_world = rigid_contact_point0[idx]

    if body_id_1 >= 0:
        p1_world = wp.transform_point(body_q[body_id_1], rigid_contact_point1[idx])
    else:
        p1_world = rigid_contact_point1[idx]

    n = rigid_contact_normal[idx]
    d = p1_world - p0_world
    thickness_total = rigid_contact_margin0[idx] + rigid_contact_margin1[idx]

    if hard_contacts == 1:
        k = contact_penalty_k[idx]
        C0_vec = contact_C0[idx]
        lam_vec = contact_lambda[idx]
        mu = contact_material_mu[idx]

        C_vec = n * thickness_total - d
        C_stab = C_vec - avbd_alpha * C0_vec
        C_stab_n = wp.dot(C_stab, n)
        C_stab_t = C_stab - n * C_stab_n

        lam_n_old = wp.dot(lam_vec, n)
        lam_t_old = lam_vec - n * lam_n_old

        lam_n_new = wp.max(lam_n_old + k * C_stab_n, 0.0)

        lam_t_new = lam_t_old + k * C_stab_t
        lam_t_len = wp.length(lam_t_new)
        cone_limit = mu * lam_n_new
        inside_cone = lam_t_len <= cone_limit
        if lam_t_len > cone_limit and lam_t_len > 0.0:
            lam_t_new = lam_t_new * (cone_limit / lam_t_len)

        contact_lambda[idx] = n * lam_n_new + lam_t_new

        C_stab_t_len = wp.length(C_stab_t)

        stick = int(0)
        if inside_cone and C_stab_t_len < stick_tangential_eps and lam_n_new > 0.0:
            stick = int(1)
        contact_stick_flag[idx] = stick
    else:
        contact_stick_flag[idx] = int(0)


# -----------------------------
# Post-iteration kernels (after all iterations)
# -----------------------------
@wp.kernel
def update_body_velocity(
    dt: float,
    body_q: wp.array(dtype=wp.transform),
    body_com: wp.array(dtype=wp.vec3),
    body_contact_buffer_pre_alloc: int,
    body_contact_counts: wp.array(dtype=wp.int32),
    body_contact_indices: wp.array(dtype=wp.int32),
    contact_stick_flag: wp.array(dtype=wp.int32),
    apply_stick_deadzone: int,
    stick_freeze_translation_eps: float,
    stick_freeze_angular_eps: float,
    body_q_prev: wp.array(dtype=wp.transform),
    body_qd: wp.array(dtype=wp.spatial_vector),
    body_q_out: wp.array(dtype=wp.transform),
):
    """
    Update body velocities from position changes (world frame).

    Optionally applies a tiny body-level stick-contact deadzone before
    finite-difference velocity computation.
    Computes linear and angular velocities using finite differences.
    Also transfers the final body poses to body_q_out (fused copy from
    the in-place Gauss-Seidel iteration buffer to state_out).

    Linear: v = (com_current - com_prev) / dt
    Angular: omega from quaternion difference dq = q * q_prev^-1

    Args:
        dt: Time step.
        body_q: Current body transforms (world), from state_in (in-place iteration buffer).
        body_com: Center of mass offsets (local frame).
        body_contact_buffer_pre_alloc: Per-body contact-list capacity.
        body_contact_counts: Number of body-body contacts adjacent to each body.
        body_contact_indices: Flat per-body contact index lists.
        contact_stick_flag: Final per-contact stick classification for this step.
        apply_stick_deadzone: If nonzero, snap tiny sticky body motion back to the
            previous pose before computing velocity.
        stick_freeze_translation_eps: Translation deadzone [m] for snapping tiny sticky motion.
        stick_freeze_angular_eps: Angular deadzone [rad] for snapping tiny sticky motion.
        body_q_prev: Previous body transforms (input/output, advanced to current pose for next step).
        body_qd: Output body velocities (spatial vectors, world frame).
        body_q_out: Output body transforms (state_out), fused copy of body_q.
    """
    tid = wp.tid()

    # Read transforms
    pose = body_q[tid]
    pose_prev = body_q_prev[tid]

    x = wp.transform_get_translation(pose)
    x_prev = wp.transform_get_translation(pose_prev)
    q = wp.transform_get_rotation(pose)
    q_prev = wp.transform_get_rotation(pose_prev)

    if apply_stick_deadzone != 0:
        count = wp.min(body_contact_counts[tid], body_contact_buffer_pre_alloc)
        offset = tid * body_contact_buffer_pre_alloc
        sticky = int(0)
        for i in range(count):
            contact_idx = body_contact_indices[offset + i]
            if contact_stick_flag[contact_idx] != 0:
                sticky = int(1)
                break

        if sticky != 0:
            translation_delta = wp.length(x - x_prev)
            # Use dt=1 to measure quaternion delta magnitude as an angular step proxy.
            angular_delta = wp.length(quat_velocity(q, q_prev, 1.0))
            if translation_delta < stick_freeze_translation_eps and angular_delta < stick_freeze_angular_eps:
                pose = pose_prev
                x = x_prev
                q = q_prev

    # Compute COM positions
    com_local = body_com[tid]
    x_com = x + wp.quat_rotate(q, com_local)
    x_com_prev = x_prev + wp.quat_rotate(q_prev, com_local)

    # Linear velocity
    v = (x_com - x_com_prev) / dt

    # Angular velocity
    omega = quat_velocity(q, q_prev, dt)

    body_qd[tid] = wp.spatial_vector(v, omega)

    # Advance body_q_prev for next step (for kinematic bodies this is the only write).
    body_q_prev[tid] = pose

    body_q_out[tid] = pose


@wp.kernel
def update_cable_dahl_state(
    # Joint geometry
    joint_type: wp.array(dtype=int),
    joint_enabled: wp.array(dtype=bool),
    joint_parent: wp.array(dtype=int),
    joint_child: wp.array(dtype=int),
    joint_X_p: wp.array(dtype=wp.transform),
    joint_X_c: wp.array(dtype=wp.transform),
    joint_constraint_start: wp.array(dtype=int),
    joint_penalty_k: wp.array(dtype=float),
    joint_is_hard: wp.array(dtype=wp.int32),
    # Body states (final, after solver convergence)
    body_q: wp.array(dtype=wp.transform),
    body_q_rest: wp.array(dtype=wp.transform),
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
        joint_constraint_start: Start index per joint in the solver constraint layout
        joint_penalty_k: Per-constraint fixed stiffness; for cables, bend slot stores effective per-joint bend stiffness [N*m]
        body_q: Final body transforms (after convergence)
        body_q_rest: Rest body transforms
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

    # World-parent joints are valid; child body must exist.
    if child < 0:
        return

    # Dahl guard: skip when bend is hard-constrained (AL drives error to zero,
    # contradicting Dahl hysteresis which requires nonzero bending).
    c_start_dahl = joint_constraint_start[j]
    if joint_is_hard[c_start_dahl + 1] == 1:
        return

    # Compute joint frames in world space (final state)
    if parent >= 0:
        X_wp = body_q[parent] * joint_X_p[j]
        X_wp_rest = body_q_rest[parent] * joint_X_p[j]
    else:
        X_wp = joint_X_p[j]
        X_wp_rest = joint_X_p[j]
    X_wc = body_q[child] * joint_X_c[j]
    X_wc_rest = body_q_rest[child] * joint_X_c[j]

    q_wp = wp.transform_get_rotation(X_wp)
    q_wc = wp.transform_get_rotation(X_wc)
    q_wp_rest = wp.transform_get_rotation(X_wp_rest)
    q_wc_rest = wp.transform_get_rotation(X_wc_rest)

    # Compute final curvature vector at end of timestep
    kappa_final = cable_get_kappa(q_wp, q_wc, q_wp_rest, q_wc_rest)

    if not joint_enabled[j]:
        # Refresh Dahl state to current configuration with zero preload so that
        # re-enabling the joint does not see a stale kappa delta.
        joint_kappa_prev[j] = kappa_final
        joint_sigma_prev[j] = wp.vec3(0.0)
        joint_dkappa_prev[j] = wp.vec3(0.0)
        return

    # Read stored Dahl state (component-wise vectors)
    kappa_old = joint_kappa_prev[j]  # stored curvature
    d_kappa_old = joint_dkappa_prev[j]  # stored Delta kappa
    sigma_old = joint_sigma_prev[j]  # stored friction stress

    # Read per-joint Dahl parameters (isotropic)
    eps_max = joint_eps_max[j]  # Maximum persistent strain [rad]
    tau = joint_tau[j]  # Memory decay length [rad]

    # Bend stiffness is stored in constraint slot 1 for cable joints.
    c_start = joint_constraint_start[j]
    k_bend_target = joint_penalty_k[c_start + 1]  # [N*m]

    # Friction envelope: sigma_max = k_bend_target * eps_max.
    sigma_max = k_bend_target * eps_max  # [N*m]

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
