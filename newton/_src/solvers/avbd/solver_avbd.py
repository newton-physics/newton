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

from __future__ import annotations

import numpy as np
import warp as wp
from warp.types import float32, vector

from ...core.types import override
from ...sim import Contacts, Control, JointType, Model, State
from ..solver import SolverBase, integrate_rigid_body

# TODO: Grab changes from Warp that has fixed the backward pass
wp.set_module_options({"enable_backward": False})

# Small-angle threshold (radians)
SMALL_ANGLE_EPS = wp.constant(1.0e-7)

# Multi-threading per body for contact accumulation
# Multiple threads cooperatively process one body's contacts using strided iteration
NUM_THREADS_PER_BODY = 4  # 1 = serial, 4 = parallel

# Temporary test flags.

# Rotational dynamics mode: False=first-order small-angle; True=closed-form (small-angle below eps)
USE_EXACT_ROTATIONAL_DYNAMICS = wp.constant(False)  # Temporary test flag

# Soft constraint model switch (applies to contacts and joints):
# True  = AVBD-style soft constraints (adaptive penalty, warmstart + dual updates)
# False = VBD-style soft constraints (fixed material stiffness, no adaptation)
use_avbd = wp.constant(True)

# Stretch constraints: True = Cosserat stretch/shear, False = Simple pinning
use_cosserat_stretch = wp.constant(False)


@wp.func
def quat_rotate_z_axis(q: wp.quat) -> wp.vec3:
    """
    Optimized rotation of +Z axis (0, 0, 1) by quaternion.

    Closed-form solution for quat_rotate(q, (0,0,1)):
        result = (2*(qx*qz + qy*qw), 2*(qy*qz - qx*qw), 1 - 2*(qx^2 + qy^2))

    Args:
        q: Quaternion (x, y, z, w)

    Returns:
        Rotated +Z axis in world frame
    """
    qx = q[0]
    qy = q[1]
    qz = q[2]
    qw = q[3]

    return wp.vec3(2.0 * (qx * qz + qy * qw), 2.0 * (qy * qz - qx * qw), 1.0 - 2.0 * (qx * qx + qy * qy))


class vec6(vector(length=6, dtype=float32)):
    pass


@wp.struct
class ForceElementAdjacencyInfo:
    r"""
    Stores adjacency information for rigid bodies and their connected joints using CSR (Compressed Sparse Row) format.

    - body_adj_joints: Flattened array of joint IDs. Size is \sum_{i=0}^{|B|} N_i, where N_i is the
      number of joints connected to body i.

    - body_adj_joints_offsets: Offset array indicating where each body's joint list starts.
      Size is |B|+1 (number of bodies + 1).
      The number of joints adjacent to body i is: body_adj_joints_offsets[i+1] - body_adj_joints_offsets[i]
    """

    # Rigid body joint adjacency
    body_adj_joints: wp.array(dtype=int)
    body_adj_joints_offsets: wp.array(dtype=int)

    def to(self, device):
        if device == self.body_adj_joints.device:
            return self
        else:
            adjacency_gpu = ForceElementAdjacencyInfo()
            adjacency_gpu.body_adj_joints = self.body_adj_joints.to(device)
            adjacency_gpu.body_adj_joints_offsets = self.body_adj_joints_offsets.to(device)

            return adjacency_gpu


@wp.func
def get_body_num_adjacent_joints(adjacency: ForceElementAdjacencyInfo, body: wp.int32):
    return adjacency.body_adj_joints_offsets[body + 1] - adjacency.body_adj_joints_offsets[body]


@wp.func
def get_body_adjacent_joint_id(adjacency: ForceElementAdjacencyInfo, body: wp.int32, joint: wp.int32):
    offset = adjacency.body_adj_joints_offsets[body]
    return adjacency.body_adj_joints[offset + joint]


@wp.func
def evaluate_rigid_contact_from_collision(
    body_a_index: int,
    body_b_index: int,
    body_q: wp.array(dtype=wp.transform),
    body_q_prev: wp.array(dtype=wp.transform),
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
    """
    Compute contact forces and Hessian blocks for contacts.

    Args:
        body_a_index: Body A index (-1 for static/kinematic body)
        body_b_index: Body B index (-1 for static/kinematic body)
        body_q: Current body transforms (world space)
        body_q_prev: Previous body transforms (world space)
        body_com: Body center-of-mass offsets (local body coordinates)
        contact_point_a_local: Contact point on body A (local body coordinates)
        contact_point_b_local: Contact point on body B (local body coordinates)
        contact_normal: Unit contact normal from collision detection (A to B, world coordinates)
        penetration_depth: Penetration depth from collision detection (> 0 when penetrating)
        contact_ke: Contact normal stiffness
        contact_kd: Contact damping coefficient
        friction_mu: Coulomb friction coefficient
        friction_epsilon: Friction regularization parameter (smoothing distance)
        dt: Time step

    Returns:
        Tuple of (force_a, torque_a, h_ll_a, h_al_a, h_aa_a,
                  force_b, torque_b, h_ll_b, h_al_b, h_aa_b):
        Per-body forces, torques, and Hessian blocks:
        - h_ll: Linear-linear coupling (3x3)
        - h_al: Angular-linear coupling (3x3)
        - h_aa: Angular-angular coupling (3x3)
    """

    # Early exit: no penetration or zero stiffness
    if penetration_depth <= 0.0 or contact_ke <= 0.0:
        zero_vec = wp.vec3(0.0)
        zero_mat = wp.mat33(0.0)
        return (zero_vec, zero_vec, zero_mat, zero_mat, zero_mat, zero_vec, zero_vec, zero_mat, zero_mat, zero_mat)

    # Handle static bodies (index < 0) with identity transforms
    X_wa = wp.transform_identity() if body_a_index < 0 else body_q[body_a_index]
    X_wa_prev = wp.transform_identity() if body_a_index < 0 else body_q_prev[body_a_index]
    body_a_com_local = wp.vec3(0.0) if body_a_index < 0 else body_com[body_a_index]

    X_wb = wp.transform_identity() if body_b_index < 0 else body_q[body_b_index]
    X_wb_prev = wp.transform_identity() if body_b_index < 0 else body_q_prev[body_b_index]
    body_b_com_local = wp.vec3(0.0) if body_b_index < 0 else body_com[body_b_index]

    # Centers of mass in world coordinates
    x_com_a_now = wp.transform_point(X_wa, body_a_com_local)
    x_com_b_now = wp.transform_point(X_wb, body_b_com_local)

    # Contact points in world coordinates
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
    f_total = contact_normal * (contact_ke * penetration_depth)
    K_total = contact_ke * wp.outer(contact_normal, contact_normal)

    # Compute relative velocity for damping and friction
    v_rel = dx_rel / dt
    v_dot_n = wp.dot(contact_normal, v_rel)

    # Apply damping only when contacts are compressing (v_n < 0, bodies approaching)
    if contact_kd > 0.0 and v_dot_n < 0.0:
        damping_coeff = contact_kd * contact_ke
        damping_force = -damping_coeff * v_dot_n * contact_normal
        damping_hessian = (damping_coeff / dt) * wp.outer(contact_normal, contact_normal)
        f_total = f_total + damping_force
        K_total = K_total + damping_hessian

    # Normal force magnitude for friction (elastic only, excluding damping)
    elastic_normal_force = contact_ke * penetration_depth
    normal_load = wp.max(0.0, elastic_normal_force)

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
def compute_projected_isotropic_friction(
    friction_mu: float,
    normal_load: float,
    n_hat: wp.vec3,
    slip_u: wp.vec3,
    eps_u: float,
) -> tuple[wp.vec3, wp.mat33]:
    """
    Isotropic friction aligned with tangential slip using projector P = I - n n^T.

    Returns force and Hessian in world coordinates without constructing a tangent basis.
    """
    # Tangential slip in the contact tangent plane without forming P: u_t = u - n (n·u)
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
def cable_get_kappa(q_wp: wp.quat, q_wc: wp.quat, q_wp_rest: wp.quat, q_wc_rest: wp.quat) -> wp.vec3:
    """
    Compute cable bending curvature vector (kappa) from relative rotation.

    Kappa is the rotation vector representation: kappa = 2*theta*axis
    where theta is the bend angle and axis is the bend axis.

    Args:
        q_wp: Parent segment orientation (world frame)
        q_wc: Child segment orientation (world frame)
        q_wp_rest: Parent segment rest orientation
        q_wc_rest: Child segment rest orientation

    Returns:
        wp.vec3: Curvature vector kappa = 2*theta*axis (rotation vector form)
    """
    # Compute relative rotations in current and rest configurations
    r_current = wp.mul(wp.quat_inverse(q_wp), q_wc)
    r_rest = wp.mul(wp.quat_inverse(q_wp_rest), q_wc_rest)

    # Rotation from rest to current: rel_q = r_current * r_rest^-1
    rel_q = wp.mul(r_current, wp.quat_inverse(r_rest))

    # Ensure positive quaternion (q and -q represent same rotation, convention: w >= 0)
    if rel_q[3] < 0.0:
        rel_q = wp.quat(-rel_q[0], -rel_q[1], -rel_q[2], -rel_q[3])
    rel_q = wp.normalize(rel_q)

    # Extract axis-angle: quaternion q = [axis*sin(theta/2), cos(theta/2)]
    v = wp.vec3(rel_q[0], rel_q[1], rel_q[2])  # v = axis * sin(theta/2)
    v_norm = wp.length(v)
    w = rel_q[3]  # w = cos(theta/2)

    # Convert to rotation vector: kappa = 2*theta*axis
    # Small angle: sin(theta/2) ~= theta/2, so v ~= axis*theta/2, thus kappa = 2*v
    # Large angle: kappa = (axis) * (2*theta) = (v/sin(theta/2)) * (2*atan2(sin(theta/2), cos(theta/2)))
    return v * 2.0 if v_norm < SMALL_ANGLE_EPS else (v / v_norm) * (2.0 * wp.atan2(v_norm, w))


@wp.kernel
def forward_step_rigid_bodies(
    dt: float,
    gravity: wp.array(dtype=wp.vec3),
    body_q_prev: wp.array(dtype=wp.transform),
    body_q: wp.array(dtype=wp.transform),
    body_qd: wp.array(dtype=wp.spatial_vector),
    body_f: wp.array(dtype=wp.spatial_vector),
    body_com: wp.array(dtype=wp.vec3),
    body_inertia: wp.array(dtype=wp.mat33),
    body_inv_mass: wp.array(dtype=float),
    body_inv_inertia: wp.array(dtype=wp.mat33),
    body_inertia_q: wp.array(dtype=wp.transform),
):
    """
    Forward integration step for rigid bodies in AVBD solver.

    Integrates rigid body motion using semi-implicit Euler integration.
    Stores previous transforms for velocity computation and sets inertial targets
    for the VBD-style solve. No angular damping is applied (consistent with VBD).

    Args:
        dt: Time step
        gravity: Gravity vector array (world frame)
        body_q_prev: Previous body transforms (output, for velocity computation)
        body_q: Current body transforms (input/output)
        body_qd: Current body velocities (spatial vectors)
        body_f: External forces on bodies (spatial wrenches)
        body_com: Center of mass offsets (local body frame)
        body_inertia: Inertia tensors (local body frame)
        body_inv_mass: Inverse masses (0 for kinematic bodies)
        body_inv_inertia: Inverse inertia tensors (local body frame)
        body_inertia_q: Inertial body transforms (output, VBD solve target)
    """
    tid = wp.tid()

    # Read current transform
    q_current = body_q[tid]

    # Store previous transform for velocity computation
    body_q_prev[tid] = q_current

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
    q_new, _ = integrate_rigid_body(
        q_current,
        qd_current,
        f_current,
        com_local,
        I_local,
        inv_m,
        inv_I,
        gravity,
        0.0,  # angular_damping = 0 (consistent with VBD)
        dt,
    )

    # Update current transform and set inertial target
    body_q[tid] = q_new
    body_inertia_q[tid] = q_new


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

    Complexity: 9 multiplies, 6 divides (optimal for 3x3)
    """
    # Unpack Cholesky factor for readability
    L00 = Lp[0]
    L10 = Lp[1]
    L11 = Lp[2]
    L20 = Lp[3]
    L21 = Lp[4]
    L22 = Lp[5]

    # Forward substitution: L*y = b
    # Solve lower triangular system:
    #   [L00   0   0 ] [y0]   [b0]
    #   [L10 L11   0 ] [y1] = [b1]
    #   [L20 L21 L22] [y2]   [b2]
    y0 = b[0] / L00  # L00*y0 = b0
    y1 = (b[1] - L10 * y0) / L11  # L10*y0 + L11*y1 = b1
    y2 = (b[2] - L20 * y0 - L21 * y1) / L22  # L20*y0 + L21*y1 + L22*y2 = b2

    # Backward substitution: L^T*x = y
    # Solve upper triangular system:
    #   [L00 L10 L20] [x0]   [y0]
    #   [ 0  L11 L21] [x1] = [y1]
    #   [ 0   0  L22] [x2]   [y2]
    x2 = y2 / L22  # L22*x2 = y2
    x1 = (y1 - L21 * x2) / L11  # L11*x1 + L21*x2 = y1
    x0 = (y0 - L10 * x1 - L20 * x2) / L00  # L00*x0 + L10*x1 + L20*x2 = y0

    return wp.vec3(x0, x1, x2)


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
    adjacency: ForceElementAdjacencyInfo,
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
      1. Compute inertial forces/Hessians (m/I * (target - current) / dt^2)
      2. Accumulate external forces/Hessians from rigid contacts
      3. Accumulate joint forces/Hessians from adjacent joints
      4. Solve 6x6 block system via Schur complement: S = A - C*M^-1*C^T
      5. Update pose: rotation from angular increment, position from linear increment

    Args:
        dt: Time step
        body_ids_in_color: Body indices in current color group (for parallel coloring)
        body_q: Current body transforms (input/output via body_q_new)
        body_q_prev: Previous body transforms (for damping)
        body_q_rest: Rest transforms (for joint targets)
        body_mass: Body masses
        body_inv_mass: Inverse masses (0 for kinematic bodies)
        body_inertia: Inertia tensors (local body frame)
        body_inertia_q: Inertial target transforms (from forward integration)
        body_com: Center of mass offsets (local body frame)
        adjacency: Body-joint adjacency (CSR format)
        joint_*: Joint configuration arrays
        joint_penalty_k: AVBD per-DOF penalty stiffness
        external_forces: External linear forces from rigid contacts
        external_torques: External angular torques from rigid contacts
        external_hessian_ll: Linear-linear Hessian block (3x3)
        external_hessian_al: Angular-linear coupling Hessian block (3x3)
        external_hessian_aa: Angular-angular Hessian block (3x3)
        body_q_new: Output updated body transforms

    Note: Consistent with VBD particle solve (see solver_vbd.py:solve_trimesh_with_self_contact_penetration_free)
          Both use: inertia force/hessian + external + constraints, then solve linear system.
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

    # Compute relative rotation
    dq = wp.mul(wp.quat_inverse(rot_current), rot_star)

    # Enforce shortest arc
    if dq[3] < 0.0:
        dq = wp.quat(-dq[0], -dq[1], -dq[2], -dq[3])

    # Convert quaternion to rotation vector
    if not USE_EXACT_ROTATIONAL_DYNAMICS:
        v = wp.vec3(dq[0], dq[1], dq[2])
        theta_body = 2.0 * v
    else:
        v = wp.vec3(dq[0], dq[1], dq[2])
        angle = wp.length(v)
        w_scalar = dq[3]
        if angle > SMALL_ANGLE_EPS:
            theta_magnitude = 2.0 * wp.atan2(angle, w_scalar)
            theta_body = theta_magnitude * (v / angle)
        else:
            theta_body = 2.0 * v

    # Angular inertial torque
    tau_body = I_body * (theta_body * dt_sqr_reciprocal)
    tau_world = wp.quat_rotate(rot_current, tau_body)

    # Angular Hessian in world frame
    R = wp.quat_to_matrix(rot_current)
    ex = wp.vec3(R[0, 0], R[1, 0], R[2, 0])
    ey = wp.vec3(R[0, 1], R[1, 1], R[2, 1])
    ez = wp.vec3(R[0, 2], R[1, 2], R[2, 2])
    Ixx = I_body[0, 0]
    Iyy = I_body[1, 1]
    Izz = I_body[2, 2]
    angular_hessian = dt_sqr_reciprocal * (Ixx * wp.outer(ex, ex) + Iyy * wp.outer(ey, ey) + Izz * wp.outer(ez, ez))

    # Accumulate external forces (rigid contacts)
    # Read external contributions
    ext_torque = external_torques[body_index]
    ext_force = external_forces[body_index]
    ext_h_aa = external_hessian_aa[body_index]
    ext_h_al = external_hessian_al[body_index]
    ext_h_ll = external_hessian_ll[body_index]

    # Initialize accumulators
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
            joint_parent,
            joint_child,
            joint_X_p,
            joint_X_c,
            joint_qd_start,
            joint_target_kd,
            joint_penalty_k,
            dt,
        )

        f_force = f_force + joint_force
        f_torque = f_torque + joint_torque
        h_ll = h_ll + joint_H_ll
        h_al = h_al + joint_H_al
        h_aa = h_aa + joint_H_aa

    # Solve 6x6 block system via Schur complement
    # Regularize for numerical stability
    trM = wp.trace(h_ll) / 3.0
    trA = wp.trace(h_aa) / 3.0
    epsM = 1.0e-6 * (trM + 1.0)
    epsA = 1.0e-6 * (trA + 1.0)

    M_reg = wp.mat33(
        h_ll[0, 0] + epsM,
        h_ll[0, 1],
        h_ll[0, 2],
        h_ll[1, 0],
        h_ll[1, 1] + epsM,
        h_ll[1, 2],
        h_ll[2, 0],
        h_ll[2, 1],
        h_ll[2, 2] + epsM,
    )
    A_reg = wp.mat33(
        h_aa[0, 0] + epsA,
        h_aa[0, 1],
        h_aa[0, 2],
        h_aa[1, 0],
        h_aa[1, 1] + epsA,
        h_aa[1, 2],
        h_aa[2, 0],
        h_aa[2, 1],
        h_aa[2, 2] + epsA,
    )

    # Factorize M
    Lm_p = chol33(M_reg)

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

    # Compute and regularize Schur complement
    S = A_reg - (h_al * MinvCt)
    trS = wp.trace(S) / 3.0
    epsS = 1.0e-9 * (trS + 1.0)
    S_reg = wp.mat33(
        S[0, 0] + epsS, S[0, 1], S[0, 2], S[1, 0], S[1, 1] + epsS, S[1, 2], S[2, 0], S[2, 1], S[2, 2] + epsS
    )

    # Factorize S
    Ls_p = chol33(S_reg)

    # Solve for angular increment
    rhs_w = f_torque - (h_al * MinvF)
    w_world = chol33_solve(Ls_p, rhs_w)

    # Solve for linear increment
    Ct_w = wp.transpose(h_al) * w_world
    x_inc = chol33_solve(Lm_p, f_force - Ct_w)

    # Update pose from increments
    # Convert angular increment to quaternion
    if USE_EXACT_ROTATIONAL_DYNAMICS:
        ang_mag = wp.length(w_world)
        if ang_mag > SMALL_ANGLE_EPS:
            dq_world = wp.quat_from_axis_angle(w_world / ang_mag, ang_mag)
        else:
            half_w = w_world * 0.5
            dq_world = wp.quat(half_w[0], half_w[1], half_w[2], 1.0)
            dq_world = wp.normalize(dq_world)
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
    body_q: wp.array(dtype=wp.transform),
    body_q_new: wp.array(dtype=wp.transform),
):
    """
    Copy solved body transforms back to main array.

    Equivalent to VBD's copy_particle_positions_back. Used after parallel
    solve to write updated transforms from temporary buffer to main state.

    Args:
        body_ids_in_color: Body indices in current color group
        body_q: Main body transforms array (output)
        body_q_new: Temporary solved transforms (input)
    """
    tid = wp.tid()
    body_index = body_ids_in_color[tid]
    body_q[body_index] = body_q_new[body_index]


@wp.kernel
def update_body_velocity(
    dt: float,
    body_q: wp.array(dtype=wp.transform),
    body_q_prev: wp.array(dtype=wp.transform),
    body_com: wp.array(dtype=wp.vec3),
    body_qd: wp.array(dtype=wp.spatial_vector),
):
    """
    Update body velocities from position changes.

    Equivalent to VBD's update_velocity but for rigid bodies (6-DOF).
    Computes linear and angular velocities using finite differences.

    Linear: v = (com_current - com_prev) / dt
    Angular: omega from quaternion difference dq = q * q_prev^-1

    Args:
        dt: Time step
        body_q: Current body transforms
        body_q_prev: Previous body transforms
        body_com: Center of mass offsets (local frame)
        body_qd: Output body velocities (spatial vectors)
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

    # Compute quaternion difference
    dq = q * wp.quat_inverse(q_prev)
    dq = wp.normalize(dq)

    # Enforce shortest arc
    if dq[3] < 0.0:
        dq = wp.quat(-dq[0], -dq[1], -dq[2], -dq[3])

    # Convert to angular velocity
    if USE_EXACT_ROTATIONAL_DYNAMICS:
        v_part = wp.vec3(dq[0], dq[1], dq[2])
        w_scalar = dq[3]
        v_norm = wp.length(v_part)
        if v_norm > SMALL_ANGLE_EPS:
            theta = 2.0 * wp.atan2(v_norm, w_scalar)
            omega = (theta / dt) * (v_part / v_norm)
        else:
            omega = (2.0 / dt) * v_part
    else:
        omega = (2.0 / dt) * wp.vec3(dq[0], dq[1], dq[2])

    body_qd[tid] = wp.spatial_vector(v, omega)


# AVBD helper kernels
@wp.kernel
def avbd_warmstart_joints(
    joint_penalty_k: wp.array(dtype=float),
    joint_target_ke: wp.array(dtype=float),
    gamma: float,
    penalty_min: float,
    penalty_max: float,
):
    """
    Warm-start AVBD penalty parameters for soft constraints.

    Decay penalty k from previous step and clamp to valid range.
    """
    i = wp.tid()

    # Decay penalty from previous step
    k_new = gamma * joint_penalty_k[i]
    k_new = wp.max(k_new, penalty_min)
    k_new = wp.min(k_new, penalty_max)

    # Cap by material stiffness
    k_new = wp.min(k_new, joint_target_ke[i])
    joint_penalty_k[i] = k_new


@wp.kernel
def avbd_warmstart_contacts(
    rigid_contact_count: wp.array(dtype=int),
    rigid_contact_shape0: wp.array(dtype=int),
    rigid_contact_shape1: wp.array(dtype=int),
    shape_material_ke: wp.array(dtype=float),
    shape_material_kd: wp.array(dtype=float),
    shape_material_mu: wp.array(dtype=float),
    gamma: float,
    penalty_min: float,
    penalty_max: float,
    # Outputs
    contact_penalty_k: wp.array(dtype=float),
    contact_material_ke: wp.array(dtype=float),
    contact_material_kd: wp.array(dtype=float),
    contact_material_mu: wp.array(dtype=float),
):
    """
    Warm-start contact penalties and cache material properties.

    Performs two tasks per contact per step:
    1. Cache averaged material properties (ke, kd, mu) for reuse in all solver iterations
    2. Decay penalty parameter from previous step: k_new = clamp(gamma * k_old, min, max)

    Algorithm:
    - Penalty decay: k *= gamma
    - Clamping: [penalty_min, penalty_max] for numerical stability
    - Material cap: k <= material_stiffness (soft constraints only)

    Args:
        rigid_contact_count: Number of active contacts
        rigid_contact_shape0: Shape index for first body
        rigid_contact_shape1: Shape index for second body
        shape_material_ke: Contact stiffness per shape
        shape_material_kd: Contact damping per shape
        shape_material_mu: Friction coefficient per shape
        gamma: Decay factor
        penalty_min: Minimum penalty value
        penalty_max: Maximum penalty value
        contact_penalty_k: Penalty parameters (input/output)
        contact_material_ke: Cached averaged stiffness (output)
        contact_material_kd: Cached averaged damping (output)
        contact_material_mu: Cached averaged friction (output)
    """
    i = wp.tid()
    if i >= rigid_contact_count[0]:
        return

    # Read shape indices
    shape_id_0 = rigid_contact_shape0[i]
    shape_id_1 = rigid_contact_shape1[i]

    # Cache averaged material properties (arithmetic mean)
    # Computed once per step, reused in all iterations
    avg_ke = 0.5 * (shape_material_ke[shape_id_0] + shape_material_ke[shape_id_1])
    avg_kd = 0.5 * (shape_material_kd[shape_id_0] + shape_material_kd[shape_id_1])
    avg_mu = 0.5 * (shape_material_mu[shape_id_0] + shape_material_mu[shape_id_1])

    contact_material_ke[i] = avg_ke
    contact_material_kd[i] = avg_kd
    contact_material_mu[i] = avg_mu

    # Warm-start penalty: decay and clamp
    k_new = wp.clamp(gamma * contact_penalty_k[i], penalty_min, penalty_max)

    # Cap by material stiffness (soft constraints only)
    k_new = wp.min(k_new, avg_ke)

    contact_penalty_k[i] = k_new


@wp.kernel
def compute_contact_material_properties(
    rigid_contact_count: wp.array(dtype=int),
    rigid_contact_shape0: wp.array(dtype=int),
    rigid_contact_shape1: wp.array(dtype=int),
    shape_material_ke: wp.array(dtype=float),
    shape_material_kd: wp.array(dtype=float),
    shape_material_mu: wp.array(dtype=float),
    # Outputs
    contact_material_ke: wp.array(dtype=float),
    contact_material_kd: wp.array(dtype=float),
    contact_material_mu: wp.array(dtype=float),
):
    """
    Cache averaged material properties for rigid contacts.

    Computes and stores averaged material properties (ke, kd, mu) for each contact
    once per step. Used by VBD soft contact mode where material properties are
    reused across all solver iterations.

    Note: AVBD mode uses avbd_warmstart_contacts which combines this with penalty warmstart.

    Args:
        rigid_contact_count: Number of active contacts
        rigid_contact_shape0: Shape index for first body
        rigid_contact_shape1: Shape index for second body
        shape_material_ke: Contact stiffness per shape
        shape_material_kd: Contact damping per shape
        shape_material_mu: Friction coefficient per shape
        contact_material_ke: Cached averaged stiffness (output)
        contact_material_kd: Cached averaged damping (output)
        contact_material_mu: Cached averaged friction (output)
    """
    i = wp.tid()
    if i >= rigid_contact_count[0]:
        return

    # Read shape indices
    shape_id_0 = rigid_contact_shape0[i]
    shape_id_1 = rigid_contact_shape1[i]

    # Cache averaged material properties (arithmetic mean)
    # Computed once per step, reused in all iterations
    contact_material_ke[i] = 0.5 * (shape_material_ke[shape_id_0] + shape_material_ke[shape_id_1])
    contact_material_kd[i] = 0.5 * (shape_material_kd[shape_id_0] + shape_material_kd[shape_id_1])
    contact_material_mu[i] = 0.5 * (shape_material_mu[shape_id_0] + shape_material_mu[shape_id_1])


@wp.kernel
def avbd_update_duals_joint(
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
    joint_penalty_k: wp.array(dtype=float),
    beta: float,
    penalty_max: float,
):
    """
    Update AVBD penalty parameters for joint constraints.

    Adaptively grows penalty parameters based on constraint violation:
        penalty_new = min(penalty + beta * |C|, min(penalty_max, stiffness))

    Algorithm:
    - Compute constraint violation C for each DOF
    - Increase penalty: k += beta * |C|
    - Clamp to valid range: [0, min(penalty_max, stiffness)]

    Args:
        joint_type: Joint type IDs
        joint_parent: Parent body indices
        joint_child: Child body indices
        joint_X_p: Parent joint frames
        joint_X_c: Child joint frames
        joint_qd_start: DOF start indices
        joint_dof_dim: DOF dimensions [linear, angular]
        joint_target_ke: Target stiffness per DOF
        body_q: Current body transforms
        body_q_rest: Rest body transforms
        body_com: Center of mass offsets
        joint_penalty_k: Penalty parameters (input/output)
        beta: Penalty growth rate
        penalty_max: Maximum penalty value
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
    ang_axes = joint_dof_dim[j, 1]

    # Compute joint frames in world space
    X_wp = body_q[parent] * joint_X_p[j]
    X_wc = body_q[child] * joint_X_c[j]
    X_wp_rest = body_q_rest[parent] * joint_X_p[j]
    X_wc_rest = body_q_rest[child] * joint_X_c[j]

    # Cable joint: adaptive penalty for stretch and bend constraints
    if joint_type[j] == JointType.CABLE:
        # Read body poses
        parent_pose = body_q[parent]
        child_pose = body_q[child]
        x_p_origin = wp.transform_get_translation(parent_pose)
        x_c_origin = wp.transform_get_translation(child_pose)

        # Read joint frame rotations
        q_wp = wp.transform_get_rotation(X_wp)
        q_wc = wp.transform_get_rotation(X_wc)
        q_wp_rest = wp.transform_get_rotation(X_wp_rest)
        q_wc_rest = wp.transform_get_rotation(X_wc_rest)

        # Compute stretch constraint violation
        if use_cosserat_stretch:
            # Cosserat stretch/shear: C = (x_c - x_p)/L - director
            parent_com_local = body_com[parent]
            rest_length = 2.0 * parent_com_local[2]

            # Skip degenerate segments
            if rest_length < 1.0e-9:
                return

            # Normalized displacement
            d = x_c_origin - x_p_origin
            inv_l0 = 1.0 / rest_length

            # Director: parent +Z axis in world frame (optimized rotation)
            q_p = wp.transform_get_rotation(parent_pose)
            director = quat_rotate_z_axis(q_p)

            # Constraint violation magnitude
            C_stretch = wp.length(d * inv_l0 - director)
        else:
            # Simple pinning: C = ||x_c - x_p||
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
        k_stretch_new = wp.min(k_stretch + beta * C_stretch, wp.min(penalty_max, stiffness_stretch))
        joint_penalty_k[stretch_idx] = k_stretch_new

        # Update bend penalty (DOF 1, if present)
        if ang_axes > 0:
            bend_idx = dof_start + lin_axes
            stiffness_bend = joint_target_ke[bend_idx]
            k_bend = joint_penalty_k[bend_idx]
            k_bend_new = wp.min(k_bend + beta * C_bend, wp.min(penalty_max, stiffness_bend))
            joint_penalty_k[bend_idx] = k_bend_new


@wp.kernel
def avbd_update_duals_contact(
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
    contact_penalty_k: wp.array(dtype=float),
    contact_material_ke: wp.array(dtype=float),
    beta: float,
    penalty_max: float,
):
    """
    Update AVBD penalty parameters for contact constraints.

    Adaptively grows penalty parameters based on penetration depth:
        penalty_new = min(penalty + beta * |C|, min(penalty_max, stiffness))

    Algorithm:
    - Compute constraint violation C (penetration depth)
    - Increase penalty: k += beta * |C|
    - Clamp to valid range: [0, min(penalty_max, stiffness)]

    Args:
        rigid_contact_count: Number of active contacts
        rigid_contact_shape0: Shape index for first body
        rigid_contact_shape1: Shape index for second body
        rigid_contact_point0: Contact points in body 0 local frame
        rigid_contact_point1: Contact points in body 1 local frame
        rigid_contact_normal: Contact normals in world frame
        rigid_contact_thickness0: Shape thickness (collision margin) for body 0
        rigid_contact_thickness1: Shape thickness (collision margin) for body 1
        shape_body: Body index per shape
        body_q: Current body transforms
        contact_penalty_k: Penalty parameters (input/output)
        contact_material_ke: Cached averaged stiffness
        beta: Penalty growth rate
        penalty_max: Maximum penalty value
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
    # Normal points from body 0 to body 1, so flip it
    normal = -rigid_contact_normal[idx]
    dist = wp.dot(normal, p1_world - p0_world)
    thickness_total = rigid_contact_thickness0[idx] + rigid_contact_thickness1[idx]
    penetration = wp.max(0.0, thickness_total - dist)

    # Update penalty: k_new = min(k + beta * |C|, min(penalty_max, stiffness))
    k = contact_penalty_k[idx]
    k_new = wp.min(k + beta * penetration, wp.min(penalty_max, stiffness))
    contact_penalty_k[idx] = k_new


@wp.func
def compute_right_jacobian_inverse(kappa: wp.vec3) -> wp.mat33:
    """
    Compute inverse of right Jacobian Jr^{-1}(kappa) for SO(3) rotation vectors.

    The right Jacobian relates angular velocity omega to time derivative of rotation vector:
        d(kappa)/dt = Jr(kappa) * omega

    Therefore: omega = Jr^{-1}(kappa) * d(kappa)/dt

    Formula (closed-form):
        Jr^{-1}(kappa) = I + (1/2)*[kappa]_x + b*[kappa]_x^2

    where:
        b = (1/theta^2) - (1 + cos(theta))/(2*theta*sin(theta))
        theta = ||kappa||

    Small-angle approximation (theta < eps):
        Jr^{-1}(kappa) ≈ I + (1/2)*[kappa]_x

    This is used to compute the Jacobian mapping rotation-vector forces to angular torques
    in Cosserat rod bending constraints.

    Args:
        kappa: Rotation vector (axis-angle representation, theta*axis)

    Returns:
        3x3 matrix: Jr^{-1}(kappa)
    """
    theta = wp.length(kappa)
    kappa_skew = wp.skew(kappa)

    # Small-angle approximation for numerical stability
    if (theta < SMALL_ANGLE_EPS) or (not USE_EXACT_ROTATIONAL_DYNAMICS):
        # Jr^{-1} ≈ I + (1/2)*[kappa]_x  (first-order Taylor series)
        return wp.identity(3, float) + 0.5 * kappa_skew

    # Full formula for general rotations
    sin_theta = wp.sin(theta)
    cos_theta = wp.cos(theta)

    # Coefficient for [kappa]_x^2 term
    b = (1.0 / (theta * theta)) - (1.0 + cos_theta) / (2.0 * theta * sin_theta)

    # Jr^{-1} = I + (1/2)*[kappa]_x + b*[kappa]_x^2
    return wp.identity(3, float) + 0.5 * kappa_skew + b * (kappa_skew * kappa_skew)


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
    damping: float,
    dt: float,
):
    """
    Compute AVBD cable bending force and Hessian using Cosserat rod theory.

    Physics:
        Continuum Cosserat bending energy:
            E = integral[ (1/2) * EI * ||curvature||^2 ] ds

        Discretized over segment length L:
            E = (1/2) * (EI/L) * ||kappa||^2

        where:
            - EI: bending stiffness [N*m^2] (Young's modulus times second moment of area)
            - kappa: rotation vector between segments [rad] (discrete curvature)
            - L: segment rest length [m]

    Implementation:
        1. Compute rotation vector: kappa = 2*axis*theta (relative rotation)
        2. Map to world torque: tau = R_wp * Jr^{-1}^T * (k_eff * kappa)
        3. Compute Gauss-Newton Hessian: H = k_eff * J_world * J_world^T
        4. Add Rayleigh damping: f_d = -D * dkappa/dt, H_d = (D/dt) * H

    Jacobian mapping:
        J_world = R_wp * Jr^{-1}^T

        Maps rotation-vector space forces to world-space angular torques.
        Jr^{-1} converts rotation-vector velocities to angular velocities.

    Args:
        q_wp: Parent segment orientation (world frame)
        q_wc: Child segment orientation (world frame)
        q_wp_rest: Parent rest orientation
        q_wc_rest: Child rest orientation
        q_wp_prev: Parent previous orientation (for damping)
        q_wc_prev: Child previous orientation (for damping)
        is_parent: True for parent body, False for child (flips torque sign)
        k_eff: Effective stiffness [N*m/rad] (pre-scaled: k_eff = EI/L)
        damping: Rayleigh damping coefficient (dimensionless multiplier on H)
        dt: Time step [s]

    Returns:
        tuple: (tau_world, H_aa)
            - tau_world: World-space angular torque [N*m] (3D vector)
            - H_aa: Angular-angular Hessian block [N*m/rad] (3x3 SPD matrix)

    Note:
        Caller must pre-scale stiffness (k_eff = EI/L) to avoid redundant division.
        Linear 1/L scaling (not 1/L^2) follows Cosserat rod physics.
    """
    # Compute relative rotation vector (current configuration)
    kappa_now = cable_get_kappa(q_wp, q_wc, q_wp_rest, q_wc_rest)

    # Compute world-space Jacobian: J_world = R_wp * Jr^{-1}^T
    # This maps rotation-vector forces (kappa space) to world-space angular torques
    Jr_inv = compute_right_jacobian_inverse(kappa_now)
    R_wp = wp.quat_to_matrix(q_wp)
    J_world = R_wp * wp.transpose(Jr_inv)

    # Elastic torque: tau = J_world * (k_eff * kappa)
    f_local = k_eff * kappa_now
    tau_world = J_world * f_local

    # Flip sign for child body (action-reaction pair)
    if not is_parent:
        tau_world = -tau_world

    # Gauss-Newton Hessian (symmetric positive definite)
    # H = k_eff * J * J^T
    H_aa = k_eff * (J_world * wp.transpose(J_world))

    # Rayleigh damping (consistent with VBD cloth bending)
    if damping > 0.0:
        # Compute rotation-vector velocity: dkappa/dt (finite difference)
        kappa_prev = cable_get_kappa(q_wp_prev, q_wc_prev, q_wp_rest, q_wc_rest)
        inv_dt = 1.0 / dt
        dkappa_dt = (kappa_now - kappa_prev) * inv_dt

        # Damping torque: tau_d = J_world * (-D * dkappa/dt)
        # where D = damping * k_eff (damping coefficient)
        f_damp_local = -(damping * k_eff) * dkappa_dt
        tau_damp_world = J_world * f_damp_local
        if not is_parent:
            tau_damp_world = -tau_damp_world

        # Damping Hessian: H_d = (D/dt) * H = (damping/dt) * H_aa
        H_aa_damp = (damping * inv_dt) * H_aa

        # Accumulate damping contributions
        tau_world = tau_world + tau_damp_world
        H_aa = H_aa + H_aa_damp

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
    Compute AVBD cable stretch force and Hessian using simple pinning constraint.

    Physics:
        Pinning constraint: C = x_c - x_p (displacement between attachment points)
        Energy: E = (1/2) * k * ||C||^2
        Force: f = k * C (penalty force)

    Implementation:
        1. Compute constraint violation: C = x_c - x_p
        2. Compute elastic force: f_point = k * C
        3. Map to body forces/torques using moment arm r
        4. Compute Gauss-Newton Hessian: H = J^T * K * J where J = [-[r]_x, I]
        5. Add Rayleigh damping: f_d = -D * dC/dt, H_d = (D/dt) * H

    Note:
        This is a simple 3D pinning constraint.
        Used when use_cosserat_stretch = False.

    Args:
        X_wp: Parent joint frame (world space)
        X_wc: Child joint frame (world space)
        X_wp_prev: Parent joint frame previous step (for damping)
        X_wc_prev: Child joint frame previous step (for damping)
        parent_pose: Parent body pose (for COM)
        child_pose: Child body pose (for COM)
        parent_com: Parent COM in body frame
        child_com: Child COM in body frame
        is_parent: True for parent body, False for child
        penalty_k: Penalty parameter [N/m] (linear spring stiffness)
        damping: Rayleigh damping coefficient (dimensionless)
        dt: Time step [s]

    Returns:
        tuple: (force, torque, H_ll, H_al, H_aa)
            - force: Linear force on body [N] (3D vector)
            - torque: Angular torque on body [N*m] (3D vector)
            - H_ll: Linear-linear Hessian block [N/m] (3x3 SPD)
            - H_al: Angular-linear coupling Hessian [N/rad] (3x3)
            - H_aa: Angular-angular Hessian block [N*m/rad] (3x3 SPD)
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

    # Elastic force at attachment point: f_point = k * C
    f_point = penalty_k * C_vec

    # Map to body linear force (sign flip for child)
    f_lin = f_point if is_parent else -f_point

    # Body torque: tau = r x f_lin
    torque = wp.cross(r, f_lin)

    # Gauss-Newton Hessian: H = J^T * K * J
    # where J = [-[r]_x, I] is the contact-point Jacobian
    # and K = k * I is the point stiffness
    rx = wp.skew(r)
    K_point = penalty_k * wp.identity(3, float)

    H_ll = K_point
    H_al = rx * K_point
    H_aa = wp.transpose(rx) * K_point * rx

    force = f_lin

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

        # Damping force at attachment point: f_damp_point = -D * dC/dt
        f_damp_point = -damping_coeff * dC_dt
        f_damp_lin = f_damp_point if is_parent else -f_damp_point

        # Damping torque: tau_damp = r x f_damp_lin
        torque_damp = wp.cross(r, f_damp_lin)

        # Damping Hessian: H_damp = (D/dt) * H_elastic = (damping/dt) * H_elastic
        damp_scale = damping * inv_dt
        H_ll_damp = damp_scale * K_point
        H_al_damp = damp_scale * (rx * K_point)
        H_aa_damp = damp_scale * (wp.transpose(rx) * K_point * rx)

        # Accumulate damping contributions
        force = force + f_damp_lin
        torque = torque + torque_damp
        H_ll = H_ll + H_ll_damp
        H_al = H_al + H_al_damp
        H_aa = H_aa + H_aa_damp

    return force, torque, H_ll, H_al, H_aa


@wp.func
def evaluate_cable_stretch_force_hessian_cosserat(
    parent_pose: wp.transform,
    child_pose: wp.transform,
    parent_pose_prev: wp.transform,
    child_pose_prev: wp.transform,
    parent_com: wp.vec3,
    child_com: wp.vec3,
    is_parent: bool,
    penalty_k: float,
    rest_length: float,
    damping: float,
    dt: float,
):
    """
    Compute AVBD Cosserat rod stretch/shear force and Hessian.

    Physics (Cosserat rod theory):
        Constraint (dimensionless): C = d/S - director
        where:
            - d = x_c - x_p: displacement between body origins
            - S: segment rest length
            - director: parent body +Z axis (unit vector)

        In rest state: d = S * director => C = 0

        Energy: E = (1/2) * k * ||C||^2  [J]
        Lagrange multiplier: lambda = k * C  [N*m] (torque per dimensionless violation)
        Linear force: f = (1/S) * lambda  [N]
        Hessian: H = k * (1/S^2)  [N/m]

    Implementation:
        1. Compute director: t = parent_+Z_axis (world frame)
        2. Compute constraint: C = d/S - t
        3. Compute lambda: lambda = k * C
        4. Compute forces:
           - Parent: f_p = +(1/S) * lambda, tau_p = r_p x f_p + t x lambda
           - Child:  f_c = -(1/S) * lambda, tau_c = r_c x f_c
        5. Compute Hessian: H = J^T * k * J with 1/S and 1/S^2 scaling
        6. Add Rayleigh damping: f_d = -D * dC/dt, H_d = (D/dt) * H

    Args:
        parent_pose: Parent body pose (world space)
        child_pose: Child body pose (world space)
        parent_pose_prev: Parent body pose previous step (for damping)
        child_pose_prev: Child body pose previous step (for damping)
        parent_com: Parent COM in body frame
        child_com: Child COM in body frame
        is_parent: True for parent body, False for child
        penalty_k: Penalty parameter [N*m] (torque per dimensionless C)
        rest_length: Segment rest length S [m]
        damping: Rayleigh damping coefficient (dimensionless)
        dt: Time step [s]

    Returns:
        tuple: (force, torque, H_ll, H_al, H_aa)
            - force: Linear force on body [N] (3D vector)
            - torque: Angular torque on body [N*m] (3D vector)
            - H_ll: Linear-linear Hessian block [N/m] (3x3 SPD)
            - H_al: Angular-linear coupling Hessian [N/rad] (3x3)
            - H_aa: Angular-angular Hessian block [N*m/rad] (3x3 SPD)
    """
    # Geometry: use BODY ORIGINS (Cosserat nodes)
    x_p = wp.transform_get_translation(parent_pose)
    x_c = wp.transform_get_translation(child_pose)
    d = x_c - x_p

    # Normalize by rest length
    # rest_length is guaranteed > 1e-6 by caller's early-exit check, no need to clamp
    inv_S = 1.0 / rest_length
    inv_S_sqr = inv_S * inv_S

    # Director: material frame +Z axis in world space (from parent body orientation)
    q_p = wp.transform_get_rotation(parent_pose)
    t = quat_rotate_z_axis(q_p)  # Optimized rotation of +Z axis

    # Constraint violation (dimensionless): C = d/S - t
    # In rest state: d = S * t => C = t - t = 0
    C = d * inv_S - t

    # Penalty parameter (scalar, units [N*m])
    # Clamp to avoid negative penalties from numerical errors
    k = wp.max(penalty_k, 0.0)

    # Lagrange multiplier: lambda = k * C (avoid building K matrix)
    # lambda has units [N*m], so f = (1/S) * lambda has units [N]
    lam = k * C

    # Forces and torques for this body
    if is_parent:
        # Parent body
        com_w = wp.transform_point(parent_pose, parent_com)
        r = x_p - com_w  # Lever arm from COM to body origin

        # Linear force: f_p = +(1/S) * lambda
        # Derivation: dC/dx_p = -1/S * I => f_p = (1/S) * lambda
        f_lin = inv_S * lam

        # Orientational torque: tau_q = t x lambda (director wrench on parent)
        tau_q = wp.cross(t, lam)
    else:
        # Child body
        com_w = wp.transform_point(child_pose, child_com)
        r = x_c - com_w  # Lever arm from COM to body origin

        # Linear force: f_c = -(1/S) * lambda
        # Derivation: dC/dx_c = +1/S * I => f_c = -(1/S) * lambda
        f_lin = -inv_S * lam

        # No orientational torque on child (director lives on parent)
        tau_q = wp.vec3(0.0, 0.0, 0.0)

    force = f_lin
    torque = wp.cross(r, f_lin) + tau_q

    # Gauss-Newton SPD Hessian: H = J^T K J
    # Jacobians:
    #   J_lin = (1/S) * I (positional part)
    #   J_q = -[t]_x (orientational part, parent only)

    # Factor out k (avoid building K matrix)
    rx = wp.skew(r)

    # H_ll: Positional-positional block
    # J_lin^T K J_lin = k * (1/S^2) * I
    I3 = wp.identity(3, float)
    H_ll = k * inv_S_sqr * I3

    # H_al (H_la^T): Angular-linear coupling
    # For parent: includes both positional and orientational contributions
    if is_parent:
        # Compute tx only for parent (optimization)
        tx = wp.skew(t)
        # H_la = -k * (rx * (1/S^2) + tx * (1/S))
        H_la = -k * (rx * inv_S_sqr + tx * inv_S)
    else:
        # No orientational coupling for child
        H_la = -k * (rx * inv_S_sqr)
    H_al = wp.transpose(H_la)

    # H_aa: Angular-angular block
    # Positional contribution: k * rx^T rx * (1/S^2)
    H_aa = k * inv_S_sqr * (wp.transpose(rx) * rx)

    if is_parent:
        # Orientational contribution: k * tx^T tx
        H_aa += k * (wp.transpose(tx) * tx)
        # Cross-coupling: k * (rx^T tx + tx^T rx) * (1/S)
        H_aa += k * inv_S * (wp.transpose(rx) * tx + wp.transpose(tx) * rx)

    # Rayleigh-style damping
    if damping > 0.0:
        inv_dt = 1.0 / dt

        # Compute velocity of constraint violation
        x_p_prev = wp.transform_get_translation(parent_pose_prev)
        x_c_prev = wp.transform_get_translation(child_pose_prev)
        d_prev = x_c_prev - x_p_prev
        q_p_prev = wp.transform_get_rotation(parent_pose_prev)
        t_prev = quat_rotate_z_axis(q_p_prev)
        C_prev = d_prev * inv_S - t_prev
        dC_dt = (C - C_prev) * inv_dt

        # Damping coefficient proportional to stiffness (Rayleigh damping)
        damping_coeff = damping * k

        # Damping force: lambda_damp = -D * dC/dt where D = beta * K
        lam_damp = -damping_coeff * dC_dt

        # Linear damping force
        f_damp_lin = inv_S * lam_damp if is_parent else -inv_S * lam_damp

        # Damping torque
        tau_q_damp = wp.cross(t, lam_damp) if is_parent else wp.vec3(0.0, 0.0, 0.0)
        torque_damp = wp.cross(r, f_damp_lin) + tau_q_damp

        # Damping Hessian: H_d = D/dt = (damping/dt) * K_elastic
        damp_scale = damping * inv_dt
        H_ll_damp = damp_scale * k * inv_S_sqr * I3

        if is_parent:
            H_la_damp = -damp_scale * k * (rx * inv_S_sqr + tx * inv_S)
            H_aa_damp = damp_scale * k * inv_S_sqr * (wp.transpose(rx) * rx)
            H_aa_damp += damp_scale * k * (wp.transpose(tx) * tx)
            H_aa_damp += damp_scale * k * inv_S * (wp.transpose(rx) * tx + wp.transpose(tx) * rx)
        else:
            H_la_damp = -damp_scale * k * (rx * inv_S_sqr)
            H_aa_damp = damp_scale * k * inv_S_sqr * (wp.transpose(rx) * rx)

        H_al_damp = wp.transpose(H_la_damp)

        force_damp = f_damp_lin

        # Add damping contributions
        force = force + force_damp
        torque = torque + torque_damp
        H_ll = H_ll + H_ll_damp
        H_al = H_al + H_al_damp
        H_aa = H_aa + H_aa_damp

    return force, torque, H_ll, H_al, H_aa


@wp.func
def evaluate_joint_force_hessian(
    body_index: int,
    joint_index: int,
    body_q: wp.array(dtype=wp.transform),
    body_q_prev: wp.array(dtype=wp.transform),
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
    dt: float,
):
    """
    Compute AVBD force and Hessian contributions from a joint on a specific body.

    Currently supports cable joints only (2 DOFs: stretch + bend).
    Cable joints use AVBD soft constraints with adaptive penalty parameters.

    Physics:
        Stretch: C = displacement (simple pinning) or C = d/S - director (Cosserat)
        Bend: kappa = rotation vector between segments (Cosserat rod theory)

        Both use penalty formulation:
            Force: f = k * C (or tau = J * k * kappa for bend)
            Hessian: H = J^T * k * J (Gauss-Newton)
            Damping: f_d = -D * dC/dt, H_d = (D/dt) * H (Rayleigh)

    Args:
        body_index: Body to compute forces for
        joint_index: Joint constraint index
        body_q: Current body transforms
        body_q_prev: Previous body transforms (for damping)
        body_q_rest: Rest body transforms (for bend constraint)
        body_com: Body COM in local frame
        joint_type: Joint type
        joint_parent: Parent body index per joint
        joint_child: Child body index per joint
        joint_X_p: Parent joint frame (local)
        joint_X_c: Child joint frame (local)
        joint_qd_start: DOF start index per joint
        joint_target_kd: Damping coefficient per DOF
        joint_penalty_k: AVBD adaptive penalty per DOF
        dt: Time step [s]

    Returns:
        tuple: (force, torque, H_ll, H_al, H_aa)
            - force: Linear force [N] (3D vector)
            - torque: Angular torque [N*m] (3D vector)
            - H_ll: Linear-linear Hessian [N/m] (3x3 SPD)
            - H_al: Angular-linear Hessian [N/rad] (3x3)
            - H_aa: Angular-angular Hessian [N*m/rad] (3x3 SPD)
    """
    # Early exit: only cable joints supported
    if joint_type[joint_index] != JointType.CABLE:
        return wp.vec3(0.0), wp.vec3(0.0), wp.mat33(0.0), wp.mat33(0.0), wp.mat33(0.0)

    # Get parent and child body indices
    parent_index = joint_parent[joint_index]
    child_index = joint_child[joint_index]

    # Early exit: invalid parent or body not connected to this joint
    if parent_index < 0 or (body_index != parent_index and body_index != child_index):
        return wp.vec3(0.0), wp.vec3(0.0), wp.mat33(0.0), wp.mat33(0.0), wp.mat33(0.0)

    # Determine if this body is the parent
    is_parent_body = body_index == parent_index

    # Read joint configuration
    X_pj = joint_X_p[joint_index]
    X_cj = joint_X_c[joint_index]

    # Read body state
    parent_pose = body_q[parent_index]
    child_pose = body_q[child_index]
    parent_pose_prev = body_q_prev[parent_index]
    child_pose_prev = body_q_prev[child_index]
    parent_pose_rest = body_q_rest[parent_index]
    child_pose_rest = body_q_rest[child_index]
    parent_com = body_com[parent_index]
    child_com = body_com[child_index]

    # Transform joint frames to world space
    X_wp = parent_pose * X_pj
    X_wc = child_pose * X_cj
    X_wp_prev = parent_pose_prev * X_pj
    X_wc_prev = child_pose_prev * X_cj
    X_wp_rest = parent_pose_rest * X_pj
    X_wc_rest = child_pose_rest * X_cj

    # Read cable constraint parameters (2 DOFs: stretch, bend)
    dof_start = joint_qd_start[joint_index]
    stretch_penalty_k = joint_penalty_k[dof_start]
    bend_penalty_k = joint_penalty_k[dof_start + 1]
    stretch_damping = joint_target_kd[dof_start]
    bend_damping = joint_target_kd[dof_start + 1]

    # Compute rest segment length from parent body capsule
    # For rod mesh: rest_length = full capsule height = 2 * parent_com[2]
    rest_length = 2.0 * parent_com[2]

    # Early exit for degenerate segments (< 1 nm threshold)
    if rest_length < 1.0e-9:
        return wp.vec3(0.0), wp.vec3(0.0), wp.mat33(0.0), wp.mat33(0.0), wp.mat33(0.0)

    # Compute cable constraints
    total_force = wp.vec3(0.0)
    total_torque = wp.vec3(0.0)
    total_H_ll = wp.mat33(0.0)
    total_H_al = wp.mat33(0.0)
    total_H_aa = wp.mat33(0.0)

    # Bend constraint (rotation-based, angular components only)
    if bend_penalty_k > 0.0:
        # Extract quaternions from joint frames
        q_wp = wp.transform_get_rotation(X_wp)
        q_wc = wp.transform_get_rotation(X_wc)
        q_wp_rest = wp.transform_get_rotation(X_wp_rest)
        q_wc_rest = wp.transform_get_rotation(X_wc_rest)
        q_wp_prev = wp.transform_get_rotation(X_wp_prev)
        q_wc_prev = wp.transform_get_rotation(X_wc_prev)

        # Pre-scale stiffness: k_eff = EI / L (Cosserat rod theory)
        bend_k_eff = bend_penalty_k / rest_length

        bend_torque, bend_H_aa = evaluate_cable_bend_force_hessian_avbd(
            q_wp, q_wc, q_wp_rest, q_wc_rest, q_wp_prev, q_wc_prev, is_parent_body, bend_k_eff, bend_damping, dt
        )

        total_torque = total_torque + bend_torque
        total_H_aa = total_H_aa + bend_H_aa

    # Stretch constraint (switchable: Cosserat vs simple pinning)
    if stretch_penalty_k > 0.0:
        if use_cosserat_stretch:
            # Cosserat: C = d/S - director (dimensionless, shear-aware)
            f_s, t_s, Hll_s, Hal_s, Haa_s = evaluate_cable_stretch_force_hessian_cosserat(
                parent_pose,
                child_pose,
                parent_pose_prev,
                child_pose_prev,
                parent_com,
                child_com,
                is_parent_body,
                stretch_penalty_k,
                rest_length,
                stretch_damping,
                dt,
            )
        else:
            # Simple pinning: C = x_c - x_p (dimensional, faster)
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


class SolverAVBD(SolverBase):
    """Augmented VBD solver for rigid bodies (joints + rigid contacts)."""

    def __init__(
        self,
        model: Model,
        iterations: int = 10,
        friction_epsilon: float = 1e-2,
        avbd_beta: float = 1.0e5,
        avbd_gamma: float = 0.99,
        penalty_min: float = 1.0e6,
        penalty_max: float = 1.0e18,
        k_start: float | None = None,
        body_contact_capacity: int = 64,
    ):
        """
        Args:
            model: The `Model` associated with this solver. Must be the same object passed to `step`.
            iterations: Number of AVBD iterations per step.
            friction_epsilon: Friction regularization epsilon used in contact (velocity smoothing threshold).
            avbd_beta: Penalty ramp rate (how fast k grows with constraint violation).
            avbd_gamma: Warmstart decay for penalty k (cross-step decay factor).
            penalty_min: Lower clamp for penalty k (applied in warmstart).
            penalty_max: Upper clamp for penalty k (applied in dual updates).
            k_start: Optional initial floor for penalty k. If provided, broadcast to all DOFs; if None, zeros are used
                and the effective floor comes from `penalty_min`.
            body_contact_capacity: Max contacts per body for per-body contact lists (tune based on contact density).
        """
        super().__init__(model)
        self.iterations = iterations

        # Rigid body storage for forward stepping
        self.body_q_prev = wp.zeros_like(model.body_q, device=self.device)
        self.body_inertia_q = wp.zeros_like(model.body_q, device=self.device)

        self.adjacency = self.compute_force_element_adjacency(model).to(self.device)

        self.rigid_contact_launch_size = model.shape_count * model.shape_count

        # Store torques and forces separately for better performance
        self.body_torques = wp.zeros(self.model.body_count, dtype=wp.vec3, device=self.device)
        self.body_forces = wp.zeros(self.model.body_count, dtype=wp.vec3, device=self.device)

        # Collision Hessian blocks
        self.body_hessian_aa = wp.zeros(self.model.body_count, dtype=wp.mat33, device=self.device)  # Angular-angular
        self.body_hessian_al = wp.zeros(self.model.body_count, dtype=wp.mat33, device=self.device)  # Angular-linear
        self.body_hessian_ll = wp.zeros(self.model.body_count, dtype=wp.mat33, device=self.device)  # Linear-linear

        # Per-body contact lists (fixed capacity, GPU-only)
        # Each body has a fixed-size buffer to store contact indices
        self.body_contact_capacity = body_contact_capacity
        self.body_contact_counts = wp.zeros(self.model.body_count, dtype=wp.int32, device=self.device)
        self.body_contact_indices = wp.zeros(
            self.model.body_count * self.body_contact_capacity, dtype=wp.int32, device=self.device
        )

        self.friction_epsilon = friction_epsilon

        # AVBD algorithm parameters
        self.avbd_beta = avbd_beta
        self.avbd_gamma = avbd_gamma

        # User-configurable AVBD penalty bounds
        self.penalty_min = penalty_min
        self.penalty_max = penalty_max

        # Basic validation for user-configured penalty bounds
        if self.penalty_min <= 0.0:
            raise ValueError("penalty_min must be > 0.0")
        if self.penalty_max < self.penalty_min:
            raise ValueError("penalty_max must be >= penalty_min")

        # AVBD constraint state: adaptive penalty parameters (soft constraints only)
        dof_count = self.model.joint_dof_count
        self.joint_penalty_k = wp.full((dof_count,), self.penalty_min, dtype=float, device=self.device)

        max_contacts = self.rigid_contact_launch_size
        self.contact_penalty_k = wp.full((max_contacts,), self.penalty_min, dtype=float, device=self.device)

        # Pre-computed averaged material properties (computed once per step in warmstart, reused in all iterations)
        self.contact_material_ke = wp.zeros(max_contacts, dtype=float, device=self.device)
        self.contact_material_kd = wp.zeros(max_contacts, dtype=float, device=self.device)
        self.contact_material_mu = wp.zeros(max_contacts, dtype=float, device=self.device)

        # Check that we have coloring information for rigid bodies
        has_bodies = self.model.body_count > 0
        has_body_coloring = len(self.model.body_color_groups) > 0

        if not has_bodies:
            raise ValueError("Model has no rigid bodies! SolverAVBD is rigid-only and requires at least one body.")

        if has_bodies and not has_body_coloring:
            raise ValueError(
                "model.body_color_groups is empty but rigid bodies are present! When using the SolverAVBD you must call ModelBuilder.color() "
                "or ModelBuilder.set_coloring() before calling ModelBuilder.finalize()."
            )

    def compute_force_element_adjacency(self, model):
        adjacency = ForceElementAdjacencyInfo()

        with wp.ScopedDevice("cpu"):
            # Build body-joint adjacency data (rigid-only)
            if model.joint_count > 0:
                joint_parent_cpu = model.joint_parent.to("cpu")
                joint_child_cpu = model.joint_child.to("cpu")

                num_body_adjacent_joints = wp.zeros(shape=(model.body_count,), dtype=wp.int32)
                wp.launch(
                    kernel=self.count_num_adjacent_joints,
                    inputs=[joint_parent_cpu, joint_child_cpu, num_body_adjacent_joints],
                    dim=1,
                )

                num_body_adjacent_joints = num_body_adjacent_joints.numpy()
                body_adjacent_joints_offsets = np.empty(shape=(model.body_count + 1,), dtype=wp.int32)
                body_adjacent_joints_offsets[1:] = np.cumsum(num_body_adjacent_joints)[:]
                body_adjacent_joints_offsets[0] = 0
                adjacency.body_adj_joints_offsets = wp.array(body_adjacent_joints_offsets, dtype=wp.int32)

                body_adjacent_joints_fill_count = wp.zeros(shape=(model.body_count,), dtype=wp.int32)
                adjacency.body_adj_joints = wp.empty(shape=(num_body_adjacent_joints.sum(),), dtype=wp.int32)

                wp.launch(
                    kernel=self.fill_adjacent_joints,
                    inputs=[
                        joint_parent_cpu,
                        joint_child_cpu,
                        adjacency.body_adj_joints_offsets,
                        body_adjacent_joints_fill_count,
                        adjacency.body_adj_joints,
                    ],
                    dim=1,
                )
            else:
                # No joints: create offset array of zeros (size body_count + 1) so indexing works
                adjacency.body_adj_joints_offsets = wp.zeros(shape=(model.body_count + 1,), dtype=wp.int32)
                adjacency.body_adj_joints = wp.empty(shape=(0,), dtype=wp.int32)

        return adjacency

    @override
    def step(self, state_in: State, state_out: State, control: Control, contacts: Contacts, dt: float):
        model = self.model
        body_color_groups = self.model.body_color_groups

        # Forward integrate rigid bodies
        wp.launch(
            kernel=forward_step_rigid_bodies,
            inputs=[
                dt,
                model.gravity,
                self.body_q_prev,
                state_in.body_q,
                state_in.body_qd,
                state_in.body_f,
                model.body_com,
                model.body_inertia,
                model.body_inv_mass,
                model.body_inv_inertia,
                self.body_inertia_q,
            ],
            dim=self.model.body_count,
            device=self.device,
        )

        # Joint penalty handling: AVBD (adaptive) vs VBD (fixed)
        if use_avbd:
            # Warmstart AVBD penalty parameters for joints
            wp.launch(
                kernel=avbd_warmstart_joints,
                inputs=[
                    self.joint_penalty_k,
                    self.model.joint_target_ke,
                    self.avbd_gamma,
                    self.penalty_min,
                    self.penalty_max,
                ],
                dim=self.model.joint_dof_count,
                device=self.device,
            )
        # Get actual contact count for optimal kernel launch (avoids wasted threads)
        # Note: rigid_contact_count is a device array, so we use it as upper bound in kernel
        # The kernels check `if i >= rigid_contact_count[0]: return` for early exit
        # We use self.rigid_contact_launch_size as a conservative upper bound
        contact_launch_dim = self.rigid_contact_launch_size

        # Warmstart contact penalties and/or pre-compute material properties
        if use_avbd:
            wp.launch(
                kernel=avbd_warmstart_contacts,
                inputs=[
                    contacts.rigid_contact_count,
                    contacts.rigid_contact_shape0,
                    contacts.rigid_contact_shape1,
                    model.shape_material_ke,
                    model.shape_material_kd,
                    model.shape_material_mu,
                    self.avbd_gamma,
                    self.penalty_min,
                    self.penalty_max,
                ],
                outputs=[
                    self.contact_penalty_k,
                    self.contact_material_ke,
                    self.contact_material_kd,
                    self.contact_material_mu,
                ],
                dim=contact_launch_dim,
                device=self.device,
            )
        else:
            # VBD soft: compute material properties and initialize penalty to material stiffness
            wp.launch(
                kernel=compute_contact_material_properties,
                inputs=[
                    contacts.rigid_contact_count,
                    contacts.rigid_contact_shape0,
                    contacts.rigid_contact_shape1,
                    model.shape_material_ke,
                    model.shape_material_kd,
                    model.shape_material_mu,
                ],
                outputs=[
                    self.contact_material_ke,
                    self.contact_material_kd,
                    self.contact_material_mu,
                ],
                dim=contact_launch_dim,
                device=self.device,
            )
            # VBD mode: pass contact_material_ke directly (no assign needed)

        # Build per-body contact lists once per step (GPU-only, no CPU sync)
        self.body_contact_counts.zero_()
        wp.launch(
            kernel=build_body_contact_lists,
            dim=contact_launch_dim,
            inputs=[
                contacts.rigid_contact_count,
                contacts.rigid_contact_shape0,
                contacts.rigid_contact_shape1,
                self.model.shape_body,
                self.body_contact_capacity,
                self.body_contact_counts,
                self.body_contact_indices,
            ],
            device=self.device,
        )

        for _iter in range(self.iterations):
            self.body_torques.zero_()
            self.body_forces.zero_()
            self.body_hessian_aa.zero_()
            self.body_hessian_al.zero_()
            self.body_hessian_ll.zero_()

            # Select stiffness arrays based on mode (compile-time decision)
            # AVBD mode: use adaptive penalties; VBD mode: use fixed material stiffness
            contact_stiffness_array = self.contact_penalty_k if use_avbd else self.contact_material_ke
            joint_stiffness_array = self.joint_penalty_k if use_avbd else self.model.joint_target_ke

            for color in range(len(body_color_groups)):
                color_group = body_color_groups[color]
                # Per-body accumulation: NUM_THREADS_PER_BODY threads per body
                wp.launch(
                    kernel=accumulate_rigid_contact_per_body,
                    dim=color_group.size * NUM_THREADS_PER_BODY,
                    inputs=[
                        dt,
                        color_group,
                        self.body_q_prev,
                        state_in.body_q,
                        self.model.body_com,
                        self.model.body_inv_mass,
                        self.friction_epsilon,
                        contact_stiffness_array,
                        self.contact_material_kd,
                        self.contact_material_mu,
                        contacts.rigid_contact_count,
                        contacts.rigid_contact_shape0,
                        contacts.rigid_contact_shape1,
                        contacts.rigid_contact_point0,
                        contacts.rigid_contact_point1,
                        contacts.rigid_contact_normal,
                        contacts.rigid_contact_thickness0,
                        contacts.rigid_contact_thickness1,
                        model.shape_body,
                        self.body_contact_capacity,
                        self.body_contact_counts,
                        self.body_contact_indices,
                    ],
                    outputs=[
                        self.body_forces,
                        self.body_torques,
                        self.body_hessian_ll,
                        self.body_hessian_al,
                        self.body_hessian_aa,
                    ],
                    device=self.device,
                )

                wp.launch(
                    kernel=solve_rigid_body,
                    inputs=[
                        dt,
                        color_group,
                        state_in.body_q,
                        self.body_q_prev,
                        self.model.body_q,
                        self.model.body_mass,
                        self.model.body_inv_mass,
                        self.model.body_inertia,
                        self.body_inertia_q,
                        self.model.body_com,
                        self.adjacency,
                        self.model.joint_type,
                        self.model.joint_parent,
                        self.model.joint_child,
                        self.model.joint_X_p,
                        self.model.joint_X_c,
                        self.model.joint_qd_start,
                        self.model.joint_target_kd,
                        joint_stiffness_array,
                        self.body_forces,
                        self.body_torques,
                        self.body_hessian_ll,
                        self.body_hessian_al,
                        self.body_hessian_aa,
                    ],
                    outputs=[
                        state_out.body_q,
                    ],
                    dim=color_group.size,
                    device=self.device,
                )

                wp.launch(
                    kernel=copy_rigid_body_transforms_back,
                    inputs=[color_group, state_in.body_q],
                    outputs=[state_out.body_q],
                    dim=color_group.size,
                    device=self.device,
                )

            # AVBD dual update: update adaptive penalties based on constraint violation
            if use_avbd:
                # Update contact penalties
                wp.launch(
                    kernel=avbd_update_duals_contact,
                    dim=contact_launch_dim,
                    inputs=[
                        contacts.rigid_contact_count,
                        contacts.rigid_contact_shape0,
                        contacts.rigid_contact_shape1,
                        contacts.rigid_contact_point0,
                        contacts.rigid_contact_point1,
                        contacts.rigid_contact_normal,
                        contacts.rigid_contact_thickness0,
                        contacts.rigid_contact_thickness1,
                        model.shape_body,
                        state_out.body_q,
                        self.contact_penalty_k,
                        self.contact_material_ke,
                        self.avbd_beta,
                        self.penalty_max,
                    ],
                    device=self.device,
                )

                # Update joint penalties at NEW positions
                wp.launch(
                    kernel=avbd_update_duals_joint,
                    dim=self.model.joint_count,
                    inputs=[
                        self.model.joint_type,
                        self.model.joint_parent,
                        self.model.joint_child,
                        self.model.joint_X_p,
                        self.model.joint_X_c,
                        self.model.joint_qd_start,
                        self.model.joint_dof_dim,
                        self.model.joint_target_ke,
                        state_out.body_q,
                        self.model.body_q,
                        self.model.body_com,
                        self.joint_penalty_k,
                        self.avbd_beta,
                        self.penalty_max,
                    ],
                    device=self.device,
                )

        # Velocity update (BDF1) after all iterations
        wp.launch(
            kernel=update_body_velocity,
            inputs=[dt, state_out.body_q, self.body_q_prev, self.model.body_com],
            outputs=[state_out.body_qd],
            dim=model.body_count,
            device=self.device,
        )

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


@wp.func
def get_both_bodies_from_contact_with_thickness(
    t_id: int,
    rigid_contact_shape0: wp.array(dtype=int),
    rigid_contact_shape1: wp.array(dtype=int),
    rigid_contact_point0: wp.array(dtype=wp.vec3),
    rigid_contact_point1: wp.array(dtype=wp.vec3),
    rigid_contact_thickness0: wp.array(dtype=float),
    rigid_contact_thickness1: wp.array(dtype=float),
    shape_body: wp.array(dtype=wp.int32),
):
    """Extract both bodies from a rigid contact pair."""
    shape_id_0 = rigid_contact_shape0[t_id]
    shape_id_1 = rigid_contact_shape1[t_id]

    body_id_0 = shape_body[shape_id_0] if shape_id_0 >= 0 else -1
    body_id_1 = shape_body[shape_id_1] if shape_id_1 >= 0 else -1

    return (
        body_id_0,
        shape_id_0,
        rigid_contact_point0[t_id],
        rigid_contact_thickness0[t_id],
        body_id_1,
        shape_id_1,
        rigid_contact_point1[t_id],
        rigid_contact_thickness1[t_id],
    )


@wp.kernel
def build_body_contact_lists(
    rigid_contact_count: wp.array(dtype=int),
    rigid_contact_shape0: wp.array(dtype=int),
    rigid_contact_shape1: wp.array(dtype=int),
    shape_body: wp.array(dtype=wp.int32),
    body_contact_capacity: int,
    body_contact_counts: wp.array(dtype=wp.int32),
    body_contact_indices: wp.array(dtype=wp.int32),
):
    """
    Build per-body contact lists. Each contact is added to both bodies' lists.
    Simple version: just store contact index (not packed).
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
        if idx < body_contact_capacity:
            body_contact_indices[b0 * body_contact_capacity + idx] = t_id

    # Add contact to body1's list
    if b1 >= 0:
        idx = wp.atomic_add(body_contact_counts, b1, 1)
        if idx < body_contact_capacity:
            body_contact_indices[b1 * body_contact_capacity + idx] = t_id


@wp.kernel
def accumulate_rigid_contact_per_body(
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
    rigid_contact_count: wp.array(dtype=int),
    rigid_contact_shape0: wp.array(dtype=int),
    rigid_contact_shape1: wp.array(dtype=int),
    rigid_contact_point0: wp.array(dtype=wp.vec3),
    rigid_contact_point1: wp.array(dtype=wp.vec3),
    rigid_contact_normal: wp.array(dtype=wp.vec3),
    rigid_contact_thickness0: wp.array(dtype=float),
    rigid_contact_thickness1: wp.array(dtype=float),
    shape_body: wp.array(dtype=wp.int32),
    body_contact_capacity: int,
    body_contact_counts: wp.array(dtype=wp.int32),
    body_contact_indices: wp.array(dtype=wp.int32),
    body_forces: wp.array(dtype=wp.vec3),
    body_torques: wp.array(dtype=wp.vec3),
    body_hessian_ll: wp.array(dtype=wp.mat33),
    body_hessian_al: wp.array(dtype=wp.mat33),
    body_hessian_aa: wp.array(dtype=wp.mat33),
):
    """
    Per-body contact accumulation with multi-threading.
    NUM_THREADS_PER_BODY threads cooperatively process one body's contacts using strided iteration.
    Much more efficient than global scan when each body has few contacts.
    """
    tid = wp.tid()

    # Determine which body and which thread within that body
    body_idx_in_group = tid // NUM_THREADS_PER_BODY
    thread_id_within_body = tid % NUM_THREADS_PER_BODY

    if body_idx_in_group >= color_group.shape[0]:
        return

    body_id = color_group[body_idx_in_group]
    if body_inv_mass[body_id] <= 0.0:
        return

    # Get this body's contact range
    num_contacts = body_contact_counts[body_id]
    if num_contacts > body_contact_capacity:
        num_contacts = body_contact_capacity  # Clamp to capacity

    # Strided iteration: each thread processes every NUM_THREADS_PER_BODY-th contact
    i = thread_id_within_body
    while i < num_contacts:
        contact_idx = body_contact_indices[body_id * body_contact_capacity + i]

        # Bounds check
        if contact_idx >= rigid_contact_count[0]:
            i += NUM_THREADS_PER_BODY
            continue

        # Get contact data
        s0 = rigid_contact_shape0[contact_idx]
        s1 = rigid_contact_shape1[contact_idx]
        b0 = shape_body[s0] if s0 >= 0 else -1
        b1 = shape_body[s1] if s1 >= 0 else -1

        # Determine which body we are (0 or 1)
        if b0 != body_id and b1 != body_id:
            i += NUM_THREADS_PER_BODY
            continue  # Safety: contact doesn't involve this body

        # Recompute penetration at current positions
        cp0_local = rigid_contact_point0[contact_idx]
        cp1_local = rigid_contact_point1[contact_idx]
        contact_normal = -rigid_contact_normal[contact_idx]

        cp0_world = wp.transform_point(body_q[b0], cp0_local) if b0 >= 0 else cp0_local
        cp1_world = wp.transform_point(body_q[b1], cp1_local) if b1 >= 0 else cp1_local
        thickness = rigid_contact_thickness0[contact_idx] + rigid_contact_thickness1[contact_idx]
        dist = wp.dot(contact_normal, cp1_world - cp0_world)
        penetration = wp.max(0.0, thickness - dist)

        if penetration <= 1.0e-9:
            i += NUM_THREADS_PER_BODY
            continue

        contact_ke = contact_penalty_k[contact_idx]
        contact_kd = contact_material_kd[contact_idx]
        contact_mu = contact_material_mu[contact_idx]

        # Compute forces for both bodies
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
            penetration,
            contact_ke,
            contact_kd,
            contact_mu,
            friction_epsilon,
            dt,
        )

        # Accumulate only this body's side (atomics needed due to parallel threads)
        if body_id == b0:
            wp.atomic_add(body_forces, body_id, force_0)
            wp.atomic_add(body_torques, body_id, torque_0)
            wp.atomic_add(body_hessian_ll, body_id, h_ll_0)
            wp.atomic_add(body_hessian_al, body_id, h_al_0)
            wp.atomic_add(body_hessian_aa, body_id, h_aa_0)
        else:  # body_id == b1
            wp.atomic_add(body_forces, body_id, force_1)
            wp.atomic_add(body_torques, body_id, torque_1)
            wp.atomic_add(body_hessian_ll, body_id, h_ll_1)
            wp.atomic_add(body_hessian_al, body_id, h_al_1)
            wp.atomic_add(body_hessian_aa, body_id, h_aa_1)

        # Move to next contact for this thread (strided)
        i += NUM_THREADS_PER_BODY


@wp.kernel
def accumulate_rigid_contact_force_and_hessian(
    dt: float,
    current_color: int,
    body_colors: wp.array(dtype=int),
    body_q_prev: wp.array(dtype=wp.transform),
    body_q: wp.array(dtype=wp.transform),
    body_com: wp.array(dtype=wp.vec3),
    body_inv_mass: wp.array(dtype=float),
    friction_epsilon: float,
    contact_penalty_k: wp.array(dtype=float),
    contact_material_kd: wp.array(dtype=float),
    contact_material_mu: wp.array(dtype=float),
    rigid_contact_count: wp.array(dtype=int),
    rigid_contact_max: int,
    rigid_contact_shape0: wp.array(dtype=int),
    rigid_contact_shape1: wp.array(dtype=int),
    rigid_contact_point0: wp.array(dtype=wp.vec3),
    rigid_contact_point1: wp.array(dtype=wp.vec3),
    rigid_contact_normal: wp.array(dtype=wp.vec3),
    rigid_contact_thickness0: wp.array(dtype=float),
    rigid_contact_thickness1: wp.array(dtype=float),
    shape_body: wp.array(dtype=wp.int32),
    body_forces: wp.array(dtype=wp.vec3),
    body_torques: wp.array(dtype=wp.vec3),
    body_hessian_ll: wp.array(dtype=wp.mat33),
    body_hessian_al: wp.array(dtype=wp.mat33),
    body_hessian_aa: wp.array(dtype=wp.mat33),
):
    """
    Rigid body collision accumulation kernel.

    Processes rigid body-body contacts, computing contact forces and Hessians
    using penalty method with friction. Only applies forces to dynamic bodies
    in the current color group for parallel processing.
    """
    t_id = wp.tid()

    rigid_body_contact_count = min(rigid_contact_max, rigid_contact_count[0])

    if t_id < rigid_body_contact_count:
        (
            body_id_0,
            shape_id_0,
            contact_point_0,
            collision_thickness_0,
            body_id_1,
            shape_id_1,
            contact_point_1,
            collision_thickness_1,
        ) = get_both_bodies_from_contact_with_thickness(
            t_id,
            rigid_contact_shape0,
            rigid_contact_shape1,
            rigid_contact_point0,
            rigid_contact_point1,
            rigid_contact_thickness0,
            rigid_contact_thickness1,
            shape_body,
        )

        # Determine which bodies are in the current color set
        apply_to_body_0 = body_id_0 >= 0 and body_colors[body_id_0] == current_color and body_inv_mass[body_id_0] > 0.0
        apply_to_body_1 = body_id_1 >= 0 and body_colors[body_id_1] == current_color and body_inv_mass[body_id_1] > 0.0

        if apply_to_body_0 or apply_to_body_1:
            contact_normal = -rigid_contact_normal[t_id]

            # The contact points are on the surfaces of the collision shapes.
            contact_point_0_world = (
                wp.transform_point(body_q[body_id_0], contact_point_0) if body_id_0 >= 0 else contact_point_0
            )
            contact_point_1_world = (
                wp.transform_point(body_q[body_id_1], contact_point_1) if body_id_1 >= 0 else contact_point_1
            )

            # Penetration is the geometric overlap, calculated consistently for all types.
            thickness = collision_thickness_0 + collision_thickness_1
            dist = wp.dot(contact_normal, contact_point_1_world - contact_point_0_world)
            actual_penetration = wp.max(0.0, thickness - dist)

            # Process contact forces only if there is penetration
            if actual_penetration > 1.0e-9:
                # Read pre-computed averaged material properties (computed once in warmstart)
                # Use adaptive penalty k for soft AVBD contacts
                contact_ke = contact_penalty_k[t_id]
                contact_kd = contact_material_kd[t_id]
                contact_mu = contact_material_mu[t_id]

                (force_a, torque_a, h_ll_a, h_al_a, h_aa_a, force_b, torque_b, h_ll_b, h_al_b, h_aa_b) = (
                    evaluate_rigid_contact_from_collision(
                        body_id_0,
                        body_id_1,
                        body_q,
                        body_q_prev,
                        body_com,
                        contact_point_0,
                        contact_point_1,
                        contact_normal,
                        actual_penetration,
                        contact_ke,
                        contact_kd,
                        contact_mu,
                        friction_epsilon,
                        dt,
                    )
                )

                # Apply forces only to bodies in current color
                if apply_to_body_0:
                    wp.atomic_add(body_forces, body_id_0, force_a)
                    wp.atomic_add(body_torques, body_id_0, torque_a)
                    wp.atomic_add(body_hessian_ll, body_id_0, h_ll_a)
                    wp.atomic_add(body_hessian_al, body_id_0, h_al_a)
                    wp.atomic_add(body_hessian_aa, body_id_0, h_aa_a)

                if apply_to_body_1:
                    wp.atomic_add(body_forces, body_id_1, force_b)
                    wp.atomic_add(body_torques, body_id_1, torque_b)
                    wp.atomic_add(body_hessian_ll, body_id_1, h_ll_b)
                    wp.atomic_add(body_hessian_al, body_id_1, h_al_b)
                    wp.atomic_add(body_hessian_aa, body_id_1, h_aa_b)
