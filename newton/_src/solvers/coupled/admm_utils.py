# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Warp kernels supporting the ADMM coupling scheme.

See :class:`SolverCoupled` and ``docs/plans/2026-04-23-admm-coupling.tex``
for the overall algorithm. Kernels here cover proximal velocity shifts,
point-velocity computation, ``J^T W`` force splatting, and the per-constraint
``u`` / ``lambda`` updates for model-joint attachments and detected contacts.
"""

from __future__ import annotations

import warp as wp

from ...math.spatial import velocity_at_point


@wp.kernel(enable_backward=False)
def velocity_proximal_shift_body_kernel(
    v_n: wp.array[wp.spatial_vector],
    v_k: wp.array[wp.spatial_vector],
    gamma: float,
    v_out: wp.array[wp.spatial_vector],
):
    """Write ``(v_n + gamma * v_k) / (1 + gamma)`` into ``v_out`` for each body.

    Paired with a ``(1 + gamma)`` rescaling of the body mass on the sub-solver's
    :class:`ModelView`, this produces the ADMM proximal term
    ``(gamma/2) ||v - v_k||^2_M`` in the sub-solver's per-step optimization.
    """
    i = wp.tid()
    inv_denom = 1.0 / (1.0 + gamma)
    v_out[i] = (v_n[i] + gamma * v_k[i]) * inv_denom


@wp.kernel(enable_backward=False)
def velocity_proximal_shift_particle_kernel(
    v_n: wp.array[wp.vec3],
    v_k: wp.array[wp.vec3],
    gamma: float,
    v_out: wp.array[wp.vec3],
):
    """Particle analogue of :func:`velocity_proximal_shift_body_kernel`."""
    i = wp.tid()
    inv_denom = 1.0 / (1.0 + gamma)
    v_out[i] = (v_n[i] + gamma * v_k[i]) * inv_denom


@wp.kernel(enable_backward=False)
def velocity_proximal_shift_joint_kernel(
    v_n: wp.array[float],
    v_k: wp.array[float],
    gamma: float,
    v_out: wp.array[float],
):
    """Joint-space analogue of :func:`velocity_proximal_shift_body_kernel`.

    Operates on the flat ``joint_qd`` array so the shift covers generalized
    DOFs for solvers whose authoritative velocity state is joint-space
    (e.g. :class:`~newton.solvers.SolverMuJoCo`).
    """
    i = wp.tid()
    inv_denom = 1.0 / (1.0 + gamma)
    v_out[i] = (v_n[i] + gamma * v_k[i]) * inv_denom


# ----------------------------------------------------------------------
# Quadratic attachment local solve
# ----------------------------------------------------------------------
#
# The ADMM update rules for a quadratic coupling energy
# ``E_c(u) = (kappa/2) ||u - u_target||^2 + (damping/2) ||u||^2`` are:
#
#     u^{k+1}      = (rho W^2 Jv + kappa u_target - W lambda) / (kappa + damping + rho W^2)
#     lambda^{k+1} = lambda^k + rho W (u^{k+1} - Jv)
#
# With ``u_target = 0`` the coupling damps the relative velocity to zero;
# a non-zero ``u_target`` acts as a Baumgarte-style position stabiliser.


@wp.kernel(enable_backward=False)
def u_update_quadratic_kernel(
    kappa: wp.array[float],
    damping: wp.array[float],
    W: wp.array[float],
    rho: float,
    lambda_k: wp.array[wp.vec3],
    Jv: wp.array[wp.vec3],
    u_target: wp.array[wp.vec3],
    u_out: wp.array[wp.vec3],
):
    """Closed-form u-update for a quadratic coupling energy."""
    i = wp.tid()
    W_i = W[i]
    W2 = W_i * W_i
    denom = kappa[i] + damping[i] + rho * W2
    u_out[i] = (rho * W2 * Jv[i] + kappa[i] * u_target[i] - W_i * lambda_k[i]) / denom


@wp.kernel(enable_backward=False)
def lambda_update_kernel(
    rho: float,
    W: wp.array[float],
    u: wp.array[wp.vec3],
    Jv: wp.array[wp.vec3],
    lambda_inout: wp.array[wp.vec3],
):
    """Dual-variable update ``lambda += rho * W * (u - Jv)``."""
    i = wp.tid()
    lambda_inout[i] = lambda_inout[i] + rho * W[i] * (u[i] - Jv[i])


@wp.func
def _soft_threshold_box(value: float, threshold: float) -> float:
    threshold = wp.max(0.0, threshold)
    if value > threshold:
        return value - threshold
    if value < -threshold:
        return value + threshold
    return 0.0


@wp.kernel(enable_backward=False)
def joint_box_friction_u_update_kernel(
    friction: wp.array[wp.vec3],
    W: wp.array[float],
    rho: float,
    lambda_k: wp.array[wp.vec3],
    Jv: wp.array[wp.vec3],
    u_out: wp.array[wp.vec3],
):
    """Local ADMM u-update for per-axis box dry friction.

    ``friction`` stores positive physical force/torque limits. The proximal
    solve is the component-wise soft-threshold of the relative velocity in the
    joint frame, giving a maximum-dissipation box-friction law.
    """
    i = wp.tid()
    W_i = W[i]
    p = Jv[i]
    denom = rho * W_i
    if denom > 0.0:
        p = p - lambda_k[i] / denom

    threshold = wp.vec3(0.0, 0.0, 0.0)
    force_denom = rho * W_i * W_i
    if force_denom > 0.0:
        threshold = friction[i] / force_denom

    u_out[i] = wp.vec3(
        _soft_threshold_box(p[0], threshold[0]),
        _soft_threshold_box(p[1], threshold[1]),
        _soft_threshold_box(p[2], threshold[2]),
    )


@wp.func
def solve_coulomb_isotropic(mu: float, normal: wp.vec3, u: wp.vec3):
    """Solve the isotropic local Coulomb law in velocity space.

    This is the local maximum-dissipation solve used by Daviet's contact
    projection: separating contacts keep their relative velocity, sticking
    contacts return zero velocity, and sliding contacts keep zero normal
    velocity while reducing the tangential velocity so the force lies on the
    Coulomb boundary.
    """
    u_n = wp.dot(u, normal)
    if u_n < 0.0:
        u = u - u_n * normal
        tau = wp.length_sq(u)
        alpha = mu * u_n
        if tau <= alpha * alpha:
            u = wp.vec3(0.0, 0.0, 0.0)
        else:
            u = u * (1.0 + mu * u_n / wp.sqrt(tau))

    return u


@wp.kernel(enable_backward=False)
def contact_u_update_kernel(
    u_min: wp.array[float],
    W: wp.array[float],
    rho: float,
    friction: wp.array[float],
    normal: wp.array[wp.vec3],
    lambda_k: wp.array[wp.vec3],
    Jv: wp.array[wp.vec3],
    u_out: wp.array[wp.vec3],
):
    """Local ADMM u-update for unilateral Coulomb contact.

    ``u_min`` is the minimum admissible normal relative velocity. For active
    penetration this is a non-negative separating velocity; for inactive
    candidate contacts it is a large negative value so the projection releases
    any warm-started contact force.
    """
    i = wp.tid()
    W_i = W[i]
    p = Jv[i]
    denom = rho * W_i
    if denom > 0.0:
        p = p - lambda_k[i] / denom

    u_min_i = u_min[i]
    if u_min_i < -1.0e7:
        u_out[i] = p
        return

    n = normal[i]
    mu = wp.max(0.0, friction[i])
    shifted = p - u_min_i * n
    u_out[i] = solve_coulomb_isotropic(mu, n, shifted) + u_min_i * n


@wp.kernel(enable_backward=False)
def contact_lambda_update_kernel(
    rho: float,
    W: wp.array[float],
    u: wp.array[wp.vec3],
    Jv: wp.array[wp.vec3],
    lambda_inout: wp.array[wp.vec3],
):
    """Dual update for unilateral contact after the Coulomb local solve."""
    i = wp.tid()
    lambda_inout[i] = lambda_inout[i] + rho * W[i] * (u[i] - Jv[i])


# ----------------------------------------------------------------------
# Rigid-particle attachment kernels
# ----------------------------------------------------------------------
#
# Custom model annotations can bind a rigid-body anchor to a deformable
# particle. The sign convention is
#
#     Jv = v_body_at_anchor - v_particle
#
# so side A receives ``+f`` and the particle receives ``-f``.


@wp.kernel(enable_backward=False)
def attach_rp_compute_Jv_kernel(
    body_a: wp.array[int],
    point_a_local: wp.array[wp.vec3],
    particle_b: wp.array[int],
    body_q: wp.array[wp.transform],
    body_com: wp.array[wp.vec3],
    body_qd: wp.array[wp.spatial_vector],
    particle_qd: wp.array[wp.vec3],
    Jv: wp.array[wp.vec3],
):
    """Compute ``Jv = v_body_at_anchor - v_particle`` per attachment."""
    i = wp.tid()
    ba = body_a[i]
    pb = particle_b[i]
    xform_a = body_q[ba]
    world_pt_a = wp.transform_point(xform_a, point_a_local[i])
    arm_a = world_pt_a - wp.transform_point(xform_a, body_com[ba])
    Jv[i] = velocity_at_point(body_qd[ba], arm_a) - particle_qd[pb]


@wp.kernel(enable_backward=False)
def attach_rp_compute_u_target_kernel(
    body_a: wp.array[int],
    point_a_local: wp.array[wp.vec3],
    particle_b: wp.array[int],
    body_q: wp.array[wp.transform],
    particle_q: wp.array[wp.vec3],
    baumgarte: float,
    dt: float,
    u_target: wp.array[wp.vec3],
):
    """Compute Baumgarte target velocity for a rigid-particle attachment."""
    i = wp.tid()
    ba = body_a[i]
    pb = particle_b[i]
    anchor = wp.transform_point(body_q[ba], point_a_local[i])
    gap = particle_q[pb] - anchor
    u_target[i] = (baumgarte / dt) * gap


@wp.kernel(enable_backward=False)
def attach_rp_accumulate_forces_kernel(
    body_a: wp.array[int],
    point_a_local: wp.array[wp.vec3],
    particle_b: wp.array[int],
    body_q: wp.array[wp.transform],
    body_com: wp.array[wp.vec3],
    rho: float,
    W: wp.array[float],
    lambda_k: wp.array[wp.vec3],
    u_k: wp.array[wp.vec3],
    Jv_k: wp.array[wp.vec3],
    body_f: wp.array[wp.spatial_vector],
    particle_f: wp.array[wp.vec3],
):
    """Splat rigid-particle attachment forces into body and particle buffers."""
    i = wp.tid()
    ba = body_a[i]
    pb = particle_b[i]
    W_i = W[i]
    force = W_i * (lambda_k[i] + rho * W_i * (u_k[i] - Jv_k[i]))

    xform_a = body_q[ba]
    world_pt_a = wp.transform_point(xform_a, point_a_local[i])
    arm_a = world_pt_a - wp.transform_point(xform_a, body_com[ba])
    wp.atomic_add(body_f, ba, wp.spatial_vector(force, wp.cross(arm_a, force)))
    wp.atomic_sub(particle_f, pb, force)


# ----------------------------------------------------------------------
# Rigid-rigid attachment kernels
# ----------------------------------------------------------------------
#
# Body-body model joints are converted to quadratic ADMM attachments. The
# translational row uses anchor point velocities:
#
#     Jv = v_a_at_anchor - v_b_at_anchor
#
# The fixed-joint angular row uses world angular velocities:
#
#     Jw = w_a - w_b


@wp.kernel(enable_backward=False)
def attach_rr_compute_u_target_kernel(
    body_a: wp.array[int],
    point_a_local: wp.array[wp.vec3],
    body_b: wp.array[int],
    point_b_local: wp.array[wp.vec3],
    body_q_a: wp.array[wp.transform],
    body_q_b: wp.array[wp.transform],
    baumgarte: float,
    dt: float,
    u_target: wp.array[wp.vec3],
):
    """Compute Baumgarte target velocity for a rigid-rigid anchor attachment."""
    i = wp.tid()
    ba = body_a[i]
    bb = body_b[i]
    point_a = wp.transform_point(body_q_a[ba], point_a_local[i])
    point_b = wp.transform_point(body_q_b[bb], point_b_local[i])
    gap = point_b - point_a
    u_target[i] = (baumgarte / dt) * gap


@wp.kernel(enable_backward=False)
def attach_rr_angular_compute_Jv_kernel(
    body_a: wp.array[int],
    body_b: wp.array[int],
    body_qd_a: wp.array[wp.spatial_vector],
    body_qd_b: wp.array[wp.spatial_vector],
    Jv: wp.array[wp.vec3],
):
    """Compute ``Jv = w_a - w_b`` per angular rigid-rigid attachment."""
    i = wp.tid()
    Jv[i] = wp.spatial_bottom(body_qd_a[body_a[i]]) - wp.spatial_bottom(body_qd_b[body_b[i]])


@wp.kernel(enable_backward=False)
def attach_rr_angular_local_compute_Jv_kernel(
    body_a: wp.array[int],
    frame_a: wp.array[wp.transform],
    body_b: wp.array[int],
    body_q_a: wp.array[wp.transform],
    body_qd_a: wp.array[wp.spatial_vector],
    body_qd_b: wp.array[wp.spatial_vector],
    Jv: wp.array[wp.vec3],
):
    """Compute angular relative velocity in body A's attachment frame."""
    i = wp.tid()
    ba = body_a[i]
    rel_w_world = wp.spatial_bottom(body_qd_a[ba]) - wp.spatial_bottom(body_qd_b[body_b[i]])
    frame_world = body_q_a[ba] * frame_a[i]
    Jv[i] = wp.quat_rotate_inv(wp.transform_get_rotation(frame_world), rel_w_world)


@wp.kernel(enable_backward=False)
def attach_rr_revolute_angular_local_compute_Jv_kernel(
    body_a: wp.array[int],
    frame_a: wp.array[wp.transform],
    body_b: wp.array[int],
    body_q_a: wp.array[wp.transform],
    body_qd_a: wp.array[wp.spatial_vector],
    body_qd_b: wp.array[wp.spatial_vector],
    Jv: wp.array[wp.vec3],
):
    """Compute the two constrained angular velocity components of a revolute joint."""
    i = wp.tid()
    ba = body_a[i]
    rel_w_world = wp.spatial_bottom(body_qd_a[ba]) - wp.spatial_bottom(body_qd_b[body_b[i]])
    frame_world = body_q_a[ba] * frame_a[i]
    rel_w_local = wp.quat_rotate_inv(wp.transform_get_rotation(frame_world), rel_w_world)
    Jv[i] = wp.vec3(0.0, rel_w_local[1], rel_w_local[2])


@wp.kernel(enable_backward=False)
def attach_rr_angular_compute_u_target_kernel(
    body_a: wp.array[int],
    frame_a: wp.array[wp.transform],
    body_b: wp.array[int],
    frame_b: wp.array[wp.transform],
    body_q_a: wp.array[wp.transform],
    body_q_b: wp.array[wp.transform],
    baumgarte: float,
    dt: float,
    u_target: wp.array[wp.vec3],
):
    """Compute Baumgarte target angular velocity for a fixed-joint row."""
    i = wp.tid()
    rot_a = wp.transform_get_rotation(body_q_a[body_a[i]] * frame_a[i])
    rot_b = wp.transform_get_rotation(body_q_b[body_b[i]] * frame_b[i])
    dq = wp.normalize(wp.mul(rot_b, wp.quat_inverse(rot_a)))
    axis, angle = wp.quat_to_axis_angle(dq)
    u_target[i] = axis * (baumgarte * angle / dt)


@wp.kernel(enable_backward=False)
def attach_rr_revolute_angular_local_compute_u_target_kernel(
    body_a: wp.array[int],
    frame_a: wp.array[wp.transform],
    body_b: wp.array[int],
    frame_b: wp.array[wp.transform],
    body_q_a: wp.array[wp.transform],
    body_q_b: wp.array[wp.transform],
    baumgarte: float,
    dt: float,
    u_target: wp.array[wp.vec3],
):
    """Compute Baumgarte target for the two constrained revolute angular axes."""
    i = wp.tid()
    rot_a = wp.transform_get_rotation(body_q_a[body_a[i]] * frame_a[i])
    rot_b = wp.transform_get_rotation(body_q_b[body_b[i]] * frame_b[i])
    dq = wp.normalize(wp.mul(rot_b, wp.quat_inverse(rot_a)))
    axis, angle = wp.quat_to_axis_angle(dq)
    target_world = axis * (baumgarte * angle / dt)
    target_local = wp.quat_rotate_inv(rot_a, target_world)
    u_target[i] = wp.vec3(0.0, target_local[1], target_local[2])


@wp.kernel(enable_backward=False)
def attach_rr_angular_accumulate_forces_kernel(
    body_a: wp.array[int],
    body_b: wp.array[int],
    rho: float,
    W: wp.array[float],
    lambda_k: wp.array[wp.vec3],
    u_k: wp.array[wp.vec3],
    Jv_k: wp.array[wp.vec3],
    body_f_a: wp.array[wp.spatial_vector],
    body_f_b: wp.array[wp.spatial_vector],
):
    """Splat angular attachment torques into both rigid-body force buffers."""
    i = wp.tid()
    W_i = W[i]
    torque_a = W_i * (lambda_k[i] + rho * W_i * (u_k[i] - Jv_k[i]))
    wp.atomic_add(body_f_a, body_a[i], wp.spatial_vector(wp.vec3(0.0, 0.0, 0.0), torque_a))
    wp.atomic_sub(body_f_b, body_b[i], wp.spatial_vector(wp.vec3(0.0, 0.0, 0.0), torque_a))


@wp.kernel(enable_backward=False)
def attach_rr_angular_local_accumulate_forces_kernel(
    body_a: wp.array[int],
    frame_a: wp.array[wp.transform],
    body_b: wp.array[int],
    body_q_a: wp.array[wp.transform],
    rho: float,
    W: wp.array[float],
    lambda_k: wp.array[wp.vec3],
    u_k: wp.array[wp.vec3],
    Jv_k: wp.array[wp.vec3],
    body_f_a: wp.array[wp.spatial_vector],
    body_f_b: wp.array[wp.spatial_vector],
):
    """Splat local angular attachment torques in world coordinates."""
    i = wp.tid()
    ba = body_a[i]
    W_i = W[i]
    torque_local = W_i * (lambda_k[i] + rho * W_i * (u_k[i] - Jv_k[i]))
    frame_world = body_q_a[ba] * frame_a[i]
    torque_world = wp.quat_rotate(wp.transform_get_rotation(frame_world), torque_local)
    wp.atomic_add(body_f_a, ba, wp.spatial_vector(wp.vec3(0.0, 0.0, 0.0), torque_world))
    wp.atomic_sub(body_f_b, body_b[i], wp.spatial_vector(wp.vec3(0.0, 0.0, 0.0), torque_world))


@wp.kernel(enable_backward=False)
def attach_rr_revolute_angular_local_accumulate_forces_kernel(
    body_a: wp.array[int],
    frame_a: wp.array[wp.transform],
    body_b: wp.array[int],
    body_q_a: wp.array[wp.transform],
    rho: float,
    W: wp.array[float],
    lambda_k: wp.array[wp.vec3],
    u_k: wp.array[wp.vec3],
    Jv_k: wp.array[wp.vec3],
    body_f_a: wp.array[wp.spatial_vector],
    body_f_b: wp.array[wp.spatial_vector],
):
    """Splat revolute angular constraint torques, leaving the hinge axis free."""
    i = wp.tid()
    ba = body_a[i]
    W_i = W[i]
    torque_local = W_i * (lambda_k[i] + rho * W_i * (u_k[i] - Jv_k[i]))
    torque_local = wp.vec3(0.0, torque_local[1], torque_local[2])
    frame_world = body_q_a[ba] * frame_a[i]
    torque_world = wp.quat_rotate(wp.transform_get_rotation(frame_world), torque_local)
    wp.atomic_add(body_f_a, ba, wp.spatial_vector(wp.vec3(0.0, 0.0, 0.0), torque_world))
    wp.atomic_sub(body_f_b, body_b[i], wp.spatial_vector(wp.vec3(0.0, 0.0, 0.0), torque_world))


# ----------------------------------------------------------------------
# Rigid-rigid contact kernels
# ----------------------------------------------------------------------
#
# Contact normals point from endpoint B toward endpoint A. A positive scalar
# contact force applies +normal to endpoint A and -normal to endpoint B.


@wp.kernel(enable_backward=False)
def contact_rr_compute_Jv_kernel(
    body_a: wp.array[int],
    point_a_local: wp.array[wp.vec3],
    body_b: wp.array[int],
    point_b_local: wp.array[wp.vec3],
    body_q_a: wp.array[wp.transform],
    body_com_a: wp.array[wp.vec3],
    body_qd_a: wp.array[wp.spatial_vector],
    body_q_b: wp.array[wp.transform],
    body_com_b: wp.array[wp.vec3],
    body_qd_b: wp.array[wp.spatial_vector],
    Jv: wp.array[wp.vec3],
):
    """Compute relative point velocity for a rigid-rigid contact."""
    i = wp.tid()
    ba = body_a[i]
    bb = body_b[i]

    xform_a = body_q_a[ba]
    point_a = wp.transform_point(xform_a, point_a_local[i])
    arm_a = point_a - wp.transform_point(xform_a, body_com_a[ba])
    vel_a = velocity_at_point(body_qd_a[ba], arm_a)

    xform_b = body_q_b[bb]
    point_b = wp.transform_point(xform_b, point_b_local[i])
    arm_b = point_b - wp.transform_point(xform_b, body_com_b[bb])
    vel_b = velocity_at_point(body_qd_b[bb], arm_b)

    Jv[i] = vel_a - vel_b


@wp.kernel(enable_backward=False)
def contact_rr_compute_u_min_kernel(
    body_a: wp.array[int],
    point_a_local: wp.array[wp.vec3],
    body_b: wp.array[int],
    point_b_local: wp.array[wp.vec3],
    normal: wp.array[wp.vec3],
    contact_distance: wp.array[float],
    body_q_a: wp.array[wp.transform],
    body_q_b: wp.array[wp.transform],
    baumgarte: float,
    dt: float,
    u_min: wp.array[float],
):
    """Compute the minimum normal velocity for a rigid-rigid contact."""
    i = wp.tid()
    ba = body_a[i]
    bb = body_b[i]
    point_a = wp.transform_point(body_q_a[ba], point_a_local[i])
    point_b = wp.transform_point(body_q_b[bb], point_b_local[i])
    gap = wp.dot(normal[i], point_a - point_b)
    violation = contact_distance[i] - gap
    if violation > 0.0 and dt > 0.0:
        u_min[i] = baumgarte * violation / dt
    else:
        u_min[i] = -1.0e8


@wp.kernel(enable_backward=False)
def contact_rr_accumulate_forces_kernel(
    body_a: wp.array[int],
    point_a_local: wp.array[wp.vec3],
    body_b: wp.array[int],
    point_b_local: wp.array[wp.vec3],
    body_q_a: wp.array[wp.transform],
    body_com_a: wp.array[wp.vec3],
    body_q_b: wp.array[wp.transform],
    body_com_b: wp.array[wp.vec3],
    rho: float,
    W: wp.array[float],
    lambda_k: wp.array[wp.vec3],
    u_k: wp.array[wp.vec3],
    Jv_k: wp.array[wp.vec3],
    body_f_a: wp.array[wp.spatial_vector],
    body_f_b: wp.array[wp.spatial_vector],
):
    """Splat Coulomb contact wrenches for a rigid-rigid contact."""
    i = wp.tid()
    ba = body_a[i]
    bb = body_b[i]
    W_i = W[i]
    force_a = W_i * (lambda_k[i] + rho * W_i * (u_k[i] - Jv_k[i]))

    xform_a = body_q_a[ba]
    point_a = wp.transform_point(xform_a, point_a_local[i])
    arm_a = point_a - wp.transform_point(xform_a, body_com_a[ba])
    wp.atomic_add(body_f_a, ba, wp.spatial_vector(force_a, wp.cross(arm_a, force_a)))

    force_b = -force_a
    xform_b = body_q_b[bb]
    point_b = wp.transform_point(xform_b, point_b_local[i])
    arm_b = point_b - wp.transform_point(xform_b, body_com_b[bb])
    wp.atomic_add(body_f_b, bb, wp.spatial_vector(force_b, wp.cross(arm_b, force_b)))


@wp.kernel(enable_backward=False)
def contact_rr_snapshot_kernel(
    body_a: wp.array[int],
    body_b: wp.array[int],
    shape_a: wp.array[int],
    shape_b: wp.array[int],
    point_id: wp.array[int],
    active: wp.array[int],
    u: wp.array[wp.vec3],
    lambda_: wp.array[wp.vec3],
    prev_body_a: wp.array[int],
    prev_body_b: wp.array[int],
    prev_shape_a: wp.array[int],
    prev_shape_b: wp.array[int],
    prev_point_id: wp.array[int],
    prev_active: wp.array[int],
    prev_u: wp.array[wp.vec3],
    prev_lambda: wp.array[wp.vec3],
):
    """Snapshot dynamic rigid-rigid contacts for key-based warm starting."""
    i = wp.tid()
    prev_body_a[i] = body_a[i]
    prev_body_b[i] = body_b[i]
    prev_shape_a[i] = shape_a[i]
    prev_shape_b[i] = shape_b[i]
    prev_point_id[i] = point_id[i]
    prev_active[i] = active[i]
    prev_u[i] = u[i]
    prev_lambda[i] = lambda_[i]


@wp.kernel(enable_backward=False)
def contact_rr_reset_kernel(
    active_count: wp.array[int],
    body_a: wp.array[int],
    point_a_local: wp.array[wp.vec3],
    body_b: wp.array[int],
    point_b_local: wp.array[wp.vec3],
    shape_a: wp.array[int],
    shape_b: wp.array[int],
    point_id: wp.array[int],
    active: wp.array[int],
    normal: wp.array[wp.vec3],
    contact_distance: wp.array[float],
    W: wp.array[float],
    friction: wp.array[float],
    u: wp.array[wp.vec3],
    lambda_: wp.array[wp.vec3],
    Jv: wp.array[wp.vec3],
    u_min: wp.array[float],
):
    """Clear a fixed-capacity dynamic rigid-rigid contact group."""
    i = wp.tid()
    if i == 0:
        active_count[0] = 0

    body_a[i] = 0
    point_a_local[i] = wp.vec3(0.0, 0.0, 0.0)
    body_b[i] = 0
    point_b_local[i] = wp.vec3(0.0, 0.0, 0.0)
    shape_a[i] = -1
    shape_b[i] = -1
    point_id[i] = -1
    active[i] = 0
    normal[i] = wp.vec3(0.0, 0.0, 0.0)
    contact_distance[i] = 0.0
    W[i] = 0.0
    friction[i] = 0.0
    u[i] = wp.vec3(0.0, 0.0, 0.0)
    lambda_[i] = wp.vec3(0.0, 0.0, 0.0)
    Jv[i] = wp.vec3(0.0, 0.0, 0.0)
    u_min[i] = -1.0e8


@wp.kernel(enable_backward=False)
def contact_rr_fill_from_rigid_contacts_kernel(
    rigid_contact_count: wp.array[int],
    rigid_contact_shape0: wp.array[int],
    rigid_contact_shape1: wp.array[int],
    rigid_contact_point0: wp.array[wp.vec3],
    rigid_contact_point1: wp.array[wp.vec3],
    rigid_contact_normal: wp.array[wp.vec3],
    rigid_contact_margin0: wp.array[float],
    rigid_contact_margin1: wp.array[float],
    rigid_contact_point_id: wp.array[int],
    shape_body: wp.array[int],
    body_mask_a: wp.array[int],
    body_mask_b: wp.array[int],
    shape_mask_a: wp.array[int],
    shape_mask_b: wp.array[int],
    body_global_to_local_a: wp.array[int],
    body_global_to_local_b: wp.array[int],
    body_mass_a: wp.array[float],
    body_mass_b: wp.array[float],
    shape_material_mu: wp.array[float],
    contact_distance_value: float,
    use_contact_margins: int,
    capacity: int,
    active_count: wp.array[int],
    active_count_max: wp.array[int],
    prev_shape_a: wp.array[int],
    prev_shape_b: wp.array[int],
    prev_point_id: wp.array[int],
    prev_active: wp.array[int],
    prev_u: wp.array[wp.vec3],
    prev_lambda: wp.array[wp.vec3],
    body_a: wp.array[int],
    point_a_local: wp.array[wp.vec3],
    body_b: wp.array[int],
    point_b_local: wp.array[wp.vec3],
    shape_a: wp.array[int],
    shape_b: wp.array[int],
    point_id: wp.array[int],
    active: wp.array[int],
    normal: wp.array[wp.vec3],
    contact_distance: wp.array[float],
    W: wp.array[float],
    friction: wp.array[float],
    u: wp.array[wp.vec3],
    lambda_: wp.array[wp.vec3],
):
    """Convert detected rigid contacts into oriented ADMM rigid-rigid rows."""
    i = wp.tid()
    if i >= rigid_contact_count[0]:
        return

    s0 = rigid_contact_shape0[i]
    s1 = rigid_contact_shape1[i]
    if s0 < 0 or s1 < 0:
        return

    b0 = shape_body[s0]
    b1 = shape_body[s1]
    if b0 < 0 or b1 < 0:
        return

    ba = int(0)
    bb = int(0)
    sa = int(-1)
    sb = int(-1)
    pa = wp.vec3(0.0, 0.0, 0.0)
    pb = wp.vec3(0.0, 0.0, 0.0)
    n = wp.vec3(0.0, 0.0, 0.0)

    if body_mask_a[b1] != 0 and body_mask_b[b0] != 0 and shape_mask_a[s1] != 0 and shape_mask_b[s0] != 0:
        ba = b1
        bb = b0
        sa = s1
        sb = s0
        pa = rigid_contact_point1[i]
        pb = rigid_contact_point0[i]
        n = rigid_contact_normal[i]
    elif body_mask_a[b0] != 0 and body_mask_b[b1] != 0 and shape_mask_a[s0] != 0 and shape_mask_b[s1] != 0:
        ba = b0
        bb = b1
        sa = s0
        sb = s1
        pa = rigid_contact_point0[i]
        pb = rigid_contact_point1[i]
        n = -rigid_contact_normal[i]
    else:
        return

    ba_local = body_global_to_local_a[ba]
    bb_local = body_global_to_local_b[bb]
    if ba_local < 0 or bb_local < 0:
        return

    dst = wp.atomic_add(active_count, 0, 1)
    if dst >= capacity:
        wp.atomic_min(active_count, 0, capacity)
        wp.atomic_max(active_count_max, 0, capacity)
        return
    wp.atomic_max(active_count_max, 0, dst + 1)

    body_a[dst] = ba_local
    point_a_local[dst] = pa
    body_b[dst] = bb_local
    point_b_local[dst] = pb
    shape_a[dst] = sa
    shape_b[dst] = sb
    pid = rigid_contact_point_id[i]
    point_id[dst] = pid
    active[dst] = 1
    normal[dst] = n
    if use_contact_margins != 0:
        contact_distance[dst] = rigid_contact_margin0[i] + rigid_contact_margin1[i]
    else:
        contact_distance[dst] = contact_distance_value

    ma = body_mass_a[ba]
    mb = body_mass_b[bb]
    if ma > 0.0 and mb > 0.0:
        W[dst] = wp.sqrt((ma * mb) / (ma + mb))
    else:
        W[dst] = 1.0
    # Geometric-mean combining of shape material friction (matches AVBD).
    friction[dst] = wp.sqrt(shape_material_mu[sa] * shape_material_mu[sb])

    u_out = wp.vec3(0.0, 0.0, 0.0)
    lambda_out = wp.vec3(0.0, 0.0, 0.0)
    for j in range(capacity):
        if prev_active[j] != 0 and prev_shape_a[j] == sa and prev_shape_b[j] == sb and prev_point_id[j] == pid:
            u_out = prev_u[j]
            lambda_out = prev_lambda[j]
            break
    u[dst] = u_out
    lambda_[dst] = lambda_out


# ----------------------------------------------------------------------
# Rigid-particle contact kernels
# ----------------------------------------------------------------------
#
# Contact normals point from endpoint B toward endpoint A. A positive scalar
# contact force applies +normal to endpoint A and -normal to endpoint B.


@wp.kernel(enable_backward=False)
def contact_rp_compute_Jv_kernel(
    body_id: wp.array[int],
    point_body_local: wp.array[wp.vec3],
    particle_id: wp.array[int],
    body_sign: wp.array[int],
    body_q: wp.array[wp.transform],
    body_com: wp.array[wp.vec3],
    body_qd: wp.array[wp.spatial_vector],
    particle_qd: wp.array[wp.vec3],
    Jv: wp.array[wp.vec3],
):
    """Compute relative velocity for a rigid-particle contact."""
    i = wp.tid()
    b = body_id[i]
    p = particle_id[i]
    xform = body_q[b]
    world_pt = wp.transform_point(xform, point_body_local[i])
    arm = world_pt - wp.transform_point(xform, body_com[b])
    body_v = velocity_at_point(body_qd[b], arm)
    particle_v = particle_qd[p]
    if body_sign[i] > 0:
        Jv[i] = body_v - particle_v
    else:
        Jv[i] = particle_v - body_v


@wp.kernel(enable_backward=False)
def contact_rp_compute_u_min_kernel(
    body_id: wp.array[int],
    point_body_local: wp.array[wp.vec3],
    particle_id: wp.array[int],
    normal: wp.array[wp.vec3],
    body_sign: wp.array[int],
    contact_distance: wp.array[float],
    body_q: wp.array[wp.transform],
    particle_q: wp.array[wp.vec3],
    baumgarte: float,
    dt: float,
    u_min: wp.array[float],
):
    """Compute the minimum normal velocity for a rigid-particle contact."""
    i = wp.tid()
    b = body_id[i]
    p = particle_id[i]
    body_pt = wp.transform_point(body_q[b], point_body_local[i])
    particle_pt = particle_q[p]
    n = normal[i]
    gap = float(0.0)
    if body_sign[i] > 0:
        gap = wp.dot(n, body_pt - particle_pt)
    else:
        gap = wp.dot(n, particle_pt - body_pt)

    violation = contact_distance[i] - gap
    if violation > 0.0 and dt > 0.0:
        u_min[i] = baumgarte * violation / dt
    else:
        u_min[i] = -1.0e8


@wp.kernel(enable_backward=False)
def contact_rp_accumulate_forces_kernel(
    body_id: wp.array[int],
    point_body_local: wp.array[wp.vec3],
    particle_id: wp.array[int],
    body_sign: wp.array[int],
    body_q: wp.array[wp.transform],
    body_com: wp.array[wp.vec3],
    rho: float,
    W: wp.array[float],
    lambda_k: wp.array[wp.vec3],
    u_k: wp.array[wp.vec3],
    Jv_k: wp.array[wp.vec3],
    body_f: wp.array[wp.spatial_vector],
    particle_f: wp.array[wp.vec3],
):
    """Splat Coulomb contact forces for a rigid-particle contact."""
    i = wp.tid()
    b = body_id[i]
    p = particle_id[i]
    W_i = W[i]
    force = W_i * (lambda_k[i] + rho * W_i * (u_k[i] - Jv_k[i]))
    force_body = float(body_sign[i]) * force

    xform = body_q[b]
    world_pt = wp.transform_point(xform, point_body_local[i])
    arm = world_pt - wp.transform_point(xform, body_com[b])
    wp.atomic_add(body_f, b, wp.spatial_vector(force_body, wp.cross(arm, force_body)))
    wp.atomic_sub(particle_f, p, force_body)


@wp.kernel(enable_backward=False)
def contact_rp_snapshot_kernel(
    body_id: wp.array[int],
    particle_id: wp.array[int],
    shape_id: wp.array[int],
    active: wp.array[int],
    u: wp.array[wp.vec3],
    lambda_: wp.array[wp.vec3],
    prev_body_id: wp.array[int],
    prev_particle_id: wp.array[int],
    prev_shape_id: wp.array[int],
    prev_active: wp.array[int],
    prev_u: wp.array[wp.vec3],
    prev_lambda: wp.array[wp.vec3],
):
    """Snapshot dynamic rigid-particle contacts for key-based warm starting."""
    i = wp.tid()
    prev_body_id[i] = body_id[i]
    prev_particle_id[i] = particle_id[i]
    prev_shape_id[i] = shape_id[i]
    prev_active[i] = active[i]
    prev_u[i] = u[i]
    prev_lambda[i] = lambda_[i]


@wp.kernel(enable_backward=False)
def contact_rp_reset_kernel(
    active_count: wp.array[int],
    body_id: wp.array[int],
    point_body_local: wp.array[wp.vec3],
    particle_id: wp.array[int],
    shape_id: wp.array[int],
    active: wp.array[int],
    normal: wp.array[wp.vec3],
    body_sign: wp.array[int],
    contact_distance: wp.array[float],
    W: wp.array[float],
    friction: wp.array[float],
    u: wp.array[wp.vec3],
    lambda_: wp.array[wp.vec3],
    Jv: wp.array[wp.vec3],
    u_min: wp.array[float],
):
    """Clear a fixed-capacity dynamic rigid-particle contact group."""
    i = wp.tid()
    if i == 0:
        active_count[0] = 0

    active[i] = 0
    body_id[i] = 0
    point_body_local[i] = wp.vec3(0.0, 0.0, 0.0)
    particle_id[i] = 0
    shape_id[i] = -1
    normal[i] = wp.vec3(0.0, 0.0, 0.0)
    body_sign[i] = -1
    contact_distance[i] = 0.0
    W[i] = 0.0
    friction[i] = 0.0
    u[i] = wp.vec3(0.0, 0.0, 0.0)
    lambda_[i] = wp.vec3(0.0, 0.0, 0.0)
    Jv[i] = wp.vec3(0.0, 0.0, 0.0)
    u_min[i] = -1.0e8


@wp.kernel(enable_backward=False)
def contact_rp_fill_from_soft_contacts_kernel(
    soft_contact_count: wp.array[int],
    soft_contact_particle: wp.array[int],
    soft_contact_shape: wp.array[int],
    soft_contact_body_pos: wp.array[wp.vec3],
    soft_contact_normal: wp.array[wp.vec3],
    shape_body: wp.array[int],
    particle_owner_mask: wp.array[int],
    body_owner_mask: wp.array[int],
    shape_filter_mask: wp.array[int],
    body_global_to_local: wp.array[int],
    particle_radius: wp.array[float],
    body_mass: wp.array[float],
    particle_mass: wp.array[float],
    shape_material_mu: wp.array[float],
    particle_mu: float,
    contact_distance_value: float,
    use_particle_radius: int,
    capacity: int,
    active_count: wp.array[int],
    active_count_max: wp.array[int],
    prev_particle_id: wp.array[int],
    prev_shape_id: wp.array[int],
    prev_active: wp.array[int],
    prev_u: wp.array[wp.vec3],
    prev_lambda: wp.array[wp.vec3],
    body_id: wp.array[int],
    point_body_local: wp.array[wp.vec3],
    particle_id: wp.array[int],
    shape_id: wp.array[int],
    active: wp.array[int],
    normal: wp.array[wp.vec3],
    body_sign: wp.array[int],
    contact_distance: wp.array[float],
    W: wp.array[float],
    friction: wp.array[float],
    u: wp.array[wp.vec3],
    lambda_: wp.array[wp.vec3],
):
    """Populate a dynamic rigid-particle group from soft particle-shape contacts."""
    i = wp.tid()
    if i >= soft_contact_count[0]:
        return

    p = soft_contact_particle[i]
    s = soft_contact_shape[i]
    if p < 0 or s < 0:
        return
    if particle_owner_mask[p] == 0 or shape_filter_mask[s] == 0:
        return

    b = shape_body[s]
    if b < 0 or body_owner_mask[b] == 0:
        return

    b_local = body_global_to_local[b]
    if b_local < 0:
        return

    n = soft_contact_normal[i]
    n_len = wp.length(n)
    if n_len <= 0.0:
        return
    n = n / n_len

    dst = wp.atomic_add(active_count, 0, 1)
    if dst >= capacity:
        wp.atomic_min(active_count, 0, capacity)
        wp.atomic_max(active_count_max, 0, capacity)
        return
    wp.atomic_max(active_count_max, 0, dst + 1)

    m_a = body_mass[b]
    m_b = particle_mass[p]
    weight = float(1.0)
    if m_a > 0.0 and m_b > 0.0:
        weight = wp.sqrt((m_a * m_b) / (m_a + m_b))

    distance = contact_distance_value
    if use_particle_radius != 0:
        distance = particle_radius[p]

    u0 = wp.vec3(0.0, 0.0, 0.0)
    lambda0 = wp.vec3(0.0, 0.0, 0.0)
    for j in range(capacity):
        if prev_active[j] != 0 and prev_particle_id[j] == p and prev_shape_id[j] == s:
            u0 = prev_u[j]
            lambda0 = prev_lambda[j]
            break

    active[dst] = 1
    body_id[dst] = b_local
    point_body_local[dst] = soft_contact_body_pos[i]
    particle_id[dst] = p
    shape_id[dst] = s
    normal[dst] = n
    body_sign[dst] = -1
    contact_distance[dst] = distance
    W[dst] = weight
    # Geometric-mean combining of shape and particle friction.
    friction[dst] = wp.sqrt(shape_material_mu[s] * particle_mu)
    u[dst] = u0
    lambda_[dst] = lambda0


# ----------------------------------------------------------------------
# Particle-particle contact kernels
# ----------------------------------------------------------------------


@wp.kernel(enable_backward=False)
def contact_pp_compute_Jv_kernel(
    particle_a: wp.array[int],
    particle_b: wp.array[int],
    particle_qd_a: wp.array[wp.vec3],
    particle_qd_b: wp.array[wp.vec3],
    Jv: wp.array[wp.vec3],
):
    """Compute relative velocity for a particle-particle contact."""
    i = wp.tid()
    pa = particle_a[i]
    pb = particle_b[i]
    Jv[i] = particle_qd_a[pa] - particle_qd_b[pb]


@wp.kernel(enable_backward=False)
def contact_pp_compute_u_min_kernel(
    particle_a: wp.array[int],
    particle_b: wp.array[int],
    normal: wp.array[wp.vec3],
    contact_distance: wp.array[float],
    particle_q_a: wp.array[wp.vec3],
    particle_q_b: wp.array[wp.vec3],
    baumgarte: float,
    dt: float,
    u_min: wp.array[float],
):
    """Compute the minimum normal velocity for a particle-particle contact."""
    i = wp.tid()
    pa = particle_a[i]
    pb = particle_b[i]
    gap = wp.dot(normal[i], particle_q_a[pa] - particle_q_b[pb])
    violation = contact_distance[i] - gap
    if violation > 0.0 and dt > 0.0:
        u_min[i] = baumgarte * violation / dt
    else:
        u_min[i] = -1.0e8


@wp.kernel(enable_backward=False)
def contact_pp_accumulate_forces_kernel(
    particle_a: wp.array[int],
    particle_b: wp.array[int],
    rho: float,
    W: wp.array[float],
    lambda_k: wp.array[wp.vec3],
    u_k: wp.array[wp.vec3],
    Jv_k: wp.array[wp.vec3],
    particle_f_a: wp.array[wp.vec3],
    particle_f_b: wp.array[wp.vec3],
):
    """Splat Coulomb contact forces for a particle-particle contact."""
    i = wp.tid()
    pa = particle_a[i]
    pb = particle_b[i]
    W_i = W[i]
    force = W_i * (lambda_k[i] + rho * W_i * (u_k[i] - Jv_k[i]))
    wp.atomic_add(particle_f_a, pa, force)
    wp.atomic_sub(particle_f_b, pb, force)


@wp.kernel(enable_backward=False)
def contact_pp_snapshot_kernel(
    particle_a: wp.array[int],
    particle_b: wp.array[int],
    active: wp.array[int],
    u: wp.array[wp.vec3],
    lambda_: wp.array[wp.vec3],
    prev_particle_a: wp.array[int],
    prev_particle_b: wp.array[int],
    prev_active: wp.array[int],
    prev_u: wp.array[wp.vec3],
    prev_lambda: wp.array[wp.vec3],
):
    """Snapshot dynamic particle-particle contacts for key-based warm starting."""
    i = wp.tid()
    prev_particle_a[i] = particle_a[i]
    prev_particle_b[i] = particle_b[i]
    prev_active[i] = active[i]
    prev_u[i] = u[i]
    prev_lambda[i] = lambda_[i]


@wp.kernel(enable_backward=False)
def contact_pp_reset_kernel(
    active_count: wp.array[int],
    particle_a: wp.array[int],
    particle_b: wp.array[int],
    active: wp.array[int],
    normal: wp.array[wp.vec3],
    contact_distance: wp.array[float],
    W: wp.array[float],
    friction: wp.array[float],
    u: wp.array[wp.vec3],
    lambda_: wp.array[wp.vec3],
    Jv: wp.array[wp.vec3],
    u_min: wp.array[float],
):
    """Clear a fixed-capacity dynamic particle-particle contact group."""
    i = wp.tid()
    if i == 0:
        active_count[0] = 0

    particle_a[i] = 0
    particle_b[i] = 0
    active[i] = 0
    normal[i] = wp.vec3(0.0, 0.0, 0.0)
    contact_distance[i] = 0.0
    W[i] = 0.0
    friction[i] = 0.0
    u[i] = wp.vec3(0.0, 0.0, 0.0)
    lambda_[i] = wp.vec3(0.0, 0.0, 0.0)
    Jv[i] = wp.vec3(0.0, 0.0, 0.0)
    u_min[i] = -1.0e8


@wp.kernel(enable_backward=False)
def particle_contact_count_reset_kernel(particle_contact_count: wp.array[int]):
    """Reset a particle-particle contact stream count."""
    particle_contact_count[0] = 0


@wp.kernel(enable_backward=False)
def particle_particle_contacts_hashgrid_kernel(
    grid: wp.uint64,
    particle_q: wp.array[wp.vec3],
    particle_radius: wp.array[float],
    particle_flags: wp.array[wp.int32],
    particle_world: wp.array[int],
    particle_mask_a: wp.array[int],
    particle_mask_b: wp.array[int],
    contact_distance_value: float,
    use_radius_sum: int,
    detection_margin: float,
    query_radius: float,
    capacity: int,
    particle_contact_count: wp.array[int],
    particle_contact_count_max: wp.array[int],
    particle_contact_particle0: wp.array[int],
    particle_contact_particle1: wp.array[int],
    particle_contact_normal: wp.array[wp.vec3],
    particle_contact_distance: wp.array[float],
    particle_contact_tids: wp.array[int],
):
    """Detect particle-particle contacts into a contacts-like stream."""
    tid = wp.tid()
    pa = wp.hash_grid_point_id(grid, tid)
    if pa == -1:
        return
    if particle_mask_a[pa] == 0:
        return
    if (particle_flags[pa] & wp.int32(1)) == 0:
        return

    qa = particle_q[pa]
    world_a = particle_world[pa]
    query = wp.hash_grid_query(grid, qa, query_radius)
    pb = int(0)

    while wp.hash_grid_query_next(query, pb):
        if pb == pa:
            continue
        if particle_mask_b[pb] == 0:
            continue
        if (particle_flags[pb] & wp.int32(1)) == 0:
            continue

        world_b = particle_world[pb]
        if world_a != -1 and world_b != -1 and world_a != world_b:
            continue

        distance = contact_distance_value
        if use_radius_sum != 0:
            distance = particle_radius[pa] + particle_radius[pb]

        delta = qa - particle_q[pb]
        gap = wp.length(delta)
        if gap >= distance + detection_margin:
            continue

        n = wp.vec3(1.0, 0.0, 0.0)
        if gap > 1.0e-8:
            n = delta / gap

        dst = wp.atomic_add(particle_contact_count, 0, 1)
        if dst >= capacity:
            wp.atomic_min(particle_contact_count, 0, capacity)
            wp.atomic_max(particle_contact_count_max, 0, capacity)
            continue
        wp.atomic_max(particle_contact_count_max, 0, dst + 1)

        particle_contact_particle0[dst] = pa
        particle_contact_particle1[dst] = pb
        particle_contact_normal[dst] = n
        particle_contact_distance[dst] = distance
        particle_contact_tids[dst] = tid


@wp.kernel(enable_backward=False)
def contact_pp_fill_from_particle_contacts_kernel(
    particle_contact_count: wp.array[int],
    particle_contact_particle0: wp.array[int],
    particle_contact_particle1: wp.array[int],
    particle_contact_normal: wp.array[wp.vec3],
    particle_contact_distance: wp.array[float],
    particle_mass_a: wp.array[float],
    particle_mass_b: wp.array[float],
    particle_mu: float,
    capacity: int,
    active_count: wp.array[int],
    active_count_max: wp.array[int],
    prev_particle_a: wp.array[int],
    prev_particle_b: wp.array[int],
    prev_active: wp.array[int],
    prev_u: wp.array[wp.vec3],
    prev_lambda: wp.array[wp.vec3],
    particle_a: wp.array[int],
    particle_b: wp.array[int],
    active: wp.array[int],
    normal: wp.array[wp.vec3],
    contact_distance: wp.array[float],
    W: wp.array[float],
    friction: wp.array[float],
    u: wp.array[wp.vec3],
    lambda_: wp.array[wp.vec3],
):
    """Populate a dynamic particle-particle group from a contacts-like stream."""
    i = wp.tid()
    if i >= particle_contact_count[0]:
        return

    pa = particle_contact_particle0[i]
    pb = particle_contact_particle1[i]
    dst = wp.atomic_add(active_count, 0, 1)
    if dst >= capacity:
        wp.atomic_min(active_count, 0, capacity)
        wp.atomic_max(active_count_max, 0, capacity)
        return
    wp.atomic_max(active_count_max, 0, dst + 1)

    m_a = particle_mass_a[pa]
    m_b = particle_mass_b[pb]
    weight = float(1.0)
    if m_a > 0.0 and m_b > 0.0:
        weight = wp.sqrt((m_a * m_b) / (m_a + m_b))

    u0 = wp.vec3(0.0, 0.0, 0.0)
    lambda0 = wp.vec3(0.0, 0.0, 0.0)
    for j in range(capacity):
        if prev_active[j] != 0 and prev_particle_a[j] == pa and prev_particle_b[j] == pb:
            u0 = prev_u[j]
            lambda0 = prev_lambda[j]
            break

    particle_a[dst] = pa
    particle_b[dst] = pb
    active[dst] = 1
    normal[dst] = particle_contact_normal[i]
    contact_distance[dst] = particle_contact_distance[i]
    W[dst] = weight
    friction[dst] = particle_mu
    u[dst] = u0
    lambda_[dst] = lambda0
