# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0


import warp as wp

from ...math.spatial import transform_twist
from ...sim import BodyFlags, JointType
from ...sim.articulation import (
    compute_2d_rotational_dofs,
    compute_3d_rotational_dofs,
)
from ...sim.contacts import GENERATION_SENTINEL
from .contact_filters import contact_friction_eligible, contact_normal_gap_limit
from .friction import contact_tangent_basis
from .friction_patches import FrictionPatches, warmstart_dt_scale

PGS_CONSTRAINT_TYPE_CONTACT = 0
# PGS joint-drive row; the velocity-only iterations can freeze these rows.
PGS_CONSTRAINT_TYPE_JOINT_TARGET = 1
PGS_CONSTRAINT_TYPE_FRICTION = 2
PGS_CONSTRAINT_TYPE_JOINT_LIMIT = 3
# Joint velocity-limit row: a per-DOF velocity clamp with one unilateral row per
# bound and no position bias.
PGS_CONSTRAINT_TYPE_JOINT_VELOCITY_LIMIT = 4

# Positive gaps below this slop [m] count as touching: it absorbs float32 residuals of
# rows that land exactly at contact.
_FPGS_CONTACT_END_GAP_SLOP = wp.constant(1.0e-6)

# Warm-start history generation of a step without a contact set; no contact set has it.
CONTACT_GENERATION_NONE = wp.constant(GENERATION_SENTINEL)

# Keep launch-geometry-specific dynamics kernels out of the large general
# kernel module. Warp compiles one whole module variant per block dimension.
_KINEMATICS_KERNEL_MODULE = wp.Module(f"{__name__}.kinematics")
_INVERSE_DYNAMICS_KERNEL_MODULE = wp.Module(f"{__name__}.inverse_dynamics")
_MASS_DYNAMICS_KERNEL_MODULE = wp.Module(f"{__name__}.mass_dynamics")


@wp.kernel
def compute_spatial_inertia(
    body_inertia: wp.array[wp.mat33],
    body_mass: wp.array[float],
    # outputs
    body_I_m: wp.array[wp.spatial_matrix],
):
    tid = wp.tid()
    I = body_inertia[tid]
    m = body_mass[tid]
    # fmt: off
    body_I_m[tid] = wp.spatial_matrix(
        m,   0.0, 0.0, 0.0,     0.0,     0.0,
        0.0, m,   0.0, 0.0,     0.0,     0.0,
        0.0, 0.0, m,   0.0,     0.0,     0.0,
        0.0, 0.0, 0.0, I[0, 0], I[0, 1], I[0, 2],
        0.0, 0.0, 0.0, I[1, 0], I[1, 1], I[1, 2],
        0.0, 0.0, 0.0, I[2, 0], I[2, 1], I[2, 2],
    )
    # fmt: on


@wp.kernel
def compute_com_transforms(
    body_com: wp.array[wp.vec3],
    # outputs
    body_X_com: wp.array[wp.transform],
):
    tid = wp.tid()
    com = body_com[tid]
    body_X_com[tid] = wp.transform(com, wp.quat_identity())


@wp.kernel
def prescale_joint_velocity_limits(
    articulation_start: wp.array[int],
    joint_type: wp.array[int],
    joint_child: wp.array[int],
    joint_qd_start: wp.array[int],
    joint_dof_dim: wp.array2d[int],
    joint_velocity_limit: wp.array[float],
    body_flags: wp.array[wp.int32],
    joint_qd: wp.array[float],
):
    """PhysX-style pre-solve joint velocity scaling.

    PhysX computes a single ratio per articulation from maxJointVelocity and
    applies that ratio to all articulation DOFs before building link velocities.
    This is separate from the velocity-limit constraint rows solved later.
    """
    art = wp.tid()
    joint_start = articulation_start[art]
    joint_end = articulation_start[art + 1]

    ratio = float(1.0)
    for j in range(joint_start, joint_end):
        if (body_flags[joint_child[j]] & BodyFlags.KINEMATIC) != 0:
            continue
        jtype = joint_type[j]
        if jtype != JointType.PRISMATIC and jtype != JointType.REVOLUTE and jtype != JointType.D6:
            continue

        lin_count = joint_dof_dim[j, 0]
        ang_count = joint_dof_dim[j, 1]
        axis_count = lin_count + ang_count
        qd_start = joint_qd_start[j]

        for axis in range(axis_count):
            dof = qd_start + axis
            limit = joint_velocity_limit[dof]
            qd_abs = wp.abs(joint_qd[dof])
            if limit > 0.0 and wp.isfinite(limit) and qd_abs > 0.0:
                scale = limit / qd_abs
                if scale < ratio:
                    ratio = scale

    if ratio >= 1.0:
        return

    for j in range(joint_start, joint_end):
        if (body_flags[joint_child[j]] & BodyFlags.KINEMATIC) != 0:
            continue
        jtype = joint_type[j]
        if jtype != JointType.PRISMATIC and jtype != JointType.REVOLUTE and jtype != JointType.D6:
            continue

        lin_count = joint_dof_dim[j, 0]
        ang_count = joint_dof_dim[j, 1]
        axis_count = lin_count + ang_count
        qd_start = joint_qd_start[j]

        for axis in range(axis_count):
            dof = qd_start + axis
            joint_qd[dof] = joint_qd[dof] * ratio


@wp.func
def transform_spatial_inertia(t: wp.transform, I: wp.spatial_matrix):
    """
    Transform a spatial inertia tensor to a new coordinate frame.

    This computes the change of coordinates for a spatial inertia tensor under a rigid-body
    transformation `t`. The result is mathematically equivalent to:

        adj_t^-T * I * adj_t^-1

    where `adj_t` is the adjoint transformation matrix of `t`, and `I` is the spatial inertia
    tensor in the original frame. This operation is described in Frank & Park, "Modern Robotics",
    Section 8.2.3 (pg. 290).

    Args:
        t (wp.transform): The rigid-body transform (destination ← source).
        I (wp.spatial_matrix): The spatial inertia tensor in the source frame.

    Returns:
        wp.spatial_matrix: The spatial inertia tensor expressed in the destination frame.
    """
    t_inv = wp.transform_inverse(t)

    q = wp.transform_get_rotation(t_inv)
    p = wp.transform_get_translation(t_inv)

    r1 = wp.quat_rotate(q, wp.vec3(1.0, 0.0, 0.0))
    r2 = wp.quat_rotate(q, wp.vec3(0.0, 1.0, 0.0))
    r3 = wp.quat_rotate(q, wp.vec3(0.0, 0.0, 1.0))

    R = wp.matrix_from_cols(r1, r2, r3)
    S = wp.skew(p) @ R

    T = wp.spatial_matrix(
        R[0, 0],
        R[0, 1],
        R[0, 2],
        S[0, 0],
        S[0, 1],
        S[0, 2],
        R[1, 0],
        R[1, 1],
        R[1, 2],
        S[1, 0],
        S[1, 1],
        S[1, 2],
        R[2, 0],
        R[2, 1],
        R[2, 2],
        S[2, 0],
        S[2, 1],
        S[2, 2],
        0.0,
        0.0,
        0.0,
        R[0, 0],
        R[0, 1],
        R[0, 2],
        0.0,
        0.0,
        0.0,
        R[1, 0],
        R[1, 1],
        R[1, 2],
        0.0,
        0.0,
        0.0,
        R[2, 0],
        R[2, 1],
        R[2, 2],
    )

    return wp.mul(wp.mul(wp.transpose(T), I), T)


@wp.func
def transform_com_inertia_terms(t: wp.transform, mass: float, inertia_com: wp.mat33):
    """Rotate COM inertia and shift its angular block to the solve origin."""
    rotation = wp.quat_to_matrix(wp.transform_get_rotation(t))
    com = wp.transform_get_translation(t)
    com_cross = wp.skew(com)
    inertia_origin = rotation * inertia_com * wp.transpose(rotation) - mass * com_cross * com_cross
    return com, inertia_origin


@wp.func
def assemble_com_spatial_inertia(mass: float, com: wp.vec3, inertia_origin: wp.mat33):
    """Assemble a solve-frame spatial inertia from compact COM terms."""
    mass_com_cross = mass * wp.skew(com)
    # fmt: off
    return wp.spatial_matrix(
        mass, 0.0,  0.0,  -mass_com_cross[0, 0], -mass_com_cross[0, 1], -mass_com_cross[0, 2],
        0.0,  mass, 0.0,  -mass_com_cross[1, 0], -mass_com_cross[1, 1], -mass_com_cross[1, 2],
        0.0,  0.0,  mass, -mass_com_cross[2, 0], -mass_com_cross[2, 1], -mass_com_cross[2, 2],
        mass_com_cross[0, 0], mass_com_cross[0, 1], mass_com_cross[0, 2],
        inertia_origin[0, 0], inertia_origin[0, 1], inertia_origin[0, 2],
        mass_com_cross[1, 0], mass_com_cross[1, 1], mass_com_cross[1, 2],
        inertia_origin[1, 0], inertia_origin[1, 1], inertia_origin[1, 2],
        mass_com_cross[2, 0], mass_com_cross[2, 1], mass_com_cross[2, 2],
        inertia_origin[2, 0], inertia_origin[2, 1], inertia_origin[2, 2],
    )
    # fmt: on


@wp.func
def mul_com_spatial_inertia(mass: float, com: wp.vec3, inertia_origin: wp.mat33, velocity: wp.spatial_vector):
    """Multiply a solve-frame twist by a COM-centered rigid-body inertia."""
    linear = wp.spatial_top(velocity)
    angular = wp.spatial_bottom(velocity)
    return wp.spatial_vector(
        mass * (linear - wp.cross(com, angular)),
        mass * wp.cross(com, linear) + inertia_origin * angular,
    )


# compute transform across a joint
@wp.func
def jcalc_transform(
    type: int,
    joint_axis: wp.array[wp.vec3],
    axis_start: int,
    lin_axis_count: int,
    ang_axis_count: int,
    joint_q: wp.array[float],
    q_start: int,
):
    if type == JointType.PRISMATIC:
        q = joint_q[q_start]
        axis = joint_axis[axis_start]
        X_jc = wp.transform(axis * q, wp.quat_identity())
        return X_jc

    if type == JointType.REVOLUTE:
        q = joint_q[q_start]
        axis = joint_axis[axis_start]
        X_jc = wp.transform(wp.vec3(), wp.quat_from_axis_angle(axis, q))
        return X_jc

    if type == JointType.BALL:
        qx = joint_q[q_start + 0]
        qy = joint_q[q_start + 1]
        qz = joint_q[q_start + 2]
        qw = joint_q[q_start + 3]

        X_jc = wp.transform(wp.vec3(), wp.quat(qx, qy, qz, qw))
        return X_jc

    if type == JointType.FIXED:
        X_jc = wp.transform_identity()
        return X_jc

    if type == JointType.FREE or type == JointType.DISTANCE:
        px = joint_q[q_start + 0]
        py = joint_q[q_start + 1]
        pz = joint_q[q_start + 2]

        qx = joint_q[q_start + 3]
        qy = joint_q[q_start + 4]
        qz = joint_q[q_start + 5]
        qw = joint_q[q_start + 6]

        X_jc = wp.transform(wp.vec3(px, py, pz), wp.quat(qx, qy, qz, qw))
        return X_jc

    if type == JointType.D6:
        pos = wp.vec3(0.0)
        rot = wp.quat_identity()

        # unroll for loop to ensure joint actions remain differentiable
        # (since differentiating through a for loop that updates a local variable is not supported)

        if lin_axis_count > 0:
            axis = joint_axis[axis_start + 0]
            pos += axis * joint_q[q_start + 0]
        if lin_axis_count > 1:
            axis = joint_axis[axis_start + 1]
            pos += axis * joint_q[q_start + 1]
        if lin_axis_count > 2:
            axis = joint_axis[axis_start + 2]
            pos += axis * joint_q[q_start + 2]

        ia = axis_start + lin_axis_count
        iq = q_start + lin_axis_count
        if ang_axis_count == 1:
            axis = joint_axis[ia]
            rot = wp.quat_from_axis_angle(axis, joint_q[iq])
        if ang_axis_count == 2:
            rot, _ = compute_2d_rotational_dofs(
                joint_axis[ia + 0],
                joint_axis[ia + 1],
                joint_q[iq + 0],
                joint_q[iq + 1],
                0.0,
                0.0,
            )
        if ang_axis_count == 3:
            rot, _ = compute_3d_rotational_dofs(
                joint_axis[ia + 0],
                joint_axis[ia + 1],
                joint_axis[ia + 2],
                joint_q[iq + 0],
                joint_q[iq + 1],
                joint_q[iq + 2],
                0.0,
                0.0,
                0.0,
            )

        X_jc = wp.transform(pos, rot)
        return X_jc

    # default case
    return wp.transform_identity()


# compute motion subspace and velocity for a joint
@wp.func
def jcalc_motion(
    type: int,
    joint_axis: wp.array[wp.vec3],
    lin_axis_count: int,
    ang_axis_count: int,
    X_sc: wp.transform,
    joint_qd: wp.array[float],
    qd_start: int,
    # outputs
    joint_S_s: wp.array[wp.spatial_vector],
):
    if type == JointType.PRISMATIC:
        axis = joint_axis[qd_start]
        S_s = transform_twist(X_sc, wp.spatial_vector(axis, wp.vec3()))
        v_j_s = S_s * joint_qd[qd_start]
        joint_S_s[qd_start] = S_s
        return v_j_s

    if type == JointType.REVOLUTE:
        axis = joint_axis[qd_start]
        S_s = transform_twist(X_sc, wp.spatial_vector(wp.vec3(), axis))
        v_j_s = S_s * joint_qd[qd_start]
        joint_S_s[qd_start] = S_s
        return v_j_s

    if type == JointType.D6:
        v_j_s = wp.spatial_vector()
        if lin_axis_count > 0:
            axis = joint_axis[qd_start + 0]
            S_s = transform_twist(X_sc, wp.spatial_vector(axis, wp.vec3()))
            v_j_s += S_s * joint_qd[qd_start + 0]
            joint_S_s[qd_start + 0] = S_s
        if lin_axis_count > 1:
            axis = joint_axis[qd_start + 1]
            S_s = transform_twist(X_sc, wp.spatial_vector(axis, wp.vec3()))
            v_j_s += S_s * joint_qd[qd_start + 1]
            joint_S_s[qd_start + 1] = S_s
        if lin_axis_count > 2:
            axis = joint_axis[qd_start + 2]
            S_s = transform_twist(X_sc, wp.spatial_vector(axis, wp.vec3()))
            v_j_s += S_s * joint_qd[qd_start + 2]
            joint_S_s[qd_start + 2] = S_s
        if ang_axis_count > 0:
            axis = joint_axis[qd_start + lin_axis_count + 0]
            S_s = transform_twist(X_sc, wp.spatial_vector(wp.vec3(), axis))
            v_j_s += S_s * joint_qd[qd_start + lin_axis_count + 0]
            joint_S_s[qd_start + lin_axis_count + 0] = S_s
        if ang_axis_count > 1:
            axis = joint_axis[qd_start + lin_axis_count + 1]
            S_s = transform_twist(X_sc, wp.spatial_vector(wp.vec3(), axis))
            v_j_s += S_s * joint_qd[qd_start + lin_axis_count + 1]
            joint_S_s[qd_start + lin_axis_count + 1] = S_s
        if ang_axis_count > 2:
            axis = joint_axis[qd_start + lin_axis_count + 2]
            S_s = transform_twist(X_sc, wp.spatial_vector(wp.vec3(), axis))
            v_j_s += S_s * joint_qd[qd_start + lin_axis_count + 2]
            joint_S_s[qd_start + lin_axis_count + 2] = S_s

        return v_j_s

    if type == JointType.BALL:
        S_0 = transform_twist(X_sc, wp.spatial_vector(0.0, 0.0, 0.0, 1.0, 0.0, 0.0))
        S_1 = transform_twist(X_sc, wp.spatial_vector(0.0, 0.0, 0.0, 0.0, 1.0, 0.0))
        S_2 = transform_twist(X_sc, wp.spatial_vector(0.0, 0.0, 0.0, 0.0, 0.0, 1.0))

        joint_S_s[qd_start + 0] = S_0
        joint_S_s[qd_start + 1] = S_1
        joint_S_s[qd_start + 2] = S_2

        return S_0 * joint_qd[qd_start + 0] + S_1 * joint_qd[qd_start + 1] + S_2 * joint_qd[qd_start + 2]

    if type == JointType.FIXED:
        return wp.spatial_vector()

    if type == JointType.FREE or type == JointType.DISTANCE:
        # For FREE/DISTANCE joints we treat linear/angular velocity components as
        # referenced at the root COM world point to avoid world-origin conditioning.
        q_sc = wp.transform_get_rotation(X_sc)

        v_local = wp.vec3(joint_qd[qd_start + 0], joint_qd[qd_start + 1], joint_qd[qd_start + 2])
        w_local = wp.vec3(joint_qd[qd_start + 3], joint_qd[qd_start + 4], joint_qd[qd_start + 5])
        v_j_s = wp.spatial_vector(wp.quat_rotate(q_sc, v_local), wp.quat_rotate(q_sc, w_local))

        ex = wp.quat_rotate(q_sc, wp.vec3(1.0, 0.0, 0.0))
        ey = wp.quat_rotate(q_sc, wp.vec3(0.0, 1.0, 0.0))
        ez = wp.quat_rotate(q_sc, wp.vec3(0.0, 0.0, 1.0))

        joint_S_s[qd_start + 0] = wp.spatial_vector(ex, wp.vec3())
        joint_S_s[qd_start + 1] = wp.spatial_vector(ey, wp.vec3())
        joint_S_s[qd_start + 2] = wp.spatial_vector(ez, wp.vec3())
        joint_S_s[qd_start + 3] = wp.spatial_vector(wp.vec3(), ex)
        joint_S_s[qd_start + 4] = wp.spatial_vector(wp.vec3(), ey)
        joint_S_s[qd_start + 5] = wp.spatial_vector(wp.vec3(), ez)

        return v_j_s

    wp.printf("jcalc_motion not implemented for joint type %d\n", type)

    # default case
    return wp.spatial_vector()


# computes joint space forces/torques in tau
@wp.func
def jcalc_tau(
    type: int,
    joint_S_s: wp.array[wp.spatial_vector],
    joint_f: wp.array[float],
    joint_q: wp.array[float],
    joint_qd: wp.array[float],
    joint_spring_stiffness: wp.array[float],
    joint_spring_ref: wp.array[float],
    joint_damping: wp.array[float],
    coord_start: int,
    dof_start: int,
    lin_axis_count: int,
    ang_axis_count: int,
    body_f_s: wp.spatial_vector,
    add_existing_tau: int,
    # outputs
    tau: wp.array[float],
):
    if type == JointType.BALL:
        # target_ke = joint_target_ke[dof_start]
        # target_kd = joint_target_kd[dof_start]

        for i in range(3):
            S_s = joint_S_s[dof_start + i]

            # w = joint_qd[dof_start + i]
            # r = joint_q[coord_start + i]

            value = -wp.dot(S_s, body_f_s) + joint_f[dof_start + i]
            if add_existing_tau != 0:
                value += tau[dof_start + i]
            tau[dof_start + i] = value
            # tau -= w * target_kd - r * target_ke

        return

    if type == JointType.FREE or type == JointType.DISTANCE:
        for i in range(6):
            S_s = joint_S_s[dof_start + i]
            value = -wp.dot(S_s, body_f_s) + joint_f[dof_start + i]
            if add_existing_tau != 0:
                value += tau[dof_start + i]
            tau[dof_start + i] = value

        return

    if type == JointType.PRISMATIC or type == JointType.REVOLUTE or type == JointType.D6:
        axis_count = lin_axis_count + ang_axis_count

        for i in range(axis_count):
            j = dof_start + i
            S_s = joint_S_s[j]
            # Passive spring/damping applied explicitly; the drive gains stay on the
            # implicit augmented path. These joint types have one coordinate per DOF,
            # so coord_start + i addresses the axis position.
            passive_f = joint_spring_stiffness[j] * (joint_spring_ref[j] - joint_q[coord_start + i])
            passive_f -= joint_damping[j] * joint_qd[j]
            # total torque / force on the joint (drive forces handled via augmented mass)
            value = -wp.dot(S_s, body_f_s) + joint_f[j] + passive_f
            if add_existing_tau != 0:
                value += tau[j]
            tau[j] = value

        return


@wp.func
def jcalc_integrate(
    type: int,
    child: int,
    body_com: wp.array[wp.vec3],
    X_cj: wp.transform,
    joint_q: wp.array[float],
    joint_qd: wp.array[float],
    joint_qdd: wp.array[float],
    coord_start: int,
    dof_start: int,
    lin_axis_count: int,
    ang_axis_count: int,
    dt: float,
    angular_damping: float,
    parent: int,
    # outputs
    joint_q_new: wp.array[float],
    joint_qd_new: wp.array[float],
):
    if type == JointType.FIXED:
        return

    # prismatic / revolute
    if type == JointType.PRISMATIC or type == JointType.REVOLUTE:
        qdd = joint_qdd[dof_start]
        qd = joint_qd[dof_start]
        q = joint_q[coord_start]

        qd_new = qd + qdd * dt
        q_new = q + qd_new * dt

        joint_qd_new[dof_start] = qd_new
        joint_q_new[coord_start] = q_new

        return

    # ball
    if type == JointType.BALL:
        m_j = wp.vec3(joint_qdd[dof_start + 0], joint_qdd[dof_start + 1], joint_qdd[dof_start + 2])
        w_j = wp.vec3(joint_qd[dof_start + 0], joint_qd[dof_start + 1], joint_qd[dof_start + 2])

        r_j = wp.quat(
            joint_q[coord_start + 0], joint_q[coord_start + 1], joint_q[coord_start + 2], joint_q[coord_start + 3]
        )

        # symplectic Euler
        w_j_new = w_j + m_j * dt

        drdt_j = wp.quat(w_j_new, 0.0) * r_j * 0.5

        # new orientation (normalized)
        r_j_new = wp.normalize(r_j + drdt_j * dt)

        # update joint coords
        joint_q_new[coord_start + 0] = r_j_new[0]
        joint_q_new[coord_start + 1] = r_j_new[1]
        joint_q_new[coord_start + 2] = r_j_new[2]
        joint_q_new[coord_start + 3] = r_j_new[3]

        # update joint vel
        joint_qd_new[dof_start + 0] = w_j_new[0]
        joint_qd_new[dof_start + 1] = w_j_new[1]
        joint_qd_new[dof_start + 2] = w_j_new[2]

        return

    if type == JointType.FREE or type == JointType.DISTANCE:
        a_s = wp.vec3(joint_qdd[dof_start + 0], joint_qdd[dof_start + 1], joint_qdd[dof_start + 2])
        m_s = wp.vec3(joint_qdd[dof_start + 3], joint_qdd[dof_start + 4], joint_qdd[dof_start + 5])

        v_com = wp.vec3(joint_qd[dof_start + 0], joint_qd[dof_start + 1], joint_qd[dof_start + 2])
        w_s = wp.vec3(joint_qd[dof_start + 3], joint_qd[dof_start + 4], joint_qd[dof_start + 5])

        # symplectic Euler. joint_qdd's linear rows give the acceleration of the articulation-frame
        # origin, a point fixed in the root body, so its velocity also changes by transport as the
        # body rotates: that is the omega x v term. SolverFeatherstone performs the same conversion
        # explicitly (a_com = a + alpha x x_com + omega x v_com); omitting it leaves the free base
        # short by a term proportional to the spin. The same term must appear in the velocity
        # predictor and be removed by the velocity-to-acceleration conversion (see
        # apply_free_root_transport_to_predictor / remove_free_root_transport_from_qdd), or
        # constraint rows are built against a velocity this integration never realizes.
        w_prev = w_s
        w_s = w_s + m_s * dt
        if parent < 0:
            v_com = v_com + (a_s + wp.cross(w_prev, v_com)) * dt
        else:
            # A descendant free joint's coordinate is a RELATIVE twist in the parent anchor
            # frame; the root transport rule above is not derived for it, so integrate
            # component-wise until the parent-frame transport is.
            v_com = v_com + a_s * dt
        w_s_integrate = w_s

        p_s = wp.vec3(joint_q[coord_start + 0], joint_q[coord_start + 1], joint_q[coord_start + 2])

        r_s = wp.quat(
            joint_q[coord_start + 3], joint_q[coord_start + 4], joint_q[coord_start + 5], joint_q[coord_start + 6]
        )
        # (p_s, r_s) track the child ANCHOR frame, so the lever to the COM must go through the
        # child anchor transform: with a non-identity X_cj the COM does not sit at
        # body_com[child] in anchor coordinates.
        r_ac = wp.transform_point(wp.transform_inverse(X_cj), body_com[child])

        drdt_s = wp.quat(w_s_integrate, 0.0) * r_s * 0.5
        r_s_new = wp.normalize(r_s + drdt_s * dt)

        if parent < 0:
            # Reconstruct the root anchor from the integrated COM instead of
            # advancing it with the linearized lever velocity: the COM is the
            # point whose velocity the coordinate stores, so integrate it
            # directly and place the anchor at x_com - R_new * r_ac.  A
            # force-free COM then stays stationary to roundoff, where the
            # linearized form (p += (v - w x R_old*r_ac) * dt) has
            # O(omega^2 * |r_ac| * dt^2) local error, which accumulates into
            # first-order global drift.
            # SolverFeatherstone integrates body poses around the COM the
            # same way.
            x_com_new = p_s + wp.quat_rotate(r_s, r_ac) + v_com * dt
            p_s_new = x_com_new - wp.quat_rotate(r_s_new, r_ac)
        else:
            # Descendant free joints keep the linearized relative update until
            # the moving-parent-frame transport is derived.
            dpdt_s = v_com - wp.cross(w_s_integrate, wp.quat_rotate(r_s, r_ac))
            p_s_new = p_s + dpdt_s * dt

        if parent < 0:
            w_s = w_s * (1.0 - angular_damping * dt)

        # update transform
        joint_q_new[coord_start + 0] = p_s_new[0]
        joint_q_new[coord_start + 1] = p_s_new[1]
        joint_q_new[coord_start + 2] = p_s_new[2]

        joint_q_new[coord_start + 3] = r_s_new[0]
        joint_q_new[coord_start + 4] = r_s_new[1]
        joint_q_new[coord_start + 5] = r_s_new[2]
        joint_q_new[coord_start + 6] = r_s_new[3]

        joint_qd_new[dof_start + 0] = v_com[0]
        joint_qd_new[dof_start + 1] = v_com[1]
        joint_qd_new[dof_start + 2] = v_com[2]
        joint_qd_new[dof_start + 3] = w_s[0]
        joint_qd_new[dof_start + 4] = w_s[1]
        joint_qd_new[dof_start + 5] = w_s[2]

        return

    # other joint types (compound, universal, D6)
    if type == JointType.D6:
        axis_count = lin_axis_count + ang_axis_count

        for i in range(axis_count):
            qdd = joint_qdd[dof_start + i]
            qd = joint_qd[dof_start + i]
            q = joint_q[coord_start + i]

            qd_new = qd + qdd * dt
            q_new = q + qd_new * dt

            joint_qd_new[dof_start + i] = qd_new
            joint_q_new[coord_start + i] = q_new

        return


@wp.func
def compute_link_transform(
    i: int,
    joint_type: wp.array[int],
    joint_parent: wp.array[int],
    joint_child: wp.array[int],
    joint_q_start: wp.array[int],
    joint_qd_start: wp.array[int],
    joint_q: wp.array[float],
    joint_X_p: wp.array[wp.transform],
    joint_X_c: wp.array[wp.transform],
    body_X_com: wp.array[wp.transform],
    joint_axis: wp.array[wp.vec3],
    joint_dof_dim: wp.array2d[int],
    # outputs
    body_q: wp.array[wp.transform],
    body_q_com: wp.array[wp.transform],
):
    # parent transform
    parent = joint_parent[i]
    child = joint_child[i]

    # parent transform in spatial coordinates
    X_pj = joint_X_p[i]
    X_cj = joint_X_c[i]
    # parent anchor frame in world space
    X_wpj = X_pj
    if parent >= 0:
        X_wp = body_q[parent]
        X_wpj = X_wp * X_wpj

    type = joint_type[i]
    qd_start = joint_qd_start[i]
    lin_axis_count = joint_dof_dim[i, 0]
    ang_axis_count = joint_dof_dim[i, 1]
    coord_start = joint_q_start[i]

    # compute transform across joint
    X_j = jcalc_transform(type, joint_axis, qd_start, lin_axis_count, ang_axis_count, joint_q, coord_start)

    # transform from world to joint anchor frame at child body
    X_wcj = X_wpj * X_j
    # transform from world to child body frame
    X_wc = X_wcj * wp.transform_inverse(X_cj)

    # compute transform of center of mass
    X_cm = body_X_com[child]
    X_sm = X_wc * X_cm

    # store geometry transforms
    body_q[child] = X_wc
    body_q_com[child] = X_sm


@wp.func
def spatial_cross(a: wp.spatial_vector, b: wp.spatial_vector):
    w_a = wp.spatial_bottom(a)
    v_a = wp.spatial_top(a)

    w_b = wp.spatial_bottom(b)
    v_b = wp.spatial_top(b)

    w = wp.cross(w_a, w_b)
    v = wp.cross(w_a, v_b) + wp.cross(v_a, w_b)

    return wp.spatial_vector(v, w)


@wp.func
def spatial_cross_dual(a: wp.spatial_vector, b: wp.spatial_vector):
    w_a = wp.spatial_bottom(a)
    v_a = wp.spatial_top(a)

    w_b = wp.spatial_bottom(b)
    v_b = wp.spatial_top(b)

    w = wp.cross(w_a, w_b) + wp.cross(v_a, v_b)
    v = wp.cross(w_a, v_b)

    return wp.spatial_vector(v, w)


@wp.func
def compute_link_kinematics(
    i: int,
    parent: int,
    child: int,
    parent_v_s: wp.spatial_vector,
    parent_a_s: wp.spatial_vector,
    origin: wp.vec3,
    joint_type: wp.array[int],
    joint_qd_start: wp.array[int],
    joint_qd: wp.array[float],
    joint_axis: wp.array[wp.vec3],
    joint_dof_dim: wp.array2d[int],
    body_q: wp.array[wp.transform],
    joint_X_p: wp.array[wp.transform],
    # outputs
    joint_S_s: wp.array[wp.spatial_vector],
    body_v_s: wp.array[wp.spatial_vector],
    body_a_s: wp.array[wp.spatial_vector],
):
    type = joint_type[i]
    qd_start = joint_qd_start[i]

    X_pj = joint_X_p[i]
    # X_cj = joint_X_c[i]

    # parent anchor frame in world space
    X_wpj = X_pj
    if parent >= 0:
        X_wp = body_q[parent]
        X_wpj = X_wp * X_wpj
    X_wpj_local = wp.transform(
        wp.transform_get_translation(X_wpj) - origin,
        wp.transform_get_rotation(X_wpj),
    )

    # compute motion subspace and velocity across the joint (also stores S_s to global memory)
    lin_axis_count = joint_dof_dim[i, 0]
    ang_axis_count = joint_dof_dim[i, 1]
    v_j_s = jcalc_motion(
        type,
        joint_axis,
        lin_axis_count,
        ang_axis_count,
        X_wpj_local,
        joint_qd,
        qd_start,
        joint_S_s,
    )

    # body velocity, acceleration
    v_s = parent_v_s + v_j_s
    a_s = parent_a_s + spatial_cross(v_s, v_j_s)

    body_v_s[child] = v_s
    body_a_s[child] = a_s
    return v_s, a_s


@wp.func
def compute_link_velocity(
    i: int,
    parent: int,
    child: int,
    parent_v_s: wp.spatial_vector,
    parent_a_s: wp.spatial_vector,
    origin: wp.vec3,
    gravity: wp.vec3,
    joint_type: wp.array[int],
    joint_qd_start: wp.array[int],
    joint_qd: wp.array[float],
    joint_axis: wp.array[wp.vec3],
    joint_dof_dim: wp.array2d[int],
    body_mass: wp.array[float],
    body_inertia: wp.array[wp.mat33],
    write_body_inertia: int,
    write_body_inertia_terms: int,
    body_q: wp.array[wp.transform],
    body_q_com: wp.array[wp.transform],
    joint_X_p: wp.array[wp.transform],
    # outputs
    joint_S_s: wp.array[wp.spatial_vector],
    body_I_s: wp.array[wp.spatial_matrix],
    body_inertia_terms: wp.array2d[float],
    body_v_s: wp.array[wp.spatial_vector],
    body_f_s: wp.array[wp.spatial_vector],
    body_a_s: wp.array[wp.spatial_vector],
):
    v_s, a_s = compute_link_kinematics(
        i,
        parent,
        child,
        parent_v_s,
        parent_a_s,
        origin,
        joint_type,
        joint_qd_start,
        joint_qd,
        joint_axis,
        joint_dof_dim,
        body_q,
        joint_X_p,
        joint_S_s,
        body_v_s,
        body_a_s,
    )

    # compute body forces
    X_sm = body_q_com[child]
    X_sm_local = wp.transform(
        wp.transform_get_translation(X_sm) - origin,
        wp.transform_get_rotation(X_sm),
    )
    mass = body_mass[child]

    # gravity and external forces (expressed in frame aligned with s but centered at body mass)
    f_g = mass * gravity
    com, inertia_origin = transform_com_inertia_terms(X_sm_local, mass, body_inertia[child])
    f_g_s = wp.spatial_vector(f_g, wp.cross(com, f_g))

    # body forces
    if write_body_inertia != 0:
        body_I_s[child] = assemble_com_spatial_inertia(mass, com, inertia_origin)
    if write_body_inertia_terms != 0:
        body_inertia_terms[child, 0] = com[0]
        body_inertia_terms[child, 1] = com[1]
        body_inertia_terms[child, 2] = com[2]
        for row in range(3):
            for col in range(3):
                body_inertia_terms[child, 3 + 3 * row + col] = inertia_origin[row, col]

    # The root's linear inertial wrench is NOT spurious: the solve frame is centred on a material
    # point of the root body, so that point accelerates as the body rotates and this term is what
    # carries it. SolverFeatherstone keeps it and conserves momentum; zeroing it here leaked
    # momentum on every rotating multi-link articulation.
    coriolis = spatial_cross_dual(v_s, mul_com_spatial_inertia(mass, com, inertia_origin, v_s))

    f_b_s = mul_com_spatial_inertia(mass, com, inertia_origin, a_s) + coriolis

    body_f_s[child] = f_b_s - f_g_s
    return v_s, a_s


@wp.kernel(module=_KINEMATICS_KERNEL_MODULE)
def eval_rigid_fk_id(
    articulation_start: wp.array[int],
    articulation_joint_end: wp.array[int],
    joint_type: wp.array[int],
    joint_parent: wp.array[int],
    joint_child: wp.array[int],
    joint_q_start: wp.array[int],
    joint_qd_start: wp.array[int],
    joint_q: wp.array[float],
    joint_qd: wp.array[float],
    joint_X_p: wp.array[wp.transform],
    joint_X_c: wp.array[wp.transform],
    body_X_com: wp.array[wp.transform],
    joint_axis: wp.array[wp.vec3],
    joint_dof_dim: wp.array2d[int],
    body_com: wp.array[wp.vec3],
    body_mass: wp.array[float],
    body_inertia: wp.array[wp.mat33],
    is_free_rigid: wp.array[int],
    materialize_all_body_inertia: int,
    materialize_body_inertia_terms: int,
    body_world: wp.array[int],
    gravity: wp.array[wp.vec3],
    # outputs
    body_q: wp.array[wp.transform],
    body_q_com: wp.array[wp.transform],
    articulation_origin: wp.array[wp.vec3],
    joint_S_s: wp.array[wp.spatial_vector],
    body_I_s: wp.array[wp.spatial_matrix],
    body_inertia_terms: wp.array2d[float],
    body_v_s: wp.array[wp.spatial_vector],
    body_f_s: wp.array[wp.spatial_vector],
    body_a_s: wp.array[wp.spatial_vector],
):
    """Evaluate articulation poses and the inverse-dynamics bias wrenches."""
    index = wp.tid()
    start = articulation_start[index]
    end = articulation_joint_end[index]

    for i in range(start, end):
        compute_link_transform(
            i,
            joint_type,
            joint_parent,
            joint_child,
            joint_q_start,
            joint_qd_start,
            joint_q,
            joint_X_p,
            joint_X_c,
            body_X_com,
            joint_axis,
            joint_dof_dim,
            body_q,
            body_q_com,
        )

    origin = wp.vec3()
    if start < articulation_start[index + 1]:
        root_body = joint_child[start]
        if root_body >= 0:
            origin = wp.transform_point(body_q[root_body], body_com[root_body])
    articulation_origin[index] = origin

    write_body_inertia = materialize_all_body_inertia
    if is_free_rigid[index] != 0:
        write_body_inertia = 1
    cached_child = int(-1)
    cached_v_s = wp.spatial_vector()
    cached_a_s = wp.spatial_vector()
    for i in range(start, end):
        parent = joint_parent[i]
        child = joint_child[i]
        gravity_s = gravity[body_world[child]]
        parent_v_s = wp.spatial_vector()
        parent_a_s = wp.spatial_vector()
        if parent >= 0:
            if parent == cached_child:
                parent_v_s = cached_v_s
                parent_a_s = cached_a_s
            else:
                parent_v_s = body_v_s[parent]
                parent_a_s = body_a_s[parent]
        cached_v_s, cached_a_s = compute_link_velocity(
            i,
            parent,
            child,
            parent_v_s,
            parent_a_s,
            origin,
            gravity_s,
            joint_type,
            joint_qd_start,
            joint_qd,
            joint_axis,
            joint_dof_dim,
            body_mass,
            body_inertia,
            write_body_inertia,
            materialize_body_inertia_terms,
            body_q,
            body_q_com,
            joint_X_p,
            joint_S_s,
            body_I_s,
            body_inertia_terms,
            body_v_s,
            body_f_s,
            body_a_s,
        )
        cached_child = child


@wp.kernel
def refresh_masked_body_inertia(
    articulation_joint_end: wp.array[int],
    joint_articulation: wp.array[int],
    joint_child: wp.array[int],
    mass_update_mask: wp.array[int],
    body_q_com: wp.array[wp.transform],
    articulation_origin: wp.array[wp.vec3],
    body_I_m: wp.array[wp.spatial_matrix],
    body_mass: wp.array[float],
    body_inertia: wp.array[wp.mat33],
    write_body_inertia_terms: int,
    # outputs
    body_I_s: wp.array[wp.spatial_matrix],
    body_inertia_terms: wp.array2d[float],
):
    """Materialize current link inertias selected by a reuse-step mass update mask.

    The compact COM terms feed the direct diagonal inertia owner, which bypasses ``body_I_s``; refresh them
    under the same mask so a masked inertial update reaches every mass consumer.
    """
    joint = wp.tid()
    articulation = joint_articulation[joint]
    if articulation < 0 or joint >= articulation_joint_end[articulation] or mass_update_mask[articulation] == 0:
        return
    child = joint_child[joint]
    X_sm = body_q_com[child]
    X_sm_local = wp.transform(
        wp.transform_get_translation(X_sm) - articulation_origin[articulation],
        wp.transform_get_rotation(X_sm),
    )
    body_I_s[child] = transform_spatial_inertia(X_sm_local, body_I_m[child])
    if write_body_inertia_terms != 0:
        com, inertia_origin = transform_com_inertia_terms(X_sm_local, body_mass[child], body_inertia[child])
        body_inertia_terms[child, 0] = com[0]
        body_inertia_terms[child, 1] = com[1]
        body_inertia_terms[child, 2] = com[2]
        for row in range(3):
            for col in range(3):
                body_inertia_terms[child, 3 + 3 * row + col] = inertia_origin[row, col]


@wp.func
def _compute_body_net_wrench(
    child: int,
    f_t_s: wp.spatial_vector,
    origin: wp.vec3,
    body_fb_s: wp.array[wp.spatial_vector],
    body_f_ext: wp.array[wp.spatial_vector],
    body_flags: wp.array[wp.int32],
    body_q: wp.array[wp.transform],
    body_com: wp.array[wp.vec3],
):
    """Subtract the external COM wrench in the articulation's origin frame."""
    f_ext_com = wp.spatial_vector()
    if (body_flags[child] & BodyFlags.KINEMATIC) == 0:
        f_ext_com = body_f_ext[child]
    f_ext_f = wp.spatial_bottom(f_ext_com)
    f_ext_t = wp.spatial_top(f_ext_com)
    com_world = wp.transform_point(body_q[child], body_com[child])
    com_rel = com_world - origin
    tau_origin = f_ext_f + wp.cross(com_rel, f_ext_t)
    f_ext_origin = wp.spatial_vector(f_ext_t, tau_origin)
    return body_fb_s[child] + f_t_s - f_ext_origin


@wp.func
def accumulate_articulation_tau(
    index: int,
    articulation_start: wp.array[int],
    articulation_joint_end: wp.array[int],
    joint_type: wp.array[int],
    joint_parent: wp.array[int],
    joint_child: wp.array[int],
    joint_articulation: wp.array[int],
    joint_qd_start: wp.array[int],
    joint_q_start: wp.array[int],
    joint_dof_dim: wp.array2d[int],
    joint_f: wp.array[float],
    joint_q: wp.array[float],
    joint_qd: wp.array[float],
    joint_spring_stiffness: wp.array[float],
    joint_spring_ref: wp.array[float],
    joint_damping: wp.array[float],
    joint_S_s: wp.array[wp.spatial_vector],
    body_fb_s: wp.array[wp.spatial_vector],
    body_f_ext: wp.array[wp.spatial_vector],
    body_flags: wp.array[wp.int32],
    body_q: wp.array[wp.transform],
    body_com: wp.array[wp.vec3],
    articulation_origin: wp.array[wp.vec3],
    add_existing_tau: int,
    # outputs
    body_ft_s: wp.array[wp.spatial_vector],
    tau: wp.array[float],
):
    start = articulation_start[index]
    # Tree prefix only: trailing loop-closing joints are handled as constraint rows.
    end = articulation_joint_end[index]
    count = end - start

    # compute joint forces
    for offset in range(count):
        # for backwards traversal
        i = end - offset - 1

        type = joint_type[i]
        parent = joint_parent[i]
        child = joint_child[i]
        articulation = joint_articulation[i]
        dof_start = joint_qd_start[i]
        lin_axis_count = joint_dof_dim[i, 0]
        ang_axis_count = joint_dof_dim[i, 1]
        origin = wp.vec3()
        if articulation >= 0:
            origin = articulation_origin[articulation]

        f_s = _compute_body_net_wrench(
            child, body_ft_s[child], origin, body_fb_s, body_f_ext, body_flags, body_q, body_com
        )

        # compute joint-space forces, writes out tau
        jcalc_tau(
            type,
            joint_S_s,
            joint_f,
            joint_q,
            joint_qd,
            joint_spring_stiffness,
            joint_spring_ref,
            joint_damping,
            joint_q_start[i],
            dof_start,
            lin_axis_count,
            ang_axis_count,
            f_s,
            add_existing_tau,
            tau,
        )

        if parent >= 0:
            # One thread owns the complete articulation and visits children
            # before parents, so no other thread can update this accumulator.
            body_ft_s[parent] = body_ft_s[parent] + f_s


@wp.kernel(module=_INVERSE_DYNAMICS_KERNEL_MODULE)
def eval_rigid_tau(
    articulation_start: wp.array[int],
    articulation_joint_end: wp.array[int],
    joint_type: wp.array[int],
    joint_parent: wp.array[int],
    joint_child: wp.array[int],
    joint_articulation: wp.array[int],
    joint_qd_start: wp.array[int],
    joint_q_start: wp.array[int],
    joint_dof_dim: wp.array2d[int],
    joint_f: wp.array[float],
    joint_q: wp.array[float],
    joint_qd: wp.array[float],
    joint_spring_stiffness: wp.array[float],
    joint_spring_ref: wp.array[float],
    joint_damping: wp.array[float],
    joint_S_s: wp.array[wp.spatial_vector],
    body_fb_s: wp.array[wp.spatial_vector],
    body_f_ext: wp.array[wp.spatial_vector],
    body_flags: wp.array[wp.int32],
    body_q: wp.array[wp.transform],
    body_com: wp.array[wp.vec3],
    articulation_origin: wp.array[wp.vec3],
    body_ft_s: wp.array[wp.spatial_vector],
    tau: wp.array[float],
):
    # one thread per articulation
    accumulate_articulation_tau(
        wp.tid(),
        articulation_start,
        articulation_joint_end,
        joint_type,
        joint_parent,
        joint_child,
        joint_articulation,
        joint_qd_start,
        joint_q_start,
        joint_dof_dim,
        joint_f,
        joint_q,
        joint_qd,
        joint_spring_stiffness,
        joint_spring_ref,
        joint_damping,
        joint_S_s,
        body_fb_s,
        body_f_ext,
        body_flags,
        body_q,
        body_com,
        articulation_origin,
        0,
        body_ft_s,
        tau,
    )


@wp.kernel(module=_MASS_DYNAMICS_KERNEL_MODULE)
def compute_composite_inertia(
    articulation_start: wp.array[int],
    articulation_joint_end: wp.array[int],
    mass_update_mask: wp.array[int],
    joint_ancestor: wp.array[int],
    joint_child: wp.array[int],
    body_I_s: wp.array[wp.spatial_matrix],
    # outputs
    body_I_c: wp.array[wp.spatial_matrix],
):
    art_idx = wp.tid()

    if mass_update_mask[art_idx] == 0:
        return

    start = articulation_start[art_idx]
    # Tree prefix only: trailing loop-closing joints carry no link inertia.
    end = articulation_joint_end[art_idx]
    count = end - start

    # body_I_s/body_I_c are BODY-indexed (see compute_link_velocity); index them through
    # joint_child. Joint index and child body index only coincide in loop-free models —
    # a loop joint shifts every later joint index off its body row.
    for i in range(count):
        body_I_c[joint_child[start + i]] = body_I_s[joint_child[start + i]]

    for i in range(count - 1, -1, -1):
        joint_i = start + i
        parent_joint = joint_ancestor[joint_i]

        if parent_joint >= start:
            body_I_c[joint_child[parent_joint]] += body_I_c[joint_child[joint_i]]


@wp.kernel
def cholesky_loop(
    H_group: wp.array3d[float],  # [n_arts, n_dofs, n_dofs]
    R_group: wp.array2d[float],  # [n_arts, n_dofs]
    group_to_art: wp.array[int],
    mass_update_mask: wp.array[int],
    n_dofs: int,
    # output
    L_group: wp.array3d[float],  # [n_arts, n_dofs, n_dofs]
):
    """Non-tiled Cholesky for grouped articulation storage.

    One thread per articulation, loop-based Cholesky decomposition.
    Efficient for small articulations where tile overhead dominates.
    """
    group_idx = wp.tid()
    art_idx = group_to_art[group_idx]

    if mass_update_mask[art_idx] == 0:
        return

    # Cholesky decomposition with regularization: L L^T = H + diag(R)
    for j in range(n_dofs):
        # Compute diagonal element L[j,j]
        s = H_group[group_idx, j, j] + R_group[group_idx, j]

        for k in range(j):
            r = L_group[group_idx, j, k]
            s -= r * r

        s = wp.sqrt(s)
        inv_s = 1.0 / s
        L_group[group_idx, j, j] = s

        # Compute off-diagonal elements L[i,j] for i > j
        for i in range(j + 1, n_dofs):
            s = H_group[group_idx, i, j]

            for k in range(j):
                s -= L_group[group_idx, i, k] * L_group[group_idx, j, k]

            L_group[group_idx, i, j] = s * inv_s


@wp.func
def _active_free_root_dof_start(
    free_root_joint_indices: wp.array[int],
    joint_qd_start: wp.array[int],
    kinematic_joint_mask: wp.array[int],
    root_index: int,
):
    """DOF start of an active free/distance root, or ``-1`` when kinematic.

    Structural eligibility is precomputed once; the device-side kinematic
    check remains live so model-property notifications and graph replay keep
    the same launch topology. Eligibility never depends on the numeric value
    of ``omega x v`` because its forward value can be zero while its derivative
    is not. Each root owns all six of its DOFs, so writing its three linear
    entries unconditionally races with nobody.
    """
    joint = free_root_joint_indices[root_index]
    if kinematic_joint_mask[joint] != 0:
        return -1
    return joint_qd_start[joint]


@wp.kernel
def apply_free_root_transport_to_predictor(
    free_root_joint_indices: wp.array[int],
    joint_qd_start: wp.array[int],
    kinematic_joint_mask: wp.array[int],
    joint_qd: wp.array[float],
    dt: float,
    v_hat: wp.array[float],
):
    """Lift the free root's velocity predictor onto the integrator's convention.

    ``jcalc_integrate`` realizes ``qd + (qdd + omega x v) * dt`` for the root's
    linear coordinate. Constraint rows are built against ``v_hat``, so without
    the same term here every contact, friction, and velocity-limit row sees a
    COM velocity the integrator never produces, off by ``dt * (omega x v)``.
    """
    root_index = wp.tid()
    d = _active_free_root_dof_start(free_root_joint_indices, joint_qd_start, kinematic_joint_mask, root_index)
    if d < 0:
        return
    v = wp.vec3(joint_qd[d + 0], joint_qd[d + 1], joint_qd[d + 2])
    w = wp.vec3(joint_qd[d + 3], joint_qd[d + 4], joint_qd[d + 5])
    c = wp.cross(w, v)
    v_hat[d + 0] = v_hat[d + 0] + c[0] * dt
    v_hat[d + 1] = v_hat[d + 1] + c[1] * dt
    v_hat[d + 2] = v_hat[d + 2] + c[2] * dt


@wp.func
def _gyro_skew(v: wp.vec3):
    return wp.mat33(0.0, -v[2], v[1], v[2], 0.0, -v[0], -v[1], v[0], 0.0)


@wp.func
def _gyroscopic_velocity(inertia: wp.mat33, effective_inertia: wp.mat33, omega: wp.vec3, predicted: wp.vec3, dt: float):
    """Replace the explicit gyroscopic kick with energy-preserving Cayley updates.

    Every solve has the form ``(A-S) w = (A+S) u``, with symmetric positive
    definite ``A`` and skew ``S``. Thus ``w.T A w == u.T A u`` independently
    of fixed-point convergence. Updating S from the midpoint approximates
    implicit midpoint without an unconverged Newton step injecting energy.
    External torque remains in u; only the gyroscopic kick is replaced.
    """
    scale = wp.max(effective_inertia[0, 0], wp.max(effective_inertia[1, 1], effective_inertia[2, 2]))
    a = effective_inertia / scale
    physical = inertia / scale
    inverse = wp.inverse(a)
    u = predicted + dt * (inverse * wp.cross(omega, physical * omega))
    # An energy bound on angular speed chooses inexpensive local gyro
    # microsteps. Geometry and the constraint solver still run once per step.
    # The fixed cap bounds work; energy preservation does not depend on it.
    speed_bound = wp.sqrt(wp.max(wp.dot(u, a * u) * wp.trace(inverse), 0.0))
    microsteps = wp.int32(wp.clamp(wp.ceil(2.0 * wp.abs(dt) * speed_bound), 1.0, 32.0))
    h = dt / float(microsteps)
    w = u
    for _ in range(microsteps):
        u = w
        energy = wp.dot(u, a * u)
        for _iteration in range(8):
            s = (0.5 * h) * _gyro_skew(physical * (0.5 * (u + w)))
            candidate = wp.inverse(a - s) * ((a + s) * u)
            delta = candidate - w
            w = candidate
            if wp.dot(delta, a * delta) <= 1.0e-12 * energy:
                break
    return w


@wp.kernel
def apply_free_root_velocity_corrections(
    free_root_joint_indices: wp.array[int],
    joint_qd_start: wp.array[int],
    joint_child: wp.array[int],
    body_to_articulation: wp.array[int],
    is_free_rigid: wp.array[int],
    art_group_index: wp.array[int],
    kinematic_joint_mask: wp.array[int],
    body_q: wp.array[wp.transform],
    body_inertia: wp.array[wp.mat33],
    cholesky: wp.array3d[float],
    joint_qd: wp.array[float],
    dt: float,
    v_hat: wp.array[float],
):
    """Fuse free-root transport with the isolated rigid-body gyroscopic update."""
    root_index = wp.tid()
    d = _active_free_root_dof_start(free_root_joint_indices, joint_qd_start, kinematic_joint_mask, root_index)
    if d < 0:
        return
    v = wp.vec3(joint_qd[d + 0], joint_qd[d + 1], joint_qd[d + 2])
    w = wp.vec3(joint_qd[d + 3], joint_qd[d + 4], joint_qd[d + 5])
    c = wp.cross(w, v)
    v_hat[d + 0] = v_hat[d + 0] + c[0] * dt
    v_hat[d + 1] = v_hat[d + 1] + c[1] * dt
    v_hat[d + 2] = v_hat[d + 2] + c[2] * dt

    body = joint_child[free_root_joint_indices[root_index]]
    art = body_to_articulation[body]
    if is_free_rigid[art] == 0:
        return
    predicted_world = wp.vec3(v_hat[d + 3], v_hat[d + 4], v_hat[d + 5])
    # A stationary angular predictor has no gyroscopic work to do.
    if (
        w[0] == 0.0
        and w[1] == 0.0
        and w[2] == 0.0
        and predicted_world[0] == 0.0
        and predicted_world[1] == 0.0
        and predicted_world[2] == 0.0
    ):
        return
    inertia = body_inertia[body]
    # Isotropic inertia has identically zero gyroscopic bias: keep the predictor exact.
    if (
        inertia[0, 0] == inertia[1, 1]
        and inertia[1, 1] == inertia[2, 2]
        and inertia[0, 1] == 0.0
        and inertia[0, 2] == 0.0
        and inertia[1, 0] == 0.0
        and inertia[1, 2] == 0.0
        and inertia[2, 0] == 0.0
        and inertia[2, 1] == 0.0
    ):
        return
    group = art_group_index[art]
    # At the root COM, translation and rotation decouple. The angular
    # Cholesky block includes armature and the factorization's pivot floor.
    lower = wp.mat33(0.0)
    for r in range(3):
        for c in range(r + 1):
            lower[r, c] = cholesky[group, r + 3, c + 3]
    rotation = wp.transform_get_rotation(body_q[body])
    basis = wp.quat_to_matrix(rotation)
    effective = wp.transpose(basis) * (lower * wp.transpose(lower)) * basis
    omega = wp.quat_rotate_inv(rotation, w)
    predicted = wp.quat_rotate_inv(rotation, predicted_world)
    corrected = wp.quat_rotate(rotation, _gyroscopic_velocity(inertia, effective, omega, predicted, dt))
    for k in range(3):
        v_hat[d + 3 + k] = corrected[k]


@wp.kernel
def remove_free_root_transport_from_qdd(
    free_root_joint_indices: wp.array[int],
    joint_qd_start: wp.array[int],
    kinematic_joint_mask: wp.array[int],
    joint_qd: wp.array[float],
    joint_qdd: wp.array[float],
):
    """Make ``jcalc_integrate`` reproduce the solved velocity exactly.

    The solver commits to ``v_out``; ``qdd = (v_out - qd) / dt`` alone would let
    the integrator's transport term push the realized root velocity to
    ``v_out + dt * (omega x v)``. Subtracting the term here closes the loop, and
    in the contact-free case recovers the dynamics' own ``qdd`` bit for bit.
    """
    root_index = wp.tid()
    d = _active_free_root_dof_start(free_root_joint_indices, joint_qd_start, kinematic_joint_mask, root_index)
    if d < 0:
        return
    v = wp.vec3(joint_qd[d + 0], joint_qd[d + 1], joint_qd[d + 2])
    w = wp.vec3(joint_qd[d + 3], joint_qd[d + 4], joint_qd[d + 5])
    c = wp.cross(w, v)
    joint_qdd[d + 0] = joint_qdd[d + 0] - c[0]
    joint_qdd[d + 1] = joint_qdd[d + 1] - c[1]
    joint_qdd[d + 2] = joint_qdd[d + 2] - c[2]


@wp.kernel
def integrate_generalized_joints(
    joint_type: wp.array[int],
    joint_parent: wp.array[int],
    joint_child: wp.array[int],
    joint_q_start: wp.array[int],
    joint_qd_start: wp.array[int],
    kinematic_joint_mask: wp.array[int],
    joint_dof_dim: wp.array2d[int],
    body_com: wp.array[wp.vec3],
    joint_X_c: wp.array[wp.transform],
    joint_q: wp.array[float],
    joint_qd: wp.array[float],
    joint_qdd: wp.array[float],
    dt: float,
    body_angular_damping: wp.array[float],
    joint_q_new: wp.array[float],
    joint_qd_new: wp.array[float],
):
    # one thread per joint
    index = wp.tid()

    type = joint_type[index]
    parent = joint_parent[index]
    child = joint_child[index]
    coord_start = joint_q_start[index]
    dof_start = joint_qd_start[index]
    if kinematic_joint_mask[index] != 0:
        for coord in range(coord_start, joint_q_start[index + 1]):
            joint_q_new[coord] = joint_q[coord]
        for dof in range(dof_start, joint_qd_start[index + 1]):
            joint_qd_new[dof] = joint_qd[dof]
        return

    lin_axis_count = joint_dof_dim[index, 0]
    ang_axis_count = joint_dof_dim[index, 1]

    jcalc_integrate(
        type,
        child,
        body_com,
        joint_X_c[index],
        joint_q,
        joint_qd,
        joint_qdd,
        coord_start,
        dof_start,
        lin_axis_count,
        ang_axis_count,
        dt,
        body_angular_damping[child],
        parent,
        joint_q_new,
        joint_qd_new,
    )


@wp.kernel
def compute_velocity_predictor(
    joint_qd: wp.array[float],
    kinematic_dof_mask: wp.array[int],
    dt: float,
    joint_qdd: wp.array[float],
    # outputs
    v_hat: wp.array[float],
):
    tid = wp.tid()
    if kinematic_dof_mask[tid] != 0:
        joint_qdd[tid] = 0.0
    v_hat[tid] = joint_qd[tid] + joint_qdd[tid] * dt


@wp.kernel
def update_qdd_from_velocity(
    joint_qd: wp.array[float],
    kinematic_dof_mask: wp.array[int],
    inv_dt: float,
    v_new: wp.array[float],
    # output
    joint_qdd: wp.array[float],
):
    tid = wp.tid()
    if kinematic_dof_mask[tid] != 0:
        v_new[tid] = joint_qd[tid]
        joint_qdd[tid] = 0.0
    else:
        joint_qdd[tid] = (v_new[tid] - joint_qd[tid]) * inv_dt


@wp.kernel
def compute_contact_linear_force_from_impulses(
    contact_count: wp.array[wp.int32],
    contact_normal: wp.array[wp.vec3],
    contact_world: wp.array[wp.int32],
    contact_slot: wp.array[wp.int32],
    contact_path: wp.array[wp.int32],
    contact_slots_needed: wp.array[wp.int32],
    world_impulses: wp.array2d[wp.float32],
    mf_impulses: wp.array2d[wp.float32],
    world_constraint_count: wp.array[wp.int32],
    mf_constraint_count: wp.array[wp.int32],
    inv_dt: float,
    # outputs
    rigid_contact_force: wp.array[wp.vec3],
):
    """Convert the solved normal and friction impulses of each contact into a world-frame force.

    The force acts on shape 0's body; contacts whose rows were dropped report zero.
    Contacts without friction rows (filtered by a gap threshold, or not a friction
    anchor of their patch) report their normal force only.
    """
    c = wp.tid()
    if c >= contact_count[0]:
        return

    force = wp.vec3(0.0)
    slot = contact_slot[c]
    path = contact_path[c]
    if slot >= 0 and path >= 0 and inv_dt > 0.0:
        world = contact_world[c]
        # Rows use the normal from shape 1 toward shape 0, i.e. the force on shape 0.
        normal = -contact_normal[c]
        count = mf_constraint_count[world]
        if path == 0:
            count = world_constraint_count[world]
        has_friction = contact_slots_needed[c] == 3 and slot + 2 < count
        lam_n = float(0.0)
        lam_t0 = float(0.0)
        lam_t1 = float(0.0)
        if slot < count:
            if path == 0:
                lam_n = world_impulses[world, slot]
                if has_friction:
                    lam_t0 = world_impulses[world, slot + 1]
                    lam_t1 = world_impulses[world, slot + 2]
            else:
                lam_n = mf_impulses[world, slot]
                if has_friction:
                    lam_t0 = mf_impulses[world, slot + 1]
                    lam_t1 = mf_impulses[world, slot + 2]
        tangent0, tangent1 = contact_tangent_basis(normal)
        force = lam_n * normal
        force += lam_t0 * tangent0 + lam_t1 * tangent1
        force *= inv_dt

    rigid_contact_force[c] = force


@wp.kernel
def pack_contact_linear_force_as_spatial(
    contact_count: wp.array[wp.int32],
    rigid_contact_force: wp.array[wp.vec3],
    # outputs
    contact_force: wp.array[wp.spatial_vector],
):
    """Pack linear contact forces into Newton's spatial-force contact buffer."""
    c = wp.tid()
    total_contacts = contact_count[0]
    if c >= total_contacts:
        return

    contact_force[c] = wp.spatial_vector(rigid_contact_force[c], wp.vec3(0.0))


@wp.func
def prepare_articulation_augmented_drives(
    articulation: int,
    articulation_start: wp.array[int],
    articulation_H_rows: wp.array[int],
    joint_type: wp.array[int],
    joint_qd_start: wp.array[int],
    joint_q_start: wp.array[int],
    joint_dof_dim: wp.array2d[int],
    joint_q: wp.array[float],
    joint_qd: wp.array[float],
    joint_target_ke: wp.array[float],
    joint_target_kd: wp.array[float],
    joint_target_pos: wp.array[float],
    joint_target_q_start: wp.array[int],
    joint_target_vel: wp.array[float],
    joint_effort_limit: wp.array[float],
    max_dofs: int,
    dt: float,
    reset_tau: int,
    # outputs
    row_counts: wp.array[int],
    row_dof_index: wp.array[int],
    row_K: wp.array[float],
    tau: wp.array[float],
):
    """Prepare one articulation's implicit drive rows and explicit force."""
    joint_start = articulation_start[articulation]
    joint_end = articulation_start[articulation + 1]
    if reset_tau != 0:
        dof_start = joint_qd_start[joint_start]
        dof_end = dof_start + articulation_H_rows[articulation]
        for dof_index in range(dof_start, dof_end):
            tau[dof_index] = 0.0

    if articulation_H_rows[articulation] == 0:
        row_counts[articulation] = 0
        return

    slot = int(0)
    for joint_index in range(joint_start, joint_end):
        type = joint_type[joint_index]
        if type != JointType.PRISMATIC and type != JointType.REVOLUTE and type != JointType.D6:
            continue

        axis_count = joint_dof_dim[joint_index, 0] + joint_dof_dim[joint_index, 1]
        qd_start = joint_qd_start[joint_index]
        coord_start = joint_q_start[joint_index]
        for axis in range(axis_count):
            if slot >= max_dofs:
                break
            dof_index = qd_start + axis
            ke = joint_target_ke[dof_index]
            kd = joint_target_kd[dof_index]
            if ke <= 0.0 and kd <= 0.0:
                continue

            K = ke * dt * dt + kd * dt
            if K <= 0.0:
                continue

            q = joint_q[coord_start + axis]
            qd = joint_qd[dof_index]
            target_pos = joint_target_pos[joint_target_q_start[joint_index] + axis]
            u0 = -(ke * (q - target_pos + dt * qd) + kd * (qd - joint_target_vel[dof_index]))
            effort_limit = joint_effort_limit[dof_index]
            if effort_limit > 0.0:
                u0 = wp.clamp(u0, -effort_limit, effort_limit)

            row_index = articulation * max_dofs + slot
            row_dof_index[row_index] = dof_index
            row_K[row_index] = K
            if reset_tau != 0:
                tau[dof_index] = u0
            else:
                tau[dof_index] = tau[dof_index] + u0
            slot += 1
            if slot >= max_dofs:
                break

    row_counts[articulation] = slot


@wp.kernel(module=_INVERSE_DYNAMICS_KERNEL_MODULE)
def eval_rigid_tau_and_augmented_drives(
    articulation_start: wp.array[int],
    articulation_joint_end: wp.array[int],
    articulation_H_rows: wp.array[int],
    joint_type: wp.array[int],
    joint_parent: wp.array[int],
    joint_child: wp.array[int],
    joint_articulation: wp.array[int],
    joint_qd_start: wp.array[int],
    joint_q_start: wp.array[int],
    joint_dof_dim: wp.array2d[int],
    joint_f: wp.array[float],
    joint_q: wp.array[float],
    joint_qd: wp.array[float],
    joint_spring_stiffness: wp.array[float],
    joint_spring_ref: wp.array[float],
    joint_damping: wp.array[float],
    joint_S_s: wp.array[wp.spatial_vector],
    body_fb_s: wp.array[wp.spatial_vector],
    body_f_ext: wp.array[wp.spatial_vector],
    body_flags: wp.array[wp.int32],
    body_q: wp.array[wp.transform],
    body_com: wp.array[wp.vec3],
    articulation_origin: wp.array[wp.vec3],
    joint_target_ke: wp.array[float],
    joint_target_kd: wp.array[float],
    joint_target_pos: wp.array[float],
    joint_target_q_start: wp.array[int],
    joint_target_vel: wp.array[float],
    joint_effort_limit: wp.array[float],
    max_dofs: int,
    dt: float,
    body_ft_s: wp.array[wp.spatial_vector],
    row_counts: wp.array[int],
    row_dof_index: wp.array[int],
    row_K: wp.array[float],
    tau: wp.array[float],
):
    """Accumulate articulation forces and augmented drives in one launch."""
    articulation = wp.tid()
    accumulate_articulation_tau(
        articulation,
        articulation_start,
        articulation_joint_end,
        joint_type,
        joint_parent,
        joint_child,
        joint_articulation,
        joint_qd_start,
        joint_q_start,
        joint_dof_dim,
        joint_f,
        joint_q,
        joint_qd,
        joint_spring_stiffness,
        joint_spring_ref,
        joint_damping,
        joint_S_s,
        body_fb_s,
        body_f_ext,
        body_flags,
        body_q,
        body_com,
        articulation_origin,
        0,
        body_ft_s,
        tau,
    )

    prepare_articulation_augmented_drives(
        articulation,
        articulation_start,
        articulation_H_rows,
        joint_type,
        joint_qd_start,
        joint_q_start,
        joint_dof_dim,
        joint_q,
        joint_qd,
        joint_target_ke,
        joint_target_kd,
        joint_target_pos,
        joint_target_q_start,
        joint_target_vel,
        joint_effort_limit,
        max_dofs,
        dt,
        0,
        row_counts,
        row_dof_index,
        row_K,
        tau,
    )


@wp.kernel
def build_mass_update_mask(
    global_flag: int,
    mass_update_requested: wp.array[int],
    mass_update_mask: wp.array[int],
):
    tid = wp.tid()
    flag = 1 if global_flag != 0 else 0
    if mass_update_requested[tid] != 0:
        flag = 1
    mass_update_mask[tid] = flag


# =============================================================================
# Joint Velocity-Limit Constraint Kernels
# =============================================================================
# These kernels clamp per-DOF joint speeds. They reuse the allocation / populate
# shape of the joint-position-limit kernels; a finite velocity limit allocates
# both bounds once the speed reaches the activation fraction of the limit.


@wp.kernel
def allocate_joint_velocity_limit_slots(
    articulation_start: wp.array[int],
    articulation_dof_start: wp.array[int],
    articulation_H_rows: wp.array[int],
    joint_type: wp.array[int],
    joint_qd_start: wp.array[int],
    joint_dof_dim: wp.array2d[int],
    joint_velocity_limit: wp.array[float],
    joint_qd: wp.array[float],
    velocity_limit_activation_fraction: float,
    art_to_world: wp.array[int],
    max_constraints: int,
    velocity_limit_slot: wp.array[int],
    velocity_limit_sign: wp.array[float],
    world_slot_counter: wp.array[int],
):
    """Allocate lower/upper velocity-limit rows for every finitely limited DOF.

    For each non-locked DOF of a PRISMATIC / REVOLUTE / D6 joint
    with ``joint_velocity_limit[i] > 0``, two slots are atomically reserved in
    the per-world counter. The sign encodes which side of the bilateral box
    ``[-qdot_max, +qdot_max]`` each row enforces:

    * ``sign = +1`` — lower-limit violation (``qdot_i < -qdot_max``). The row
      pushes velocity back up (``J = +e_i``, ``target_vel = -qdot_max``).
    * ``sign = -1`` — upper-limit violation (``qdot_i > +qdot_max``). The row
      pushes velocity back down (``J = -e_i``, ``target_vel = -qdot_max``).

    The matrix-free solver treats these rows as stateless PhysX-style clamp
    rows: satisfied sides apply no impulse, and a violated side applies only
    the current velocity overshoot correction.

    ``velocity_limit_activation_fraction`` proximity-gates the allocation:
    with a positive fraction the lower/upper pair is reserved only when
    ``|joint_qd[dof]| >= fraction * qdot_max``, i.e. the DOF is close enough
    to the velocity box edge that the rows could act. A fraction of ``0.0``
    short-circuits the gate (``joint_qd`` is not even read) so the default
    allocation and slot ordering are bit-identical to the historical
    always-allocate behavior. Because the gate samples the pre-solve
    velocity, a DOF that crosses the threshold during a step is clamped one
    step late.

    Outputs two entries per DOF in ``velocity_limit_slot`` (world-constraint
    row, or -1) and ``velocity_limit_sign`` (+1 / -1).
    """
    art = wp.tid()
    world = art_to_world[art]

    # Initialize all DOFs of this articulation to "no limit active"
    dof_base = articulation_dof_start[art]
    dof_count = articulation_H_rows[art]
    for d in range(dof_count):
        lower_idx = 2 * (dof_base + d)
        upper_idx = lower_idx + 1
        velocity_limit_slot[lower_idx] = -1
        velocity_limit_slot[upper_idx] = -1
        velocity_limit_sign[lower_idx] = 0.0
        velocity_limit_sign[upper_idx] = 0.0

    joint_start = articulation_start[art]
    joint_end = articulation_start[art + 1]

    for j in range(joint_start, joint_end):
        jtype = joint_type[j]
        if jtype != JointType.PRISMATIC and jtype != JointType.REVOLUTE and jtype != JointType.D6:
            continue

        lin_count = joint_dof_dim[j, 0]
        ang_count = joint_dof_dim[j, 1]
        axis_count = lin_count + ang_count
        qd_start = joint_qd_start[j]

        for axis in range(axis_count):
            dof = qd_start + axis
            qdot_max = joint_velocity_limit[dof]

            # Guard against degenerate limits. PhysX pins ``recipResponse``
            # off for ``unitResponse <= 0``; here we drop the row entirely if
            # the stored limit is non-positive (treated as "unlimited").
            if qdot_max <= 0.0:
                continue

            # Proximity gate: only reserve the row pair when the DOF speed is
            # within ``fraction * qdot_max`` of the box edge. The fraction==0
            # branch short-circuits before reading ``joint_qd`` so the default
            # path allocates exactly as before (same slots, same order).
            if velocity_limit_activation_fraction > 0.0:
                if wp.abs(joint_qd[dof]) < velocity_limit_activation_fraction * qdot_max:
                    continue

            lower_idx = 2 * dof
            upper_idx = lower_idx + 1

            lower_slot = wp.atomic_add(world_slot_counter, world, 1)
            if lower_slot < max_constraints:
                velocity_limit_slot[lower_idx] = lower_slot
                velocity_limit_sign[lower_idx] = 1.0

            upper_slot = wp.atomic_add(world_slot_counter, world, 1)
            if upper_slot < max_constraints:
                velocity_limit_slot[upper_idx] = upper_slot
                velocity_limit_sign[upper_idx] = -1.0


@wp.kernel
def populate_joint_velocity_limit_J_for_size(
    articulation_start: wp.array[int],
    articulation_dof_start: wp.array[int],
    joint_type: wp.array[int],
    joint_qd_start: wp.array[int],
    joint_dof_dim: wp.array2d[int],
    joint_velocity_limit: wp.array[float],
    art_to_world: wp.array[int],
    velocity_limit_slot: wp.array[int],
    velocity_limit_sign: wp.array[float],
    group_to_art: wp.array[int],
    # outputs
    J_group: wp.array3d[float],
    world_row_type: wp.array2d[int],
    world_row_parent: wp.array2d[int],
    world_row_mu: wp.array2d[float],
    world_phi: wp.array2d[float],
    world_target_velocity: wp.array2d[float],
):
    """Populate Jacobian and metadata for joint velocity-limit rows.

    Launched once per size group with ``dim = n_arts_of_size``. For every DOF
    whose two ``velocity_limit_slot`` entries are non-negative, writes signed
    ±1 entries into the local DOF column of the grouped Jacobian and sets the
    constraint metadata. The rows have no Baumgarte bias (``phi = 0``). The
    target velocity is ``-qdot_max`` for both sides of the box: combined with
    the sign flip on ``J``, this encodes the bilateral projection as two
    unilateral ``J*v >= target_vel`` rows.

    DOFs skipped by :func:`allocate_joint_velocity_limit_slots` (activation
    gate) carry ``velocity_limit_slot == -1`` and are skipped here through the
    same per-row slot check.
    """
    group_idx = wp.tid()
    art = group_to_art[group_idx]
    world = art_to_world[art]
    dof_start = articulation_dof_start[art]

    for j in range(articulation_start[art], articulation_start[art + 1]):
        jtype = joint_type[j]
        if jtype != JointType.PRISMATIC and jtype != JointType.REVOLUTE and jtype != JointType.D6:
            continue

        axis_count = joint_dof_dim[j, 0] + joint_dof_dim[j, 1]
        qd_start = joint_qd_start[j]
        for axis in range(axis_count):
            dof = qd_start + axis
            qdot_max = joint_velocity_limit[dof]
            if qdot_max <= 0.0:
                continue

            local_dof = dof - dof_start
            for side in range(2):
                row_idx = 2 * dof + side
                slot = velocity_limit_slot[row_idx]
                if slot < 0:
                    continue
                # Selector row J = sign * e_i on the generalized velocity; its
                # articulated response J M^-1 J^T comes from the ordinary
                # H^-1 J^T and diagonal kernels.
                J_group[group_idx, slot, local_dof] = velocity_limit_sign[row_idx]
                world_row_type[world, slot] = PGS_CONSTRAINT_TYPE_JOINT_VELOCITY_LIMIT
                world_row_parent[world, slot] = -1
                world_row_mu[world, slot] = 0.0
                world_phi[world, slot] = 0.0
                # rhs = -target + J v = qdot_max +/- qdot_i, negative exactly when
                # that side of the velocity box is violated.
                world_target_velocity[world, slot] = -qdot_max


# =============================================================================
# Multi-Articulation Contact Building Kernels
# =============================================================================
# These kernels enable contacts between multiple articulations within the same
# world. The constraint system becomes world-level instead of per-articulation.


@wp.func
def _contact_points_world(
    c: int,
    body_a: int,
    body_b: int,
    normal: wp.vec3,
    contact_point0: wp.array[wp.vec3],
    contact_point1: wp.array[wp.vec3],
    contact_thickness0: wp.array[float],
    contact_thickness1: wp.array[float],
    body_q: wp.array[wp.transform],
):
    """Return both world-space contact points on the shape surfaces (thickness applied)."""
    point_a_world = contact_point0[c] - contact_thickness0[c] * normal
    point_b_world = contact_point1[c] + contact_thickness1[c] * normal
    if body_a >= 0:
        point_a_world = wp.transform_point(body_q[body_a], contact_point0[c]) - contact_thickness0[c] * normal
    if body_b >= 0:
        point_b_world = wp.transform_point(body_q[body_b], contact_point1[c]) + contact_thickness1[c] * normal
    return point_a_world, point_b_world


@wp.func
def _allocate_world_contact_slot(
    c: int,
    contact_shape0: wp.array[int],
    contact_shape1: wp.array[int],
    contact_point0: wp.array[wp.vec3],
    contact_point1: wp.array[wp.vec3],
    contact_normal: wp.array[wp.vec3],
    contact_thickness0: wp.array[float],
    contact_thickness1: wp.array[float],
    body_q: wp.array[wp.transform],
    shape_body: wp.array[int],
    body_to_articulation: wp.array[int],
    art_to_world: wp.array[int],
    articulation_response_dof_count: wp.array[int],
    body_flags: wp.array[wp.int32],
    body_has_response_dofs: wp.array[int],
    is_free_rigid: wp.array[int],
    has_free_rigid: int,
    max_constraints: int,
    mf_max_constraints: int,
    art_model_world: wp.array[int],
    contact_gap_gate: float,
    same_articulation_contact_gap_gate: float,
    articulation_pair_contact_gap_gate: float,
    contact_friction_gap_threshold: float,
    contact_friction_articulation_pairs_only: int,
    friction_patches: FrictionPatches,
    # outputs
    contact_world: wp.array[int],
    contact_slot: wp.array[int],
    contact_art_a: wp.array[int],
    contact_art_b: wp.array[int],
    contact_slots_needed: wp.array[int],
    world_slot_counter: wp.array[int],
    contact_path: wp.array[int],
    mf_slot_counter: wp.array[int],
    dense_dropped_contact_rows: wp.array[int],
    mf_dropped_contact_rows: wp.array[int],
    dense_first_rejected_slot: wp.array[int],
    mf_first_rejected_slot: wp.array[int],
    cross_world_contacts: wp.array[int],
):
    """Classify one contact and reserve its normal row and, when it has friction, two friction rows.

    Contacts whose responding sides are only free bodies (or ground) go to the
    free-body rows (``contact_path == 1``); every other contact with a responding
    side goes to the dense rows of its world (``contact_path == 0``). Contacts
    without a responding side, beyond their gap gate or past a capacity get
    ``contact_path == -1``.

    A contact reserves one normal row, plus two adjacent friction rows when its gap
    is within the friction gap threshold and, with friction patches, when it is one
    of its region's friction anchors. ``contact_slots_needed`` records the reservation
    (1 or 3), which every later row builder follows.

    The solve world comes from the responding sides only, so a global (world ``-1``)
    kinematic or prescribed body touches the bodies of every world; its motion enters
    through the row target velocity. A contact that would couple two solve worlds (a
    dynamic global body, solved in world 0, touching a body of another world), or whose
    non-responding side keeps response DOFs in another world, cannot be solved. It gets
    ``contact_path == -1`` and is counted in ``cross_world_contacts`` for each side's
    model world, the final entry standing for global bodies.
    """
    shape_a = contact_shape0[c]
    shape_b = contact_shape1[c]

    body_a = -1
    body_b = -1
    if shape_a >= 0:
        body_a = shape_body[shape_a]
    if shape_b >= 0:
        body_b = shape_body[shape_b]

    art_a = -1
    art_b = -1
    if body_a >= 0:
        art_a = body_to_articulation[body_a]
    if body_b >= 0:
        art_b = body_to_articulation[body_b]

    a_has_dofs = art_a >= 0 and articulation_response_dof_count[art_a] > 0
    b_has_dofs = art_b >= 0 and articulation_response_dof_count[art_b] > 0
    a_can_respond = (
        a_has_dofs and body_has_response_dofs[body_a] != 0 and (body_flags[body_a] & BodyFlags.KINEMATIC) == 0
    )
    b_can_respond = (
        b_has_dofs and body_has_response_dofs[body_b] != 0 and (body_flags[body_b] & BodyFlags.KINEMATIC) == 0
    )
    contact_slots_needed[c] = 0
    if not a_can_respond and not b_can_respond:
        contact_slot[c] = -1
        contact_path[c] = -1
        return

    # Gap gates drop wide speculative contacts before any row is reserved, using the
    # same eligibility as the friction patch selection.
    normal = -contact_normal[c]
    point_a_world, point_b_world = _contact_points_world(
        c, body_a, body_b, normal, contact_point0, contact_point1, contact_thickness0, contact_thickness1, body_q
    )
    phi = wp.dot(normal, point_a_world - point_b_world)
    a_non_free = art_a >= 0 and is_free_rigid[art_a] == 0
    b_non_free = art_b >= 0 and is_free_rigid[art_b] == 0
    normal_gap_limit = contact_normal_gap_limit(
        a_non_free,
        b_non_free,
        art_a == art_b,
        contact_gap_gate,
        articulation_pair_contact_gap_gate,
        same_articulation_contact_gap_gate,
    )
    if phi > normal_gap_limit:
        contact_slot[c] = -1
        contact_path[c] = -1
        return

    # The responding sides select the solve world; a non-responding side must not need
    # response DOFs of another world.
    world = -1
    if a_can_respond:
        world = art_to_world[art_a]
    if b_can_respond:
        world_b = art_to_world[art_b]
        if world >= 0 and world_b != world:
            world = -2
        elif world != -2:
            world = world_b
    if world >= 0:
        if not a_can_respond and a_has_dofs and art_to_world[art_a] != world:
            world = -2
        if not b_can_respond and b_has_dofs and art_to_world[art_b] != world:
            world = -2
    if world == -2:
        global_slot = cross_world_contacts.shape[0] - 1
        for side in range(2):
            art = art_a
            if side == 1:
                art = art_b
            if art >= 0:
                model_world = art_model_world[art]
                if model_world < 0 or model_world >= global_slot:
                    model_world = global_slot
                wp.atomic_add(cross_world_contacts, model_world, 1)
        contact_slot[c] = -1
        contact_path[c] = -1
        return
    if world < 0:
        contact_slot[c] = -1
        contact_path[c] = -1
        return

    is_mf = False
    if has_free_rigid != 0:
        a_is_mf_compatible = not a_has_dofs or is_free_rigid[art_a] != 0
        b_is_mf_compatible = not b_has_dofs or is_free_rigid[art_b] != 0
        is_mf = a_is_mf_compatible and b_is_mf_compatible

    # Every normal row survives; only the selected patch anchors receive friction rows.
    slots_needed = 1
    add_friction = contact_friction_eligible(
        phi, a_non_free, b_non_free, contact_friction_articulation_pairs_only, contact_friction_gap_threshold
    )
    if friction_patches.enabled != 0:
        add_friction = add_friction and friction_patches.weight[c] > 0.0
    if add_friction:
        slots_needed = 3
    contact_slots_needed[c] = slots_needed

    # A reservation that does not fit records its slot instead of rolling the counter
    # back, so the finalize kernels truncate every world's row count there.
    if is_mf:
        slot = wp.atomic_add(mf_slot_counter, world, slots_needed)
        if slot + slots_needed > mf_max_constraints:
            wp.atomic_min(mf_first_rejected_slot, world, slot)
            wp.atomic_add(mf_dropped_contact_rows, world, slots_needed)
            contact_slot[c] = -1
            contact_path[c] = -1
            return
        contact_path[c] = 1
    else:
        slot = wp.atomic_add(world_slot_counter, world, slots_needed)
        if slot + slots_needed > max_constraints:
            wp.atomic_min(dense_first_rejected_slot, world, slot)
            wp.atomic_add(dense_dropped_contact_rows, world, slots_needed)
            contact_slot[c] = -1
            contact_path[c] = -1
            return
        contact_path[c] = 0
    contact_world[c] = world
    contact_slot[c] = slot
    contact_art_a[c] = art_a
    contact_art_b[c] = art_b


@wp.kernel
def allocate_world_contact_slots(
    contact_count: wp.array[int],
    total_num_threads: int,
    contact_shape0: wp.array[int],
    contact_shape1: wp.array[int],
    contact_point0: wp.array[wp.vec3],
    contact_point1: wp.array[wp.vec3],
    contact_normal: wp.array[wp.vec3],
    contact_thickness0: wp.array[float],
    contact_thickness1: wp.array[float],
    body_q: wp.array[wp.transform],
    shape_body: wp.array[int],
    body_to_articulation: wp.array[int],
    art_to_world: wp.array[int],
    articulation_response_dof_count: wp.array[int],
    body_flags: wp.array[wp.int32],
    body_has_response_dofs: wp.array[int],
    is_free_rigid: wp.array[int],
    has_free_rigid: int,
    max_constraints: int,
    mf_max_constraints: int,
    art_model_world: wp.array[int],
    contact_gap_gate: float,
    same_articulation_contact_gap_gate: float,
    articulation_pair_contact_gap_gate: float,
    contact_friction_gap_threshold: float,
    contact_friction_articulation_pairs_only: int,
    friction_patches: FrictionPatches,
    # outputs
    contact_world: wp.array[int],
    contact_slot: wp.array[int],
    contact_art_a: wp.array[int],
    contact_art_b: wp.array[int],
    contact_slots_needed: wp.array[int],
    world_slot_counter: wp.array[int],
    contact_path: wp.array[int],
    mf_slot_counter: wp.array[int],
    dense_dropped_contact_rows: wp.array[int],
    mf_dropped_contact_rows: wp.array[int],
    dense_first_rejected_slot: wp.array[int],
    mf_first_rejected_slot: wp.array[int],
    cross_world_contacts: wp.array[int],
):
    """Allocate the rows of every active contact with a grid-stride loop.

    The narrow phase increments ``contact_count`` before checking its output
    capacity, so a count above the buffer size does not describe a fully
    materialized prefix: such a frame routes no contact (and is reported through
    the solver's capacity status).
    """
    thread = wp.tid()
    total_contacts = contact_count[0]
    capacity = contact_shape0.shape[0]
    if total_contacts > capacity:
        for c in range(thread, capacity, total_num_threads):
            contact_slot[c] = -1
            contact_path[c] = -1
            contact_slots_needed[c] = 0
        return

    for c in range(thread, total_contacts, total_num_threads):
        _allocate_world_contact_slot(
            c,
            contact_shape0,
            contact_shape1,
            contact_point0,
            contact_point1,
            contact_normal,
            contact_thickness0,
            contact_thickness1,
            body_q,
            shape_body,
            body_to_articulation,
            art_to_world,
            articulation_response_dof_count,
            body_flags,
            body_has_response_dofs,
            is_free_rigid,
            has_free_rigid,
            max_constraints,
            mf_max_constraints,
            art_model_world,
            contact_gap_gate,
            same_articulation_contact_gap_gate,
            articulation_pair_contact_gap_gate,
            contact_friction_gap_threshold,
            contact_friction_articulation_pairs_only,
            friction_patches,
            contact_world,
            contact_slot,
            contact_art_a,
            contact_art_b,
            contact_slots_needed,
            world_slot_counter,
            contact_path,
            mf_slot_counter,
            dense_dropped_contact_rows,
            mf_dropped_contact_rows,
            dense_first_rejected_slot,
            mf_first_rejected_slot,
            cross_world_contacts,
        )


@wp.func
def accumulate_jacobian_row_world(
    body_index: int,
    sign: float,
    point_world: wp.vec3,
    origin: wp.vec3,
    direction: wp.vec3,
    body_to_joint: wp.array[int],
    joint_ancestor: wp.array[int],
    joint_qd_start: wp.array[int],
    joint_S_s: wp.array[wp.spatial_vector],
    art_dof_start: int,
    n_dofs: int,
    group_idx: int,
    row: int,
    J_group: wp.array3d[float],
):
    """Accumulate Jacobian contributions by walking up the kinematic tree."""
    if body_index < 0:
        return

    point_rel = point_world - origin
    curr_joint = body_to_joint[body_index]

    while curr_joint >= 0:
        dof_start = joint_qd_start[curr_joint]
        dof_end = joint_qd_start[curr_joint + 1]

        for global_dof in range(dof_start, dof_end):
            S = joint_S_s[global_dof]
            lin = wp.vec3(S[0], S[1], S[2])
            ang = wp.vec3(S[3], S[4], S[5])

            # Velocity at contact point from this joint
            v = lin + wp.cross(ang, point_rel)
            proj = wp.dot(direction, v)

            local_dof = global_dof - art_dof_start
            if local_dof >= 0 and local_dof < n_dofs:
                J_group[group_idx, row, local_dof] += sign * proj

        curr_joint = joint_ancestor[curr_joint]


@wp.func
def prescribed_contact_velocity(
    body: int,
    art: int,
    sign: float,
    point_world: wp.vec3,
    direction: wp.vec3,
    prescribed_articulation: wp.array[int],
    articulation_origin: wp.array[wp.vec3],
    body_v_s: wp.array[wp.spatial_vector],
):
    """Return one prescribed articulation's signed contact-point velocity."""
    value = float(0.0)
    if body >= 0 and art >= 0 and prescribed_articulation[art] != 0:
        twist = body_v_s[body]
        linear = wp.spatial_top(twist)
        angular = wp.spatial_bottom(twist)
        point_velocity = linear + wp.cross(angular, point_world - articulation_origin[art])
        value = sign * wp.dot(direction, point_velocity)
    return value


@wp.func
def prescribed_relative_contact_target(
    body_a: int,
    art_a: int,
    body_b: int,
    art_b: int,
    point_a_world: wp.vec3,
    point_b_world: wp.vec3,
    direction: wp.vec3,
    prescribed_articulation: wp.array[int],
    articulation_origin: wp.array[wp.vec3],
    body_v_s: wp.array[wp.spatial_vector],
):
    """Return the target velocity induced by prescribed contact motion."""
    known_jv = prescribed_contact_velocity(
        body_a,
        art_a,
        1.0,
        point_a_world,
        direction,
        prescribed_articulation,
        articulation_origin,
        body_v_s,
    )
    known_jv += prescribed_contact_velocity(
        body_b,
        art_b,
        -1.0,
        point_b_world,
        direction,
        prescribed_articulation,
        articulation_origin,
        body_v_s,
    )
    return -known_jv


@wp.func
def _contact_mu(shape_a: int, shape_b: int, shape_material_mu: wp.array[float]):
    """Return the arithmetic-mean friction coefficient of one shape pair."""
    mu = float(0.0)
    material_count = int(0)
    if shape_a >= 0:
        mu += shape_material_mu[shape_a]
        material_count += 1
    if shape_b >= 0:
        mu += shape_material_mu[shape_b]
        material_count += 1
    if material_count > 0:
        mu /= float(material_count)
    return mu


@wp.func
def mixed_contact_restitution(
    shape_a: int,
    shape_b: int,
    shape_material_restitution: wp.array[float],
):
    """Return the arithmetic-mean restitution coefficient of one shape pair.

    Finite coefficients are clamped to ``[0, 1]``; non-finite coefficients count as zero.
    """
    restitution = float(0.0)
    material_count = int(0)
    if shape_a >= 0:
        value_a = shape_material_restitution[shape_a]
        if wp.isfinite(value_a):
            restitution += wp.clamp(value_a, 0.0, 1.0)
        material_count += 1
    if shape_b >= 0:
        value_b = shape_material_restitution[shape_b]
        if wp.isfinite(value_b):
            restitution += wp.clamp(value_b, 0.0, 1.0)
        material_count += 1
    if material_count > 0:
        restitution /= float(material_count)
    return restitution


@wp.func
def contact_restitution_fires(
    phi: float,
    relative_incident: float,
    dt: float,
    restitution_velocity_threshold: float,
):
    """Return whether an incident impact qualifies for a rebound target.

    Fires only for a closing contact faster than the threshold that is already at the
    surface or is predicted to reach it during the step; the end-gap slop absorbs
    float32 residuals of rows that land exactly at contact.
    """
    if relative_incident >= -restitution_velocity_threshold:
        return False
    if phi <= _FPGS_CONTACT_END_GAP_SLOP:
        return True
    return phi + dt * relative_incident <= _FPGS_CONTACT_END_GAP_SLOP


@wp.func
def world_contact_row_dot(
    world_dof_count: wp.array[int],
    world_dof_indices: wp.array2d[int],
    world_J: wp.array3d[float],
    velocity: wp.array[float],
    world: int,
    i: int,
):
    """Dot one dense row's Jacobian with a generalized velocity."""
    out = float(0.0)
    for d in range(world_dof_count[world]):
        global_dof = world_dof_indices[world, d]
        if global_dof >= 0:
            out += world_J[world, i, d] * velocity[global_dof]
    return out


@wp.func
def mf_contact_row_dot(
    mf_J_a: wp.array3d[float],
    mf_J_b: wp.array3d[float],
    dof_a: int,
    dof_b: int,
    world_dof_indices: wp.array2d[int],
    velocity: wp.array[float],
    world: int,
    i: int,
):
    """Dot one free-body row's two body Jacobians with a generalized velocity."""
    out = float(0.0)
    if dof_a >= 0:
        for k in range(6):
            global_dof = world_dof_indices[world, dof_a + k]
            if global_dof >= 0:
                out += mf_J_a[world, i, k] * velocity[global_dof]
    if dof_b >= 0:
        for k in range(6):
            global_dof = world_dof_indices[world, dof_b + k]
            if global_dof >= 0:
                out += mf_J_b[world, i, k] * velocity[global_dof]
    return out


@wp.kernel
def prepare_world_contact_rows(
    contact_count: wp.array[int],
    total_num_threads: int,
    contact_point0: wp.array[wp.vec3],
    contact_point1: wp.array[wp.vec3],
    contact_normal: wp.array[wp.vec3],
    contact_shape0: wp.array[int],
    contact_shape1: wp.array[int],
    contact_thickness0: wp.array[float],
    contact_thickness1: wp.array[float],
    contact_world: wp.array[int],
    contact_slot: wp.array[int],
    contact_art_a: wp.array[int],
    contact_art_b: wp.array[int],
    contact_path: wp.array[int],
    contact_slots_needed: wp.array[int],
    shape_body: wp.array[int],
    body_q: wp.array[wp.transform],
    body_v_s: wp.array[wp.spatial_vector],
    prescribed_articulation: wp.array[int],
    articulation_origin: wp.array[wp.vec3],
    shape_material_mu: wp.array[float],
    shape_material_restitution: wp.array[float],
    friction_patches: FrictionPatches,
    # outputs
    world_row_type: wp.array2d[int],
    world_row_parent: wp.array2d[int],
    world_row_mu: wp.array2d[float],
    world_phi: wp.array2d[float],
    world_target_velocity: wp.array2d[float],
    world_row_restitution: wp.array2d[float],
):
    """Write the metadata of every dense contact's normal row and, when allocated, its friction rows.

    Rows of prescribed (kinematic) articulations enter through the target velocity.
    With friction patches the friction rows act at the patch anchor points and their
    ``phi`` holds the anchor's tangential displacement.
    """
    total_contacts = wp.min(contact_count[0], contact_point0.shape[0])
    for c in range(wp.tid(), total_contacts, total_num_threads):
        if contact_path[c] != 0:
            continue
        slot = contact_slot[c]
        if slot < 0:
            continue

        world = contact_world[c]
        art_a = contact_art_a[c]
        art_b = contact_art_b[c]
        # The contact normal is stored A-to-B; rows use B-to-A.
        normal = -contact_normal[c]
        shape_a = contact_shape0[c]
        shape_b = contact_shape1[c]
        body_a = -1
        body_b = -1
        if shape_a >= 0:
            body_a = shape_body[shape_a]
        if shape_b >= 0:
            body_b = shape_body[shape_b]
        point_a_world, point_b_world = _contact_points_world(
            c, body_a, body_b, normal, contact_point0, contact_point1, contact_thickness0, contact_thickness1, body_q
        )
        phi = wp.dot(normal, point_a_world - point_b_world)
        mu = _contact_mu(shape_a, shape_b, shape_material_mu)
        tangent0, tangent1 = contact_tangent_basis(normal)

        world_row_type[world, slot] = PGS_CONSTRAINT_TYPE_CONTACT
        world_row_parent[world, slot] = -1
        world_row_mu[world, slot] = mu
        world_phi[world, slot] = phi
        world_row_restitution[world, slot] = mixed_contact_restitution(shape_a, shape_b, shape_material_restitution)
        world_target_velocity[world, slot] = prescribed_relative_contact_target(
            body_a,
            art_a,
            body_b,
            art_b,
            point_a_world,
            point_b_world,
            normal,
            prescribed_articulation,
            articulation_origin,
            body_v_s,
        )
        # The allocation owns the row extent; recomputing the friction eligibility here
        # could cross a floating-point threshold and overwrite the next contact's rows.
        if contact_slots_needed[c] < 3:
            continue
        point_a_friction = point_a_world
        point_b_friction = point_b_world
        if friction_patches.enabled != 0:
            point_a_friction = friction_patches.point_a[c]
            point_b_friction = friction_patches.point_b[c]
        for k in range(2):
            tangent = tangent0
            if k == 1:
                tangent = tangent1
            row = slot + 1 + k
            world_row_type[world, row] = PGS_CONSTRAINT_TYPE_FRICTION
            world_row_parent[world, row] = slot
            world_row_mu[world, row] = mu
            world_phi[world, row] = friction_patches.phi[c][k]
            world_row_restitution[world, row] = 0.0
            world_target_velocity[world, row] = prescribed_relative_contact_target(
                body_a,
                art_a,
                body_b,
                art_b,
                point_a_friction,
                point_b_friction,
                tangent,
                prescribed_articulation,
                articulation_origin,
                body_v_s,
            )


@wp.kernel
def populate_world_J_for_compact_size(
    contact_count: wp.array[int],
    total_num_workers: int,
    contact_point0: wp.array[wp.vec3],
    contact_point1: wp.array[wp.vec3],
    contact_normal: wp.array[wp.vec3],
    contact_shape0: wp.array[int],
    contact_shape1: wp.array[int],
    contact_thickness0: wp.array[float],
    contact_thickness1: wp.array[float],
    contact_slot: wp.array[int],
    contact_art_a: wp.array[int],
    contact_art_b: wp.array[int],
    contact_path: wp.array[int],
    contact_slots_needed: wp.array[int],
    target_size: int,
    articulation_response_dof_count: wp.array[int],
    art_group_idx: wp.array[int],
    art_dof_start: wp.array[int],
    articulation_origin: wp.array[wp.vec3],
    body_response_dof_mask: wp.array[wp.uint32],
    joint_S_s: wp.array[wp.spatial_vector],
    shape_body: wp.array[int],
    body_q: wp.array[wp.transform],
    friction_patches: FrictionPatches,
    # output
    J_group: wp.array3d[float],
):
    """Project one dense contact row and DOF per lane for compact articulations."""
    worker, lane = wp.tid()
    total_contacts = wp.min(contact_count[0], contact_point0.shape[0])
    row = lane // target_size
    local_dof = lane - row * target_size
    if row >= 3 or local_dof >= target_size:
        return

    for c in range(worker, total_contacts, total_num_workers):
        if contact_path[c] != 0 or contact_slot[c] < 0 or row >= contact_slots_needed[c]:
            continue

        shape_a = contact_shape0[c]
        shape_b = contact_shape1[c]
        body_a = -1
        body_b = -1
        if shape_a >= 0:
            body_a = shape_body[shape_a]
        if shape_b >= 0:
            body_b = shape_body[shape_b]

        normal = -contact_normal[c]
        point_a, point_b = _contact_points_world(
            c, body_a, body_b, normal, contact_point0, contact_point1, contact_thickness0, contact_thickness1, body_q
        )
        direction = normal
        if row > 0:
            tangent0, tangent1 = contact_tangent_basis(normal)
            if row == 1:
                direction = tangent0
            else:
                direction = tangent1
            if friction_patches.enabled != 0:
                point_a = friction_patches.point_a[c]
                point_b = friction_patches.point_b[c]

        art_a = contact_art_a[c]
        art_b = contact_art_b[c]
        group_a = -1
        group_b = -1
        value_a = float(0.0)
        value_b = float(0.0)
        bit = wp.uint32(1) << wp.uint32(local_dof)

        if art_a >= 0 and articulation_response_dof_count[art_a] == target_size:
            group_a = art_group_idx[art_a]
            if body_a >= 0 and (body_response_dof_mask[body_a] & bit) != wp.uint32(0):
                motion_a = joint_S_s[art_dof_start[art_a] + local_dof]
                linear_a = wp.vec3(motion_a[0], motion_a[1], motion_a[2])
                angular_a = wp.vec3(motion_a[3], motion_a[4], motion_a[5])
                velocity_a = linear_a + wp.cross(angular_a, point_a - articulation_origin[art_a])
                value_a = wp.dot(direction, velocity_a)

        if art_b >= 0 and articulation_response_dof_count[art_b] == target_size:
            group_b = art_group_idx[art_b]
            if body_b >= 0 and (body_response_dof_mask[body_b] & bit) != wp.uint32(0):
                motion_b = joint_S_s[art_dof_start[art_b] + local_dof]
                linear_b = wp.vec3(motion_b[0], motion_b[1], motion_b[2])
                angular_b = wp.vec3(motion_b[3], motion_b[4], motion_b[5])
                velocity_b = linear_b + wp.cross(angular_b, point_b - articulation_origin[art_b])
                value_b = -wp.dot(direction, velocity_b)

        slot = contact_slot[c] + row
        if group_a >= 0:
            if group_b == group_a:
                J_group[group_a, slot, local_dof] = value_a + value_b
            else:
                J_group[group_a, slot, local_dof] = value_a
        if group_b >= 0 and group_b != group_a:
            J_group[group_b, slot, local_dof] = value_b


@wp.func
def _populate_world_J_for_size_contact(
    c: int,
    contact_point0: wp.array[wp.vec3],
    contact_point1: wp.array[wp.vec3],
    contact_normal: wp.array[wp.vec3],
    contact_shape0: wp.array[int],
    contact_shape1: wp.array[int],
    contact_thickness0: wp.array[float],
    contact_thickness1: wp.array[float],
    contact_slot: wp.array[int],
    contact_art_a: wp.array[int],
    contact_art_b: wp.array[int],
    contact_path: wp.array[int],
    contact_slots_needed: wp.array[int],
    target_size: int,
    articulation_response_dof_count: wp.array[int],
    art_group_idx: wp.array[int],
    art_dof_start: wp.array[int],
    articulation_origin: wp.array[wp.vec3],
    body_to_joint: wp.array[int],
    joint_ancestor: wp.array[int],
    joint_qd_start: wp.array[int],
    joint_S_s: wp.array[wp.spatial_vector],
    shape_body: wp.array[int],
    body_q: wp.array[wp.transform],
    friction_patches: FrictionPatches,
    # outputs
    J_group: wp.array3d[float],
):
    """Accumulate one dense contact's Jacobian rows into the size group's articulations."""
    if contact_path[c] != 0:
        return
    slot = contact_slot[c]
    if slot < 0:
        return

    art_a = contact_art_a[c]
    art_b = contact_art_b[c]
    normal = -contact_normal[c]
    shape_a = contact_shape0[c]
    shape_b = contact_shape1[c]
    body_a = -1
    body_b = -1
    if shape_a >= 0:
        body_a = shape_body[shape_a]
    if shape_b >= 0:
        body_b = shape_body[shape_b]
    point_a_world, point_b_world = _contact_points_world(
        c, body_a, body_b, normal, contact_point0, contact_point1, contact_thickness0, contact_thickness1, body_q
    )
    t0, t1 = contact_tangent_basis(normal)
    row_count = contact_slots_needed[c]
    friction_point_a = point_a_world
    friction_point_b = point_b_world
    if friction_patches.enabled != 0:
        friction_point_a = friction_patches.point_a[c]
        friction_point_b = friction_patches.point_b[c]

    for side in range(2):
        art = art_a
        body = body_a
        point = point_a_world
        friction_point = friction_point_a
        sign = 1.0
        if side == 1:
            art = art_b
            body = body_b
            point = point_b_world
            friction_point = friction_point_b
            sign = -1.0
        if art < 0 or articulation_response_dof_count[art] != target_size:
            continue
        group_idx = art_group_idx[art]
        dof_start = art_dof_start[art]
        origin = articulation_origin[art]
        for row in range(row_count):
            direction = normal
            row_point = point
            if row == 1:
                direction = t0
                row_point = friction_point
            elif row == 2:
                direction = t1
                row_point = friction_point
            accumulate_jacobian_row_world(
                body,
                sign,
                row_point,
                origin,
                direction,
                body_to_joint,
                joint_ancestor,
                joint_qd_start,
                joint_S_s,
                dof_start,
                target_size,
                group_idx,
                slot + row,
                J_group,
            )


@wp.kernel
def populate_world_J_for_size(
    contact_count: wp.array[int],
    total_num_threads: int,
    contact_point0: wp.array[wp.vec3],
    contact_point1: wp.array[wp.vec3],
    contact_normal: wp.array[wp.vec3],
    contact_shape0: wp.array[int],
    contact_shape1: wp.array[int],
    contact_thickness0: wp.array[float],
    contact_thickness1: wp.array[float],
    contact_slot: wp.array[int],
    contact_art_a: wp.array[int],
    contact_art_b: wp.array[int],
    contact_path: wp.array[int],
    contact_slots_needed: wp.array[int],
    target_size: int,
    articulation_response_dof_count: wp.array[int],
    art_group_idx: wp.array[int],
    art_dof_start: wp.array[int],
    articulation_origin: wp.array[wp.vec3],
    body_to_joint: wp.array[int],
    joint_ancestor: wp.array[int],
    joint_qd_start: wp.array[int],
    joint_S_s: wp.array[wp.spatial_vector],
    shape_body: wp.array[int],
    body_q: wp.array[wp.transform],
    friction_patches: FrictionPatches,
    # outputs
    J_group: wp.array3d[float],
):
    """Populate the dense contact Jacobians of one articulation-size group."""
    total_contacts = wp.min(contact_count[0], contact_point0.shape[0])
    for c in range(wp.tid(), total_contacts, total_num_threads):
        _populate_world_J_for_size_contact(
            c,
            contact_point0,
            contact_point1,
            contact_normal,
            contact_shape0,
            contact_shape1,
            contact_thickness0,
            contact_thickness1,
            contact_slot,
            contact_art_a,
            contact_art_b,
            contact_path,
            contact_slots_needed,
            target_size,
            articulation_response_dof_count,
            art_group_idx,
            art_dof_start,
            articulation_origin,
            body_to_joint,
            joint_ancestor,
            joint_qd_start,
            joint_S_s,
            shape_body,
            body_q,
            friction_patches,
            J_group,
        )


@wp.kernel
def finalize_world_constraint_counts(
    world_slot_counter: wp.array[int],
    max_constraints: int,
    slots_per_contact: int,
    first_rejected_slot: wp.array[int],
    # outputs
    world_constraint_count: wp.array[int],
):
    """Turn the monotone slot counter into the row count.

    The count is the counter truncated at the first rejected reservation and at
    ``max_constraints``: rows at or past the first rejected slot were reserved by
    contacts the allocator dropped, so every accepted contact lies below it.

    The ``slots_per_contact`` argument is accepted for backwards
    compatibility but is no longer used for rounding, because the
    constraint buffer may now contain a mix of 3-row contact groups and
    single-row joint-limit constraints.
    """
    world = wp.tid()
    count = wp.min(world_slot_counter[world], first_rejected_slot[world])
    if count > max_constraints:
        count = max_constraints
    world_constraint_count[world] = count


@wp.kernel
def apply_augmented_mass_diagonal_grouped(
    group_to_art: wp.array[int],
    articulation_dof_start: wp.array[int],
    n_dofs: int,
    max_dofs: int,
    mass_update_mask: wp.array[int],
    row_counts: wp.array[int],
    row_dof_index: wp.array[int],
    row_K: wp.array[float],
    # outputs
    H_group: wp.array3d[float],  # [n_arts, n_dofs, n_dofs]
):
    """Apply augmented mass diagonal for grouped H storage."""
    idx = wp.tid()
    articulation = group_to_art[idx]

    if mass_update_mask[articulation] == 0:
        return

    count = row_counts[articulation]
    if count == 0:
        return

    dof_start = articulation_dof_start[articulation]

    for i in range(count):
        row_index = articulation * max_dofs + i
        dof = row_dof_index[row_index]
        local = dof - dof_start
        if local < 0 or local >= n_dofs:
            continue

        K = row_K[row_index]
        if K <= 0.0:
            continue

        H_group[idx, local, local] += K


@wp.kernel
def update_body_qd_from_featherstone(
    body_v_s: wp.array[wp.spatial_vector],
    body_q: wp.array[wp.transform],
    body_com: wp.array[wp.vec3],
    body_to_articulation: wp.array[int],
    articulation_origin: wp.array[wp.vec3],
    body_qd_out: wp.array[wp.spatial_vector],
):
    tid = wp.tid()

    twist = body_v_s[tid]  # spatial twist about origin
    v0 = wp.spatial_top(twist)
    w = wp.spatial_bottom(twist)

    X_wb = body_q[tid]
    com_local = body_com[tid]
    com_world = wp.transform_point(X_wb, com_local)
    art = body_to_articulation[tid]
    origin = wp.vec3()
    if art >= 0:
        origin = articulation_origin[art]
    com_rel = com_world - origin

    v_com = v0 + wp.cross(w, com_rel)

    body_qd_out[tid] = wp.spatial_vector(v_com, w)


# =============================================================================
# World-Level PGS and Velocity Kernels for Multi-Articulation
# =============================================================================


@wp.kernel
def compute_world_contact_bias(
    world_constraint_count: wp.array[int],
    world_phi: wp.array2d[float],
    world_row_type: wp.array2d[int],
    world_target_velocity: wp.array2d[float],
    pgs_beta: float,
    friction_anchor_beta: float,
    contact_speculative_scale: float,
    contact_w: float,
    dt: float,
    # outputs
    world_rhs: wp.array2d[float],
    world_row_w: wp.array2d[float],
):
    """Compute the bias of each dense row; the solve adds ``J v`` every iteration.

    ``rhs = -target_velocity + bias``. Penetrating contacts and violated joint limits
    get the Baumgarte term ``beta * phi / dt``; separated contacts may close
    ``contact_speculative_scale`` times their gap during the step, and limits within
    their activation gap the remaining gap. Friction rows of persistent patches get
    ``friction_anchor_beta * phi / dt`` from the anchor displacement ``phi`` (zero for
    point friction). Velocity-limit rows have no position bias.

    ``world_row_w`` receives the regularization weight of each row: ``contact_w`` for
    penetrating contacts and 1 otherwise. It is written only when regularization is on
    (``contact_w < 1``).
    """
    world = wp.tid()
    inv_dt = 1.0 / dt
    for i in range(world_constraint_count[world]):
        phi = world_phi[world, i]
        row_type = world_row_type[world, i]
        rhs = -world_target_velocity[world, i]
        row_w = float(1.0)
        if row_type == PGS_CONSTRAINT_TYPE_CONTACT:
            if phi <= 0.0:
                rhs += pgs_beta * phi * inv_dt
                row_w = contact_w
            else:
                rhs += contact_speculative_scale * phi * inv_dt
        elif row_type == PGS_CONSTRAINT_TYPE_FRICTION:
            rhs += friction_anchor_beta * phi * inv_dt
        elif row_type == PGS_CONSTRAINT_TYPE_JOINT_LIMIT:
            if phi < 0.0:
                rhs += pgs_beta * phi * inv_dt
            else:
                rhs += phi * inv_dt
        world_rhs[world, i] = rhs
        if contact_w < 1.0:
            world_row_w[world, i] = row_w


@wp.kernel
def apply_world_contact_restitution(
    world_constraint_count: wp.array[int],
    max_constraints: int,
    world_dof_count: wp.array[int],
    world_phi: wp.array2d[float],
    world_row_type: wp.array2d[int],
    world_target_velocity: wp.array2d[float],
    world_row_restitution: wp.array2d[float],
    world_incident_velocity: wp.array[float],
    world_dof_indices: wp.array2d[int],
    world_J: wp.array3d[float],
    dt: float,
    restitution_velocity_threshold: float,
    write_row_w: int,
    # in/out
    world_rhs: wp.array2d[float],
    world_row_w: wp.array2d[float],
):
    """Replace the bias of an impacting dense contact by its rebound target.

    The incident velocity is the unconstrained velocity of the step. A contact whose
    rebound fires (see :func:`contact_restitution_fires`) gets ``rhs = -target +
    e * u_incident``, solved rigidly (weight 1); the solve adds the live ``J v``.
    """
    tid = wp.tid()
    world = tid // max_constraints
    i = tid - world * max_constraints
    if i >= world_constraint_count[world]:
        return
    if world_row_type[world, i] != PGS_CONSTRAINT_TYPE_CONTACT:
        return
    restitution = world_row_restitution[world, i]
    if restitution <= 0.0:
        return

    phi = world_phi[world, i]
    target_vel = world_target_velocity[world, i]
    relative_incident = (
        world_contact_row_dot(world_dof_count, world_dof_indices, world_J, world_incident_velocity, world, i)
        - target_vel
    )
    if contact_restitution_fires(phi, relative_incident, dt, restitution_velocity_threshold):
        world_rhs[world, i] = -target_vel + restitution * relative_incident
        if write_row_w != 0:
            world_row_w[world, i] = 1.0


@wp.kernel
def compute_world_contact_velocity_bias(
    world_constraint_count: wp.array[int],
    max_constraints: int,
    world_dof_count: wp.array[int],
    world_phi: wp.array2d[float],
    world_row_type: wp.array2d[int],
    world_target_velocity: wp.array2d[float],
    world_row_restitution: wp.array2d[float],
    world_position_velocity: wp.array[float],
    world_incident_velocity: wp.array[float],
    world_dof_indices: wp.array2d[int],
    world_J: wp.array3d[float],
    dt: float,
    restitution_velocity_threshold: float,
    # outputs
    world_rhs: wp.array2d[float],
):
    """Build the dense right-hand side of the velocity-only iterations.

    The rows keep their position-solve Jacobians but drop the position bias. A positive
    gap row whose end gap after the position solve, ``phi + dt * (J v_position -
    target)``, stays above the slop keeps its speculative allowance ``phi / dt``, so a
    falling body is not stopped short of the surface; impacting contacts keep their
    rebound target. Joint limits keep their speculative allowance.
    """
    tid = wp.tid()
    world = tid // max_constraints
    i = tid - world * max_constraints
    if i >= world_constraint_count[world]:
        return
    inv_dt = 1.0 / dt
    phi = world_phi[world, i]
    row_type = world_row_type[world, i]
    target_vel = world_target_velocity[world, i]
    rhs = -target_vel

    if row_type == PGS_CONSTRAINT_TYPE_CONTACT:
        restitution = world_row_restitution[world, i]
        relative_incident = float(0.0)
        bounce = False
        if restitution > 0.0:
            relative_incident = (
                world_contact_row_dot(world_dof_count, world_dof_indices, world_J, world_incident_velocity, world, i)
                - target_vel
            )
            bounce = contact_restitution_fires(phi, relative_incident, dt, restitution_velocity_threshold)
        if bounce:
            rhs += restitution * relative_incident
        elif phi > 0.0:
            jv_position = world_contact_row_dot(
                world_dof_count, world_dof_indices, world_J, world_position_velocity, world, i
            )
            end_gap = phi + dt * (jv_position - target_vel)
            if end_gap > _FPGS_CONTACT_END_GAP_SLOP:
                rhs += phi * inv_dt
    elif row_type == PGS_CONSTRAINT_TYPE_JOINT_LIMIT:
        if phi >= 0.0:
            rhs += phi * inv_dt

    world_rhs[world, i] = rhs


@wp.kernel
def prepare_world_impulses(
    world_constraint_count: wp.array[int],
    max_constraints: int,
    warmstart: int,
    # in/out
    world_impulses: wp.array2d[float],
):
    """Cold-start the dense rows of each world before the contact warm-start gather.

    Cold solves clear the active rows only. Warm-started solves clear the whole
    capacity: only contact and friction rows carry an identity across steps, so every
    other row starts cold and no stale impulse survives a change of the row layout.
    """
    world = wp.tid()
    clear_count = max_constraints
    if warmstart == 0:
        clear_count = world_constraint_count[world]
    for i in range(clear_count):
        world_impulses[world, i] = 0.0


# =============================================================================
# Contact warm start
# =============================================================================


@wp.func
def _transport_warmstart_contact(
    normal: wp.vec3,
    previous_normal: wp.vec3,
    world: int,
    slot: int,
    count: int,
    row_type: wp.array2d[int],
    row_parent: wp.array2d[int],
    row_mu: wp.array2d[float],
    impulses: wp.array2d[float],
):
    """Rotate a carried tangent impulse into the current tangent frame and clamp it to the current cone.

    A carried normal impulse is dropped when it is not finite or the normal reversed.
    """
    normal_impulse = impulses[world, slot]
    if not wp.isfinite(normal_impulse) or wp.dot(normal, previous_normal) <= 0.0:
        normal_impulse = 0.0
    normal_impulse = wp.max(normal_impulse, 0.0)
    impulses[world, slot] = normal_impulse
    if slot + 2 >= count:
        return
    if (
        row_type[world, slot + 1] != PGS_CONSTRAINT_TYPE_FRICTION
        or row_type[world, slot + 2] != PGS_CONSTRAINT_TYPE_FRICTION
        or row_parent[world, slot + 1] != slot
        or row_parent[world, slot + 2] != slot
    ):
        return

    # Collision normals are A-to-B; every contact Jacobian uses B-to-A.
    old_t0, old_t1 = contact_tangent_basis(-previous_normal)
    new_t0, new_t1 = contact_tangent_basis(-normal)
    tangent_world = impulses[world, slot + 1] * old_t0 + impulses[world, slot + 2] * old_t1
    tangent = wp.vec2(wp.dot(tangent_world, new_t0), wp.dot(tangent_world, new_t1))
    magnitude = wp.length(tangent)
    radius = wp.max(row_mu[world, slot + 1] * normal_impulse, 0.0)
    if not wp.isfinite(magnitude) or radius <= 0.0:
        tangent = wp.vec2(0.0)
    elif magnitude > radius:
        tangent *= radius / magnitude
    impulses[world, slot + 1] = tangent[0]
    impulses[world, slot + 2] = tangent[1]


@wp.kernel
def snapshot_step_warmstart(
    dt: float,
    contact_generation: wp.array[wp.int32],
    contact_stream: int,
    # outputs
    history: wp.array[float],
    history_generation: wp.array[wp.int32],
    history_stream: wp.array[wp.int32],
):
    """Record the step and the contact set it solved on the device, so graph replays stay current.

    ``contact_stream`` identifies the contact buffer (0: the step had none); generations
    are only comparable within one buffer.
    """
    history[0] = dt
    history_stream[0] = contact_stream
    if contact_stream != 0:
        history_generation[0] = contact_generation[0]
    else:
        history_generation[0] = CONTACT_GENERATION_NONE


@wp.func
def warmstart_history_relation(
    contact_generation: wp.array[wp.int32],
    contact_stream: int,
    history_generation: wp.array[wp.int32],
    history_stream: wp.array[wp.int32],
):
    """Relate the current contact set to the solved history: 0 unrelated, 1 same set, 2 next collision pass."""
    if contact_stream == 0 or history_stream[0] != contact_stream:
        return 0
    generation = contact_generation[0]
    previous = history_generation[0]
    if previous == CONTACT_GENERATION_NONE:
        return 0
    if generation == previous:
        return 1
    # Collision passes advance the generation by one, wrapping like Contacts does.
    following = previous + 1
    if previous == 2147483647:
        following = 0
    if generation == following:
        return 2
    return 0


@wp.kernel
def gather_contact_warmstart(
    contact_count: wp.array[int],
    route: int,
    contact_path: wp.array[int],
    contact_slot: wp.array[int],
    contact_world: wp.array[int],
    match_index: wp.array[int],
    prev_slot_sorted: wp.array[int],
    prev_impulses: wp.array2d[float],
    prev_row_type: wp.array2d[int],
    prev_row_parent: wp.array2d[int],
    constraint_count: wp.array[int],
    row_type: wp.array2d[int],
    row_parent: wp.array2d[int],
    contact_normal: wp.array[wp.vec3],
    previous_normal: wp.array[wp.vec3],
    row_mu: wp.array2d[float],
    decay: float,
    dt: float,
    history: wp.array[float],
    contact_generation: wp.array[wp.int32],
    contact_stream: int,
    history_generation: wp.array[wp.int32],
    history_stream: wp.array[wp.int32],
    max_constraints: int,
    # in/out
    impulses: wp.array2d[float],
):
    """Seed the impulses of one row family (dense or free-body) from the previous step by contact identity.

    ``prev_slot_sorted`` maps a contact's index in the contact set the previous step
    solved to its first row then, so contacts keep their impulses when their rows move.
    When the solver steps again on the same contact set (same buffer ``contact_stream``,
    unchanged generation) the index is ``c`` itself. After exactly one collision pass
    into that buffer, ``match_index[c]`` is contact ``c``'s index in the solved set.
    Any other relation (another buffer, skipped collision passes, no history) starts
    every contact cold, since the match indices then refer to an unsolved set.
    A friction row is seeded only when both steps allocated it for the contact
    (contacts may change between one and three rows). Unmatched contacts stay cold.
    Seeded impulses are scaled by ``decay`` and by ``dt / history[0]``, the ratio of the
    step to the previous one (impulses are proportional to the step for steady loads),
    then the tangent impulse is transported to the current tangent frame and clamped to
    the current friction cone.
    """
    c = wp.tid()
    if c >= contact_count[0] or contact_path[c] != route:
        return
    new_slot = contact_slot[c]
    if new_slot < 0:
        return
    world = contact_world[c]
    count = constraint_count[world]
    if new_slot >= count:
        return

    relation = warmstart_history_relation(contact_generation, contact_stream, history_generation, history_stream)
    if relation == 0:
        return
    mi = match_index[c]
    if relation == 1:
        mi = c
    if mi < 0:
        return
    dt_scale = warmstart_dt_scale(dt, history)
    prev_slot = prev_slot_sorted[mi]
    if prev_slot < 0 or prev_slot >= max_constraints:
        return

    if (
        row_type[world, new_slot] == PGS_CONSTRAINT_TYPE_CONTACT
        and prev_row_type[world, prev_slot] == PGS_CONSTRAINT_TYPE_CONTACT
    ):
        impulses[world, new_slot] = decay * dt_scale * prev_impulses[world, prev_slot]
    for r in range(1, 3):
        new_r = new_slot + r
        prev_r = prev_slot + r
        if (
            new_r < count
            and prev_r < max_constraints
            and row_type[world, new_r] == PGS_CONSTRAINT_TYPE_FRICTION
            and row_parent[world, new_r] == new_slot
            and prev_row_type[world, prev_r] == PGS_CONSTRAINT_TYPE_FRICTION
            and prev_row_parent[world, prev_r] == prev_slot
        ):
            impulses[world, new_r] = decay * dt_scale * prev_impulses[world, prev_r]

    _transport_warmstart_contact(
        contact_normal[c],
        previous_normal[mi],
        world,
        new_slot,
        count,
        row_type,
        row_parent,
        row_mu,
        impulses,
    )


@wp.kernel
def snapshot_contact_warmstart(
    contact_count: wp.array[int],
    contact_path: wp.array[int],
    contact_slot: wp.array[int],
    contact_normal: wp.array[wp.vec3],
    previous_dense_slot: wp.array[int],
    previous_mf_slot: wp.array[int],
    previous_normal: wp.array[wp.vec3],
):
    """Record each contact's first row per family and its normal for the next step's gather."""
    contact = wp.tid()
    count = contact_count[0]
    active = contact < count and count <= contact_normal.shape[0]
    path = int(-1)
    slot = int(-1)
    if active:
        path = contact_path[contact]
        slot = contact_slot[contact]
        previous_normal[contact] = contact_normal[contact]
    previous_dense_slot[contact] = wp.where(path == 0, slot, -1)
    previous_mf_slot[contact] = wp.where(path == 1, slot, -1)


@wp.kernel
def snapshot_row_warmstart(
    impulses: wp.array2d[float],
    row_type: wp.array2d[int],
    row_parent: wp.array2d[int],
    has_contacts: int,
    # outputs
    prev_impulses: wp.array2d[float],
    prev_row_type: wp.array2d[int],
    prev_row_parent: wp.array2d[int],
):
    """Save one row family's impulses and row layout; a step without contacts carries nothing."""
    world, row = wp.tid()
    prev_impulses[world, row] = impulses[world, row]
    if has_contacts != 0:
        prev_row_type[world, row] = row_type[world, row]
        prev_row_parent[world, row] = row_parent[world, row]
    else:
        prev_row_type[world, row] = -1
        prev_row_parent[world, row] = -1


@wp.kernel
def reset_row_warmstart(
    world_mask: wp.array[wp.bool],
    world_count: int,
    global_shares_world_zero: int,
    prev_impulses: wp.array2d[float],
    prev_row_type: wp.array2d[int],
    prev_row_parent: wp.array2d[int],
):
    """Discard the carried impulses of the worlds selected by a ``(world_count + 1,)`` mask.

    Global (world ``-1``) articulations are solved in world 0's rows, so a reset of the
    global entry also discards world 0's carry while a responding global articulation
    exists.
    """
    world, row = wp.tid()
    if world_mask:
        selected = False
        if world < world_count:
            selected = world_mask[world]
        if world == 0 and global_shares_world_zero != 0 and world_mask[world_count]:
            selected = True
        if not selected:
            return
    prev_impulses[world, row] = 0.0
    prev_row_type[world, row] = -1
    prev_row_parent[world, row] = -1


@wp.kernel
def apply_world_impulses_to_velocity(
    world_constraint_count: wp.array[int],
    world_dof_indices: wp.array2d[int],
    max_world_dofs: int,
    Y_world: wp.array3d[float],
    world_impulses: wp.array2d[float],
    # in/out
    velocity: wp.array[float],
):
    """Add the velocity ``Y lambda`` of the seeded dense impulses to the world velocity."""
    tid = wp.tid()
    world = tid // max_world_dofs
    d = tid - world * max_world_dofs
    global_dof = world_dof_indices[world, d]
    if global_dof < 0:
        return
    delta_v = float(0.0)
    for i in range(world_constraint_count[world]):
        delta_v += Y_world[world, i, d] * world_impulses[world, i]
    velocity[global_dof] += delta_v


@wp.kernel
def build_mf_body_map(
    mf_constraint_count: wp.array[int],
    mf_body_a: wp.array2d[int],
    mf_body_b: wp.array2d[int],
    body_to_articulation: wp.array[int],
    art_dof_start: wp.array[int],
    max_mf_bodies: int,
    # outputs
    mf_body_list: wp.array2d[int],
    mf_body_dof_start: wp.array2d[int],
    mf_body_count: wp.array[int],
    mf_local_body_a: wp.array2d[int],
    mf_local_body_b: wp.array2d[int],
):
    """Build each world's table of free bodies with rows and map every row's bodies into it."""
    world = wp.tid()
    m = mf_constraint_count[world]
    n_bodies = int(0)
    for i in range(m):
        for side in range(2):
            body = mf_body_a[world, i]
            if side == 1:
                body = mf_body_b[world, i]
            local = int(-1)
            if body >= 0:
                for b in range(n_bodies):
                    if mf_body_list[world, b] == body:
                        local = b
                        break
                if local < 0 and n_bodies < max_mf_bodies:
                    local = n_bodies
                    mf_body_list[world, n_bodies] = body
                    mf_body_dof_start[world, n_bodies] = art_dof_start[body_to_articulation[body]]
                    n_bodies += 1
            if side == 0:
                mf_local_body_a[world, i] = local
            else:
                mf_local_body_b[world, i] = local
    mf_body_count[world] = n_bodies


@wp.kernel
def apply_mf_warmstart_impulses(
    mf_constraint_count: wp.array[int],
    mf_body_count: wp.array[int],
    mf_body_dof_start: wp.array2d[int],
    mf_local_body_a: wp.array2d[int],
    mf_local_body_b: wp.array2d[int],
    mf_MiJt_a: wp.array3d[float],
    mf_MiJt_b: wp.array3d[float],
    mf_impulses: wp.array2d[float],
    max_mf_bodies: int,
    # in/out
    velocity: wp.array[float],
):
    """Add the velocity of the seeded free-body impulses, so velocity and impulses start consistent.

    The solve updates the velocity by ``M^-1 J^T (lambda_new - lambda_old)``; without
    this, retracting a seeded normal impulse would pull the bodies together.
    """
    tid = wp.tid()
    component = tid % 6
    local_body = (tid // 6) % max_mf_bodies
    world = tid // (6 * max_mf_bodies)
    if local_body >= mf_body_count[world]:
        return

    global_dof = mf_body_dof_start[world, local_body] + component
    delta_velocity = float(0.0)
    for i in range(mf_constraint_count[world]):
        impulse = mf_impulses[world, i]
        if impulse == 0.0:
            continue
        if mf_local_body_a[world, i] == local_body:
            delta_velocity += mf_MiJt_a[world, i, component] * impulse
        if mf_local_body_b[world, i] == local_body:
            delta_velocity += mf_MiJt_b[world, i, component] * impulse
    velocity[global_dof] += delta_velocity


# =============================================================================
# Fully Matrix-Free PGS Kernels (velocity-space Jacobi)
# =============================================================================


@wp.kernel
def diag_from_JY_par_art(
    J_group: wp.array3d[float],  # [n_arts_of_size, max_constraints, n_dofs]
    Y_group: wp.array3d[float],  # [n_arts_of_size, max_constraints, n_dofs]
    group_to_art: wp.array[int],
    art_to_world: wp.array[int],
    world_constraint_count: wp.array[int],
    n_dofs: int,
    max_constraints: int,
    n_arts: int,
    # output
    world_diag: wp.array2d[float],
):
    """Compute diagonal of Delassus from J and Y without assembling the full matrix.

    diag[w,c] += sum_k J[idx,c,k] * Y[idx,c,k]. Thread dim: n_arts * max_constraints.
    """
    tid = wp.tid()
    c = tid % max_constraints
    idx = tid // max_constraints
    if idx >= n_arts:
        return
    art = group_to_art[idx]
    world = art_to_world[art]
    if c >= world_constraint_count[world]:
        return
    val = float(0.0)
    for k in range(n_dofs):
        val += J_group[idx, c, k] * Y_group[idx, c, k]
    if val != 0.0:
        wp.atomic_add(world_diag, world, c, val)


@wp.kernel
def accumulate_group_diag_worlds(
    group_diag: wp.array2d[float],
    world_group_art_start: wp.array[int],
    world_group_to_art: wp.array[int],
    art_group_idx: wp.array[int],
    world_constraint_count: wp.array[int],
    max_constraints: int,
    # output
    world_diag: wp.array2d[float],
):
    """Accumulate one size group's response diagonal in deterministic world order."""
    tid = wp.tid()
    c = tid % max_constraints
    world = tid // max_constraints
    if c >= world_constraint_count[world]:
        return
    value = float(0.0)
    for offset in range(world_group_art_start[world], world_group_art_start[world + 1]):
        art = world_group_to_art[offset]
        value += group_diag[art_group_idx[art], c]
    if value != 0.0:
        world_diag[world, c] += value


@wp.kernel
def gather_JY_to_world(
    group_to_art: wp.array[int],
    art_to_world: wp.array[int],
    articulation_world_dof_offset: wp.array[int],
    world_constraint_count: wp.array[int],
    J_group: wp.array3d[float],
    Y_group: wp.array3d[float],
    n_dofs: int,
    max_constraints: int,
    n_arts: int,
    # outputs
    J_world: wp.array3d[float],
    Y_world: wp.array3d[float],
):
    """Gather per-size-group J/Y into world-indexed arrays.

    Thread dim: n_arts * max_constraints * n_dofs.
    """
    tid = wp.tid()
    d = tid % n_dofs
    remainder = tid // n_dofs
    c = remainder % max_constraints
    idx = remainder // max_constraints
    if idx >= n_arts:
        return
    art = group_to_art[idx]
    world = art_to_world[art]
    if c >= world_constraint_count[world]:
        return
    local_d = articulation_world_dof_offset[art] + d
    # Write unconditionally (including zeros) so J_world/Y_world don't need pre-zeroing
    J_world[world, c, local_d] = J_group[idx, c, d]
    Y_world[world, c, local_d] = Y_group[idx, c, d]


@wp.kernel
def diag_from_JY_world(
    world_constraint_count: wp.array[int],
    world_dof_count: wp.array[int],
    J_world: wp.array3d[float],
    Y_world: wp.array3d[float],
    max_constraints: int,
    # output
    world_diag: wp.array2d[float],
):
    """Compute the matrix-free Delassus diagonal from world-indexed J/Y buffers."""
    tid = wp.tid()
    row = tid % max_constraints
    world = tid // max_constraints
    if row >= world_constraint_count[world]:
        return

    value = float(0.0)
    for dof in range(world_dof_count[world]):
        value += J_world[world, row, dof] * Y_world[world, row, dof]
    world_diag[world, row] = value


# =============================================================================
# Matrix-Free PGS Kernels for Free Rigid Bodies
# =============================================================================


@wp.func
def _build_mf_contact_row(
    c: int,
    contact_point0: wp.array[wp.vec3],
    contact_point1: wp.array[wp.vec3],
    contact_normal: wp.array[wp.vec3],
    contact_shape0: wp.array[int],
    contact_shape1: wp.array[int],
    contact_thickness0: wp.array[float],
    contact_thickness1: wp.array[float],
    contact_world: wp.array[int],
    contact_slot: wp.array[int],
    contact_path: wp.array[int],
    contact_art_a: wp.array[int],
    contact_art_b: wp.array[int],
    articulation_response_dof_count: wp.array[int],
    articulation_origin: wp.array[wp.vec3],
    shape_body: wp.array[int],
    body_q: wp.array[wp.transform],
    body_v_s: wp.array[wp.spatial_vector],
    prescribed_articulation: wp.array[int],
    has_target_velocity: int,
    shape_material_mu: wp.array[float],
    contact_slots_needed: wp.array[int],
    shape_material_restitution: wp.array[float],
    friction_patches: FrictionPatches,
    friction_anchor_beta: float,
    # outputs
    mf_body_a: wp.array2d[int],
    mf_body_b: wp.array2d[int],
    mf_J_a: wp.array3d[float],
    mf_J_b: wp.array3d[float],
    mf_row_type: wp.array2d[int],
    mf_row_parent: wp.array2d[int],
    mf_row_mu: wp.array2d[float],
    mf_phi: wp.array2d[float],
    mf_target_velocity: wp.array2d[float],
    mf_row_restitution: wp.array2d[float],
):
    """Build the normal row and, when allocated, the two friction rows of a free-body contact.

    A free body's generalized velocity is ``[v, omega]`` of its articulation origin, so
    each side's row is ``J = [d, r x d]`` with ``r`` the contact point relative to that
    origin. With friction patches the friction rows act at the patch anchor points and
    their ``phi`` holds ``friction_anchor_beta`` times the anchor's tangential displacement.
    """
    if contact_path[c] != 1:
        return
    slot = contact_slot[c]
    if slot < 0:
        return

    world = contact_world[c]
    shape_a = contact_shape0[c]
    shape_b = contact_shape1[c]
    normal = -contact_normal[c]
    body_a = -1
    body_b = -1
    if shape_a >= 0:
        body_a = shape_body[shape_a]
    if shape_b >= 0:
        body_b = shape_body[shape_b]

    # Zero-DOF articulations contribute collision geometry but own no response;
    # treat them like ground after using their transforms for the contact points.
    response_body_a = -1
    response_body_b = -1
    if body_a >= 0 and contact_art_a[c] >= 0 and articulation_response_dof_count[contact_art_a[c]] > 0:
        response_body_a = body_a
    if body_b >= 0 and contact_art_b[c] >= 0 and articulation_response_dof_count[contact_art_b[c]] > 0:
        response_body_b = body_b

    point_a_world, point_b_world = _contact_points_world(
        c, body_a, body_b, normal, contact_point0, contact_point1, contact_thickness0, contact_thickness1, body_q
    )
    phi = wp.dot(normal, point_a_world - point_b_world)
    mu = _contact_mu(shape_a, shape_b, shape_material_mu)
    restitution = mixed_contact_restitution(shape_a, shape_b, shape_material_restitution)
    t0, t1 = contact_tangent_basis(normal)

    for row_offset in range(contact_slots_needed[c]):
        row_idx = slot + row_offset
        d = normal
        if row_offset == 1:
            d = t0
        elif row_offset == 2:
            d = t1
        row_point_a = point_a_world
        row_point_b = point_b_world
        if row_offset > 0 and friction_patches.enabled != 0:
            row_point_a = friction_patches.point_a[c]
            row_point_b = friction_patches.point_b[c]

        if response_body_a >= 0:
            ang_a = wp.cross(row_point_a - articulation_origin[contact_art_a[c]], d)
            mf_J_a[world, row_idx, 0] = d[0]
            mf_J_a[world, row_idx, 1] = d[1]
            mf_J_a[world, row_idx, 2] = d[2]
            mf_J_a[world, row_idx, 3] = ang_a[0]
            mf_J_a[world, row_idx, 4] = ang_a[1]
            mf_J_a[world, row_idx, 5] = ang_a[2]
        if response_body_b >= 0:
            ang_b = wp.cross(row_point_b - articulation_origin[contact_art_b[c]], d)
            mf_J_b[world, row_idx, 0] = -d[0]
            mf_J_b[world, row_idx, 1] = -d[1]
            mf_J_b[world, row_idx, 2] = -d[2]
            mf_J_b[world, row_idx, 3] = -ang_b[0]
            mf_J_b[world, row_idx, 4] = -ang_b[1]
            mf_J_b[world, row_idx, 5] = -ang_b[2]

        mf_body_a[world, row_idx] = response_body_a
        mf_body_b[world, row_idx] = response_body_b
        mf_row_mu[world, row_idx] = mu
        if row_offset == 0:
            mf_row_type[world, row_idx] = PGS_CONSTRAINT_TYPE_CONTACT
            mf_row_parent[world, row_idx] = -1
            mf_phi[world, row_idx] = phi
            mf_row_restitution[world, row_idx] = restitution
        else:
            mf_row_type[world, row_idx] = PGS_CONSTRAINT_TYPE_FRICTION
            mf_row_parent[world, row_idx] = slot
            mf_phi[world, row_idx] = friction_anchor_beta * friction_patches.phi[c][row_offset - 1]
            mf_row_restitution[world, row_idx] = 0.0
        if has_target_velocity != 0:
            mf_target_velocity[world, row_idx] = prescribed_relative_contact_target(
                body_a,
                contact_art_a[c],
                body_b,
                contact_art_b[c],
                row_point_a,
                row_point_b,
                d,
                prescribed_articulation,
                articulation_origin,
                body_v_s,
            )


@wp.kernel
def build_mf_contact_rows(
    contact_count: wp.array[int],
    total_num_threads: int,
    contact_point0: wp.array[wp.vec3],
    contact_point1: wp.array[wp.vec3],
    contact_normal: wp.array[wp.vec3],
    contact_shape0: wp.array[int],
    contact_shape1: wp.array[int],
    contact_thickness0: wp.array[float],
    contact_thickness1: wp.array[float],
    contact_world: wp.array[int],
    contact_slot: wp.array[int],
    contact_path: wp.array[int],
    contact_art_a: wp.array[int],
    contact_art_b: wp.array[int],
    articulation_response_dof_count: wp.array[int],
    articulation_origin: wp.array[wp.vec3],
    shape_body: wp.array[int],
    body_q: wp.array[wp.transform],
    body_v_s: wp.array[wp.spatial_vector],
    prescribed_articulation: wp.array[int],
    has_target_velocity: int,
    shape_material_mu: wp.array[float],
    contact_slots_needed: wp.array[int],
    shape_material_restitution: wp.array[float],
    friction_patches: FrictionPatches,
    friction_anchor_beta: float,
    # outputs
    mf_body_a: wp.array2d[int],
    mf_body_b: wp.array2d[int],
    mf_J_a: wp.array3d[float],
    mf_J_b: wp.array3d[float],
    mf_row_type: wp.array2d[int],
    mf_row_parent: wp.array2d[int],
    mf_row_mu: wp.array2d[float],
    mf_phi: wp.array2d[float],
    mf_target_velocity: wp.array2d[float],
    mf_row_restitution: wp.array2d[float],
):
    """Build the free-body contact rows with a grid-stride loop."""
    total_contacts = wp.min(contact_count[0], contact_point0.shape[0])
    for c in range(wp.tid(), total_contacts, total_num_threads):
        _build_mf_contact_row(
            c,
            contact_point0,
            contact_point1,
            contact_normal,
            contact_shape0,
            contact_shape1,
            contact_thickness0,
            contact_thickness1,
            contact_world,
            contact_slot,
            contact_path,
            contact_art_a,
            contact_art_b,
            articulation_response_dof_count,
            articulation_origin,
            shape_body,
            body_q,
            body_v_s,
            prescribed_articulation,
            has_target_velocity,
            shape_material_mu,
            contact_slots_needed,
            shape_material_restitution,
            friction_patches,
            friction_anchor_beta,
            mf_body_a,
            mf_body_b,
            mf_J_a,
            mf_J_b,
            mf_row_type,
            mf_row_parent,
            mf_row_mu,
            mf_phi,
            mf_target_velocity,
            mf_row_restitution,
        )


@wp.kernel
def allocate_rigid_velocity_limit_slots(
    free_rigid_body_indices: wp.array[int],
    body_to_articulation: wp.array[int],
    art_to_world: wp.array[int],
    is_free_rigid: wp.array[int],
    body_flags: wp.array[wp.int32],
    rigid_body_max_linear_velocity: wp.array[float],
    rigid_body_max_angular_velocity: wp.array[float],
    articulation_root_dof_start: wp.array[int],
    joint_qd: wp.array[float],
    velocity_limit_activation_fraction: float,
    mf_max_constraints: int,
    rigid_velocity_limit_slot: wp.array[int],
    rigid_velocity_limit_sign: wp.array[float],
    mf_slot_counter: wp.array[int],
):
    """Allocate two signed MF velocity-limit rows per limited rigid velocity axis.

    Free-rigid generalized velocity is stored as six root DOFs:
    ``[lin_x, lin_y, lin_z, ang_x, ang_y, ang_z]``.  Each finite max speed
    contributes lower/upper unilateral rows with Jacobians ``+e_i`` and
    ``-e_i`` so the same stateless row projection used by articulated joint
    velocity limits can clamp the current scalar speed.

    ``velocity_limit_activation_fraction`` proximity-gates the allocation per
    axis, mirroring :func:`allocate_joint_velocity_limit_slots`: a positive
    fraction reserves the lower/upper pair only when the axis speed read from
    ``joint_qd`` at the free root's DOFs satisfies
    ``|qd[axis]| >= fraction * limit``. A fraction of ``0.0`` short-circuits
    the gate (``articulation_root_dof_start`` / ``joint_qd`` are not read) so
    the default allocation and slot ordering match the historical behavior
    exactly.
    """
    candidate = wp.tid()
    body = free_rigid_body_indices[candidate]
    base = 12 * candidate

    for row in range(12):
        rigid_velocity_limit_slot[base + row] = -1
        rigid_velocity_limit_sign[base + row] = 0.0

    art = body_to_articulation[body]
    if art < 0:
        return
    if is_free_rigid[art] == 0:
        return
    if (body_flags[body] & BodyFlags.KINEMATIC) != 0:
        return

    world = art_to_world[art]
    if world < 0:
        return

    lin_limit = rigid_body_max_linear_velocity[body]
    ang_limit = rigid_body_max_angular_velocity[body]

    for axis in range(6):
        limit = lin_limit
        if axis >= 3:
            limit = ang_limit
        if limit <= 0.0 or not wp.isfinite(limit):
            continue

        # Proximity gate: only reserve the pair when this axis is within
        # ``fraction * limit`` of the box edge. The fraction==0 branch
        # short-circuits before any velocity read so the default path
        # allocates exactly as before (same slots, same order).
        if velocity_limit_activation_fraction > 0.0:
            root_dof = articulation_root_dof_start[art] + axis
            if wp.abs(joint_qd[root_dof]) < velocity_limit_activation_fraction * limit:
                continue

        lower_idx = base + 2 * axis
        upper_idx = lower_idx + 1

        lower_slot = wp.atomic_add(mf_slot_counter, world, 1)
        if lower_slot < mf_max_constraints:
            rigid_velocity_limit_slot[lower_idx] = lower_slot
            rigid_velocity_limit_sign[lower_idx] = 1.0

        upper_slot = wp.atomic_add(mf_slot_counter, world, 1)
        if upper_slot < mf_max_constraints:
            rigid_velocity_limit_slot[upper_idx] = upper_slot
            rigid_velocity_limit_sign[upper_idx] = -1.0


@wp.kernel
def populate_rigid_velocity_limit_rows(
    free_rigid_body_indices: wp.array[int],
    body_to_articulation: wp.array[int],
    art_to_world: wp.array[int],
    is_free_rigid: wp.array[int],
    rigid_body_max_linear_velocity: wp.array[float],
    rigid_body_max_angular_velocity: wp.array[float],
    rigid_velocity_limit_slot: wp.array[int],
    rigid_velocity_limit_sign: wp.array[float],
    # outputs
    mf_body_a: wp.array2d[int],
    mf_body_b: wp.array2d[int],
    mf_J_a: wp.array3d[float],
    mf_J_b: wp.array3d[float],
    mf_row_type: wp.array2d[int],
    mf_row_parent: wp.array2d[int],
    mf_row_mu: wp.array2d[float],
    mf_phi: wp.array2d[float],
):
    """Populate MF rows for rigid-body linear/angular velocity limits."""
    candidate = wp.tid()
    body = free_rigid_body_indices[candidate]
    art = body_to_articulation[body]
    if art < 0:
        return
    if is_free_rigid[art] == 0:
        return

    world = art_to_world[art]
    if world < 0:
        return

    lin_limit = rigid_body_max_linear_velocity[body]
    ang_limit = rigid_body_max_angular_velocity[body]
    base = 12 * candidate

    for axis in range(6):
        limit = lin_limit
        if axis >= 3:
            limit = ang_limit

        for side in range(2):
            row_idx = base + 2 * axis + side
            slot = rigid_velocity_limit_slot[row_idx]
            if slot < 0:
                continue

            sign = rigid_velocity_limit_sign[row_idx]
            for k in range(6):
                mf_J_a[world, slot, k] = 0.0
                mf_J_b[world, slot, k] = 0.0
            mf_J_a[world, slot, axis] = sign

            mf_body_a[world, slot] = body
            mf_body_b[world, slot] = -1
            mf_row_type[world, slot] = PGS_CONSTRAINT_TYPE_JOINT_VELOCITY_LIMIT
            mf_row_parent[world, slot] = -1
            mf_row_mu[world, slot] = 0.0
            # For velocity-limit rows mf_phi stores qdot_max; contact rows
            # use it as geometric gap. Row type disambiguates the meaning.
            mf_phi[world, slot] = limit


@wp.func
def spatial_matrix_block_inverse(M: wp.spatial_matrix):
    """Invert a 6x6 spatial matrix using 3x3 block inversion.

    Partition M = [A B; C D] into 3x3 blocks, then:
        S = D - C * A^-1 * B   (Schur complement)
        M^-1 = [A^-1 + A^-1*B*S^-1*C*A^-1,  -A^-1*B*S^-1]
               [-S^-1*C*A^-1,                 S^-1]
    """
    A = wp.mat33(
        M[0, 0],
        M[0, 1],
        M[0, 2],
        M[1, 0],
        M[1, 1],
        M[1, 2],
        M[2, 0],
        M[2, 1],
        M[2, 2],
    )
    B = wp.mat33(
        M[0, 3],
        M[0, 4],
        M[0, 5],
        M[1, 3],
        M[1, 4],
        M[1, 5],
        M[2, 3],
        M[2, 4],
        M[2, 5],
    )
    C = wp.mat33(
        M[3, 0],
        M[3, 1],
        M[3, 2],
        M[4, 0],
        M[4, 1],
        M[4, 2],
        M[5, 0],
        M[5, 1],
        M[5, 2],
    )
    D = wp.mat33(
        M[3, 3],
        M[3, 4],
        M[3, 5],
        M[4, 3],
        M[4, 4],
        M[4, 5],
        M[5, 3],
        M[5, 4],
        M[5, 5],
    )

    Ainv = wp.inverse(A)
    AinvB = Ainv * B
    S = D - C * AinvB
    Sinv = wp.inverse(S)
    SinvCAinv = Sinv * C * Ainv

    # Top-left: Ainv + AinvB * SinvCAinv
    TL = Ainv + AinvB * SinvCAinv
    # Top-right: -AinvB * Sinv
    TR = -AinvB * Sinv
    # Bottom-left: -SinvCAinv
    BL = -SinvCAinv
    # Bottom-right: Sinv
    BR = Sinv

    return wp.spatial_matrix(
        TL[0, 0],
        TL[0, 1],
        TL[0, 2],
        TR[0, 0],
        TR[0, 1],
        TR[0, 2],
        TL[1, 0],
        TL[1, 1],
        TL[1, 2],
        TR[1, 0],
        TR[1, 1],
        TR[1, 2],
        TL[2, 0],
        TL[2, 1],
        TL[2, 2],
        TR[2, 0],
        TR[2, 1],
        TR[2, 2],
        BL[0, 0],
        BL[0, 1],
        BL[0, 2],
        BR[0, 0],
        BR[0, 1],
        BR[0, 2],
        BL[1, 0],
        BL[1, 1],
        BL[1, 2],
        BR[1, 0],
        BR[1, 1],
        BR[1, 2],
        BL[2, 0],
        BL[2, 1],
        BL[2, 2],
        BR[2, 0],
        BR[2, 1],
        BR[2, 2],
    )


@wp.kernel
def compute_mf_body_Hinv(
    free_rigid_body_indices: wp.array[int],
    body_I_s: wp.array[wp.spatial_matrix],
    is_free_rigid: wp.array[int],
    body_to_articulation: wp.array[int],
    body_flags: wp.array[wp.int32],
    articulation_dof_start: wp.array[int],
    joint_armature: wp.array[float],
    # outputs
    mf_body_Hinv: wp.array[wp.spatial_matrix],
):
    """Compute H^-1 = inverse(body_I_s + diag(armature)) for free rigid bodies.

    For root free joints, H = body_I_s in articulation-local coordinates, plus the free
    joint's armature on the diagonal as in the articulated mass matrix. This remains a
    full 6x6 matrix for bodies with non-zero CoM offsets.
    """
    b = free_rigid_body_indices[wp.tid()]
    art = body_to_articulation[b]
    if art < 0:
        return
    if is_free_rigid[art] == 0:
        return
    if (body_flags[b] & BodyFlags.KINEMATIC) != 0:
        mf_body_Hinv[b] = wp.spatial_matrix(0.0)
        return

    H = body_I_s[b]
    dof_start = articulation_dof_start[art]
    for k in range(6):
        H[k, k] = H[k, k] + joint_armature[dof_start + k]
    mf_body_Hinv[b] = spatial_matrix_block_inverse(H)


@wp.kernel
def compute_mf_effective_mass_and_rhs(
    mf_constraint_count: wp.array[int],
    mf_body_a: wp.array2d[int],
    mf_body_b: wp.array2d[int],
    mf_J_a: wp.array3d[float],
    mf_J_b: wp.array3d[float],
    mf_body_Hinv: wp.array[wp.spatial_matrix],
    mf_phi: wp.array2d[float],
    mf_row_type: wp.array2d[int],
    mf_target_velocity: wp.array2d[float],
    mf_row_restitution: wp.array2d[float],
    has_target_velocity: int,
    body_to_articulation: wp.array[int],
    articulation_dof_start: wp.array[int],
    incident_velocity: wp.array[float],
    rigid_body_max_depenetration_velocity: wp.array[float],
    pgs_cfm: float,
    pgs_beta: float,
    contact_w: float,
    dt: float,
    contact_speculative_scale: float,
    restitution_velocity_threshold: float,
    mf_max_constraints: int,
    # outputs
    mf_eff_mass_inv: wp.array2d[float],
    mf_MiJt_a: wp.array3d[float],
    mf_MiJt_b: wp.array3d[float],
    mf_rhs: wp.array2d[float],
    mf_row_w: wp.array2d[float],
):
    """Compute the response ``H^-1 J^T``, inverse effective mass and bias of each free-body row.

    The effective mass is ``J_a H_a^-1 J_a^T + J_b H_b^-1 J_b^T + cfm`` with the full
    6x6 inverse spatial inertia of each free body. The right-hand side holds only the
    bias; the solve recomputes ``J v`` every iteration. Penetrating contacts get the
    Baumgarte term, bounded by the bodies' maximum depenetration velocity; separated
    contacts may close ``contact_speculative_scale`` times their gap during the step.
    An impacting contact whose rebound fires gets its restitution target instead.
    Patch friction rows get their anchor bias (``phi`` already holds the gain).

    ``mf_row_w`` receives the regularization weight (``contact_w`` for penetrating
    contacts without a rebound, 1 otherwise) when regularization is on.
    """
    tid = wp.tid()
    world = tid // mf_max_constraints
    i = tid % mf_max_constraints
    if i >= mf_constraint_count[world]:
        return

    ba = mf_body_a[world, i]
    bb = mf_body_b[world, i]
    Ja = wp.spatial_vector(
        mf_J_a[world, i, 0],
        mf_J_a[world, i, 1],
        mf_J_a[world, i, 2],
        mf_J_a[world, i, 3],
        mf_J_a[world, i, 4],
        mf_J_a[world, i, 5],
    )
    Jb = wp.spatial_vector(
        mf_J_b[world, i, 0],
        mf_J_b[world, i, 1],
        mf_J_b[world, i, 2],
        mf_J_b[world, i, 3],
        mf_J_b[world, i, 4],
        mf_J_b[world, i, 5],
    )

    d = pgs_cfm
    if ba >= 0:
        MiJt_a = mf_body_Hinv[ba] * Ja
        d += wp.dot(Ja, MiJt_a)
        for k in range(6):
            mf_MiJt_a[world, i, k] = MiJt_a[k]
    if bb >= 0:
        MiJt_b = mf_body_Hinv[bb] * Jb
        d += wp.dot(Jb, MiJt_b)
        for k in range(6):
            mf_MiJt_b[world, i, k] = MiJt_b[k]

    if d > 0.0:
        mf_eff_mass_inv[world, i] = 1.0 / d
    else:
        mf_eff_mass_inv[world, i] = 0.0

    bias = float(0.0)
    row_w = float(1.0)
    rtype = mf_row_type[world, i]
    if rtype == PGS_CONSTRAINT_TYPE_CONTACT:
        phi_val = mf_phi[world, i]
        if phi_val <= 0.0:
            # Speculative rows stay rigid so a closing contact reaches the surface
            # instead of leaking closing speed into penetration.
            row_w = contact_w
        if phi_val < 0.0:
            bias = pgs_beta * phi_val / dt
            max_depen = 1.0e20
            if ba >= 0:
                max_depen = rigid_body_max_depenetration_velocity[ba]
            if bb >= 0:
                max_depen_b = rigid_body_max_depenetration_velocity[bb]
                if max_depen_b > 0.0 and wp.isfinite(max_depen_b):
                    if max_depen_b < max_depen:
                        max_depen = max_depen_b
            if max_depen > 0.0 and wp.isfinite(max_depen):
                bias = wp.max(bias, -max_depen)
        else:
            bias = contact_speculative_scale * phi_val / dt
        restitution = mf_row_restitution[world, i]
        if restitution > 0.0:
            relative_incident = float(0.0)
            if ba >= 0:
                dof_start = articulation_dof_start[body_to_articulation[ba]]
                for k in range(6):
                    relative_incident += mf_J_a[world, i, k] * incident_velocity[dof_start + k]
            if bb >= 0:
                dof_start = articulation_dof_start[body_to_articulation[bb]]
                for k in range(6):
                    relative_incident += mf_J_b[world, i, k] * incident_velocity[dof_start + k]
            if has_target_velocity != 0:
                relative_incident -= mf_target_velocity[world, i]
            if contact_restitution_fires(phi_val, relative_incident, dt, restitution_velocity_threshold):
                bias = restitution * relative_incident
                # An impact is impulsive, not a spring: keep the rebound exact.
                row_w = 1.0
    elif rtype == PGS_CONSTRAINT_TYPE_FRICTION:
        bias = mf_phi[world, i] / dt
    elif rtype == PGS_CONSTRAINT_TYPE_JOINT_VELOCITY_LIMIT:
        bias = mf_phi[world, i]

    if has_target_velocity != 0:
        bias -= mf_target_velocity[world, i]
    mf_rhs[world, i] = bias
    if contact_w < 1.0:
        mf_row_w[world, i] = row_w


@wp.kernel
def compute_mf_velocity_rhs(
    mf_constraint_count: wp.array[int],
    mf_dof_a: wp.array2d[int],
    mf_dof_b: wp.array2d[int],
    mf_J_a: wp.array3d[float],
    mf_J_b: wp.array3d[float],
    world_dof_indices: wp.array2d[int],
    mf_phi: wp.array2d[float],
    mf_row_type: wp.array2d[int],
    mf_target_velocity: wp.array2d[float],
    mf_row_restitution: wp.array2d[float],
    has_target_velocity: int,
    dt: float,
    position_velocity: wp.array[float],
    incident_velocity: wp.array[float],
    restitution_velocity_threshold: float,
    mf_max_constraints: int,
    # outputs
    mf_rhs: wp.array2d[float],
):
    """Build the free-body right-hand side of the velocity-only iterations.

    Mirrors :func:`compute_world_contact_velocity_bias`: no position bias, except that a
    positive-gap contact whose end gap after the position solve stays above the slop
    keeps ``phi / dt`` and an impacting contact keeps its rebound target. Velocity-limit
    rows keep their bound.
    """
    tid = wp.tid()
    world = tid // mf_max_constraints
    i = tid % mf_max_constraints
    if i >= mf_constraint_count[world]:
        return

    target_velocity = float(0.0)
    if has_target_velocity != 0:
        target_velocity = mf_target_velocity[world, i]
    bias = float(0.0)
    row_type = mf_row_type[world, i]
    if row_type == PGS_CONSTRAINT_TYPE_CONTACT:
        phi_val = mf_phi[world, i]
        dof_a = mf_dof_a[world, i]
        dof_b = mf_dof_b[world, i]
        restitution = mf_row_restitution[world, i]
        relative_incident = float(0.0)
        bounce = False
        if restitution > 0.0:
            relative_incident = (
                mf_contact_row_dot(mf_J_a, mf_J_b, dof_a, dof_b, world_dof_indices, incident_velocity, world, i)
                - target_velocity
            )
            bounce = contact_restitution_fires(phi_val, relative_incident, dt, restitution_velocity_threshold)
        if bounce:
            bias = restitution * relative_incident
        elif phi_val > 0.0:
            jv_position = mf_contact_row_dot(
                mf_J_a, mf_J_b, dof_a, dof_b, world_dof_indices, position_velocity, world, i
            )
            end_gap = phi_val + dt * (jv_position - target_velocity)
            if end_gap > _FPGS_CONTACT_END_GAP_SLOP:
                bias = phi_val / dt
    elif row_type == PGS_CONSTRAINT_TYPE_JOINT_VELOCITY_LIMIT:
        bias = mf_phi[world, i]

    mf_rhs[world, i] = bias - target_velocity


@wp.kernel
def finalize_mf_constraint_counts(
    mf_slot_counter: wp.array[int],
    mf_max_constraints: int,
    slots_per_contact: int,
    first_rejected_slot: wp.array[int],
    # outputs
    mf_constraint_count: wp.array[int],
):
    """Turn the monotone MF slot counter into the row count.

    Truncates at the first rejected reservation and at ``mf_max_constraints``
    (see :func:`finalize_world_constraint_counts`). ``slots_per_contact`` is
    kept for call-site compatibility.  The MF buffer may contain a mix of 3-row
    normal+friction contacts and 1-row speculative normal contacts, so rounding
    to a fixed stride would drop valid rows.
    """
    world = wp.tid()
    count = wp.min(mf_slot_counter[world], first_rejected_slot[world])
    if count > mf_max_constraints:
        count = mf_max_constraints
    mf_constraint_count[world] = count


@wp.kernel
def compute_mf_world_dof_offsets(
    mf_constraint_count: wp.array[int],
    mf_body_a: wp.array2d[int],
    mf_body_b: wp.array2d[int],
    body_to_articulation: wp.array[int],
    articulation_world_dof_offset: wp.array[int],
    mf_max_constraints: int,
    # outputs
    mf_dof_a: wp.array2d[int],
    mf_dof_b: wp.array2d[int],
):
    """Compute world-relative DOF offsets for each MF contact body.

    For each MF constraint, stores the articulation's compact response
    offset. The two-phase GS kernel uses these offsets to index its shared
    velocity vector.
    """
    tid = wp.tid()
    world = tid // mf_max_constraints
    c = tid % mf_max_constraints
    if c >= mf_constraint_count[world]:
        return
    ba = mf_body_a[world, c]
    bb = mf_body_b[world, c]
    if ba >= 0:
        mf_dof_a[world, c] = articulation_world_dof_offset[body_to_articulation[ba]]
    else:
        mf_dof_a[world, c] = -1
    if bb >= 0:
        mf_dof_b[world, c] = articulation_world_dof_offset[body_to_articulation[bb]]
    else:
        mf_dof_b[world, c] = -1


@wp.kernel
def finalize_world_diag_cfm(
    world_constraint_count: wp.array[int],
    pgs_cfm: float,
    # in/out
    world_diag: wp.array2d[float],
):
    """Add constraint force mixing to every dense row's diagonal."""
    world = wp.tid()
    for i in range(world_constraint_count[world]):
        world_diag[world, i] += pgs_cfm


# =============================================================================
# Parallelized Non-Tiled Kernels for Heterogeneous Multi-Articulation
# =============================================================================
# These kernels parallelize across constraints (and constraint pairs) to achieve
# much better GPU utilization than the single-thread-per-articulation versions.


@wp.kernel
def hinv_jt_par_row(
    # Grouped Cholesky factor storage [n_arts, n_dofs, n_dofs]
    L_group: wp.array3d[float],
    # Size-grouped Jacobian [n_arts_of_size, max_constraints, n_dofs]
    J_group: wp.array3d[float],
    # Indirection arrays
    group_to_art: wp.array[int],
    art_to_world: wp.array[int],
    articulation_world_dof_offset: wp.array[int],
    world_constraint_count: wp.array[int],
    # Size parameters
    n_dofs: int,
    max_constraints: int,
    n_arts: int,
    write_world: int,
    # Output: Y = H^-1 * J^T [n_arts_of_size, max_constraints, n_dofs]
    Y_group: wp.array3d[float],
    J_world: wp.array3d[float],
    Y_world: wp.array3d[float],
):
    """
    Compute Y = H^-1 * J^T for one size group using forward/backward substitution.

    Uses L_group (3D array) grouped by DOF size.
    Efficient for small articulations where tile overhead dominates.

    Each thread handles one (articulation, constraint) pair.

    For each articulation in the group, solves:
        L * L^T * Y = J^T
    Using:
        1. Forward substitution: L * Z = J^T
        2. Backward substitution: L^T * Y = Z

    Thread dimension: n_arts_of_size * max_constraints
    """
    tid = wp.tid()

    # Decode thread index
    c = tid % max_constraints  # constraint index
    idx = tid // max_constraints  # group index (articulation within size group)

    # Bounds check for articulation
    if idx >= n_arts:
        return

    art = group_to_art[idx]
    world = art_to_world[art]
    n_constraints = world_constraint_count[world]

    # Early exit if this constraint is beyond the actual count
    if c >= n_constraints:
        return

    # ----------------------------------------------------------------
    # Forward substitution: L * z = j
    # L is lower triangular, so solve from top to bottom
    # ----------------------------------------------------------------
    for i in range(n_dofs):
        # z[i] = (j[i] - sum_{k<i} L[i,k] * z[k]) / L[i,i]
        val = J_group[idx, c, i]

        for k in range(i):
            # z[k] is stored in Y_group temporarily
            val -= L_group[idx, i, k] * Y_group[idx, c, k]

        L_ii = L_group[idx, i, i]
        if L_ii != 0.0:
            Y_group[idx, c, i] = val / L_ii
        else:
            Y_group[idx, c, i] = 0.0

    # ----------------------------------------------------------------
    # Backward substitution: L^T * y = z
    # L^T is upper triangular, so solve from bottom to top
    # z is currently stored in Y_group, we overwrite with y
    # ----------------------------------------------------------------
    for i_rev in range(n_dofs):
        i = n_dofs - 1 - i_rev

        # y[i] = (z[i] - sum_{k>i} L[k,i] * y[k]) / L[i,i]
        # Note: L^T[i,k] = L[k,i], so we read L[k,i] for k > i
        val = Y_group[idx, c, i]  # This is z[i] from forward pass

        for k in range(i + 1, n_dofs):
            val -= L_group[idx, k, i] * Y_group[idx, c, k]

        L_ii = L_group[idx, i, i]
        if L_ii != 0.0:
            Y_group[idx, c, i] = val / L_ii
        else:
            Y_group[idx, c, i] = 0.0

    if write_world != 0:
        dof_offset = articulation_world_dof_offset[art]
        for i in range(n_dofs):
            J_world[world, c, dof_offset + i] = J_group[idx, c, i]
            Y_world[world, c, dof_offset + i] = Y_group[idx, c, i]


# =============================================================================
# Tiled kernels for homogenous multi-articulation support
# =============================================================================


@wp.kernel(module=_MASS_DYNAMICS_KERNEL_MODULE)
def crba_fill_par_dof(
    articulation_start: wp.array[int],
    articulation_dof_start: wp.array[int],
    mass_update_mask: wp.array[int],
    joint_ancestor: wp.array[int],
    joint_child: wp.array[int],
    joint_qd_start: wp.array[int],
    joint_dof_dim: wp.array2d[int],
    joint_S_s: wp.array[wp.spatial_vector],
    body_I_c: wp.array[wp.spatial_matrix],
    # Size-group parameters
    group_to_art: wp.array[int],
    n_dofs: int,  # = TILE_DOF for tiled path
    # outputs
    H_group: wp.array3d[float],  # [n_arts_of_size, n_dofs, n_dofs]
):
    """
    CRBA fill kernel that writes directly to size-grouped H storage.

    Thread dimension: n_arts_of_size * n_dofs (one thread per articulation-column pair)

    This version is for homogenous multi-articulation where all articulations have
    the same DOF count equal to TILE_DOF.
    """
    tid = wp.tid()

    group_idx = tid // n_dofs
    col_idx = tid % n_dofs

    art_idx = group_to_art[group_idx]

    if mass_update_mask[art_idx] == 0:
        return

    # All articulations in this group have exactly n_dofs DOFs
    if col_idx >= n_dofs:
        return

    global_dof_start = articulation_dof_start[art_idx]
    target_dof_global = global_dof_start + col_idx

    joint_start = articulation_start[art_idx]
    joint_end = articulation_start[art_idx + 1]

    # Find the joint that owns this DOF
    pivot_joint = int(-1)
    for j in range(joint_start, joint_end):
        q_start = joint_qd_start[j]
        q_end = joint_qd_start[j + 1]
        if target_dof_global >= q_start and target_dof_global < q_end:
            pivot_joint = j
            break

    if pivot_joint == -1:
        return

    # Compute Force F = I_c[pivot] * S[column]
    S_col = joint_S_s[target_dof_global]
    # body_I_c is BODY-indexed; joint index only coincides with the child body
    # index in loop-free models.
    I_comp = body_I_c[joint_child[pivot_joint]]
    F = I_comp * S_col

    # Walk up the tree and project F onto ancestors
    # H[row, col] = S[row] * F
    curr = pivot_joint

    while curr != -1:
        if curr < joint_start:
            break

        q_start = joint_qd_start[curr]
        q_dim = joint_dof_dim[curr]
        count = q_dim[0] + q_dim[1]

        dof_offset_local = q_start - global_dof_start

        for k in range(count):
            row_idx = dof_offset_local + k

            S_row = joint_S_s[q_start + k]
            val = wp.dot(S_row, F)

            # Write to grouped 3D array
            H_group[group_idx, row_idx, col_idx] = val
            H_group[group_idx, col_idx, row_idx] = val

        curr = joint_ancestor[curr]


@wp.kernel
def trisolve_loop(
    L_group: wp.array3d[float],  # [n_arts_of_size, n_dofs, n_dofs]
    group_to_art: wp.array[int],
    articulation_dof_start: wp.array[int],
    n_dofs: int,
    joint_tau: wp.array[float],  # [total_dofs]
    joint_qdd: wp.array[float],  # [total_dofs]
):
    """
    Solve L * L^T * qdd = tau for grouped articulations using forward/backward substitution.

    Thread dimension: n_arts_of_size (one thread per articulation in this size group)
    """
    idx = wp.tid()
    art = group_to_art[idx]
    dof_start = articulation_dof_start[art]

    # Forward substitution: L * z = tau
    # z is stored temporarily in joint_qdd
    for i in range(n_dofs):
        val = joint_tau[dof_start + i]
        for k in range(i):
            L_ik = L_group[idx, i, k]
            val -= L_ik * joint_qdd[dof_start + k]

        L_ii = L_group[idx, i, i]
        if L_ii != 0.0:
            joint_qdd[dof_start + i] = val / L_ii
        else:
            joint_qdd[dof_start + i] = 0.0

    # Backward substitution: L^T * qdd = z
    for i_rev in range(n_dofs):
        i = n_dofs - 1 - i_rev

        val = joint_qdd[dof_start + i]
        for k in range(i + 1, n_dofs):
            L_ki = L_group[idx, k, i]
            val -= L_ki * joint_qdd[dof_start + k]

        L_ii = L_group[idx, i, i]
        if L_ii != 0.0:
            joint_qdd[dof_start + i] = val / L_ii
        else:
            joint_qdd[dof_start + i] = 0.0


@wp.kernel
def gather_tau_to_groups(
    joint_tau: wp.array[float],  # [total_dofs]
    group_to_art: wp.array[int],
    articulation_dof_start: wp.array[int],
    n_dofs: int,
    tau_group: wp.array3d[float],  # [n_arts, n_dofs, 1]
):
    """Gather joint_tau from 1D array into grouped 3D buffer for tiled solve.

    Thread dimension: n_arts_of_size (one thread per articulation in this size group)
    """
    idx = wp.tid()
    art = group_to_art[idx]
    dof_start = articulation_dof_start[art]
    for i in range(n_dofs):
        tau_group[idx, i, 0] = joint_tau[dof_start + i]


@wp.kernel
def scatter_qdd_from_groups(
    qdd_group: wp.array3d[float],  # [n_arts, n_dofs, 1]
    group_to_art: wp.array[int],
    articulation_dof_start: wp.array[int],
    n_dofs: int,
    joint_qdd: wp.array[float],  # [total_dofs]
):
    """Scatter qdd from grouped 3D buffer back to 1D array after tiled solve.

    Thread dimension: n_arts_of_size (one thread per articulation in this size group)
    """
    idx = wp.tid()
    art = group_to_art[idx]
    dof_start = articulation_dof_start[art]
    for i in range(n_dofs):
        joint_qdd[dof_start + i] = qdd_group[idx, i, 0]
