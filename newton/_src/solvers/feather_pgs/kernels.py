# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

from functools import cache

import warp as wp

from ...math.spatial import transform_twist
from ...sim import BodyFlags, JointType
from ...sim.articulation import (
    compute_2d_rotational_dofs,
    compute_3d_rotational_dofs,
    eval_single_articulation_fk,
)
from ...sim.contacts import GENERATION_SENTINEL
from .contact_filters import contact_friction_eligible, contact_normal_gap_limit
from .friction import contact_tangent_basis, friction_pair_candidate
from .friction_patches import FrictionPatches, warmstart_dt_scale

PGS_CONSTRAINT_TYPE_CONTACT = 0
# PGS joint drive row (``drive_mode="physx_pgs"``): a bilateral row per driven DOF whose
# impulse follows the PhysX force-drive update each iteration; the velocity-only
# iterations can freeze these rows.
PGS_CONSTRAINT_TYPE_JOINT_TARGET = 1
PGS_CONSTRAINT_TYPE_FRICTION = 2
PGS_CONSTRAINT_TYPE_JOINT_LIMIT = 3
# Joint velocity-limit row: a per-DOF velocity clamp with one unilateral row per
# bound and no position bias.
PGS_CONSTRAINT_TYPE_JOINT_VELOCITY_LIMIT = 4
# Mimic row: the bilateral equality ``q_follower - coef1 * q_leader - coef0 = 0`` between
# two DOFs of one articulation, with an unbounded impulse.
PGS_CONSTRAINT_TYPE_MIMIC = 5
# Connect row: one world axis of the point coincidence of a loop-closing BALL joint's
# parent and child anchors (three rows per closure), with an unbounded impulse.
PGS_CONSTRAINT_TYPE_CONNECT = 6
# Experimental contact torsion row: one bounded angular row per contact group about the
# group normal (see ``contact_torsion.py``).
PGS_CONSTRAINT_TYPE_TORSION = 7

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

# Owner of a world's dense and free-body rows in the matrix-free solve: the general
# world sweep, or one of the articulation-local solves (one articulation, one
# articulation with a free body, or that pair together with its free-body rows).
PGS_LOCAL_SOLVE_OWNER_GENERAL = 0
PGS_LOCAL_SOLVE_OWNER_SINGLE = 1
PGS_LOCAL_SOLVE_OWNER_PAIR = 2
PGS_LOCAL_SOLVE_OWNER_PAIR_RESIDUAL = 3


@wp.kernel
def local_solve_launch_gate():
    """Create a minimal graph dependency ahead of the bulk local solves."""
    pass


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
    drive_slot: wp.array[int],
    skip_driven: int,
    articulation_active: wp.array[int],
    joint_qd: wp.array[float],
):
    """PhysX-style pre-solve joint velocity scaling.

    PhysX computes a single ratio per articulation from maxJointVelocity and
    applies that ratio to all articulation DOFs before building link velocities.
    This is separate from the velocity-limit constraint rows solved later.

    ``skip_driven != 0`` (``fuse_joint_velocity_limits``) excludes DOFs with a drive
    row (``drive_slot[dof] >= 0``) from the ratio and the scaling: the fused clamp
    limits them inside the solve. With ``skip_driven == 0`` ``drive_slot`` is not read.
    """
    art = wp.tid()
    if articulation_active[art] == 0:
        return
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
            if skip_driven != 0:
                if drive_slot[dof] >= 0:
                    continue
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
            if skip_driven != 0:
                if drive_slot[dof] >= 0:
                    continue
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
    articulation_active: wp.array[int],
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
    if articulation_active[index] == 0:
        return
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


@wp.func_native("""
#if defined(__CUDA_ARCH__)
    __syncwarp();
#endif
""")
def _tree_warp_sync():
    """Order dependent tree levels within one complete CUDA warp."""


@cache
def _get_tree_fk_kernel(lanes: int, mode: str):
    """Build a cooperative traversal of independent tree branches.

    Each articulation is assigned ``lanes`` threads of a warp. Its branches are
    grouped into levels; the branches of one level run in parallel and levels are
    separated by a warp barrier. The per-joint mathematics is the serial kernels'.

    ``mode`` ``"id"`` evaluates poses and the inverse-dynamics bias with the
    :func:`eval_rigid_fk_id` signature after the six schedule arguments;
    ``"public"`` publishes ``body_q`` / ``body_qd`` like :func:`newton.eval_fk`.
    Launch complete 32-thread warps.
    """
    if lanes not in (1, 2, 4, 8, 16, 32):
        raise ValueError("Tree traversal lanes must be a power of two between 1 and 32")
    if mode not in ("id", "public"):
        raise ValueError(f"Unknown tree traversal mode: {mode}")
    module = wp.Module(f"{__name__}.tree_fk_{mode}_{lanes}")

    if mode == "public":

        @wp.kernel(module=module, enable_backward=False)
        def tree_fk_public(
            group_count: int,
            max_levels: int,
            art_indices: wp.array[int],
            level_offsets: wp.array2d[int],
            segment_offsets: wp.array[int],
            segment_joints: wp.array[int],
            joint_articulation: wp.array[int],
            joint_q: wp.array[float],
            joint_qd: wp.array[float],
            joint_q_start: wp.array[int],
            joint_qd_start: wp.array[int],
            joint_type: wp.array[int],
            joint_parent: wp.array[int],
            joint_child: wp.array[int],
            joint_X_p: wp.array[wp.transform],
            joint_X_c: wp.array[wp.transform],
            joint_axis: wp.array[wp.vec3],
            joint_dof_dim: wp.array2d[int],
            body_com: wp.array[wp.vec3],
            body_flags: wp.array[wp.int32],
            body_flag_filter: int,
            articulation_active: wp.array[int],
            body_q: wp.array[wp.transform],
            body_qd: wp.array[wp.spatial_vector],
        ):
            block, lane = wp.tid()
            group = block * (32 // lanes) + lane // lanes
            local = lane % lanes
            active = group < group_count
            if active:
                active = articulation_active[art_indices[group]] != 0
            for level in range(max_levels):
                if active:
                    segment = level_offsets[group, level] + local
                    end = level_offsets[group, level + 1]
                    while segment < end:
                        for entry in range(segment_offsets[segment], segment_offsets[segment + 1]):
                            joint = segment_joints[entry]
                            eval_single_articulation_fk(
                                joint,
                                joint + 1,
                                joint_articulation,
                                joint_q,
                                joint_qd,
                                joint_q_start,
                                joint_qd_start,
                                joint_type,
                                joint_parent,
                                joint_child,
                                joint_X_p,
                                joint_X_c,
                                joint_axis,
                                joint_dof_dim,
                                body_com,
                                body_flags,
                                body_flag_filter,
                                body_q,
                                body_qd,
                            )
                        segment += lanes
                # Inactive groups still participate in every full-warp barrier.
                if wp.static(lanes > 1):
                    _tree_warp_sync()

        return tree_fk_public

    @wp.kernel(module=module, enable_backward=False)
    def tree_fk_id(
        group_count: int,
        max_levels: int,
        art_indices: wp.array[int],
        level_offsets: wp.array2d[int],
        segment_offsets: wp.array[int],
        segment_joints: wp.array[int],
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
        articulation_active: wp.array[int],
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
        block, lane = wp.tid()
        group = block * (32 // lanes) + lane // lanes
        local = lane % lanes
        active = group < group_count
        articulation = int(0)
        if active:
            articulation = art_indices[group]
            if articulation_active[articulation] == 0:
                active = False

        # The first root joint defines the articulation's frame origin, as in the serial kernel.
        seed_joint = int(-1)
        if active:
            start = articulation_start[articulation]
            if start < articulation_joint_end[articulation]:
                seed_joint = start
            if local == 0:
                origin = wp.vec3()
                if seed_joint >= 0:
                    compute_link_transform(
                        seed_joint,
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
                    root_body = joint_child[seed_joint]
                    if root_body >= 0:
                        origin = wp.transform_point(body_q[root_body], body_com[root_body])
                articulation_origin[articulation] = origin
        if wp.static(lanes > 1):
            _tree_warp_sync()

        origin = wp.vec3()
        write_body_inertia = materialize_all_body_inertia
        if active:
            origin = articulation_origin[articulation]
            if is_free_rigid[articulation] != 0:
                write_body_inertia = 1

        # A joint needs only its parent's pose and motion and the articulation origin, so
        # one pass evaluates both, carrying the parent motion along each unary chain.
        for level in range(max_levels):
            if active:
                segment = level_offsets[group, level] + local
                end = level_offsets[group, level + 1]
                while segment < end:
                    begin = segment_offsets[segment]
                    finish = segment_offsets[segment + 1]
                    parent = joint_parent[segment_joints[begin]]
                    parent_v_s = wp.spatial_vector()
                    parent_a_s = wp.spatial_vector()
                    if parent >= 0:
                        parent_v_s = body_v_s[parent]
                        parent_a_s = body_a_s[parent]
                    for entry in range(begin, finish):
                        joint = segment_joints[entry]
                        parent = joint_parent[joint]
                        child = joint_child[joint]
                        if joint != seed_joint:
                            compute_link_transform(
                                joint,
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
                        gravity_s = gravity[body_world[child]]
                        parent_v_s, parent_a_s = compute_link_velocity(
                            joint,
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
                    segment += lanes
            # Inactive groups still participate in every full-warp barrier.
            if wp.static(lanes > 1):
                _tree_warp_sync()

    return tree_fk_id


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
    articulation_active: wp.array[int],
    body_ft_s: wp.array[wp.spatial_vector],
    tau: wp.array[float],
):
    # one thread per articulation
    if articulation_active[wp.tid()] == 0:
        return
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


@cache
def _get_tree_tau_kernel(lanes: int):
    """Build the cooperative backward pass of :func:`eval_rigid_tau` over independent branches.

    Branch wrenches are reduced without atomics; the immediate children of a branch are
    summed in descending joint order, the order of the serial pass.
    """
    if lanes not in (1, 2, 4, 8, 16, 32):
        raise ValueError("Tree traversal lanes must be a power of two between 1 and 32")
    module = wp.Module(f"{__name__}.tree_tau_{lanes}")

    @wp.kernel(module=module, enable_backward=False)
    def tree_tau(
        group_count: int,
        max_levels: int,
        art_indices: wp.array[int],
        level_offsets: wp.array2d[int],
        segment_offsets: wp.array[int],
        segment_joints: wp.array[int],
        child_offsets: wp.array[int],
        child_segments: wp.array[int],
        joint_type: wp.array[int],
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
        articulation_active: wp.array[int],
        body_ft_s: wp.array[wp.spatial_vector],
        tau: wp.array[float],
        segment_net: wp.array[wp.spatial_vector],
    ):
        block, lane = wp.tid()
        group = block * (32 // lanes) + lane // lanes
        local = lane % lanes
        active = group < group_count
        if active:
            active = articulation_active[art_indices[group]] != 0
        for reverse_level in range(max_levels):
            level = max_levels - reverse_level - 1
            if active:
                segment = level_offsets[group, level] + local
                end = level_offsets[group, level + 1]
                while segment < end:
                    begin = segment_offsets[segment]
                    finish = segment_offsets[segment + 1]
                    tail_body = joint_child[segment_joints[finish - 1]]
                    f_t_s = body_ft_s[tail_body]
                    for entry in range(child_offsets[segment], child_offsets[segment + 1]):
                        f_t_s = f_t_s + segment_net[child_segments[entry]]
                    f_s = wp.spatial_vector()
                    for offset in range(finish - begin):
                        joint = segment_joints[finish - offset - 1]
                        child = joint_child[joint]
                        if offset > 0:
                            f_t_s = body_ft_s[child] + f_s
                        body_ft_s[child] = f_t_s
                        articulation = joint_articulation[joint]
                        origin = wp.vec3()
                        if articulation >= 0:
                            origin = articulation_origin[articulation]
                        f_s = _compute_body_net_wrench(
                            child, f_t_s, origin, body_fb_s, body_f_ext, body_flags, body_q, body_com
                        )
                        jcalc_tau(
                            joint_type[joint],
                            joint_S_s,
                            joint_f,
                            joint_q,
                            joint_qd,
                            joint_spring_stiffness,
                            joint_spring_ref,
                            joint_damping,
                            joint_q_start[joint],
                            joint_qd_start[joint],
                            joint_dof_dim[joint, 0],
                            joint_dof_dim[joint, 1],
                            f_s,
                            0,
                            tau,
                        )
                    segment_net[segment] = f_s
                    segment += lanes
            # Inactive groups still participate in every full-warp barrier.
            if wp.static(lanes > 1):
                _tree_warp_sync()

    return tree_tau


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
    joint_active: wp.array[int],
    v_hat: wp.array[float],
):
    """Lift the free root's velocity predictor onto the integrator's convention.

    ``jcalc_integrate`` realizes ``qd + (qdd + omega x v) * dt`` for the root's
    linear coordinate. Constraint rows are built against ``v_hat``, so without
    the same term here every contact, friction, and velocity-limit row sees a
    COM velocity the integrator never produces, off by ``dt * (omega x v)``.
    """
    root_index = wp.tid()
    if joint_active[free_root_joint_indices[root_index]] == 0:
        return
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
    joint_active: wp.array[int],
    v_hat: wp.array[float],
):
    """Fuse free-root transport with the isolated rigid-body gyroscopic update."""
    root_index = wp.tid()
    if joint_active[free_root_joint_indices[root_index]] == 0:
        return
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
    joint_active: wp.array[int],
    joint_qdd: wp.array[float],
):
    """Make ``jcalc_integrate`` reproduce the solved velocity exactly.

    The solver commits to ``v_out``; ``qdd = (v_out - qd) / dt`` alone would let
    the integrator's transport term push the realized root velocity to
    ``v_out + dt * (omega x v)``. Subtracting the term here closes the loop, and
    in the contact-free case recovers the dynamics' own ``qdd`` bit for bit.
    """
    root_index = wp.tid()
    if joint_active[free_root_joint_indices[root_index]] == 0:
        return
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
def apply_free_root_angular_damping(
    free_root_joint_indices: wp.array[int],
    joint_qd_start: wp.array[int],
    kinematic_joint_mask: wp.array[int],
    joint_child: wp.array[int],
    body_angular_damping: wp.array[float],
    dt: float,
    joint_qd: wp.array[float],
):
    """Scale each active free root's angular velocity by the decay ``jcalc_integrate`` applies."""
    root_index = wp.tid()
    d = _active_free_root_dof_start(free_root_joint_indices, joint_qd_start, kinematic_joint_mask, root_index)
    if d < 0:
        return
    scale = 1.0 - body_angular_damping[joint_child[free_root_joint_indices[root_index]]] * dt
    for i in range(3, 6):
        joint_qd[d + i] = joint_qd[d + i] * scale


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
    joint_active: wp.array[int],
    joint_q_new: wp.array[float],
    joint_qd_new: wp.array[float],
):
    # one thread per joint
    index = wp.tid()
    if joint_active[index] == 0:
        return

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
    dof_active: wp.array[int],
    # outputs
    joint_qdd: wp.array[float],
    v_hat: wp.array[float],
):
    tid = wp.tid()
    if dof_active[tid] == 0:
        return
    if kinematic_dof_mask[tid] != 0:
        joint_qdd[tid] = 0.0
    v_hat[tid] = joint_qd[tid] + joint_qdd[tid] * dt


@wp.kernel
def update_qdd_from_velocity(
    joint_qd: wp.array[float],
    kinematic_dof_mask: wp.array[int],
    inv_dt: float,
    dof_active: wp.array[int],
    # output
    v_new: wp.array[float],
    joint_qdd: wp.array[float],
):
    tid = wp.tid()
    if dof_active[tid] == 0:
        return
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
    propagation_impulses: wp.array2d[wp.float32],
    world_constraint_count: wp.array[wp.int32],
    mf_constraint_count: wp.array[wp.int32],
    propagation_constraint_count: wp.array[wp.int32],
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
        elif path == 2:
            count = propagation_constraint_count[world]
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
            elif path == 1:
                lam_n = mf_impulses[world, slot]
                if has_friction:
                    lam_t0 = mf_impulses[world, slot + 1]
                    lam_t1 = mf_impulses[world, slot + 2]
            else:
                lam_n = propagation_impulses[world, slot]
                if has_friction:
                    lam_t0 = propagation_impulses[world, slot + 1]
                    lam_t1 = propagation_impulses[world, slot + 2]
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
def eval_augmented_drives(
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
    articulation_active: wp.array[int],
    row_counts: wp.array[int],
    row_dof_index: wp.array[int],
    row_K: wp.array[float],
    tau: wp.array[float],
):
    """Add the augmented drives to an already accumulated ``tau``, one thread per articulation."""
    if articulation_active[wp.tid()] == 0:
        return
    prepare_articulation_augmented_drives(
        wp.tid(),
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
    articulation_active: wp.array[int],
    body_ft_s: wp.array[wp.spatial_vector],
    row_counts: wp.array[int],
    row_dof_index: wp.array[int],
    row_K: wp.array[float],
    tau: wp.array[float],
):
    """Accumulate articulation forces and augmented drives in one launch."""
    articulation = wp.tid()
    if articulation_active[articulation] == 0:
        return
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
    articulation_active: wp.array[int],
    mass_update_mask: wp.array[int],
):
    tid = wp.tid()
    flag = 1 if global_flag != 0 else 0
    if mass_update_requested[tid] != 0:
        flag = 1
    if articulation_active[tid] == 0:
        flag = 0
    mass_update_mask[tid] = flag


# =============================================================================
# Bilateral Constraint Kernels (mimic and connect rows)
# =============================================================================
# Mimic rows enforce ``q_follower = coef0 + coef1 * q_leader`` between two DOFs of
# one articulation, one row per follower coordinate. Connect rows close a kinematic
# loop: three rows per loop-closing BALL joint pin its parent and child anchors
# together. Both are bilateral equality rows with an unbounded impulse and the
# Baumgarte bias ``pgs_beta * phi / dt``. Coefficients, enable flags and anchors are
# read from device arrays every step, so runtime changes need no re-initialization
# and stay compatible with CUDA graph capture.


@wp.func
def dense_index(stride: int, i: int, j: int):
    return i * stride + j


@wp.func
def _reserve_bilateral_rows(
    world: int,
    row_count: int,
    max_constraints: int,
    world_slot_counter: wp.array[int],
    first_rejected_slot: wp.array[int],
    dropped_rows: wp.array[int],
) -> int:
    """Reserve ``row_count`` consecutive rows of a world, or reject the whole group.

    A rejected group records its first slot so the row count is truncated before it,
    and counts its rows as dropped; the raised slot counter latches the overflow status.
    """
    slot = wp.atomic_add(world_slot_counter, world, row_count)
    if slot + row_count <= max_constraints:
        return slot
    wp.atomic_min(first_rejected_slot, world, slot)
    wp.atomic_add(dropped_rows, world, row_count)
    return -1


@wp.kernel
def allocate_mimic_slots(
    mimic_legacy: wp.array[int],
    mimic_enabled: wp.array[wp.bool],
    mimic_world: wp.array[int],
    max_constraints: int,
    # outputs
    mimic_slot: wp.array[int],
    world_slot_counter: wp.array[int],
    first_rejected_slot: wp.array[int],
    dropped_rows: wp.array[int],
):
    """Allocate one dense row per enabled mimic row.

    Launched with one thread per mimic row. A row from a disabled
    :attr:`~newton.Model.constraint_mimic_enabled` entry gets ``mimic_slot = -1``;
    joint-owned rows are always enabled.
    """
    k = wp.tid()
    mimic_slot[k] = -1
    legacy = mimic_legacy[k]
    if legacy >= 0:
        if not mimic_enabled[legacy]:
            return
    mimic_slot[k] = _reserve_bilateral_rows(
        mimic_world[k], 1, max_constraints, world_slot_counter, first_rejected_slot, dropped_rows
    )


@wp.kernel
def populate_mimic_J_for_size(
    articulation_dof_start: wp.array[int],
    art_to_world: wp.array[int],
    group_to_art: wp.array[int],
    mimic_slot: wp.array[int],
    mimic_art_start: wp.array[int],
    mimic_art_list: wp.array[int],
    mimic_dof0: wp.array[int],
    mimic_dof1: wp.array[int],
    mimic_q0: wp.array[int],
    mimic_q1: wp.array[int],
    mimic_legacy: wp.array[int],
    mimic_owner: wp.array[int],
    mimic_coef0: wp.array[float],
    mimic_coef1: wp.array[float],
    joint_mimic_coeffs: wp.array[wp.vec2],
    joint_q: wp.array[float],
    # outputs
    J_group: wp.array3d[float],
    world_row_type: wp.array2d[int],
    world_row_parent: wp.array2d[int],
    world_row_mu: wp.array2d[float],
    world_phi: wp.array2d[float],
    world_target_velocity: wp.array2d[float],
):
    """Fill the Jacobian and metadata of the mimic rows of one size group.

    One thread per articulation of the group visits the articulation's range of the
    mimic table. The row is ``J = e_follower - coef1 * e_leader`` with the signed
    violation ``phi = q_follower - coef1 * q_leader - coef0``.
    """
    group_idx = wp.tid()
    art = group_to_art[group_idx]
    world = art_to_world[art]
    dof_start = articulation_dof_start[art]

    for m in range(mimic_art_start[art], mimic_art_start[art + 1]):
        k = mimic_art_list[m]
        slot = mimic_slot[k]
        if slot < 0:
            continue

        legacy = mimic_legacy[k]
        c0 = float(0.0)
        c1 = float(0.0)
        if legacy >= 0:
            c0 = mimic_coef0[legacy]
            c1 = mimic_coef1[legacy]
        else:
            coeffs = joint_mimic_coeffs[mimic_owner[k]]
            c0 = coeffs[0]
            c1 = coeffs[1]

        # The construction-time plan guarantees two distinct DOFs of this articulation.
        J_group[group_idx, slot, mimic_dof0[k] - dof_start] = 1.0
        J_group[group_idx, slot, mimic_dof1[k] - dof_start] = -c1

        world_row_type[world, slot] = PGS_CONSTRAINT_TYPE_MIMIC
        world_row_parent[world, slot] = -1
        world_row_mu[world, slot] = 0.0
        world_phi[world, slot] = joint_q[mimic_q0[k]] - c1 * joint_q[mimic_q1[k]] - c0
        world_target_velocity[world, slot] = 0.0


@wp.kernel
def allocate_connect_slots(
    connect_enabled: wp.array[int],
    connect_world: wp.array[int],
    max_constraints: int,
    # outputs
    connect_slot: wp.array[int],
    world_slot_counter: wp.array[int],
    first_rejected_slot: wp.array[int],
    dropped_rows: wp.array[int],
):
    """Allocate three consecutive dense rows per enabled loop closure.

    A closure that does not fit is dropped whole (``connect_slot = -1``) rather than
    enforced along some axes only.
    """
    k = wp.tid()
    connect_slot[k] = -1
    if connect_enabled[k] == 0:
        return
    connect_slot[k] = _reserve_bilateral_rows(
        connect_world[k], 3, max_constraints, world_slot_counter, first_rejected_slot, dropped_rows
    )


@wp.kernel
def populate_connect_J_for_size(
    articulation_dof_start: wp.array[int],
    art_to_world: wp.array[int],
    group_to_art: wp.array[int],
    n_dofs: int,
    connect_slot: wp.array[int],
    connect_art: wp.array[int],
    connect_body_p: wp.array[int],
    connect_body_c: wp.array[int],
    connect_anchor_p: wp.array[wp.vec3],
    connect_anchor_c: wp.array[wp.vec3],
    connect_parent_prescribed: wp.array[int],
    body_q: wp.array[wp.transform],
    body_qd: wp.array[wp.spatial_vector],
    body_com: wp.array[wp.vec3],
    body_to_joint: wp.array[int],
    joint_ancestor: wp.array[int],
    joint_qd_start: wp.array[int],
    joint_S_s: wp.array[wp.spatial_vector],
    articulation_origin: wp.array[wp.vec3],
    # outputs
    J_group: wp.array3d[float],
    world_row_type: wp.array2d[int],
    world_row_parent: wp.array2d[int],
    world_row_mu: wp.array2d[float],
    world_phi: wp.array2d[float],
    world_target_velocity: wp.array2d[float],
):
    """Fill the Jacobian and metadata of the connect rows of one size group.

    One thread per articulation of the group scans the closure table and writes three
    rows per owned closure: for world axis ``e``, ``phi = e . (p_A - p_B)`` and
    ``J = e . (J_point(parent, p_A) - J_point(child, p_B))``, built with the ancestor
    walk of the contact rows. Shared ancestors cancel.

    A prescribed parent (a kinematic body outside the child's articulation, or the world
    when ``connect_body_p[k] < 0``) contributes no DOFs; its anchor velocity enters the
    row target instead (``J v = -e . v_A``), so the child anchor follows the moving
    parent anchor.

    Each row is normalized. On a planar linkage one axis is nearly redundant with the
    tree (``|J|`` around ``1e-4``), and an unnormalized row would relax a real velocity
    error against a near-zero diagonal. Degenerate rows are left inert (zero ``J``,
    ``phi`` and target).
    """
    group_idx = wp.tid()
    art = group_to_art[group_idx]
    world = art_to_world[art]
    dof_start = articulation_dof_start[art]

    n_connect = connect_art.shape[0]
    for k in range(n_connect):
        if connect_art[k] != art:
            continue
        base_slot = connect_slot[k]
        if base_slot < 0:
            continue

        body_p = connect_body_p[k]
        body_c = connect_body_c[k]
        prescribed = connect_parent_prescribed[k] != 0
        p_a = connect_anchor_p[k]
        v_a = wp.vec3(0.0, 0.0, 0.0)
        if body_p >= 0:
            p_a = wp.transform_point(body_q[body_p], connect_anchor_p[k])
            if prescribed:
                # The anchor velocity of a prescribed parent comes from its body twist.
                twist = body_qd[body_p]
                x_com = wp.transform_point(body_q[body_p], body_com[body_p])
                v_a = wp.spatial_top(twist) + wp.cross(wp.spatial_bottom(twist), p_a - x_com)
        p_b = wp.transform_point(body_q[body_c], connect_anchor_c[k])
        origin = articulation_origin[art]
        rel_a = p_a - origin
        rel_b = p_b - origin
        delta = p_a - p_b

        for axis in range(3):
            slot = base_slot + axis
            e = wp.vec3(0.0, 0.0, 0.0)
            e[axis] = 1.0
            target = float(0.0)

            # Both walks accumulate into the row, which the caller zeroed this step.
            if prescribed:
                target = -wp.dot(e, v_a)
            else:
                curr = body_to_joint[body_p]
                while curr != -1:
                    for d in range(joint_qd_start[curr], joint_qd_start[curr + 1]):
                        S = joint_S_s[d]
                        lin = wp.vec3(S[0], S[1], S[2])
                        ang = wp.vec3(S[3], S[4], S[5])
                        J_group[group_idx, slot, d - dof_start] += wp.dot(e, lin + wp.cross(ang, rel_a))
                    curr = joint_ancestor[curr]
            curr = body_to_joint[body_c]
            while curr != -1:
                for d in range(joint_qd_start[curr], joint_qd_start[curr + 1]):
                    S = joint_S_s[d]
                    lin = wp.vec3(S[0], S[1], S[2])
                    ang = wp.vec3(S[3], S[4], S[5])
                    J_group[group_idx, slot, d - dof_start] -= wp.dot(e, lin + wp.cross(ang, rel_b))
                curr = joint_ancestor[curr]

            norm_sq = float(0.0)
            for d in range(n_dofs):
                norm_sq += J_group[group_idx, slot, d] * J_group[group_idx, slot, d]
            phi_axis = delta[axis]
            if norm_sq > 1.0e-8:
                inv_norm = 1.0 / wp.sqrt(norm_sq)
                for d in range(n_dofs):
                    J_group[group_idx, slot, d] *= inv_norm
                phi_axis *= inv_norm
                target *= inv_norm
            else:
                for d in range(n_dofs):
                    J_group[group_idx, slot, d] = 0.0
                phi_axis = 0.0
                target = 0.0

            world_row_type[world, slot] = PGS_CONSTRAINT_TYPE_CONNECT
            world_row_parent[world, slot] = -1
            world_row_mu[world, slot] = 0.0
            world_phi[world, slot] = phi_axis
            world_target_velocity[world, slot] = target


# =============================================================================
# Bilateral Pre-elimination Kernels
# =============================================================================
# Fold the bilateral rows B of an articulation into the response of its other rows
# (a Schur complement): with Y_B = H^-1 J_B^T and S = J_B Y_B (+ regularization),
# every other row's response becomes Y'_i = Y_i - Y_B S^-1 (J_B Y_i), so J_B Y'_i is
# only a regularization residual and sweep impulses nearly preserve the closures. The
# corrected row diagonals follow from the unchanged J Y diagonal pass. The predictor
# velocity is projected once (J_B v + b_B ~ 0, up to the same residual), which replaces
# the Baumgarte work of the eliminated rows.

PREELIM_MAX_ROWS = 8
"""Per-articulation capacity of the pre-eliminated bilateral block, for example one
mimic row plus two three-row loop closures of a parallel gripper."""

_preelim_vec = wp.types.vector(length=PREELIM_MAX_ROWS, dtype=wp.float32)


@wp.func
def dense_cholesky(
    n: int,
    A: wp.array[float],
    R: wp.array[float],
    A_start: int,
    R_start: int,
    # outputs
    L: wp.array[float],
):
    """Factor ``A + diag(R) = L L^T`` for a packed row-major ``n x n`` block."""
    for j in range(n):
        s = A[A_start + dense_index(n, j, j)] + R[R_start + j]

        for k in range(j):
            r = L[A_start + dense_index(n, j, k)]
            s -= r * r

        s = wp.sqrt(s)
        invS = 1.0 / s

        L[A_start + dense_index(n, j, j)] = s

        for i in range(j + 1, n):
            s = A[A_start + dense_index(n, i, j)]

            for k in range(j):
                s -= L[A_start + dense_index(n, i, k)] * L[A_start + dense_index(n, j, k)]

            L[A_start + dense_index(n, i, j)] = s * invS


@wp.func
def preelim_solve(
    n: int,
    L: wp.array[float],
    L_start: int,
    b: _preelim_vec,
) -> _preelim_vec:
    """Solve ``(L L^T) x = b`` for a packed per-articulation Cholesky factor."""
    x = _preelim_vec()
    for i in range(n):
        s = b[i]
        for j in range(i):
            s -= L[L_start + dense_index(n, i, j)] * x[j]
        x[i] = s / L[L_start + dense_index(n, i, i)]
    for ii in range(n):
        i = n - 1 - ii
        s = x[i]
        for j in range(i + 1, n):
            s -= L[L_start + dense_index(n, j, i)] * x[j]
        x[i] = s / L[L_start + dense_index(n, i, i)]
    return x


@wp.kernel
def preelim_setup_for_size(
    group_to_art: wp.array[int],
    art_to_preelim: wp.array[int],
    mimic_slot: wp.array[int],
    mimic_art_start: wp.array[int],
    mimic_art_list: wp.array[int],
    n_mimic: int,
    connect_slot: wp.array[int],
    connect_art: wp.array[int],
    n_connect: int,
    J_group: wp.array3d[float],
    Y_group: wp.array3d[float],
    n_dofs: int,
    reg_rel: float,
    reg_floor: float,
    # outputs
    preelim_slots: wp.array[int],
    preelim_nB: wp.array[int],
    S_scratch: wp.array[float],
    reg: wp.array[float],
    LS: wp.array[float],
):
    """Gather the bilateral block of each articulation, form ``S = J_B Y_B`` and factor it.

    One thread per articulation of the group, after ``Y = H^-1 J^T``. Ownership comes
    from the per-row articulation tables, not from the per-world row type, which would
    admit zero rows of other articulations and make ``S`` singular. Zero rows (inert
    degenerate connect axes) are dropped for the same reason.
    """
    group_idx = wp.tid()
    art = group_to_art[group_idx]
    pe = art_to_preelim[art]
    if pe < 0:
        return
    base = pe * PREELIM_MAX_ROWS

    n = int(0)
    if n_mimic > 0:
        for m in range(mimic_art_start[art], mimic_art_start[art + 1]):
            k = mimic_art_list[m]
            if mimic_slot[k] >= 0 and n < PREELIM_MAX_ROWS:
                preelim_slots[base + n] = mimic_slot[k]
                n += 1
    for k in range(n_connect):
        if connect_art[k] == art and connect_slot[k] >= 0:
            for a in range(3):
                if n < PREELIM_MAX_ROWS:
                    preelim_slots[base + n] = connect_slot[k] + a
                    n += 1

    m = int(0)
    for p in range(n):
        s = preelim_slots[base + p]
        nrm = float(0.0)
        for d in range(n_dofs):
            nrm += J_group[group_idx, s, d] * J_group[group_idx, s, d]
        if nrm > 1.0e-10:
            preelim_slots[base + m] = s
            m += 1
    for p in range(m, PREELIM_MAX_ROWS):
        preelim_slots[base + p] = -1
    preelim_nB[pe] = m
    if m == 0:
        return

    s_base = pe * PREELIM_MAX_ROWS * PREELIM_MAX_ROWS
    for p in range(m):
        sp = preelim_slots[base + p]
        for q in range(m):
            sq = preelim_slots[base + q]
            acc = float(0.0)
            for d in range(n_dofs):
                acc += J_group[group_idx, sp, d] * Y_group[group_idx, sq, d]
            S_scratch[s_base + dense_index(m, p, q)] = acc

    # Relative diagonal regularization. A planar four-bar closure has a nearly dependent
    # axis whose last pivot is float32 cancellation noise; an absolute epsilon below the
    # Delassus scale of light links can turn it negative. Scaling by each row's own
    # diagonal keeps the factor positive definite at any mass scale and leaves the
    # redundant axis slightly soft, a direction the tree already enforces.
    for p in range(m):
        reg[base + p] = reg_rel * S_scratch[s_base + dense_index(m, p, p)] + reg_floor

    dense_cholesky(m, S_scratch, reg, s_base, base, LS)


@wp.kernel
def preelim_correct_Y_for_size(
    group_to_art: wp.array[int],
    art_to_world: wp.array[int],
    art_to_preelim: wp.array[int],
    constraint_count: wp.array[int],
    preelim_slots: wp.array[int],
    preelim_nB: wp.array[int],
    LS: wp.array[float],
    J_group: wp.array3d[float],
    n_dofs: int,
    max_constraints: int,
    n_arts: int,
    # outputs
    Y_group: wp.array3d[float],
):
    """Correct the response of every row outside the block: ``Y_i -= Y_B (S + R)^-1 (J_B Y_i)``.

    One thread per (articulation, row). Rows of the block keep their response for the
    projection and stay in the sweep, where they only see the small residual that the
    regularization ``R`` leaves, see :func:`preelim_project_velocity_for_size`.
    """
    idx = wp.tid()
    group_idx = idx // max_constraints
    i = idx % max_constraints
    if group_idx >= n_arts:
        return
    art = group_to_art[group_idx]
    pe = art_to_preelim[art]
    if pe < 0:
        return
    world = art_to_world[art]
    if i >= constraint_count[world]:
        return
    base = pe * PREELIM_MAX_ROWS
    m = preelim_nB[pe]
    if m == 0:
        return
    for p in range(m):
        if preelim_slots[base + p] == i:
            return

    w = _preelim_vec()
    nonzero = int(0)
    for p in range(m):
        sp = preelim_slots[base + p]
        acc = float(0.0)
        for d in range(n_dofs):
            acc += J_group[group_idx, sp, d] * Y_group[group_idx, i, d]
        w[p] = acc
        if acc != 0.0:
            nonzero = 1
    if nonzero == 0:
        return

    z = preelim_solve(m, LS, pe * PREELIM_MAX_ROWS * PREELIM_MAX_ROWS, w)

    for d in range(n_dofs):
        acc = float(0.0)
        for p in range(m):
            acc += Y_group[group_idx, preelim_slots[base + p], d] * z[p]
        Y_group[group_idx, i, d] -= acc


@wp.kernel
def preelim_project_velocity_for_size(
    group_to_art: wp.array[int],
    art_to_world: wp.array[int],
    art_to_preelim: wp.array[int],
    articulation_dof_start: wp.array[int],
    preelim_slots: wp.array[int],
    preelim_nB: wp.array[int],
    LS: wp.array[float],
    J_group: wp.array3d[float],
    Y_group: wp.array3d[float],
    world_rhs: wp.array2d[float],
    n_dofs: int,
    # outputs
    v_out: wp.array[float],
):
    """Project the predictor velocity once: ``v -= Y_B (S + R)^-1 (J_B v + b_B)``.

    Runs after the solve velocity is seeded with the predictor. ``R`` is the diagonal
    regularization of :func:`preelim_setup_for_size`, so the projection is not exact:
    afterwards ``J_B v + b_B = R (S + R)^-1 (J_B v_0 + b_B)`` for the predictor ``v_0``,
    and a corrected response ``Y_i`` still changes ``J_B v`` by ``R (S + R)^-1 J_B Y_i``
    per unit impulse. The bilateral rows stay in the sweep and reduce this residual. Each
    thread owns one articulation's DOF range.
    """
    group_idx = wp.tid()
    art = group_to_art[group_idx]
    pe = art_to_preelim[art]
    if pe < 0:
        return
    m = preelim_nB[pe]
    if m == 0:
        return
    base = pe * PREELIM_MAX_ROWS
    world = art_to_world[art]
    dof_start = articulation_dof_start[art]

    r = _preelim_vec()
    for p in range(m):
        sp = preelim_slots[base + p]
        acc = world_rhs[world, sp]
        for d in range(n_dofs):
            acc += J_group[group_idx, sp, d] * v_out[dof_start + d]
        r[p] = acc

    z = preelim_solve(m, LS, pe * PREELIM_MAX_ROWS * PREELIM_MAX_ROWS, r)

    for d in range(n_dofs):
        acc = float(0.0)
        for p in range(m):
            acc += Y_group[group_idx, preelim_slots[base + p], d] * z[p]
        if acc != 0.0:
            v_out[dof_start + d] -= acc


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
    drive_slot: wp.array[int],
    skip_driven: int,
    art_to_world: wp.array[int],
    max_constraints: int,
    articulation_rows_active: wp.array[int],
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

    ``skip_driven != 0`` (``fuse_joint_velocity_limits``) skips DOFs with a drive
    row (``drive_slot[dof] >= 0``): the solve clamps their velocity at the end of
    every iteration instead. DOFs without a drive row keep their rows. With
    ``skip_driven == 0`` ``drive_slot`` is not read.

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
    # Sleeping articulations reserve no rows.
    if articulation_rows_active[art] == 0:
        return

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

            # Fused clamp: driven DOFs are limited inside the solve.
            if skip_driven != 0:
                if drive_slot[dof] >= 0:
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


@wp.kernel
def allocate_physx_drive_slots(
    articulation_start: wp.array[int],
    articulation_dof_start: wp.array[int],
    articulation_H_rows: wp.array[int],
    joint_type: wp.array[int],
    joint_qd_start: wp.array[int],
    joint_dof_dim: wp.array2d[int],
    joint_target_ke: wp.array[float],
    joint_target_kd: wp.array[float],
    art_to_world: wp.array[int],
    max_constraints: int,
    # outputs
    drive_slot: wp.array[int],
    world_slot_counter: wp.array[int],
):
    """Reserve one dense PGS drive row for every driven PRISMATIC, REVOLUTE or D6 DOF.

    A DOF is driven when its target stiffness or damping is positive. Reservations
    past ``max_constraints`` leave ``drive_slot[dof] = -1``; the raw counter still
    counts them, so the world's capacity status records the loss.
    """
    art = wp.tid()
    world = art_to_world[art]

    dof_base = articulation_dof_start[art]
    for d in range(articulation_H_rows[art]):
        drive_slot[dof_base + d] = -1

    for j in range(articulation_start[art], articulation_start[art + 1]):
        jtype = joint_type[j]
        if jtype != JointType.PRISMATIC and jtype != JointType.REVOLUTE and jtype != JointType.D6:
            continue

        axis_count = joint_dof_dim[j, 0] + joint_dof_dim[j, 1]
        qd_start = joint_qd_start[j]
        for axis in range(axis_count):
            dof = qd_start + axis
            if joint_target_ke[dof] <= 0.0 and joint_target_kd[dof] <= 0.0:
                continue
            slot = wp.atomic_add(world_slot_counter, world, 1)
            if slot < max_constraints:
                drive_slot[dof] = slot


@wp.kernel
def populate_physx_drive_J_for_size(
    articulation_start: wp.array[int],
    articulation_dof_start: wp.array[int],
    joint_type: wp.array[int],
    joint_q_start: wp.array[int],
    joint_qd_start: wp.array[int],
    joint_dof_dim: wp.array2d[int],
    joint_target_ke: wp.array[float],
    joint_target_kd: wp.array[float],
    joint_effort_limit: wp.array[float],
    joint_q: wp.array[float],
    joint_target_pos: wp.array[float],
    joint_target_q_start: wp.array[int],
    joint_target_vel: wp.array[float],
    joint_velocity_limit: wp.array[float],
    fuse_vel_limits: int,
    art_to_world: wp.array[int],
    drive_slot: wp.array[int],
    group_to_art: wp.array[int],
    # outputs
    J_group: wp.array3d[float],
    world_row_type: wp.array2d[int],
    world_row_parent: wp.array2d[int],
    world_row_mu: wp.array2d[float],
    world_phi: wp.array2d[float],
    world_target_velocity: wp.array2d[float],
    world_drive_stiffness: wp.array2d[float],
    world_drive_damping: wp.array2d[float],
    world_drive_geom_error: wp.array2d[float],
    world_drive_max_force: wp.array2d[float],
    world_drive_vel_limit: wp.array2d[float],
):
    """Fill the selector Jacobian ``J = e_i`` and the drive parameters of each drive row.

    Launched once per size group with ``dim = n_arts_of_size``. With
    ``fuse_vel_limits != 0`` each row also records its DOF's velocity limit for the
    fused clamp; non-positive or non-finite limits are stored as ``inf`` (no clamp).
    With ``fuse_vel_limits == 0`` neither ``joint_velocity_limit`` nor
    ``world_drive_vel_limit`` is accessed.
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
        q_start = joint_q_start[j]
        for axis in range(axis_count):
            dof = qd_start + axis
            slot = drive_slot[dof]
            if slot < 0:
                continue

            J_group[group_idx, slot, dof - dof_start] = 1.0
            world_row_type[world, slot] = PGS_CONSTRAINT_TYPE_JOINT_TARGET
            world_row_parent[world, slot] = -1
            world_row_mu[world, slot] = 0.0
            world_phi[world, slot] = 0.0
            world_target_velocity[world, slot] = joint_target_vel[dof]
            world_drive_stiffness[world, slot] = joint_target_ke[dof]
            world_drive_damping[world, slot] = joint_target_kd[dof]
            world_drive_geom_error[world, slot] = (
                joint_target_pos[joint_target_q_start[j] + axis] - joint_q[q_start + axis]
            )
            world_drive_max_force[world, slot] = joint_effort_limit[dof]
            if fuse_vel_limits != 0:
                qdot_max = joint_velocity_limit[dof]
                if qdot_max <= 0.0 or not wp.isfinite(qdot_max):
                    qdot_max = float(wp.inf)
                world_drive_vel_limit[world, slot] = qdot_max


@wp.kernel
def compute_physx_pgs_drive_desc(
    world_constraint_count: wp.array[int],
    world_row_type: wp.array2d[int],
    world_diag: wp.array2d[float],
    world_target_velocity: wp.array2d[float],
    world_drive_stiffness: wp.array2d[float],
    world_drive_damping: wp.array2d[float],
    world_drive_geom_error: wp.array2d[float],
    world_drive_max_force: wp.array2d[float],
    dt: float,
    # outputs
    world_drive_target_vel_bias: wp.array2d[float],
    world_drive_vel_multiplier: wp.array2d[float],
    world_drive_impulse_multiplier: wp.array2d[float],
    world_drive_max_impulse: wp.array2d[float],
):
    """Precompute the force-drive impulse update of every drive row.

    With stiffness ``ke``, damping ``kd``, unit response ``r = J H^-1 J^T`` (the row
    diagonal), ``a = dt * (dt * ke + kd)`` and ``x = 1 / (1 + a * r)``, the solve updates
    a drive row's impulse each iteration as

    ``lambda = (1 - x) * lambda - x * a * (J v) + x * dt * kd * v_target + x * dt * ke * (q_target - q)``

    clamped to ``+/- joint_effort_limit * dt`` when the effort limit is positive and
    finite. This is the PGS force-drive update of PhysX articulations (zero elapsed
    time, no accumulated position delta).
    """
    world = wp.tid()
    for i in range(world_constraint_count[world]):
        if world_row_type[world, i] != PGS_CONSTRAINT_TYPE_JOINT_TARGET:
            world_drive_target_vel_bias[world, i] = 0.0
            world_drive_vel_multiplier[world, i] = 0.0
            world_drive_impulse_multiplier[world, i] = 0.0
            world_drive_max_impulse[world, i] = 0.0
            continue

        stiffness = world_drive_stiffness[world, i]
        damping = world_drive_damping[world, i]
        unit_response = world_diag[world, i]

        a = dt * (dt * stiffness + damping)
        b = dt * (damping * world_target_velocity[world, i])
        x = float(0.0)
        if unit_response > 0.0:
            x = 1.0 / (1.0 + a * unit_response)

        world_drive_target_vel_bias[world, i] = x * b + stiffness * x * dt * world_drive_geom_error[world, i]
        world_drive_vel_multiplier[world, i] = -x * a
        world_drive_impulse_multiplier[world, i] = 1.0 - x

        max_force = world_drive_max_force[world, i]
        max_impulse = float(1.0e20)
        if max_force > 0.0 and wp.isfinite(max_force):
            max_impulse = max_force * dt
        world_drive_max_impulse[world, i] = max_impulse


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
def contact_row_points(
    c: int,
    row_offset: int,
    point_a_world: wp.vec3,
    point_b_world: wp.vec3,
    contact_shared_anchor: int,
    contact_friction_shared_anchor: int,
    friction_patches: FrictionPatches,
):
    """Return the Jacobian points of row ``row_offset`` (0 normal, 1-2 friction) of contact ``c``.

    Shared anchors put both bodies' points at the witness midpoint; patch friction rows act at
    the patch anchors. ``phi`` always comes from the witness points.
    """
    if row_offset > 0 and friction_patches.enabled != 0:
        return friction_patches.point_a[c], friction_patches.point_b[c]
    if contact_shared_anchor != 0 or (row_offset > 0 and contact_friction_shared_anchor != 0):
        midpoint = 0.5 * (point_a_world + point_b_world)
        return midpoint, midpoint
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
    propagation_articulated: int,
    propagation_same_articulation: int,
    propagation_free_free: int,
    max_constraints: int,
    mf_max_constraints: int,
    propagation_max_constraints: int,
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
    propagation_slot_counter: wp.array[int],
    dense_contact_world_flag: wp.array[int],
    dense_dropped_contact_rows: wp.array[int],
    mf_dropped_contact_rows: wp.array[int],
    propagation_dropped_contact_rows: wp.array[int],
    dense_first_rejected_slot: wp.array[int],
    mf_first_rejected_slot: wp.array[int],
    propagation_first_rejected_slot: wp.array[int],
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

    With a propagation response (``propagation_articulated``), contacts touching an
    articulation that is not a free body go to the body-space propagation rows
    (``contact_path == 2``) instead, except contacts between two links of the same
    articulation unless ``propagation_same_articulation`` is set.
    ``propagation_free_free`` also moves the free-body contacts to the propagation
    rows. ``dense_contact_world_flag`` marks the worlds that keep dense contact rows.

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
    is_propagation = False
    if propagation_free_free != 0 and is_mf:
        is_mf = False
        is_propagation = True
    if propagation_articulated != 0 and not is_mf and not is_propagation:
        a_non_free = art_a >= 0 and is_free_rigid[art_a] == 0
        b_non_free = art_b >= 0 and is_free_rigid[art_b] == 0
        same_articulation = a_non_free and b_non_free and art_a == art_b
        if (a_non_free or b_non_free) and (propagation_same_articulation != 0 or not same_articulation):
            is_propagation = True

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
    elif is_propagation:
        slot = wp.atomic_add(propagation_slot_counter, world, slots_needed)
        if slot + slots_needed > propagation_max_constraints:
            wp.atomic_min(propagation_first_rejected_slot, world, slot)
            wp.atomic_add(propagation_dropped_contact_rows, world, slots_needed)
            contact_slot[c] = -1
            contact_path[c] = -1
            return
        contact_path[c] = 2
    else:
        slot = wp.atomic_add(world_slot_counter, world, slots_needed)
        if slot + slots_needed > max_constraints:
            wp.atomic_min(dense_first_rejected_slot, world, slot)
            wp.atomic_add(dense_dropped_contact_rows, world, slots_needed)
            contact_slot[c] = -1
            contact_path[c] = -1
            return
        contact_path[c] = 0
        dense_contact_world_flag[world] = 1
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
    propagation_articulated: int,
    propagation_same_articulation: int,
    propagation_free_free: int,
    max_constraints: int,
    mf_max_constraints: int,
    propagation_max_constraints: int,
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
    propagation_slot_counter: wp.array[int],
    dense_contact_world_flag: wp.array[int],
    dense_dropped_contact_rows: wp.array[int],
    mf_dropped_contact_rows: wp.array[int],
    propagation_dropped_contact_rows: wp.array[int],
    dense_first_rejected_slot: wp.array[int],
    mf_first_rejected_slot: wp.array[int],
    propagation_first_rejected_slot: wp.array[int],
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
            propagation_articulated,
            propagation_same_articulation,
            propagation_free_free,
            max_constraints,
            mf_max_constraints,
            propagation_max_constraints,
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
            propagation_slot_counter,
            dense_contact_world_flag,
            dense_dropped_contact_rows,
            mf_dropped_contact_rows,
            propagation_dropped_contact_rows,
            dense_first_rejected_slot,
            mf_first_rejected_slot,
            propagation_first_rejected_slot,
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
    contact_shared_anchor: int,
    contact_friction_shared_anchor: int,
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
        point_a_normal, point_b_normal = contact_row_points(
            c, 0, point_a_world, point_b_world, contact_shared_anchor, contact_friction_shared_anchor, friction_patches
        )
        world_target_velocity[world, slot] = prescribed_relative_contact_target(
            body_a,
            art_a,
            body_b,
            art_b,
            point_a_normal,
            point_b_normal,
            normal,
            prescribed_articulation,
            articulation_origin,
            body_v_s,
        )
        # The allocation owns the row extent; recomputing the friction eligibility here
        # could cross a floating-point threshold and overwrite the next contact's rows.
        if contact_slots_needed[c] < 3:
            continue
        point_a_friction, point_b_friction = contact_row_points(
            c, 1, point_a_world, point_b_world, contact_shared_anchor, contact_friction_shared_anchor, friction_patches
        )
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
    contact_shared_anchor: int,
    contact_friction_shared_anchor: int,
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
        point_a_world, point_b_world = _contact_points_world(
            c, body_a, body_b, normal, contact_point0, contact_point1, contact_thickness0, contact_thickness1, body_q
        )
        point_a, point_b = contact_row_points(
            c,
            row,
            point_a_world,
            point_b_world,
            contact_shared_anchor,
            contact_friction_shared_anchor,
            friction_patches,
        )
        direction = normal
        if row > 0:
            tangent0, tangent1 = contact_tangent_basis(normal)
            if row == 1:
                direction = tangent0
            else:
                direction = tangent1

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
    contact_shared_anchor: int,
    contact_friction_shared_anchor: int,
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
    normal_point_a, normal_point_b = contact_row_points(
        c, 0, point_a_world, point_b_world, contact_shared_anchor, contact_friction_shared_anchor, friction_patches
    )
    friction_point_a, friction_point_b = contact_row_points(
        c, 1, point_a_world, point_b_world, contact_shared_anchor, contact_friction_shared_anchor, friction_patches
    )

    for side in range(2):
        art = art_a
        body = body_a
        point = normal_point_a
        friction_point = friction_point_a
        sign = 1.0
        if side == 1:
            art = art_b
            body = body_b
            point = normal_point_b
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
    contact_shared_anchor: int,
    contact_friction_shared_anchor: int,
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
            contact_shared_anchor,
            contact_friction_shared_anchor,
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
def scatter_augmented_drive_dof_K(
    group_to_art: wp.array[int],
    articulation_dof_start: wp.array[int],
    n_dofs: int,
    max_dofs: int,
    mass_update_mask: wp.array[int],
    row_counts: wp.array[int],
    row_dof_index: wp.array[int],
    row_K: wp.array[float],
    # outputs
    dof_K: wp.array[float],
):
    """Write each DOF's implicit drive term ``dt * kd + dt^2 * ke`` for a mass refresh.

    The DOF-indexed form of :func:`apply_augmented_mass_diagonal_grouped`, consumed by the
    sparse mass factors, which assemble the mass matrix without dense storage.
    """
    idx = wp.tid()
    articulation = group_to_art[idx]
    if mass_update_mask[articulation] == 0:
        return
    dof_start = articulation_dof_start[articulation]
    for local in range(n_dofs):
        dof_K[dof_start + local] = 0.0
    for i in range(row_counts[articulation]):
        row_index = articulation * max_dofs + i
        dof = row_dof_index[row_index]
        local = dof - dof_start
        K = row_K[row_index]
        if local >= 0 and local < n_dofs and K > 0.0:
            dof_K[dof] = K


@wp.kernel
def invert_lower_factor_grouped(
    group_to_art: wp.array[int],
    mass_update_mask: wp.array[int],
    n_dofs: int,
    L_group: wp.array3d[float],
    # outputs
    Linv_group: wp.array3d[float],
):
    """Invert each refreshed lower Cholesky factor of a size group by forward substitution."""
    idx = wp.tid()
    if mass_update_mask[group_to_art[idx]] == 0:
        return
    for col in range(n_dofs):
        for row in range(n_dofs):
            if row < col:
                Linv_group[idx, row, col] = 0.0
                continue
            value = float(0.0)
            if row == col:
                value = 1.0
            for k in range(col, row):
                value -= L_group[idx, row, k] * Linv_group[idx, k, col]
            Linv_group[idx, row, col] = value / L_group[idx, row, row]


@wp.kernel
def update_body_qd_from_featherstone(
    body_v_s: wp.array[wp.spatial_vector],
    body_q: wp.array[wp.transform],
    body_com: wp.array[wp.vec3],
    body_to_articulation: wp.array[int],
    articulation_origin: wp.array[wp.vec3],
    body_active: wp.array[int],
    body_qd_out: wp.array[wp.spatial_vector],
):
    tid = wp.tid()
    if body_active[tid] == 0:
        return

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
    their activation gap the remaining gap. Mimic and connect rows get the Baumgarte
    term for either sign of ``phi``. Friction rows of persistent patches get
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
        elif row_type == PGS_CONSTRAINT_TYPE_MIMIC or row_type == PGS_CONSTRAINT_TYPE_CONNECT:
            # Equality rows correct the violation in both directions.
            rhs += pgs_beta * phi * inv_dt
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
    match_generation: wp.array[wp.int32],
    contact_stream: int,
    history_generation: wp.array[wp.int32],
    history_stream: wp.array[wp.int32],
):
    """Relate the current contact set to the solved history: 0 unrelated, 1 same set, 2 matched to it.

    ``match_generation`` is the generation of this buffer that the match indices refer
    to (:attr:`~newton.Contacts.rigid_contact_match_generation`). Only the solved
    generation makes them usable: after a pass into another buffer or a skipped pass,
    they refer to a contact set this solver never solved.
    """
    if contact_stream == 0 or history_stream[0] != contact_stream:
        return 0
    previous = history_generation[0]
    if previous == CONTACT_GENERATION_NONE:
        return 0
    if contact_generation[0] == previous:
        return 1
    if match_generation[0] == previous:
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
    match_generation: wp.array[wp.int32],
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
    unchanged generation) the index is ``c`` itself. When the buffer's last collision
    pass matched against the solved set (``match_generation`` equals the solved
    generation), ``match_index[c]`` is contact ``c``'s index in that set. Any other
    relation (another buffer, skipped collision passes, a pass after the pipeline wrote
    another buffer, no history) starts every contact cold, since the match indices then
    refer to an unsolved set.
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

    relation = warmstart_history_relation(
        contact_generation, match_generation, contact_stream, history_generation, history_stream
    )
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
    previous_propagation_slot: wp.array[int],
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
    previous_propagation_slot[contact] = wp.where(path == 2, slot, -1)


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
    local_solve_owner: wp.array[int],
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
    # Local owners compute their response diagonals in their fused solve.
    if local_solve_owner[world] != PGS_LOCAL_SOLVE_OWNER_GENERAL:
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
    contact_shared_anchor: int,
    contact_friction_shared_anchor: int,
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
        row_point_a, row_point_b = contact_row_points(
            c,
            row_offset,
            point_a_world,
            point_b_world,
            contact_shared_anchor,
            contact_friction_shared_anchor,
            friction_patches,
        )

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
    contact_shared_anchor: int,
    contact_friction_shared_anchor: int,
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
            contact_shared_anchor,
            contact_friction_shared_anchor,
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
    articulation_rows_active: wp.array[int],
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
    if articulation_rows_active[art] == 0:
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
    world_row_type: wp.array2d[int],
    pgs_cfm: float,
    # in/out
    world_diag: wp.array2d[float],
):
    """Add constraint force mixing to the diagonal of every dense row except drive rows.

    A drive row's diagonal is its unit response ``J H^-1 J^T``: the force-drive update
    and the fused velocity clamp divide by the exact response, and the drive is already
    regularized by its own stiffness and damping.
    """
    world = wp.tid()
    for i in range(world_constraint_count[world]):
        if world_row_type[world, i] != PGS_CONSTRAINT_TYPE_JOINT_TARGET:
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
    articulation_active: wp.array[int],
    joint_qdd: wp.array[float],  # [total_dofs]
):
    """
    Solve L * L^T * qdd = tau for grouped articulations using forward/backward substitution.

    Thread dimension: n_arts_of_size (one thread per articulation in this size group)
    """
    idx = wp.tid()
    art = group_to_art[idx]
    if articulation_active[art] == 0:
        return
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
def factor_diagonal_mass(
    H_group: wp.array3d[float],  # [n_arts, n_dofs, n_dofs]
    R_group: wp.array2d[float],  # [n_arts, n_dofs]
    group_to_art: wp.array[int],
    mass_update_mask: wp.array[int],
    n_dofs: int,
    # output
    L_group: wp.array3d[float],  # [n_arts, n_dofs, n_dofs]
):
    """Factor structurally diagonal mass matrices, one thread per DOF.

    Writes only the diagonal of ``L``; its off-diagonal entries stay zero, so the result
    equals :func:`cholesky_loop` on the same matrix.
    """
    element = wp.tid()
    group = element // n_dofs
    dof = element - group * n_dofs
    art = group_to_art[group]
    if mass_update_mask[art] != 0:
        L_group[group, dof, dof] = wp.sqrt(H_group[group, dof, dof] + R_group[group, dof])


@wp.kernel
def solve_diagonal_mass(
    L_group: wp.array3d[float],  # [n_arts, n_dofs, n_dofs]
    group_to_art: wp.array[int],
    articulation_dof_start: wp.array[int],
    n_dofs: int,
    joint_tau: wp.array[float],  # [total_dofs]
    articulation_active: wp.array[int],
    # output
    joint_qdd: wp.array[float],  # [total_dofs]
):
    """Solve ``L L^T qdd = tau`` for a diagonal factor, one thread per DOF.

    Divides by the factor twice, in the order of :func:`trisolve_loop`.
    """
    element = wp.tid()
    group = element // n_dofs
    if articulation_active[group_to_art[group]] == 0:
        return
    dof = element - group * n_dofs
    global_dof = articulation_dof_start[group_to_art[group]] + dof
    factor = L_group[group, dof, dof]
    value = float(0.0)
    if factor != 0.0:
        value = joint_tau[global_dof] / factor
        value = value / factor
    joint_qdd[global_dof] = value


@wp.kernel
def hinv_jt_diagonal(
    L_group: wp.array3d[float],  # [n_arts, n_dofs, n_dofs]
    J_group: wp.array3d[float],  # [n_arts, max_constraints, n_dofs]
    group_to_art: wp.array[int],
    art_to_world: wp.array[int],
    articulation_world_dof_offset: wp.array[int],
    world_constraint_count: wp.array[int],
    n_dofs: int,
    max_constraints: int,
    n_arts: int,
    write_world: int,
    # outputs
    Y_group: wp.array3d[float],
    J_world: wp.array3d[float],
    Y_world: wp.array3d[float],
):
    """Compute ``Y = H^-1 J^T`` for a diagonal factor, one thread per row.

    The same outputs and division order as :func:`hinv_jt_par_row`, without its
    triangular substitutions.
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
    dof_offset = int(0)
    if write_world != 0:
        dof_offset = articulation_world_dof_offset[art]
    for i in range(n_dofs):
        jacobian = J_group[idx, c, i]
        factor = L_group[idx, i, i]
        response = float(0.0)
        if factor != 0.0:
            response = jacobian / factor
            response = response / factor
        Y_group[idx, c, i] = response
        if write_world != 0:
            J_world[world, c, dof_offset + i] = jacobian
            Y_world[world, c, dof_offset + i] = response


@wp.kernel
def gather_tau_to_groups(
    joint_tau: wp.array[float],  # [total_dofs]
    group_to_art: wp.array[int],
    articulation_dof_start: wp.array[int],
    n_dofs: int,
    articulation_active: wp.array[int],
    tau_group: wp.array3d[float],  # [n_arts, n_dofs, 1]
):
    """Gather joint_tau from 1D array into grouped 3D buffer for tiled solve.

    Thread dimension: n_arts_of_size (one thread per articulation in this size group)
    """
    idx = wp.tid()
    art = group_to_art[idx]
    if articulation_active[art] == 0:
        return
    dof_start = articulation_dof_start[art]
    for i in range(n_dofs):
        tau_group[idx, i, 0] = joint_tau[dof_start + i]


@wp.kernel
def scatter_qdd_from_groups(
    qdd_group: wp.array3d[float],  # [n_arts, n_dofs, 1]
    group_to_art: wp.array[int],
    articulation_dof_start: wp.array[int],
    n_dofs: int,
    articulation_active: wp.array[int],
    joint_qdd: wp.array[float],  # [total_dofs]
):
    """Scatter qdd from grouped 3D buffer back to 1D array after tiled solve.

    Thread dimension: n_arts_of_size (one thread per articulation in this size group)
    """
    idx = wp.tid()
    art = group_to_art[idx]
    if articulation_active[art] == 0:
        return
    dof_start = articulation_dof_start[art]
    for i in range(n_dofs):
        joint_qdd[dof_start + i] = qdd_group[idx, i, 0]


# =============================================================================
# Split-Mode Kernels (dense Delassus solve and standalone free-body solve)
# =============================================================================
# Split mode assembles the dense rows of each world into a Delassus matrix
# ``C = J H^-1 J^T`` and solves them in impulse space, then solves the free-body
# rows against the resulting velocity. These Warp kernels run on every device and
# are the CPU implementation; CUDA selects native one-warp-per-world variants built
# in ``solver_feather_pgs.py`` when they fit shared memory.


@wp.kernel
def build_joint_limit_rows(
    articulation_dof_start: wp.array[int],
    art_to_world: wp.array[int],
    group_to_art: wp.array[int],
    limit_q_index: wp.array[int],
    joint_limit_lower: wp.array[float],
    joint_limit_upper: wp.array[float],
    joint_q: wp.array[float],
    activation_gap: float,
    max_constraints: int,
    n_dofs: int,
    # outputs
    world_slot_counter: wp.array[int],
    J_group: wp.array3d[float],
    world_row_type: wp.array2d[int],
    world_row_parent: wp.array2d[int],
    world_row_mu: wp.array2d[float],
    world_phi: wp.array2d[float],
    world_target_velocity: wp.array2d[float],
):
    """Allocate and fill the active joint-limit rows of one articulation per thread.

    Visits the lower and then the upper bound of each limited DOF in DOF order, the
    candidate order of the one-warp-per-articulation CUDA builder.
    """
    group_idx = wp.tid()
    art = group_to_art[group_idx]
    world = art_to_world[art]
    dof_start = articulation_dof_start[art]
    for local_dof in range(n_dofs):
        dof = dof_start + local_dof
        q_index = limit_q_index[dof]
        if q_index < 0:
            continue
        q = joint_q[q_index]
        for side in range(2):
            bound = joint_limit_lower[dof]
            phi = q - bound
            active = wp.isfinite(bound) and q <= bound + activation_gap
            sign = 1.0
            if side == 1:
                bound = joint_limit_upper[dof]
                phi = bound - q
                active = wp.isfinite(bound) and q >= bound - activation_gap
                sign = -1.0
            if not active:
                continue
            slot = wp.atomic_add(world_slot_counter, world, 1)
            if slot >= max_constraints:
                continue
            J_group[group_idx, slot, local_dof] = sign
            world_row_type[world, slot] = PGS_CONSTRAINT_TYPE_JOINT_LIMIT
            world_row_parent[world, slot] = -1
            world_row_mu[world, slot] = 0.0
            world_phi[world, slot] = phi
            world_target_velocity[world, slot] = 0.0


@wp.kernel
def delassus_par_row_col(
    J_group: wp.array3d[float],
    Y_group: wp.array3d[float],
    group_to_art: wp.array[int],
    art_to_world: wp.array[int],
    world_constraint_count: wp.array[int],
    n_dofs: int,
    max_constraints: int,
    n_arts: int,
    # outputs
    world_C: wp.array3d[float],
    world_diag: wp.array2d[float],
):
    """Accumulate one size group's Delassus contribution ``C += J Y^T``, one entry per thread.

    Launched over ``n_arts * max_constraints * max_constraints`` threads. The diagonal is
    accumulated separately so the constraint force mixing can be added to it alone.
    """
    tid = wp.tid()
    j = tid % max_constraints
    i = (tid // max_constraints) % max_constraints
    idx = tid // (max_constraints * max_constraints)
    if idx >= n_arts:
        return
    art = group_to_art[idx]
    world = art_to_world[art]
    n_constraints = world_constraint_count[world]
    if i >= n_constraints or j >= n_constraints:
        return
    val = float(0.0)
    for k in range(n_dofs):
        val += J_group[idx, i, k] * Y_group[idx, j, k]
    if val != 0.0:
        wp.atomic_add(world_C, world, i, j, val)
        if i == j:
            wp.atomic_add(world_diag, world, i, val)


@wp.kernel
def rhs_accum_world_par_art(
    world_constraint_count: wp.array[int],
    art_to_world: wp.array[int],
    art_dof_start: wp.array[int],
    velocity: wp.array[float],
    group_to_art: wp.array[int],
    J_group: wp.array3d[float],
    n_dofs: int,
    # outputs
    world_rhs: wp.array2d[float],
):
    """Add one size group's ``J v`` to the dense right-hand side, one articulation per thread."""
    idx = wp.tid()
    art = group_to_art[idx]
    world = art_to_world[art]
    n_constraints = world_constraint_count[world]
    dof_start = art_dof_start[art]
    for c in range(n_constraints):
        jv = float(0.0)
        for d in range(n_dofs):
            jv += J_group[idx, c, d] * velocity[dof_start + d]
        wp.atomic_add(world_rhs, world, c, jv)


@wp.kernel
def pgs_solve_loop(
    world_constraint_count: wp.array[int],
    world_diag: wp.array2d[float],
    world_C: wp.array3d[float],
    world_rhs: wp.array2d[float],
    iterations: int,
    omega: float,
    world_row_type: wp.array2d[int],
    world_row_parent: wp.array2d[int],
    world_row_mu: wp.array2d[float],
    friction_start_iteration: int,
    iteration_offset: int,
    # in/out
    world_impulses: wp.array2d[float],
):
    """Projected Gauss-Seidel on the dense Delassus system of one world per thread.

    The residual of row ``i`` is ``rhs_i + sum_j C_ij lambda_j``. Contact and joint-limit
    rows are unilateral; the first friction row of a contact solves both tangents on the
    Coulomb disk of the current normal impulse (:func:`friction_pair_candidate`). Friction
    rows hold zero impulse in iterations before ``friction_start_iteration``, counted from
    ``iteration_offset``.
    """
    world = wp.tid()
    m = world_constraint_count[world]
    if m == 0:
        return
    for it in range(iterations):
        for i in range(m):
            row_type = world_row_type[world, i]
            if row_type == PGS_CONSTRAINT_TYPE_FRICTION and iteration_offset + it < friction_start_iteration:
                world_impulses[world, i] = 0.0
                continue
            if row_type == PGS_CONSTRAINT_TYPE_FRICTION and i != world_row_parent[world, i] + 1:
                continue

            w = world_rhs[world, i]
            for j in range(m):
                w += world_C[world, i, j] * world_impulses[world, j]

            denom = world_diag[world, i]
            if denom <= 0.0 and row_type != PGS_CONSTRAINT_TYPE_FRICTION:
                continue

            if row_type == PGS_CONSTRAINT_TYPE_FRICTION:
                parent_idx = world_row_parent[world, i]
                radius = wp.max(world_row_mu[world, i] * world_impulses[world, parent_idx], 0.0)
                if radius <= 0.0:
                    world_impulses[world, i] = 0.0
                    world_impulses[world, i + 1] = 0.0
                    continue
                sib = parent_idx + 2
                sibling_residual = world_rhs[world, sib]
                for j in range(m):
                    sibling_residual += world_C[world, sib, j] * world_impulses[world, j]
                trial = friction_pair_candidate(
                    denom,
                    world_C[world, i, sib],
                    world_diag[world, sib],
                    wp.vec2(w, sibling_residual),
                    wp.vec2(world_impulses[world, i], world_impulses[world, sib]),
                    radius,
                    omega,
                )
                magnitude = wp.length(trial)
                if magnitude > radius:
                    trial *= radius / magnitude
                world_impulses[world, i] = trial[0]
                world_impulses[world, sib] = trial[1]
            else:
                delta = -w / denom
                new_impulse = world_impulses[world, i] + omega * delta
                if new_impulse < 0.0:
                    new_impulse = 0.0
                world_impulses[world, i] = new_impulse


@wp.kernel
def apply_impulses_world_par_dof(
    group_to_art: wp.array[int],
    art_to_world: wp.array[int],
    art_dof_start: wp.array[int],
    n_dofs: int,
    n_arts: int,
    world_constraint_count: wp.array[int],
    Y_group: wp.array3d[float],
    world_impulses: wp.array2d[float],
    v_hat: wp.array[float],
    # outputs
    v_out: wp.array[float],
):
    """Write ``v_out = v_hat + Y lambda`` for one size group, one (articulation, DOF) per thread."""
    tid = wp.tid()
    local_dof = tid % n_dofs
    idx = tid // n_dofs
    if idx >= n_arts:
        return
    art = group_to_art[idx]
    world = art_to_world[art]
    delta_v = float(0.0)
    for c in range(world_constraint_count[world]):
        delta_v += Y_group[idx, c, local_dof] * world_impulses[world, c]
    global_dof = art_dof_start[art] + local_dof
    v_out[global_dof] = v_hat[global_dof] + delta_v


@wp.func
def _mf_row_velocity(
    world: int,
    row: int,
    mf_J_a: wp.array3d[float],
    mf_J_b: wp.array3d[float],
    dof_a: int,
    dof_b: int,
    v_out: wp.array[float],
):
    jv = float(0.0)
    if dof_a >= 0:
        for k in range(6):
            jv += mf_J_a[world, row, k] * v_out[dof_a + k]
    if dof_b >= 0:
        for k in range(6):
            jv += mf_J_b[world, row, k] * v_out[dof_b + k]
    return jv


@wp.func
def _mf_apply_impulse(
    world: int,
    row: int,
    mf_MiJt_a: wp.array3d[float],
    mf_MiJt_b: wp.array3d[float],
    dof_a: int,
    dof_b: int,
    delta_impulse: float,
    v_out: wp.array[float],
):
    for k in range(6):
        if dof_a >= 0:
            v_out[dof_a + k] = v_out[dof_a + k] + mf_MiJt_a[world, row, k] * delta_impulse
        if dof_b >= 0:
            v_out[dof_b + k] = v_out[dof_b + k] + mf_MiJt_b[world, row, k] * delta_impulse


@wp.kernel
def pgs_solve_mf_loop(
    mf_constraint_count: wp.array[int],
    mf_body_a: wp.array2d[int],
    mf_body_b: wp.array2d[int],
    mf_MiJt_a: wp.array3d[float],
    mf_MiJt_b: wp.array3d[float],
    mf_J_a: wp.array3d[float],
    mf_J_b: wp.array3d[float],
    mf_eff_mass_inv: wp.array2d[float],
    mf_rhs: wp.array2d[float],
    mf_row_type: wp.array2d[int],
    mf_row_parent: wp.array2d[int],
    mf_row_mu: wp.array2d[float],
    body_to_articulation: wp.array[int],
    art_dof_start: wp.array[int],
    iterations: int,
    omega: float,
    friction_start_iteration: int,
    iteration_offset: int,
    # in/out
    mf_impulses: wp.array2d[float],
    v_out: wp.array[float],
):
    """Projected Gauss-Seidel on the free-body rows of one world per thread.

    ``J v`` is recomputed from ``v_out`` for every row and each impulse change is applied
    immediately through ``M^-1 J^T``. Rows are laid out as ``[contacts and friction]
    [velocity limits]``, so the velocity limits have the last word in each sweep; they
    are stateless and apply only the impulse needed for the current overshoot. Friction
    rows hold zero impulse in iterations before ``friction_start_iteration``, counted from
    ``iteration_offset``.
    """
    world = wp.tid()
    m = mf_constraint_count[world]
    for it in range(iterations):
        for i in range(m):
            row_type = mf_row_type[world, i]
            if row_type == PGS_CONSTRAINT_TYPE_FRICTION and iteration_offset + it < friction_start_iteration:
                mf_impulses[world, i] = 0.0
                continue
            parent_idx = mf_row_parent[world, i]
            if row_type == PGS_CONSTRAINT_TYPE_FRICTION and i != parent_idx + 1:
                continue
            eff_inv = mf_eff_mass_inv[world, i]
            if eff_inv <= 0.0 and row_type != PGS_CONSTRAINT_TYPE_FRICTION:
                continue

            ba = mf_body_a[world, i]
            bb = mf_body_b[world, i]
            dof_a = int(-1)
            dof_b = int(-1)
            if ba >= 0:
                dof_a = art_dof_start[body_to_articulation[ba]]
            if bb >= 0:
                dof_b = art_dof_start[body_to_articulation[bb]]

            residual = _mf_row_velocity(world, i, mf_J_a, mf_J_b, dof_a, dof_b, v_out) + mf_rhs[world, i]
            old_impulse = mf_impulses[world, i]
            delta = -residual * eff_inv
            new_impulse = old_impulse + omega * delta
            delta_impulse = float(0.0)
            if row_type == PGS_CONSTRAINT_TYPE_JOINT_VELOCITY_LIMIT:
                new_impulse = float(0.0)
                if residual < 0.0:
                    new_impulse = delta
                delta_impulse = new_impulse
            elif row_type == PGS_CONSTRAINT_TYPE_FRICTION:
                sib = parent_idx + 2
                radius = wp.max(mf_row_mu[world, i] * mf_impulses[world, parent_idx], 0.0)
                pair_residual = wp.vec2(mf_rhs[world, i], mf_rhs[world, sib])
                cross = float(0.0)
                for k in range(6):
                    if dof_a >= 0:
                        pair_residual[0] += mf_J_a[world, i, k] * v_out[dof_a + k]
                        pair_residual[1] += mf_J_a[world, sib, k] * v_out[dof_a + k]
                        cross += mf_J_a[world, i, k] * mf_MiJt_a[world, sib, k]
                    if dof_b >= 0:
                        pair_residual[0] += mf_J_b[world, i, k] * v_out[dof_b + k]
                        pair_residual[1] += mf_J_b[world, sib, k] * v_out[dof_b + k]
                        cross += mf_J_b[world, i, k] * mf_MiJt_b[world, sib, k]
                first_diag = float(0.0)
                if eff_inv > 0.0:
                    first_diag = 1.0 / eff_inv
                sibling_inv = mf_eff_mass_inv[world, sib]
                sibling_diag = float(0.0)
                if sibling_inv > 0.0:
                    sibling_diag = 1.0 / sibling_inv
                trial = friction_pair_candidate(
                    first_diag,
                    cross,
                    sibling_diag,
                    pair_residual,
                    wp.vec2(old_impulse, mf_impulses[world, sib]),
                    radius,
                    omega,
                )
                magnitude = wp.length(trial)
                if magnitude > radius:
                    trial *= radius / magnitude
                sibling_delta = trial[1] - mf_impulses[world, sib]
                mf_impulses[world, sib] = trial[1]
                _mf_apply_impulse(world, sib, mf_MiJt_a, mf_MiJt_b, dof_a, dof_b, sibling_delta, v_out)
                new_impulse = trial[0]
                delta_impulse = new_impulse - old_impulse
            else:
                if new_impulse < 0.0:
                    new_impulse = 0.0
                delta_impulse = new_impulse - old_impulse
            mf_impulses[world, i] = new_impulse
            _mf_apply_impulse(world, i, mf_MiJt_a, mf_MiJt_b, dof_a, dof_b, delta_impulse, v_out)


@wp.kernel
def scale_array_inplace(values: wp.array[float], scale: float):
    """Multiply every entry of ``values`` by ``scale``."""
    i = wp.tid()
    values[i] = values[i] * scale


@wp.kernel
def vector_add_inplace(a: wp.array[float], b: wp.array[float]):
    """Add ``b`` to ``a`` element-wise."""
    i = wp.tid()
    a[i] = a[i] + b[i]


@wp.kernel
def compute_delta_and_accumulate(
    v_out: wp.array[float],
    v_snap: wp.array[float],
    v_accum: wp.array[float],
):
    """Accumulate ``delta = v_out - v_snap`` into ``v_accum`` and store ``delta`` in ``v_snap``."""
    i = wp.tid()
    delta = v_out[i] - v_snap[i]
    v_accum[i] = v_accum[i] + delta
    v_snap[i] = delta


# ---------------------------------------------------------------------------
# Propagation contact response
#
# Contacts touching an articulated (non-free) body are solved as fixed-size
# body-space rows ``J = [d, r x d]`` at the body's center of mass. The row solve
# accumulates each impulse on the touched bodies, and the articulated-body
# factorization of the tree (Featherstone's articulated-body algorithm) carries
# the accumulated impulses to the joint velocities once per iteration. Body
# twists and wrenches are world-aligned and referenced at each body's center of
# mass, whose position relative to the articulation origin is
# ``propagation_body_com_rel``.
# ---------------------------------------------------------------------------


@wp.func
def translate_twist_between_parallel_frames(twist: wp.spatial_vector, dest_minus_source: wp.vec3):
    """Translate a world-aligned twist from one origin to another."""
    lin = wp.spatial_top(twist)
    ang = wp.spatial_bottom(twist)
    return wp.spatial_vector(lin + wp.cross(ang, dest_minus_source), ang)


@wp.func
def translate_wrench_between_parallel_frames(wrench: wp.spatial_vector, source_minus_dest: wp.vec3):
    """Translate a world-aligned wrench from one origin to another."""
    force = wp.spatial_top(wrench)
    torque = wp.spatial_bottom(wrench)
    return wp.spatial_vector(force, torque + wp.cross(source_minus_dest, force))


@wp.func
def _spatial_row(values: wp.array2d[float], row: int):
    """Load one row of a ``[n, 6]`` array as a spatial vector."""
    return wp.spatial_vector(
        values[row, 0], values[row, 1], values[row, 2], values[row, 3], values[row, 4], values[row, 5]
    )


@wp.func
def _com_edge(propagation_body_com_rel: wp.array2d[float], child: int, parent: int):
    """Return the vector from the parent's to the child's center of mass."""
    return wp.vec3(
        propagation_body_com_rel[child, 0] - propagation_body_com_rel[parent, 0],
        propagation_body_com_rel[child, 1] - propagation_body_com_rel[parent, 1],
        propagation_body_com_rel[child, 2] - propagation_body_com_rel[parent, 2],
    )


@wp.func
def _unit_spatial_vector(k: int):
    """Return the ``k``-th spatial basis vector."""
    e = wp.spatial_vector()
    e[k] = 1.0
    return e


@wp.kernel
def build_propagation_contact_rows(
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
    shape_body: wp.array[int],
    body_q: wp.array[wp.transform],
    body_com: wp.array[wp.vec3],
    body_v_s: wp.array[wp.spatial_vector],
    prescribed_articulation: wp.array[int],
    articulation_origin: wp.array[wp.vec3],
    shape_material_mu: wp.array[float],
    shape_material_restitution: wp.array[float],
    contact_slots_needed: wp.array[int],
    contact_shared_anchor: int,
    contact_friction_shared_anchor: int,
    friction_patches: FrictionPatches,
    friction_anchor_beta: float,
    # outputs
    propagation_body_a: wp.array2d[int],
    propagation_body_b: wp.array2d[int],
    propagation_J_a: wp.array3d[float],
    propagation_J_b: wp.array3d[float],
    propagation_row_type: wp.array2d[int],
    propagation_row_parent: wp.array2d[int],
    propagation_row_mu: wp.array2d[float],
    propagation_phi: wp.array2d[float],
    propagation_target_velocity: wp.array2d[float],
    propagation_row_restitution: wp.array2d[float],
):
    """Build the body-space normal row and, when allocated, the two friction rows of every propagation contact.

    Each side's row is ``J = [d, r x d]`` with ``r`` the contact point relative to the
    body's center of mass. A side without response DOFs (ground, a zero-DOF articulation
    or a prescribed kinematic body) gets body ``-1``; its prescribed motion enters
    through the row target velocity. With friction patches the friction rows act at the
    patch anchor points and their ``phi`` holds ``friction_anchor_beta`` times the anchor's
    tangential displacement.
    """
    total_contacts = wp.min(contact_count[0], contact_point0.shape[0])
    for c in range(wp.tid(), total_contacts, total_num_threads):
        if contact_path[c] != 2:
            continue
        slot = contact_slot[c]
        if slot < 0:
            continue

        world = contact_world[c]
        art_a = contact_art_a[c]
        art_b = contact_art_b[c]
        shape_a = contact_shape0[c]
        shape_b = contact_shape1[c]
        # The contact normal is stored A-to-B; rows use B-to-A.
        normal = -contact_normal[c]
        body_a = -1
        body_b = -1
        if shape_a >= 0:
            body_a = shape_body[shape_a]
        if shape_b >= 0:
            body_b = shape_body[shape_b]
        response_body_a = -1
        response_body_b = -1
        if body_a >= 0 and art_a >= 0 and articulation_response_dof_count[art_a] > 0:
            response_body_a = body_a
        if body_b >= 0 and art_b >= 0 and articulation_response_dof_count[art_b] > 0:
            response_body_b = body_b

        point_a_world, point_b_world = _contact_points_world(
            c, body_a, body_b, normal, contact_point0, contact_point1, contact_thickness0, contact_thickness1, body_q
        )
        phi = wp.dot(normal, point_a_world - point_b_world)
        mu = _contact_mu(shape_a, shape_b, shape_material_mu)
        tangent0, tangent1 = contact_tangent_basis(normal)
        com_a = wp.vec3(0.0)
        com_b = wp.vec3(0.0)
        if response_body_a >= 0:
            com_a = wp.transform_point(body_q[response_body_a], body_com[response_body_a])
        if response_body_b >= 0:
            com_b = wp.transform_point(body_q[response_body_b], body_com[response_body_b])

        restitution = mixed_contact_restitution(shape_a, shape_b, shape_material_restitution)
        for row_offset in range(contact_slots_needed[c]):
            row = slot + row_offset
            d = normal
            if row_offset == 1:
                d = tangent0
            elif row_offset == 2:
                d = tangent1
            row_point_a, row_point_b = contact_row_points(
                c,
                row_offset,
                point_a_world,
                point_b_world,
                contact_shared_anchor,
                contact_friction_shared_anchor,
                friction_patches,
            )
            if response_body_a >= 0:
                ang_a = wp.cross(row_point_a - com_a, d)
                for k in range(3):
                    propagation_J_a[world, row, k] = d[k]
                    propagation_J_a[world, row, 3 + k] = ang_a[k]
            if response_body_b >= 0:
                ang_b = wp.cross(row_point_b - com_b, d)
                for k in range(3):
                    propagation_J_b[world, row, k] = -d[k]
                    propagation_J_b[world, row, 3 + k] = -ang_b[k]
            propagation_body_a[world, row] = response_body_a
            propagation_body_b[world, row] = response_body_b
            propagation_row_mu[world, row] = mu
            if row_offset == 0:
                propagation_row_type[world, row] = PGS_CONSTRAINT_TYPE_CONTACT
                propagation_row_parent[world, row] = -1
                propagation_phi[world, row] = phi
                propagation_row_restitution[world, row] = restitution
            else:
                propagation_row_type[world, row] = PGS_CONSTRAINT_TYPE_FRICTION
                propagation_row_parent[world, row] = slot
                propagation_phi[world, row] = friction_anchor_beta * friction_patches.phi[c][row_offset - 1]
                propagation_row_restitution[world, row] = 0.0
            propagation_target_velocity[world, row] = prescribed_relative_contact_target(
                body_a,
                art_a,
                body_b,
                art_b,
                row_point_a,
                row_point_b,
                d,
                prescribed_articulation,
                articulation_origin,
                body_v_s,
            )


@wp.kernel
def copy_free_rigid_propagation_body_response(
    free_rigid_body_indices: wp.array[int],
    mf_body_Hinv: wp.array[wp.spatial_matrix],
    # outputs
    propagation_body_response: wp.array3d[float],
):
    """Seed the 6x6 propagation response of free bodies with their inverse inertia.

    Articulated links get their response from the tree factorization; a free body's
    response is the same inverse spatial inertia its free-body rows use.
    """
    body = free_rigid_body_indices[wp.tid()]
    Hinv = mf_body_Hinv[body]
    for r in range(6):
        for c in range(6):
            propagation_body_response[body, r, c] = Hinv[r, c]


@wp.func
def propagation_contact_row_dot(
    J_a: wp.array3d[float],
    J_b: wp.array3d[float],
    body_qd: wp.array2d[float],
    world: int,
    row: int,
    body_a: int,
    body_b: int,
):
    """Return ``J v`` of one propagation row from the body twists."""
    value = float(0.0)
    for k in range(6):
        if body_a >= 0:
            value += J_a[world, row, k] * body_qd[body_a, k]
        if body_b >= 0:
            value += J_b[world, row, k] * body_qd[body_b, k]
    return value


@wp.kernel
def count_propagation_coupled_bodies(
    propagation_constraint_count: wp.array[int],
    propagation_body_a: wp.array2d[int],
    propagation_body_b: wp.array2d[int],
    propagation_max_constraints: int,
    propagation_body_coupling_group: wp.array[int],
    # outputs
    propagation_body_split_seen: wp.array[int],
    propagation_coupling_group_body_count: wp.array[int],
):
    """Count the distinct row-bearing bodies in each coupling group."""
    tid = wp.tid()
    world = tid // propagation_max_constraints
    i = tid - world * propagation_max_constraints
    if i >= wp.min(propagation_constraint_count[world], propagation_max_constraints):
        return
    ba = propagation_body_a[world, i]
    if ba >= 0 and wp.atomic_add(propagation_body_split_seen, ba, 1) == 0:
        wp.atomic_add(propagation_coupling_group_body_count, propagation_body_coupling_group[ba], 1)
    bb = propagation_body_b[world, i]
    if bb >= 0 and wp.atomic_add(propagation_body_split_seen, bb, 1) == 0:
        wp.atomic_add(propagation_coupling_group_body_count, propagation_body_coupling_group[bb], 1)


@wp.func
def propagation_body_split(
    body: int,
    propagation_body_coupling_group: wp.array[int],
    propagation_coupling_group_body_count: wp.array[int],
) -> float:
    return float(wp.max(propagation_coupling_group_body_count[propagation_body_coupling_group[body]], 1))


@wp.kernel
def compute_propagation_effective_mass_and_rhs(
    propagation_constraint_count: wp.array[int],
    propagation_body_a: wp.array2d[int],
    propagation_body_b: wp.array2d[int],
    propagation_J_a: wp.array3d[float],
    propagation_J_b: wp.array3d[float],
    propagation_body_response: wp.array3d[float],
    propagation_body_coupling_group: wp.array[int],
    propagation_coupling_group_body_count: wp.array[int],
    propagation_phi: wp.array2d[float],
    propagation_row_type: wp.array2d[int],
    propagation_target_velocity: wp.array2d[float],
    propagation_row_restitution: wp.array2d[float],
    propagation_body_qd: wp.array2d[float],
    rigid_body_max_depenetration_velocity: wp.array[float],
    pgs_cfm: float,
    pgs_beta: float,
    contact_w: float,
    dt: float,
    contact_speculative_scale: float,
    restitution_velocity_threshold: float,
    propagation_max_constraints: int,
    # outputs
    propagation_eff_mass_inv: wp.array2d[float],
    propagation_MiJt_a: wp.array3d[float],
    propagation_MiJt_b: wp.array3d[float],
    propagation_rhs: wp.array2d[float],
    propagation_restitution_bias: wp.array2d[float],
    propagation_row_w: wp.array2d[float],
):
    """Compute the response ``M^-1 J^T``, inverse effective mass and bias of each propagation row.

    ``M^-1`` is each touched body's own 6x6 response, so a row between two links of one
    articulation misses their cross term here; see
    :func:`refine_same_articulation_propagation_rows`. Each response is scaled by the
    number of row-bearing bodies in the body's coupling group (mass splitting). The bias follows the free-body
    rows: penetrating contacts get the Baumgarte term, bounded by the bodies' maximum
    depenetration velocity, and separated contacts may close ``contact_speculative_scale``
    times their gap during the step. An impacting contact whose rebound fires gets its
    restitution target, from the twists in ``propagation_body_qd`` (the unconstrained
    velocity), which ``propagation_restitution_bias`` keeps for the velocity-only
    iterations. Patch friction rows get their anchor bias. ``propagation_row_w`` receives
    the regularization weight when regularization is on.
    """
    tid = wp.tid()
    world = tid // propagation_max_constraints
    i = tid - world * propagation_max_constraints
    if i >= propagation_constraint_count[world]:
        return

    ba = propagation_body_a[world, i]
    bb = propagation_body_b[world, i]
    d = pgs_cfm
    d_unsplit = pgs_cfm
    # Mass splitting: a sweep sees only each body's own response, so a body shares its coupling
    # group's mobility with the group's other row-bearing bodies to damp the Jacobi update.
    if ba >= 0:
        split_a = propagation_body_split(ba, propagation_body_coupling_group, propagation_coupling_group_body_count)
        for r in range(6):
            value = float(0.0)
            for c in range(6):
                value += propagation_body_response[ba, r, c] * propagation_J_a[world, i, c]
            d_unsplit += propagation_J_a[world, i, r] * value
            value *= split_a
            propagation_MiJt_a[world, i, r] = value
            d += propagation_J_a[world, i, r] * value
    if bb >= 0:
        split_b = propagation_body_split(bb, propagation_body_coupling_group, propagation_coupling_group_body_count)
        for r in range(6):
            value = float(0.0)
            for c in range(6):
                value += propagation_body_response[bb, r, c] * propagation_J_b[world, i, c]
            d_unsplit += propagation_J_b[world, i, r] * value
            value *= split_b
            propagation_MiJt_b[world, i, r] = value
            d += propagation_J_b[world, i, r] * value
    if d > 0.0:
        propagation_eff_mass_inv[world, i] = 1.0 / d
    else:
        propagation_eff_mass_inv[world, i] = 0.0

    bias = float(0.0)
    restitution_bias = float(0.0)
    row_w = float(1.0)
    row_type = propagation_row_type[world, i]
    target_velocity = propagation_target_velocity[world, i]
    if row_type == PGS_CONSTRAINT_TYPE_CONTACT:
        phi = propagation_phi[world, i]
        if phi <= 0.0:
            # Speculative rows stay rigid so a closing contact reaches the surface.
            row_w = contact_w
        if phi < 0.0:
            bias = pgs_beta * phi / dt
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
            bias = contact_speculative_scale * phi / dt
        restitution = propagation_row_restitution[world, i]
        if restitution > 0.0:
            relative_incident = (
                propagation_contact_row_dot(propagation_J_a, propagation_J_b, propagation_body_qd, world, i, ba, bb)
                - target_velocity
            )
            if contact_restitution_fires(phi, relative_incident, dt, restitution_velocity_threshold):
                bias = restitution * relative_incident
                restitution_bias = bias
                # An impact is impulsive, not a spring: keep the rebound exact.
                row_w = 1.0
    elif row_type == PGS_CONSTRAINT_TYPE_FRICTION:
        bias = propagation_phi[world, i] / dt
    propagation_rhs[world, i] = bias - target_velocity
    propagation_restitution_bias[world, i] = restitution_bias
    if contact_w < 1.0:
        if row_w < 1.0 and d != d_unsplit:
            # Regularize against the unsplit diagonal so splitting changes the step, not the fixed point.
            row_w = contact_w * d / (contact_w * d + (1.0 - contact_w) * d_unsplit)
        propagation_row_w[world, i] = row_w


@wp.kernel
def compute_propagation_velocity_rhs(
    propagation_constraint_count: wp.array[int],
    propagation_body_a: wp.array2d[int],
    propagation_body_b: wp.array2d[int],
    propagation_J_a: wp.array3d[float],
    propagation_J_b: wp.array3d[float],
    propagation_phi: wp.array2d[float],
    propagation_row_type: wp.array2d[int],
    propagation_target_velocity: wp.array2d[float],
    propagation_restitution_bias: wp.array2d[float],
    position_body_qd: wp.array2d[float],
    dt: float,
    propagation_max_constraints: int,
    # outputs
    propagation_rhs: wp.array2d[float],
):
    """Build the propagation right-hand side of the velocity-only iterations.

    Mirrors :func:`compute_mf_velocity_rhs`: no position bias, except that a positive-gap
    contact whose end gap after the position solve stays above the slop keeps
    ``phi / dt`` and an impacting contact keeps the rebound target frozen before the
    position solve. ``position_body_qd`` holds the body twists after the position solve.
    """
    tid = wp.tid()
    world = tid // propagation_max_constraints
    i = tid - world * propagation_max_constraints
    if i >= propagation_constraint_count[world]:
        return

    target_velocity = propagation_target_velocity[world, i]
    bias = float(0.0)
    if propagation_row_type[world, i] == PGS_CONSTRAINT_TYPE_CONTACT:
        restitution_bias = propagation_restitution_bias[world, i]
        phi = propagation_phi[world, i]
        if restitution_bias != 0.0:
            bias = restitution_bias
        elif phi > 0.0:
            jv_position = propagation_contact_row_dot(
                propagation_J_a,
                propagation_J_b,
                position_body_qd,
                world,
                i,
                propagation_body_a[world, i],
                propagation_body_b[world, i],
            )
            end_gap = phi + dt * (jv_position - target_velocity)
            if end_gap > _FPGS_CONTACT_END_GAP_SLOP:
                bias = phi / dt
    propagation_rhs[world, i] = bias - target_velocity


@wp.kernel
def accumulate_propagation_warmstart_body_impulses(
    propagation_constraint_count: wp.array[int],
    propagation_body_a: wp.array2d[int],
    propagation_body_b: wp.array2d[int],
    propagation_J_a: wp.array3d[float],
    propagation_J_b: wp.array3d[float],
    propagation_MiJt_a: wp.array3d[float],
    propagation_MiJt_b: wp.array3d[float],
    propagation_impulses: wp.array2d[float],
    propagation_max_constraints: int,
    # in/out
    propagation_body_qd: wp.array2d[float],
    propagation_body_impulses: wp.array2d[float],
):
    """Turn the seeded propagation impulses into body twists and pending body impulses.

    The tree propagation then applies the pending impulses to the joint velocities.
    """
    tid = wp.tid()
    world = tid // propagation_max_constraints
    row = tid - world * propagation_max_constraints
    if row >= propagation_constraint_count[world]:
        return
    impulse = propagation_impulses[world, row]
    if impulse == 0.0:
        return
    ba = propagation_body_a[world, row]
    bb = propagation_body_b[world, row]
    for k in range(6):
        if ba >= 0:
            wp.atomic_add(propagation_body_qd, ba, k, propagation_MiJt_a[world, row, k] * impulse)
            wp.atomic_add(propagation_body_impulses, ba, k, propagation_J_a[world, row, k] * impulse)
        if bb >= 0:
            wp.atomic_add(propagation_body_qd, bb, k, propagation_MiJt_b[world, row, k] * impulse)
            wp.atomic_add(propagation_body_impulses, bb, k, propagation_J_b[world, row, k] * impulse)


# Coloring of the propagation contact units ("propagation-colored").
#
# Two units conflict when they share a body with a response: their twist updates and deferred
# impulses overlap. The units of one color are body-disjoint and solve in parallel; colors run in
# sequence, a Gauss-Seidel reordering of the serial sweep. A unit's friction rows share its bodies,
# so a friction pair's sibling write has one writer. The coloring is first-fit greedy over the
# world's units in contact-index order, which uses at most 2 * degree - 1 colors; units left
# without a color below the cap run in an ordered serial tail.

PROPAGATION_COLOR_TAIL = 512
"""Color cap of the unit coloring; the index of the serial tail bucket."""

PROPAGATION_UNIT_RING = 4
"""Patch-ring members carried in a unit record; longer rings resume the walk after them."""

PROPAGATION_UNIT_META = 12
"""Words per unit record of the colored solve, ``7 + PROPAGATION_UNIT_RING`` padded for 16-byte copies."""


@wp.kernel(enable_backward=False)
def collect_propagation_units(
    contact_count: wp.array[int],
    contact_path: wp.array[int],
    contact_world: wp.array[int],
    contact_shape0: wp.array[int],
    contact_shape1: wp.array[int],
    contact_art_a: wp.array[int],
    contact_art_b: wp.array[int],
    articulation_response_dof_count: wp.array[int],
    shape_body: wp.array[int],
    contact_slots_needed: wp.array[int],
    propagation_max_constraints: int,
    # in/out
    world_unit_cursor: wp.array[int],
    # outputs
    unit_contact: wp.array[int],
    unit_body_a: wp.array[int],
    unit_body_b: wp.array[int],
    unit_len: wp.array[int],
):
    """Gather the propagation contacts into per-world unit lists for the coloring.

    A side without response DOFs (ground or a prescribed kinematic body) is recorded as ``-1``,
    as in the rows, so it never conflicts.
    """
    c = wp.tid()
    if c >= contact_count[0]:
        return
    if contact_path[c] != 2:
        return
    world = contact_world[c]
    idx = wp.atomic_add(world_unit_cursor, world, 1)
    if idx >= propagation_max_constraints:
        return
    base = world * propagation_max_constraints
    body_a = -1
    body_b = -1
    art_a = contact_art_a[c]
    art_b = contact_art_b[c]
    shape_a = contact_shape0[c]
    shape_b = contact_shape1[c]
    if shape_a >= 0 and art_a >= 0 and articulation_response_dof_count[art_a] > 0:
        body_a = shape_body[shape_a]
    if shape_b >= 0 and art_b >= 0 and articulation_response_dof_count[art_b] > 0:
        body_b = shape_body[shape_b]
    unit_contact[base + idx] = c
    unit_body_a[base + idx] = body_a
    unit_body_b[base + idx] = body_b
    unit_len[base + idx] = contact_slots_needed[c]


@wp.kernel(enable_backward=False)
def gather_propagation_unit_meta(
    propagation_max_constraints: int,
    n_color_entries: int,
    world_color_offsets: wp.array[int],
    world_row_order: wp.array[int],
    row_type: wp.array2d[int],
    row_parent: wp.array2d[int],
    body_a: wp.array2d[int],
    body_b: wp.array2d[int],
    body_local_slot: wp.array[int],
    constraint_count: wp.array[int],
    ring_overflow_cursor: wp.array[int],
    straight_line: int,
    # out
    unit_meta: wp.array[int],
    ring_overflow: wp.array[int],
):
    """Pack the static record of every colored unit, indexed by its position in color order.

    Words: start slot, local slots of bodies a and b, the stored patch-ring length, the first
    ``PROPAGATION_UNIT_RING`` ring members, the member after them, the row count, the
    straight-line kind (1 a contact row with its friction pair, 2 a lone contact row, 0 the
    row loop) and the rest of the ring in ``ring_overflow`` (offset | count << 20, or -1).
    """
    tid = wp.tid()
    world = tid // propagation_max_constraints
    pos = tid - world * propagation_max_constraints
    if pos >= world_color_offsets[world * n_color_entries + n_color_entries - 1]:
        return
    slot = world_row_order[tid]
    ba = body_a[world, slot]
    bb = body_b[world, slot]
    la = int(-1)
    lb = int(-1)
    if ba >= 0:
        la = body_local_slot[ba]
    if bb >= 0:
        lb = body_local_slot[bb]
    base = tid * PROPAGATION_UNIT_META
    unit_meta[base + 0] = slot
    unit_meta[base + 1] = la
    unit_meta[base + 2] = lb
    n = int(0)
    member = int(-1)
    if row_type[world, slot] == PGS_CONSTRAINT_TYPE_CONTACT:
        member = row_parent[world, slot]
        while member >= 0 and member != slot and n < PROPAGATION_UNIT_RING:
            unit_meta[base + 4 + n] = member
            n += 1
            member = row_parent[world, member]
    for q in range(n, PROPAGATION_UNIT_RING):
        unit_meta[base + 4 + q] = -1
    # Past the stored members: -1 when the ring closed within them.
    if member == slot:
        member = -1
    unit_meta[base + 3] = n
    unit_meta[base + 4 + PROPAGATION_UNIT_RING] = member
    # The rest of a long ring, in ring order, goes to the world's overflow list for parallel loads.
    overflow = int(-1)
    if member >= 0:
        count = int(0)
        walk = member
        while walk >= 0 and walk != slot:
            count += 1
            walk = row_parent[world, walk]
        offset = wp.atomic_add(ring_overflow_cursor, world, count)
        if offset + count <= propagation_max_constraints and offset < (1 << 20) and count < 1024:
            walk = member
            q = int(0)
            while walk >= 0 and walk != slot:
                ring_overflow[world * propagation_max_constraints + offset + q] = walk
                q += 1
                walk = row_parent[world, walk]
            overflow = offset | (count << 20)
    unit_meta[base + 7 + PROPAGATION_UNIT_RING] = overflow
    # Rows of the unit: its first row and the friction rows following it.
    m = wp.min(constraint_count[world], int(row_type.shape[1]))
    rows = int(1)
    while slot + rows < m and row_type[world, slot + rows] == PGS_CONSTRAINT_TYPE_FRICTION:
        rows += 1
    unit_meta[base + 5 + PROPAGATION_UNIT_RING] = rows
    # Straight-line kinds: a contact row with its friction pair on the unit's two bodies, or a lone contact row.
    standard = int(0)
    if (
        straight_line != 0
        and rows == 3
        and row_type[world, slot] == PGS_CONSTRAINT_TYPE_CONTACT
        and (la != lb or la < 0)
    ):
        if row_parent[world, slot + 1] == slot and row_parent[world, slot + 2] == slot:
            if body_a[world, slot + 1] == ba and body_b[world, slot + 1] == bb:
                if body_a[world, slot + 2] == ba and body_b[world, slot + 2] == bb:
                    standard = 1
    if (
        straight_line != 0
        and rows == 1
        and row_type[world, slot] == PGS_CONSTRAINT_TYPE_CONTACT
        and (la != lb or la < 0)
    ):
        standard = 2  # a lone contact row
    unit_meta[base + 6 + PROPAGATION_UNIT_RING] = standard


@wp.kernel
def build_propagation_body_map(
    propagation_constraint_count: wp.array[int],
    propagation_body_a: wp.array2d[int],
    propagation_body_b: wp.array2d[int],
    propagation_max_constraints: int,
    max_propagation_bodies: int,
    propagation_body_seen: wp.array[int],
    # outputs
    propagation_body_list: wp.array2d[int],
    propagation_body_count: wp.array[int],
    propagation_body_local_slot: wp.array[int],
):
    """List the distinct bodies touched by each world's propagation rows."""
    tid = wp.tid()
    world = tid // propagation_max_constraints
    i = tid - world * propagation_max_constraints
    m = wp.min(propagation_constraint_count[world], propagation_max_constraints)
    if i >= m:
        return

    for side in range(2):
        body = propagation_body_a[world, i]
        if side == 1:
            body = propagation_body_b[world, i]
        if body < 0:
            continue
        if wp.atomic_add(propagation_body_seen, body, 1) == 0:
            slot = wp.atomic_add(propagation_body_count, world, 1)
            if slot < max_propagation_bodies:
                propagation_body_list[world, slot] = body
                propagation_body_local_slot[body] = slot


@wp.kernel
def build_propagation_body_map_partitioned(
    propagation_constraint_count: wp.array[int],
    propagation_body_a: wp.array2d[int],
    propagation_body_b: wp.array2d[int],
    propagation_max_constraints: int,
    max_propagation_bodies: int,
    body_to_articulation: wp.array[int],
    propagation_cache_art_eligible: wp.array[int],
    want_eligible: int,
    propagation_body_seen: wp.array[int],
    # outputs
    propagation_body_list: wp.array2d[int],
    propagation_body_count: wp.array[int],
    propagation_body_local_slot: wp.array[int],
):
    """One partition pass of the body-map build.

    Same claim logic as :func:`build_propagation_body_map`, restricted to
    bodies whose articulation's cache eligibility equals ``want_eligible``.
    The cached-response path launches this twice — eligible bodies first,
    everything else second — so cache-eligible bodies occupy a contiguous
    slot prefix and the capacity gate can ignore free-rigid clutter and
    non-cacheable articulations (their impulses go through the flush / the
    unconditional tree walk, never through the cache). The eligibility
    predicate runs BEFORE the seen-claim so the second pass can still claim
    the bodies the first pass skipped.
    """
    tid = wp.tid()
    world = tid // propagation_max_constraints
    i = tid - world * propagation_max_constraints
    m = propagation_constraint_count[world]
    if m > propagation_max_constraints:
        m = propagation_max_constraints
    if i >= m:
        return

    ba = propagation_body_a[world, i]
    if ba >= 0:
        elig_a = int(0)
        art_a = body_to_articulation[ba]
        if art_a >= 0:
            elig_a = propagation_cache_art_eligible[art_a]
        if elig_a == want_eligible:
            old = wp.atomic_add(propagation_body_seen, ba, 1)
            if old == 0:
                slot = wp.atomic_add(propagation_body_count, world, 1)
                if slot < max_propagation_bodies:
                    propagation_body_list[world, slot] = ba
                    propagation_body_local_slot[ba] = slot

    bb = propagation_body_b[world, i]
    if bb >= 0:
        elig_b = int(0)
        art_b = body_to_articulation[bb]
        if art_b >= 0:
            elig_b = propagation_cache_art_eligible[art_b]
        if elig_b == want_eligible:
            old = wp.atomic_add(propagation_body_seen, bb, 1)
            if old == 0:
                slot = wp.atomic_add(propagation_body_count, world, 1)
                if slot < max_propagation_bodies:
                    propagation_body_list[world, slot] = bb
                    propagation_body_local_slot[bb] = slot


@wp.kernel
def compute_propagation_cache_world_flag(
    propagation_cache_body_count: wp.array[int],
    cache_max_bodies: int,
    # outputs
    propagation_cache_world_flag: wp.array[int],
):
    """Mark worlds whose cache-ELIGIBLE active bodies fit the response cache.

    The count input is the length of the eligible slot prefix written by the
    partitioned body-map build — free-rigid clutter and non-cacheable
    articulations never take the GEMV path, so they must not evict the robot
    from the cache. Worlds whose eligible count exceeds the capacity keep the
    exact per-iteration tree-walk fallback (flag 0); worlds under the cap
    take the cached-response GEMV path (flag 1). A zero-body world is
    trivially "cached": both paths are no-ops there.
    """
    world = wp.tid()
    flag = int(0)
    if propagation_cache_body_count[world] <= cache_max_bodies:
        flag = int(1)
    propagation_cache_world_flag[world] = flag


@wp.kernel
def snapshot_propagation_cache_qd_base(
    propagation_cache_world_flag: wp.array[int],
    propagation_body_count: wp.array[int],
    propagation_body_list: wp.array2d[int],
    cache_max_bodies: int,
    propagation_body_qd: wp.array2d[float],
    # outputs
    propagation_cache_qd_base: wp.array3d[float],
):
    """Snapshot active bodies' live COM twists before a propagation GS sweep.

    The sweep updates ``propagation_body_qd`` in place with diagonal-response
    estimates; the cached-response GEMV afterwards must rebuild the exact
    velocities as (pre-sweep twist) + (response x accumulated impulses), so
    the consistent pre-sweep value is captured here. Pre-sweep consistency
    holds because either the forced refresh just recomputed body_qd from
    v_out, or v_out is unchanged since the previous exact update.
    """
    tid = wp.tid()
    world = tid // cache_max_bodies
    slot = tid - world * cache_max_bodies
    if propagation_cache_world_flag[world] == 0:
        return
    n = propagation_body_count[world]
    if n > cache_max_bodies:
        n = cache_max_bodies
    if slot >= n:
        return
    body = propagation_body_list[world, slot]
    if body < 0:
        return
    for r in range(6):
        propagation_cache_qd_base[world, slot, r] = propagation_body_qd[body, r]


@wp.kernel
def compute_propagation_body_com_rel(
    body_to_articulation: wp.array[int],
    body_q: wp.array[wp.transform],
    body_com: wp.array[wp.vec3],
    articulation_origin: wp.array[wp.vec3],
    # outputs
    propagation_body_com_rel: wp.array2d[float],
):
    """Store each articulated body's center of mass relative to its articulation origin."""
    body = wp.tid()
    art = body_to_articulation[body]
    if art < 0:
        return
    rel = wp.transform_point(body_q[body], body_com[body]) - articulation_origin[art]
    for k in range(3):
        propagation_body_com_rel[body, k] = rel[k]


@wp.kernel
def flatten_propagation_joint_S(
    joint_child: wp.array[int],
    joint_qd_start: wp.array[int],
    joint_S_s: wp.array[wp.spatial_vector],
    propagation_body_com_rel: wp.array2d[float],
    # outputs
    propagation_joint_S_flat: wp.array2d[float],
):
    """Re-reference each joint motion subspace from the articulation origin to the child's center of mass."""
    joint = wp.tid()
    child = joint_child[joint]
    if child < 0:
        return
    child_rel = wp.vec3(
        propagation_body_com_rel[child, 0], propagation_body_com_rel[child, 1], propagation_body_com_rel[child, 2]
    )
    for dof in range(joint_qd_start[joint], joint_qd_start[joint + 1]):
        S = joint_S_s[dof]
        ang = wp.spatial_bottom(S)
        lin_child = wp.spatial_top(S) + wp.cross(ang, child_rel)
        for k in range(3):
            propagation_joint_S_flat[dof, k] = lin_child[k]
            propagation_joint_S_flat[dof, 3 + k] = ang[k]


@wp.kernel
def factor_propagation_tree_for_size(
    group_to_art: wp.array[int],
    articulation_start: wp.array[int],
    joint_parent: wp.array[int],
    joint_child: wp.array[int],
    joint_qd_start: wp.array[int],
    joint_dof_dim: wp.array2d[int],
    propagation_joint_S_flat: wp.array2d[float],
    joint_armature: wp.array[float],
    max_dofs: int,
    aug_row_counts: wp.array[int],
    aug_row_dof_index: wp.array[int],
    aug_row_K: wp.array[float],
    body_I_m: wp.array[wp.spatial_matrix],
    body_q_com: wp.array[wp.transform],
    propagation_body_com_rel: wp.array2d[float],
    # outputs
    propagation_tree_Ia: wp.array3d[float],
    propagation_tree_U: wp.array2d[float],
    propagation_tree_D_chol: wp.array3d[float],
    propagation_tree_D_inv: wp.array3d[float],
):
    """Factor the articulations of one size group into articulated-body inertia terms.

    One thread per articulation, for any joint DOF count. Stores the articulated
    inertia ``I_a`` of each link, ``U = I_a S`` and ``D^-1 = (S^T U + armature + K)^-1``
    of each inbound joint, where ``K`` is the implicit drive term folded into the mass
    matrix. ``joint_armature`` is the effective armature, so kinematic DOFs do not
    respond.
    """
    group_idx = wp.tid()
    art = group_to_art[group_idx]
    joint_start = articulation_start[art]
    joint_end = articulation_start[art + 1]

    for joint in range(joint_start, joint_end):
        body = joint_child[joint]
        X_com_world = wp.transform(wp.vec3(), wp.transform_get_rotation(body_q_com[body]))
        I = transform_spatial_inertia(X_com_world, body_I_m[body])
        for r in range(6):
            for c in range(6):
                propagation_tree_Ia[body, r, c] = I[r, c]
        for dof in range(joint_qd_start[joint], joint_qd_start[joint + 1]):
            for r in range(6):
                propagation_tree_U[dof, r] = 0.0
        for r in range(6):
            for c in range(6):
                propagation_tree_D_chol[joint, r, c] = 0.0
                propagation_tree_D_inv[joint, r, c] = 0.0

    for offset in range(joint_end - joint_start):
        joint = joint_end - 1 - offset
        child = joint_child[joint]
        parent = joint_parent[joint]
        dof_start = joint_qd_start[joint]
        dof_count = joint_dof_dim[joint, 0] + joint_dof_dim[joint, 1]

        for a in range(dof_count):
            gdof = dof_start + a
            for r in range(6):
                value = float(0.0)
                for c in range(6):
                    value += propagation_tree_Ia[child, r, c] * propagation_joint_S_flat[gdof, c]
                propagation_tree_U[gdof, r] = value

        for a in range(dof_count):
            gdof_a = dof_start + a
            for b in range(dof_count):
                gdof_b = dof_start + b
                value = float(0.0)
                for r in range(6):
                    value += propagation_joint_S_flat[gdof_a, r] * propagation_tree_U[gdof_b, r]
                if a == b:
                    value += joint_armature[gdof_a]
                    for aug_i in range(aug_row_counts[art]):
                        row_index = art * max_dofs + aug_i
                        if aug_row_dof_index[row_index] == gdof_a:
                            K = aug_row_K[row_index]
                            if K > 0.0:
                                value += K
                propagation_tree_D_chol[joint, a, b] = value

        # Cholesky factorization of the joint-space block D.
        for j in range(dof_count):
            s = propagation_tree_D_chol[joint, j, j]
            for k in range(j):
                chol_jk = propagation_tree_D_chol[joint, j, k]
                s -= chol_jk * chol_jk
            if s <= 1.0e-12:
                s = 1.0e-12
            s = wp.sqrt(s)
            propagation_tree_D_chol[joint, j, j] = s
            inv_s = 1.0 / s
            for i in range(j + 1, dof_count):
                v = propagation_tree_D_chol[joint, i, j]
                for k in range(j):
                    v -= propagation_tree_D_chol[joint, i, k] * propagation_tree_D_chol[joint, j, k]
                propagation_tree_D_chol[joint, i, j] = v * inv_s

        # Invert D one column at a time with forward and backward solves.
        for col in range(dof_count):
            for i in range(dof_count):
                v = float(0.0)
                if i == col:
                    v = 1.0
                for k in range(i):
                    v -= propagation_tree_D_chol[joint, i, k] * propagation_tree_D_inv[joint, k, col]
                propagation_tree_D_inv[joint, i, col] = v / propagation_tree_D_chol[joint, i, i]
            for i_rev in range(dof_count):
                i = dof_count - 1 - i_rev
                v = propagation_tree_D_inv[joint, i, col]
                for k in range(i + 1, dof_count):
                    v -= propagation_tree_D_chol[joint, k, i] * propagation_tree_D_inv[joint, k, col]
                propagation_tree_D_inv[joint, i, col] = v / propagation_tree_D_chol[joint, i, i]

        # Reduce the child's articulated inertia across this joint.
        for r in range(6):
            for c in range(6):
                reduced = propagation_tree_Ia[child, r, c]
                for a in range(dof_count):
                    U_ar = propagation_tree_U[dof_start + a, r]
                    for b in range(dof_count):
                        reduced -= U_ar * propagation_tree_D_inv[joint, a, b] * propagation_tree_U[dof_start + b, c]
                propagation_tree_Ia[child, r, c] = reduced

        if parent >= 0:
            # Add the reduced child inertia to the parent, referenced at the parent's COM.
            edge = _com_edge(propagation_body_com_rel, child, parent)
            I_child = wp.spatial_matrix()
            for r in range(6):
                for c in range(6):
                    I_child[r, c] = propagation_tree_Ia[child, r, c]
            for c in range(6):
                v_child = translate_twist_between_parallel_frames(_unit_spatial_vector(c), edge)
                w_parent = translate_wrench_between_parallel_frames(I_child * v_child, edge)
                for r in range(6):
                    propagation_tree_Ia[parent, r, c] = propagation_tree_Ia[parent, r, c] + w_parent[r]


@wp.func
def _propagation_tree_backward(
    joint_start: int,
    joint_end: int,
    joint_parent: wp.array[int],
    joint_child: wp.array[int],
    joint_qd_start: wp.array[int],
    joint_dof_dim: wp.array2d[int],
    propagation_joint_S_flat: wp.array2d[float],
    propagation_body_com_rel: wp.array2d[float],
    propagation_tree_U: wp.array2d[float],
    propagation_tree_D_inv: wp.array3d[float],
    propagation_tree_pA: wp.array2d[float],
    propagation_tree_u: wp.array[float],
):
    """Leaf-to-root pass: articulated bias forces and joint terms ``u`` of the impulses in ``pA``."""
    for offset in range(joint_end - joint_start):
        joint = joint_end - 1 - offset
        child = joint_child[joint]
        parent = joint_parent[joint]
        dof_start = joint_qd_start[joint]
        dof_count = joint_dof_dim[joint, 0] + joint_dof_dim[joint, 1]

        for a in range(dof_count):
            gdof = dof_start + a
            v = float(0.0)
            for r in range(6):
                v -= propagation_joint_S_flat[gdof, r] * propagation_tree_pA[child, r]
            propagation_tree_u[gdof] = v

        if parent >= 0:
            propagated_child = _spatial_row(propagation_tree_pA, child)
            for a in range(dof_count):
                coeff = float(0.0)
                for b in range(dof_count):
                    coeff += propagation_tree_D_inv[joint, a, b] * propagation_tree_u[dof_start + b]
                propagated_child += _spatial_row(propagation_tree_U, dof_start + a) * coeff
            propagated_parent = translate_wrench_between_parallel_frames(
                propagated_child, _com_edge(propagation_body_com_rel, child, parent)
            )
            for r in range(6):
                propagation_tree_pA[parent, r] = propagation_tree_pA[parent, r] + propagated_parent[r]


@wp.func
def _propagation_joint_qdd(
    joint: int,
    child: int,
    parent: int,
    dof_start: int,
    dof_count: int,
    propagation_body_com_rel: wp.array2d[float],
    propagation_tree_U: wp.array2d[float],
    propagation_tree_D_inv: wp.array3d[float],
    propagation_tree_u: wp.array[float],
    propagation_tree_body_delta: wp.array2d[float],
    propagation_tree_qdd: wp.array[float],
):
    """Root-to-leaf step of one joint: store its ``qdd`` and return the parent's twist change at the child."""
    parent_delta_child = wp.spatial_vector()
    if parent >= 0:
        parent_delta_child = translate_twist_between_parallel_frames(
            _spatial_row(propagation_tree_body_delta, parent), _com_edge(propagation_body_com_rel, child, parent)
        )
    for a in range(dof_count):
        qdd = float(0.0)
        for b in range(dof_count):
            gdof_b = dof_start + b
            parent_term = float(0.0)
            if parent >= 0:
                parent_term = wp.dot(_spatial_row(propagation_tree_U, gdof_b), parent_delta_child)
            qdd += propagation_tree_D_inv[joint, a, b] * (propagation_tree_u[gdof_b] - parent_term)
        propagation_tree_qdd[dof_start + a] = qdd
    return parent_delta_child


@wp.kernel
def refine_same_articulation_propagation_rows(
    is_free_rigid: wp.array[int],
    art_to_world: wp.array[int],
    body_to_articulation: wp.array[int],
    articulation_start: wp.array[int],
    joint_parent: wp.array[int],
    joint_child: wp.array[int],
    joint_qd_start: wp.array[int],
    joint_dof_dim: wp.array2d[int],
    propagation_joint_S_flat: wp.array2d[float],
    propagation_body_com_rel: wp.array2d[float],
    propagation_tree_U: wp.array2d[float],
    propagation_tree_D_inv: wp.array3d[float],
    propagation_constraint_count: wp.array[int],
    propagation_body_a: wp.array2d[int],
    propagation_body_b: wp.array2d[int],
    propagation_J_a: wp.array3d[float],
    propagation_J_b: wp.array3d[float],
    pgs_cfm: float,
    contact_w: float,
    propagation_max_constraints: int,
    # scratch
    propagation_tree_pA: wp.array2d[float],
    propagation_tree_u: wp.array[float],
    propagation_tree_qdd: wp.array[float],
    propagation_tree_body_delta: wp.array2d[float],
    # outputs
    propagation_eff_mass_inv: wp.array2d[float],
    propagation_MiJt_a: wp.array3d[float],
    propagation_MiJt_b: wp.array3d[float],
    propagation_row_w: wp.array2d[float],
):
    """Exact response of rows whose two bodies are links of one articulation.

    The per-link responses miss the cross term ``J_a (X_a H^-1 X_b^T) J_b^T``. The
    row's combined test impulse (``J_a`` at body a, ``J_b`` at body b) is propagated
    once through the articulated-body factorization and the resulting link velocity
    changes replace ``M^-1 J^T`` of both sides, so the effective mass includes the cross
    term. One thread per articulation, serial over its same-articulation rows.
    """
    art = wp.tid()
    if is_free_rigid[art] != 0:
        return
    world = art_to_world[art]
    m_count = wp.min(propagation_constraint_count[world], propagation_max_constraints)
    joint_start = articulation_start[art]
    joint_end = articulation_start[art + 1]

    for i in range(m_count):
        ba = propagation_body_a[world, i]
        bb = propagation_body_b[world, i]
        if ba < 0 or bb < 0:
            continue
        if body_to_articulation[ba] != art or body_to_articulation[bb] != art:
            continue

        for joint in range(joint_start, joint_end):
            body = joint_child[joint]
            for r in range(6):
                propagation_tree_pA[body, r] = 0.0
                propagation_tree_body_delta[body, r] = 0.0
            for dof in range(joint_qd_start[joint], joint_qd_start[joint + 1]):
                propagation_tree_u[dof] = 0.0
                propagation_tree_qdd[dof] = 0.0
        for r in range(6):
            propagation_tree_pA[ba, r] = propagation_tree_pA[ba, r] - propagation_J_a[world, i, r]
            propagation_tree_pA[bb, r] = propagation_tree_pA[bb, r] - propagation_J_b[world, i, r]

        _propagation_tree_backward(
            joint_start,
            joint_end,
            joint_parent,
            joint_child,
            joint_qd_start,
            joint_dof_dim,
            propagation_joint_S_flat,
            propagation_body_com_rel,
            propagation_tree_U,
            propagation_tree_D_inv,
            propagation_tree_pA,
            propagation_tree_u,
        )
        for joint in range(joint_start, joint_end):
            child = joint_child[joint]
            dof_start = joint_qd_start[joint]
            dof_count = joint_dof_dim[joint, 0] + joint_dof_dim[joint, 1]
            value = _propagation_joint_qdd(
                joint,
                child,
                joint_parent[joint],
                dof_start,
                dof_count,
                propagation_body_com_rel,
                propagation_tree_U,
                propagation_tree_D_inv,
                propagation_tree_u,
                propagation_tree_body_delta,
                propagation_tree_qdd,
            )
            for a in range(dof_count):
                value += _spatial_row(propagation_joint_S_flat, dof_start + a) * propagation_tree_qdd[dof_start + a]
            for r in range(6):
                propagation_tree_body_delta[child, r] = value[r]

        d = pgs_cfm
        for r in range(6):
            mi_a = propagation_tree_body_delta[ba, r]
            mi_b = propagation_tree_body_delta[bb, r]
            propagation_MiJt_a[world, i, r] = mi_a
            propagation_MiJt_b[world, i, r] = mi_b
            d += propagation_J_a[world, i, r] * mi_a
            d += propagation_J_b[world, i, r] * mi_b
        if d > 0.0:
            propagation_eff_mass_inv[world, i] = 1.0 / d
        else:
            propagation_eff_mass_inv[world, i] = 0.0
        # The exact response is unsplit, so the row keeps the uniform regularization weight.
        if contact_w < 1.0 and propagation_row_w[world, i] < 1.0:
            propagation_row_w[world, i] = contact_w


@wp.kernel
def compute_propagation_tree_body_response_for_size(
    propagation_body_count: wp.array[int],
    propagation_body_list: wp.array2d[int],
    body_to_articulation: wp.array[int],
    body_to_joint: wp.array[int],
    group_to_art: wp.array[int],
    art_to_world: wp.array[int],
    articulation_start: wp.array[int],
    joint_parent: wp.array[int],
    joint_child: wp.array[int],
    joint_qd_start: wp.array[int],
    joint_dof_dim: wp.array2d[int],
    propagation_joint_S_flat: wp.array2d[float],
    max_propagation_bodies: int,
    propagation_body_com_rel: wp.array2d[float],
    propagation_tree_U: wp.array2d[float],
    propagation_tree_D_inv: wp.array3d[float],
    # scratch
    propagation_tree_pA: wp.array2d[float],
    propagation_tree_u: wp.array[float],
    propagation_tree_qdd: wp.array[float],
    propagation_tree_body_delta: wp.array2d[float],
    # outputs
    propagation_body_response: wp.array3d[float],
):
    """Compute the 6x6 center-of-mass response of each contact-touched link by tree solves.

    For any joint DOF count. A unit wrench at a link only couples to the joints on its
    path to the root, so each of the six basis solves walks that path only. The walk is
    capped at the articulation's joint count.
    """
    group_idx = wp.tid()
    art = group_to_art[group_idx]
    world = art_to_world[art]
    joint_start = articulation_start[art]
    joint_end = articulation_start[art + 1]

    for local_body in range(wp.min(propagation_body_count[world], max_propagation_bodies)):
        target_body = propagation_body_list[world, local_body]
        if target_body < 0 or body_to_articulation[target_body] != art:
            continue

        path_len = int(0)
        walk = body_to_joint[target_body]
        for _cap in range(joint_end - joint_start):
            if walk < 0:
                break
            path_len += 1
            walk_parent = joint_parent[walk]
            if walk_parent >= 0:
                walk = body_to_joint[walk_parent]
            else:
                walk = int(-1)

        for basis in range(6):
            walk = body_to_joint[target_body]
            for _k in range(path_len):
                joint = walk
                walk_parent = joint_parent[joint]
                if walk_parent >= 0:
                    walk = body_to_joint[walk_parent]
                else:
                    walk = int(-1)
                body = joint_child[joint]
                for r in range(6):
                    propagation_tree_pA[body, r] = 0.0
                    propagation_tree_body_delta[body, r] = 0.0
                for dof in range(joint_qd_start[joint], joint_qd_start[joint + 1]):
                    propagation_tree_u[dof] = 0.0
                    propagation_tree_qdd[dof] = 0.0
            for r in range(6):
                propagation_tree_pA[target_body, r] = 0.0
            propagation_tree_pA[target_body, basis] = -1.0

            # Backward sweep, target to root along the path.
            walk = body_to_joint[target_body]
            for _k in range(path_len):
                joint = walk
                walk_parent = joint_parent[joint]
                if walk_parent >= 0:
                    walk = body_to_joint[walk_parent]
                else:
                    walk = int(-1)
                child = joint_child[joint]
                parent = joint_parent[joint]
                dof_start = joint_qd_start[joint]
                dof_count = joint_dof_dim[joint, 0] + joint_dof_dim[joint, 1]
                for a in range(dof_count):
                    gdof = dof_start + a
                    v = float(0.0)
                    for r in range(6):
                        v -= propagation_joint_S_flat[gdof, r] * propagation_tree_pA[child, r]
                    propagation_tree_u[gdof] = v
                if parent >= 0:
                    propagated_child = _spatial_row(propagation_tree_pA, child)
                    for a in range(dof_count):
                        coeff = float(0.0)
                        for b in range(dof_count):
                            coeff += propagation_tree_D_inv[joint, a, b] * propagation_tree_u[dof_start + b]
                        propagated_child += _spatial_row(propagation_tree_U, dof_start + a) * coeff
                    propagated_parent = translate_wrench_between_parallel_frames(
                        propagated_child, _com_edge(propagation_body_com_rel, child, parent)
                    )
                    for r in range(6):
                        propagation_tree_pA[parent, r] = propagation_tree_pA[parent, r] + propagated_parent[r]

            # Forward sweep, root to target: level i visits the (path_len - 1 - i)-th ancestor.
            for level in range(path_len):
                joint = body_to_joint[target_body]
                for _s in range(path_len - 1 - level):
                    joint = body_to_joint[joint_parent[joint]]
                child = joint_child[joint]
                dof_start = joint_qd_start[joint]
                dof_count = joint_dof_dim[joint, 0] + joint_dof_dim[joint, 1]
                value = _propagation_joint_qdd(
                    joint,
                    child,
                    joint_parent[joint],
                    dof_start,
                    dof_count,
                    propagation_body_com_rel,
                    propagation_tree_U,
                    propagation_tree_D_inv,
                    propagation_tree_u,
                    propagation_tree_body_delta,
                    propagation_tree_qdd,
                )
                for a in range(dof_count):
                    value += _spatial_row(propagation_joint_S_flat, dof_start + a) * propagation_tree_qdd[dof_start + a]
                for r in range(6):
                    propagation_tree_body_delta[child, r] = value[r]

            for r in range(6):
                propagation_body_response[target_body, r, basis] = propagation_tree_body_delta[target_body, r]


@wp.kernel
def refresh_propagation_tree_body_qd_for_size(
    group_to_art: wp.array[int],
    art_to_world: wp.array[int],
    dense_contact_world_flag: wp.array[int],
    force_refresh: int,
    articulation_start: wp.array[int],
    joint_parent: wp.array[int],
    joint_child: wp.array[int],
    joint_qd_start: wp.array[int],
    propagation_joint_S_flat: wp.array2d[float],
    propagation_body_com_rel: wp.array2d[float],
    v_out: wp.array[float],
    # outputs
    propagation_body_qd: wp.array2d[float],
):
    """Recompute the center-of-mass twists of an articulation's links from ``v_out``.

    One root-to-leaf pass. Without ``force_refresh`` the pass is skipped for worlds
    without dense contact rows, whose generalized velocities only change through the
    propagation solve (which leaves the twists consistent).
    """
    group_idx = wp.tid()
    art = group_to_art[group_idx]
    if force_refresh == 0:
        world = art_to_world[art]
        if world >= 0 and dense_contact_world_flag[world] == 0:
            return

    for joint in range(articulation_start[art], articulation_start[art + 1]):
        child = joint_child[joint]
        parent = joint_parent[joint]
        value = wp.spatial_vector()
        if parent >= 0:
            value = translate_twist_between_parallel_frames(
                _spatial_row(propagation_body_qd, parent), _com_edge(propagation_body_com_rel, child, parent)
            )
        for dof in range(joint_qd_start[joint], joint_qd_start[joint + 1]):
            value += _spatial_row(propagation_joint_S_flat, dof) * v_out[dof]
        for r in range(6):
            propagation_body_qd[child, r] = value[r]


@wp.kernel
def flush_propagation_free_body_qd_to_vout(
    free_rigid_body_indices: wp.array[int],
    body_to_articulation: wp.array[int],
    articulation_dof_start: wp.array[int],
    # in/out
    propagation_body_qd: wp.array2d[float],
    propagation_body_impulses: wp.array2d[float],
    v_out: wp.array[float],
):
    """Write the solved free-body twists back to ``v_out`` and clear their impulses.

    A free body's generalized velocity and its propagation twist are both referenced at
    its center of mass, so the write-back is a copy.
    """
    body = free_rigid_body_indices[wp.tid()]
    dof_start = articulation_dof_start[body_to_articulation[body]]
    for r in range(6):
        v_out[dof_start + r] = propagation_body_qd[body, r]
        propagation_body_impulses[body, r] = 0.0


@wp.kernel
def refresh_propagation_free_body_qd_from_vout(
    free_rigid_body_indices: wp.array[int],
    body_to_articulation: wp.array[int],
    articulation_dof_start: wp.array[int],
    v_out: wp.array[float],
    # outputs
    propagation_body_qd: wp.array2d[float],
):
    """Copy the free-body generalized velocities in ``v_out`` to their propagation twists."""
    body = free_rigid_body_indices[wp.tid()]
    dof_start = articulation_dof_start[body_to_articulation[body]]
    for r in range(6):
        propagation_body_qd[body, r] = v_out[dof_start + r]


@wp.kernel
def propagate_tree_impulses_for_size(
    group_to_art: wp.array[int],
    articulation_start: wp.array[int],
    joint_parent: wp.array[int],
    joint_child: wp.array[int],
    joint_qd_start: wp.array[int],
    joint_dof_dim: wp.array2d[int],
    propagation_joint_S_flat: wp.array2d[float],
    propagation_body_com_rel: wp.array2d[float],
    propagation_tree_U: wp.array2d[float],
    propagation_tree_D_inv: wp.array3d[float],
    # scratch
    propagation_tree_pA: wp.array2d[float],
    propagation_tree_u: wp.array[float],
    propagation_tree_qdd: wp.array[float],
    propagation_tree_body_delta: wp.array2d[float],
    # in/out
    propagation_body_impulses: wp.array2d[float],
    propagation_body_qd: wp.array2d[float],
    v_out: wp.array[float],
):
    """Apply an articulation's accumulated link impulses to its joint velocities.

    For any joint DOF count. Runs the articulated-body backward and forward passes on
    the impulses the row solve accumulated, adds the joint velocity change to ``v_out``,
    recomputes the links' twists from ``v_out`` (replacing the row solve's per-link
    estimates) and clears the accumulated impulses. Articulations without impulses are
    already consistent and are skipped.
    """
    group_idx = wp.tid()
    art = group_to_art[group_idx]
    joint_start = articulation_start[art]
    joint_end = articulation_start[art + 1]

    has_impulse = int(0)
    for joint in range(joint_start, joint_end):
        body = joint_child[joint]
        for r in range(6):
            impulse = propagation_body_impulses[body, r]
            if impulse != 0.0:
                has_impulse = int(1)
            propagation_tree_pA[body, r] = -impulse
            propagation_tree_body_delta[body, r] = 0.0
        for dof in range(joint_qd_start[joint], joint_qd_start[joint + 1]):
            propagation_tree_u[dof] = 0.0
            propagation_tree_qdd[dof] = 0.0
    if has_impulse == 0:
        return

    _propagation_tree_backward(
        joint_start,
        joint_end,
        joint_parent,
        joint_child,
        joint_qd_start,
        joint_dof_dim,
        propagation_joint_S_flat,
        propagation_body_com_rel,
        propagation_tree_U,
        propagation_tree_D_inv,
        propagation_tree_pA,
        propagation_tree_u,
    )
    for joint in range(joint_start, joint_end):
        child = joint_child[joint]
        dof_start = joint_qd_start[joint]
        dof_count = joint_dof_dim[joint, 0] + joint_dof_dim[joint, 1]
        value = _propagation_joint_qdd(
            joint,
            child,
            joint_parent[joint],
            dof_start,
            dof_count,
            propagation_body_com_rel,
            propagation_tree_U,
            propagation_tree_D_inv,
            propagation_tree_u,
            propagation_tree_body_delta,
            propagation_tree_qdd,
        )
        for a in range(dof_count):
            gdof = dof_start + a
            qdd = propagation_tree_qdd[gdof]
            v_out[gdof] = v_out[gdof] + qdd
            value += _spatial_row(propagation_joint_S_flat, gdof) * qdd
        for r in range(6):
            propagation_tree_body_delta[child, r] = value[r]

    for joint in range(joint_start, joint_end):
        child = joint_child[joint]
        parent = joint_parent[joint]
        value = wp.spatial_vector()
        if parent >= 0:
            value = translate_twist_between_parallel_frames(
                _spatial_row(propagation_body_qd, parent), _com_edge(propagation_body_com_rel, child, parent)
            )
        for dof in range(joint_qd_start[joint], joint_qd_start[joint + 1]):
            value += _spatial_row(propagation_joint_S_flat, dof) * v_out[dof]
        for r in range(6):
            propagation_body_qd[child, r] = value[r]
            propagation_body_impulses[child, r] = 0.0


# Slots of the row high-water array, per row family (dense, free-body, propagation) and the contact count.
ROW_WATERMARK_FAMILY_STRIDE = 5
ROW_WATERMARK_CONTACT_SLOT = 3 * ROW_WATERMARK_FAMILY_STRIDE


@wp.kernel
def accumulate_row_watermarks(
    constraint_count: wp.array[wp.int32],
    slot_counter: wp.array[wp.int32],
    dropped_contact_rows: wp.array[wp.int32],
    capacity: int,
    base: int,
    # outputs
    watermarks: wp.array[wp.int32],
):
    """Accumulate one row family's high-water marks without changing solver state.

    Slots from ``base``: retained rows, requested rows (accepted plus rejected reservations),
    dropped contact rows, excess over ``capacity`` (maxima) and overflowing world-steps (a sum).
    """
    world = wp.tid()
    requested = slot_counter[world]
    wp.atomic_max(watermarks, base, constraint_count[world])
    wp.atomic_max(watermarks, base + 1, requested)
    wp.atomic_max(watermarks, base + 2, dropped_contact_rows[world])
    if requested > capacity:
        wp.atomic_max(watermarks, base + 3, requested - capacity)
        wp.atomic_add(watermarks, base + 4, 1)


@wp.kernel
def accumulate_contact_watermark(
    contact_count: wp.array[wp.int32],
    # outputs
    watermarks: wp.array[wp.int32],
):
    """Accumulate the high-water mark of the rigid contact count."""
    wp.atomic_max(watermarks, ROW_WATERMARK_CONTACT_SLOT, contact_count[0])


@wp.kernel
def compute_dense_contact_bounds(
    world_constraint_count: wp.array[int],
    world_row_type: wp.array2d[int],
    # outputs
    dense_contact_bounds: wp.array2d[int],
):
    """Write each world's first contact or friction row, where its internal-row prefix ends, twice.

    The two entries end the drive and position-limit prefix and the velocity-limit segment; the
    paired factor-coordinate solve has no velocity-limit rows, so both name the contact start.
    """
    world = wp.tid()
    count = wp.min(world_constraint_count[world], world_row_type.shape[1])
    start = count
    for i in range(count):
        row_type = world_row_type[world, i]
        if row_type == PGS_CONSTRAINT_TYPE_CONTACT or row_type == PGS_CONSTRAINT_TYPE_FRICTION:
            start = i
            break
    dense_contact_bounds[world, 0] = start
    dense_contact_bounds[world, 1] = start


# ---------------------------------------------------------------------------
# Sparse-diagonal contact response
#
# Worlds made of one articulation with an uncoupled (diagonal) mass matrix and one small
# dense articulation solve their dense rows over the dense articulation's DOFs plus at most
# two diagonal coordinates per row. The diagonal articulation's inverse mass is stored per
# DOF, its contact rows store only the two touched coordinates, and its position limits are
# projected per DOF inside the solve instead of occupying dense rows.
# ---------------------------------------------------------------------------


@wp.kernel(module=_MASS_DYNAMICS_KERNEL_MODULE)
def compute_compact_diagonal_inverse_mass(
    articulation_start: wp.array[int],
    articulation_dof_start: wp.array[int],
    mass_update_mask: wp.array[int],
    joint_child: wp.array[int],
    joint_S_s: wp.array[wp.spatial_vector],
    body_I_c: wp.array[wp.spatial_matrix],
    group_to_art: wp.array[int],
    dof_joint_offset: wp.array[int],
    armature: wp.array2d[float],
    drive_dof_K: wp.array[float],
    n_dofs: int,
    # output
    diagonal_inverse_mass: wp.array[float],
):
    """Store the inverse diagonal mass of independent articulation branches, drive stiffness included."""
    element = wp.tid()
    group = element // n_dofs
    dof = element - group * n_dofs
    art = group_to_art[group]
    if mass_update_mask[art] == 0:
        return
    global_dof = articulation_dof_start[art] + dof
    joint = articulation_start[art] + dof_joint_offset[dof]
    motion = joint_S_s[global_dof]
    mass = wp.dot(motion, body_I_c[joint_child[joint]] * motion) + armature[group, dof]
    stiffness = drive_dof_K[global_dof]
    if stiffness > 0.0:
        mass += stiffness
    diagonal_inverse_mass[global_dof] = 1.0 / mass


@wp.kernel(module=_MASS_DYNAMICS_KERNEL_MODULE)
def solve_compact_diagonal_mass(
    diagonal_inverse_mass: wp.array[float],
    group_to_art: wp.array[int],
    articulation_dof_start: wp.array[int],
    n_dofs: int,
    joint_tau: wp.array[float],
    articulation_active: wp.array[int],
    # output
    joint_qdd: wp.array[float],
):
    """Apply a stored inverse diagonal mass to the generalized forces."""
    element = wp.tid()
    group = element // n_dofs
    dof = element - group * n_dofs
    art = group_to_art[group]
    if articulation_active[art] == 0:
        return
    global_dof = articulation_dof_start[art] + dof
    joint_qdd[global_dof] = joint_tau[global_dof] * diagonal_inverse_mass[global_dof]


@wp.kernel
def prepare_fused_diagonal_joint_limits(
    world_dof_indices: wp.array2d[int],
    max_world_dofs: int,
    fused_limit_dof_mask: wp.array[int],
    limit_q_index: wp.array[int],
    joint_limit_lower: wp.array[float],
    joint_limit_upper: wp.array[float],
    joint_q: wp.array[float],
    activation_gap: float,
    pgs_beta: float,
    dt: float,
    # outputs
    active_sides: wp.array2d[int],
    lower_rhs: wp.array2d[float],
    upper_rhs: wp.array2d[float],
):
    """Prepare the per-coordinate position-limit projections of the diagonal articulation."""
    element = wp.tid()
    world = element // max_world_dofs
    local_dof = element - world * max_world_dofs
    global_dof = world_dof_indices[world, local_dof]

    active = int(0)
    rhs_lower = float(0.0)
    rhs_upper = float(0.0)
    if global_dof >= 0 and fused_limit_dof_mask[global_dof] != 0:
        q_index = limit_q_index[global_dof]
        if q_index >= 0:
            q = joint_q[q_index]
            lower = joint_limit_lower[global_dof]
            upper = joint_limit_upper[global_dof]
            inv_dt = 1.0 / dt
            if wp.isfinite(lower) and q <= lower + activation_gap:
                phi = q - lower
                scale = 1.0
                if phi < 0.0:
                    scale = pgs_beta
                rhs_lower = scale * phi * inv_dt
                active |= 1
            if wp.isfinite(upper) and q >= upper - activation_gap:
                phi = upper - q
                scale = 1.0
                if phi < 0.0:
                    scale = pgs_beta
                rhs_upper = scale * phi * inv_dt
                active |= 2

    active_sides[world, local_dof] = active
    lower_rhs[world, local_dof] = rhs_lower
    upper_rhs[world, local_dof] = rhs_upper


@wp.kernel
def snapshot_dense_contact_row_start(
    world_slot_counter: wp.array[int],
    max_constraints: int,
    # output
    dense_contact_row_start: wp.array[int],
):
    """Record where each world's dense contact rows begin: after its internal rows."""
    world = wp.tid()
    dense_contact_row_start[world] = wp.min(world_slot_counter[world], max_constraints)


@wp.kernel
def populate_sparse_diagonal_contact_response(
    contact_count: wp.array[int],
    total_num_workers: int,
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
    target_size: int,
    articulation_response_dof_count: wp.array[int],
    articulation_dof_start: wp.array[int],
    articulation_world_dof_offset: wp.array[int],
    articulation_origin: wp.array[wp.vec3],
    body_single_response_dof: wp.array[int],
    diagonal_inverse_mass: wp.array[float],
    joint_S_s: wp.array[wp.spatial_vector],
    shape_body: wp.array[int],
    body_q: wp.array[wp.transform],
    friction_patches: FrictionPatches,
    contact_shared_anchor: int,
    contact_friction_shared_anchor: int,
    # outputs
    sparse_row_dof: wp.array3d[int],
    sparse_row_jy: wp.array3d[float],
):
    """Store the two diagonal-articulation coordinates of each dense contact row with their ``J`` and ``Y``.

    Row points follow the dense builders (:func:`contact_row_points`), so the sparse entries see the same
    row geometry. Entries are ``[J_a, Y_a, J_b, Y_b]``; both sides on one coordinate are merged.
    """
    worker = wp.tid()
    total_contacts = wp.min(contact_count[0], contact_point0.shape[0])
    for c in range(worker, total_contacts, total_num_workers):
        slot = contact_slot[c]
        if contact_path[c] != 0 or slot < 0:
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
        point_a_world, point_b_world = _contact_points_world(
            c, body_a, body_b, normal, contact_point0, contact_point1, contact_thickness0, contact_thickness1, body_q
        )
        tangent0, tangent1 = contact_tangent_basis(normal)
        world = contact_world[c]
        art_a = contact_art_a[c]
        art_b = contact_art_b[c]
        rows = contact_slots_needed[c]
        for row in range(3):
            if row >= rows:
                continue
            direction = normal
            if row == 1:
                direction = tangent0
            elif row == 2:
                direction = tangent1
            point_a, point_b = contact_row_points(
                c,
                row,
                point_a_world,
                point_b_world,
                contact_shared_anchor,
                contact_friction_shared_anchor,
                friction_patches,
            )
            dof_a = int(-1)
            dof_b = int(-1)
            coord_a = int(-1)
            coord_b = int(-1)
            value_a = float(0.0)
            value_b = float(0.0)
            if art_a >= 0 and body_a >= 0 and articulation_response_dof_count[art_a] == target_size:
                dof_a = body_single_response_dof[body_a]
                if dof_a >= 0:
                    local_dof_a = dof_a - articulation_dof_start[art_a]
                    if local_dof_a >= 0 and local_dof_a < target_size:
                        motion_a = joint_S_s[dof_a]
                        linear_a = wp.vec3(motion_a[0], motion_a[1], motion_a[2])
                        angular_a = wp.vec3(motion_a[3], motion_a[4], motion_a[5])
                        value_a = wp.dot(
                            direction, linear_a + wp.cross(angular_a, point_a - articulation_origin[art_a])
                        )
                        coord_a = articulation_world_dof_offset[art_a] + local_dof_a
            if art_b >= 0 and body_b >= 0 and articulation_response_dof_count[art_b] == target_size:
                dof_b = body_single_response_dof[body_b]
                if dof_b >= 0:
                    local_dof_b = dof_b - articulation_dof_start[art_b]
                    if local_dof_b >= 0 and local_dof_b < target_size:
                        motion_b = joint_S_s[dof_b]
                        linear_b = wp.vec3(motion_b[0], motion_b[1], motion_b[2])
                        angular_b = wp.vec3(motion_b[3], motion_b[4], motion_b[5])
                        value_b = -wp.dot(
                            direction, linear_b + wp.cross(angular_b, point_b - articulation_origin[art_b])
                        )
                        coord_b = articulation_world_dof_offset[art_b] + local_dof_b
            response_a = float(0.0)
            response_b = float(0.0)
            if dof_a >= 0:
                response_a = value_a * diagonal_inverse_mass[dof_a]
            if dof_b >= 0:
                response_b = value_b * diagonal_inverse_mass[dof_b]
            if coord_a >= 0 and coord_a == coord_b:
                value_a += value_b
                response_a += response_b
                coord_b = -1
                value_b = 0.0
                response_b = 0.0
            output_row = slot + row
            sparse_row_dof[world, output_row, 0] = coord_a
            sparse_row_dof[world, output_row, 1] = coord_b
            sparse_row_jy[world, output_row, 0] = value_a
            sparse_row_jy[world, output_row, 1] = response_a
            sparse_row_jy[world, output_row, 2] = value_b
            sparse_row_jy[world, output_row, 3] = response_b


@wp.kernel
def accumulate_sparse_diagonal_response_diag(
    world_constraint_count: wp.array[int],
    max_constraints: int,
    sparse_row_dof: wp.array3d[int],
    sparse_row_jy: wp.array3d[float],
    # in/out
    world_diag: wp.array2d[float],
):
    """Add the two sparse response entries of each active row to its diagonal."""
    tid = wp.tid()
    world = tid // max_constraints
    row = tid - world * max_constraints
    if row >= world_constraint_count[world]:
        return
    value = float(0.0)
    if sparse_row_dof[world, row, 0] >= 0:
        value += sparse_row_jy[world, row, 0] * sparse_row_jy[world, row, 1]
    if sparse_row_dof[world, row, 1] >= 0:
        value += sparse_row_jy[world, row, 2] * sparse_row_jy[world, row, 3]
    world_diag[world, row] += value


@wp.kernel
def apply_sparse_diagonal_contact_restitution(
    world_constraint_count: wp.array[int],
    max_constraints: int,
    world_phi: wp.array2d[float],
    world_row_type: wp.array2d[int],
    world_target_velocity: wp.array2d[float],
    world_row_restitution: wp.array2d[float],
    world_incident_velocity: wp.array[float],
    world_dof_indices: wp.array2d[int],
    dense_offsets: wp.array[int],
    dense_groups: wp.array[int],
    dense_dofs: int,
    dense_J: wp.array3d[float],
    sparse_row_dof: wp.array3d[int],
    sparse_row_jy: wp.array3d[float],
    dt: float,
    restitution_velocity_threshold: float,
    # in/out
    world_rhs: wp.array2d[float],
):
    """Replace the bias of an impacting contact by its rebound target, from the dense and sparse coordinates.

    The counterpart of :func:`apply_world_contact_restitution` for sparse-diagonal worlds.
    """
    tid = wp.tid()
    world = tid // max_constraints
    row = tid - world * max_constraints
    if row >= world_constraint_count[world] or world_row_type[world, row] != PGS_CONSTRAINT_TYPE_CONTACT:
        return
    restitution = world_row_restitution[world, row]
    if restitution <= 0.0:
        return
    relative_incident = float(0.0)
    dense_offset = dense_offsets[world]
    dense_group = dense_groups[world]
    for local_dof in range(dense_dofs):
        global_dof = world_dof_indices[world, dense_offset + local_dof]
        if global_dof >= 0:
            relative_incident += dense_J[dense_group, row, local_dof] * world_incident_velocity[global_dof]
    for sparse_slot in range(2):
        world_dof = sparse_row_dof[world, row, sparse_slot]
        if world_dof >= 0:
            global_dof = world_dof_indices[world, world_dof]
            if global_dof >= 0:
                relative_incident += sparse_row_jy[world, row, sparse_slot * 2] * world_incident_velocity[global_dof]
    target_vel = world_target_velocity[world, row]
    relative_incident -= target_vel
    if contact_restitution_fires(world_phi[world, row], relative_incident, dt, restitution_velocity_threshold):
        world_rhs[world, row] = -target_vel + restitution * relative_incident


@wp.kernel
def hinv_jt_par_row_contact_fallback(
    L_group: wp.array3d[float],
    J_group: wp.array3d[float],
    group_to_art: wp.array[int],
    art_to_world: wp.array[int],
    articulation_world_dof_offset: wp.array[int],
    world_constraint_count: wp.array[int],
    local_solve_owner: wp.array[int],
    world_row_restitution: wp.array2d[float],
    n_dofs: int,
    n_arts: int,
    write_world: int,
    Y_group: wp.array3d[float],
    J_world: wp.array3d[float],
    Y_world: wp.array3d[float],
):
    """Compute ``Y = H^-1 J^T`` only for worlds that the general sweep solves, one warp per articulation."""
    tid = wp.tid()
    group_index = tid // 32
    lane = tid % 32
    if group_index >= n_arts:
        return

    art = group_to_art[group_index]
    world = art_to_world[art]
    constraint_count = world_constraint_count[world]
    if local_solve_owner[world] != PGS_LOCAL_SOLVE_OWNER_GENERAL:
        # Local owners build their response in their fused solve. Only impact rows need a
        # world Jacobian, for the restitution target pass.
        if write_world != 0:
            dof_offset = articulation_world_dof_offset[art]
            constraint = lane
            while constraint < constraint_count:
                if world_row_restitution[world, constraint] > 0.0:
                    for i in range(n_dofs):
                        J_world[world, constraint, dof_offset + i] = J_group[group_index, constraint, i]
                constraint += 32
        return

    constraint = lane
    while constraint < constraint_count:
        for i in range(n_dofs):
            value = J_group[group_index, constraint, i]
            for k in range(i):
                value -= L_group[group_index, i, k] * Y_group[group_index, constraint, k]

            diagonal = L_group[group_index, i, i]
            if diagonal != 0.0:
                Y_group[group_index, constraint, i] = value / diagonal
            else:
                Y_group[group_index, constraint, i] = 0.0

        for reverse in range(n_dofs):
            i = n_dofs - 1 - reverse
            value = Y_group[group_index, constraint, i]
            for k in range(i + 1, n_dofs):
                value -= L_group[group_index, k, i] * Y_group[group_index, constraint, k]

            diagonal = L_group[group_index, i, i]
            if diagonal != 0.0:
                Y_group[group_index, constraint, i] = value / diagonal
            else:
                Y_group[group_index, constraint, i] = 0.0

        if write_world != 0:
            dof_offset = articulation_world_dof_offset[art]
            for i in range(n_dofs):
                J_world[world, constraint, dof_offset + i] = J_group[group_index, constraint, i]
                Y_world[world, constraint, dof_offset + i] = Y_group[group_index, constraint, i]
        constraint += 32


@wp.kernel
def classify_local_solve_worlds(
    world_constraint_count: wp.array[int],
    world_row_type: wp.array2d[int],
    mf_constraint_count: wp.array[int],
    mf_body_a: wp.array2d[int],
    mf_body_b: wp.array2d[int],
    mf_row_type: wp.array2d[int],
    body_to_articulation: wp.array[int],
    articulation_dof_count: wp.array[int],
    local_primary_articulation: wp.array[int],
    local_pair_articulation: wp.array[int],
    local_residual_pair_articulation: wp.array[int],
    local_max_constraints: int,
    local_residual_max_constraints: int,
    local_residual_mf_max_constraints: int,
    # outputs
    local_solve_owner: wp.array[int],
    general_world_count: wp.array[int],
    general_worlds: wp.array[int],
):
    """Assign each world's solver owner and compact the worlds left to the general sweep.

    A world whose dense rows are all internal (no contact rows) and fit the articulation's
    DOF count goes to the single-articulation solve; a world whose articulation also touches
    its one free body goes to the pair solve; the residual pair solve also takes that free
    body's own contact rows. Every other world with rows stays with the general sweep.
    """
    world = wp.tid()
    row_count = world_constraint_count[world]
    mf_count = mf_constraint_count[world]
    primary_articulation = local_primary_articulation[world]
    pair_articulation = local_pair_articulation[world]
    residual_pair_articulation = local_residual_pair_articulation[world]
    # A world without contact rows solves internal rows only.
    contact_rows = int(0)
    for row in range(row_count):
        row_type = world_row_type[world, row]
        if row_type == PGS_CONSTRAINT_TYPE_CONTACT or row_type == PGS_CONSTRAINT_TYPE_FRICTION:
            contact_rows += 1
    single_phase = contact_rows == 0

    local_mf = mf_count > 0 and mf_count <= local_residual_mf_max_constraints and residual_pair_articulation >= 0
    mf_row = int(0)
    while mf_row < mf_count and local_mf:
        body_a = mf_body_a[world, mf_row]
        body_b = mf_body_b[world, mf_row]
        if body_a >= 0 and body_to_articulation[body_a] != residual_pair_articulation:
            local_mf = False
        if body_b >= 0 and body_to_articulation[body_b] != residual_pair_articulation:
            local_mf = False
        # The residual loop does not implement the free-body velocity-limit law.
        if mf_row_type[world, mf_row] == PGS_CONSTRAINT_TYPE_JOINT_VELOCITY_LIMIT:
            local_mf = False
        mf_row += 1

    owner = PGS_LOCAL_SOLVE_OWNER_GENERAL
    single_row_capacity = int(0)
    if primary_articulation >= 0:
        single_row_capacity = wp.min(local_max_constraints, articulation_dof_count[primary_articulation])
    if row_count > 0 and mf_count == 0:
        if single_phase and row_count <= single_row_capacity:
            owner = PGS_LOCAL_SOLVE_OWNER_SINGLE
        elif not single_phase and row_count <= local_max_constraints and pair_articulation >= 0:
            owner = PGS_LOCAL_SOLVE_OWNER_PAIR
    if owner == PGS_LOCAL_SOLVE_OWNER_GENERAL and (
        row_count > 0
        and row_count <= local_residual_max_constraints
        and residual_pair_articulation >= 0
        and ((mf_count == 0 and not single_phase) or local_mf)
    ):
        owner = PGS_LOCAL_SOLVE_OWNER_PAIR_RESIDUAL
    local_solve_owner[world] = owner
    if owner == PGS_LOCAL_SOLVE_OWNER_GENERAL and (row_count > 0 or mf_count > 0):
        general_index = wp.atomic_add(general_world_count, 0, 1)
        general_worlds[general_index] = world


@wp.kernel
def compact_local_pair_candidates(
    candidate_articulations: wp.array[int],
    candidate_secondary_articulations: wp.array[int],
    articulation_world: wp.array[int],
    local_solve_owner: wp.array[int],
    expected_owner: int,
    # outputs
    active_count: wp.array[int],
    active_articulations: wp.array[int],
    active_secondary_articulations: wp.array[int],
):
    """Compact the pair candidates whose world selected the given local owner."""
    candidate = wp.tid()
    articulation = candidate_articulations[candidate]
    world = articulation_world[articulation]
    if local_solve_owner[world] == expected_owner:
        active_index = wp.atomic_add(active_count, 0, 1)
        active_articulations[active_index] = articulation
        active_secondary_articulations[active_index] = candidate_secondary_articulations[candidate]


@wp.kernel
def clear_local_solve_diag(
    world_constraint_count: wp.array[int],
    local_solve_owner: wp.array[int],
    max_constraints: int,
    # output
    world_diag: wp.array2d[float],
):
    """Discard stale response diagonals of locally owned worlds."""
    tid = wp.tid()
    row = tid % max_constraints
    world = tid // max_constraints
    if local_solve_owner[world] != PGS_LOCAL_SOLVE_OWNER_GENERAL and row < world_constraint_count[world]:
        world_diag[world, row] = 0.0
