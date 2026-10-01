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

from functools import cache

import warp as wp

from ...math.spatial import transform_twist
from ...sim import BodyFlags, JointType
from ...sim.articulation import (
    compute_2d_rotational_dofs,
    compute_3d_rotational_dofs,
    eval_single_articulation_fk,
)
from .contact_filters import contact_friction_eligible, contact_normal_gap_limit
from .friction import friction_pair_candidate
from .friction_patches import FrictionPatches, contact_tangent_basis, patch_normal_load

PGS_CONSTRAINT_TYPE_CONTACT = 0
PGS_CONSTRAINT_TYPE_JOINT_TARGET = 1
PGS_CONSTRAINT_TYPE_FRICTION = 2
PGS_CONSTRAINT_TYPE_JOINT_LIMIT = 3
# Joint velocity-limit row. Mirrors the PhysX per-DOF velocity clamp (see
# ``notes/investigations/velocity-spike/physx-deep-dive.md`` §4, math
# appendix). Finite limits create two unilateral rows every step: one lower
# bound row and one upper bound row. No Baumgarte / ERP bias.
PGS_CONSTRAINT_TYPE_JOINT_VELOCITY_LIMIT = 4
# Mimic (joint coupling) row: the bilateral equality
# ``q_follower - coef1 * q_leader - coef0 = 0`` between two 1-DoF joints of the
# same articulation, from ``Model.constraint_mimic_*``. Jacobian has two
# entries (+1 on the follower DOF, -coef1 on the leader DOF); lambda is
# unbounded; drift is removed with the standard Baumgarte ``pgs_beta`` bias.
PGS_CONSTRAINT_TYPE_MIMIC = 5
# Connect (loop-closure) row: one axis of the bilateral 3-DoF point-coincidence equality
# ``p_anchor(parent) - p_anchor(child) = 0`` closing a kinematic loop between two links of
# the same articulation (imported MJCF ``connect`` equalities become trailing BALL loop
# joints; see ``_FeatherPGSModelPlan.build``). Three rows per closure, world-axis
# directions, unbounded lambda, Baumgarte drift correction.
PGS_CONSTRAINT_TYPE_CONNECT = 6
PGS_CONSTRAINT_TYPE_TORSION = 7
PGS_CONSTRAINT_TYPE_COUNT = 8

# Keep launch-geometry-specific dynamics kernels out of the large general
# kernel module. Warp compiles one whole module variant per block dimension.
_KINEMATICS_KERNEL_MODULE = wp.Module(f"{__name__}.kinematics")
_INVERSE_DYNAMICS_KERNEL_MODULE = wp.Module(f"{__name__}.inverse_dynamics")
_MASS_DYNAMICS_KERNEL_MODULE = wp.Module(f"{__name__}.mass_dynamics")

PGS_LOCAL_SOLVE_OWNER_GENERAL = 0
PGS_LOCAL_SOLVE_OWNER_SINGLE = 1
PGS_LOCAL_SOLVE_OWNER_PAIR = 2
PGS_LOCAL_SOLVE_OWNER_PAIR_RESIDUAL = 3


@wp.kernel
def local_solve_launch_gate():
    """Create a minimal graph dependency ahead of the bulk local solves."""

    pass


# Numeric IDs for the ``friction_mode`` argument passed to the matrix-free
# PGS solver kernels.  Mirrors the Python-side string enum on
# :class:`~newton.solvers.SolverFeatherPGS` (``"current"`` /
# ``"bisection"`` / ``"bisection_desaxce"`` / ``"coulomb_newton"``).
# The matrix-free kernel branches on the numeric id at each PGS row to
# avoid duplicating the kernel body per friction strategy (see the
# ``[FPGS Friction Modes]`` issue series).
FRICTION_MODE_CURRENT = 0
FRICTION_MODE_BISECTION = 1
FRICTION_MODE_BISECTION_DESAXCE = 2
# Gilles Daviet's scalar bracketed-Newton on the tangential-force ratio
# alpha (FPGS Friction Modes 7/13).  The @wp.func implementation is
# :func:`friction_step_coulomb_newton`; the core solver is ported from
# ``artifacts/2026-04-16-slack-raisim/coulomb_root_finding_warp.py``.
FRICTION_MODE_COULOMB_NEWTON = 3

# RAISim-style bisection step count for :func:`friction_step_bisection`.
# Matches the ``BISECTION_ITERS`` constant in Miles Macklin's
# ``solvers/raisim/kernels.py``.
_FPGS_BISECTION_ITERS = 20

# Bracketed-Newton expansion / iteration bounds for
# :func:`friction_step_coulomb_newton`.  Match the constants in
# ``coulomb_root_finding_warp.py::solve_coulomb`` so the in-solver
# behaviour is byte-for-byte identical to the reference self-test
# (``|phi| < ~5e-6`` at solution).
_FPGS_COULOMB_NEWTON_EXPAND_ITERS = 30
_FPGS_COULOMB_NEWTON_NEWTON_ITERS = 50

# One-sided distance slop [m] for float32 cancellation at the speculative
# bound. Rows within 1e-6 m are treated as reaching the surface, bounding early
# activation to 1e-6 m of linearized separation.
_FPGS_CONTACT_END_GAP_SLOP = wp.constant(1.0e-6)


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


@wp.kernel(module=_KINEMATICS_KERNEL_MODULE)
def update_articulation_origins(
    articulation_start: wp.array[int],
    joint_child: wp.array[int],
    body_q: wp.array[wp.transform],
    body_com: wp.array[wp.vec3],
    # outputs
    articulation_origin: wp.array[wp.vec3],
):
    art = wp.tid()

    start = articulation_start[art]
    end = articulation_start[art + 1]

    if start >= end:
        articulation_origin[art] = wp.vec3()
        return

    root_body = joint_child[start]
    if root_body >= 0:
        # Store the absolute world-space COM position of the articulation root body.
        articulation_origin[art] = wp.transform_point(body_q[root_body], body_com[root_body])
    else:
        articulation_origin[art] = wp.vec3()


@wp.kernel
def clamp_free_root_velocity_limits(
    articulation_start: wp.array[int],
    joint_child: wp.array[int],
    body_flags: wp.array[wp.int32],
    articulation_root_is_free: wp.array[int],
    articulation_root_dof_start: wp.array[int],
    rigid_body_max_linear_velocity: wp.array[float],
    rigid_body_max_angular_velocity: wp.array[float],
    articulation_active: wp.array[int],
    # outputs
    qd: wp.array[float],
):
    """Clamp free-root linear/angular velocity magnitudes to PhysX rigid-body limits."""
    art = wp.tid()
    if articulation_active[art] == 0:
        return
    if articulation_root_is_free[art] == 0:
        return

    root_joint = articulation_start[art]
    root_body = joint_child[root_joint]
    if root_body < 0:
        return
    if (body_flags[root_body] & BodyFlags.KINEMATIC) != 0:
        return

    ds = articulation_root_dof_start[art]

    max_lin = rigid_body_max_linear_velocity[root_body]
    if max_lin > 0.0 and wp.isfinite(max_lin):
        lin = wp.vec3(qd[ds + 0], qd[ds + 1], qd[ds + 2])
        lin_speed = wp.length(lin)
        if lin_speed > max_lin:
            scale = max_lin / lin_speed
            qd[ds + 0] = lin[0] * scale
            qd[ds + 1] = lin[1] * scale
            qd[ds + 2] = lin[2] * scale

    max_ang = rigid_body_max_angular_velocity[root_body]
    if max_ang > 0.0 and wp.isfinite(max_ang):
        ang = wp.vec3(qd[ds + 3], qd[ds + 4], qd[ds + 5])
        ang_speed = wp.length(ang)
        if ang_speed > max_ang:
            scale = max_ang / ang_speed
            qd[ds + 3] = ang[0] * scale
            qd[ds + 4] = ang[1] * scale
            qd[ds + 5] = ang[2] * scale


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
    # in/out
    joint_qd: wp.array[float],
):
    """PhysX-style pre-solve joint velocity scaling.

    PhysX computes a single ratio per articulation from maxJointVelocity and
    applies that ratio to all articulation DOFs before building link velocities.
    This is separate from the velocity-limit constraint rows solved later.

    ``skip_driven != 0`` (the ``fuse_joint_velocity_limits`` path) excludes
    DOFs with a PhysX drive row (``drive_slot[dof] >= 0``) from both the
    ratio computation and the scaling: those DOFs are clamped in-solve by the
    fused end-of-iteration velocity clamp instead. ``drive_slot`` holds the previous
    step's allocation at this point in the step (it is initialized to -1, so
    the very first step prescales exactly as before); the driven-DOF set is
    derived from ``joint_target_ke/kd > 0`` and is stable across steps.
    When ``skip_driven == 0``, ``drive_slot`` is never read (a 1-element
    dummy is safe).
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


@wp.kernel(module=_KINEMATICS_KERNEL_MODULE)
def eval_rigid_fk(
    articulation_start: wp.array[int],
    articulation_joint_end: wp.array[int],
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
    # one thread per-articulation
    index = wp.tid()

    start = articulation_start[index]
    # Tree prefix only: trailing loop-closing joints would overwrite their child's pose.
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
def dense_index(stride: int, i: int, j: int):
    return i * stride + j


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
def eval_rigid_fk_kinematics(
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
    articulation_active: wp.array[int],
    # outputs
    body_q: wp.array[wp.transform],
    body_q_com: wp.array[wp.transform],
    articulation_origin: wp.array[wp.vec3],
    joint_S_s: wp.array[wp.spatial_vector],
    body_v_s: wp.array[wp.spatial_vector],
    body_a_s: wp.array[wp.spatial_vector],
    fk_id_cache_valid: wp.array[int],
):
    """Propagate articulation poses, motion subspaces, velocities, and accelerations."""
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

    cached_child = int(-1)
    cached_v_s = wp.spatial_vector()
    cached_a_s = wp.spatial_vector()
    for i in range(start, end):
        parent = joint_parent[i]
        child = joint_child[i]
        parent_v_s = wp.spatial_vector()
        parent_a_s = wp.spatial_vector()
        if parent >= 0:
            if parent == cached_child:
                parent_v_s = cached_v_s
                parent_a_s = cached_a_s
            else:
                parent_v_s = body_v_s[parent]
                parent_a_s = body_a_s[parent]
        cached_v_s, cached_a_s = compute_link_kinematics(
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
        cached_child = child
    fk_id_cache_valid[index] = 1


@wp.kernel(module=_MASS_DYNAMICS_KERNEL_MODULE)
def finalize_body_dynamics(
    body_to_articulation: wp.array[int],
    body_q: wp.array[wp.transform],
    body_q_com: wp.array[wp.transform],
    body_com: wp.array[wp.vec3],
    body_mass: wp.array[float],
    body_inertia: wp.array[wp.mat33],
    is_free_rigid: wp.array[int],
    articulation_origin: wp.array[wp.vec3],
    materialize_all_body_inertia: int,
    materialize_body_inertia_terms: int,
    body_world: wp.array[int],
    body_disable_gravity: wp.array[bool],
    gravity: wp.array[wp.vec3],
    body_v_s: wp.array[wp.spatial_vector],
    body_a_s: wp.array[wp.spatial_vector],
    body_active: wp.array[int],
    # outputs
    body_I_s: wp.array[wp.spatial_matrix],
    body_inertia_terms: wp.array2d[float],
    body_f_s: wp.array[wp.spatial_vector],
    body_qd: wp.array[wp.spatial_vector],
):
    """Build independent body dynamics and publish COM velocity in parallel."""
    body = wp.tid()
    if body_active[body] == 0:
        return
    articulation = body_to_articulation[body]
    v_s = body_v_s[body]
    if articulation < 0:
        com_world = wp.transform_point(body_q[body], body_com[body])
        v_com = wp.spatial_top(v_s) + wp.cross(wp.spatial_bottom(v_s), com_world)
        body_qd[body] = wp.spatial_vector(v_com, wp.spatial_bottom(v_s))
        return

    origin = articulation_origin[articulation]
    X_sm = body_q_com[body]
    X_sm_local = wp.transform(
        wp.transform_get_translation(X_sm) - origin,
        wp.transform_get_rotation(X_sm),
    )
    mass = body_mass[body]
    com, inertia_origin = transform_com_inertia_terms(X_sm_local, mass, body_inertia[body])

    write_body_inertia = materialize_all_body_inertia
    if is_free_rigid[articulation] != 0:
        write_body_inertia = 1
    if write_body_inertia != 0:
        body_I_s[body] = assemble_com_spatial_inertia(mass, com, inertia_origin)
    if materialize_body_inertia_terms != 0:
        body_inertia_terms[body, 0] = com[0]
        body_inertia_terms[body, 1] = com[1]
        body_inertia_terms[body, 2] = com[2]
        for row in range(3):
            for col in range(3):
                body_inertia_terms[body, 3 + 3 * row + col] = inertia_origin[row, col]

    a_s = body_a_s[body]
    coriolis = spatial_cross_dual(v_s, mul_com_spatial_inertia(mass, com, inertia_origin, v_s))
    f_b_s = mul_com_spatial_inertia(mass, com, inertia_origin, a_s) + coriolis
    f_g = wp.vec3()
    if not body_disable_gravity[body]:
        f_g = mass * gravity[body_world[body]]
    body_f_s[body] = f_b_s - wp.spatial_vector(f_g, wp.cross(com, f_g))

    com_world = wp.transform_point(body_q[body], body_com[body])
    com_rel = com_world - origin
    v_com = wp.spatial_top(v_s) + wp.cross(wp.spatial_bottom(v_s), com_rel)
    body_qd[body] = wp.spatial_vector(v_com, wp.spatial_bottom(v_s))


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
    reuse_cached: int,
    body_world: wp.array[int],
    body_disable_gravity: wp.array[bool],
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
    fk_id_cache_valid: wp.array[int],
):
    """Evaluate and cache articulation poses and inverse dynamics."""
    index = wp.tid()
    if reuse_cached != 0 and fk_id_cache_valid[index] != 0:
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
        gravity_s = wp.vec3()
        if not body_disable_gravity[child]:
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
    fk_id_cache_valid[index] = 1


@wp.func_native("""
#if defined(__CUDA_ARCH__)
    __syncwarp();
#endif
""")
def _tree_warp_sync():
    """Order dependent tree levels within one complete CUDA warp."""


@cache
def _get_tree_fk_kernel(lanes: int, mode: str):
    """Build a cooperative tree traversal without replacing its per-joint mathematics.

    ``id`` and ``kinematics`` share the ordinary FK/ID signature after the six
    schedule arguments. ``public`` instead takes the public FK helper's arrays.
    Unary chains use joint-sized segments when that shortens the topology span.
    Launch complete 32-thread warps;
    CPU and differentiable execution stay serial.
    """
    if lanes not in (1, 2, 4, 8, 16, 32):
        raise ValueError("Tree traversal lanes must be a power of two between 1 and 32")
    if mode not in ("id", "kinematics", "public"):
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
                            # Retain the public FREE/D6 COM-velocity conventions.
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
                if wp.static(lanes > 1):
                    _tree_warp_sync()

        return tree_fk_public

    compute_dynamics = mode == "id"

    @wp.kernel(module=module, enable_backward=False)
    def tree_fk(
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
        reuse_cached: int,
        body_world: wp.array[int],
        body_disable_gravity: wp.array[bool],
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
        fk_id_cache_valid: wp.array[int],
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
            if wp.static(compute_dynamics):
                if reuse_cached != 0 and fk_id_cache_valid[articulation] != 0:
                    active = False

        # The authored first root defines the common frame, including forests.
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
            if wp.static(compute_dynamics):
                if is_free_rigid[articulation] != 0:
                    write_body_inertia = 1

        # A joint needs only its current parent and the root frame, not descendant
        # poses. Fuse both passes and retain parent motion along each unary chain.
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
                        if wp.static(compute_dynamics):
                            gravity_s = wp.vec3()
                            if not body_disable_gravity[child]:
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
                        else:
                            parent_v_s, parent_a_s = compute_link_kinematics(
                                joint,
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
                    segment += lanes
            # Inactive groups still participate in every full-warp barrier.
            if wp.static(lanes > 1):
                _tree_warp_sync()
        if active and local == 0:
            fk_id_cache_valid[articulation] = 1

    return tree_fk


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


# Inverse dynamics via Recursive Newton-Euler algorithm (Featherstone Table 5.1)
@wp.kernel(module=_KINEMATICS_KERNEL_MODULE)
def eval_rigid_id(
    articulation_start: wp.array[int],
    articulation_joint_end: wp.array[int],
    joint_type: wp.array[int],
    joint_parent: wp.array[int],
    joint_child: wp.array[int],
    joint_qd_start: wp.array[int],
    joint_qd: wp.array[float],
    joint_axis: wp.array[wp.vec3],
    joint_dof_dim: wp.array2d[int],
    body_mass: wp.array[float],
    body_inertia: wp.array[wp.mat33],
    is_free_rigid: wp.array[int],
    materialize_all_body_inertia: int,
    materialize_body_inertia_terms: int,
    body_q: wp.array[wp.transform],
    body_q_com: wp.array[wp.transform],
    joint_X_p: wp.array[wp.transform],
    articulation_origin: wp.array[wp.vec3],
    body_world: wp.array[int],
    body_disable_gravity: wp.array[bool],
    gravity: wp.array[wp.vec3],
    # outputs
    joint_S_s: wp.array[wp.spatial_vector],
    body_I_s: wp.array[wp.spatial_matrix],
    body_inertia_terms: wp.array2d[float],
    body_v_s: wp.array[wp.spatial_vector],
    body_f_s: wp.array[wp.spatial_vector],
    body_a_s: wp.array[wp.spatial_vector],
):
    # one thread per-articulation
    index = wp.tid()

    start = articulation_start[index]
    # Tree prefix only: trailing loop-closing joints carry no motion subspaces.
    end = articulation_joint_end[index]
    origin = articulation_origin[index]
    write_body_inertia = materialize_all_body_inertia
    if is_free_rigid[index] != 0:
        write_body_inertia = 1
    cached_child = int(-1)
    cached_v_s = wp.spatial_vector()
    cached_a_s = wp.spatial_vector()

    # compute link velocities and coriolis forces
    for i in range(start, end):
        parent = joint_parent[i]
        child = joint_child[i]
        gravity_s = wp.vec3()
        if not body_disable_gravity[child]:
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


@cache
def _get_tree_tau_kernel(lanes: int):
    """Reduce chain wrenches without atomics, preserving descending child order."""
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
                    # The host orders immediate children by descending joint index,
                    # exactly matching the serial articulation's accumulation order.
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
                            add_existing_tau,
                            tau,
                        )
                    segment_net[segment] = f_s
                    segment += lanes
            # Tau is never skipped on FK cache hits: controls and wrenches can change.
            if wp.static(lanes > 1):
                _tree_warp_sync()

    return tree_tau


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
    # outputs
    body_ft_s: wp.array[wp.spatial_vector],
    tau: wp.array[float],
):
    if articulation_active[wp.tid()] == 0:
        return
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


@wp.kernel(module=_INVERSE_DYNAMICS_KERNEL_MODULE)
def eval_rigid_tau_add(
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
    # outputs
    body_ft_s: wp.array[wp.spatial_vector],
    tau: wp.array[float],
):
    """Accumulate inverse dynamics onto a previously prepared drive-force bucket."""
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
        1,
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
    # compute the Cholesky factorization of A = L L^T with diagonal regularization R
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


@wp.kernel(module=_MASS_DYNAMICS_KERNEL_MODULE)
def factor_diagonal_mass(
    H_group: wp.array3d[float],
    R_group: wp.array2d[float],
    group_to_art: wp.array[int],
    mass_update_mask: wp.array[int],
    n_dofs: int,
    # output
    L_group: wp.array3d[float],
):
    """Factor structurally diagonal generalized mass matrices."""
    element = wp.tid()
    group = element // n_dofs
    dof = element - group * n_dofs
    art = group_to_art[group]
    if mass_update_mask[art] != 0:
        mass = H_group[group, dof, dof] + R_group[group, dof]
        L_group[group, dof, dof] = wp.sqrt(mass)


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
    drive_row_by_dof: wp.array[int],
    drive_row_K: wp.array[float],
    n_dofs: int,
    # output
    diagonal_inverse_mass: wp.array[float],
):
    """Build an inverse diagonal directly from independent articulation branches."""
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
    drive_row = drive_row_by_dof[global_dof]
    if drive_row >= 0:
        stiffness = drive_row_K[drive_row]
        if stiffness > 0.0:
            mass += stiffness
    diagonal_inverse_mass[global_dof] = 1.0 / mass


@wp.kernel(module=_MASS_DYNAMICS_KERNEL_MODULE)
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
    """Prepare independent diagonal-articulation limit projections per world coordinate."""
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


@wp.kernel(module=_MASS_DYNAMICS_KERNEL_MODULE)
def solve_diagonal_mass(
    L_group: wp.array3d[float],
    group_to_art: wp.array[int],
    articulation_dof_start: wp.array[int],
    n_dofs: int,
    joint_tau: wp.array[float],
    articulation_active: wp.array[int],
    # output
    joint_qdd: wp.array[float],
):
    """Solve a structurally diagonal generalized mass system."""
    element = wp.tid()
    group = element // n_dofs
    dof = element - group * n_dofs
    art = group_to_art[group]
    if articulation_active[art] == 0:
        return
    global_dof = articulation_dof_start[art] + dof
    diagonal = L_group[group, dof, dof]
    value = joint_tau[global_dof] / diagonal
    joint_qdd[global_dof] = value / diagonal


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
    """Apply a directly stored inverse diagonal to generalized forces."""
    element = wp.tid()
    group = element // n_dofs
    dof = element - group * n_dofs
    art = group_to_art[group]
    if articulation_active[art] == 0:
        return
    global_dof = articulation_dof_start[art] + dof
    joint_qdd[global_dof] = joint_tau[global_dof] * diagonal_inverse_mass[global_dof]


@wp.func
def dense_subs(
    n: int,
    L_start: int,
    b_start: int,
    L: wp.array[float],
    b: wp.array[float],
    # outputs
    x: wp.array[float],
):
    # Solves (L L^T) x = b for x given the Cholesky factor L
    # forward substitution solves the lower triangular system L y = b for y
    for i in range(n):
        s = b[b_start + i]

        for j in range(i):
            s -= L[L_start + dense_index(n, i, j)] * x[b_start + j]

        x[b_start + i] = s / L[L_start + dense_index(n, i, i)]

    # backward substitution solves the upper triangular system L^T x = y for x
    for i in range(n - 1, -1, -1):
        s = x[b_start + i]

        for j in range(i + 1, n):
            s -= L[L_start + dense_index(n, j, i)] * x[b_start + j]

        x[b_start + i] = s / L[L_start + dense_index(n, i, i)]


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
    # in/out
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
    requires_grad: bool,
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
    if not requires_grad:
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
    if not requires_grad:
        # Isotropic inertia has identically zero gyroscopic bias. Preserve
        # its predictor exactly and avoid local solves for spheres and cubes.
        # The derivative with respect to inertia need not vanish here.
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
    # in/out
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
    angular_damping: float,
    joint_active: wp.array[int],
    # outputs
    joint_q_new: wp.array[float],
    joint_qd_new: wp.array[float],
):
    # one thread per-articulation
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
        angular_damping,
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
    # in/out
    joint_qdd: wp.array[float],
    # outputs
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
    # in/out
    v_new: wp.array[float],
    # output
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
def reset_friction_anchor_history(
    world_mask: wp.array[wp.bool],
    prev_world: wp.array[int],
    prev_valid: wp.array[int],
):
    """Drop the carried friction anchors of reset worlds (all worlds for an empty mask)."""
    c = wp.tid()
    if world_mask:
        w = prev_world[c]
        if w >= 0 and w < world_mask.shape[0] and not world_mask[w]:
            return
    prev_valid[c] = 0


@wp.kernel
def compute_contact_linear_force_from_impulses(
    contact_count: wp.array[wp.int32],
    contact_normal: wp.array[wp.vec3],
    contact_world: wp.array[wp.int32],
    contact_slot: wp.array[wp.int32],
    contact_path: wp.array[wp.int32],
    world_impulses: wp.array2d[wp.float32],
    mf_impulses: wp.array2d[wp.float32],
    propagation_impulses: wp.array2d[wp.float32],
    world_constraint_count: wp.array[wp.int32],
    mf_constraint_count: wp.array[wp.int32],
    propagation_constraint_count: wp.array[wp.int32],
    world_row_type: wp.array2d[wp.int32],
    world_row_parent: wp.array2d[wp.int32],
    mf_row_type: wp.array2d[wp.int32],
    mf_row_parent: wp.array2d[wp.int32],
    propagation_row_type: wp.array2d[wp.int32],
    propagation_row_parent: wp.array2d[wp.int32],
    enable_friction: int,
    inv_dt: float,
    # outputs
    rigid_contact_force: wp.array[wp.vec3],
):
    """Convert solved FeatherPGS contact impulses into world-frame forces."""
    c = wp.tid()
    total_contacts = contact_count[0]
    if c >= total_contacts:
        return

    force = wp.vec3(0.0)
    slot = contact_slot[c]
    path = contact_path[c]

    if slot >= 0 and path >= 0 and inv_dt > 0.0:
        world = contact_world[c]
        # Contacts store normals from shape 0 toward shape 1 (A-to-B). FeatherPGS
        # solves along the opposite direction internally, which corresponds to the
        # force on shape/body 0 from shape/body 1.
        normal = -contact_normal[c]

        lam_n = 0.0
        lam_t0 = 0.0
        lam_t1 = 0.0
        if path == 0:
            lam_n = world_impulses[world, slot]
            count = world_constraint_count[world]
            if (
                enable_friction != 0
                and slot + 2 < count
                and world_row_type[world, slot + 1] == PGS_CONSTRAINT_TYPE_FRICTION
                and world_row_parent[world, slot + 1] == slot
                and world_row_type[world, slot + 2] == PGS_CONSTRAINT_TYPE_FRICTION
                and world_row_parent[world, slot + 2] == slot
            ):
                lam_t0 = world_impulses[world, slot + 1]
                lam_t1 = world_impulses[world, slot + 2]
        elif path == 1:
            lam_n = mf_impulses[world, slot]
            count = mf_constraint_count[world]
            if (
                enable_friction != 0
                and slot + 2 < count
                and mf_row_type[world, slot + 1] == PGS_CONSTRAINT_TYPE_FRICTION
                and mf_row_parent[world, slot + 1] == slot
                and mf_row_type[world, slot + 2] == PGS_CONSTRAINT_TYPE_FRICTION
                and mf_row_parent[world, slot + 2] == slot
            ):
                lam_t0 = mf_impulses[world, slot + 1]
                lam_t1 = mf_impulses[world, slot + 2]
        elif path == 2:
            lam_n = propagation_impulses[world, slot]
            count = propagation_constraint_count[world]
            if (
                enable_friction != 0
                and slot + 2 < count
                and propagation_row_type[world, slot + 1] == PGS_CONSTRAINT_TYPE_FRICTION
                and propagation_row_parent[world, slot + 1] == slot
                and propagation_row_type[world, slot + 2] == PGS_CONSTRAINT_TYPE_FRICTION
                and propagation_row_parent[world, slot + 2] == slot
            ):
                lam_t0 = propagation_impulses[world, slot + 1]
                lam_t1 = propagation_impulses[world, slot + 2]

        force = lam_n * normal
        if enable_friction != 0:
            tangent0, tangent1 = contact_tangent_basis(normal)
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
    articulation_active: wp.array[int],
    # outputs
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


@wp.kernel(module=_INVERSE_DYNAMICS_KERNEL_MODULE)
def prepare_augmented_joint_drives(
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
    # outputs
    row_counts: wp.array[int],
    row_dof_index: wp.array[int],
    row_K: wp.array[float],
    drive_tau: wp.array[float],
):
    """Prepare implicit drive rows and their explicit force independently of inverse dynamics."""
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
        1,
        row_counts,
        row_dof_index,
        row_K,
        drive_tau,
    )


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
    """Allocate one dense PhysX-style drive row for each driven scalar DOF."""
    art = wp.tid()
    world = art_to_world[art]

    dof_base = articulation_dof_start[art]
    dof_count = articulation_H_rows[art]
    for d in range(dof_count):
        drive_slot[dof_base + d] = -1

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
            stiffness = joint_target_ke[dof]
            damping = joint_target_kd[dof]
            if stiffness <= 0.0 and damping <= 0.0:
                continue

            slot = wp.atomic_add(world_slot_counter, world, 1)
            if slot < max_constraints:
                drive_slot[dof] = slot


@wp.kernel
def clear_grouped_jacobian_active_rows(
    group_to_art: wp.array[int],
    art_to_world: wp.array[int],
    world_constraint_count: wp.array[int],
    dof_count: int,
    max_constraints: int,
    # outputs
    J_group: wp.array3d[float],
):
    """Clear only Jacobian rows that were active in the completed solve.

    The launch assigns one warp to each articulation. Packing those warps in
    ordinary CUDA blocks avoids the one-block-per-world scheduling overhead,
    while lanes clear the compact active prefix cooperatively.
    """
    tid = wp.tid()
    group_idx = tid // 32
    lane = tid % 32
    if group_idx >= group_to_art.shape[0]:
        return

    art = group_to_art[group_idx]
    world = art_to_world[art]
    row_count = wp.min(world_constraint_count[world], max_constraints)
    element_count = row_count * dof_count
    for element in range(lane, element_count, 32):
        row = element // dof_count
        dof = element - row * dof_count
        J_group[group_idx, row, dof] = 0.0


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
    world_row_beta: wp.array2d[float],
    world_row_cfm: wp.array2d[float],
    world_phi: wp.array2d[float],
    world_target_velocity: wp.array2d[float],
    world_drive_stiffness: wp.array2d[float],
    world_drive_damping: wp.array2d[float],
    world_drive_geom_error: wp.array2d[float],
    world_drive_max_force: wp.array2d[float],
    world_drive_vel_limit: wp.array2d[float],
):
    """Populate PhysX-style drive rows and row data for one articulation size group.

    ``fuse_vel_limits != 0`` (the ``fuse_joint_velocity_limits`` path)
    additionally records the DOF's joint velocity limit per drive row in
    ``world_drive_vel_limit`` so the solve kernels can apply the PhysX-style
    stateless velocity clamp at the end of each iteration in place of
    dedicated velocity-limit rows. Non-positive / non-finite limits are stored as
    ``+inf`` so the fused clamp is a no-op for unlimited DOFs. When
    ``fuse_vel_limits == 0`` neither ``joint_velocity_limit`` nor
    ``world_drive_vel_limit`` is touched (1-element dummies are safe).
    """
    group_idx = wp.tid()
    art = group_to_art[group_idx]
    world = art_to_world[art]
    dof_start = articulation_dof_start[art]

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
        q_start = joint_q_start[j]

        for axis in range(axis_count):
            dof = qd_start + axis
            slot = drive_slot[dof]
            if slot < 0:
                continue

            local_dof = dof - dof_start
            J_group[group_idx, slot, local_dof] = 1.0

            stiffness = joint_target_ke[dof]
            damping = joint_target_kd[dof]
            target_pos = joint_target_pos[joint_target_q_start[j] + axis]
            target_vel = joint_target_vel[dof]
            q = joint_q[q_start + axis]

            world_row_type[world, slot] = PGS_CONSTRAINT_TYPE_JOINT_TARGET
            world_row_parent[world, slot] = -1
            world_row_mu[world, slot] = 0.0
            world_row_beta[world, slot] = 0.0
            world_row_cfm[world, slot] = 0.0
            world_phi[world, slot] = 0.0
            world_target_velocity[world, slot] = target_vel
            world_drive_stiffness[world, slot] = stiffness
            world_drive_damping[world, slot] = damping
            world_drive_geom_error[world, slot] = target_pos - q
            world_drive_max_force[world, slot] = joint_effort_limit[dof]
            if fuse_vel_limits != 0:
                qdot_max = joint_velocity_limit[dof]
                if qdot_max <= 0.0 or not wp.isfinite(qdot_max):
                    qdot_max = float(wp.inf)
                world_drive_vel_limit[world, slot] = qdot_max


@wp.kernel
def compute_physx_pgs_drive_desc(
    world_constraint_count: wp.array[int],
    max_constraints: int,
    world_dof_indices: wp.array2d[int],
    max_world_dofs: int,
    world_row_type: wp.array2d[int],
    world_diag: wp.array2d[float],
    world_J: wp.array3d[float],
    position_delta_velocity: wp.array[float],
    world_target_velocity: wp.array2d[float],
    world_drive_stiffness: wp.array2d[float],
    world_drive_damping: wp.array2d[float],
    world_drive_geom_error: wp.array2d[float],
    world_drive_max_force: wp.array2d[float],
    dt: float,
    position_bias_scale: float,
    position_delta_scale: float,
    # outputs
    world_drive_target_vel_bias: wp.array2d[float],
    world_drive_vel_multiplier: wp.array2d[float],
    world_drive_impulse_multiplier: wp.array2d[float],
    world_drive_max_impulse: wp.array2d[float],
):
    """Build PhysX PGS force-drive descriptor fields for dense joint target rows."""
    world = wp.tid()
    m = world_constraint_count[world]

    for i in range(m):
        if world_row_type[world, i] != PGS_CONSTRAINT_TYPE_JOINT_TARGET:
            world_drive_target_vel_bias[world, i] = 0.0
            world_drive_vel_multiplier[world, i] = 0.0
            world_drive_impulse_multiplier[world, i] = 0.0
            world_drive_max_impulse[world, i] = 0.0
            continue

        stiffness = world_drive_stiffness[world, i]
        damping = world_drive_damping[world, i]
        target_vel = world_target_velocity[world, i]
        geom_error = world_drive_geom_error[world, i]
        unit_response = world_diag[world, i]

        a = dt * (dt * stiffness + damping)
        b = dt * (damping * target_vel)
        x = float(0.0)
        if unit_response > 0.0:
            x = 1.0 / (1.0 + a * unit_response)

        drive_bias_coeff = stiffness * x * dt
        position_delta = float(0.0)
        if position_delta_scale != 0.0:
            for d in range(max_world_dofs):
                global_dof = world_dof_indices[world, d]
                if global_dof >= 0:
                    position_delta += world_J[world, i, d] * position_delta_velocity[global_dof]
            position_delta = position_delta_scale * dt * position_delta

        world_drive_target_vel_bias[world, i] = x * b + drive_bias_coeff * (
            position_bias_scale * geom_error - position_delta
        )
        world_drive_vel_multiplier[world, i] = -x * a
        # PGS path: PhysX uses 1 - x. TGS would use 1 and additional position bias.
        world_drive_impulse_multiplier[world, i] = 1.0 - x

        max_force = world_drive_max_force[world, i]
        max_impulse = float(1.0e20)
        if max_force > 0.0 and wp.isfinite(max_force):
            max_impulse = max_force * dt
        world_drive_max_impulse[world, i] = max_impulse


@wp.kernel
def build_mass_update_mask(
    global_flag: int,
    mass_update_requested: wp.array[int],
    articulation_active: wp.array[int],
    # outputs
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
# Joint Limit Constraint Kernels
# =============================================================================


@wp.kernel
def build_joint_limit_rows_for_size(
    articulation_start: wp.array[int],
    articulation_dof_start: wp.array[int],
    joint_type: wp.array[int],
    joint_q_start: wp.array[int],
    joint_qd_start: wp.array[int],
    joint_dof_dim: wp.array2d[int],
    joint_limit_lower: wp.array[float],
    joint_limit_upper: wp.array[float],
    joint_q: wp.array[float],
    joint_limit_activation_gap: float,
    art_to_world: wp.array[int],
    group_to_art: wp.array[int],
    max_constraints: int,
    pgs_beta: float,
    pgs_cfm: float,
    # outputs
    world_slot_counter: wp.array[int],
    J_group: wp.array3d[float],
    world_row_type: wp.array2d[int],
    world_row_parent: wp.array2d[int],
    world_row_mu: wp.array2d[float],
    world_row_beta: wp.array2d[float],
    world_row_cfm: wp.array2d[float],
    world_phi: wp.array2d[float],
    world_target_velocity: wp.array2d[float],
):
    """Allocate and populate active joint-limit rows in one articulation pass."""
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
            q_val = joint_q[q_start + axis]
            lower = joint_limit_lower[dof]
            upper = joint_limit_upper[dof]

            local_dof = dof - dof_start

            for side in range(2):
                sign = 1.0
                phi = q_val - lower
                active = wp.isfinite(lower) and q_val <= lower + joint_limit_activation_gap
                if side == 1:
                    sign = -1.0
                    phi = upper - q_val
                    active = wp.isfinite(upper) and q_val >= upper - joint_limit_activation_gap
                if not active:
                    continue

                slot = wp.atomic_add(world_slot_counter, world, 1)
                if slot >= max_constraints:
                    continue

                J_group[group_idx, slot, local_dof] = sign
                world_row_type[world, slot] = PGS_CONSTRAINT_TYPE_JOINT_LIMIT
                world_row_parent[world, slot] = -1
                world_row_mu[world, slot] = 0.0
                world_row_beta[world, slot] = pgs_beta
                world_row_cfm[world, slot] = pgs_cfm
                world_phi[world, slot] = phi
                world_target_velocity[world, slot] = 0.0


# =============================================================================
# Mimic (Joint Coupling) Constraint Kernels
# =============================================================================
# Bilateral equality rows enforcing ``q_follower = coef0 + coef1 * q_leader``
# between two DOFs of the same articulation, sourced from joint-owned
# ``Model.joint_mimic_*`` (one row per coordinate) or the deprecated
# ``Model.constraint_mimic_*``. This is the FeatherPGS analogue of PhysX's
# ``PxArticulationMimicJoint`` (``qA + gearRatio*qB + offset = 0`` with
# ``gearRatio = -coef1``, ``offset = -coef0``): a joint-space row with two
# Jacobian entries, an unbounded multiplier, and standard Baumgarte drift
# correction. See ``reports/vishal/fpgs_mimic_design.md`` in the parent repo.
#
# ``coef0``/``coef1``/``enabled`` are read from the model arrays at row-build
# time, so runtime mutation (NotifyFlags.CONSTRAINT_PROPERTIES) takes effect
# without solver re-initialization and stays CUDA-graph-safe. The static
# validity mask (same articulation, 1-DoF joint types, distinct DOFs) is
# precomputed on the host at solver init.


@wp.kernel
def allocate_mimic_slots(
    mimic_valid: wp.array[int],
    mimic_legacy: wp.array[int],
    mimic_enabled: wp.array[wp.bool],
    mimic_world: wp.array[int],
    max_constraints: int,
    # outputs
    mimic_slot: wp.array[int],
    world_slot_counter: wp.array[int],
):
    """Allocate one dense constraint slot per enabled, valid mimic constraint.

    Launched with one thread per mimic row. Disabled or invalid rows get
    ``mimic_slot = -1`` and consume no slot; joint-owned rows are always enabled.
    """
    k = wp.tid()
    mimic_slot[k] = -1
    if mimic_valid[k] == 0:
        return
    legacy = mimic_legacy[k]
    if legacy >= 0 and not mimic_enabled[legacy]:
        return
    slot = wp.atomic_add(world_slot_counter, mimic_world[k], 1)
    if slot < max_constraints:
        mimic_slot[k] = slot


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
    pgs_beta: float,
    pgs_cfm: float,
    # outputs
    J_group: wp.array3d[float],
    world_row_type: wp.array2d[int],
    world_row_parent: wp.array2d[int],
    world_row_mu: wp.array2d[float],
    world_row_beta: wp.array2d[float],
    world_row_cfm: wp.array2d[float],
    world_phi: wp.array2d[float],
    world_target_velocity: wp.array2d[float],
):
    """Populate Jacobian and metadata for mimic constraint rows.

    Launched once per size group with ``dim = n_arts_of_size``, matching the
    joint-limit populate kernel. Each thread visits only its articulation's
    range in the precomputed mimic table:

    * Jacobian ``J = e_follower - coef1 * e_leader`` — two entries.
    * ``phi = q_follower - coef1 * q_leader - coef0`` — the signed violation.
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
        if legacy >= 0:
            c0 = mimic_coef0[legacy]
            c1 = mimic_coef1[legacy]
        else:
            coeffs = joint_mimic_coeffs[mimic_owner[k]]
            c0 = coeffs[0]
            c1 = coeffs[1]

        # Jacobian: +1 on the follower DOF, -coef1 on the leader DOF. The two
        # DOFs are guaranteed distinct by the host-side validity mask.
        J_group[group_idx, slot, mimic_dof0[k] - dof_start] = 1.0
        J_group[group_idx, slot, mimic_dof1[k] - dof_start] = -c1

        world_row_type[world, slot] = PGS_CONSTRAINT_TYPE_MIMIC
        world_row_parent[world, slot] = -1
        world_row_mu[world, slot] = 0.0
        world_row_beta[world, slot] = pgs_beta
        world_row_cfm[world, slot] = pgs_cfm
        world_phi[world, slot] = joint_q[mimic_q0[k]] - c1 * joint_q[mimic_q1[k]] - c0
        world_target_velocity[world, slot] = 0.0


# =============================================================================
# Connect (Loop-Closure) Constraint Kernels
# =============================================================================
# Three bilateral rows per loop closure enforcing point coincidence of the loop
# joint's parent/child anchors — the FeatherPGS realization of MJCF ``connect``
# equalities (design: ``reports/vishal/fpgs_connect_design.md`` in the parent
# repo). The Jacobian per axis is the anchor-point Jacobian difference of the
# two bodies, built with the same ancestor walk contact rows use.


@wp.kernel
def invalidate_fk_id_cache_for_articulations(
    articulations: wp.array[int],
    # outputs
    fk_id_cache_valid: wp.array[int],
):
    """Clear the FK/ID cache flag of the listed articulations only."""
    k = wp.tid()
    art = articulations[k]
    if art >= 0 and art < fk_id_cache_valid.shape[0]:
        fk_id_cache_valid[art] = 0


@wp.kernel
def allocate_connect_slots(
    connect_valid: wp.array[int],
    connect_enabled: wp.array[int],
    connect_world: wp.array[int],
    max_constraints: int,
    # outputs
    connect_slot: wp.array[int],
    world_slot_counter: wp.array[int],
):
    """Allocate three consecutive dense slots per enabled, valid loop closure.

    Launched with ``dim = n_connect``. If the buffer cannot hold all three rows the
    closure is dropped whole (``connect_slot = -1``) rather than partially enforced.
    """
    k = wp.tid()
    connect_slot[k] = -1
    if connect_valid[k] == 0 or connect_enabled[k] == 0:
        return
    slot = wp.atomic_add(world_slot_counter, connect_world[k], 3)
    if slot + 2 < max_constraints:
        connect_slot[k] = slot


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
    body_to_articulation: wp.array[int],
    joint_ancestor: wp.array[int],
    joint_qd_start: wp.array[int],
    joint_S_s: wp.array[wp.spatial_vector],
    articulation_origin: wp.array[wp.vec3],
    pgs_beta: float,
    pgs_cfm: float,
    # outputs
    J_group: wp.array3d[float],
    world_row_type: wp.array2d[int],
    world_row_parent: wp.array2d[int],
    world_row_mu: wp.array2d[float],
    world_row_beta: wp.array2d[float],
    world_row_cfm: wp.array2d[float],
    world_phi: wp.array2d[float],
    world_target_velocity: wp.array2d[float],
):
    """Populate Jacobian and metadata for connect (loop-closure) rows.

    Launched once per size group with ``dim = n_arts_of_size``. Each thread scans the
    (small) closure table and, for closures belonging to its articulation, writes three
    rows: for world axis ``e_k``, ``phi = (p_A - p_B) . e_k`` and
    ``J = e_k . (Jpoint(parent, p_A) - Jpoint(child, p_B))`` via the contact ancestor
    walk. One near-redundant axis on planar linkages is expected; its vanishing Delassus
    diagonal makes the sweep skip it.

    A closure whose parent is prescribed (``connect_parent_prescribed[k] != 0``: a
    kinematic body outside the child's articulation, or the world when
    ``connect_body_p[k] < 0``) contributes no parent DOFs. Its anchor velocity is
    instead moved to the row target, ``J v = -(e_k . v_A)``, so the child anchor tracks
    the moving parent anchor exactly as a same-articulation closure would.
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
                # Prescribed parent: the anchor velocity comes from the body twist
                # (v_com_world, omega_world) rather than from tree DOFs.
                twist = body_qd[body_p]
                v_com = wp.spatial_top(twist)
                omega = wp.spatial_bottom(twist)
                x_com = wp.transform_point(body_q[body_p], body_com[body_p])
                v_a = v_com + wp.cross(omega, p_a - x_com)
        p_b = wp.transform_point(body_q[body_c], connect_anchor_c[k])
        origin = articulation_origin[art]
        rel_a = p_a - origin
        rel_b = p_b - origin
        delta = p_a - p_b

        for axis in range(3):
            slot = base_slot + axis
            # The two ancestor walks below ACCUMULATE (shared ancestors partially cancel),
            # so the row must start from zero regardless of buffer reuse policy.
            for d in range(n_dofs):
                J_group[group_idx, slot, d] = 0.0
            e = wp.vec3(0.0, 0.0, 0.0)
            e[axis] = 1.0
            target = float(0.0)

            # J = e . (Jpoint(parent, p_a) - Jpoint(child, p_b)): ancestor walks with
            # opposite signs; shared ancestors partially cancel automatically.
            if prescribed:
                target = -wp.dot(e, v_a)
            else:
                curr = body_to_joint[body_p]
                while curr != -1:
                    d0 = joint_qd_start[curr]
                    d1 = joint_qd_start[curr + 1]
                    for d in range(d0, d1):
                        S = joint_S_s[d]
                        lin = wp.vec3(S[0], S[1], S[2])
                        ang = wp.vec3(S[3], S[4], S[5])
                        J_group[group_idx, slot, d - dof_start] += wp.dot(e, lin + wp.cross(ang, rel_a))
                    curr = joint_ancestor[curr]
            curr = body_to_joint[body_c]
            while curr != -1:
                d0 = joint_qd_start[curr]
                d1 = joint_qd_start[curr + 1]
                for d in range(d0, d1):
                    S = joint_S_s[d]
                    lin = wp.vec3(S[0], S[1], S[2])
                    ang = wp.vec3(S[3], S[4], S[5])
                    J_group[group_idx, slot, d - dof_start] -= wp.dot(e, lin + wp.cross(ang, rel_b))
                curr = joint_ancestor[curr]

            # Normalize the row: on planar linkages one axis is near-redundant with the
            # tree, leaving ||J|| ~ 1e-4 — its Delassus diagonal collapses to pure CFM
            # while H^-1 J^T stays large against small link inertias, and PGS then relaxes
            # a real velocity effect against a near-zero denominator (measured to NaN in
            # one sweep). Normalizing keeps the identical constraint with a well-scaled
            # diagonal; truly degenerate rows are left inert (zero J, zero phi).
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
            world_row_beta[world, slot] = pgs_beta
            world_row_cfm[world, slot] = pgs_cfm
            world_phi[world, slot] = phi_axis
            world_target_velocity[world, slot] = target


# =============================================================================
# Bilateral Pre-elimination Kernels (mimic + connect Schur complement)
# =============================================================================
# Fold the selected bilateral internal equality rows (MIMIC and/or CONNECT) into the
# response operator so every other row sees the closed-loop effective mass:
# with B the bilateral block of one articulation, Y_B = H^-1 J_B^T and
# S = J_B Y_B (+ regularization), every other row's response is corrected to
# Y'_i = Y_i - Y_B S^-1 (J_B Y_i), which makes J_B Y'_i ~ 0 — sweep impulses
# then preserve the closures exactly, and the corrected row diagonals fall out
# of the unchanged diag_from_JY pass. The predictor velocity is projected once
# (J_B v + b_B -> 0) to replace the eliminated rows' Baumgarte work. See
# reports/vishal/robotiq/fpgs_preelimination_design.md (skild-IL-solver).

PREELIM_MAX_ROWS = 8
"""Per-articulation capacity of the pre-eliminated bilateral block (the
Robotiq 2F-85 needs 7: one mimic + two 3-row connect closures)."""

_preelim_vec = wp.types.vector(length=PREELIM_MAX_ROWS, dtype=wp.float32)


@wp.func
def preelim_solve(
    n: int,
    L: wp.array[float],
    L_start: int,
    b: _preelim_vec,
) -> _preelim_vec:
    """Solve (L L^T) x = b for the packed per-articulation Cholesky factor.

    Mirrors :func:`dense_subs` but keeps the right-hand side in registers so
    per-row callers need no global scratch.
    """
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
    """Gather the selected bilateral block, form S = J_B Y_B, and factor it.

    Launched once per size group with ``dim = n_arts_of_size`` after the
    ``Y = H^-1 J^T`` stage; one thread per articulation (the block is tiny).
    Ownership comes from the per-constraint ``*_art``/``*_slot`` tables — NOT
    ``row_type``, which is per-world and would admit other articulations'
    zero-J rows and make S singular. Zero-norm rows (the degenerate connect
    axis the populate kernel already zeroes) are dropped here for the same
    reason.
    """
    group_idx = wp.tid()
    art = group_to_art[group_idx]
    pe = art_to_preelim[art]
    if pe < 0:
        return
    base = pe * PREELIM_MAX_ROWS

    # Gather candidate slots owned by this articulation.
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

    # Drop zero-norm rows (inert degenerate axes) by compaction.
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

    # S[p, q] = J_B[p] . Y_B[q]  (n x n, row-major with stride m).
    s_base = pe * PREELIM_MAX_ROWS * PREELIM_MAX_ROWS
    for p in range(m):
        sp = preelim_slots[base + p]
        for q in range(m):
            sq = preelim_slots[base + q]
            acc = float(0.0)
            for d in range(n_dofs):
                acc += J_group[group_idx, sp, d] * Y_group[group_idx, sq, d]
            S_scratch[s_base + dense_index(m, p, q)] = acc

    # RELATIVE diagonal regularization: a planar four-bar's closure has one
    # near-dependent axis, so after eliminating the well-conditioned rows the
    # last pivot is pure float32 cancellation noise around zero — an absolute
    # epsilon under the Delassus scale (which is huge for gram-scale links)
    # goes negative in the sqrt. Scaling by the row's own diagonal keeps the
    # factor SPD at any mass scale; the redundant axis just turns ~reg_rel
    # soft, which is benign (the tree already enforces that direction).
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
    """Correct every non-B row's response: Y_i -= Y_B S^-1 (J_B Y_i).

    Launched per size group with ``dim = n_arts_of_size * max_constraints``.
    B rows themselves are left intact (the projection kernel needs Y_B; their
    sweep visits self-neutralize because the projected velocity already
    satisfies J_B v = -b_B, so their residual is ~0).
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
            return  # B row: keep its response for the projection

    # w = J_B . Y_i
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
        return  # row does not touch this articulation (e.g. other art's row)

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
    """One-shot bilateral projection: v_out -= Y_B S^-1 (J_B v_out + b_B).

    Launched per size group with ``dim = n_arts_of_size`` after the sweep's
    velocity buffer is seeded with the predictor (v_out = v_hat). Afterwards
    J_B v_out = -b_B holds, and it stays held through the sweep because every
    corrected column satisfies J_B Y'_i ~ 0. Each thread owns one
    articulation's contiguous DOF window, so there are no write races.
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
# These kernels mirror the PhysX per-DOF velocity-limit formulation documented
# in ``notes/investigations/velocity-spike/physx-deep-dive.md`` §4 and the math
# appendix. They reuse the same allocation / populate shape as the
# joint-position-limit kernels above, but finite velocity limits allocate both
# sides of the bilateral velocity box rather than waiting for a violation.


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
    # outputs
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

    ``skip_driven != 0`` (the ``fuse_joint_velocity_limits`` path) skips DOFs
    with a PhysX drive row (``drive_slot[dof] >= 0``): their velocity limit is
    enforced as a stateless clamp at the end of each solver iteration
    instead of a dedicated row pair (PhysX ``PxClamp`` math, run after the
    contact phases like the rows it replaces). DOFs with a velocity limit but no drive row keep their
    dedicated rows. When ``skip_driven == 0``, ``drive_slot`` is never read
    (a 1-element dummy is safe).

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

            # Fused path: driven DOFs are clamped at the end of each iteration.
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
    pgs_cfm: float,
    # outputs
    J_group: wp.array3d[float],
    world_row_type: wp.array2d[int],
    world_row_parent: wp.array2d[int],
    world_row_mu: wp.array2d[float],
    world_row_beta: wp.array2d[float],
    world_row_cfm: wp.array2d[float],
    world_phi: wp.array2d[float],
    world_target_velocity: wp.array2d[float],
):
    """Populate Jacobian and metadata for joint velocity-limit rows.

    Launched once per size group with ``dim = n_arts_of_size``. For every DOF
    whose two ``velocity_limit_slot`` entries are non-negative, writes signed
    ±1 entries into the local DOF column of the grouped Jacobian and sets the
    constraint metadata. The rows have **no Baumgarte bias** (``beta = 0``,
    ``phi = 0``) — PhysX's velocity-limit row has no ``data.erp`` either.
    The target velocity is ``-qdot_max`` for both sides of the box: combined
    with the sign flip on ``J``, this encodes the bilateral projection as
    two unilateral ``J*v >= target_vel`` rows with ``lambda >= 0``.

    DOFs skipped by :func:`allocate_joint_velocity_limit_slots` (activation
    gate, or driven DOFs under ``fuse_joint_velocity_limits``) carry
    ``velocity_limit_slot == -1`` and are skipped here through the same
    per-row slot check; this kernel needs no separate skip flag.
    """
    group_idx = wp.tid()
    art = group_to_art[group_idx]
    world = art_to_world[art]
    dof_start = articulation_dof_start[art]

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
            if qdot_max <= 0.0:
                continue

            local_dof = dof - dof_start
            lower_idx = 2 * dof

            for side in range(2):
                row_idx = lower_idx + side
                slot = velocity_limit_slot[row_idx]
                if slot < 0:
                    continue

                sign = velocity_limit_sign[row_idx]

                # Single signed ±1 entry at the local DOF column. The selector
                # row on generalised velocity is ``J = sign * e_i``; the
                # articulated-body response ``J M^-1 J^T`` is exactly PhysX's
                # ``recipResponse`` on the same axis and is computed by the
                # existing ``hinv_jt_par_row`` / ``diag_from_JY_par_art`` path.
                J_group[group_idx, slot, local_dof] = sign

                world_row_type[world, slot] = PGS_CONSTRAINT_TYPE_JOINT_VELOCITY_LIMIT
                world_row_parent[world, slot] = -1
                world_row_mu[world, slot] = 0.0
                # No Baumgarte / ERP — matches PhysX vel-limit row (§4).
                world_row_beta[world, slot] = 0.0
                world_row_cfm[world, slot] = pgs_cfm
                world_phi[world, slot] = 0.0
                # ``target_vel = -qdot_max`` for both signs: rhs = -target + J*v
                # = qdot_max +/- qdot_i, which is negative exactly when the
                # corresponding side of the box is violated.
                world_target_velocity[world, slot] = -qdot_max


# =============================================================================
# Multi-Articulation Contact Building Kernels
# =============================================================================
# These kernels enable contacts between multiple articulations within the same
# world. The constraint system becomes world-level instead of per-articulation.


@wp.func
def _allocate_world_contact_slot(
    c: int,
    total_contacts: int,
    contact_shape0: wp.array[int],
    contact_shape1: wp.array[int],
    contact_point0: wp.array[wp.vec3],
    contact_point1: wp.array[wp.vec3],
    contact_normal: wp.array[wp.vec3],
    contact_thickness0: wp.array[float],
    contact_thickness1: wp.array[float],
    body_q: wp.array[wp.transform],
    shape_transform: wp.array[wp.transform],
    shape_body: wp.array[int],
    body_to_articulation: wp.array[int],
    art_to_world: wp.array[int],
    articulation_response_dof_count: wp.array[int],
    body_flags: wp.array[wp.int32],
    body_has_response_dofs: wp.array[int],
    is_free_rigid: wp.array[int],
    has_free_rigid: int,
    propagation_articulated_contacts: int,
    propagation_same_articulation: int,
    propagation_free_free: int,
    contact_gap_gate: float,
    same_articulation_contact_gap_gate: float,
    articulation_pair_contact_gap_gate: float,
    max_constraints: int,
    mf_max_constraints: int,
    propagation_max_constraints: int,
    enable_friction: int,
    contact_friction_gap_threshold: float,
    contact_friction_articulation_pairs_only: int,
    row_capacity_telemetry: int,
    friction_patches: FrictionPatches,
    # outputs
    contact_world: wp.array[int],
    contact_slot: wp.array[int],
    contact_art_a: wp.array[int],
    contact_art_b: wp.array[int],
    world_slot_counter: wp.array[int],
    contact_path: wp.array[int],
    mf_slot_counter: wp.array[int],
    propagation_slot_counter: wp.array[int],
    dense_contact_world_flag: wp.array[int],
    contact_slots_needed: wp.array[int],
    dense_dropped_contact_rows: wp.array[int],
    mf_dropped_contact_rows: wp.array[int],
    propagation_dropped_contact_rows: wp.array[int],
    dense_first_rejected_slot: wp.array[int],
    mf_first_rejected_slot: wp.array[int],
    propagation_first_rejected_slot: wp.array[int],
):
    """Classify and allocate rows for one active contact.

    Allocates world-level constraint slots for each contact and records
    which articulations are involved. Contacts where both sides are free
    rigid bodies (or ground) are routed to the matrix-free path.

    Each contact reserves 1 slot for the normal row and, when enabled below the
    friction gap threshold, 2 adjacent slots for Coulomb friction rows.
    A positive contact gap gate excludes wider speculative contacts before any
    route reserves slots; zero disables the gate.

    """
    shape_a = contact_shape0[c]
    shape_b = contact_shape1[c]

    # Get bodies and articulations
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
    if not a_can_respond and not b_can_respond:
        contact_slot[c] = -1
        contact_path[c] = -1
        return

    # Determine world (both bodies must be in same world, or one is ground)
    world = -1
    if art_a >= 0:
        world = art_to_world[art_a]
    if art_b >= 0:
        world_b = art_to_world[art_b]
        if world >= 0 and world_b != world:
            # Cross-world contact - shouldn't happen, skip
            contact_slot[c] = -1
            contact_path[c] = -1
            return
        world = world_b

    if world < 0:
        # No articulation involved (ground-ground?)
        contact_slot[c] = -1
        contact_path[c] = -1
        return

    # Compute phi (same logic as populate_world_J_for_size)
    # contact normal stored as A-to-B; negate to get B-to-A used internally
    normal = -contact_normal[c]
    point_a_local = contact_point0[c]
    point_b_local = contact_point1[c]
    thickness_a = contact_thickness0[c]
    thickness_b = contact_thickness1[c]

    point_a_world = wp.vec3(0.0)
    point_b_world = wp.vec3(0.0)

    if body_a >= 0:
        X_wb_a = body_q[body_a]
        # Contact points are stored in body frame by collision detection
        point_a_world = wp.transform_point(X_wb_a, point_a_local) - thickness_a * normal
    else:
        point_a_world = point_a_local - thickness_a * normal

    if body_b >= 0:
        X_wb_b = body_q[body_b]
        # Contact points are stored in body frame by collision detection
        point_b_world = wp.transform_point(X_wb_b, point_b_local) + thickness_b * normal
    else:
        point_b_world = point_b_local + thickness_b * normal
    phi = wp.dot(normal, point_a_world - point_b_world)

    # Use the same eligibility as patch selection before reserving row storage.
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

    # Classify: MF path if both sides are free rigid or ground.
    # Propagation path is an opt-in matrix-free route for any contact touching a
    # non-free articulation. It stores fixed-size body-space rows instead of
    # D-wide generalized rows.
    is_mf = 0
    is_propagation = 0
    a_non_free = art_a >= 0 and is_free_rigid[art_a] == 0
    b_non_free = art_b >= 0 and is_free_rigid[art_b] == 0
    if has_free_rigid != 0:
        a_is_mf_compatible = not a_has_dofs or is_free_rigid[art_a] != 0
        b_is_mf_compatible = not b_has_dofs or is_free_rigid[art_b] != 0
        if a_is_mf_compatible and b_is_mf_compatible:
            is_mf = 1
    if propagation_free_free != 0 and is_mf == 1:
        # Opt-in: route free/free (and free/ground) contact rows to the
        # propagation family. The row math is identical (free bodies use the
        # plain spatial inverse inertia response seeded by
        # copy_free_rigid_propagation_body_response); this puts pile rows on
        # the colored solve path. Velocity-limit rows stay on the MF family.
        is_mf = 0
        is_propagation = 1
    if propagation_articulated_contacts != 0 and is_mf == 0 and is_propagation == 0:
        # Same-articulation two-link contacts need cross response terms between
        # the two touched links. They route to propagation rows only when the
        # cross-response path is enabled; otherwise they stay on the dense
        # generalized-row path.
        same_non_free_articulation = a_non_free and b_non_free and art_a == art_b
        if a_non_free or b_non_free:
            if propagation_same_articulation != 0 or not same_non_free_articulation:
                is_propagation = 1

    # Every normal survives; only the selected patch anchors receive tangent rows.
    slots_needed = 1
    add_friction = enable_friction != 0 and contact_friction_eligible(
        phi, a_non_free, b_non_free, contact_friction_articulation_pairs_only, contact_friction_gap_threshold
    )
    if friction_patches.enabled != 0:
        add_friction = add_friction and friction_patches.weight[c] > 0.0
    if add_friction:
        slots_needed = 3
    contact_slots_needed[c] = slots_needed

    if is_mf != 0:
        # Matrix-free path
        slot = wp.atomic_add(mf_slot_counter, world, slots_needed)
        if slot + slots_needed > mf_max_constraints:
            wp.atomic_min(mf_first_rejected_slot, world, slot)
            if row_capacity_telemetry != 0:
                wp.atomic_add(mf_dropped_contact_rows, world, slots_needed)
            contact_slot[c] = -1
            contact_path[c] = -1
            return
        contact_world[c] = world
        contact_slot[c] = slot
        contact_art_a[c] = art_a
        contact_art_b[c] = art_b
        contact_path[c] = 1
    elif is_propagation != 0:
        # Propagation articulated matrix-free path
        slot = wp.atomic_add(propagation_slot_counter, world, slots_needed)
        if slot + slots_needed > propagation_max_constraints:
            wp.atomic_min(propagation_first_rejected_slot, world, slot)
            if row_capacity_telemetry != 0:
                wp.atomic_add(propagation_dropped_contact_rows, world, slots_needed)
            contact_slot[c] = -1
            contact_path[c] = -1
            return
        contact_world[c] = world
        contact_slot[c] = slot
        contact_art_a[c] = art_a
        contact_art_b[c] = art_b
        contact_path[c] = 2
    else:
        # Dense path
        slot = wp.atomic_add(world_slot_counter, world, slots_needed)
        if slot + slots_needed > max_constraints:
            wp.atomic_min(dense_first_rejected_slot, world, slot)
            if row_capacity_telemetry != 0:
                wp.atomic_add(dense_dropped_contact_rows, world, slots_needed)
            contact_slot[c] = -1
            contact_path[c] = -1
            return
        contact_world[c] = world
        contact_slot[c] = slot
        contact_art_a[c] = art_a
        contact_art_b[c] = art_b
        contact_path[c] = 0
        # Dense contact rows write articulated generalized velocities during
        # split GS phases; propagation tree refresh keys off this per world.
        dense_contact_world_flag[world] = 1


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
    shape_transform: wp.array[wp.transform],
    shape_body: wp.array[int],
    body_to_articulation: wp.array[int],
    art_to_world: wp.array[int],
    articulation_response_dof_count: wp.array[int],
    body_flags: wp.array[wp.int32],
    body_has_response_dofs: wp.array[int],
    is_free_rigid: wp.array[int],
    has_free_rigid: int,
    propagation_articulated_contacts: int,
    propagation_same_articulation: int,
    propagation_free_free: int,
    contact_gap_gate: float,
    same_articulation_contact_gap_gate: float,
    articulation_pair_contact_gap_gate: float,
    max_constraints: int,
    mf_max_constraints: int,
    propagation_max_constraints: int,
    enable_friction: int,
    contact_friction_gap_threshold: float,
    contact_friction_articulation_pairs_only: int,
    row_capacity_telemetry: int,
    friction_patches: FrictionPatches,
    # outputs
    contact_world: wp.array[int],
    contact_slot: wp.array[int],
    contact_art_a: wp.array[int],
    contact_art_b: wp.array[int],
    world_slot_counter: wp.array[int],
    contact_path: wp.array[int],
    mf_slot_counter: wp.array[int],
    propagation_slot_counter: wp.array[int],
    dense_contact_world_flag: wp.array[int],
    contact_slots_needed: wp.array[int],
    dense_dropped_contact_rows: wp.array[int],
    mf_dropped_contact_rows: wp.array[int],
    propagation_dropped_contact_rows: wp.array[int],
    dense_first_rejected_slot: wp.array[int],
    mf_first_rejected_slot: wp.array[int],
    propagation_first_rejected_slot: wp.array[int],
):
    """Allocate active contacts with work proportional to the materialized prefix.

    The narrow phase increments :paramref:`contact_count` before checking its
    output capacity. An overflowed count therefore does not describe a fully
    materialized prefix; clear the routing arrays and reject that frame.

    Slot counters only grow. A contact whose reservation does not fit records its
    slot in the family's ``*_first_rejected_slot`` (per world, preset to a large
    value) instead of rolling the counter back: every later reservation starts at
    or past that slot, so the finalize kernels truncate the row count there and
    no accepted contact can hold a slot at or beyond the count. Rolling back
    raced with concurrent reservations and left accepted rows past the count,
    which the friction patch links then chained into.
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
            total_contacts,
            contact_shape0,
            contact_shape1,
            contact_point0,
            contact_point1,
            contact_normal,
            contact_thickness0,
            contact_thickness1,
            body_q,
            shape_transform,
            shape_body,
            body_to_articulation,
            art_to_world,
            articulation_response_dof_count,
            body_flags,
            body_has_response_dofs,
            is_free_rigid,
            has_free_rigid,
            propagation_articulated_contacts,
            propagation_same_articulation,
            propagation_free_free,
            contact_gap_gate,
            same_articulation_contact_gap_gate,
            articulation_pair_contact_gap_gate,
            max_constraints,
            mf_max_constraints,
            propagation_max_constraints,
            enable_friction,
            contact_friction_gap_threshold,
            contact_friction_articulation_pairs_only,
            row_capacity_telemetry,
            friction_patches,
            contact_world,
            contact_slot,
            contact_art_a,
            contact_art_b,
            world_slot_counter,
            contact_path,
            mf_slot_counter,
            propagation_slot_counter,
            dense_contact_world_flag,
            contact_slots_needed,
            dense_dropped_contact_rows,
            mf_dropped_contact_rows,
            propagation_dropped_contact_rows,
            dense_first_rejected_slot,
            mf_first_rejected_slot,
            propagation_first_rejected_slot,
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
    contact_friction_shared_anchor: int,
    contact_friction_scale: float,
    contact_shared_anchor: int,
    pgs_beta: float,
    pgs_cfm: float,
    friction_patches: FrictionPatches,
    friction_anchor_beta: float,
    # outputs
    world_row_type: wp.array2d[int],
    world_row_parent: wp.array2d[int],
    world_row_mu: wp.array2d[float],
    world_row_beta: wp.array2d[float],
    world_row_cfm: wp.array2d[float],
    world_phi: wp.array2d[float],
    world_target_velocity: wp.array2d[float],
    world_row_restitution: wp.array2d[float],
):
    """Build dense-contact metadata once, independently of articulation size groups."""
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
        normal = -contact_normal[c]
        shape_a = contact_shape0[c]
        shape_b = contact_shape1[c]

        body_a = -1
        body_b = -1
        if shape_a >= 0:
            body_a = shape_body[shape_a]
        if shape_b >= 0:
            body_b = shape_body[shape_b]

        point_a_world = contact_point0[c] - contact_thickness0[c] * normal
        point_b_world = contact_point1[c] + contact_thickness1[c] * normal
        if body_a >= 0:
            point_a_world = wp.transform_point(body_q[body_a], contact_point0[c]) - contact_thickness0[c] * normal
        if body_b >= 0:
            point_b_world = wp.transform_point(body_q[body_b], contact_point1[c]) + contact_thickness1[c] * normal

        phi = wp.dot(normal, point_a_world - point_b_world)
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
        restitution = mixed_contact_restitution(shape_a, shape_b, shape_material_restitution)

        friction_mu = mu * contact_friction_scale

        tangent0, tangent1 = contact_tangent_basis(normal)
        # Allocation owns the row extent; recomputing eligibility can cross a
        # floating-point threshold and overwrite the following contact.
        add_friction = contact_slots_needed[c] >= 3

        contact_anchor_world = 0.5 * (point_a_world + point_b_world)
        point_a_normal = point_a_world
        point_b_normal = point_b_world
        if contact_shared_anchor != 0:
            point_a_normal = contact_anchor_world
            point_b_normal = contact_anchor_world
        point_a_friction = point_a_world
        point_b_friction = point_b_world
        if contact_shared_anchor != 0 or contact_friction_shared_anchor != 0:
            point_a_friction = contact_anchor_world
            point_b_friction = contact_anchor_world
        if friction_patches.enabled != 0:
            point_a_friction = friction_patches.point_a[c]
            point_b_friction = friction_patches.point_b[c]

        normal_target = prescribed_relative_contact_target(
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
        friction0_target = prescribed_relative_contact_target(
            body_a,
            art_a,
            body_b,
            art_b,
            point_a_friction,
            point_b_friction,
            tangent0,
            prescribed_articulation,
            articulation_origin,
            body_v_s,
        )
        friction1_target = prescribed_relative_contact_target(
            body_a,
            art_a,
            body_b,
            art_b,
            point_a_friction,
            point_b_friction,
            tangent1,
            prescribed_articulation,
            articulation_origin,
            body_v_s,
        )

        world_row_type[world, slot] = PGS_CONSTRAINT_TYPE_CONTACT
        world_row_parent[world, slot] = -1
        world_row_mu[world, slot] = mu
        world_row_beta[world, slot] = pgs_beta
        world_row_cfm[world, slot] = pgs_cfm
        world_phi[world, slot] = phi
        world_target_velocity[world, slot] = normal_target
        world_row_restitution[world, slot] = restitution

        if add_friction:
            world_row_type[world, slot + 1] = PGS_CONSTRAINT_TYPE_FRICTION
            world_row_parent[world, slot + 1] = slot
            world_row_mu[world, slot + 1] = friction_mu
            world_row_beta[world, slot + 1] = friction_anchor_beta
            world_row_cfm[world, slot + 1] = pgs_cfm
            world_phi[world, slot + 1] = friction_patches.phi[c][0]
            world_target_velocity[world, slot + 1] = friction0_target
            world_row_restitution[world, slot + 1] = 0.0

            world_row_type[world, slot + 2] = PGS_CONSTRAINT_TYPE_FRICTION
            world_row_parent[world, slot + 2] = slot
            world_row_mu[world, slot + 2] = friction_mu
            world_row_beta[world, slot + 2] = friction_anchor_beta
            world_row_cfm[world, slot + 2] = pgs_cfm
            world_phi[world, slot + 2] = friction_patches.phi[c][1]
            world_target_velocity[world, slot + 2] = friction1_target
            world_row_restitution[world, slot + 2] = 0.0


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
    contact_friction_shared_anchor: int,
    friction_patches: FrictionPatches,
    contact_shared_anchor: int,
    # output
    J_group: wp.array3d[float],
):
    """Project one dense contact row/DOF per lane for compact articulations."""
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
        point_a_world = contact_point0[c] - contact_thickness0[c] * normal
        point_b_world = contact_point1[c] + contact_thickness1[c] * normal
        if body_a >= 0:
            point_a_world = wp.transform_point(body_q[body_a], contact_point0[c]) - contact_thickness0[c] * normal
        if body_b >= 0:
            point_b_world = wp.transform_point(body_q[body_b], contact_point1[c]) + contact_thickness1[c] * normal

        direction = normal
        point_a = point_a_world
        point_b = point_b_world
        contact_anchor_world = 0.5 * (point_a_world + point_b_world)
        if row == 0:
            if contact_shared_anchor != 0:
                point_a = contact_anchor_world
                point_b = contact_anchor_world
        else:
            tangent0, tangent1 = contact_tangent_basis(normal)
            if row == 1:
                direction = tangent0
            else:
                direction = tangent1
            if contact_shared_anchor != 0 or contact_friction_shared_anchor != 0:
                point_a = contact_anchor_world
                point_b = contact_anchor_world
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
                global_dof_a = art_dof_start[art_a] + local_dof
                motion_a = joint_S_s[global_dof_a]
                linear_a = wp.vec3(motion_a[0], motion_a[1], motion_a[2])
                angular_a = wp.vec3(motion_a[3], motion_a[4], motion_a[5])
                velocity_a = linear_a + wp.cross(angular_a, point_a - articulation_origin[art_a])
                value_a = wp.dot(direction, velocity_a)

        if art_b >= 0 and articulation_response_dof_count[art_b] == target_size:
            group_b = art_group_idx[art_b]
            if body_b >= 0 and (body_response_dof_mask[body_b] & bit) != wp.uint32(0):
                global_dof_b = art_dof_start[art_b] + local_dof
                motion_b = joint_S_s[global_dof_b]
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


@wp.kernel
def clear_sparse_diagonal_response_prefix(
    dense_phase_bounds: wp.array2d[int],
    # in/out
    sparse_row_dof: wp.array3d[int],
):
    """Remove stale sparse entries from the non-contact row prefix."""
    world = wp.tid()
    for row in range(dense_phase_bounds[world, 1]):
        sparse_row_dof[world, row, 0] = -1
        sparse_row_dof[world, row, 1] = -1


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
    contact_friction_shared_anchor: int,
    friction_patches: FrictionPatches,
    contact_shared_anchor: int,
    # outputs
    sparse_row_dof: wp.array3d[int],
    sparse_row_jy: wp.array3d[float],
):
    """Build exact ``dense + two sparse`` contact responses for an independent articulation.

    Tangent rows use the persistent patch anchors when patch friction is enabled, exactly as the dense
    contact Jacobian builders do, so the sparse coordinates see the same row geometry.
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
        point_a_world = contact_point0[c] - contact_thickness0[c] * normal
        point_b_world = contact_point1[c] + contact_thickness1[c] * normal
        if body_a >= 0:
            point_a_world = wp.transform_point(body_q[body_a], contact_point0[c]) - contact_thickness0[c] * normal
        if body_b >= 0:
            point_b_world = wp.transform_point(body_q[body_b], contact_point1[c]) + contact_thickness1[c] * normal

        tangent0, tangent1 = contact_tangent_basis(normal)
        contact_anchor_world = 0.5 * (point_a_world + point_b_world)
        world = contact_world[c]
        art_a = contact_art_a[c]
        art_b = contact_art_b[c]
        rows = contact_slots_needed[c]

        for row in range(3):
            if row >= rows:
                continue
            direction = normal
            point_a = point_a_world
            point_b = point_b_world
            if row == 0:
                if contact_shared_anchor != 0:
                    point_a = contact_anchor_world
                    point_b = contact_anchor_world
            else:
                if row == 1:
                    direction = tangent0
                else:
                    direction = tangent1
                if contact_shared_anchor != 0 or contact_friction_shared_anchor != 0:
                    point_a = contact_anchor_world
                    point_b = contact_anchor_world
                if friction_patches.enabled != 0:
                    point_a = friction_patches.point_a[c]
                    point_b = friction_patches.point_b[c]

            dof_a = -1
            dof_b = -1
            coord_a = -1
            coord_b = -1
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
                        velocity_a = linear_a + wp.cross(angular_a, point_a - articulation_origin[art_a])
                        value_a = wp.dot(direction, velocity_a)
                        coord_a = articulation_world_dof_offset[art_a] + local_dof_a

            if art_b >= 0 and body_b >= 0 and articulation_response_dof_count[art_b] == target_size:
                dof_b = body_single_response_dof[body_b]
                if dof_b >= 0:
                    local_dof_b = dof_b - articulation_dof_start[art_b]
                    if local_dof_b >= 0 and local_dof_b < target_size:
                        motion_b = joint_S_s[dof_b]
                        linear_b = wp.vec3(motion_b[0], motion_b[1], motion_b[2])
                        angular_b = wp.vec3(motion_b[3], motion_b[4], motion_b[5])
                        velocity_b = linear_b + wp.cross(angular_b, point_b - articulation_origin[art_b])
                        value_b = -wp.dot(direction, velocity_b)
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
    """Accumulate the two sparse response entries into each active row diagonal."""
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
def apply_sparse_diagonal_contact_restitution_matrix_free(
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
    """Apply restitution from a compact dense component plus two sparse response coordinates."""
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
        world_dof = dense_offset + local_dof
        global_dof = world_dof_indices[world, world_dof]
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
    phi = world_phi[world, row]
    if contact_restitution_fires(phi, relative_incident, dt, restitution_velocity_threshold):
        world_rhs[world, row] = -target_vel + restitution * relative_incident


@wp.func
def world_contact_row_dot(
    world_dof_count: wp.array[int],
    world_dof_indices: wp.array2d[int],
    world_J: wp.array3d[float],
    velocity: wp.array[float],
    world: int,
    i: int,
):
    """Dot one dense world row's Jacobian with a global velocity state."""
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
    """Dot one matrix-free row's two body Jacobians with a global velocity."""
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


@wp.func
def propagation_contact_row_dot(
    propagation_J_a: wp.array3d[float],
    propagation_J_b: wp.array3d[float],
    body_qd: wp.array2d[float],
    world: int,
    i: int,
    body_a: int,
    body_b: int,
):
    """Dot one propagation row's two body Jacobians with body velocities."""
    out = float(0.0)
    if body_a >= 0:
        for k in range(6):
            out += propagation_J_a[world, i, k] * body_qd[body_a, k]
    if body_b >= 0:
        for k in range(6):
            out += propagation_J_b[world, i, k] * body_qd[body_b, k]
    return out


@wp.func
def contact_restitution_fires(
    phi: float,
    relative_incident: float,
    dt: float,
    restitution_velocity_threshold: float,
):
    """Return whether a frozen incident impact qualifies for a rebound target.

    Fires only for a sufficiently fast closing contact that is already at the
    surface or is predicted to reach it during the step; the end-gap slop
    absorbs float32 residuals on rows that land exactly at contact.
    """
    if relative_incident >= -restitution_velocity_threshold:
        return False
    if phi <= _FPGS_CONTACT_END_GAP_SLOP:
        return True
    return phi + dt * relative_incident <= _FPGS_CONTACT_END_GAP_SLOP


@wp.func
def mixed_contact_restitution(
    shape_a: int,
    shape_b: int,
    shape_material_restitution: wp.array[float],
):
    """Return the arithmetic-mean coefficient for one shape pair."""
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
def _populate_world_J_for_size_contact(
    c: int,
    total_contacts: int,
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
    body_v_s: wp.array[wp.spatial_vector],
    prescribed_articulation: wp.array[int],
    shape_transform: wp.array[wp.transform],
    shape_material_mu: wp.array[float],
    shape_material_restitution: wp.array[float],
    enable_friction: int,
    contact_friction_gap_threshold: float,
    contact_friction_shared_anchor: int,
    contact_friction_articulation_pairs_only: int,
    is_free_rigid: wp.array[int],
    contact_friction_scale: float,
    contact_shared_anchor: int,
    pgs_beta: float,
    pgs_cfm: float,
    friction_patches: FrictionPatches,
    friction_anchor_beta: float,
    # outputs
    J_group: wp.array3d[float],
    world_row_type: wp.array2d[int],
    world_row_parent: wp.array2d[int],
    world_row_mu: wp.array2d[float],
    world_row_beta: wp.array2d[float],
    world_row_cfm: wp.array2d[float],
    world_phi: wp.array2d[float],
    world_target_velocity: wp.array2d[float],
    world_row_restitution: wp.array2d[float],
):
    """Populate one contact for a specific articulation-size group.

    Each contact may contribute to multiple articulations' Jacobian matrices.
    Contacts routed to the matrix-free path (contact_path==1) are skipped.
    """
    # Skip contacts routed to MF path
    if contact_path[c] != 0:
        return

    slot = contact_slot[c]
    if slot < 0:
        return

    world = contact_world[c]
    art_a = contact_art_a[c]
    art_b = contact_art_b[c]

    # Get contact geometry
    # contact normal stored as A-to-B; negate to get B-to-A used internally
    normal = -contact_normal[c]
    shape_a = contact_shape0[c]
    shape_b = contact_shape1[c]

    body_a = -1
    body_b = -1
    if shape_a >= 0:
        body_a = shape_body[shape_a]
    if shape_b >= 0:
        body_b = shape_body[shape_b]

    thickness_a = contact_thickness0[c]
    thickness_b = contact_thickness1[c]

    # Compute contact points in world frame
    # Contact points are stored in body frame by collision detection
    point_a_local = contact_point0[c]
    point_b_local = contact_point1[c]
    point_a_world = wp.vec3(0.0)
    point_b_world = wp.vec3(0.0)

    if body_a >= 0:
        X_wb_a = body_q[body_a]
        point_a_world = wp.transform_point(X_wb_a, point_a_local) - thickness_a * normal
    else:
        point_a_world = point_a_local - thickness_a * normal

    if body_b >= 0:
        X_wb_b = body_q[body_b]
        point_b_world = wp.transform_point(X_wb_b, point_b_local) + thickness_b * normal
    else:
        point_b_world = point_b_local + thickness_b * normal

    # Compute penetration depth
    phi = wp.dot(normal, point_a_world - point_b_world)

    # Compute friction coefficient
    mu = 0.0
    mat_count = 0
    if shape_a >= 0:
        mu += shape_material_mu[shape_a]
        mat_count += 1
    if shape_b >= 0:
        mu += shape_material_mu[shape_b]
        mat_count += 1
    if mat_count > 0:
        mu /= float(mat_count)
    restitution = mixed_contact_restitution(shape_a, shape_b, shape_material_restitution)
    a_non_free = art_a >= 0 and is_free_rigid[art_a] == 0
    b_non_free = art_b >= 0 and is_free_rigid[art_b] == 0
    friction_mu = mu * contact_friction_scale

    # Compute tangent basis for friction
    t0, t1 = contact_tangent_basis(normal)
    will_add_friction = enable_friction != 0 and contact_friction_eligible(
        phi, a_non_free, b_non_free, contact_friction_articulation_pairs_only, contact_friction_gap_threshold
    )
    if friction_patches.enabled != 0:
        will_add_friction = will_add_friction and friction_patches.weight[c] > 0.0
    contact_anchor_world = 0.5 * (point_a_world + point_b_world)
    target_point_a_normal = point_a_world
    target_point_b_normal = point_b_world
    if contact_shared_anchor != 0:
        target_point_a_normal = contact_anchor_world
        target_point_b_normal = contact_anchor_world
    target_point_a_friction = point_a_world
    target_point_b_friction = point_b_world
    if contact_shared_anchor != 0 or contact_friction_shared_anchor != 0:
        target_point_a_friction = contact_anchor_world
        target_point_b_friction = contact_anchor_world
    if friction_patches.enabled != 0:
        target_point_a_friction = friction_patches.point_a[c]
        target_point_b_friction = friction_patches.point_b[c]

    normal_target = prescribed_relative_contact_target(
        body_a,
        art_a,
        body_b,
        art_b,
        target_point_a_normal,
        target_point_b_normal,
        normal,
        prescribed_articulation,
        articulation_origin,
        body_v_s,
    )
    friction0_target = prescribed_relative_contact_target(
        body_a,
        art_a,
        body_b,
        art_b,
        target_point_a_friction,
        target_point_b_friction,
        t0,
        prescribed_articulation,
        articulation_origin,
        body_v_s,
    )
    friction1_target = prescribed_relative_contact_target(
        body_a,
        art_a,
        body_b,
        art_b,
        target_point_a_friction,
        target_point_b_friction,
        t1,
        prescribed_articulation,
        articulation_origin,
        body_v_s,
    )

    # Handle articulation A if it matches target size
    if art_a >= 0 and articulation_response_dof_count[art_a] == target_size:
        group_idx_a = art_group_idx[art_a]
        dof_start_a = art_dof_start[art_a]
        origin_a = articulation_origin[art_a]

        # Normal row (slot + 0)
        point_a_normal_world = point_a_world
        if contact_shared_anchor != 0:
            point_a_normal_world = contact_anchor_world

        accumulate_jacobian_row_world(
            body_a,
            1.0,
            point_a_normal_world,
            origin_a,
            normal,
            body_to_joint,
            joint_ancestor,
            joint_qd_start,
            joint_S_s,
            dof_start_a,
            target_size,
            group_idx_a,
            slot,
            J_group,
        )

        if will_add_friction:
            point_a_friction_world = point_a_world
            if contact_shared_anchor != 0 or contact_friction_shared_anchor != 0:
                point_a_friction_world = contact_anchor_world
            if friction_patches.enabled != 0:
                point_a_friction_world = friction_patches.point_a[c]

            # Friction row 1 (slot + 1)
            accumulate_jacobian_row_world(
                body_a,
                1.0,
                point_a_friction_world,
                origin_a,
                t0,
                body_to_joint,
                joint_ancestor,
                joint_qd_start,
                joint_S_s,
                dof_start_a,
                target_size,
                group_idx_a,
                slot + 1,
                J_group,
            )
            # Friction row 2 (slot + 2)
            accumulate_jacobian_row_world(
                body_a,
                1.0,
                point_a_friction_world,
                origin_a,
                t1,
                body_to_joint,
                joint_ancestor,
                joint_qd_start,
                joint_S_s,
                dof_start_a,
                target_size,
                group_idx_a,
                slot + 2,
                J_group,
            )

    # Handle articulation B if it matches target size
    if art_b >= 0 and articulation_response_dof_count[art_b] == target_size:
        group_idx_b = art_group_idx[art_b]
        dof_start_b = art_dof_start[art_b]
        origin_b = articulation_origin[art_b]

        # Opposite sign for body B
        point_b_normal_world = point_b_world
        if contact_shared_anchor != 0:
            point_b_normal_world = contact_anchor_world

        accumulate_jacobian_row_world(
            body_b,
            -1.0,
            point_b_normal_world,
            origin_b,
            normal,
            body_to_joint,
            joint_ancestor,
            joint_qd_start,
            joint_S_s,
            dof_start_b,
            target_size,
            group_idx_b,
            slot,
            J_group,
        )

        if will_add_friction:
            point_b_friction_world = point_b_world
            if contact_shared_anchor != 0 or contact_friction_shared_anchor != 0:
                point_b_friction_world = contact_anchor_world
            if friction_patches.enabled != 0:
                point_b_friction_world = friction_patches.point_b[c]

            accumulate_jacobian_row_world(
                body_b,
                -1.0,
                point_b_friction_world,
                origin_b,
                t0,
                body_to_joint,
                joint_ancestor,
                joint_qd_start,
                joint_S_s,
                dof_start_b,
                target_size,
                group_idx_b,
                slot + 1,
                J_group,
            )
            accumulate_jacobian_row_world(
                body_b,
                -1.0,
                point_b_friction_world,
                origin_b,
                t1,
                body_to_joint,
                joint_ancestor,
                joint_qd_start,
                joint_S_s,
                dof_start_b,
                target_size,
                group_idx_b,
                slot + 2,
                J_group,
            )

    # Set row metadata (only once per contact, from whichever articulation runs first)
    # Use art_a preferentially to avoid double-writes
    if art_a >= 0 and articulation_response_dof_count[art_a] == target_size:
        # Normal contact row
        world_row_type[world, slot] = PGS_CONSTRAINT_TYPE_CONTACT
        world_row_parent[world, slot] = -1
        world_row_mu[world, slot] = mu
        world_row_beta[world, slot] = pgs_beta
        world_row_cfm[world, slot] = pgs_cfm
        world_phi[world, slot] = phi
        world_target_velocity[world, slot] = normal_target
        world_row_restitution[world, slot] = restitution

        if will_add_friction:
            # Friction row 1
            world_row_type[world, slot + 1] = PGS_CONSTRAINT_TYPE_FRICTION
            world_row_parent[world, slot + 1] = slot
            world_row_mu[world, slot + 1] = friction_mu
            world_row_beta[world, slot + 1] = friction_anchor_beta
            world_row_cfm[world, slot + 1] = pgs_cfm
            world_phi[world, slot + 1] = friction_patches.phi[c][0]
            world_target_velocity[world, slot + 1] = friction0_target
            world_row_restitution[world, slot + 1] = 0.0

            # Friction row 2
            world_row_type[world, slot + 2] = PGS_CONSTRAINT_TYPE_FRICTION
            world_row_parent[world, slot + 2] = slot
            world_row_mu[world, slot + 2] = friction_mu
            world_row_beta[world, slot + 2] = friction_anchor_beta
            world_row_cfm[world, slot + 2] = pgs_cfm
            world_phi[world, slot + 2] = friction_patches.phi[c][1]
            world_row_restitution[world, slot + 2] = 0.0
            world_target_velocity[world, slot + 2] = friction1_target

    elif art_b >= 0 and articulation_response_dof_count[art_b] == target_size:
        # Only write metadata from art_b if art_a didn't match this size
        world_row_type[world, slot] = PGS_CONSTRAINT_TYPE_CONTACT
        world_row_parent[world, slot] = -1
        world_row_mu[world, slot] = mu
        world_row_beta[world, slot] = pgs_beta
        world_row_cfm[world, slot] = pgs_cfm
        world_phi[world, slot] = phi
        world_target_velocity[world, slot] = normal_target
        world_row_restitution[world, slot] = restitution

        if will_add_friction:
            world_row_type[world, slot + 1] = PGS_CONSTRAINT_TYPE_FRICTION
            world_row_parent[world, slot + 1] = slot
            world_row_mu[world, slot + 1] = friction_mu
            world_row_beta[world, slot + 1] = friction_anchor_beta
            world_row_cfm[world, slot + 1] = pgs_cfm
            world_phi[world, slot + 1] = friction_patches.phi[c][0]
            world_target_velocity[world, slot + 1] = friction0_target
            world_row_restitution[world, slot + 1] = 0.0

            world_row_type[world, slot + 2] = PGS_CONSTRAINT_TYPE_FRICTION
            world_row_parent[world, slot + 2] = slot
            world_row_mu[world, slot + 2] = friction_mu
            world_row_beta[world, slot + 2] = friction_anchor_beta
            world_row_cfm[world, slot + 2] = pgs_cfm
            world_phi[world, slot + 2] = friction_patches.phi[c][1]
            world_target_velocity[world, slot + 2] = friction1_target
            world_row_restitution[world, slot + 2] = 0.0


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
    contact_world: wp.array[int],
    contact_slot: wp.array[int],
    contact_art_a: wp.array[int],
    contact_art_b: wp.array[int],
    contact_path: wp.array[int],
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
    body_v_s: wp.array[wp.spatial_vector],
    prescribed_articulation: wp.array[int],
    shape_transform: wp.array[wp.transform],
    shape_material_mu: wp.array[float],
    shape_material_restitution: wp.array[float],
    enable_friction: int,
    contact_friction_gap_threshold: float,
    contact_friction_shared_anchor: int,
    contact_friction_articulation_pairs_only: int,
    is_free_rigid: wp.array[int],
    contact_friction_scale: float,
    contact_shared_anchor: int,
    pgs_beta: float,
    pgs_cfm: float,
    friction_patches: FrictionPatches,
    friction_anchor_beta: float,
    # outputs
    J_group: wp.array3d[float],
    world_row_type: wp.array2d[int],
    world_row_parent: wp.array2d[int],
    world_row_mu: wp.array2d[float],
    world_row_beta: wp.array2d[float],
    world_row_cfm: wp.array2d[float],
    world_phi: wp.array2d[float],
    world_target_velocity: wp.array2d[float],
    world_row_restitution: wp.array2d[float],
):
    """Populate active dense contacts with a capacity-independent launch."""
    total_contacts = wp.min(contact_count[0], contact_point0.shape[0])
    for c in range(wp.tid(), total_contacts, total_num_threads):
        _populate_world_J_for_size_contact(
            c,
            total_contacts,
            contact_point0,
            contact_point1,
            contact_normal,
            contact_shape0,
            contact_shape1,
            contact_thickness0,
            contact_thickness1,
            contact_world,
            contact_slot,
            contact_art_a,
            contact_art_b,
            contact_path,
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
            body_v_s,
            prescribed_articulation,
            shape_transform,
            shape_material_mu,
            shape_material_restitution,
            enable_friction,
            contact_friction_gap_threshold,
            contact_friction_shared_anchor,
            contact_friction_articulation_pairs_only,
            is_free_rigid,
            contact_friction_scale,
            contact_shared_anchor,
            pgs_beta,
            pgs_cfm,
            friction_patches,
            friction_anchor_beta,
            J_group,
            world_row_type,
            world_row_parent,
            world_row_mu,
            world_row_beta,
            world_row_cfm,
            world_phi,
            world_target_velocity,
            world_row_restitution,
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
def snapshot_dense_phase_bound(
    world_slot_counter: wp.array[int],
    bound_index: int,
    # outputs
    dense_phase_bounds: wp.array2d[int],
):
    """Record the current dense slot watermark for one row-family boundary.

    The dense per-world slot layout is [drive][joint-limit][joint-vel-limit]
    [contact/friction]. Launched right after the joint-limit allocation
    (``bound_index`` 0) and the joint-velocity-limit allocation
    (``bound_index`` 1) so the PhysX-grasp GS phases can loop only their own
    contiguous row range instead of scanning every dense row. The raw counter
    is stored; consumers clamp against the per-world constraint count.
    """
    world = wp.tid()
    dense_phase_bounds[world, bound_index] = world_slot_counter[world]


@wp.kernel
def apply_augmented_mass_diagonal(
    articulation_H_start: wp.array[int],
    articulation_H_rows: wp.array[int],
    articulation_dof_start: wp.array[int],
    max_dofs: int,
    mass_update_mask: wp.array[int],
    row_counts: wp.array[int],
    row_dof_index: wp.array[int],
    row_K: wp.array[float],
    # outputs
    H: wp.array[float],
):
    articulation = wp.tid()
    if mass_update_mask[articulation] == 0:
        return

    n = articulation_H_rows[articulation]
    if n == 0 or max_dofs == 0:
        return

    count = row_counts[articulation]
    if count == 0:
        return

    H_start = articulation_H_start[articulation]
    dof_start = articulation_dof_start[articulation]

    for i in range(count):
        row_index = articulation * max_dofs + i
        dof = row_dof_index[row_index]
        local = dof - dof_start
        if local < 0 or local >= n:
            continue

        K = row_K[row_index]
        if K <= 0.0:
            continue

        diag_index = H_start + dense_index(n, local, local)
        H[diag_index] += K


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


# --- Tile configuration for contact system build ---
# Kernel naming: {op}_{parallelism}
# parallelism: tiled | loop | par_row | par_row_col | par_dof

# Max generalized dofs per articulation we support in the tiled path.
# joint_dof_count per articulation must be <= TILE_DOF or we use fall back
TILE_DOF = wp.constant(49)

# Max constraints per articulation we support in the tiled path.
# dense_max_constraints must be <= TILE_CONSTRAINTS or we use fall back
TILE_CONSTRAINTS = wp.constant(128)

# Threads per tile/block for tile kernels
TILE_THREADS = 64


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
    max_constraints: int,
    world_phi: wp.array2d[float],
    world_row_beta: wp.array2d[float],
    world_row_type: wp.array2d[int],
    world_target_velocity: wp.array2d[float],
    dt: float,
    bias_scale: float,
    contact_speculative_scale: float,
    joint_limit_speculative_scale: float,
    contact_w: float,
    # outputs
    world_rhs: wp.array2d[float],
    world_row_w: wp.array2d[float],
):
    """Compute the RHS bias term for world-level PGS solve.

    The RHS follows the convention: rhs = J*v + stabilization
    For contacts with penetration (phi < 0): rhs = J*v + beta * phi / dt (negative)
    This leads to positive impulses when resolved by PGS.
    """
    world = wp.tid()
    m = world_constraint_count[world]

    inv_dt = 1.0 / dt

    for i in range(m):
        phi = world_phi[world, i]
        beta = world_row_beta[world, i]
        row_type = world_row_type[world, i]
        target_vel = world_target_velocity[world, i]

        # Initialize with -target_velocity (will add J*v later)
        rhs = -target_vel
        row_w = float(1.0)

        # Contacts inside the speculative gap should not become sticky ghost
        # contacts.  For separation (phi > 0), allow closing by the current gap
        # over this substep: Jv + phi / dt >= 0.  For penetration, keep the
        # Baumgarte correction and let velocity-only passes scale it to zero.
        if row_type == PGS_CONSTRAINT_TYPE_CONTACT:
            if phi <= 0.0:
                rhs += bias_scale * beta * phi * inv_dt  # Negative for penetration
                row_w = contact_w
            else:
                rhs += contact_speculative_scale * phi * inv_dt
        elif row_type == PGS_CONSTRAINT_TYPE_FRICTION:
            # Positional friction anchor (``phi`` = tangential anchor separation,
            # ``beta`` = friction_anchor_beta; both zero when anchors are off).
            rhs += bias_scale * beta * phi * inv_dt
        elif row_type == PGS_CONSTRAINT_TYPE_JOINT_LIMIT:
            if phi < 0.0:
                rhs += bias_scale * beta * phi * inv_dt  # Negative for violation
            else:
                # Speculative finite-limit row: allow motion up to the bound
                # during the step, matching PhysX's nextErr limit branch.
                rhs += joint_limit_speculative_scale * phi * inv_dt
        elif row_type == PGS_CONSTRAINT_TYPE_JOINT_TARGET:
            # PhysX-style drive rows use world_target_velocity as drive-row
            # input, not as a generic constraint target. Their RHS is handled
            # by the per-row drive descriptor in the matrix-free GS kernel.
            rhs = 0.0
        elif row_type == PGS_CONSTRAINT_TYPE_MIMIC or row_type == PGS_CONSTRAINT_TYPE_CONNECT:
            # Bilateral equality: Baumgarte-correct the violation from BOTH
            # sides (contacts/limits only bias on penetration; an equality has
            # no speculative branch).
            rhs += bias_scale * beta * phi * inv_dt
        # PGS_CONSTRAINT_TYPE_JOINT_VELOCITY_LIMIT: no phi-based bias. The
        # constraint is an instantaneous velocity-space projection; the only
        # RHS contribution is ``-target_vel`` (already set above) plus the
        # ``J*v_hat`` term added by ``rhs_accum_world_par_art``.

        world_rhs[world, i] = rhs
        if contact_w < 1.0:
            world_row_w[world, i] = row_w


@wp.kernel
def apply_world_contact_restitution_matrix_free(
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
    """Replace a matrix-free contact bias with a one-shot restitution target."""
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
        # Matrix-free GS adds live J*v itself, so store only
        # -target + e*u_incident as the row bias.
        world_rhs[world, i] = -target_vel + restitution * relative_incident
        if write_row_w != 0:
            world_row_w[world, i] = 1.0


@wp.kernel
def apply_world_contact_restitution_accumulated(
    world_constraint_count: wp.array[int],
    max_constraints: int,
    world_phi: wp.array2d[float],
    world_row_beta: wp.array2d[float],
    world_row_type: wp.array2d[int],
    world_row_restitution: wp.array2d[float],
    dt: float,
    contact_speculative_scale: float,
    restitution_velocity_threshold: float,
    write_row_w: int,
    # in/out
    world_rhs: wp.array2d[float],
    world_row_w: wp.array2d[float],
):
    """Replace an accumulated impulse-space contact RHS with restitution."""
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
    geometric_bias = contact_speculative_scale * phi / dt
    if phi < 0.0:
        geometric_bias = world_row_beta[world, i] * phi / dt
    relative_incident = world_rhs[world, i] - geometric_bias
    if contact_restitution_fires(phi, relative_incident, dt, restitution_velocity_threshold):
        # Impulse-space RHS contains u_incident already.  Replacing geometric
        # bias with the Newton target yields (1+e)*u_incident.
        world_rhs[world, i] = (1.0 + restitution) * relative_incident
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
    apply_restitution: int,
    restitution_velocity_threshold: float,
    # outputs
    world_rhs: wp.array2d[float],
):
    """Build the velocity-pass RHS from the position solution's end gap."""
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
        restitution = float(0.0)
        relative_incident = float(0.0)
        bounce = int(0)
        if apply_restitution != 0:
            restitution = world_row_restitution[world, i]
        if restitution > 0.0:
            relative_incident = (
                world_contact_row_dot(world_dof_count, world_dof_indices, world_J, world_incident_velocity, world, i)
                - target_vel
            )
            if contact_restitution_fires(phi, relative_incident, dt, restitution_velocity_threshold):
                bounce = int(1)

        if bounce != 0:
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
    elif row_type == PGS_CONSTRAINT_TYPE_JOINT_TARGET:
        rhs = 0.0

    world_rhs[world, i] = rhs


@wp.kernel
def rhs_accum_world_par_art(
    world_constraint_count: wp.array[int],
    max_constraints: int,
    art_to_world: wp.array[int],
    art_dof_start: wp.array[int],
    v_hat: wp.array[float],
    group_to_art: wp.array[int],
    J_group: wp.array3d[float],
    n_dofs: int,
    # outputs
    world_rhs: wp.array2d[float],
):
    """
    Accumulate J*v_hat into world RHS for a single size group.

    RHS = J*v + stabilization (already includes stabilization from compute_world_contact_bias)
    This kernel is launched once per size group to accumulate velocity contributions.
    """
    idx = wp.tid()
    art = group_to_art[idx]
    world = art_to_world[art]
    n_constraints = world_constraint_count[world]

    if n_constraints == 0:
        return

    dof_start = art_dof_start[art]

    for c in range(n_constraints):
        jv = float(0.0)
        for d in range(n_dofs):
            jv += J_group[idx, c, d] * v_hat[dof_start + d]
        wp.atomic_add(world_rhs, world, c, jv)  # Add J*v (positive)


@wp.kernel
def prepare_world_impulses(
    world_constraint_count: wp.array[int],
    max_constraints: int,
    warmstart: int,
    # in/out
    world_impulses: wp.array2d[float],
):
    """Cold-initialize dense rows before identity-gathering contact impulses.

    No current non-contact row family has a cross-step identity contract:
    limits are activation-allocated, while mimic/connect rows can be enabled
    dynamically and allocated atomically. Warm start therefore clears the
    full capacity before the contact/friction identity gather; cold solves only
    need to clear the current active prefix.
    """
    world = wp.tid()
    m = world_constraint_count[world]

    # Cold-started solves consume only the current active prefix. Rows outside
    # that prefix are ignored, and a row that becomes active on a later step is
    # cleared then as part of that step's prefix. Warm-started solves clear full
    # capacity so no inactive cache can survive a later layout growth.
    clear_count = max_constraints
    if warmstart == 0:
        clear_count = m

    for i in range(clear_count):
        world_impulses[world, i] = 0.0


@wp.kernel
def reset_world_warmstart_buffers(
    world_mask: wp.array[wp.bool],
    dense_impulses: wp.array2d[float],
    prev_mf_impulses: wp.array2d[float],
    prev_mf_row_type: wp.array2d[int],
    prev_mf_row_parent: wp.array2d[int],
):
    """Clear persistent warm-start state for selected worlds."""
    world = wp.tid()
    if world_mask and not world_mask[world]:
        return

    if world < dense_impulses.shape[0]:
        for row in range(dense_impulses.shape[1]):
            dense_impulses[world, row] = 0.0
    if world < prev_mf_impulses.shape[0]:
        for row in range(prev_mf_impulses.shape[1]):
            prev_mf_impulses[world, row] = 0.0
    if world < prev_mf_row_type.shape[0]:
        for row in range(prev_mf_row_type.shape[1]):
            prev_mf_row_type[world, row] = -1
    if world < prev_mf_row_parent.shape[0]:
        for row in range(prev_mf_row_parent.shape[1]):
            prev_mf_row_parent[world, row] = -1


@wp.kernel
def invalidate_articulation_fk_id_cache(
    world_mask: wp.array[wp.bool],
    art_to_world: wp.array[int],
    fk_id_cache_valid: wp.array[int],
):
    """Invalidate cached articulation dynamics for selected worlds."""
    articulation = wp.tid()
    world = art_to_world[articulation]
    if world_mask and not world_mask[world]:
        return
    fk_id_cache_valid[articulation] = 0


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
    total_contacts: int,
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
    shape_material_restitution: wp.array[float],
    enable_friction: int,
    contact_friction_gap_threshold: float,
    contact_friction_shared_anchor: int,
    contact_friction_articulation_pairs_only: int,
    contact_friction_scale: float,
    contact_shared_anchor: int,
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
    """Build MF constraint rows for contacts between free rigid bodies / ground.

    For root free joints, the internal qd used here stores the COM-point linear
    term and angular velocity, i.e. `qd = [v_com_point, omega]`, where the COM
    point is the root body's world-space COM position. The MF contact Jacobian
    uses contact position relative to that point:
        J = [d, r x d]   (r = p_contact - p_com_world)
    """
    if contact_path[c] != 1:
        return

    slot = contact_slot[c]
    if slot < 0:
        return

    world = contact_world[c]
    shape_a = contact_shape0[c]
    shape_b = contact_shape1[c]
    # contact normal stored as A-to-B; negate to get B-to-A used internally
    normal = -contact_normal[c]

    body_a = -1
    body_b = -1
    if shape_a >= 0:
        body_a = shape_body[shape_a]
    if shape_b >= 0:
        body_b = shape_body[shape_b]

    # Zero-DOF articulations contribute collision geometry but own no
    # generalized response. Treat them like ground after using their body
    # transforms to reconstruct the contact points below.
    response_body_a = -1
    response_body_b = -1
    if body_a >= 0 and contact_art_a[c] >= 0 and articulation_response_dof_count[contact_art_a[c]] > 0:
        response_body_a = body_a
    if body_b >= 0 and contact_art_b[c] >= 0 and articulation_response_dof_count[contact_art_b[c]] > 0:
        response_body_b = body_b

    thickness_a = contact_thickness0[c]
    thickness_b = contact_thickness1[c]

    # Compute contact points in world frame
    point_a_local = contact_point0[c]
    point_b_local = contact_point1[c]
    point_a_world = wp.vec3(0.0)
    point_b_world = wp.vec3(0.0)

    if body_a >= 0:
        X_wb_a = body_q[body_a]
        point_a_world = wp.transform_point(X_wb_a, point_a_local) - thickness_a * normal
    else:
        point_a_world = point_a_local - thickness_a * normal

    if body_b >= 0:
        X_wb_b = body_q[body_b]
        point_b_world = wp.transform_point(X_wb_b, point_b_local) + thickness_b * normal
    else:
        point_b_world = point_b_local + thickness_b * normal

    phi = wp.dot(normal, point_a_world - point_b_world)

    # Friction coefficient
    mu = 0.0
    mat_count = 0
    if shape_a >= 0:
        mu += shape_material_mu[shape_a]
        mat_count += 1
    if shape_b >= 0:
        mu += shape_material_mu[shape_b]
        mat_count += 1
    if mat_count > 0:
        mu /= float(mat_count)
    restitution = mixed_contact_restitution(shape_a, shape_b, shape_material_restitution)
    a_non_free = False
    b_non_free = False
    friction_mu = mu * contact_friction_scale

    # Tangent basis
    t0, t1 = contact_tangent_basis(normal)
    will_add_friction = enable_friction != 0 and contact_friction_eligible(
        phi, a_non_free, b_non_free, contact_friction_articulation_pairs_only, contact_friction_gap_threshold
    )
    if friction_patches.enabled != 0:
        will_add_friction = will_add_friction and friction_patches.weight[c] > 0.0
    contact_anchor_world = 0.5 * (point_a_world + point_b_world)

    # Write rows for normal + friction
    for row_offset in range(3):
        if row_offset > 0 and not will_add_friction:
            break

        row_idx = slot + row_offset

        if row_offset == 0:
            d = normal
        elif row_offset == 1:
            d = t0
        else:
            d = t1

        point_a_row_world = point_a_world
        point_b_row_world = point_b_world
        if contact_shared_anchor != 0 or (row_offset > 0 and contact_friction_shared_anchor != 0):
            point_a_row_world = contact_anchor_world
            point_b_row_world = contact_anchor_world
        if row_offset > 0 and friction_patches.enabled != 0:
            point_a_row_world = friction_patches.point_a[c]
            point_b_row_world = friction_patches.point_b[c]

        # Body A Jacobian in articulation-local frame: J = [d, r_a x d], where
        # r_a is the contact point relative to articulation A's fixed origin.
        if response_body_a >= 0:
            art_a = contact_art_a[c]
            origin_a = articulation_origin[art_a]
            r_a = point_a_row_world - origin_a
            ang_a = wp.cross(r_a, d)
            mf_J_a[world, row_idx, 0] = d[0]
            mf_J_a[world, row_idx, 1] = d[1]
            mf_J_a[world, row_idx, 2] = d[2]
            mf_J_a[world, row_idx, 3] = ang_a[0]
            mf_J_a[world, row_idx, 4] = ang_a[1]
            mf_J_a[world, row_idx, 5] = ang_a[2]

        # Body B Jacobian in articulation-local frame (opposite sign).
        if response_body_b >= 0:
            art_b = contact_art_b[c]
            origin_b = articulation_origin[art_b]
            r_b = point_b_row_world - origin_b
            ang_b = wp.cross(r_b, d)
            mf_J_b[world, row_idx, 0] = -d[0]
            mf_J_b[world, row_idx, 1] = -d[1]
            mf_J_b[world, row_idx, 2] = -d[2]
            mf_J_b[world, row_idx, 3] = -ang_b[0]
            mf_J_b[world, row_idx, 4] = -ang_b[1]
            mf_J_b[world, row_idx, 5] = -ang_b[2]

        mf_body_a[world, row_idx] = response_body_a
        mf_body_b[world, row_idx] = response_body_b

        if row_offset == 0:
            mf_row_type[world, row_idx] = PGS_CONSTRAINT_TYPE_CONTACT
            mf_row_parent[world, row_idx] = -1
            mf_phi[world, row_idx] = phi
            mf_row_restitution[world, row_idx] = restitution
        else:
            mf_row_type[world, row_idx] = PGS_CONSTRAINT_TYPE_FRICTION
            mf_row_parent[world, row_idx] = slot
            if row_offset == 1:
                mf_phi[world, row_idx] = friction_anchor_beta * friction_patches.phi[c][0]
            else:
                mf_phi[world, row_idx] = friction_anchor_beta * friction_patches.phi[c][1]
            mf_row_restitution[world, row_idx] = 0.0
        if row_offset == 0:
            mf_row_mu[world, row_idx] = mu
        else:
            mf_row_mu[world, row_idx] = friction_mu
        if has_target_velocity != 0:
            mf_target_velocity[world, row_idx] = prescribed_relative_contact_target(
                body_a,
                contact_art_a[c],
                body_b,
                contact_art_b[c],
                point_a_row_world,
                point_b_row_world,
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
    shape_material_restitution: wp.array[float],
    enable_friction: int,
    contact_friction_gap_threshold: float,
    contact_friction_shared_anchor: int,
    contact_friction_articulation_pairs_only: int,
    contact_friction_scale: float,
    contact_shared_anchor: int,
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
    """Build active matrix-free contacts with a capacity-independent launch."""
    total_contacts = wp.min(contact_count[0], contact_point0.shape[0])
    for c in range(wp.tid(), total_contacts, total_num_threads):
        _build_mf_contact_row(
            c,
            total_contacts,
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
            shape_material_restitution,
            enable_friction,
            contact_friction_gap_threshold,
            contact_friction_shared_anchor,
            contact_friction_articulation_pairs_only,
            contact_friction_scale,
            contact_shared_anchor,
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
def build_propagation_contact_rows(
    contact_count: wp.array[int],
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
    shape_body: wp.array[int],
    body_q: wp.array[wp.transform],
    body_com: wp.array[wp.vec3],
    is_free_rigid: wp.array[int],
    shape_material_mu: wp.array[float],
    shape_material_restitution: wp.array[float],
    enable_friction: int,
    contact_friction_gap_threshold: float,
    contact_friction_shared_anchor: int,
    contact_friction_articulation_pairs_only: int,
    contact_friction_scale: float,
    contact_shared_anchor: int,
    friction_patches: FrictionPatches,
    friction_anchor_beta: float,
    unit_order: wp.array[int],
    world_unit_count: wp.array[int],
    unit_capacity: int,
    use_unit_order: int,
    # outputs
    propagation_body_a: wp.array2d[int],
    propagation_body_b: wp.array2d[int],
    propagation_J_a: wp.array3d[float],
    propagation_J_b: wp.array3d[float],
    propagation_row_type: wp.array2d[int],
    propagation_row_parent: wp.array2d[int],
    propagation_row_mu: wp.array2d[float],
    propagation_phi: wp.array2d[float],
    propagation_row_restitution: wp.array2d[float],
):
    """Build fixed-size body-space rows for contacts touching non-free articulations.

    With ``use_unit_order`` the launch runs over color-sorted unit positions
    (worlds x capacity) instead of raw contact indices, so row WRITES land
    sequentially in the color-ordered slot layout (reads scatter instead,
    avoiding the write-allocate round trip on the row arrays).
    """
    tid = wp.tid()
    c = tid
    if use_unit_order != 0:
        w = tid // unit_capacity
        pos = tid - w * unit_capacity
        if pos >= world_unit_count[w]:
            return
        c = unit_order[tid]
    total_contacts = contact_count[0]
    if c >= total_contacts:
        return

    if contact_path[c] != 2:
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

    thickness_a = contact_thickness0[c]
    thickness_b = contact_thickness1[c]

    point_a_local = contact_point0[c]
    point_b_local = contact_point1[c]
    point_a_world = wp.vec3(0.0)
    point_b_world = wp.vec3(0.0)

    if body_a >= 0:
        X_wb_a = body_q[body_a]
        point_a_world = wp.transform_point(X_wb_a, point_a_local) - thickness_a * normal
    else:
        point_a_world = point_a_local - thickness_a * normal

    if body_b >= 0:
        X_wb_b = body_q[body_b]
        point_b_world = wp.transform_point(X_wb_b, point_b_local) + thickness_b * normal
    else:
        point_b_world = point_b_local + thickness_b * normal

    phi = wp.dot(normal, point_a_world - point_b_world)

    mu = 0.0
    mat_count = 0
    if shape_a >= 0:
        mu += shape_material_mu[shape_a]
        mat_count += 1
    if shape_b >= 0:
        mu += shape_material_mu[shape_b]
        mat_count += 1
    if mat_count > 0:
        mu /= float(mat_count)
    restitution = mixed_contact_restitution(shape_a, shape_b, shape_material_restitution)

    art_a = contact_art_a[c]
    art_b = contact_art_b[c]
    a_non_free = art_a >= 0 and is_free_rigid[art_a] == 0
    b_non_free = art_b >= 0 and is_free_rigid[art_b] == 0
    friction_mu = mu * contact_friction_scale

    t0, t1 = contact_tangent_basis(normal)
    will_add_friction = enable_friction != 0 and contact_friction_eligible(
        phi, a_non_free, b_non_free, contact_friction_articulation_pairs_only, contact_friction_gap_threshold
    )
    if friction_patches.enabled != 0:
        will_add_friction = will_add_friction and friction_patches.weight[c] > 0.0
    contact_anchor_world = 0.5 * (point_a_world + point_b_world)

    com_a = wp.vec3(0.0)
    com_b = wp.vec3(0.0)
    if body_a >= 0:
        com_a = wp.transform_point(body_q[body_a], body_com[body_a])
    if body_b >= 0:
        com_b = wp.transform_point(body_q[body_b], body_com[body_b])

    for row_offset in range(3):
        if row_offset > 0 and not will_add_friction:
            break

        row_idx = slot + row_offset

        if row_offset == 0:
            d = normal
        elif row_offset == 1:
            d = t0
        else:
            d = t1

        if body_a >= 0:
            point_a_row_world = point_a_world
            if contact_shared_anchor != 0 or (row_offset > 0 and contact_friction_shared_anchor != 0):
                point_a_row_world = contact_anchor_world
            if row_offset > 0 and friction_patches.enabled != 0:
                point_a_row_world = friction_patches.point_a[c]
            r_a = point_a_row_world - com_a
            ang_a = wp.cross(r_a, d)
            propagation_J_a[world, row_idx, 0] = d[0]
            propagation_J_a[world, row_idx, 1] = d[1]
            propagation_J_a[world, row_idx, 2] = d[2]
            propagation_J_a[world, row_idx, 3] = ang_a[0]
            propagation_J_a[world, row_idx, 4] = ang_a[1]
            propagation_J_a[world, row_idx, 5] = ang_a[2]

        if body_b >= 0:
            point_b_row_world = point_b_world
            if contact_shared_anchor != 0 or (row_offset > 0 and contact_friction_shared_anchor != 0):
                point_b_row_world = contact_anchor_world
            if row_offset > 0 and friction_patches.enabled != 0:
                point_b_row_world = friction_patches.point_b[c]
            r_b = point_b_row_world - com_b
            ang_b = wp.cross(r_b, d)
            propagation_J_b[world, row_idx, 0] = -d[0]
            propagation_J_b[world, row_idx, 1] = -d[1]
            propagation_J_b[world, row_idx, 2] = -d[2]
            propagation_J_b[world, row_idx, 3] = -ang_b[0]
            propagation_J_b[world, row_idx, 4] = -ang_b[1]
            propagation_J_b[world, row_idx, 5] = -ang_b[2]

        propagation_body_a[world, row_idx] = body_a
        propagation_body_b[world, row_idx] = body_b

        if row_offset == 0:
            propagation_row_type[world, row_idx] = PGS_CONSTRAINT_TYPE_CONTACT
            propagation_row_parent[world, row_idx] = -1
            propagation_phi[world, row_idx] = phi
            propagation_row_mu[world, row_idx] = mu
            propagation_row_restitution[world, row_idx] = restitution
        else:
            propagation_row_type[world, row_idx] = PGS_CONSTRAINT_TYPE_FRICTION
            propagation_row_parent[world, row_idx] = slot
            if row_offset == 1:
                propagation_phi[world, row_idx] = friction_anchor_beta * friction_patches.phi[c][0]
            else:
                propagation_phi[world, row_idx] = friction_anchor_beta * friction_patches.phi[c][1]
            propagation_row_mu[world, row_idx] = friction_mu
            propagation_row_restitution[world, row_idx] = 0.0


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
    """Transport the cached tangent frame and enforce the current contact cone."""
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
def gather_mf_warmstart(
    contact_count: wp.array[int],
    contact_path: wp.array[int],
    contact_slot: wp.array[int],
    contact_world: wp.array[int],
    match_index: wp.array[int],  # rigid_contact_match_index (sorted-current -> prev-sorted idx)
    prev_slot_sorted: wp.array[int],  # prev-sorted contact idx -> prev base MF slot (or -1)
    prev_mf_impulses: wp.array2d[float],
    prev_mf_row_type: wp.array2d[int],
    prev_mf_row_parent: wp.array2d[int],
    mf_constraint_count: wp.array[int],
    mf_row_type: wp.array2d[int],  # THIS step's row types (already built)
    mf_row_parent: wp.array2d[int],
    contact_normal: wp.array[wp.vec3],
    previous_normal: wp.array[wp.vec3],
    mf_row_mu: wp.array2d[float],
    decay: float,
    dt_scale: float,
    mf_max_c: int,
    # in-out
    mf_impulses: wp.array2d[float],
):
    """Seed this step's MF impulse buffer from the previous step's converged
    impulses, matched by contact identity.

    Runs after ``allocate_world_contact_slots`` + ``build_mf_contact_rows`` (so
    ``contact_slot`` and ``mf_row_type`` are populated for this step) and after
    the warm-start branch has zeroed ``mf_impulses``. One thread per contact;
    each writes only its own disjoint slot range, so unmatched / cold contacts
    keep the zero left by the memset and no stale slot survives.

    ``dt_scale`` = dt_now / dt_prev rescales carried impulses across step-size
    changes (impulse is proportional to dt for quasi-static loads); 1.0 at fixed dt.

    ``match_index[c] >= 0`` is the previous frame's *sorted* contact index;
    ``prev_slot_sorted`` is keyed the same way (see solver writeback), so
    ``prev_slot = prev_slot_sorted[match_index[c]]`` is the base slot that
    contact occupied last step. Friction rows are only seeded when BOTH this
    step and the previous step allocated a friction row at the corresponding
    offset (guards the variable 1-vs-3 stride).
    """
    c = wp.tid()
    if c >= contact_count[0]:
        return
    if contact_path[c] != 1:  # MF path only
        return
    new_slot = contact_slot[c]
    if new_slot < 0:
        return

    world = contact_world[c]
    count = mf_constraint_count[world]
    if new_slot >= count:
        return

    mi = match_index[c]
    matched = mi >= 0
    prev_slot = int(-1)
    if matched:
        prev_slot = prev_slot_sorted[mi]

    # Normal row (offset 0): always present for an MF contact.
    if (
        matched
        and prev_slot >= 0
        and prev_slot < mf_max_c
        and mf_row_type[world, new_slot] == PGS_CONSTRAINT_TYPE_CONTACT
        and prev_mf_row_type[world, prev_slot] == PGS_CONSTRAINT_TYPE_CONTACT
    ):
        mf_impulses[world, new_slot] = decay * dt_scale * prev_mf_impulses[world, prev_slot]
    # else: leave 0 (already memset)

    # Friction rows (offsets 1..2): only if THIS step allocated them here and
    # they belong to this contact's block (parent == base slot).
    for r in range(1, 3):
        new_r = new_slot + r
        if new_r < count and new_r < mf_max_c:
            if mf_row_type[world, new_r] == PGS_CONSTRAINT_TYPE_FRICTION and mf_row_parent[world, new_r] == new_slot:
                prev_r = prev_slot + r
                if (
                    matched
                    and prev_slot >= 0
                    and prev_r < mf_max_c
                    and prev_mf_row_type[world, prev_r] == PGS_CONSTRAINT_TYPE_FRICTION
                    and prev_mf_row_parent[world, prev_r] == prev_slot
                ):
                    mf_impulses[world, new_r] = decay * dt_scale * prev_mf_impulses[world, prev_r]
                # else: leave 0

    if mi >= 0 and prev_slot >= 0 and prev_slot < mf_max_c:
        _transport_warmstart_contact(
            contact_normal[c],
            previous_normal[mi],
            world,
            new_slot,
            count,
            mf_row_type,
            mf_row_parent,
            mf_row_mu,
            mf_impulses,
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
    """Save all contact routes and live normals in one launch."""
    contact = wp.tid()
    count = contact_count[0]
    active = contact < count and count <= contact_normal.shape[0]
    path = int(-1)
    slot = int(-1)
    if active:
        path = contact_path[contact]
        slot = contact_slot[contact]
        previous_normal[contact] = contact_normal[contact]
    if contact < previous_dense_slot.shape[0]:
        previous_dense_slot[contact] = wp.where(path == 0, slot, -1)
    if contact < previous_mf_slot.shape[0]:
        previous_mf_slot[contact] = wp.where(path == 1, slot, -1)
    if contact < previous_propagation_slot.shape[0]:
        previous_propagation_slot[contact] = wp.where(path == 2, slot, -1)


@wp.kernel
def snapshot_mf_prev_slots(
    contact_count: wp.array[int],
    contact_path: wp.array[int],
    contact_slot: wp.array[int],
    # out
    prev_slot_sorted: wp.array[int],
):
    """Record, per current sorted contact index, the base MF slot it occupied
    this step (or -1 if it was not MF-routed / inactive). Run at the END of the
    step so next step's ``rigid_contact_match_index`` (referencing this frame's
    *sorted* index) resolves through it. One thread per contact-array slot;
    indices beyond ``contact_count`` are written -1 so stale entries from a
    larger previous frame can't leak.
    """
    c = wp.tid()
    if c >= contact_count[0]:
        prev_slot_sorted[c] = -1
        return
    if contact_path[c] != 1:
        prev_slot_sorted[c] = -1
        return
    slot = contact_slot[c]
    prev_slot_sorted[c] = slot


@wp.kernel
def snapshot_dense_prev_slots(
    contact_count: wp.array[int],
    contact_path: wp.array[int],
    contact_slot: wp.array[int],
    # out
    prev_slot_sorted: wp.array[int],
):
    """Dense-path mirror of :func:`snapshot_mf_prev_slots`.

    Records, per current sorted contact index, the absolute dense world row the
    contact's normal row occupied this step (or -1 when it was not
    dense-routed / inactive), so next step's ``rigid_contact_match_index`` can
    resolve previous impulses by contact identity.
    """
    c = wp.tid()
    if c >= contact_count[0]:
        prev_slot_sorted[c] = -1
        return
    if contact_path[c] != 0:
        prev_slot_sorted[c] = -1
        return
    prev_slot_sorted[c] = contact_slot[c]


@wp.kernel
def snapshot_propagation_prev_slots(
    contact_count: wp.array[int],
    contact_path: wp.array[int],
    contact_slot: wp.array[int],
    # out
    prev_slot_sorted: wp.array[int],
):
    """Propagation-path mirror of :func:`snapshot_mf_prev_slots`."""
    c = wp.tid()
    if c >= contact_count[0]:
        prev_slot_sorted[c] = -1
        return
    if contact_path[c] != 2:
        prev_slot_sorted[c] = -1
        return
    prev_slot_sorted[c] = contact_slot[c]


@wp.kernel
def gather_dense_warmstart(
    contact_count: wp.array[int],
    contact_path: wp.array[int],
    contact_slot: wp.array[int],
    contact_world: wp.array[int],
    match_index: wp.array[int],  # rigid_contact_match_index (sorted-current -> prev-sorted idx)
    prev_slot_sorted: wp.array[int],  # prev-sorted contact idx -> prev dense normal row (or -1)
    prev_dense_impulses: wp.array2d[float],
    prev_dense_row_type: wp.array2d[int],
    prev_dense_row_parent: wp.array2d[int],
    world_constraint_count: wp.array[int],
    world_row_type: wp.array2d[int],  # THIS step's dense row types (already built)
    world_row_parent: wp.array2d[int],
    contact_normal: wp.array[wp.vec3],
    previous_normal: wp.array[wp.vec3],
    world_row_mu: wp.array2d[float],
    decay: float,
    dt_scale: float,
    max_constraints: int,
    # in-out
    world_impulses: wp.array2d[float],
):
    """Seed this step's DENSE impulse buffer from the previous step's converged
    impulses, matched by contact identity — the dense mirror of
    :func:`gather_mf_warmstart`.

    Runs after ``prepare_world_impulses`` has cold-zeroed every dense row, so
    unmatched or fresh contacts stay cold and no stale impulse survives a
    slot-layout shift. Friction rows
    (offsets 1, 2 after the normal row) are seeded only when BOTH this step and
    the previous step allocated a friction row at the corresponding offset.
    ``decay`` scales every seeded impulse: at persistent quasi-static contacts
    it damps the accumulate-on-impact overshoot while keeping the sweep's
    starting point near the converged solution.
    """
    c = wp.tid()
    if c >= contact_count[0]:
        return
    if contact_path[c] != 0:  # dense path only
        return
    new_slot = contact_slot[c]
    if new_slot < 0:
        return

    world = contact_world[c]
    count = world_constraint_count[world]
    if new_slot >= count:
        return

    mi = match_index[c]
    matched = mi >= 0
    prev_slot = int(-1)
    if matched:
        prev_slot = prev_slot_sorted[mi]

    # Normal row (offset 0).
    if (
        matched
        and prev_slot >= 0
        and prev_slot < max_constraints
        and prev_dense_row_type[world, prev_slot] == PGS_CONSTRAINT_TYPE_CONTACT
    ):
        if world_row_type[world, new_slot] == PGS_CONSTRAINT_TYPE_CONTACT:
            world_impulses[world, new_slot] = decay * dt_scale * prev_dense_impulses[world, prev_slot]

    # Friction rows (offsets 1, 2): only if THIS step allocated them here.
    for r in range(1, 3):
        new_r = new_slot + r
        if new_r < count and new_r < max_constraints:
            if (
                world_row_type[world, new_r] == PGS_CONSTRAINT_TYPE_FRICTION
                and world_row_parent[world, new_r] == new_slot
            ):
                prev_r = prev_slot + r
                if (
                    matched
                    and prev_slot >= 0
                    and prev_r < max_constraints
                    and prev_dense_row_type[world, prev_r] == PGS_CONSTRAINT_TYPE_FRICTION
                    and prev_dense_row_parent[world, prev_r] == prev_slot
                ):
                    world_impulses[world, new_r] = decay * dt_scale * prev_dense_impulses[world, prev_r]

    if mi >= 0 and prev_slot >= 0 and prev_slot < max_constraints:
        _transport_warmstart_contact(
            contact_normal[c],
            previous_normal[mi],
            world,
            new_slot,
            count,
            world_row_type,
            world_row_parent,
            world_row_mu,
            world_impulses,
        )


@wp.kernel
def gather_propagation_warmstart(
    contact_count: wp.array[int],
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
    dt_scale: float,
    max_constraints: int,
    # in-out
    impulses: wp.array2d[float],
):
    """Identity-gather converged impulses for propagation-routed contacts."""
    c = wp.tid()
    if c >= contact_count[0] or contact_path[c] != 2:
        return
    new_slot = contact_slot[c]
    if new_slot < 0:
        return
    world = contact_world[c]
    count = constraint_count[world]
    if new_slot >= count:
        return

    mi = match_index[c]
    prev_slot = int(-1)
    if mi >= 0:
        prev_slot = prev_slot_sorted[mi]

    if (
        prev_slot >= 0
        and prev_slot < max_constraints
        and row_type[world, new_slot] == PGS_CONSTRAINT_TYPE_CONTACT
        and prev_row_type[world, prev_slot] == PGS_CONSTRAINT_TYPE_CONTACT
    ):
        impulses[world, new_slot] = decay * dt_scale * prev_impulses[world, prev_slot]

    for r in range(1, 3):
        new_r = new_slot + r
        prev_r = prev_slot + r
        if (
            new_r < count
            and new_r < max_constraints
            and prev_slot >= 0
            and prev_r < max_constraints
            and row_type[world, new_r] == PGS_CONSTRAINT_TYPE_FRICTION
            and row_parent[world, new_r] == new_slot
            and prev_row_type[world, prev_r] == PGS_CONSTRAINT_TYPE_FRICTION
            and prev_row_parent[world, prev_r] == prev_slot
        ):
            impulses[world, new_r] = decay * dt_scale * prev_impulses[world, prev_r]

    if mi >= 0 and prev_slot >= 0 and prev_slot < max_constraints:
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
def accumulate_propagation_warmstart_body_impulses(
    constraint_count: wp.array[int],
    body_a: wp.array2d[int],
    body_b: wp.array2d[int],
    J_a: wp.array3d[float],
    J_b: wp.array3d[float],
    MiJt_a: wp.array3d[float],
    MiJt_b: wp.array3d[float],
    impulses: wp.array2d[float],
    max_constraints: int,
    # in-out
    body_qd: wp.array2d[float],
    body_impulses: wp.array2d[float],
):
    """Convert seeded propagation row impulses to deferred body wrenches."""
    tid = wp.tid()
    world = tid // max_constraints
    row = tid - world * max_constraints
    if row >= constraint_count[world]:
        return
    impulse = impulses[world, row]
    if impulse == 0.0:
        return
    ba = body_a[world, row]
    bb = body_b[world, row]
    for k in range(6):
        if ba >= 0:
            wp.atomic_add(body_qd, ba, k, MiJt_a[world, row, k] * impulse)
            wp.atomic_add(body_impulses, ba, k, J_a[world, row, k] * impulse)
        if bb >= 0:
            wp.atomic_add(body_qd, bb, k, MiJt_b[world, row, k] * impulse)
            wp.atomic_add(body_impulses, bb, k, J_b[world, row, k] * impulse)


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
    # outputs
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
    # outputs
    mf_body_Hinv: wp.array[wp.spatial_matrix],
):
    """Compute H^-1 = inverse(body_I_s) for free rigid bodies.

    For root free joints, H = body_I_s in articulation-local coordinates.
    This remains a full 6x6 matrix for bodies with non-zero CoM offsets.
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

    mf_body_Hinv[b] = spatial_matrix_block_inverse(body_I_s[b])


@wp.kernel
def copy_free_rigid_propagation_body_response(
    is_free_rigid: wp.array[int],
    body_to_articulation: wp.array[int],
    mf_body_Hinv: wp.array[wp.spatial_matrix],
    # outputs
    propagation_body_response: wp.array3d[float],
):
    """Seed propagation's 6D body response for free rigid bodies.

    Propagation rows use ``propagation_body_response`` for both contact sides.
    Non-free articulated links are filled by the tree-response setup. Free rigid
    bodies do not run that tree solve, so copy the same inverse spatial inertia
    used by the existing free/free MF path.
    """
    body = wp.tid()
    art = body_to_articulation[body]
    if art < 0:
        return
    if is_free_rigid[art] == 0:
        return

    Hinv = mf_body_Hinv[body]
    for r in range(6):
        for c in range(6):
            propagation_body_response[body, r, c] = Hinv[r, c]


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
    """Compute effective mass diagonal, H^-1*J^T, and RHS bias for MF constraints.

    The effective mass for constraint i is:
        d_ii = J_a^T * H_a_inv * J_a + J_b^T * H_b_inv * J_b + cfm

    H_inv is the full 6x6 inverse of the spatial inertia in articulation-local
    coordinates for each free rigid articulation.

    RHS stores only the stabilization bias (not J*v), since the MF PGS
    recomputes J*v each iteration from the live velocity array.
    """
    tid = wp.tid()
    world = tid // mf_max_constraints
    i = tid % mf_max_constraints
    if i >= mf_constraint_count[world]:
        return

    ba = mf_body_a[world, i]
    bb = mf_body_b[world, i]

    # Load Jacobian as spatial_vector
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

    # Side A: MiJt_a = H_a_inv * J_a, d += J_a^T * MiJt_a
    if ba >= 0:
        Hinv_a = mf_body_Hinv[ba]
        MiJt_a = Hinv_a * Ja
        d += wp.dot(Ja, MiJt_a)
        mf_MiJt_a[world, i, 0] = MiJt_a[0]
        mf_MiJt_a[world, i, 1] = MiJt_a[1]
        mf_MiJt_a[world, i, 2] = MiJt_a[2]
        mf_MiJt_a[world, i, 3] = MiJt_a[3]
        mf_MiJt_a[world, i, 4] = MiJt_a[4]
        mf_MiJt_a[world, i, 5] = MiJt_a[5]

    # Side B
    if bb >= 0:
        Hinv_b = mf_body_Hinv[bb]
        MiJt_b = Hinv_b * Jb
        d += wp.dot(Jb, MiJt_b)
        mf_MiJt_b[world, i, 0] = MiJt_b[0]
        mf_MiJt_b[world, i, 1] = MiJt_b[1]
        mf_MiJt_b[world, i, 2] = MiJt_b[2]
        mf_MiJt_b[world, i, 3] = MiJt_b[3]
        mf_MiJt_b[world, i, 4] = MiJt_b[4]
        mf_MiJt_b[world, i, 5] = MiJt_b[5]

    if d > 0.0:
        mf_eff_mass_inv[world, i] = 1.0 / d
    else:
        mf_eff_mass_inv[world, i] = 0.0

    # Contact bias only (not J*v -- recomputed each PGS iter). Penetrating
    # contacts use Baumgarte stabilization; separated speculative contacts
    # allow closing by the current positive gap over this substep.
    bias = float(0.0)
    row_w = float(1.0)
    rtype = mf_row_type[world, i]
    if rtype == PGS_CONSTRAINT_TYPE_CONTACT:
        phi_val = mf_phi[world, i]
        if phi_val <= 0.0:
            # Positive-gap (speculative) rows stay rigid so a closing contact
            # reaches the surface instead of leaking closing speed into penetration.
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
            target_vel = float(0.0)
            if has_target_velocity != 0:
                target_vel = mf_target_velocity[world, i]
            relative_incident -= target_vel
            if contact_restitution_fires(phi_val, relative_incident, dt, restitution_velocity_threshold):
                bias = restitution * relative_incident
                # An impact is impulsive, not a spring: keep the rebound exact.
                row_w = 1.0
    elif rtype == PGS_CONSTRAINT_TYPE_FRICTION:
        bias = mf_phi[world, i] / dt  # anchor gain pre-multiplied by the row builder
    elif rtype == PGS_CONSTRAINT_TYPE_JOINT_VELOCITY_LIMIT:
        bias = mf_phi[world, i]

    if has_target_velocity != 0:
        bias -= mf_target_velocity[world, i]
    mf_rhs[world, i] = bias
    if contact_w < 1.0:
        mf_row_w[world, i] = row_w


@wp.kernel
def compute_mf_rhs_bias(
    mf_constraint_count: wp.array[int],
    mf_body_a: wp.array2d[int],
    mf_body_b: wp.array2d[int],
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
    rigid_body_max_depenetration_velocity: wp.array[float],
    pgs_beta: float,
    dt: float,
    bias_scale: float,
    speculative_scale: float,
    position_velocity: wp.array[float],
    incident_velocity: wp.array[float],
    preserve_unreached_speculative: int,
    apply_restitution: int,
    restitution_velocity_threshold: float,
    mf_max_constraints: int,
    # outputs
    mf_rhs: wp.array2d[float],
):
    """Compute only the MF contact RHS bias for a previously-built contact set."""
    tid = wp.tid()
    world = tid // mf_max_constraints
    i = tid % mf_max_constraints
    if i >= mf_constraint_count[world]:
        return

    bias = float(0.0)
    row_type = mf_row_type[world, i]
    if row_type == PGS_CONSTRAINT_TYPE_CONTACT:
        phi_val = mf_phi[world, i]
        restitution = float(0.0)
        relative_incident = float(0.0)
        bounce = int(0)
        if apply_restitution != 0:
            restitution = mf_row_restitution[world, i]
        if restitution > 0.0:
            relative_incident = mf_contact_row_dot(
                mf_J_a, mf_J_b, mf_dof_a[world, i], mf_dof_b[world, i], world_dof_indices, incident_velocity, world, i
            )
            target_velocity = float(0.0)
            if has_target_velocity != 0:
                target_velocity = mf_target_velocity[world, i]
            relative_incident -= target_velocity
            if contact_restitution_fires(phi_val, relative_incident, dt, restitution_velocity_threshold):
                bounce = int(1)

        if bounce != 0:
            bias = restitution * relative_incident
        elif phi_val < 0.0:
            ba = mf_body_a[world, i]
            bb = mf_body_b[world, i]
            bias = bias_scale * pgs_beta * phi_val / dt
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
            end_gap = float(0.0)
            if preserve_unreached_speculative != 0 and phi_val > 0.0:
                # Evaluate the same linearized end gap constrained by the
                # position solve, using its realized velocity.
                jv_position = mf_contact_row_dot(
                    mf_J_a,
                    mf_J_b,
                    mf_dof_a[world, i],
                    mf_dof_b[world, i],
                    world_dof_indices,
                    position_velocity,
                    world,
                    i,
                )
                target_velocity = float(0.0)
                if has_target_velocity != 0:
                    target_velocity = mf_target_velocity[world, i]
                end_gap = phi_val + dt * (jv_position - target_velocity)
            if preserve_unreached_speculative != 0 and end_gap > _FPGS_CONTACT_END_GAP_SLOP:
                bias = phi_val / dt
            else:
                bias = speculative_scale * phi_val / dt
    elif row_type == PGS_CONSTRAINT_TYPE_FRICTION:
        bias = bias_scale * mf_phi[world, i] / dt  # anchor gain pre-multiplied by the row builder
    elif row_type == PGS_CONSTRAINT_TYPE_JOINT_VELOCITY_LIMIT:
        bias = mf_phi[world, i]

    if has_target_velocity != 0:
        bias -= mf_target_velocity[world, i]
    mf_rhs[world, i] = bias


@wp.kernel
def compute_propagation_effective_mass_and_rhs(
    propagation_constraint_count: wp.array[int],
    propagation_body_a: wp.array2d[int],
    propagation_body_b: wp.array2d[int],
    propagation_J_a: wp.array3d[float],
    propagation_J_b: wp.array3d[float],
    propagation_body_response: wp.array3d[float],
    propagation_phi: wp.array2d[float],
    propagation_row_type: wp.array2d[int],
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
    propagation_restitution_target: wp.array2d[float],
    propagation_row_w: wp.array2d[float],
):
    tid = wp.tid()
    world = tid // propagation_max_constraints
    i = tid - world * propagation_max_constraints
    if i >= propagation_constraint_count[world]:
        return

    ba = propagation_body_a[world, i]
    bb = propagation_body_b[world, i]
    row_type = propagation_row_type[world, i]

    d = pgs_cfm

    if ba >= 0:
        for r in range(6):
            value = float(0.0)
            for c in range(6):
                value += propagation_body_response[ba, r, c] * propagation_J_a[world, i, c]
            propagation_MiJt_a[world, i, r] = value
            d += propagation_J_a[world, i, r] * value

    if bb >= 0:
        for r in range(6):
            value = float(0.0)
            for c in range(6):
                value += propagation_body_response[bb, r, c] * propagation_J_b[world, i, c]
            propagation_MiJt_b[world, i, r] = value
            d += propagation_J_b[world, i, r] * value

    if d > 0.0:
        propagation_eff_mass_inv[world, i] = 1.0 / d
    else:
        propagation_eff_mass_inv[world, i] = 0.0

    bias = float(0.0)
    restitution_target = float(0.0)
    row_w = float(1.0)
    if row_type == PGS_CONSTRAINT_TYPE_CONTACT:
        phi_val = propagation_phi[world, i]
        if phi_val <= 0.0:
            # Positive-gap (speculative) rows stay rigid so a closing contact
            # reaches the surface instead of leaking closing speed into penetration.
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
        restitution = propagation_row_restitution[world, i]
        if restitution > 0.0:
            relative_incident = propagation_contact_row_dot(
                propagation_J_a, propagation_J_b, propagation_body_qd, world, i, ba, bb
            )
            if contact_restitution_fires(phi_val, relative_incident, dt, restitution_velocity_threshold):
                restitution_target = -restitution * relative_incident
                # The propagation solve adds live J*v to this bias, so the
                # Newton target replaces (rather than augments) speculative or
                # Baumgarte geometry for a qualifying impact.
                bias = -restitution_target
                row_w = 1.0
    if row_type == PGS_CONSTRAINT_TYPE_FRICTION:
        bias = propagation_phi[world, i] / dt  # anchor gain pre-multiplied by the row builder
    propagation_rhs[world, i] = bias
    propagation_restitution_target[world, i] = restitution_target
    if contact_w < 1.0:
        propagation_row_w[world, i] = row_w


@wp.kernel
def compute_propagation_rhs_bias(
    propagation_constraint_count: wp.array[int],
    propagation_body_a: wp.array2d[int],
    propagation_body_b: wp.array2d[int],
    propagation_J_a: wp.array3d[float],
    propagation_J_b: wp.array3d[float],
    propagation_phi: wp.array2d[float],
    propagation_row_type: wp.array2d[int],
    propagation_restitution_target: wp.array2d[float],
    rigid_body_max_depenetration_velocity: wp.array[float],
    pgs_beta: float,
    dt: float,
    bias_scale: float,
    speculative_scale: float,
    position_body_qd: wp.array2d[float],
    preserve_unreached_speculative: int,
    apply_restitution: int,
    propagation_max_constraints: int,
    # outputs
    propagation_rhs: wp.array2d[float],
):
    tid = wp.tid()
    world = tid // propagation_max_constraints
    i = tid - world * propagation_max_constraints
    if i >= propagation_constraint_count[world]:
        return

    bias = float(0.0)
    row_type = propagation_row_type[world, i]
    if row_type == PGS_CONSTRAINT_TYPE_CONTACT:
        reached = int(1)
        phi_val = propagation_phi[world, i]
        if phi_val < 0.0:
            ba = propagation_body_a[world, i]
            bb = propagation_body_b[world, i]
            bias = bias_scale * pgs_beta * phi_val / dt
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
            end_gap = float(0.0)
            if preserve_unreached_speculative != 0 and phi_val > 0.0:
                jv_position = propagation_contact_row_dot(
                    propagation_J_a,
                    propagation_J_b,
                    position_body_qd,
                    world,
                    i,
                    propagation_body_a[world, i],
                    propagation_body_b[world, i],
                )
                end_gap = phi_val + dt * jv_position
            if preserve_unreached_speculative != 0 and end_gap > _FPGS_CONTACT_END_GAP_SLOP:
                bias = phi_val / dt
                reached = int(0)
            else:
                bias = speculative_scale * phi_val / dt
        restitution_target = propagation_restitution_target[world, i]
        if apply_restitution != 0 and restitution_target > 0.0:
            # This target was frozen before the position solve and already
            # includes the impact/reach predicate.  A rebound can leave the
            # realized end gap positive, so do not reclassify it as an
            # unreached speculative row here.
            bias = -restitution_target
        elif apply_restitution != 0 and reached != 0:
            bias -= propagation_restitution_target[world, i]
    if row_type == PGS_CONSTRAINT_TYPE_FRICTION:
        bias = bias_scale * propagation_phi[world, i] / dt  # anchor gain pre-multiplied by the row builder
    propagation_rhs[world, i] = bias


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
        old = wp.atomic_add(propagation_body_seen, ba, 1)
        if old == 0:
            slot = wp.atomic_add(propagation_body_count, world, 1)
            if slot < max_propagation_bodies:
                propagation_body_list[world, slot] = ba
                propagation_body_local_slot[ba] = slot

    bb = propagation_body_b[world, i]
    if bb >= 0:
        old = wp.atomic_add(propagation_body_seen, bb, 1)
        if old == 0:
            slot = wp.atomic_add(propagation_body_count, world, 1)
            if slot < max_propagation_bodies:
                propagation_body_list[world, slot] = bb
                propagation_body_local_slot[bb] = slot


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
    body = wp.tid()
    art = body_to_articulation[body]
    if art < 0:
        return

    com_world = wp.transform_point(body_q[body], body_com[body])
    rel = com_world - articulation_origin[art]
    propagation_body_com_rel[body, 0] = rel[0]
    propagation_body_com_rel[body, 1] = rel[1]
    propagation_body_com_rel[body, 2] = rel[2]


@wp.kernel
def flatten_propagation_joint_S(
    joint_child: wp.array[int],
    joint_qd_start: wp.array[int],
    joint_S_s: wp.array[wp.spatial_vector],
    propagation_body_com_rel: wp.array2d[float],
    # outputs
    propagation_joint_S_flat: wp.array2d[float],
):
    joint = wp.tid()
    child = joint_child[joint]
    if child < 0:
        return

    child_rel = wp.vec3(
        propagation_body_com_rel[child, 0],
        propagation_body_com_rel[child, 1],
        propagation_body_com_rel[child, 2],
    )
    dof_start = joint_qd_start[joint]
    dof_end = joint_qd_start[joint + 1]
    for dof in range(dof_start, dof_end):
        S = joint_S_s[dof]
        lin = wp.vec3(S[0], S[1], S[2])
        ang = wp.vec3(S[3], S[4], S[5])
        lin_child = lin + wp.cross(ang, child_rel)
        propagation_joint_S_flat[dof, 0] = lin_child[0]
        propagation_joint_S_flat[dof, 1] = lin_child[1]
        propagation_joint_S_flat[dof, 2] = lin_child[2]
        propagation_joint_S_flat[dof, 3] = ang[0]
        propagation_joint_S_flat[dof, 4] = ang[1]
        propagation_joint_S_flat[dof, 5] = ang[2]


@wp.kernel
def factor_propagation_tree_for_size(
    group_to_art: wp.array[int],
    articulation_start: wp.array[int],
    articulation_dof_start: wp.array[int],
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
    """Factor one size group into articulated-body inertia terms.

    This is the tree-space equivalent of the dense CRBA+Cholesky response used
    by the diagnostic path. It stores per-link articulated inertia, U = I_a S,
    and D^-1 for each inbound joint without constructing D-wide contact rows.
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

        dof_start = joint_qd_start[joint]
        dof_end = joint_qd_start[joint + 1]
        for dof in range(dof_start, dof_end):
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
        lin_count = joint_dof_dim[joint, 0]
        ang_count = joint_dof_dim[joint, 1]
        dof_count = lin_count + ang_count

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
                    aug_count = aug_row_counts[art]
                    for aug_i in range(aug_count):
                        row_index = art * max_dofs + aug_i
                        if aug_row_dof_index[row_index] == gdof_a:
                            K = aug_row_K[row_index]
                            if K > 0.0:
                                value += K
                propagation_tree_D_chol[joint, a, b] = value

        # Cholesky factorization of the small joint-space block D.
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

        # Invert D one column at a time using the Cholesky factor. D_inv[:, col]
        # first holds the forward solve, then the final inverse column.
        for col in range(dof_count):
            for i in range(dof_count):
                v = float(0.0)
                if i == col:
                    v = 1.0
                for k in range(i):
                    v -= propagation_tree_D_chol[joint, i, k] * propagation_tree_D_inv[joint, k, col]
                diag = propagation_tree_D_chol[joint, i, i]
                propagation_tree_D_inv[joint, i, col] = v / diag

            for i_rev in range(dof_count):
                i = dof_count - 1 - i_rev
                v = propagation_tree_D_inv[joint, i, col]
                for k in range(i + 1, dof_count):
                    v -= propagation_tree_D_chol[joint, k, i] * propagation_tree_D_inv[joint, k, col]
                diag = propagation_tree_D_chol[joint, i, i]
                propagation_tree_D_inv[joint, i, col] = v / diag

        # Reduce the child's articulated inertia across this joint in the child
        # COM frame.
        for r in range(6):
            for c in range(6):
                reduced = propagation_tree_Ia[child, r, c]
                for a in range(dof_count):
                    gdof_a = dof_start + a
                    U_ar = propagation_tree_U[gdof_a, r]
                    for b in range(dof_count):
                        gdof_b = dof_start + b
                        reduced -= U_ar * propagation_tree_D_inv[joint, a, b] * propagation_tree_U[gdof_b, c]
                propagation_tree_Ia[child, r, c] = reduced

        if parent >= 0:
            child_rel = wp.vec3(
                propagation_body_com_rel[child, 0],
                propagation_body_com_rel[child, 1],
                propagation_body_com_rel[child, 2],
            )
            parent_rel = wp.vec3(
                propagation_body_com_rel[parent, 0],
                propagation_body_com_rel[parent, 1],
                propagation_body_com_rel[parent, 2],
            )
            child_minus_parent = child_rel - parent_rel
            for c in range(6):
                basis = wp.spatial_vector()
                if c == 0:
                    basis = wp.spatial_vector(1.0, 0.0, 0.0, 0.0, 0.0, 0.0)
                elif c == 1:
                    basis = wp.spatial_vector(0.0, 1.0, 0.0, 0.0, 0.0, 0.0)
                elif c == 2:
                    basis = wp.spatial_vector(0.0, 0.0, 1.0, 0.0, 0.0, 0.0)
                elif c == 3:
                    basis = wp.spatial_vector(0.0, 0.0, 0.0, 1.0, 0.0, 0.0)
                elif c == 4:
                    basis = wp.spatial_vector(0.0, 0.0, 0.0, 0.0, 1.0, 0.0)
                else:
                    basis = wp.spatial_vector(0.0, 0.0, 0.0, 0.0, 0.0, 1.0)

                v_child = translate_twist_between_parallel_frames(basis, child_minus_parent)
                I_child = wp.spatial_matrix(
                    propagation_tree_Ia[child, 0, 0],
                    propagation_tree_Ia[child, 0, 1],
                    propagation_tree_Ia[child, 0, 2],
                    propagation_tree_Ia[child, 0, 3],
                    propagation_tree_Ia[child, 0, 4],
                    propagation_tree_Ia[child, 0, 5],
                    propagation_tree_Ia[child, 1, 0],
                    propagation_tree_Ia[child, 1, 1],
                    propagation_tree_Ia[child, 1, 2],
                    propagation_tree_Ia[child, 1, 3],
                    propagation_tree_Ia[child, 1, 4],
                    propagation_tree_Ia[child, 1, 5],
                    propagation_tree_Ia[child, 2, 0],
                    propagation_tree_Ia[child, 2, 1],
                    propagation_tree_Ia[child, 2, 2],
                    propagation_tree_Ia[child, 2, 3],
                    propagation_tree_Ia[child, 2, 4],
                    propagation_tree_Ia[child, 2, 5],
                    propagation_tree_Ia[child, 3, 0],
                    propagation_tree_Ia[child, 3, 1],
                    propagation_tree_Ia[child, 3, 2],
                    propagation_tree_Ia[child, 3, 3],
                    propagation_tree_Ia[child, 3, 4],
                    propagation_tree_Ia[child, 3, 5],
                    propagation_tree_Ia[child, 4, 0],
                    propagation_tree_Ia[child, 4, 1],
                    propagation_tree_Ia[child, 4, 2],
                    propagation_tree_Ia[child, 4, 3],
                    propagation_tree_Ia[child, 4, 4],
                    propagation_tree_Ia[child, 4, 5],
                    propagation_tree_Ia[child, 5, 0],
                    propagation_tree_Ia[child, 5, 1],
                    propagation_tree_Ia[child, 5, 2],
                    propagation_tree_Ia[child, 5, 3],
                    propagation_tree_Ia[child, 5, 4],
                    propagation_tree_Ia[child, 5, 5],
                )
                w_child = I_child * v_child
                w_parent = translate_wrench_between_parallel_frames(w_child, child_minus_parent)
                for r in range(6):
                    propagation_tree_Ia[parent, r, c] = propagation_tree_Ia[parent, r, c] + w_parent[r]


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
    propagation_body_local_slot: wp.array[int],
    propagation_constraint_count: wp.array[int],
    propagation_body_a: wp.array2d[int],
    propagation_body_b: wp.array2d[int],
    propagation_J_a: wp.array3d[float],
    propagation_J_b: wp.array3d[float],
    propagation_phi: wp.array2d[float],
    propagation_row_type: wp.array2d[int],
    pgs_cfm: float,
    dt: float,
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
):
    """Exact response for rows whose two bodies are links of one articulation.

    The per-link diagonal response misses the cross operational-space term
    J_a (X_a H^-1 X_b^T) J_b^T. Here the row's combined test impulse (J_a at
    body a, J_b at body b) is propagated once through the articulated-body
    factorization and the resulting link velocity changes replace MiJt_a/b,
    so the effective mass picks up the cross terms exactly. Runs one thread
    per articulation, serial over its same-articulation rows, reusing the
    tree response scratch (launched after the body-response and effective
    mass kernels)."""
    art = wp.tid()
    if is_free_rigid[art] != 0:
        return
    world = art_to_world[art]
    m_count = propagation_constraint_count[world]
    if m_count > propagation_max_constraints:
        m_count = propagation_max_constraints
    joint_start = articulation_start[art]
    joint_end = articulation_start[art + 1]

    for i in range(m_count):
        ba = propagation_body_a[world, i]
        bb = propagation_body_b[world, i]
        if ba < 0 or bb < 0:
            continue
        if body_to_articulation[ba] != art or body_to_articulation[bb] != art:
            continue
        if propagation_body_local_slot[ba] < 0 or propagation_body_local_slot[bb] < 0:
            continue

        for joint in range(joint_start, joint_end):
            body = joint_child[joint]
            for r in range(6):
                propagation_tree_pA[body, r] = 0.0
                propagation_tree_body_delta[body, r] = 0.0
            dof_start = joint_qd_start[joint]
            dof_end = joint_qd_start[joint + 1]
            for dof in range(dof_start, dof_end):
                propagation_tree_u[dof] = 0.0
                propagation_tree_qdd[dof] = 0.0

        for r in range(6):
            propagation_tree_pA[ba, r] = propagation_tree_pA[ba, r] - propagation_J_a[world, i, r]
            propagation_tree_pA[bb, r] = propagation_tree_pA[bb, r] - propagation_J_b[world, i, r]

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
                p0 = propagation_tree_pA[child, 0]
                p1 = propagation_tree_pA[child, 1]
                p2 = propagation_tree_pA[child, 2]
                p3 = propagation_tree_pA[child, 3]
                p4 = propagation_tree_pA[child, 4]
                p5 = propagation_tree_pA[child, 5]
                for a in range(dof_count):
                    gdof_a = dof_start + a
                    coeff = float(0.0)
                    for b in range(dof_count):
                        gdof_b = dof_start + b
                        coeff += propagation_tree_D_inv[joint, a, b] * propagation_tree_u[gdof_b]
                    p0 += propagation_tree_U[gdof_a, 0] * coeff
                    p1 += propagation_tree_U[gdof_a, 1] * coeff
                    p2 += propagation_tree_U[gdof_a, 2] * coeff
                    p3 += propagation_tree_U[gdof_a, 3] * coeff
                    p4 += propagation_tree_U[gdof_a, 4] * coeff
                    p5 += propagation_tree_U[gdof_a, 5] * coeff

                child_rel = wp.vec3(
                    propagation_body_com_rel[child, 0],
                    propagation_body_com_rel[child, 1],
                    propagation_body_com_rel[child, 2],
                )
                parent_rel = wp.vec3(
                    propagation_body_com_rel[parent, 0],
                    propagation_body_com_rel[parent, 1],
                    propagation_body_com_rel[parent, 2],
                )
                child_minus_parent = child_rel - parent_rel
                propagated_child = wp.spatial_vector(p0, p1, p2, p3, p4, p5)
                propagated_parent = translate_wrench_between_parallel_frames(propagated_child, child_minus_parent)
                for r in range(6):
                    propagation_tree_pA[parent, r] = propagation_tree_pA[parent, r] + propagated_parent[r]

        for joint in range(joint_start, joint_end):
            child = joint_child[joint]
            parent = joint_parent[joint]
            dof_start = joint_qd_start[joint]
            dof_count = joint_dof_dim[joint, 0] + joint_dof_dim[joint, 1]
            parent_delta_child = wp.spatial_vector()
            if parent >= 0:
                parent_delta_parent = wp.spatial_vector(
                    propagation_tree_body_delta[parent, 0],
                    propagation_tree_body_delta[parent, 1],
                    propagation_tree_body_delta[parent, 2],
                    propagation_tree_body_delta[parent, 3],
                    propagation_tree_body_delta[parent, 4],
                    propagation_tree_body_delta[parent, 5],
                )
                child_rel = wp.vec3(
                    propagation_body_com_rel[child, 0],
                    propagation_body_com_rel[child, 1],
                    propagation_body_com_rel[child, 2],
                )
                parent_rel = wp.vec3(
                    propagation_body_com_rel[parent, 0],
                    propagation_body_com_rel[parent, 1],
                    propagation_body_com_rel[parent, 2],
                )
                parent_delta_child = translate_twist_between_parallel_frames(
                    parent_delta_parent, child_rel - parent_rel
                )

            for a in range(dof_count):
                gdof_a = dof_start + a
                qdd = float(0.0)
                for b in range(dof_count):
                    gdof_b = dof_start + b
                    parent_term = float(0.0)
                    if parent >= 0:
                        for r in range(6):
                            parent_term += propagation_tree_U[gdof_b, r] * parent_delta_child[r]
                    qdd += propagation_tree_D_inv[joint, a, b] * (propagation_tree_u[gdof_b] - parent_term)
                propagation_tree_qdd[gdof_a] = qdd

            for r in range(6):
                value = parent_delta_child[r]
                for a in range(dof_count):
                    gdof = dof_start + a
                    value += propagation_joint_S_flat[gdof, r] * propagation_tree_qdd[gdof]
                propagation_tree_body_delta[child, r] = value

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
    """Compute exact 6x6 COM response for active propagation bodies by tree solves."""
    group_idx = wp.tid()
    art = group_to_art[group_idx]
    world = art_to_world[art]
    joint_start = articulation_start[art]
    joint_end = articulation_start[art + 1]

    for local_body in range(max_propagation_bodies):
        if local_body >= propagation_body_count[world]:
            break
        target_body = propagation_body_list[world, local_body]
        if target_body < 0:
            continue
        if body_to_articulation[target_body] != art:
            continue

        # Root-path restriction: a unit wrench at target_body couples only to
        # the joints on target's root path. Off-path pA/u/qdd stay zero in the
        # backward pass, and body deltas flow root->leaf, so off-path qdd can
        # never feed the delta read back at target_body. Walking just the path
        # is exact and turns each basis solve from O(joints) into O(depth).
        # The walk is capped at the articulation's joint count: a well-formed
        # tree exits via wparent < 0 first, and the cap keeps a malformed or
        # cyclic model from hanging the GPU in an unbounded loop.
        path_len = int(0)
        walk = body_to_joint[target_body]
        for _cap in range(joint_end - joint_start):
            if walk < 0:
                break
            path_len += 1
            wparent = joint_parent[walk]
            if wparent >= 0:
                walk = body_to_joint[wparent]
            else:
                walk = int(-1)

        for basis in range(6):
            walk = body_to_joint[target_body]
            for _k in range(path_len):
                joint = walk
                wparent = joint_parent[joint]
                if wparent >= 0:
                    walk = body_to_joint[wparent]
                else:
                    walk = int(-1)
                body = joint_child[joint]
                for r in range(6):
                    propagation_tree_pA[body, r] = 0.0
                    propagation_tree_body_delta[body, r] = 0.0
                dof_start = joint_qd_start[joint]
                dof_end = joint_qd_start[joint + 1]
                for dof in range(dof_start, dof_end):
                    propagation_tree_u[dof] = 0.0
                    propagation_tree_qdd[dof] = 0.0

            force = wp.vec3(0.0)
            torque_com = wp.vec3(0.0)
            if basis == 0:
                force = wp.vec3(1.0, 0.0, 0.0)
            elif basis == 1:
                force = wp.vec3(0.0, 1.0, 0.0)
            elif basis == 2:
                force = wp.vec3(0.0, 0.0, 1.0)
            elif basis == 3:
                torque_com = wp.vec3(1.0, 0.0, 0.0)
            elif basis == 4:
                torque_com = wp.vec3(0.0, 1.0, 0.0)
            else:
                torque_com = wp.vec3(0.0, 0.0, 1.0)

            propagation_tree_pA[target_body, 0] = -force[0]
            propagation_tree_pA[target_body, 1] = -force[1]
            propagation_tree_pA[target_body, 2] = -force[2]
            propagation_tree_pA[target_body, 3] = -torque_com[0]
            propagation_tree_pA[target_body, 4] = -torque_com[1]
            propagation_tree_pA[target_body, 5] = -torque_com[2]

            # Backward sweep, target -> root along the path (children first).
            walk = body_to_joint[target_body]
            for _k in range(path_len):
                joint = walk
                wparent = joint_parent[joint]
                if wparent >= 0:
                    walk = body_to_joint[wparent]
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
                    p0 = propagation_tree_pA[child, 0]
                    p1 = propagation_tree_pA[child, 1]
                    p2 = propagation_tree_pA[child, 2]
                    p3 = propagation_tree_pA[child, 3]
                    p4 = propagation_tree_pA[child, 4]
                    p5 = propagation_tree_pA[child, 5]
                    for a in range(dof_count):
                        gdof_a = dof_start + a
                        coeff = float(0.0)
                        for b in range(dof_count):
                            gdof_b = dof_start + b
                            coeff += propagation_tree_D_inv[joint, a, b] * propagation_tree_u[gdof_b]
                        p0 += propagation_tree_U[gdof_a, 0] * coeff
                        p1 += propagation_tree_U[gdof_a, 1] * coeff
                        p2 += propagation_tree_U[gdof_a, 2] * coeff
                        p3 += propagation_tree_U[gdof_a, 3] * coeff
                        p4 += propagation_tree_U[gdof_a, 4] * coeff
                        p5 += propagation_tree_U[gdof_a, 5] * coeff

                    child_rel = wp.vec3(
                        propagation_body_com_rel[child, 0],
                        propagation_body_com_rel[child, 1],
                        propagation_body_com_rel[child, 2],
                    )
                    parent_rel = wp.vec3(
                        propagation_body_com_rel[parent, 0],
                        propagation_body_com_rel[parent, 1],
                        propagation_body_com_rel[parent, 2],
                    )
                    child_minus_parent = child_rel - parent_rel
                    propagated_child = wp.spatial_vector(p0, p1, p2, p3, p4, p5)
                    propagated_parent = translate_wrench_between_parallel_frames(propagated_child, child_minus_parent)
                    for r in range(6):
                        propagation_tree_pA[parent, r] = propagation_tree_pA[parent, r] + propagated_parent[r]

            # Forward sweep, root -> target along the path (parents first):
            # level i visits the (path_len-1-i)-th ancestor of target_body.
            for level in range(path_len):
                joint = body_to_joint[target_body]
                steps = path_len - 1 - level
                for _s in range(steps):
                    joint = body_to_joint[joint_parent[joint]]
                child = joint_child[joint]
                parent = joint_parent[joint]
                dof_start = joint_qd_start[joint]
                dof_count = joint_dof_dim[joint, 0] + joint_dof_dim[joint, 1]
                parent_delta_child = wp.spatial_vector()
                if parent >= 0:
                    parent_delta_parent = wp.spatial_vector(
                        propagation_tree_body_delta[parent, 0],
                        propagation_tree_body_delta[parent, 1],
                        propagation_tree_body_delta[parent, 2],
                        propagation_tree_body_delta[parent, 3],
                        propagation_tree_body_delta[parent, 4],
                        propagation_tree_body_delta[parent, 5],
                    )
                    child_rel = wp.vec3(
                        propagation_body_com_rel[child, 0],
                        propagation_body_com_rel[child, 1],
                        propagation_body_com_rel[child, 2],
                    )
                    parent_rel = wp.vec3(
                        propagation_body_com_rel[parent, 0],
                        propagation_body_com_rel[parent, 1],
                        propagation_body_com_rel[parent, 2],
                    )
                    parent_delta_child = translate_twist_between_parallel_frames(
                        parent_delta_parent, child_rel - parent_rel
                    )

                for a in range(dof_count):
                    gdof_a = dof_start + a
                    qdd = float(0.0)
                    for b in range(dof_count):
                        gdof_b = dof_start + b
                        parent_term = float(0.0)
                        if parent >= 0:
                            for r in range(6):
                                parent_term += propagation_tree_U[gdof_b, r] * parent_delta_child[r]
                        qdd += propagation_tree_D_inv[joint, a, b] * (propagation_tree_u[gdof_b] - parent_term)
                    propagation_tree_qdd[gdof_a] = qdd

                for r in range(6):
                    value = parent_delta_child[r]
                    for a in range(dof_count):
                        gdof = dof_start + a
                        value += propagation_joint_S_flat[gdof, r] * propagation_tree_qdd[gdof]
                    propagation_tree_body_delta[child, r] = value

            for r in range(6):
                propagation_body_response[target_body, r, basis] = propagation_tree_body_delta[target_body, r]


@wp.kernel
def compute_propagation_tree_body_response_revolute_for_size(
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
    # scratch/output: overwritten with local-COM body response matrices
    propagation_tree_Ia: wp.array3d[float],
    propagation_tree_body_delta: wp.array2d[float],
    # outputs
    propagation_body_response: wp.array3d[float],
):
    """Compute per-link response matrices for 0/1-DOF joint trees."""
    group_idx = wp.tid()
    art = group_to_art[group_idx]
    joint_start = articulation_start[art]
    joint_end = articulation_start[art + 1]

    for joint in range(joint_start, joint_end):
        child = joint_child[joint]
        parent = joint_parent[joint]
        dof_start = joint_qd_start[joint]
        dof_count = joint_dof_dim[joint, 0] + joint_dof_dim[joint, 1]
        edge = wp.vec3()
        if parent >= 0:
            child_rel = wp.vec3(
                propagation_body_com_rel[child, 0],
                propagation_body_com_rel[child, 1],
                propagation_body_com_rel[child, 2],
            )
            parent_rel = wp.vec3(
                propagation_body_com_rel[parent, 0],
                propagation_body_com_rel[parent, 1],
                propagation_body_com_rel[parent, 2],
            )
            edge = child_rel - parent_rel

        if dof_count == 0:
            for col in range(6):
                basis = wp.spatial_vector()
                if col == 0:
                    basis = wp.spatial_vector(1.0, 0.0, 0.0, 0.0, 0.0, 0.0)
                elif col == 1:
                    basis = wp.spatial_vector(0.0, 1.0, 0.0, 0.0, 0.0, 0.0)
                elif col == 2:
                    basis = wp.spatial_vector(0.0, 0.0, 1.0, 0.0, 0.0, 0.0)
                elif col == 3:
                    basis = wp.spatial_vector(0.0, 0.0, 0.0, 1.0, 0.0, 0.0)
                elif col == 4:
                    basis = wp.spatial_vector(0.0, 0.0, 0.0, 0.0, 1.0, 0.0)
                else:
                    basis = wp.spatial_vector(0.0, 0.0, 0.0, 0.0, 0.0, 1.0)

                child_delta = wp.spatial_vector()
                if parent >= 0:
                    parent_wrench = translate_wrench_between_parallel_frames(basis, edge)
                    parent_delta = wp.spatial_vector()
                    for row in range(6):
                        v = float(0.0)
                        for p_col in range(6):
                            v += propagation_tree_Ia[parent, row, p_col] * parent_wrench[p_col]
                        propagation_tree_body_delta[child, row] = v
                    parent_delta = wp.spatial_vector(
                        propagation_tree_body_delta[child, 0],
                        propagation_tree_body_delta[child, 1],
                        propagation_tree_body_delta[child, 2],
                        propagation_tree_body_delta[child, 3],
                        propagation_tree_body_delta[child, 4],
                        propagation_tree_body_delta[child, 5],
                    )
                    child_delta = translate_twist_between_parallel_frames(parent_delta, edge)

                for row in range(6):
                    propagation_tree_Ia[child, row, col] = child_delta[row]
                    propagation_body_response[child, row, col] = child_delta[row]
        else:
            gdof = dof_start
            inv_d = propagation_tree_D_inv[joint, 0, 0]
            for col in range(6):
                basis = wp.spatial_vector()
                if col == 0:
                    basis = wp.spatial_vector(1.0, 0.0, 0.0, 0.0, 0.0, 0.0)
                elif col == 1:
                    basis = wp.spatial_vector(0.0, 1.0, 0.0, 0.0, 0.0, 0.0)
                elif col == 2:
                    basis = wp.spatial_vector(0.0, 0.0, 1.0, 0.0, 0.0, 0.0)
                elif col == 3:
                    basis = wp.spatial_vector(0.0, 0.0, 0.0, 1.0, 0.0, 0.0)
                elif col == 4:
                    basis = wp.spatial_vector(0.0, 0.0, 0.0, 0.0, 1.0, 0.0)
                else:
                    basis = wp.spatial_vector(0.0, 0.0, 0.0, 0.0, 0.0, 1.0)

                s_dot_f = propagation_joint_S_flat[gdof, col]
                p_child = wp.spatial_vector(
                    basis[0] - propagation_tree_U[gdof, 0] * inv_d * s_dot_f,
                    basis[1] - propagation_tree_U[gdof, 1] * inv_d * s_dot_f,
                    basis[2] - propagation_tree_U[gdof, 2] * inv_d * s_dot_f,
                    basis[3] - propagation_tree_U[gdof, 3] * inv_d * s_dot_f,
                    basis[4] - propagation_tree_U[gdof, 4] * inv_d * s_dot_f,
                    basis[5] - propagation_tree_U[gdof, 5] * inv_d * s_dot_f,
                )

                parent_delta_child = wp.spatial_vector()
                if parent >= 0:
                    p_parent = translate_wrench_between_parallel_frames(p_child, edge)
                    for row in range(6):
                        value = float(0.0)
                        for p_col in range(6):
                            value += propagation_tree_Ia[parent, row, p_col] * p_parent[p_col]
                        propagation_tree_body_delta[child, row] = value
                    parent_delta_parent = wp.spatial_vector(
                        propagation_tree_body_delta[child, 0],
                        propagation_tree_body_delta[child, 1],
                        propagation_tree_body_delta[child, 2],
                        propagation_tree_body_delta[child, 3],
                        propagation_tree_body_delta[child, 4],
                        propagation_tree_body_delta[child, 5],
                    )
                    parent_delta_child = translate_twist_between_parallel_frames(parent_delta_parent, edge)

                parent_dot = float(0.0)
                for row in range(6):
                    parent_dot += propagation_tree_U[gdof, row] * parent_delta_child[row]
                qdd = inv_d * (s_dot_f - parent_dot)

                for row in range(6):
                    value = parent_delta_child[row] + propagation_joint_S_flat[gdof, row] * qdd
                    propagation_tree_Ia[child, row, col] = value
                    propagation_body_response[child, row, col] = value


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
    joint_dof_dim: wp.array2d[int],
    propagation_joint_S_flat: wp.array2d[float],
    propagation_body_com_rel: wp.array2d[float],
    v_out: wp.array[float],
    # scratch
    propagation_tree_body_delta: wp.array2d[float],
    # outputs
    propagation_body_qd: wp.array2d[float],
):
    """Refresh propagation live COM velocities from generalized velocity by a tree pass.

    ``propagate_tree_impulses_for_size`` already leaves ``propagation_body_qd``
    consistent with ``v_out``. Between propagation iterations the tree
    generalized velocities only change when a dense GS phase ran rows for this
    world, so when ``force_refresh`` is 0 the pass is skipped for worlds with
    no dense contact rows (drive/limit/velocity-limit phases force refresh
    through ``force_refresh`` instead).
    """
    group_idx = wp.tid()
    art = group_to_art[group_idx]
    if force_refresh == 0:
        world = art_to_world[art]
        if world >= 0 and dense_contact_world_flag[world] == 0:
            return
    joint_start = articulation_start[art]
    joint_end = articulation_start[art + 1]

    for joint in range(joint_start, joint_end):
        child = joint_child[joint]
        parent = joint_parent[joint]
        dof_start = joint_qd_start[joint]
        dof_count = joint_dof_dim[joint, 0] + joint_dof_dim[joint, 1]
        parent_delta_child = wp.spatial_vector()
        if parent >= 0:
            parent_delta_parent = wp.spatial_vector(
                propagation_tree_body_delta[parent, 0],
                propagation_tree_body_delta[parent, 1],
                propagation_tree_body_delta[parent, 2],
                propagation_tree_body_delta[parent, 3],
                propagation_tree_body_delta[parent, 4],
                propagation_tree_body_delta[parent, 5],
            )
            child_rel = wp.vec3(
                propagation_body_com_rel[child, 0],
                propagation_body_com_rel[child, 1],
                propagation_body_com_rel[child, 2],
            )
            parent_rel = wp.vec3(
                propagation_body_com_rel[parent, 0],
                propagation_body_com_rel[parent, 1],
                propagation_body_com_rel[parent, 2],
            )
            parent_delta_child = translate_twist_between_parallel_frames(parent_delta_parent, child_rel - parent_rel)

        for r in range(6):
            value = parent_delta_child[r]
            for a in range(dof_count):
                gdof = dof_start + a
                value += propagation_joint_S_flat[gdof, r] * v_out[gdof]
            propagation_tree_body_delta[child, r] = value
            propagation_body_qd[child, r] = value


@wp.kernel
def flush_propagation_free_body_qd_to_vout(
    body_to_articulation: wp.array[int],
    is_free_rigid: wp.array[int],
    articulation_root_dof_start: wp.array[int],
    # in/out
    propagation_body_qd: wp.array2d[float],
    propagation_body_impulses: wp.array2d[float],
    v_out: wp.array[float],
):
    """Write propagation live free-rigid velocities to v_out after propagation GS.

    The generalized linear coordinate and the propagation body velocity both refer to the root
    body's centre of mass, so the write-back is a plain copy.
    """
    body = wp.tid()
    art = body_to_articulation[body]
    if art < 0:
        return
    if is_free_rigid[art] == 0:
        return

    dof_start = articulation_root_dof_start[art]
    for r in range(6):
        v_out[dof_start + r] = propagation_body_qd[body, r]

    for r in range(6):
        propagation_body_impulses[body, r] = 0.0


@wp.kernel
def refresh_propagation_free_body_qd_from_vout(
    body_to_articulation: wp.array[int],
    is_free_rigid: wp.array[int],
    articulation_root_dof_start: wp.array[int],
    v_out: wp.array[float],
    # out
    propagation_body_qd: wp.array2d[float],
):
    """Refresh propagation live COM velocities for free-rigid bodies from v_out.

    Both sides use the root body's centre of mass as the linear reference point, so the refresh is
    a plain copy.
    """
    body = wp.tid()
    art = body_to_articulation[body]
    if art < 0:
        return
    if is_free_rigid[art] == 0:
        return

    dof_start = articulation_root_dof_start[art]
    for r in range(6):
        propagation_body_qd[body, r] = v_out[dof_start + r]


@wp.kernel
def propagate_tree_impulses_for_size(
    group_to_art: wp.array[int],
    art_to_world: wp.array[int],
    propagation_cache_world_flag: wp.array[int],
    articulation_start: wp.array[int],
    joint_parent: wp.array[int],
    joint_child: wp.array[int],
    joint_qd_start: wp.array[int],
    joint_dof_dim: wp.array2d[int],
    propagation_joint_S_flat: wp.array2d[float],
    propagation_body_com_rel: wp.array2d[float],
    propagation_tree_U: wp.array2d[float],
    propagation_tree_D_inv: wp.array3d[float],
    propagation_body_impulses: wp.array2d[float],
    # scratch
    propagation_tree_pA: wp.array2d[float],
    propagation_tree_u: wp.array[float],
    propagation_tree_qdd: wp.array[float],
    propagation_tree_body_delta: wp.array2d[float],
    # in/out
    propagation_body_qd: wp.array2d[float],
    v_out: wp.array[float],
):
    """Propagate deferred propagation body impulses through the articulation tree.

    ``propagation_cache_world_flag`` gates the cached-response fast path: a
    nonzero flag means this world's deferred impulses were already converted
    to joint velocities by the precomputed response-matrix GEMV (exactly the
    same linear map), so this articulation is skipped. Callers with the
    cached path disabled pass an all-zeros flag, which leaves behavior
    unchanged.
    """
    group_idx = wp.tid()
    art = group_to_art[group_idx]
    world = art_to_world[art]
    if world >= 0 and propagation_cache_world_flag[world] != 0:
        return
    joint_start = articulation_start[art]
    joint_end = articulation_start[art + 1]

    has_impulse = int(0)
    for joint in range(joint_start, joint_end):
        body = joint_child[joint]
        for r in range(6):
            propagation_tree_pA[body, r] = 0.0
            propagation_tree_body_delta[body, r] = 0.0
        dof_start = joint_qd_start[joint]
        dof_end = joint_qd_start[joint + 1]
        for dof in range(dof_start, dof_end):
            propagation_tree_u[dof] = 0.0
            propagation_tree_qdd[dof] = 0.0

        force = wp.vec3(
            propagation_body_impulses[body, 0],
            propagation_body_impulses[body, 1],
            propagation_body_impulses[body, 2],
        )
        torque_com = wp.vec3(
            propagation_body_impulses[body, 3],
            propagation_body_impulses[body, 4],
            propagation_body_impulses[body, 5],
        )
        if wp.length_sq(force) + wp.length_sq(torque_com) > 0.0:
            has_impulse = int(1)

        propagation_tree_pA[body, 0] = -force[0]
        propagation_tree_pA[body, 1] = -force[1]
        propagation_tree_pA[body, 2] = -force[2]
        propagation_tree_pA[body, 3] = -torque_com[0]
        propagation_tree_pA[body, 4] = -torque_com[1]
        propagation_tree_pA[body, 5] = -torque_com[2]

    if has_impulse != 0:
        for offset in range(joint_end - joint_start):
            joint = joint_end - 1 - offset
            child = joint_child[joint]
            parent = joint_parent[joint]
            dof_start = joint_qd_start[joint]
            dof_count = joint_dof_dim[joint, 0] + joint_dof_dim[joint, 1]

            for a in range(dof_count):
                gdof = dof_start + a
                value = float(0.0)
                for r in range(6):
                    value -= propagation_joint_S_flat[gdof, r] * propagation_tree_pA[child, r]
                propagation_tree_u[gdof] = value

            if parent >= 0:
                p0 = propagation_tree_pA[child, 0]
                p1 = propagation_tree_pA[child, 1]
                p2 = propagation_tree_pA[child, 2]
                p3 = propagation_tree_pA[child, 3]
                p4 = propagation_tree_pA[child, 4]
                p5 = propagation_tree_pA[child, 5]
                for a in range(dof_count):
                    gdof_a = dof_start + a
                    coeff = float(0.0)
                    for b in range(dof_count):
                        gdof_b = dof_start + b
                        coeff += propagation_tree_D_inv[joint, a, b] * propagation_tree_u[gdof_b]
                    p0 += propagation_tree_U[gdof_a, 0] * coeff
                    p1 += propagation_tree_U[gdof_a, 1] * coeff
                    p2 += propagation_tree_U[gdof_a, 2] * coeff
                    p3 += propagation_tree_U[gdof_a, 3] * coeff
                    p4 += propagation_tree_U[gdof_a, 4] * coeff
                    p5 += propagation_tree_U[gdof_a, 5] * coeff

                child_rel = wp.vec3(
                    propagation_body_com_rel[child, 0],
                    propagation_body_com_rel[child, 1],
                    propagation_body_com_rel[child, 2],
                )
                parent_rel = wp.vec3(
                    propagation_body_com_rel[parent, 0],
                    propagation_body_com_rel[parent, 1],
                    propagation_body_com_rel[parent, 2],
                )
                child_minus_parent = child_rel - parent_rel
                propagated_child = wp.spatial_vector(p0, p1, p2, p3, p4, p5)
                propagated_parent = translate_wrench_between_parallel_frames(propagated_child, child_minus_parent)
                for r in range(6):
                    propagation_tree_pA[parent, r] = propagation_tree_pA[parent, r] + propagated_parent[r]

        for joint in range(joint_start, joint_end):
            child = joint_child[joint]
            parent = joint_parent[joint]
            dof_start = joint_qd_start[joint]
            dof_count = joint_dof_dim[joint, 0] + joint_dof_dim[joint, 1]
            parent_delta_child = wp.spatial_vector()
            if parent >= 0:
                parent_delta_parent = wp.spatial_vector(
                    propagation_tree_body_delta[parent, 0],
                    propagation_tree_body_delta[parent, 1],
                    propagation_tree_body_delta[parent, 2],
                    propagation_tree_body_delta[parent, 3],
                    propagation_tree_body_delta[parent, 4],
                    propagation_tree_body_delta[parent, 5],
                )
                child_rel = wp.vec3(
                    propagation_body_com_rel[child, 0],
                    propagation_body_com_rel[child, 1],
                    propagation_body_com_rel[child, 2],
                )
                parent_rel = wp.vec3(
                    propagation_body_com_rel[parent, 0],
                    propagation_body_com_rel[parent, 1],
                    propagation_body_com_rel[parent, 2],
                )
                parent_delta_child = translate_twist_between_parallel_frames(
                    parent_delta_parent, child_rel - parent_rel
                )

            for a in range(dof_count):
                gdof_a = dof_start + a
                qdd = float(0.0)
                for b in range(dof_count):
                    gdof_b = dof_start + b
                    parent_term = float(0.0)
                    if parent >= 0:
                        for r in range(6):
                            parent_term += propagation_tree_U[gdof_b, r] * parent_delta_child[r]
                    qdd += propagation_tree_D_inv[joint, a, b] * (propagation_tree_u[gdof_b] - parent_term)
                propagation_tree_qdd[gdof_a] = qdd
                v_out[gdof_a] = v_out[gdof_a] + qdd

            for r in range(6):
                value = parent_delta_child[r]
                for a in range(dof_count):
                    gdof = dof_start + a
                    value += propagation_joint_S_flat[gdof, r] * propagation_tree_qdd[gdof]
                propagation_tree_body_delta[child, r] = value

    # Recompute full live body velocities from updated generalized velocities
    # and clear the deferred body impulse buffer for the next GS iteration.
    # With no deferred impulses, v_out and propagation_body_qd are both
    # untouched since the previous consistent recompute, so skip the pass.
    if has_impulse != 0:
        for joint in range(joint_start, joint_end):
            child = joint_child[joint]
            parent = joint_parent[joint]
            dof_start = joint_qd_start[joint]
            dof_count = joint_dof_dim[joint, 0] + joint_dof_dim[joint, 1]
            parent_delta_child = wp.spatial_vector()
            if parent >= 0:
                parent_delta_parent = wp.spatial_vector(
                    propagation_tree_body_delta[parent, 0],
                    propagation_tree_body_delta[parent, 1],
                    propagation_tree_body_delta[parent, 2],
                    propagation_tree_body_delta[parent, 3],
                    propagation_tree_body_delta[parent, 4],
                    propagation_tree_body_delta[parent, 5],
                )
                child_rel = wp.vec3(
                    propagation_body_com_rel[child, 0],
                    propagation_body_com_rel[child, 1],
                    propagation_body_com_rel[child, 2],
                )
                parent_rel = wp.vec3(
                    propagation_body_com_rel[parent, 0],
                    propagation_body_com_rel[parent, 1],
                    propagation_body_com_rel[parent, 2],
                )
                parent_delta_child = translate_twist_between_parallel_frames(
                    parent_delta_parent, child_rel - parent_rel
                )

            for r in range(6):
                value = parent_delta_child[r]
                for a in range(dof_count):
                    gdof = dof_start + a
                    value += propagation_joint_S_flat[gdof, r] * v_out[gdof]
                propagation_tree_body_delta[child, r] = value
                propagation_body_qd[child, r] = value
            for r in range(6):
                propagation_body_impulses[child, r] = 0.0


@wp.kernel
def pgs_solve_propagation_contact_loop(
    propagation_constraint_count: wp.array[int],
    propagation_body_a: wp.array2d[int],
    propagation_body_b: wp.array2d[int],
    propagation_MiJt_a: wp.array3d[float],
    propagation_MiJt_b: wp.array3d[float],
    propagation_J_a: wp.array3d[float],
    propagation_J_b: wp.array3d[float],
    propagation_eff_mass_inv: wp.array2d[float],
    propagation_rhs: wp.array2d[float],
    propagation_row_w: wp.array2d[float],
    propagation_row_type: wp.array2d[int],
    propagation_row_parent: wp.array2d[int],
    propagation_row_mu: wp.array2d[float],
    propagation_max_constraints: int,
    iterations: int,
    omega: float,
    regularize: int,
    friction_start_iteration: int,
    iteration_offset: int,
    # in/out
    propagation_impulses: wp.array2d[float],
    propagation_body_qd: wp.array2d[float],
    propagation_body_impulses: wp.array2d[float],
):
    """Serial per-world propagation contact GS over fixed-size body-space rows."""
    world = wp.tid()
    m_count = propagation_constraint_count[world]
    if m_count == 0:
        return
    if m_count > propagation_max_constraints:
        m_count = propagation_max_constraints

    for it in range(iterations):
        global_iter = iteration_offset + it
        for i in range(m_count):
            row_type = propagation_row_type[world, i]
            if row_type == PGS_CONSTRAINT_TYPE_FRICTION and global_iter < friction_start_iteration:
                propagation_impulses[world, i] = 0.0
                continue

            eff_inv = propagation_eff_mass_inv[world, i]
            if eff_inv <= 0.0 and row_type != PGS_CONSTRAINT_TYPE_FRICTION:
                continue

            ba = propagation_body_a[world, i]
            bb = propagation_body_b[world, i]

            jv = float(0.0)
            if ba >= 0:
                for k in range(6):
                    jv += propagation_J_a[world, i, k] * propagation_body_qd[ba, k]
            if bb >= 0:
                for k in range(6):
                    jv += propagation_J_b[world, i, k] * propagation_body_qd[bb, k]

            residual = jv + propagation_rhs[world, i]
            old_impulse = propagation_impulses[world, i]
            w = float(1.0)
            if regularize != 0:
                w = propagation_row_w[world, i]
            delta = -residual * eff_inv * w - (1.0 - w) * old_impulse
            new_impulse = old_impulse + omega * delta

            if row_type == PGS_CONSTRAINT_TYPE_CONTACT:
                if new_impulse < 0.0:
                    new_impulse = 0.0
            elif row_type == PGS_CONSTRAINT_TYPE_FRICTION:
                parent_idx = propagation_row_parent[world, i]
                lambda_n = patch_normal_load(propagation_row_parent, propagation_impulses, world, parent_idx)
                mu_val = propagation_row_mu[world, i]
                radius = wp.max(mu_val * lambda_n, 0.0)

                if i != parent_idx + 1:
                    new_impulse = old_impulse
                else:
                    sib = parent_idx + 2
                    other = propagation_impulses[world, sib]
                    sibling_residual = propagation_rhs[world, sib]
                    cross = float(0.0)
                    for k in range(6):
                        if ba >= 0:
                            sibling_residual += propagation_J_a[world, sib, k] * propagation_body_qd[ba, k]
                            cross += propagation_J_a[world, i, k] * propagation_MiJt_a[world, sib, k]
                        if bb >= 0:
                            sibling_residual += propagation_J_b[world, sib, k] * propagation_body_qd[bb, k]
                            cross += propagation_J_b[world, i, k] * propagation_MiJt_b[world, sib, k]
                    inv_sib = propagation_eff_mass_inv[world, sib]
                    sibling_diag = float(0.0)
                    if inv_sib > 0.0:
                        sibling_diag = 1.0 / inv_sib
                    first_diag = float(0.0)
                    if eff_inv > 0.0:
                        first_diag = 1.0 / eff_inv
                    trial = friction_pair_candidate(
                        first_diag,
                        cross,
                        sibling_diag,
                        wp.vec2(residual, sibling_residual),
                        wp.vec2(old_impulse, other),
                        radius,
                        omega,
                    )
                    magnitude = wp.length(trial)
                    if magnitude > radius:
                        trial *= radius / magnitude
                    new_impulse = trial[0]
                    sib_delta = trial[1] - other
                    propagation_impulses[world, sib] = trial[1]
                    if sib_delta != 0.0:
                        sib_ba = propagation_body_a[world, sib]
                        sib_bb = propagation_body_b[world, sib]
                        if sib_ba >= 0:
                            for k in range(6):
                                propagation_body_qd[sib_ba, k] = (
                                    propagation_body_qd[sib_ba, k] + propagation_MiJt_a[world, sib, k] * sib_delta
                                )
                                propagation_body_impulses[sib_ba, k] = (
                                    propagation_body_impulses[sib_ba, k] + propagation_J_a[world, sib, k] * sib_delta
                                )
                        if sib_bb >= 0:
                            for k in range(6):
                                propagation_body_qd[sib_bb, k] = (
                                    propagation_body_qd[sib_bb, k] + propagation_MiJt_b[world, sib, k] * sib_delta
                                )
                                propagation_body_impulses[sib_bb, k] = (
                                    propagation_body_impulses[sib_bb, k] + propagation_J_b[world, sib, k] * sib_delta
                                )

            delta_impulse = new_impulse - old_impulse
            propagation_impulses[world, i] = new_impulse

            if delta_impulse != 0.0:
                if ba >= 0:
                    for k in range(6):
                        propagation_body_qd[ba, k] = (
                            propagation_body_qd[ba, k] + propagation_MiJt_a[world, i, k] * delta_impulse
                        )
                        propagation_body_impulses[ba, k] = (
                            propagation_body_impulses[ba, k] + propagation_J_a[world, i, k] * delta_impulse
                        )
                if bb >= 0:
                    for k in range(6):
                        propagation_body_qd[bb, k] = (
                            propagation_body_qd[bb, k] + propagation_MiJt_b[world, i, k] * delta_impulse
                        )
                        propagation_body_impulses[bb, k] = (
                            propagation_body_impulses[bb, k] + propagation_J_b[world, i, k] * delta_impulse
                        )


# ---------------------------------------------------------------------------
# Gilles Daviet's 1D Coulomb Newton — ported from
# ``artifacts/2026-04-16-slack-raisim/coulomb_root_finding_warp.py``.
#
# The scalar bracketed-Newton on the tangential-force ratio alpha solves the
# Coulomb cone coupling directly (no lagged de Saxce correction, no
# quartic).  See the artifact's module docstring for the full
# derivation; summary:
#
#     A_T = W_T - w_NT w_NT^T / W_N          (2x2 Schur complement)
#     c_T = b_T - (b_N / W_N) w_NT           (2-vector)
#
#     phi(alpha) = |s(alpha)| - mu (w_NT . s(alpha) - b_N) / W_N
#     where  s(alpha) = (A_T + alpha I)^{-1} c_T
#
#     Sticking:  phi(0) <= 0  =>  alpha = 0
#     Sliding:   find alpha > 0 s.t. phi(alpha) = 0 via bracketed Newton
#
# The FeatherPGS matrix-free path exposes each contact triple through a
# 3x3 effective-mass (Delassus) block ``G = J H^{-1} J^T`` assembled in
# :func:`friction_step_coulomb_newton` on the fly from ``mf_J_*`` /
# ``mf_MiJt_*`` — exactly the same data consumed by
# :func:`friction_step_bisection`.  The ``W`` argument of
# :func:`solve_coulomb_row` is that block, ``b`` the per-contact
# velocity bias (``u_free`` shifted by the normal-row target velocity
# so the reference's ``b_N < 0`` convention is preserved), and ``mu``
# the row's Coulomb coefficient.
# ---------------------------------------------------------------------------


@wp.func
def _fpgs_mat22_solve(M: wp.mat22, rhs: wp.vec2) -> wp.vec2:
    """Solve ``M x = rhs`` via Cramer's rule (cheaper than a full inverse).

    Ported verbatim from ``coulomb_root_finding_warp.py::_mat22_solve``;
    renamed with the ``_fpgs_`` prefix to avoid colliding with any other
    2x2 helpers that may be registered as ``@wp.func``s.
    """
    inv_det = 1.0 / (M[0, 0] * M[1, 1] - M[0, 1] * M[1, 0])
    return wp.vec2(
        (M[1, 1] * rhs[0] - M[0, 1] * rhs[1]) * inv_det,
        (M[0, 0] * rhs[1] - M[1, 0] * rhs[0]) * inv_det,
    )


@wp.func
def _fpgs_phi_dphi_and_s(
    AT: wp.mat22,
    cT: wp.vec2,
    wNT: wp.vec2,
    bN: float,
    WN: float,
    mu: float,
    alpha: float,
) -> wp.vec4:
    """Return ``(phi, phi', s[0], s[1])`` with ``s = (A_T + alpha I)^{-1} c_T``.

    Shares the 2x2 determinant across both solves (``s`` and
    ``t = M^{-1} s``).  Ported from
    ``coulomb_root_finding_warp.py::_phi_dphi_and_s``.
    """
    a = AT[0, 0] + alpha
    b_ = AT[0, 1]
    c = AT[1, 0]
    d = AT[1, 1] + alpha
    inv_det = 1.0 / (a * d - b_ * c)

    # s = M^{-1} cT
    s0 = (d * cT[0] - b_ * cT[1]) * inv_det
    s1 = (a * cT[1] - c * cT[0]) * inv_det
    # t = M^{-1} s  (reuses same inv_det)
    t0 = (d * s0 - b_ * s1) * inv_det
    t1 = (a * s1 - c * s0) * inv_det

    norm_s = wp.sqrt(s0 * s0 + s1 * s1)
    wNT_s = wNT[0] * s0 + wNT[1] * s1
    wNT_t = wNT[0] * t0 + wNT[1] * t1

    val = norm_s - mu * (wNT_s - bN) / WN
    dval = -(s0 * t0 + s1 * t1) / norm_s + mu * wNT_t / WN
    return wp.vec4(val, dval, s0, s1)


# Return type matches the reference batch kernel:
# ``[alpha, r_N, r_T[0], r_T[1], float(iterations), float(status)]`` where
# ``status`` is ``0.0`` for sticking (``alpha = 0``, ``|r_T| <= mu r_N``)
# or ``1.0`` for sliding (``alpha > 0``, ``|r_T| = mu r_N``).
FPGSCoulombNewtonResult = wp.types.vector(6, float)


@wp.func
def solve_coulomb_row(W: wp.mat33, b: wp.vec3, mu: float) -> FPGSCoulombNewtonResult:
    """Solve a single 3-D Coulomb friction contact (Gilles Daviet 1D Newton).

    Given a 3x3 SPD Delassus block ``W``, velocity-level rhs ``b`` (with
    ``b_N < 0`` for penetrating / sliding-into-plane contacts), and the
    row's Coulomb coefficient ``mu``, returns the 6-vector
    ``[alpha, r_N, r_T[0], r_T[1], iterations, status]`` described
    above.  Ported from
    ``artifacts/2026-04-16-slack-raisim/coulomb_root_finding_warp.py::
    solve_coulomb`` (renamed here so the in-solver symbol does not
    collide with the reference batch kernel).

    Args:
        W: 3x3 SPD effective-mass (Delassus) block for the contact
            triple.  Index 0 is the normal row, indices 1 / 2 are the
            two tangential rows.
        b: Velocity-level rhs; ``b[0]`` is the normal component
            (``bN < 0`` on penetrating contacts) and ``b[1:]`` are the
            two tangential components.
        mu: Row-level Coulomb friction coefficient.

    Returns:
        ``[alpha, r_N, r_T[0], r_T[1], iterations, status]`` with status
        ``0.0`` for sticking and ``1.0`` for sliding.
    """
    WN = W[0, 0]
    wNT = wp.vec2(W[1, 0], W[2, 0])
    WT = wp.mat22(W[1, 1], W[1, 2], W[2, 1], W[2, 2])
    bN = b[0]
    bT = wp.vec2(b[1], b[2])

    AT = WT - wp.outer(wNT, wNT) / WN
    cT = bT - (bN / WN) * wNT

    # Sticking check: alpha = 0.
    s0 = _fpgs_mat22_solve(AT, cT)
    phi0 = wp.length(s0) - mu * (wp.dot(wNT, s0) - bN) / WN

    if phi0 <= 0.0:
        rT_stick = -s0
        rN_stick = -(wp.dot(wNT, rT_stick) + bN) / WN
        return FPGSCoulombNewtonResult(0.0, rN_stick, rT_stick[0], rT_stick[1], 0.0, 0.0)

    # Find upper bracket by doubling.
    hi = float(1.0)
    for _expand in range(wp.static(_FPGS_COULOMB_NEWTON_EXPAND_ITERS)):
        vhi = _fpgs_phi_dphi_and_s(AT, cT, wNT, bN, WN, mu, hi)
        if vhi[0] < 0.0:
            break
        hi = hi * 2.0

    lo = float(0.0)
    x = 0.5 * (lo + hi)
    iterations = int(0)
    tol = 1.0e-6 + 1.0e-6 * phi0
    # s from the last evaluation, used for final r_T / r_N recovery.
    last_s = wp.vec2(0.0, 0.0)

    for _it in range(wp.static(_FPGS_COULOMB_NEWTON_NEWTON_ITERS)):
        vd = _fpgs_phi_dphi_and_s(AT, cT, wNT, bN, WN, mu, x)
        fx = vd[0]
        dfx = vd[1]
        last_s = wp.vec2(vd[2], vd[3])
        iterations = iterations + 1

        if wp.abs(fx) < tol or wp.abs(hi - lo) < 1.0e-6 * (1.0 + hi):
            break

        if fx > 0.0:
            lo = x
        else:
            hi = x

        x_new = float(0.5) * (lo + hi)
        if dfx != 0.0:
            x_newton = x - fx / dfx
            if x_newton > lo and x_newton < hi:
                x_new = x_newton
        x = x_new

    # Recover r_T, r_N from the last evaluated s (avoids a redundant solve).
    rT_final = -last_s
    rN_final = -(wp.dot(wNT, rT_final) + bN) / WN

    return FPGSCoulombNewtonResult(x, rN_final, rT_final[0], rT_final[1], float(iterations), 1.0)


@wp.func
def friction_step_current(
    world: int,
    i: int,
    omega: float,
    mf_eff_mass_inv: wp.array2d[float],
    mf_rhs: wp.array2d[float],
    mf_J_a: wp.array3d[float],
    mf_J_b: wp.array3d[float],
    mf_body_a: wp.array2d[int],
    mf_body_b: wp.array2d[int],
    mf_MiJt_a: wp.array3d[float],
    mf_MiJt_b: wp.array3d[float],
    mf_row_parent: wp.array2d[int],
    mf_row_mu: wp.array2d[float],
    body_to_articulation: wp.array[int],
    art_dof_start: wp.array[int],
    mf_impulses: wp.array2d[float],
    v_out: wp.array[float],
):
    """Solve the tangent pair at the current patch or point normal load.

    The caller applies the first tangent's velocity change; this function
    applies the second and makes its subsequent row visit a no-op.
    """
    parent_idx = mf_row_parent[world, i]
    if i != parent_idx + 1:
        return mf_impulses[world, i]
    sib = parent_idx + 2
    lambda_n = patch_normal_load(mf_row_parent, mf_impulses, world, parent_idx)
    radius = wp.max(mf_row_mu[world, i] * lambda_n, 0.0)
    residual = wp.vec2(mf_rhs[world, i], mf_rhs[world, sib])
    ba = mf_body_a[world, i]
    bb = mf_body_b[world, i]
    cross = float(0.0)
    for k in range(6):
        if ba >= 0:
            ds = art_dof_start[body_to_articulation[ba]]
            residual[0] += mf_J_a[world, i, k] * v_out[ds + k]
            residual[1] += mf_J_a[world, sib, k] * v_out[ds + k]
            cross += mf_J_a[world, i, k] * mf_MiJt_a[world, sib, k]
        if bb >= 0:
            ds = art_dof_start[body_to_articulation[bb]]
            residual[0] += mf_J_b[world, i, k] * v_out[ds + k]
            residual[1] += mf_J_b[world, sib, k] * v_out[ds + k]
            cross += mf_J_b[world, i, k] * mf_MiJt_b[world, sib, k]
    inv0 = mf_eff_mass_inv[world, i]
    inv1 = mf_eff_mass_inv[world, sib]
    sibling_diag = float(0.0)
    if inv1 > 0.0:
        sibling_diag = 1.0 / inv1
    first_diag = float(0.0)
    if inv0 > 0.0:
        first_diag = 1.0 / inv0
    trial = friction_pair_candidate(
        first_diag,
        cross,
        sibling_diag,
        residual,
        wp.vec2(mf_impulses[world, i], mf_impulses[world, sib]),
        radius,
        omega,
    )
    magnitude = wp.length(trial)
    if magnitude > radius:
        trial *= radius / magnitude
    sibling_delta = trial[1] - mf_impulses[world, sib]
    mf_impulses[world, sib] = trial[1]
    for k in range(6):
        if ba >= 0:
            ds = art_dof_start[body_to_articulation[ba]]
            v_out[ds + k] += mf_MiJt_a[world, sib, k] * sibling_delta
        if bb >= 0:
            ds = art_dof_start[body_to_articulation[bb]]
            v_out[ds + k] += mf_MiJt_b[world, sib, k] * sibling_delta
    return trial[0]


@wp.func
def friction_step_bisection(
    world: int,
    i: int,
    mf_body_a: wp.array2d[int],
    mf_body_b: wp.array2d[int],
    mf_J_a: wp.array3d[float],
    mf_J_b: wp.array3d[float],
    mf_MiJt_a: wp.array3d[float],
    mf_MiJt_b: wp.array3d[float],
    mf_row_parent: wp.array2d[int],
    mf_row_mu: wp.array2d[float],
    mf_rhs: wp.array2d[float],
    body_to_articulation: wp.array[int],
    art_dof_start: wp.array[int],
    use_de_saxce: int,
    mf_impulses: wp.array2d[float],
    v_out: wp.array[float],
):
    """RAISim-style bisection-on-λ_n Coulomb friction step.

    Drop-in replacement for :func:`friction_step_current` selected by
    ``friction_mode="bisection"`` (``use_de_saxce == 0``) or
    ``friction_mode="bisection_desaxce"`` (``use_de_saxce == 1``) on
    ``pgs_mode="matrix_free"``.  Ported from the ``USE_BISECTION`` branch
    of ``artifacts/2026-04-16-slack-raisim/repos/mmacklin-newton-solver-raisim
    /newton/_src/solvers/raisim/kernels.py`` (``gs_contact_sweep``, lines
    ~687-803).  The de Saxce branch augments the normal target velocity
    with ``μ · ‖c_T‖`` (Le Lidec & Carpentier 2024) to enforce the
    maximum dissipation principle on sliding contacts — the 5/13 vs 6/13
    split in the ``[FPGS Friction Modes]`` series.

    The matrix-free layout stores each contact as three consecutive rows
    (normal + 2 friction).  The RAISim step is naturally a per-contact
    3-DOF solve, so this function centralises the bisection on the first
    friction row of the triple (``i == parent + 1``) and becomes a no-op
    on the second friction row (``i == parent + 2``).  It therefore:

    * Recomputes ``u_n, u_t1, u_t2`` from the current ``v_out``.
    * Builds the 3x3 Delassus block ``G = J H⁻¹ Jᵀ`` on the fly from the
      per-row ``mf_J_*`` / ``mf_MiJt_*`` arrays.
    * Bisects ``λ_n`` in ``[0, hi]`` with
      :data:`_FPGS_BISECTION_ITERS` steps.  At each probe, solves the
      2x2 tangential sub-problem and projects onto the Coulomb cone.
    * Writes the final ``(λ_n, λ_t2)`` to ``mf_impulses`` directly and
      applies the corresponding ``v_out`` corrections for the normal
      row and the second friction row.  The first-friction-row ``v_out``
      correction is returned as the new impulse so the outer PGS loop
      applies it via its usual ``delta_impulse`` path.

    Args:
        world: World index for the current row.
        i: Constraint row index within the world.
        mf_body_a: Matrix-free body-a indices [shape: world_count,
            mf_max_constraints].
        mf_body_b: Matrix-free body-b indices [shape: world_count,
            mf_max_constraints].
        mf_J_a: Per-row Jacobian ``J`` for body a [shape: world_count,
            mf_max_constraints, 6].
        mf_J_b: Per-row Jacobian ``J`` for body b [shape: world_count,
            mf_max_constraints, 6].
        mf_MiJt_a: ``H^{-1} J^T`` for body a per row [shape: world_count,
            mf_max_constraints, 6].
        mf_MiJt_b: ``H^{-1} J^T`` for body b per row [shape: world_count,
            mf_max_constraints, 6].
        mf_row_parent: Parent normal-row index for each friction row
            [shape: world_count, mf_max_constraints].
        mf_row_mu: Coulomb friction coefficient per row [shape:
            world_count, mf_max_constraints].
        mf_rhs: Baumgarte normal-row bias ``b_n`` (and 0 on friction
            rows) [shape: world_count, mf_max_constraints].
        body_to_articulation: Body-to-articulation index map.
        art_dof_start: First DOF index per articulation.
        use_de_saxce: When non-zero, augment the normal target velocity
            by ``μ · ‖c_T‖`` (de Saxce maximum-dissipation correction,
            Le Lidec & Carpentier 2024).  ``0`` matches the pure RAISim
            ``USE_BISECTION`` branch; ``1`` matches the
            ``USE_BISECTION + USE_DE_SAXCE`` branch.
        mf_impulses: Current matrix-free impulses; updated in place for
            the normal and second-friction siblings with the bisection
            solution [shape: world_count, mf_max_constraints].
        v_out: Generalized velocity buffer; updated in place with the
            normal and second-friction row velocity corrections [N].

    Returns:
        The projected friction impulse for row ``i`` [N·s].  On the
        first friction row this is the new ``λ_t1`` from the bisection;
        on the second friction row it is the pre-stored ``λ_t2`` so the
        outer PGS loop applies a zero delta.
    """
    parent_idx = mf_row_parent[world, i]
    i_n = parent_idx
    i_t1 = parent_idx + 1
    i_t2 = parent_idx + 2

    # Second friction row of the triple: the bisection already ran at
    # i == i_t1 and wrote mf_impulses[i_t2].  Returning that value makes
    # the outer kernel's ``delta_impulse = new - old`` a no-op.
    if i != i_t1:
        return mf_impulses[world, i]

    mu_val = mf_row_mu[world, i]
    ba = mf_body_a[world, i_n]
    bb = mf_body_b[world, i_n]

    # --- Recompute u_n, u_t1, u_t2 from current v_out --------------------
    u_n = float(0.0)
    u_t1 = float(0.0)
    u_t2 = float(0.0)

    ds_a = int(0)
    ds_b = int(0)
    if ba >= 0:
        art_a = body_to_articulation[ba]
        ds_a = art_dof_start[art_a]
        for k in range(6):
            va_k = v_out[ds_a + k]
            u_n = u_n + mf_J_a[world, i_n, k] * va_k
            u_t1 = u_t1 + mf_J_a[world, i_t1, k] * va_k
            u_t2 = u_t2 + mf_J_a[world, i_t2, k] * va_k
    if bb >= 0:
        art_b = body_to_articulation[bb]
        ds_b = art_dof_start[art_b]
        for k in range(6):
            vb_k = v_out[ds_b + k]
            u_n = u_n + mf_J_b[world, i_n, k] * vb_k
            u_t1 = u_t1 + mf_J_b[world, i_t1, k] * vb_k
            u_t2 = u_t2 + mf_J_b[world, i_t2, k] * vb_k

    # --- Build the 3x3 block G = J H^{-1} J^T ---------------------------
    G_nn = float(0.0)
    G_nt1 = float(0.0)
    G_nt2 = float(0.0)
    G_t1t1 = float(0.0)
    G_t1t2 = float(0.0)
    G_t2t2 = float(0.0)

    if ba >= 0:
        for k in range(6):
            Jna = mf_J_a[world, i_n, k]
            Jt1a = mf_J_a[world, i_t1, k]
            Jt2a = mf_J_a[world, i_t2, k]
            Mna = mf_MiJt_a[world, i_n, k]
            Mt1a = mf_MiJt_a[world, i_t1, k]
            Mt2a = mf_MiJt_a[world, i_t2, k]
            G_nn = G_nn + Jna * Mna
            G_nt1 = G_nt1 + Jna * Mt1a
            G_nt2 = G_nt2 + Jna * Mt2a
            G_t1t1 = G_t1t1 + Jt1a * Mt1a
            G_t1t2 = G_t1t2 + Jt1a * Mt2a
            G_t2t2 = G_t2t2 + Jt2a * Mt2a
    if bb >= 0:
        for k in range(6):
            Jnb = mf_J_b[world, i_n, k]
            Jt1b = mf_J_b[world, i_t1, k]
            Jt2b = mf_J_b[world, i_t2, k]
            Mnb = mf_MiJt_b[world, i_n, k]
            Mt1b = mf_MiJt_b[world, i_t1, k]
            Mt2b = mf_MiJt_b[world, i_t2, k]
            G_nn = G_nn + Jnb * Mnb
            G_nt1 = G_nt1 + Jnb * Mt1b
            G_nt2 = G_nt2 + Jnb * Mt2b
            G_t1t1 = G_t1t1 + Jt1b * Mt1b
            G_t1t2 = G_t1t2 + Jt1b * Mt2b
            G_t2t2 = G_t2t2 + Jt2b * Mt2b

    # Degenerate contact: leave impulses alone.
    if G_nn < 1.0e-20:
        return mf_impulses[world, i_t1]

    old_lambda_n = mf_impulses[world, i_n]
    old_lambda_t1 = mf_impulses[world, i_t1]
    old_lambda_t2 = mf_impulses[world, i_t2]

    # FeatherPGS stores ``rhs = beta * phi / dt`` on contact rows —
    # *negative* for penetration — because its PGS step solves
    # ``delta = -(J·v + rhs) / d_ii`` (converging to ``J·v + rhs = 0``).
    # RAISim's bisection instead targets ``u_n >= b_n`` with a
    # *positive* ``b_n = -erp * gap / dt`` for penetration.  Convert
    # once here so the rest of the bisection mirrors RAISim's sign
    # convention verbatim.
    target_vel_n = -mf_rhs[world, i_n]

    # De Saxce correction (Le Lidec & Carpentier 2024): when requested,
    # augment the normal target by ``μ · ‖c_T‖`` where ``c_T`` is the
    # current (pre-solve) tangential velocity.  This enforces the
    # maximum-dissipation principle on sliding contacts and drives the
    # ``r_mdp_dir`` / ``r_ds_compl`` residuals down — see
    # ``raisim/kernels.py`` (lines ~687-698) for the reference impl.
    if use_de_saxce != 0:
        c_T_mag = wp.sqrt(u_t1 * u_t1 + u_t2 * u_t2)
        target_vel_n = target_vel_n + mu_val * c_T_mag

    new_lambda_n = old_lambda_n
    new_lambda_t1 = old_lambda_t1
    new_lambda_t2 = old_lambda_t2

    # Check whether lambda_n = 0 already satisfies complementarity
    # (separating contact).  ``u_n_at_zero`` is the normal velocity we
    # would see if we removed the current normal impulse entirely.
    u_n_at_zero = u_n + G_nn * (0.0 - old_lambda_n)
    if u_n_at_zero >= target_vel_n:
        new_lambda_n = 0.0
        new_lambda_t1 = 0.0
        new_lambda_t2 = 0.0
    else:
        # --- Bisect lambda_n in [lo, hi] --------------------------------
        lo = float(0.0)
        hi = wp.max(old_lambda_n * 2.0, (target_vel_n - u_n) / G_nn + old_lambda_n)
        hi = wp.max(hi, 1.0)

        for _bi in range(wp.static(_FPGS_BISECTION_ITERS)):
            mid = 0.5 * (lo + hi)
            d_n = mid - old_lambda_n
            ut1_eff = u_t1 + G_nt1 * d_n
            ut2_eff = u_t2 + G_nt2 * d_n

            det = G_t1t1 * G_t2t2 - G_t1t2 * G_t1t2
            d_t1 = float(0.0)
            d_t2 = float(0.0)
            if wp.abs(det) > 1.0e-20:
                d_t1 = (-ut1_eff * G_t2t2 + ut2_eff * G_t1t2) / det
                d_t2 = (ut1_eff * G_t1t2 - ut2_eff * G_t1t1) / det

            trial_t1 = old_lambda_t1 + d_t1
            trial_t2 = old_lambda_t2 + d_t2

            flimit = mu_val * mid
            tmag = wp.sqrt(trial_t1 * trial_t1 + trial_t2 * trial_t2)
            if tmag > flimit and tmag > 1.0e-20:
                sc = flimit / tmag
                trial_t1 = trial_t1 * sc
                trial_t2 = trial_t2 * sc

            d_t1_actual = trial_t1 - old_lambda_t1
            d_t2_actual = trial_t2 - old_lambda_t2
            u_n_trial = u_n + G_nn * d_n + G_nt1 * d_t1_actual + G_nt2 * d_t2_actual

            if u_n_trial < target_vel_n:
                lo = mid
            else:
                hi = mid

        new_lambda_n = 0.5 * (lo + hi)

        # --- Final friction solve at converged lambda_n -----------------
        d_n_final = new_lambda_n - old_lambda_n
        ut1_f = u_t1 + G_nt1 * d_n_final
        ut2_f = u_t2 + G_nt2 * d_n_final

        det_f = G_t1t1 * G_t2t2 - G_t1t2 * G_t1t2
        d_t1_f = float(0.0)
        d_t2_f = float(0.0)
        if wp.abs(det_f) > 1.0e-20:
            d_t1_f = (-ut1_f * G_t2t2 + ut2_f * G_t1t2) / det_f
            d_t2_f = (ut1_f * G_t1t2 - ut2_f * G_t1t1) / det_f

        new_lambda_t1 = old_lambda_t1 + d_t1_f
        new_lambda_t2 = old_lambda_t2 + d_t2_f

        flimit_f = mu_val * new_lambda_n
        tmag_f = wp.sqrt(new_lambda_t1 * new_lambda_t1 + new_lambda_t2 * new_lambda_t2)
        if tmag_f > flimit_f and tmag_f > 1.0e-20:
            sc_f = flimit_f / tmag_f
            new_lambda_t1 = new_lambda_t1 * sc_f
            new_lambda_t2 = new_lambda_t2 * sc_f

    # --- Apply v_out deltas for normal and t2 rows ----------------------
    d_n_total = new_lambda_n - old_lambda_n
    d_t2_total = new_lambda_t2 - old_lambda_t2

    if ba >= 0:
        for k in range(6):
            v_out[ds_a + k] = (
                v_out[ds_a + k] + mf_MiJt_a[world, i_n, k] * d_n_total + mf_MiJt_a[world, i_t2, k] * d_t2_total
            )
    if bb >= 0:
        for k in range(6):
            v_out[ds_b + k] = (
                v_out[ds_b + k] + mf_MiJt_b[world, i_n, k] * d_n_total + mf_MiJt_b[world, i_t2, k] * d_t2_total
            )

    # --- Store normal + second-friction impulses ------------------------
    # The first-friction row's impulse is returned so the outer PGS loop
    # applies its own ``delta_impulse`` and v_out correction.
    mf_impulses[world, i_n] = new_lambda_n
    mf_impulses[world, i_t2] = new_lambda_t2

    return new_lambda_t1


@wp.func
def friction_step_coulomb_newton(
    world: int,
    i: int,
    mf_body_a: wp.array2d[int],
    mf_body_b: wp.array2d[int],
    mf_J_a: wp.array3d[float],
    mf_J_b: wp.array3d[float],
    mf_MiJt_a: wp.array3d[float],
    mf_MiJt_b: wp.array3d[float],
    mf_row_parent: wp.array2d[int],
    mf_row_mu: wp.array2d[float],
    mf_rhs: wp.array2d[float],
    body_to_articulation: wp.array[int],
    art_dof_start: wp.array[int],
    mf_impulses: wp.array2d[float],
    v_out: wp.array[float],
):
    """Gilles Daviet's 1D Coulomb Newton per-row friction step.

    Drop-in replacement for :func:`friction_step_current` selected by
    ``friction_mode="coulomb_newton"`` on ``pgs_mode="matrix_free"``.
    FPGS Friction Modes 7/13.  Invokes :func:`solve_coulomb_row`
    (ported from ``coulomb_root_finding_warp.py``) on the per-contact
    3x3 effective-mass (Delassus) block and rewrites the contact
    triple's impulses directly from its ``(r_N, r_T1, r_T2)`` return.
    Shares the matrix-free row data layout with
    :func:`friction_step_bisection`.

    The matrix-free layout stores each contact as three consecutive
    rows (normal + two friction).  The 1D Newton is naturally a
    per-contact 3-DOF solve, so this function centralises the work on
    the first friction row (``i == parent + 1``) and becomes a no-op
    on the second friction row (``i == parent + 2``) — same pattern as
    the bisection step.

    Mapping of matrix-free row data to the reference ``solve_coulomb``
    interface (docstring required by the acceptance criterion):

    * ``W`` is the per-contact ``3x3`` Delassus block ``G = J H^{-1}
      J^T`` assembled on the fly from ``mf_J_*`` / ``mf_MiJt_*``.
      Layout: ``W[0, 0] = G_nn`` (normal/normal), ``W[0, 1:]`` =
      ``W[1:, 0]`` = ``(G_nt1, G_nt2)`` (normal/tangential coupling),
      and the remaining ``2x2`` tangential block is populated from the
      two friction rows' dot products.
    * ``b`` is the velocity-level rhs.  The matrix-free PGS stores the
      Baumgarte bias as ``mf_rhs[i_n] = beta * phi / dt`` (negative on
      penetration), while the reference expects ``b_N < 0`` on
      contacts that would penetrate at zero impulse.  We set
      ``target_vel_n = -mf_rhs[i_n]`` to convert the FPGS convention
      to RAISim's positive-target convention (matching
      :func:`friction_step_bisection`), then use
      ``b_N = u_free_n - target_vel_n`` so the ``b_N < 0`` invariant
      holds in the shifted frame.  ``b[1:]`` are the tangential
      components of ``u_free`` (no bias on friction rows).
    * ``u_free = u_current - G · lambda_old`` is recomputed from
      ``v_out`` (``J_row · v_out``) and the current impulses so the
      step is Gauss-Seidel consistent with the rest of the sweep.
    * ``mu`` comes directly from ``mf_row_mu[i]``.

    The returned ``(r_N, r_T1, r_T2)`` are full (non-delta) impulses
    for the triple.  We write them to ``mf_impulses`` for the normal
    and second-friction rows directly, apply the corresponding
    ``v_out`` corrections, and return ``r_T1`` as the first-friction
    row's new impulse so the outer PGS loop applies the t1
    ``delta_impulse`` via its usual path.

    Args:
        world: World index for the current row.
        i: Constraint row index within the world.
        mf_body_a: Matrix-free body-a indices [shape: world_count,
            mf_max_constraints].
        mf_body_b: Matrix-free body-b indices [shape: world_count,
            mf_max_constraints].
        mf_J_a: Per-row Jacobian ``J`` for body a [shape: world_count,
            mf_max_constraints, 6].
        mf_J_b: Per-row Jacobian ``J`` for body b [shape: world_count,
            mf_max_constraints, 6].
        mf_MiJt_a: ``H^{-1} J^T`` for body a per row [shape:
            world_count, mf_max_constraints, 6].
        mf_MiJt_b: ``H^{-1} J^T`` for body b per row [shape:
            world_count, mf_max_constraints, 6].
        mf_row_parent: Parent normal-row index for each friction row
            [shape: world_count, mf_max_constraints].
        mf_row_mu: Coulomb friction coefficient per row [shape:
            world_count, mf_max_constraints].
        mf_rhs: Baumgarte normal-row bias ``beta * phi / dt`` (and 0
            on friction rows) [shape: world_count, mf_max_constraints].
        body_to_articulation: Body-to-articulation index map.
        art_dof_start: First DOF index per articulation.
        mf_impulses: Current matrix-free impulses; updated in place
            for the normal and second-friction siblings with the
            Newton solution [shape: world_count, mf_max_constraints].
        v_out: Generalized velocity buffer; updated in place with the
            normal and second-friction row velocity corrections [N].

    Returns:
        The projected friction impulse for row ``i`` [N·s].  On the
        first friction row this is ``r_T1``; on the second friction
        row it is the pre-stored ``r_T2`` so the outer PGS loop
        applies a zero delta.
    """
    parent_idx = mf_row_parent[world, i]
    i_n = parent_idx
    i_t1 = parent_idx + 1
    i_t2 = parent_idx + 2

    # Second friction row of the triple: the Newton solve already ran
    # at ``i == i_t1`` and wrote ``mf_impulses[i_t2]``.  Returning that
    # value makes the outer kernel's ``delta_impulse = new - old`` a
    # no-op.
    if i != i_t1:
        return mf_impulses[world, i]

    mu_val = mf_row_mu[world, i]
    ba = mf_body_a[world, i_n]
    bb = mf_body_b[world, i_n]

    # --- Recompute u_n, u_t1, u_t2 from current v_out --------------------
    u_n = float(0.0)
    u_t1 = float(0.0)
    u_t2 = float(0.0)

    ds_a = int(0)
    ds_b = int(0)
    if ba >= 0:
        art_a = body_to_articulation[ba]
        ds_a = art_dof_start[art_a]
        for k in range(6):
            va_k = v_out[ds_a + k]
            u_n = u_n + mf_J_a[world, i_n, k] * va_k
            u_t1 = u_t1 + mf_J_a[world, i_t1, k] * va_k
            u_t2 = u_t2 + mf_J_a[world, i_t2, k] * va_k
    if bb >= 0:
        art_b = body_to_articulation[bb]
        ds_b = art_dof_start[art_b]
        for k in range(6):
            vb_k = v_out[ds_b + k]
            u_n = u_n + mf_J_b[world, i_n, k] * vb_k
            u_t1 = u_t1 + mf_J_b[world, i_t1, k] * vb_k
            u_t2 = u_t2 + mf_J_b[world, i_t2, k] * vb_k

    # --- Build the 3x3 Delassus block G = J H^{-1} J^T ------------------
    G_nn = float(0.0)
    G_nt1 = float(0.0)
    G_nt2 = float(0.0)
    G_t1t1 = float(0.0)
    G_t1t2 = float(0.0)
    G_t2t2 = float(0.0)

    if ba >= 0:
        for k in range(6):
            Jna = mf_J_a[world, i_n, k]
            Jt1a = mf_J_a[world, i_t1, k]
            Jt2a = mf_J_a[world, i_t2, k]
            Mna = mf_MiJt_a[world, i_n, k]
            Mt1a = mf_MiJt_a[world, i_t1, k]
            Mt2a = mf_MiJt_a[world, i_t2, k]
            G_nn = G_nn + Jna * Mna
            G_nt1 = G_nt1 + Jna * Mt1a
            G_nt2 = G_nt2 + Jna * Mt2a
            G_t1t1 = G_t1t1 + Jt1a * Mt1a
            G_t1t2 = G_t1t2 + Jt1a * Mt2a
            G_t2t2 = G_t2t2 + Jt2a * Mt2a
    if bb >= 0:
        for k in range(6):
            Jnb = mf_J_b[world, i_n, k]
            Jt1b = mf_J_b[world, i_t1, k]
            Jt2b = mf_J_b[world, i_t2, k]
            Mnb = mf_MiJt_b[world, i_n, k]
            Mt1b = mf_MiJt_b[world, i_t1, k]
            Mt2b = mf_MiJt_b[world, i_t2, k]
            G_nn = G_nn + Jnb * Mnb
            G_nt1 = G_nt1 + Jnb * Mt1b
            G_nt2 = G_nt2 + Jnb * Mt2b
            G_t1t1 = G_t1t1 + Jt1b * Mt1b
            G_t1t2 = G_t1t2 + Jt1b * Mt2b
            G_t2t2 = G_t2t2 + Jt2b * Mt2b

    # Degenerate contact: leave impulses alone.
    if G_nn < 1.0e-20:
        return mf_impulses[world, i_t1]

    old_lambda_n = mf_impulses[world, i_n]
    old_lambda_t1 = mf_impulses[world, i_t1]
    old_lambda_t2 = mf_impulses[world, i_t2]

    # --- u_free = u_current - G * lambda_old (velocity with impulses
    # removed).  Mirrors the frame ``solve_coulomb_row`` expects: b is
    # the rhs at zero impulse.
    u_free_n = u_n - (G_nn * old_lambda_n + G_nt1 * old_lambda_t1 + G_nt2 * old_lambda_t2)
    u_free_t1 = u_t1 - (G_nt1 * old_lambda_n + G_t1t1 * old_lambda_t1 + G_t1t2 * old_lambda_t2)
    u_free_t2 = u_t2 - (G_nt2 * old_lambda_n + G_t1t2 * old_lambda_t1 + G_t2t2 * old_lambda_t2)

    # FeatherPGS stores rhs = beta * phi / dt (negative on penetration).
    # Shift to the positive-target convention (``target_vel_n = -rhs``)
    # so ``b_N < 0`` iff the contact would penetrate at zero impulse.
    target_vel_n = -mf_rhs[world, i_n]

    # If the contact separates already (u_free_n >= target_vel_n), the
    # Coulomb problem has the trivial solution λ = 0.  Short-circuit
    # to avoid the full Newton solve.
    new_lambda_n = float(0.0)
    new_lambda_t1 = float(0.0)
    new_lambda_t2 = float(0.0)

    if u_free_n < target_vel_n:
        W = wp.mat33(
            G_nn,
            G_nt1,
            G_nt2,
            G_nt1,
            G_t1t1,
            G_t1t2,
            G_nt2,
            G_t1t2,
            G_t2t2,
        )
        b_vec = wp.vec3(u_free_n - target_vel_n, u_free_t1, u_free_t2)
        res = solve_coulomb_row(W, b_vec, mu_val)
        new_lambda_n = res[1]
        new_lambda_t1 = res[2]
        new_lambda_t2 = res[3]

    # --- Apply v_out deltas for the normal and t2 rows ------------------
    d_n_total = new_lambda_n - old_lambda_n
    d_t2_total = new_lambda_t2 - old_lambda_t2

    if ba >= 0:
        for k in range(6):
            v_out[ds_a + k] = (
                v_out[ds_a + k] + mf_MiJt_a[world, i_n, k] * d_n_total + mf_MiJt_a[world, i_t2, k] * d_t2_total
            )
    if bb >= 0:
        for k in range(6):
            v_out[ds_b + k] = (
                v_out[ds_b + k] + mf_MiJt_b[world, i_n, k] * d_n_total + mf_MiJt_b[world, i_t2, k] * d_t2_total
            )

    # Store normal + second-friction impulses.  The first-friction
    # row's impulse is returned so the outer PGS loop applies its own
    # ``delta_impulse`` and v_out correction.
    mf_impulses[world, i_n] = new_lambda_n
    mf_impulses[world, i_t2] = new_lambda_t2

    return new_lambda_t1


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
    mf_row_w: wp.array2d[float],
    body_to_articulation: wp.array[int],
    art_dof_start: wp.array[int],
    iterations: int,
    omega: float,
    regularize: int,
    friction_mode: int,
    friction_start_iteration: int,
    iteration_offset: int,
    # in/out
    mf_impulses: wp.array2d[float],
    v_out: wp.array[float],
):
    """Matrix-free PGS solver for free rigid body contacts.

    Operates directly on body velocities stored in v_out (generalized coordinates).
    Each iteration recomputes J*v from v_out and applies velocity corrections
    immediately (Gauss-Seidel style).

    The per-row Coulomb friction projection is delegated to
    :func:`friction_step_current` (``friction_mode == FRICTION_MODE_CURRENT``),
    :func:`friction_step_bisection` (``friction_mode ==
    FRICTION_MODE_BISECTION`` for pure RAISim bisection, or
    ``friction_mode == FRICTION_MODE_BISECTION_DESAXCE`` for bisection
    augmented with the de Saxce max-dissipation bias), or
    :func:`friction_step_coulomb_newton` (``friction_mode ==
    FRICTION_MODE_COULOMB_NEWTON`` for Gilles Daviet's 1D Coulomb
    Newton — FPGS Friction Modes 7/13) so alternate strategies can be
    plugged in without rewriting this kernel body.
    """
    world = wp.tid()
    m_count = mf_constraint_count[world]
    if m_count == 0:
        return

    for it in range(iterations):
        for i in range(m_count):
            row_type = mf_row_type[world, i]
            if row_type == PGS_CONSTRAINT_TYPE_FRICTION and iteration_offset + it < friction_start_iteration:
                mf_impulses[world, i] = 0.0
                continue

            eff_inv = mf_eff_mass_inv[world, i]
            if eff_inv <= 0.0 and not (
                row_type == PGS_CONSTRAINT_TYPE_FRICTION and friction_mode == FRICTION_MODE_CURRENT
            ):
                continue

            ba = mf_body_a[world, i]
            bb = mf_body_b[world, i]

            # Compute current J * v
            jv = float(0.0)
            if ba >= 0:
                art_a = body_to_articulation[ba]
                ds_a = art_dof_start[art_a]
                for k in range(6):
                    jv += mf_J_a[world, i, k] * v_out[ds_a + k]
            if bb >= 0:
                art_b = body_to_articulation[bb]
                ds_b = art_dof_start[art_b]
                for k in range(6):
                    jv += mf_J_b[world, i, k] * v_out[ds_b + k]

            # PGS update: delta = -(J*v_current + bias) / d_ii
            residual = jv + mf_rhs[world, i]
            old_impulse = mf_impulses[world, i]
            delta = -residual * eff_inv
            if row_type == PGS_CONSTRAINT_TYPE_CONTACT:
                # Proximal regularization in w-form (w = 1/(1+g)): identical
                # algebra, but no g*lambda product that could overflow for
                # extreme finite g; w = 1 (g = 0) is the exact hard update.
                w = float(1.0)
                if regularize != 0:
                    w = mf_row_w[world, i]
                delta = -residual * eff_inv * w - (1.0 - w) * old_impulse
            new_impulse = old_impulse + omega * delta
            delta_impulse = float(0.0)

            # Project
            if row_type == PGS_CONSTRAINT_TYPE_CONTACT:
                if new_impulse < 0.0:
                    new_impulse = 0.0
            elif row_type == PGS_CONSTRAINT_TYPE_JOINT_VELOCITY_LIMIT:
                if residual < 0.0:
                    delta_impulse = delta
                    new_impulse = delta
                else:
                    new_impulse = 0.0
            elif row_type == PGS_CONSTRAINT_TYPE_FRICTION:
                if friction_mode == FRICTION_MODE_BISECTION or friction_mode == FRICTION_MODE_BISECTION_DESAXCE:
                    # Shared RAISim bisection step; the de Saxce branch
                    # of the [FPGS Friction Modes] series toggles the
                    # μ·‖c_T‖ bias correction via ``use_de_saxce``.
                    use_de_saxce = int(0)
                    if friction_mode == FRICTION_MODE_BISECTION_DESAXCE:
                        use_de_saxce = int(1)
                    new_impulse = friction_step_bisection(
                        world,
                        i,
                        mf_body_a,
                        mf_body_b,
                        mf_J_a,
                        mf_J_b,
                        mf_MiJt_a,
                        mf_MiJt_b,
                        mf_row_parent,
                        mf_row_mu,
                        mf_rhs,
                        body_to_articulation,
                        art_dof_start,
                        use_de_saxce,
                        mf_impulses,
                        v_out,
                    )
                elif friction_mode == FRICTION_MODE_COULOMB_NEWTON:
                    # Gilles Daviet's 1D Coulomb Newton (7/13): scalar
                    # bracketed-Newton on alpha solves the cone coupling
                    # directly.  See :func:`friction_step_coulomb_newton`.
                    new_impulse = friction_step_coulomb_newton(
                        world,
                        i,
                        mf_body_a,
                        mf_body_b,
                        mf_J_a,
                        mf_J_b,
                        mf_MiJt_a,
                        mf_MiJt_b,
                        mf_row_parent,
                        mf_row_mu,
                        mf_rhs,
                        body_to_articulation,
                        art_dof_start,
                        mf_impulses,
                        v_out,
                    )
                else:
                    new_impulse = friction_step_current(
                        world,
                        i,
                        omega,
                        mf_eff_mass_inv,
                        mf_rhs,
                        mf_J_a,
                        mf_J_b,
                        mf_body_a,
                        mf_body_b,
                        mf_MiJt_a,
                        mf_MiJt_b,
                        mf_row_parent,
                        mf_row_mu,
                        body_to_articulation,
                        art_dof_start,
                        mf_impulses,
                        v_out,
                    )

            if row_type != PGS_CONSTRAINT_TYPE_JOINT_VELOCITY_LIMIT:
                delta_impulse = new_impulse - old_impulse
            mf_impulses[world, i] = new_impulse

            # Apply velocity correction: v += M_inv * J^T * delta_impulse
            if ba >= 0:
                art_a2 = body_to_articulation[ba]
                ds_a2 = art_dof_start[art_a2]
                for k in range(6):
                    v_out[ds_a2 + k] = v_out[ds_a2 + k] + mf_MiJt_a[world, i, k] * delta_impulse
            if bb >= 0:
                art_b2 = body_to_articulation[bb]
                ds_b2 = art_dof_start[art_b2]
                for k in range(6):
                    v_out[ds_b2 + k] = v_out[ds_b2 + k] + mf_MiJt_b[world, i, k] * delta_impulse


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
    """Build per-world local body table and local body index mapping.

    Scans all MF constraint body indices, builds a unique body list per world,
    and maps each constraint's body indices to local indices.
    """
    world = wp.tid()
    m = mf_constraint_count[world]
    if m == 0:
        mf_body_count[world] = 0
        return

    n_bodies = int(0)

    for i in range(m):
        # Process body A
        ba = mf_body_a[world, i]
        if ba >= 0:
            # Search for ba in body_list
            found_a = int(-1)
            for b in range(n_bodies):
                if mf_body_list[world, b] == ba:
                    found_a = b
                    break
            if found_a < 0 and n_bodies < max_mf_bodies:
                found_a = n_bodies
                mf_body_list[world, n_bodies] = ba
                art_a = body_to_articulation[ba]
                mf_body_dof_start[world, n_bodies] = art_dof_start[art_a]
                n_bodies += 1
            mf_local_body_a[world, i] = found_a
        else:
            mf_local_body_a[world, i] = -1

        # Process body B
        bb = mf_body_b[world, i]
        if bb >= 0:
            found_b = int(-1)
            for b in range(n_bodies):
                if mf_body_list[world, b] == bb:
                    found_b = b
                    break
            if found_b < 0 and n_bodies < max_mf_bodies:
                found_b = n_bodies
                mf_body_list[world, n_bodies] = bb
                mf_body_dof_start[world, n_bodies] = art_dof_start[body_to_articulation[bb]]
                n_bodies += 1
            mf_local_body_b[world, i] = found_b
        else:
            mf_local_body_b[world, i] = -1

    mf_body_count[world] = n_bodies


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
def pgs_solve_loop(
    world_constraint_count: wp.array[int],
    max_constraints: int,
    world_diag: wp.array2d[float],
    world_C: wp.array3d[float],
    world_rhs: wp.array2d[float],
    world_impulses: wp.array2d[float],
    iterations: int,
    omega: float,
    world_row_type: wp.array2d[int],
    world_row_parent: wp.array2d[int],
    world_row_mu: wp.array2d[float],
    friction_start_iteration: int,
    iteration_offset: int,
):
    """
    World-level Projected Gauss-Seidel solver.

    Similar to pgs_solve_contacts but operates on 2D world-indexed arrays.
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

            # Compute residual: w = rhs_i + sum_j C_ij * lambda_j
            w = world_rhs[world, i]
            for j in range(m):
                w += world_C[world, i, j] * world_impulses[world, j]

            denom = world_diag[world, i]
            if denom <= 0.0 and row_type != PGS_CONSTRAINT_TYPE_FRICTION:
                continue

            delta = float(0.0)
            if denom > 0.0:
                delta = -w / denom
            new_impulse = world_impulses[world, i] + omega * delta

            # --- Normal contact, joint limit, or joint velocity limit:
            #     lambda_n >= 0. The velocity-limit row uses a signed Jacobian
            #     so the unilateral projector handles both sides of the
            #     bilateral ``[-qdot_max, +qdot_max]`` box.
            if (
                row_type == PGS_CONSTRAINT_TYPE_CONTACT
                or row_type == PGS_CONSTRAINT_TYPE_JOINT_LIMIT
                or row_type == PGS_CONSTRAINT_TYPE_JOINT_VELOCITY_LIMIT
            ):
                if new_impulse < 0.0:
                    new_impulse = 0.0
                world_impulses[world, i] = new_impulse

            # --- Friction: isotropic Coulomb ---
            elif row_type == PGS_CONSTRAINT_TYPE_FRICTION:
                parent_idx = world_row_parent[world, i]
                lambda_n = patch_normal_load(world_row_parent, world_impulses, world, parent_idx)
                mu = world_row_mu[world, i]
                radius = wp.max(mu * lambda_n, 0.0)

                if radius <= 0.0:
                    world_impulses[world, i] = 0.0
                    world_impulses[world, i + 1] = 0.0
                    continue

                if i != parent_idx + 1:
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
                a = trial[0]
                b = trial[1]

                mag = wp.sqrt(a * a + b * b)
                if mag > radius:
                    scale = radius / mag
                    a *= scale
                    b *= scale
                world_impulses[world, i] = a
                world_impulses[world, sib] = b

            else:
                world_impulses[world, i] = new_impulse


@wp.kernel
def apply_impulses_world_par_dof(
    group_to_art: wp.array[int],
    art_to_world: wp.array[int],
    art_dof_start: wp.array[int],
    n_dofs: int,
    n_arts: int,
    world_constraint_count: wp.array[int],
    max_constraints: int,
    Y_group: wp.array3d[float],
    world_impulses: wp.array2d[float],
    v_hat: wp.array[float],
    # outputs
    v_out: wp.array[float],
):
    """
    Accumulate velocity changes from world impulses for a single size group.
    DOF-parallelized: each thread handles one (articulation, DOF) pair.

    v_out = v_hat + Y * impulses
    """
    tid = wp.tid()

    # Decode thread index
    local_dof = tid % n_dofs
    idx = tid // n_dofs  # group index

    if idx >= n_arts:
        return

    art = group_to_art[idx]
    world = art_to_world[art]
    n_constraints = world_constraint_count[world]
    dof_start = art_dof_start[art]

    # Inner loop only over constraints
    delta_v = float(0.0)
    for c in range(n_constraints):
        delta_v += Y_group[idx, c, local_dof] * world_impulses[world, c]

    global_dof = dof_start + local_dof
    v_out[global_dof] = v_hat[global_dof] + delta_v


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
    set_output_to_delta: int,
    # in/out
    velocity: wp.array[float],
):
    """Apply cached MF impulses so live velocity and lambda share one state.

    PGS updates velocity by ``M^-1 J^T (lambda_new-lambda_old)``.  A nonzero
    warm-start lambda must therefore have its existing velocity contribution
    installed before the first iteration; otherwise retracting a stale normal
    impulse creates an attractive impulse.
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

    if set_output_to_delta != 0:
        velocity[global_dof] = delta_velocity
    else:
        velocity[global_dof] += delta_velocity


@wp.kernel
def finalize_world_diag_cfm(
    world_constraint_count: wp.array[int],
    world_row_cfm: wp.array2d[float],
    # in/out
    world_diag: wp.array2d[float],
):
    """Add CFM to world diagonal after Delassus accumulation."""
    world = wp.tid()
    m = world_constraint_count[world]

    for i in range(m):
        world_diag[world, i] += world_row_cfm[world, i]


@wp.kernel
def apply_contact_regularization(
    world_constraint_count: wp.array[int],
    world_row_type: wp.array2d[int],
    world_row_w: wp.array2d[float],
    # in/out
    world_diag: wp.array2d[float],
    world_C: wp.array3d[float],
):
    """Fold the per-row weight of dense contact rows into the assembled Delassus system.

    Adding ``reg = (1/w - 1) * d`` to the matrix diagonal and the divisor is the
    w-form update ``delta = -w*r/d - (1-w)*lambda`` used by every other route.
    Only the split mode, which solves with the assembled matrix, needs this; the
    fused velocity-residual routes read ``world_row_w`` directly.
    """
    world = wp.tid()
    m = world_constraint_count[world]
    for i in range(m):
        if world_row_type[world, i] == PGS_CONSTRAINT_TYPE_CONTACT:
            w = world_row_w[world, i]
            if w < 1.0:
                reg = (1.0 / w - 1.0) * world_diag[world, i]
                world_diag[world, i] += reg
                world_C[world, i, i] += reg


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


@wp.kernel(module=_MASS_DYNAMICS_KERNEL_MODULE)
def hinv_jt_diagonal(
    L_group: wp.array3d[float],
    J_group: wp.array3d[float],
    group_to_art: wp.array[int],
    art_to_world: wp.array[int],
    articulation_world_dof_offset: wp.array[int],
    world_constraint_count: wp.array[int],
    n_dofs: int,
    max_constraints: int,
    n_arts: int,
    write_group: int,
    write_world: int,
    compute_diag: int,
    # outputs
    Y_group: wp.array3d[float],
    J_world: wp.array3d[float],
    Y_world: wp.array3d[float],
    diag_group: wp.array2d[float],
):
    """Apply a structurally diagonal inverse to one Jacobian row."""
    element = wp.tid()
    constraint = element % max_constraints
    group = element // max_constraints
    if group >= n_arts:
        return

    art = group_to_art[group]
    world = art_to_world[art]
    if constraint >= world_constraint_count[world]:
        return

    dof_offset = int(0)
    if write_world != 0:
        dof_offset = articulation_world_dof_offset[art]
    diagonal_sum = float(0.0)
    for dof in range(n_dofs):
        jacobian = J_group[group, constraint, dof]
        factor = L_group[group, dof, dof]
        response = jacobian / factor
        response = response / factor
        if write_group != 0:
            Y_group[group, constraint, dof] = response
        if write_world != 0:
            J_world[world, constraint, dof_offset + dof] = jacobian
            Y_world[world, constraint, dof_offset + dof] = response
        if compute_diag != 0:
            diagonal_sum += jacobian * response

    if compute_diag != 0:
        diag_group[group, constraint] = diagonal_sum


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
    max_constraints: int,
    n_arts: int,
    write_world: int,
    Y_group: wp.array3d[float],
    J_world: wp.array3d[float],
    Y_world: wp.array3d[float],
):
    """Compute response only for worlds that need the general contact solver."""
    tid = wp.tid()
    group_index = tid // 32
    lane = tid % 32
    if group_index >= n_arts:
        return

    art = group_to_art[group_index]
    world = art_to_world[art]
    constraint_count = world_constraint_count[world]
    if local_solve_owner[world] != PGS_LOCAL_SOLVE_OWNER_GENERAL:
        # Local owners construct response in their fused solver.  Only impact
        # rows need a world Jacobian for the separate restitution target pass;
        # keep every other local row in articulation-local storage.
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
    dense_phase_bounds: wp.array2d[int],
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
    """Assign exact solver ownership and compact active general worlds."""
    world = wp.tid()
    row_count = world_constraint_count[world]
    mf_count = mf_constraint_count[world]
    primary_articulation = local_primary_articulation[world]
    pair_articulation = local_pair_articulation[world]
    residual_pair_articulation = local_residual_pair_articulation[world]
    single_phase = dense_phase_bounds[world, 1] == row_count

    local_mf = mf_count > 0 and mf_count <= local_residual_mf_max_constraints and residual_pair_articulation >= 0
    mf_row = int(0)
    while mf_row < mf_count and local_mf:
        body_a = mf_body_a[world, mf_row]
        body_b = mf_body_b[world, mf_row]
        if body_a >= 0 and body_to_articulation[body_a] != residual_pair_articulation:
            local_mf = False
        if body_b >= 0 and body_to_articulation[body_b] != residual_pair_articulation:
            local_mf = False
        # Free-body velocity-limit rows follow the general owner's stateless unilateral law, which the
        # residual local loop does not implement: keep such worlds on the general owner.
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
    """Compact topology candidates that selected paired local ownership."""
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
    """Discard stale response diagonals for locally owned worlds."""
    tid = wp.tid()
    row = tid % max_constraints
    world = tid // max_constraints
    if local_solve_owner[world] != PGS_LOCAL_SOLVE_OWNER_GENERAL and row < world_constraint_count[world]:
        world_diag[world, row] = 0.0


@wp.kernel
def delassus_par_row_col(
    # Size-grouped arrays
    J_group: wp.array3d[float],  # [n_arts_of_size, max_constraints, n_dofs]
    Y_group: wp.array3d[float],  # [n_arts_of_size, max_constraints, n_dofs]
    # Indirection arrays
    group_to_art: wp.array[int],
    art_to_world: wp.array[int],
    world_constraint_count: wp.array[int],
    # Size parameters
    n_dofs: int,
    max_constraints: int,
    n_arts: int,
    # Output: Delassus matrix C and diagonal (accumulated via atomics)
    world_C: wp.array3d[float],  # [world_count, max_constraints, max_constraints]
    world_diag: wp.array2d[float],  # [world_count, max_constraints]
):
    """
    Accumulate Delassus matrix contribution C += J * Y^T from one size group.

    PARALLELIZED VERSION: Each thread handles one (articulation, i, j) triplet.

    The Delassus matrix is: C = sum_art J_art * H_art^-1 * J_art^T = sum_art J_art * Y_art^T

    Since Y is stored as [constraint, dof], we compute:
        C[i,j] = sum_k J[i,k] * Y[j,k]

    Thread dimension: n_arts_of_size * max_constraints * max_constraints
    """
    tid = wp.tid()

    # Decode thread index
    j = tid % max_constraints
    i = (tid // max_constraints) % max_constraints
    idx = tid // (max_constraints * max_constraints)

    # Bounds check for articulation
    if idx >= n_arts:
        return

    art = group_to_art[idx]
    world = art_to_world[art]
    n_constraints = world_constraint_count[world]

    # Early exit if this (i, j) is beyond the actual constraint count
    if i >= n_constraints or j >= n_constraints:
        return

    # Compute C[i,j] = sum_k J[i,k] * Y[j,k]
    val = float(0.0)
    for k in range(n_dofs):
        val += J_group[idx, i, k] * Y_group[idx, j, k]

    if val != 0.0:
        wp.atomic_add(world_C, world, i, j, val)

    # Also accumulate diagonal separately (only when i == j)
    if i == j and val != 0.0:
        wp.atomic_add(world_diag, world, i, val)


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
    fused_augmented_drive: int,
    drive_row_by_dof: wp.array[int],
    row_K: wp.array[float],
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

            # The column thread uniquely owns its diagonal. Fold the
            # augmented-drive inertia into that write so the immutable drive
            # topology does not require a second read/modify/write pass over H.
            if fused_augmented_drive != 0 and row_idx == col_idx:
                drive_row = drive_row_by_dof[target_dof_global]
                if drive_row >= 0:
                    K = row_K[drive_row]
                    if K > 0.0:
                        val += K

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
    # output
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


@wp.kernel
def vector_add_inplace(a: wp.array[float], b: wp.array[float]):
    """a[i] += b[i]"""
    i = wp.tid()
    a[i] = a[i] + b[i]


@wp.kernel
def compute_delta_and_accumulate(
    v_out: wp.array[float],
    v_snap: wp.array[float],
    v_accum: wp.array[float],
):
    """delta = v_out - v_snap; v_accum += delta; v_snap = delta (reuse buffer for rhs_accum input)"""
    i = wp.tid()
    delta = v_out[i] - v_snap[i]
    v_accum[i] = v_accum[i] + delta
    v_snap[i] = delta


@wp.func
def world_response_velocity(
    v_out: wp.array[float],
    world_dof_indices: wp.array2d[int],
    world: int,
    local_dof: int,
) -> float:
    """Read a compact response velocity, returning zero for padding."""
    global_dof = world_dof_indices[world, local_dof]
    if global_dof >= 0:
        return v_out[global_dof]
    return 0.0


# =============================================================================
# PGS Convergence Diagnostic Kernel (velocity-space mode)
# =============================================================================


@wp.kernel
def pgs_convergence_diagnostic_velocity(
    # Dense constraints
    constraint_count: wp.array[int],
    world_dof_indices: wp.array2d[int],
    rhs: wp.array2d[float],
    impulses: wp.array2d[float],
    prev_impulses: wp.array2d[float],
    row_type: wp.array2d[int],
    row_parent: wp.array2d[int],
    row_mu: wp.array2d[float],
    J_world: wp.array3d[float],
    max_constraints: int,
    max_world_dofs: int,
    # MF constraints
    mf_constraint_count: wp.array[int],
    mf_rhs: wp.array2d[float],
    mf_impulses: wp.array2d[float],
    prev_mf_impulses: wp.array2d[float],
    mf_row_type: wp.array2d[int],
    mf_row_parent: wp.array2d[int],
    mf_row_mu: wp.array2d[float],
    mf_J_a: wp.array3d[float],
    mf_J_b: wp.array3d[float],
    mf_dof_a: wp.array2d[int],
    mf_dof_b: wp.array2d[int],
    mf_max_constraints: int,
    # Propagation articulated body-space constraints
    propagation_constraint_count: wp.array[int],
    propagation_rhs: wp.array2d[float],
    propagation_impulses: wp.array2d[float],
    prev_propagation_impulses: wp.array2d[float],
    propagation_row_type: wp.array2d[int],
    propagation_row_parent: wp.array2d[int],
    propagation_row_mu: wp.array2d[float],
    propagation_J_a: wp.array3d[float],
    propagation_J_b: wp.array3d[float],
    propagation_body_a: wp.array2d[int],
    propagation_body_b: wp.array2d[int],
    propagation_body_qd: wp.array2d[float],
    propagation_max_constraints: int,
    # Velocity
    v_out: wp.array[float],
    # Output: [worlds, 4]
    metrics: wp.array2d[float],
):
    """Compute per-world PGS convergence metrics for velocity-space mode.

    Metrics:
        [0] max|delta_lambda| across all constraint rows
        [1] sum(lambda_n * residual_n) for normal contacts (complementarity gap)
        [2] sum(residual_t^2) for sticking friction contacts (tangent residual energy)
        [3] sum(FB(lambda_n, residual_n)^2) for normal contacts (Fischer-Burmeister)
    """
    world = wp.tid()

    m_dense = constraint_count[world]
    m_mf = mf_constraint_count[world]
    m_propagation = propagation_constraint_count[world]

    max_dl = float(0.0)
    comp_gap = float(0.0)
    tang_res = float(0.0)
    fb_merit = float(0.0)

    # --- Dense constraints ---
    for i in range(m_dense):
        lam = impulses[world, i]
        prev_lam = prev_impulses[world, i]
        dl = wp.abs(lam - prev_lam)
        if dl > max_dl:
            max_dl = dl

        # Compute residual: r_i = J_i * v + bias_i
        jv = float(0.0)
        for d in range(max_world_dofs):
            jv += J_world[world, i, d] * world_response_velocity(v_out, world_dof_indices, world, d)
        residual = jv + rhs[world, i]

        rt = row_type[world, i]
        if rt == PGS_CONSTRAINT_TYPE_CONTACT:
            # Normal: complementarity gap and FB
            comp_gap += lam * residual
            fb_val = wp.sqrt(lam * lam + residual * residual) - lam - residual
            fb_merit += fb_val * fb_val
        elif rt == PGS_CONSTRAINT_TYPE_FRICTION:
            # Friction: tangent residual for sticking contacts
            parent_idx = row_parent[world, i]
            lambda_n = patch_normal_load(row_parent, impulses, world, parent_idx)
            mu = row_mu[world, i]
            radius = mu * lambda_n
            if radius > 0.0:
                # Check if sticking: |lambda_t| < mu * lambda_n
                # Get sibling
                if i == parent_idx + 1:
                    sib = parent_idx + 2
                else:
                    sib = parent_idx + 1
                lam_t1 = impulses[world, i]
                lam_t2 = impulses[world, sib]
                t_mag = wp.sqrt(lam_t1 * lam_t1 + lam_t2 * lam_t2)
                if t_mag < radius * 0.999:  # sticking (with small tolerance)
                    tang_res += residual * residual

    # --- MF constraints ---
    for i in range(m_mf):
        lam = mf_impulses[world, i]
        prev_lam = prev_mf_impulses[world, i]
        dl = wp.abs(lam - prev_lam)
        if dl > max_dl:
            max_dl = dl

        # Compute residual: r = J_a * v_a + J_b * v_b + bias
        dof_a = mf_dof_a[world, i]
        dof_b = mf_dof_b[world, i]
        jv = float(0.0)
        if dof_a >= 0:
            for k in range(6):
                jv += mf_J_a[world, i, k] * world_response_velocity(v_out, world_dof_indices, world, dof_a + k)
        if dof_b >= 0:
            for k in range(6):
                jv += mf_J_b[world, i, k] * world_response_velocity(v_out, world_dof_indices, world, dof_b + k)
        residual = jv + mf_rhs[world, i]

        rt = mf_row_type[world, i]
        if rt == PGS_CONSTRAINT_TYPE_CONTACT:
            comp_gap += lam * residual
            fb_val = wp.sqrt(lam * lam + residual * residual) - lam - residual
            fb_merit += fb_val * fb_val
        elif rt == PGS_CONSTRAINT_TYPE_FRICTION:
            parent_idx = mf_row_parent[world, i]
            lambda_n = patch_normal_load(mf_row_parent, mf_impulses, world, parent_idx)
            mu = mf_row_mu[world, i]
            radius = mu * lambda_n
            if radius > 0.0:
                if i == parent_idx + 1:
                    sib = parent_idx + 2
                else:
                    sib = parent_idx + 1
                lam_t1 = mf_impulses[world, i]
                lam_t2 = mf_impulses[world, sib]
                t_mag = wp.sqrt(lam_t1 * lam_t1 + lam_t2 * lam_t2)
                if t_mag < radius * 0.999:
                    tang_res += residual * residual

    # --- Propagation articulated body-space constraints ---
    for i in range(m_propagation):
        lam = propagation_impulses[world, i]
        prev_lam = prev_propagation_impulses[world, i]
        dl = wp.abs(lam - prev_lam)
        if dl > max_dl:
            max_dl = dl

        ba = propagation_body_a[world, i]
        bb = propagation_body_b[world, i]
        jv = float(0.0)
        if ba >= 0:
            for k in range(6):
                jv += propagation_J_a[world, i, k] * propagation_body_qd[ba, k]
        if bb >= 0:
            for k in range(6):
                jv += propagation_J_b[world, i, k] * propagation_body_qd[bb, k]
        residual = jv + propagation_rhs[world, i]

        rt = propagation_row_type[world, i]
        if rt == PGS_CONSTRAINT_TYPE_CONTACT:
            comp_gap += lam * residual
            fb_val = wp.sqrt(lam * lam + residual * residual) - lam - residual
            fb_merit += fb_val * fb_val
        elif rt == PGS_CONSTRAINT_TYPE_FRICTION:
            parent_idx = propagation_row_parent[world, i]
            lambda_n = patch_normal_load(propagation_row_parent, propagation_impulses, world, parent_idx)
            mu = propagation_row_mu[world, i]
            radius = mu * lambda_n
            if radius > 0.0:
                if i == parent_idx + 1:
                    sib = parent_idx + 2
                else:
                    sib = parent_idx + 1
                if sib < propagation_max_constraints and sib < m_propagation:
                    lam_t1 = propagation_impulses[world, i]
                    lam_t2 = propagation_impulses[world, sib]
                    t_mag = wp.sqrt(lam_t1 * lam_t1 + lam_t2 * lam_t2)
                    if t_mag < radius * 0.999:
                        tang_res += residual * residual

    metrics[world, 0] = max_dl
    metrics[world, 1] = comp_gap
    metrics[world, 2] = tang_res
    metrics[world, 3] = fb_merit


# =============================================================================
# PGS NCP / MDP Residual Diagnostic Kernel (velocity-space mode)
# =============================================================================


@wp.func
def contact_friction_residuals(
    normal_impulse: float,
    friction_load: float,
    mu: float,
    normal_velocity: float,
    tangent_velocity: wp.vec2,
    tangent_impulse: wp.vec2,
):
    """Physical cone/MDP residuals with a separate normal budget for patch anchors."""
    speed = wp.length(tangent_velocity)
    radius = mu * friction_load
    cone = wp.max(wp.length(tangent_impulse) - radius, 0.0)
    # Normal complementarity stays local; tangent dissipation uses patch support.
    complementarity = wp.abs(
        normal_impulse * normal_velocity + radius * speed + wp.dot(tangent_impulse, tangent_velocity)
    )
    dual = wp.max(-normal_velocity, 0.0)
    direction = float(0.0)
    if speed > 1.0e-8 and friction_load > 1.0e-8:
        direction = wp.length(tangent_impulse + radius * tangent_velocity / speed)
        if radius > 1.0e-8:
            direction /= radius
    return wp.vec4(cone, complementarity, dual, direction)


@wp.kernel
def pgs_ncp_residuals_diagnostic_velocity(
    # Dense constraints
    constraint_count: wp.array[int],
    world_dof_indices: wp.array2d[int],
    rhs: wp.array2d[float],
    impulses: wp.array2d[float],
    row_type: wp.array2d[int],
    row_parent: wp.array2d[int],
    row_mu: wp.array2d[float],
    row_phi: wp.array2d[float],
    J_world: wp.array3d[float],
    max_constraints: int,
    max_world_dofs: int,
    # MF constraints
    mf_constraint_count: wp.array[int],
    mf_rhs: wp.array2d[float],
    mf_impulses: wp.array2d[float],
    mf_row_type: wp.array2d[int],
    mf_row_parent: wp.array2d[int],
    mf_row_mu: wp.array2d[float],
    mf_row_phi: wp.array2d[float],
    mf_J_a: wp.array3d[float],
    mf_J_b: wp.array3d[float],
    mf_dof_a: wp.array2d[int],
    mf_dof_b: wp.array2d[int],
    mf_max_constraints: int,
    # Propagation articulated body-space constraints
    propagation_constraint_count: wp.array[int],
    propagation_rhs: wp.array2d[float],
    propagation_impulses: wp.array2d[float],
    propagation_row_type: wp.array2d[int],
    propagation_row_parent: wp.array2d[int],
    propagation_row_mu: wp.array2d[float],
    propagation_row_phi: wp.array2d[float],
    propagation_J_a: wp.array3d[float],
    propagation_J_b: wp.array3d[float],
    propagation_body_a: wp.array2d[int],
    propagation_body_b: wp.array2d[int],
    propagation_body_qd: wp.array2d[float],
    propagation_max_constraints: int,
    # Velocity
    v_out: wp.array[float],
    # Output: [worlds, 6]
    metrics: wp.array2d[float],
):
    """Compute per-world NCP / MDP residuals on the matrix_free PGS path.

    The six residuals per world are reduced with ``max`` across contact
    groups and follow the formulation in ``SolverRaisim.residuals`` (see
    ``artifacts/2026-04-16-slack-raisim/repos/mmacklin-newton-solver-raisim/
    newton/_src/solvers/raisim/residuals.py``):

    ``[0] r_compl``
        ``max_i |min(lambda_n_i, u_n_i + b_n_i)|`` — standard NCP
        complementarity residual.
    ``[1] r_cone``
        ``max_i max(||lambda_t_i|| - mu_i * lambda_n_i, 0)`` — friction
        cone violation (Coulomb).
    ``[2] r_gap``
        ``max_i max(-phi_i, 0)`` — signed-distance penetration.
    ``[3] r_ds_compl``
        ``max_i |<lambda, c + Gamma(c, mu)>|`` — de Saxcé / MDP
        complementarity with Gamma = (0, 0, mu*||c_T||).
    ``[4] r_ds_dual``
        ``max_i max(-u_n_i, 0)`` — dual-cone feasibility for the
        augmented velocity ``c + Gamma``, which simplifies to
        ``max(-u_n, 0)``.
    ``[5] r_mdp_dir``
        ``max_i ||lambda_t - (-mu*lambda_n)*(c_T / ||c_T||)|| / (mu*lambda_n)``
        — MDP direction error for actively sliding contacts
        (``||c_T|| > 1e-8`` and ``lambda_n > 1e-8``).

    For patch friction, tangent cone and MDP terms use the complete region's
    normal impulse and the anchor's share of its friction coefficient. Normal
    complementarity and gap remain per contact.

    Residual computation distinguishes row kinds:
    * Only rows of type ``PGS_CONSTRAINT_TYPE_CONTACT`` drive a contact
      iteration (both in the dense articulated buffer and the
      matrix-free free-rigid buffer).
    * Joint-limit rows (``PGS_CONSTRAINT_TYPE_JOINT_LIMIT``) and
      joint-target rows (``PGS_CONSTRAINT_TYPE_JOINT_TARGET``) are
      **skipped** — friction, cone, and MDP residuals do not apply to
      them. ``r_gap`` is also skipped for these rows (the gap concept is
      specific to contact normals).
    * Friction rows (``PGS_CONSTRAINT_TYPE_FRICTION``) are **not**
      iterated directly — they are read via the parent CONTACT row at
      ``parent_idx + 1`` and ``parent_idx + 2``, which avoids
      double-counting and correctly pairs tangent basis components.
    * ``u_n``, ``u_t1`` and ``u_t2`` are computed as the bias-free
      ``J_row * v_out`` (matching the raisim reference which reads
      joint-velocity directly). ``r_compl`` still uses ``u_n + b_n`` via
      ``rhs[row]`` to match the NCP statement.
    """
    world = wp.tid()

    m_dense = constraint_count[world]
    m_mf = mf_constraint_count[world]
    m_propagation = propagation_constraint_count[world]

    r_compl = float(0.0)
    r_cone = float(0.0)
    r_gap = float(0.0)
    r_ds_compl = float(0.0)
    r_ds_dual = float(0.0)
    r_mdp_dir = float(0.0)

    # ---- Dense constraints (articulated contacts + joint limits) ----
    for i in range(m_dense):
        rt = row_type[world, i]
        if rt != PGS_CONSTRAINT_TYPE_CONTACT:
            # skip friction rows (handled via parent), joint-limit rows,
            # and joint-target rows — they do not contribute NCP/MDP
            # contact residuals.
            continue

        # Normal row velocity (bias-free): u_n = J_n * v
        u_n = float(0.0)
        for d in range(max_world_dofs):
            u_n += J_world[world, i, d] * world_response_velocity(v_out, world_dof_indices, world, d)
        b_n = rhs[world, i]
        ln = impulses[world, i]

        # r_compl: |min(ln, u_n + b_n)|
        ubn = u_n + b_n
        if ln < ubn:
            compl = wp.abs(ln)
        else:
            compl = wp.abs(ubn)
        if compl > r_compl:
            r_compl = compl

        # r_gap: max(-phi, 0)
        neg_phi = -row_phi[world, i]
        if neg_phi > r_gap:
            r_gap = neg_phi

        # Friction rows at i+1, i+2 (if present and parented to i)
        lt1 = float(0.0)
        lt2 = float(0.0)
        u_t1 = float(0.0)
        u_t2 = float(0.0)
        mu = float(0.0)

        i1 = i + 1
        if i1 < max_constraints and i1 < m_dense:
            if row_type[world, i1] == PGS_CONSTRAINT_TYPE_FRICTION and row_parent[world, i1] == i:
                lt1 = impulses[world, i1]
                mu = row_mu[world, i1]
                for d in range(max_world_dofs):
                    u_t1 += J_world[world, i1, d] * world_response_velocity(v_out, world_dof_indices, world, d)

        i2 = i + 2
        if i2 < max_constraints and i2 < m_dense:
            if row_type[world, i2] == PGS_CONSTRAINT_TYPE_FRICTION and row_parent[world, i2] == i:
                lt2 = impulses[world, i2]
                if mu == 0.0:
                    mu = row_mu[world, i2]
                for d in range(max_world_dofs):
                    u_t2 += J_world[world, i2, d] * world_response_velocity(v_out, world_dof_indices, world, d)

        # r_cone
        friction_load = ln
        if mu > 0.0:
            friction_load = patch_normal_load(row_parent, impulses, world, i)
        friction_residual = contact_friction_residuals(
            ln, friction_load, mu, u_n, wp.vec2(u_t1, u_t2), wp.vec2(lt1, lt2)
        )
        r_cone = wp.max(r_cone, friction_residual[0])
        r_ds_compl = wp.max(r_ds_compl, friction_residual[1])
        r_ds_dual = wp.max(r_ds_dual, friction_residual[2])
        r_mdp_dir = wp.max(r_mdp_dir, friction_residual[3])

    # ---- Matrix-free constraints (free-rigid contacts) ----
    for i in range(m_mf):
        rt = mf_row_type[world, i]
        if rt != PGS_CONSTRAINT_TYPE_CONTACT:
            continue

        dof_a = mf_dof_a[world, i]
        dof_b = mf_dof_b[world, i]
        u_n = float(0.0)
        if dof_a >= 0:
            for k in range(6):
                u_n += mf_J_a[world, i, k] * world_response_velocity(v_out, world_dof_indices, world, dof_a + k)
        if dof_b >= 0:
            for k in range(6):
                u_n += mf_J_b[world, i, k] * world_response_velocity(v_out, world_dof_indices, world, dof_b + k)
        b_n = mf_rhs[world, i]
        ln = mf_impulses[world, i]

        ubn = u_n + b_n
        if ln < ubn:
            compl = wp.abs(ln)
        else:
            compl = wp.abs(ubn)
        if compl > r_compl:
            r_compl = compl

        neg_phi = -mf_row_phi[world, i]
        if neg_phi > r_gap:
            r_gap = neg_phi

        lt1 = float(0.0)
        lt2 = float(0.0)
        u_t1 = float(0.0)
        u_t2 = float(0.0)
        mu = float(0.0)

        i1 = i + 1
        if i1 < mf_max_constraints and i1 < m_mf:
            if mf_row_type[world, i1] == PGS_CONSTRAINT_TYPE_FRICTION and mf_row_parent[world, i1] == i:
                lt1 = mf_impulses[world, i1]
                mu = mf_row_mu[world, i1]
                dof_a1 = mf_dof_a[world, i1]
                dof_b1 = mf_dof_b[world, i1]
                if dof_a1 >= 0:
                    for k in range(6):
                        u_t1 += mf_J_a[world, i1, k] * world_response_velocity(
                            v_out, world_dof_indices, world, dof_a1 + k
                        )
                if dof_b1 >= 0:
                    for k in range(6):
                        u_t1 += mf_J_b[world, i1, k] * world_response_velocity(
                            v_out, world_dof_indices, world, dof_b1 + k
                        )

        i2 = i + 2
        if i2 < mf_max_constraints and i2 < m_mf:
            if mf_row_type[world, i2] == PGS_CONSTRAINT_TYPE_FRICTION and mf_row_parent[world, i2] == i:
                lt2 = mf_impulses[world, i2]
                if mu == 0.0:
                    mu = mf_row_mu[world, i2]
                dof_a2 = mf_dof_a[world, i2]
                dof_b2 = mf_dof_b[world, i2]
                if dof_a2 >= 0:
                    for k in range(6):
                        u_t2 += mf_J_a[world, i2, k] * world_response_velocity(
                            v_out, world_dof_indices, world, dof_a2 + k
                        )
                if dof_b2 >= 0:
                    for k in range(6):
                        u_t2 += mf_J_b[world, i2, k] * world_response_velocity(
                            v_out, world_dof_indices, world, dof_b2 + k
                        )

        friction_load = ln
        if mu > 0.0:
            friction_load = patch_normal_load(mf_row_parent, mf_impulses, world, i)
        friction_residual = contact_friction_residuals(
            ln, friction_load, mu, u_n, wp.vec2(u_t1, u_t2), wp.vec2(lt1, lt2)
        )
        r_cone = wp.max(r_cone, friction_residual[0])
        r_ds_compl = wp.max(r_ds_compl, friction_residual[1])
        r_ds_dual = wp.max(r_ds_dual, friction_residual[2])
        r_mdp_dir = wp.max(r_mdp_dir, friction_residual[3])

    # ---- Propagation articulated body-space constraints ----
    for i in range(m_propagation):
        rt = propagation_row_type[world, i]
        if rt != PGS_CONSTRAINT_TYPE_CONTACT:
            continue

        ba = propagation_body_a[world, i]
        bb = propagation_body_b[world, i]
        u_n = float(0.0)
        if ba >= 0:
            for k in range(6):
                u_n += propagation_J_a[world, i, k] * propagation_body_qd[ba, k]
        if bb >= 0:
            for k in range(6):
                u_n += propagation_J_b[world, i, k] * propagation_body_qd[bb, k]
        b_n = propagation_rhs[world, i]
        ln = propagation_impulses[world, i]

        ubn = u_n + b_n
        if ln < ubn:
            compl = wp.abs(ln)
        else:
            compl = wp.abs(ubn)
        if compl > r_compl:
            r_compl = compl

        neg_phi = -propagation_row_phi[world, i]
        if neg_phi > r_gap:
            r_gap = neg_phi

        lt1 = float(0.0)
        lt2 = float(0.0)
        u_t1 = float(0.0)
        u_t2 = float(0.0)
        mu = float(0.0)

        i1 = i + 1
        if i1 < propagation_max_constraints and i1 < m_propagation:
            if (
                propagation_row_type[world, i1] == PGS_CONSTRAINT_TYPE_FRICTION
                and propagation_row_parent[world, i1] == i
            ):
                lt1 = propagation_impulses[world, i1]
                mu = propagation_row_mu[world, i1]
                ba1 = propagation_body_a[world, i1]
                bb1 = propagation_body_b[world, i1]
                if ba1 >= 0:
                    for k in range(6):
                        u_t1 += propagation_J_a[world, i1, k] * propagation_body_qd[ba1, k]
                if bb1 >= 0:
                    for k in range(6):
                        u_t1 += propagation_J_b[world, i1, k] * propagation_body_qd[bb1, k]

        i2 = i + 2
        if i2 < propagation_max_constraints and i2 < m_propagation:
            if (
                propagation_row_type[world, i2] == PGS_CONSTRAINT_TYPE_FRICTION
                and propagation_row_parent[world, i2] == i
            ):
                lt2 = propagation_impulses[world, i2]
                if mu == 0.0:
                    mu = propagation_row_mu[world, i2]
                ba2 = propagation_body_a[world, i2]
                bb2 = propagation_body_b[world, i2]
                if ba2 >= 0:
                    for k in range(6):
                        u_t2 += propagation_J_a[world, i2, k] * propagation_body_qd[ba2, k]
                if bb2 >= 0:
                    for k in range(6):
                        u_t2 += propagation_J_b[world, i2, k] * propagation_body_qd[bb2, k]

        friction_load = ln
        if mu > 0.0:
            friction_load = patch_normal_load(propagation_row_parent, propagation_impulses, world, i)
        friction_residual = contact_friction_residuals(
            ln, friction_load, mu, u_n, wp.vec2(u_t1, u_t2), wp.vec2(lt1, lt2)
        )
        r_cone = wp.max(r_cone, friction_residual[0])
        r_ds_compl = wp.max(r_ds_compl, friction_residual[1])
        r_ds_dual = wp.max(r_ds_dual, friction_residual[2])
        r_mdp_dir = wp.max(r_mdp_dir, friction_residual[3])

    metrics[world, 0] = r_compl
    metrics[world, 1] = r_cone
    metrics[world, 2] = r_gap
    metrics[world, 3] = r_ds_compl
    metrics[world, 4] = r_ds_dual
    metrics[world, 5] = r_mdp_dir


# ---------------------------------------------------------------------------
# Graph coloring for propagation rows (articulated_contact_response=
# "propagation-colored").
#
# Two propagation rows conflict iff they share a dynamic body (their velocity
# writes and deferred-impulse accumulations overlap). Rows of one color are
# body-disjoint, so a color can be solved by one flat thread-per-row launch
# across all worlds with plain (non-atomic) body writes; colors run
# sequentially, giving a Gauss-Seidel ordering permutation of the serial
# per-world sweep. The friction-cone sibling write stays single-writer under
# coloring because a row's siblings share its bodies and therefore can never
# share its color.
#
# Coloring is first-fit greedy edge coloring over the world's contact units taken
# in global-contact-index order (the pre-build kernel sorts them first, so the
# schedule is independent of the atomic list-build order): a unit takes the lowest
# color neither of its bodies uses yet, tracked as one bitmask per body. That uses
# at most 2*degree-1 colors and makes every color a near-maximal matching. Units
# that find no free color below the cap go to a serial tail bucket (color
# PROPAGATION_COLOR_TAIL) processed by a per-world ordered sweep — measured and
# reported, never silent. A kinematic free rigid body has no response, so it is
# recorded as -1 like the world and never counts as a conflict.
#
# Edge coloring needs at least max-degree colors and first-fit uses up to
# 2*degree-1. A dense raw-mesh pile (P12: 60 GraspNet meshes, ~7000 contact
# units on 61 bodies) puts ~230 units on a body, so 256 colors left a quarter of
# the units in the serial tail; 512 leaves under 2%. Small scenes are unaffected:
# the solve kernels stop at the last non-empty color.

PROPAGATION_MAX_COLORS = 512
PROPAGATION_COLOR_TAIL = 512


PROPAGATION_UNIT_RING = 4
"""Patch-ring members carried in a unit record; longer rings resume the walk after them."""

PROPAGATION_UNIT_META = 5 + PROPAGATION_UNIT_RING
"""Per-unit static record for the colored solve (see :func:`gather_propagation_unit_meta`)."""


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
    # out
    unit_meta: wp.array[int],
):
    """Pack what the colored solve needs before it can solve a unit.

    Indexed by the unit's position in color order: ``[start slot, local slot of body a,
    local slot of body b, patch-ring length, the first PROPAGATION_UNIT_RING ring
    members, the member after them]``. A contact row's ``row_parent`` links the other
    contact rows sharing its friction patch into a ring. All of it is fixed for a
    sweep, so the solve loads the next colors' records while the current color runs
    and loads the ring members' impulses in parallel instead of walking the ring one
    dependent load at a time.
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


@wp.kernel(enable_backward=False)
def collect_propagation_units(
    contact_count: wp.array[int],
    contact_path: wp.array[int],
    contact_world: wp.array[int],
    contact_shape0: wp.array[int],
    contact_shape1: wp.array[int],
    shape_body: wp.array[int],
    body_prescribed: wp.array[int],
    contact_slots_needed: wp.array[int],
    propagation_max_constraints: int,
    # in/out
    world_unit_cursor: wp.array[int],
    # out
    unit_contact: wp.array[int],
    unit_body_a: wp.array[int],
    unit_body_b: wp.array[int],
    unit_len: wp.array[int],
):
    """Gather propagation-path contacts into per-world unit lists for pre-build coloring.

    A prescribed body (``body_prescribed`` marks kinematic free rigid bodies, whose
    response is identically zero) never receives a velocity update from a row, so rows
    touching it do not conflict: it is recorded as ``-1`` like the world. Otherwise a
    kinematic hub (a tray, a conveyor) forces every one of its contacts into a separate
    color or the serial tail.
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
    sa = contact_shape0[c]
    sb = contact_shape1[c]
    if sa >= 0:
        body_a = shape_body[sa]
        if body_a >= 0 and body_prescribed[body_a] != 0:
            body_a = -1
    if sb >= 0:
        body_b = shape_body[sb]
        if body_b >= 0 and body_prescribed[body_b] != 0:
            body_b = -1
    unit_contact[base + idx] = c
    unit_body_a[base + idx] = body_a
    unit_body_b[base + idx] = body_b
    unit_len[base + idx] = contact_slots_needed[c]
