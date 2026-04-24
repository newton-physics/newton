# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Warp kernels for SolverMuJoCo."""

from __future__ import annotations

from typing import Any

import warp as wp

from ...core.types import vec5
from ...sim import BodyFlags, EqType, JointTargetMode, JointType
from ...sim.articulation import com_twist_to_point_velocity, origin_twist_to_com_twist


def _import_contact_force_fn():
    from mujoco_warp._src.support import contact_force_fn

    return contact_force_fn


# Custom vector types
vec10 = wp.types.vector(length=10, dtype=wp.float32)
vec11 = wp.types.vector(length=11, dtype=wp.float32)


# Constants
MJ_MINVAL = 2.220446049250313e-16
MJ_MINMU = 1e-5


# Utility functions
@wp.func
def safe_div(x: float, y: float) -> float:
    return x / wp.where(y != 0.0, y, MJ_MINVAL)


@wp.func
def orthogonals(a: wp.vec3):
    y = wp.vec3(0.0, 1.0, 0.0)
    z = wp.vec3(0.0, 0.0, 1.0)
    b = wp.where((-0.5 < a[1]) and (a[1] < 0.5), y, z)
    b = b - a * wp.dot(a, b)
    b = wp.normalize(b)
    if wp.length(a) == 0.0:
        b = wp.vec3(0.0, 0.0, 0.0)
    c = wp.cross(a, b)

    return b, c


@wp.func
def make_frame(a: wp.vec3):
    a = wp.normalize(a)
    b, c = orthogonals(a)

    # fmt: off
    return wp.mat33(
    a.x, a.y, a.z,
    b.x, b.y, b.z,
    c.x, c.y, c.z
  )
    # fmt: on


@wp.func
def write_contact(
    # Data in:
    # In:
    dist_in: float,
    pos_in: wp.vec3,
    frame_in: wp.mat33,
    margin_in: float,
    gap_in: float,
    condim_in: int,
    friction_in: vec5,
    solref_in: wp.vec2f,
    solreffriction_in: wp.vec2f,
    solimp_in: vec5,
    geoms_in: wp.vec2i,
    worldid_in: int,
    contact_id_in: int,
    # Data out:
    contact_dist_out: wp.array[float],
    contact_pos_out: wp.array[wp.vec3],
    contact_frame_out: wp.array[wp.mat33],
    contact_includemargin_out: wp.array[float],
    contact_friction_out: wp.array[vec5],
    contact_solref_out: wp.array[wp.vec2],
    contact_solreffriction_out: wp.array[wp.vec2],
    contact_solimp_out: wp.array[vec5],
    contact_dim_out: wp.array[int],
    contact_geom_out: wp.array[wp.vec2i],
    contact_efc_address_out: wp.array2d[int],
    contact_worldid_out: wp.array[int],
):
    # See function write_contact in mujoco_warp, file collision_primitive.py

    cid = contact_id_in
    contact_dist_out[cid] = dist_in
    contact_pos_out[cid] = pos_in
    contact_frame_out[cid] = frame_in
    contact_geom_out[cid] = geoms_in
    contact_worldid_out[cid] = worldid_in
    contact_includemargin_out[cid] = margin_in - gap_in
    contact_dim_out[cid] = condim_in
    contact_friction_out[cid] = friction_in
    contact_solref_out[cid] = solref_in
    contact_solreffriction_out[cid] = solreffriction_in
    contact_solimp_out[cid] = solimp_in

    # initialize constraint address to -1 (max 10 elements; populated during constraint generation)
    for i in range(contact_efc_address_out.shape[1]):
        contact_efc_address_out[cid, i] = -1


@wp.func
def contact_params(
    geom_condim: wp.array[int],
    geom_priority: wp.array[int],
    geom_solmix: wp.array2d[float],
    geom_solref: wp.array2d[wp.vec2],
    geom_solimp: wp.array2d[vec5],
    geom_friction: wp.array2d[wp.vec3],
    geom_margin: wp.array2d[float],
    geom_gap: wp.array2d[float],
    geoms: wp.vec2i,
    worldid: int,
):
    # See function contact_params in mujoco_warp, file collision_core.py

    g1 = geoms[0]
    g2 = geoms[1]

    p1 = geom_priority[g1]
    p2 = geom_priority[g2]

    condim1 = geom_condim[g1]
    condim2 = geom_condim[g2]

    if p1 > p2:
        mix = 1.0
        condim = condim1
        resolved_friction = geom_friction[worldid, g1]
    elif p2 > p1:
        mix = 0.0
        condim = condim2
        resolved_friction = geom_friction[worldid, g2]
    else:
        solmix1 = geom_solmix[worldid, g1]
        solmix2 = geom_solmix[worldid, g2]
        mix = safe_div(solmix1, solmix1 + solmix2)
        mix = wp.where((solmix1 < MJ_MINVAL) and (solmix2 < MJ_MINVAL), 0.5, mix)
        mix = wp.where((solmix1 < MJ_MINVAL) and (solmix2 >= MJ_MINVAL), 0.0, mix)
        mix = wp.where((solmix1 >= MJ_MINVAL) and (solmix2 < MJ_MINVAL), 1.0, mix)
        condim = wp.max(condim1, condim2)
        resolved_friction = wp.max(geom_friction[worldid, g1], geom_friction[worldid, g2])

    friction = vec5(
        wp.max(MJ_MINMU, resolved_friction[0]),
        wp.max(MJ_MINMU, resolved_friction[0]),
        wp.max(MJ_MINMU, resolved_friction[1]),
        wp.max(MJ_MINMU, resolved_friction[2]),
        wp.max(MJ_MINMU, resolved_friction[2]),
    )

    # Sum margins for consistency with thickness summing
    margin = geom_margin[worldid, g1] + geom_margin[worldid, g2]
    gap = geom_gap[worldid, g1] + geom_gap[worldid, g2]

    if geom_solref[worldid, g1].x > 0.0 and geom_solref[worldid, g2].x > 0.0:
        solref = mix * geom_solref[worldid, g1] + (1.0 - mix) * geom_solref[worldid, g2]
    else:
        solref = wp.min(geom_solref[worldid, g1], geom_solref[worldid, g2])

    solreffriction = wp.vec2(0.0, 0.0)

    solimp = mix * geom_solimp[worldid, g1] + (1.0 - mix) * geom_solimp[worldid, g2]

    return margin, gap, condim, friction, solref, solreffriction, solimp


@wp.func
def convert_solref(ke: float, kd: float, d_width: float, d_r: float) -> wp.vec2:
    """Convert from stiffness and damping to time constant and damp ratio
    based on d(r) and d(width)."""

    if ke > 0.0 and kd > 0.0:
        # ke = d(r) / (d_width^2 * timeconst^2 * dampratio^2)
        # kd = 2 / (d_width * timeconst)
        timeconst = 2.0 / (kd * d_width)
        dampratio = kd / 2.0 * wp.sqrt(d_r / ke)
    else:
        timeconst = 0.02
        dampratio = 1.0
    # see https://mujoco.readthedocs.io/en/latest/modeling.html#solver-parameters

    return wp.vec2(timeconst, dampratio)


@wp.func
def quat_wxyz_to_xyzw(q: wp.quat) -> wp.quat:
    """Convert a quaternion from MuJoCo wxyz storage to Warp xyzw format."""
    return wp.quat(q[1], q[2], q[3], q[0])


@wp.func
def quat_xyzw_to_wxyz(q: wp.quat) -> wp.quat:
    """Convert a Warp xyzw quaternion to MuJoCo wxyz storage order.

    The returned wp.quat is NOT valid for Warp math — it is a container
    for writing components to MuJoCo arrays.
    """
    return wp.quat(q[3], q[0], q[1], q[2])


# Kernel functions
@wp.kernel
def convert_newton_contacts_to_mjwarp_kernel(
    body_q: wp.array[wp.transform],
    shape_body: wp.array[int],
    body_flags: wp.array[int],
    # Model:
    geom_bodyid: wp.array[int],
    body_weldid: wp.array[int],
    geom_condim: wp.array[int],
    geom_priority: wp.array[int],
    geom_solmix: wp.array2d[float],
    geom_solref: wp.array2d[wp.vec2],
    geom_solimp: wp.array2d[vec5],
    geom_friction: wp.array2d[wp.vec3],
    geom_margin: wp.array2d[float],
    geom_gap: wp.array2d[float],
    # Newton contacts
    rigid_contact_count: wp.array[wp.int32],
    rigid_contact_shape0: wp.array[wp.int32],
    rigid_contact_shape1: wp.array[wp.int32],
    rigid_contact_point0: wp.array[wp.vec3],
    rigid_contact_point1: wp.array[wp.vec3],
    rigid_contact_normal: wp.array[wp.vec3],
    rigid_contact_margin0: wp.array[wp.float32],
    rigid_contact_margin1: wp.array[wp.float32],
    rigid_contact_stiffness: wp.array[wp.float32],
    rigid_contact_damping: wp.array[wp.float32],
    rigid_contact_friction: wp.array[wp.float32],
    shape_margin: wp.array[float],
    bodies_per_world: int,
    newton_shape_to_mjc_geom: wp.array[wp.int32],
    # Mujoco warp contacts
    naconmax: int,
    nacon_out: wp.array[int],
    contact_dist_out: wp.array[float],
    contact_pos_out: wp.array[wp.vec3],
    contact_frame_out: wp.array[wp.mat33],
    contact_includemargin_out: wp.array[float],
    contact_friction_out: wp.array[vec5],
    contact_solref_out: wp.array[wp.vec2],
    contact_solreffriction_out: wp.array[wp.vec2],
    contact_solimp_out: wp.array[vec5],
    contact_dim_out: wp.array[int],
    contact_geom_out: wp.array[wp.vec2i],
    contact_efc_address_out: wp.array2d[int],
    contact_worldid_out: wp.array[int],
    # Values to clear - see _zero_collision_arrays kernel from mujoco_warp
    nworld_in: int,
    ncollision_out: wp.array[int],
    # Fast-path generation tracking
    contact_generation: wp.array[wp.int32],
    last_contact_generation: wp.array[wp.int32],
    tid_to_cid: wp.array[wp.int32],
    last_nacon_count: wp.array[wp.int32],
):
    # nacon_out must be zeroed before this kernel is launched so that
    # wp.atomic_add below produces the correct compacted count.
    #
    # When the contact set hasn't changed since the last full pass
    # (contact_generation == last_contact_generation), the kernel takes a
    # fast path that only recomputes the body-q-dependent fields (dist, pos)
    # and resets efc_address.  All other MJWarp contact fields (frame,
    # friction, solref, solimp, condim, geom, worldid, includemargin) are
    # still valid from the previous full pass.

    tid = wp.tid()

    count = rigid_contact_count[0]

    gen = contact_generation[0]
    last_gen = last_contact_generation[0]
    needs_full = gen != last_gen

    if needs_full:
        # ── FULL PATH ────────────────────────────────────────────────────
        # Runs on the first substep after collision detection.  Identical to
        # the original kernel plus recording the tid→cid mapping.

        if tid == 0:
            if count > naconmax:
                wp.printf(
                    "Number of Newton contacts (%d) exceeded MJWarp limit (%d). Increase nconmax.\n",
                    count,
                    naconmax,
                )
            ncollision_out[0] = 0

        if count > naconmax:
            count = naconmax

        if tid >= count:
            tid_to_cid[tid] = -1
            return

        shape_a = rigid_contact_shape0[tid]
        shape_b = rigid_contact_shape1[tid]

        if shape_a < 0 or shape_b < 0:
            tid_to_cid[tid] = -1
            return

        geom_a = newton_shape_to_mjc_geom[shape_a]
        geom_b = newton_shape_to_mjc_geom[shape_b]

        body_a = shape_body[shape_a]
        body_b = shape_body[shape_b]

        mj_body_a = geom_bodyid[geom_a]
        mj_body_b = geom_bodyid[geom_b]

        # A body is "immovable" in three cases:
        #  1. body < 0 → static shape (no body)
        #  2. BodyFlags.KINEMATIC → kinematic body (e.g. armature=1e10)
        #  3. body_weldid == 0 → fixed root body (worldbody)
        # Pairs where both sides are immovable produce degenerate efc_D values
        # in MuJoCo's solver, so we skip them.
        a_immovable = body_a < 0 or (body_flags[body_a] & BodyFlags.KINEMATIC) != 0 or body_weldid[mj_body_a] == 0
        b_immovable = body_b < 0 or (body_flags[body_b] & BodyFlags.KINEMATIC) != 0 or body_weldid[mj_body_b] == 0

        if a_immovable and b_immovable:
            tid_to_cid[tid] = -1
            return

        X_wb_a = wp.transform_identity()
        X_wb_b = wp.transform_identity()
        if body_a >= 0:
            X_wb_a = body_q[body_a]
        if body_b >= 0:
            X_wb_b = body_q[body_b]

        bx_a = wp.transform_point(X_wb_a, rigid_contact_point0[tid])
        bx_b = wp.transform_point(X_wb_b, rigid_contact_point1[tid])

        # rigid_contact_margin0/1 = radius_eff + shape_margin per shape.
        # Subtract shape_margin so dist is the surface-to-surface distance;
        # shape_margin is handled by geom_margin (MuJoCo's includemargin).
        radius_eff = (rigid_contact_margin0[tid] - shape_margin[shape_a]) + (
            rigid_contact_margin1[tid] - shape_margin[shape_b]
        )

        n = rigid_contact_normal[tid]
        dist = wp.dot(n, bx_b - bx_a) - radius_eff
        pos = 0.5 * (bx_a + bx_b)

        frame = make_frame(n)

        geoms = wp.vec2i(geom_a, geom_b)

        worldid = body_a // bodies_per_world
        if body_a < 0:
            worldid = body_b // bodies_per_world

        margin, gap, condim, friction, solref, solreffriction, solimp = contact_params(
            geom_condim,
            geom_priority,
            geom_solmix,
            geom_solref,
            geom_solimp,
            geom_friction,
            geom_margin,
            geom_gap,
            geoms,
            worldid,
        )

        # Convert Newton per-contact stiffness/damping to MuJoCo solref
        # (timeconst, dampratio).  solimp is set to approximate a linear
        # force-displacement relationship at rest, compensating for impedance
        # scaling.  See https://mujoco.readthedocs.io/en/latest/modeling.html#solver-parameters
        if rigid_contact_stiffness:
            contact_ke = rigid_contact_stiffness[tid]
            if contact_ke > 0.0:
                imp = solimp[1]
                solimp = vec5(imp, imp, 0.001, 1.0, 0.5)
                contact_ke = contact_ke * (1.0 - imp)
                kd = rigid_contact_damping[tid]
                if kd > 0.0:
                    timeconst = 2.0 / kd
                    dampratio = wp.sqrt(1.0 / (timeconst * timeconst * contact_ke))
                else:
                    timeconst = wp.sqrt(1.0 / contact_ke)
                    dampratio = 1.0
                solref = wp.vec2(timeconst, dampratio)

            friction_scale = rigid_contact_friction[tid]
            if friction_scale > 0.0:
                friction = vec5(
                    friction[0] * friction_scale,
                    friction[1] * friction_scale,
                    friction[2],
                    friction[3],
                    friction[4],
                )

        cid = wp.atomic_add(nacon_out, 0, 1)
        if cid >= naconmax:
            tid_to_cid[tid] = -1
            return

        tid_to_cid[tid] = cid

        write_contact(
            dist_in=dist,
            pos_in=pos,
            frame_in=frame,
            margin_in=margin,
            gap_in=gap,
            condim_in=condim,
            friction_in=friction,
            solref_in=solref,
            solreffriction_in=solreffriction,
            solimp_in=solimp,
            geoms_in=geoms,
            worldid_in=worldid,
            contact_id_in=cid,
            contact_dist_out=contact_dist_out,
            contact_pos_out=contact_pos_out,
            contact_frame_out=contact_frame_out,
            contact_includemargin_out=contact_includemargin_out,
            contact_friction_out=contact_friction_out,
            contact_solref_out=contact_solref_out,
            contact_solreffriction_out=contact_solreffriction_out,
            contact_solimp_out=contact_solimp_out,
            contact_dim_out=contact_dim_out,
            contact_geom_out=contact_geom_out,
            contact_efc_address_out=contact_efc_address_out,
            contact_worldid_out=contact_worldid_out,
        )
    else:
        # ── FAST PATH ────────────────────────────────────────────────────
        # Subsequent substeps with the same contact set.  Only dist, pos,
        # and efc_address need updating; all other MJWarp fields are still
        # valid from the full pass.
        #
        # NOTE: rigid_contact_normal is computed once by the narrow phase
        # and is invariant across substeps.  The fast path is only correct
        # when collide() has not been called since the last full pass.

        if tid == 0:
            ncollision_out[0] = 0
            # Restore the compacted contact count from the full pass
            nacon_out[0] = last_nacon_count[0]

        cid = tid_to_cid[tid]
        if cid < 0:
            return

        shape_a = rigid_contact_shape0[tid]
        shape_b = rigid_contact_shape1[tid]
        body_a = shape_body[shape_a]
        body_b = shape_body[shape_b]

        X_wb_a = wp.transform_identity()
        X_wb_b = wp.transform_identity()
        if body_a >= 0:
            X_wb_a = body_q[body_a]
        if body_b >= 0:
            X_wb_b = body_q[body_b]

        bx_a = wp.transform_point(X_wb_a, rigid_contact_point0[tid])
        bx_b = wp.transform_point(X_wb_b, rigid_contact_point1[tid])

        radius_eff = (rigid_contact_margin0[tid] - shape_margin[shape_a]) + (
            rigid_contact_margin1[tid] - shape_margin[shape_b]
        )

        n = rigid_contact_normal[tid]
        contact_dist_out[cid] = wp.dot(n, bx_b - bx_a) - radius_eff
        contact_pos_out[cid] = 0.5 * (bx_a + bx_b)

        for i in range(contact_efc_address_out.shape[1]):
            contact_efc_address_out[cid, i] = -1


@wp.kernel(enable_backward=False)
def _snapshot_nacon_count(
    nacon: wp.array[wp.int32],
    last_nacon_count: wp.array[wp.int32],
    contact_generation: wp.array[wp.int32],
    last_contact_generation: wp.array[wp.int32],
):
    last_nacon_count[0] = nacon[0]
    last_contact_generation[0] = contact_generation[0]


@wp.kernel
def convert_mj_coords_to_warp_kernel(
    qpos: wp.array2d[wp.float32],
    qvel: wp.array2d[wp.float32],
    joints_per_world: int,
    joint_type: wp.array[wp.int32],
    joint_q_start: wp.array[wp.int32],
    joint_qd_start: wp.array[wp.int32],
    joint_dof_dim: wp.array2d[wp.int32],
    joint_child: wp.array[wp.int32],
    body_com: wp.array[wp.vec3],
    dof_ref: wp.array[wp.float32],
    body_flags: wp.array[wp.int32],
    joint_q_in: wp.array[wp.float32],
    joint_qd_in: wp.array[wp.float32],
    mj_q_start: wp.array[wp.int32],
    mj_qd_start: wp.array[wp.int32],
    # outputs
    joint_q: wp.array[wp.float32],
    joint_qd: wp.array[wp.float32],
):
    worldid, jntid = wp.tid()

    joint_id = joints_per_world * worldid + jntid

    # Skip loop joints — they have no MuJoCo qpos/qvel entries
    q_i = mj_q_start[jntid]
    if q_i < 0:
        return

    qd_i = mj_qd_start[jntid]
    type = joint_type[joint_id]
    wq_i = joint_q_start[joint_id]
    wqd_i = joint_qd_start[joint_id]
    child = joint_child[joint_id]

    if (body_flags[child] & BodyFlags.KINEMATIC) != 0:
        # Previous joint states pass through for kinematic bodies
        wq_end = joint_q_start[joint_id + 1]
        for i in range(wq_i, wq_end):
            joint_q[i] = joint_q_in[i]

        wqd_end = joint_qd_start[joint_id + 1]
        for i in range(wqd_i, wqd_end):
            joint_qd[i] = joint_qd_in[i]
        return

    if type == JointType.FREE:
        # convert position components
        for i in range(3):
            joint_q[wq_i + i] = qpos[worldid, q_i + i]

        # change quaternion order from wxyz to xyzw
        rot = quat_wxyz_to_xyzw(
            wp.quat(
                qpos[worldid, q_i + 3],
                qpos[worldid, q_i + 4],
                qpos[worldid, q_i + 5],
                qpos[worldid, q_i + 6],
            )
        )
        joint_q[wq_i + 3] = rot[0]
        joint_q[wq_i + 4] = rot[1]
        joint_q[wq_i + 5] = rot[2]
        joint_q[wq_i + 6] = rot[3]

        # MuJoCo qvel: linear velocity of body ORIGIN (world frame), angular velocity (body frame)
        # Newton's MuJoCo FREE-root bridge uses a CoM/world twist. More generally,
        # descendant FREE/DISTANCE joint_qd remains expressed in the joint parent frame.
        #
        # Relationship: v_com = v_origin + ω x com_offset_world
        # where com_offset_world = quat_rotate(body_rotation, body_com)

        # Get angular velocity in body frame from MuJoCo and convert to world frame
        w_body = wp.vec3(qvel[worldid, qd_i + 3], qvel[worldid, qd_i + 4], qvel[worldid, qd_i + 5])
        w_world = wp.quat_rotate(rot, w_body)

        # Get CoM offset in world frame
        com_local = body_com[child]
        com_world = wp.quat_rotate(rot, com_local)

        # Get body origin velocity from MuJoCo
        v_origin = wp.vec3(qvel[worldid, qd_i + 0], qvel[worldid, qd_i + 1], qvel[worldid, qd_i + 2])

        # Convert to CoM velocity for Newton: v_com = v_origin + ω x com_offset
        v_com = v_origin + wp.cross(w_world, com_world)
        joint_qd[wqd_i + 0] = v_com[0]
        joint_qd[wqd_i + 1] = v_com[1]
        joint_qd[wqd_i + 2] = v_com[2]

        # Angular velocity: convert from body frame (MuJoCo) to world frame (Newton)
        joint_qd[wqd_i + 3] = w_world[0]
        joint_qd[wqd_i + 4] = w_world[1]
        joint_qd[wqd_i + 5] = w_world[2]
    elif type == JointType.BALL:
        # change quaternion order from wxyz to xyzw
        rot = quat_wxyz_to_xyzw(
            wp.quat(
                qpos[worldid, q_i],
                qpos[worldid, q_i + 1],
                qpos[worldid, q_i + 2],
                qpos[worldid, q_i + 3],
            )
        )
        joint_q[wq_i] = rot[0]
        joint_q[wq_i + 1] = rot[1]
        joint_q[wq_i + 2] = rot[2]
        joint_q[wq_i + 3] = rot[3]
        for i in range(3):
            # convert velocity components
            joint_qd[wqd_i + i] = qvel[worldid, qd_i + i]
    else:
        axis_count = joint_dof_dim[joint_id, 0] + joint_dof_dim[joint_id, 1]
        for i in range(axis_count):
            ref = float(0.0)
            if dof_ref:
                ref = dof_ref[wqd_i + i]
            joint_q[wq_i + i] = qpos[worldid, q_i + i] - ref
        for i in range(axis_count):
            # convert velocity components
            joint_qd[wqd_i + i] = qvel[worldid, qd_i + i]


@wp.kernel
def convert_warp_coords_to_mj_kernel(
    joint_q: wp.array[wp.float32],
    joint_qd: wp.array[wp.float32],
    joints_per_world: int,
    joint_type: wp.array[wp.int32],
    joint_q_start: wp.array[wp.int32],
    joint_qd_start: wp.array[wp.int32],
    joint_dof_dim: wp.array2d[wp.int32],
    joint_child: wp.array[wp.int32],
    body_com: wp.array[wp.vec3],
    dof_ref: wp.array[wp.float32],
    mj_q_start: wp.array[wp.int32],
    mj_qd_start: wp.array[wp.int32],
    # outputs
    qpos: wp.array2d[wp.float32],
    qvel: wp.array2d[wp.float32],
):
    worldid, jntid = wp.tid()

    joint_id = joints_per_world * worldid + jntid

    # Skip loop joints — they have no MuJoCo qpos/qvel entries
    q_i = mj_q_start[jntid]
    if q_i < 0:
        return

    qd_i = mj_qd_start[jntid]
    jtype = joint_type[joint_id]
    wq_i = joint_q_start[joint_id]
    wqd_i = joint_qd_start[joint_id]

    if jtype == JointType.FREE:
        # convert position components
        for i in range(3):
            qpos[worldid, q_i + i] = joint_q[wq_i + i]

        rot = wp.quat(
            joint_q[wq_i + 3],
            joint_q[wq_i + 4],
            joint_q[wq_i + 5],
            joint_q[wq_i + 6],
        )
        # change quaternion order from xyzw to wxyz
        rot_wxyz = quat_xyzw_to_wxyz(rot)
        qpos[worldid, q_i + 3] = rot_wxyz[0]
        qpos[worldid, q_i + 4] = rot_wxyz[1]
        qpos[worldid, q_i + 5] = rot_wxyz[2]
        qpos[worldid, q_i + 6] = rot_wxyz[3]

        # Newton's MuJoCo FREE-root bridge uses a CoM/world twist. More generally,
        # descendant FREE/DISTANCE joint_qd remains expressed in the joint parent frame.
        # MuJoCo qvel: linear velocity of body ORIGIN (world frame), angular velocity (body frame)
        #
        # Relationship: v_origin = v_com - ω x com_offset_world
        # where com_offset_world = quat_rotate(body_rotation, body_com)

        # Get angular velocity in world frame
        w_world = wp.vec3(joint_qd[wqd_i + 3], joint_qd[wqd_i + 4], joint_qd[wqd_i + 5])

        # Get CoM offset in world frame
        child = joint_child[joint_id]
        com_local = body_com[child]
        com_world = wp.quat_rotate(rot, com_local)

        # Get CoM velocity from Newton
        v_com = wp.vec3(joint_qd[wqd_i + 0], joint_qd[wqd_i + 1], joint_qd[wqd_i + 2])

        # Convert to body origin velocity for MuJoCo: v_origin = v_com - ω x com_offset
        v_origin = v_com - wp.cross(w_world, com_world)
        qvel[worldid, qd_i + 0] = v_origin[0]
        qvel[worldid, qd_i + 1] = v_origin[1]
        qvel[worldid, qd_i + 2] = v_origin[2]

        # Angular velocity: convert from world frame (Newton) to body frame (MuJoCo)
        w_body = wp.quat_rotate_inv(rot, w_world)
        qvel[worldid, qd_i + 3] = w_body[0]
        qvel[worldid, qd_i + 4] = w_body[1]
        qvel[worldid, qd_i + 5] = w_body[2]

    elif jtype == JointType.BALL:
        # change quaternion order from xyzw to wxyz
        ball_q = wp.quat(joint_q[wq_i], joint_q[wq_i + 1], joint_q[wq_i + 2], joint_q[wq_i + 3])
        ball_q_wxyz = quat_xyzw_to_wxyz(ball_q)
        qpos[worldid, q_i + 0] = ball_q_wxyz[0]
        qpos[worldid, q_i + 1] = ball_q_wxyz[1]
        qpos[worldid, q_i + 2] = ball_q_wxyz[2]
        qpos[worldid, q_i + 3] = ball_q_wxyz[3]
        for i in range(3):
            # convert velocity components
            qvel[worldid, qd_i + i] = joint_qd[wqd_i + i]
    else:
        axis_count = joint_dof_dim[joint_id, 0] + joint_dof_dim[joint_id, 1]
        for i in range(axis_count):
            ref = float(0.0)
            if dof_ref:
                ref = dof_ref[wqd_i + i]
            qpos[worldid, q_i + i] = joint_q[wq_i + i] + ref
        for i in range(axis_count):
            # convert velocity components
            qvel[worldid, qd_i + i] = joint_qd[wqd_i + i]


@wp.kernel
def sync_qpos0_kernel(
    joints_per_world: int,
    bodies_per_world: int,
    joint_type: wp.array[wp.int32],
    joint_q_start: wp.array[wp.int32],
    joint_qd_start: wp.array[wp.int32],
    joint_dof_dim: wp.array2d[wp.int32],
    joint_child: wp.array[wp.int32],
    body_q: wp.array[wp.transform],
    dof_ref: wp.array[wp.float32],
    dof_springref: wp.array[wp.float32],
    mj_q_start: wp.array[wp.int32],
    # outputs
    qpos0: wp.array2d[wp.float32],
    qpos_spring: wp.array2d[wp.float32],
):
    """Sync MuJoCo qpos0 and qpos_spring from Newton model data.

    For hinge/slide: qpos0 = ref, qpos_spring = springref.
    For free: qpos0 from body_q (pos + quat in wxyz order).
    For ball: qpos0 = [1, 0, 0, 0] (identity quaternion in wxyz).
    """
    worldid, jntid = wp.tid()

    # Skip loop joints — they have no MuJoCo qpos entries
    q_i = mj_q_start[jntid]
    if q_i < 0:
        return

    type = joint_type[jntid]
    wqd_i = joint_qd_start[joints_per_world * worldid + jntid]

    if type == JointType.FREE:
        child = joint_child[jntid]
        world_body = worldid * bodies_per_world + child
        bq = body_q[world_body]
        pos = wp.transform_get_translation(bq)
        rot = wp.transform_get_rotation(bq)

        # Position
        for i in range(3):
            qpos0[worldid, q_i + i] = pos[i]
            qpos_spring[worldid, q_i + i] = pos[i]

        # Quaternion: Newton stores xyzw, MuJoCo uses wxyz
        rot_wxyz = quat_xyzw_to_wxyz(rot)
        for i in range(4):
            qpos0[worldid, q_i + 3 + i] = rot_wxyz[i]
            qpos_spring[worldid, q_i + 3 + i] = rot_wxyz[i]
    elif type == JointType.BALL:
        # Identity quaternion in wxyz order
        qpos0[worldid, q_i + 0] = 1.0
        qpos0[worldid, q_i + 1] = 0.0
        qpos0[worldid, q_i + 2] = 0.0
        qpos0[worldid, q_i + 3] = 0.0
        qpos_spring[worldid, q_i + 0] = 1.0
        qpos_spring[worldid, q_i + 1] = 0.0
        qpos_spring[worldid, q_i + 2] = 0.0
        qpos_spring[worldid, q_i + 3] = 0.0
    else:
        axis_count = joint_dof_dim[jntid, 0] + joint_dof_dim[jntid, 1]
        for i in range(axis_count):
            ref = float(0.0)
            springref = float(0.0)
            if dof_ref:
                ref = dof_ref[wqd_i + i]
            if dof_springref:
                springref = dof_springref[wqd_i + i]
            qpos0[worldid, q_i + i] = ref
            qpos_spring[worldid, q_i + i] = springref


@wp.kernel
def build_ref_q_kernel(
    joint_type: wp.array[wp.int32],
    joint_q_start: wp.array[wp.int32],
    joint_qd_start: wp.array[wp.int32],
    joint_dof_dim: wp.array2d[wp.int32],
    joint_child: wp.array[wp.int32],
    body_q: wp.array[wp.transform],
    dof_ref: wp.array[wp.float32],
    # output
    ref_q: wp.array[wp.float32],
):
    """Build reference joint coordinates from joint types and ``dof_ref``.

    Iterates over joints ``[j]``. Produces joint coordinates in Newton
    convention (xyzw quaternions) suitable for ``eval_articulation_fk``.
    Per joint type:

    - **FREE / DISTANCE**: position and quaternion [xyzw] from ``body_q``
      of the child body.
    - **BALL**: identity quaternion [xyzw].
    - **PRISMATIC / REVOLUTE / D6**: copies ``dof_ref`` values [m or rad]
      (or zero when ``dof_ref`` is ``None``).
    - **FIXED** and others: no DOFs, no writes.

    Args:
        joint_type: Joint type enum per joint, shape ``[joint_count]``.
        joint_q_start: Start index into ``ref_q`` for each joint,
            shape ``[joint_count]``.
        joint_qd_start: Start index into ``dof_ref`` for each joint,
            shape ``[joint_count]``.
        joint_dof_dim: Positional and rotational DOF counts per joint,
            shape ``[joint_count, 2]``.
        joint_child: Child body index per joint, shape ``[joint_count]``.
        body_q: Body transforms [m], shape ``[body_count]``,
            dtype ``wp.transform``.
        dof_ref: Reference DOF values [m or rad], shape ``[joint_dof_count]``.
            May be ``None``, in which case zeros are used.
        ref_q: *(output)* Reference joint coordinates [m or rad],
            shape ``[joint_coord_count]``.
    """
    j = wp.tid()
    jtype = joint_type[j]
    q_start = joint_q_start[j]
    qd_start = joint_qd_start[j]

    if jtype == JointType.FREE or jtype == JointType.DISTANCE:
        child = joint_child[j]
        bq = body_q[child]
        pos = wp.transform_get_translation(bq)
        rot = wp.transform_get_rotation(bq)
        ref_q[q_start + 0] = pos[0]
        ref_q[q_start + 1] = pos[1]
        ref_q[q_start + 2] = pos[2]
        ref_q[q_start + 3] = rot[0]
        ref_q[q_start + 4] = rot[1]
        ref_q[q_start + 5] = rot[2]
        ref_q[q_start + 6] = rot[3]
    elif jtype == JointType.BALL:
        ref_q[q_start + 0] = 0.0
        ref_q[q_start + 1] = 0.0
        ref_q[q_start + 2] = 0.0
        ref_q[q_start + 3] = 1.0
    elif jtype == JointType.PRISMATIC or jtype == JointType.REVOLUTE or jtype == JointType.D6:
        coord_count = joint_dof_dim[j, 0] + joint_dof_dim[j, 1]
        for k in range(coord_count):
            ref_val = float(0.0)
            if dof_ref:
                ref_val = dof_ref[qd_start + k]
            ref_q[q_start + k] = ref_val


@wp.kernel
def update_connect_constraint_rel_body_poses_at_qref_kernel(
    eq_constraint_type: wp.array[wp.int32],
    eq_constraint_body1: wp.array[wp.int32],
    eq_constraint_body2: wp.array[wp.int32],
    ref_body_q: wp.array[wp.transform],
    # outputs
    q_rel_out: wp.array[wp.quat],
    t_rel_out: wp.array[wp.vec3],
):
    """Compute relative body transforms for CONNECT constraints at the reference pose.

    Iterates over equality constraints ``[i]``. For each CONNECT constraint,
    computes ``q_rel`` and ``t_rel`` from the reference body poses such that::

        anchor2 = quat_rotate(q_rel, anchor1) + t_rel

    where ``q_rel = inv(q2) * q1`` and
    ``t_rel = quat_rotate(inv(q2), pos1 - pos2)``.

    These values are constant for a given reference configuration, so when
    ``anchor1`` changes at runtime ``anchor2`` can be recomputed without
    re-running forward kinematics. Non-CONNECT constraints are skipped.

    Args:
        eq_constraint_type: Constraint type enum per constraint,
            shape ``[equality_constraint_count]``.
        eq_constraint_body1: First body index per constraint (-1 for world),
            shape ``[equality_constraint_count]``.
        eq_constraint_body2: Second body index per constraint (-1 for world),
            shape ``[equality_constraint_count]``.
        ref_body_q: Body transforms at the reference pose [m],
            shape ``[body_count]``, dtype ``wp.transform``.
        q_rel_out: *(output)* Relative rotation ``inv(q2) * q1`` per
            constraint, shape ``[equality_constraint_count]``,
            dtype ``wp.quat``.
        t_rel_out: *(output)* Relative translation [m] per constraint,
            shape ``[equality_constraint_count]``, dtype ``wp.vec3``.
    """
    i = wp.tid()

    if eq_constraint_type[i] != EqType.CONNECT:
        return

    body1 = eq_constraint_body1[i]
    body2 = eq_constraint_body2[i]

    # Extract world-space pose for body1
    if body1 == -1:
        pos1 = wp.vec3(0.0, 0.0, 0.0)
        q1 = wp.quat_identity()
    else:
        tf1 = ref_body_q[body1]
        pos1 = wp.transform_get_translation(tf1)
        q1 = wp.transform_get_rotation(tf1)

    # Extract world-space pose for body2
    if body2 == -1:
        pos2 = wp.vec3(0.0, 0.0, 0.0)
        q2 = wp.quat_identity()
    else:
        tf2 = ref_body_q[body2]
        pos2 = wp.transform_get_translation(tf2)
        q2 = wp.transform_get_rotation(tf2)

    # q_rel = inv(q2) * q1
    # t = quat_rotate(inv(q2), pos1 - pos2)
    q2_inv = wp.quat_inverse(q2)
    q_rel_out[i] = q2_inv * q1
    t_rel_out[i] = wp.quat_rotate(q2_inv, pos1 - pos2)


@wp.kernel
def update_connect_constraint_anchors_kernel(
    mjc_eq_to_newton_eq: wp.array2d[wp.int32],
    eq_constraint_type: wp.array[wp.int32],
    eq_constraint_anchor: wp.array[wp.vec3],
    connect_anchor2_q: wp.array[wp.quat],
    connect_anchor2_t: wp.array[wp.vec3],
    # output
    eq_data_out: wp.array2d[vec11],
):
    """Write CONNECT constraint anchors into MuJoCo ``eq_data``.

    Iterates over MuJoCo equality constraints ``[world, eq]``. For each
    CONNECT constraint, copies ``anchor1`` [m] from Newton into
    ``eq_data[0:3]`` and computes::

        anchor2 = quat_rotate(q_rel, anchor1) + t_rel

    into ``eq_data[3:6]``. Non-CONNECT constraints and unmapped entries
    (``newton_eq < 0``) are skipped.

    Args:
        mjc_eq_to_newton_eq: Mapping from MuJoCo ``[world, eq]`` to Newton
            equality constraint index, shape ``[world_count, neq]``.
            Negative values indicate unmapped entries.
        eq_constraint_type: Constraint type enum per Newton constraint,
            shape ``[equality_constraint_count]``.
        eq_constraint_anchor: Anchor position on body 1 [m] per Newton
            constraint, shape ``[equality_constraint_count]``,
            dtype ``wp.vec3``.
        connect_anchor2_q: Precomputed relative rotation per constraint,
            shape ``[equality_constraint_count]``, dtype ``wp.quat``.
        connect_anchor2_t: Precomputed relative translation [m] per
            constraint, shape ``[equality_constraint_count]``,
            dtype ``wp.vec3``.
        eq_data_out: *(output)* MuJoCo equality constraint data,
            shape ``[world_count, neq]``, dtype ``vec11``.
            Slots ``[0:3]`` receive ``anchor1`` and ``[3:6]`` receive
            ``anchor2``.
    """
    world, mjc_eq = wp.tid()
    newton_eq = mjc_eq_to_newton_eq[world, mjc_eq]
    if newton_eq < 0:
        return

    if eq_constraint_type[newton_eq] != EqType.CONNECT:
        return

    anchor = eq_constraint_anchor[newton_eq]
    q = connect_anchor2_q[newton_eq]
    t = connect_anchor2_t[newton_eq]
    anchor2 = wp.quat_rotate(q, anchor) + t

    data = eq_data_out[world, mjc_eq]
    data[0] = anchor[0]
    data[1] = anchor[1]
    data[2] = anchor[2]
    data[3] = anchor2[0]
    data[4] = anchor2[1]
    data[5] = anchor2[2]
    eq_data_out[world, mjc_eq] = data


@wp.kernel
def update_jnt_connect_constraint_rel_body_poses_at_qref_kernel(
    mjc_eq_to_newton_jnt: wp.array2d[wp.int32],
    joint_parent: wp.array[wp.int32],
    joint_child: wp.array[wp.int32],
    ref_body_q: wp.array[wp.transform],
    # outputs
    q_rel_out: wp.array2d[wp.quat],
    t_rel_out: wp.array2d[wp.vec3],
):
    """Compute relative body transforms for joint-synthesized CONNECT constraints.

    For each MuJoCo equality constraint that maps to a Newton joint (via
    ``mjc_eq_to_newton_jnt``), computes ``q_rel`` and ``t_rel`` from the
    reference body poses of the joint's parent and child bodies such that::

        anchor2 = quat_rotate(q_rel, anchor1) + t_rel

    where ``q_rel = inv(q_child) * q_parent`` and
    ``t_rel = quat_rotate(inv(q_child), pos_parent - pos_child)``.

    Unmapped entries (``newton_jnt < 0``) are skipped.

    Args:
        mjc_eq_to_newton_jnt: Mapping from MuJoCo ``[world, eq]`` to Newton
            joint index, shape ``[world_count, neq]``.
            Negative values indicate unmapped entries.
        joint_parent: Parent body index per joint,
            shape ``[joint_count]``, dtype ``wp.int32``.
        joint_child: Child body index per joint,
            shape ``[joint_count]``, dtype ``wp.int32``.
        ref_body_q: Body transforms at the reference pose [m],
            shape ``[body_count]``, dtype ``wp.transform``.
        q_rel_out: *(output)* Relative rotation per ``[world, eq]``,
            shape ``[world_count, neq]``, dtype ``wp.quat``.
        t_rel_out: *(output)* Relative translation [m] per ``[world, eq]``,
            shape ``[world_count, neq]``, dtype ``wp.vec3``.
    """
    world, mjc_eq = wp.tid()
    newton_jnt = mjc_eq_to_newton_jnt[world, mjc_eq]
    if newton_jnt < 0:
        return

    body1 = joint_parent[newton_jnt]
    body2 = joint_child[newton_jnt]

    # Extract world-space pose for body1 (parent)
    if body1 == -1:
        pos1 = wp.vec3(0.0, 0.0, 0.0)
        q1 = wp.quat_identity()
    else:
        tf1 = ref_body_q[body1]
        pos1 = wp.transform_get_translation(tf1)
        q1 = wp.transform_get_rotation(tf1)

    # Extract world-space pose for body2 (child)
    if body2 == -1:
        pos2 = wp.vec3(0.0, 0.0, 0.0)
        q2 = wp.quat_identity()
    else:
        tf2 = ref_body_q[body2]
        pos2 = wp.transform_get_translation(tf2)
        q2 = wp.transform_get_rotation(tf2)

    # q_rel = inv(q_child) * q_parent
    # t_rel = quat_rotate(inv(q_child), pos_parent - pos_child)
    q2_inv = wp.quat_inverse(q2)
    q_rel_out[world, mjc_eq] = q2_inv * q1
    t_rel_out[world, mjc_eq] = wp.quat_rotate(q2_inv, pos1 - pos2)


@wp.kernel
def recompute_jnt_eq_anchor1_kernel(
    mjc_eq_to_newton_jnt: wp.array2d[wp.int32],
    has_axis_offset: wp.array2d[wp.int32],
    axis_offset_distance: float,
    joint_X_p: wp.array[wp.transform],
    joint_axis: wp.array[wp.vec3],
    joint_qd_start: wp.array[wp.int32],
    # outputs
    jnt_eq_anchor1: wp.array2d[wp.vec3],
):
    """Recompute body1-local anchor positions for joint-synthesized CONNECT constraints.

    For each mapped ``[world, eq]`` entry, reads the translation from the
    joint's parent transform (``joint_X_p``).  When ``has_axis_offset`` is
    set, adds ``axis_offset_distance`` along the hinge axis rotated into
    the parent body frame.

    Args:
        mjc_eq_to_newton_jnt: Mapping from MuJoCo ``[world, eq]`` to Newton
            joint index, shape ``[world_count, neq]``.
            Negative values indicate unmapped entries.
        has_axis_offset: ``1`` for the second hinge CONNECT that is offset
            along the joint axis, ``0`` otherwise,
            shape ``[world_count, neq]``.
        axis_offset_distance: Distance [m] along the hinge axis for the
            second CONNECT constraint point.
        joint_X_p: Parent-body-local joint transform [m],
            shape ``[joint_count]``, dtype ``wp.transform``.
        joint_axis: Joint axis in joint-local frame,
            shape ``[joint_dof_count]``, dtype ``wp.vec3``.
        joint_qd_start: Start index into ``joint_axis`` for each joint,
            shape ``[joint_count]``, dtype ``wp.int32``.
        jnt_eq_anchor1: *(output)* Body1-local anchor [m] per
            ``[world, eq]``, shape ``[world_count, neq]``,
            dtype ``wp.vec3``.
    """
    world, mjc_eq = wp.tid()
    newton_jnt = mjc_eq_to_newton_jnt[world, mjc_eq]
    if newton_jnt < 0:
        return

    xform = joint_X_p[newton_jnt]
    anchor = wp.transform_get_translation(xform)

    if has_axis_offset[world, mjc_eq] != 0:
        qd_start = joint_qd_start[newton_jnt]
        axis_local = joint_axis[qd_start]
        axis_parent = wp.quat_rotate(wp.transform_get_rotation(xform), axis_local)
        anchor = anchor + axis_offset_distance * axis_parent

    jnt_eq_anchor1[world, mjc_eq] = anchor


@wp.kernel
def update_jnt_connect_constraint_anchors_kernel(
    mjc_eq_to_newton_jnt: wp.array2d[wp.int32],
    jnt_eq_anchor1: wp.array2d[wp.vec3],
    jnt_eq_q_rel: wp.array2d[wp.quat],
    jnt_eq_t_rel: wp.array2d[wp.vec3],
    # output
    eq_data_out: wp.array2d[vec11],
):
    """Write joint-synthesized CONNECT constraint anchors into MuJoCo ``eq_data``.

    For each MuJoCo equality constraint that maps to a Newton joint,
    copies ``anchor1`` [m] into ``eq_data[0:3]`` and computes::

        anchor2 = quat_rotate(q_rel, anchor1) + t_rel

    into ``eq_data[3:6]``. Unmapped entries (``newton_jnt < 0``) are skipped.

    Args:
        mjc_eq_to_newton_jnt: Mapping from MuJoCo ``[world, eq]`` to Newton
            joint index, shape ``[world_count, neq]``.
            Negative values indicate unmapped entries.
        jnt_eq_anchor1: Pre-computed anchor on body1 [m] per ``[world, eq]``,
            shape ``[world_count, neq]``, dtype ``wp.vec3``.
        jnt_eq_q_rel: Relative rotation per ``[world, eq]``,
            shape ``[world_count, neq]``, dtype ``wp.quat``.
        jnt_eq_t_rel: Relative translation [m] per ``[world, eq]``,
            shape ``[world_count, neq]``, dtype ``wp.vec3``.
        eq_data_out: *(output)* MuJoCo equality constraint data,
            shape ``[world_count, neq]``, dtype ``vec11``.
            Slots ``[0:3]`` receive ``anchor1`` and ``[3:6]`` receive
            ``anchor2``.
    """
    world, mjc_eq = wp.tid()
    newton_jnt = mjc_eq_to_newton_jnt[world, mjc_eq]
    if newton_jnt < 0:
        return

    anchor = jnt_eq_anchor1[world, mjc_eq]
    q = jnt_eq_q_rel[world, mjc_eq]
    t = jnt_eq_t_rel[world, mjc_eq]
    anchor2 = wp.quat_rotate(q, anchor) + t

    data = eq_data_out[world, mjc_eq]
    data[0] = anchor[0]
    data[1] = anchor[1]
    data[2] = anchor[2]
    data[3] = anchor2[0]
    data[4] = anchor2[1]
    data[5] = anchor2[2]
    eq_data_out[world, mjc_eq] = data


def create_convert_mjw_contacts_to_newton_kernel():
    """Create contact conversion kernel; deferred so ``wp.static`` doesn't import mujoco_warp at module load."""

    @wp.kernel
    def convert_mjw_contacts_to_newton_kernel(
        # inputs
        mjc_geom_to_newton_shape: wp.array2d[wp.int32],
        mj_opt_cone: int,
        mj_nacon: wp.array[wp.int32],
        mj_contact_pos: wp.array[wp.vec3],
        mj_contact_frame: wp.array[wp.mat33f],
        mj_contact_friction: wp.array[vec5],
        mj_contact_dist: wp.array[float],
        mj_contact_dim: wp.array[int],
        mj_contact_geom: wp.array[wp.vec2i],
        mj_contact_efc_address: wp.array2d[int],
        mj_contact_worldid: wp.array[wp.int32],
        mj_efc_force: wp.array2d[float],
        mj_geom_bodyid: wp.array[int],
        mj_xpos: wp.array2d[wp.vec3],
        mj_xquat: wp.array2d[wp.quatf],
        njmax: int,
        # outputs
        rigid_contact_count: wp.array[wp.int32],
        rigid_contact_shape0: wp.array[wp.int32],
        rigid_contact_shape1: wp.array[wp.int32],
        rigid_contact_point0: wp.array[wp.vec3],
        rigid_contact_point1: wp.array[wp.vec3],
        rigid_contact_normal: wp.array[wp.vec3],
        contact_force: wp.array[wp.spatial_vector],
    ):
        """Convert MuJoCo contacts to Newton contact format.

        Uses mjc_geom_to_newton_shape to convert MuJoCo geom indices to Newton shape indices.
        Contact positions are converted from MuJoCo world frame to Newton body-local frame.
        Contact forces are computed via ``mujoco_warp`` ``contact_force_fn``.
        """
        contact_idx = wp.tid()
        n_contacts = mj_nacon[0]

        if contact_idx == 0:
            rigid_contact_count[0] = n_contacts

        if contact_idx >= n_contacts:
            return

        world = mj_contact_worldid[contact_idx]
        geoms_mjw = mj_contact_geom[contact_idx]

        normal = mj_contact_frame[contact_idx][0]
        pos_world = mj_contact_pos[contact_idx]

        rigid_contact_shape0[contact_idx] = mjc_geom_to_newton_shape[world, geoms_mjw[0]]
        rigid_contact_shape1[contact_idx] = mjc_geom_to_newton_shape[world, geoms_mjw[1]]
        rigid_contact_normal[contact_idx] = normal

        # Convert contact position from world frame to body-local frame for each shape.
        # MuJoCo contact.pos is the midpoint in world frame; we transform it into each
        # body's local frame to match Newton's convention (see collide.py write_contact).
        body_a = mj_geom_bodyid[geoms_mjw[0]]
        body_b = mj_geom_bodyid[geoms_mjw[1]]

        X_wb_a = wp.transform_identity()
        X_wb_b = wp.transform_identity()
        if body_a > 0:
            X_wb_a = wp.transform(mj_xpos[world, body_a], quat_wxyz_to_xyzw(mj_xquat[world, body_a]))
        if body_b > 0:
            X_wb_b = wp.transform(mj_xpos[world, body_b], quat_wxyz_to_xyzw(mj_xquat[world, body_b]))

        dist = mj_contact_dist[contact_idx]
        point0_world = pos_world - 0.5 * dist * normal
        point1_world = pos_world + 0.5 * dist * normal

        rigid_contact_point0[contact_idx] = wp.transform_point(wp.transform_inverse(X_wb_a), point0_world)
        rigid_contact_point1[contact_idx] = wp.transform_point(wp.transform_inverse(X_wb_b), point1_world)

        if contact_force:
            # Negate: contact_force_fn returns force on geom2; Newton stores force on shape0 (geom1).
            contact_force[contact_idx] = -wp.static(_import_contact_force_fn())(
                mj_opt_cone,
                mj_contact_frame,
                mj_contact_friction,
                mj_contact_dim,
                mj_contact_efc_address,
                mj_efc_force,
                njmax,
                mj_nacon,
                world,
                contact_idx,
                True,
            )

    return convert_mjw_contacts_to_newton_kernel


# Import control source/type enums and create warp constants

CTRL_SOURCE_JOINT_TARGET = wp.constant(0)
CTRL_SOURCE_CTRL_DIRECT = wp.constant(1)


@wp.kernel
def apply_mjc_control_kernel(
    mjc_actuator_ctrl_source: wp.array[wp.int32],
    mjc_actuator_to_newton_idx: wp.array[wp.int32],
    joint_target_pos: wp.array[wp.float32],
    joint_target_vel: wp.array[wp.float32],
    mujoco_ctrl: wp.array[wp.float32],
    dofs_per_world: wp.int32,
    ctrls_per_world: wp.int32,
    # outputs
    mj_ctrl: wp.array2d[wp.float32],
):
    """Apply Newton control inputs to MuJoCo control array.

    For JOINT_TARGET (source=0), uses sign encoding in mjc_actuator_to_newton_idx:
    - Positive value (>=0): position actuator, newton_axis = value
    - Value of -1: unmapped/skip
    - Negative value (<=-2): velocity actuator, newton_axis = -(value + 2)

    For CTRL_DIRECT (source=1), mjc_actuator_to_newton_idx is the ctrl index.

    Args:
        mjc_actuator_ctrl_source: 0=JOINT_TARGET, 1=CTRL_DIRECT
        mjc_actuator_to_newton_idx: Index into Newton array (sign-encoded for JOINT_TARGET)
        joint_target_pos: Per-DOF position targets
        joint_target_vel: Per-DOF velocity targets
        mujoco_ctrl: Direct control inputs (from control.mujoco.ctrl)
        dofs_per_world: Number of DOFs per world
        ctrls_per_world: Number of ctrl inputs per world
        mj_ctrl: Output MuJoCo control array
    """
    world, actuator = wp.tid()
    source = mjc_actuator_ctrl_source[actuator]
    idx = mjc_actuator_to_newton_idx[actuator]

    if source == CTRL_SOURCE_JOINT_TARGET:
        if idx >= 0:
            # Position actuator
            world_dof = world * dofs_per_world + idx
            mj_ctrl[world, actuator] = joint_target_pos[world_dof]
        elif idx == -1:
            # Unmapped/skip
            return
        else:
            # Velocity actuator: newton_axis = -(idx + 2)
            newton_axis = -(idx + 2)
            world_dof = world * dofs_per_world + newton_axis
            mj_ctrl[world, actuator] = joint_target_vel[world_dof]
    else:  # CTRL_SOURCE_CTRL_DIRECT
        world_ctrl_idx = world * ctrls_per_world + idx
        if world_ctrl_idx < mujoco_ctrl.shape[0]:
            mj_ctrl[world, actuator] = mujoco_ctrl[world_ctrl_idx]


@wp.kernel
def apply_mjc_body_f_kernel(
    mjc_body_to_newton: wp.array2d[wp.int32],
    body_flags: wp.array[wp.int32],
    body_f: wp.array[wp.spatial_vector],
    # outputs
    xfrc_applied: wp.array2d[wp.spatial_vector],
):
    """Apply Newton body forces to MuJoCo xfrc_applied array.

    Iterates over MuJoCo bodies [world, mjc_body], looks up Newton body index,
    and copies the force.
    """
    world, mjc_body = wp.tid()
    newton_body = mjc_body_to_newton[world, mjc_body]
    if newton_body < 0 or (body_flags[newton_body] & BodyFlags.KINEMATIC) != 0:
        xfrc_applied[world, mjc_body] = wp.spatial_vector(wp.vec3(0.0, 0.0, 0.0), wp.vec3(0.0, 0.0, 0.0))
        return

    f = body_f[newton_body]
    v = wp.vec3(f[0], f[1], f[2])
    w = wp.vec3(f[3], f[4], f[5])
    xfrc_applied[world, mjc_body] = wp.spatial_vector(v, w)


@wp.kernel
def apply_mjc_qfrc_kernel(
    joint_f: wp.array[wp.float32],
    joint_type: wp.array[wp.int32],
    joint_child: wp.array[wp.int32],
    body_flags: wp.array[wp.int32],
    joint_qd_start: wp.array[wp.int32],
    joint_dof_dim: wp.array2d[wp.int32],
    joints_per_world: int,
    mj_qd_start: wp.array[wp.int32],
    # outputs
    qfrc_applied: wp.array2d[wp.float32],
):
    worldid, jntid = wp.tid()

    # Skip loop joints — they have no MuJoCo DOF entries
    qd_i = mj_qd_start[jntid]
    if qd_i < 0:
        return

    wqd_i = joint_qd_start[joints_per_world * worldid + jntid]
    joint_id = joints_per_world * worldid + jntid
    jtype = joint_type[jntid]
    dof_count = joint_dof_dim[jntid, 0] + joint_dof_dim[jntid, 1]

    for i in range(dof_count):
        qfrc_applied[worldid, qd_i + i] = 0.0

    if (body_flags[joint_child[joint_id]] & BodyFlags.KINEMATIC) != 0:
        return

    # Free/DISTANCE joint forces are routed via xfrc_applied in a separate kernel
    # to preserve COM-wrench semantics; skip them here.
    if jtype == JointType.FREE or jtype == JointType.DISTANCE:
        return
    elif jtype == JointType.BALL:
        qfrc_applied[worldid, qd_i + 0] = joint_f[wqd_i + 0]
        qfrc_applied[worldid, qd_i + 1] = joint_f[wqd_i + 1]
        qfrc_applied[worldid, qd_i + 2] = joint_f[wqd_i + 2]
    else:
        for i in range(dof_count):
            qfrc_applied[worldid, qd_i + i] = joint_f[wqd_i + i]


@wp.kernel
def apply_mjc_free_joint_f_to_body_f_kernel(
    mjc_body_to_newton: wp.array2d[wp.int32],
    body_flags: wp.array[wp.int32],
    body_free_qd_start: wp.array[wp.int32],
    joint_f: wp.array[wp.float32],
    # outputs
    xfrc_applied: wp.array2d[wp.spatial_vector],
):
    worldid, mjc_body = wp.tid()
    newton_body = mjc_body_to_newton[worldid, mjc_body]
    if newton_body < 0 or (body_flags[newton_body] & BodyFlags.KINEMATIC) != 0:
        return

    qd_start = body_free_qd_start[newton_body]
    if qd_start < 0:
        return

    v = wp.vec3(joint_f[qd_start + 0], joint_f[qd_start + 1], joint_f[qd_start + 2])
    w = wp.vec3(joint_f[qd_start + 3], joint_f[qd_start + 4], joint_f[qd_start + 5])
    xfrc = xfrc_applied[worldid, mjc_body]
    xfrc_applied[worldid, mjc_body] = wp.spatial_vector(
        wp.spatial_top(xfrc) + v,
        wp.spatial_bottom(xfrc) + w,
    )


@wp.func
def eval_single_articulation_fk(
    joint_start: int,
    joint_end: int,
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
    # outputs
    body_q: wp.array[wp.transform],
    body_qd: wp.array[wp.spatial_vector],
):
    for i in range(joint_start, joint_end):
        articulation = joint_articulation[i]
        if articulation == -1:
            continue

        parent = joint_parent[i]
        child = joint_child[i]

        # compute transform across the joint
        type = joint_type[i]

        X_pj = joint_X_p[i]
        X_cj = joint_X_c[i]

        q_start = joint_q_start[i]
        qd_start = joint_qd_start[i]
        lin_axis_count = joint_dof_dim[i, 0]
        ang_axis_count = joint_dof_dim[i, 1]

        X_j = wp.transform_identity()
        v_j = wp.spatial_vector(wp.vec3(), wp.vec3())

        if type == JointType.PRISMATIC:
            axis = joint_axis[qd_start]

            q = joint_q[q_start]
            qd = joint_qd[qd_start]

            X_j = wp.transform(axis * q, wp.quat_identity())
            v_j = wp.spatial_vector(axis * qd, wp.vec3())

        if type == JointType.REVOLUTE:
            axis = joint_axis[qd_start]

            q = joint_q[q_start]
            qd = joint_qd[qd_start]

            X_j = wp.transform(wp.vec3(), wp.quat_from_axis_angle(axis, q))
            v_j = wp.spatial_vector(wp.vec3(), axis * qd)

        if type == JointType.BALL:
            r = wp.quat(joint_q[q_start + 0], joint_q[q_start + 1], joint_q[q_start + 2], joint_q[q_start + 3])

            w = wp.vec3(joint_qd[qd_start + 0], joint_qd[qd_start + 1], joint_qd[qd_start + 2])

            X_j = wp.transform(wp.vec3(), r)
            v_j = wp.spatial_vector(wp.vec3(), w)

        if type == JointType.FREE or type == JointType.DISTANCE:
            t = wp.transform(
                wp.vec3(joint_q[q_start + 0], joint_q[q_start + 1], joint_q[q_start + 2]),
                wp.quat(joint_q[q_start + 3], joint_q[q_start + 4], joint_q[q_start + 5], joint_q[q_start + 6]),
            )

            v = wp.spatial_vector(
                wp.vec3(joint_qd[qd_start + 0], joint_qd[qd_start + 1], joint_qd[qd_start + 2]),
                wp.vec3(joint_qd[qd_start + 3], joint_qd[qd_start + 4], joint_qd[qd_start + 5]),
            )

            X_j = t
            v_j = v

        if type == JointType.D6:
            pos = wp.vec3(0.0)
            rot = wp.quat_identity()
            vel_v = wp.vec3(0.0)
            vel_w = wp.vec3(0.0)

            for j in range(lin_axis_count):
                axis = joint_axis[qd_start + j]
                pos += axis * joint_q[q_start + j]
                vel_v += axis * joint_qd[qd_start + j]

            iq = q_start + lin_axis_count
            iqd = qd_start + lin_axis_count
            for j in range(ang_axis_count):
                axis = joint_axis[iqd + j]
                rot = rot * wp.quat_from_axis_angle(axis, joint_q[iq + j])
                vel_w += joint_qd[iqd + j] * axis

            X_j = wp.transform(pos, rot)
            v_j = wp.spatial_vector(vel_v, vel_w)  # vel_v=linear, vel_w=angular

        # transform from world to parent joint anchor frame
        X_wpj = X_pj
        if parent >= 0:
            X_wp = body_q[parent]
            X_wpj = X_wp * X_wpj

        # transform from world to joint anchor frame at child body
        X_wcj = X_wpj * X_j
        # transform from world to child body frame
        X_wc = X_wcj * wp.transform_inverse(X_cj)

        x_child_origin = wp.transform_get_translation(X_wc)
        v_parent_origin = wp.vec3()
        w_parent = wp.vec3()
        if parent >= 0:
            v_wp = body_qd[parent]
            w_parent = wp.spatial_bottom(v_wp)
            v_parent_origin = com_twist_to_point_velocity(v_wp, X_wp, body_com[parent], x_child_origin)

        linear_joint_world = wp.transform_vector(X_wpj, wp.spatial_top(v_j))
        angular_joint_world = wp.transform_vector(X_wpj, wp.spatial_bottom(v_j))
        if type == JointType.FREE or type == JointType.DISTANCE:
            linear_joint_origin = linear_joint_world - wp.cross(
                angular_joint_world, wp.transform_vector(X_wc, body_com[child])
            )
        else:
            child_origin_offset_world = x_child_origin - wp.transform_get_translation(X_wcj)
            linear_joint_origin = linear_joint_world + wp.cross(angular_joint_world, child_origin_offset_world)

        v_wc_origin = wp.spatial_vector(
            v_parent_origin + linear_joint_origin,
            w_parent + angular_joint_world,
        )  # spatial vector with (linear, angular) ordering

        body_q[child] = X_wc
        body_qd[child] = origin_twist_to_com_twist(v_wc_origin, X_wc, body_com[child])


@wp.kernel
def eval_articulation_fk(
    articulation_start: wp.array[int],
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
    # outputs
    body_q: wp.array[wp.transform],
    body_qd: wp.array[wp.spatial_vector],
):
    tid = wp.tid()

    joint_start = articulation_start[tid]
    joint_end = articulation_start[tid + 1]

    eval_single_articulation_fk(
        joint_start,
        joint_end,
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
        # outputs
        body_q,
        body_qd,
    )


@wp.kernel
def convert_body_xforms_to_warp_kernel(
    mjc_body_to_newton: wp.array2d[wp.int32],
    xpos: wp.array2d[wp.vec3],
    xquat: wp.array2d[wp.quat],
    # outputs
    body_q: wp.array[wp.transform],
):
    """Convert MuJoCo body transforms to Newton body_q array.

    Iterates over MuJoCo bodies [world, mjc_body], looks up Newton body index,
    reads MuJoCo position/quaternion, and writes to Newton body_q.
    """
    world, mjc_body = wp.tid()
    newton_body = mjc_body_to_newton[world, mjc_body]
    if newton_body >= 0:
        pos = xpos[world, mjc_body]
        quat = xquat[world, mjc_body]
        # convert from wxyz to xyzw
        quat = quat_wxyz_to_xyzw(quat)
        body_q[newton_body] = wp.transform(pos, quat)


@wp.kernel
def update_body_mass_ipos_kernel(
    mjc_body_to_newton: wp.array2d[wp.int32],
    body_com: wp.array[wp.vec3f],
    body_mass: wp.array[float],
    body_gravcomp: wp.array[float],
    # outputs
    body_ipos: wp.array2d[wp.vec3f],
    body_mass_out: wp.array2d[float],
    body_gravcomp_out: wp.array2d[float],
):
    """Update MuJoCo body mass and inertial position from Newton body properties.

    Iterates over MuJoCo bodies [world, mjc_body], looks up Newton body index,
    and copies mass, COM, and gravcomp.
    """
    world, mjc_body = wp.tid()
    newton_body = mjc_body_to_newton[world, mjc_body]
    if newton_body < 0:
        return

    # update COM position
    body_ipos[world, mjc_body] = body_com[newton_body]

    # update mass
    body_mass_out[world, mjc_body] = body_mass[newton_body]

    # update gravcomp
    if body_gravcomp:
        body_gravcomp_out[world, mjc_body] = body_gravcomp[newton_body]


@wp.func
def _sort_eigenpairs_descending(eigenvalues: wp.vec3f, eigenvectors: wp.mat33f) -> tuple[wp.vec3f, wp.mat33f]:
    """Sort eigenvalues descending and reorder eigenvector columns to match."""
    # Transpose to work with rows (easier swapping)
    vecs_t = wp.transpose(eigenvectors)
    vals = eigenvalues

    # Bubble sort for 3 elements
    for i in range(2):
        for j in range(2 - i):
            if vals[j] < vals[j + 1]:
                # Swap eigenvalues
                tmp_val = vals[j]
                vals[j] = vals[j + 1]
                vals[j + 1] = tmp_val
                # Swap eigenvector rows
                tmp_vec = vecs_t[j]
                vecs_t[j] = vecs_t[j + 1]
                vecs_t[j + 1] = tmp_vec

    return vals, wp.transpose(vecs_t)


@wp.func
def _ensure_proper_rotation(V: wp.mat33f) -> wp.mat33f:
    """Ensure matrix is a proper rotation (det=+1) by negating a column if needed.

    wp.eig3 can return eigenvector matrices with det=-1 (reflections), which
    cannot be converted to valid quaternions. This fixes it by negating the
    third column when det < 0.
    """
    if wp.determinant(V) < 0.0:
        # Negate third column to flip determinant sign
        return wp.mat33(
            V[0, 0],
            V[0, 1],
            -V[0, 2],
            V[1, 0],
            V[1, 1],
            -V[1, 2],
            V[2, 0],
            V[2, 1],
            -V[2, 2],
        )
    return V


@wp.kernel
def update_body_inertia_kernel(
    mjc_body_to_newton: wp.array2d[wp.int32],
    body_inertia: wp.array[wp.mat33f],
    # outputs
    body_inertia_out: wp.array2d[wp.vec3f],
    body_iquat_out: wp.array2d[wp.quatf],
):
    """Update MuJoCo body inertia from Newton body inertia tensor.

    Iterates over MuJoCo bodies [world, mjc_body], looks up Newton body index,
    computes eigendecomposition, and writes to MuJoCo arrays.
    """
    world, mjc_body = wp.tid()
    newton_body = mjc_body_to_newton[world, mjc_body]
    if newton_body < 0:
        return

    # Eigendecomposition of inertia tensor
    eigenvectors, eigenvalues = wp.eig3(body_inertia[newton_body])

    # Sort descending (MuJoCo convention)
    eigenvalues, V = _sort_eigenpairs_descending(eigenvalues, eigenvectors)

    # Ensure proper rotation matrix (det=+1) for valid quaternion conversion
    V = _ensure_proper_rotation(V)

    # Convert to quaternion (Warp uses xyzw, mujoco_warp stores wxyz)
    q = wp.normalize(wp.quat_from_matrix(V))

    # Convert from xyzw to wxyz
    q = quat_xyzw_to_wxyz(q)

    # Store results
    body_inertia_out[world, mjc_body] = eigenvalues
    body_iquat_out[world, mjc_body] = q


@wp.kernel(module="unique", enable_backward=False)
def repeat_array_kernel(
    src: wp.array[Any],
    nelems_per_world: int,
    dst: wp.array[Any],
):
    tid = wp.tid()
    src_idx = tid % nelems_per_world
    dst[tid] = src[src_idx]


@wp.kernel
def update_solver_options_kernel(
    # WORLD frequency inputs (None if overridden/unavailable)
    newton_impratio: wp.array[float],
    newton_tolerance: wp.array[float],
    newton_ls_tolerance: wp.array[float],
    newton_ccd_tolerance: wp.array[float],
    newton_density: wp.array[float],
    newton_viscosity: wp.array[float],
    newton_wind: wp.array[wp.vec3],
    newton_magnetic: wp.array[wp.vec3],
    # outputs - MuJoCo per-world arrays
    opt_impratio_invsqrt: wp.array[float],
    opt_tolerance: wp.array[float],
    opt_ls_tolerance: wp.array[float],
    opt_ccd_tolerance: wp.array[float],
    opt_density: wp.array[float],
    opt_viscosity: wp.array[float],
    opt_wind: wp.array[wp.vec3],
    opt_magnetic: wp.array[wp.vec3],
):
    """Update per-world solver options from Newton model.

    Args:
        newton_impratio: Per-world impratio values from Newton model (None if overridden)
        newton_tolerance: Per-world tolerance values (None if overridden)
        newton_ls_tolerance: Per-world line search tolerance values (None if overridden)
        newton_ccd_tolerance: Per-world CCD tolerance values (None if overridden)
        newton_density: Per-world medium density values (None if overridden)
        newton_viscosity: Per-world medium viscosity values (None if overridden)
        newton_wind: Per-world wind velocity vectors (None if overridden)
        newton_magnetic: Per-world magnetic flux vectors (None if overridden)
        opt_impratio_invsqrt: MuJoCo Warp opt.impratio_invsqrt array (shape: nworld)
        opt_tolerance: MuJoCo Warp opt.tolerance array (shape: nworld)
        opt_ls_tolerance: MuJoCo Warp opt.ls_tolerance array (shape: nworld)
        opt_ccd_tolerance: MuJoCo Warp opt.ccd_tolerance array (shape: nworld)
        opt_density: MuJoCo Warp opt.density array (shape: nworld)
        opt_viscosity: MuJoCo Warp opt.viscosity array (shape: nworld)
        opt_wind: MuJoCo Warp opt.wind array (shape: nworld)
        opt_magnetic: MuJoCo Warp opt.magnetic array (shape: nworld)
    """
    worldid = wp.tid()

    # Only update if Newton array exists (None means overridden or not available)
    if newton_impratio:
        # MuJoCo stores impratio as inverse square root
        # Guard against zero/negative values to avoid NaN/Inf
        impratio_val = newton_impratio[worldid]
        if impratio_val > 0.0:
            opt_impratio_invsqrt[worldid] = 1.0 / wp.sqrt(impratio_val)
        # else: skip update, keep existing MuJoCo default value

    if newton_tolerance:
        # MuJoCo Warp clamps tolerance to 1e-6 for float32 precision
        # See mujoco_warp/_src/io.py: opt.tolerance = max(opt.tolerance, 1e-6)
        opt_tolerance[worldid] = wp.max(newton_tolerance[worldid], 1.0e-6)

    if newton_ls_tolerance:
        opt_ls_tolerance[worldid] = newton_ls_tolerance[worldid]

    if newton_ccd_tolerance:
        opt_ccd_tolerance[worldid] = newton_ccd_tolerance[worldid]

    if newton_density:
        opt_density[worldid] = newton_density[worldid]

    if newton_viscosity:
        opt_viscosity[worldid] = newton_viscosity[worldid]

    if newton_wind:
        opt_wind[worldid] = newton_wind[worldid]

    if newton_magnetic:
        opt_magnetic[worldid] = newton_magnetic[worldid]


@wp.kernel
def update_axis_properties_kernel(
    mjc_actuator_ctrl_source: wp.array[wp.int32],
    mjc_actuator_to_newton_idx: wp.array[wp.int32],
    joint_target_ke: wp.array[float],
    joint_target_kd: wp.array[float],
    joint_target_mode: wp.array[wp.int32],
    dofs_per_world: wp.int32,
    # outputs
    actuator_bias: wp.array2d[vec10],
    actuator_gain: wp.array2d[vec10],
):
    """Update MuJoCo actuator gains from Newton per-DOF arrays.

    Only updates JOINT_TARGET actuators. CTRL_DIRECT actuators keep their gains
    from custom attributes.

    For JOINT_TARGET, uses sign encoding in mjc_actuator_to_newton_idx:
    - Positive value (>=0): position actuator, newton_axis = value
    - Value of -1: unmapped/skip
    - Negative value (<=-2): velocity actuator, newton_axis = -(value + 2)

    For POSITION-only actuators (joint_target_mode == JointTargetMode.POSITION), both
    kp and kd are synced since the position actuator includes damping. For
    POSITION_VELOCITY mode, only kp is synced to the position actuator (kd goes
    to the separate velocity actuator).

    Args:
        mjc_actuator_ctrl_source: 0=JOINT_TARGET, 1=CTRL_DIRECT
        mjc_actuator_to_newton_idx: Index into Newton array (sign-encoded for JOINT_TARGET)
        joint_target_ke: Per-DOF position gains (kp)
        joint_target_kd: Per-DOF velocity/damping gains (kd)
        joint_target_mode: Per-DOF target mode from Model.joint_target_mode
        dofs_per_world: Number of DOFs per world
    """
    world, actuator = wp.tid()
    source = mjc_actuator_ctrl_source[actuator]

    if source != CTRL_SOURCE_JOINT_TARGET:
        # CTRL_DIRECT: gains unchanged (set from custom attributes)
        return

    idx = mjc_actuator_to_newton_idx[actuator]
    if idx >= 0:
        # Position actuator - get kp from per-DOF array
        world_dof = world * dofs_per_world + idx
        kp = joint_target_ke[world_dof]
        actuator_bias[world, actuator][1] = -kp
        actuator_gain[world, actuator][0] = kp

        # For POSITION-only mode, also sync kd (damping) to the position actuator
        # For POSITION_VELOCITY mode, kd is handled by the separate velocity actuator
        mode = joint_target_mode[idx]  # Use template DOF index (idx) not world_dof
        if mode == JointTargetMode.POSITION:
            kd = joint_target_kd[world_dof]
            actuator_bias[world, actuator][2] = -kd
    elif idx == -1:
        # Unmapped/skip
        return
    else:
        # Velocity actuator - get kd from per-DOF array
        newton_axis = -(idx + 2)
        world_dof = world * dofs_per_world + newton_axis
        kd = joint_target_kd[world_dof]
        actuator_bias[world, actuator][2] = -kd
        actuator_gain[world, actuator][0] = kd


@wp.kernel
def update_ctrl_direct_actuator_properties_kernel(
    mjc_actuator_ctrl_source: wp.array[wp.int32],
    mjc_actuator_to_newton_idx: wp.array[wp.int32],
    newton_actuator_gainprm: wp.array[vec10],
    newton_actuator_biasprm: wp.array[vec10],
    newton_actuator_dynprm: wp.array[vec10],
    newton_actuator_ctrlrange: wp.array[wp.vec2],
    newton_actuator_forcerange: wp.array[wp.vec2],
    newton_actuator_actrange: wp.array[wp.vec2],
    newton_actuator_gear: wp.array[wp.spatial_vector],
    newton_actuator_cranklength: wp.array[float],
    actuators_per_world: wp.int32,
    # outputs
    actuator_gain: wp.array2d[vec10],
    actuator_bias: wp.array2d[vec10],
    actuator_dynprm: wp.array2d[vec10],
    actuator_ctrlrange: wp.array2d[wp.vec2],
    actuator_forcerange: wp.array2d[wp.vec2],
    actuator_actrange: wp.array2d[wp.vec2],
    actuator_gear: wp.array2d[wp.spatial_vector],
    actuator_cranklength: wp.array2d[float],
):
    """Update MuJoCo actuator properties for CTRL_DIRECT actuators from Newton custom attributes.

    Only updates actuators where mjc_actuator_ctrl_source == CTRL_DIRECT.
    Uses mjc_actuator_to_newton_idx to map from MuJoCo actuator index to Newton's
    mujoco:actuator frequency index.

    Args:
        mjc_actuator_ctrl_source: 0=JOINT_TARGET, 1=CTRL_DIRECT
        mjc_actuator_to_newton_idx: Index into Newton's mujoco:actuator arrays
        newton_actuator_gainprm: Newton's model.mujoco.actuator_gainprm
        newton_actuator_biasprm: Newton's model.mujoco.actuator_biasprm
        newton_actuator_dynprm: Newton's model.mujoco.actuator_dynprm
        newton_actuator_ctrlrange: Newton's model.mujoco.actuator_ctrlrange
        newton_actuator_forcerange: Newton's model.mujoco.actuator_forcerange
        newton_actuator_actrange: Newton's model.mujoco.actuator_actrange
        newton_actuator_gear: Newton's model.mujoco.actuator_gear
        newton_actuator_cranklength: Newton's model.mujoco.actuator_cranklength
        actuators_per_world: Number of actuators per world in Newton model
    """
    world, actuator = wp.tid()
    source = mjc_actuator_ctrl_source[actuator]

    if source != CTRL_SOURCE_CTRL_DIRECT:
        return

    newton_idx = mjc_actuator_to_newton_idx[actuator]
    if newton_idx < 0:
        return

    world_newton_idx = world * actuators_per_world + newton_idx
    actuator_gain[world, actuator] = newton_actuator_gainprm[world_newton_idx]
    actuator_bias[world, actuator] = newton_actuator_biasprm[world_newton_idx]
    actuator_dynprm[world, actuator] = newton_actuator_dynprm[world_newton_idx]
    actuator_ctrlrange[world, actuator] = newton_actuator_ctrlrange[world_newton_idx]
    actuator_forcerange[world, actuator] = newton_actuator_forcerange[world_newton_idx]
    actuator_actrange[world, actuator] = newton_actuator_actrange[world_newton_idx]
    actuator_gear[world, actuator] = newton_actuator_gear[world_newton_idx]
    actuator_cranklength[world, actuator] = newton_actuator_cranklength[world_newton_idx]


@wp.kernel
def update_dof_properties_kernel(
    mjc_dof_to_newton_dof: wp.array2d[wp.int32],
    newton_dof_to_body: wp.array[wp.int32],
    body_flags: wp.array[wp.int32],
    joint_armature: wp.array[float],
    joint_friction: wp.array[float],
    joint_damping: wp.array[float],
    dof_solimp: wp.array[vec5],
    dof_solref: wp.array[wp.vec2],
    # outputs
    dof_armature: wp.array2d[float],
    dof_frictionloss: wp.array2d[float],
    dof_damping: wp.array2d[float],
    dof_solimp_out: wp.array2d[vec5],
    dof_solref_out: wp.array2d[wp.vec2],
):
    """Update MuJoCo DOF properties from Newton DOF properties.

    Iterates over MuJoCo DOFs [world, dof], looks up Newton DOF,
    and copies armature, friction, damping, solimp, solref.
    Armature updates are skipped for DOFs whose child body is marked kinematic.
    """
    world, mjc_dof = wp.tid()
    newton_dof = mjc_dof_to_newton_dof[world, mjc_dof]
    if newton_dof < 0:
        return

    newton_body = newton_dof_to_body[newton_dof]
    if newton_body < 0 or (body_flags[newton_body] & BodyFlags.KINEMATIC) == 0:
        dof_armature[world, mjc_dof] = joint_armature[newton_dof]
    dof_frictionloss[world, mjc_dof] = joint_friction[newton_dof]
    if joint_damping:
        dof_damping[world, mjc_dof] = joint_damping[newton_dof]
    if dof_solimp:
        dof_solimp_out[world, mjc_dof] = dof_solimp[newton_dof]
    if dof_solref:
        dof_solref_out[world, mjc_dof] = dof_solref[newton_dof]


@wp.kernel
def update_body_properties_kernel(
    mjc_dof_to_newton_dof: wp.array2d[wp.int32],
    newton_dof_to_body: wp.array[wp.int32],
    body_flags: wp.array[wp.int32],
    joint_armature: wp.array[float],
    kinematic_armature: float,
    # outputs
    dof_armature: wp.array2d[float],
):
    """Update MuJoCo dof_armature from Newton body flags.

    For each MuJoCo DOF, the mapped Newton child body controls armature source:
    - kinematic body -> ``kinematic_armature``
    - dynamic body   -> Newton ``joint_armature``
    """
    world, mjc_dof = wp.tid()
    newton_dof = mjc_dof_to_newton_dof[world, mjc_dof]
    if newton_dof < 0:
        return

    newton_body = newton_dof_to_body[newton_dof]
    if newton_body >= 0 and (body_flags[newton_body] & BodyFlags.KINEMATIC) != 0:
        dof_armature[world, mjc_dof] = kinematic_armature
    else:
        dof_armature[world, mjc_dof] = joint_armature[newton_dof]


@wp.kernel
def update_jnt_properties_kernel(
    mjc_jnt_to_newton_dof: wp.array2d[wp.int32],
    joint_limit_ke: wp.array[float],
    joint_limit_kd: wp.array[float],
    joint_limit_lower: wp.array[float],
    joint_limit_upper: wp.array[float],
    joint_effort_limit: wp.array[float],
    solimplimit: wp.array[vec5],
    joint_stiffness: wp.array[float],
    limit_margin: wp.array[float],
    # outputs
    jnt_solimp: wp.array2d[vec5],
    jnt_solref: wp.array2d[wp.vec2],
    jnt_stiffness: wp.array2d[float],
    jnt_margin: wp.array2d[float],
    jnt_range: wp.array2d[wp.vec2],
    jnt_actfrcrange: wp.array2d[wp.vec2],
):
    """Update MuJoCo joint properties from Newton DOF properties.

    Iterates over MuJoCo joints [world, jnt], looks up Newton DOF,
    and copies joint-level properties (limits, stiffness, solref, solimp).
    """
    world, mjc_jnt = wp.tid()
    newton_dof = mjc_jnt_to_newton_dof[world, mjc_jnt]
    if newton_dof < 0:
        return

    # Update joint limit solref using negative convention
    if joint_limit_ke[newton_dof] > 0.0:
        jnt_solref[world, mjc_jnt] = wp.vec2(-joint_limit_ke[newton_dof], -joint_limit_kd[newton_dof])

    # Update solimplimit
    if solimplimit:
        jnt_solimp[world, mjc_jnt] = solimplimit[newton_dof]

    # Update passive stiffness
    if joint_stiffness:
        jnt_stiffness[world, mjc_jnt] = joint_stiffness[newton_dof]

    # Update limit margin
    if limit_margin:
        jnt_margin[world, mjc_jnt] = limit_margin[newton_dof]

    # Update joint range
    jnt_range[world, mjc_jnt] = wp.vec2(joint_limit_lower[newton_dof], joint_limit_upper[newton_dof])
    # update joint actuator force range (effort limit)
    effort_limit = joint_effort_limit[newton_dof]
    jnt_actfrcrange[world, mjc_jnt] = wp.vec2(-effort_limit, effort_limit)


@wp.kernel
def update_mocap_transforms_kernel(
    mjc_mocap_to_newton_jnt: wp.array2d[wp.int32],
    newton_joint_X_p: wp.array[wp.transform],
    newton_joint_X_c: wp.array[wp.transform],
    # outputs
    mocap_pos: wp.array2d[wp.vec3],
    mocap_quat: wp.array2d[wp.quat],
):
    """Update MuJoCo mocap body transforms from Newton joint data.

    Iterates over MuJoCo mocap bodies [world, mocap_idx]. Each mocap body maps
    to a fixed Newton joint and is updated from ``joint_X_p * inv(joint_X_c)``.
    """
    world, mocap_idx = wp.tid()
    newton_jnt = mjc_mocap_to_newton_jnt[world, mocap_idx]
    if newton_jnt < 0:
        return

    parent_xform = newton_joint_X_p[newton_jnt]
    child_xform = newton_joint_X_c[newton_jnt]
    tf = parent_xform * wp.transform_inverse(child_xform)

    mocap_pos[world, mocap_idx] = tf.p
    mocap_quat[world, mocap_idx] = quat_xyzw_to_wxyz(tf.q)


@wp.kernel
def update_joint_transforms_kernel(
    mjc_jnt_to_newton_jnt: wp.array2d[wp.int32],
    mjc_jnt_to_newton_dof: wp.array2d[wp.int32],
    mjc_jnt_bodyid: wp.array[wp.int32],
    mjc_jnt_type: wp.array[wp.int32],
    # Newton model data (joint-indexed)
    newton_joint_X_p: wp.array[wp.transform],
    newton_joint_X_c: wp.array[wp.transform],
    # Newton model data (DOF-indexed)
    newton_joint_axis: wp.array[wp.vec3],
    # outputs
    jnt_pos: wp.array2d[wp.vec3],
    jnt_axis: wp.array2d[wp.vec3],
    body_pos: wp.array2d[wp.vec3],
    body_quat: wp.array2d[wp.quat],
):
    """Update MuJoCo joint transforms and body positions from Newton joint data.

    Iterates over MuJoCo joints [world, jnt]. For each joint:
    - Updates MuJoCo body_pos/body_quat from Newton joint transforms
    - Updates MuJoCo jnt_pos and jnt_axis

    Free joints are skipped because their motion is encoded directly in qpos/qvel.
    """
    world, mjc_jnt = wp.tid()

    # Get the Newton joint index for this MuJoCo joint (for joint-indexed arrays)
    newton_jnt = mjc_jnt_to_newton_jnt[world, mjc_jnt]
    if newton_jnt < 0:
        return

    # Get the Newton DOF for this MuJoCo joint (for DOF-indexed arrays like axis)
    newton_dof = mjc_jnt_to_newton_dof[world, mjc_jnt]

    # Skip free joints
    jtype = mjc_jnt_type[mjc_jnt]
    if jtype == 0:  # mjJNT_FREE
        return

    # Get transforms from Newton (indexed by Newton joint)
    child_xform = newton_joint_X_c[newton_jnt]
    parent_xform = newton_joint_X_p[newton_jnt]

    # Update body pos and quat from parent joint transform
    tf = parent_xform * wp.transform_inverse(child_xform)

    # Get the MuJoCo body for this joint and update its transform
    mjc_body = mjc_jnt_bodyid[mjc_jnt]
    body_pos[world, mjc_body] = tf.p
    body_quat[world, mjc_body] = quat_xyzw_to_wxyz(tf.q)

    # Update joint axis and position (DOF-indexed for axis)
    if newton_dof >= 0:
        axis = newton_joint_axis[newton_dof]
        jnt_axis[world, mjc_jnt] = wp.quat_rotate(child_xform.q, axis)
    jnt_pos[world, mjc_jnt] = child_xform.p


@wp.kernel(enable_backward=False)
def update_shape_mappings_kernel(
    geom_to_shape_idx: wp.array[wp.int32],
    geom_is_static: wp.array[bool],
    shape_range_len: int,
    first_env_shape_base: int,
    # output - MuJoCo[world, geom] -> Newton shape
    mjc_geom_to_newton_shape: wp.array2d[wp.int32],
):
    """
    Build the mapping from MuJoCo [world, geom] to Newton shape index.
    This is the primary mapping direction for the new unified design.
    """
    world, geom_idx = wp.tid()
    template_or_static_idx = geom_to_shape_idx[geom_idx]
    if template_or_static_idx < 0:
        return

    # Check if this is a static shape using the precomputed mask
    # For static shapes, template_or_static_idx is the absolute Newton shape index
    # For non-static shapes, template_or_static_idx is 0-based offset from first env's first shape
    is_static = geom_is_static[geom_idx]

    if is_static:
        # Static shape - use absolute index (same for all worlds)
        newton_shape_idx = template_or_static_idx
    else:
        # Non-static shape - compute the absolute Newton shape index for this world
        # template_or_static_idx is 0-based offset within first_group shapes
        newton_shape_idx = first_env_shape_base + template_or_static_idx + world * shape_range_len

    mjc_geom_to_newton_shape[world, geom_idx] = newton_shape_idx


@wp.kernel
def update_model_properties_kernel(
    # Newton model properties
    gravity_src: wp.array[wp.vec3],
    # MuJoCo model properties
    gravity_dst: wp.array[wp.vec3f],
):
    world_idx = wp.tid()
    gravity_dst[world_idx] = gravity_src[world_idx]


@wp.kernel
def update_geom_properties_kernel(
    shape_mu: wp.array[float],
    shape_ke: wp.array[float],
    shape_kd: wp.array[float],
    shape_size: wp.array[wp.vec3f],
    shape_transform: wp.array[wp.transform],
    mjc_geom_to_newton_shape: wp.array2d[wp.int32],
    geom_type: wp.array[int],
    GEOM_TYPE_MESH: int,
    geom_dataid: wp.array2d[int],
    mesh_pos: wp.array[wp.vec3],
    mesh_quat: wp.array[wp.quat],
    shape_mu_torsional: wp.array[float],
    shape_mu_rolling: wp.array[float],
    shape_geom_solimp: wp.array[vec5],
    shape_geom_solmix: wp.array[float],
    shape_margin: wp.array[float],
    zero_margin: int,
    # outputs
    geom_friction: wp.array2d[wp.vec3f],
    geom_solref: wp.array2d[wp.vec2f],
    geom_size: wp.array2d[wp.vec3f],
    geom_pos: wp.array2d[wp.vec3f],
    geom_quat: wp.array2d[wp.quatf],
    geom_solimp: wp.array2d[vec5],
    geom_solmix: wp.array2d[float],
    geom_gap: wp.array2d[float],
    geom_margin: wp.array2d[float],
):
    """Update MuJoCo geom properties from Newton shape properties.

    Iterates over MuJoCo geoms [world, geom], looks up Newton shape index,
    and copies shape properties to geom properties.

    Note: geom_rbound (collision radius) is not updated here. MuJoCo computes
    this internally based on the geometry, and Newton's shape_collision_radius
    is not compatible with MuJoCo's bounding sphere calculation.

    Note: geom_gap is always set to 0 because Newton does not use MuJoCo's
    gap concept.  geom_margin is zeroed when MuJoCo handles collisions
    because mujoco_warp's NATIVECCD broadphase rejects non-zero margins at
    put_model() time (#2106).  When Newton provides contacts, margins are
    restored from shape_margin so that ``convert_newton_contacts_to_mjwarp_kernel``
    can compute correct ``includemargin`` thresholds via ``contact_params``.
    """
    world, geom_idx = wp.tid()

    shape_idx = mjc_geom_to_newton_shape[world, geom_idx]
    if shape_idx < 0:
        return

    # update friction (slide, torsion, roll)
    mu = shape_mu[shape_idx]
    torsional = shape_mu_torsional[shape_idx]
    rolling = shape_mu_rolling[shape_idx]
    geom_friction[world, geom_idx] = wp.vec3f(mu, torsional, rolling)

    # update geom_solref (timeconst, dampratio) using stiffness and damping
    # we don't use the negative convention to support controlling the mixing of shapes' stiffnesses via solmix
    # use approximation of d(0) = d(width) = 1
    geom_solref[world, geom_idx] = convert_solref(shape_ke[shape_idx], shape_kd[shape_idx], 1.0, 1.0)

    # update geom_solimp from custom attribute
    if shape_geom_solimp:
        geom_solimp[world, geom_idx] = shape_geom_solimp[shape_idx]

    # update geom_solmix from custom attribute
    if shape_geom_solmix:
        geom_solmix[world, geom_idx] = shape_geom_solmix[shape_idx]

    geom_gap[world, geom_idx] = 0.0
    if zero_margin:
        geom_margin[world, geom_idx] = 0.0
    else:
        geom_margin[world, geom_idx] = shape_margin[shape_idx]

    # update size
    geom_size[world, geom_idx] = shape_size[shape_idx]

    # update position and orientation

    # get shape transform
    tf = shape_transform[shape_idx]

    # check if this is a mesh geom and apply mesh transformation
    if geom_type[geom_idx] == GEOM_TYPE_MESH:
        mesh_id = geom_dataid[world, geom_idx]
        mesh_p = mesh_pos[mesh_id]
        mesh_q = mesh_quat[mesh_id]
        mesh_tf = wp.transform(mesh_p, quat_wxyz_to_xyzw(mesh_q))
        tf = tf * mesh_tf

    # store position and orientation
    geom_pos[world, geom_idx] = tf.p
    geom_quat[world, geom_idx] = quat_xyzw_to_wxyz(tf.q)


@wp.kernel(enable_backward=False)
def create_inverse_shape_mapping_kernel(
    mjc_geom_to_newton_shape: wp.array2d[wp.int32],
    # output
    newton_shape_to_mjc_geom: wp.array[wp.int32],
):
    """
    Create partial inverse mapping from Newton shape index to MuJoCo geom index.

    Note: The full inverse mapping (Newton [shape] -> MuJoCo [world, geom]) is not possible because
    shape-to-geom is one-to-many: the same global Newton shape maps to one MuJoCo geom in every
    world. This kernel only stores the geom index; world ID is computed from body indices
    in the contact conversion kernel.
    """
    world, geom_idx = wp.tid()
    newton_shape_idx = mjc_geom_to_newton_shape[world, geom_idx]
    if newton_shape_idx >= 0:
        newton_shape_to_mjc_geom[newton_shape_idx] = geom_idx


@wp.kernel
def update_eq_properties_kernel(
    mjc_eq_to_newton_eq: wp.array2d[wp.int32],
    eq_solref: wp.array[wp.vec2],
    eq_solimp: wp.array[vec5],
    # outputs
    eq_solref_out: wp.array2d[wp.vec2],
    eq_solimp_out: wp.array2d[vec5],
):
    """Update MuJoCo equality constraint properties from Newton equality constraint properties.

    Iterates over MuJoCo equality constraints [world, eq], looks up Newton eq constraint,
    and copies solref and solimp.
    """
    world, mjc_eq = wp.tid()
    newton_eq = mjc_eq_to_newton_eq[world, mjc_eq]
    if newton_eq < 0:
        return

    if eq_solref:
        eq_solref_out[world, mjc_eq] = eq_solref[newton_eq]

    if eq_solimp:
        eq_solimp_out[world, mjc_eq] = eq_solimp[newton_eq]


@wp.kernel
def update_tendon_properties_kernel(
    mjc_tendon_to_newton_tendon: wp.array2d[wp.int32],
    # Newton tendon properties (inputs)
    tendon_stiffness: wp.array[wp.float32],
    tendon_damping: wp.array[wp.float32],
    tendon_frictionloss: wp.array[wp.float32],
    tendon_range: wp.array[wp.vec2],
    tendon_margin: wp.array[wp.float32],
    tendon_solref_limit: wp.array[wp.vec2],
    tendon_solimp_limit: wp.array[vec5],
    tendon_solref_friction: wp.array[wp.vec2],
    tendon_solimp_friction: wp.array[vec5],
    tendon_armature: wp.array[wp.float32],
    tendon_actfrcrange: wp.array[wp.vec2],
    # MuJoCo tendon properties (outputs)
    tendon_stiffness_out: wp.array2d[wp.float32],
    tendon_damping_out: wp.array2d[wp.float32],
    tendon_frictionloss_out: wp.array2d[wp.float32],
    tendon_range_out: wp.array2d[wp.vec2],
    tendon_margin_out: wp.array2d[wp.float32],
    tendon_solref_lim_out: wp.array2d[wp.vec2],
    tendon_solimp_lim_out: wp.array2d[vec5],
    tendon_solref_fri_out: wp.array2d[wp.vec2],
    tendon_solimp_fri_out: wp.array2d[vec5],
    tendon_armature_out: wp.array2d[wp.float32],
    tendon_actfrcrange_out: wp.array2d[wp.vec2],
):
    """Update MuJoCo tendon properties from Newton tendon custom attributes.

    Iterates over MuJoCo tendons [world, tendon], looks up Newton tendon,
    and copies properties.

    Note: tendon_lengthspring is NOT updated at runtime because it has special
    initialization semantics in MuJoCo (value -1.0 means auto-compute from initial state).
    """
    world, mjc_tendon = wp.tid()
    newton_tendon = mjc_tendon_to_newton_tendon[world, mjc_tendon]
    if newton_tendon < 0:
        return

    if tendon_stiffness:
        tendon_stiffness_out[world, mjc_tendon] = tendon_stiffness[newton_tendon]
    if tendon_damping:
        tendon_damping_out[world, mjc_tendon] = tendon_damping[newton_tendon]
    if tendon_frictionloss:
        tendon_frictionloss_out[world, mjc_tendon] = tendon_frictionloss[newton_tendon]
    if tendon_range:
        tendon_range_out[world, mjc_tendon] = tendon_range[newton_tendon]
    if tendon_margin:
        tendon_margin_out[world, mjc_tendon] = tendon_margin[newton_tendon]
    if tendon_solref_limit:
        tendon_solref_lim_out[world, mjc_tendon] = tendon_solref_limit[newton_tendon]
    if tendon_solimp_limit:
        tendon_solimp_lim_out[world, mjc_tendon] = tendon_solimp_limit[newton_tendon]
    if tendon_solref_friction:
        tendon_solref_fri_out[world, mjc_tendon] = tendon_solref_friction[newton_tendon]
    if tendon_solimp_friction:
        tendon_solimp_fri_out[world, mjc_tendon] = tendon_solimp_friction[newton_tendon]
    if tendon_armature:
        tendon_armature_out[world, mjc_tendon] = tendon_armature[newton_tendon]
    if tendon_actfrcrange:
        tendon_actfrcrange_out[world, mjc_tendon] = tendon_actfrcrange[newton_tendon]


@wp.kernel
def update_eq_data_and_active_kernel(
    mjc_eq_to_newton_eq: wp.array2d[wp.int32],
    # Newton equality constraint data
    eq_constraint_type: wp.array[wp.int32],
    eq_constraint_anchor: wp.array[wp.vec3],
    eq_constraint_relpose: wp.array[wp.transform],
    eq_constraint_polycoef: wp.array2d[wp.float32],
    eq_constraint_torquescale: wp.array[wp.float32],
    eq_constraint_enabled: wp.array[wp.bool],
    # outputs
    eq_data_out: wp.array2d[vec11],
    eq_active_out: wp.array2d[wp.bool],
):
    """Update MuJoCo equality constraint data and active status from Newton properties.

    Iterates over MuJoCo equality constraints [world, eq], looks up Newton eq constraint,
    and copies:
    - eq_data based on constraint type:
      - CONNECT: data[0:3] = anchor
      - JOINT: data[0:5] = polycoef
      - WELD: data[0:3] = anchor, data[3:6] = relpose translation, data[6:10] = relpose quaternion, data[10] = torquescale
    - eq_active from equality_constraint_enabled
    """
    world, mjc_eq = wp.tid()
    newton_eq = mjc_eq_to_newton_eq[world, mjc_eq]
    if newton_eq < 0:
        return

    constraint_type = eq_constraint_type[newton_eq]

    # Read existing data to preserve fields we don't update
    data = eq_data_out[world, mjc_eq]

    if constraint_type == EqType.CONNECT:
        # CONNECT: data[0:3] = anchor
        anchor = eq_constraint_anchor[newton_eq]
        data[0] = anchor[0]
        data[1] = anchor[1]
        data[2] = anchor[2]

    elif constraint_type == EqType.JOINT:
        # JOINT: data[0:5] = polycoef
        for i in range(5):
            data[i] = eq_constraint_polycoef[newton_eq, i]

    elif constraint_type == EqType.WELD:
        # WELD: data[0:3] = anchor
        anchor = eq_constraint_anchor[newton_eq]
        data[0] = anchor[0]
        data[1] = anchor[1]
        data[2] = anchor[2]

        # data[3:6] = relpose translation
        relpose = eq_constraint_relpose[newton_eq]
        pos = wp.transform_get_translation(relpose)
        data[3] = pos[0]
        data[4] = pos[1]
        data[5] = pos[2]

        # data[6:10] = relpose quaternion in MuJoCo order (wxyz)
        quat = quat_xyzw_to_wxyz(wp.transform_get_rotation(relpose))
        for i in range(4):
            data[6 + i] = quat[i]

        # data[10] = torquescale
        data[10] = eq_constraint_torquescale[newton_eq]

    eq_data_out[world, mjc_eq] = data
    eq_active_out[world, mjc_eq] = eq_constraint_enabled[newton_eq]


@wp.kernel
def update_mimic_eq_data_and_active_kernel(
    mjc_eq_to_newton_mimic: wp.array2d[wp.int32],
    # Newton mimic constraint data
    constraint_mimic_coef0: wp.array[wp.float32],
    constraint_mimic_coef1: wp.array[wp.float32],
    constraint_mimic_enabled: wp.array[wp.bool],
    # outputs
    eq_data_out: wp.array2d[vec11],
    eq_active_out: wp.array2d[wp.bool],
):
    """Update MuJoCo equality constraint data and active status from Newton mimic constraint properties.

    Iterates over MuJoCo equality constraints [world, eq], looks up Newton mimic constraint,
    and copies:
    - eq_data: polycoef = [coef0, coef1, 0, 0, 0] (linear mimic relationship)
    - eq_active from constraint_mimic_enabled
    """
    world, mjc_eq = wp.tid()
    newton_mimic = mjc_eq_to_newton_mimic[world, mjc_eq]
    if newton_mimic < 0:
        return

    data = eq_data_out[world, mjc_eq]

    # polycoef: data[0] + data[1]*q2 - q1 = 0  =>  q1 = coef0 + coef1*q2
    data[0] = constraint_mimic_coef0[newton_mimic]
    data[1] = constraint_mimic_coef1[newton_mimic]
    data[2] = 0.0
    data[3] = 0.0
    data[4] = 0.0

    eq_data_out[world, mjc_eq] = data
    eq_active_out[world, mjc_eq] = constraint_mimic_enabled[newton_mimic]


@wp.func
def mj_body_acceleration(
    body_rootid: wp.array[int],
    xipos_in: wp.array2d[wp.vec3],
    subtree_com_in: wp.array2d[wp.vec3],
    cvel_in: wp.array2d[wp.spatial_vector],
    cacc_in: wp.array2d[wp.spatial_vector],
    worldid: int,
    bodyid: int,
) -> wp.vec3:
    """Compute accelerations for bodies from mjwarp data."""
    cacc = cacc_in[worldid, bodyid]
    cvel = cvel_in[worldid, bodyid]
    offset = xipos_in[worldid, bodyid] - subtree_com_in[worldid, body_rootid[bodyid]]
    ang = wp.spatial_top(cvel)
    lin = wp.spatial_bottom(cvel) - wp.cross(offset, ang)
    acc = wp.spatial_bottom(cacc) - wp.cross(offset, wp.spatial_top(cacc))
    correction = wp.cross(ang, lin)

    return acc + correction


@wp.kernel
def convert_rigid_forces_from_mj_kernel(
    mjc_body_to_newton: wp.array2d[wp.int32],
    # mjw sources
    mjw_body_rootid: wp.array[wp.int32],
    mjw_gravity: wp.array[wp.vec3],
    mjw_xipos: wp.array2d[wp.vec3],
    mjw_subtree_com: wp.array2d[wp.vec3],
    mjw_cacc: wp.array2d[wp.spatial_vector],
    mjw_cvel: wp.array2d[wp.spatial_vector],
    mjw_cint: wp.array2d[wp.spatial_vector],
    # outputs
    body_qdd: wp.array[wp.spatial_vector],
    body_parent_f: wp.array[wp.spatial_vector],
):
    """Update RNE-computed rigid forces from mj_warp com-based forces."""
    world, mjc_body = wp.tid()
    newton_body = mjc_body_to_newton[world, mjc_body]

    if newton_body < 0:
        return

    if body_qdd:
        cacc = mjw_cacc[world, mjc_body]
        qdd_lin = mj_body_acceleration(
            mjw_body_rootid,
            mjw_xipos,
            mjw_subtree_com,
            mjw_cvel,
            mjw_cacc,
            world,
            mjc_body,
        )
        body_qdd[newton_body] = wp.spatial_vector(qdd_lin + mjw_gravity[world], wp.spatial_top(cacc))

    if body_parent_f:
        cint = mjw_cint[world, mjc_body]
        parent_f_ang = wp.spatial_top(cint)
        parent_f_lin = wp.spatial_bottom(cint)

        offset = mjw_xipos[world, mjc_body] - mjw_subtree_com[world, mjw_body_rootid[mjc_body]]

        body_parent_f[newton_body] = wp.spatial_vector(parent_f_lin, parent_f_ang - wp.cross(offset, parent_f_lin))


@wp.kernel
def convert_qfrc_actuator_from_mj_kernel(
    mjw_qfrc_actuator: wp.array2d[wp.float32],
    qpos: wp.array2d[wp.float32],
    joints_per_world: int,
    joint_type: wp.array[wp.int32],
    joint_q_start: wp.array[wp.int32],
    joint_qd_start: wp.array[wp.int32],
    joint_dof_dim: wp.array2d[wp.int32],
    joint_child: wp.array[wp.int32],
    body_com: wp.array[wp.vec3],
    mj_q_start: wp.array[wp.int32],
    mj_qd_start: wp.array[wp.int32],
    # output
    qfrc_actuator: wp.array[wp.float32],
):
    """Convert MuJoCo qfrc_actuator [nworld, nv] into Newton flat DOF array.

    Uses the same joint-based DOF mapping as the coordinate conversion
    kernels. For free joints the wrench is transformed from MuJoCo's
    (origin, body-frame) convention to the CoM/world convention used on the
    MuJoCo side of Newton. Ball and other joints are copied directly.
    """
    worldid, jntid = wp.tid()

    joint_id = joints_per_world * worldid + jntid

    # Skip loop joints — they have no MuJoCo DOF entries
    q_i = mj_q_start[jntid]
    if q_i < 0:
        return

    qd_i = mj_qd_start[jntid]
    wqd_i = joint_qd_start[joint_id]

    jtype = joint_type[joint_id]

    if jtype == JointType.FREE:
        # MuJoCo qfrc_actuator for free joint:
        #   [f_x, f_y, f_z] = linear force at body origin (world frame)
        #   [τ_x, τ_y, τ_z] = torque in body frame
        # Newton convention (dual of velocity transform):
        #   f_lin_newton = f_lin_mujoco            (unchanged)
        #   tau_newton    = R * tau_body - r_com x f_lin

        f_lin = wp.vec3(
            mjw_qfrc_actuator[worldid, qd_i + 0],
            mjw_qfrc_actuator[worldid, qd_i + 1],
            mjw_qfrc_actuator[worldid, qd_i + 2],
        )
        tau_body = wp.vec3(
            mjw_qfrc_actuator[worldid, qd_i + 3],
            mjw_qfrc_actuator[worldid, qd_i + 4],
            mjw_qfrc_actuator[worldid, qd_i + 5],
        )

        # Body rotation (MuJoCo quaternion wxyz)
        rot = wp.quat(
            qpos[worldid, q_i + 4],
            qpos[worldid, q_i + 5],
            qpos[worldid, q_i + 6],
            qpos[worldid, q_i + 3],
        )

        # CoM offset in world frame
        child = joint_child[joint_id]
        com_world = wp.quat_rotate(rot, body_com[child])

        # Rotate torque body -> world and shift reference origin -> CoM
        tau_world = wp.quat_rotate(rot, tau_body) - wp.cross(com_world, f_lin)

        qfrc_actuator[wqd_i + 0] = f_lin[0]
        qfrc_actuator[wqd_i + 1] = f_lin[1]
        qfrc_actuator[wqd_i + 2] = f_lin[2]
        qfrc_actuator[wqd_i + 3] = tau_world[0]
        qfrc_actuator[wqd_i + 4] = tau_world[1]
        qfrc_actuator[wqd_i + 5] = tau_world[2]
    elif jtype == JointType.BALL:
        for i in range(3):
            qfrc_actuator[wqd_i + i] = mjw_qfrc_actuator[worldid, qd_i + i]
    else:
        axis_count = joint_dof_dim[joint_id, 0] + joint_dof_dim[joint_id, 1]
        for i in range(axis_count):
            qfrc_actuator[wqd_i + i] = mjw_qfrc_actuator[worldid, qd_i + i]


@wp.kernel
def update_pair_properties_kernel(
    pairs_per_world: int,
    pair_solref_in: wp.array[wp.vec2],
    pair_solreffriction_in: wp.array[wp.vec2],
    pair_solimp_in: wp.array[vec5],
    pair_margin_in: wp.array[float],
    pair_gap_in: wp.array[float],
    pair_friction_in: wp.array[vec5],
    # outputs
    pair_solref_out: wp.array2d[wp.vec2],
    pair_solreffriction_out: wp.array2d[wp.vec2],
    pair_solimp_out: wp.array2d[vec5],
    pair_margin_out: wp.array2d[float],
    pair_gap_out: wp.array2d[float],
    pair_friction_out: wp.array2d[vec5],
):
    """Update MuJoCo contact pair properties from Newton custom attributes.

    Iterates over MuJoCo pairs [world, pair] and copies solver properties
    (solref, solimp, margin, gap, friction) from Newton custom attributes.
    """
    world, mjc_pair = wp.tid()
    newton_pair = world * pairs_per_world + mjc_pair

    if pair_solref_in:
        pair_solref_out[world, mjc_pair] = pair_solref_in[newton_pair]

    if pair_solreffriction_in:
        pair_solreffriction_out[world, mjc_pair] = pair_solreffriction_in[newton_pair]

    if pair_solimp_in:
        pair_solimp_out[world, mjc_pair] = pair_solimp_in[newton_pair]

    if pair_margin_in:
        pair_margin_out[world, mjc_pair] = pair_margin_in[newton_pair]

    if pair_gap_in:
        pair_gap_out[world, mjc_pair] = pair_gap_in[newton_pair]

    if pair_friction_in:
        pair_friction_out[world, mjc_pair] = pair_friction_in[newton_pair]
