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

"""Warp kernels for SolverMuJoCo."""

from __future__ import annotations

from typing import Any

import warp as wp

from ...core.types import vec5
from ...sim import JointType

# Custom vector types
vec10 = wp.types.vector(length=10, dtype=wp.float32)


# Constants
MJ_MINVAL = 2.220446049250313e-16


# Utility functions
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
    contact_dist_out: wp.array(dtype=float),
    contact_pos_out: wp.array(dtype=wp.vec3),
    contact_frame_out: wp.array(dtype=wp.mat33),
    contact_includemargin_out: wp.array(dtype=float),
    contact_friction_out: wp.array(dtype=vec5),
    contact_solref_out: wp.array(dtype=wp.vec2),
    contact_solreffriction_out: wp.array(dtype=wp.vec2),
    contact_solimp_out: wp.array(dtype=vec5),
    contact_dim_out: wp.array(dtype=int),
    contact_geom_out: wp.array(dtype=wp.vec2i),
    contact_worldid_out: wp.array(dtype=int),
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


@wp.func
def contact_params(
    geom_condim: wp.array(dtype=int),
    geom_priority: wp.array(dtype=int),
    geom_solmix: wp.array2d(dtype=float),
    geom_solref: wp.array2d(dtype=wp.vec2),
    geom_solimp: wp.array2d(dtype=vec5),
    geom_friction: wp.array2d(dtype=wp.vec3),
    geom_margin: wp.array2d(dtype=float),
    geom_gap: wp.array2d(dtype=float),
    geoms: wp.vec2i,
    worldid: int,
):
    # See function contact_params in mujoco_warp, file collision_primitive.py

    g1 = geoms[0]
    g2 = geoms[1]

    p1 = geom_priority[g1]
    p2 = geom_priority[g2]

    solmix1 = geom_solmix[worldid, g1]
    solmix2 = geom_solmix[worldid, g2]

    mix = solmix1 / (solmix1 + solmix2)
    mix = wp.where((solmix1 < MJ_MINVAL) and (solmix2 < MJ_MINVAL), 0.5, mix)
    mix = wp.where((solmix1 < MJ_MINVAL) and (solmix2 >= MJ_MINVAL), 0.0, mix)
    mix = wp.where((solmix1 >= MJ_MINVAL) and (solmix2 < MJ_MINVAL), 1.0, mix)
    mix = wp.where(p1 == p2, mix, wp.where(p1 > p2, 1.0, 0.0))

    # Sum margins for consistency with thickness summing
    margin = geom_margin[worldid, g1] + geom_margin[worldid, g2]
    gap = geom_gap[worldid, g1] + geom_gap[worldid, g2]

    condim1 = geom_condim[g1]
    condim2 = geom_condim[g2]
    condim = wp.where(p1 == p2, wp.max(condim1, condim2), wp.where(p1 > p2, condim1, condim2))

    max_geom_friction = wp.max(geom_friction[worldid, g1], geom_friction[worldid, g2])
    friction = vec5(
        max_geom_friction[0],
        max_geom_friction[0],
        max_geom_friction[1],
        max_geom_friction[2],
        max_geom_friction[2],
    )

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


# Kernel functions
@wp.kernel
def convert_newton_contacts_to_mjwarp_kernel(
    body_q: wp.array(dtype=wp.transform),
    shape_body: wp.array(dtype=int),
    # Model:
    geom_condim: wp.array(dtype=int),
    geom_priority: wp.array(dtype=int),
    geom_solmix: wp.array2d(dtype=float),
    geom_solref: wp.array2d(dtype=wp.vec2),
    geom_solimp: wp.array2d(dtype=vec5),
    geom_friction: wp.array2d(dtype=wp.vec3),
    geom_margin: wp.array2d(dtype=float),
    geom_gap: wp.array2d(dtype=float),
    # Newton contacts
    rigid_contact_count: wp.array(dtype=wp.int32),
    rigid_contact_shape0: wp.array(dtype=wp.int32),
    rigid_contact_shape1: wp.array(dtype=wp.int32),
    rigid_contact_point0: wp.array(dtype=wp.vec3),
    rigid_contact_point1: wp.array(dtype=wp.vec3),
    rigid_contact_normal: wp.array(dtype=wp.vec3),
    rigid_contact_thickness0: wp.array(dtype=wp.float32),
    rigid_contact_thickness1: wp.array(dtype=wp.float32),
    rigid_contact_stiffness: wp.array(dtype=wp.float32),
    rigid_contact_damping: wp.array(dtype=wp.float32),
    rigid_contact_friction_scale: wp.array(dtype=wp.float32),
    bodies_per_world: int,
    newton_shape_to_mjc_geom: wp.array(dtype=wp.int32),
    # Mujoco warp contacts
    naconmax: int,
    nacon_out: wp.array(dtype=int),
    contact_dist_out: wp.array(dtype=float),
    contact_pos_out: wp.array(dtype=wp.vec3),
    contact_frame_out: wp.array(dtype=wp.mat33),
    contact_includemargin_out: wp.array(dtype=float),
    contact_friction_out: wp.array(dtype=vec5),
    contact_solref_out: wp.array(dtype=wp.vec2),
    contact_solreffriction_out: wp.array(dtype=wp.vec2),
    contact_solimp_out: wp.array(dtype=vec5),
    contact_dim_out: wp.array(dtype=int),
    contact_geom_out: wp.array(dtype=wp.vec2i),
    contact_worldid_out: wp.array(dtype=int),
    # Values to clear - see _zero_collision_arrays kernel from mujoco_warp
    nworld_in: int,
    ncollision_out: wp.array(dtype=int),
):
    # See kernel solve_body_contact_positions for reference

    tid = wp.tid()

    count = rigid_contact_count[0]

    # Set number of contacts (for a single world)
    if tid == 0:
        if count > naconmax:
            wp.printf(
                "Number of Newton contacts (%d) exceeded MJWarp limit (%d). Increase nconmax.\n",
                count,
                naconmax,
            )
            count = naconmax
        nacon_out[0] = count
        ncollision_out[0] = 0

    if count > naconmax:
        count = naconmax

    if tid >= count:
        return

    shape_a = rigid_contact_shape0[tid]
    shape_b = rigid_contact_shape1[tid]

    # Skip invalid contacts - both shapes must be specified
    if shape_a < 0 or shape_b < 0:
        return

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

    thickness = rigid_contact_thickness0[tid] + rigid_contact_thickness1[tid]

    n = -rigid_contact_normal[tid]
    dist = wp.dot(n, bx_b - bx_a) - thickness

    # Contact position: use midpoint between contact points (as in XPBD kernel)
    pos = 0.5 * (bx_a + bx_b)

    # Build contact frame
    frame = make_frame(n)

    geom_a = newton_shape_to_mjc_geom[shape_a]
    geom_b = newton_shape_to_mjc_geom[shape_b]
    geoms = wp.vec2i(geom_a, geom_b)

    # Compute world ID from body indices (more reliable than shape mapping for static shapes)
    # Static shapes like ground planes share the same Newton shape index across all worlds,
    # so the inverse shape mapping may have the wrong world ID for them.
    # Using body indices: body_index = world * bodies_per_world + body_in_world
    # Note: At least one shape must be attached to a body (body >= 0) since collisions
    # between two static shapes (not attached to any body) are not supported.
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

    if rigid_contact_stiffness:
        # Use per-contact stiffness/damping parameters
        contact_ke = rigid_contact_stiffness[tid]
        if contact_ke > 0.0:
            # set solimp to approximate linear force-to-displacement relationship at rest
            # see https://mujoco.readthedocs.io/en/latest/modeling.html#solver-parameters
            imp = solimp[1]
            solimp = vec5(imp, imp, 0.001, 1.0, 0.5)
            contact_ke = contact_ke * (1.0 - imp)  # compensate for impedance scaling
            kd = rigid_contact_damping[tid]
            # convert from stiffness/damping to MuJoCo's solref timeconst and dampratio
            if kd > 0.0:
                timeconst = 2.0 / kd
                dampratio = wp.sqrt(1.0 / (timeconst * timeconst * contact_ke))
            else:
                # if no damping was set, use default damping ratio
                timeconst = wp.sqrt(1.0 / contact_ke)
                dampratio = 1.0

            solref = wp.vec2(timeconst, dampratio)

        friction_scale = rigid_contact_friction_scale[tid]
        if friction_scale > 0.0:
            friction = vec5(
                friction[0] * friction_scale,
                friction[1] * friction_scale,
                friction[2],
                friction[3],
                friction[4],
            )

    # Use the write_contact function to write all the data
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
        contact_id_in=tid,
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
        contact_worldid_out=contact_worldid_out,
    )


@wp.kernel
def convert_mj_coords_to_warp_kernel(
    qpos: wp.array2d(dtype=wp.float32),
    qvel: wp.array2d(dtype=wp.float32),
    joints_per_world: int,
    joint_type: wp.array(dtype=wp.int32),
    joint_q_start: wp.array(dtype=wp.int32),
    joint_qd_start: wp.array(dtype=wp.int32),
    joint_dof_dim: wp.array(dtype=wp.int32, ndim=2),
    joint_child: wp.array(dtype=wp.int32),
    body_com: wp.array(dtype=wp.vec3),
    # outputs
    joint_q: wp.array(dtype=wp.float32),
    joint_qd: wp.array(dtype=wp.float32),
):
    worldid, jntid = wp.tid()

    type = joint_type[jntid]
    q_i = joint_q_start[jntid]
    qd_i = joint_qd_start[jntid]
    wq_i = joint_q_start[joints_per_world * worldid + jntid]
    wqd_i = joint_qd_start[joints_per_world * worldid + jntid]

    if type == JointType.FREE:
        # convert position components
        for i in range(3):
            joint_q[wq_i + i] = qpos[worldid, q_i + i]

        # change quaternion order from wxyz to xyzw
        rot = wp.quat(
            qpos[worldid, q_i + 4],
            qpos[worldid, q_i + 5],
            qpos[worldid, q_i + 6],
            qpos[worldid, q_i + 3],
        )
        joint_q[wq_i + 3] = rot[0]
        joint_q[wq_i + 4] = rot[1]
        joint_q[wq_i + 5] = rot[2]
        joint_q[wq_i + 6] = rot[3]

        # MuJoCo qvel: linear velocity of body ORIGIN (world frame), angular velocity (body frame)
        # Newton joint_qd: linear velocity of CoM (world frame), angular velocity (world frame)
        #
        # Relationship: v_com = v_origin + ω x com_offset_world
        # where com_offset_world = quat_rotate(body_rotation, body_com)

        # Get angular velocity in body frame from MuJoCo and convert to world frame
        w_body = wp.vec3(qvel[worldid, qd_i + 3], qvel[worldid, qd_i + 4], qvel[worldid, qd_i + 5])
        w_world = wp.quat_rotate(rot, w_body)

        # Get CoM offset in world frame
        child = joint_child[jntid]
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
        rot = wp.quat(
            qpos[worldid, q_i + 1],
            qpos[worldid, q_i + 2],
            qpos[worldid, q_i + 3],
            qpos[worldid, q_i],
        )
        joint_q[wq_i] = rot[0]
        joint_q[wq_i + 1] = rot[1]
        joint_q[wq_i + 2] = rot[2]
        joint_q[wq_i + 3] = rot[3]
        for i in range(3):
            # convert velocity components
            joint_qd[wqd_i + i] = qvel[worldid, qd_i + i]
    else:
        axis_count = joint_dof_dim[jntid, 0] + joint_dof_dim[jntid, 1]
        for i in range(axis_count):
            # convert position components
            joint_q[wq_i + i] = qpos[worldid, q_i + i]
        for i in range(axis_count):
            # convert velocity components
            joint_qd[wqd_i + i] = qvel[worldid, qd_i + i]


@wp.kernel
def convert_warp_coords_to_mj_kernel(
    joint_q: wp.array(dtype=wp.float32),
    joint_qd: wp.array(dtype=wp.float32),
    joints_per_world: int,
    joint_type: wp.array(dtype=wp.int32),
    joint_q_start: wp.array(dtype=wp.int32),
    joint_qd_start: wp.array(dtype=wp.int32),
    joint_dof_dim: wp.array(dtype=wp.int32, ndim=2),
    joint_child: wp.array(dtype=wp.int32),
    body_com: wp.array(dtype=wp.vec3),
    # outputs
    qpos: wp.array2d(dtype=wp.float32),
    qvel: wp.array2d(dtype=wp.float32),
):
    worldid, jntid = wp.tid()

    type = joint_type[jntid]
    q_i = joint_q_start[jntid]
    qd_i = joint_qd_start[jntid]
    wq_i = joint_q_start[joints_per_world * worldid + jntid]
    wqd_i = joint_qd_start[joints_per_world * worldid + jntid]

    if type == JointType.FREE:
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
        qpos[worldid, q_i + 3] = rot[3]
        qpos[worldid, q_i + 4] = rot[0]
        qpos[worldid, q_i + 5] = rot[1]
        qpos[worldid, q_i + 6] = rot[2]

        # Newton joint_qd: linear velocity of CoM (world frame), angular velocity (world frame)
        # MuJoCo qvel: linear velocity of body ORIGIN (world frame), angular velocity (body frame)
        #
        # Relationship: v_origin = v_com - ω x com_offset_world
        # where com_offset_world = quat_rotate(body_rotation, body_com)

        # Get angular velocity in world frame
        w_world = wp.vec3(joint_qd[wqd_i + 3], joint_qd[wqd_i + 4], joint_qd[wqd_i + 5])

        # Get CoM offset in world frame
        child = joint_child[jntid]
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

    elif type == JointType.BALL:
        # change quaternion order from xyzw to wxyz
        qpos[worldid, q_i + 0] = joint_q[wq_i + 3]
        qpos[worldid, q_i + 1] = joint_q[wq_i + 0]
        qpos[worldid, q_i + 2] = joint_q[wq_i + 1]
        qpos[worldid, q_i + 3] = joint_q[wq_i + 2]
        for i in range(3):
            # convert velocity components
            qvel[worldid, qd_i + i] = joint_qd[wqd_i + i]
    else:
        axis_count = joint_dof_dim[jntid, 0] + joint_dof_dim[jntid, 1]
        for i in range(axis_count):
            # convert position components
            qpos[worldid, q_i + i] = joint_q[wq_i + i]
        for i in range(axis_count):
            # convert velocity components
            qvel[worldid, qd_i + i] = joint_qd[wqd_i + i]


@wp.kernel
def convert_mjw_contact_to_warp_kernel(
    # inputs
    mjc_geom_to_newton_shape: wp.array2d(dtype=wp.int32),
    pyramidal_cone: bool,
    mj_nacon: wp.array(dtype=wp.int32),
    mj_contact_frame: wp.array(dtype=wp.mat33f),
    mj_contact_dim: wp.array(dtype=int),
    mj_contact_geom: wp.array(dtype=wp.vec2i),
    mj_contact_efc_address: wp.array2d(dtype=int),
    mj_contact_worldid: wp.array(dtype=wp.int32),
    mj_efc_force: wp.array2d(dtype=float),
    # outputs
    contact_pair: wp.array(dtype=wp.vec2i),
    contact_normal: wp.array(dtype=wp.vec3f),
    contact_force: wp.array(dtype=float),
):
    """Convert MuJoCo contacts to Newton contact format.

    Uses mjc_geom_to_newton_shape to convert MuJoCo geom indices to Newton shape indices.
    """
    n_contacts = mj_nacon[0]
    contact_idx = wp.tid()

    if contact_idx >= n_contacts:
        return

    world = mj_contact_worldid[contact_idx]
    geoms_mjw = mj_contact_geom[contact_idx]

    normalforce = wp.float(-1.0)

    efc_address0 = mj_contact_efc_address[contact_idx, 0]
    if efc_address0 >= 0:
        normalforce = mj_efc_force[world, efc_address0]

        if pyramidal_cone:
            dim = mj_contact_dim[contact_idx]
            for i in range(1, 2 * (dim - 1)):
                normalforce += mj_efc_force[world, mj_contact_efc_address[contact_idx, i]]

    pair = wp.vec2i()
    for i in range(2):
        pair[i] = mjc_geom_to_newton_shape[world, geoms_mjw[i]]
    contact_pair[contact_idx] = pair
    contact_normal[contact_idx] = wp.transpose(mj_contact_frame[contact_idx])[0]
    contact_force[contact_idx] = wp.where(normalforce > 0.0, normalforce, 0.0)


@wp.kernel
def apply_mjc_control_kernel(
    mjc_actuator_to_newton_axis: wp.array2d(dtype=wp.int32),
    joint_target_pos: wp.array(dtype=wp.float32),
    joint_target_vel: wp.array(dtype=wp.float32),
    # outputs
    mj_ctrl: wp.array2d(dtype=wp.float32),
):
    """Apply Newton joint targets to MuJoCo control array.

    Iterates over MuJoCo actuators [world, actuator], looks up Newton axis,
    and copies position or velocity target based on actuator type.

    The actuator type is encoded in the sign of mjc_actuator_to_newton_axis:
    - Positive value: position actuator, newton_axis = value
    - Negative value: velocity actuator, newton_axis = -(value + 2)
    """
    world, mjc_actuator = wp.tid()
    raw_value = mjc_actuator_to_newton_axis[world, mjc_actuator]
    if raw_value >= 0:
        # Position actuator
        newton_axis = raw_value
        mj_ctrl[world, mjc_actuator] = joint_target_pos[newton_axis]
    elif raw_value != -1:  # raw_value == -1 means unmapped
        # Velocity actuator
        newton_axis = -raw_value - 2  # Decode: -(newton_axis + 2) -> newton_axis
        mj_ctrl[world, mjc_actuator] = joint_target_vel[newton_axis]


@wp.kernel
def apply_mjc_body_f_kernel(
    mjc_body_to_newton: wp.array2d(dtype=wp.int32),
    body_f: wp.array(dtype=wp.spatial_vector),
    # outputs
    xfrc_applied: wp.array2d(dtype=wp.spatial_vector),
):
    """Apply Newton body forces to MuJoCo xfrc_applied array.

    Iterates over MuJoCo bodies [world, mjc_body], looks up Newton body index,
    and copies the force.
    """
    world, mjc_body = wp.tid()
    newton_body = mjc_body_to_newton[world, mjc_body]
    if newton_body >= 0:
        f = body_f[newton_body]
        v = wp.vec3(f[0], f[1], f[2])
        w = wp.vec3(f[3], f[4], f[5])
        xfrc_applied[world, mjc_body] = wp.spatial_vector(v, w)


@wp.kernel
def apply_mjc_qfrc_kernel(
    body_q: wp.array(dtype=wp.transform),
    joint_f: wp.array(dtype=wp.float32),
    joint_type: wp.array(dtype=wp.int32),
    body_com: wp.array(dtype=wp.vec3),
    joint_child: wp.array(dtype=wp.int32),
    joint_q_start: wp.array(dtype=wp.int32),
    joint_qd_start: wp.array(dtype=wp.int32),
    joint_dof_dim: wp.array2d(dtype=wp.int32),
    joints_per_world: int,
    bodies_per_world: int,
    # outputs
    qfrc_applied: wp.array2d(dtype=wp.float32),
):
    worldid, jntid = wp.tid()
    child = joint_child[jntid]
    # q_i = joint_q_start[jntid]
    qd_i = joint_qd_start[jntid]
    # wq_i = joint_q_start[joints_per_world * worldid + jntid]
    wqd_i = joint_qd_start[joints_per_world * worldid + jntid]
    jtype = joint_type[jntid]
    if jtype == JointType.FREE or jtype == JointType.DISTANCE:
        tf = body_q[worldid * bodies_per_world + child]
        rot = wp.transform_get_rotation(tf)
        # com_world = wp.transform_point(tf, body_com[child])
        v = wp.vec3(joint_f[wqd_i + 0], joint_f[wqd_i + 1], joint_f[wqd_i + 2])
        w = wp.vec3(joint_f[wqd_i + 3], joint_f[wqd_i + 4], joint_f[wqd_i + 5])

        # rotate angular torque to world frame
        w = wp.quat_rotate_inv(rot, w)

        qfrc_applied[worldid, qd_i + 0] = v[0]
        qfrc_applied[worldid, qd_i + 1] = v[1]
        qfrc_applied[worldid, qd_i + 2] = v[2]
        qfrc_applied[worldid, qd_i + 3] = w[0]
        qfrc_applied[worldid, qd_i + 4] = w[1]
        qfrc_applied[worldid, qd_i + 5] = w[2]
    elif jtype == JointType.BALL:
        qfrc_applied[worldid, qd_i + 0] = joint_f[wqd_i + 0]
        qfrc_applied[worldid, qd_i + 1] = joint_f[wqd_i + 1]
        qfrc_applied[worldid, qd_i + 2] = joint_f[wqd_i + 2]
    else:
        for i in range(joint_dof_dim[jntid, 0] + joint_dof_dim[jntid, 1]):
            qfrc_applied[worldid, qd_i + i] = joint_f[wqd_i + i]


@wp.func
def eval_single_articulation_fk(
    joint_start: int,
    joint_end: int,
    joint_articulation: wp.array(dtype=int),
    joint_q: wp.array(dtype=float),
    joint_qd: wp.array(dtype=float),
    joint_q_start: wp.array(dtype=int),
    joint_qd_start: wp.array(dtype=int),
    joint_type: wp.array(dtype=int),
    joint_parent: wp.array(dtype=int),
    joint_child: wp.array(dtype=int),
    joint_X_p: wp.array(dtype=wp.transform),
    joint_X_c: wp.array(dtype=wp.transform),
    joint_axis: wp.array(dtype=wp.vec3),
    joint_dof_dim: wp.array(dtype=int, ndim=2),
    body_com: wp.array(dtype=wp.vec3),
    # outputs
    body_q: wp.array(dtype=wp.transform),
    body_qd: wp.array(dtype=wp.spatial_vector),
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

        # parent anchor frame in world space
        X_wpj = X_pj
        # velocity of parent anchor point in world space
        v_wpj = wp.spatial_vector()
        if parent >= 0:
            X_wp = body_q[parent]
            X_wpj = X_wp * X_wpj
            r_p = wp.transform_get_translation(X_wpj) - wp.transform_point(X_wp, body_com[parent])

            v_wp = body_qd[parent]
            w_p = wp.spatial_bottom(v_wp)
            v_p = wp.spatial_top(v_wp) + wp.cross(w_p, r_p)
            v_wpj = wp.spatial_vector(v_p, w_p)

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

        # transform from world to joint anchor frame at child body
        X_wcj = X_wpj * X_j
        # transform from world to child body frame
        X_wc = X_wcj * wp.transform_inverse(X_cj)

        # transform velocity across the joint to world space
        linear_vel = wp.transform_vector(X_wpj, wp.spatial_top(v_j))
        angular_vel = wp.transform_vector(X_wpj, wp.spatial_bottom(v_j))

        v_wc = v_wpj + wp.spatial_vector(linear_vel, angular_vel)  # spatial vector with (linear, angular) ordering

        body_q[child] = X_wc
        body_qd[child] = v_wc


@wp.kernel
def eval_articulation_fk(
    articulation_start: wp.array(dtype=int),
    joint_articulation: wp.array(dtype=int),
    joint_q: wp.array(dtype=float),
    joint_qd: wp.array(dtype=float),
    joint_q_start: wp.array(dtype=int),
    joint_qd_start: wp.array(dtype=int),
    joint_type: wp.array(dtype=int),
    joint_parent: wp.array(dtype=int),
    joint_child: wp.array(dtype=int),
    joint_X_p: wp.array(dtype=wp.transform),
    joint_X_c: wp.array(dtype=wp.transform),
    joint_axis: wp.array(dtype=wp.vec3),
    joint_dof_dim: wp.array(dtype=int, ndim=2),
    body_com: wp.array(dtype=wp.vec3),
    # outputs
    body_q: wp.array(dtype=wp.transform),
    body_qd: wp.array(dtype=wp.spatial_vector),
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
    mjc_body_to_newton: wp.array2d(dtype=wp.int32),
    xpos: wp.array2d(dtype=wp.vec3),
    xquat: wp.array2d(dtype=wp.quat),
    # outputs
    body_q: wp.array(dtype=wp.transform),
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
        quat = wp.quat(quat[1], quat[2], quat[3], quat[0])
        body_q[newton_body] = wp.transform(pos, quat)


@wp.kernel
def update_body_mass_ipos_kernel(
    mjc_body_to_newton: wp.array2d(dtype=wp.int32),
    body_com: wp.array(dtype=wp.vec3f),
    body_mass: wp.array(dtype=float),
    body_gravcomp: wp.array(dtype=float),
    up_axis: int,
    # outputs
    body_ipos: wp.array2d(dtype=wp.vec3f),
    body_mass_out: wp.array2d(dtype=float),
    body_gravcomp_out: wp.array2d(dtype=float),
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
    if up_axis == 1:
        body_ipos[world, mjc_body] = wp.vec3f(
            body_com[newton_body][0], -body_com[newton_body][2], body_com[newton_body][1]
        )
    else:
        body_ipos[world, mjc_body] = body_com[newton_body]

    # update mass
    body_mass_out[world, mjc_body] = body_mass[newton_body]

    # update gravcomp
    if body_gravcomp:
        body_gravcomp_out[world, mjc_body] = body_gravcomp[newton_body]


@wp.kernel
def update_body_inertia_kernel(
    mjc_body_to_newton: wp.array2d(dtype=wp.int32),
    body_inertia: wp.array(dtype=wp.mat33f),
    # outputs
    body_inertia_out: wp.array2d(dtype=wp.vec3f),
    body_iquat_out: wp.array2d(dtype=wp.quatf),
):
    """Update MuJoCo body inertia from Newton body inertia tensor.

    Iterates over MuJoCo bodies [world, mjc_body], looks up Newton body index,
    computes eigendecomposition, and writes to MuJoCo arrays.
    """
    world, mjc_body = wp.tid()
    newton_body = mjc_body_to_newton[world, mjc_body]
    if newton_body < 0:
        return

    # Get inertia tensor
    I = body_inertia[newton_body]

    # Calculate eigenvalues and eigenvectors
    eigenvectors, eigenvalues = wp.eig3(I)

    # transpose eigenvectors to allow reshuffling by indexing rows.
    vecs_transposed = wp.transpose(eigenvectors)

    # Bubble sort for 3 elements in descending order
    for i in range(2):
        for j in range(2 - i):
            if eigenvalues[j] < eigenvalues[j + 1]:
                # Swap eigenvalues
                temp_val = eigenvalues[j]
                eigenvalues[j] = eigenvalues[j + 1]
                eigenvalues[j + 1] = temp_val
                # Swap eigenvectors
                temp_vec = vecs_transposed[j]
                vecs_transposed[j] = vecs_transposed[j + 1]
                vecs_transposed[j + 1] = temp_vec

    # Convert eigenvectors to quaternion (xyzw format)
    q = wp.quat_from_matrix(wp.transpose(vecs_transposed))
    q = wp.normalize(q)

    # Convert from xyzw to wxyz format
    q = wp.quat(q[1], q[2], q[3], q[0])

    # Store results
    body_inertia_out[world, mjc_body] = eigenvalues
    body_iquat_out[world, mjc_body] = q


@wp.kernel(module="unique", enable_backward=False)
def repeat_array_kernel(
    src: wp.array(dtype=Any),
    nelems_per_world: int,
    dst: wp.array(dtype=Any),
):
    tid = wp.tid()
    src_idx = tid % nelems_per_world
    dst[tid] = src[src_idx]


@wp.kernel
def update_axis_properties_kernel(
    mjc_actuator_to_newton_axis: wp.array2d(dtype=wp.int32),
    joint_target_kp: wp.array(dtype=float),
    joint_target_kv: wp.array(dtype=float),
    # outputs
    actuator_bias: wp.array2d(dtype=vec10),
    actuator_gain: wp.array2d(dtype=vec10),
):
    """Update MuJoCo actuator properties from Newton joint properties.

    Iterates over MuJoCo actuators [world, actuator], looks up Newton axis,
    and updates bias/gain/forcerange based on actuator type (position or velocity).

    The actuator type is encoded in the sign of mjc_actuator_to_newton_axis:
    - Positive value: position actuator, newton_axis = value
    - Negative value: velocity actuator, newton_axis = -(value + 2)
    - Value of -1: unmapped actuator
    """
    world, mjc_actuator = wp.tid()
    raw_value = mjc_actuator_to_newton_axis[world, mjc_actuator]

    if raw_value >= 0:
        # Position actuator
        newton_axis = raw_value
        kp = joint_target_kp[newton_axis]
        actuator_bias[world, mjc_actuator][1] = -kp
        actuator_gain[world, mjc_actuator][0] = kp
    elif raw_value != -1:  # raw_value == -1 means unmapped
        # Velocity actuator
        newton_axis = -raw_value - 2  # Decode: -(newton_axis + 2) -> newton_axis
        kv = joint_target_kv[newton_axis]
        actuator_bias[world, mjc_actuator][2] = -kv
        actuator_gain[world, mjc_actuator][0] = kv


@wp.kernel
def update_dof_properties_kernel(
    mjc_dof_to_newton_dof: wp.array2d(dtype=wp.int32),
    joint_armature: wp.array(dtype=float),
    joint_friction: wp.array(dtype=float),
    joint_damping: wp.array(dtype=float),
    dof_solimp: wp.array(dtype=vec5),
    dof_solref: wp.array(dtype=wp.vec2),
    # outputs
    dof_armature: wp.array2d(dtype=float),
    dof_frictionloss: wp.array2d(dtype=float),
    dof_damping: wp.array2d(dtype=float),
    dof_solimp_out: wp.array2d(dtype=vec5),
    dof_solref_out: wp.array2d(dtype=wp.vec2),
):
    """Update MuJoCo DOF properties from Newton DOF properties.

    Iterates over MuJoCo DOFs [world, dof], looks up Newton DOF,
    and copies armature, friction, damping, solimp, solref.
    """
    world, mjc_dof = wp.tid()
    newton_dof = mjc_dof_to_newton_dof[world, mjc_dof]
    if newton_dof < 0:
        return

    dof_armature[world, mjc_dof] = joint_armature[newton_dof]
    dof_frictionloss[world, mjc_dof] = joint_friction[newton_dof]
    if joint_damping:
        dof_damping[world, mjc_dof] = joint_damping[newton_dof]
    if dof_solimp:
        dof_solimp_out[world, mjc_dof] = dof_solimp[newton_dof]
    if dof_solref:
        dof_solref_out[world, mjc_dof] = dof_solref[newton_dof]


@wp.kernel
def update_jnt_properties_kernel(
    mjc_jnt_to_newton_dof: wp.array2d(dtype=wp.int32),
    joint_limit_ke: wp.array(dtype=float),
    joint_limit_kd: wp.array(dtype=float),
    joint_limit_lower: wp.array(dtype=float),
    joint_limit_upper: wp.array(dtype=float),
    joint_effort_limit: wp.array(dtype=float),
    solimplimit: wp.array(dtype=vec5),
    joint_stiffness: wp.array(dtype=float),
    limit_margin: wp.array(dtype=float),
    # outputs
    jnt_solimp: wp.array2d(dtype=vec5),
    jnt_solref: wp.array2d(dtype=wp.vec2),
    jnt_stiffness: wp.array2d(dtype=float),
    jnt_margin: wp.array2d(dtype=float),
    jnt_range: wp.array2d(dtype=wp.vec2),
    jnt_actfrcrange: wp.array2d(dtype=wp.vec2),
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
    mjc_mocap_to_newton_jnt: wp.array2d(dtype=wp.int32),
    newton_joint_X_p: wp.array(dtype=wp.transform),
    newton_joint_X_c: wp.array(dtype=wp.transform),
    # outputs
    mocap_pos: wp.array2d(dtype=wp.vec3),
    mocap_quat: wp.array2d(dtype=wp.quat),
):
    """Update mocap body positions and orientations from Newton joint data.

    Iterates over MuJoCo mocap bodies [world, mocap_idx].
    Mocap bodies are fixed-base articulations with no MuJoCo joint.
    """
    world, mocap_idx = wp.tid()

    # Get the Newton joint index for this mocap body
    newton_jnt = mjc_mocap_to_newton_jnt[world, mocap_idx]
    if newton_jnt < 0:
        return

    # Get transforms from Newton
    parent_xform = newton_joint_X_p[newton_jnt]
    child_xform = newton_joint_X_c[newton_jnt]

    # Compute body transform: X_p * inv(X_c)
    tf = parent_xform * wp.transform_inverse(child_xform)

    # Update mocap position and orientation
    mocap_pos[world, mocap_idx] = tf.p
    mocap_quat[world, mocap_idx] = wp.quat(tf.q.w, tf.q.x, tf.q.y, tf.q.z)


@wp.kernel
def update_joint_transforms_kernel(
    mjc_jnt_to_newton_jnt: wp.array2d(dtype=wp.int32),
    mjc_jnt_to_newton_dof: wp.array2d(dtype=wp.int32),
    mjc_jnt_bodyid: wp.array(dtype=wp.int32),
    mjc_jnt_type: wp.array(dtype=wp.int32),
    # Newton model data (joint-indexed)
    newton_joint_X_p: wp.array(dtype=wp.transform),
    newton_joint_X_c: wp.array(dtype=wp.transform),
    # Newton model data (DOF-indexed)
    newton_joint_axis: wp.array(dtype=wp.vec3),
    # outputs
    jnt_pos: wp.array2d(dtype=wp.vec3),
    jnt_axis: wp.array2d(dtype=wp.vec3),
    body_pos: wp.array2d(dtype=wp.vec3),
    body_quat: wp.array2d(dtype=wp.quat),
):
    """Update MuJoCo joint transforms and body positions from Newton joint data.

    Iterates over MuJoCo joints [world, jnt]. For each joint:
    - Updates MuJoCo body_pos/body_quat from Newton joint transforms
    - Updates MuJoCo jnt_pos and jnt_axis

    Note: Mocap bodies are handled by update_mocap_transforms_kernel.
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
    # Note: Mocap bodies don't have MuJoCo joints, so they're handled
    # separately by update_mocap_transforms_kernel
    mjc_body = mjc_jnt_bodyid[mjc_jnt]
    body_pos[world, mjc_body] = tf.p
    body_quat[world, mjc_body] = wp.quat(tf.q.w, tf.q.x, tf.q.y, tf.q.z)

    # Update joint axis and position (DOF-indexed for axis)
    if newton_dof >= 0:
        axis = newton_joint_axis[newton_dof]
        jnt_axis[world, mjc_jnt] = wp.quat_rotate(child_xform.q, axis)
    jnt_pos[world, mjc_jnt] = child_xform.p


@wp.kernel(enable_backward=False)
def update_shape_mappings_kernel(
    geom_to_shape_idx: wp.array(dtype=wp.int32),
    geom_is_static: wp.array(dtype=bool),
    shape_range_len: int,
    first_env_shape_base: int,
    # output - MuJoCo[world, geom] -> Newton shape
    mjc_geom_to_newton_shape: wp.array(dtype=wp.int32, ndim=2),
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
    gravity_src: wp.array(dtype=wp.vec3),
    # MuJoCo model properties
    gravity_dst: wp.array(dtype=wp.vec3f),
):
    world_idx = wp.tid()
    gravity_dst[world_idx] = gravity_src[0]


@wp.kernel
def update_geom_properties_kernel(
    shape_collision_radius: wp.array(dtype=float),
    shape_mu: wp.array(dtype=float),
    shape_ke: wp.array(dtype=float),
    shape_kd: wp.array(dtype=float),
    shape_size: wp.array(dtype=wp.vec3f),
    shape_transform: wp.array(dtype=wp.transform),
    mjc_geom_to_newton_shape: wp.array2d(dtype=wp.int32),
    geom_type: wp.array(dtype=int),
    GEOM_TYPE_MESH: int,
    geom_dataid: wp.array(dtype=int),
    mesh_pos: wp.array(dtype=wp.vec3),
    mesh_quat: wp.array(dtype=wp.quat),
    shape_torsional_friction: wp.array(dtype=float),
    shape_rolling_friction: wp.array(dtype=float),
    shape_geom_solimp: wp.array(dtype=vec5),
    shape_geom_solmix: wp.array(dtype=float),
    shape_geom_gap: wp.array(dtype=float),
    # outputs
    geom_rbound: wp.array2d(dtype=float),
    geom_friction: wp.array2d(dtype=wp.vec3f),
    geom_solref: wp.array2d(dtype=wp.vec2f),
    geom_size: wp.array2d(dtype=wp.vec3f),
    geom_pos: wp.array2d(dtype=wp.vec3f),
    geom_quat: wp.array2d(dtype=wp.quatf),
    geom_solimp: wp.array2d(dtype=vec5),
    geom_solmix: wp.array2d(dtype=float),
    geom_gap: wp.array2d(dtype=float),
):
    """Update MuJoCo geom properties from Newton shape properties.

    Iterates over MuJoCo geoms [world, geom], looks up Newton shape index,
    and copies shape properties to geom properties.
    """
    world, geom_idx = wp.tid()

    shape_idx = mjc_geom_to_newton_shape[world, geom_idx]
    if shape_idx < 0:
        return

    # update bounding radius
    geom_rbound[world, geom_idx] = shape_collision_radius[shape_idx]

    # update friction (slide, torsion, roll)
    mu = shape_mu[shape_idx]
    torsional = shape_torsional_friction[shape_idx]
    rolling = shape_rolling_friction[shape_idx]
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

    # update geom_gap from custom attribute
    if shape_geom_gap:
        geom_gap[world, geom_idx] = shape_geom_gap[shape_idx]

    # update size
    geom_size[world, geom_idx] = shape_size[shape_idx]

    # update position and orientation

    # get shape transform
    tf = shape_transform[shape_idx]

    # check if this is a mesh geom and apply mesh transformation
    if geom_type[geom_idx] == GEOM_TYPE_MESH:
        mesh_id = geom_dataid[geom_idx]
        mesh_p = mesh_pos[mesh_id]
        mesh_q = mesh_quat[mesh_id]
        mesh_tf = wp.transform(mesh_p, wp.quat(mesh_q.y, mesh_q.z, mesh_q.w, mesh_q.x))
        tf = tf * mesh_tf

    # store position and orientation
    geom_pos[world, geom_idx] = tf.p
    geom_quat[world, geom_idx] = wp.quat(tf.q.w, tf.q.x, tf.q.y, tf.q.z)


@wp.kernel(enable_backward=False)
def _create_inverse_shape_mapping_kernel(
    mjc_geom_to_newton_shape: wp.array2d(dtype=wp.int32),
    # output
    newton_shape_to_mjc_geom: wp.array(dtype=wp.int32),
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


@wp.func
def mj_body_acceleration(
    body_rootid: wp.array(dtype=int),
    xipos_in: wp.array2d(dtype=wp.vec3),
    subtree_com_in: wp.array2d(dtype=wp.vec3),
    cvel_in: wp.array2d(dtype=wp.spatial_vector),
    cacc_in: wp.array2d(dtype=wp.spatial_vector),
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
def update_eq_properties_kernel(
    mjc_eq_to_newton_eq: wp.array2d(dtype=wp.int32),
    eq_solref: wp.array(dtype=wp.vec2),
    # outputs
    eq_solref_out: wp.array2d(dtype=wp.vec2),
):
    """Update MuJoCo equality constraint properties from Newton equality constraint properties.

    Iterates over MuJoCo equality constraints [world, eq], looks up Newton eq constraint,
    and copies solref.
    """
    world, mjc_eq = wp.tid()
    newton_eq = mjc_eq_to_newton_eq[world, mjc_eq]
    if newton_eq < 0:
        return

    if eq_solref:
        eq_solref_out[world, mjc_eq] = eq_solref[newton_eq]


@wp.kernel
def convert_rigid_forces_from_mj_kernel(
    mjc_body_to_newton: wp.array2d(dtype=wp.int32),
    # mjw sources
    mjw_body_rootid: wp.array(dtype=wp.int32),
    mjw_gravity: wp.array(dtype=wp.vec3),
    mjw_xpos: wp.array2d(dtype=wp.vec3),
    mjw_subtree_com: wp.array2d(dtype=wp.vec3),
    mjw_cacc: wp.array2d(dtype=wp.spatial_vector),
    mjw_cvel: wp.array2d(dtype=wp.spatial_vector),
    # outputs
    body_qdd: wp.array(dtype=wp.spatial_vector),
    body_parent_f: wp.array(dtype=wp.spatial_vector),
):
    """Update RNE-computed rigid forces from mj_warp com-based forces."""
    world, mjc_body = wp.tid()
    newton_body = mjc_body_to_newton[world, mjc_body]

    if newton_body < 0:
        return

    if body_qdd:
        cacc = mjw_cacc[world, mjc_body]
        lin = mj_body_acceleration(
            mjw_body_rootid,
            mjw_xpos,
            mjw_subtree_com,
            mjw_cvel,
            mjw_cacc,
            world,
            mjc_body,
        )
        body_qdd[newton_body] = wp.spatial_vector(lin + mjw_gravity[0], wp.spatial_top(cacc))

    if body_parent_f:
        # TODO: implement link incoming forces
        pass
