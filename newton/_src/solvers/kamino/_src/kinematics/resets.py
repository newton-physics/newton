# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Provides a set of operations to reset the state of a physics simulation."""

from __future__ import annotations

import warp as wp

from ..core.bodies import transform_body_inertial_properties
from ..core.data import DataKamino
from ..core.joints import JointDoFType
from ..core.math import quat_from_x_rot, quat_from_y_rot, quat_from_z_rot, screw, screw_angular, screw_linear
from ..core.model import ModelKamino
from ..core.state import StateKamino
from ..core.types import float32, int32, mat33f, quatf, transformf, vec3f, vec6f
from ..kinematics.joints import (
    compute_joint_pose_and_relative_motion,
    get_joint_coords_mapping_function,
    joint_constraint_velocity_residual_universal,
    make_write_joint_data,
)
from ..solvers.fk.kernels import _correct_joint_angle, _correct_joint_quaternion, read_quat_from_array

###
# Module interface
###

__all__ = [
    "get_base_q_from_joint_q_and_body_q",
    "get_base_u_from_joint_u_and_body_u",
    "reset_body_net_wrenches",
    "reset_body_velocities",
    "reset_body_wrenches",
    "reset_joint_constraint_reactions",
    "reset_joints_state_from_bodies_state",
    "reset_select_worlds_to_initial_state",
    "reset_select_worlds_to_state",
    "reset_state_from_base_state",
    "reset_state_from_bodies_state",
    "reset_state_to_model_default",
    "reset_time",
    "set_body_q",
    "set_floating_base",
    "set_joint_state_masked",
]


###
# Module configs
###

wp.set_module_options({"enable_backward": False})


###
# Kernels
###


def make_correct_joint_coords(dof_type: JointDoFType):
    CoordsType = dof_type.coords_storage_type

    def _correct_joint_coords(
        coords,  # CoordsType,
        coords_ref,  # wp.array[float32],
    ):  # -> CoordsType
        if wp.static(
            dof_type == JointDoFType.CARTESIAN or dof_type == JointDoFType.FIXED or dof_type == JointDoFType.PRISMATIC
        ):
            pass  # No correction needed

        elif wp.static(dof_type == JointDoFType.CYLINDRICAL):  # Correct angle up to +/- 2 pi
            coords[1] = _correct_joint_angle(coords[1], coords_ref[1])

        elif wp.static(dof_type == JointDoFType.FREE):  # Correct quaternion up to sign
            quat = wp.vec4f(coords[3], coords[4], coords[5], coords[6])
            quat_ref = wp.vec4f(coords_ref[3], coords_ref[4], coords_ref[5], coords_ref[6])
            quat_corrected = _correct_joint_quaternion(quat, quat_ref)
            for i in range(4):
                coords[3 + i] = quat_corrected[i]

        elif wp.static(dof_type == JointDoFType.GIMBAL):  # Correct angles up to +/- 2 pi
            coords[0] = _correct_joint_angle(coords[0], coords_ref[0])
            coords[1] = _correct_joint_angle(coords[1], coords_ref[1])
            coords[2] = _correct_joint_angle(coords[2], coords_ref[2])

        elif wp.static(dof_type == JointDoFType.REVOLUTE):  # Correct angle up to +/- 2 pi
            coords[0] = _correct_joint_angle(coords[0], coords_ref[0])

        elif wp.static(dof_type == JointDoFType.SPHERICAL):  # Correct quaternion up to sign
            quat_ref = wp.vec4f(coords_ref[0], coords_ref[1], coords_ref[2], coords_ref[3])
            coords = _correct_joint_quaternion(coords, quat_ref)

        elif wp.static(dof_type == JointDoFType.UNIVERSAL):  # Correct angles up to +/- 2 pi
            coords[0] = _correct_joint_angle(coords[0], coords_ref[0])
            coords[1] = _correct_joint_angle(coords[1], coords_ref[1])

        return coords

    # Set type annotations manually (else CoordsType fails to resolve)
    _correct_joint_coords.__annotations__ = {
        "coords": CoordsType,
        "coords_ref": wp.array[float32],
        "return": CoordsType,
    }

    return wp.func(_correct_joint_coords)


def make_compute_and_write_joint_coords(dof_type: JointDoFType):
    """
    Generate a function computing and writing joint coordinates from the relative translation/rotation
    of the follower w.r.t. the base body, in joint frame.
    """
    num_coords = dof_type.num_coords
    assert num_coords > 0

    @wp.func
    def _compute_and_write_joint_coords(
        r_j: vec3f,
        q_j: quatf,
        coords_offset: int32,
        joint_q_ref: wp.array[float32],
        joint_q: wp.array[float32],
    ):
        # Compute joint coordinates
        coords = wp.static(get_joint_coords_mapping_function(dof_type))(r_j, q_j)

        # Apply correction up to +/- 2pi and quaternion sign
        coords_ref = joint_q_ref[coords_offset : coords_offset + num_coords]
        coords = wp.static(make_correct_joint_coords(dof_type))(coords, coords_ref)

        # Write out joint coordinates
        for i in range(num_coords):
            joint_q[coords_offset + i] = coords[i]

    return _compute_and_write_joint_coords


def make_compute_and_write_joint_vel(dof_type: JointDoFType):
    """
    Generate a function computing and writing joint velocity from the relative velocity
    of the follower w.r.t. the base body, in joint frame (as well as the relative orientation,
    in the case of the universal joint)
    """
    num_dofs = dof_type.num_dofs
    assert num_dofs > 0
    dof_axes = dof_type.dofs_axes

    @wp.func
    def _compute_and_write_joint_vel(
        q_j: quatf,
        u_j: vec6f,
        dofs_offset: int32,
        joint_u: wp.array[float32],
    ):
        # Convert angular velocity to intermediary body frame for universal joint
        if wp.static(dof_type == JointDoFType.UNIVERSAL):
            u_j = joint_constraint_velocity_residual_universal(q_j, u_j)

        # Write out joint velocity (=components of relative velocity along unconstrained axes)
        for i in range(num_dofs):
            joint_u[dofs_offset + i] = u_j[dof_axes[i]]

    return _compute_and_write_joint_vel


@wp.func
def _compute_and_write_joint_coords_and_vel(
    dof_type: int32,
    r_j: vec3f,
    q_j: quatf,
    u_j: vec6f,
    coords_offset: int32,
    dofs_offset: int32,
    joint_q_ref: wp.array[float32],
    joint_q: wp.array[float32],
    joint_u: wp.array[float32],
):
    if dof_type == JointDoFType.CARTESIAN:
        wp.static(make_compute_and_write_joint_coords(JointDoFType.CARTESIAN))(
            r_j, q_j, coords_offset, joint_q_ref, joint_q
        )
        wp.static(make_compute_and_write_joint_vel(JointDoFType.CARTESIAN))(q_j, u_j, dofs_offset, joint_u)

    elif dof_type == JointDoFType.CYLINDRICAL:
        wp.static(make_compute_and_write_joint_coords(JointDoFType.CYLINDRICAL))(
            r_j, q_j, coords_offset, joint_q_ref, joint_q
        )
        wp.static(make_compute_and_write_joint_vel(JointDoFType.CYLINDRICAL))(q_j, u_j, dofs_offset, joint_u)

    elif dof_type == JointDoFType.FIXED:
        pass  # 0 coords and dofs

    elif dof_type == JointDoFType.FREE:
        wp.static(make_compute_and_write_joint_coords(JointDoFType.FREE))(r_j, q_j, coords_offset, joint_q_ref, joint_q)
        wp.static(make_compute_and_write_joint_vel(JointDoFType.FREE))(q_j, u_j, dofs_offset, joint_u)

    elif dof_type == JointDoFType.GIMBAL:
        wp.static(make_compute_and_write_joint_coords(JointDoFType.GIMBAL))(
            r_j, q_j, coords_offset, joint_q_ref, joint_q
        )
        wp.static(make_compute_and_write_joint_vel(JointDoFType.GIMBAL))(q_j, u_j, dofs_offset, joint_u)

    elif dof_type == JointDoFType.PRISMATIC:
        wp.static(make_compute_and_write_joint_coords(JointDoFType.PRISMATIC))(
            r_j, q_j, coords_offset, joint_q_ref, joint_q
        )
        wp.static(make_compute_and_write_joint_vel(JointDoFType.PRISMATIC))(q_j, u_j, dofs_offset, joint_u)

    elif dof_type == JointDoFType.REVOLUTE:
        wp.static(make_compute_and_write_joint_coords(JointDoFType.REVOLUTE))(
            r_j, q_j, coords_offset, joint_q_ref, joint_q
        )
        wp.static(make_compute_and_write_joint_vel(JointDoFType.REVOLUTE))(q_j, u_j, dofs_offset, joint_u)

    elif dof_type == JointDoFType.SPHERICAL:
        wp.static(make_compute_and_write_joint_coords(JointDoFType.SPHERICAL))(
            r_j, q_j, coords_offset, joint_q_ref, joint_q
        )
        wp.static(make_compute_and_write_joint_vel(JointDoFType.SPHERICAL))(q_j, u_j, dofs_offset, joint_u)

    elif dof_type == JointDoFType.UNIVERSAL:
        wp.static(make_compute_and_write_joint_coords(JointDoFType.UNIVERSAL))(
            r_j, q_j, coords_offset, joint_q_ref, joint_q
        )
        wp.static(make_compute_and_write_joint_vel(JointDoFType.UNIVERSAL))(q_j, u_j, dofs_offset, joint_u)


@wp.func
def _get_joint_rel_transform_from_coords(
    dof_type: int32,
    coord_offset: int32,
    joint_q: wp.array[float32],
) -> transformf:
    # Initialize transform to identity
    t = wp.vec3f(0.0, 0.0, 0.0)
    q = wp.quatf(0.0, 0.0, 0.0, 1.0)

    # Overwrite transform base on joint type and coords
    if dof_type == JointDoFType.CARTESIAN:
        t[0] = joint_q[coord_offset]
        t[1] = joint_q[coord_offset + 1]
        t[2] = joint_q[coord_offset + 2]
    elif dof_type == JointDoFType.CYLINDRICAL:
        t[0] = joint_q[coord_offset]
        q = quat_from_x_rot(joint_q[coord_offset + 1])
    elif dof_type == JointDoFType.FIXED:
        pass  # No dofs to apply
    elif dof_type == JointDoFType.FREE:
        t[0] = joint_q[coord_offset]
        t[1] = joint_q[coord_offset + 1]
        t[2] = joint_q[coord_offset + 2]
        q = read_quat_from_array(joint_q, coord_offset + 3, True)
    elif dof_type == JointDoFType.GIMBAL:
        q_x = quat_from_x_rot(joint_q[coord_offset])
        q_y = quat_from_y_rot(joint_q[coord_offset + 1])
        q_z = quat_from_z_rot(joint_q[coord_offset + 2])
        q = q_x * q_y * q_z
    elif dof_type == JointDoFType.PRISMATIC:
        t[0] = joint_q[coord_offset]
    elif dof_type == JointDoFType.REVOLUTE:
        q = quat_from_x_rot(joint_q[coord_offset])
    elif dof_type == JointDoFType.SPHERICAL:
        q = read_quat_from_array(joint_q, coord_offset, True)
    elif dof_type == JointDoFType.UNIVERSAL:
        q_x = quat_from_x_rot(joint_q[coord_offset])
        q_y = quat_from_y_rot(joint_q[coord_offset + 1])
        q = q_x * q_y
    else:
        assert False, "Unexpected joint dof type"  # noqa: B011

    return transformf(t, q)


@wp.kernel
def _get_base_q_from_joint_q_and_body_q(
    # Inputs:
    model_base_joint_index: wp.array[int32],
    model_base_body_index: wp.array[int32],
    model_joint_dof_type: wp.array[int32],
    model_joint_coords_offset: wp.array[int32],
    state_joint_q: wp.array[float32],
    state_body_q: wp.array[transformf],
    world_mask: wp.array[bool],
    # Outputs:
    base_q: wp.array[transformf],
):
    # Get thread id as world id
    wid = wp.tid()

    # Early return based on mask
    if not world_mask[wid]:
        return

    # Read base_q from joint_q if a base joint was set for this world
    base_joint_id = model_base_joint_index[wid]
    if base_joint_id >= 0:
        dof_type = model_joint_dof_type[base_joint_id]
        coords_offset = model_joint_coords_offset[base_joint_id]
        base_q[wid] = _get_joint_rel_transform_from_coords(dof_type, coords_offset, state_joint_q)

    # Otherwise read base_q from body_q if a base body was set for this world
    else:
        base_body_id = model_base_body_index[wid]
        assert base_body_id >= 0
        base_q[wid] = state_body_q[base_body_id]


@wp.kernel
def _get_base_u_from_joint_u_and_body_u(
    # Inputs:
    model_base_joint_index: wp.array[int32],
    model_base_body_index: wp.array[int32],
    model_joint_dofs_offset: wp.array[int32],
    state_joint_u: wp.array[float32],
    state_body_u: wp.array[vec6f],
    world_mask: wp.array[bool],
    # Outputs:
    base_u: wp.array[vec6f],
):
    # Get thread id as world id
    wid = wp.tid()

    # Early return based on mask
    if not world_mask[wid]:
        return

    # Read base_u from joint_u if a base joint was set for this world
    base_joint_id = model_base_joint_index[wid]
    if base_joint_id >= 0:
        dofs_offset = model_joint_dofs_offset[base_joint_id]
        num_dofs = model_joint_dofs_offset[base_joint_id + 1] - dofs_offset
        assert num_dofs == 6  # Note: only free joint as base joint supported right now
        base_u[wid] = vec6f(
            state_joint_u[dofs_offset],
            state_joint_u[dofs_offset + 1],
            state_joint_u[dofs_offset + 2],
            state_joint_u[dofs_offset + 3],
            state_joint_u[dofs_offset + 4],
            state_joint_u[dofs_offset + 5],
        )

    # Otherwise read base_u from body_u if a base body was set for this world
    else:
        base_body_id = model_base_body_index[wid]
        assert base_body_id >= 0
        base_u[wid] = state_body_u[base_body_id]


@wp.kernel
def _set_body_q(
    # Inputs:
    body_world_id: wp.array[int32],
    body_q_in: wp.array[transformf],
    world_mask: wp.array[bool],
    # Outputs:
    body_q_out: wp.array[transformf],
):
    body_id = wp.tid()
    wid = body_world_id[body_id]
    if not world_mask[wid]:
        return
    body_q_out[body_id] = body_q_in[body_id]


@wp.kernel
def _reset_joints_state_from_bodies_state(
    # Inputs
    joint_world_id: wp.array[int32],
    joint_dof_type: wp.array[int32],
    joint_coords_offset: wp.array[int32],
    joint_dofs_offset: wp.array[int32],
    joint_cts_offset: wp.array[int32],
    joint_bid_B: wp.array[int32],
    joint_bid_F: wp.array[int32],
    joint_B_r_Bj: wp.array[vec3f],
    joint_F_r_Fj: wp.array[vec3f],
    joint_X_Bj: wp.array[mat33f],
    joint_X_Fj: wp.array[mat33f],
    joint_q_0: wp.array[float32],
    body_q: wp.array[transformf],
    body_u: wp.array[vec6f],
    world_mask: wp.array[bool],
    # Outputs
    joint_q: wp.array[float32],
    joint_q_prev: wp.array[float32],
    joint_u: wp.array[float32],
    joint_lambda: wp.array[float32],
):
    # Get thread id as joint id
    jid = wp.tid()

    # Early return based on mask
    wid = joint_world_id[jid]
    if not world_mask[wid]:
        return

    # Retrieve the joint model data
    dof_type = joint_dof_type[jid]
    coords_offset = joint_coords_offset[jid]
    num_coords = joint_coords_offset[jid + 1] - coords_offset
    dofs_offset = joint_dofs_offset[jid]
    cts_offset = joint_cts_offset[jid]
    num_cts = joint_cts_offset[jid + 1] - cts_offset
    bid_B = joint_bid_B[jid]
    bid_F = joint_bid_F[jid]
    r_B = joint_B_r_Bj[jid]
    r_F = joint_F_r_Fj[jid]
    X_B = joint_X_Bj[jid]
    X_F = joint_X_Fj[jid]

    # Get pose and velocity of base/follower bodies
    T_B = wp.transform_identity(dtype=float32)
    u_B = vec6f(0.0)
    if bid_B > -1:
        T_B = body_q[bid_B]
        u_B = body_u[bid_B]
    T_F = body_q[bid_F]
    u_F = body_u[bid_F]

    # Compute the relative motion of the follower w.r.t. the base body, in joint frame
    _, r_j, q_j, u_j = compute_joint_pose_and_relative_motion(T_B, T_F, u_B, u_F, r_B, r_F, X_B, X_F)

    # Evaluate joint coordinates/velocity from relative motion
    _compute_and_write_joint_coords_and_vel(
        dof_type, r_j, q_j, u_j, coords_offset, dofs_offset, joint_q_0, joint_q, joint_u
    )
    for i in range(num_coords):
        joint_q_prev[coords_offset + i] = joint_q[coords_offset + i]

    # Set lambda to zero
    for i in range(num_cts):
        joint_lambda[cts_offset + i] = 0.0


@wp.kernel
def _reset_body_velocities(
    # Inputs
    body_world_id: wp.array[int32],
    world_mask: wp.array[bool],
    # Outputs
    body_u: wp.array[vec6f],
):
    # Get thread id as body id
    body_id = wp.tid()

    # Early return based on mask
    wid = body_world_id[body_id]
    if not world_mask[wid]:
        return

    # Reset velocities to zero
    body_u[body_id] = vec6f(0.0)


@wp.kernel
def _reset_body_wrenches(
    # Inputs
    body_world_id: wp.array[int32],
    world_mask: wp.array[bool],
    # Outputs
    body_w: wp.array[vec6f],
    body_w_e: wp.array[vec6f],
):
    # Get thread id as body id
    body_id = wp.tid()

    # Early return based on mask
    wid = body_world_id[body_id]
    if not world_mask[wid]:
        return

    # Reset wrenches to zero
    body_w[body_id] = vec6f(0.0)
    body_w_e[body_id] = vec6f(0.0)


@wp.kernel
def _reset_time_of_select_worlds(
    # Inputs:
    world_mask: wp.array[bool],
    # Outputs:
    data_time: wp.array[float32],
    data_steps: wp.array[int32],
):
    # Retrieve the world index from the 1D thread index
    wid = wp.tid()

    # Skip resetting time if the world has not been marked for reset
    if not world_mask[wid]:
        return

    # Reset both the physical time and step count to zero
    data_time[wid] = 0.0
    data_steps[wid] = 0


@wp.kernel
def _reset_body_state_of_select_worlds(
    # Inputs:
    world_mask: wp.array[bool],
    model_body_wid: wp.array[int32],
    model_body_q_i_0: wp.array[transformf],
    model_body_u_i_0: wp.array[vec6f],
    # Outputs:
    state_q_i: wp.array[transformf],
    state_u_i: wp.array[vec6f],
    state_w_i: wp.array[vec6f],
    state_w_i_e: wp.array[vec6f],
):
    # Retrieve the body index from the 1D thread index
    bid = wp.tid()

    # Retrieve the world index for this body
    wid = model_body_wid[bid]

    # Skip resetting this body if the world has not been marked for reset
    if not world_mask[wid]:
        return

    # Retrieve the target state for this body
    q_i_0 = model_body_q_i_0[bid]
    u_i_0 = model_body_u_i_0[bid]

    # Store the reset state in the output arrays and zero-out wrenches
    state_q_i[bid] = q_i_0
    state_u_i[bid] = u_i_0
    state_w_i[bid] = vec6f(0.0)
    state_w_i_e[bid] = vec6f(0.0)


@wp.kernel
def _eval_floating_base_relative_transform(
    # Inputs:
    model_base_joint_index: wp.array[int32],
    model_base_body_index: wp.array[int32],
    joint_B_r_Bj: wp.array[vec3f],
    joint_F_r_Fj: wp.array[vec3f],
    joint_X_Bj: wp.array[mat33f],
    joint_X_Fj: wp.array[mat33f],
    base_q: wp.array[transformf],  # None also supported
    base_u: wp.array[vec6f],  # None also supported
    body_q: wp.array[transformf],
    body_u: wp.array[vec6f],
    world_mask: wp.array[bool],
    relative_base_u: bool,
    # Outputs:
    rel_transform: wp.array[transformf],
    rel_velocity: wp.array[vec6f],
    new_base_pos: wp.array[vec3f],
):
    # Get thread id as world id
    wid = wp.tid()

    # Early return based on mask
    if not world_mask[wid]:
        return

    # Determine new pose of the base body (= follower of the base joint if there is a base joint)
    base_joint_id = model_base_joint_index[wid]
    base_body_id = model_base_body_index[wid]
    if not base_q:  # No prescribed base_q: take new base body pose as its current pose
        base_body_pose = body_q[base_body_id]
    elif base_joint_id >= 0:  # If there is a base joint, base_q is the transform in joint frame
        # body_q_B * T_B * base_q = body_q_F * T_F, and body_q_B = identity for a unary joint
        # This gives body_q_F = T_B * base_q * T_F ^-1
        r_B = joint_B_r_Bj[base_joint_id]
        r_F = joint_F_r_Fj[base_joint_id]
        X_B = joint_X_Bj[base_joint_id]
        X_F = joint_X_Fj[base_joint_id]
        T_B = transformf(r_B, wp.quat_from_matrix(X_B))
        T_F = transformf(r_F, wp.quat_from_matrix(X_F))
        T_F_inv = wp.transform_inverse(T_F)
        base_body_pose = wp.transform_multiply(wp.transform_multiply(T_B, base_q[wid]), T_F_inv)
    else:  # Directly interpret base_q as the new base body pose if no base joint
        base_body_pose = base_q[wid]
    new_base_pos[wid] = wp.transform_get_translation(base_body_pose)

    # Determine relative transform to apply, from current to target base body pose
    if not base_q:
        T_rel = wp.transform_identity(float32)
        # Ensure we get a bit-accurate identity (although the formula below would yield the identity)
    else:
        T_rel = wp.transform_multiply(base_body_pose, wp.transform_inverse(body_q[base_body_id]))
    rel_transform[wid] = T_rel

    # Determine new velocity of the base body
    if not base_u:  # No prescribed base_u: use zero additional relative velocity to apply to the base
        rel_velocity[wid] = vec6f(0.0)
        return
    base_u_ = base_u[wid]
    if base_joint_id >= 0:  # If there is a base joint, base_u is the velocity in joint frame
        # For a unary joint, the joint velocity is simply the follower velocity in base joint frame
        # i.e. base_u = (base_v, base_omega) = X_B^T * (v_F + omega_F x R_F r_F), X_B^T * omega_F
        if not base_q:  # Read joint data that was not read above in the base_q = None path
            r_F = joint_F_r_Fj[base_joint_id]
            X_B = joint_X_Bj[base_joint_id]
        base_v = base_u_[:3]
        base_omega = base_u_[3:]
        omega_F = X_B * base_omega
        q_F = wp.transform_get_rotation(base_body_pose)
        v_F = X_B * base_v - wp.cross(omega_F, wp.quat_rotate(q_F, r_F))
    else:  # Directly interpret base_u as the new base body velocity if no base joint
        v_F = base_u_[:3]
        omega_F = base_u_[3:]

    # Determine relative velocity change to apply to base body (after applying T_rel)
    u_curr = body_u[base_body_id]
    v_curr = u_curr[:3]
    omega_curr = u_curr[3:]
    if relative_base_u:
        v_rel = wp.transform_vector(T_rel, v_F - v_curr)
        omega_rel = wp.transform_vector(T_rel, omega_F - omega_curr)
    else:
        v_rel = v_F - wp.transform_vector(T_rel, v_curr)
        omega_rel = omega_F - wp.transform_vector(T_rel, omega_curr)
    rel_velocity[wid] = vec6f(*v_rel, *omega_rel)


@wp.kernel
def _apply_floating_base_transform(
    # Inputs:
    body_world_id: wp.array[int32],
    rel_transform: wp.array[transformf],
    rel_velocity: wp.array[vec6f],
    new_base_pos: wp.array[vec3f],
    world_mask: wp.array[bool],
    # Outputs:
    body_q: wp.array[transformf],
    body_u: wp.array[vec6f],
):
    # Get thread id as body id
    body_id = wp.tid()

    # Early return based on mask
    wid = body_world_id[body_id]
    if not world_mask[wid]:
        return

    # Transform body pose
    T_rel = rel_transform[wid]
    body_q_new = wp.transform_multiply(T_rel, body_q[body_id])
    body_q[body_id] = body_q_new

    # Transform body velocity
    body_u_curr = body_u[body_id]
    body_v_new = wp.transform_vector(T_rel, body_u_curr[:3])
    body_omega_new = wp.transform_vector(T_rel, body_u_curr[3:])

    # Compose with new base velocity
    u_rel = rel_velocity[wid]
    omega_rel = u_rel[3:]
    body_pos_new = wp.transform_get_translation(body_q_new)
    body_v_new += u_rel[:3] + wp.cross(omega_rel, body_pos_new - new_base_pos[wid])
    body_omega_new += omega_rel
    body_u[body_id] = vec6f(*body_v_new, *body_omega_new)


@wp.kernel
def _reset_body_state_from_base(
    # Inputs:
    world_mask: wp.array[bool],
    model_info_base_body_index: wp.array[int32],
    model_body_wid: wp.array[int32],
    model_bodies_q_i_0: wp.array[transformf],
    base_q: wp.array[transformf],
    base_u: wp.array[vec6f],
    # Outputs:
    state_q_i: wp.array[transformf],
    state_u_i: wp.array[vec6f],
    state_w_i: wp.array[vec6f],
):
    # Retrieve the body index from the 1D thread index
    bid = wp.tid()

    # Retrieve the world index for this body
    wid = model_body_wid[bid]

    # Skip resetting this body if the world has not been marked for reset
    if not world_mask[wid]:
        return

    # Retrieve the index of the base body for this world
    base_bid = model_info_base_body_index[wid]

    # Retrieve the initial pose of the base body
    if base_bid >= 0:
        q_b_0 = model_bodies_q_i_0[base_bid]
    else:
        # If there is no base body, use the identity transform
        q_b_0 = wp.transform_identity(dtype=float32)

    # Retrieve the initial pose for this body
    q_i_0 = model_bodies_q_i_0[bid]

    # Retrieve the target state of the base body
    q_b = base_q[wid]
    u_b = base_u[wid]

    # Compute the relative pose transform that
    # moves the base body to the target pose
    X_b = wp.transform_multiply(q_b, wp.transform_inverse(q_b_0))

    # Retrieve the position vectors of the base and current body
    r_b_0 = wp.transform_get_translation(q_b_0)
    r_i_0 = wp.transform_get_translation(q_i_0)

    # Decompose the base body's target twist
    v_b = screw_linear(u_b)
    omega_b = screw_angular(u_b)

    # Compute the target pose and twist for this body
    q_i = wp.transform_multiply(X_b, q_i_0)
    u_i = screw(v_b + wp.cross(omega_b, r_i_0 - r_b_0), omega_b)

    # Store the reset state in the output arrays and zero-out wrenches
    state_q_i[bid] = q_i
    state_u_i[bid] = u_i
    state_w_i[bid] = vec6f(0.0)


@wp.kernel
def _reset_joint_state_of_select_worlds(
    # Inputs:
    world_mask: wp.array[bool],
    model_joint_wid: wp.array[int32],
    model_joint_num_dynamic_cts: wp.array[int32],
    model_joint_num_kinematic_cts: wp.array[int32],
    model_joint_coords_offset: wp.array[int32],
    model_joint_dofs_offset: wp.array[int32],
    model_joint_dynamic_cts_offset_joint_cts: wp.array[int32],
    model_joint_kinematic_cts_offset_joint_cts: wp.array[int32],
    model_joint_q_j_ref: wp.array[float32],
    # Outputs:
    state_q_j: wp.array[float32],
    state_q_j_p: wp.array[float32],
    state_dq_j: wp.array[float32],
    state_lambda_j: wp.array[float32],
):
    # Retrieve the body index from the 1D thread index
    jid = wp.tid()

    # Retrieve the world index for this body
    wid = model_joint_wid[jid]

    # Skip resetting this joint if the world has not been marked for reset
    if not world_mask[wid]:
        return

    # Retrieve the joint model data
    coords_offset = model_joint_coords_offset[jid]
    num_coords = model_joint_coords_offset[jid + 1] - coords_offset
    dofs_offset = model_joint_dofs_offset[jid]
    num_dofs = model_joint_dofs_offset[jid + 1] - dofs_offset
    dynamic_cts_offset = model_joint_dynamic_cts_offset_joint_cts[jid]
    kinematic_cts_offset = model_joint_kinematic_cts_offset_joint_cts[jid]
    num_dynamic_cts = model_joint_num_dynamic_cts[jid]
    num_kinematic_cts = model_joint_num_kinematic_cts[jid]

    # Reset all joint state data
    for j in range(num_coords):
        q_j_ref = model_joint_q_j_ref[coords_offset + j]
        state_q_j[coords_offset + j] = q_j_ref
        state_q_j_p[coords_offset + j] = q_j_ref
    for j in range(num_dofs):
        state_dq_j[dofs_offset + j] = 0.0
    for j in range(num_dynamic_cts):
        state_lambda_j[dynamic_cts_offset + j] = 0.0
    for j in range(num_kinematic_cts):
        state_lambda_j[kinematic_cts_offset + j] = 0.0


@wp.kernel
def _set_joint_state_of_select_worlds(
    # Inputs:
    write_velocities: bool,
    world_mask: wp.array[bool],
    model_joint_wid: wp.array[int32],
    model_joint_coords_offset: wp.array[int32],
    model_joint_dofs_offset: wp.array[int32],
    src_q: wp.array[float32],
    src_u: wp.array[float32],
    # Outputs:
    dst_q: wp.array[float32],
    dst_q_p: wp.array[float32],
    dst_dq: wp.array[float32],
):
    # Retrieve the joint index from the 1D thread index
    jid = wp.tid()

    # Retrieve the world index for this joint
    wid = model_joint_wid[jid]

    # Skip writing this joint's state if the world has not been marked for reset
    if not world_mask[wid]:
        return

    # Write the joint's coordinate block to both `q` and `q_p` (TWOPI reference)
    coords_offset = model_joint_coords_offset[jid]
    num_coords = model_joint_coords_offset[jid + 1] - coords_offset
    for j in range(num_coords):
        v = src_q[coords_offset + j]
        dst_q[coords_offset + j] = v
        dst_q_p[coords_offset + j] = v

    # Optionally write the joint's DoF velocities
    if write_velocities:
        dofs_offset = model_joint_dofs_offset[jid]
        num_dofs = model_joint_dofs_offset[jid + 1] - dofs_offset
        for j in range(num_dofs):
            dst_dq[dofs_offset + j] = src_u[dofs_offset + j]


@wp.kernel
def _reset_bodies_of_select_worlds(
    # Inputs:
    mask: wp.array[bool],
    # Inputs:
    model_bid: wp.array[int32],
    model_i_I_i: wp.array[mat33f],
    model_inv_i_I_i: wp.array[mat33f],
    state_q_i: wp.array[transformf],
    state_u_i: wp.array[vec6f],
    # Outputs:
    data_q_i: wp.array[transformf],
    data_u_i: wp.array[vec6f],
    data_I_i: wp.array[mat33f],
    data_inv_I_i: wp.array[mat33f],
    data_w_i: wp.array[vec6f],
    data_w_a_i: wp.array[vec6f],
    data_w_j_i: wp.array[vec6f],
    data_w_l_i: wp.array[vec6f],
    data_w_c_i: wp.array[vec6f],
    data_w_e_i: wp.array[vec6f],
):
    # Retrieve the body index from the 1D thread index
    bid = wp.tid()

    # Retrieve the world index for this body
    wid = model_bid[bid]

    # Retrieve the reset flag for the corresponding world
    world_has_reset = mask[wid]

    # Skip resetting this body if the world has not been marked for reset
    if not world_has_reset:
        return

    # Create a zero-valued vec6 to zero-out wrenches
    zero6 = vec6f(0.0)

    # Retrieve the target state for this body
    q_i_0 = state_q_i[bid]
    u_i_0 = state_u_i[bid]

    # Retrieve the model data for this body
    i_I_i = model_i_I_i[bid]
    inv_i_I_i = model_inv_i_I_i[bid]

    # Compute the moment of inertia matrices in world coordinates
    I_i, inv_I_i = transform_body_inertial_properties(q_i_0, i_I_i, inv_i_I_i)

    # Store the reset state and inertial properties
    # in the output arrays and zero-out wrenches
    data_q_i[bid] = q_i_0
    data_u_i[bid] = u_i_0
    data_I_i[bid] = I_i
    data_inv_I_i[bid] = inv_I_i
    data_w_i[bid] = zero6
    data_w_a_i[bid] = zero6
    data_w_j_i[bid] = zero6
    data_w_l_i[bid] = zero6
    data_w_c_i[bid] = zero6
    data_w_e_i[bid] = zero6


@wp.kernel
def _reset_body_net_wrenches(
    # Inputs:
    world_mask: wp.array[bool],
    body_wid: wp.array[int32],
    # Outputs:
    body_w_i: wp.array[vec6f],
):
    # Retrieve the body index from the 1D thread grid
    bid = wp.tid()

    # Retrieve the world index for this body
    wid = body_wid[bid]

    # Skip resetting this body if the world has not been marked for reset
    if not world_mask[wid]:
        return

    # Zero-out wrenches
    body_w_i[bid] = vec6f(0.0)


@wp.kernel
def _reset_joint_constraint_reactions(
    # Inputs:
    world_mask: wp.array[bool],
    model_joint_wid: wp.array[int32],
    model_joint_num_dynamic_cts: wp.array[int32],
    model_joint_num_kinematic_cts: wp.array[int32],
    model_joint_dynamic_cts_offset_total_cts: wp.array[int32],
    model_joint_kinematic_cts_offset_total_cts: wp.array[int32],
    # Outputs:
    lambda_j: wp.array[float32],
):
    # Retrieve the joint index from the thread grid
    jid = wp.tid()

    # Retrieve the world index and actuation type of the joint
    wid = model_joint_wid[jid]

    # Early exit the operation if the joint's world is flagged as skipped or if the joint is not actuated
    if not world_mask[wid]:
        return

    # Retrieve the joint model data
    num_dynamic_cts = model_joint_num_dynamic_cts[jid]
    num_kinematic_cts = model_joint_num_kinematic_cts[jid]
    dynamic_cts_offset = model_joint_dynamic_cts_offset_total_cts[jid]
    kinematic_cts_offset = model_joint_kinematic_cts_offset_total_cts[jid]

    # Reset the joint constraint reactions
    for j in range(num_dynamic_cts):
        lambda_j[dynamic_cts_offset + j] = 0.0
    for j in range(num_kinematic_cts):
        lambda_j[kinematic_cts_offset + j] = 0.0


@wp.kernel
def _reset_joints_of_select_worlds(
    # Inputs:
    reset_constraints: bool,
    mask: wp.array[bool],
    model_joint_wid: wp.array[int32],
    model_joint_dof_type: wp.array[int32],
    model_joint_num_dynamic_cts: wp.array[int32],
    model_joint_num_kinematic_cts: wp.array[int32],
    model_joint_coords_offset: wp.array[int32],
    model_joint_dofs_offset: wp.array[int32],
    model_joint_dynamic_cts_offset_joint_cts: wp.array[int32],
    model_joint_kinematic_cts_offset_joint_cts: wp.array[int32],
    model_joint_bid_B: wp.array[int32],
    model_joint_bid_F: wp.array[int32],
    model_joint_B_r_Bj: wp.array[vec3f],
    model_joint_F_r_Fj: wp.array[vec3f],
    model_joint_X_Bj: wp.array[mat33f],
    model_joint_X_Fj: wp.array[mat33f],
    model_joint_q_j_ref: wp.array[float32],
    state_q_i: wp.array[transformf],
    state_u_i: wp.array[vec6f],
    state_lambda_j: wp.array[float32],
    # Outputs:
    data_p_j: wp.array[transformf],
    data_r_j: wp.array[float32],
    data_dr_j: wp.array[float32],
    data_q_j: wp.array[float32],
    data_dq_j: wp.array[float32],
    data_lambda_j: wp.array[float32],
):
    # Retrieve the body index from the 1D thread index
    jid = wp.tid()

    # Retrieve the world index for this body
    wid = model_joint_wid[jid]

    # Retrieve the reset flag for the corresponding world
    world_has_reset = mask[wid]

    # Skip resetting this joint if the world has not been marked for reset
    if not world_has_reset:
        return

    # Retrieve the joint model data
    dof_type = model_joint_dof_type[jid]
    num_dynamic_cts = model_joint_num_dynamic_cts[jid]
    num_kinematic_cts = model_joint_num_kinematic_cts[jid]
    coords_offset = model_joint_coords_offset[jid]
    dofs_offset = model_joint_dofs_offset[jid]
    dynamic_cts_offset = model_joint_dynamic_cts_offset_joint_cts[jid]
    kinematic_cts_offset = model_joint_kinematic_cts_offset_joint_cts[jid]
    bid_B = model_joint_bid_B[jid]
    bid_F = model_joint_bid_F[jid]
    B_r_Bj = model_joint_B_r_Bj[jid]
    F_r_Fj = model_joint_F_r_Fj[jid]
    X_Bj = model_joint_X_Bj[jid]
    X_Fj = model_joint_X_Fj[jid]

    # If the Base body is the world (bid=-1), use the identity transform (frame
    # of the world's origin), otherwise retrieve the Base body's pose and twist
    T_B_j = wp.transform_identity(dtype=float32)
    u_B_j = vec6f(0.0)
    if bid_B > -1:
        T_B_j = state_q_i[bid_B]
        u_B_j = state_u_i[bid_B]

    # Retrieve the Follower body's pose and twist
    T_F_j = state_q_i[bid_F]
    u_F_j = state_u_i[bid_F]

    # Compute the joint frame pose and relative motion
    p_j, j_r_j, j_q_j, j_u_j = compute_joint_pose_and_relative_motion(
        T_B_j, T_F_j, u_B_j, u_F_j, B_r_Bj, F_r_Fj, X_Bj, X_Fj
    )

    # Store the absolute pose of the joint frame in world coordinates
    data_p_j[jid] = p_j

    # Store the joint constraint residuals and motion
    wp.static(make_write_joint_data())(
        dof_type,
        kinematic_cts_offset,
        dofs_offset,
        coords_offset,
        j_r_j,
        j_q_j,
        j_u_j,
        model_joint_q_j_ref,
        data_r_j,
        data_dr_j,
        data_q_j,
        data_dq_j,
    )

    # If requested, reset the joint constraint reactions to zero
    if reset_constraints:
        for j in range(num_dynamic_cts):
            data_lambda_j[dynamic_cts_offset + j] = 0.0
        for j in range(num_kinematic_cts):
            data_lambda_j[kinematic_cts_offset + j] = 0.0
    # Otherwise, copy the target constraint reactions from the target state
    else:
        for j in range(num_dynamic_cts):
            data_lambda_j[dynamic_cts_offset + j] = state_lambda_j[dynamic_cts_offset + j]
        for j in range(num_kinematic_cts):
            data_lambda_j[kinematic_cts_offset + j] = state_lambda_j[kinematic_cts_offset + j]


###
# Launchers
###


def reset_time(
    model: ModelKamino,
    time: wp.array,
    steps: wp.array,
    world_mask: wp.array,
):
    wp.launch(
        _reset_time_of_select_worlds,
        dim=model.size.num_worlds,
        inputs=[
            # Inputs:
            world_mask,
            # Outputs:
            time,
            steps,
        ],
        device=model.device,
    )


def reset_body_net_wrenches(
    model: ModelKamino,
    body_w: wp.array,
    world_mask: wp.array,
):
    """
    Reset the body constraint wrenches of the selected worlds given an array of per-world flags.

    Args:
        model: Input model container holding the time-invariant data of the system.
        body_w: Array of body constraint wrenches to be reset.
        world_mask: Array of per-world flags indicating which worlds should be reset.
    """
    wp.launch(
        _reset_body_net_wrenches,
        dim=model.size.sum_of_num_bodies,
        inputs=[
            # Inputs:
            world_mask,
            model.bodies.wid,
            # Outputs:
            body_w,
        ],
        device=model.device,
    )


def reset_joint_constraint_reactions(
    model: ModelKamino,
    lambda_j: wp.array,
    world_mask: wp.array,
):
    """
    Resets the joint constraint reaction forces/torques to zero.

    This function is typically called at the beginning of a simulation step
    to clear out any accumulated reaction forces from the previous step.

    Args:
        model (ModelKamino):
            The model container holding the time-invariant data of the simulation.
        lambda_j (wp.array):
            The array of joint constraint reaction forces/torques.\n
            Shape of ``(sum_of_num_joint_constraints,)`` and type :class:`float`.
        world_mask (wp.array):
            An array indicating which worlds are active (True) or skipped (False).\n
            Shape of ``(num_worlds,)`` and type :class:`bool`.
    """
    wp.launch(
        _reset_joint_constraint_reactions,
        dim=model.size.sum_of_num_joints,
        inputs=[
            # Inputs:
            world_mask,
            model.joints.wid,
            model.joints.num_dynamic_cts,
            model.joints.num_kinematic_cts,
            model.joints.dynamic_cts_offset_total_cts,
            model.joints.kinematic_cts_offset_total_cts,
            # Outputs:
            lambda_j,
        ],
        device=model.device,
    )


def set_joint_state_masked(
    model: ModelKamino,
    world_mask: wp.array,
    src_q: wp.array,
    src_u: wp.array | None,
    dst_q: wp.array,
    dst_q_p: wp.array,
    dst_dq: wp.array | None,
):
    """
    Writes joint state into ``dst_q`` and ``dst_q_p`` from ``src_q``, and optionally
    writes joint velocities into ``dst_dq`` from ``src_u``, for the worlds selected
    by ``world_mask``. Joints whose world is masked out are left untouched.

    Velocities are written only when both ``src_u`` and ``dst_dq`` are provided. When
    either is ``None``, the velocity write is skipped and ``src_q`` / ``dst_q`` are
    passed as placeholders to satisfy the kernel signature.
    """
    write_velocities = src_u is not None and dst_dq is not None
    _src_u = src_u if src_u is not None else src_q
    _dst_dq = dst_dq if dst_dq is not None else dst_q
    wp.launch(
        _set_joint_state_of_select_worlds,
        dim=model.size.sum_of_num_joints,
        inputs=[
            # Inputs:
            write_velocities,
            world_mask,
            model.joints.wid,
            model.joints.coords_offset,
            model.joints.dofs_offset,
            src_q,
            _src_u,
            # Outputs:
            dst_q,
            dst_q_p,
            _dst_dq,
        ],
        device=model.device,
    )


def reset_state_to_model_default(
    model: ModelKamino,
    state_out: StateKamino,
    world_mask: wp.array,
):
    """
    Reset the given `state_out` container to the initial state defined
    in the model, but only for the worlds specified by the `world_mask`.

    Args:
        model (ModelKamino):
            Input model container holding the time-invariant data of the system.
        state_out (StateKamino):
            Output state container to be reset to the model's default state.
        world_mask (wp.array):
            Array of per-world flags indicating which worlds should be reset.\n
            Shape of ``(num_worlds,)`` and type :class:`bool`.
    """
    reset_state_from_bodies_state(
        model,
        state_out,
        world_mask,
        model.bodies.q_i_0,
        model.bodies.u_i_0,
    )


def reset_state_from_bodies_state(
    model: ModelKamino,
    state_out: StateKamino,
    world_mask: wp.array,
    bodies_q: wp.array,
    bodies_u: wp.array,
):
    """
    Resets the state of all bodies in the selected worlds based on their provided state.
    The result is stored in the provided `state_out` container.

    Args:
        model (ModelKamino):
            Input model container holding the time-invariant data of the system.
        state_out (StateKamino):
            Output state container to be reset to the model's default state.
        world_mask (wp.array):
            Array of per-world flags indicating which worlds should be reset.\n
            Shape of ``(num_worlds,)`` and type :class:`bool`.
        bodies_q (wp.array):
            Array of target poses for the rigid bodies of each world.\n
            Shape of ``(num_bodies,)`` and type :class:`transformf`.
        bodies_u (wp.array):
            Array of target twists for the rigid bodies of each world.\n
            Shape of ``(num_bodies,)`` and type :class:`vec6f`.
    """
    # Reset bodies
    wp.launch(
        _reset_body_state_of_select_worlds,
        dim=model.size.sum_of_num_bodies,
        inputs=[
            # Inputs:
            world_mask,
            model.bodies.wid,
            bodies_q,
            bodies_u,
            # Outputs:
            state_out.q_i,
            state_out.u_i,
            state_out.w_i,
            state_out.w_i_e,
        ],
        device=model.device,
    )

    # Reset joints
    wp.launch(
        _reset_joint_state_of_select_worlds,
        dim=model.size.sum_of_num_joints,
        inputs=[
            # Inputs:
            world_mask,
            model.joints.wid,
            model.joints.num_dynamic_cts,
            model.joints.num_kinematic_cts,
            model.joints.coords_offset,
            model.joints.dofs_offset,
            model.joints.dynamic_cts_offset_joint_cts,
            model.joints.kinematic_cts_offset_joint_cts,
            model.joints.q_j_0,
            # Outputs:
            state_out.q_j,
            state_out.q_j_p,
            state_out.dq_j,
            state_out.lambda_j,
        ],
        device=model.device,
    )


def reset_state_from_base_state(
    model: ModelKamino,
    state_out: StateKamino,
    world_mask: wp.array,
    base_q: wp.array,
    base_u: wp.array,
):
    """
    Resets the state of all bodies in the selected worlds based on the state of their
    respective base bodies. The result is stored in the provided `state_out` container.

    More specifically, in each world, the reset operation rigidly transforms the initial pose of the
    system so as to match the target pose of the base body, and sets body poses accordingly.
    Furthermore, the twists of all bodies are set to that of the base body, but transformed to account
    for the relative pose offset.

    Args:
        model (ModelKamino):
            Input model container holding the time-invariant data of the system.
        state_out (StateKamino):
            Output state container to be reset based on the base body states.
        world_mask (wp.array):
            Array of per-world flags indicating which worlds should be reset.\n
            Shape of ``(num_worlds,)`` and type :class:`bool`.
        base_q (wp.array):
            Array of target poses for the base bodies of each world.\n
            Shape of ``(num_worlds,)`` and type :class:`transformf`.
        base_u (wp.array):
            Array of target twists for the base bodies of each world.\n
            Shape of ``(num_worlds,)`` and type :class:`vec6f`.
    """
    # Reset bodies based on base body states
    wp.launch(
        _reset_body_state_from_base,
        dim=model.size.sum_of_num_bodies,
        inputs=[
            # Inputs:
            world_mask,
            model.info.base_body_index,
            model.bodies.wid,
            model.bodies.q_i_0,
            base_q,
            base_u,
            # Outputs:
            state_out.q_i,
            state_out.u_i,
            state_out.w_i,
        ],
        device=model.device,
    )


def reset_select_worlds_to_initial_state(
    model: ModelKamino,
    mask: wp.array,
    data: DataKamino,
    reset_constraints: bool = True,
):
    """
    Reset the state of the selected worlds to the initial state
    defined in the model given an array of per-world flags.

    Args:
        model: Input model container holding the time-invariant data of the system.
        state: Input state container specifying the target state to be reset to.
        mask: Array of per-world flags indicating which worlds should be reset.
        data: Output solver data to be configured for the target state.
        reset_constraints: Whether to reset joint constraint reactions to zero.
    """
    # Reset time
    wp.launch(
        _reset_time_of_select_worlds,
        dim=model.size.num_worlds,
        inputs=[
            # Inputs:
            mask,
            # Outputs:
            data.time.time,
            data.time.steps,
        ],
        device=model.device,
    )

    # Reset bodies
    wp.launch(
        _reset_bodies_of_select_worlds,
        dim=model.size.sum_of_num_bodies,
        inputs=[
            # Inputs:
            mask,
            model.bodies.wid,
            model.bodies.i_I_i,
            model.bodies.inv_i_I_i,
            model.bodies.q_i_0,
            model.bodies.u_i_0,
            # Outputs:
            data.bodies.q_i,
            data.bodies.u_i,
            data.bodies.I_i,
            data.bodies.inv_I_i,
            data.bodies.w_i,
            data.bodies.w_a_i,
            data.bodies.w_j_i,
            data.bodies.w_l_i,
            data.bodies.w_c_i,
            data.bodies.w_e_i,
        ],
        device=model.device,
    )

    # Reset joints
    wp.launch(
        _reset_joints_of_select_worlds,
        dim=model.size.sum_of_num_joints,
        inputs=[
            # Inputs:
            reset_constraints,
            mask,
            model.joints.wid,
            model.joints.dof_type,
            model.joints.num_dynamic_cts,
            model.joints.num_kinematic_cts,
            model.joints.coords_offset,
            model.joints.dofs_offset,
            model.joints.dynamic_cts_offset_joint_cts,
            model.joints.kinematic_cts_offset_joint_cts,
            model.joints.bid_B,
            model.joints.bid_F,
            model.joints.B_r_Bj,
            model.joints.F_r_Fj,
            model.joints.X_Bj,
            model.joints.X_Fj,
            model.joints.q_j_0,
            model.bodies.q_i_0,
            model.bodies.u_i_0,
            data.joints.lambda_j,
            # Outputs:
            data.joints.p_j,
            data.joints.r_j,
            data.joints.dr_j,
            data.joints.q_j,
            data.joints.dq_j,
            data.joints.lambda_j,
        ],
        device=model.device,
    )


def reset_select_worlds_to_state(
    model: ModelKamino,
    state: StateKamino,
    mask: wp.array,
    data: DataKamino,
    reset_constraints: bool = True,
):
    """
    Reset the state of the selected worlds given an array of per-world flags.

    Args:
        model: Input model container holding the time-invariant data of the system.
        state: Input state container specifying the target state to be reset to.
        mask: Array of per-world flags indicating which worlds should be reset.
        data: Output solver data to be configured for the target state.
    """
    # Reset time
    wp.launch(
        _reset_time_of_select_worlds,
        dim=model.size.num_worlds,
        inputs=[
            # Inputs:
            mask,
            # Outputs:
            data.time.time,
            data.time.steps,
        ],
        device=model.device,
    )

    # Reset bodies
    wp.launch(
        _reset_bodies_of_select_worlds,
        dim=model.size.sum_of_num_bodies,
        inputs=[
            # Inputs:
            mask,
            model.bodies.wid,
            model.bodies.i_I_i,
            model.bodies.inv_i_I_i,
            state.q_i,
            state.u_i,
            # Outputs:
            data.bodies.q_i,
            data.bodies.u_i,
            data.bodies.I_i,
            data.bodies.inv_I_i,
            data.bodies.w_i,
            data.bodies.w_a_i,
            data.bodies.w_j_i,
            data.bodies.w_l_i,
            data.bodies.w_c_i,
            data.bodies.w_e_i,
        ],
        device=model.device,
    )

    # Reset joints
    wp.launch(
        _reset_joints_of_select_worlds,
        dim=model.size.sum_of_num_joints,
        inputs=[
            # Inputs:
            reset_constraints,
            mask,
            model.joints.wid,
            model.joints.dof_type,
            model.joints.num_dynamic_cts,
            model.joints.num_kinematic_cts,
            model.joints.coords_offset,
            model.joints.dofs_offset,
            model.joints.dynamic_cts_offset_joint_cts,
            model.joints.kinematic_cts_offset_joint_cts,
            model.joints.bid_B,
            model.joints.bid_F,
            model.joints.B_r_Bj,
            model.joints.F_r_Fj,
            model.joints.X_Bj,
            model.joints.X_Fj,
            model.joints.q_j_0,
            state.q_i,
            state.u_i,
            state.lambda_j,
            # Outputs:
            data.joints.p_j,
            data.joints.r_j,
            data.joints.dr_j,
            data.joints.q_j,
            data.joints.dq_j,
            data.joints.lambda_j,
        ],
        device=model.device,
    )


def get_base_q_from_joint_q_and_body_q(
    model: ModelKamino,
    joint_q: wp.array[float32],
    body_q: wp.array[transformf],
    base_q: wp.array[transformf],
    world_mask: wp.array[bool],
):
    """
    Infer the floating base pose from joint coordinates, if a base joint was set, or from body poses,
    if only a base body was set.

    Args:
        model: Kamino model.
        joint_q: joint coordinates array.
        body_q: body poses array.
        base_q: array of per-world floating base pose, to set from joint_q/body_q as applicable.
        world_mask: Per-world boolean mask, indicating in which worlds to perform the operation.
    """
    wp.launch(
        _get_base_q_from_joint_q_and_body_q,
        dim=model.size.num_worlds,
        inputs=[
            model.info.base_joint_index,
            model.info.base_body_index,
            model.joints.dof_type,
            model.joints.coords_offset,
            joint_q,
            body_q,
            world_mask,
            base_q,
        ],
        device=model.device,
    )


def get_base_u_from_joint_u_and_body_u(
    model: ModelKamino,
    joint_u: wp.array[float32],
    body_u: wp.array[vec6f],
    base_u: wp.array[vec6f],
    world_mask: wp.array[bool],
):
    """
    Infer the floating base velocity from joint velocities, if a base joint was set, or from body velocities,
    if only a base body was set.

    Args:
        model: Kamino model.
        joint_u: joint velocities array.
        body_u: body velocities array.
        base_u: array of per-world floating base velocity, to set from joint_u/body_u as applicable.
        world_mask: Per-world boolean mask, indicating in which worlds to perform the operation.
    """
    wp.launch(
        _get_base_u_from_joint_u_and_body_u,
        dim=model.size.num_worlds,
        inputs=[
            model.info.base_joint_index,
            model.info.base_body_index,
            model.joints.dofs_offset,
            joint_u,
            body_u,
            world_mask,
            base_u,
        ],
        device=model.device,
    )


def set_body_q(
    model: ModelKamino,
    body_q_in: wp.array[transformf],
    body_q_out: wp.array[transformf],
    world_mask: wp.array[bool],
):
    """
    Set the body poses of select worlds to prescribed values.

    Args:
        model: Kamino model.
        body_q_in: prescribed body poses.
        body_q_out: body poses to overwrite with those in body_q_in, in active worlds.
        world_mask: Per-world boolean mask, indicating in which worlds to perform the operation.
    """
    wp.launch(
        _set_body_q,
        dim=model.size.sum_of_num_bodies,
        inputs=[model.bodies.wid, body_q_in, world_mask, body_q_out],
        device=model.device,
    )


def set_floating_base(
    model: ModelKamino,
    base_q: wp.array[transformf] | None,
    base_u: wp.array[vec6f] | None,
    body_q: wp.array[transformf],
    body_u: wp.array[vec6f],
    world_mask: wp.array[bool],
    relative_base_u: bool = False,
):
    """
    Transforms body poses and velocities so as to match a new prescribed floating base pose and
    velocity, while preserving relative body poses and velocities.

    Args:
        model: Kamino model.
        base_q: prescribed base pose (for the base joint if applicable, else for the base body).
                If None, no transformation is applied to match the base pose.
        base_u: prescribed base velocity (for the base joint if applicable, else for the base body).
                If None, no additional velocity is composed to match the base velocity.
        body_q: body poses to update.
        body_u: body velocities to update.
        world_mask: Per-world boolean mask, indicating in which worlds to perform the operation.
        relative_base_u: Boolean indicating whether base_u should be interpreted as expressed relative
                         to the new pose (after transforming so as to match base_q).
    """
    # Early return if nothing to do
    if base_q is None and base_u is None:
        return

    # Compute relative transformation and velocity change applied to base body
    # Note: we also cache the new base body position to avoid a race condition as the base body is updated
    rel_transform = wp.empty(shape=model.size.num_worlds, dtype=transformf, device=model.device)
    rel_velocity = wp.empty(shape=model.size.num_worlds, dtype=vec6f, device=model.device)
    new_base_pos = wp.empty(shape=model.size.num_worlds, dtype=vec3f, device=model.device)
    wp.launch(
        _eval_floating_base_relative_transform,
        dim=model.size.num_worlds,
        inputs=[
            model.info.base_joint_index,
            model.info.base_body_index,
            model.joints.B_r_Bj,
            model.joints.F_r_Fj,
            model.joints.X_Bj,
            model.joints.X_Fj,
            base_q,
            base_u,
            body_q,
            body_u,
            world_mask,
            relative_base_u,
            rel_transform,
            rel_velocity,
            new_base_pos,
        ],
        device=model.device,
    )

    # Apply transformation to all bodies and compose velocities
    wp.launch(
        _apply_floating_base_transform,
        dim=model.size.sum_of_num_bodies,
        inputs=[
            model.bodies.wid,
            rel_transform,
            rel_velocity,
            new_base_pos,
            world_mask,
            body_q,
            body_u,
        ],
        device=model.device,
    )


def reset_joints_state_from_bodies_state(
    model: ModelKamino,
    state: StateKamino,
    world_mask: wp.array[bool],
):
    """
    Reset joint-based components of the state given body poses and velocities, inferring consistent
    joint coordinates and velocities, and setting joint forces to zero.

    Args:
        model: Kamino model.
        state: Kamino state.
        world_mask: Per-world boolean mask, indicating in which worlds to perform the operation.
    """
    wp.launch(
        _reset_joints_state_from_bodies_state,
        dim=model.size.sum_of_num_joints,
        inputs=[
            model.joints.wid,
            model.joints.dof_type,
            model.joints.coords_offset,
            model.joints.dofs_offset,
            model.joints.cts_offset,
            model.joints.bid_B,
            model.joints.bid_F,
            model.joints.B_r_Bj,
            model.joints.F_r_Fj,
            model.joints.X_Bj,
            model.joints.X_Fj,
            model.joints.q_j_0,
            state.q_i,
            state.u_i,
            world_mask,
            state.q_j,
            state.q_j_p,
            state.dq_j,
            state.lambda_j,
        ],
        device=model.device,
    )


def reset_body_velocities(
    model: ModelKamino,
    state: StateKamino,
    world_mask: wp.array[bool],
):
    """
    Reset body velocities in the state to zero.

    Args:
        model: Kamino model.
        state: Kamino state.
        world_mask: Per-world boolean mask, indicating in which worlds to perform the operation.
    """
    wp.launch(
        _reset_body_velocities,
        dim=model.size.sum_of_num_bodies,
        inputs=[model.bodies.wid, world_mask, state.u_i],
        device=model.device,
    )


def reset_body_wrenches(
    model: ModelKamino,
    state: StateKamino,
    world_mask: wp.array[bool],
):
    """
    Reset body wrenches in the state to zero.

    Args:
        model: Kamino model.
        state: Kamino state.
        world_mask: Per-world boolean mask, indicating in which worlds to perform the operation.
    """
    wp.launch(
        _reset_body_wrenches,
        dim=model.size.sum_of_num_bodies,
        inputs=[model.bodies.wid, world_mask, state.w_i, state.w_i_e],
        device=model.device,
    )
