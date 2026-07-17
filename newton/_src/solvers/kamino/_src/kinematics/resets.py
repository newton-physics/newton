# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Provides a set of operations to reset the state of a physics simulation."""

from __future__ import annotations

from typing import Any

import warp as wp

from ..core.joints import JointDoFType
from ..core.model import ModelKamino
from ..core.state import StateKamino
from ..kinematics.joints import (
    compute_joint_pose_and_relative_motion,
    convert_angular_vel_to_universal_joint_intermediary_frame,
    get_joint_coords_mapping_function,
)
from ..solvers.fk.kernels import _correct_joint_angle, _correct_joint_quaternion

###
# Module interface
###

__all__ = [
    "get_base_q_from_joint_q_and_body_q",
    "get_base_u_from_joint_u_and_body_u",
    "reset_body_velocities",
    "reset_body_wrenches",
    "reset_joints_state_from_bodies_state",
    "reset_time",
    "set_body_q",
    "set_floating_base",
]


###
# Module configs
###

wp.set_module_options({"enable_backward": False})


###
# Kernels
###


def make_correct_joint_coords(dof_type: JointDoFType):
    @wp.func
    def _correct_joint_coords(
        coords: Any,  # dof_type.coords_storage_type,
        coords_ref: wp.array[wp.float32],
    ) -> Any:  # dof_type.coords_storage_type
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

        elif wp.static(dof_type == JointDoFType.REVOLUTE):  # Correct angle up to +/- 2 pi
            coords[0] = _correct_joint_angle(coords[0], coords_ref[0])

        elif wp.static(dof_type == JointDoFType.SPHERICAL):  # Correct quaternion up to sign
            quat_ref = wp.vec4f(coords_ref[0], coords_ref[1], coords_ref[2], coords_ref[3])
            coords = _correct_joint_quaternion(coords, quat_ref)

        elif wp.static(dof_type == JointDoFType.UNIVERSAL):  # Correct angles up to +/- 2 pi
            coords[0] = _correct_joint_angle(coords[0], coords_ref[0])
            coords[1] = _correct_joint_angle(coords[1], coords_ref[1])

        return coords

    return _correct_joint_coords


def make_compute_and_write_joint_coords(dof_type: JointDoFType):
    """
    Generate a function computing and writing joint coordinates from the relative translation/rotation
    of the follower w.r.t. the base body, in joint frame.
    """
    num_coords = dof_type.num_coords
    assert num_coords > 0

    @wp.func
    def _compute_and_write_joint_coords(
        r_j: wp.vec3f,
        q_j: wp.quatf,
        coords_offset: wp.int32,
        joint_q_ref: wp.array[wp.float32],
        joint_q: wp.array[wp.float32],
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
    Generate a function computing the dof-space joint velocity, from the joint-frame relative
    body velocity (and from the joint-frame relative body orientation, for universal joints).
    """
    num_dofs = dof_type.num_dofs
    assert num_dofs > 0
    dof_axes = dof_type.dofs_axes

    @wp.func
    def _compute_and_write_joint_vel(
        q_j: wp.quatf,
        u_j: wp.spatial_vectorf,
        dofs_offset: wp.int32,
        joint_u: wp.array[wp.float32],
    ):
        # Convert angular velocity to intermediary body frame for universal joint
        if wp.static(dof_type == JointDoFType.UNIVERSAL):
            u_j = convert_angular_vel_to_universal_joint_intermediary_frame(q_j, u_j)

        # Write out joint velocity (=components of relative velocity along unconstrained axes)
        for i in range(num_dofs):
            joint_u[dofs_offset + i] = u_j[dof_axes[i]]

    return _compute_and_write_joint_vel


@wp.func
def _compute_and_write_joint_coords_and_vel(
    dof_type: wp.int32,
    r_j: wp.vec3f,
    q_j: wp.quatf,
    u_j: wp.spatial_vectorf,
    coords_offset: wp.int32,
    dofs_offset: wp.int32,
    joint_q_ref: wp.array[wp.float32],
    joint_q: wp.array[wp.float32],
    joint_u: wp.array[wp.float32],
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


@wp.kernel
def _get_base_q_from_joint_q_and_body_q(
    # Inputs:
    model_base_joint_index: wp.array[wp.int32],
    model_base_body_index: wp.array[wp.int32],
    model_joint_coords_offset: wp.array[wp.int32],
    state_joint_q: wp.array[wp.float32],
    state_body_q: wp.array[wp.transformf],
    world_mask: wp.array[wp.bool],
    # Outputs:
    base_q: wp.array[wp.transformf],
):
    # Get thread id as world id
    wid = wp.tid()

    # Early return based on mask
    if not world_mask[wid]:
        return

    # Read base_q from joint_q if a base joint was set for this world
    base_joint_id = model_base_joint_index[wid]
    if base_joint_id >= 0:
        coords_offset = model_joint_coords_offset[base_joint_id]
        assert model_joint_coords_offset[base_joint_id + 1] - coords_offset == 7  # Base joint must be free
        base_q[wid] = wp.transformf(
            state_joint_q[coords_offset],
            state_joint_q[coords_offset + 1],
            state_joint_q[coords_offset + 2],
            state_joint_q[coords_offset + 3],
            state_joint_q[coords_offset + 4],
            state_joint_q[coords_offset + 5],
            state_joint_q[coords_offset + 6],
        )

    # Otherwise read base_q from body_q if a base body was set for this world
    else:
        base_body_id = model_base_body_index[wid]
        if base_body_id >= 0:
            base_q[wid] = state_body_q[base_body_id]


@wp.kernel
def _get_base_u_from_joint_u_and_body_u(
    # Inputs:
    model_base_joint_index: wp.array[wp.int32],
    model_base_body_index: wp.array[wp.int32],
    model_joint_dofs_offset: wp.array[wp.int32],
    state_joint_u: wp.array[wp.float32],
    state_body_u: wp.array[wp.spatial_vectorf],
    world_mask: wp.array[wp.bool],
    # Outputs:
    base_u: wp.array[wp.spatial_vectorf],
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
        assert model_joint_dofs_offset[base_joint_id + 1] - dofs_offset == 6  # Base joint must be free
        base_u[wid] = wp.spatial_vectorf(
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
        if base_body_id >= 0:
            base_u[wid] = state_body_u[base_body_id]


@wp.kernel
def _set_body_q(
    # Inputs:
    body_world_id: wp.array[wp.int32],
    body_q_in: wp.array[wp.transformf],
    world_mask: wp.array[wp.bool],
    # Outputs:
    body_q_out: wp.array[wp.transformf],
):
    body_id = wp.tid()
    wid = body_world_id[body_id]
    if not world_mask[wid]:
        return
    body_q_out[body_id] = body_q_in[body_id]


@wp.kernel
def _reset_joints_state_from_bodies_state(
    # Inputs
    joint_world_id: wp.array[wp.int32],
    joint_dof_type: wp.array[wp.int32],
    joint_coords_offset: wp.array[wp.int32],
    joint_dofs_offset: wp.array[wp.int32],
    joint_cts_offset: wp.array[wp.int32],
    joint_bid_B: wp.array[wp.int32],
    joint_bid_F: wp.array[wp.int32],
    joint_B_r_Bj: wp.array[wp.vec3f],
    joint_F_r_Fj: wp.array[wp.vec3f],
    joint_X_Bj: wp.array[wp.mat33f],
    joint_X_Fj: wp.array[wp.mat33f],
    joint_q_0: wp.array[wp.float32],
    body_q: wp.array[wp.transformf],
    body_u: wp.array[wp.spatial_vectorf],
    world_mask: wp.array[wp.bool],
    # Outputs
    joint_q: wp.array[wp.float32],
    joint_q_prev: wp.array[wp.float32],
    joint_u: wp.array[wp.float32],
    joint_lambda: wp.array[wp.float32],
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
    T_B = wp.transform_identity(dtype=wp.float32)
    u_B = wp.spatial_vectorf(0.0)
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
    body_world_id: wp.array[wp.int32],
    world_mask: wp.array[wp.bool],
    # Outputs
    body_u: wp.array[wp.spatial_vectorf],
):
    # Get thread id as body id
    body_id = wp.tid()

    # Early return based on mask
    wid = body_world_id[body_id]
    if not world_mask[wid]:
        return

    # Reset velocities to zero
    body_u[body_id] = wp.spatial_vectorf(0.0)


@wp.kernel
def _reset_body_wrenches(
    # Inputs
    body_world_id: wp.array[wp.int32],
    world_mask: wp.array[wp.bool],
    # Outputs
    body_w: wp.array[wp.spatial_vectorf],
    body_w_e: wp.array[wp.spatial_vectorf],
):
    # Get thread id as body id
    body_id = wp.tid()

    # Early return based on mask
    wid = body_world_id[body_id]
    if not world_mask[wid]:
        return

    # Reset wrenches to zero
    body_w[body_id] = wp.spatial_vectorf(0.0)
    body_w_e[body_id] = wp.spatial_vectorf(0.0)


@wp.kernel
def _reset_time_of_select_worlds(
    # Inputs:
    world_mask: wp.array[wp.bool],
    # Outputs:
    data_time: wp.array[wp.float32],
    data_steps: wp.array[wp.int32],
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
def _eval_floating_base_relative_transform(
    # Inputs:
    model_base_joint_index: wp.array[wp.int32],
    model_base_body_index: wp.array[wp.int32],
    joint_B_r_Bj: wp.array[wp.vec3f],
    joint_F_r_Fj: wp.array[wp.vec3f],
    joint_X_Bj: wp.array[wp.mat33f],
    joint_X_Fj: wp.array[wp.mat33f],
    base_q: wp.array[wp.transformf],  # None also supported
    base_u: wp.array[wp.spatial_vectorf],  # None also supported
    body_q: wp.array[wp.transformf],
    body_u: wp.array[wp.spatial_vectorf],
    world_mask: wp.array[wp.bool],
    relative_base_u: wp.bool,
    # Outputs:
    rel_transform: wp.array[wp.transformf],
    rel_velocity: wp.array[wp.spatial_vectorf],
    new_base_pos: wp.array[wp.vec3f],
):
    # Get thread id as world id
    wid = wp.tid()

    # Early return based on mask
    if not world_mask[wid]:
        return

    # Determine new pose of the base body (= follower of the base joint if there is a base joint)
    base_joint_id = model_base_joint_index[wid]
    base_body_id = model_base_body_index[wid]
    if base_body_id < 0:  # No floating base prescribed
        return
    base_body_curr_pose = body_q[base_body_id]
    if not base_q:  # No prescribed base_q: take new base body pose as its current pose
        base_body_pose = base_body_curr_pose
    elif base_joint_id >= 0:  # If there is a base joint, base_q is the transform in joint frame
        # body_q_B * T_B * base_q = body_q_F * T_F, and body_q_B = identity for a unary joint
        # This gives body_q_F = T_B * base_q * T_F ^-1
        r_B = joint_B_r_Bj[base_joint_id]
        r_F = joint_F_r_Fj[base_joint_id]
        X_B = joint_X_Bj[base_joint_id]
        X_F = joint_X_Fj[base_joint_id]
        T_B = wp.transformf(r_B, wp.quat_from_matrix(X_B))
        T_F = wp.transformf(r_F, wp.quat_from_matrix(X_F))
        T_F_inv = wp.transform_inverse(T_F)
        base_body_pose = wp.transform_multiply(wp.transform_multiply(T_B, base_q[wid]), T_F_inv)
    else:  # Directly interpret base_q as the new base body pose if base body but not base joint
        base_body_pose = base_q[wid]
    new_base_pos[wid] = wp.transform_get_translation(base_body_pose)

    # Determine relative transform to apply, from current to target base body pose
    if not base_q:
        T_rel = wp.transform_identity(wp.float32)
        # Ensure we get a bit-accurate identity (although the formula below would yield the identity)
    else:
        T_rel = wp.transform_multiply(base_body_pose, wp.transform_inverse(base_body_curr_pose))
    rel_transform[wid] = T_rel

    # Determine new velocity of the base body
    if not base_u:  # No prescribed base_u: use zero additional relative velocity to apply to the base
        rel_velocity[wid] = wp.spatial_vectorf(0.0)
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
        if relative_base_u:
            q_F = wp.transform_get_rotation(base_body_curr_pose)
        else:
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
    rel_velocity[wid] = wp.spatial_vectorf(*v_rel, *omega_rel)


@wp.kernel
def _apply_floating_base_transform(
    # Inputs:
    model_base_body_index: wp.array[wp.int32],
    body_world_id: wp.array[wp.int32],
    rel_transform: wp.array[wp.transformf],
    rel_velocity: wp.array[wp.spatial_vectorf],
    new_base_pos: wp.array[wp.vec3f],
    world_mask: wp.array[wp.bool],
    # Outputs:
    body_q: wp.array[wp.transformf],
    body_u: wp.array[wp.spatial_vectorf],
):
    # Get thread id as body id
    body_id = wp.tid()

    # Early return based on mask or absence of floating base
    wid = body_world_id[body_id]
    if not world_mask[wid] or model_base_body_index[wid] < 0:
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
    body_u[body_id] = wp.spatial_vectorf(*body_v_new, *body_omega_new)


###
# Launchers
###


def reset_time(
    model: ModelKamino,
    time: wp.array[wp.float32],
    steps: wp.array[wp.int32],
    world_mask: wp.array[wp.bool],
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


def get_base_q_from_joint_q_and_body_q(
    model: ModelKamino,
    joint_q: wp.array[wp.float32],
    body_q: wp.array[wp.transformf],
    base_q: wp.array[wp.transformf],
    world_mask: wp.array[wp.bool],
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
    joint_u: wp.array[wp.float32],
    body_u: wp.array[wp.spatial_vectorf],
    base_u: wp.array[wp.spatial_vectorf],
    world_mask: wp.array[wp.bool],
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
    body_q_in: wp.array[wp.transformf],
    body_q_out: wp.array[wp.transformf],
    world_mask: wp.array[wp.bool],
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
    base_q: wp.array[wp.transformf] | None,
    base_u: wp.array[wp.spatial_vectorf] | None,
    body_q: wp.array[wp.transformf],
    body_u: wp.array[wp.spatial_vectorf],
    world_mask: wp.array[wp.bool],
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
    rel_transform = wp.empty(shape=model.size.num_worlds, dtype=wp.transformf, device=model.device)
    rel_velocity = wp.empty(shape=model.size.num_worlds, dtype=wp.spatial_vectorf, device=model.device)
    new_base_pos = wp.empty(shape=model.size.num_worlds, dtype=wp.vec3f, device=model.device)
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
            model.info.base_body_index,
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
    world_mask: wp.array[wp.bool],
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
    world_mask: wp.array[wp.bool],
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
    world_mask: wp.array[wp.bool],
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
