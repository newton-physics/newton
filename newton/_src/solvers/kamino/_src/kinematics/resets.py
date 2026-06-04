# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Provides a set of operations to reset the state of a physics simulation."""

import warp as wp

from ..core.bodies import transform_body_inertial_properties
from ..core.data import DataKamino
from ..core.math import screw, screw_angular, screw_linear
from ..core.model import ModelKamino
from ..core.state import StateKamino
from ..core.types import float32, int32, mat33f, transformf, vec3f, vec6f
from ..kinematics.joints import compute_joint_pose_and_relative_motion, make_write_joint_data

###
# Module interface
###

__all__ = [
    "reset_body_net_wrenches",
    "reset_joint_constraint_reactions",
    "reset_select_worlds_to_initial_state",
    "reset_select_worlds_to_state",
    "reset_state_from_base_state",
    "reset_state_from_bodies_state",
    "reset_state_to_model_default",
    "reset_time",
    "set_joint_state_masked",
]


###
# Module configs
###

wp.set_module_options({"enable_backward": False})


###
# Kernels
###


@wp.kernel
def _reset_time_of_select_worlds(
    # Inputs:
    world_mask: wp.array[int32],
    # Outputs:
    data_time: wp.array[float32],
    data_steps: wp.array[int32],
):
    # Retrieve the world index from the 1D thread index
    wid = wp.tid()

    # Skip resetting time if the world has not been marked for reset
    if world_mask[wid] == 0:
        return

    # Reset both the physical time and step count to zero
    data_time[wid] = 0.0
    data_steps[wid] = 0


@wp.kernel
def _reset_body_state_of_select_worlds(
    # Inputs:
    world_mask: wp.array[int32],
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
    if world_mask[wid] == 0:
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
def _reset_body_state_from_base(
    # Inputs:
    world_mask: wp.array[int32],
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
    if world_mask[wid] == 0:
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
    world_mask: wp.array[int32],
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
    if world_mask[wid] == 0:
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
    world_mask: wp.array[int32],
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
    if world_mask[wid] == 0:
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
    mask: wp.array[int32],
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
    world_mask: wp.array[int32],
    body_wid: wp.array[int32],
    # Outputs:
    body_w_i: wp.array[vec6f],
):
    # Retrieve the body index from the 1D thread grid
    bid = wp.tid()

    # Retrieve the world index for this body
    wid = body_wid[bid]

    # Skip resetting this body if the world has not been marked for reset
    if world_mask[wid] == 0:
        return

    # Zero-out wrenches
    body_w_i[bid] = vec6f(0.0)


@wp.kernel
def _reset_joint_constraint_reactions(
    # Inputs:
    world_mask: wp.array[int32],
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
    if world_mask[wid] == 0:
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
    mask: wp.array[int32],
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
    model_joint_X_j: wp.array[mat33f],
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
    X_j = model_joint_X_j[jid]

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
    p_j, j_r_j, j_q_j, j_u_j = compute_joint_pose_and_relative_motion(T_B_j, T_F_j, u_B_j, u_F_j, B_r_Bj, F_r_Fj, X_j)

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
            An array indicating which worlds are active (1) or skipped (0).\n
            Shape of ``(num_worlds,)`` and type :class:`int32`.
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
            Shape of ``(num_worlds,)`` and type :class:`int32`.
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
            Shape of ``(num_worlds,)`` and type :class:`int32`.
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
            Shape of ``(num_worlds,)`` and type :class:`int32`.
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
            model.joints.X_j,
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
            model.joints.X_j,
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
