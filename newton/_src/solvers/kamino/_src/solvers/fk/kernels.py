# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Defines the Warp kernels used by the Forward Kinematics solver."""

from functools import cache

import warp as wp

from ...core.joints import JointActuationType, JointDoFType
from ...core.math import (
    TWO_PI,
    G_of,
    quat_left_jacobian_inverse,
    quat_log,
    squared_norm,
    unit_quat_apply,
    unit_quat_apply_jacobian,
    unit_quat_conj_apply,
    unit_quat_conj_apply_jacobian,
    unit_quat_conj_to_rotation_matrix,
)
from ...core.types import vec6f
from ...kinematics.joints import get_joint_coords_mapping_function
from ...linalg.sparse_matrix import BlockDType
from .types import FKJointDoFType

###
# Module interface
###

__all__ = [
    "_apply_line_search_step",
    "_correct_actuator_coords",
    "_eval_actuator_coords",
    "_eval_body_velocities",
    "_eval_fk_actuated_dofs_or_coords",
    "_eval_incremental_target_actuator_coords",
    "_eval_linear_combination",
    "_eval_position_control_transformations",
    "_eval_rhs",
    "_eval_stepped_state",
    "_eval_target_constraint_velocities",
    "_eval_unit_quaternion_constraints",
    "_eval_unit_quaternion_constraints_jacobian",
    "_eval_unit_quaternion_constraints_sparse_jacobian",
    "_line_search_check",
    "_newton_check",
    "_reset_state",
    "_reset_state_base_q",
    "_update_cg_tolerance_kernel",
    "create_1d_tile_based_kernels",
    "create_2d_tile_based_kernels",
    "create_eval_joint_constraints_jacobian_kernel",
    "create_eval_joint_constraints_kernel",
    "create_eval_joint_constraints_sparse_jacobian_kernel",
    "create_eval_min_num_iterations_kernel",
    "read_quat_from_array",
]


###
# Module configs
###

wp.set_module_options({"enable_backward": False})


###
# Types
###


class block_type(BlockDType(dtype=wp.float32, shape=(7,)).warp_type):
    pass


###
# Functions
###


@wp.func
def read_quat_from_array(array: wp.array[wp.float32], offset: int, normalize: bool) -> wp.quatf:
    """
    Utility function to read a quaternion from a flat array
    """
    q = wp.quatf(array[offset], array[offset + 1], array[offset + 2], array[offset + 3])
    if normalize:
        return wp.normalize(q)
    return q


###
# Kernels
###


@wp.kernel
def _reset_state(
    # Inputs
    num_bodies: wp.array[wp.int32],
    first_body_id: wp.array[wp.int32],
    bodies_q_0_flat: wp.array[wp.float32],
    world_mask: wp.array[wp.int32],
    # Outputs
    bodies_q_flat: wp.array[wp.float32],
):
    """
    A kernel resetting the fk state (body poses) to the reference state

    Inputs:
        num_bodies: Num bodies per world
        first_body_id: First body id per world
        bodies_q_0_flat: Reference state, flattened
        world_mask: Per-world flag to perform the operation (0 = skip)
    Outputs:
        bodies_q_flat: State to reset, flattened
    """
    wd_id, state_id_loc = wp.tid()  # Thread indices (= world index, state index)
    rb_id_loc = state_id_loc // 7
    if wd_id < num_bodies.shape[0] and world_mask[wd_id] != 0 and rb_id_loc < num_bodies[wd_id]:
        state_id_tot = 7 * first_body_id[wd_id] + state_id_loc
        bodies_q_flat[state_id_tot] = bodies_q_0_flat[state_id_tot]


@wp.kernel
def _reset_state_base_q(
    # Inputs
    base_joint_id: wp.array[wp.int32],
    base_q: wp.array[wp.transformf],
    joints_bid_F: wp.array[wp.int32],
    joints_X: wp.array[wp.mat33f],
    joints_B_r_B: wp.array[wp.vec3f],
    joints_F_r_F: wp.array[wp.vec3f],
    num_bodies: wp.array[wp.int32],
    first_body_id: wp.array[wp.int32],
    bodies_q_0: wp.array[wp.transformf],
    world_mask: wp.array[wp.int32],
    # Outputs
    bodies_q: wp.array[wp.transformf],
):
    """
    A kernel resetting the fk state (body poses) to a rigid transformation of the reference state,
    computed so that the base body is aligned on its prescribed pose.

    Inputs:
        base_joint_id: Base joint id per world (-1 = None)
        base_q: Base body pose per world, in base joint coordinates
        joints_bid_F: Joint follower body id
        joints_X: Joint frame (local axes, valid both on base and follower)
        joints_B_r_B: Joint local position on base body
        joints_F_r_F: Joint local position on follower body
        num_bodies: Num bodies per world
        first_body_id: First body id per world
        bodies_q_0: Reference body poses
        world_mask: Per-world flag to perform the operation (0 = skip)
    Outputs:
        bodies_q: Body poses to reset
    """
    wd_id, rb_id_loc = wp.tid()  # Thread indices (= world index, body index)
    if wd_id < num_bodies.shape[0] and world_mask[wd_id] != 0 and rb_id_loc < num_bodies[wd_id]:
        # Worlds without base joint: just copy the reference pose
        rb_id_tot = first_body_id[wd_id] + rb_id_loc
        base_jt_id = base_joint_id[wd_id]
        body_q_0 = bodies_q_0[rb_id_tot]
        if base_jt_id < 0:
            bodies_q[rb_id_tot] = body_q_0
            return

        # Read memory
        base_q_wd = base_q[wd_id]
        bid_F = joints_bid_F[base_jt_id]
        X = joints_X[base_jt_id]
        x_B = joints_B_r_B[base_jt_id]
        x_F = joints_F_r_F[base_jt_id]
        body_q_F_0 = bodies_q_0[bid_F]

        # Compute pose of the base body (follower of the base joint) given current joint coordinates
        # Note: the relative transform from base to follower can be written
        # t_jt = X^T * R_B^T * (c_F + R_F * x_F - c_B - R_B * x_B)
        # q_jt = X^T * R_B^T * R_F * X
        # We invert these equations, using R_B = I and c_B = 0 (base body = world)
        t_jt = wp.transform_get_translation(base_q_wd)
        q_jt = wp.transform_get_rotation(base_q_wd)
        q_X = wp.quat_from_matrix(X)
        q_F = q_X * q_jt * wp.quat_inverse(q_X)
        c_F = wp.quat_rotate(q_X, t_jt) - wp.quat_rotate(q_F, x_F) + x_B
        body_q_F = wp.transformf(c_F, q_F)

        # Compute the transform that was applied to the base body relative to the base pose,
        # and apply that transform to all rigid bodies
        transform_tot = wp.transform_multiply(body_q_F, wp.transform_inverse(body_q_F_0))
        bodies_q[rb_id_tot] = wp.transform_multiply(transform_tot, body_q_0)


@wp.kernel
def _eval_fk_actuated_dofs_or_coords(
    # Inputs
    model_base_dofs: wp.array[wp.float32],
    model_actuated_dofs: wp.array[wp.float32],
    actuated_dofs_map: wp.array[wp.int32],
    # Outputs
    fk_actuated_dofs: wp.array[wp.float32],
):
    """
    A kernel mapping actuated and base dofs/coordinates of the main model to actuated dofs/coordinates of the fk model,
    which has a modified version of the joints, notably actuated free joints to control floating bases.

    This uses a map from fk to model dofs/cords, that has >= 0 indices for fk dofs/coords that correspond to
    main model actuated dofs/coords, and negative indices for base dofs/coords (base dof/coord i is stored as -i - 1)

    Inputs:
        model_base_dofs:
            Base dofs or coordinates of the main model (as a flat vector with 6 dofs or 7 coordinates per world)
        model_actuated_dofs:
            Actuated dofs/coords of the main model
        actuated_dofs_map:
            Map of fk to main model actuated/base dofs/coords
    Outputs:
        fk_actuated_dofs: Actuated dofs or coordinates of the fk model
    """

    # Retrieve the thread index (= fk actuated dof or coordinate index)
    # Note: we use "dof" in variables naming to mean either dof or coordinate
    fk_dof_id = wp.tid()

    if fk_dof_id < fk_actuated_dofs.shape[0]:
        model_dof_id = actuated_dofs_map[fk_dof_id]
        if model_dof_id >= 0:
            fk_actuated_dofs[fk_dof_id] = model_actuated_dofs[model_dof_id]
        else:  # Base dofs/coordinates are encoded as negative indices
            base_dof_id = -(model_dof_id + 1)  # Recover base dof/coord id
            fk_actuated_dofs[fk_dof_id] = model_base_dofs[base_dof_id]


def _make_typed_joint_transform_to_coords_func(dof_type: JointDoFType):
    """Factory returning a function extracting joint coords from joint transform, for a single joint type."""
    num_coords = dof_type.num_coords

    @wp.func
    def _typed_joint_transform_to_coords(
        pos_rel: wp.vec3f,
        q_rel: wp.quatf,
        offset: wp.int32,
        output: wp.array[wp.float32],
    ):
        coords = wp.static(get_joint_coords_mapping_function(dof_type))(pos_rel, q_rel)
        for i in range(num_coords):
            output[offset + i] = coords[i]

    return _typed_joint_transform_to_coords


@wp.func
def _joint_transform_to_coords(
    dof_type: wp.int32,
    pos_rel: wp.vec3f,
    q_rel: wp.quatf,
    offset: wp.int32,
    output: wp.array[wp.float32],
):
    """Function extracting joint coordinates from the joint transform, and writing them out in an array."""
    if dof_type == FKJointDoFType.CARTESIAN:
        wp.static(_make_typed_joint_transform_to_coords_func(JointDoFType.CARTESIAN))(pos_rel, q_rel, offset, output)
    elif dof_type == FKJointDoFType.CYLINDRICAL:
        wp.static(_make_typed_joint_transform_to_coords_func(JointDoFType.CYLINDRICAL))(pos_rel, q_rel, offset, output)
    elif dof_type == FKJointDoFType.FREE:
        wp.static(_make_typed_joint_transform_to_coords_func(JointDoFType.FREE))(pos_rel, q_rel, offset, output)
    elif dof_type == FKJointDoFType.PRISMATIC:
        wp.static(_make_typed_joint_transform_to_coords_func(JointDoFType.PRISMATIC))(pos_rel, q_rel, offset, output)
    elif dof_type == FKJointDoFType.REVOLUTE:
        wp.static(_make_typed_joint_transform_to_coords_func(JointDoFType.REVOLUTE))(pos_rel, q_rel, offset, output)
    elif dof_type == FKJointDoFType.SPHERICAL:
        wp.static(_make_typed_joint_transform_to_coords_func(JointDoFType.SPHERICAL))(pos_rel, q_rel, offset, output)
    elif dof_type == FKJointDoFType.UNIVERSAL:
        wp.static(_make_typed_joint_transform_to_coords_func(JointDoFType.UNIVERSAL))(pos_rel, q_rel, offset, output)


@wp.kernel
def _eval_actuator_coords(
    num_joints: wp.array[wp.int32],
    first_joint_id: wp.array[wp.int32],
    joints_dof_type: wp.array[wp.int32],
    joints_bid_B: wp.array[wp.int32],
    joints_bid_F: wp.array[wp.int32],
    joints_X: wp.array[wp.mat33f],
    joints_B_r_B: wp.array[wp.vec3f],
    joints_F_r_F: wp.array[wp.vec3f],
    bodies_q: wp.array[wp.transformf],
    actuated_coord_offsets: wp.array[wp.int32],
    actuators_q: wp.array[wp.float32],
):
    """
    A kernel evaluating effective actuator coordinates based on body poses.

    Inputs:
        num_joints: Num joints per world.
        first_joint_id: First joint id per world.
        joints_dof_type: Joint dof type (i.e. revolute, spherical, ...).
        joints_bid_B: Joint base body id.
        joints_bid_F: Joint follower body id.
        joints_X: Joint frame (local axes, valid both on base and follower).
        joints_B_r_B: Joint local position on base body.
        joints_F_r_F: Joint local position on follower body.
        bodies_q: Body poses.
        actuated_coord_offsets: Joint first actuated coordinate id, among all actuated coordinates in all worlds.
    Outputs:
        actuators_q: Actuator coordinates.
    """
    # Retrieve the thread index (= world index, joint index within world)
    wd_id, jt_id_loc = wp.tid()

    # Get global joint index
    if jt_id_loc >= num_joints[wd_id]:
        return
    jt_id = first_joint_id[wd_id] + jt_id_loc

    # Get joint actuated coords size and offset
    coord_id = actuated_coord_offsets[jt_id]
    num_coords = actuated_coord_offsets[jt_id + 1] - coord_id
    if num_coords == 0:
        return

    # Get joint dof type, local positions and local orientation
    dof_type = joints_dof_type[jt_id]
    x_base = joints_B_r_B[jt_id]
    x_follower = joints_F_r_F[jt_id]
    q_X = wp.quat_from_matrix(joints_X[jt_id])

    # Get base and follower transformations
    base_id = joints_bid_B[jt_id]
    if base_id < 0:
        c_base = wp.vec3f(0.0, 0.0, 0.0)
        q_base = wp.quatf(0.0, 0.0, 0.0, 1.0)
    else:
        c_base = wp.transform_get_translation(bodies_q[base_id])
        q_base = wp.transform_get_rotation(bodies_q[base_id])
    follower_id = joints_bid_F[jt_id]
    c_follower = wp.transform_get_translation(bodies_q[follower_id])
    q_follower = wp.transform_get_rotation(bodies_q[follower_id])

    # Compute relative pose of follower body in joint frame of base body
    pos_base = c_base + wp.quat_rotate(q_base, x_base)
    pos_follower = c_follower + wp.quat_rotate(q_follower, x_follower)
    ori_base_T = wp.quat_inverse(q_base * q_X)
    ori_follower = q_follower * q_X
    pos_rel = wp.quat_rotate(ori_base_T, pos_follower - pos_base)
    q_rel = ori_base_T * ori_follower

    # Extract joint coordinates from relative pose
    _joint_transform_to_coords(dof_type, pos_rel, q_rel, coord_id, actuators_q)


@wp.func
def _correct_joint_angle(angle: wp.float32, angle_ref: wp.float32) -> wp.float32:
    """Function adding multiples of 2 pi to an angle, so that it is the closest to a reference."""
    return angle + wp.round((angle_ref - angle) / TWO_PI) * TWO_PI


@wp.func
def _correct_joint_quaternion(quat: wp.vec4f, quat_ref: wp.vec4f) -> wp.vec4f:
    """Function flipping the sign of a quaternion if needed, so it is the closest to a reference."""
    if squared_norm(quat + quat_ref) < squared_norm(quat - quat_ref):
        return -quat
    return quat


@wp.kernel
def _correct_actuator_coords(
    # Inputs
    actuated_coord_offsets: wp.array[wp.int32],
    joints_dof_type: wp.array[wp.int32],
    actuators_q_ref: wp.array[wp.float32],
    # Outputs
    actuators_q: wp.array[wp.float32],
):
    """
    A kernel correcting actuator coordinates w.r.t. reference coordinates, ensuring that
    angles are within +/- 2 pi of the reference, and quaternions are closer to the reference
    than to its opposite.

    Inputs:
        actuated_coord_offsets: Joint first actuated coordinate id, among all actuated coordinates in all worlds.
        joints_dof_type: Joint dof type (i.e. revolute, spherical, ...).
        actuators_q_ref: Reference actuator coordinates.
    Outputs:
        actuators_q: Actuator coordinates to correct w.r.t. the reference.
    """
    # Retrieve the thread index (= joint index)
    joint_id = wp.tid()

    # Get joint actuated coords size and offset
    coord_id = actuated_coord_offsets[joint_id]
    num_coords = actuated_coord_offsets[joint_id + 1] - coord_id
    if num_coords == 0:
        return

    # Apply correction based on DoFs
    dof_type = joints_dof_type[joint_id]
    if (
        dof_type == FKJointDoFType.CARTESIAN or dof_type == FKJointDoFType.FIXED or dof_type == FKJointDoFType.PRISMATIC
    ):  # No correction needed
        return
    elif dof_type == FKJointDoFType.CYLINDRICAL:  # Correct angle up to +/- 2 pi
        angle = actuators_q[coord_id + 1]
        angle_ref = actuators_q_ref[coord_id + 1]
        actuators_q[coord_id + 1] = _correct_joint_angle(angle, angle_ref)
    elif dof_type == FKJointDoFType.FREE:  # Correct quaternion up to sign
        quat = wp.vec4f(
            actuators_q[coord_id + 3], actuators_q[coord_id + 4], actuators_q[coord_id + 5], actuators_q[coord_id + 6]
        )
        quat_ref = wp.vec4f(
            actuators_q_ref[coord_id + 3],
            actuators_q_ref[coord_id + 4],
            actuators_q_ref[coord_id + 5],
            actuators_q_ref[coord_id + 6],
        )
        quat_corrected = _correct_joint_quaternion(quat, quat_ref)
        for i in range(4):
            actuators_q[coord_id + 3 + i] = quat_corrected[i]
    elif dof_type == FKJointDoFType.REVOLUTE:  # Correct angle up to +/- 2 pi
        angle = actuators_q[coord_id]
        angle_ref = actuators_q_ref[coord_id]
        actuators_q[coord_id] = _correct_joint_angle(angle, angle_ref)
    elif dof_type == FKJointDoFType.SPHERICAL:  # Correct quaternion up to sign
        quat = wp.vec4f(
            actuators_q[coord_id], actuators_q[coord_id + 1], actuators_q[coord_id + 2], actuators_q[coord_id + 3]
        )
        quat_ref = wp.vec4f(
            actuators_q_ref[coord_id],
            actuators_q_ref[coord_id + 1],
            actuators_q_ref[coord_id + 2],
            actuators_q_ref[coord_id + 3],
        )
        quat_corrected = _correct_joint_quaternion(quat, quat_ref)
        for i in range(4):
            actuators_q[coord_id + i] = quat_corrected[i]
    elif dof_type == FKJointDoFType.UNIVERSAL:  # Correct angles up to +/- 2 pi
        angle = actuators_q[coord_id]
        angle_ref = actuators_q_ref[coord_id]
        actuators_q[coord_id] = _correct_joint_angle(angle, angle_ref)
        angle = actuators_q[coord_id + 1]
        angle_ref = actuators_q_ref[coord_id + 1]
        actuators_q[coord_id + 1] = _correct_joint_angle(angle, angle_ref)
    else:
        assert False, "Unexpected actuator dof type"  # noqa: B011


@wp.kernel
def _eval_incremental_target_actuator_coords(
    # Inputs
    world_actuated_coord_offsets: wp.array[wp.int32],
    actuators_q_prev: wp.array[wp.float32],
    actuators_q_next: wp.array[wp.float32],
    delta_q_max: wp.array[wp.float32],
    iteration: wp.array[wp.int32],
    world_mask: wp.array[wp.int32],
    # Outputs
    actuators_q_curr: wp.array[wp.float32],
):
    """
    A kernel evaluating the actuator coordinates to solve for given the Newton iteration
    number and the target actuator coordinates, by interpolating between initial and target
    coordinates if necessary to avoid too large jumps per iteration.

    Inputs:
        world_actuated_coord_offsets: World first actuated coordinate id, among all actuated coordinates in all worlds.
        actuators_q_prev: Previous actuator coordinates.
        actuators_q_next: Next actuator coordinates (= target).
        delta_q_max: Maximal allowed step per coordinate, for one Newton iteration.
        iteration: Current Newton iteration per world.
        world_mask: Per-world flag to perform the computation (0 = skip).
    Outputs:
        actuators_q_curr: Actuator coordinates to use as target for the current iteration (= incremental target).
    """
    # Retrieve the thread index (= world index, coordinate index in world)
    wd_id, coord_id_loc = wp.tid()

    # Early return based on world mask
    if world_mask[wd_id] == 0:
        return

    # Read data
    coord_id = world_actuated_coord_offsets[wd_id] + coord_id_loc
    if coord_id >= world_actuated_coord_offsets[wd_id + 1]:
        return
    q_prev = actuators_q_prev[coord_id]
    q_next = actuators_q_next[coord_id]
    delta = delta_q_max[coord_id]
    it = iteration[wd_id]

    # Interpolate coordinate
    sign = wp.where(q_prev > q_next, -1.0, 1.0)
    actuators_q_curr[coord_id] = sign * wp.min(sign * q_prev + wp.float32(it + 1) * delta, sign * q_next)


@wp.func
def min_iteration_op(v0: wp.float32, v1: wp.float32, d: wp.float32) -> wp.int32:
    """Function returning the floor of the absolute division of (v1 - v0) by d."""
    eps = 1e-7  # Epsilon to ensure that we round down if |v1 - v0| = d
    return wp.int32(wp.floor((wp.abs(v1 - v0) - eps) / d))


@wp.func
def less_than_op(i: wp.int32, threshold: wp.int32) -> wp.int32:
    """Thresholding operation."""
    return wp.where(i < threshold, 1, 0)


@wp.func
def mul_mask(mask: wp.int32, value: wp.int32):
    """Return value if mask is positive, else 0"""
    return wp.where(mask > 0, value, 0)


@cache
def create_eval_min_num_iterations_kernel(TILE_SIZE: int):
    @wp.kernel
    def _eval_min_num_iterations(
        # Inputs
        world_actuated_coord_offsets: wp.array[wp.int32],
        actuators_q_prev: wp.array[wp.float32],
        actuators_q_next: wp.array[wp.float32],
        delta_q_max: wp.array[wp.float32],
        # Outputs
        min_iterations: wp.array[wp.int32],
    ):
        """
        A kernel evaluating the minimal number of Newton iterations needed per world, for incremental steps
        in actuator coordinates to have converged to the target coordinates.

        Inputs:
            world_actuated_coord_offsets: World first actuated coordinate id, among all actuated coordinates in all worlds.
            actuators_q_prev: Previous actuator coordinates.
            actuators_q_next: Next actuator coordinates (= target).
            delta_q_max: Maximal allowed step per coordinate, for one Newton iteration.
        Outputs:
            min_iterations: Minimum iterations needed per world.
        """
        # Retrieve the thread index (= world index, input tile index, thread index in block)
        wd_id, i, tid = wp.tid()

        # Read data
        world_offset = world_actuated_coord_offsets[wd_id]
        next_world_offset = world_actuated_coord_offsets[wd_id + 1]
        offset = world_offset + i * TILE_SIZE
        q_prev = wp.tile_load(actuators_q_prev, shape=TILE_SIZE, offset=offset)
        q_next = wp.tile_load(actuators_q_next, shape=TILE_SIZE, offset=offset)
        delta = wp.tile_load(delta_q_max, shape=TILE_SIZE, offset=offset)

        # Compute min iterations count per coordinate, and take the maximum per world
        min_it = wp.tile_map(min_iteration_op, q_prev, q_next, delta)
        if offset + TILE_SIZE > next_world_offset:  # Mask out values from next world if needed
            mask = wp.tile_map(less_than_op, wp.tile_arange(TILE_SIZE, dtype=wp.int32), next_world_offset - offset)
            min_it = wp.tile_map(mul_mask, mask, min_it)
        min_it_max = wp.tile_max(min_it)[0]
        if tid == 0:
            wp.atomic_max(min_iterations, wd_id, min_it_max)

    return _eval_min_num_iterations


@wp.kernel
def _eval_position_control_transformations(
    # Inputs
    joints_dof_type: wp.array[wp.int32],
    joints_act_type: wp.array[wp.int32],
    actuated_coords_offset: wp.array[wp.int32],
    joints_X: wp.array[wp.mat33f],
    actuators_q: wp.array[wp.float32],
    normalize_quaternions: wp.bool,
    # Outputs
    pos_control_transforms: wp.array[wp.transformf],
):
    """
    A kernel computing a transformation per joint corresponding to position-control parameters
    More specifically, this is the identity (no translation, no rotation) for passive joints
    and a transformation corresponding to joint generalized coordinates for actuators

    The translation part is expressed in joint frame (e.g., translation is along [1,0,0] for a prismatic joint)
    The rotation part is expressed in body frame (e.g., rotation is about X[:,0] for a revolute joint)

    Inputs:
        joints_dof_type: Joint dof type (i.e. revolute, spherical, ...)
        joints_act_type: Joint actuation type (i.e. passive or actuated)
        actuated_coords_offset: Joint first actuated coordinate id, among all actuated coordinates in all worlds
        joints_X: Joint frame (local axes, valid both on base and follower)
        actuators_q: Actuated coordinates
        normalize_quaternions: Whether to normalize quaternions in actuators_q (else unit length is assumed)
    Outputs:
        pos_control_transforms: Joint position-control transformation
    """

    # Retrieve the thread index (= joint index)
    jt_id = wp.tid()

    if jt_id < joints_dof_type.shape[0]:
        # Retrieve the joint model data
        dof_type_j = joints_dof_type[jt_id]
        act_type_j = joints_act_type[jt_id]
        X = joints_X[jt_id]

        # Initialize transform to identity (already covers the passive case)
        t = wp.vec3f(0.0, 0.0, 0.0)
        q = wp.quatf(0.0, 0.0, 0.0, 1.0)

        # In the actuated case, set translation/rotation as per joint generalized coordinates
        if act_type_j != JointActuationType.PASSIVE:
            offset_q_j = actuated_coords_offset[jt_id]
            if dof_type_j == FKJointDoFType.CARTESIAN:
                t[0] = actuators_q[offset_q_j]
                t[1] = actuators_q[offset_q_j + 1]
                t[2] = actuators_q[offset_q_j + 2]
            elif dof_type_j == FKJointDoFType.CYLINDRICAL:
                t[0] = actuators_q[offset_q_j]
                q = wp.quat_from_axis_angle(wp.vec3f(X[0, 0], X[1, 0], X[2, 0]), actuators_q[offset_q_j + 1])
            elif dof_type_j == FKJointDoFType.FIXED:
                pass  # No dofs to apply
            elif dof_type_j == FKJointDoFType.FREE:
                t[0] = actuators_q[offset_q_j]
                t[1] = actuators_q[offset_q_j + 1]
                t[2] = actuators_q[offset_q_j + 2]
                q_X = wp.quat_from_matrix(X)
                q_loc = read_quat_from_array(actuators_q, offset_q_j + 3, normalize_quaternions)
                q = q_X * q_loc * wp.quat_inverse(q_X)
            elif dof_type_j == FKJointDoFType.PRISMATIC:
                t[0] = actuators_q[offset_q_j]
            elif dof_type_j == FKJointDoFType.REVOLUTE:
                q = wp.quat_from_axis_angle(wp.vec3f(X[0, 0], X[1, 0], X[2, 0]), actuators_q[offset_q_j])
            elif dof_type_j == FKJointDoFType.SPHERICAL:
                q_X = wp.quat_from_matrix(X)
                q_loc = read_quat_from_array(actuators_q, offset_q_j, normalize_quaternions)
                q = q_X * q_loc * wp.quat_inverse(q_X)
            elif dof_type_j == FKJointDoFType.UNIVERSAL:
                q_x = wp.quat_from_axis_angle(wp.vec3f(X[0, 0], X[1, 0], X[2, 0]), actuators_q[offset_q_j])
                q_y = wp.quat_from_axis_angle(wp.vec3f(X[0, 1], X[1, 1], X[2, 1]), actuators_q[offset_q_j + 1])
                q = q_x * q_y
            else:
                assert False, "Unexpected actuator dof type"  # noqa: B011

        # Write out transformation
        pos_control_transforms[jt_id] = wp.transformf(t, q)


@wp.kernel
def _eval_unit_quaternion_constraints(
    # Inputs
    num_bodies: wp.array[wp.int32],
    first_body_id: wp.array[wp.int32],
    bodies_q: wp.array[wp.transformf],
    world_mask: wp.array[wp.int32],
    # Outputs
    constraints: wp.array2d[wp.float32],
):
    """
        A kernel computing unit norm quaternion constraints for each body, written at the top of the constraints vector

        Inputs:
            num_bodies: Num bodies per world
            first_body_id: First body id per world
            bodies_q: Body poses
            world_mask: Per-world flag to perform the computation (0 = skip)
        Outputs:
            constraints: Constraint vector per world
    ):
    """

    # Retrieve the thread indices (= world index, body index)
    wd_id, rb_id_loc = wp.tid()

    if wd_id < num_bodies.shape[0] and world_mask[wd_id] != 0 and rb_id_loc < num_bodies[wd_id]:
        # Get overall body id
        rb_id_tot = first_body_id[wd_id] + rb_id_loc

        # Evaluate unit quaternion constraint
        q = wp.transform_get_rotation(bodies_q[rb_id_tot])
        constraints[wd_id, rb_id_loc] = wp.dot(q, q) - 1.0


@cache
def create_eval_joint_constraints_kernel(has_universal_joints: bool):
    """
    Returns the joint constraints evaluation kernel, statically baking in whether there are universal joints
    or not (these joints need a separate handling)
    """

    @wp.kernel
    def _eval_joint_constraints(
        # Inputs
        num_joints: wp.array[wp.int32],
        first_joint_id: wp.array[wp.int32],
        joints_dof_type: wp.array[wp.int32],
        joints_act_type: wp.array[wp.int32],
        joints_bid_B: wp.array[wp.int32],
        joints_bid_F: wp.array[wp.int32],
        joints_X: wp.array[wp.mat33f],
        joints_B_r_B: wp.array[wp.vec3f],
        joints_F_r_F: wp.array[wp.vec3f],
        bodies_q: wp.array[wp.transformf],
        pos_control_transforms: wp.array[wp.transformf],
        ct_full_to_red_map: wp.array[wp.int32],
        world_mask: wp.array[wp.int32],
        # Outputs
        constraints: wp.array2d[wp.float32],
    ):
        """
        A kernel computing joint constraints with the log map formulation, first computing 6 constraints per
        joint (treating it as a fixed joint), then writing out the relevant subset of constraints (only along
        relevant directions) using a precomputed full to reduced map.

        Note: the log map formulation doesn't allow to formulate passive universal joints. If such joints are
        present, the right number of (incorrect) constraints is first written with the log map, then the result
        is overwritten in a second pass with the correct constraints.

        Inputs:
            num_joints: Num joints per world
            first_joint_id: First joint id per world
            joints_dof_type: Joint dof type (i.e. revolute, spherical, ...)
            joints_act_type: Joint actuation type (i.e. passive or actuated)
            joints_bid_B: Joint base body id
            joints_bid_F: Joint follower body id
            joints_X: Joint frame (local axes, valid both on base and follower)
            joints_B_r_B: Joint local position on base body
            joints_F_r_F: Joint local position on follower body
            bodies_q: Body poses
            pos_control_transforms: Joint position-control transformation
            ct_full_to_red_map: Map from full to reduced constraint id
            world_mask: Per-world flag to perform the computation (0 = skip)
        Outputs:
            constraints: Constraint vector per world
        """

        # Retrieve the thread indices (= world index, joint index)
        wd_id, jt_id_loc = wp.tid()

        if wd_id < num_joints.shape[0] and world_mask[wd_id] != 0 and jt_id_loc < num_joints[wd_id]:
            # Get overall joint id
            jt_id_tot = first_joint_id[wd_id] + jt_id_loc

            # Get reduced constraint ids (-1 meaning constraint is not used)
            first_ct_id_full = 6 * jt_id_tot
            trans_ct_ids_red = wp.vec3i(
                ct_full_to_red_map[first_ct_id_full],
                ct_full_to_red_map[first_ct_id_full + 1],
                ct_full_to_red_map[first_ct_id_full + 2],
            )
            rot_ct_ids_red = wp.vec3i(
                ct_full_to_red_map[first_ct_id_full + 3],
                ct_full_to_red_map[first_ct_id_full + 4],
                ct_full_to_red_map[first_ct_id_full + 5],
            )

            # Get joint local positions and orientation
            x_base = joints_B_r_B[jt_id_tot]
            x_follower = joints_F_r_F[jt_id_tot]
            X_T = wp.transpose(joints_X[jt_id_tot])

            # Get base and follower transformations
            base_id = joints_bid_B[jt_id_tot]
            if base_id < 0:
                c_base = wp.vec3f(0.0, 0.0, 0.0)
                q_base = wp.quatf(0.0, 0.0, 0.0, 1.0)
            else:
                c_base = wp.transform_get_translation(bodies_q[base_id])
                q_base = wp.transform_get_rotation(bodies_q[base_id])
            follower_id = joints_bid_F[jt_id_tot]
            c_follower = wp.transform_get_translation(bodies_q[follower_id])
            q_follower = wp.transform_get_rotation(bodies_q[follower_id])

            # Get position control transformation, in joint/body frame for translation/rotation part
            t_control_joint = wp.transform_get_translation(pos_control_transforms[jt_id_tot])
            q_control_body = wp.transform_get_rotation(pos_control_transforms[jt_id_tot])

            # Translation constraints: compute "error" translation, in joint frame
            pos_follower_world = unit_quat_apply(q_follower, x_follower) + c_follower
            pos_follower_base = unit_quat_conj_apply(q_base, pos_follower_world - c_base)
            pos_rel_base = (
                pos_follower_base - x_base
            )  # Relative position on base body (should match translation from controls)
            t_error = X_T * pos_rel_base - t_control_joint  # Error in joint frame

            # Rotation constraints: compute "error" rotation with the log map, in joint frame
            q_error_base = wp.quat_inverse(q_base) * q_follower * wp.quat_inverse(q_control_body)
            rot_error = X_T * quat_log(q_error_base)

            # Write out constraint
            for i in range(3):
                if trans_ct_ids_red[i] >= 0:
                    constraints[wd_id, trans_ct_ids_red[i]] = t_error[i]
                if rot_ct_ids_red[i] >= 0:
                    constraints[wd_id, rot_ct_ids_red[i]] = rot_error[i]

            # Correct constraints for passive universal joints
            if wp.static(has_universal_joints):
                # Check for a passive universal joint
                dof_type_j = joints_dof_type[jt_id_tot]
                act_type_j = joints_act_type[jt_id_tot]
                if dof_type_j != FKJointDoFType.UNIVERSAL or act_type_j != JointActuationType.PASSIVE:
                    return

                # Compute constraint (dot product between x axis on base and y axis on follower)
                a_x = X_T[0]
                a_y = X_T[1]
                a_x_base = unit_quat_apply(q_base, a_x)
                a_y_follower = unit_quat_apply(q_follower, a_y)
                ct = -wp.dot(a_x_base, a_y_follower)

                # Set constraint in output (at a location corresponding to z rotational constraint)
                constraints[wd_id, rot_ct_ids_red[2]] = ct

    return _eval_joint_constraints


@wp.kernel
def _eval_unit_quaternion_constraints_jacobian(
    # Inputs
    num_bodies: wp.array[wp.int32],
    first_body_id: wp.array[wp.int32],
    bodies_q: wp.array[wp.transformf],
    world_mask: wp.array[wp.int32],
    # Outputs
    constraints_jacobian: wp.array3d[wp.float32],
):
    """
    A kernel computing the Jacobian of unit norm quaternion constraints for each body, written at the top of the
    constraints Jacobian

    Inputs:
        num_bodies: Num bodies per world
        first_body_id: First body id per world
        bodies_q: Body poses
        world_mask: Per-world flag to perform the computation (0 = skip)
    Outputs:
        constraints_jacobian: Constraints Jacobian per world
    """

    # Retrieve the thread indices (= world index, body index)
    wd_id, rb_id_loc = wp.tid()

    if wd_id < num_bodies.shape[0] and world_mask[wd_id] != 0 and rb_id_loc < num_bodies[wd_id]:
        # Get overall body id
        rb_id_tot = first_body_id[wd_id] + rb_id_loc

        # Evaluate constraint Jacobian
        q = wp.transform_get_rotation(bodies_q[rb_id_tot])
        state_offset = 7 * rb_id_loc + 3
        constraints_jacobian[wd_id, rb_id_loc, state_offset] = 2.0 * q.x
        constraints_jacobian[wd_id, rb_id_loc, state_offset + 1] = 2.0 * q.y
        constraints_jacobian[wd_id, rb_id_loc, state_offset + 2] = 2.0 * q.z
        constraints_jacobian[wd_id, rb_id_loc, state_offset + 3] = 2.0 * q.w


@wp.kernel
def _eval_unit_quaternion_constraints_sparse_jacobian(
    # Inputs
    num_bodies: wp.array[wp.int32],
    first_body_id: wp.array[wp.int32],
    bodies_q: wp.array[wp.transformf],
    rb_nzb_id: wp.array[wp.int32],
    world_mask: wp.array[wp.int32],
    # Outputs
    jacobian_nzb: wp.array[block_type],
):
    """
    A kernel computing the sparse Jacobian of unit norm quaternion constraints for each body, written at the top of the
    constraints Jacobian

    Inputs:
        num_bodies: Num bodies per world
        first_body_id: First body id per world
        bodies_q: Body poses
        rb_nzb_id: Id of the nzb corresponding to the constraint per body
        world_mask: Per-world flag to perform the computation (0 = skip)
    Outputs:
        jacobian_nzb: Non-zero blocks of the sparse Jacobian
    """

    # Retrieve the thread indices (= world index, body index)
    wd_id, rb_id_loc = wp.tid()

    if wd_id < num_bodies.shape[0] and world_mask[wd_id] != 0 and rb_id_loc < num_bodies[wd_id]:
        # Get overall body id
        rb_id_tot = first_body_id[wd_id] + rb_id_loc

        # Evaluate constraint Jacobian
        q = wp.transform_get_rotation(bodies_q[rb_id_tot])
        nzb_id = rb_nzb_id[rb_id_tot]
        jacobian_nzb[nzb_id][3] = 2.0 * q.x
        jacobian_nzb[nzb_id][4] = 2.0 * q.y
        jacobian_nzb[nzb_id][5] = 2.0 * q.z
        jacobian_nzb[nzb_id][6] = 2.0 * q.w


@cache
def create_eval_joint_constraints_jacobian_kernel(has_universal_joints: bool):
    """
    Returns the joint constraints Jacobian evaluation kernel, statically baking in whether there are universal joints
    or not (these joints need a separate handling)
    """

    @wp.kernel
    def _eval_joint_constraints_jacobian(
        # Inputs
        num_joints: wp.array[wp.int32],
        first_joint_id: wp.array[wp.int32],
        first_body_id: wp.array[wp.int32],
        joints_dof_type: wp.array[wp.int32],
        joints_act_type: wp.array[wp.int32],
        joints_bid_B: wp.array[wp.int32],
        joints_bid_F: wp.array[wp.int32],
        joints_X: wp.array[wp.mat33f],
        joints_B_r_B: wp.array[wp.vec3f],
        joints_F_r_F: wp.array[wp.vec3f],
        bodies_q: wp.array[wp.transformf],
        pos_control_transforms: wp.array[wp.transformf],
        ct_full_to_red_map: wp.array[wp.int32],
        world_mask: wp.array[wp.int32],
        # Outputs
        constraints_jacobian: wp.array3d[wp.float32],
    ):
        """
        A kernel computing the Jacobian of the joint constraints.
        The Jacobian is assumed to have already been filled with zeros, at least in the coefficients that
        are always zero due to joint connectivity.

        Inputs:
            num_joints: Num joints per world
            first_joint_id: First joint id per world
            first_body_id: First body id per world
            joints_dof_type: Joint dof type (i.e. revolute, spherical, ...)
            joints_act_type: Joint actuation type (i.e. passive or actuated)
            joints_bid_B: Joint base body id
            joints_bid_F: Joint follower body id
            joints_X: Joint frame (local axes, valid both on base and follower)
            joints_B_r_B: Joint local position on base body
            joints_F_r_F: Joint local position on follower body
            bodies_q: Body poses
            pos_control_transforms: Joint position-control transformation
            ct_full_to_red_map: Map from full to reduced constraint id
            world_mask: Per-world flag to perform the computation (0 = skip)
        Outputs:
            constraints_jacobian: Constraint Jacobian per world
        """

        # Retrieve the thread indices (= world index, joint index)
        wd_id, jt_id_loc = wp.tid()

        if wd_id < num_joints.shape[0] and world_mask[wd_id] != 0 and jt_id_loc < num_joints[wd_id]:
            # Get overall joint id
            jt_id_tot = first_joint_id[wd_id] + jt_id_loc

            # Get reduced constraint ids (-1 meaning constraint is not used)
            first_ct_id_full = 6 * jt_id_tot
            trans_ct_ids_red = wp.vec3i(
                ct_full_to_red_map[first_ct_id_full],
                ct_full_to_red_map[first_ct_id_full + 1],
                ct_full_to_red_map[first_ct_id_full + 2],
            )
            rot_ct_ids_red = wp.vec3i(
                ct_full_to_red_map[first_ct_id_full + 3],
                ct_full_to_red_map[first_ct_id_full + 4],
                ct_full_to_red_map[first_ct_id_full + 5],
            )

            # Get joint local positions and orientation
            x_follower = joints_F_r_F[jt_id_tot]
            X_T = wp.transpose(joints_X[jt_id_tot])

            # Get base and follower transformations
            base_id_tot = joints_bid_B[jt_id_tot]
            if base_id_tot < 0:
                c_base = wp.vec3f(0.0, 0.0, 0.0)
                q_base = wp.quatf(0.0, 0.0, 0.0, 1.0)
            else:
                c_base = wp.transform_get_translation(bodies_q[base_id_tot])
                q_base = wp.transform_get_rotation(bodies_q[base_id_tot])
            follower_id_tot = joints_bid_F[jt_id_tot]
            c_follower = wp.transform_get_translation(bodies_q[follower_id_tot])
            q_follower = wp.transform_get_rotation(bodies_q[follower_id_tot])
            base_id_loc = base_id_tot - first_body_id[wd_id]
            follower_id_loc = follower_id_tot - first_body_id[wd_id]

            # Get position control transformation (rotation part only, as translation part doesn't affect the Jacobian)
            q_control_body = wp.transform_get_rotation(pos_control_transforms[jt_id_tot])

            # Translation constraints
            X_T_R_base_T = X_T * unit_quat_conj_to_rotation_matrix(q_base)
            if base_id_tot >= 0:
                jac_trans_c_base = -X_T_R_base_T
                delta_pos = unit_quat_apply(q_follower, x_follower) + c_follower - c_base
                jac_trans_q_base = X_T * unit_quat_conj_apply_jacobian(q_base, delta_pos)
            jac_trans_c_follower = X_T_R_base_T
            jac_trans_q_follower = X_T_R_base_T * unit_quat_apply_jacobian(q_follower, x_follower)

            # Rotation constraints
            q_base_sq_norm = wp.dot(q_base, q_base)
            q_follower_sq_norm = wp.dot(q_follower, q_follower)
            R_base_T = unit_quat_conj_to_rotation_matrix(q_base / wp.sqrt(q_base_sq_norm))
            q_rel = q_follower * wp.quat_inverse(q_control_body) * wp.quat_inverse(q_base)
            temp = X_T * R_base_T * quat_left_jacobian_inverse(q_rel)
            if base_id_tot >= 0:
                jac_rot_q_base = (-2.0 / q_base_sq_norm) * temp * G_of(q_base)
            jac_rot_q_follower = (2.0 / q_follower_sq_norm) * temp * G_of(q_follower)
            # Note: we need X^T * R_base^T both for translation and rotation constraints, but to get the correct
            # derivatives for non-unit quaternions (which may be encountered before convergence) we end up needing
            # to use a separate formula to evaluate R_base in either case

            # Write out Jacobian
            base_offset = 7 * base_id_loc
            follower_offset = 7 * follower_id_loc
            for i in range(3):
                trans_ct_id_red = trans_ct_ids_red[i]
                if trans_ct_id_red >= 0:
                    for j in range(3):
                        if base_id_tot >= 0:
                            constraints_jacobian[wd_id, trans_ct_id_red, base_offset + j] = jac_trans_c_base[i, j]
                        constraints_jacobian[wd_id, trans_ct_id_red, follower_offset + j] = jac_trans_c_follower[i, j]
                    for j in range(4):
                        if base_id_tot >= 0:
                            constraints_jacobian[wd_id, trans_ct_id_red, base_offset + 3 + j] = jac_trans_q_base[i, j]
                        constraints_jacobian[wd_id, trans_ct_id_red, follower_offset + 3 + j] = jac_trans_q_follower[
                            i, j
                        ]
                rot_ct_id_red = rot_ct_ids_red[i]
                if rot_ct_id_red >= 0:
                    for j in range(4):
                        if base_id_tot >= 0:
                            constraints_jacobian[wd_id, rot_ct_id_red, base_offset + 3 + j] = jac_rot_q_base[i, j]
                        constraints_jacobian[wd_id, rot_ct_id_red, follower_offset + 3 + j] = jac_rot_q_follower[i, j]

            # Correct Jacobian for passive universal joints
            if wp.static(has_universal_joints):
                # Check for a passive universal joint
                dof_type_j = joints_dof_type[jt_id_tot]
                act_type_j = joints_act_type[jt_id_tot]
                if dof_type_j != FKJointDoFType.UNIVERSAL or act_type_j != JointActuationType.PASSIVE:
                    return

                # Compute constraint Jacobian (cross product between x axis on base and y axis on follower)
                a_x = X_T[0]
                a_y = X_T[1]
                if base_id_tot >= 0:
                    a_y_follower = unit_quat_apply(q_follower, a_y)
                    jac_q_base = -a_y_follower * unit_quat_apply_jacobian(q_base, a_x)
                a_x_base = unit_quat_apply(q_base, a_x)
                jac_q_follower = -a_x_base * unit_quat_apply_jacobian(q_follower, a_y)

                # Write out Jacobian
                for i in range(4):
                    rot_ct_id_red = rot_ct_ids_red[2]
                    if base_id_tot >= 0:
                        constraints_jacobian[wd_id, rot_ct_id_red, base_offset + 3 + i] = jac_q_base[i]
                    constraints_jacobian[wd_id, rot_ct_id_red, follower_offset + 3 + i] = jac_q_follower[i]

    return _eval_joint_constraints_jacobian


@cache
def create_eval_joint_constraints_sparse_jacobian_kernel(has_universal_joints: bool):
    """
    Returns the joint constraints sparse Jacobian evaluation kernel,
    statically baking in whether there are universal joints or not
    (these joints need a separate handling)
    """

    @wp.kernel
    def _eval_joint_constraints_sparse_jacobian(
        # Inputs
        num_joints: wp.array[wp.int32],
        first_joint_id: wp.array[wp.int32],
        first_body_id: wp.array[wp.int32],
        joints_dof_type: wp.array[wp.int32],
        joints_act_type: wp.array[wp.int32],
        joints_bid_B: wp.array[wp.int32],
        joints_bid_F: wp.array[wp.int32],
        joints_X: wp.array[wp.mat33f],
        joints_B_r_B: wp.array[wp.vec3f],
        joints_F_r_F: wp.array[wp.vec3f],
        bodies_q: wp.array[wp.transformf],
        pos_control_transforms: wp.array[wp.transformf],
        ct_nzb_id_base: wp.array[wp.int32],
        ct_nzb_id_follower: wp.array[wp.int32],
        world_mask: wp.array[wp.int32],
        # Outputs
        jacobian_nzb: wp.array[block_type],
    ):
        """
        A kernel computing the Jacobian of the joint constraints.
        The Jacobian is assumed to have already been filled with zeros, at least in the coefficients that
        are always zero due to joint connectivity.

        Inputs:
            num_joints: Num joints per world
            first_joint_id: First joint id per world
            first_body_id: First body id per world
            joints_dof_type: Joint dof type (i.e. revolute, spherical, ...)
            joints_act_type: Joint actuation type (i.e. passive or actuated)
            joints_bid_B: Joint base body id
            joints_bid_F: Joint follower body id
            joints_X: Joint frame (local axes, valid both on base and follower)
            joints_B_r_B: Joint local position on base body
            joints_F_r_F: Joint local position on follower body
            bodies_q: Body poses
            pos_control_transforms: Joint position-control transformation
            ct_nzb_id_base: Map from full constraint id to nzb id, for the base body blocks
            ct_nzb_id_base: Map from full constraint id to nzb id, for the follower body blocks
            world_mask: Per-world flag to perform the computation (0 = skip)
        Outputs:
            jacobian_nzb: Non-zero blocks of the sparse Jacobian
        """

        # Retrieve the thread indices (= world index, joint index)
        wd_id, jt_id_loc = wp.tid()

        if wd_id < num_joints.shape[0] and world_mask[wd_id] != 0 and jt_id_loc < num_joints[wd_id]:
            # Get overall joint id
            jt_id_tot = first_joint_id[wd_id] + jt_id_loc

            # Get nzb ids (-1 meaning constraint is not used)
            start = 6 * jt_id_tot
            end = start + 6
            nzb_ids_base = ct_nzb_id_base[start:end]
            nzb_ids_follower = ct_nzb_id_follower[start:end]

            # Get joint local positions and orientation
            x_follower = joints_F_r_F[jt_id_tot]
            X_T = wp.transpose(joints_X[jt_id_tot])

            # Get base and follower transformations
            base_id = joints_bid_B[jt_id_tot]
            if base_id < 0:
                c_base = wp.vec3f(0.0, 0.0, 0.0)
                q_base = wp.quatf(0.0, 0.0, 0.0, 1.0)
            else:
                c_base = wp.transform_get_translation(bodies_q[base_id])
                q_base = wp.transform_get_rotation(bodies_q[base_id])
            follower_id = joints_bid_F[jt_id_tot]
            c_follower = wp.transform_get_translation(bodies_q[follower_id])
            q_follower = wp.transform_get_rotation(bodies_q[follower_id])

            # Get position control transformation (rotation part only, as translation part doesn't affect the Jacobian)
            q_control_body = wp.transform_get_rotation(pos_control_transforms[jt_id_tot])

            # Translation constraints
            X_T_R_base_T = X_T * unit_quat_conj_to_rotation_matrix(q_base)
            if base_id >= 0:
                jac_trans_c_base = -X_T_R_base_T
                delta_pos = unit_quat_apply(q_follower, x_follower) + c_follower - c_base
                jac_trans_q_base = X_T * unit_quat_conj_apply_jacobian(q_base, delta_pos)
            jac_trans_c_follower = X_T_R_base_T
            jac_trans_q_follower = X_T_R_base_T * unit_quat_apply_jacobian(q_follower, x_follower)

            # Rotation constraints
            q_base_sq_norm = wp.dot(q_base, q_base)
            q_follower_sq_norm = wp.dot(q_follower, q_follower)
            R_base_T = unit_quat_conj_to_rotation_matrix(q_base / wp.sqrt(q_base_sq_norm))
            q_rel = q_follower * wp.quat_inverse(q_control_body) * wp.quat_inverse(q_base)
            temp = X_T * R_base_T * quat_left_jacobian_inverse(q_rel)
            if base_id >= 0:
                jac_rot_q_base = (-2.0 / q_base_sq_norm) * temp * G_of(q_base)
            jac_rot_q_follower = (2.0 / q_follower_sq_norm) * temp * G_of(q_follower)
            # Note: we need X^T * R_base^T both for translation and rotation constraints, but to get the correct
            # derivatives for non-unit quaternions (which may be encountered before convergence) we end up needing
            # to use a separate formula to evaluate R_base in either case

            # Write out Jacobian
            if base_id >= 0:
                for i in range(3):
                    nzb_id = nzb_ids_base[i]
                    if nzb_id >= 0:
                        for j in range(3):
                            jacobian_nzb[nzb_id][j] = jac_trans_c_base[i, j]
                        for j in range(4):
                            jacobian_nzb[nzb_id][3 + j] = jac_trans_q_base[i, j]
                for i in range(3):
                    nzb_id = nzb_ids_base[i + 3]
                    if nzb_id >= 0:
                        for j in range(4):
                            jacobian_nzb[nzb_id][3 + j] = jac_rot_q_base[i, j]
            for i in range(3):
                nzb_id = nzb_ids_follower[i]
                if nzb_id >= 0:
                    for j in range(3):
                        jacobian_nzb[nzb_id][j] = jac_trans_c_follower[i, j]
                    for j in range(4):
                        jacobian_nzb[nzb_id][3 + j] = jac_trans_q_follower[i, j]
            for i in range(3):
                nzb_id = nzb_ids_follower[i + 3]
                if nzb_id >= 0:
                    for j in range(4):
                        jacobian_nzb[nzb_id][3 + j] = jac_rot_q_follower[i, j]

            # Correct Jacobian for passive universal joints
            if wp.static(has_universal_joints):
                # Check for a passive universal joint
                dof_type_j = joints_dof_type[jt_id_tot]
                act_type_j = joints_act_type[jt_id_tot]
                if dof_type_j != FKJointDoFType.UNIVERSAL or act_type_j != JointActuationType.PASSIVE:
                    return

                # Compute constraint Jacobian (cross product between x axis on base and y axis on follower)
                a_x = X_T[0]
                a_y = X_T[1]
                if base_id >= 0:
                    a_y_follower = unit_quat_apply(q_follower, a_y)
                    jac_q_base = -a_y_follower * unit_quat_apply_jacobian(q_base, a_x)
                a_x_base = unit_quat_apply(q_base, a_x)
                jac_q_follower = -a_x_base * unit_quat_apply_jacobian(q_follower, a_y)

                # Write out Jacobian
                if base_id >= 0:
                    nzb_id = nzb_ids_base[5]
                    for j in range(4):
                        jacobian_nzb[nzb_id][3 + j] = jac_q_base[j]
                nzb_id = nzb_ids_follower[5]
                for j in range(4):
                    jacobian_nzb[nzb_id][3 + j] = jac_q_follower[j]

    return _eval_joint_constraints_sparse_jacobian


@cache
def create_2d_tile_based_kernels(TILE_SIZE_CTS: wp.int32, TILE_SIZE_VRS: wp.int32):
    """
    Generates and returns all kernels based on 2d tiles in this module, given the tile size to use along the constraints
    and variables (i.e. body poses) dimensions in the constraint vector, Jacobian, step vector etc.

    These are _eval_pattern_T_pattern, _eval_jacobian_T_jacobian, eval_jacobian_T_constraints
    (returned in this order)
    """

    @wp.func
    def clip_to_one(x: wp.float32):
        """
        Clips an number to 1 if it is above
        """
        return wp.min(x, 1.0)

    @wp.kernel
    def _eval_pattern_T_pattern(
        # Inputs
        sparsity_pattern: wp.array3d[wp.float32],
        # Outputs
        pattern_T_pattern: wp.array3d[wp.float32],
    ):
        """
        A kernel computing the sparsity pattern of J^T * J given that of J, in each world
        More specifically, given an integer matrix of zeros and ones representing a sparsity pattern, multiply it by
        its transpose and clip values to [0, 1] to get the sparsity pattern of J^T * J
        Note: mostly redundant with _eval_jacobian_T_jacobian apart from the clipping, could possibly be removed
        (was initially written to take int32, but float32 is actually faster)

        Inputs:
            sparsity_pattern: Jacobian sparsity pattern per world
        Outputs:
            pattern_T_pattern: Jacobian^T * Jacobian sparsity pattern per world
        """
        wd_id, i, j = wp.tid()  # Thread indices (= world index, output tile indices)

        if (
            wd_id < pattern_T_pattern.shape[0]
            and i * TILE_SIZE_VRS < pattern_T_pattern.shape[1]
            and j * TILE_SIZE_VRS < pattern_T_pattern.shape[2]
        ):
            tile_out = wp.tile_zeros(shape=(TILE_SIZE_VRS, TILE_SIZE_VRS), dtype=wp.float32)

            num_cts = sparsity_pattern.shape[1]
            num_tiles_K = (num_cts + TILE_SIZE_CTS - 1) // TILE_SIZE_CTS  # Equivalent to ceil(num_cts / TILE_SIZE_CTS)

            for k in range(num_tiles_K):
                tile_i_3d = wp.tile_load(
                    sparsity_pattern,
                    shape=(1, TILE_SIZE_CTS, TILE_SIZE_VRS),
                    offset=(wd_id, k * TILE_SIZE_CTS, i * TILE_SIZE_VRS),
                )
                tile_i = wp.tile_reshape(tile_i_3d, (TILE_SIZE_CTS, TILE_SIZE_VRS))
                tile_i_T = wp.tile_transpose(tile_i)
                tile_j_3d = wp.tile_load(
                    sparsity_pattern,
                    shape=(1, TILE_SIZE_CTS, TILE_SIZE_VRS),
                    offset=(wd_id, k * TILE_SIZE_CTS, j * TILE_SIZE_VRS),
                )
                tile_j = wp.tile_reshape(tile_j_3d, (TILE_SIZE_CTS, TILE_SIZE_VRS))
                wp.tile_matmul(tile_i_T, tile_j, tile_out)

            tile_out_3d = wp.tile_reshape(tile_out, (1, TILE_SIZE_VRS, TILE_SIZE_VRS))
            tile_out_3d_clipped = wp.tile_map(clip_to_one, tile_out_3d)
            wp.tile_store(pattern_T_pattern, tile_out_3d_clipped, offset=(wd_id, i * TILE_SIZE_VRS, j * TILE_SIZE_VRS))

    @wp.kernel
    def _eval_jacobian_T_jacobian(
        # Inputs
        constraints_jacobian: wp.array3d[wp.float32],
        tile_sparsity_pattern: wp.array3d[wp.int32],
        world_mask: wp.array[wp.int32],
        # Outputs
        jacobian_T_jacobian: wp.array3d[wp.float32],
    ):
        """
        A kernel computing the matrix product J^T * J given the Jacobian J, in each world

        Inputs:
            constraints_jacobian: Constraint Jacobian per world
            tile_sparsity_pattern: Per-tile sparsity pattern of the Jacobian (0 = tile is fully zero)
            world_mask: Per-world flag to perform the computation (0 = skip)
        Outputs:
            jacobian_T_jacobian: Jacobian^T * Jacobian per world
        """
        wd_id, i, j = wp.tid()  # Thread indices (= world index, output tile indices)

        if (
            wd_id < jacobian_T_jacobian.shape[0]
            and world_mask[wd_id] != 0
            and i * TILE_SIZE_VRS < jacobian_T_jacobian.shape[1]
            and j * TILE_SIZE_VRS < jacobian_T_jacobian.shape[2]
        ):
            tile_out = wp.tile_zeros(shape=(TILE_SIZE_VRS, TILE_SIZE_VRS), dtype=wp.float32)

            num_cts = constraints_jacobian.shape[1]
            num_tiles_K = (num_cts + TILE_SIZE_CTS - 1) // TILE_SIZE_CTS  # Equivalent to ceil(num_cts / TILE_SIZE_CTS)

            for k in range(num_tiles_K):
                if tile_sparsity_pattern[wd_id, k, i] == 0 or tile_sparsity_pattern[wd_id, k, j] == 0:
                    continue
                tile_i_3d = wp.tile_load(
                    constraints_jacobian,
                    shape=(1, TILE_SIZE_CTS, TILE_SIZE_VRS),
                    offset=(wd_id, k * TILE_SIZE_CTS, i * TILE_SIZE_VRS),
                )
                tile_i = wp.tile_reshape(tile_i_3d, (TILE_SIZE_CTS, TILE_SIZE_VRS))
                tile_i_T = wp.tile_transpose(tile_i)
                tile_j_3d = wp.tile_load(
                    constraints_jacobian,
                    shape=(1, TILE_SIZE_CTS, TILE_SIZE_VRS),
                    offset=(wd_id, k * TILE_SIZE_CTS, j * TILE_SIZE_VRS),
                )
                tile_j = wp.tile_reshape(tile_j_3d, (TILE_SIZE_CTS, TILE_SIZE_VRS))
                wp.tile_matmul(tile_i_T, tile_j, tile_out)

            tile_out_3d = wp.tile_reshape(tile_out, (1, TILE_SIZE_VRS, TILE_SIZE_VRS))
            wp.tile_store(jacobian_T_jacobian, tile_out_3d, offset=(wd_id, i * TILE_SIZE_VRS, j * TILE_SIZE_VRS))

    @wp.kernel
    def _eval_jacobian_T_constraints(
        # Inputs
        constraints_jacobian: wp.array3d[wp.float32],
        constraints: wp.array2d[wp.float32],
        tile_sparsity_pattern: wp.array3d[wp.int32],
        world_mask: wp.array[wp.int32],
        # Outputs
        jacobian_T_constraints: wp.array2d[wp.float32],
    ):
        """
        A kernel computing the matrix product J^T * C given the Jacobian J and the constraints vector C, in each world

        Inputs:
            constraints_jacobian: Constraint Jacobian per world
            constraints: Constraint vector per world
            tile_sparsity_pattern: Per-tile sparsity pattern of the Jacobian (0 = tile is fully zero)
            world_mask: Per-world flag to perform the computation (0 = skip)
        Outputs:
            jacobian_T_constraints: Jacobian^T * Constraints per world
        """
        wd_id, i = wp.tid()  # Thread indices (= world index, output tile index)

        if (
            wd_id < jacobian_T_constraints.shape[0]
            and world_mask[wd_id] != 0
            and i * TILE_SIZE_VRS < jacobian_T_constraints.shape[1]
        ):
            segment_out = wp.tile_zeros(shape=(TILE_SIZE_VRS, 1), dtype=wp.float32)

            num_cts = constraints_jacobian.shape[1]
            num_tiles_K = (num_cts + TILE_SIZE_CTS - 1) // TILE_SIZE_CTS  # Equivalent to ceil(num_cts / TILE_SIZE_CTS)

            for k in range(num_tiles_K):
                if tile_sparsity_pattern[wd_id, k, i] == 0:
                    continue
                tile_i_3d = wp.tile_load(
                    constraints_jacobian,
                    shape=(1, TILE_SIZE_CTS, TILE_SIZE_VRS),
                    offset=(wd_id, k * TILE_SIZE_CTS, i * TILE_SIZE_VRS),
                )
                tile_i = wp.tile_reshape(tile_i_3d, (TILE_SIZE_CTS, TILE_SIZE_VRS))
                tile_i_T = wp.tile_transpose(tile_i)
                segment_k_2d = wp.tile_load(constraints, shape=(1, TILE_SIZE_CTS), offset=(wd_id, k * TILE_SIZE_CTS))
                segment_k = wp.tile_reshape(segment_k_2d, (TILE_SIZE_CTS, 1))  # Technically still 2d...
                wp.tile_matmul(tile_i_T, segment_k, segment_out)

            segment_out_2d = wp.tile_reshape(
                segment_out,
                (
                    1,
                    TILE_SIZE_VRS,
                ),
            )
            wp.tile_store(
                jacobian_T_constraints,
                segment_out_2d,
                offset=(
                    wd_id,
                    i * TILE_SIZE_VRS,
                ),
            )

    return _eval_pattern_T_pattern, _eval_jacobian_T_jacobian, _eval_jacobian_T_constraints


@cache
def create_1d_tile_based_kernels(TILE_SIZE_CTS: wp.int32, TILE_SIZE_VRS: wp.int32):
    """
    Generates and returns all kernels based on 1d tiles in this module, given the tile size to use along the constraints
    and variables (i.e. body poses) dimensions in the constraint vector, Jacobian, step vector etc.

    These are _eval_max_constraint, _eval_merit_function, _eval_merit_function_gradient
    (returned in this order)
    """

    @wp.func
    def _isnan(x: wp.float32) -> wp.int32:
        """Calls wp.isnan and converts the result to int32"""
        return wp.int32(wp.isnan(x))

    @wp.kernel
    def _eval_max_constraint(
        # Inputs
        constraints: wp.array2d[wp.float32],
        # Outputs
        max_constraint: wp.array[wp.float32],
    ):
        """
        A kernel computing the max absolute constraint from the constraints vector, in each world.

        Inputs:
            constraints: Constraint vector per world
        Outputs:
            max_constraint: Max absolute constraint per world; must be zero-initialized
        """
        wd_id, i, tid = wp.tid()  # Thread indices (= world index, input tile index, thread index in block)

        if wd_id < constraints.shape[0] and i * TILE_SIZE_CTS < constraints.shape[1]:
            segment = wp.tile_load(constraints, shape=(1, TILE_SIZE_CTS), offset=(wd_id, i * TILE_SIZE_CTS))
            segment_max = wp.tile_max(wp.tile_map(wp.abs, segment))[0]
            segment_has_nan = wp.tile_max(wp.tile_map(_isnan, segment))[0]

            if tid == 0:
                if segment_has_nan:
                    # Write NaN in max (non-atomically, as this will overwrite any non-NaN value)
                    max_constraint[wd_id] = wp.nan
                else:
                    # Atomically update the max, only if it is not yet NaN (in CUDA, the max() operation only
                    # considers non-NaN values, so the NaN value would get overwritten by a non-NaN otherwise)
                    while True:
                        curr_val = max_constraint[wd_id]
                        if wp.isnan(curr_val):
                            break
                        check_val = wp.atomic_cas(max_constraint, wd_id, curr_val, wp.max(curr_val, segment_max))
                        if check_val == curr_val:
                            break

    @wp.kernel
    def _eval_merit_function(
        # Inputs
        constraints: wp.array2d[wp.float32],
        # Outputs
        merit_function_val: wp.array[wp.float32],
    ):
        """
        A kernel computing the merit function, i.e. the least-squares error 1/2 * ||C||^2, from the constraints
        vector C, in each world

        Inputs:
            constraints: Constraint vector per world
        Outputs:
            merit_function_val: Merit function value per world; must be zero-initialized
        """
        wd_id, i, tid = wp.tid()  # Thread indices (= world index, input tile index, thread index in block)

        if wd_id < constraints.shape[0] and i * TILE_SIZE_CTS < constraints.shape[1]:
            segment = wp.tile_load(constraints, shape=(1, TILE_SIZE_CTS), offset=(wd_id, i * TILE_SIZE_CTS))
            segment_error = 0.5 * wp.tile_sum(wp.tile_map(wp.mul, segment, segment))[0]

            if tid == 0:
                wp.atomic_add(merit_function_val, wd_id, segment_error)

    @wp.kernel
    def _eval_merit_function_gradient(
        # Inputs
        step: wp.array2d[wp.float32],
        grad: wp.array2d[wp.float32],
        # Outputs
        merit_function_grad: wp.array[wp.float32],
    ):
        """
        A kernel computing the merit function gradient w.r.t. line search step size, from the step direction
        and the gradient in state space (= dC_ds^T * C). This is simply the dot product between these two vectors.

        Inputs:
            step: Step in variables per world
            grad: Gradient w.r.t. state (i.e. body poses) per world
        Outputs:
            merit_function_grad: Merit function gradient per world; must be zero-initialized
        """
        wd_id, i, tid = wp.tid()  # Thread indices (= world index, input tile index, thread index in block)

        if wd_id < step.shape[0] and i * TILE_SIZE_VRS < step.shape[1]:
            step_segment = wp.tile_load(step, shape=(1, TILE_SIZE_VRS), offset=(wd_id, i * TILE_SIZE_VRS))
            grad_segment = wp.tile_load(grad, shape=(1, TILE_SIZE_VRS), offset=(wd_id, i * TILE_SIZE_VRS))
            tile_dot_prod = wp.tile_sum(wp.tile_map(wp.mul, step_segment, grad_segment))[0]

            if tid == 0:
                wp.atomic_add(merit_function_grad, wd_id, tile_dot_prod)

    return _eval_max_constraint, _eval_merit_function, _eval_merit_function_gradient


@wp.kernel
def _eval_rhs(
    # Inputs
    grad: wp.array2d[wp.float32],
    # Outputs
    rhs: wp.array2d[wp.float32],
):
    """
    A kernel computing rhs := -grad (where rhs has shape (num_worlds, num_states_max, 1))

    Inputs:
        grad: Merit function gradient w.r.t. state (i.e. body poses) per world
    Outputs:
        rhs: Gauss-Newton right-hand side per world
    """
    wd_id, state_id_loc = wp.tid()  # Thread indices (= world index, state index)
    if wd_id < grad.shape[0] and state_id_loc < grad.shape[1]:
        rhs[wd_id, state_id_loc] = -grad[wd_id, state_id_loc]


@wp.kernel
def _eval_linear_combination(
    # Inputs
    alpha: wp.float32,
    x: wp.array2d[wp.float32],
    beta: wp.float32,
    y: wp.array2d[wp.float32],
    num_rows: wp.array[wp.int32],
    world_mask: wp.array[wp.int32],
    # Outputs
    z: wp.array2d[wp.float32],
):
    """
    A kernel computing z := alpha * x + beta * y

    Inputs:
        alpha: Scalar coefficient
        x: Stack of vectors (one per world) to be multiplied by alpha
        beta: Scalar coefficient
        y: Stack of vectors (one per world) to be multiplied by beta
        num_rows: Active size of the vectors (x, y and z) per world
        world_mask: Per-world flag to perform the computation (0 = skip)
    Outputs:
        z: Output stack of vectors
    """
    wd_id, row_id = wp.tid()  # Thread indices (= world index, row index)
    if wd_id < num_rows.shape[0] and world_mask[wd_id] != 0 and row_id < num_rows[wd_id]:
        z[wd_id, row_id] = alpha * x[wd_id, row_id] + beta * y[wd_id, row_id]


@wp.kernel
def _eval_stepped_state(
    # Inputs
    num_bodies: wp.array[wp.int32],
    first_body_id: wp.array[wp.int32],
    bodies_q_0_flat: wp.array[wp.float32],
    alpha: wp.array[wp.float32],
    step: wp.array2d[wp.float32],
    world_mask: wp.array[wp.int32],
    # Outputs
    bodies_q_alpha_flat: wp.array[wp.float32],
):
    """
    A kernel computing states_alpha := states_0 + alpha * step

    Inputs:
        num_bodies: Num bodies per world
        first_body_id: First body id per world
        bodies_q_0_flat: Previous state (for step size 0), flattened
        alpha: Step size per world
        step: Step direction per world
        world_mask: Per-world flag to perform the computation (0 = skip)
    Outputs:
        bodies_q_alpha_flat: New state (for step size alpha), flattened
    """
    wd_id, state_id_loc = wp.tid()  # Thread indices (= world index, state index)
    rb_id_loc = state_id_loc // 7
    if wd_id < num_bodies.shape[0] and world_mask[wd_id] != 0 and rb_id_loc < num_bodies[wd_id]:
        state_id_tot = 7 * first_body_id[wd_id] + state_id_loc
        bodies_q_alpha_flat[state_id_tot] = bodies_q_0_flat[state_id_tot] + alpha[wd_id] * step[wd_id, state_id_loc]


@wp.kernel
def _apply_line_search_step(
    # Inputs
    num_bodies: wp.array[wp.int32],
    first_body_id: wp.array[wp.int32],
    bodies_q_alpha: wp.array[wp.transformf],
    line_search_success: wp.array[wp.int32],
    # Outputs
    bodies_q: wp.array[wp.transformf],
):
    """
    A kernel replacing the state with the line search result, in worlds where line search succeeded
    Note: relies on the fact that the success flag is left at zero for worlds that don't run line search
    (otherwise would also need to check against line search mask)

    Inputs
        num_bodies: Num bodies per world
        first_body_id: First body id per world
        bodies_q_alpha: Stepped states (line search result)
        line_search_success: Per-world line search success flag
    Outputs
        bodies_q: Output state (rigid body poses)
    """
    wd_id, rb_id_loc = wp.tid()  # Thread indices (= world index, body index)
    if wd_id < num_bodies.shape[0] and line_search_success[wd_id] != 0 and rb_id_loc < num_bodies[wd_id]:
        rb_id_tot = first_body_id[wd_id] + rb_id_loc
        bodies_q[rb_id_tot] = bodies_q_alpha[rb_id_tot]


@wp.kernel
def _line_search_check(
    # Inputs
    val_0: wp.array[wp.float32],
    grad_0: wp.array[wp.float32],
    alpha: wp.array[wp.float32],
    val_alpha: wp.array[wp.float32],
    iteration: wp.array[wp.int32],
    max_iterations: wp.array[wp.int32],
    # Outputs
    line_search_success: wp.array[wp.int32],
    line_search_mask: wp.array[wp.int32],
    line_search_loop_condition: wp.array[wp.int32],
):
    """
    A kernel checking the sufficient decrease condition in line search in each world, and updating the looping
    condition (zero if max iterations reached, or all worlds successful)

    Inputs:
        val_0: Merit function value at 0, per world
        grad_0: Merit function gradient at 0, per world
        alpha: Step size per world (in/out)
        val_alpha: Merit function value at alpha, per world
        iteration: Iteration count, per world
        max_iterations: Max iterations (size 1 array)
    Outputs:
        line_search_success: Convergence per world
        line_search_mask: Per-world flag to continue line search (0 = skip)
        line_search_loop_condition: Loop condition; must be zero-initialized (size 1 array)
    """
    wd_id = wp.tid()  # Thread index (= world index)
    if wd_id < val_0.shape[0] and line_search_mask[wd_id] != 0:
        iteration[wd_id] += 1
        line_search_success[wd_id] = int(
            wp.isfinite(val_alpha[wd_id]) and val_alpha[wd_id] <= val_0[wd_id] + 1e-4 * alpha[wd_id] * grad_0[wd_id]
        )
        continue_loop_world = iteration[wd_id] < max_iterations[0] and not line_search_success[wd_id]
        line_search_mask[wd_id] = int(continue_loop_world)
        if continue_loop_world:
            alpha[wd_id] *= 0.5
        wp.atomic_max(line_search_loop_condition, 0, int(continue_loop_world))


@wp.kernel
def _newton_check(
    # Inputs
    max_constraint: wp.array[wp.float32],
    tolerance: wp.array[wp.float32],
    iteration: wp.array[wp.int32],
    min_iterations: wp.array[wp.int32],
    max_iterations: wp.array[wp.int32],
    line_search_success: wp.array[wp.int32],
    # Outputs
    newton_success: wp.array[wp.int32],
    newton_mask: wp.array[wp.int32],
    newton_loop_condition: wp.array[wp.int32],
):
    """
    A kernel checking the convergence (max constraint vs tolerance) in each world, and updating the looping
    condition (zero if max iterations reached, or all worlds successful)

    Inputs
        max_constraint: Max absolute constraint per world
        tolerance: Tolerance on max constraint (size 1 array)
        iteration: Iteration count, per world
        min_iterations: Min iterations per world (may be > 0 if incremental solve is enabled)
        max_iterations: Max iterations (size 1 array)
        line_search_success: Per-world line search success flag
    Outputs
        newton_success: Convergence per world
        newton_mask: Flag to keep iterating per world
        newton_loop_condition: Loop condition; must be zero-initialized (size 1 array)
    """
    wd_id = wp.tid()  # Thread index (= world index)
    if wd_id < max_constraint.shape[0] and newton_mask[wd_id] != 0:
        iteration_prev = iteration[wd_id]  # Index of the iteration that just ran
        iteration_next = iteration_prev + 1  # Index of the iteration that is about to run
        min_iterations_wd = min_iterations[wd_id]
        iteration[wd_id] = iteration_next
        reached_min_it = iteration_prev >= min_iterations_wd
        max_constraint_wd = max_constraint[wd_id]
        is_finite = wp.isfinite(max_constraint_wd)
        newton_success[wd_id] = int(is_finite and reached_min_it and max_constraint_wd <= tolerance[0])
        newton_continue_world = int(
            iteration_next < max_iterations[0]
            and not newton_success[wd_id]
            and is_finite  # Abort when encountering NaN / Inf values
            and line_search_success[wd_id]  # Abort in case of line search failure
        )
        newton_mask[wd_id] = newton_continue_world
        wp.atomic_max(newton_loop_condition, 0, newton_continue_world)


@wp.kernel
def _eval_target_constraint_velocities(
    # Inputs
    num_joints: wp.array[wp.int32],
    first_joint_id: wp.array[wp.int32],
    joints_dof_type: wp.array[wp.int32],
    joints_act_type: wp.array[wp.int32],
    actuated_dofs_offset: wp.array[wp.int32],
    ct_full_to_red_map: wp.array[wp.int32],
    actuators_u: wp.array[wp.float32],
    world_mask: wp.array[wp.int32],
    # Outputs
    target_cts_u: wp.array2d[wp.float32],
):
    """
    A kernel computing the target constraint velocities, i.e. zero for passive constraints
    and the prescribed dof velocity for actuated constraints.

    Inputs:
        num_joints: Num joints per world
        first_joint_id: First joint id per world
        joints_dof_type: Joint dof type (i.e. revolute, spherical, ...)
        joints_act_type: Joint actuation type (i.e. passive or actuated)
        actuated_dofs_offset: Joint first actuated dof id, among all actuated dofs in all worlds
        ct_full_to_red_map: Map from full to reduced constraint id
        actuators_u: Actuated joint velocities
        world_mask: Per-world flag to perform the computation (0 = skip)
    Outputs:
        target_cts_u: Target constraint velocities (assumed to be zero-initialized)
    """
    # Retrieve the thread indices (= world index, joint index)
    wd_id, jt_id_loc = wp.tid()

    if wd_id < world_mask.shape[0] and world_mask[wd_id] != 0 and jt_id_loc < num_joints[wd_id]:
        # Retrieve the joint model data
        jt_id_tot = first_joint_id[wd_id] + jt_id_loc
        if joints_act_type[jt_id_tot] == JointActuationType.PASSIVE:
            return
        dof_type_j = joints_dof_type[jt_id_tot]
        offset_u_j = actuated_dofs_offset[jt_id_tot]
        offset_cts_j = ct_full_to_red_map[6 * jt_id_tot]

        if dof_type_j == FKJointDoFType.CARTESIAN:
            target_cts_u[wd_id, offset_cts_j] = actuators_u[offset_u_j]
            target_cts_u[wd_id, offset_cts_j + 1] = actuators_u[offset_u_j + 1]
            target_cts_u[wd_id, offset_cts_j + 2] = actuators_u[offset_u_j + 2]
        elif dof_type_j == FKJointDoFType.CYLINDRICAL:
            target_cts_u[wd_id, offset_cts_j] = actuators_u[offset_u_j]
            target_cts_u[wd_id, offset_cts_j + 3] = actuators_u[offset_u_j + 1]
        elif dof_type_j == FKJointDoFType.FIXED:
            pass  # No dofs to apply
        elif dof_type_j == FKJointDoFType.FREE:
            target_cts_u[wd_id, offset_cts_j] = actuators_u[offset_u_j]
            target_cts_u[wd_id, offset_cts_j + 1] = actuators_u[offset_u_j + 1]
            target_cts_u[wd_id, offset_cts_j + 2] = actuators_u[offset_u_j + 2]
            target_cts_u[wd_id, offset_cts_j + 3] = actuators_u[offset_u_j + 3]
            target_cts_u[wd_id, offset_cts_j + 4] = actuators_u[offset_u_j + 4]
            target_cts_u[wd_id, offset_cts_j + 5] = actuators_u[offset_u_j + 5]
        elif dof_type_j == FKJointDoFType.PRISMATIC:
            target_cts_u[wd_id, offset_cts_j] = actuators_u[offset_u_j]
        elif dof_type_j == FKJointDoFType.REVOLUTE:
            target_cts_u[wd_id, offset_cts_j + 3] = actuators_u[offset_u_j]
        elif dof_type_j == FKJointDoFType.SPHERICAL:
            target_cts_u[wd_id, offset_cts_j + 3] = actuators_u[offset_u_j]
            target_cts_u[wd_id, offset_cts_j + 4] = actuators_u[offset_u_j + 1]
            target_cts_u[wd_id, offset_cts_j + 5] = actuators_u[offset_u_j + 2]
        elif dof_type_j == FKJointDoFType.UNIVERSAL:
            target_cts_u[wd_id, offset_cts_j + 3] = actuators_u[offset_u_j]
            target_cts_u[wd_id, offset_cts_j + 4] = actuators_u[offset_u_j + 1]
        else:
            assert False, "Unexpected actuator dof type"  # noqa: B011


@wp.kernel
def _eval_body_velocities(
    # Inputs
    num_bodies: wp.array[wp.int32],
    first_body_id: wp.array[wp.int32],
    bodies_q: wp.array[wp.transformf],
    bodies_q_dot: wp.array2d[wp.float32],
    world_mask: wp.array[wp.int32],
    # Outputs
    bodies_u: wp.array[vec6f],
):
    """
    A kernel computing the body velocities (twists) from the time derivative of body poses,
    computing in particular angular velocities omega = 2G(q)q_dot

    Inputs:
        num_bodies: Number of bodies per world
        first_body_id: First body id per world
        bodies_q: Body poses
        bodies_q_dot: Time derivative of body poses
        world_mask: Per-world flag to perform the computation (0 = skip)
    Outputs:
        bodies_u: Body velocities (twists)
    """
    wd_id, rb_id_loc = wp.tid()  # Thread indices (= world index, body index)
    if wd_id < world_mask.shape[0] and world_mask[wd_id] != 0 and rb_id_loc < num_bodies[wd_id]:
        # Indices / offsets
        rb_id_tot = first_body_id[wd_id] + rb_id_loc
        offset_q_dot = 7 * rb_id_loc

        # Copy linear velocity
        bodies_u[rb_id_tot][0] = bodies_q_dot[wd_id, offset_q_dot]
        bodies_u[rb_id_tot][1] = bodies_q_dot[wd_id, offset_q_dot + 1]
        bodies_u[rb_id_tot][2] = bodies_q_dot[wd_id, offset_q_dot + 2]

        # Compute angular velocities
        q = wp.transform_get_rotation(bodies_q[rb_id_tot])
        q_dot = wp.vec4f(
            bodies_q_dot[wd_id, offset_q_dot + 3],
            bodies_q_dot[wd_id, offset_q_dot + 4],
            bodies_q_dot[wd_id, offset_q_dot + 5],
            bodies_q_dot[wd_id, offset_q_dot + 6],
        )
        omega = 2.0 * (G_of(q) * q_dot)
        bodies_u[rb_id_tot][3] = omega[0]
        bodies_u[rb_id_tot][4] = omega[1]
        bodies_u[rb_id_tot][5] = omega[2]


@wp.kernel
def _update_cg_tolerance_kernel(
    # Input
    max_constraint: wp.array[wp.float32],
    world_mask: wp.array[wp.int32],
    # Output
    atol: wp.array[wp.float32],
    rtol: wp.array[wp.float32],
):
    """
    A kernel heuristically adapting the CG tolerance based on the current constraint residual
    (starting with a loose tolerance, and tightening it as we converge)
    Note: needs to be refined, until then we are still using a fixed tolerance
    """
    wd_id = wp.tid()
    if wd_id >= world_mask.shape[0] or world_mask[wd_id] == 0:
        return
    tol = wp.max(1e-8, wp.min(1e-5, 1e-3 * max_constraint[wd_id]))
    atol[wd_id] = tol
    rtol[wd_id] = tol
