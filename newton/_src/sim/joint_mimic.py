# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import warnings

import warp as wp

from .articulation import (
    invert_2d_rotational_dofs,
    invert_3d_rotational_dofs,
    transform_2d_rotational_axes,
    transform_3d_rotational_axes,
)
from .enums import JointType
from .model import Model
from .state import State

_SUPPORTED_JOINT_TYPES = {int(JointType.PRISMATIC), int(JointType.REVOLUTE), int(JointType.D6)}
_MAX_REPORTED_UNSUPPORTED_JOINTS = 10


@wp.kernel
def eval_mimic_joints(
    joint_mimic_joint: wp.array[int],
    joint_mimic_coeffs: wp.array[wp.vec2],
    joint_q_start: wp.array[int],
    joint_qd_start: wp.array[int],
    # outputs
    joint_q: wp.array[float],
    joint_qd: wp.array[float],
):
    """Apply joint-owned mimic relationships to generalized coordinates."""
    joint = wp.tid()
    reference_joint = joint_mimic_joint[joint]
    if reference_joint < 0:
        return

    coeffs = joint_mimic_coeffs[joint]
    q_start = joint_q_start[joint]
    reference_q_start = joint_q_start[reference_joint]
    for coordinate in range(joint_q_start[joint + 1] - q_start):
        joint_q[q_start + coordinate] = coeffs[0] + coeffs[1] * joint_q[reference_q_start + coordinate]

    qd_start = joint_qd_start[joint]
    reference_qd_start = joint_qd_start[reference_joint]
    for dof in range(joint_qd_start[joint + 1] - qd_start):
        joint_qd[qd_start + dof] = coeffs[1] * joint_qd[reference_qd_start + dof]


def eval_mimic(model: Model, state_in: State, state_out: State | None = None) -> None:
    """Update follower joint coordinates from their reference joints.

    For each follower, this function reads every position and velocity
    coordinate of the reference joint, then writes the matching follower
    coordinates according to :attr:`Model.joint_mimic_coeffs`. Independent
    joints are left unchanged. Only :attr:`State.joint_q` and
    :attr:`State.joint_qd` are written.

    If ``state_out`` is omitted, ``state_in`` is updated in place. Otherwise,
    all joint coordinates are first copied from ``state_in`` to ``state_out``
    and the followers are updated in ``state_out``.

    Args:
        model: Model containing the joint mimic metadata.
        state_in: State providing the input joint coordinates.
        state_out: State receiving the updated joint coordinates. If ``None``,
            update ``state_in`` in place.

    Raises:
        ValueError: If either state does not contain joint coordinate arrays.
    """
    if state_in.joint_q is None or state_in.joint_qd is None:
        raise ValueError("state_in must contain joint_q and joint_qd arrays")

    if state_out is None:
        state_out = state_in
    elif state_out.joint_q is None or state_out.joint_qd is None:
        raise ValueError("state_out must contain joint_q and joint_qd arrays")
    elif state_out is not state_in:
        state_out.joint_q.assign(state_in.joint_q)
        state_out.joint_qd.assign(state_in.joint_qd)

    if model.joint_count == 0:
        return

    wp.launch(
        kernel=eval_mimic_joints,
        dim=model.joint_count,
        inputs=[
            model.joint_mimic_joint,
            model.joint_mimic_coeffs,
            model.joint_q_start,
            model.joint_qd_start,
        ],
        outputs=[state_out.joint_q, state_out.joint_qd],
        device=model.device,
    )


@wp.func
def eval_joint_mimic_coordinate(
    joint: int,
    component: int,
    body_q: wp.array[wp.transform],
    body_com: wp.array[wp.vec3],
    joint_type: wp.array[int],
    joint_parent: wp.array[int],
    joint_child: wp.array[int],
    joint_X_p: wp.array[wp.transform],
    joint_X_c: wp.array[wp.transform],
    joint_qd_start: wp.array[int],
    joint_dof_dim: wp.array2d[int],
    joint_axis: wp.array[wp.vec3],
):
    """Return one joint coordinate and its parent/child maximal-coordinate gradients."""
    type = joint_type[joint]
    parent = joint_parent[joint]
    child = joint_child[joint]

    X_wp = joint_X_p[joint]
    pose_p = X_wp
    if parent >= 0:
        pose_p = body_q[parent]
        X_wp = pose_p * X_wp
    pose_c = body_q[child]
    X_wc = pose_c * joint_X_c[joint]

    q_p = wp.transform_get_rotation(X_wp)
    q_c = wp.transform_get_rotation(X_wc)
    rel_q = wp.quat_inverse(q_p) * q_c
    x_err = wp.transform_get_translation(X_wc) - wp.transform_get_translation(X_wp)
    x_err_p = wp.quat_rotate_inv(q_p, x_err)

    qd_start = joint_qd_start[joint]
    lin_axis_count = joint_dof_dim[joint, 0]
    ang_axis_count = joint_dof_dim[joint, 1]

    coordinate = float(0.0)
    linear_axis = wp.vec3(0.0)
    angular_covector = wp.vec3(0.0)

    if type == JointType.PRISMATIC:
        axis = joint_axis[qd_start]
        coordinate = wp.dot(x_err_p, axis)
        linear_axis = wp.quat_rotate(q_p, axis)
    elif type == JointType.REVOLUTE:
        axis = joint_axis[qd_start]
        coordinate = wp.quat_twist_angle_signed(axis, rel_q)
        angular_covector = wp.quat_rotate(q_p, axis)
    elif type == JointType.D6:
        if component < lin_axis_count:
            axis = joint_axis[qd_start + component]
            coordinate = wp.dot(x_err_p, axis)
            linear_axis = wp.quat_rotate(q_p, axis)
        else:
            angular_component = component - lin_axis_count
            angular_start = qd_start + lin_axis_count
            local_covector = wp.vec3(0.0)
            if ang_axis_count == 1:
                axis = joint_axis[angular_start]
                coordinate = wp.quat_twist_angle_signed(axis, rel_q)
                local_covector[0] = axis[0]
                local_covector[1] = axis[1]
                local_covector[2] = axis[2]
            elif ang_axis_count == 2:
                axis_0 = joint_axis[angular_start + 0]
                axis_1 = joint_axis[angular_start + 1]
                coordinates_2, _unused_velocities_2 = invert_2d_rotational_dofs(axis_0, axis_1, q_p, q_c, wp.vec3(0.0))
                coordinate = coordinates_2[angular_component]
                axis_0_q, axis_1_q = transform_2d_rotational_axes(axis_0, axis_1, coordinates_2[0])
                if angular_component == 0:
                    local_covector[0] = axis_0_q[0]
                    local_covector[1] = axis_0_q[1]
                    local_covector[2] = axis_0_q[2]
                else:
                    local_covector[0] = axis_1_q[0]
                    local_covector[1] = axis_1_q[1]
                    local_covector[2] = axis_1_q[2]
            elif ang_axis_count == 3:
                axis_0 = joint_axis[angular_start + 0]
                axis_1 = joint_axis[angular_start + 1]
                axis_2 = joint_axis[angular_start + 2]
                coordinates_3, _unused_velocities_3 = invert_3d_rotational_dofs(
                    axis_0, axis_1, axis_2, q_p, q_c, wp.vec3(0.0)
                )
                coordinate = coordinates_3[angular_component]
                axis_0_q, axis_1_q, axis_2_q = transform_3d_rotational_axes(
                    axis_0, axis_1, axis_2, coordinates_3[0], coordinates_3[1]
                )
                if angular_component == 0:
                    local_covector[0] = axis_0_q[0]
                    local_covector[1] = axis_0_q[1]
                    local_covector[2] = axis_0_q[2]
                elif angular_component == 1:
                    local_covector[0] = axis_1_q[0]
                    local_covector[1] = axis_1_q[1]
                    local_covector[2] = axis_1_q[2]
                else:
                    local_covector[0] = axis_2_q[0]
                    local_covector[1] = axis_2_q[1]
                    local_covector[2] = axis_2_q[2]
            angular_covector = wp.quat_rotate(q_p, local_covector)

    r_p = wp.vec3(0.0)
    if parent >= 0:
        r_p = wp.transform_get_translation(X_wp) - wp.transform_point(pose_p, body_com[parent])
    r_c = wp.transform_get_translation(X_wc) - wp.transform_point(pose_c, body_com[child])

    gradient_parent = wp.spatial_vector(-linear_axis, -wp.cross(r_p, linear_axis) - angular_covector)
    gradient_child = wp.spatial_vector(linear_axis, wp.cross(r_c, linear_axis) + angular_covector)
    return coordinate, gradient_parent, gradient_child


@wp.func
def eval_joint_mimic_velocity(
    parent: int,
    child: int,
    parent_gradient: wp.spatial_vector,
    child_gradient: wp.spatial_vector,
    body_qd: wp.array[wp.spatial_vector],
) -> float:
    """Return one joint-coordinate velocity from maximal body velocities."""
    velocity = float(0.0)
    if parent >= 0:
        parent_twist = body_qd[parent]
        velocity += wp.dot(wp.spatial_top(parent_gradient), wp.spatial_top(parent_twist))
        velocity += wp.dot(wp.spatial_bottom(parent_gradient), wp.spatial_bottom(parent_twist))
    if child >= 0:
        child_twist = body_qd[child]
        velocity += wp.dot(wp.spatial_top(child_gradient), wp.spatial_top(child_twist))
        velocity += wp.dot(wp.spatial_bottom(child_gradient), wp.spatial_bottom(child_twist))
    return velocity


def has_supported_joint_mimics(model: Model, solver_name: str) -> bool:
    """Return whether a model has supported mimics and warn about unsupported ones."""
    joint_mimic_joint = model.joint_mimic_joint.numpy()
    joint_type = model.joint_type.numpy()
    has_supported = False
    unsupported_count = 0
    unsupported_sample = []
    for follower, reference in enumerate(joint_mimic_joint):
        if reference < 0:
            continue
        if int(joint_type[follower]) in _SUPPORTED_JOINT_TYPES and int(joint_type[reference]) in _SUPPORTED_JOINT_TYPES:
            has_supported = True
            continue
        unsupported_count += 1
        if len(unsupported_sample) < _MAX_REPORTED_UNSUPPORTED_JOINTS:
            unsupported_sample.append(follower)

    if unsupported_count:
        omitted_count = unsupported_count - len(unsupported_sample)
        omitted_suffix = f"; {omitted_count} additional indices omitted" if omitted_count else ""
        warnings.warn(
            f"{solver_name} ignores joint-owned mimic relationships unless both joints are PRISMATIC, "
            f"REVOLUTE, or D6; unsupported follower joint indices: {unsupported_sample}{omitted_suffix}.",
            stacklevel=3,
        )
    return has_supported
