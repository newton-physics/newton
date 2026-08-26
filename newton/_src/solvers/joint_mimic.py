# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import warnings

import warp as wp

from ..sim import JointType, Model

_SUPPORTED_JOINT_TYPES = {int(JointType.PRISMATIC), int(JointType.REVOLUTE), int(JointType.D6)}


def has_supported_joint_mimics(model: Model, solver_name: str) -> bool:
    """Return whether a model has supported mimics and warn about unsupported ones."""
    joint_mimic_joint = model.joint_mimic_joint.numpy()
    joint_type = model.joint_type.numpy()
    mimic_followers = [follower for follower, reference in enumerate(joint_mimic_joint) if reference >= 0]
    supported_mimic_followers = [
        follower
        for follower in mimic_followers
        if int(joint_type[follower]) in _SUPPORTED_JOINT_TYPES
        and int(joint_type[joint_mimic_joint[follower]]) in _SUPPORTED_JOINT_TYPES
    ]
    unsupported_mimic_followers = sorted(set(mimic_followers) - set(supported_mimic_followers))
    if unsupported_mimic_followers:
        warnings.warn(
            f"{solver_name} ignores joint-owned mimic relationships unless both joints are PRISMATIC, "
            f"REVOLUTE, or D6; unsupported follower joint indices: {unsupported_mimic_followers}.",
            stacklevel=3,
        )
    return bool(supported_mimic_followers)


@wp.kernel
def reduce_mimic_inertia(
    articulation_start: wp.array[int],
    articulation_end: wp.array[int],
    articulation_H_start: wp.array[int],
    articulation_H_rows: wp.array[int],
    articulation_dof_start: wp.array[int],
    joint_qd_start: wp.array[int],
    joint_mimic_joint: wp.array[int],
    joint_mimic_coeffs: wp.array[wp.vec2],
    joint_armature: wp.array[float],
    H: wp.array[float],
    H_reduced: wp.array[float],
):
    """Reduce a Featherstone inertia matrix to its independent coordinates."""
    articulation = wp.tid()
    joint_start = articulation_start[articulation]
    joint_end = articulation_end[articulation]
    matrix_start = articulation_H_start[articulation]
    matrix_rows = articulation_H_rows[articulation]
    dof_start = articulation_dof_start[articulation]

    for row_joint in range(joint_start, joint_end):
        row_reference = joint_mimic_joint[row_joint]
        row_multiplier = float(1.0)
        row_destination_start = joint_qd_start[row_joint]
        if row_reference >= 0:
            row_multiplier = joint_mimic_coeffs[row_joint][1]
            row_destination_start = joint_qd_start[row_reference]

        row_count = joint_qd_start[row_joint + 1] - joint_qd_start[row_joint]
        for row_axis in range(row_count):
            source_row = joint_qd_start[row_joint] + row_axis
            destination_row = row_destination_start + row_axis

            for column_joint in range(joint_start, joint_end):
                column_reference = joint_mimic_joint[column_joint]
                column_multiplier = float(1.0)
                column_destination_start = joint_qd_start[column_joint]
                if column_reference >= 0:
                    column_multiplier = joint_mimic_coeffs[column_joint][1]
                    column_destination_start = joint_qd_start[column_reference]

                column_count = joint_qd_start[column_joint + 1] - joint_qd_start[column_joint]
                for column_axis in range(column_count):
                    source_column = joint_qd_start[column_joint] + column_axis
                    destination_column = column_destination_start + column_axis
                    value = H[matrix_start + (source_row - dof_start) * matrix_rows + (source_column - dof_start)]
                    if source_row == source_column:
                        value += joint_armature[source_row]
                    wp.atomic_add(
                        H_reduced,
                        matrix_start + (destination_row - dof_start) * matrix_rows + (destination_column - dof_start),
                        row_multiplier * column_multiplier * value,
                    )

    # Follower columns are unused by the reduced mapping. Give each one a unit
    # diagonal so the fixed-size matrix remains positive definite.
    for joint in range(joint_start, joint_end):
        if joint_mimic_joint[joint] >= 0:
            dof_count = joint_qd_start[joint + 1] - joint_qd_start[joint]
            for axis in range(dof_count):
                follower_dof = joint_qd_start[joint] + axis - dof_start
                H_reduced[matrix_start + follower_dof * matrix_rows + follower_dof] = 1.0


@wp.kernel
def reduce_mimic_forces(
    articulation_start: wp.array[int],
    articulation_end: wp.array[int],
    joint_qd_start: wp.array[int],
    joint_mimic_joint: wp.array[int],
    joint_mimic_coeffs: wp.array[wp.vec2],
    joint_tau: wp.array[float],
    joint_tau_reduced: wp.array[float],
):
    """Transfer follower generalized forces to their independent references."""
    articulation = wp.tid()
    joint_start = articulation_start[articulation]
    joint_end = articulation_end[articulation]

    for joint in range(joint_start, joint_end):
        reference = joint_mimic_joint[joint]
        multiplier = float(1.0)
        destination_start = joint_qd_start[joint]
        if reference >= 0:
            multiplier = joint_mimic_coeffs[joint][1]
            destination_start = joint_qd_start[reference]

        dof_count = joint_qd_start[joint + 1] - joint_qd_start[joint]
        for axis in range(dof_count):
            wp.atomic_add(
                joint_tau_reduced,
                destination_start + axis,
                multiplier * joint_tau[joint_qd_start[joint] + axis],
            )


@wp.kernel
def expand_mimic_accelerations(
    joint_qd_start: wp.array[int],
    joint_mimic_joint: wp.array[int],
    joint_mimic_coeffs: wp.array[wp.vec2],
    joint_qdd: wp.array[float],
):
    """Expand independent accelerations into follower coordinates."""
    joint = wp.tid()
    reference = joint_mimic_joint[joint]
    if reference < 0:
        return

    multiplier = joint_mimic_coeffs[joint][1]
    dof_count = joint_qd_start[joint + 1] - joint_qd_start[joint]
    for axis in range(dof_count):
        joint_qdd[joint_qd_start[joint] + axis] = multiplier * joint_qdd[joint_qd_start[reference] + axis]


@wp.kernel
def apply_joint_mimic_deltas(
    body_q: wp.array[wp.transform],
    body_qd: wp.array[wp.spatial_vector],
    body_com: wp.array[wp.vec3],
    body_inv_m: wp.array[float],
    body_inv_I: wp.array[wp.mat33],
    deltas: wp.array[wp.spatial_vector],
    dt: float,
):
    """Apply velocity-like mimic corrections to maximal body state in place."""
    body = wp.tid()
    inv_m = body_inv_m[body]
    if inv_m == 0.0:
        return

    pose = body_q[body]
    rotation = wp.transform_get_rotation(pose)
    delta = deltas[body]
    linear_delta = wp.spatial_top(delta) * inv_m
    angular_delta = wp.quat_rotate(
        rotation,
        body_inv_I[body] * wp.quat_rotate_inv(rotation, wp.spatial_bottom(delta)),
    )

    rotation_new = wp.normalize(rotation + 0.5 * wp.quat(angular_delta * dt, 0.0) * rotation)
    com = body_com[body]
    com_world = wp.transform_get_translation(pose) + wp.quat_rotate(rotation, com)
    position_new = com_world + linear_delta * dt - wp.quat_rotate(rotation_new, com)

    velocity = body_qd[body]
    body_q[body] = wp.transform(position_new, rotation_new)
    body_qd[body] = wp.spatial_vector(
        wp.spatial_top(velocity) + linear_delta,
        wp.spatial_bottom(velocity) + angular_delta,
    )


def project_joint_mimics(
    model: Model,
    body_q: wp.array[wp.transform],
    body_qd: wp.array[wp.spatial_vector],
    body_inv_m: wp.array[float],
    body_inv_I: wp.array[wp.mat33],
    deltas: wp.array[wp.spatial_vector],
    dt: float,
    iterations: int,
) -> None:
    """Project supported mimic relationships in maximal coordinates."""
    # Import lazily so importing another solver does not initialize SolverXPBD.
    from .xpbd.kernels import solve_joint_mimics  # noqa: PLC0415

    for _ in range(iterations):
        deltas.zero_()
        wp.launch(
            kernel=solve_joint_mimics,
            dim=model.joint_count,
            inputs=[
                body_q,
                model.body_com,
                body_inv_m,
                body_inv_I,
                model.joint_type,
                model.joint_enabled,
                model.joint_parent,
                model.joint_child,
                model.joint_X_p,
                model.joint_X_c,
                model.joint_qd_start,
                model.joint_dof_dim,
                model.joint_axis,
                model.joint_mimic_joint,
                model.joint_mimic_coeffs,
                1.0,
                1.0,
                dt,
            ],
            outputs=[deltas, None],
            device=model.device,
        )
        wp.launch(
            kernel=apply_joint_mimic_deltas,
            dim=model.body_count,
            inputs=[body_q, body_qd, model.body_com, body_inv_m, body_inv_I, deltas, dt],
            device=model.device,
        )
