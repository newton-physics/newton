# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Project rigid-body inertia through user-defined body-twist bases."""

import warp as wp


@wp.kernel
def _eval_body_inertia_projection_kernel(
    body_world_start: wp.array[wp.int32],
    body_mass: wp.array[wp.float32],
    body_inertia: wp.array[wp.mat33f],
    body_q: wp.array[wp.transformf],
    body_velocity_basis: wp.array2d[wp.spatial_vectorf],
    world_mask: wp.array[wp.bool],
    projection: wp.array3d[wp.float32],
):
    world_id, row, column = wp.tid()
    if column < row:
        return

    value = float(0.0)
    if not world_mask or world_mask[world_id]:
        body_start = body_world_start[world_id]
        body_end = body_world_start[world_id + 1]
        for body_id in range(body_start, body_end):
            twist_row = body_velocity_basis[row, body_id]
            twist_column = body_velocity_basis[column, body_id]
            linear_row = wp.spatial_top(twist_row)
            linear_column = wp.spatial_top(twist_column)

            rotation = wp.transform_get_rotation(body_q[body_id])
            angular_row = wp.quat_rotate_inv(rotation, wp.spatial_bottom(twist_row))
            angular_column = wp.quat_rotate_inv(rotation, wp.spatial_bottom(twist_column))

            value += body_mass[body_id] * wp.dot(linear_row, linear_column)
            value += wp.dot(angular_row, body_inertia[body_id] @ angular_column)

    projection[world_id, row, column] = value
    projection[world_id, column, row] = value


def eval_body_inertia_projection(
    body_world_start: wp.array[wp.int32],
    body_mass: wp.array[wp.float32],
    body_inertia: wp.array[wp.mat33f],
    body_q: wp.array[wp.transformf],
    body_velocity_basis: wp.array2d[wp.spatial_vectorf],
    projection: wp.array3d[wp.float32],
    world_mask: wp.array[wp.bool] | None = None,
) -> None:
    """Launch the body-inertia projection kernel."""
    wp.launch(
        _eval_body_inertia_projection_kernel,
        dim=(projection.shape[0], projection.shape[1], projection.shape[2]),
        inputs=[
            body_world_start,
            body_mass,
            body_inertia,
            body_q,
            body_velocity_basis,
            world_mask,
        ],
        outputs=[projection],
        device=projection.device,
    )
