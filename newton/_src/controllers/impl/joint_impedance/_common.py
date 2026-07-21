# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Shared Warp kernels for :class:`~newton.controllers.ControllerJointImpedance`."""

import numpy as np
import warp as wp


def _idx_max(idx: wp.array[wp.uint32]) -> int:
    """Return the minimum flat-array size needed to hold all indices."""
    return int(np.max(idx.numpy())) + 1


@wp.kernel
def _pd_term_kernel(
    joint_q: wp.array2d[wp.float32],  # (num_robots, max_dofs)
    joint_qd: wp.array2d[wp.float32],  # (num_robots, max_dofs)
    joint_q_des: wp.array2d[wp.float32],  # (num_robots, max_dofs)
    joint_qd_des: wp.array2d[wp.float32],  # (num_robots, max_dofs)
    stiffness: wp.array2d[wp.float32],  # (num_robots, max_dofs)
    damping: wp.array2d[wp.float32],  # (num_robots, max_dofs)
    dofs_per_robot: wp.array[wp.int32],  # (num_robots,)
    out: wp.array2d[wp.float32],  # (num_robots, max_dofs)
):
    robot, dof = wp.tid()
    if dof >= dofs_per_robot[robot]:
        return
    out[robot, dof] = stiffness[robot, dof] * (joint_q_des[robot, dof] - joint_q[robot, dof]) + damping[robot, dof] * (
        joint_qd_des[robot, dof] - joint_qd[robot, dof]
    )


@wp.kernel
def _add_term_kernel(
    term: wp.array2d[wp.float32],  # (num_robots, max_dofs)
    dofs_per_robot: wp.array[wp.int32],  # (num_robots,)
    tau: wp.array2d[wp.float32],  # (num_robots, max_dofs)
):
    robot, dof = wp.tid()
    if dof >= dofs_per_robot[robot]:
        return
    tau[robot, dof] = tau[robot, dof] + term[robot, dof]


@wp.kernel
def _mass_matrix_multiply_kernel(
    M: wp.array3d[wp.float32],  # (num_robots, max_dofs, max_dofs)
    vec: wp.array2d[wp.float32],  # (num_robots, max_dofs)
    dofs_per_robot: wp.array[wp.int32],  # (num_robots,)
    out: wp.array2d[wp.float32],  # (num_robots, max_dofs)
):
    robot, dof = wp.tid()
    if dof >= dofs_per_robot[robot]:
        return
    acc = float(0.0)
    for col in range(dofs_per_robot[robot]):
        acc = acc + M[robot, dof, col] * vec[robot, col]
    out[robot, dof] = acc


@wp.kernel
def _gather_dof_flat_kernel(
    src: wp.array[wp.float32],  # flat sim array
    indices: wp.array[wp.uint32],  # (total_dofs,) — concatenated per-robot, no padding
    dst: wp.array[wp.float32],  # flat output (total_dofs,)
):
    flat = wp.tid()
    dst[flat] = src[indices[flat]]


@wp.kernel
def _gather_dof_kernel(
    src: wp.array[wp.float32],  # flat sim array
    dof_indices: wp.array[wp.uint32],  # (total_dofs,) — concatenated per-robot indices
    dof_offsets: wp.array[wp.int32],  # (num_robots,) — start of each robot in dof_indices
    dofs_per_robot: wp.array[wp.int32],  # (num_robots,)
    dst: wp.array2d[wp.float32],  # (num_robots, max_dofs)
):
    robot, dof = wp.tid()
    if dof >= dofs_per_robot[robot]:
        return
    dst[robot, dof] = src[dof_indices[dof_offsets[robot] + dof]]


@wp.kernel
def _scatter_dof_kernel(
    src: wp.array2d[wp.float32],  # (num_robots, max_dofs)
    dof_indices: wp.array[wp.uint32],  # (total_dofs,) — concatenated per-robot indices
    dof_offsets: wp.array[wp.int32],  # (num_robots,) — start of each robot in dof_indices
    dofs_per_robot: wp.array[wp.int32],  # (num_robots,)
    dst: wp.array[wp.float32],  # flat sim output
):
    robot, dof = wp.tid()
    if dof >= dofs_per_robot[robot]:
        return
    dst[dof_indices[dof_offsets[robot] + dof]] = src[robot, dof]
