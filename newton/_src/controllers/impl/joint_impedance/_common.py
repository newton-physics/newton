# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Shared Warp kernels for :class:`~newton.controllers.ControllerJointImpedance`.

Every buffer here is compact — one entry per controlled DOF, robot 0's DOFs
first, then robot 1's — so the elementwise kernels are flat 1-D launches with no
padding to skip. Only the mass-matrix kernels are per-robot blocked, since
:func:`~newton.eval_mass_matrix` genuinely produces one square block per
articulation.
"""

import warp as wp


@wp.kernel
def _pd_term_kernel(
    joint_q: wp.array[wp.float32],  # (total_dofs,)
    joint_qd: wp.array[wp.float32],  # (total_dofs,)
    joint_q_des: wp.array[wp.float32],  # (total_dofs,)
    joint_qd_des: wp.array[wp.float32],  # (total_dofs,)
    stiffness: wp.array[wp.float32],  # (total_dofs,)
    damping: wp.array[wp.float32],  # (total_dofs,)
    out: wp.array[wp.float32],  # (total_dofs,)
):
    dof = wp.tid()
    out[dof] = stiffness[dof] * (joint_q_des[dof] - joint_q[dof]) + damping[dof] * (joint_qd_des[dof] - joint_qd[dof])


@wp.kernel
def _add_term_kernel(
    term: wp.array[wp.float32],  # (total_dofs,)
    tau: wp.array[wp.float32],  # (total_dofs,)
):
    dof = wp.tid()
    tau[dof] = tau[dof] + term[dof]


@wp.kernel
def _mass_matrix_multiply_kernel(
    mass_matrix: wp.array3d[wp.float32],  # (robot_count, max_dofs, max_dofs)
    vec: wp.array[wp.float32],  # (total_dofs,)
    robot_of_dof: wp.array[wp.int32],  # (total_dofs,) -> owning robot
    slot_of_dof: wp.array[wp.int32],  # (total_dofs,) -> row within that robot's block
    dof_offsets: wp.array[wp.int32],  # (robot_count,) -> first flat DOF of each robot
    dofs_per_robot: wp.array[wp.int32],  # (robot_count,)
    out: wp.array[wp.float32],  # (total_dofs,)
):
    dof = wp.tid()
    robot = robot_of_dof[dof]
    row = slot_of_dof[dof]
    row_base = dof_offsets[robot]
    acc = float(0.0)
    for col in range(dofs_per_robot[robot]):
        acc = acc + mass_matrix[robot, row, col] * vec[row_base + col]
    out[dof] = acc


@wp.kernel
def _gather_mass_matrix_blocks_kernel(
    mass_matrix_full: wp.array3d[wp.float32],  # (robot_count, model_max_dofs, model_max_dofs)
    local_dof_idx: wp.array2d[wp.int32],  # (robot_count, max_dofs) -> DOF index within the articulation
    dofs_per_robot: wp.array[wp.int32],  # (robot_count,)
    out: wp.array3d[wp.float32],  # (robot_count, max_dofs, max_dofs)
):
    robot, row, col = wp.tid()
    if row >= dofs_per_robot[robot] or col >= dofs_per_robot[robot]:
        return
    out[robot, row, col] = mass_matrix_full[robot, local_dof_idx[robot, row], local_dof_idx[robot, col]]
