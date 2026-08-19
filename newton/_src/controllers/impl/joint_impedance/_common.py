# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Shared Warp kernels for the joint impedance controllers.

Every 1-D buffer here is compact — one entry per controlled DOF, robot 0's DOFs
first, then robot 1's — so every kernel is a flat 1-D launch with no padding to
skip. The exception is the mass matrix, which :func:`~newton.eval_mass_matrix`
produces as one square block per articulation: the multiply kernel stays flat
and indexes into those blocks, while the gather kernel launches over them.
"""

import warp as wp


@wp.kernel
def _pd_term_kernel(
    joint_q: wp.array[wp.float32],  # (total_controlled_dofs,)
    joint_qd: wp.array[wp.float32],  # (total_controlled_dofs,)
    joint_q_des: wp.array[wp.float32],  # (total_controlled_dofs,)
    joint_qd_des: wp.array[wp.float32],  # (total_controlled_dofs,)
    stiffness: wp.array[wp.float32],  # (total_controlled_dofs,)
    damping: wp.array[wp.float32],  # (total_controlled_dofs,)
    out: wp.array[wp.float32],  # (total_controlled_dofs,)
):
    dof = wp.tid()
    out[dof] = stiffness[dof] * (joint_q_des[dof] - joint_q[dof]) + damping[dof] * (joint_qd_des[dof] - joint_qd[dof])


@wp.kernel
def _add_term_kernel(
    term: wp.array[wp.float32],  # (total_controlled_dofs,)
    tau: wp.array[wp.float32],  # (total_controlled_dofs,)
):
    dof = wp.tid()
    tau[dof] = tau[dof] + term[dof]


@wp.kernel
def _mass_matrix_multiply_kernel(
    mass_matrix: wp.array3d[wp.float32],  # (model_robot_count, max_controlled_dofs, max_controlled_dofs)
    vec: wp.array[wp.float32],  # (total_controlled_dofs,)
    robot_of_dof: wp.array[wp.int32],  # (total_controlled_dofs,) -> owning robot
    slot_of_dof: wp.array[wp.int32],  # (total_controlled_dofs,) -> row within that robot's block
    dof_offsets: wp.array[wp.int32],  # (model_robot_count,) -> first flat DOF of each robot
    controlled_dofs_per_robot: wp.array[wp.int32],  # (model_robot_count,)
    out: wp.array[wp.float32],  # (total_controlled_dofs,)
):
    dof = wp.tid()
    robot = robot_of_dof[dof]
    row = slot_of_dof[dof]
    row_base = dof_offsets[robot]
    acc = float(0.0)
    for col in range(controlled_dofs_per_robot[robot]):
        acc = acc + mass_matrix[robot, row, col] * vec[row_base + col]
    out[dof] = acc


@wp.kernel
def _gather_mass_matrix_blocks_kernel(
    model_mass_matrix: wp.array3d[wp.float32],  # (model_robot_count, model_max_dofs, model_max_dofs)
    model_robot_index: wp.array[wp.int32],  # (controlled_robot_count,) -> that robot's index in the model
    local_dof_idx: wp.array2d[wp.int32],  # (controlled_robot_count, max_controlled_dofs) -> DOF index within its robot
    controlled_dofs_per_robot: wp.array[wp.int32],  # (controlled_robot_count,)
    out: wp.array3d[wp.float32],  # (controlled_robot_count, max_controlled_dofs, max_controlled_dofs)
):
    robot, row, col = wp.tid()
    if row >= controlled_dofs_per_robot[robot] or col >= controlled_dofs_per_robot[robot]:
        return
    model_robot = model_robot_index[robot]
    out[robot, row, col] = model_mass_matrix[model_robot, local_dof_idx[robot, row], local_dof_idx[robot, col]]


# wp.copy is not recordable under APIC graph capture when either side is
# non-contiguous, which every indexed-view port is. These two kernels do the
# same work in a form that captures and serialises.


@wp.kernel
def _gather_port_kernel(
    port: wp.indexedarray[wp.float32],  # (total_controlled_dofs,) view of a simulation-sized array
    out: wp.array[wp.float32],  # (total_controlled_dofs,)
):
    dof = wp.tid()
    out[dof] = port[dof]


@wp.kernel
def _scatter_port_kernel(
    values: wp.array[wp.float32],  # (total_controlled_dofs,)
    port: wp.indexedarray[wp.float32],  # (total_controlled_dofs,) view of a simulation-sized array
):
    dof = wp.tid()
    port[dof] = values[dof]
