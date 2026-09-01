# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Shared Warp kernels for the joint impedance controllers.

Every 1-D buffer here is compact — one entry per controlled DOF, robot 0's DOFs
first, then robot 1's — so every kernel is a flat 1-D launch with no padding to
skip. The exception is the mass matrix, which :func:`~newton.eval_mass_matrix`
produces as one square block per articulation: the multiply kernel stays flat
and indexes into those blocks, while the gather kernel launches over them.
"""

from __future__ import annotations

import warp as wp

from ....core.types import Devicelike


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
    mass_matrix: wp.array3d[wp.float32],  # (controlled_robot_count, max_controlled_dofs, max_controlled_dofs)
    vec: wp.array[wp.float32],  # (total_controlled_dofs,)
    robot_of_dof: wp.array[wp.int32],  # (total_controlled_dofs,) -> owning robot
    slot_of_dof: wp.array[wp.int32],  # (total_controlled_dofs,) -> row within that robot's block
    dof_offsets: wp.array[wp.int32],  # (controlled_robot_count,) -> first flat DOF of each robot
    controlled_dofs_per_robot: wp.array[wp.int32],  # (controlled_robot_count,)
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
# same work in a form that captures and serialises. Both controllers launch them
# at their own port length: one entry per controlled DOF for a compact port, one
# per model coordinate or DOF for the model-based controller's whole-model ports.


@wp.kernel
def _gather_port_kernel(
    port: wp.indexedarray[wp.float32],  # view of a simulation-sized array
    out: wp.array[wp.float32],  # one entry per element the view addresses
):
    dof = wp.tid()
    out[dof] = port[dof]


@wp.kernel
def _gather_mass_matrix_port_kernel(
    port: wp.indexedarray(dtype=wp.float32, ndim=3),  # view selecting robots from a larger set of blocks
    out: wp.array3d[wp.float32],  # (controlled_robot_count, max_controlled_dofs, max_controlled_dofs)
):
    robot, row, col = wp.tid()
    out[robot, row, col] = port[robot, row, col]


@wp.kernel
def _scatter_port_kernel(
    values: wp.array[wp.float32],  # one entry per element the view addresses
    port: wp.indexedarray[wp.float32],  # view of a simulation-sized array
):
    dof = wp.tid()
    port[dof] = values[dof]


def _read_port(
    port: wp.array[wp.float32] | wp.array3d[wp.float32] | wp.indexedarray[wp.float32],
    buffer: wp.array[wp.float32] | wp.array3d[wp.float32],
    shape: int | tuple[int, ...],
    device: Devicelike,
) -> None:
    """Copy a bound port into an internal buffer, whatever it is bound to.

    A view has to go through a kernel: :func:`warp.copy` is not recordable under
    APIC graph capture when either side is non-contiguous, so using it here would
    make a controller that reports ``is_graphable()`` fail to export.

    Args:
        port: The caller-bound port, a :class:`warp.array` or a view of one.
            1-D for a compact or whole-model port, 3-D for a mass matrix; a 3-D
            view has no bracket spelling and is
            ``wp.indexedarray(dtype=wp.float32, ndim=3)``.
        buffer: Destination, matching ``port`` in shape and dtype.
        shape: Launch shape — the length for a 1-D port, ``(robots, rows, cols)``
            for a mass matrix.
        device: Device to launch on.
    """
    if not isinstance(port, wp.indexedarray):
        wp.copy(buffer, port)
        return

    # A kernel parameter's dimensionality is part of its type, so a view needs
    # the kernel that matches its rank.
    kernel = _gather_port_kernel if port.ndim == 1 else _gather_mass_matrix_port_kernel
    wp.launch(kernel, dim=shape, inputs=[port], outputs=[buffer], device=device)
