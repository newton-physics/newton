# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Shared Warp kernels used by more than one controller family.

Every 1-D buffer here is compact — one entry per controlled DOF, robot 0's DOFs
first, then robot 1's — so every kernel is a flat 1-D launch with no padding to
skip. The exception is a padded per-robot matrix (e.g. a mass matrix), which
:func:`~newton.eval_mass_matrix` produces as one square block per articulation:
:func:`_block_matrix_vector_multiply_kernel` stays a flat 1-D launch and indexes
into those blocks, while the gather kernels launch over them directly.

A kernel belongs here, rather than in one controller family's own ``_common.py``,
once a second family needs the identical operation — see
:class:`~newton.controllers.ControllerJointImpedanceModelFree` (joint-space PD,
compact-vector accumulation, and the mass-matrix multiply/gather) and the
operational-space controller family's null-space posture term (the same
joint-space PD and accumulation, plus the same block-matrix-vector multiply for
its ``N @ (M @ a)`` combine step) for the two current users.
"""

from __future__ import annotations

from typing import Any

import warp as wp

from ...core.types import Devicelike


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
def _block_matrix_vector_multiply_kernel(
    block_matrix: wp.array3d[wp.float32],  # (controlled_robot_count, max_controlled_dofs, max_controlled_dofs)
    vec: wp.array[wp.float32],  # (total_controlled_dofs,)
    robot_of_dof: wp.array[wp.int32],  # (total_controlled_dofs,) -> owning robot
    slot_of_dof: wp.array[wp.int32],  # (total_controlled_dofs,) -> row within that robot's block
    dof_offsets: wp.array[wp.int32],  # (controlled_robot_count,) -> first flat DOF of each robot
    controlled_dofs_per_robot: wp.array[wp.int32],  # (controlled_robot_count,)
    out: wp.array[wp.float32],  # (total_controlled_dofs,)
):
    """Multiply a compact per-DOF vector by a padded per-robot square matrix, ``out = block_matrix @ vec``.

    ``block_matrix`` need not be a mass matrix — any per-robot square matrix in
    the same padded ``(controlled_robot_count, max_controlled_dofs,
    max_controlled_dofs)`` layout works, e.g. a null-space projector.
    """
    dof = wp.tid()
    robot = robot_of_dof[dof]
    row = slot_of_dof[dof]
    row_base = dof_offsets[robot]
    acc = float(0.0)
    for col in range(controlled_dofs_per_robot[robot]):
        acc = acc + block_matrix[robot, row, col] * vec[row_base + col]
    out[dof] = acc


@wp.kernel
def _gather_mass_matrix_blocks_kernel(
    model_mass_matrix: wp.array3d[wp.float32],  # (model_robot_count, model_max_dofs, model_max_dofs)
    model_robot_index: wp.array[wp.int32],  # (controlled_robot_count,) -> that robot's index in the model
    articulation_dof_idx_of_padded_dof_idx: wp.array2d[
        wp.int32
    ],  # (controlled_robot_count, max_controlled_dofs) padded_dof_idx -> DOF index within its robot
    controlled_dofs_per_robot: wp.array[wp.int32],  # (controlled_robot_count,)
    out: wp.array3d[wp.float32],  # (controlled_robot_count, max_controlled_dofs, max_controlled_dofs)
):
    robot, padded_row_dof_idx, padded_col_dof_idx = wp.tid()
    if padded_row_dof_idx >= controlled_dofs_per_robot[robot] or padded_col_dof_idx >= controlled_dofs_per_robot[robot]:
        return
    model_robot = model_robot_index[robot]
    articulation_row_dof_idx = articulation_dof_idx_of_padded_dof_idx[robot, padded_row_dof_idx]
    articulation_col_dof_idx = articulation_dof_idx_of_padded_dof_idx[robot, padded_col_dof_idx]
    out[robot, padded_row_dof_idx, padded_col_dof_idx] = model_mass_matrix[
        model_robot, articulation_row_dof_idx, articulation_col_dof_idx
    ]


# wp.copy is not recordable under APIC graph capture when either side is
# non-contiguous, which every indexed-view port is. These two kernels do the
# same work in a form that captures and serialises. Controllers launch them
# at their own port length: one entry per controlled DOF for a compact port, one
# per model coordinate or DOF for a model-based controller's whole-model ports.


@wp.kernel
def _gather_rank1_port_kernel(
    port: wp.indexedarray(dtype=Any),  # view of a simulation-sized array
    out: wp.array[Any],  # one entry per element the view addresses
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


# dtype -> (rank -> gather kernel), the set of dtype/rank combinations any
# controller's ports currently use. Extend this table, not _read_port itself,
# when a controller needs a new port dtype or rank. Every rank-1 dtype shares
# _gather_rank1_port_kernel: it's generic over dtype (Any), so Warp compiles
# one concrete kernel per dtype the table actually uses, from a single body.
_GATHER_KERNELS_BY_DTYPE_AND_RANK = {
    wp.float32: {1: _gather_rank1_port_kernel, 3: _gather_mass_matrix_port_kernel},
    wp.transform: {1: _gather_rank1_port_kernel},
    wp.spatial_vector: {1: _gather_rank1_port_kernel},
    wp.quat: {1: _gather_rank1_port_kernel},
}


def _read_port(
    port: wp.array | wp.indexedarray,
    buffer: wp.array,
    shape: int | tuple[int, ...],
    device: Devicelike,
) -> None:
    """Copy a bound port into an internal buffer, whatever it is bound to.

    A view has to go through a kernel: :func:`warp.copy` is not recordable under
    APIC graph capture when either side is non-contiguous, so using it here would
    make a controller that reports ``is_graphable()`` fail to export.

    Args:
        port: The caller-bound port, a :class:`warp.array` or a view of one.
            Any dtype/rank combination in :data:`_GATHER_KERNELS_BY_DTYPE_AND_RANK`
            is supported when ``port`` is a view; a plain array supports any
            dtype/rank, since :func:`warp.copy` doesn't care.
        buffer: Destination, matching ``port`` in shape and dtype.
        shape: Launch shape — the length for a 1-D port, ``(robots, rows, cols)``
            for a padded per-robot matrix.
        device: Device to launch on.
    """
    if not isinstance(port, wp.indexedarray):
        wp.copy(buffer, port)
        return

    # A kernel parameter's dtype and dimensionality are part of its type, so
    # a view needs the kernel that matches both.
    kernels_by_rank = _GATHER_KERNELS_BY_DTYPE_AND_RANK.get(port.dtype)
    kernel = kernels_by_rank.get(port.ndim) if kernels_by_rank is not None else None
    if kernel is None:
        raise TypeError(
            f"_read_port has no gather kernel for a {port.ndim}-D indexed array of dtype {port.dtype}; "
            f"add one to _GATHER_KERNELS_BY_DTYPE_AND_RANK in controllers/impl/_common.py."
        )
    wp.launch(kernel, dim=shape, inputs=[port], outputs=[buffer], device=device)
