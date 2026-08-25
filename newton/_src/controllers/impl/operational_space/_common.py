# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Shared Warp kernels for the operational-space controller family.

The operational-space frame is called the *tool* frame throughout. It is
implemented as a Newton *site* (a body-fixed offset, ``tool_body`` +
``coordinate_change_body_from_tool``), resolved once per robot, one entry per
robot (one task per environment). "Site" only appears where these kernels are
literally reading that underlying Newton primitive; every resolved,
per-step quantity is named "tool". These kernels shift the COM twists
:func:`~newton.eval_jacobian` and :class:`~newton.State` produce — each one
about the body's COM point, expressed in world-frame coordinates,
per-articulation-local link/DOF indexing — to the tool point, still
expressed in world-frame coordinates.

Every transform variable is named ``coordinate_change_TARGET_from_SOURCE``:
given a point's coordinates in the SOURCE frame, it produces that same
point's coordinates in the TARGET frame. Equivalently, its translation is the
SOURCE frame's origin, expressed in the TARGET frame — so
``coordinate_change_world_from_body``'s translation is directly the body's
position in world coordinates, with no inversion needed. Warp's ``*``
composes transforms as ``(A * B)(p) = A(B(p))`` (right operand applied
first), so this naming makes a chain of transforms cancel visibly, left to
right: ``coordinate_change_world_from_body * coordinate_change_body_from_tool
== coordinate_change_world_from_tool`` — the adjacent ``body``s are the frame
the right transform's output and the left transform's input agree on.
"""

from __future__ import annotations

import numpy as np
import warp as wp

from ....math import velocity_at_point

# Cholesky pivots are clamped above this, scaled by the pivot's own magnitude,
# so float32 cancellation noise on a near-singular matrix can't drive a
# pivot negative (which would make the square root below NaN).
_FLOAT32_EPS = wp.constant(wp.float32(np.finfo(np.float32).eps))


@wp.kernel
def _tool_pose_and_twist_kernel(
    body_q: wp.array[wp.transform],  # (body_count,) coordinate_change_world_from_body per body
    body_qd_world: wp.array[wp.spatial_vector],  # (body_count,) twist about the COM point, in world coords (v_com, w)
    body_com_body: wp.array[wp.vec3],  # (body_count,) COM position, in the body's own local frame
    tool_body: wp.array[wp.int32],  # (robot_count,) -> body index of each robot's tool site
    coordinate_change_body_from_tool: wp.array[wp.transform],  # (robot_count,) tool site's body-local transform
    # outputs
    coordinate_change_world_from_tool: wp.array[wp.transform],  # (robot_count,) world pose of the tool frame
    tool_twist_world: wp.array[
        wp.spatial_vector
    ],  # (robot_count,) twist about the tool point, in world coords (v_tool, w)
):
    robot_idx = wp.tid()
    tool_body_idx = tool_body[robot_idx]
    coordinate_change_world_from_body = body_q[tool_body_idx]
    coordinate_change_world_from_tool[robot_idx] = (
        coordinate_change_world_from_body * coordinate_change_body_from_tool[robot_idx]
    )

    tool_point_world = wp.transform_get_translation(coordinate_change_world_from_tool[robot_idx])
    body_com_world = wp.transform_point(coordinate_change_world_from_body, body_com_body[tool_body_idx])
    com_to_tool_offset_world = tool_point_world - body_com_world
    # Angular velocity is the same everywhere on a rigid body, so only the
    # linear part changes when shifting the twist's reference point.
    body_twist_com_world = body_qd_world[tool_body_idx]
    tool_twist_world[robot_idx] = wp.spatial_vector(
        velocity_at_point(body_twist_com_world, com_to_tool_offset_world), wp.spatial_bottom(body_twist_com_world)
    )


@wp.kernel
def _shift_jacobian_to_tool_kernel(
    jacobian_com_world: wp.array3d[
        float
    ],  # (articulation_count, max_links*6, max_dofs) columns are twists about each link's COM point, in world coords
    body_q: wp.array[wp.transform],  # (body_count,) coordinate_change_world_from_body per body
    body_com_body: wp.array[wp.vec3],  # (body_count,) COM position, in the body's own local frame
    tool_body: wp.array[wp.int32],  # (robot_count,) -> body index of each robot's tool site
    coordinate_change_body_from_tool: wp.array[wp.transform],  # (robot_count,) tool site's body-local transform
    robot_articulation: wp.array[wp.int32],  # (robot_count,) -> articulation index into jacobian_com_world
    robot_link_idx: wp.array[wp.int32],  # (robot_count,) -> row-block index of the tool's link, within its articulation
    # outputs
    jacobian_tool_world: wp.array3d[
        float
    ],  # (robot_count, 6, max_dofs) columns are twists about the tool point, in world coords
):
    robot_idx, dof_idx = wp.tid()
    articulation_idx = robot_articulation[robot_idx]
    link_row_start = robot_link_idx[robot_idx] * 6

    tool_body_idx = tool_body[robot_idx]
    coordinate_change_world_from_body = body_q[tool_body_idx]
    coordinate_change_world_from_tool = coordinate_change_world_from_body * coordinate_change_body_from_tool[robot_idx]
    tool_point_world = wp.transform_get_translation(coordinate_change_world_from_tool)
    body_com_world = wp.transform_point(coordinate_change_world_from_body, body_com_body[tool_body_idx])
    com_to_tool_offset_world = tool_point_world - body_com_world

    jacobian_column_com_world = wp.spatial_vector(
        jacobian_com_world[articulation_idx, link_row_start + 0, dof_idx],
        jacobian_com_world[articulation_idx, link_row_start + 1, dof_idx],
        jacobian_com_world[articulation_idx, link_row_start + 2, dof_idx],
        jacobian_com_world[articulation_idx, link_row_start + 3, dof_idx],
        jacobian_com_world[articulation_idx, link_row_start + 4, dof_idx],
        jacobian_com_world[articulation_idx, link_row_start + 5, dof_idx],
    )
    jacobian_column_tool_world = wp.spatial_vector(
        velocity_at_point(jacobian_column_com_world, com_to_tool_offset_world),
        wp.spatial_bottom(jacobian_column_com_world),
    )
    for row in range(6):
        jacobian_tool_world[robot_idx, row, dof_idx] = jacobian_column_tool_world[row]


# ---------------------------------------------------------------------------
# Operational-space mass matrix: Lambda = (J M^-1 J^T)^-1.
#
# Both M (the joint-space mass matrix) and Lambda^-1 = J M^-1 J^T are
# symmetric positive-definite, so _invert_spd_block_kernel below is used
# twice: once to invert M (block_dim = each robot's controlled-DOF count),
# once to invert Lambda^-1 (block_dim = 6, the fixed task dimension).
# ---------------------------------------------------------------------------


@wp.kernel
def _invert_spd_block_kernel(
    spd_matrix: wp.array3d[float],  # (block_count, max_dim, max_dim) symmetric positive-definite matrix per block
    block_dim: wp.array[wp.int32],  # (block_count,) size of the used top-left submatrix of each block
    # scratch, preallocated by the caller (not valid on entry; written and then read within this kernel)
    cholesky_factor: wp.array3d[
        float
    ],  # (block_count, max_dim, max_dim) lower-triangular L such that spd_matrix = L L^T
    # outputs
    spd_matrix_inv: wp.array3d[
        float
    ],  # (block_count, max_dim, max_dim) inverse of the top-left block_dim x block_dim submatrix; untouched elsewhere
):
    """Explicit inverse of a batch of small SPD matrices, via Cholesky factorization.

    Column c of the inverse solves ``spd_matrix @ x = e_c`` (e_c the c'th
    standard basis vector), found by forward-substituting ``L y = e_c`` and
    then back-substituting ``L^T x = y``. No dense-inverse routine (cofactor
    expansion, Gauss-Jordan) is used — this is the numerically standard way to
    invert a small SPD matrix, and the same recipe
    ``newton/_src/actuators/response_oracle.py`` uses for the same reason.
    """
    block_idx = wp.tid()
    n = block_dim[block_idx]

    # Cholesky factorization: spd_matrix == cholesky_factor @ cholesky_factor^T.
    for col in range(n):
        diagonal_term = spd_matrix[block_idx, col, col]
        for k in range(col):
            diagonal_term -= cholesky_factor[block_idx, col, k] * cholesky_factor[block_idx, col, k]
        diagonal_term = wp.max(diagonal_term, _FLOAT32_EPS * wp.max(wp.abs(spd_matrix[block_idx, col, col]), 1.0))
        diagonal_value = wp.sqrt(diagonal_term)
        cholesky_factor[block_idx, col, col] = diagonal_value
        for row in range(col + 1, n):
            off_diagonal_term = spd_matrix[block_idx, row, col]
            for k in range(col):
                off_diagonal_term -= cholesky_factor[block_idx, row, k] * cholesky_factor[block_idx, col, k]
            cholesky_factor[block_idx, row, col] = off_diagonal_term / diagonal_value

    # Solve spd_matrix @ x = e_c for every column c, writing x into column c of the inverse.
    for c in range(n):
        # Forward substitution: cholesky_factor @ y = e_c.
        for row in range(n):
            right_hand_side = float(0.0)
            if row == c:
                right_hand_side = 1.0
            for k in range(row):
                right_hand_side -= cholesky_factor[block_idx, row, k] * spd_matrix_inv[block_idx, k, c]
            spd_matrix_inv[block_idx, row, c] = right_hand_side / cholesky_factor[block_idx, row, row]
        # Back substitution: cholesky_factor^T @ x = y, overwriting y with x in place.
        for reverse_row in range(n):
            row = n - 1 - reverse_row
            right_hand_side = spd_matrix_inv[block_idx, row, c]
            for k in range(row + 1, n):
                right_hand_side -= cholesky_factor[block_idx, k, row] * spd_matrix_inv[block_idx, k, c]
            spd_matrix_inv[block_idx, row, c] = right_hand_side / cholesky_factor[block_idx, row, row]


@wp.kernel
def _operational_space_mass_matrix_inverse_kernel(
    jacobian_tool_world: wp.array3d[
        float
    ],  # (robot_count, 6, max_dofs) columns are twists about the tool point, in world coords
    mass_matrix_inv: wp.array3d[
        float
    ],  # (robot_count, max_dofs, max_dofs) inverse of the controlled-DOF mass matrix; zero beyond dof_count
    dof_count: wp.array[wp.int32],  # (robot_count,) number of controlled DOFs for each robot
    # outputs
    operational_space_mass_matrix_inv: wp.array3d[
        float
    ],  # (robot_count, 6, 6) = jacobian_tool_world @ mass_matrix_inv @ jacobian_tool_world^T
):
    """The inverse operational-space mass matrix, ``Lambda^-1 = J M^-1 J^T``.

    Still needs a 6x6 inverse (via :func:`_invert_spd_block_kernel`) to become
    the operational-space mass matrix Lambda that maps a desired task-space
    acceleration to the task-space force that would produce it.
    """
    robot_idx, row, col = wp.tid()
    n = dof_count[robot_idx]

    total = float(0.0)
    for a in range(n):
        for b in range(n):
            total += (
                jacobian_tool_world[robot_idx, row, a]
                * mass_matrix_inv[robot_idx, a, b]
                * jacobian_tool_world[robot_idx, col, b]
            )
    operational_space_mass_matrix_inv[robot_idx, row, col] = total
