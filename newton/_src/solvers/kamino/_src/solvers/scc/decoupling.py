# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Decoupled joint rows in the sparse SCC path.

With joint decoupling (see :class:`~newton._src.solvers.kamino._src.dynamics.joint_decoupling.DynamicJointDecoupling`),
the friction and effort rows of a DoF act on the bodies with the Jacobian ``s J``, and they couple
to the joint dynamics row of the DoF with ``c r`` and to each other with ``c^2 r``. The sparse
Delassus operator applies these terms in its own products. The sparse SCC kernels that form row
products or matrix entries from the Jacobian blocks instead scale the Jacobian rows with the row
scale of the operator, which includes ``s``, and read the operator's coupled-row table (see
:meth:`~newton._src.solvers.kamino._src.dynamics.delassus.DelassusOperator.update_decoupled_coupled_rows`).
"""

from __future__ import annotations

import warp as wp

wp.set_module_options({"enable_backward": False})

__all__ = ["SparseDecoupledRows", "coupled_rows_entry", "coupled_rows_product"]


@wp.func
def coupled_rows_product(
    vec_idx: wp.int32,
    vio: wp.int32,
    coupled_row: wp.array2d[wp.int32],
    coupled_coef: wp.array2d[wp.float32],
    row_scale: wp.array[wp.float32],
    x: wp.array[wp.float32],
) -> wp.float32:
    """Product of the joint-space couplings of the global row ``vec_idx`` with ``x``.

    Args:
        vec_idx: Global index of the row.
        vio: Vector offset of the row's world.
        coupled_row: World-local indices of the up to two coupled rows of each row, or ``-1``.
        coupled_coef: Coupling to each coupled row, relative to its entry of ``P S x``.
        row_scale: Row scale ``P S`` of the Jacobian.
        x: Vector of preconditioned impulses.
    """
    value = wp.float32(0.0)
    for k in range(2):
        coupled = coupled_row[vec_idx, k]
        if coupled >= 0:
            value += coupled_coef[vec_idx, k] * row_scale[vio + coupled] * x[vio + coupled]
    return value


@wp.func
def coupled_rows_entry(
    vec_idx: wp.int32,
    vio: wp.int32,
    local_col: wp.int32,
    coupled_row: wp.array2d[wp.int32],
    coupled_coef: wp.array2d[wp.float32],
    row_scale: wp.array[wp.float32],
) -> wp.float32:
    """Joint-space coupling between the global row ``vec_idx`` and the world-local row ``local_col``.

    Args:
        vec_idx: Global index of the row.
        vio: Vector offset of the row's world.
        local_col: World-local index of the column.
        coupled_row: World-local indices of the up to two coupled rows of each row, or ``-1``.
        coupled_coef: Coupling to each coupled row, relative to its entry of ``P S x``.
        row_scale: Row scale ``P S`` of the Jacobian.
    """
    value = wp.float32(0.0)
    for k in range(2):
        if coupled_row[vec_idx, k] == local_col:
            value += coupled_coef[vec_idx, k]
    return value * row_scale[vio + local_col]


class SparseDecoupledRows:
    """Kernel inputs for the decoupled joint rows of sparse SCC.

    The table and the row scale belong to the sparse Delassus operator, which refreshes them
    together with the mass-weighted Jacobian blocks that the same kernels read. Without a joint
    decoupling, the inputs are placeholders that kernels skip, and the row scale is the
    preconditioner.
    """

    def __init__(self, device: wp.DeviceLike):
        """Allocates the placeholder table before any graph capture.

        Args:
            device: Device of the arrays.
        """
        self.coupled_row = wp.full((1, 2), -1, dtype=wp.int32, device=device)
        self.coupled_coef = wp.zeros((1, 2), dtype=wp.float32, device=device)

    @staticmethod
    def enabled(problem) -> bool:
        """Whether ``problem`` has decoupled joint rows."""
        return problem.joint_decoupling is not None

    def inputs(self, problem) -> tuple:
        """The ``(enabled, coupled_row, coupled_coef, row_scale)`` kernel inputs of ``problem``."""
        if not self.enabled(problem):
            return self.disabled_inputs(problem.data.P)
        delassus = problem.delassus
        return (
            wp.bool(True),
            delassus._decoupled_coupled_row,
            delassus._decoupled_coupled_coef,
            self.row_scale(problem),
        )

    def disabled_inputs(self, row_scale: wp.array) -> tuple:
        """The kernel inputs without decoupled joint rows, with the row scale ``row_scale``."""
        return (wp.bool(False), self.coupled_row, self.coupled_coef, row_scale)

    def row_scale(self, problem) -> wp.array:
        """Scale of the Jacobian rows: the preconditioner, with the friction and effort rows of
        decoupled DoFs also scaled by their body share.
        """
        if not self.enabled(problem):
            return problem.data.P
        return problem.delassus._jacobian_row_scale
