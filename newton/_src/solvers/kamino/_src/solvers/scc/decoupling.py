# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Decoupled joint rows in the sparse SCC path.

With joint decoupling (see :class:`~newton._src.solvers.kamino._src.dynamics.joint_decoupling.DynamicJointDecoupling`),
the friction and effort rows of a DoF act on the bodies with the Jacobian ``s J``, and they couple
to the joint dynamics row of the DoF with ``c r`` and to each other with ``c^2 r``. The sparse
Delassus operator applies these terms in its own products. The sparse SCC kernels that form row
products or matrix entries from the Jacobian blocks instead scale the Jacobian rows with the row
scale of the operator, which includes ``s``, and read the coupled-row table of
:class:`SparseDecoupledRows`.
"""

from __future__ import annotations

import warp as wp

from ...dynamics.joint_decoupling import DecoupledJointDof, DynamicJointDecoupling

wp.set_module_options({"enable_backward": False})

__all__ = ["SparseDecoupledRows", "coupled_rows_entry", "coupled_rows_product"]


@wp.func
def coupled_rows_product(
    vec_idx: wp.int32,
    vio: wp.int32,
    coupled_row: wp.array2d[wp.int32],
    coupled_value: wp.array2d[wp.float32],
    x: wp.array[wp.float32],
) -> wp.float32:
    """Product of the joint-space couplings of the global row ``vec_idx`` with ``x``.

    Args:
        vec_idx: Global index of the row.
        vio: Vector offset of the row's world.
        coupled_row: World-local indices of the up to two coupled rows of each row, or ``-1``.
        coupled_value: Preconditioned coupling ``P_i C_ij P_j`` to each coupled row.
        x: Vector of preconditioned impulses.
    """
    value = wp.float32(0.0)
    for k in range(2):
        coupled = coupled_row[vec_idx, k]
        if coupled >= 0:
            value += coupled_value[vec_idx, k] * x[vio + coupled]
    return value


@wp.func
def coupled_rows_entry(
    vec_idx: wp.int32,
    local_col: wp.int32,
    coupled_row: wp.array2d[wp.int32],
    coupled_value: wp.array2d[wp.float32],
) -> wp.float32:
    """Joint-space coupling between the global row ``vec_idx`` and the world-local row ``local_col``."""
    value = wp.float32(0.0)
    for k in range(2):
        if coupled_row[vec_idx, k] == local_col:
            value += coupled_value[vec_idx, k]
    return value


@wp.func
def _set_coupled_row(
    row: wp.int32,
    slot: wp.int32,
    other: wp.int32,
    value: wp.float32,
    vio: wp.int32,
    P: wp.array[wp.float32],
    coupled_row: wp.array2d[wp.int32],
    coupled_value: wp.array2d[wp.float32],
):
    coupled_row[row, slot] = other - vio
    coupled_value[row, slot] = P[row] * value * P[other]


@wp.kernel
def _set_coupled_rows(
    # Inputs:
    dofs: wp.array[DecoupledJointDof],
    body_share: wp.array[wp.float32],
    inv_m_j: wp.array[wp.float32],
    vio: wp.array[wp.int32],
    P: wp.array[wp.float32],
    # Outputs:
    coupled_row: wp.array2d[wp.int32],
    coupled_value: wp.array2d[wp.float32],
):
    """Write the joint-space couplings ``c r`` and ``c^2 r`` of the rows of each decoupled DoF.

    The joint dynamics row uses slot 0 for the friction row and slot 1 for the effort row, and
    the friction and effort rows use slot 0 for the joint dynamics row and slot 1 for each other.
    """
    tid = wp.tid()
    dof = dofs[tid]
    offset = vio[dof.wid]
    c = body_share[tid] - 1.0
    coupling = c * inv_m_j[dof.dynamic_cts]
    q = dof.dynamic_row
    f = dof.friction_row
    e = dof.effort_row
    if f >= 0:
        _set_coupled_row(q, 0, f, coupling, offset, P, coupled_row, coupled_value)
        _set_coupled_row(f, 0, q, coupling, offset, P, coupled_row, coupled_value)
    if e >= 0:
        _set_coupled_row(q, 1, e, coupling, offset, P, coupled_row, coupled_value)
        _set_coupled_row(e, 0, q, coupling, offset, P, coupled_row, coupled_value)
        if f >= 0:
            _set_coupled_row(f, 1, e, c * coupling, offset, P, coupled_row, coupled_value)
            _set_coupled_row(e, 1, f, c * coupling, offset, P, coupled_row, coupled_value)


class SparseDecoupledRows:
    """Row scale and coupled-row table of decoupled joint rows for sparse SCC.

    Without a joint decoupling, :attr:`enabled` is ``False``, :meth:`row_scale` returns the
    preconditioner, and the table holds placeholders that kernels skip.
    """

    def __init__(self, decoupling: DynamicJointDecoupling | None, num_rows: int, device: wp.DeviceLike):
        """Allocates the table before any graph capture.

        Args:
            decoupling: The joint decoupling of the problem, if any.
            num_rows: Number of constraint rows of all worlds.
            device: Device of the arrays.
        """
        self.decoupling = decoupling if decoupling is not None and decoupling.num_dofs > 0 else None
        self.device = device
        rows = num_rows if self.enabled else 1
        self.coupled_row = wp.full((rows, 2), -1, dtype=wp.int32, device=device)
        self.coupled_value = wp.zeros((rows, 2), dtype=wp.float32, device=device)

    @property
    def enabled(self) -> bool:
        """Whether the problem has decoupled joint rows."""
        return self.decoupling is not None

    @property
    def inputs(self) -> tuple:
        """The ``(enabled, coupled_row, coupled_value)`` kernel inputs."""
        return (wp.bool(self.enabled), self.coupled_row, self.coupled_value)

    def update(self, problem) -> None:
        """Refreshes the coupled-row table for the current build of ``problem``."""
        if not self.enabled:
            return
        decoupling = self.decoupling
        wp.launch(
            _set_coupled_rows,
            dim=decoupling.num_dofs,
            inputs=[
                decoupling.dofs,
                decoupling.body_share,
                problem.delassus._data.joints.inv_m_j,
                problem.data.vio,
                problem.data.P,
            ],
            outputs=[self.coupled_row, self.coupled_value],
            device=self.device,
        )

    def row_scale(self, problem) -> wp.array:
        """Scale of the Jacobian rows: the preconditioner, with the friction and effort rows of
        decoupled DoFs also scaled by their body share.

        The sparse Delassus operator refreshes it together with the mass-weighted Jacobian blocks
        that the same kernels read, so it is current whenever they are.
        """
        if not self.enabled:
            return problem.data.P
        return problem.delassus._jacobian_row_scale
