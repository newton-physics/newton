# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Decoupled joint rows in the sparse SCC path.

With joint decoupling (see :class:`~newton._src.solvers.kamino._src.dynamics.joint_decoupling.DynamicJointDecoupling`),
the friction and effort rows of a DoF act on the bodies with the Jacobian ``s J``, and they couple
to the joint dynamics row of the DoF with ``c r`` and to each other with ``c^2 r``. The sparse
Delassus operator applies these terms in its own products. The sparse SCC kernels that form row
products or matrix entries from the Jacobian blocks instead read the row-scaled Jacobian values
and the coupled-row table of :class:`SparseDecoupledRows`.
"""

from __future__ import annotations

import warp as wp

from ...core.types import vec6f
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


@wp.kernel
def _set_row_body_share(
    # Inputs:
    dofs: wp.array[DecoupledJointDof],
    body_share: wp.array[wp.float32],
    # Outputs:
    row_body_share: wp.array[wp.float32],
):
    tid = wp.tid()
    dof = dofs[tid]
    s = body_share[tid]
    if dof.friction_row >= 0:
        row_body_share[dof.friction_row] = s
    if dof.effort_row >= 0:
        row_body_share[dof.effort_row] = s


@wp.kernel
def _scale_jacobian_rows(
    # Inputs:
    num_nzb: wp.array[wp.int32],
    nzb_start: wp.array[wp.int32],
    nzb_coords: wp.array2d[wp.int32],
    row_start: wp.array[wp.int32],
    row_body_share: wp.array[wp.float32],
    values: wp.array[vec6f],
    # Outputs:
    scaled_values: wp.array[vec6f],
):
    mat, block = wp.tid()
    if block >= num_nzb[mat]:
        return
    idx = nzb_start[mat] + block
    scaled_values[idx] = values[idx] * row_body_share[row_start[mat] + nzb_coords[idx, 0]]


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
    """Row-scaled Jacobian values and coupled-row table of decoupled joint rows for sparse SCC.

    Without a joint decoupling, :attr:`enabled` is ``False``, :meth:`jacobian_values` returns the
    constraint Jacobian values unchanged, and the table holds placeholders that kernels skip.
    """

    def __init__(
        self,
        decoupling: DynamicJointDecoupling | None,
        num_rows: int,
        device: wp.DeviceLike,
        jacobian_values: wp.array | None = None,
    ):
        """Allocates the table and the scaled copy before any graph capture.

        Args:
            decoupling: The joint decoupling of the problem, if any.
            num_rows: Number of constraint rows of all worlds.
            device: Device of the arrays.
            jacobian_values: Block values of the constraint Jacobian, which set the size of the scaled copy.
        """
        self.decoupling = decoupling if decoupling is not None and decoupling.num_dofs > 0 else None
        self.device = device
        rows = num_rows if self.enabled else 1
        self.coupled_row = wp.full((rows, 2), -1, dtype=wp.int32, device=device)
        self.coupled_value = wp.zeros((rows, 2), dtype=wp.float32, device=device)
        self.row_body_share = wp.ones(rows, dtype=wp.float32, device=device) if self.enabled else None
        self._scaled_values: wp.array | None = None
        if self.enabled:
            if jacobian_values is None:
                raise ValueError("Decoupled joint rows require the constraint Jacobian values.")
            self._scaled_values = wp.zeros_like(jacobian_values)

    @property
    def enabled(self) -> bool:
        """Whether the problem has decoupled joint rows."""
        return self.decoupling is not None

    @property
    def inputs(self) -> tuple:
        """The ``(enabled, coupled_row, coupled_value)`` kernel inputs."""
        return (wp.bool(self.enabled), self.coupled_row, self.coupled_value)

    def update(self, problem) -> None:
        """Refreshes the scaled Jacobian values and the coupled-row table for the current build of ``problem``."""
        if not self.enabled:
            return
        decoupling = self.decoupling
        delassus = problem.delassus
        jacobian = delassus.constraint_jacobian
        wp.launch(
            _set_row_body_share,
            dim=decoupling.num_dofs,
            inputs=[decoupling.dofs, decoupling.body_share],
            outputs=[self.row_body_share],
            device=self.device,
        )
        wp.launch(
            _scale_jacobian_rows,
            dim=(jacobian.num_matrices, jacobian.max_of_num_nzb),
            inputs=[
                jacobian.num_nzb,
                jacobian.nzb_start,
                jacobian.nzb_coords,
                jacobian.row_start,
                self.row_body_share,
                jacobian.nzb_values,
            ],
            outputs=[self._scaled_values],
            device=self.device,
        )
        wp.launch(
            _set_coupled_rows,
            dim=decoupling.num_dofs,
            inputs=[
                decoupling.dofs,
                decoupling.body_share,
                delassus._data.joints.inv_m_j,
                problem.data.vio,
                problem.data.P,
            ],
            outputs=[self.coupled_row, self.coupled_value],
            device=self.device,
        )

    def jacobian_values(self, problem) -> wp.array:
        """Jacobian block values with the friction and effort rows of decoupled DoFs scaled by their body share."""
        if not self.enabled:
            return problem.delassus.constraint_jacobian.nzb_values
        return self._scaled_values
