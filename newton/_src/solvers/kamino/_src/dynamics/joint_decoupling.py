# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""
Decouples the Coulomb friction and effort-limit rows of a joint DoF from its joint dynamics row.

A joint DoF with armature, damping or implicit PD has a dynamic constraint row ``q`` with
compliance ``r = inv_m_j``. If the DoF also has a friction row ``f`` and/or an effort-limit
row ``e``, all of these rows share the same body Jacobian ``J``, and with ``d = J M^-1 J^T``
their Delassus block is::

    [ d + r   d        d           ]
    [ d       d        d           ]
    [ d       d        d + inv_m_a ]

When the joint inertia ``1 / r`` is much larger than the body inertia ``1 / d``, this block is
nearly singular, and the joint dynamics impulse nearly cancels the friction and effort impulses.

The dual problem is therefore solved in the variables ``lambda_q' = lambda_q - c (lambda_f +
lambda_e)``, with ``lambda_f`` and ``lambda_e`` unchanged so that their bounds still apply. With
the *body share* ``s = r / (d + r)`` and ``c = s - 1 = -d / (d + r)``, this eliminates the
coupling between ``q`` and the other two rows. The friction and effort rows then act on the
bodies with the scaled Jacobian ``s J`` and on the joint dynamics with the compliance ``c^2 r``,
which sum to ``d r / (d + r)``, the inverse of the combined body and joint inertia of the DoF.

``s`` is always computed directly, since ``1 + c`` cancels when ``c`` tends to ``-1``, whereas
``c = s - 1`` is exact in floating point.
"""

from __future__ import annotations

import numpy as np
import warp as wp

from ..core.model import ModelKamino

###
# Module interface
###

__all__ = [
    "DecoupledJointDof",
    "DynamicJointDecoupling",
    "compute_body_share",
]


###
# Module configs
###

wp.set_module_options({"enable_backward": False})


###
# Types
###


@wp.struct
class DecoupledJointDof:
    """A joint DoF whose friction and/or effort row is decoupled from its joint dynamics row."""

    wid: wp.int32
    """Index of the world of the DoF."""

    dynamic_cts: wp.int32
    """Index of the DoF in the joint dynamic constraint arrays (e.g. ``JointsData.inv_m_j``)."""

    effort_cts: wp.int32
    """Index of the DoF in the joint effort constraint arrays (e.g. ``JointsData.inv_m_a``), or ``-1``."""

    dynamic_row: wp.int32
    """Index of the joint dynamics row in the model-wide constraint vector."""

    friction_row: wp.int32
    """Index of the Coulomb friction row in the model-wide constraint vector, or ``-1``."""

    effort_row: wp.int32
    """Index of the effort-limit row in the model-wide constraint vector, or ``-1``."""


###
# Functions
###


@wp.func
def compute_body_share(d: wp.float32, inv_m_j: wp.float32) -> wp.float32:
    """Body share ``s = r / (d + r)`` of a decoupled DoF.

    Args:
        d: Body-space Delassus diagonal ``J M^-1 J^T`` of the DoF.
        inv_m_j: Compliance ``r`` of the joint dynamics row of the DoF.
    """
    return inv_m_j / (d + inv_m_j)


###
# Kernels
###


@wp.kernel
def _decouple_free_velocity(
    # Inputs:
    dofs: wp.array[DecoupledJointDof],
    body_share: wp.array[wp.float32],
    # Outputs:
    v_f: wp.array[wp.float32],
):
    """Apply the change of variables to the free velocity of the friction and effort rows."""
    tid = wp.tid()
    dof = dofs[tid]
    shift = (body_share[tid] - 1.0) * v_f[dof.dynamic_row]
    if dof.friction_row >= 0:
        v_f[dof.friction_row] += shift
    if dof.effort_row >= 0:
        v_f[dof.effort_row] += shift


@wp.func
def _decouple_warmstart_row(
    row: wp.int32,
    row_dyn: wp.int32,
    c: wp.float32,
    P: wp.array[wp.float32],
    x: wp.array[wp.float32],
    y: wp.array[wp.float32],
    z: wp.array[wp.float32],
):
    """Apply the change of variables to the preconditioned iterates of one friction or effort row, if any."""
    if row < 0:
        return
    ratio = c * P[row] / P[row_dyn]
    x[row_dyn] -= ratio * x[row]
    y[row_dyn] -= ratio * y[row]
    z[row] += ratio * z[row_dyn]


@wp.kernel
def _decouple_warmstart(
    # Inputs:
    dofs: wp.array[DecoupledJointDof],
    body_share: wp.array[wp.float32],
    P: wp.array[wp.float32],
    # Outputs:
    x: wp.array[wp.float32],
    y: wp.array[wp.float32],
    z: wp.array[wp.float32],
):
    """Apply the change of variables to the preconditioned warm-start iterates of each DoF."""
    tid = wp.tid()
    dof = dofs[tid]
    c = body_share[tid] - 1.0
    _decouple_warmstart_row(dof.friction_row, dof.dynamic_row, c, P, x, y, z)
    _decouple_warmstart_row(dof.effort_row, dof.dynamic_row, c, P, x, y, z)


@wp.kernel
def _rebase_solution(
    # Inputs:
    dofs: wp.array[DecoupledJointDof],
    body_share: wp.array[wp.float32],
    # Inputs and outputs:
    solution_body_share: wp.array[wp.float32],
    lambdas: wp.array[wp.float32],
    v_plus: wp.array[wp.float32],
):
    """Express a solution in the variables of the current body share instead of those it was computed with.

    The original impulses ``lambda_q = lambda_q' + c lambda_f`` and velocities ``v_f = v_f' - c v_q``
    are unchanged, so ``lambda_q'`` changes by ``(c_old - c_new) lambda_f`` and ``v_f'`` by
    ``(c_new - c_old) v_q``, where ``lambda_f`` sums the friction and effort impulses.
    """
    tid = wp.tid()
    dof = dofs[tid]
    s_new = body_share[tid]
    dc = solution_body_share[tid] - s_new  # c_old - c_new
    solution_body_share[tid] = s_new
    lambda_sum = wp.float32(0.0)
    v_shift = -dc * v_plus[dof.dynamic_row]
    if dof.friction_row >= 0:
        lambda_sum += lambdas[dof.friction_row]
        v_plus[dof.friction_row] += v_shift
    if dof.effort_row >= 0:
        lambda_sum += lambdas[dof.effort_row]
        v_plus[dof.effort_row] += v_shift
    lambdas[dof.dynamic_row] += dc * lambda_sum


@wp.kernel
def _recouple_impulses(
    # Inputs:
    dofs: wp.array[DecoupledJointDof],
    body_share: wp.array[wp.float32],
    lambdas: wp.array[wp.float32],
    # Outputs:
    body_lambdas: wp.array[wp.float32],
    joint_lambdas: wp.array[wp.float32],
):
    """Write the body-space and the original impulses of the rows of each DoF."""
    tid = wp.tid()
    dof = dofs[tid]
    s = body_share[tid]
    lambda_sum = wp.float32(0.0)
    if dof.friction_row >= 0:
        lambda_f = lambdas[dof.friction_row]
        body_lambdas[dof.friction_row] = s * lambda_f
        lambda_sum += lambda_f
    if dof.effort_row >= 0:
        lambda_e = lambdas[dof.effort_row]
        body_lambdas[dof.effort_row] = s * lambda_e
        lambda_sum += lambda_e
    joint_lambdas[dof.dynamic_row] = lambdas[dof.dynamic_row] + (s - 1.0) * lambda_sum


@wp.kernel
def _compute_body_share_dense(
    # Inputs:
    dofs: wp.array[DecoupledJointDof],
    inv_m_j: wp.array[wp.float32],
    dim: wp.array[wp.int32],
    mio: wp.array[wp.int32],
    vio: wp.array[wp.int32],
    D: wp.array2d[wp.float32],
    # Outputs:
    body_share: wp.array[wp.float32],
):
    """Compute the body share of each DoF from the dense body-space Delassus matrix."""
    tid = wp.tid()
    dof = dofs[tid]
    ncts = dim[dof.wid]
    vio_w = vio[dof.wid]
    # The off-diagonal entry between the joint dynamics row and either other row is not
    # regularized, so it is the body-space term ``d`` of the DoF
    row = wp.where(dof.friction_row >= 0, dof.friction_row, dof.effort_row)
    d = D[dof.wid, mio[dof.wid] + ncts * (dof.dynamic_row - vio_w) + row - vio_w]
    body_share[tid] = compute_body_share(d, inv_m_j[dof.dynamic_cts])


@wp.func
def _scale_dense_row(
    row: wp.int32,
    k: wp.int32,
    s: wp.float32,
    ncts: wp.int32,
    m: wp.int32,
    vio_w: wp.int32,
    columns: wp.bool,
    D: wp.array[wp.float32],
):
    """Scale entry ``k`` of one friction or effort row of a dense Delassus matrix, or of its column, if any."""
    if row < 0:
        return
    local = row - vio_w
    if columns:
        D[m + ncts * k + local] *= s
    else:
        D[m + ncts * local + k] *= s


@wp.kernel
def _scale_rows_dense(
    # Inputs:
    dofs: wp.array[DecoupledJointDof],
    body_share: wp.array[wp.float32],
    dim: wp.array[wp.int32],
    mio: wp.array[wp.int32],
    vio: wp.array[wp.int32],
    columns: wp.bool,
    # Outputs:
    D: wp.array2d[wp.float32],
):
    """Scale the friction and effort rows, or columns, of each DoF in a dense Delassus matrix by its body share."""
    # Columns are launched with the DoF index varying fastest, so that the threads of a warp write
    # neighboring entries of a matrix row instead of entries strided by a row length
    i, j = wp.tid()
    tid, k = i, j
    if columns:
        tid, k = j, i
    dof = dofs[tid]
    ncts = dim[dof.wid]
    if k >= ncts:
        return
    m = mio[dof.wid]
    vio_w = vio[dof.wid]
    s = body_share[tid]
    _scale_dense_row(dof.friction_row, k, s, ncts, m, vio_w, columns, D[dof.wid])
    _scale_dense_row(dof.effort_row, k, s, ncts, m, vio_w, columns, D[dof.wid])


@wp.kernel
def _add_joint_terms_dense(
    # Inputs:
    dofs: wp.array[DecoupledJointDof],
    body_share: wp.array[wp.float32],
    inv_m_j: wp.array[wp.float32],
    inv_m_a: wp.array[wp.float32],
    dim: wp.array[wp.int32],
    mio: wp.array[wp.int32],
    vio: wp.array[wp.int32],
    # Outputs:
    D: wp.array2d[wp.float32],
):
    """Add the joint-space terms of the friction and effort rows to the row-scaled dense Delassus matrix."""
    tid = wp.tid()
    dof = dofs[tid]
    ncts = dim[dof.wid]
    m = mio[dof.wid]
    vio_w = vio[dof.wid]
    row_q = dof.dynamic_row - vio_w
    row_f = dof.friction_row - vio_w
    row_e = dof.effort_row - vio_w
    s = body_share[tid]
    c = s - 1.0
    compliance = c * c * inv_m_j[dof.dynamic_cts]
    # The change of variables decouples the rows, so set the cancelling terms to exact zeros
    if dof.friction_row >= 0:
        D[dof.wid, m + ncts * row_f + row_f] += compliance
        D[dof.wid, m + ncts * row_q + row_f] = 0.0
        D[dof.wid, m + ncts * row_f + row_q] = 0.0
    if dof.effort_row >= 0:
        # Undo the row scaling of the effort compliance, which the change of variables leaves unchanged
        D[dof.wid, m + ncts * row_e + row_e] += compliance + (1.0 - s * s) * inv_m_a[dof.effort_cts]
        D[dof.wid, m + ncts * row_q + row_e] = 0.0
        D[dof.wid, m + ncts * row_e + row_q] = 0.0
        if dof.friction_row >= 0:
            D[dof.wid, m + ncts * row_f + row_e] += compliance
            D[dof.wid, m + ncts * row_e + row_f] += compliance


###
# Interfaces
###


class DynamicJointDecoupling:
    """
    Decouples joint friction and effort-limit rows from the joint dynamics rows of the same DoFs.

    Holds the table of the decoupled DoFs and their body share ``s``, and applies the change
    of variables described in the module docstring to the vectors of the dual problem. The
    Delassus operators apply it to their matrix or matrix-free products.

    The dual problem, and thus its solution, is in the decoupled variables. Use
    :meth:`recouple_impulses` to obtain the impulses that act on the bodies, and the impulses
    of the original rows.
    """

    def __init__(self, model: ModelKamino | None = None, device: wp.DeviceLike = None):
        """
        Initializes the decoupling.

        Args:
            model: The model to decouple the joint DoFs of. If provided, :meth:`finalize` is called.
            device: The device to allocate on. Defaults to the device of the model.
        """
        # Declare the device cache
        self._device: wp.DeviceLike = None

        self.dofs: wp.array[DecoupledJointDof] | None = None
        """The decoupled DoFs, or ``None`` if the model has none."""

        self.body_share: wp.array[wp.float32] | None = None
        """Body share ``s`` of each decoupled DoF, updated whenever the dual problem is built."""

        self.solution_body_share: wp.array[wp.float32] | None = None
        """Body share of each decoupled DoF with which the solver's stored solution was computed,
        updated by :meth:`rebase_solution`."""

        # If a model is provided, finalize the allocations
        if model is not None:
            self.finalize(model, device=device)

    def finalize(self, model: ModelKamino, device: wp.DeviceLike = None):
        """
        Finds all joint DoFs with a joint dynamics row and a friction or effort row, and
        allocates the decoupling data. Leaves :attr:`dofs` as ``None`` if there are no such DoFs.

        Args:
            model: The model to decouple the joint DoFs of.
            device: The device to allocate on. Defaults to the device of the model.
        """
        # Ensure the model is valid
        if not isinstance(model, ModelKamino):
            raise TypeError("Expected 'model' to be of type ModelKamino.")

        self._device = device if device is not None else model.device
        self.dofs = None
        self.body_share = None
        self.solution_body_share = None

        table = self._build_dof_table(model)
        if table is None:
            return

        with wp.ScopedDevice(self._device):
            self.dofs = wp.array(table, dtype=DecoupledJointDof)
            self.body_share = wp.zeros(table.size, dtype=wp.float32)
            self.solution_body_share = wp.zeros(table.size, dtype=wp.float32)

    @staticmethod
    def _build_dof_table(model: ModelKamino) -> np.ndarray | None:
        """Returns the :class:`DecoupledJointDof` table of the model as a NumPy array, or ``None`` if it is empty."""
        size = model.size
        joints = model.joints
        if size.sum_of_num_dynamic_joint_cts == 0:
            return None
        if size.sum_of_num_friction_joint_cts + size.sum_of_num_effort_joint_cts == 0:
            return None

        num_joints = size.sum_of_num_joints
        joint_ids = np.arange(num_joints, dtype=np.int64)

        def joint_and_axis(offset: wp.array, axis: wp.array) -> tuple[np.ndarray, np.ndarray]:
            # Joint index of each row of a group, and a key that sorts the rows by joint and axis
            offset_np = offset.numpy()
            jid = np.repeat(joint_ids, np.diff(offset_np))
            return jid, jid * 8 + axis.numpy()

        dyn_offset = joints.dynamic_cts_offset.numpy()
        dyn_jid, dyn_key = joint_and_axis(joints.dynamic_cts_offset, joints.dynamic_cts_axis)

        def rows_of_same_dof(offset: wp.array, axis: wp.array, offset_total_cts: wp.array):
            # Index into the group arrays and constraint-vector row of each dynamic row's DoF, or -1
            _, key = joint_and_axis(offset, axis)
            if key.size == 0:
                none = np.full(dyn_key.shape, -1, dtype=np.int32)
                return none, none
            pos = np.minimum(np.searchsorted(key, dyn_key), key.size - 1)
            found = key[pos] == dyn_key
            row = offset_total_cts.numpy()[dyn_jid] + pos - offset.numpy()[dyn_jid]
            return np.where(found, pos, -1).astype(np.int32), np.where(found, row, -1).astype(np.int32)

        _, friction_row = rows_of_same_dof(
            joints.friction_cts_offset, joints.friction_cts_axis, joints.friction_cts_offset_total_cts
        )
        effort_cts, effort_row = rows_of_same_dof(
            joints.effort_cts_offset, joints.effort_cts_axis, joints.effort_cts_offset_total_cts
        )
        decoupled = (friction_row >= 0) | (effort_row >= 0)
        if not decoupled.any():
            return None

        dynamic_cts = np.flatnonzero(decoupled)
        dyn_jid = dyn_jid[dynamic_cts]
        table = np.zeros(dynamic_cts.size, dtype=DecoupledJointDof.numpy_dtype())
        table["wid"] = joints.wid.numpy()[dyn_jid]
        table["dynamic_cts"] = dynamic_cts
        table["effort_cts"] = effort_cts[dynamic_cts]
        table["dynamic_row"] = joints.dynamic_cts_offset_total_cts.numpy()[dyn_jid] + dynamic_cts - dyn_offset[dyn_jid]
        table["friction_row"] = friction_row[dynamic_cts]
        table["effort_row"] = effort_row[dynamic_cts]
        return table

    @property
    def num_dofs(self) -> int:
        """Number of decoupled DoFs."""
        return 0 if self.dofs is None else self.dofs.shape[0]

    @property
    def device(self) -> wp.DeviceLike:
        """Device of the decoupling arrays."""
        return self._device

    def compute_body_share_dense(
        self, inv_m_j: wp.array, dim: wp.array, mio: wp.array, vio: wp.array, D: wp.array2d[wp.float32]
    ):
        """
        Updates the body share from a dense body-space Delassus matrix.

        Args:
            inv_m_j: Compliance of the joint dynamics rows.
            dim, mio, vio: Per-world dimension, matrix offset and vector offset of the Delassus matrix.
            D: The dense Delassus matrix, before any change of variables.
        """
        wp.launch(
            _compute_body_share_dense,
            dim=self.num_dofs,
            inputs=[self.dofs, inv_m_j, dim, mio, vio, D],
            outputs=[self.body_share],
            device=self.device,
        )

    def scale_rows_dense(self, dim: wp.array, mio: wp.array, vio: wp.array, max_rows: int, D: wp.array2d[wp.float32]):
        """
        Scales the friction and effort rows and columns of a dense Delassus matrix by the body share.

        Only the rows and columns of the decoupled DoFs are visited, so the cost is proportional
        to their number rather than to the size of the matrix.

        Args:
            dim, mio, vio: Per-world dimension, matrix offset and vector offset of the Delassus matrix.
            max_rows: Maximum number of constraint rows of any world.
            D: The dense Delassus matrix, modified in place.
        """
        # Scale the rows, then the columns, so that the entries of two decoupled rows get both shares
        for columns in (False, True):
            wp.launch(
                _scale_rows_dense,
                dim=(max_rows, self.num_dofs) if columns else (self.num_dofs, max_rows),
                inputs=[self.dofs, self.body_share, dim, mio, vio, columns],
                outputs=[D],
                device=self.device,
            )

    def add_joint_terms_dense(
        self,
        inv_m_j: wp.array,
        inv_m_a: wp.array,
        dim: wp.array,
        mio: wp.array,
        vio: wp.array,
        D: wp.array2d[wp.float32],
    ):
        """
        Completes the change of variables of a dense Delassus matrix that is already scaled by
        :meth:`scale_rows_dense`.

        Args:
            inv_m_j: Compliance of the joint dynamics rows.
            inv_m_a: Compliance of the effort rows.
            dim, mio, vio: Per-world dimension, matrix offset and vector offset of the Delassus matrix.
            D: The dense, row-scaled Delassus matrix, modified in place.
        """
        wp.launch(
            _add_joint_terms_dense,
            dim=self.num_dofs,
            inputs=[self.dofs, self.body_share, inv_m_j, inv_m_a, dim, mio, vio],
            outputs=[D],
            device=self.device,
        )

    def decouple_free_velocity(self, v_f: wp.array[wp.float32]):
        """
        Applies the change of variables to a free-velocity vector.

        Args:
            v_f: Free velocity, modified in place.
        """
        wp.launch(
            _decouple_free_velocity,
            dim=self.num_dofs,
            inputs=[self.dofs, self.body_share],
            outputs=[v_f],
            device=self.device,
        )

    def decouple_warmstart(
        self, P: wp.array[wp.float32], x: wp.array[wp.float32], y: wp.array[wp.float32], z: wp.array[wp.float32]
    ):
        """
        Applies the change of variables to preconditioned warm-start iterates of the original rows.

        Args:
            P: Diagonal preconditioner of the dual problem.
            x: Preconditioned primal iterate, modified in place.
            y: Preconditioned slack iterate, modified in place.
            z: Preconditioned dual (velocity) iterate, modified in place.
        """
        wp.launch(
            _decouple_warmstart,
            dim=self.num_dofs,
            inputs=[self.dofs, self.body_share, P],
            outputs=[x, y, z],
            device=self.device,
        )

    def rebase_solution(self, lambdas: wp.array[wp.float32], v_plus: wp.array[wp.float32]):
        """
        Expresses a stored solution in the variables of the current body share, e.g. to warm-start from it.

        The body share changes with the Jacobian, the body inertias and the joint compliance, and the
        solution is in the variables of the body share it was computed with, which
        :attr:`solution_body_share` records, see :meth:`record_solution_body_share`. Rebasing records
        the current body share, so a second call has no effect.

        Args:
            lambdas: Unpreconditioned constraint impulses of the solution, modified in place.
            v_plus: Unpreconditioned post-event constraint velocities of the solution, modified in place.
        """
        wp.launch(
            _rebase_solution,
            dim=self.num_dofs,
            inputs=[self.dofs, self.body_share],
            outputs=[self.solution_body_share, lambdas, v_plus],
            device=self.device,
        )

    def record_solution_body_share(self):
        """Records the current body share as the one of a solution that was just computed in its variables."""
        wp.copy(self.solution_body_share, self.body_share)

    def recouple_impulses(
        self,
        lambdas: wp.array[wp.float32],
        body_lambdas: wp.array[wp.float32],
        joint_lambdas: wp.array[wp.float32],
    ):
        """
        Computes the impulses that act on the bodies and the impulses of the original rows.

        Body wrenches computed from ``body_lambdas`` with the constraint Jacobians avoid summing
        the nearly cancelling impulses of the original rows.

        Args:
            lambdas: Constraint impulses in the decoupled variables.
            body_lambdas: Output impulses that act on the bodies through the constraint Jacobians:
                the friction and effort rows are scaled by ``s``, and all other rows are copied.
            joint_lambdas: Output impulses of the original rows: the joint dynamics rows are
                ``lambda_q = lambda_q' + c (lambda_f + lambda_e)``, and all other rows are copied.
        """
        wp.copy(body_lambdas, lambdas)
        wp.copy(joint_lambdas, lambdas)
        wp.launch(
            _recouple_impulses,
            dim=self.num_dofs,
            inputs=[self.dofs, self.body_share, lambdas],
            outputs=[body_lambdas, joint_lambdas],
            device=self.device,
        )
