# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Sparse DVI solve path for Kamino dual systems."""

from __future__ import annotations

import warp as wp

from ...core.data import DataKamino
from ...core.model import ModelKamino
from ...dynamics.delassus import BlockSparseMatrixFreeDelassusOperator
from ...dynamics.dual import DualProblem
from ...geometry.contacts import ContactsKamino
from ...kinematics.jacobians import SparseSystemJacobians
from ...kinematics.limits import LimitsKamino
from . import sparse_kernels
from .kernels import (
    _initialize_dvi_status,
    _scatter_bilateral_solution,
    _set_dvi_bilateral_active_dim,
    _set_dvi_direct_status_iterations,
)
from .sparse_kernels import (
    _build_sparse_bilateral_block,
    _build_sparse_bilateral_rhs,
    _color_mapped_dvi_inequalities,
    _compute_dvi_sparse_solution_vectors,
    _map_active_contacts,
    _map_active_limits,
    _set_dvi_sparse_status_iterations,
    _set_sparse_bilateral_diagonal,
    _solve_dvi_sparse_inequalities_pgs,
    _sparse_delassus_gemv_rows,
    _zero_bilateral_lambdas,
)

wp.set_module_options({"enable_backward": False})

int32 = wp.int32


_SPARSE_DELASSUS_ROWS_JOINTS = 0
_SPARSE_DELASSUS_ROWS_UNILATERAL = 1


class SparseDVIPath:
    """Own workspace and operations for the sparse Kamino DVI solve path."""

    def __init__(
        self,
        device: wp.DeviceLike,
        size,
        data,
        model: ModelKamino,
        model_data: DataKamino | None,
        limits: LimitsKamino | None,
        contacts: ContactsKamino | None,
        jacobians: SparseSystemJacobians | None,
        bilateral_solver,
        max_iterations: int,
        max_block_iterations: int,
        max_contact_iterations: int,
        has_contact_block_preconditioner: bool,
        has_unilateral_constraints: bool,
        all_worlds_mask: wp.array[wp.bool],
        should_solve_bilateral_after_block,
    ):
        """Initialize the sparse-path workspace references."""
        self.device = device
        self.size = size
        self.data = data
        self.model = model
        self.model_data = model_data
        self.limits = limits
        self.contacts = contacts
        self.jacobians = jacobians
        self.body_space = wp.empty(shape=size.sum_of_num_body_dofs, dtype=wp.float32, device=device)
        self.bilateral_solver = bilateral_solver
        self.max_iterations = max_iterations
        self.max_block_iterations = max_block_iterations
        self.max_contact_iterations = max_contact_iterations
        self.has_contact_block_preconditioner = has_contact_block_preconditioner
        self.has_unilateral_constraints = has_unilateral_constraints
        self.all_worlds_mask = all_worlds_mask
        self.should_solve_bilateral_after_block = should_solve_bilateral_after_block
        self.bilateral_nzb_pairs: (
            tuple[
                wp.array[wp.int32],
                wp.array[wp.int32],
                wp.array[wp.int32],
                wp.array[wp.int32],
                wp.array[wp.int32],
                wp.array[wp.int32],
            ]
            | None
        ) = None

    def prepare(self, problem: DualProblem) -> None:
        """Precompute host-derived sparse topology before the first solve."""
        _get_sparse_delassus(problem)
        if self.model_data is None or self.jacobians is None:
            raise RuntimeError("Sparse DVI requires model data and sparse Jacobians.")
        if self.bilateral_solver is not None and self.data.bilateral_operator is not None:
            _build_sparse_bilateral_pairs(self, problem)

    def solve(self, problem: DualProblem) -> None:
        """Solve a sparse Kamino DVI problem without materializing dense Delassus."""
        if self.has_contact_block_preconditioner and self.size.max_of_max_contacts > 0:
            _compute_sparse_contact_block_inverse(self, problem)

        if self.bilateral_solver is not None and self.data.bilateral_operator is not None:
            _solve_sparse_with_bilateral_direct_block(self, problem)
        elif _can_use_sparse_inequality_pgs(self):
            _solve_sparse_inequality_pgs(self, problem)
        elif self.has_unilateral_constraints:
            raise RuntimeError("Sparse DVI inequalities require limit/contact topology and sparse Jacobians.")
        else:
            _compute_sparse_solution_vectors(self, problem)


def _can_use_sparse_inequality_pgs(path: SparseDVIPath) -> bool:
    return _can_use_sparse_colored_inequalities(path)


def _can_use_sparse_colored_inequalities(path: SparseDVIPath) -> bool:
    has_limits = path.limits is not None and path.size.max_of_max_limits > 0
    has_contacts = path.contacts is not None and path.size.max_of_max_contacts > 0
    return path.jacobians is not None and (has_limits or has_contacts)


def _prepare_sparse_inequality_pgs(path: SparseDVIPath, problem: DualProblem) -> None:
    """Map and color all active inequalities for sparse PGS."""
    state = path.data.state
    limits = path.limits
    if limits is not None and limits.model_max_limits_host > 0:
        wp.launch(
            kernel=_map_active_limits,
            dim=limits.model_max_limits_host,
            inputs=[
                limits.model_active_limits,
                limits.wid,
                limits.lid,
                limits.bids,
                problem.data.lio,
                problem.data.uio,
                state.limit_indices,
                state.inequality_bodies,
            ],
            device=path.device,
        )
    contacts = path.contacts
    if contacts is not None and contacts.model_max_contacts_host > 0:
        wp.launch(
            kernel=_map_active_contacts,
            dim=contacts.model_max_contacts_host,
            inputs=[
                contacts.model_active_contacts,
                contacts.wid,
                contacts.cid,
                contacts.bid_AB,
                problem.data.nl,
                problem.data.cio,
                problem.data.uio,
                state.contact_indices,
                state.inequality_bodies,
            ],
            device=path.device,
        )
    wp.launch(
        kernel=_color_mapped_dvi_inequalities,
        dim=path.size.num_worlds,
        inputs=[
            problem.data.nl,
            problem.data.nc,
            problem.data.uio,
            state.inequality_bodies,
            state.inequality_colors,
            state.inequality_num_colors,
        ],
        device=path.device,
    )


def _launch_sparse_inequality_pgs(path: SparseDVIPath, problem: DualProblem, block_iteration: int) -> None:
    """Apply colored sparse PGS from the current full dual iterate."""
    state = path.data.state
    jacobians = path.jacobians
    if jacobians is None:
        raise RuntimeError("Sparse inequality PGS requires Jacobian topology.")
    delassus = _get_sparse_delassus(problem)
    bsm = delassus.bsm
    if bsm is None:
        raise RuntimeError("Sparse inequality PGS requires an initialized Delassus operator.")

    path.body_space.zero_()
    delassus.apply_jacobian_transpose(path.data.solution.lambdas, path.body_space, path.all_worlds_mask)
    threads_per_world = 64 if path.device.is_cuda else 1
    wp.launch(
        kernel=_solve_dvi_sparse_inequalities_pgs,
        dim=path.size.num_worlds * threads_per_world,
        inputs=[
            bsm.num_nzb,
            bsm.nzb_start,
            bsm.nzb_coords,
            bsm.nzb_values,
            delassus.constraint_jacobian.nzb_values,
            bsm.row_start,
            bsm.col_start,
            jacobians.limit_constraint_nzb_offsets,
            jacobians.contact_constraint_nzb_offsets,
            state.limit_indices,
            state.contact_indices,
            problem.data.nl,
            problem.data.nc,
            problem.data.lio,
            problem.data.cio,
            problem.data.uio,
            problem.data.lcgo,
            problem.data.ccgo,
            problem.data.vio,
            problem.data.mu,
            problem.data.P,
            problem.data.v_f,
            state.scratch,
            delassus.regularization,
            state.inequality_colors,
            state.inequality_num_colors,
            block_iteration,
            path.data.config,
            path.body_space,
            path.data.solution.lambdas,
        ],
        device=path.device,
        block_dim=threads_per_world,
    )


def _solve_sparse_inequality_pgs(path: SparseDVIPath, problem: DualProblem) -> None:
    delassus = _get_sparse_delassus(problem)
    delassus.diagonal(path.data.state.scratch)
    _prepare_sparse_inequality_pgs(path, problem)
    _launch_sparse_inequality_pgs(path, problem, block_iteration=-1)
    _compute_sparse_solution_vectors(path, problem)
    wp.launch(
        kernel=_set_dvi_sparse_status_iterations,
        dim=path.size.num_worlds,
        inputs=[problem.data.dim, path.data.config, path.data.status],
        device=path.device,
    )


def _get_sparse_delassus(problem: DualProblem) -> BlockSparseMatrixFreeDelassusOperator:
    delassus = problem.delassus
    if not isinstance(delassus, BlockSparseMatrixFreeDelassusOperator):
        raise TypeError("Sparse DVI requires a `BlockSparseMatrixFreeDelassusOperator`.")
    return delassus


def _compute_sparse_solution_vectors(path: SparseDVIPath, problem: DualProblem) -> None:
    state = path.data.state
    problem.delassus.matvec(
        x=path.data.solution.lambdas,
        y=state.v_aug,
        world_mask=path.all_worlds_mask,
    )
    wp.launch(
        kernel=_compute_dvi_sparse_solution_vectors,
        dim=(path.size.num_worlds, path.size.max_of_max_total_cts),
        inputs=[
            problem.data.dim,
            problem.data.vio,
            problem.data.v_f,
            state.s,
            state.v_aug,
            path.data.solution.v_plus,
        ],
        device=path.device,
    )


def _sparse_delassus_matvec_rows_path(path: SparseDVIPath, problem: DualProblem, row_kind: int) -> None:
    delassus = _get_sparse_delassus(problem)
    state = path.data.state
    regularization = delassus.regularization
    body_space = path.body_space
    bsm = delassus.bsm
    if bsm is None:
        raise RuntimeError("Sparse DVI row products require initialized Delassus sparse operators.")

    # Evaluate selected rows of D * lambda = J * M^-1 * J^T * lambda + R * lambda
    # without materializing the Delassus matrix.
    delassus.apply_jacobian_transpose(path.data.solution.lambdas, body_space, path.all_worlds_mask)
    state.v_aug.zero_()
    wp.launch(
        kernel=_sparse_delassus_gemv_rows,
        dim=(bsm.num_matrices, bsm.max_of_num_nzb),
        inputs=[
            bsm.dims,
            bsm.num_nzb,
            bsm.nzb_start,
            bsm.nzb_coords,
            bsm.nzb_values,
            bsm.row_start,
            bsm.col_start,
            problem.data.dim,
            problem.data.njc,
            row_kind,
            regularization,
            body_space,
            state.v_aug,
            path.data.solution.lambdas,
            path.all_worlds_mask,
        ],
        device=path.device,
    )


def _sparse_delassus_matvec_rows(solver, problem: DualProblem, row_kind: int) -> None:
    """Compatibility wrapper for sparse Delassus row products."""
    if solver._sparse_path is None:
        raise RuntimeError("Sparse DVI path has not been allocated. Call `finalize()` first.")
    _sparse_delassus_matvec_rows_path(solver._sparse_path, problem, row_kind)


def _compute_sparse_contact_block_inverse(path: SparseDVIPath, problem: DualProblem) -> None:
    jacobian = problem.delassus.constraint_jacobian
    wp.launch(
        kernel=sparse_kernels._compute_sparse_contact_block_inverse,
        dim=(path.size.num_worlds, path.size.max_of_max_contacts),
        inputs=[
            path.model.info.bodies_offset,
            path.model.bodies.inv_m_i,
            path.model_data.bodies.inv_I_i,
            jacobian.nzb_start,
            jacobian.num_nzb,
            jacobian.nzb_coords,
            jacobian.nzb_values,
            problem.data.nc,
            problem.data.ccgo,
            problem.data.cio,
            problem.data.vio,
            problem.data.P,
            path.data.config,
            jacobian.max_of_num_nzb,
            path.data.state.contact_block_inv,
        ],
        device=path.device,
    )


def _factor_sparse_bilateral_block(path: SparseDVIPath, problem: DualProblem) -> None:
    operator = path.data.bilateral_operator
    state = path.data.state
    operator.info.dim = operator.info.maxdim
    operator.mat.zero_()
    state.bilateral_preconditioner.zero_()
    problem.delassus.diagonal(state.scratch)

    jacobian = problem.delassus.constraint_jacobian
    if path.bilateral_nzb_pairs is None:
        raise RuntimeError("Sparse DVI topology is not prepared. Call `SparseDVIPath.prepare()` before solving.")
    wp.launch(
        kernel=_set_sparse_bilateral_diagonal,
        dim=(path.size.num_worlds, path.size.max_of_num_joint_cts),
        inputs=[
            problem.data.njc,
            problem.data.vio,
            operator.info.mio,
            operator.info.vio,
            state.scratch,
            operator.mat,
            state.bilateral_preconditioner,
        ],
        device=path.device,
    )
    pair_wid, pair_row, pair_col, pair_bid, pair_i, pair_j = path.bilateral_nzb_pairs
    if pair_wid.size > 0:
        wp.launch(
            kernel=_build_sparse_bilateral_block,
            dim=pair_wid.size,
            inputs=[
                path.model.bodies.inv_m_i,
                path.model_data.bodies.inv_I_i,
                pair_wid,
                pair_row,
                pair_col,
                pair_bid,
                pair_i,
                pair_j,
                jacobian.nzb_values,
                problem.data.njc,
                operator.info.mio,
                operator.info.vio,
                state.bilateral_preconditioner,
                operator.mat,
            ],
            device=path.device,
        )
    path.bilateral_solver.compute(A=operator.mat)


def _build_sparse_bilateral_pairs(path: SparseDVIPath, problem: DualProblem) -> None:
    """Cache joint Jacobian block pairs that contribute to the bilateral matrix."""
    jacobian = problem.delassus.constraint_jacobian
    counts = path.jacobians.joint_constraint_nzb_count.numpy().tolist()
    starts = jacobian.nzb_start.numpy().tolist()
    coords = jacobian.nzb_coords.numpy()
    joint_counts = problem.data.njc.numpy().tolist()
    body_offsets = path.model.info.bodies_offset.numpy().tolist()

    pair_wid: list[int] = []
    pair_row: list[int] = []
    pair_col: list[int] = []
    pair_bid: list[int] = []
    pair_i: list[int] = []
    pair_j: list[int] = []
    for wid, count in enumerate(counts):
        start = starts[wid]
        njc = joint_counts[wid]
        for local_i in range(count):
            nzb_i = start + local_i
            row = int(coords[nzb_i, 0])
            body_col = int(coords[nzb_i, 1])
            if row >= njc:
                continue
            for local_j in range(count):
                nzb_j = start + local_j
                col = int(coords[nzb_j, 0])
                if row < col < njc and body_col == int(coords[nzb_j, 1]):
                    pair_wid.append(wid)
                    pair_row.append(row)
                    pair_col.append(col)
                    pair_bid.append(body_offsets[wid] + body_col // 6)
                    pair_i.append(nzb_i)
                    pair_j.append(nzb_j)

    path.bilateral_nzb_pairs = tuple(
        wp.array(values, dtype=int32, device=path.device)
        for values in (pair_wid, pair_row, pair_col, pair_bid, pair_i, pair_j)
    )


def _solve_sparse_bilateral_block(
    path: SparseDVIPath, problem: DualProblem, active_dim: wp.array[int32] | None = None
) -> None:
    operator = path.data.bilateral_operator
    state = path.data.state
    wp.launch(
        kernel=_zero_bilateral_lambdas,
        dim=(path.size.num_worlds, path.size.max_of_num_joint_cts),
        inputs=[
            problem.data.njc,
            problem.data.vio,
            path.data.solution.lambdas,
        ],
        device=path.device,
    )
    _sparse_delassus_matvec_rows_path(path, problem, _SPARSE_DELASSUS_ROWS_JOINTS)
    wp.launch(
        kernel=_build_sparse_bilateral_rhs,
        dim=(path.size.num_worlds, path.size.max_of_num_joint_cts),
        inputs=[
            problem.data.vio,
            problem.data.njc,
            problem.data.v_f,
            state.v_aug,
            operator.info.vio,
            state.bilateral_preconditioner,
            state.bilateral_rhs,
        ],
        device=path.device,
    )
    full_dim = operator.info.dim
    if active_dim is not None:
        operator.info.dim = active_dim
    try:
        path.bilateral_solver.solve(b=state.bilateral_rhs, x=state.bilateral_solution)
    finally:
        operator.info.dim = full_dim
    wp.launch(
        kernel=_scatter_bilateral_solution,
        dim=(path.size.num_worlds, path.size.max_of_num_joint_cts),
        inputs=[
            problem.data.vio,
            problem.data.njc,
            operator.info.vio,
            state.bilateral_preconditioner,
            state.bilateral_solution,
            path.data.solution.lambdas,
        ],
        device=path.device,
    )


def _solve_sparse_with_bilateral_direct_block(path: SparseDVIPath, problem: DualProblem) -> None:
    """Alternate a direct ``D_bb`` solve with projected sparse unilateral sweeps."""
    state = path.data.state
    _factor_sparse_bilateral_block(path, problem)
    _solve_sparse_bilateral_block(path, problem)
    if not path.has_unilateral_constraints:
        _compute_sparse_solution_vectors(path, problem)
        return

    wp.launch(
        kernel=_initialize_dvi_status,
        dim=path.size.num_worlds,
        inputs=[
            path.data.config,
            path.data.status,
        ],
        device=path.device,
    )
    wp.launch(
        kernel=_set_dvi_bilateral_active_dim,
        dim=path.size.num_worlds,
        inputs=[
            problem.data.njc,
            problem.data.nl,
            problem.data.nc,
            state.bilateral_active_dim,
        ],
        device=path.device,
    )

    use_inequality_pgs = _can_use_sparse_colored_inequalities(path)
    if use_inequality_pgs:
        _prepare_sparse_inequality_pgs(path, problem)

    for block_iteration in range(path.max_block_iterations):
        if not use_inequality_pgs:
            raise RuntimeError("Sparse DVI inequalities require limit/contact topology and sparse Jacobians.")
        _launch_sparse_inequality_pgs(path, problem, block_iteration)

        if path.should_solve_bilateral_after_block(block_iteration):
            _solve_sparse_bilateral_block(path, problem, active_dim=state.bilateral_active_dim)

    _solve_sparse_bilateral_block(path, problem, active_dim=state.bilateral_active_dim)
    wp.launch(
        kernel=_set_dvi_direct_status_iterations,
        dim=path.size.num_worlds,
        inputs=[
            problem.data.nl,
            problem.data.nc,
            path.data.config,
            path.data.status,
        ],
        device=path.device,
    )
    _compute_sparse_solution_vectors(path, problem)
