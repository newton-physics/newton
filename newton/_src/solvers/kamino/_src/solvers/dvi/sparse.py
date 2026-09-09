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
from .apgd import ContactAPGDSolver
from .apgd_sparse import SparseContactOperator
from .kernels import (
    _FUSED_SINGLE_FAMILY_BLOCK,
    _INEQUALITY_FAMILY_CONTACTS,
    _INEQUALITY_FAMILY_LIMITS,
    _accumulate_dvi_apgd_status,
    _initialize_dvi_status,
    _scatter_bilateral_solution,
    _set_dvi_contact_active_mask,
    _set_dvi_direct_status_iterations,
)
from .sparse_kernels import (
    _build_sparse_bilateral_block,
    _build_sparse_bilateral_rhs,
    _color_mapped_dvi_inequalities,
    _compute_dvi_sparse_solution_vectors,
    _map_active_contacts,
    _map_active_limits,
    _map_bounded_constraints,
    _set_sparse_bilateral_diagonal,
    _solve_dvi_sparse_inequalities_pgs,
    _sparse_delassus_gemv_rows,
    _zero_bilateral_lambdas,
)

wp.set_module_options({"enable_backward": False})

int32 = wp.int32


_SPARSE_DELASSUS_ROWS_JOINTS = 0
_SPARSE_DELASSUS_ROWS_UNILATERAL = 1

_SPARSE_INEQUALITY_TOPOLOGY_ERROR = "Sparse DVI inequalities require limit/contact topology and sparse Jacobians."


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
        contact_solver: str,
        contact_apgd: ContactAPGDSolver | None,
        max_alternating_iterations: int,
        has_unilateral_constraints: bool,
        has_limit_constraints: bool,
        has_contact_constraints: bool,
        has_post_stabilization_bilateral: bool,
        all_worlds_mask: wp.array[wp.bool],
        set_bilateral_active_dim,
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
        self.contact_solver = contact_solver
        self.contact_apgd = contact_apgd
        self.contact_operator: SparseContactOperator | None = None
        self.max_alternating_iterations = max_alternating_iterations
        self.has_unilateral_constraints = has_unilateral_constraints
        self.has_limit_constraints = has_limit_constraints
        self.has_contact_constraints = has_contact_constraints
        self.has_post_stabilization_bilateral = has_post_stabilization_bilateral
        self.all_worlds_mask = all_worlds_mask
        self.set_bilateral_active_dim = set_bilateral_active_dim
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
        if self.contact_solver == "apgd" and self.contact_operator is None:
            self.contact_operator = _make_sparse_contact_operator(self, problem)
        if self.bilateral_solver is not None and self.data.bilateral_operator is not None:
            _build_sparse_bilateral_pairs(self, problem)

    def solve(self, problem: DualProblem) -> None:
        """Solve a sparse Kamino DVI problem without materializing dense Delassus."""
        if self.bilateral_solver is not None and self.data.bilateral_operator is not None:
            _solve_sparse_with_bilateral_direct_block(self, problem)
        elif _can_use_sparse_colored_inequalities(self):
            _solve_sparse_family_pgs(self, problem)
        elif self.has_unilateral_constraints:
            raise RuntimeError(_SPARSE_INEQUALITY_TOPOLOGY_ERROR)
        else:
            _compute_sparse_solution_vectors(self, problem)


def _can_use_sparse_colored_inequalities(path: SparseDVIPath) -> bool:
    has_limit_capacity = path.size.max_of_max_limits > 0
    has_contact_capacity = path.size.max_of_max_contacts > 0
    has_bounded_capacity = path.size.max_of_num_bounded_joint_cts > 0
    limits_ready = not has_limit_capacity or path.limits is not None
    contacts_ready = not has_contact_capacity or path.contacts is not None
    return (
        path.jacobians is not None
        and (has_limit_capacity or has_contact_capacity or has_bounded_capacity)
        and limits_ready
        and contacts_ready
    )


def _prepare_sparse_inequality_pgs(path: SparseDVIPath, problem: DualProblem) -> None:
    """Map and color active inequalities with the multi-world fast path."""
    state = path.data.state
    num_joints = path.size.sum_of_num_joints
    if num_joints > 0 and path.size.max_of_num_bounded_joint_cts > 0:
        joints = path.model.joints
        wp.launch(
            kernel=_map_bounded_constraints,
            dim=num_joints,
            inputs=[
                joints.wid,
                joints.bid_B,
                joints.bid_F,
                joints.bounded_cts_offset,
                problem.data.bcio,
                problem.data.iio,
                state.inequality_bodies,
            ],
            device=path.device,
        )
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
                problem.data.iio,
                problem.data.nbc,
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
                problem.data.nbc,
                problem.data.nl,
                problem.data.cio,
                problem.data.iio,
                state.contact_indices,
                state.inequality_bodies,
            ],
            device=path.device,
        )
    state.inequality_body_color_masks.zero_()
    wp.launch(
        kernel=_color_mapped_dvi_inequalities,
        dim=path.size.num_worlds,
        inputs=[
            problem.data.nbc,
            problem.data.nl,
            problem.data.nc,
            problem.data.iio,
            state.inequality_bodies,
            state.inequality_body_color_masks,
            state.inequality_colors,
            state.inequality_num_colors,
            state.inequality_ids_by_color,
            state.inequality_color_starts,
        ],
        device=path.device,
    )


def _launch_sparse_inequality_pgs(
    path: SparseDVIPath,
    problem: DualProblem,
    block_iteration: int,
    inequality_family: int,
) -> None:
    """Apply selected colored sparse PGS rows from the current full dual iterate."""
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
            jacobians.bounded_constraint_nzb_offsets,
            jacobians.limit_constraint_nzb_offsets,
            jacobians.contact_constraint_nzb_offsets,
            state.limit_indices,
            state.contact_indices,
            problem.data.nbc,
            problem.data.nl,
            problem.data.nc,
            problem.data.bcio,
            problem.data.lio,
            problem.data.cio,
            problem.data.iio,
            problem.data.bcgo,
            problem.data.lcgo,
            problem.data.ccgo,
            problem.data.vio,
            problem.data.mu,
            problem.data.bound_lower,
            problem.data.bound_upper,
            problem.data.P,
            problem.data.v_f,
            problem.data.E_hat,
            state.scratch,
            delassus.regularization,
            state.inequality_num_colors,
            state.inequality_ids_by_color,
            state.inequality_color_starts,
            block_iteration,
            inequality_family,
            path.data.config,
            path.body_space,
            path.data.solution.lambdas,
        ],
        device=path.device,
        block_dim=threads_per_world,
    )


def _solve_sparse_limit_phase(path: SparseDVIPath, problem: DualProblem, block_iteration: int) -> None:
    """Solve bounded joint rows and joint limits from the latest iterate."""
    _launch_sparse_inequality_pgs(path, problem, block_iteration, _INEQUALITY_FAMILY_LIMITS)


def _solve_sparse_contact_phase(path: SparseDVIPath, problem: DualProblem, block_iteration: int) -> None:
    """Solve contact triplets from the latest iterate with the selected backend."""
    if path.contact_solver == "apgd":
        _solve_sparse_contact_apgd(path, problem, block_iteration)
    else:
        _launch_sparse_inequality_pgs(path, problem, block_iteration, _INEQUALITY_FAMILY_CONTACTS)


def _make_sparse_contact_operator(path: SparseDVIPath, problem: DualProblem) -> SparseContactOperator:
    """Bind APGD to Kamino's raw sparse Jacobian and represented dual arrays."""
    if path.contact_apgd is None:
        raise RuntimeError("The APGD contact solver has not been allocated.")
    if path.model_data is None or path.jacobians is None:
        raise RuntimeError("Sparse contact APGD requires model data and sparse Jacobians.")
    delassus = _get_sparse_delassus(problem)
    jacobian = delassus.constraint_jacobian
    return SparseContactOperator(
        problem_dim=problem.data.dim,
        problem_vio=problem.data.vio,
        problem_nc=problem.data.nc,
        problem_cio=problem.data.cio,
        problem_ccgo=problem.data.ccgo,
        contact_indices=path.data.state.contact_indices,
        contact_nzb_offsets=path.jacobians.contact_constraint_nzb_offsets,
        jacobian_num_nzb=jacobian.num_nzb,
        jacobian_nzb_start=jacobian.nzb_start,
        jacobian_nzb_coords=jacobian.nzb_coords,
        jacobian_nzb_values=jacobian.nzb_values,
        jacobian_row_start=jacobian.row_start,
        jacobian_col_start=jacobian.col_start,
        body_offset=path.model.info.bodies_offset,
        body_inv_mass=path.model.bodies.inv_m_i,
        body_inv_inertia=path.model_data.bodies.inv_I_i,
        preconditioner=problem.data.P,
        regularization=delassus.regularization,
        represented_compliance=problem.data.E_hat,
        free_velocity=problem.data.v_f,
        contact_row_offset=path.contact_apgd.contact_row_offset,
        num_worlds=path.size.num_worlds,
        max_contacts_per_world=path.contact_apgd.max_contacts_per_world,
        max_bodies_per_world=path.size.max_of_num_bodies,
        total_body_dofs=path.size.sum_of_num_body_dofs,
        device=path.device,
    )


def _rebind_sparse_contact_operator(path: SparseDVIPath, problem: DualProblem) -> None:
    """Rebind graph-time workspace when a solver is reused with a new problem."""
    if path.contact_operator is None:
        path.contact_operator = _make_sparse_contact_operator(path, problem)
        return

    delassus = _get_sparse_delassus(problem)
    jacobian = delassus.constraint_jacobian
    operator = path.contact_operator
    operator.problem_dim = problem.data.dim
    operator.problem_vio = problem.data.vio
    operator.problem_nc = problem.data.nc
    operator.problem_cio = problem.data.cio
    operator.problem_ccgo = problem.data.ccgo
    operator.jacobian_num_nzb = jacobian.num_nzb
    operator.jacobian_nzb_start = jacobian.nzb_start
    operator.jacobian_nzb_coords = jacobian.nzb_coords
    operator.jacobian_nzb_values = jacobian.nzb_values
    operator.jacobian_row_start = jacobian.row_start
    operator.jacobian_col_start = jacobian.col_start
    operator.preconditioner = problem.data.P
    operator.regularization = delassus.regularization
    operator.represented_compliance = problem.data.E_hat
    operator.free_velocity = problem.data.v_f


def _prepare_sparse_contact_apgd(path: SparseDVIPath, problem: DualProblem) -> None:
    """Bind APGD arrays and build topology once for the complete family solve."""
    if path.contact_solver != "apgd" or not path.has_contact_constraints:
        return
    _rebind_sparse_contact_operator(path, problem)
    path.contact_operator.prepare()


def _solve_sparse_contact_apgd(path: SparseDVIPath, problem: DualProblem, block_iteration: int) -> None:
    """Solve the associated sparse contact subproblem with L and B fixed."""
    if path.contact_apgd is None:
        raise RuntimeError("The APGD contact solver has not been allocated.")
    wp.launch(
        kernel=_set_dvi_contact_active_mask,
        dim=path.size.num_worlds,
        inputs=[
            problem.data.nc,
            problem.data.ccgo,
            problem.data.vio,
            problem.data.P,
            block_iteration,
            path.data.config,
            path.data.status,
            path.data.state.contact_active_mask,
        ],
        device=path.device,
    )
    operator = path.contact_operator
    phase_mask = path.data.state.contact_active_mask
    operator.build_rhs(path.data.solution.lambdas, path.contact_apgd.rhs, phase_mask)
    operator.gather(path.data.solution.lambdas, path.contact_apgd.solution, phase_mask)
    apgd_status = path.contact_apgd.solve(
        problem.data.nc,
        problem.data.mu,
        path.contact_apgd.rhs,
        path.contact_apgd.solution,
        operator.matvec,
        phase_mask=phase_mask,
        contact_offset=problem.data.cio,
    )
    operator.scatter(path.contact_apgd.solution, path.data.solution.lambdas, phase_mask)
    wp.launch(
        kernel=_accumulate_dvi_apgd_status,
        dim=path.size.num_worlds,
        inputs=[apgd_status, phase_mask, path.data.status],
        device=path.device,
    )


def _solve_sparse_family_pgs(path: SparseDVIPath, problem: DualProblem) -> None:
    """Apply explicit ``L -> C`` family coupling without a bilateral block."""
    delassus = _get_sparse_delassus(problem)
    delassus.diagonal(path.data.state.scratch)
    _prepare_sparse_inequality_pgs(path, problem)
    _prepare_sparse_contact_apgd(path, problem)
    single_contact_phase = path.contact_solver == "apgd" and not path.has_limit_constraints
    fused_pgs_family = path.contact_solver == "pgs" and (path.has_limit_constraints != path.has_contact_constraints)
    if fused_pgs_family:
        if path.has_limit_constraints:
            _solve_sparse_limit_phase(path, problem, _FUSED_SINGLE_FAMILY_BLOCK)
        else:
            _solve_sparse_contact_phase(path, problem, _FUSED_SINGLE_FAMILY_BLOCK)
    else:
        family_iterations = 1 if single_contact_phase else path.max_alternating_iterations
        for block_iteration in range(family_iterations):
            if path.has_limit_constraints:
                _solve_sparse_limit_phase(path, problem, block_iteration)
            if path.has_contact_constraints:
                _solve_sparse_contact_phase(path, problem, block_iteration)
    _compute_sparse_solution_vectors(path, problem)
    wp.launch(
        kernel=_set_dvi_direct_status_iterations,
        dim=path.size.num_worlds,
        inputs=[
            problem.data.nbc,
            problem.data.nl,
            problem.data.nc,
            single_contact_phase,
            path.data.config,
            path.data.status,
        ],
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
            problem.data.E_hat,
            path.data.solution.lambdas,
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


def _factor_sparse_bilateral_block(path: SparseDVIPath, problem: DualProblem) -> None:
    """Build and factor the represented ``D_bb + E_hat_bb`` operator."""
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
        dim=(path.size.num_worlds, path.size.max_of_num_bilateral_joint_cts),
        inputs=[
            problem.data.njc,
            problem.data.vio,
            operator.info.mio,
            operator.info.vio,
            state.scratch,
            problem.data.P,
            problem.data.E_hat,
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
                problem.data.vio,
                problem.data.P,
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
        dim=(path.size.num_worlds, path.size.max_of_num_bilateral_joint_cts),
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
        dim=(path.size.num_worlds, path.size.max_of_num_bilateral_joint_cts),
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
        dim=(path.size.num_worlds, path.size.max_of_num_bilateral_joint_cts),
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
    """Factor ``D_bb + E_hat_bb`` once and execute the sparse coupling schedule."""
    _factor_sparse_bilateral_block(path, problem)
    if not path.has_unilateral_constraints:
        _solve_sparse_bilateral_block(path, problem)
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
    if not _can_use_sparse_colored_inequalities(path):
        raise RuntimeError(_SPARSE_INEQUALITY_TOPOLOGY_ERROR)
    _prepare_sparse_inequality_pgs(path, problem)
    _prepare_sparse_contact_apgd(path, problem)

    _solve_sparse_explicit_family_coupling(path, problem)

    wp.launch(
        kernel=_set_dvi_direct_status_iterations,
        dim=path.size.num_worlds,
        inputs=[
            problem.data.nbc,
            problem.data.nl,
            problem.data.nc,
            False,
            path.data.config,
            path.data.status,
        ],
        device=path.device,
    )
    _compute_sparse_solution_vectors(path, problem)


def _solve_sparse_explicit_family_coupling(path: SparseDVIPath, problem: DualProblem) -> None:
    """Apply exactly ``L -> B -> C`` per sparse coupling sweep."""
    state = path.data.state
    for block_iteration in range(path.max_alternating_iterations):
        if path.has_limit_constraints:
            _solve_sparse_limit_phase(path, problem, block_iteration)

        path.set_bilateral_active_dim(problem, block_iteration)
        _solve_sparse_bilateral_block(path, problem, active_dim=state.bilateral_active_dim)

        if path.has_contact_constraints:
            # Rebuilding body_space in this launch makes C see the B impulse
            # that was just scattered into the unified lambda vector.
            _solve_sparse_contact_phase(path, problem, block_iteration)

    _solve_sparse_post_stabilization_bilateral(path, problem)


def _solve_sparse_post_stabilization_bilateral(path: SparseDVIPath, problem: DualProblem) -> None:
    """Apply the configured optional terminal sparse bilateral refresh."""
    if not path.has_post_stabilization_bilateral:
        return
    path.set_bilateral_active_dim(problem, -1)
    _solve_sparse_bilateral_block(path, problem, active_dim=path.data.state.bilateral_active_dim)
