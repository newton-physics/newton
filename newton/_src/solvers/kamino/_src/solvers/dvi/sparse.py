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
from ...geometry.keying import KeySorter
from ...kinematics.jacobians import SparseSystemJacobians
from ...kinematics.limits import LimitsKamino
from ...linalg import LLTBlockedRCMSolver
from .cublas import is_batched_trsm_available, solve_llt_batched
from .kernels import (
    _FUSED_BILATERAL_BLOCK,
    _FUSED_INEQUALITY_BLOCK,
    _initialize_dvi_status,
    _pack_batched_bilateral_response,
    _scatter_batched_bilateral_response,
    _scatter_bilateral_solution,
    _set_dvi_direct_status_iterations,
    _solve_bilateral_unilateral_response,
    _solve_bilateral_unilateral_response_cooperative,
)
from .sparse_kernels import (
    _assemble_compact_unilateral_schur,
    _assemble_sparse_bilateral_unilateral_coupling,
    _build_sparse_bilateral_block,
    _build_sparse_bilateral_rhs,
    _cache_sparse_contact_diagonal,
    _cache_sparse_projected_diagonal,
    _color_compact_contact_groups,
    _compact_contact_group_starts,
    _compare_compact_contact_topology,
    _compute_dvi_sparse_solution_vectors,
    _expand_colored_contact_groups,
    _group_mapped_dvi_inequalities,
    _map_active_contacts,
    _map_active_limits,
    _map_ordered_active_contacts,
    _mark_contact_group_boundaries,
    _prefix_active_contacts_by_world,
    _prepare_colored_contact_group_sizes,
    _prepare_contact_pair_sort,
    _prepare_contact_world_sort,
    _reconstruct_fused_bilateral_solution,
    _reset_active_bilateral_delta,
    _select_parallel_contact_colors,
    _set_sparse_bilateral_diagonal,
    _solve_dvi_sparse_contacts_pgs,
    _solve_dvi_sparse_inequalities_pgs,
    _solve_dvi_sparse_inequalities_pgs_cooperative,
    _sparse_delassus_gemv_rows,
    _zero_bilateral_lambdas,
)

wp.set_module_options({"enable_backward": False})

int32 = wp.int32


_SPARSE_DELASSUS_ROWS_JOINTS = 0
_SPARSE_DELASSUS_ROWS_UNILATERAL = 1
_CONTACT_PAIR_SORT_MIN_CAPACITY = 4096
_PARALLEL_CONTACT_MAX_COLORS = 8
_PARALLEL_CONTACT_MIN_CAPACITY = 32768
_BATCHED_RESPONSE_MIN_WORLDS = 16
_BATCHED_RESPONSE_RHS = 64
_BATCHED_RESPONSE_TAIL_TASKS = 8

_SPARSE_INEQUALITY_TOPOLOGY_ERROR = "Sparse DVI inequalities require limit/contact topology and sparse Jacobians."


def _use_parallel_contact_colors(num_worlds: int, max_limits: int, max_contacts: int, is_cuda: bool) -> bool:
    """Use fixed color nodes only for contact-rich one-world CUDA capacity."""
    return is_cuda and num_worlds == 1 and max_limits == 0 and max_contacts >= _PARALLEL_CONTACT_MIN_CAPACITY


def _parallel_contact_group_width(sm_count: int, group_capacity: int) -> int:
    """Trade idle SIMD lanes for enough independent warps on low-world contact solves."""
    # A color commonly occupies about half the body-derived group capacity. Aim
    # for two useful warps per SM, retain at least two lanes/group to amortize
    # the runtime mapping, and cap the deliberate SIMD underfill.
    required = max(2, (4 * sm_count * 32 + group_capacity - 1) // group_capacity)
    return min(32, 1 << (required - 1).bit_length())


class SparseDVIPath:
    """Own workspace and operations for the sparse Kamino DVI solve path."""

    def __init__(
        self,
        device: wp.DeviceLike,
        size,
        joint_rows_host: list[int],
        unilateral_strides_host: list[int],
        data,
        model: ModelKamino,
        model_data: DataKamino | None,
        limits: LimitsKamino | None,
        contacts: ContactsKamino | None,
        jacobians: SparseSystemJacobians | None,
        bilateral_solver,
        max_alternating_iterations: int,
        max_inequality_sweeps_per_iteration: int,
        has_unilateral_constraints: bool,
        all_worlds_mask: wp.array[wp.bool],
        should_solve_bilateral_after_block,
        set_bilateral_active_dim,
    ):
        """Initialize the sparse-path workspace references."""
        self.device = device
        self.size = size
        self.joint_rows_host = joint_rows_host
        self.unilateral_strides_host = unilateral_strides_host
        self.data = data
        self.model = model
        self.model_data = model_data
        self.limits = limits
        self.contacts = contacts
        self.jacobians = jacobians
        self.body_space = wp.empty(shape=size.sum_of_num_body_dofs, dtype=wp.float32, device=device)
        self.parallel_contact_colors = wp.zeros(shape=1, dtype=wp.int32, device=device)
        self.bilateral_solver = bilateral_solver
        self.max_alternating_iterations = max_alternating_iterations
        self.max_inequality_sweeps_per_iteration = max_inequality_sweeps_per_iteration
        self.has_unilateral_constraints = has_unilateral_constraints
        self.all_worlds_mask = all_worlds_mask
        self.should_solve_bilateral_after_block = should_solve_bilateral_after_block
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
        self.bilateral_row_nzb_topology: tuple[wp.array[wp.int32], wp.array[wp.int32], wp.array[wp.int32]] | None = None
        self.batched_response_factor_ptrs: wp.array[wp.uint64] | None = None
        self.batched_response_rhs_ptrs: wp.array[wp.uint64] | None = None
        self.batched_response_rows = 0
        self.contact_sorter: KeySorter | None = None
        self.contact_world_starts: wp.array[wp.int32] | None = None
        if device.is_cuda and size.max_of_max_contacts >= _CONTACT_PAIR_SORT_MIN_CAPACITY:
            num_contacts = size.sum_of_max_contacts
            self.contact_sorter = KeySorter(max_num_keys=num_contacts, device=device)
            self.contact_world_starts = wp.zeros(size.num_worlds + 1, dtype=int32, device=device)
            self.contact_group_flags = wp.array(
                ptr=self.contact_sorter.sorted_to_unsorted_map.ptr + num_contacts * wp.types.type_size_in_bytes(int32),
                shape=num_contacts,
                dtype=int32,
                device=device,
                copy=False,
            )
            self.contact_group_starts = wp.array(
                ptr=self.contact_sorter.sorted_keys.ptr,
                shape=num_contacts,
                dtype=int32,
                device=device,
                copy=False,
            )
            if size.num_worlds == 1 and size.max_of_max_limits == 0:
                self.contact_groups_by_color = wp.empty(num_contacts, dtype=int32, device=device)
                self.cached_contact_group_pairs = wp.empty(num_contacts, dtype=wp.vec2i, device=device)
                self.cached_contact_group_count = wp.zeros(1, dtype=int32, device=device)
                self.cached_contact_num_colors = wp.zeros(1, dtype=int32, device=device)
                self.cached_contact_color_starts = wp.empty(num_contacts + 1, dtype=int32, device=device)
                self.contact_topology_cache_valid = wp.zeros(1, dtype=int32, device=device)
                self.contact_topology_changed = wp.zeros(1, dtype=int32, device=device)

    def prepare(self, problem: DualProblem) -> None:
        """Precompute host-derived sparse topology before the first solve."""
        _get_sparse_delassus(problem)
        if self.model_data is None or self.jacobians is None:
            raise RuntimeError("Sparse DVI requires model data and sparse Jacobians.")
        if self.bilateral_solver is not None and self.data.bilateral_operator is not None:
            _build_sparse_bilateral_pairs(self, problem)
            _build_sparse_bilateral_row_nzb_topology(self, problem)

            _prepare_batched_bilateral_response(self)

    def solve(self, problem: DualProblem) -> None:
        """Solve a sparse Kamino DVI problem without materializing dense Delassus."""
        if self.bilateral_solver is not None and self.data.bilateral_operator is not None:
            _solve_sparse_with_bilateral_direct_block(self, problem)
        elif _can_use_sparse_colored_inequalities(self):
            _solve_sparse_inequality_pgs(self, problem)
        elif self.has_unilateral_constraints:
            raise RuntimeError(_SPARSE_INEQUALITY_TOPOLOGY_ERROR)
        else:
            _compute_sparse_solution_vectors(self, problem)


def _can_use_sparse_colored_inequalities(path: SparseDVIPath) -> bool:
    has_limit_capacity = path.size.max_of_max_limits > 0
    has_contact_capacity = path.size.max_of_max_contacts > 0
    limits_ready = not has_limit_capacity or path.limits is not None
    contacts_ready = not has_contact_capacity or path.contacts is not None
    return (
        path.jacobians is not None and (has_limit_capacity or has_contact_capacity) and limits_ready and contacts_ready
    )


def _prepare_sparse_inequality_pgs(path: SparseDVIPath, problem: DualProblem) -> None:
    """Map and color active inequalities with the multi-world fast path."""
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
                path.model.bodies.effective_inv_m_i,
                problem.data.lio,
                problem.data.uio,
                state.limit_indices,
                state.inequality_bodies,
            ],
            device=path.device,
        )
    contacts = path.contacts
    use_contact_order = path.contact_sorter is not None and contacts is not None
    if contacts is not None and contacts.model_max_contacts_host > 0:
        if use_contact_order:
            sorter = path.contact_sorter
            wp.launch(
                kernel=_prepare_contact_pair_sort,
                dim=contacts.model_max_contacts_host,
                inputs=[
                    contacts.model_active_contacts,
                    contacts.gid_AB,
                    sorter.sorted_keys,
                    sorter.sorted_to_unsorted_map,
                ],
                device=path.device,
            )
            wp.utils.radix_sort_pairs(
                sorter.sorted_keys_int64,
                sorter.sorted_to_unsorted_map,
                contacts.model_max_contacts_host,
            )
            wp.launch(
                kernel=_prepare_contact_world_sort,
                dim=contacts.model_max_contacts_host,
                inputs=[
                    contacts.model_active_contacts,
                    contacts.wid,
                    sorter.sorted_to_unsorted_map,
                    sorter.sorted_keys,
                ],
                device=path.device,
            )
            wp.utils.radix_sort_pairs(
                sorter.sorted_keys_int64,
                sorter.sorted_to_unsorted_map,
                contacts.model_max_contacts_host,
            )
            wp.launch(
                kernel=_prefix_active_contacts_by_world,
                dim=1,
                inputs=[path.size.num_worlds, problem.data.nc, path.contact_world_starts],
                device=path.device,
            )
            wp.launch(
                kernel=_map_ordered_active_contacts,
                dim=contacts.model_max_contacts_host,
                inputs=[
                    contacts.model_active_contacts,
                    contacts.wid,
                    contacts.cid,
                    contacts.bid_AB,
                    sorter.sorted_to_unsorted_map,
                    path.contact_world_starts,
                    path.model.bodies.effective_inv_m_i,
                    problem.data.nl,
                    problem.data.cio,
                    problem.data.uio,
                    state.contact_indices,
                    state.inequality_bodies,
                    state.inequality_group_starts,
                ],
                device=path.device,
            )
        else:
            wp.launch(
                kernel=_map_active_contacts,
                dim=contacts.model_max_contacts_host,
                inputs=[
                    contacts.model_active_contacts,
                    contacts.wid,
                    contacts.cid,
                    contacts.bid_AB,
                    path.model.bodies.effective_inv_m_i,
                    problem.data.nl,
                    problem.data.cio,
                    problem.data.uio,
                    state.contact_indices,
                    state.inequality_bodies,
                ],
                device=path.device,
            )
    state.inequality_body_color_masks.zero_()
    use_parallel_groups = use_contact_order and path.size.num_worlds == 1 and path.size.max_of_max_limits == 0
    if not use_parallel_groups:
        wp.launch(
            kernel=_group_mapped_dvi_inequalities,
            dim=path.size.num_worlds,
            inputs=[
                problem.data.nl,
                problem.data.nc,
                problem.data.uio,
                state.inequality_bodies,
                state.inequality_body_color_masks,
                state.inequality_colors,
                state.inequality_num_colors,
                state.inequality_ids_by_color,
                state.inequality_color_starts,
                state.inequality_group_starts,
                state.inequality_group_starts,
                wp.bool(use_contact_order),
            ],
            device=path.device,
        )
        return

    sorter = path.contact_sorter
    wp.launch(
        kernel=_mark_contact_group_boundaries,
        dim=contacts.model_max_contacts_host,
        inputs=[
            contacts.model_active_contacts,
            sorter.sorted_to_unsorted_map,
            contacts.cid,
            problem.data.uio,
            state.inequality_bodies,
            path.contact_group_flags,
        ],
        device=path.device,
    )
    wp.utils.array_scan(path.contact_group_flags, state.inequality_colors, inclusive=True)
    wp.launch(
        kernel=_compact_contact_group_starts,
        dim=contacts.model_max_contacts_host,
        inputs=[
            contacts.model_active_contacts,
            path.contact_group_flags,
            state.inequality_colors,
            path.contact_group_starts,
        ],
        device=path.device,
    )
    path.contact_topology_changed.zero_()
    wp.launch(
        kernel=_compare_compact_contact_topology,
        dim=contacts.model_max_contacts_host,
        inputs=[
            contacts.model_active_contacts,
            state.inequality_colors,
            problem.data.uio,
            sorter.sorted_to_unsorted_map,
            contacts.cid,
            state.inequality_bodies,
            path.contact_group_starts,
            path.cached_contact_group_pairs,
            path.cached_contact_group_count,
            path.contact_topology_cache_valid,
            path.contact_topology_changed,
        ],
        device=path.device,
    )
    wp.launch(
        kernel=_color_compact_contact_groups,
        dim=1,
        inputs=[
            contacts.model_active_contacts,
            state.inequality_colors,
            problem.data.uio,
            sorter.sorted_to_unsorted_map,
            contacts.cid,
            state.inequality_bodies,
            state.inequality_body_color_masks,
            path.contact_group_starts,
            state.inequality_colors,
            state.inequality_num_colors,
            path.contact_world_starts,
            state.inequality_color_starts,
            path.contact_groups_by_color,
            path.cached_contact_group_pairs,
            path.cached_contact_group_count,
            path.cached_contact_num_colors,
            path.cached_contact_color_starts,
            path.contact_topology_cache_valid,
            path.contact_topology_changed,
        ],
        device=path.device,
    )
    state.inequality_colors.zero_()
    group_dim = path.size.max_of_max_contacts + 1
    wp.launch(
        kernel=_prepare_colored_contact_group_sizes,
        dim=group_dim,
        inputs=[
            contacts.model_active_contacts,
            path.contact_world_starts,
            path.contact_group_starts,
            path.contact_groups_by_color,
            state.inequality_colors,
        ],
        device=path.device,
    )
    wp.utils.array_scan(state.inequality_colors, state.inequality_colors, inclusive=True)
    wp.launch(
        kernel=_expand_colored_contact_groups,
        dim=group_dim,
        inputs=[
            contacts.model_active_contacts,
            path.contact_world_starts,
            problem.data.uio,
            sorter.sorted_to_unsorted_map,
            contacts.cid,
            path.contact_group_starts,
            path.contact_groups_by_color,
            state.inequality_colors,
            state.inequality_ids_by_color,
            state.inequality_group_starts,
        ],
        device=path.device,
    )


def _launch_sparse_inequality_pgs(
    path: SparseDVIPath,
    problem: DualProblem,
    block_iteration: int,
    enable_compact_schur: bool = False,
) -> None:
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
    bilateral_vio = (
        path.data.bilateral_operator.info.vio if path.data.bilateral_operator is not None else problem.data.vio
    )
    delassus.apply_jacobian_transpose(path.data.solution.lambdas, path.body_space, path.all_worlds_mask)
    threads_per_world = 1
    if path.device.is_cuda:
        threads_per_world = 64
        if path.size.max_of_max_contacts >= 2048:
            # This kernel exceeds CUDA graph resource limits at 512 threads on some devices.
            threads_per_world = 256
    contact_only = path.size.max_of_max_limits == 0 and path.bilateral_solver is None
    parallel_contact_path = contact_only and _use_parallel_contact_colors(
        path.size.num_worlds,
        path.size.max_of_max_limits,
        path.size.max_of_max_contacts,
        path.device.is_cuda,
    )
    kernel = _solve_dvi_sparse_contacts_pgs if contact_only else _solve_dvi_sparse_inequalities_pgs
    if contact_only:
        wp.launch(
            kernel=_cache_sparse_contact_diagonal,
            dim=(path.size.num_worlds, path.size.max_of_max_contacts),
            inputs=[
                problem.data.nc,
                problem.data.ccgo,
                problem.data.vio,
                problem.data.P,
                state.scratch,
                state.inequality_projected_diagonal,
            ],
            device=path.device,
        )
    cooperative_articulation = (
        path.device.is_cuda and path.bilateral_solver is not None and path.size.max_of_num_joint_cts >= 64
    )
    if cooperative_articulation:
        kernel = _solve_dvi_sparse_inequalities_pgs_cooperative
        threads_per_world = 32
    common_inputs = [
        bsm.num_nzb,
        bsm.nzb_start,
        bsm.nzb_coords,
        bsm.nzb_values,
        delassus.constraint_jacobian.nzb_values,
        bsm.row_start,
        bsm.col_start,
    ]
    if contact_only:
        kernel_inputs = [
            *common_inputs,
            jacobians.contact_constraint_nzb_offsets,
            state.contact_indices,
            path.contacts.bid_AB,
            path.model.info.bodies_offset,
            problem.data.nc,
            problem.data.cio,
            problem.data.uio,
            problem.data.ccgo,
            problem.data.vio,
            problem.data.mu,
            problem.data.P,
            problem.data.v_f,
            problem.data.v_b,
            state.inequality_projected_diagonal,
            delassus.regularization,
            state.inequality_num_colors,
            state.inequality_ids_by_color,
            state.inequality_color_starts,
            state.inequality_group_starts,
            state.inequality_tangent_cross,
        ]
        parallel_mode_offset = len(kernel_inputs)
        kernel_inputs.extend(
            [
                path.parallel_contact_colors,
                wp.bool(False),
                int32(-1),
                int32(-1),
                threads_per_world,
                int32(1),
                block_iteration,
                path.data.config,
                path.body_space,
                path.data.solution.lambdas,
            ]
        )
    else:
        kernel_inputs = [
            *common_inputs,
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
            problem.data.v_b,
            state.scratch,
            state.inequality_projected_diagonal,
            delassus.regularization,
            problem.data.njc,
            bilateral_vio,
            state.bilateral_response_mio,
            state.bilateral_response_stride,
            state.bilateral_coupling,
            state.bilateral_response,
            state.bilateral_delta,
        ]
        if cooperative_articulation:
            kernel_inputs.extend(
                [
                    state.bilateral_response_factor,
                    state.s,
                    wp.bool(enable_compact_schur),
                ]
            )
        kernel_inputs.extend(
            [
                state.inequality_num_colors,
                state.inequality_ids_by_color,
                state.inequality_color_starts,
                state.inequality_group_starts,
                state.inequality_tangent_cross,
                block_iteration,
                path.data.config,
                *([path.data.status] if cooperative_articulation else []),
                path.body_space,
                path.data.solution.lambdas,
            ]
        )
    if parallel_contact_path:
        wp.launch(
            kernel=_select_parallel_contact_colors,
            dim=1,
            inputs=[
                problem.data.nc,
                state.inequality_num_colors,
                _CONTACT_PAIR_SORT_MIN_CAPACITY,
                _PARALLEL_CONTACT_MAX_COLORS,
                path.parallel_contact_colors,
            ],
            device=path.device,
        )
        base_sweeps = path.max_inequality_sweeps_per_iteration
        alternating_iterations = path.max_alternating_iterations
        tangent_sweeps = base_sweeps * alternating_iterations // 2
        total_sweeps = (base_sweeps + 1) * alternating_iterations + tangent_sweeps + 1
        group_dim = path.size.max_of_num_bodies + 1
        group_width = _parallel_contact_group_width(path.device.sm_count, group_dim)
        for sweep in range(total_sweeps):
            for color_ordinal in range(_PARALLEL_CONTACT_MAX_COLORS):
                node_inputs = kernel_inputs.copy()
                node_inputs[parallel_mode_offset + 1] = wp.bool(True)
                node_inputs[parallel_mode_offset + 2] = int32(sweep)
                node_inputs[parallel_mode_offset + 3] = int32(color_ordinal)
                node_inputs[parallel_mode_offset + 4] = group_dim
                node_inputs[parallel_mode_offset + 5] = int32(group_width)
                wp.launch(
                    kernel=kernel,
                    dim=group_dim * group_width,
                    inputs=node_inputs,
                    device=path.device,
                    block_dim=64,
                )
    wp.launch(
        kernel=kernel,
        dim=path.size.num_worlds * threads_per_world,
        inputs=kernel_inputs,
        device=path.device,
        block_dim=threads_per_world,
    )


def _solve_sparse_inequality_pgs(path: SparseDVIPath, problem: DualProblem) -> None:
    delassus = _get_sparse_delassus(problem)
    delassus.diagonal(path.data.state.scratch)
    _prepare_sparse_inequality_pgs(path, problem)
    # Inequality-only solves need no host work between PGS blocks.
    for block_iteration in (_FUSED_INEQUALITY_BLOCK,):
        _launch_sparse_inequality_pgs(path, problem, block_iteration)
    _compute_sparse_solution_vectors(path, problem)
    wp.launch(
        kernel=_set_dvi_direct_status_iterations,
        dim=path.size.num_worlds,
        inputs=[problem.data.nl, problem.data.nc, path.data.config, False, path.data.status],
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
                path.model.bodies.effective_inv_m_i,
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


def _build_sparse_bilateral_row_nzb_topology(path: SparseDVIPath, problem: DualProblem) -> None:
    """Cache joint Jacobian blocks by bilateral row in their original storage order."""
    jacobian = problem.delassus.constraint_jacobian
    counts = path.jacobians.joint_constraint_nzb_count.numpy().tolist()
    matrix_starts = jacobian.nzb_start.numpy().tolist()
    coords = jacobian.nzb_coords.numpy()
    joint_counts = problem.data.njc.numpy().tolist()

    world_row_offsets = []
    row_starts = [0]
    row_nzb_indices = []
    row_offset = 0
    for count, matrix_start, njc in zip(counts, matrix_starts, joint_counts, strict=True):
        world_row_offsets.append(row_offset)
        for row in range(njc):
            for local_block in range(count):
                block = matrix_start + local_block
                if int(coords[block, 0]) == row:
                    row_nzb_indices.append(block)
            row_starts.append(len(row_nzb_indices))
        row_offset += njc

    path.bilateral_row_nzb_topology = tuple(
        wp.array(values, dtype=int32, device=path.device) for values in (world_row_offsets, row_starts, row_nzb_indices)
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


def _prepare_batched_bilateral_response(path: SparseDVIPath) -> None:
    """Prepare stable pointer arrays for the many-world batched response solve."""
    path.batched_response_factor_ptrs = None
    path.batched_response_rhs_ptrs = None
    path.batched_response_rows = 0
    if (
        not path.device.is_cuda
        or path.size.num_worlds < _BATCHED_RESPONSE_MIN_WORLDS
        or not path.joint_rows_host
        or path.data.state.bilateral_response_factor is None
        or not is_batched_trsm_available(path.device)
    ):
        return
    rows = path.joint_rows_host[0]
    if (
        rows <= 0
        or any(world_rows != rows for world_rows in path.joint_rows_host)
        or any(stride < _BATCHED_RESPONSE_RHS for stride in path.unilateral_strides_host)
    ):
        return

    factor_base = path.bilateral_solver.L.ptr
    rhs_base = path.data.state.bilateral_response_factor.ptr
    scalar_bytes = wp.types.type_size_in_bytes(wp.float32)
    factor_ptrs = []
    rhs_ptrs = []
    factor_offset = 0
    rhs_offset = 0
    for stride in path.unilateral_strides_host:
        factor_ptrs.append(factor_base + scalar_bytes * factor_offset)
        rhs_ptrs.append(rhs_base + scalar_bytes * rhs_offset)
        factor_offset += rows * rows
        rhs_offset += rows * stride
    path.batched_response_factor_ptrs = wp.array(factor_ptrs, dtype=wp.uint64, device=path.device)
    path.batched_response_rhs_ptrs = wp.array(rhs_ptrs, dtype=wp.uint64, device=path.device)
    path.batched_response_rows = rows


def _solve_batched_bilateral_response(
    path: SparseDVIPath,
    problem: DualProblem,
    permutation: wp.array[wp.int32],
    use_permutation: bool,
) -> None:
    """Solve a fixed RHS prefix through cuBLAS and any active tail in Warp."""
    factor_ptrs = path.batched_response_factor_ptrs
    rhs_ptrs = path.batched_response_rhs_ptrs
    if factor_ptrs is None or rhs_ptrs is None:
        raise RuntimeError("Batched response pointer arrays were not prepared.")
    state = path.data.state
    operator = path.data.bilateral_operator
    wp.launch(
        kernel=_pack_batched_bilateral_response,
        dim=(path.size.num_worlds, _BATCHED_RESPONSE_RHS, path.batched_response_rows),
        inputs=[
            problem.data.dim,
            problem.data.njc,
            operator.info.vio,
            state.bilateral_preconditioner,
            permutation,
            use_permutation,
            state.bilateral_response_mio,
            state.bilateral_response_stride,
            False,
            state.bilateral_response_mio,
            state.bilateral_coupling,
            state.bilateral_response_factor,
        ],
        device=path.device,
    )
    solve_llt_batched(
        factor_ptrs,
        rhs_ptrs,
        path.batched_response_rows,
        _BATCHED_RESPONSE_RHS,
        path.size.num_worlds,
    )
    wp.launch(
        kernel=_scatter_batched_bilateral_response,
        dim=(path.size.num_worlds, _BATCHED_RESPONSE_RHS, path.batched_response_rows),
        inputs=[
            problem.data.dim,
            problem.data.njc,
            operator.info.vio,
            state.bilateral_preconditioner,
            permutation,
            use_permutation,
            state.bilateral_response_mio,
            state.bilateral_response_mio,
            state.bilateral_response_stride,
            False,
            True,
            state.bilateral_response_factor,
            state.bilateral_response,
        ],
        device=path.device,
    )
    wp.launch(
        kernel=_solve_bilateral_unilateral_response_cooperative,
        dim=path.size.num_worlds * _BATCHED_RESPONSE_TAIL_TASKS * 32,
        inputs=[
            problem.data.dim,
            problem.data.njc,
            operator.info.mio,
            operator.info.vio,
            state.bilateral_preconditioner,
            path.bilateral_solver.L,
            permutation,
            use_permutation,
            state.bilateral_response_mio,
            state.bilateral_response_stride,
            state.bilateral_coupling,
            state.bilateral_response_factor,
            state.bilateral_response,
            _BATCHED_RESPONSE_RHS,
            _BATCHED_RESPONSE_TAIL_TASKS,
        ],
        device=path.device,
        block_dim=32,
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
    if not _can_use_sparse_colored_inequalities(path):
        raise RuntimeError(_SPARSE_INEQUALITY_TOPOLOGY_ERROR)
    _prepare_sparse_inequality_pgs(path, problem)

    delassus = _get_sparse_delassus(problem)
    bsm = delassus.bsm
    if path.bilateral_row_nzb_topology is None:
        raise RuntimeError("Sparse DVI topology is not prepared. Call `SparseDVIPath.prepare()` before solving.")
    world_row_offsets, row_starts, row_nzb_indices = path.bilateral_row_nzb_topology
    max_joint_rows = path.size.max_of_num_joint_cts
    max_unilateral_rows = path.size.max_of_max_limits + 3 * path.size.max_of_max_contacts
    # Coupling and response kernels overwrite every active entry; only the
    # accumulated bilateral correction must start from zero.
    state.bilateral_delta.zero_()
    wp.launch(
        kernel=_assemble_sparse_bilateral_unilateral_coupling,
        dim=(path.size.num_worlds, max_joint_rows, max_unilateral_rows),
        inputs=[
            bsm.num_nzb,
            bsm.nzb_start,
            bsm.nzb_coords,
            bsm.nzb_values,
            delassus.constraint_jacobian.nzb_values,
            problem.data.dim,
            problem.data.njc,
            problem.data.nl,
            problem.data.nc,
            problem.data.lio,
            problem.data.cio,
            problem.data.vio,
            problem.data.P,
            state.limit_indices,
            state.contact_indices,
            path.jacobians.limit_constraint_nzb_offsets,
            path.jacobians.contact_constraint_nzb_offsets,
            world_row_offsets,
            row_starts,
            row_nzb_indices,
            state.bilateral_response_mio,
            state.bilateral_response_stride,
            state.bilateral_coupling,
        ],
        device=path.device,
    )
    use_permutation = isinstance(path.bilateral_solver, LLTBlockedRCMSolver)
    permutation = path.bilateral_solver.P if use_permutation else state.projected_mio
    has_intermediate_bilateral_solve = any(
        path.should_solve_bilateral_after_block(block_iteration)
        for block_iteration in range(path.max_alternating_iterations)
    )
    enable_compact_schur = (
        path.device.is_cuda
        and path.size.max_of_num_joint_cts >= 64
        and path.max_alternating_iterations >= 4
        and not has_intermediate_bilateral_solve
    )
    response_kernel = _solve_bilateral_unilateral_response
    response_block_dim = 1
    response_tasks_per_world = 0
    response_dim = path.size.num_worlds
    if path.device.is_cuda:
        response_kernel = _solve_bilateral_unilateral_response_cooperative
        response_block_dim = 32
        response_tasks_per_world = (max_unilateral_rows + 1) // 2
        response_dim = path.size.num_worlds * response_tasks_per_world * response_block_dim
    if path.batched_response_factor_ptrs is not None:
        _solve_batched_bilateral_response(path, problem, permutation, use_permutation)
    else:
        wp.launch(
            kernel=response_kernel,
            dim=response_dim,
            inputs=[
                problem.data.dim,
                problem.data.njc,
                path.data.bilateral_operator.info.mio,
                path.data.bilateral_operator.info.vio,
                state.bilateral_preconditioner,
                path.bilateral_solver.L,
                permutation,
                use_permutation,
                state.bilateral_response_mio,
                state.bilateral_response_stride,
                state.bilateral_coupling,
                state.bilateral_response_factor,
                state.bilateral_response,
                *([0, response_tasks_per_world] if path.device.is_cuda else []),
            ],
            device=path.device,
            block_dim=response_block_dim,
        )
    if enable_compact_schur:
        wp.launch(
            kernel=_assemble_compact_unilateral_schur,
            dim=path.size.num_worlds * 256,
            inputs=[
                problem.data.dim,
                problem.data.njc,
                problem.data.vio,
                state.bilateral_response_mio,
                state.bilateral_response_stride,
                state.bilateral_coupling,
                state.bilateral_response,
                state.bilateral_response_factor,
                state.s,
            ],
            device=path.device,
            block_dim=256,
        )
    wp.launch(
        kernel=_cache_sparse_projected_diagonal,
        dim=(path.size.num_worlds, max_unilateral_rows),
        inputs=[
            problem.data.dim,
            problem.data.njc,
            problem.data.vio,
            problem.data.P,
            state.scratch,
            state.bilateral_response_mio,
            state.bilateral_response_stride,
            state.bilateral_coupling,
            state.bilateral_response,
            path.data.solution.lambdas,
            state.v_aug,
            state.inequality_projected_diagonal,
        ],
        device=path.device,
    )
    if not has_intermediate_bilateral_solve:
        # A fixed bilateral response lets block-local barriers preserve colored GS across all sweeps.
        _launch_sparse_inequality_pgs(path, problem, _FUSED_BILATERAL_BLOCK, enable_compact_schur=enable_compact_schur)
    else:
        for block_iteration in range(path.max_alternating_iterations):
            _launch_sparse_inequality_pgs(path, problem, block_iteration)
            if not path.should_solve_bilateral_after_block(block_iteration):
                continue
            path.set_bilateral_active_dim(problem, block_iteration)
            _solve_sparse_bilateral_block(path, problem, active_dim=state.bilateral_active_dim)
            wp.launch(
                kernel=_reset_active_bilateral_delta,
                dim=(path.size.num_worlds, path.size.max_of_num_joint_cts),
                inputs=[
                    state.bilateral_active_dim,
                    path.data.bilateral_operator.info.vio,
                    state.bilateral_delta,
                ],
                device=path.device,
            )

    cooperative_fused_pgs = path.device.is_cuda and path.size.max_of_num_joint_cts >= 64
    if has_intermediate_bilateral_solve or not cooperative_fused_pgs:
        path.set_bilateral_active_dim(problem, -1)
        _solve_sparse_bilateral_block(path, problem, active_dim=state.bilateral_active_dim)
    else:
        wp.launch(
            kernel=_reconstruct_fused_bilateral_solution,
            dim=(path.size.num_worlds, path.size.max_of_num_joint_cts),
            inputs=[
                problem.data.dim,
                problem.data.njc,
                problem.data.vio,
                path.data.bilateral_operator.info.vio,
                state.bilateral_response_mio,
                state.bilateral_response_stride,
                state.bilateral_response,
                state.v_aug,
                state.bilateral_delta,
                wp.bool(enable_compact_schur),
                path.data.solution.lambdas,
            ],
            device=path.device,
        )
    if has_intermediate_bilateral_solve or not cooperative_fused_pgs:
        wp.launch(
            kernel=_set_dvi_direct_status_iterations,
            dim=path.size.num_worlds,
            inputs=[
                problem.data.nl,
                problem.data.nc,
                path.data.config,
                False,
                path.data.status,
            ],
            device=path.device,
        )
    _compute_sparse_solution_vectors(path, problem)
