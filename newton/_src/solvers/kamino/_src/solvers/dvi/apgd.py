# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Per-world accelerated projected-gradient solves for Kamino contact phases.

This module owns the APGD iteration and compact contact workspace.  The
operator is supplied as a callback, so the same iteration drives either a
dense represented Delassus block or Kamino's matrix-free sparse contact product.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

import warp as wp

from .apgd_kernels import (
    _ACC_GAMMA_NORM_SQUARED,
    _ACC_OBJ_CANDIDATE,
    _ACC_OBJ_MODEL,
    _ACC_RAYLEIGH_NORM_SQUARED,
    _ACC_RAYLEIGH_PRODUCT_NORM_SQUARED,
    _ACC_RESIDUAL_SQUARED,
    _ACC_RESTART_DOT,
    _GAMMA_NORM_SQUARED,
    _NUM_APGD_ACCUMULATORS,
    _NUM_APGD_FLAGS,
    _NUM_APGD_REDUCTION_BLOCKS,
    _NUM_APGD_SCALARS,
    _OBJ_CANDIDATE,
    _OBJ_MODEL,
    _RAYLEIGH_NORM_SQUARED,
    _RAYLEIGH_PRODUCT_NORM_SQUARED,
    _RESIDUAL_SQUARED,
    _RESTART_DOT,
    ContactAPGDConfigStruct,
    ContactAPGDStatus,
    build_dense_contact_rhs,
    combine_apgd_block_sums,
    compute_apgd_gradient,
    copy_best_to_solution,
    dense_contact_matvec,
    finalize_rayleigh_estimate,
    gather_dense_contact_solution,
    initialize_apgd_vectors,
    initialize_apgd_worlds,
    prepare_apgd_iteration,
    project_apgd_step,
    reduce_apgd_objectives,
    reduce_apgd_partials_to_blocks,
    reduce_restart_and_res4,
    scatter_dense_contact_solution,
    update_apgd_vectors,
    update_apgd_worlds,
    update_backtracking_condition,
    update_outer_condition,
    write_rayleigh_partials,
)

wp.set_module_options({"enable_backward": False})

float32 = wp.float32
int32 = wp.int32


class ContactMatvec(Protocol):
    """Callable contract for a represented compact contact operator."""

    def __call__(
        self,
        x: wp.array[float32],
        y: wp.array[float32],
        world_mask: wp.array[bool],
    ) -> None:
        """Overwrite ``y`` with ``A*x`` for every active world."""


@dataclass(frozen=True, slots=True)
class ContactAPGDOptions:
    """Host-side controls for one world's contact APGD phase."""

    max_iterations: int = 20
    """Maximum APGD outer iterations."""

    max_backtrack_iterations: int = 20
    """Backtracking passes per outer iteration, including the initial check.

    A value of zero disables the descent check. If the last permitted check
    doubles the Lipschitz estimate, the new step is used by the next outer
    iteration, matching the final newton-dvi APGD schedule.
    """

    tolerance: float = 1.0e-3
    """Absolute tolerance on the running minimum Res4 norm."""

    min_iterations: int = 1
    """Minimum completed iterations before tolerance may stop the solve."""

    early_exit: bool = True
    """Whether a converged world stops participating in later iterations."""

    use_graph_conditionals: bool = True
    """Whether supported devices use nested device-driven while loops."""

    def __post_init__(self) -> None:
        """Validate iteration budgets and the stopping tolerance."""
        if not isinstance(self.max_iterations, int) or isinstance(self.max_iterations, bool) or self.max_iterations < 1:
            raise ValueError("`max_iterations` must be a positive integer.")
        if (
            not isinstance(self.max_backtrack_iterations, int)
            or isinstance(self.max_backtrack_iterations, bool)
            or self.max_backtrack_iterations < 0
        ):
            raise ValueError("`max_backtrack_iterations` must be a non-negative integer.")
        if (
            not isinstance(self.min_iterations, int)
            or isinstance(self.min_iterations, bool)
            or not 1 <= self.min_iterations <= self.max_iterations
        ):
            raise ValueError("`min_iterations` must be in [1, max_iterations].")
        if not math.isfinite(self.tolerance) or self.tolerance < 0.0:
            raise ValueError("`tolerance` must be finite and non-negative.")


class DenseContactOperator:
    """Dense represented contact block and right-hand-side adapter.

    ``matrix`` is Kamino's represented Delassus matrix ``D`` and
    ``represented_compliance`` is its represented physical contact diagonal
    ``E_hat``.  Contact preconditioner entries must be identical within each
    ``[t0, t1, n]`` triplet, as they are in Kamino, so the associated Coulomb
    cone remains invariant in represented coordinates.
    """

    def __init__(
        self,
        *,
        problem_dim: wp.array[int32],
        problem_mio: wp.array[int32],
        problem_vio: wp.array[int32],
        problem_nc: wp.array[int32],
        problem_cio: wp.array[int32],
        problem_ccgo: wp.array[int32],
        matrix: wp.array[float32],
        represented_compliance: wp.array[float32],
        free_velocity: wp.array[float32],
        contact_row_offset: wp.array[int32],
        num_worlds: int,
        max_contacts_per_world: int,
        device: wp.DeviceLike = None,
    ):
        """Bind the dense problem arrays without allocating graph-time storage."""
        self.device = wp.get_device(device)
        self.num_worlds = num_worlds
        self.max_contacts_per_world = max_contacts_per_world
        self.max_contact_rows_per_world = 3 * max_contacts_per_world
        self.problem_dim = problem_dim
        self.problem_mio = problem_mio
        self.problem_vio = problem_vio
        self.problem_nc = problem_nc
        self.problem_cio = problem_cio
        self.problem_ccgo = problem_ccgo
        self.matrix = matrix
        self.represented_compliance = represented_compliance
        self.free_velocity = free_velocity
        self.contact_row_offset = contact_row_offset

        if num_worlds < 1:
            raise ValueError("`num_worlds` must be positive.")
        if max_contacts_per_world < 0:
            raise ValueError("`max_contacts_per_world` must be non-negative.")
        for name, array, dtype in (
            ("problem_dim", problem_dim, int32),
            ("problem_mio", problem_mio, int32),
            ("problem_vio", problem_vio, int32),
            ("problem_nc", problem_nc, int32),
            ("problem_cio", problem_cio, int32),
            ("problem_ccgo", problem_ccgo, int32),
            ("matrix", matrix, float32),
            ("represented_compliance", represented_compliance, float32),
            ("free_velocity", free_velocity, float32),
            ("contact_row_offset", contact_row_offset, int32),
        ):
            self._validate_array(name, array, dtype)
        for name, array in (
            ("problem_dim", problem_dim),
            ("problem_mio", problem_mio),
            ("problem_vio", problem_vio),
            ("problem_nc", problem_nc),
            ("problem_cio", problem_cio),
            ("problem_ccgo", problem_ccgo),
        ):
            if array.shape[0] != num_worlds:
                raise ValueError(f"`{name}` must contain one entry per world.")
        if contact_row_offset.shape[0] != num_worlds + 1:
            raise ValueError("`contact_row_offset` must have shape (num_worlds + 1,).")

    def _validate_array(self, name: str, array: wp.array, dtype: type) -> None:
        """Require a one-dimensional array of the expected type and device."""
        if array is None or array.ndim != 1:
            raise ValueError(f"`{name}` must be a one-dimensional Warp array.")
        if array.dtype != dtype:
            raise TypeError(f"`{name}` must have dtype {dtype}.")
        if array.device != self.device:
            raise ValueError(f"`{name}` must be allocated on {self.device}.")

    def matvec(
        self,
        x: wp.array[float32],
        y: wp.array[float32],
        world_mask: wp.array[bool],
    ) -> None:
        """Apply ``(D_CC + E_hat_CC) * x`` in compact contact storage."""
        wp.launch(
            kernel=dense_contact_matvec,
            dim=(self.num_worlds, max(1, self.max_contact_rows_per_world)),
            inputs=[
                self.problem_dim,
                self.problem_mio,
                self.problem_vio,
                self.problem_nc,
                self.problem_ccgo,
                self.contact_row_offset,
                world_mask,
                self.matrix,
                self.represented_compliance,
                x,
                y,
            ],
            device=self.device,
        )

    def build_rhs(
        self,
        full_solution: wp.array[float32],
        rhs: wp.array[float32],
        world_mask: wp.array[bool],
    ) -> None:
        """Build the contact RHS with all current non-contact reactions fixed."""
        wp.launch(
            kernel=build_dense_contact_rhs,
            dim=(self.num_worlds, max(1, self.max_contact_rows_per_world)),
            inputs=[
                self.problem_dim,
                self.problem_mio,
                self.problem_vio,
                self.problem_nc,
                self.problem_ccgo,
                self.contact_row_offset,
                world_mask,
                self.matrix,
                self.free_velocity,
                full_solution,
                rhs,
            ],
            device=self.device,
        )

    def gather(
        self,
        full_solution: wp.array[float32],
        compact_solution: wp.array[float32],
        world_mask: wp.array[bool],
    ) -> None:
        """Gather the current contact reactions as an APGD warm start."""
        wp.launch(
            kernel=gather_dense_contact_solution,
            dim=(self.num_worlds, max(1, self.max_contact_rows_per_world)),
            inputs=[
                self.problem_vio,
                self.problem_nc,
                self.problem_ccgo,
                self.contact_row_offset,
                world_mask,
                full_solution,
                compact_solution,
            ],
            device=self.device,
        )

    def scatter(
        self,
        compact_solution: wp.array[float32],
        full_solution: wp.array[float32],
        world_mask: wp.array[bool],
    ) -> None:
        """Scatter the minimum-Res4 reactions to Kamino's unified vector."""
        wp.launch(
            kernel=scatter_dense_contact_solution,
            dim=(self.num_worlds, max(1, self.max_contact_rows_per_world)),
            inputs=[
                self.problem_vio,
                self.problem_nc,
                self.problem_ccgo,
                self.contact_row_offset,
                world_mask,
                compact_solution,
                full_solution,
            ],
            device=self.device,
        )


class ContactAPGDSolver:
    """Batched contact APGD engine with deterministic per-world reductions."""

    def __init__(
        self,
        contact_capacities: Sequence[int],
        config: ContactAPGDOptions | Sequence[ContactAPGDOptions] | None = None,
        *,
        device: wp.DeviceLike = None,
    ):
        """Allocate all workspace required by eager or captured APGD solves."""
        capacities = tuple(contact_capacities)
        if not capacities:
            raise ValueError("`contact_capacities` must contain at least one world.")
        if any(not isinstance(capacity, int) or isinstance(capacity, bool) or capacity < 0 for capacity in capacities):
            raise ValueError("Every contact capacity must be a non-negative integer.")

        self.device = wp.get_device(device)
        self.num_worlds = len(capacities)
        self.contact_capacities = capacities
        self.max_contacts_per_world = max(capacities)
        self.max_contact_rows_per_world = 3 * self.max_contacts_per_world
        self._num_reduction_blocks = min(_NUM_APGD_REDUCTION_BLOCKS, max(1, self.max_contacts_per_world))

        if config is None:
            configs = (ContactAPGDOptions(),) * self.num_worlds
        elif isinstance(config, ContactAPGDOptions):
            configs = (config,) * self.num_worlds
        else:
            configs = tuple(config)
            if len(configs) != self.num_worlds:
                raise ValueError("`config` must contain one ContactAPGDOptions per world.")
            if not all(isinstance(item, ContactAPGDOptions) for item in configs):
                raise TypeError("Every config entry must be a ContactAPGDOptions instance.")
        conditional_modes = {item.use_graph_conditionals for item in configs}
        if len(conditional_modes) != 1:
            raise ValueError("All worlds must use the same graph-conditional mode.")

        self.config = configs
        self._use_graph_conditionals = configs[0].use_graph_conditionals
        self._max_iterations = max(item.max_iterations for item in configs)
        self._max_backtrack_iterations = max(item.max_backtrack_iterations for item in configs)

        contact_offsets = [0]
        for capacity in capacities:
            contact_offsets.append(contact_offsets[-1] + capacity)
        contact_row_offsets = [3 * offset for offset in contact_offsets]
        self._contact_storage_size = max(1, contact_row_offsets[-1])
        self._contact_launch_dim = (self.num_worlds, max(1, self.max_contacts_per_world))
        self._row_launch_dim = (self.num_worlds, max(1, self.max_contact_rows_per_world))

        config_structs = []
        for item in configs:
            config_struct = ContactAPGDConfigStruct()
            config_struct.max_iterations = item.max_iterations
            config_struct.max_backtrack_iterations = item.max_backtrack_iterations
            config_struct.tolerance = item.tolerance
            config_struct.min_iterations = item.min_iterations
            config_struct.early_exit = int(item.early_exit)
            config_structs.append(config_struct)

        with wp.ScopedDevice(self.device):
            self.contact_offset = wp.array(contact_offsets, dtype=int32, device=self.device)
            self.contact_row_offset = wp.array(contact_row_offsets, dtype=int32, device=self.device)
            self._config = wp.array(config_structs, dtype=ContactAPGDConfigStruct, device=self.device)
            self.status = wp.zeros(self.num_worlds, dtype=ContactAPGDStatus, device=self.device)
            self.rhs = wp.zeros(self._contact_storage_size, dtype=float32, device=self.device)
            self.solution = wp.zeros(self._contact_storage_size, dtype=float32, device=self.device)
            self._y = wp.zeros_like(self.solution)
            self._gamma = wp.zeros_like(self.solution)
            self._gamma_new = wp.zeros_like(self.solution)
            self._gamma_best = wp.zeros_like(self.solution)
            self._gradient = wp.zeros_like(self.solution)
            self._operator_product = wp.zeros_like(self.solution)
            self._operator_gamma_new = wp.zeros_like(self.solution)
            self._rayleigh_vector = wp.zeros_like(self.solution)
            self._rayleigh_product = wp.zeros_like(self.solution)
            self._scalars = wp.zeros((self.num_worlds, _NUM_APGD_SCALARS), dtype=float32, device=self.device)
            self._flags = wp.zeros((self.num_worlds, _NUM_APGD_FLAGS), dtype=int32, device=self.device)
            self._partials = wp.zeros(
                (_NUM_APGD_ACCUMULATORS, max(1, contact_offsets[-1])), dtype=float32, device=self.device
            )
            self._block_sums = wp.zeros(
                (2 * self.num_worlds, self._num_reduction_blocks), dtype=float32, device=self.device
            )
            self._outer_mask = wp.zeros(self.num_worlds, dtype=wp.bool, device=self.device)
            self._backtrack_mask = wp.zeros(self.num_worlds, dtype=wp.bool, device=self.device)
            self._all_worlds_mask = wp.ones(self.num_worlds, dtype=wp.bool, device=self.device)
            self._outer_continue = wp.zeros(1, dtype=int32, device=self.device)
            self._backtrack_continue = wp.zeros(1, dtype=int32, device=self.device)

    def make_dense_operator(
        self,
        *,
        problem_dim: wp.array[int32],
        problem_mio: wp.array[int32],
        problem_vio: wp.array[int32],
        problem_nc: wp.array[int32],
        problem_cio: wp.array[int32],
        problem_ccgo: wp.array[int32],
        matrix: wp.array[float32],
        represented_compliance: wp.array[float32],
        free_velocity: wp.array[float32],
    ) -> DenseContactOperator:
        """Create a dense adapter sharing this solver's compact row layout."""
        return DenseContactOperator(
            problem_dim=problem_dim,
            problem_mio=problem_mio,
            problem_vio=problem_vio,
            problem_nc=problem_nc,
            problem_cio=problem_cio,
            problem_ccgo=problem_ccgo,
            matrix=matrix,
            represented_compliance=represented_compliance,
            free_velocity=free_velocity,
            contact_row_offset=self.contact_row_offset,
            num_worlds=self.num_worlds,
            max_contacts_per_world=self.max_contacts_per_world,
            device=self.device,
        )

    def _validate_vector(
        self,
        name: str,
        array: wp.array,
        dtype: type,
        minimum_size: int,
    ) -> None:
        """Require a compatible one-dimensional caller-owned array."""
        if array is None or array.ndim != 1:
            raise ValueError(f"`{name}` must be a one-dimensional Warp array.")
        if array.dtype != dtype:
            raise TypeError(f"`{name}` must have dtype {dtype}.")
        if array.device != self.device:
            raise ValueError(f"`{name}` must be allocated on {self.device}.")
        if array.shape[0] < minimum_size:
            raise ValueError(f"`{name}` must contain at least {minimum_size} entries.")

    def _reduce_partials(
        self,
        contact_count: wp.array[int32],
        world_mask: wp.array[bool],
        partial_slot_a: int,
        partial_slot_b: int,
        scalar_slot_a: int,
        scalar_slot_b: int,
    ) -> None:
        """Reduce per-contact partials in a deterministic fixed order."""
        wp.launch(
            kernel=reduce_apgd_partials_to_blocks,
            dim=(self.num_worlds, self._num_reduction_blocks),
            inputs=[
                contact_count,
                self.contact_offset,
                world_mask,
                self._num_reduction_blocks,
                partial_slot_a,
                partial_slot_b,
                self._partials,
                self._block_sums,
            ],
            device=self.device,
        )
        wp.launch(
            kernel=combine_apgd_block_sums,
            dim=self.num_worlds,
            inputs=[
                world_mask,
                self._num_reduction_blocks,
                int(partial_slot_b >= 0),
                scalar_slot_a,
                scalar_slot_b,
                self._block_sums,
                self._scalars,
            ],
            device=self.device,
        )

    def solve(
        self,
        contact_count: wp.array[int32],
        friction: wp.array[float32],
        rhs: wp.array[float32],
        solution: wp.array[float32],
        matvec: ContactMatvec,
        *,
        phase_mask: wp.array[bool] | None = None,
        contact_offset: wp.array[int32] | None = None,
    ) -> wp.array[ContactAPGDStatus]:
        """Solve compact associated contact QPs through a caller-owned operator.

        The contact rows are ``[t0, t1, n]`` and each world solves
        ``min 0.5*x' A x - b'x`` subject to ``x`` belonging to the product of
        its Coulomb cones.  ``matvec`` must overwrite active output rows and
        honor the supplied per-world mask.  All arrays and callback storage
        must already exist when this method is CUDA-graph captured.
        """
        self._validate_vector("contact_count", contact_count, int32, self.num_worlds)
        self._validate_vector("friction", friction, float32, sum(self.contact_capacities))
        self._validate_vector("rhs", rhs, float32, self._contact_storage_size)
        self._validate_vector("solution", solution, float32, self._contact_storage_size)
        if phase_mask is None:
            phase_mask = self._all_worlds_mask
        self._validate_vector("phase_mask", phase_mask, wp.bool, self.num_worlds)
        if contact_offset is None:
            contact_offset = self.contact_offset
        self._validate_vector("contact_offset", contact_offset, int32, self.num_worlds)

        self._outer_continue.zero_()
        wp.launch(
            kernel=initialize_apgd_worlds,
            dim=self.num_worlds,
            inputs=[
                contact_count,
                self.contact_row_offset,
                phase_mask,
                self._config,
                self._scalars,
                self._flags,
                self._outer_mask,
                self._backtrack_mask,
                self._outer_continue,
                self.status,
            ],
            device=self.device,
        )
        wp.launch(
            kernel=initialize_apgd_vectors,
            dim=self._contact_launch_dim,
            inputs=[
                contact_count,
                self.contact_row_offset,
                phase_mask,
                solution,
                self._y,
                self._gamma,
                self._gamma_new,
                self._gamma_best,
                self._rayleigh_vector,
            ],
            device=self.device,
        )
        matvec(self._rayleigh_vector, self._rayleigh_product, self._outer_mask)
        wp.launch(
            kernel=write_rayleigh_partials,
            dim=self._contact_launch_dim,
            inputs=[
                contact_count,
                self.contact_row_offset,
                self.contact_offset,
                self._outer_mask,
                self._rayleigh_vector,
                self._rayleigh_product,
                self._partials,
            ],
            device=self.device,
        )
        self._reduce_partials(
            contact_count,
            self._outer_mask,
            _ACC_RAYLEIGH_NORM_SQUARED,
            _ACC_RAYLEIGH_PRODUCT_NORM_SQUARED,
            _RAYLEIGH_NORM_SQUARED,
            _RAYLEIGH_PRODUCT_NORM_SQUARED,
        )
        wp.launch(
            kernel=finalize_rayleigh_estimate,
            dim=self.num_worlds,
            inputs=[self._outer_mask, self._scalars],
            device=self.device,
        )

        use_conditionals = self._use_graph_conditionals and (self.device.is_cpu or wp.is_conditional_graph_supported())

        def backtrack_body() -> None:
            """Execute one masked re-projection and descent-bound check."""
            wp.launch(
                kernel=project_apgd_step,
                dim=self._contact_launch_dim,
                inputs=[
                    contact_count,
                    self.contact_row_offset,
                    contact_offset,
                    self._backtrack_mask,
                    friction,
                    self._y,
                    self._gradient,
                    self._scalars,
                    self._gamma_new,
                ],
                device=self.device,
            )
            matvec(self._gamma_new, self._operator_gamma_new, self._backtrack_mask)
            wp.launch(
                kernel=reduce_apgd_objectives,
                dim=self._contact_launch_dim,
                inputs=[
                    contact_count,
                    self.contact_row_offset,
                    self.contact_offset,
                    self._backtrack_mask,
                    rhs,
                    self._y,
                    self._gradient,
                    self._gamma_new,
                    self._operator_gamma_new,
                    self._scalars,
                    self._partials,
                ],
                device=self.device,
            )
            self._reduce_partials(
                contact_count,
                self._backtrack_mask,
                _ACC_OBJ_CANDIDATE,
                _ACC_OBJ_MODEL,
                _OBJ_CANDIDATE,
                _OBJ_MODEL,
            )
            self._backtrack_continue.zero_()
            wp.launch(
                kernel=update_backtracking_condition,
                dim=self.num_worlds,
                inputs=[
                    self._backtrack_mask,
                    self._config,
                    self._scalars,
                    self._flags,
                    self._backtrack_mask,
                    self._backtrack_continue,
                    self.status,
                ],
                device=self.device,
            )

        def outer_body() -> None:
            """Execute one APGD iteration for every currently active world."""
            wp.launch(
                kernel=prepare_apgd_iteration,
                dim=self.num_worlds,
                inputs=[self._outer_mask, self._scalars, self._flags, self._backtrack_mask],
                device=self.device,
            )
            matvec(self._y, self._operator_product, self._outer_mask)
            wp.launch(
                kernel=compute_apgd_gradient,
                dim=self._row_launch_dim,
                inputs=[
                    contact_count,
                    self.contact_row_offset,
                    self._outer_mask,
                    self._operator_product,
                    rhs,
                    self._gradient,
                ],
                device=self.device,
            )
            wp.launch(
                kernel=project_apgd_step,
                dim=self._contact_launch_dim,
                inputs=[
                    contact_count,
                    self.contact_row_offset,
                    contact_offset,
                    self._outer_mask,
                    friction,
                    self._y,
                    self._gradient,
                    self._scalars,
                    self._gamma_new,
                ],
                device=self.device,
            )
            matvec(self._gamma_new, self._operator_gamma_new, self._outer_mask)
            wp.launch(
                kernel=reduce_apgd_objectives,
                dim=self._contact_launch_dim,
                inputs=[
                    contact_count,
                    self.contact_row_offset,
                    self.contact_offset,
                    self._outer_mask,
                    rhs,
                    self._y,
                    self._gradient,
                    self._gamma_new,
                    self._operator_gamma_new,
                    self._scalars,
                    self._partials,
                ],
                device=self.device,
            )
            self._reduce_partials(
                contact_count,
                self._outer_mask,
                _ACC_OBJ_CANDIDATE,
                _ACC_OBJ_MODEL,
                _OBJ_CANDIDATE,
                _OBJ_MODEL,
            )
            self._backtrack_continue.zero_()
            wp.launch(
                kernel=update_backtracking_condition,
                dim=self.num_worlds,
                inputs=[
                    self._outer_mask,
                    self._config,
                    self._scalars,
                    self._flags,
                    self._backtrack_mask,
                    self._backtrack_continue,
                    self.status,
                ],
                device=self.device,
            )
            if use_conditionals and self._max_backtrack_iterations > 1:
                wp.capture_while(self._backtrack_continue, while_body=backtrack_body)
            else:
                for _ in range(max(0, self._max_backtrack_iterations - 1)):
                    backtrack_body()

            wp.launch(
                kernel=reduce_restart_and_res4,
                dim=self._contact_launch_dim,
                inputs=[
                    contact_count,
                    self.contact_row_offset,
                    self.contact_offset,
                    contact_offset,
                    self._outer_mask,
                    friction,
                    rhs,
                    self._gamma,
                    self._gamma_new,
                    self._gradient,
                    self._operator_gamma_new,
                    self._scalars,
                    self._partials,
                ],
                device=self.device,
            )
            self._reduce_partials(
                contact_count,
                self._outer_mask,
                _ACC_RESIDUAL_SQUARED,
                _ACC_RESTART_DOT,
                _RESIDUAL_SQUARED,
                _RESTART_DOT,
            )
            self._reduce_partials(
                contact_count,
                self._outer_mask,
                _ACC_GAMMA_NORM_SQUARED,
                -1,
                _GAMMA_NORM_SQUARED,
                -1,
            )
            wp.launch(
                kernel=update_apgd_worlds,
                dim=self.num_worlds,
                inputs=[self._outer_mask, self._config, self._scalars, self._flags, self.status],
                device=self.device,
            )
            wp.launch(
                kernel=update_apgd_vectors,
                dim=self._row_launch_dim,
                inputs=[
                    contact_count,
                    self.contact_row_offset,
                    self._outer_mask,
                    self._scalars,
                    self._flags,
                    self._gamma_new,
                    self._y,
                    self._gamma,
                    self._gamma_best,
                ],
                device=self.device,
            )
            self._outer_continue.zero_()
            wp.launch(
                kernel=update_outer_condition,
                dim=self.num_worlds,
                inputs=[phase_mask, self._config, self._outer_mask, self._outer_continue, self.status],
                device=self.device,
            )

        if use_conditionals:
            wp.capture_while(self._outer_continue, while_body=outer_body)
        else:
            for _ in range(self._max_iterations):
                outer_body()

        wp.launch(
            kernel=copy_best_to_solution,
            dim=self._row_launch_dim,
            inputs=[contact_count, self.contact_row_offset, phase_mask, self._gamma_best, solution],
            device=self.device,
        )
        return self.status

    def solve_dense(
        self,
        operator: DenseContactOperator,
        friction: wp.array[float32],
        full_solution: wp.array[float32],
        *,
        phase_mask: wp.array[bool] | None = None,
    ) -> wp.array[ContactAPGDStatus]:
        """Build and solve one dense contact phase in unified Kamino storage."""
        if operator.device != self.device or operator.num_worlds != self.num_worlds:
            raise ValueError("The dense operator must use this APGD solver's device and world count.")
        if operator.contact_row_offset is not self.contact_row_offset:
            raise ValueError("The dense operator must share this APGD solver's compact row layout.")
        self._validate_vector("full_solution", full_solution, float32, 1)
        if phase_mask is None:
            phase_mask = self._all_worlds_mask
        self._validate_vector("phase_mask", phase_mask, wp.bool, self.num_worlds)
        operator.build_rhs(full_solution, self.rhs, phase_mask)
        operator.gather(full_solution, self.solution, phase_mask)
        status = self.solve(
            operator.problem_nc,
            friction,
            self.rhs,
            self.solution,
            operator.matvec,
            phase_mask=phase_mask,
            contact_offset=operator.problem_cio,
        )
        operator.scatter(self.solution, full_solution, phase_mask)
        return status
