# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Sparse represented contact operator for Kamino APGD."""

from __future__ import annotations

import warp as wp

from ...core.types import vec6f
from .apgd_kernels import gather_dense_contact_solution, scatter_dense_contact_solution
from .apgd_sparse_kernels import (
    build_sparse_body_adjacency,
    build_sparse_contact_rhs,
    sparse_contact_matvec,
    sparse_contact_transpose_mass,
    sparse_noncontact_transpose_mass,
)

wp.set_module_options({"enable_backward": False})

float32 = wp.float32
int32 = wp.int32
mat33f = wp.mat33f


class SparseContactOperator:
    """Apply a sparse represented Delassus contact block without atomics.

    The operator consumes Kamino's raw block-sparse Jacobian and the existing
    dynamic-contact map. It computes ``P J M^-1 J^T P + R + E_hat`` directly,
    preserving the compact APGD contact layout ``[t0, t1, n]``. All workspace
    is allocated during construction. Call :meth:`prepare` after sparse
    topology changes; it builds deterministic per-body traversals used by
    :meth:`matvec` and :meth:`build_rhs`. Every operation is safe to CUDA-graph
    capture.
    """

    def __init__(
        self,
        *,
        problem_dim: wp.array[int32],
        problem_vio: wp.array[int32],
        problem_nc: wp.array[int32],
        problem_cio: wp.array[int32],
        problem_ccgo: wp.array[int32],
        contact_indices: wp.array[int32],
        contact_nzb_offsets: wp.array[int32],
        jacobian_num_nzb: wp.array[int32],
        jacobian_nzb_start: wp.array[int32],
        jacobian_nzb_coords: wp.array2d[int32],
        jacobian_nzb_values: wp.array[vec6f],
        jacobian_row_start: wp.array[int32],
        jacobian_col_start: wp.array[int32],
        body_offset: wp.array[int32],
        body_inv_mass: wp.array[float32],
        body_inv_inertia: wp.array[mat33f],
        preconditioner: wp.array[float32],
        regularization: wp.array[float32],
        represented_compliance: wp.array[float32],
        free_velocity: wp.array[float32],
        contact_row_offset: wp.array[int32],
        num_worlds: int,
        max_contacts_per_world: int,
        max_bodies_per_world: int,
        total_body_dofs: int,
        device: wp.DeviceLike = None,
    ):
        """Bind sparse problem storage and allocate one body-space vector."""
        self.device = wp.get_device(device)
        self.num_worlds = num_worlds
        self.max_contacts_per_world = max_contacts_per_world
        self.max_contact_rows_per_world = 3 * max_contacts_per_world
        self.max_bodies_per_world = max_bodies_per_world
        self.problem_dim = problem_dim
        self.problem_vio = problem_vio
        self.problem_nc = problem_nc
        self.problem_cio = problem_cio
        self.problem_ccgo = problem_ccgo
        self.contact_indices = contact_indices
        self.contact_nzb_offsets = contact_nzb_offsets
        self.jacobian_num_nzb = jacobian_num_nzb
        self.jacobian_nzb_start = jacobian_nzb_start
        self.jacobian_nzb_coords = jacobian_nzb_coords
        self.jacobian_nzb_values = jacobian_nzb_values
        self.jacobian_row_start = jacobian_row_start
        self.jacobian_col_start = jacobian_col_start
        self.body_offset = body_offset
        self.body_inv_mass = body_inv_mass
        self.body_inv_inertia = body_inv_inertia
        self.preconditioner = preconditioner
        self.regularization = regularization
        self.represented_compliance = represented_compliance
        self.free_velocity = free_velocity
        self.contact_row_offset = contact_row_offset

        for name, value in (
            ("num_worlds", num_worlds),
            ("max_contacts_per_world", max_contacts_per_world),
            ("max_bodies_per_world", max_bodies_per_world),
            ("total_body_dofs", total_body_dofs),
        ):
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError(f"`{name}` must be a non-negative integer.")
        if num_worlds < 1:
            raise ValueError("`num_worlds` must be positive.")
        if total_body_dofs % 6 != 0:
            raise ValueError("`total_body_dofs` must be divisible by six.")

        for name, array, dtype in (
            ("problem_dim", problem_dim, int32),
            ("problem_vio", problem_vio, int32),
            ("problem_nc", problem_nc, int32),
            ("problem_cio", problem_cio, int32),
            ("problem_ccgo", problem_ccgo, int32),
            ("contact_indices", contact_indices, int32),
            ("contact_nzb_offsets", contact_nzb_offsets, int32),
            ("jacobian_num_nzb", jacobian_num_nzb, int32),
            ("jacobian_nzb_start", jacobian_nzb_start, int32),
            ("jacobian_nzb_values", jacobian_nzb_values, vec6f),
            ("jacobian_row_start", jacobian_row_start, int32),
            ("jacobian_col_start", jacobian_col_start, int32),
            ("body_offset", body_offset, int32),
            ("body_inv_mass", body_inv_mass, float32),
            ("body_inv_inertia", body_inv_inertia, mat33f),
            ("preconditioner", preconditioner, float32),
            ("regularization", regularization, float32),
            ("represented_compliance", represented_compliance, float32),
            ("free_velocity", free_velocity, float32),
            ("contact_row_offset", contact_row_offset, int32),
        ):
            self._validate_vector(name, array, dtype)
        if jacobian_nzb_coords is None or jacobian_nzb_coords.ndim != 2:
            raise ValueError("`jacobian_nzb_coords` must be a two-dimensional Warp array.")
        if jacobian_nzb_coords.dtype != int32:
            raise TypeError("`jacobian_nzb_coords` must have dtype wp.int32.")
        if jacobian_nzb_coords.device != self.device:
            raise ValueError(f"`jacobian_nzb_coords` must be allocated on {self.device}.")

        for name, array in (
            ("problem_dim", problem_dim),
            ("problem_vio", problem_vio),
            ("problem_nc", problem_nc),
            ("problem_cio", problem_cio),
            ("problem_ccgo", problem_ccgo),
            ("jacobian_num_nzb", jacobian_num_nzb),
            ("jacobian_nzb_start", jacobian_nzb_start),
            ("jacobian_row_start", jacobian_row_start),
            ("jacobian_col_start", jacobian_col_start),
        ):
            if array.shape[0] != num_worlds:
                raise ValueError(f"`{name}` must contain one entry per world.")
        if body_offset.shape[0] != num_worlds + 1:
            raise ValueError("`body_offset` must have shape (num_worlds + 1,).")
        if contact_row_offset.shape[0] != num_worlds + 1:
            raise ValueError("`contact_row_offset` must have shape (num_worlds + 1,).")

        with wp.ScopedDevice(self.device):
            self.body_space = wp.zeros(max(1, total_body_dofs), dtype=float32)
            body_entry_count = max(1, num_worlds * max_bodies_per_world)
            body_offset_count = max(1, num_worlds * (max_bodies_per_world + 1))
            contact_slot_count = max(1, 2 * num_worlds * max_contacts_per_world)
            self.body_contact_count = wp.zeros(body_entry_count, dtype=int32)
            self.body_block_count = wp.zeros(body_entry_count, dtype=int32)
            self.body_contact_offsets = wp.zeros(body_offset_count, dtype=int32)
            self.body_contact_slots = wp.empty(contact_slot_count, dtype=wp.vec2i)
            self.body_block_offsets = wp.zeros(body_offset_count, dtype=int32)
            self.body_block_slots = wp.empty(max(1, jacobian_nzb_values.shape[0]), dtype=int32)
            self.body_block_scratch = wp.empty_like(self.body_block_slots)

        self._body_launch_dim = (num_worlds, max(1, max_bodies_per_world))
        self._row_launch_dim = (num_worlds, max(1, self.max_contact_rows_per_world))

    def prepare(self) -> None:
        """Build deterministic per-body traversals for the current topology."""
        wp.launch(
            kernel=build_sparse_body_adjacency,
            dim=self.num_worlds,
            inputs=[
                self.problem_dim,
                self.problem_nc,
                self.problem_cio,
                self.problem_ccgo,
                self.contact_indices,
                self.contact_nzb_offsets,
                self.jacobian_num_nzb,
                self.jacobian_nzb_start,
                self.jacobian_nzb_coords,
                self.body_offset,
                self.max_contacts_per_world,
                self.max_bodies_per_world,
                self.body_contact_count,
                self.body_block_count,
                self.body_contact_offsets,
                self.body_contact_slots,
                self.body_block_offsets,
                self.body_block_slots,
                self.body_block_scratch,
            ],
            device=self.device,
        )

    def _validate_vector(self, name: str, array: wp.array, dtype: type) -> None:
        """Require a one-dimensional array of the expected type and device."""
        if array is None or array.ndim != 1:
            raise ValueError(f"`{name}` must be a one-dimensional Warp array.")
        # Kamino's block-sparse storage creates a private vector class for each
        # block shape.  Warp treats that generated six-float class as compatible
        # with ``vec6f`` at kernel boundaries, even though the Python class
        # identities differ.
        compatible_vector = (
            wp.types.type_is_vector(array.dtype)
            and wp.types.type_is_vector(dtype)
            and array.dtype._shape_ == dtype._shape_
            and array.dtype._wp_scalar_type_ == dtype._wp_scalar_type_
        )
        if array.dtype != dtype and not compatible_vector:
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
            kernel=sparse_contact_transpose_mass,
            dim=self._body_launch_dim,
            inputs=[
                self.problem_ccgo,
                self.problem_vio,
                self.contact_row_offset,
                self.body_contact_offsets,
                self.body_contact_slots,
                self.jacobian_nzb_coords,
                self.jacobian_nzb_values,
                self.jacobian_col_start,
                self.body_offset,
                self.body_inv_mass,
                self.body_inv_inertia,
                self.preconditioner,
                self.max_bodies_per_world,
                x,
                world_mask,
                self.body_space,
            ],
            device=self.device,
        )
        wp.launch(
            kernel=sparse_contact_matvec,
            dim=self._row_launch_dim,
            inputs=[
                self.problem_nc,
                self.problem_cio,
                self.problem_ccgo,
                self.problem_vio,
                self.contact_row_offset,
                self.contact_indices,
                self.contact_nzb_offsets,
                self.jacobian_num_nzb,
                self.jacobian_nzb_start,
                self.jacobian_nzb_coords,
                self.jacobian_nzb_values,
                self.jacobian_row_start,
                self.jacobian_col_start,
                self.preconditioner,
                self.regularization,
                self.represented_compliance,
                self.body_space,
                x,
                world_mask,
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
            kernel=sparse_noncontact_transpose_mass,
            dim=self._body_launch_dim,
            inputs=[
                self.problem_dim,
                self.problem_nc,
                self.problem_ccgo,
                self.problem_vio,
                self.body_block_offsets,
                self.body_block_slots,
                self.jacobian_nzb_coords,
                self.jacobian_nzb_values,
                self.jacobian_col_start,
                self.body_offset,
                self.body_inv_mass,
                self.body_inv_inertia,
                self.preconditioner,
                self.max_bodies_per_world,
                full_solution,
                world_mask,
                self.body_space,
            ],
            device=self.device,
        )
        wp.launch(
            kernel=build_sparse_contact_rhs,
            dim=self._row_launch_dim,
            inputs=[
                self.problem_nc,
                self.problem_cio,
                self.problem_ccgo,
                self.problem_vio,
                self.contact_row_offset,
                self.contact_indices,
                self.contact_nzb_offsets,
                self.jacobian_num_nzb,
                self.jacobian_nzb_start,
                self.jacobian_nzb_coords,
                self.jacobian_nzb_values,
                self.jacobian_col_start,
                self.preconditioner,
                self.free_velocity,
                self.body_space,
                world_mask,
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
            dim=self._row_launch_dim,
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
        """Scatter solved compact contacts into Kamino's unified vector."""
        wp.launch(
            kernel=scatter_dense_contact_solution,
            dim=self._row_launch_dim,
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
