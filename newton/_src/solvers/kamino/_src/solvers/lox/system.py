# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Block-diagonal primal body systems for the LOX backend.

This module owns only solver-internal, contact-free body-space assembly. Each
factor block is one connected dynamic-body component and each dynamic body
contributes six linear-first velocity unknowns. Prescribed bodies retain their
global packed body indices but map to ``-1`` in the matrix layout.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np
import warp as wp

from ...core.types import vec6f
from ...linalg import DenseLinearOperatorData
from .llt import LLTBlockedRCMTraversalSolver
from .system_kernels import (
    _assemble_body_inertial_systems,
    _assemble_joint_rows,
    _assemble_rod_geometric_stiffness,
    _build_candidate_right_hand_side,
    _invert_single_body_blocks,
    _solve_single_body_blocks,
)
from .types import (
    DenseBlockInfo,
    DynamicRows,
    EffortRows,
    LOXSystemData,
    StructuralRows,
    capacity_offsets,
    segment_local_indices,
)

###
# Module interface
###

__all__ = [
    "LOXSystem",
]

###
# Module configs
###

wp.set_module_options({"enable_backward": False})

_WIDE_TILE_MAX_BLOCKS_PER_SM = 4
"""Multi-body blocks per multiprocessor below which the sparse LLT factorizes and solves with wide tiles."""


###
# Interfaces
###


class LOXSystem:
    """Batched dense smooth body systems over independent body components.

    World-level bookkeeping remains in model body order. The dense matrices and
    vectors are packed by ``body_components`` so disconnected articulations can
    be factorized independently. Once the factorization has ordered the rows of
    each block, the matrices are assembled directly in that order. When every
    body is prescribed, the system holds one empty placeholder block.
    The arrays live in :class:`LOXSystemData`, exposed as :attr:`data`.
    """

    def __init__(
        self,
        body_counts: Sequence[int],
        body_world: wp.array[wp.int32],
        body_components: Sequence[Sequence[int]],
        body_edges: Sequence[tuple[int, int]],
        device: wp.DeviceLike,
    ):
        self.device = wp.get_device(device)
        self.body_world = body_world
        """World of each body, from the model."""
        self.num_worlds = len(body_counts)
        self.num_bodies = sum(body_counts)
        body_world_host = np.repeat(np.arange(self.num_worlds, dtype=np.int32), np.asarray(body_counts, dtype=np.int32))

        # Pack each joint-connected component of dynamic bodies into one dense factor block. The
        # multi-body blocks come first, so that the sparse factorization holds a prefix of the blocks.
        components = [np.asarray(component, dtype=np.int32) for component in body_components]
        components.sort(key=lambda component: component.size == 1)
        if components:
            block_body_counts = np.asarray([component.size for component in components], dtype=np.int32)
            block_offsets = capacity_offsets(block_body_counts)
            packed_bodies = np.concatenate(components)
            self.block_world_host = tuple(body_world_host[packed_bodies[block_offsets[:-1]]].tolist())
            storage_dimensions = (6 * block_body_counts).tolist()
            active_dimensions = storage_dimensions
        else:
            # Dense multi-linear storage requires one positive allocation size,
            # while the active dimension remains zero.
            block_body_counts = np.zeros(1, dtype=np.int32)
            block_offsets = np.zeros(2, dtype=np.int32)
            packed_bodies = np.empty(0, dtype=np.int32)
            self.block_world_host = (0,)
            storage_dimensions = [1]
            active_dimensions = [0]
        self.dynamic_bodies = tuple(sorted(packed_bodies.tolist()))
        """Bodies of the smooth system, in model order."""
        self.block_body_counts = tuple(block_body_counts.tolist())
        """Number of bodies of each factor block."""
        self.num_blocks = len(self.block_body_counts)
        # Single-body blocks keep a direct 6 x 6 inverse; the sparse LLT factorizes the blocks of several bodies
        self.num_sparse_blocks = sum(count > 1 for count in self.block_body_counts)

        body_block = np.full(self.num_bodies, -1, dtype=np.int32)
        body_local = np.full(self.num_bodies, -1, dtype=np.int32)
        body_block[packed_bodies] = np.repeat(np.arange(self.num_blocks, dtype=np.int32), block_body_counts)
        body_local[packed_bodies] = segment_local_indices(block_body_counts, block_offsets)

        edges = np.asarray(body_edges, dtype=np.int32).reshape((-1, 2))
        edges.sort(axis=1)
        edges = np.unique(edges, axis=0)
        self._body_edge_keys_host = edges[:, 0].astype(np.int64) * self.num_bodies + edges[:, 1]

        body_vector_index = np.where(body_block >= 0, 6 * (block_offsets[body_block.clip(min=0)] + body_local), -1)
        self.body_block_host = tuple(body_block.tolist())
        self.body_local_host = tuple(body_local.tolist())
        self.body_vector_index_host = tuple(body_vector_index.tolist())

        info = DenseBlockInfo()
        info.finalize(dimensions=storage_dimensions, dtype=wp.float32, itype=wp.int32, device=self.device)
        if active_dimensions != storage_dimensions:
            info.dim = wp.array(active_dimensions, dtype=wp.int32, device=self.device)
        self._data = LOXSystemData(body_block, body_local, body_vector_index, info, self.device)

        # The first body of each edge between two dynamic bodies clears the off-diagonal blocks of the edge
        # before an assembly, see :func:`.system_kernels._assemble_body_inertial_systems`
        dynamic_edges = edges[np.all(body_block[edges] >= 0, axis=1) & (edges[:, 0] != edges[:, 1])]
        body_edge_offset = capacity_offsets(np.bincount(dynamic_edges[:, 0], minlength=self.num_bodies))
        self._body_edge_offset = wp.array(body_edge_offset, dtype=wp.int32, device=self.device)
        self._body_edge_neighbor = wp.array(
            dynamic_edges[:, 1] if dynamic_edges.size else [0], dtype=wp.int32, device=self.device
        )
        self.linear_solver: LLTBlockedRCMTraversalSolver | None = None
        """Sparse LLT of the multi-body blocks, or ``None`` when every block holds a single body."""
        if self.num_sparse_blocks > 0:
            # The totals of the sparse prefix cover the single-body blocks, so that captured steps
            # assemble every block into the matrix of the factorization.
            sparse_info = DenseBlockInfo()
            sparse_info.finalize(
                dimensions=storage_dimensions[: self.num_sparse_blocks],
                dtype=wp.float32,
                itype=wp.int32,
                device=self.device,
            )
            if active_dimensions != storage_dimensions:
                sparse_info.dim = wp.array(active_dimensions, dtype=wp.int32, device=self.device)
            sparse_info.total_mat_size = info.total_mat_size
            sparse_info.total_vec_size = info.total_vec_size
            # Batches that fill the device take small tiles, which skip most of the zero
            # blocks of a tree-like body graph, and one warp per world, which matches the
            # work of a single-RHS tile solve. Smaller batches take wide tiles, wide thread
            # blocks and panel-parallel factorization, which halve the sequential tile
            # steps of each block.
            wide_tiles = (
                self.device.is_cuda and self.num_sparse_blocks < _WIDE_TILE_MAX_BLOCKS_PER_SM * self.device.sm_count
            )
            self.linear_solver = LLTBlockedRCMTraversalSolver(
                operator=DenseLinearOperatorData(info=sparse_info, mat=self._data.matrix),
                block_size=32 if wide_tiles else 16,
                solve_block_dim=256 if wide_tiles else 32,
                parallel_factorization=wide_tiles,
                reuse_permutation=True,
                dtype=wp.float32,
                device=self.device,
            )
            # The body blocks of the joints bound the entries that the assembly writes
            pair_block, pair_row, pair_col = self._structural_pairs(edges, body_block, body_local)
            self.linear_solver.configure_sparse_assembly(
                wp.array(pair_block, dtype=wp.int32, device=self.device),
                wp.array(pair_row, dtype=wp.int32, device=self.device),
                wp.array(pair_col, dtype=wp.int32, device=self.device),
            )
        self._matrix = self._data.matrix
        self._inverse_permutation: wp.array[wp.int32] | None = None

    ###
    # Properties
    ###

    @property
    def data(self) -> LOXSystemData:
        """Returns the arrays of the smooth body systems."""
        return self._data

    @property
    def matrix(self) -> wp.array[wp.float32]:
        """Matrix that the assembly writes: :attr:`LOXSystemData.matrix`, or the matrix of the factorization
        during :meth:`assemble_and_factorize`."""
        return self._matrix

    @property
    def inverse_permutation(self) -> wp.array[wp.int32] | None:
        """Row of each packed vector entry in :attr:`matrix`, or ``None`` for the packed order."""
        return self._inverse_permutation

    ###
    # Public API
    ###

    def validate_body_pairs(
        self,
        name: str,
        body_a: wp.array[wp.int32],
        body_b: wp.array[wp.int32],
    ) -> None:
        """Verify fixed topology covers every dynamic off-diagonal assembly pair."""
        a_values = body_a.numpy().astype(np.int32, copy=False)
        b_values = body_b.numpy().astype(np.int32, copy=False)
        if a_values.size != b_values.size:
            raise ValueError(f"{name} endpoint arrays must have identical lengths.")
        body_block = np.asarray(self.body_block_host, dtype=np.int32)
        valid = (a_values >= 0) & (b_values >= 0)
        valid &= body_block[a_values.clip(min=0)] >= 0
        valid &= body_block[b_values.clip(min=0)] >= 0
        pairs = np.column_stack((a_values[valid], b_values[valid]))
        pairs.sort(axis=1)
        pair_keys = pairs[:, 0].astype(np.int64) * self.num_bodies + pairs[:, 1]
        missing = np.unique(pairs[~np.isin(pair_keys, self._body_edge_keys_host)], axis=0)
        if missing.size:
            raise ValueError(f"Fixed body topology does not cover {name} pairs: {list(map(tuple, missing.tolist()))}.")

    def assemble_bodies(
        self,
        mass: wp.array[wp.float32],
        inertia_world: wp.array[wp.mat33f],
        velocity_previous: wp.array[vec6f],
        external_wrench: wp.array[vec6f],
        actuation_wrench: wp.array[vec6f],
        gravity: wp.array[wp.vec3f],
        time_step: wp.array[wp.float32],
    ) -> None:
        """Assemble body inertia and explicit-force terms, and clear the other matrix entries of the assembly.

        The body terms write the whole diagonal block of each body and clear the off-diagonal blocks of the
        body edges, which hold every matrix entry that :meth:`add_joint_rows` adds to. The packed matrix holds
        zeros elsewhere from its allocation, and the factorization copies them into its ordered matrix with
        each new row order.
        """
        wp.launch(
            _assemble_body_inertial_systems,
            dim=(self.num_bodies, 6),
            inputs=[
                self.body_world,
                self._data.body_block,
                self._data.body_local,
                self._data.info.dim,
                self._data.info.mio,
                self._data.info.vio,
                self._body_edge_offset,
                self._body_edge_neighbor,
                mass,
                inertia_world,
                velocity_previous,
                external_wrench,
                actuation_wrench,
                gravity,
                time_step,
                self._inverse_permutation,
            ],
            outputs=[self._matrix, self._data.right_hand_side],
            device=self.device,
        )

    def add_joint_rows(
        self,
        joint_world: wp.array[wp.int32],
        joint_body_a: wp.array[wp.int32],
        joint_body_b: wp.array[wp.int32],
        dynamic_rows: DynamicRows,
        structural_rows: StructuralRows,
        structural_offset: wp.array[wp.int32],
        structural_count: wp.array[wp.int32],
        time_step: wp.array[wp.float32],
        joint_penalty_scale: wp.array[wp.float32] | None,
        joint_max_correction: float,
        joint_translation_dead_zone: float,
        joint_rotation_dead_zone: float,
        prescribed_twist: wp.array[vec6f] | None = None,
    ) -> None:
        """Add the implicit joint-dynamics rows and the frozen augmented penalty terms of the structural rows.

        Each joint adds its rows to the blocks of its two bodies at once, with one thread per unknown of
        the two bodies. :meth:`assemble_bodies` must be called first. The structural rows enter
        the system with a ``joint_penalty_scale``, which also writes their penalties. The multiplier
        terms are added to the candidate right-hand side of each iteration. Each structural row
        corrects the part of its residual beyond ``joint_translation_dead_zone``, or
        ``joint_rotation_dead_zone`` for the rotational rows, by at most ``joint_max_correction``
        in one step.

        Args:
            joint_world: World of each joint.
            joint_body_a: Body A of each joint, or ``-1``.
            joint_body_b: Body B of each joint.
            dynamic_rows: Dynamic rows, with their Jacobians, effective inertias and free velocities.
            structural_rows: Structural rows, with their Jacobians, residuals and effective masses.
            structural_offset: First structural row of each joint.
            structural_count: Number of structural rows of each joint.
            time_step: Time step of each world [s].
            joint_penalty_scale: Dimensionless structural ALM penalty scale of each body, or ``None``
                to leave the structural rows out of the system.
            joint_max_correction: Largest joint residual corrected in one step [m or rad].
            joint_translation_dead_zone: Translational joint residual left uncorrected [m].
            joint_rotation_dead_zone: Rotational joint residual left uncorrected [rad].
            prescribed_twist: Twist of each body outside the system, or ``None`` for zero twists.
        """
        joint_count = joint_world.shape[0]
        if joint_count == 0 or (
            dynamic_rows.count == 0 and (structural_rows.count == 0 or joint_penalty_scale is None)
        ):
            return
        wp.launch(
            _assemble_joint_rows,
            dim=(joint_count, 12),
            inputs=[
                self._data.info.dim,
                self._data.info.mio,
                self._data.info.vio,
                joint_world,
                joint_body_a,
                joint_body_b,
                self._data.body_block,
                self._data.body_local,
                prescribed_twist,
                time_step,
                dynamic_rows.joint_offset,
                dynamic_rows.joint_count,
                dynamic_rows.jacobian_a,
                dynamic_rows.jacobian_b,
                dynamic_rows.effective_inertia,
                dynamic_rows.free_velocity,
                structural_offset,
                structural_count,
                structural_rows.jacobian_a,
                structural_rows.jacobian_b,
                structural_rows.residual,
                structural_rows.effective_mass,
                structural_rows.reaction,
                joint_penalty_scale,
                joint_max_correction,
                joint_translation_dead_zone,
                joint_rotation_dead_zone,
                self._inverse_permutation,
            ],
            outputs=[structural_rows.penalty, structural_rows.body_impulse, self._matrix, self._data.right_hand_side],
            device=self.device,
        )

    def add_rod_geometric_stiffness(
        self,
        joint_world: wp.array[wp.int32],
        joint_dof_type: wp.array[wp.int32],
        joint_dofs_offset: wp.array[wp.int32],
        joint_coords_offset: wp.array[wp.int32],
        joint_body_a: wp.array[wp.int32],
        joint_body_b: wp.array[wp.int32],
        joint_q: wp.array[wp.float32],
        joint_q_ref: wp.array[wp.float32],
        joint_pose: wp.array[wp.transformf],
        body_pose: wp.array[wp.transformf],
        dynamic_rows: DynamicRows,
        time_step: wp.array[wp.float32],
        prescribed_twist: wp.array[vec6f] | None = None,
    ) -> None:
        """Add the rotational geometric stiffness of the stretch and shear rows of the stretched rod joints.

        :meth:`add_joint_rows` adds the Gauss-Newton term ``effective_inertia J^T J`` of each row. The
        elastic force of the stretch and shear rows of a rod also turns with the arm from each body's center
        of mass to the joint point. For each rod longer than its rest length, this adds the stiffness of the
        tension of that force to the angular block of the body, see
        :func:`.system_kernels._assemble_rod_geometric_stiffness`.

        Args:
            joint_world: World of each joint.
            joint_dof_type: DoF type of each joint.
            joint_dofs_offset: First DoF of each joint.
            joint_coords_offset: First coordinate of each joint.
            joint_body_a: Body A of each joint, or ``-1``.
            joint_body_b: Body B of each joint.
            joint_q: Joint coordinates.
            joint_q_ref: Reference joint coordinates, holding the rest strain of the rods.
            joint_pose: World pose of each joint frame.
            body_pose: World pose of each body's center of mass.
            dynamic_rows: Dynamic rows, with their Jacobians, effective inertias and free velocities.
            time_step: Time step of each world [s].
            prescribed_twist: Twist of each body outside the system, or ``None`` for zero twists.
        """
        joint_count = joint_world.shape[0]
        if joint_count == 0 or dynamic_rows.count == 0:
            return
        wp.launch(
            _assemble_rod_geometric_stiffness,
            dim=(joint_count, 2),
            inputs=[
                self._data.info.dim,
                self._data.info.mio,
                self._data.info.vio,
                joint_world,
                joint_dof_type,
                joint_dofs_offset,
                joint_coords_offset,
                joint_body_a,
                joint_body_b,
                joint_q,
                joint_q_ref,
                joint_pose,
                body_pose,
                self._data.body_block,
                self._data.body_local,
                prescribed_twist,
                time_step,
                dynamic_rows.joint_offset,
                dynamic_rows.joint_count,
                dynamic_rows.dof_index,
                dynamic_rows.jacobian_a,
                dynamic_rows.jacobian_b,
                dynamic_rows.effective_inertia,
                dynamic_rows.free_velocity,
                self._inverse_permutation,
            ],
            outputs=[self._matrix],
            device=self.device,
        )

    def factorize(self) -> None:
        """Invert the single-body blocks of the current matrices and factorize the others with the sparse LLT."""
        if self.num_sparse_blocks < self.num_blocks:
            wp.launch(
                _invert_single_body_blocks,
                dim=self.num_bodies,
                inputs=[
                    self._data.body_block,
                    self._data.body_local,
                    self._data.info.dim,
                    self._data.info.mio,
                    self._data.matrix,
                ],
                outputs=[self._data.single_body_inverse],
                device=self.device,
            )
        if self.linear_solver is not None:
            self.linear_solver.compute(self._data.matrix)

    def assemble_and_factorize(self, assemble: Callable[[], None]) -> None:
        """Assemble the weighted matrices with ``assemble`` and factorize them with the RCM-ordered semi-sparse LLT.

        ``assemble`` writes and reads the matrix through :attr:`matrix` and :attr:`inverse_permutation`,
        and inverts the single-body blocks with the patch weights. Once the factorization
        has ordered the rows, captured steps assemble directly in that order.
        """
        if self.linear_solver is None:
            assemble()
            return

        def assemble_into(matrix: wp.array[wp.float32], inverse_permutation: wp.array[wp.int32] | None) -> None:
            self._matrix = matrix
            self._inverse_permutation = inverse_permutation
            try:
                assemble()
            finally:
                self._matrix = self._data.matrix
                self._inverse_permutation = None

        self.linear_solver.compute_sparse(assemble_into)

    def build_candidate_right_hand_side(
        self,
        projected_twist: wp.array[vec6f],
        splitting_dual: wp.array[vec6f],
        dynamic_rows: DynamicRows,
        effort_rows: EffortRows,
        structural_rows: StructuralRows,
        anchor_scale: float = 1.0,
    ) -> None:
        """Build the right-hand side of the candidate solve of one iteration.

        Adds the splitting target ``W (s p + u)`` and the current row impulses to the frozen
        right-hand side ``f``: the structural multiplier impulses ``h J_s^T lambda`` and the
        effort-limit counter-impulses ``J_d^T c``. The single-body blocks also write their solution.

        Args:
            projected_twist: Anchor twist ``p`` of each body, or ``None`` for no anchor.
            splitting_dual: Scaled splitting dual ``u`` of each body.
            dynamic_rows: Dynamic rows, which hold the Jacobians of the effort rows.
            effort_rows: Effort-limit rows and their counter-impulses.
            structural_rows: Structural rows and their multiplier impulses.
            anchor_scale: Factor ``s`` of the anchor twist.
        """
        wp.launch(
            _build_candidate_right_hand_side,
            dim=self.num_bodies,
            inputs=[
                self._data.body_vector_index,
                self._data.right_hand_side,
                self._data.weight,
                projected_twist,
                anchor_scale,
                splitting_dual,
                effort_rows.body_offset,
                effort_rows.body_index,
                effort_rows.body_side,
                effort_rows.dynamic_row_index,
                dynamic_rows.jacobian_a,
                dynamic_rows.jacobian_b,
                effort_rows.counter,
                structural_rows.body_impulse,
                self._data.body_block,
                self._data.info.dim,
                self._data.single_body_inverse,
            ],
            outputs=[self._data.candidate_right_hand_side, self._data.packed_solution],
            device=self.device,
        )

    def solve_candidate(self) -> None:
        """Solve the prepared candidate right-hand side into the packed solution.

        Read the twist of each body with :func:`.system_kernels.unpack_body_solution`.
        """
        if self.linear_solver is not None:
            self.linear_solver.solve(self._data.candidate_right_hand_side, self._data.packed_solution)

    def solve(self, right_hand_side: wp.array[wp.float32], solution: wp.array[wp.float32]) -> None:
        """Solve the factorized matrices for a packed right-hand side.

        The candidate solve of the iterations uses :meth:`solve_candidate`, whose right-hand-side
        assembly already applies the single-body inverses.
        """
        if self.num_sparse_blocks < self.num_blocks:
            wp.launch(
                _solve_single_body_blocks,
                dim=self.num_bodies,
                inputs=[
                    self._data.body_block,
                    self._data.body_vector_index,
                    self._data.info.dim,
                    self._data.single_body_inverse,
                    right_hand_side,
                ],
                outputs=[solution],
                device=self.device,
            )
        if self.linear_solver is not None:
            self.linear_solver.solve(right_hand_side, solution)

    ###
    # Internals
    ###

    def _structural_pairs(
        self,
        edges: np.ndarray,
        body_block: np.ndarray,
        body_local: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return the factor block, row and column of every matrix entry that the assembly can write.

        These are the entries of the diagonal block of each dynamic body and of the off-diagonal
        blocks of its body edges.
        """
        if edges.size:
            edge_blocks = body_block[edges]
            dynamic_edge = np.all(edge_blocks >= 0, axis=1)
            if np.any(edge_blocks[dynamic_edge, 0] != edge_blocks[dynamic_edge, 1]):
                raise ValueError("A body edge cannot span factor blocks.")
            edges = edges[dynamic_edge]
        dynamic_bodies = np.flatnonzero((body_block >= 0) & (body_block < self.num_sparse_blocks))
        body_a = np.concatenate((dynamic_bodies, edges[:, 0])).astype(np.int32)
        body_b = np.concatenate((dynamic_bodies, edges[:, 1])).astype(np.int32)
        axes = np.arange(6, dtype=np.int32)
        rows = 6 * body_local[body_a][:, None, None] + axes[None, :, None]
        cols = 6 * body_local[body_b][:, None, None] + axes[None, None, :]
        rows, cols = np.broadcast_arrays(rows, cols)
        pair_block = np.repeat(body_block[body_a], 36)
        return pair_block, rows.reshape(-1), cols.reshape(-1)
