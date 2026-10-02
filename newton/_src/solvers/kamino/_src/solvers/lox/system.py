# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Block-diagonal primal body systems for the LOX backend.

This module owns only solver-internal, contact-free body-space assembly. Each
factor block is one connected dynamic-body component and each dynamic body
contributes six linear-first velocity unknowns. Prescribed bodies retain their
global packed body indices but map to ``-1`` in the matrix layout.
"""

from __future__ import annotations

from collections.abc import Sequence
from functools import lru_cache

import numpy as np
import warp as wp

from ...core.types import vec6f
from ...linalg import DenseLinearOperatorData
from .llt import DenseBlockInfo, HybridLLTBlockedSolver
from .system_kernels import (
    _assemble_body_inertial_systems,
    _assemble_dynamic_joint_rows,
    _assemble_structural_joint_rows,
    _build_candidate_right_hand_side,
    _compute_body_weights_and_add,
)
from .types import DynamicRows, EffortRows, LOXSystemData, StructuralRows, capacity_offsets, segment_local_indices

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


###
# Constants
###


_RCM_MIN_DIMENSION = 256


###
# Functions
###


@lru_cache(maxsize=32)
def _expand_body_symbolic_adjacency(
    body_count: int,
    body_edges: tuple[tuple[int, int], ...],
) -> tuple[tuple[int, ...], ...]:
    """Expand a reusable body graph into its six-DoF scalar adjacency."""
    body_adjacency = np.eye(body_count, dtype=bool)
    if body_edges:
        edges = np.asarray(body_edges, dtype=np.int32)
        body_adjacency[edges[:, 0], edges[:, 1]] = True
        body_adjacency[edges[:, 1], edges[:, 0]] = True
    scalar_adjacency = np.kron(body_adjacency, np.ones((6, 6), dtype=bool))
    np.fill_diagonal(scalar_adjacency, False)
    return tuple(tuple(np.flatnonzero(row).tolist()) for row in scalar_adjacency)


###
# Interfaces
###


class LOXSystem:
    """Batched dense smooth body systems over independent body components.

    World-level bookkeeping remains in model body order. The dense matrices and
    vectors are packed by ``body_components`` so disconnected articulations can
    be factorized independently. When every body is prescribed, the system holds
    one empty placeholder block.
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

        # Pack each joint-connected component of dynamic bodies into one dense factor block.
        components = [np.asarray(component, dtype=np.int32) for component in body_components]
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

        body_block = np.full(self.num_bodies, -1, dtype=np.int32)
        body_local = np.full(self.num_bodies, -1, dtype=np.int32)
        body_block[packed_bodies] = np.repeat(np.arange(self.num_blocks, dtype=np.int32), block_body_counts)
        body_local[packed_bodies] = segment_local_indices(block_body_counts, block_offsets)
        body_vector_index = np.where(body_block >= 0, 6 * (block_offsets[body_block.clip(min=0)] + body_local), -1)
        self.body_block_host = tuple(body_block.tolist())
        self.body_local_host = tuple(body_local.tolist())
        self.body_vector_index_host = tuple(body_vector_index.tolist())

        edges = np.asarray(body_edges, dtype=np.int32).reshape((-1, 2))
        edges.sort(axis=1)
        edges = np.unique(edges, axis=0)
        self._body_edge_keys_host = edges[:, 0].astype(np.int64) * self.num_bodies + edges[:, 1]

        info = DenseBlockInfo()
        info.finalize(dimensions=storage_dimensions, dtype=wp.float32, itype=wp.int32, device=self.device)
        if active_dimensions != storage_dimensions:
            info.dim = wp.array(active_dimensions, dtype=wp.int32, device=self.device)
        self._data = LOXSystemData(body_block, body_local, body_vector_index, info, self.device)
        self.operator = DenseLinearOperatorData(info=info, mat=self._data.matrix)
        # Factorization benefits from fewer wide dense panels, while repeated
        # single-RHS solves retain the smaller tile for better occupancy.
        self.linear_solver = HybridLLTBlockedSolver(
            operator=self.operator,
            factorize_block_size=64,
            solve_block_dim=256,
            rcm_min_dimension=_RCM_MIN_DIMENSION,
            symbolic_adjacency=self._build_symbolic_adjacency(edges),
            dtype=wp.float32,
            device=self.device,
        )
        self._mass: wp.array[wp.float32] | None = None
        self._inertia_world: wp.array[wp.mat33f] | None = None

    ###
    # Properties
    ###

    @property
    def data(self) -> LOXSystemData:
        """Returns the arrays of the smooth body systems."""
        return self._data

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
        """Reset and assemble body inertia and explicit-force terms."""
        self._reset()
        self._mass = mass
        self._inertia_world = inertia_world
        wp.launch(
            _assemble_body_inertial_systems,
            dim=self.num_bodies,
            inputs=[
                self.body_world,
                self._data.body_block,
                self._data.body_local,
                self._data.info.dim,
                self._data.info.mio,
                self._data.info.vio,
                mass,
                inertia_world,
                velocity_previous,
                external_wrench,
                actuation_wrench,
                gravity,
                time_step,
            ],
            outputs=[self._data.matrix, self._data.right_hand_side],
            device=self.device,
        )

    def add_dynamic_rows(
        self,
        row_world: wp.array[wp.int32],
        body_a: wp.array[wp.int32],
        body_b: wp.array[wp.int32],
        jacobian_a: wp.array[vec6f],
        jacobian_b: wp.array[vec6f],
        effective_inertia: wp.array[wp.float32],
        free_velocity: wp.array[wp.float32],
        prescribed_twist: wp.array[vec6f] | None = None,
    ) -> None:
        """Add implicit joint-dynamics rows to the smooth system."""
        row_count = row_world.shape[0]
        if row_count == 0:
            return
        wp.launch(
            _assemble_dynamic_joint_rows,
            dim=row_count,
            inputs=[
                self._data.info.dim,
                self._data.info.mio,
                self._data.info.vio,
                body_a,
                body_b,
                self._data.body_block,
                self._data.body_local,
                jacobian_a,
                jacobian_b,
                effective_inertia,
                free_velocity,
                prescribed_twist,
            ],
            outputs=[self._data.matrix, self._data.right_hand_side],
            device=self.device,
        )

    def add_structural_rows(
        self,
        row_world: wp.array[wp.int32],
        body_a: wp.array[wp.int32],
        body_b: wp.array[wp.int32],
        jacobian_a: wp.array[vec6f],
        jacobian_b: wp.array[vec6f],
        residual: wp.array[wp.float32],
        effective_mass: wp.array[wp.float32],
        time_step: wp.array[wp.float32],
        joint_penalty_scale: wp.array[wp.float32],
        penalty: wp.array[wp.float32],
        joint_max_correction: float,
        prescribed_twist: wp.array[vec6f] | None = None,
    ) -> None:
        """Add the frozen augmented penalty terms of the structural rows and write their penalties.

        The multiplier terms are added to the candidate right-hand side of each iteration.
        Each row corrects its residual up to ``joint_max_correction`` in one step.
        """
        row_count = row_world.shape[0]
        if row_count == 0:
            return
        wp.launch(
            _assemble_structural_joint_rows,
            dim=row_count,
            inputs=[
                self._data.info.dim,
                self._data.info.mio,
                self._data.info.vio,
                row_world,
                body_a,
                body_b,
                self._data.body_block,
                self._data.body_local,
                jacobian_a,
                jacobian_b,
                residual,
                effective_mass,
                prescribed_twist,
                time_step,
                joint_penalty_scale,
                joint_max_correction,
            ],
            outputs=[penalty, self._data.matrix, self._data.right_hand_side],
            device=self.device,
        )

    def build_weighted_matrix(
        self,
        body_weight_enabled: wp.array[wp.int32],
        *,
        sigma: float,
        beta: float,
    ) -> None:
        """Compute the splitting weights ``W`` of the enabled bodies and add them to the matrix ``A``.

        Args:
            body_weight_enabled: Whether each body receives a splitting weight; other bodies get zero.
            sigma: Relative lower scale of the inertia-normalized body-weight clamp.
            beta: Normalized smooth-weight transition threshold of the body-weight heuristic.
        """
        wp.launch(
            _compute_body_weights_and_add,
            dim=self.num_bodies,
            inputs=[
                self._data.body_block,
                self._data.body_local,
                body_weight_enabled,
                self._data.info.dim,
                self._data.info.mio,
                self._mass,
                self._inertia_world,
                sigma,
                beta,
            ],
            outputs=[
                self._data.weight,
                self._data.inverse_weight,
                self._data.matrix,
            ],
            device=self.device,
        )

    def factorize(self) -> None:
        """Factorize the current weighted matrices with batched dense LLT."""
        self.linear_solver.compute(self._data.matrix)

    def build_candidate_right_hand_side(
        self,
        projected_twist: wp.array[vec6f],
        splitting_dual: wp.array[vec6f],
        dynamic_rows: DynamicRows,
        effort_rows: EffortRows,
        structural_rows: StructuralRows,
    ) -> None:
        """Build the right-hand side of the candidate solve of one iteration.

        Adds the splitting target ``W (p + u)`` and the current row impulses to the frozen
        right-hand side ``f``: the structural multiplier impulses ``h J_s^T lambda`` and the
        effort-limit counter-impulses ``J_d^T c``.
        """
        wp.launch(
            _build_candidate_right_hand_side,
            dim=self.num_bodies,
            inputs=[
                self._data.body_vector_index,
                self._data.right_hand_side,
                self._data.weight,
                projected_twist,
                splitting_dual,
                effort_rows.body_offset,
                effort_rows.body_index,
                effort_rows.body_side,
                effort_rows.dynamic_row_index,
                dynamic_rows.jacobian_a,
                dynamic_rows.jacobian_b,
                effort_rows.counter,
                structural_rows.body_impulse,
            ],
            outputs=[self._data.candidate_right_hand_side],
            device=self.device,
        )

    def solve_candidate(self) -> None:
        """Solve the prepared candidate right-hand side into the packed solution.

        Read the twist of each body with :func:`.system_kernels.unpack_body_solution`.
        """
        self.linear_solver.solve(self._data.candidate_right_hand_side, self._data.packed_solution)

    ###
    # Internals
    ###

    def _reset(self) -> None:
        """Clear the matrix before an assembly; the factorization overwrites its buffers."""
        self._data.zero()

    def _build_symbolic_adjacency(self, edges: np.ndarray) -> tuple[tuple[tuple[int, ...], ...], ...]:
        """Expand fixed body topology into scalar adjacency per factor block."""
        body_block = np.asarray(self.body_block_host, dtype=np.int32)
        body_local = np.asarray(self.body_local_host, dtype=np.int32)
        if edges.size:
            edge_blocks = body_block[edges]
            dynamic_edge = np.all(edge_blocks >= 0, axis=1)
            if np.any(edge_blocks[dynamic_edge, 0] != edge_blocks[dynamic_edge, 1]):
                raise ValueError("A body edge cannot span factor blocks.")
            edges = edges[dynamic_edge]
            edge_block = edge_blocks[dynamic_edge, 0]
            local_edges = body_local[edges]
            local_edges.sort(axis=1)
            order = np.lexsort((local_edges[:, 1], local_edges[:, 0], edge_block))
            edge_block = edge_block[order]
            local_edges = local_edges[order]
            edge_counts = np.bincount(edge_block, minlength=self.num_blocks).astype(np.int32, copy=False)
            edge_offsets = capacity_offsets(edge_counts)
        else:
            local_edges = edges
            edge_offsets = np.zeros(self.num_blocks + 1, dtype=np.int32)

        adjacency_blocks = []
        for block, body_count in enumerate(self.block_body_counts):
            if 6 * body_count < _RCM_MIN_DIMENSION:
                adjacency_blocks.append(())
                continue
            begin = edge_offsets[block]
            end = edge_offsets[block + 1]
            block_edges = tuple(map(tuple, local_edges[begin:end].tolist()))
            adjacency_blocks.append(_expand_body_symbolic_adjacency(body_count, block_edges))
        return tuple(adjacency_blocks)
