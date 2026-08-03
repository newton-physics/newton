# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Batched two-particle distance constraints."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import warp as wp

from ..block_csr import BlockCsrBuilder, BlockCsrMatrix


@wp.kernel
def _accumulate_distance_force(
    index_pairs: wp.array2d[int],
    rest_lengths: wp.array[float],
    stiffnesses: wp.array[float],
    positions: wp.array[wp.vec3],
    forces: wp.array[wp.vec3],
):
    constraint = wp.tid()
    particle_i = index_pairs[constraint, 0]
    particle_j = index_pairs[constraint, 1]
    displacement = positions[particle_j] - positions[particle_i]
    length = wp.length(displacement)
    if length > 1.0e-8:
        direction = displacement / length
        force = stiffnesses[constraint] * (length - rest_lengths[constraint]) * direction
        wp.atomic_add(forces, particle_i, force)
        wp.atomic_sub(forces, particle_j, force)


@wp.kernel
def _accumulate_distance_force_and_hessian(
    index_pairs: wp.array2d[int],
    rest_lengths: wp.array[float],
    stiffnesses: wp.array[float],
    hessian_block_indices: wp.array2d[int],
    positions: wp.array[wp.vec3],
    forces: wp.array[wp.vec3],
    hessian_values: wp.array[wp.mat33],
):
    constraint = wp.tid()
    particle_i = index_pairs[constraint, 0]
    particle_j = index_pairs[constraint, 1]
    displacement = positions[particle_j] - positions[particle_i]
    length = wp.length(displacement)
    if length > 1.0e-8:
        rest_length = rest_lengths[constraint]
        stiffness = stiffnesses[constraint]
        direction = displacement / length
        normal_outer = wp.outer(direction, direction)
        tangent_projection = wp.identity(3, float) - normal_outer
        transverse_eigenvalue = wp.max(stiffness * (1.0 - rest_length / length), 0.0)
        hessian = stiffness * normal_outer + transverse_eigenvalue * tangent_projection
        force = stiffness * (length - rest_length) * direction

        wp.atomic_add(forces, particle_i, force)
        wp.atomic_sub(forces, particle_j, force)
        wp.atomic_add(hessian_values, hessian_block_indices[constraint, 0], hessian)
        wp.atomic_add(hessian_values, hessian_block_indices[constraint, 1], -hessian)
        wp.atomic_add(hessian_values, hessian_block_indices[constraint, 2], -hessian)
        wp.atomic_add(hessian_values, hessian_block_indices[constraint, 3], hessian)


class ConstraintDistance:
    """A batch of Hookean springs between pairs of particles."""

    def __init__(
        self,
        index_pairs: Sequence[tuple[int, int]],
        rest_lengths: Sequence[float],
        stiffnesses: Sequence[float],
        particle_count: int,
        device: Any,
    ):
        """Create a distance constraint batch.

        Args:
            index_pairs: Particle index pair for every spring.
            rest_lengths: Positive spring rest lengths [m].
            stiffnesses: Positive spring stiffnesses [N/m].
            particle_count: Number of particles in the associated model.
            device: Warp device storing runtime arrays.
        """
        if particle_count <= 0:
            raise ValueError("particle_count must be positive")
        if len(index_pairs) == 0 or len(index_pairs) != len(rest_lengths) or len(index_pairs) != len(stiffnesses):
            raise ValueError("Distance pairs, rest lengths, and stiffnesses must have the same nonzero length")

        self.host_index_pairs = tuple((int(pair[0]), int(pair[1])) for pair in index_pairs)
        self.host_rest_lengths = tuple(float(length) for length in rest_lengths)
        self.host_stiffnesses = tuple(float(stiffness) for stiffness in stiffnesses)
        for particle_i, particle_j in self.host_index_pairs:
            if particle_i < 0 or particle_i >= particle_count or particle_j < 0 or particle_j >= particle_count:
                raise ValueError(
                    f"Distance pair ({particle_i}, {particle_j}) is outside particle_count={particle_count}"
                )
            if particle_i == particle_j:
                raise ValueError("Distance constraint endpoints must differ")
        for length in self.host_rest_lengths:
            if not np.isfinite(length) or length <= 0.0:
                raise ValueError("Distance rest lengths must be finite and positive")
        for stiffness in self.host_stiffnesses:
            if not np.isfinite(stiffness) or stiffness <= 0.0:
                raise ValueError("Distance stiffnesses must be finite and positive")

        self.particle_count = particle_count
        self.device = wp.get_device(device)
        self.index_pairs = wp.array2d(self.host_index_pairs, dtype=int, device=self.device)
        self.rest_lengths = wp.array(self.host_rest_lengths, dtype=float, device=self.device)
        self.stiffnesses = wp.array(self.host_stiffnesses, dtype=float, device=self.device)
        self.hessian_block_indices: wp.array2d[int] | None = None

    def append_hessian_structure(self, builder: BlockCsrBuilder) -> None:
        """Append block coordinates required by this constraint batch."""
        if builder.row_count != self.particle_count:
            raise ValueError("Constraint and block matrix particle counts differ")
        for particle_i, particle_j in self.host_index_pairs:
            builder.ensure_block(particle_i, particle_i)
            builder.ensure_block(particle_i, particle_j)
            builder.ensure_block(particle_j, particle_i)
            builder.ensure_block(particle_j, particle_j)

    def bind_hessian(self, matrix: BlockCsrMatrix) -> None:
        """Bind distance constraints to finalized block-CSR value indices."""
        if matrix.row_count != self.particle_count or matrix.device != self.device:
            raise ValueError("Constraint and block matrix must have matching particle counts and devices")
        block_indices = [
            (
                matrix.block_index(particle_i, particle_i),
                matrix.block_index(particle_i, particle_j),
                matrix.block_index(particle_j, particle_i),
                matrix.block_index(particle_j, particle_j),
            )
            for particle_i, particle_j in self.host_index_pairs
        ]
        self.hessian_block_indices = wp.array2d(block_indices, dtype=int, device=self.device)

    def append_hessian(self, builder: BlockCsrBuilder) -> None:
        """Append the fixed projective-dynamics Hessian blocks."""
        if builder.row_count != self.particle_count:
            raise ValueError("Constraint and block matrix particle counts differ")
        for (particle_i, particle_j), stiffness in zip(self.host_index_pairs, self.host_stiffnesses, strict=True):
            builder.add_scaled_identity(particle_i, particle_i, stiffness)
            builder.add_scaled_identity(particle_i, particle_j, -stiffness)
            builder.add_scaled_identity(particle_j, particle_i, -stiffness)
            builder.add_scaled_identity(particle_j, particle_j, stiffness)

    def accumulate_force(self, positions: wp.array[wp.vec3], output: wp.array[wp.vec3]) -> None:
        """Add distance-spring forces evaluated at ``positions`` to ``output``."""
        self._validate_runtime_arrays(positions, output)
        wp.launch(
            _accumulate_distance_force,
            dim=len(self.rest_lengths),
            inputs=[self.index_pairs, self.rest_lengths, self.stiffnesses, positions],
            outputs=[output],
            device=self.device,
        )

    def accumulate_force_and_hessian(
        self,
        positions: wp.array[wp.vec3],
        force_output: wp.array[wp.vec3],
        hessian_values: wp.array[wp.mat33],
    ) -> None:
        """Add spring force and projected Hessian blocks evaluated at ``positions``."""
        self._validate_runtime_arrays(positions, force_output)
        if self.hessian_block_indices is None:
            raise RuntimeError("bind_hessian() must be called before Hessian assembly")
        if hessian_values.device != self.device:
            raise ValueError("Constraint and Hessian values must use the same device")
        wp.launch(
            _accumulate_distance_force_and_hessian,
            dim=len(self.rest_lengths),
            inputs=[
                self.index_pairs,
                self.rest_lengths,
                self.stiffnesses,
                self.hessian_block_indices,
                positions,
            ],
            outputs=[force_output, hessian_values],
            device=self.device,
        )

    def _validate_runtime_arrays(self, positions: wp.array[wp.vec3], output: wp.array[wp.vec3]) -> None:
        if len(positions) != self.particle_count or len(output) != self.particle_count:
            raise ValueError(f"Expected {self.particle_count} particle rows")
        if positions.device != self.device or output.device != self.device:
            raise ValueError("Constraint and runtime arrays must use the same device")
