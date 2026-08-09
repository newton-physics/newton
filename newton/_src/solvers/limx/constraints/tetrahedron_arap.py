# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Batched tetrahedral as-rigid-as-possible constraints."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import warp as wp

from ..block_csr import BlockCsrBuilder, BlockCsrMatrix


@wp.func
def _signed_svd3(deformation: wp.mat33) -> tuple[wp.mat33, wp.vec3, wp.mat33]:
    """Return proper singular-vector bases while preserving the factorization."""
    left, singular_values, right = wp.svd3(deformation)
    if wp.determinant(left) < 0.0:
        for row in range(3):
            left[row, 2] = -left[row, 2]
        singular_values[2] = -singular_values[2]
    if wp.determinant(right) < 0.0:
        for row in range(3):
            right[row, 2] = -right[row, 2]
        singular_values[2] = -singular_values[2]
    return left, singular_values, right


@wp.func
def _deformation_gradient(
    position_0: wp.vec3,
    position_1: wp.vec3,
    position_2: wp.vec3,
    position_3: wp.vec3,
    inverse_rest: wp.mat33,
) -> wp.mat33:
    """Compute ``Ds * Dm_inverse`` with spatial edge vectors as columns."""
    edge_01 = position_1 - position_0
    edge_02 = position_2 - position_0
    edge_03 = position_3 - position_0
    current_edges = wp.mat33(0.0)
    for row in range(3):
        current_edges[row, 0] = edge_01[row]
        current_edges[row, 1] = edge_02[row]
        current_edges[row, 2] = edge_03[row]
    return current_edges * inverse_rest


@wp.func
def _material_gradient(inverse_rest: wp.mat33, local_vertex: int) -> wp.vec3:
    """Return one vertex gradient of the tetrahedral material coordinates."""
    if local_vertex == 1:
        return wp.vec3(inverse_rest[0, 0], inverse_rest[0, 1], inverse_rest[0, 2])
    if local_vertex == 2:
        return wp.vec3(inverse_rest[1, 0], inverse_rest[1, 1], inverse_rest[1, 2])
    if local_vertex == 3:
        return wp.vec3(inverse_rest[2, 0], inverse_rest[2, 1], inverse_rest[2, 2])
    return -wp.vec3(
        inverse_rest[0, 0] + inverse_rest[1, 0] + inverse_rest[2, 0],
        inverse_rest[0, 1] + inverse_rest[1, 1] + inverse_rest[2, 1],
        inverse_rest[0, 2] + inverse_rest[1, 2] + inverse_rest[2, 2],
    )


@wp.func
def _arap_energy(deformation: wp.mat33, stiffness: float, rest_volume: float) -> float:
    """Evaluate ``kappa * V0 * ||F - R||_F^2``."""
    left, _singular_values, right = _signed_svd3(deformation)
    rotation = left * wp.transpose(right)
    strain = deformation - rotation
    return stiffness * rest_volume * wp.ddot(strain, strain)


@wp.func
def _arap_gradient(deformation: wp.mat33, stiffness: float, rest_volume: float) -> wp.mat33:
    """Evaluate the first derivative of ARAP energy with respect to ``F``."""
    left, _singular_values, right = _signed_svd3(deformation)
    rotation = left * wp.transpose(right)
    return 2.0 * stiffness * rest_volume * (deformation - rotation)


@wp.kernel
def _accumulate_tetrahedron_arap_force(
    tetrahedron_indices: wp.array2d[int],
    inverse_rest_matrices: wp.array[wp.mat33],
    rest_volumes: wp.array[float],
    stiffnesses: wp.array[float],
    positions: wp.array[wp.vec3],
    forces: wp.array[wp.vec3],
):
    tetrahedron = wp.tid()
    particle_0 = tetrahedron_indices[tetrahedron, 0]
    particle_1 = tetrahedron_indices[tetrahedron, 1]
    particle_2 = tetrahedron_indices[tetrahedron, 2]
    particle_3 = tetrahedron_indices[tetrahedron, 3]
    inverse_rest = inverse_rest_matrices[tetrahedron]
    deformation = _deformation_gradient(
        positions[particle_0],
        positions[particle_1],
        positions[particle_2],
        positions[particle_3],
        inverse_rest,
    )
    gradient = _arap_gradient(deformation, stiffnesses[tetrahedron], rest_volumes[tetrahedron])

    for local_vertex in range(4):
        material_gradient = _material_gradient(inverse_rest, local_vertex)
        particle = tetrahedron_indices[tetrahedron, local_vertex]
        wp.atomic_sub(forces, particle, gradient * material_gradient)


class ConstraintTetrahedronARAP:
    """A batch of tetrahedral as-rigid-as-possible elastic constraints."""

    def __init__(
        self,
        tetrahedron_indices: Sequence[tuple[int, int, int, int]],
        inverse_rest_matrices: Sequence[wp.mat33],
        stiffnesses: Sequence[float],
        particle_count: int,
        device: Any,
    ):
        """Create a tetrahedral ARAP constraint batch.

        Args:
            tetrahedron_indices: Four particle indices per tetrahedron.
            inverse_rest_matrices: Inverse rest matrices per tetrahedron [1/m].
            stiffnesses: Positive ARAP stiffness per tetrahedron [Pa].
            particle_count: Number of particles in the associated model.
            device: Warp device storing runtime arrays.
        """
        if particle_count <= 0:
            raise ValueError("particle_count must be positive")
        tetrahedron_count = len(tetrahedron_indices)
        if (
            tetrahedron_count == 0
            or tetrahedron_count != len(inverse_rest_matrices)
            or tetrahedron_count != len(stiffnesses)
        ):
            raise ValueError(
                "Tetrahedron indices, inverse rest matrices, and stiffnesses must have equal nonzero length"
            )

        self.host_tetrahedron_indices = tuple(
            tuple(int(index) for index in tetrahedron) for tetrahedron in tetrahedron_indices
        )
        self.host_inverse_rest_matrices = tuple(
            np.asarray(matrix, dtype=np.float32).reshape(3, 3) for matrix in inverse_rest_matrices
        )
        self.host_stiffnesses = tuple(float(stiffness) for stiffness in stiffnesses)

        for tetrahedron in self.host_tetrahedron_indices:
            if len(tetrahedron) != 4 or len(set(tetrahedron)) != 4:
                raise ValueError("Tetrahedra must contain exactly four distinct particle indices")
            if any(index < 0 or index >= particle_count for index in tetrahedron):
                raise ValueError(f"Tetrahedron {tetrahedron} is outside particle_count={particle_count}")

        rest_volumes = []
        for inverse_rest in self.host_inverse_rest_matrices:
            if not np.isfinite(inverse_rest).all():
                raise ValueError("Inverse rest matrices must be finite and nonsingular")
            determinant = float(np.linalg.det(inverse_rest))
            if not np.isfinite(determinant) or abs(determinant) <= 1.0e-12:
                raise ValueError("Inverse rest matrices must be finite and nonsingular")
            rest_volume = 1.0 / (6.0 * determinant)
            if not np.isfinite(rest_volume) or rest_volume <= 0.0:
                raise ValueError("Inverse rest matrices must define positive rest volumes")
            rest_volumes.append(rest_volume)
        self.host_rest_volumes = tuple(rest_volumes)

        for stiffness in self.host_stiffnesses:
            if not np.isfinite(stiffness) or stiffness <= 0.0:
                raise ValueError("ARAP stiffnesses must be finite and positive")

        self.particle_count = particle_count
        self.device = wp.get_device(device)
        self.tetrahedron_indices = wp.array2d(self.host_tetrahedron_indices, dtype=int, device=self.device)
        self.inverse_rest_matrices = wp.array(
            np.asarray(self.host_inverse_rest_matrices), dtype=wp.mat33, device=self.device
        )
        self.rest_volumes = wp.array(self.host_rest_volumes, dtype=float, device=self.device)
        self.stiffnesses = wp.array(self.host_stiffnesses, dtype=float, device=self.device)
        self.hessian_block_indices: wp.array2d[int] | None = None
        self.hessian_value_count: int | None = None

    def append_hessian_structure(self, builder: BlockCsrBuilder) -> None:
        """Append all sixteen ordered particle-pair blocks per tetrahedron."""
        if builder.row_count != self.particle_count:
            raise ValueError("Constraint and block matrix particle counts differ")
        for tetrahedron in self.host_tetrahedron_indices:
            for particle_i in tetrahedron:
                for particle_j in tetrahedron:
                    builder.ensure_block(particle_i, particle_j)

    def bind_hessian(self, matrix: BlockCsrMatrix) -> None:
        """Bind tetrahedron blocks to finalized block-CSR value indices."""
        if matrix.row_count != self.particle_count or matrix.device != self.device:
            raise ValueError("Constraint and block matrix must have matching particle counts and devices")
        block_indices = [
            tuple(
                matrix.block_index(particle_i, particle_j) for particle_i in tetrahedron for particle_j in tetrahedron
            )
            for tetrahedron in self.host_tetrahedron_indices
        ]
        self.hessian_block_indices = wp.array2d(block_indices, dtype=int, device=self.device)
        self.hessian_value_count = len(matrix.values)

    def accumulate_force(self, positions: wp.array[wp.vec3], output: wp.array[wp.vec3]) -> None:
        """Add ARAP forces evaluated at ``positions`` to ``output``."""
        wp.launch(
            _accumulate_tetrahedron_arap_force,
            dim=len(self.rest_volumes),
            inputs=[
                self.tetrahedron_indices,
                self.inverse_rest_matrices,
                self.rest_volumes,
                self.stiffnesses,
                positions,
            ],
            outputs=[output],
            device=self.device,
        )
