# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Batched tetrahedral as-rigid-as-possible constraints."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import warp as wp

from ..block_csr import BlockCsrBuilder, BlockCsrMatrix


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
