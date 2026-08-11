# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Batched tetrahedral as-rigid-as-possible constraints."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import warp as wp
from warp.fem.linalg import symmetric_eigenvalues_qr

from ..block_csr import BlockCsrBuilder, BlockCsrMatrix


class vec9(wp.types.vector(length=9, dtype=wp.float32)):
    """Nine-vector used for column-major deformation-gradient coordinates."""


class mat99(wp.types.matrix(shape=(9, 9), dtype=wp.float32)):
    """Nine-by-nine deformation-gradient Hessian."""


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


@wp.func
def _flatten_matrix_column_major(matrix: wp.mat33) -> vec9:
    """Flatten a three-by-three matrix in libuipc's column-major order."""
    result = vec9(0.0)
    for column in range(3):
        for row in range(3):
            result[3 * column + row] = matrix[row, column]
    return result


@wp.func
def _guarded_singular_sum(value_0: float, value_1: float) -> float:
    """Keep an analytical twist denominator finite near a singular state."""
    value = value_0 + value_1
    if wp.abs(value) < 1.0e-6:
        if value < 0.0:
            return -1.0e-6
        return 1.0e-6
    return value


@wp.func
def _subtract_twist_mode(
    hessian: mat99,
    left: wp.mat33,
    right: wp.mat33,
    twist: wp.mat33,
    singular_sum: float,
) -> mat99:
    """Subtract one exact rotational mode from the ARAP Hessian."""
    rotated_twist = left * twist * wp.transpose(right) / wp.sqrt(2.0)
    mode = _flatten_matrix_column_major(rotated_twist)
    return hessian - (4.0 / singular_sum) * wp.outer(mode, mode)


@wp.func
def _arap_hessian_unscaled(deformation: wp.mat33) -> mat99:
    """Evaluate libuipc's exact unscaled ARAP Hessian in ``vec(F)``."""
    left, singular_values, right = _signed_svd3(deformation)
    hessian = mat99(0.0)
    for diagonal in range(9):
        hessian[diagonal, diagonal] = 2.0

    twist_0 = wp.mat33(0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    twist_1 = wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, -1.0, 0.0)
    twist_2 = wp.mat33(0.0, 0.0, 1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0)
    hessian = _subtract_twist_mode(
        hessian,
        left,
        right,
        twist_0,
        _guarded_singular_sum(singular_values[0], singular_values[1]),
    )
    hessian = _subtract_twist_mode(
        hessian,
        left,
        right,
        twist_1,
        _guarded_singular_sum(singular_values[1], singular_values[2]),
    )
    return _subtract_twist_mode(
        hessian,
        left,
        right,
        twist_2,
        _guarded_singular_sum(singular_values[0], singular_values[2]),
    )


@wp.func
def _project_psd(hessian: mat99) -> mat99:
    """Project a complete symmetric Hessian with generic EVD clamping."""
    eigenvalues, eigenvectors_by_row = symmetric_eigenvalues_qr(hessian, 1.0e-6)
    projected = mat99(0.0)
    for mode in range(9):
        eigenvalue = wp.max(eigenvalues[mode], 0.0)
        for row in range(9):
            for column in range(9):
                projected[row, column] += (
                    eigenvalue * eigenvectors_by_row[mode, row] * eigenvectors_by_row[mode, column]
                )
    return projected


@wp.func
def _map_hessian_block(
    hessian: mat99,
    material_gradient_i: wp.vec3,
    material_gradient_j: wp.vec3,
) -> wp.mat33:
    """Map a deformation Hessian into one ordered particle-pair block."""
    block = wp.mat33(0.0)
    for spatial_i in range(3):
        for spatial_j in range(3):
            value = float(0.0)
            for material_i in range(3):
                for material_j in range(3):
                    value += (
                        material_gradient_i[material_i]
                        * hessian[3 * material_i + spatial_i, 3 * material_j + spatial_j]
                        * material_gradient_j[material_j]
                    )
            block[spatial_i, spatial_j] = value
    return block


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


@wp.kernel
def _accumulate_tetrahedron_arap_force_and_hessian(
    tetrahedron_indices: wp.array2d[int],
    inverse_rest_matrices: wp.array[wp.mat33],
    rest_volumes: wp.array[float],
    stiffnesses: wp.array[float],
    hessian_block_indices: wp.array2d[int],
    positions: wp.array[wp.vec3],
    forces: wp.array[wp.vec3],
    hessian_values: wp.array[wp.mat33],
):
    tetrahedron = wp.tid()
    particle_0 = tetrahedron_indices[tetrahedron, 0]
    particle_1 = tetrahedron_indices[tetrahedron, 1]
    particle_2 = tetrahedron_indices[tetrahedron, 2]
    particle_3 = tetrahedron_indices[tetrahedron, 3]
    inverse_rest = inverse_rest_matrices[tetrahedron]
    rest_volume = rest_volumes[tetrahedron]
    stiffness = stiffnesses[tetrahedron]
    deformation = _deformation_gradient(
        positions[particle_0],
        positions[particle_1],
        positions[particle_2],
        positions[particle_3],
        inverse_rest,
    )
    gradient = _arap_gradient(deformation, stiffness, rest_volume)
    hessian = stiffness * rest_volume * _project_psd(_arap_hessian_unscaled(deformation))

    for local_i in range(4):
        material_gradient_i = _material_gradient(inverse_rest, local_i)
        particle_i = tetrahedron_indices[tetrahedron, local_i]
        wp.atomic_sub(forces, particle_i, gradient * material_gradient_i)
        for local_j in range(4):
            material_gradient_j = _material_gradient(inverse_rest, local_j)
            block = _map_hessian_block(hessian, material_gradient_i, material_gradient_j)
            block_index = hessian_block_indices[tetrahedron, 4 * local_i + local_j]
            wp.atomic_add(hessian_values, block_index, block)


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
        builder.ensure_stencil_blocks(np.asarray(self.host_tetrahedron_indices, dtype=np.int32))

    def bind_hessian(self, matrix: BlockCsrMatrix) -> None:
        """Bind tetrahedron blocks to finalized block-CSR value indices."""
        if matrix.row_count != self.particle_count or matrix.device != self.device:
            raise ValueError("Constraint and block matrix must have matching particle counts and devices")
        block_indices = matrix.stencil_block_indices(np.asarray(self.host_tetrahedron_indices, dtype=np.int32))
        self.hessian_block_indices = wp.array2d(block_indices, dtype=int, device=self.device)
        self.hessian_value_count = len(matrix.values)

    def accumulate_force(self, positions: wp.array[wp.vec3], output: wp.array[wp.vec3]) -> None:
        """Add ARAP forces evaluated at ``positions`` to ``output``."""
        self._validate_runtime_arrays(positions, output)
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

    def accumulate_force_and_hessian(
        self,
        positions: wp.array[wp.vec3],
        force_output: wp.array[wp.vec3],
        hessian_values: wp.array[wp.mat33],
    ) -> None:
        """Add ARAP forces and full analytical PSD Hessian blocks."""
        self._validate_runtime_arrays(positions, force_output)
        if self.hessian_block_indices is None:
            raise RuntimeError("bind_hessian() must be called before Hessian assembly")
        if hessian_values.dtype != wp.mat33:
            raise ValueError("Hessian values must contain wp.mat33 blocks")
        if hessian_values.device != self.device:
            raise ValueError("Constraint and Hessian values must use the same device")
        if len(hessian_values) != self.hessian_value_count:
            raise ValueError(f"Expected {self.hessian_value_count} Hessian blocks")
        wp.launch(
            _accumulate_tetrahedron_arap_force_and_hessian,
            dim=len(self.rest_volumes),
            inputs=[
                self.tetrahedron_indices,
                self.inverse_rest_matrices,
                self.rest_volumes,
                self.stiffnesses,
                self.hessian_block_indices,
                positions,
            ],
            outputs=[force_output, hessian_values],
            device=self.device,
        )

    def _validate_runtime_arrays(self, positions: wp.array[wp.vec3], output: wp.array[wp.vec3]) -> None:
        if positions.dtype != wp.vec3 or output.dtype != wp.vec3:
            raise ValueError("Positions and forces must contain wp.vec3 rows")
        if len(positions) != self.particle_count or len(output) != self.particle_count:
            raise ValueError(f"Expected {self.particle_count} particle rows")
        if positions.device != self.device or output.device != self.device:
            raise ValueError("Constraint and runtime arrays must use the same device")
