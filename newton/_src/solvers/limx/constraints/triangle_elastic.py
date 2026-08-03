# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Batched anisotropic triangle membrane constraints."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import warp as wp

from ..block_csr import BlockCsrBuilder, BlockCsrMatrix


@wp.kernel
def _accumulate_triangle_elastic_force(
    triangle_indices: wp.array2d[int],
    inverse_rest_matrices: wp.array[wp.mat22],
    rest_areas: wp.array[float],
    stiffnesses: wp.array[wp.vec3],
    positions: wp.array[wp.vec3],
    forces: wp.array[wp.vec3],
):
    triangle = wp.tid()
    particle_0 = triangle_indices[triangle, 0]
    particle_1 = triangle_indices[triangle, 1]
    particle_2 = triangle_indices[triangle, 2]
    inverse_rest = inverse_rest_matrices[triangle]

    edge_01 = positions[particle_1] - positions[particle_0]
    edge_02 = positions[particle_2] - positions[particle_0]
    deformation_u = edge_01 * inverse_rest[0, 0] + edge_02 * inverse_rest[1, 0]
    deformation_v = edge_01 * inverse_rest[0, 1] + edge_02 * inverse_rest[1, 1]
    length_u = wp.length(deformation_u)
    length_v = wp.length(deformation_v)

    derivative_u = wp.vec3(
        -inverse_rest[0, 0] - inverse_rest[1, 0],
        inverse_rest[0, 0],
        inverse_rest[1, 0],
    )
    derivative_v = wp.vec3(
        -inverse_rest[0, 1] - inverse_rest[1, 1],
        inverse_rest[0, 1],
        inverse_rest[1, 1],
    )
    stiffness = stiffnesses[triangle]
    shear = wp.dot(deformation_u, deformation_v)
    face = wp.vec3i(particle_0, particle_1, particle_2)

    for local_vertex in range(3):
        gradient = (
            stiffness[2]
            * shear
            * (derivative_u[local_vertex] * deformation_v + derivative_v[local_vertex] * deformation_u)
        )
        if length_u > 1.0e-8:
            gradient += stiffness[0] * derivative_u[local_vertex] * (length_u - 1.0) * deformation_u / length_u
        if length_v > 1.0e-8:
            gradient += stiffness[1] * derivative_v[local_vertex] * (length_v - 1.0) * deformation_v / length_v
        wp.atomic_sub(forces, face[local_vertex], rest_areas[triangle] * gradient)


@wp.kernel
def _accumulate_triangle_elastic_force_and_hessian(
    triangle_indices: wp.array2d[int],
    inverse_rest_matrices: wp.array[wp.mat22],
    rest_areas: wp.array[float],
    stiffnesses: wp.array[wp.vec3],
    hessian_block_indices: wp.array2d[int],
    positions: wp.array[wp.vec3],
    forces: wp.array[wp.vec3],
    hessian_values: wp.array[wp.mat33],
):
    triangle = wp.tid()
    particle_0 = triangle_indices[triangle, 0]
    particle_1 = triangle_indices[triangle, 1]
    particle_2 = triangle_indices[triangle, 2]
    inverse_rest = inverse_rest_matrices[triangle]

    edge_01 = positions[particle_1] - positions[particle_0]
    edge_02 = positions[particle_2] - positions[particle_0]
    deformation_u = edge_01 * inverse_rest[0, 0] + edge_02 * inverse_rest[1, 0]
    deformation_v = edge_01 * inverse_rest[0, 1] + edge_02 * inverse_rest[1, 1]
    length_u = wp.length(deformation_u)
    length_v = wp.length(deformation_v)

    derivative_u = wp.vec3(
        -inverse_rest[0, 0] - inverse_rest[1, 0],
        inverse_rest[0, 0],
        inverse_rest[1, 0],
    )
    derivative_v = wp.vec3(
        -inverse_rest[0, 1] - inverse_rest[1, 1],
        inverse_rest[0, 1],
        inverse_rest[1, 1],
    )
    stiffness = stiffnesses[triangle]
    shear = wp.dot(deformation_u, deformation_v)
    face = wp.vec3i(particle_0, particle_1, particle_2)
    area = rest_areas[triangle]

    identity = wp.identity(3, float)
    curvature_u = wp.mat33(0.0)
    curvature_v = wp.mat33(0.0)
    if length_u > 1.0e-8:
        direction_u = deformation_u / length_u
        normal_outer_u = wp.outer(direction_u, direction_u)
        transverse_u = wp.max(1.0 - 1.0 / length_u, 0.0)
        curvature_u = stiffness[0] * (normal_outer_u + transverse_u * (identity - normal_outer_u))
    if length_v > 1.0e-8:
        direction_v = deformation_v / length_v
        normal_outer_v = wp.outer(direction_v, direction_v)
        transverse_v = wp.max(1.0 - 1.0 / length_v, 0.0)
        curvature_v = stiffness[1] * (normal_outer_v + transverse_v * (identity - normal_outer_v))

    for local_i in range(3):
        shear_gradient_i = derivative_u[local_i] * deformation_v + derivative_v[local_i] * deformation_u
        gradient = stiffness[2] * shear * shear_gradient_i
        if length_u > 1.0e-8:
            gradient += stiffness[0] * derivative_u[local_i] * (length_u - 1.0) * deformation_u / length_u
        if length_v > 1.0e-8:
            gradient += stiffness[1] * derivative_v[local_i] * (length_v - 1.0) * deformation_v / length_v
        wp.atomic_sub(forces, face[local_i], area * gradient)

        for local_j in range(3):
            shear_gradient_j = derivative_u[local_j] * deformation_v + derivative_v[local_j] * deformation_u
            hessian = area * (
                derivative_u[local_i] * derivative_u[local_j] * curvature_u
                + derivative_v[local_i] * derivative_v[local_j] * curvature_v
                + stiffness[2] * wp.outer(shear_gradient_i, shear_gradient_j)
            )
            wp.atomic_add(hessian_values, hessian_block_indices[triangle, 3 * local_i + local_j], hessian)


class ConstraintTriangleElastic:
    """A batch of anisotropic triangle membrane constraints."""

    def __init__(
        self,
        triangle_indices: Sequence[tuple[int, int, int]],
        inverse_rest_matrices: Sequence[wp.mat22],
        rest_areas: Sequence[float],
        stiffnesses: Sequence[wp.vec3],
        particle_count: int,
        device: Any,
    ):
        """Create an anisotropic triangle membrane constraint batch.

        Args:
            triangle_indices: Three particle indices per triangle.
            inverse_rest_matrices: Inverse 2D rest matrices per triangle [1/m].
            rest_areas: Positive material-space rest areas per triangle [m^2].
            stiffnesses: Nonnegative warp, weft, and shear stiffness per
                triangle [N/m].
            particle_count: Number of particles in the associated model.
            device: Warp device storing runtime arrays.
        """
        if particle_count <= 0:
            raise ValueError("particle_count must be positive")
        triangle_count = len(triangle_indices)
        if (
            triangle_count == 0
            or triangle_count != len(inverse_rest_matrices)
            or triangle_count != len(rest_areas)
            or triangle_count != len(stiffnesses)
        ):
            raise ValueError(
                "Triangle indices, rest matrices, rest areas, and stiffnesses must have equal nonzero length"
            )

        self.host_triangle_indices = tuple(tuple(int(index) for index in triangle) for triangle in triangle_indices)
        self.host_inverse_rest_matrices = tuple(
            np.asarray(matrix, dtype=np.float32).reshape(2, 2) for matrix in inverse_rest_matrices
        )
        self.host_rest_areas = tuple(float(area) for area in rest_areas)
        self.host_stiffnesses = tuple(np.asarray(stiffness, dtype=np.float32).reshape(3) for stiffness in stiffnesses)

        for triangle in self.host_triangle_indices:
            if len(triangle) != 3 or len(set(triangle)) != 3:
                raise ValueError("Triangles must contain exactly three distinct particle indices")
            if any(index < 0 or index >= particle_count for index in triangle):
                raise ValueError(f"Triangle {triangle} is outside particle_count={particle_count}")
        for inverse_rest in self.host_inverse_rest_matrices:
            if not np.isfinite(inverse_rest).all() or abs(float(np.linalg.det(inverse_rest))) <= 1.0e-12:
                raise ValueError("Inverse rest matrices must be finite and nonsingular")
        for area in self.host_rest_areas:
            if not np.isfinite(area) or area <= 0.0:
                raise ValueError("Triangle rest areas must be finite and positive")
        for stiffness in self.host_stiffnesses:
            if not np.isfinite(stiffness).all() or np.any(stiffness < 0.0):
                raise ValueError("Triangle stiffness components must be finite and nonnegative")

        self.particle_count = particle_count
        self.device = wp.get_device(device)
        self.triangle_indices = wp.array2d(self.host_triangle_indices, dtype=int, device=self.device)
        self.inverse_rest_matrices = wp.array(
            np.asarray(self.host_inverse_rest_matrices), dtype=wp.mat22, device=self.device
        )
        self.rest_areas = wp.array(self.host_rest_areas, dtype=float, device=self.device)
        self.stiffnesses = wp.array(np.asarray(self.host_stiffnesses), dtype=wp.vec3, device=self.device)
        self.hessian_block_indices: wp.array2d[int] | None = None
        self.hessian_value_count: int | None = None

    def append_hessian_structure(self, builder: BlockCsrBuilder) -> None:
        """Append all nine ordered particle-pair blocks for every triangle."""
        if builder.row_count != self.particle_count:
            raise ValueError("Constraint and block matrix particle counts differ")
        for triangle in self.host_triangle_indices:
            for particle_i in triangle:
                for particle_j in triangle:
                    builder.ensure_block(particle_i, particle_j)

    def bind_hessian(self, matrix: BlockCsrMatrix) -> None:
        """Bind triangle blocks to finalized block-CSR value indices."""
        if matrix.row_count != self.particle_count or matrix.device != self.device:
            raise ValueError("Constraint and block matrix must have matching particle counts and devices")
        block_indices = [
            tuple(matrix.block_index(particle_i, particle_j) for particle_i in triangle for particle_j in triangle)
            for triangle in self.host_triangle_indices
        ]
        self.hessian_block_indices = wp.array2d(block_indices, dtype=int, device=self.device)
        self.hessian_value_count = len(matrix.values)

    def accumulate_force(self, positions: wp.array[wp.vec3], output: wp.array[wp.vec3]) -> None:
        """Add membrane forces evaluated at ``positions`` to ``output``."""
        self._validate_runtime_arrays(positions, output)
        wp.launch(
            _accumulate_triangle_elastic_force,
            dim=len(self.rest_areas),
            inputs=[
                self.triangle_indices,
                self.inverse_rest_matrices,
                self.rest_areas,
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
        """Add membrane force and analytic PSD Hessian blocks."""
        self._validate_runtime_arrays(positions, force_output)
        if self.hessian_block_indices is None:
            raise RuntimeError("bind_hessian() must be called before Hessian assembly")
        if hessian_values.device != self.device:
            raise ValueError("Constraint and Hessian values must use the same device")
        if len(hessian_values) != self.hessian_value_count:
            raise ValueError(f"Expected {self.hessian_value_count} Hessian blocks")
        wp.launch(
            _accumulate_triangle_elastic_force_and_hessian,
            dim=len(self.rest_areas),
            inputs=[
                self.triangle_indices,
                self.inverse_rest_matrices,
                self.rest_areas,
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
