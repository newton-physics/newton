# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Batched four-particle dihedral-angle bending constraints."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import warp as wp

from ..block_csr import BlockCsrBuilder, BlockCsrMatrix

_MIN_GEOMETRY_NORM = 1.0e-8


def _host_signed_dihedral_angle(positions: np.ndarray, dihedral: tuple[int, int, int, int]) -> float:
    edge_v0, edge_v1, left_opposite, right_opposite = dihedral
    edge = positions[edge_v1] - positions[edge_v0]
    left_edge = positions[left_opposite] - positions[edge_v0]
    right_edge = positions[right_opposite] - positions[edge_v0]
    edge_length = float(np.linalg.norm(edge))
    left_normal_raw = np.cross(left_edge, edge)
    right_normal_raw = np.cross(edge, right_edge)
    left_normal_length = float(np.linalg.norm(left_normal_raw))
    right_normal_length = float(np.linalg.norm(right_normal_raw))
    if min(edge_length, left_normal_length, right_normal_length) <= _MIN_GEOMETRY_NORM:
        raise ValueError(f"Rest dihedral {dihedral} is degenerate")

    edge_direction = edge / edge_length
    left_normal = left_normal_raw / left_normal_length
    right_normal = right_normal_raw / right_normal_length
    edge_length_squared = edge_length * edge_length
    left_projection = float(np.dot(edge, left_edge) / edge_length_squared)
    right_projection = float(np.dot(edge, right_edge) / edge_length_squared)
    left_height = float(np.linalg.norm(left_edge - left_projection * edge))
    right_height = float(np.linalg.norm(right_edge - right_projection * edge))
    if min(left_height, right_height) <= _MIN_GEOMETRY_NORM:
        raise ValueError(f"Rest dihedral {dihedral} is degenerate")

    return float(
        np.arctan2(
            np.dot(np.cross(left_normal, right_normal), edge_direction),
            np.dot(left_normal, right_normal),
        )
    )


@wp.func
def _dihedral_frame(
    edge_position_0: wp.vec3,
    edge_position_1: wp.vec3,
    left_position: wp.vec3,
    right_position: wp.vec3,
):
    edge = edge_position_1 - edge_position_0
    left_edge = left_position - edge_position_0
    right_edge = right_position - edge_position_0
    edge_length = wp.length(edge)
    left_normal_raw = wp.cross(left_edge, edge)
    right_normal_raw = wp.cross(edge, right_edge)
    left_normal_length = wp.length(left_normal_raw)
    right_normal_length = wp.length(right_normal_raw)
    if (
        edge_length <= _MIN_GEOMETRY_NORM
        or left_normal_length <= _MIN_GEOMETRY_NORM
        or right_normal_length <= _MIN_GEOMETRY_NORM
    ):
        return False, float(0.0), wp.vec3(0.0), wp.vec3(0.0), wp.vec3(0.0), wp.vec3(0.0)

    inverse_edge_length = 1.0 / edge_length
    edge_direction = edge * inverse_edge_length
    left_normal = left_normal_raw / left_normal_length
    right_normal = right_normal_raw / right_normal_length
    sine = wp.clamp(wp.dot(wp.cross(left_normal, right_normal), edge_direction), -1.0, 1.0)
    cosine = wp.clamp(wp.dot(left_normal, right_normal), -1.0, 1.0)
    angle = wp.atan2(sine, cosine)

    inverse_edge_length_squared = inverse_edge_length * inverse_edge_length
    left_projection = wp.dot(edge, left_edge) * inverse_edge_length_squared
    right_projection = wp.dot(edge, right_edge) * inverse_edge_length_squared
    left_height = wp.length(left_edge - left_projection * edge)
    right_height = wp.length(right_edge - right_projection * edge)
    if left_height <= _MIN_GEOMETRY_NORM or right_height <= _MIN_GEOMETRY_NORM:
        return False, float(0.0), wp.vec3(0.0), wp.vec3(0.0), wp.vec3(0.0), wp.vec3(0.0)

    inverse_left_height = 1.0 / left_height
    inverse_right_height = 1.0 / right_height
    gradient_0 = (left_projection - 1.0) * inverse_left_height * left_normal + (
        right_projection - 1.0
    ) * inverse_right_height * right_normal
    gradient_1 = (
        -left_projection * inverse_left_height * left_normal - right_projection * inverse_right_height * right_normal
    )
    gradient_2 = inverse_left_height * left_normal
    gradient_3 = inverse_right_height * right_normal
    return True, angle, gradient_0, gradient_1, gradient_2, gradient_3


@wp.kernel
def _accumulate_dihedral_bending_force(
    dihedral_indices: wp.array2d[int],
    rest_angles: wp.array[float],
    stiffness: float,
    positions: wp.array[wp.vec3],
    forces: wp.array[wp.vec3],
):
    dihedral = wp.tid()
    particle_0 = dihedral_indices[dihedral, 0]
    particle_1 = dihedral_indices[dihedral, 1]
    particle_2 = dihedral_indices[dihedral, 2]
    particle_3 = dihedral_indices[dihedral, 3]
    valid, angle, gradient_0, gradient_1, gradient_2, gradient_3 = _dihedral_frame(
        positions[particle_0],
        positions[particle_1],
        positions[particle_2],
        positions[particle_3],
    )
    if not valid:
        return

    angle_difference = angle - rest_angles[dihedral]
    residual = wp.atan2(wp.sin(angle_difference), wp.cos(angle_difference))
    scale = stiffness * residual
    wp.atomic_sub(forces, particle_0, scale * gradient_0)
    wp.atomic_sub(forces, particle_1, scale * gradient_1)
    wp.atomic_sub(forces, particle_2, scale * gradient_2)
    wp.atomic_sub(forces, particle_3, scale * gradient_3)


@wp.kernel
def _accumulate_dihedral_bending_force_and_hessian(
    dihedral_indices: wp.array2d[int],
    rest_angles: wp.array[float],
    stiffness: float,
    hessian_block_indices: wp.array2d[int],
    positions: wp.array[wp.vec3],
    forces: wp.array[wp.vec3],
    hessian_values: wp.array[wp.mat33],
):
    dihedral = wp.tid()
    particle_0 = dihedral_indices[dihedral, 0]
    particle_1 = dihedral_indices[dihedral, 1]
    particle_2 = dihedral_indices[dihedral, 2]
    particle_3 = dihedral_indices[dihedral, 3]
    valid, angle, gradient_0, gradient_1, gradient_2, gradient_3 = _dihedral_frame(
        positions[particle_0],
        positions[particle_1],
        positions[particle_2],
        positions[particle_3],
    )
    if not valid:
        return

    angle_difference = angle - rest_angles[dihedral]
    residual = wp.atan2(wp.sin(angle_difference), wp.cos(angle_difference))
    force_scale = stiffness * residual
    for local_i in range(4):
        gradient_i = gradient_0
        if local_i == 1:
            gradient_i = gradient_1
        elif local_i == 2:
            gradient_i = gradient_2
        elif local_i == 3:
            gradient_i = gradient_3
        particle_i = dihedral_indices[dihedral, local_i]
        wp.atomic_sub(forces, particle_i, force_scale * gradient_i)

        for local_j in range(4):
            gradient_j = gradient_0
            if local_j == 1:
                gradient_j = gradient_1
            elif local_j == 2:
                gradient_j = gradient_2
            elif local_j == 3:
                gradient_j = gradient_3
            block = stiffness * wp.outer(gradient_i, gradient_j)
            wp.atomic_add(hessian_values, hessian_block_indices[dihedral, 4 * local_i + local_j], block)


class ConstraintDihedralBending:
    """A batch of four-particle dihedral-angle bending constraints."""

    def __init__(
        self,
        dihedral_indices: Sequence[tuple[int, int, int, int]],
        rest_positions: Sequence[wp.vec3],
        stiffness: float,
        particle_count: int,
        device: Any,
    ):
        """Create a dihedral-angle bending constraint batch.

        Args:
            dihedral_indices: Shared-edge endpoints followed by the left and
                right opposite particle indices for each dihedral.
            rest_positions: Particle positions used to compute rest angles [m].
            stiffness: Shared positive bending stiffness [N·m].
            particle_count: Number of particles in the associated model.
            device: Warp device storing runtime arrays.
        """
        if particle_count <= 0:
            raise ValueError("particle_count must be positive")
        if not np.isfinite(stiffness) or stiffness <= 0.0:
            raise ValueError("stiffness must be finite and positive")

        self.host_dihedral_indices = tuple(tuple(int(index) for index in dihedral) for dihedral in dihedral_indices)
        if not self.host_dihedral_indices:
            raise ValueError("dihedral_indices must be nonempty")
        for dihedral in self.host_dihedral_indices:
            if len(dihedral) != 4 or len(set(dihedral)) != 4:
                raise ValueError("Dihedrals must contain exactly four distinct particle indices")
            if any(index < 0 or index >= particle_count for index in dihedral):
                raise ValueError(f"Dihedral {dihedral} is outside particle_count={particle_count}")

        host_rest_positions = np.asarray(rest_positions, dtype=np.float32)
        if host_rest_positions.shape != (particle_count, 3):
            raise ValueError(f"Expected {particle_count} rest-position rows")
        if not np.isfinite(host_rest_positions).all():
            raise ValueError("rest_positions must be finite")
        host_rest_angles = tuple(
            _host_signed_dihedral_angle(host_rest_positions, dihedral) for dihedral in self.host_dihedral_indices
        )

        self.particle_count = particle_count
        self.stiffness = float(stiffness)
        self.device = wp.get_device(device)
        self.dihedral_indices = wp.array2d(self.host_dihedral_indices, dtype=int, device=self.device)
        self.rest_angles = wp.array(host_rest_angles, dtype=float, device=self.device)
        self.hessian_block_indices: wp.array2d[int] | None = None
        self.hessian_value_count: int | None = None

    def append_hessian_structure(self, builder: BlockCsrBuilder) -> None:
        """Append all sixteen ordered particle-pair blocks per dihedral."""
        if builder.row_count != self.particle_count:
            raise ValueError("Constraint and block matrix particle counts differ")
        for dihedral in self.host_dihedral_indices:
            for particle_i in dihedral:
                for particle_j in dihedral:
                    builder.ensure_block(particle_i, particle_j)

    def bind_hessian(self, matrix: BlockCsrMatrix) -> None:
        """Bind dihedral blocks to finalized block-CSR value indices."""
        if matrix.row_count != self.particle_count or matrix.device != self.device:
            raise ValueError("Constraint and block matrix must have matching particle counts and devices")
        block_indices = [
            tuple(matrix.block_index(particle_i, particle_j) for particle_i in dihedral for particle_j in dihedral)
            for dihedral in self.host_dihedral_indices
        ]
        self.hessian_block_indices = wp.array2d(block_indices, dtype=int, device=self.device)
        self.hessian_value_count = len(matrix.values)

    def accumulate_force(self, positions: wp.array[wp.vec3], output: wp.array[wp.vec3]) -> None:
        """Add bending forces evaluated at ``positions`` to ``output``."""
        self._validate_runtime_arrays(positions, output)
        wp.launch(
            _accumulate_dihedral_bending_force,
            dim=len(self.rest_angles),
            inputs=[self.dihedral_indices, self.rest_angles, self.stiffness, positions],
            outputs=[output],
            device=self.device,
        )

    def accumulate_force_and_hessian(
        self,
        positions: wp.array[wp.vec3],
        force_output: wp.array[wp.vec3],
        hessian_values: wp.array[wp.mat33],
    ) -> None:
        """Add exact force and Gauss-Newton positive-semidefinite Hessian blocks."""
        self._validate_runtime_arrays(positions, force_output)
        if self.hessian_block_indices is None:
            raise RuntimeError("bind_hessian() must be called before Hessian assembly")
        if hessian_values.device != self.device:
            raise ValueError("Constraint and Hessian values must use the same device")
        if len(hessian_values) != self.hessian_value_count:
            raise ValueError(f"Expected {self.hessian_value_count} Hessian blocks")
        wp.launch(
            _accumulate_dihedral_bending_force_and_hessian,
            dim=len(self.rest_angles),
            inputs=[
                self.dihedral_indices,
                self.rest_angles,
                self.stiffness,
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
