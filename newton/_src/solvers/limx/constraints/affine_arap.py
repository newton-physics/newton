# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Batched direct-affine as-rigid-as-possible constraints."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import warp as wp

from ..affine_types import mat1212, vec12
from ..block_csr_12 import BlockCsrBuilder12, BlockCsrMatrix12
from .tetrahedron_arap import (
    _arap_energy,
    _arap_gradient,
    _arap_hessian_unscaled,
    _project_psd,
    mat99,
)


@wp.func
def _affine_matrix(state: vec12) -> wp.mat33:
    """Extract the row-major affine matrix from a generalized state."""
    matrix = wp.mat33(0.0)
    for row in range(3):
        for column in range(3):
            matrix[row, column] = state[3 + 3 * row + column]
    return matrix


@wp.func
def _hessian_to_row_major(hessian: mat99) -> mat99:
    """Permute a column-major deformation Hessian into affine-state order."""
    result = mat99(0.0)
    for row_i in range(3):
        for column_i in range(3):
            state_i = 3 * row_i + column_i
            deformation_i = 3 * column_i + row_i
            for row_j in range(3):
                for column_j in range(3):
                    state_j = 3 * row_j + column_j
                    deformation_j = 3 * column_j + row_j
                    result[state_i, state_j] = hessian[deformation_i, deformation_j]
    return result


@wp.func
def _affine_arap_energy(state: vec12, rigidity: float, volume: float) -> float:
    """Evaluate direct-affine ARAP energy for one generalized state."""
    return _arap_energy(_affine_matrix(state), rigidity, volume)


@wp.func
def _affine_arap_hessian_unscaled(state: vec12) -> mat99:
    """Evaluate the unscaled ARAP Hessian in row-major affine-state order."""
    return _hessian_to_row_major(_arap_hessian_unscaled(_affine_matrix(state)))


@wp.kernel
def _accumulate_affine_arap_force_and_hessian(
    rigidities: wp.array[float],
    volumes: wp.array[float],
    hessian_block_indices: wp.array[int],
    states: wp.array[vec12],
    forces: wp.array[vec12],
    hessian_values: wp.array[mat1212],
):
    body = wp.tid()
    state = states[body]
    rigidity = rigidities[body]
    volume = volumes[body]
    gradient = _arap_gradient(_affine_matrix(state), rigidity, volume)
    generalized_force = vec12(0.0)
    for row in range(3):
        for column in range(3):
            generalized_force[3 + 3 * row + column] = -gradient[row, column]
    forces[body] += generalized_force

    affine_hessian = rigidity * volume * _project_psd(_affine_arap_hessian_unscaled(state))
    block = mat1212(0.0)
    for row in range(9):
        for column in range(9):
            block[3 + row, 3 + column] = affine_hessian[row, column]
    hessian_values[hessian_block_indices[body]] += block


class ConstraintAffineARAP:
    """A batch of direct-affine as-rigid-as-possible constraints."""

    def __init__(
        self,
        rigidities: Sequence[float],
        volumes: Sequence[float],
        body_count: int,
        device: Any,
    ):
        """Create direct-affine ARAP constraints.

        Args:
            rigidities: Non-negative ARAP rigidity per affine body.
            volumes: Positive rest volume per affine body [m^3].
            body_count: Number of affine bodies.
            device: Warp device storing runtime arrays.
        """
        if not isinstance(body_count, (int, np.integer)) or body_count <= 0:
            raise ValueError("body_count must be positive")
        host_rigidities = np.asarray(rigidities, dtype=np.float64)
        host_volumes = np.asarray(volumes, dtype=np.float64)
        if host_rigidities.shape != (body_count,) or host_volumes.shape != (body_count,):
            raise ValueError("Rigidities and volumes must contain one value per affine body")
        if not np.isfinite(host_rigidities).all() or np.any(host_rigidities < 0.0):
            raise ValueError("Rigidities must be finite and non-negative")
        if not np.isfinite(host_volumes).all() or np.any(host_volumes <= 0.0):
            raise ValueError("Volumes must be finite and positive")

        self.body_count = int(body_count)
        self.device = wp.get_device(device)
        self.host_rigidities = tuple(float(value) for value in host_rigidities)
        self.host_volumes = tuple(float(value) for value in host_volumes)
        self.rigidities = wp.array(self.host_rigidities, dtype=float, device=self.device)
        self.volumes = wp.array(self.host_volumes, dtype=float, device=self.device)
        self.hessian_block_indices: wp.array[int] | None = None
        self.hessian_value_count: int | None = None

    def append_hessian_structure(self, builder: BlockCsrBuilder12) -> None:
        """Append one native diagonal block per affine body."""
        if builder.row_count != self.body_count:
            raise ValueError("Constraint and block matrix body counts differ")
        for body in range(self.body_count):
            builder.ensure_block(body, body)

    def bind_hessian(self, matrix: BlockCsrMatrix12) -> None:
        """Bind each affine body to its finalized diagonal block."""
        if matrix.row_count != self.body_count or matrix.device != self.device:
            raise ValueError("Constraint and block matrix must have matching body counts and devices")
        block_indices = [matrix.block_index(body, body) for body in range(self.body_count)]
        self.hessian_block_indices = wp.array(block_indices, dtype=int, device=self.device)
        self.hessian_value_count = len(matrix.values)

    def accumulate_force_and_hessian(
        self,
        q: wp.array[vec12],
        force: wp.array[vec12],
        values: wp.array[mat1212],
    ) -> None:
        """Add direct-affine ARAP forces and projected Hessian blocks."""
        if q.dtype != vec12 or force.dtype != vec12:
            raise ValueError("Affine states and forces must contain vec12 rows")
        if len(q) != self.body_count or len(force) != self.body_count:
            raise ValueError(f"Expected {self.body_count} affine body rows")
        if q.device != self.device or force.device != self.device:
            raise ValueError("Constraint and affine runtime arrays must use the same device")
        if self.hessian_block_indices is None:
            raise RuntimeError("bind_hessian() must be called before Hessian assembly")
        if values.dtype != mat1212:
            raise ValueError("Hessian values must contain mat1212 blocks")
        if values.device != self.device:
            raise ValueError("Constraint and Hessian values must use the same device")
        if len(values) != self.hessian_value_count:
            raise ValueError(f"Expected {self.hessian_value_count} Hessian blocks")

        wp.launch(
            _accumulate_affine_arap_force_and_hessian,
            dim=self.body_count,
            inputs=[self.rigidities, self.volumes, self.hessian_block_indices, q],
            outputs=[force, values],
            device=self.device,
        )
