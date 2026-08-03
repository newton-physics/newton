# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Linear operators for LIMX particle systems."""

from __future__ import annotations

from typing import Any

import warp as wp

from .block_csr import BlockCsrMatrix


@wp.kernel
def _initialize_diagonal(
    masses: wp.array[float],
    static_diagonal: wp.array[wp.mat33],
    inv_dt_squared: float,
    diagonal: wp.array[wp.mat33],
):
    particle = wp.tid()
    diagonal[particle] = static_diagonal[particle] + masses[particle] * inv_dt_squared * wp.identity(3, float)


@wp.kernel
def _invert_diagonal(diagonal: wp.array[wp.mat33], inverse_diagonal: wp.array[wp.mat33]):
    particle = wp.tid()
    inverse_diagonal[particle] = wp.inverse(diagonal[particle])


@wp.kernel
def _add_mass_multiply(
    masses: wp.array[float],
    inv_dt_squared: float,
    vector: wp.array[wp.vec3],
    output: wp.array[wp.vec3],
):
    particle = wp.tid()
    output[particle] += masses[particle] * inv_dt_squared * vector[particle]


class EmptyDynamicConstraintOperator:
    """A matrix-free dynamic constraint operator that contributes nothing."""

    def prepare(self, positions: wp.array[wp.vec3]) -> None:
        """Perform no preparation for the current Newton linearization."""

    def accumulate_force(self, positions: wp.array[wp.vec3], output: wp.array[wp.vec3]) -> None:
        """Leave ``output`` unchanged."""

    def hessian_multiply(
        self,
        positions: wp.array[wp.vec3],
        vector: wp.array[wp.vec3],
        output: wp.array[wp.vec3],
    ) -> None:
        """Leave ``output`` unchanged."""

    def accumulate_diagonal(self, positions: wp.array[wp.vec3], output: wp.array[wp.mat33]) -> None:
        """Leave ``output`` unchanged."""


class CompositeLinearOperator:
    """Combine inertia, assembled block-CSR elasticity, and matrix-free Hessians."""

    def __init__(
        self,
        masses: wp.array[float],
        static_matrix: BlockCsrMatrix,
        dynamic_operator: Any,
        device: Any,
    ):
        self.device = wp.get_device(device)
        self.particle_count = len(masses)
        if self.particle_count == 0:
            raise ValueError("masses must not be empty")
        if static_matrix.row_count != self.particle_count:
            raise ValueError("Mass and static matrix dimensions must match")
        if masses.device != self.device or static_matrix.device != self.device:
            raise ValueError("Masses, static matrix, and operator must use the same device")

        self.masses = masses
        self.static_matrix = static_matrix
        self.dynamic_operator = dynamic_operator
        self.diagonal = wp.empty(self.particle_count, dtype=wp.mat33, device=self.device)
        self.inverse_diagonal = wp.empty_like(self.diagonal)
        self._positions: wp.array[wp.vec3] | None = None
        self._inv_dt_squared = 0.0

    def prepare(self, positions: wp.array[wp.vec3], dt: float) -> None:
        """Prepare the current operator and its block-Jacobi inverse diagonal.

        Args:
            positions: Linearization positions [m], shape ``[particle_count, 3]``.
            dt: Simulation time step [s].
        """
        if dt <= 0.0:
            raise ValueError("dt must be positive")
        self._validate_vector(positions, "positions")

        self._positions = positions
        self._inv_dt_squared = 1.0 / (dt * dt)
        wp.launch(
            _initialize_diagonal,
            dim=self.particle_count,
            inputs=[self.masses, self.static_matrix.diagonal, self._inv_dt_squared],
            outputs=[self.diagonal],
            device=self.device,
        )
        self.dynamic_operator.accumulate_diagonal(positions, self.diagonal)
        wp.launch(
            _invert_diagonal,
            dim=self.particle_count,
            inputs=[self.diagonal],
            outputs=[self.inverse_diagonal],
            device=self.device,
        )

    def multiply(self, vector: wp.array[wp.vec3], output: wp.array[wp.vec3]) -> None:
        """Compute the complete Hessian-vector product."""
        if self._positions is None:
            raise RuntimeError("prepare() must be called before multiply()")
        self._validate_vector(vector, "vector")
        self._validate_vector(output, "output")

        self.static_matrix.multiply(vector, output)
        wp.launch(
            _add_mass_multiply,
            dim=self.particle_count,
            inputs=[self.masses, self._inv_dt_squared, vector],
            outputs=[output],
            device=self.device,
        )
        self.dynamic_operator.hessian_multiply(self._positions, vector, output)

    def _validate_vector(self, vector: wp.array[wp.vec3], name: str) -> None:
        if len(vector) != self.particle_count:
            raise ValueError(f"{name} must contain {self.particle_count} vectors")
        if vector.device != self.device:
            raise ValueError(f"{name} must use device {self.device}")
