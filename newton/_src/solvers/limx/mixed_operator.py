# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Heterogeneous particle-affine linear operators for LIMX."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import warp as wp

from .affine_types import mat1212, vec12
from .block_csr_12 import BlockCsrMatrix12
from .mixed_linear_solver import _factor_affine_diagonal
from .operator import CompositeLinearOperator


@wp.kernel
def _invert_particle_diagonal(diagonal: wp.array[wp.mat33], inverse_diagonal: wp.array[wp.mat33]):
    row = wp.tid()
    inverse_diagonal[row] = wp.inverse(diagonal[row])


@dataclass
class MixedVector3x12:
    """Pair native particle and affine generalized vectors."""

    particle: wp.array[wp.vec3]
    affine: wp.array[vec12]


class EmptyMixedDynamicOperator:
    """A matrix-free mixed operator that contributes nothing."""

    def begin_step(self, q: wp.array[vec12], qd: wp.array[vec12], dt: float) -> None:
        """Leave affine step-start data unused."""

    def prepare(self, q: wp.array[vec12]) -> None:
        """Leave affine linearization data unused."""

    def accumulate_force(self, q: wp.array[vec12], output: wp.array[vec12]) -> None:
        """Leave affine force output unchanged."""

    def multiply(
        self,
        particle_input: wp.array[wp.vec3],
        affine_input: wp.array[vec12],
        particle_output: wp.array[wp.vec3],
        affine_output: wp.array[vec12],
    ) -> None:
        """Leave both outputs unchanged."""

    def accumulate_diagonal(
        self,
        particle_diagonal: wp.array[wp.mat33],
        affine_diagonal: wp.array[mat1212],
    ) -> None:
        """Leave both block diagonals unchanged."""


class MixedLinearOperator:
    """Combine native particle and affine operators with matrix-free coupling."""

    def __init__(
        self,
        particle_operator: CompositeLinearOperator | None,
        affine_matrix: BlockCsrMatrix12 | None,
        mixed_dynamic_operator: Any,
        device: Any,
    ):
        self.device = wp.get_device(device)
        if particle_operator is not None and particle_operator.device != self.device:
            raise ValueError("Particle operator and mixed operator must use the same device")
        if affine_matrix is not None and affine_matrix.device != self.device:
            raise ValueError("Affine matrix and mixed operator must use the same device")
        if particle_operator is None and affine_matrix is None:
            raise ValueError("At least one particle or affine operator must be provided")

        self.particle_operator = particle_operator
        self.affine_matrix = affine_matrix
        self.mixed_dynamic_operator = mixed_dynamic_operator
        self.particle_count = 0 if particle_operator is None else particle_operator.particle_count
        self.affine_count = 0 if affine_matrix is None else affine_matrix.row_count
        self._empty_particle_diagonal = wp.empty(0, dtype=wp.mat33, device=self.device)
        self.affine_diagonal = wp.empty(self.affine_count, dtype=mat1212, device=self.device)
        self.affine_factors = wp.empty_like(self.affine_diagonal)
        self.affine_regularization = wp.zeros(self.affine_count, dtype=int, device=self.device)
        self._prepared = False

    def prepare(self, particle_positions: wp.array[wp.vec3] | None, dt: float) -> None:
        """Prepare native block-Jacobi preconditioners for one linearization.

        Args:
            particle_positions: Particle linearization positions [m], or ``None`` when absent.
            dt: Simulation time step [s].
        """
        if self.particle_operator is not None:
            if particle_positions is None:
                raise ValueError("particle_positions are required for a particle operator")
            self.particle_operator.prepare(particle_positions, dt)
        elif particle_positions is not None and len(particle_positions) != 0:
            raise ValueError("particle_positions must be empty without a particle operator")
        elif dt <= 0.0:
            raise ValueError("dt must be positive")

        if self.affine_matrix is not None:
            wp.copy(self.affine_diagonal, self.affine_matrix.diagonal)

        self.mixed_dynamic_operator.accumulate_diagonal(
            self._particle_diagonal(),
            self.affine_diagonal,
        )
        if self.particle_count:
            wp.launch(
                _invert_particle_diagonal,
                dim=self.particle_count,
                inputs=[self.particle_operator.diagonal],
                outputs=[self.particle_operator.inverse_diagonal],
                device=self.device,
            )
        if self.affine_count:
            wp.launch(
                _factor_affine_diagonal,
                dim=self.affine_count,
                inputs=[self.affine_diagonal],
                outputs=[self.affine_factors, self.affine_regularization],
                device=self.device,
            )
        self._prepared = True

    def multiply(
        self,
        particle_input: wp.array[wp.vec3],
        affine_input: wp.array[vec12],
        particle_output: wp.array[wp.vec3],
        affine_output: wp.array[vec12],
    ) -> None:
        """Compute the complete split Hessian-vector product."""
        if not self._prepared:
            raise RuntimeError("prepare() must be called before multiply()")
        self._validate_particle_vector(particle_input, "particle_input")
        self._validate_particle_vector(particle_output, "particle_output")
        self._validate_affine_vector(affine_input, "affine_input")
        self._validate_affine_vector(affine_output, "affine_output")

        particle_output.zero_()
        affine_output.zero_()
        if self.particle_operator is not None:
            self.particle_operator.multiply(particle_input, particle_output)
        if self.affine_matrix is not None:
            self.affine_matrix.multiply(affine_input, affine_output)
        self.mixed_dynamic_operator.multiply(particle_input, affine_input, particle_output, affine_output)

    def _particle_diagonal(self) -> wp.array[wp.mat33]:
        if self.particle_operator is not None:
            return self.particle_operator.diagonal
        return self._empty_particle_diagonal

    def _validate_particle_vector(self, vector: wp.array[wp.vec3], name: str) -> None:
        if len(vector) != self.particle_count:
            raise ValueError(f"{name} must contain {self.particle_count} vectors")
        if vector.device != self.device:
            raise ValueError(f"{name} must use device {self.device}")

    def _validate_affine_vector(self, vector: wp.array[vec12], name: str) -> None:
        if len(vector) != self.affine_count:
            raise ValueError(f"{name} must contain {self.affine_count} vectors")
        if vector.device != self.device:
            raise ValueError(f"{name} must use device {self.device}")
