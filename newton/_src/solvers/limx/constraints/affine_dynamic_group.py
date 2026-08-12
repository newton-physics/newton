# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Composition for matrix-free LIMX affine dynamic constraints."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import warp as wp

from ..affine_types import mat1212, vec12


class ConstraintGroupAffine:
    """Compose matrix-free dynamic constraints over one affine-body domain."""

    def __init__(self, constraints: Sequence[Any]):
        """Create an ordered affine dynamic constraint group.

        Args:
            constraints: Dynamic constraints sharing one affine body count and device.
        """
        self.constraints = tuple(constraints)
        if not self.constraints:
            raise ValueError("constraints must not be empty")

        self.body_count = self.constraints[0].body_count
        self.device = wp.get_device(self.constraints[0].device)
        for constraint in self.constraints:
            if constraint.body_count != self.body_count:
                raise ValueError("Every affine dynamic constraint must use the same body count")
            if wp.get_device(constraint.device) != self.device:
                raise ValueError("Every affine dynamic constraint must use the same device")

    def begin_step(self, q: wp.array[vec12], qd: wp.array[vec12], dt: float) -> None:
        """Forward time-step preparation to every child in order.

        Args:
            q: Step-start affine generalized states.
            qd: Step-start affine generalized velocities [m/s, 1/s].
            dt: Simulation time step [s].
        """
        for constraint in self.constraints:
            constraint.begin_step(q, qd, dt)

    def prepare(self, q: wp.array[vec12]) -> None:
        """Prepare every child for the current Newton linearization.

        Args:
            q: Current affine generalized states.
        """
        for constraint in self.constraints:
            constraint.prepare(q)

    def accumulate_force(self, q: wp.array[vec12], output: wp.array[vec12]) -> None:
        """Accumulate every child's affine generalized force.

        Args:
            q: Current affine generalized states.
            output: Affine generalized force accumulation buffer [N, N·m].
        """
        for constraint in self.constraints:
            constraint.accumulate_force(q, output)

    def multiply(
        self,
        particle_input: wp.array[wp.vec3],
        affine_input: wp.array[vec12],
        particle_output: wp.array[wp.vec3],
        affine_output: wp.array[vec12],
    ) -> None:
        """Accumulate every child's mixed Hessian-vector product."""
        for constraint in self.constraints:
            constraint.multiply(particle_input, affine_input, particle_output, affine_output)

    def accumulate_diagonal(
        self,
        particle_diagonal: wp.array[wp.mat33],
        affine_diagonal: wp.array[mat1212],
    ) -> None:
        """Accumulate every child's native block-Jacobi diagonal."""
        for constraint in self.constraints:
            constraint.accumulate_diagonal(particle_diagonal, affine_diagonal)
