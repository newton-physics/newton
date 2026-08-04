# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Composition for matrix-free LIMX dynamic constraints."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import warp as wp


class ConstraintGroupDynamic:
    """Compose matrix-free dynamic constraints over one particle domain."""

    def __init__(self, constraints: Sequence[Any]):
        """Create an ordered dynamic constraint group.

        Args:
            constraints: Dynamic constraints sharing one particle count and device.
        """
        self.constraints = tuple(constraints)
        if not self.constraints:
            raise ValueError("constraints must not be empty")

        self.particle_count = self.constraints[0].particle_count
        self.device = wp.get_device(self.constraints[0].device)
        for constraint in self.constraints:
            if constraint.particle_count != self.particle_count:
                raise ValueError("Every dynamic constraint must use the same particle count")
            if wp.get_device(constraint.device) != self.device:
                raise ValueError("Every dynamic constraint must use the same device")

    def bind_static_system(
        self,
        static_diagonal: wp.array[wp.mat33],
        masses: wp.array[float],
    ) -> None:
        """Bind assembled static-system data to children that consume it.

        Args:
            static_diagonal: Current assembled diagonal blocks [N/m], shape
                ``[particle_count, 3, 3]``.
            masses: Particle masses [kg], shape ``[particle_count]``.
        """
        for constraint in self.constraints:
            bind_static_system = getattr(constraint, "bind_static_system", None)
            if bind_static_system is not None:
                bind_static_system(static_diagonal, masses)

    def begin_step(
        self,
        positions: wp.array[wp.vec3],
        velocities: wp.array[wp.vec3],
        dt: float,
    ) -> None:
        """Forward time-step preparation to every child in order.

        Args:
            positions: Step-start particle positions [m], shape ``[particle_count, 3]``.
            velocities: Step-start particle velocities [m/s], shape ``[particle_count, 3]``.
            dt: Simulation time step [s].
        """
        for constraint in self.constraints:
            constraint.begin_step(positions, velocities, dt)

    def prepare(self, positions: wp.array[wp.vec3]) -> None:
        """Prepare every child for the current Newton linearization.

        Args:
            positions: Linearization positions [m], shape ``[particle_count, 3]``.
        """
        for constraint in self.constraints:
            constraint.prepare(positions)

    def accumulate_force(self, positions: wp.array[wp.vec3], output: wp.array[wp.vec3]) -> None:
        """Accumulate every child's physical force in order.

        Args:
            positions: Linearization positions [m], shape ``[particle_count, 3]``.
            output: Force accumulation buffer [N], shape ``[particle_count, 3]``.
        """
        for constraint in self.constraints:
            constraint.accumulate_force(positions, output)

    def hessian_multiply(
        self,
        positions: wp.array[wp.vec3],
        vector: wp.array[wp.vec3],
        output: wp.array[wp.vec3],
    ) -> None:
        """Accumulate every child's Hessian-vector product in order.

        Args:
            positions: Linearization positions [m], shape ``[particle_count, 3]``.
            vector: Particle-space input vector, shape ``[particle_count, 3]``.
            output: Particle-space accumulation buffer, shape ``[particle_count, 3]``.
        """
        for constraint in self.constraints:
            constraint.hessian_multiply(positions, vector, output)

    def accumulate_diagonal(self, positions: wp.array[wp.vec3], output: wp.array[wp.mat33]) -> None:
        """Accumulate every child's exact diagonal blocks in order.

        Args:
            positions: Linearization positions [m], shape ``[particle_count, 3]``.
            output: Hessian block accumulation buffer [N/m], shape ``[particle_count, 3, 3]``.
        """
        for constraint in self.constraints:
            constraint.accumulate_diagonal(positions, output)
