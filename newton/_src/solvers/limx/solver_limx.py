# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Implicit particle solver using decoupled LIMX constraints and PCG."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import warp as wp

from ...core.types import override
from ...sim import Contacts, Control, Model, State
from ..solver import SolverBase
from .block_csr import BlockCsrBuilder
from .linear_solver import PcgSolver
from .operator import CompositeLinearOperator, EmptyDynamicConstraintOperator


@wp.kernel
def _initialize_step(
    positions: wp.array[wp.vec3],
    velocities: wp.array[wp.vec3],
    external_forces: wp.array[wp.vec3],
    masses: wp.array[float],
    particle_world: wp.array[int],
    gravity: wp.array[wp.vec3],
    dt: float,
    previous_positions: wp.array[wp.vec3],
    inertia_positions: wp.array[wp.vec3],
    iterate_positions: wp.array[wp.vec3],
):
    particle = wp.tid()
    position = positions[particle]
    acceleration = gravity[wp.max(particle_world[particle], 0)] + external_forces[particle] / masses[particle]
    previous_positions[particle] = position
    inertia_positions[particle] = position + dt * velocities[particle] + dt * dt * acceleration
    iterate_positions[particle] = position


@wp.kernel
def _initialize_rhs(
    masses: wp.array[float],
    inv_dt_squared: float,
    inertia_positions: wp.array[wp.vec3],
    iterate_positions: wp.array[wp.vec3],
    rhs: wp.array[wp.vec3],
):
    particle = wp.tid()
    rhs[particle] = masses[particle] * inv_dt_squared * (
        inertia_positions[particle] - iterate_positions[particle]
    )


@wp.kernel
def _apply_increment(increment: wp.array[wp.vec3], iterate_positions: wp.array[wp.vec3]):
    particle = wp.tid()
    iterate_positions[particle] += increment[particle]


@wp.kernel
def _finish_step(
    previous_positions: wp.array[wp.vec3],
    iterate_positions: wp.array[wp.vec3],
    inv_dt: float,
    velocity_damping: float,
    output_positions: wp.array[wp.vec3],
    output_velocities: wp.array[wp.vec3],
):
    particle = wp.tid()
    output_positions[particle] = iterate_positions[particle]
    output_velocities[particle] = (
        velocity_damping * inv_dt * (iterate_positions[particle] - previous_positions[particle])
    )


class SolverLIMX(SolverBase):
    r"""Implicit particle solver with block-CSR elasticity and matrix-free dynamics.

    Static constraints evaluate their own forces and assemble a fixed
    projective-dynamics Hessian approximation. The resulting ``3 x 3``
    block-CSR system is solved independently with block-Jacobi PCG. Dynamic
    constraints, such as future collision terms, can add matrix-free force,
    Hessian-vector, and diagonal contributions through ``dynamic_operator``.
    """

    def __init__(
        self,
        model: Model,
        constraints: Sequence[Any],
        nonlinear_iterations: int = 4,
        linear_iterations: int = 32,
        velocity_damping: float = 0.998,
        dynamic_operator: Any | None = None,
    ):
        """Create a LIMX particle solver.

        Args:
            model: Model containing the particles to integrate.
            constraints: Static constraint batches that provide force and
                fixed-Hessian assembly methods.
            nonlinear_iterations: Nonlinear position iterations per step.
            linear_iterations: Fixed PCG iterations per nonlinear iteration.
            velocity_damping: Per-step velocity multiplier.
            dynamic_operator: Optional matrix-free dynamic constraint operator.
        """
        super().__init__(model)
        if model.particle_count <= 0 or model.particle_mass is None:
            raise ValueError("SolverLIMX requires at least one particle")
        masses = model.particle_mass.numpy()
        if not np.isfinite(masses).all() or np.any(masses <= 0.0):
            raise ValueError("SolverLIMX requires finite positive particle masses")
        if nonlinear_iterations <= 0:
            raise ValueError("nonlinear_iterations must be positive")
        if linear_iterations <= 0:
            raise ValueError("linear_iterations must be positive")
        if not np.isfinite(velocity_damping) or velocity_damping < 0.0 or velocity_damping > 1.0:
            raise ValueError("velocity_damping must be finite and between zero and one")

        self.constraints = tuple(constraints)
        self.nonlinear_iterations = nonlinear_iterations
        self.linear_iterations = linear_iterations
        self.velocity_damping = float(velocity_damping)
        self.dynamic_operator = (
            dynamic_operator if dynamic_operator is not None else EmptyDynamicConstraintOperator()
        )

        matrix_builder = BlockCsrBuilder(model.particle_count)
        for constraint in self.constraints:
            if getattr(constraint, "particle_count", None) != model.particle_count:
                raise ValueError("Every constraint must match the model particle count")
            if getattr(constraint, "device", None) != self.device:
                raise ValueError("Every constraint must use the model device")
            constraint.append_hessian(matrix_builder)
        self.static_matrix = matrix_builder.finalize(self.device)
        self.operator = CompositeLinearOperator(
            masses=model.particle_mass,
            static_matrix=self.static_matrix,
            dynamic_operator=self.dynamic_operator,
            device=self.device,
        )
        self.linear_solver = PcgSolver(model.particle_count, self.device)

        self.previous_positions = wp.empty(model.particle_count, dtype=wp.vec3, device=self.device)
        self.inertia_positions = wp.empty_like(self.previous_positions)
        self.iterate_positions = wp.empty_like(self.previous_positions)
        self.rhs = wp.empty_like(self.previous_positions)
        self.increment = wp.empty_like(self.previous_positions)

    @override
    def step(
        self,
        state_in: State,
        state_out: State,
        control: Control | None,
        contacts: Contacts | None,
        dt: float,
    ) -> None:
        """Advance particles by one implicit-Euler time step.

        ``control`` and ``contacts`` are currently unused. Collision constraints
        can be supplied through the solver's matrix-free dynamic operator.

        Args:
            state_in: Input state, which remains unchanged.
            state_out: State receiving updated particle positions and velocities.
            control: Unused control input.
            contacts: Unused Newton contact data.
            dt: Simulation time step [s].
        """
        if dt <= 0.0:
            raise ValueError("dt must be positive")

        model = self.model
        wp.launch(
            _initialize_step,
            dim=model.particle_count,
            inputs=[
                state_in.particle_q,
                state_in.particle_qd,
                state_in.particle_f,
                model.particle_mass,
                model.particle_world,
                model.gravity,
                dt,
            ],
            outputs=[self.previous_positions, self.inertia_positions, self.iterate_positions],
            device=self.device,
        )

        inv_dt_squared = 1.0 / (dt * dt)
        for _ in range(self.nonlinear_iterations):
            wp.launch(
                _initialize_rhs,
                dim=model.particle_count,
                inputs=[
                    model.particle_mass,
                    inv_dt_squared,
                    self.inertia_positions,
                    self.iterate_positions,
                ],
                outputs=[self.rhs],
                device=self.device,
            )
            for constraint in self.constraints:
                constraint.accumulate_force(self.iterate_positions, self.rhs)
            self.dynamic_operator.accumulate_force(self.iterate_positions, self.rhs)

            self.operator.prepare(self.iterate_positions, dt)
            self.linear_solver.solve(
                self.operator,
                self.rhs,
                self.increment,
                iterations=self.linear_iterations,
            )
            wp.launch(
                _apply_increment,
                dim=model.particle_count,
                inputs=[self.increment],
                outputs=[self.iterate_positions],
                device=self.device,
            )

        wp.launch(
            _finish_step,
            dim=model.particle_count,
            inputs=[self.previous_positions, self.iterate_positions, 1.0 / dt, self.velocity_damping],
            outputs=[state_out.particle_q, state_out.particle_qd],
            device=self.device,
        )
