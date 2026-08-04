# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Projected-Newton particle solver using block-CSR elasticity and PCG."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import warp as wp

from ...core.types import override
from ...geometry import ParticleFlags
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
    rhs[particle] = masses[particle] * inv_dt_squared * (inertia_positions[particle] - iterate_positions[particle])


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
    r"""Implicit projected-Newton particle solver.

    Static constraints assemble forces and analytic positive-semidefinite
    Hessian blocks at the current Newton iterate. Their fixed topology is
    stored in a ``3 x 3`` block-CSR matrix whose values are rebuilt before
    every PCG solve. Dynamic constraints, such as future collision terms, can
    add matrix-free force, Hessian-vector, and diagonal contributions through
    ``dynamic_operator``.
    """

    def __init__(
        self,
        model: Model,
        constraints: Sequence[Any],
        nonlinear_iterations: int = 4,
        linear_iterations: int = 32,
        velocity_damping: float = 1.0,
        dynamic_operator: Any | None = None,
    ):
        """Create a LIMX projected-Newton particle solver.

        Args:
            model: Particle-only model containing active, positive-mass particles.
            constraints: Static constraint batches that provide current-position
                force and Hessian assembly methods.
            nonlinear_iterations: Newton position iterations per step.
            linear_iterations: Fixed PCG iterations per Newton iteration.
            velocity_damping: Per-step velocity multiplier.
            dynamic_operator: Optional matrix-free dynamic constraint operator.
        """
        super().__init__(model)
        if model.body_count > 0:
            raise ValueError("SolverLIMX is particle-only and does not accept rigid bodies")
        if model.particle_count <= 0 or model.particle_mass is None:
            raise ValueError("SolverLIMX requires at least one particle")
        masses = model.particle_mass.numpy()
        if not np.isfinite(masses).all() or np.any(masses <= 0.0):
            raise ValueError("SolverLIMX requires finite positive particle masses")
        flags = model.particle_flags.numpy()
        if np.any((flags & ParticleFlags.ACTIVE) == 0):
            raise ValueError("SolverLIMX requires active particles; use ConstraintAnchor to fix particle positions")
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
        self.dynamic_operator = dynamic_operator if dynamic_operator is not None else EmptyDynamicConstraintOperator()

        matrix_builder = BlockCsrBuilder(model.particle_count)
        for constraint in self.constraints:
            if getattr(constraint, "particle_count", None) != model.particle_count:
                raise ValueError("Every constraint must match the model particle count")
            if getattr(constraint, "device", None) != self.device:
                raise ValueError("Every constraint must use the model device")
            constraint.append_hessian_structure(matrix_builder)
        self.static_matrix = matrix_builder.finalize(self.device)
        for constraint in self.constraints:
            constraint.bind_hessian(self.static_matrix)

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
        self.increment = wp.zeros_like(self.previous_positions)

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

        begin_dynamic_step = getattr(self.dynamic_operator, "begin_step", None)
        if begin_dynamic_step is not None:
            begin_dynamic_step(state_in.particle_q, state_in.particle_qd, dt)

        inv_dt_squared = 1.0 / (dt * dt)
        for nonlinear_iteration in range(self.nonlinear_iterations):
            prepare_dynamic_constraints = getattr(self.dynamic_operator, "prepare", None)
            if prepare_dynamic_constraints is not None:
                prepare_dynamic_constraints(self.iterate_positions)
            self.static_matrix.clear_values()
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
                constraint.accumulate_force_and_hessian(
                    self.iterate_positions,
                    self.rhs,
                    self.static_matrix.values,
                )
            self.dynamic_operator.accumulate_force(self.iterate_positions, self.rhs)

            self.static_matrix.update_diagonal()
            self.operator.prepare(self.iterate_positions, dt)
            self.linear_solver.solve(
                self.operator,
                self.rhs,
                self.increment,
                iterations=self.linear_iterations,
                zero_initial_guess=nonlinear_iteration > 0,
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
