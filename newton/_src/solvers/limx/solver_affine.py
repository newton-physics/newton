# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Collision-free projected-Newton stepping for LIMX affine bodies."""

from __future__ import annotations

import numpy as np
import warp as wp

from .affine_body import AffineBodyModel
from .affine_types import mat1212, vec12
from .block_csr_12 import BlockCsrBuilder12
from .constraints.affine_arap import ConstraintAffineARAP
from .mixed_linear_solver import MixedPcgSolver
from .mixed_operator import EmptyMixedDynamicOperator, MixedLinearOperator, MixedVector3x12


@wp.kernel
def _initialize_affine_step(
    q: wp.array[vec12],
    qd: wp.array[vec12],
    gravity: wp.array[vec12],
    dt: float,
    previous_q: wp.array[vec12],
    inertia_q: wp.array[vec12],
):
    body = wp.tid()
    state = q[body]
    previous_q[body] = state
    inertia_q[body] = state + dt * qd[body] + dt * dt * gravity[body]


@wp.kernel
def _initialize_affine_system(
    mass_matrices: wp.array[mat1212],
    diagonal_block_indices: wp.array[int],
    inv_dt_squared: float,
    inertia_q: wp.array[vec12],
    q: wp.array[vec12],
    rhs: wp.array[vec12],
    hessian_values: wp.array[mat1212],
):
    body = wp.tid()
    scaled_mass = inv_dt_squared * mass_matrices[body]
    rhs[body] = scaled_mass * (inertia_q[body] - q[body])
    hessian_values[diagonal_block_indices[body]] += scaled_mass


@wp.kernel
def _apply_affine_increment(increment: wp.array[vec12], q: wp.array[vec12]):
    body = wp.tid()
    q[body] += increment[body]


@wp.kernel
def _finish_affine_step(
    previous_q: wp.array[vec12],
    q: wp.array[vec12],
    inv_dt: float,
    velocity_damping: float,
    qd: wp.array[vec12],
):
    body = wp.tid()
    qd[body] = velocity_damping * inv_dt * (q[body] - previous_q[body])


class SolverLIMXAffine:
    """Advance collision-free affine bodies with implicit Euler and Newton's method."""

    def __init__(
        self,
        body_model: AffineBodyModel,
        nonlinear_iterations: int = 4,
        linear_iterations: int = 32,
        velocity_damping: float = 1.0,
    ):
        """Create a collision-free LIMX affine-body solver.

        Args:
            body_model: Affine body mass, gravity, surface, and ARAP data.
            nonlinear_iterations: Newton position iterations per step.
            linear_iterations: Fixed PCG iterations per Newton iteration.
            velocity_damping: Per-step generalized-velocity multiplier.
        """
        if not isinstance(body_model, AffineBodyModel):
            raise TypeError("body_model must be an AffineBodyModel")
        if nonlinear_iterations <= 0:
            raise ValueError("nonlinear_iterations must be positive")
        if linear_iterations <= 0:
            raise ValueError("linear_iterations must be positive")
        if not np.isfinite(velocity_damping) or velocity_damping < 0.0 or velocity_damping > 1.0:
            raise ValueError("velocity_damping must be finite and between zero and one")

        self.body_model = body_model
        self.device = body_model.device
        self.body_count = body_model.body_count
        self.nonlinear_iterations = nonlinear_iterations
        self.linear_iterations = linear_iterations
        self.velocity_damping = float(velocity_damping)

        self.arap_constraint = ConstraintAffineARAP(
            body_model.rigidities.numpy(),
            body_model.volumes.numpy(),
            self.body_count,
            self.device,
        )
        matrix_builder = BlockCsrBuilder12(self.body_count)
        self.arap_constraint.append_hessian_structure(matrix_builder)
        self.static_matrix = matrix_builder.finalize(self.device)
        self.arap_constraint.bind_hessian(self.static_matrix)
        self.diagonal_block_indices = wp.array(
            [self.static_matrix.block_index(body, body) for body in range(self.body_count)],
            dtype=int,
            device=self.device,
        )

        self.operator = MixedLinearOperator(
            particle_operator=None,
            affine_matrix=self.static_matrix,
            mixed_dynamic_operator=EmptyMixedDynamicOperator(),
            device=self.device,
        )
        self.linear_solver = MixedPcgSolver(0, self.body_count, self.device)

        self.q = wp.clone(body_model.q)
        self.qd = wp.clone(body_model.qd)
        self.previous_q = wp.empty_like(self.q)
        self.inertia_q = wp.empty_like(self.q)
        self.rhs = wp.empty_like(self.q)
        self.increment = wp.zeros_like(self.q)
        self._empty_particles = wp.empty(0, dtype=wp.vec3, device=self.device)
        self._mixed_rhs = MixedVector3x12(self._empty_particles, self.rhs)
        self._mixed_increment = MixedVector3x12(self._empty_particles, self.increment)
        self.last_linear_iterations = 0

    def step(self, dt: float) -> None:
        """Advance the affine state by one implicit-Euler time step.

        Args:
            dt: Simulation time step [s].
        """
        if not np.isfinite(dt) or dt <= 0.0:
            raise ValueError("dt must be finite and positive")

        wp.launch(
            _initialize_affine_step,
            dim=self.body_count,
            inputs=[self.q, self.qd, self.body_model.gravity, dt],
            outputs=[self.previous_q, self.inertia_q],
            device=self.device,
        )

        inv_dt_squared = 1.0 / (dt * dt)
        for nonlinear_iteration in range(self.nonlinear_iterations):
            self.static_matrix.clear_values()
            wp.launch(
                _initialize_affine_system,
                dim=self.body_count,
                inputs=[
                    self.body_model.mass_matrices,
                    self.diagonal_block_indices,
                    inv_dt_squared,
                    self.inertia_q,
                    self.q,
                ],
                outputs=[self.rhs, self.static_matrix.values],
                device=self.device,
            )
            self.arap_constraint.accumulate_force_and_hessian(
                self.q,
                self.rhs,
                self.static_matrix.values,
            )
            self.static_matrix.update_diagonal()
            self.operator.prepare(None, dt)
            self.last_linear_iterations = self.linear_solver.solve(
                self.operator,
                self._mixed_rhs,
                self._mixed_increment,
                iterations=self.linear_iterations,
                zero_initial_guess=nonlinear_iteration > 0,
            )
            wp.launch(
                _apply_affine_increment,
                dim=self.body_count,
                inputs=[self.increment],
                outputs=[self.q],
                device=self.device,
            )

        wp.launch(
            _finish_affine_step,
            dim=self.body_count,
            inputs=[self.previous_q, self.q, 1.0 / dt, self.velocity_damping],
            outputs=[self.qd],
            device=self.device,
        )

    def update_surface_positions(self, output: wp.array[wp.vec3]) -> None:
        """Write current world-space surface positions.

        Args:
            output: World-space surface positions [m], shape ``(surface_vertex_count, 3)``.
        """
        self.body_model.update_surface_positions(self.q, output)
