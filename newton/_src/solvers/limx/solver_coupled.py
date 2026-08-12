# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Coupled projected-Newton stepping for LIMX particles and affine bodies."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import warp as wp

from ...core.types import override
from ...geometry import ParticleFlags
from ...sim import Contacts, Control, Model, State
from ..solver import SolverBase
from .affine_body import AffineBodyModel
from .affine_types import mat1212, vec12
from .block_csr import BlockCsrBuilder
from .block_csr_12 import BlockCsrBuilder12
from .constraints.affine_arap import ConstraintAffineARAP
from .mixed_linear_solver import MixedPcgSolver
from .mixed_operator import MixedLinearOperator, MixedVector3x12
from .operator import CompositeLinearOperator, EmptyDynamicConstraintOperator
from .solver_affine import (
    _apply_affine_increment,
    _finish_affine_step,
    _initialize_affine_step,
    _initialize_affine_system,
)
from .solver_newton import _apply_increment, _finish_step, _initialize_rhs, _initialize_step


class _EmptyCoupledDynamicOperator:
    def __init__(self, particle_count: int, body_count: int, device: Any):
        self.particle_count = particle_count
        self.body_count = body_count
        self.device = wp.get_device(device)

    def begin_step(
        self,
        particle_q: wp.array[wp.vec3],
        particle_qd: wp.array[wp.vec3],
        affine_q: wp.array[vec12],
        affine_qd: wp.array[vec12],
        dt: float,
    ) -> None:
        pass

    def prepare(self, particle_q: wp.array[wp.vec3], affine_q: wp.array[vec12]) -> None:
        pass

    def accumulate_force(
        self,
        particle_q: wp.array[wp.vec3],
        affine_q: wp.array[vec12],
        particle_output: wp.array[wp.vec3],
        affine_output: wp.array[vec12],
    ) -> None:
        pass

    def multiply(
        self,
        particle_input: wp.array[wp.vec3],
        affine_input: wp.array[vec12],
        particle_output: wp.array[wp.vec3],
        affine_output: wp.array[vec12],
    ) -> None:
        pass

    def accumulate_diagonal(
        self,
        particle_diagonal: wp.array[wp.mat33],
        affine_diagonal: wp.array[mat1212],
    ) -> None:
        pass


class SolverLIMXCoupled(SolverBase):
    """Advance one particle model and one affine-body model in a shared Newton solve."""

    def __init__(
        self,
        model: Model,
        constraints: Sequence[Any],
        body_model: AffineBodyModel,
        nonlinear_iterations: int = 4,
        linear_iterations: int = 32,
        velocity_damping: float = 1.0,
        dynamic_operator: Any | None = None,
    ):
        """Create a coupled LIMX particle-affine solver.

        Args:
            model: Particle-only model containing active, positive-mass particles.
            constraints: Static particle constraint batches.
            body_model: Affine body mass, gravity, surface, and ARAP data.
            nonlinear_iterations: Newton position iterations per step.
            linear_iterations: Fixed mixed-PCG iterations per Newton iteration.
            velocity_damping: Per-step velocity multiplier for both domains.
            dynamic_operator: Optional matrix-free particle-affine operator.
        """
        super().__init__(model)
        if not isinstance(body_model, AffineBodyModel):
            raise TypeError("body_model must be an AffineBodyModel")
        if model.body_count > 0:
            raise ValueError("SolverLIMXCoupled accepts particles through model and affine bodies through body_model")
        if model.particle_count <= 0 or model.particle_mass is None:
            raise ValueError("SolverLIMXCoupled requires at least one particle")
        masses = model.particle_mass.numpy()
        if not np.isfinite(masses).all() or np.any(masses <= 0.0):
            raise ValueError("SolverLIMXCoupled requires finite positive particle masses")
        flags = model.particle_flags.numpy()
        if np.any((flags & ParticleFlags.ACTIVE) == 0):
            raise ValueError("SolverLIMXCoupled requires active particles; use ConstraintAnchor to fix positions")
        if body_model.device != self.device:
            raise ValueError("Particle and affine models must use the same device")
        if nonlinear_iterations <= 0:
            raise ValueError("nonlinear_iterations must be positive")
        if linear_iterations <= 0:
            raise ValueError("linear_iterations must be positive")
        if not np.isfinite(velocity_damping) or velocity_damping < 0.0 or velocity_damping > 1.0:
            raise ValueError("velocity_damping must be finite and between zero and one")

        self.constraints = tuple(constraints)
        self.body_model = body_model
        self.body_count = body_model.body_count
        self.nonlinear_iterations = nonlinear_iterations
        self.linear_iterations = linear_iterations
        self.velocity_damping = float(velocity_damping)
        if dynamic_operator is None:
            self.dynamic_operator = _EmptyCoupledDynamicOperator(
                model.particle_count,
                self.body_count,
                self.device,
            )
        else:
            if getattr(dynamic_operator, "particle_count", None) != model.particle_count:
                raise ValueError("Dynamic operator and solver must have matching particle counts")
            if getattr(dynamic_operator, "body_count", None) != self.body_count:
                raise ValueError("Dynamic operator and solver must have matching body counts")
            operator_device = getattr(dynamic_operator, "device", None)
            if operator_device is None or wp.get_device(operator_device) != self.device:
                raise ValueError("Dynamic operator and solver must use the same device")
            self.dynamic_operator = dynamic_operator

        particle_matrix_builder = BlockCsrBuilder(model.particle_count)
        for constraint in self.constraints:
            if getattr(constraint, "particle_count", None) != model.particle_count:
                raise ValueError("Every particle constraint must match the model particle count")
            if getattr(constraint, "device", None) != self.device:
                raise ValueError("Every particle constraint must use the model device")
            constraint.append_hessian_structure(particle_matrix_builder)
        self.particle_static_matrix = particle_matrix_builder.finalize(self.device)
        for constraint in self.constraints:
            constraint.bind_hessian(self.particle_static_matrix)

        self.particle_operator = CompositeLinearOperator(
            masses=model.particle_mass,
            static_matrix=self.particle_static_matrix,
            dynamic_operator=EmptyDynamicConstraintOperator(),
            device=self.device,
        )

        self.arap_constraint = ConstraintAffineARAP(
            body_model.rigidities.numpy(),
            body_model.volumes.numpy(),
            self.body_count,
            self.device,
        )
        affine_matrix_builder = BlockCsrBuilder12(self.body_count)
        self.arap_constraint.append_hessian_structure(affine_matrix_builder)
        self.affine_static_matrix = affine_matrix_builder.finalize(self.device)
        self.arap_constraint.bind_hessian(self.affine_static_matrix)
        self.affine_diagonal_block_indices = wp.array(
            [self.affine_static_matrix.block_index(body, body) for body in range(self.body_count)],
            dtype=int,
            device=self.device,
        )

        self.operator = MixedLinearOperator(
            particle_operator=self.particle_operator,
            affine_matrix=self.affine_static_matrix,
            mixed_dynamic_operator=self.dynamic_operator,
            device=self.device,
        )
        self.linear_solver = MixedPcgSolver(model.particle_count, self.body_count, self.device)

        self.previous_positions = wp.empty(model.particle_count, dtype=wp.vec3, device=self.device)
        self.inertia_positions = wp.empty_like(self.previous_positions)
        self.iterate_positions = wp.empty_like(self.previous_positions)
        self.particle_rhs = wp.empty_like(self.previous_positions)
        self.particle_increment = wp.zeros_like(self.previous_positions)

        self.q = wp.clone(body_model.q)
        self.qd = wp.clone(body_model.qd)
        self.previous_q = wp.empty_like(self.q)
        self.inertia_q = wp.empty_like(self.q)
        self.affine_rhs = wp.empty_like(self.q)
        self.affine_increment = wp.zeros_like(self.q)
        self._mixed_rhs = MixedVector3x12(self.particle_rhs, self.affine_rhs)
        self._mixed_increment = MixedVector3x12(self.particle_increment, self.affine_increment)
        self.last_linear_iterations = 0

    @override
    def step(
        self,
        state_in: State,
        state_out: State,
        control: Control | None,
        contacts: Contacts | None,
        dt: float,
    ) -> None:
        """Advance particles and affine bodies by one implicit-Euler step.

        Args:
            state_in: Input particle state, which remains unchanged.
            state_out: State receiving updated particle positions and velocities.
            control: Unused control input.
            contacts: Unused Newton contact data.
            dt: Simulation time step [s].
        """
        if not np.isfinite(dt) or dt <= 0.0:
            raise ValueError("dt must be finite and positive")

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
        wp.launch(
            _initialize_affine_step,
            dim=self.body_count,
            inputs=[self.q, self.qd, self.body_model.gravity, dt],
            outputs=[self.previous_q, self.inertia_q],
            device=self.device,
        )
        self.dynamic_operator.begin_step(
            state_in.particle_q,
            state_in.particle_qd,
            self.q,
            self.qd,
            dt,
        )

        inv_dt_squared = 1.0 / (dt * dt)
        for nonlinear_iteration in range(self.nonlinear_iterations):
            self.dynamic_operator.prepare(self.iterate_positions, self.q)

            self.particle_static_matrix.clear_values()
            wp.launch(
                _initialize_rhs,
                dim=model.particle_count,
                inputs=[
                    model.particle_mass,
                    inv_dt_squared,
                    self.inertia_positions,
                    self.iterate_positions,
                ],
                outputs=[self.particle_rhs],
                device=self.device,
            )
            for constraint in self.constraints:
                constraint.accumulate_force_and_hessian(
                    self.iterate_positions,
                    self.particle_rhs,
                    self.particle_static_matrix.values,
                )
            self.particle_static_matrix.update_diagonal()

            self.affine_static_matrix.clear_values()
            wp.launch(
                _initialize_affine_system,
                dim=self.body_count,
                inputs=[
                    self.body_model.mass_matrices,
                    self.affine_diagonal_block_indices,
                    inv_dt_squared,
                    self.inertia_q,
                    self.q,
                ],
                outputs=[self.affine_rhs, self.affine_static_matrix.values],
                device=self.device,
            )
            self.arap_constraint.accumulate_force_and_hessian(
                self.q,
                self.affine_rhs,
                self.affine_static_matrix.values,
            )
            self.dynamic_operator.accumulate_force(
                self.iterate_positions,
                self.q,
                self.particle_rhs,
                self.affine_rhs,
            )
            self.affine_static_matrix.update_diagonal()

            self.operator.prepare(self.iterate_positions, dt)
            self.last_linear_iterations = self.linear_solver.solve(
                self.operator,
                self._mixed_rhs,
                self._mixed_increment,
                iterations=self.linear_iterations,
                zero_initial_guess=nonlinear_iteration > 0,
            )
            wp.launch(
                _apply_increment,
                dim=model.particle_count,
                inputs=[self.particle_increment],
                outputs=[self.iterate_positions],
                device=self.device,
            )
            wp.launch(
                _apply_affine_increment,
                dim=self.body_count,
                inputs=[self.affine_increment],
                outputs=[self.q],
                device=self.device,
            )

        wp.launch(
            _finish_step,
            dim=model.particle_count,
            inputs=[self.previous_positions, self.iterate_positions, 1.0 / dt, self.velocity_damping],
            outputs=[state_out.particle_q, state_out.particle_qd],
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
        """Write current affine-body surface positions to ``output`` [m]."""
        self.body_model.update_surface_positions(self.q, output)
