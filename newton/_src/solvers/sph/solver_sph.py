# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""SPH (Smoothed Particle Hydrodynamics) solver for Newton.

Implements two solvers:

- **SolverSPH** -- classical SPH using the Müller et al. 2003 formulation with
  explicit pressure and viscosity forces.
- **SolverPBF** -- Position-Based Fluids (Macklin & Müller 2013) that enforces
  incompressibility through iterative position corrections on predicted
  particle positions.
"""

import numpy as np
import warp as wp

import newton

from ...core.types import override
from ...geometry import ParticleFlags
from ..solver import SolverBase
from .kernels_sph import (
    compute_density,
    compute_sph_forces,
    pbf_compute_lambda,
    pbf_compute_position_correction,
    pbf_apply_position_correction,
    pbf_update_velocity,
)

__all__ = ["SolverSPH", "SolverPBF"]


# ---------------------------------------------------------------------------
# PBF device kernels
# ---------------------------------------------------------------------------


@wp.kernel
def predict_positions(
    particle_q: wp.array[wp.vec3],
    particle_qd: wp.array[wp.vec3],
    particle_flags: wp.array[wp.int32],
    particle_world: wp.array[wp.int32],
    gravity: wp.array[wp.vec3],
    dt: float,
    # outputs
    p_pred: wp.array[wp.vec3],
):
    """Predict particle positions under gravity (semi-implicit Euler step)."""
    tid = wp.tid()
    if (particle_flags[tid] & ParticleFlags.ACTIVE) == 0:
        p_pred[tid] = particle_q[tid]
        return

    world_idx = particle_world[tid]
    g = gravity[wp.max(world_idx, 0)]
    v = particle_qd[tid]
    p_pred[tid] = particle_q[tid] + v * dt + 0.5 * g * dt * dt


@wp.kernel
def copy_corrected_positions(
    particle_flags: wp.array[wp.int32],
    p_pred: wp.array[wp.vec3],
    # outputs
    particle_q: wp.array[wp.vec3],
):
    """Copy corrected predicted positions into output state positions."""
    tid = wp.tid()
    if (particle_flags[tid] & ParticleFlags.ACTIVE) == 0:
        return
    particle_q[tid] = p_pred[tid]


@wp.kernel
def update_velocity_pbf(
    particle_q: wp.array[wp.vec3],
    particle_flags: wp.array[wp.int32],
    particle_world: wp.array[wp.int32],
    p_pred: wp.array[wp.vec3],
    v_max: float,
    dt: float,
    viscosity: float,
    # outputs
    particle_qd: wp.array[wp.vec3],
):
    """Update velocities from corrected positions and apply viscous damping."""
    tid = wp.tid()
    if (particle_flags[tid] & ParticleFlags.ACTIVE) == 0:
        return

    v = (p_pred[tid] - particle_q[tid]) / dt
    v = v * (1.0 - viscosity * dt)

    v_mag = wp.length(v)
    if v_mag > v_max:
        v = v * (v_max / v_mag)

    particle_qd[tid] = v


@wp.kernel
def scale_and_apply_correction(
    particle_flags: wp.array[wp.int32],
    delta_p: wp.array[wp.vec3],
    omega: float,
    # inputs/outputs
    p_pred: wp.array[wp.vec3],
):
    """Apply relaxed position correction: p_pred += omega * delta_p."""
    tid = wp.tid()
    if (particle_flags[tid] & ParticleFlags.ACTIVE) == 0:
        return
    p_pred[tid] = p_pred[tid] + omega * delta_p[tid]


# ---------------------------------------------------------------------------
# SolverSPH -- classical SPH (Müller 2003)
# ---------------------------------------------------------------------------


class SolverSPH(SolverBase):
    """Smoothed Particle Hydrodynamics solver using the Müller et al. 2003 formulation."""

    def __init__(
        self,
        model: newton.Model,
        smoothing_length: float | None = None,
        pressure_stiffness: float = 20.0,
        rest_density: float = 1.0,
        dynamic_viscosity: float = 0.025,
    ):
        super().__init__(model=model)

        default_h = 2.0 * model.particle_max_radius if model.particle_max_radius > 0.0 else 1.0
        self.smoothing_length = (
            smoothing_length if smoothing_length is not None else default_h
        )
        self.pressure_stiffness = pressure_stiffness
        self.rest_density = rest_density
        self.dynamic_viscosity = dynamic_viscosity

        h = self.smoothing_length
        self.density_normalization = 315.0 / (64.0 * np.pi * h**9)
        self.pressure_normalization = -45.0 / (np.pi * h**6)
        self.viscous_normalization = 45.0 * dynamic_viscosity / (np.pi * h**6)

    @override
    def step(self, state_in, state_out, control, contacts, dt):
        model = self.model
        if model.particle_count == 0:
            return

        particle_rho = state_in.sph.density
        state_in.particle_f.zero_()

        if model.particle_grid is None:
            model.particle_grid = wp.HashGrid(128, 128, 128, device=model.device)
        model.particle_grid.build(state_in.particle_q, self.smoothing_length)

        wp.launch(
            kernel=compute_density,
            dim=model.particle_count,
            inputs=[
                model.particle_grid.id,
                state_in.particle_q,
                model.particle_mass,
                model.particle_flags,
                model.particle_world,
                self.density_normalization,
                self.smoothing_length,
            ],
            outputs=[particle_rho],
            device=model.device,
        )

        wp.launch(
            kernel=compute_sph_forces,
            dim=model.particle_count,
            inputs=[
                model.particle_grid.id,
                state_in.particle_q,
                state_in.particle_qd,
                model.particle_mass,
                model.particle_flags,
                model.particle_world,
                particle_rho,
                self.pressure_stiffness,
                self.rest_density,
                self.pressure_normalization,
                self.viscous_normalization,
                self.smoothing_length,
            ],
            outputs=[state_in.particle_f],
            device=model.device,
        )

        self.integrate_particles(model, state_in, state_out, dt)

    @override
    def update_contacts(self, contacts, state=None):
        pass

    @classmethod
    def register_custom_attributes(cls, builder: newton.ModelBuilder) -> None:
        builder.add_custom_attribute(
            newton.ModelBuilder.CustomAttribute(
                name="density",
                frequency=newton.Model.AttributeFrequency.PARTICLE,
                assignment=newton.Model.AttributeAssignment.STATE,
                dtype=wp.float32,
                default=0.0,
                namespace="sph",
            )
        )


# ---------------------------------------------------------------------------
# SolverPBF -- Position-Based Fluids (Macklin & Müller 2013)
# ---------------------------------------------------------------------------


class SolverPBF(SolverBase):
    """Position-Based Fluids solver enforcing incompressibility iteratively."""

    def __init__(
        self,
        model: newton.Model,
        smoothing_length: float | None = None,
        iterations: int = 4,
        rest_density: float = 1000.0,
        epsilon: float = 1e-6,
        surface_tension: float = 0.5,
        dynamic_viscosity: float = 0.02,
        omega: float = 0.03,
    ):
        super().__init__(model=model)

        default_h = 2.0 * model.particle_max_radius if model.particle_max_radius > 0.0 else 1.0
        self.smoothing_length = (
            smoothing_length if smoothing_length is not None else default_h
        )
        self.iterations = iterations
        self.rest_density = rest_density
        self.epsilon = epsilon
        self.surface_tension = surface_tension
        self.dynamic_viscosity = dynamic_viscosity
        self.omega = omega

        h = self.smoothing_length

        # Kernel normalization constants (Müller 2003)
        self.density_normalization = 315.0 / (64.0 * np.pi * h**9)
        self.spiky_normalization = 45.0 / (np.pi * h**6)

        n = model.particle_count
        self._p_pred = wp.zeros(n, dtype=wp.vec3, device=model.device)
        self._delta_p = wp.zeros(n, dtype=wp.vec3, device=model.device)

        if n > 1 and model.particle_grid is not None:
            with wp.ScopedDevice(model.device):
                model.particle_grid.reserve(n)

    @override
    def step(self, state_in, state_out, control, contacts, dt):
        model = self.model
        if model.particle_count == 0:
            return

        particle_rho = state_in.sph.density
        particle_lambda = state_in.sph.lambda_attr

        # --- Step 1: Predict positions under gravity ---
        wp.launch(
            kernel=predict_positions,
            dim=model.particle_count,
            inputs=[
                state_in.particle_q,
                state_in.particle_qd,
                model.particle_flags,
                model.particle_world,
                model.gravity,
                dt,
            ],
            outputs=[self._p_pred],
            device=model.device,
        )

        # Build spatial hash grid on predicted positions
        if model.particle_grid is None:
            model.particle_grid = wp.HashGrid(128, 128, 128, device=model.device)
        model.particle_grid.build(self._p_pred, self.smoothing_length)

        # --- Step 2: Compute density on predicted positions ---
        wp.launch(
            kernel=compute_density,
            dim=model.particle_count,
            inputs=[
                model.particle_grid.id,
                self._p_pred,
                model.particle_mass,
                model.particle_flags,
                model.particle_world,
                self.density_normalization,
                self.smoothing_length,
            ],
            outputs=[particle_rho],
            device=model.device,
        )

        # --- Step 3: Iterative constraint solving ---
        for iteration in range(self.iterations):
            # 3a. Compute Lagrange multipliers
            wp.launch(
                kernel=pbf_compute_lambda,
                dim=model.particle_count,
                inputs=[
                    model.particle_grid.id,
                    self._p_pred,
                    model.particle_mass,
                    model.particle_flags,
                    model.particle_world,
                    particle_rho,
                    self.rest_density,
                    self.smoothing_length,
                    self.spiky_normalization,
                    self.epsilon,
                ],
                outputs=[particle_lambda],
                device=model.device,
            )

            # 3b. Compute position corrections
            self._delta_p.zero_()
            wp.launch(
                kernel=pbf_compute_position_correction,
                dim=model.particle_count,
                inputs=[
                    model.particle_grid.id,
                    self._p_pred,
                    state_in.particle_qd,
                    model.particle_mass,
                    model.particle_flags,
                    model.particle_world,
                    particle_rho,
                    particle_lambda,
                    self.rest_density,
                    self.smoothing_length,
                    self.spiky_normalization,
                    self.surface_tension,
                    self.density_normalization,
                    dt,
                ],
                outputs=[self._delta_p],
                device=model.device,
            )

            # 3c. Apply position corrections with relaxation
            wp.launch(
                kernel=scale_and_apply_correction,
                dim=model.particle_count,
                inputs=[
                    model.particle_flags,
                    self._delta_p,
                    self.omega,
                ],
                outputs=[self._p_pred],
                device=model.device,
            )

            # 3d. Recompute density on corrected positions (skip on last iter)
            if iteration < self.iterations - 1:
                model.particle_grid.build(self._p_pred, self.smoothing_length)
                wp.launch(
                    kernel=compute_density,
                    dim=model.particle_count,
                    inputs=[
                        model.particle_grid.id,
                        self._p_pred,
                        model.particle_mass,
                        model.particle_flags,
                        model.particle_world,
                        self.density_normalization,
                        self.smoothing_length,
                    ],
                    outputs=[particle_rho],
                    device=model.device,
                )

        # --- Step 4: Derive velocity from position delta ---
        wp.launch(
            kernel=update_velocity_pbf,
            dim=model.particle_count,
            inputs=[
                state_in.particle_q,
                model.particle_flags,
                model.particle_world,
                self._p_pred,
                model.particle_max_velocity,
                dt,
                self.dynamic_viscosity,
            ],
            outputs=[state_out.particle_qd],
            device=model.device,
        )

        # --- Step 5: Copy corrected positions and density to output state ---
        state_out.particle_q.assign(state_in.particle_q)
        wp.launch(
            kernel=copy_corrected_positions,
            dim=model.particle_count,
            inputs=[
                model.particle_flags,
                self._p_pred,
            ],
            outputs=[state_out.particle_q],
            device=model.device,
        )

        # Copy density and lambda to output state so they survive the state swap
        state_out.sph.density.assign(state_in.sph.density)
        state_out.sph.lambda_attr.assign(state_in.sph.lambda_attr)

    @override
    def update_contacts(self, contacts, state=None):
        pass

    @classmethod
    def register_custom_attributes(cls, builder: newton.ModelBuilder) -> None:
        builder.add_custom_attribute(
            newton.ModelBuilder.CustomAttribute(
                name="density",
                frequency=newton.Model.AttributeFrequency.PARTICLE,
                assignment=newton.Model.AttributeAssignment.STATE,
                dtype=wp.float32,
                default=0.0,
                namespace="sph",
            )
        )
        builder.add_custom_attribute(
            newton.ModelBuilder.CustomAttribute(
                name="lambda_attr",
                frequency=newton.Model.AttributeFrequency.PARTICLE,
                assignment=newton.Model.AttributeAssignment.STATE,
                dtype=wp.float32,
                default=0.0,
                namespace="sph",
            )
        )
