# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""SPH (Smoothed Particle Hydrodynamics) solver for Newton."""

import numpy as np
import warp as wp

import newton

from ...core.types import override
from ..solver import SolverBase
from .kernels_sph import compute_density, compute_sph_forces

__all__ = ["SolverSPH"]


class SolverSPH(SolverBase):
    """Smoothed Particle Hydrodynamics solver using the Müller et al. 2003 formulation.

    This solver computes pressure and viscous forces between particles using SPH
    kernels (Poly6 for density, Spiky for pressure, viscosity Laplacian). Gravity
    is applied through Newton's standard integration pipeline.

    Call :meth:`register_custom_attributes` on your :class:`~newton.ModelBuilder`
    before calling :meth:`~newton.ModelBuilder.finalize` to register the
    ``sph:density`` attribute used by this solver.

    Example::

        builder = newton.ModelBuilder()
        SolverSPH.register_custom_attributes(builder)
        builder.add_particle_grid(...)
        model = builder.finalize()
        solver = SolverSPH(model)
    """

    def __init__(
        self,
        model: newton.Model,
        smoothing_length: float | None = None,
        pressure_stiffness: float = 20.0,
        rest_density: float = 1.0,
        dynamic_viscosity: float = 0.025,
    ):
        """Initialize the SPH solver.

        Args:
            model: The Newton model to simulate.
            smoothing_length: SPH smoothing length [m]. If ``None``, defaults to
                ``2.0 * model.particle_max_radius``.
            pressure_stiffness: Stiffness constant for the Tait equation of state [Pa].
            rest_density: Rest density of the fluid [kg/m³].
            dynamic_viscosity: Dynamic viscosity coefficient [Pa·s].
        """
        super().__init__(model=model)

        default_h = 2.0 * model.particle_max_radius if model.particle_max_radius > 0.0 else 1.0
        self.smoothing_length = (
            smoothing_length if smoothing_length is not None else default_h
        )
        self.pressure_stiffness = pressure_stiffness
        self.rest_density = rest_density
        self.dynamic_viscosity = dynamic_viscosity

        h = self.smoothing_length

        # Precompute kernel normalization constants (Müller 2003)
        self.density_normalization = 315.0 / (64.0 * np.pi * h**9)
        self.pressure_normalization = -45.0 / (np.pi * h**6)
        self.viscous_normalization = 45.0 * dynamic_viscosity / (np.pi * h**6)

    @override
    def step(self, state_in, state_out, control, contacts, dt):
        """Advance the SPH simulation by one time step.

        The solver computes SPH density and forces, delegates boundary handling
        to Newton's collision pipeline via the ``contacts`` argument, then
        integrates particles forward using semi-implicit Euler.

        Args:
            state_in: Input particle state.
            state_out: Output particle state (written to).
            control: Control input (unused by SPH).
            contacts: Contact data from ``model.collide()``.
            dt: Time step [s].
        """
        model = self.model
        if model.particle_count == 0:
            return

        # Get density array from state custom attributes
        particle_rho = state_in.sph.density

        # Clear forces
        state_in.particle_f.zero_()

        # Build spatial hash grid with SPH smoothing length
        if model.particle_grid is None:
            model.particle_grid = wp.HashGrid(
                128, 128, 128, device=model.device
            )
        model.particle_grid.build(state_in.particle_q, self.smoothing_length)

        # Compute SPH density
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

        # Compute SPH pressure and viscosity forces
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

        # Integrate particles (handles gravity and velocity clamping)
        self.integrate_particles(model, state_in, state_out, dt)

    @override
    def update_contacts(self, contacts, state=None):
        """SPH does not generate solver-internal contacts."""

    @classmethod
    def register_custom_attributes(cls, builder: newton.ModelBuilder) -> None:
        """Register SPH-specific custom attributes.

        Registers the ``sph:density`` per-particle attribute on State, which
        stores the computed SPH density for each particle.

        Args:
            builder: The model builder to register attributes on.
        """
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
