# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Jacobi-style iterative constraint solver for Cosserat rods."""

from typing import TYPE_CHECKING

import warp as wp

from newton.examples.cosserat2.solvers.base import (
    ConstraintSolverBase,
    ConstraintSolverType,
    FrictionMethod,
)
from newton.examples.cosserat2.kernels import (
    zero_vec3_kernel,
    zero_quat_kernel,
    solve_stretch_shear_constraint_kernel,
    solve_bend_twist_constraint_kernel,
    solve_bend_twist_with_strain_rate_damping_kernel,
    solve_bend_twist_with_dahl_friction_kernel,
    apply_particle_corrections_kernel,
    apply_quaternion_corrections_kernel,
)

if TYPE_CHECKING:
    from newton.examples.cosserat2.cosserat_rod import CosseratRod


class ConstraintSolverJacobi(ConstraintSolverBase):
    """Jacobi-style iterative constraint solver.

    Uses parallel kernel launches with atomic accumulation of corrections.
    Supports all three friction models for bend/twist constraints.

    This is the default solver, ported from 10_sim_aorta.py.
    """

    def __init__(self, rod: "CosseratRod", device: str = "cuda:0"):
        super().__init__(rod, device)

    @property
    def solver_type(self) -> ConstraintSolverType:
        return ConstraintSolverType.JACOBI

    def solve_stretch_shear(
        self,
        particle_q: wp.array,
        particle_q_out: wp.array,
        stretch_shear_stiffness: wp.vec3,
    ):
        """Solve stretch/shear constraints using Jacobi iteration.

        1. Zero out correction accumulators
        2. Launch parallel constraint kernel (atomic accumulation)
        3. Apply accumulated corrections
        """
        rod = self.rod

        # Zero out correction accumulators
        wp.launch(
            kernel=zero_vec3_kernel,
            dim=self.num_particles,
            inputs=[rod.particle_delta],
            device=self.device,
        )
        wp.launch(
            kernel=zero_quat_kernel,
            dim=self.num_stretch,
            inputs=[rod.edge_q_delta],
            device=self.device,
        )

        # Solve stretch/shear constraints (accumulates corrections)
        wp.launch(
            kernel=solve_stretch_shear_constraint_kernel,
            dim=self.num_stretch,
            inputs=[
                particle_q,
                rod.particle_inv_mass,
                rod.edge_q,
                rod.edge_inv_mass,
                rod.rest_length,
                stretch_shear_stiffness,
                self.num_stretch,
            ],
            outputs=[rod.particle_delta, rod.edge_q_delta],
            device=self.device,
        )

        # Apply accumulated position corrections
        wp.launch(
            kernel=apply_particle_corrections_kernel,
            dim=self.num_particles,
            inputs=[
                particle_q,
                rod.particle_delta,
                rod.particle_inv_mass,
            ],
            outputs=[particle_q_out],
            device=self.device,
        )

        # Apply accumulated quaternion corrections
        wp.launch(
            kernel=apply_quaternion_corrections_kernel,
            dim=self.num_stretch,
            inputs=[
                rod.edge_q,
                rod.edge_q_delta,
                rod.edge_inv_mass,
            ],
            outputs=[rod.edge_q_new],
            device=self.device,
        )
        rod.swap_edge_q()

    def solve_bend_twist(
        self,
        bend_twist_stiffness: wp.vec3,
        friction_method: FrictionMethod,
        friction_params: dict,
        dt: float,
    ):
        """Solve bend/twist constraints with optional friction.

        Supports three friction methods:
        - NONE: Standard elastic constraint
        - STRAIN_RATE_DAMPING: Rayleigh-style damping
        - DAHL_HYSTERESIS: Path-dependent friction
        """
        if self.num_bend <= 0:
            return

        rod = self.rod

        # Zero out quaternion correction accumulator
        wp.launch(
            kernel=zero_quat_kernel,
            dim=self.num_stretch,
            inputs=[rod.edge_q_delta],
            device=self.device,
        )

        if friction_method == FrictionMethod.STRAIN_RATE_DAMPING:
            damping_coeff = friction_params.get("damping_coeff", 0.1)
            wp.launch(
                kernel=solve_bend_twist_with_strain_rate_damping_kernel,
                dim=self.num_bend,
                inputs=[
                    rod.edge_q,
                    rod.edge_inv_mass,
                    rod.rest_darboux,
                    bend_twist_stiffness,
                    rod.friction_state.kappa_prev,
                    damping_coeff,
                    dt,
                    self.num_bend,
                ],
                outputs=[rod.edge_q_delta, rod.friction_state.kappa_current],
                device=self.device,
            )
        elif friction_method == FrictionMethod.DAHL_HYSTERESIS:
            eps_max = friction_params.get("eps_max", 0.01)
            tau = friction_params.get("tau", 0.005)
            wp.launch(
                kernel=solve_bend_twist_with_dahl_friction_kernel,
                dim=self.num_bend,
                inputs=[
                    rod.edge_q,
                    rod.edge_inv_mass,
                    rod.rest_darboux,
                    bend_twist_stiffness,
                    rod.friction_state.kappa_prev,
                    rod.friction_state.sigma_prev,
                    rod.friction_state.dkappa_prev,
                    eps_max,
                    tau,
                    self.num_bend,
                ],
                outputs=[
                    rod.edge_q_delta,
                    rod.friction_state.kappa_current,
                    rod.friction_state.sigma_current,
                    rod.friction_state.dkappa_current,
                ],
                device=self.device,
            )
        else:
            # No friction or velocity damping (applied separately)
            wp.launch(
                kernel=solve_bend_twist_constraint_kernel,
                dim=self.num_bend,
                inputs=[
                    rod.edge_q,
                    rod.edge_inv_mass,
                    rod.rest_darboux,
                    bend_twist_stiffness,
                    self.num_bend,
                ],
                outputs=[rod.edge_q_delta],
                device=self.device,
            )

        # Apply accumulated quaternion corrections
        wp.launch(
            kernel=apply_quaternion_corrections_kernel,
            dim=self.num_stretch,
            inputs=[
                rod.edge_q,
                rod.edge_q_delta,
                rod.edge_inv_mass,
            ],
            outputs=[rod.edge_q_new],
            device=self.device,
        )
        rod.swap_edge_q()
