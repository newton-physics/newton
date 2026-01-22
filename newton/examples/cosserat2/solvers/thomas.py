# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Thomas algorithm (TDMA) constraint solver for Cosserat rods.

Uses O(n) tridiagonal solver for stretch constraints, Jacobi for bend.
"""

from typing import TYPE_CHECKING

import warp as wp

from newton.examples.cosserat2.solvers.base import (
    ConstraintSolverBase,
    ConstraintSolverType,
    FrictionMethod,
)
from newton.examples.cosserat2.kernels import (
    zero_quat_kernel,
    compute_stretch_constraint_data_kernel,
    assemble_stretch_tridiagonal_system_kernel,
    thomas_solve_kernel,
    apply_stretch_corrections_kernel,
    solve_bend_twist_constraint_kernel,
    solve_bend_twist_with_strain_rate_damping_kernel,
    solve_bend_twist_with_dahl_friction_kernel,
    apply_quaternion_corrections_kernel,
)

if TYPE_CHECKING:
    from newton.examples.cosserat2.cosserat_rod import CosseratRod


class ConstraintSolverThomas(ConstraintSolverBase):
    """Thomas algorithm (TDMA) constraint solver.

    Uses O(n) tridiagonal solver for stretch constraints.
    Falls back to Jacobi for bend/twist constraints.

    Adapted from 01_global_pbd_chain_thomas.py, extended for Cosserat rods.
    """

    def __init__(self, rod: "CosseratRod", device: str = "cuda:0", compliance: float = 1.0e-6):
        super().__init__(rod, device)
        self.compliance = compliance

        # Constraint data buffers
        self.stretch_violation = wp.zeros(self.num_stretch, dtype=float, device=device)
        self.stretch_direction = wp.zeros(self.num_stretch, dtype=wp.vec3, device=device)
        self.stretch_quat_direction = wp.zeros(self.num_stretch, dtype=wp.quat, device=device)

        # Tridiagonal system storage
        self.diag = wp.zeros(self.num_stretch, dtype=float, device=device)
        self.off_diag = wp.zeros(max(1, self.num_stretch - 1), dtype=float, device=device)
        self.rhs = wp.zeros(self.num_stretch, dtype=float, device=device)
        self.delta_lambda = wp.zeros(self.num_stretch, dtype=float, device=device)

        # Thomas algorithm workspace
        self.c_prime = wp.zeros(self.num_stretch, dtype=float, device=device)
        self.d_prime = wp.zeros(self.num_stretch, dtype=float, device=device)

    @property
    def solver_type(self) -> ConstraintSolverType:
        return ConstraintSolverType.THOMAS

    def solve_stretch_shear(
        self,
        particle_q: wp.array,
        particle_q_out: wp.array,
        stretch_shear_stiffness: wp.vec3,
    ):
        """Solve stretch/shear constraints using Thomas algorithm.

        1. Compute constraint data (violations, directions)
        2. Assemble tridiagonal system
        3. Solve using Thomas algorithm O(n)
        4. Apply corrections
        """
        rod = self.rod

        # Compute compliance factor (using stretch component of stiffness)
        # Note: For Thomas, we use a simpler compliance model
        compliance_factor = self.compliance

        # Compute constraint data
        wp.launch(
            kernel=compute_stretch_constraint_data_kernel,
            dim=self.num_stretch,
            inputs=[
                particle_q,
                rod.edge_q,
                rod.rest_length,
                self.num_stretch,
            ],
            outputs=[
                self.stretch_violation,
                self.stretch_direction,
                self.stretch_quat_direction,
            ],
            device=self.device,
        )

        # Assemble tridiagonal system
        wp.launch(
            kernel=assemble_stretch_tridiagonal_system_kernel,
            dim=1,
            inputs=[
                rod.particle_inv_mass,
                rod.edge_inv_mass,
                rod.rest_length,
                self.stretch_direction,
                self.stretch_violation,
                compliance_factor,
                self.num_stretch,
            ],
            outputs=[self.diag, self.off_diag, self.rhs],
            device=self.device,
        )

        # Solve using Thomas algorithm
        wp.launch(
            kernel=thomas_solve_kernel,
            dim=1,
            inputs=[
                self.diag,
                self.off_diag,
                self.rhs,
                self.num_stretch,
                self.c_prime,
                self.d_prime,
            ],
            outputs=[self.delta_lambda],
            device=self.device,
        )

        # Apply corrections
        wp.launch(
            kernel=apply_stretch_corrections_kernel,
            dim=max(self.num_particles, self.num_stretch),
            inputs=[
                particle_q,
                rod.edge_q,
                rod.particle_inv_mass,
                rod.edge_inv_mass,
                rod.rest_length,
                self.stretch_direction,
                self.stretch_quat_direction,
                self.delta_lambda,
                self.num_stretch,
                self.num_particles,
            ],
            outputs=[particle_q_out, rod.edge_q_new],
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
        """Solve bend/twist constraints using Jacobi (same as ConstraintSolverJacobi).

        Thomas algorithm is primarily for stretch constraints; bend/twist
        uses the same Jacobi approach.
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
