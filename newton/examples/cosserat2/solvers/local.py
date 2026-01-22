# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Local iterative constraint solver for Cosserat rods.

This implements the local iterative approach from 02_local_cosserat_rod.py,
where velocity is updated incrementally during constraint solving iterations.
"""

from typing import TYPE_CHECKING

import warp as wp

from newton.examples.cosserat2.kernels import (
    apply_particle_corrections_kernel,
    apply_particle_corrections_with_velocity_kernel,
    apply_quaternion_corrections_kernel,
    solve_bend_twist_constraint_kernel,
    solve_stretch_shear_constraint_kernel,
    zero_quat_kernel,
    zero_vec3_kernel,
)
from newton.examples.cosserat2.solvers.base import (
    ConstraintSolverBase,
    ConstraintSolverType,
    FrictionMethod,
)

if TYPE_CHECKING:
    from newton.examples.cosserat2.cosserat_rod import CosseratRod


class ConstraintSolverLocal(ConstraintSolverBase):
    """Local iterative constraint solver with velocity update.

    This solver implements the approach from 02_local_cosserat_rod.py:
    - Uses parallel kernel launches with atomic accumulation of corrections
    - Updates velocity incrementally during position correction application
    - Does not support friction models (simpler than Jacobi solver)

    The key difference from Jacobi is that velocity is updated as part of
    the position correction (delta/dt), not computed from the final position
    change after all constraint iterations.

    Args:
        rod: The CosseratRod data structure.
        device: Warp device to use.
    """

    def __init__(self, rod: "CosseratRod", device: str = "cuda:0"):
        super().__init__(rod, device)

        # Allocate temporary buffers for velocity tracking during iterations
        self._particle_qd_temp = wp.zeros(rod.num_particles, dtype=wp.vec3, device=device)

        # Store dt for velocity updates (will be set during solve)
        self._dt = 1.0 / 60.0 / 8.0  # Default, will be updated

    @property
    def solver_type(self) -> ConstraintSolverType:
        return ConstraintSolverType.LOCAL

    def set_dt(self, dt: float):
        """Set the time step for velocity updates.

        Args:
            dt: Time step in seconds.
        """
        self._dt = dt

    def solve_stretch_shear(
        self,
        particle_q: wp.array,
        particle_q_out: wp.array,
        stretch_shear_stiffness: wp.vec3,
        particle_qd: wp.array = None,
        particle_qd_out: wp.array = None,
    ):
        """Solve stretch/shear constraints with velocity update.

        1. Zero out correction accumulators
        2. Launch parallel constraint kernel (atomic accumulation)
        3. Apply accumulated corrections with velocity update

        Args:
            particle_q: Current particle positions [num_particles].
            particle_q_out: Output corrected positions [num_particles].
            stretch_shear_stiffness: Stiffness vector (shear_d1, shear_d2, stretch_d3).
            particle_qd: Current particle velocities (optional for local solver).
            particle_qd_out: Output updated velocities (optional for local solver).
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

        # Apply accumulated position corrections with velocity update
        if particle_qd is not None and particle_qd_out is not None:
            wp.launch(
                kernel=apply_particle_corrections_with_velocity_kernel,
                dim=self.num_particles,
                inputs=[
                    particle_q,
                    particle_qd,
                    rod.particle_delta,
                    rod.particle_inv_mass,
                    self._dt,
                ],
                outputs=[particle_q_out, particle_qd_out],
                device=self.device,
            )
        else:
            # Fallback: just update positions (for compatibility with base interface)
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
        """Solve bend/twist constraints without friction.

        The local solver does not support friction models. It uses only
        the basic elastic constraint solving.

        Args:
            bend_twist_stiffness: Stiffness vector (bend_d1, twist, bend_d2).
            friction_method: Ignored (local solver doesn't support friction).
            friction_params: Ignored (local solver doesn't support friction).
            dt: Time step (stored for velocity updates).
        """
        if self.num_bend <= 0:
            return

        # Store dt for velocity updates in solve_stretch_shear
        self._dt = dt

        rod = self.rod

        # Zero out quaternion correction accumulator
        wp.launch(
            kernel=zero_quat_kernel,
            dim=self.num_stretch,
            inputs=[rod.edge_q_delta],
            device=self.device,
        )

        # Solve bend/twist constraints (no friction support in local solver)
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
