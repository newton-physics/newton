# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Cholesky decomposition constraint solver for Cosserat rods.

Uses tile-based Cholesky factorization for efficient GPU solving.
"""

from typing import TYPE_CHECKING

import warp as wp

from newton.examples.cosserat2.solvers.base import (
    ConstraintSolverBase,
    ConstraintSolverType,
    FrictionMethod,
)
from newton.examples.cosserat2.kernels import (
    compute_stretch_constraint_data_kernel,
    assemble_stretch_global_system_kernel,
    cholesky_solve_kernel,
    apply_stretch_corrections_kernel,
    compute_bend_constraint_data_kernel,
    assemble_bend_global_system_kernel,
    apply_bend_corrections_kernel,
    TILE,
    BLOCK_DIM,
)

if TYPE_CHECKING:
    from newton.examples.cosserat2.cosserat_rod import CosseratRod


class ConstraintSolverCholesky(ConstraintSolverBase):
    """Cholesky decomposition constraint solver.

    Uses tile-based Cholesky factorization for direct solving.
    Limited to TILE x TILE constraints per system (default 32x32).

    Adapted from 07_global_cosserat_rod_cholesky.py.
    """

    def __init__(
        self,
        rod: "CosseratRod",
        device: str = "cuda:0",
        stretch_compliance: float = 1.0e-6,
        bend_compliance: float = 1.0e-5,
        multi_tile: bool = False,
    ):
        super().__init__(rod, device)
        self.stretch_compliance = stretch_compliance
        self.bend_compliance = bend_compliance
        self.multi_tile = multi_tile

        # Validate tile size
        if self.num_stretch > TILE and not multi_tile:
            raise ValueError(
                f"Single-tile Cholesky requires num_stretch <= {TILE}, "
                f"got {self.num_stretch}. Use multi_tile=True for larger rods."
            )

        # Stretch constraint data
        self.stretch_violation = wp.zeros(self.num_stretch, dtype=float, device=device)
        self.stretch_direction = wp.zeros(self.num_stretch, dtype=wp.vec3, device=device)
        self.stretch_quat_direction = wp.zeros(self.num_stretch, dtype=wp.quat, device=device)

        # Bend constraint data
        self.bend_violation = wp.zeros(max(1, self.num_bend), dtype=float, device=device)
        self.bend_direction = wp.zeros(max(1, self.num_bend), dtype=wp.vec3, device=device)

        # Global system matrices (TILE x TILE for Cholesky)
        self.stretch_matrix = wp.zeros((TILE, TILE), dtype=float, device=device)
        self.stretch_rhs = wp.zeros(TILE, dtype=float, device=device)
        self.stretch_delta_lambda = wp.zeros(TILE, dtype=float, device=device)

        self.bend_matrix = wp.zeros((TILE, TILE), dtype=float, device=device)
        self.bend_rhs = wp.zeros(TILE, dtype=float, device=device)
        self.bend_delta_lambda = wp.zeros(TILE, dtype=float, device=device)

    @property
    def solver_type(self) -> ConstraintSolverType:
        if self.multi_tile:
            return ConstraintSolverType.CHOLESKY_MULTI
        return ConstraintSolverType.CHOLESKY_SINGLE

    def solve_stretch_shear(
        self,
        particle_q: wp.array,
        particle_q_out: wp.array,
        stretch_shear_stiffness: wp.vec3,
    ):
        """Solve stretch/shear constraints using Cholesky decomposition.

        1. Compute constraint data (violations, directions)
        2. Assemble global system matrix
        3. Solve using tile Cholesky
        4. Apply corrections
        """
        rod = self.rod

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

        # Assemble global system
        wp.launch(
            kernel=assemble_stretch_global_system_kernel,
            dim=1,
            inputs=[
                rod.particle_inv_mass,
                rod.edge_inv_mass,
                rod.rest_length,
                self.stretch_direction,
                self.stretch_violation,
                self.stretch_compliance,
                self.num_stretch,
                TILE,
            ],
            outputs=[self.stretch_matrix, self.stretch_rhs],
            device=self.device,
        )

        # Solve via Cholesky
        wp.launch_tiled(
            kernel=cholesky_solve_kernel,
            dim=[1, 1],
            inputs=[self.stretch_matrix, self.stretch_rhs],
            outputs=[self.stretch_delta_lambda],
            block_dim=BLOCK_DIM,
            device=self.device,
        )

        # Apply stretch corrections
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
                self.stretch_delta_lambda,
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
        """Solve bend/twist constraints using Cholesky decomposition.

        Note: Friction models are not directly supported with Cholesky solver.
        For friction, the elastic solve is performed, and friction effects
        are handled separately.
        """
        if self.num_bend <= 0:
            return

        rod = self.rod

        # Compute bend constraint data
        wp.launch(
            kernel=compute_bend_constraint_data_kernel,
            dim=self.num_bend,
            inputs=[
                rod.edge_q,
                rod.rest_darboux,
                self.num_bend,
            ],
            outputs=[self.bend_violation, self.bend_direction],
            device=self.device,
        )

        # Assemble bend system
        wp.launch(
            kernel=assemble_bend_global_system_kernel,
            dim=1,
            inputs=[
                rod.edge_inv_mass,
                self.bend_direction,
                self.bend_violation,
                self.bend_compliance,
                self.num_bend,
                TILE,
            ],
            outputs=[self.bend_matrix, self.bend_rhs],
            device=self.device,
        )

        # Solve via Cholesky
        wp.launch_tiled(
            kernel=cholesky_solve_kernel,
            dim=[1, 1],
            inputs=[self.bend_matrix, self.bend_rhs],
            outputs=[self.bend_delta_lambda],
            block_dim=BLOCK_DIM,
            device=self.device,
        )

        # Apply bend corrections
        wp.launch(
            kernel=apply_bend_corrections_kernel,
            dim=self.num_stretch,
            inputs=[
                rod.edge_q,
                rod.edge_inv_mass,
                self.bend_direction,
                self.bend_delta_lambda,
                self.num_bend,
                self.num_stretch,
            ],
            outputs=[rod.edge_q_new],
            device=self.device,
        )
        rod.swap_edge_q()
