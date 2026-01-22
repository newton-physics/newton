# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Main Cosserat XPBD solver with pluggable constraint solving.

Maintains a constant simulation loop with pluggable constraint solvers.
"""

from dataclasses import dataclass, field
from typing import Optional

import warp as wp

from newton.examples.cosserat2.cosserat_rod import CosseratRod
from newton.examples.cosserat2.solvers import (
    ConstraintSolverBase,
    ConstraintSolverType,
    FrictionMethod,
    SOLVER_REGISTRY,
)
from newton.examples.cosserat2.kernels import (
    integrate_particles_kernel,
    update_velocities_kernel,
    apply_velocity_damping_kernel,
    solve_ground_collision_kernel,
    collide_particles_vs_triangles_bvh_kernel,
    compute_current_kappa_kernel,
)


@dataclass
class SolverConfig:
    """Configuration for SolverCosseratXPBD.

    Args:
        constraint_iterations: Number of constraint iterations per substep.
        gravity: Gravity acceleration vector.
        ground_level: Z-coordinate of the ground plane.
        solver_type: Type of constraint solver to use.
        friction_method: Internal friction method.
        stretch_stiffness: Stretch stiffness (0 to 1).
        shear_stiffness: Shear stiffness (0 to 1).
        bend_stiffness: Bend stiffness (0 to 1).
        twist_stiffness: Twist stiffness (0 to 1).
        velocity_damping: Velocity damping coefficient (for VELOCITY_DAMPING).
        strain_rate_damping: Strain-rate damping coefficient (for STRAIN_RATE_DAMPING).
        dahl_eps_max: Maximum persistent strain for Dahl friction.
        dahl_tau: Memory decay length for Dahl friction.
        stretch_compliance: Compliance for direct solvers (stretch).
        bend_compliance: Compliance for direct solvers (bend).
    """

    constraint_iterations: int = 2
    gravity: wp.vec3 = field(default_factory=lambda: wp.vec3(0.0, 0.0, 0.0))
    ground_level: float = 0.0
    solver_type: ConstraintSolverType = ConstraintSolverType.JACOBI

    # Friction settings
    friction_method: FrictionMethod = FrictionMethod.NONE
    velocity_damping: float = 0.99
    strain_rate_damping: float = 0.1
    dahl_eps_max: float = 0.01
    dahl_tau: float = 0.005

    # Stiffness parameters
    stretch_stiffness: float = 1.0
    shear_stiffness: float = 1.0
    bend_stiffness: float = 0.1
    twist_stiffness: float = 0.1

    # Compliance for direct solvers
    stretch_compliance: float = 1.0e-6
    bend_compliance: float = 1.0e-5


class SolverCosseratXPBD:
    """Main Cosserat XPBD solver with pluggable constraint solving.

    The simulation loop is constant across all solver methods:
    1. Integration (semi-implicit Euler)
    2. Constraint solving (pluggable via ConstraintSolverBase)
    3. Collision handling (ground + optional mesh)
    4. Velocity update
    5. Friction post-processing (if VELOCITY_DAMPING)

    Args:
        rod: CosseratRod data structure.
        particle_q: Reference to Newton State particle positions.
        particle_qd: Reference to Newton State particle velocities.
        particle_radius: Reference to Newton Model particle radii.
        config: Solver configuration.
        device: Warp device to use.
    """

    def __init__(
        self,
        rod: CosseratRod,
        particle_q: wp.array,
        particle_qd: wp.array,
        particle_radius: wp.array,
        config: SolverConfig = None,
        device: str = "cuda:0",
    ):
        self.rod = rod
        self.particle_q = particle_q
        self.particle_qd = particle_qd
        self.particle_radius = particle_radius
        self.config = config or SolverConfig()
        self.device = device

        # Temporary buffers
        self.particle_q_predicted = wp.zeros(rod.num_particles, dtype=wp.vec3, device=device)
        self.particle_q_temp = wp.zeros(rod.num_particles, dtype=wp.vec3, device=device)
        self.particle_qd_temp = wp.zeros(rod.num_particles, dtype=wp.vec3, device=device)

        # Initialize constraint solver
        self._constraint_solver = self._create_solver(self.config.solver_type)

        # Optional mesh collision
        self._mesh_collision_enabled = False
        self._mesh_vertices = None
        self._mesh_indices = None
        self._mesh_bvh = None

        # Initialize friction state
        self._initialize_kappa()

    def _create_solver(self, solver_type: ConstraintSolverType) -> ConstraintSolverBase:
        """Create a constraint solver of the specified type."""
        solver_class = SOLVER_REGISTRY[solver_type]

        if solver_type in (ConstraintSolverType.CHOLESKY_SINGLE, ConstraintSolverType.CHOLESKY_MULTI):
            return solver_class(
                self.rod,
                device=self.device,
                stretch_compliance=self.config.stretch_compliance,
                bend_compliance=self.config.bend_compliance,
                multi_tile=(solver_type == ConstraintSolverType.CHOLESKY_MULTI),
            )
        elif solver_type == ConstraintSolverType.THOMAS:
            return solver_class(
                self.rod,
                device=self.device,
                compliance=self.config.stretch_compliance,
            )
        else:
            return solver_class(self.rod, device=self.device)

    def set_constraint_solver(self, solver_type: ConstraintSolverType):
        """Change solver method at runtime (called from UI)."""
        self.config.solver_type = solver_type
        self._constraint_solver = self._create_solver(solver_type)

    @property
    def constraint_solver(self) -> ConstraintSolverBase:
        """Get the current constraint solver."""
        return self._constraint_solver

    def enable_mesh_collision(
        self,
        vertices: wp.array,
        indices: wp.array,
        bvh: wp.Bvh,
    ):
        """Enable collision with a triangle mesh.

        Args:
            vertices: Mesh vertex positions.
            indices: Triangle indices (num_triangles x 3).
            bvh: Pre-built BVH for the mesh.
        """
        self._mesh_collision_enabled = True
        self._mesh_vertices = vertices
        self._mesh_indices = indices
        self._mesh_bvh = bvh

    def disable_mesh_collision(self):
        """Disable mesh collision."""
        self._mesh_collision_enabled = False

    def _initialize_kappa(self):
        """Initialize curvature state arrays for friction models."""
        if self.rod.num_bend > 0:
            wp.launch(
                kernel=compute_current_kappa_kernel,
                dim=self.rod.num_bend,
                inputs=[
                    self.rod.edge_q,
                    self.rod.rest_darboux,
                    self.rod.num_bend,
                ],
                outputs=[self.rod.friction_state.kappa_prev],
                device=self.device,
            )

    def substep(
        self,
        particle_q_in: wp.array,
        particle_qd_in: wp.array,
        particle_q_out: wp.array,
        particle_qd_out: wp.array,
        dt: float,
    ):
        """Perform one substep of the simulation.

        Args:
            particle_q_in: Input particle positions.
            particle_qd_in: Input particle velocities.
            particle_q_out: Output particle positions.
            particle_qd_out: Output particle velocities.
            dt: Time step.
        """
        config = self.config
        rod = self.rod

        # Store old positions for velocity update
        wp.copy(self.particle_q_temp, particle_q_in)

        # Build stiffness vectors from config
        stretch_shear_ks = wp.vec3(
            config.shear_stiffness,
            config.shear_stiffness,
            config.stretch_stiffness,
        )
        bend_twist_ks = wp.vec3(
            config.bend_stiffness,
            config.twist_stiffness,
            config.bend_stiffness,
        )

        # Phase 1: Integration (CONSTANT - same for all methods)
        wp.launch(
            kernel=integrate_particles_kernel,
            dim=rod.num_particles,
            inputs=[
                particle_q_in,
                particle_qd_in,
                rod.particle_inv_mass,
                config.gravity,
                dt,
            ],
            outputs=[self.particle_q_predicted, particle_qd_out],
            device=self.device,
        )

        # Copy predicted to working buffer
        wp.copy(particle_q_out, self.particle_q_predicted)

        # Phase 2: Constraint solving (PLUGGABLE via callback)
        friction_params = {
            "damping_coeff": config.strain_rate_damping,
            "eps_max": config.dahl_eps_max,
            "tau": config.dahl_tau,
        }

        for _ in range(config.constraint_iterations):
            # Solve stretch/shear
            self._constraint_solver.solve_stretch_shear(
                particle_q_out,
                self.particle_q_predicted,
                stretch_shear_ks,
            )
            wp.copy(particle_q_out, self.particle_q_predicted)

            # Solve bend/twist
            self._constraint_solver.solve_bend_twist(
                bend_twist_ks,
                config.friction_method,
                friction_params,
                dt,
            )

        # Phase 3: Collision (CONSTANT - same for all methods)
        # Ground collision
        wp.launch(
            kernel=solve_ground_collision_kernel,
            dim=rod.num_particles,
            inputs=[
                particle_q_out,
                rod.particle_inv_mass,
                self.particle_radius,
                config.ground_level,
            ],
            outputs=[self.particle_q_predicted],
            device=self.device,
        )
        wp.copy(particle_q_out, self.particle_q_predicted)

        # Mesh collision (if enabled)
        if self._mesh_collision_enabled:
            wp.launch(
                kernel=collide_particles_vs_triangles_bvh_kernel,
                dim=rod.num_particles,
                inputs=[
                    particle_q_out,
                    self.particle_radius,
                    rod.particle_inv_mass,
                    self._mesh_vertices,
                    self._mesh_indices,
                    self._mesh_bvh.id,
                ],
                outputs=[self.particle_q_predicted],
                device=self.device,
            )
            wp.copy(particle_q_out, self.particle_q_predicted)

        # Phase 4: Velocity update (CONSTANT)
        wp.launch(
            kernel=update_velocities_kernel,
            dim=rod.num_particles,
            inputs=[
                self.particle_q_temp,
                particle_q_out,
                rod.particle_inv_mass,
                dt,
            ],
            outputs=[particle_qd_out],
            device=self.device,
        )

        # Phase 5: Friction post-processing (CONSTANT)
        if config.friction_method == FrictionMethod.VELOCITY_DAMPING:
            wp.launch(
                kernel=apply_velocity_damping_kernel,
                dim=rod.num_particles,
                inputs=[
                    particle_qd_out,
                    rod.particle_inv_mass,
                    config.velocity_damping,
                ],
                outputs=[self.particle_qd_temp],
                device=self.device,
            )
            wp.copy(particle_qd_out, self.particle_qd_temp)

        # Update friction state for next substep
        if config.friction_method == FrictionMethod.STRAIN_RATE_DAMPING:
            rod.update_friction_state_strain_rate()
        elif config.friction_method == FrictionMethod.DAHL_HYSTERESIS:
            rod.update_friction_state_dahl()
