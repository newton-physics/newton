# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""NumPy-based Position Based Dynamics solver for Cosserat rods.

Reference implementation of "Position And Orientation Based Cosserat Rods"
(https://animation.rwth-aachen.de/publication/0550/)

This implementation follows the pbd_rods C++ reference code closely,
providing a non-GPU, sequential solver for validation and testing.
"""

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from newton.examples.cosserat2.reference.cosserat_rod_numpy import CosseratRodNumpy
from newton.examples.cosserat2.reference.quaternion_ops import (
    quat_conjugate,
    quat_e3_bar,
    quat_multiply,
    quat_normalize,
    quat_rotate_e3,
    quat_rotate_vector,
    quat_rotate_vector_inv,
)


@dataclass
class SolverConfig:
    """Configuration for the NumPy Cosserat rod solver.

    Attributes:
        dt: Time step size.
        substeps: Number of substeps per frame.
        constraint_iterations: Number of constraint solver iterations per substep.
        gravity: Gravity acceleration vector [3].
        stretch_stiffness: Stiffness for stretch constraint (0 to 1).
        shear_stiffness: Stiffness for shear constraint (0 to 1).
        bend_stiffness: Stiffness for bend constraint (0 to 1).
        twist_stiffness: Stiffness for twist constraint (0 to 1).
        particle_damping: Particle velocity damping factor (0 to 1, 1 = no damping).
        quaternion_damping: Quaternion angular velocity damping (0 to 1, 1 = no damping).
    """

    dt: float = 1.0 / 60.0
    substeps: int = 4
    constraint_iterations: int = 2
    gravity: NDArray = field(default_factory=lambda: np.array([0.0, 0.0, -9.81]))
    stretch_stiffness: float = 1.0
    shear_stiffness: float = 1.0
    bend_stiffness: float = 0.5
    twist_stiffness: float = 0.5
    particle_damping: float = 0.99
    quaternion_damping: float = 0.99


class PositionBasedCosseratRods:
    """Position Based Dynamics solver for Cosserat rods.

    Implements the constraint solving from the paper:
    "Position And Orientation Based Cosserat Rods"
    (Kugelstadt & Schömer, 2016)

    This is a direct NumPy port of the pbd_rods C++ reference implementation.
    """

    EPS = 1.0e-6

    @staticmethod
    def solve_stretch_shear_constraint(
        p0: NDArray,
        inv_mass_p0: float,
        p1: NDArray,
        inv_mass_p1: float,
        q0: NDArray,
        inv_mass_q0: float,
        stretch_shear_ks: NDArray,
        rest_length: float,
    ) -> tuple[NDArray, NDArray, NDArray]:
        """Solve the stretch and shear constraint (Eq. 37 in paper).

        Constraint: gamma = (p1 - p0) / L - d3(q0) = 0

        The constraint ensures:
        - The edge vector matches the rest length (stretch)
        - The edge vector aligns with the third director d3 (shear)

        Args:
            p0: Position of first particle [3].
            inv_mass_p0: Inverse mass of first particle.
            p1: Position of second particle [3].
            inv_mass_p1: Inverse mass of second particle.
            q0: Edge quaternion [x, y, z, w].
            inv_mass_q0: Inverse mass of the quaternion.
            stretch_shear_ks: Stiffness coefficients [shear_d1, shear_d2, stretch_d3].
            rest_length: Rest length of the edge.

        Returns:
            Tuple of (corr_p0, corr_p1, corr_q0):
            - corr_p0: Position correction for p0 [3].
            - corr_p1: Position correction for p1 [3].
            - corr_q0: Quaternion correction for q0 [4].
        """
        eps = PositionBasedCosseratRods.EPS
        L = rest_length

        # Compute third director: d3 = q0 * e3 * conjugate(q0)
        d3 = quat_rotate_e3(q0)

        # Compute constraint violation: gamma = (p1 - p0) / L - d3
        edge_vec = p1 - p0
        gamma = edge_vec / L - d3

        # Compute effective mass denominator
        # From C++: (invMass1 + invMass0) / restLength + invMassq0 * 4.0 * restLength
        denom = (inv_mass_p0 + inv_mass_p1) / L + inv_mass_q0 * 4.0 * L + eps

        # Scale gamma by inverse denominator
        gamma = gamma / denom

        # Apply anisotropic stiffness in LOCAL frame coordinates
        # Transform gamma to local space: gamma_loc = R^T(q0) * gamma
        gamma_loc = quat_rotate_vector_inv(q0, gamma)

        # Apply stiffness: [shear_d1, shear_d2, stretch_d3]
        gamma_loc = gamma_loc * stretch_shear_ks

        # Transform back to world space: gamma = R(q0) * gamma_loc
        gamma = quat_rotate_vector(q0, gamma_loc)

        # Compute position corrections
        corr_p0 = inv_mass_p0 * gamma
        corr_p1 = -inv_mass_p1 * gamma

        # Compute quaternion correction using q * e3_bar formula
        # From C++: corrq0 = Quaternion(0.0, gamma.x, gamma.y, gamma.z) * q_e_3_bar
        #           corrq0.coeffs *= 2.0 * invMassq0 * restLength
        q_e3_bar = quat_e3_bar(q0)
        gamma_quat = np.array([gamma[0], gamma[1], gamma[2], 0.0])
        corr_q0 = quat_multiply(gamma_quat, q_e3_bar)
        corr_q0 = corr_q0 * (2.0 * inv_mass_q0 * L)

        return corr_p0, corr_p1, corr_q0

    @staticmethod
    def solve_bend_twist_constraint(
        q0: NDArray,
        inv_mass_q0: float,
        q1: NDArray,
        inv_mass_q1: float,
        bend_twist_ks: NDArray,
        rest_darboux: NDArray,
    ) -> tuple[NDArray, NDArray]:
        """Solve the bend and twist constraint (Eq. 40 in paper).

        Constraint: omega = conjugate(q0) * q1 should match rest_darboux

        The Darboux vector omega represents the relative rotation between
        adjacent edge frames, encoding bending and twisting.

        Args:
            q0: First edge quaternion [x, y, z, w].
            inv_mass_q0: Inverse mass of first quaternion.
            q1: Second edge quaternion [x, y, z, w].
            inv_mass_q1: Inverse mass of second quaternion.
            bend_twist_ks: Stiffness coefficients [bend_d1, bend_d2, twist].
            rest_darboux: Rest Darboux vector as quaternion [x, y, z, w].

        Returns:
            Tuple of (corr_q0, corr_q1):
            - corr_q0: Quaternion correction for q0 [4].
            - corr_q1: Quaternion correction for q1 [4].
        """
        eps = PositionBasedCosseratRods.EPS

        # Compute Darboux vector: omega = conjugate(q0) * q1
        q0_conj = quat_conjugate(q0)
        omega = quat_multiply(q0_conj, q1)

        # Handle quaternion double-cover: choose shorter rotation path
        # Compare full quaternion norms |omega + rest|^2 vs |omega - rest|^2
        omega_plus = omega + rest_darboux
        omega_minus = omega - rest_darboux

        # Use full quaternion squared norm (all 4 components)
        if np.dot(omega_minus, omega_minus) > np.dot(omega_plus, omega_plus):
            omega = omega_plus
        else:
            omega = omega_minus

        # Apply bending and twisting stiffness
        denom = inv_mass_q0 + inv_mass_q1 + eps

        # Scale the imaginary components by stiffness
        omega_corrected = np.array(
            [
                omega[0] * bend_twist_ks[0] / denom,
                omega[1] * bend_twist_ks[1] / denom,
                omega[2] * bend_twist_ks[2] / denom,
                0.0,  # Discrete Darboux vector has vanishing scalar part
            ]
        )

        # Compute quaternion corrections
        # From C++: corrq0 = q1 * omega; corrq1 = q0 * omega
        corr_q0 = quat_multiply(q1, omega_corrected)
        corr_q1 = quat_multiply(q0, omega_corrected)

        # Scale by inverse masses
        corr_q0 = corr_q0 * inv_mass_q0
        corr_q1 = corr_q1 * (-inv_mass_q1)

        return corr_q0, corr_q1


class SolverCosseratNumpy:
    """Main solver class for Cosserat rod simulation using NumPy.

    Provides the complete simulation loop including:
    1. External forces (gravity)
    2. Time integration (semi-implicit Euler)
    3. Constraint projection (stretch/shear and bend/twist)
    4. Velocity update
    5. Damping
    """

    def __init__(self, rod: CosseratRodNumpy, config: SolverConfig = None):
        """Initialize the solver.

        Args:
            rod: Cosserat rod data structure.
            config: Solver configuration.
        """
        self.rod = rod
        self.config = config or SolverConfig()

        # Pre-allocated temporary arrays
        self._positions_predicted = np.zeros_like(rod.particle_positions)
        self._positions_old = np.zeros_like(rod.particle_positions)
        self._quaternions_old = np.zeros_like(rod.edge_quaternions)

    def step(self) -> None:
        """Advance the simulation by one frame (dt seconds)."""
        sub_dt = self.config.dt / self.config.substeps

        for _ in range(self.config.substeps):
            self._substep(sub_dt)

    def _substep(self, dt: float) -> None:
        """Perform one substep of the simulation.

        Args:
            dt: Substep time increment.
        """
        rod = self.rod
        config = self.config

        # Store old states for velocity update
        np.copyto(self._positions_old, rod.particle_positions)
        np.copyto(self._quaternions_old, rod.edge_quaternions)

        # Phase 1: Integration (semi-implicit Euler)
        # Apply gravity and predict positions
        for i in range(rod.num_particles):
            if rod.particle_inv_mass[i] > 0:
                # Update velocity with gravity
                rod.particle_velocities[i] += config.gravity * dt

                # Predict position
                self._positions_predicted[i] = rod.particle_positions[i] + rod.particle_velocities[i] * dt
            else:
                # Fixed particle
                self._positions_predicted[i] = rod.particle_positions[i]

        # Copy predicted to current positions for constraint solving
        np.copyto(rod.particle_positions, self._positions_predicted)

        # Phase 2: Constraint projection
        stretch_shear_ks = np.array(
            [
                config.shear_stiffness,
                config.shear_stiffness,
                config.stretch_stiffness,
            ]
        )
        bend_twist_ks = np.array(
            [
                config.bend_stiffness,   # bend_d1
                config.bend_stiffness,   # bend_d2
                config.twist_stiffness,  # twist
            ]
        )

        for _ in range(config.constraint_iterations):
            # Solve stretch/shear constraints (Gauss-Seidel)
            self._solve_stretch_shear_constraints(stretch_shear_ks)

            # Solve bend/twist constraints (Gauss-Seidel)
            self._solve_bend_twist_constraints(bend_twist_ks)

        # Normalize quaternions after constraint solving
        rod.normalize_quaternions()

        # Phase 3: Velocity update for particles
        for i in range(rod.num_particles):
            if rod.particle_inv_mass[i] > 0:
                rod.particle_velocities[i] = (rod.particle_positions[i] - self._positions_old[i]) / dt

        # Phase 4: Angular velocity update for quaternions
        # Compute angular velocity from quaternion change: ω ≈ 2 * Im(q_new * conj(q_old)) / dt
        for i in range(rod.num_edges):
            if rod.edge_inv_mass[i] > 0:
                q_old = self._quaternions_old[i]
                q_new = rod.edge_quaternions[i]
                # q_delta = q_new * conj(q_old) represents the rotation from old to new
                q_old_conj = quat_conjugate(q_old)
                q_delta = quat_multiply(q_new, q_old_conj)
                # Angular velocity is approximately 2 * imaginary part / dt
                rod.edge_angular_velocities[i] = 2.0 * q_delta[:3] / dt

        # Phase 5: Damping
        rod.particle_velocities *= config.particle_damping
        rod.edge_angular_velocities *= config.quaternion_damping

    def _solve_stretch_shear_constraints(self, stretch_shear_ks: NDArray) -> None:
        """Solve all stretch/shear constraints using Gauss-Seidel iteration.

        Args:
            stretch_shear_ks: Stiffness vector [shear_d1, shear_d2, stretch_d3].
        """
        rod = self.rod

        for i in range(rod.num_edges):
            p0 = rod.particle_positions[i]
            p1 = rod.particle_positions[i + 1]
            q0 = rod.edge_quaternions[i]

            inv_mass_p0 = rod.particle_inv_mass[i]
            inv_mass_p1 = rod.particle_inv_mass[i + 1]
            inv_mass_q0 = rod.edge_inv_mass[i]
            rest_length = rod.rest_lengths[i]

            corr_p0, corr_p1, corr_q0 = PositionBasedCosseratRods.solve_stretch_shear_constraint(
                p0,
                inv_mass_p0,
                p1,
                inv_mass_p1,
                q0,
                inv_mass_q0,
                stretch_shear_ks,
                rest_length,
            )

            # Apply corrections immediately (Gauss-Seidel)
            rod.particle_positions[i] += corr_p0
            rod.particle_positions[i + 1] += corr_p1
            rod.edge_quaternions[i] = quat_normalize(rod.edge_quaternions[i] + corr_q0)

    def _solve_bend_twist_constraints(self, bend_twist_ks: NDArray) -> None:
        """Solve all bend/twist constraints using Gauss-Seidel iteration.

        Args:
            bend_twist_ks: Stiffness vector [bend_d1, bend_d2, twist].
        """
        rod = self.rod

        for i in range(rod.num_bend):
            q0 = rod.edge_quaternions[i]
            q1 = rod.edge_quaternions[i + 1]

            inv_mass_q0 = rod.edge_inv_mass[i]
            inv_mass_q1 = rod.edge_inv_mass[i + 1]
            rest_darboux = rod.rest_darboux[i]

            corr_q0, corr_q1 = PositionBasedCosseratRods.solve_bend_twist_constraint(
                q0,
                inv_mass_q0,
                q1,
                inv_mass_q1,
                bend_twist_ks,
                rest_darboux,
            )

            # Apply corrections immediately (Gauss-Seidel)
            rod.edge_quaternions[i] = quat_normalize(rod.edge_quaternions[i] + corr_q0)
            rod.edge_quaternions[i + 1] = quat_normalize(rod.edge_quaternions[i + 1] + corr_q1)

    def get_particle_positions(self) -> NDArray:
        """Get current particle positions.

        Returns:
            Particle positions [num_particles, 3].
        """
        return self.rod.particle_positions.copy()

    def get_edge_quaternions(self) -> NDArray:
        """Get current edge quaternions.

        Returns:
            Edge quaternions [num_edges, 4].
        """
        return self.rod.edge_quaternions.copy()
