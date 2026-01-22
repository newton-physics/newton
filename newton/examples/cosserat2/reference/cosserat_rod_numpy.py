# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""NumPy-based Cosserat rod data structure.

Reference implementation matching the Warp-based CosseratRod class
for validation and testing purposes.
"""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from newton.examples.cosserat2.reference.quaternion_ops import (
    quat_conjugate,
    quat_multiply,
    quat_normalize,
)


@dataclass
class CosseratRodNumpy:
    """NumPy-based Cosserat rod data structure.

    Attributes:
        num_particles: Number of particles in the rod.
        particle_positions: Particle positions [num_particles, 3].
        particle_velocities: Particle velocities [num_particles, 3].
        particle_inv_mass: Inverse mass per particle [num_particles].
        edge_quaternions: Edge quaternions [num_edges, 4] as [x, y, z, w].
        edge_angular_velocities: Edge angular velocities [num_edges, 3].
        edge_inv_mass: Inverse mass per edge [num_edges].
        rest_lengths: Rest length per edge [num_edges].
        rest_darboux: Rest Darboux vectors [num_bend, 4] as quaternions.
    """

    num_particles: int
    particle_positions: NDArray
    particle_velocities: NDArray
    particle_inv_mass: NDArray
    edge_quaternions: NDArray
    edge_angular_velocities: NDArray
    edge_inv_mass: NDArray
    rest_lengths: NDArray
    rest_darboux: NDArray

    @property
    def num_edges(self) -> int:
        """Number of edges (stretch constraints)."""
        return self.num_particles - 1

    @property
    def num_bend(self) -> int:
        """Number of bend/twist constraints."""
        return self.num_particles - 2

    @classmethod
    def create_straight_rod(
        cls,
        num_particles: int,
        start_pos: NDArray,
        direction: NDArray,
        segment_length: float,
        particle_mass: float = 1.0,
        edge_mass: float = 1.0,
        fixed_particles: list[int] | None = None,
        fixed_edges: list[int] | None = None,
    ) -> "CosseratRodNumpy":
        """Create a straight rod along a given direction.

        Args:
            num_particles: Number of particles.
            start_pos: Starting position [3].
            direction: Direction vector (will be normalized) [3].
            segment_length: Length of each segment.
            particle_mass: Mass of each particle.
            edge_mass: Mass of each edge (for quaternion inertia).
            fixed_particles: List of particle indices to fix (inv_mass = 0).
            fixed_edges: List of edge indices to fix (inv_mass = 0). If None and
                fixed_particles contains index 0, edge 0 will be automatically fixed
                to match C++ pbd_rods behavior.

        Returns:
            Initialized CosseratRodNumpy instance.
        """
        # Normalize direction
        direction = np.array(direction, dtype=np.float64)
        direction = direction / np.linalg.norm(direction)

        # Create particle positions along the direction
        positions = np.zeros((num_particles, 3), dtype=np.float64)
        for i in range(num_particles):
            positions[i] = start_pos + i * segment_length * direction

        # Initialize velocities to zero
        velocities = np.zeros((num_particles, 3), dtype=np.float64)

        # Set inverse masses (0 for fixed particles)
        particle_inv_mass = np.ones(num_particles, dtype=np.float64) / particle_mass
        if fixed_particles:
            for idx in fixed_particles:
                particle_inv_mass[idx] = 0.0

        # Number of edges
        num_edges = num_particles - 1

        # Edge inverse masses
        edge_inv_mass = np.ones(num_edges, dtype=np.float64) / edge_mass

        # Fix edges (set inverse mass to 0)
        # If fixed_edges is not provided but particle 0 is fixed, auto-fix edge 0
        # to match C++ pbd_rods behavior
        if fixed_edges is not None:
            for idx in fixed_edges:
                edge_inv_mass[idx] = 0.0
        elif fixed_particles and 0 in fixed_particles:
            edge_inv_mass[0] = 0.0

        # Rest lengths
        rest_lengths = np.full(num_edges, segment_length, dtype=np.float64)

        # Compute initial edge quaternions
        # The quaternion should rotate e3 = [0, 0, 1] to the edge direction
        edge_quaternions = np.zeros((num_edges, 4), dtype=np.float64)
        for i in range(num_edges):
            edge_quaternions[i] = _compute_quaternion_from_direction(direction)

        # Initialize angular velocities to zero
        edge_angular_velocities = np.zeros((num_edges, 3), dtype=np.float64)

        # Compute rest Darboux vectors (for a straight rod, these should be zero)
        num_bend = num_particles - 2
        rest_darboux = np.zeros((max(1, num_bend), 4), dtype=np.float64)

        if num_bend > 0:
            # Identity quaternion in [x, y, z, w] format
            identity_quat = np.array([0.0, 0.0, 0.0, 1.0])

            for i in range(num_bend):
                q0 = edge_quaternions[i]
                q1 = edge_quaternions[i + 1]
                # Rest Darboux = conjugate(q0) * q1
                omega = quat_multiply(quat_conjugate(q0), q1)

                # Ensure rest Darboux is in hemisphere closer to identity quaternion
                # This matches the C++ behavior which checks:
                # if |omega - identity|^2 > |omega + identity|^2 then flip omega
                omega_plus = omega + identity_quat
                omega_minus = omega - identity_quat
                if np.dot(omega_minus, omega_minus) > np.dot(omega_plus, omega_plus):
                    omega = -omega

                rest_darboux[i] = omega

        return cls(
            num_particles=num_particles,
            particle_positions=positions,
            particle_velocities=velocities,
            particle_inv_mass=particle_inv_mass,
            edge_quaternions=edge_quaternions,
            edge_angular_velocities=edge_angular_velocities,
            edge_inv_mass=edge_inv_mass,
            rest_lengths=rest_lengths,
            rest_darboux=rest_darboux,
        )

    def normalize_quaternions(self) -> None:
        """Normalize all edge quaternions to unit length."""
        for i in range(self.num_edges):
            self.edge_quaternions[i] = quat_normalize(self.edge_quaternions[i])

    def get_particle_position(self, idx: int) -> NDArray:
        """Get position of particle at index."""
        return self.particle_positions[idx].copy()

    def set_particle_position(self, idx: int, pos: NDArray) -> None:
        """Set position of particle at index."""
        self.particle_positions[idx] = pos

    def get_edge_quaternion(self, idx: int) -> NDArray:
        """Get quaternion of edge at index."""
        return self.edge_quaternions[idx].copy()

    def set_edge_quaternion(self, idx: int, q: NDArray) -> None:
        """Set quaternion of edge at index."""
        self.edge_quaternions[idx] = quat_normalize(q)


def _compute_quaternion_from_direction(direction: NDArray) -> NDArray:
    """Compute a quaternion that rotates e3 = [0, 0, 1] to the given direction.

    Args:
        direction: Target direction (should be normalized).

    Returns:
        Quaternion [x, y, z, w].
    """
    e3 = np.array([0.0, 0.0, 1.0])
    direction = direction / np.linalg.norm(direction)

    # Handle parallel/anti-parallel cases
    dot = np.dot(e3, direction)

    if dot > 0.9999:
        # Already aligned with e3
        return np.array([0.0, 0.0, 0.0, 1.0])
    elif dot < -0.9999:
        # Anti-parallel: rotate 180 degrees around any perpendicular axis
        # Use x-axis if e3 is [0, 0, 1]
        return np.array([1.0, 0.0, 0.0, 0.0])
    else:
        # General case: rotation axis = e3 x direction
        axis = np.cross(e3, direction)
        axis = axis / np.linalg.norm(axis)

        # Rotation angle
        angle = np.arccos(np.clip(dot, -1.0, 1.0))

        # Quaternion from axis-angle
        half_angle = angle / 2.0
        sin_half = np.sin(half_angle)
        cos_half = np.cos(half_angle)

        return np.array(
            [
                axis[0] * sin_half,
                axis[1] * sin_half,
                axis[2] * sin_half,
                cos_half,
            ]
        )
