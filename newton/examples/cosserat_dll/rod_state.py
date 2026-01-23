# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""State management for Cosserat rod simulation using DefKit DLL."""

import numpy as np


class RodState:
    """Manages all state arrays for a Cosserat rod.

    This class holds all the arrays needed for the Position and Orientation
    Based Cosserat Rods simulation, with proper memory layout for the DLL.

    Note: btVector3 and btQuaternion both use 4 floats for SIMD alignment.
    Quaternions use (x, y, z, w) order (Bullet convention).
    """

    def __init__(self, n_particles: int):
        """Initialize rod state arrays.

        Args:
            n_particles: Number of particles in the rod.
        """
        self.n_particles = n_particles
        self.n_edges = n_particles - 1

        # Position state (btVector3 = 4 floats for SIMD alignment)
        self.positions = np.zeros((n_particles, 4), dtype=np.float32)
        self.predicted_positions = np.zeros((n_particles, 4), dtype=np.float32)
        self.velocities = np.zeros((n_particles, 4), dtype=np.float32)
        self.forces = np.zeros((n_particles, 4), dtype=np.float32)
        self.inv_masses = np.ones(n_particles, dtype=np.float32)

        # Orientation state (btQuaternion = 4 floats: x, y, z, w)
        # Initialize to identity quaternion (0, 0, 0, 1)
        self.orientations = np.zeros((n_particles, 4), dtype=np.float32)
        self.orientations[:, 3] = 1.0  # w = 1 for identity
        self.predicted_orientations = np.zeros((n_particles, 4), dtype=np.float32)
        self.predicted_orientations[:, 3] = 1.0
        self.prev_orientations = np.zeros((n_particles, 4), dtype=np.float32)
        self.prev_orientations[:, 3] = 1.0
        self.angular_velocities = np.zeros((n_particles, 4), dtype=np.float32)
        self.torques = np.zeros((n_particles, 4), dtype=np.float32)
        self.quat_inv_masses = np.ones(n_particles, dtype=np.float32)

        # Rod constraint properties (per edge)
        self.rest_lengths = np.zeros(self.n_edges, dtype=np.float32)
        # Rest Darboux vector as quaternion (identity = straight rod)
        self.rest_darboux = np.zeros((self.n_edges, 4), dtype=np.float32)
        self.rest_darboux[:, 3] = 1.0  # Identity quaternion
        # Bending/twisting stiffness coefficients (btVector3 = 4 floats)
        self.bend_twist_ks = np.ones((self.n_edges, 4), dtype=np.float32)

        # Global stiffness parameters
        self.stretch_ks = 1.0
        self.bend_twist_ks_mult = 1.0

    def set_position(self, index: int, x: float, y: float, z: float):
        """Set position of a particle.

        Args:
            index: Particle index.
            x, y, z: Position coordinates.
        """
        self.positions[index, 0] = x
        self.positions[index, 1] = y
        self.positions[index, 2] = z
        self.positions[index, 3] = 0.0
        self.predicted_positions[index] = self.positions[index].copy()

    def set_orientation(self, index: int, qx: float, qy: float, qz: float, qw: float):
        """Set orientation of a particle as quaternion.

        Args:
            index: Particle index.
            qx, qy, qz, qw: Quaternion components (Bullet order: x, y, z, w).
        """
        self.orientations[index] = [qx, qy, qz, qw]
        self.predicted_orientations[index] = self.orientations[index].copy()
        self.prev_orientations[index] = self.orientations[index].copy()

    def fix_particle(self, index: int):
        """Fix a particle (make it static).

        Args:
            index: Particle index to fix.
        """
        self.inv_masses[index] = 0.0
        self.quat_inv_masses[index] = 0.0

    def get_positions_3d(self) -> np.ndarray:
        """Get positions as (n, 3) array for visualization.

        Returns:
            Positions with shape (n_particles, 3).
        """
        return self.positions[:, :3]

    def copy_positions_to_predicted(self):
        """Copy current positions to predicted positions."""
        np.copyto(self.predicted_positions, self.positions)

    def copy_orientations_to_predicted(self):
        """Copy current orientations to predicted orientations."""
        np.copyto(self.predicted_orientations, self.orientations)

    def clear_forces(self):
        """Clear all forces and torques."""
        self.forces.fill(0)
        self.torques.fill(0)


def create_straight_rod(
    n_particles: int,
    start_pos: tuple[float, float, float],
    direction: tuple[float, float, float],
    segment_length: float,
    fix_first: bool = True,
) -> RodState:
    """Create a straight rod state.

    Args:
        n_particles: Number of particles.
        start_pos: Starting position (x, y, z).
        direction: Unit direction vector (x, y, z).
        segment_length: Length of each segment.
        fix_first: Whether to fix the first particle.

    Returns:
        Initialized RodState.
    """
    state = RodState(n_particles)

    # Normalize direction
    dir_arr = np.array(direction, dtype=np.float32)
    dir_arr = dir_arr / np.linalg.norm(dir_arr)

    # Set positions along the line
    for i in range(n_particles):
        pos = np.array(start_pos) + i * segment_length * dir_arr
        state.set_position(i, pos[0], pos[1], pos[2])

    # Set rest lengths
    state.rest_lengths[:] = segment_length

    # Compute orientation quaternion that aligns Z axis with rod direction
    # Default material frame has d3 (third director) along the edge
    # We need quaternion q such that q * (0,0,1) * q^-1 = direction
    z_axis = np.array([0, 0, 1], dtype=np.float32)
    quat = _quaternion_from_two_vectors(z_axis, dir_arr)

    for i in range(n_particles):
        state.set_orientation(i, quat[0], quat[1], quat[2], quat[3])

    # Fix first particle if requested
    if fix_first:
        state.fix_particle(0)

    return state


def _quaternion_from_two_vectors(v1: np.ndarray, v2: np.ndarray) -> np.ndarray:
    """Compute quaternion that rotates v1 to v2.

    Args:
        v1: Source vector (normalized).
        v2: Target vector (normalized).

    Returns:
        Quaternion as (x, y, z, w) array.
    """
    # Handle parallel/anti-parallel cases
    dot = np.dot(v1, v2)

    if dot > 0.9999:
        # Vectors are parallel, return identity
        return np.array([0, 0, 0, 1], dtype=np.float32)

    if dot < -0.9999:
        # Vectors are anti-parallel, rotate 180 degrees around any perpendicular axis
        perp = np.cross(v1, np.array([1, 0, 0]))
        if np.linalg.norm(perp) < 0.001:
            perp = np.cross(v1, np.array([0, 1, 0]))
        perp = perp / np.linalg.norm(perp)
        return np.array([perp[0], perp[1], perp[2], 0], dtype=np.float32)

    # General case: axis = v1 x v2, angle from dot product
    axis = np.cross(v1, v2)
    s = np.sqrt((1 + dot) * 2)
    inv_s = 1.0 / s

    quat = np.array([axis[0] * inv_s, axis[1] * inv_s, axis[2] * inv_s, s * 0.5], dtype=np.float32)

    # Normalize
    quat = quat / np.linalg.norm(quat)
    return quat
