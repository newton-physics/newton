# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Cosserat rod model - unified data structure for rod simulation state.

This module defines the CosseratRodModel class that holds all arrays needed for
Cosserat rod simulation, independent of the solver backend.
"""

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class MaterialProperties:
    """Material properties for the rod."""

    young_modulus: float = 1.0e6  # Pa - affects bending stiffness
    torsion_modulus: float = 1.0e6  # Pa - affects twist stiffness
    radius: float = 0.01  # m - cross-section radius

    @property
    def cross_section_area(self) -> float:
        """Cross-sectional area (pi * r^2)."""
        return np.pi * self.radius * self.radius

    @property
    def second_moment_area(self) -> float:
        """Second moment of area for bending (pi * r^4 / 4)."""
        return np.pi * self.radius**4 / 4.0

    @property
    def polar_moment(self) -> float:
        """Polar moment of inertia for torsion (pi * r^4 / 2)."""
        return np.pi * self.radius**4 / 2.0


@dataclass
class SimulationConfig:
    """Simulation configuration parameters."""

    position_damping: float = 0.001  # Velocity damping (0-1)
    rotation_damping: float = 0.001  # Angular velocity damping (0-1)
    gravity: np.ndarray = field(
        default_factory=lambda: np.array([0.0, 0.0, -9.81], dtype=np.float32)
    )

    def __post_init__(self):
        self.gravity = np.asarray(self.gravity, dtype=np.float32)


class CosseratRodModel:
    """Unified data structure for Cosserat rod simulation state.

    This class holds all arrays needed for Position and Orientation Based
    Cosserat Rods simulation, with proper memory layout for interoperability
    with various backends (DLL, NumPy, Warp).

    Memory Layout:
        - Positions/velocities/forces: (n_particles, 4) float32
          The 4th component is padding for SIMD alignment.
        - Orientations: (n_particles, 4) float32 as quaternions (x, y, z, w)
          Using Bullet physics convention for DLL compatibility.
        - Per-edge properties: (n_edges,) or (n_edges, 3/4) arrays

    Attributes:
        n_particles: Number of particles (nodes) in the rod.
        n_edges: Number of edges (segments) = n_particles - 1.
        material: Material properties (Young's modulus, torsion modulus, radius).
        config: Simulation configuration (damping, gravity).

    State Arrays:
        positions: Current particle positions (n, 4).
        predicted_positions: Predicted positions after integration (n, 4).
        velocities: Particle velocities (n, 4).
        forces: External forces on particles (n, 4).
        inv_masses: Inverse particle masses (n,).
        orientations: Current quaternion orientations (n, 4).
        predicted_orientations: Predicted orientations (n, 4).
        prev_orientations: Previous orientations for velocity calc (n, 4).
        angular_velocities: Angular velocities (n, 4).
        torques: External torques (n, 4).
        quat_inv_masses: Inverse rotational masses (n,).

    Rod Properties:
        rest_lengths: Rest length per edge (n_edges,).
        rest_darboux: Rest Darboux vector per edge (n_edges, 3).
        bend_stiffness: Bending/twist stiffness coefficients (n_edges, 3).
    """

    def __init__(
        self,
        n_particles: int,
        material: Optional[MaterialProperties] = None,
        config: Optional[SimulationConfig] = None,
    ):
        """Initialize rod model with empty state arrays.

        Args:
            n_particles: Number of particles in the rod.
            material: Material properties (uses defaults if None).
            config: Simulation configuration (uses defaults if None).
        """
        if n_particles < 2:
            raise ValueError("Rod must have at least 2 particles")

        self.n_particles = n_particles
        self.n_edges = n_particles - 1
        self.material = material or MaterialProperties()
        self.config = config or SimulationConfig()

        # Position state (4 floats for SIMD alignment)
        self.positions = np.zeros((n_particles, 4), dtype=np.float32)
        self.predicted_positions = np.zeros((n_particles, 4), dtype=np.float32)
        self.velocities = np.zeros((n_particles, 4), dtype=np.float32)
        self.forces = np.zeros((n_particles, 4), dtype=np.float32)
        self.inv_masses = np.ones(n_particles, dtype=np.float32)

        # Orientation state (quaternion: x, y, z, w)
        self.orientations = np.zeros((n_particles, 4), dtype=np.float32)
        self.orientations[:, 3] = 1.0  # Identity quaternion
        self.predicted_orientations = np.zeros((n_particles, 4), dtype=np.float32)
        self.predicted_orientations[:, 3] = 1.0
        self.prev_orientations = np.zeros((n_particles, 4), dtype=np.float32)
        self.prev_orientations[:, 3] = 1.0
        self.angular_velocities = np.zeros((n_particles, 4), dtype=np.float32)
        self.torques = np.zeros((n_particles, 4), dtype=np.float32)
        self.quat_inv_masses = np.ones(n_particles, dtype=np.float32)

        # Rod constraint properties (per edge)
        self.rest_lengths = np.zeros(self.n_edges, dtype=np.float32)
        # Rest Darboux vector (kappa1, kappa2, tau) - NOT quaternion
        self.rest_darboux = np.zeros((self.n_edges, 3), dtype=np.float32)
        # Bending/twist stiffness coefficients (k_bend1, k_bend2, k_twist)
        self.bend_stiffness = np.ones((self.n_edges, 3), dtype=np.float32)

    # =========================================================================
    # Position/orientation setters
    # =========================================================================

    def set_position(self, index: int, x: float, y: float, z: float):
        """Set position of a particle.

        Args:
            index: Particle index.
            x, y, z: Position coordinates.
        """
        self.positions[index, :3] = [x, y, z]
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
        """Fix a particle in place (zero inverse mass).

        Args:
            index: Particle index to fix.
        """
        self.inv_masses[index] = 0.0
        self.quat_inv_masses[index] = 0.0

    def unfix_particle(self, index: int, mass: float = 1.0):
        """Unfix a particle (restore inverse mass).

        Args:
            index: Particle index to unfix.
            mass: Particle mass (default 1.0).
        """
        self.inv_masses[index] = 1.0 / mass
        self.quat_inv_masses[index] = 1.0 / mass

    # =========================================================================
    # State management
    # =========================================================================

    def copy_positions_to_predicted(self):
        """Copy current positions to predicted positions."""
        np.copyto(self.predicted_positions, self.positions)

    def copy_orientations_to_predicted(self):
        """Copy current orientations to predicted orientations."""
        np.copyto(self.predicted_orientations, self.orientations)

    def clear_forces(self):
        """Clear all external forces and torques."""
        self.forces.fill(0.0)
        self.torques.fill(0.0)

    def reset_to_initial(self):
        """Reset velocities and forces (keep positions/orientations)."""
        self.velocities.fill(0.0)
        self.angular_velocities.fill(0.0)
        self.clear_forces()

    # =========================================================================
    # Convenience accessors
    # =========================================================================

    def get_positions_3d(self) -> np.ndarray:
        """Get positions as (n, 3) array for visualization.

        Returns:
            Positions with shape (n_particles, 3).
        """
        return self.positions[:, :3].copy()

    def get_tip_position(self) -> np.ndarray:
        """Get position of the last (tip) particle.

        Returns:
            Tip position as (3,) array.
        """
        return self.positions[-1, :3].copy()

    def get_root_position(self) -> np.ndarray:
        """Get position of the first (root) particle.

        Returns:
            Root position as (3,) array.
        """
        return self.positions[0, :3].copy()

    # =========================================================================
    # Rest shape configuration
    # =========================================================================

    def set_rest_curvature(self, kappa1: float, kappa2: float, tau: float):
        """Set uniform rest curvature for the entire rod.

        Args:
            kappa1: First bending curvature component.
            kappa2: Second bending curvature component.
            tau: Twist (torsion).
        """
        self.rest_darboux[:, 0] = kappa1
        self.rest_darboux[:, 1] = kappa2
        self.rest_darboux[:, 2] = tau

    def set_bend_stiffness_uniform(self, k1: float, k2: float, k_tau: float):
        """Set uniform bending/twist stiffness for the entire rod.

        Args:
            k1: First bending stiffness coefficient.
            k2: Second bending stiffness coefficient.
            k_tau: Twist stiffness coefficient.
        """
        self.bend_stiffness[:, 0] = k1
        self.bend_stiffness[:, 1] = k2
        self.bend_stiffness[:, 2] = k_tau

    # =========================================================================
    # Serialization / cloning
    # =========================================================================

    def clone(self) -> "CosseratRodModel":
        """Create a deep copy of the model.

        Returns:
            New CosseratRodModel with copied state.
        """
        model = CosseratRodModel(
            n_particles=self.n_particles,
            material=MaterialProperties(
                young_modulus=self.material.young_modulus,
                torsion_modulus=self.material.torsion_modulus,
                radius=self.material.radius,
            ),
            config=SimulationConfig(
                position_damping=self.config.position_damping,
                rotation_damping=self.config.rotation_damping,
                gravity=self.config.gravity.copy(),
            ),
        )

        # Copy all state arrays
        np.copyto(model.positions, self.positions)
        np.copyto(model.predicted_positions, self.predicted_positions)
        np.copyto(model.velocities, self.velocities)
        np.copyto(model.forces, self.forces)
        np.copyto(model.inv_masses, self.inv_masses)
        np.copyto(model.orientations, self.orientations)
        np.copyto(model.predicted_orientations, self.predicted_orientations)
        np.copyto(model.prev_orientations, self.prev_orientations)
        np.copyto(model.angular_velocities, self.angular_velocities)
        np.copyto(model.torques, self.torques)
        np.copyto(model.quat_inv_masses, self.quat_inv_masses)
        np.copyto(model.rest_lengths, self.rest_lengths)
        np.copyto(model.rest_darboux, self.rest_darboux)
        np.copyto(model.bend_stiffness, self.bend_stiffness)

        return model


# =============================================================================
# Factory functions
# =============================================================================


def _quaternion_from_two_vectors(v1: np.ndarray, v2: np.ndarray) -> np.ndarray:
    """Compute quaternion that rotates v1 to v2.

    Args:
        v1: Source vector (normalized).
        v2: Target vector (normalized).

    Returns:
        Quaternion as (x, y, z, w) array.
    """
    dot = np.dot(v1, v2)

    if dot > 0.9999:
        # Vectors are parallel
        return np.array([0, 0, 0, 1], dtype=np.float32)

    if dot < -0.9999:
        # Vectors are anti-parallel
        perp = np.cross(v1, np.array([1, 0, 0]))
        if np.linalg.norm(perp) < 0.001:
            perp = np.cross(v1, np.array([0, 1, 0]))
        perp = perp / np.linalg.norm(perp)
        return np.array([perp[0], perp[1], perp[2], 0], dtype=np.float32)

    # General case
    axis = np.cross(v1, v2)
    s = np.sqrt((1 + dot) * 2)
    inv_s = 1.0 / s

    quat = np.array(
        [axis[0] * inv_s, axis[1] * inv_s, axis[2] * inv_s, s * 0.5], dtype=np.float32
    )
    return quat / np.linalg.norm(quat)


def create_straight_rod(
    n_particles: int,
    start_pos: tuple[float, float, float],
    direction: tuple[float, float, float],
    segment_length: float,
    fix_first: bool = True,
    material: Optional[MaterialProperties] = None,
    config: Optional[SimulationConfig] = None,
) -> CosseratRodModel:
    """Create a straight rod model.

    Args:
        n_particles: Number of particles.
        start_pos: Starting position (x, y, z).
        direction: Unit direction vector (x, y, z).
        segment_length: Length of each segment.
        fix_first: Whether to fix the first particle.
        material: Material properties (uses defaults if None).
        config: Simulation configuration (uses defaults if None).

    Returns:
        Initialized CosseratRodModel.
    """
    model = CosseratRodModel(
        n_particles=n_particles,
        material=material,
        config=config,
    )

    # Normalize direction
    dir_arr = np.array(direction, dtype=np.float32)
    dir_arr = dir_arr / np.linalg.norm(dir_arr)

    # Set positions along the line
    for i in range(n_particles):
        pos = np.array(start_pos) + i * segment_length * dir_arr
        model.set_position(i, pos[0], pos[1], pos[2])

    # Set rest lengths
    model.rest_lengths[:] = segment_length

    # Compute orientation quaternion that aligns Z axis with rod direction
    z_axis = np.array([0, 0, 1], dtype=np.float32)
    quat = _quaternion_from_two_vectors(z_axis, dir_arr)

    for i in range(n_particles):
        model.set_orientation(i, quat[0], quat[1], quat[2], quat[3])

    # Fix first particle if requested
    if fix_first:
        model.fix_particle(0)

    return model


def create_curved_rod(
    n_particles: int,
    start_pos: tuple[float, float, float],
    initial_direction: tuple[float, float, float],
    segment_length: float,
    curvature: float,
    twist: float = 0.0,
    fix_first: bool = True,
    material: Optional[MaterialProperties] = None,
    config: Optional[SimulationConfig] = None,
) -> CosseratRodModel:
    """Create a curved rod with constant curvature.

    Args:
        n_particles: Number of particles.
        start_pos: Starting position (x, y, z).
        initial_direction: Initial tangent direction.
        segment_length: Length of each segment.
        curvature: Constant curvature (1/radius).
        twist: Constant twist rate.
        fix_first: Whether to fix the first particle.
        material: Material properties (uses defaults if None).
        config: Simulation configuration (uses defaults if None).

    Returns:
        Initialized CosseratRodModel with curved rest shape.
    """
    # Start with a straight rod
    model = create_straight_rod(
        n_particles=n_particles,
        start_pos=start_pos,
        direction=initial_direction,
        segment_length=segment_length,
        fix_first=fix_first,
        material=material,
        config=config,
    )

    # Set rest curvature
    model.set_rest_curvature(curvature, 0.0, twist)

    return model
