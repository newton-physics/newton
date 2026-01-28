# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Abstract base class for Cosserat rod state."""

from __future__ import annotations

import math
from abc import ABC, abstractmethod

import numpy as np

from newton.examples.cosserat_codex.math_utils import quat_from_axis_angle


class RodStateBase(ABC):
    """Abstract base class defining the interface and shared data for rod state.
    
    This class provides the common data structures and methods shared by all
    rod implementations (C++ DLL, NumPy, Warp GPU).
    
    Attributes:
        num_points: Number of particles in the rod.
        num_edges: Number of edges (segments) in the rod.
        segment_length: Rest length of each segment.
        rod_radius: Radius of the rod for collision detection.
        young_modulus: Young's modulus for stretch stiffness.
        torsion_modulus: Torsion modulus for twist stiffness.
        positions: Particle positions as (num_points, 4) array.
        predicted_positions: Predicted positions for constraint projection.
        velocities: Particle velocities as (num_points, 4) array.
        forces: External forces on particles.
        orientations: Particle orientations as (num_points, 4) quaternions [x, y, z, w].
        predicted_orientations: Predicted orientations for constraint projection.
        prev_orientations: Previous orientations for angular velocity computation.
        angular_velocities: Angular velocities as (num_points, 4) array.
        torques: External torques on particles.
        inv_masses: Inverse masses (0 for fixed particles).
        quat_inv_masses: Inverse rotational masses (0 for rotation-locked particles).
        rest_lengths: Rest length for each edge.
        rest_darboux: Rest Darboux vector (curvature) for each edge.
        bend_stiffness: Bend/twist stiffness coefficients for each edge.
        gravity: Gravity vector as (1, 4) array.
        root_locked: Whether the root particle is position-locked.
    """

    def __init__(
        self,
        num_points: int,
        segment_length: float,
        mass: float,
        particle_height: float,
        rod_radius: float,
        bend_stiffness: float,
        twist_stiffness: float,
        rest_bend_d1: float,
        rest_bend_d2: float,
        rest_twist: float,
        young_modulus: float,
        torsion_modulus: float,
        gravity: np.ndarray,
        lock_root_rotation: bool,
    ):
        """Initialize rod state with common data structures.
        
        Args:
            num_points: Number of particles in the rod.
            segment_length: Rest length of each segment.
            mass: Mass of each particle.
            particle_height: Initial Z height of particles.
            rod_radius: Radius of the rod for collision detection.
            bend_stiffness: Bending stiffness coefficient.
            twist_stiffness: Twist stiffness coefficient.
            rest_bend_d1: Rest curvature in d1 direction.
            rest_bend_d2: Rest curvature in d2 direction.
            rest_twist: Rest twist angle.
            young_modulus: Young's modulus for stretch stiffness.
            torsion_modulus: Torsion modulus for twist stiffness.
            gravity: 3D gravity vector.
            lock_root_rotation: Whether to lock root particle rotation.
        """
        self.num_points = num_points
        self.num_edges = max(0, num_points - 1)
        self.segment_length = segment_length
        self.rod_radius = rod_radius
        self.young_modulus = young_modulus
        self.torsion_modulus = torsion_modulus

        # Position and velocity arrays (4-component for alignment)
        self.positions = np.zeros((num_points, 4), dtype=np.float32)
        self.predicted_positions = np.zeros((num_points, 4), dtype=np.float32)
        self.velocities = np.zeros((num_points, 4), dtype=np.float32)
        self.forces = np.zeros((num_points, 4), dtype=np.float32)

        # Initialize positions along X-axis at specified height
        for i in range(num_points):
            self.positions[i, 0] = i * segment_length
            self.positions[i, 2] = particle_height
        self.predicted_positions[:] = self.positions

        # Orientation arrays - align rod along X-axis (rotate from Z to X)
        q_align = quat_from_axis_angle(np.array([0.0, 1.0, 0.0], dtype=np.float32), math.pi * 0.5)
        self.orientations = np.tile(q_align, (num_points, 1)).astype(np.float32)
        self.predicted_orientations = self.orientations.copy()
        self.prev_orientations = self.orientations.copy()

        # Angular velocity and torque arrays
        self.angular_velocities = np.zeros((num_points, 4), dtype=np.float32)
        self.torques = np.zeros((num_points, 4), dtype=np.float32)

        # Mass and inertia
        inv_mass_value = 0.0 if mass == 0.0 else 1.0 / mass
        self.inv_masses = np.full(num_points, inv_mass_value, dtype=np.float32)
        self.inv_masses[0] = 0.0  # Root is fixed by default
        self._root_inv_mass_unlocked = inv_mass_value

        self.quat_inv_masses = np.full(num_points, 1.0, dtype=np.float32)
        # If inv_mass is 0 (static), rotation is also locked
        static_mask = self.inv_masses == 0.0
        self.quat_inv_masses[static_mask] = 0.0
        self._root_quat_inv_mass_unlocked = np.float32(1.0)

        if lock_root_rotation:
            self.quat_inv_masses[0] = 0.0
        self.root_locked = True

        # Edge properties
        self.rest_lengths = np.full(self.num_edges, segment_length, dtype=np.float32)

        self.rest_darboux = np.zeros((self.num_edges, 4), dtype=np.float32)
        self.set_rest_darboux(rest_bend_d1, rest_bend_d2, rest_twist)

        self.bend_stiffness = np.zeros((self.num_edges, 4), dtype=np.float32)
        self.set_bend_stiffness(bend_stiffness, twist_stiffness)

        # Correction arrays for constraint projection
        self.pos_corrections = np.zeros((num_points, 4), dtype=np.float32)
        self.rot_corrections = np.zeros((num_points, 4), dtype=np.float32)

        # Gravity
        self.gravity = np.zeros((1, 4), dtype=np.float32)
        self.set_gravity(gravity)

        # Store initial state for reset
        self._initial_positions = self.positions.copy()
        self._initial_orientations = self.orientations.copy()

    def set_gravity(self, gravity: np.ndarray) -> None:
        """Set the gravity vector.
        
        Args:
            gravity: 3D gravity vector.
        """
        self.gravity[0, 0:3] = gravity.astype(np.float32)

    def set_bend_stiffness(self, bend_stiffness: float, twist_stiffness: float) -> None:
        """Set bend and twist stiffness coefficients.
        
        Args:
            bend_stiffness: Bending stiffness coefficient.
            twist_stiffness: Twist stiffness coefficient.
        """
        self.bend_stiffness[:, 0] = bend_stiffness
        self.bend_stiffness[:, 1] = bend_stiffness
        self.bend_stiffness[:, 2] = twist_stiffness

    def set_rest_darboux(self, rest_bend_d1: float, rest_bend_d2: float, rest_twist: float) -> None:
        """Set rest curvature (Darboux vector) for all edges.
        
        Args:
            rest_bend_d1: Rest curvature in d1 direction.
            rest_bend_d2: Rest curvature in d2 direction.
            rest_twist: Rest twist angle.
        """
        self.rest_darboux[:, 0] = rest_bend_d1
        self.rest_darboux[:, 1] = rest_bend_d2
        self.rest_darboux[:, 2] = rest_twist

    def set_root_locked(self, locked: bool) -> None:
        """Lock or unlock the root particle position and rotation.
        
        Args:
            locked: Whether to lock the root particle.
        """
        self.root_locked = locked
        if locked:
            self.inv_masses[0] = 0.0
            self.quat_inv_masses[0] = 0.0
            self.velocities[0, 0:3] = 0.0
            self.angular_velocities[0, 0:3] = 0.0
        else:
            self.inv_masses[0] = self._root_inv_mass_unlocked
            self.quat_inv_masses[0] = self._root_quat_inv_mass_unlocked

    def toggle_root_lock(self) -> None:
        """Toggle the root particle lock state."""
        self.set_root_locked(not self.root_locked)

    @abstractmethod
    def step(self, dt: float, linear_damping: float, angular_damping: float) -> None:
        """Advance the simulation by one time step.
        
        Args:
            dt: Time step size in seconds.
            linear_damping: Linear velocity damping factor.
            angular_damping: Angular velocity damping factor.
        """
        pass

    @abstractmethod
    def reset(self) -> None:
        """Reset the rod to its initial state."""
        pass

    @abstractmethod
    def apply_floor_collisions(self, floor_z: float, restitution: float = 0.0) -> None:
        """Apply floor collision constraints.
        
        Args:
            floor_z: Z coordinate of the floor plane.
            restitution: Coefficient of restitution for bouncing.
        """
        pass

    def destroy(self) -> None:
        """Clean up any allocated resources.
        
        Override this method in subclasses that allocate external resources.
        """
        pass


__all__ = ["RodStateBase"]
