# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Cosserat rod simulation using DefKit DLL."""

import numpy as np

from .defkit_wrapper import DefKitWrapper
from .rod_state import RodState


class CosseratRodSimulation:
    """Cosserat rod simulation using DefKit DLL backend.

    This class orchestrates the Position and Orientation Based Cosserat Rods
    simulation using the native DLL functions for constraint projection.

    The simulation loop follows the standard PBD pattern:
    1. Predict positions and orientations (semi-implicit Euler)
    2. Project constraints iteratively
    3. Integrate (update positions/orientations and derive velocities)
    """

    def __init__(self, state: RodState, dll_path: str = "unity_ref"):
        """Initialize simulation.

        Args:
            state: Rod state to simulate.
            dll_path: Path to directory containing DefKit DLLs.
        """
        self.dll = DefKitWrapper(dll_path)
        self.state = state

        # Simulation parameters
        self.position_damping = 0.001
        self.rotation_damping = 0.001
        self.gravity = np.array([0.0, 0.0, -9.81, 0.0], dtype=np.float32)
        self.constraint_iterations = 4

    def step(self, dt: float):
        """Advance simulation by one timestep.

        Args:
            dt: Time step size in seconds.
        """
        s = self.state

        # 1. Predict positions (apply gravity and external forces)
        self.dll.predict_positions(
            dt,
            self.position_damping,
            s.positions,
            s.predicted_positions,
            s.velocities,
            s.forces,
            s.inv_masses,
            self.gravity,
        )

        # 2. Predict orientations (apply external torques)
        self.dll.predict_rotations(
            dt,
            self.rotation_damping,
            s.orientations,
            s.predicted_orientations,
            s.angular_velocities,
            s.torques,
            s.quat_inv_masses,
        )

        # 3. Project constraints iteratively
        for _ in range(self.constraint_iterations):
            self.dll.project_elastic_rod_constraints(
                s.predicted_positions,
                s.predicted_orientations,
                s.inv_masses,
                s.quat_inv_masses,
                s.rest_darboux,
                s.bend_twist_ks,
                s.rest_lengths,
                s.stretch_ks,
                s.bend_twist_ks_mult,
            )

        # 4. Integrate positions (update positions and velocities)
        self.dll.integrate_positions(
            dt,
            s.positions,
            s.predicted_positions,
            s.velocities,
            s.inv_masses,
        )

        # 5. Integrate orientations (update orientations and angular velocities)
        self.dll.integrate_rotations(
            dt,
            s.orientations,
            s.predicted_orientations,
            s.prev_orientations,
            s.angular_velocities,
            s.quat_inv_masses,
        )

        # 6. Clear forces for next step
        s.clear_forces()

    def set_gravity(self, gx: float, gy: float, gz: float):
        """Set gravity vector.

        Args:
            gx, gy, gz: Gravity components.
        """
        self.gravity[0] = gx
        self.gravity[1] = gy
        self.gravity[2] = gz

    def apply_force(self, particle_index: int, fx: float, fy: float, fz: float):
        """Apply external force to a particle.

        Args:
            particle_index: Index of the particle.
            fx, fy, fz: Force components.
        """
        self.state.forces[particle_index, 0] += fx
        self.state.forces[particle_index, 1] += fy
        self.state.forces[particle_index, 2] += fz

    def apply_torque(self, particle_index: int, tx: float, ty: float, tz: float):
        """Apply external torque to a particle.

        Args:
            particle_index: Index of the particle.
            tx, ty, tz: Torque components.
        """
        self.state.torques[particle_index, 0] += tx
        self.state.torques[particle_index, 1] += ty
        self.state.torques[particle_index, 2] += tz
