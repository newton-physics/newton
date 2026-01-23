# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Direct Cosserat rod simulation using DefKit DLL.

This implements the "Direct Position-Based Solver for Stiff Rods" from
Deul et al., which uses a global (direct) solve instead of iterative
constraint projection for better handling of stiff rods.
"""

import numpy as np

from .defkit_wrapper import DefKitWrapper
from .rod_state import RodState


class DirectCosseratRodSimulation:
    """Direct Cosserat rod simulation using DefKit DLL backend.

    This class implements the Direct Position-Based Solver for Stiff Rods,
    which solves the constraint system globally using a banded matrix solver.
    This is more accurate for stiff rods compared to the iterative solver.

    The simulation loop:
    1. Predict positions and orientations (semi-implicit Euler)
    2. Prepare constraints (reset lambdas, update stiffness)
    3. Update constraint state
    4. Compute Jacobians
    5. Assemble JMJT banded matrix
    6. Solve system and apply corrections
    7. Integrate (update positions/orientations and derive velocities)
    """

    def __init__(self, state: RodState, dll_path: str = "unity_ref"):
        """Initialize direct solver simulation.

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

        # Material parameters
        self.radius = 0.01  # Rod cross-section radius
        self.young_modulus = 1.0  # Base Young's modulus (set to 1 so multiplier = effective value)
        self.torsion_modulus = 1.0  # Base torsion modulus (set to 1 so multiplier = effective value)
        self.young_modulus_mult = 1.0e6  # Effective Young's modulus (Pa)
        self.torsion_modulus_mult = 1.0e6  # Effective torsion modulus (Pa)

        # Direct solver uses Vector3 for rest Darboux (not quaternion)
        # Shape: (n_edges, 4) but only first 3 components used
        self.rest_darboux_vec = np.zeros((state.n_edges, 4), dtype=np.float32)

        # Bend stiffness coefficients (kappa1, kappa2, tau stiffness)
        self.bend_stiffness = np.ones((state.n_edges, 4), dtype=np.float32)

        # Initialize the direct solver
        self._rod_ptr = self.dll.init_direct_elastic_rod(
            state.positions,
            state.orientations,
            self.radius,
            state.rest_lengths,
            self.young_modulus,
            self.torsion_modulus,
        )

    def __del__(self):
        """Clean up native resources."""
        if hasattr(self, "_rod_ptr") and self._rod_ptr is not None:
            self.dll.destroy_direct_elastic_rod(self._rod_ptr)
            self._rod_ptr = None

    def step(self, dt: float):
        """Advance simulation by one timestep.

        Args:
            dt: Time step size in seconds.
        """
        s = self.state
        n_constraints = s.n_edges

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

        # 3. Prepare direct solver for this timestep
        self.dll.prepare_direct_elastic_rod_constraints(
            self._rod_ptr,
            n_constraints,
            dt,
            self.bend_stiffness,
            self.rest_darboux_vec,
            s.rest_lengths,
            self.young_modulus_mult,
            self.torsion_modulus_mult,
        )

        # 4. Update constraint state with predicted positions/orientations
        self.dll.update_direct_constraints(
            self._rod_ptr,
            s.predicted_positions,
            s.predicted_orientations,
            s.inv_masses,
        )

        # 5. Compute Jacobians (for all constraints)
        self.dll.compute_jacobians_direct(
            self._rod_ptr,
            0,  # start_id
            n_constraints,  # count
            s.predicted_positions,
            s.predicted_orientations,
            s.inv_masses,
        )

        # 6. Assemble JMJT banded matrix
        self.dll.assemble_jmjt_direct(
            self._rod_ptr,
            0,  # start_id
            n_constraints,  # count
            s.predicted_positions,
            s.predicted_orientations,
            s.inv_masses,
        )

        # 7. Solve and apply corrections (modifies predicted_positions/orientations in-place)
        self.dll.solve_direct_constraints(
            self._rod_ptr,
            s.predicted_positions,
            s.predicted_orientations,
            s.inv_masses,
        )

        # 8. Integrate positions (update positions and velocities)
        self.dll.integrate_positions(
            dt,
            s.positions,
            s.predicted_positions,
            s.velocities,
            s.inv_masses,
        )

        # 9. Integrate orientations (update orientations and angular velocities)
        self.dll.integrate_rotations(
            dt,
            s.orientations,
            s.predicted_orientations,
            s.prev_orientations,
            s.angular_velocities,
            s.quat_inv_masses,
        )

        # 10. Clear forces for next step
        s.clear_forces()

    def set_gravity(self, gx: float, gy: float, gz: float):
        """Set gravity vector.

        Args:
            gx, gy, gz: Gravity components.
        """
        self.gravity[0] = gx
        self.gravity[1] = gy
        self.gravity[2] = gz

    def set_rest_curvature(self, kappa1: float, kappa2: float, tau: float):
        """Set uniform rest curvature for the entire rod.

        Args:
            kappa1: Bending curvature around local X axis
            kappa2: Bending curvature around local Y axis
            tau: Twist around local Z axis
        """
        self.rest_darboux_vec[:, 0] = kappa1
        self.rest_darboux_vec[:, 1] = kappa2
        self.rest_darboux_vec[:, 2] = tau

    def set_bend_stiffness(self, k1: float, k2: float, k_tau: float):
        """Set uniform bending/twist stiffness for the entire rod.

        Args:
            k1: Stiffness for bending around local X axis
            k2: Stiffness for bending around local Y axis
            k_tau: Stiffness for twist around local Z axis
        """
        self.bend_stiffness[:, 0] = k1
        self.bend_stiffness[:, 1] = k2
        self.bend_stiffness[:, 2] = k_tau
