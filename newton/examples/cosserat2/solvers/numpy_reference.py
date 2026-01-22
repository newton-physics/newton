# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""NumPy reference solver wrapper for integration with Warp-based simulation.

This module provides a constraint solver that uses the pure NumPy reference
implementation from the `reference` subpackage. It bridges between Warp arrays
and NumPy arrays, allowing the reference implementation to be used as a
drop-in replacement for the GPU-based solvers.

This is useful for:
- Validation/testing against a known-correct implementation
- Debugging without GPU complexity
- Educational purposes (understanding the algorithm)

Note: This solver runs on CPU and is significantly slower than GPU-based solvers.
"""

from typing import TYPE_CHECKING

import numpy as np
import warp as wp

from newton.examples.cosserat2.reference.quaternion_ops import quat_normalize
from newton.examples.cosserat2.reference.solver_numpy import PositionBasedCosseratRods
from newton.examples.cosserat2.solvers.base import (
    ConstraintSolverBase,
    ConstraintSolverType,
    FrictionMethod,
)

if TYPE_CHECKING:
    from newton.examples.cosserat2.cosserat_rod import CosseratRod


class ConstraintSolverNumpyReference(ConstraintSolverBase):
    """NumPy reference implementation solver wrapper.

    This solver wraps the pure NumPy implementation from the `reference`
    subpackage, allowing it to be used with the Warp-based simulation.

    Data is copied between Warp and NumPy arrays for each solve call.
    This adds overhead but allows the reference implementation to be
    used for validation and debugging.

    Args:
        rod: The CosseratRod data structure.
        device: Warp device (used for array transfers).
    """

    def __init__(self, rod: "CosseratRod", device: str = "cuda:0"):
        super().__init__(rod, device)

        # Pre-allocate NumPy arrays for data transfer
        self._particle_q_np = np.zeros((self.num_particles, 3), dtype=np.float64)
        self._particle_q_out_np = np.zeros((self.num_particles, 3), dtype=np.float64)
        self._edge_q_np = np.zeros((self.num_stretch, 4), dtype=np.float64)

        # Cache for rest lengths and inverse masses (copy once from Warp)
        self._rest_lengths_np = rod.rest_length.numpy().astype(np.float64)
        self._particle_inv_mass_np = rod.particle_inv_mass.numpy().astype(np.float64)
        self._edge_inv_mass_np = rod.edge_inv_mass.numpy().astype(np.float64)
        self._rest_darboux_np = rod.rest_darboux.numpy().astype(np.float64)

    @property
    def solver_type(self) -> ConstraintSolverType:
        """Return the solver type enum value."""
        return ConstraintSolverType.NUMPY_REFERENCE

    def solve_stretch_shear(
        self,
        particle_q: wp.array,
        particle_q_out: wp.array,
        stretch_shear_stiffness: wp.vec3,
        **kwargs,
    ):
        """Solve stretch/shear constraints using NumPy reference implementation.

        Args:
            particle_q: Current particle positions [num_particles].
            particle_q_out: Output corrected positions [num_particles].
            stretch_shear_stiffness: Stiffness vector (shear_d1, shear_d2, stretch_d3).
        """
        # Copy data from Warp to NumPy
        particle_q_warp = particle_q.numpy()
        self._particle_q_np[:] = particle_q_warp

        edge_q_warp = self.rod.edge_q.numpy()
        self._edge_q_np[:] = edge_q_warp

        # Convert stiffness to numpy
        ks = np.array([stretch_shear_stiffness[0], stretch_shear_stiffness[1], stretch_shear_stiffness[2]])

        # Solve each stretch/shear constraint (Gauss-Seidel)
        for i in range(self.num_stretch):
            p0 = self._particle_q_np[i]
            p1 = self._particle_q_np[i + 1]
            q0 = self._edge_q_np[i]

            inv_mass_p0 = self._particle_inv_mass_np[i]
            inv_mass_p1 = self._particle_inv_mass_np[i + 1]
            inv_mass_q0 = self._edge_inv_mass_np[i]
            rest_length = self._rest_lengths_np[i]

            corr_p0, corr_p1, corr_q0 = PositionBasedCosseratRods.solve_stretch_shear_constraint(
                p0,
                inv_mass_p0,
                p1,
                inv_mass_p1,
                q0,
                inv_mass_q0,
                ks,
                rest_length,
            )

            # Apply corrections immediately (Gauss-Seidel)
            self._particle_q_np[i] += corr_p0
            self._particle_q_np[i + 1] += corr_p1
            self._edge_q_np[i] = quat_normalize(self._edge_q_np[i] + corr_q0)

        # Copy results back to Warp
        particle_q_out.assign(wp.array(self._particle_q_np, dtype=wp.vec3, device=self.device))
        self.rod.edge_q.assign(wp.array(self._edge_q_np, dtype=wp.quat, device=self.device))

    def solve_bend_twist(
        self,
        bend_twist_stiffness: wp.vec3,
        friction_method: FrictionMethod,
        friction_params: dict,
        dt: float,
    ):
        """Solve bend/twist constraints using NumPy reference implementation.

        Note: Friction methods are not supported in the NumPy reference.
        They are ignored and a warning is printed on first call.

        Args:
            bend_twist_stiffness: Stiffness vector (bend_d1, twist, bend_d2).
            friction_method: Which friction model to use (ignored).
            friction_params: Parameters for the friction model (ignored).
            dt: Time step for friction calculations (ignored).
        """
        if friction_method != FrictionMethod.NONE:
            if not hasattr(self, "_friction_warning_shown"):
                print("Warning: NumPy reference solver does not support friction methods. Ignoring.")
                self._friction_warning_shown = True

        if self.num_bend == 0:
            return

        # Re-read edge quaternions (may have been updated by stretch/shear)
        edge_q_warp = self.rod.edge_q.numpy()
        self._edge_q_np[:] = edge_q_warp

        # Re-read rest Darboux (may have been updated by UI)
        self._rest_darboux_np = self.rod.rest_darboux.numpy().astype(np.float64)

        # Convert stiffness to numpy
        ks = np.array([bend_twist_stiffness[0], bend_twist_stiffness[1], bend_twist_stiffness[2]])

        # Solve each bend/twist constraint (Gauss-Seidel)
        for i in range(self.num_bend):
            q0 = self._edge_q_np[i]
            q1 = self._edge_q_np[i + 1]

            inv_mass_q0 = self._edge_inv_mass_np[i]
            inv_mass_q1 = self._edge_inv_mass_np[i + 1]
            rest_darboux = self._rest_darboux_np[i]

            corr_q0, corr_q1 = PositionBasedCosseratRods.solve_bend_twist_constraint(
                q0,
                inv_mass_q0,
                q1,
                inv_mass_q1,
                ks,
                rest_darboux,
            )

            # Apply corrections immediately (Gauss-Seidel)
            self._edge_q_np[i] = quat_normalize(self._edge_q_np[i] + corr_q0)
            self._edge_q_np[i + 1] = quat_normalize(self._edge_q_np[i + 1] + corr_q1)

        # Copy results back to Warp
        self.rod.edge_q.assign(wp.array(self._edge_q_np, dtype=wp.quat, device=self.device))
