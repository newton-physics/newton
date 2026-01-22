# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Direct Position Based Solver for Stiff Rods wrapper.

This module provides a constraint solver that uses the direct solver from
Deul et al. 2017 "Direct Position-Based Solver for Stiff Rods".

Unlike iterative PBD solvers, the direct solver formulates the entire
constraint system as a single linear system solved via LDLT factorization.
This provides better handling of stiff constraints.

Reference:
    https://animation.rwth-aachen.de/publication/0557/

Note: This solver runs on CPU and is significantly slower than GPU-based solvers.
It is primarily intended for validation and testing.
"""

from typing import TYPE_CHECKING

import numpy as np
import warp as wp

from newton.examples.cosserat2.reference.direct_solver_numpy import (
    DirectPositionBasedSolverForStiffRods,
    RodSegment,
)
from newton.examples.cosserat2.reference.quaternion_ops import (
    quat_multiply,
    quat_normalize,
    quat_to_rotation_matrix,
)
from newton.examples.cosserat2.solvers.base import (
    ConstraintSolverBase,
    ConstraintSolverType,
    FrictionMethod,
)

if TYPE_CHECKING:
    from newton.examples.cosserat2.cosserat_rod import CosseratRod


class ConstraintSolverDirectStiffRods(ConstraintSolverBase):
    """Direct solver for stiff rod constraints wrapper.

    This solver wraps the pure NumPy implementation of the Direct Position
    Based Solver for Stiff Rods, allowing it to be used with the Warp-based
    simulation framework.

    The direct solver builds a tree structure of segments and constraints,
    then solves the coupled system using LDLT factorization.

    Args:
        rod: The CosseratRod data structure.
        device: Warp device (used for array transfers).
        youngs_modulus: Young's modulus (Pa) for bending stiffness. Default 10 GPa.
        torsion_modulus: Torsion/shear modulus (Pa). Default 5 GPa.
        radius_fraction: Fraction of segment length to use as rod radius.
            Default 0.25, matching the C++ reference implementation.
    """

    def __init__(
        self,
        rod: "CosseratRod",
        device: str = "cuda:0",
        youngs_modulus: float = 1.0e10,
        torsion_modulus: float = 0.5e10,
        radius_fraction: float = 0.25,
    ):
        super().__init__(rod, device)

        self._youngs_modulus = youngs_modulus
        self._torsion_modulus = torsion_modulus
        self._radius_fraction = radius_fraction

        # Create internal direct solver
        self._direct_solver = DirectPositionBasedSolverForStiffRods()

        # Pre-allocate NumPy arrays for data transfer
        self._particle_q_np = np.zeros((self.num_particles, 3), dtype=np.float64)
        self._edge_q_np = np.zeros((self.num_stretch, 4), dtype=np.float64)

        # Cache for rod properties
        self._rest_lengths_np = rod.rest_length.numpy().astype(np.float64)
        self._particle_inv_mass_np = rod.particle_inv_mass.numpy().astype(np.float64)
        self._edge_inv_mass_np = rod.edge_inv_mass.numpy().astype(np.float64)
        self._rest_darboux_np = rod.rest_darboux.numpy().astype(np.float64)

        # Initialize will happen on first solve (needs particle positions)
        self._initialized = False
        self._dt = 1.0 / 60.0

        # Store reference to output array for particle position propagation
        self._particle_q_out_ref = None

    @property
    def solver_type(self) -> ConstraintSolverType:
        """Return the solver type enum value."""
        return ConstraintSolverType.DIRECT_STIFF_RODS

    def _initialize_solver(self) -> None:
        """Initialize the direct solver with current rod state."""
        solver = self._direct_solver

        # Create segments from rod edges
        solver.segments = []
        for i in range(self.num_stretch):
            p0 = self._particle_q_np[i]
            p1 = self._particle_q_np[i + 1]
            center = 0.5 * (p0 + p1)

            rotation = self._edge_q_np[i].copy()

            # Compute effective mass
            # A segment is static if its edge quaternion is fixed (edge_inv_mass = 0)
            # This ensures proper tree rooting in the direct solver
            if self._edge_inv_mass_np[i] <= 0:
                mass = 0.0  # Static segment (fixed quaternion)
            else:
                m0 = 1.0 / self._particle_inv_mass_np[i] if self._particle_inv_mass_np[i] > 0 else 0.0
                m1 = 1.0 / self._particle_inv_mass_np[i + 1] if self._particle_inv_mass_np[i + 1] > 0 else 0.0
                mass = m0 + m1

            # Simple inertia estimate using radius proportional to segment length
            length = self._rest_lengths_np[i]
            radius = self._radius_fraction * length
            Ix = (1.0 / 12.0) * mass * (3 * radius**2 + length**2) if mass > 0 else 1.0
            Iz = 0.5 * mass * radius**2 if mass > 0 else 1.0
            inertia = np.array([Ix, Iz, Ix])

            segment = RodSegment(
                index=i,
                position=center,
                rotation=rotation,
                mass=mass,
                inertia_tensor=inertia,
            )
            solver.segments.append(segment)

        # Create constraints between adjacent segments
        solver.constraints = []
        for i in range(self.num_bend):
            constraint_pos = self._particle_q_np[i + 1].copy()

            seg0 = solver.segments[i]
            seg1 = solver.segments[i + 1]

            avg_length = 0.5 * (self._rest_lengths_np[i] + self._rest_lengths_np[i + 1])
            avg_radius = self._radius_fraction * avg_length

            constraint = solver.init_constraint(
                segment0=seg0,
                segment1=seg1,
                constraint_position=constraint_pos,
                average_radius=avg_radius,
                average_segment_length=avg_length,
                youngs_modulus=self._youngs_modulus,
                torsion_modulus=self._torsion_modulus,
            )

            # Override rest Darboux
            if i < len(self._rest_darboux_np):
                rest_q = self._rest_darboux_np[i]
                constraint.rest_darboux_vector = (2.0 / avg_length) * rest_q[:3] / (np.abs(rest_q[3]) + 1e-10)

            solver.constraints.append(constraint)

        # Initialize tree structure
        if len(solver.constraints) > 0:
            solver.init_tree()

        # Allocate working arrays
        n_constraints = len(solver.constraints)
        n_segments = len(solver.segments)

        solver.rhs = [np.zeros(6) for _ in range(n_constraints)]
        solver.lambda_sums = [np.zeros(6) for _ in range(n_constraints)]
        solver.bending_torsion_jacobians = [[np.zeros((3, 3)), np.zeros((3, 3))] for _ in range(n_constraints)]
        solver.corr_x = [np.zeros(3) for _ in range(n_segments)]
        solver.corr_q = [np.zeros(4) for _ in range(n_segments)]

        self._initialized = True

    def _sync_segments_from_arrays(self) -> None:
        """Update segment positions and rotations from numpy arrays.

        Static segments (those with mass=0) are NOT updated - they serve as
        fixed anchors in the direct solver's tree structure.

        For dynamic segments, rotations are computed from particle positions
        using a minimal rotation approach to preserve twist.
        """
        for i, segment in enumerate(self._direct_solver.segments):
            # Only update dynamic segments - static segments stay fixed
            if segment.is_dynamic():
                p0 = self._particle_q_np[i]
                p1 = self._particle_q_np[i + 1]
                segment.position = 0.5 * (p0 + p1)

                # Compute segment direction from particle positions
                edge_dir = p1 - p0
                edge_len = np.linalg.norm(edge_dir)
                if edge_len > 1e-10:
                    edge_dir = edge_dir / edge_len
                else:
                    edge_dir = np.array([0.0, 0.0, 1.0])

                # Get the current d3 (local Z-axis) from stored quaternion
                q_stored = self._edge_q_np[i]
                R_stored = quat_to_rotation_matrix(q_stored)
                d3_stored = R_stored[:, 2]

                # Compute minimal rotation from d3_stored to edge_dir
                segment.rotation = self._compute_minimal_rotation(
                    q_stored, d3_stored, edge_dir
                )

    def _compute_minimal_rotation(
        self, q_base: np.ndarray, from_dir: np.ndarray, to_dir: np.ndarray
    ) -> np.ndarray:
        """Compute minimal rotation quaternion to align directions.

        Args:
            q_base: Base quaternion.
            from_dir: Current direction (unit vector).
            to_dir: Target direction (unit vector).

        Returns:
            Updated quaternion.
        """
        dot = np.clip(np.dot(from_dir, to_dir), -1.0, 1.0)

        if dot > 0.9999:
            return q_base.copy()
        elif dot < -0.9999:
            perp = np.array([1.0, 0.0, 0.0])
            if abs(np.dot(from_dir, perp)) > 0.9:
                perp = np.array([0.0, 1.0, 0.0])
            axis = np.cross(from_dir, perp)
            axis = axis / np.linalg.norm(axis)
            q_rot = np.array([axis[0], axis[1], axis[2], 0.0])
        else:
            axis = np.cross(from_dir, to_dir)
            axis_len = np.linalg.norm(axis)
            if axis_len > 1e-10:
                axis = axis / axis_len
            angle = np.arccos(dot)
            half_angle = angle / 2.0
            sin_half = np.sin(half_angle)
            cos_half = np.cos(half_angle)
            q_rot = np.array([
                axis[0] * sin_half,
                axis[1] * sin_half,
                axis[2] * sin_half,
                cos_half,
            ])

        return quat_normalize(quat_multiply(q_rot, q_base))

    def _sync_arrays_from_segments(self) -> None:
        """Update numpy arrays from segment positions and rotations."""
        solver = self._direct_solver

        # Update edge quaternions
        for i, segment in enumerate(solver.segments):
            self._edge_q_np[i] = quat_normalize(segment.rotation)

        # Update constraint world-space connectors
        for constraint in solver.constraints:
            solver.update_constraint(constraint)

        # Extract particle positions from constraint connectors
        # Particle 0 stays fixed (it's the root)
        for i, constraint in enumerate(solver.constraints):
            particle_idx = i + 1  # Constraint i gives us particle i+1
            # Use average of both connectors
            connector0 = constraint.constraint_info[:, 2]
            connector1 = constraint.constraint_info[:, 3]
            self._particle_q_np[particle_idx] = 0.5 * (connector0 + connector1)

        # The last particle (tip) is computed from the last segment
        if self.num_stretch > 0:
            last_seg_idx = self.num_stretch - 1
            last_segment = solver.segments[last_seg_idx]
            last_particle_idx = self.num_particles - 1
            prev_particle = self._particle_q_np[last_particle_idx - 1]
            self._particle_q_np[last_particle_idx] = 2 * last_segment.position - prev_particle

    def _apply_segment_corrections(self) -> None:
        """Apply position and rotation corrections to segments."""
        for i, segment in enumerate(self._direct_solver.segments):
            if segment.is_dynamic():
                segment.position = segment.position + self._direct_solver.corr_x[i]
                delta_q = self._direct_solver.corr_q[i]
                new_q = segment.rotation + delta_q
                segment.rotation = quat_normalize(new_q)

    def solve_stretch_shear(
        self,
        particle_q: wp.array,
        particle_q_out: wp.array,
        stretch_shear_stiffness: wp.vec3,
        **kwargs,
    ):
        """Solve stretch/shear constraints using the direct solver.

        Note: The direct solver solves stretch and bend/twist constraints
        together in a coupled system. This method handles the stretch part
        and stores the particle positions for the combined solve.

        Args:
            particle_q: Current particle positions [num_particles].
            particle_q_out: Output corrected positions [num_particles].
            stretch_shear_stiffness: Stiffness vector (ignored, uses material params).
        """
        # Copy data from Warp to NumPy
        particle_q_warp = particle_q.numpy()
        self._particle_q_np[:] = particle_q_warp

        edge_q_warp = self.rod.edge_q.numpy()
        self._edge_q_np[:] = edge_q_warp

        # Re-read rest Darboux (may have been updated by UI)
        self._rest_darboux_np = self.rod.rest_darboux.numpy().astype(np.float64)

        # Initialize solver on first call
        if not self._initialized:
            self._initialize_solver()

        # Sync segments from rod state
        self._sync_segments_from_arrays()

        # Store reference to output array - actual solving happens in solve_bend_twist
        # since the direct solver couples stretch and bend/twist constraints
        self._particle_q_out_ref = particle_q_out

        # Copy current positions to output (will be updated in solve_bend_twist)
        particle_q_out.assign(wp.array(self._particle_q_np, dtype=wp.vec3, device=self.device))

    def solve_bend_twist(
        self,
        bend_twist_stiffness: wp.vec3,
        friction_method: FrictionMethod,
        friction_params: dict,
        dt: float,
    ):
        """Solve bend/twist constraints using the direct solver.

        The direct solver solves all constraints (stretch + bend/twist) together
        in a single coupled system. This method performs the actual solve.

        Args:
            bend_twist_stiffness: Stiffness vector (ignored, uses material params).
            friction_method: Friction model (ignored - not supported).
            friction_params: Friction parameters (ignored).
            dt: Time step.
        """
        if friction_method != FrictionMethod.NONE:
            if not hasattr(self, "_friction_warning_shown"):
                print("Warning: Direct stiff rods solver does not support friction methods. Ignoring.")
                self._friction_warning_shown = True

        if self.num_bend == 0 or not self._initialized:
            return

        self._dt = dt

        # Initialize solver for this time step
        self._direct_solver.init_before_projection(1.0 / dt)

        # Update rest Darboux in constraints (may have been changed by UI)
        for i, constraint in enumerate(self._direct_solver.constraints):
            if i < len(self._rest_darboux_np):
                rest_q = self._rest_darboux_np[i]
                avg_length = constraint.average_segment_length
                constraint.rest_darboux_vector = (2.0 / avg_length) * rest_q[:3] / (np.abs(rest_q[3]) + 1e-10)

        # Factor and solve the system
        self._direct_solver.factor()
        self._direct_solver.solve()

        # Apply corrections
        self._apply_segment_corrections()

        # Sync back to arrays
        self._sync_arrays_from_segments()

        # Copy results back to Warp
        self.rod.edge_q.assign(wp.array(self._edge_q_np, dtype=wp.quat, device=self.device))

        # CRITICAL: Copy updated particle positions to output array
        # The direct solver modifies segment positions which updates particle positions
        if self._particle_q_out_ref is not None:
            self._particle_q_out_ref.assign(wp.array(self._particle_q_np, dtype=wp.vec3, device=self.device))
