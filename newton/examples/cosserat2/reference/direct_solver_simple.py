# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Simplified Direct Position Based Solver for Stiff Rods.

Minimal implementation for a SINGLE LINEAR ROD (no branching).

Strips away:
- Tree structure (not needed for linear chain)
- RodSegment/RodConstraint/Node classes (use plain arrays)
- Complex stiffness calculations (use simple compliance)
- Mass/inertia calculations (use identity)

Reference: "Direct Position-Based Solver for Stiff Rods" (Deul et al., 2017)
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy import linalg

from newton.examples.cosserat2.reference.quaternion_ops import (
    quat_conjugate,
    quat_multiply,
    quat_normalize,
    quat_to_rotation_matrix,
)


def compute_darboux_vector(q0: NDArray, q1: NDArray, avg_length: float) -> NDArray:
    """Compute discrete Darboux vector: omega = (2/L) * Im(conj(q0) * q1)."""
    product = quat_multiply(quat_conjugate(q0), q1)
    return (2.0 / avg_length) * product[:3]


def compute_matrix_G(q: NDArray) -> NDArray:
    """Compute G matrix: q_dot = G * omega (quaternion from angular velocity)."""
    x, y, z, w = q
    return 0.5 * np.array([
        [w, z, -y],
        [-z, w, x],
        [y, -x, w],
        [-x, -y, -z],
    ])


def skew(v: NDArray) -> NDArray:
    """Skew-symmetric cross product matrix."""
    return np.array([
        [0, -v[2], v[1]],
        [v[2], 0, -v[0]],
        [-v[1], v[0], 0],
    ])


def compute_bending_jacobians(q0: NDArray, q1: NDArray, avg_length: float) -> tuple[NDArray, NDArray]:
    """Compute d(omega)/d(omega_0) and d(omega)/d(omega_1) (both 3x3)."""
    x0, y0, z0, w0 = q0
    x1, y1, z1, w1 = q1
    scale = 2.0 / avg_length

    dOmega_dq0 = scale * np.array([
        [-w1, -z1, y1, x1],
        [z1, -w1, -x1, y1],
        [-y1, x1, -w1, z1],
    ])

    dOmega_dq1 = scale * np.array([
        [w0, z0, -y0, -x0],
        [-z0, w0, x0, -y0],
        [y0, -x0, w0, -z0],
    ])

    G0 = compute_matrix_G(q0)
    G1 = compute_matrix_G(q1)

    return dOmega_dq0 @ G0, dOmega_dq1 @ G1


class SimpleDirectRodSolver:
    """Minimal direct solver for a single linear rod.

    Constraints:
    - Stretch: adjacent segment endpoints should meet at shared particle
    - Bending: Darboux vector should match rest Darboux

    For a linear chain, the system is block-tridiagonal.
    """

    def __init__(
        self,
        n_particles: int,
        positions: NDArray,
        quaternions: NDArray,
        rest_lengths: NDArray,
        particle_inv_mass: NDArray,
        edge_inv_mass: NDArray,
        stretch_stiffness: float = 1.0,
        bend_stiffness: float = 1.0,
    ):
        """Initialize solver.

        Args:
            n_particles: Number of particles.
            positions: Particle positions [n_particles, 3].
            quaternions: Edge quaternions [n_particles-1, 4].
            rest_lengths: Rest lengths [n_particles-1].
            particle_inv_mass: Inverse mass per particle [n_particles].
            edge_inv_mass: Inverse mass per edge [n_particles-1].
            stretch_stiffness: Stretch stiffness (0 to 1).
            bend_stiffness: Bend/twist stiffness (0 to 1).
        """
        self.n_particles = n_particles
        self.n_seg = n_particles - 1
        self.n_con = n_particles - 2

        # Copy input arrays
        self.positions = positions.copy().astype(np.float64)
        self.quaternions = quaternions.copy().astype(np.float64)
        self.rest_lengths = rest_lengths.copy().astype(np.float64)
        self.particle_inv_mass = particle_inv_mass.copy().astype(np.float64)
        self.edge_inv_mass = edge_inv_mass.copy().astype(np.float64)

        # Segment state
        self.seg_pos = np.zeros((self.n_seg, 3), dtype=np.float64)
        self.seg_quat = np.zeros((self.n_seg, 4), dtype=np.float64)
        self.seg_is_static = np.zeros(self.n_seg, dtype=bool)

        # Segment masses (sum of endpoint particle masses)
        self.seg_mass = np.zeros(self.n_seg, dtype=np.float64)

        for i in range(self.n_seg):
            self.seg_pos[i] = 0.5 * (self.positions[i] + self.positions[i + 1])
            self.seg_quat[i] = self.quaternions[i]
            self.seg_is_static[i] = self.edge_inv_mass[i] <= 0

            # Compute segment mass from particle masses
            m0 = 1.0 / self.particle_inv_mass[i] if self.particle_inv_mass[i] > 0 else 0.0
            m1 = 1.0 / self.particle_inv_mass[i + 1] if self.particle_inv_mass[i + 1] > 0 else 0.0
            self.seg_mass[i] = m0 + m1 if not self.seg_is_static[i] else 0.0

        # Compliance: larger = softer constraint
        # These values are chosen to match the original solver's behavior:
        # - Stretch compliance ~5e-6 (very stiff, keeps segment lengths exact)
        # - Bend compliance ~1e6 for soft materials (allows natural bending)
        #
        # stiffness=1.0: matches original's soft material behavior
        # stiffness=0.0: even softer (more bending)
        self.stretch_compliance_base = 1e-10 / max(stretch_stiffness, 1e-6)
        # Bend compliance inversely proportional to stiffness
        # With stiffness=1.0, compliance_base ~ 30 -> alpha ~ 1.7e6 (similar to original)
        self.bend_compliance_base = 30.0 / max(bend_stiffness, 1e-6)

        # Rest Darboux vectors (computed from initial quaternions)
        self.rest_darboux = np.zeros((max(1, self.n_con), 3), dtype=np.float64)
        for i in range(self.n_con):
            avg_len = 0.5 * (self.rest_lengths[i] + self.rest_lengths[i + 1])
            self.rest_darboux[i] = compute_darboux_vector(
                self.seg_quat[i], self.seg_quat[i + 1], avg_len
            )

        # Half-lengths for connector computation (from rest lengths)
        self.half_lengths = 0.5 * self.rest_lengths

        # Working arrays
        self.lambda_sum = np.zeros((max(1, self.n_con), 6), dtype=np.float64)
        self.velocities = np.zeros((self.n_particles, 3), dtype=np.float64)
        self.positions_old = np.zeros_like(self.positions)

    def _get_segment_d3(self, i: int) -> NDArray:
        """Get the d3 (local Z) axis of segment i in world frame."""
        R = quat_to_rotation_matrix(self.seg_quat[i])
        return R[:, 2]

    def _get_connector(self, seg_idx: int, end: int) -> NDArray:
        """Get the connector position for segment at specified end.

        Args:
            seg_idx: Segment index.
            end: 0 for start (toward particle seg_idx), 1 for end (toward particle seg_idx+1).

        Returns:
            World-space connector position.
        """
        d3 = self._get_segment_d3(seg_idx)
        half_len = self.half_lengths[seg_idx]
        if end == 0:
            return self.seg_pos[seg_idx] - half_len * d3
        else:
            return self.seg_pos[seg_idx] + half_len * d3

    def sync_segments_from_particles(self) -> None:
        """Update segment positions and orientations from particle positions."""
        for i in range(self.n_seg):
            p0 = self.positions[i]
            p1 = self.positions[i + 1]

            # For anchor segment (segment 0 when particle 0 is fixed):
            # Its START must be at particle 0, so its center = p0 + half_len * d3
            # But we need to compute d3 from the edge direction first
            if i == 0 and self.particle_inv_mass[0] <= 0:
                # Segment 0: start is at fixed particle 0
                # Compute edge direction and update quaternion
                edge_dir = p1 - p0
                edge_len = np.linalg.norm(edge_dir)
                if edge_len > 1e-10:
                    edge_dir /= edge_len
                    self._align_quaternion_to_direction(i, edge_dir)
                # Now set center based on fixed start point
                d3 = self._get_segment_d3(i)
                self.seg_pos[i] = p0 + self.half_lengths[i] * d3
            elif not self.seg_is_static[i]:
                self.seg_pos[i] = 0.5 * (p0 + p1)
                edge_dir = p1 - p0
                edge_len = np.linalg.norm(edge_dir)
                if edge_len > 1e-10:
                    edge_dir /= edge_len
                    self._align_quaternion_to_direction(i, edge_dir)

    def _align_quaternion_to_direction(self, seg_idx: int, target_dir: NDArray) -> None:
        """Align segment's d3 axis to target direction using minimal rotation."""
        d3 = self._get_segment_d3(seg_idx)
        dot = np.clip(np.dot(d3, target_dir), -1.0, 1.0)
        if dot < 0.9999:
            if dot > -0.9999:
                axis = np.cross(d3, target_dir)
                axis /= np.linalg.norm(axis)
                angle = np.arccos(dot)
                s, c = np.sin(angle / 2), np.cos(angle / 2)
                q_rot = np.array([axis[0] * s, axis[1] * s, axis[2] * s, c])
            else:
                perp = np.array([1.0, 0.0, 0.0])
                if abs(np.dot(d3, perp)) > 0.9:
                    perp = np.array([0.0, 1.0, 0.0])
                axis = np.cross(d3, perp)
                axis /= np.linalg.norm(axis)
                q_rot = np.array([axis[0], axis[1], axis[2], 0.0])
            self.seg_quat[seg_idx] = quat_normalize(quat_multiply(q_rot, self.seg_quat[seg_idx]))

    def sync_particles_from_segments(self) -> None:
        """Update particle positions from segment state."""
        self.quaternions[:] = self.seg_quat

        # Particle 0 is fixed (anchor)
        # First, enforce segment 0's center is consistent with fixed particle 0
        # Segment 0's start must be at particle 0, so:
        # seg_pos[0] = particle_0 + half_len * d3[0]
        if self.n_seg > 0 and self.particle_inv_mass[0] <= 0:
            d3_0 = self._get_segment_d3(0)
            self.seg_pos[0] = self.positions[0] + self.half_lengths[0] * d3_0

        # For each constraint i, compute particle i+1 from connector positions
        for i in range(self.n_con):
            # Connector from segment i (end=1) and segment i+1 (end=0) should meet
            c0 = self._get_connector(i, 1)      # End of segment i
            c1 = self._get_connector(i + 1, 0)  # Start of segment i+1
            self.positions[i + 1] = 0.5 * (c0 + c1)

        # Last particle
        if self.n_seg > 0:
            self.positions[self.n_particles - 1] = self._get_connector(self.n_seg - 1, 1)

    def _get_inverse_mass_matrix(self, seg_idx: int) -> NDArray:
        """Get 6x6 inverse mass matrix for a segment.

        Returns diagonal matrix with:
        - [0:3, 0:3]: 1/mass * I for translational DOFs
        - [3:6, 3:6]: 1/inertia * I for rotational DOFs (simplified to identity)
        """
        M_inv = np.zeros((6, 6))
        if not self.seg_is_static[seg_idx] and self.seg_mass[seg_idx] > 0:
            inv_mass = 1.0 / self.seg_mass[seg_idx]
            M_inv[:3, :3] = inv_mass * np.eye(3)
            # For rotational DOFs, use simplified inertia (identity-like scaling)
            # The inertia tensor would typically be computed from mass distribution,
            # but for simplicity we use a similar scale as the translational part
            M_inv[3:, 3:] = inv_mass * np.eye(3)
        return M_inv

    def project_constraints(self, dt: float, iterations: int = 1) -> None:
        """Project constraints using direct solve with mass-weighted corrections."""
        if self.n_con == 0:
            return

        inv_dt_sq = 1.0 / (dt * dt)
        alpha_stretch = self.stretch_compliance_base * inv_dt_sq
        alpha_bend = self.bend_compliance_base * inv_dt_sq

        # Pre-compute inverse mass matrices for all segments
        M_inv_list = [self._get_inverse_mass_matrix(i) for i in range(self.n_seg)]

        for _ in range(iterations):
            n_seg = self.n_seg
            n_con = self.n_con

            rhs = np.zeros((n_con, 6))
            J0_list = []
            J1_list = []

            for i in range(n_con):
                q0 = self.seg_quat[i]
                q1 = self.seg_quat[i + 1]

                # Connectors: segment i end (toward particle i+1) and segment i+1 start
                d3_0 = self._get_segment_d3(i)
                d3_1 = self._get_segment_d3(i + 1)
                half_len_0 = self.half_lengths[i]
                half_len_1 = self.half_lengths[i + 1]

                c0 = self.seg_pos[i] + half_len_0 * d3_0
                c1 = self.seg_pos[i + 1] - half_len_1 * d3_1

                # Stretch violation: c0 - c1 should be zero
                stretch_err = c0 - c1

                # Bending violation
                avg_len = 0.5 * (self.rest_lengths[i] + self.rest_lengths[i + 1])
                omega = compute_darboux_vector(q0, q1, avg_len)
                bend_err = omega - self.rest_darboux[i]

                # RHS
                rhs[i, :3] = -stretch_err - alpha_stretch * self.lambda_sum[i, :3]
                rhs[i, 3:] = -bend_err - alpha_bend * self.lambda_sum[i, 3:]

                # Jacobians for stretch:
                # c0 = seg_pos[i] + half_len_0 * d3_0
                # d(c0)/d(seg_pos[i]) = I
                # d(c0)/d(omega[i]) = half_len_0 * d(d3_0)/d(omega) = -half_len_0 * skew(d3_0)
                #
                # c1 = seg_pos[i+1] - half_len_1 * d3_1
                # d(c1)/d(seg_pos[i+1]) = I
                # d(c1)/d(omega[i+1]) = -half_len_1 * d(d3_1)/d(omega) = half_len_1 * skew(d3_1)

                J0 = np.zeros((6, 6))
                J1 = np.zeros((6, 6))

                # Stretch: C = c0 - c1
                J0[:3, :3] = np.eye(3)
                J0[:3, 3:] = -half_len_0 * skew(d3_0)
                J1[:3, :3] = -np.eye(3)
                J1[:3, 3:] = half_len_1 * skew(d3_1)

                # Bending
                dOmega_dOmega0, dOmega_dOmega1 = compute_bending_jacobians(q0, q1, avg_len)
                J0[3:, 3:] = dOmega_dOmega0
                J1[3:, 3:] = dOmega_dOmega1

                J0_list.append(J0)
                J1_list.append(J1)

            # Build system matrix: A = J * M_inv * J^T + alpha
            A = np.zeros((6 * n_con, 6 * n_con))
            b = rhs.flatten()

            for i in range(n_con):
                J0 = J0_list[i]
                J1 = J1_list[i]
                M_inv_0 = M_inv_list[i]
                M_inv_1 = M_inv_list[i + 1]
                row_start = 6 * i

                seg0_dynamic = not self.seg_is_static[i]
                seg1_dynamic = not self.seg_is_static[i + 1]

                # Diagonal block: J0 * M_inv_0 * J0^T + J1 * M_inv_1 * J1^T + alpha
                diag = np.zeros((6, 6))
                if seg0_dynamic:
                    diag += J0 @ M_inv_0 @ J0.T
                if seg1_dynamic:
                    diag += J1 @ M_inv_1 @ J1.T

                diag[:3, :3] += alpha_stretch * np.eye(3)
                diag[3:, 3:] += alpha_bend * np.eye(3)

                A[row_start:row_start + 6, row_start:row_start + 6] = diag

                # Off-diagonal blocks for coupled constraints
                if i + 1 < n_con:
                    J1_curr = J1_list[i]
                    J0_next = J0_list[i + 1]
                    M_inv_shared = M_inv_list[i + 1]  # Shared segment between constraints i and i+1

                    if not self.seg_is_static[i + 1]:
                        # Off-diagonal: J1_curr * M_inv_shared * J0_next^T
                        off_diag = J1_curr @ M_inv_shared @ J0_next.T
                        A[row_start:row_start + 6, row_start + 6:row_start + 12] = off_diag
                        A[row_start + 6:row_start + 12, row_start:row_start + 6] = off_diag.T

            # Solve
            try:
                lambdas = linalg.solve(A, b, assume_a='sym')
            except linalg.LinAlgError:
                lambdas = np.linalg.lstsq(A, b, rcond=None)[0]

            lambdas = lambdas.reshape((n_con, 6))
            self.lambda_sum += lambdas

            # Apply mass-weighted corrections: delta = M_inv * J^T * lambda
            for i in range(n_seg):
                if self.seg_is_static[i]:
                    continue

                M_inv = M_inv_list[i]
                raw_delta = np.zeros(6)

                if i > 0:
                    raw_delta += J1_list[i - 1].T @ lambdas[i - 1]
                if i < n_con:
                    raw_delta += J0_list[i].T @ lambdas[i]

                # Apply mass weighting
                delta = M_inv @ raw_delta

                self.seg_pos[i] += delta[:3]

                G = compute_matrix_G(self.seg_quat[i])
                delta_q = G @ delta[3:]
                self.seg_quat[i] = quat_normalize(self.seg_quat[i] + delta_q)

    def step(self, dt: float, gravity: NDArray, iterations: int = 1, damping: float = 0.99) -> None:
        """Perform one simulation step."""
        np.copyto(self.positions_old, self.positions)

        # Semi-implicit Euler
        for i in range(self.n_particles):
            if self.particle_inv_mass[i] > 0:
                self.velocities[i] += gravity * dt
                self.positions[i] += self.velocities[i] * dt

        self.sync_segments_from_particles()
        self.lambda_sum[:] = 0
        self.project_constraints(dt, iterations)
        self.sync_particles_from_segments()

        # Update velocities from position change
        for i in range(self.n_particles):
            if self.particle_inv_mass[i] > 0:
                self.velocities[i] = (self.positions[i] - self.positions_old[i]) / dt
                self.velocities[i] *= damping
