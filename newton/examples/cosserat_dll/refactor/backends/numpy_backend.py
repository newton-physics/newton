# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Pure NumPy backend for Cosserat rod simulation.

This backend implements the complete direct solver algorithm in pure Python/NumPy.
It is optimized for clarity and debugging rather than performance.
"""

from typing import TYPE_CHECKING

import numpy as np

from .base import BackendBase

if TYPE_CHECKING:
    from ..model import CosseratRodModel


class NumPyBackend(BackendBase):
    """Pure NumPy backend for Cosserat rod simulation.

    This backend implements the direct solver algorithm entirely in NumPy,
    making it easy to debug and understand. It follows the same structure
    as the C++ reference implementation.

    The solver uses:
    - Banded Cholesky factorization for the JMJT system
    - 6 DOF per constraint (3 stretch-shear + 3 bend-twist)
    - Block tri-diagonal matrix structure
    """

    # Bandwidth for banded system (6x6 blocks -> bandwidth 11)
    BANDWIDTH = 11

    def __init__(self, model: "CosseratRodModel"):
        """Initialize NumPy backend.

        Args:
            model: The rod model to operate on.
        """
        super().__init__(model)

        n_edges = model.n_edges

        # Lagrange multipliers (accumulated constraint forces)
        self.lambdas = np.zeros((n_edges, 6), dtype=np.float32)

        # Compliance values (inverse stiffness scaled by dt^2)
        self.compliance = np.zeros((n_edges, 6), dtype=np.float32)

        # Constraint values/errors
        self.constraint_values = np.zeros((n_edges, 6), dtype=np.float32)

        # Full 6x6 Jacobians per constraint
        self.J_fwd = np.zeros((n_edges, 6, 6), dtype=np.float32)
        self.J_bwd = np.zeros((n_edges, 6, 6), dtype=np.float32)

        # Banded matrix storage
        n_dofs = 6 * n_edges
        self.A_banded = np.zeros((2 * self.BANDWIDTH + 1, max(1, n_dofs)), dtype=np.float32)
        self.rhs = np.zeros(max(1, n_dofs), dtype=np.float32)

    @property
    def name(self) -> str:
        return "NumPy (CPU)"

    def step(self, dt: float):
        """Advance simulation by one timestep."""
        m = self.model

        # 1. Predict positions
        self._predict_positions(dt)

        # 2. Predict rotations
        self._predict_rotations(dt)

        # 3. Prepare constraints
        self._prepare_constraints(dt)

        # 4-7. Update, Jacobians, Assemble, Solve
        self._solve_constraints()

        # 8. Integrate positions
        self._integrate_positions(dt)

        # 9. Integrate rotations
        self._integrate_rotations(dt)

        # 10. Clear forces
        m.clear_forces()

    # =========================================================================
    # Prediction
    # =========================================================================

    def _predict_positions(self, dt: float):
        """Semi-implicit Euler position prediction."""
        m = self.model
        damping = m.config.position_damping
        gravity = m.config.gravity

        for i in range(m.n_particles):
            if m.inv_masses[i] > 0:
                # Apply damping
                m.velocities[i, :3] *= (1.0 - damping)
                # Acceleration
                accel = m.inv_masses[i] * (gravity + m.forces[i, :3])
                m.velocities[i, :3] += dt * accel
                # Predict
                m.predicted_positions[i, :3] = m.positions[i, :3] + dt * m.velocities[i, :3]
            else:
                m.predicted_positions[i, :3] = m.positions[i, :3]

    def _predict_rotations(self, dt: float):
        """Quaternion rotation prediction."""
        m = self.model
        damping = m.config.rotation_damping

        for i in range(m.n_particles):
            if m.quat_inv_masses[i] > 0:
                # Apply damping
                m.angular_velocities[i, :3] *= (1.0 - damping)
                # Angular acceleration
                m.angular_velocities[i, :3] += dt * m.torques[i, :3]

                # Quaternion integration: q' = q + 0.5 * dt * omega_quat * q
                q = m.orientations[i].copy()
                omega = m.angular_velocities[i, :3]

                # omega_quat * q
                wx, wy, wz = omega
                qx, qy, qz, qw = q

                dq = np.array([
                    wx * qw + wy * qz - wz * qy,
                    -wx * qz + wy * qw + wz * qx,
                    wx * qy - wy * qx + wz * qw,
                    -wx * qx - wy * qy - wz * qz,
                ], dtype=np.float32)

                q_new = q + 0.5 * dt * dq
                norm = np.linalg.norm(q_new)
                if norm > 1e-10:
                    q_new /= norm

                m.predicted_orientations[i] = q_new
            else:
                m.predicted_orientations[i] = m.orientations[i].copy()

    # =========================================================================
    # Integration
    # =========================================================================

    def _integrate_positions(self, dt: float):
        """Derive velocities and update positions."""
        m = self.model
        inv_dt = 1.0 / dt if dt > 1e-10 else 0.0

        for i in range(m.n_particles):
            if m.inv_masses[i] > 0:
                m.velocities[i, :3] = (m.predicted_positions[i, :3] - m.positions[i, :3]) * inv_dt
                m.positions[i, :3] = m.predicted_positions[i, :3]
            else:
                m.positions[i, :3] = m.predicted_positions[i, :3]
                m.velocities[i, :3] = 0.0

    def _integrate_rotations(self, dt: float):
        """Derive angular velocities and update orientations."""
        m = self.model
        inv_dt = 1.0 / dt if dt > 1e-10 else 0.0

        for i in range(m.n_particles):
            if m.quat_inv_masses[i] > 0:
                q_old = m.orientations[i].copy()
                m.prev_orientations[i] = q_old
                q_new = m.predicted_orientations[i].copy()
                m.orientations[i] = q_new

                # Derive angular velocity
                q_old_inv = self._quat_conjugate(q_old)
                q_rel = self._quat_multiply(q_new, q_old_inv)

                sign = 1.0 if q_rel[3] >= 0 else -1.0
                m.angular_velocities[i, :3] = sign * 2.0 * inv_dt * q_rel[:3]
            else:
                m.prev_orientations[i] = m.orientations[i].copy()
                m.orientations[i] = m.predicted_orientations[i].copy()
                m.angular_velocities[i, :3] = 0.0

    # =========================================================================
    # Constraint solving
    # =========================================================================

    def _prepare_constraints(self, dt: float):
        """Prepare constraint system (reset lambdas, compute compliance)."""
        m = self.model
        n_edges = m.n_edges

        self.lambdas.fill(0.0)

        dt2 = dt * dt
        inv_dt2 = 1.0 / dt2

        # Stretch regularization (nearly inextensible)
        stretch_reg = 1e-12

        for i in range(n_edges):
            L = m.rest_lengths[i]

            # Stretch compliance
            stretch_compliance = stretch_reg * inv_dt2
            self.compliance[i, 0] = stretch_compliance
            self.compliance[i, 1] = stretch_compliance
            self.compliance[i, 2] = stretch_compliance

            # Bend/twist stiffness
            K_bend1 = m.bend_stiffness[i, 0] * m.material.young_modulus
            K_bend2 = m.bend_stiffness[i, 1] * m.material.young_modulus
            K_twist = m.bend_stiffness[i, 2] * m.material.torsion_modulus

            eps = 1e-10
            self.compliance[i, 3] = inv_dt2 / (K_bend1 + eps) / L
            self.compliance[i, 4] = inv_dt2 / (K_bend2 + eps) / L
            self.compliance[i, 5] = inv_dt2 / (K_twist + eps) / L

    def _solve_constraints(self):
        """Complete constraint solve: update, Jacobians, assemble, solve."""
        m = self.model
        n_edges = m.n_edges

        if n_edges == 0:
            return

        # 1. Update constraint values
        self._update_constraints()

        # 2. Compute Jacobians
        self._compute_jacobians()

        # 3. Assemble JMJT
        self._assemble_system()

        # 4. Solve and apply corrections
        self._solve_and_apply()

    def _update_constraints(self):
        """Compute constraint violations."""
        m = self.model

        for i in range(m.n_edges):
            p0 = m.predicted_positions[i, :3]
            p1 = m.predicted_positions[i + 1, :3]
            q0 = m.predicted_orientations[i]
            q1 = m.predicted_orientations[i + 1]
            L = m.rest_lengths[i]

            # Stretch-Shear
            d3_0 = self._quat_rotate_vector(q0, np.array([0, 0, 1], dtype=np.float32))
            d3_1 = self._quat_rotate_vector(q1, np.array([0, 0, 1], dtype=np.float32))

            connector0 = p0 + 0.5 * L * d3_0
            connector1 = p1 - 0.5 * L * d3_1
            stretch_violation = connector0 - connector1

            self.constraint_values[i, :3] = stretch_violation

            # Bend-Twist (Darboux vector)
            q0_inv = self._quat_conjugate(q0)
            q_rel = self._quat_multiply(q0_inv, q1)
            omega = q_rel[:3]
            omega_rest = m.rest_darboux[i]

            self.constraint_values[i, 3:6] = omega - omega_rest

    def _compute_jacobians(self):
        """Compute 6x6 Jacobians for each constraint."""
        m = self.model

        for i in range(m.n_edges):
            q0 = m.predicted_orientations[i]
            q1 = m.predicted_orientations[i + 1]
            L = max(m.rest_lengths[i], 1e-8)

            d3_0 = self._quat_rotate_vector(q0, np.array([0, 0, 1], dtype=np.float32))
            d3_1 = self._quat_rotate_vector(q1, np.array([0, 0, 1], dtype=np.float32))

            r0 = 0.5 * L * d3_0
            r1 = -0.5 * L * d3_1

            G0 = self._compute_matrix_G(q0)
            G1 = self._compute_matrix_G(q1)
            jOmega0, jOmega1 = self._compute_jOmega(q0, q1)

            jOmegaG0 = jOmega0 @ G0
            jOmegaG1 = jOmega1 @ G1

            # Forward Jacobian
            self.J_fwd[i, 0:3, 0:3] = np.eye(3, dtype=np.float32)
            self.J_fwd[i, 0:3, 3:6] = -self._skew_symmetric(r0)
            self.J_fwd[i, 3:6, 0:3] = 0.0
            self.J_fwd[i, 3:6, 3:6] = jOmegaG0

            # Backward Jacobian
            self.J_bwd[i, 0:3, 0:3] = -np.eye(3, dtype=np.float32)
            self.J_bwd[i, 0:3, 3:6] = self._skew_symmetric(r1)
            self.J_bwd[i, 3:6, 0:3] = 0.0
            self.J_bwd[i, 3:6, 3:6] = jOmegaG1

    def _assemble_system(self):
        """Assemble banded JMJT matrix."""
        m = self.model
        n_edges = m.n_edges

        self.A_banded.fill(0.0)

        for i in range(n_edges):
            J_fwd = self.J_fwd[i]
            J_bwd = self.J_bwd[i]

            # Diagonal block
            JMJT_block = J_fwd @ J_fwd.T + J_bwd @ J_bwd.T
            for k in range(6):
                JMJT_block[k, k] += self.compliance[i, k]

            # Store diagonal block
            block_start = 6 * i
            for row in range(6):
                for col in range(6):
                    global_row = block_start + row
                    global_col = block_start + col
                    band_row = self.BANDWIDTH + global_row - global_col
                    self.A_banded[band_row, global_col] = JMJT_block[row, col]

            # Off-diagonal coupling
            if i < n_edges - 1:
                J_fwd_next = self.J_fwd[i + 1]
                coupling = J_bwd @ J_fwd_next.T

                for row in range(6):
                    for col in range(6):
                        global_row = block_start + row
                        global_col = block_start + 6 + col
                        band_row = self.BANDWIDTH + global_row - global_col
                        self.A_banded[band_row, global_col] = coupling[row, col]

                        global_row = block_start + 6 + row
                        global_col = block_start + col
                        band_row = self.BANDWIDTH + global_row - global_col
                        self.A_banded[band_row, global_col] = coupling[col, row]

    def _solve_and_apply(self):
        """Solve the system and apply corrections."""
        m = self.model
        n_edges = m.n_edges

        # Build RHS
        for i in range(n_edges):
            for k in range(6):
                self.rhs[6 * i + k] = -self.constraint_values[i, k]

        # Solve using banded Cholesky
        delta_lambda = self._solve_banded_cholesky()

        # Apply corrections
        for i in range(n_edges):
            dl = delta_lambda[6 * i: 6 * i + 6]

            J_fwd = self.J_fwd[i]
            J_bwd = self.J_bwd[i]

            # Segment i correction
            if m.inv_masses[i] > 0.0:
                correction0 = J_fwd.T @ dl
                dp0 = m.inv_masses[i] * correction0[:3]
                m.predicted_positions[i, :3] += dp0

            if m.quat_inv_masses[i] > 0.0:
                correction0 = J_fwd.T @ dl
                dtheta0 = correction0[3:6]
                self._apply_quaternion_correction(m.predicted_orientations, i, dtheta0)

            # Segment i+1 correction
            if m.inv_masses[i + 1] > 0.0:
                correction1 = J_bwd.T @ dl
                dp1 = m.inv_masses[i + 1] * correction1[:3]
                m.predicted_positions[i + 1, :3] += dp1

            if m.quat_inv_masses[i + 1] > 0.0:
                correction1 = J_bwd.T @ dl
                dtheta1 = correction1[3:6]
                self._apply_quaternion_correction(m.predicted_orientations, i + 1, dtheta1)

    def _solve_banded_cholesky(self) -> np.ndarray:
        """Solve banded system with in-place Cholesky."""
        kd = self.BANDWIDTH
        n = self.rhs.shape[0]

        if n == 0:
            return self.rhs.copy()

        # Copy for in-place factorization
        ab = self.A_banded[:kd + 1, :].copy()
        b = self.rhs.copy()

        # Cholesky factorization
        for j in range(n):
            sum_sq = 0.0
            kmax = min(j, kd)
            for k in range(1, kmax + 1):
                u = ab[kd - k, j]
                sum_sq += u * u

            ajj = ab[kd, j] - sum_sq
            if ajj <= 1e-6:
                ajj = 1e-6
            ujj = np.sqrt(ajj)
            ab[kd, j] = ujj

            imax = min(n - j - 1, kd)
            for i in range(1, imax + 1):
                dot = 0.0
                k2max = min(j, kd - i)
                for k in range(1, k2max + 1):
                    dot += ab[kd - k, j] * ab[kd - i - k, j + i]
                aji = ab[kd - i, j + i] - dot
                ab[kd - i, j + i] = aji / ujj

        # Forward solve
        for i in range(n):
            sum_val = 0.0
            k0 = max(0, i - kd)
            for k in range(k0, i):
                sum_val += ab[kd + k - i, i] * b[k]
            b[i] = (b[i] - sum_val) / ab[kd, i]

        # Backward solve
        for i in range(n - 1, -1, -1):
            sum_val = 0.0
            k1 = min(i + kd, n - 1)
            for k in range(i + 1, k1 + 1):
                sum_val += ab[kd + i - k, k] * b[k]
            b[i] = (b[i] - sum_val) / ab[kd, i]

        return b

    # =========================================================================
    # Quaternion utilities
    # =========================================================================

    def _quat_rotate_vector(self, q: np.ndarray, v: np.ndarray) -> np.ndarray:
        """Rotate vector v by quaternion q."""
        x, y, z, w = q
        vx, vy, vz = v

        tx = 2.0 * (y * vz - z * vy)
        ty = 2.0 * (z * vx - x * vz)
        tz = 2.0 * (x * vy - y * vx)

        return np.array([
            vx + w * tx + y * tz - z * ty,
            vy + w * ty + z * tx - x * tz,
            vz + w * tz + x * ty - y * tx,
        ], dtype=np.float32)

    def _quat_conjugate(self, q: np.ndarray) -> np.ndarray:
        """Quaternion conjugate."""
        return np.array([-q[0], -q[1], -q[2], q[3]], dtype=np.float32)

    def _quat_multiply(self, q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
        """Multiply two quaternions."""
        x1, y1, z1, w1 = q1
        x2, y2, z2, w2 = q2
        return np.array([
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        ], dtype=np.float32)

    def _compute_matrix_G(self, q: np.ndarray) -> np.ndarray:
        """G matrix: converts angular velocity to quaternion derivative."""
        x, y, z, w = q
        return np.array([
            [0.5 * w, 0.5 * z, -0.5 * y],
            [-0.5 * z, 0.5 * w, 0.5 * x],
            [0.5 * y, -0.5 * x, 0.5 * w],
            [-0.5 * x, -0.5 * y, -0.5 * z]
        ], dtype=np.float32)

    def _compute_jOmega(self, q0: np.ndarray, q1: np.ndarray):
        """Jacobians of Darboux vector w.r.t. quaternion components."""
        x0, y0, z0, w0 = q0
        x1, y1, z1, w1 = q1

        jOmega0 = np.array([
            [-w1, -z1, y1, x1],
            [z1, -w1, -x1, y1],
            [-y1, x1, -w1, z1]
        ], dtype=np.float32)

        jOmega1 = np.array([
            [w0, z0, -y0, -x0],
            [-z0, w0, x0, -y0],
            [y0, -x0, w0, -z0]
        ], dtype=np.float32)

        return jOmega0, jOmega1

    def _skew_symmetric(self, v: np.ndarray) -> np.ndarray:
        """Skew-symmetric matrix from vector."""
        return np.array([
            [0, -v[2], v[1]],
            [v[2], 0, -v[0]],
            [-v[1], v[0], 0],
        ], dtype=np.float32)

    def _apply_quaternion_correction(
        self, orientations: np.ndarray, idx: int, dtheta: np.ndarray
    ):
        """Apply rotation correction using G matrix."""
        if np.linalg.norm(dtheta) < 1e-10:
            return

        q = orientations[idx]
        G = self._compute_matrix_G(q)
        dq = G @ dtheta
        q_new = q + dq

        norm = np.linalg.norm(q_new)
        if norm > 1e-10:
            q_new /= norm

        orientations[idx] = q_new
