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

"""NumPy-based rod state implementation with optional Warp acceleration.

This module provides a hybrid implementation that can use NumPy or Warp
for different simulation steps, useful for debugging and comparison.
"""

from __future__ import annotations

import numpy as np
import warp as wp

from newton.examples.cosserat_codex.constants import (
    DIRECT_SOLVE_BACKENDS,
    DIRECT_SOLVE_CPU_NUMPY,
    DIRECT_SOLVE_WARP_BLOCK_THOMAS,
)

from .dll_rod import DefKitDirectRodState


class NumpyDirectRodState(DefKitDirectRodState):
    """Hybrid rod state with NumPy fallback and Warp acceleration options.
    
    This class extends DefKitDirectRodState to allow switching between:
    - Native C++ DLL implementation
    - Pure NumPy implementation
    - Warp GPU-accelerated implementation
    
    Useful for debugging, comparison, and platforms without the native DLL.
    """

    def __init__(self, *args, **kwargs):
        """Initialize the hybrid rod state."""
        super().__init__(*args, **kwargs)
        
        # Track which steps have NumPy implementations
        self.numpy_available = {
            "predict_positions": True,
            "integrate_positions": True,
            "predict_rotations": True,
            "integrate_rotations": True,
            "prepare_constraints": True,
            "update_constraints_banded": True,
            "compute_jacobians_banded": True,
            "assemble_jmjt_banded": True,
            "project_jmjt_banded": True,
            "project_direct": True,
        }
        
        # Enable NumPy by default for all available steps
        self.numpy_enabled = {k: True for k in self.numpy_available}
        
        # Check Warp device availability
        self.warp_device = wp.get_device()
        warp_default = self.warp_device.is_cuda
        
        # Track which steps have Warp implementations
        self.warp_available = {step_name: False for step_name in self.numpy_available}
        self.warp_enabled = {step_name: False for step_name in self.numpy_available}
        
        # Enable Warp for supported steps on CUDA devices
        for step_name in (
            "predict_positions",
            "integrate_positions",
            "predict_rotations",
            "integrate_rotations",
            "prepare_constraints",
            "project_direct",
        ):
            self.warp_available[step_name] = True
            self.warp_enabled[step_name] = warp_default
        
        # Solver backend selection
        self.direct_solve_backend = (
            DIRECT_SOLVE_WARP_BLOCK_THOMAS if warp_default else DIRECT_SOLVE_CPU_NUMPY
        )
        
        # Constraint state arrays
        self.lambdas = np.zeros((self.num_edges, 6), dtype=np.float32)
        self.compliance = np.zeros((self.num_edges, 6), dtype=np.float32)
        self.constraint_values = np.zeros((self.num_edges, 6), dtype=np.float32)
        self.lambda_sum = np.zeros((self.num_edges, 6), dtype=np.float32)
        self.current_rest_lengths = self.rest_lengths.copy()
        self.current_rest_darboux = np.zeros((self.num_edges, 3), dtype=np.float32)
        
        # Convergence diagnostics
        self.last_constraint_max = 0.0
        self.last_delta_lambda_max = 0.0
        self.last_correction_max = 0.0

        # Initialize cross-section properties for physical calculations
        self._update_cross_section_properties()

    def _update_cross_section_properties(self) -> None:
        """Compute cross-section geometric properties from rod radius."""
        radius = np.float32(self.rod_radius)
        self.cross_section_area = np.float32(np.pi) * radius * radius
        self.second_moment_area = np.float32(np.pi) * radius**4 / np.float32(4.0)
        self.polar_moment = np.float32(np.pi) * radius**4 / np.float32(2.0)

    def _requires_native_constraint_pipeline(self) -> bool:
        """Check if native DLL is required for constraint projection."""
        if not self.use_banded:
            return not self.numpy_enabled.get("project_direct", False)
        return not (
            self.numpy_enabled.get("update_constraints_banded", False)
            and self.numpy_enabled.get("compute_jacobians_banded", False)
            and self.numpy_enabled.get("assemble_jmjt_banded", False)
            and self.numpy_enabled.get("project_jmjt_banded", False)
        )

    def set_numpy_override(self, step_name: str, enabled: bool) -> bool:
        """Enable or disable NumPy implementation for a specific step.
        
        Args:
            step_name: Name of the simulation step.
            enabled: Whether to enable NumPy for this step.
        
        Returns:
            Whether the setting was applied successfully.
        
        Raises:
            ValueError: If step_name is not recognized.
        """
        if step_name not in self.numpy_available:
            raise ValueError(f"Unknown step: {step_name}")
        if not self.numpy_available[step_name]:
            self.numpy_enabled[step_name] = False
            return False
        self.numpy_enabled[step_name] = enabled
        return True

    def set_warp_override(self, step_name: str, enabled: bool) -> bool:
        """Enable or disable Warp implementation for a specific step.
        
        Args:
            step_name: Name of the simulation step.
            enabled: Whether to enable Warp for this step.
        
        Returns:
            Whether the setting was applied successfully.
        
        Raises:
            ValueError: If step_name is not recognized.
        """
        if step_name not in self.warp_available:
            raise ValueError(f"Unknown step: {step_name}")
        if not self.warp_available[step_name]:
            self.warp_enabled[step_name] = False
            return False
        self.warp_enabled[step_name] = enabled
        return True

    def set_direct_solve_backend(self, backend: str) -> None:
        """Set the direct solver backend.
        
        Args:
            backend: One of the DIRECT_SOLVE_BACKENDS constants.
        
        Raises:
            ValueError: If backend is not recognized.
        """
        if backend not in DIRECT_SOLVE_BACKENDS:
            raise ValueError(f"Unknown direct solve backend: {backend}")
        self.direct_solve_backend = backend

    def _use_warp_step(self, step_name: str) -> bool:
        """Check if Warp should be used for a given step."""
        return self.warp_available.get(step_name, False) and self.warp_enabled.get(step_name, False)

    # NumPy implementations of quaternion operations
    
    def _numpy_quat_mul(self, q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
        """Multiply two quaternion arrays (N x 4)."""
        x1, y1, z1, w1 = q1[:, 0], q1[:, 1], q1[:, 2], q1[:, 3]
        x2, y2, z2, w2 = q2[:, 0], q2[:, 1], q2[:, 2], q2[:, 3]
        result = np.zeros_like(q1)
        result[:, 0] = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
        result[:, 1] = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
        result[:, 2] = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
        result[:, 3] = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
        return result

    def _numpy_quat_normalize(self, q: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
        """Normalize quaternion array."""
        norm = np.linalg.norm(q, axis=1, keepdims=True)
        norm = np.where(norm < 1e-8, 1.0, norm)
        result = q / norm
        if mask is not None:
            result[~mask] = q[~mask]
        return result

    @staticmethod
    def _numpy_quat_mul_single(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
        """Multiply two single quaternions."""
        x1, y1, z1, w1 = q1
        x2, y2, z2, w2 = q2
        return np.array(
            [
                w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
                w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            ],
            dtype=np.float32,
        )

    @staticmethod
    def _numpy_quat_conjugate(q: np.ndarray) -> np.ndarray:
        """Return conjugate of a single quaternion."""
        return np.array([-q[0], -q[1], -q[2], q[3]], dtype=np.float32)

    @staticmethod
    def _numpy_quat_rotate_vector(q: np.ndarray, v: np.ndarray) -> np.ndarray:
        """Rotate a vector by a quaternion."""
        x, y, z, w = q
        vx, vy, vz = v

        tx = np.float32(2.0) * (y * vz - z * vy)
        ty = np.float32(2.0) * (z * vx - x * vz)
        tz = np.float32(2.0) * (x * vy - y * vx)

        return np.array(
            [
                vx + w * tx + y * tz - z * ty,
                vy + w * ty + z * tx - x * tz,
                vz + w * tz + x * ty - y * tx,
            ],
            dtype=np.float32,
        )

    @staticmethod
    def _numpy_skew_symmetric(v: np.ndarray) -> np.ndarray:
        """Return 3x3 skew-symmetric matrix from vector."""
        return np.array(
            [
                [0.0, -v[2], v[1]],
                [v[2], 0.0, -v[0]],
                [-v[1], v[0], 0.0],
            ],
            dtype=np.float32,
        )

    @staticmethod
    def _numpy_compute_bending_torsion_jacobians(
        q0: np.ndarray, q1: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute Jacobians for bending/torsion constraints."""
        x0, y0, z0, w0 = q0
        x1, y1, z1, w1 = q1
        jomega0 = np.array(
            [
                [-w1, -z1, y1, x1],
                [z1, -w1, -x1, y1],
                [-y1, x1, -w1, z1],
            ],
            dtype=np.float32,
        )
        jomega1 = np.array(
            [
                [w0, z0, -y0, -x0],
                [-z0, w0, x0, -y0],
                [y0, -x0, w0, -z0],
            ],
            dtype=np.float32,
        )
        return jomega0, jomega1

    @staticmethod
    def _numpy_compute_matrix_g(q: np.ndarray) -> np.ndarray:
        """Compute G matrix for quaternion angular velocity mapping."""
        x, y, z, w = q
        return np.array(
            [
                [0.5 * w, 0.5 * z, -0.5 * y],
                [-0.5 * z, 0.5 * w, 0.5 * x],
                [0.5 * y, -0.5 * x, 0.5 * w],
                [-0.5 * x, -0.5 * y, -0.5 * z],
            ],
            dtype=np.float32,
        )

    def _apply_quaternion_correction_g(
        self, orientations: np.ndarray, idx: int, dtheta: np.ndarray
    ) -> None:
        """Apply quaternion correction using G matrix."""
        if np.linalg.norm(dtheta) < 1.0e-10:
            return
        q = orientations[idx]
        g = self._numpy_compute_matrix_g(q)
        corr_q = (g @ dtheta).astype(np.float32)
        q_new = q + corr_q
        q_new /= np.linalg.norm(q_new)
        orientations[idx] = q_new

    # NumPy implementations of simulation steps
    
    def _numpy_predict_positions(self, dt: float, linear_damping: float) -> None:
        """NumPy implementation of position prediction."""
        dt_f32 = np.float32(dt)
        damp = np.float32(1.0 - linear_damping)

        inv_mass = self.inv_masses[:, None]
        positions = self.positions[:, 0:3]
        velocities = self.velocities[:, 0:3]
        forces = self.forces[:, 0:3]
        gravity = self.gravity[0, 0:3]

        velocities[:] = velocities + (forces * inv_mass + gravity) * dt_f32
        velocities[:] = velocities * damp

        predicted = positions + velocities * dt_f32

        static_mask = self.inv_masses == 0.0
        velocities[static_mask] = 0.0
        predicted[static_mask] = positions[static_mask]

        self.predicted_positions[:, 0:3] = predicted
        self.predicted_positions[:, 3] = 0.0
        self.velocities[:, 3] = 0.0

    def _numpy_integrate_positions(self, dt: float) -> None:
        """NumPy implementation of position integration."""
        dt_inv = np.float32(1.0 / dt)
        positions = self.positions[:, 0:3]
        predicted = self.predicted_positions[:, 0:3]
        velocities = self.velocities[:, 0:3]

        dynamic_mask = self.inv_masses != 0.0
        velocities[dynamic_mask] = (predicted[dynamic_mask] - positions[dynamic_mask]) * dt_inv
        positions[dynamic_mask] = predicted[dynamic_mask]

        self.positions[:, 3] = 0.0
        self.velocities[:, 3] = 0.0

    def _numpy_predict_rotations(self, dt: float, angular_damping: float) -> None:
        """NumPy implementation of rotation prediction."""
        dt_f32 = np.float32(dt)
        half_dt = np.float32(0.5 * dt_f32)
        damp = np.float32(1.0 - angular_damping)

        inv_mass = self.quat_inv_masses[:, None]
        ang_vel = self.angular_velocities[:, 0:3]
        torques = self.torques[:, 0:3]
        orientations = self.orientations

        dynamic_mask = self.quat_inv_masses != 0.0
        ang_vel[dynamic_mask] = (ang_vel[dynamic_mask] + torques[dynamic_mask] * inv_mass[dynamic_mask] * dt_f32) * damp
        ang_vel[~dynamic_mask] = 0.0

        ang_vel_q = np.zeros_like(orientations)
        ang_vel_q[:, 0:3] = ang_vel

        qdot = self._numpy_quat_mul(ang_vel_q, orientations)
        predicted = orientations + qdot * half_dt
        predicted = self._numpy_quat_normalize(predicted, dynamic_mask)
        predicted[~dynamic_mask] = orientations[~dynamic_mask]

        self.predicted_orientations = predicted.astype(np.float32)
        self.angular_velocities[:, 3] = 0.0

    def _numpy_integrate_rotations(self, dt: float) -> None:
        """NumPy implementation of rotation integration."""
        dt_inv2 = np.float32(2.0 / dt)
        dynamic_mask = self.quat_inv_masses != 0.0

        predicted = self.predicted_orientations
        orientations = self.orientations

        conj = orientations.copy()
        conj[:, 0:3] *= -1.0

        rel = self._numpy_quat_mul(predicted, conj)
        self.angular_velocities[dynamic_mask, 0:3] = rel[dynamic_mask, 0:3] * dt_inv2

        self.prev_orientations[dynamic_mask] = orientations[dynamic_mask]
        self.orientations[dynamic_mask] = predicted[dynamic_mask]
        self.angular_velocities[:, 3] = 0.0

    def _numpy_prepare_constraints(self, dt: float) -> None:
        """NumPy implementation of constraint preparation."""
        self.lambdas.fill(0.0)
        self.lambda_sum.fill(0.0)
        self.current_rest_lengths[:] = self.rest_lengths
        self.current_rest_darboux[:] = self.rest_darboux[:, 0:3]

        dt2 = np.float32(dt * dt)
        eps = np.float32(1.0e-10)

        E = np.float32(self.young_modulus)
        G = np.float32(self.torsion_modulus)

        L = self.current_rest_lengths.astype(np.float32)

        k_bend1_eff = E * self.bend_stiffness[:, 0] * L
        k_bend2_eff = E * self.bend_stiffness[:, 1] * L
        k_twist_eff = G * self.bend_stiffness[:, 2] * L

        stretch_compliance_val = np.float32(1.0e-10)

        self.compliance[:, 0] = stretch_compliance_val
        self.compliance[:, 1] = stretch_compliance_val
        self.compliance[:, 2] = stretch_compliance_val
        self.compliance[:, 3] = np.float32(1.0) / (k_bend1_eff * dt2 + eps)
        self.compliance[:, 4] = np.float32(1.0) / (k_bend2_eff * dt2 + eps)
        self.compliance[:, 5] = np.float32(1.0) / (k_twist_eff * dt2 + eps)

    def _numpy_update_constraints_banded(self) -> None:
        """NumPy implementation of constraint value update."""
        positions = self.predicted_positions[:, 0:3]
        orientations = self.predicted_orientations
        rest_lengths = self.current_rest_lengths
        rest_darboux = self.current_rest_darboux

        max_constraint = 0.0
        for i in range(self.num_edges):
            p0 = positions[i]
            p1 = positions[i + 1]
            q0 = orientations[i]
            q1 = orientations[i + 1]

            L = rest_lengths[i]
            half_L = np.float32(0.5) * L

            # Local offsets to midpoint (assuming rod aligns with local Z)
            r0_local = np.array([0.0, 0.0, half_L], dtype=np.float32)
            r1_local = np.array([0.0, 0.0, -half_L], dtype=np.float32)

            r0_world = self._numpy_quat_rotate_vector(q0, r0_local)
            r1_world = self._numpy_quat_rotate_vector(q1, r1_local)

            c0 = p0 + r0_world
            c1 = p1 + r1_world

            # Stretch violation
            stretch_error = c0 - c1
            self.constraint_values[i, 0:3] = stretch_error

            # Bending/torsion violation
            q_rel = self._numpy_quat_mul_single(self._numpy_quat_conjugate(q0), q1)
            omega = q_rel[:3]
            darboux_error = omega - rest_darboux[i]
            self.constraint_values[i, 3:6] = darboux_error
            max_constraint = max(max_constraint, float(np.linalg.norm(self.constraint_values[i])))

        self.last_constraint_max = max_constraint

    def _numpy_compute_jacobians_direct(self) -> None:
        """NumPy implementation of Jacobian computation."""
        n_edges = self.num_edges
        if n_edges == 0:
            return

        if not hasattr(self, "jacobian_pos") or self.jacobian_pos.shape[0] != n_edges:
            self.jacobian_pos = np.zeros((n_edges, 6, 6), dtype=np.float32)
            self.jacobian_rot = np.zeros((n_edges, 6, 6), dtype=np.float32)

        orientations = self.predicted_orientations
        rest_lengths = self.current_rest_lengths

        for i in range(n_edges):
            q0 = orientations[i]
            q1 = orientations[i + 1]
            L = rest_lengths[i]
            half_L = np.float32(0.5) * L

            r0_local = np.array([0.0, 0.0, half_L], dtype=np.float32)
            r1_local = np.array([0.0, 0.0, -half_L], dtype=np.float32)

            r0_world = self._numpy_quat_rotate_vector(q0, r0_local)
            r1_world = self._numpy_quat_rotate_vector(q1, r1_local)

            I3 = np.eye(3, dtype=np.float32)

            self.jacobian_pos[i, 0:3, 0:3] = I3
            self.jacobian_pos[i, 0:3, 3:6] = -I3

            r0_skew = self._numpy_skew_symmetric(r0_world)
            r1_skew = self._numpy_skew_symmetric(r1_world)

            self.jacobian_rot[i, 0:3, 0:3] = -r0_skew
            self.jacobian_rot[i, 0:3, 3:6] = r1_skew

            jomega0, jomega1 = self._numpy_compute_bending_torsion_jacobians(q0, q1)
            g0 = self._numpy_compute_matrix_g(q0)
            g1 = self._numpy_compute_matrix_g(q1)
            self.jacobian_rot[i, 3:6, 0:3] = (jomega0 @ g0).astype(np.float32)
            self.jacobian_rot[i, 3:6, 3:6] = (jomega1 @ g1).astype(np.float32)

            self.jacobian_pos[i, 3:6, 0:3] = 0.0
            self.jacobian_pos[i, 3:6, 3:6] = 0.0

    def _numpy_assemble_jmjt_banded(self) -> None:
        """NumPy implementation of banded JMJT assembly."""
        n_edges = self.num_edges
        if n_edges == 0:
            return
        n_dofs = 6 * n_edges
        bandwidth = 6
        self.bandwidth = bandwidth

        if not hasattr(self, "A_banded") or self.A_banded.shape[1] != n_dofs:
            self.A_banded = np.zeros((2 * bandwidth + 1, n_dofs), dtype=np.float32)
            self.rhs = np.zeros(n_dofs, dtype=np.float32)

        self.A_banded.fill(0.0)

        # Use unit mass/inertia for LHS assembly (matching C++ reference)
        inv_masses = np.ones(self.num_points, dtype=np.float32)
        inv_I = np.ones(self.num_points, dtype=np.float32)

        for i in range(n_edges):
            J_pos = self.jacobian_pos[i]
            J_rot = self.jacobian_rot[i]
            J_p0 = J_pos[:, 0:3]
            J_p1 = J_pos[:, 3:6]
            J_t0 = J_rot[:, 0:3]
            J_t1 = J_rot[:, 3:6]

            inv_m0 = inv_masses[i]
            inv_m1 = inv_masses[i + 1]
            inv_I0 = inv_I[i]
            inv_I1 = inv_I[i + 1]

            JMJT = (
                inv_m0 * (J_p0 @ J_p0.T)
                + inv_m1 * (J_p1 @ J_p1.T)
                + inv_I0 * (J_t0 @ J_t0.T)
                + inv_I1 * (J_t1 @ J_t1.T)
            )
            JMJT += np.diag(self.compliance[i])

            block_start = 6 * i
            for row in range(6):
                for col in range(6):
                    global_row = block_start + row
                    global_col = block_start + col
                    band_row = bandwidth + global_row - global_col
                    if 0 <= band_row < 2 * bandwidth + 1:
                        self.A_banded[band_row, global_col] += JMJT[row, col]

            if i > 0:
                J_pos_prev = self.jacobian_pos[i - 1]
                J_rot_prev = self.jacobian_rot[i - 1]
                J_p1_prev = J_pos_prev[:, 3:6]
                J_t1_prev = J_rot_prev[:, 3:6]
                coupling = inv_m0 * (J_p1_prev @ J_p0.T) + inv_I0 * (J_t1_prev @ J_t0.T)

                prev_block = 6 * (i - 1)
                for row in range(6):
                    for col in range(6):
                        global_row = prev_block + row
                        global_col = block_start + col
                        band_row = bandwidth + global_row - global_col
                        if 0 <= band_row < 2 * bandwidth + 1:
                            self.A_banded[band_row, global_col] += coupling[row, col]

                        global_row = block_start + col
                        global_col = prev_block + row
                        band_row = bandwidth + global_row - global_col
                        if 0 <= band_row < 2 * bandwidth + 1:
                            self.A_banded[band_row, global_col] += coupling[row, col]

    def _numpy_project_jmjt_banded(self) -> None:
        """NumPy implementation of banded constraint projection."""
        n_edges = self.num_edges
        if n_edges == 0:
            return

        n_dofs = 6 * n_edges
        self.rhs[:n_dofs] = (-self.constraint_values).reshape(n_dofs)

        # Regularization for stability
        regularization = np.float32(1.0e-6)
        self.A_banded[self.bandwidth, :n_dofs] += regularization

        try:
            from scipy.linalg import solve_banded

            delta_lambda = solve_banded(
                (self.bandwidth, self.bandwidth),
                self.A_banded,
                self.rhs[:n_dofs],
                overwrite_ab=False,
                overwrite_b=False,
            )
        except Exception:
            A_dense = np.zeros((n_dofs, n_dofs), dtype=np.float32)
            for col in range(n_dofs):
                for band_row in range(2 * self.bandwidth + 1):
                    row = col + band_row - self.bandwidth
                    if 0 <= row < n_dofs:
                        A_dense[row, col] = self.A_banded[band_row, col]
            delta_lambda = np.linalg.solve(A_dense, self.rhs[:n_dofs])

        inv_masses = self.inv_masses

        self.last_delta_lambda_max = float(np.max(np.abs(delta_lambda))) if delta_lambda.size > 0 else 0.0
        corr_max = 0.0
        for i in range(n_edges):
            dl = delta_lambda[6 * i : 6 * i + 6]
            J_pos = self.jacobian_pos[i]
            J_rot = self.jacobian_rot[i]
            J_p0 = J_pos[:, 0:3]
            J_p1 = J_pos[:, 3:6]
            J_t0 = J_rot[:, 0:3]
            J_t1 = J_rot[:, 3:6]

            inv_m0 = inv_masses[i]
            inv_m1 = inv_masses[i + 1]

            if inv_m0 > 0.0:
                dp0 = inv_m0 * (J_p0.T @ dl)
                self.predicted_positions[i, 0:3] += dp0
                corr_max = max(corr_max, float(np.linalg.norm(dp0)))
            if inv_m1 > 0.0:
                dp1 = inv_m1 * (J_p1.T @ dl)
                self.predicted_positions[i + 1, 0:3] += dp1
                corr_max = max(corr_max, float(np.linalg.norm(dp1)))

            if self.quat_inv_masses[i] > 0.0:
                dtheta0 = 1.0 * (J_t0.T @ dl)
                self._apply_quaternion_correction_g(self.predicted_orientations, i, dtheta0)
                corr_max = max(corr_max, float(np.linalg.norm(dtheta0)))
            if self.quat_inv_masses[i + 1] > 0.0:
                dtheta1 = 1.0 * (J_t1.T @ dl)
                self._apply_quaternion_correction_g(self.predicted_orientations, i + 1, dtheta1)
                corr_max = max(corr_max, float(np.linalg.norm(dtheta1)))

        self.last_correction_max = corr_max

    def _numpy_project_direct(self) -> None:
        """NumPy implementation of non-banded direct constraint projection."""
        n_edges = self.num_edges
        if n_edges == 0:
            return

        self._numpy_update_constraints_banded()
        self._numpy_compute_jacobians_direct()

        n_dofs = 6 * n_edges
        A = np.zeros((n_dofs, n_dofs), dtype=np.float32)
        rhs = (-self.constraint_values).reshape(n_dofs)
        rhs -= (self.compliance * self.lambda_sum).reshape(n_dofs)

        inv_masses = self.inv_masses

        # Use unit mass/inertia for LHS (matching C++ reference)
        inv_masses_lhs = np.ones_like(inv_masses)
        inv_I_lhs = np.ones_like(self.quat_inv_masses)

        for i in range(n_edges):
            J_pos = self.jacobian_pos[i]
            J_rot = self.jacobian_rot[i]
            J_p0 = J_pos[:, 0:3]
            J_p1 = J_pos[:, 3:6]
            J_t0 = J_rot[:, 0:3]
            J_t1 = J_rot[:, 3:6]

            inv_m0 = inv_masses_lhs[i]
            inv_m1 = inv_masses_lhs[i + 1]
            inv_I0 = inv_I_lhs[i]
            inv_I1 = inv_I_lhs[i + 1]

            JMJT = (
                inv_m0 * (J_p0 @ J_p0.T)
                + inv_m1 * (J_p1 @ J_p1.T)
                + inv_I0 * (J_t0 @ J_t0.T)
                + inv_I1 * (J_t1 @ J_t1.T)
            )
            JMJT += np.diag(self.compliance[i])

            block = slice(6 * i, 6 * i + 6)
            A[block, block] += JMJT

            if i > 0:
                J_pos_prev = self.jacobian_pos[i - 1]
                J_rot_prev = self.jacobian_rot[i - 1]
                J_p1_prev = J_pos_prev[:, 3:6]
                J_t1_prev = J_rot_prev[:, 3:6]

                coupling = inv_m0 * (J_p1_prev @ J_p0.T) + inv_I0 * (J_t1_prev @ J_t0.T)
                prev_block = slice(6 * (i - 1), 6 * (i - 1) + 6)
                A[prev_block, block] += coupling
                A[block, prev_block] += coupling.T

        try:
            delta_lambda = np.linalg.solve(A, rhs)
        except np.linalg.LinAlgError:
            delta_lambda = np.linalg.lstsq(A, rhs, rcond=None)[0]

        self.last_delta_lambda_max = float(np.max(np.abs(delta_lambda))) if delta_lambda.size > 0 else 0.0
        self.lambda_sum += delta_lambda.reshape(n_edges, 6)

        corr_max = 0.0
        for i in range(n_edges):
            dl = delta_lambda[6 * i : 6 * i + 6]
            J_pos = self.jacobian_pos[i]
            J_rot = self.jacobian_rot[i]
            J_p0 = J_pos[:, 0:3]
            J_p1 = J_pos[:, 3:6]
            J_t0 = J_rot[:, 0:3]
            J_t1 = J_rot[:, 3:6]

            inv_m0 = inv_masses[i]
            inv_m1 = inv_masses[i + 1]

            if inv_m0 > 0.0:
                dp0 = inv_m0 * (J_p0.T @ dl)
                self.predicted_positions[i, 0:3] += dp0
                corr_max = max(corr_max, float(np.linalg.norm(dp0)))
            if inv_m1 > 0.0:
                dp1 = inv_m1 * (J_p1.T @ dl)
                self.predicted_positions[i + 1, 0:3] += dp1
                corr_max = max(corr_max, float(np.linalg.norm(dp1)))

            if self.quat_inv_masses[i] > 0.0:
                dtheta0 = 1.0 * (J_t0.T @ dl)
                self._apply_quaternion_correction_g(self.predicted_orientations, i, dtheta0)
                corr_max = max(corr_max, float(np.linalg.norm(dtheta0)))
            if self.quat_inv_masses[i + 1] > 0.0:
                dtheta1 = 1.0 * (J_t1.T @ dl)
                self._apply_quaternion_correction_g(self.predicted_orientations, i + 1, dtheta1)
                corr_max = max(corr_max, float(np.linalg.norm(dtheta1)))

        self.last_correction_max = corr_max

    # Override parent methods to use NumPy/Warp when enabled
    
    def predict_positions(self, dt: float, linear_damping: float) -> None:
        """Predict positions using selected implementation."""
        if self._use_warp_step("predict_positions"):
            # Use parent's DLL implementation as fallback for now
            super().predict_positions(dt, linear_damping)
        elif self.numpy_enabled["predict_positions"]:
            self._numpy_predict_positions(dt, linear_damping)
        else:
            super().predict_positions(dt, linear_damping)

    def integrate_positions(self, dt: float) -> None:
        """Integrate positions using selected implementation."""
        if self._use_warp_step("integrate_positions"):
            super().integrate_positions(dt)
        elif self.numpy_enabled["integrate_positions"]:
            self._numpy_integrate_positions(dt)
        else:
            super().integrate_positions(dt)

    def predict_rotations(self, dt: float, angular_damping: float) -> None:
        """Predict rotations using selected implementation."""
        if self._use_warp_step("predict_rotations"):
            super().predict_rotations(dt, angular_damping)
        elif self.numpy_enabled["predict_rotations"]:
            self._numpy_predict_rotations(dt, angular_damping)
        else:
            super().predict_rotations(dt, angular_damping)

    def integrate_rotations(self, dt: float) -> None:
        """Integrate rotations using selected implementation."""
        if self._use_warp_step("integrate_rotations"):
            super().integrate_rotations(dt)
        elif self.numpy_enabled["integrate_rotations"]:
            self._numpy_integrate_rotations(dt)
        else:
            super().integrate_rotations(dt)

    def prepare_constraints(self, dt: float) -> None:
        """Prepare constraints using selected implementation."""
        if self.numpy_enabled["prepare_constraints"]:
            self._numpy_prepare_constraints(dt)
            if self._requires_native_constraint_pipeline():
                super().prepare_constraints(dt)
        else:
            super().prepare_constraints(dt)

    def update_constraints_banded(self) -> None:
        """Update constraints using selected implementation."""
        if self.numpy_enabled["update_constraints_banded"]:
            self._numpy_update_constraints_banded()
            if self._requires_native_constraint_pipeline():
                super().update_constraints_banded()
        else:
            super().update_constraints_banded()

    def compute_jacobians_banded(self) -> None:
        """Compute Jacobians using selected implementation."""
        if self.numpy_enabled["compute_jacobians_banded"]:
            self._numpy_compute_jacobians_direct()
            if self._requires_native_constraint_pipeline():
                super().compute_jacobians_banded()
        else:
            super().compute_jacobians_banded()

    def assemble_jmjt_banded(self) -> None:
        """Assemble JMJT using selected implementation."""
        if self.numpy_enabled["assemble_jmjt_banded"]:
            self._numpy_assemble_jmjt_banded()
            if self._requires_native_constraint_pipeline():
                super().assemble_jmjt_banded()
        else:
            super().assemble_jmjt_banded()

    def project_jmjt_banded(self) -> None:
        """Project constraints using banded solver."""
        if self.numpy_enabled["project_jmjt_banded"]:
            self._numpy_project_jmjt_banded()
        else:
            super().project_jmjt_banded()

    def project_direct(self) -> None:
        """Project constraints using direct solver."""
        if self.numpy_enabled["project_direct"]:
            self._numpy_project_direct()
        else:
            super().project_direct()


__all__ = ["NumpyDirectRodState"]
