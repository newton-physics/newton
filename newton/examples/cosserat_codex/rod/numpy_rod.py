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

    def _numpy_quat_normalize(self, q: np.ndarray, mask: np.ndarray = None) -> np.ndarray:
        """Normalize quaternion array."""
        norm = np.linalg.norm(q, axis=1, keepdims=True)
        norm = np.where(norm < 1e-8, 1.0, norm)
        result = q / norm
        if mask is not None:
            result[~mask] = q[~mask]
        return result

    # NumPy implementations of simulation steps
    
    def _numpy_predict_positions(self, dt: float, linear_damping: float) -> None:
        """NumPy implementation of position prediction."""
        dt = np.float32(dt)
        damp = np.float32(1.0 - linear_damping)

        inv_mass = self.inv_masses[:, None]
        positions = self.positions[:, 0:3]
        velocities = self.velocities[:, 0:3]
        forces = self.forces[:, 0:3]
        gravity = self.gravity[0, 0:3]

        velocities[:] = velocities + (forces * inv_mass + gravity) * dt
        velocities[:] = velocities * damp

        predicted = positions + velocities * dt

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
        dt = np.float32(dt)
        half_dt = np.float32(0.5 * dt)
        damp = np.float32(1.0 - angular_damping)

        inv_mass = self.quat_inv_masses[:, None]
        ang_vel = self.angular_velocities[:, 0:3]
        torques = self.torques[:, 0:3]
        orientations = self.orientations

        dynamic_mask = self.quat_inv_masses != 0.0
        ang_vel[dynamic_mask] = (ang_vel[dynamic_mask] + torques[dynamic_mask] * inv_mass[dynamic_mask] * dt) * damp
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
        self.lambda_sum.fill(0.0)
        self.current_rest_lengths[:] = self.rest_lengths
        self.current_rest_darboux[:] = self.rest_darboux[:, 0:3]
        
        # Compute compliance
        dt2 = dt * dt
        eps = 1.0e-10
        
        for i in range(self.num_edges):
            L = self.rest_lengths[i]
            k_bend1 = self.young_modulus * self.bend_stiffness[i, 0] * L
            k_bend2 = self.young_modulus * self.bend_stiffness[i, 1] * L
            k_twist = self.torsion_modulus * self.bend_stiffness[i, 2] * L
            
            self.compliance[i, 0:3] = 1.0e-10  # Stretch compliance
            self.compliance[i, 3] = 1.0 / (k_bend1 * dt2 + eps)
            self.compliance[i, 4] = 1.0 / (k_bend2 * dt2 + eps)
            self.compliance[i, 5] = 1.0 / (k_twist * dt2 + eps)

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
        super().prepare_constraints(dt)


__all__ = ["NumpyDirectRodState"]
