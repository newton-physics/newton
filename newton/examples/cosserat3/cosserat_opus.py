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

###########################################################################
# Example: Cosserat Rod via DefKitAdv DLL (Position and Orientation Based)
#
# Demonstrates Cosserat rod simulation using the DefKitAdv.dll C++ library
# via ctypes. Uses Newton as the visualization backend.
#
# This example implements the iterative "Position and Orientation Based
# Cosserat Rods" solver from the paper by Kugelstadt & Schömer (2016).
#
# The simulation uses:
# - StretchShear constraints: maintain edge lengths and shear resistance
# - BendTwist constraints: maintain bending and twisting stiffness
#
# Command: uv run python newton/examples/cosserat3/cosserat_opus.py
#
###########################################################################

from __future__ import annotations

import ctypes
import math
import os
import sys
from ctypes import (
    POINTER,
    Structure,
    byref,
    c_float,
    c_int,
    c_void_p,
    pointer,
)
from pathlib import Path
from typing import Optional

import numpy as np
import warp as wp

import newton
import newton.examples


# =============================================================================
# ctypes Data Structures matching C++ layout
# =============================================================================


class btVector3(Structure):
    """ctypes structure matching Bullet's btVector3 (4 floats for alignment)."""

    _fields_ = [
        ("x", c_float),
        ("y", c_float),
        ("z", c_float),
        ("w", c_float),  # Padding for 16-byte alignment
    ]

    def __init__(self, x: float = 0.0, y: float = 0.0, z: float = 0.0):
        super().__init__()
        self.x = x
        self.y = y
        self.z = z
        self.w = 0.0

    def to_numpy(self) -> np.ndarray:
        return np.array([self.x, self.y, self.z], dtype=np.float32)

    @classmethod
    def from_numpy(cls, arr: np.ndarray) -> "btVector3":
        return cls(float(arr[0]), float(arr[1]), float(arr[2]))


class btQuaternion(Structure):
    """ctypes structure matching Bullet's btQuaternion (x, y, z, w order)."""

    _fields_ = [
        ("x", c_float),
        ("y", c_float),
        ("z", c_float),
        ("w", c_float),
    ]

    def __init__(self, x: float = 0.0, y: float = 0.0, z: float = 0.0, w: float = 1.0):
        super().__init__()
        self.x = x
        self.y = y
        self.z = z
        self.w = w

    def to_numpy(self) -> np.ndarray:
        """Return as [x, y, z, w] array."""
        return np.array([self.x, self.y, self.z, self.w], dtype=np.float32)

    @classmethod
    def from_numpy(cls, arr: np.ndarray) -> "btQuaternion":
        """Create from [x, y, z, w] array."""
        return cls(float(arr[0]), float(arr[1]), float(arr[2]), float(arr[3]))

    def normalize(self):
        """Normalize the quaternion in place."""
        norm = math.sqrt(self.x**2 + self.y**2 + self.z**2 + self.w**2)
        if norm > 1e-10:
            self.x /= norm
            self.y /= norm
            self.z /= norm
            self.w /= norm


# =============================================================================
# DefKitAdv DLL Wrapper
# =============================================================================


class DefKitDLL:
    """Wrapper for DefKitAdv.dll providing Cosserat rod simulation functions."""

    def __init__(self, dll_path: Optional[str] = None):
        """Load the DefKitAdv DLL.

        Args:
            dll_path: Path to DefKitAdv.dll. If None, searches in common locations.
        """
        self._dll = None
        self._loaded = False

        if dll_path is None:
            # Search for DLL in common locations
            search_paths = [
                Path(__file__).parent / "DefKitAdv.dll",
                Path(__file__).parent.parent.parent.parent / "unity_ref" / "Native" / "x64" / "Release" / "DefKitAdv.dll",
                Path(__file__).parent.parent.parent.parent / "unity_ref" / "Native" / "build" / "Release" / "DefKitAdv.dll",
                Path(os.environ.get("DEFKIT_DLL_PATH", "")) / "DefKitAdv.dll",
            ]
            for path in search_paths:
                if path.exists():
                    dll_path = str(path)
                    break

        if dll_path is None or not Path(dll_path).exists():
            print(f"Warning: DefKitAdv.dll not found. Simulation will use fallback Python implementation.")
            self._loaded = False
            return

        try:
            self._dll = ctypes.CDLL(dll_path)
            self._setup_function_signatures()
            self._loaded = True
            print(f"Loaded DefKitAdv.dll from: {dll_path}")
        except OSError as e:
            print(f"Warning: Failed to load DefKitAdv.dll: {e}")
            self._loaded = False

    def _setup_function_signatures(self):
        """Set up ctypes function signatures for all exported functions."""
        if self._dll is None:
            return

        # InitDirectElasticRod
        # IntPtr InitDirectElasticRod(int pointsCount, btVector3* positions, btQuaternion* orientations,
        #                             float radius, float* restLengths, float youngModulus, float torsionModulus)
        self._dll.InitDirectElasticRod.argtypes = [
            c_int,  # pointsCount
            POINTER(btVector3),  # positions
            POINTER(btQuaternion),  # orientations
            c_float,  # radius
            POINTER(c_float),  # restLengths
            c_float,  # youngModulus
            c_float,  # torsionModulus
        ]
        self._dll.InitDirectElasticRod.restype = c_void_p

        # PrepareDirectElasticRodConstraints
        self._dll.PrepareDirectElasticRodConstraints.argtypes = [
            c_void_p,  # rod pointer
            c_int,  # pointsCount (num constraints = pointsCount - 1)
            c_float,  # dt
            POINTER(btVector3),  # bendStiffness (actually Vector4 but we use btVector3)
            POINTER(btVector3),  # restDarboux
            POINTER(c_float),  # restLengths
            c_float,  # youngModulusMult
            c_float,  # torsionModulusMult
        ]
        self._dll.PrepareDirectElasticRodConstraints.restype = None

        # UpdateConstraints_DirectElasticRodConstraintsBanded
        self._dll.UpdateConstraints_DirectElasticRodConstraintsBanded.argtypes = [
            c_void_p,  # rod pointer
            c_int,  # pointsCount
            POINTER(btVector3),  # positions
            POINTER(btQuaternion),  # orientations
            POINTER(c_float),  # invMasses
        ]
        self._dll.UpdateConstraints_DirectElasticRodConstraintsBanded.restype = None

        # ComputeJacobians_DirectElasticRodConstraintsBanded
        self._dll.ComputeJacobians_DirectElasticRodConstraintsBanded.argtypes = [
            c_void_p,  # rod pointer
            c_int,  # startId
            c_int,  # count
            POINTER(btVector3),  # positions
            POINTER(btQuaternion),  # orientations
            POINTER(c_float),  # invMasses
        ]
        self._dll.ComputeJacobians_DirectElasticRodConstraintsBanded.restype = None

        # AssembleJMJT_DirectElasticRodConstraintsBanded
        self._dll.AssembleJMJT_DirectElasticRodConstraintsBanded.argtypes = [
            c_void_p,  # rod pointer
            c_int,  # startId
            c_int,  # count
            POINTER(btVector3),  # positions
            POINTER(btQuaternion),  # orientations
            POINTER(c_float),  # invMasses
        ]
        self._dll.AssembleJMJT_DirectElasticRodConstraintsBanded.restype = None

        # ProjectJMJT_DirectElasticRodConstraintsBanded
        self._dll.ProjectJMJT_DirectElasticRodConstraintsBanded.argtypes = [
            c_void_p,  # rod pointer
            c_int,  # pointsCount
            POINTER(btVector3),  # positions
            POINTER(btQuaternion),  # orientations
            POINTER(c_float),  # invMasses
            POINTER(btVector3),  # posCorr
            POINTER(btQuaternion),  # rotCorr
        ]
        self._dll.ProjectJMJT_DirectElasticRodConstraintsBanded.restype = None

        # DestroyDirectElasticRod
        self._dll.DestroyDirectElasticRod.argtypes = [c_void_p]
        self._dll.DestroyDirectElasticRod.restype = None

        # PredictRotationsPBD (from DefKit.dll, but may be in DefKitAdv too)
        try:
            self._dll.PredictRotationsPBD.argtypes = [
                c_float,  # dt
                c_float,  # damping
                c_int,  # pointsCount
                POINTER(btQuaternion),  # orientations
                POINTER(btQuaternion),  # predictedOrientations
                POINTER(btVector3),  # angVelocities
                POINTER(btVector3),  # torques
                POINTER(c_float),  # quatInvMass
            ]
            self._dll.PredictRotationsPBD.restype = None
            self._has_predict_rotations = True
        except AttributeError:
            self._has_predict_rotations = False

        # IntegrateRotationsPBD
        try:
            self._dll.IntegrateRotationsPBD.argtypes = [
                c_float,  # dt
                c_int,  # pointsCount
                POINTER(btQuaternion),  # orientations
                POINTER(btQuaternion),  # predictedOrientations
                POINTER(btQuaternion),  # prevOrientations
                POINTER(btVector3),  # angVelocities
                POINTER(c_float),  # quatInvMass
            ]
            self._dll.IntegrateRotationsPBD.restype = None
            self._has_integrate_rotations = True
        except AttributeError:
            self._has_integrate_rotations = False

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def init_direct_elastic_rod(
        self,
        positions: np.ndarray,
        orientations: np.ndarray,
        radius: float,
        rest_lengths: np.ndarray,
        young_modulus: float,
        torsion_modulus: float,
    ) -> c_void_p:
        """Initialize a direct elastic rod solver.

        Args:
            positions: (N, 3) array of particle positions
            orientations: (N-1, 4) array of quaternions [x, y, z, w]
            radius: Rod radius
            rest_lengths: (N-1,) array of rest lengths
            young_modulus: Young's modulus
            torsion_modulus: Torsion modulus

        Returns:
            Pointer to the rod constraint object
        """
        if not self._loaded:
            return None

        n = len(positions)
        pos_array = (btVector3 * n)()
        for i in range(n):
            pos_array[i] = btVector3.from_numpy(positions[i])

        quat_array = (btQuaternion * (n - 1))()
        for i in range(n - 1):
            quat_array[i] = btQuaternion.from_numpy(orientations[i])

        rest_len_array = (c_float * (n - 1))(*rest_lengths.astype(np.float32))

        return self._dll.InitDirectElasticRod(
            n,
            pos_array,
            quat_array,
            c_float(radius),
            rest_len_array,
            c_float(young_modulus),
            c_float(torsion_modulus),
        )

    def prepare_constraints(
        self,
        rod_ptr: c_void_p,
        num_constraints: int,
        dt: float,
        bend_stiffness: np.ndarray,
        rest_darboux: np.ndarray,
        rest_lengths: np.ndarray,
        young_mult: float = 1.0,
        torsion_mult: float = 1.0,
    ):
        """Prepare constraints before projection iterations."""
        if not self._loaded or rod_ptr is None:
            return

        bend_array = (btVector3 * num_constraints)()
        darboux_array = (btVector3 * num_constraints)()
        for i in range(num_constraints):
            bend_array[i] = btVector3.from_numpy(bend_stiffness[i])
            darboux_array[i] = btVector3.from_numpy(rest_darboux[i])

        rest_len_array = (c_float * num_constraints)(*rest_lengths.astype(np.float32))

        self._dll.PrepareDirectElasticRodConstraints(
            rod_ptr,
            num_constraints,
            c_float(dt),
            bend_array,
            darboux_array,
            rest_len_array,
            c_float(young_mult),
            c_float(torsion_mult),
        )

    def update_constraints(
        self,
        rod_ptr: c_void_p,
        positions: np.ndarray,
        orientations: np.ndarray,
        inv_masses: np.ndarray,
    ):
        """Update constraint state with current positions and orientations."""
        if not self._loaded or rod_ptr is None:
            return

        n = len(positions)
        pos_array = (btVector3 * n)()
        for i in range(n):
            pos_array[i] = btVector3.from_numpy(positions[i])

        quat_array = (btQuaternion * n)()
        for i in range(n):
            if i < len(orientations):
                quat_array[i] = btQuaternion.from_numpy(orientations[i])
            else:
                quat_array[i] = btQuaternion(0, 0, 0, 1)

        mass_array = (c_float * n)(*inv_masses.astype(np.float32))

        self._dll.UpdateConstraints_DirectElasticRodConstraintsBanded(
            rod_ptr, n, pos_array, quat_array, mass_array
        )

    def compute_jacobians(
        self,
        rod_ptr: c_void_p,
        start_id: int,
        count: int,
        positions: np.ndarray,
        orientations: np.ndarray,
        inv_masses: np.ndarray,
    ):
        """Compute Jacobians for the banded solver."""
        if not self._loaded or rod_ptr is None:
            return

        n = len(positions)
        pos_array = (btVector3 * n)()
        for i in range(n):
            pos_array[i] = btVector3.from_numpy(positions[i])

        quat_array = (btQuaternion * n)()
        for i in range(n):
            if i < len(orientations):
                quat_array[i] = btQuaternion.from_numpy(orientations[i])
            else:
                quat_array[i] = btQuaternion(0, 0, 0, 1)

        mass_array = (c_float * n)(*inv_masses.astype(np.float32))

        self._dll.ComputeJacobians_DirectElasticRodConstraintsBanded(
            rod_ptr, start_id, count, pos_array, quat_array, mass_array
        )

    def assemble_jmjt(
        self,
        rod_ptr: c_void_p,
        start_id: int,
        count: int,
        positions: np.ndarray,
        orientations: np.ndarray,
        inv_masses: np.ndarray,
    ):
        """Assemble the JMJT matrix for the banded solver."""
        if not self._loaded or rod_ptr is None:
            return

        n = len(positions)
        pos_array = (btVector3 * n)()
        for i in range(n):
            pos_array[i] = btVector3.from_numpy(positions[i])

        quat_array = (btQuaternion * n)()
        for i in range(n):
            if i < len(orientations):
                quat_array[i] = btQuaternion.from_numpy(orientations[i])
            else:
                quat_array[i] = btQuaternion(0, 0, 0, 1)

        mass_array = (c_float * n)(*inv_masses.astype(np.float32))

        self._dll.AssembleJMJT_DirectElasticRodConstraintsBanded(
            rod_ptr, start_id, count, pos_array, quat_array, mass_array
        )

    def project_constraints(
        self,
        rod_ptr: c_void_p,
        positions: np.ndarray,
        orientations: np.ndarray,
        inv_masses: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Project constraints and get position/orientation corrections.

        Returns:
            Tuple of (position_corrections, orientation_corrections)
        """
        if not self._loaded or rod_ptr is None:
            return np.zeros_like(positions), np.zeros_like(orientations)

        n = len(positions)
        pos_array = (btVector3 * n)()
        for i in range(n):
            pos_array[i] = btVector3.from_numpy(positions[i])

        quat_array = (btQuaternion * n)()
        for i in range(n):
            if i < len(orientations):
                quat_array[i] = btQuaternion.from_numpy(orientations[i])
            else:
                quat_array[i] = btQuaternion(0, 0, 0, 1)

        mass_array = (c_float * n)(*inv_masses.astype(np.float32))

        pos_corr = (btVector3 * n)()
        rot_corr = (btQuaternion * n)()

        self._dll.ProjectJMJT_DirectElasticRodConstraintsBanded(
            rod_ptr, n, pos_array, quat_array, mass_array, pos_corr, rot_corr
        )

        # Extract corrected positions and orientations
        new_positions = np.array([[pos_array[i].x, pos_array[i].y, pos_array[i].z] for i in range(n)], dtype=np.float32)
        new_orientations = np.array(
            [[quat_array[i].x, quat_array[i].y, quat_array[i].z, quat_array[i].w] for i in range(n)], dtype=np.float32
        )

        return new_positions, new_orientations

    def destroy_rod(self, rod_ptr: c_void_p):
        """Clean up rod resources."""
        if self._loaded and rod_ptr is not None:
            self._dll.DestroyDirectElasticRod(rod_ptr)


# =============================================================================
# Rod State Manager
# =============================================================================


class RodState:
    """Manages the state of a Cosserat rod for PBD simulation."""

    def __init__(
        self,
        num_particles: int,
        positions: np.ndarray,
        orientations: np.ndarray,
        rest_lengths: np.ndarray,
        particle_inv_mass: np.ndarray,
        quat_inv_mass: np.ndarray,
        radius: float = 0.01,
        young_modulus: float = 1e8,
        torsion_modulus: float = 1e8,
    ):
        """Initialize rod state.

        Args:
            num_particles: Number of particles in the rod
            positions: (N, 3) initial positions
            orientations: (N-1, 4) initial quaternions [x, y, z, w]
            rest_lengths: (N-1,) rest lengths between particles
            particle_inv_mass: (N,) inverse masses for positions
            quat_inv_mass: (N-1,) inverse masses for quaternions
            radius: Rod radius
            young_modulus: Young's modulus
            torsion_modulus: Torsion modulus
        """
        self.num_particles = num_particles
        self.num_edges = num_particles - 1

        # Position state
        self.positions = positions.astype(np.float32).copy()
        self.predicted_positions = self.positions.copy()
        self.prev_positions = self.positions.copy()
        self.velocities = np.zeros((num_particles, 3), dtype=np.float32)
        self.forces = np.zeros((num_particles, 3), dtype=np.float32)
        self.particle_inv_mass = particle_inv_mass.astype(np.float32).copy()

        # Orientation state
        self.orientations = orientations.astype(np.float32).copy()
        self.predicted_orientations = self.orientations.copy()
        self.prev_orientations = self.orientations.copy()
        self.angular_velocities = np.zeros((self.num_edges, 3), dtype=np.float32)
        self.torques = np.zeros((self.num_edges, 3), dtype=np.float32)
        self.quat_inv_mass = quat_inv_mass.astype(np.float32).copy()

        # Rest configuration
        self.rest_lengths = rest_lengths.astype(np.float32).copy()
        self.rest_darboux = self._compute_rest_darboux()

        # Material properties
        self.radius = radius
        self.young_modulus = young_modulus
        self.torsion_modulus = torsion_modulus

        # Compute bending stiffness coefficients
        self.bend_stiffness = self._compute_bend_stiffness()

    def _compute_rest_darboux(self) -> np.ndarray:
        """Compute rest Darboux vectors from initial orientations."""
        rest_darboux = np.zeros((self.num_edges, 3), dtype=np.float32)

        for i in range(self.num_edges - 1):
            q0 = self.orientations[i]
            q1 = self.orientations[i + 1] if i + 1 < len(self.orientations) else q0

            # Darboux vector: omega = 2 * (q0^-1 * q1).vec / avg_length
            q0_conj = np.array([-q0[0], -q0[1], -q0[2], q0[3]], dtype=np.float32)
            omega = self._quat_multiply(q0_conj, q1)
            rest_darboux[i] = omega[:3]  # Just the vector part

        return rest_darboux

    def _compute_bend_stiffness(self) -> np.ndarray:
        """Compute bending/torsion stiffness coefficients."""
        # Second moment of area for circular cross-section
        I = (math.pi / 4.0) * self.radius**4
        bend_stiff = self.young_modulus * I
        torsion_stiff = 2.0 * self.torsion_modulus * I

        # Stiffness vector: [bend_x, torsion_y, bend_z]
        stiffness = np.zeros((self.num_edges, 3), dtype=np.float32)
        for i in range(self.num_edges):
            stiffness[i] = [bend_stiff, torsion_stiff, bend_stiff]

        return stiffness

    @staticmethod
    def _quat_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
        """Multiply two quaternions [x, y, z, w]."""
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
    def _normalize_quat(q: np.ndarray) -> np.ndarray:
        """Normalize a quaternion."""
        norm = np.linalg.norm(q)
        if norm > 1e-10:
            return q / norm
        return np.array([0, 0, 0, 1], dtype=np.float32)


# =============================================================================
# PBD Integration Functions (Python fallback)
# =============================================================================


def predict_positions(
    state: RodState,
    dt: float,
    gravity: np.ndarray,
    damping: float = 0.01,
):
    """Predict positions using semi-implicit Euler integration."""
    damp = 1.0 - damping

    for i in range(state.num_particles):
        if state.particle_inv_mass[i] > 0:
            # v += (f * inv_m + g) * dt
            state.velocities[i] += (state.forces[i] * state.particle_inv_mass[i] + gravity) * dt
            state.velocities[i] *= damp

            # x_pred = x + v * dt
            state.predicted_positions[i] = state.positions[i] + state.velocities[i] * dt
        else:
            state.velocities[i] = np.zeros(3, dtype=np.float32)
            state.predicted_positions[i] = state.positions[i].copy()


def predict_rotations(
    state: RodState,
    dt: float,
    damping: float = 0.001,
):
    """Predict orientations using quaternion integration."""
    damp = 1.0 - damping
    half_dt = dt * 0.5

    for i in range(state.num_edges):
        if state.quat_inv_mass[i] > 0:
            # Simple angular velocity integration (no nutation)
            state.angular_velocities[i] += state.torques[i] * state.quat_inv_mass[i] * dt
            state.angular_velocities[i] *= damp

            # q_pred = q + 0.5 * omega * q * dt
            q = state.orientations[i]
            omega = state.angular_velocities[i]

            # omega as quaternion: [omega_x, omega_y, omega_z, 0]
            omega_q = np.array([omega[0], omega[1], omega[2], 0], dtype=np.float32)

            # dq = 0.5 * omega_q * q
            dq = 0.5 * state._quat_multiply(omega_q, q)

            state.predicted_orientations[i] = state._normalize_quat(q + dq * dt)
        else:
            state.angular_velocities[i] = np.zeros(3, dtype=np.float32)
            state.predicted_orientations[i] = state.orientations[i].copy()


def integrate_positions(state: RodState, dt: float):
    """Integrate velocities from position changes."""
    dt_inv = 1.0 / dt

    for i in range(state.num_particles):
        if state.particle_inv_mass[i] > 0:
            state.velocities[i] = (state.predicted_positions[i] - state.positions[i]) * dt_inv
            state.positions[i] = state.predicted_positions[i].copy()


def integrate_rotations(state: RodState, dt: float):
    """Integrate angular velocities from orientation changes."""
    dt_inv_2 = 2.0 / dt

    for i in range(state.num_edges):
        if state.quat_inv_mass[i] > 0:
            # relRot = predicted * current^-1
            q_curr = state.orientations[i]
            q_pred = state.predicted_orientations[i]
            q_curr_conj = np.array([-q_curr[0], -q_curr[1], -q_curr[2], q_curr[3]], dtype=np.float32)

            rel_rot = state._quat_multiply(q_pred, q_curr_conj)
            state.angular_velocities[i] = rel_rot[:3] * dt_inv_2

            state.prev_orientations[i] = state.orientations[i].copy()
            state.orientations[i] = state.predicted_orientations[i].copy()


# =============================================================================
# Quaternion Utilities
# =============================================================================


def rotate_vector_by_quaternion(v: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Rotate vector v by quaternion q [x, y, z, w]."""
    x, y, z, w = q[0], q[1], q[2], q[3]
    vx, vy, vz = v[0], v[1], v[2]

    # Optimized q * v * q^-1
    tx = 2.0 * (y * vz - z * vy)
    ty = 2.0 * (z * vx - x * vz)
    tz = 2.0 * (x * vy - y * vx)

    return np.array(
        [
            vx + w * tx + y * tz - z * ty,
            vy + w * ty + z * tx - x * tz,
            vz + w * tz + x * ty - y * tx,
        ],
        dtype=np.float32,
    )


def quaternion_from_two_vectors(v1: np.ndarray, v2: np.ndarray) -> np.ndarray:
    """Compute quaternion that rotates v1 to v2."""
    v1 = v1 / np.linalg.norm(v1)
    v2 = v2 / np.linalg.norm(v2)

    dot = np.dot(v1, v2)

    if dot > 0.999999:
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    elif dot < -0.999999:
        ortho = np.cross(v1, [1, 0, 0]) if abs(v1[0]) < 0.9 else np.cross(v1, [0, 1, 0])
        ortho = ortho / np.linalg.norm(ortho)
        return np.array([ortho[0], ortho[1], ortho[2], 0.0], dtype=np.float32)

    cross = np.cross(v1, v2)
    w = 1.0 + dot
    q = np.array([cross[0], cross[1], cross[2], w], dtype=np.float32)
    return q / np.linalg.norm(q)


# =============================================================================
# Example Class
# =============================================================================


class Example:
    """Cosserat rod simulation using DefKitAdv.dll with Newton visualization."""

    def __init__(self, viewer, args=None):
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_time = 0.0
        self.substeps = 8  # More substeps for stability
        self.iterations = 5  # More iterations for constraint solver

        self.viewer = viewer
        self.args = args

        # Rod parameters
        self.num_particles = 20
        self.segment_length = 0.1
        self.particle_radius = 0.02
        self.rod_radius = 0.01

        # Material properties (from RodParams.txt reference)
        self.young_modulus = 1e6
        self.torsion_modulus = 1e6

        # Gravity (Z-down for Newton's coordinate system)
        self.gravity = np.array([0.0, 0.0, -9.81], dtype=np.float32)
        self.gravity_enabled = True

        # Damping
        self.position_damping = 0.01
        self.rotation_damping = 0.001

        # Initialize DLL wrapper
        self.dll = DefKitDLL()
        self.rod_ptr = None

        # Build rod state
        self._build_rod()
        
        print(f"Rod initialized: {self.num_particles} particles")
        print(f"Initial tip position: {self.state.positions[-1]}")
        print(f"Gravity: {self.gravity}")

        # Initialize DLL rod if available
        if self.dll.is_loaded:
            self.rod_ptr = self.dll.init_direct_elastic_rod(
                self.state.positions,
                self.state.orientations,
                self.rod_radius,
                self.state.rest_lengths,
                self.young_modulus,
                self.torsion_modulus,
            )

        # Build Newton model for visualization
        self._build_newton_model()

        # Director visualization
        self.show_directors = True
        self.director_scale = 0.05

        # UI state
        self._gravity_key_was_down = False
        self._reset_key_was_down = False

    def _build_rod(self):
        """Build initial rod configuration as a horizontal cantilever."""
        # Create horizontal rod along X axis (extending from anchor)
        # This will bend downward under gravity (Z-down)
        positions = np.zeros((self.num_particles, 3), dtype=np.float32)
        for i in range(self.num_particles):
            positions[i] = [i * self.segment_length, 0.0, 1.5]  # Horizontal along X, at z=1.5

        # Identity quaternion (local frame aligned with world)
        q_init = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
        orientations = np.tile(q_init, (self.num_particles - 1, 1))

        rest_lengths = np.full(self.num_particles - 1, self.segment_length, dtype=np.float32)

        # Fix first particle (anchor point)
        particle_inv_mass = np.ones(self.num_particles, dtype=np.float32)
        particle_inv_mass[0] = 0.0  # Fixed anchor

        quat_inv_mass = np.ones(self.num_particles - 1, dtype=np.float32)
        # First quaternion can still rotate (cantilever can bend at root)

        self.state = RodState(
            num_particles=self.num_particles,
            positions=positions,
            orientations=orientations,
            rest_lengths=rest_lengths,
            particle_inv_mass=particle_inv_mass,
            quat_inv_mass=quat_inv_mass,
            radius=self.rod_radius,
            young_modulus=self.young_modulus,
            torsion_modulus=self.torsion_modulus,
        )

        # Store initial state for reset
        self._initial_positions = positions.copy()
        self._initial_orientations = orientations.copy()

    def _build_newton_model(self):
        """Build Newton model for visualization."""
        builder = newton.ModelBuilder()
        builder.add_ground_plane()

        # Add particles
        for i in range(self.num_particles):
            mass = 0.0 if i == 0 else 1.0
            pos = tuple(self.state.positions[i])
            builder.add_particle(pos=pos, vel=(0.0, 0.0, 0.0), mass=mass, radius=self.particle_radius)

        self.model = builder.finalize()
        self.newton_state = self.model.state()

        self._sync_state_to_newton()

        self.viewer.set_model(self.model)
        self.viewer.show_particles = True

        # Pre-allocate visualization buffers
        device = self.model.device
        num_director_lines = (self.num_particles - 1) * 3
        self.director_starts = wp.zeros(num_director_lines, dtype=wp.vec3, device=device)
        self.director_ends = wp.zeros(num_director_lines, dtype=wp.vec3, device=device)
        self.director_colors = wp.zeros(num_director_lines, dtype=wp.vec3, device=device)

    def _sync_state_to_newton(self):
        """Sync simulation state to Newton for visualization."""
        positions_wp = wp.array(self.state.positions, dtype=wp.vec3, device=self.model.device)
        self.newton_state.particle_q.assign(positions_wp)

    def _simulate_substep(self, dt: float):
        """Run one simulation substep."""
        gravity = self.gravity if self.gravity_enabled else np.zeros(3, dtype=np.float32)

        # Predict positions and rotations
        predict_positions(self.state, dt, gravity, self.position_damping)
        predict_rotations(self.state, dt, self.rotation_damping)

        # Project constraints
        if self.dll.is_loaded and self.rod_ptr is not None:
            # Use DLL for constraint projection
            self.dll.prepare_constraints(
                self.rod_ptr,
                self.state.num_edges,
                dt,
                self.state.bend_stiffness,
                self.state.rest_darboux,
                self.state.rest_lengths,
                1.0,
                1.0,
            )

            for _ in range(self.iterations):
                # Update constraints with predicted state
                self.dll.update_constraints(
                    self.rod_ptr,
                    self.state.predicted_positions,
                    self.state.predicted_orientations,
                    self.state.particle_inv_mass,
                )

                # Compute Jacobians
                self.dll.compute_jacobians(
                    self.rod_ptr,
                    0,
                    self.state.num_edges,
                    self.state.predicted_positions,
                    self.state.predicted_orientations,
                    self.state.particle_inv_mass,
                )

                # Assemble system matrix
                self.dll.assemble_jmjt(
                    self.rod_ptr,
                    0,
                    self.state.num_edges,
                    self.state.predicted_positions,
                    self.state.predicted_orientations,
                    self.state.particle_inv_mass,
                )

                # Project and get corrections
                new_pos, new_quat = self.dll.project_constraints(
                    self.rod_ptr,
                    self.state.predicted_positions,
                    self.state.predicted_orientations,
                    self.state.particle_inv_mass,
                )

                self.state.predicted_positions = new_pos
                self.state.predicted_orientations = new_quat[: self.state.num_edges]
        else:
            # Fallback: simple distance constraints
            self._project_distance_constraints()

        # Integrate velocities
        integrate_positions(self.state, dt)
        integrate_rotations(self.state, dt)

    def _project_distance_constraints(self):
        """Simple distance constraint projection (fallback when DLL not available)."""
        # Use many more iterations for better convergence
        num_iterations = max(self.iterations * 10, 20)
        
        for _ in range(num_iterations):
            for i in range(self.state.num_edges):
                p0 = self.state.predicted_positions[i].copy()
                p1 = self.state.predicted_positions[i + 1].copy()

                diff = p1 - p0
                dist = np.linalg.norm(diff)
                if dist < 1e-10:
                    continue

                rest = self.state.rest_lengths[i]
                error = dist - rest

                w0 = self.state.particle_inv_mass[i]
                w1 = self.state.particle_inv_mass[i + 1]
                w_sum = w0 + w1
                if w_sum < 1e-10:
                    continue

                # Correction direction (unit vector from p0 to p1)
                direction = diff / dist
                # Total correction magnitude
                correction_mag = error / w_sum
                
                # Apply corrections
                self.state.predicted_positions[i] += w0 * correction_mag * direction
                self.state.predicted_positions[i + 1] -= w1 * correction_mag * direction

    def step(self):
        """Advance simulation by one frame."""
        self._handle_keyboard_input()

        sub_dt = self.frame_dt / self.substeps
        for _ in range(self.substeps):
            self._simulate_substep(sub_dt)

        self._sync_state_to_newton()
        self.sim_time += self.frame_dt

    def _handle_keyboard_input(self):
        """Handle keyboard input for interactive controls."""
        if not hasattr(self.viewer, "is_key_down"):
            return

        try:
            import pyglet.window.key as key
        except ImportError:
            return

        # Gravity toggle
        g_down = self.viewer.is_key_down(key.G)
        if g_down and not self._gravity_key_was_down:
            self.gravity_enabled = not self.gravity_enabled
            print(f"Gravity: {'ON' if self.gravity_enabled else 'OFF'}")
        self._gravity_key_was_down = g_down

        # Reset
        r_down = self.viewer.is_key_down(key.R)
        if r_down and not self._reset_key_was_down:
            self._reset_simulation()
            print("Reset simulation")
        self._reset_key_was_down = r_down

    def _reset_simulation(self):
        """Reset simulation to initial state."""
        self.state.positions = self._initial_positions.copy()
        self.state.predicted_positions = self._initial_positions.copy()
        self.state.velocities = np.zeros_like(self.state.velocities)
        self.state.forces = np.zeros_like(self.state.forces)

        self.state.orientations = self._initial_orientations.copy()
        self.state.predicted_orientations = self._initial_orientations.copy()
        self.state.angular_velocities = np.zeros_like(self.state.angular_velocities)
        self.state.torques = np.zeros_like(self.state.torques)

        self._sync_state_to_newton()
        self.sim_time = 0.0

    def _update_director_visualization(self):
        """Update director frame lines for visualization."""
        starts = []
        ends = []
        colors = []

        for i in range(self.state.num_edges):
            # Get edge midpoint
            p0 = self.state.positions[i]
            p1 = self.state.positions[i + 1]
            midpoint = 0.5 * (p0 + p1)

            # Get quaternion
            q = self.state.orientations[i]

            # Compute directors
            d1 = rotate_vector_by_quaternion(np.array([1, 0, 0], dtype=np.float32), q)
            d2 = rotate_vector_by_quaternion(np.array([0, 1, 0], dtype=np.float32), q)
            d3 = rotate_vector_by_quaternion(np.array([0, 0, 1], dtype=np.float32), q)

            scale = self.director_scale

            # d1 - red (X)
            starts.append(midpoint)
            ends.append(midpoint + d1 * scale)
            colors.append([1.0, 0.0, 0.0])

            # d2 - green (Y)
            starts.append(midpoint)
            ends.append(midpoint + d2 * scale)
            colors.append([0.0, 1.0, 0.0])

            # d3 - blue (Z / tangent)
            starts.append(midpoint)
            ends.append(midpoint + d3 * scale)
            colors.append([0.0, 0.0, 1.0])

        self.director_starts = wp.array(starts, dtype=wp.vec3, device=self.model.device)
        self.director_ends = wp.array(ends, dtype=wp.vec3, device=self.model.device)
        self.director_colors = wp.array(colors, dtype=wp.vec3, device=self.model.device)

    def render(self):
        """Render current state."""
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.newton_state)

        # Draw rod segments
        starts = wp.array(self.state.positions[:-1], dtype=wp.vec3, device=self.model.device)
        ends = wp.array(self.state.positions[1:], dtype=wp.vec3, device=self.model.device)
        rod_colors = wp.array([[0.2, 0.6, 1.0]] * self.state.num_edges, dtype=wp.vec3, device=self.model.device)
        self.viewer.log_lines("/rod", starts, ends, rod_colors)

        # Draw director frames
        if self.show_directors:
            self._update_director_visualization()
            self.viewer.log_lines("/directors", self.director_starts, self.director_ends, self.director_colors)
        else:
            self.viewer.log_lines("/directors", None, None, None)

        self.viewer.end_frame()

    def gui(self, ui):
        """GUI controls."""
        ui.text("DefKitAdv Cosserat Rod")
        ui.text(f"DLL Loaded: {'Yes' if self.dll.is_loaded else 'No (using fallback)'}")
        ui.text(f"Particles: {self.num_particles}")
        ui.text(f"Simulation time: {self.sim_time:.2f}s")

        ui.separator()
        ui.text("Simulation Parameters")
        _, self.gravity_enabled = ui.checkbox("Gravity (G)", self.gravity_enabled)
        _, self.substeps = ui.slider_int("Substeps", self.substeps, 1, 16)
        _, self.iterations = ui.slider_int("Iterations", self.iterations, 1, 8)

        ui.separator()
        ui.text("Visualization")
        _, self.show_directors = ui.checkbox("Show Directors", self.show_directors)
        _, self.director_scale = ui.slider_float("Director Scale", self.director_scale, 0.01, 0.2)

        ui.separator()
        ui.text("Material Properties")
        _, ym = ui.slider_float("Young's Modulus (log)", math.log10(self.young_modulus), 4, 10)
        self.young_modulus = 10**ym
        _, tm = ui.slider_float("Torsion Modulus (log)", math.log10(self.torsion_modulus), 4, 10)
        self.torsion_modulus = 10**tm

        ui.separator()
        ui.text("Controls:")
        ui.text("  G: Toggle gravity")
        ui.text("  R: Reset simulation")

        ui.separator()
        ui.text("Debug Info:")
        tip_pos = self.state.positions[-1]
        ui.text(f"  Tip: ({tip_pos[0]:.3f}, {tip_pos[1]:.3f}, {tip_pos[2]:.3f})")

        # Show segment lengths
        lengths = [
            np.linalg.norm(self.state.positions[i + 1] - self.state.positions[i]) for i in range(min(3, self.state.num_edges))
        ]
        ui.text(f"  Lengths: {[f'{l:.4f}' for l in lengths]}...")

    def test_final(self):
        """Validation method run after simulation completes."""
        # Verify first particle (anchor) hasn't moved
        anchor_pos = self.state.positions[0]
        initial_anchor = self._initial_positions[0]
        dist = np.linalg.norm(anchor_pos - initial_anchor)
        assert dist < 0.01, f"Anchor moved: distance = {dist}"

        # Verify segment lengths are preserved within tolerance
        for i in range(self.state.num_edges):
            actual = np.linalg.norm(self.state.positions[i + 1] - self.state.positions[i])
            expected = self.state.rest_lengths[i]
            error = abs(actual - expected) / expected
            assert error < 0.2, f"Segment {i} length error {error*100:.1f}% exceeds 20%"

        # Verify quaternions are normalized
        for i in range(self.state.num_edges):
            q = self.state.orientations[i]
            norm = np.linalg.norm(q)
            assert abs(norm - 1.0) < 0.1, f"Quaternion {i} not normalized: norm = {norm}"

    def __del__(self):
        """Clean up resources."""
        if hasattr(self, "dll") and hasattr(self, "rod_ptr"):
            if self.rod_ptr is not None:
                self.dll.destroy_rod(self.rod_ptr)


# =============================================================================
# Main Entry Point
# =============================================================================


if __name__ == "__main__":
    viewer, args = newton.examples.init()

    if isinstance(viewer, newton.viewer.ViewerGL):
        viewer.show_particles = True

    example = Example(viewer, args)
    newton.examples.run(example, args)
