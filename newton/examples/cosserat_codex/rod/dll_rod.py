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

"""DefKit DLL-based rod state implementation.

This module provides the native C++ DLL interface for Cosserat rod simulation
using the DefKitAdv library.
"""

from __future__ import annotations

import atexit
import ctypes
import os

import numpy as np

from .base import RodStateBase


# ==============================================================================
# Ctypes structures
# ==============================================================================


class BtVector3(ctypes.Structure):
    """4-component vector structure matching Bullet Physics layout."""

    _fields_ = [
        ("x", ctypes.c_float),
        ("y", ctypes.c_float),
        ("z", ctypes.c_float),
        ("w", ctypes.c_float),
    ]


class BtQuaternion(ctypes.Structure):
    """4-component quaternion structure matching Bullet Physics layout."""

    _fields_ = [
        ("x", ctypes.c_float),
        ("y", ctypes.c_float),
        ("z", ctypes.c_float),
        ("w", ctypes.c_float),
    ]


# ==============================================================================
# Helper functions
# ==============================================================================


def _as_ptr(array: np.ndarray, ctype):
    """Convert numpy array to ctypes pointer."""
    if array.dtype != np.float32:
        raise TypeError(f"Expected float32 array, got {array.dtype}")
    if not array.flags["C_CONTIGUOUS"]:
        raise ValueError("Expected C-contiguous array")
    return array.ctypes.data_as(ctypes.POINTER(ctype))


def _as_float_ptr(array: np.ndarray):
    """Convert numpy array to float pointer."""
    if array.dtype != np.float32:
        raise TypeError(f"Expected float32 array, got {array.dtype}")
    if not array.flags["C_CONTIGUOUS"]:
        raise ValueError("Expected C-contiguous array")
    return array.ctypes.data_as(ctypes.POINTER(ctypes.c_float))


# ==============================================================================
# DLL Library wrapper
# ==============================================================================


class DefKitDirectLibrary:
    """Wrapper for the DefKitAdv DLL providing direct rod constraint functions."""

    def __init__(self, dll_path: str | None, calling_convention: str):
        """Initialize the DLL wrapper.
        
        Args:
            dll_path: Path to the DLL file, or None to use default.
            calling_convention: Either "stdcall" or "cdecl".
        """
        self.dll_path = self._resolve_dll_path(dll_path)
        self.dll = self._load_library(self.dll_path, calling_convention)
        self._bind_functions()

    @staticmethod
    def _resolve_dll_path(dll_path: str | None) -> str:
        """Resolve the DLL path to an absolute path."""
        if dll_path:
            abs_path = os.path.abspath(dll_path)
            if not os.path.exists(abs_path):
                raise FileNotFoundError(f"DefKit DLL not found: {abs_path}")
            return abs_path
        return "DefKitAdv.dll"

    @staticmethod
    def _load_library(dll_path: str, calling_convention: str):
        """Load the DLL with the specified calling convention."""
        loader = ctypes.WinDLL if calling_convention == "stdcall" else ctypes.CDLL
        try:
            return loader(dll_path)
        except OSError as exc:
            raise RuntimeError(
                "Failed to load DefKit DLL. Provide --dll-path or add it to PATH. "
                f"Path: {dll_path}. Error: {exc}"
            ) from exc

    def _get_function(self, name: str, argtypes, restype=None):
        """Get a required function from the DLL."""
        try:
            fn = getattr(self.dll, name)
        except AttributeError as exc:
            raise RuntimeError(f"Required symbol '{name}' not found in {self.dll_path}") from exc
        fn.argtypes = argtypes
        fn.restype = restype
        return fn

    def _get_optional_function(self, name: str, argtypes, restype=None):
        """Get an optional function from the DLL, returning None if not found."""
        fn = getattr(self.dll, name, None)
        if fn is None:
            return None
        fn.argtypes = argtypes
        fn.restype = restype
        return fn

    def _bind_functions(self):
        """Bind all DLL functions."""
        self.PredictPositions_native = self._get_function(
            "PredictPositions_native",
            [
                ctypes.c_float,
                ctypes.c_float,
                ctypes.c_int,
                ctypes.POINTER(BtVector3),
                ctypes.POINTER(BtVector3),
                ctypes.POINTER(BtVector3),
                ctypes.POINTER(BtVector3),
                ctypes.POINTER(ctypes.c_float),
                ctypes.POINTER(BtVector3),
            ],
        )
        self.Integrate_native = self._get_function(
            "Integrate_native",
            [
                ctypes.c_float,
                ctypes.c_int,
                ctypes.POINTER(BtVector3),
                ctypes.POINTER(BtVector3),
                ctypes.POINTER(BtVector3),
                ctypes.POINTER(ctypes.c_float),
            ],
        )
        self.PredictRotationsPBD = self._get_function(
            "PredictRotationsPBD",
            [
                ctypes.c_float,
                ctypes.c_float,
                ctypes.c_int,
                ctypes.POINTER(BtQuaternion),
                ctypes.POINTER(BtQuaternion),
                ctypes.POINTER(BtVector3),
                ctypes.POINTER(BtVector3),
                ctypes.POINTER(ctypes.c_float),
            ],
        )
        self.IntegrateRotationsPBD = self._get_function(
            "IntegrateRotationsPBD",
            [
                ctypes.c_float,
                ctypes.c_int,
                ctypes.POINTER(BtQuaternion),
                ctypes.POINTER(BtQuaternion),
                ctypes.POINTER(BtQuaternion),
                ctypes.POINTER(BtVector3),
                ctypes.POINTER(ctypes.c_float),
            ],
        )
        self.InitDirectElasticRod = self._get_function(
            "InitDirectElasticRod",
            [
                ctypes.c_int,
                ctypes.POINTER(BtVector3),
                ctypes.POINTER(BtQuaternion),
                ctypes.c_float,
                ctypes.POINTER(ctypes.c_float),
                ctypes.c_float,
                ctypes.c_float,
            ],
            restype=ctypes.c_void_p,
        )
        self.PrepareDirectElasticRodConstraints = self._get_function(
            "PrepareDirectElasticRodConstraints",
            [
                ctypes.c_void_p,
                ctypes.c_int,
                ctypes.c_float,
                ctypes.POINTER(BtVector3),
                ctypes.POINTER(BtVector3),
                ctypes.POINTER(ctypes.c_float),
                ctypes.c_float,
                ctypes.c_float,
            ],
        )
        self.UpdateConstraints_DirectElasticRodConstraintsBanded = self._get_function(
            "UpdateConstraints_DirectElasticRodConstraintsBanded",
            [
                ctypes.c_void_p,
                ctypes.c_int,
                ctypes.POINTER(BtVector3),
                ctypes.POINTER(BtQuaternion),
                ctypes.POINTER(ctypes.c_float),
            ],
        )
        self.ComputeJacobians_DirectElasticRodConstraintsBanded = self._get_function(
            "ComputeJacobians_DirectElasticRodConstraintsBanded",
            [
                ctypes.c_void_p,
                ctypes.c_int,
                ctypes.c_int,
                ctypes.POINTER(BtVector3),
                ctypes.POINTER(BtQuaternion),
                ctypes.POINTER(ctypes.c_float),
            ],
        )
        self.AssembleJMJT_DirectElasticRodConstraintsBanded = self._get_function(
            "AssembleJMJT_DirectElasticRodConstraintsBanded",
            [
                ctypes.c_void_p,
                ctypes.c_int,
                ctypes.c_int,
                ctypes.POINTER(BtVector3),
                ctypes.POINTER(BtQuaternion),
                ctypes.POINTER(ctypes.c_float),
            ],
        )
        self.ProjectJMJT_DirectElasticRodConstraintsBanded = self._get_function(
            "ProjectJMJT_DirectElasticRodConstraintsBanded",
            [
                ctypes.c_void_p,
                ctypes.c_int,
                ctypes.POINTER(BtVector3),
                ctypes.POINTER(BtQuaternion),
                ctypes.POINTER(ctypes.c_float),
                ctypes.POINTER(BtVector3),
                ctypes.POINTER(BtQuaternion),
            ],
        )
        self.UpdateDirectElasticRodConstraints = self._get_optional_function(
            "UpdateDirectElasticRodConstraints",
            [
                ctypes.c_void_p,
                ctypes.c_int,
                ctypes.POINTER(BtVector3),
                ctypes.POINTER(BtQuaternion),
                ctypes.POINTER(ctypes.c_float),
            ],
        )
        self.ProjectDirectElasticRodConstraints = self._get_optional_function(
            "ProjectDirectElasticRodConstraints",
            [
                ctypes.c_void_p,
                ctypes.c_int,
                ctypes.POINTER(BtVector3),
                ctypes.POINTER(BtQuaternion),
                ctypes.POINTER(ctypes.c_float),
                ctypes.POINTER(BtVector3),
                ctypes.POINTER(BtQuaternion),
            ],
        )
        self.ProjectDirectElasticRodConstraintsBanded = self._get_optional_function(
            "ProjectDirectElasticRodConstraintsBanded",
            [
                ctypes.c_void_p,
                ctypes.c_int,
                ctypes.POINTER(BtVector3),
                ctypes.POINTER(BtQuaternion),
                ctypes.POINTER(ctypes.c_float),
                ctypes.POINTER(BtVector3),
                ctypes.POINTER(BtQuaternion),
            ],
        )
        self.DestroyDirectElasticRod = self._get_function(
            "DestroyDirectElasticRod",
            [ctypes.c_void_p],
        )


# ==============================================================================
# DefKit rod state
# ==============================================================================


class DefKitDirectRodState(RodStateBase):
    """Rod state implementation using the DefKitAdv DLL.
    
    This implementation uses the native C++ library for constraint projection
    via FFI (Foreign Function Interface).
    """

    def __init__(
        self,
        lib: DefKitDirectLibrary,
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
        use_banded: bool = False,
    ):
        """Initialize the rod state with DLL backend.
        
        Args:
            lib: DefKitDirectLibrary instance.
            num_points: Number of particles.
            segment_length: Rest length of each segment.
            mass: Mass of each particle.
            particle_height: Initial Z height.
            rod_radius: Rod radius for collision.
            bend_stiffness: Bending stiffness coefficient.
            twist_stiffness: Twist stiffness coefficient.
            rest_bend_d1: Rest curvature in d1 direction.
            rest_bend_d2: Rest curvature in d2 direction.
            rest_twist: Rest twist angle.
            young_modulus: Young's modulus.
            torsion_modulus: Torsion modulus.
            gravity: Gravity vector.
            lock_root_rotation: Whether to lock root rotation.
            use_banded: Whether to use banded solver.
        """
        super().__init__(
            num_points=num_points,
            segment_length=segment_length,
            mass=mass,
            particle_height=particle_height,
            rod_radius=rod_radius,
            bend_stiffness=bend_stiffness,
            twist_stiffness=twist_stiffness,
            rest_bend_d1=rest_bend_d1,
            rest_bend_d2=rest_bend_d2,
            rest_twist=rest_twist,
            young_modulus=young_modulus,
            torsion_modulus=torsion_modulus,
            gravity=gravity,
            lock_root_rotation=lock_root_rotation,
        )

        self.lib = lib
        self.supports_non_banded = self.lib.ProjectDirectElasticRodConstraints is not None
        self.use_banded = use_banded or not self.supports_non_banded
        self.use_iterative_refinement = False
        self.iterative_refinement_iters = 2

        # Initialize native rod
        self.rod_ptr = self.lib.InitDirectElasticRod(
            ctypes.c_int(self.num_points),
            _as_ptr(self.positions, BtVector3),
            _as_ptr(self.orientations, BtQuaternion),
            ctypes.c_float(rod_radius),
            _as_float_ptr(self.rest_lengths),
            ctypes.c_float(self.young_modulus),
            ctypes.c_float(self.torsion_modulus),
        )
        if not self.rod_ptr:
            raise RuntimeError("InitDirectElasticRod returned a null pointer.")
        self._destroyed = False
        atexit.register(self.destroy)

    def destroy(self) -> None:
        """Destroy the native rod and free resources."""
        if self._destroyed:
            return
        if self.rod_ptr:
            self.lib.DestroyDirectElasticRod(self.rod_ptr)
            self.rod_ptr = None
        self._destroyed = True

    def set_solver_mode(self, use_banded: bool) -> None:
        """Set whether to use banded solver.
        
        Args:
            use_banded: Whether to use banded solver.
        """
        self.use_banded = use_banded or not self.supports_non_banded

    def set_iterative_refinement(self, enabled: bool, iters: int = 2) -> None:
        """Enable/disable iterative refinement for banded Cholesky solver.
        
        Args:
            enabled: Whether to use iterative refinement.
            iters: Number of refinement iterations.
        """
        self.use_iterative_refinement = enabled
        self.iterative_refinement_iters = max(1, iters)

    def predict_positions(self, dt: float, linear_damping: float) -> None:
        """Predict positions using the native library."""
        self.lib.PredictPositions_native(
            ctypes.c_float(dt),
            ctypes.c_float(linear_damping),
            ctypes.c_int(self.num_points),
            _as_ptr(self.positions, BtVector3),
            _as_ptr(self.predicted_positions, BtVector3),
            _as_ptr(self.velocities, BtVector3),
            _as_ptr(self.forces, BtVector3),
            _as_float_ptr(self.inv_masses),
            _as_ptr(self.gravity, BtVector3),
        )

    def predict_rotations(self, dt: float, angular_damping: float) -> None:
        """Predict rotations using the native library."""
        self.lib.PredictRotationsPBD(
            ctypes.c_float(dt),
            ctypes.c_float(angular_damping),
            ctypes.c_int(self.num_points),
            _as_ptr(self.orientations, BtQuaternion),
            _as_ptr(self.predicted_orientations, BtQuaternion),
            _as_ptr(self.angular_velocities, BtVector3),
            _as_ptr(self.torques, BtVector3),
            _as_float_ptr(self.quat_inv_masses),
        )

    def prepare_constraints(self, dt: float) -> None:
        """Prepare constraints for projection."""
        self.lib.PrepareDirectElasticRodConstraints(
            self.rod_ptr,
            ctypes.c_int(self.num_edges),
            ctypes.c_float(dt),
            _as_ptr(self.bend_stiffness, BtVector3),
            _as_ptr(self.rest_darboux, BtVector3),
            _as_float_ptr(self.rest_lengths),
            ctypes.c_float(self.young_modulus),
            ctypes.c_float(self.torsion_modulus),
        )

    def update_constraints_banded(self) -> None:
        """Update constraints for banded solver."""
        self.lib.UpdateConstraints_DirectElasticRodConstraintsBanded(
            self.rod_ptr,
            ctypes.c_int(self.num_points),
            _as_ptr(self.predicted_positions, BtVector3),
            _as_ptr(self.predicted_orientations, BtQuaternion),
            _as_float_ptr(self.inv_masses),
        )

    def compute_jacobians_banded(self) -> None:
        """Compute Jacobians for banded solver."""
        self.lib.ComputeJacobians_DirectElasticRodConstraintsBanded(
            self.rod_ptr,
            ctypes.c_int(0),
            ctypes.c_int(self.num_edges),
            _as_ptr(self.predicted_positions, BtVector3),
            _as_ptr(self.predicted_orientations, BtQuaternion),
            _as_float_ptr(self.inv_masses),
        )

    def assemble_jmjt_banded(self) -> None:
        """Assemble JMJT matrix for banded solver."""
        self.lib.AssembleJMJT_DirectElasticRodConstraintsBanded(
            self.rod_ptr,
            ctypes.c_int(0),
            ctypes.c_int(self.num_edges),
            _as_ptr(self.predicted_positions, BtVector3),
            _as_ptr(self.predicted_orientations, BtQuaternion),
            _as_float_ptr(self.inv_masses),
        )

    def project_jmjt_banded(self) -> None:
        """Project constraints using banded solver."""
        self.lib.ProjectJMJT_DirectElasticRodConstraintsBanded(
            self.rod_ptr,
            ctypes.c_int(self.num_points),
            _as_ptr(self.predicted_positions, BtVector3),
            _as_ptr(self.predicted_orientations, BtQuaternion),
            _as_float_ptr(self.inv_masses),
            _as_ptr(self.pos_corrections, BtVector3),
            _as_ptr(self.rot_corrections, BtQuaternion),
        )

    def project_direct(self) -> None:
        """Project constraints using direct solver."""
        self.lib.ProjectDirectElasticRodConstraints(
            self.rod_ptr,
            ctypes.c_int(self.num_points),
            _as_ptr(self.predicted_positions, BtVector3),
            _as_ptr(self.predicted_orientations, BtQuaternion),
            _as_float_ptr(self.inv_masses),
            _as_ptr(self.pos_corrections, BtVector3),
            _as_ptr(self.rot_corrections, BtQuaternion),
        )

    def project_direct_banded(self) -> None:
        """Project constraints using banded direct solver."""
        self.lib.ProjectDirectElasticRodConstraintsBanded(
            self.rod_ptr,
            ctypes.c_int(self.num_points),
            _as_ptr(self.predicted_positions, BtVector3),
            _as_ptr(self.predicted_orientations, BtQuaternion),
            _as_float_ptr(self.inv_masses),
            _as_ptr(self.pos_corrections, BtVector3),
            _as_ptr(self.rot_corrections, BtQuaternion),
        )

    def integrate_positions(self, dt: float) -> None:
        """Integrate positions after constraint projection."""
        self.lib.Integrate_native(
            ctypes.c_float(dt),
            ctypes.c_int(self.num_points),
            _as_ptr(self.positions, BtVector3),
            _as_ptr(self.predicted_positions, BtVector3),
            _as_ptr(self.velocities, BtVector3),
            _as_float_ptr(self.inv_masses),
        )

    def integrate_rotations(self, dt: float) -> None:
        """Integrate rotations after constraint projection."""
        self.lib.IntegrateRotationsPBD(
            ctypes.c_float(dt),
            ctypes.c_int(self.num_points),
            _as_ptr(self.orientations, BtQuaternion),
            _as_ptr(self.predicted_orientations, BtQuaternion),
            _as_ptr(self.prev_orientations, BtQuaternion),
            _as_ptr(self.angular_velocities, BtVector3),
            _as_float_ptr(self.quat_inv_masses),
        )

    def apply_floor_collisions(self, floor_z: float, restitution: float = 0.0) -> None:
        """Apply floor collision constraints."""
        if self.num_points == 0:
            return
        min_z = np.float32(floor_z + self.rod_radius)
        for i in range(self.num_points):
            z = self.positions[i, 2]
            if z < min_z:
                self.positions[i, 2] = min_z
                self.predicted_positions[i, 2] = min_z
                if self.velocities[i, 2] < 0.0:
                    self.velocities[i, 2] = -np.float32(restitution) * self.velocities[i, 2]

    def reset(self) -> None:
        """Reset the rod to its initial state."""
        self.positions[:] = self._initial_positions
        self.predicted_positions[:] = self._initial_positions
        self.velocities.fill(0.0)
        self.forces.fill(0.0)

        self.orientations[:] = self._initial_orientations
        self.predicted_orientations[:] = self._initial_orientations
        self.prev_orientations[:] = self._initial_orientations
        self.angular_velocities.fill(0.0)
        self.torques.fill(0.0)

    def step(self, dt: float, linear_damping: float, angular_damping: float) -> None:
        """Advance the simulation by one time step."""
        self.predict_positions(dt, linear_damping)
        self.predict_rotations(dt, angular_damping)
        self.prepare_constraints(dt)

        if self.use_banded or not self.supports_non_banded:
            self.update_constraints_banded()
            self.compute_jacobians_banded()
            self.assemble_jmjt_banded()
            self.project_jmjt_banded()
        else:
            self.project_direct()

        self.integrate_positions(dt)
        self.integrate_rotations(dt)


__all__ = [
    "BtQuaternion",
    "BtVector3",
    "DefKitDirectLibrary",
    "DefKitDirectRodState",
]
