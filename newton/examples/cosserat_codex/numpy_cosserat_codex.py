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

"""Direct Cosserat rod: native reference + NumPy port scaffold.

This example runs two direct rods side-by-side:
- Reference rod: full native DefKitAdv.dll pipeline.
- NumPy rod: hybrid pipeline where steps can be replaced by NumPy.

Command:
    uv run python newton/examples/cosserat_codex/numpy_cosserat_codex.py --dll-path "C:\\path\\to\\DefKitAdv.dll"
"""

from __future__ import annotations

import atexit
import ctypes
import math
import os

import numpy as np
import warp as wp

import newton
import newton.examples


class BtVector3(ctypes.Structure):
    _fields_ = [
        ("x", ctypes.c_float),
        ("y", ctypes.c_float),
        ("z", ctypes.c_float),
        ("w", ctypes.c_float),
    ]


class BtQuaternion(ctypes.Structure):
    _fields_ = [
        ("x", ctypes.c_float),
        ("y", ctypes.c_float),
        ("z", ctypes.c_float),
        ("w", ctypes.c_float),
    ]


def _as_ptr(array: np.ndarray, ctype):
    if array.dtype != np.float32:
        raise TypeError(f"Expected float32 array, got {array.dtype}")
    if not array.flags["C_CONTIGUOUS"]:
        raise ValueError("Expected C-contiguous array")
    return array.ctypes.data_as(ctypes.POINTER(ctype))


def _as_float_ptr(array: np.ndarray):
    if array.dtype != np.float32:
        raise TypeError(f"Expected float32 array, got {array.dtype}")
    if not array.flags["C_CONTIGUOUS"]:
        raise ValueError("Expected C-contiguous array")
    return array.ctypes.data_as(ctypes.POINTER(ctypes.c_float))


def _quat_from_axis_angle(axis: np.ndarray, angle: float) -> np.ndarray:
    axis = np.asarray(axis, dtype=np.float32)
    norm = np.linalg.norm(axis)
    if norm < 1.0e-8:
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    axis = axis / norm
    half = angle * 0.5
    s = math.sin(half)
    return np.array([axis[0] * s, axis[1] * s, axis[2] * s, math.cos(half)], dtype=np.float32)


def _quat_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
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


class DefKitDirectLibrary:
    def __init__(self, dll_path: str | None, calling_convention: str):
        self.dll_path = self._resolve_dll_path(dll_path)
        self.dll = self._load_library(self.dll_path, calling_convention)
        self._bind_functions()

    @staticmethod
    def _resolve_dll_path(dll_path: str | None) -> str:
        if dll_path:
            abs_path = os.path.abspath(dll_path)
            if not os.path.exists(abs_path):
                raise FileNotFoundError(f"DefKit DLL not found: {abs_path}")
            return abs_path
        return "DefKitAdv.dll"

    @staticmethod
    def _load_library(dll_path: str, calling_convention: str):
        loader = ctypes.WinDLL if calling_convention == "stdcall" else ctypes.CDLL
        try:
            return loader(dll_path)
        except OSError as exc:
            raise RuntimeError(
                "Failed to load DefKit DLL. Provide --dll-path or add it to PATH. "
                f"Path: {dll_path}. Error: {exc}"
            ) from exc

    def _get_function(self, name: str, argtypes, restype=None):
        try:
            fn = getattr(self.dll, name)
        except AttributeError as exc:
            raise RuntimeError(f"Required symbol '{name}' not found in {self.dll_path}") from exc
        fn.argtypes = argtypes
        fn.restype = restype
        return fn

    def _get_optional_function(self, name: str, argtypes, restype=None):
        fn = getattr(self.dll, name, None)
        if fn is None:
            return None
        fn.argtypes = argtypes
        fn.restype = restype
        return fn

    def _bind_functions(self):
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
        self.DestroyDirectElasticRod = self._get_function(
            "DestroyDirectElasticRod",
            [ctypes.c_void_p],
        )


class DefKitDirectRodState:
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
        use_banded: bool,
    ):
        self.lib = lib
        self.num_points = num_points
        self.num_edges = max(0, num_points - 1)
        self.segment_length = segment_length
        self.young_modulus = young_modulus
        self.torsion_modulus = torsion_modulus
        self.rod_radius = rod_radius
        self.supports_non_banded = self.lib.ProjectDirectElasticRodConstraints is not None
        self.use_banded = use_banded or not self.supports_non_banded

        self.positions = np.zeros((num_points, 4), dtype=np.float32)
        self.predicted_positions = np.zeros((num_points, 4), dtype=np.float32)
        self.velocities = np.zeros((num_points, 4), dtype=np.float32)
        self.forces = np.zeros((num_points, 4), dtype=np.float32)

        for i in range(num_points):
            self.positions[i, 0] = i * segment_length
            self.positions[i, 2] = particle_height

        self.predicted_positions[:] = self.positions

        q_align = _quat_from_axis_angle(np.array([0.0, 1.0, 0.0], dtype=np.float32), math.pi * 0.5)
        self.orientations = np.tile(q_align, (num_points, 1)).astype(np.float32)
        self.predicted_orientations = self.orientations.copy()
        self.prev_orientations = self.orientations.copy()

        self.angular_velocities = np.zeros((num_points, 4), dtype=np.float32)
        self.torques = np.zeros((num_points, 4), dtype=np.float32)

        inv_mass_value = 0.0 if mass == 0.0 else 1.0 / mass
        self.inv_masses = np.full(num_points, inv_mass_value, dtype=np.float32)
        self.inv_masses[0] = 0.0

        self.quat_inv_masses = np.full(num_points, 1.0, dtype=np.float32)
        if lock_root_rotation:
            self.quat_inv_masses[0] = 0.0

        self.rest_lengths = np.full(self.num_edges, segment_length, dtype=np.float32)

        self.rest_darboux = np.zeros((self.num_edges, 4), dtype=np.float32)
        self.set_rest_darboux(rest_bend_d1, rest_bend_d2, rest_twist)

        self.bend_stiffness = np.zeros((self.num_edges, 4), dtype=np.float32)
        self.set_bend_stiffness(bend_stiffness, twist_stiffness)

        self.pos_corrections = np.zeros((num_points, 4), dtype=np.float32)
        self.rot_corrections = np.zeros((num_points, 4), dtype=np.float32)

        self.gravity = np.zeros((1, 4), dtype=np.float32)
        self.set_gravity(gravity)

        self._initial_positions = self.positions.copy()
        self._initial_orientations = self.orientations.copy()

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

    def destroy(self):
        if self._destroyed:
            return
        if self.rod_ptr:
            self.lib.DestroyDirectElasticRod(self.rod_ptr)
            self.rod_ptr = None
        self._destroyed = True

    def set_gravity(self, gravity: np.ndarray):
        self.gravity[0, 0:3] = gravity.astype(np.float32)

    def set_bend_stiffness(self, bend_stiffness: float, twist_stiffness: float):
        self.bend_stiffness[:, 0] = bend_stiffness
        self.bend_stiffness[:, 1] = bend_stiffness
        self.bend_stiffness[:, 2] = twist_stiffness

    def set_rest_darboux(self, rest_bend_d1: float, rest_bend_d2: float, rest_twist: float):
        self.rest_darboux[:, 0] = rest_bend_d1
        self.rest_darboux[:, 1] = rest_bend_d2
        self.rest_darboux[:, 2] = rest_twist

    def set_solver_mode(self, use_banded: bool):
        self.use_banded = use_banded or not self.supports_non_banded

    def reset(self):
        self.positions[:] = self._initial_positions
        self.predicted_positions[:] = self._initial_positions
        self.velocities.fill(0.0)
        self.forces.fill(0.0)

        self.orientations[:] = self._initial_orientations
        self.predicted_orientations[:] = self._initial_orientations
        self.prev_orientations[:] = self._initial_orientations
        self.angular_velocities.fill(0.0)
        self.torques.fill(0.0)

    def predict_positions(self, dt: float, linear_damping: float):
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

    def predict_rotations(self, dt: float, angular_damping: float):
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

    def prepare_constraints(self, dt: float):
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

    def update_constraints_banded(self):
        self.lib.UpdateConstraints_DirectElasticRodConstraintsBanded(
            self.rod_ptr,
            ctypes.c_int(self.num_points),
            _as_ptr(self.predicted_positions, BtVector3),
            _as_ptr(self.predicted_orientations, BtQuaternion),
            _as_float_ptr(self.inv_masses),
        )

    def compute_jacobians_banded(self):
        self.lib.ComputeJacobians_DirectElasticRodConstraintsBanded(
            self.rod_ptr,
            ctypes.c_int(0),
            ctypes.c_int(self.num_edges),
            _as_ptr(self.predicted_positions, BtVector3),
            _as_ptr(self.predicted_orientations, BtQuaternion),
            _as_float_ptr(self.inv_masses),
        )

    def assemble_jmjt_banded(self):
        self.lib.AssembleJMJT_DirectElasticRodConstraintsBanded(
            self.rod_ptr,
            ctypes.c_int(0),
            ctypes.c_int(self.num_edges),
            _as_ptr(self.predicted_positions, BtVector3),
            _as_ptr(self.predicted_orientations, BtQuaternion),
            _as_float_ptr(self.inv_masses),
        )

    def project_jmjt_banded(self):
        self.lib.ProjectJMJT_DirectElasticRodConstraintsBanded(
            self.rod_ptr,
            ctypes.c_int(self.num_points),
            _as_ptr(self.predicted_positions, BtVector3),
            _as_ptr(self.predicted_orientations, BtQuaternion),
            _as_float_ptr(self.inv_masses),
            _as_ptr(self.pos_corrections, BtVector3),
            _as_ptr(self.rot_corrections, BtQuaternion),
        )

    def project_direct(self):
        self.lib.ProjectDirectElasticRodConstraints(
            self.rod_ptr,
            ctypes.c_int(self.num_points),
            _as_ptr(self.predicted_positions, BtVector3),
            _as_ptr(self.predicted_orientations, BtQuaternion),
            _as_float_ptr(self.inv_masses),
            _as_ptr(self.pos_corrections, BtVector3),
            _as_ptr(self.rot_corrections, BtQuaternion),
        )

    def integrate_positions(self, dt: float):
        self.lib.Integrate_native(
            ctypes.c_float(dt),
            ctypes.c_int(self.num_points),
            _as_ptr(self.positions, BtVector3),
            _as_ptr(self.predicted_positions, BtVector3),
            _as_ptr(self.velocities, BtVector3),
            _as_float_ptr(self.inv_masses),
        )

    def integrate_rotations(self, dt: float):
        self.lib.IntegrateRotationsPBD(
            ctypes.c_float(dt),
            ctypes.c_int(self.num_points),
            _as_ptr(self.orientations, BtQuaternion),
            _as_ptr(self.predicted_orientations, BtQuaternion),
            _as_ptr(self.prev_orientations, BtQuaternion),
            _as_ptr(self.angular_velocities, BtVector3),
            _as_float_ptr(self.quat_inv_masses),
        )

    def step(
        self,
        dt: float,
        linear_damping: float,
        angular_damping: float,
    ):
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


class NumpyDirectRodState(DefKitDirectRodState):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.numpy_available = {
            "predict_positions": True,
            "integrate_positions": True,
            "predict_rotations": True,
            "integrate_rotations": True,
            "prepare_constraints": True,
            "update_constraints_banded": True,
            "compute_jacobians_banded": False,
            "assemble_jmjt_banded": False,
            "project_jmjt_banded": False,
            "project_direct": False,
        }
        self.numpy_enabled = {
            "predict_positions": True,
            "integrate_positions": True,
            "predict_rotations": True,
            "integrate_rotations": True,
            "prepare_constraints": True,
            "update_constraints_banded": True,
            "compute_jacobians_banded": False,
            "assemble_jmjt_banded": False,
            "project_jmjt_banded": False,
            "project_direct": False,
        }
        self.lambdas = np.zeros((self.num_edges, 6), dtype=np.float32)
        self.compliance = np.zeros((self.num_edges, 6), dtype=np.float32)
        self.constraint_values = np.zeros((self.num_edges, 6), dtype=np.float32)
        self.current_rest_lengths = self.rest_lengths.copy()
        self.current_rest_darboux = np.zeros((self.num_edges, 3), dtype=np.float32)
        self._update_cross_section_properties()

    def set_numpy_override(self, step_name: str, enabled: bool):
        if step_name not in self.numpy_available:
            raise ValueError(f"Unknown step: {step_name}")
        if not self.numpy_available[step_name]:
            self.numpy_enabled[step_name] = False
            return False
        self.numpy_enabled[step_name] = enabled
        return True

    def predict_positions(self, dt: float, linear_damping: float):
        if self.numpy_enabled["predict_positions"]:
            self._numpy_predict_positions(dt, linear_damping)
        else:
            super().predict_positions(dt, linear_damping)

    def integrate_positions(self, dt: float):
        if self.numpy_enabled["integrate_positions"]:
            self._numpy_integrate_positions(dt)
        else:
            super().integrate_positions(dt)

    def predict_rotations(self, dt: float, angular_damping: float):
        if self.numpy_enabled["predict_rotations"]:
            self._numpy_predict_rotations(dt, angular_damping)
        else:
            super().predict_rotations(dt, angular_damping)

    def integrate_rotations(self, dt: float):
        if self.numpy_enabled["integrate_rotations"]:
            self._numpy_integrate_rotations(dt)
        else:
            super().integrate_rotations(dt)

    def prepare_constraints(self, dt: float):
        if self.numpy_enabled["prepare_constraints"]:
            self._numpy_prepare_constraints(dt)
            if self._requires_native_constraint_pipeline():
                super().prepare_constraints(dt)
        else:
            super().prepare_constraints(dt)

    def update_constraints_banded(self):
        if self.numpy_enabled["update_constraints_banded"]:
            self._numpy_update_constraints_banded()
            if self._requires_native_constraint_pipeline():
                super().update_constraints_banded()
        else:
            super().update_constraints_banded()

    def compute_jacobians_banded(self):
        if self.numpy_enabled["compute_jacobians_banded"]:
            raise NotImplementedError("NumPy ComputeJacobians_Banded is not implemented yet.")
        super().compute_jacobians_banded()

    def assemble_jmjt_banded(self):
        if self.numpy_enabled["assemble_jmjt_banded"]:
            raise NotImplementedError("NumPy AssembleJMJT_Banded is not implemented yet.")
        super().assemble_jmjt_banded()

    def project_jmjt_banded(self):
        if self.numpy_enabled["project_jmjt_banded"]:
            raise NotImplementedError("NumPy ProjectJMJT_Banded is not implemented yet.")
        super().project_jmjt_banded()

    def project_direct(self):
        if self.numpy_enabled["project_direct"]:
            raise NotImplementedError("NumPy ProjectDirectElasticRodConstraints is not implemented yet.")
        super().project_direct()

    def _numpy_predict_positions(self, dt: float, linear_damping: float):
        dt = np.float32(dt)
        damping = np.float32(linear_damping)
        damp = np.float32(1.0) - damping

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

    def _numpy_integrate_positions(self, dt: float):
        dt_inv = np.float32(1.0) / np.float32(dt)
        positions = self.positions[:, 0:3]
        predicted = self.predicted_positions[:, 0:3]
        velocities = self.velocities[:, 0:3]

        dynamic_mask = self.inv_masses != 0.0
        velocities[dynamic_mask] = (predicted[dynamic_mask] - positions[dynamic_mask]) * dt_inv
        positions[dynamic_mask] = predicted[dynamic_mask]

        self.positions[:, 3] = 0.0
        self.velocities[:, 3] = 0.0

    def _numpy_predict_rotations(self, dt: float, angular_damping: float):
        dt = np.float32(dt)
        half_dt = np.float32(0.5) * dt
        damp = np.float32(1.0) - np.float32(angular_damping)

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

    def _numpy_integrate_rotations(self, dt: float):
        dt_inv2 = np.float32(2.0) / np.float32(dt)
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

    def _update_cross_section_properties(self):
        radius = np.float32(self.rod_radius)
        self.cross_section_area = np.float32(np.pi) * radius * radius
        self.second_moment_area = np.float32(np.pi) * radius**4 / np.float32(4.0)
        self.polar_moment = np.float32(np.pi) * radius**4 / np.float32(2.0)

    def _requires_native_constraint_pipeline(self) -> bool:
        if not self.use_banded:
            return not self.numpy_enabled.get("project_direct", False)
        return not (
            self.numpy_enabled.get("update_constraints_banded", False)
            and self.numpy_enabled.get("compute_jacobians_banded", False)
            and self.numpy_enabled.get("assemble_jmjt_banded", False)
            and self.numpy_enabled.get("project_jmjt_banded", False)
        )

    def _numpy_prepare_constraints(self, dt: float):
        self.lambdas.fill(0.0)

        self.current_rest_lengths = self.rest_lengths.copy()
        self.current_rest_darboux[:, 0] = self.rest_darboux[:, 0]
        self.current_rest_darboux[:, 1] = self.rest_darboux[:, 1]
        self.current_rest_darboux[:, 2] = self.rest_darboux[:, 2]

        dt2 = np.float32(dt * dt)
        eps = np.float32(1.0e-10)

        E = np.float32(self.young_modulus)
        G = np.float32(self.torsion_modulus)
        A = self.cross_section_area
        I = self.second_moment_area
        J = self.polar_moment

        L = self.current_rest_lengths.astype(np.float32)
        inv_L = np.where(L > 0.0, np.float32(1.0) / L, np.float32(0.0))

        k_stretch = E * A * inv_L
        k_shear = k_stretch
        k_bend1 = E * I * inv_L * self.bend_stiffness[:, 0]
        k_bend2 = E * I * inv_L * self.bend_stiffness[:, 1]
        k_twist = G * J * inv_L * self.bend_stiffness[:, 2]

        self.compliance[:, 0] = np.float32(1.0) / (k_stretch * dt2 + eps)
        self.compliance[:, 1] = np.float32(1.0) / (k_shear * dt2 + eps)
        self.compliance[:, 2] = np.float32(1.0) / (k_shear * dt2 + eps)
        self.compliance[:, 3] = np.float32(1.0) / (k_bend1 * dt2 + eps)
        self.compliance[:, 4] = np.float32(1.0) / (k_bend2 * dt2 + eps)
        self.compliance[:, 5] = np.float32(1.0) / (k_twist * dt2 + eps)

    def _numpy_update_constraints_banded(self):
        positions = self.predicted_positions[:, 0:3]
        orientations = self.predicted_orientations
        rest_lengths = self.current_rest_lengths
        rest_darboux = self.current_rest_darboux

        for i in range(self.num_edges):
            p0 = positions[i]
            p1 = positions[i + 1]
            q0 = orientations[i]
            q1 = orientations[i + 1]

            edge = p1 - p0
            L = rest_lengths[i]
            if L <= 1.0e-8:
                L = np.float32(1.0e-8)

            d1 = self._numpy_quat_rotate_vector(q0, np.array([1.0, 0.0, 0.0], dtype=np.float32))
            d2 = self._numpy_quat_rotate_vector(q0, np.array([0.0, 1.0, 0.0], dtype=np.float32))
            d3 = self._numpy_quat_rotate_vector(q0, np.array([0.0, 0.0, 1.0], dtype=np.float32))

            edge_error = edge - d3 * L

            self.constraint_values[i, 0] = np.dot(edge_error, d3)
            self.constraint_values[i, 1] = np.dot(edge_error, d1)
            self.constraint_values[i, 2] = np.dot(edge_error, d2)

            q0_inv = self._numpy_quat_conjugate(q0)
            q_rel = self._numpy_quat_mul_single(q0_inv, q1)
            omega = np.float32(2.0) * q_rel[:3] / L
            darboux_error = omega - rest_darboux[i]
            self.constraint_values[i, 3:6] = darboux_error

    @staticmethod
    def _numpy_quat_rotate_vector(q: np.ndarray, v: np.ndarray) -> np.ndarray:
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
    def _numpy_quat_conjugate(q: np.ndarray) -> np.ndarray:
        return np.array([-q[0], -q[1], -q[2], q[3]], dtype=np.float32)

    @staticmethod
    def _numpy_quat_mul_single(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
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
    def _numpy_quat_mul(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
        x1, y1, z1, w1 = q1[:, 0], q1[:, 1], q1[:, 2], q1[:, 3]
        x2, y2, z2, w2 = q2[:, 0], q2[:, 1], q2[:, 2], q2[:, 3]
        return np.stack(
            [
                w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
                w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            ],
            axis=1,
        ).astype(np.float32)

    @staticmethod
    def _numpy_quat_normalize(quats: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
        norms = np.linalg.norm(quats, axis=1, keepdims=True)
        norms = np.where(norms < 1.0e-8, 1.0, norms)
        normalized = quats / norms
        if mask is None:
            return normalized
        result = quats.copy()
        result[mask] = normalized[mask]
        return result

class Example:
    def __init__(self, viewer, args=None):
        self.viewer = viewer
        self.args = args

        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_time = 0.0

        self.substeps = args.substeps
        self.linear_damping = args.linear_damping
        self.angular_damping = args.angular_damping
        self.bend_stiffness = args.bend_stiffness
        self.twist_stiffness = args.twist_stiffness
        self.rest_bend_d1 = args.rest_bend_d1
        self.rest_bend_d2 = args.rest_bend_d2
        self.rest_twist = args.rest_twist
        self.young_modulus_scale = args.young_modulus / 1.0e6
        self.torsion_modulus_scale = args.torsion_modulus / 1.0e6
        self.use_banded = args.use_banded
        self.compare_offset = args.compare_offset
        half_offset = 0.5 * self.compare_offset
        self.ref_offset = np.array([0.0, -half_offset, 0.0], dtype=np.float32)
        self.numpy_offset = np.array([0.0, half_offset, 0.0], dtype=np.float32)

        self.base_gravity = np.array(args.gravity, dtype=np.float32)
        self.gravity_enabled = True
        self.gravity_scale = 1.0

        self.show_segments = True
        self.show_directors = False
        self.director_scale = 0.1

        self.root_move_speed = 1.0
        self.root_rotate_speed = 1.0
        self.root_rotation = 0.0

        self._gravity_key_was_down = False
        self._reset_key_was_down = False

        self.lib = DefKitDirectLibrary(args.dll_path, args.calling_convention)
        self.supports_non_banded = self.lib.ProjectDirectElasticRodConstraints is not None
        if not self.supports_non_banded:
            self.use_banded = True

        rod_radius = args.rod_radius if args.rod_radius is not None else args.particle_radius
        self.ref_rod = DefKitDirectRodState(
            lib=self.lib,
            num_points=args.num_points,
            segment_length=args.segment_length,
            mass=args.particle_mass,
            particle_height=args.particle_height,
            rod_radius=rod_radius,
            bend_stiffness=self.bend_stiffness,
            twist_stiffness=self.twist_stiffness,
            rest_bend_d1=self.rest_bend_d1,
            rest_bend_d2=self.rest_bend_d2,
            rest_twist=self.rest_twist,
            young_modulus=args.young_modulus,
            torsion_modulus=args.torsion_modulus,
            gravity=self.base_gravity,
            lock_root_rotation=args.lock_root_rotation,
            use_banded=self.use_banded,
        )
        self.numpy_rod = NumpyDirectRodState(
            lib=self.lib,
            num_points=args.num_points,
            segment_length=args.segment_length,
            mass=args.particle_mass,
            particle_height=args.particle_height,
            rod_radius=rod_radius,
            bend_stiffness=self.bend_stiffness,
            twist_stiffness=self.twist_stiffness,
            rest_bend_d1=self.rest_bend_d1,
            rest_bend_d2=self.rest_bend_d2,
            rest_twist=self.rest_twist,
            young_modulus=args.young_modulus,
            torsion_modulus=args.torsion_modulus,
            gravity=self.base_gravity,
            lock_root_rotation=args.lock_root_rotation,
            use_banded=self.use_banded,
        )

        builder = newton.ModelBuilder()
        builder.add_ground_plane()

        for i in range(args.num_points):
            mass = 0.0 if i == 0 else args.particle_mass
            ref_pos = tuple(self.ref_rod.positions[i, 0:3] + self.ref_offset)
            builder.add_particle(pos=ref_pos, vel=(0.0, 0.0, 0.0), mass=mass, radius=args.particle_radius)
        for i in range(args.num_points):
            mass = 0.0 if i == 0 else args.particle_mass
            numpy_pos = tuple(self.numpy_rod.positions[i, 0:3] + self.numpy_offset)
            builder.add_particle(pos=numpy_pos, vel=(0.0, 0.0, 0.0), mass=mass, radius=args.particle_radius)

        self.model = builder.finalize()
        self.state = self.model.state()

        self.viewer.set_model(self.model)
        self.viewer.show_particles = True

        self._ref_segment_colors = np.tile(np.array([0.2, 0.6, 1.0], dtype=np.float32), (args.num_points - 1, 1))
        self._numpy_segment_colors = np.tile(np.array([1.0, 0.6, 0.2], dtype=np.float32), (args.num_points - 1, 1))
        self._numpy_step_labels = {
            "predict_positions": "Predict Positions",
            "integrate_positions": "Integrate Positions",
            "predict_rotations": "Predict Rotations",
            "integrate_rotations": "Integrate Rotations",
            "prepare_constraints": "Prepare Constraints",
            "update_constraints_banded": "Update Constraints (banded)",
            "compute_jacobians_banded": "Compute Jacobians (banded)",
            "assemble_jmjt_banded": "Assemble JMJT (banded)",
            "project_jmjt_banded": "Project JMJT (banded)",
            "project_direct": "Project Direct (non-banded)",
        }

        self._sync_state_from_rods()
        self._update_gravity()
        self._ref_root_base_orientation = self.ref_rod.orientations[0].copy()
        self._numpy_root_base_orientation = self.numpy_rod.orientations[0].copy()

    def __del__(self):
        if hasattr(self, "ref_rod"):
            self.ref_rod.destroy()
        if hasattr(self, "numpy_rod"):
            self.numpy_rod.destroy()

    def _update_gravity(self):
        if self.gravity_enabled:
            gravity = self.base_gravity * self.gravity_scale
        else:
            gravity = np.zeros(3, dtype=np.float32)
        self.ref_rod.set_gravity(gravity)
        self.numpy_rod.set_gravity(gravity)

    def _sync_state_from_rods(self):
        ref_positions = self.ref_rod.positions[:, 0:3].astype(np.float32) + self.ref_offset
        numpy_positions = self.numpy_rod.positions[:, 0:3].astype(np.float32) + self.numpy_offset
        ref_velocities = self.ref_rod.velocities[:, 0:3].astype(np.float32)
        numpy_velocities = self.numpy_rod.velocities[:, 0:3].astype(np.float32)

        positions = np.vstack([ref_positions, numpy_positions])
        velocities = np.vstack([ref_velocities, numpy_velocities])

        self.state.particle_q.assign(wp.array(positions, dtype=wp.vec3, device=self.model.device))
        self.state.particle_qd.assign(wp.array(velocities, dtype=wp.vec3, device=self.model.device))

    def _handle_keyboard_input(self):
        if not hasattr(self.viewer, "is_key_down"):
            return

        try:
            import pyglet.window.key as key
        except ImportError:
            return

        g_down = self.viewer.is_key_down(key.G)
        if g_down and not self._gravity_key_was_down:
            self.gravity_enabled = not self.gravity_enabled
            self._update_gravity()
        self._gravity_key_was_down = g_down

        r_down = self.viewer.is_key_down(key.R)
        if r_down and not self._reset_key_was_down:
            self.ref_rod.reset()
            self.numpy_rod.reset()
            self.root_rotation = 0.0
            self._apply_root_rotation()
            self.sim_time = 0.0
            self._sync_state_from_rods()
        self._reset_key_was_down = r_down

        dx = 0.0
        dy = 0.0
        dz = 0.0

        if self.viewer.is_key_down(key.NUM_6):
            dx += self.root_move_speed * self.frame_dt
        if self.viewer.is_key_down(key.NUM_4):
            dx -= self.root_move_speed * self.frame_dt
        if self.viewer.is_key_down(key.NUM_8):
            dy += self.root_move_speed * self.frame_dt
        if self.viewer.is_key_down(key.NUM_2):
            dy -= self.root_move_speed * self.frame_dt
        if self.viewer.is_key_down(key.NUM_9):
            dz += self.root_move_speed * self.frame_dt
        if self.viewer.is_key_down(key.NUM_3):
            dz -= self.root_move_speed * self.frame_dt

        rotation_changed = False
        if self.viewer.is_key_down(key.NUM_7):
            self.root_rotation += self.root_rotate_speed * self.frame_dt
            rotation_changed = True
        if self.viewer.is_key_down(key.NUM_1):
            self.root_rotation -= self.root_rotate_speed * self.frame_dt
            rotation_changed = True

        if dx != 0.0 or dy != 0.0 or dz != 0.0:
            self._apply_root_translation(dx, dy, dz)
        if rotation_changed:
            self._apply_root_rotation()

    def step(self):
        self._handle_keyboard_input()

        sub_dt = self.frame_dt / max(self.substeps, 1)
        for _ in range(self.substeps):
            self.ref_rod.step(sub_dt, self.linear_damping, self.angular_damping)
            self.numpy_rod.step(sub_dt, self.linear_damping, self.angular_damping)

        self._sync_state_from_rods()
        self.sim_time += self.frame_dt

    def _apply_root_translation(self, dx: float, dy: float, dz: float):
        delta = np.array([dx, dy, dz], dtype=np.float32)
        for rod in (self.ref_rod, self.numpy_rod):
            pos = rod.positions[0, 0:3]
            new_pos = pos + delta
            rod.positions[0, 0:3] = new_pos
            rod.predicted_positions[0, 0:3] = new_pos
            rod.velocities[0, 0:3] = 0.0

    def _apply_root_rotation(self):
        q_twist = _quat_from_axis_angle(np.array([0.0, 0.0, 1.0], dtype=np.float32), self.root_rotation)
        q_ref = _quat_multiply(self._ref_root_base_orientation, q_twist)
        q_numpy = _quat_multiply(self._numpy_root_base_orientation, q_twist)
        self.ref_rod.orientations[0] = q_ref
        self.ref_rod.predicted_orientations[0] = q_ref
        self.ref_rod.prev_orientations[0] = q_ref
        self.numpy_rod.orientations[0] = q_numpy
        self.numpy_rod.predicted_orientations[0] = q_numpy
        self.numpy_rod.prev_orientations[0] = q_numpy

    def _rotate_vector_by_quaternion(self, v: np.ndarray, q: np.ndarray) -> np.ndarray:
        x, y, z, w = q
        vx, vy, vz = v

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

    def _build_director_lines(self, rod: DefKitDirectRodState, offset: np.ndarray):
        num_edges = rod.num_points - 1
        positions = rod.positions[:, 0:3] + offset
        orientations = rod.orientations

        starts = np.zeros((num_edges * 3, 3), dtype=np.float32)
        ends = np.zeros((num_edges * 3, 3), dtype=np.float32)
        colors = np.zeros((num_edges * 3, 3), dtype=np.float32)

        for i in range(num_edges):
            midpoint = 0.5 * (positions[i] + positions[i + 1])
            q = orientations[i]

            d1 = self._rotate_vector_by_quaternion(np.array([1.0, 0.0, 0.0], dtype=np.float32), q)
            d2 = self._rotate_vector_by_quaternion(np.array([0.0, 1.0, 0.0], dtype=np.float32), q)
            d3 = self._rotate_vector_by_quaternion(np.array([0.0, 0.0, 1.0], dtype=np.float32), q)

            base = i * 3
            starts[base] = midpoint
            ends[base] = midpoint + d1 * self.director_scale
            colors[base] = [1.0, 0.0, 0.0]

            starts[base + 1] = midpoint
            ends[base + 1] = midpoint + d2 * self.director_scale
            colors[base + 1] = [0.0, 1.0, 0.0]

            starts[base + 2] = midpoint
            ends[base + 2] = midpoint + d3 * self.director_scale
            colors[base + 2] = [0.0, 0.0, 1.0]

        return starts, ends, colors

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state)

        if self.show_segments:
            ref_starts = (self.ref_rod.positions[:-1, 0:3] + self.ref_offset).astype(np.float32)
            ref_ends = (self.ref_rod.positions[1:, 0:3] + self.ref_offset).astype(np.float32)
            numpy_starts = (self.numpy_rod.positions[:-1, 0:3] + self.numpy_offset).astype(np.float32)
            numpy_ends = (self.numpy_rod.positions[1:, 0:3] + self.numpy_offset).astype(np.float32)
            self.viewer.log_lines(
                "/rod_reference",
                wp.array(ref_starts, dtype=wp.vec3, device=self.model.device),
                wp.array(ref_ends, dtype=wp.vec3, device=self.model.device),
                wp.array(self._ref_segment_colors, dtype=wp.vec3, device=self.model.device),
            )
            self.viewer.log_lines(
                "/rod_numpy",
                wp.array(numpy_starts, dtype=wp.vec3, device=self.model.device),
                wp.array(numpy_ends, dtype=wp.vec3, device=self.model.device),
                wp.array(self._numpy_segment_colors, dtype=wp.vec3, device=self.model.device),
            )
        else:
            self.viewer.log_lines("/rod_reference", None, None, None)
            self.viewer.log_lines("/rod_numpy", None, None, None)

        if self.show_directors:
            ref_starts, ref_ends, ref_colors = self._build_director_lines(self.ref_rod, self.ref_offset)
            numpy_starts, numpy_ends, numpy_colors = self._build_director_lines(self.numpy_rod, self.numpy_offset)
            self.viewer.log_lines(
                "/directors_reference",
                wp.array(ref_starts, dtype=wp.vec3, device=self.model.device),
                wp.array(ref_ends, dtype=wp.vec3, device=self.model.device),
                wp.array(ref_colors, dtype=wp.vec3, device=self.model.device),
            )
            self.viewer.log_lines(
                "/directors_numpy",
                wp.array(numpy_starts, dtype=wp.vec3, device=self.model.device),
                wp.array(numpy_ends, dtype=wp.vec3, device=self.model.device),
                wp.array(numpy_colors, dtype=wp.vec3, device=self.model.device),
            )
        else:
            self.viewer.log_lines("/directors_reference", None, None, None)
            self.viewer.log_lines("/directors_numpy", None, None, None)

        self.viewer.end_frame()

    def gui(self, ui):
        ui.text("Direct Cosserat Rod: Reference + NumPy")
        ui.text(f"Particles per rod: {self.ref_rod.num_points}")
        ui.text("Reference: blue, NumPy: orange")
        ui.separator()

        _changed, self.substeps = ui.slider_int("Substeps", self.substeps, 1, 16)
        _changed, self.linear_damping = ui.slider_float("Linear Damping", self.linear_damping, 0.0, 0.05)
        _changed, self.angular_damping = ui.slider_float("Angular Damping", self.angular_damping, 0.0, 0.05)

        ui.separator()
        offset_changed, self.compare_offset = ui.slider_float("Compare Offset", self.compare_offset, 0.1, 5.0)
        if offset_changed:
            half_offset = 0.5 * self.compare_offset
            self.ref_offset = np.array([0.0, -half_offset, 0.0], dtype=np.float32)
            self.numpy_offset = np.array([0.0, half_offset, 0.0], dtype=np.float32)
            self._sync_state_from_rods()

        ui.separator()
        changed_bend_k, self.bend_stiffness = ui.slider_float("Bend Stiffness", self.bend_stiffness, 0.0, 1.0)
        changed_twist_k, self.twist_stiffness = ui.slider_float(
            "Twist Stiffness", self.twist_stiffness, 0.0, 1.0
        )
        if changed_bend_k or changed_twist_k:
            self.ref_rod.set_bend_stiffness(self.bend_stiffness, self.twist_stiffness)
            self.numpy_rod.set_bend_stiffness(self.bend_stiffness, self.twist_stiffness)

        ui.separator()
        ui.text("Material Moduli")
        changed_young, self.young_modulus_scale = ui.slider_float(
            "Young Modulus (x1e6)", self.young_modulus_scale, 0.01, 100.0
        )
        changed_torsion, self.torsion_modulus_scale = ui.slider_float(
            "Torsion Modulus (x1e6)", self.torsion_modulus_scale, 0.01, 100.0
        )
        if changed_young or changed_torsion:
            young_modulus = float(self.young_modulus_scale) * 1.0e6
            torsion_modulus = float(self.torsion_modulus_scale) * 1.0e6
            self.ref_rod.young_modulus = young_modulus
            self.ref_rod.torsion_modulus = torsion_modulus
            self.numpy_rod.young_modulus = young_modulus
            self.numpy_rod.torsion_modulus = torsion_modulus

        ui.separator()
        ui.text("Rest Shape (Darboux Vector)")
        changed_rest_d1, self.rest_bend_d1 = ui.slider_float("Rest Bend d1", self.rest_bend_d1, -0.5, 0.5)
        changed_rest_d2, self.rest_bend_d2 = ui.slider_float("Rest Bend d2", self.rest_bend_d2, -0.5, 0.5)
        changed_rest_twist, self.rest_twist = ui.slider_float("Rest Twist", self.rest_twist, -0.5, 0.5)
        if changed_rest_d1 or changed_rest_d2 or changed_rest_twist:
            self.ref_rod.set_rest_darboux(self.rest_bend_d1, self.rest_bend_d2, self.rest_twist)
            self.numpy_rod.set_rest_darboux(self.rest_bend_d1, self.rest_bend_d2, self.rest_twist)

        ui.separator()
        gravity_changed, self.gravity_enabled = ui.checkbox("Gravity (G)", self.gravity_enabled)
        scale_changed, self.gravity_scale = ui.slider_float("Gravity Scale", self.gravity_scale, 0.0, 2.0)
        if gravity_changed or scale_changed:
            self._update_gravity()

        ui.separator()
        if self.supports_non_banded:
            changed_banded, self.use_banded = ui.checkbox("Use Banded Solver", self.use_banded)
            if changed_banded:
                self.ref_rod.set_solver_mode(self.use_banded)
                self.numpy_rod.set_solver_mode(self.use_banded)
                self.use_banded = self.ref_rod.use_banded
        else:
            ui.text("Non-banded solver not available in this DLL build.")

        ui.separator()
        ui.text("NumPy Overrides (candidate rod)")
        available_steps = []
        for step_name, label in self._numpy_step_labels.items():
            if self.numpy_rod.numpy_available.get(step_name, False):
                available_steps.append((step_name, label))
        for step_name, label in available_steps:
            current = self.numpy_rod.numpy_enabled.get(step_name, False)
            changed, enabled = ui.checkbox(f"{label} (NumPy)", current)
            if changed:
                self.numpy_rod.set_numpy_override(step_name, enabled)
        pending_count = len(self._numpy_step_labels) - len(available_steps)
        if pending_count > 0:
            ui.text(f"Pending NumPy steps: {pending_count} (see DEFKIT_TO_NUMPY.md)")

        ui.separator()
        _changed, self.show_segments = ui.checkbox("Show Rod Segments", self.show_segments)
        _changed, self.show_directors = ui.checkbox("Show Directors", self.show_directors)
        _changed, self.director_scale = ui.slider_float("Director Scale", self.director_scale, 0.01, 0.3)

        ui.separator()
        ui.text("Root Control (Numpad, both rods)")
        _changed, self.root_move_speed = ui.slider_float("Move Speed", self.root_move_speed, 0.1, 5.0)
        _changed, self.root_rotate_speed = ui.slider_float("Rotate Speed", self.root_rotate_speed, 0.1, 3.0)
        ui.text(f"  Rotation: {self.root_rotation:.2f} rad")
        ui.text("  4/6: X-, X+  8/2: Y+, Y-  9/3: Z+, Z-")
        ui.text("  7/1: Rotate +Z/-Z")

        ui.separator()
        ui.text("Controls:")
        ui.text("  G: Toggle gravity")
        ui.text("  R: Reset")

    def test_final(self):
        ref_anchor = self.ref_rod.positions[0, 0:3]
        ref_initial = self.ref_rod._initial_positions[0, 0:3]
        ref_dist = float(np.linalg.norm(ref_anchor - ref_initial))
        assert ref_dist < 1.0e-3, f"Reference anchor moved too far: {ref_dist}"

        numpy_anchor = self.numpy_rod.positions[0, 0:3]
        numpy_initial = self.numpy_rod._initial_positions[0, 0:3]
        numpy_dist = float(np.linalg.norm(numpy_anchor - numpy_initial))
        assert numpy_dist < 1.0e-3, f"NumPy anchor moved too far: {numpy_dist}"

        if not np.all(np.isfinite(self.ref_rod.positions[:, 0:3])):
            raise AssertionError("Non-finite reference positions detected")
        if not np.all(np.isfinite(self.numpy_rod.positions[:, 0:3])):
            raise AssertionError("Non-finite NumPy positions detected")


def create_parser():
    import argparse  # noqa: PLC0415

    parser = newton.examples.create_parser()
    parser.add_argument(
        "--dll-path",
        type=str,
        default=None,
        help="Path to DefKitAdv.dll. If omitted, attempts to load from PATH.",
    )
    parser.add_argument(
        "--calling-convention",
        type=str,
        choices=["cdecl", "stdcall"],
        default="cdecl",
        help="Calling convention used by the DLL (cdecl or stdcall).",
    )
    parser.add_argument("--num-points", type=int, default=64, help="Number of rod points.")
    parser.add_argument("--segment-length", type=float, default=0.1, help="Rest length per segment.")
    parser.add_argument("--particle-mass", type=float, default=1.0, help="Mass per particle (root fixed).")
    parser.add_argument("--particle-radius", type=float, default=0.02, help="Particle visualization radius.")
    parser.add_argument("--particle-height", type=float, default=1.0, help="Initial rod height (z).")
    parser.add_argument(
        "--rod-radius",
        type=float,
        default=None,
        help="Physical rod radius for direct solver (defaults to particle-radius).",
    )
    parser.add_argument(
        "--compare-offset",
        type=float,
        default=0.5,
        help="Y-offset separating reference and NumPy rods.",
    )
    parser.add_argument("--substeps", type=int, default=4, help="Integration substeps per frame.")
    parser.add_argument("--bend-stiffness", type=float, default=0.1, help="Per-edge bend stiffness.")
    parser.add_argument("--twist-stiffness", type=float, default=0.1, help="Per-edge twist stiffness.")
    parser.add_argument("--rest-bend-d1", type=float, default=0.0, help="Rest bend around d1 axis (rad/segment).")
    parser.add_argument("--rest-bend-d2", type=float, default=0.0, help="Rest bend around d2 axis (rad/segment).")
    parser.add_argument("--rest-twist", type=float, default=0.0, help="Rest twist around d3 axis (rad/segment).")
    parser.add_argument("--young-modulus", type=float, default=1.0e6, help="Young's modulus multiplier.")
    parser.add_argument("--torsion-modulus", type=float, default=1.0e6, help="Torsion modulus multiplier.")
    parser.add_argument(
        "--use-banded",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use banded direct solver (disable to use non-banded if available).",
    )
    parser.add_argument("--linear-damping", type=float, default=0.001, help="Linear damping coefficient.")
    parser.add_argument("--angular-damping", type=float, default=0.001, help="Angular damping coefficient.")
    parser.add_argument(
        "--gravity",
        type=float,
        nargs=3,
        default=[0.0, 0.0, -9.81],
        help="Gravity vector (x y z).",
    )
    parser.add_argument(
        "--lock-root-rotation",
        action="store_true",
        default=False,
        help="Lock root rotation by zeroing quaternion inverse mass.",
    )
    return parser


if __name__ == "__main__":
    viewer, args = newton.examples.init(create_parser())

    if isinstance(viewer, newton.viewer.ViewerGL):
        viewer.show_particles = True

    example = Example(viewer, args)
    newton.examples.run(example, args)
