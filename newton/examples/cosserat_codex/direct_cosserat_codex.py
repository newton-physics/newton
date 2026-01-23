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

"""DefKitAdv direct (global) Cosserat rod demo (DLL-backed).

This example runs the Direct Position Based Solver for Stiff Rods from
DefKitAdv.dll and uses Newton only for visualization.

Command:
    uv run python newton/examples/cosserat_codex/direct_cosserat_codex.py --dll-path "C:\\path\\to\\DefKitAdv.dll"
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

    def step(
        self,
        dt: float,
        linear_damping: float,
        angular_damping: float,
    ):
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

        if self.use_banded or not self.supports_non_banded:
            self.lib.UpdateConstraints_DirectElasticRodConstraintsBanded(
                self.rod_ptr,
                ctypes.c_int(self.num_points),
                _as_ptr(self.predicted_positions, BtVector3),
                _as_ptr(self.predicted_orientations, BtQuaternion),
                _as_float_ptr(self.inv_masses),
            )
            self.lib.ComputeJacobians_DirectElasticRodConstraintsBanded(
                self.rod_ptr,
                ctypes.c_int(0),
                ctypes.c_int(self.num_edges),
                _as_ptr(self.predicted_positions, BtVector3),
                _as_ptr(self.predicted_orientations, BtQuaternion),
                _as_float_ptr(self.inv_masses),
            )
            self.lib.AssembleJMJT_DirectElasticRodConstraintsBanded(
                self.rod_ptr,
                ctypes.c_int(0),
                ctypes.c_int(self.num_edges),
                _as_ptr(self.predicted_positions, BtVector3),
                _as_ptr(self.predicted_orientations, BtQuaternion),
                _as_float_ptr(self.inv_masses),
            )
            self.lib.ProjectJMJT_DirectElasticRodConstraintsBanded(
                self.rod_ptr,
                ctypes.c_int(self.num_points),
                _as_ptr(self.predicted_positions, BtVector3),
                _as_ptr(self.predicted_orientations, BtQuaternion),
                _as_float_ptr(self.inv_masses),
                _as_ptr(self.pos_corrections, BtVector3),
                _as_ptr(self.rot_corrections, BtQuaternion),
            )
        else:
            self.lib.ProjectDirectElasticRodConstraints(
                self.rod_ptr,
                ctypes.c_int(self.num_points),
                _as_ptr(self.predicted_positions, BtVector3),
                _as_ptr(self.predicted_orientations, BtQuaternion),
                _as_float_ptr(self.inv_masses),
                _as_ptr(self.pos_corrections, BtVector3),
                _as_ptr(self.rot_corrections, BtQuaternion),
            )

        self.lib.Integrate_native(
            ctypes.c_float(dt),
            ctypes.c_int(self.num_points),
            _as_ptr(self.positions, BtVector3),
            _as_ptr(self.predicted_positions, BtVector3),
            _as_ptr(self.velocities, BtVector3),
            _as_float_ptr(self.inv_masses),
        )

        self.lib.IntegrateRotationsPBD(
            ctypes.c_float(dt),
            ctypes.c_int(self.num_points),
            _as_ptr(self.orientations, BtQuaternion),
            _as_ptr(self.predicted_orientations, BtQuaternion),
            _as_ptr(self.prev_orientations, BtQuaternion),
            _as_ptr(self.angular_velocities, BtVector3),
            _as_float_ptr(self.quat_inv_masses),
        )


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
        self.rod_state = DefKitDirectRodState(
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
            pos = tuple(self.rod_state.positions[i, 0:3])
            builder.add_particle(pos=pos, vel=(0.0, 0.0, 0.0), mass=mass, radius=args.particle_radius)

        self.model = builder.finalize()
        self.state = self.model.state()

        self.viewer.set_model(self.model)
        self.viewer.show_particles = True

        self._segment_colors = np.tile(np.array([0.2, 0.6, 1.0], dtype=np.float32), (args.num_points - 1, 1))

        self._sync_state_from_rod()
        self._update_gravity()
        self._root_base_orientation = self.rod_state.orientations[0].copy()

    def __del__(self):
        if hasattr(self, "rod_state"):
            self.rod_state.destroy()

    def _update_gravity(self):
        if self.gravity_enabled:
            gravity = self.base_gravity * self.gravity_scale
        else:
            gravity = np.zeros(3, dtype=np.float32)
        self.rod_state.set_gravity(gravity)

    def _sync_state_from_rod(self):
        positions = self.rod_state.positions[:, 0:3].astype(np.float32)
        velocities = self.rod_state.velocities[:, 0:3].astype(np.float32)
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
            self.rod_state.reset()
            self.root_rotation = 0.0
            self._apply_root_rotation()
            self.sim_time = 0.0
            self._sync_state_from_rod()
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
            self.rod_state.step(sub_dt, self.linear_damping, self.angular_damping)

        self._sync_state_from_rod()
        self.sim_time += self.frame_dt

    def _apply_root_translation(self, dx: float, dy: float, dz: float):
        pos = self.rod_state.positions[0, 0:3]
        new_pos = pos + np.array([dx, dy, dz], dtype=np.float32)
        self.rod_state.positions[0, 0:3] = new_pos
        self.rod_state.predicted_positions[0, 0:3] = new_pos
        self.rod_state.velocities[0, 0:3] = 0.0

    def _apply_root_rotation(self):
        q_twist = _quat_from_axis_angle(np.array([0.0, 0.0, 1.0], dtype=np.float32), self.root_rotation)
        q_new = _quat_multiply(self._root_base_orientation, q_twist)
        self.rod_state.orientations[0] = q_new
        self.rod_state.predicted_orientations[0] = q_new
        self.rod_state.prev_orientations[0] = q_new

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

    def _build_director_lines(self):
        num_edges = self.rod_state.num_points - 1
        positions = self.rod_state.positions[:, 0:3]
        orientations = self.rod_state.orientations

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
            starts = self.rod_state.positions[:-1, 0:3].astype(np.float32)
            ends = self.rod_state.positions[1:, 0:3].astype(np.float32)
            self.viewer.log_lines(
                "/rod",
                wp.array(starts, dtype=wp.vec3, device=self.model.device),
                wp.array(ends, dtype=wp.vec3, device=self.model.device),
                wp.array(self._segment_colors, dtype=wp.vec3, device=self.model.device),
            )
        else:
            self.viewer.log_lines("/rod", None, None, None)

        if self.show_directors:
            starts, ends, colors = self._build_director_lines()
            self.viewer.log_lines(
                "/directors",
                wp.array(starts, dtype=wp.vec3, device=self.model.device),
                wp.array(ends, dtype=wp.vec3, device=self.model.device),
                wp.array(colors, dtype=wp.vec3, device=self.model.device),
            )
        else:
            self.viewer.log_lines("/directors", None, None, None)

        self.viewer.end_frame()

    def gui(self, ui):
        ui.text("DefKit Direct Cosserat Rod (DLL)")
        ui.text(f"Particles: {self.rod_state.num_points}")
        ui.separator()

        _changed, self.substeps = ui.slider_int("Substeps", self.substeps, 1, 16)
        _changed, self.linear_damping = ui.slider_float("Linear Damping", self.linear_damping, 0.0, 0.05)
        _changed, self.angular_damping = ui.slider_float("Angular Damping", self.angular_damping, 0.0, 0.05)

        ui.separator()
        changed_bend_k, self.bend_stiffness = ui.slider_float("Bend Stiffness", self.bend_stiffness, 0.0, 1.0)
        changed_twist_k, self.twist_stiffness = ui.slider_float(
            "Twist Stiffness", self.twist_stiffness, 0.0, 1.0
        )
        if changed_bend_k or changed_twist_k:
            self.rod_state.set_bend_stiffness(self.bend_stiffness, self.twist_stiffness)

        ui.separator()
        ui.text("Material Moduli")
        changed_young, self.young_modulus_scale = ui.slider_float(
            "Young Modulus (x1e6)", self.young_modulus_scale, 0.01, 100.0
        )
        changed_torsion, self.torsion_modulus_scale = ui.slider_float(
            "Torsion Modulus (x1e6)", self.torsion_modulus_scale, 0.01, 100.0
        )
        if changed_young or changed_torsion:
            self.rod_state.young_modulus = float(self.young_modulus_scale) * 1.0e6
            self.rod_state.torsion_modulus = float(self.torsion_modulus_scale) * 1.0e6

        ui.separator()
        ui.text("Rest Shape (Darboux Vector)")
        changed_rest_d1, self.rest_bend_d1 = ui.slider_float("Rest Bend d1", self.rest_bend_d1, -0.5, 0.5)
        changed_rest_d2, self.rest_bend_d2 = ui.slider_float("Rest Bend d2", self.rest_bend_d2, -0.5, 0.5)
        changed_rest_twist, self.rest_twist = ui.slider_float("Rest Twist", self.rest_twist, -0.5, 0.5)
        if changed_rest_d1 or changed_rest_d2 or changed_rest_twist:
            self.rod_state.set_rest_darboux(self.rest_bend_d1, self.rest_bend_d2, self.rest_twist)

        ui.separator()
        gravity_changed, self.gravity_enabled = ui.checkbox("Gravity (G)", self.gravity_enabled)
        scale_changed, self.gravity_scale = ui.slider_float("Gravity Scale", self.gravity_scale, 0.0, 2.0)
        if gravity_changed or scale_changed:
            self._update_gravity()

        ui.separator()
        if self.supports_non_banded:
            changed_banded, self.use_banded = ui.checkbox("Use Banded Solver", self.use_banded)
            if changed_banded:
                self.rod_state.set_solver_mode(self.use_banded)
                self.use_banded = self.rod_state.use_banded
        else:
            ui.text("Non-banded solver not available in this DLL build.")

        ui.separator()
        _changed, self.show_segments = ui.checkbox("Show Rod Segments", self.show_segments)
        _changed, self.show_directors = ui.checkbox("Show Directors", self.show_directors)
        _changed, self.director_scale = ui.slider_float("Director Scale", self.director_scale, 0.01, 0.3)

        ui.separator()
        ui.text("Root Control (Numpad)")
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
        anchor_pos = self.state.particle_q.numpy()[0]
        initial_pos = self.rod_state._initial_positions[0, 0:3]
        dist = float(np.linalg.norm(anchor_pos - initial_pos))
        assert dist < 1.0e-3, f"Anchor moved too far: {dist}"

        if not np.all(np.isfinite(self.rod_state.positions[:, 0:3])):
            raise AssertionError("Non-finite particle positions detected")


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
    parser.add_argument("--num-points", type=int, default=20, help="Number of rod points.")
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
