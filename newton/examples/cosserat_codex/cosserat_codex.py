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

"""DefKitAdv iterative Cosserat rod demo (DLL-backed).

This example loads DefKitAdv.dll and runs the iterative Position and Orientation
Based Cosserat rod solver, using Newton only for visualization.

Command:
    uv run python newton/examples/cosserat_codex/cosserat_codex.py --dll-path "C:\\path\\to\\DefKitAdv.dll"
"""

from __future__ import annotations

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


class DefKitLibrary:
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
        if calling_convention == "stdcall":
            loader = ctypes.WinDLL
        else:
            loader = ctypes.CDLL
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
        self.ProjectElasticRodConstraints = self._get_function(
            "ProjectElasticRodConstraints",
            [
                ctypes.c_int,
                ctypes.POINTER(BtVector3),
                ctypes.POINTER(BtQuaternion),
                ctypes.POINTER(ctypes.c_float),
                ctypes.POINTER(ctypes.c_float),
                ctypes.POINTER(BtQuaternion),
                ctypes.POINTER(BtVector3),
                ctypes.POINTER(ctypes.c_float),
                ctypes.c_float,
                ctypes.c_float,
            ],
        )


class DefKitRodState:
    def __init__(
        self,
        num_points: int,
        segment_length: float,
        mass: float,
        particle_height: float,
        bend_stiffness: float,
        twist_stiffness: float,
        rest_bend_d1: float,
        rest_bend_d2: float,
        rest_twist: float,
        gravity: np.ndarray,
        lock_root_rotation: bool,
    ):
        self.num_points = num_points
        self.segment_length = segment_length

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

        self.rest_lengths = np.full(num_points, segment_length, dtype=np.float32)

        self.rest_darboux = np.zeros((num_points, 4), dtype=np.float32)
        self.set_rest_darboux(rest_bend_d1, rest_bend_d2, rest_twist)

        self.bend_twist_ks = np.zeros((num_points, 4), dtype=np.float32)
        self.set_bend_twist_stiffness(bend_stiffness, twist_stiffness)

        self.gravity = np.zeros((1, 4), dtype=np.float32)
        self.set_gravity(gravity)

        self._initial_positions = self.positions.copy()
        self._initial_orientations = self.orientations.copy()

    def set_gravity(self, gravity: np.ndarray):
        self.gravity[0, 0:3] = gravity.astype(np.float32)

    def set_bend_twist_stiffness(self, bend_stiffness: float, twist_stiffness: float):
        self.bend_twist_ks[:, 0] = bend_stiffness
        self.bend_twist_ks[:, 1] = bend_stiffness
        self.bend_twist_ks[:, 2] = twist_stiffness

    def set_rest_darboux(self, rest_bend_d1: float, rest_bend_d2: float, rest_twist: float):
        half_bend_d1 = 0.5 * rest_bend_d1
        half_bend_d2 = 0.5 * rest_bend_d2
        half_twist = 0.5 * rest_twist
        angle = math.sqrt(half_bend_d1 * half_bend_d1 + half_bend_d2 * half_bend_d2 + half_twist * half_twist)
        if angle < 1.0e-8:
            quat = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
        else:
            s = math.sin(angle) / angle
            quat = np.array(
                [half_bend_d1 * s, half_bend_d2 * s, half_twist * s, math.cos(angle)], dtype=np.float32
            )
        self.rest_darboux[:, 0:3] = quat[0:3]
        self.rest_darboux[:, 3] = quat[3]

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
        lib: DefKitLibrary,
        dt: float,
        linear_damping: float,
        angular_damping: float,
        iterations: int,
        stretch_shear_ks: float,
        bend_twist_ks: float,
    ):
        lib.PredictPositions_native(
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

        lib.PredictRotationsPBD(
            ctypes.c_float(dt),
            ctypes.c_float(angular_damping),
            ctypes.c_int(self.num_points),
            _as_ptr(self.orientations, BtQuaternion),
            _as_ptr(self.predicted_orientations, BtQuaternion),
            _as_ptr(self.angular_velocities, BtVector3),
            _as_ptr(self.torques, BtVector3),
            _as_float_ptr(self.quat_inv_masses),
        )

        for _ in range(iterations):
            lib.ProjectElasticRodConstraints(
                ctypes.c_int(self.num_points),
                _as_ptr(self.predicted_positions, BtVector3),
                _as_ptr(self.predicted_orientations, BtQuaternion),
                _as_float_ptr(self.inv_masses),
                _as_float_ptr(self.quat_inv_masses),
                _as_ptr(self.rest_darboux, BtQuaternion),
                _as_ptr(self.bend_twist_ks, BtVector3),
                _as_float_ptr(self.rest_lengths),
                ctypes.c_float(stretch_shear_ks),
                ctypes.c_float(bend_twist_ks),
            )

        lib.Integrate_native(
            ctypes.c_float(dt),
            ctypes.c_int(self.num_points),
            _as_ptr(self.positions, BtVector3),
            _as_ptr(self.predicted_positions, BtVector3),
            _as_ptr(self.velocities, BtVector3),
            _as_float_ptr(self.inv_masses),
        )

        lib.IntegrateRotationsPBD(
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
        self.constraint_iterations = args.constraint_iterations
        self.linear_damping = args.linear_damping
        self.angular_damping = args.angular_damping
        self.stretch_shear_ks = args.stretch_shear_ks
        self.bend_twist_ks = args.bend_twist_ks
        self.bend_stiffness = args.bend_stiffness
        self.twist_stiffness = args.twist_stiffness
        self.rest_bend_d1 = args.rest_bend_d1
        self.rest_bend_d2 = args.rest_bend_d2
        self.rest_twist = args.rest_twist

        self.base_gravity = np.array(args.gravity, dtype=np.float32)
        self.gravity_enabled = True
        self.gravity_scale = 1.0

        self.show_segments = True
        self.show_directors = False
        self.director_scale = 0.1

        self._gravity_key_was_down = False
        self._reset_key_was_down = False

        self.lib = DefKitLibrary(args.dll_path, args.calling_convention)

        self.rod_state = DefKitRodState(
            num_points=args.num_points,
            segment_length=args.segment_length,
            mass=args.particle_mass,
            particle_height=args.particle_height,
            bend_stiffness=self.bend_stiffness,
            twist_stiffness=self.twist_stiffness,
            rest_bend_d1=self.rest_bend_d1,
            rest_bend_d2=self.rest_bend_d2,
            rest_twist=self.rest_twist,
            gravity=self.base_gravity,
            lock_root_rotation=args.lock_root_rotation,
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
            self.sim_time = 0.0
            self._sync_state_from_rod()
        self._reset_key_was_down = r_down

    def step(self):
        self._handle_keyboard_input()

        sub_dt = self.frame_dt / max(self.substeps, 1)
        for _ in range(self.substeps):
            self.rod_state.step(
                self.lib,
                sub_dt,
                self.linear_damping,
                self.angular_damping,
                self.constraint_iterations,
                self.stretch_shear_ks,
                self.bend_twist_ks,
            )

        self._sync_state_from_rod()
        self.sim_time += self.frame_dt

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
        ui.text("DefKit PBD Cosserat Rod (DLL)")
        ui.text(f"Particles: {self.rod_state.num_points}")
        ui.separator()

        _changed, self.substeps = ui.slider_int("Substeps", self.substeps, 1, 16)
        _changed, self.constraint_iterations = ui.slider_int(
            "Constraint Iterations", self.constraint_iterations, 1, 16
        )
        _changed, self.linear_damping = ui.slider_float("Linear Damping", self.linear_damping, 0.0, 0.05)
        _changed, self.angular_damping = ui.slider_float("Angular Damping", self.angular_damping, 0.0, 0.05)

        ui.separator()
        changed_stretch, self.stretch_shear_ks = ui.slider_float(
            "Stretch/Shear Ks", self.stretch_shear_ks, 0.0, 1.0
        )
        changed_bend, self.bend_twist_ks = ui.slider_float("Bend/Twist Ks", self.bend_twist_ks, 0.0, 1.0)
        changed_bend_k, self.bend_stiffness = ui.slider_float("Bend Stiffness", self.bend_stiffness, 0.0, 1.0)
        changed_twist_k, self.twist_stiffness = ui.slider_float(
            "Twist Stiffness", self.twist_stiffness, 0.0, 1.0
        )
        if changed_bend_k or changed_twist_k:
            self.rod_state.set_bend_twist_stiffness(self.bend_stiffness, self.twist_stiffness)

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
        _changed, self.show_segments = ui.checkbox("Show Rod Segments", self.show_segments)
        _changed, self.show_directors = ui.checkbox("Show Directors", self.show_directors)
        _changed, self.director_scale = ui.slider_float("Director Scale", self.director_scale, 0.01, 0.3)

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
    parser.add_argument("--substeps", type=int, default=4, help="Integration substeps per frame.")
    parser.add_argument("--constraint-iterations", type=int, default=4, help="Constraint iterations per substep.")
    parser.add_argument("--stretch-shear-ks", type=float, default=1.0, help="Stretch/shear stiffness multiplier.")
    parser.add_argument("--bend-twist-ks", type=float, default=1.0, help="Bend/twist stiffness multiplier.")
    parser.add_argument("--bend-stiffness", type=float, default=0.1, help="Per-edge bend stiffness.")
    parser.add_argument("--twist-stiffness", type=float, default=0.1, help="Per-edge twist stiffness.")
    parser.add_argument("--rest-bend-d1", type=float, default=0.0, help="Rest bend around d1 axis (rad/segment).")
    parser.add_argument("--rest-bend-d2", type=float, default=0.0, help="Rest bend around d2 axis (rad/segment).")
    parser.add_argument("--rest-twist", type=float, default=0.0, help="Rest twist around d3 axis (rad/segment).")
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
