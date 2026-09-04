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
# Example Basic Reduced Elastic Chair Stick Slip
#
# Demonstrates a reduced elastic monobloc plastic chair on an inclined ramp.
# The chair mesh is the CC0 "Plastic Monobloc Chair 01" asset from Poly Haven.
# A POD basis is generated from leg-bending exemplars so that the contact feet
# can store and release elastic energy during frictional stick/slip motion.
#
# Command: python -m newton.examples basic_reduced_elastic_chair_stick_slip
#
###########################################################################

from __future__ import annotations

import json
import math
import os
import urllib.request
from pathlib import Path

import numpy as np
import warp as wp

import newton
import newton.examples
from newton.examples.basic._reduced_elastic import (
    elastic_shape_deformed_vertices,
    init_elastic_solver_metric_tracking,
    transform_point,
    update_elastic_solver_metric_tracking,
)
from newton.examples.basic._reduced_elastic_contact import (
    contact_shape_config,
    run_example_test,
    validate_elastic_vertices,
)

ASSET_NAME = "plastic_monobloc_chair_01"
GLTF_NAME = "plastic_monobloc_chair_01_4k.gltf"
BIN_NAME = "plastic_monobloc_chair_01.bin"
POLYHAVEN_URLS = {
    GLTF_NAME: "https://dl.polyhaven.org/file/ph-assets/Models/gltf/4k/plastic_monobloc_chair_01/plastic_monobloc_chair_01_4k.gltf",
    BIN_NAME: "https://dl.polyhaven.org/file/ph-assets/Models/gltf/4k/plastic_monobloc_chair_01/plastic_monobloc_chair_01.bin",
}


def _asset_cache_dir() -> Path:
    explicit = os.environ.get("NEWTON_POLYHAVEN_ASSET_DIR")
    if explicit:
        return Path(explicit).expanduser()

    staging = Path.home() / "newton-assets-staging" / "polyhaven" / ASSET_NAME
    if staging.exists():
        return staging

    cache_root = Path(os.environ.get("NEWTON_CACHE_PATH", Path.home() / ".cache" / "newton"))
    return cache_root / "polyhaven" / ASSET_NAME


def _download_polyhaven_chair() -> Path:
    asset_dir = _asset_cache_dir()
    asset_dir.mkdir(parents=True, exist_ok=True)
    for filename, url in POLYHAVEN_URLS.items():
        path = asset_dir / filename
        if path.exists() and path.stat().st_size > 0:
            continue
        req = urllib.request.Request(url, headers={"User-Agent": "newton-reduced-elastic-chair-example/1.0"})
        with urllib.request.urlopen(req, timeout=60) as response:
            path.write_bytes(response.read())
    return asset_dir


def _read_accessor(gltf: dict, buffer: bytes, accessor_index: int) -> np.ndarray:
    components = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4}
    dtypes = {5121: np.uint8, 5123: np.uint16, 5125: np.uint32, 5126: np.float32}
    accessor = gltf["accessors"][accessor_index]
    view = gltf["bufferViews"][accessor["bufferView"]]
    dtype = dtypes[accessor["componentType"]]
    component_count = components[accessor["type"]]
    count = int(accessor["count"])
    byte_offset = int(view.get("byteOffset", 0)) + int(accessor.get("byteOffset", 0))
    byte_stride = int(view.get("byteStride", 0))

    if byte_stride == 0:
        return np.frombuffer(buffer, dtype=dtype, count=count * component_count, offset=byte_offset).reshape(
            count, component_count
        )

    values = np.empty((count, component_count), dtype=dtype)
    item_bytes = np.dtype(dtype).itemsize * component_count
    for i in range(count):
        start = byte_offset + i * byte_stride
        values[i] = np.frombuffer(buffer[start : start + item_bytes], dtype=dtype, count=component_count)
    return values


def _load_polyhaven_chair_mesh() -> tuple[np.ndarray, np.ndarray]:
    asset_dir = _download_polyhaven_chair()
    gltf = json.loads((asset_dir / GLTF_NAME).read_text())
    buffer = (asset_dir / BIN_NAME).read_bytes()
    primitive = gltf["meshes"][0]["primitives"][0]
    positions = np.array(_read_accessor(gltf, buffer, primitive["attributes"]["POSITION"]), dtype=np.float32)
    indices = np.array(_read_accessor(gltf, buffer, primitive["indices"]).reshape((-1,)), dtype=np.int32)

    # Poly Haven's glTF is Y-up. Newton examples use Z-up.
    vertices = np.column_stack((positions[:, 0], positions[:, 2], positions[:, 1])).astype(np.float32)
    bbox_min = vertices.min(axis=0)
    bbox_max = vertices.max(axis=0)
    center = 0.5 * (bbox_min + bbox_max)
    vertices -= center.reshape((1, 3))
    return vertices, indices


def _smoothstep01(x: np.ndarray) -> np.ndarray:
    s = np.clip(x, 0.0, 1.0)
    return s * s * (3.0 - 2.0 * s)


def _chair_leg_centers(vertices: np.ndarray) -> list[np.ndarray]:
    z_min = float(vertices[:, 2].min())
    lower = np.nonzero(vertices[:, 2] < z_min + 0.08)[0]
    centers: list[np.ndarray] = []
    for sx in (-1.0, 1.0):
        for sy in (-1.0, 1.0):
            ids = lower[(sx * vertices[lower, 0] > 0.0) & (sy * vertices[lower, 1] > 0.0)]
            if ids.shape[0] == 0:
                continue
            centers.append(vertices[ids].mean(axis=0))
    return centers


def _foot_weight(vertices: np.ndarray, center: np.ndarray) -> np.ndarray:
    dx = (vertices[:, 0] - float(center[0])) / 0.085
    dy = (vertices[:, 1] - float(center[1])) / 0.10
    z_min = float(vertices[:, 2].min())
    z_max = float(vertices[:, 2].max())
    leg_height = max(0.34 * (z_max - z_min), 1.0e-6)
    lower = 1.0 - _smoothstep01((vertices[:, 2] - z_min) / leg_height)
    return np.exp(-0.5 * (dx * dx + dy * dy)) * lower


def _build_chair_modal_basis(
    vertices: np.ndarray,
    mass: float,
    stiffness_scale: float = 1.0,
    damping_ratio: float = 0.005,
) -> newton.ModalBasis:
    centers = _chair_leg_centers(vertices)
    if len(centers) < 4:
        raise RuntimeError("Could not find four chair leg contact patches in the Poly Haven mesh")

    z_min = float(vertices[:, 2].min())
    z_max = float(vertices[:, 2].max())
    height = max(z_max - z_min, 1.0e-6)
    lower_profile = 1.0 - _smoothstep01((vertices[:, 2] - z_min) / (0.42 * height))
    seat_profile = _smoothstep01((vertices[:, 2] - (z_min + 0.22 * height)) / (0.42 * height))
    y_extent = max(float(np.max(np.abs(vertices[:, 1]))), 1.0e-6)
    z_unit = np.clip((vertices[:, 2] - z_min) / height, 0.0, 1.0)
    y_unit = np.clip(vertices[:, 1] / y_extent, -1.0, 1.0)

    weights = [_foot_weight(vertices, center) for center in centers]
    all_legs = np.clip(np.sum(weights, axis=0), 0.0, 1.0)
    lateral_profile = lower_profile * (0.35 + 0.65 * all_legs)

    snapshots: list[np.ndarray] = []
    for center, weight in zip(centers, weights, strict=True):
        disp = np.zeros_like(vertices)
        # Tangential leg bending. The downhill ramp axis is body-local +Y.
        disp[:, 1] = -0.055 * weight
        # Slight normal compression and lateral splay keep the mode visually
        # chair-like without adding nonlinear constraints.
        disp[:, 2] = 0.014 * weight
        disp[:, 0] = 0.010 * math.copysign(1.0, float(center[0])) * weight
        snapshots.append(disp)

    for sign in (-1.0, 1.0):
        disp = np.zeros_like(vertices)
        front_back = np.clip(vertices[:, 1] / max(float(np.max(np.abs(vertices[:, 1]))), 1.0e-6), -1.0, 1.0)
        disp[:, 1] = -0.045 * sign * front_back * all_legs
        disp[:, 2] = 0.010 * all_legs
        snapshots.append(disp)

    disp = np.zeros_like(vertices)
    disp[:, 1] = -0.040 * all_legs
    disp[:, 2] = 0.012 * all_legs
    snapshots.append(disp)

    disp = np.zeros_like(vertices)
    disp[:, 0] = (
        0.030 * lower_profile * np.clip(vertices[:, 0] / max(float(np.max(np.abs(vertices[:, 0]))), 1.0e-6), -1.0, 1.0)
    )
    disp[:, 1] = -0.018 * all_legs
    snapshots.append(disp)

    disp = np.zeros_like(vertices)
    disp[:, 1] = -0.030 * seat_profile
    disp[:, 2] = (
        -0.018 * seat_profile * np.clip(vertices[:, 1] / max(float(np.max(np.abs(vertices[:, 1]))), 1.0e-6), -1.0, 1.0)
    )
    snapshots.append(disp)

    disp = np.zeros_like(vertices)
    # Sideways sliding excites local-X foot shear when the chair is yawed
    # across the ramp instead of facing downhill.
    disp[:, 0] = -0.065 * lateral_profile
    disp[:, 1] = -0.012 * lateral_profile * y_unit
    disp[:, 2] = 0.012 * lateral_profile * (1.0 - z_unit)
    snapshots.append(disp)

    disp = np.zeros_like(vertices)
    disp[:, 0] = -0.040 * lower_profile * y_unit
    disp[:, 1] = 0.028 * lateral_profile
    disp[:, 2] = 0.014 * lateral_profile * y_unit
    snapshots.append(disp)

    generator = newton.ModalGeneratorPOD(
        sample_points=vertices,
        displacements=np.asarray(snapshots, dtype=np.float32),
        mode_count=10,
        total_mass=mass,
        stiffness_scale=1.0,
        damping_ratio=0.0,
        subtract_mean=False,
        label="polyhaven_monobloc_chair_pod",
    )
    basis = generator.build()
    stiffness = (
        np.array(
            [8500.0, 7750.0, 7000.0, 5950.0, 4910.0, 4060.0, 3310.0, 2650.0, 2130.0, 1770.0],
            dtype=np.float32,
        )
        * stiffness_scale
    )
    damping = np.array(
        [
            2.0 * damping_ratio * math.sqrt(max(float(m), 0.0) * max(float(k), 0.0))
            for m, k in zip(basis.mode_mass, stiffness, strict=True)
        ],
        dtype=np.float32,
    )
    basis.mode_stiffness = stiffness
    basis.mode_damping = damping
    return basis


def _foot_collision_mesh(vertices: np.ndarray, _indices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    centers = _chair_leg_centers(vertices)
    z_min = float(vertices[:, 2].min())
    hx = 0.055
    hy = 0.060
    hz = 0.012
    box_vertices: list[tuple[float, float, float]] = []
    indices: list[int] = []

    def add_foot(center: np.ndarray) -> None:
        base = len(box_vertices)
        cx = float(center[0])
        cy = float(center[1])
        cz = z_min + hz
        box_vertices.extend(
            (
                (cx - hx, cy - hy, cz - hz),
                (cx + hx, cy - hy, cz - hz),
                (cx + hx, cy + hy, cz - hz),
                (cx - hx, cy + hy, cz - hz),
                (cx - hx, cy - hy, cz + hz),
                (cx + hx, cy - hy, cz + hz),
                (cx + hx, cy + hy, cz + hz),
                (cx - hx, cy + hy, cz + hz),
            )
        )
        faces = (
            (0, 2, 1, 0, 3, 2),
            (4, 5, 6, 4, 6, 7),
            (0, 1, 5, 0, 5, 4),
            (1, 2, 6, 1, 6, 5),
            (2, 3, 7, 2, 7, 6),
            (3, 0, 4, 3, 4, 7),
        )
        for face in faces:
            indices.extend(base + i for i in face)

    for center in centers:
        add_foot(center)

    return np.asarray(box_vertices, dtype=np.float32), np.asarray(indices, dtype=np.int32)


def _rotation_matrix_x(angle: float) -> np.ndarray:
    c = math.cos(angle)
    s = math.sin(angle)
    return np.array(((1.0, 0.0, 0.0), (0.0, c, -s), (0.0, s, c)), dtype=np.float32)


def _rotation_matrix_z(angle: float) -> np.ndarray:
    c = math.cos(angle)
    s = math.sin(angle)
    return np.array(((c, -s, 0.0), (s, c, 0.0), (0.0, 0.0, 1.0)), dtype=np.float32)


def _quat_x(angle: float) -> wp.quat:
    return wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), angle)


def _quat_z(angle: float) -> wp.quat:
    return wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), angle)


def _arg_float(args, name: str, default: float) -> float:
    if args is None:
        return default
    return float(getattr(args, name, default))


class Example:
    solver_iterations = 12

    def __init__(self, viewer, args):
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_substeps = 8
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.sim_time = 0.0
        self.viewer = viewer
        self.args = args
        self.show_elastic_strain = False

        self.chair_mass = _arg_float(args, "chair_mass", 3.0)
        self.ramp_angle = math.radians(_arg_float(args, "ramp_angle_deg", 24.0))
        self.gravity = _arg_float(args, "gravity", -9.81)
        self.contact_stiffness = _arg_float(args, "contact_stiffness", 6.0e3)
        self.contact_damping = _arg_float(args, "contact_damping", 0.008)
        self.friction_mu = _arg_float(args, "friction_mu", 0.66)
        self.modal_stiffness_scale = _arg_float(args, "modal_stiffness_scale", 1.0)
        self.modal_damping_ratio = _arg_float(args, "modal_damping_ratio", 0.005)
        self.chair_yaw_angle = math.radians(_arg_float(args, "chair_yaw_angle_deg", 0.0))
        self.initial_ramp_y = _arg_float(args, "initial_ramp_y", -1.55)
        self.stop_enabled = _arg_float(args, "stop_enabled", 0.0) > 0.0
        self.stop_ramp_y = _arg_float(args, "stop_ramp_y", 0.35)
        self.stop_hy = _arg_float(args, "stop_hy", 0.025)
        self.stop_hz = _arg_float(args, "stop_hz", 0.040)
        self.pusher_enabled = _arg_float(args, "pusher_enabled", 0.0) > 0.0
        self.pusher_start_y = _arg_float(args, "pusher_start_y", -1.61)
        self.pusher_end_y = _arg_float(args, "pusher_end_y", -0.55)
        self.pusher_speed = _arg_float(args, "pusher_speed", 0.45)
        self.pusher_hy = _arg_float(args, "pusher_hy", 0.035)
        self.pusher_hz = _arg_float(args, "pusher_hz", 0.10)
        self.ramp_hz = 0.035
        self.ramp_top_origin = np.array([0.0, 0.0, 0.18], dtype=np.float32)
        self.ramp_rot = _quat_x(-self.ramp_angle)
        self.ramp_matrix = _rotation_matrix_x(-self.ramp_angle)
        self.chair_yaw_rot = _quat_z(self.chair_yaw_angle)
        self.chair_yaw_matrix = _rotation_matrix_z(self.chair_yaw_angle)
        self.chair_rot = wp.mul(self.ramp_rot, self.chair_yaw_rot)
        self.downhill_axis = self.ramp_matrix @ np.array([0.0, 1.0, 0.0], dtype=np.float32)
        self.ramp_normal = self.ramp_matrix @ np.array([0.0, 0.0, 1.0], dtype=np.float32)
        self.current_downhill_axis = np.array(self.downhill_axis, dtype=np.float32)

        vertices, indices = _load_polyhaven_chair_mesh()
        self.rest_vertices = vertices
        self.rest_indices = indices
        self.basis = _build_chair_modal_basis(
            vertices,
            self.chair_mass,
            stiffness_scale=self.modal_stiffness_scale,
            damping_ratio=self.modal_damping_ratio,
        )
        self.chair_height = float(vertices[:, 2].max() - vertices[:, 2].min())
        self.chair_foot_z = float(vertices[:, 2].min())

        contact_cfg = contact_shape_config(ke=self.contact_stiffness, kd=self.contact_damping, mu=self.friction_mu)
        contact_cfg.margin = 0.0015
        contact_cfg.gap = 0.002
        ramp_cfg = contact_shape_config(ke=self.contact_stiffness, kd=self.contact_damping, mu=self.friction_mu)
        ramp_cfg.margin = 0.0
        ramp_cfg.gap = 0.004
        ramp_cfg.is_visible = False
        ramp_visual_cfg = contact_shape_config(ke=0.0, kd=0.0, mu=0.0)
        ramp_visual_cfg.has_shape_collision = False
        ramp_visual_cfg.has_particle_collision = False
        stop_cfg = contact_shape_config(ke=1.6e4, kd=0.02, mu=0.8)
        stop_cfg.margin = 0.0015
        stop_cfg.gap = 0.0015
        pusher_cfg = contact_shape_config(ke=1.6e4, kd=0.02, mu=0.8)
        pusher_cfg.margin = 0.0015
        pusher_cfg.gap = 0.0015

        builder = newton.ModelBuilder(gravity=(0.0, 0.0, self.gravity), up_axis="Z")
        builder.num_rigid_contacts_per_world = 16384
        ramp_pos = self.ramp_top_origin - self.ramp_normal * self.ramp_hz
        self.ramp_body = builder.add_body(
            xform=wp.transform(wp.vec3(*ramp_pos), self.ramp_rot),
            mass=0.0,
            inertia=wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
            is_kinematic=True,
            label="fixed_slanted_ramp",
        )
        self.ramp_joint = len(builder.joint_type) - 1
        builder.add_shape_box(
            self.ramp_body,
            hx=0.72,
            hy=12.0,
            hz=self.ramp_hz,
            cfg=ramp_visual_cfg,
            label="inclined_plastic_ramp_visual",
        )
        builder.add_shape_plane(
            xform=wp.transform(wp.vec3(*self.ramp_top_origin), self.ramp_rot),
            width=1.44,
            length=24.0,
            cfg=ramp_cfg,
            label="inclined_plastic_ramp",
        )
        self.stop_shape = -1
        if self.stop_enabled:
            stop_world = self.ramp_top_origin + self.ramp_matrix @ np.array(
                [0.0, self.stop_ramp_y, self.stop_hz], dtype=np.float32
            )
            self.stop_shape = builder.add_shape_box(
                -1,
                xform=wp.transform(wp.vec3(*stop_world), self.ramp_rot),
                hx=0.68,
                hy=self.stop_hy,
                hz=self.stop_hz,
                cfg=stop_cfg,
                label="inclined_ramp_hard_stop",
            )
        self.pusher_body = -1
        self.pusher_joint = -1
        self.pusher_shape = -1
        if self.pusher_enabled:
            pusher_pos, pusher_quat = self._pusher_pose(0.0)
            self.pusher_body = builder.add_body(
                xform=wp.transform(wp.vec3(*pusher_pos), pusher_quat),
                mass=0.0,
                inertia=wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
                is_kinematic=True,
                label="kinematic_chair_pusher",
            )
            self.pusher_joint = len(builder.joint_type) - 1
            self.pusher_shape = builder.add_shape_box(
                self.pusher_body,
                hx=0.46,
                hy=self.pusher_hy,
                hz=self.pusher_hz,
                cfg=pusher_cfg,
                label="kinematic_chair_pusher_pad",
            )

        initial_local_on_ramp = np.array([0.0, self.initial_ramp_y, -self.chair_foot_z + 0.004], dtype=np.float32)
        initial_world = self.ramp_top_origin + self.ramp_matrix @ initial_local_on_ramp
        chair_mesh = newton.Mesh(
            vertices,
            indices,
            compute_inertia=False,
            is_solid=False,
            color=wp.vec3(0.16, 0.72, 0.24),
            roughness=0.62,
            metallic=0.0,
        )
        inertia = wp.mat33(0.22, 0.0, 0.0, 0.0, 0.24, 0.0, 0.0, 0.0, 0.18)
        self.chair = builder.add_body_elastic(
            xform=wp.transform(wp.vec3(*initial_world), self.chair_rot),
            mass=self.chair_mass,
            inertia=inertia,
            modal_basis=self.basis,
            label="polyhaven_monobloc_chair",
        )
        builder.add_shape_mesh(self.chair, mesh=chair_mesh, cfg=contact_cfg, label="polyhaven_monobloc_chair_mesh")

        builder.color()
        self.model = builder.finalize()
        self.model.rigid_contact_max = 16384
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.contacts = self.model.contacts()
        self.solver = newton.solvers.SolverVBD(
            self.model,
            iterations=self.solver_iterations,
            elastic_solver_metrics=True,
            rigid_contact_k_start=5.0e3,
            rigid_body_contact_buffer_size=256,
            friction_epsilon=1.5e-3,
            elastic_body_relaxation=0.6,
        )

        elastic_index = int(self.model.body_elastic_index.numpy()[self.chair])
        owner_joint = int(self.model.elastic_joint.numpy()[elastic_index])
        self.elastic_q_start = int(self.model.joint_q_start.numpy()[owner_joint])
        self.ramp_q_start = int(self.model.joint_q_start.numpy()[self.ramp_joint])
        self.ramp_qd_start = int(self.model.joint_qd_start.numpy()[self.ramp_joint])
        self.pusher_q_start = -1
        self.pusher_qd_start = -1
        if self.pusher_joint >= 0:
            self.pusher_q_start = int(self.model.joint_q_start.numpy()[self.pusher_joint])
            self.pusher_qd_start = int(self.model.joint_qd_start.numpy()[self.pusher_joint])
        self._elastic_sample_body = np.full(int(self.model.elastic_shape_vertex_total_count), -1, dtype=np.int32)
        for elastic_shape in range(int(self.model.elastic_shape_count)):
            start = int(self.model.elastic_shape_vertex_start.numpy()[elastic_shape])
            count = int(self.model.elastic_shape_vertex_count.numpy()[elastic_shape])
            body = int(self.model.elastic_shape_body.numpy()[elastic_shape])
            self._elastic_sample_body[start : start + count] = body
        self._elastic_sample_rest = np.asarray(self.model.elastic_shape_vertex_local.numpy(), dtype=np.float64)
        self._elastic_sample_phi = np.asarray(self.model.elastic_shape_vertex_phi.numpy(), dtype=np.float64).reshape(
            (-1, int(self.model.elastic_max_mode_count), 3)
        )
        self.max_contact_count = 0
        self.contact_dropouts_after_settle = 0
        self.max_mode_abs = 0.0
        self.max_displacement = 0.0
        self.max_lateral_displacement = 0.0
        self.max_stop_lateral_displacement = 0.0
        self.max_speed = 0.0
        self.latest_speed = 0.0
        self.min_speed_after_settle = 1.0e9
        self.stick_frames = 0
        self.slip_frames = 0
        self.stick_to_slip_events = 0
        self._was_sticking = False
        self.contact_stick_frames = 0
        self.contact_slip_frames = 0
        self.contact_stick_to_slip_events = 0
        self.contact_tail_stick_frames = 0
        self.contact_tail_slip_frames = 0
        self.contact_velocity_frames = 0
        self.max_contact_tangent_speed = 0.0
        self.latest_contact_tangent_speed = 0.0
        self.latest_contact_stick_speed = 0.0
        self.max_contact_normal_speed = 0.0
        self.contact_normal_speed_sum = 0.0
        self.contact_normal_speed_count = 0
        self.max_stop_contact_count = 0
        self.stop_contact_frames = 0
        self.stop_first_time = -1.0
        self.max_stop_contact_tangent_speed = 0.0
        self.max_pusher_contact_count = 0
        self.pusher_contact_frames = 0
        self.pusher_first_time = -1.0
        self._was_contact_sticking = False
        self._previous_contact_sample_world: dict[int, np.ndarray] = {}
        self.slide_start = self._slide_position()
        self.slide_end = self.slide_start
        self.previous_slide = self.slide_start
        init_elastic_solver_metric_tracking(self)

        self.viewer.set_model(self.model)
        self.viewer.show_elastic_strain = False
        self.viewer.elastic_strain_color_max = 0.045
        self._update_camera()

    def _update_camera(self):
        self.viewer.set_camera(
            pos=wp.vec3(3.4, -0.1, 1.05),
            pitch=-8.0,
            yaw=180.0,
        )
        if hasattr(self.viewer, "camera") and hasattr(self.viewer.camera, "fov"):
            self.viewer.camera.fov = 52.0

    def _ramp_pose(self, t: float) -> tuple[np.ndarray, wp.quat, float]:
        return (
            (self.ramp_top_origin - self.ramp_normal * self.ramp_hz).astype(np.float32),
            self.ramp_rot,
            self.ramp_angle,
        )

    def _pusher_pose(self, t: float) -> tuple[np.ndarray, wp.quat]:
        ramp_y = min(self.pusher_start_y + self.pusher_speed * t, self.pusher_end_y)
        local = np.array([0.0, ramp_y, self.pusher_hz], dtype=np.float32)
        return (self.ramp_top_origin + self.ramp_matrix @ local).astype(np.float32), self.ramp_rot

    def _apply_ramp_pose(self, t: float, previous_t: float):
        pos, quat, angle = self._ramp_pose(t)
        self.current_downhill_axis = _rotation_matrix_x(-angle) @ np.array([0.0, 1.0, 0.0], dtype=np.float32)
        linear = np.zeros(3, dtype=np.float32)
        angular = np.zeros(3, dtype=np.float32)

        body_q = self.state_0.body_q.numpy()
        body_qd = self.state_0.body_qd.numpy()
        joint_q = self.state_0.joint_q.numpy()
        joint_qd = self.state_0.joint_qd.numpy()

        body_q[self.ramp_body] = [pos[0], pos[1], pos[2], quat[0], quat[1], quat[2], quat[3]]
        body_qd[self.ramp_body] = [linear[0], linear[1], linear[2], angular[0], angular[1], angular[2]]
        joint_q[self.ramp_q_start : self.ramp_q_start + 7] = [
            pos[0],
            pos[1],
            pos[2],
            quat[0],
            quat[1],
            quat[2],
            quat[3],
        ]
        joint_qd[self.ramp_qd_start : self.ramp_qd_start + 6] = [
            linear[0],
            linear[1],
            linear[2],
            angular[0],
            angular[1],
            angular[2],
        ]

        if self.pusher_body >= 0 and self.pusher_q_start >= 0:
            pusher_pos, pusher_quat = self._pusher_pose(t)
            previous_pos, _ = self._pusher_pose(previous_t)
            pusher_linear = (pusher_pos - previous_pos) / max(t - previous_t, self.sim_dt)
            body_q[self.pusher_body] = [
                pusher_pos[0],
                pusher_pos[1],
                pusher_pos[2],
                pusher_quat[0],
                pusher_quat[1],
                pusher_quat[2],
                pusher_quat[3],
            ]
            body_qd[self.pusher_body] = [pusher_linear[0], pusher_linear[1], pusher_linear[2], 0.0, 0.0, 0.0]
            joint_q[self.pusher_q_start : self.pusher_q_start + 7] = [
                pusher_pos[0],
                pusher_pos[1],
                pusher_pos[2],
                pusher_quat[0],
                pusher_quat[1],
                pusher_quat[2],
                pusher_quat[3],
            ]
            joint_qd[self.pusher_qd_start : self.pusher_qd_start + 6] = [
                pusher_linear[0],
                pusher_linear[1],
                pusher_linear[2],
                0.0,
                0.0,
                0.0,
            ]
        self.state_0.body_q.assign(body_q)
        self.state_0.body_qd.assign(body_qd)
        self.state_0.joint_q.assign(joint_q)
        self.state_0.joint_qd.assign(joint_qd)

    def _slide_position(self) -> float:
        body_q = self.state_0.body_q.numpy()[self.chair]
        return float(
            np.dot(np.asarray(body_q[:3], dtype=np.float32) - self.ramp_top_origin, self.current_downhill_axis)
        )

    def _elastic_sample_world(self, sample: int) -> np.ndarray:
        body = int(self._elastic_sample_body[sample])
        elastic = int(self.model.body_elastic_index.numpy()[body])
        owner_joint = int(self.model.elastic_joint.numpy()[elastic])
        q_start = int(self.model.joint_q_start.numpy()[owner_joint]) + 7
        mode_count = int(self.model.elastic_mode_count.numpy()[elastic])
        q = self.state_0.joint_q.numpy()[q_start : q_start + mode_count]
        local = self._elastic_sample_rest[sample].copy()
        local += np.einsum("mc,m->c", self._elastic_sample_phi[sample, :mode_count], q)
        return transform_point(self.state_0.body_q.numpy()[body], local)

    def _update_contact_velocity_metrics(self, contact_count: int):
        shape0 = self.contacts.rigid_contact_shape0.numpy()[:contact_count]
        shape1 = self.contacts.rigid_contact_shape1.numpy()[:contact_count]
        sample0 = self.contacts.rigid_contact_elastic_sample0.numpy()[:contact_count]
        sample1 = self.contacts.rigid_contact_elastic_sample1.numpy()[:contact_count]
        normals = self.contacts.rigid_contact_normal.numpy()[:contact_count]
        next_sample_world: dict[int, np.ndarray] = {}
        tangent_speeds: list[float] = []
        normal_speeds: list[float] = []
        stop_tangent_speeds: list[float] = []

        for contact, (s0, s1) in enumerate(zip(sample0, sample1, strict=True)):
            sample = int(s0 if s0 >= 0 else s1)
            if sample < 0:
                continue

            point_world = self._elastic_sample_world(sample)
            previous = self._previous_contact_sample_world.get(sample)
            next_sample_world[sample] = point_world
            if previous is None:
                continue

            velocity = (point_world - previous) / self.sim_dt
            normal = np.asarray(normals[contact], dtype=np.float64)
            normal_norm = float(np.linalg.norm(normal))
            if normal_norm <= 1.0e-8:
                continue
            normal /= normal_norm
            normal_speed = abs(float(np.dot(velocity, normal)))
            tangent = velocity - normal * float(np.dot(velocity, normal))
            tangent_speed = float(np.linalg.norm(tangent))
            is_stop_contact = self.stop_shape >= 0 and (
                int(shape0[contact]) == self.stop_shape or int(shape1[contact]) == self.stop_shape
            )
            is_pusher_contact = self.pusher_shape >= 0 and (
                int(shape0[contact]) == self.pusher_shape or int(shape1[contact]) == self.pusher_shape
            )
            if is_stop_contact or is_pusher_contact:
                stop_tangent_speeds.append(tangent_speed)
                continue
            tangent_speeds.append(tangent_speed)
            normal_speeds.append(normal_speed)

        self._previous_contact_sample_world = next_sample_world
        if stop_tangent_speeds:
            self.max_stop_contact_tangent_speed = max(
                self.max_stop_contact_tangent_speed, float(np.percentile(stop_tangent_speeds, 75.0))
            )
        if not tangent_speeds:
            return

        stick_speed = float(np.percentile(tangent_speeds, 25.0))
        tangent_speed = float(np.percentile(tangent_speeds, 75.0))
        normal_speed = float(np.percentile(normal_speeds, 75.0))
        self.latest_contact_stick_speed = stick_speed
        self.latest_contact_tangent_speed = tangent_speed
        self.max_contact_tangent_speed = max(self.max_contact_tangent_speed, tangent_speed)
        self.max_contact_normal_speed = max(self.max_contact_normal_speed, normal_speed)
        self.contact_normal_speed_sum += normal_speed
        self.contact_normal_speed_count += 1

        if self.sim_time <= 0.5:
            return

        self.contact_velocity_frames += 1
        sticking = stick_speed < 0.035
        slipping = tangent_speed > 0.12
        self.contact_stick_frames += int(sticking)
        self.contact_slip_frames += int(slipping)
        if self.sim_time > 2.0:
            self.contact_tail_stick_frames += int(sticking)
            self.contact_tail_slip_frames += int(slipping)
        if self._was_contact_sticking and slipping:
            self.contact_stick_to_slip_events += 1
        self._was_contact_sticking = sticking or (self._was_contact_sticking and not slipping)

    def _update_metrics(self):
        update_elastic_solver_metric_tracking(self)
        contact_count = min(int(self.contacts.rigid_contact_count.numpy()[0]), int(self.model.rigid_contact_max))
        self.max_contact_count = max(self.max_contact_count, contact_count)
        if self.sim_time > 0.75 and contact_count == 0:
            self.contact_dropouts_after_settle += 1
        if self.stop_shape >= 0 and contact_count > 0:
            shape0 = self.contacts.rigid_contact_shape0.numpy()[:contact_count]
            shape1 = self.contacts.rigid_contact_shape1.numpy()[:contact_count]
            stop_contact_count = int(np.count_nonzero((shape0 == self.stop_shape) | (shape1 == self.stop_shape)))
            self.max_stop_contact_count = max(self.max_stop_contact_count, stop_contact_count)
            if stop_contact_count > 0:
                self.stop_contact_frames += 1
                if self.stop_first_time < 0.0:
                    self.stop_first_time = self.sim_time
            if self.pusher_shape >= 0:
                pusher_contact_count = int(
                    np.count_nonzero((shape0 == self.pusher_shape) | (shape1 == self.pusher_shape))
                )
                self.max_pusher_contact_count = max(self.max_pusher_contact_count, pusher_contact_count)
                if pusher_contact_count > 0:
                    self.pusher_contact_frames += 1
                    if self.pusher_first_time < 0.0:
                        self.pusher_first_time = self.sim_time

        q = self.state_0.joint_q.numpy()
        modes = q[self.elastic_q_start + 7 : self.elastic_q_start + 7 + self.basis.mode_count]
        self.max_mode_abs = max(self.max_mode_abs, float(np.max(np.abs(modes))))

        slide_now = self._slide_position()
        speed = abs((slide_now - self.previous_slide) / self.sim_dt)
        self.latest_speed = speed
        self.previous_slide = slide_now
        self.max_speed = max(self.max_speed, speed)
        if self.sim_time > 0.25:
            self.min_speed_after_settle = min(self.min_speed_after_settle, speed)
            sticking = speed < 0.10 and contact_count > 0
            slipping = speed > 0.10
            self.stick_frames += int(sticking)
            self.slip_frames += int(slipping)
            if self._was_sticking and slipping:
                self.stick_to_slip_events += 1
            self._was_sticking = sticking or (self._was_sticking and not slipping)
        if contact_count > 0:
            self._update_contact_velocity_metrics(contact_count)
        else:
            self._previous_contact_sample_world = {}

        self.slide_end = slide_now

        if int(getattr(self.model, "elastic_shape_count", 0)) > 0:
            deformed = elastic_shape_deformed_vertices(self.model, self.state_0)
            displacement = deformed - self.rest_vertices.astype(np.float64)
            lateral_displacement = float(np.max(np.abs(displacement[:, 1])))
            self.max_lateral_displacement = max(self.max_lateral_displacement, lateral_displacement)
            if self.max_stop_contact_count > 0:
                self.max_stop_lateral_displacement = max(self.max_stop_lateral_displacement, lateral_displacement)
            self.max_displacement = max(
                self.max_displacement,
                float(np.max(np.linalg.norm(displacement, axis=1))),
            )

    def simulate(self):
        for substep in range(self.sim_substeps):
            t = self.sim_time + substep * self.sim_dt
            self.state_0.clear_forces()
            self.viewer.apply_forces(self.state_0)
            self._apply_ramp_pose(t, max(t - self.sim_dt, 0.0))
            self.model.collide(self.state_0, self.contacts)
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0
            self._update_metrics()

    def step(self):
        self.simulate()
        self.sim_time += self.frame_dt

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.log_lines("/model/elastic_bodies/centerlines", None, None, None)
        self.viewer.log_points("/model/elastic_bodies/samples", None)
        self.viewer.log_points("/model/elastic_bodies/endpoints", None)
        self.viewer.log_contacts(self.contacts, self.state_0)
        self.viewer.end_frame()

    def test_final(self):
        if self.max_contact_count == 0:
            raise AssertionError("chair stick/slip example did not generate ramp contacts")
        slide_distance = self.slide_end - self.slide_start
        if slide_distance < 0.06:
            raise AssertionError(f"chair did not move down the ramp enough: {slide_distance}")
        if self.max_displacement < 0.004:
            raise AssertionError(f"chair leg modal deformation too small: {self.max_displacement}")
        if self.slip_frames < 4:
            raise AssertionError(
                "chair did not enter a sustained downhill sliding regime: "
                f"stick_frames={self.stick_frames}, slip_frames={self.slip_frames}, max_speed={self.max_speed}"
            )
        if self.contact_stick_frames < 4 or self.contact_slip_frames < 4:
            raise AssertionError(
                "chair foot contacts did not show both sticking and slipping tangential speed regimes: "
                f"contact_stick_frames={self.contact_stick_frames}, "
                f"contact_slip_frames={self.contact_slip_frames}, "
                f"max_contact_tangent_speed={self.max_contact_tangent_speed}"
            )
        if self.contact_stick_to_slip_events < 1:
            raise AssertionError("chair foot contacts did not transition from stick to slip")
        if self.contact_tail_stick_frames < 4:
            raise AssertionError(
                "chair foot contacts entered a pure slide regime in the tail: "
                f"tail_stick_frames={self.contact_tail_stick_frames}, "
                f"tail_slip_frames={self.contact_tail_slip_frames}"
            )
        if self.stop_shape >= 0:
            if self.pusher_shape >= 0 and self.max_pusher_contact_count == 0:
                raise AssertionError("chair was not contacted by the kinematic pusher")
            if self.max_stop_contact_count == 0:
                raise AssertionError("chair did not hit the downstream hard stop")
            if self.max_stop_lateral_displacement < 0.0065:
                raise AssertionError(
                    "chair hard-stop impact did not produce enough lateral leg deformation: "
                    f"max_stop_lateral_displacement={self.max_stop_lateral_displacement}"
                )
        normal_mean = self.contact_normal_speed_sum / max(self.contact_normal_speed_count, 1)
        if normal_mean > 0.16:
            raise AssertionError(f"chair contact normal motion is too bouncy: normal_speed_mean={normal_mean}")
        if self.final_modal_solve_residual_ratio > 0.08:
            raise AssertionError(f"chair modal solve residual ratio too high: {self.final_modal_solve_residual_ratio}")
        if self.final_modal_update_norm > 5.0e-5 or self.max_modal_update_norm > 1.3e-3:
            raise AssertionError(
                f"chair modal update too large: final={self.final_modal_update_norm}, max={self.max_modal_update_norm}"
            )
        validate_elastic_vertices(self.model, self.state_0)


def test(device=None, frame_count: int = 240):
    return run_example_test(Example, frame_count, device)


if __name__ == "__main__":
    parser = newton.examples.create_parser()
    viewer, args = newton.examples.init(parser)
    example = Example(viewer, args)
    newton.examples.run(example, args)
