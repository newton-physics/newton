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
# Example Robot UR10 Elastic Panel
#
# Shows a UR10-like industrial arm carrying a reduced elastic automotive panel.
# The robot is treated as a kinematic trajectory source, while the panel is a
# floating-frame reduced elastic body attached to the end effector through
# multiple fixed gripper samples.
#
# Command: python -m newton.examples robot_ur10_elastic_panel
#
###########################################################################

from __future__ import annotations

import itertools
import math
import struct
from pathlib import Path

import numpy as np
import warp as wp

import newton
import newton.examples
import newton.utils
from newton.examples.basic._reduced_elastic import elastic_shape_deformed_vertices, joint_endpoint_world


def _quat_from_axis_angle(axis: tuple[float, float, float], angle: float) -> wp.quat:
    ax = wp.vec3(*axis)
    return wp.quat_from_axis_angle(wp.normalize(ax), float(angle))


def _quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return np.array(
        [
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
            aw * bw - ax * bx - ay * by - az * bz,
        ],
        dtype=np.float32,
    )


def _quat_conjugate(q: np.ndarray) -> np.ndarray:
    return np.array([-q[0], -q[1], -q[2], q[3]], dtype=np.float32)


def _quat_rotate_np(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    qv = np.asarray(q[:3], dtype=np.float32)
    v = np.asarray(v, dtype=np.float32)
    return v + 2.0 * np.cross(qv, np.cross(qv, v) + float(q[3]) * v)


def _quat_rotate_points_np(q: np.ndarray, points: np.ndarray) -> np.ndarray:
    qv = np.asarray(q[:3], dtype=np.float32)
    points = np.asarray(points, dtype=np.float32)
    return points + 2.0 * np.cross(qv.reshape((1, 3)), np.cross(qv.reshape((1, 3)), points) + float(q[3]) * points)


def _axis_angle_np(axis: tuple[float, float, float], angle: float) -> np.ndarray:
    axis_np = np.asarray(axis, dtype=np.float32)
    axis_np /= max(float(np.linalg.norm(axis_np)), 1.0e-8)
    half = 0.5 * float(angle)
    return np.array([*(math.sin(half) * axis_np), math.cos(half)], dtype=np.float32)


def _quat_from_z_axis(direction: np.ndarray) -> wp.quat:
    direction = np.asarray(direction, dtype=np.float32)
    length = float(np.linalg.norm(direction))
    if length < 1.0e-8:
        return wp.quat_identity()
    axis_to = direction / length
    axis_from = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    dot = float(np.clip(np.dot(axis_from, axis_to), -1.0, 1.0))
    if dot > 1.0 - 1.0e-6:
        return wp.quat_identity()
    if dot < -1.0 + 1.0e-6:
        return _quat_from_axis_angle((1.0, 0.0, 0.0), math.pi)
    axis = np.cross(axis_from, axis_to)
    return _quat_from_axis_angle((float(axis[0]), float(axis[1]), float(axis[2])), math.acos(dot))


def _cylinder_xform_between(start: np.ndarray, end: np.ndarray) -> tuple[wp.transform, float]:
    start = np.asarray(start, dtype=np.float32)
    end = np.asarray(end, dtype=np.float32)
    delta = end - start
    length = float(np.linalg.norm(delta))
    center = 0.5 * (start + end)
    return wp.transform(wp.vec3(*center), _quat_from_z_axis(delta)), 0.5 * length


def _make_procedural_hood_panel(
    length: float = 1.18,
    width: float = 0.81,
    camber: float = 0.065,
    nx: int = 34,
    ny: int = 24,
) -> tuple[np.ndarray, np.ndarray]:
    """Return a hood-frame-like triangulated shell used until CarHoods STL is extracted."""
    xs = np.linspace(-0.5 * length, 0.5 * length, nx + 1, dtype=np.float32)
    ys = np.linspace(-0.5 * width, 0.5 * width, ny + 1, dtype=np.float32)

    def is_cutout(cx: float, cy: float) -> bool:
        if abs(cx) > 0.34 * length or abs(cy) > 0.38 * width:
            return False
        pockets = (
            (-0.20 * length, -0.16 * width, 0.10 * length, 0.11 * width),
            (-0.20 * length, 0.16 * width, 0.10 * length, 0.11 * width),
            (0.14 * length, -0.15 * width, 0.11 * length, 0.10 * width),
            (0.14 * length, 0.15 * width, 0.11 * length, 0.10 * width),
        )
        for px, py, rx, ry in pockets:
            if ((cx - px) / rx) ** 2 + ((cy - py) / ry) ** 2 < 1.0:
                return True
        return False

    active: list[tuple[int, int]] = []
    used: set[tuple[int, int]] = set()
    for i in range(nx):
        for j in range(ny):
            cx = 0.5 * (float(xs[i]) + float(xs[i + 1]))
            cy = 0.5 * (float(ys[j]) + float(ys[j + 1]))
            if is_cutout(cx, cy):
                continue
            active.append((i, j))
            used.update(((i, j), (i + 1, j), (i + 1, j + 1), (i, j + 1)))

    node_lookup: dict[tuple[int, int], int] = {}
    vertices: list[tuple[float, float, float]] = []
    for key in sorted(used):
        x = float(xs[key[0]])
        y = float(ys[key[1]])
        xi = 2.0 * x / length
        eta = 2.0 * y / width
        z = camber * (1.0 - 0.55 * xi * xi - 0.35 * eta * eta)
        node_lookup[key] = len(vertices)
        vertices.append((x, y, z))

    indices: list[int] = []
    for i, j in active:
        a = node_lookup[(i, j)]
        b = node_lookup[(i + 1, j)]
        c = node_lookup[(i + 1, j + 1)]
        d = node_lookup[(i, j + 1)]
        indices.extend((a, b, c, a, c, d))

    return np.asarray(vertices, dtype=np.float32), np.asarray(indices, dtype=np.int32)


def _read_stl_mesh(path: Path) -> tuple[np.ndarray, np.ndarray]:
    data = path.read_bytes()
    if len(data) >= 84:
        tri_count = struct.unpack_from("<I", data, 80)[0]
        if 84 + 50 * tri_count == len(data):
            triangles = np.empty((tri_count, 3, 3), dtype=np.float32)
            offset = 84
            for tri in range(tri_count):
                offset += 12
                triangles[tri, 0] = struct.unpack_from("<3f", data, offset)
                triangles[tri, 1] = struct.unpack_from("<3f", data, offset + 12)
                triangles[tri, 2] = struct.unpack_from("<3f", data, offset + 24)
                offset += 38
            return _deduplicate_triangle_vertices(triangles)

    vertices: list[tuple[float, float, float]] = []
    for raw_line in data.decode("utf-8", errors="ignore").splitlines():
        parts = raw_line.strip().split()
        if len(parts) == 4 and parts[0].lower() == "vertex":
            vertices.append((float(parts[1]), float(parts[2]), float(parts[3])))
    if len(vertices) % 3 != 0 or not vertices:
        raise ValueError(f"Could not parse STL triangles from {path}")
    return _deduplicate_triangle_vertices(np.asarray(vertices, dtype=np.float32).reshape((-1, 3, 3)))


def _deduplicate_triangle_vertices(triangles: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    flat_vertices = np.asarray(triangles, dtype=np.float32).reshape((-1, 3))
    rounded = np.round(flat_vertices / 1.0e-6).astype(np.int64)
    _, first_indices, inverse = np.unique(rounded, axis=0, return_index=True, return_inverse=True)
    vertices = flat_vertices[np.sort(first_indices)]
    remap = np.empty(first_indices.shape[0], dtype=np.int32)
    remap[np.argsort(first_indices)] = np.arange(first_indices.shape[0], dtype=np.int32)
    indices = remap[inverse].astype(np.int32)
    return vertices.astype(np.float32, copy=False), indices


def _normalize_panel_mesh(vertices: np.ndarray, target_length: float = 1.18) -> np.ndarray:
    points = np.asarray(vertices, dtype=np.float32)
    center = 0.5 * (np.min(points, axis=0) + np.max(points, axis=0))
    centered = points - center.reshape((1, 3))
    extents = np.ptp(centered, axis=0)
    order = np.argsort(extents)
    mapped = np.empty_like(centered)
    mapped[:, 0] = centered[:, order[2]]
    mapped[:, 1] = centered[:, order[1]]
    mapped[:, 2] = centered[:, order[0]]
    mapped[:, 2] -= float(np.mean(mapped[:, 2]))
    length = max(float(np.ptp(mapped[:, 0])), 1.0e-6)
    mapped *= float(target_length) / length
    return mapped.astype(np.float32, copy=False)


def _find_carhood_stl(carhoods_dir: Path) -> Path | None:
    if not carhoods_dir.exists():
        return None
    candidates = sorted(carhoods_dir.glob("**/*_decimated_*.stl"))
    if not candidates:
        candidates = sorted(carhoods_dir.glob("**/*_decimated_*.STL"))
    if not candidates:
        candidates = sorted(carhoods_dir.glob("**/*.stl"))
    if not candidates:
        candidates = sorted(carhoods_dir.glob("**/*.STL"))
    return candidates[0] if candidates else None


def _load_panel_mesh(args) -> tuple[np.ndarray, np.ndarray]:
    mesh_path = getattr(args, "panel_mesh", None) if args is not None else None
    if mesh_path is None:
        carhoods_dir = getattr(args, "carhoods_dir", None) if args is not None else None
        mesh_path = _find_carhood_stl(Path(carhoods_dir)) if carhoods_dir is not None else None

    if mesh_path is None:
        return _make_procedural_hood_panel()

    mesh_path = Path(mesh_path)
    vertices, indices = _read_stl_mesh(mesh_path)
    vertices = _normalize_panel_mesh(vertices)
    return vertices, indices


def _nearest_panel_surface_point(vertices: np.ndarray, x: float, y: float) -> np.ndarray:
    d2 = (vertices[:, 0] - float(x)) ** 2 + (vertices[:, 1] - float(y)) ** 2
    return np.asarray(vertices[int(np.argmin(d2))], dtype=np.float32)


def _build_panel_rib_segments(vertices: np.ndarray) -> np.ndarray:
    x = vertices[:, 0]
    y = vertices[:, 1]
    length = max(float(np.max(x) - np.min(x)), 1.0e-6)
    width = max(float(np.max(y) - np.min(y)), 1.0e-6)
    max_dist = 0.055
    max_jump = 0.13
    segments: list[tuple[int, int]] = []

    def nearest_index(target_x: float, target_y: float) -> int | None:
        d2 = (x - target_x) ** 2 + (y - target_y) ** 2
        index = int(np.argmin(d2))
        if float(d2[index]) > max_dist * max_dist:
            return None
        return index

    def add_polyline(targets: list[tuple[float, float]]):
        indices = [nearest_index(target_x, target_y) for target_x, target_y in targets]
        for i0, i1 in itertools.pairwise(indices):
            if i0 is None or i1 is None or i0 == i1:
                continue
            if float(np.linalg.norm(vertices[i1] - vertices[i0])) > max_jump:
                continue
            segments.append((i0, i1))

    for y_frac in (-0.28, 0.0, 0.28):
        add_polyline([(x_frac * length, y_frac * width) for x_frac in np.linspace(-0.42, 0.42, 19)])
    for x_frac in (-0.24, 0.02, 0.28):
        add_polyline([(x_frac * length, y_frac * width) for y_frac in np.linspace(-0.36, 0.36, 13)])

    return np.asarray(segments, dtype=np.int32).reshape((-1, 2))


def _panel_box_inertia(vertices: np.ndarray, mass: float) -> wp.mat33:
    extents = np.maximum(np.ptp(vertices, axis=0).astype(np.float64), 1.0e-4)
    ix = mass * (extents[1] * extents[1] + extents[2] * extents[2]) / 12.0
    iy = mass * (extents[0] * extents[0] + extents[2] * extents[2]) / 12.0
    iz = mass * (extents[0] * extents[0] + extents[1] * extents[1]) / 12.0
    return wp.mat33(float(ix), 0.0, 0.0, 0.0, float(iy), 0.0, 0.0, 0.0, float(iz))


def _project_out_rigid_modes(points: np.ndarray, phi: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    modes = np.asarray(phi, dtype=np.float64).copy()
    weights = np.ones(points.shape[0], dtype=np.float64)
    total_mass = float(np.sum(weights))
    center = np.sum(points * weights[:, None], axis=0) / total_mass
    rel = points - center
    inertia = np.zeros((3, 3), dtype=np.float64)
    for r, w in zip(rel, weights, strict=True):
        inertia += w * ((float(np.dot(r, r)) * np.eye(3)) - np.outer(r, r))
    inertia += 1.0e-9 * np.eye(3)

    for mode in range(modes.shape[1]):
        u = modes[:, mode, :]
        translation = np.sum(u * weights[:, None], axis=0) / total_mass
        v = u - translation
        angular_momentum = np.sum(weights[:, None] * np.cross(rel, v), axis=0)
        omega = np.linalg.solve(inertia, angular_momentum)
        modes[:, mode, :] = u - translation - np.cross(omega[None, :], rel)

    return modes.astype(np.float32)


def _build_panel_basis(vertices: np.ndarray) -> newton.ModalBasis:
    x = vertices[:, 0]
    y = vertices[:, 1]
    length = max(float(np.max(x) - np.min(x)), 1.0e-6)
    width = max(float(np.max(y) - np.min(y)), 1.0e-6)
    xi = 2.0 * x / length
    eta = 2.0 * y / width
    mode_count = 6
    phi = np.zeros((vertices.shape[0], mode_count, 3), dtype=np.float32)

    phi[:, 0, 2] = 1.0 - xi * xi
    phi[:, 1, 2] = 1.0 - eta * eta
    phi[:, 2, 2] = xi * eta
    phi[:, 3, 2] = np.sin(math.pi * (xi + 1.0)) * (1.0 - 0.25 * eta * eta)
    phi[:, 4, 0] = 0.18 * xi * (1.0 - eta * eta)
    phi[:, 4, 2] = -0.20 * (1.0 - xi * xi)
    phi[:, 5, 1] = 0.16 * eta * (1.0 - xi * xi)
    phi[:, 5, 2] = 0.16 * xi * eta

    phi = _project_out_rigid_modes(vertices, phi)
    max_abs = np.max(np.linalg.norm(phi, axis=2), axis=0)
    phi /= np.maximum(max_abs.reshape((1, mode_count, 1)), 1.0e-6)

    return newton.ModalBasis(
        sample_points=vertices,
        sample_phi=phi,
        mode_mass=np.full(mode_count, 0.18, dtype=np.float32),
        mode_stiffness=np.array([8200.0, 9800.0, 7000.0, 11500.0, 15000.0, 14000.0], dtype=np.float32),
        mode_damping=np.array([9.0, 9.5, 8.0, 10.0, 12.0, 11.0], dtype=np.float32),
        label="ur10_car_panel_procedural_modes",
    )


class Example:
    def __init__(self, viewer, args):
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_substeps = 4
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.sim_time = 0.0
        self.step_count = 0

        self.viewer = viewer
        self.args = args
        self.show_elastic_strain = True
        self.show_panel_deflection_guides = True
        self.panel_deflection_guide_scale = 4.0
        self.max_mode_abs = 0.0
        self.max_panel_deflection = 0.0
        self.max_attachment_residual = 0.0
        self.min_panel_z = float("inf")

        panel_vertices, panel_indices = _load_panel_mesh(args)
        self.panel_vertices = panel_vertices
        self.panel_indices = panel_indices
        self.panel_rib_segments = _build_panel_rib_segments(panel_vertices)
        self.panel_mass = 5.0
        panel_basis = _build_panel_basis(panel_vertices)
        self.mode_count = panel_basis.mode_count
        panel_extents = np.ptp(panel_vertices, axis=0)
        print(
            f"UR10 elastic panel: {self.mode_count} modes, {panel_vertices.shape[0]} vertices, "
            f"{panel_indices.size // 3} triangles, extents {panel_extents[0]:.2f} x {panel_extents[1]:.2f} m"
        )

        builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
        newton.solvers.SolverMuJoCo.register_custom_attributes(builder)
        asset_path = newton.utils.download_asset("universal_robots_ur10")
        robot_mount_height = 1.65
        builder.add_usd(
            str(asset_path / "usd" / "ur10_instanceable.usda"),
            xform=wp.transform(wp.vec3(0.0, 0.0, robot_mount_height), wp.quat_identity()),
            collapse_fixed_joints=False,
            enable_self_collisions=False,
            hide_collision_shapes=True,
        )
        for body in range(len(builder.body_flags)):
            builder.body_flags[body] = int(newton.BodyFlags.KINEMATIC)
        self.ee_body = builder.body_label.index("/ur10/ee_link")

        shape_cfg = newton.ModelBuilder.ShapeConfig()
        shape_cfg.density = 0.0
        shape_cfg.has_shape_collision = False
        shape_cfg.has_particle_collision = False
        builder.add_shape_cylinder(
            -1,
            xform=wp.transform(wp.vec3(0.0, 0.0, 0.5 * robot_mount_height), wp.quat_identity()),
            radius=0.095,
            half_height=0.5 * robot_mount_height,
            cfg=shape_cfg,
            label="ur10_pedestal",
        )

        self.panel = builder.add_body_elastic(
            xform=wp.transform(wp.vec3(0.68, -0.38, 1.12), _quat_from_axis_angle((0.0, 1.0, 0.0), -0.18)),
            mass=self.panel_mass,
            inertia=_panel_box_inertia(panel_vertices, self.panel_mass),
            modal_basis=panel_basis,
            label="elastic_car_panel",
        )
        builder.body_flags[self.panel] = int(newton.BodyFlags.KINEMATIC)
        builder.add_shape_mesh(
            self.panel,
            mesh=newton.Mesh(panel_vertices, panel_indices, compute_inertia=False),
            cfg=shape_cfg,
            label="elastic_car_panel_mesh",
        )

        gripper_angle = -0.18
        gripper_q = _quat_from_axis_angle((0.0, 1.0, 0.0), gripper_angle)
        gripper_q_np = _axis_angle_np((0.0, 1.0, 0.0), gripper_angle)
        gripper_origin_np = np.array([0.28, 0.0, 0.0], dtype=np.float32)
        gripper_hub_np = gripper_origin_np + np.array([-0.06, 0.0, 0.0], dtype=np.float32)
        panel_length = max(float(panel_extents[0]), 1.0e-6)
        panel_width = max(float(panel_extents[1]), 1.0e-6)

        self.panel_attachment_locals = tuple(
            _nearest_panel_surface_point(panel_vertices, x, y)
            for x, y in (
                (-0.28 * panel_length, -0.28 * panel_width),
                (-0.28 * panel_length, 0.28 * panel_width),
                (-0.08 * panel_length, -0.28 * panel_width),
                (-0.08 * panel_length, 0.28 * panel_width),
            )
        )
        self.attachment_joints: list[int] = []
        builder.add_shape_cylinder(
            self.ee_body,
            xform=wp.transform(wp.vec3(*gripper_hub_np), gripper_q),
            radius=0.055,
            half_height=0.016,
            cfg=shape_cfg,
            label="panel_gripper_hub",
        )
        for i, local in enumerate(self.panel_attachment_locals):
            parent_local = gripper_origin_np + _quat_rotate_np(gripper_q_np, local)
            cup_center = parent_local + _quat_rotate_np(gripper_q_np, np.array([0.0, 0.0, 0.012], dtype=np.float32))
            strut_xform, strut_half_height = _cylinder_xform_between(gripper_hub_np, cup_center)
            builder.add_shape_cylinder(
                self.ee_body,
                xform=strut_xform,
                radius=0.012,
                half_height=strut_half_height,
                cfg=shape_cfg,
                label=f"panel_gripper_strut_{i}",
            )
            builder.add_shape_cylinder(
                self.ee_body,
                xform=wp.transform(wp.vec3(*cup_center), gripper_q),
                radius=0.038,
                half_height=0.012,
                cfg=shape_cfg,
                label=f"panel_vacuum_cup_{i}",
            )
            self.attachment_joints.append(
                builder.add_joint_fixed(
                    parent=self.ee_body,
                    child=self.panel,
                    parent_xform=wp.transform(wp.vec3(*parent_local), gripper_q),
                    child_xform=wp.transform(wp.vec3(*local), wp.quat_identity()),
                    label=f"ur10_gripper_to_panel_{i}",
                )
            )

        builder.color()
        self.model = builder.finalize()
        self.device = self.model.device
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.contacts = None

        labels = list(self.model.joint_label)
        self.robot_joint_ids = [
            labels.index("/ur10/base_link/shoulder_pan_joint"),
            labels.index("/ur10/shoulder_link/shoulder_lift_joint"),
            labels.index("/ur10/upper_arm_link/elbow_joint"),
            labels.index("/ur10/forearm_link/wrist_1_joint"),
            labels.index("/ur10/wrist_1_link/wrist_2_joint"),
            labels.index("/ur10/wrist_2_link/wrist_3_joint"),
        ]
        self.robot_q_starts = [int(self.model.joint_q_start.numpy()[joint]) for joint in self.robot_joint_ids]
        self.robot_qd_starts = [int(self.model.joint_qd_start.numpy()[joint]) for joint in self.robot_joint_ids]
        elastic_index = int(self.model.body_elastic_index.numpy()[self.panel])
        self.elastic_joint = int(self.model.elastic_joint.numpy()[elastic_index])
        self.elastic_q_start = int(self.model.joint_q_start.numpy()[self.elastic_joint])
        self.elastic_qd_start = int(self.model.joint_qd_start.numpy()[self.elastic_joint])
        self._joint_f = self.control.joint_f.numpy()
        self._base_robot_q = np.array([0.48, -1.05, 1.18, -0.98, -1.50, -0.12], dtype=np.float32)
        self._last_robot_q = self._base_robot_q.copy()
        self._modal_pressure_gain = self._build_modal_pressure_gain(panel_basis)
        self._set_robot_trajectory(0.0)
        self._align_panel_to_gripper(reset_modes=True)

        self.solver = newton.solvers.SolverVBD(
            self.model,
            iterations=48,
            rigid_joint_linear_k_start=3.0e5,
            rigid_joint_angular_k_start=7.5e4,
            rigid_joint_linear_ke=3.0e6,
            rigid_joint_angular_ke=7.5e5,
            rigid_joint_linear_kd=1.0e-3,
        )
        wp.copy(self.solver.body_q_prev, self.state_0.body_q)

        self.viewer.set_model(self.model)
        self.viewer.show_elastic_strain = True
        self.viewer.elastic_strain_color_max = 0.035
        self._frame_camera()

    def _build_modal_pressure_gain(self, basis: newton.ModalBasis) -> np.ndarray:
        sample_points = basis.sample_points
        length = max(float(np.max(sample_points[:, 0]) - np.min(sample_points[:, 0])), 1.0e-6)
        width = max(float(np.max(sample_points[:, 1]) - np.min(sample_points[:, 1])), 1.0e-6)
        xi = np.clip(2.0 * sample_points[:, 0] / length, -1.0, 1.0)
        eta = np.clip(2.0 * sample_points[:, 1] / width, -1.0, 1.0)
        free_edge_profile = np.clip(0.5 * (xi + 1.0), 0.0, 1.0) ** 1.4
        lateral_profile = 1.0 - 0.25 * eta * eta
        area_per_sample = length * width / max(basis.sample_count, 1)
        force_dir = np.array([0.0, 0.0, -1.0], dtype=np.float32)
        generalized = area_per_sample * np.einsum(
            "smc,s,c->m", basis.sample_phi, free_edge_profile * lateral_profile, force_dir
        )
        return generalized.astype(np.float32)

    def _align_panel_to_gripper(self, *, reset_modes: bool = False):
        joint = self.attachment_joints[0]
        body_q = self.state_0.body_q.numpy()
        parent_q = np.asarray(body_q[self.ee_body], dtype=np.float32)
        parent_pos = parent_q[:3]
        parent_rot = parent_q[3:7]
        joint_xp = self.model.joint_X_p.numpy()[joint]
        joint_xc = self.model.joint_X_c.numpy()[joint]
        parent_local = np.asarray(joint_xp[:3], dtype=np.float32)
        parent_joint_rot = np.asarray(joint_xp[3:7], dtype=np.float32)
        child_local = np.asarray(joint_xc[:3], dtype=np.float32)
        child_joint_rot = np.asarray(joint_xc[3:7], dtype=np.float32)

        panel_rot = _quat_mul(_quat_mul(parent_rot, parent_joint_rot), _quat_conjugate(child_joint_rot))
        panel_rot /= max(float(np.linalg.norm(panel_rot)), 1.0e-8)
        panel_pos = parent_pos + _quat_rotate_np(parent_rot, parent_local) - _quat_rotate_np(panel_rot, child_local)

        joint_q = self.state_0.joint_q.numpy()
        joint_qd = self.state_0.joint_qd.numpy()
        joint_q[self.elastic_q_start : self.elastic_q_start + 3] = panel_pos
        joint_q[self.elastic_q_start + 3 : self.elastic_q_start + 7] = panel_rot
        if reset_modes:
            joint_q[self.elastic_q_start + 7 : self.elastic_q_start + 7 + self.mode_count] = 0.0
            joint_qd[self.elastic_qd_start : self.elastic_qd_start + 6 + self.mode_count] = 0.0
        self.state_0.joint_q.assign(joint_q)
        self.state_0.joint_qd.assign(joint_qd)
        newton.eval_fk(self.model, self.state_0.joint_q, self.state_0.joint_qd, self.state_0)

    def _mode_values(self) -> np.ndarray:
        return self.state_0.joint_q.numpy()[self.elastic_q_start + 7 : self.elastic_q_start + 7 + self.mode_count]

    def _panel_world_vertices(self) -> np.ndarray:
        local_vertices = elastic_shape_deformed_vertices(self.model, self.state_0)
        return self._panel_points_to_world(local_vertices)

    def _panel_points_to_world(self, local_points: np.ndarray) -> np.ndarray:
        panel_xform = np.asarray(self.state_0.body_q.numpy()[self.panel], dtype=np.float32)
        panel_pos = panel_xform[:3]
        panel_rot = panel_xform[3:7]
        return panel_pos.reshape((1, 3)) + _quat_rotate_points_np(panel_rot, local_points)

    def _frame_camera(self):
        if not hasattr(self.viewer, "camera"):
            return

        body_positions = self.state_0.body_q.numpy()[:, :3]
        panel_vertices = self._panel_world_vertices()
        points = np.vstack((body_positions, panel_vertices))
        bounds_min = np.min(points, axis=0)
        bounds_max = np.max(points, axis=0)
        center = 0.5 * (bounds_min + bounds_max)
        extent = max(float(np.max(bounds_max - bounds_min)), 1.0)

        yaw = 112.0
        pitch = -14.0
        pitch_rad = math.radians(pitch)
        yaw_rad = math.radians(yaw)
        front = np.array(
            [
                math.cos(yaw_rad) * math.cos(pitch_rad),
                math.sin(yaw_rad) * math.cos(pitch_rad),
                math.sin(pitch_rad),
            ],
            dtype=np.float32,
        )
        front /= max(float(np.linalg.norm(front)), 1.0e-8)
        distance = 1.65 * extent / (2.0 * math.tan(math.radians(self.viewer.camera.fov) * 0.5))
        camera_pos = center - front * distance
        camera_pos[2] += 0.10
        self.viewer.set_camera(wp.vec3(*camera_pos), pitch, yaw)

    def _robot_trajectory(self, t: float) -> tuple[np.ndarray, np.ndarray]:
        phase = 2.0 * math.pi * 0.14 * t
        amplitudes = np.array([0.38, 0.05, 0.06, 0.05, 0.08, 0.14], dtype=np.float32)
        phase_offsets = np.array([0.0, 0.8, 1.4, 2.1, 2.8, 3.4], dtype=np.float32)
        q = self._base_robot_q + amplitudes * np.sin(phase + phase_offsets)
        qd = amplitudes * (2.0 * math.pi * 0.14) * np.cos(phase + phase_offsets)
        return q.astype(np.float32), qd.astype(np.float32)

    def _set_robot_trajectory(self, t: float):
        q_target, qd_target = self._robot_trajectory(t)
        joint_q = self.state_0.joint_q.numpy()
        joint_qd = self.state_0.joint_qd.numpy()
        for value, velocity, q_start, qd_start in zip(
            q_target, qd_target, self.robot_q_starts, self.robot_qd_starts, strict=True
        ):
            joint_q[q_start] = float(value)
            joint_qd[qd_start] = float(velocity)
        self.state_0.joint_q.assign(joint_q)
        self.state_0.joint_qd.assign(joint_qd)
        newton.eval_fk(
            self.model,
            self.state_0.joint_q,
            self.state_0.joint_qd,
            self.state_0,
            body_flag_filter=newton.BodyFlags.KINEMATIC,
        )
        self._last_robot_q = q_target

    def _set_panel_forces(self):
        pulse = math.sin(2.0 * math.pi * 0.42 * self.sim_time)
        pressure = 37200.0 * pulse + 13200.0 * math.sin(2.0 * math.pi * 0.73 * self.sim_time + 0.35)
        self._joint_f[:] = 0.0
        modal_start = self.elastic_qd_start + 6
        self._joint_f[modal_start : modal_start + self.mode_count] = self._modal_pressure_gain * pressure
        self.control.joint_f.assign(self._joint_f)

    def _update_metrics(self):
        modes = self._mode_values()
        self.max_mode_abs = max(self.max_mode_abs, float(np.max(np.abs(modes))))
        deformed = elastic_shape_deformed_vertices(self.model, self.state_0)
        rest = self.model.elastic_shape_vertex_local.numpy()[: deformed.shape[0]]
        self.max_panel_deflection = max(
            self.max_panel_deflection, float(np.max(np.linalg.norm(deformed - rest, axis=1)))
        )
        self.min_panel_z = min(self.min_panel_z, float(np.min(self._panel_world_vertices()[:, 2])))
        residuals = []
        for joint in self.attachment_joints:
            residuals.append(
                float(
                    np.linalg.norm(
                        joint_endpoint_world(self.model, self.state_0, joint, "parent")
                        - joint_endpoint_world(self.model, self.state_0, joint, "child")
                    )
                )
            )
        self.max_attachment_residual = max(self.max_attachment_residual, *residuals)

    def _log_panel_deflection_guides(self):
        if not self.show_panel_deflection_guides or self.panel_rib_segments.size == 0:
            self.viewer.log_lines("/diagnostics/ur10_panel/rest_ribs", None, None, None)
            self.viewer.log_lines("/diagnostics/ur10_panel/deformed_ribs", None, None, None)
            self.viewer.log_lines("/diagnostics/ur10_panel/deflection_whiskers", None, None, None)
            return

        deformed_local = elastic_shape_deformed_vertices(self.model, self.state_0).astype(np.float32)
        rest_local = self.model.elastic_shape_vertex_local.numpy()[: deformed_local.shape[0]]
        scaled_local = rest_local + self.panel_deflection_guide_scale * (deformed_local - rest_local)
        rest_world = self._panel_points_to_world(rest_local)
        scaled_world = self._panel_points_to_world(scaled_local)

        segment_starts = self.panel_rib_segments[:, 0]
        segment_ends = self.panel_rib_segments[:, 1]
        rest_starts = wp.array(rest_world[segment_starts], dtype=wp.vec3, device=self.viewer.device)
        rest_ends = wp.array(rest_world[segment_ends], dtype=wp.vec3, device=self.viewer.device)
        scaled_starts = wp.array(scaled_world[segment_starts], dtype=wp.vec3, device=self.viewer.device)
        scaled_ends = wp.array(scaled_world[segment_ends], dtype=wp.vec3, device=self.viewer.device)

        self.viewer.log_lines(
            "/diagnostics/ur10_panel/rest_ribs",
            rest_starts,
            rest_ends,
            (0.55, 0.58, 0.62),
            width=0.004,
        )
        self.viewer.log_lines(
            "/diagnostics/ur10_panel/deformed_ribs",
            scaled_starts,
            scaled_ends,
            (1.0, 0.86, 0.10),
            width=0.007,
        )

        whisker_indices = np.unique(self.panel_rib_segments.reshape((-1,)))[::4]
        whisker_starts = wp.array(rest_world[whisker_indices], dtype=wp.vec3, device=self.viewer.device)
        whisker_ends = wp.array(scaled_world[whisker_indices], dtype=wp.vec3, device=self.viewer.device)
        self.viewer.log_lines(
            "/diagnostics/ur10_panel/deflection_whiskers",
            whisker_starts,
            whisker_ends,
            (1.0, 0.20, 0.85),
            width=0.003,
        )

    def simulate(self):
        for _substep in range(self.sim_substeps):
            self.state_0.clear_forces()
            self.viewer.apply_forces(self.state_0)
            self._set_robot_trajectory(self.sim_time)
            self._align_panel_to_gripper(reset_modes=False)
            self._set_panel_forces()
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0
            self.sim_time += self.sim_dt

    def step(self):
        self.simulate()
        self.step_count += 1
        self._update_metrics()

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.log_lines("/model/elastic_bodies/centerlines", None, None, None)
        self.viewer.log_points("/model/elastic_bodies/samples", None)
        self.viewer.log_points("/model/elastic_bodies/endpoints", None)
        self._log_panel_deflection_guides()
        self.viewer.end_frame()

    def test_final(self):
        if not np.isfinite(self.state_0.body_q.numpy()).all():
            raise AssertionError("UR10 elastic panel body transforms contain non-finite values")
        if self.model.elastic_mode_count.numpy()[0] != self.mode_count:
            raise AssertionError("UR10 elastic panel has the wrong number of retained modes")
        if self.max_mode_abs < 1.0e-3:
            raise AssertionError(f"UR10 elastic panel modes were not excited enough: {self.max_mode_abs}")
        if self.max_mode_abs > 0.35:
            raise AssertionError(f"UR10 elastic panel modes are out of range: {self._mode_values()}")
        if self.max_panel_deflection < 1.0e-3:
            raise AssertionError(f"UR10 elastic panel deformation too small: {self.max_panel_deflection}")
        if self.min_panel_z < 1.0:
            raise AssertionError(f"UR10 elastic panel dipped below the raised workcell clearance: {self.min_panel_z}")
        if self.max_attachment_residual > 7.5e-2:
            raise AssertionError(f"UR10 elastic panel attachment residual too large: {self.max_attachment_residual}")

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()
        parser.add_argument(
            "--carhoods-dir",
            type=Path,
            default=None,
            help="Directory containing the downloaded CarHoods10k dataset.",
        )
        parser.add_argument(
            "--panel-mesh",
            type=Path,
            default=None,
            help="Optional STL panel mesh to use instead of the procedural hood-frame fallback.",
        )
        return parser


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)
    example = Example(viewer, args)
    newton.examples.run(example, args)
