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
# Example Basic Reduced Elastic Matrix ROM
#
# Demonstrates a Simscape-style reduced flexible body workflow on a robot
# handling beam. A front-facing perforated crossbar is converted from nodal
# mass, stiffness, and damping matrices into sampled modal shape functions
# with ModalGeneratorFEM. A robot wrist clamps the left edge, two prismatic
# gripper carriages attach to solid bottom-edge material, rigid arms reach
# toward the camera, and a distributed picked-part load is projected into modal
# forces with Phi.T @ f.
#
# Command: python -m newton.examples basic_reduced_elastic_matrix_rom
#
###########################################################################

import numpy as np
import warp as wp

import newton
import newton.examples
from newton.examples.basic._reduced_elastic import (
    elastic_shape_deformed_vertices,
    joint_endpoint_world,
    transform_point,
)


def _add_matrix_spring(stiffness: np.ndarray, node_a: int, node_b: int, stiffness_xyz: np.ndarray):
    for axis in range(3):
        value = float(stiffness_xyz[axis])
        ia = 3 * node_a + axis
        ib = 3 * node_b + axis
        stiffness[ia, ia] += value
        stiffness[ib, ib] += value
        stiffness[ia, ib] -= value
        stiffness[ib, ia] -= value


def _smoothstep(t: float, t0: float, t1: float) -> float:
    if t <= t0:
        return 0.0
    if t >= t1:
        return 1.0
    s = (t - t0) / (t1 - t0)
    return s * s * (3.0 - 2.0 * s)


def _make_handling_crossbar(
    length: float,
    height: float,
    thickness: float,
    nx: int,
    nz: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[tuple[int, int, int, int]]]:
    x = np.linspace(-0.5 * length, 0.5 * length, nx + 1, dtype=np.float32)
    z = np.linspace(-0.5 * height, 0.5 * height, nz + 1, dtype=np.float32)

    def is_cutout(cx: float, cz: float) -> bool:
        if abs(cz) > 0.28 * height:
            return False
        for sx, rx in (
            (-0.30 * length, 0.09 * length),
            (-0.04 * length, 0.11 * length),
            (0.24 * length, 0.10 * length),
        ):
            rz = 0.18 * height
            if ((cx - sx) / rx) ** 2 + (cz / rz) ** 2 < 1.0:
                return True
        return False

    active_cells: list[tuple[int, int]] = []
    used_grid_nodes: set[tuple[int, int]] = set()
    for i in range(nx):
        for j in range(nz):
            cx = 0.5 * (float(x[i]) + float(x[i + 1]))
            cz = 0.5 * (float(z[j]) + float(z[j + 1]))
            if is_cutout(cx, cz):
                continue
            active_cells.append((i, j))
            used_grid_nodes.update(((i, j), (i + 1, j), (i + 1, j + 1), (i, j + 1)))

    node_lookup: dict[tuple[int, int], int] = {}
    nodes: list[tuple[float, float, float]] = []
    for key in sorted(used_grid_nodes):
        node_lookup[key] = len(nodes)
        nodes.append((float(x[key[0]]), 0.0, float(z[key[1]])))

    cells: list[tuple[int, int, int, int]] = []
    active_set = set(active_cells)
    for i, j in active_cells:
        cells.append(
            (node_lookup[(i, j)], node_lookup[(i + 1, j)], node_lookup[(i + 1, j + 1)], node_lookup[(i, j + 1)])
        )

    vertices: list[tuple[float, float, float]] = []
    vertex_node: list[int] = []
    front: dict[int, int] = {}
    back: dict[int, int] = {}
    for node, point in enumerate(nodes):
        px, _py, pz = point
        front[node] = len(vertices)
        vertices.append((px, -0.5 * thickness, pz))
        vertex_node.append(node)
        back[node] = len(vertices)
        vertices.append((px, 0.5 * thickness, pz))
        vertex_node.append(node)

    indices: list[int] = []

    def add_quad(a: int, b: int, c: int, d: int) -> None:
        indices.extend((a, b, c, a, c, d))

    for (i, j), (a, b, c, d) in zip(active_cells, cells, strict=True):
        add_quad(front[a], front[b], front[c], front[d])
        add_quad(back[a], back[d], back[c], back[b])

        edge_specs = (
            ((i, j - 1), a, b),
            ((i + 1, j), b, c),
            ((i, j + 1), d, c),
            ((i - 1, j), a, d),
        )
        for neighbor, n0, n1 in edge_specs:
            if neighbor not in active_set:
                add_quad(front[n0], front[n1], back[n1], back[n0])

    return (
        np.asarray(nodes, dtype=np.float32),
        np.asarray(vertices, dtype=np.float32),
        np.asarray(indices, dtype=np.int32),
        np.asarray(vertex_node, dtype=np.int32),
        cells,
    )


def _assemble_plate_matrices(
    nodes: np.ndarray,
    cells: list[tuple[int, int, int, int]],
    density: float,
    thickness: float,
    in_plane_stiffness: float,
    out_of_plane_stiffness: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    dof_count = 3 * int(nodes.shape[0])
    mass = np.zeros((dof_count, dof_count), dtype=np.float64)
    stiffness = np.zeros((dof_count, dof_count), dtype=np.float64)

    cell_area = 0.0
    nodal_area = np.zeros(nodes.shape[0], dtype=np.float64)
    edge_pairs: set[tuple[int, int]] = set()
    diagonal_pairs: set[tuple[int, int]] = set()
    for a, b, c, d in cells:
        p = nodes[[a, b, c, d]]
        area = 0.5 * np.linalg.norm(np.cross(p[1] - p[0], p[2] - p[0]))
        area += 0.5 * np.linalg.norm(np.cross(p[2] - p[0], p[3] - p[0]))
        cell_area += float(area)
        for node in (a, b, c, d):
            nodal_area[node] += 0.25 * float(area)
        for pair in ((a, b), (b, c), (c, d), (d, a)):
            edge_pairs.add(tuple(sorted(pair)))
        for pair in ((a, c), (b, d)):
            diagonal_pairs.add(tuple(sorted(pair)))

    total_mass = max(float(density) * float(thickness) * cell_area, 1.0e-6)
    for node, area in enumerate(nodal_area):
        lumped = max(total_mass * area / max(cell_area, 1.0e-9), 1.0e-5)
        for axis in range(3):
            mass[3 * node + axis, 3 * node + axis] = lumped

    for node_a, node_b in sorted(edge_pairs):
        edge_length = max(float(np.linalg.norm(nodes[node_b] - nodes[node_a])), 1.0e-6)
        _add_matrix_spring(
            stiffness,
            node_a,
            node_b,
            np.array(
                [
                    in_plane_stiffness / edge_length,
                    out_of_plane_stiffness / edge_length,
                    in_plane_stiffness / edge_length,
                ]
            ),
        )
    for node_a, node_b in sorted(diagonal_pairs):
        edge_length = max(float(np.linalg.norm(nodes[node_b] - nodes[node_a])), 1.0e-6)
        _add_matrix_spring(
            stiffness,
            node_a,
            node_b,
            np.array(
                [
                    0.28 * in_plane_stiffness / edge_length,
                    0.28 * out_of_plane_stiffness / edge_length,
                    0.28 * in_plane_stiffness / edge_length,
                ]
            ),
        )

    damping = 0.004 * mass + 0.0008 * stiffness
    return mass, stiffness, damping


def _select_x_patch_nodes(nodes: np.ndarray, x_center: float, x_columns: int, z_count: int) -> np.ndarray:
    x_values = np.unique(np.round(nodes[:, 0], decimals=7))
    columns = sorted((float(x) for x in x_values), key=lambda value: abs(value - x_center))[:x_columns]
    z_min = float(np.min(nodes[:, 2]))
    z_max = float(np.max(nodes[:, 2]))
    z_targets = np.linspace(0.72 * z_min, 0.72 * z_max, z_count)

    selected: list[int] = []
    for x_value in sorted(columns):
        column = np.where(np.abs(nodes[:, 0] - x_value) <= 1.0e-6)[0]
        for z_value in z_targets:
            node = int(column[np.argmin(np.abs(nodes[column, 2] - float(z_value)))])
            if node not in selected:
                selected.append(node)

    if not selected:
        raise ValueError(f"No nodes selected near x={x_center}")

    selected.sort(key=lambda node: (abs(float(nodes[node, 0]) - x_center), abs(float(nodes[node, 2]))))
    return np.asarray(selected, dtype=np.int32)


def _select_lower_patch_nodes(nodes: np.ndarray, x_center: float, x_columns: int = 2, z_rows: int = 2) -> np.ndarray:
    x_values = np.unique(np.round(nodes[:, 0], decimals=7))
    z_values = np.unique(np.round(nodes[:, 2], decimals=7))
    columns = sorted((float(x) for x in x_values), key=lambda value: abs(value - x_center))[:x_columns]
    rows = sorted(float(z) for z in z_values)[:z_rows]

    selected: list[int] = []
    for x_value in sorted(columns):
        for z_value in rows:
            candidates = np.where(
                (np.abs(nodes[:, 0] - x_value) <= 1.0e-6) & (np.abs(nodes[:, 2] - z_value) <= 1.0e-6)
            )[0]
            if candidates.shape[0] == 0:
                continue
            node = int(candidates[0])
            if node not in selected:
                selected.append(node)

    if not selected:
        raise ValueError(f"No lower patch nodes selected near x={x_center}")

    selected.sort(key=lambda node: (abs(float(nodes[node, 0]) - x_center), float(nodes[node, 2])))
    return np.asarray(selected, dtype=np.int32)


def _select_bottom_attachment_node(nodes: np.ndarray, x_center: float) -> int:
    z_min = float(np.min(nodes[:, 2]))
    bottom = np.where(np.abs(nodes[:, 2] - z_min) <= 1.0e-6)[0]
    if bottom.shape[0] == 0:
        raise ValueError("No bottom-edge nodes found")
    return int(bottom[np.argmin(np.abs(nodes[bottom, 0] - x_center))])


def _make_centerline_columns(nodes: np.ndarray) -> tuple[np.ndarray, list[np.ndarray]]:
    x_values = np.unique(np.round(nodes[:, 0], decimals=7))
    columns: list[np.ndarray] = []
    centers: list[float] = []
    for x_value in x_values:
        column = np.where(np.abs(nodes[:, 0] - float(x_value)) <= 1.0e-6)[0].astype(np.int32)
        if column.shape[0] == 0:
            continue
        centers.append(float(np.mean(nodes[column, 0])))
        columns.append(column)
    return np.asarray(centers, dtype=np.float32), columns


def _build_static_reference(
    stiffness_matrix: np.ndarray,
    modal_matrix: np.ndarray,
    mode_stiffness: np.ndarray,
    fixed_nodes: np.ndarray,
    load_nodes: np.ndarray,
    total_force_z: float,
) -> dict[str, np.ndarray | float]:
    dof_count = int(stiffness_matrix.shape[0])
    force = np.zeros(dof_count, dtype=np.float64)
    per_node_force = float(total_force_z) / float(load_nodes.shape[0])
    for node in load_nodes:
        force[3 * int(node) + 2] += per_node_force

    fixed_dofs: list[int] = []
    for node in fixed_nodes:
        fixed_dofs.extend((3 * int(node), 3 * int(node) + 1, 3 * int(node) + 2))
    free_mask = np.ones(dof_count, dtype=bool)
    free_mask[np.asarray(fixed_dofs, dtype=np.int32)] = False
    free = np.nonzero(free_mask)[0]

    u_full = np.zeros(dof_count, dtype=np.float64)
    k_free = stiffness_matrix[np.ix_(free, free)]
    f_free = force[free]
    u_full[free] = np.linalg.solve(k_free, f_free)

    modal_force = modal_matrix.T @ force
    q_static = modal_force / np.maximum(mode_stiffness.astype(np.float64), 1.0e-12)
    u_rom = modal_matrix @ q_static
    load_z_dofs = np.asarray([3 * int(node) + 2 for node in load_nodes], dtype=np.int32)
    full_sag = float(np.mean(u_full[load_z_dofs]))
    rom_sag = float(np.mean(u_rom[load_z_dofs]))
    relative_error = abs(rom_sag - full_sag) / max(abs(full_sag), 1.0e-9)

    return {
        "force": force,
        "modal_force": modal_force,
        "q_static": q_static,
        "u_full": u_full,
        "u_rom": u_rom,
        "full_sag": full_sag,
        "rom_sag": rom_sag,
        "relative_error": relative_error,
    }


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

        self.length = 1.32
        self.height = 0.22
        self.thickness = 0.035
        self.mode_count = 8
        self.max_joint_residual = 0.0
        self.max_mode_abs = 0.0
        self.max_slide_motion = 0.0
        self.max_wrist_motion = 0.0
        self.max_sag_abs = 0.0
        self.max_centerline_deflection = 0.0

        nodes, surface_vertices, surface_indices, surface_node_indices, cells = _make_handling_crossbar(
            self.length, self.height, self.thickness, nx=32, nz=10
        )
        self.nodes = nodes
        self.deflection_x, self.deflection_columns = _make_centerline_columns(nodes)
        self.deflection_visual_scale = 8.0
        self.deflection_overlay_y = -0.5 * self.thickness - 0.035
        self.show_straight_deflection = True
        self.support_nodes = _select_x_patch_nodes(nodes, -0.5 * self.length, x_columns=2, z_count=5)
        self.left_grip_x = -0.18 * self.length
        self.right_grip_x = 0.39 * self.length
        self.left_attachment_node = _select_bottom_attachment_node(nodes, self.left_grip_x)
        self.right_attachment_node = _select_bottom_attachment_node(nodes, self.right_grip_x)
        self.left_grip_local = np.array(nodes[self.left_attachment_node], dtype=np.float32)
        self.right_grip_local = np.array(nodes[self.right_attachment_node], dtype=np.float32)
        self.left_grip_x = float(self.left_grip_local[0])
        self.right_grip_x = float(self.right_grip_local[0])
        self.left_load_nodes = _select_lower_patch_nodes(nodes, self.left_grip_x, x_columns=2, z_rows=2)
        self.right_load_nodes = _select_lower_patch_nodes(nodes, self.right_grip_x, x_columns=2, z_rows=2)
        self.load_nodes = np.unique(np.concatenate((self.left_load_nodes, self.right_load_nodes))).astype(np.int32)

        mass_matrix, stiffness_matrix, damping_matrix = _assemble_plate_matrices(
            nodes,
            cells,
            density=1400.0,
            thickness=self.thickness,
            in_plane_stiffness=980.0,
            out_of_plane_stiffness=120.0,
        )
        sample_points = np.vstack((surface_vertices, nodes)).astype(np.float32)
        sample_node_indices = np.concatenate((surface_node_indices, np.arange(nodes.shape[0], dtype=np.int32)))
        matrix_generator = newton.ModalGeneratorFEM(
            node_positions=nodes,
            mass_matrix=mass_matrix,
            stiffness_matrix=stiffness_matrix,
            damping_matrix=damping_matrix,
            sample_points=sample_points,
            sample_node_indices=sample_node_indices,
            fixed_node_indices=self.support_nodes,
            mode_count=self.mode_count,
            label="handling_beam_matrix_rom",
        )
        matrix_basis = matrix_generator.build()
        self.modal_frequencies_hz = matrix_generator.frequencies
        self.modal_matrix = matrix_generator.modal_matrix
        self.mode_stiffness = np.asarray(matrix_basis.mode_stiffness, dtype=np.float64)
        self.part_weight = 150.0
        self.static_reference = _build_static_reference(
            stiffness_matrix,
            self.modal_matrix,
            self.mode_stiffness,
            self.support_nodes,
            self.load_nodes,
            total_force_z=-self.part_weight,
        )
        self.modal_load = np.asarray(self.static_reference["modal_force"], dtype=np.float32)
        self.reference_full_sag = float(self.static_reference["full_sag"])
        self.reference_rom_sag = float(self.static_reference["rom_sag"])
        self.reference_relative_error = float(self.static_reference["relative_error"])

        self.panel_origin = np.array([0.0, 0.0, 0.82], dtype=np.float32)
        self.support_local = np.array([-0.5 * self.length, 0.0, 0.0], dtype=np.float32)
        self.wrist_origin = self.panel_origin + self.support_local
        self.beam_anchor_z = float(np.min(nodes[:, 2]))
        self.gripper_drop = 0.18
        self.arm_reach = 0.22
        self.payload_origin = self.panel_origin + np.array(
            [
                0.5 * (self.left_grip_x + self.right_grip_x),
                -self.arm_reach,
                self.beam_anchor_z - self.gripper_drop - 0.08,
            ],
            dtype=np.float32,
        )

        builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
        inertia = wp.mat33(0.025, 0.0, 0.0, 0.0, 0.025, 0.0, 0.0, 0.0, 0.025)
        self.wrist = builder.add_body(
            xform=wp.transform(wp.vec3(*self.wrist_origin), wp.quat_identity()),
            mass=8.0,
            inertia=inertia,
            is_kinematic=True,
            label="robot_wrist_proxy",
        )
        self.j_wrist = len(builder.joint_type) - 1
        self.panel = builder.add_body_elastic(
            xform=wp.transform(wp.vec3(*self.panel_origin), wp.quat_identity()),
            mass=2.2,
            inertia=inertia,
            modal_basis=matrix_basis,
            label="matrix_rom_handling_beam",
        )

        gripper_inertia = wp.mat33(0.004, 0.0, 0.0, 0.0, 0.004, 0.0, 0.0, 0.0, 0.004)
        left_gripper_origin = self.panel_origin + np.array(
            [self.left_grip_x, -self.arm_reach, self.beam_anchor_z - self.gripper_drop], dtype=np.float32
        )
        right_gripper_origin = self.panel_origin + np.array(
            [self.right_grip_x, -self.arm_reach, self.beam_anchor_z - self.gripper_drop], dtype=np.float32
        )
        self.left_gripper = builder.add_body(
            xform=wp.transform(wp.vec3(*left_gripper_origin), wp.quat_identity()),
            mass=0.45,
            inertia=gripper_inertia,
            label="left_prismatic_gripper",
        )
        self.right_gripper = builder.add_body(
            xform=wp.transform(wp.vec3(*right_gripper_origin), wp.quat_identity()),
            mass=0.45,
            inertia=gripper_inertia,
            label="right_prismatic_gripper",
        )
        self.payload = builder.add_body(
            xform=wp.transform(wp.vec3(*self.payload_origin), wp.quat_identity()),
            mass=0.25,
            inertia=wp.mat33(0.01, 0.0, 0.0, 0.0, 0.01, 0.0, 0.0, 0.0, 0.01),
            label="picked_panel_visual_payload",
        )

        shape_cfg = newton.ModelBuilder.ShapeConfig()
        shape_cfg.density = 0.0
        shape_cfg.has_shape_collision = False
        shape_cfg.has_particle_collision = False
        builder.add_shape_box(
            self.wrist,
            xform=wp.transform(wp.vec3(-0.16, 0.035, 0.02), wp.quat_identity()),
            hx=0.17,
            hy=0.035,
            hz=0.055,
            cfg=shape_cfg,
            label="robot_arm_proxy",
        )
        builder.add_shape_box(self.wrist, hx=0.055, hy=0.045, hz=0.11, cfg=shape_cfg, label="wrist_flange")
        builder.add_shape_mesh(
            self.panel,
            mesh=newton.Mesh(surface_vertices, surface_indices, compute_inertia=False),
            cfg=shape_cfg,
            label="matrix_rom_handling_beam_mesh",
        )
        for gripper in (self.left_gripper, self.right_gripper):
            builder.add_shape_box(
                gripper,
                xform=wp.transform(wp.vec3(0.0, 0.5 * self.arm_reach, self.gripper_drop), wp.quat_identity()),
                hx=0.022,
                hy=0.5 * self.arm_reach,
                hz=0.016,
                cfg=shape_cfg,
                label="camera_facing_gripper_arm",
            )
            builder.add_shape_box(
                gripper,
                xform=wp.transform(wp.vec3(0.0, 0.0, 0.07), wp.quat_identity()),
                hx=0.018,
                hy=0.025,
                hz=0.095,
                cfg=shape_cfg,
                label="gripper_drop_link",
            )
            builder.add_shape_box(
                gripper,
                xform=wp.transform(wp.vec3(-0.035, 0.0, -0.02), wp.quat_identity()),
                hx=0.035,
                hy=0.024,
                hz=0.018,
                cfg=shape_cfg,
                label="left_jaw",
            )
            builder.add_shape_box(
                gripper,
                xform=wp.transform(wp.vec3(0.035, 0.0, -0.02), wp.quat_identity()),
                hx=0.035,
                hy=0.024,
                hz=0.018,
                cfg=shape_cfg,
                label="right_jaw",
            )
        builder.add_shape_box(
            self.payload,
            hx=0.5 * abs(self.right_grip_x - self.left_grip_x) + 0.16,
            hy=0.04,
            hz=0.035,
            cfg=shape_cfg,
            label="picked_car_panel_proxy",
        )

        self.support_joints: list[int] = []
        for i, node in enumerate(self.support_nodes):
            local = nodes[int(node)]
            self.support_joints.append(
                builder.add_joint_fixed(
                    parent=self.wrist,
                    child=self.panel,
                    parent_xform=wp.transform(wp.vec3(*(local - self.support_local)), wp.quat_identity()),
                    child_xform=wp.transform(wp.vec3(*local), wp.quat_identity()),
                    label=f"wrist_to_handling_beam_{i:02d}",
                )
            )

        self.j_left_slide = builder.add_joint_prismatic(
            parent=self.panel,
            child=self.left_gripper,
            axis=(0.0, 0.0, 1.0),
            parent_xform=wp.transform(wp.vec3(*self.left_grip_local), wp.quat_identity()),
            child_xform=wp.transform(wp.vec3(0.0, self.arm_reach, self.gripper_drop), wp.quat_identity()),
            target_pos=0.0,
            target_ke=2.0e5,
            target_kd=80.0,
            label="left_gripper_vertical_slide",
        )
        self.j_right_slide = builder.add_joint_prismatic(
            parent=self.panel,
            child=self.right_gripper,
            axis=(0.0, 0.0, 1.0),
            parent_xform=wp.transform(wp.vec3(*self.right_grip_local), wp.quat_identity()),
            child_xform=wp.transform(wp.vec3(0.0, self.arm_reach, self.gripper_drop), wp.quat_identity()),
            target_pos=0.0,
            target_ke=2.0e5,
            target_kd=80.0,
            label="right_gripper_vertical_slide",
        )

        payload_left = left_gripper_origin - self.payload_origin
        payload_right = right_gripper_origin - self.payload_origin
        self.payload_joints = [
            builder.add_joint_fixed(
                parent=self.left_gripper,
                child=self.payload,
                parent_xform=wp.transform_identity(),
                child_xform=wp.transform(wp.vec3(*payload_left), wp.quat_identity()),
                label="left_gripper_to_picked_panel",
            ),
            builder.add_joint_fixed(
                parent=self.right_gripper,
                child=self.payload,
                parent_xform=wp.transform_identity(),
                child_xform=wp.transform(wp.vec3(*payload_right), wp.quat_identity()),
                label="right_gripper_to_picked_panel",
            ),
        ]
        self.residual_joints = tuple(self.support_joints + self.payload_joints)
        builder.color()

        self.model = builder.finalize()
        self.device = self.model.device
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.contacts = None

        self.elastic_joint = int(self.model.elastic_joint.numpy()[0])
        self.elastic_q_start = int(self.model.joint_q_start.numpy()[self.elastic_joint])
        self.elastic_qd_start = int(self.model.joint_qd_start.numpy()[self.elastic_joint])
        self.wrist_q_start = int(self.model.joint_q_start.numpy()[self.j_wrist])
        self.wrist_qd_start = int(self.model.joint_qd_start.numpy()[self.j_wrist])
        self.left_slide_qd_start = int(self.model.joint_qd_start.numpy()[self.j_left_slide])
        self.right_slide_qd_start = int(self.model.joint_qd_start.numpy()[self.j_right_slide])
        self.left_slide_q_start = int(self.model.joint_target_q_start.numpy()[self.j_left_slide])
        self.right_slide_q_start = int(self.model.joint_target_q_start.numpy()[self.j_right_slide])
        self._joint_f = self.control.joint_f.numpy()
        self._target_pos = self.control.joint_target_q.numpy()
        self._target_vel = self.control.joint_target_qd.numpy()
        self.initial_wrist_z = float(self.state_0.body_q.numpy()[self.wrist, 2])

        self.solver = newton.solvers.SolverVBD(
            self.model,
            iterations=36,
            rigid_compliant_alm=True,
            rigid_avbd_beta=1.0e5,
            rigid_joint_linear_k_start=1.0e5,
            rigid_joint_angular_k_start=1.0e4,
            rigid_joint_linear_ke=2.5e6,
            rigid_joint_angular_ke=7.5e5,
            elastic_body_relaxation=0.6,
        )

        self.viewer.set_model(self.model)
        self.viewer.show_elastic_strain = True
        self.viewer.elastic_strain_color_max = 0.035
        self.viewer.set_camera(wp.vec3(0.06, -1.85, 0.80), -3.0, 90.0)

    def _mode_values(self) -> np.ndarray:
        return self.state_0.joint_q.numpy()[self.elastic_q_start + 7 : self.elastic_q_start + 7 + self.mode_count]

    def _current_gripper_sag(self) -> float:
        modes = self._mode_values().astype(np.float64)
        u = self.modal_matrix @ modes
        z_dofs = np.asarray([3 * int(node) + 2 for node in self.load_nodes], dtype=np.int32)
        return float(np.mean(u[z_dofs]))

    def _centerline_deflection_world(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        modes = self._mode_values().astype(np.float64)
        nodal_displacement = (self.modal_matrix @ modes).reshape((-1, 3))
        reference_local = np.zeros((len(self.deflection_columns), 3), dtype=np.float64)
        deformed_local = np.zeros_like(reference_local)
        unscaled_deflection = np.zeros_like(reference_local)

        for i, column in enumerate(self.deflection_columns):
            displacement = np.mean(nodal_displacement[column], axis=0)
            reference_local[i] = [float(self.deflection_x[i]), self.deflection_overlay_y, 0.0]
            deformed_local[i] = reference_local[i] + self.deflection_visual_scale * displacement
            unscaled_deflection[i] = displacement

        panel_xform = self.state_0.body_q.numpy()[self.panel]
        reference_world = np.asarray(
            [transform_point(panel_xform, point) for point in reference_local], dtype=np.float32
        )
        deformed_world = np.asarray([transform_point(panel_xform, point) for point in deformed_local], dtype=np.float32)
        return reference_world, deformed_world, unscaled_deflection

    def _log_straight_deflection_visualizer(self):
        if not self.show_straight_deflection:
            self.viewer.log_lines("/diagnostics/matrix_rom/straight_reference", None, None, None)
            self.viewer.log_lines("/diagnostics/matrix_rom/deflected_centerline", None, None, None)
            self.viewer.log_lines("/diagnostics/matrix_rom/deflection_whiskers", None, None, None)
            return

        reference_world, deformed_world, _unscaled_deflection = self._centerline_deflection_world()
        if reference_world.shape[0] < 2:
            return

        reference_starts = wp.array(reference_world[:-1], dtype=wp.vec3, device=self.viewer.device)
        reference_ends = wp.array(reference_world[1:], dtype=wp.vec3, device=self.viewer.device)
        deformed_starts = wp.array(deformed_world[:-1], dtype=wp.vec3, device=self.viewer.device)
        deformed_ends = wp.array(deformed_world[1:], dtype=wp.vec3, device=self.viewer.device)
        whisker_starts = wp.array(reference_world, dtype=wp.vec3, device=self.viewer.device)
        whisker_ends = wp.array(deformed_world, dtype=wp.vec3, device=self.viewer.device)

        self.viewer.log_lines(
            "/diagnostics/matrix_rom/straight_reference",
            reference_starts,
            reference_ends,
            (0.75, 0.75, 0.75),
            width=0.006,
        )
        self.viewer.log_lines(
            "/diagnostics/matrix_rom/deflected_centerline",
            deformed_starts,
            deformed_ends,
            (1.0, 0.1, 0.85),
            width=0.009,
        )
        self.viewer.log_lines(
            "/diagnostics/matrix_rom/deflection_whiskers",
            whisker_starts,
            whisker_ends,
            (1.0, 0.85, 0.05),
            width=0.004,
        )

    def _set_controls(self):
        load_scale = _smoothstep(self.sim_time, 0.45, 1.25)
        lift = 0.045 * _smoothstep(self.sim_time, 0.25, 1.05)
        lift -= 0.018 * _smoothstep(self.sim_time, 3.0, 4.2)
        wrist_lift = 0.035 * _smoothstep(self.sim_time, 0.95, 1.55)
        wrist_lift -= 0.025 * _smoothstep(self.sim_time, 3.25, 4.25)

        joint_q = self.state_0.joint_q.numpy()
        joint_qd = self.state_0.joint_qd.numpy()
        wrist_pos = self.wrist_origin + np.array([0.0, 0.0, wrist_lift], dtype=np.float32)
        joint_q[self.wrist_q_start : self.wrist_q_start + 3] = wrist_pos
        joint_q[self.wrist_q_start + 3 : self.wrist_q_start + 7] = [0.0, 0.0, 0.0, 1.0]
        joint_qd[self.wrist_qd_start : self.wrist_qd_start + 3] = [0.0, 0.0, 0.0]
        joint_qd[self.wrist_qd_start + 3 : self.wrist_qd_start + 6] = [0.0, 0.0, 0.0]
        self.state_0.joint_q.assign(joint_q)
        self.state_0.joint_qd.assign(joint_qd)
        newton.eval_fk(
            self.model,
            self.state_0.joint_q,
            self.state_0.joint_qd,
            self.state_0,
            body_flag_filter=newton.BodyFlags.KINEMATIC,
        )

        self._target_pos[self.left_slide_q_start] = lift
        self._target_pos[self.right_slide_q_start] = lift
        self._target_vel[self.left_slide_qd_start] = 0.0
        self._target_vel[self.right_slide_qd_start] = 0.0
        self.control.joint_target_q.assign(self._target_pos)
        self.control.joint_target_qd.assign(self._target_vel)

        self._joint_f[:] = 0.0
        modal_start = self.elastic_qd_start + 6
        self._joint_f[modal_start : modal_start + self.mode_count] = self.modal_load * load_scale
        self.control.joint_f.assign(self._joint_f)

        self.max_slide_motion = max(self.max_slide_motion, abs(lift))
        self.max_wrist_motion = max(self.max_wrist_motion, abs(wrist_lift))

    def _update_metrics(self):
        residuals = [
            float(
                np.linalg.norm(
                    joint_endpoint_world(self.model, self.state_0, joint, "parent")
                    - joint_endpoint_world(self.model, self.state_0, joint, "child")
                )
            )
            for joint in self.residual_joints
        ]
        self.max_joint_residual = max(self.max_joint_residual, *residuals)
        modes = self._mode_values()
        self.max_mode_abs = max(self.max_mode_abs, float(np.max(np.abs(modes))))
        self.max_sag_abs = max(self.max_sag_abs, abs(self._current_gripper_sag()))
        _reference_world, _deformed_world, centerline_deflection = self._centerline_deflection_world()
        self.max_centerline_deflection = max(
            self.max_centerline_deflection, float(np.max(np.linalg.norm(centerline_deflection, axis=1)))
        )

    def simulate(self):
        for _substep in range(self.sim_substeps):
            self.state_0.clear_forces()
            self.viewer.apply_forces(self.state_0)
            self._set_controls()
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
        self._log_straight_deflection_visualizer()
        self.viewer.end_frame()

    def test_final(self):
        if not np.isfinite(self.state_0.body_q.numpy()).all():
            raise AssertionError("body transforms contain non-finite values")
        if not np.isfinite(self.state_0.joint_q.numpy()).all():
            raise AssertionError("joint coordinates contain non-finite values")
        if self.model.elastic_mode_count.numpy()[0] != self.mode_count:
            raise AssertionError("matrix ROM example has the wrong number of retained modes")
        if len(self.support_joints) != 10:
            raise AssertionError(f"matrix ROM expected 10 wrist support joints, got {len(self.support_joints)}")
        if np.any(self.modal_frequencies_hz <= 0.0):
            raise AssertionError(f"matrix ROM frequencies must be positive, got {self.modal_frequencies_hz}")
        if self.reference_relative_error > 0.35:
            raise AssertionError(
                "matrix ROM static reference error too large: "
                f"full={self.reference_full_sag}, rom={self.reference_rom_sag}, rel={self.reference_relative_error}"
            )
        if self.max_joint_residual > 3.5e-2:
            raise AssertionError(f"matrix ROM fixed-joint residual too large: {self.max_joint_residual}")
        if self.max_mode_abs < 5.0e-3:
            raise AssertionError(f"matrix ROM modes were not visibly excited: {self.max_mode_abs}")
        if self.max_mode_abs > 0.65:
            raise AssertionError(f"matrix ROM modal response is out of range: {self._mode_values()}")
        if self.max_sag_abs < 0.30 * abs(self.reference_rom_sag):
            raise AssertionError(
                f"matrix ROM gripper sag too small: sag={self.max_sag_abs}, reference={self.reference_rom_sag}"
            )
        if self.max_centerline_deflection < 0.30 * abs(self.reference_rom_sag):
            raise AssertionError(
                "matrix ROM centerline deflection visualizer did not observe enough motion: "
                f"deflection={self.max_centerline_deflection}, reference={self.reference_rom_sag}"
            )
        if self.max_slide_motion < 0.035:
            raise AssertionError(f"matrix ROM gripper slides did not move enough: {self.max_slide_motion}")
        if self.max_wrist_motion < 0.025:
            raise AssertionError(f"matrix ROM wrist trajectory did not move enough: {self.max_wrist_motion}")
        deformed = elastic_shape_deformed_vertices(self.model, self.state_0)
        if not np.isfinite(deformed).all():
            raise AssertionError("deformed matrix ROM vertices contain non-finite values")


if __name__ == "__main__":
    parser = newton.examples.create_parser()
    viewer, args = newton.examples.init(parser)
    example = Example(viewer, args)
    newton.examples.run(example, args)
