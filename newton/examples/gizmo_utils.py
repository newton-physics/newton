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

import math

import numpy as np
import pyglet.window.mouse


def rotate_vector_by_quaternion(v, q):
    q_vec = q[:3]
    q_scalar = q[3]
    return (
        2.0 * np.dot(q_vec, v) * q_vec + (q_scalar**2 - np.dot(q_vec, q_vec)) * v + 2.0 * q_scalar * np.cross(q_vec, v)
    )


def get_rotation_quaternion(v_from, v_to):
    v_from = np.asarray(v_from, dtype=np.float32)
    v_to = np.asarray(v_to, dtype=np.float32)

    norm_from = np.linalg.norm(v_from)
    norm_to = np.linalg.norm(v_to)

    if norm_from < 1e-6 or norm_to < 1e-6:
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)

    v_from_norm = v_from / norm_from
    v_to_norm = v_to / norm_to

    dot_product = np.dot(v_from_norm, v_to_norm)

    if dot_product > 0.999999:
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    elif dot_product < -0.999999:
        axis = np.cross(v_from_norm, np.array([1.0, 0.0, 0.0]))
        if np.linalg.norm(axis) < 1e-6:
            axis = np.cross(v_from_norm, np.array([0.0, 1.0, 0.0]))
        axis = axis / np.linalg.norm(axis)
        angle = math.pi
    else:
        axis = np.cross(v_from_norm, v_to_norm)
        axis = axis / np.linalg.norm(axis)
        angle = math.acos(dot_product)

    s = math.sin(angle / 2.0)
    c = math.cos(angle / 2.0)
    return np.array([axis[0] * s, axis[1] * s, axis[2] * s, c], dtype=np.float32)


def quaternion_from_axis_angle(axis, angle):
    axis = np.asarray(axis, dtype=np.float32)
    axis = axis / np.linalg.norm(axis)
    s = math.sin(angle / 2.0)
    c = math.cos(angle / 2.0)
    return np.array([axis[0] * s, axis[1] * s, axis[2] * s, c], dtype=np.float32)


def quaternion_multiply(q1, q2):
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


def closest_point_on_line_to_ray(line_origin, line_dir, ray_origin, ray_dir):
    line_dir = line_dir / np.linalg.norm(line_dir)
    ray_dir = ray_dir / np.linalg.norm(ray_dir)

    r = line_origin - ray_origin
    d1d1 = 1.0
    d2d2 = 1.0
    d1d2 = np.dot(line_dir, ray_dir)

    det = d1d2 * d1d2 - d1d1 * d2d2

    if abs(det) < 1e-6:
        t1 = np.dot(ray_origin - line_origin, line_dir) / d1d1
        return t1

    t1 = (np.dot(r, line_dir) - np.dot(r, ray_dir) * d1d2) / det
    return t1


def ray_sphere_intersect(ray_origin, ray_dir, sphere_center, sphere_radius):
    L = sphere_center - ray_origin
    tca = np.dot(L, ray_dir)
    d2 = np.dot(L, L) - tca * tca
    if d2 > sphere_radius**2:
        return float("inf")
    thc = np.sqrt(sphere_radius**2 - d2)
    t0 = tca - thc
    t1 = tca + thc
    if t0 < 0 and t1 < 0:
        return float("inf")
    return t1 if t0 < 0 else t0


def ray_capsule_intersect(ray_origin, ray_dir, p_start, p_end, radius):
    ca = p_end - p_start
    oa = ray_origin - p_start
    caca = np.dot(ca, ca)
    card = np.dot(ca, ray_dir)
    caoa = np.dot(ca, oa)
    a = caca - card * card
    b = caca * np.dot(oa, ray_dir) - caoa * card
    c = caca * np.dot(oa, oa) - caoa * caoa - radius * radius * caca
    h = b * b - a * c
    if h < 0.0:
        return float("inf")
    t = (-b - np.sqrt(h)) / a
    y = caoa + t * card
    if y > 0.0 and y < caca:
        return t if t > 0 else float("inf")
    t_cap1 = ray_sphere_intersect(ray_origin, ray_dir, p_start, radius)
    t_cap2 = ray_sphere_intersect(ray_origin, ray_dir, p_end, radius)
    if y <= 0.0:
        return t_cap1 if t_cap1 > 0 else float("inf")
    elif y >= caca:
        return t_cap2 if t_cap2 > 0 else float("inf")
    return float("inf")


def ray_box_intersect(ray_origin, ray_dir, box_center, box_extents, box_orientation):
    inv_orientation = np.array(
        [-box_orientation[0], -box_orientation[1], -box_orientation[2], box_orientation[3]], dtype=np.float32
    )

    local_ray_origin = rotate_vector_by_quaternion(ray_origin - box_center, inv_orientation)
    local_ray_dir = rotate_vector_by_quaternion(ray_dir, inv_orientation)

    tmin = float("-inf")
    tmax = float("inf")

    for i in range(3):
        if abs(local_ray_dir[i]) < 1e-6:
            if abs(local_ray_origin[i]) > box_extents[i] / 2.0:
                return float("inf")
        else:
            t1 = (-box_extents[i] / 2.0 - local_ray_origin[i]) / local_ray_dir[i]
            t2 = (box_extents[i] / 2.0 - local_ray_origin[i]) / local_ray_dir[i]
            tmin = max(tmin, min(t1, t2))
            tmax = min(tmax, max(t1, t2))

    if tmin > tmax or tmax < 0:
        return float("inf")

    return tmin if tmin >= 0 else tmax


class Transform:
    def __init__(self, position, rotation=None):
        self.position = np.array(position, dtype=np.float32)
        self.rotation = np.array(rotation if rotation is not None else [0, 0, 0, 1], dtype=np.float32)


class Arrow:
    def __init__(self, axis_name):
        self.axis_name = axis_name
        self.axis_vector = np.array(
            {"X": [1.0, 0.0, 0.0], "Y": [0.0, 1.0, 0.0], "Z": [0.0, 0.0, 1.0]}[axis_name], dtype=np.float32
        )

        self.original_color = {"X": (0.8, 0.2, 0.2), "Y": (0.2, 0.8, 0.2), "Z": (0.2, 0.2, 0.8)}[axis_name]

        self.color = self.original_color
        self.capsule_radius_factor = 0.08
        self.collision_radius_factor = 0.2


class Plane:
    def __init__(self, plane_name):
        self.plane_name = plane_name

        self.normal = np.array(
            {"XY": [0.0, 0.0, 1.0], "XZ": [0.0, 1.0, 0.0], "YZ": [1.0, 0.0, 0.0]}[plane_name], dtype=np.float32
        )

        self.axis1 = np.array(
            {"XY": [1.0, 0.0, 0.0], "XZ": [1.0, 0.0, 0.0], "YZ": [0.0, 1.0, 0.0]}[plane_name], dtype=np.float32
        )

        self.axis2 = np.array(
            {"XY": [0.0, 1.0, 0.0], "XZ": [0.0, 0.0, 1.0], "YZ": [0.0, 0.0, 1.0]}[plane_name], dtype=np.float32
        )

        stable_axis = {"XY": "Z", "XZ": "Y", "YZ": "X"}[plane_name]

        self.original_color = {"X": (0.8, 0.2, 0.2), "Y": (0.2, 0.8, 0.2), "Z": (0.2, 0.2, 0.8)}[stable_axis]

        self.color = self.original_color
        self.size = 0.3
        self.thickness = 0.1
        self.distance_factor = 0.7
        self.display_size = 0.3
        self.edge_radius_factor = 0.01  # Radius for edge capsules


class Arc:
    def __init__(self, axis_name):
        self.axis_name = axis_name

        self.axis_vector = np.array(
            {"X": [1.0, 0.0, 0.0], "Y": [0.0, 1.0, 0.0], "Z": [0.0, 0.0, 1.0]}[axis_name], dtype=np.float32
        )

        self.from_axis = {"X": "Y", "Y": "Z", "Z": "X"}[axis_name]
        self.to_axis = {"X": "Z", "Y": "X", "Z": "Y"}[axis_name]

        self.original_color = {"X": (0.8, 0.2, 0.2), "Y": (0.2, 0.8, 0.2), "Z": (0.2, 0.2, 0.8)}[axis_name]

        self.color = self.original_color
        self.capsules_per_arc = 4
        self.capsule_radius = 0.015


class GizmoTarget:
    def __init__(self, target_id, renderer, position, rotation, offset, scale_factor):
        self.id = target_id
        self.renderer = renderer
        self.scale_factor = scale_factor

        self.target_transform = Transform(position, rotation)
        self.offset = np.array(offset, dtype=np.float32)

        self.arrows = {"X": Arrow("X"), "Y": Arrow("Y"), "Z": Arrow("Z")}
        self.planes = {"XY": Plane("XY"), "XZ": Plane("XZ"), "YZ": Plane("YZ")}
        self.arcs = {"X": Arc("X"), "Y": Arc("Y"), "Z": Arc("Z")}

        self._last_position_for_arcs = None
        self._last_rotation_for_arcs = None
        self._last_position_for_planes = None
        self._last_rotation_for_planes = None

        self.gizmo_coll_radius = 0.3 * scale_factor
        self.gizmo_coll_half_height = 0.5 * scale_factor
        self.arc_radius_factor = 1.0
        self.arc_capsule_radius = 0.1 * scale_factor

        self._setup_visuals()

    def _compute_visual_transform(self):
        visual_position = self.target_transform.position + self.offset
        visual_rotation = self.target_transform.rotation
        return visual_position, visual_rotation

    def update_position(self, position):
        self.target_transform.position = np.array(position, dtype=np.float32)
        self._update_visuals()

    def update_rotation(self, rotation):
        self.target_transform.rotation = np.array(rotation, dtype=np.float32)
        self._update_visuals()

    def get_position(self):
        return self.target_transform.position

    def get_rotation(self):
        return self.target_transform.rotation

        self._update_visuals()

    def find_hit(self, ray_origin, ray_dir):
        visual_pos, visual_rot = self._compute_visual_transform()
        min_t = float("inf")
        hit_component = None

        for arrow in self.arrows.values():
            world_axis = rotate_vector_by_quaternion(arrow.axis_vector, visual_rot)
            p_start = visual_pos
            p_end = visual_pos + world_axis * (self.gizmo_coll_half_height * 2)
            radius = self.scale_factor * arrow.collision_radius_factor

            t = ray_capsule_intersect(ray_origin, ray_dir, p_start, p_end, radius)
            if 0 < t < min_t:
                min_t = t
                hit_component = arrow

        # Use box collision for planes
        for plane in self.planes.values():
            center_pos, orientation = self._get_plane_transform(plane, visual_pos, visual_rot)
            size = plane.size * self.scale_factor
            thickness = plane.thickness * self.scale_factor

            t = ray_box_intersect(ray_origin, ray_dir, center_pos, np.array([size, size, thickness]), orientation)
            if 0 < t < min_t:
                min_t = t
                hit_component = plane

        for arc in self.arcs.values():
            segments = self._get_arc_collision_segments(arc, visual_pos, visual_rot)
            for p_start, p_end in segments:
                t = ray_capsule_intersect(ray_origin, ray_dir, p_start, p_end, self.arc_capsule_radius)
                if 0 < t < min_t:
                    min_t = t
                    hit_component = arc

        return hit_component if min_t < float("inf") else None

    def _get_arrow_transform(self, arrow, visual_pos, visual_rot):
        world_axis = rotate_vector_by_quaternion(arrow.axis_vector, visual_rot)
        position = visual_pos + self.gizmo_coll_half_height * world_axis
        axis_alignment = get_rotation_quaternion(np.array([0.0, 1.0, 0.0]), arrow.axis_vector)
        orientation = quaternion_multiply(visual_rot, axis_alignment)
        return position, orientation

    def _get_plane_transform(self, plane, visual_pos, visual_rot):
        world_axis1 = rotate_vector_by_quaternion(plane.axis1, visual_rot)
        world_axis2 = rotate_vector_by_quaternion(plane.axis2, visual_rot)
        world_normal = rotate_vector_by_quaternion(plane.normal, visual_rot)

        total_arrow_length = self.gizmo_coll_half_height * 2
        plane_distance = total_arrow_length * plane.distance_factor

        diagonal_dir = (world_axis1 + world_axis2) / np.sqrt(2.0)
        plane_center = visual_pos + diagonal_dir * plane_distance

        rot_matrix = np.array(
            [
                [world_axis1[0], world_axis2[0], world_normal[0]],
                [world_axis1[1], world_axis2[1], world_normal[1]],
                [world_axis1[2], world_axis2[2], world_normal[2]],
            ],
            dtype=np.float32,
        )

        trace = rot_matrix[0, 0] + rot_matrix[1, 1] + rot_matrix[2, 2]

        if trace > 0:
            s = 0.5 / np.sqrt(trace + 1.0)
            w = 0.25 / s
            x = (rot_matrix[2, 1] - rot_matrix[1, 2]) * s
            y = (rot_matrix[0, 2] - rot_matrix[2, 0]) * s
            z = (rot_matrix[1, 0] - rot_matrix[0, 1]) * s
        elif rot_matrix[0, 0] > rot_matrix[1, 1] and rot_matrix[0, 0] > rot_matrix[2, 2]:
            s = 2.0 * np.sqrt(1.0 + rot_matrix[0, 0] - rot_matrix[1, 1] - rot_matrix[2, 2])
            w = (rot_matrix[2, 1] - rot_matrix[1, 2]) / s
            x = 0.25 * s
            y = (rot_matrix[0, 1] + rot_matrix[1, 0]) / s
            z = (rot_matrix[0, 2] + rot_matrix[2, 0]) / s
        elif rot_matrix[1, 1] > rot_matrix[2, 2]:
            s = 2.0 * np.sqrt(1.0 + rot_matrix[1, 1] - rot_matrix[0, 0] - rot_matrix[2, 2])
            w = (rot_matrix[0, 2] - rot_matrix[2, 0]) / s
            x = (rot_matrix[0, 1] + rot_matrix[1, 0]) / s
            y = 0.25 * s
            z = (rot_matrix[1, 2] + rot_matrix[2, 1]) / s
        else:
            s = 2.0 * np.sqrt(1.0 + rot_matrix[2, 2] - rot_matrix[0, 0] - rot_matrix[1, 1])
            w = (rot_matrix[1, 0] - rot_matrix[0, 1]) / s
            x = (rot_matrix[0, 2] + rot_matrix[2, 0]) / s
            y = (rot_matrix[1, 2] + rot_matrix[2, 1]) / s
            z = 0.25 * s

        orientation = np.array([x, y, z, w], dtype=np.float32)
        return plane_center, orientation

    def _get_plane_quad_vertices(self, plane, visual_pos, visual_rot):
        plane_center, _ = self._get_plane_transform(plane, visual_pos, visual_rot)

        world_axis1 = rotate_vector_by_quaternion(plane.axis1, visual_rot)
        world_axis2 = rotate_vector_by_quaternion(plane.axis2, visual_rot)

        half_size = plane.display_size * self.scale_factor / 2

        vertices = [
            plane_center + half_size * world_axis1 + half_size * world_axis2,
            plane_center + half_size * world_axis1 - half_size * world_axis2,
            plane_center - half_size * world_axis1 - half_size * world_axis2,
            plane_center - half_size * world_axis1 + half_size * world_axis2,
        ]

        return vertices

    def _get_plane_edge_transforms(self, plane, visual_pos, visual_rot):
        """Get transforms for the 4 edge capsules of a plane"""
        vertices = self._get_plane_quad_vertices(plane, visual_pos, visual_rot)

        edge_transforms = []
        edges = [(0, 1), (1, 2), (2, 3), (3, 0)]

        for _, (start_idx, end_idx) in enumerate(edges):
            p_start = vertices[start_idx]
            p_end = vertices[end_idx]

            # Midpoint
            position = (p_start + p_end) / 2

            # Direction and half-height
            edge_dir = p_end - p_start
            edge_length = np.linalg.norm(edge_dir)
            half_height = edge_length / 2

            # Orientation - align Y axis with edge direction
            if edge_length > 1e-6:
                edge_dir_norm = edge_dir / edge_length
                orientation = get_rotation_quaternion(np.array([0.0, 1.0, 0.0]), edge_dir_norm)
            else:
                orientation = np.array([0.0, 0.0, 0.0, 1.0])

            edge_transforms.append((position, orientation, half_height))

        return edge_transforms

    def _get_arc_vertices(self, arc, visual_pos, visual_rot):
        from_vec = np.array(
            {"X": [1.0, 0.0, 0.0], "Y": [0.0, 1.0, 0.0], "Z": [0.0, 0.0, 1.0]}[arc.from_axis], dtype=np.float32
        )

        to_vec = np.array(
            {"X": [1.0, 0.0, 0.0], "Y": [0.0, 1.0, 0.0], "Z": [0.0, 0.0, 1.0]}[arc.to_axis], dtype=np.float32
        )

        from_world = rotate_vector_by_quaternion(from_vec, visual_rot)
        to_world = rotate_vector_by_quaternion(to_vec, visual_rot)

        radius = self.gizmo_coll_half_height * 2 * self.arc_radius_factor

        vertices = []
        num_segments = 16

        for i in range(num_segments + 1):
            t = i / num_segments
            t * (math.pi / 2)

            sin_total = math.sin(math.pi / 2)
            a = math.sin((1 - t) * (math.pi / 2)) / sin_total
            b = math.sin(t * (math.pi / 2)) / sin_total

            interpolated = a * from_world + b * to_world
            point = visual_pos + interpolated * radius
            vertices.append(point.tolist())

        return vertices

    def _get_arc_collision_segments(self, arc, visual_pos, visual_rot):
        from_vec = np.array(
            {"X": [1.0, 0.0, 0.0], "Y": [0.0, 1.0, 0.0], "Z": [0.0, 0.0, 1.0]}[arc.from_axis], dtype=np.float32
        )

        to_vec = np.array(
            {"X": [1.0, 0.0, 0.0], "Y": [0.0, 1.0, 0.0], "Z": [0.0, 0.0, 1.0]}[arc.to_axis], dtype=np.float32
        )

        from_world = rotate_vector_by_quaternion(from_vec, visual_rot)
        to_world = rotate_vector_by_quaternion(to_vec, visual_rot)

        radius = self.gizmo_coll_half_height * 2 * self.arc_radius_factor

        segments = []
        for i in range(arc.capsules_per_arc):
            t1 = i / arc.capsules_per_arc
            t2 = (i + 1) / arc.capsules_per_arc

            sin_total = math.sin(math.pi / 2)

            a1 = math.sin((1 - t1) * (math.pi / 2)) / sin_total
            b1 = math.sin(t1 * (math.pi / 2)) / sin_total
            p1 = visual_pos + (a1 * from_world + b1 * to_world) * radius

            a2 = math.sin((1 - t2) * (math.pi / 2)) / sin_total
            b2 = math.sin(t2 * (math.pi / 2)) / sin_total
            p2 = visual_pos + (a2 * from_world + b2 * to_world) * radius

            segments.append((p1, p2))

        return segments

    def _get_arc_segment_transforms(self, arc, visual_pos, visual_rot):
        """Get transforms for arc segment capsules"""
        segments = self._get_arc_collision_segments(arc, visual_pos, visual_rot)

        segment_transforms = []
        for p_start, p_end in segments:
            # Midpoint
            position = (p_start + p_end) / 2

            # Direction and half-height
            segment_dir = p_end - p_start
            segment_length = np.linalg.norm(segment_dir)
            half_height = segment_length / 2

            # Orientation - align Y axis with segment direction
            if segment_length > 1e-6:
                segment_dir_norm = segment_dir / segment_length
                orientation = get_rotation_quaternion(np.array([0.0, 1.0, 0.0]), segment_dir_norm)
            else:
                orientation = np.array([0.0, 0.0, 0.0, 1.0])

            segment_transforms.append((position, orientation, half_height))

        return segment_transforms

    def _setup_visuals(self):
        visual_pos, visual_rot = self._compute_visual_transform()

        # Setup arrows (existing code)
        for name, arrow in self.arrows.items():
            pos, rot = self._get_arrow_transform(arrow, visual_pos, visual_rot)
            self.renderer.render_capsule(
                name=f"arrow_{self.id}_{name}",
                pos=tuple(pos),
                rot=tuple(rot),
                half_height=self.gizmo_coll_half_height,
                radius=self.scale_factor * arrow.capsule_radius_factor,
                up_axis=1,
                color=arrow.color,
            )
            # Fix initial color
            self.renderer.update_shape_instance(
                name=f"arrow_{self.id}_{name}", pos=tuple(pos), rot=tuple(rot), color1=arrow.color, color2=arrow.color
            )

        # Setup planes with capsules
        for name, plane in self.planes.items():
            edge_transforms = self._get_plane_edge_transforms(plane, visual_pos, visual_rot)

            for i, (pos, rot, half_height) in enumerate(edge_transforms):
                self.renderer.render_capsule(
                    name=f"plane_{self.id}_{name}_edge{i}",
                    pos=tuple(pos),
                    rot=tuple(rot),
                    half_height=half_height,
                    radius=plane.edge_radius_factor * self.scale_factor,
                    up_axis=1,
                    color=plane.color,
                )
                # Fix initial color
                self.renderer.update_shape_instance(
                    name=f"plane_{self.id}_{name}_edge{i}",
                    pos=tuple(pos),
                    rot=tuple(rot),
                    color1=plane.color,
                    color2=plane.color,
                )

        # Setup arcs with capsules
        for name, arc in self.arcs.items():
            segment_transforms = self._get_arc_segment_transforms(arc, visual_pos, visual_rot)

            for i, (pos, rot, half_height) in enumerate(segment_transforms):
                self.renderer.render_capsule(
                    name=f"arc_{self.id}_{name}_seg{i}",
                    pos=tuple(pos),
                    rot=tuple(rot),
                    half_height=half_height,
                    radius=arc.capsule_radius * self.scale_factor,
                    up_axis=1,
                    color=arc.color,
                )
                # Fix initial color
                self.renderer.update_shape_instance(
                    name=f"arc_{self.id}_{name}_seg{i}",
                    pos=tuple(pos),
                    rot=tuple(rot),
                    color1=arc.color,
                    color2=arc.color,
                )

        self._last_position_for_arcs = np.copy(visual_pos)
        self._last_rotation_for_arcs = np.copy(visual_rot)
        self._last_position_for_planes = np.copy(visual_pos)
        self._last_rotation_for_planes = np.copy(visual_rot)

    def _update_visuals(self):
        visual_pos, visual_rot = self._compute_visual_transform()

        for name, arrow in self.arrows.items():
            pos, rot = self._get_arrow_transform(arrow, visual_pos, visual_rot)
            self.renderer.update_shape_instance(
                name=f"arrow_{self.id}_{name}", pos=tuple(pos), rot=tuple(rot), color1=None, color2=None
            )

        # Update planes if position or rotation changed
        position_changed_planes = self._last_position_for_planes is None or not np.array_equal(
            self._last_position_for_planes, visual_pos
        )
        rotation_changed_planes = self._last_rotation_for_planes is None or not np.array_equal(
            self._last_rotation_for_planes, visual_rot
        )

        if position_changed_planes or rotation_changed_planes:
            for name, plane in self.planes.items():
                edge_transforms = self._get_plane_edge_transforms(plane, visual_pos, visual_rot)

                for i, (pos, rot, _) in enumerate(edge_transforms):
                    self.renderer.update_shape_instance(
                        name=f"plane_{self.id}_{name}_edge{i}", pos=tuple(pos), rot=tuple(rot), color1=None, color2=None
                    )

            self._last_position_for_planes = np.copy(visual_pos)
            self._last_rotation_for_planes = np.copy(visual_rot)

        # Update arcs if either position or rotation changed
        position_changed = self._last_position_for_arcs is None or not np.array_equal(
            self._last_position_for_arcs, visual_pos
        )
        rotation_changed = self._last_rotation_for_arcs is None or not np.array_equal(
            self._last_rotation_for_arcs, visual_rot
        )

        if position_changed or rotation_changed:
            for name, arc in self.arcs.items():
                segment_transforms = self._get_arc_segment_transforms(arc, visual_pos, visual_rot)

                for i, (pos, rot, _) in enumerate(segment_transforms):
                    self.renderer.update_shape_instance(
                        name=f"arc_{self.id}_{name}_seg{i}", pos=tuple(pos), rot=tuple(rot), color1=None, color2=None
                    )

            self._last_position_for_arcs = np.copy(visual_pos)
            self._last_rotation_for_arcs = np.copy(visual_rot)


class DragState:
    def __init__(self, target, component, initial_ray_origin, initial_ray_dir, renderer):
        self.target = target
        self.component = component
        self.renderer = renderer

        visual_pos, visual_rot = target._compute_visual_transform()
        self.start_visual_position = visual_pos
        self.start_visual_rotation = visual_rot
        self.start_target_position = np.copy(target.target_transform.position)
        self.start_target_rotation = np.copy(target.target_transform.rotation)

        if isinstance(component, Arrow):
            self.mode = "translate"
            self.axis = rotate_vector_by_quaternion(component.axis_vector, visual_rot)
            self.initial_t = closest_point_on_line_to_ray(visual_pos, self.axis, initial_ray_origin, initial_ray_dir)

        elif isinstance(component, Plane):
            self.mode = "translate_plane"
            self.normal = rotate_vector_by_quaternion(component.normal, visual_rot)
            self.axis1 = rotate_vector_by_quaternion(component.axis1, visual_rot)
            self.axis2 = rotate_vector_by_quaternion(component.axis2, visual_rot)

            plane_d = np.dot(self.normal, visual_pos)
            denom = np.dot(initial_ray_dir, self.normal)
            if abs(denom) > 1e-6:
                t = (plane_d - np.dot(initial_ray_origin, self.normal)) / denom
                self.initial_plane_point = initial_ray_origin + t * initial_ray_dir
            else:
                self.initial_plane_point = visual_pos

        elif isinstance(component, Arc):
            self.mode = "rotate"
            self.rotation_axis = rotate_vector_by_quaternion(component.axis_vector, visual_rot)
            self.prev_angle = None
            self.accumulated_angle = 0.0

    def update(self, ray_origin, ray_dir, mouse_x=None, mouse_y=None, rotation_sensitivity=0.01):
        if self.mode == "translate":
            t = closest_point_on_line_to_ray(self.start_visual_position, self.axis, ray_origin, ray_dir)
            delta = (t - self.initial_t) * self.axis
            new_target_position = self.start_target_position + delta
            self.target.update_position(new_target_position)
        elif self.mode == "translate_plane":
            plane_d = np.dot(self.normal, self.start_visual_position)
            denom = np.dot(ray_dir, self.normal)
            if abs(denom) > 1e-6:
                t = (plane_d - np.dot(ray_origin, self.normal)) / denom
                if t > 0:
                    current_point = ray_origin + t * ray_dir
                    delta = current_point - self.initial_plane_point
                    movement = np.dot(delta, self.axis1) * self.axis1 + np.dot(delta, self.axis2) * self.axis2
                    new_target_position = self.start_target_position + movement
                    self.target.update_position(new_target_position)
        elif self.mode == "rotate":
            # Get camera info
            camera_pos = np.array(
                [self.renderer.camera_pos.x, self.renderer.camera_pos.y, self.renderer.camera_pos.z], dtype=np.float32
            )

            visual_pos, _ = self.target._compute_visual_transform()

            # Project gizmo center to screen
            view_matrix = self.renderer._view_matrix.reshape(4, 4).T
            proj_matrix = self.renderer._projection_matrix.reshape(4, 4).T

            # Transform to clip space
            visual_pos, _ = self.target._compute_visual_transform()

            # Instead, you need to scale it to render space before projection:
            scaling_factor = 1.0
            if hasattr(self.renderer, "scaling"):
                scaling_factor = self.renderer.scaling

            # Scale position to render space for screen projection
            visual_pos_render = visual_pos * scaling_factor

            # Then use visual_pos_render for the screen projection:
            gizmo_clip = proj_matrix @ view_matrix @ np.append(visual_pos_render, 1.0)
            if abs(gizmo_clip[3]) > 1e-6:
                gizmo_ndc = gizmo_clip[:3] / gizmo_clip[3]
                gizmo_screen_x = (gizmo_ndc[0] + 1.0) * self.renderer.screen_width / 2.0
                gizmo_screen_y = (gizmo_ndc[1] + 1.0) * self.renderer.screen_height / 2.0

                # Vector from gizmo center to mouse
                to_mouse_x = mouse_x - gizmo_screen_x
                to_mouse_y = mouse_y - gizmo_screen_y

                # Normalize to get direction (avoid divide by zero)
                dist = math.sqrt(to_mouse_x * to_mouse_x + to_mouse_y * to_mouse_y)
                if dist > 1e-6:
                    to_mouse_x /= dist
                    to_mouse_y /= dist

                    # Tangent is perpendicular to radial direction
                    # For counterclockwise: tangent = (-y, x)
                    tangent_x = -to_mouse_y
                    tangent_y = to_mouse_x

                    # Get mouse delta
                    if self.prev_angle is not None:  # We're reusing prev_angle to store prev mouse pos
                        mouse_delta_x = mouse_x - self.prev_angle[0]
                        mouse_delta_y = mouse_y - self.prev_angle[1]

                        # Project mouse movement onto tangent direction
                        tangent_movement = mouse_delta_x * tangent_x + mouse_delta_y * tangent_y

                        # Convert to rotation angle
                        # Negative because we want clockwise drag = clockwise rotation
                        rotation_delta = -tangent_movement * rotation_sensitivity * 0.01

                        # Get camera position (which is in render space when scaled)
                        camera_pos = np.array(
                            [self.renderer.camera_pos.x, self.renderer.camera_pos.y, self.renderer.camera_pos.z],
                            dtype=np.float32,
                        )

                        # Calculate view direction using render space positions
                        view_dir = visual_pos_render - camera_pos  # Both in render space now
                        view_dir = view_dir / np.linalg.norm(view_dir)

                        # The rotation axis also needs to be consistent
                        # It's already in world/sim space from rotate_vector_by_quaternion
                        axis_dot_view = np.dot(self.rotation_axis, view_dir)

                        # Flip rotation direction if we're looking from the opposite side
                        if axis_dot_view < 0:
                            rotation_delta = -rotation_delta

                        # Accumulate rotation
                        self.accumulated_angle += rotation_delta

                        # Apply rotation from start position
                        if abs(self.accumulated_angle) > 1e-6:
                            rotation_quat = quaternion_from_axis_angle(self.rotation_axis, self.accumulated_angle)
                            new_rotation = quaternion_multiply(rotation_quat, self.start_target_rotation)
                            self.target.update_rotation(new_rotation)

                    # Store current mouse position for next frame
                    self.prev_angle = (mouse_x, mouse_y)
            else:
                # If we can't project, still update prev position
                self.prev_angle = (mouse_x, mouse_y)

    def get_drag_axis_points(self):
        visual_pos, _ = self.target._compute_visual_transform()
        far_length = 1000.0

        if self.mode == "translate":
            p1 = visual_pos - self.axis * far_length
            p2 = visual_pos + self.axis * far_length
            return [p1.tolist(), p2.tolist()]

        elif self.mode == "translate_plane":
            points = []
            p1 = visual_pos - self.axis1 * far_length
            p2 = visual_pos + self.axis1 * far_length
            points.extend([p1.tolist(), p2.tolist()])

            points.append(p2.tolist())

            p3 = visual_pos - self.axis2 * far_length
            p4 = visual_pos + self.axis2 * far_length
            points.extend([p3.tolist(), p4.tolist()])

            return points

        elif self.mode == "rotate":
            p1 = visual_pos - self.rotation_axis * far_length
            p2 = visual_pos + self.rotation_axis * far_length
            return [p1.tolist(), p2.tolist()]

        return []


class GizmoSystem:
    def __init__(self, renderer, scale_factor=1.0, rotation_sensitivity=0.01):
        self.renderer = renderer
        self.scale_factor = scale_factor
        self.rotation_sensitivity = rotation_sensitivity
        self.targets = {}
        self.drag_state = None
        self.position_callback = None
        self.rotation_callback = None

    def create_target(self, target_id, position, rotation=None, world_offset=None):
        if world_offset is None:
            world_offset = [0.0, 0.0, 0.0]

        self.targets[target_id] = GizmoTarget(
            target_id, self.renderer, position, rotation, world_offset, self.scale_factor
        )

    def update_target_position(self, target_id, position):
        if target_id in self.targets:
            self.targets[target_id].update_position(position)

    def update_target_rotation(self, target_id, rotation):
        if target_id in self.targets:
            self.targets[target_id].update_rotation(rotation)

    def set_callbacks(self, position_callback=None, rotation_callback=None):
        self.position_callback = position_callback
        self.rotation_callback = rotation_callback

    def on_mouse_press(self, x, y, button, modifiers):
        if button != pyglet.window.mouse.LEFT:
            return False

        ray_origin, ray_dir = self._cast_ray_from_screen(x, y)
        if ray_origin is None:
            return False

        for target in self.targets.values():
            hit = target.find_hit(ray_origin, ray_dir)
            if hit:
                self.drag_state = DragState(target, hit, ray_origin, ray_dir, self.renderer)
                self._update_drag_axis()
                return True

        return False

    def on_mouse_drag(self, x, y, dx, dy, buttons, modifiers):
        if not self.drag_state or not (buttons & pyglet.window.mouse.LEFT):
            return False

        ray_origin, ray_dir = self._cast_ray_from_screen(x, y)
        if ray_origin is None:
            return False

        prev_pos = np.copy(self.drag_state.target.target_transform.position)
        prev_rot = np.copy(self.drag_state.target.target_transform.rotation)

        self.drag_state.update(ray_origin, ray_dir, x, y, self.rotation_sensitivity)

        if self.drag_state.mode in ["translate", "translate_plane"] and self.position_callback:
            if not np.array_equal(prev_pos, self.drag_state.target.target_transform.position):
                self.position_callback(self.drag_state.target.id, self.drag_state.target.get_position())
            return self.drag_state.mode == "rotate"

        if self.drag_state.mode == "rotate" and self.rotation_callback:
            if not np.array_equal(prev_rot, self.drag_state.target.target_transform.rotation):
                self.rotation_callback(self.drag_state.target.id, self.drag_state.target.get_rotation())

        return self.drag_state.mode == "rotate"

    def on_mouse_release(self, x, y, button, modifiers):
        if button == pyglet.window.mouse.LEFT and self.drag_state:
            self.drag_state = None
            self._hide_drag_axis()
            return True
        return False

    def _cast_ray_from_screen(self, x, y):
        screen_width = self.renderer.screen_width
        screen_height = self.renderer.screen_height
        if screen_width == 0 or screen_height == 0:
            return None, None

        # Convert to NDC
        ndc_x = (2.0 * x) / screen_width - 1.0
        ndc_y = (2.0 * y) / screen_height - 1.0

        # Get matrices
        if not (hasattr(self.renderer, "_projection_matrix") and hasattr(self.renderer, "_view_matrix")):
            return None, None

        try:
            # Get projection and view matrices
            proj_matrix = self.renderer._projection_matrix.reshape(4, 4).T
            view_matrix = self.renderer._view_matrix.reshape(4, 4).T

            # Combined view-projection inverse
            vp_matrix = proj_matrix @ view_matrix
            inv_vp_matrix = np.linalg.inv(vp_matrix)

            # Transform NDC points to world/render space
            near_point_ndc = np.array([ndc_x, ndc_y, -1.0, 1.0])
            far_point_ndc = np.array([ndc_x, ndc_y, 1.0, 1.0])

            near_point_world = inv_vp_matrix @ near_point_ndc
            far_point_world = inv_vp_matrix @ far_point_ndc

            # Perspective divide
            near_point_world = near_point_world[:3] / near_point_world[3]
            far_point_world = far_point_world[:3] / far_point_world[3]

            # Ray in render space
            ray_origin_render = near_point_world
            ray_direction_render = far_point_world - near_point_world
            ray_direction_render = ray_direction_render / np.linalg.norm(ray_direction_render)

            # Transform ray to simulation space if scaling is applied
            if hasattr(self.renderer, "_inv_model_matrix") and self.renderer._inv_model_matrix is not None:
                inv_model_matrix = self.renderer._inv_model_matrix.reshape(4, 4).T

                # Transform ray origin to sim space
                ray_origin_sim_h = inv_model_matrix @ np.append(ray_origin_render, 1.0)
                ray_origin_sim = ray_origin_sim_h[:3] / ray_origin_sim_h[3]

                # Transform ray direction to sim space (no w division needed for directions)
                ray_dir_sim_h = inv_model_matrix @ np.append(ray_direction_render, 0.0)
                ray_dir_sim = ray_dir_sim_h[:3]
                ray_dir_sim = ray_dir_sim / np.linalg.norm(ray_dir_sim)

                # Use sim space ray
                ray_origin = ray_origin_sim
                ray_direction = ray_dir_sim
            else:
                # No scaling, use render space ray directly
                ray_origin = ray_origin_render
                ray_direction = ray_direction_render

            return ray_origin, ray_direction

        except Exception:
            return None, None

    def _update_drag_axis(self):
        if self.drag_state:
            points = self.drag_state.get_drag_axis_points()
            self.renderer.render_line_strip(
                name="drag_axis_visualization",
                vertices=points,
                color=(1.0, 1.0, 0.0, 0.8),
                radius=0.015 * self.scale_factor,
            )

    def _hide_drag_axis(self):
        self.renderer.render_line_strip(
            name="drag_axis_visualization", vertices=[], color=(1.0, 1.0, 0.0, 0.8), radius=0.015 * self.scale_factor
        )
