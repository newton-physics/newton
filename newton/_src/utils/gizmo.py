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
from enum import Enum

import numpy as np
import warp as wp
from warp.render.render_opengl import ShapeInstancer, update_vbo_transforms

# Coordinate system axes
AXES = {
    "X": np.array([1.0, 0.0, 0.0], dtype=np.float32),
    "Y": np.array([0.0, 1.0, 0.0], dtype=np.float32),
    "Z": np.array([0.0, 0.0, 1.0], dtype=np.float32),
}

# Axis colors
COLORS = {
    "X": (0.8, 0.2, 0.2),
    "Y": (0.2, 0.8, 0.2),
    "Z": (0.2, 0.2, 0.8),
}

# Arc axis mappings
ARC_MAP = {
    "X": ("Y", "Z"),
    "Y": ("Z", "X"),
    "Z": ("X", "Y"),
}


class ComponentType(Enum):
    """Component type identifiers for collision detection."""

    ARROW = 0
    ARC = 1


def get_rotation_quaternion(v_from, v_to):
    """Compute quaternion that rotates v_from to v_to."""
    v_from = np.asarray(v_from, dtype=np.float32)
    v_to = np.asarray(v_to, dtype=np.float32)

    norm_from = np.linalg.norm(v_from)
    norm_to = np.linalg.norm(v_to)

    if norm_from < 1e-6 or norm_to < 1e-6:
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)

    v_from_norm = v_from / norm_from
    v_to_norm = v_to / norm_to

    dot = np.dot(v_from_norm, v_to_norm)

    if dot > 0.999999:
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)

    elif dot < -0.999999:
        axis = np.cross(v_from_norm, np.array([1.0, 0.0, 0.0]))
        if np.linalg.norm(axis) < 1e-6:
            axis = np.cross(v_from_norm, np.array([0.0, 1.0, 0.0]))
        axis = axis / np.linalg.norm(axis)
        angle = math.pi

    else:
        axis = np.cross(v_from_norm, v_to_norm)
        axis = axis / np.linalg.norm(axis)
        angle = math.acos(dot)

    half_angle = angle / 2.0
    s = math.sin(half_angle)
    c = math.cos(half_angle)

    return np.array([axis[0] * s, axis[1] * s, axis[2] * s, c], dtype=np.float32)


def closest_point_on_line_to_ray(line_origin, line_dir, ray_origin, ray_dir):
    """Find parameter t such that line_origin + t*line_dir is closest to ray."""
    line_dir_norm = line_dir / np.linalg.norm(line_dir)
    ray_dir_norm = ray_dir / np.linalg.norm(ray_dir)

    offset = line_origin - ray_origin
    dot_product = np.dot(line_dir_norm, ray_dir_norm)
    determinant = dot_product * dot_product - 1.0

    if abs(determinant) < 1e-6:
        return np.dot(ray_origin - line_origin, line_dir_norm)

    t = (np.dot(offset, line_dir_norm) - np.dot(offset, ray_dir_norm) * dot_product) / determinant
    return t


@wp.kernel
def compute_ray_from_screen(
    screen_x: float,
    screen_y: float,
    screen_width: float,
    screen_height: float,
    view_matrix: wp.mat44,
    proj_matrix: wp.mat44,
    has_model_matrix: int,
    inv_model_matrix: wp.mat44,
    # outputs
    ray_origin: wp.array(dtype=wp.vec3),
    ray_direction: wp.array(dtype=wp.vec3),
):
    """Compute ray from screen coordinates through view frustum."""
    ndc_x = (2.0 * screen_x) / screen_width - 1.0
    ndc_y = (2.0 * screen_y) / screen_height - 1.0

    inv_view_proj_matrix = wp.inverse(proj_matrix * view_matrix)

    near_point_homogeneous = inv_view_proj_matrix * wp.vec4(ndc_x, ndc_y, -1.0, 1.0)
    far_point_homogeneous = inv_view_proj_matrix * wp.vec4(ndc_x, ndc_y, 1.0, 1.0)

    near_point_world = wp.vec3(
        near_point_homogeneous[0] / near_point_homogeneous[3],
        near_point_homogeneous[1] / near_point_homogeneous[3],
        near_point_homogeneous[2] / near_point_homogeneous[3],
    )
    far_point_world = wp.vec3(
        far_point_homogeneous[0] / far_point_homogeneous[3],
        far_point_homogeneous[1] / far_point_homogeneous[3],
        far_point_homogeneous[2] / far_point_homogeneous[3],
    )

    ray_origin_render = near_point_world
    ray_direction_render = wp.normalize(far_point_world - near_point_world)

    if has_model_matrix > 0:
        ray_origin_homogeneous = inv_model_matrix * wp.vec4(
            ray_origin_render[0], ray_origin_render[1], ray_origin_render[2], 1.0
        )
        ray_origin_scaled = wp.vec3(
            ray_origin_homogeneous[0] / ray_origin_homogeneous[3],
            ray_origin_homogeneous[1] / ray_origin_homogeneous[3],
            ray_origin_homogeneous[2] / ray_origin_homogeneous[3],
        )

        ray_dir_homogeneous = inv_model_matrix * wp.vec4(
            ray_direction_render[0], ray_direction_render[1], ray_direction_render[2], 0.0
        )
        ray_dir_scaled = wp.normalize(wp.vec3(ray_dir_homogeneous[0], ray_dir_homogeneous[1], ray_dir_homogeneous[2]))

        ray_origin[0] = ray_origin_scaled
        ray_direction[0] = ray_dir_scaled
    else:
        ray_origin[0] = ray_origin_render
        ray_direction[0] = ray_direction_render


@wp.kernel
def test_all_capsule_hits(
    ray_origin_world: wp.vec3,
    ray_dir_world: wp.vec3,
    body_transforms: wp.array(dtype=wp.transform),
    local_capsule_starts: wp.array(dtype=wp.vec3),
    local_capsule_ends: wp.array(dtype=wp.vec3),
    local_capsule_radii: wp.array(dtype=float),
    capsule_to_gizmo: wp.array(dtype=int),
    num_active_capsules: int,
    # outputs
    hit_distances: wp.array(dtype=float),
):
    """Test ray intersection with all capsules."""
    tid = wp.tid()

    if tid >= num_active_capsules:
        hit_distances[tid] = float(1e10)
        return

    inv_transform = wp.transform_inverse(body_transforms[capsule_to_gizmo[tid]])
    ray_origin_gizmo_local = wp.transform_point(inv_transform, ray_origin_world)
    ray_dir_gizmo_local = wp.transform_vector(inv_transform, ray_dir_world)

    point_start = local_capsule_starts[tid]
    point_end = local_capsule_ends[tid]
    radius = local_capsule_radii[tid]

    capsule_axis = point_end - point_start
    origin_axis = ray_origin_gizmo_local - point_start

    axis_length_sq = wp.dot(capsule_axis, capsule_axis)
    axis_ray_dot = wp.dot(capsule_axis, ray_dir_gizmo_local)
    axis_origin_dot = wp.dot(capsule_axis, origin_axis)

    a = axis_length_sq - axis_ray_dot * axis_ray_dot
    b = axis_length_sq * wp.dot(origin_axis, ray_dir_gizmo_local) - axis_origin_dot * axis_ray_dot
    c = (
        axis_length_sq * wp.dot(origin_axis, origin_axis)
        - axis_origin_dot * axis_origin_dot
        - radius * radius * axis_length_sq
    )

    discriminant = b * b - a * c

    if discriminant < 0.0:
        hit_distances[tid] = float(1e10)
        return

    t = (-b - wp.sqrt(discriminant)) / a
    y = axis_origin_dot + t * axis_ray_dot

    if y > 0.0 and y < axis_length_sq:
        hit_distances[tid] = t if t > 0.0 else float(1e10)
        return

    t_cap = float(1e10)

    if y <= 0.0:
        L = point_start - ray_origin_gizmo_local
    else:
        L = point_end - ray_origin_gizmo_local

    tca = wp.dot(L, ray_dir_gizmo_local)
    d2 = wp.dot(L, L) - tca * tca

    if d2 <= radius * radius:
        thc = wp.sqrt(radius * radius - d2)
        t0 = tca - thc
        t1 = tca + thc

        if t0 > 0.0:
            t_cap = t0
        elif t1 > 0.0:
            t_cap = t1

    hit_distances[tid] = t_cap


@wp.kernel
def update_single_body_transform(
    body_transforms: wp.array(dtype=wp.transform),
    body_id: int,
    new_transform: wp.transform,
):
    """Update transform for a single body."""
    body_transforms[body_id] = new_transform


@wp.kernel
def find_closest_hit(
    capsule_hit_distances: wp.array(dtype=float),
    num_capsules: int,
    # outputs
    hit_type: wp.array(dtype=int),
    hit_index: wp.array(dtype=int),
):
    """Find closest hit among all tested geometries."""
    min_capsule_dist = float(1e10)
    min_capsule_idx = int(-1)

    for i in range(num_capsules):
        if capsule_hit_distances[i] < min_capsule_dist:
            min_capsule_dist = capsule_hit_distances[i]
            min_capsule_idx = i

    if min_capsule_dist < 1e9:
        hit_type[0] = 1
        hit_index[0] = min_capsule_idx
    else:
        hit_type[0] = 0
        hit_index[0] = -1


class Component:
    """Base class for gizmo components."""

    def __init__(self, name, color, **kwargs):
        self.name = name
        self.color = color

        for key, value in kwargs.items():
            setattr(self, key, value)


class Arrow(Component):
    """Arrow component for translation along an axis."""

    def __init__(self, axis_name):
        super().__init__(
            axis_name,
            COLORS[axis_name],
            axis_vector=AXES[axis_name],
            capsule_radius_factor=0.08,
            collision_radius_factor=0.2,
        )


class Arc(Component):
    """Arc component for rotation around an axis."""

    def __init__(self, axis_name):
        super().__init__(
            axis_name,
            COLORS[axis_name],
            axis_vector=AXES[axis_name],
            from_axis=ARC_MAP[axis_name][0],
            to_axis=ARC_MAP[axis_name][1],
            capsules_per_arc=4,
            capsule_radius=0.015,
        )


class GizmoTarget:
    """Interactive 3D manipulation gizmo."""

    def __init__(self, target_id, renderer, position, rotation, offset, scale_factor, gizmo_system):
        self.id = target_id
        self.renderer = renderer
        self.scale_factor = scale_factor
        self.gizmo_system = gizmo_system

        self.body_id = None
        self.target_transform = wp.transform(
            tuple(position), tuple(rotation if rotation is not None else [0.0, 0.0, 0.0, 1.0])
        )
        self.offset = np.array(offset, dtype=np.float32)

        self.arrows = {name: Arrow(name) for name in "XYZ"}
        self.arcs = {name: Arc(name) for name in "XYZ"}

        self.gizmo_coll_half_height = 0.5 * scale_factor
        self.arc_radius_factor = 1.0
        self.arc_capsule_radius = 0.1 * scale_factor

    def _compute_visual_transform(self):
        """Compute visual transform including offset."""
        position = np.array(self.target_transform[:3]) + self.offset
        rotation = np.array(self.target_transform[3:7])
        return position, rotation

    def update_position(self, position):
        """Update gizmo position."""
        self.target_transform = wp.transform(tuple(position), tuple(self.target_transform[3:7]))
        self._update_visuals()

    def update_rotation(self, rotation):
        """Update gizmo rotation."""
        self.target_transform = wp.transform(tuple(self.target_transform[:3]), tuple(rotation))
        self._update_visuals()

    def get_position(self):
        """Get current position."""
        return np.array(self.target_transform[:3])

    def get_rotation(self):
        """Get current rotation."""
        return np.array(self.target_transform[3:7])

    def _get_arc_collision_segments(self, arc, visual_pos, visual_rot):
        """Get collision segments for arc component."""
        from_world = wp.quat_rotate(wp.quat(visual_rot), wp.vec3(AXES[arc.from_axis]))
        to_world = wp.quat_rotate(wp.quat(visual_rot), wp.vec3(AXES[arc.to_axis]))

        radius = self.gizmo_coll_half_height * 2 * self.arc_radius_factor

        segments = []
        for i in range(arc.capsules_per_arc):
            t1 = i / arc.capsules_per_arc
            t2 = (i + 1) / arc.capsules_per_arc

            def arc_point(t):
                sin_t1 = math.sin((1 - t) * (math.pi / 2))
                sin_t2 = math.sin(t * (math.pi / 2))
                direction = sin_t1 * from_world + sin_t2 * to_world
                return visual_pos + direction * radius

            segments.append((arc_point(t1), arc_point(t2)))

        return segments

    def _get_arc_segment_transforms(self, arc, visual_pos, visual_rot):
        """Get transforms for arc collision segments."""
        segments = self._get_arc_collision_segments(arc, visual_pos, visual_rot)
        transforms = []

        for point_start, point_end in segments:
            position = (point_start + point_end) / 2
            segment_dir = point_end - point_start
            segment_length = np.linalg.norm(segment_dir)
            half_height = segment_length / 2

            if segment_length > 1e-6:
                orientation = get_rotation_quaternion(np.array([0.0, 1.0, 0.0]), segment_dir / segment_length)
            else:
                orientation = np.array([0.0, 0.0, 0.0, 1.0])

            transforms.append((position, orientation, half_height))

        return transforms

    def get_capsule_data_for_render(self):
        """Collect capsule data for rendering."""
        data = {
            "positions": [],
            "rotations": [],
            "scalings": [],
            "colors": [],
        }

        for arrow in self.arrows.values():
            data["positions"].append(self.gizmo_coll_half_height * arrow.axis_vector)
            data["rotations"].append(get_rotation_quaternion(np.array([0.0, 1.0, 0.0]), arrow.axis_vector))
            data["scalings"].append((self.scale_factor * arrow.capsule_radius_factor, self.gizmo_coll_half_height))
            data["colors"].append(arrow.color)

        for arc in self.arcs.values():
            arc_transforms = self._get_arc_segment_transforms(arc, np.zeros(3), np.array([0, 0, 0, 1]))

            for pos, rot, half_height in arc_transforms:
                data["positions"].append(pos)
                data["rotations"].append(rot)
                data["scalings"].append((arc.capsule_radius * self.scale_factor, half_height))
                data["colors"].append(arc.color)

        return (
            data["positions"],
            data["rotations"],
            data["scalings"],
            data["colors"],
        )

    def _update_visuals(self):
        """Update visual representation after transform change."""
        if self.body_id is not None:
            visual_pos, visual_rot = self._compute_visual_transform()
            transform = (*visual_pos, *visual_rot)
            self.gizmo_system._update_body_transform(self.body_id, transform)


class GizmoInstancer:
    """Manages GPU instancing for gizmo rendering."""

    def __init__(self, renderer):
        self.renderer = renderer
        self.device = renderer._device

        self.instancer = ShapeInstancer(renderer._shape_shader, self.device)

        vertices, indices = renderer._create_cylinder_mesh(1.0, 1.0, up_axis=1)
        self.instancer.register_shape(vertices, indices)

        self.num_instances = 0
        self.num_gizmos = 0
        self.body_transforms = None

    def allocate_gizmos(self, gizmo_data_list, body_transforms_array):
        """Allocate GPU resources for all gizmos."""
        all_data = {
            "positions": [],
            "rotations": [],
            "scalings": [],
            "colors": [],
            "bodies": [],
        }

        for positions, rotations, scalings, colors, body_id in gizmo_data_list:
            all_data["positions"].extend(positions)
            all_data["rotations"].extend(rotations)
            all_data["scalings"].extend(scalings)
            all_data["colors"].extend(colors)
            all_data["bodies"].extend([body_id] * len(positions))

        self.num_instances = len(all_data["positions"])
        self.num_gizmos = len(gizmo_data_list)

        self.instance_ids = wp.array(np.arange(self.num_instances), dtype=wp.int32, device=self.device)

        self.instance_transforms = wp.array(
            [(*pos, *rot) for pos, rot in zip(all_data["positions"], all_data["rotations"])],
            dtype=wp.transform,
            device=self.device,
        )

        self.instance_scalings = wp.array(
            [[scale[0], scale[1], scale[0]] for scale in all_data["scalings"]], dtype=wp.vec3, device=self.device
        )

        self.instance_bodies = wp.array(all_data["bodies"], dtype=wp.int32, device=self.device)

        self.body_transforms = body_transforms_array

        self.instancer.allocate_instances(
            positions=[(0, 0, 0)] * self.num_instances,
            rotations=[(0, 0, 0, 1)] * self.num_instances,
            colors1=all_data["colors"],
            colors2=all_data["colors"],
        )

    def update_instance_buffer(self):
        """Update GPU instance buffer with current transforms."""
        with self.instancer:
            wp.launch(
                update_vbo_transforms,
                dim=self.num_instances,
                inputs=[
                    self.instance_ids,
                    self.instance_bodies,
                    self.instance_transforms,
                    self.instance_scalings,
                    self.body_transforms,
                ],
                outputs=[self.instancer.vbo_transforms],
                device=self.device,
                record_tape=False,
            )


class DragState:
    """Tracks state during interactive dragging."""

    def __init__(self, target, component, initial_ray_origin, initial_ray_dir, renderer):
        self.target = target
        self.component = component
        self.renderer = renderer

        visual_pos, visual_rot = target._compute_visual_transform()
        self.start_visual_position = visual_pos
        self.start_visual_rotation = visual_rot
        self.start_target_position = target.get_position()
        self.start_target_rotation = target.get_rotation()

        if isinstance(component, Arrow):
            self.mode = "translate"
            self.axis = wp.quat_rotate(wp.quat(visual_rot), wp.vec3(component.axis_vector))
            self.initial_t = closest_point_on_line_to_ray(visual_pos, self.axis, initial_ray_origin, initial_ray_dir)

        elif isinstance(component, Arc):
            self.mode = "rotate"
            self.rotation_axis = wp.quat_rotate(wp.quat(visual_rot), wp.vec3(component.axis_vector))
            self.prev_angle = None
            self.accumulated_angle = 0.0

    def update(self, ray_origin, ray_direction, mouse_x=None, mouse_y=None, rotation_sensitivity=0.01):
        """Update drag state with new input."""
        if self.mode == "translate":
            t = closest_point_on_line_to_ray(self.start_visual_position, self.axis, ray_origin, ray_direction)
            delta_position = (t - self.initial_t) * self.axis
            self.target.update_position(self.start_target_position + delta_position)

        elif self.mode == "rotate":
            if mouse_x is None or mouse_y is None:
                return

            visual_pos, _ = self.target._compute_visual_transform()
            view_matrix = self.renderer._view_matrix.reshape(4, 4).T
            proj_matrix = self.renderer._projection_matrix.reshape(4, 4).T

            scale_factor = getattr(self.renderer, "scaling", 1.0)
            visual_pos_render = visual_pos * scale_factor

            gizmo_clip = proj_matrix @ view_matrix @ np.append(visual_pos_render, 1.0)

            if abs(gizmo_clip[3]) > 1e-6:
                gizmo_ndc = gizmo_clip[:3] / gizmo_clip[3]
                gizmo_screen_x = (gizmo_ndc[0] + 1.0) * self.renderer.screen_width / 2.0
                gizmo_screen_y = (gizmo_ndc[1] + 1.0) * self.renderer.screen_height / 2.0

                to_mouse = np.array([mouse_x - gizmo_screen_x, mouse_y - gizmo_screen_y])
                distance = np.linalg.norm(to_mouse)

                if distance > 1e-6:
                    to_mouse /= distance
                    tangent = np.array([-to_mouse[1], to_mouse[0]])

                    if self.prev_angle is not None:
                        mouse_delta = np.array([mouse_x - self.prev_angle[0], mouse_y - self.prev_angle[1]])
                        tangent_movement = np.dot(mouse_delta, tangent)
                        rotation_delta = -tangent_movement * rotation_sensitivity * 0.01

                        camera_pos = np.array(
                            [self.renderer.camera_pos.x, self.renderer.camera_pos.y, self.renderer.camera_pos.z],
                            dtype=np.float32,
                        )
                        view_dir = visual_pos_render - camera_pos
                        view_dir /= np.linalg.norm(view_dir)

                        if np.dot(self.rotation_axis, view_dir) < 0:
                            rotation_delta = -rotation_delta

                        self.accumulated_angle += rotation_delta

                        if abs(self.accumulated_angle) > 1e-6:
                            rotation_quat = wp.quat_from_axis_angle(
                                wp.vec3(self.rotation_axis), wp.float32(self.accumulated_angle)
                            )
                            new_rotation = wp.quat(rotation_quat) * wp.quat(self.start_target_rotation)
                            self.target.update_rotation(new_rotation)

                    self.prev_angle = (mouse_x, mouse_y)
            else:
                self.prev_angle = (mouse_x, mouse_y)

    def get_drag_axis_points(self):
        """Get axis visualization points."""
        visual_pos, _ = self.target._compute_visual_transform()
        far_length = 1000.0

        if self.mode == "translate":
            return [
                (visual_pos - self.axis * far_length).tolist(),
                (visual_pos + self.axis * far_length).tolist(),
            ]

        elif self.mode == "rotate":
            return [
                (visual_pos - self.rotation_axis * far_length).tolist(),
                (visual_pos + self.rotation_axis * far_length).tolist(),
            ]

        return []


class GizmoSystem:
    """Manages collection of interactive gizmos."""

    def __init__(self, renderer, scale_factor=1.0, rotation_sensitivity=0.01, max_gizmos=800):
        self.renderer = renderer
        self.scale_factor = scale_factor
        self.rotation_sensitivity = rotation_sensitivity

        self.targets = {}
        self.drag_state = None
        self.position_callback = None
        self.rotation_callback = None
        self.instancer = None
        self.next_body_id = 0
        self._needs_reallocation = False

        self.device = renderer._device

        self.MAX_GIZMOS = max_gizmos
        self.MAX_CAPSULES = self.MAX_GIZMOS * 27

        self._allocate_gpu_arrays()

        self.num_active_capsules = 0
        self.num_gizmos = 0

    def _allocate_gpu_arrays(self):
        """Allocate GPU arrays for collision detection."""
        self.body_transforms_gpu = wp.zeros((self.MAX_GIZMOS,), dtype=wp.transform, device=self.device)
        self.ray_origin_array = wp.zeros((1,), dtype=wp.vec3, device=self.device)
        self.ray_direction_array = wp.zeros((1,), dtype=wp.vec3, device=self.device)
        self.capsule_hit_distances = wp.zeros((self.MAX_CAPSULES,), dtype=float, device=self.device)
        self.hit_type = wp.zeros((1,), dtype=int, device=self.device)
        self.hit_index = wp.zeros((1,), dtype=int, device=self.device)
        self.local_capsule_starts = wp.zeros((self.MAX_CAPSULES,), dtype=wp.vec3, device=self.device)
        self.local_capsule_ends = wp.zeros((self.MAX_CAPSULES,), dtype=wp.vec3, device=self.device)
        self.local_capsule_radii = wp.zeros((self.MAX_CAPSULES,), dtype=float, device=self.device)
        self.capsule_to_gizmo = wp.zeros((self.MAX_CAPSULES,), dtype=int, device=self.device)
        self.capsule_to_component_type = wp.zeros((self.MAX_CAPSULES,), dtype=int, device=self.device)
        self.capsule_to_component_idx = wp.zeros((self.MAX_CAPSULES,), dtype=int, device=self.device)

    def create_target(self, target_id, position, rotation=None, world_offset=None):
        """Create a new gizmo target."""
        if world_offset is None:
            world_offset = [0.0, 0.0, 0.0]

        target = GizmoTarget(target_id, self.renderer, position, rotation, world_offset, self.scale_factor, self)

        target.body_id = self.next_body_id
        self.next_body_id += 1

        self.targets[target_id] = target
        self._needs_reallocation = True

        visual_pos, visual_rot = target._compute_visual_transform()
        transform = wp.transform(tuple(visual_pos), tuple(visual_rot))

        wp.launch(
            update_single_body_transform,
            dim=1,
            inputs=[self.body_transforms_gpu, target.body_id, transform],
            device=self.device,
        )

    def finalize(self):
        """Finalize gizmo allocation after all targets are created."""
        if self._needs_reallocation:
            self._allocate_all_gizmos()

    def _allocate_all_gizmos(self):
        """Allocate rendering and collision resources for all gizmos."""
        if not self.targets:
            return

        render_data_list = []

        collision_data = {
            "capsule_starts": [],
            "capsule_ends": [],
            "capsule_radii": [],
            "capsule_gizmo_ids": [],
            "capsule_comp_types": [],
            "capsule_comp_indices": [],
        }

        local_space_pos = np.zeros(3)
        local_space_rot = np.array([0, 0, 0, 1])

        for target in self.targets.values():
            positions, rotations, scalings, colors = target.get_capsule_data_for_render()
            render_data_list.append((positions, rotations, scalings, colors, target.body_id))

            for i, arrow in enumerate(target.arrows.values()):
                collision_data["capsule_starts"].append(local_space_pos)
                collision_data["capsule_ends"].append(
                    local_space_pos + arrow.axis_vector * (target.gizmo_coll_half_height * 2)
                )
                collision_data["capsule_radii"].append(target.scale_factor * arrow.collision_radius_factor)
                collision_data["capsule_gizmo_ids"].append(target.body_id)
                collision_data["capsule_comp_types"].append(ComponentType.ARROW.value)
                collision_data["capsule_comp_indices"].append(i)

            arc_segments = []
            for arc in target.arcs.values():
                segments = target._get_arc_collision_segments(arc, local_space_pos, local_space_rot)
                arc_segments.extend(segments)

            for i, (start_point, end_point) in enumerate(arc_segments):
                collision_data["capsule_starts"].append(start_point)
                collision_data["capsule_ends"].append(end_point)
                collision_data["capsule_radii"].append(target.arc_capsule_radius)
                collision_data["capsule_gizmo_ids"].append(target.body_id)
                collision_data["capsule_comp_types"].append(ComponentType.ARC.value)
                collision_data["capsule_comp_indices"].append(i)

        self.num_active_capsules = len(collision_data["capsule_starts"])
        self.num_gizmos = len(self.targets)

        if self.num_active_capsules > 0:
            self.local_capsule_starts.assign(np.array(collision_data["capsule_starts"], dtype=np.float32))
            self.local_capsule_ends.assign(np.array(collision_data["capsule_ends"], dtype=np.float32))
            self.local_capsule_radii.assign(np.array(collision_data["capsule_radii"], dtype=np.float32))
            self.capsule_to_gizmo.assign(np.array(collision_data["capsule_gizmo_ids"], dtype=np.int32))
            self.capsule_to_component_type.assign(np.array(collision_data["capsule_comp_types"], dtype=np.int32))
            self.capsule_to_component_idx.assign(np.array(collision_data["capsule_comp_indices"], dtype=np.int32))

        if self.instancer is None:
            self.instancer = GizmoInstancer(self.renderer)
            self.renderer._shape_instancers["gizmo_system"] = self.instancer.instancer

        self.instancer.allocate_gizmos(render_data_list, self.body_transforms_gpu)
        self._needs_reallocation = False

        self._update_all_bodies()

    def _update_body_transform(self, body_id, transform):
        """Update transform for a single body."""
        wp.launch(
            update_single_body_transform,
            dim=1,
            inputs=[self.body_transforms_gpu, body_id, wp.transform(transform[:3], transform[3:])],
            device=self.device,
        )

        if self.instancer and not self._needs_reallocation:
            self._update_all_bodies()

    def _update_all_bodies(self):
        """Update all body transforms in renderer."""
        if self.instancer:
            self.instancer.update_instance_buffer()

    def update_target_position(self, target_id, position):
        """Update target position."""
        if target_id in self.targets:
            if self._needs_reallocation:
                self._allocate_all_gizmos()
            self.targets[target_id].update_position(position)

    def update_target_rotation(self, target_id, rotation):
        """Update target rotation."""
        if target_id in self.targets:
            if self._needs_reallocation:
                self._allocate_all_gizmos()
            self.targets[target_id].update_rotation(rotation)

    def set_callbacks(self, position_callback=None, rotation_callback=None):
        """Set callbacks for position/rotation changes."""
        self.position_callback = position_callback
        self.rotation_callback = rotation_callback

    def on_mouse_press(self, x, y, button, modifiers):
        """Handle mouse press events."""
        import pyglet.window.mouse  # noqa: PLC0415

        if button != pyglet.window.mouse.LEFT or self.num_gizmos == 0:
            return False

        if self._needs_reallocation:
            self._allocate_all_gizmos()

        ray_origin, ray_direction = self._cast_ray_from_screen(x, y)
        if ray_origin is None:
            return False

        self.capsule_hit_distances.fill_(1e10)

        if self.num_active_capsules > 0:
            if self.num_active_capsules > self.MAX_CAPSULES:
                raise ValueError(f"Active capsules ({self.num_active_capsules}) exceeds maximum ({self.MAX_CAPSULES})")

            wp.launch(
                test_all_capsule_hits,
                dim=self.MAX_CAPSULES,
                inputs=[
                    wp.vec3(ray_origin),
                    wp.vec3(ray_direction),
                    self.body_transforms_gpu,
                    self.local_capsule_starts,
                    self.local_capsule_ends,
                    self.local_capsule_radii,
                    self.capsule_to_gizmo,
                    self.num_active_capsules,
                ],
                outputs=[self.capsule_hit_distances],
                device=self.device,
            )

        wp.launch(
            find_closest_hit,
            dim=1,
            inputs=[
                self.capsule_hit_distances,
                self.num_active_capsules,
            ],
            outputs=[self.hit_type, self.hit_index],
            device=self.device,
        )

        hit_type_value = self.hit_type.numpy()[0]
        if hit_type_value == 0:
            return False

        hit_index_value = self.hit_index.numpy()[0]
        target = None
        component = None

        if hit_type_value == 1:
            gizmo_id = self.capsule_to_gizmo.numpy()[hit_index_value]
            component_type = self.capsule_to_component_type.numpy()[hit_index_value]
            component_idx = self.capsule_to_component_idx.numpy()[hit_index_value]

            target = next((t for t in self.targets.values() if t.body_id == gizmo_id), None)

            if target:
                if component_type == ComponentType.ARROW.value:
                    component = list(target.arrows.values())[component_idx]
                elif component_type == ComponentType.ARC.value:
                    arc_idx = component_idx // target.arcs["X"].capsules_per_arc
                    component = list(target.arcs.values())[arc_idx]

        if target and component:
            self.drag_state = DragState(target, component, ray_origin, ray_direction, self.renderer)
            self._update_drag_axis()
            return True

        return False

    def on_mouse_drag(self, x, y, dx, dy, button, modifiers):
        """Handle mouse drag events."""
        import pyglet.window.mouse  # noqa: PLC0415

        if not self.drag_state or not (button & pyglet.window.mouse.LEFT):
            return False

        ray_origin, ray_direction = self._cast_ray_from_screen(x, y)
        if ray_origin is None:
            return False

        prev_position = self.drag_state.target.get_position()
        prev_rotation = self.drag_state.target.get_rotation()

        self.drag_state.update(ray_origin, ray_direction, x, y, self.rotation_sensitivity)

        if self.drag_state.mode == "translate" and self.position_callback:
            if not np.array_equal(prev_position, self.drag_state.target.get_position()):
                self.position_callback(self.drag_state.target.id, self.drag_state.target.get_position())

        if self.drag_state.mode == "rotate" and self.rotation_callback:
            if not np.array_equal(prev_rotation, self.drag_state.target.get_rotation()):
                self.rotation_callback(self.drag_state.target.id, self.drag_state.target.get_rotation())

        # True: block view-pan while rotating; False: keep view following translation
        return self.drag_state.mode == "rotate"

    def on_mouse_release(self, x, y, button, modifiers):
        """Handle mouse release events."""
        import pyglet.window.mouse  # noqa: PLC0415

        if button == pyglet.window.mouse.LEFT and self.drag_state:
            self.drag_state = None
            self._hide_drag_axis()
            return True

        return False

    def _cast_ray_from_screen(self, x, y):
        """Cast ray from screen coordinates."""
        if self.renderer.screen_width == 0 or self.renderer.screen_height == 0:
            return None, None

        if not (hasattr(self.renderer, "_projection_matrix") and hasattr(self.renderer, "_view_matrix")):
            return None, None

        proj_matrix = wp.mat44(self.renderer._projection_matrix.reshape(4, 4).T.flatten())
        view_matrix = wp.mat44(self.renderer._view_matrix.reshape(4, 4).T.flatten())

        has_model = hasattr(self.renderer, "_inv_model_matrix") and self.renderer._inv_model_matrix is not None
        inv_model_matrix = (
            wp.mat44(self.renderer._inv_model_matrix.reshape(4, 4).T.flatten()) if has_model else wp.mat44()
        )

        wp.launch(
            compute_ray_from_screen,
            dim=1,
            inputs=[
                float(x),
                float(y),
                float(self.renderer.screen_width),
                float(self.renderer.screen_height),
                view_matrix,
                proj_matrix,
                int(has_model),
                inv_model_matrix,
            ],
            outputs=[self.ray_origin_array, self.ray_direction_array],
            device=self.device,
        )

        return self.ray_origin_array.numpy()[0], self.ray_direction_array.numpy()[0]

    def _update_drag_axis(self):
        """Update drag axis visualization."""
        if self.drag_state:
            self.renderer.render_line_strip(
                name="drag_axis_visualization",
                vertices=self.drag_state.get_drag_axis_points(),
                color=(1.0, 1.0, 0.0, 0.8),
                radius=0.015 * self.scale_factor,
            )

    def _hide_drag_axis(self):
        """Hide drag axis visualization."""
        self.renderer.render_line_strip(
            name="drag_axis_visualization",
            vertices=[],
            color=(1.0, 1.0, 0.0, 0.8),
            radius=0.015 * self.scale_factor,
        )
