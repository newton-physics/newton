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
import warp as wp

from ..geometry import ShapeFlags
from ..sim import Model, State
from ..viewer.camera import Camera
from .warp_raytrace import GeomType, LightType, RenderContext


@wp.kernel
def convert_newton_transform(
    in_body_transforms: wp.array(dtype=wp.transform),
    in_shape_body: wp.array(dtype=wp.int32),
    in_transform: wp.array(dtype=wp.transformf),
    in_scale: wp.array(dtype=wp.vec3f),
    out_position: wp.array(dtype=wp.vec3f),
    out_matrix: wp.array(dtype=wp.mat33f),
    out_sizes: wp.array(dtype=wp.vec3f),
):
    tid = wp.tid()

    body = in_shape_body[tid]
    body_transform = wp.transform_identity()
    if body >= 0:
        body_transform = in_body_transforms[body]

    transform = wp.mul(body_transform, in_transform[tid])
    out_position[tid] = wp.transform_get_translation(transform)
    out_matrix[tid] = wp.quat_to_matrix(wp.transform_get_rotation(transform))
    out_sizes[tid] = in_scale[tid]


@wp.kernel
def compute_mesh_bounds(in_meshes: wp.array(dtype=wp.uint64), out_bounds: wp.array2d(dtype=wp.vec3f)):
    tid = wp.tid()

    min_point = wp.vec3f()
    max_point = wp.vec3f()

    if in_meshes[tid] != 0:
        mesh = wp.mesh_get(in_meshes[tid])
        for i in range(mesh.points.shape[0]):
            min_point = wp.min(min_point, mesh.points[i])
            max_point = wp.max(max_point, mesh.points[i])

    out_bounds[tid][0] = min_point
    out_bounds[tid][1] = max_point


@wp.func
def is_supported_shape_type(shape_type: wp.int32) -> wp.bool:
    if shape_type == GeomType.BOX:
        return True
    if shape_type == GeomType.CAPSULE:
        return True
    if shape_type == GeomType.CYLINDER:
        return True
    if shape_type == GeomType.ELLIPSOID:
        return True
    if shape_type == GeomType.PLANE:
        return True
    if shape_type == GeomType.SPHERE:
        return True
    if shape_type == GeomType.CONE:
        return True
    if shape_type == GeomType.MESH:
        return True
    return False


@wp.kernel
def compute_enabled_shapes(
    shape_type: wp.array(dtype=wp.int32),
    shape_flags: wp.array(dtype=wp.int32),
    out_geom_enabled: wp.array(dtype=wp.int32),
    out_mesh_indices: wp.array(dtype=wp.int32),
    out_geom_enabled_count: wp.array(dtype=wp.int32),
):
    tid = wp.tid()

    out_mesh_indices[tid] = tid

    if not bool(shape_flags[tid] & ShapeFlags.VISIBLE):
        return

    if not is_supported_shape_type(shape_type[tid]):
        return

    index = wp.atomic_add(out_geom_enabled_count, 0, 1)
    out_geom_enabled[index] = tid


class TiledCameraSensor:
    def __init__(self, model: Model, num_cameras: int, width: int, height: int):
        self.model = model

        self.render_context = RenderContext(width, height, True, True, False, False, self.model.num_worlds, True)
        self.render_context.num_cameras = num_cameras
        self.render_context.mesh_ids = model.shape_source_ptr
        self.render_context.geom_mesh_indices = wp.empty(self.model.shape_count, dtype=wp.int32)
        self.render_context.mesh_bounds = wp.empty((self.model.shape_count, 2), dtype=wp.vec3f, ndim=2)

        self.render_context.geom_enabled = wp.empty(self.model.shape_count, dtype=wp.int32)
        self.render_context.geom_types = model.shape_type
        self.render_context.geom_sizes = wp.empty(self.model.shape_count, dtype=wp.vec3f)
        self.render_context.geom_positions = wp.empty(self.model.shape_count, dtype=wp.vec3f)
        self.render_context.geom_orientations = wp.empty(self.model.shape_count, dtype=wp.mat33f)
        self.render_context.geom_materials = wp.array(
            np.full(self.model.shape_count, fill_value=-1, dtype=np.int32), dtype=wp.int32
        )
        self.render_context.geom_colors = wp.array(
            np.full((self.model.shape_count, 4), fill_value=1.0, dtype=wp.float32), dtype=wp.vec4f
        )
        self.render_context.geom_world_index = self.model.shape_world

        num_enabled_geoms = wp.zeros(1, dtype=wp.int32)
        wp.launch(
            kernel=compute_enabled_shapes,
            dim=self.model.shape_count,
            inputs=[
                model.shape_type,
                model.shape_flags,
                self.render_context.geom_enabled,
                self.render_context.geom_mesh_indices,
                num_enabled_geoms,
            ],
        )
        self.render_context.num_geom_in_bvh = int(num_enabled_geoms.numpy()[0])
        self.render_context.init_outputs()

        wp.launch(
            kernel=compute_mesh_bounds,
            dim=self.model.shape_count,
            inputs=[self.render_context.mesh_ids, self.render_context.mesh_bounds],
        )

    def render(self, state: State):
        if self.model.shape_count:
            wp.launch(
                kernel=convert_newton_transform,
                dim=self.model.shape_count,
                inputs=[
                    state.body_q,
                    self.model.shape_body,
                    self.model.shape_transform,
                    self.model.shape_scale,
                    self.render_context.geom_positions,
                    self.render_context.geom_orientations,
                    self.render_context.geom_sizes,
                ],
            )
            self.render_context.render()

    @property
    def color_image(self) -> wp.array:
        return self.render_context.output.color_image

    @property
    def depth_image(self) -> wp.array:
        return self.render_context.output.depth_image

    def update_cameras(self, cameras: list[Camera]):
        assert len(cameras) == self.render_context.num_cameras, "Number of cameras does not match initial setup."

        self.render_context.camera_fovs = wp.array([math.radians(camera.fov) for camera in cameras], dtype=wp.float32)
        self.render_context.camera_positions = wp.array([camera.pos for camera in cameras], dtype=wp.vec3f)
        self.render_context.camera_orientations = wp.array(
            [np.delete(camera.get_view_matrix(), np.arange(3, 16, 4))[:9] for camera in cameras], dtype=wp.mat33f
        )

    def save_color_image(self, filename: str) -> bool:
        try:
            from PIL import Image  # noqa: PLC0415
        except ImportError:
            print("Failed to import PIL.Image, not saving image.")
            return False

        num_worlds_and_cameras = self.render_context.num_worlds * self.render_context.num_cameras
        rows = math.ceil(math.sqrt(num_worlds_and_cameras))
        cols = math.ceil(num_worlds_and_cameras / rows)

        tile_data = self.color_image.numpy().astype(np.uint32)
        tile_data = tile_data.reshape(num_worlds_and_cameras, self.render_context.width * self.render_context.height)

        if rows * cols > num_worlds_and_cameras:
            extended_data = np.zeros(
                (rows * cols, self.render_context.width * self.render_context.height), dtype=np.uint32
            )
            extended_data[: tile_data.shape[0]] = tile_data
            tile_data = extended_data

        r = (tile_data & 0xFF).astype(np.uint8)
        g = ((tile_data >> 8) & 0xFF).astype(np.uint8)
        b = ((tile_data >> 16) & 0xFF).astype(np.uint8)

        tile_data = np.dstack([r, g, b])
        tile_data = tile_data.reshape(rows, cols, self.render_context.width, self.render_context.height, 3)
        tile_data = tile_data.transpose(0, 2, 1, 3, 4)
        tile_data = tile_data.reshape(rows * self.render_context.width, cols * self.render_context.height, 3)
        Image.fromarray(tile_data).save(filename)
        return True

    def save_depth_image(self, filename: str) -> bool:
        try:
            from PIL import Image  # noqa: PLC0415
        except ImportError:
            print("Failed to import PIL.Image, not saving image.")
            return False

        num_worlds_and_cameras = self.render_context.num_worlds * self.render_context.num_cameras
        rows = math.ceil(math.sqrt(num_worlds_and_cameras))
        cols = math.ceil(num_worlds_and_cameras / rows)

        tile_data = self.color_image.numpy().astype(np.float32)
        tile_data = tile_data.reshape(num_worlds_and_cameras, self.render_context.width * self.render_context.height)

        tile_data[tile_data < 0] = 0

        if rows * cols > num_worlds_and_cameras:
            extended_data = np.zeros(
                (rows * cols, self.render_context.width * self.render_context.height), dtype=np.float32
            )
            extended_data[: tile_data.shape[0]] = tile_data
            tile_data = extended_data

        # Normalize positive values to 0-255 range
        pos_mask = tile_data > 0
        if np.any(pos_mask):
            pos_vals = tile_data[pos_mask]
            min_depth = pos_vals.min()
            max_depth = pos_vals.max()
            denom = max(max_depth - min_depth, 1e-6)
            # Invert: closer objects = brighter, farther = darker
            # Scale to 50-255 range (so background/no-hit stays at 0)
            tile_data[pos_mask] = 255 - ((pos_vals - min_depth) / denom) * 205

        tile_data = np.clip(tile_data, 0, 255).astype(np.uint8)
        tile_data = tile_data.reshape(rows, cols, self.render_context.width, self.render_context.height)
        tile_data = tile_data.transpose(0, 2, 1, 3)
        tile_data = tile_data.reshape(rows * self.render_context.width, cols * self.render_context.height)
        Image.fromarray(tile_data).save(filename)
        return True

    def assign_debug_colors_per_world(self, seed: int = 100):
        colors = np.random.default_rng(seed).random((self.model.shape_count, 4)) * 0.5 + 0.5
        colors[:, -1] = 1.0
        self.render_context.geom_colors = wp.array(colors[self.model.shape_world.numpy() % len(colors)], dtype=wp.vec4f)

    def assign_debug_colors_per_shape(self, seed: int = 100):
        colors = np.random.default_rng(seed).random((self.model.shape_count, 4)) * 0.5 + 0.5
        colors[:, -1] = 1.0
        self.render_context.geom_colors = wp.array(colors, dtype=wp.vec4f)

    def create_default_light(self):
        self.render_context.num_lights = 1
        self.render_context.use_shadows = True
        self.render_context.lights_active = wp.array([True], dtype=wp.bool)
        self.render_context.lights_type = wp.array([LightType.DIRECTIONAL], dtype=wp.int32)
        self.render_context.lights_cast_shadow = wp.array([True], dtype=wp.bool)
        self.render_context.lights_position = wp.array([wp.vec3f(0.0)], dtype=wp.vec3f)
        self.render_context.lights_orientation = wp.array(
            [wp.vec3f(-0.57735026, 0.57735026, -0.57735026)], dtype=wp.vec3f
        )
