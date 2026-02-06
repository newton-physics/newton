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

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np
import warp as wp

from .ray import MAXVAL
from .types import RenderLightType

if TYPE_CHECKING:
    from .render_context import RenderContext


@wp.kernel(enable_backward=False)
def compute_mesh_bounds(in_meshes: wp.array(dtype=wp.uint64), out_bounds: wp.array2d(dtype=wp.vec3f)):
    tid = wp.tid()

    min_point = wp.vec3(MAXVAL)
    max_point = wp.vec3(-MAXVAL)

    if in_meshes[tid] != 0:
        mesh = wp.mesh_get(in_meshes[tid])
        for i in range(mesh.points.shape[0]):
            min_point = wp.min(min_point, mesh.points[i])
            max_point = wp.max(max_point, mesh.points[i])

    out_bounds[tid, 0] = min_point
    out_bounds[tid, 1] = max_point


@wp.kernel(enable_backward=False)
def compute_pinhole_camera_rays(
    width: int,
    height: int,
    camera_fovs: wp.array(dtype=wp.float32),
    out_rays: wp.array(dtype=wp.vec3f, ndim=4),
):
    camera_index, py, px = wp.tid()
    aspect_ratio = float(width) / float(height)
    u = (float(px) + 0.5) / float(width) - 0.5
    v = (float(py) + 0.5) / float(height) - 0.5
    h = wp.tan(camera_fovs[camera_index] / 2.0)
    ray_direction_camera_space = wp.vec3f(u * 2.0 * h * aspect_ratio, -v * 2.0 * h, -1.0)
    out_rays[camera_index, py, px, 0] = wp.vec3f(0.0)
    out_rays[camera_index, py, px, 1] = wp.normalize(ray_direction_camera_space)


@wp.kernel(enable_backward=False)
def flatten_color_image(
    color_image: wp.array(dtype=wp.uint32, ndim=4),
    buffer: wp.array(dtype=wp.uint8, ndim=3),
    width: wp.int32,
    height: wp.int32,
    num_cameras: wp.int32,
    num_worlds_per_row: wp.int32,
):
    world_id, camera_id, y, x = wp.tid()

    view_id = world_id * num_cameras + camera_id

    row = view_id // num_worlds_per_row
    col = view_id % num_worlds_per_row

    px = col * width + x
    py = row * height + y
    color = color_image[world_id, camera_id, y, x]

    buffer[py, px, 0] = wp.uint8((color >> wp.uint32(0)) & wp.uint32(0xFF))
    buffer[py, px, 1] = wp.uint8((color >> wp.uint32(8)) & wp.uint32(0xFF))
    buffer[py, px, 2] = wp.uint8((color >> wp.uint32(16)) & wp.uint32(0xFF))
    buffer[py, px, 3] = wp.uint8((color >> wp.uint32(24)) & wp.uint32(0xFF))


@wp.kernel(enable_backward=False)
def flatten_normal_image(
    normal_image: wp.array(dtype=wp.vec3f, ndim=4),
    buffer: wp.array(dtype=wp.uint8, ndim=3),
    width: wp.int32,
    height: wp.int32,
    num_cameras: wp.int32,
    num_worlds_per_row: wp.int32,
):
    world_id, camera_id, y, x = wp.tid()

    view_id = world_id * num_cameras + camera_id

    row = view_id // num_worlds_per_row
    col = view_id % num_worlds_per_row

    px = col * width + x
    py = row * height + y
    normal = normal_image[world_id, camera_id, y, x] * 0.5 + wp.vec3f(0.5)

    buffer[py, px, 0] = wp.uint8(normal[0] * 255.0)
    buffer[py, px, 1] = wp.uint8(normal[1] * 255.0)
    buffer[py, px, 2] = wp.uint8(normal[2] * 255.0)
    buffer[py, px, 3] = wp.uint8(255)


@wp.kernel(enable_backward=False)
def find_depth_range(depth_image: wp.array(dtype=wp.float32, ndim=4), depth_range: wp.array(dtype=wp.float32)):
    world_id, camera_id, y, x = wp.tid()
    depth = depth_image[world_id, camera_id, y, x]
    if depth > 0:
        wp.atomic_min(depth_range, 0, depth)
        wp.atomic_max(depth_range, 1, depth)


@wp.kernel(enable_backward=False)
def flatten_depth_image(
    depth_image: wp.array(dtype=wp.float32, ndim=4),
    buffer: wp.array(dtype=wp.uint8, ndim=3),
    depth_range: wp.array(dtype=wp.float32),
    width: wp.int32,
    height: wp.int32,
    num_cameras: wp.int32,
    num_worlds_per_row: wp.int32,
):
    world_id, camera_id, y, x = wp.tid()

    view_id = world_id * num_cameras + camera_id

    row = view_id // num_worlds_per_row
    col = view_id % num_worlds_per_row

    px = col * width + x
    py = row * height + y

    value = wp.uint8(0)
    depth = depth_image[world_id, camera_id, y, x]
    if depth > 0:
        denom = wp.max(depth_range[1] - depth_range[0], 1e-6)
        value = wp.uint8(255.0 - ((depth - depth_range[0]) / denom) * 205.0)

    buffer[py, px, 0] = value
    buffer[py, px, 1] = value
    buffer[py, px, 2] = value
    buffer[py, px, 3] = value


class Utils:
    def __init__(self, render_context: RenderContext):
        self.__render_context = render_context

    def compute_mesh_bounds(self):
        wp.launch(
            kernel=compute_mesh_bounds,
            dim=self.__render_context.mesh_ids.size,
            inputs=[self.__render_context.mesh_ids, self.__render_context.mesh_bounds],
            device=self.__render_context.device,
        )

    def compute_pinhole_camera_rays(self, width: int, height: int, camera_fovs: wp.array(dtype=wp.float32)) -> wp.array(
        dtype=wp.vec3f, ndim=4
    ):
        num_cameras = camera_fovs.size

        camera_rays = wp.empty((num_cameras, height, width, 2), dtype=wp.vec3f, device=self.__render_context.device)

        wp.launch(
            kernel=compute_pinhole_camera_rays,
            dim=(num_cameras, height, width),
            inputs=[
                width,
                height,
                camera_fovs,
                camera_rays,
            ],
            device=self.__render_context.device,
        )

        return camera_rays

    def flatten_color_image_to_rgba(
        self,
        image: wp.array(dtype=wp.uint32, ndim=4),
        out_buffer: wp.array(dtype=wp.uint8, ndim=3) | None = None,
        num_worlds_per_row: int | None = None,
    ) -> wp.array(dtype=wp.uint8, ndim=3):
        num_cameras = image.shape[1]
        height = image.shape[2]
        width = image.shape[3]

        out_buffer, num_worlds_per_row = self.__reshape_buffer_for_flatten(
            width, height, num_cameras, out_buffer, num_worlds_per_row
        )

        wp.launch(
            flatten_color_image,
            (
                self.__render_context.num_worlds,
                num_cameras,
                height,
                width,
            ),
            [
                image,
                out_buffer,
                width,
                height,
                num_cameras,
                num_worlds_per_row,
            ],
            device=self.__render_context.device,
        )
        return out_buffer

    def flatten_normal_image_to_rgba(
        self,
        image: wp.array(dtype=wp.vec3f, ndim=4),
        out_buffer: wp.array(dtype=wp.uint8, ndim=3) | None = None,
        num_worlds_per_row: int | None = None,
    ) -> wp.array(dtype=wp.uint8, ndim=3):
        num_cameras = image.shape[1]
        height = image.shape[2]
        width = image.shape[3]

        out_buffer, num_worlds_per_row = self.__reshape_buffer_for_flatten(
            width, height, num_cameras, out_buffer, num_worlds_per_row
        )

        wp.launch(
            flatten_normal_image,
            (
                self.__render_context.num_worlds,
                num_cameras,
                height,
                width,
            ),
            [
                image,
                out_buffer,
                width,
                height,
                num_cameras,
                num_worlds_per_row,
            ],
            device=self.__render_context.device,
        )
        return out_buffer

    def flatten_depth_image_to_rgba(
        self,
        image: wp.array(dtype=wp.float32, ndim=4),
        out_buffer: wp.array(dtype=wp.uint8, ndim=3) | None = None,
        num_worlds_per_row: int | None = None,
        depth_range: wp.array(dtype=wp.float32) | None = None,
    ) -> wp.array(dtype=wp.uint8, ndim=3):
        num_cameras = image.shape[1]
        height = image.shape[2]
        width = image.shape[3]

        out_buffer, num_worlds_per_row = self.__reshape_buffer_for_flatten(
            width, height, num_cameras, out_buffer, num_worlds_per_row
        )

        if depth_range is None:
            depth_range = wp.array([MAXVAL, 0.0], dtype=wp.float32, device=self.__render_context.device)
            wp.launch(find_depth_range, image.shape, [image, depth_range], device=self.__render_context.device)

        wp.launch(
            flatten_depth_image,
            (
                self.__render_context.num_worlds,
                num_cameras,
                height,
                width,
            ),
            [
                image,
                out_buffer,
                depth_range,
                width,
                height,
                num_cameras,
                num_worlds_per_row,
            ],
            device=self.__render_context.device,
        )
        return out_buffer

    def assign_random_colors_per_world(self, seed: int = 100):
        if not self.__render_context.num_shapes_total:
            return
        colors = np.random.default_rng(seed).random((self.__render_context.num_shapes_total, 4)) * 0.5 + 0.5
        colors[:, -1] = 1.0
        self.__render_context.shape_colors = wp.array(
            colors[self.__render_context.shape_world_index.numpy() % len(colors)],
            dtype=wp.vec4f,
            device=self.__render_context.device,
        )

    def assign_random_colors_per_shape(self, seed: int = 100):
        colors = np.random.default_rng(seed).random((self.__render_context.num_shapes_total, 4)) * 0.5 + 0.5
        colors[:, -1] = 1.0
        self.__render_context.shape_colors = wp.array(colors, dtype=wp.vec4f, device=self.__render_context.device)

    def create_default_light(self, enable_shadows: bool = True, direction: wp.vec3f | None = None):
        self.__render_context.options.enable_shadows = enable_shadows
        self.__render_context.lights_active = wp.array([True], dtype=wp.bool, device=self.__render_context.device)
        self.__render_context.lights_type = wp.array(
            [RenderLightType.DIRECTIONAL], dtype=wp.int32, device=self.__render_context.device
        )
        self.__render_context.lights_cast_shadow = wp.array([True], dtype=wp.bool, device=self.__render_context.device)
        self.__render_context.lights_position = wp.array(
            [wp.vec3f(0.0)], dtype=wp.vec3f, device=self.__render_context.device
        )
        self.__render_context.lights_orientation = wp.array(
            [direction if direction is not None else wp.vec3f(-0.57735026, 0.57735026, -0.57735026)],
            dtype=wp.vec3f,
            device=self.__render_context.device,
        )

    def assign_checkerboard_material_to_all_shapes(self, resolution: int = 64, checker_size: int = 32):
        checkerboard = (
            (np.arange(resolution) // checker_size)[:, None] + (np.arange(resolution) // checker_size)
        ) % 2 == 0
        pixels = np.where(checkerboard, 0xFF808080, 0xFFBFBFBF).astype(np.uint32).flatten()

        self.__render_context.options.enable_textures = True
        self.__render_context.texture_data = wp.array(pixels, dtype=wp.uint32, device=self.__render_context.device)
        self.__render_context.texture_offsets = wp.array([0], dtype=wp.int32, device=self.__render_context.device)
        self.__render_context.texture_width = wp.array(
            [resolution], dtype=wp.int32, device=self.__render_context.device
        )
        self.__render_context.texture_height = wp.array(
            [resolution], dtype=wp.int32, device=self.__render_context.device
        )

        self.__render_context.material_texture_ids = wp.array([0], dtype=wp.int32, device=self.__render_context.device)
        self.__render_context.material_texture_repeat = wp.array(
            [wp.vec2f(1.0)], dtype=wp.vec2f, device=self.__render_context.device
        )
        self.__render_context.material_rgba = wp.array(
            [wp.vec4f(1.0)], dtype=wp.vec4f, device=self.__render_context.device
        )

        self.__render_context.shape_materials = wp.array(
            np.full(self.__render_context.num_shapes_total, fill_value=0, dtype=np.int32),
            dtype=wp.int32,
            device=self.__render_context.device,
        )

    def __reshape_buffer_for_flatten(
        self,
        width: int,
        height: int,
        num_cameras: int,
        out_buffer: wp.array | None = None,
        num_worlds_per_row: int | None = None,
    ) -> wp.array():
        num_worlds_and_cameras = self.__render_context.num_worlds * num_cameras
        if not num_worlds_per_row:
            num_worlds_per_row = math.ceil(math.sqrt(num_worlds_and_cameras))
        num_worlds_per_col = math.ceil(num_worlds_and_cameras / num_worlds_per_row)

        if out_buffer is None:
            return wp.empty(
                (
                    num_worlds_per_col * height,
                    num_worlds_per_row * width,
                    4,
                ),
                dtype=wp.uint8,
                device=self.__render_context.device,
            ), num_worlds_per_row

        return out_buffer.reshape((num_worlds_per_col * height, num_worlds_per_row * width, 4)), num_worlds_per_row
