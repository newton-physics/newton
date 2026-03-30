# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np
import warp as wp

from ...core import MAXVAL
from .types import RenderLightType, TextureData

if TYPE_CHECKING:
    from .render_context import RenderContext


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
    camera_count: wp.int32,
    worlds_per_row: wp.int32,
):
    world_id, camera_id, y, x = wp.tid()

    view_id = world_id * camera_count + camera_id

    row = view_id // worlds_per_row
    col = view_id % worlds_per_row

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
    camera_count: wp.int32,
    worlds_per_row: wp.int32,
):
    world_id, camera_id, y, x = wp.tid()

    view_id = world_id * camera_count + camera_id

    row = view_id // worlds_per_row
    col = view_id % worlds_per_row

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
    camera_count: wp.int32,
    worlds_per_row: wp.int32,
):
    world_id, camera_id, y, x = wp.tid()

    view_id = world_id * camera_count + camera_id

    row = view_id // worlds_per_row
    col = view_id % worlds_per_row

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
    """Utility functions for the RenderContext."""

    def __init__(self, render_context: RenderContext):
        self.__render_context = render_context

    def create_color_image_output(self, width: int, height: int, camera_count: int = 1) -> wp.array(
        dtype=wp.uint32, ndim=4
    ):
        """Create a color output array for :meth:`update`.

        Args:
            width: Image width [px].
            height: Image height [px].
            camera_count: Number of cameras.

        Returns:
            Array of shape ``(world_count, camera_count, height, width)``, dtype ``uint32``.
        """
        return wp.zeros(
            (self.__render_context.world_count, camera_count, height, width),
            dtype=wp.uint32,
            device=self.__render_context.device,
        )

    def create_depth_image_output(self, width: int, height: int, camera_count: int = 1) -> wp.array(
        dtype=wp.float32, ndim=4
    ):
        """Create a depth output array for :meth:`update`.

        Args:
            width: Image width [px].
            height: Image height [px].
            camera_count: Number of cameras.

        Returns:
            Array of shape ``(world_count, camera_count, height, width)``, dtype ``float32``.
        """
        return wp.zeros(
            (self.__render_context.world_count, camera_count, height, width),
            dtype=wp.float32,
            device=self.__render_context.device,
        )

    def create_shape_index_image_output(self, width: int, height: int, camera_count: int = 1) -> wp.array(
        dtype=wp.uint32, ndim=4
    ):
        """Create a shape-index output array for :meth:`update`.

        Args:
            width: Image width [px].
            height: Image height [px].
            camera_count: Number of cameras.

        Returns:
            Array of shape ``(world_count, camera_count, height, width)``, dtype ``uint32``.
        """
        return wp.zeros(
            (self.__render_context.world_count, camera_count, height, width),
            dtype=wp.uint32,
            device=self.__render_context.device,
        )

    def create_normal_image_output(self, width: int, height: int, camera_count: int = 1) -> wp.array(
        dtype=wp.vec3f, ndim=4
    ):
        """Create a normal output array for :meth:`update`.

        Args:
            width: Image width [px].
            height: Image height [px].
            camera_count: Number of cameras.

        Returns:
            Array of shape ``(world_count, camera_count, height, width)``, dtype ``vec3f``.
        """
        return wp.zeros(
            (self.__render_context.world_count, camera_count, height, width),
            dtype=wp.vec3f,
            device=self.__render_context.device,
        )

    def create_albedo_image_output(self, width: int, height: int, camera_count: int = 1) -> wp.array(
        dtype=wp.uint32, ndim=4
    ):
        """Create an albedo output array for :meth:`update`.

        Args:
            width: Image width [px].
            height: Image height [px].
            camera_count: Number of cameras.

        Returns:
            Array of shape ``(world_count, camera_count, height, width)``, dtype ``uint32``.
        """
        return wp.zeros(
            (self.__render_context.world_count, camera_count, height, width),
            dtype=wp.uint32,
            device=self.__render_context.device,
        )

    def compute_pinhole_camera_rays(
        self, width: int, height: int, camera_fovs: float | list[float] | np.ndarray | wp.array(dtype=wp.float32)
    ) -> wp.array(dtype=wp.vec3f, ndim=4):
        """Compute camera-space ray directions for pinhole cameras.

        Generates rays in camera space (origin at the camera center, direction normalized) for each pixel based on the
        vertical field of view.

        Args:
            width: Image width [px].
            height: Image height [px].
            camera_fovs: Vertical FOV angles [rad], shape ``(camera_count,)``.

        Returns:
            camera_rays: Shape ``(camera_count, height, width, 2)``, dtype ``vec3f``.
        """
        if isinstance(camera_fovs, float):
            camera_fovs = wp.array([camera_fovs], dtype=wp.float32, device=self.__render_context.device)
        elif isinstance(camera_fovs, list):
            camera_fovs = wp.array(camera_fovs, dtype=wp.float32, device=self.__render_context.device)
        elif isinstance(camera_fovs, np.ndarray):
            camera_fovs = wp.array(camera_fovs, dtype=wp.float32, device=self.__render_context.device)

        camera_count = camera_fovs.size

        camera_rays = wp.empty((camera_count, height, width, 2), dtype=wp.vec3f, device=self.__render_context.device)

        wp.launch(
            kernel=compute_pinhole_camera_rays,
            dim=(camera_count, height, width),
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
        worlds_per_row: int | None = None,
    ) -> wp.array(dtype=wp.uint8, ndim=3):
        """Flatten rendered color image to a tiled RGBA buffer.

        Arranges ``(world_count * camera_count)`` tiles in a grid. Each tile shows one camera's view of one world.

        Args:
            image: Color output from :meth:`update`, shape ``(world_count, camera_count, height, width)``.
            out_buffer: Pre-allocated RGBA buffer. If None, allocates a new one.
            worlds_per_row: Tiles per row in the grid. If None, picks a square-ish layout.
        """
        camera_count = image.shape[1]
        height = image.shape[2]
        width = image.shape[3]

        out_buffer, worlds_per_row = self.__reshape_buffer_for_flatten(
            width, height, camera_count, out_buffer, worlds_per_row
        )

        wp.launch(
            flatten_color_image,
            (
                self.__render_context.world_count,
                camera_count,
                height,
                width,
            ),
            [
                image,
                out_buffer,
                width,
                height,
                camera_count,
                worlds_per_row,
            ],
            device=self.__render_context.device,
        )
        return out_buffer

    def flatten_normal_image_to_rgba(
        self,
        image: wp.array(dtype=wp.vec3f, ndim=4),
        out_buffer: wp.array(dtype=wp.uint8, ndim=3) | None = None,
        worlds_per_row: int | None = None,
    ) -> wp.array(dtype=wp.uint8, ndim=3):
        """Flatten rendered normal image to a tiled RGBA buffer.

        Arranges ``(world_count * camera_count)`` tiles in a grid. Each tile shows one camera's view of one world.

        Args:
            image: Normal output from :meth:`update`, shape ``(world_count, camera_count, height, width)``.
            out_buffer: Pre-allocated RGBA buffer. If None, allocates a new one.
            worlds_per_row: Tiles per row in the grid. If None, picks a square-ish layout.
        """
        camera_count = image.shape[1]
        height = image.shape[2]
        width = image.shape[3]

        out_buffer, worlds_per_row = self.__reshape_buffer_for_flatten(
            width, height, camera_count, out_buffer, worlds_per_row
        )

        wp.launch(
            flatten_normal_image,
            (
                self.__render_context.world_count,
                camera_count,
                height,
                width,
            ),
            [
                image,
                out_buffer,
                width,
                height,
                camera_count,
                worlds_per_row,
            ],
            device=self.__render_context.device,
        )
        return out_buffer

    def flatten_depth_image_to_rgba(
        self,
        image: wp.array(dtype=wp.float32, ndim=4),
        out_buffer: wp.array(dtype=wp.uint8, ndim=3) | None = None,
        worlds_per_row: int | None = None,
        depth_range: wp.array(dtype=wp.float32) | None = None,
    ) -> wp.array(dtype=wp.uint8, ndim=3):
        """Flatten rendered depth image to a tiled RGBA buffer.

        Encodes depth as grayscale: inverts values (closer = brighter) and normalizes to the ``[50, 255]``
        range. Background pixels (no hit) remain black.

        Args:
            image: Depth output from :meth:`update`, shape ``(world_count, camera_count, height, width)``.
            out_buffer: Pre-allocated RGBA buffer. If None, allocates a new one.
            worlds_per_row: Tiles per row in the grid. If None, picks a square-ish layout.
            depth_range: Depth range to normalize to, shape ``(2,)`` ``[near, far]``. If None, computes from *image*.
        """
        camera_count = image.shape[1]
        height = image.shape[2]
        width = image.shape[3]

        out_buffer, worlds_per_row = self.__reshape_buffer_for_flatten(
            width, height, camera_count, out_buffer, worlds_per_row
        )

        if depth_range is None:
            depth_range = wp.array([MAXVAL, 0.0], dtype=wp.float32, device=self.__render_context.device)
            wp.launch(find_depth_range, image.shape, [image, depth_range], device=self.__render_context.device)

        wp.launch(
            flatten_depth_image,
            (
                self.__render_context.world_count,
                camera_count,
                height,
                width,
            ),
            [
                image,
                out_buffer,
                depth_range,
                width,
                height,
                camera_count,
                worlds_per_row,
            ],
            device=self.__render_context.device,
        )
        return out_buffer

    def assign_random_colors_per_world(self, seed: int = 100):
        """Assign each world a random color, applied to all its shapes.

        Args:
            seed: Random seed.
        """
        if not self.__render_context.shape_count_total:
            return
        colors = np.random.default_rng(seed).random((self.__render_context.shape_count_total, 4)) * 0.5 + 0.5
        colors[:, -1] = 1.0
        self.__render_context.shape_colors = wp.array(
            colors[self.__render_context.shape_world_index.numpy() % len(colors)],
            dtype=wp.vec4f,
            device=self.__render_context.device,
        )

    def assign_random_colors_per_shape(self, seed: int = 100):
        """Assign a random color to each shape.

        Args:
            seed: Random seed.
        """
        colors = np.random.default_rng(seed).random((self.__render_context.shape_count_total, 4)) * 0.5 + 0.5
        colors[:, -1] = 1.0
        self.__render_context.shape_colors = wp.array(colors, dtype=wp.vec4f, device=self.__render_context.device)

    def create_default_light(self, enable_shadows: bool = True, direction: wp.vec3f | None = None):
        """Create a default directional light oriented at ``(-1, 1, -1)``.

        Args:
            enable_shadows: Enable shadow casting for this light.
            direction: Normalized light direction. If ``None``, defaults to
                (normalized ``(-1, 1, -1)``).
        """
        self.__render_context.config.enable_shadows = enable_shadows
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
        """Assign a gray checkerboard texture material to all shapes.
        Creates a gray checkerboard pattern texture and applies it to all shapes
        in the scene.

        Args:
            resolution: Texture resolution in pixels (square texture).
            checker_size: Size of each checkerboard square in pixels.
        """
        checkerboard = (
            (np.arange(resolution) // checker_size)[:, None] + (np.arange(resolution) // checker_size)
        ) % 2 == 0

        pixels = np.where(checkerboard, 0xFF808080, 0xFFBFBFBF).astype(np.uint32)

        texture_ids = np.full(self.__render_context.shape_count_total, fill_value=0, dtype=np.int32)

        self.__checkerboard_data = TextureData()
        self.__checkerboard_data.texture = wp.Texture2D(
            pixels.view(np.uint8).reshape(resolution, resolution, 4),
            filter_mode=wp.TextureFilterMode.CLOSEST,
            address_mode=wp.TextureAddressMode.WRAP,
            normalized_coords=True,
            dtype=wp.uint8,
            num_channels=4,
            device=self.__render_context.device,
        )

        self.__checkerboard_data.repeat = wp.vec2f(1.0, 1.0)

        self.__render_context.config.enable_textures = True
        self.__render_context.texture_data = wp.array(
            [self.__checkerboard_data], dtype=TextureData, device=self.__render_context.device
        )
        self.__render_context.shape_texture_ids = wp.array(
            texture_ids, dtype=wp.int32, device=self.__render_context.device
        )

    def __reshape_buffer_for_flatten(
        self,
        width: int,
        height: int,
        camera_count: int,
        out_buffer: wp.array | None = None,
        worlds_per_row: int | None = None,
    ) -> wp.array():
        world_and_camera_count = self.__render_context.world_count * camera_count
        if not worlds_per_row:
            worlds_per_row = math.ceil(math.sqrt(world_and_camera_count))
        worlds_per_col = math.ceil(world_and_camera_count / worlds_per_row)

        if out_buffer is None:
            return wp.empty(
                (
                    worlds_per_col * height,
                    worlds_per_row * width,
                    4,
                ),
                dtype=wp.uint8,
                device=self.__render_context.device,
            ), worlds_per_row

        return out_buffer.reshape((worlds_per_col * height, worlds_per_row * width, 4)), worlds_per_row
