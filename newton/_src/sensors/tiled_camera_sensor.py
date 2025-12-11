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
from dataclasses import dataclass

import numpy as np
import warp as wp

from ..geometry import ShapeFlags
from ..sim import Model, State
from .warp_raytrace import ClearData, GeomType, LightType, RenderContext

DEFAULT_CLEAR_DATA = ClearData(0xFF666666)


@wp.kernel(enable_backward=False)
def convert_newton_transform(
    in_body_transforms: wp.array(dtype=wp.transform),
    in_shape_body: wp.array(dtype=wp.int32),
    in_transform: wp.array(dtype=wp.transformf),
    in_scale: wp.array(dtype=wp.vec3f),
    out_transforms: wp.array(dtype=wp.transformf),
    out_sizes: wp.array(dtype=wp.vec3f),
):
    tid = wp.tid()

    body = in_shape_body[tid]
    body_transform = wp.transform_identity()
    if body >= 0:
        body_transform = in_body_transforms[body]

    out_transforms[tid] = wp.mul(body_transform, in_transform[tid])
    out_sizes[tid] = in_scale[tid]


@wp.kernel(enable_backward=False)
def compute_mesh_bounds(in_meshes: wp.array(dtype=wp.uint64), out_bounds: wp.array2d(dtype=wp.vec3f)):
    tid = wp.tid()

    min_point = wp.vec3(wp.inf)
    max_point = wp.vec3(-wp.inf)

    if in_meshes[tid] != 0:
        mesh = wp.mesh_get(in_meshes[tid])
        for i in range(mesh.points.shape[0]):
            min_point = wp.min(min_point, mesh.points[i])
            max_point = wp.max(max_point, mesh.points[i])

    out_bounds[tid, 0] = min_point
    out_bounds[tid, 1] = max_point


@wp.func
def is_supported_shape_geom_type(shape_geom_type: wp.int32) -> wp.bool:
    if shape_geom_type == GeomType.BOX:
        return True
    if shape_geom_type == GeomType.CAPSULE:
        return True
    if shape_geom_type == GeomType.CYLINDER:
        return True
    if shape_geom_type == GeomType.ELLIPSOID:
        return True
    if shape_geom_type == GeomType.PLANE:
        return True
    if shape_geom_type == GeomType.SPHERE:
        return True
    if shape_geom_type == GeomType.CONE:
        return True
    if shape_geom_type == GeomType.MESH:
        return True
    wp.printf("Unsupported shape geom type: %d\n", shape_geom_type)
    return False


@wp.kernel(enable_backward=False)
def compute_enabled_shapes(
    shape_geom_type: wp.array(dtype=wp.int32),
    shape_flags: wp.array(dtype=wp.int32),
    out_shape_enabled: wp.array(dtype=wp.uint32),
    out_mesh_indices: wp.array(dtype=wp.int32),
    out_shape_enabled_count: wp.array(dtype=wp.int32),
):
    tid = wp.tid()

    out_mesh_indices[tid] = tid

    if not bool(shape_flags[tid] & ShapeFlags.VISIBLE):
        return

    if not is_supported_shape_geom_type(shape_geom_type[tid]):
        return

    index = wp.atomic_add(out_shape_enabled_count, 0, 1)
    out_shape_enabled[index] = wp.uint32(tid)


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
    color_image: wp.array(dtype=wp.uint32, ndim=3),
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
    color = color_image[world_id, camera_id, y * width + x]

    buffer[py, px, 0] = wp.uint8((color >> wp.uint32(0)) & wp.uint32(0xFF))
    buffer[py, px, 1] = wp.uint8((color >> wp.uint32(8)) & wp.uint32(0xFF))
    buffer[py, px, 2] = wp.uint8((color >> wp.uint32(16)) & wp.uint32(0xFF))
    buffer[py, px, 3] = wp.uint8((color >> wp.uint32(24)) & wp.uint32(0xFF))


@wp.kernel(enable_backward=False)
def flatten_normal_image(
    normal_image: wp.array(dtype=wp.vec3f, ndim=3),
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
    normal = normal_image[world_id, camera_id, y * width + x] * 0.5 + wp.vec3f(0.5)

    buffer[py, px, 0] = wp.uint8(normal[0] * 255.0)
    buffer[py, px, 1] = wp.uint8(normal[1] * 255.0)
    buffer[py, px, 2] = wp.uint8(normal[2] * 255.0)
    buffer[py, px, 3] = wp.uint8(255)


@wp.kernel(enable_backward=False)
def find_depth_range(depth_image: wp.array(dtype=wp.float32, ndim=3), depth_range: wp.array(dtype=wp.float32)):
    world_id, camera_id, yx = wp.tid()
    depth = depth_image[world_id, camera_id, yx]
    if depth > 0:
        wp.atomic_min(depth_range, 0, depth)
        wp.atomic_max(depth_range, 1, depth)


@wp.kernel(enable_backward=False)
def flatten_depth_image(
    depth_image: wp.array(dtype=wp.float32, ndim=3),
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
    depth = depth_image[world_id, camera_id, y * width + x]
    if depth > 0:
        denom = wp.max(depth_range[1] - depth_range[0], 1e-6)
        value = wp.uint8(255.0 - ((depth - depth_range[0]) / denom) * 205.0)

    buffer[py, px, 0] = value
    buffer[py, px, 1] = value
    buffer[py, px, 2] = value
    buffer[py, px, 3] = value


class TiledCameraSensor:
    """
    A Warp-based tiled camera sensor for raytraced rendering across multiple worlds.

    Renders color and depth images for multiple cameras and worlds, organizing the
    output as tiles in a grid layout.

    Args:
        model: The Newton Model containing shapes to render.
        num_cameras: Number of cameras per world.
        width: Image width in pixels for each camera.
        height: Image height in pixels for each camera.
    """

    RenderContext = RenderContext
    LightType = LightType
    GeomType = GeomType

    @dataclass
    class Options:
        checkerboard_texture: bool = False
        default_light: bool = False
        default_light_shadows: bool = False
        colors_per_world: bool = False
        colors_per_shape: bool = False

    def __init__(self, model: Model, num_cameras: int, width: int, height: int, options: Options | None = None):
        self.model = model

        self.render_context = RenderContext(
            width, height, False, False, True, True, self.model.num_worlds, num_cameras, True
        )
        self.render_context.mesh_ids = model.shape_source_ptr
        self.render_context.shape_mesh_indices = wp.empty(self.model.shape_count, dtype=wp.int32)
        self.render_context.mesh_bounds = wp.empty((self.model.shape_count, 2), dtype=wp.vec3f, ndim=2)

        if model.particle_q is not None and model.particle_q.shape[0]:
            self.render_context.particles_position = model.particle_q
            self.render_context.particles_radius = model.particle_radius
            self.render_context.particles_world_index = model.particle_world
            if model.tri_indices is not None and model.tri_indices.shape[0]:
                self.render_context.triangle_points = model.particle_q
                self.render_context.triangle_indices = model.tri_indices.flatten()
                self.render_context.enable_particles = False

        self.render_context.shape_enabled = wp.empty(self.model.shape_count, dtype=wp.uint32)
        self.render_context.shape_geom_types = model.shape_type
        self.render_context.shape_sizes = wp.empty(self.model.shape_count, dtype=wp.vec3f)
        self.render_context.shape_transforms = wp.empty(self.model.shape_count, dtype=wp.transformf)
        self.render_context.shape_materials = wp.array(
            np.full(self.model.shape_count, fill_value=-1, dtype=np.int32), dtype=wp.int32
        )
        self.render_context.shape_colors = wp.array(
            np.full((self.model.shape_count, 4), fill_value=1.0, dtype=wp.float32), dtype=wp.vec4f
        )
        self.render_context.shape_world_index = self.model.shape_world

        num_enabled_shapes = wp.zeros(1, dtype=wp.int32)
        wp.launch(
            kernel=compute_enabled_shapes,
            dim=self.model.shape_count,
            inputs=[
                model.shape_type,
                model.shape_flags,
                self.render_context.shape_enabled,
                self.render_context.shape_mesh_indices,
                num_enabled_shapes,
            ],
        )
        self.render_context.num_shapes = int(num_enabled_shapes.numpy()[0])

        wp.launch(
            kernel=compute_mesh_bounds,
            dim=self.model.shape_count,
            inputs=[self.render_context.mesh_ids, self.render_context.mesh_bounds],
        )

        if options is not None:
            if options.checkerboard_texture:
                self.assign_checkerboard_material_to_all_shapes()
            if options.default_light:
                self.create_default_light(options.default_light_shadows)
            if options.colors_per_world:
                self.assign_random_colors_per_world()
            elif options.colors_per_shape:
                self.assign_random_colors_per_shape()

    def update_from_state(self, state: State):
        """
        Update data from Newton State.

        Args:
            state: The current simulation state containing body transforms.
        """
        if self.render_context.has_shapes:
            wp.launch(
                kernel=convert_newton_transform,
                dim=self.model.shape_count,
                inputs=[
                    state.body_q,
                    self.model.shape_body,
                    self.model.shape_transform,
                    self.model.shape_scale,
                    self.render_context.shape_transforms,
                    self.render_context.shape_sizes,
                ],
            )

        if self.render_context.has_triangle_mesh:
            self.render_context.triangle_points = state.particle_q

        if self.render_context.has_particles:
            self.render_context.particles_position = state.particle_q

    def render(
        self,
        state: State | None,
        camera_transforms: wp.array(dtype=wp.transformf, ndim=2),
        camera_rays: wp.array(dtype=wp.vec3f, ndim=4),
        color_image: wp.array(dtype=wp.uint32, ndim=3) | None = None,
        depth_image: wp.array(dtype=wp.float32, ndim=3) | None = None,
        shape_index_image: wp.array(dtype=wp.uint32, ndim=3) | None = None,
        normal_image: wp.array(dtype=wp.vec3f, ndim=3) | None = None,
        refit_bvh: bool = True,
        clear_data: ClearData | None = DEFAULT_CLEAR_DATA,
    ):
        """
        Render color and depth images for all worlds and cameras.

        Args:
            state: The current simulation state containing body transforms.
            camera_transforms: Array of camera transforms in world space, shape (num_cameras, num_worlds).
            camera_rays: Array of camera rays in camera space, shape (num_cameras, height, width, 2).
            color_image: Optional output array for color data (num_worlds, num_cameras, width*height).
                        If None, no color rendering is performed.
            depth_image: Optional output array for depth data (num_worlds, num_cameras, width*height).
                        If None, no depth rendering is performed.
            shape_index_image: Optional output array for shape index data (num_worlds, num_cameras, width*height).
                        If None, no shape index rendering is performed.
            normal_image: Optional output array for normal data (num_worlds, num_cameras, width*height).
                        If None, no normal rendering is performed.
            refit_bvh: Whether to refit the BVH or not.
            clear_data: The data to clear the image buffers with (or skip if None).
        """
        if state is not None:
            self.update_from_state(state)

        self.render_context.render(
            camera_transforms,
            camera_rays,
            color_image,
            depth_image,
            shape_index_image,
            normal_image,
            refit_bvh=refit_bvh,
            clear_data=clear_data,
        )

    def compute_pinhole_camera_rays(
        self, camera_fovs: float | list[float] | np.ndarray | wp.array(dtype=wp.float32)
    ) -> wp.array(dtype=wp.vec3f, ndim=4):
        """
        Compute camera-space ray directions for pinhole cameras.

        Generates rays in camera space (origin at [0,0,0], direction normalized) for each
        pixel in each camera based on the specified field-of-view angles.

        Args:
            camera_fovs: Array of vertical FOV angles in radians, shape (num_cameras,).

        Returns:
            camera_rays: Array of camera rays in camera space, shape (num_cameras, height, width, 2).
        """

        camera_rays = wp.empty(
            (self.render_context.num_cameras, self.render_context.height, self.render_context.width, 2), dtype=wp.vec3f
        )

        if isinstance(camera_fovs, float):
            camera_fovs = wp.array([camera_fovs] * self.render_context.num_cameras, dtype=wp.float32)
        elif isinstance(camera_fovs, list):
            assert len(camera_fovs) == self.render_context.num_cameras, (
                "Length of camera_fovs does not match the number of cameras"
            )
            camera_fovs = wp.array(camera_fovs, dtype=wp.float32)
        elif isinstance(camera_fovs, np.ndarray):
            assert camera_fovs.size == self.render_context.num_cameras, (
                "Length of camera_fovs does not match the number of cameras"
            )
            camera_fovs = wp.array(camera_fovs, dtype=wp.float32)

        wp.launch(
            kernel=compute_pinhole_camera_rays,
            dim=(self.render_context.num_cameras, self.render_context.height, self.render_context.width),
            inputs=[
                self.render_context.width,
                self.render_context.height,
                camera_fovs,
                camera_rays,
            ],
        )

        return camera_rays

    def __reshape_buffer_for_flatten(self, out_buffer: wp.array | None = None, num_worlds_per_row: int | None = None):
        num_worlds_and_cameras = self.render_context.num_worlds * self.render_context.num_cameras
        if not num_worlds_per_row:
            num_worlds_per_row = math.ceil(math.sqrt(num_worlds_and_cameras))
        num_worlds_per_col = math.ceil(num_worlds_and_cameras / num_worlds_per_row)

        if out_buffer is None:
            return wp.empty(
                (num_worlds_per_col * self.render_context.height, num_worlds_per_row * self.render_context.width, 4),
                dtype=wp.uint8,
            ), num_worlds_per_row

        return out_buffer.reshape(
            (num_worlds_per_col * self.render_context.height, num_worlds_per_row * self.render_context.width, 4)
        ), num_worlds_per_row

    def flatten_color_image_to_rgba(
        self,
        image: wp.array(dtype=wp.uint32, ndim=3),
        out_buffer: wp.array(dtype=wp.uint8, ndim=3) | None = None,
        num_worlds_per_row: int | None = None,
    ):
        """
        Flatten rendered color image to a tiled image buffer.

        Arranges (num_worlds x num_cameras) tiles in a grid layout. Each tile
        shows one camera's view of one world.

        Args:
            image: Color output array from render(), shape (num_worlds, num_cameras, width*height).
            out_buffer: Optional output array
            num_worlds_per_row: Optional number of rows
        """

        out_buffer, num_worlds_per_row = self.__reshape_buffer_for_flatten(out_buffer, num_worlds_per_row)

        wp.launch(
            flatten_color_image,
            (
                self.render_context.num_worlds,
                self.render_context.num_cameras,
                self.render_context.height,
                self.render_context.width,
            ),
            [
                image,
                out_buffer,
                self.render_context.width,
                self.render_context.height,
                self.render_context.num_cameras,
                num_worlds_per_row,
            ],
        )
        return out_buffer

    def flatten_normal_image_to_rgba(
        self,
        image: wp.array(dtype=wp.vec3f, ndim=3),
        out_buffer: wp.array(dtype=wp.uint8, ndim=3) | None = None,
        num_worlds_per_row: int | None = None,
    ):
        """
        Flatten rendered normal image to a tiled image buffer.

        Arranges (num_worlds x num_cameras) tiles in a grid layout. Each tile
        shows one camera's view of one world.

        Args:
            image: Normal output array from render(), shape (num_worlds, num_cameras, width*height).
            out_buffer: Optional output array
            num_worlds_per_row: Optional number of rows
        """

        out_buffer, num_worlds_per_row = self.__reshape_buffer_for_flatten(out_buffer, num_worlds_per_row)

        wp.launch(
            flatten_normal_image,
            (
                self.render_context.num_worlds,
                self.render_context.num_cameras,
                self.render_context.height,
                self.render_context.width,
            ),
            [
                image,
                out_buffer,
                self.render_context.width,
                self.render_context.height,
                self.render_context.num_cameras,
                num_worlds_per_row,
            ],
        )
        return out_buffer

    def flatten_depth_image_to_rgba(
        self,
        image: wp.array(dtype=wp.float32, ndim=3),
        out_buffer: wp.array(dtype=wp.uint8, ndim=3) | None = None,
        num_worlds_per_row: int | None = None,
    ):
        """
        Flatten rendered depth image to a tiled grayscale image buffer.

        Arranges (num_worlds x num_cameras) tiles in a grid. Depth values are
        inverted (closer = brighter) and normalized to [50, 255] range. Background (depth < 0
        or no hit) remains black.

        Args:
            image: Depth output array from render(), shape (num_worlds, num_cameras, width*height).
            out_buffer: Optional output array
            num_worlds_per_row: Optional number of rows
        """

        out_buffer, num_worlds_per_row = self.__reshape_buffer_for_flatten(out_buffer, num_worlds_per_row)

        depth_range = wp.array([100000000.0, 0.0], dtype=wp.float32)
        wp.launch(find_depth_range, image.shape, [image, depth_range])
        wp.launch(
            flatten_depth_image,
            (
                self.render_context.num_worlds,
                self.render_context.num_cameras,
                self.render_context.height,
                self.render_context.width,
            ),
            [
                image,
                out_buffer,
                depth_range,
                self.render_context.width,
                self.render_context.height,
                self.render_context.num_cameras,
                num_worlds_per_row,
            ],
        )
        return out_buffer

    def assign_random_colors_per_world(self, seed: int = 100):
        """
        Assign a random color to all shapes, per world.

        Args:
            seed: The seed to use for the randomizer.
        """

        colors = np.random.default_rng(seed).random((self.model.shape_count, 4)) * 0.5 + 0.5
        colors[:, -1] = 1.0
        self.render_context.shape_colors = wp.array(
            colors[self.model.shape_world.numpy() % len(colors)], dtype=wp.vec4f
        )

    def assign_random_colors_per_shape(self, seed: int = 100):
        """
        Assign a random color to all shapes.

        Args:
            seed: The seed to use for the randomizer.
        """

        colors = np.random.default_rng(seed).random((self.model.shape_count, 4)) * 0.5 + 0.5
        colors[:, -1] = 1.0
        self.render_context.shape_colors = wp.array(colors, dtype=wp.vec4f)

    def create_default_light(self, enable_shadows: bool = True):
        """
        Create a default directional light for the scene.

        Sets up a single directional light oriented at (-1, 1, -1) with shadow casting enabled.
        """

        self.render_context.enable_shadows = enable_shadows
        self.render_context.lights_active = wp.array([True], dtype=wp.bool)
        self.render_context.lights_type = wp.array([LightType.DIRECTIONAL], dtype=wp.int32)
        self.render_context.lights_cast_shadow = wp.array([True], dtype=wp.bool)
        self.render_context.lights_position = wp.array([wp.vec3f(0.0)], dtype=wp.vec3f)
        self.render_context.lights_orientation = wp.array(
            [wp.vec3f(-0.57735026, 0.57735026, -0.57735026)], dtype=wp.vec3f
        )

    def assign_checkerboard_material_to_all_shapes(self, resolution: int = 64, checker_size: int = 32):
        """
        Assign a checkerboard texture material to all shapes.

        Creates a gray checkerboard pattern texture and applies it to all shapes
        in the scene.

        Args:
            resolution: Texture resolution in pixels (square texture).
            checker_size: Size of each checkerboard square in pixels.
        """

        checkerboard = (
            (np.arange(resolution) // checker_size)[:, None] + (np.arange(resolution) // checker_size)
        ) % 2 == 0
        pixels = np.where(checkerboard, 0xFF808080, 0xFFBFBFBF).astype(np.uint32).flatten()

        self.render_context.enable_textures = True
        self.render_context.texture_data = wp.array(pixels, dtype=wp.uint32)
        self.render_context.texture_offsets = wp.array([0], dtype=wp.int32)
        self.render_context.texture_width = wp.array([resolution], dtype=wp.int32)
        self.render_context.texture_height = wp.array([resolution], dtype=wp.int32)

        self.render_context.material_texture_ids = wp.array([0], dtype=wp.int32)
        self.render_context.material_texture_repeat = wp.array([wp.vec2f(1.0)], dtype=wp.vec2f)
        self.render_context.material_rgba = wp.array([wp.vec4f(1.0)], dtype=wp.vec4f)

        self.render_context.shape_materials = wp.array(
            np.full(self.model.shape_count, fill_value=0, dtype=np.int32), dtype=wp.int32
        )

    def create_color_image_output(self):
        """
        Create a Warp array for color image output.

        Returns:
            wp.array of shape (num_worlds, num_cameras, width*height) with dtype uint32.
        """
        return self.render_context.create_color_image_output()

    def create_depth_image_output(self):
        """
        Create a Warp array for depth image output.

        Returns:
            wp.array of shape (num_worlds, num_cameras, width*height) with dtype float32.
        """
        return self.render_context.create_depth_image_output()

    def create_shape_index_image_output(self):
        """
        Create a Warp array for shape index image output.

        Returns:
            wp.array of shape (num_worlds, num_cameras, width*height) with dtype uint32.
        """
        return self.render_context.create_shape_index_image_output()

    def create_normal_image_output(self):
        """
        Create a Warp array for normal image output.

        Returns:
            wp.array of shape (num_worlds, num_cameras, width*height) with dtype vec3f.
        """
        return self.render_context.create_normal_image_output()
