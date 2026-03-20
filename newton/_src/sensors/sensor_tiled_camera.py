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

from dataclasses import dataclass
from typing import Any

import numpy as np
import warp as wp

from ..geometry import GeoType, Mesh, ShapeFlags
from ..sim import Model, State
from ..utils import load_texture, normalize_texture
from .warp_raytrace import (
    GaussianRenderMode,
    MeshData,
    RenderContext,
    RenderLightType,
    RenderOrder,
    TextureData,
)


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


@wp.func
def is_supported_shape_type(shape_type: wp.int32) -> wp.bool:
    if shape_type == GeoType.BOX:
        return True
    if shape_type == GeoType.CAPSULE:
        return True
    if shape_type == GeoType.CYLINDER:
        return True
    if shape_type == GeoType.ELLIPSOID:
        return True
    if shape_type == GeoType.PLANE:
        return True
    if shape_type == GeoType.SPHERE:
        return True
    if shape_type == GeoType.CONE:
        return True
    if shape_type == GeoType.MESH:
        return True
    if shape_type == GeoType.GAUSSIAN:
        return True
    wp.printf("Unsupported shape geom type: %d\n", shape_type)
    return False


@wp.kernel(enable_backward=False)
def compute_enabled_shapes(
    shape_type: wp.array(dtype=wp.int32),
    shape_flags: wp.array(dtype=wp.int32),
    out_shape_enabled: wp.array(dtype=wp.uint32),
    out_shape_enabled_count: wp.array(dtype=wp.int32),
):
    tid = wp.tid()

    if not bool(shape_flags[tid] & ShapeFlags.VISIBLE):
        return

    if not is_supported_shape_type(shape_type[tid]):
        return

    index = wp.atomic_add(out_shape_enabled_count, 0, 1)
    out_shape_enabled[index] = wp.uint32(tid)


class SensorTiledCamera:
    """Warp-based tiled camera sensor for raytraced rendering across multiple worlds.

    Renders up to five image channels per (world, camera) pair:

    - **color** -- RGBA shaded image (``uint32``).
    - **depth** -- ray-hit distance [m] (``float32``); negative means no hit.
    - **normal** -- surface normal at hit point (``vec3f``).
    - **albedo** -- unshaded surface color (``uint32``).
    - **shape_index** -- shape id per pixel (``uint32``).

    All output arrays have shape ``(world_count, camera_count, height, width)``. Use the ``flatten_*`` helpers to
    rearrange them into tiled RGBA buffers for display, with one tile per (world, camera) pair laid out in a grid.

    Shapes without the ``VISIBLE`` flag are excluded.

    Example:
        ::

            sensor = SensorTiledCamera(model)
            rays = sensor.compute_pinhole_camera_rays(width, height, fov)
            color = sensor.create_color_image_output(width, height)

            # each step
            sensor.update(state, camera_transforms, rays, color_image=color)

    See :class:`Config` for optional rendering settings and :attr:`ClearData` / :attr:`DEFAULT_CLEAR_DATA` /
    :attr:`GRAY_CLEAR_DATA` for image-clear presets.
    """

    RenderContext = RenderContext
    RenderLightType = RenderLightType
    RenderOrder = RenderOrder
    GaussianRenderMode = GaussianRenderMode
    ClearData = RenderContext.ClearData

    DEFAULT_CLEAR_DATA = ClearData()
    GRAY_CLEAR_DATA = ClearData(clear_color=0xFF666666, clear_albedo=0xFF000000)

    @dataclass
    class Config:
        """Rendering configuration."""

        checkerboard_texture: bool = False
        """Apply a checkerboard texture to all shapes."""

        default_light: bool = False
        """Add a default directional light to the scene."""

        default_light_shadows: bool = False
        """Enable shadows for the default light (requires ``default_light``)."""

        enable_ambient_lighting: bool = True
        """Enable ambient lighting for the scene."""

        colors_per_world: bool = False
        """Assign a random color palette per world."""

        colors_per_shape: bool = False
        """Assign a random color per shape (ignored when ``colors_per_world`` is True)."""

        backface_culling: bool = True
        """Cull back-facing triangles."""

        enable_textures: bool = False
        """Enable texturing."""

        enable_particles: bool = True
        """Enable particle rendering."""

    def __init__(self, model: Model, *, config: Config | None = None):
        self.model = model

        if config is None:
            config = SensorTiledCamera.Config()

        self.render_context = RenderContext(
            world_count=self.model.world_count,
            config=RenderContext.Config(
                enable_global_world=True,
                enable_shadows=False,
                enable_textures=config.enable_textures,
                enable_ambient_lighting=config.enable_ambient_lighting,
                enable_particles=config.enable_particles,
                enable_backface_culling=config.backface_culling,
            ),
            device=self.model.device,
        )
        self.render_context.shape_source_ptr = model.shape_source_ptr
        self.render_context.shape_bounds = wp.empty(
            (self.model.shape_count, 2), dtype=wp.vec3f, ndim=2, device=self.render_context.device
        )

        if model.particle_q is not None and model.particle_q.shape[0]:
            self.render_context.particles_position = model.particle_q
            self.render_context.particles_radius = model.particle_radius
            self.render_context.particles_world_index = model.particle_world
            if model.tri_indices is not None and model.tri_indices.shape[0]:
                self.render_context.triangle_points = model.particle_q
                self.render_context.triangle_indices = model.tri_indices.flatten()
                self.render_context.config.enable_particles = False

        self.render_context.shape_enabled = wp.empty(
            self.model.shape_count, dtype=wp.uint32, device=self.render_context.device
        )
        self.render_context.shape_types = model.shape_type
        self.render_context.shape_sizes = wp.empty(
            self.model.shape_count, dtype=wp.vec3f, device=self.render_context.device
        )
        self.render_context.shape_transforms = wp.empty(
            self.model.shape_count, dtype=wp.transformf, device=self.render_context.device
        )

        self.render_context.shape_world_index = self.model.shape_world
        self.render_context.gaussians_data = self.model.gaussians_data

        self.__load_texture_and_mesh_data(config)

        colors = [(*self.__get_shape_color(i, shape), 1.0) for i, shape in enumerate(self.model.shape_source)]
        self.render_context.shape_colors = wp.array(colors, dtype=wp.vec4f, device=self.render_context.device)

        num_enabled_shapes = wp.zeros(1, dtype=wp.int32, device=self.render_context.device)
        wp.launch(
            kernel=compute_enabled_shapes,
            dim=self.model.shape_count,
            inputs=[
                model.shape_type,
                model.shape_flags,
                self.render_context.shape_enabled,
                num_enabled_shapes,
            ],
            device=self.render_context.device,
        )
        self.render_context.shape_count_total = self.model.shape_count
        self.render_context.shape_count_enabled = int(num_enabled_shapes.numpy()[0])

        self.render_context.utils.compute_shape_bounds()

        if config.checkerboard_texture:
            self.assign_checkerboard_material_to_all_shapes()
        if config.default_light:
            self.create_default_light(config.default_light_shadows)
        if config.colors_per_world:
            self.assign_random_colors_per_world()
        elif config.colors_per_shape:
            self.assign_random_colors_per_shape()

    def sync_transforms(self, state: State):
        """Synchronize shape transforms from the simulation state.

        :meth:`update` calls this automatically when *state* is not None.

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
                device=self.render_context.device,
            )

        if self.render_context.has_triangle_mesh:
            self.render_context.triangle_points = state.particle_q

        if self.render_context.has_particles:
            self.render_context.particles_position = state.particle_q

    def update(
        self,
        state: State | None,
        camera_transforms: wp.array(dtype=wp.transformf, ndim=2),
        camera_rays: wp.array(dtype=wp.vec3f, ndim=4),
        *,
        color_image: wp.array(dtype=wp.uint32, ndim=4) | None = None,
        depth_image: wp.array(dtype=wp.float32, ndim=4) | None = None,
        shape_index_image: wp.array(dtype=wp.uint32, ndim=4) | None = None,
        normal_image: wp.array(dtype=wp.vec3f, ndim=4) | None = None,
        albedo_image: wp.array(dtype=wp.uint32, ndim=4) | None = None,
        refit_bvh: bool = True,
        clear_data: SensorTiledCamera.ClearData | None = DEFAULT_CLEAR_DATA,
    ):
        """Render output images for all worlds and cameras.

        Each output array has shape ``(world_count, camera_count, height, width)`` where element
        ``[world_id, camera_id, y, x]`` corresponds to the ray in ``camera_rays[camera_id, y, x]``. Each output
        channel is optional -- pass None to skip that channel's rendering entirely.

        Args:
            state: Simulation state with body transforms. If not None, calls :meth:`sync_transforms` first.
            camera_transforms: Camera-to-world transforms, shape ``(camera_count, world_count)``.
            camera_rays: Camera-space rays from :meth:`compute_pinhole_camera_rays`, shape
                ``(camera_count, height, width, 2)``.
            color_image: Output for RGBA color. None to skip.
            depth_image: Output for ray-hit distance [m]. None to skip.
            shape_index_image: Output for per-pixel shape id. None to skip.
            normal_image: Output for surface normals. None to skip.
            albedo_image: Output for unshaded surface color. None to skip.
            refit_bvh: Refit the BVH before rendering.
            clear_data: Values to clear output buffers with.
                See :attr:`DEFAULT_CLEAR_DATA`, :attr:`GRAY_CLEAR_DATA`.
        """
        if state is not None:
            self.sync_transforms(state)

        self.render_context.render(
            camera_transforms,
            camera_rays,
            color_image,
            depth_image,
            shape_index_image,
            normal_image,
            albedo_image,
            refit_bvh=refit_bvh,
            clear_data=clear_data,
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
            camera_fovs = wp.array([camera_fovs], dtype=wp.float32, device=self.render_context.device)
        elif isinstance(camera_fovs, list):
            camera_fovs = wp.array(camera_fovs, dtype=wp.float32, device=self.render_context.device)
        elif isinstance(camera_fovs, np.ndarray):
            camera_fovs = wp.array(camera_fovs, dtype=wp.float32, device=self.render_context.device)
        return self.render_context.utils.compute_pinhole_camera_rays(width, height, camera_fovs)

    def flatten_color_image_to_rgba(
        self,
        image: wp.array(dtype=wp.uint32, ndim=4),
        out_buffer: wp.array(dtype=wp.uint8, ndim=3) | None = None,
        worlds_per_row: int | None = None,
    ):
        """Flatten rendered color image to a tiled RGBA buffer.

        Arranges ``(world_count * camera_count)`` tiles in a grid. Each tile shows one camera's view of one world.

        Args:
            image: Color output from :meth:`update`, shape ``(world_count, camera_count, height, width)``.
            out_buffer: Pre-allocated RGBA buffer. If None, allocates a new one.
            worlds_per_row: Tiles per row in the grid. If None, picks a square-ish layout.
        """
        return self.render_context.utils.flatten_color_image_to_rgba(image, out_buffer, worlds_per_row)

    def flatten_normal_image_to_rgba(
        self,
        image: wp.array(dtype=wp.vec3f, ndim=4),
        out_buffer: wp.array(dtype=wp.uint8, ndim=3) | None = None,
        worlds_per_row: int | None = None,
    ):
        """Flatten rendered normal image to a tiled RGBA buffer.

        Arranges ``(world_count * camera_count)`` tiles in a grid. Each tile shows one camera's view of one world.

        Args:
            image: Normal output from :meth:`update`, shape ``(world_count, camera_count, height, width)``.
            out_buffer: Pre-allocated RGBA buffer. If None, allocates a new one.
            worlds_per_row: Tiles per row in the grid. If None, picks a square-ish layout.
        """
        return self.render_context.utils.flatten_normal_image_to_rgba(image, out_buffer, worlds_per_row)

    def flatten_depth_image_to_rgba(
        self,
        image: wp.array(dtype=wp.float32, ndim=4),
        out_buffer: wp.array(dtype=wp.uint8, ndim=3) | None = None,
        worlds_per_row: int | None = None,
        depth_range: wp.array(dtype=wp.float32) | None = None,
    ):
        """Flatten rendered depth image to a tiled RGBA buffer.

        Encodes depth as grayscale: inverts values (closer = brighter) and normalizes to the ``[50, 255]``
        range. Background pixels (no hit) remain black.

        Args:
            image: Depth output from :meth:`update`, shape ``(world_count, camera_count, height, width)``.
            out_buffer: Pre-allocated RGBA buffer. If None, allocates a new one.
            worlds_per_row: Tiles per row in the grid. If None, picks a square-ish layout.
            depth_range: Depth range to normalize to, shape ``(2,)`` ``[near, far]``. If None, computes from *image*.
        """
        return self.render_context.utils.flatten_depth_image_to_rgba(image, out_buffer, worlds_per_row, depth_range)

    def assign_random_colors_per_world(self, seed: int = 100):
        """Assign each world a random color, applied to all its shapes.

        Args:
            seed: Random seed.
        """
        self.render_context.utils.assign_random_colors_per_world(seed)

    def assign_random_colors_per_shape(self, seed: int = 100):
        """Assign a random color to each shape.

        Args:
            seed: Random seed.
        """
        self.render_context.utils.assign_random_colors_per_shape(seed)

    def create_default_light(self, enable_shadows: bool = True):
        """Create a default directional light oriented at ``(-1, 1, -1)``.

        Args:
            enable_shadows: Enable shadow casting for this light.
        """
        self.render_context.utils.create_default_light(enable_shadows)

    def assign_checkerboard_material_to_all_shapes(self, resolution: int = 64, checker_size: int = 32):
        """Assign a gray checkerboard texture material to all shapes.
        Creates a gray checkerboard pattern texture and applies it to all shapes
        in the scene.

        Args:
            resolution: Texture resolution in pixels (square texture).
            checker_size: Size of each checkerboard square in pixels.
        """
        self.render_context.utils.assign_checkerboard_material_to_all_shapes(resolution, checker_size)

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
        return self.render_context.create_color_image_output(width, height, camera_count)

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
        return self.render_context.create_depth_image_output(width, height, camera_count)

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
        return self.render_context.create_shape_index_image_output(width, height, camera_count)

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
        return self.render_context.create_normal_image_output(width, height, camera_count)

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
        return self.render_context.create_albedo_image_output(width, height, camera_count)

    def __get_shape_color(self, index: int, shape: Any):
        SHAPE_COLOR_MAP = [
            (68 / 255.0, 119 / 255.0, 170 / 255.0),  # blue
            (102 / 255.0, 204 / 255.0, 238 / 255.0),  # cyan
            (34 / 255.0, 136 / 255.0, 51 / 255.0),  # green
            (204 / 255.0, 187 / 255.0, 68 / 255.0),  # yellow
            (238 / 255.0, 102 / 255.0, 119 / 255.0),  # red
            (170 / 255.0, 51 / 255.0, 119 / 255.0),  # magenta
            (187 / 255.0, 187 / 255.0, 187 / 255.0),  # grey
            (238 / 255.0, 153 / 255.0, 51 / 255.0),  # orange
            (0 / 255.0, 153 / 255.0, 136 / 255.0),  # teal
        ]

        if color := getattr(shape, "color", None):
            return color
        return SHAPE_COLOR_MAP[index % len(SHAPE_COLOR_MAP)]

    def __load_texture_and_mesh_data(self, config: Config):
        """Load textures and mesh data into the render context.

        Deduplicates textures by hash and meshes by identity, storing each
        unique texture as a :class:`TextureData` struct and each unique Mesh
        as a :class:`MeshData` struct.  Per-shape index arrays map each
        shape to its texture and mesh data entry (``-1`` when absent).

        Args:
            config: Sensor configuration controlling whether textures are enabled.
        """
        self.__mesh_data = []
        self.__texture_data = []

        texture_hashes = {}
        mesh_hashes = {}

        mesh_data_ids = []
        texture_data_ids = []

        for shape in self.model.shape_source:
            if isinstance(shape, Mesh):
                if shape.texture is not None and config.enable_textures and not config.checkerboard_texture:
                    if shape.texture_hash not in texture_hashes:
                        pixels = load_texture(shape.texture)
                        if pixels is None:
                            raise ValueError(f"Failed to load texture: {shape.texture}")

                        # Normalize texture to ensure a consistent channel layout and dtype
                        pixels = normalize_texture(pixels, require_channels=True)
                        if pixels.dtype != np.uint8:
                            pixels = pixels.astype(np.uint8, copy=False)

                        texture_hashes[shape.texture_hash] = len(self.__texture_data)

                        data = TextureData()
                        data.texture = wp.Texture2D(
                            pixels,
                            filter_mode=wp.TextureFilterMode.LINEAR,
                            address_mode=wp.TextureAddressMode.WRAP,
                            normalized_coords=True,
                            dtype=wp.uint8,
                            num_channels=4,
                            device=self.render_context.device,
                        )
                        data.repeat = wp.vec2f(1.0, 1.0)
                        self.__texture_data.append(data)

                    texture_data_ids.append(texture_hashes[shape.texture_hash])
                else:
                    texture_data_ids.append(-1)

                if shape.uvs is not None or shape.normals is not None:
                    if shape not in mesh_hashes:
                        mesh_hashes[shape] = len(self.__mesh_data)

                        data = MeshData()
                        if shape.uvs is not None:
                            data.uvs = wp.array(shape.uvs, dtype=wp.vec2f, device=self.render_context.device)
                        if shape.normals is not None:
                            data.normals = wp.array(shape.normals, dtype=wp.vec3f, device=self.render_context.device)
                        self.__mesh_data.append(data)

                    mesh_data_ids.append(mesh_hashes[shape])
                else:
                    mesh_data_ids.append(-1)
            else:
                texture_data_ids.append(-1)
                mesh_data_ids.append(-1)

        self.render_context.texture_data = wp.array(
            self.__texture_data, dtype=TextureData, device=self.render_context.device
        )
        self.render_context.shape_texture_ids = wp.array(
            texture_data_ids, dtype=wp.int32, device=self.render_context.device
        )

        self.render_context.mesh_data = wp.array(self.__mesh_data, dtype=MeshData, device=self.render_context.device)
        self.render_context.shape_mesh_data_ids = wp.array(
            mesh_data_ids, dtype=wp.int32, device=self.render_context.device
        )
