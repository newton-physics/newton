# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np
import warp as wp

from ..sim import Model, State
from .warp_raytrace import (
    GaussianRenderMode,
    RenderContext,
    RenderLightType,
    RenderOrder,
)


class DeprecatedFields(type):
    @property
    def RenderContext(cls) -> RenderContext:
        warnings.warn(
            "Access to SensorTiledCamera.RenderContext is deprecated.",
            DeprecationWarning,
            stacklevel=2,
        )
        return RenderContext


class SensorTiledCamera(metaclass=DeprecatedFields):
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

    RenderLightType = RenderLightType
    RenderOrder = RenderOrder
    GaussianRenderMode = GaussianRenderMode
    RenderConfig = RenderContext.Config
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

        enable_ambient_lighting: bool | None = None
        """Deprecated: use ``render_config.enable_ambient_lighting`` instead."""

        colors_per_world: bool = False
        """Assign a random color palette per world."""

        colors_per_shape: bool = False
        """Assign a random color per shape (ignored when ``colors_per_world`` is True)."""

        backface_culling: bool | None = None
        """Deprecated: use ``render_config.enable_backface_culling`` instead."""

        enable_textures: bool | None = None
        """Deprecated: use ``render_config.enable_textures`` instead."""

        enable_particles: bool | None = None
        """Deprecated: use ``render_config.enable_particles`` instead."""

        render_config: SensorTiledCamera.RenderConfig = field(default_factory=RenderContext.Config)
        """Low-level raytrace settings forwarded to the internal :class:`RenderContext`."""

    def __init__(self, model: Model, *, config: Config | None = None):
        self.model = model

        if config is None:
            config = SensorTiledCamera.Config()

        if config.enable_ambient_lighting is not None:
            warnings.warn(
                "SensorTiledCamera.Config.enable_ambient_lighting is deprecated, use SensorTiledCamera.Config.render_config.enable_ambient_lighting instead.",
                category=DeprecationWarning,
                stacklevel=2,
            )
            config.render_config.enable_ambient_lighting = config.enable_ambient_lighting
        if config.backface_culling is not None:
            warnings.warn(
                "SensorTiledCamera.Config.backface_culling is deprecated, use SensorTiledCamera.Config.render_config.enable_backface_culling instead.",
                category=DeprecationWarning,
                stacklevel=2,
            )
            config.render_config.enable_backface_culling = config.backface_culling
        if config.enable_textures is not None:
            warnings.warn(
                "SensorTiledCamera.Config.enable_textures is deprecated, use SensorTiledCamera.Config.render_config.enable_textures instead.",
                category=DeprecationWarning,
                stacklevel=2,
            )
            config.render_config.enable_textures = config.enable_textures
        if config.enable_particles is not None:
            warnings.warn(
                "SensorTiledCamera.Config.enable_particles is deprecated, use SensorTiledCamera.Config.render_config.enable_particles instead.",
                category=DeprecationWarning,
                stacklevel=2,
            )
            config.render_config.enable_particles = config.enable_particles

        self.__render_context = RenderContext(
            world_count=self.model.world_count,
            config=config.render_config,
            device=self.model.device,
        )

        self.__render_context.init_from_model(self.model, not config.checkerboard_texture)

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
        self.__render_context.update(self.model, state)

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

        self.__render_context.render(
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
            camera_fovs = wp.array([camera_fovs], dtype=wp.float32, device=self.__render_context.device)
        elif isinstance(camera_fovs, list):
            camera_fovs = wp.array(camera_fovs, dtype=wp.float32, device=self.__render_context.device)
        elif isinstance(camera_fovs, np.ndarray):
            camera_fovs = wp.array(camera_fovs, dtype=wp.float32, device=self.__render_context.device)
        return self.__render_context.utils.compute_pinhole_camera_rays(width, height, camera_fovs)

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
        return self.__render_context.utils.flatten_color_image_to_rgba(image, out_buffer, worlds_per_row)

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
        return self.__render_context.utils.flatten_normal_image_to_rgba(image, out_buffer, worlds_per_row)

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
        return self.__render_context.utils.flatten_depth_image_to_rgba(image, out_buffer, worlds_per_row, depth_range)

    def assign_random_colors_per_world(self, seed: int = 100):
        """Assign each world a random color, applied to all its shapes.

        Args:
            seed: Random seed.
        """
        self.__render_context.utils.assign_random_colors_per_world(seed)

    def assign_random_colors_per_shape(self, seed: int = 100):
        """Assign a random color to each shape.

        Args:
            seed: Random seed.
        """
        self.__render_context.utils.assign_random_colors_per_shape(seed)

    def create_default_light(self, enable_shadows: bool = True):
        """Create a default directional light oriented at ``(-1, 1, -1)``.

        Args:
            enable_shadows: Enable shadow casting for this light.
        """
        self.__render_context.utils.create_default_light(enable_shadows)

    def assign_checkerboard_material_to_all_shapes(self, resolution: int = 64, checker_size: int = 32):
        """Assign a gray checkerboard texture material to all shapes.
        Creates a gray checkerboard pattern texture and applies it to all shapes
        in the scene.

        Args:
            resolution: Texture resolution in pixels (square texture).
            checker_size: Size of each checkerboard square in pixels.
        """
        self.__render_context.utils.assign_checkerboard_material_to_all_shapes(resolution, checker_size)

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
        return self.__render_context.utils.create_color_image_output(width, height, camera_count)

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
        return self.__render_context.utils.create_depth_image_output(width, height, camera_count)

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
        return self.__render_context.utils.create_shape_index_image_output(width, height, camera_count)

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
        return self.__render_context.utils.create_normal_image_output(width, height, camera_count)

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
        return self.__render_context.utils.create_albedo_image_output(width, height, camera_count)

    @property
    def render_context(self) -> RenderContext:
        """Internal Warp raytracing context used by :meth:`update` and buffer helpers.

        Deprecated: direct access is deprecated and will be removed; prefer this class's
        public methods, or `SensorTiledCamera.render_config` for `RenderConfig` access.

        Returns:
            The shared :class:`RenderContext` instance.
        """
        warnings.warn(
            "Direct access to SensorTiledCamera.render_context is deprecated and will be removed in a future release.",
            category=DeprecationWarning,
            stacklevel=2,
        )
        return self.__render_context

    @property
    def render_config(self) -> SensorTiledCamera.RenderConfig:
        """Low-level raytrace settings on the internal :class:`RenderContext`.

        Populated at construction from :class:`Config` and from fixed defaults
        (for example global world and shadow flags on the context). Attributes may
        be modified to change behavior for subsequent :meth:`update` calls.

        Returns:
            The live :class:`SensorTiledCamera.RenderConfig` instance (same object as
            ``render_context.config`` without triggering deprecation warnings).
        """
        return self.__render_context.config
