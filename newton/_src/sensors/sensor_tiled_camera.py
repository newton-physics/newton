# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import warp as wp

from ..sim import Model, State
from .warp_raytrace import (
    ClearData,
    GaussianRenderMode,
    RenderConfig,
    RenderContext,
    RenderLightType,
    RenderOrder,
    Utils,
)


class _SensorTiledCameraMeta(type):
    @property
    def RenderContext(cls) -> type[RenderContext]:
        warnings.warn(
            "Access to SensorTiledCamera.RenderContext is deprecated.",
            DeprecationWarning,
            stacklevel=2,
        )
        return RenderContext


class SensorTiledCamera(metaclass=_SensorTiledCameraMeta):
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
            rays = sensor.utils.compute_pinhole_camera_rays(width, height, fov)
            color = sensor.utils.create_color_image_output(width, height)

            # each step
            sensor.update(state, camera_transforms, rays, color_image=color)

    See :class:`RenderConfig` for optional rendering settings and :attr:`ClearData` / :attr:`DEFAULT_CLEAR_DATA` /
    :attr:`GRAY_CLEAR_DATA` for image-clear presets.
    """

    RenderLightType = RenderLightType
    RenderOrder = RenderOrder
    GaussianRenderMode = GaussianRenderMode
    RenderConfig = RenderConfig
    ClearData = ClearData
    Utils = Utils

    DEFAULT_CLEAR_DATA = ClearData()
    GRAY_CLEAR_DATA = ClearData(clear_color=0xFF666666, clear_albedo=0xFF000000)

    @dataclass
    class Config:
        """Sensor configuration.

        .. deprecated::
            Use :class:`RenderConfig` and ``SensorTiledCamera.utils.*`` instead.
        """

        checkerboard_texture: bool = False
        """.. deprecated:: Use ``SensorTiledCamera.utils.assign_checkerboard_material_to_all_shapes()`` instead."""

        default_light: bool = False
        """.. deprecated:: Use ``SensorTiledCamera.utils.create_default_light()`` instead."""

        default_light_shadows: bool = False
        """.. deprecated:: Use ``SensorTiledCamera.utils.create_default_light(enable_shadows=True)`` instead."""

        enable_ambient_lighting: bool = True
        """.. deprecated:: Use ``render_config.enable_ambient_lighting`` instead."""

        colors_per_world: bool = False
        """.. deprecated:: Use shape colors instead (e.g. ``builder.add_shape_cylinder(..., color=(r, g, b))``)."""

        colors_per_shape: bool = False
        """.. deprecated:: Use shape colors instead (e.g. ``builder.add_shape_cylinder(..., color=(r, g, b))``)."""

        backface_culling: bool = True
        """.. deprecated:: Use ``render_config.enable_backface_culling`` instead."""

        enable_textures: bool = False
        """.. deprecated:: Use ``render_config.enable_textures`` instead."""

        enable_particles: bool = True
        """.. deprecated:: Use ``render_config.enable_particles`` instead."""

    def __init__(self, model: Model, *, config: Config | RenderConfig | None = None, load_textures: bool = True):
        """Initialize the tiled camera sensor from a simulation model.

        Builds the internal :class:`RenderContext`, loads shape geometry (and
        optionally textures) from *model*, and exposes :attr:`utils` for
        creating output buffers, computing rays, and assigning materials.

        Args:
            model: Simulation model whose shapes will be rendered.
            config: Rendering configuration. Pass a :class:`RenderConfig` to
                control raytrace settings directly, or ``None`` to use
                defaults. The legacy :class:`Config` dataclass is still
                accepted but deprecated.
            load_textures: Load texture data from the model. Set to ``False``
                to skip texture loading when textures are not needed.
        """
        self.model = model

        render_config = config

        if render_config is None:
            render_config = RenderConfig()

        elif isinstance(config, SensorTiledCamera.Config):
            warnings.warn(
                "SensorTiledCamera.Config is deprecated, use SensorTiledCamera.RenderConfig and SensorTiledCamera.utils.* functions instead.",
                category=DeprecationWarning,
                stacklevel=2,
            )

            render_config = RenderConfig()
            render_config.enable_ambient_lighting = config.enable_ambient_lighting
            render_config.enable_backface_culling = config.backface_culling
            render_config.enable_textures = config.enable_textures
            render_config.enable_particles = config.enable_particles

        self.__render_context = RenderContext(
            world_count=self.model.world_count,
            config=render_config,
            device=self.model.device,
        )

        self.__render_context.init_from_model(self.model, load_textures)

        if isinstance(config, SensorTiledCamera.Config):
            if config.checkerboard_texture:
                self.utils.assign_checkerboard_material_to_all_shapes()
            if config.default_light:
                self.utils.create_default_light(config.default_light_shadows)
            if config.colors_per_world:
                self.utils.assign_random_colors_per_world()
            elif config.colors_per_shape:
                self.utils.assign_random_colors_per_shape()

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
        clear_data: ClearData | None = DEFAULT_CLEAR_DATA,
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

        .. deprecated::
            Use ``SensorTiledCamera.utils.compute_pinhole_camera_rays`` instead.

        Args:
            width: Image width [px].
            height: Image height [px].
            camera_fovs: Vertical FOV angles [rad], shape ``(camera_count,)``.

        Returns:
            camera_rays: Shape ``(camera_count, height, width, 2)``, dtype ``vec3f``.
        """
        warnings.warn(
            "Deprecated: SensorTiledCamera.compute_pinhole_camera_rays is deprecated, use SensorTiledCamera.utils.compute_pinhole_camera_rays instead.",
            category=DeprecationWarning,
            stacklevel=2,
        )

        return self.__render_context.utils.compute_pinhole_camera_rays(width, height, camera_fovs)

    def flatten_color_image_to_rgba(
        self,
        image: wp.array(dtype=wp.uint32, ndim=4),
        out_buffer: wp.array(dtype=wp.uint8, ndim=3) | None = None,
        worlds_per_row: int | None = None,
    ):
        """Flatten rendered color image to a tiled RGBA buffer.

        Arranges ``(world_count * camera_count)`` tiles in a grid. Each tile shows one camera's view of one world.

        .. deprecated::
            Use ``SensorTiledCamera.utils.flatten_color_image_to_rgba`` instead.

        Args:
            image: Color output from :meth:`update`, shape ``(world_count, camera_count, height, width)``.
            out_buffer: Pre-allocated RGBA buffer. If None, allocates a new one.
            worlds_per_row: Tiles per row in the grid. If None, picks a square-ish layout.
        """
        warnings.warn(
            "Deprecated: SensorTiledCamera.flatten_color_image_to_rgba is deprecated, use SensorTiledCamera.utils.flatten_color_image_to_rgba instead.",
            category=DeprecationWarning,
            stacklevel=2,
        )
        return self.utils.flatten_color_image_to_rgba(image, out_buffer, worlds_per_row)

    def flatten_normal_image_to_rgba(
        self,
        image: wp.array(dtype=wp.vec3f, ndim=4),
        out_buffer: wp.array(dtype=wp.uint8, ndim=3) | None = None,
        worlds_per_row: int | None = None,
    ):
        """Flatten rendered normal image to a tiled RGBA buffer.

        Arranges ``(world_count * camera_count)`` tiles in a grid. Each tile shows one camera's view of one world.

        .. deprecated::
            Use ``SensorTiledCamera.utils.flatten_normal_image_to_rgba`` instead.

        Args:
            image: Normal output from :meth:`update`, shape ``(world_count, camera_count, height, width)``.
            out_buffer: Pre-allocated RGBA buffer. If None, allocates a new one.
            worlds_per_row: Tiles per row in the grid. If None, picks a square-ish layout.
        """
        warnings.warn(
            "Deprecated: SensorTiledCamera.flatten_normal_image_to_rgba is deprecated, use SensorTiledCamera.utils.flatten_normal_image_to_rgba instead.",
            category=DeprecationWarning,
            stacklevel=2,
        )
        return self.utils.flatten_normal_image_to_rgba(image, out_buffer, worlds_per_row)

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

        .. deprecated::
            Use ``SensorTiledCamera.utils.flatten_depth_image_to_rgba`` instead.

        Args:
            image: Depth output from :meth:`update`, shape ``(world_count, camera_count, height, width)``.
            out_buffer: Pre-allocated RGBA buffer. If None, allocates a new one.
            worlds_per_row: Tiles per row in the grid. If None, picks a square-ish layout.
            depth_range: Depth range to normalize to, shape ``(2,)`` ``[near, far]``. If None, computes from *image*.
        """
        warnings.warn(
            "Deprecated: SensorTiledCamera.flatten_depth_image_to_rgba is deprecated, use SensorTiledCamera.utils.flatten_depth_image_to_rgba instead.",
            category=DeprecationWarning,
            stacklevel=2,
        )
        return self.utils.flatten_depth_image_to_rgba(image, out_buffer, worlds_per_row, depth_range)

    def assign_random_colors_per_world(self, seed: int = 100):
        """Assign each world a random color, applied to all its shapes.

        .. deprecated::
            Use shape colors instead (e.g. ``builder.add_shape_cylinder(..., color=(r, g, b))``).

        Args:
            seed: Random seed.
        """
        warnings.warn(
            "``SensorTiledCamera.assign_random_colors_per_world`` is deprecated. Use shape colors instead (e.g. ``builder.add_shape_cylinder(..., color=(r, g, b))``).",
            category=DeprecationWarning,
            stacklevel=2,
        )
        self.utils.assign_random_colors_per_world(seed)

    def assign_random_colors_per_shape(self, seed: int = 100):
        """Assign a random color to each shape.

        .. deprecated::
            Use shape colors instead (e.g. ``builder.add_shape_cylinder(..., color=(r, g, b))``).

        Args:
            seed: Random seed.
        """
        warnings.warn(
            "``SensorTiledCamera.assign_random_colors_per_shape`` is deprecated. Use shape colors instead (e.g. ``builder.add_shape_cylinder(..., color=(r, g, b))``).",
            category=DeprecationWarning,
            stacklevel=2,
        )
        self.utils.assign_random_colors_per_shape(seed)

    def create_default_light(self, enable_shadows: bool = True):
        """Create a default directional light oriented at ``(-1, 1, -1)``.

        .. deprecated::
            Use ``SensorTiledCamera.utils.create_default_light`` instead.

        Args:
            enable_shadows: Enable shadow casting for this light.
        """
        warnings.warn(
            "Deprecated: SensorTiledCamera.create_default_light is deprecated, use SensorTiledCamera.utils.create_default_light instead.",
            category=DeprecationWarning,
            stacklevel=2,
        )
        self.utils.create_default_light(enable_shadows)

    def assign_checkerboard_material_to_all_shapes(self, resolution: int = 64, checker_size: int = 32):
        """Assign a gray checkerboard texture material to all shapes.

        Creates a gray checkerboard pattern texture and applies it to all shapes
        in the scene.

        .. deprecated::
            Use ``SensorTiledCamera.utils.assign_checkerboard_material_to_all_shapes`` instead.

        Args:
            resolution: Texture resolution in pixels (square texture).
            checker_size: Size of each checkerboard square in pixels.
        """
        warnings.warn(
            "Deprecated: SensorTiledCamera.assign_checkerboard_material_to_all_shapes is deprecated, use SensorTiledCamera.utils.assign_checkerboard_material_to_all_shapes instead.",
            category=DeprecationWarning,
            stacklevel=2,
        )
        self.utils.assign_checkerboard_material_to_all_shapes(resolution, checker_size)

    def create_color_image_output(self, width: int, height: int, camera_count: int = 1) -> wp.array(
        dtype=wp.uint32, ndim=4
    ):
        """Create a color output array for :meth:`update`.

        .. deprecated::
            Use ``SensorTiledCamera.utils.create_color_image_output`` instead.

        Args:
            width: Image width [px].
            height: Image height [px].
            camera_count: Number of cameras.

        Returns:
            Array of shape ``(world_count, camera_count, height, width)``, dtype ``uint32``.
        """
        warnings.warn(
            "Deprecated: SensorTiledCamera.create_color_image_output is deprecated, use SensorTiledCamera.utils.create_color_image_output instead.",
            category=DeprecationWarning,
            stacklevel=2,
        )
        return self.utils.create_color_image_output(width, height, camera_count)

    def create_depth_image_output(self, width: int, height: int, camera_count: int = 1) -> wp.array(
        dtype=wp.float32, ndim=4
    ):
        """Create a depth output array for :meth:`update`.

        .. deprecated::
            Use ``SensorTiledCamera.utils.create_depth_image_output`` instead.

        Args:
            width: Image width [px].
            height: Image height [px].
            camera_count: Number of cameras.

        Returns:
            Array of shape ``(world_count, camera_count, height, width)``, dtype ``float32``.
        """
        warnings.warn(
            "Deprecated: SensorTiledCamera.create_depth_image_output is deprecated, use SensorTiledCamera.utils.create_depth_image_output instead.",
            category=DeprecationWarning,
            stacklevel=2,
        )
        return self.utils.create_depth_image_output(width, height, camera_count)

    def create_shape_index_image_output(self, width: int, height: int, camera_count: int = 1) -> wp.array(
        dtype=wp.uint32, ndim=4
    ):
        """Create a shape-index output array for :meth:`update`.

        .. deprecated::
            Use ``SensorTiledCamera.utils.create_shape_index_image_output`` instead.

        Args:
            width: Image width [px].
            height: Image height [px].
            camera_count: Number of cameras.

        Returns:
            Array of shape ``(world_count, camera_count, height, width)``, dtype ``uint32``.
        """
        warnings.warn(
            "Deprecated: SensorTiledCamera.create_shape_index_image_output is deprecated, use SensorTiledCamera.utils.create_shape_index_image_output instead.",
            category=DeprecationWarning,
            stacklevel=2,
        )
        return self.utils.create_shape_index_image_output(width, height, camera_count)

    def create_normal_image_output(self, width: int, height: int, camera_count: int = 1) -> wp.array(
        dtype=wp.vec3f, ndim=4
    ):
        """Create a normal output array for :meth:`update`.

        .. deprecated::
            Use ``SensorTiledCamera.utils.create_normal_image_output`` instead.

        Args:
            width: Image width [px].
            height: Image height [px].
            camera_count: Number of cameras.

        Returns:
            Array of shape ``(world_count, camera_count, height, width)``, dtype ``vec3f``.
        """
        warnings.warn(
            "Deprecated: SensorTiledCamera.create_normal_image_output is deprecated, use SensorTiledCamera.utils.create_normal_image_output instead.",
            category=DeprecationWarning,
            stacklevel=2,
        )
        return self.utils.create_normal_image_output(width, height, camera_count)

    def create_albedo_image_output(self, width: int, height: int, camera_count: int = 1) -> wp.array(
        dtype=wp.uint32, ndim=4
    ):
        """Create an albedo output array for :meth:`update`.

        .. deprecated::
            Use ``SensorTiledCamera.utils.create_albedo_image_output`` instead.

        Args:
            width: Image width [px].
            height: Image height [px].
            camera_count: Number of cameras.

        Returns:
            Array of shape ``(world_count, camera_count, height, width)``, dtype ``uint32``.
        """
        warnings.warn(
            "Deprecated: SensorTiledCamera.create_albedo_image_output is deprecated, use SensorTiledCamera.utils.create_albedo_image_output instead.",
            category=DeprecationWarning,
            stacklevel=2,
        )
        return self.utils.create_albedo_image_output(width, height, camera_count)

    @property
    def render_context(self) -> RenderContext:
        """Internal Warp raytracing context used by :meth:`update` and buffer helpers.

        .. deprecated::
            Direct access is deprecated and will be removed. Prefer this
            class's public methods, or :attr:`render_config` for
            :class:`RenderConfig` access.

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
    def render_config(self) -> RenderConfig:
        """Low-level raytrace settings on the internal :class:`RenderContext`.

        Populated at construction from :class:`Config` and from fixed defaults
        (for example global world and shadow flags on the context). Attributes may
        be modified to change behavior for subsequent :meth:`update` calls.

        Returns:
            The live :class:`RenderConfig` instance (same object as
            ``render_context.config`` without triggering deprecation warnings).
        """
        return self.__render_context.config

    @property
    def utils(self) -> Utils:
        """Utility helpers for creating output buffers, computing rays, and assigning materials/lights."""
        return self.__render_context.utils
