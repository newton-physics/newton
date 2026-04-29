# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import warp as wp

from ..geometry import (
    build_bvh_particle,
    build_bvh_shape,
    refit_bvh_particle,
    refit_bvh_shape,
)
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

            # After setup, build BVHs once for the initial state.
            state = model.state()
            newton.geometry.build_bvh_shape(model, state)
            newton.geometry.build_bvh_particle(model, state)

            # Before each later frame that changes geometry, refit BVHs.
            newton.geometry.refit_bvh_shape(model, state)
            newton.geometry.refit_bvh_particle(model, state)
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

        .. deprecated:: 1.1
            Use :class:`RenderConfig` and ``SensorTiledCamera.utils.*`` instead.
        """

        checkerboard_texture: bool = False
        """.. deprecated:: 1.1 Use ``SensorTiledCamera.utils.assign_checkerboard_material_to_all_shapes()`` instead."""

        default_light: bool = False
        """.. deprecated:: 1.1 Use ``SensorTiledCamera.utils.create_default_light()`` instead."""

        default_light_shadows: bool = False
        """.. deprecated:: 1.1 Use ``SensorTiledCamera.utils.create_default_light(enable_shadows=True)`` instead."""

        enable_ambient_lighting: bool = True
        """.. deprecated:: 1.1 Use ``render_config.enable_ambient_lighting`` instead."""

        colors_per_world: bool = False
        """.. deprecated:: 1.1 Use shape colors instead (e.g. ``builder.add_shape_cylinder(..., color=(r, g, b))``)."""

        colors_per_shape: bool = False
        """.. deprecated:: 1.1 Use shape colors instead (e.g. ``builder.add_shape_cylinder(..., color=(r, g, b))``)."""

        backface_culling: bool = True
        """.. deprecated:: 1.1 Use ``render_config.enable_backface_culling`` instead."""

        enable_textures: bool = False
        """.. deprecated:: 1.1 Use ``render_config.enable_textures`` instead."""

        enable_particles: bool = True
        """.. deprecated:: 1.1 Use ``render_config.enable_particles`` instead."""

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
        """Synchronize triangle-mesh points from the simulation state.

        :meth:`update` calls this automatically when *state* is not None.

        Shape and particle BVHs on :attr:`model` must be built once via
        :func:`~newton.geometry.build_bvh_shape` and
        :func:`~newton.geometry.build_bvh_particle` before first use. Before
        later frames that change geometry, refit them via
        :func:`~newton.geometry.refit_bvh_shape` and
        :func:`~newton.geometry.refit_bvh_particle` prior to calling
        :meth:`update`.

        Args:
            state: The current simulation state containing particle positions.
        """
        self.__render_context.update(self.model, state)

    def update(
        self,
        state: State | None = None,
        camera_transforms: wp.array2d[wp.transformf] | None = None,
        camera_rays: wp.array4d[wp.vec3f] | None = None,
        *,
        color_image: wp.array4d[wp.uint32] | None = None,
        depth_image: wp.array4d[wp.float32] | None = None,
        shape_index_image: wp.array4d[wp.uint32] | None = None,
        normal_image: wp.array4d[wp.vec3f] | None = None,
        albedo_image: wp.array4d[wp.uint32] | None = None,
        clear_data: ClearData | None = DEFAULT_CLEAR_DATA,
        refit_bvh: bool | None = None,
    ):
        """Render output images for all worlds and cameras.

        Each output array has shape ``(world_count, camera_count, height, width)`` where element
        ``[world_id, camera_id, y, x]`` corresponds to the ray in ``camera_rays[camera_id, y, x]``. Each output
        channel is optional -- pass None to skip that channel's rendering entirely.

        Shape and particle BVHs on :attr:`model` must be built once for the
        initial state via :func:`~newton.geometry.build_bvh_shape` and
        :func:`~newton.geometry.build_bvh_particle` before first use. Before
        later frames that change geometry, refit them for *state* via
        :func:`~newton.geometry.refit_bvh_shape` and
        :func:`~newton.geometry.refit_bvh_particle` before calling this method.

        Args:
            state: Simulation state with body and particle transforms.
                Passing ``None`` is deprecated and will be removed in a future release.
            camera_transforms: Camera-to-world transforms, shape ``(camera_count, world_count)``.
            camera_rays: Camera-space rays from :meth:`compute_pinhole_camera_rays`, shape
                ``(camera_count, height, width, 2)``.
            color_image: Output for RGBA color. None to skip.
            depth_image: Output for ray-hit distance [m]. None to skip.
            shape_index_image: Output for per-pixel shape id. None to skip.
            normal_image: Output for surface normals. None to skip.
            albedo_image: Output for unshaded surface color. None to skip.
            clear_data: Values to clear output buffers with.
                See :attr:`DEFAULT_CLEAR_DATA`, :attr:`GRAY_CLEAR_DATA`.
            refit_bvh: Refit the BVH before rendering. This is deprecated, use
                :func:`~newton.geometry.build_bvh_shape`,
                :func:`~newton.geometry.refit_bvh_shape`,
                :func:`~newton.geometry.build_bvh_particle`, and
                :func:`~newton.geometry.refit_bvh_particle` explicitly
                before calling this method instead.
        """

        # TODO: Remove this deprecation behaviour in the next release.
        # state will be required and refit_bvh will be removed.
        render_state = state if state is not None else self.model.state()

        if state is None or refit_bvh is not None:
            warnings.warn(
                "Passing state=None or refit_bvh to SensorTiledCamera.update() is deprecated. "
                "Call SensorTiledCamera.sync_transforms(state) and manage BVHs explicitly with "
                "newton.geometry.build_bvh_*() / refit_bvh_*() before update().",
                category=DeprecationWarning,
                stacklevel=2,
            )
            should_refit = True if refit_bvh is None else refit_bvh

            if self.model.shape_count:
                if self.model.bvh_shapes is None:
                    build_bvh_shape(self.model, render_state)
                elif should_refit:
                    refit_bvh_shape(self.model, render_state)

            if render_state.particle_q is not None and render_state.particle_count:
                if self.model.bvh_particles is None:
                    build_bvh_particle(self.model, render_state)
                elif should_refit:
                    refit_bvh_particle(self.model, render_state)

        self.sync_transforms(render_state)

        self.__render_context.render(
            self.model,
            render_state,
            camera_transforms,
            camera_rays,
            color_image,
            depth_image,
            shape_index_image,
            normal_image,
            albedo_image,
            clear_data=clear_data,
        )

    def compute_pinhole_camera_rays(
        self, width: int, height: int, camera_fovs: float | list[float] | np.ndarray | wp.array[wp.float32]
    ) -> wp.array4d[wp.vec3f]:
        """Compute camera-space ray directions for pinhole cameras.

        Generates rays in camera space (origin at the camera center, direction normalized) for each pixel based on the
        vertical field of view.

        .. deprecated:: 1.1
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
        image: wp.array4d[wp.uint32],
        out_buffer: wp.array3d[wp.uint8] | None = None,
        worlds_per_row: int | None = None,
    ):
        """Flatten rendered color image to a tiled RGBA buffer.

        Arranges ``(world_count * camera_count)`` tiles in a grid. Each tile shows one camera's view of one world.

        .. deprecated:: 1.1
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
        image: wp.array4d[wp.vec3f],
        out_buffer: wp.array3d[wp.uint8] | None = None,
        worlds_per_row: int | None = None,
    ):
        """Flatten rendered normal image to a tiled RGBA buffer.

        Arranges ``(world_count * camera_count)`` tiles in a grid. Each tile shows one camera's view of one world.

        .. deprecated:: 1.1
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
        image: wp.array4d[wp.float32],
        out_buffer: wp.array3d[wp.uint8] | None = None,
        worlds_per_row: int | None = None,
        depth_range: wp.array[wp.float32] | None = None,
    ):
        """Flatten rendered depth image to a tiled RGBA buffer.

        Encodes depth as grayscale: inverts values (closer = brighter) and normalizes to the ``[50, 255]``
        range. Background pixels (no hit) remain black.

        .. deprecated:: 1.1
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

        .. deprecated:: 1.1
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

        .. deprecated:: 1.1
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

        .. deprecated:: 1.1
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

        .. deprecated:: 1.1
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

    def create_color_image_output(self, width: int, height: int, camera_count: int = 1) -> wp.array4d[wp.uint32]:
        """Create a color output array for :meth:`update`.

        .. deprecated:: 1.1
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

    def create_depth_image_output(self, width: int, height: int, camera_count: int = 1) -> wp.array4d[wp.float32]:
        """Create a depth output array for :meth:`update`.

        .. deprecated:: 1.1
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

    def create_shape_index_image_output(self, width: int, height: int, camera_count: int = 1) -> wp.array4d[wp.uint32]:
        """Create a shape-index output array for :meth:`update`.

        .. deprecated:: 1.1
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

    def create_normal_image_output(self, width: int, height: int, camera_count: int = 1) -> wp.array4d[wp.vec3f]:
        """Create a normal output array for :meth:`update`.

        .. deprecated:: 1.1
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

    def create_albedo_image_output(self, width: int, height: int, camera_count: int = 1) -> wp.array4d[wp.uint32]:
        """Create an albedo output array for :meth:`update`.

        .. deprecated:: 1.1
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

        .. deprecated:: 1.1
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
