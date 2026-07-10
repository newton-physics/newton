# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

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


class SensorTiledCamera:
    """Warp-based tiled camera sensor for raytraced rendering across multiple worlds.

    Renders up to six image channels per (world, camera) pair:

    - **color** -- RGBA shaded image (``uint32``).
    - **hdr_color** -- linear shaded RGB image (``vec3f``).
    - **depth** -- ray-hit distance [m] (``float32``); negative means no hit.
    - **normal** -- surface normal at hit point (``vec3f``).
    - **albedo** -- unshaded surface color (``uint32``).
    - **shape_index** -- shape id per pixel (``uint32``).

    All output arrays have shape ``(world_count, camera_count, height, width)``. Use the ``flatten_*`` helpers to
    rearrange them into tiled RGBA buffers for display, with one tile per (world, camera) pair laid out in a grid.

    Shapes without the ``VISIBLE`` flag are excluded.

    Shape colors and base-color textures are interpreted as display/sRGB RGB,
    converted to linear RGB internally for shading, and packed according to
    :attr:`RenderConfig.output_color_space` at the output boundary.

    Example:
        ::

            sensor = SensorTiledCamera(model)
            rays = sensor.utils.compute_camera_rays_pinhole(width, height, camera_fovs=fov)
            color = sensor.utils.create_color_image_output(width, height)

            # BVHs are built for the initial state by ModelBuilder.finalize().
            state = model.state()

            # Before each frame that changes geometry, refit BVHs.
            model.bvh_refit_shapes(state)
            model.bvh_refit_particles(state)
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

    def __init__(self, model: Model, *, config: RenderConfig | None = None, load_textures: bool = True):
        """Initialize the tiled camera sensor from a simulation model.

        Builds the internal :class:`RenderContext`, loads shape geometry (and
        optionally textures) from *model*, and exposes :attr:`utils` for
        creating output buffers, computing rays, and assigning materials.

        Args:
            model: Simulation model whose shapes will be rendered.
            config: Rendering configuration. Pass a :class:`RenderConfig` to
                control raytrace settings directly, or ``None`` to use
                defaults. Use ``RenderConfig.output_color_space`` to control
                whether packed ``color`` and ``albedo`` outputs are
                display-encoded or left linear.
            load_textures: Load texture data from the model. Set to ``False``
                to skip texture loading when textures are not needed.
        """
        self.model = model

        render_config = config if config is not None else RenderConfig()

        self.__render_context = RenderContext(
            world_count=self.model.world_count,
            config=render_config,
            device=self.model.device,
        )

        self.__render_context.init_from_model(self.model, load_textures)

    def sync_transforms(self, state: State):
        """Synchronize triangle-mesh points from the simulation state.

        :meth:`update` calls this automatically when *state* is not None.

        Shape and particle BVHs on :attr:`model` are built for the initial
        state by :meth:`~newton.ModelBuilder.finalize`. Before later frames
        that change geometry, refit them via
        :meth:`~newton.Model.bvh_refit_shapes` and
        :meth:`~newton.Model.bvh_refit_particles` prior to calling
        :meth:`update`.

        Args:
            state: The current simulation state containing particle positions.
        """
        self.__render_context.update(self.model, state)

    def update(
        self,
        state: State,
        camera_transforms: wp.array2d[wp.transformf] | None = None,
        camera_rays: wp.array4d[wp.vec3f] | None = None,
        *,
        color_image: wp.array4d[wp.uint32] | None = None,
        depth_image: wp.array4d[wp.float32] | None = None,
        shape_index_image: wp.array4d[wp.uint32] | None = None,
        normal_image: wp.array4d[wp.vec3f] | None = None,
        albedo_image: wp.array4d[wp.uint32] | None = None,
        clear_data: ClearData | None = DEFAULT_CLEAR_DATA,
        hdr_color_image: wp.array4d[wp.vec3f] | None = None,
        kernel_block_dim: int = 64,
    ):
        """Render output images for all worlds and cameras.

        Each output array has shape ``(world_count, camera_count, height, width)`` where element
        ``[world_id, camera_id, y, x]`` corresponds to the ray in ``camera_rays[camera_id, y, x]``. Each output
        channel is optional -- pass None to skip that channel's rendering entirely.

        Shape and particle BVHs on :attr:`model` are built for the initial
        state by :meth:`~newton.ModelBuilder.finalize`. Before later frames
        that change geometry, refit them for *state* via
        :meth:`~newton.Model.bvh_refit_shapes` and
        :meth:`~newton.Model.bvh_refit_particles` before calling this method.

        Args:
            state: Simulation state with body and particle transforms.
            camera_transforms: Camera-to-world transforms, shape ``(camera_count, world_count)``.
            camera_rays: Camera-space rays from ``SensorTiledCamera.utils`` ray helpers, shape
                ``(camera_count, height, width, 2)``.
            color_image: Output for packed RGBA color. The bytes are
                display/sRGB by default, or linear when
                ``self.render_config.output_color_space`` is
                ``newton.utils.ColorSpace.LINEAR``. None to skip.
            depth_image: Output for ray-hit distance [m]. None to skip.
            shape_index_image: Output for per-pixel shape id. None to skip.
            normal_image: Output for surface normals. None to skip.
            albedo_image: Output for packed unshaded surface color, using the
                same output color space as ``color_image``. None to skip.
            clear_data: Values to clear output buffers with. Packed color and
                albedo clear values are specified as display/sRGB RGBA and
                converted to linear when linear output is requested. See
                :attr:`DEFAULT_CLEAR_DATA`, :attr:`GRAY_CLEAR_DATA`.
            hdr_color_image: Output for linear HDR color. None to skip.
            kernel_block_dim: Thread block dimension forwarded to ``wp.launch``
                for the render megakernel.
        """

        self.sync_transforms(state)

        self.__render_context.render(
            self.model,
            state,
            camera_transforms,
            camera_rays,
            color_image,
            depth_image,
            shape_index_image,
            normal_image,
            albedo_image,
            clear_data=clear_data,
            hdr_color_image=hdr_color_image,
            kernel_block_dim=kernel_block_dim,
        )

    @property
    def render_config(self) -> RenderConfig:
        """Low-level raytrace settings on the internal :class:`RenderContext`.

        Populated at construction from fixed defaults (for example global
        world and shadow flags on the context). Attributes may be modified to
        change behavior for subsequent :meth:`update` calls.

        Returns:
            The live :class:`RenderConfig` instance (same object as
            ``render_context.config`` without triggering deprecation warnings).
        """
        return self.__render_context.config

    @property
    def utils(self) -> Utils:
        """Utility helpers for creating output buffers, computing rays, and assigning materials/lights."""
        return self.__render_context.utils
