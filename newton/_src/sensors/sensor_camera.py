# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math
import os
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import numpy as np
import warp as wp

from ..core.types import Devicelike
from .sensor_camera_render import Utils
from .sensor_camera_render.types import (
    ClearData,
    GaussianRenderMode,
    LightType,
    RenderConfig,
    RenderOrder,
    TextureProjectionMode,
    WorldRenderFlag,
)

if TYPE_CHECKING:
    from ..sim.model import Model
    from ..sim.state import State
    from .sensor_camera_render.render_context import RenderContext

# Enable NVTX ranges / timing around SensorCamera.update() when NEWTON_PROFILE is set.
PROFILE_ENABLED = os.environ.get("NEWTON_PROFILE", "0") != "0"


def _resolve_fisheye_image_size(
    axis: str,
    image_size: float | None,
    nominal_size: float | None,
    default_size: int,
) -> float:
    if image_size is not None and nominal_size is not None and image_size != nominal_size:
        raise ValueError(f"image_{axis} and nominal_{axis} must match when both are provided.")
    if image_size is not None:
        return float(image_size)
    if nominal_size is not None:
        return float(nominal_size)
    return float(default_size)


def _validate_camera_ray_output(
    width: int,
    height: int,
    out_rays: wp.array3d[wp.vec3f] | None,
    device: Devicelike = None,
) -> tuple[int, int, wp.array3d[wp.vec3f], wp.Device]:
    width = int(width)
    height = int(height)
    if width <= 0 or height <= 0:
        raise ValueError("width and height must be positive.")

    expected_shape = (height, width, 2)
    target_device = wp.get_device(device) if device is not None else None

    if out_rays is None:
        out_rays = wp.empty(expected_shape, dtype=wp.vec3f, device=device)
    else:
        if not isinstance(out_rays, wp.array):
            raise TypeError(f"out_rays must be a Warp array, got {type(out_rays).__name__}")
        if out_rays.dtype != wp.vec3f:
            raise ValueError(f"out_rays must have dtype vec3f, got {out_rays.dtype}")
        if out_rays.shape != expected_shape:
            raise ValueError(f"out_rays must have shape {expected_shape}, got {out_rays.shape}")
        if target_device is not None and out_rays.device != target_device:
            raise ValueError(f"out_rays is on {out_rays.device}, expected {target_device}")

    return width, height, out_rays, out_rays.device


class SensorCamera:
    """Raytraced camera sensor that renders a Newton model.

    A camera sensor owns an internal renderer built for a model plus the default
    render settings (:attr:`default_clear_data`, :attr:`default_render_config`)
    used when :meth:`update` is not given per-call overrides. The caller supplies
    the camera-space rays, the world-space per-view camera transforms, and the
    output image buffers to :meth:`update`; the number of views is inferred from
    the leading dimension of the camera transforms.

    The render configuration types are exposed as nested attributes (e.g.
    ``SensorCamera.RenderConfig``, ``SensorCamera.ClearData``,
    ``SensorCamera.WorldRenderFlag``); they are not part of the top-level
    ``newton`` namespace.
    """

    ClearData = ClearData
    GaussianRenderMode = GaussianRenderMode
    LightType = LightType
    RenderConfig = RenderConfig
    RenderOrder = RenderOrder
    TextureProjectionMode = TextureProjectionMode
    Utils = Utils
    WorldRenderFlag = WorldRenderFlag

    def __init__(
        self,
        model: Model | None = None,
        *,
        default_clear_data: ClearData | None = None,
        default_render_config: RenderConfig | None = None,
        load_textures: bool = True,
    ):
        """Construct a camera sensor for a model.

        Args:
            model: Newton simulation model to render. The sensor builds its
                internal renderer for the model. :meth:`update` requires a model.
            default_clear_data: Clear values used by :meth:`update` when its
                ``clear_data`` argument is ``None``. Defaults to ``ClearData()``.
            default_render_config: Render settings used by :meth:`update` when
                its ``render_config`` argument is ``None``. Defaults to
                ``RenderConfig()``.
            load_textures: Load mesh textures from disk. Set ``False`` for
                checkerboard or custom-texture workflows (see
                :meth:`assign_checkerboard_material`).
        """
        self.default_clear_data: ClearData = default_clear_data if default_clear_data is not None else ClearData()
        """Clear values used by :meth:`update` when its ``clear_data`` argument is ``None``."""
        self.default_render_config: RenderConfig = (
            default_render_config if default_render_config is not None else RenderConfig()
        )
        """Render settings used by :meth:`update` when its ``render_config`` argument is ``None``."""

        self._render_context = None
        if model is not None:
            from .sensor_camera_render.render_context import RenderContext  # noqa: PLC0415

            self._render_context = RenderContext(model, load_textures=load_textures)

    @property
    def device(self) -> wp.Device:
        """Device of the model this sensor renders."""
        return self._get_render_context().model.device

    def utils(self, view_count: int) -> Utils:
        """Return post-processing helpers (``to_rgba``/``flatten``/depth conversion) for ``view_count`` views.

        Args:
            view_count: Number of views the processed images carry (their leading dimension).

        Returns:
            A :class:`~newton.sensors.SensorCamera.Utils` bound to ``view_count`` and the model device.
        """
        return Utils(view_count=int(view_count), device=self.device)

    def create_image_output(self, view_count: int, width: int, height: int, dtype: Any) -> wp.array[Any]:
        """Create an output image array with shape ``(view_count, height, width)``."""
        return wp.zeros((int(view_count), int(height), int(width)), dtype=dtype, device=self.device)

    def create_color_image_output(self, view_count: int, width: int, height: int) -> wp.array3d[wp.uint32]:
        """Create an RGBA color output array (packed ``uint32``), shape ``(view_count, height, width)``."""
        return self.create_image_output(view_count, width, height, wp.uint32)

    def create_depth_image_output(self, view_count: int, width: int, height: int) -> wp.array3d[wp.float32]:
        """Create a depth output array [m], shape ``(view_count, height, width)``."""
        return self.create_image_output(view_count, width, height, wp.float32)

    def create_forward_depth_image_output(self, view_count: int, width: int, height: int) -> wp.array3d[wp.float32]:
        """Create a forward-depth output array [m], shape ``(view_count, height, width)``."""
        return self.create_depth_image_output(view_count, width, height)

    def create_shape_index_image_output(self, view_count: int, width: int, height: int) -> wp.array3d[wp.uint32]:
        """Create a shape-index output array, shape ``(view_count, height, width)``."""
        return self.create_image_output(view_count, width, height, wp.uint32)

    def create_normal_image_output(self, view_count: int, width: int, height: int) -> wp.array3d[wp.vec3f]:
        """Create a world-space surface-normal output array (``vec3f``), shape ``(view_count, height, width)``."""
        return self.create_image_output(view_count, width, height, wp.vec3f)

    def create_albedo_image_output(self, view_count: int, width: int, height: int) -> wp.array3d[wp.uint32]:
        """Create an RGBA albedo output array (packed ``uint32``), shape ``(view_count, height, width)``."""
        return self.create_image_output(view_count, width, height, wp.uint32)

    def create_hdr_color_image_output(self, view_count: int, width: int, height: int) -> wp.array3d[wp.vec3f]:
        """Create a linear HDR color output array, shape ``(view_count, height, width)``."""
        return self.create_image_output(view_count, width, height, wp.vec3f)

    @staticmethod
    def compute_camera_rays_pinhole(
        width: int,
        height: int,
        camera_fov: float | None = None,
        *,
        focal_length: float | None = None,
        horizontal_aperture: float | None = None,
        vertical_aperture: float | None = None,
        horizontal_aperture_offset: float = 0.0,
        vertical_aperture_offset: float = 0.0,
        out_rays: wp.array3d[wp.vec3f] | None = None,
        device: Devicelike = None,
    ) -> wp.array3d[wp.vec3f]:
        """Compute camera-space rays for one pinhole camera.

        Provide either ``camera_fov`` or the aperture triple (``focal_length``,
        ``horizontal_aperture``, ``vertical_aperture``), not both. The focal
        length and apertures share consistent units; only their ratios affect
        the ray directions.

        Args:
            width: Image width [px].
            height: Image height [px].
            camera_fov: Horizontal field of view [rad], in ``(0, pi)``. Mutually
                exclusive with the aperture parameters.
            focal_length: Lens focal length; must be positive.
            horizontal_aperture: Horizontal sensor aperture; must be positive.
            vertical_aperture: Vertical sensor aperture; must be positive.
            horizontal_aperture_offset: Horizontal principal-point offset.
            vertical_aperture_offset: Vertical principal-point offset.
            out_rays: Optional output buffer, shape ``(height, width, 2)`` of ``vec3f``.
            device: Device for the ray bundle. Defaults to the current Warp device.

        Returns:
            Ray origins and directions, shape ``(height, width, 2)`` of ``vec3f``.
        """
        from .sensor_camera_render import camera_utils  # noqa: PLC0415

        width, height, out_rays, device = _validate_camera_ray_output(width, height, out_rays, device)

        use_aperture = focal_length is not None or horizontal_aperture is not None or vertical_aperture is not None
        if use_aperture:
            if camera_fov is not None:
                raise ValueError("camera_fov cannot be provided with aperture parameters.")
            if focal_length is None or horizontal_aperture is None or vertical_aperture is None:
                raise ValueError("focal_length, horizontal_aperture, and vertical_aperture must be provided together.")
            if float(focal_length) <= 0.0 or float(horizontal_aperture) <= 0.0 or float(vertical_aperture) <= 0.0:
                raise ValueError("focal_length, horizontal_aperture, and vertical_aperture must be positive.")

            wp.launch(
                kernel=camera_utils.compute_camera_rays_pinhole_from_aperture_kernel,
                dim=(height, width),
                inputs=[
                    width,
                    height,
                    float(focal_length),
                    float(horizontal_aperture),
                    float(vertical_aperture),
                    float(horizontal_aperture_offset),
                    float(vertical_aperture_offset),
                    out_rays,
                ],
                device=device,
            )

            return out_rays

        if camera_fov is None:
            raise ValueError("camera_fov must be provided when aperture parameters are not used.")
        if not 0.0 < float(camera_fov) < math.pi:
            raise ValueError(f"camera_fov must be in (0, pi) radians, got {float(camera_fov)}.")

        wp.launch(
            kernel=camera_utils.compute_camera_rays_pinhole,
            dim=(height, width),
            inputs=[
                width,
                height,
                float(camera_fov),
                out_rays,
            ],
            device=device,
        )

        return out_rays

    @staticmethod
    def compute_camera_rays_usd_pinhole(
        width: int,
        height: int,
        camera: Any,
        *,
        time: Any | None = None,
        out_rays: wp.array3d[wp.vec3f] | None = None,
        device: Devicelike = None,
    ) -> wp.array3d[wp.vec3f]:
        """Compute camera-space rays for one USD pinhole camera."""
        from .sensor_camera_render import camera_utils  # noqa: PLC0415

        width, height, out_rays, device = _validate_camera_ray_output(width, height, out_rays, device)
        camera_utils.compute_camera_rays_usd_pinhole(
            width,
            height,
            camera,
            device=device,
            time=time,
            out_rays=out_rays,
        )
        return out_rays

    @staticmethod
    def compute_camera_rays_fisheye_opencv(
        width: int,
        height: int,
        fx: float,
        fy: float,
        cx: float,
        cy: float,
        *,
        image_width: float | None = None,
        image_height: float | None = None,
        k1: float = 0.0,
        k2: float = 0.0,
        k3: float = 0.0,
        k4: float = 0.0,
        max_fov: float = 2.0 * math.pi,
        out_rays: wp.array3d[wp.vec3f] | None = None,
        device: Devicelike = None,
    ) -> wp.array3d[wp.vec3f]:
        """Compute camera-space rays for one OpenCV fisheye camera."""
        from .sensor_camera_render import camera_utils  # noqa: PLC0415

        width, height, out_rays, device = _validate_camera_ray_output(width, height, out_rays, device)
        image_width = float(width) if image_width is None else float(image_width)
        image_height = float(height) if image_height is None else float(image_height)

        wp.launch(
            kernel=camera_utils.compute_camera_rays_fisheye_opencv_kernel,
            dim=(height, width),
            inputs=[
                width,
                height,
                image_width,
                image_height,
                fx,
                fy,
                cx,
                cy,
                k1,
                k2,
                k3,
                k4,
                max_fov,
                out_rays,
            ],
            device=device,
        )

        return out_rays

    @staticmethod
    def compute_camera_rays_fisheye_ftheta(
        width: int,
        height: int,
        optical_center_x: float,
        optical_center_y: float,
        *,
        image_width: float | None = None,
        image_height: float | None = None,
        nominal_width: float | None = None,
        nominal_height: float | None = None,
        k0: float = 0.0,
        k1: float = 1.0,
        k2: float = 0.0,
        k3: float = 0.0,
        k4: float = 0.0,
        max_fov: float = 2.0 * math.pi,
        out_rays: wp.array3d[wp.vec3f] | None = None,
        device: Devicelike = None,
    ) -> wp.array3d[wp.vec3f]:
        """Compute camera-space rays for one F-theta fisheye camera."""
        from .sensor_camera_render import camera_utils  # noqa: PLC0415

        width, height, out_rays, device = _validate_camera_ray_output(width, height, out_rays, device)
        image_width = _resolve_fisheye_image_size("width", image_width, nominal_width, width)
        image_height = _resolve_fisheye_image_size("height", image_height, nominal_height, height)

        wp.launch(
            kernel=camera_utils.compute_camera_rays_fisheye_ftheta_kernel,
            dim=(height, width),
            inputs=[
                width,
                height,
                image_width,
                image_height,
                optical_center_x,
                optical_center_y,
                k0,
                k1,
                k2,
                k3,
                k4,
                max_fov,
                out_rays,
            ],
            device=device,
        )

        return out_rays

    @staticmethod
    def compute_camera_rays_fisheye_kannala_brandt(
        width: int,
        height: int,
        optical_center_x: float,
        optical_center_y: float,
        *,
        image_width: float | None = None,
        image_height: float | None = None,
        nominal_width: float | None = None,
        nominal_height: float | None = None,
        k0: float = 1.0,
        k1: float = 0.0,
        k2: float = 0.0,
        k3: float = 0.0,
        max_fov: float = 2.0 * math.pi,
        out_rays: wp.array3d[wp.vec3f] | None = None,
        device: Devicelike = None,
    ) -> wp.array3d[wp.vec3f]:
        """Compute camera-space rays for one Kannala-Brandt fisheye camera."""
        from .sensor_camera_render import camera_utils  # noqa: PLC0415

        width, height, out_rays, device = _validate_camera_ray_output(width, height, out_rays, device)
        image_width = _resolve_fisheye_image_size("width", image_width, nominal_width, width)
        image_height = _resolve_fisheye_image_size("height", image_height, nominal_height, height)

        wp.launch(
            kernel=camera_utils.compute_camera_rays_fisheye_kannala_brandt_kernel,
            dim=(height, width),
            inputs=[
                width,
                height,
                image_width,
                image_height,
                optical_center_x,
                optical_center_y,
                k0,
                k1,
                k2,
                k3,
                max_fov,
                out_rays,
            ],
            device=device,
        )

        return out_rays

    def _get_render_context(self) -> RenderContext:
        if self._render_context is None:
            raise RuntimeError("SensorCamera has no model; construct it with SensorCamera(model).")
        return self._render_context

    def create_default_light(self, enable_shadows: bool = True, direction: wp.vec3f | None = None) -> None:
        """Create a default directional light for the rendered scene.

        Args:
            enable_shadows: Enable shadow casting for this light.
            direction: Normalized light direction. If ``None``, defaults to
                normalized ``(-1, 1, -1)``.
        """
        self._get_render_context().create_default_light(enable_shadows=enable_shadows, direction=direction)

    def assign_checkerboard_material(
        self,
        *,
        shape_indices: Sequence[int] | np.ndarray,
        resolution: int = 64,
        checker_size: int = 32,
    ) -> None:
        """Assign a gray checkerboard texture material to selected shapes.

        Args:
            shape_indices: Shape indices that should use the checkerboard texture.
            resolution: Texture resolution [px] (square texture).
            checker_size: Size of each checkerboard square [px].
        """
        self._get_render_context().assign_checkerboard_material(
            shape_indices=shape_indices, resolution=resolution, checker_size=checker_size
        )

    def sync_transforms(self, state: State) -> None:
        """Synchronize render-only state (deformable triangle meshes) from *state*.

        Call this before :meth:`update` on any frame whose geometry changed; the
        ray tracer reads the synchronized mesh points. Rigid-only scenes need no
        sync (this is a no-op for them). Shape and particle BVHs are refit
        separately via :meth:`~newton.Model.bvh_refit_shapes` and
        :meth:`~newton.Model.bvh_refit_particles`.

        Args:
            state: Current simulation state with particle positions.
        """
        self._get_render_context().update(state)

    @staticmethod
    def _validate_render_array(name: str, array: Any, dtype: Any, device: wp.Device) -> None:
        if not isinstance(array, wp.array):
            raise TypeError(f"{name} must be a Warp array, got {type(array).__name__}.")
        if array.dtype != dtype:
            raise ValueError(f"{name} must have dtype {dtype}, got {array.dtype}.")
        if array.device != device:
            raise RuntimeError(f"{name} must be on the model device ({device}), got {array.device}.")

    def update(
        self,
        state: State,
        camera_transforms: wp.array[wp.transformf],
        camera_rays: wp.array3d[wp.vec3f],
        *,
        color_image: wp.array3d[wp.uint32] | None = None,
        depth_image: wp.array3d[wp.float32] | None = None,
        forward_depth_image: wp.array3d[wp.float32] | None = None,
        shape_index_image: wp.array3d[wp.uint32] | None = None,
        normal_image: wp.array3d[wp.vec3f] | None = None,
        albedo_image: wp.array3d[wp.uint32] | None = None,
        hdr_color_image: wp.array3d[wp.vec3f] | None = None,
        world_indices: wp.array[wp.int32] | None = None,
        clear_data: ClearData | None = None,
        render_config: RenderConfig | None = None,
        kernel_block_dim: int = 64,
    ) -> None:
        """Render this camera sensor.

        The number of views is inferred from ``camera_transforms.shape[0]``; all
        non-``None`` output arrays must have shape ``(view_count, height, width)``
        matching the ``camera_rays`` image dimensions.

        Before calling this on any frame whose geometry moved, call
        :meth:`sync_transforms` to synchronize render-only state (deformable
        triangle meshes) from *state*, and refit the model's shape and particle
        BVHs with :meth:`~newton.Model.bvh_refit_shapes` and
        :meth:`~newton.Model.bvh_refit_particles` (both are built initially by
        :meth:`~newton.ModelBuilder.finalize`); otherwise the render reads stale
        points and bounds.

        Args:
            state: Simulation state with body and particle transforms.
            camera_transforms: World-space camera transform per view [m, rad],
                shape ``(view_count,)`` of ``transformf``, on the model device.
            camera_rays: Camera-space ray origins and directions, shape
                ``(height, width, 2)`` of ``vec3f``, on the model device.
            color_image: Output RGBA color buffer (packed ``uint32``).
            depth_image: Output depth buffer [m].
            forward_depth_image: Output forward-depth buffer [m].
            shape_index_image: Output shape-index buffer.
            normal_image: Output world-space surface normals.
            albedo_image: Output albedo buffer (packed ``uint32``).
            hdr_color_image: Output linear HDR color buffer.
            world_indices: Optional per-view world selector, shape
                ``(view_count,)``. Defaults to the identity mapping (view ``i``
                renders world ``i``). A non-negative entry is the world index
                rendered for that view; a negative
                :class:`~newton.sensors.SensorCamera.WorldRenderFlag` sentinel
                disables it (``DISABLE_CLEAR`` clears the outputs,
                ``DISABLE_PRESERVE`` leaves them unchanged).
            clear_data: Clear values for this call. Defaults to
                :attr:`default_clear_data`.
            render_config: Render settings for this call. Defaults to
                :attr:`default_render_config`.
            kernel_block_dim: Thread block dimension forwarded to ``wp.launch``.
        """
        render_context = self._get_render_context()
        model = render_context.model

        self._validate_render_array("camera_transforms", camera_transforms, wp.transformf, model.device)
        if camera_transforms.ndim != 1 or camera_transforms.shape[0] <= 0:
            raise ValueError(f"camera_transforms must have shape (view_count,), got {tuple(camera_transforms.shape)}.")

        self._validate_render_array("camera_rays", camera_rays, wp.vec3f, model.device)
        if camera_rays.ndim != 3 or camera_rays.shape[0] <= 0 or camera_rays.shape[1] <= 0 or camera_rays.shape[2] != 2:
            raise ValueError(f"camera_rays must have shape (height, width, 2), got {tuple(camera_rays.shape)}.")

        view_count = int(camera_transforms.shape[0])
        # Without an explicit mapping the renderer uses ``world_index == view_index``
        # (no array is built), which is only valid when there are at least as many
        # worlds as views; otherwise it would index past the model's per-world data.
        if world_indices is None and view_count > int(model.world_count):
            raise ValueError(
                f"view_count ({view_count}) exceeds model.world_count ({int(model.world_count)}); "
                "pass an explicit world_indices mapping for the extra views."
            )

        with wp.ScopedTimer("Newton::SensorCamera::update", active=PROFILE_ENABLED, use_nvtx=True, synchronize=True):
            render_context.render(
                state,
                camera_transforms=camera_transforms,
                camera_rays=camera_rays,
                world_indices=world_indices,
                color_image=color_image,
                hdr_color_image=hdr_color_image,
                depth_image=depth_image,
                forward_depth_image=forward_depth_image,
                shape_index_image=shape_index_image,
                normal_image=normal_image,
                albedo_image=albedo_image,
                clear_data=clear_data if clear_data is not None else self.default_clear_data,
                config=render_config if render_config is not None else self.default_render_config,
                kernel_block_dim=kernel_block_dim,
            )
