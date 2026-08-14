# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

import numpy as np
import warp as wp

from ..core.types import Devicelike
from ..render.types import (
    ClearData,
    RenderConfig,
)

if TYPE_CHECKING:
    from ..render.render_context import RenderContext
    from ..sim.state import State


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
    """Raytraced camera sensor rendered through a shared :class:`RenderContext`.

    A camera sensor holds the camera-space rays (the image model) and renders
    one view per entry of the ``camera_transforms`` array passed to
    :meth:`update`. The caller owns those world-space camera poses; the sensor
    only owns the ray bundle, the per-view :attr:`world_indices` selector, and
    the render settings.
    """

    def __init__(
        self,
        rays: wp.array | np.ndarray,
        render_context: RenderContext | None = None,
        *,
        view_count: int | None = None,
    ):
        """Construct a camera sensor from a ray bundle.

        Args:
            rays: Camera-space ray origins and directions, shape
                ``(height, width, 2)`` of ``vec3f``.
            render_context: Render context used by :meth:`update`. It is
                read-only after construction. When provided, the rays are moved
                to the model device and ``view_count`` defaults to the model's
                world count.
            view_count: Number of render views, i.e. the required leading
                dimension of the ``camera_transforms`` passed to :meth:`update`
                and of the output images. Defaults to the render context's
                ``model.world_count``.
        """
        self.rays = self._coerce_rays(rays)
        self._render_context = render_context

        self.world_indices: wp.array[wp.int32] | None = None
        """Per-view world selector, shape ``(view_count,)``. A non-negative entry is the model world index rendered
        for that view; a negative :class:`~newton.WorldRenderFlag` sentinel disables it. Defaults to the identity
        mapping (view ``i`` renders world ``i``)."""

        self.clear_data: ClearData | None = ClearData()
        """Values used to clear output images before rendering."""
        self.render_config: RenderConfig | None = RenderConfig()
        """Render settings used by :meth:`update`."""

        if view_count is not None and int(view_count) <= 0:
            raise ValueError(f"view_count must be positive, got {view_count}.")

        if render_context is not None:
            device = wp.get_device(render_context.model.device)
            if self.rays.device != device:
                self.rays = wp.clone(self.rays, device=device)
            default_view_count = int(render_context.model.world_count)
        else:
            default_view_count = 0

        self._view_count = int(view_count) if view_count is not None else default_view_count
        if self._view_count > 0:
            self.world_indices = wp.array(
                np.arange(self._view_count, dtype=np.int32), dtype=wp.int32, device=self.rays.device
            )

    @property
    def render_context(self) -> RenderContext | None:
        """Render context used by :meth:`update` (read-only; set in the constructor)."""
        return self._render_context

    @property
    def width(self) -> int:
        """Image width [px]."""
        return int(self.rays.shape[1])

    @property
    def height(self) -> int:
        """Image height [px]."""
        return int(self.rays.shape[0])

    @property
    def view_count(self) -> int:
        """Number of render views produced by this camera sensor."""
        return self._view_count

    @property
    def device(self) -> wp.Device:
        """Device storing this camera sensor's rays."""
        return self.rays.device

    @property
    def utils(self):
        """Renderer utility helpers for this camera sensor."""
        self._ensure_ready()

        from ..render import Utils  # noqa: PLC0415

        return Utils(view_count=self.view_count, device=self.device)

    def _ensure_ready(self) -> None:
        if self._view_count <= 0:
            raise RuntimeError(
                "SensorCamera has no views; construct it with a render context or an explicit view_count."
            )

    def create_image_output(self, dtype: Any) -> wp.array:
        """Create an output image array with shape ``(view_count, height, width)``."""
        self._ensure_ready()
        return wp.zeros((self.view_count, self.height, self.width), dtype=dtype, device=self.device)

    def create_color_image_output(self) -> wp.array3d[wp.uint32]:
        """Create a color output array for this camera sensor."""
        return self.create_image_output(wp.uint32)

    def create_depth_image_output(self) -> wp.array3d[wp.float32]:
        """Create a depth output array for this camera sensor [m]."""
        return self.create_image_output(wp.float32)

    def create_forward_depth_image_output(self) -> wp.array3d[wp.float32]:
        """Create a forward-depth output array for this camera sensor [m]."""
        return self.create_depth_image_output()

    def create_shape_index_image_output(self) -> wp.array3d[wp.uint32]:
        """Create a shape-index output array for this camera sensor."""
        return self.create_image_output(wp.uint32)

    def create_normal_image_output(self) -> wp.array3d[wp.vec3f]:
        """Create a normal output array for this camera sensor."""
        return self.create_image_output(wp.vec3f)

    def create_albedo_image_output(self) -> wp.array3d[wp.uint32]:
        """Create an albedo output array for this camera sensor."""
        return self.create_image_output(wp.uint32)

    def create_hdr_color_image_output(self) -> wp.array3d[wp.vec3f]:
        """Create a linear HDR color output array for this camera sensor."""
        return self.create_image_output(wp.vec3f)

    def _coerce_rays(self, rays: wp.array | np.ndarray) -> wp.array3d[wp.vec3f]:
        if isinstance(rays, np.ndarray):
            rays = wp.array(np.ascontiguousarray(rays, dtype=np.float32), dtype=wp.vec3f)

        if not isinstance(rays, wp.array):
            raise TypeError(f"rays must be a Warp or NumPy array, got {type(rays).__name__}")
        if rays.dtype != wp.vec3f:
            raise ValueError(f"SensorCamera rays must have dtype vec3f, got {rays.dtype}")

        if rays.ndim != 3 or rays.shape[0] <= 0 or rays.shape[1] <= 0 or rays.shape[2] != 2:
            raise ValueError(f"SensorCamera rays must have shape (height, width, 2), got {rays.shape}")

        return rays

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
        """Compute camera-space rays for one pinhole camera."""
        from ..render import camera_utils  # noqa: PLC0415

        width, height, out_rays, device = _validate_camera_ray_output(width, height, out_rays, device)

        use_aperture = focal_length is not None or horizontal_aperture is not None or vertical_aperture is not None
        if use_aperture:
            if camera_fov is not None:
                raise ValueError("camera_fov cannot be provided with aperture parameters.")
            if focal_length is None or horizontal_aperture is None or vertical_aperture is None:
                raise ValueError("focal_length, horizontal_aperture, and vertical_aperture must be provided together.")

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
        from ..render import camera_utils  # noqa: PLC0415

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
        from ..render import camera_utils  # noqa: PLC0415

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
        from ..render import camera_utils  # noqa: PLC0415

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
        from ..render import camera_utils  # noqa: PLC0415

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
        if self.render_context is None:
            raise RuntimeError("SensorCamera.update() requires a RenderContext.")
        return self.render_context

    def update(
        self,
        state: State,
        camera_transforms: wp.array[wp.transformf],
        *,
        color_image: wp.array3d[wp.uint32] | None = None,
        depth_image: wp.array3d[wp.float32] | None = None,
        forward_depth_image: wp.array3d[wp.float32] | None = None,
        shape_index_image: wp.array3d[wp.uint32] | None = None,
        normal_image: wp.array3d[wp.vec3f] | None = None,
        albedo_image: wp.array3d[wp.uint32] | None = None,
        hdr_color_image: wp.array3d[wp.vec3f] | None = None,
        world_indices: wp.array[wp.int32] | None = None,
        kernel_block_dim: int = 64,
    ) -> None:
        """Render this camera sensor from the given camera transforms.

        Output arrays must have shape ``(view_count, height, width)``.

        Args:
            state: Simulation state with body and particle transforms.
            camera_transforms: World-space camera transform per view [m, rad],
                shape ``(view_count,)`` of ``transformf``, on the model device.
            color_image: Output RGBA color buffer (packed ``uint32``).
            depth_image: Output depth buffer [m].
            forward_depth_image: Output forward-depth buffer [m].
            shape_index_image: Output shape-index buffer.
            normal_image: Output world-space surface normals.
            albedo_image: Output albedo buffer (packed ``uint32``).
            hdr_color_image: Output linear HDR color buffer.
            world_indices: Optional per-view world selector, shape
                ``(view_count,)``. Defaults to :attr:`world_indices`. A
                non-negative entry is the world index rendered for that view; a
                negative :class:`~newton.WorldRenderFlag` sentinel disables it
                (``DISABLE_CLEAR`` clears the outputs, ``DISABLE_PRESERVE``
                leaves them unchanged).
            kernel_block_dim: Thread block dimension forwarded to ``wp.launch``.
        """
        render_context = self._get_render_context()
        model = render_context.model
        if camera_transforms.dtype != wp.transformf:
            raise ValueError(f"camera_transforms must have dtype transformf, got {camera_transforms.dtype}.")
        if camera_transforms.shape != (self._view_count,):
            raise ValueError(
                f"camera_transforms must have shape ({self._view_count},), got {tuple(camera_transforms.shape)}."
            )
        if self.rays.device != model.device:
            raise RuntimeError(
                "SensorCamera rays are not on the model device; construct it with this model's render context."
            )
        if camera_transforms.device != model.device:
            raise RuntimeError(
                f"camera_transforms must be on the model device ({model.device}), got {camera_transforms.device}."
            )

        if world_indices is None:
            world_indices = self.world_indices

        render_context.render(
            state,
            camera_transforms=camera_transforms,
            camera_rays=self.rays,
            world_indices=world_indices,
            color_image=color_image,
            hdr_color_image=hdr_color_image,
            depth_image=depth_image,
            forward_depth_image=forward_depth_image,
            shape_index_image=shape_index_image,
            normal_image=normal_image,
            albedo_image=albedo_image,
            clear_data=self.clear_data,
            config=self.render_config,
            kernel_block_dim=kernel_block_dim,
        )
