# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
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
    AntiAliasing,
    ClearData,
    GaussianRenderMode,
    RenderConfig,
    RenderOrder,
    TextureProjectionMode,
    WorldRenderFlag,
)

if TYPE_CHECKING:
    from ..sim.model import Model
    from ..sim.state import State

# Enable NVTX ranges / timing around SensorCamera.update() when NEWTON_PROFILE is set.
PROFILE_ENABLED = os.environ.get("NEWTON_PROFILE", "0") != "0"


def _validate_camera_ray_output(
    width: int,
    height: int,
    out_rays: wp.array4d[wp.vec3f] | None,
    device: Devicelike = None,
    sample_count: int = 1,
) -> tuple[int, int, int, wp.array4d[wp.vec3f], wp.Device]:
    width = int(width)
    height = int(height)
    if width <= 0 or height <= 0:
        raise ValueError("width and height must be positive.")

    sample_count = int(sample_count)
    if sample_count < 1:
        raise ValueError("sample_count must be positive.")

    expected_shape = (height, width, sample_count, 2)
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

    return width, height, sample_count, out_rays, out_rays.device


class SensorCamera:
    """Raytraced camera sensor that renders a Newton model.

    A camera sensor owns an internal renderer built for a model plus the default
    render settings (:attr:`default_clear_data`, :attr:`default_render_config`)
    used when :meth:`update` is not given per-call overrides. The caller supplies
    the camera-space rays, the world-space per-view camera transforms, and the
    output image buffers to :meth:`update`; the number of views is inferred from
    the leading dimension of the camera transforms.

    Camera frame convention: each camera looks along its local ``-Z`` axis, with
    ``+Y`` up and ``+X`` right (the USD/OpenGL convention). The per-view
    ``camera_transforms`` place that camera frame in world space, and the
    ``camera_rays`` bundle stores per-pixel origins in ``[..., 0]`` and
    directions in ``[..., 1]``, both expressed in camera space.

    Deformable triangle surfaces use :attr:`Model.tri_color` as sRGB albedo,
    like shape colors, and fall back to white when triangle colors are
    unavailable. Set :attr:`RenderConfig.enable_backface_culling` to ``False``
    to render both sides with camera-facing normals.

    The render configuration types are exposed as nested attributes (e.g.
    ``SensorCamera.RenderConfig``, ``SensorCamera.ClearData``,
    ``SensorCamera.WorldRenderFlag``); they are not part of the top-level
    ``newton`` namespace.

    Example::

        import warp as wp
        import newton
        from newton.sensors import SensorCamera

        camera = SensorCamera(model)
        camera.create_default_light()

        width, height = 640, 480
        camera_rays = SensorCamera.compute_camera_rays_pinhole(width, height, camera_fov=1.0, device=model.device)
        camera_transforms = wp.array([wp.transform_identity()], dtype=wp.transformf, device=model.device)
        color = camera.create_color_image_output(camera_transforms.shape[0], width, height)

        camera.update(state, camera_transforms, camera_rays, color_image=color)
    """

    AntiAliasing = AntiAliasing
    ClearData = ClearData
    GaussianRenderMode = GaussianRenderMode
    RenderConfig = RenderConfig
    RenderOrder = RenderOrder
    TextureProjectionMode = TextureProjectionMode
    Utils = Utils
    WorldRenderFlag = WorldRenderFlag

    def __init__(
        self,
        model: Model,
        *,
        default_clear_data: ClearData | None = None,
        default_render_config: RenderConfig | None = None,
        load_textures: bool = True,
    ):
        """Construct a camera sensor for a model.

        Args:
            model: Newton simulation model to render. The sensor builds and owns
                its internal renderer for this model.
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

        from .sensor_camera_render.render_context import RenderContext  # noqa: PLC0415

        self._render_context = RenderContext(model, load_textures=load_textures)

    @property
    def device(self) -> wp.Device:
        """Device of the model this sensor renders."""
        return self._render_context.model.device

    def create_image_output(self, view_count: int, width: int, height: int, dtype: Any) -> wp.array[Any]:
        """Create a zeroed output image array on the model device.

        Args:
            view_count: Number of views (the array's leading dimension).
            width: Image width [px].
            height: Image height [px].
            dtype: Warp element type for the array (e.g. ``wp.uint32``,
                ``wp.float32``, ``wp.vec3f``).

        Returns:
            Zeroed array of shape ``(view_count, height, width)`` and dtype
            *dtype*, on the model device.
        """
        return wp.zeros((int(view_count), int(height), int(width)), dtype=dtype, device=self.device)

    def create_color_image_output(self, view_count: int, width: int, height: int) -> wp.array3d[wp.uint32]:
        """Create an RGBA color output array (packed ``uint32``).

        Args:
            view_count: Number of views (the array's leading dimension).
            width: Image width [px].
            height: Image height [px].

        Returns:
            Zeroed array of shape ``(view_count, height, width)``, dtype ``uint32``.
        """
        return self.create_image_output(view_count, width, height, wp.uint32)

    def create_depth_image_output(self, view_count: int, width: int, height: int) -> wp.array3d[wp.float32]:
        """Create a ray-distance depth output array [m].

        Args:
            view_count: Number of views (the array's leading dimension).
            width: Image width [px].
            height: Image height [px].

        Returns:
            Zeroed array of shape ``(view_count, height, width)``, dtype ``float32``.
        """
        return self.create_image_output(view_count, width, height, wp.float32)

    def create_forward_depth_image_output(self, view_count: int, width: int, height: int) -> wp.array3d[wp.float32]:
        """Create a forward (planar) depth output array [m].

        Args:
            view_count: Number of views (the array's leading dimension).
            width: Image width [px].
            height: Image height [px].

        Returns:
            Zeroed array of shape ``(view_count, height, width)``, dtype ``float32``.
        """
        return self.create_depth_image_output(view_count, width, height)

    def create_shape_index_image_output(self, view_count: int, width: int, height: int) -> wp.array3d[wp.uint32]:
        """Create a shape-index output array.

        Args:
            view_count: Number of views (the array's leading dimension).
            width: Image width [px].
            height: Image height [px].

        Returns:
            Zeroed array of shape ``(view_count, height, width)``, dtype ``uint32``.
        """
        return self.create_image_output(view_count, width, height, wp.uint32)

    def create_normal_image_output(self, view_count: int, width: int, height: int) -> wp.array3d[wp.vec3f]:
        """Create a world-space surface-normal output array (``vec3f``).

        Args:
            view_count: Number of views (the array's leading dimension).
            width: Image width [px].
            height: Image height [px].

        Returns:
            Zeroed array of shape ``(view_count, height, width)``, dtype ``vec3f``.
        """
        return self.create_image_output(view_count, width, height, wp.vec3f)

    def create_albedo_image_output(self, view_count: int, width: int, height: int) -> wp.array3d[wp.uint32]:
        """Create an RGBA albedo output array (packed ``uint32``).

        Args:
            view_count: Number of views (the array's leading dimension).
            width: Image width [px].
            height: Image height [px].

        Returns:
            Zeroed array of shape ``(view_count, height, width)``, dtype ``uint32``.
        """
        return self.create_image_output(view_count, width, height, wp.uint32)

    def create_hdr_color_image_output(self, view_count: int, width: int, height: int) -> wp.array3d[wp.vec3f]:
        """Create a linear HDR color output array (``vec3f``).

        Args:
            view_count: Number of views (the array's leading dimension).
            width: Image width [px].
            height: Image height [px].

        Returns:
            Zeroed array of shape ``(view_count, height, width)``, dtype ``vec3f``.
        """
        return self.create_image_output(view_count, width, height, wp.vec3f)

    @staticmethod
    def compute_camera_rays_pinhole(
        width: int,
        height: int,
        *,
        camera_fov: float | None = None,
        focal_length: float | None = None,
        horizontal_aperture: float | None = None,
        vertical_aperture: float | None = None,
        horizontal_aperture_offset: float = 0.0,
        vertical_aperture_offset: float = 0.0,
        sample_count: int = 1,
        out_rays: wp.array4d[wp.vec3f] | None = None,
        device: Devicelike = None,
    ) -> wp.array4d[wp.vec3f]:
        """Compute camera-space rays for one pinhole camera.

        Provide either ``camera_fov`` or the aperture triple (``focal_length``,
        ``horizontal_aperture``, ``vertical_aperture``), not both. The focal
        length and apertures share consistent units; only their ratios affect
        the ray directions.

        Args:
            width: Image width [px].
            height: Image height [px].
            camera_fov: Vertical field of view [rad], in ``(0, pi)``. Mutually
                exclusive with the aperture parameters.
            focal_length: Lens focal length; must be positive.
            horizontal_aperture: Horizontal sensor aperture; must be positive.
            vertical_aperture: Vertical sensor aperture; must be positive.
            horizontal_aperture_offset: Horizontal principal-point offset.
            vertical_aperture_offset: Vertical principal-point offset.
            sample_count: Number of rays to generate per pixel. Use one for a
                centered ray per pixel.
            out_rays: Optional output buffer, shape
                ``(height, width, sample_count, 2)`` of ``vec3f``.
            device: Device for the ray bundle. Defaults to the current Warp device.

        Returns:
            Ray origins and directions, shape
            ``(height, width, sample_count, 2)`` of ``vec3f``.
        """
        from .sensor_camera_render import camera_utils  # noqa: PLC0415

        width, height, sample_count, out_rays, device = _validate_camera_ray_output(
            width, height, out_rays, device, sample_count=sample_count
        )
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
                dim=(height, width, sample_count),
                inputs=[
                    width,
                    height,
                    float(focal_length),
                    float(horizontal_aperture),
                    float(vertical_aperture),
                    float(horizontal_aperture_offset),
                    float(vertical_aperture_offset),
                    sample_count,
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
            dim=(height, width, sample_count),
            inputs=[
                width,
                height,
                float(camera_fov),
                sample_count,
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
        sample_count: int = 1,
        out_rays: wp.array4d[wp.vec3f] | None = None,
        device: Devicelike = None,
    ) -> wp.array4d[wp.vec3f]:
        """Compute camera-space rays for one USD pinhole camera.

        Reads the perspective intrinsics (focal length and aperture) from a USD
        camera and builds the matching pinhole ray bundle.

        Args:
            width: Image width [px].
            height: Image height [px].
            camera: A ``UsdGeom.Camera`` or a ``Usd.Prim`` that is a camera. Its
                projection must be ``perspective``.
            time: USD time to sample the camera attributes at (a
                ``Usd.TimeCode`` or a frame number). If ``None``, the default
                time code is used.
            sample_count: Number of rays per pixel. Use one for a centered ray.
            out_rays: Optional output buffer, shape ``(height, width, sample_count, 2)`` of
                ``vec3f``. If ``None``, a new one is allocated.
            device: Device for the ray bundle. Defaults to the current Warp
                device.

        Returns:
            Ray origins (``[..., 0]``) and directions (``[..., 1]``), shape
            ``(height, width, sample_count, 2)`` of ``vec3f``.
        """
        from .sensor_camera_render import camera_utils  # noqa: PLC0415

        width, height, sample_count, out_rays, device = _validate_camera_ray_output(
            width, height, out_rays, device, sample_count=sample_count
        )
        camera_utils.compute_camera_rays_usd_pinhole(
            width,
            height,
            camera,
            device=device,
            time=time,
            sample_count=sample_count,
            out_rays=out_rays,
        )
        return out_rays

    @staticmethod
    def compute_camera_rays_pinhole_opencv(
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
        k5: float = 0.0,
        k6: float = 0.0,
        p1: float = 0.0,
        p2: float = 0.0,
        s1: float = 0.0,
        s2: float = 0.0,
        s3: float = 0.0,
        s4: float = 0.0,
        sample_count: int = 1,
        out_rays: wp.array4d[wp.vec3f] | None = None,
        device: Devicelike = None,
    ) -> wp.array4d[wp.vec3f]:
        """Compute camera-space rays for one OpenCV pinhole camera.

        Inverts OpenCV's rational radial, tangential, and thin-prism distortion
        model with damped Newton iteration. The four- and five-coefficient
        variants are represented by leaving unused coefficients at zero. Pixels
        whose inverse cannot be verified within the solver tolerance receive a
        zero direction.

        Args:
            width: Output image width [px].
            height: Output image height [px].
            fx: Horizontal focal length [px].
            fy: Vertical focal length [px].
            cx: Principal point x-coordinate [px].
            cy: Principal point y-coordinate [px].
            image_width: Calibration image width [px]. If ``None``, uses *width*.
            image_height: Calibration image height [px]. If ``None``, uses
                *height*.
            k1: First numerator radial distortion coefficient.
            k2: Second numerator radial distortion coefficient.
            k3: Third numerator radial distortion coefficient.
            k4: First denominator radial distortion coefficient.
            k5: Second denominator radial distortion coefficient.
            k6: Third denominator radial distortion coefficient.
            p1: First tangential distortion coefficient.
            p2: Second tangential distortion coefficient.
            s1: First thin-prism distortion coefficient.
            s2: Second thin-prism distortion coefficient.
            s3: Third thin-prism distortion coefficient.
            s4: Fourth thin-prism distortion coefficient.
            sample_count: Number of rays per pixel. Use one for a centered ray.
            out_rays: Optional output buffer, shape ``(height, width, sample_count, 2)`` of
                ``vec3f``. If ``None``, a new one is allocated.
            device: Device for the ray bundle. Defaults to the current Warp
                device.

        Returns:
            Ray origins (``[..., 0]``) and directions (``[..., 1]``), shape
            ``(height, width, sample_count, 2)`` of ``vec3f``.

        Raises:
            ValueError: If any focal length or calibration image dimension is
                non-positive, or if any calibration value is non-finite.
        """
        from .sensor_camera_render import camera_utils  # noqa: PLC0415

        width, height, sample_count, out_rays, device = _validate_camera_ray_output(
            width, height, out_rays, device, sample_count=sample_count
        )
        image_width = float(width) if image_width is None else float(image_width)
        image_height = float(height) if image_height is None else float(image_height)
        if not (math.isfinite(fx) and math.isfinite(fy) and fx > 0.0 and fy > 0.0):
            raise ValueError("fx and fy must be finite and positive.")
        if not (
            math.isfinite(image_width) and math.isfinite(image_height) and image_width > 0.0 and image_height > 0.0
        ):
            raise ValueError("image_width and image_height must be finite and positive.")
        if not all(math.isfinite(value) for value in (cx, cy, k1, k2, k3, k4, k5, k6, p1, p2, s1, s2, s3, s4)):
            raise ValueError("cx, cy, and distortion coefficients must be finite.")

        wp.launch(
            kernel=camera_utils.compute_camera_rays_pinhole_opencv_kernel,
            dim=(height, width, sample_count),
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
                k5,
                k6,
                p1,
                p2,
                s1,
                s2,
                s3,
                s4,
                sample_count,
                out_rays,
            ],
            device=device,
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
        sample_count: int = 1,
        out_rays: wp.array4d[wp.vec3f] | None = None,
        device: Devicelike = None,
    ) -> wp.array4d[wp.vec3f]:
        """Compute camera-space rays for one OpenCV fisheye camera.

        Inverts the OpenCV fisheye radius polynomial
        ``r = theta (1 + k1 theta^2 + k2 theta^4 + k3 theta^6 + k4 theta^8)``,
        which must be monotonic over ``[0, min(max_fov / 2, pi)]``.

        Args:
            width: Output image width [px].
            height: Output image height [px].
            fx: Horizontal focal length [px].
            fy: Vertical focal length [px].
            cx: Principal point x-coordinate [px].
            cy: Principal point y-coordinate [px].
            image_width: Calibration image width [px]. If ``None``, uses *width*.
            image_height: Calibration image height [px]. If ``None``, uses
                *height*.
            k1: First OpenCV fisheye distortion coefficient.
            k2: Second OpenCV fisheye distortion coefficient.
            k3: Third OpenCV fisheye distortion coefficient.
            k4: Fourth OpenCV fisheye distortion coefficient.
            max_fov: Maximum field of view [rad]. Pixels whose undistorted angle
                exceeds ``max_fov / 2`` receive a zero ray.
            sample_count: Number of rays per pixel. Use one for a centered ray.
            out_rays: Optional output buffer, shape ``(height, width, sample_count, 2)`` of
                ``vec3f``. If ``None``, a new one is allocated.
            device: Device for the ray bundle. Defaults to the current Warp
                device.

        Returns:
            Ray origins (``[..., 0]``) and directions (``[..., 1]``), shape
            ``(height, width, sample_count, 2)`` of ``vec3f``.
        """
        from .sensor_camera_render import camera_utils  # noqa: PLC0415

        width, height, sample_count, out_rays, device = _validate_camera_ray_output(
            width, height, out_rays, device, sample_count=sample_count
        )
        image_width = float(width) if image_width is None else float(image_width)
        image_height = float(height) if image_height is None else float(image_height)

        wp.launch(
            kernel=camera_utils.compute_camera_rays_fisheye_opencv_kernel,
            dim=(height, width, sample_count),
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
                sample_count,
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
        k0: float = 0.0,
        k1: float = 1.0,
        k2: float = 0.0,
        k3: float = 0.0,
        k4: float = 0.0,
        max_fov: float = 2.0 * math.pi,
        sample_count: int = 1,
        out_rays: wp.array4d[wp.vec3f] | None = None,
        device: Devicelike = None,
    ) -> wp.array4d[wp.vec3f]:
        """Compute camera-space rays for one F-theta fisheye camera.

        Inverts the F-theta radius polynomial
        ``r = k0 + k1 theta + k2 theta^2 + k3 theta^3 + k4 theta^4``, which must
        be monotonic over ``[0, min(max_fov / 2, pi)]``.

        Args:
            width: Output image width [px].
            height: Output image height [px].
            optical_center_x: Optical center x-coordinate [px].
            optical_center_y: Optical center y-coordinate [px].
            image_width: Calibration image width [px] (the F-theta nominal
                width). If ``None``, uses *width*.
            image_height: Calibration image height [px] (the F-theta nominal
                height). If ``None``, uses *height*.
            k0: Constant F-theta polynomial coefficient [px].
            k1: Linear F-theta polynomial coefficient [px/rad].
            k2: Quadratic F-theta polynomial coefficient [px/rad^2].
            k3: Cubic F-theta polynomial coefficient [px/rad^3].
            k4: Quartic F-theta polynomial coefficient [px/rad^4].
            max_fov: Maximum field of view [rad]. Pixels whose undistorted angle
                exceeds ``max_fov / 2`` receive a zero ray.
            sample_count: Number of rays per pixel. Use one for a centered ray.
            out_rays: Optional output buffer, shape ``(height, width, sample_count, 2)`` of
                ``vec3f``. If ``None``, a new one is allocated.
            device: Device for the ray bundle. Defaults to the current Warp
                device.

        Returns:
            Ray origins (``[..., 0]``) and directions (``[..., 1]``), shape
            ``(height, width, sample_count, 2)`` of ``vec3f``.
        """
        from .sensor_camera_render import camera_utils  # noqa: PLC0415

        width, height, sample_count, out_rays, device = _validate_camera_ray_output(
            width, height, out_rays, device, sample_count=sample_count
        )
        image_width = float(width) if image_width is None else float(image_width)
        image_height = float(height) if image_height is None else float(image_height)

        wp.launch(
            kernel=camera_utils.compute_camera_rays_fisheye_ftheta_kernel,
            dim=(height, width, sample_count),
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
                sample_count,
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
        k0: float = 1.0,
        k1: float = 0.0,
        k2: float = 0.0,
        k3: float = 0.0,
        max_fov: float = 2.0 * math.pi,
        sample_count: int = 1,
        out_rays: wp.array4d[wp.vec3f] | None = None,
        device: Devicelike = None,
    ) -> wp.array4d[wp.vec3f]:
        """Compute camera-space rays for one Kannala-Brandt fisheye camera.

        Inverts the Kannala-Brandt radius polynomial
        ``r = k0 theta + k1 theta^3 + k2 theta^5 + k3 theta^7``, which must be
        monotonic over ``[0, min(max_fov / 2, pi)]``.

        Args:
            width: Output image width [px].
            height: Output image height [px].
            optical_center_x: Optical center x-coordinate [px].
            optical_center_y: Optical center y-coordinate [px].
            image_width: Calibration image width [px]. If ``None``, uses *width*.
            image_height: Calibration image height [px]. If ``None``, uses
                *height*.
            k0: First Kannala-Brandt polynomial coefficient [px/rad].
            k1: Second Kannala-Brandt polynomial coefficient [px/rad^3].
            k2: Third Kannala-Brandt polynomial coefficient [px/rad^5].
            k3: Fourth Kannala-Brandt polynomial coefficient [px/rad^7].
            max_fov: Maximum field of view [rad]. Pixels whose undistorted angle
                exceeds ``max_fov / 2`` receive a zero ray.
            sample_count: Number of rays per pixel. Use one for a centered ray.
            out_rays: Optional output buffer, shape ``(height, width, sample_count, 2)`` of
                ``vec3f``. If ``None``, a new one is allocated.
            device: Device for the ray bundle. Defaults to the current Warp
                device.

        Returns:
            Ray origins (``[..., 0]``) and directions (``[..., 1]``), shape
            ``(height, width, sample_count, 2)`` of ``vec3f``.
        """
        from .sensor_camera_render import camera_utils  # noqa: PLC0415

        width, height, sample_count, out_rays, device = _validate_camera_ray_output(
            width, height, out_rays, device, sample_count=sample_count
        )
        image_width = float(width) if image_width is None else float(image_width)
        image_height = float(height) if image_height is None else float(image_height)

        wp.launch(
            kernel=camera_utils.compute_camera_rays_fisheye_kannala_brandt_kernel,
            dim=(height, width, sample_count),
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
                sample_count,
                out_rays,
            ],
            device=device,
        )

        return out_rays

    def compute_camera_transforms_usd(
        self,
        cameras: Any,
        *,
        time: Any | None = None,
        xform: Any | None = None,
    ) -> wp.array[wp.transformf]:
        """Read world-space camera transforms from one USD camera per view.

        Converts each camera pose from its USD stage up axis to the rendered
        model's up axis, then composes an optional scene *xform* (e.g. the
        transform passed to ``ModelBuilder.add_usd``). Use the result as the
        ``camera_transforms`` argument to :meth:`update`.

        Args:
            cameras: A single ``UsdGeom.Camera``/``Usd.Prim`` (one view) or a
                sequence of them (one per view, in view order).
            time: USD time to sample the camera poses at (a ``Usd.TimeCode`` or a
                frame number). If ``None``, the default time code is used.
            xform: Optional scene transform ``(pos, quat)`` applied on top of the
                up-axis conversion, matching the pose used when importing the
                stage.

        Returns:
            World-space camera transforms, shape ``(view_count,)`` of
            ``transformf``, on the model device.
        """
        from ..core import Axis  # noqa: PLC0415
        from .sensor_camera_render import camera_utils  # noqa: PLC0415

        model = self._render_context.model
        return camera_utils.compute_camera_transforms_usd(
            cameras,
            device=model.device,
            target_up_axis=Axis(int(model.up_axis)),
            time=time,
            xform=xform,
        )

    def create_default_light(self, *, enable_shadows: bool = True, direction: wp.vec3f | None = None) -> None:
        """Create a default directional light for the rendered scene.

        Args:
            enable_shadows: Enable shadow casting for this light. Shadows are
                only rendered when the render config also enables them, i.e.
                ``enable_shadows=True`` here **and**
                ``render_config.enable_shadows=True`` (the latter defaults to
                ``False``); both switches must be set.
            direction: Normalized light direction. If ``None``, defaults to
                normalized ``(-1, 1, -1)``.
        """
        self._render_context.create_default_light(enable_shadows=enable_shadows, direction=direction)

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
        self._render_context.assign_checkerboard_material(
            shape_indices=shape_indices, resolution=resolution, checker_size=checker_size
        )

    def sync_deformable_meshes(self, state: State) -> None:
        """Synchronize render-only deformable triangle-mesh points from *state*.

        :meth:`update` calls this by default (pass ``sync_deformables=False`` to
        skip it when you already synced). Call it explicitly only when you opt
        out of the automatic sync. It syncs deformable mesh points (rigid-only
        scenes are a no-op) but does not touch transforms; shape and particle
        BVHs are refit separately via :meth:`~newton.Model.bvh_refit_shapes` and
        :meth:`~newton.Model.bvh_refit_particles`.

        Args:
            state: Current simulation state with particle positions.
        """
        self._render_context.update(state)

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
        camera_rays: wp.array4d[wp.vec3f],
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
        sync_deformables: bool = True,
    ) -> None:
        """Render this camera sensor.

        The number of views is inferred from ``camera_transforms.shape[0]``; all
        non-``None`` output arrays must have shape ``(view_count, height, width)``
        matching the ``camera_rays`` image dimensions.

        On any frame whose geometry moved, refit the model's shape and particle
        BVHs with :meth:`~newton.Model.bvh_refit_shapes` and
        :meth:`~newton.Model.bvh_refit_particles` (both are built initially by
        :meth:`~newton.ModelBuilder.finalize`) before calling this; otherwise the
        render reads stale bounds. Deformable triangle-mesh points are synced from
        *state* automatically (see ``sync_deformables``).

        The camera looks along its local ``-Z`` axis with ``+Y`` up and ``+X``
        right (the USD/OpenGL convention); ``camera_transforms`` place that frame
        in world space. ``camera_rays[..., 0]`` are per-sample ray origins and
        ``camera_rays[..., 1]`` are ray directions, both in camera space.

        Args:
            state: Simulation state with body and particle transforms.
            camera_transforms: World-space camera transform per view [m, rad],
                shape ``(view_count,)`` of ``transformf``, on the model device.
            camera_rays: Camera-space ray origins and directions, shape
                ``(height, width, sample_count, 2)`` of ``vec3f``, on the
                model device. Bundles with ``sample_count > 1`` require
                ``render_config.anti_aliasing`` to be :attr:`AntiAliasing.SSAA`
                or :attr:`AntiAliasing.MSAA`.
            color_image: Output RGBA color buffer (packed ``uint32``).
            depth_image: Output depth buffer [m].
            forward_depth_image: Output forward-depth buffer [m].
            shape_index_image: Output shape-index buffer.
            normal_image: Output world-space surface normals.
            albedo_image: Output albedo buffer (packed ``uint32``).
            hdr_color_image: Output linear HDR color buffer.
            world_indices: Optional per-view world selector, shape
                ``(view_count,)``. Defaults to the identity mapping (view ``i``
                renders world ``i``). A valid entry is a world index in
                ``[0, model.world_count)``. A
                :class:`~newton.sensors.SensorCamera.WorldRenderFlag` sentinel
                disables the view (``DISABLE_CLEAR`` clears the outputs,
                ``DISABLE_PRESERVE`` leaves them unchanged). Any other value,
                including ``-1`` (reserved for future global-world rendering) and
                indices ``>= model.world_count``, is treated as ``DISABLE_CLEAR``.
            clear_data: Clear values for this call. Defaults to
                :attr:`default_clear_data`.
            render_config: Render settings for this call. Defaults to
                :attr:`default_render_config`.
            sync_deformables: Sync deformable triangle-mesh points from *state*
                before rendering (a no-op for rigid-only scenes). Set ``False``
                if you already called :meth:`sync_deformable_meshes` this frame.
        """
        render_context = self._render_context
        model = render_context.model

        if sync_deformables:
            render_context.update(state)

        self._validate_render_array("camera_transforms", camera_transforms, wp.transformf, model.device)
        if camera_transforms.ndim != 1 or camera_transforms.shape[0] <= 0:
            raise ValueError(f"camera_transforms must have shape (view_count,), got {tuple(camera_transforms.shape)}.")

        self._validate_render_array("camera_rays", camera_rays, wp.vec3f, model.device)
        multisample_shape = camera_rays.ndim == 4 and camera_rays.shape[2] > 0 and camera_rays.shape[3] == 2
        if not multisample_shape or camera_rays.shape[0] <= 0 or camera_rays.shape[1] <= 0:
            raise ValueError(
                f"camera_rays must have shape (height, width, sample_count, 2), got {tuple(camera_rays.shape)}."
            )

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
            )
