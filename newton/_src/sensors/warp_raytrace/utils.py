# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math
import warnings
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import numpy as np
import warp as wp

from ...core import MAXVAL
from . import camera_utils
from .types import RenderConfig, RenderLightType, TextureData

if TYPE_CHECKING:
    from .render_context import RenderContext


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


@wp.kernel(enable_backward=False)
def flatten_color_image(
    color_image: wp.array4d[wp.uint32],
    buffer: wp.array3d[wp.uint8],
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
    normal_image: wp.array4d[wp.vec3f],
    buffer: wp.array3d[wp.uint8],
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
def find_depth_range(depth_image: wp.array4d[wp.float32], depth_range: wp.array[wp.float32]):
    world_id, camera_id, y, x = wp.tid()
    depth = depth_image[world_id, camera_id, y, x]
    if depth > 0:
        wp.atomic_min(depth_range, 0, depth)
        wp.atomic_max(depth_range, 1, depth)


@wp.kernel(enable_backward=False)
def flatten_depth_image(
    depth_image: wp.array4d[wp.float32],
    buffer: wp.array3d[wp.uint8],
    depth_range: wp.array[wp.float32],
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


@wp.kernel(enable_backward=False)
def convert_ray_depth_to_forward_depth_kernel(
    depth_image: wp.array4d[wp.float32],
    camera_rays: wp.array4d[wp.vec3f],
    camera_transforms: wp.array2d[wp.transformf],
    out_depth: wp.array4d[wp.float32],
):
    world_index, camera_index, py, px = wp.tid()

    ray_depth = depth_image[world_index, camera_index, py, px]
    camera_transform = camera_transforms[camera_index, world_index]
    camera_ray = camera_rays[camera_index, py, px, 1]
    ray_dir_world = wp.transform_vector(camera_transform, camera_ray)
    cam_forward_world = wp.normalize(wp.transform_vector(camera_transform, wp.vec3f(0.0, 0.0, -1.0)))

    if ray_depth <= 0.0 or wp.dot(ray_dir_world, ray_dir_world) <= 1.0e-12:
        out_depth[world_index, camera_index, py, px] = ray_depth
        return

    out_depth[world_index, camera_index, py, px] = ray_depth * wp.dot(ray_dir_world, cam_forward_world)


@wp.kernel(enable_backward=False)
def unpack_normal_to_rgba_kernel(
    image: wp.array4d[wp.vec3f],
    out: wp.array4d[wp.uint8],
):
    """Unpack (world, camera, H, W) vec3 normals into (N, H, W, 4) uint8 RGB.

    Maps each component from [-1, 1] to [0, 255]. Alpha = 255.
    """
    # NOTE(reviewers): The legacy `flatten_normal_image` kernel does
    # `wp.uint8(normal * 0.5 + 0.5) * 255` with no clamp, which wraps for
    # un-normalized inputs. We clamp here to saturate instead. Identical for
    # normalized normals; different (saturate vs. wrap) for out-of-range
    # inputs. Keep the clamp, or match the old wrap-on-overflow behavior?
    world, camera, y, x = wp.tid()
    camera_count = image.shape[1]
    n = world * camera_count + camera
    nrm = image[world, camera, y, x]
    r = wp.uint8(wp.int32(wp.clamp((nrm[0] + 1.0) * 0.5, 0.0, 1.0) * 255.0))
    g = wp.uint8(wp.int32(wp.clamp((nrm[1] + 1.0) * 0.5, 0.0, 1.0) * 255.0))
    b = wp.uint8(wp.int32(wp.clamp((nrm[2] + 1.0) * 0.5, 0.0, 1.0) * 255.0))
    out[n, y, x, 0] = r
    out[n, y, x, 1] = g
    out[n, y, x, 2] = b
    out[n, y, x, 3] = wp.uint8(255)


@wp.kernel(enable_backward=False)
def unpack_depth_to_rgba_kernel(
    image: wp.array4d[wp.float32],
    depth_range: wp.array[wp.float32],
    out: wp.array4d[wp.uint8],
):
    """Unpack (world, camera, H, W) depth into (N, H, W, 4) uint8 grayscale.

    Invert and normalize to ``[50, 255]`` (closer = brighter). Miss pixels
    (depth <= 0; matches the default ``ClearData.clear_depth = 0.0`` sentinel)
    render black. Alpha = 255. ``depth_range`` is a 2-element array
    ``[near, far]`` consumed on device so the kernel composes with the
    GPU-side ``find_depth_range`` reduction without a host sync.
    """
    world, camera, y, x = wp.tid()
    camera_count = image.shape[1]
    n = world * camera_count + camera
    d = image[world, camera, y, x]
    if d <= 0.0:
        out[n, y, x, 0] = wp.uint8(0)
        out[n, y, x, 1] = wp.uint8(0)
        out[n, y, x, 2] = wp.uint8(0)
        out[n, y, x, 3] = wp.uint8(255)
        return
    near = depth_range[0]
    far = depth_range[1]
    denom = wp.max(far - near, 1e-6)
    t = wp.clamp((d - near) / denom, 0.0, 1.0)
    # Closer -> brighter: near=255, far=50.
    v = wp.uint8(wp.int32((1.0 - t) * 205.0 + 50.0))
    out[n, y, x, 0] = v
    out[n, y, x, 1] = v
    out[n, y, x, 2] = v
    out[n, y, x, 3] = wp.uint8(255)


@wp.kernel(enable_backward=False)
def unpack_shape_index_hash_to_rgba_kernel(
    image: wp.array4d[wp.uint32],
    out: wp.array4d[wp.uint8],
):
    """Colorize shape index with a deterministic hash palette."""
    world, camera, y, x = wp.tid()
    camera_count = image.shape[1]
    n = world * camera_count + camera
    idx = image[world, camera, y, x]
    # Knuth multiplicative hash, masked to 24 bits. ``idx + 1`` keeps shape 0
    # away from the all-zero hash that collides with the miss color; the
    # miss sentinel ``0xFFFFFFFF`` wraps back to 0 and intentionally renders black.
    h = ((idx + wp.uint32(1)) * wp.uint32(2654435761)) & wp.uint32(0xFFFFFF)
    out[n, y, x, 0] = wp.uint8((h >> wp.uint32(16)) & wp.uint32(0xFF))
    out[n, y, x, 1] = wp.uint8((h >> wp.uint32(8)) & wp.uint32(0xFF))
    out[n, y, x, 2] = wp.uint8(h & wp.uint32(0xFF))
    out[n, y, x, 3] = wp.uint8(255)


@wp.kernel(enable_backward=False)
def colorize_shape_index_with_palette_kernel(
    image: wp.array4d[wp.uint32],
    colors: wp.array2d[wp.uint8],
    out: wp.array4d[wp.uint8],
):
    """Colorize shape index by indexing into a caller-provided RGB palette.

    Indices out of range of the palette are rendered black.
    """
    world, camera, y, x = wp.tid()
    camera_count = image.shape[1]
    n = world * camera_count + camera
    idx = image[world, camera, y, x]
    num = wp.uint32(colors.shape[0])
    if idx >= num:
        out[n, y, x, 0] = wp.uint8(0)
        out[n, y, x, 1] = wp.uint8(0)
        out[n, y, x, 2] = wp.uint8(0)
        out[n, y, x, 3] = wp.uint8(255)
        return
    i = wp.int32(idx)
    out[n, y, x, 0] = colors[i, 0]
    out[n, y, x, 1] = colors[i, 1]
    out[n, y, x, 2] = colors[i, 2]
    out[n, y, x, 3] = wp.uint8(255)


def _validate_rgba_out_buffer(
    name: str,
    out_buffer: wp.array[Any],
    expected_shape: tuple[int, int, int, int],
    expected_device: wp.Device,
) -> None:
    """Raise ``ValueError`` if *out_buffer* is not a canonical RGBA sink."""
    if tuple(out_buffer.shape) != expected_shape:
        raise ValueError(f"{name}: out_buffer shape {tuple(out_buffer.shape)} does not match expected {expected_shape}")
    if out_buffer.dtype != wp.uint8:
        raise ValueError(f"{name}: out_buffer dtype must be wp.uint8, got {out_buffer.dtype}")
    if out_buffer.device != expected_device:
        raise ValueError(f"{name}: out_buffer is on {out_buffer.device} but input is on {expected_device}")


class Utils:
    """Utility functions for the RenderContext."""

    def __init__(self, render_context: RenderContext, render_config: RenderConfig | None = None):
        self.__render_context = render_context
        self.__render_config = render_config

    def __warn_implicit_render_config_update(self, method_name: str, config_field: str, value: bool) -> None:
        warnings.warn(
            f"SensorTiledCamera.utils.{method_name}() changed SensorTiledCamera.default_render_config.{config_field}. "
            "This side effect is deprecated as of Newton 1.4 and will be removed in a future release. "
            f"Set sensor.default_render_config.{config_field} = {value!r} explicitly.",
            category=DeprecationWarning,
            stacklevel=3,
        )

    def create_color_image_output(self, width: int, height: int, camera_count: int = 1) -> wp.array4d[wp.uint32]:
        """Create a color output array for :meth:`~newton.sensors.SensorTiledCamera.update`.

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

    def create_depth_image_output(self, width: int, height: int, camera_count: int = 1) -> wp.array4d[wp.float32]:
        """Create a depth output array for :meth:`~newton.sensors.SensorTiledCamera.update`.

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

    def create_shape_index_image_output(self, width: int, height: int, camera_count: int = 1) -> wp.array4d[wp.uint32]:
        """Create a shape-index output array for :meth:`~newton.sensors.SensorTiledCamera.update`.

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

    def create_normal_image_output(self, width: int, height: int, camera_count: int = 1) -> wp.array4d[wp.vec3f]:
        """Create a normal output array for :meth:`~newton.sensors.SensorTiledCamera.update`.

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

    def create_albedo_image_output(self, width: int, height: int, camera_count: int = 1) -> wp.array4d[wp.uint32]:
        """Create an albedo output array for :meth:`~newton.sensors.SensorTiledCamera.update`.

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

    def create_hdr_color_image_output(self, width: int, height: int, camera_count: int = 1) -> wp.array4d[wp.vec3f]:
        """Create a linear HDR color output array for :meth:`~SensorTiledCamera.update`.

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

    def compute_camera_rays_pinhole(
        self,
        width: int,
        height: int,
        *,
        camera_fovs: float | list[float] | np.ndarray | wp.array[wp.float32] | None = None,
        focal_length: float | list[float] | np.ndarray | wp.array[wp.float32] | None = None,
        horizontal_aperture: float | list[float] | np.ndarray | wp.array[wp.float32] | None = None,
        vertical_aperture: float | list[float] | np.ndarray | wp.array[wp.float32] | None = None,
        horizontal_aperture_offset: float | list[float] | np.ndarray | wp.array[wp.float32] = 0.0,
        vertical_aperture_offset: float | list[float] | np.ndarray | wp.array[wp.float32] = 0.0,
        out_rays: wp.array4d[wp.vec3f] | None = None,
        camera_index: int = 0,
    ) -> wp.array4d[wp.vec3f]:
        """Compute camera-space ray directions for pinhole cameras.

        Generates rays in camera space (origin at the camera center,
        direction normalized) for each pixel. Use either vertical field of
        view values or aperture/focal-length values.

        Physical camera parameters accept any consistent length unit, such as
        USD-style millimeters. The unit must match across *focal_length*,
        apertures, and aperture offsets.

        Leave *focal_length*, *horizontal_aperture*, and
        *vertical_aperture* as ``None`` to use *camera_fovs*. Supplying any of
        those aperture parameters selects aperture mode, requires all three,
        and ignores *camera_fovs*.

        Args:
            width: Image width [px].
            height: Image height [px].
            camera_fovs: Vertical FOV angles [rad], scalar or shape
                ``(camera_count,)``. If ``None``, aperture mode must be used;
                if no aperture parameters are provided, raises
                ``ValueError``.
            focal_length: Focal length [mm or any consistent unit]. If
                ``None`` and the other aperture parameters are also ``None``,
                uses *camera_fovs*.
            horizontal_aperture: Horizontal aperture [mm or any consistent
                unit]. If ``None`` and the other aperture parameters are
                also ``None``, uses *camera_fovs*.
            vertical_aperture: Vertical aperture [mm or any consistent unit].
                If ``None`` and the other aperture parameters are also
                ``None``, uses *camera_fovs*.
            horizontal_aperture_offset: Horizontal aperture offset [mm or any
                consistent unit]. Defaults to ``0.0`` for a centered aperture.
            vertical_aperture_offset: Vertical aperture offset [mm or any
                consistent unit]. Defaults to ``0.0`` for a centered aperture.
            out_rays: Optional output array to write into, shape
                ``(out_camera_count, height, width, 2)``. If ``None``,
                allocates a new array.
            camera_index: Camera index in *out_rays* at which to start
                writing. Ignored when *out_rays* is ``None``.

        Returns:
            camera_rays: *out_rays* if provided, otherwise a new array with
                shape ``(camera_count, height, width, 2)`` and dtype
                ``vec3f``.
        """
        use_aperture = focal_length is not None or horizontal_aperture is not None or vertical_aperture is not None
        if use_aperture:
            if focal_length is None or horizontal_aperture is None or vertical_aperture is None:
                raise ValueError("focal_length, horizontal_aperture, and vertical_aperture must be provided together.")

            camera_count = max(
                camera_utils._camera_param_count(focal_length),
                camera_utils._camera_param_count(horizontal_aperture),
                camera_utils._camera_param_count(vertical_aperture),
                camera_utils._camera_param_count(horizontal_aperture_offset),
                camera_utils._camera_param_count(vertical_aperture_offset),
            )
            focal_lengths = camera_utils._camera_param_array(
                "focal_length", focal_length, camera_count, self.__render_context.device
            )
            horizontal_apertures = camera_utils._camera_param_array(
                "horizontal_aperture", horizontal_aperture, camera_count, self.__render_context.device
            )
            vertical_apertures = camera_utils._camera_param_array(
                "vertical_aperture", vertical_aperture, camera_count, self.__render_context.device
            )
            horizontal_aperture_offsets = camera_utils._camera_param_array(
                "horizontal_aperture_offset", horizontal_aperture_offset, camera_count, self.__render_context.device
            )
            vertical_aperture_offsets = camera_utils._camera_param_array(
                "vertical_aperture_offset", vertical_aperture_offset, camera_count, self.__render_context.device
            )
            out_rays, camera_index = camera_utils._validate_camera_ray_output(
                width, height, camera_count, out_rays, camera_index, self.__render_context.device
            )

            wp.launch(
                kernel=camera_utils.compute_camera_rays_pinhole_from_aperture_kernel,
                dim=(camera_count, height, width),
                inputs=[
                    width,
                    height,
                    focal_lengths,
                    horizontal_apertures,
                    vertical_apertures,
                    horizontal_aperture_offsets,
                    vertical_aperture_offsets,
                    camera_index,
                    out_rays,
                ],
                device=self.__render_context.device,
            )

            return out_rays

        if camera_fovs is None:
            raise ValueError("camera_fovs must be provided when aperture parameters are not used.")

        camera_count = camera_utils._camera_param_count(camera_fovs)
        camera_fovs = camera_utils._camera_param_array(
            "camera_fovs", camera_fovs, camera_count, self.__render_context.device
        )
        out_rays, camera_index = camera_utils._validate_camera_ray_output(
            width, height, camera_count, out_rays, camera_index, self.__render_context.device
        )

        wp.launch(
            kernel=camera_utils.compute_camera_rays_pinhole,
            dim=(camera_count, height, width),
            inputs=[
                width,
                height,
                camera_fovs,
                camera_index,
                out_rays,
            ],
            device=self.__render_context.device,
        )

        return out_rays

    def compute_pinhole_camera_rays(
        self,
        width: int,
        height: int,
        camera_fovs: float | list[float] | np.ndarray | wp.array[wp.float32] | None = None,
        *,
        focal_length: float | list[float] | np.ndarray | wp.array[wp.float32] | None = None,
        horizontal_aperture: float | list[float] | np.ndarray | wp.array[wp.float32] | None = None,
        vertical_aperture: float | list[float] | np.ndarray | wp.array[wp.float32] | None = None,
        horizontal_aperture_offset: float | list[float] | np.ndarray | wp.array[wp.float32] = 0.0,
        vertical_aperture_offset: float | list[float] | np.ndarray | wp.array[wp.float32] = 0.0,
        out_rays: wp.array4d[wp.vec3f] | None = None,
        camera_index: int = 0,
    ) -> wp.array4d[wp.vec3f]:
        """Compute camera-space ray directions for pinhole cameras.

        .. deprecated:: 1.4
            Use :meth:`compute_camera_rays_pinhole` instead.

        Physical camera parameters accept any consistent length unit, such as
        USD-style millimeters. The unit must match across *focal_length*,
        apertures, and aperture offsets.

        Leave *focal_length*, *horizontal_aperture*, and
        *vertical_aperture* as ``None`` to use *camera_fovs*. Supplying any of
        those aperture parameters selects aperture mode, requires all three,
        and ignores *camera_fovs*.

        Args:
            width: Image width [px].
            height: Image height [px].
            camera_fovs: Vertical FOV angles [rad], scalar or shape
                ``(camera_count,)``. If ``None``, aperture mode must be used;
                if no aperture parameters are provided, raises
                ``ValueError``.
            focal_length: Focal length [mm or any consistent unit]. If
                ``None`` and the other aperture parameters are also ``None``,
                uses *camera_fovs*.
            horizontal_aperture: Horizontal aperture [mm or any consistent
                unit]. If ``None`` and the other aperture parameters are
                also ``None``, uses *camera_fovs*.
            vertical_aperture: Vertical aperture [mm or any consistent unit].
                If ``None`` and the other aperture parameters are also
                ``None``, uses *camera_fovs*.
            horizontal_aperture_offset: Horizontal aperture offset [mm or any
                consistent unit]. Defaults to ``0.0`` for a centered aperture.
            vertical_aperture_offset: Vertical aperture offset [mm or any
                consistent unit]. Defaults to ``0.0`` for a centered aperture.
            out_rays: Optional output array to write into, shape
                ``(out_camera_count, height, width, 2)``. If ``None``,
                allocates a new array.
            camera_index: Camera index in *out_rays* at which to start
                writing. Ignored when *out_rays* is ``None``.

        Returns:
            camera_rays: *out_rays* if provided, otherwise a new array with
                shape ``(camera_count, height, width, 2)`` and dtype
                ``vec3f``.
        """
        warnings.warn(
            "``SensorTiledCamera.utils.compute_pinhole_camera_rays`` is deprecated. "
            "Use ``SensorTiledCamera.utils.compute_camera_rays_pinhole`` instead.",
            category=DeprecationWarning,
            stacklevel=2,
        )
        return self.compute_camera_rays_pinhole(
            width,
            height,
            camera_fovs=camera_fovs,
            focal_length=focal_length,
            horizontal_aperture=horizontal_aperture,
            vertical_aperture=vertical_aperture,
            horizontal_aperture_offset=horizontal_aperture_offset,
            vertical_aperture_offset=vertical_aperture_offset,
            out_rays=out_rays,
            camera_index=camera_index,
        )

    def compute_camera_rays_usd_pinhole(
        self,
        width: int,
        height: int,
        cameras: camera_utils.UsdCameraInput,
        *,
        time: camera_utils.UsdTime | None = None,
        out_rays: wp.array4d[wp.vec3f] | None = None,
        camera_index: int = 0,
    ) -> wp.array4d[wp.vec3f]:
        """Compute camera-space ray directions for USD pinhole cameras.

        Reads standard ``UsdGeom.Camera`` perspective attributes and forwards
        them to :meth:`compute_camera_rays_pinhole`. The returned
        ``camera_rays`` array has no world axis, so rays are shared by every
        world in a render call. Pair these rays with per-world camera
        transforms only when the cameras at each camera index have matching
        intrinsics across all worlds.

        Args:
            width: Image width [px].
            height: Image height [px].
            cameras: USD camera prim, ``UsdGeom.Camera``, or a flat sequence
                of either. Per-world camera grids are not accepted because
                camera rays cannot vary by world.
            time: Optional USD time code or numeric frame used for camera
                attributes.
            out_rays: Optional output array to write into, shape
                ``(out_camera_count, height, width, 2)``. If ``None``,
                allocates a new array.
            camera_index: Camera index in *out_rays* at which to start
                writing. Ignored when *out_rays* is ``None``.

        Returns:
            camera_rays: *out_rays* if provided, otherwise a new array with
                shape ``(camera_count, height, width, 2)`` and dtype
                ``vec3f``.
        """
        return camera_utils.compute_camera_rays_usd_pinhole(
            width,
            height,
            cameras,
            device=self.__render_context.device,
            time=time,
            out_rays=out_rays,
            camera_index=camera_index,
        )

    def compute_camera_rays_fisheye_opencv(
        self,
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
        out_rays: wp.array4d[wp.vec3f] | None = None,
        camera_index: int = 0,
    ) -> wp.array4d[wp.vec3f]:
        """Compute camera-space ray directions for OpenCV fisheye cameras.

        The distorted radius polynomial
        ``r = theta * (1 + k1 theta^2 + k2 theta^4 + k3 theta^6 + k4 theta^8)``
        must be monotonically increasing over the supported field of view,
        ``[0, min(max_fov / 2, pi)]``.

        Args:
            width: Output image width [px].
            height: Output image height [px].
            fx: Horizontal focal length [px].
            fy: Vertical focal length [px].
            cx: Principal point x-coordinate [px].
            cy: Principal point y-coordinate [px].
            image_width: Calibration image width [px]. If ``None``, uses
                *width*.
            image_height: Calibration image height [px]. If ``None``, uses
                *height*.
            k1: First OpenCV fisheye distortion coefficient.
            k2: Second OpenCV fisheye distortion coefficient.
            k3: Third OpenCV fisheye distortion coefficient.
            k4: Fourth OpenCV fisheye distortion coefficient.
            max_fov: Maximum field of view [rad]. Pixels whose undistorted
                angle exceeds ``max_fov / 2`` receive a zero ray.
            out_rays: Optional output array to write into, shape
                ``(out_camera_count, height, width, 2)``. If ``None``,
                allocates a new array.
            camera_index: Camera index in *out_rays* to write. Ignored when
                *out_rays* is ``None``.

        Returns:
            camera_rays: *out_rays* if provided, otherwise a new array with
                shape ``(1, height, width, 2)`` and dtype ``vec3f``.
        """
        image_width = width if image_width is None else image_width
        image_height = height if image_height is None else image_height
        out_rays, camera_index = camera_utils._validate_camera_ray_output(
            width, height, 1, out_rays, camera_index, self.__render_context.device
        )

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
                camera_index,
                out_rays,
            ],
            device=self.__render_context.device,
        )

        return out_rays

    def compute_camera_rays_fisheye_ftheta(
        self,
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
        out_rays: wp.array4d[wp.vec3f] | None = None,
        camera_index: int = 0,
    ) -> wp.array4d[wp.vec3f]:
        """Compute camera-space ray directions for F-theta fisheye cameras.

        The F-theta radius polynomial
        ``r = k0 + k1 theta + k2 theta^2 + k3 theta^3 + k4 theta^4``
        must be monotonically increasing over the supported field of view,
        ``[0, min(max_fov / 2, pi)]``.

        Args:
            width: Output image width [px].
            height: Output image height [px].
            optical_center_x: Optical center x-coordinate [px].
            optical_center_y: Optical center y-coordinate [px].
            image_width: Calibration image width [px]. If ``None``, uses
                *nominal_width*, then *width*.
            image_height: Calibration image height [px]. If ``None``, uses
                *nominal_height*, then *height*.
            nominal_width: Alias for *image_width* using F-theta model
                terminology. If both are provided, they must match.
            nominal_height: Alias for *image_height* using F-theta model
                terminology. If both are provided, they must match.
            k0: Constant F-theta polynomial coefficient [px].
            k1: Linear F-theta polynomial coefficient [px/rad].
            k2: Quadratic F-theta polynomial coefficient [px/rad^2].
            k3: Cubic F-theta polynomial coefficient [px/rad^3].
            k4: Quartic F-theta polynomial coefficient [px/rad^4].
            max_fov: Maximum field of view [rad].
            out_rays: Optional output array to write into, shape
                ``(out_camera_count, height, width, 2)``. If ``None``,
                allocates a new array.
            camera_index: Camera index in *out_rays* to write. Ignored when
                *out_rays* is ``None``.

        Returns:
            camera_rays: *out_rays* if provided, otherwise a new array with
                shape ``(1, height, width, 2)`` and dtype ``vec3f``.
        """
        image_width = _resolve_fisheye_image_size("width", image_width, nominal_width, width)
        image_height = _resolve_fisheye_image_size("height", image_height, nominal_height, height)
        out_rays, camera_index = camera_utils._validate_camera_ray_output(
            width, height, 1, out_rays, camera_index, self.__render_context.device
        )

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
                camera_index,
                out_rays,
            ],
            device=self.__render_context.device,
        )

        return out_rays

    def compute_camera_rays_fisheye_kannala_brandt(
        self,
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
        out_rays: wp.array4d[wp.vec3f] | None = None,
        camera_index: int = 0,
    ) -> wp.array4d[wp.vec3f]:
        """Compute camera-space ray directions for Kannala-Brandt fisheye cameras.

        Uses the ``r = k0 theta + k1 theta^3 + k2 theta^5 + k3 theta^7``
        polynomial form.

        The radius polynomial must be monotonically increasing over the
        supported field of view, ``[0, min(max_fov / 2, pi)]``.

        Args:
            width: Output image width [px].
            height: Output image height [px].
            optical_center_x: Optical center x-coordinate [px].
            optical_center_y: Optical center y-coordinate [px].
            image_width: Calibration image width [px]. If ``None``, uses
                *nominal_width*, then *width*.
            image_height: Calibration image height [px]. If ``None``, uses
                *nominal_height*, then *height*.
            nominal_width: Alias for *image_width* using Kannala-Brandt model
                terminology. If both are provided, they must match.
            nominal_height: Alias for *image_height* using Kannala-Brandt model
                terminology. If both are provided, they must match.
            k0: Linear Kannala-Brandt coefficient [px/rad].
            k1: Cubic Kannala-Brandt coefficient [px/rad^3].
            k2: Quintic Kannala-Brandt coefficient [px/rad^5].
            k3: Septic Kannala-Brandt coefficient [px/rad^7].
            max_fov: Maximum field of view [rad].
            out_rays: Optional output array to write into, shape
                ``(out_camera_count, height, width, 2)``. If ``None``,
                allocates a new array.
            camera_index: Camera index in *out_rays* to write. Ignored when
                *out_rays* is ``None``.

        Returns:
            camera_rays: *out_rays* if provided, otherwise a new array with
                shape ``(1, height, width, 2)`` and dtype ``vec3f``.
        """
        image_width = _resolve_fisheye_image_size("width", image_width, nominal_width, width)
        image_height = _resolve_fisheye_image_size("height", image_height, nominal_height, height)
        out_rays, camera_index = camera_utils._validate_camera_ray_output(
            width, height, 1, out_rays, camera_index, self.__render_context.device
        )

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
                camera_index,
                out_rays,
            ],
            device=self.__render_context.device,
        )

        return out_rays

    def compute_camera_transforms_usd(
        self,
        cameras: camera_utils.UsdCameraGridInput,
        *,
        time: camera_utils.UsdTime | None = None,
        xform: Any | None = None,
    ) -> wp.array2d[wp.transformf]:
        """Compute camera-to-world transforms from USD camera prims.

        Transforms are rotated from each USD camera's stage up-axis into the
        associated :class:`~newton.Model` up-axis. Pass the same *xform* used
        for :meth:`~newton.ModelBuilder.add_usd` when the USD scene was
        imported with a placement transform.

        The returned transform array may vary by world, but
        :meth:`compute_camera_rays_usd_pinhole` and
        :meth:`~newton.sensors.SensorTiledCamera.update` use ``camera_rays``
        without a world axis. When using a 2D per-world camera layout, every
        camera at the same camera index must therefore share the same
        intrinsics across worlds.

        Args:
            cameras: One or more USD camera prims or ``UsdGeom.Camera``
                schemas. A 1D sequence shares the same transforms across all
                worlds. A 2D sequence indexed ``cameras[world_index][camera_index]``
                assigns a distinct camera per world; the outer dimension must
                equal ``world_count`` and each row must have the same length.
                Note: the input is world-major (outer = world) but the returned
                array is camera-major (outer = camera), i.e., shape
                ``(camera_count, world_count)``.
            time: Optional USD time code or numeric frame used for authored
                camera attributes and transforms.
            xform: Optional scene placement transform to compose with each
                camera transform. Use the same value passed as
                ``ModelBuilder.add_usd(..., xform=xform)`` so cameras and
                imported geometry share the same model-space placement.

        Returns:
            Camera-to-world transforms, shape ``(camera_count, world_count)``.
        """
        return camera_utils.compute_camera_transforms_usd(
            cameras,
            world_count=self.__render_context.world_count,
            device=self.__render_context.device,
            target_up_axis=self.__render_context.up_axis,
            time=time,
            xform=xform,
        )

    def convert_ray_depth_to_forward_depth(
        self,
        depth_image: wp.array4d[wp.float32],
        camera_transforms: wp.array2d[wp.transformf],
        camera_rays: wp.array4d[wp.vec3f],
        out_depth: wp.array4d[wp.float32] | None = None,
    ) -> wp.array4d[wp.float32]:
        """Convert ray-distance depth to forward (planar) depth.

        Projects each pixel's hit distance along its ray onto the camera's
        forward axis, producing depth measured perpendicular to the image
        plane. The forward axis is derived from each camera transform by
        transforming camera-space ``(0, 0, -1)`` into world space.

        Args:
            depth_image: Ray-distance depth [m] from
                :meth:`~newton.sensors.SensorTiledCamera.update`, shape
                ``(world_count, camera_count, height, width)``.
            camera_transforms: World-space camera transforms, shape
                ``(camera_count, world_count)``.
            camera_rays: Camera-space rays from
                :meth:`compute_camera_rays_pinhole` or the fisheye camera ray
                helpers, shape ``(camera_count, height, width, 2)``.
            out_depth: Output forward-depth array [m] with the same shape as
                *depth_image*. If ``None``, allocates a new one.

        Returns:
            Forward (planar) depth array, same shape as *depth_image* [m].
        """
        world_count = depth_image.shape[0]
        camera_count = depth_image.shape[1]
        height = depth_image.shape[2]
        width = depth_image.shape[3]

        if out_depth is None:
            out_depth = wp.empty_like(depth_image, device=self.__render_context.device)

        wp.launch(
            kernel=convert_ray_depth_to_forward_depth_kernel,
            dim=(world_count, camera_count, height, width),
            inputs=[
                depth_image,
                camera_rays,
                camera_transforms,
                out_depth,
            ],
            device=self.__render_context.device,
        )

        return out_depth

    def flatten_color_image_to_rgba(
        self,
        image: wp.array4d[wp.uint32],
        out_buffer: wp.array3d[wp.uint8] | None = None,
        worlds_per_row: int | None = None,
    ) -> wp.array3d[wp.uint8]:
        """Flatten rendered color image to a tiled RGBA buffer.

        Arranges ``(world_count * camera_count)`` tiles in a grid. Each tile shows one camera's view of one world.
        Useful for writing a single pre-tiled image to disk; use :meth:`to_rgba_from_color`
        with :meth:`~newton.viewer.ViewerBase.log_image` for in-viewer display.

        Args:
            image: Color output from :meth:`~newton.sensors.SensorTiledCamera.update`, shape ``(world_count, camera_count, height, width)``.
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

    def to_rgba_from_color(
        self,
        image: wp.array4d[wp.uint32],
    ) -> wp.array4d[wp.uint8]:
        """Reinterpret packed ``uint32`` RGBA color sensor output as ``uint8`` RGBA.

        Returns a zero-copy view: each ``uint32``
        (``R | G<<8 | B<<16 | A<<24``) aliases 4 contiguous ``uint8``
        channels and the ``(world_count, camera_count)`` axes are flattened.
        The returned array shares memory with *image*; do not write into it.

        The returned array plugs directly into :meth:`~newton.viewer.ViewerBase.log_image`.
        World is the slower-changing axis: tile ``i`` has
        ``world = i // camera_count`` and ``camera = i % camera_count``.

        Args:
            image: Color sensor output, shape
                ``(world_count, camera_count, H, W)``, dtype ``uint32``
                (packed RGBA: ``R | G<<8 | B<<16 | A<<24``). Must be
                contiguous; arrays returned by
                :meth:`~newton.sensors.SensorTiledCamera.update` always satisfy this.

        Returns:
            Array of shape ``(world_count * camera_count, H, W, 4)``,
            dtype ``uint8``, aliasing *image*.
        """
        world_count, camera_count, h, w = image.shape
        n = world_count * camera_count
        return image.view(wp.vec4ub).reshape((n, h, w)).view(wp.uint8)

    def to_rgba_from_normal(
        self,
        image: wp.array4d[wp.vec3f],
        out_buffer: wp.array4d[wp.uint8] | None = None,
    ) -> wp.array4d[wp.uint8]:
        """Convert vec3 normal sensor output to ``uint8`` RGBA.

        Args:
            image: Normal output, shape ``(world_count, camera_count, H, W)``,
                dtype ``vec3f``.
            out_buffer: Optional pre-allocated output of shape
                ``(world_count * camera_count, H, W, 4)``, dtype ``uint8``.

        Returns:
            Array of shape ``(world_count * camera_count, H, W, 4)``, dtype
            ``uint8``. Suitable for :meth:`~newton.viewer.ViewerBase.log_image`.
        """
        world_count = image.shape[0]
        camera_count = image.shape[1]
        h = image.shape[2]
        w = image.shape[3]
        n = world_count * camera_count

        if out_buffer is None:
            out_buffer = wp.empty((n, h, w, 4), dtype=wp.uint8, device=self.__render_context.device)
        else:
            _validate_rgba_out_buffer("to_rgba_from_normal", out_buffer, (n, h, w, 4), image.device)

        wp.launch(
            unpack_normal_to_rgba_kernel,
            dim=(world_count, camera_count, h, w),
            inputs=[image],
            outputs=[out_buffer],
            device=self.__render_context.device,
        )
        return out_buffer

    def to_rgba_from_depth(
        self,
        image: wp.array4d[wp.float32],
        depth_range: wp.array[wp.float32] | tuple[float, float] | None = None,
        out_buffer: wp.array4d[wp.uint8] | None = None,
    ) -> wp.array4d[wp.uint8]:
        """Convert float32 depth sensor output to ``uint8`` grayscale RGBA.

        Closer pixels render brighter; miss pixels (depth <= 0; matches the
        default ``ClearData.clear_depth = 0.0`` sentinel) render black.
        Alpha = 255.

        Args:
            image: Depth output, shape ``(world_count, camera_count, H, W)``,
                dtype ``float32``. Non-positive values denote ray misses.
            depth_range: Optional ``(near, far)`` [m] for normalization.
                Accepts a 2-element ``wp.array[wp.float32]`` or a Python
                ``(near, far)`` tuple. If ``None``, the per-frame range is
                computed on device by :func:`find_depth_range` (matches
                :meth:`flatten_depth_image_to_rgba`).
            out_buffer: Optional pre-allocated output of shape
                ``(world_count * camera_count, H, W, 4)``, dtype ``uint8``.

        Returns:
            Array of shape ``(world_count * camera_count, H, W, 4)``, dtype
            ``uint8``. Suitable for :meth:`~newton.viewer.ViewerBase.log_image`.
        """
        world_count = image.shape[0]
        camera_count = image.shape[1]
        h = image.shape[2]
        w = image.shape[3]
        n = world_count * camera_count
        device = self.__render_context.device

        if depth_range is None:
            depth_range_arr = wp.array([MAXVAL, 0.0], dtype=wp.float32, device=device)
            wp.launch(find_depth_range, image.shape, [image, depth_range_arr], device=device)
        elif isinstance(depth_range, wp.array):
            depth_range_arr = depth_range
        else:
            near, far = float(depth_range[0]), float(depth_range[1])
            if not (near < far):
                raise ValueError(f"to_rgba_from_depth: depth_range must satisfy near < far, got near={near}, far={far}")
            depth_range_arr = wp.array([near, far], dtype=wp.float32, device=device)

        if out_buffer is None:
            out_buffer = wp.empty((n, h, w, 4), dtype=wp.uint8, device=device)
        else:
            _validate_rgba_out_buffer("to_rgba_from_depth", out_buffer, (n, h, w, 4), image.device)

        wp.launch(
            unpack_depth_to_rgba_kernel,
            dim=(world_count, camera_count, h, w),
            inputs=[image, depth_range_arr],
            outputs=[out_buffer],
            device=device,
        )
        return out_buffer

    def to_rgba_from_shape_index(
        self,
        image: wp.array4d[wp.uint32],
        colors: wp.array2d[wp.uint8] | None = None,
        out_buffer: wp.array4d[wp.uint8] | None = None,
    ) -> wp.array4d[wp.uint8]:
        """Convert uint32 shape-index sensor output to ``uint8`` RGBA.

        Args:
            image: Shape-index output, shape
                ``(world_count, camera_count, H, W)``, dtype ``uint32``.
            colors: Optional RGB palette of shape ``(num_entries, 3)``, dtype
                ``uint8``. If provided, each pixel is colored by looking up
                its shape index in this palette (indices past the palette
                length render black). If ``None``, a deterministic hash
                palette is used (good for debugging which shape hit which
                pixel without a predefined class map).
            out_buffer: Optional pre-allocated output of shape
                ``(world_count * camera_count, H, W, 4)``, dtype ``uint8``.

        Returns:
            Array of shape ``(world_count * camera_count, H, W, 4)``, dtype
            ``uint8``. Suitable for :meth:`~newton.viewer.ViewerBase.log_image`.
        """
        world_count = image.shape[0]
        camera_count = image.shape[1]
        h = image.shape[2]
        w = image.shape[3]
        n = world_count * camera_count

        if out_buffer is None:
            out_buffer = wp.empty((n, h, w, 4), dtype=wp.uint8, device=self.__render_context.device)
        else:
            _validate_rgba_out_buffer("to_rgba_from_shape_index", out_buffer, (n, h, w, 4), image.device)

        if colors is None:
            wp.launch(
                unpack_shape_index_hash_to_rgba_kernel,
                dim=(world_count, camera_count, h, w),
                inputs=[image],
                outputs=[out_buffer],
                device=self.__render_context.device,
            )
        else:
            wp.launch(
                colorize_shape_index_with_palette_kernel,
                dim=(world_count, camera_count, h, w),
                inputs=[image, colors],
                outputs=[out_buffer],
                device=self.__render_context.device,
            )
        return out_buffer

    def flatten_normal_image_to_rgba(
        self,
        image: wp.array4d[wp.vec3f],
        out_buffer: wp.array3d[wp.uint8] | None = None,
        worlds_per_row: int | None = None,
    ) -> wp.array3d[wp.uint8]:
        """Flatten rendered normal image to a tiled RGBA buffer.

        Arranges ``(world_count * camera_count)`` tiles in a grid. Each tile shows one camera's view of one world.
        Useful for writing a single pre-tiled image to disk; use :meth:`to_rgba_from_normal`
        with :meth:`~newton.viewer.ViewerBase.log_image` for in-viewer display.

        Args:
            image: Normal output from :meth:`~newton.sensors.SensorTiledCamera.update`, shape ``(world_count, camera_count, height, width)``.
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
        image: wp.array4d[wp.float32],
        out_buffer: wp.array3d[wp.uint8] | None = None,
        worlds_per_row: int | None = None,
        depth_range: wp.array[wp.float32] | None = None,
    ) -> wp.array3d[wp.uint8]:
        """Flatten rendered depth image to a tiled RGBA buffer.

        Encodes depth as grayscale: inverts values (closer = brighter) and normalizes to the ``[50, 255]``
        range. Background pixels (no hit) remain black. Useful for writing a single pre-tiled image to disk;
        use :meth:`to_rgba_from_depth` with :meth:`~newton.viewer.ViewerBase.log_image` for in-viewer display.

        Args:
            image: Depth output from :meth:`~newton.sensors.SensorTiledCamera.update`, shape ``(world_count, camera_count, height, width)``.
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

    def create_default_light(
        self,
        enable_shadows: bool = True,
        direction: wp.vec3f | None = None,
    ):
        """Create a default directional light oriented at ``(-1, 1, -1)``.

        Args:
            enable_shadows: Enable shadow casting for this light.
            direction: Normalized light direction. If ``None``, defaults to
                (normalized ``(-1, 1, -1)``).
        """
        if self.__render_config is not None:
            if self.__render_config.enable_shadows != enable_shadows:
                self.__warn_implicit_render_config_update("create_default_light", "enable_shadows", enable_shadows)
            self.__render_config.enable_shadows = enable_shadows
        self.__render_context.lights_active = wp.array([True], dtype=wp.bool, device=self.__render_context.device)
        self.__render_context.lights_type = wp.array(
            [RenderLightType.DIRECTIONAL], dtype=wp.int32, device=self.__render_context.device
        )
        self.__render_context.lights_cast_shadow = wp.array(
            [enable_shadows], dtype=wp.bool, device=self.__render_context.device
        )
        self.__render_context.lights_position = wp.array(
            [wp.vec3f(0.0)], dtype=wp.vec3f, device=self.__render_context.device
        )
        self.__render_context.lights_orientation = wp.array(
            [direction if direction is not None else wp.vec3f(-0.57735026, 0.57735026, -0.57735026)],
            dtype=wp.vec3f,
            device=self.__render_context.device,
        )

    def assign_checkerboard_material(
        self,
        *,
        shape_indices: Sequence[int] | np.ndarray,
        resolution: int = 64,
        checker_size: int = 32,
    ):
        """Assign a gray checkerboard texture material to selected shapes.

        Args:
            shape_indices: Shape indices that should use the checkerboard texture.
            resolution: Texture resolution in pixels (square texture).
            checker_size: Size of each checkerboard square in pixels.
        """
        shape_indices = np.asarray(shape_indices, dtype=np.int64).reshape(-1)
        invalid = (shape_indices < 0) | (shape_indices >= self.__render_context.shape_count_total)
        if invalid.any():
            raise ValueError("shape_indices contains an out-of-range shape index")

        checkerboard = (
            (np.arange(resolution) // checker_size)[:, None] + (np.arange(resolution) // checker_size)
        ) % 2 == 0

        pixels = np.where(checkerboard, 0xFF808080, 0xFFBFBFBF).astype(np.uint32)

        texture_ids = np.full(self.__render_context.shape_count_total, fill_value=-1, dtype=np.int32)
        texture_ids[shape_indices] = 0

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

        if self.__render_config is not None:
            if not self.__render_config.enable_textures:
                self.__warn_implicit_render_config_update("assign_checkerboard_material", "enable_textures", True)
            self.__render_config.enable_textures = True
        self.__render_context.texture_data = wp.array(
            [self.__checkerboard_data], dtype=TextureData, device=self.__render_context.device
        )
        self.__render_context.shape_texture_ids = wp.array(
            texture_ids, dtype=wp.int32, device=self.__render_context.device
        )

    def assign_checkerboard_material_to_all_shapes(self, resolution: int = 64, checker_size: int = 32):
        """Assign a gray checkerboard texture material to all shapes.

        .. deprecated:: 1.4
            Use :meth:`assign_checkerboard_material` with explicit shape
            indices instead.

        Args:
            resolution: Texture resolution in pixels (square texture).
            checker_size: Size of each checkerboard square in pixels.
        """
        warnings.warn(
            "``SensorTiledCamera.utils.assign_checkerboard_material_to_all_shapes`` is deprecated as of Newton 1.4. "
            "Use ``SensorTiledCamera.utils.assign_checkerboard_material(shape_indices=...)`` instead.",
            category=DeprecationWarning,
            stacklevel=2,
        )
        self.assign_checkerboard_material(
            shape_indices=np.arange(self.__render_context.shape_count_total, dtype=np.int32),
            resolution=resolution,
            checker_size=checker_size,
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
        if worlds_per_row is None:
            worlds_per_row = math.ceil(math.sqrt(world_and_camera_count))
        elif worlds_per_row < 1:
            raise ValueError(f"worlds_per_row must be >= 1, got {worlds_per_row}")
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
