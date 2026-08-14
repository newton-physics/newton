# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math
from typing import Any

import warp as wp

from ..core import MAXVAL

# Knuth multiplicative hash constant (2^32 / golden ratio).
# Typed uint32 so kernel codegen doesn't overflow an int32 constant.
HASH_MULTIPLIER = wp.uint32(2654435761)


@wp.kernel(enable_backward=False)
def flatten_color_image(
    color_image: wp.array3d[wp.uint32],
    buffer: wp.array3d[wp.uint8],
    width: wp.int32,
    height: wp.int32,
    worlds_per_row: wp.int32,
):
    world_id, y, x = wp.tid()

    row = world_id // worlds_per_row
    col = world_id % worlds_per_row

    px = col * width + x
    py = row * height + y
    color = color_image[world_id, y, x]

    buffer[py, px, 0] = wp.uint8((color >> wp.uint32(0)) & wp.uint32(0xFF))
    buffer[py, px, 1] = wp.uint8((color >> wp.uint32(8)) & wp.uint32(0xFF))
    buffer[py, px, 2] = wp.uint8((color >> wp.uint32(16)) & wp.uint32(0xFF))
    buffer[py, px, 3] = wp.uint8((color >> wp.uint32(24)) & wp.uint32(0xFF))


@wp.kernel(enable_backward=False)
def flatten_normal_image(
    normal_image: wp.array3d[wp.vec3f],
    buffer: wp.array3d[wp.uint8],
    width: wp.int32,
    height: wp.int32,
    worlds_per_row: wp.int32,
):
    world_id, y, x = wp.tid()

    row = world_id // worlds_per_row
    col = world_id % worlds_per_row

    px = col * width + x
    py = row * height + y
    normal = normal_image[world_id, y, x] * 0.5 + wp.vec3f(0.5)

    buffer[py, px, 0] = wp.uint8(normal[0] * 255.0)
    buffer[py, px, 1] = wp.uint8(normal[1] * 255.0)
    buffer[py, px, 2] = wp.uint8(normal[2] * 255.0)
    buffer[py, px, 3] = wp.uint8(255)


@wp.kernel(enable_backward=False)
def find_depth_range(depth_image: wp.array3d[wp.float32], depth_range: wp.array[wp.float32]):
    world_id, y, x = wp.tid()
    depth = depth_image[world_id, y, x]
    if depth > 0:
        wp.atomic_min(depth_range, 0, depth)
        wp.atomic_max(depth_range, 1, depth)


@wp.kernel(enable_backward=False)
def flatten_depth_image(
    depth_image: wp.array3d[wp.float32],
    buffer: wp.array3d[wp.uint8],
    depth_range: wp.array[wp.float32],
    width: wp.int32,
    height: wp.int32,
    worlds_per_row: wp.int32,
):
    world_id, y, x = wp.tid()

    row = world_id // worlds_per_row
    col = world_id % worlds_per_row

    px = col * width + x
    py = row * height + y

    value = wp.uint8(0)
    depth = depth_image[world_id, y, x]
    if depth > 0:
        denom = wp.max(depth_range[1] - depth_range[0], 1e-6)
        value = wp.uint8(255.0 - ((depth - depth_range[0]) / denom) * 205.0)

    buffer[py, px, 0] = value
    buffer[py, px, 1] = value
    buffer[py, px, 2] = value
    buffer[py, px, 3] = value


@wp.kernel(enable_backward=False)
def convert_ray_depth_to_forward_depth_kernel(
    depth_image: wp.array3d[wp.float32],
    camera_rays: wp.array3d[wp.vec3f],
    camera_transforms: wp.array[wp.transformf],
    out_depth: wp.array3d[wp.float32],
):
    world_index, py, px = wp.tid()

    ray_depth = depth_image[world_index, py, px]
    camera_transform = camera_transforms[world_index]
    camera_ray = camera_rays[py, px, 1]
    ray_dir_world = wp.transform_vector(camera_transform, camera_ray)
    cam_forward_world = wp.normalize(wp.transform_vector(camera_transform, wp.vec3f(0.0, 0.0, -1.0)))

    if ray_depth <= 0.0 or wp.dot(ray_dir_world, ray_dir_world) <= 1.0e-12:
        out_depth[world_index, py, px] = ray_depth
        return

    out_depth[world_index, py, px] = ray_depth * wp.dot(ray_dir_world, cam_forward_world)


@wp.kernel(enable_backward=False)
def unpack_normal_to_rgba_kernel(
    image: wp.array3d[wp.vec3f],
    out: wp.array4d[wp.uint8],
):
    """Unpack (world, H, W) vec3 normals into (world, H, W, 4) uint8 RGB.

    Maps each component from [-1, 1] to [0, 255]. Alpha = 255.
    """
    # Rendered normals should be normalized, but clamp debug inputs to avoid
    # wrapping out-of-range values through uint8 conversion.
    world, y, x = wp.tid()
    nrm = image[world, y, x]
    r = wp.uint8(wp.int32(wp.clamp((nrm[0] + 1.0) * 0.5, 0.0, 1.0) * 255.0))
    g = wp.uint8(wp.int32(wp.clamp((nrm[1] + 1.0) * 0.5, 0.0, 1.0) * 255.0))
    b = wp.uint8(wp.int32(wp.clamp((nrm[2] + 1.0) * 0.5, 0.0, 1.0) * 255.0))
    out[world, y, x, 0] = r
    out[world, y, x, 1] = g
    out[world, y, x, 2] = b
    out[world, y, x, 3] = wp.uint8(255)


@wp.kernel(enable_backward=False)
def unpack_depth_to_rgba_kernel(
    image: wp.array3d[wp.float32],
    depth_range: wp.array[wp.float32],
    out: wp.array4d[wp.uint8],
):
    """Unpack (world, H, W) depth into (world, H, W, 4) uint8 grayscale.

    Invert and normalize to ``[50, 255]`` (closer = brighter). Miss pixels
    (depth <= 0; matches the default ``ClearData.clear_depth = 0.0`` sentinel)
    render black. Alpha = 255. ``depth_range`` is a 2-element array
    ``[near, far]`` consumed on device so the kernel composes with the
    GPU-side ``find_depth_range`` reduction without a host sync.
    """
    world, y, x = wp.tid()
    d = image[world, y, x]
    if d <= 0.0:
        out[world, y, x, 0] = wp.uint8(0)
        out[world, y, x, 1] = wp.uint8(0)
        out[world, y, x, 2] = wp.uint8(0)
        out[world, y, x, 3] = wp.uint8(255)
        return
    near = depth_range[0]
    far = depth_range[1]
    denom = wp.max(far - near, 1e-6)
    t = wp.clamp((d - near) / denom, 0.0, 1.0)
    # Closer -> brighter: near=255, far=50.
    v = wp.uint8(wp.int32((1.0 - t) * 205.0 + 50.0))
    out[world, y, x, 0] = v
    out[world, y, x, 1] = v
    out[world, y, x, 2] = v
    out[world, y, x, 3] = wp.uint8(255)


@wp.kernel(enable_backward=False)
def unpack_shape_index_hash_to_rgba_kernel(
    image: wp.array3d[wp.uint32],
    out: wp.array4d[wp.uint8],
):
    """Colorize shape index with a deterministic hash palette."""
    world, y, x = wp.tid()
    idx = image[world, y, x]
    # Knuth multiplicative hash, masked to 24 bits. ``idx + 1`` keeps shape 0
    # away from the all-zero hash that collides with the miss color; the
    # miss sentinel ``0xFFFFFFFF`` wraps back to 0 and intentionally renders black.
    h = ((idx + wp.uint32(1)) * HASH_MULTIPLIER) & wp.uint32(0xFFFFFF)
    out[world, y, x, 0] = wp.uint8((h >> wp.uint32(16)) & wp.uint32(0xFF))
    out[world, y, x, 1] = wp.uint8((h >> wp.uint32(8)) & wp.uint32(0xFF))
    out[world, y, x, 2] = wp.uint8(h & wp.uint32(0xFF))
    out[world, y, x, 3] = wp.uint8(255)


@wp.kernel(enable_backward=False)
def colorize_shape_index_with_palette_kernel(
    image: wp.array3d[wp.uint32],
    colors: wp.array2d[wp.uint8],
    out: wp.array4d[wp.uint8],
):
    """Colorize shape index by indexing into a caller-provided RGB palette.

    Indices out of range of the palette are rendered black.
    """
    world, y, x = wp.tid()
    idx = image[world, y, x]
    num = wp.uint32(colors.shape[0])
    if idx >= num:
        out[world, y, x, 0] = wp.uint8(0)
        out[world, y, x, 1] = wp.uint8(0)
        out[world, y, x, 2] = wp.uint8(0)
        out[world, y, x, 3] = wp.uint8(255)
        return
    i = wp.int32(idx)
    out[world, y, x, 0] = colors[i, 0]
    out[world, y, x, 1] = colors[i, 1]
    out[world, y, x, 2] = colors[i, 2]
    out[world, y, x, 3] = wp.uint8(255)


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
    """Utility functions for a SensorCamera."""

    def __init__(self, view_count: int, device: wp.Device):
        self.__view_count = int(view_count)
        self.__device = device

    def __image_shape(self, name: str, image: wp.array[Any]) -> tuple[int, int, int]:
        view_count = self.__view_count
        device = self.__device
        if image.shape[0] != view_count:
            raise ValueError(
                f"{name}: image leading dimension {image.shape[0]} must match SensorCamera.view_count {view_count}"
            )
        if image.device != device:
            raise ValueError(f"{name}: image is on {image.device} but SensorCamera is on {device}")
        return view_count, image.shape[1], image.shape[2]

    def convert_ray_depth_to_forward_depth(
        self,
        depth_image: wp.array3d[wp.float32],
        camera_transforms: wp.array[wp.transformf],
        camera_rays: wp.array3d[wp.vec3f],
        out_depth: wp.array3d[wp.float32] | None = None,
    ) -> wp.array3d[wp.float32]:
        """Convert ray-distance depth to forward (planar) depth.

        Projects each pixel's hit distance along its ray onto the camera's
        forward axis, producing depth measured perpendicular to the image
        plane. The forward axis is derived from each camera transform by
        transforming camera-space ``(0, 0, -1)`` into world space.

        Args:
            depth_image: Ray-distance depth [m] from
                :meth:`~newton.sensors.SensorCamera.update`, shape
                ``(view_count, height, width)``.
            camera_transforms: World-space camera transforms, shape
                ``(view_count,)``.
            camera_rays: Camera-space rays from
                :class:`~newton.sensors.SensorCamera`, shape
                ``(height, width, 2)``. Ray direction vectors must be unit
                length; non-unit directions scale the converted depth.
            out_depth: Output forward-depth array [m] with the same shape as
                *depth_image*. If ``None``, allocates a new one.

        Returns:
            Forward (planar) depth array, same shape as *depth_image* [m].
        """
        view_count, height, width = self.__image_shape("convert_ray_depth_to_forward_depth", depth_image)
        device = self.__device

        if out_depth is None:
            out_depth = wp.empty_like(depth_image, device=device)

        wp.launch(
            kernel=convert_ray_depth_to_forward_depth_kernel,
            dim=(view_count, height, width),
            inputs=[
                depth_image,
                camera_rays,
                camera_transforms,
                out_depth,
            ],
            device=device,
        )

        return out_depth

    def flatten_color_image_to_rgba(
        self,
        image: wp.array3d[wp.uint32],
        out_buffer: wp.array3d[wp.uint8] | None = None,
        worlds_per_row: int | None = None,
    ) -> wp.array3d[wp.uint8]:
        """Flatten rendered color image to a tiled RGBA buffer.

        Arranges one tile per view in a grid.
        Useful for writing a single pre-tiled image to disk; use :meth:`to_rgba_from_color`
        with :meth:`~newton.viewer.ViewerBase.log_image` for in-viewer display.

        Args:
            image: Color output from :meth:`~newton.sensors.SensorCamera.update`, shape ``(view_count, height, width)``.
            out_buffer: Pre-allocated RGBA buffer. If None, allocates a new one.
            worlds_per_row: Views per row in the grid. If None, picks a square-ish layout.
        """
        view_count, height, width = self.__image_shape("flatten_color_image_to_rgba", image)
        device = self.__device

        out_buffer, worlds_per_row = self.__reshape_buffer_for_flatten(width, height, out_buffer, worlds_per_row)

        wp.launch(
            flatten_color_image,
            (
                view_count,
                height,
                width,
            ),
            [
                image,
                out_buffer,
                width,
                height,
                worlds_per_row,
            ],
            device=device,
        )
        return out_buffer

    def to_rgba_from_color(
        self,
        image: wp.array3d[wp.uint32],
    ) -> wp.array4d[wp.uint8]:
        """Reinterpret packed ``uint32`` RGBA color sensor output as ``uint8`` RGBA.

        Returns a zero-copy view: each ``uint32``
        (``R | G<<8 | B<<16 | A<<24``) aliases 4 contiguous ``uint8``
        channels.
        The returned array shares memory with *image*; do not write into it.

        The returned array plugs directly into :meth:`~newton.viewer.ViewerBase.log_image`.

        Args:
            image: Color sensor output, shape
                ``(view_count, H, W)``, dtype ``uint32``
                (packed RGBA: ``R | G<<8 | B<<16 | A<<24``). Must be
                contiguous.

        Returns:
            Array of shape ``(view_count, H, W, 4)``,
            dtype ``uint8``, aliasing *image*.
        """
        view_count, h, w = self.__image_shape("to_rgba_from_color", image)
        return image.view(wp.vec4ub).reshape((view_count, h, w)).view(wp.uint8)

    def to_rgba_from_normal(
        self,
        image: wp.array3d[wp.vec3f],
        out_buffer: wp.array4d[wp.uint8] | None = None,
    ) -> wp.array4d[wp.uint8]:
        """Convert vec3 normal sensor output to ``uint8`` RGBA.

        Args:
            image: Normal output, shape ``(view_count, H, W)``, dtype ``vec3f``.
            out_buffer: Optional pre-allocated output of shape
                ``(view_count, H, W, 4)``, dtype ``uint8``.

        Returns:
            Array of shape ``(view_count, H, W, 4)``, dtype
            ``uint8``. Suitable for :meth:`~newton.viewer.ViewerBase.log_image`.
        """
        view_count, h, w = self.__image_shape("to_rgba_from_normal", image)
        device = self.__device

        if out_buffer is None:
            out_buffer = wp.empty((view_count, h, w, 4), dtype=wp.uint8, device=device)
        else:
            _validate_rgba_out_buffer("to_rgba_from_normal", out_buffer, (view_count, h, w, 4), device)

        wp.launch(
            unpack_normal_to_rgba_kernel,
            dim=(view_count, h, w),
            inputs=[image],
            outputs=[out_buffer],
            device=device,
        )
        return out_buffer

    def to_rgba_from_depth(
        self,
        image: wp.array3d[wp.float32],
        depth_range: wp.array[wp.float32] | tuple[float, float] | None = None,
        out_buffer: wp.array4d[wp.uint8] | None = None,
    ) -> wp.array4d[wp.uint8]:
        """Convert float32 depth sensor output to ``uint8`` grayscale RGBA.

        Closer pixels render brighter; miss pixels (depth <= 0; matches the
        default ``ClearData.clear_depth = 0.0`` sentinel) render black.
        Alpha = 255.

        Args:
            image: Depth output, shape ``(view_count, H, W)``, dtype
                ``float32``. Non-positive values denote ray misses.
            depth_range: Optional ``(near, far)`` [m] for normalization.
                Accepts a 2-element ``wp.array[wp.float32]`` or a Python
                ``(near, far)`` tuple. If ``None``, the per-frame range is
                computed on device by :func:`find_depth_range` (matches
                :meth:`flatten_depth_image_to_rgba`).
            out_buffer: Optional pre-allocated output of shape
                ``(view_count, H, W, 4)``, dtype ``uint8``.

        Returns:
            Array of shape ``(view_count, H, W, 4)``, dtype
            ``uint8``. Suitable for :meth:`~newton.viewer.ViewerBase.log_image`.
        """
        view_count, h, w = self.__image_shape("to_rgba_from_depth", image)
        device = self.__device

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
            out_buffer = wp.empty((view_count, h, w, 4), dtype=wp.uint8, device=device)
        else:
            _validate_rgba_out_buffer("to_rgba_from_depth", out_buffer, (view_count, h, w, 4), device)

        wp.launch(
            unpack_depth_to_rgba_kernel,
            dim=(view_count, h, w),
            inputs=[image, depth_range_arr],
            outputs=[out_buffer],
            device=device,
        )
        return out_buffer

    def to_rgba_from_shape_index(
        self,
        image: wp.array3d[wp.uint32],
        colors: wp.array2d[wp.uint8] | None = None,
        out_buffer: wp.array4d[wp.uint8] | None = None,
    ) -> wp.array4d[wp.uint8]:
        """Convert uint32 shape-index sensor output to ``uint8`` RGBA.

        Args:
            image: Shape-index output, shape
                ``(view_count, H, W)``, dtype ``uint32``.
            colors: Optional RGB palette of shape ``(num_entries, 3)``, dtype
                ``uint8``. If provided, each pixel is colored by looking up
                its shape index in this palette (indices past the palette
                length render black). If ``None``, a deterministic hash
                palette is used (good for debugging which shape hit which
                pixel without a predefined class map).
            out_buffer: Optional pre-allocated output of shape
                ``(view_count, H, W, 4)``, dtype ``uint8``.

        Returns:
            Array of shape ``(view_count, H, W, 4)``, dtype
            ``uint8``. Suitable for :meth:`~newton.viewer.ViewerBase.log_image`.
        """
        view_count, h, w = self.__image_shape("to_rgba_from_shape_index", image)
        device = self.__device

        if out_buffer is None:
            out_buffer = wp.empty((view_count, h, w, 4), dtype=wp.uint8, device=device)
        else:
            _validate_rgba_out_buffer("to_rgba_from_shape_index", out_buffer, (view_count, h, w, 4), device)

        if colors is None:
            wp.launch(
                unpack_shape_index_hash_to_rgba_kernel,
                dim=(view_count, h, w),
                inputs=[image],
                outputs=[out_buffer],
                device=device,
            )
        else:
            wp.launch(
                colorize_shape_index_with_palette_kernel,
                dim=(view_count, h, w),
                inputs=[image, colors],
                outputs=[out_buffer],
                device=device,
            )
        return out_buffer

    def flatten_normal_image_to_rgba(
        self,
        image: wp.array3d[wp.vec3f],
        out_buffer: wp.array3d[wp.uint8] | None = None,
        worlds_per_row: int | None = None,
    ) -> wp.array3d[wp.uint8]:
        """Flatten rendered normal image to a tiled RGBA buffer.

        Arranges one tile per view in a grid.
        Useful for writing a single pre-tiled image to disk; use :meth:`to_rgba_from_normal`
        with :meth:`~newton.viewer.ViewerBase.log_image` for in-viewer display.

        Args:
            image: Normal output from :meth:`~newton.sensors.SensorCamera.update`, shape ``(view_count, height, width)``.
            out_buffer: Pre-allocated RGBA buffer. If None, allocates a new one.
            worlds_per_row: Views per row in the grid. If None, picks a square-ish layout.
        """
        view_count, height, width = self.__image_shape("flatten_normal_image_to_rgba", image)
        device = self.__device

        out_buffer, worlds_per_row = self.__reshape_buffer_for_flatten(width, height, out_buffer, worlds_per_row)

        wp.launch(
            flatten_normal_image,
            (
                view_count,
                height,
                width,
            ),
            [
                image,
                out_buffer,
                width,
                height,
                worlds_per_row,
            ],
            device=device,
        )
        return out_buffer

    def flatten_depth_image_to_rgba(
        self,
        image: wp.array3d[wp.float32],
        out_buffer: wp.array3d[wp.uint8] | None = None,
        worlds_per_row: int | None = None,
        depth_range: wp.array[wp.float32] | None = None,
    ) -> wp.array3d[wp.uint8]:
        """Flatten rendered depth image to a tiled RGBA buffer.

        Encodes depth as grayscale: inverts values (closer = brighter) and normalizes to the ``[50, 255]``
        range. Background pixels (no hit) remain black. Useful for writing a single pre-tiled image to disk;
        use :meth:`to_rgba_from_depth` with :meth:`~newton.viewer.ViewerBase.log_image` for in-viewer display.

        Args:
            image: Depth output from :meth:`~newton.sensors.SensorCamera.update`, shape ``(view_count, height, width)``.
            out_buffer: Pre-allocated RGBA buffer. If None, allocates a new one.
            worlds_per_row: Views per row in the grid. If None, picks a square-ish layout.
            depth_range: Depth range to normalize to, shape ``(2,)`` ``[near, far]``. If None, computes from *image*.
        """
        view_count, height, width = self.__image_shape("flatten_depth_image_to_rgba", image)
        device = self.__device

        out_buffer, worlds_per_row = self.__reshape_buffer_for_flatten(width, height, out_buffer, worlds_per_row)

        if depth_range is None:
            depth_range = wp.array([MAXVAL, 0.0], dtype=wp.float32, device=device)
            wp.launch(find_depth_range, image.shape, [image, depth_range], device=device)

        wp.launch(
            flatten_depth_image,
            (
                view_count,
                height,
                width,
            ),
            [
                image,
                out_buffer,
                depth_range,
                width,
                height,
                worlds_per_row,
            ],
            device=device,
        )
        return out_buffer

    def __reshape_buffer_for_flatten(
        self,
        width: int,
        height: int,
        out_buffer: wp.array | None = None,
        worlds_per_row: int | None = None,
    ) -> tuple[wp.array3d[wp.uint8], int]:
        view_count = self.__view_count
        if worlds_per_row is None:
            worlds_per_row = math.ceil(math.sqrt(view_count))
        elif worlds_per_row < 1:
            raise ValueError(f"worlds_per_row must be >= 1, got {worlds_per_row}")
        worlds_per_col = math.ceil(view_count / worlds_per_row)

        if out_buffer is None:
            return wp.empty(
                (
                    worlds_per_col * height,
                    worlds_per_row * width,
                    4,
                ),
                dtype=wp.uint8,
                device=self.__device,
            ), worlds_per_row

        return out_buffer.reshape((worlds_per_col * height, worlds_per_row * width, 4)), worlds_per_row
