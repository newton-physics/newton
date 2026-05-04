# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math
import warnings
from typing import TYPE_CHECKING, Any

import numpy as np
import warp as wp

from ...core import MAXVAL
from .types import RenderLightType, TextureData

if TYPE_CHECKING:
    from .render_context import RenderContext


@wp.kernel(enable_backward=False)
def compute_pinhole_camera_rays(
    width: int,
    height: int,
    camera_fovs: wp.array[wp.float32],
    out_rays: wp.array4d[wp.vec3f],
):
    camera_index, py, px = wp.tid()
    aspect_ratio = float(width) / float(height)
    u = (float(px) + 0.5) / float(width) - 0.5
    v = (float(py) + 0.5) / float(height) - 0.5
    h = wp.tan(camera_fovs[camera_index] / 2.0)
    ray_direction_camera_space = wp.vec3f(u * 2.0 * h * aspect_ratio, -v * 2.0 * h, -1.0)
    out_rays[camera_index, py, px, 0] = wp.vec3f(0.0)
    out_rays[camera_index, py, px, 1] = wp.normalize(ray_direction_camera_space)


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

    def __init__(self, render_context: RenderContext):
        self.__render_context = render_context

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

    def compute_pinhole_camera_rays(
        self, width: int, height: int, camera_fovs: float | list[float] | np.ndarray | wp.array[wp.float32]
    ) -> wp.array4d[wp.vec3f]:
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

        camera_count = camera_fovs.size

        camera_rays = wp.empty((camera_count, height, width, 2), dtype=wp.vec3f, device=self.__render_context.device)

        wp.launch(
            kernel=compute_pinhole_camera_rays,
            dim=(camera_count, height, width),
            inputs=[
                width,
                height,
                camera_fovs,
                camera_rays,
            ],
            device=self.__render_context.device,
        )

        return camera_rays

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
                :meth:`compute_pinhole_camera_rays`, shape
                ``(camera_count, height, width, 2)``.
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

    def assign_random_colors_per_world(self, seed: int = 100):
        """Assign each world a random color, applied to all its shapes.

        .. deprecated:: 1.1
            Use shape colors instead (e.g. ``builder.add_shape_cylinder(..., color=(r, g, b))``).

        Args:
            seed: Random seed.
        """
        warnings.warn(
            "``SensorTiledCamera.utils.assign_random_colors_per_world`` is deprecated. Use shape colors instead (e.g. ``builder.add_shape_cylinder(..., color=(r, g, b))``).",
            category=DeprecationWarning,
            stacklevel=2,
        )

        if not self.__render_context.shape_count_total:
            return
        colors = np.random.default_rng(seed).random((self.__render_context.shape_count_total, 3)) * 0.5 + 0.5
        self.__render_context.shape_colors = wp.array(
            colors[self.__render_context.shape_world_index.numpy() % len(colors)],
            dtype=wp.vec3f,
            device=self.__render_context.device,
        )

    def assign_random_colors_per_shape(self, seed: int = 100):
        """Assign a random color to each shape.

        .. deprecated:: 1.1
            Use shape colors instead (e.g. ``builder.add_shape_cylinder(..., color=(r, g, b))``).

        Args:
            seed: Random seed.
        """
        warnings.warn(
            "``SensorTiledCamera.utils.assign_random_colors_per_shape`` is deprecated. Use shape colors instead (e.g. ``builder.add_shape_cylinder(..., color=(r, g, b))``).",
            category=DeprecationWarning,
            stacklevel=2,
        )

        colors = np.random.default_rng(seed).random((self.__render_context.shape_count_total, 3)) * 0.5 + 0.5
        self.__render_context.shape_colors = wp.array(colors, dtype=wp.vec3f, device=self.__render_context.device)

    def create_default_light(self, enable_shadows: bool = True, direction: wp.vec3f | None = None):
        """Create a default directional light oriented at ``(-1, 1, -1)``.

        Args:
            enable_shadows: Enable shadow casting for this light.
            direction: Normalized light direction. If ``None``, defaults to
                (normalized ``(-1, 1, -1)``).
        """
        self.__render_context.config.enable_shadows = enable_shadows
        self.__render_context.lights_active = wp.array([True], dtype=wp.bool, device=self.__render_context.device)
        self.__render_context.lights_type = wp.array(
            [RenderLightType.DIRECTIONAL], dtype=wp.int32, device=self.__render_context.device
        )
        self.__render_context.lights_cast_shadow = wp.array([True], dtype=wp.bool, device=self.__render_context.device)
        self.__render_context.lights_position = wp.array(
            [wp.vec3f(0.0)], dtype=wp.vec3f, device=self.__render_context.device
        )
        self.__render_context.lights_orientation = wp.array(
            [direction if direction is not None else wp.vec3f(-0.57735026, 0.57735026, -0.57735026)],
            dtype=wp.vec3f,
            device=self.__render_context.device,
        )

    def assign_checkerboard_material_to_all_shapes(self, resolution: int = 64, checker_size: int = 32):
        """Assign a gray checkerboard texture material to all shapes.
        Creates a gray checkerboard pattern texture and applies it to all shapes
        in the scene.

        Args:
            resolution: Texture resolution in pixels (square texture).
            checker_size: Size of each checkerboard square in pixels.
        """
        checkerboard = (
            (np.arange(resolution) // checker_size)[:, None] + (np.arange(resolution) // checker_size)
        ) % 2 == 0

        pixels = np.where(checkerboard, 0xFF808080, 0xFFBFBFBF).astype(np.uint32)

        texture_ids = np.full(self.__render_context.shape_count_total, fill_value=0, dtype=np.int32)

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

        self.__render_context.config.enable_textures = True
        self.__render_context.texture_data = wp.array(
            [self.__checkerboard_data], dtype=TextureData, device=self.__render_context.device
        )
        self.__render_context.shape_texture_ids = wp.array(
            texture_ids, dtype=wp.int32, device=self.__render_context.device
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
        elif worlds_per_row == 0:
            # Older callers passed 0 to mean "auto layout" because the original
            # check was a falsy test. Preserve that behavior with a deprecation
            # warning so we can require >=1 in a future release.
            warnings.warn(
                "worlds_per_row=0 is deprecated; pass None for auto layout.",
                category=DeprecationWarning,
                stacklevel=3,
            )
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
