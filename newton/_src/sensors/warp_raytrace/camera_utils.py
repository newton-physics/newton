# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math
import numbers
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, TypeAlias

import numpy as np
import warp as wp

if TYPE_CHECKING:
    from pxr import Usd, UsdGeom

    UsdCameraLike: TypeAlias = Usd.Prim | UsdGeom.Camera
    UsdTime: TypeAlias = Usd.TimeCode | float
else:
    UsdCameraLike: TypeAlias = Any
    UsdTime: TypeAlias = Any

UsdCameraInput: TypeAlias = UsdCameraLike | Sequence[UsdCameraLike]
UsdCameraGridInput: TypeAlias = UsdCameraInput | Sequence[Sequence[UsdCameraLike]]


def _is_camera_sequence(cameras: Any) -> bool:
    return isinstance(cameras, Sequence) and not isinstance(cameras, (str, bytes, bytearray))


def _camera_param_count(param: Any) -> int:
    if isinstance(param, numbers.Real):
        return 1
    if isinstance(param, list | tuple | np.ndarray):
        return int(np.asarray(param).size)
    return int(param.size)


def _camera_param_array(
    name: str,
    param: Any,
    camera_count: int,
    device: wp.Device,
) -> wp.array[wp.float32]:
    if isinstance(param, numbers.Real):
        return wp.full((camera_count,), value=float(param), dtype=wp.float32, device=device)

    if isinstance(param, list | tuple | np.ndarray):
        values = np.asarray(param, dtype=np.float32).reshape(-1)
        if values.size == camera_count:
            return wp.array(values, dtype=wp.float32, device=device)
        if values.size == 1:
            return wp.full((camera_count,), value=float(values[0]), dtype=wp.float32, device=device)
        raise ValueError(f"{name} must have length 1 or {camera_count}.")

    if param.size == 1:
        value = float(param.numpy().reshape(-1)[0])
        return wp.full((camera_count,), value=value, dtype=wp.float32, device=device)
    if param.size != camera_count:
        raise ValueError(f"{name} must have length 1 or {camera_count}.")
    if param.dtype != wp.float32 or param.device != device:
        return wp.array(param.numpy().reshape(-1).astype(np.float32), dtype=wp.float32, device=device)
    return param


def _validate_camera_ray_output(
    width: int,
    height: int,
    camera_count: int,
    out_rays: wp.array4d[wp.vec3f] | None,
    camera_index: int,
    device: wp.Device,
) -> tuple[wp.array4d[wp.vec3f], int]:
    if width <= 0 or height <= 0:
        raise ValueError("width and height must be positive.")

    camera_index = int(camera_index)
    if camera_index < 0:
        raise ValueError("camera_index must be non-negative.")

    if out_rays is None:
        out_rays = wp.empty((camera_count, height, width, 2), dtype=wp.vec3f, device=device)
        camera_index = 0
    elif (
        out_rays.shape[0] < camera_index + camera_count
        or out_rays.shape[1] != height
        or out_rays.shape[2] != width
        or out_rays.shape[3] != 2
    ):
        raise ValueError("out_rays must have shape (out_camera_count, height, width, 2) with enough camera slots.")

    return out_rays, camera_index


def _coerce_usd_time(time: Any) -> Any:
    try:
        from pxr import Usd
    except ImportError as e:
        raise ImportError("USD camera ray helpers require the pxr USD Python modules.") from e

    if time is None:
        return Usd.TimeCode.Default()
    if isinstance(time, Usd.TimeCode):
        return time
    return Usd.TimeCode(float(time))


def _normalize_usd_cameras(cameras: UsdCameraInput) -> list[Any]:
    try:
        from pxr import Usd, UsdGeom
    except ImportError as e:
        raise ImportError("USD camera ray helpers require the pxr USD Python modules.") from e

    if _is_camera_sequence(cameras):
        camera_items = list(cameras)
    else:
        camera_items = [cameras]

    if not camera_items:
        raise ValueError("At least one USD camera is required.")

    usd_cameras = []
    for camera in camera_items:
        if isinstance(camera, UsdGeom.Camera):
            usd_camera = camera
            prim = usd_camera.GetPrim()
        elif isinstance(camera, Usd.Prim):
            prim = camera
            if not prim.IsValid():
                raise TypeError("Expected a valid UsdGeom.Camera prim.")
            usd_camera = UsdGeom.Camera(prim)
        else:
            raise TypeError("Expected a UsdGeom.Camera or Usd.Prim.")

        if not prim.IsValid():
            raise TypeError("Expected a valid UsdGeom.Camera prim.")
        if not prim.IsA(UsdGeom.Camera):
            raise TypeError(f"Expected a UsdGeom.Camera prim, got {prim.GetPath()!r}.")
        usd_cameras.append(usd_camera)

    return usd_cameras


def compute_camera_rays_usd_pinhole(
    width: int,
    height: int,
    cameras: UsdCameraInput,
    *,
    device: wp.Device,
    time: UsdTime | None = None,
    out_rays: wp.array4d[wp.vec3f] | None = None,
    camera_index: int = 0,
) -> wp.array4d[wp.vec3f]:
    time_code = _coerce_usd_time(time)
    usd_cameras = _normalize_usd_cameras(cameras)
    camera_count = len(usd_cameras)
    out_rays, camera_index = _validate_camera_ray_output(width, height, camera_count, out_rays, camera_index, device)

    focal_lengths = []
    horizontal_apertures = []
    vertical_apertures = []
    horizontal_aperture_offsets = []
    vertical_aperture_offsets = []
    for usd_camera in usd_cameras:
        projection = str(usd_camera.GetProjectionAttr().Get(time_code))
        if projection != "perspective":
            prim = usd_camera.GetPrim()
            raise NotImplementedError(f"USD camera {prim.GetPath()} uses unsupported projection {projection!r}.")

        focal_lengths.append(float(usd_camera.GetFocalLengthAttr().Get(time_code)))
        horizontal_apertures.append(float(usd_camera.GetHorizontalApertureAttr().Get(time_code)))
        vertical_apertures.append(float(usd_camera.GetVerticalApertureAttr().Get(time_code)))
        horizontal_aperture_offsets.append(float(usd_camera.GetHorizontalApertureOffsetAttr().Get(time_code)))
        vertical_aperture_offsets.append(float(usd_camera.GetVerticalApertureOffsetAttr().Get(time_code)))

    wp.launch(
        kernel=compute_camera_rays_pinhole_from_aperture_kernel,
        dim=(camera_count, height, width),
        inputs=[
            width,
            height,
            wp.array(focal_lengths, dtype=wp.float32, device=device),
            wp.array(horizontal_apertures, dtype=wp.float32, device=device),
            wp.array(vertical_apertures, dtype=wp.float32, device=device),
            wp.array(horizontal_aperture_offsets, dtype=wp.float32, device=device),
            wp.array(vertical_aperture_offsets, dtype=wp.float32, device=device),
            camera_index,
            out_rays,
        ],
        device=device,
    )

    return out_rays


def compute_camera_transforms_usd(
    cameras: UsdCameraGridInput,
    *,
    world_count: int,
    device: wp.Device,
    target_up_axis: Any | None = None,
    time: UsdTime | None = None,
    xform: Any | None = None,
) -> wp.array2d[wp.transformf]:
    try:
        from pxr import UsdGeom
    except ImportError as e:
        raise ImportError("USD camera ray helpers require the pxr USD Python modules.") from e

    from ...core import Axis, quat_between_axes  # noqa: PLC0415
    from ...usd.utils import get_transform  # noqa: PLC0415

    time_code = _coerce_usd_time(time)
    xform_cache = UsdGeom.XformCache(time_code)
    scene_xform = wp.transform(*xform) if xform is not None else None

    def world_transform(usd_camera: Any) -> wp.transformf:
        transform = get_transform(usd_camera.GetPrim(), local=False, xform_cache=xform_cache)
        if target_up_axis is not None:
            stage_up_axis = Axis.from_string(str(UsdGeom.GetStageUpAxis(usd_camera.GetPrim().GetStage())))
            axis_xform = wp.transform(wp.vec3(0.0), quat_between_axes(stage_up_axis, target_up_axis))
            transform = axis_xform * transform
        if scene_xform is not None:
            transform = scene_xform * transform
        return transform

    is_per_world = _is_camera_sequence(cameras) and len(cameras) > 0 and _is_camera_sequence(cameras[0])

    if is_per_world:
        if len(cameras) != world_count:
            raise ValueError(
                f"compute_camera_transforms_usd: per-world cameras outer dimension {len(cameras)} "
                f"must match world_count {world_count}."
            )
        rows = [_normalize_usd_cameras(row) for row in cameras]
        camera_count = len(rows[0])
        for world_index, row in enumerate(rows):
            if len(row) != camera_count:
                raise ValueError(
                    f"compute_camera_transforms_usd: per-world cameras row {world_index} has "
                    f"{len(row)} cameras, expected {camera_count}."
                )
        transforms = [
            [world_transform(rows[world_index][camera_index]) for world_index in range(world_count)]
            for camera_index in range(camera_count)
        ]
    else:
        usd_cameras = _normalize_usd_cameras(cameras)
        transforms = [[world_transform(usd_camera)] * world_count for usd_camera in usd_cameras]

    return wp.array(
        transforms,
        dtype=wp.transformf,
        device=device,
    )


@wp.func
def _opencv_fisheye_radius(theta: wp.float32, k0: wp.float32, k1: wp.float32, k2: wp.float32, k3: wp.float32):
    theta2 = theta * theta
    theta4 = theta2 * theta2
    theta6 = theta4 * theta2
    theta8 = theta4 * theta4
    return theta * (1.0 + k0 * theta2 + k1 * theta4 + k2 * theta6 + k3 * theta8)


@wp.func
def _ftheta_radius(
    theta: wp.float32,
    k0: wp.float32,
    k1: wp.float32,
    k2: wp.float32,
    k3: wp.float32,
    k4: wp.float32,
):
    theta2 = theta * theta
    theta3 = theta2 * theta
    theta4 = theta2 * theta2
    return k0 + k1 * theta + k2 * theta2 + k3 * theta3 + k4 * theta4


@wp.func
def _kannala_brandt_k3_radius(
    theta: wp.float32,
    k0: wp.float32,
    k1: wp.float32,
    k2: wp.float32,
    k3: wp.float32,
):
    theta2 = theta * theta
    theta3 = theta2 * theta
    theta5 = theta3 * theta2
    theta7 = theta5 * theta2
    return k0 * theta + k1 * theta3 + k2 * theta5 + k3 * theta7


@wp.func
def _solve_opencv_fisheye_theta(
    radius: wp.float32,
    k0: wp.float32,
    k1: wp.float32,
    k2: wp.float32,
    k3: wp.float32,
    max_theta: wp.float32,
):
    if radius <= 1.0e-7:
        return wp.float32(0.0)

    # This endpoint check and the binary search assume r(theta) is monotonic.
    max_radius = _opencv_fisheye_radius(max_theta, k0, k1, k2, k3)
    if radius > max_radius + 1.0e-5:
        return wp.float32(-1.0)

    lo = wp.float32(0.0)
    hi = max_theta
    for _i in range(24):
        mid = (lo + hi) * 0.5
        if _opencv_fisheye_radius(mid, k0, k1, k2, k3) < radius:
            lo = mid
        else:
            hi = mid
    return (lo + hi) * 0.5


@wp.func
def _solve_ftheta_theta(
    radius: wp.float32,
    k0: wp.float32,
    k1: wp.float32,
    k2: wp.float32,
    k3: wp.float32,
    k4: wp.float32,
    max_theta: wp.float32,
):
    if radius <= 1.0e-7:
        return wp.float32(0.0)

    # When k0 != 0 the polynomial has a nonzero floor at theta=0 (r(0) = k0).
    # Pixels inside that central circle are undefined by the model; return theta=0 (forward).
    min_radius = _ftheta_radius(0.0, k0, k1, k2, k3, k4)
    if radius <= min_radius:
        return wp.float32(0.0)

    # This endpoint check and the binary search assume r(theta) is monotonic.
    max_radius = _ftheta_radius(max_theta, k0, k1, k2, k3, k4)
    if radius > max_radius + 1.0e-5:
        return wp.float32(-1.0)

    lo = wp.float32(0.0)
    hi = max_theta
    for _i in range(24):
        mid = (lo + hi) * 0.5
        if _ftheta_radius(mid, k0, k1, k2, k3, k4) < radius:
            lo = mid
        else:
            hi = mid
    return (lo + hi) * 0.5


@wp.func
def _solve_kannala_brandt_k3_theta(
    radius: wp.float32,
    k0: wp.float32,
    k1: wp.float32,
    k2: wp.float32,
    k3: wp.float32,
    max_theta: wp.float32,
):
    if radius <= 1.0e-7:
        return wp.float32(0.0)

    # This endpoint check and the binary search assume r(theta) is monotonic.
    max_radius = _kannala_brandt_k3_radius(max_theta, k0, k1, k2, k3)
    if radius > max_radius + 1.0e-5:
        return wp.float32(-1.0)

    lo = wp.float32(0.0)
    hi = max_theta
    for _i in range(24):
        mid = (lo + hi) * 0.5
        if _kannala_brandt_k3_radius(mid, k0, k1, k2, k3) < radius:
            lo = mid
        else:
            hi = mid
    return (lo + hi) * 0.5


@wp.func
def _fisheye_direction_from_theta(x: wp.float32, y: wp.float32, radius: wp.float32, theta: wp.float32):
    # Valid fisheye rays are unit-length by construction; zero is reserved for invalid rays.
    if theta < 0.0:
        return wp.vec3f(0.0)
    if radius <= 1.0e-7:
        return wp.vec3f(0.0, 0.0, -1.0)

    sin_theta = wp.sin(theta)
    return wp.vec3f((x / radius) * sin_theta, (y / radius) * sin_theta, -wp.cos(theta))


@wp.kernel(enable_backward=False)
def compute_camera_rays_pinhole(
    width: int,
    height: int,
    camera_fovs: wp.array[wp.float32],
    camera_index_start: int,
    out_rays: wp.array4d[wp.vec3f],
):
    camera_index, py, px = wp.tid()
    output_camera_index = camera_index_start + camera_index
    aspect_ratio = float(width) / float(height)
    u = (float(px) + 0.5) / float(width) - 0.5
    v = (float(py) + 0.5) / float(height) - 0.5
    h = wp.tan(camera_fovs[camera_index] / 2.0)
    ray_direction_camera_space = wp.vec3f(u * 2.0 * h * aspect_ratio, -v * 2.0 * h, -1.0)
    out_rays[output_camera_index, py, px, 0] = wp.vec3f(0.0)
    out_rays[output_camera_index, py, px, 1] = wp.normalize(ray_direction_camera_space)


@wp.kernel(enable_backward=False)
def compute_camera_rays_pinhole_from_aperture_kernel(
    width: int,
    height: int,
    focal_lengths: wp.array[wp.float32],
    horizontal_apertures: wp.array[wp.float32],
    vertical_apertures: wp.array[wp.float32],
    horizontal_aperture_offsets: wp.array[wp.float32],
    vertical_aperture_offsets: wp.array[wp.float32],
    camera_index_start: int,
    out_rays: wp.array4d[wp.vec3f],
):
    camera_index, py, px = wp.tid()
    output_camera_index = camera_index_start + camera_index
    u = (float(px) + 0.5) / float(width)
    v = (float(py) + 0.5) / float(height)
    film_x = (u - 0.5) * horizontal_apertures[camera_index] + horizontal_aperture_offsets[camera_index]
    film_y = (0.5 - v) * vertical_apertures[camera_index] + vertical_aperture_offsets[camera_index]
    focal_length = focal_lengths[camera_index]
    ray_direction_camera_space = wp.vec3f(film_x / focal_length, film_y / focal_length, -1.0)
    out_rays[output_camera_index, py, px, 0] = wp.vec3f(0.0)
    out_rays[output_camera_index, py, px, 1] = wp.normalize(ray_direction_camera_space)


@wp.kernel(enable_backward=False)
def compute_camera_rays_fisheye_opencv_kernel(
    width: int,
    height: int,
    image_width: wp.float32,
    image_height: wp.float32,
    fx: wp.float32,
    fy: wp.float32,
    cx: wp.float32,
    cy: wp.float32,
    k1: wp.float32,
    k2: wp.float32,
    k3: wp.float32,
    k4: wp.float32,
    max_fov: wp.float32,
    camera_index: int,
    out_rays: wp.array4d[wp.vec3f],
):
    py, px = wp.tid()
    u = ((float(px) + 0.5) / float(width)) * image_width
    v = ((float(py) + 0.5) / float(height)) * image_height
    x = (u - cx) / fx
    y = -(v - cy) / fy
    radius = wp.sqrt(x * x + y * y)
    theta = _solve_opencv_fisheye_theta(
        radius,
        k1,
        k2,
        k3,
        k4,
        wp.min(max_fov * wp.float32(0.5), wp.float32(math.pi)),
    )
    ray_direction_camera_space = _fisheye_direction_from_theta(x, y, radius, theta)

    out_rays[camera_index, py, px, 0] = wp.vec3f(0.0)
    out_rays[camera_index, py, px, 1] = ray_direction_camera_space


@wp.kernel(enable_backward=False)
def compute_camera_rays_fisheye_ftheta_kernel(
    width: int,
    height: int,
    nominal_width: wp.float32,
    nominal_height: wp.float32,
    optical_center_x: wp.float32,
    optical_center_y: wp.float32,
    k0: wp.float32,
    k1: wp.float32,
    k2: wp.float32,
    k3: wp.float32,
    k4: wp.float32,
    max_fov: wp.float32,
    camera_index: int,
    out_rays: wp.array4d[wp.vec3f],
):
    py, px = wp.tid()
    u = ((float(px) + 0.5) / float(width)) * nominal_width
    v = ((float(py) + 0.5) / float(height)) * nominal_height
    x = u - optical_center_x
    y = -(v - optical_center_y)
    radius = wp.sqrt(x * x + y * y)
    max_theta = wp.min(max_fov * 0.5, wp.float32(math.pi))
    theta = _solve_ftheta_theta(
        radius,
        k0,
        k1,
        k2,
        k3,
        k4,
        max_theta,
    )
    ray_direction_camera_space = _fisheye_direction_from_theta(x, y, radius, theta)

    out_rays[camera_index, py, px, 0] = wp.vec3f(0.0)
    out_rays[camera_index, py, px, 1] = ray_direction_camera_space


@wp.kernel(enable_backward=False)
def compute_camera_rays_fisheye_kannala_brandt_kernel(
    width: int,
    height: int,
    nominal_width: wp.float32,
    nominal_height: wp.float32,
    optical_center_x: wp.float32,
    optical_center_y: wp.float32,
    k0: wp.float32,
    k1: wp.float32,
    k2: wp.float32,
    k3: wp.float32,
    max_fov: wp.float32,
    camera_index: int,
    out_rays: wp.array4d[wp.vec3f],
):
    py, px = wp.tid()
    u = ((float(px) + 0.5) / float(width)) * nominal_width
    v = ((float(py) + 0.5) / float(height)) * nominal_height
    x = u - optical_center_x
    y = -(v - optical_center_y)
    radius = wp.sqrt(x * x + y * y)
    max_theta = wp.min(max_fov * 0.5, wp.float32(math.pi))
    theta = _solve_kannala_brandt_k3_theta(
        radius,
        k0,
        k1,
        k2,
        k3,
        max_theta,
    )
    ray_direction_camera_space = _fisheye_direction_from_theta(x, y, radius, theta)

    out_rays[camera_index, py, px, 0] = wp.vec3f(0.0)
    out_rays[camera_index, py, px, 1] = ray_direction_camera_space
