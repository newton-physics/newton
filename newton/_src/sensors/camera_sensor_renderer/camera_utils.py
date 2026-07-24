# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any, TypeAlias

import warp as wp

if TYPE_CHECKING:
    from pxr import Usd, UsdGeom

    UsdCameraLike: TypeAlias = Usd.Prim | UsdGeom.Camera
    UsdTime: TypeAlias = Usd.TimeCode | float
else:
    UsdCameraLike: TypeAlias = Any
    UsdTime: TypeAlias = Any

UsdCameraInput: TypeAlias = UsdCameraLike


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


def _normalize_usd_camera(camera: UsdCameraInput) -> Any:
    try:
        from pxr import Usd, UsdGeom
    except ImportError as e:
        raise ImportError("USD camera ray helpers require the pxr USD Python modules.") from e

    if isinstance(camera, UsdGeom.Camera):
        usd_camera = camera
        prim = usd_camera.GetPrim()
    elif isinstance(camera, Usd.Prim):
        prim = camera
        if not prim.IsValid():
            raise TypeError("Expected a valid UsdGeom.Camera prim.")
        usd_camera = UsdGeom.Camera(prim)
    else:
        raise TypeError("Expected a single UsdGeom.Camera or Usd.Prim.")

    if not prim.IsValid():
        raise TypeError("Expected a valid UsdGeom.Camera prim.")
    if not prim.IsA(UsdGeom.Camera):
        raise TypeError(f"Expected a UsdGeom.Camera prim, got {prim.GetPath()!r}.")
    return usd_camera


def compute_camera_rays_usd_pinhole(
    width: int,
    height: int,
    camera: UsdCameraInput,
    *,
    device: wp.Device,
    time: UsdTime | None = None,
    out_rays: wp.array3d[wp.vec3f],
) -> wp.array3d[wp.vec3f]:
    time_code = _coerce_usd_time(time)
    usd_camera = _normalize_usd_camera(camera)
    projection = str(usd_camera.GetProjectionAttr().Get(time_code))
    if projection != "perspective":
        prim = usd_camera.GetPrim()
        raise NotImplementedError(f"USD camera {prim.GetPath()} uses unsupported projection {projection!r}.")

    wp.launch(
        kernel=compute_camera_rays_pinhole_from_aperture_kernel,
        dim=(height, width),
        inputs=[
            width,
            height,
            float(usd_camera.GetFocalLengthAttr().Get(time_code)),
            float(usd_camera.GetHorizontalApertureAttr().Get(time_code)),
            float(usd_camera.GetVerticalApertureAttr().Get(time_code)),
            float(usd_camera.GetHorizontalApertureOffsetAttr().Get(time_code)),
            float(usd_camera.GetVerticalApertureOffsetAttr().Get(time_code)),
            out_rays,
        ],
        device=device,
    )

    return out_rays


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
    camera_fov: wp.float32,
    out_rays: wp.array3d[wp.vec3f],
):
    py, px = wp.tid()
    aspect_ratio = float(width) / float(height)
    u = (float(px) + 0.5) / float(width) - 0.5
    v = (float(py) + 0.5) / float(height) - 0.5
    h = wp.tan(camera_fov / 2.0)
    ray_direction_camera_space = wp.vec3f(u * 2.0 * h * aspect_ratio, -v * 2.0 * h, -1.0)
    out_rays[py, px, 0] = wp.vec3f(0.0)
    out_rays[py, px, 1] = wp.normalize(ray_direction_camera_space)


@wp.kernel(enable_backward=False)
def compute_camera_rays_pinhole_from_aperture_kernel(
    width: int,
    height: int,
    focal_length: wp.float32,
    horizontal_aperture: wp.float32,
    vertical_aperture: wp.float32,
    horizontal_aperture_offset: wp.float32,
    vertical_aperture_offset: wp.float32,
    out_rays: wp.array3d[wp.vec3f],
):
    py, px = wp.tid()
    u = (float(px) + 0.5) / float(width)
    v = (float(py) + 0.5) / float(height)
    film_x = (u - 0.5) * horizontal_aperture + horizontal_aperture_offset
    film_y = (0.5 - v) * vertical_aperture + vertical_aperture_offset
    ray_direction_camera_space = wp.vec3f(film_x / focal_length, film_y / focal_length, -1.0)
    out_rays[py, px, 0] = wp.vec3f(0.0)
    out_rays[py, px, 1] = wp.normalize(ray_direction_camera_space)


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
    out_rays: wp.array3d[wp.vec3f],
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

    out_rays[py, px, 0] = wp.vec3f(0.0)
    out_rays[py, px, 1] = ray_direction_camera_space


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
    out_rays: wp.array3d[wp.vec3f],
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

    out_rays[py, px, 0] = wp.vec3f(0.0)
    out_rays[py, px, 1] = ray_direction_camera_space


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
    out_rays: wp.array3d[wp.vec3f],
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

    out_rays[py, px, 0] = wp.vec3f(0.0)
    out_rays[py, px, 1] = ray_direction_camera_space
