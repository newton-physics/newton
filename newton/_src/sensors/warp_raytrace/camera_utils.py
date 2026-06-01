# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math
from typing import Any

import numpy as np
import warp as wp

_USD_CAMERA_MODEL_PINHOLE = 0
_USD_CAMERA_MODEL_OPENCV_FISHEYE = 1
_USD_CAMERA_MODEL_FTHETA = 2
_USD_CAMERA_MODEL_KANNALA_BRANDT_K3 = 3

_USD_CAMERA_PARAM_COUNT = 18
_USD_CAMERA_PARAM_H_APERTURE = 0
_USD_CAMERA_PARAM_V_APERTURE = 1
_USD_CAMERA_PARAM_H_OFFSET = 2
_USD_CAMERA_PARAM_V_OFFSET = 3
_USD_CAMERA_PARAM_FOCAL_LENGTH = 4
_USD_CAMERA_PARAM_FX = 5
_USD_CAMERA_PARAM_FY = 6
_USD_CAMERA_PARAM_CX = 7
_USD_CAMERA_PARAM_CY = 8
_USD_CAMERA_PARAM_IMAGE_WIDTH = 9
_USD_CAMERA_PARAM_IMAGE_HEIGHT = 10
_USD_CAMERA_PARAM_K0 = 11
_USD_CAMERA_PARAM_K1 = 12
_USD_CAMERA_PARAM_K2 = 13
_USD_CAMERA_PARAM_K3 = 14
_USD_CAMERA_PARAM_K4 = 15
_USD_CAMERA_PARAM_MAX_THETA = 16

_PI = 3.141592653589793


def _coerce_usd_time(time: Any, Usd: Any) -> Any:
    if time is None:
        return Usd.TimeCode.Default()
    if isinstance(time, Usd.TimeCode):
        return time
    return Usd.TimeCode(float(time))


def _normalize_usd_cameras(cameras: Any, UsdGeom: Any) -> list[Any]:
    if isinstance(cameras, (list, tuple)):
        camera_items = list(cameras)
    else:
        camera_items = [cameras]

    if not camera_items:
        raise ValueError("At least one USD camera is required.")

    usd_cameras = []
    for camera in camera_items:
        if hasattr(camera, "GetProjectionAttr") and hasattr(camera, "GetPrim"):
            usd_camera = camera
        elif hasattr(camera, "IsA"):
            usd_camera = UsdGeom.Camera(camera)
        else:
            raise TypeError("Expected a UsdGeom.Camera or Usd.Prim.")

        prim = usd_camera.GetPrim()
        if not prim or not prim.IsValid() or not prim.IsA(UsdGeom.Camera):
            raise TypeError(f"Expected a UsdGeom.Camera prim, got {prim.GetPath() if prim else camera!r}.")
        usd_cameras.append(usd_camera)

    return usd_cameras


def _get_usd_attr(prim: Any, name: str, default: Any, time_code: Any) -> Any:
    attr = prim.GetAttribute(name)
    if not attr or not attr.IsValid():
        return default
    value = attr.Get(time_code)
    return default if value is None else value


def _get_usd_float_attr(prim: Any, name: str, default: float, time_code: Any) -> float:
    value = _get_usd_attr(prim, name, default, time_code)
    try:
        value = float(value)
    except (TypeError, ValueError):
        return default
    return value if math.isfinite(value) else default


def _get_usd_vec2_attr(prim: Any, name: str, default: tuple[float, float], time_code: Any) -> tuple[float, float]:
    value = _get_usd_attr(prim, name, default, time_code)
    try:
        arr = np.asarray(value, dtype=np.float32).reshape(-1)
    except (TypeError, ValueError):
        return default
    if arr.size < 2 or not np.isfinite(arr[:2]).all():
        return default
    return (float(arr[0]), float(arr[1]))


def _get_applied_api_schemas(prim: Any) -> list[str]:
    schemas = list(prim.GetAppliedSchemas())
    if schemas:
        return schemas

    listop = prim.GetMetadata("apiSchemas")
    if listop is None:
        return []
    return (
        list(getattr(listop, "prependedItems", []))
        + list(getattr(listop, "appendedItems", []))
        + list(getattr(listop, "explicitItems", []))
    )


def _has_authored_usd_attr(prim: Any, name: str) -> bool:
    attr = prim.GetAttribute(name)
    return bool(attr and attr.IsValid() and attr.HasAuthoredValue())


def _has_any_authored_usd_attr(prim: Any, names: tuple[str, ...]) -> bool:
    return any(_has_authored_usd_attr(prim, name) for name in names)


def _max_theta_from_fov(max_fov_degrees: float) -> float:
    if max_fov_degrees <= 0.0 or not math.isfinite(max_fov_degrees):
        return math.pi
    return min(math.radians(max_fov_degrees) * 0.5, math.pi)


def extract_usd_camera_ray_params(
    width: int,
    height: int,
    cameras: Any | list[Any] | tuple[Any, ...],
    *,
    time: Any | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    try:
        from pxr import Usd, UsdGeom
    except ImportError as e:
        raise ImportError("USD camera ray helpers require the pxr USD Python modules.") from e

    time_code = _coerce_usd_time(time, Usd)
    usd_cameras = _normalize_usd_cameras(cameras, UsdGeom)

    model_ids = np.empty(len(usd_cameras), dtype=np.int32)
    params = np.zeros((len(usd_cameras), _USD_CAMERA_PARAM_COUNT), dtype=np.float32)

    for camera_index, usd_camera in enumerate(usd_cameras):
        prim = usd_camera.GetPrim()
        lens_model = str(_get_usd_attr(prim, "omni:lensdistortion:model", "", time_code))
        applied_schemas = set(_get_applied_api_schemas(prim))

        has_opencv_fisheye = (
            lens_model == "opencvFisheye"
            or "OmniLensDistortionOpenCvFisheyeAPI" in applied_schemas
            or _has_any_authored_usd_attr(
                prim,
                (
                    "omni:lensdistortion:opencvFisheye:fx",
                    "omni:lensdistortion:opencvFisheye:fy",
                    "omni:lensdistortion:opencvFisheye:k1",
                ),
            )
        )
        has_ftheta = (
            lens_model == "ftheta"
            or "OmniLensDistortionFthetaAPI" in applied_schemas
            or _has_any_authored_usd_attr(
                prim,
                (
                    "omni:lensdistortion:ftheta:k0",
                    "omni:lensdistortion:ftheta:k1",
                    "omni:lensdistortion:ftheta:maxFov",
                ),
            )
        )
        has_kannala_brandt = (
            lens_model == "kannalaBrandtK3"
            or "OmniLensDistortionKannalaBrandtK3API" in applied_schemas
            or _has_any_authored_usd_attr(
                prim,
                (
                    "omni:lensdistortion:kannalaBrandtK3:k0",
                    "omni:lensdistortion:kannalaBrandtK3:k1",
                    "omni:lensdistortion:kannalaBrandtK3:maxFov",
                ),
            )
        )

        if has_opencv_fisheye:
            image_size = _get_usd_vec2_attr(
                prim, "omni:lensdistortion:opencvFisheye:imageSize", (2048.0, 1024.0), time_code
            )
            fx = _get_usd_float_attr(prim, "omni:lensdistortion:opencvFisheye:fx", 900.0, time_code)
            fy = _get_usd_float_attr(prim, "omni:lensdistortion:opencvFisheye:fy", 800.0, time_code)
            if fx <= 0.0 or fy <= 0.0 or image_size[0] <= 0.0 or image_size[1] <= 0.0:
                raise ValueError(f"USD camera {prim.GetPath()} has invalid OpenCV fisheye calibration.")

            model_ids[camera_index] = _USD_CAMERA_MODEL_OPENCV_FISHEYE
            params[camera_index, _USD_CAMERA_PARAM_FX] = fx
            params[camera_index, _USD_CAMERA_PARAM_FY] = fy
            params[camera_index, _USD_CAMERA_PARAM_CX] = _get_usd_float_attr(
                prim, "omni:lensdistortion:opencvFisheye:cx", 1024.0, time_code
            )
            params[camera_index, _USD_CAMERA_PARAM_CY] = _get_usd_float_attr(
                prim, "omni:lensdistortion:opencvFisheye:cy", 512.0, time_code
            )
            params[camera_index, _USD_CAMERA_PARAM_IMAGE_WIDTH] = image_size[0]
            params[camera_index, _USD_CAMERA_PARAM_IMAGE_HEIGHT] = image_size[1]
            params[camera_index, _USD_CAMERA_PARAM_K0] = _get_usd_float_attr(
                prim, "omni:lensdistortion:opencvFisheye:k1", 0.00245, time_code
            )
            params[camera_index, _USD_CAMERA_PARAM_K1] = _get_usd_float_attr(
                prim, "omni:lensdistortion:opencvFisheye:k2", 0.0, time_code
            )
            params[camera_index, _USD_CAMERA_PARAM_K2] = _get_usd_float_attr(
                prim, "omni:lensdistortion:opencvFisheye:k3", 0.0, time_code
            )
            params[camera_index, _USD_CAMERA_PARAM_K3] = _get_usd_float_attr(
                prim, "omni:lensdistortion:opencvFisheye:k4", 0.0, time_code
            )
            continue

        if has_ftheta:
            nominal_width = _get_usd_float_attr(
                prim, "omni:lensdistortion:ftheta:nominalWidth", float(width), time_code
            )
            nominal_height = _get_usd_float_attr(
                prim, "omni:lensdistortion:ftheta:nominalHeight", float(height), time_code
            )
            if nominal_width <= 0.0 or nominal_height <= 0.0:
                raise ValueError(f"USD camera {prim.GetPath()} has invalid F-theta calibration dimensions.")
            optical_center = _get_usd_vec2_attr(
                prim,
                "omni:lensdistortion:ftheta:opticalCenter",
                (nominal_width * 0.5, nominal_height * 0.5),
                time_code,
            )

            model_ids[camera_index] = _USD_CAMERA_MODEL_FTHETA
            params[camera_index, _USD_CAMERA_PARAM_IMAGE_WIDTH] = nominal_width
            params[camera_index, _USD_CAMERA_PARAM_IMAGE_HEIGHT] = nominal_height
            params[camera_index, _USD_CAMERA_PARAM_CX] = optical_center[0]
            params[camera_index, _USD_CAMERA_PARAM_CY] = optical_center[1]
            params[camera_index, _USD_CAMERA_PARAM_K0] = _get_usd_float_attr(
                prim, "omni:lensdistortion:ftheta:k0", 0.0, time_code
            )
            params[camera_index, _USD_CAMERA_PARAM_K1] = _get_usd_float_attr(
                prim, "omni:lensdistortion:ftheta:k1", 1.0, time_code
            )
            params[camera_index, _USD_CAMERA_PARAM_K2] = _get_usd_float_attr(
                prim, "omni:lensdistortion:ftheta:k2", 0.0, time_code
            )
            params[camera_index, _USD_CAMERA_PARAM_K3] = _get_usd_float_attr(
                prim, "omni:lensdistortion:ftheta:k3", 0.0, time_code
            )
            params[camera_index, _USD_CAMERA_PARAM_K4] = _get_usd_float_attr(
                prim, "omni:lensdistortion:ftheta:k4", 0.0, time_code
            )
            params[camera_index, _USD_CAMERA_PARAM_MAX_THETA] = _max_theta_from_fov(
                _get_usd_float_attr(prim, "omni:lensdistortion:ftheta:maxFov", 0.0, time_code)
            )
            continue

        if has_kannala_brandt:
            nominal_width = _get_usd_float_attr(
                prim, "omni:lensdistortion:kannalaBrandtK3:nominalWidth", float(width), time_code
            )
            nominal_height = _get_usd_float_attr(
                prim, "omni:lensdistortion:kannalaBrandtK3:nominalHeight", float(height), time_code
            )
            if nominal_width <= 0.0 or nominal_height <= 0.0:
                raise ValueError(f"USD camera {prim.GetPath()} has invalid Kannala-Brandt K3 dimensions.")
            optical_center = _get_usd_vec2_attr(
                prim,
                "omni:lensdistortion:kannalaBrandtK3:opticalCenter",
                (nominal_width * 0.5, nominal_height * 0.5),
                time_code,
            )

            model_ids[camera_index] = _USD_CAMERA_MODEL_KANNALA_BRANDT_K3
            params[camera_index, _USD_CAMERA_PARAM_IMAGE_WIDTH] = nominal_width
            params[camera_index, _USD_CAMERA_PARAM_IMAGE_HEIGHT] = nominal_height
            params[camera_index, _USD_CAMERA_PARAM_CX] = optical_center[0]
            params[camera_index, _USD_CAMERA_PARAM_CY] = optical_center[1]
            params[camera_index, _USD_CAMERA_PARAM_K0] = _get_usd_float_attr(
                prim, "omni:lensdistortion:kannalaBrandtK3:k0", 1.0, time_code
            )
            params[camera_index, _USD_CAMERA_PARAM_K1] = _get_usd_float_attr(
                prim, "omni:lensdistortion:kannalaBrandtK3:k1", 0.0, time_code
            )
            params[camera_index, _USD_CAMERA_PARAM_K2] = _get_usd_float_attr(
                prim, "omni:lensdistortion:kannalaBrandtK3:k2", 0.0, time_code
            )
            params[camera_index, _USD_CAMERA_PARAM_K3] = _get_usd_float_attr(
                prim, "omni:lensdistortion:kannalaBrandtK3:k3", 0.0, time_code
            )
            params[camera_index, _USD_CAMERA_PARAM_MAX_THETA] = _max_theta_from_fov(
                _get_usd_float_attr(prim, "omni:lensdistortion:kannalaBrandtK3:maxFov", 0.0, time_code)
            )
            continue

        projection = str(usd_camera.GetProjectionAttr().Get(time_code))
        if projection != "perspective":
            raise NotImplementedError(f"USD camera {prim.GetPath()} uses unsupported projection {projection!r}.")

        h_aperture = float(usd_camera.GetHorizontalApertureAttr().Get(time_code))
        v_aperture = float(usd_camera.GetVerticalApertureAttr().Get(time_code))
        focal_length = float(usd_camera.GetFocalLengthAttr().Get(time_code))
        if h_aperture <= 0.0 or v_aperture <= 0.0 or focal_length <= 0.0:
            raise ValueError(f"USD camera {prim.GetPath()} has invalid pinhole aperture or focal length.")

        model_ids[camera_index] = _USD_CAMERA_MODEL_PINHOLE
        params[camera_index, _USD_CAMERA_PARAM_H_APERTURE] = h_aperture
        params[camera_index, _USD_CAMERA_PARAM_V_APERTURE] = v_aperture
        params[camera_index, _USD_CAMERA_PARAM_H_OFFSET] = float(
            usd_camera.GetHorizontalApertureOffsetAttr().Get(time_code)
        )
        params[camera_index, _USD_CAMERA_PARAM_V_OFFSET] = float(
            usd_camera.GetVerticalApertureOffsetAttr().Get(time_code)
        )
        params[camera_index, _USD_CAMERA_PARAM_FOCAL_LENGTH] = focal_length

    return model_ids, params


def compute_usd_camera_transforms(
    cameras: Any | list[Any] | tuple[Any, ...],
    *,
    world_count: int,
    device: wp.Device,
    time: Any | None = None,
) -> wp.array2d[wp.transformf]:
    try:
        from pxr import Usd, UsdGeom
    except ImportError as e:
        raise ImportError("USD camera ray helpers require the pxr USD Python modules.") from e

    from ...usd.utils import get_transform  # noqa: PLC0415

    time_code = _coerce_usd_time(time, Usd)
    usd_cameras = _normalize_usd_cameras(cameras, UsdGeom)
    xform_cache = UsdGeom.XformCache(time_code)
    transforms = []
    for usd_camera in usd_cameras:
        transform = get_transform(usd_camera.GetPrim(), local=False, xform_cache=xform_cache)
        transforms.append([transform for _world_index in range(world_count)])

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

    max_radius = _opencv_fisheye_radius(max_theta, k0, k1, k2, k3)
    if radius > max_radius + 1.0e-5:
        return wp.float32(-1.0)

    lo = wp.float32(0.0)
    hi = max_theta
    for _i in range(32):
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

    min_radius = _ftheta_radius(0.0, k0, k1, k2, k3, k4)
    if radius <= min_radius:
        return wp.float32(0.0)

    max_radius = _ftheta_radius(max_theta, k0, k1, k2, k3, k4)
    if radius > max_radius + 1.0e-5:
        return wp.float32(-1.0)

    lo = wp.float32(0.0)
    hi = max_theta
    for _i in range(32):
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

    max_radius = _kannala_brandt_k3_radius(max_theta, k0, k1, k2, k3)
    if radius > max_radius + 1.0e-5:
        return wp.float32(-1.0)

    lo = wp.float32(0.0)
    hi = max_theta
    for _i in range(32):
        mid = (lo + hi) * 0.5
        if _kannala_brandt_k3_radius(mid, k0, k1, k2, k3) < radius:
            lo = mid
        else:
            hi = mid
    return (lo + hi) * 0.5


@wp.func
def _fisheye_direction_from_theta(x: wp.float32, y: wp.float32, radius: wp.float32, theta: wp.float32):
    if theta < 0.0:
        return wp.vec3f(0.0)
    if radius <= 1.0e-7:
        return wp.vec3f(0.0, 0.0, -1.0)

    sin_theta = wp.sin(theta)
    return wp.vec3f((x / radius) * sin_theta, (y / radius) * sin_theta, -wp.cos(theta))


@wp.kernel(enable_backward=False)
def compute_usd_camera_rays_kernel(
    width: int,
    height: int,
    camera_model_ids: wp.array[wp.int32],
    camera_params: wp.array2d[wp.float32],
    out_rays: wp.array4d[wp.vec3f],
):
    camera_index, py, px = wp.tid()
    model_id = camera_model_ids[camera_index]

    ray_direction_camera_space = wp.vec3f(0.0)

    if model_id == _USD_CAMERA_MODEL_PINHOLE:
        h_aperture = camera_params[camera_index, _USD_CAMERA_PARAM_H_APERTURE]
        v_aperture = camera_params[camera_index, _USD_CAMERA_PARAM_V_APERTURE]
        h_offset = camera_params[camera_index, _USD_CAMERA_PARAM_H_OFFSET]
        v_offset = camera_params[camera_index, _USD_CAMERA_PARAM_V_OFFSET]
        focal_length = camera_params[camera_index, _USD_CAMERA_PARAM_FOCAL_LENGTH]

        u = (float(px) + 0.5) / float(width)
        v = (float(py) + 0.5) / float(height)
        film_x = (u - 0.5) * h_aperture + h_offset
        film_y = (0.5 - v) * v_aperture + v_offset
        ray_direction_camera_space = wp.normalize(wp.vec3f(film_x / focal_length, film_y / focal_length, -1.0))

    elif model_id == _USD_CAMERA_MODEL_OPENCV_FISHEYE:
        fx = camera_params[camera_index, _USD_CAMERA_PARAM_FX]
        fy = camera_params[camera_index, _USD_CAMERA_PARAM_FY]
        cx = camera_params[camera_index, _USD_CAMERA_PARAM_CX]
        cy = camera_params[camera_index, _USD_CAMERA_PARAM_CY]
        image_width = camera_params[camera_index, _USD_CAMERA_PARAM_IMAGE_WIDTH]
        image_height = camera_params[camera_index, _USD_CAMERA_PARAM_IMAGE_HEIGHT]

        u = ((float(px) + 0.5) / float(width)) * image_width
        v = ((float(py) + 0.5) / float(height)) * image_height
        x = (u - cx) / fx
        y = -(v - cy) / fy
        radius = wp.sqrt(x * x + y * y)
        theta = _solve_opencv_fisheye_theta(
            radius,
            camera_params[camera_index, _USD_CAMERA_PARAM_K0],
            camera_params[camera_index, _USD_CAMERA_PARAM_K1],
            camera_params[camera_index, _USD_CAMERA_PARAM_K2],
            camera_params[camera_index, _USD_CAMERA_PARAM_K3],
            wp.float32(_PI),
        )
        ray_direction_camera_space = _fisheye_direction_from_theta(x, y, radius, theta)

    elif model_id == _USD_CAMERA_MODEL_FTHETA:
        image_width = camera_params[camera_index, _USD_CAMERA_PARAM_IMAGE_WIDTH]
        image_height = camera_params[camera_index, _USD_CAMERA_PARAM_IMAGE_HEIGHT]
        cx = camera_params[camera_index, _USD_CAMERA_PARAM_CX]
        cy = camera_params[camera_index, _USD_CAMERA_PARAM_CY]

        u = ((float(px) + 0.5) / float(width)) * image_width
        v = ((float(py) + 0.5) / float(height)) * image_height
        x = u - cx
        y = -(v - cy)
        radius = wp.sqrt(x * x + y * y)
        theta = _solve_ftheta_theta(
            radius,
            camera_params[camera_index, _USD_CAMERA_PARAM_K0],
            camera_params[camera_index, _USD_CAMERA_PARAM_K1],
            camera_params[camera_index, _USD_CAMERA_PARAM_K2],
            camera_params[camera_index, _USD_CAMERA_PARAM_K3],
            camera_params[camera_index, _USD_CAMERA_PARAM_K4],
            camera_params[camera_index, _USD_CAMERA_PARAM_MAX_THETA],
        )
        ray_direction_camera_space = _fisheye_direction_from_theta(x, y, radius, theta)

    elif model_id == _USD_CAMERA_MODEL_KANNALA_BRANDT_K3:
        image_width = camera_params[camera_index, _USD_CAMERA_PARAM_IMAGE_WIDTH]
        image_height = camera_params[camera_index, _USD_CAMERA_PARAM_IMAGE_HEIGHT]
        cx = camera_params[camera_index, _USD_CAMERA_PARAM_CX]
        cy = camera_params[camera_index, _USD_CAMERA_PARAM_CY]

        u = ((float(px) + 0.5) / float(width)) * image_width
        v = ((float(py) + 0.5) / float(height)) * image_height
        x = u - cx
        y = -(v - cy)
        radius = wp.sqrt(x * x + y * y)
        theta = _solve_kannala_brandt_k3_theta(
            radius,
            camera_params[camera_index, _USD_CAMERA_PARAM_K0],
            camera_params[camera_index, _USD_CAMERA_PARAM_K1],
            camera_params[camera_index, _USD_CAMERA_PARAM_K2],
            camera_params[camera_index, _USD_CAMERA_PARAM_K3],
            camera_params[camera_index, _USD_CAMERA_PARAM_MAX_THETA],
        )
        ray_direction_camera_space = _fisheye_direction_from_theta(x, y, radius, theta)

    out_rays[camera_index, py, px, 0] = wp.vec3f(0.0)
    out_rays[camera_index, py, px, 1] = ray_direction_camera_space
