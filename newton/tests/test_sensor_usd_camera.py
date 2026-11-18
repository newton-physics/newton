# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import math
import unittest

import numpy as np
import warp as wp

from newton.sensors import SensorCamera
from newton.tests.unittest_utils import requires

try:
    from pxr import Gf, Usd, UsdGeom
except ImportError:
    Gf = None
    Usd = None
    UsdGeom = None


def _make_camera():
    stage = Usd.Stage.CreateInMemory()
    camera = UsdGeom.Camera.Define(stage, "/World/Camera")
    return stage, camera


def _forward_pinhole_opencv(x: float, y: float, coefficients: dict[str, float]) -> tuple[float, float]:
    r2 = x * x + y * y
    r4 = r2 * r2
    r6 = r4 * r2
    radial = (1.0 + coefficients["k1"] * r2 + coefficients["k2"] * r4 + coefficients["k3"] * r6) / (
        1.0 + coefficients["k4"] * r2 + coefficients["k5"] * r4 + coefficients["k6"] * r6
    )
    x_distorted = (
        x * radial
        + 2.0 * coefficients["p1"] * x * y
        + coefficients["p2"] * (r2 + 2.0 * x * x)
        + coefficients["s1"] * r2
        + coefficients["s2"] * r4
    )
    y_distorted = (
        y * radial
        + coefficients["p1"] * (r2 + 2.0 * y * y)
        + 2.0 * coefficients["p2"] * x * y
        + coefficients["s3"] * r2
        + coefficients["s4"] * r4
    )
    return x_distorted, y_distorted


def _ray_to_opencv_normalized(direction: np.ndarray) -> tuple[float, float]:
    return float(direction[0] / -direction[2]), float(direction[1] / direction[2])


class TestSensorCameraRays(unittest.TestCase):
    @requires("pxr.Usd")
    def test_sensor_camera_usd_pinhole_rays_match_aperture_form(self):
        """Verify SensorCamera.compute_camera_rays_pinhole_usd reads USD intrinsics and matches the aperture form."""
        width, height = 5, 3
        _stage, camera = _make_camera()
        camera.GetProjectionAttr().Set(UsdGeom.Tokens.perspective)
        camera.GetFocalLengthAttr().Set(1.5)
        camera.GetHorizontalApertureAttr().Set(2.0)
        camera.GetVerticalApertureAttr().Set(1.0)
        camera.GetHorizontalApertureOffsetAttr().Set(0.1)
        camera.GetVerticalApertureOffsetAttr().Set(0.2)
        expected = SensorCamera.compute_camera_rays_pinhole(
            width,
            height,
            focal_length=1.5,
            horizontal_aperture=2.0,
            vertical_aperture=1.0,
            horizontal_aperture_offset=0.1,
            vertical_aperture_offset=0.2,
            device="cpu",
        ).numpy()

        got_prim = SensorCamera.compute_camera_rays_pinhole_usd(width, height, camera.GetPrim(), device="cpu")
        got_camera = SensorCamera.compute_camera_rays_pinhole_usd(width, height, camera, device="cpu")
        expected_multisample = SensorCamera.compute_camera_rays_pinhole(
            width,
            height,
            focal_length=1.5,
            horizontal_aperture=2.0,
            vertical_aperture=1.0,
            horizontal_aperture_offset=0.1,
            vertical_aperture_offset=0.2,
            sample_count=4,
            device="cpu",
        )
        got_multisample = SensorCamera.compute_camera_rays_pinhole_usd(
            width, height, camera, sample_count=4, device="cpu"
        )

        self.assertEqual(got_prim.shape, (height, width, 1, 2))
        self.assertEqual(got_prim.dtype, wp.vec3f)
        np.testing.assert_allclose(got_prim.numpy(), expected, atol=1e-6)
        np.testing.assert_allclose(got_camera.numpy(), expected, atol=1e-6)
        np.testing.assert_allclose(got_multisample.numpy(), expected_multisample.numpy(), atol=1e-6)

    @requires("pxr.Usd")
    def test_sensor_camera_usd_pinhole_rejects_invalid_prim(self):
        """Verify SensorCamera.compute_camera_rays_pinhole_usd rejects a non-camera prim."""
        with self.assertRaisesRegex(TypeError, "Expected a valid UsdGeom.Camera prim"):
            SensorCamera.compute_camera_rays_pinhole_usd(1, 1, Usd.Prim(), device="cpu")

    def test_sensor_camera_opencv_pinhole_round_trips(self):
        """Verify SensorCamera.compute_camera_rays_pinhole_opencv inverts the distortion model."""
        width, height = 8, 6
        image_width, image_height = 640.0, 480.0
        fx, fy, cx, cy = 339.26592887, 338.82010626, 323.55809091, 250.27360914
        coefficients = {
            "k1": 0.1,
            "k2": -0.05,
            "k3": 0.01,
            "k4": 0.005,
            "k5": -0.002,
            "k6": 0.0005,
            "p1": 0.001,
            "p2": -0.002,
            "s1": 0.0005,
            "s2": -0.0002,
            "s3": 0.0003,
            "s4": -0.0001,
        }

        rays = SensorCamera.compute_camera_rays_pinhole_opencv(
            width,
            height,
            fx,
            fy,
            cx,
            cy,
            image_width=image_width,
            image_height=image_height,
            device="cpu",
            **coefficients,
        )
        self.assertEqual(rays.shape, (height, width, 1, 2))
        self.assertEqual(rays.dtype, wp.vec3f)

        rays = rays.numpy()[:, :, 0]
        np.testing.assert_array_equal(rays[..., 0, :], np.zeros_like(rays[..., 0, :]))
        np.testing.assert_allclose(np.linalg.norm(rays[..., 1, :], axis=-1), 1.0, atol=1e-6)
        for py in range(height):
            for px in range(width):
                x, y = _ray_to_opencv_normalized(rays[py, px, 1])
                x_distorted, y_distorted = _forward_pinhole_opencv(x, y, coefficients)
                expected_x = (((px + 0.5) / width) * image_width - cx) / fx
                expected_y = (((py + 0.5) / height) * image_height - cy) / fy
                self.assertAlmostEqual(x_distorted, expected_x, delta=1.0e-5)
                self.assertAlmostEqual(y_distorted, expected_y, delta=1.0e-5)

    def test_sensor_camera_opencv_pinhole_rejects_invalid_calibration(self):
        """Verify SensorCamera.compute_camera_rays_pinhole_opencv rejects invalid calibration values."""
        calibration = {"width": 1, "height": 1, "fx": 1.0, "fy": 1.0, "cx": 0.5, "cy": 0.5, "device": "cpu"}

        for name in ("fx", "fy"):
            for value in (0.0, -1.0, math.nan, math.inf):
                with self.subTest(name=name, value=value):
                    with self.assertRaisesRegex(ValueError, "fx and fy must be finite and positive"):
                        SensorCamera.compute_camera_rays_pinhole_opencv(**(calibration | {name: value}))

        for name in ("cx", "cy", "k1", "p1", "s1"):
            with self.subTest(name=name):
                with self.assertRaisesRegex(ValueError, "cx, cy, and distortion coefficients must be finite"):
                    SensorCamera.compute_camera_rays_pinhole_opencv(**(calibration | {name: math.nan}))


if __name__ == "__main__":
    unittest.main()
