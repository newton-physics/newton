# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import math
import unittest

import numpy as np
import warp as wp

import newton
from newton.sensors import SensorCamera
from newton.tests.unittest_utils import requires

try:
    from pxr import Gf, Usd, UsdGeom
except ImportError:
    Gf = None
    Usd = None
    UsdGeom = None


def _make_sensor(up_axis: newton.Axis = newton.Axis.Z) -> SensorCamera:
    builder = newton.ModelBuilder(up_axis=up_axis)
    builder.add_shape_sphere(-1, radius=0.5)
    return SensorCamera(builder.finalize(device="cpu"))


def _make_camera():
    stage = Usd.Stage.CreateInMemory()
    camera = UsdGeom.Camera.Define(stage, "/World/Camera")
    return stage, camera


def _direction(theta: float, x_sign: float = 1.0) -> np.ndarray:
    return np.array([x_sign * math.sin(theta), 0.0, -math.cos(theta)], dtype=np.float32)


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
    def test_usd_camera_transform_matches_model_up_axis(self):
        from newton.math import quat_between_axes  # noqa: PLC0415

        sensor = _make_sensor(up_axis=newton.Axis.Z)
        stage, camera = _make_camera()
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.y)
        camera.AddTranslateOp().Set(Gf.Vec3d(0.0, 1.0, 0.0))

        got = sensor.compute_camera_transforms_usd(camera).numpy()[0]
        expected = wp.transform(wp.vec3(0.0), quat_between_axes(newton.Axis.Y, newton.Axis.Z)) * wp.transform(
            wp.vec3(0.0, 1.0, 0.0),
            wp.quat_identity(),
        )

        np.testing.assert_allclose(got[:3], np.array(expected.p), atol=1e-6)
        got_q = got[3:]
        expected_q = np.array(expected.q)
        if np.dot(got_q, expected_q) < 0.0:
            got_q = -got_q
        np.testing.assert_allclose(got_q, expected_q, atol=1e-6)

    @requires("pxr.Usd")
    def test_usd_camera_transform_composes_import_xform(self):
        from newton.math import quat_between_axes  # noqa: PLC0415

        sensor = _make_sensor(up_axis=newton.Axis.Z)
        stage, camera = _make_camera()
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.y)
        camera.AddTranslateOp().Set(Gf.Vec3d(0.0, 1.0, 0.0))
        import_xform = wp.transform(
            wp.vec3(1.0, 2.0, 3.0),
            wp.quat(0.0, 0.0, 0.70710678, 0.70710678),
        )

        got = sensor.compute_camera_transforms_usd(camera, xform=import_xform).numpy()[0]
        expected = (
            import_xform
            * wp.transform(wp.vec3(0.0), quat_between_axes(newton.Axis.Y, newton.Axis.Z))
            * wp.transform(wp.vec3(0.0, 1.0, 0.0), wp.quat_identity())
        )

        np.testing.assert_allclose(got[:3], np.array(expected.p), atol=1e-6)
        got_q = got[3:]
        expected_q = np.array(expected.q)
        if np.dot(got_q, expected_q) < 0.0:
            got_q = -got_q
        np.testing.assert_allclose(got_q, expected_q, atol=1e-6)

    @requires("pxr.Usd")
    def test_usd_camera_transform_keeps_view_and_up_under_mirrored_parent(self):
        sensor = _make_sensor(up_axis=newton.Axis.Z)
        for parent_scale in ((-1.0, 1.0, 1.0), (-1.0, -1.0, -1.0), (1.0, 1.0, -1.0)):
            with self.subTest(parent_scale=parent_scale):
                stage = Usd.Stage.CreateInMemory()
                UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
                parent = UsdGeom.Xform.Define(stage, "/World/Parent")
                parent.AddTranslateOp().Set(Gf.Vec3d(0.5, -1.0, 2.0))
                parent.AddScaleOp().Set(Gf.Vec3d(*parent_scale))
                camera = UsdGeom.Camera.Define(stage, "/World/Parent/Camera")
                camera.AddTranslateOp().Set(Gf.Vec3d(1.0, 2.0, 3.0))
                camera.AddRotateXYZOp().Set(Gf.Vec3f(20.0, 30.0, 40.0))

                world = camera.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
                # USD transforms row vectors, so the authored directions are M applied to -Z and +Y.
                expected_view = np.array(world.TransformDir(Gf.Vec3d(0.0, 0.0, -1.0)))
                expected_up = np.array(world.TransformDir(Gf.Vec3d(0.0, 1.0, 0.0)))
                expected_view /= np.linalg.norm(expected_view)
                expected_up /= np.linalg.norm(expected_up)
                expected_pos = np.array(world.ExtractTranslation())

                got = sensor.compute_camera_transforms_usd(camera).numpy()[0]
                q = wp.quat(*got[3:].tolist())
                view = np.array(wp.quat_rotate(q, wp.vec3(0.0, 0.0, -1.0)))
                up = np.array(wp.quat_rotate(q, wp.vec3(0.0, 1.0, 0.0)))
                np.testing.assert_allclose(got[:3], expected_pos, atol=1e-5)
                np.testing.assert_allclose(view, expected_view, atol=1e-5)
                np.testing.assert_allclose(up, expected_up, atol=1e-5)

    def test_opencv_fisheye_zero_distortion(self):
        got = SensorCamera.compute_camera_rays_fisheye_opencv(
            3, 3, fx=1.0, fy=1.0, cx=1.5, cy=1.5, device="cpu"
        ).numpy()[1, 2, 0, 1]
        expected = _direction(1.0)

        np.testing.assert_allclose(got, expected, atol=1e-6)

    def test_opencv_pinhole_zero_distortion(self):
        """Verify zero distortion produces calibrated pinhole rays."""
        got = SensorCamera.compute_camera_rays_pinhole_opencv(
            3, 3, fx=2.0, fy=4.0, cx=1.0, cy=2.0, device="cpu"
        ).numpy()[1, 2, 0, 1]
        expected = np.array([0.75, 0.125, -1.0], dtype=np.float32)
        expected /= np.linalg.norm(expected)

        np.testing.assert_allclose(got, expected, atol=1e-6)

    def test_opencv_pinhole_strong_distortion_round_trips(self):
        """Verify inversion remains accurate for strong invertible distortion."""
        width, height = 32, 24
        image_width, image_height = 640.0, 480.0
        fx = fy = 320.0
        cx, cy = 320.0, 240.0
        coefficients = {
            "k1": -0.45,
            "k2": 0.25,
            "k3": 0.0,
            "k4": 0.0,
            "k5": 0.0,
            "k6": 0.0,
            "p1": 0.0,
            "p2": 0.0,
            "s1": 0.0,
            "s2": 0.0,
            "s3": 0.0,
            "s4": 0.0,
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
            **coefficients,
        ).numpy()

        np.testing.assert_allclose(np.linalg.norm(rays[..., 1, :], axis=-1), 1.0, atol=1e-6)
        for py in range(height):
            for px in range(width):
                x, y = _ray_to_opencv_normalized(rays[py, px, 0, 1])
                x_distorted, y_distorted = _forward_pinhole_opencv(x, y, coefficients)
                expected_x = (((px + 0.5) / width) * image_width - cx) / fx
                expected_y = (((py + 0.5) / height) * image_height - cy) / fy
                self.assertAlmostEqual(x_distorted, expected_x, delta=1.0e-5)
                self.assertAlmostEqual(y_distorted, expected_y, delta=1.0e-5)

    def test_opencv_pinhole_non_invertible_distortion_returns_sentinel(self):
        """Verify pixels without a verifiable inverse receive the zero sentinel."""
        width, height = 64, 48
        image_width, image_height = 640.0, 480.0
        fx = fy = 150.0
        cx, cy = 320.0, 240.0

        # A strong barrel term folds the forward map, leaving outer pixels non-invertible.
        rays = SensorCamera.compute_camera_rays_pinhole_opencv(
            width,
            height,
            fx,
            fy,
            cx,
            cy,
            image_width=image_width,
            image_height=image_height,
            k1=-0.9,
        ).numpy()

        directions = rays[:, :, 0, 1]
        self.assertTrue(np.isfinite(directions).all())

        norms = np.linalg.norm(directions, axis=-1)
        is_zero = norms == 0.0
        is_unit = np.isclose(norms, 1.0, atol=1e-6)
        # Every ray is either a valid unit direction or the exact zero sentinel.
        self.assertTrue(np.logical_or(is_zero, is_unit).all())
        # The fold produces both resolvable and unresolvable pixels.
        self.assertTrue(is_zero.any())
        self.assertTrue(is_unit.any())
        # The central pixel is near the principal point and must resolve.
        self.assertTrue(is_unit[height // 2, width // 2])

    def test_opencv_pinhole_rejects_invalid_calibration(self):
        """Verify invalid OpenCV pinhole calibration values are rejected."""
        calibration = {
            "width": 1,
            "height": 1,
            "fx": 1.0,
            "fy": 1.0,
            "cx": 0.5,
            "cy": 0.5,
            "image_width": 1.0,
            "image_height": 1.0,
        }

        for name in ("fx", "fy"):
            for value in (0.0, -1.0, math.nan, math.inf, -math.inf):
                with self.subTest(name=name, value=value):
                    invalid_calibration = calibration | {name: value}
                    with self.assertRaisesRegex(ValueError, "fx and fy must be finite and positive"):
                        SensorCamera.compute_camera_rays_pinhole_opencv(**invalid_calibration)

        for name in ("image_width", "image_height"):
            for value in (0.0, -1.0, math.nan, math.inf, -math.inf):
                with self.subTest(name=name, value=value):
                    invalid_calibration = calibration | {name: value}
                    with self.assertRaisesRegex(ValueError, "image_width and image_height must be finite and positive"):
                        SensorCamera.compute_camera_rays_pinhole_opencv(**invalid_calibration)

        for name in ("cx", "cy", "k1", "k2", "k3", "k4", "k5", "k6", "p1", "p2", "s1", "s2", "s3", "s4"):
            for value in (math.nan, math.inf, -math.inf):
                with self.subTest(name=name, value=value):
                    invalid_calibration = calibration | {name: value}
                    with self.assertRaisesRegex(ValueError, "cx, cy, and distortion coefficients must be finite"):
                        SensorCamera.compute_camera_rays_pinhole_opencv(**invalid_calibration)

    def test_pinhole_rays_require_keyword_fov(self):
        with self.assertRaises(TypeError):
            SensorCamera.compute_camera_rays_pinhole(1, 1, math.radians(45.0))

    def test_pinhole_aperture_matches_fov_helper(self):
        width, height = 5, 3
        fov = math.radians(60.0)
        vertical_aperture = 2.0 * math.tan(fov * 0.5)
        horizontal_aperture = vertical_aperture * (width / height)

        got = SensorCamera.compute_camera_rays_pinhole(
            width,
            height,
            focal_length=1.0,
            horizontal_aperture=horizontal_aperture,
            vertical_aperture=vertical_aperture,
        ).numpy()
        expected = SensorCamera.compute_camera_rays_pinhole(width, height, fov=fov).numpy()

        np.testing.assert_allclose(got, expected, atol=1e-6)

    def test_pinhole_aperture_offsets_shift_principal_ray(self):
        got = SensorCamera.compute_camera_rays_pinhole(
            1,
            1,
            focal_length=1.0,
            horizontal_aperture=1.0,
            vertical_aperture=1.0,
            horizontal_aperture_offset=0.1,
            vertical_aperture_offset=0.2,
        ).numpy()[0, 0, 0, 1]
        expected = np.array([0.1, 0.2, -1.0], dtype=np.float32)
        expected /= np.linalg.norm(expected)

        np.testing.assert_allclose(got, expected, atol=1e-6)

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

    def test_opencv_pinhole_full_distortion_round_trips(self):
        """Verify inversion of the full OpenCV pinhole coefficient set."""
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

    def test_opencv_fisheye_distortion_solves_theta(self):
        theta = 0.5
        k1 = 0.25
        radius = theta * (1.0 + k1 * theta * theta)

        got = SensorCamera.compute_camera_rays_fisheye_opencv(
            1,
            1,
            fx=1.0,
            fy=1.0,
            cx=0.5 - radius,
            cy=0.5,
            k1=k1,
        ).numpy()[0, 0, 0, 1]

        np.testing.assert_allclose(got, _direction(theta), atol=1e-6)

    def test_ftheta_solves_known_angle(self):
        theta = 0.4
        radius = 2.0 * theta

        got = SensorCamera.compute_camera_rays_fisheye_ftheta(
            1,
            1,
            optical_center_x=0.5 - radius,
            optical_center_y=0.5,
            k1=2.0,
            max_fov=math.pi,
        ).numpy()[0, 0, 0, 1]

        np.testing.assert_allclose(got, _direction(theta), atol=1e-6)

    def test_kannala_brandt_k3_solves_known_angle(self):
        theta = 0.3
        radius = 2.0 * theta

        got = SensorCamera.compute_camera_rays_fisheye_kannala_brandt(
            1,
            1,
            optical_center_x=0.5 - radius,
            optical_center_y=0.5,
            k0=2.0,
            max_fov=math.pi,
        ).numpy()[0, 0, 0, 1]

        np.testing.assert_allclose(got, _direction(theta), atol=1e-6)

    def test_fisheye_max_fov_masks_invalid_ray(self):
        got = SensorCamera.compute_camera_rays_fisheye_ftheta(
            1,
            1,
            optical_center_x=-0.5,
            optical_center_y=0.5,
            k1=1.0,
            max_fov=math.radians(60.0),
        ).numpy()[0, 0, 0, 1]

        np.testing.assert_array_equal(got, np.zeros(3, dtype=np.float32))

    def test_zero_direction_ray_renders_clear_values(self):
        builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
        body = builder.add_body(xform=wp.transform(p=wp.vec3(0.0, 0.0, -2.0), q=wp.quat_identity()))
        builder.add_shape_sphere(body, radius=0.5)
        model = builder.finalize(device="cpu")
        state = model.state()
        model.bvh_build_shapes(state)

        sensor = SensorCamera(model)
        camera_transforms = wp.array([wp.transform_identity()], dtype=wp.transformf, device="cpu")
        camera_rays = wp.zeros((1, 1, 1, 2), dtype=wp.vec3f, device="cpu")
        color = sensor.allocate_image_color(1, 1, 1)
        depth = sensor.allocate_image_depth(1, 1, 1)
        clear_data = SensorCamera.ClearData(clear_color=0xFF112233, clear_depth=-1.0)

        sensor.update(
            state, camera_transforms, camera_rays, color_image=color, depth_image=depth, clear_data=clear_data
        )

        self.assertEqual(int(color.numpy()[0, 0, 0]), 0xFF112233)
        self.assertEqual(float(depth.numpy()[0, 0, 0]), -1.0)


if __name__ == "__main__":
    unittest.main()
