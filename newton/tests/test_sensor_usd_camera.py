# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import math
import types
import unittest

import numpy as np
import warp as wp

import newton
from newton.sensors import SensorTiledCamera

try:
    from pxr import Gf, Usd, UsdGeom
except ImportError:
    Gf = None
    Usd = None
    UsdGeom = None


def _make_utils(device: str = "cpu", up_axis: newton.Axis = newton.Axis.Z):
    from newton._src.sensors.warp_raytrace.utils import Utils  # noqa: PLC0415

    render_context = types.SimpleNamespace(world_count=2, device=wp.get_device(device), up_axis=up_axis)
    return Utils(render_context)


def _make_camera():
    stage = Usd.Stage.CreateInMemory()
    camera = UsdGeom.Camera.Define(stage, "/World/Camera")
    return stage, camera


def _direction(theta: float, x_sign: float = 1.0) -> np.ndarray:
    return np.array([x_sign * math.sin(theta), 0.0, -math.cos(theta)], dtype=np.float32)


class TestSensorCameraRays(unittest.TestCase):
    @unittest.skipIf(Usd is None, "Requires USD Python bindings")
    def test_usd_camera_transform_matches_model_up_axis(self):
        from newton.math import quat_between_axes  # noqa: PLC0415

        utils = _make_utils(up_axis=newton.Axis.Z)
        stage, camera = _make_camera()
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.y)
        camera.AddTranslateOp().Set(Gf.Vec3d(0.0, 1.0, 0.0))

        got = utils.compute_camera_transforms_usd(camera).numpy()[0, 0]
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

    @unittest.skipIf(Usd is None, "Requires USD Python bindings")
    def test_usd_camera_transform_composes_import_xform(self):
        from newton.math import quat_between_axes  # noqa: PLC0415

        utils = _make_utils(up_axis=newton.Axis.Z)
        stage, camera = _make_camera()
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.y)
        camera.AddTranslateOp().Set(Gf.Vec3d(0.0, 1.0, 0.0))
        import_xform = wp.transform(
            wp.vec3(1.0, 2.0, 3.0),
            wp.quat(0.0, 0.0, 0.70710678, 0.70710678),
        )

        got = utils.compute_camera_transforms_usd(camera, xform=import_xform).numpy()[0, 0]
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

    def test_opencv_fisheye_zero_distortion(self):
        utils = _make_utils()

        got = utils.compute_camera_rays_fisheye_opencv(3, 3, fx=1.0, fy=1.0, cx=1.5, cy=1.5).numpy()[0, 1, 2, 1]
        expected = _direction(1.0)

        np.testing.assert_allclose(got, expected, atol=1e-6)

    def test_pinhole_rays_write_preallocated_camera_index(self):
        utils = _make_utils()
        width, height = 3, 3
        fov = math.radians(45.0)
        expected = utils.compute_camera_rays_pinhole(width, height, camera_fovs=fov).numpy()[0]
        out_rays = wp.zeros((2, height, width, 2), dtype=wp.vec3f, device="cpu")

        got = utils.compute_camera_rays_pinhole(
            width, height, camera_fovs=fov, out_rays=out_rays, camera_index=1
        ).numpy()

        np.testing.assert_array_equal(got[0], np.zeros_like(got[0]))
        np.testing.assert_allclose(got[1], expected, atol=1e-6)

    def test_pinhole_rays_require_keyword_camera_fovs(self):
        utils = _make_utils()

        with self.assertRaises(TypeError):
            utils.compute_camera_rays_pinhole(1, 1, math.radians(45.0))

    def test_pinhole_aperture_matches_fov_helper(self):
        utils = _make_utils()
        width, height = 5, 3
        fov = math.radians(60.0)
        vertical_aperture = 2.0 * math.tan(fov * 0.5)
        horizontal_aperture = vertical_aperture * (width / height)

        got = utils.compute_camera_rays_pinhole(
            width,
            height,
            focal_length=1.0,
            horizontal_aperture=horizontal_aperture,
            vertical_aperture=vertical_aperture,
        ).numpy()
        expected = utils.compute_camera_rays_pinhole(width, height, camera_fovs=fov).numpy()

        np.testing.assert_allclose(got, expected, atol=1e-6)

    def test_pinhole_length_one_warp_intrinsic_broadcasts(self):
        utils = _make_utils()
        width, height = 5, 3
        horizontal_aperture = 2.0
        vertical_apertures = [1.0, 1.5]

        got = utils.compute_camera_rays_pinhole(
            width,
            height,
            focal_length=[1.0, 1.0],
            horizontal_aperture=wp.array([horizontal_aperture], dtype=wp.float32, device="cpu"),
            vertical_aperture=vertical_apertures,
        ).numpy()
        expected = utils.compute_camera_rays_pinhole(
            width,
            height,
            focal_length=[1.0, 1.0],
            horizontal_aperture=horizontal_aperture,
            vertical_aperture=vertical_apertures,
        ).numpy()

        np.testing.assert_allclose(got, expected, atol=1e-6)

    def test_pinhole_aperture_offsets_shift_principal_ray(self):
        utils = _make_utils()

        got = utils.compute_camera_rays_pinhole(
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

    @unittest.skipIf(Usd is None, "Requires USD Python bindings")
    def test_usd_pinhole_camera_rays_accepts_prim_and_camera(self):
        utils = _make_utils()
        width, height = 5, 3
        _stage, camera = _make_camera()
        camera.GetProjectionAttr().Set(UsdGeom.Tokens.perspective)
        camera.GetFocalLengthAttr().Set(1.5)
        camera.GetHorizontalApertureAttr().Set(2.0)
        camera.GetVerticalApertureAttr().Set(1.0)
        camera.GetHorizontalApertureOffsetAttr().Set(0.1)
        camera.GetVerticalApertureOffsetAttr().Set(0.2)
        expected = utils.compute_camera_rays_pinhole(
            width,
            height,
            focal_length=1.5,
            horizontal_aperture=2.0,
            vertical_aperture=1.0,
            horizontal_aperture_offset=0.1,
            vertical_aperture_offset=0.2,
        ).numpy()

        got_prim = utils.compute_camera_rays_usd_pinhole(width, height, camera.GetPrim()).numpy()
        got_camera = utils.compute_camera_rays_usd_pinhole(width, height, camera).numpy()

        np.testing.assert_allclose(got_prim, expected, atol=1e-6)
        np.testing.assert_allclose(got_camera, expected, atol=1e-6)

    @unittest.skipIf(Usd is None, "Requires USD Python bindings")
    def test_usd_pinhole_camera_rays_rejects_invalid_prim(self):
        utils = _make_utils()

        with self.assertRaisesRegex(TypeError, "Expected a valid UsdGeom.Camera prim"):
            utils.compute_camera_rays_usd_pinhole(1, 1, Usd.Prim())

    def test_opencv_fisheye_distortion_solves_theta(self):
        utils = _make_utils()
        theta = 0.5
        k1 = 0.25
        radius = theta * (1.0 + k1 * theta * theta)

        got = utils.compute_camera_rays_fisheye_opencv(
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
        utils = _make_utils()
        theta = 0.4
        radius = 2.0 * theta

        got = utils.compute_camera_rays_fisheye_ftheta(
            1,
            1,
            optical_center_x=0.5 - radius,
            optical_center_y=0.5,
            k1=2.0,
            max_fov=math.pi,
        ).numpy()[0, 0, 0, 1]

        np.testing.assert_allclose(got, _direction(theta), atol=1e-6)

    def test_fisheye_image_size_aliases_match_nominal_names(self):
        utils = _make_utils()

        ftheta_from_image_size = utils.compute_camera_rays_fisheye_ftheta(
            2,
            2,
            optical_center_x=2.0,
            optical_center_y=2.0,
            image_width=4.0,
            image_height=4.0,
            k1=2.0,
            max_fov=math.pi,
        ).numpy()
        ftheta_from_nominal_size = utils.compute_camera_rays_fisheye_ftheta(
            2,
            2,
            optical_center_x=2.0,
            optical_center_y=2.0,
            nominal_width=4.0,
            nominal_height=4.0,
            k1=2.0,
            max_fov=math.pi,
        ).numpy()
        kb_from_image_size = utils.compute_camera_rays_fisheye_kannala_brandt(
            2,
            2,
            optical_center_x=2.0,
            optical_center_y=2.0,
            image_width=4.0,
            image_height=4.0,
            k0=2.0,
            max_fov=math.pi,
        ).numpy()
        kb_from_nominal_size = utils.compute_camera_rays_fisheye_kannala_brandt(
            2,
            2,
            optical_center_x=2.0,
            optical_center_y=2.0,
            nominal_width=4.0,
            nominal_height=4.0,
            k0=2.0,
            max_fov=math.pi,
        ).numpy()

        np.testing.assert_allclose(ftheta_from_image_size, ftheta_from_nominal_size, atol=1e-6)
        np.testing.assert_allclose(kb_from_image_size, kb_from_nominal_size, atol=1e-6)

    def test_fisheye_image_size_alias_conflicts_raise(self):
        utils = _make_utils()

        for helper in (
            utils.compute_camera_rays_fisheye_ftheta,
            utils.compute_camera_rays_fisheye_kannala_brandt,
        ):
            with self.assertRaisesRegex(ValueError, "image_width and nominal_width"):
                helper(
                    1,
                    1,
                    optical_center_x=0.5,
                    optical_center_y=0.5,
                    image_width=2.0,
                    nominal_width=3.0,
                )

    def test_fisheye_rays_write_preallocated_camera_index(self):
        utils = _make_utils()
        theta = 0.4
        radius = 2.0 * theta
        expected = utils.compute_camera_rays_fisheye_ftheta(
            1,
            1,
            optical_center_x=0.5 - radius,
            optical_center_y=0.5,
            k1=2.0,
            max_fov=math.pi,
        ).numpy()[0]
        out_rays = wp.zeros((2, 1, 1, 2), dtype=wp.vec3f, device="cpu")

        got = utils.compute_camera_rays_fisheye_ftheta(
            1,
            1,
            optical_center_x=0.5 - radius,
            optical_center_y=0.5,
            k1=2.0,
            max_fov=math.pi,
            out_rays=out_rays,
            camera_index=1,
        ).numpy()

        np.testing.assert_array_equal(got[0], np.zeros_like(got[0]))
        np.testing.assert_allclose(got[1], expected, atol=1e-6)

    def test_kannala_brandt_k3_solves_known_angle(self):
        utils = _make_utils()
        theta = 0.3
        radius = 2.0 * theta

        got = utils.compute_camera_rays_fisheye_kannala_brandt(
            1,
            1,
            optical_center_x=0.5 - radius,
            optical_center_y=0.5,
            k0=2.0,
            max_fov=math.pi,
        ).numpy()[0, 0, 0, 1]

        np.testing.assert_allclose(got, _direction(theta), atol=1e-6)

    def test_fisheye_max_fov_masks_invalid_ray(self):
        utils = _make_utils()

        got = utils.compute_camera_rays_fisheye_ftheta(
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

        sensor = SensorTiledCamera(model)
        camera_transforms = wp.array(
            [[wp.transformf(wp.vec3f(0.0), wp.quatf(0.0, 0.0, 0.0, 1.0))]],
            dtype=wp.transformf,
            device="cpu",
        )
        camera_rays = wp.zeros((1, 1, 1, 2), dtype=wp.vec3f, device="cpu")
        color = sensor.utils.create_color_image_output(1, 1)
        depth = sensor.utils.create_depth_image_output(1, 1)
        clear_data = SensorTiledCamera.ClearData(clear_color=0xFF112233, clear_depth=-1.0)

        sensor.update(
            state, camera_transforms, camera_rays, color_image=color, depth_image=depth, clear_data=clear_data
        )

        self.assertEqual(int(color.numpy()[0, 0, 0, 0]), 0xFF112233)
        self.assertEqual(float(depth.numpy()[0, 0, 0, 0]), -1.0)


if __name__ == "__main__":
    unittest.main()
