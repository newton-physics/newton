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
    from pxr import Gf, Sdf, Usd, UsdGeom
except ImportError:
    Gf = None
    Sdf = None
    Usd = None
    UsdGeom = None


def _make_utils(device: str = "cpu"):
    from newton._src.sensors.warp_raytrace.utils import Utils  # noqa: PLC0415

    render_context = types.SimpleNamespace(world_count=2, device=wp.get_device(device))
    return Utils(render_context)


def _make_camera():
    stage = Usd.Stage.CreateInMemory()
    camera = UsdGeom.Camera.Define(stage, "/World/Camera")
    return stage, camera


def _set_attr(prim, name: str, type_name, value) -> None:
    prim.CreateAttribute(name, type_name, custom=True).Set(value)


def _direction(theta: float, x_sign: float = 1.0) -> np.ndarray:
    return np.array([x_sign * math.sin(theta), 0.0, -math.cos(theta)], dtype=np.float32)


@unittest.skipIf(Usd is None, "Requires USD Python bindings")
class TestSensorUsdCameraRays(unittest.TestCase):
    def test_usd_pinhole_matches_fov_helper(self):
        utils = _make_utils()
        width, height = 5, 3
        fov = math.radians(60.0)
        vertical_aperture = 2.0 * math.tan(fov * 0.5)
        horizontal_aperture = vertical_aperture * (width / height)

        _stage, camera = _make_camera()
        camera.GetFocalLengthAttr().Set(1.0)
        camera.GetHorizontalApertureAttr().Set(horizontal_aperture)
        camera.GetVerticalApertureAttr().Set(vertical_aperture)

        got = utils.compute_usd_camera_rays(width, height, camera).numpy()
        expected = utils.compute_pinhole_camera_rays(width, height, fov).numpy()

        np.testing.assert_allclose(got, expected, atol=1e-6)

    def test_usd_pinhole_aperture_offsets_shift_principal_ray(self):
        utils = _make_utils()
        _stage, camera = _make_camera()
        camera.GetFocalLengthAttr().Set(1.0)
        camera.GetHorizontalApertureAttr().Set(1.0)
        camera.GetVerticalApertureAttr().Set(1.0)
        camera.GetHorizontalApertureOffsetAttr().Set(0.1)
        camera.GetVerticalApertureOffsetAttr().Set(0.2)

        got = utils.compute_usd_camera_rays(1, 1, camera).numpy()[0, 0, 0, 1]
        expected = np.array([0.1, 0.2, -1.0], dtype=np.float32)
        expected /= np.linalg.norm(expected)

        np.testing.assert_allclose(got, expected, atol=1e-6)

    def test_opencv_fisheye_zero_distortion(self):
        utils = _make_utils()
        _stage, camera = _make_camera()
        prim = camera.GetPrim()
        _set_attr(prim, "omni:lensdistortion:model", Sdf.ValueTypeNames.Token, "opencvFisheye")
        _set_attr(prim, "omni:lensdistortion:opencvFisheye:imageSize", Sdf.ValueTypeNames.Int2, Gf.Vec2i(3, 3))
        _set_attr(prim, "omni:lensdistortion:opencvFisheye:fx", Sdf.ValueTypeNames.Float, 1.0)
        _set_attr(prim, "omni:lensdistortion:opencvFisheye:fy", Sdf.ValueTypeNames.Float, 1.0)
        _set_attr(prim, "omni:lensdistortion:opencvFisheye:cx", Sdf.ValueTypeNames.Float, 1.5)
        _set_attr(prim, "omni:lensdistortion:opencvFisheye:cy", Sdf.ValueTypeNames.Float, 1.5)
        for coeff in ("k1", "k2", "k3", "k4"):
            _set_attr(prim, f"omni:lensdistortion:opencvFisheye:{coeff}", Sdf.ValueTypeNames.Float, 0.0)

        got = utils.compute_usd_camera_rays(3, 3, camera).numpy()[0, 1, 2, 1]
        expected = _direction(1.0)

        np.testing.assert_allclose(got, expected, atol=1e-6)

    def test_opencv_fisheye_distortion_solves_theta(self):
        utils = _make_utils()
        theta = 0.5
        k1 = 0.25
        radius = theta * (1.0 + k1 * theta * theta)
        _stage, camera = _make_camera()
        prim = camera.GetPrim()
        _set_attr(prim, "omni:lensdistortion:model", Sdf.ValueTypeNames.Token, "opencvFisheye")
        _set_attr(prim, "omni:lensdistortion:opencvFisheye:imageSize", Sdf.ValueTypeNames.Int2, Gf.Vec2i(1, 1))
        _set_attr(prim, "omni:lensdistortion:opencvFisheye:fx", Sdf.ValueTypeNames.Float, 1.0)
        _set_attr(prim, "omni:lensdistortion:opencvFisheye:fy", Sdf.ValueTypeNames.Float, 1.0)
        _set_attr(prim, "omni:lensdistortion:opencvFisheye:cx", Sdf.ValueTypeNames.Float, 0.5 - radius)
        _set_attr(prim, "omni:lensdistortion:opencvFisheye:cy", Sdf.ValueTypeNames.Float, 0.5)
        _set_attr(prim, "omni:lensdistortion:opencvFisheye:k1", Sdf.ValueTypeNames.Float, k1)
        _set_attr(prim, "omni:lensdistortion:opencvFisheye:k2", Sdf.ValueTypeNames.Float, 0.0)
        _set_attr(prim, "omni:lensdistortion:opencvFisheye:k3", Sdf.ValueTypeNames.Float, 0.0)
        _set_attr(prim, "omni:lensdistortion:opencvFisheye:k4", Sdf.ValueTypeNames.Float, 0.0)

        got = utils.compute_usd_camera_rays(1, 1, camera).numpy()[0, 0, 0, 1]

        np.testing.assert_allclose(got, _direction(theta), atol=1e-6)

    def test_ftheta_solves_known_angle(self):
        utils = _make_utils()
        theta = 0.4
        radius = 2.0 * theta
        _stage, camera = _make_camera()
        prim = camera.GetPrim()
        _set_attr(prim, "omni:lensdistortion:model", Sdf.ValueTypeNames.Token, "ftheta")
        _set_attr(prim, "omni:lensdistortion:ftheta:nominalWidth", Sdf.ValueTypeNames.Float, 1.0)
        _set_attr(prim, "omni:lensdistortion:ftheta:nominalHeight", Sdf.ValueTypeNames.Float, 1.0)
        _set_attr(
            prim,
            "omni:lensdistortion:ftheta:opticalCenter",
            Sdf.ValueTypeNames.Float2,
            Gf.Vec2f(0.5 - radius, 0.5),
        )
        _set_attr(prim, "omni:lensdistortion:ftheta:k0", Sdf.ValueTypeNames.Float, 0.0)
        _set_attr(prim, "omni:lensdistortion:ftheta:k1", Sdf.ValueTypeNames.Float, 2.0)
        _set_attr(prim, "omni:lensdistortion:ftheta:k2", Sdf.ValueTypeNames.Float, 0.0)
        _set_attr(prim, "omni:lensdistortion:ftheta:k3", Sdf.ValueTypeNames.Float, 0.0)
        _set_attr(prim, "omni:lensdistortion:ftheta:k4", Sdf.ValueTypeNames.Float, 0.0)
        _set_attr(prim, "omni:lensdistortion:ftheta:maxFov", Sdf.ValueTypeNames.Float, 180.0)

        got = utils.compute_usd_camera_rays(1, 1, camera).numpy()[0, 0, 0, 1]

        np.testing.assert_allclose(got, _direction(theta), atol=1e-6)

    def test_explicit_lens_model_ignores_stale_fisheye_attrs(self):
        utils = _make_utils()
        theta = 0.4
        radius = 2.0 * theta
        _stage, camera = _make_camera()
        prim = camera.GetPrim()
        _set_attr(prim, "omni:lensdistortion:model", Sdf.ValueTypeNames.Token, "ftheta")
        _set_attr(prim, "omni:lensdistortion:opencvFisheye:imageSize", Sdf.ValueTypeNames.Int2, Gf.Vec2i(1, 1))
        _set_attr(prim, "omni:lensdistortion:opencvFisheye:fx", Sdf.ValueTypeNames.Float, 1.0)
        _set_attr(prim, "omni:lensdistortion:opencvFisheye:fy", Sdf.ValueTypeNames.Float, 1.0)
        _set_attr(prim, "omni:lensdistortion:opencvFisheye:cx", Sdf.ValueTypeNames.Float, 0.5)
        _set_attr(prim, "omni:lensdistortion:opencvFisheye:cy", Sdf.ValueTypeNames.Float, 0.5)
        _set_attr(prim, "omni:lensdistortion:opencvFisheye:k1", Sdf.ValueTypeNames.Float, 0.0)
        _set_attr(prim, "omni:lensdistortion:ftheta:nominalWidth", Sdf.ValueTypeNames.Float, 1.0)
        _set_attr(prim, "omni:lensdistortion:ftheta:nominalHeight", Sdf.ValueTypeNames.Float, 1.0)
        _set_attr(
            prim,
            "omni:lensdistortion:ftheta:opticalCenter",
            Sdf.ValueTypeNames.Float2,
            Gf.Vec2f(0.5 - radius, 0.5),
        )
        _set_attr(prim, "omni:lensdistortion:ftheta:k0", Sdf.ValueTypeNames.Float, 0.0)
        _set_attr(prim, "omni:lensdistortion:ftheta:k1", Sdf.ValueTypeNames.Float, 2.0)
        _set_attr(prim, "omni:lensdistortion:ftheta:maxFov", Sdf.ValueTypeNames.Float, 180.0)

        got = utils.compute_usd_camera_rays(1, 1, camera).numpy()[0, 0, 0, 1]

        np.testing.assert_allclose(got, _direction(theta), atol=1e-6)

    def test_kannala_brandt_k3_solves_known_angle(self):
        utils = _make_utils()
        theta = 0.3
        radius = 2.0 * theta
        _stage, camera = _make_camera()
        prim = camera.GetPrim()
        _set_attr(prim, "omni:lensdistortion:model", Sdf.ValueTypeNames.Token, "kannalaBrandtK3")
        _set_attr(prim, "omni:lensdistortion:kannalaBrandtK3:nominalWidth", Sdf.ValueTypeNames.Float, 1.0)
        _set_attr(prim, "omni:lensdistortion:kannalaBrandtK3:nominalHeight", Sdf.ValueTypeNames.Float, 1.0)
        _set_attr(
            prim,
            "omni:lensdistortion:kannalaBrandtK3:opticalCenter",
            Sdf.ValueTypeNames.Float2,
            Gf.Vec2f(0.5 - radius, 0.5),
        )
        _set_attr(prim, "omni:lensdistortion:kannalaBrandtK3:k0", Sdf.ValueTypeNames.Float, 2.0)
        _set_attr(prim, "omni:lensdistortion:kannalaBrandtK3:k1", Sdf.ValueTypeNames.Float, 0.0)
        _set_attr(prim, "omni:lensdistortion:kannalaBrandtK3:k2", Sdf.ValueTypeNames.Float, 0.0)
        _set_attr(prim, "omni:lensdistortion:kannalaBrandtK3:k3", Sdf.ValueTypeNames.Float, 0.0)
        _set_attr(prim, "omni:lensdistortion:kannalaBrandtK3:maxFov", Sdf.ValueTypeNames.Float, 180.0)

        got = utils.compute_usd_camera_rays(1, 1, camera).numpy()[0, 0, 0, 1]

        np.testing.assert_allclose(got, _direction(theta), atol=1e-6)

    def test_fisheye_max_fov_masks_invalid_ray(self):
        utils = _make_utils()
        _stage, camera = _make_camera()
        prim = camera.GetPrim()
        _set_attr(prim, "omni:lensdistortion:model", Sdf.ValueTypeNames.Token, "ftheta")
        _set_attr(prim, "omni:lensdistortion:ftheta:nominalWidth", Sdf.ValueTypeNames.Float, 1.0)
        _set_attr(prim, "omni:lensdistortion:ftheta:nominalHeight", Sdf.ValueTypeNames.Float, 1.0)
        _set_attr(
            prim,
            "omni:lensdistortion:ftheta:opticalCenter",
            Sdf.ValueTypeNames.Float2,
            Gf.Vec2f(-0.5, 0.5),
        )
        _set_attr(prim, "omni:lensdistortion:ftheta:k0", Sdf.ValueTypeNames.Float, 0.0)
        _set_attr(prim, "omni:lensdistortion:ftheta:k1", Sdf.ValueTypeNames.Float, 1.0)
        _set_attr(prim, "omni:lensdistortion:ftheta:maxFov", Sdf.ValueTypeNames.Float, 60.0)

        got = utils.compute_usd_camera_rays(1, 1, camera).numpy()[0, 0, 0, 1]

        np.testing.assert_array_equal(got, np.zeros(3, dtype=np.float32))

    def test_zero_direction_ray_renders_clear_values(self):
        builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
        body = builder.add_body(xform=wp.transform(p=wp.vec3(0.0, 0.0, -2.0), q=wp.quat_identity()))
        builder.add_shape_sphere(body, radius=0.5)
        model = builder.finalize(device="cpu")
        state = model.state()
        newton.geometry.build_bvh_shape(model, state)

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
