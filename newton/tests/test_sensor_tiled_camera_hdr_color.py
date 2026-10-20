# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import math
import unittest

import numpy as np
import warp as wp

import newton
from newton.sensors import SensorCamera, SensorTiledCamera
from newton.tests.unittest_utils import ignore_sensor_tiled_camera_deprecation


def setUpModule():
    # SensorTiledCamera is deprecated; these tests exercise it intentionally.
    ignore_sensor_tiled_camera_deprecation()


def _build_single_sphere_model():
    builder = newton.ModelBuilder()
    body = builder.add_body(xform=wp.transform(p=wp.vec3(0.0, 0.0, 0.0), q=wp.quat_identity()))
    builder.add_shape_sphere(body, radius=1.0, color=(0.5, 0.5, 0.5))
    return builder.finalize(device="cpu")


def _render_tiny_color_and_hdr(model, *, output_color_space=newton.utils.ColorSpace.SRGB):
    sensor = SensorTiledCamera(
        model=model,
        default_render_config=SensorTiledCamera.RenderConfig(output_color_space=output_color_space),
    )
    state = model.state()

    camera_transforms = wp.array(
        [[wp.transformf(wp.vec3f(3.0, 0.0, 0.0), wp.quatf(0.5, 0.5, 0.5, 0.5))]],
        dtype=wp.transformf,
        device="cpu",
    )
    camera_rays = sensor.utils.compute_camera_rays_pinhole(4, 4, camera_fovs=math.radians(45.0))
    color_image = sensor.utils.create_color_image_output(4, 4, 1)
    hdr_color_image = sensor.utils.create_hdr_color_image_output(4, 4, 1)

    sensor.update(
        state,
        camera_transforms,
        camera_rays,
        color_image=color_image,
        hdr_color_image=hdr_color_image,
    )
    return np.asarray(color_image.numpy(), dtype=np.uint32), np.asarray(hdr_color_image.numpy(), dtype=np.float32)


class TestSensorTiledCameraHdrColor(unittest.TestCase):
    def _render_gaussian_hdr(self, color_space):
        gaussian = newton.Gaussian(
            positions=np.array([[0.0, 0.0, -2.0]], dtype=np.float32),
            scales=np.array([[0.5, 0.5, 0.5]], dtype=np.float32),
            opacities=np.array([1.0], dtype=np.float32),
            sh_coeffs=np.array([[-4.0, 0.0, 4.0]], dtype=np.float32),
            sh_color_space=color_space,
        )
        builder = newton.ModelBuilder()
        builder.add_shape_gaussian(body=-1, gaussian=gaussian)
        model = builder.finalize(device="cpu")

        camera = SensorCamera(model)
        rays = camera.compute_camera_rays_pinhole(1, 1, camera_fov=math.radians(45.0), device="cpu")
        transforms = wp.array([wp.transform_identity()], dtype=wp.transformf, device="cpu")
        hdr = camera.allocate_image_color_hdr(1, 1, 1)
        camera.update(model.state(), transforms, rays, hdr_color_image=hdr)
        return hdr.numpy().reshape(-1, 3)[0]

    def test_gaussian_srgb_radiance_is_clamped_and_decoded(self):
        color = self._render_gaussian_hdr(newton.utils.ColorSpace.SRGB)
        np.testing.assert_allclose(color[:2], [0.0, 0.21404114], atol=1e-5)
        self.assertGreater(color[2], 1.0)

    def test_gaussian_linear_radiance_is_clamped_not_decoded(self):
        color = self._render_gaussian_hdr(newton.utils.ColorSpace.LINEAR)
        np.testing.assert_allclose(color[:2], [0.0, 0.5], atol=1e-5)
        self.assertAlmostEqual(float(color[2]), 0.28209479 * 4.0 + 0.5, places=4)

    def test_gaussian_sh_color_space_affects_identity(self):
        kwargs = {
            "positions": np.zeros((1, 3), dtype=np.float32),
            "sh_coeffs": np.zeros((1, 3), dtype=np.float32),
        }
        self.assertEqual(newton.Gaussian(**kwargs).sh_color_space, newton.utils.ColorSpace.SRGB)
        srgb = newton.Gaussian(**kwargs, sh_color_space=newton.utils.ColorSpace.SRGB)
        linear = newton.Gaussian(**kwargs, sh_color_space=newton.utils.ColorSpace.LINEAR)
        self.assertNotEqual(srgb, linear)
        self.assertNotEqual(hash(srgb), hash(linear))

    def test_hdr_color_is_available_next_to_packed_color(self):
        model = _build_single_sphere_model()

        color, hdr_color = _render_tiny_color_and_hdr(model)

        self.assertEqual(color.shape, (1, 1, 4, 4))
        self.assertEqual(hdr_color.shape, (1, 1, 4, 4, 3))
        self.assertEqual(color.dtype, np.uint32)
        self.assertEqual(hdr_color.dtype, np.float32)
        self.assertTrue(np.isfinite(hdr_color).all())
        self.assertGreater(hdr_color.max(), 0.0)

    def test_hdr_color_matches_srgb_packed_color_after_encoding(self):
        model = _build_single_sphere_model()

        color, hdr_color = _render_tiny_color_and_hdr(model)
        clipped_hdr_color = np.clip(hdr_color, 0.0, 1.0)
        expected_packed_rgb = np.where(
            clipped_hdr_color <= 0.0031308,
            clipped_hdr_color * 12.92,
            1.055 * np.power(clipped_hdr_color, 1.0 / 2.4) - 0.055,
        )
        packed_rgb = color.view(np.uint8).reshape(*color.shape, 4)[..., :3].astype(np.float32) / 255.0

        np.testing.assert_allclose(expected_packed_rgb, packed_rgb, atol=1.0 / 255.0)

    def test_hdr_color_matches_linear_color_when_packing_linear_output(self):
        model = _build_single_sphere_model()

        color, hdr_color = _render_tiny_color_and_hdr(model, output_color_space=newton.utils.ColorSpace.LINEAR)
        packed_rgb = color.view(np.uint8).reshape(*color.shape, 4)[..., :3].astype(np.float32) / 255.0

        np.testing.assert_allclose(np.clip(hdr_color, 0.0, 1.0), packed_rgb, atol=1.0 / 255.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
