# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import math
import unittest

import numpy as np
import warp as wp

import newton
from newton.sensors import SensorTiledCamera


def _build_single_sphere_model():
    builder = newton.ModelBuilder()
    body = builder.add_body(xform=wp.transform(p=wp.vec3(0.0, 0.0, 0.0), q=wp.quat_identity()))
    builder.add_shape_sphere(body, radius=1.0, color=(0.5, 0.5, 0.5))
    return builder.finalize(device="cpu")


def _render_tiny_color_and_hdr(model):
    sensor = SensorTiledCamera(model=model)
    state = model.state()

    camera_transforms = wp.array(
        [[wp.transformf(wp.vec3f(3.0, 0.0, 0.0), wp.quatf(0.5, 0.5, 0.5, 0.5))]],
        dtype=wp.transformf,
        device="cpu",
    )
    camera_rays = sensor.utils.compute_pinhole_camera_rays(4, 4, math.radians(45.0))
    color_image = sensor.utils.create_color_image_output(4, 4, 1)
    hdr_color_image = sensor.utils.create_hdr_color_image_output(4, 4, 1)

    newton.geometry.build_bvh_shape(model, state)
    newton.geometry.build_bvh_particle(model, state)
    sensor.update(
        state,
        camera_transforms,
        camera_rays,
        color_image=color_image,
        hdr_color_image=hdr_color_image,
    )
    return np.asarray(color_image.numpy(), dtype=np.uint32), np.asarray(hdr_color_image.numpy(), dtype=np.float32)


class TestSensorTiledCameraHdrColor(unittest.TestCase):
    def test_hdr_color_is_available_next_to_packed_color(self):
        model = _build_single_sphere_model()

        color, hdr_color = _render_tiny_color_and_hdr(model)

        self.assertEqual(color.shape, (1, 1, 4, 4))
        self.assertEqual(hdr_color.shape, (1, 1, 4, 4, 3))
        self.assertEqual(color.dtype, np.uint32)
        self.assertEqual(hdr_color.dtype, np.float32)
        self.assertTrue(np.isfinite(hdr_color).all())
        self.assertGreater(hdr_color.max(), 0.0)

    def test_hdr_color_matches_linear_color_before_packing(self):
        model = _build_single_sphere_model()

        color, hdr_color = _render_tiny_color_and_hdr(model)
        packed_rgb = color.view(np.uint8).reshape(*color.shape, 4)[..., :3].astype(np.float32) / 255.0

        np.testing.assert_allclose(np.clip(hdr_color, 0.0, 1.0), packed_rgb, atol=1.0 / 255.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
