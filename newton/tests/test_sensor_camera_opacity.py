# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import math
import unittest

import numpy as np
import warp as wp

import newton
from newton.sensors import SensorCamera


class TestSensorCameraOpacity(unittest.TestCase):
    def test_fully_transparent_shape_does_not_occlude(self) -> None:
        """A shape with opacity 0 is skipped, so the shape behind it stays visible."""
        builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
        ball_body = builder.add_body(xform=wp.transform(p=wp.vec3(0.0, 0.0, -3.0), q=wp.quat_identity()))
        ball = builder.add_shape_sphere(ball_body, radius=0.5, color=(1.0, 0.0, 0.0))
        helper_body = builder.add_body(xform=wp.transform(p=wp.vec3(0.0, 0.0, -1.5), q=wp.quat_identity()))
        helper = builder.add_shape_box(helper_body, hx=1.0, hy=1.0, hz=0.05, opacity=0.0)
        model = builder.finalize(device="cpu")

        camera = SensorCamera(model)
        rays = SensorCamera.compute_camera_rays_pinhole(8, 8, camera_fov=math.radians(40.0), device="cpu")
        transforms = wp.array(
            np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]], dtype=np.float32), dtype=wp.transformf, device="cpu"
        )
        depth = camera.create_depth_image_output(1, 8, 8)
        shape_index = camera.create_shape_index_image_output(1, 8, 8)
        camera.update(model.state(), transforms, rays, depth_image=depth, shape_index_image=shape_index)

        self.assertEqual(int(shape_index.numpy()[0, 4, 4]), ball)
        self.assertNotIn(helper, np.unique(shape_index.numpy()))
        self.assertAlmostEqual(float(depth.numpy()[0, 4, 4]), 2.5, delta=0.05)


if __name__ == "__main__":
    unittest.main()
