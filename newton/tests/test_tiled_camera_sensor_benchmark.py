# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
import unittest

import warp as wp

import newton
from newton.sensors import TiledCameraSensor


class TestTiledCameraSensorBenchmark(unittest.TestCase):
    def test_benchmark(self):
        franka = newton.ModelBuilder()
        franka.add_urdf(
            newton.utils.download_asset("franka_emika_panda") / "urdf/fr3_franka_hand.urdf",
            floating=False,
        )

        resolution = 64
        num_worlds = 4096
        warmup = 20
        iterations = 50

        scene = newton.ModelBuilder()
        scene.replicate(franka, num_worlds)
        scene.add_ground_plane()

        model = scene.finalize()
        state = model.state()
        newton.eval_fk(model, model.joint_q, model.joint_qd, state)

        # camera_positions = wp.array([wp.vec3f(2.0, 0.0, 0.6)], dtype=wp.vec3f)
        # camera_orientations = wp.array([wp.mat33f(0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0)], dtype=wp.mat33f)
        camera_positions = wp.array([2.4, 0.0, 0.8], dtype=wp.vec3f)
        camera_orientations = wp.array(
            [
                -0.008726535,
                -0.29236057,
                0.95626837,
                0.9999619,
                -0.002551392,
                0.008345228,
                1.3010426e-18,
                0.9563047,
                0.2923717,
            ],
            dtype=wp.mat33f,
        )

        tiled_camera_sensor = TiledCameraSensor(model=model, num_cameras=1, width=resolution, height=resolution)
        tiled_camera_sensor.assign_debug_colors_per_shape()
        tiled_camera_sensor.create_default_light(False)
        tiled_camera_sensor.compute_camera_rays(wp.array([math.radians(45.0)], dtype=wp.float32))
        color_image = tiled_camera_sensor.create_color_image_output()
        depth_image = tiled_camera_sensor.create_depth_image_output()

        tiled_camera_sensor.update_cameras(camera_positions, camera_orientations)

        tiled_camera_sensor.update_from_state(state)
        with wp.ScopedTimer("Refit BVH", synchronize=True):
            tiled_camera_sensor.render_context.refit_bvh()

        with wp.ScopedTimer("Warmup", synchronize=True):
            for _ in range(warmup):
                tiled_camera_sensor.render(color_image, depth_image, refit_bvh=False, clear_images=False)

        with wp.ScopedTimer("Rendering", synchronize=True) as timer:
            for _ in range(iterations):
                tiled_camera_sensor.render(color_image, depth_image, refit_bvh=False, clear_images=False)

        print(f"Average: {timer.elapsed / iterations:.2f} ms")

        tiled_camera_sensor.save_color_image(color_image, "example_color.png")
        tiled_camera_sensor.save_depth_image(depth_image, "example_depth.png")


if __name__ == "__main__":
    unittest.main()
