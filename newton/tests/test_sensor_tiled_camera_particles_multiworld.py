# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
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

import numpy as np
import warp as wp

import newton
from newton.sensors import SensorTiledCamera
from newton.tests.unittest_utils import add_function_test, get_test_devices


def _build_multiworld_particle_model(*, worlds: int, spacing: float):
    """Build a tiny multi-world particle model for SensorTiledCamera regression tests.

    This scene is intentionally minimal and deterministic:
    - each world has the same particle grid
    - worlds are translated in +X to avoid overlap
    - each world gets its own camera placed above the particle block

    The rendered depth image should be identical across worlds.
    """
    if worlds <= 0:
        raise ValueError("worlds must be positive")
    if spacing <= 0.0:
        raise ValueError("spacing must be positive")

    # Per-world particle blueprint.
    # Keep particle counts small so this can run on CPU CI as well.
    cell = 0.10
    radius = 0.04
    blueprint = newton.ModelBuilder(up_axis=newton.Axis.Z)
    blueprint.default_particle_radius = radius
    blueprint.add_particle_grid(
        pos=wp.vec3(-0.4, -0.4, 0.1),
        rot=wp.quat_identity(),
        vel=wp.vec3(0.0, 0.0, 0.0),
        dim_x=8,
        dim_y=8,
        dim_z=2,
        cell_x=cell,
        cell_y=cell,
        cell_z=cell,
        mass=0.0,
        jitter=0.0,
        radius_mean=radius,
        radius_std=0.0,
    )

    # Multi-world model (world-local positions are translated by add_world() xform).
    builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
    for world_id in range(int(worlds)):
        builder.add_world(
            blueprint,
            xform=wp.transform(wp.vec3(float(world_id) * float(spacing), 0.0, 0.0), wp.quat_identity()),
        )

    return builder


def test_sensor_tiled_camera_multiworld_particles_consistent(test: unittest.TestCase, device):
    """Regression test: multi-world particle depth should be consistent across worlds.

    This catches incorrect BVH particle index mapping across worlds, which can cause
    wrong depth images and, on CUDA, illegal memory accesses.
    """
    wp.init()

    worlds = 4
    spacing = 10.0

    width = 32
    height = 24
    fov = math.radians(60.0)
    max_distance = 10.0

    cam_local_pos = wp.vec3f(0.0, 0.0, 3.0)

    builder = _build_multiworld_particle_model(worlds=worlds, spacing=spacing)
    with wp.ScopedDevice(device):
        model = builder.finalize()

    state = model.state()

    sensor = SensorTiledCamera(model=model, options=SensorTiledCamera.Options())
    sensor.render_context.options.max_distance = max_distance
    camera_rays = sensor.compute_pinhole_camera_rays(width, height, fov)

    cam_quat = wp.quat_identity()
    camera_transforms = wp.array(
        [
            [
                wp.transformf(
                    wp.vec3f(cam_local_pos[0] + float(world_id) * float(spacing), cam_local_pos[1], cam_local_pos[2]),
                    cam_quat,
                )
                for world_id in range(worlds)
            ]
        ],
        dtype=wp.transformf,
        device=device,
    )

    depth_image = sensor.create_depth_image_output(width, height, num_cameras=1)
    sensor.render(state, camera_transforms, camera_rays, depth_image=depth_image)
    wp.synchronize()

    depth_np = depth_image.numpy()  # (num_worlds, num_cameras, H, W)

    # Sanity: ensure this scene actually produces hits.
    hit_count = int(np.count_nonzero(depth_np[0, 0] > 0.0))
    test.assertGreater(hit_count, 0, "Expected at least one particle hit in world 0.")

    # Regression: all worlds should render the same depth image.
    # (Identical scene, identical relative camera pose).
    for w in range(1, worlds):
        # We expect numerical agreement up to small fp32 differences introduced by world translations.
        np.testing.assert_allclose(depth_np[0, 0], depth_np[w, 0], rtol=0.0, atol=1e-4)


class TestSensorTiledCameraParticlesMultiworld(unittest.TestCase):
    """Unittest harness for device-parametrized SensorTiledCamera particle regression tests."""

    pass


devices = get_test_devices()
add_function_test(
    TestSensorTiledCameraParticlesMultiworld,
    "test_sensor_tiled_camera_multiworld_particles_consistent",
    test_sensor_tiled_camera_multiworld_particles_consistent,
    devices=devices,
)


if __name__ == "__main__":
    unittest.main(verbosity=2)
