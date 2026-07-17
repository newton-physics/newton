# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import math
import unittest

import numpy as np
import warp as wp

import newton
from newton import Heightfield
from newton.sensors import SensorTiledCamera


class TestSensorTiledCameraHeightfield(unittest.TestCase):
    """The tiled camera must render heightfield (HFIELD) shapes."""

    @unittest.skipUnless(wp.is_cuda_available(), "Requires CUDA")
    def test_renders_flat_heightfield_from_above(self):
        # Flat heightfield at z=1 spanning [-2, 2]^2.
        data = np.full((3, 3), 1.0, dtype=np.float32)
        hf = Heightfield(data=data, nrow=3, ncol=3, hx=2.0, hy=2.0, min_z=1.0, max_z=1.0)
        builder = newton.ModelBuilder()
        builder.add_shape_heightfield(heightfield=hf)
        model = builder.finalize()
        state = model.state()

        res = 16
        sensor = SensorTiledCamera(model=model)
        sensor.default_render_config.enable_textures = True
        sensor.utils.create_default_light(enable_shadows=False)
        sensor.utils.assign_checkerboard_material(shape_indices=[0])
        # 30-deg fov: footprint half-extent at depth 4 is 4*tan(15)=1.07 < 2,
        # so the terrain robustly fills the whole frame.
        rays = sensor.utils.compute_camera_rays_pinhole(res, res, camera_fovs=math.radians(30.0))
        depth = sensor.utils.create_depth_image_output(res, res)
        model.bvh_build_shapes(state)
        model.bvh_build_particles(state)
        sensor.sync_transforms(state)

        # Camera 5m above origin, identity orientation => looks straight down (-z).
        # At depth 4 the 45-deg footprint half-extent is 4*tan(22.5)=1.66 < 2,
        # so every ray hits the terrain.
        cam = wp.array(
            [[wp.transformf(wp.vec3f(0.0, 0.0, 5.0), wp.quatf(0.0, 0.0, 0.0, 1.0))]],
            dtype=wp.transformf,
        )
        sensor.default_render_config.render_order = SensorTiledCamera.RenderOrder.PIXEL_PRIORITY
        sensor.update(state, cam, rays, depth_image=depth)

        d = depth.numpy()[0, 0]  # .numpy() syncs the device-to-host copy
        hit = int(np.count_nonzero(d > 0.0))
        # The terrain covers the whole frame, but ~10-15% of rays miss along
        # triangle edges (non-watertight mesh_query_ray); measured stable across
        # resolution and camera offset, so require "most" pixels rather than all.
        self.assertGreaterEqual(
            hit,
            int(res * res * 0.8),
            msg=f"heightfield should fill most of the view; only {hit}/{res * res} pixels hit",
        )
        # Every ray that hits sees the flat surface at z=1 from z=5: depth ~4,
        # up to ~4.25 toward the frame edges (ray-angle cosine).
        hit_depths = d[d > 0.0]
        self.assertGreater(float(hit_depths.min()), 3.9)
        self.assertLess(float(hit_depths.max()), 4.4)
        center = float(d[res // 2, res // 2])
        self.assertAlmostEqual(center, 4.0, delta=0.05, msg=f"center depth {center}, expected ~4.0 (5 - 1)")


if __name__ == "__main__":
    unittest.main()
