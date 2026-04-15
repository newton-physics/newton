# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import math
import unittest

import numpy as np
import warp as wp

from newton._src.sensors.warp_raytrace.utils import (
    compute_pinhole_camera_rays,
    convert_ray_depth_to_forward_depth_kernel,
)


class TestConvertRayDepthToForwardDepth(unittest.TestCase):
    """Headless tests for ray-depth to forward-depth conversion.

    Only requires Warp arrays (no OpenGL / CUDA), so it can run in any CI
    environment.
    """

    device = "cpu"

    def _launch_pinhole_rays(self, width: int, height: int, fov_rad: float):
        """Launch the pinhole camera rays kernel and return the rays array."""
        camera_fovs = wp.array([fov_rad], dtype=wp.float32, device=self.device)
        camera_count = 1
        camera_rays = wp.empty((camera_count, height, width, 2), dtype=wp.vec3f, device=self.device)
        wp.launch(
            kernel=compute_pinhole_camera_rays,
            dim=(camera_count, height, width),
            inputs=[width, height, camera_fovs, camera_rays],
            device=self.device,
        )
        return camera_rays

    @staticmethod
    def _expected_cos_theta(px: int, py: int, width: int, height: int, fov_rad: float) -> float:
        """Compute cos(theta) between a pixel's ray and the optical axis.

        Mirrors the logic in ``compute_pinhole_camera_rays``:
        the unnormalized ray direction is ``(u*2*h*ar, -v*2*h, -1)`` so
        ``cos(theta) = 1 / ||ray||``.
        """
        aspect = width / height
        u = (px + 0.5) / width - 0.5
        v = (py + 0.5) / height - 0.5
        h = math.tan(fov_rad / 2.0)
        dx = u * 2.0 * h * aspect
        dy = -v * 2.0 * h
        dz = -1.0
        length = math.sqrt(dx * dx + dy * dy + dz * dz)
        return 1.0 / length

    def test_identity_transform_3x3(self):
        """Forward depth equals ray_depth * cos(theta) for an identity camera."""
        width, height = 3, 3
        fov_rad = math.radians(90.0)
        uniform_depth = 10.0

        camera_rays = self._launch_pinhole_rays(width, height, fov_rad)

        world_count, camera_count = 1, 1
        depth_image = wp.full(
            (world_count, camera_count, height, width),
            value=uniform_depth,
            dtype=wp.float32,
            device=self.device,
        )
        out_depth = wp.zeros_like(depth_image)

        camera_transforms = wp.array(
            [[wp.transformf(wp.vec3f(0.0, 0.0, 0.0), wp.quatf(0.0, 0.0, 0.0, 1.0))]],
            dtype=wp.transformf,
            device=self.device,
        )

        wp.launch(
            kernel=convert_ray_depth_to_forward_depth_kernel,
            dim=(world_count, camera_count, height, width),
            inputs=[depth_image, camera_rays, camera_transforms, out_depth],
            device=self.device,
        )

        result = out_depth.numpy()

        for py in range(height):
            for px in range(width):
                cos_theta = self._expected_cos_theta(px, py, width, height, fov_rad)
                expected = uniform_depth * cos_theta
                actual = float(result[0, 0, py, px])
                self.assertAlmostEqual(
                    actual,
                    expected,
                    places=5,
                    msg=f"pixel ({px},{py}): expected {expected:.6f}, got {actual:.6f}",
                )

    def test_center_pixel_unchanged(self):
        """The center pixel's ray is on-axis, so forward depth equals ray depth."""
        width, height = 3, 3
        fov_rad = math.radians(60.0)
        uniform_depth = 5.0

        camera_rays = self._launch_pinhole_rays(width, height, fov_rad)

        depth_image = wp.full((1, 1, height, width), value=uniform_depth, dtype=wp.float32, device=self.device)
        out_depth = wp.zeros_like(depth_image)

        camera_transforms = wp.array(
            [[wp.transformf(wp.vec3f(0.0, 0.0, 0.0), wp.quatf(0.0, 0.0, 0.0, 1.0))]],
            dtype=wp.transformf,
            device=self.device,
        )

        wp.launch(
            kernel=convert_ray_depth_to_forward_depth_kernel,
            dim=(1, 1, height, width),
            inputs=[depth_image, camera_rays, camera_transforms, out_depth],
            device=self.device,
        )

        center = float(out_depth.numpy()[0, 0, 1, 1])
        self.assertAlmostEqual(
            center,
            uniform_depth,
            places=5,
            msg=f"Center pixel forward depth should equal ray depth, got {center}",
        )

    def test_off_axis_strictly_less(self):
        """Off-axis pixels must produce strictly smaller forward depth than the ray depth."""
        width, height = 3, 3
        fov_rad = math.radians(90.0)
        uniform_depth = 8.0

        camera_rays = self._launch_pinhole_rays(width, height, fov_rad)

        depth_image = wp.full((1, 1, height, width), value=uniform_depth, dtype=wp.float32, device=self.device)
        out_depth = wp.zeros_like(depth_image)

        camera_transforms = wp.array(
            [[wp.transformf(wp.vec3f(0.0, 0.0, 0.0), wp.quatf(0.0, 0.0, 0.0, 1.0))]],
            dtype=wp.transformf,
            device=self.device,
        )

        wp.launch(
            kernel=convert_ray_depth_to_forward_depth_kernel,
            dim=(1, 1, height, width),
            inputs=[depth_image, camera_rays, camera_transforms, out_depth],
            device=self.device,
        )

        result = out_depth.numpy()
        for py in range(height):
            for px in range(width):
                if px == 1 and py == 1:
                    continue
                self.assertLess(
                    float(result[0, 0, py, px]),
                    uniform_depth,
                    msg=f"Off-axis pixel ({px},{py}) should have forward depth < ray depth",
                )

    def test_varying_depth(self):
        """Per-pixel ray depths are each scaled by the correct cos(theta)."""
        width, height = 3, 3
        fov_rad = math.radians(70.0)

        camera_rays = self._launch_pinhole_rays(width, height, fov_rad)

        depths_np = np.arange(1.0, 1.0 + width * height, dtype=np.float32).reshape(1, 1, height, width)
        depth_image = wp.array(depths_np, dtype=wp.float32, device=self.device)
        out_depth = wp.zeros_like(depth_image)

        camera_transforms = wp.array(
            [[wp.transformf(wp.vec3f(0.0, 0.0, 0.0), wp.quatf(0.0, 0.0, 0.0, 1.0))]],
            dtype=wp.transformf,
            device=self.device,
        )

        wp.launch(
            kernel=convert_ray_depth_to_forward_depth_kernel,
            dim=(1, 1, height, width),
            inputs=[depth_image, camera_rays, camera_transforms, out_depth],
            device=self.device,
        )

        result = out_depth.numpy()
        for py in range(height):
            for px in range(width):
                cos_theta = self._expected_cos_theta(px, py, width, height, fov_rad)
                ray_depth = depths_np[0, 0, py, px]
                expected = ray_depth * cos_theta
                actual = float(result[0, 0, py, px])
                self.assertAlmostEqual(
                    actual,
                    expected,
                    places=4,
                    msg=f"pixel ({px},{py}): depth={ray_depth}, expected {expected:.6f}, got {actual:.6f}",
                )

    def test_symmetry(self):
        """Pixels equidistant from center should produce equal forward depth."""
        width, height = 3, 3
        fov_rad = math.radians(90.0)
        uniform_depth = 10.0

        camera_rays = self._launch_pinhole_rays(width, height, fov_rad)

        depth_image = wp.full((1, 1, height, width), value=uniform_depth, dtype=wp.float32, device=self.device)
        out_depth = wp.zeros_like(depth_image)

        camera_transforms = wp.array(
            [[wp.transformf(wp.vec3f(0.0, 0.0, 0.0), wp.quatf(0.0, 0.0, 0.0, 1.0))]],
            dtype=wp.transformf,
            device=self.device,
        )

        wp.launch(
            kernel=convert_ray_depth_to_forward_depth_kernel,
            dim=(1, 1, height, width),
            inputs=[depth_image, camera_rays, camera_transforms, out_depth],
            device=self.device,
        )

        result = out_depth.numpy()[0, 0]

        corner_values = [result[0, 0], result[0, 2], result[2, 0], result[2, 2]]
        for v in corner_values[1:]:
            self.assertAlmostEqual(float(v), float(corner_values[0]), places=5)

        edge_values = [result[0, 1], result[1, 0], result[1, 2], result[2, 1]]
        for v in edge_values[1:]:
            self.assertAlmostEqual(float(v), float(edge_values[0]), places=5)


if __name__ == "__main__":
    unittest.main()
