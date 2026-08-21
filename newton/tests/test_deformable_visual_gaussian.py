# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Tests for Gaussian visual payloads embedded in deformable bodies."""

import math
import unittest

import numpy as np
import warp as wp

import newton
from newton.sensors import SensorTiledCamera


def _soft_builder():
    builder = newton.ModelBuilder()
    builder.add_soft_grid(
        pos=wp.vec3(0.0),
        rot=wp.quat_identity(),
        vel=wp.vec3(0.0),
        dim_x=1,
        dim_y=1,
        dim_z=1,
        cell_x=1.0,
        cell_y=1.0,
        cell_z=1.0,
        density=100.0,
        k_mu=1.0e4,
        k_lambda=1.0e4,
        k_damp=0.0,
    )
    return builder


def _quat_matrix(quaternion):
    x, y, z, w = quaternion
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float32,
    )


class TestDeformableVisualGaussianBuilder(unittest.TestCase):
    """Public builder and finalized model behavior."""

    def test_add_tet_bound_gaussian_visual(self):
        """Create a stable model record for a tetrahedron-bound Gaussian field."""
        builder = _soft_builder()
        positions = np.array([[0.2, 0.2, 0.2], [0.4, 0.2, 0.2], [0.2, 0.4, 0.2], [0.2, 0.2, 0.4]], dtype=np.float32)
        gaussian = newton.Gaussian(positions=positions, scales=np.full((4, 3), 0.05, dtype=np.float32))
        index = builder.add_deformable_visual_gaussian(
            gaussian,
            kind="tet",
            tet_range=(0, builder.tet_count),
            parent=np.zeros(4, dtype=np.int32),
            weights=np.full((4, 4), 0.25, dtype=np.float32),
            label="soft_splats",
        )

        model = builder.finalize()

        self.assertEqual(index, 0)
        self.assertEqual(model.deformable_visual_gaussian_count, 1)
        visual = model.deformable_visual_gaussians[index]
        self.assertIsInstance(visual, newton.DeformableVisualGaussian)
        self.assertIs(visual.gaussian, gaussian)
        self.assertEqual(visual.kind, newton.DeformableVisualBinding.Kind.TET)
        self.assertEqual(visual.count, 4)
        self.assertEqual(visual.label, "soft_splats")
        self.assertEqual(visual.index, index)

    def test_replicate_offsets_gaussian_drivers(self):
        """Replicate Gaussian bindings into distinct worlds without copying rest appearance."""
        source = _soft_builder()
        positions = np.array([[0.2, 0.2, 0.2], [0.4, 0.2, 0.2]], dtype=np.float32)
        gaussian = newton.Gaussian(positions=positions, scales=np.full((2, 3), 0.05, dtype=np.float32))
        source.add_deformable_visual_gaussian(
            gaussian,
            kind="tet",
            tet_range=(0, source.tet_count),
            parent=np.array([0, 1], dtype=np.int32),
            weights=np.full((2, 4), 0.25, dtype=np.float32),
            label="soft_splats",
        )

        builder = newton.ModelBuilder()
        builder.replicate(source, 2)
        model = builder.finalize()

        self.assertEqual(model.deformable_visual_gaussian_count, 2)
        self.assertEqual([visual.world for visual in model.deformable_visual_gaussians], [0, 1])
        np.testing.assert_array_equal(model.deformable_visual_gaussians[0].parent.numpy(), [0, 1])
        np.testing.assert_array_equal(
            model.deformable_visual_gaussians[1].parent.numpy(), np.array([0, 1]) + source.tet_count
        )
        self.assertIs(model.deformable_visual_gaussians[0].gaussian, gaussian)
        self.assertIs(model.deformable_visual_gaussians[1].gaussian, gaussian)

    def test_rejects_invalid_gaussian_visual_data(self):
        """Reject malformed appearance and unsupported bindings before finalization."""
        builder = _soft_builder()
        positions = np.array([[0.2, 0.2, 0.2]], dtype=np.float32)
        bad_sh = np.array([[np.nan, 0.0, 0.0]], dtype=np.float32)
        gaussian = newton.Gaussian(positions=positions, sh_coeffs=bad_sh)

        with self.assertRaisesRegex(ValueError, "Gaussian data must be finite"):
            builder.add_deformable_visual_gaussian(
                gaussian,
                kind="tet",
                tet_range=(0, builder.tet_count),
                parent=[0],
                weights=np.full((1, 4), 0.25, dtype=np.float32),
            )

        gaussian = newton.Gaussian(positions=positions)
        with self.assertRaisesRegex(ValueError, "supports only kind='tet'"):
            builder.add_deformable_visual_gaussian(
                gaussian,
                kind="particle",
                tet_range=(0, builder.tet_count),
            )


class TestDeformableVisualGaussianEvaluation(unittest.TestCase):
    """Reusable current Gaussian transforms and scales."""

    def test_update_evaluates_tet_center_and_covariance(self):
        """Evaluate one Gaussian center and covariance through the shared visual result."""
        builder = _soft_builder()
        tet_indices = np.asarray(builder.tet_indices, dtype=np.int32).reshape(-1, 4)
        corners = np.asarray(builder.particle_q, dtype=np.float32)[tet_indices[0]]
        center = corners.mean(axis=0, keepdims=True)
        rest_scale = np.array([[0.1, 0.2, 0.3]], dtype=np.float32)
        gaussian = newton.Gaussian(positions=center, scales=rest_scale)
        builder.add_deformable_visual_gaussian(
            gaussian,
            kind="tet",
            tet_range=(0, builder.tet_count),
            parent=[0],
            weights=np.full((1, 4), 0.25, dtype=np.float32),
        )
        model = builder.finalize()
        state = model.state()
        visuals = model.deformable_visuals()

        returned = model.update_deformable_visuals(state, visuals)

        self.assertIs(returned, visuals)
        transforms = visuals.get_gaussian_transforms(0).numpy()
        scales = visuals.get_gaussian_scales(0).numpy()
        np.testing.assert_allclose(transforms[0, :3], center[0], atol=1.0e-6)
        np.testing.assert_allclose(np.sort(scales[0]), np.sort(rest_scale[0]), atol=1.0e-6)
        self.assertIs(visuals.state, state)

    def test_shear_updates_full_covariance(self):
        """Preserve the independently computed Gaussian covariance under affine shear."""
        builder = _soft_builder()
        tet_indices = np.asarray(builder.tet_indices, dtype=np.int32).reshape(-1, 4)
        rest_particles = np.asarray(builder.particle_q, dtype=np.float32)
        center = rest_particles[tet_indices[0]].mean(axis=0, keepdims=True)
        rest_scale = np.array([[0.08, 0.13, 0.21]], dtype=np.float32)
        gaussian = newton.Gaussian(positions=center, scales=rest_scale)
        builder.add_deformable_visual_gaussian(
            gaussian,
            kind="tet",
            tet_range=(0, builder.tet_count),
            parent=[0],
            weights=np.full((1, 4), 0.25, dtype=np.float32),
        )
        model = builder.finalize()
        state = model.state()
        shear = np.array([[1.0, 0.35, 0.0], [0.0, 1.0, 0.2], [0.0, 0.0, 0.7]], dtype=np.float32)
        state.particle_q.assign(rest_particles @ shear.T)
        visuals = model.deformable_visuals()

        model.update_deformable_visuals(state, visuals)

        transform = visuals.get_gaussian_transforms(0).numpy()[0]
        scales = visuals.get_gaussian_scales(0).numpy()[0]
        rotation = _quat_matrix(transform[3:7])
        actual_covariance = rotation @ np.diag(scales**2) @ rotation.T
        expected_covariance = shear @ np.diag(rest_scale[0] ** 2) @ shear.T
        np.testing.assert_allclose(actual_covariance, expected_covariance, atol=2.0e-6)
        np.testing.assert_allclose(transform[:3], (center @ shear.T)[0], atol=1.0e-6)

    def test_update_is_cuda_graph_capturable(self):
        """Replay Gaussian evaluation into the same output buffers during CUDA capture."""
        builder = _soft_builder()
        tet_indices = np.asarray(builder.tet_indices, dtype=np.int32).reshape(-1, 4)
        rest_particles = np.asarray(builder.particle_q, dtype=np.float32)
        center = rest_particles[tet_indices[0]].mean(axis=0, keepdims=True)
        builder.add_deformable_visual_gaussian(
            newton.Gaussian(positions=center, scales=np.full((1, 3), 0.1, dtype=np.float32)),
            kind="tet",
            tet_range=(0, builder.tet_count),
            parent=[0],
            weights=np.full((1, 4), 0.25, dtype=np.float32),
        )
        model = builder.finalize()
        if not model.device.is_cuda:
            self.skipTest("CUDA graph capture requires a CUDA device")
        state = model.state()
        visuals = model.deformable_visuals()
        model.update_deformable_visuals(state, visuals)
        transforms_ptr = visuals.gaussian_transforms.ptr
        scales_ptr = visuals.gaussian_scales.ptr

        with wp.ScopedCapture(model.device) as capture:
            model.update_deformable_visuals(state, visuals)

        moved = rest_particles.copy()
        moved[:, 2] += 0.4
        state.particle_q.assign(moved)
        wp.capture_launch(capture.graph)
        visuals.wait()

        self.assertEqual(visuals.gaussian_transforms.ptr, transforms_ptr)
        self.assertEqual(visuals.gaussian_scales.ptr, scales_ptr)
        np.testing.assert_allclose(visuals.get_gaussian_transforms(0).numpy()[0, :3], center[0] + [0, 0, 0.4])


class TestDeformableVisualGaussianSensor(unittest.TestCase):
    """Camera consumption through the public deformable visual output."""

    def test_tiled_camera_tracks_tet_bound_gaussian(self):
        """Render a Gaussian visual without a separate static Gaussian shape."""
        builder = _soft_builder()
        tet_indices = np.asarray(builder.tet_indices, dtype=np.int32).reshape(-1, 4)
        rest_particles = np.asarray(builder.particle_q, dtype=np.float32)
        center = rest_particles[tet_indices[0]].mean(axis=0, keepdims=True)
        gaussian = newton.Gaussian(
            positions=center,
            scales=np.full((1, 3), 0.18, dtype=np.float32),
            opacities=np.array([0.95], dtype=np.float32),
        )
        builder.add_deformable_visual_gaussian(
            gaussian,
            kind="tet",
            tet_range=(0, builder.tet_count),
            parent=[0],
            weights=np.full((1, 4), 0.25, dtype=np.float32),
            label="soft_splat",
        )
        model = builder.finalize()
        state = model.state()

        sensor = SensorTiledCamera(
            model,
            default_render_config=SensorTiledCamera.RenderConfig(
                enable_particles=False,
                enable_simulation_triangles=False,
                gaussians_mode=SensorTiledCamera.GaussianRenderMode.QUALITY,
                max_distance=10.0,
            ),
        )
        width = 32
        height = 32
        camera_rays = sensor.utils.compute_camera_rays_pinhole(width, height, camera_fovs=math.radians(40.0))
        camera_transforms = wp.array(
            [[wp.transformf(wp.vec3f(float(center[0, 0]), float(center[0, 1]), 2.0), wp.quat_identity())]],
            dtype=wp.transformf,
            device=model.device,
        )
        depth_image = sensor.utils.create_depth_image_output(width, height, camera_count=1)

        sensor.update(state, camera_transforms, camera_rays, depth_image=depth_image)
        rest_depth = float(depth_image.numpy()[0, 0, height // 2, width // 2])

        moved = rest_particles.copy()
        moved[:, 2] += 0.4
        state.particle_q.assign(moved)
        sensor.update(state, camera_transforms, camera_rays, depth_image=depth_image)
        moved_depth = float(depth_image.numpy()[0, 0, height // 2, width // 2])

        self.assertGreater(rest_depth, 0.0)
        self.assertGreater(moved_depth, 0.0)
        self.assertAlmostEqual(rest_depth - moved_depth, 0.4, delta=0.1)

    def test_tiled_camera_keeps_replicated_gaussians_independent(self):
        """Render one independently deformed Gaussian field in each world."""
        prototype = _soft_builder()
        tet_indices = np.asarray(prototype.tet_indices, dtype=np.int32).reshape(-1, 4)
        rest_particles = np.asarray(prototype.particle_q, dtype=np.float32)
        center = rest_particles[tet_indices[0]].mean(axis=0, keepdims=True)
        prototype.add_deformable_visual_gaussian(
            newton.Gaussian(
                positions=center,
                scales=np.full((1, 3), 0.18, dtype=np.float32),
                opacities=np.array([0.95], dtype=np.float32),
            ),
            kind="tet",
            tet_range=(0, prototype.tet_count),
            parent=[0],
            weights=np.full((1, 4), 0.25, dtype=np.float32),
        )
        builder = newton.ModelBuilder()
        builder.replicate(prototype, 2)
        model = builder.finalize()
        state = model.state()

        moved = state.particle_q.numpy()
        particle_world = model.particle_world.numpy()
        moved[particle_world == 0, 2] += 0.4
        state.particle_q.assign(moved)

        sensor = SensorTiledCamera(
            model,
            default_render_config=SensorTiledCamera.RenderConfig(
                enable_particles=False,
                enable_simulation_triangles=False,
                gaussians_mode=SensorTiledCamera.GaussianRenderMode.QUALITY,
                max_distance=10.0,
            ),
        )
        width = 32
        height = 32
        camera_rays = sensor.utils.compute_camera_rays_pinhole(width, height, camera_fovs=math.radians(40.0))
        camera = wp.transformf(wp.vec3f(float(center[0, 0]), float(center[0, 1]), 2.0), wp.quat_identity())
        camera_transforms = wp.array([[camera, camera]], dtype=wp.transformf, device=model.device)
        depth_image = sensor.utils.create_depth_image_output(width, height, camera_count=1)

        sensor.update(state, camera_transforms, camera_rays, depth_image=depth_image)
        depth = depth_image.numpy()[:, 0, height // 2, width // 2]

        self.assertTrue(np.all(depth > 0.0))
        self.assertAlmostEqual(float(depth[1] - depth[0]), 0.4, delta=0.1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
