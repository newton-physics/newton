# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import math
import tempfile
import unittest
from pathlib import Path

import numpy as np
import warp as wp

import newton
from newton.sensors import SensorCamera
from newton.tests.unittest_utils import get_test_devices


class TestSensorCameraDeformable(unittest.TestCase):
    """Render native visual meshes without USD or a viewer window."""

    @staticmethod
    def _builder():
        builder = newton.ModelBuilder()
        points = np.array([[-0.5, -0.5, 0], [0.5, -0.5, 0], [0.5, 0.5, 0], [-0.5, 0.5, 0]])
        particles = [builder.add_particle(pos=tuple(p), vel=(0, 0, 0), mass=1, radius=0) for p in points]
        builder.add_deformable_visual_mesh(points, [0, 1, 2, 0, 2, 3], kind="particle", parent=particles)
        return builder

    @staticmethod
    def _camera_arrays(model, view_count=1):
        rays = SensorCamera.compute_camera_rays_pinhole(16, 16, device=model.device, camera_fov=math.radians(45))
        transforms = wp.array(
            [wp.transform(wp.vec3(0, 0, 2), wp.quat_identity())] * view_count,
            dtype=wp.transform,
            device=model.device,
        )
        return rays, transforms

    @staticmethod
    def _mixed_builder(texture=None):
        builder = newton.ModelBuilder()
        points = np.array([[-0.5, -0.5, 0], [0.5, -0.5, 0], [0.5, 0.5, 0], [-0.5, 0.5, 0]])
        for point in points:
            builder.add_particle(pos=wp.vec3(*point), vel=wp.vec3(0), mass=1, radius=0.8)
        builder.add_triangle(0, 1, 2)
        builder.add_triangle(0, 2, 3)
        # These large particles would obscure the skin if hiding the cloth exposed them.
        body = builder.add_body(xform=wp.transform(wp.vec3(0, 0, -0.5), wp.quat_identity()))
        builder.add_deformable_visual_mesh(
            points + np.array([0, 0, -0.5]),
            [0, 1, 2, 0, 2, 3],
            kind="body",
            bodies=[body],
            uvs=[[1, 0], [1.5, 0], [1.5, 1], [1, 1]],
            texture=texture,
        )
        builder.add_shape_sphere(-1, xform=wp.transform(wp.vec3(-2, 0, 0), wp.quat_identity()), radius=0.3)
        builder.add_particle(pos=(2, 0, 0), vel=(0, 0, 0), mass=1, radius=0.3)
        return builder

    @staticmethod
    def _rgb(packed):
        return np.array([(int(packed) >> shift) & 255 for shift in (0, 8, 16)])

    def test_visual_depth_and_world_mapping(self):
        """Render independently moved visual meshes in the requested world order."""
        prototype = self._builder()
        builder = newton.ModelBuilder()
        builder.replicate(prototype, 2)
        for device in get_test_devices():
            with self.subTest(device=device):
                model = builder.finalize(device=device)
                state = model.state()
                camera = SensorCamera(model, default_render_config=SensorCamera.RenderConfig(enable_particles=False))
                rays, transforms = self._camera_arrays(model, 3)
                worlds = wp.array([1, 0, 1], dtype=wp.int32, device=device)
                depth = camera.allocate_image_depth(3, 16, 16)
                camera.update(state, transforms, rays, world_indices=worlds, depth_image=depth)
                np.testing.assert_allclose(depth.numpy()[:, 8, 8], [2, 2, 2], atol=0.03)

                moved = state.particle_q.numpy().copy()
                moved[model.particle_world.numpy() == 0, 2] += 0.4
                state.particle_q.assign(moved)
                camera.update(state, transforms, rays, world_indices=worlds, depth_image=depth)
                np.testing.assert_allclose(depth.numpy()[:, 8, 8], [2, 1.6, 2], atol=0.03)

    def test_simulation_surface_visibility(self):
        """Hide the coarse surface without exposing its particles or hiding other geometry."""
        for device in get_test_devices():
            model = self._mixed_builder().finalize(device=device)
            state = model.state()
            rays, _ = self._camera_arrays(model)
            transforms = wp.array(
                [wp.transform(wp.vec3(x, 0, 2), wp.quat_identity()) for x in (0, -2, 2)],
                dtype=wp.transform,
                device=device,
            )
            worlds = wp.zeros(3, dtype=wp.int32, device=device)
            for show_simulation in (True, False):
                with self.subTest(device=device, show_simulation=show_simulation):
                    camera = SensorCamera(
                        model,
                        default_render_config=SensorCamera.RenderConfig(enable_simulation_triangles=show_simulation),
                    )
                    depth = camera.allocate_image_depth(3, 16, 16)
                    albedo = camera.allocate_image_albedo(3, 16, 16)
                    camera.update(state, transforms, rays, world_indices=worlds, depth_image=depth, albedo_image=albedo)
                    np.testing.assert_allclose(
                        depth.numpy()[:, 8, 8], [2 if show_simulation else 2.5, 1.7, 1.7], atol=0.03
                    )
                    expected = model.tri_color.numpy()[0] * 255 if show_simulation else [255, 255, 255]
                    np.testing.assert_allclose(self._rgb(albedo.numpy()[0, 8, 8]), expected, atol=2)

    def test_shared_visuals_and_capture(self):
        """Consume supplied buffers as-is and replay updates from alternating states."""
        for device in get_test_devices():
            with self.subTest(device=device):
                model = self._mixed_builder().finalize(device=device)
                state_a, state_b = model.state(), model.state()
                visuals = model.deformable_visuals()
                model.update_deformable_visuals(state_a, visuals)
                moved = state_a.body_q.numpy().copy()
                moved[:, 2] += 1
                state_a.body_q.assign(moved)
                camera = SensorCamera(
                    model,
                    default_render_config=SensorCamera.RenderConfig(enable_simulation_triangles=False),
                )
                rays, transforms = self._camera_arrays(model)
                depth = camera.allocate_image_depth(1, 16, 16)
                camera.update(state_a, transforms, rays, depth_image=depth, deformable_visuals=visuals)
                self.assertAlmostEqual(float(depth.numpy()[0, 8, 8]), 2.5, delta=0.03)

                camera.sync_deformable_meshes(state_a, deformable_visuals=visuals)
                camera.update(state_a, transforms, rays, depth_image=depth, sync_deformables=False)
                self.assertAlmostEqual(float(depth.numpy()[0, 8, 8]), 2.5, delta=0.03)
                camera.update(state_a, transforms, rays, depth_image=depth)
                self.assertAlmostEqual(float(depth.numpy()[0, 8, 8]), 1.5, delta=0.03)

                if device.is_cuda:
                    # Warm both evaluation and rendering before capturing the full path.
                    graphs = []
                    for state in (state_a, state_b):
                        model.update_deformable_visuals(state, visuals)
                        camera.update(state, transforms, rays, depth_image=depth, deformable_visuals=visuals)
                        with wp.ScopedCapture(device) as capture:
                            model.update_deformable_visuals(state, visuals)
                            camera.update(state, transforms, rays, depth_image=depth, deformable_visuals=visuals)
                        graphs.append(capture.graph)
                    for graph, expected in ((graphs[0], 1.5), (graphs[1], 2.5), (graphs[0], 1.5)):
                        wp.capture_launch(graph)
                        self.assertAlmostEqual(float(depth.numpy()[0, 8, 8]), expected, delta=0.03)

                with self.assertRaisesRegex(ValueError, "requires sync_deformables=True"):
                    camera.update(
                        state_a,
                        transforms,
                        rays,
                        depth_image=depth,
                        sync_deformables=False,
                        deformable_visuals=visuals,
                    )
                with self.assertRaisesRegex(ValueError, "another model"):
                    other = self._builder().finalize(device=device)
                    camera.sync_deformable_meshes(state_a, deformable_visuals=other.deformable_visuals())

    def test_textures_and_checkerboard(self):
        """Sample wrapped UVs and preserve skin textures when assigning a ground checkerboard."""
        rgb = np.empty((4, 4, 3), dtype=np.uint8)
        rgb[:, :2] = [200, 40, 10]
        rgb[:, 2:] = [10, 200, 40]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "skin.ppm"
            path.write_bytes(b"P6\n4 4\n255\n" + rgb.tobytes())
            sources = (
                (rgb, [200, 40, 10]),
                (np.concatenate([rgb, np.full((4, 4, 1), 255, np.uint8)], axis=2), [200, 40, 10]),
                (np.full((4, 4), 90, np.uint8), [90, 90, 90]),
                (str(path), [200, 40, 10]),
            )
            for device in get_test_devices():
                for source, expected in sources:
                    model = self._mixed_builder(source).finalize(device=device)
                    state = model.state()
                    camera = SensorCamera(
                        model,
                        default_render_config=SensorCamera.RenderConfig(
                            enable_simulation_triangles=False, enable_textures=True
                        ),
                    )
                    _, transforms = self._camera_arrays(model)
                    albedo = camera.allocate_image_albedo(1, 16, 16)
                    color = camera.allocate_image_color(1, 16, 16)
                    camera.set_ambient_light((1, 1, 1))
                    for mode in SensorCamera.AntiAliasing:
                        with self.subTest(device=device, source=type(source).__name__, mode=mode):
                            rays = SensorCamera.compute_camera_rays_pinhole(
                                16,
                                16,
                                camera_fov=math.radians(45),
                                device=device,
                                sample_count=1 if mode == SensorCamera.AntiAliasing.NONE else 4,
                            )
                            camera.default_render_config.anti_aliasing = mode
                            camera.update(state, transforms, rays, albedo_image=albedo, color_image=color)
                            np.testing.assert_allclose(self._rgb(albedo.numpy()[0, 8, 8]), expected, atol=2)
                            np.testing.assert_allclose(self._rgb(color.numpy()[0, 8, 8]), expected, atol=2)
                    for _ in range(2):
                        camera.assign_checkerboard_material(shape_indices=[0])
                        camera.update(state, transforms, rays, albedo_image=albedo, color_image=color)
                        np.testing.assert_allclose(self._rgb(albedo.numpy()[0, 8, 8]), expected, atol=2)


if __name__ == "__main__":
    unittest.main()
