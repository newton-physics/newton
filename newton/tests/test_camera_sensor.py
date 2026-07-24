# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import inspect
import math
import unittest
from unittest import mock

import numpy as np
import warp as wp

import newton
import newton.geometry as geometry
from newton._src.sensors.camera_sensor_renderer.utils import Utils
from newton.sensors import CameraSensor


class TestCameraSensor(unittest.TestCase):
    @staticmethod
    def _make_pinhole_camera(width: int, height: int, fov: float = math.radians(45.0)) -> CameraSensor:
        rays = CameraSensor.compute_camera_rays_pinhole(width, height, fov, device="cpu")
        return CameraSensor(rays)

    @staticmethod
    def _build_sphere_camera_scene(
        width: int,
        height: int,
        *,
        camera: CameraSensor | None = None,
        camera_label: str = "camera",
    ) -> tuple[newton.Model, CameraSensor]:
        builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
        sphere_body = builder.add_body(xform=wp.transform(p=wp.vec3(0.0, 0.0, -2.0), q=wp.quat_identity()))
        builder.add_shape_sphere(sphere_body, radius=0.75, color=(0.25, 0.5, 0.75))

        camera_body = builder.add_body(xform=wp.transform(p=wp.vec3(0.0, 0.0, 0.0), q=wp.quat_identity()))
        camera = camera or TestCameraSensor._make_pinhole_camera(width, height)
        builder.add_shape_camera(body=camera_body, camera=camera, label=camera_label)

        return builder.finalize(device="cpu"), camera

    @staticmethod
    def _build_sphere_world(
        camera: CameraSensor,
        *,
        camera_label: str = "camera",
    ) -> newton.ModelBuilder:
        builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
        sphere_body = builder.add_body(xform=wp.transform(p=wp.vec3(0.0, 0.0, -2.0), q=wp.quat_identity()))
        builder.add_shape_sphere(sphere_body, radius=0.75, color=(0.25, 0.5, 0.75))

        camera_body = builder.add_body(xform=wp.transform(p=wp.vec3(0.0, 0.0, 0.0), q=wp.quat_identity()))
        builder.add_shape_camera(body=camera_body, camera=camera, label=camera_label)

        return builder

    def test_camera_sensor_public_imports_resolve_to_same_class(self) -> None:
        self.assertIs(newton.CameraSensor, CameraSensor)
        self.assertIs(geometry.CameraSensor, CameraSensor)
        self.assertFalse(hasattr(CameraSensor, "create_pinhole"))
        self.assertFalse(hasattr(CameraSensor, "get_render_utils"))

    def test_constructor_derives_dimensions_from_rays(self) -> None:
        width, height = 5, 4
        rays = CameraSensor.compute_camera_rays_pinhole(width, height, math.radians(45.0), device="cpu")

        camera = CameraSensor(rays)

        self.assertEqual(camera.width, width)
        self.assertEqual(camera.height, height)
        self.assertEqual(camera.rays.shape, (height, width, 2))
        with self.assertRaises(AttributeError):
            camera.width = 1
        with self.assertRaises(AttributeError):
            camera.height = 1

        with self.assertRaisesRegex(ValueError, "CameraSensor rays must have shape"):
            CameraSensor(rays.reshape((1, height, width, 2)))

    def test_camera_ray_helpers_live_on_camera_sensor(self) -> None:
        sensor_helper_names = (
            "compute_camera_rays_pinhole",
            "compute_camera_rays_usd_pinhole",
            "compute_camera_rays_fisheye_opencv",
            "compute_camera_rays_fisheye_ftheta",
            "compute_camera_rays_fisheye_kannala_brandt",
        )
        for helper_name in sensor_helper_names:
            self.assertTrue(hasattr(CameraSensor, helper_name))
            self.assertFalse(hasattr(Utils, helper_name))
        self.assertFalse(hasattr(Utils, "compute_pinhole_camera_rays"))
        self.assertFalse(hasattr(Utils, "compute_camera_transforms_usd"))
        self.assertFalse(hasattr(Utils, "assign_checkerboard_material_to_all_shapes"))

        width, height = 3, 3
        rays = [
            CameraSensor.compute_camera_rays_pinhole(width, height, math.radians(45.0), device="cpu"),
            CameraSensor.compute_camera_rays_pinhole(
                width,
                height,
                focal_length=1.0,
                horizontal_aperture=2.0,
                vertical_aperture=2.0,
                device="cpu",
            ),
            CameraSensor.compute_camera_rays_fisheye_opencv(
                width, height, fx=1.0, fy=1.0, cx=1.5, cy=1.5, device="cpu"
            ),
            CameraSensor.compute_camera_rays_fisheye_ftheta(
                width, height, optical_center_x=1.5, optical_center_y=1.5, device="cpu"
            ),
            CameraSensor.compute_camera_rays_fisheye_kannala_brandt(
                width, height, optical_center_x=1.5, optical_center_y=1.5, device="cpu"
            ),
        ]

        for ray_bundle in rays:
            self.assertEqual(ray_bundle.shape, (height, width, 2))
            self.assertEqual(ray_bundle.dtype, wp.vec3f)

    def test_camera_ray_helpers_support_preallocated_output(self) -> None:
        width, height = 4, 3
        out_rays = wp.zeros((height, width, 2), dtype=wp.vec3f, device="cpu")

        rays = CameraSensor.compute_camera_rays_pinhole(
            width, height, math.radians(45.0), out_rays=out_rays, device="cpu"
        )

        self.assertIs(rays, out_rays)
        self.assertFalse(np.allclose(rays.numpy(), 0.0))

    def test_camera_ray_helpers_reject_batched_inputs(self) -> None:
        width, height = 4, 3

        with self.assertRaises(TypeError):
            CameraSensor.compute_camera_rays_pinhole(width, height, [math.radians(45.0)], device="cpu")

        with self.assertRaises(TypeError):
            CameraSensor.compute_camera_rays_pinhole(
                width,
                height,
                focal_length=wp.array([1.0], dtype=wp.float32, device="cpu"),
                horizontal_aperture=2.0,
                vertical_aperture=2.0,
                device="cpu",
            )

        out_rays = wp.zeros((1, height, width, 2), dtype=wp.vec3f, device="cpu")
        with self.assertRaisesRegex(ValueError, "out_rays must have shape"):
            CameraSensor.compute_camera_rays_pinhole(width, height, math.radians(45.0), out_rays=out_rays)

    def test_utils_property_uses_finalized_model_render_context(self) -> None:
        width, height = 4, 3
        camera = self._make_pinhole_camera(width, height)
        with self.assertRaisesRegex(RuntimeError, "finalized into a model"):
            _ = camera.utils

        model, camera = self._build_sphere_camera_scene(width, height, camera=camera)

        utils = camera.utils

        self.assertIsInstance(utils, Utils)
        self.assertIsNot(camera.utils, utils)
        self.assertIsNotNone(model.render_context)
        self.assertEqual(utils.create_depth_image_output(width, height).shape, (model.world_count, height, width))
        utils.assign_checkerboard_material(shape_indices=[0])
        self.assertEqual(len(model.render_context._texture_data_source), 1)

    def test_update_renders_from_shape_transform(self) -> None:
        width, height = 16, 12
        model, camera = self._build_sphere_camera_scene(width, height)
        state = model.state()

        depth = wp.zeros((model.world_count, height, width), dtype=wp.float32, device="cpu")
        shape_index = wp.zeros((model.world_count, height, width), dtype=wp.uint32, device="cpu")

        model.update_render_context(state)
        camera.update(model, state, depth_image=depth, shape_index_image=shape_index)

        depth_np = depth.numpy()
        shape_index_np = shape_index.numpy()
        self.assertGreater(float(depth_np[0, height // 2, width // 2]), 0.0)
        self.assertTrue(np.any(shape_index_np != 0xFFFFFFFF))
        self.assertIsNotNone(model.render_context)

    def test_update_respects_world_enabled_mask(self) -> None:
        width, height = 16, 12
        camera = self._make_pinhole_camera(width, height)

        scene = newton.ModelBuilder(up_axis=newton.Axis.Z)
        scene.add_world(self._build_sphere_world(camera, camera_label="camera_0"))
        scene.add_world(self._build_sphere_world(camera, camera_label="camera_1"))
        model = scene.finalize(device="cpu")
        state = model.state()

        camera.clear_data = CameraSensor.ClearData(clear_depth=-2.0, clear_shape_index=123)
        world_enabled = wp.array([True, False], dtype=wp.bool, device="cpu")
        depth = wp.zeros((model.world_count, height, width), dtype=wp.float32, device="cpu")
        shape_index = wp.zeros((model.world_count, height, width), dtype=wp.uint32, device="cpu")

        model.update_render_context(state)
        camera.update(model, state, depth_image=depth, shape_index_image=shape_index, world_enabled=world_enabled)

        depth_np = depth.numpy()
        shape_index_np = shape_index.numpy()
        self.assertGreater(float(depth_np[0, height // 2, width // 2]), 0.0)
        self.assertEqual(float(depth_np[1, height // 2, width // 2]), -2.0)
        self.assertEqual(int(shape_index_np[1, height // 2, width // 2]), 123)

    def test_update_reuses_default_world_enabled_mask(self) -> None:
        width, height = 16, 12
        model, camera = self._build_sphere_camera_scene(width, height)
        state = model.state()
        depth = wp.zeros((model.world_count, height, width), dtype=wp.float32, device="cpu")

        model.update_render_context(state)
        camera.update(model, state, depth_image=depth)
        default_world_enabled = camera._all_world_enabled

        self.assertIsNotNone(default_world_enabled)
        camera.update(model, state, depth_image=depth)
        self.assertIs(camera._all_world_enabled, default_world_enabled)

    def test_update_uses_instance_render_settings(self) -> None:
        parameters = inspect.signature(CameraSensor.update).parameters
        self.assertNotIn("clear_data", parameters)
        self.assertNotIn("render_config", parameters)
        self.assertNotIn("load_textures", parameters)

        width, height = 16, 12
        model, camera = self._build_sphere_camera_scene(width, height)
        state = model.state()

        camera.clear_data = CameraSensor.ClearData(clear_depth=-2.0, clear_shape_index=123)
        camera.render_config = CameraSensor.RenderConfig(max_distance=0.1)
        self.assertFalse(hasattr(camera, "load_textures"))

        depth = wp.zeros((model.world_count, height, width), dtype=wp.float32, device="cpu")
        shape_index = wp.zeros((model.world_count, height, width), dtype=wp.uint32, device="cpu")

        model.init_render_context(load_textures=False)
        model.update_render_context(state)
        camera.update(model, state, depth_image=depth, shape_index_image=shape_index)

        self.assertEqual(float(depth.numpy()[0, height // 2, width // 2]), -2.0)
        self.assertEqual(int(shape_index.numpy()[0, height // 2, width // 2]), 123)

    def test_update_supports_all_render_orders_with_3d_outputs(self) -> None:
        width, height = 16, 12

        for render_order in CameraSensor.RenderOrder:
            with self.subTest(render_order=render_order):
                model, camera = self._build_sphere_camera_scene(width, height)
                state = model.state()
                camera.render_config = CameraSensor.RenderConfig(render_order=render_order)

                depth = wp.zeros((model.world_count, height, width), dtype=wp.float32, device="cpu")

                model.update_render_context(state)
                camera.update(model, state, depth_image=depth)

                self.assertGreater(float(depth.numpy()[0, height // 2, width // 2]), 0.0)

    def test_multiple_camera_sensors_share_model_render_context(self) -> None:
        width, height = 8, 6
        camera_a = self._make_pinhole_camera(width, height)
        camera_b = self._make_pinhole_camera(width, height)

        builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
        sphere_body = builder.add_body(xform=wp.transform(p=wp.vec3(0.0, 0.0, -2.0), q=wp.quat_identity()))
        builder.add_shape_sphere(sphere_body, radius=0.75, color=(0.25, 0.5, 0.75))
        builder.add_shape_camera(camera=camera_a, label="camera_a")
        builder.add_shape_camera(camera=camera_b, label="camera_b")
        model = builder.finalize(device="cpu")
        state = model.state()

        depth_a = wp.zeros((model.world_count, height, width), dtype=wp.float32, device="cpu")
        depth_b = wp.zeros((model.world_count, height, width), dtype=wp.float32, device="cpu")

        model.update_render_context(state)
        camera_a.update(model, state, depth_image=depth_a)
        render_context = model.render_context

        with mock.patch.object(render_context, "update", wraps=render_context.update) as update_mock:
            camera_b.update(model, state, depth_image=depth_b)
            update_mock.assert_not_called()

        self.assertIs(model.render_context, render_context)
        self.assertGreater(float(depth_a.numpy()[0, height // 2, width // 2]), 0.0)
        self.assertGreater(float(depth_b.numpy()[0, height // 2, width // 2]), 0.0)


if __name__ == "__main__":
    unittest.main()
