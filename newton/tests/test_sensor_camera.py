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
from newton._src.sensors.sensor_camera_renderer.utils import Utils
from newton.sensors import SensorCamera


class TestSensorCamera(unittest.TestCase):
    @staticmethod
    def _make_pinhole_camera(width: int, height: int, fov: float = math.radians(45.0)) -> SensorCamera:
        rays = SensorCamera.compute_camera_rays_pinhole(width, height, fov, device="cpu")
        return SensorCamera(rays)

    @staticmethod
    def _build_sphere_camera_scene(
        width: int,
        height: int,
        *,
        camera: SensorCamera | None = None,
        camera_label: str = "camera",
    ) -> tuple[newton.Model, SensorCamera]:
        builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
        sphere_body = builder.add_body(xform=wp.transform(p=wp.vec3(0.0, 0.0, -2.0), q=wp.quat_identity()))
        builder.add_shape_sphere(sphere_body, radius=0.75, color=(0.25, 0.5, 0.75))

        camera_body = builder.add_body(xform=wp.transform(p=wp.vec3(0.0, 0.0, 0.0), q=wp.quat_identity()))
        camera = camera or TestSensorCamera._make_pinhole_camera(width, height)
        builder.add_shape_camera(body=camera_body, camera=camera, label=camera_label)

        return builder.finalize(device="cpu"), camera

    @staticmethod
    def _build_sphere_world(
        camera: SensorCamera,
        *,
        camera_label: str = "camera",
    ) -> newton.ModelBuilder:
        builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
        sphere_body = builder.add_body(xform=wp.transform(p=wp.vec3(0.0, 0.0, -2.0), q=wp.quat_identity()))
        builder.add_shape_sphere(sphere_body, radius=0.75, color=(0.25, 0.5, 0.75))

        camera_body = builder.add_body(xform=wp.transform(p=wp.vec3(0.0, 0.0, 0.0), q=wp.quat_identity()))
        builder.add_shape_camera(body=camera_body, camera=camera, label=camera_label)

        return builder

    def test_sensor_camera_public_imports_resolve_to_same_class(self) -> None:
        """Verify public SensorCamera imports and removed helpers."""
        self.assertIs(newton.SensorCamera, SensorCamera)
        self.assertIs(geometry.SensorCamera, SensorCamera)
        self.assertFalse(hasattr(SensorCamera, "create_pinhole"))
        self.assertFalse(hasattr(SensorCamera, "get_render_utils"))
        self.assertFalse(hasattr(SensorCamera, "_bind_model"))
        self.assertFalse(hasattr(SensorCamera, "_ensure_shape_index_by_world"))
        self.assertFalse(hasattr(SensorCamera, "_ensure_render_buffers"))
        self.assertEqual(int(SensorCamera.WorldRenderFlag.ENABLE), 1)
        self.assertEqual(int(SensorCamera.WorldRenderFlag.DISABLE_CLEAR), 2)
        self.assertEqual(int(SensorCamera.WorldRenderFlag.DISABLE_PRESERVE), 0)

    def test_constructor_derives_dimensions_from_rays(self) -> None:
        """Verify SensorCamera derives read-only image dimensions from rays."""
        width, height = 5, 4
        rays = SensorCamera.compute_camera_rays_pinhole(width, height, math.radians(45.0), device="cpu")

        camera = SensorCamera(rays)

        self.assertEqual(camera.width, width)
        self.assertEqual(camera.height, height)
        self.assertEqual(camera.rays.shape, (height, width, 2))
        with self.assertRaises(AttributeError):
            camera.width = 1
        with self.assertRaises(AttributeError):
            camera.height = 1

        with self.assertRaisesRegex(ValueError, "SensorCamera rays must have shape"):
            SensorCamera(rays.reshape((1, height, width, 2)))

    def test_camera_ray_helpers_live_on_sensor_camera(self) -> None:
        """Verify camera ray helpers live on SensorCamera."""
        sensor_helper_names = (
            "compute_camera_rays_pinhole",
            "compute_camera_rays_usd_pinhole",
            "compute_camera_rays_fisheye_opencv",
            "compute_camera_rays_fisheye_ftheta",
            "compute_camera_rays_fisheye_kannala_brandt",
        )
        for helper_name in sensor_helper_names:
            self.assertTrue(hasattr(SensorCamera, helper_name))
            self.assertFalse(hasattr(Utils, helper_name))
        self.assertFalse(hasattr(Utils, "compute_pinhole_camera_rays"))
        self.assertFalse(hasattr(Utils, "compute_camera_transforms_usd"))
        self.assertFalse(hasattr(Utils, "create_default_light"))
        self.assertFalse(hasattr(Utils, "assign_checkerboard_material"))
        self.assertFalse(hasattr(Utils, "assign_checkerboard_material_to_all_shapes"))
        for helper_name in (
            "_create_image_output",
            "create_color_image_output",
            "create_depth_image_output",
            "create_forward_depth_image_output",
            "create_shape_index_image_output",
            "create_normal_image_output",
            "create_albedo_image_output",
            "create_hdr_color_image_output",
        ):
            self.assertFalse(hasattr(Utils, helper_name))

        width, height = 3, 3
        rays = [
            SensorCamera.compute_camera_rays_pinhole(width, height, math.radians(45.0), device="cpu"),
            SensorCamera.compute_camera_rays_pinhole(
                width,
                height,
                focal_length=1.0,
                horizontal_aperture=2.0,
                vertical_aperture=2.0,
                device="cpu",
            ),
            SensorCamera.compute_camera_rays_fisheye_opencv(
                width, height, fx=1.0, fy=1.0, cx=1.5, cy=1.5, device="cpu"
            ),
            SensorCamera.compute_camera_rays_fisheye_ftheta(
                width, height, optical_center_x=1.5, optical_center_y=1.5, device="cpu"
            ),
            SensorCamera.compute_camera_rays_fisheye_kannala_brandt(
                width, height, optical_center_x=1.5, optical_center_y=1.5, device="cpu"
            ),
        ]

        for ray_bundle in rays:
            self.assertEqual(ray_bundle.shape, (height, width, 2))
            self.assertEqual(ray_bundle.dtype, wp.vec3f)

    def test_camera_ray_helpers_support_preallocated_output(self) -> None:
        """Verify camera ray helpers can write into caller output arrays."""
        width, height = 4, 3
        out_rays = wp.zeros((height, width, 2), dtype=wp.vec3f, device="cpu")

        rays = SensorCamera.compute_camera_rays_pinhole(
            width, height, math.radians(45.0), out_rays=out_rays, device="cpu"
        )

        self.assertIs(rays, out_rays)
        self.assertFalse(np.allclose(rays.numpy(), 0.0))

    def test_camera_ray_helpers_reject_batched_inputs(self) -> None:
        """Verify camera ray helpers accept only single-camera parameters."""
        width, height = 4, 3

        with self.assertRaisesRegex(ValueError, "camera_fov cannot be provided with aperture parameters"):
            SensorCamera.compute_camera_rays_pinhole(
                width,
                height,
                math.radians(45.0),
                focal_length=1.0,
                horizontal_aperture=2.0,
                vertical_aperture=2.0,
                device="cpu",
            )

        with self.assertRaises(TypeError):
            SensorCamera.compute_camera_rays_pinhole(width, height, [math.radians(45.0)], device="cpu")

        with self.assertRaises(TypeError):
            SensorCamera.compute_camera_rays_pinhole(
                width,
                height,
                focal_length=wp.array([1.0], dtype=wp.float32, device="cpu"),
                horizontal_aperture=2.0,
                vertical_aperture=2.0,
                device="cpu",
            )

        out_rays = wp.zeros((1, height, width, 2), dtype=wp.vec3f, device="cpu")
        with self.assertRaisesRegex(ValueError, "out_rays must have shape"):
            SensorCamera.compute_camera_rays_pinhole(width, height, math.radians(45.0), out_rays=out_rays)

    def test_utils_property_uses_finalized_sensor_camera(self) -> None:
        """Verify finalized SensorCamera instances own render utility state."""
        width, height = 4, 3
        camera = self._make_pinhole_camera(width, height)
        with self.assertRaisesRegex(RuntimeError, "finalized into a model"):
            _ = camera.utils
        with self.assertRaisesRegex(RuntimeError, "finalized into a model"):
            camera.create_image_output(wp.float32)

        model, camera = self._build_sphere_camera_scene(width, height, camera=camera)

        utils = camera.utils

        self.assertIsInstance(utils, Utils)
        self.assertIsNot(camera.utils, utils)
        self.assertFalse(hasattr(utils, "_Utils__sensor_camera"))
        self.assertIsNone(model.render_context)
        self.assertEqual(camera.device, model.device)
        self.assertFalse(hasattr(camera, "_model_ref"))
        self.assertIsNotNone(camera._shape_index_by_world)
        self.assertEqual(camera._shape_index_by_world.shape, (camera.view_count,))
        self.assertIsNotNone(camera._camera_transforms)
        self.assertEqual(camera._camera_transforms.shape, (camera.view_count,))
        self.assertIsNotNone(camera._all_world_render_flags)
        self.assertEqual(camera._all_world_render_flags.shape, (camera.view_count,))
        np.testing.assert_array_equal(
            camera._all_world_render_flags.numpy(),
            np.full(camera.view_count, int(SensorCamera.WorldRenderFlag.ENABLE), dtype=np.int32),
        )
        output_specs = (
            (camera.create_image_output(wp.float32), wp.float32),
            (camera.create_color_image_output(), wp.uint32),
            (camera.create_depth_image_output(), wp.float32),
            (camera.create_forward_depth_image_output(), wp.float32),
            (camera.create_shape_index_image_output(), wp.uint32),
            (camera.create_normal_image_output(), wp.vec3f),
            (camera.create_albedo_image_output(), wp.uint32),
            (camera.create_hdr_color_image_output(), wp.vec3f),
        )
        for output, dtype in output_specs:
            with self.subTest(dtype=dtype):
                self.assertEqual(output.shape, (camera.view_count, height, width))
                self.assertEqual(output.dtype, dtype)
                self.assertEqual(output.device, model.device)
        color_rgba = utils.to_rgba_from_color(camera.create_color_image_output())
        self.assertEqual(color_rgba.shape, (camera.view_count, height, width, 4))
        render_context = model.init_render_context()
        render_context.create_default_light(enable_shadows=True)
        self.assertEqual(render_context.light_count, 1)
        render_context.assign_checkerboard_material(shape_indices=[0])
        self.assertEqual(len(model.render_context._texture_data_source), 1)

    def test_shape_index_by_world_rejects_negative_indices(self) -> None:
        """Verify SensorCamera rejects negative shape indices."""
        with self.assertRaisesRegex(ValueError, "shape_indices must be non-negative"):
            SensorCamera._compute_shape_index_by_world(
                shape_indices=np.array([-1], dtype=np.int32),
                shape_world=np.array([0, 1], dtype=np.int32),
                world_count=2,
            )

        with self.assertRaises(IndexError):
            SensorCamera._compute_shape_index_by_world(
                shape_indices=np.array([2], dtype=np.int32),
                shape_world=np.array([0, 1], dtype=np.int32),
                world_count=2,
            )

    @unittest.skipUnless(wp.is_cuda_available(), "Requires CUDA")
    def test_finalize_uses_rays_device_when_device_is_none(self) -> None:
        """Verify finalize allocates buffers on the rays device by default."""
        camera = self._make_pinhole_camera(2, 2)
        expected_device = camera.rays.device

        with wp.ScopedDevice("cuda:0"):
            camera.finalize(
                shape_indices=[0],
                shape_world=np.array([0], dtype=np.int32),
                world_count=1,
            )

        self.assertEqual(camera.rays.device, expected_device)
        self.assertEqual(camera.shape_indices.device, expected_device)
        self.assertEqual(camera._shape_index_by_world.device, expected_device)
        self.assertEqual(camera._camera_transforms.device, expected_device)
        self.assertEqual(camera._all_world_render_flags.device, expected_device)

    def test_update_renders_from_shape_transform(self) -> None:
        """Verify SensorCamera renders from model shape transforms."""
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

    def test_update_respects_disable_clear_flag(self) -> None:
        """Verify SensorCamera clears output images for DISABLE_CLEAR worlds."""
        width, height = 16, 12
        camera = self._make_pinhole_camera(width, height)

        scene = newton.ModelBuilder(up_axis=newton.Axis.Z)
        scene.add_world(self._build_sphere_world(camera, camera_label="camera_0"))
        scene.add_world(self._build_sphere_world(camera, camera_label="camera_1"))
        model = scene.finalize(device="cpu")
        state = model.state()

        camera.clear_data = SensorCamera.ClearData(clear_depth=-2.0, clear_shape_index=123)
        world_render_flags = wp.array(
            [int(SensorCamera.WorldRenderFlag.ENABLE), int(SensorCamera.WorldRenderFlag.DISABLE_CLEAR)],
            dtype=wp.int32,
            device="cpu",
        )
        depth = wp.zeros((model.world_count, height, width), dtype=wp.float32, device="cpu")
        shape_index = wp.zeros((model.world_count, height, width), dtype=wp.uint32, device="cpu")

        model.update_render_context(state)
        camera.update(
            model,
            state,
            depth_image=depth,
            shape_index_image=shape_index,
            world_render_flags=world_render_flags,
        )

        depth_np = depth.numpy()
        shape_index_np = shape_index.numpy()
        self.assertGreater(float(depth_np[0, height // 2, width // 2]), 0.0)
        self.assertEqual(float(depth_np[1, height // 2, width // 2]), -2.0)
        self.assertEqual(int(shape_index_np[1, height // 2, width // 2]), 123)

    def test_update_respects_disable_preserve_flag(self) -> None:
        """Verify SensorCamera preserves output images for DISABLE_PRESERVE worlds."""
        width, height = 16, 12
        camera = self._make_pinhole_camera(width, height)

        scene = newton.ModelBuilder(up_axis=newton.Axis.Z)
        scene.add_world(self._build_sphere_world(camera, camera_label="camera_0"))
        scene.add_world(self._build_sphere_world(camera, camera_label="camera_1"))
        model = scene.finalize(device="cpu")
        state = model.state()

        world_render_flags = wp.array(
            [int(SensorCamera.WorldRenderFlag.ENABLE), int(SensorCamera.WorldRenderFlag.DISABLE_PRESERVE)],
            dtype=wp.int32,
            device="cpu",
        )
        depth = wp.full((model.world_count, height, width), value=42.0, dtype=wp.float32, device="cpu")
        shape_index = wp.full((model.world_count, height, width), value=456, dtype=wp.uint32, device="cpu")

        model.update_render_context(state)
        camera.update(
            model,
            state,
            depth_image=depth,
            shape_index_image=shape_index,
            world_render_flags=world_render_flags,
        )

        depth_np = depth.numpy()
        shape_index_np = shape_index.numpy()
        self.assertGreater(float(depth_np[0, height // 2, width // 2]), 0.0)
        np.testing.assert_allclose(depth_np[1], 42.0)
        np.testing.assert_array_equal(shape_index_np[1], np.full((height, width), 456, dtype=np.uint32))

    def test_update_reuses_default_world_render_flags(self) -> None:
        """Verify SensorCamera reuses its finalized default world flags."""
        width, height = 16, 12
        model, camera = self._build_sphere_camera_scene(width, height)
        state = model.state()
        depth = wp.zeros((model.world_count, height, width), dtype=wp.float32, device="cpu")
        default_world_render_flags = camera._all_world_render_flags

        self.assertIsNotNone(default_world_render_flags)
        model.update_render_context(state)
        camera.update(model, state, depth_image=depth)
        camera.update(model, state, depth_image=depth)
        self.assertIs(camera._all_world_render_flags, default_world_render_flags)

    def test_update_uses_instance_render_settings(self) -> None:
        """Verify SensorCamera update uses instance render settings."""
        parameters = inspect.signature(SensorCamera.update).parameters
        self.assertNotIn("clear_data", parameters)
        self.assertNotIn("render_config", parameters)
        self.assertNotIn("load_textures", parameters)
        self.assertNotIn("world_enabled", parameters)
        self.assertIn("world_render_flags", parameters)

        width, height = 16, 12
        model, camera = self._build_sphere_camera_scene(width, height)
        state = model.state()

        camera.clear_data = SensorCamera.ClearData(clear_depth=-2.0, clear_shape_index=123)
        camera.render_config = SensorCamera.RenderConfig(max_distance=0.1)
        self.assertFalse(hasattr(camera, "load_textures"))

        depth = wp.zeros((model.world_count, height, width), dtype=wp.float32, device="cpu")
        shape_index = wp.zeros((model.world_count, height, width), dtype=wp.uint32, device="cpu")

        model.init_render_context(load_textures=False)
        model.update_render_context(state)
        camera.update(model, state, depth_image=depth, shape_index_image=shape_index)

        self.assertEqual(float(depth.numpy()[0, height // 2, width // 2]), -2.0)
        self.assertEqual(int(shape_index.numpy()[0, height // 2, width // 2]), 123)

    def test_update_supports_all_render_orders_with_3d_outputs(self) -> None:
        """Verify SensorCamera renders every render order into 3-D outputs."""
        width, height = 16, 12

        for render_order in SensorCamera.RenderOrder:
            with self.subTest(render_order=render_order):
                model, camera = self._build_sphere_camera_scene(width, height)
                state = model.state()
                camera.render_config = SensorCamera.RenderConfig(render_order=render_order)

                depth = wp.zeros((model.world_count, height, width), dtype=wp.float32, device="cpu")

                model.update_render_context(state)
                camera.update(model, state, depth_image=depth)

                self.assertGreater(float(depth.numpy()[0, height // 2, width // 2]), 0.0)

    def test_multiple_sensor_cameras_share_model_render_context(self) -> None:
        """Verify multiple SensorCamera instances share model render context."""
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
