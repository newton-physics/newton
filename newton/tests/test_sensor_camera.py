# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import inspect
import math
import unittest

import numpy as np
import warp as wp

import newton
import newton._src.render as internal_render
import newton.geometry as geometry
from newton._src.render.utils import Utils
from newton.sensors import (
    SensorCamera,
)

# Transform placing a camera at the origin looking down -Z (identity pose). A
# camera with this transform sees a sphere placed at z = -2.
_IDENTITY_XFORM = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float32)


class TestSensorCamera(unittest.TestCase):
    @staticmethod
    def _make_pinhole_camera(width: int, height: int, fov: float = math.radians(45.0)) -> SensorCamera:
        rays = SensorCamera.compute_camera_rays_pinhole(width, height, fov, device="cpu")
        return SensorCamera(rays)

    @staticmethod
    def _sphere_world_builder() -> newton.ModelBuilder:
        """A single-world scene with a sphere in front of an identity camera."""
        builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
        sphere_body = builder.add_body(xform=wp.transform(p=wp.vec3(0.0, 0.0, -2.0), q=wp.quat_identity()))
        builder.add_shape_sphere(sphere_body, radius=0.75, color=(0.25, 0.5, 0.75))
        return builder

    @classmethod
    def _build_sphere_scene(
        cls,
        width: int,
        height: int,
        *,
        world_count: int = 1,
        assign_render_context: bool = True,
        view_count: int | None = None,
    ) -> tuple[newton.Model, SensorCamera]:
        if world_count == 1:
            builder = cls._sphere_world_builder()
        else:
            builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
            for _ in range(world_count):
                builder.add_world(cls._sphere_world_builder())

        model = builder.finalize(device="cpu")
        rays = SensorCamera.compute_camera_rays_pinhole(width, height, math.radians(45.0), device="cpu")
        render_context = newton.RenderContext(model) if assign_render_context else None
        camera = SensorCamera(rays, render_context, view_count=view_count)
        return model, camera

    @staticmethod
    def _identity_transforms(view_count: int, device: str = "cpu") -> wp.array:
        """World-space identity camera poses, shape ``(view_count,)``."""
        return wp.array(np.tile(_IDENTITY_XFORM, (view_count, 1)), dtype=wp.transformf, device=device)

    @staticmethod
    def _camera_with_context(
        width: int, height: int, model: newton.Model, *, fov: float = math.radians(45.0), **context_kwargs
    ) -> SensorCamera:
        """A SensorCamera constructed with a fresh render context for ``model``."""
        rays = SensorCamera.compute_camera_rays_pinhole(width, height, fov, device="cpu")
        return SensorCamera(rays, newton.RenderContext(model, **context_kwargs))

    def test_sensor_camera_public_imports_resolve_to_same_class(self) -> None:
        """Verify public SensorCamera imports and removed site-attachment helpers."""
        # SensorCamera lives in ``newton.sensors`` like every other sensor, not at
        # the top level or in ``newton.geometry``.
        self.assertIs(newton.sensors.SensorCamera, SensorCamera)
        self.assertFalse(hasattr(newton, "SensorCamera"))
        # The camera model spec classes were removed along with USD/MJCF camera import.
        for spec_name in (
            "CameraSpec",
            "CameraPinholeSpec",
            "CameraFisheyeOpenCVSpec",
            "CameraFisheyeFThetaSpec",
            "CameraFisheyeKannalaBrandtSpec",
        ):
            self.assertFalse(hasattr(newton, spec_name), spec_name)
            self.assertFalse(hasattr(newton.sensors, spec_name), spec_name)
        self.assertIs(newton.RenderContext, newton.render.RenderContext)
        self.assertNotIn("RenderContext", internal_render.__all__)
        self.assertFalse(hasattr(internal_render, "RenderContext"))
        self.assertIs(newton.ClearData, newton.render.ClearData)
        self.assertIs(newton.GaussianRenderMode, newton.render.GaussianRenderMode)
        self.assertIs(newton.LightType, newton.render.LightType)
        self.assertIs(newton.RenderConfig, newton.render.RenderConfig)
        self.assertIs(newton.RenderOrder, newton.render.RenderOrder)
        self.assertIs(newton.WorldRenderFlag, newton.render.WorldRenderFlag)
        self.assertIs(newton.TextureProjectionMode, newton.render.TextureProjectionMode)
        self.assertFalse(hasattr(geometry, "SensorCamera"))

        # The caller owns the transforms; the camera is no longer attached to sites.
        camera = self._make_pinhole_camera(2, 2)
        self.assertFalse(hasattr(camera, "camera_transforms"))
        for attr in ("shape_indices", "_shape_index_by_world", "_camera_transforms"):
            self.assertFalse(hasattr(camera, attr), attr)
        for attr in (
            "_compute_shape_index_by_world",
            "_update_transforms",
            "_ensure_shape_index_by_world",
            "_ensure_render_buffers",
        ):
            self.assertFalse(hasattr(SensorCamera, attr), attr)
        self.assertFalse(hasattr(internal_render, "_compute_camera_transforms"))

        self.assertFalse(hasattr(newton.ModelBuilder, "add_shape_camera"))
        self.assertFalse(hasattr(newton.ModelBuilder, "set_site_camera"))
        self.assertNotIn("camera", inspect.signature(newton.ModelBuilder.add_site).parameters)

        # Negative disable sentinels for the per-view world_indices array; no ENABLE.
        self.assertFalse(hasattr(newton.WorldRenderFlag, "ENABLE"))
        self.assertEqual(int(newton.WorldRenderFlag.DISABLE_PRESERVE), -1)
        self.assertEqual(int(newton.WorldRenderFlag.DISABLE_CLEAR), -2)

    def test_constructor_derives_dimensions_from_rays(self) -> None:
        """Verify SensorCamera derives read-only image dimensions from rays."""
        width, height = 5, 4
        rays = SensorCamera.compute_camera_rays_pinhole(width, height, math.radians(45.0), device="cpu")

        camera = SensorCamera(rays)

        self.assertEqual(camera.width, width)
        self.assertEqual(camera.height, height)
        self.assertIsNone(camera.render_context)
        self.assertFalse(hasattr(camera, "camera_transforms"))
        self.assertEqual(camera.view_count, 0)
        self.assertEqual(camera.rays.shape, (height, width, 2))
        with self.assertRaises(AttributeError):
            camera.width = 1
        with self.assertRaises(AttributeError):
            camera.height = 1
        # render_context is read-only: it can only be set in the constructor.
        with self.assertRaises(AttributeError):
            camera.render_context = None

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

    def test_render_context_sets_view_count_and_world_indices(self) -> None:
        """Verify a render context sizes view_count and world indices to the world count."""
        width, height = 4, 3
        # A camera without a render context or view_count has no views yet.
        camera = self._make_pinhole_camera(width, height)
        with self.assertRaisesRegex(RuntimeError, "render context"):
            _ = camera.utils
        with self.assertRaisesRegex(RuntimeError, "render context"):
            camera.create_image_output(wp.float32)

        model, camera = self._build_sphere_scene(width, height, world_count=2)

        self.assertEqual(camera.view_count, model.world_count)
        # The camera no longer owns the camera transforms; the caller passes them.
        self.assertFalse(hasattr(camera, "camera_transforms"))
        self.assertIsNotNone(camera.world_indices)
        self.assertEqual(camera.world_indices.shape, (camera.view_count,))
        # Default identity mapping: view i renders world i.
        np.testing.assert_array_equal(
            camera.world_indices.numpy(),
            np.arange(camera.view_count, dtype=np.int32),
        )
        self.assertEqual(camera.device, model.device)
        self.assertFalse(hasattr(model, "render_context"))

    def test_utils_property_uses_render_context(self) -> None:
        """Verify SensorCamera instances own render utility state after a render context is assigned."""
        width, height = 4, 3
        model, camera = self._build_sphere_scene(width, height)

        utils = camera.utils

        self.assertIsInstance(utils, Utils)
        self.assertIsNot(camera.utils, utils)
        self.assertFalse(hasattr(utils, "_Utils__sensor_camera"))
        self.assertFalse(hasattr(model, "render_context"))
        self.assertIs(camera.render_context.model, model)
        self.assertEqual(camera.device, model.device)
        self.assertFalse(hasattr(camera, "_model_ref"))

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
        render_context = camera.render_context
        self.assertIsNotNone(render_context)
        self.assertEqual(render_context.world_count, model.world_count)
        self.assertEqual(render_context.device, model.device)
        render_context.create_default_light(enable_shadows=True)
        self.assertEqual(render_context.light_count, 1)
        render_context.assign_checkerboard_material(shape_indices=[0])

    @unittest.skipUnless(wp.is_cuda_available(), "Requires CUDA")
    def test_render_context_moves_buffers_to_model_device(self) -> None:
        """Verify constructing with a render context clones rays and allocates buffers on the model device."""
        rays = SensorCamera.compute_camera_rays_pinhole(2, 2, math.radians(45.0), device="cpu")
        self.assertEqual(rays.device, wp.get_device("cpu"))

        model = self._sphere_world_builder().finalize(device="cuda:0")
        camera = SensorCamera(rays, newton.RenderContext(model))

        self.assertEqual(camera.rays.device, model.device)
        self.assertEqual(camera.world_indices.device, model.device)

    def test_update_renders_from_camera_transforms(self) -> None:
        """Verify SensorCamera renders from the camera transforms passed to update."""
        width, height = 16, 12
        model, camera = self._build_sphere_scene(width, height)
        state = model.state()

        depth = wp.zeros((model.world_count, height, width), dtype=wp.float32, device="cpu")
        shape_index = wp.zeros((model.world_count, height, width), dtype=wp.uint32, device="cpu")

        # Identity transforms see the sphere placed in front of the camera.
        camera.render_context.update(state)
        camera_transforms = self._identity_transforms(camera.view_count)
        camera.update(state, camera_transforms, depth_image=depth, shape_index_image=shape_index)

        center = (0, height // 2, width // 2)
        identity_center_depth = float(depth.numpy()[center])
        self.assertGreater(identity_center_depth, 0.0)
        self.assertTrue(np.any(shape_index.numpy() != 0xFFFFFFFF))
        self.assertFalse(hasattr(model, "render_context"))

        # Move the camera behind the sphere; the center ray no longer hits it.
        behind = np.tile(np.array([0.0, 0.0, -4.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float32), (model.world_count, 1))
        camera_transforms.assign(behind)
        depth.zero_()
        camera.update(state, camera_transforms, depth_image=depth)
        self.assertEqual(float(depth.numpy()[center]), 0.0)

    def test_update_respects_disable_clear_flag(self) -> None:
        """Verify SensorCamera clears output images for DISABLE_CLEAR worlds."""
        width, height = 16, 12
        model, camera = self._build_sphere_scene(width, height, world_count=2)
        state = model.state()

        camera.clear_data = newton.ClearData(clear_depth=-2.0, clear_shape_index=123)
        world_indices = wp.array(
            [0, int(newton.WorldRenderFlag.DISABLE_CLEAR)],
            dtype=wp.int32,
            device="cpu",
        )
        depth = wp.zeros((model.world_count, height, width), dtype=wp.float32, device="cpu")
        shape_index = wp.zeros((model.world_count, height, width), dtype=wp.uint32, device="cpu")

        camera.render_context.update(state)
        camera.update(
            state,
            self._identity_transforms(camera.view_count),
            depth_image=depth,
            shape_index_image=shape_index,
            world_indices=world_indices,
        )

        depth_np = depth.numpy()
        shape_index_np = shape_index.numpy()
        self.assertGreater(float(depth_np[0, height // 2, width // 2]), 0.0)
        self.assertEqual(float(depth_np[1, height // 2, width // 2]), -2.0)
        self.assertEqual(int(shape_index_np[1, height // 2, width // 2]), 123)

    def test_update_requires_render_context(self) -> None:
        """Verify SensorCamera rendering requires an explicit render context."""
        width, height = 8, 6
        model, camera = self._build_sphere_scene(width, height, assign_render_context=False)
        state = model.state()
        depth = wp.zeros((model.world_count, height, width), dtype=wp.float32, device="cpu")

        with self.assertRaisesRegex(RuntimeError, "requires a RenderContext"):
            camera.update(state, self._identity_transforms(model.world_count), depth_image=depth)

    def test_update_respects_disable_preserve_flag(self) -> None:
        """Verify SensorCamera preserves output images for DISABLE_PRESERVE worlds."""
        width, height = 16, 12
        model, camera = self._build_sphere_scene(width, height, world_count=2)
        state = model.state()

        world_indices = wp.array(
            [0, int(newton.WorldRenderFlag.DISABLE_PRESERVE)],
            dtype=wp.int32,
            device="cpu",
        )
        depth = wp.full((model.world_count, height, width), value=42.0, dtype=wp.float32, device="cpu")
        shape_index = wp.full((model.world_count, height, width), value=456, dtype=wp.uint32, device="cpu")

        camera.render_context.update(state)
        camera.update(
            state,
            self._identity_transforms(camera.view_count),
            depth_image=depth,
            shape_index_image=shape_index,
            world_indices=world_indices,
        )

        depth_np = depth.numpy()
        shape_index_np = shape_index.numpy()
        self.assertGreater(float(depth_np[0, height // 2, width // 2]), 0.0)
        np.testing.assert_allclose(depth_np[1], 42.0)
        np.testing.assert_array_equal(shape_index_np[1], np.full((height, width), 456, dtype=np.uint32))

    def test_update_reuses_default_world_indices(self) -> None:
        """Verify SensorCamera reuses its default world indices across updates."""
        width, height = 16, 12
        model, camera = self._build_sphere_scene(width, height)
        state = model.state()
        depth = wp.zeros((model.world_count, height, width), dtype=wp.float32, device="cpu")
        default_world_indices = camera.world_indices

        self.assertIsNotNone(default_world_indices)
        camera.render_context.update(state)
        camera_transforms = self._identity_transforms(camera.view_count)
        camera.update(state, camera_transforms, depth_image=depth)
        camera.update(state, camera_transforms, depth_image=depth)
        self.assertIs(camera.world_indices, default_world_indices)

    def test_world_indices_decouple_views_from_worlds(self) -> None:
        """Verify multiple views can render one shared world from different poses."""
        width, height = 8, 6
        # One world (sphere at z=-2) but three views, all rendering world 0.
        model, camera = self._build_sphere_scene(width, height, view_count=3)
        state = model.state()

        self.assertEqual(camera.view_count, 3)
        self.assertEqual(model.world_count, 1)

        # Three views of world 0 from progressively closer poses.
        transforms = np.tile(_IDENTITY_XFORM, (3, 1))
        transforms[1, 2] = -0.5
        transforms[2, 2] = -1.0
        camera_transforms = wp.array(transforms, dtype=wp.transformf, device="cpu")
        camera.world_indices = wp.array(np.zeros(3, dtype=np.int32), dtype=wp.int32, device="cpu")

        depth = camera.create_depth_image_output()
        self.assertEqual(depth.shape, (3, height, width))
        camera.render_context.update(state)
        camera.update(state, camera_transforms, depth_image=depth)

        d = depth.numpy()
        center = (height // 2, width // 2)
        self.assertTrue(all(float(d[v][center]) > 0.0 for v in range(3)))
        # The closer camera measures a smaller hit distance.
        self.assertGreater(float(d[0][center]), float(d[2][center]))

    def test_texture_projection_modes_texture_uvless_shapes(self) -> None:
        """Verify cubic and triplanar projection texture UV-less shapes and differ.

        A checkerboard is projected onto a UV-less sphere; both projection modes
        must texture it, and they must produce distinct results on the curved
        surface.
        """
        width, height = 32, 32

        def render(mode: int) -> np.ndarray:
            builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
            sphere_body = builder.add_body(xform=wp.transform(p=wp.vec3(0.0, 0.0, -2.5), q=wp.quat_identity()))
            sphere = builder.add_shape_sphere(sphere_body, radius=1.2, color=(1.0, 1.0, 1.0))
            model = builder.finalize(device="cpu")
            camera = self._camera_with_context(width, height, model, fov=math.radians(60.0))
            camera.render_config = newton.RenderConfig(enable_textures=True, texture_projection_mode=mode)
            camera.render_context.assign_checkerboard_material(shape_indices=[sphere])
            state = model.state()
            camera.render_context.update(state)
            albedo = camera.create_albedo_image_output()
            camera.update(state, self._identity_transforms(camera.view_count), albedo_image=albedo)
            return albedo.numpy()

        cubic = render(newton.TextureProjectionMode.CUBIC)
        triplanar = render(newton.TextureProjectionMode.TRIPLANAR)

        # Both modes project the checkerboard onto the UV-less sphere (not flat white).
        self.assertGreater(len(np.unique(cubic)), 1)
        self.assertGreater(len(np.unique(triplanar)), 1)
        # The two projection modes produce distinct results on a curved surface.
        self.assertFalse(np.array_equal(cubic, triplanar))

    def test_update_uses_instance_render_settings(self) -> None:
        """Verify SensorCamera update uses instance render settings."""
        parameters = inspect.signature(SensorCamera.update).parameters
        self.assertNotIn("clear_data", parameters)
        self.assertNotIn("render_config", parameters)
        self.assertNotIn("load_textures", parameters)
        self.assertNotIn("world_enabled", parameters)
        self.assertNotIn("model", parameters)
        self.assertIn("world_indices", parameters)

        width, height = 16, 12
        model = self._sphere_world_builder().finalize(device="cpu")
        camera = self._camera_with_context(width, height, model, load_textures=False)

        camera.clear_data = newton.ClearData(clear_depth=-2.0, clear_shape_index=123)
        camera.render_config = newton.RenderConfig(max_distance=0.1)
        self.assertFalse(hasattr(camera, "load_textures"))

        state = model.state()

        depth = wp.zeros((model.world_count, height, width), dtype=wp.float32, device="cpu")
        shape_index = wp.zeros((model.world_count, height, width), dtype=wp.uint32, device="cpu")

        camera.render_context.update(state)
        camera.update(
            state,
            self._identity_transforms(camera.view_count),
            depth_image=depth,
            shape_index_image=shape_index,
        )

        self.assertEqual(float(depth.numpy()[0, height // 2, width // 2]), -2.0)
        self.assertEqual(int(shape_index.numpy()[0, height // 2, width // 2]), 123)

    def test_update_supports_all_render_orders_with_3d_outputs(self) -> None:
        """Verify SensorCamera renders every render order into 3-D outputs."""
        width, height = 16, 12

        for render_order in newton.RenderOrder:
            with self.subTest(render_order=render_order):
                model, camera = self._build_sphere_scene(width, height)
                state = model.state()
                camera.render_config = newton.RenderConfig(render_order=render_order)

                depth = wp.zeros((model.world_count, height, width), dtype=wp.float32, device="cpu")

                camera.render_context.update(state)
                camera.update(state, self._identity_transforms(camera.view_count), depth_image=depth)

                self.assertGreater(float(depth.numpy()[0, height // 2, width // 2]), 0.0)

    def test_multiple_sensor_cameras_share_explicit_render_context(self) -> None:
        """Verify multiple SensorCamera instances share an explicit render context."""
        width, height = 8, 6
        model = self._sphere_world_builder().finalize(device="cpu")
        state = model.state()

        depth_a = wp.zeros((model.world_count, height, width), dtype=wp.float32, device="cpu")
        depth_b = wp.zeros((model.world_count, height, width), dtype=wp.float32, device="cpu")

        # Both cameras are constructed with the same render context instance.
        render_context = newton.RenderContext(model)
        rays_a = SensorCamera.compute_camera_rays_pinhole(width, height, math.radians(45.0), device="cpu")
        rays_b = SensorCamera.compute_camera_rays_pinhole(width, height, math.radians(45.0), device="cpu")
        camera_a = SensorCamera(rays_a, render_context)
        camera_b = SensorCamera(rays_b, render_context)

        render_context.update(state)
        camera_a.update(state, self._identity_transforms(camera_a.view_count), depth_image=depth_a)
        camera_b.update(state, self._identity_transforms(camera_b.view_count), depth_image=depth_b)

        self.assertIs(camera_a.render_context, render_context)
        self.assertIs(camera_b.render_context, render_context)
        self.assertGreater(float(depth_a.numpy()[0, height // 2, width // 2]), 0.0)
        self.assertGreater(float(depth_b.numpy()[0, height // 2, width // 2]), 0.0)

    # --- Rendering output channels (ported from SensorTiledCamera coverage) ---

    @staticmethod
    def _shaded_sphere_model(color: tuple[float, float, float] = (0.5, 0.5, 0.5)) -> newton.Model:
        builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
        body = builder.add_body(xform=wp.transform(p=wp.vec3(0.0, 0.0, -2.0), q=wp.quat_identity()))
        builder.add_shape_sphere(body, radius=1.0, color=color)
        return builder.finalize(device="cpu")

    def _render_color_and_hdr(self, output_color_space) -> tuple[np.ndarray, np.ndarray]:
        model = self._shaded_sphere_model()
        camera = self._camera_with_context(4, 4, model)
        camera.render_config = newton.RenderConfig(output_color_space=output_color_space)
        state = model.state()
        camera.render_context.update(state)
        color = camera.create_color_image_output()
        hdr = camera.create_hdr_color_image_output()
        camera.update(state, self._identity_transforms(camera.view_count), color_image=color, hdr_color_image=hdr)
        return np.asarray(color.numpy(), dtype=np.uint32), np.asarray(hdr.numpy(), dtype=np.float32)

    def test_render_hdr_color_output(self) -> None:
        """Verify SensorCamera produces a finite, non-zero HDR color channel."""
        color, hdr = self._render_color_and_hdr(newton.utils.ColorSpace.SRGB)
        self.assertEqual(color.shape, (1, 4, 4))
        self.assertEqual(hdr.shape, (1, 4, 4, 3))
        self.assertEqual(color.dtype, np.uint32)
        self.assertEqual(hdr.dtype, np.float32)
        self.assertTrue(np.isfinite(hdr).all())
        self.assertGreater(hdr.max(), 0.0)

    def test_hdr_color_matches_srgb_packed_color(self) -> None:
        """Verify packed color is the sRGB encoding of the HDR color for SRGB output."""
        color, hdr = self._render_color_and_hdr(newton.utils.ColorSpace.SRGB)
        clipped = np.clip(hdr, 0.0, 1.0)
        expected = np.where(clipped <= 0.0031308, clipped * 12.92, 1.055 * np.power(clipped, 1.0 / 2.4) - 0.055)
        packed = color.view(np.uint8).reshape(*color.shape, 4)[..., :3].astype(np.float32) / 255.0
        np.testing.assert_allclose(expected, packed, atol=1.0 / 255.0)

    def test_hdr_color_matches_linear_packed_color(self) -> None:
        """Verify packed color equals the clipped HDR color for LINEAR output."""
        color, hdr = self._render_color_and_hdr(newton.utils.ColorSpace.LINEAR)
        packed = color.view(np.uint8).reshape(*color.shape, 4)[..., :3].astype(np.float32) / 255.0
        np.testing.assert_allclose(np.clip(hdr, 0.0, 1.0), packed, atol=1.0 / 255.0)

    def test_albedo_output_follows_output_color_space(self) -> None:
        """Verify albedo packing honors the render-config output color space."""
        model = self._shaded_sphere_model(color=(0.25, 0.5, 0.75))

        def render_albedo(space) -> np.ndarray:
            camera = self._camera_with_context(8, 8, model)
            camera.render_config = newton.RenderConfig(output_color_space=space)
            state = model.state()
            camera.render_context.update(state)
            albedo = camera.create_albedo_image_output()
            camera.update(state, self._identity_transforms(camera.view_count), albedo_image=albedo)
            return albedo.numpy()

        srgb = render_albedo(newton.utils.ColorSpace.SRGB)
        linear = render_albedo(newton.utils.ColorSpace.LINEAR)
        self.assertFalse(np.array_equal(srgb, linear))

    def test_render_forward_depth_output(self) -> None:
        """Verify forward-depth is positive and never exceeds ray-hit distance."""
        width, height = 16, 12
        model, camera = self._build_sphere_scene(width, height)
        state = model.state()
        depth = wp.zeros((model.world_count, height, width), dtype=wp.float32, device="cpu")
        forward = wp.zeros((model.world_count, height, width), dtype=wp.float32, device="cpu")
        camera.render_context.update(state)
        camera.update(
            state, self._identity_transforms(camera.view_count), depth_image=depth, forward_depth_image=forward
        )
        center = (0, height // 2, width // 2)
        fwd = float(forward.numpy()[center])
        ray = float(depth.numpy()[center])
        self.assertGreater(fwd, 0.0)
        self.assertLessEqual(fwd, ray + 1.0e-4)

    # --- Utils to_rgba / flatten helpers (ported; new 3-D Utils) ---

    def test_utils_to_rgba_helpers_produce_canonical_outputs(self) -> None:
        """Verify the Utils to_rgba helpers return ``(view, H, W, 4)`` uint8 arrays."""
        width, height = 8, 6
        model, camera = self._build_sphere_scene(width, height)
        state = model.state()
        camera.render_context.update(state)
        color = camera.create_color_image_output()
        depth = camera.create_depth_image_output()
        normal = camera.create_normal_image_output()
        shape_index = camera.create_shape_index_image_output()
        camera.update(
            state,
            self._identity_transforms(camera.view_count),
            color_image=color,
            depth_image=depth,
            normal_image=normal,
            shape_index_image=shape_index,
        )

        utils = camera.utils
        for rgba in (
            utils.to_rgba_from_color(color),
            utils.to_rgba_from_depth(depth, depth_range=(0.0, 10.0)),
            utils.to_rgba_from_normal(normal),
            utils.to_rgba_from_shape_index(shape_index),
        ):
            self.assertEqual(rgba.shape, (camera.view_count, height, width, 4))
            self.assertEqual(rgba.dtype, wp.uint8)

    def test_utils_shape_index_hash_colors_differ_by_index(self) -> None:
        """Verify the shape-index hash palette assigns distinct colors (uint32 hash)."""
        width, height = 8, 6
        model, camera = self._build_sphere_scene(width, height)
        state = model.state()
        camera.render_context.update(state)
        shape_index = camera.create_shape_index_image_output()
        camera.update(state, self._identity_transforms(camera.view_count), shape_index_image=shape_index)
        rgba = camera.utils.to_rgba_from_shape_index(shape_index).numpy()
        colors = {tuple(c) for c in rgba.reshape(-1, 4)[:, :3]}
        self.assertGreater(len(colors), 1)

    def test_utils_flatten_rejects_worlds_per_row_below_one(self) -> None:
        """Verify the flatten helpers reject a non-positive ``worlds_per_row``."""
        width, height = 4, 3
        model, camera = self._build_sphere_scene(width, height)
        state = model.state()
        camera.render_context.update(state)
        color = camera.create_color_image_output()
        camera.update(state, self._identity_transforms(camera.view_count), color_image=color)
        with self.assertRaisesRegex(ValueError, "worlds_per_row"):
            camera.utils.flatten_color_image_to_rgba(color, worlds_per_row=0)


if __name__ == "__main__":
    unittest.main()
