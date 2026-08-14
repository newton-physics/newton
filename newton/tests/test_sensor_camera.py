# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import inspect
import math
import unittest

import numpy as np
import warp as wp

import newton
import newton._src.sensors.sensor_camera_render as internal_render
import newton.geometry as geometry
from newton._src.sensors.sensor_camera_render.utils import Utils
from newton.sensors import (
    SensorCamera,
)

# Transform placing a camera at the origin looking down -Z (identity pose). A
# camera with this transform sees a sphere placed at z = -2.
_IDENTITY_XFORM = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float32)


class TestSensorCamera(unittest.TestCase):
    @staticmethod
    def _rays(width: int, height: int, fov: float = math.radians(45.0), device: str = "cpu") -> wp.array3d[wp.vec3f]:
        """Camera-space pinhole rays, shape ``(height, width, 2)``."""
        return SensorCamera.compute_camera_rays_pinhole(width, height, fov, device=device)

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
        *,
        world_count: int = 1,
        assign_render_context: bool = True,
    ) -> tuple[newton.Model, SensorCamera]:
        if world_count == 1:
            builder = cls._sphere_world_builder()
        else:
            builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
            for _ in range(world_count):
                builder.add_world(cls._sphere_world_builder())

        model = builder.finalize(device="cpu")
        camera = SensorCamera(model if assign_render_context else None)
        return model, camera

    @staticmethod
    def _identity_transforms(view_count: int, device: str = "cpu") -> wp.array:
        """World-space identity camera poses, shape ``(view_count,)``."""
        return wp.array(np.tile(_IDENTITY_XFORM, (view_count, 1)), dtype=wp.transformf, device=device)

    @staticmethod
    def _camera_with_model(model: newton.Model, **camera_kwargs) -> SensorCamera:
        """A SensorCamera that renders ``model`` (owns an internal render context)."""
        return SensorCamera(model, **camera_kwargs)

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
        # RenderContext is an internal implementation detail owned by SensorCamera;
        # it is not part of the public API.
        self.assertFalse(hasattr(newton, "RenderContext"))
        self.assertNotIn("RenderContext", internal_render.__all__)
        self.assertFalse(hasattr(internal_render, "RenderContext"))
        # The render config/enum types are exposed as SensorCamera nested attributes,
        # not on the top-level namespace, and the ``newton.render`` module is gone.
        self.assertFalse(hasattr(newton, "render"))
        render_types = (
            "ClearData",
            "GaussianRenderMode",
            "LightType",
            "RenderConfig",
            "RenderOrder",
            "TextureProjectionMode",
            "WorldRenderFlag",
        )
        for type_name in render_types:
            self.assertFalse(hasattr(newton, type_name), type_name)
            self.assertNotIn(type_name, newton.__all__)
            self.assertTrue(hasattr(SensorCamera, type_name), type_name)
            self.assertIs(getattr(SensorCamera, type_name), getattr(internal_render, type_name))
        # The post-processing Utils and the gray clear preset are nested on SensorCamera.
        self.assertIs(SensorCamera.Utils, Utils)
        self.assertFalse(hasattr(geometry, "SensorCamera"))

        # The caller owns the rays, transforms, and output buffers; the sensor holds
        # none of them, and is not attached to model sites.
        camera = SensorCamera()
        for attr in (
            "rays",
            "view_count",
            "width",
            "height",
            "world_indices",
            "camera_transforms",
            "shape_indices",
            "_shape_index_by_world",
            "_camera_transforms",
        ):
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
        self.assertFalse(hasattr(SensorCamera.WorldRenderFlag, "ENABLE"))
        self.assertEqual(int(SensorCamera.WorldRenderFlag.DISABLE_PRESERVE), -1)
        self.assertEqual(int(SensorCamera.WorldRenderFlag.DISABLE_CLEAR), -2)

    def test_constructor_without_model_is_inert(self) -> None:
        """Verify a model-less SensorCamera owns only render settings and rejects rendering."""
        camera = SensorCamera()

        # Only render settings; no render context, rays, dimensions, or buffers.
        self.assertFalse(hasattr(camera, "render_context"))
        self.assertIsInstance(camera.default_render_config, SensorCamera.RenderConfig)
        self.assertIsInstance(camera.default_clear_data, SensorCamera.ClearData)

        state = self._sphere_world_builder().finalize(device="cpu").state()
        camera_transforms = self._identity_transforms(1)
        rays = self._rays(4, 4)
        with self.assertRaisesRegex(RuntimeError, "no model"):
            camera.update(state, camera_transforms, rays)
        with self.assertRaisesRegex(RuntimeError, "no model"):
            _ = camera.device
        with self.assertRaisesRegex(RuntimeError, "no model"):
            camera.create_color_image_output(1, 4, 4)
        with self.assertRaisesRegex(RuntimeError, "no model"):
            camera.utils(1)

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

    def test_update_validates_rays_and_transforms(self) -> None:
        """Verify update rejects mistyped or misshaped rays and camera transforms."""
        width, height = 8, 6
        model, camera = self._build_sphere_scene()
        state = model.state()
        rays = self._rays(width, height)
        camera_transforms = self._identity_transforms(model.world_count)

        with self.assertRaisesRegex(ValueError, "camera_transforms must have dtype"):
            camera.update(state, rays, rays)
        with self.assertRaisesRegex(ValueError, "camera_transforms must have shape"):
            camera.update(state, camera_transforms.reshape((model.world_count, 1)), rays)
        with self.assertRaisesRegex(ValueError, "camera_rays must have dtype"):
            camera.update(state, camera_transforms, camera_transforms)
        with self.assertRaisesRegex(ValueError, "camera_rays must have shape"):
            camera.update(state, camera_transforms, rays.reshape((1, height, width, 2)))
        with self.assertRaises(TypeError):
            camera.update(state, camera_transforms, np.zeros((height, width, 2), dtype=np.float32))

    def test_model_required_for_outputs_and_utils(self) -> None:
        """Verify output and utility helpers require a model, and report the model device."""
        # A camera without a model cannot produce buffers or utils.
        camera = SensorCamera()
        with self.assertRaisesRegex(RuntimeError, "no model"):
            camera.create_image_output(1, 4, 3, wp.float32)
        with self.assertRaisesRegex(RuntimeError, "no model"):
            camera.utils(1)

        model, camera = self._build_sphere_scene(world_count=2)
        self.assertEqual(camera.device, model.device)
        self.assertFalse(hasattr(model, "render_context"))
        self.assertFalse(hasattr(camera, "render_context"))

    def test_utils_and_scene_config_from_model(self) -> None:
        """Verify a model-backed SensorCamera exposes utils, output buffers, and scene config."""
        width, height = 4, 3
        model, camera = self._build_sphere_scene()
        view_count = model.world_count

        utils = camera.utils(view_count)

        self.assertIsInstance(utils, Utils)
        self.assertIsNot(camera.utils(view_count), utils)
        self.assertFalse(hasattr(utils, "_Utils__sensor_camera"))
        self.assertFalse(hasattr(model, "render_context"))
        self.assertFalse(hasattr(camera, "render_context"))
        self.assertEqual(camera.device, model.device)
        self.assertFalse(hasattr(camera, "_model_ref"))

        output_specs = (
            (camera.create_image_output(view_count, width, height, wp.float32), wp.float32),
            (camera.create_color_image_output(view_count, width, height), wp.uint32),
            (camera.create_depth_image_output(view_count, width, height), wp.float32),
            (camera.create_forward_depth_image_output(view_count, width, height), wp.float32),
            (camera.create_shape_index_image_output(view_count, width, height), wp.uint32),
            (camera.create_normal_image_output(view_count, width, height), wp.vec3f),
            (camera.create_albedo_image_output(view_count, width, height), wp.uint32),
            (camera.create_hdr_color_image_output(view_count, width, height), wp.vec3f),
        )
        for output, dtype in output_specs:
            with self.subTest(dtype=dtype):
                self.assertEqual(output.shape, (view_count, height, width))
                self.assertEqual(output.dtype, dtype)
                self.assertEqual(output.device, model.device)
        color_rgba = utils.to_rgba_from_color(camera.create_color_image_output(view_count, width, height))
        self.assertEqual(color_rgba.shape, (view_count, height, width, 4))
        # Scene configuration is surfaced on the camera; the render context is private.
        camera.create_default_light(enable_shadows=True)
        camera.assign_checkerboard_material(shape_indices=[0])

    @unittest.skipUnless(wp.is_cuda_available(), "Requires CUDA")
    def test_update_requires_arrays_on_model_device(self) -> None:
        """Verify update rejects rays or transforms that are not on the model device."""
        width, height = 2, 2
        model = self._sphere_world_builder().finalize(device="cuda:0")
        camera = SensorCamera(model)
        self.assertEqual(camera.device, model.device)
        state = model.state()

        cpu_rays = self._rays(width, height, device="cpu")
        cpu_transforms = self._identity_transforms(model.world_count, device="cpu")
        cuda_rays = self._rays(width, height, device="cuda:0")
        cuda_transforms = self._identity_transforms(model.world_count, device="cuda:0")

        with self.assertRaisesRegex(RuntimeError, "camera_transforms must be on the model device"):
            camera.update(state, cpu_transforms, cuda_rays)
        with self.assertRaisesRegex(RuntimeError, "camera_rays must be on the model device"):
            camera.update(state, cuda_transforms, cpu_rays)

    def test_update_renders_from_camera_transforms(self) -> None:
        """Verify SensorCamera renders from the camera transforms passed to update."""
        width, height = 16, 12
        model, camera = self._build_sphere_scene()
        state = model.state()
        rays = self._rays(width, height)
        view_count = model.world_count

        depth = wp.zeros((view_count, height, width), dtype=wp.float32, device="cpu")
        shape_index = wp.zeros((view_count, height, width), dtype=wp.uint32, device="cpu")

        # Identity transforms see the sphere placed in front of the camera.
        camera_transforms = self._identity_transforms(view_count)
        camera.update(state, camera_transforms, rays, depth_image=depth, shape_index_image=shape_index)

        center = (0, height // 2, width // 2)
        identity_center_depth = float(depth.numpy()[center])
        self.assertGreater(identity_center_depth, 0.0)
        self.assertTrue(np.any(shape_index.numpy() != 0xFFFFFFFF))
        self.assertFalse(hasattr(model, "render_context"))

        # Move the camera behind the sphere; the center ray no longer hits it.
        behind = np.tile(np.array([0.0, 0.0, -4.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float32), (view_count, 1))
        camera_transforms.assign(behind)
        depth.zero_()
        camera.update(state, camera_transforms, rays, depth_image=depth)
        self.assertEqual(float(depth.numpy()[center]), 0.0)

    def test_update_respects_disable_clear_flag(self) -> None:
        """Verify SensorCamera clears output images for DISABLE_CLEAR worlds."""
        width, height = 16, 12
        model, camera = self._build_sphere_scene(world_count=2)
        state = model.state()
        rays = self._rays(width, height)
        view_count = model.world_count

        camera.default_clear_data = SensorCamera.ClearData(clear_depth=-2.0, clear_shape_index=123)
        world_indices = wp.array(
            [0, int(SensorCamera.WorldRenderFlag.DISABLE_CLEAR)],
            dtype=wp.int32,
            device="cpu",
        )
        depth = wp.zeros((view_count, height, width), dtype=wp.float32, device="cpu")
        shape_index = wp.zeros((view_count, height, width), dtype=wp.uint32, device="cpu")

        camera.update(
            state,
            self._identity_transforms(view_count),
            rays,
            depth_image=depth,
            shape_index_image=shape_index,
            world_indices=world_indices,
        )

        depth_np = depth.numpy()
        shape_index_np = shape_index.numpy()
        self.assertGreater(float(depth_np[0, height // 2, width // 2]), 0.0)
        self.assertEqual(float(depth_np[1, height // 2, width // 2]), -2.0)
        self.assertEqual(int(shape_index_np[1, height // 2, width // 2]), 123)

    def test_update_requires_model(self) -> None:
        """Verify SensorCamera rendering requires a model given at construction."""
        width, height = 8, 6
        model, camera = self._build_sphere_scene(assign_render_context=False)
        state = model.state()
        depth = wp.zeros((model.world_count, height, width), dtype=wp.float32, device="cpu")

        with self.assertRaisesRegex(RuntimeError, "no model"):
            camera.update(
                state,
                self._identity_transforms(model.world_count),
                self._rays(width, height),
                depth_image=depth,
            )

    def test_update_respects_disable_preserve_flag(self) -> None:
        """Verify SensorCamera preserves output images for DISABLE_PRESERVE worlds."""
        width, height = 16, 12
        model, camera = self._build_sphere_scene(world_count=2)
        state = model.state()
        rays = self._rays(width, height)
        view_count = model.world_count

        world_indices = wp.array(
            [0, int(SensorCamera.WorldRenderFlag.DISABLE_PRESERVE)],
            dtype=wp.int32,
            device="cpu",
        )
        depth = wp.full((view_count, height, width), value=42.0, dtype=wp.float32, device="cpu")
        shape_index = wp.full((view_count, height, width), value=456, dtype=wp.uint32, device="cpu")

        camera.update(
            state,
            self._identity_transforms(view_count),
            rays,
            depth_image=depth,
            shape_index_image=shape_index,
            world_indices=world_indices,
        )

        depth_np = depth.numpy()
        shape_index_np = shape_index.numpy()
        self.assertGreater(float(depth_np[0, height // 2, width // 2]), 0.0)
        np.testing.assert_allclose(depth_np[1], 42.0)
        np.testing.assert_array_equal(shape_index_np[1], np.full((height, width), 456, dtype=np.uint32))

    def test_update_defaults_world_indices_to_identity(self) -> None:
        """Verify update maps view i to world i when world_indices is omitted."""
        width, height = 16, 12
        model, camera = self._build_sphere_scene(world_count=2)
        state = model.state()
        rays = self._rays(width, height)
        view_count = model.world_count
        depth = wp.zeros((view_count, height, width), dtype=wp.float32, device="cpu")

        # No world_indices passed: each view renders its own world (identity mapping).
        camera.update(state, self._identity_transforms(view_count), rays, depth_image=depth)

        center = (height // 2, width // 2)
        self.assertTrue(all(float(depth.numpy()[v][center]) > 0.0 for v in range(view_count)))

    def test_default_world_indices_cache_is_reused_and_grown(self) -> None:
        """Verify the default world-indices cache is reused across calls and grown only when needed."""
        width, height = 8, 6
        model, camera = self._build_sphere_scene(world_count=5)
        state = model.state()
        rays = self._rays(width, height)

        def render(view_count: int) -> None:
            depth = wp.zeros((view_count, height, width), dtype=wp.float32, device="cpu")
            camera.update(state, self._identity_transforms(view_count), rays, depth_image=depth)

        # Allocated lazily on the first update that needs a default.
        self.assertIsNone(camera._default_world_indices)

        render(3)
        cache = camera._default_world_indices
        self.assertIsNotNone(cache)
        self.assertEqual(cache.shape, (3,))
        np.testing.assert_array_equal(cache.numpy(), np.arange(3, dtype=np.int32))

        # Same or smaller view counts reuse the cached array (sliced to fit).
        render(3)
        self.assertIs(camera._default_world_indices, cache)
        render(2)
        self.assertIs(camera._default_world_indices, cache)

        # A larger view count grows the cache to the new maximum.
        render(5)
        grown = camera._default_world_indices
        self.assertIsNot(grown, cache)
        self.assertEqual(grown.shape, (5,))
        np.testing.assert_array_equal(grown.numpy(), np.arange(5, dtype=np.int32))

        # Below the maximum, the grown cache is reused, not reallocated.
        render(4)
        self.assertIs(camera._default_world_indices, grown)

        # Explicit world_indices bypass the cache entirely.
        depth = wp.zeros((5, height, width), dtype=wp.float32, device="cpu")
        explicit = wp.array(np.zeros(5, dtype=np.int32), dtype=wp.int32, device="cpu")
        camera.update(state, self._identity_transforms(5), rays, depth_image=depth, world_indices=explicit)
        self.assertIs(camera._default_world_indices, grown)

    def test_world_indices_decouple_views_from_worlds(self) -> None:
        """Verify multiple views can render one shared world from different poses."""
        width, height = 8, 6
        # One world (sphere at z=-2) but three views, all rendering world 0.
        model, camera = self._build_sphere_scene()
        state = model.state()
        rays = self._rays(width, height)
        self.assertEqual(model.world_count, 1)

        # Three views of world 0 from progressively closer poses.
        transforms = np.tile(_IDENTITY_XFORM, (3, 1))
        transforms[1, 2] = -0.5
        transforms[2, 2] = -1.0
        camera_transforms = wp.array(transforms, dtype=wp.transformf, device="cpu")
        world_indices = wp.array(np.zeros(3, dtype=np.int32), dtype=wp.int32, device="cpu")

        depth = camera.create_depth_image_output(3, width, height)
        self.assertEqual(depth.shape, (3, height, width))
        camera.update(state, camera_transforms, rays, depth_image=depth, world_indices=world_indices)

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
            camera = self._camera_with_model(model)
            camera.default_render_config = SensorCamera.RenderConfig(enable_textures=True, texture_projection_mode=mode)
            camera.assign_checkerboard_material(shape_indices=[sphere])
            state = model.state()
            rays = self._rays(width, height, math.radians(60.0))
            albedo = camera.create_albedo_image_output(model.world_count, width, height)
            camera.update(state, self._identity_transforms(model.world_count), rays, albedo_image=albedo)
            return albedo.numpy()

        cubic = render(SensorCamera.TextureProjectionMode.CUBIC)
        triplanar = render(SensorCamera.TextureProjectionMode.TRIPLANAR)

        # Both modes project the checkerboard onto the UV-less sphere (not flat white).
        self.assertGreater(len(np.unique(cubic)), 1)
        self.assertGreater(len(np.unique(triplanar)), 1)
        # The two projection modes produce distinct results on a curved surface.
        self.assertFalse(np.array_equal(cubic, triplanar))

    def test_update_uses_default_render_settings(self) -> None:
        """Verify update falls back to the default clear data and render config."""
        parameters = inspect.signature(SensorCamera.update).parameters
        self.assertIn("camera_transforms", parameters)
        self.assertIn("camera_rays", parameters)
        self.assertIn("world_indices", parameters)
        self.assertIn("clear_data", parameters)
        self.assertIn("render_config", parameters)
        self.assertNotIn("load_textures", parameters)
        self.assertNotIn("world_enabled", parameters)
        self.assertNotIn("model", parameters)

        width, height = 16, 12
        model = self._sphere_world_builder().finalize(device="cpu")
        # Defaults may also be provided at construction.
        camera = SensorCamera(
            model,
            default_clear_data=SensorCamera.ClearData(clear_depth=-2.0, clear_shape_index=123),
            default_render_config=SensorCamera.RenderConfig(max_distance=0.1),
            load_textures=False,
        )
        self.assertFalse(hasattr(camera, "load_textures"))

        state = model.state()
        rays = self._rays(width, height)
        view_count = model.world_count

        depth = wp.zeros((view_count, height, width), dtype=wp.float32, device="cpu")
        shape_index = wp.zeros((view_count, height, width), dtype=wp.uint32, device="cpu")

        # No per-call overrides: max_distance=0.1 misses the sphere, so the depth
        # and shape-index outputs take the default clear values.
        camera.update(
            state,
            self._identity_transforms(view_count),
            rays,
            depth_image=depth,
            shape_index_image=shape_index,
        )

        self.assertEqual(float(depth.numpy()[0, height // 2, width // 2]), -2.0)
        self.assertEqual(int(shape_index.numpy()[0, height // 2, width // 2]), 123)

    def test_update_overrides_default_render_settings(self) -> None:
        """Verify per-call clear_data and render_config override the defaults."""
        width, height = 16, 12
        model, camera = self._build_sphere_scene()
        state = model.state()
        rays = self._rays(width, height)
        view_count = model.world_count
        center = (0, height // 2, width // 2)

        # Defaults would clear to -7.0 and cull the sphere (max_distance=0.1)...
        camera.default_clear_data = SensorCamera.ClearData(clear_depth=-7.0)
        camera.default_render_config = SensorCamera.RenderConfig(max_distance=0.1)

        depth = wp.zeros((view_count, height, width), dtype=wp.float32, device="cpu")
        # ...but the per-call overrides raise max_distance so the sphere is hit.
        camera.update(
            state,
            self._identity_transforms(view_count),
            rays,
            depth_image=depth,
            clear_data=SensorCamera.ClearData(clear_depth=-3.0),
            render_config=SensorCamera.RenderConfig(max_distance=1000.0),
        )
        self.assertGreater(float(depth.numpy()[center]), 0.0)

        # A miss with the override clear_data writes the override's clear value.
        depth.zero_()
        behind = np.tile(np.array([0.0, 0.0, -4.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float32), (view_count, 1))
        camera.update(
            state,
            wp.array(behind, dtype=wp.transformf, device="cpu"),
            rays,
            depth_image=depth,
            clear_data=SensorCamera.ClearData(clear_depth=-3.0),
            render_config=SensorCamera.RenderConfig(max_distance=1000.0),
        )
        self.assertEqual(float(depth.numpy()[center]), -3.0)

    def test_update_supports_all_render_orders_with_3d_outputs(self) -> None:
        """Verify SensorCamera renders every render order into 3-D outputs."""
        width, height = 16, 12

        for render_order in SensorCamera.RenderOrder:
            with self.subTest(render_order=render_order):
                model, camera = self._build_sphere_scene()
                state = model.state()
                rays = self._rays(width, height)
                camera.default_render_config = SensorCamera.RenderConfig(render_order=render_order)

                depth = wp.zeros((model.world_count, height, width), dtype=wp.float32, device="cpu")

                camera.update(state, self._identity_transforms(model.world_count), rays, depth_image=depth)

                self.assertGreater(float(depth.numpy()[0, height // 2, width // 2]), 0.0)

    def test_multiple_sensor_cameras_render_same_model(self) -> None:
        """Verify multiple independent SensorCamera instances can render the same model."""
        width, height = 8, 6
        model = self._sphere_world_builder().finalize(device="cpu")
        state = model.state()
        rays = self._rays(width, height)
        view_count = model.world_count

        depth_a = wp.zeros((view_count, height, width), dtype=wp.float32, device="cpu")
        depth_b = wp.zeros((view_count, height, width), dtype=wp.float32, device="cpu")

        # Each camera owns its own private render context for the same model.
        camera_a = self._camera_with_model(model)
        camera_b = self._camera_with_model(model)

        camera_a.update(state, self._identity_transforms(view_count), rays, depth_image=depth_a)
        camera_b.update(state, self._identity_transforms(view_count), rays, depth_image=depth_b)

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
        width, height = 4, 4
        model = self._shaded_sphere_model()
        camera = self._camera_with_model(model)
        camera.default_render_config = SensorCamera.RenderConfig(output_color_space=output_color_space)
        state = model.state()
        rays = self._rays(width, height)
        view_count = model.world_count
        color = camera.create_color_image_output(view_count, width, height)
        hdr = camera.create_hdr_color_image_output(view_count, width, height)
        camera.update(state, self._identity_transforms(view_count), rays, color_image=color, hdr_color_image=hdr)
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
        width, height = 8, 8
        model = self._shaded_sphere_model(color=(0.25, 0.5, 0.75))

        def render_albedo(space) -> np.ndarray:
            camera = self._camera_with_model(model)
            camera.default_render_config = SensorCamera.RenderConfig(output_color_space=space)
            state = model.state()
            rays = self._rays(width, height)
            albedo = camera.create_albedo_image_output(model.world_count, width, height)
            camera.update(state, self._identity_transforms(model.world_count), rays, albedo_image=albedo)
            return albedo.numpy()

        srgb = render_albedo(newton.utils.ColorSpace.SRGB)
        linear = render_albedo(newton.utils.ColorSpace.LINEAR)
        self.assertFalse(np.array_equal(srgb, linear))

    def test_render_forward_depth_output(self) -> None:
        """Verify forward-depth is positive and never exceeds ray-hit distance."""
        width, height = 16, 12
        model, camera = self._build_sphere_scene()
        state = model.state()
        rays = self._rays(width, height)
        view_count = model.world_count
        depth = wp.zeros((view_count, height, width), dtype=wp.float32, device="cpu")
        forward = wp.zeros((view_count, height, width), dtype=wp.float32, device="cpu")
        camera.update(
            state, self._identity_transforms(view_count), rays, depth_image=depth, forward_depth_image=forward
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
        model, camera = self._build_sphere_scene()
        state = model.state()
        rays = self._rays(width, height)
        view_count = model.world_count
        color = camera.create_color_image_output(view_count, width, height)
        depth = camera.create_depth_image_output(view_count, width, height)
        normal = camera.create_normal_image_output(view_count, width, height)
        shape_index = camera.create_shape_index_image_output(view_count, width, height)
        camera.update(
            state,
            self._identity_transforms(view_count),
            rays,
            color_image=color,
            depth_image=depth,
            normal_image=normal,
            shape_index_image=shape_index,
        )

        utils = camera.utils(view_count)
        for rgba in (
            utils.to_rgba_from_color(color),
            utils.to_rgba_from_depth(depth, depth_range=(0.0, 10.0)),
            utils.to_rgba_from_normal(normal),
            utils.to_rgba_from_shape_index(shape_index),
        ):
            self.assertEqual(rgba.shape, (view_count, height, width, 4))
            self.assertEqual(rgba.dtype, wp.uint8)

    def test_utils_postprocessing_helpers(self) -> None:
        """Verify forward-depth conversion, normal/depth flatten, palette colorize, and depth-range branches."""
        width, height, worlds_per_row = 6, 4, 2
        model, camera = self._build_sphere_scene(world_count=4)
        state = model.state()
        rays = self._rays(width, height)
        view_count = model.world_count
        camera_transforms = self._identity_transforms(view_count)
        depth = camera.create_depth_image_output(view_count, width, height)
        normal = camera.create_normal_image_output(view_count, width, height)
        shape_index = camera.create_shape_index_image_output(view_count, width, height)
        camera.update(
            state, camera_transforms, rays, depth_image=depth, normal_image=normal, shape_index_image=shape_index
        )

        utils = camera.utils(view_count)
        center = (0, height // 2, width // 2)
        self.assertGreater(float(depth.numpy()[center]), 0.0)

        # Ray-distance depth -> forward (planar) depth; must not exceed ray depth.
        forward = utils.convert_ray_depth_to_forward_depth(depth, camera_transforms, rays)
        self.assertEqual(forward.shape, depth.shape)
        self.assertEqual(forward.dtype, wp.float32)
        self.assertLessEqual(float(forward.numpy()[center]), float(depth.numpy()[center]) + 1.0e-4)

        # Flatten normal/depth into one tiled (rows*H, cols*W, 4) grid buffer.
        worlds_per_col = -(-view_count // worlds_per_row)
        for flat in (
            utils.flatten_normal_image_to_rgba(normal, worlds_per_row=worlds_per_row),
            utils.flatten_depth_image_to_rgba(depth, worlds_per_row=worlds_per_row),
        ):
            self.assertEqual(flat.shape, (worlds_per_col * height, worlds_per_row * width, 4))
            self.assertEqual(flat.dtype, wp.uint8)

        # Shape-index colorized via a caller palette (out-of-range indices -> black).
        palette = wp.array(np.array([[10, 20, 30]], dtype=np.uint8), dtype=wp.uint8, device="cpu")
        colored = utils.to_rgba_from_shape_index(shape_index, colors=palette)
        self.assertEqual(colored.shape, (view_count, height, width, 4))

        # to_rgba_from_depth: on-device auto range (depth_range=None) and the near<far guard.
        auto = utils.to_rgba_from_depth(depth)
        self.assertEqual(auto.shape, (view_count, height, width, 4))
        with self.assertRaisesRegex(ValueError, "near < far"):
            utils.to_rgba_from_depth(depth, depth_range=(5.0, 1.0))

    def test_utils_shape_index_hash_colors_differ_by_index(self) -> None:
        """Verify the shape-index hash palette assigns distinct colors (uint32 hash)."""
        width, height = 8, 6
        model, camera = self._build_sphere_scene()
        state = model.state()
        rays = self._rays(width, height)
        view_count = model.world_count
        shape_index = camera.create_shape_index_image_output(view_count, width, height)
        camera.update(state, self._identity_transforms(view_count), rays, shape_index_image=shape_index)
        rgba = camera.utils(view_count).to_rgba_from_shape_index(shape_index).numpy()
        colors = {tuple(c) for c in rgba.reshape(-1, 4)[:, :3]}
        self.assertGreater(len(colors), 1)

    def test_utils_flatten_rejects_worlds_per_row_below_one(self) -> None:
        """Verify the flatten helpers reject a non-positive ``worlds_per_row``."""
        width, height = 4, 3
        model, camera = self._build_sphere_scene()
        state = model.state()
        rays = self._rays(width, height)
        view_count = model.world_count
        color = camera.create_color_image_output(view_count, width, height)
        camera.update(state, self._identity_transforms(view_count), rays, color_image=color)
        with self.assertRaisesRegex(ValueError, "worlds_per_row"):
            camera.utils(view_count).flatten_color_image_to_rgba(color, worlds_per_row=0)


if __name__ == "__main__":
    unittest.main()
