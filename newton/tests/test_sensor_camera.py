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
from newton._src.sensors.sensor_camera_render.types import LightType
from newton._src.sensors.sensor_camera_render.utils import Utils
from newton.sensors import (
    SensorCamera,
)
from newton.tests.unittest_utils import get_test_devices

# Transform placing a camera at the origin looking down -Z (identity pose). A
# camera with this transform sees a sphere placed at z = -2.
_IDENTITY_XFORM = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float32)


class TestSensorCamera(unittest.TestCase):
    @staticmethod
    def _rays(width: int, height: int, fov: float = math.radians(45.0), device: str = "cpu") -> wp.array4d[wp.vec3f]:
        """Camera-space pinhole rays, shape ``(height, width, 1, 2)``."""
        return SensorCamera.compute_camera_rays_pinhole(width, height, camera_fov=fov, device=device)

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
    ) -> tuple[newton.Model, SensorCamera]:
        if world_count == 1:
            builder = cls._sphere_world_builder()
        else:
            builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
            for _ in range(world_count):
                builder.add_world(cls._sphere_world_builder())

        model = builder.finalize(device="cpu")
        camera = SensorCamera(model)
        return model, camera

    @staticmethod
    def _identity_transforms(view_count: int, device: str = "cpu") -> wp.array[wp.transformf]:
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
            "AntiAliasing",
            "ClearData",
            "GaussianRenderMode",
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
        # LightType is intentionally not exposed on SensorCamera (no public API accepts it yet).
        self.assertFalse(hasattr(SensorCamera, "LightType"))
        # The post-processing Utils and the gray clear preset are nested on SensorCamera.
        self.assertIs(SensorCamera.Utils, Utils)
        self.assertFalse(hasattr(geometry, "SensorCamera"))

        # The caller owns the rays, transforms, and output buffers; the sensor holds
        # none of them, and is not attached to model sites.
        _, camera = self._build_sphere_scene()
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
        self.assertEqual(int(SensorCamera.WorldRenderFlag.DISABLE_PRESERVE), -101)
        self.assertEqual(int(SensorCamera.WorldRenderFlag.DISABLE_CLEAR), -102)

    def test_camera_ray_helpers_live_on_sensor_camera(self) -> None:
        """Verify camera ray helpers live on SensorCamera."""
        sensor_helper_names = (
            "compute_camera_rays_pinhole",
            "compute_camera_rays_usd_pinhole",
            "compute_camera_rays_pinhole_opencv",
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
            SensorCamera.compute_camera_rays_pinhole(width, height, camera_fov=math.radians(45.0), device="cpu"),
            SensorCamera.compute_camera_rays_pinhole(
                width,
                height,
                focal_length=1.0,
                horizontal_aperture=2.0,
                vertical_aperture=2.0,
                device="cpu",
            ),
            SensorCamera.compute_camera_rays_pinhole_opencv(
                width, height, fx=2.0, fy=4.0, cx=1.0, cy=2.0, device="cpu"
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
            self.assertEqual(ray_bundle.shape, (height, width, 1, 2))
            self.assertEqual(ray_bundle.dtype, wp.vec3f)

    def test_camera_ray_helpers_support_preallocated_output(self) -> None:
        """Verify camera ray helpers can write into caller output arrays."""
        width, height = 4, 3
        out_rays = wp.zeros((height, width, 1, 2), dtype=wp.vec3f, device="cpu")

        rays = SensorCamera.compute_camera_rays_pinhole(
            width, height, camera_fov=math.radians(45.0), out_rays=out_rays, device="cpu"
        )

        self.assertIs(rays, out_rays)
        self.assertFalse(np.allclose(rays.numpy(), 0.0))

    def test_camera_ray_helpers_generate_multisamples(self) -> None:
        """Generate distinct subpixel rays from every camera model."""
        width, height = 4, 3
        default_rays = SensorCamera.compute_camera_rays_pinhole(
            width, height, camera_fov=math.radians(45.0), device="cpu"
        )
        one_sample_rays = SensorCamera.compute_camera_rays_pinhole(
            width, height, camera_fov=math.radians(45.0), sample_count=1, device="cpu"
        )
        multisample_rays = SensorCamera.compute_camera_rays_pinhole(
            width, height, camera_fov=math.radians(45.0), sample_count=4, device="cpu"
        )

        self.assertEqual(default_rays.shape, (height, width, 1, 2))
        self.assertEqual(one_sample_rays.shape, (height, width, 1, 2))
        self.assertEqual(multisample_rays.shape, (height, width, 4, 2))
        np.testing.assert_allclose(one_sample_rays.numpy(), default_rays.numpy(), atol=1.0e-6)
        sample_directions = multisample_rays.numpy()[1, 1, :, 1]
        self.assertFalse(np.allclose(sample_directions, sample_directions[0]))
        np.testing.assert_allclose(np.linalg.norm(sample_directions, axis=1), 1.0, atol=1.0e-6)

        calibrated_rays = (
            SensorCamera.compute_camera_rays_pinhole_opencv(
                width, height, fx=2.0, fy=2.0, cx=2.0, cy=1.5, sample_count=4, device="cpu"
            ),
            SensorCamera.compute_camera_rays_fisheye_opencv(
                width, height, fx=2.0, fy=2.0, cx=2.0, cy=1.5, sample_count=4, device="cpu"
            ),
            SensorCamera.compute_camera_rays_fisheye_ftheta(
                width, height, optical_center_x=2.0, optical_center_y=1.5, sample_count=4, device="cpu"
            ),
            SensorCamera.compute_camera_rays_fisheye_kannala_brandt(
                width, height, optical_center_x=2.0, optical_center_y=1.5, sample_count=4, device="cpu"
            ),
        )
        for rays in calibrated_rays:
            self.assertEqual(rays.shape, (height, width, 4, 2))
            directions = rays.numpy()[1, 1, :, 1]
            self.assertFalse(np.allclose(directions, directions[0]))

        out_rays = wp.empty((height, width, 4, 2), dtype=wp.vec3f, device="cpu")
        rays = SensorCamera.compute_camera_rays_pinhole(
            width,
            height,
            focal_length=1.0,
            horizontal_aperture=2.0,
            vertical_aperture=1.5,
            sample_count=4,
            out_rays=out_rays,
        )
        self.assertIs(rays, out_rays)
        for sample_count in (0, -1):
            with self.subTest(sample_count=sample_count):
                with self.assertRaisesRegex(ValueError, "sample_count must be positive"):
                    SensorCamera.compute_camera_rays_pinhole(
                        width, height, camera_fov=1.0, sample_count=sample_count, device="cpu"
                    )

    def test_camera_multisample_offsets_center_on_pixel(self) -> None:
        """Center the pattern and place its first ray at the center when possible."""
        center_ray = SensorCamera.compute_camera_rays_pinhole(1, 1, camera_fov=1.0, device="cpu").numpy()[0, 0, 0, 1]
        for sample_count in (2, 3, 4, 5, 6, 8, 16):
            with self.subTest(sample_count=sample_count):
                rays = SensorCamera.compute_camera_rays_pinhole(
                    1, 1, camera_fov=1.0, sample_count=sample_count, device="cpu"
                ).numpy()[0, 0, :, 1]
                ray_slopes = -rays[:, :2] / rays[:, 2, None]
                np.testing.assert_allclose(ray_slopes.mean(axis=0), (0.0, 0.0), atol=1.0e-6)
                self.assertEqual(len(np.unique(np.round(ray_slopes, 6), axis=0)), sample_count)
                if sample_count == 2:
                    self.assertFalse(np.allclose(rays[0], center_ray))
                else:
                    np.testing.assert_allclose(rays[0], center_ray, atol=1.0e-6)
                    ring = ray_slopes[1:]
                    radii = np.linalg.norm(ring, axis=1)
                    np.testing.assert_allclose(radii, radii[0], atol=1.0e-6)
                    angles = np.sort(np.mod(np.arctan2(ring[:, 1], ring[:, 0]), 2.0 * math.pi))
                    gaps = np.diff(np.append(angles, angles[0] + 2.0 * math.pi))
                    np.testing.assert_allclose(gaps, 2.0 * math.pi / (sample_count - 1), atol=1.0e-6)

    def test_camera_ray_helpers_reject_batched_inputs(self) -> None:
        """Verify camera ray helpers accept only single-camera parameters."""
        width, height = 4, 3

        with self.assertRaisesRegex(ValueError, "camera_fov cannot be provided with aperture parameters"):
            SensorCamera.compute_camera_rays_pinhole(
                width,
                height,
                camera_fov=math.radians(45.0),
                focal_length=1.0,
                horizontal_aperture=2.0,
                vertical_aperture=2.0,
                device="cpu",
            )

        with self.assertRaises(TypeError):
            SensorCamera.compute_camera_rays_pinhole(width, height, camera_fov=[math.radians(45.0)], device="cpu")

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
            SensorCamera.compute_camera_rays_pinhole(width, height, camera_fov=math.radians(45.0), out_rays=out_rays)

    def test_pinhole_rays_reject_out_of_range_parameters(self) -> None:
        """Verify pinhole ray generation rejects non-positive focal length and out-of-range fov."""
        width, height = 4, 3
        for bad_fov in (0.0, math.pi, -0.1, math.pi + 0.1):
            with self.assertRaisesRegex(ValueError, r"camera_fov must be in \(0, pi\)"):
                SensorCamera.compute_camera_rays_pinhole(width, height, camera_fov=bad_fov, device="cpu")
        with self.assertRaisesRegex(ValueError, "must be positive"):
            SensorCamera.compute_camera_rays_pinhole(
                width, height, focal_length=0.0, horizontal_aperture=2.0, vertical_aperture=2.0, device="cpu"
            )

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
            camera.update(state, camera_transforms, wp.zeros((height, width, 2, 3), dtype=wp.vec3f, device="cpu"))
        with self.assertRaisesRegex(ValueError, "camera_rays must have shape"):
            camera.update(state, camera_transforms, rays.reshape((height, width, 2)))
        with self.assertRaises(TypeError):
            camera.update(state, camera_transforms, np.zeros((height, width, 2), dtype=np.float32))

    def _resolve_multisampled_against_single_sample(self, anti_aliasing) -> None:
        """Render a one-hit/one-miss bundle and compare it to the single-ray baseline.

        The bundle's first sample hits the sphere and its second sample has a zero
        direction (a forced miss), so SSAA and MSAA both blend the sphere against the
        clear color while keeping nearest-hit depth, normal, and shape index.
        """
        model, camera = self._build_sphere_scene()
        state = model.state()
        transforms = self._identity_transforms(1)
        single_sample_rays = self._rays(1, 1)
        ray_values = single_sample_rays.numpy()
        multisample_values = np.zeros((1, 1, 2, 2, 3), dtype=np.float32)
        multisample_values[:, :, 0] = ray_values[:, :, 0]
        multisample_rays = wp.array(multisample_values, dtype=wp.vec3f, device="cpu")

        single_color = camera.create_color_image_output(1, 1, 1)
        multisample_color = camera.create_color_image_output(1, 1, 1)
        single_depth = camera.create_depth_image_output(1, 1, 1)
        multisample_depth = camera.create_depth_image_output(1, 1, 1)
        single_forward_depth = camera.create_forward_depth_image_output(1, 1, 1)
        multisample_forward_depth = camera.create_forward_depth_image_output(1, 1, 1)
        single_normal = camera.create_normal_image_output(1, 1, 1)
        multisample_normal = camera.create_normal_image_output(1, 1, 1)
        single_shape_index = camera.create_shape_index_image_output(1, 1, 1)
        multisample_shape_index = camera.create_shape_index_image_output(1, 1, 1)
        camera.update(
            state,
            transforms,
            single_sample_rays,
            color_image=single_color,
            depth_image=single_depth,
            forward_depth_image=single_forward_depth,
            normal_image=single_normal,
            shape_index_image=single_shape_index,
        )
        camera.update(
            state,
            transforms,
            multisample_rays,
            color_image=multisample_color,
            depth_image=multisample_depth,
            forward_depth_image=multisample_forward_depth,
            normal_image=multisample_normal,
            shape_index_image=multisample_shape_index,
            render_config=camera.RenderConfig(anti_aliasing=anti_aliasing),
        )

        self.assertNotEqual(int(multisample_color.numpy()[0, 0, 0]), int(single_color.numpy()[0, 0, 0]))
        np.testing.assert_allclose(multisample_depth.numpy(), single_depth.numpy(), atol=1.0e-6)
        np.testing.assert_allclose(multisample_forward_depth.numpy(), single_forward_depth.numpy(), atol=1.0e-6)
        np.testing.assert_allclose(multisample_normal.numpy(), single_normal.numpy(), atol=1.0e-6)
        np.testing.assert_array_equal(multisample_shape_index.numpy(), single_shape_index.numpy())

    def test_update_resolves_ssaa_rays(self) -> None:
        """Supersample color across all pixel rays and retain nearest-hit geometry."""
        self._resolve_multisampled_against_single_sample(SensorCamera.AntiAliasing.SSAA)

    def test_update_resolves_msaa_rays(self) -> None:
        """Multisample coverage while shading only the nearest hit once."""
        self._resolve_multisampled_against_single_sample(SensorCamera.AntiAliasing.MSAA)

    def test_update_rejects_invalid_anti_aliasing(self) -> None:
        """Reject an unknown resolve mode before rendering the output."""
        model, camera = self._build_sphere_scene()
        color = camera.create_color_image_output(1, 1, 1)
        with self.assertRaisesRegex(ValueError, "Invalid anti_aliasing mode"):
            camera.update(
                model.state(),
                self._identity_transforms(1),
                self._rays(1, 1),
                color_image=color,
                render_config=camera.RenderConfig(anti_aliasing=99),
            )

    def test_msaa_blends_overlapping_objects_without_background(self) -> None:
        """Blend two objects at an edge instead of bleeding the background.

        When a pixel's subsamples land on two different objects (never the background),
        MSAA must shade both surfaces and composite them - matching SSAA and staying
        fully opaque - rather than painting the nearest object or blending toward the
        clear color.
        """
        builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
        red_body = builder.add_body(xform=wp.transform(p=wp.vec3(-0.3, 0.0, -2.0), q=wp.quat_identity()))
        builder.add_shape_sphere(red_body, radius=0.5, color=(1.0, 0.0, 0.0))
        blue_body = builder.add_body(xform=wp.transform(p=wp.vec3(0.3, 0.0, -2.0), q=wp.quat_identity()))
        builder.add_shape_sphere(blue_body, radius=0.5, color=(0.0, 0.0, 1.0))
        model = builder.finalize(device="cpu")
        camera = SensorCamera(model)
        state = model.state()
        transforms = self._identity_transforms(1)

        dir_red = np.array([-0.3, 0.0, -2.0], dtype=np.float32)
        dir_red /= np.linalg.norm(dir_red)
        dir_blue = np.array([0.3, 0.0, -2.0], dtype=np.float32)
        dir_blue /= np.linalg.norm(dir_blue)

        def _single_ray(direction: np.ndarray) -> wp.array4d[wp.vec3f]:
            values = np.zeros((1, 1, 1, 2, 3), dtype=np.float32)
            values[0, 0, 0, 1] = direction
            return wp.array(values, dtype=wp.vec3f, device="cpu")

        bundle_values = np.zeros((1, 1, 2, 2, 3), dtype=np.float32)
        bundle_values[0, 0, 0, 1] = dir_red
        bundle_values[0, 0, 1, 1] = dir_blue
        bundle_rays = wp.array(bundle_values, dtype=wp.vec3f, device="cpu")

        red_color = camera.create_color_image_output(1, 1, 1)
        blue_color = camera.create_color_image_output(1, 1, 1)
        msaa_color = camera.create_color_image_output(1, 1, 1)
        ssaa_color = camera.create_color_image_output(1, 1, 1)
        camera.update(state, transforms, _single_ray(dir_red), color_image=red_color)
        camera.update(state, transforms, _single_ray(dir_blue), color_image=blue_color)
        camera.update(
            state,
            transforms,
            bundle_rays,
            color_image=msaa_color,
            render_config=camera.RenderConfig(anti_aliasing=SensorCamera.AntiAliasing.MSAA),
        )
        camera.update(
            state,
            transforms,
            bundle_rays,
            color_image=ssaa_color,
            render_config=camera.RenderConfig(anti_aliasing=SensorCamera.AntiAliasing.SSAA),
        )

        red_packed = int(red_color.numpy()[0, 0, 0])
        blue_packed = int(blue_color.numpy()[0, 0, 0])
        msaa_packed = int(msaa_color.numpy()[0, 0, 0])
        # Both surfaces contribute: the blend matches neither object rendered alone.
        self.assertNotIn(msaa_packed, (red_packed, blue_packed))
        # Every subsample hit geometry, so the pixel stays fully opaque - no background bleed.
        self.assertEqual((msaa_packed >> 24) & 0xFF, 255)
        # Shading each surface once yields the same composite as supersampling here.
        self.assertEqual(msaa_packed, int(ssaa_color.numpy()[0, 0, 0]))
        # The red and blue channels are both present in the blend.
        self.assertGreater(msaa_packed & 0xFF, 0)
        self.assertGreater((msaa_packed >> 16) & 0xFF, 0)

    def test_msaa_shades_fully_covered_shape_at_center(self) -> None:
        """Use the center ray first when generated rays all hit one surface."""
        builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
        body = builder.add_body(xform=wp.transform(p=wp.vec3(0.0, 0.0, -4.0), q=wp.quat_identity()))
        builder.add_shape_sphere(body, radius=2.0, color=(1.0, 0.5, 0.2))
        model = builder.finalize(device="cpu")
        camera = SensorCamera(model)
        camera.create_default_light()
        state = model.state()
        transforms = self._identity_transforms(1)
        center_rays = SensorCamera.compute_camera_rays_pinhole(1, 1, camera_fov=0.8, device="cpu")
        center_hdr = camera.create_hdr_color_image_output(1, 1, 1)
        camera.update(state, transforms, center_rays, hdr_color_image=center_hdr)

        for sample_count in (3, 4, 8):
            with self.subTest(sample_count=sample_count):
                rays = SensorCamera.compute_camera_rays_pinhole(
                    1, 1, camera_fov=0.8, sample_count=sample_count, device="cpu"
                )
                msaa_hdr = camera.create_hdr_color_image_output(1, 1, 1)
                camera.update(
                    state,
                    transforms,
                    rays,
                    hdr_color_image=msaa_hdr,
                    render_config=camera.RenderConfig(anti_aliasing=camera.AntiAliasing.MSAA),
                )
                np.testing.assert_allclose(msaa_hdr.numpy(), center_hdr.numpy(), atol=1.0e-6)

    def test_msaa_shades_distinct_particles_separately(self) -> None:
        """Keep shading from adjacent particles independent despite their shared hit ID."""
        builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
        for x in (-0.3, 0.3):
            builder.add_particle(pos=wp.vec3(x, 0.0, -2.0), vel=wp.vec3(0.0), mass=1.0, radius=0.25)
        model = builder.finalize(device="cpu")
        camera = SensorCamera(model)
        camera.create_default_light()
        ray_values = np.zeros((1, 1, 2, 2, 3), dtype=np.float32)
        for sample_index, x in enumerate((-0.3, 0.3)):
            direction = np.array([x, 0.0, -2.0], dtype=np.float32)
            ray_values[0, 0, sample_index, 1] = direction / np.linalg.norm(direction)
        rays = wp.array(ray_values, dtype=wp.vec3f, device="cpu")
        transforms = self._identity_transforms(1)
        ssaa_hdr = camera.create_hdr_color_image_output(1, 1, 1)
        msaa_hdr = camera.create_hdr_color_image_output(1, 1, 1)

        camera.update(
            model.state(),
            transforms,
            rays,
            hdr_color_image=ssaa_hdr,
            render_config=camera.RenderConfig(anti_aliasing=camera.AntiAliasing.SSAA),
        )
        camera.update(
            model.state(),
            transforms,
            rays,
            hdr_color_image=msaa_hdr,
            render_config=camera.RenderConfig(anti_aliasing=camera.AntiAliasing.MSAA),
        )

        np.testing.assert_allclose(msaa_hdr.numpy(), ssaa_hdr.numpy(), atol=1.0e-6)

    def test_msaa_reuses_shading_for_one_particle(self) -> None:
        """Shade repeated hits on one particle using the first covered sample."""
        builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
        builder.add_particle(pos=wp.vec3(0.0, 0.0, -2.0), vel=wp.vec3(0.0), mass=1.0, radius=0.75)
        model = builder.finalize(device="cpu")
        camera = SensorCamera(model)
        camera.create_default_light()
        transforms = self._identity_transforms(1)
        rays = SensorCamera.compute_camera_rays_pinhole(1, 1, camera_fov=0.8, sample_count=3, device="cpu")
        first_rays = wp.array(rays.numpy()[:, :, 0:1].copy(), dtype=wp.vec3f, device="cpu")
        first_hdr = camera.create_hdr_color_image_output(1, 1, 1)
        msaa_hdr = camera.create_hdr_color_image_output(1, 1, 1)

        camera.update(model.state(), transforms, first_rays, hdr_color_image=first_hdr)
        camera.update(
            model.state(),
            transforms,
            rays,
            hdr_color_image=msaa_hdr,
            render_config=camera.RenderConfig(anti_aliasing=camera.AntiAliasing.MSAA),
        )

        np.testing.assert_allclose(msaa_hdr.numpy(), first_hdr.numpy(), atol=1.0e-6)

    def test_msaa_reuses_shading_per_deformable_face(self) -> None:
        """Group repeated hits on one cloth face without merging different faces."""
        builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
        builder.add_cloth_mesh(
            pos=wp.vec3(0.0),
            rot=wp.quat_identity(),
            scale=1.0,
            vel=wp.vec3(0.0),
            vertices=[
                wp.vec3(-0.8, -0.3, -2.0),
                wp.vec3(-0.2, -0.3, -2.0),
                wp.vec3(-0.5, 0.5, -2.0),
                wp.vec3(0.2, -0.3, -2.0),
                wp.vec3(0.8, -0.3, -2.3),
                wp.vec3(0.5, 0.5, -2.0),
            ],
            indices=[0, 1, 2, 3, 4, 5],
            density=1.0,
        )
        model = builder.finalize(device="cpu")
        camera = SensorCamera(model)
        camera.create_default_light()
        # A nearby spotlight makes shading vary across one planar face, so this
        # test can distinguish per-face reuse from per-sample shading.
        camera._render_context._lights_type = wp.array([LightType.SPOTLIGHT], dtype=wp.int32, device="cpu")
        camera._render_context._lights_position = wp.array([wp.vec3f(-0.5, 0.0, 0.0)], dtype=wp.vec3f, device="cpu")
        camera._render_context._lights_orientation = wp.array([wp.vec3f(0.0, 0.0, -1.0)], dtype=wp.vec3f, device="cpu")
        state = model.state()
        transforms = self._identity_transforms(1)
        ray_values = np.zeros((1, 1, 3, 2, 3), dtype=np.float32)
        for sample_index, target in enumerate(((-0.65, 0.0, -2.0), (-0.45, 0.0, -2.0), (0.5, 0.0, -2.1))):
            direction = np.array(target, dtype=np.float32)
            ray_values[0, 0, sample_index, 1] = direction / np.linalg.norm(direction)
        rays = wp.array(ray_values, dtype=wp.vec3f, device="cpu")
        msaa_hdr = camera.create_hdr_color_image_output(1, 1, 1)
        single_hdr = camera.create_hdr_color_image_output(1, 1, 1)
        single_colors = []
        for sample_index in range(3):
            single_ray = wp.array(ray_values[:, :, sample_index : sample_index + 1], dtype=wp.vec3f, device="cpu")
            camera.update(state, transforms, single_ray, hdr_color_image=single_hdr)
            single_colors.append(single_hdr.numpy().copy())
        camera.update(
            state,
            transforms,
            rays,
            hdr_color_image=msaa_hdr,
            render_config=camera.RenderConfig(anti_aliasing=camera.AntiAliasing.MSAA),
        )

        self.assertGreater(np.max(np.abs(single_colors[0] - single_colors[1])), 1.0e-4)
        self.assertGreater(np.max(np.abs(single_colors[1] - single_colors[2])), 1.0e-4)
        np.testing.assert_allclose(msaa_hdr.numpy(), (2.0 * single_colors[0] + single_colors[2]) / 3.0, atol=1.0e-6)

    def test_update_rejects_multisample_rays_without_anti_aliasing(self) -> None:
        """Reject multisampled rays when the resolve mode is left at NONE."""
        model, camera = self._build_sphere_scene()
        rays = SensorCamera.compute_camera_rays_pinhole(1, 1, camera_fov=1.0, sample_count=4, device="cpu")
        color = camera.create_color_image_output(1, 1, 1)

        with self.assertRaisesRegex(ValueError, "anti_aliasing"):
            camera.update(model.state(), self._identity_transforms(1), rays, color_image=color)

    def test_update_syncs_deformables_by_default(self) -> None:
        """Verify update() syncs deformable meshes by default and skips it with sync_deformables=False."""
        width, height = 8, 6
        model, camera = self._build_sphere_scene()
        state = model.state()
        rays = self._rays(width, height)
        transforms = self._identity_transforms(model.world_count)
        depth = wp.zeros((model.world_count, height, width), dtype=wp.float32, device="cpu")

        # Spy on the internal render-context sync to observe who triggers it.
        calls = []
        real_update = camera._render_context.update
        camera._render_context.update = calls.append
        try:
            camera.update(state, transforms, rays, depth_image=depth)
            self.assertEqual(len(calls), 1, "update() must sync deformable meshes by default")

            camera.update(state, transforms, rays, depth_image=depth, sync_deformables=False)
            self.assertEqual(len(calls), 1, "sync_deformables=False must skip the sync")

            camera.sync_deformable_meshes(state)
            self.assertEqual(len(calls), 2, "sync_deformable_meshes() must sync explicitly")
        finally:
            camera._render_context.update = real_update

        # The render still produced a valid frame (rigid scene needs no sync).
        self.assertGreater(float(depth.numpy()[0, height // 2, width // 2]), 0.0)

    def test_scene_config_from_model(self) -> None:
        """Verify a model-backed SensorCamera exposes output buffers, static utils, and scene config."""
        width, height = 4, 3
        model, camera = self._build_sphere_scene()
        view_count = model.world_count

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
        color_rgba = SensorCamera.Utils.to_rgba_from_color(camera.create_color_image_output(view_count, width, height))
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

    def test_convex_hull_renders_like_its_mesh(self) -> None:
        """Verify convex-hull shapes render, matching the same geometry added as a triangle mesh.

        The box mesh has per-face vertices with normals and UVs, which the hull's collision mesh
        deduplicates, so the hull must not shade with the source mesh's per-vertex normals.
        """
        width, height = 16, 16
        depths, normals = [], []
        for convex in (True, False):
            builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
            mesh = newton.Mesh.create_box(0.2, 0.2, 0.1, compute_inertia=False)
            if convex:
                builder.add_shape_convex_hull(-1, mesh=mesh)
            else:
                builder.add_shape_mesh(-1, mesh=mesh)
            model = builder.finalize(device="cpu")
            camera = SensorCamera(model)
            # Camera 2 m above the origin looking down -Z at the box's top face (z = 0.1 m).
            above = np.array([[0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 1.0]], dtype=np.float32)
            depth = camera.create_depth_image_output(1, width, height)
            normal = camera.create_normal_image_output(1, width, height)
            camera.update(
                model.state(),
                wp.array(above, dtype=wp.transformf, device="cpu"),
                self._rays(width, height, math.radians(30.0)),
                depth_image=depth,
                normal_image=normal,
            )
            depths.append(depth.numpy()[0])
            normals.append(normal.numpy()[0])

        np.testing.assert_allclose(depths[0][height // 2, width // 2], 1.9, atol=1e-3)
        np.testing.assert_allclose(depths[0], depths[1], atol=1e-3)
        np.testing.assert_allclose(normals[0][height // 2, width // 2], (0.0, 0.0, 1.0), atol=1e-3)
        np.testing.assert_allclose(normals[0], normals[1], atol=1e-3)

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

    def test_update_clears_reserved_and_out_of_range_world_indices(self) -> None:
        """Verify -1 (reserved global) and out-of-range world indices clear, not preserve."""
        width, height = 16, 12
        model, camera = self._build_sphere_scene(world_count=2)
        state = model.state()
        rays = self._rays(width, height)
        view_count = model.world_count

        camera.default_clear_data = SensorCamera.ClearData(clear_depth=-2.0)
        # View 0: -1 is reserved for future global-world rendering (unsupported);
        # view 1: an index >= world_count. Neither is a sentinel, so both clear.
        world_indices = wp.array([-1, model.world_count], dtype=wp.int32, device="cpu")
        depth = wp.full((view_count, height, width), value=42.0, dtype=wp.float32, device="cpu")

        camera.update(
            state,
            self._identity_transforms(view_count),
            rays,
            depth_image=depth,
            world_indices=world_indices,
        )

        depth_np = depth.numpy()
        # Both views are cleared to the clear value, not left at the 42.0 prefill.
        np.testing.assert_allclose(depth_np[0], -2.0)
        np.testing.assert_allclose(depth_np[1], -2.0)

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

    def test_update_without_world_indices_maps_view_to_world(self) -> None:
        """Verify omitting world_indices renders view i into world i, with no cached mapping array."""
        width, height = 8, 6
        model, camera = self._build_sphere_scene(world_count=5)
        state = model.state()
        rays = self._rays(width, height)
        center = (height // 2, width // 2)

        # The sensor holds no default-mapping array; the renderer uses the view index.
        self.assertFalse(hasattr(camera, "_default_world_indices"))

        # Rendering different view counts (each <= world_count) works with no mapping;
        # each view renders its own world, so every view sees its sphere.
        for view_count in (3, 5, 2, 4):
            depth = wp.zeros((view_count, height, width), dtype=wp.float32, device="cpu")
            camera.update(state, self._identity_transforms(view_count), rays, depth_image=depth)
            self.assertTrue(all(float(depth.numpy()[v][center]) > 0.0 for v in range(view_count)))

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

    def test_update_rejects_default_world_indices_exceeding_world_count(self) -> None:
        """Verify the default identity mapping is rejected when there are more views than worlds."""
        width, height = 8, 6
        model, camera = self._build_sphere_scene()  # 1 world
        state = model.state()
        rays = self._rays(width, height)
        # Two views on a one-world model with no explicit mapping would index world 1.
        camera_transforms = self._identity_transforms(2)
        depth = wp.zeros((2, height, width), dtype=wp.float32, device="cpu")
        with self.assertRaisesRegex(ValueError, "exceeds model.world_count"):
            camera.update(state, camera_transforms, rays, depth_image=depth)

        # An explicit mapping to the valid world renders both views.
        world_indices = wp.array(np.zeros(2, dtype=np.int32), dtype=wp.int32, device="cpu")
        camera.update(state, camera_transforms, rays, depth_image=depth, world_indices=world_indices)
        center = (height // 2, width // 2)
        self.assertTrue(all(float(depth.numpy()[v][center]) > 0.0 for v in range(2)))

    @staticmethod
    def _cloth_color_scene(device):
        """Build two overlapping worlds with distinct face colors and views from both sides."""
        colors = np.array([[0.8, 0.1, 0.2], [0.1, 0.6, 0.3], [0.2, 0.3, 0.7], [0.6, 0.4, 0.1]], dtype=np.float32)
        builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
        for world in range(2):
            cloth = newton.ModelBuilder(up_axis=newton.Axis.Z)
            cloth.add_cloth_mesh(
                pos=wp.vec3(0.0, 0.0, 1.0),
                rot=wp.quat_identity(),
                scale=1.0,
                vel=wp.vec3(0.0),
                vertices=[(x + dx, y, 0.0) for x in (-0.5, 0.5) for dx, y in ((-0.3, -0.3), (0.3, -0.3), (0.0, 0.6))],
                indices=[0, 1, 2, 3, 4, 5],
                density=1.0,
                color=colors[2 * world : 2 * world + 2],
            )
            builder.add_world(cloth)
        model = builder.finalize(device=device)
        camera = SensorCamera(model)
        # Orthographic rays hit the two triangle interiors, away from shared edges.
        rays = np.zeros((1, 2, 1, 2, 3), dtype=np.float32)
        rays[0, :, 0, 0, 0] = (-0.5, 0.5)
        rays[0, :, 0, 1, 2] = -1.0
        rays = wp.array(rays, dtype=wp.vec3f, device=device)
        flip = wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), math.pi)
        poses = wp.array(
            [wp.transform((0.0, 0.0, z), q) for z, q in ((3.0, wp.quat_identity()), (-1.0, flip)) for _ in range(2)],
            dtype=wp.transformf,
            device=device,
        )
        world_indices = wp.array([1, 0, 1, 0], dtype=wp.int32, device=device)
        expected = colors.reshape(2, 1, 2, 3)[[1, 0, 1, 0]]
        return model, camera, rays, poses, world_indices, expected

    @staticmethod
    def _unpack_rgb(image):
        """Unpack RGB channels from a packed RGBA image."""
        packed = image.numpy()
        return np.stack([(packed >> shift) & 0xFF for shift in (0, 8, 16)], axis=-1)

    def test_cloth_albedo_uses_per_face_colors(self) -> None:
        """Verify albedo-only renders select each face's color across worlds and viewing sides."""
        for device in get_test_devices():
            model, camera, rays, poses, worlds, expected_srgb = self._cloth_color_scene(device)
            state = model.state()
            albedo = camera.create_albedo_image_output(4, 2, 1)
            for color_space in (newton.utils.ColorSpace.SRGB, newton.utils.ColorSpace.LINEAR):
                expected = expected_srgb
                if color_space == newton.utils.ColorSpace.LINEAR:
                    expected = np.apply_along_axis(newton.utils.color_srgb_to_linear, -1, expected)
                for textures in (False, True):
                    with self.subTest(device=device, color_space=color_space, textures=textures):
                        camera.update(
                            state,
                            poses,
                            rays,
                            world_indices=worlds,
                            albedo_image=albedo,
                            render_config=SensorCamera.RenderConfig(
                                enable_textures=textures,
                                output_color_space=color_space,
                                enable_backface_culling=False,
                            ),
                        )
                        np.testing.assert_allclose(self._unpack_rgb(albedo), expected * 255.0, atol=2)

    def test_cloth_renders_triangle_colors_from_both_sides(self) -> None:
        """Verify two-sided depth, camera-facing normals, and per-face color under directional light."""
        for device in get_test_devices():
            model, camera, rays, poses, worlds, expected = self._cloth_color_scene(device)
            state = model.state()
            depth = camera.create_depth_image_output(4, 2, 1)
            albedo = camera.create_albedo_image_output(4, 2, 1)
            normal = camera.create_normal_image_output(4, 2, 1)
            color = camera.create_color_image_output(4, 2, 1)
            hdr = camera.create_hdr_color_image_output(4, 2, 1)
            with self.subTest(device=device, output="depth-only"):
                camera.update(
                    state,
                    poses,
                    rays,
                    world_indices=worlds,
                    depth_image=depth,
                    render_config=SensorCamera.RenderConfig(enable_backface_culling=False),
                )
                np.testing.assert_allclose(depth.numpy(), 2.0, atol=1e-5)

            for light_direction in (-1.0, 1.0):
                camera.create_default_light(enable_shadows=False, direction=wp.vec3(0.0, 0.0, light_direction))
                with self.subTest(device=device, light_direction=light_direction):
                    camera.update(
                        state,
                        poses,
                        rays,
                        world_indices=worlds,
                        albedo_image=albedo,
                        normal_image=normal,
                        color_image=color,
                        hdr_color_image=hdr,
                        render_config=SensorCamera.RenderConfig(
                            enable_ambient_lighting=False, enable_backface_culling=False
                        ),
                    )
                    np.testing.assert_allclose(self._unpack_rgb(albedo), expected * 255.0, atol=2)
                    expected_normals = np.zeros((4, 1, 2, 3), dtype=np.float32)
                    expected_normals[:2, :, :, 2] = 1.0
                    expected_normals[2:, :, :, 2] = -1.0
                    np.testing.assert_allclose(normal.numpy(), expected_normals, atol=1e-5)
                    lit = (expected_normals[..., 2:] * light_direction < 0.0).astype(np.float32)
                    np.testing.assert_allclose(self._unpack_rgb(color), expected * lit * 255.0, atol=2)
                    np.testing.assert_allclose(
                        hdr.numpy(),
                        np.apply_along_axis(newton.utils.color_srgb_to_linear, -1, expected) * lit,
                        atol=1e-5,
                    )

    def test_cloth_respects_backface_culling(self) -> None:
        """Honor default and explicit culling settings in both triangle intersection paths."""
        for device in get_test_devices():
            model, camera, rays, poses, worlds, _ = self._cloth_color_scene(device)
            state = model.state()
            depth = camera.create_depth_image_output(4, 2, 1)
            normal = camera.create_normal_image_output(4, 2, 1)
            for render_normals in (False, True):
                for culling in (None, False, True):
                    with self.subTest(device=device, render_normals=render_normals, culling=culling):
                        config = None if culling is None else SensorCamera.RenderConfig(enable_backface_culling=culling)
                        camera.update(
                            state,
                            poses,
                            rays,
                            world_indices=worlds,
                            depth_image=depth,
                            normal_image=normal if render_normals else None,
                            render_config=config,
                        )
                        expected_depth = np.full((4, 1, 2), 2.0)
                        if culling is not False:
                            expected_depth[2:] = 0.0
                        np.testing.assert_allclose(depth.numpy(), expected_depth, atol=1e-5)

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

    def test_in_memory_rgb_and_grayscale_textures(self) -> None:
        """Verify meshes with in-memory RGB ``(H, W, 3)`` or grayscale ``(H, W)`` textures render opaque."""
        width, height = 8, 8
        for name, texture, expected in (
            ("rgb", np.tile(np.array([200, 40, 10], dtype=np.uint8), (4, 4, 1)), (200, 40, 10)),
            ("gray", np.full((4, 4), 90, dtype=np.uint8), (90, 90, 90)),
        ):
            with self.subTest(texture=name):
                mesh = newton.Mesh(
                    np.array([[-1, -1, 0], [1, -1, 0], [1, 1, 0], [-1, 1, 0]], dtype=np.float32),
                    np.array([0, 1, 2, 0, 2, 3], dtype=np.int32),
                    uvs=np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float32),
                    compute_inertia=False,
                    texture=texture,
                )
                builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
                builder.add_shape_mesh(-1, mesh=mesh, color=(1.0, 1.0, 1.0))
                model = builder.finalize(device="cpu")
                camera = SensorCamera(model, default_render_config=SensorCamera.RenderConfig(enable_textures=True))
                above = np.array([[0.0, 0.0, 1.8, 0.0, 0.0, 0.0, 1.0]], dtype=np.float32)
                albedo = camera.create_albedo_image_output(1, width, height)
                camera.update(
                    model.state(),
                    wp.array(above, dtype=wp.transformf, device="cpu"),
                    self._rays(width, height, math.radians(60.0)),
                    albedo_image=albedo,
                )
                packed = int(albedo.numpy()[0, height // 2, width // 2])
                rgb = np.array([packed & 0xFF, (packed >> 8) & 0xFF, (packed >> 16) & 0xFF])
                np.testing.assert_allclose(rgb, expected, atol=2)

    def test_mesh_texture_transform_maps_uvs(self) -> None:
        """Verify ``Mesh.texture_transform`` is applied to mesh UVs, as in the viewers."""
        width, height = 16, 16
        # Red for u < 0.5 and green for u >= 0.5.
        texture = np.zeros((8, 16, 4), dtype=np.uint8)
        texture[..., 3] = 255
        texture[:, :8, 0] = 255
        texture[:, 8:, 1] = 255

        def albedo_at_u_quarter(texture_transform) -> np.ndarray:
            mesh = newton.Mesh(
                np.array([[-1, -1, 0], [1, -1, 0], [1, 1, 0], [-1, 1, 0]], dtype=np.float32),
                np.array([0, 1, 2, 0, 2, 3], dtype=np.int32),
                uvs=np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float32),
                compute_inertia=False,
                texture=texture,
                texture_transform=texture_transform,
            )
            builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
            builder.add_shape_mesh(-1, mesh=mesh, color=(1.0, 1.0, 1.0))
            model = builder.finalize(device="cpu")
            camera = SensorCamera(model, default_render_config=SensorCamera.RenderConfig(enable_textures=True))
            # The quad fills the view of a camera 1.8 m above it; pixel column 4 sees u = 0.25.
            above = np.array([[0.0, 0.0, 1.8, 0.0, 0.0, 0.0, 1.0]], dtype=np.float32)
            albedo = camera.create_albedo_image_output(1, width, height)
            camera.update(
                model.state(),
                wp.array(above, dtype=wp.transformf, device="cpu"),
                self._rays(width, height, math.radians(60.0)),
                albedo_image=albedo,
            )
            packed = int(albedo.numpy()[0, height // 2, 4])
            return np.array([packed & 0xFF, (packed >> 8) & 0xFF, (packed >> 16) & 0xFF])

        identity = albedo_at_u_quarter(((1.0, 0.0, 0.0), (0.0, 1.0, 0.0)))
        shifted = albedo_at_u_quarter(((1.0, 0.0, 0.5), (0.0, 1.0, 0.0)))
        np.testing.assert_allclose(identity, (255, 0, 0), atol=2)
        np.testing.assert_allclose(shifted, (0, 255, 0), atol=2)

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

        for rgba in (
            SensorCamera.Utils.to_rgba_from_color(color),
            SensorCamera.Utils.to_rgba_from_depth(depth, depth_range=(0.0, 10.0)),
            SensorCamera.Utils.to_rgba_from_normal(normal),
            SensorCamera.Utils.to_rgba_from_shape_index(shape_index),
        ):
            self.assertEqual(rgba.shape, (view_count, height, width, 4))
            self.assertEqual(rgba.dtype, wp.uint8)

    def test_utils_postprocessing_helpers(self) -> None:
        """Verify forward-depth conversion, normal/depth flatten, palette colorize, and depth-range branches."""
        width, height, views_per_row = 6, 4, 2
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

        center = (0, height // 2, width // 2)
        self.assertGreater(float(depth.numpy()[center]), 0.0)

        # Ray-distance depth -> forward (planar) depth; must not exceed ray depth.
        forward = SensorCamera.Utils.convert_ray_depth_to_forward_depth(depth, camera_transforms, rays)
        self.assertEqual(forward.shape, depth.shape)
        self.assertEqual(forward.dtype, wp.float32)
        self.assertLessEqual(float(forward.numpy()[center]), float(depth.numpy()[center]) + 1.0e-4)

        # Flatten normal/depth into one tiled (rows*H, cols*W, 4) grid buffer.
        views_per_col = -(-view_count // views_per_row)
        for flat in (
            SensorCamera.Utils.flatten_normal_image_to_rgba(normal, views_per_row=views_per_row),
            SensorCamera.Utils.flatten_depth_image_to_rgba(depth, views_per_row=views_per_row),
        ):
            self.assertEqual(flat.shape, (views_per_col * height, views_per_row * width, 4))
            self.assertEqual(flat.dtype, wp.uint8)

        # Shape-index colorized via a caller palette (out-of-range indices -> black).
        palette = wp.array(np.array([[10, 20, 30]], dtype=np.uint8), dtype=wp.uint8, device="cpu")
        colored = SensorCamera.Utils.to_rgba_from_shape_index(shape_index, colors=palette)
        self.assertEqual(colored.shape, (view_count, height, width, 4))

        # to_rgba_from_depth: on-device auto range (depth_range=None) and the near<far guard.
        auto = SensorCamera.Utils.to_rgba_from_depth(depth)
        self.assertEqual(auto.shape, (view_count, height, width, 4))
        with self.assertRaisesRegex(ValueError, "near < far"):
            SensorCamera.Utils.to_rgba_from_depth(depth, depth_range=(5.0, 1.0))

    def test_utils_shape_index_hash_colors_differ_by_index(self) -> None:
        """Verify the shape-index hash palette maps two distinct valid indices to distinct colors."""
        # One view, one row, two pixels holding shape indices 0 and 1.
        shape_index = wp.array(np.array([[[0, 1]]], dtype=np.uint32), dtype=wp.uint32, device="cpu")
        rgba = SensorCamera.Utils.to_rgba_from_shape_index(shape_index).numpy()
        color_0 = tuple(int(c) for c in rgba[0, 0, 0, :3])
        color_1 = tuple(int(c) for c in rgba[0, 0, 1, :3])
        self.assertNotEqual(color_0, color_1)

    def test_utils_flatten_rejects_views_per_row_below_one(self) -> None:
        """Verify the flatten helpers reject a non-positive ``views_per_row``."""
        width, height = 4, 3
        model, camera = self._build_sphere_scene()
        state = model.state()
        rays = self._rays(width, height)
        view_count = model.world_count
        color = camera.create_color_image_output(view_count, width, height)
        camera.update(state, self._identity_transforms(view_count), rays, color_image=color)
        with self.assertRaisesRegex(ValueError, "views_per_row"):
            SensorCamera.Utils.flatten_color_image_to_rgba(color, views_per_row=0)


if __name__ == "__main__":
    unittest.main()
