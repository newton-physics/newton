# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import math
import os
import unittest

import numpy as np
import warp as wp

import newton
from newton.sensors import SensorBatchedCamera


class TestSensorBatchedCamera(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not wp.is_cuda_available():
            return
        cls._shared_model = cls._build_scene()

    @staticmethod
    def _build_scene():
        from pxr import Usd, UsdGeom

        builder = newton.ModelBuilder()

        # add ground plane
        builder.add_ground_plane(color=(0.91749084, 0.798277, 0.64443165))

        # SPHERE
        sphere_pos = wp.vec3(0.0, -2.0, 0.5)
        body_sphere = builder.add_body(xform=wp.transform(p=sphere_pos, q=wp.quat_identity()), label="sphere")
        builder.add_shape_sphere(body_sphere, radius=0.5, color=(0.5214758, 0.9868272, 0.79823583))

        # CAPSULE
        capsule_pos = wp.vec3(0.0, 0.0, 0.75)
        body_capsule = builder.add_body(xform=wp.transform(p=capsule_pos, q=wp.quat_identity()), label="capsule")
        builder.add_shape_capsule(body_capsule, radius=0.25, half_height=0.5, color=(0.8951316, 0.9551697, 0.8440772))

        # CYLINDER
        cylinder_pos = wp.vec3(0.0, -4.0, 0.5)
        body_cylinder = builder.add_body(xform=wp.transform(p=cylinder_pos, q=wp.quat_identity()), label="cylinder")
        builder.add_shape_cylinder(
            body_cylinder, radius=0.4, half_height=0.5, color=(0.59499574, 0.99073946, 0.64237005)
        )

        # BOX
        box_pos = wp.vec3(0.0, 2.0, 0.5)
        body_box = builder.add_body(xform=wp.transform(p=box_pos, q=wp.quat_identity()), label="box")
        builder.add_shape_box(body_box, hx=0.5, hy=0.35, hz=0.5, color=(0.8146366, 0.7905182, 0.79995614))

        # MESH (bunny)
        bunny_filename = os.path.join(os.path.dirname(__file__), "..", "examples", "assets", "bunny.usd")
        assert os.path.exists(bunny_filename), f"File not found: {bunny_filename}"
        usd_stage = Usd.Stage.Open(bunny_filename)
        usd_geom = UsdGeom.Mesh(usd_stage.GetPrimAtPath("/root/bunny"))

        mesh_vertices = np.array(usd_geom.GetPointsAttr().Get())
        mesh_indices = np.array(usd_geom.GetFaceVertexIndicesAttr().Get())

        demo_mesh = newton.Mesh(mesh_vertices, mesh_indices)

        mesh_pos = wp.vec3(0.0, 4.0, 0.0)
        body_mesh = builder.add_body(xform=wp.transform(p=mesh_pos, q=wp.quat(0.5, 0.5, 0.5, 0.5)), label="mesh")
        builder.add_shape_mesh(body_mesh, mesh=demo_mesh, color=(0.7676241, 0.99788857, 0.75097305))

        return builder.finalize()

    def __compare_images(self, test_image: np.ndarray, gold_image: np.ndarray, allowed_difference: float = 0.0):
        self.assertEqual(test_image.dtype, gold_image.dtype, "Images have different data types")
        self.assertEqual(test_image.shape, gold_image.shape, "Images have different shapes")

        if test_image.dtype == np.uint32:
            test_image = np.stack(
                [
                    test_image & np.uint32(0xFF),
                    (test_image >> np.uint32(8)) & np.uint32(0xFF),
                    (test_image >> np.uint32(16)) & np.uint32(0xFF),
                    (test_image >> np.uint32(24)) & np.uint32(0xFF),
                ],
                axis=-1,
            ).astype(np.uint8)
            gold_image = np.stack(
                [
                    gold_image & np.uint32(0xFF),
                    (gold_image >> np.uint32(8)) & np.uint32(0xFF),
                    (gold_image >> np.uint32(16)) & np.uint32(0xFF),
                    (gold_image >> np.uint32(24)) & np.uint32(0xFF),
                ],
                axis=-1,
            ).astype(np.uint8)

        # Promote to a wide type before subtracting: int64 avoids unsigned underflow for
        # integer images, float64 preserves fractional deltas for float (e.g. depth) images.
        wide_dtype = np.int64 if np.issubdtype(test_image.dtype, np.integer) else np.float64
        diff = np.abs(test_image.astype(wide_dtype) - gold_image.astype(wide_dtype))

        divider = 1.0
        if np.issubdtype(test_image.dtype, np.integer):
            divider = np.iinfo(test_image.dtype).max

        percentage_diff = float(np.average(diff)) / divider * 100.0
        self.assertLessEqual(
            percentage_diff,
            allowed_difference,
            f"Images differ more than {allowed_difference:.2f}%, total difference is {percentage_diff:.2f}%",
        )

    @staticmethod
    def _build_single_sphere_scene(color: tuple[float, float, float]) -> newton.Model:
        builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
        body = builder.add_body(xform=wp.transform(p=wp.vec3(0.0, 0.0, -2.0), q=wp.quat_identity()))
        builder.add_shape_sphere(body, radius=0.75, color=color)
        return builder.finalize(device="cpu")

    @staticmethod
    def _unpack_rgba(packed: int) -> np.ndarray:
        value = int(packed)
        return np.array(
            [
                value & 0xFF,
                (value >> 8) & 0xFF,
                (value >> 16) & 0xFF,
                (value >> 24) & 0xFF,
            ],
            dtype=np.uint8,
        )

    def test_render_config_uses_utils_color_space_enum(self) -> None:
        self.assertEqual(SensorBatchedCamera.RenderConfig().output_color_space, newton.utils.ColorSpace.SRGB)
        config = SensorBatchedCamera.RenderConfig(output_color_space=newton.utils.ColorSpace.LINEAR)
        self.assertEqual(config.output_color_space, newton.utils.ColorSpace.LINEAR)

        linear = newton.utils.color_srgb_to_linear((0.5, 0.25, 0.1))
        np.testing.assert_allclose(newton.utils.color_linear_to_srgb(linear), (0.5, 0.25, 0.1), atol=1e-6)

    def test_albedo_output_follows_output_color_space(self) -> None:
        color = (0.25, 0.5, 0.75)
        model = self._build_single_sphere_scene(color)
        camera_transforms = wp.array(
            [wp.transformf(wp.vec3f(0.0), wp.quatf(0.0, 0.0, 0.0, 1.0))],
            dtype=wp.transformf,
            device="cpu",
        )
        camera_indices = wp.array([[0, 0]], dtype=wp.int32, device="cpu")
        state = model.state()

        for output_color_space in (newton.utils.ColorSpace.SRGB, newton.utils.ColorSpace.LINEAR):
            sensor = SensorBatchedCamera(
                model=model,
                config=SensorBatchedCamera.RenderConfig(output_color_space=output_color_space),
            )
            camera_rays = sensor.utils.compute_pinhole_camera_rays(1, 1, math.radians(30.0))
            albedo_image = sensor.utils.create_albedo_image_output(1, 1, 1)

            sensor.update(state, camera_transforms, camera_rays, camera_indices, albedo_image=albedo_image)

            packed = self._unpack_rgba(albedo_image.numpy()[0, 0, 0])
            expected_rgb = (
                np.array([63, 127, 191], dtype=np.uint8)
                if output_color_space == newton.utils.ColorSpace.SRGB
                else np.array([12, 54, 133], dtype=np.uint8)
            )
            np.testing.assert_array_equal(packed[:3], expected_rgb)
            self.assertEqual(packed[3], 255)

    def test_cloth_renders_via_triangle_mesh_construction(self) -> None:
        """wp.Mesh must be lazily constructed on the first render call for cloth models.

        Cloth models have both ``particle_q`` and ``tri_indices``, which routes rendering
        through RenderContext's triangle-mesh path. The first ``update`` then constructs
        a :class:`wp.Mesh` from those geometry arrays.
        """
        # Cloth is the minimal model with both particle_q and tri_indices. During
        # init_from_model, RenderContext maps those to triangle_points and
        # triangle_indices (cloth vertices live in particle_q; tri_indices are the
        # faces). That pairing is what enables the triangle-mesh construction.
        builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
        builder.add_cloth_grid(
            pos=wp.vec3(0.0, 0.0, 0.0),
            rot=wp.quat_identity(),
            vel=wp.vec3(0.0),
            dim_x=2,
            dim_y=2,
            cell_x=0.1,
            cell_y=0.1,
            mass=0.1,
            fix_top=True,
        )
        model = builder.finalize(device="cpu")

        sensor = SensorBatchedCamera(model=model)
        # The public render_context alias was removed; this regression test needs
        # the internal mesh state that drives first-render construction.
        render_context = sensor._SensorBatchedCamera__render_context

        # init_from_model copies model.particle_q/tri_indices into triangle_points/
        # triangle_indices but does not build wp.Mesh until the first render call.
        self.assertTrue(render_context.has_triangle_mesh)
        self.assertIsNone(render_context.triangle_mesh)

        width, height = 8, 8
        camera_transforms = wp.array(
            [wp.transformf(wp.vec3f(0.0, 0.0, 0.5), wp.quatf(0.0, 0.0, 0.0, 1.0))],
            dtype=wp.transformf,
            device="cpu",
        )
        camera_indices = wp.array([[0, 0]], dtype=wp.int32, device="cpu")
        camera_rays = sensor.utils.compute_pinhole_camera_rays(width, height, math.radians(60.0))
        depth_image = sensor.utils.create_depth_image_output(1, width, height)

        sensor.update(model.state(), camera_transforms, camera_rays, camera_indices, depth_image=depth_image)

        # update() must construct the cloth triangle mesh on this first render call.
        self.assertIsNotNone(render_context.triangle_mesh)

        # Depth hits prove the mesh was passed into the render kernel, not just created.
        self.assertGreater(int(np.sum(depth_image.numpy() > 0.0)), 0)

    def test_checkerboard_material_requires_keyword_arguments(self) -> None:
        model = self._build_single_sphere_scene((0.25, 0.5, 0.75))
        sensor = SensorBatchedCamera(model=model)

        with self.assertRaises(TypeError):
            sensor.utils.assign_checkerboard_material([0])

    @unittest.skipUnless(wp.is_cuda_available(), "Requires CUDA")
    def test_golden_image(self):
        model = self._shared_model

        width = 320
        height = 240
        camera_count = 1

        camera_transforms = wp.array(
            [wp.transformf(wp.vec3f(10.0, 0.0, 2.0), wp.quatf(0.5, 0.5, 0.5, 0.5))],
            dtype=wp.transformf,
        )
        camera_indices = wp.array([[0, 0]], dtype=wp.int32)

        batched_camera_sensor = SensorBatchedCamera(model=model)
        batched_camera_sensor.utils.create_default_light(enable_shadows=True)
        batched_camera_sensor.utils.assign_checkerboard_material(
            shape_indices=np.arange(model.shape_count, dtype=np.int32)
        )

        camera_rays = batched_camera_sensor.utils.compute_pinhole_camera_rays(width, height, math.radians(45.0))
        color_image = batched_camera_sensor.utils.create_color_image_output(camera_count, width, height)
        depth_image = batched_camera_sensor.utils.create_depth_image_output(camera_count, width, height)

        state = model.state()
        batched_camera_sensor.update(
            state, camera_transforms, camera_rays, camera_indices, color_image=color_image, depth_image=depth_image
        )

        golden_color_data = np.load(
            os.path.join(os.path.dirname(__file__), "golden_data", "test_sensor_tiled_camera", "color.npy")
        )[:, 0]
        golden_depth_data = np.load(
            os.path.join(os.path.dirname(__file__), "golden_data", "test_sensor_tiled_camera", "depth.npy")
        )[:, 0]

        self.__compare_images(color_image.numpy(), golden_color_data, allowed_difference=3.0)
        self.__compare_images(depth_image.numpy(), golden_depth_data, allowed_difference=0.1)

    @unittest.skipUnless(wp.is_cuda_available(), "Requires CUDA")
    def test_deprecated_checkerboard_material_to_all_shapes_warns(self):
        model = self._shared_model
        batched_camera_sensor = SensorBatchedCamera(model=model)

        with self.assertWarnsRegex(DeprecationWarning, "assign_checkerboard_material"):
            batched_camera_sensor.utils.assign_checkerboard_material_to_all_shapes()

    @unittest.skipUnless(wp.is_cuda_available(), "Requires CUDA")
    def test_output_image_parameters(self):
        model = self._shared_model

        width = 640
        height = 480
        camera_count = 1

        camera_transforms = wp.array(
            [wp.transformf(wp.vec3f(10.0, 0.0, 2.0), wp.quatf(0.5, 0.5, 0.5, 0.5))],
            dtype=wp.transformf,
        )
        camera_indices = wp.array([[0, 0]], dtype=wp.int32)

        batched_camera_sensor = SensorBatchedCamera(model=model)
        camera_rays = batched_camera_sensor.utils.compute_pinhole_camera_rays(width, height, math.radians(45.0))

        state = model.state()

        color_image = batched_camera_sensor.utils.create_color_image_output(camera_count, width, height)
        depth_image = batched_camera_sensor.utils.create_depth_image_output(camera_count, width, height)
        batched_camera_sensor.update(
            state, camera_transforms, camera_rays, camera_indices, color_image=color_image, depth_image=depth_image
        )
        self.assertTrue(np.any(color_image.numpy() != 0), "Color image should contain rendered data")
        self.assertTrue(np.any(depth_image.numpy() != 0), "Depth image should contain rendered data")

        color_image = batched_camera_sensor.utils.create_color_image_output(camera_count, width, height)
        depth_image = batched_camera_sensor.utils.create_depth_image_output(camera_count, width, height)
        batched_camera_sensor.update(
            state, camera_transforms, camera_rays, camera_indices, color_image=color_image, depth_image=None
        )
        self.assertTrue(np.any(color_image.numpy() != 0), "Color image should contain rendered data")
        self.assertFalse(np.any(depth_image.numpy() != 0), "Depth image should NOT contain rendered data")

        color_image = batched_camera_sensor.utils.create_color_image_output(camera_count, width, height)
        depth_image = batched_camera_sensor.utils.create_depth_image_output(camera_count, width, height)
        batched_camera_sensor.update(
            state, camera_transforms, camera_rays, camera_indices, color_image=None, depth_image=depth_image
        )
        self.assertFalse(np.any(color_image.numpy() != 0), "Color image should NOT contain rendered data")
        self.assertTrue(np.any(depth_image.numpy() != 0), "Depth image should contain rendered data")

        color_image = batched_camera_sensor.utils.create_color_image_output(camera_count, width, height)
        depth_image = batched_camera_sensor.utils.create_depth_image_output(camera_count, width, height)
        batched_camera_sensor.update(
            state, camera_transforms, camera_rays, camera_indices, color_image=None, depth_image=None
        )
        self.assertFalse(np.any(color_image.numpy() != 0), "Color image should NOT contain rendered data")
        self.assertFalse(np.any(depth_image.numpy() != 0), "Depth image should NOT contain rendered data")


if __name__ == "__main__":
    unittest.main()
