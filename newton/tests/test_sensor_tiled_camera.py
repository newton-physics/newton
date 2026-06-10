# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import math
import os
import unittest

import numpy as np
import warp as wp

import newton
from newton.sensors import SensorTiledCamera


class TestSensorTiledCamera(unittest.TestCase):
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
        self.assertEqual(test_image.size, gold_image.size, "Images have different data shapes")

        gold_image = gold_image.reshape(test_image.shape)

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
    def _build_single_particle_scene() -> newton.Model:
        builder = newton.ModelBuilder()
        builder.add_particle(pos=wp.vec3(0.0), vel=wp.vec3(0.0), mass=1.0, radius=0.1)
        return builder.finalize(device="cpu")

    @staticmethod
    def _build_global_textured_quad_scene() -> tuple[newton.Model, int]:
        builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
        for _ in range(2):
            builder.begin_world()
            builder.end_world()

        vertices = np.array(
            [
                [-1.0, -1.0, -2.0],
                [1.0, -1.0, -2.0],
                [1.0, 1.0, -2.0],
                [-1.0, 1.0, -2.0],
            ],
            dtype=np.float32,
        )
        indices = np.array([0, 1, 2, 0, 2, 3], dtype=np.int32)
        uvs = np.array(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [1.0, 1.0],
                [0.0, 1.0],
            ],
            dtype=np.float32,
        )

        red = np.full((2, 2, 4), (255, 0, 0, 255), dtype=np.uint8)
        blue = np.full((2, 2, 4), (0, 0, 255, 255), dtype=np.uint8)

        visible_mesh = newton.Mesh(vertices, indices, uvs=uvs, compute_inertia=False, texture=red)
        hidden_mesh = newton.Mesh(
            vertices + np.array([100.0, 0.0, 0.0], dtype=np.float32),
            indices,
            uvs=uvs,
            compute_inertia=False,
            texture=blue,
        )
        visible_shape_index = builder.add_shape_mesh(-1, mesh=visible_mesh, color=(1.0, 1.0, 1.0))
        builder.add_shape_mesh(-1, mesh=hidden_mesh, color=(1.0, 1.0, 1.0))
        return builder.finalize(device="cpu"), visible_shape_index

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

    def test_per_world_shape_texture_ids_render_shared_shape_differently(self) -> None:
        model, visible_shape_index = self._build_global_textured_quad_scene()
        sensor = SensorTiledCamera(
            model=model,
            config=SensorTiledCamera.RenderConfig(enable_textures=True, enable_ambient_lighting=False),
        )
        texture_ids = np.full((2, model.shape_count), -1, dtype=np.int32)
        texture_ids[:, visible_shape_index] = [0, 1]
        sensor.set_shape_texture_ids(texture_ids, per_world=True)

        camera_transforms = wp.array(
            [
                [
                    wp.transformf(wp.vec3f(0.0), wp.quatf(0.0, 0.0, 0.0, 1.0)),
                    wp.transformf(wp.vec3f(0.0), wp.quatf(0.0, 0.0, 0.0, 1.0)),
                ]
            ],
            dtype=wp.transformf,
            device="cpu",
        )
        camera_rays = sensor.utils.compute_pinhole_camera_rays(3, 3, math.radians(30.0))
        albedo_image = sensor.utils.create_albedo_image_output(3, 3, camera_count=1)
        shape_index_image = sensor.utils.create_shape_index_image_output(3, 3, camera_count=1)

        sensor.update(
            model.state(),
            camera_transforms,
            camera_rays,
            albedo_image=albedo_image,
            shape_index_image=shape_index_image,
        )

        albedo = albedo_image.numpy()
        shape_index = shape_index_image.numpy()
        self.assertEqual(shape_index[0, 0, 1, 1], visible_shape_index)
        self.assertEqual(shape_index[1, 0, 1, 1], visible_shape_index)
        np.testing.assert_allclose(self._unpack_rgba(albedo[0, 0, 1, 1])[:3], [255, 0, 0], atol=1)
        np.testing.assert_allclose(self._unpack_rgba(albedo[1, 0, 1, 1])[:3], [0, 0, 255], atol=1)

    def test_set_shape_texture_ids_reuses_same_layout_allocation(self) -> None:
        model, visible_shape_index = self._build_global_textured_quad_scene()
        sensor = SensorTiledCamera(model=model, config=SensorTiledCamera.RenderConfig(enable_textures=True))
        texture_ids = np.full((2, model.shape_count), -1, dtype=np.int32)
        texture_ids[:, visible_shape_index] = [0, 1]
        sensor.set_shape_texture_ids(texture_ids, per_world=True)

        with self.assertWarns(DeprecationWarning):
            render_context = sensor.render_context

        ptr = render_context.shape_texture_ids.ptr
        texture_ids[:, visible_shape_index] = [1, 0]
        sensor.set_shape_texture_ids(texture_ids, per_world=True)

        self.assertTrue(render_context.shape_texture_ids_per_world)
        self.assertEqual(render_context.shape_texture_ids.ptr, ptr)
        np.testing.assert_array_equal(render_context.shape_texture_ids.numpy(), texture_ids.reshape(-1))

    def test_render_config_uses_utils_color_space_enum(self) -> None:
        self.assertEqual(SensorTiledCamera.RenderConfig().output_color_space, newton.utils.ColorSpace.SRGB)
        config = SensorTiledCamera.RenderConfig(output_color_space=newton.utils.ColorSpace.LINEAR)
        self.assertEqual(config.output_color_space, newton.utils.ColorSpace.LINEAR)

        linear = newton.utils.color_srgb_to_linear((0.5, 0.25, 0.1))
        np.testing.assert_allclose(newton.utils.color_linear_to_srgb(linear), (0.5, 0.25, 0.1), atol=1e-6)

    def test_albedo_output_follows_output_color_space(self) -> None:
        color = (0.25, 0.5, 0.75)
        model = self._build_single_sphere_scene(color)
        camera_transforms = wp.array(
            [[wp.transformf(wp.vec3f(0.0), wp.quatf(0.0, 0.0, 0.0, 1.0))]],
            dtype=wp.transformf,
            device="cpu",
        )
        state = model.state()

        for output_color_space in (newton.utils.ColorSpace.SRGB, newton.utils.ColorSpace.LINEAR):
            sensor = SensorTiledCamera(
                model=model,
                config=SensorTiledCamera.RenderConfig(output_color_space=output_color_space),
            )
            camera_rays = sensor.utils.compute_pinhole_camera_rays(1, 1, math.radians(30.0))
            albedo_image = sensor.utils.create_albedo_image_output(1, 1, camera_count=1)

            sensor.update(state, camera_transforms, camera_rays, albedo_image=albedo_image)

            packed = self._unpack_rgba(albedo_image.numpy()[0, 0, 0, 0])
            expected_rgb = (
                np.array([63, 127, 191], dtype=np.uint8)
                if output_color_space == newton.utils.ColorSpace.SRGB
                else np.array([12, 54, 133], dtype=np.uint8)
            )
            np.testing.assert_array_equal(packed[:3], expected_rgb)
            self.assertEqual(packed[3], 255)

    @unittest.skipUnless(wp.is_cuda_available(), "Requires CUDA")
    def test_golden_image(self):
        model = self._shared_model

        width = 320
        height = 240
        camera_count = 1

        camera_transforms = wp.array(
            [[wp.transformf(wp.vec3f(10.0, 0.0, 2.0), wp.quatf(0.5, 0.5, 0.5, 0.5))]], dtype=wp.transformf
        )

        tiled_camera_sensor = SensorTiledCamera(model=model)
        tiled_camera_sensor.utils.create_default_light(enable_shadows=True)
        tiled_camera_sensor.utils.assign_checkerboard_material_to_all_shapes()

        camera_rays = tiled_camera_sensor.utils.compute_pinhole_camera_rays(width, height, math.radians(45.0))
        color_image = tiled_camera_sensor.utils.create_color_image_output(width, height, camera_count)
        depth_image = tiled_camera_sensor.utils.create_depth_image_output(width, height, camera_count)

        state = model.state()
        tiled_camera_sensor.update(
            state, camera_transforms, camera_rays, color_image=color_image, depth_image=depth_image
        )

        golden_color_data = np.load(
            os.path.join(os.path.dirname(__file__), "golden_data", "test_sensor_tiled_camera", "color.npy")
        )
        golden_depth_data = np.load(
            os.path.join(os.path.dirname(__file__), "golden_data", "test_sensor_tiled_camera", "depth.npy")
        )

        self.__compare_images(color_image.numpy(), golden_color_data, allowed_difference=0.1)
        self.__compare_images(depth_image.numpy(), golden_depth_data, allowed_difference=0.1)

    @unittest.skipUnless(wp.is_cuda_available(), "Requires CUDA")
    def test_output_image_parameters(self):
        model = self._shared_model

        width = 640
        height = 480
        camera_count = 1

        camera_transforms = wp.array(
            [[wp.transformf(wp.vec3f(10.0, 0.0, 2.0), wp.quatf(0.5, 0.5, 0.5, 0.5))]], dtype=wp.transformf
        )

        tiled_camera_sensor = SensorTiledCamera(model=model)
        camera_rays = tiled_camera_sensor.utils.compute_pinhole_camera_rays(width, height, math.radians(45.0))

        state = model.state()

        color_image = tiled_camera_sensor.utils.create_color_image_output(width, height, camera_count)
        depth_image = tiled_camera_sensor.utils.create_depth_image_output(width, height, camera_count)
        tiled_camera_sensor.update(
            state, camera_transforms, camera_rays, color_image=color_image, depth_image=depth_image
        )
        self.assertTrue(np.any(color_image.numpy() != 0), "Color image should contain rendered data")
        self.assertTrue(np.any(depth_image.numpy() != 0), "Depth image should contain rendered data")

        color_image = tiled_camera_sensor.utils.create_color_image_output(width, height, camera_count)
        depth_image = tiled_camera_sensor.utils.create_depth_image_output(width, height, camera_count)
        tiled_camera_sensor.update(state, camera_transforms, camera_rays, color_image=color_image, depth_image=None)
        self.assertTrue(np.any(color_image.numpy() != 0), "Color image should contain rendered data")
        self.assertFalse(np.any(depth_image.numpy() != 0), "Depth image should NOT contain rendered data")

        color_image = tiled_camera_sensor.utils.create_color_image_output(width, height, camera_count)
        depth_image = tiled_camera_sensor.utils.create_depth_image_output(width, height, camera_count)
        tiled_camera_sensor.update(state, camera_transforms, camera_rays, color_image=None, depth_image=depth_image)
        self.assertFalse(np.any(color_image.numpy() != 0), "Color image should NOT contain rendered data")
        self.assertTrue(np.any(depth_image.numpy() != 0), "Depth image should contain rendered data")

        color_image = tiled_camera_sensor.utils.create_color_image_output(width, height, camera_count)
        depth_image = tiled_camera_sensor.utils.create_depth_image_output(width, height, camera_count)
        tiled_camera_sensor.update(state, camera_transforms, camera_rays, color_image=None, depth_image=None)
        self.assertFalse(np.any(color_image.numpy() != 0), "Color image should NOT contain rendered data")
        self.assertFalse(np.any(depth_image.numpy() != 0), "Depth image should NOT contain rendered data")

    def test_deprecated_geometry_bvh_helpers_forward_to_model_methods(self) -> None:
        model = self._build_single_sphere_scene((0.25, 0.5, 0.75))
        state = model.state()

        with self.assertWarns(DeprecationWarning):
            newton.geometry.build_bvh_shape(model, state, bvh_constructor="median")
        self.assertIsNotNone(model.bvh_shapes)

        with self.assertWarns(DeprecationWarning):
            newton.geometry.refit_bvh_shape(model, state)

        particle_model = self._build_single_particle_scene()
        particle_state = particle_model.state()

        with self.assertWarns(DeprecationWarning):
            newton.geometry.build_bvh_particle(particle_model, particle_state, bvh_constructor="median")
        self.assertIsNotNone(particle_model.bvh_particles)

        with self.assertWarns(DeprecationWarning):
            newton.geometry.refit_bvh_particle(particle_model, particle_state)

    def test_model_bvh_build_accepts_constructor(self) -> None:
        model = self._build_single_sphere_scene((0.25, 0.5, 0.75))
        state = model.state()

        model.bvh_build_shapes(state, bvh_constructor="median")
        self.assertIsNotNone(model.bvh_shapes)

        particle_model = self._build_single_particle_scene()
        particle_state = particle_model.state()

        particle_model.bvh_build_particles(particle_state, bvh_constructor="median")
        self.assertIsNotNone(particle_model.bvh_particles)

    def test_model_bvhs_are_built_by_finalize_and_refit(self) -> None:
        model = self._build_single_sphere_scene((0.25, 0.5, 0.75))
        state = model.state()

        self.assertIsNotNone(model.bvh_shapes)
        model.bvh_refit_shapes(state)

        particle_model = self._build_single_particle_scene()
        particle_state = particle_model.state()

        self.assertIsNotNone(particle_model.bvh_particles)
        particle_model.bvh_refit_particles(particle_state)


if __name__ == "__main__":
    unittest.main()
