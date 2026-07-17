# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import math
import os
import unittest
import warnings

import numpy as np
import warp as wp

import newton
from newton._src.sensors.warp_raytrace.raytrace import PARTICLES_SHAPE_ID, TRIANGLE_MESH_SHAPE_ID
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
    def _build_mixed_cloth_particle_scene() -> newton.Model:
        builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
        builder.add_cloth_grid(
            pos=wp.vec3(2.0, 0.0, 0.0),
            rot=wp.quat_identity(),
            vel=wp.vec3(0.0),
            dim_x=2,
            dim_y=2,
            cell_x=0.1,
            cell_y=0.1,
            mass=0.1,
            fix_top=True,
        )
        builder.add_particle(pos=wp.vec3(0.0, 0.0, -2.0), vel=wp.vec3(0.0), mass=1.0, radius=0.25)
        return builder.finalize(device="cpu")

    @staticmethod
    def _build_soft_grid_scene() -> newton.Model:
        builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
        builder.add_soft_grid(
            pos=wp.vec3(-0.1, -0.1, -2.1),
            rot=wp.quat_identity(),
            vel=wp.vec3(0.0),
            dim_x=2,
            dim_y=2,
            dim_z=2,
            cell_x=0.1,
            cell_y=0.1,
            cell_z=0.1,
            density=1000.0,
            k_mu=1.0,
            k_lambda=1.0,
            k_damp=0.0,
            particle_radius=0.25,
        )
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
        self.assertEqual(SensorTiledCamera.RenderConfig().output_color_space, newton.utils.ColorSpace.SRGB)
        config = SensorTiledCamera.RenderConfig(output_color_space=newton.utils.ColorSpace.LINEAR)
        self.assertEqual(config.output_color_space, newton.utils.ColorSpace.LINEAR)

        linear = newton.utils.color_srgb_to_linear((0.5, 0.25, 0.1))
        np.testing.assert_allclose(newton.utils.color_linear_to_srgb(linear), (0.5, 0.25, 0.1), atol=1e-6)

    def test_render_config_alias_deprecated(self) -> None:
        model = self._build_single_sphere_scene((0.25, 0.5, 0.75))
        sensor = SensorTiledCamera(model=model)

        with self.assertWarnsRegex(DeprecationWarning, "SensorTiledCamera.render_config.*default_render_config"):
            render_config = sensor.render_config

        self.assertIs(render_config, sensor.default_render_config)

    def test_constructor_config_alias_deprecated(self) -> None:
        model = self._build_single_sphere_scene((0.25, 0.5, 0.75))
        config = SensorTiledCamera.RenderConfig(output_color_space=newton.utils.ColorSpace.LINEAR)

        with self.assertWarnsRegex(DeprecationWarning, r"config=.*default_render_config"):
            sensor = SensorTiledCamera(model=model, config=config)

        self.assertIs(sensor.default_render_config, config)

    def test_constructor_config_none_warns_deprecated(self) -> None:
        model = self._build_single_sphere_scene((0.25, 0.5, 0.75))

        with self.assertWarnsRegex(DeprecationWarning, r"config=.*default_render_config"):
            sensor = SensorTiledCamera(model=model, config=None)

        self.assertIsInstance(sensor.default_render_config, SensorTiledCamera.RenderConfig)

    def test_constructor_rejects_default_render_config_and_config(self) -> None:
        model = self._build_single_sphere_scene((0.25, 0.5, 0.75))

        with self.assertWarnsRegex(DeprecationWarning, r"config=.*default_render_config"):
            with self.assertRaisesRegex(TypeError, "default_render_config.*config"):
                SensorTiledCamera(
                    model=model,
                    default_render_config=SensorTiledCamera.RenderConfig(),
                    config=SensorTiledCamera.RenderConfig(),
                )

    def test_utils_implicit_default_render_config_update_warns(self) -> None:
        model = self._build_single_sphere_scene((0.25, 0.5, 0.75))
        sensor = SensorTiledCamera(model=model)

        self.assertFalse(sensor.default_render_config.enable_shadows)
        self.assertFalse(sensor.default_render_config.enable_textures)

        with self.assertWarnsRegex(DeprecationWarning, "create_default_light.*default_render_config"):
            sensor.utils.create_default_light(enable_shadows=True)
        with self.assertWarnsRegex(DeprecationWarning, "assign_checkerboard_material.*default_render_config"):
            sensor.utils.assign_checkerboard_material(shape_indices=[0])

        self.assertTrue(sensor.default_render_config.enable_shadows)
        self.assertTrue(sensor.default_render_config.enable_textures)

    def test_utils_explicit_render_config_field_update_does_not_warn(self) -> None:
        model = self._build_single_sphere_scene((0.25, 0.5, 0.75))
        sensor = SensorTiledCamera(model=model)
        sensor.default_render_config.enable_shadows = True
        sensor.default_render_config.enable_textures = True

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            sensor.utils.create_default_light(enable_shadows=True)
            sensor.utils.assign_checkerboard_material(shape_indices=[0])

        self.assertFalse(any(issubclass(w.category, DeprecationWarning) for w in caught))
        self.assertTrue(sensor.default_render_config.enable_shadows)
        self.assertTrue(sensor.default_render_config.enable_textures)

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
                default_render_config=SensorTiledCamera.RenderConfig(output_color_space=output_color_space),
            )
            camera_rays = sensor.utils.compute_camera_rays_pinhole(1, 1, camera_fovs=math.radians(30.0))
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

    def test_render_context_none_config_uses_default(self) -> None:
        model = self._build_single_sphere_scene((0.25, 0.5, 0.75))
        sensor = SensorTiledCamera(model=model)
        render_context = sensor._SensorTiledCamera__render_context

        camera_transforms = wp.array(
            [[wp.transformf(wp.vec3f(0.0), wp.quatf(0.0, 0.0, 0.0, 1.0))]],
            dtype=wp.transformf,
            device="cpu",
        )
        camera_rays = sensor.utils.compute_camera_rays_pinhole(1, 1, camera_fovs=math.radians(30.0))
        depth_image = sensor.utils.create_depth_image_output(1, 1, camera_count=1)

        render_context.render(
            model,
            model.state(),
            camera_transforms=camera_transforms,
            camera_rays=camera_rays,
            depth_image=depth_image,
            config=None,
        )

        self.assertGreater(depth_image.numpy()[0, 0, 0, 0], 0.0)

    def test_forward_depth_image_matches_utility(self) -> None:
        builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
        body = builder.add_body(xform=wp.transform(p=wp.vec3(0.0, 0.0, -5.0), q=wp.quat_identity()))
        builder.add_shape_box(body, hx=10.0, hy=10.0, hz=0.1, color=(0.25, 0.5, 0.75))
        model = builder.finalize(device="cpu")

        sensor = SensorTiledCamera(model=model)
        width, height = 5, 5
        camera_transforms = wp.array(
            [[wp.transformf(wp.vec3f(0.0), wp.quatf(0.0, 0.0, 0.0, 1.0))]],
            dtype=wp.transformf,
            device="cpu",
        )
        camera_rays = sensor.utils.compute_camera_rays_pinhole(width, height, camera_fovs=math.radians(90.0))
        depth_image = sensor.utils.create_depth_image_output(width, height)
        forward_depth_image = sensor.utils.create_forward_depth_image_output(width, height)

        state = model.state()
        sensor.update(
            state,
            camera_transforms,
            camera_rays,
            depth_image=depth_image,
            forward_depth_image=forward_depth_image,
        )

        converted_depth = sensor.utils.convert_ray_depth_to_forward_depth(depth_image, camera_transforms, camera_rays)
        np.testing.assert_allclose(forward_depth_image.numpy(), converted_depth.numpy(), rtol=1.0e-5, atol=1.0e-5)

        depth_np = depth_image.numpy()[0, 0]
        forward_depth_np = forward_depth_image.numpy()[0, 0]
        self.assertLess(float(forward_depth_np[0, 0]), float(depth_np[0, 0]))
        self.assertAlmostEqual(
            float(forward_depth_np[height // 2, width // 2]), float(depth_np[height // 2, width // 2])
        )

        forward_depth_only_image = sensor.utils.create_forward_depth_image_output(width, height)
        sensor.update(state, camera_transforms, camera_rays, forward_depth_image=forward_depth_only_image)
        np.testing.assert_allclose(
            forward_depth_only_image.numpy(), forward_depth_image.numpy(), rtol=1.0e-5, atol=1.0e-5
        )

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

        sensor = SensorTiledCamera(model=model)
        # The public render_context alias was removed; this regression test needs
        # the internal mesh state that drives first-render construction.
        render_context = sensor._SensorTiledCamera__render_context

        # init_from_model copies model.particle_q/tri_indices into triangle_points/
        # triangle_indices but does not build wp.Mesh until the first render call.
        self.assertTrue(render_context.has_triangle_mesh)
        self.assertIsNone(render_context.triangle_mesh)

        width, height = 8, 8
        camera_transforms = wp.array(
            [[wp.transformf(wp.vec3f(0.0, 0.0, 0.5), wp.quatf(0.0, 0.0, 0.0, 1.0))]],
            dtype=wp.transformf,
            device="cpu",
        )
        camera_rays = sensor.utils.compute_camera_rays_pinhole(width, height, camera_fovs=math.radians(60.0))
        depth_image = sensor.utils.create_depth_image_output(width, height)

        sensor.update(model.state(), camera_transforms, camera_rays, depth_image=depth_image)

        # update() must construct the cloth triangle mesh on this first render call.
        self.assertIsNotNone(render_context.triangle_mesh)

        # Depth hits prove the mesh was passed into the render kernel, not just created.
        self.assertGreater(int(np.sum(depth_image.numpy() > 0.0)), 0)

    def test_render_config_can_enable_particles_with_triangle_mesh(self) -> None:
        model = self._build_mixed_cloth_particle_scene()
        sensor = SensorTiledCamera(
            model=model,
            default_render_config=SensorTiledCamera.RenderConfig(enable_particles=False, max_distance=10.0),
        )

        camera_transforms = wp.array(
            [[wp.transformf(wp.vec3f(0.0), wp.quatf(0.0, 0.0, 0.0, 1.0))]],
            dtype=wp.transformf,
            device="cpu",
        )
        camera_rays = sensor.utils.compute_camera_rays_pinhole(1, 1, camera_fovs=math.radians(30.0))
        state = model.state()

        disabled_depth_image = sensor.utils.create_depth_image_output(1, 1)
        sensor.update(state, camera_transforms, camera_rays, depth_image=disabled_depth_image)
        self.assertEqual(disabled_depth_image.numpy()[0, 0, 0, 0], 0.0)

        enabled_depth_image = sensor.utils.create_depth_image_output(1, 1)
        enabled_shape_index_image = sensor.utils.create_shape_index_image_output(1, 1)
        sensor.update(
            state,
            camera_transforms,
            camera_rays,
            depth_image=enabled_depth_image,
            shape_index_image=enabled_shape_index_image,
            render_config=SensorTiledCamera.RenderConfig(enable_particles=True, max_distance=10.0),
        )
        self.assertGreater(enabled_depth_image.numpy()[0, 0, 0, 0], 0.0)
        self.assertEqual(int(enabled_shape_index_image.numpy()[0, 0, 0, 0]), int(PARTICLES_SHAPE_ID))

        cloth_vertex = model.particle_q.numpy()[0]
        cloth_camera_transforms = wp.array(
            [
                [
                    wp.transformf(
                        wp.vec3f(float(cloth_vertex[0]), float(cloth_vertex[1]), float(cloth_vertex[2] + 0.5)),
                        wp.quatf(0.0, 0.0, 0.0, 1.0),
                    )
                ]
            ],
            dtype=wp.transformf,
            device="cpu",
        )
        cloth_depth_image = sensor.utils.create_depth_image_output(1, 1)
        cloth_shape_index_image = sensor.utils.create_shape_index_image_output(1, 1)
        sensor.update(
            state,
            cloth_camera_transforms,
            camera_rays,
            depth_image=cloth_depth_image,
            shape_index_image=cloth_shape_index_image,
            render_config=SensorTiledCamera.RenderConfig(
                enable_backface_culling=False,
                enable_particles=True,
                max_distance=10.0,
            ),
        )
        self.assertGreater(cloth_depth_image.numpy()[0, 0, 0, 0], 0.0)
        self.assertEqual(int(cloth_shape_index_image.numpy()[0, 0, 0, 0]), int(TRIANGLE_MESH_SHAPE_ID))

    def test_render_config_masks_volume_deformable_particles(self) -> None:
        model = self._build_soft_grid_scene()
        sensor = SensorTiledCamera(model=model)

        camera_transforms = wp.array(
            [[wp.transformf(wp.vec3f(0.0), wp.quatf(0.0, 0.0, 0.0, 1.0))]],
            dtype=wp.transformf,
            device="cpu",
        )
        camera_rays = sensor.utils.compute_camera_rays_pinhole(1, 1, camera_fovs=math.radians(30.0))
        depth_image = sensor.utils.create_depth_image_output(1, 1)
        shape_index_image = sensor.utils.create_shape_index_image_output(1, 1)

        sensor.update(
            model.state(),
            camera_transforms,
            camera_rays,
            depth_image=depth_image,
            shape_index_image=shape_index_image,
            render_config=SensorTiledCamera.RenderConfig(
                enable_backface_culling=False,
                enable_particles=True,
                max_distance=10.0,
            ),
        )

        self.assertGreater(depth_image.numpy()[0, 0, 0, 0], 0.0)
        self.assertEqual(int(shape_index_image.numpy()[0, 0, 0, 0]), int(TRIANGLE_MESH_SHAPE_ID))

    def test_checkerboard_material_requires_keyword_arguments(self) -> None:
        model = self._build_single_sphere_scene((0.25, 0.5, 0.75))
        sensor = SensorTiledCamera(model=model)

        with self.assertRaises(TypeError):
            sensor.utils.assign_checkerboard_material([0])

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
        tiled_camera_sensor.default_render_config.enable_shadows = True
        tiled_camera_sensor.default_render_config.enable_textures = True
        tiled_camera_sensor.utils.create_default_light(enable_shadows=True)
        tiled_camera_sensor.utils.assign_checkerboard_material(
            shape_indices=np.arange(model.shape_count, dtype=np.int32)
        )

        camera_rays = tiled_camera_sensor.utils.compute_camera_rays_pinhole(
            width, height, camera_fovs=math.radians(45.0)
        )
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
    def test_deprecated_checkerboard_material_to_all_shapes_warns(self):
        model = self._shared_model
        tiled_camera_sensor = SensorTiledCamera(model=model)

        with self.assertWarnsRegex(DeprecationWarning, "assign_checkerboard_material"):
            tiled_camera_sensor.utils.assign_checkerboard_material_to_all_shapes()

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
        camera_rays = tiled_camera_sensor.utils.compute_camera_rays_pinhole(
            width, height, camera_fovs=math.radians(45.0)
        )

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
