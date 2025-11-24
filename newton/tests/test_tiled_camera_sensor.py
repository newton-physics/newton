# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
import os
import unittest

import numpy as np
import warp as wp
from pxr import Usd, UsdGeom

import newton
from newton.sensors import TiledCameraSensor


class TestTiledCameraSensor(unittest.TestCase):
    def __build_scene(self):
        builder = newton.ModelBuilder()

        # add ground plane
        builder.add_ground_plane()

        # SPHERE
        sphere_pos = wp.vec3(0.0, -2.0, 0.5)
        body_sphere = builder.add_body(xform=wp.transform(p=sphere_pos, q=wp.quat_identity()), key="sphere")
        builder.add_shape_sphere(body_sphere, radius=0.5)

        # CAPSULE
        capsule_pos = wp.vec3(0.0, 0.0, 0.75)
        body_capsule = builder.add_body(xform=wp.transform(p=capsule_pos, q=wp.quat_identity()), key="capsule")
        builder.add_shape_capsule(body_capsule, radius=0.25, half_height=0.5)

        # CYLINDER
        cylinder_pos = wp.vec3(0.0, -4.0, 0.5)
        body_cylinder = builder.add_body(xform=wp.transform(p=cylinder_pos, q=wp.quat_identity()), key="cylinder")
        builder.add_shape_cylinder(body_cylinder, radius=0.4, half_height=0.5)

        # BOX
        box_pos = wp.vec3(0.0, 2.0, 0.5)
        body_box = builder.add_body(xform=wp.transform(p=box_pos, q=wp.quat_identity()), key="box")
        builder.add_shape_box(body_box, hx=0.5, hy=0.35, hz=0.5)

        # MESH (bunny)
        bunny_filename = os.path.join(os.path.dirname(__file__), "..", "examples", "assets", "bunny.usd")
        self.assertTrue(os.path.exists(bunny_filename), f"File not found: {bunny_filename}")
        usd_stage = Usd.Stage.Open(bunny_filename)
        usd_geom = UsdGeom.Mesh(usd_stage.GetPrimAtPath("/root/bunny"))

        mesh_vertices = np.array(usd_geom.GetPointsAttr().Get())
        mesh_indices = np.array(usd_geom.GetFaceVertexIndicesAttr().Get())

        demo_mesh = newton.Mesh(mesh_vertices, mesh_indices)

        mesh_pos = wp.vec3(0.0, 4.0, 0.0)
        body_mesh = builder.add_body(xform=wp.transform(p=mesh_pos, q=wp.quat(0.5, 0.5, 0.5, 0.5)), key="mesh")
        builder.add_shape_mesh(body_mesh, mesh=demo_mesh)

        return builder.finalize()

    def __compare_images(self, test_image: np.ndarray, gold_image: np.ndarray, allowed_difference: float = 0.0):
        self.assertEqual(test_image.dtype, gold_image.dtype, "Images have different data types")
        self.assertEqual(test_image.shape, gold_image.shape, "Images have different data shapes")

        def _absdiff(x, y):
            if x > y:
                return x - y
            return y - x

        absdiff = np.vectorize(_absdiff)

        diff = absdiff(test_image, gold_image)

        divider = 1.0
        if np.issubdtype(test_image.dtype, np.integer):
            divider = np.iinfo(test_image.dtype).max

        percentage_diff = np.average(diff) / divider * 100.0
        self.assertLessEqual(
            percentage_diff,
            allowed_difference,
            f"Images differ more than {allowed_difference:.2f}%, total difference is {percentage_diff:.2f}%",
        )

    @unittest.skipUnless(wp.is_cuda_available(), "Requires CUDA")
    def test_golden_image(self):
        model = self.__build_scene()

        camera_position = wp.vec3f(10.0, 0.0, 2.0)
        camera_orientation = wp.mat33f(0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0)
        camera_transform = wp.array(
            wp.transformf(camera_position, wp.quat_from_matrix(camera_orientation)), dtype=wp.transformf
        )

        tiled_camera_sensor = TiledCameraSensor(model=model, num_cameras=1, width=320, height=240)
        tiled_camera_sensor.create_default_light()
        tiled_camera_sensor.assign_debug_colors_per_shape()
        tiled_camera_sensor.assign_default_checkerboard_material()
        tiled_camera_sensor.update_cameras(camera_transform)
        tiled_camera_sensor.compute_camera_rays(wp.array([math.radians(45.0)], dtype=wp.float32))
        color_image = tiled_camera_sensor.create_color_image_output()
        depth_image = tiled_camera_sensor.create_depth_image_output()

        tiled_camera_sensor.update_from_state(model.state())
        tiled_camera_sensor.render(color_image, depth_image)

        golden_color_data = np.load(
            os.path.join(os.path.dirname(__file__), "golden_data", "test_tiled_camera_sensor", "color.npy")
        )
        golden_depth_data = np.load(
            os.path.join(os.path.dirname(__file__), "golden_data", "test_tiled_camera_sensor", "depth.npy")
        )

        self.__compare_images(color_image.numpy(), golden_color_data, allowed_difference=0.1)
        self.__compare_images(depth_image.numpy(), golden_depth_data, allowed_difference=0.1)

    @unittest.skipUnless(wp.is_cuda_available(), "Requires CUDA")
    def test_output_image_parameters(self):
        model = self.__build_scene()

        camera_position = wp.vec3f(10.0, 0.0, 2.0)
        camera_orientation = wp.mat33f(0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0)
        camera_transform = wp.array(
            wp.transformf(camera_position, wp.quat_from_matrix(camera_orientation)), dtype=wp.transformf
        )

        tiled_camera_sensor = TiledCameraSensor(model=model, num_cameras=1, width=640, height=460)
        tiled_camera_sensor.update_cameras(camera_transform)
        tiled_camera_sensor.compute_camera_rays(wp.array([math.radians(45.0)], dtype=wp.float32))
        tiled_camera_sensor.update_from_state(model.state())

        color_image = tiled_camera_sensor.create_color_image_output()
        depth_image = tiled_camera_sensor.create_depth_image_output()
        tiled_camera_sensor.render(color_image, depth_image)
        self.assertTrue(np.any(color_image.numpy() != 0), "Color image should contain rendered data")
        self.assertTrue(np.any(depth_image.numpy() != 0), "Depth image should contain rendered data")

        color_image = tiled_camera_sensor.create_color_image_output()
        depth_image = tiled_camera_sensor.create_depth_image_output()
        tiled_camera_sensor.render(color_image, None)
        self.assertTrue(np.any(color_image.numpy() != 0), "Color image should contain rendered data")
        self.assertFalse(np.any(depth_image.numpy() != 0), "Depth image should NOT contain rendered data")

        color_image = tiled_camera_sensor.create_color_image_output()
        depth_image = tiled_camera_sensor.create_depth_image_output()
        tiled_camera_sensor.render(None, depth_image)
        self.assertFalse(np.any(color_image.numpy() != 0), "Color image should NOT contain rendered data")
        self.assertTrue(np.any(depth_image.numpy() != 0), "Depth image should contain rendered data")

        color_image = tiled_camera_sensor.create_color_image_output()
        depth_image = tiled_camera_sensor.create_depth_image_output()
        tiled_camera_sensor.render(None, None)
        self.assertFalse(np.any(color_image.numpy() != 0), "Color image should NOT contain rendered data")
        self.assertFalse(np.any(depth_image.numpy() != 0), "Depth image should NOT contain rendered data")


if __name__ == "__main__":
    unittest.main()
