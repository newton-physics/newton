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

import os
import unittest

import numpy as np
import warp as wp

import newton
import newton.examples
from newton.geometry.utils import create_box_mesh, transform_points
from newton.tests.unittest_utils import USD_AVAILABLE, assert_np_equal, get_test_devices
from newton.utils import parse_usd

devices = get_test_devices()


class TestImportUsd(unittest.TestCase):
    @unittest.skipUnless(USD_AVAILABLE, "Requires usd-core")
    def test_import_articulation(self):
        builder = newton.ModelBuilder()

        results = parse_usd(
            os.path.join(os.path.dirname(__file__), "assets", "ant.usda"),
            builder,
            collapse_fixed_joints=True,
        )
        self.assertEqual(builder.body_count, 9)
        self.assertEqual(builder.shape_count, 26)
        self.assertEqual(len(builder.shape_key), len(set(builder.shape_key)))
        self.assertEqual(len(builder.body_key), len(set(builder.body_key)))
        self.assertEqual(len(builder.joint_key), len(set(builder.joint_key)))
        # 8 joints + 1 free joint for the root body
        self.assertEqual(builder.joint_count, 9)
        self.assertEqual(builder.joint_dof_count, 14)
        self.assertEqual(builder.joint_coord_count, 15)
        self.assertEqual(builder.joint_type, [newton.JOINT_FREE] + [newton.JOINT_REVOLUTE] * 8)
        self.assertEqual(len(results["path_body_map"]), 9)
        self.assertEqual(len(results["path_shape_map"]), 26)

        collision_shapes = [
            i
            for i in range(builder.shape_count)
            if builder.shape_flags[i] & int(newton.geometry.SHAPE_FLAG_COLLIDE_SHAPES)
        ]
        self.assertEqual(len(collision_shapes), 13)

    @unittest.skipUnless(USD_AVAILABLE, "Requires usd-core")
    def test_import_articulation_no_visuals(self):
        builder = newton.ModelBuilder()

        results = parse_usd(
            os.path.join(os.path.dirname(__file__), "assets", "ant.usda"),
            builder,
            collapse_fixed_joints=True,
            load_non_physics_prims=False,
        )
        self.assertEqual(builder.body_count, 9)
        self.assertEqual(builder.shape_count, 13)
        self.assertEqual(len(builder.shape_key), len(set(builder.shape_key)))
        self.assertEqual(len(builder.body_key), len(set(builder.body_key)))
        self.assertEqual(len(builder.joint_key), len(set(builder.joint_key)))
        # 8 joints + 1 free joint for the root body
        self.assertEqual(builder.joint_count, 9)
        self.assertEqual(builder.joint_dof_count, 14)
        self.assertEqual(builder.joint_coord_count, 15)
        self.assertEqual(builder.joint_type, [newton.JOINT_FREE] + [newton.JOINT_REVOLUTE] * 8)
        self.assertEqual(len(results["path_body_map"]), 9)
        self.assertEqual(len(results["path_shape_map"]), 13)

        collision_shapes = [
            i
            for i in range(builder.shape_count)
            if builder.shape_flags[i] & int(newton.geometry.SHAPE_FLAG_COLLIDE_SHAPES)
        ]
        self.assertEqual(len(collision_shapes), 13)

    @unittest.skipUnless(USD_AVAILABLE, "Requires usd-core")
    def test_import_articulation_with_mesh(self):
        builder = newton.ModelBuilder()

        _ = parse_usd(
            os.path.join(os.path.dirname(__file__), "assets", "simple_articulation_with_mesh.usda"),
            builder,
            collapse_fixed_joints=True,
        )

    @unittest.skipUnless(USD_AVAILABLE, "Requires usd-core")
    def test_joint_ordering(self):
        builder_dfs = newton.ModelBuilder()
        parse_usd(
            os.path.join(os.path.dirname(__file__), "assets", "ant.usda"),
            builder_dfs,
            collapse_fixed_joints=True,
            joint_ordering="dfs",
        )
        expected = [
            "front_left_leg",
            "front_left_foot",
            "front_right_leg",
            "front_right_foot",
            "left_back_leg",
            "left_back_foot",
            "right_back_leg",
            "right_back_foot",
        ]
        for i in range(8):
            self.assertTrue(builder_dfs.joint_key[i + 1].endswith(expected[i]))

        builder_bfs = newton.ModelBuilder()
        parse_usd(
            os.path.join(os.path.dirname(__file__), "assets", "ant.usda"),
            builder_bfs,
            collapse_fixed_joints=True,
            joint_ordering="bfs",
        )
        expected = [
            "front_left_leg",
            "front_right_leg",
            "left_back_leg",
            "right_back_leg",
            "front_left_foot",
            "front_right_foot",
            "left_back_foot",
            "right_back_foot",
        ]
        for i in range(8):
            self.assertTrue(builder_bfs.joint_key[i + 1].endswith(expected[i]))

    @unittest.skipUnless(USD_AVAILABLE, "Requires usd-core")
    def test_env_cloning(self):
        builder_no_cloning = newton.ModelBuilder()
        builder_cloning = newton.ModelBuilder()
        parse_usd(
            os.path.join(os.path.dirname(__file__), "assets", "ant_multi.usda"),
            builder_no_cloning,
            collapse_fixed_joints=True,
        )
        parse_usd(
            os.path.join(os.path.dirname(__file__), "assets", "ant_multi.usda"),
            builder_cloning,
            collapse_fixed_joints=True,
            cloned_env="/World/envs/env_0",
        )
        self.assertEqual(builder_cloning.articulation_key, builder_no_cloning.articulation_key)
        # ordering of the shape keys may differ
        shape_key_cloning = set(builder_cloning.shape_key)
        shape_key_no_cloning = set(builder_no_cloning.shape_key)
        self.assertEqual(len(shape_key_cloning), len(shape_key_no_cloning))
        for key in shape_key_cloning:
            self.assertIn(key, shape_key_no_cloning)
        self.assertEqual(builder_cloning.body_key, builder_no_cloning.body_key)
        # ignore keys that are not USD paths (e.g. "joint_0" gets repeated N times)
        joint_key_cloning = [k for k in builder_cloning.joint_key if k.startswith("/World")]
        joint_key_no_cloning = [k for k in builder_no_cloning.joint_key if k.startswith("/World")]
        self.assertEqual(joint_key_cloning, joint_key_no_cloning)

    @unittest.skipUnless(USD_AVAILABLE, "Requires usd-core")
    def test_mass_calculations(self):
        builder = newton.ModelBuilder()

        _ = parse_usd(
            os.path.join(os.path.dirname(__file__), "assets", "ant.usda"),
            builder,
            collapse_fixed_joints=True,
        )

        np.testing.assert_allclose(
            np.array(builder.body_mass),
            np.array(
                [
                    0.09677605,
                    0.00783155,
                    0.01351844,
                    0.00783155,
                    0.01351844,
                    0.00783155,
                    0.01351844,
                    0.00783155,
                    0.01351844,
                ]
            ),
            rtol=1e-5,
            atol=1e-7,
        )

    @unittest.skipUnless(USD_AVAILABLE, "Requires usd-core")
    def test_import_cube_cylinder_joint_count(self):
        builder = newton.ModelBuilder()
        import_results = parse_usd(
            os.path.join(os.path.dirname(__file__), "assets", "cube_cylinder.usda"),
            builder,
            collapse_fixed_joints=True,
            invert_rotations=True,
        )
        self.assertEqual(builder.body_count, 1)
        self.assertEqual(builder.shape_count, 2)
        self.assertEqual(builder.joint_count, 1)

        usd_path_to_shape = import_results["path_shape_map"]
        expected = {
            "/World/Cylinder_dynamic/cylinder_reverse/mesh_0": {"mu": 0.2, "restitution": 0.3},
            "/World/Cube_static/cube2/mesh_0": {"mu": 0.75, "restitution": 0.3},
        }
        # Reverse mapping: shape index -> USD path
        shape_idx_to_usd_path = {v: k for k, v in usd_path_to_shape.items()}
        for shape_idx in range(builder.shape_count):
            usd_path = shape_idx_to_usd_path[shape_idx]
            if usd_path in expected:
                self.assertAlmostEqual(builder.shape_material_mu[shape_idx], expected[usd_path]["mu"], places=5)
                self.assertAlmostEqual(
                    builder.shape_material_restitution[shape_idx], expected[usd_path]["restitution"], places=5
                )

    def test_mesh_approximation(self):
        from pxr import Gf, Usd, UsdGeom, UsdPhysics  # noqa: PLC0415

        def box_mesh(scale=(1.0, 1.0, 1.0), transform: wp.transform | None = None):
            vertices, indices = create_box_mesh(scale)
            if transform is not None:
                vertices = transform_points(vertices, transform)
            return (vertices, indices)

        def create_collision_mesh(name, vertices, indices, approximation_method):
            mesh = UsdGeom.Mesh.Define(stage, name)
            UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())

            mesh.CreateFaceVertexCountsAttr().Set([3] * (len(indices) // 3))
            mesh.CreateFaceVertexIndicesAttr().Set(indices.tolist())
            mesh.CreatePointsAttr().Set([Gf.Vec3f(*p) for p in vertices.tolist()])
            mesh.CreateDoubleSidedAttr().Set(False)

            prim = mesh.GetPrim()
            meshColAPI = UsdPhysics.MeshCollisionAPI.Apply(prim)
            meshColAPI.GetApproximationAttr().Set(approximation_method)
            return prim

        def npsorted(x):
            return np.array(sorted(x))

        stage = Usd.Stage.CreateInMemory()
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
        UsdGeom.SetStageMetersPerUnit(stage, 1.0)
        self.assertTrue(stage)

        scene = UsdPhysics.Scene.Define(stage, "/physicsScene")
        self.assertTrue(scene)

        scale = wp.vec3(1.0, 3.0, 0.2)
        tf = wp.transform(wp.vec3(1.0, 2.0, 3.0), wp.quat_identity())
        vertices, indices = box_mesh(scale=scale, transform=tf)

        create_collision_mesh("/meshOriginal", vertices, indices, UsdPhysics.Tokens.none)
        create_collision_mesh("/meshConvexHull", vertices, indices, UsdPhysics.Tokens.convexHull)
        create_collision_mesh("/meshBoundingSphere", vertices, indices, UsdPhysics.Tokens.boundingSphere)
        create_collision_mesh("/meshBoundingCube", vertices, indices, UsdPhysics.Tokens.boundingCube)

        builder = newton.ModelBuilder()
        newton.geometry.MESH_MAXHULLVERT = 4
        parse_usd(
            stage,
            builder,
        )

        self.assertEqual(builder.body_count, 0)
        self.assertEqual(builder.shape_count, 4)
        self.assertEqual(
            builder.shape_type, [newton.GeoType.MESH, newton.GeoType.MESH, newton.GeoType.SPHERE, newton.GeoType.BOX]
        )

        # original mesh
        mesh_original = builder.shape_source[0]
        self.assertEqual(mesh_original.vertices.shape, (8, 3))
        assert_np_equal(mesh_original.vertices, vertices)
        assert_np_equal(mesh_original.indices, indices)

        # convex hull
        mesh_convex_hull = builder.shape_source[1]
        self.assertEqual(mesh_convex_hull.vertices.shape, (4, 3))

        # bounding sphere
        self.assertIsNone(builder.shape_source[2])
        self.assertEqual(builder.shape_type[2], newton.geometry.GeoType.SPHERE)
        self.assertAlmostEqual(builder.shape_scale[2][0], wp.length(scale))
        assert_np_equal(np.array(builder.shape_transform[2].p), np.array(tf.p), tol=1.0e-4)

        # bounding box
        assert_np_equal(npsorted(builder.shape_scale[3]), npsorted(scale), tol=1.0e-6)
        # only compare the position since the rotation is not guaranteed to be the same
        assert_np_equal(np.array(builder.shape_transform[3].p), np.array(tf.p), tol=1.0e-4)

    @unittest.skipUnless(USD_AVAILABLE, "Requires usd-core")
    def test_visual_match_collision_shapes(self):
        builder = newton.ModelBuilder()
        parse_usd(
            newton.examples.get_asset("humanoid.usda"),
            builder,
        )
        self.assertEqual(builder.shape_count, 38)
        self.assertEqual(builder.body_count, 16)
        visual_shape_keys = [k for k in builder.shape_key if "visuals" in k]
        collision_shape_keys = [k for k in builder.shape_key if "collisions" in k]
        self.assertEqual(len(visual_shape_keys), 19)
        self.assertEqual(len(collision_shape_keys), 19)
        visual_shapes = [i for i, k in enumerate(builder.shape_key) if "visuals" in k]
        # corresponding collision shapes
        collision_shapes = [builder.shape_key.index(k.replace("visuals", "collisions")) for k in visual_shape_keys]
        # ensure that the visual and collision shapes match
        for i in range(len(visual_shapes)):
            vi = visual_shapes[i]
            ci = collision_shapes[i]
            self.assertEqual(builder.shape_type[vi], builder.shape_type[ci])
            self.assertEqual(builder.shape_source[vi], builder.shape_source[ci])
            assert_np_equal(np.array(builder.shape_transform[vi]), np.array(builder.shape_transform[ci]), tol=1e-5)
            assert_np_equal(np.array(builder.shape_scale[vi]), np.array(builder.shape_scale[ci]), tol=1e-5)
            self.assertFalse(builder.shape_flags[vi] & int(newton.geometry.SHAPE_FLAG_COLLIDE_SHAPES))
            self.assertTrue(builder.shape_flags[ci] & int(newton.geometry.SHAPE_FLAG_COLLIDE_SHAPES))


if __name__ == "__main__":
    unittest.main(verbosity=2, failfast=True)
