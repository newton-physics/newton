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
import tempfile
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

    @unittest.skipUnless(USD_AVAILABLE, "Requires usd-core")
    def test_joint_solref_solimp_import(self):
        """Test that joint solref and solimp parameters are correctly imported from USD."""
        try:
            from pxr import Sdf, Usd, UsdGeom, UsdPhysics  # noqa: PLC0415
        except ImportError:
            self.skipTest("USD not available")

        # Create a temporary USD file with custom joint newton:joint_solref and newton:joint_solimp attributes
        with tempfile.NamedTemporaryFile(suffix=".usda", delete=False) as f:
            usd_filename = f.name

        try:
            # Create USD stage
            stage = Usd.Stage.CreateNew(usd_filename)

            # Add physics scene
            physics_scene = UsdPhysics.Scene.Define(stage, "/physicsScene")
            physics_scene.CreateGravityDirectionAttr().Set((0.0, 0.0, -1.0))
            physics_scene.CreateGravityMagnitudeAttr().Set(9.81)

            # Create an articulation to contain our bodies and joints
            UsdPhysics.ArticulationRootAPI.Apply(stage.DefinePrim("/articulation", "Xform"))

            # Create ground body with collision shape as part of articulation
            ground_prim = stage.DefinePrim("/articulation/ground", "Xform")
            rigid_body = UsdPhysics.RigidBodyAPI.Apply(ground_prim)
            rigid_body.CreateRigidBodyEnabledAttr().Set(True)
            UsdPhysics.MassAPI.Apply(ground_prim)

            # Add collision shape to ground
            ground_shape = UsdGeom.Cube.Define(stage, "/articulation/ground/shape")
            ground_shape.CreateSizeAttr().Set(10.0)
            UsdPhysics.CollisionAPI.Apply(ground_shape.GetPrim())

            # Create first body with joint that has custom solref/solimp
            body1_prim = stage.DefinePrim("/articulation/body1", "Xform")
            UsdGeom.Xform(body1_prim).AddTranslateOp().Set((0.0, 0.0, 1.0))
            rigid_body1 = UsdPhysics.RigidBodyAPI.Apply(body1_prim)
            rigid_body1.CreateRigidBodyEnabledAttr().Set(True)
            mass_api1 = UsdPhysics.MassAPI.Apply(body1_prim)
            mass_api1.CreateMassAttr().Set(1.0)

            # Add collision shape to body1
            body1_shape = UsdGeom.Cube.Define(stage, "/articulation/body1/shape")
            body1_shape.CreateSizeAttr().Set(0.5)
            UsdPhysics.CollisionAPI.Apply(body1_shape.GetPrim())

            # Create revolute joint with custom solref/solimp
            joint1 = UsdPhysics.RevoluteJoint.Define(stage, "/articulation/joint1")
            joint1.CreateAxisAttr().Set("Z")
            joint1.CreateBody0Rel().SetTargets(["/articulation/ground"])
            joint1.CreateBody1Rel().SetTargets(["/articulation/body1"])
            joint1.CreateLowerLimitAttr().Set(-90.0)
            joint1.CreateUpperLimitAttr().Set(90.0)

            # Add custom warp attributes
            joint1_prim = joint1.GetPrim()
            joint1_prim.CreateAttribute("warp:joint_solref", Sdf.ValueTypeNames.FloatArray).Set([0.05, 2.0])
            joint1_prim.CreateAttribute("warp:joint_solimp", Sdf.ValueTypeNames.FloatArray).Set(
                [0.8, 0.9, 0.002, 0.6, 3.0]
            )

            # Create second body with joint that has only solref
            body2_prim = stage.DefinePrim("/articulation/body2", "Xform")
            UsdGeom.Xform(body2_prim).AddTranslateOp().Set((1.0, 0.0, 1.0))
            rigid_body2 = UsdPhysics.RigidBodyAPI.Apply(body2_prim)
            rigid_body2.CreateRigidBodyEnabledAttr().Set(True)
            mass_api2 = UsdPhysics.MassAPI.Apply(body2_prim)
            mass_api2.CreateMassAttr().Set(1.0)

            # Add collision shape to body2
            body2_shape = UsdGeom.Cube.Define(stage, "/articulation/body2/shape")
            body2_shape.CreateSizeAttr().Set(0.5)
            UsdPhysics.CollisionAPI.Apply(body2_shape.GetPrim())

            # Create prismatic joint with only custom solref
            joint2 = UsdPhysics.PrismaticJoint.Define(stage, "/articulation/joint2")
            joint2.CreateAxisAttr().Set("X")
            joint2.CreateBody0Rel().SetTargets(["/articulation/ground"])
            joint2.CreateBody1Rel().SetTargets(["/articulation/body2"])
            joint2.CreateLowerLimitAttr().Set(-1.0)
            joint2.CreateUpperLimitAttr().Set(1.0)

            # Add only solref attribute
            joint2_prim = joint2.GetPrim()
            joint2_prim.CreateAttribute("warp:joint_solref", Sdf.ValueTypeNames.FloatArray).Set([0.1, 1.5])

            # Create third body with joint that has no custom attributes
            body3_prim = stage.DefinePrim("/articulation/body3", "Xform")
            UsdGeom.Xform(body3_prim).AddTranslateOp().Set((2.0, 0.0, 1.0))
            rigid_body3 = UsdPhysics.RigidBodyAPI.Apply(body3_prim)
            rigid_body3.CreateRigidBodyEnabledAttr().Set(True)
            mass_api3 = UsdPhysics.MassAPI.Apply(body3_prim)
            mass_api3.CreateMassAttr().Set(1.0)

            # Add collision shape to body3
            body3_shape = UsdGeom.Cube.Define(stage, "/articulation/body3/shape")
            body3_shape.CreateSizeAttr().Set(0.5)
            UsdPhysics.CollisionAPI.Apply(body3_shape.GetPrim())

            # Create revolute joint with no custom attributes
            joint3 = UsdPhysics.RevoluteJoint.Define(stage, "/articulation/joint3")
            joint3.CreateAxisAttr().Set("Y")
            joint3.CreateBody0Rel().SetTargets(["/articulation/ground"])
            joint3.CreateBody1Rel().SetTargets(["/articulation/body3"])
            joint3.CreateLowerLimitAttr().Set(-45.0)
            joint3.CreateUpperLimitAttr().Set(45.0)

            # Save the stage
            stage.Save()

            # Import the USD file
            builder = newton.ModelBuilder()
            parse_usd(usd_filename, builder)
            model = builder.finalize()

            # Check joints - there will be free joints for each body plus our defined joints
            # Filter to only the joints we explicitly defined
            joint_types = model.joint_type.numpy()

            # Debug: print all joint types (commented out)
            # print(f"Total joints: {model.joint_count}")
            # print(f"Joint types: {joint_types}")
            # print(f"Joint type values: FREE={newton.JOINT_FREE}, REVOLUTE={newton.JOINT_REVOLUTE}, PRISMATIC={newton.JOINT_PRISMATIC}, D6={newton.JOINT_D6}")
            # print(f"Body count: {model.body_count}")

            revolute_joints = [i for i in range(model.joint_count) if joint_types[i] == newton.JOINT_REVOLUTE]
            prismatic_joints = [i for i in range(model.joint_count) if joint_types[i] == newton.JOINT_PRISMATIC]

            # The joints might be imported as D6 joints
            d6_joints = [i for i in range(model.joint_count) if joint_types[i] == newton.JOINT_D6]

            # We should have at least 3 non-free joints
            non_free_joints = len(revolute_joints) + len(prismatic_joints) + len(d6_joints)
            self.assertGreaterEqual(
                non_free_joints,
                3,
                f"Should have at least 3 non-free joints, but got {non_free_joints} (revolute: {len(revolute_joints)}, prismatic: {len(prismatic_joints)}, d6: {len(d6_joints)})",
            )

            # Check joint_solref and joint_solimp arrays exist
            self.assertIsNotNone(model.joint_solref, "joint_solref should not be None")
            self.assertIsNotNone(model.joint_solimp, "joint_solimp should not be None")

            solref_values = model.joint_solref.numpy()
            solimp_values = model.joint_solimp.numpy()

            # Check that we have the expected solref/solimp values somewhere in the arrays
            # Since free joints are added, we need to search for our specific values

            # Find DOFs with custom solref values
            custom_solref1_found = False
            custom_solref2_found = False

            for i in range(model.joint_dof_count):
                if np.allclose(solref_values[i], [0.05, 2.0]):
                    # This should be joint1 with custom solref and solimp
                    custom_solref1_found = True
                    np.testing.assert_array_almost_equal(
                        solimp_values[i],
                        [0.8, 0.9, 0.002, 0.6, 3.0],
                        err_msg="Joint with solref [0.05, 2.0] should have custom solimp values",
                    )
                elif np.allclose(solref_values[i], [0.1, 1.5]):
                    # This should be joint2 with custom solref but default solimp
                    custom_solref2_found = True
                    np.testing.assert_array_almost_equal(
                        solimp_values[i],
                        [0.9, 0.95, 0.001, 0.5, 2.0],
                        err_msg="Joint with solref [0.1, 1.5] should have default solimp values",
                    )

            self.assertTrue(custom_solref1_found, "Should find joint with custom solref [0.05, 2.0]")
            self.assertTrue(custom_solref2_found, "Should find joint with custom solref [0.1, 1.5]")

            # Also check that some joints have default values
            default_joints = [
                i
                for i in range(model.joint_dof_count)
                if np.allclose(solref_values[i], [0.02, 1.0])
                and np.allclose(solimp_values[i], [0.9, 0.95, 0.001, 0.5, 2.0])
            ]
            self.assertGreater(len(default_joints), 0, "Should have at least one joint with default solref/solimp")

        finally:
            # Clean up the temporary file
            os.unlink(usd_filename)


if __name__ == "__main__":
    unittest.main(verbosity=2, failfast=True)
