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

import newton
import newton.examples
from newton import JointType
from newton._src.geometry.utils import create_box_mesh, transform_points
from newton.solvers import SolverMuJoCo
from newton.tests.unittest_utils import USD_AVAILABLE, assert_np_equal, get_test_devices
from newton.utils import quat_between_axes

devices = get_test_devices()


class TestImportUsd(unittest.TestCase):
    @unittest.skipUnless(USD_AVAILABLE, "Requires usd-core")
    def test_import_articulation(self):
        builder = newton.ModelBuilder()

        results = builder.add_usd(
            os.path.join(os.path.dirname(__file__), "assets", "ant.usda"),
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
        self.assertEqual(builder.joint_type, [newton.JointType.FREE] + [newton.JointType.REVOLUTE] * 8)
        self.assertEqual(len(results["path_body_map"]), 9)
        self.assertEqual(len(results["path_shape_map"]), 26)

        collision_shapes = [
            i for i in range(builder.shape_count) if builder.shape_flags[i] & int(newton.ShapeFlags.COLLIDE_SHAPES)
        ]
        self.assertEqual(len(collision_shapes), 13)

    @unittest.skipUnless(USD_AVAILABLE, "Requires usd-core")
    def test_import_articulation_no_visuals(self):
        builder = newton.ModelBuilder()

        results = builder.add_usd(
            os.path.join(os.path.dirname(__file__), "assets", "ant.usda"),
            collapse_fixed_joints=True,
            load_sites=False,
            load_visual_shapes=False,
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
        self.assertEqual(builder.joint_type, [newton.JointType.FREE] + [newton.JointType.REVOLUTE] * 8)
        self.assertEqual(len(results["path_body_map"]), 9)
        self.assertEqual(len(results["path_shape_map"]), 13)

        collision_shapes = [
            i for i in range(builder.shape_count) if builder.shape_flags[i] & newton.ShapeFlags.COLLIDE_SHAPES
        ]
        self.assertEqual(len(collision_shapes), 13)

    @unittest.skipUnless(USD_AVAILABLE, "Requires usd-core")
    def test_import_articulation_with_mesh(self):
        builder = newton.ModelBuilder()

        _ = builder.add_usd(
            os.path.join(os.path.dirname(__file__), "assets", "simple_articulation_with_mesh.usda"),
            collapse_fixed_joints=True,
        )

    @unittest.skipUnless(USD_AVAILABLE, "Requires usd-core")
    def test_import_revolute_articulation(self):
        """Test importing USD with a joint that has missing body1.

        This tests the behavior where:
        - Normally: body0 is parent, body1 is child
        - When body1 is missing: body0 becomes child, world (-1) becomes parent

        The test USD file contains a FixedJoint inside CenterPivot that only
        specifies body0 (itself) but no body1, which should result in the joint
        connecting CenterPivot to the world.
        """
        builder = newton.ModelBuilder()

        results = builder.add_usd(
            os.path.join(os.path.dirname(__file__), "assets", "revolute_articulation.usda"),
            collapse_fixed_joints=False,  # Don't collapse to see all joints
        )

        # The articulation has 2 bodies
        self.assertEqual(builder.body_count, 2)
        self.assertEqual(set(builder.body_key), {"/Articulation/Arm", "/Articulation/CenterPivot"})

        # Should have 2 joints:
        # 1. Fixed joint with only body0 specified (CenterPivot to world)
        # 2. Revolute joint between CenterPivot and Arm (normal joint with both bodies)
        self.assertEqual(builder.joint_count, 2)

        # Find joints by their keys to make test robust to ordering changes
        fixed_joint_idx = builder.joint_key.index("/Articulation/CenterPivot/FixedJoint")
        revolute_joint_idx = builder.joint_key.index("/Articulation/Arm/RevoluteJoint")

        # Verify joint types
        self.assertEqual(builder.joint_type[revolute_joint_idx], newton.JointType.REVOLUTE)
        self.assertEqual(builder.joint_type[fixed_joint_idx], newton.JointType.FIXED)

        # The key test: verify the FixedJoint connects CenterPivot to world
        # because body1 was missing in the USD file
        self.assertEqual(builder.joint_parent[fixed_joint_idx], -1)  # Parent is world (-1)
        # Child should be CenterPivot (which was body0 in the USD)
        center_pivot_idx = builder.body_key.index("/Articulation/CenterPivot")
        self.assertEqual(builder.joint_child[fixed_joint_idx], center_pivot_idx)

        # Verify the import results mapping
        self.assertEqual(len(results["path_body_map"]), 2)
        self.assertEqual(len(results["path_shape_map"]), 1)

    @unittest.skipUnless(USD_AVAILABLE, "Requires usd-core")
    def test_joint_ordering(self):
        builder_dfs = newton.ModelBuilder()
        builder_dfs.add_usd(
            os.path.join(os.path.dirname(__file__), "assets", "ant.usda"),
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
        builder_bfs.add_usd(
            os.path.join(os.path.dirname(__file__), "assets", "ant.usda"),
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
    def test_joint_filtering(self):
        def test_filtering(
            msg,
            ignore_paths,
            bodies_follow_joint_ordering,
            expected_articulation_count,
            expected_joint_types,
            expected_body_keys,
            expected_joint_keys,
        ):
            builder = newton.ModelBuilder()
            builder.add_usd(
                os.path.join(os.path.dirname(__file__), "assets", "four_link_chain_articulation.usda"),
                ignore_paths=ignore_paths,
                bodies_follow_joint_ordering=bodies_follow_joint_ordering,
            )
            self.assertEqual(
                builder.joint_count,
                len(expected_joint_types),
                f"Expected {len(expected_joint_types)} joints after filtering ({msg}; {bodies_follow_joint_ordering!s}), got {builder.joint_count}",
            )
            self.assertEqual(
                builder.articulation_count,
                expected_articulation_count,
                f"Expected {expected_articulation_count} articulations after filtering ({msg}; {bodies_follow_joint_ordering!s}), got {builder.articulation_count}",
            )
            self.assertEqual(
                builder.joint_type,
                expected_joint_types,
                f"Expected {expected_joint_types} joints after filtering ({msg}; {bodies_follow_joint_ordering!s}), got {builder.joint_type}",
            )
            self.assertEqual(
                builder.body_key,
                expected_body_keys,
                f"Expected {expected_body_keys} bodies after filtering ({msg}; {bodies_follow_joint_ordering!s}), got {builder.body_key}",
            )
            self.assertEqual(
                builder.joint_key,
                expected_joint_keys,
                f"Expected {expected_joint_keys} joints after filtering ({msg}; {bodies_follow_joint_ordering!s}), got {builder.joint_key}",
            )

        for bodies_follow_joint_ordering in [True, False]:
            test_filtering(
                "filter out nothing",
                ignore_paths=[],
                bodies_follow_joint_ordering=bodies_follow_joint_ordering,
                expected_articulation_count=1,
                expected_joint_types=[
                    newton.JointType.FIXED,
                    newton.JointType.REVOLUTE,
                    newton.JointType.REVOLUTE,
                    newton.JointType.REVOLUTE,
                ],
                expected_body_keys=[
                    "/Articulation/Body0",
                    "/Articulation/Body1",
                    "/Articulation/Body2",
                    "/Articulation/Body3",
                ],
                expected_joint_keys=[
                    "/Articulation/Joint0",
                    "/Articulation/Joint1",
                    "/Articulation/Joint2",
                    "/Articulation/Joint3",
                ],
            )

            # we filter out all joints, so 4 free-body articulations are created
            test_filtering(
                "filter out all joints",
                ignore_paths=[".*Joint"],
                bodies_follow_joint_ordering=bodies_follow_joint_ordering,
                expected_articulation_count=4,
                expected_joint_types=[newton.JointType.FREE] * 4,
                expected_body_keys=[
                    "/Articulation/Body0",
                    "/Articulation/Body1",
                    "/Articulation/Body2",
                    "/Articulation/Body3",
                ],
                expected_joint_keys=["joint_1", "joint_2", "joint_3", "joint_4"],
            )

            # here we filter out the root fixed joint so that the articulation
            # becomes floating-base
            test_filtering(
                "filter out the root fixed joint",
                ignore_paths=[".*Joint0"],
                bodies_follow_joint_ordering=bodies_follow_joint_ordering,
                expected_articulation_count=1,
                expected_joint_types=[
                    newton.JointType.FREE,
                    newton.JointType.REVOLUTE,
                    newton.JointType.REVOLUTE,
                    newton.JointType.REVOLUTE,
                ],
                expected_body_keys=[
                    "/Articulation/Body0",
                    "/Articulation/Body1",
                    "/Articulation/Body2",
                    "/Articulation/Body3",
                ],
                expected_joint_keys=["joint_1", "/Articulation/Joint1", "/Articulation/Joint2", "/Articulation/Joint3"],
            )

            # filter out all the bodies
            test_filtering(
                "filter out all bodies",
                ignore_paths=[".*Body"],
                bodies_follow_joint_ordering=bodies_follow_joint_ordering,
                expected_articulation_count=0,
                expected_joint_types=[],
                expected_body_keys=[],
                expected_joint_keys=[],
            )

            # filter out the last body, which means the last joint is also filtered out
            test_filtering(
                "filter out the last body",
                ignore_paths=[".*Body3"],
                bodies_follow_joint_ordering=bodies_follow_joint_ordering,
                expected_articulation_count=1,
                expected_joint_types=[newton.JointType.FIXED, newton.JointType.REVOLUTE, newton.JointType.REVOLUTE],
                expected_body_keys=["/Articulation/Body0", "/Articulation/Body1", "/Articulation/Body2"],
                expected_joint_keys=["/Articulation/Joint0", "/Articulation/Joint1", "/Articulation/Joint2"],
            )

            # filter out the first body, which means the first two joints are also filtered out and the articulation becomes floating-base
            test_filtering(
                "filter out the first body",
                ignore_paths=[".*Body0"],
                bodies_follow_joint_ordering=bodies_follow_joint_ordering,
                expected_articulation_count=1,
                expected_joint_types=[newton.JointType.FREE, newton.JointType.REVOLUTE, newton.JointType.REVOLUTE],
                expected_body_keys=["/Articulation/Body1", "/Articulation/Body2", "/Articulation/Body3"],
                expected_joint_keys=["joint_1", "/Articulation/Joint2", "/Articulation/Joint3"],
            )

    @unittest.skipUnless(USD_AVAILABLE, "Requires usd-core")
    def test_env_cloning(self):
        builder_no_cloning = newton.ModelBuilder()
        builder_cloning = newton.ModelBuilder()
        builder_no_cloning.add_usd(
            os.path.join(os.path.dirname(__file__), "assets", "ant_multi.usda"),
            collapse_fixed_joints=True,
        )
        builder_cloning.add_usd(
            os.path.join(os.path.dirname(__file__), "assets", "ant_multi.usda"),
            collapse_fixed_joints=True,
            cloned_world="/World/envs/env_0",
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

        _ = builder.add_usd(
            os.path.join(os.path.dirname(__file__), "assets", "ant.usda"),
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
        import_results = builder.add_usd(
            os.path.join(os.path.dirname(__file__), "assets", "cube_cylinder.usda"),
            collapse_fixed_joints=True,
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
        builder.add_usd(stage, mesh_maxhullvert=4)

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
        self.assertEqual(builder.shape_type[2], newton.GeoType.SPHERE)
        self.assertAlmostEqual(builder.shape_scale[2][0], wp.length(scale))
        assert_np_equal(np.array(builder.shape_transform[2].p), np.array(tf.p), tol=1.0e-4)

        # bounding box
        assert_np_equal(npsorted(builder.shape_scale[3]), npsorted(scale), tol=1.0e-5)
        # only compare the position since the rotation is not guaranteed to be the same
        assert_np_equal(np.array(builder.shape_transform[3].p), np.array(tf.p), tol=1.0e-4)

    @unittest.skipUnless(USD_AVAILABLE, "Requires usd-core")
    def test_visual_match_collision_shapes(self):
        builder = newton.ModelBuilder()
        builder.add_usd(newton.examples.get_asset("humanoid.usda"))
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
            self.assertFalse(builder.shape_flags[vi] & newton.ShapeFlags.COLLIDE_SHAPES)
            self.assertTrue(builder.shape_flags[ci] & newton.ShapeFlags.COLLIDE_SHAPES)

    @unittest.skipUnless(USD_AVAILABLE, "Requires usd-core")
    def test_non_symmetric_inertia(self):
        """Test importing USD with inertia specified in principal axes that don't align with body frame."""
        from pxr import Gf, Usd, UsdGeom, UsdPhysics  # noqa: PLC0415

        # Create USD stage
        stage = Usd.Stage.CreateInMemory()
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)

        # Create box and apply physics APIs
        box = UsdGeom.Cube.Define(stage, "/World/Box")
        UsdPhysics.CollisionAPI.Apply(box.GetPrim())
        UsdPhysics.RigidBodyAPI.Apply(box.GetPrim())
        mass_api = UsdPhysics.MassAPI.Apply(box.GetPrim())

        # Set mass
        mass_api.CreateMassAttr().Set(1.0)

        # Set diagonal inertia in principal axes frame
        # Principal moments: [2, 4, 6] kg⋅m²
        mass_api.CreateDiagonalInertiaAttr().Set(Gf.Vec3f(2.0, 4.0, 6.0))

        # Set principal axes rotated from body frame
        # Rotate 45° around Z, then 30° around Y
        # Hardcoded quaternion values for this rotation
        q = wp.quat(0.1830127, 0.1830127, 0.6830127, 0.6830127)
        R = np.array(wp.quat_to_matrix(q)).reshape(3, 3)

        # Set principal axes using quaternion
        mass_api.CreatePrincipalAxesAttr().Set(Gf.Quatf(q.w, q.x, q.y, q.z))

        # Parse USD
        builder = newton.ModelBuilder()
        builder.add_usd(stage)

        # Verify parsing
        self.assertEqual(builder.body_count, 1)
        self.assertEqual(builder.shape_count, 1)
        self.assertAlmostEqual(builder.body_mass[0], 1.0, places=6)
        self.assertEqual(builder.body_key[0], "/World/Box")
        self.assertEqual(builder.shape_key[0], "/World/Box")

        # Ensure the body has a free joint assigned and is in an articulation
        self.assertEqual(builder.joint_count, 1)
        self.assertEqual(builder.joint_type[0], newton.JointType.FREE)
        self.assertEqual(builder.joint_parent[0], -1)
        self.assertEqual(builder.joint_child[0], 0)
        self.assertEqual(builder.articulation_count, 1)
        self.assertEqual(builder.articulation_key[0], "/World/Box")

        # Get parsed inertia tensor
        inertia_parsed = np.array(builder.body_inertia[0])

        # Calculate expected inertia tensor in body frame
        # I_body = R * I_principal * R^T
        I_principal = np.diag([2.0, 4.0, 6.0])
        I_body_expected = R @ I_principal @ R.T

        # Verify the parsed inertia matches our calculated body frame inertia
        np.testing.assert_allclose(inertia_parsed.reshape(3, 3), I_body_expected, rtol=1e-5, atol=1e-8)

    @unittest.skipUnless(USD_AVAILABLE, "Requires usd-core")
    def test_force_limits(self):
        """Test importing USD with force limits specified."""
        from pxr import Usd, UsdGeom, UsdPhysics  # noqa: PLC0415

        stage = Usd.Stage.CreateInMemory()
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
        UsdGeom.SetStageMetersPerUnit(stage, 1.0)
        self.assertTrue(stage)

        bodies = {}
        for name, is_root in [("A", True), ("B", False), ("C", False), ("D", False)]:
            path = f"/{name}"
            body = UsdGeom.Xform.Define(stage, path)
            UsdPhysics.RigidBodyAPI.Apply(body.GetPrim())
            if is_root:
                UsdPhysics.ArticulationRootAPI.Apply(body.GetPrim())
            mass_api = UsdPhysics.MassAPI.Apply(body.GetPrim())
            mass_api.CreateMassAttr().Set(1.0)
            mass_api.CreateDiagonalInertiaAttr().Set((1.0, 1.0, 1.0))
            bodies[name] = body

        # Common drive parameters
        default_stiffness = 100.0
        default_damping = 10.0

        joint_configs = {
            "/joint_AB": {
                "type": UsdPhysics.RevoluteJoint,
                "bodies": ["A", "B"],
                "drive_type": "angular",
                "max_force": 24.0,
            },
            "/joint_AC": {
                "type": UsdPhysics.PrismaticJoint,
                "bodies": ["A", "C"],
                "axis": "Z",
                "drive_type": "linear",
                "max_force": 15.0,
            },
            "/joint_AD": {
                "type": UsdPhysics.Joint,
                "bodies": ["A", "D"],
                "limits": {"transX": {"low": -1.0, "high": 1.0}},
                "drive_type": "transX",
                "max_force": 30.0,
            },
        }

        joints = {}
        for path, config in joint_configs.items():
            joint = config["type"].Define(stage, path)

            if "axis" in config:
                joint.CreateAxisAttr().Set(config["axis"])

            if "limits" in config:
                for dof, limits in config["limits"].items():
                    limit_api = UsdPhysics.LimitAPI.Apply(joint.GetPrim(), dof)
                    limit_api.CreateLowAttr().Set(limits["low"])
                    limit_api.CreateHighAttr().Set(limits["high"])

            # Set bodies using names from config
            joint.CreateBody0Rel().SetTargets([bodies[config["bodies"][0]].GetPrim().GetPath()])
            joint.CreateBody1Rel().SetTargets([bodies[config["bodies"][1]].GetPrim().GetPath()])

            # Apply drive with default stiffness/damping
            drive_api = UsdPhysics.DriveAPI.Apply(joint.GetPrim(), config["drive_type"])
            drive_api.CreateStiffnessAttr().Set(default_stiffness)
            drive_api.CreateDampingAttr().Set(default_damping)
            drive_api.CreateMaxForceAttr().Set(config["max_force"])

            joints[path] = joint

        builder = newton.ModelBuilder()
        builder.add_usd(stage)

        model = builder.finalize()

        # Test revolute joint (A-B)
        joint_idx = model.joint_key.index("/joint_AB")
        self.assertEqual(model.joint_type.numpy()[joint_idx], newton.JointType.REVOLUTE)
        joint_dof_idx = model.joint_qd_start.numpy()[joint_idx]
        self.assertEqual(model.joint_effort_limit.numpy()[joint_dof_idx], 24.0)

        # Test prismatic joint (A-C)
        joint_idx_AC = model.joint_key.index("/joint_AC")
        self.assertEqual(model.joint_type.numpy()[joint_idx_AC], newton.JointType.PRISMATIC)
        joint_dof_idx_AC = model.joint_qd_start.numpy()[joint_idx_AC]
        self.assertEqual(model.joint_effort_limit.numpy()[joint_dof_idx_AC], 15.0)

        # Test D6 joint (A-D) - check transX DOF
        joint_idx_AD = model.joint_key.index("/joint_AD")
        self.assertEqual(model.joint_type.numpy()[joint_idx_AD], newton.JointType.D6)
        joint_dof_idx_AD = model.joint_qd_start.numpy()[joint_idx_AD]
        self.assertEqual(model.joint_effort_limit.numpy()[joint_dof_idx_AD], 30.0)

    @unittest.skipUnless(USD_AVAILABLE, "Requires usd-core")
    def test_solimplimit_parsing(self):
        """Test that solimplimit attribute is parsed correctly from USD."""
        from pxr import Usd  # noqa: PLC0415

        # Create USD stage with multiple single-DOF revolute joints
        usd_content = """#usda 1.0
(
    upAxis = "Z"
)

def PhysicsScene "physicsScene"
{
}

def Xform "Articulation" (
    prepend apiSchemas = ["PhysicsArticulationRootAPI"]
)
{
    def Xform "Body1" (
        prepend apiSchemas = ["PhysicsRigidBodyAPI"]
    )
    {
        double3 xformOp:translate = (0, 0, 0)
        uniform token[] xformOpOrder = ["xformOp:translate"]

        def Cube "Collision1" (
            prepend apiSchemas = ["PhysicsCollisionAPI"]
        )
        {
            double size = 0.2
        }
    }

    def PhysicsRevoluteJoint "Joint1" (
        prepend apiSchemas = ["PhysicsDriveAPI:angular"]
    )
    {
        rel physics:body0 = </Articulation/Body1>
        point3f physics:localPos0 = (0, 0, 0)
        point3f physics:localPos1 = (0, 0, 0)
        quatf physics:localRot0 = (1, 0, 0, 0)
        quatf physics:localRot1 = (1, 0, 0, 0)
        token physics:axis = "X"
        float physics:lowerLimit = -90
        float physics:upperLimit = 90

        # MuJoCo solimplimit attribute (5 elements)
        uniform double[] mjc:solimplimit = [0.89, 0.9, 0.01, 2.1, 1.8]
    }

    def Xform "Body2" (
        prepend apiSchemas = ["PhysicsRigidBodyAPI"]
    )
    {
        double3 xformOp:translate = (1, 0, 0)
        uniform token[] xformOpOrder = ["xformOp:translate"]

        def Sphere "Collision2" (
            prepend apiSchemas = ["PhysicsCollisionAPI"]
        )
        {
            double radius = 0.1
        }
    }

    def PhysicsRevoluteJoint "Joint2" (
        prepend apiSchemas = ["PhysicsDriveAPI:angular"]
    )
    {
        rel physics:body0 = </Articulation/Body1>
        rel physics:body1 = </Articulation/Body2>
        point3f physics:localPos0 = (0, 0, 0)
        point3f physics:localPos1 = (0, 0, 0)
        quatf physics:localRot0 = (1, 0, 0, 0)
        quatf physics:localRot1 = (1, 0, 0, 0)
        token physics:axis = "Z"
        float physics:lowerLimit = -180
        float physics:upperLimit = 180

        # No solimplimit - should use defaults
    }
}
"""
        stage = Usd.Stage.CreateInMemory()
        stage.GetRootLayer().ImportFromString(usd_content)

        builder = newton.ModelBuilder()
        SolverMuJoCo.register_custom_attributes(builder)
        builder.add_usd(stage)
        model = builder.finalize()

        # Check if solimplimit custom attribute exists
        self.assertTrue(hasattr(model, "mujoco"), "Model should have mujoco namespace for custom attributes")
        self.assertTrue(hasattr(model.mujoco, "solimplimit"), "Model should have solimplimit attribute")

        solimplimit = model.mujoco.solimplimit.numpy()

        # Should have 2 joints: Joint1 (world to Body1) and Joint2 (Body1 to Body2)
        self.assertEqual(model.joint_count, 2, "Should have 2 single-DOF joints")

        # Helper to check if two arrays match within tolerance
        def arrays_match(arr, expected, tol=1e-4):
            return all(abs(arr[i] - expected[i]) < tol for i in range(len(expected)))

        # Expected values
        expected_joint1 = [0.89, 0.9, 0.01, 2.1, 1.8]  # from Joint1
        expected_joint2 = [0.9, 0.95, 0.001, 0.5, 2.0]  # from Joint2 (default values)

        # Check that both expected solimplimit values are present in the model
        num_dofs = solimplimit.shape[0]
        found_values = [solimplimit[i, :].tolist() for i in range(num_dofs)]

        found_joint1 = any(arrays_match(val, expected_joint1) for val in found_values)
        found_joint2 = any(arrays_match(val, expected_joint2) for val in found_values)

        self.assertTrue(found_joint1, f"Expected solimplimit {expected_joint1} not found in model")
        self.assertTrue(found_joint2, f"Expected default solimplimit {expected_joint2} not found in model")

    def test_limit_margin_parsing(self):
        """Test importing limit_margin from USD with mjc:margin on joint."""
        from pxr import Sdf, Usd, UsdGeom, UsdPhysics  # noqa: PLC0415

        stage = Usd.Stage.CreateInMemory()
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
        UsdGeom.SetStageMetersPerUnit(stage, 1.0)

        # Create first body with joint
        body1_path = "/body1"
        shape1 = UsdGeom.Cube.Define(stage, body1_path)
        prim1 = shape1.GetPrim()
        UsdPhysics.RigidBodyAPI.Apply(prim1)
        UsdPhysics.ArticulationRootAPI.Apply(prim1)
        UsdPhysics.CollisionAPI.Apply(prim1)

        joint1_path = "/joint1"
        joint1 = UsdPhysics.RevoluteJoint.Define(stage, joint1_path)
        joint1.CreateAxisAttr().Set("Z")
        joint1.CreateBody0Rel().SetTargets([body1_path])
        joint1_prim = joint1.GetPrim()
        joint1_prim.CreateAttribute("mjc:margin", Sdf.ValueTypeNames.FloatArray, True).Set([0.01])

        # Create second body with joint
        body2_path = "/body2"
        shape2 = UsdGeom.Cube.Define(stage, body2_path)
        prim2 = shape2.GetPrim()
        UsdPhysics.RigidBodyAPI.Apply(prim2)
        UsdPhysics.CollisionAPI.Apply(prim2)

        joint2_path = "/joint2"
        joint2 = UsdPhysics.RevoluteJoint.Define(stage, joint2_path)
        joint2.CreateAxisAttr().Set("Z")
        joint2.CreateBody0Rel().SetTargets([body1_path])
        joint2.CreateBody1Rel().SetTargets([body2_path])
        joint2_prim = joint2.GetPrim()
        joint2_prim.CreateAttribute("mjc:margin", Sdf.ValueTypeNames.FloatArray, True).Set([0.02])

        # Create third body with joint (no margin, should default to 0.0)
        body3_path = "/body3"
        shape3 = UsdGeom.Cube.Define(stage, body3_path)
        prim3 = shape3.GetPrim()
        UsdPhysics.RigidBodyAPI.Apply(prim3)
        UsdPhysics.CollisionAPI.Apply(prim3)

        joint3_path = "/joint3"
        joint3 = UsdPhysics.RevoluteJoint.Define(stage, joint3_path)
        joint3.CreateAxisAttr().Set("Z")
        joint3.CreateBody0Rel().SetTargets([body2_path])
        joint3.CreateBody1Rel().SetTargets([body3_path])

        builder = newton.ModelBuilder()
        SolverMuJoCo.register_custom_attributes(builder)
        builder.add_usd(stage)
        model = builder.finalize()

        self.assertTrue(hasattr(model, "mujoco"))
        self.assertTrue(hasattr(model.mujoco, "limit_margin"))
        np.testing.assert_allclose(model.mujoco.limit_margin.numpy(), [0.01, 0.02, 0.0])


class TestImportSampleAssets(unittest.TestCase):
    def verify_usdphysics_parser(self, file, model, compare_min_max_coords, floating):
        """Verify model based on the UsdPhysics Parsing Utils"""
        # [1] https://openusd.org/release/api/usd_physics_page_front.html
        from pxr import Sdf, Usd, UsdPhysics  # noqa: PLC0415

        stage = Usd.Stage.Open(file)
        parsed = UsdPhysics.LoadUsdPhysicsFromRange(stage, ["/"])
        # since the key is generated from USD paths we can assume that keys are unique
        body_key_to_idx = dict(zip(model.body_key, range(model.body_count), strict=False))
        shape_key_to_idx = dict(zip(model.shape_key, range(model.shape_count), strict=False))

        parsed_bodies = list(zip(*parsed[UsdPhysics.ObjectType.RigidBody], strict=False))

        # body presence
        for body_path, _ in parsed_bodies:
            assert body_key_to_idx.get(str(body_path), None) is not None
        self.assertEqual(len(parsed_bodies), model.body_count)

        # body colliders
        # TODO: exclude or handle bodies that have child shapes
        for body_path, body_desc in parsed_bodies:
            body_idx = body_key_to_idx.get(str(body_path), None)

            model_collisions = {model.shape_key[sk] for sk in model.body_shapes[body_idx]}
            parsed_collisions = {str(collider) for collider in body_desc.collisions}
            self.assertEqual(parsed_collisions, model_collisions)

        # body mass properties
        body_mass = model.body_mass.numpy()
        body_inertia = model.body_inertia.numpy()
        # in newton, only rigid bodies have mass
        for body_path, _body_desc in parsed_bodies:
            body_idx = body_key_to_idx.get(str(body_path), None)
            prim = stage.GetPrimAtPath(body_path)
            if prim.HasAPI(UsdPhysics.MassAPI):
                mass_api = UsdPhysics.MassAPI(prim)
                # Parents' explicit total masses override any mass properties specified further down in the subtree. [1]
                if mass_api.GetMassAttr().HasAuthoredValue():
                    mass = mass_api.GetMassAttr().Get()
                    self.assertAlmostEqual(body_mass[body_idx], mass, places=5)
                if mass_api.GetDiagonalInertiaAttr().HasAuthoredValue():
                    diag_inertia = mass_api.GetDiagonalInertiaAttr().Get()
                    principal_axes = mass_api.GetPrincipalAxesAttr().Get().Normalize()
                    p = np.array(wp.quat_to_matrix(wp.quat(*principal_axes.imaginary, principal_axes.real))).reshape(
                        (3, 3)
                    )
                    inertia = p @ np.diag(diag_inertia) @ p.T
                    assert_np_equal(body_inertia[body_idx], inertia, tol=1e-5)
        # Rigid bodies that don't have mass and inertia parameters authored will not be checked
        # TODO: check bodies with CollisionAPI children that have MassAPI specified

        joint_mapping = {
            JointType.PRISMATIC: UsdPhysics.ObjectType.PrismaticJoint,
            JointType.REVOLUTE: UsdPhysics.ObjectType.RevoluteJoint,
            JointType.BALL: UsdPhysics.ObjectType.SphericalJoint,
            JointType.FIXED: UsdPhysics.ObjectType.FixedJoint,
            # JointType.FREE: None,
            JointType.DISTANCE: UsdPhysics.ObjectType.DistanceJoint,
            JointType.D6: UsdPhysics.ObjectType.D6Joint,
        }

        joint_key_to_idx = dict(zip(model.joint_key, range(model.joint_count), strict=False))
        model_joint_type = model.joint_type.numpy()
        joints_found = []

        for joint_type, joint_objtype in joint_mapping.items():
            for joint_path, _joint_desc in list(zip(*parsed.get(joint_objtype, ()), strict=False)):
                joint_idx = joint_key_to_idx.get(str(joint_path), None)
                joints_found.append(joint_idx)
                assert joint_key_to_idx.get(str(joint_path), None) is not None
                assert model_joint_type[joint_idx] == joint_type

        # the parser will insert free joints as parents to floating bodies with nonzero mass
        expected_model_joints = len(joints_found) + 1 if floating else len(joints_found)
        self.assertEqual(model.joint_count, expected_model_joints)

        body_q_array = model.body_q.numpy()
        joint_dof_dim_array = model.joint_dof_dim.numpy()
        body_positions = [body_q_array[i, 0:3].tolist() for i in range(body_q_array.shape[0])]
        body_quaternions = [body_q_array[i, 3:7].tolist() for i in range(body_q_array.shape[0])]

        total_dofs = 0
        for j in range(model.joint_count):
            lin = int(joint_dof_dim_array[j][0])
            ang = int(joint_dof_dim_array[j][1])
            total_dofs += lin + ang
            jt = int(model.joint_type.numpy()[j])

            if jt == JointType.REVOLUTE:
                self.assertEqual((lin, ang), (0, 1), f"{model.joint_key[j]} DOF dim mismatch")
            elif jt == JointType.FIXED:
                self.assertEqual((lin, ang), (0, 0), f"{model.joint_key[j]} DOF dim mismatch")
            elif jt == JointType.FREE:
                self.assertGreater(lin + ang, 0, f"{model.joint_key[j]} expected nonzero DOFs for free joint")
            elif jt == JointType.PRISMATIC:
                self.assertEqual((lin, ang), (1, 0), f"{model.joint_key[j]} DOF dim mismatch")
            elif jt == JointType.BALL:
                self.assertEqual((lin, ang), (0, 3), f"{model.joint_key[j]} DOF dim mismatch")

        self.assertEqual(int(total_dofs), int(model.joint_axis.numpy().shape[0]))
        joint_enabled = model.joint_enabled.numpy()
        self.assertTrue(all(joint_enabled[i] != 0 for i in range(len(joint_enabled))))

        axis_vectors = {
            "X": [1.0, 0.0, 0.0],
            "Y": [0.0, 1.0, 0.0],
            "Z": [0.0, 0.0, 1.0],
        }

        drive_gain_scale = 1.0
        scene = UsdPhysics.Scene.Get(stage, Sdf.Path("/physicsScene"))
        if scene:
            attr = scene.GetPrim().GetAttribute("newton:joint_drive_gains_scaling")
            if attr and attr.HasAuthoredValue():
                drive_gain_scale = float(attr.Get())

        for j, key in enumerate(model.joint_key):
            prim = stage.GetPrimAtPath(key)
            if not prim:
                continue

            dof_index = (
                0 if j <= 0 else sum(int(joint_dof_dim_array[i][0] + joint_dof_dim_array[i][1]) for i in range(j))
            )

            p_rel = prim.GetRelationship("physics:body0")
            c_rel = prim.GetRelationship("physics:body1")
            p_targets = p_rel.GetTargets() if p_rel and p_rel.HasAuthoredTargets() else []
            c_targets = c_rel.GetTargets() if c_rel and c_rel.HasAuthoredTargets() else []

            if len(p_targets) == 1 and len(c_targets) == 1:
                p_path = str(p_targets[0])
                c_path = str(c_targets[0])
                if p_path in body_key_to_idx and c_path in body_key_to_idx:
                    self.assertEqual(int(model.joint_parent.numpy()[j]), body_key_to_idx[p_path])
                    self.assertEqual(int(model.joint_child.numpy()[j]), body_key_to_idx[c_path])

            if prim.IsA(UsdPhysics.RevoluteJoint) or prim.IsA(UsdPhysics.PrismaticJoint):
                axis_attr = prim.GetAttribute("physics:axis")
                axis_tok = axis_attr.Get() if axis_attr and axis_attr.HasAuthoredValue() else None
                if axis_tok:
                    expected_axis = axis_vectors[str(axis_tok)]
                    actual_axis = model.joint_axis.numpy()[dof_index].tolist()

                    self.assertTrue(
                        all(abs(actual_axis[i] - expected_axis[i]) < 1e-6 for i in range(3))
                        or all(abs(actual_axis[i] - (-expected_axis[i])) < 1e-6 for i in range(3))
                    )

                lower_attr = prim.GetAttribute("physics:lowerLimit")
                upper_attr = prim.GetAttribute("physics:upperLimit")
                lower = lower_attr.Get() if lower_attr and lower_attr.HasAuthoredValue() else None
                upper = upper_attr.Get() if upper_attr and upper_attr.HasAuthoredValue() else None

                if prim.IsA(UsdPhysics.RevoluteJoint):
                    if lower is not None:
                        self.assertAlmostEqual(
                            float(model.joint_limit_lower.numpy()[dof_index]), math.radians(lower), places=5
                        )
                    if upper is not None:
                        self.assertAlmostEqual(
                            float(model.joint_limit_upper.numpy()[dof_index]), math.radians(upper), places=5
                        )
                else:
                    if lower is not None:
                        self.assertAlmostEqual(
                            float(model.joint_limit_lower.numpy()[dof_index]), float(lower), places=5
                        )
                    if upper is not None:
                        self.assertAlmostEqual(
                            float(model.joint_limit_upper.numpy()[dof_index]), float(upper), places=5
                        )

            if prim.IsA(UsdPhysics.RevoluteJoint):
                ke_attr = prim.GetAttribute("drive:angular:physics:stiffness")
                kd_attr = prim.GetAttribute("drive:angular:physics:damping")
            elif prim.IsA(UsdPhysics.PrismaticJoint):
                ke_attr = prim.GetAttribute("drive:linear:physics:stiffness")
                kd_attr = prim.GetAttribute("drive:linear:physics:damping")
            else:
                ke_attr = kd_attr = None

            if ke_attr:
                ke_val = ke_attr.Get() if ke_attr.HasAuthoredValue() else None
                if ke_val is not None:
                    ke = float(ke_val)
                    self.assertAlmostEqual(
                        float(model.joint_target_ke.numpy()[dof_index]), ke * math.degrees(drive_gain_scale), places=2
                    )

            if kd_attr:
                kd_val = kd_attr.Get() if kd_attr.HasAuthoredValue() else None
                if kd_val is not None:
                    kd = float(kd_val)
                    self.assertAlmostEqual(
                        float(model.joint_target_kd.numpy()[dof_index]), kd * math.degrees(drive_gain_scale), places=2
                    )

        if compare_min_max_coords:
            joint_X_p_array = model.joint_X_p.numpy()
            joint_X_c_array = model.joint_X_c.numpy()
            joint_X_p_positions = [joint_X_p_array[i, 0:3].tolist() for i in range(joint_X_p_array.shape[0])]
            joint_X_p_quaternions = [joint_X_p_array[i, 3:7].tolist() for i in range(joint_X_p_array.shape[0])]
            joint_X_c_positions = [joint_X_c_array[i, 0:3].tolist() for i in range(joint_X_c_array.shape[0])]
            joint_X_c_quaternions = [joint_X_c_array[i, 3:7].tolist() for i in range(joint_X_c_array.shape[0])]

            for j in range(model.joint_count):
                p = int(model.joint_parent.numpy()[j])
                c = int(model.joint_child.numpy()[j])
                if p < 0 or c < 0:
                    continue

                parent_tf = wp.transform(wp.vec3(*body_positions[p]), wp.quat(*body_quaternions[p]))
                child_tf = wp.transform(wp.vec3(*body_positions[c]), wp.quat(*body_quaternions[c]))
                joint_parent_tf = wp.transform(wp.vec3(*joint_X_p_positions[j]), wp.quat(*joint_X_p_quaternions[j]))
                joint_child_tf = wp.transform(wp.vec3(*joint_X_c_positions[j]), wp.quat(*joint_X_c_quaternions[j]))

                lhs_tf = wp.transform_multiply(parent_tf, joint_parent_tf)
                rhs_tf = wp.transform_multiply(child_tf, joint_child_tf)

                lhs_p = wp.transform_get_translation(lhs_tf)
                rhs_p = wp.transform_get_translation(rhs_tf)
                lhs_q = wp.transform_get_rotation(lhs_tf)
                rhs_q = wp.transform_get_rotation(rhs_tf)

                self.assertTrue(all(abs(lhs_p[i] - rhs_p[i]) < 1e-6 for i in range(3)))

                q_diff = lhs_q * wp.quat_inverse(rhs_q)
                angle_diff = 2.0 * math.acos(min(1.0, abs(q_diff[3])))
                self.assertLessEqual(angle_diff, 1e-3)

        model.shape_body.numpy()
        shape_type_array = model.shape_type.numpy()
        shape_transform_array = model.shape_transform.numpy()
        shape_scale_array = model.shape_scale.numpy()
        shape_flags_array = model.shape_flags.numpy()

        shape_to_path = {}
        usd_shape_specs = {}

        shape_type_mapping = {
            newton.GeoType.BOX: UsdPhysics.ObjectType.CubeShape,
            newton.GeoType.SPHERE: UsdPhysics.ObjectType.SphereShape,
            newton.GeoType.CAPSULE: UsdPhysics.ObjectType.CapsuleShape,
            newton.GeoType.CYLINDER: UsdPhysics.ObjectType.CylinderShape,
            newton.GeoType.CONE: UsdPhysics.ObjectType.ConeShape,
            newton.GeoType.MESH: UsdPhysics.ObjectType.MeshShape,
            newton.GeoType.PLANE: UsdPhysics.ObjectType.PlaneShape,
            newton.GeoType.CONVEX_MESH: UsdPhysics.ObjectType.MeshShape,
        }

        for _shape_type, shape_objtype in shape_type_mapping.items():
            if shape_objtype not in parsed:
                continue
            for xpath, shape_spec in zip(*parsed[shape_objtype], strict=False):
                path = str(xpath)
                if path in shape_key_to_idx:
                    sid = shape_key_to_idx[path]
                    # Skip if already processed (e.g., CONVEX_MESH already matched via MESH)
                    if sid in shape_to_path:
                        continue
                    shape_to_path[sid] = path
                    usd_shape_specs[sid] = shape_spec
                    # Check that Newton's shape type maps to the correct USD type
                    newton_type = newton.GeoType(shape_type_array[sid])
                    expected_usd_type = shape_type_mapping.get(newton_type)
                    self.assertEqual(
                        expected_usd_type,
                        shape_objtype,
                        f"Shape {sid} type mismatch: Newton type {newton_type} should map to USD {expected_usd_type}, but found {shape_objtype}",
                    )

        def from_gfquat(gfquat):
            return wp.normalize(wp.quat(*gfquat.imaginary, gfquat.real))

        def quaternions_match(q1, q2, tolerance=1e-5):
            return all(abs(q1[i] - q2[i]) < tolerance for i in range(4)) or all(
                abs(q1[i] + q2[i]) < tolerance for i in range(4)
            )

        for sid, path in shape_to_path.items():
            prim = stage.GetPrimAtPath(path)
            shape_spec = usd_shape_specs[sid]
            newton_type = shape_type_array[sid]
            newton_transform = shape_transform_array[sid]
            newton_scale = shape_scale_array[sid]
            newton_flags = shape_flags_array[sid]

            collision_enabled_usd = True
            if prim.HasAPI(UsdPhysics.CollisionAPI):
                attr = prim.GetAttribute("physics:collisionEnabled")
                if attr and attr.HasAuthoredValue():
                    collision_enabled_usd = attr.Get()

            collision_enabled_newton = bool(newton_flags & int(newton.ShapeFlags.COLLIDE_SHAPES))
            self.assertEqual(
                collision_enabled_newton,
                collision_enabled_usd,
                f"Shape {sid} collision mismatch: USD={collision_enabled_usd}, Newton={collision_enabled_newton}",
            )

            usd_quat = from_gfquat(shape_spec.localRot)
            newton_pos = newton_transform[:3]
            newton_quat = newton_transform[3:7]

            for i, (n_pos, u_pos) in enumerate(zip(newton_pos, shape_spec.localPos, strict=False)):
                self.assertAlmostEqual(
                    n_pos, u_pos, places=5, msg=f"Shape {sid} position[{i}]: USD={u_pos}, Newton={n_pos}"
                )

            if newton_type in [3, 5]:
                usd_axis = int(shape_spec.axis) if hasattr(shape_spec, "axis") else 2
                axis_quat = (
                    quat_between_axes(newton.Axis.Z, newton.Axis.X)
                    if usd_axis == 0
                    else quat_between_axes(newton.Axis.Z, newton.Axis.Y)
                    if usd_axis == 1
                    else wp.quat_identity()
                )
                expected_quat = wp.mul(usd_quat, axis_quat)
            else:
                expected_quat = usd_quat

            if not quaternions_match(newton_quat, expected_quat):
                q_diff = wp.mul(newton_quat, wp.quat_inverse(expected_quat))
                angle_diff = 2.0 * math.acos(min(1.0, abs(q_diff[3])))
                self.fail(
                    f"Shape {sid} rotation mismatch: expected={expected_quat}, Newton={newton_quat}, angle_diff={math.degrees(angle_diff)}°"
                )

            if newton_type == newton.GeoType.CAPSULE:
                self.assertAlmostEqual(newton_scale[0], shape_spec.radius, places=5)
                self.assertAlmostEqual(newton_scale[1], shape_spec.halfHeight, places=5)
            elif newton_type == newton.GeoType.BOX:
                for i, (n_scale, u_extent) in enumerate(zip(newton_scale, shape_spec.halfExtents, strict=False)):
                    self.assertAlmostEqual(
                        n_scale, u_extent, places=5, msg=f"Box {sid} extent[{i}]: USD={u_extent}, Newton={n_scale}"
                    )
            elif newton_type == newton.GeoType.SPHERE:
                self.assertAlmostEqual(newton_scale[0], shape_spec.radius, places=5)
            elif newton_type == newton.GeoType.CYLINDER:
                self.assertAlmostEqual(newton_scale[0], shape_spec.radius, places=5)
                self.assertAlmostEqual(newton_scale[1], shape_spec.halfHeight, places=5)

    @unittest.skipUnless(USD_AVAILABLE, "Requires usd-core")
    def test_ant(self):
        builder = newton.ModelBuilder()

        asset_path = newton.examples.get_asset("ant.usda")
        builder.add_usd(
            asset_path,
            collapse_fixed_joints=False,
            enable_self_collisions=False,
            load_sites=False,
            load_visual_shapes=False,
        )
        model = builder.finalize()
        self.verify_usdphysics_parser(asset_path, model, compare_min_max_coords=True, floating=True)

    @unittest.skipUnless(USD_AVAILABLE, "Requires usd-core")
    def test_anymal(self):
        builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
        asset_root = newton.utils.download_asset("anybotics_anymal_d/usd")
        stage_path = None
        for root, _, files in os.walk(asset_root):
            if "anymal_d.usda" in files:
                stage_path = os.path.join(root, "anymal_d.usda")
                break
        if not stage_path or not os.path.exists(stage_path):
            raise unittest.SkipTest(f"Stage file not found: {stage_path}")

        builder.add_usd(
            stage_path,
            collapse_fixed_joints=False,
            enable_self_collisions=False,
            load_sites=False,
            load_visual_shapes=False,
        )
        model = builder.finalize()
        self.verify_usdphysics_parser(stage_path, model, True, floating=True)

    @unittest.skipUnless(USD_AVAILABLE, "Requires usd-core")
    def test_cartpole(self):
        builder = newton.ModelBuilder()

        asset_path = newton.examples.get_asset("cartpole.usda")
        builder.add_usd(
            asset_path,
            collapse_fixed_joints=False,
            enable_self_collisions=False,
            load_sites=False,
            load_visual_shapes=False,
        )
        model = builder.finalize()
        self.verify_usdphysics_parser(asset_path, model, compare_min_max_coords=True, floating=False)

    @unittest.skipUnless(USD_AVAILABLE, "Requires usd-core")
    def test_g1(self):
        builder = newton.ModelBuilder()
        asset_path = str(newton.utils.download_asset("unitree_g1/usd") / "g1_isaac.usd")

        builder.add_usd(
            asset_path,
            collapse_fixed_joints=False,
            enable_self_collisions=False,
            load_sites=False,
            load_visual_shapes=False,
        )
        model = builder.finalize()
        self.verify_usdphysics_parser(asset_path, model, compare_min_max_coords=False, floating=True)

    @unittest.skipUnless(USD_AVAILABLE, "Requires usd-core")
    def test_h1(self):
        builder = newton.ModelBuilder()
        asset_path = str(newton.utils.download_asset("unitree_h1/usd") / "h1_minimal.usda")

        builder.add_usd(
            asset_path,
            collapse_fixed_joints=False,
            enable_self_collisions=False,
            load_sites=False,
            load_visual_shapes=False,
        )
        model = builder.finalize()
        self.verify_usdphysics_parser(asset_path, model, compare_min_max_coords=True, floating=True)

    @unittest.skipUnless(USD_AVAILABLE, "Requires usd-core")
    def test_granular_loading_flags(self):
        """Test the granular control over sites and visual shapes loading."""
        from pxr import Usd  # noqa: PLC0415

        # Create USD stage in memory with sites, collision, and visual shapes
        usd_content = """#usda 1.0
(
    upAxis = "Z"
)

def PhysicsScene "physicsScene"
{
}

def Xform "TestBody" (
    prepend apiSchemas = ["PhysicsRigidBodyAPI"]
)
{
    double3 xformOp:translate = (0, 0, 1)
    uniform token[] xformOpOrder = ["xformOp:translate"]

    def Cube "CollisionBox" (
        prepend apiSchemas = ["PhysicsCollisionAPI"]
    )
    {
        double size = 1.0
    }

    def Sphere "VisualSphere"
    {
        double radius = 0.3
        double3 xformOp:translate = (1, 0, 0)
        uniform token[] xformOpOrder = ["xformOp:translate"]
    }

    def Sphere "Site1" (
        prepend apiSchemas = ["MjcSiteAPI"]
    )
    {
        double radius = 0.1
        double3 xformOp:translate = (0, 1, 0)
        uniform token[] xformOpOrder = ["xformOp:translate"]
    }

    def Cube "Site2" (
        prepend apiSchemas = ["MjcSiteAPI"]
    )
    {
        double size = 0.2
        double3 xformOp:translate = (0, -1, 0)
        uniform token[] xformOpOrder = ["xformOp:translate"]
    }
}
"""
        stage = Usd.Stage.CreateInMemory()
        stage.GetRootLayer().ImportFromString(usd_content)

        # Test 1: Load all (default behavior)
        builder_all = newton.ModelBuilder()
        builder_all.add_usd(stage)
        count_all = builder_all.shape_count
        self.assertEqual(count_all, 4, "Should load all shapes: 1 collision + 2 sites + 1 visual = 4")

        # Test 2: Load sites only, no visual shapes
        builder_sites_only = newton.ModelBuilder()
        builder_sites_only.add_usd(stage, load_sites=True, load_visual_shapes=False)
        count_sites_only = builder_sites_only.shape_count
        self.assertEqual(count_sites_only, 3, "Should load collision + sites: 1 collision + 2 sites = 3")

        # Test 3: Load visual shapes only, no sites
        builder_visuals_only = newton.ModelBuilder()
        builder_visuals_only.add_usd(stage, load_sites=False, load_visual_shapes=True)
        count_visuals_only = builder_visuals_only.shape_count
        self.assertEqual(count_visuals_only, 2, "Should load collision + visuals: 1 collision + 1 visual = 2")

        # Test 4: Load neither (physics collision shapes only)
        builder_physics_only = newton.ModelBuilder()
        builder_physics_only.add_usd(stage, load_sites=False, load_visual_shapes=False)
        count_physics_only = builder_physics_only.shape_count
        self.assertEqual(count_physics_only, 1, "Should load collision only: 1 collision = 1")

        # Verify that each filter actually reduces the count
        self.assertLess(count_sites_only, count_all, "Excluding visuals should reduce shape count")
        self.assertLess(count_visuals_only, count_all, "Excluding sites should reduce shape count")
        self.assertLess(count_physics_only, count_all, "Excluding both should reduce shape count most")

    @unittest.skipUnless(USD_AVAILABLE, "Requires usd-core")
    def test_granular_loading_with_sites(self):
        """Test loading control specifically for files with sites."""
        from pxr import Usd  # noqa: PLC0415

        # Create USD stage in memory with sites (MjcSiteAPI)
        usd_content = """#usda 1.0
(
    upAxis = "Z"
)

def PhysicsScene "physicsScene"
{
}

def Xform "TestBody" (
    prepend apiSchemas = ["PhysicsRigidBodyAPI"]
)
{
    double3 xformOp:translate = (0, 0, 1)
    uniform token[] xformOpOrder = ["xformOp:translate"]

    def Cube "CollisionBox" (
        prepend apiSchemas = ["PhysicsCollisionAPI"]
    )
    {
        double size = 1.0
    }

    def Sphere "VisualSphere"
    {
        double radius = 0.3
        double3 xformOp:translate = (1, 0, 0)
        uniform token[] xformOpOrder = ["xformOp:translate"]
    }

    def Sphere "Site1" (
        prepend apiSchemas = ["MjcSiteAPI"]
    )
    {
        double radius = 0.1
        double3 xformOp:translate = (0, 1, 0)
        uniform token[] xformOpOrder = ["xformOp:translate"]
    }

    def Cube "Site2" (
        prepend apiSchemas = ["MjcSiteAPI"]
    )
    {
        double size = 0.2
        double3 xformOp:translate = (0, -1, 0)
        uniform token[] xformOpOrder = ["xformOp:translate"]
    }
}
"""
        stage = Usd.Stage.CreateInMemory()
        stage.GetRootLayer().ImportFromString(usd_content)

        # Load everything and count shape types
        builder_all = newton.ModelBuilder()
        builder_all.add_usd(stage)

        collision_count = sum(
            1
            for i in range(builder_all.shape_count)
            if builder_all.shape_flags[i] & int(newton.ShapeFlags.COLLIDE_SHAPES)
        )
        site_count = sum(
            1 for i in range(builder_all.shape_count) if builder_all.shape_flags[i] & int(newton.ShapeFlags.SITE)
        )
        visual_count = builder_all.shape_count - collision_count - site_count

        # Verify the test asset has all three types
        self.assertGreater(collision_count, 0, "Test asset should have collision shapes")
        self.assertGreater(site_count, 0, "Test asset should have sites")
        self.assertGreater(visual_count, 0, "Test asset should have visual-only shapes")

        # Test sites-only loading
        builder_sites = newton.ModelBuilder()
        builder_sites.add_usd(stage, load_sites=True, load_visual_shapes=False)
        sites_in_result = sum(
            1 for i in range(builder_sites.shape_count) if builder_sites.shape_flags[i] & int(newton.ShapeFlags.SITE)
        )
        self.assertEqual(sites_in_result, site_count, "load_sites=True should load all sites")
        self.assertEqual(builder_sites.shape_count, collision_count + site_count, "Should have collision + sites only")

        # Test visuals-only loading (no sites)
        builder_visuals = newton.ModelBuilder()
        builder_visuals.add_usd(stage, load_sites=False, load_visual_shapes=True)
        sites_in_visuals = sum(
            1
            for i in range(builder_visuals.shape_count)
            if builder_visuals.shape_flags[i] & int(newton.ShapeFlags.SITE)
        )
        self.assertEqual(sites_in_visuals, 0, "load_sites=False should not load any sites")
        self.assertEqual(
            builder_visuals.shape_count, collision_count + visual_count, "Should have collision + visuals only"
        )

    @unittest.skipUnless(USD_AVAILABLE, "Requires usd-core")
    def test_import_usd_gravcomp(self):
        """Test parsing of gravcomp from USD"""
        from pxr import Sdf, Usd, UsdPhysics  # noqa: PLC0415

        stage = Usd.Stage.CreateInMemory()
        UsdPhysics.Scene.Define(stage, "/physicsScene")

        # Body 1 with gravcomp
        body1_path = "/Body1"
        prim1 = stage.DefinePrim(body1_path, "Xform")
        UsdPhysics.RigidBodyAPI.Apply(prim1)
        UsdPhysics.MassAPI.Apply(prim1)
        attr1 = prim1.CreateAttribute("mjc:gravcomp", Sdf.ValueTypeNames.Float)
        attr1.Set(0.5)

        # Body 2 without gravcomp
        body2_path = "/Body2"
        prim2 = stage.DefinePrim(body2_path, "Xform")
        UsdPhysics.RigidBodyAPI.Apply(prim2)
        UsdPhysics.MassAPI.Apply(prim2)

        builder = newton.ModelBuilder()
        SolverMuJoCo.register_custom_attributes(builder)
        builder.add_usd(stage)
        model = builder.finalize()

        self.assertTrue(hasattr(model, "mujoco"))
        self.assertTrue(hasattr(model.mujoco, "gravcomp"))

        gravcomp = model.mujoco.gravcomp.numpy()
        self.assertEqual(len(gravcomp), 2)

        # Check that we have one body with 0.5 and one with 0.0
        # Use assertIn/list checking since order is not strictly guaranteed without path map
        self.assertTrue(np.any(np.isclose(gravcomp, 0.5)))
        self.assertTrue(np.any(np.isclose(gravcomp, 0.0)))

    @unittest.skipUnless(USD_AVAILABLE, "Requires usd-core")
    def test_joint_stiffness_damping(self):
        """Test that joint stiffness and damping are parsed correctly from USD."""
        from pxr import Usd  # noqa: PLC0415

        usd_content = """#usda 1.0
(
    upAxis = "Z"
)

def PhysicsScene "physicsScene"
{
}

def Xform "Articulation" (
    prepend apiSchemas = ["PhysicsArticulationRootAPI"]
)
{
    def Xform "Body1" (
        prepend apiSchemas = ["PhysicsRigidBodyAPI"]
    )
    {
        double3 xformOp:translate = (0, 0, 1)
        uniform token[] xformOpOrder = ["xformOp:translate"]

        def Cube "Collision1" (
            prepend apiSchemas = ["PhysicsCollisionAPI"]
        )
        {
            double size = 0.2
        }
    }

    def PhysicsRevoluteJoint "Joint1" (
        prepend apiSchemas = ["PhysicsDriveAPI:angular"]
    )
    {
        rel physics:body0 = </Articulation/Body1>
        point3f physics:localPos0 = (0, 0, 0)
        point3f physics:localPos1 = (0, 0, 0)
        quatf physics:localRot0 = (1, 0, 0, 0)
        quatf physics:localRot1 = (1, 0, 0, 0)
        token physics:axis = "Z"
        float physics:lowerLimit = -45
        float physics:upperLimit = 45
        float mjc:stiffness = 0.05
        float mjc:damping = 0.5
        float drive:angular:physics:stiffness = 10000.0
        float drive:angular:physics:damping = 2000.0
    }

    def Xform "Body2" (
        prepend apiSchemas = ["PhysicsRigidBodyAPI"]
    )
    {
        double3 xformOp:translate = (1, 0, 1)
        uniform token[] xformOpOrder = ["xformOp:translate"]

        def Sphere "Collision2" (
            prepend apiSchemas = ["PhysicsCollisionAPI"]
        )
        {
            double radius = 0.1
        }
    }

    def PhysicsRevoluteJoint "Joint2" (
        prepend apiSchemas = ["PhysicsDriveAPI:angular"]
    )
    {
        rel physics:body0 = </Articulation/Body1>
        rel physics:body1 = </Articulation/Body2>
        point3f physics:localPos0 = (0, 0, 0)
        point3f physics:localPos1 = (0, 0, 0)
        quatf physics:localRot0 = (1, 0, 0, 0)
        quatf physics:localRot1 = (1, 0, 0, 0)
        token physics:axis = "Y"
        float physics:lowerLimit = -30
        float physics:upperLimit = 30
        float drive:angular:physics:stiffness = 5000.0
        float drive:angular:physics:damping = 1000.0
    }

    def Xform "Body3" (
        prepend apiSchemas = ["PhysicsRigidBodyAPI"]
    )
    {
        double3 xformOp:translate = (2, 0, 1)
        uniform token[] xformOpOrder = ["xformOp:translate"]

        def Sphere "Collision3" (
            prepend apiSchemas = ["PhysicsCollisionAPI"]
        )
        {
            double radius = 0.1
        }
    }

    def PhysicsRevoluteJoint "Joint3"
    {
        rel physics:body0 = </Articulation/Body2>
        rel physics:body1 = </Articulation/Body3>
        point3f physics:localPos0 = (0, 0, 0)
        point3f physics:localPos1 = (0, 0, 0)
        quatf physics:localRot0 = (1, 0, 0, 0)
        quatf physics:localRot1 = (1, 0, 0, 0)
        token physics:axis = "X"
        float physics:lowerLimit = -60
        float physics:upperLimit = 60
        float mjc:stiffness = 0.1
        float mjc:damping = 0.8
    }
}
"""
        stage = Usd.Stage.CreateInMemory()
        stage.GetRootLayer().ImportFromString(usd_content)

        builder = newton.ModelBuilder()
        SolverMuJoCo.register_custom_attributes(builder)
        builder.add_usd(stage)
        model = builder.finalize()

        self.assertTrue(hasattr(model, "mujoco"))
        self.assertTrue(hasattr(model.mujoco, "dof_passive_stiffness"))
        self.assertTrue(hasattr(model.mujoco, "dof_passive_damping"))

        joint_names = model.joint_key
        joint_qd_start = model.joint_qd_start.numpy()
        joint_stiffness = model.mujoco.dof_passive_stiffness.numpy()
        joint_damping = model.mujoco.dof_passive_damping.numpy()
        joint_target_ke = model.joint_target_ke.numpy()
        joint_target_kd = model.joint_target_kd.numpy()

        import math  # noqa: PLC0415

        angular_gain_unit_scale = math.degrees(1.0)
        expected_values = {
            "/Articulation/Joint1": {
                "stiffness": 0.05,
                "damping": 0.5,
                "target_ke": 10000.0 * angular_gain_unit_scale,
                "target_kd": 2000.0 * angular_gain_unit_scale,
            },
            "/Articulation/Joint2": {
                "stiffness": 0.0,
                "damping": 0.0,
                "target_ke": 5000.0 * angular_gain_unit_scale,
                "target_kd": 1000.0 * angular_gain_unit_scale,
            },
            "/Articulation/Joint3": {"stiffness": 0.1, "damping": 0.8, "target_ke": 0.0, "target_kd": 0.0},
        }

        for joint_name, expected in expected_values.items():
            joint_idx = joint_names.index(joint_name)
            dof_idx = joint_qd_start[joint_idx]
            self.assertAlmostEqual(joint_stiffness[dof_idx], expected["stiffness"], places=4)
            self.assertAlmostEqual(joint_damping[dof_idx], expected["damping"], places=4)
            self.assertAlmostEqual(joint_target_ke[dof_idx], expected["target_ke"], places=1)
            self.assertAlmostEqual(joint_target_kd[dof_idx], expected["target_kd"], places=1)

    @unittest.skipUnless(USD_AVAILABLE, "Requires usd-core")
    def test_jnt_actgravcomp_parsing(self):
        """Test that jnt_actgravcomp attribute is parsed correctly from USD."""
        from pxr import Usd  # noqa: PLC0415

        usd_content = """#usda 1.0
(
    upAxis = "Z"
)

def PhysicsScene "physicsScene"
{
}

def Xform "Articulation" (
    prepend apiSchemas = ["PhysicsArticulationRootAPI"]
)
{
    def Xform "Body1" (
        prepend apiSchemas = ["PhysicsRigidBodyAPI"]
    )
    {
        double3 xformOp:translate = (0, 0, 0)
        uniform token[] xformOpOrder = ["xformOp:translate"]

        def Cube "Collision1" (
            prepend apiSchemas = ["PhysicsCollisionAPI"]
        )
        {
            double size = 0.2
        }
    }

    def PhysicsRevoluteJoint "Joint1"
    {
        rel physics:body0 = </Articulation/Body1>
        point3f physics:localPos0 = (0, 0, 0)
        point3f physics:localPos1 = (0, 0, 0)
        quatf physics:localRot0 = (1, 0, 0, 0)
        quatf physics:localRot1 = (1, 0, 0, 0)
        token physics:axis = "Z"

        # MuJoCo actuatorgravcomp attribute
        bool mjc:actuatorgravcomp = true
    }

    def Xform "Body2" (
        prepend apiSchemas = ["PhysicsRigidBodyAPI"]
    )
    {
        double3 xformOp:translate = (1, 0, 0)
        uniform token[] xformOpOrder = ["xformOp:translate"]

        def Sphere "Collision2" (
            prepend apiSchemas = ["PhysicsCollisionAPI"]
        )
        {
            double radius = 0.1
        }
    }

    def PhysicsRevoluteJoint "Joint2"
    {
        rel physics:body0 = </Articulation/Body1>
        rel physics:body1 = </Articulation/Body2>
        point3f physics:localPos0 = (0, 0, 0)
        point3f physics:localPos1 = (0, 0, 0)
        quatf physics:localRot0 = (1, 0, 0, 0)
        quatf physics:localRot1 = (1, 0, 0, 0)
        token physics:axis = "Y"

        # No actuatorgravcomp - should use default (0.0)
    }
}
"""
        stage = Usd.Stage.CreateInMemory()
        stage.GetRootLayer().ImportFromString(usd_content)

        builder = newton.ModelBuilder()
        SolverMuJoCo.register_custom_attributes(builder)
        builder.add_usd(stage)
        model = builder.finalize()

        self.assertTrue(hasattr(model, "mujoco"))
        self.assertTrue(hasattr(model.mujoco, "jnt_actgravcomp"))

        jnt_actgravcomp = model.mujoco.jnt_actgravcomp.numpy()

        # Should have 2 joints
        self.assertEqual(model.joint_count, 2)

        # Find the values - one should be True, one should be False
        self.assertTrue(np.any(jnt_actgravcomp))
        self.assertTrue(np.any(not jnt_actgravcomp))


if __name__ == "__main__":
    unittest.main(verbosity=2, failfast=True)
