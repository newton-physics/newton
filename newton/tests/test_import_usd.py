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
import newton.usd as usd
from newton import JointType
from newton._src.geometry.utils import create_box_mesh, transform_points
from newton.solvers import SolverMuJoCo
from newton.tests.unittest_utils import USD_AVAILABLE, assert_np_equal, get_test_devices
from newton.utils import quat_between_axes

devices = get_test_devices()


class TestImportUsd(unittest.TestCase):
    @unittest.skipUnless(USD_AVAILABLE, "Requires usd-core")
    def test_import_usd_raises_on_stage_errors(self):
        from pxr import Usd  # noqa: PLC0415

        usd_text = """#usda 1.0
def Xform "Root" (
    references = @does_not_exist.usda@
)
{
}
"""
        stage = Usd.Stage.CreateInMemory()
        stage.GetRootLayer().ImportFromString(usd_text)

        builder = newton.ModelBuilder()
        with self.assertRaises(RuntimeError) as exc_info:
            builder.add_usd(stage)

        self.assertIn("composition errors", str(exc_info.exception))

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
    def test_import_non_articulated_joints(self):
        builder = newton.ModelBuilder()

        asset_path = newton.examples.get_asset("boxes_fourbar.usda")
        with self.assertWarns(UserWarning) as cm:
            builder.add_usd(asset_path)
        self.assertIn("No articulation was found but 4 joints were parsed", str(cm.warning))

        self.assertEqual(builder.body_count, 4)
        self.assertEqual(builder.joint_type.count(newton.JointType.REVOLUTE), 4)
        self.assertEqual(builder.joint_type.count(newton.JointType.FREE), 0)
        self.assertTrue(all(art_id == -1 for art_id in builder.joint_articulation))

        # finalize the builder and check the model
        model = builder.finalize(skip_validation_joints=True)
        # note we have to skip joint validation here because otherwise a ValueError would be
        # raised because of the orphan joints that are not part of an articulation
        self.assertEqual(model.body_count, 4)
        self.assertEqual(model.joint_type.list().count(newton.JointType.REVOLUTE), 4)
        self.assertEqual(model.joint_type.list().count(newton.JointType.FREE), 0)
        self.assertTrue(all(art_id == -1 for art_id in model.joint_articulation.numpy()))

    @unittest.skipUnless(USD_AVAILABLE, "Requires usd-core")
    def test_import_disabled_joints_create_free_joints(self):
        from pxr import Gf, Usd, UsdGeom, UsdPhysics  # noqa: PLC0415

        stage = Usd.Stage.CreateInMemory()
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
        UsdPhysics.Scene.Define(stage, "/physicsScene")

        # Regression test: if all joints are disabled (or filtered out), we still
        # need to create free joints for floating bodies so each body has DOFs.
        def define_body(path):
            body = UsdGeom.Xform.Define(stage, path)
            UsdPhysics.RigidBodyAPI.Apply(body.GetPrim())
            return body

        body0 = define_body("/World/Body0")
        body1 = define_body("/World/Body1")

        # The only joint in the stage is explicitly disabled.
        joint = UsdPhysics.RevoluteJoint.Define(stage, "/World/DisabledJoint")
        joint.CreateBody0Rel().SetTargets([body0.GetPath()])
        joint.CreateBody1Rel().SetTargets([body1.GetPath()])
        joint.CreateLocalPos0Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
        joint.CreateLocalPos1Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
        joint.CreateLocalRot0Attr().Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
        joint.CreateLocalRot1Attr().Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
        joint.CreateAxisAttr().Set("Z")
        joint.CreateJointEnabledAttr().Set(False)

        builder = newton.ModelBuilder()
        builder.add_usd(stage)

        # With no enabled joints, we should still get one free joint per body.
        self.assertEqual(builder.body_count, 2)
        self.assertEqual(builder.joint_count, 2)
        self.assertEqual(builder.joint_type.count(newton.JointType.FREE), 2)
        # Each floating body should get its own single-joint articulation.
        self.assertEqual(builder.articulation_count, 2)
        self.assertEqual(set(builder.joint_articulation), {0, 1})

    @unittest.skipUnless(USD_AVAILABLE, "Requires usd-core")
    def test_collapse_fixed_joints_preserves_orphan_joints(self):
        """collapse_fixed_joints must not drop orphan joints or their bodies.

        Regression test: when no PhysicsArticulationRootAPI exists in the USD
        and only body-to-body joints are present, collapse_fixed_joints would
        drop all bodies and joints because the DFS only started from children
        of world (-1), which was empty when free joints were not auto-inserted.
        """
        from pxr import Gf, Usd, UsdGeom, UsdPhysics  # noqa: PLC0415

        stage = Usd.Stage.CreateInMemory()
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
        UsdPhysics.Scene.Define(stage, "/physicsScene")

        # Three bodies connected by revolute joints, NO articulation root
        body_a = UsdGeom.Xform.Define(stage, "/World/BodyA")
        UsdPhysics.RigidBodyAPI.Apply(body_a.GetPrim())
        body_a.AddTranslateOp().Set(Gf.Vec3d(0, 0, 0))
        col_a = UsdGeom.Cube.Define(stage, "/World/BodyA/Collision")
        UsdPhysics.CollisionAPI.Apply(col_a.GetPrim())

        body_b = UsdGeom.Xform.Define(stage, "/World/BodyB")
        UsdPhysics.RigidBodyAPI.Apply(body_b.GetPrim())
        body_b.AddTranslateOp().Set(Gf.Vec3d(1, 0, 0))
        col_b = UsdGeom.Cube.Define(stage, "/World/BodyB/Collision")
        UsdPhysics.CollisionAPI.Apply(col_b.GetPrim())

        body_c = UsdGeom.Xform.Define(stage, "/World/BodyC")
        UsdPhysics.RigidBodyAPI.Apply(body_c.GetPrim())
        body_c.AddTranslateOp().Set(Gf.Vec3d(2, 0, 0))
        col_c = UsdGeom.Cube.Define(stage, "/World/BodyC/Collision")
        UsdPhysics.CollisionAPI.Apply(col_c.GetPrim())

        # Revolute: BodyA -> BodyB (body-to-body, no world connection)
        rev1 = UsdPhysics.RevoluteJoint.Define(stage, "/World/RevJoint1")
        rev1.CreateBody0Rel().SetTargets([body_a.GetPath()])
        rev1.CreateBody1Rel().SetTargets([body_b.GetPath()])
        rev1.CreateLocalPos0Attr().Set(Gf.Vec3f(0.5, 0, 0))
        rev1.CreateLocalPos1Attr().Set(Gf.Vec3f(-0.5, 0, 0))
        rev1.CreateLocalRot0Attr().Set(Gf.Quatf(1, 0, 0, 0))
        rev1.CreateLocalRot1Attr().Set(Gf.Quatf(1, 0, 0, 0))
        rev1.CreateAxisAttr().Set("Z")

        # Revolute: BodyB -> BodyC
        rev2 = UsdPhysics.RevoluteJoint.Define(stage, "/World/RevJoint2")
        rev2.CreateBody0Rel().SetTargets([body_b.GetPath()])
        rev2.CreateBody1Rel().SetTargets([body_c.GetPath()])
        rev2.CreateLocalPos0Attr().Set(Gf.Vec3f(0.5, 0, 0))
        rev2.CreateLocalPos1Attr().Set(Gf.Vec3f(-0.5, 0, 0))
        rev2.CreateLocalRot0Attr().Set(Gf.Quatf(1, 0, 0, 0))
        rev2.CreateLocalRot1Attr().Set(Gf.Quatf(1, 0, 0, 0))
        rev2.CreateAxisAttr().Set("Z")

        builder = newton.ModelBuilder()
        with self.assertWarns(UserWarning):
            builder.add_usd(stage, collapse_fixed_joints=True)

        # All three bodies and both revolute joints must survive collapse
        self.assertEqual(builder.body_count, 3)
        self.assertEqual(builder.joint_count, 2)
        self.assertIn("/World/RevJoint1", builder.joint_key)
        self.assertIn("/World/RevJoint2", builder.joint_key)

    @unittest.skipUnless(USD_AVAILABLE, "Requires usd-core")
    def test_import_articulation_parent_offset(self):
        from pxr import Usd  # noqa: PLC0415

        usd_text = """#usda 1.0
(
    upAxis = "Z"
)
def "World"
{
    def Xform "Env_0"
    {
        double3 xformOp:translate = (0, 0, 0)
        uniform token[] xformOpOrder = ["xformOp:translate"]

        def Xform "Robot" (
            apiSchemas = ["PhysicsArticulationRootAPI"]
        )
        {
            def Xform "Body" (
                apiSchemas = ["PhysicsRigidBodyAPI"]
            )
            {
                double3 xformOp:translate = (0, 0, 0)
                uniform token[] xformOpOrder = ["xformOp:translate"]
            }
        }
    }

    def Xform "Env_1"
    {
        double3 xformOp:translate = (2.5, 0, 0)
        uniform token[] xformOpOrder = ["xformOp:translate"]

        def Xform "Robot" (
            apiSchemas = ["PhysicsArticulationRootAPI"]
        )
        {
            def Xform "Body" (
                apiSchemas = ["PhysicsRigidBodyAPI"]
            )
            {
                double3 xformOp:translate = (0, 0, 0)
                uniform token[] xformOpOrder = ["xformOp:translate"]
            }
        }
    }
}
"""
        stage = Usd.Stage.CreateInMemory()
        stage.GetRootLayer().ImportFromString(usd_text)

        builder = newton.ModelBuilder()
        results = builder.add_usd(stage, xform=wp.transform(wp.vec3(0.0, 0.0, 1.0), wp.quat_identity()))

        body_0 = results["path_body_map"]["/World/Env_0/Robot/Body"]
        body_1 = results["path_body_map"]["/World/Env_1/Robot/Body"]

        pos_0 = np.array(builder.body_q[body_0].p)
        pos_1 = np.array(builder.body_q[body_1].p)

        np.testing.assert_allclose(pos_0, np.array([0.0, 0.0, 1.0]), atol=1e-5)
        np.testing.assert_allclose(pos_1, np.array([2.5, 0.0, 1.0]), atol=1e-5)

    @unittest.skipUnless(USD_AVAILABLE, "Requires usd-core")
    def test_import_scale_ops_units_resolve(self):
        from pxr import Usd  # noqa: PLC0415

        usd_text = """#usda 1.0
(
    upAxis = "Z"
)
def PhysicsScene "physicsScene"
{
}
def Xform "World"
{
    def Xform "Body" (
        prepend apiSchemas = ["PhysicsRigidBodyAPI"]
    )
    {
        def Xform "Scaled"
        {
            float3 xformOp:scale = (2, 2, 2)
            double xformOp:rotateX:unitsResolve = 90
            double3 xformOp:scale:unitsResolve = (0.01, 0.01, 0.01)
            uniform token[] xformOpOrder = ["xformOp:scale", "xformOp:rotateX:unitsResolve", "xformOp:scale:unitsResolve"]

            def Cube "Collision" (
                prepend apiSchemas = ["PhysicsCollisionAPI"]
            )
            {
                double size = 2
            }
        }
    }
}
"""
        stage = Usd.Stage.CreateInMemory()
        stage.GetRootLayer().ImportFromString(usd_text)

        builder = newton.ModelBuilder()
        results = builder.add_usd(stage)

        shape_id = results["path_shape_map"]["/World/Body/Scaled/Collision"]
        assert_np_equal(np.array(builder.shape_scale[shape_id]), np.array([0.02, 0.02, 0.02]), tol=1e-5)

    @unittest.skipUnless(USD_AVAILABLE, "Requires usd-core")
    def test_import_scale_ops_nested_xforms(self):
        from pxr import Usd  # noqa: PLC0415

        usd_text = """#usda 1.0
(
    upAxis = "Z"
)
def PhysicsScene "physicsScene"
{
}
def Xform "World"
{
    def Xform "Body" (
        prepend apiSchemas = ["PhysicsRigidBodyAPI"]
    )
    {
        def Xform "Parent"
        {
            float3 xformOp:scale = (2, 3, 4)
            uniform token[] xformOpOrder = ["xformOp:scale"]

            def Xform "Child"
            {
                float3 xformOp:scale = (0.5, 2, 1.5)
                uniform token[] xformOpOrder = ["xformOp:scale"]

                def Cube "Collision" (
                    prepend apiSchemas = ["PhysicsCollisionAPI"]
                )
                {
                    double size = 2
                }
            }
        }
    }
}
"""
        stage = Usd.Stage.CreateInMemory()
        stage.GetRootLayer().ImportFromString(usd_text)

        builder = newton.ModelBuilder()
        results = builder.add_usd(stage)

        shape_id = results["path_shape_map"]["/World/Body/Parent/Child/Collision"]
        assert_np_equal(np.array(builder.shape_scale[shape_id]), np.array([1.0, 6.0, 6.0]), tol=1e-5)

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
    def test_reversed_joints_in_articulation_raise(self):
        """Ensure reversed joints are reported when encountered in articulations."""
        from pxr import Gf, Usd, UsdGeom, UsdPhysics  # noqa: PLC0415

        stage = Usd.Stage.CreateInMemory()
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
        UsdPhysics.Scene.Define(stage, "/physicsScene")

        articulation = UsdGeom.Xform.Define(stage, "/World/Articulation")
        UsdPhysics.ArticulationRootAPI.Apply(articulation.GetPrim())

        def define_body(path):
            body = UsdGeom.Xform.Define(stage, path)
            UsdPhysics.RigidBodyAPI.Apply(body.GetPrim())
            return body

        body0 = define_body("/World/Articulation/Body0")
        body1 = define_body("/World/Articulation/Body1")
        body2 = define_body("/World/Articulation/Body2")

        joint0 = UsdPhysics.RevoluteJoint.Define(stage, "/World/Articulation/Joint0")
        joint0.CreateBody0Rel().SetTargets([body0.GetPath()])
        joint0.CreateBody1Rel().SetTargets([body1.GetPath()])
        joint0_pos0 = Gf.Vec3f(0.1, 0.2, 0.3)
        joint0_pos1 = Gf.Vec3f(-0.4, 0.25, 0.05)
        joint0_rot0 = Gf.Quatf(1.0, 0.0, 0.0, 0.0)
        joint0_rot1 = Gf.Quatf(0.9238795, 0.0, 0.3826834, 0.0)
        joint0.CreateLocalPos0Attr().Set(joint0_pos0)
        joint0.CreateLocalPos1Attr().Set(joint0_pos1)
        joint0.CreateLocalRot0Attr().Set(joint0_rot0)
        joint0.CreateLocalRot1Attr().Set(joint0_rot1)
        joint0.CreateAxisAttr().Set("Z")

        joint1 = UsdPhysics.RevoluteJoint.Define(stage, "/World/Articulation/Joint1")
        joint1.CreateBody0Rel().SetTargets([body2.GetPath()])
        joint1.CreateBody1Rel().SetTargets([body1.GetPath()])
        joint1_pos0 = Gf.Vec3f(0.6, -0.1, 0.2)
        joint1_pos1 = Gf.Vec3f(-0.15, 0.35, -0.25)
        joint1_rot0 = Gf.Quatf(0.9659258, 0.2588190, 0.0, 0.0)
        joint1_rot1 = Gf.Quatf(0.7071068, 0.0, 0.0, 0.7071068)
        joint1.CreateLocalPos0Attr().Set(joint1_pos0)
        joint1.CreateLocalPos1Attr().Set(joint1_pos1)
        joint1.CreateLocalRot0Attr().Set(joint1_rot0)
        joint1.CreateLocalRot1Attr().Set(joint1_rot1)
        joint1.CreateAxisAttr().Set("Z")

        builder = newton.ModelBuilder()
        with self.assertRaises(ValueError) as exc_info:
            builder.add_usd(stage)
        self.assertIn("/World/Articulation/Joint1", str(exc_info.exception))

    @unittest.skipUnless(USD_AVAILABLE, "Requires usd-core")
    def test_reversed_fixed_root_joint_to_world_is_allowed(self):
        """Ensure a fixed root joint to world (body1 unset) does not raise."""
        from pxr import Gf, Usd, UsdGeom, UsdPhysics  # noqa: PLC0415

        stage = Usd.Stage.CreateInMemory()
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
        UsdPhysics.Scene.Define(stage, "/physicsScene")

        articulation = UsdGeom.Xform.Define(stage, "/World/Articulation")
        UsdPhysics.ArticulationRootAPI.Apply(articulation.GetPrim())

        def define_body(path):
            body = UsdGeom.Xform.Define(stage, path)
            UsdPhysics.RigidBodyAPI.Apply(body.GetPrim())
            return body

        root = define_body("/World/Articulation/Root")
        link1 = define_body("/World/Articulation/Link1")
        link2 = define_body("/World/Articulation/Link2")

        fixed = UsdPhysics.FixedJoint.Define(stage, "/World/Articulation/RootToWorld")
        # Here the child body (physics:body1) is -1, so the joint is silently reversed
        fixed.CreateBody0Rel().SetTargets([root.GetPath()])
        fixed.CreateLocalPos0Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
        fixed.CreateLocalPos1Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
        fixed.CreateLocalRot0Attr().Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
        fixed.CreateLocalRot1Attr().Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))

        joint1 = UsdPhysics.RevoluteJoint.Define(stage, "/World/Articulation/Joint1")
        joint1.CreateBody0Rel().SetTargets([root.GetPath()])
        joint1.CreateBody1Rel().SetTargets([link1.GetPath()])
        joint1.CreateLocalPos0Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
        joint1.CreateLocalPos1Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
        joint1.CreateLocalRot0Attr().Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
        joint1.CreateLocalRot1Attr().Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
        joint1.CreateAxisAttr().Set("Z")

        joint2 = UsdPhysics.RevoluteJoint.Define(stage, "/World/Articulation/Joint2")
        joint2.CreateBody0Rel().SetTargets([link1.GetPath()])
        joint2.CreateBody1Rel().SetTargets([link2.GetPath()])
        joint2.CreateLocalPos0Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
        joint2.CreateLocalPos1Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
        joint2.CreateLocalRot0Attr().Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
        joint2.CreateLocalRot1Attr().Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
        joint2.CreateAxisAttr().Set("Z")

        builder = newton.ModelBuilder()
        # We must not trigger an error here regarding the reversed joint.
        builder.add_usd(stage)

        self.assertEqual(builder.body_count, 3)
        self.assertEqual(builder.joint_count, 3)

        fixed_idx = builder.joint_key.index("/World/Articulation/RootToWorld")
        root_idx = builder.body_key.index("/World/Articulation/Root")
        self.assertEqual(builder.joint_parent[fixed_idx], -1)
        self.assertEqual(builder.joint_child[fixed_idx], root_idx)

    @unittest.skipUnless(USD_AVAILABLE, "Requires usd-core")
    def test_reversed_joint_unsupported_d6_raises(self):
        """Reversing a D6 joint should raise an error."""
        from pxr import Gf, Usd, UsdGeom, UsdPhysics  # noqa: PLC0415

        stage = Usd.Stage.CreateInMemory()
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
        UsdPhysics.Scene.Define(stage, "/physicsScene")

        articulation = UsdGeom.Xform.Define(stage, "/World/Articulation")
        UsdPhysics.ArticulationRootAPI.Apply(articulation.GetPrim())

        def define_body(path):
            body = UsdGeom.Xform.Define(stage, path)
            UsdPhysics.RigidBodyAPI.Apply(body.GetPrim())
            return body

        body0 = define_body("/World/Articulation/Body0")
        body1 = define_body("/World/Articulation/Body1")
        body2 = define_body("/World/Articulation/Body2")

        joint = UsdPhysics.Joint.Define(stage, "/World/Articulation/JointD6")
        joint.CreateBody0Rel().SetTargets([body1.GetPath()])
        joint.CreateBody1Rel().SetTargets([body0.GetPath()])
        joint.CreateLocalPos0Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
        joint.CreateLocalPos1Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
        joint.CreateLocalRot0Attr().Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
        joint.CreateLocalRot1Attr().Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))

        fixed = UsdPhysics.FixedJoint.Define(stage, "/World/Articulation/FixedJoint")
        fixed.CreateBody0Rel().SetTargets([body2.GetPath()])
        fixed.CreateBody1Rel().SetTargets([body0.GetPath()])
        fixed.CreateLocalPos0Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
        fixed.CreateLocalPos1Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
        fixed.CreateLocalRot0Attr().Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
        fixed.CreateLocalRot1Attr().Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))

        builder = newton.ModelBuilder()
        with self.assertRaises(ValueError) as exc_info:
            builder.add_usd(stage)
        error_message = str(exc_info.exception)
        self.assertIn("/World/Articulation/JointD6", error_message)
        self.assertIn("/World/Articulation/FixedJoint", error_message)

    @unittest.skipUnless(USD_AVAILABLE, "Requires usd-core")
    def test_reversed_joint_unsupported_spherical_raises(self):
        """Reversing a spherical joint should raise an error."""
        from pxr import Gf, Usd, UsdGeom, UsdPhysics  # noqa: PLC0415

        stage = Usd.Stage.CreateInMemory()
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
        UsdPhysics.Scene.Define(stage, "/physicsScene")

        articulation = UsdGeom.Xform.Define(stage, "/World/Articulation")
        UsdPhysics.ArticulationRootAPI.Apply(articulation.GetPrim())

        def define_body(path):
            body = UsdGeom.Xform.Define(stage, path)
            UsdPhysics.RigidBodyAPI.Apply(body.GetPrim())
            return body

        body0 = define_body("/World/Articulation/Body0")
        body1 = define_body("/World/Articulation/Body1")
        body2 = define_body("/World/Articulation/Body2")

        joint = UsdPhysics.SphericalJoint.Define(stage, "/World/Articulation/JointBall")
        joint.CreateBody0Rel().SetTargets([body1.GetPath()])
        joint.CreateBody1Rel().SetTargets([body0.GetPath()])
        joint.CreateLocalPos0Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
        joint.CreateLocalPos1Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
        joint.CreateLocalRot0Attr().Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
        joint.CreateLocalRot1Attr().Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))

        fixed = UsdPhysics.FixedJoint.Define(stage, "/World/Articulation/FixedJoint")
        fixed.CreateBody0Rel().SetTargets([body2.GetPath()])
        fixed.CreateBody1Rel().SetTargets([body0.GetPath()])
        fixed.CreateLocalPos0Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
        fixed.CreateLocalPos1Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
        fixed.CreateLocalRot0Attr().Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
        fixed.CreateLocalRot1Attr().Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))

        builder = newton.ModelBuilder()
        with self.assertRaises(ValueError) as exc_info:
            builder.add_usd(stage)
        error_message = str(exc_info.exception)
        self.assertIn("/World/Articulation/JointBall", error_message)
        self.assertIn("/World/Articulation/FixedJoint", error_message)

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
    def test_loop_joint(self):
        """Test that an articulation with a loop joint denoted with excludeFromArticulation is correctly parsed from USD."""
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

    def PhysicsRevoluteJoint "Joint1"
    {
        rel physics:body0 = </Articulation/Body1>
        point3f physics:localPos0 = (0, 0, 0)
        point3f physics:localPos1 = (0, 0, 0)
        quatf physics:localRot0 = (1, 0, 0, 0)
        quatf physics:localRot1 = (1, 0, 0, 0)
        token physics:axis = "Z"
        float physics:lowerLimit = -45
        float physics:upperLimit = 45
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

    def PhysicsRevoluteJoint "Joint2"
    {
        rel physics:body0 = </Articulation/Body2>
        point3f physics:localPos0 = (0, 0, 0)
        point3f physics:localPos1 = (0, 0, 0)
        quatf physics:localRot0 = (1, 0, 0, 0)
        quatf physics:localRot1 = (1, 0, 0, 0)
        token physics:axis = "Z"
        float physics:lowerLimit = -45
        float physics:upperLimit = 45
    }

    def PhysicsFixedJoint "LoopJoint"
    {
        rel physics:body0 = </Articulation/Body1>
        rel physics:body1 = </Articulation/Body2>
        point3f physics:localPos0 = (0, 0, 0)
        point3f physics:localPos1 = (0, 0, 0)
        quatf physics:localRot0 = (1, 0, 0, 0)
        quatf physics:localRot1 = (1, 0, 0, 0)
        bool physics:excludeFromArticulation = true
    }
}
"""
        stage = Usd.Stage.CreateInMemory()
        stage.GetRootLayer().ImportFromString(usd_content)

        builder = newton.ModelBuilder()
        builder.add_usd(stage)

        self.assertEqual(builder.joint_count, 3)
        self.assertEqual(builder.articulation_count, 1)
        self.assertEqual(
            builder.joint_type, [newton.JointType.REVOLUTE, newton.JointType.REVOLUTE, newton.JointType.FIXED]
        )
        self.assertEqual(builder.body_key, ["/Articulation/Body1", "/Articulation/Body2"])
        self.assertEqual(builder.joint_key, ["/Articulation/Joint1", "/Articulation/Joint2", "/Articulation/LoopJoint"])
        self.assertEqual(builder.joint_articulation, [0, 0, -1])

    @unittest.skipUnless(USD_AVAILABLE, "Requires usd-core")
    def test_solimp_friction_parsing(self):
        """Test that solimp_friction attribute is parsed correctly from USD."""
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

        # MuJoCo solimpfriction attribute (5 elements)
        uniform double[] mjc:solimpfriction = [0.89, 0.9, 0.01, 2.1, 1.8]
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

        # No solimpfriction - should use defaults
    }
}
"""
        stage = Usd.Stage.CreateInMemory()
        stage.GetRootLayer().ImportFromString(usd_content)

        builder = newton.ModelBuilder()
        SolverMuJoCo.register_custom_attributes(builder)
        builder.add_usd(stage)
        model = builder.finalize()

        # Check if solimpfriction custom attribute exists
        self.assertTrue(hasattr(model, "mujoco"), "Model should have mujoco namespace for custom attributes")
        self.assertTrue(hasattr(model.mujoco, "solimpfriction"), "Model should have solimpfriction attribute")

        solimpfriction = model.mujoco.solimpfriction.numpy()

        # Should have 2 joints: Joint1 (world to Body1) and Joint2 (Body1 to Body2)
        self.assertEqual(model.joint_count, 2, "Should have 2 single-DOF joints")

        # Helper to check if two arrays match within tolerance
        def arrays_match(arr, expected, tol=1e-4):
            return all(abs(arr[i] - expected[i]) < tol for i in range(len(expected)))

        # Expected values
        expected_joint1 = [0.89, 0.9, 0.01, 2.1, 1.8]  # from Joint1
        expected_joint2 = [0.9, 0.95, 0.001, 0.5, 2.0]  # from Joint2 (default values)

        # Check that both expected solimpfriction values are present in the model
        num_dofs = solimpfriction.shape[0]
        found_values = [solimpfriction[i, :].tolist() for i in range(num_dofs)]

        found_joint1 = any(arrays_match(val, expected_joint1) for val in found_values)
        found_joint2 = any(arrays_match(val, expected_joint2) for val in found_values)

        self.assertTrue(found_joint1, f"Expected solimpfriction {expected_joint1} not found in model")
        self.assertTrue(found_joint2, f"Expected default solimpfriction {expected_joint2} not found in model")

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
            builder.shape_type,
            [newton.GeoType.MESH, newton.GeoType.CONVEX_MESH, newton.GeoType.SPHERE, newton.GeoType.BOX],
        )

        # original mesh
        mesh_original = builder.shape_source[0]
        self.assertEqual(mesh_original.vertices.shape, (8, 3))
        assert_np_equal(mesh_original.vertices, vertices)
        assert_np_equal(mesh_original.indices, indices)

        # convex hull
        mesh_convex_hull = builder.shape_source[1]
        self.assertEqual(mesh_convex_hull.vertices.shape, (4, 3))
        self.assertEqual(builder.shape_type[1], newton.GeoType.CONVEX_MESH)

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

    @unittest.skipUnless(USD_AVAILABLE, "Requires usd-core")
    def test_solreffriction_parsing(self):
        """Test that solreffriction attribute is parsed correctly from USD."""
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

        # MuJoCo solreffriction attribute (2 elements)
        uniform double[] mjc:solreffriction = [0.01, 0.5]
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

        # No solreffriction - should use defaults
    }
}
"""
        stage = Usd.Stage.CreateInMemory()
        stage.GetRootLayer().ImportFromString(usd_content)

        builder = newton.ModelBuilder()
        SolverMuJoCo.register_custom_attributes(builder)
        builder.add_usd(stage)
        model = builder.finalize()

        # Check if solreffriction custom attribute exists
        self.assertTrue(hasattr(model, "mujoco"), "Model should have mujoco namespace for custom attributes")
        self.assertTrue(hasattr(model.mujoco, "solreffriction"), "Model should have solreffriction attribute")

        solreffriction = model.mujoco.solreffriction.numpy()

        # Should have 2 joints: Joint1 (world to Body1) and Joint2 (Body1 to Body2)
        self.assertEqual(model.joint_count, 2, "Should have 2 single-DOF joints")

        # Helper to check if two arrays match within tolerance
        def arrays_match(arr, expected, tol=1e-4):
            return all(abs(arr[i] - expected[i]) < tol for i in range(len(expected)))

        # Expected values
        expected_joint1 = [0.01, 0.5]  # from Joint1
        expected_joint2 = [0.02, 1.0]  # from Joint2 (default values)

        # Check that both expected solreffriction values are present in the model
        num_dofs = solreffriction.shape[0]
        found_values = [solreffriction[i, :].tolist() for i in range(num_dofs)]

        found_joint1 = any(arrays_match(val, expected_joint1) for val in found_values)
        found_joint2 = any(arrays_match(val, expected_joint2) for val in found_values)

        self.assertTrue(found_joint1, f"Expected solreffriction {expected_joint1} not found in model")
        self.assertTrue(found_joint2, f"Expected default solreffriction {expected_joint2} not found in model")

    @unittest.skipUnless(USD_AVAILABLE, "Requires usd-core")
    def test_geom_solimp_parsing(self):
        """Test that geom_solimp attribute is parsed correctly from USD."""
        from pxr import Usd  # noqa: PLC0415

        usd_content = """#usda 1.0
(
    upAxis = "Z"
)

def PhysicsScene "physicsScene"
{
}

def Xform "Body1" (
    prepend apiSchemas = ["PhysicsRigidBodyAPI", "PhysicsArticulationRootAPI"]
)
{
    double3 xformOp:translate = (0, 0, 1)
    uniform token[] xformOpOrder = ["xformOp:translate"]

    def Cube "Collision1" (
        prepend apiSchemas = ["PhysicsCollisionAPI"]
    )
    {
        double size = 0.2
        # MuJoCo solimp attribute (5 elements)
        uniform double[] mjc:solimp = [0.8, 0.9, 0.002, 0.4, 3.0]
    }
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
        # No solimp - should use defaults
    }
}

def PhysicsRevoluteJoint "Joint1"
{
    rel physics:body0 = </Body1>
    rel physics:body1 = </Body2>
    point3f physics:localPos0 = (0, 0, 0)
    point3f physics:localPos1 = (0, 0, 0)
    quatf physics:localRot0 = (1, 0, 0, 0)
    quatf physics:localRot1 = (1, 0, 0, 0)
    token physics:axis = "Z"
}

def Xform "Body3" (
    prepend apiSchemas = ["PhysicsRigidBodyAPI"]
)
{
    double3 xformOp:translate = (2, 0, 1)
    uniform token[] xformOpOrder = ["xformOp:translate"]

    def Capsule "Collision3" (
        prepend apiSchemas = ["PhysicsCollisionAPI"]
    )
    {
        double radius = 0.05
        double height = 0.2
        # Different solimp values
        uniform double[] mjc:solimp = [0.7, 0.85, 0.003, 0.6, 2.5]
    }
}

def PhysicsRevoluteJoint "Joint2"
{
    rel physics:body0 = </Body2>
    rel physics:body1 = </Body3>
    point3f physics:localPos0 = (0, 0, 0)
    point3f physics:localPos1 = (0, 0, 0)
    quatf physics:localRot0 = (1, 0, 0, 0)
    quatf physics:localRot1 = (1, 0, 0, 0)
    token physics:axis = "Y"
}
"""
        stage = Usd.Stage.CreateInMemory()
        stage.GetRootLayer().ImportFromString(usd_content)

        builder = newton.ModelBuilder()
        SolverMuJoCo.register_custom_attributes(builder)
        builder.add_usd(stage)
        model = builder.finalize()

        self.assertTrue(hasattr(model, "mujoco"), "Model should have mujoco namespace for custom attributes")
        self.assertTrue(hasattr(model.mujoco, "geom_solimp"), "Model should have geom_solimp attribute")

        geom_solimp = model.mujoco.geom_solimp.numpy()

        def arrays_match(arr, expected, tol=1e-4):
            return all(abs(arr[i] - expected[i]) < tol for i in range(len(expected)))

        # Check that we have shapes with expected values
        expected_explicit_1 = [0.8, 0.9, 0.002, 0.4, 3.0]
        expected_default = [0.9, 0.95, 0.001, 0.5, 2.0]  # default
        expected_explicit_2 = [0.7, 0.85, 0.003, 0.6, 2.5]

        # Find shapes matching each expected value
        found_explicit_1 = any(arrays_match(geom_solimp[i], expected_explicit_1) for i in range(model.shape_count))
        found_default = any(arrays_match(geom_solimp[i], expected_default) for i in range(model.shape_count))
        found_explicit_2 = any(arrays_match(geom_solimp[i], expected_explicit_2) for i in range(model.shape_count))

        self.assertTrue(found_explicit_1, f"Expected solimp {expected_explicit_1} not found in model")
        self.assertTrue(found_default, f"Expected default solimp {expected_default} not found in model")
        self.assertTrue(found_explicit_2, f"Expected solimp {expected_explicit_2} not found in model")

    @unittest.skipUnless(USD_AVAILABLE, "Requires usd-core")
    def test_geom_solmix_parsing(self):
        """Test that geom_solmix attribute is parsed correctly from USD."""
        from pxr import Usd  # noqa: PLC0415

        usd_content = """#usda 1.0
(
    upAxis = "Z"
)

def PhysicsScene "physicsScene"
{
}

def Xform "Body1" (
    prepend apiSchemas = ["PhysicsRigidBodyAPI", "PhysicsArticulationRootAPI"]
)
{
    double3 xformOp:translate = (0, 0, 1)
    uniform token[] xformOpOrder = ["xformOp:translate"]

    def Cube "Collision1" (
        prepend apiSchemas = ["PhysicsCollisionAPI"]
    )
    {
        double size = 0.2
        # MuJoCo solmix attribute (1 float)
        double mjc:solmix = 0.8
    }
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
        # No solmix - should use defaults
    }
}

def PhysicsRevoluteJoint "Joint1"
{
    rel physics:body0 = </Body1>
    rel physics:body1 = </Body2>
    point3f physics:localPos0 = (0, 0, 0)
    point3f physics:localPos1 = (0, 0, 0)
    quatf physics:localRot0 = (1, 0, 0, 0)
    quatf physics:localRot1 = (1, 0, 0, 0)
    token physics:axis = "Z"
}

def Xform "Body3" (
    prepend apiSchemas = ["PhysicsRigidBodyAPI"]
)
{
    double3 xformOp:translate = (2, 0, 1)
    uniform token[] xformOpOrder = ["xformOp:translate"]

    def Capsule "Collision3" (
        prepend apiSchemas = ["PhysicsCollisionAPI"]
    )
    {
        double radius = 0.05
        double height = 0.2
        # Different solmix values
        double mjc:solmix = 0.7
    }
}

def PhysicsRevoluteJoint "Joint2"
{
    rel physics:body0 = </Body2>
    rel physics:body1 = </Body3>
    point3f physics:localPos0 = (0, 0, 0)
    point3f physics:localPos1 = (0, 0, 0)
    quatf physics:localRot0 = (1, 0, 0, 0)
    quatf physics:localRot1 = (1, 0, 0, 0)
    token physics:axis = "Y"
}
"""
        stage = Usd.Stage.CreateInMemory()
        stage.GetRootLayer().ImportFromString(usd_content)

        builder = newton.ModelBuilder()
        SolverMuJoCo.register_custom_attributes(builder)
        builder.add_usd(stage)
        model = builder.finalize()

        self.assertTrue(hasattr(model, "mujoco"), "Model should have mujoco namespace for custom attributes")
        self.assertTrue(hasattr(model.mujoco, "geom_solmix"), "Model should have geom_solmix attribute")

        geom_solmix = model.mujoco.geom_solmix.numpy()

        def floats_match(arr, expected, tol=1e-4):
            return abs(arr - expected) < tol

        # Check that we have shapes with expected values
        expected_explicit_1 = 0.8
        expected_default = 1.0  # default
        expected_explicit_2 = 0.7

        # Find shapes matching each expected value
        found_explicit_1 = any(floats_match(geom_solmix[i], expected_explicit_1) for i in range(model.shape_count))
        found_default = any(floats_match(geom_solmix[i], expected_default) for i in range(model.shape_count))
        found_explicit_2 = any(floats_match(geom_solmix[i], expected_explicit_2) for i in range(model.shape_count))

        self.assertTrue(found_explicit_1, f"Expected solmix {expected_explicit_1} not found in model")
        self.assertTrue(found_default, f"Expected default solmix {expected_default} not found in model")
        self.assertTrue(found_explicit_2, f"Expected solmix {expected_explicit_2} not found in model")

    @unittest.skipUnless(USD_AVAILABLE, "Requires usd-core")
    def test_geom_gap_parsing(self):
        """Test that geom_gap attribute is parsed correctly from USD."""
        from pxr import Usd  # noqa: PLC0415

        usd_content = """#usda 1.0
(
    upAxis = "Z"
)

def PhysicsScene "physicsScene"
{
}

def Xform "Body1" (
    prepend apiSchemas = ["PhysicsRigidBodyAPI", "PhysicsArticulationRootAPI"]
)
{
    double3 xformOp:translate = (0, 0, 1)
    uniform token[] xformOpOrder = ["xformOp:translate"]

    def Cube "Collision1" (
        prepend apiSchemas = ["PhysicsCollisionAPI"]
    )
    {
        double size = 0.2
        # MuJoCo gap attribute (1 float)
        double mjc:gap = 0.8
    }
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
        # No gap - should use defaults
    }
}

def PhysicsRevoluteJoint "Joint1"
{
    rel physics:body0 = </Body1>
    rel physics:body1 = </Body2>
    point3f physics:localPos0 = (0, 0, 0)
    point3f physics:localPos1 = (0, 0, 0)
    quatf physics:localRot0 = (1, 0, 0, 0)
    quatf physics:localRot1 = (1, 0, 0, 0)
    token physics:axis = "Z"
}

def Xform "Body3" (
    prepend apiSchemas = ["PhysicsRigidBodyAPI"]
)
{
    double3 xformOp:translate = (2, 0, 1)
    uniform token[] xformOpOrder = ["xformOp:translate"]

    def Capsule "Collision3" (
        prepend apiSchemas = ["PhysicsCollisionAPI"]
    )
    {
        double radius = 0.05
        double height = 0.2
        # Different gap values
        double mjc:gap = 0.7
    }
}

def PhysicsRevoluteJoint "Joint2"
{
    rel physics:body0 = </Body2>
    rel physics:body1 = </Body3>
    point3f physics:localPos0 = (0, 0, 0)
    point3f physics:localPos1 = (0, 0, 0)
    quatf physics:localRot0 = (1, 0, 0, 0)
    quatf physics:localRot1 = (1, 0, 0, 0)
    token physics:axis = "Y"
}
"""
        stage = Usd.Stage.CreateInMemory()
        stage.GetRootLayer().ImportFromString(usd_content)

        builder = newton.ModelBuilder()
        SolverMuJoCo.register_custom_attributes(builder)
        builder.add_usd(stage)
        model = builder.finalize()

        self.assertTrue(hasattr(model, "mujoco"), "Model should have mujoco namespace for custom attributes")
        self.assertTrue(hasattr(model.mujoco, "geom_gap"), "Model should have geom_gap attribute")

        geom_gap = model.mujoco.geom_gap.numpy()

        def floats_match(arr, expected, tol=1e-4):
            return abs(arr - expected) < tol

        # Check that we have shapes with expected values
        expected_explicit_1 = 0.8
        expected_default = 0.0  # default
        expected_explicit_2 = 0.7

        # Find shapes matching each expected value
        found_explicit_1 = any(floats_match(geom_gap[i], expected_explicit_1) for i in range(model.shape_count))
        found_default = any(floats_match(geom_gap[i], expected_default) for i in range(model.shape_count))
        found_explicit_2 = any(floats_match(geom_gap[i], expected_explicit_2) for i in range(model.shape_count))

        self.assertTrue(found_explicit_1, f"Expected gap {expected_explicit_1} not found in model")
        self.assertTrue(found_default, f"Expected default gap {expected_default} not found in model")
        self.assertTrue(found_explicit_2, f"Expected gap {expected_explicit_2} not found in model")

    @unittest.skipUnless(USD_AVAILABLE, "Requires usd-core")
    def test_actuator_mode_inference_from_drive(self):
        """Test that ActuatorMode is correctly inferred from USD joint drives."""
        from pxr import Usd  # noqa: PLC0415

        from newton._src.sim.joints import ActuatorMode  # noqa: PLC0415

        usd_content = """#usda 1.0
(
    upAxis = "Z"
)

def PhysicsScene "PhysicsScene"
{
}

def Xform "Root" (
    prepend apiSchemas = ["PhysicsArticulationRootAPI"]
)
{
    def Xform "Body0" (
        prepend apiSchemas = ["PhysicsRigidBodyAPI"]
    )
    {
        double3 xformOp:translate = (0, 0, 0)
        uniform token[] xformOpOrder = ["xformOp:translate"]
        def Cube "Collision0" (
            prepend apiSchemas = ["PhysicsCollisionAPI"]
        )
        {
            double size = 0.2
        }
    }

    def Xform "Body1" (
        prepend apiSchemas = ["PhysicsRigidBodyAPI"]
    )
    {
        double3 xformOp:translate = (1, 0, 0)
        uniform token[] xformOpOrder = ["xformOp:translate"]
        def Cube "Collision1" (
            prepend apiSchemas = ["PhysicsCollisionAPI"]
        )
        {
            double size = 0.2
        }
    }

    def Xform "Body2" (
        prepend apiSchemas = ["PhysicsRigidBodyAPI"]
    )
    {
        double3 xformOp:translate = (2, 0, 0)
        uniform token[] xformOpOrder = ["xformOp:translate"]
        def Cube "Collision2" (
            prepend apiSchemas = ["PhysicsCollisionAPI"]
        )
        {
            double size = 0.2
        }
    }

    def Xform "Body3" (
        prepend apiSchemas = ["PhysicsRigidBodyAPI"]
    )
    {
        double3 xformOp:translate = (3, 0, 0)
        uniform token[] xformOpOrder = ["xformOp:translate"]
        def Cube "Collision3" (
            prepend apiSchemas = ["PhysicsCollisionAPI"]
        )
        {
            double size = 0.2
        }
    }

    def Xform "Body4" (
        prepend apiSchemas = ["PhysicsRigidBodyAPI"]
    )
    {
        double3 xformOp:translate = (4, 0, 0)
        uniform token[] xformOpOrder = ["xformOp:translate"]
        def Cube "Collision4" (
            prepend apiSchemas = ["PhysicsCollisionAPI"]
        )
        {
            double size = 0.2
        }
    }

    def Xform "Body5" (
        prepend apiSchemas = ["PhysicsRigidBodyAPI"]
    )
    {
        double3 xformOp:translate = (5, 0, 0)
        uniform token[] xformOpOrder = ["xformOp:translate"]
        def Cube "Collision5" (
            prepend apiSchemas = ["PhysicsCollisionAPI"]
        )
        {
            double size = 0.2
        }
    }

    def PhysicsRevoluteJoint "joint_effort" (
        prepend apiSchemas = ["PhysicsDriveAPI:angular"]
    )
    {
        rel physics:body0 = </Root/Body0>
        rel physics:body1 = </Root/Body1>
        float drive:angular:physics:stiffness = 0.0
        float drive:angular:physics:damping = 0.0
    }

    def PhysicsRevoluteJoint "joint_passive"
    {
        rel physics:body0 = </Root/Body1>
        rel physics:body1 = </Root/Body2>
    }

    def PhysicsRevoluteJoint "joint_position" (
        prepend apiSchemas = ["PhysicsDriveAPI:angular"]
    )
    {
        rel physics:body0 = </Root/Body2>
        rel physics:body1 = </Root/Body3>
        float drive:angular:physics:stiffness = 100.0
        float drive:angular:physics:damping = 0.0
    }

    def PhysicsRevoluteJoint "joint_velocity" (
        prepend apiSchemas = ["PhysicsDriveAPI:angular"]
    )
    {
        rel physics:body0 = </Root/Body3>
        rel physics:body1 = </Root/Body4>
        float drive:angular:physics:stiffness = 0.0
        float drive:angular:physics:damping = 10.0
    }

    def PhysicsRevoluteJoint "joint_both_gains" (
        prepend apiSchemas = ["PhysicsDriveAPI:angular"]
    )
    {
        rel physics:body0 = </Root/Body4>
        rel physics:body1 = </Root/Body5>
        float drive:angular:physics:stiffness = 100.0
        float drive:angular:physics:damping = 10.0
    }
}
"""
        stage = Usd.Stage.CreateInMemory()
        stage.GetRootLayer().ImportFromString(usd_content)

        builder = newton.ModelBuilder()
        builder.add_usd(stage)

        def get_qd_start(b, joint_name):
            joint_idx = b.joint_key.index(joint_name)
            return sum(b.joint_dof_dim[i][0] + b.joint_dof_dim[i][1] for i in range(joint_idx))

        self.assertEqual(
            builder.joint_act_mode[get_qd_start(builder, "/Root/joint_effort")],
            int(ActuatorMode.EFFORT),
        )
        self.assertEqual(
            builder.joint_act_mode[get_qd_start(builder, "/Root/joint_passive")],
            int(ActuatorMode.NONE),
        )
        self.assertEqual(
            builder.joint_act_mode[get_qd_start(builder, "/Root/joint_position")],
            int(ActuatorMode.POSITION),
        )
        self.assertEqual(
            builder.joint_act_mode[get_qd_start(builder, "/Root/joint_velocity")],
            int(ActuatorMode.VELOCITY),
        )
        self.assertEqual(
            builder.joint_act_mode[get_qd_start(builder, "/Root/joint_both_gains")],
            int(ActuatorMode.POSITION),
        )

        stage2 = Usd.Stage.CreateInMemory()
        stage2.GetRootLayer().ImportFromString(usd_content)

        builder2 = newton.ModelBuilder()
        builder2.add_usd(stage2, force_position_velocity_actuation=True)

        self.assertEqual(
            builder2.joint_act_mode[get_qd_start(builder2, "/Root/joint_both_gains")],
            int(ActuatorMode.POSITION_VELOCITY),
        )
        self.assertEqual(
            builder2.joint_act_mode[get_qd_start(builder2, "/Root/joint_position")],
            int(ActuatorMode.POSITION),
        )
        self.assertEqual(
            builder2.joint_act_mode[get_qd_start(builder2, "/Root/joint_velocity")],
            int(ActuatorMode.VELOCITY),
        )


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
        self.assertTrue(all(joint_enabled))

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

                self.assertTrue(
                    all(abs(lhs_p[i] - rhs_p[i]) < 1e-6 for i in range(3)),
                    f"Joint {j} ({model.joint_key[j]}) position mismatch: expected={rhs_p}, Newton={lhs_p}",
                )

                q_diff = lhs_q * wp.quat_inverse(rhs_q)
                angle_diff = 2.0 * math.acos(min(1.0, abs(q_diff[3])))
                self.assertLessEqual(
                    angle_diff,
                    3e-3,
                    f"Joint {j} ({model.joint_key[j]}) rotation mismatch: expected={rhs_q}, Newton={lhs_q}, angle_diff={math.degrees(angle_diff)}°",
                )

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

            usd_quat = usd.from_gfquat(shape_spec.localRot)
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
    def test_geom_priority_parsing(self):
        """Test that geom_priority attribute is parsed correctly from USD."""
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
            int mjc:priority = 1
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
            # No priority - should use default (0)
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
        self.assertTrue(hasattr(model.mujoco, "geom_priority"))

        geom_priority = model.mujoco.geom_priority.numpy()

        # Should have 2 shapes
        self.assertEqual(model.shape_count, 2)

        # Find the values - one should be 1, one should be 0
        self.assertTrue(np.any(geom_priority == 1))
        self.assertTrue(np.any(geom_priority == 0))

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
        self.assertTrue(np.any(~jnt_actgravcomp))

    @unittest.skipUnless(USD_AVAILABLE, "Requires usd-core")
    def test_option_scalar_world_parsing(self):
        """Test parsing of WORLD frequency scalar options from USD PhysicsScene (6 options)."""
        from pxr import Usd  # noqa: PLC0415

        test_cases = [
            ("impratio", "1.5", 1.5, 6),
            ("tolerance", "1e-6", 1e-6, 10),
            ("ls_tolerance", "0.001", 0.001, 6),
            ("ccd_tolerance", "1e-5", 1e-5, 10),
            ("density", "1.225", 1.225, 6),
            ("viscosity", "1.8e-5", 1.8e-5, 10),
        ]

        for option_name, usd_value, expected, places in test_cases:
            with self.subTest(option=option_name):
                usd_content = f"""#usda 1.0
(
    defaultPrim = "World"
    metersPerUnit = 1.0
    upAxis = "Z"
)

def Xform "World"
{{
    def PhysicsScene "PhysicsScene" (
        prepend apiSchemas = ["MjcSceneAPI"]
    )
    {{
        float mjc:option:{option_name} = {usd_value}
    }}

    def Xform "Articulation" (
        prepend apiSchemas = ["PhysicsArticulationRootAPI"]
    )
    {{
        def Xform "Body1" (
            prepend apiSchemas = ["PhysicsRigidBodyAPI"]
        )
        {{
            double3 xformOp:translate = (0, 0, 1)
            uniform token[] xformOpOrder = ["xformOp:translate"]

            def Sphere "Collision" (
                prepend apiSchemas = ["PhysicsCollisionAPI"]
            )
            {{
                double radius = 0.1
            }}
        }}

        def PhysicsRevoluteJoint "Joint"
        {{
            rel physics:body0 = </World/Articulation/Body1>
            point3f physics:localPos0 = (0, 0, 0)
            quatf physics:localRot0 = (1, 0, 0, 0)
            token physics:axis = "Z"
        }}
    }}
}}
"""
                stage = Usd.Stage.CreateInMemory()
                stage.GetRootLayer().ImportFromString(usd_content)

                builder = newton.ModelBuilder()
                SolverMuJoCo.register_custom_attributes(builder)
                builder.add_usd(stage)
                model = builder.finalize()

                self.assertTrue(hasattr(model, "mujoco"))
                self.assertTrue(hasattr(model.mujoco, option_name))
                value = getattr(model.mujoco, option_name).numpy()
                self.assertEqual(len(value), 1)
                self.assertAlmostEqual(value[0], expected, places=places)

    @unittest.skipUnless(USD_AVAILABLE, "Requires usd-core")
    def test_option_vector_world_parsing(self):
        """Test parsing of WORLD frequency vector options from USD PhysicsScene (2 options)."""
        from pxr import Usd  # noqa: PLC0415

        test_cases = [
            ("wind", "(1, 0.5, -0.5)", [1.0, 0.5, -0.5]),
            ("magnetic", "(0, -1, 0.5)", [0.0, -1.0, 0.5]),
        ]

        for option_name, usd_value, expected in test_cases:
            with self.subTest(option=option_name):
                usd_content = f"""#usda 1.0
(
    defaultPrim = "World"
    metersPerUnit = 1.0
    upAxis = "Z"
)

def Xform "World"
{{
    def PhysicsScene "PhysicsScene" (
        prepend apiSchemas = ["MjcSceneAPI"]
    )
    {{
        float3 mjc:option:{option_name} = {usd_value}
    }}

    def Xform "Articulation" (
        prepend apiSchemas = ["PhysicsArticulationRootAPI"]
    )
    {{
        def Xform "Body1" (
            prepend apiSchemas = ["PhysicsRigidBodyAPI"]
        )
        {{
            double3 xformOp:translate = (0, 0, 1)
            uniform token[] xformOpOrder = ["xformOp:translate"]

            def Sphere "Collision" (
                prepend apiSchemas = ["PhysicsCollisionAPI"]
            )
            {{
                double radius = 0.1
            }}
        }}

        def PhysicsRevoluteJoint "Joint"
        {{
            rel physics:body0 = </World/Articulation/Body1>
            point3f physics:localPos0 = (0, 0, 0)
            quatf physics:localRot0 = (1, 0, 0, 0)
            token physics:axis = "Z"
        }}
    }}
}}
"""
                stage = Usd.Stage.CreateInMemory()
                stage.GetRootLayer().ImportFromString(usd_content)

                builder = newton.ModelBuilder()
                SolverMuJoCo.register_custom_attributes(builder)
                builder.add_usd(stage)
                model = builder.finalize()

                self.assertTrue(hasattr(model, "mujoco"))
                self.assertTrue(hasattr(model.mujoco, option_name))
                value = getattr(model.mujoco, option_name).numpy()
                self.assertEqual(len(value), 1)
                self.assertTrue(np.allclose(value[0], expected))

    @unittest.skipUnless(USD_AVAILABLE, "Requires usd-core")
    def test_option_numeric_once_parsing(self):
        """Test parsing of ONCE frequency numeric options from USD PhysicsScene (5 options)."""
        from pxr import Usd  # noqa: PLC0415

        test_cases = [
            ("iterations", "30", 30),
            ("ls_iterations", "15", 15),
            ("ccd_iterations", "25", 25),
            ("sdf_iterations", "20", 20),
            ("sdf_initpoints", "50", 50),
        ]

        for option_name, usd_value, expected in test_cases:
            with self.subTest(option=option_name):
                usd_content = f"""#usda 1.0
(
    defaultPrim = "World"
    metersPerUnit = 1.0
    upAxis = "Z"
)

def Xform "World"
{{
    def PhysicsScene "PhysicsScene" (
        prepend apiSchemas = ["MjcSceneAPI"]
    )
    {{
        int mjc:option:{option_name} = {usd_value}
    }}

    def Xform "Articulation" (
        prepend apiSchemas = ["PhysicsArticulationRootAPI"]
    )
    {{
        def Xform "Body1" (
            prepend apiSchemas = ["PhysicsRigidBodyAPI"]
        )
        {{
            double3 xformOp:translate = (0, 0, 1)
            uniform token[] xformOpOrder = ["xformOp:translate"]

            def Sphere "Collision" (
                prepend apiSchemas = ["PhysicsCollisionAPI"]
            )
            {{
                double radius = 0.1
            }}
        }}

        def PhysicsRevoluteJoint "Joint"
        {{
            rel physics:body0 = </World/Articulation/Body1>
            point3f physics:localPos0 = (0, 0, 0)
            quatf physics:localRot0 = (1, 0, 0, 0)
            token physics:axis = "Z"
        }}
    }}
}}
"""
                stage = Usd.Stage.CreateInMemory()
                stage.GetRootLayer().ImportFromString(usd_content)

                builder = newton.ModelBuilder()
                SolverMuJoCo.register_custom_attributes(builder)
                builder.add_usd(stage)
                model = builder.finalize()

                self.assertTrue(hasattr(model, "mujoco"))
                self.assertTrue(hasattr(model.mujoco, option_name))
                value = getattr(model.mujoco, option_name).numpy()
                self.assertEqual(len(value), 1)  # ONCE frequency
                self.assertEqual(value[0], expected)

    @unittest.skipUnless(USD_AVAILABLE, "Requires usd-core")
    def test_option_enum_once_parsing(self):
        """Test parsing of ONCE frequency enum options from USD PhysicsScene (4 options)."""
        from pxr import Usd  # noqa: PLC0415

        test_cases = [
            ("integrator", "0", 0),  # Euler
            ("solver", "2", 2),  # Newton
            ("cone", "1", 1),  # elliptic
            ("jacobian", "1", 1),  # sparse
        ]

        for option_name, usd_value, expected_int in test_cases:
            with self.subTest(option=option_name):
                usd_content = f"""#usda 1.0
(
    defaultPrim = "World"
    metersPerUnit = 1.0
    upAxis = "Z"
)

def Xform "World"
{{
    def PhysicsScene "PhysicsScene" (
        prepend apiSchemas = ["MjcSceneAPI"]
    )
    {{
        int mjc:option:{option_name} = {usd_value}
    }}

    def Xform "Articulation" (
        prepend apiSchemas = ["PhysicsArticulationRootAPI"]
    )
    {{
        def Xform "Body1" (
            prepend apiSchemas = ["PhysicsRigidBodyAPI"]
        )
        {{
            double3 xformOp:translate = (0, 0, 1)
            uniform token[] xformOpOrder = ["xformOp:translate"]

            def Sphere "Collision" (
                prepend apiSchemas = ["PhysicsCollisionAPI"]
            )
            {{
                double radius = 0.1
            }}
        }}

        def PhysicsRevoluteJoint "Joint"
        {{
            rel physics:body0 = </World/Articulation/Body1>
            point3f physics:localPos0 = (0, 0, 0)
            quatf physics:localRot0 = (1, 0, 0, 0)
            token physics:axis = "Z"
        }}
    }}
}}
"""
                stage = Usd.Stage.CreateInMemory()
                stage.GetRootLayer().ImportFromString(usd_content)

                builder = newton.ModelBuilder()
                SolverMuJoCo.register_custom_attributes(builder)
                builder.add_usd(stage)
                model = builder.finalize()

                self.assertTrue(hasattr(model, "mujoco"))
                self.assertTrue(hasattr(model.mujoco, option_name))
                value = getattr(model.mujoco, option_name).numpy()
                self.assertEqual(len(value), 1)  # ONCE frequency
                self.assertEqual(value[0], expected_int)

    @unittest.skipUnless(USD_AVAILABLE, "Requires usd-core")
    def test_parse_mujoco_options_disabled(self):
        """Test that MuJoCo options from PhysicsScene are not parsed when parse_mujoco_options=False."""
        from pxr import Usd  # noqa: PLC0415

        usd_content = """
#usda 1.0
(
    defaultPrim = "World"
    metersPerUnit = 1.0
    upAxis = "Z"
)
def Xform "World"
{
    def PhysicsScene "PhysicsScene"
    {
        float mjc:option:impratio = 99.0
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

            def Sphere "Collision" (
                prepend apiSchemas = ["PhysicsCollisionAPI"]
            )
            {
                double radius = 0.1
            }
        }
    }
}
"""
        stage = Usd.Stage.CreateInMemory()
        stage.GetRootLayer().ImportFromString(usd_content)

        builder = newton.ModelBuilder()
        SolverMuJoCo.register_custom_attributes(builder)
        builder.add_usd(stage, parse_mujoco_options=False)
        model = builder.finalize()

        # impratio should remain at default (1.0), not the USD value (99.0)
        self.assertAlmostEqual(model.mujoco.impratio.numpy()[0], 1.0, places=4)

    @unittest.skipUnless(USD_AVAILABLE, "Requires usd-core")
    def test_ref_attribute_parsing(self):
        """Test that 'mjc:ref' attribute is parsed."""
        from pxr import Usd  # noqa: PLC0415

        usd_content = """#usda 1.0
(
    metersPerUnit = 1.0
    upAxis = "Z"
)

def Xform "Articulation" (
    prepend apiSchemas = ["PhysicsArticulationRootAPI"]
)
{
    def Cube "base" (
        prepend apiSchemas = ["PhysicsRigidBodyAPI", "PhysicsCollisionAPI"]
    )
    {
        double3 xformOp:translate = (0, 0, 0)
        uniform token[] xformOpOrder = ["xformOp:translate"]
    }

    def Cube "child1" (
        prepend apiSchemas = ["PhysicsRigidBodyAPI", "PhysicsCollisionAPI"]
    )
    {
        double3 xformOp:translate = (0, 0, 1)
        uniform token[] xformOpOrder = ["xformOp:translate"]
    }

    def PhysicsRevoluteJoint "revolute_joint"
    {
        token physics:axis = "Y"
        rel physics:body0 = </Articulation/base>
        rel physics:body1 = </Articulation/child1>
        float mjc:ref = 90.0
    }
}
"""
        stage = Usd.Stage.CreateInMemory()
        stage.GetRootLayer().ImportFromString(usd_content)

        builder = newton.ModelBuilder()
        SolverMuJoCo.register_custom_attributes(builder)
        builder.add_usd(stage)
        model = builder.finalize()

        # Verify custom attribute parsing
        self.assertTrue(hasattr(model, "mujoco"))
        self.assertTrue(hasattr(model.mujoco, "dof_ref"))
        dof_ref = model.mujoco.dof_ref.numpy()
        qd_start = model.joint_qd_start.numpy()

        revolute_joint_idx = model.joint_key.index("/Articulation/revolute_joint")
        self.assertAlmostEqual(dof_ref[qd_start[revolute_joint_idx]], 90.0, places=4)

    @unittest.skipUnless(USD_AVAILABLE, "Requires usd-core")
    def test_springref_attribute_parsing(self):
        """Test that 'mjc:springref' attribute is parsed for revolute and prismatic joints."""
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
    def Xform "Body0" (
        prepend apiSchemas = ["PhysicsRigidBodyAPI"]
    )
    {
        double3 xformOp:translate = (0, 0, 0)
        uniform token[] xformOpOrder = ["xformOp:translate"]
        def Cube "Collision0" (
            prepend apiSchemas = ["PhysicsCollisionAPI"]
        )
        {
            double size = 0.2
        }
    }

    def Xform "Body1" (
        prepend apiSchemas = ["PhysicsRigidBodyAPI"]
    )
    {
        double3 xformOp:translate = (1, 0, 0)
        uniform token[] xformOpOrder = ["xformOp:translate"]
        def Cube "Collision1" (
            prepend apiSchemas = ["PhysicsCollisionAPI"]
        )
        {
            double size = 0.2
        }
    }

    def Xform "Body2" (
        prepend apiSchemas = ["PhysicsRigidBodyAPI"]
    )
    {
        double3 xformOp:translate = (2, 0, 0)
        uniform token[] xformOpOrder = ["xformOp:translate"]
        def Cube "Collision2" (
            prepend apiSchemas = ["PhysicsCollisionAPI"]
        )
        {
            double size = 0.2
        }
    }

    def PhysicsRevoluteJoint "revolute_joint" (
        prepend apiSchemas = ["PhysicsDriveAPI:angular"]
    )
    {
        rel physics:body0 = </Articulation/Body0>
        rel physics:body1 = </Articulation/Body1>
        float mjc:springref = 30.0
    }

    def PhysicsPrismaticJoint "prismatic_joint"
    {
        token physics:axis = "Z"
        rel physics:body0 = </Articulation/Body1>
        rel physics:body1 = </Articulation/Body2>
        float mjc:springref = 0.25
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
        self.assertTrue(hasattr(model.mujoco, "dof_springref"))
        springref = model.mujoco.dof_springref.numpy()
        qd_start = model.joint_qd_start.numpy()

        revolute_joint_idx = model.joint_key.index("/Articulation/revolute_joint")
        self.assertAlmostEqual(springref[qd_start[revolute_joint_idx]], 30.0, places=4)

        prismatic_joint_idx = model.joint_key.index("/Articulation/prismatic_joint")
        self.assertAlmostEqual(springref[qd_start[prismatic_joint_idx]], 0.25, places=4)

    @unittest.skipUnless(USD_AVAILABLE, "Requires usd-core")
    def test_material_parsing(self):
        """Test that material attributes are parsed correctly from USD."""
        from pxr import Usd, UsdGeom, UsdPhysics, UsdShade  # noqa: PLC0415

        stage = Usd.Stage.CreateInMemory()
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
        UsdGeom.SetStageMetersPerUnit(stage, 1.0)
        UsdPhysics.Scene.Define(stage, "/physicsScene")

        # Create a physics material with all relevant properties
        material_path = "/Materials/TestMaterial"
        material = UsdShade.Material.Define(stage, material_path)
        material_prim = material.GetPrim()
        material_prim.ApplyAPI("NewtonMaterialAPI")
        physics_material = UsdPhysics.MaterialAPI.Apply(material_prim)
        physics_material.GetStaticFrictionAttr().Set(0.6)
        physics_material.GetDynamicFrictionAttr().Set(0.5)
        physics_material.GetRestitutionAttr().Set(0.3)
        physics_material.GetDensityAttr().Set(1500.0)
        material_prim.GetAttribute("newton:torsionalFriction").Set(0.15)
        material_prim.GetAttribute("newton:rollingFriction").Set(0.08)

        # Create an articulation with a body and collider
        articulation = UsdGeom.Xform.Define(stage, "/Articulation")
        UsdPhysics.ArticulationRootAPI.Apply(articulation.GetPrim())

        body = UsdGeom.Xform.Define(stage, "/Articulation/Body")
        body_prim = body.GetPrim()
        UsdPhysics.RigidBodyAPI.Apply(body_prim)

        # Create a collider and bind the material
        collider = UsdGeom.Cube.Define(stage, "/Articulation/Body/Collider")
        collider_prim = collider.GetPrim()
        UsdPhysics.CollisionAPI.Apply(collider_prim)
        binding_api = UsdShade.MaterialBindingAPI.Apply(collider_prim)
        binding_api.Bind(material, "physics")

        # Import the USD
        builder = newton.ModelBuilder()
        result = builder.add_usd(stage)
        model = builder.finalize()

        # Verify the material properties were parsed correctly
        shape_idx = result["path_shape_map"]["/Articulation/Body/Collider"]

        # Check friction (mu is dynamicFriction)
        self.assertAlmostEqual(model.shape_material_mu.numpy()[shape_idx], 0.5, places=4)

        # Check restitution
        self.assertAlmostEqual(model.shape_material_restitution.numpy()[shape_idx], 0.3, places=4)

        # Check torsional friction
        torsional = model.shape_material_torsional_friction.numpy()[shape_idx]
        self.assertAlmostEqual(torsional, 0.15, places=4)

        # Check rolling friction
        rolling = model.shape_material_rolling_friction.numpy()[shape_idx]
        self.assertAlmostEqual(rolling, 0.08, places=4)

    @unittest.skipUnless(USD_AVAILABLE, "Requires usd-core")
    def test_contact_margin_parsing(self):
        """Test that contact_margin is parsed correctly from USD."""
        from pxr import Usd, UsdGeom, UsdPhysics  # noqa: PLC0415

        stage = Usd.Stage.CreateInMemory()
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
        UsdGeom.SetStageMetersPerUnit(stage, 1.0)
        UsdPhysics.Scene.Define(stage, "/physicsScene")

        # Create an articulation with a body
        articulation = UsdGeom.Xform.Define(stage, "/Articulation")
        UsdPhysics.ArticulationRootAPI.Apply(articulation.GetPrim())

        body = UsdGeom.Xform.Define(stage, "/Articulation/Body")
        body_prim = body.GetPrim()
        UsdPhysics.RigidBodyAPI.Apply(body_prim)

        # Create a collider with newton:contactMargin
        collider1 = UsdGeom.Cube.Define(stage, "/Articulation/Body/Collider1")
        collider1_prim = collider1.GetPrim()
        collider1_prim.ApplyAPI("NewtonCollisionAPI")
        UsdPhysics.CollisionAPI.Apply(collider1_prim)
        collider1_prim.GetAttribute("newton:contactMargin").Set(0.05)

        # Create another collider without contact_margin (should use default)
        collider2 = UsdGeom.Sphere.Define(stage, "/Articulation/Body/Collider2")
        collider2_prim = collider2.GetPrim()
        UsdPhysics.CollisionAPI.Apply(collider2_prim)

        # Import the USD
        builder = newton.ModelBuilder()
        builder.default_shape_cfg.contact_margin = 0.01  # set a known default
        result = builder.add_usd(stage)
        model = builder.finalize()

        # Verify contact_margin was parsed correctly
        shape1_idx = result["path_shape_map"]["/Articulation/Body/Collider1"]
        shape2_idx = result["path_shape_map"]["/Articulation/Body/Collider2"]

        # Collider1 should have the authored value
        margin1 = model.shape_contact_margin.numpy()[shape1_idx]
        self.assertAlmostEqual(margin1, 0.05, places=4)

        # Collider2 should have the default value
        margin2 = model.shape_contact_margin.numpy()[shape2_idx]
        self.assertAlmostEqual(margin2, 0.01, places=4)

    @unittest.skipUnless(USD_AVAILABLE, "Requires usd-core")
    def test_scene_gravity_enabled_parsing(self):
        """Test that gravity_enabled is parsed correctly from USD scene."""
        from pxr import Usd, UsdGeom, UsdPhysics  # noqa: PLC0415

        # Test with gravity enabled (default)
        stage = Usd.Stage.CreateInMemory()
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
        UsdPhysics.Scene.Define(stage, "/physicsScene")

        body = UsdGeom.Cube.Define(stage, "/Body")
        body_prim = body.GetPrim()
        UsdPhysics.RigidBodyAPI.Apply(body_prim)
        UsdPhysics.CollisionAPI.Apply(body_prim)

        builder = newton.ModelBuilder()
        builder.add_usd(stage)

        # Gravity should be enabled (non-zero)
        self.assertNotEqual(builder.gravity, 0.0)

        # Test with gravity disabled via newton:gravityEnabled
        stage2 = Usd.Stage.CreateInMemory()
        UsdGeom.SetStageUpAxis(stage2, UsdGeom.Tokens.z)
        scene = UsdPhysics.Scene.Define(stage2, "/physicsScene")
        scene_prim = scene.GetPrim()
        scene_prim.ApplyAPI("NewtonSceneAPI")
        scene_prim.GetAttribute("newton:gravityEnabled").Set(False)

        body2 = UsdGeom.Cube.Define(stage2, "/Body")
        body2_prim = body2.GetPrim()
        UsdPhysics.RigidBodyAPI.Apply(body2_prim)
        UsdPhysics.CollisionAPI.Apply(body2_prim)

        builder2 = newton.ModelBuilder()
        builder2.add_usd(stage2)

        # Gravity should be disabled (zero)
        self.assertEqual(builder2.gravity, 0.0)

    @unittest.skipUnless(USD_AVAILABLE, "Requires usd-core")
    def test_scene_time_steps_per_second_parsing(self):
        """Test that time_steps_per_second is parsed correctly from USD scene."""
        from pxr import Usd, UsdGeom, UsdPhysics  # noqa: PLC0415

        stage = Usd.Stage.CreateInMemory()
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
        scene = UsdPhysics.Scene.Define(stage, "/physicsScene")
        scene_prim = scene.GetPrim()
        scene_prim.ApplyAPI("NewtonSceneAPI")

        body = UsdGeom.Cube.Define(stage, "/Body")
        body_prim = body.GetPrim()
        UsdPhysics.RigidBodyAPI.Apply(body_prim)
        UsdPhysics.CollisionAPI.Apply(body_prim)

        builder = newton.ModelBuilder()
        result = builder.add_usd(stage)
        # default physics_dt should be 1/1000 = 0.001
        self.assertAlmostEqual(result["physics_dt"], 0.001, places=6)

        scene_prim.GetAttribute("newton:timeStepsPerSecond").Set(500)
        builder = newton.ModelBuilder()
        result = builder.add_usd(stage)
        # physics_dt should be 1/500 = 0.002
        self.assertAlmostEqual(result["physics_dt"], 0.002, places=6)

        # explicit bad value should be ignored and use the default fallback instead
        scene_prim.GetAttribute("newton:timeStepsPerSecond").Set(0)
        builder = newton.ModelBuilder()
        result = builder.add_usd(stage)
        # physics_dt should be 0.001
        self.assertAlmostEqual(result["physics_dt"], 0.001, places=6)

    @unittest.skipUnless(USD_AVAILABLE, "Requires usd-core")
    def test_scene_max_solver_iterations_parsing(self):
        """Test that max_solver_iterations is parsed correctly from USD scene."""
        from pxr import Usd, UsdGeom, UsdPhysics  # noqa: PLC0415

        stage = Usd.Stage.CreateInMemory()
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
        scene = UsdPhysics.Scene.Define(stage, "/physicsScene")
        scene_prim = scene.GetPrim()
        scene_prim.ApplyAPI("NewtonSceneAPI")

        body = UsdGeom.Cube.Define(stage, "/Body")
        body_prim = body.GetPrim()
        UsdPhysics.RigidBodyAPI.Apply(body_prim)
        UsdPhysics.CollisionAPI.Apply(body_prim)

        # default max_solver_iterations should be -1
        builder = newton.ModelBuilder()
        result = builder.add_usd(stage)
        self.assertEqual(result["max_solver_iterations"], -1)

        scene_prim.GetAttribute("newton:maxSolverIterations").Set(200)
        builder = newton.ModelBuilder()
        result = builder.add_usd(stage)
        # max_solver_iterations should be 200
        self.assertEqual(result["max_solver_iterations"], 200)

    @unittest.skipUnless(USD_AVAILABLE, "Requires usd-core")
    def test_mesh_max_hull_vertices_parsing(self):
        """Test that max_hull_vertices is parsed correctly from mesh collision."""
        from pxr import Gf, Usd, UsdGeom, UsdPhysics  # noqa: PLC0415

        stage = Usd.Stage.CreateInMemory()
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
        UsdPhysics.Scene.Define(stage, "/physicsScene")

        # Create a simple tetrahedron mesh
        vertices = [
            Gf.Vec3f(0, 0, 0),
            Gf.Vec3f(1, 0, 0),
            Gf.Vec3f(0.5, 1, 0),
            Gf.Vec3f(0.5, 0.5, 1),
        ]
        indices = [0, 1, 2, 0, 1, 3, 1, 2, 3, 0, 2, 3]

        mesh = UsdGeom.Mesh.Define(stage, "/Mesh")
        mesh_prim = mesh.GetPrim()
        mesh.CreateFaceVertexCountsAttr().Set([3, 3, 3, 3])
        mesh.CreateFaceVertexIndicesAttr().Set(indices)
        mesh.CreatePointsAttr().Set(vertices)

        UsdPhysics.RigidBodyAPI.Apply(mesh_prim)
        UsdPhysics.CollisionAPI.Apply(mesh_prim)
        mesh_prim.ApplyAPI("NewtonMeshCollisionAPI")

        # Default max_hull_vertices comes from the builder
        builder = newton.ModelBuilder()
        builder.add_usd(stage, mesh_maxhullvert=20)
        self.assertEqual(builder.shape_source[0].maxhullvert, 20)

        # Set max_hull_vertices to 32 on the mesh prim
        mesh_prim.GetAttribute("newton:maxHullVertices").Set(32)
        builder = newton.ModelBuilder()
        builder.add_usd(stage, mesh_maxhullvert=20)
        # the authored value should override the builder value
        self.assertEqual(builder.shape_source[0].maxhullvert, 32)


if __name__ == "__main__":
    unittest.main(verbosity=2, failfast=False)
