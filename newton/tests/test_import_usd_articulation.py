# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import contextlib
import io
import math
import os
import unittest
import warnings

import numpy as np
import warp as wp

import newton
import newton.examples
from newton import JointType
from newton._src.utils.import_usd import _is_uniform_scale
from newton.tests._usd_import_test_utils import _expect_jointless_articulation_warning
from newton.tests.unittest_utils import USD_AVAILABLE, assert_np_equal


class TestImportUsdArticulation(unittest.TestCase):
    @unittest.skipUnless(USD_AVAILABLE, "Requires usd-core")
    def test_import_usd_raises_on_stage_errors(self):
        """Reject a USD stage with composition errors."""
        from pxr import Usd

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
        """Import the ant articulation with its bodies, joints, shapes, and path maps."""
        builder = newton.ModelBuilder()

        results = builder.add_usd(
            os.path.join(os.path.dirname(__file__), "assets", "ant.usda"),
            collapse_fixed_joints=True,
        )
        self.assertEqual(builder.body_count, 9)
        self.assertEqual(builder.shape_count, 26)
        self.assertEqual(len(builder.shape_label), len(set(builder.shape_label)))
        self.assertEqual(len(builder.body_label), len(set(builder.body_label)))
        self.assertEqual(len(builder.joint_label), len(set(builder.joint_label)))
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
    def test_mirrored_body_transform_warns(self):
        """Warn when a rigid body has a negative-determinant (mirrored) transform.

        Improper transforms have no unique rotation decomposition, so the
        incoming-xform rebase can inject a spurious constant rotation into
        body and joint frames (common with mirror-scaled CAD exports).
        """
        from pxr import Gf, Usd, UsdGeom, UsdPhysics

        stage = Usd.Stage.CreateInMemory()
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
        UsdPhysics.Scene.Define(stage, "/physicsScene")

        body = UsdGeom.Xform.Define(stage, "/World/Body")
        body.AddScaleOp().Set(Gf.Vec3f(-1.0, -1.0, -1.0))
        UsdPhysics.RigidBodyAPI.Apply(body.GetPrim())
        UsdPhysics.ArticulationRootAPI.Apply(body.GetPrim())
        mass = UsdPhysics.MassAPI.Apply(body.GetPrim())
        mass.GetMassAttr().Set(1.0)
        mass.GetCenterOfMassAttr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
        mass.GetDiagonalInertiaAttr().Set(Gf.Vec3f(1.0, 1.0, 1.0))

        joint = UsdPhysics.RevoluteJoint.Define(stage, "/World/Joint")
        joint.CreateBody1Rel().SetTargets([body.GetPath()])
        joint.CreateAxisAttr().Set("Z")

        builder = newton.ModelBuilder()
        with self.assertWarnsRegex(UserWarning, "mirrored"):
            builder.add_usd(stage, load_visual_shapes=False)

    @unittest.skipUnless(USD_AVAILABLE, "Requires usd-core")
    def test_import_body_newton_armature_ignored(self):
        """Ignore body-level newton:armature without warnings or changes to inertia.

        Body-level armature was removed; joint-level armature remains supported.
        """
        from pxr import Sdf, Usd, UsdGeom, UsdPhysics

        stage = Usd.Stage.CreateInMemory()
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
        UsdPhysics.Scene.Define(stage, "/physicsScene")

        body = UsdGeom.Xform.Define(stage, "/World/Body")
        UsdPhysics.RigidBodyAPI.Apply(body.GetPrim())
        body.GetPrim().CreateAttribute("newton:armature", Sdf.ValueTypeNames.Float, True).Set(0.125)

        collider = UsdGeom.Cube.Define(stage, "/World/Body/Collision")
        UsdPhysics.CollisionAPI.Apply(collider.GetPrim())

        builder = newton.ModelBuilder()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            builder.add_usd(stage)

        self.assertFalse(
            any("newton:armature" in str(w.message) for w in caught if issubclass(w.category, DeprecationWarning)),
            "body newton:armature should be ignored silently",
        )

        # Authored armature is ignored: inertia is shape-only (default cube:
        # half-extents (1,1,1), density 1000 → mass 8000, diagonal = 16000/3).
        inertia = builder.body_inertia[0]
        expected_diag = 16000.0 / 3.0
        for j in range(3):
            self.assertAlmostEqual(float(inertia[j, j]), expected_diag, places=2)

    @unittest.skipUnless(USD_AVAILABLE, "Requires usd-core")
    def test_import_non_articulated_joints(self):
        """Import a rootless four-bar mechanism without creating an articulation."""
        builder = newton.ModelBuilder()

        asset_path = newton.examples.get_asset("boxes_fourbar.usda")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            builder.add_usd(asset_path)
        self.assertFalse(any("articulation" in str(item.message).lower() for item in caught))

        self.assertEqual(builder.body_count, 4)
        self.assertEqual(builder.joint_type.count(newton.JointType.REVOLUTE), 4)
        self.assertEqual(builder.joint_type.count(newton.JointType.FREE), 0)
        self.assertTrue(all(art_id == -1 for art_id in builder.joint_articulation))

        # Non-root orphan joints still require opting out of articulation validation.
        model = builder.finalize(skip_validation_joints=True)
        self.assertEqual(model.body_count, 4)
        self.assertEqual(model.joint_type.list().count(newton.JointType.REVOLUTE), 4)
        self.assertEqual(model.joint_type.list().count(newton.JointType.FREE), 0)
        self.assertTrue(all(art_id == -1 for art_id in model.joint_articulation.numpy()))

    def _make_rootless_fixed_stage(self, *, with_child_joint: bool):
        """Build a rootless USD mechanism with an optional articulated child."""
        from pxr import Gf, Usd, UsdGeom, UsdPhysics

        stage = Usd.Stage.CreateInMemory()
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
        UsdPhysics.Scene.Define(stage, "/physicsScene")

        base = UsdGeom.Xform.Define(stage, "/World/Base")
        UsdPhysics.RigidBodyAPI.Apply(base.GetPrim())
        base_mass = UsdPhysics.MassAPI.Apply(base.GetPrim())
        base_mass.GetMassAttr().Set(1.0)
        base_mass.GetCenterOfMassAttr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
        base_mass.GetDiagonalInertiaAttr().Set(Gf.Vec3f(1.0, 1.0, 1.0))

        fixed = UsdPhysics.FixedJoint.Define(stage, "/World/RootJoint")
        fixed.CreateBody1Rel().SetTargets([base.GetPath()])
        fixed.CreateLocalPos0Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
        fixed.CreateLocalPos1Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
        fixed.CreateLocalRot0Attr().Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
        fixed.CreateLocalRot1Attr().Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))

        if with_child_joint:
            link = UsdGeom.Xform.Define(stage, "/World/Link")
            link.AddTranslateOp().Set(Gf.Vec3d(1.0, 0.0, 0.0))
            UsdPhysics.RigidBodyAPI.Apply(link.GetPrim())
            link_mass = UsdPhysics.MassAPI.Apply(link.GetPrim())
            link_mass.GetMassAttr().Set(1.0)
            link_mass.GetCenterOfMassAttr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
            link_mass.GetDiagonalInertiaAttr().Set(Gf.Vec3f(1.0, 1.0, 1.0))

            child_joint = UsdPhysics.RevoluteJoint.Define(stage, "/World/ChildJoint")
            child_joint.CreateBody0Rel().SetTargets([base.GetPath()])
            child_joint.CreateBody1Rel().SetTargets([link.GetPath()])
            child_joint.CreateLocalPos0Attr().Set(Gf.Vec3f(0.5, 0.0, 0.0))
            child_joint.CreateLocalPos1Attr().Set(Gf.Vec3f(-0.5, 0.0, 0.0))
            child_joint.CreateLocalRot0Attr().Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
            child_joint.CreateLocalRot1Attr().Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
            child_joint.CreateAxisAttr().Set("Z")

        return stage

    @unittest.skipUnless(USD_AVAILABLE, "Requires usd-core")
    def test_body_to_world_fixed_joint_without_articulation_root_stays_orphan(self):
        """Keep a fixed joint to world rootless and finalize normally."""
        stage = self._make_rootless_fixed_stage(with_child_joint=False)
        builder = newton.ModelBuilder()
        builder.add_usd(stage, load_visual_shapes=False)

        self.assertEqual(builder.articulation_count, 0)
        self.assertEqual(builder.joint_count, 1)
        root_joint_idx = builder.joint_label.index("/World/RootJoint")
        self.assertEqual(builder.joint_parent[root_joint_idx], -1)
        self.assertEqual(builder.joint_articulation[root_joint_idx], -1)

        model = builder.finalize()
        self.assertEqual(model.articulation_count, 0)
        self.assertEqual(model.joint_articulation.numpy()[root_joint_idx], -1)

    @unittest.skipUnless(USD_AVAILABLE, "Requires usd-core")
    def test_rootless_mechanism_root_and_child_joints_stay_orphan(self):
        """Keep root and child joints outside articulations without ArticulationRootAPI."""
        stage = self._make_rootless_fixed_stage(with_child_joint=True)
        builder = newton.ModelBuilder()
        builder.add_usd(stage, load_visual_shapes=False)

        self.assertEqual(builder.articulation_count, 0)
        self.assertEqual(set(builder.joint_label), {"/World/RootJoint", "/World/ChildJoint"})
        root_joint_idx = builder.joint_label.index("/World/RootJoint")
        child_joint_idx = builder.joint_label.index("/World/ChildJoint")
        self.assertEqual(builder.joint_parent[root_joint_idx], -1)
        self.assertEqual(builder.joint_parent[child_joint_idx], builder.body_label.index("/World/Base"))
        self.assertEqual(builder.joint_articulation[root_joint_idx], -1)
        self.assertEqual(builder.joint_articulation[child_joint_idx], -1)

        model = builder.finalize(skip_validation_joints=True)
        self.assertEqual(model.articulation_count, 0)
        model_joint_articulation = model.joint_articulation.numpy().tolist()
        self.assertEqual(model_joint_articulation[root_joint_idx], -1)
        self.assertEqual(model_joint_articulation[child_joint_idx], -1)

    @unittest.skipUnless(USD_AVAILABLE, "Requires usd-core")
    def test_rootless_multi_joint_body_is_merged(self):
        """Retain all MJCF DOFs when merging multiple world joints on one orphan body."""
        from pxr import Gf, Usd, UsdGeom, UsdPhysics

        stage = Usd.Stage.CreateInMemory()
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
        UsdPhysics.Scene.Define(stage, "/physicsScene")

        body = UsdGeom.Cube.Define(stage, "/World/Body")
        UsdPhysics.RigidBodyAPI.Apply(body.GetPrim())
        UsdPhysics.CollisionAPI.Apply(body.GetPrim())

        slide = UsdPhysics.PrismaticJoint.Define(stage, "/World/Body/slide")
        slide.CreateBody1Rel().SetTargets([body.GetPath()])
        slide.CreateLocalPos0Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
        slide.CreateLocalPos1Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
        slide.CreateLocalRot0Attr().Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
        slide.CreateLocalRot1Attr().Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
        slide.CreateAxisAttr().Set("X")

        hinge = UsdPhysics.RevoluteJoint.Define(stage, "/World/Body/hinge")
        hinge.CreateBody1Rel().SetTargets([body.GetPath()])
        hinge.CreateLocalPos0Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
        hinge.CreateLocalPos1Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
        hinge.CreateLocalRot0Attr().Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
        hinge.CreateLocalRot1Attr().Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
        hinge.CreateAxisAttr().Set("Z")

        builder = newton.ModelBuilder()
        result = builder.add_usd(stage, load_visual_shapes=False)
        self.assertEqual(builder.articulation_count, 0)
        self.assertEqual(builder.joint_count, 1)
        self.assertEqual(builder.joint_type, [newton.JointType.D6])
        self.assertEqual(builder.joint_dof_dim, [(1, 1)])
        self.assertEqual(builder.joint_articulation, [-1])
        self.assertEqual(result["path_joint_map"][slide.GetPath().pathString], 0)
        self.assertEqual(result["path_joint_map"][hinge.GetPath().pathString], 0)

        model = builder.finalize()
        self.assertEqual(model.articulation_count, 0)
        self.assertEqual(model.joint_dof_count, 2)

    @unittest.skipUnless(USD_AVAILABLE, "Requires usd-core")
    def test_import_disabled_joints_create_free_joints(self):
        """Create free joints for floating bodies when all authored joints are disabled."""
        from pxr import Gf, Usd, UsdGeom, UsdPhysics

        stage = Usd.Stage.CreateInMemory()
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
        UsdPhysics.Scene.Define(stage, "/physicsScene")

        # Regression test: if all joints are disabled (or filtered out), we still
        # need to create free joints for floating bodies so each body has DOFs.
        def define_body(path):
            body = UsdGeom.Cube.Define(stage, path)
            UsdPhysics.RigidBodyAPI.Apply(body.GetPrim())
            # Adding CollisionAPI triggers mass computation from geometry (density * volume).
            # Bodies need positive mass to receive auto-inserted base joints.
            UsdPhysics.CollisionAPI.Apply(body.GetPrim())
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
        # Because the stage has no enabled mechanism joints, each body is treated
        # as standalone and receives its own articulation.
        self.assertEqual(builder.articulation_count, 2)
        self.assertEqual(set(builder.joint_articulation), {0, 1})

        model = builder.finalize()
        self.assertEqual(model.articulation_count, 2)
        self.assertEqual(set(model.joint_articulation.numpy().tolist()), {0, 1})

    @unittest.skipUnless(USD_AVAILABLE, "Requires usd-core")
    def test_unrelated_floating_body_gets_single_body_articulation(self):
        """Create standalone articulations for floating bodies outside authored articulations."""
        from pxr import Gf, Usd, UsdGeom, UsdPhysics

        stage = Usd.Stage.CreateInMemory()
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
        UsdPhysics.Scene.Define(stage, "/physicsScene")

        robot = UsdGeom.Xform.Define(stage, "/World/Robot")
        UsdPhysics.ArticulationRootAPI.Apply(robot.GetPrim())

        robot_base = UsdGeom.Cube.Define(stage, "/World/Robot/Base")
        UsdPhysics.RigidBodyAPI.Apply(robot_base.GetPrim())
        UsdPhysics.CollisionAPI.Apply(robot_base.GetPrim())

        robot_link = UsdGeom.Cube.Define(stage, "/World/Robot/Link")
        robot_link.AddTranslateOp().Set(Gf.Vec3d(1.0, 0.0, 0.0))
        UsdPhysics.RigidBodyAPI.Apply(robot_link.GetPrim())
        UsdPhysics.CollisionAPI.Apply(robot_link.GetPrim())

        robot_joint = UsdPhysics.RevoluteJoint.Define(stage, "/World/Robot/Joint")
        robot_joint.CreateBody0Rel().SetTargets([robot_base.GetPath()])
        robot_joint.CreateBody1Rel().SetTargets([robot_link.GetPath()])
        robot_joint.CreateLocalPos0Attr().Set(Gf.Vec3f(0.5, 0.0, 0.0))
        robot_joint.CreateLocalPos1Attr().Set(Gf.Vec3f(-0.5, 0.0, 0.0))
        robot_joint.CreateLocalRot0Attr().Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
        robot_joint.CreateLocalRot1Attr().Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
        robot_joint.CreateAxisAttr().Set("Z")

        loose_body = UsdGeom.Cube.Define(stage, "/World/LooseBody")
        UsdPhysics.RigidBodyAPI.Apply(loose_body.GetPrim())
        UsdPhysics.CollisionAPI.Apply(loose_body.GetPrim())

        builder = newton.ModelBuilder()
        builder.add_usd(stage, floating=False)

        self.assertEqual(builder.body_count, 3)
        self.assertEqual(builder.joint_count, 3)
        self.assertEqual(builder.articulation_count, 2)

        robot_base_joint = next(
            i for i, child in enumerate(builder.joint_child) if builder.body_label[child] == "/World/Robot/Base"
        )
        loose_joint = next(
            i for i, child in enumerate(builder.joint_child) if builder.body_label[child] == "/World/LooseBody"
        )

        self.assertEqual(builder.joint_articulation[robot_base_joint], 0)
        self.assertEqual(builder.joint_articulation[builder.joint_label.index("/World/Robot/Joint")], 0)
        self.assertEqual(builder.joint_articulation[loose_joint], 1)

        model = builder.finalize()
        self.assertEqual(model.articulation_count, 2)
        self.assertEqual(model.joint_articulation.numpy().tolist()[loose_joint], 1)

    @unittest.skipUnless(USD_AVAILABLE, "Requires usd-core")
    def test_import_orphan_joints_with_articulation_present(self):
        """Import joints outside articulations alongside an authored articulation.

        This test creates a stage with an articulation and a separate revolute joint outside it,
        and verifies that both are parsed correctly.
        """
        from pxr import Gf, Usd, UsdGeom, UsdPhysics

        stage = Usd.Stage.CreateInMemory()
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
        UsdPhysics.Scene.Define(stage, "/physicsScene")

        # Articulation: two bodies connected by a fixed joint and a revolute joint
        arm = UsdGeom.Xform.Define(stage, "/World/Arm")
        UsdPhysics.ArticulationRootAPI.Apply(arm.GetPrim())

        body_a = UsdGeom.Xform.Define(stage, "/World/Arm/BodyA")
        UsdPhysics.RigidBodyAPI.Apply(body_a.GetPrim())
        body_a.AddTranslateOp().Set(Gf.Vec3d(0, 0, 0))
        col_a = UsdGeom.Cube.Define(stage, "/World/Arm/BodyA/Collision")
        UsdPhysics.CollisionAPI.Apply(col_a.GetPrim())

        body_b = UsdGeom.Xform.Define(stage, "/World/Arm/BodyB")
        UsdPhysics.RigidBodyAPI.Apply(body_b.GetPrim())
        body_b.AddTranslateOp().Set(Gf.Vec3d(1, 0, 0))
        col_b = UsdGeom.Cube.Define(stage, "/World/Arm/BodyB/Collision")
        UsdPhysics.CollisionAPI.Apply(col_b.GetPrim())

        fixed_joint = UsdPhysics.FixedJoint.Define(stage, "/World/Arm/FixedJoint")
        fixed_joint.CreateBody1Rel().SetTargets([body_a.GetPath()])
        fixed_joint.CreateLocalPos0Attr().Set(Gf.Vec3f(0, 0, 0))
        fixed_joint.CreateLocalPos1Attr().Set(Gf.Vec3f(0, 0, 0))
        fixed_joint.CreateLocalRot0Attr().Set(Gf.Quatf(1, 0, 0, 0))
        fixed_joint.CreateLocalRot1Attr().Set(Gf.Quatf(1, 0, 0, 0))

        rev_joint = UsdPhysics.RevoluteJoint.Define(stage, "/World/Arm/RevoluteJoint")
        rev_joint.CreateBody0Rel().SetTargets([body_a.GetPath()])
        rev_joint.CreateBody1Rel().SetTargets([body_b.GetPath()])
        rev_joint.CreateLocalPos0Attr().Set(Gf.Vec3f(0.5, 0, 0))
        rev_joint.CreateLocalPos1Attr().Set(Gf.Vec3f(-0.5, 0, 0))
        rev_joint.CreateLocalRot0Attr().Set(Gf.Quatf(1, 0, 0, 0))
        rev_joint.CreateLocalRot1Attr().Set(Gf.Quatf(1, 0, 0, 0))
        rev_joint.CreateAxisAttr().Set("Z")

        # Separate bodies connected by a revolute joint, outside any articulation
        body_c = UsdGeom.Xform.Define(stage, "/World/BodyC")
        UsdPhysics.RigidBodyAPI.Apply(body_c.GetPrim())
        body_c.AddTranslateOp().Set(Gf.Vec3d(5, 0, 0))
        col_c = UsdGeom.Cube.Define(stage, "/World/BodyC/Collision")
        UsdPhysics.CollisionAPI.Apply(col_c.GetPrim())

        body_d = UsdGeom.Xform.Define(stage, "/World/BodyD")
        UsdPhysics.RigidBodyAPI.Apply(body_d.GetPrim())
        body_d.AddTranslateOp().Set(Gf.Vec3d(6, 0, 0))
        col_d = UsdGeom.Cube.Define(stage, "/World/BodyD/Collision")
        UsdPhysics.CollisionAPI.Apply(col_d.GetPrim())

        orphan_joint = UsdPhysics.RevoluteJoint.Define(stage, "/World/OrphanJoint")
        orphan_joint.CreateBody0Rel().SetTargets([body_c.GetPath()])
        orphan_joint.CreateBody1Rel().SetTargets([body_d.GetPath()])
        orphan_joint.CreateLocalPos0Attr().Set(Gf.Vec3f(0.5, 0, 0))
        orphan_joint.CreateLocalPos1Attr().Set(Gf.Vec3f(-0.5, 0, 0))
        orphan_joint.CreateLocalRot0Attr().Set(Gf.Quatf(1, 0, 0, 0))
        orphan_joint.CreateLocalRot1Attr().Set(Gf.Quatf(1, 0, 0, 0))
        orphan_joint.CreateAxisAttr().Set("Z")

        # A standalone world joint must also remain an orphan when another
        # authored articulation is present in the stage.
        body_e = UsdGeom.Cube.Define(stage, "/World/BodyE")
        UsdPhysics.RigidBodyAPI.Apply(body_e.GetPrim())
        UsdPhysics.CollisionAPI.Apply(body_e.GetPrim())
        body_e.AddTranslateOp().Set(Gf.Vec3d(8, 0, 0))
        root_slide = UsdPhysics.PrismaticJoint.Define(stage, "/World/RootSlide")
        root_slide.CreateBody1Rel().SetTargets([body_e.GetPath()])
        root_slide.CreateLocalPos0Attr().Set(Gf.Vec3f(0, 0, 0))
        root_slide.CreateLocalPos1Attr().Set(Gf.Vec3f(0, 0, 0))
        root_slide.CreateLocalRot0Attr().Set(Gf.Quatf(1, 0, 0, 0))
        root_slide.CreateLocalRot1Attr().Set(Gf.Quatf(1, 0, 0, 0))
        root_slide.CreateAxisAttr().Set("X")

        builder = newton.ModelBuilder()
        builder.add_usd(stage)

        self.assertIn("/World/Arm/RevoluteJoint", builder.joint_label)
        self.assertIn("/World/OrphanJoint", builder.joint_label)
        self.assertIn("/World/RootSlide", builder.joint_label)

        art_idx = builder.joint_label.index("/World/Arm/RevoluteJoint")
        orphan_idx = builder.joint_label.index("/World/OrphanJoint")
        self.assertEqual(builder.joint_type[art_idx], newton.JointType.REVOLUTE)
        self.assertEqual(builder.joint_type[orphan_idx], newton.JointType.REVOLUTE)

        # orphan joint stays without an articulation
        self.assertEqual(builder.joint_articulation[orphan_idx], -1)
        root_slide_idx = builder.joint_label.index("/World/RootSlide")
        self.assertEqual(builder.joint_type[root_slide_idx], newton.JointType.PRISMATIC)
        self.assertEqual(builder.joint_parent[root_slide_idx], -1)
        self.assertEqual(builder.joint_articulation[root_slide_idx], -1)

        model = builder.finalize(skip_validation_joints=True)
        self.assertEqual(model.body_count, 5)

    @unittest.skipUnless(USD_AVAILABLE, "Requires usd-core")
    def test_stray_joint_does_not_strip_unrelated_floating_bodies(self):
        """Preserve base joints for unrelated floating bodies when importing a stray joint.

        Regression test for issue #3002.
        """
        from pxr import Gf, Usd, UsdGeom, UsdPhysics

        stage = Usd.Stage.CreateInMemory()
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
        UsdPhysics.Scene.Define(stage, "/physicsScene")

        def define_body(path, pos):
            body = UsdGeom.Cube.Define(stage, path)
            UsdPhysics.RigidBodyAPI.Apply(body.GetPrim())
            # CollisionAPI gives the body positive mass so it is eligible for a base joint.
            UsdPhysics.CollisionAPI.Apply(body.GetPrim())
            body.AddTranslateOp().Set(Gf.Vec3d(*pos))
            return body

        define_body("/World/FreeBody", (0, 0, 0))

        prop_a = define_body("/World/PropA", (5, 0, 0))
        prop_b = define_body("/World/PropB", (6, 0, 0))
        stray = UsdPhysics.FixedJoint.Define(stage, "/World/StrayFixedJoint")
        stray.CreateBody0Rel().SetTargets([prop_a.GetPath()])
        stray.CreateBody1Rel().SetTargets([prop_b.GetPath()])
        stray.CreateLocalPos0Attr().Set(Gf.Vec3f(0, 0, 0))
        stray.CreateLocalPos1Attr().Set(Gf.Vec3f(0, 0, 0))
        stray.CreateLocalRot0Attr().Set(Gf.Quatf(1, 0, 0, 0))
        stray.CreateLocalRot1Attr().Set(Gf.Quatf(1, 0, 0, 0))

        builder = newton.ModelBuilder()
        builder.add_usd(stage)

        self.assertEqual(builder.body_count, 3)

        free_idx = builder.body_label.index("/World/FreeBody")
        self.assertIn(free_idx, builder.joint_child)
        free_joint = builder.joint_child.index(free_idx)
        self.assertEqual(builder.joint_type[free_joint], JointType.FREE)
        self.assertNotEqual(builder.joint_articulation[free_joint], -1)

        # The authored joint must remain orphaned (no articulation), unchanged by the fix.
        stray_joint = builder.joint_label.index("/World/StrayFixedJoint")
        self.assertEqual(builder.joint_type[stray_joint], JointType.FIXED)
        self.assertEqual(builder.joint_articulation[stray_joint], -1)

    @unittest.skipUnless(USD_AVAILABLE, "Requires usd-core")
    def test_body_to_world_fixed_joint(self):
        """Import a body-to-world PhysicsFixedJoint as FIXED without creating an articulation."""
        from pxr import Gf, Usd, UsdGeom, UsdPhysics

        stage = Usd.Stage.CreateInMemory()
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
        UsdPhysics.Scene.Define(stage, "/physicsScene")

        # Main articulation: two bodies with a revolute joint.
        arm = UsdGeom.Xform.Define(stage, "/World/Arm")
        UsdPhysics.ArticulationRootAPI.Apply(arm.GetPrim())

        base = UsdGeom.Xform.Define(stage, "/World/Arm/Base")
        UsdPhysics.RigidBodyAPI.Apply(base.GetPrim())
        base.AddTranslateOp().Set(Gf.Vec3d(0, 0, 0))
        col_base = UsdGeom.Cube.Define(stage, "/World/Arm/Base/Collision")
        UsdPhysics.CollisionAPI.Apply(col_base.GetPrim())

        link1 = UsdGeom.Xform.Define(stage, "/World/Arm/Link1")
        UsdPhysics.RigidBodyAPI.Apply(link1.GetPrim())
        link1.AddTranslateOp().Set(Gf.Vec3d(1, 0, 0))
        col_link1 = UsdGeom.Cube.Define(stage, "/World/Arm/Link1/Collision")
        UsdPhysics.CollisionAPI.Apply(col_link1.GetPrim())

        rev = UsdPhysics.RevoluteJoint.Define(stage, "/World/Arm/RevJoint")
        rev.CreateBody0Rel().SetTargets([base.GetPath()])
        rev.CreateBody1Rel().SetTargets([link1.GetPath()])
        rev.CreateLocalPos0Attr().Set(Gf.Vec3f(0.5, 0, 0))
        rev.CreateLocalPos1Attr().Set(Gf.Vec3f(-0.5, 0, 0))
        rev.CreateLocalRot0Attr().Set(Gf.Quatf(1, 0, 0, 0))
        rev.CreateLocalRot1Attr().Set(Gf.Quatf(1, 0, 0, 0))
        rev.CreateAxisAttr().Set("Z")

        # world_link: a rigid body fixed-jointed to the world (body0 unset = world).
        wl = UsdGeom.Xform.Define(stage, "/World/WorldLink")
        UsdPhysics.RigidBodyAPI.Apply(wl.GetPrim())
        wl.AddTranslateOp().Set(Gf.Vec3d(0, 0, 0))
        col_wl = UsdGeom.Cube.Define(stage, "/World/WorldLink/Collision")
        UsdPhysics.CollisionAPI.Apply(col_wl.GetPrim())

        fixed = UsdPhysics.FixedJoint.Define(stage, "/World/WorldLink/FixedJoint")
        fixed.CreateBody1Rel().SetTargets([wl.GetPath()])
        fixed.CreateLocalPos0Attr().Set(Gf.Vec3f(0, 0, 0))
        fixed.CreateLocalPos1Attr().Set(Gf.Vec3f(0, 0, 0))
        fixed.CreateLocalRot0Attr().Set(Gf.Quatf(1, 0, 0, 0))
        fixed.CreateLocalRot1Attr().Set(Gf.Quatf(1, 0, 0, 0))

        builder = newton.ModelBuilder()
        builder.add_usd(stage)

        # 3 bodies: Base, Link1, WorldLink.
        self.assertEqual(builder.body_count, 3)
        self.assertEqual(builder.articulation_count, 1)

        wl_body_idx = builder.body_label.index("/World/WorldLink")
        wl_joint_idx = next(i for i in range(builder.joint_count) if builder.joint_child[i] == wl_body_idx)

        # world_link must have a FIXED joint, not a FREE joint.
        self.assertEqual(builder.joint_type[wl_joint_idx], newton.JointType.FIXED)
        # Parent is -1 (world).
        self.assertEqual(builder.joint_parent[wl_joint_idx], -1)
        # The world-fixed joint pins a standalone body without generalized
        # coordinates, so it stays outside the authored arm articulation.
        self.assertEqual(builder.joint_articulation[wl_joint_idx], -1)

        rev_joint_idx = builder.joint_label.index("/World/Arm/RevJoint")
        arm_art = builder.joint_articulation[rev_joint_idx]
        self.assertNotEqual(arm_art, -1)

        # Model must finalize without errors (no orphan joint issues).
        model = builder.finalize()
        self.assertEqual(model.body_count, 3)

    @unittest.skipUnless(USD_AVAILABLE, "Requires usd-core")
    def test_orphan_world_fixed_joint_respects_env_offset_and_xform(self):
        """Preserve environment offsets and spawn transforms for orphan body-to-world fixed joints."""
        from pxr import Gf, Usd, UsdGeom, UsdPhysics

        local_pose0 = wp.transform(wp.vec3(0.1, 0.2, 0.3), wp.quat(0.0, 0.0, 0.7071068, 0.7071068))  # 90deg about z
        local_pose1 = wp.transform(wp.vec3(-0.2, 0.05, 0.4), wp.quat(0.7071068, 0.0, 0.0, 0.7071068))  # 90deg about x

        for side in ["body0", "body1"]:  # Test the world being on either body0 or body1
            with self.subTest(side=side):
                stage = Usd.Stage.CreateInMemory()
                UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
                UsdPhysics.Scene.Define(stage, "/physicsScene")

                env = UsdGeom.Xform.Define(stage, "/World/env")
                env.AddTranslateOp().Set(Gf.Vec3d(100.0, 200.0, 0.0))

                link = UsdGeom.Xform.Define(stage, "/World/env/PinnedLink")
                UsdPhysics.RigidBodyAPI.Apply(link.GetPrim())

                fixed = UsdPhysics.FixedJoint.Define(stage, "/World/env/PinnedLink/FixedJoint")
                if side == "body0":
                    fixed.CreateBody0Rel().SetTargets([link.GetPath()])
                else:
                    fixed.CreateBody1Rel().SetTargets([link.GetPath()])
                p0, q0 = local_pose0.p, local_pose0.q
                p1, q1 = local_pose1.p, local_pose1.q
                fixed.CreateLocalPos0Attr().Set(Gf.Vec3f(float(p0[0]), float(p0[1]), float(p0[2])))
                fixed.CreateLocalRot0Attr().Set(Gf.Quatf(float(q0[3]), float(q0[0]), float(q0[1]), float(q0[2])))
                fixed.CreateLocalPos1Attr().Set(Gf.Vec3f(float(p1[0]), float(p1[1]), float(p1[2])))
                fixed.CreateLocalRot1Attr().Set(Gf.Quatf(float(q1[3]), float(q1[0]), float(q1[1]), float(q1[2])))

                builder = newton.ModelBuilder()
                builder.add_usd(stage, xform=wp.transform(wp.vec3(5.0, 0.0, 0.0), wp.quat_identity()))

                link_idx = builder.body_label.index("/World/env/PinnedLink")
                joint_idx = builder.joint_label.index("/World/env/PinnedLink/FixedJoint")
                self.assertEqual(builder.articulation_count, 0)
                self.assertEqual(builder.joint_type[joint_idx], newton.JointType.FIXED)
                self.assertEqual(builder.joint_parent[joint_idx], -1)
                self.assertEqual(builder.joint_articulation[joint_idx], -1)

                # Check the fixed joint frame by validating the joint_X_c.
                expected_X_c = local_pose0 if side == "body0" else local_pose1
                joint_X_c = builder.joint_X_c[joint_idx]
                assert_np_equal(np.array(joint_X_c.p), np.array(expected_X_c.p), tol=1e-4)
                # Compare rotations by the angle between them (q and -q are equal).
                q_err = joint_X_c.q * wp.quat_inverse(expected_X_c.q)
                self.assertLessEqual(2.0 * math.acos(min(1.0, abs(q_err[3]))), 1e-4)

                # Check that the body is imported at spawn * USD child world pose
                # (env origin + spawn translation, identity rotation).
                body_q = builder.body_q[link_idx]
                assert_np_equal(np.array(body_q.p), np.array([105.0, 200.0, 0.0]), tol=1e-4)
                q_err = body_q.q * wp.quat_inverse(wp.quat_identity())
                self.assertLessEqual(2.0 * math.acos(min(1.0, abs(q_err[3]))), 1e-4)

                model = builder.finalize()
                self.assertEqual(model.articulation_count, 0)

    @unittest.skipUnless(USD_AVAILABLE, "Requires usd-core")
    def test_collapse_fixed_joints_preserves_orphan_joints(self):
        """Preserve orphan joints and their bodies when collapsing fixed joints."""
        from pxr import Gf, Usd, UsdGeom, UsdPhysics

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
        builder.add_usd(stage, collapse_fixed_joints=True)

        # All three bodies and both revolute joints must survive collapse
        self.assertEqual(builder.body_count, 3)
        self.assertEqual(builder.joint_count, 2)
        self.assertIn("/World/RevJoint1", builder.joint_label)
        self.assertIn("/World/RevJoint2", builder.joint_label)

    @unittest.skipUnless(USD_AVAILABLE, "Requires usd-core")
    @_expect_jointless_articulation_warning
    def test_import_articulation_parent_offset(self):
        """Combine articulation parent offsets with the import transform."""
        from pxr import Usd

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
        """Combine authored and unitsResolve scale operations for collider dimensions."""
        from pxr import Usd

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
        """Combine nested transform scales for collider dimensions."""
        from pxr import Usd

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

    def test_import_sphere_scale_uniformity_tolerance(self):
        """Treat scales within a relative tolerance as uniform, and larger spreads as non-uniform."""
        # The single-precision transform decomposition emits these for an exactly uniform
        # scale composed through a nested transform chain; they differ by one float32 ULP.
        self.assertTrue(_is_uniform_scale((0.9999999403953552, 0.9999999403953552, 1.0)))
        self.assertTrue(_is_uniform_scale((0.9999999403953552, 1.0, 0.9999999403953552)))
        self.assertTrue(_is_uniform_scale((1.0, 1.0, 1.0)))
        self.assertTrue(_is_uniform_scale((0.0, 0.0, 0.0)))
        # Genuinely non-uniform scales must still be reported.
        self.assertFalse(_is_uniform_scale((1.0, 1.0, 2.0)))
        self.assertFalse(_is_uniform_scale((1.0, 1.0, 1.001)))

    @unittest.skipUnless(USD_AVAILABLE, "Requires usd-core")
    def test_import_sphere_near_uniform_scale_does_not_warn(self):
        """Import spheres whose scale is uniform to within float32 round-off without warning.

        Both the collision and the visual code path guard against non-uniform sphere scaling.
        A scale that is exactly uniform in the source asset can still reach those guards with
        its components a ULP apart, which an exact equality comparison reports as non-uniform.
        """
        from pxr import Usd

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
        def Sphere "NearUniformCollision" (
            prepend apiSchemas = ["PhysicsCollisionAPI"]
        )
        {
            double radius = 0.5
            float3 xformOp:scale = (0.99999994, 0.99999994, 1)
            uniform token[] xformOpOrder = ["xformOp:scale"]
        }

        def Sphere "NearUniformVisual"
        {
            double radius = 0.5
            double3 xformOp:translate = (2, 0, 0)
            float3 xformOp:scale = (0.99999994, 0.99999994, 1)
            uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:scale"]
        }
    }
}
"""
        stage = Usd.Stage.CreateInMemory()
        stage.GetRootLayer().ImportFromString(usd_text)

        builder = newton.ModelBuilder()
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            builder.add_usd(stage)

        self.assertNotIn("Non-uniform scaling of spheres", stdout.getvalue())

    @unittest.skipUnless(USD_AVAILABLE, "Requires usd-core")
    def test_import_sphere_non_uniform_scale_warns(self):
        """Warn, and name the prim, when a sphere really is scaled non-uniformly."""
        from pxr import Usd

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
        def Sphere "SquashedCollision" (
            prepend apiSchemas = ["PhysicsCollisionAPI"]
        )
        {
            double radius = 0.5
            float3 xformOp:scale = (1, 1, 2)
            uniform token[] xformOpOrder = ["xformOp:scale"]
        }
    }
}
"""
        stage = Usd.Stage.CreateInMemory()
        stage.GetRootLayer().ImportFromString(usd_text)

        builder = newton.ModelBuilder()
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            builder.add_usd(stage)

        output = stdout.getvalue()
        self.assertIn("Non-uniform scaling of spheres", output)
        self.assertIn("/World/Body/SquashedCollision", output)

    @unittest.skipUnless(USD_AVAILABLE, "Requires usd-core")
    def test_import_articulation_no_visuals(self):
        """Import the ant articulation with collision shapes but no visuals or sites."""
        builder = newton.ModelBuilder()

        results = builder.add_usd(
            os.path.join(os.path.dirname(__file__), "assets", "ant.usda"),
            collapse_fixed_joints=True,
            load_sites=False,
            load_visual_shapes=False,
        )
        self.assertEqual(builder.body_count, 9)
        self.assertEqual(builder.shape_count, 13)
        self.assertEqual(len(builder.shape_label), len(set(builder.shape_label)))
        self.assertEqual(len(builder.body_label), len(set(builder.body_label)))
        self.assertEqual(len(builder.joint_label), len(set(builder.joint_label)))
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
        """Import an articulation containing a mesh without errors."""
        builder = newton.ModelBuilder()

        _ = builder.add_usd(
            os.path.join(os.path.dirname(__file__), "assets", "simple_articulation_with_mesh.usda"),
            collapse_fixed_joints=True,
        )

    @unittest.skipUnless(USD_AVAILABLE, "Requires usd-core")
    def test_import_revolute_articulation(self):
        """Connect body0 to the world when a fixed joint has no body1.

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
        self.assertEqual(set(builder.body_label), {"/Articulation/Arm", "/Articulation/CenterPivot"})

        # Should have 2 joints:
        # 1. Fixed joint with only body0 specified (CenterPivot to world)
        # 2. Revolute joint between CenterPivot and Arm (normal joint with both bodies)
        self.assertEqual(builder.joint_count, 2)

        # Find joints by their keys to make test robust to ordering changes
        fixed_joint_idx = builder.joint_label.index("/Articulation/CenterPivot/FixedJoint")
        revolute_joint_idx = builder.joint_label.index("/Articulation/Arm/RevoluteJoint")

        # Verify joint types
        self.assertEqual(builder.joint_type[revolute_joint_idx], newton.JointType.REVOLUTE)
        self.assertEqual(builder.joint_type[fixed_joint_idx], newton.JointType.FIXED)

        # The key test: verify the FixedJoint connects CenterPivot to world
        # because body1 was missing in the USD file
        self.assertEqual(builder.joint_parent[fixed_joint_idx], -1)  # Parent is world (-1)
        # Child should be CenterPivot (which was body0 in the USD)
        center_pivot_idx = builder.body_label.index("/Articulation/CenterPivot")
        self.assertEqual(builder.joint_child[fixed_joint_idx], center_pivot_idx)

        # Verify the import results mapping
        self.assertEqual(len(results["path_body_map"]), 2)
        self.assertEqual(len(results["path_shape_map"]), 1)


if __name__ == "__main__":
    unittest.main()
