# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import builtins
import functools
import hashlib
import math
import os
import posixpath
import tempfile
import types
import unittest
import warnings
from unittest import mock
from urllib.parse import urlparse

import numpy as np
import warp as wp

import newton
import newton.examples
import newton.usd as usd
from newton import BodyFlags, JointType
from newton._src.geometry.flags import ShapeFlags
from newton._src.geometry.utils import transform_points
from newton._src.solvers.mujoco.constants import (
    SOLREF_MODE_FORCE_SPACE,
    SOLREF_MODE_MJCF_DEFAULT,
    SOLREF_MODE_RAW,
)
from newton._src.solvers.mujoco.utils import MjcEqualityTargetKind
from newton.math import quat_between_axes
from newton.solvers import SolverMuJoCo
from newton.tests.unittest_utils import USD_AVAILABLE, assert_np_equal, get_test_devices, patch_sys_module

devices = get_test_devices()


_INVALID_ARTICULATION_DESC = "Warning: Invalid ArticulationDesc descriptor"


def _expect_jointless_articulation_warning(test):
    """Require the benign jointless-articulation warning on OpenUSD < 26.0.

    ``UsdPhysics.LoadUsdPhysicsFromRange`` in OpenUSD < 26.0 (e.g. the
    ``usd-exchange`` build resolved on ``aarch64``) reports an articulation root
    that has no joints as an invalid ``ArticulationDesc``, which
    :func:`~newton.utils.parse_usd` surfaces as a ``UserWarning``; usd-core
    >= 26.0 treats it as valid. The fixtures wrapped here intentionally import
    single-body (jointless) articulations -- a shape Newton parses identically
    either way. On the USD versions that emit it, assert exactly that warning
    while leaving every other warning subject to the ambient policy, so an
    unexpected ``newton.*`` warning here still fails under ``--strict-warnings``.
    """

    @functools.wraps(test)
    def wrapper(self, *args, **kwargs):
        from pxr import Usd

        if Usd.GetVersion() >= (0, 26, 0):
            return test(self, *args, **kwargs)
        with warnings.catch_warnings(record=True) as caught:
            # Record (do not escalate) only the expected warning; the inherited
            # "error" filter still applies to everything else under strict mode.
            warnings.filterwarnings("always", message=_INVALID_ARTICULATION_DESC, category=UserWarning)
            result = test(self, *args, **kwargs)
        self.assertTrue(
            any(
                issubclass(w.category, UserWarning) and str(w.message).startswith(_INVALID_ARTICULATION_DESC)
                for w in caught
            ),
            f"expected a {_INVALID_ARTICULATION_DESC!r} warning on OpenUSD < 26.0",
        )
        return result

    return wrapper


class TestImportUsdArticulation(unittest.TestCase):
    @unittest.skipUnless(USD_AVAILABLE, "Requires usd-core")
    def test_import_usd_raises_on_stage_errors(self):
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
    def test_import_body_newton_armature_ignored(self):
        # Body-level newton:armature was removed: an authored value must be
        # ignored without warning and contribute nothing to body inertia.
        # (Joint-level newton:armature is a separate, supported attribute.)
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
        """A USD fixed joint to world remains rootless and finalizes normally."""
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
        """A root joint without ArticulationRootAPI must not split the mechanism."""
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
        """Multiple world joints on one orphan body retain all MJCF DOFs."""
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
        """Floating bodies outside authored articulations get standalone articulations."""
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
        """Joints outside any articulation must not be silently dropped.
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
        """A stray authored joint under no articulation root must not suppress base-joint
        creation for unrelated floating bodies. Regression test for issue #3002.
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
        """A body connected to the world via a PhysicsFixedJoint must be imported
        with a FIXED joint (not FREE) without synthesizing a new articulation."""
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
        """Orphan body-to-world fixed joints keep env-origin + spawn xform."""
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
        """collapse_fixed_joints must not drop orphan joints or their bodies."""
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


class TestImportUsdJoints(unittest.TestCase):
    @unittest.skipUnless(USD_AVAILABLE, "Requires usd-core")
    def test_distance_joint_label(self):
        from pxr import Usd, UsdGeom, UsdPhysics

        stage = Usd.Stage.CreateInMemory()
        articulation = UsdGeom.Xform.Define(stage, "/World")
        UsdPhysics.ArticulationRootAPI.Apply(articulation.GetPrim())

        body0 = UsdGeom.Xform.Define(stage, "/World/Body0")
        UsdPhysics.RigidBodyAPI.Apply(body0.GetPrim())
        body1 = UsdGeom.Xform.Define(stage, "/World/Body1")
        UsdPhysics.RigidBodyAPI.Apply(body1.GetPrim())

        joint = UsdPhysics.DistanceJoint.Define(stage, "/World/DistanceJoint")
        joint.CreateBody0Rel().SetTargets([body0.GetPath()])
        joint.CreateBody1Rel().SetTargets([body1.GetPath()])

        builder = newton.ModelBuilder()
        builder.add_usd(stage)

        self.assertIn("/World/DistanceJoint", builder.joint_label)

    @unittest.skipUnless(USD_AVAILABLE, "Requires usd-core")
    def test_joint_collision_enabled(self):
        from pxr import Usd, UsdGeom, UsdPhysics

        def build(joints, *, enable_self_collisions=True):
            stage = Usd.Stage.CreateInMemory()
            articulation = UsdGeom.Xform.Define(stage, "/World")
            UsdPhysics.ArticulationRootAPI.Apply(articulation.GetPrim())

            bodies = []
            for name in ("Body0", "Body1"):
                body = UsdGeom.Cube.Define(stage, f"/World/{name}")
                UsdPhysics.RigidBodyAPI.Apply(body.GetPrim())
                UsdPhysics.CollisionAPI.Apply(body.GetPrim())
                bodies.append(body)

            for joint_type, name, collision_enabled in joints:
                joint = joint_type.Define(stage, f"/World/{name}")
                joint.CreateBody0Rel().SetTargets([bodies[0].GetPath()])
                joint.CreateBody1Rel().SetTargets([bodies[1].GetPath()])
                joint.CreateCollisionEnabledAttr().Set(collision_enabled)

            builder = newton.ModelBuilder()
            builder.add_usd(stage, enable_self_collisions=enable_self_collisions)
            shape_pair = tuple(sorted(builder.shape_label.index(str(body.GetPath())) for body in bodies))
            return builder, shape_pair

        for collision_enabled in (False, True):
            with self.subTest(collision_enabled=collision_enabled):
                builder, shape_pair = build([(UsdPhysics.RevoluteJoint, "Joint", collision_enabled)])
                self.assertEqual(
                    shape_pair in builder.shape_collision_filter_pairs,
                    not collision_enabled,
                )

        for collision_values in ((True, True), (True, False), (False, True)):
            with self.subTest(merged_collision_enabled=collision_values):
                builder, shape_pair = build(
                    [
                        (UsdPhysics.RevoluteJoint, "Angular", collision_values[0]),
                        (UsdPhysics.PrismaticJoint, "Linear", collision_values[1]),
                    ]
                )
                self.assertEqual(
                    shape_pair in builder.shape_collision_filter_pairs,
                    not all(collision_values),
                )

        builder, shape_pair = build(
            [(UsdPhysics.RevoluteJoint, "Joint", True)],
            enable_self_collisions=False,
        )
        self.assertIn(shape_pair, builder.shape_collision_filter_pairs)

    @unittest.skipUnless(USD_AVAILABLE, "Requires usd-core")
    def test_world_joint_does_not_filter_collisions(self):
        from pxr import Usd, UsdGeom, UsdPhysics

        for joint_type in (UsdPhysics.FixedJoint, UsdPhysics.RevoluteJoint):
            for collision_enabled in (False, True):
                with self.subTest(joint_type=joint_type, collision_enabled=collision_enabled):
                    stage = Usd.Stage.CreateInMemory()
                    articulation = UsdGeom.Xform.Define(stage, "/World")
                    UsdPhysics.ArticulationRootAPI.Apply(articulation.GetPrim())

                    ground = UsdGeom.Cube.Define(stage, "/Ground")
                    UsdPhysics.CollisionAPI.Apply(ground.GetPrim())
                    body = UsdGeom.Cube.Define(stage, "/World/Body")
                    UsdPhysics.RigidBodyAPI.Apply(body.GetPrim())
                    UsdPhysics.CollisionAPI.Apply(body.GetPrim())

                    joint = joint_type.Define(stage, "/World/Joint")
                    joint.CreateBody1Rel().SetTargets([body.GetPath()])
                    joint.CreateCollisionEnabledAttr().Set(collision_enabled)

                    builder = newton.ModelBuilder()
                    builder.add_usd(stage)
                    shape_pair = tuple(
                        sorted(builder.shape_label.index(str(prim.GetPath())) for prim in (ground, body))
                    )
                    self.assertNotIn(shape_pair, builder.shape_collision_filter_pairs)

    @unittest.skipUnless(USD_AVAILABLE, "Requires usd-core")
    def test_newton_joint_api_parsing(self):
        """NewtonJointAPI broadcast attributes parse onto a revolute joint, including sentinels."""
        from pxr import Usd

        from newton._src.utils.import_usd import _HARD_LIMIT_KE  # noqa: PLC0415

        deg2rad = math.pi / 180.0

        # Joint1: concrete NewtonJointAPI values. Joint2: hard limit (limitStiffness=inf).
        # Joint3: engine defaults (limitStiffness/limitDamping = -inf, nothing else authored).
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

        def Sphere "Collision1" (
            prepend apiSchemas = ["PhysicsCollisionAPI"]
        )
        {
            double radius = 0.1
        }
    }

    def PhysicsRevoluteJoint "Joint1"
    {
        rel physics:body0 = </Articulation/Body1>
        token physics:axis = "Z"
        float physics:lowerLimit = -45
        float physics:upperLimit = 45
        float newton:armature = 0.5
        float newton:friction = 0.1
        float newton:damping = 2.0
        float newton:velocityLimit = 100.0
        float newton:limitStiffness = 200.0
        float newton:limitDamping = 5.0
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
        rel physics:body0 = </Articulation/Body1>
        rel physics:body1 = </Articulation/Body2>
        token physics:axis = "Z"
        float physics:lowerLimit = -30
        float physics:upperLimit = 30
        float newton:limitStiffness = inf
        float newton:limitDamping = -inf
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
        token physics:axis = "Z"
        float physics:lowerLimit = -60
        float physics:upperLimit = 60
        float newton:limitStiffness = -inf
        float newton:limitDamping = -inf
    }
}
"""
        stage = Usd.Stage.CreateInMemory()
        stage.GetRootLayer().ImportFromString(usd_content)

        builder = newton.ModelBuilder()
        builder.add_usd(stage)
        model = builder.finalize()

        default_ke = builder.default_joint_cfg.limit_ke
        default_kd = builder.default_joint_cfg.limit_kd

        qd_start = model.joint_qd_start.numpy()
        limit_ke = model.joint_limit_ke.numpy()
        limit_kd = model.joint_limit_kd.numpy()
        damping = model.joint_damping.numpy()
        armature = model.joint_armature.numpy()
        friction = model.joint_friction.numpy()
        velocity_limit = model.joint_velocity_limit.numpy()

        def dof(label):
            return int(qd_start[model.joint_label.index(label)])

        # Joint1: concrete values. Angular gains are authored per-degree and stored per-radian.
        d1 = dof("/Articulation/Joint1")
        self.assertAlmostEqual(float(limit_ke[d1]), 200.0 / deg2rad, places=2)
        self.assertAlmostEqual(float(limit_kd[d1]), 5.0 / deg2rad, places=3)
        self.assertAlmostEqual(float(damping[d1]), 2.0 / deg2rad, places=3)
        self.assertAlmostEqual(float(armature[d1]), 0.5, places=5)
        self.assertAlmostEqual(float(friction[d1]), 0.1, places=5)
        self.assertAlmostEqual(float(velocity_limit[d1]), 100.0 * deg2rad, places=5)

        # Joint2: limitStiffness=inf -> hard limit, limitDamping forced to 0.
        d2 = dof("/Articulation/Joint2")
        self.assertAlmostEqual(float(limit_ke[d2]), _HARD_LIMIT_KE / deg2rad, delta=100.0)
        self.assertEqual(float(limit_kd[d2]), 0.0)

        # Joint3: -inf sentinels -> builder defaults.
        d3 = dof("/Articulation/Joint3")
        self.assertAlmostEqual(float(limit_ke[d3]), default_ke, places=2)
        self.assertAlmostEqual(float(limit_kd[d3]), default_kd, places=2)

    @unittest.skipUnless(USD_AVAILABLE, "Requires usd-core")
    def test_newton_joint_api_prismatic(self):
        """NewtonJointAPI attributes parse onto a prismatic joint without per-degree conversion."""
        from pxr import Usd

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

        def Sphere "Collision1" (
            prepend apiSchemas = ["PhysicsCollisionAPI"]
        )
        {
            double radius = 0.1
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
        }
    }

    def PhysicsPrismaticJoint "Joint1"
    {
        rel physics:body0 = </Articulation/Body1>
        rel physics:body1 = </Articulation/Body2>
        token physics:axis = "X"
        float physics:lowerLimit = -1
        float physics:upperLimit = 1
        float newton:armature = 0.5
        float newton:friction = 0.1
        float newton:damping = 2.0
        float newton:velocityLimit = 100.0
        float newton:limitStiffness = 200.0
        float newton:limitDamping = 5.0
    }
}
"""
        stage = Usd.Stage.CreateInMemory()
        stage.GetRootLayer().ImportFromString(usd_content)

        builder = newton.ModelBuilder()
        builder.add_usd(stage)
        model = builder.finalize()

        qd_start = model.joint_qd_start.numpy()
        d = int(qd_start[model.joint_label.index("/Articulation/Joint1")])

        # Linear DOFs carry the authored values directly (no per-degree conversion).
        self.assertAlmostEqual(float(model.joint_limit_ke.numpy()[d]), 200.0, places=2)
        self.assertAlmostEqual(float(model.joint_limit_kd.numpy()[d]), 5.0, places=3)
        self.assertAlmostEqual(float(model.joint_damping.numpy()[d]), 2.0, places=3)
        self.assertAlmostEqual(float(model.joint_armature.numpy()[d]), 0.5, places=5)
        self.assertAlmostEqual(float(model.joint_friction.numpy()[d]), 0.1, places=5)
        self.assertAlmostEqual(float(model.joint_velocity_limit.numpy()[d]), 100.0, places=5)

    @unittest.skipUnless(USD_AVAILABLE, "Requires usd-core")
    def test_newton_joint_api_revolute_default_damping(self):
        """builder.default_joint_cfg.damping is used as-is when newton:damping is not authored.

        Regression: the importer previously divided the builder default by DegreesToRadian,
        producing an incorrect value (e.g. 3.0 → ~171.9) for revolute joints.
        """
        from pxr import Usd

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

        def Sphere "Collision1" (
            prepend apiSchemas = ["PhysicsCollisionAPI"]
        )
        {
            double radius = 0.1
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
        }
    }

    def PhysicsRevoluteJoint "Joint1"
    {
        rel physics:body0 = </Articulation/Body1>
        rel physics:body1 = </Articulation/Body2>
        token physics:axis = "Z"
        float physics:lowerLimit = -90
        float physics:upperLimit = 90
        # newton:damping intentionally omitted — builder default must be used without conversion
    }
}
"""
        stage = Usd.Stage.CreateInMemory()
        stage.GetRootLayer().ImportFromString(usd_content)

        builder = newton.ModelBuilder()
        builder.default_joint_cfg.damping = 3.0
        builder.add_usd(stage)
        model = builder.finalize()

        qd_start = model.joint_qd_start.numpy()
        damping = model.joint_damping.numpy()
        d = int(qd_start[model.joint_label.index("/Articulation/Joint1")])
        self.assertAlmostEqual(float(damping[d]), 3.0, places=6)

    @unittest.skipUnless(USD_AVAILABLE, "Requires usd-core")
    def test_merged_joint_revolute_default_damping(self):
        """Merged-joint (D6 consolidation) path uses builder default damping as-is.

        Regression: when two single-DOF joints between the same body pair are
        merged into one D6 joint, the revolute (angular) DOF ran an unconditional
        j_damping /= DegreesToRadian on the builder default (already per-radian),
        producing e.g. 3.0 -> ~171.9. With no newton:damping authored, the builder
        default must flow through unchanged for both the linear and angular DOFs.
        """
        from pxr import Gf, Usd, UsdGeom, UsdPhysics

        stage = Usd.Stage.CreateInMemory()
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
        UsdPhysics.Scene.Define(stage, "/physicsScene")

        body = UsdGeom.Cube.Define(stage, "/World/Body")
        UsdPhysics.RigidBodyAPI.Apply(body.GetPrim())
        UsdPhysics.CollisionAPI.Apply(body.GetPrim())

        # Two single-DOF joints on the same body pair -> merged into one D6 joint,
        # exercising parse_merged_joints. A revolute DOF is required to reach the
        # angular unit-conversion block.
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
        # newton:damping intentionally omitted on both joints.

        builder = newton.ModelBuilder()
        builder.default_joint_cfg.damping = 3.0
        result = builder.add_usd(stage, load_visual_shapes=False)
        model = builder.finalize()

        # Both joints must have merged into a single D6 joint (1 linear + 1 angular DOF).
        self.assertEqual(builder.joint_type, [newton.JointType.D6])
        self.assertEqual(builder.joint_dof_dim, [(1, 1)])
        merged_joint = result["path_joint_map"][hinge.GetPath().pathString]
        self.assertEqual(result["path_joint_map"][slide.GetPath().pathString], merged_joint)

        # Both DOFs (linear DOF first, angular DOF second) must carry the builder
        # default unchanged; the revolute DOF in particular must NOT be scaled by
        # 1 / DegreesToRadian.
        qd_start = int(model.joint_qd_start.numpy()[merged_joint])
        damping = model.joint_damping.numpy()
        self.assertAlmostEqual(float(damping[qd_start]), 3.0, places=6)  # linear DOF
        self.assertAlmostEqual(float(damping[qd_start + 1]), 3.0, places=6)  # angular DOF

    @unittest.skipUnless(USD_AVAILABLE, "Requires usd-core")
    def test_newton_joint_api_d6(self):
        """NewtonJointAPI attributes broadcast uniformly across a D6 joint's linear and angular DOFs."""
        from pxr import Usd

        deg2rad = math.pi / 180.0

        # PhysicsJoint with one free translation DOF (transX) and one free rotation DOF (rotX).
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

        def Sphere "Collision1" (
            prepend apiSchemas = ["PhysicsCollisionAPI"]
        )
        {
            double radius = 0.1
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
        }
    }

    def PhysicsJoint "Joint1" (
        prepend apiSchemas = ["PhysicsLimitAPI:transX", "PhysicsLimitAPI:rotX"]
    )
    {
        rel physics:body0 = </Articulation/Body1>
        rel physics:body1 = </Articulation/Body2>
        float limit:transX:physics:low = -1
        float limit:transX:physics:high = 1
        float limit:rotX:physics:low = -45
        float limit:rotX:physics:high = 45
        float newton:armature = 0.5
        float newton:friction = 0.1
        float newton:damping = 2.0
        float newton:velocityLimit = 100.0
        float newton:limitStiffness = 200.0
        float newton:limitDamping = 5.0
    }
}
"""
        stage = Usd.Stage.CreateInMemory()
        stage.GetRootLayer().ImportFromString(usd_content)

        builder = newton.ModelBuilder()
        builder.add_usd(stage)
        model = builder.finalize()

        joint_idx = model.joint_label.index("/Articulation/Joint1")
        dof_start = int(model.joint_qd_start.numpy()[joint_idx])

        # The linear transX DOF is created before the angular rotX DOF.
        d_lin = dof_start
        d_ang = dof_start + 1

        limit_ke = model.joint_limit_ke.numpy()
        limit_kd = model.joint_limit_kd.numpy()
        damping = model.joint_damping.numpy()
        armature = model.joint_armature.numpy()
        friction = model.joint_friction.numpy()
        velocity_limit = model.joint_velocity_limit.numpy()

        # Linear DOF: authored values applied directly.
        self.assertAlmostEqual(float(limit_ke[d_lin]), 200.0, places=2)
        self.assertAlmostEqual(float(limit_kd[d_lin]), 5.0, places=3)
        self.assertAlmostEqual(float(damping[d_lin]), 2.0, places=3)
        self.assertAlmostEqual(float(velocity_limit[d_lin]), 100.0, places=5)

        # Angular DOF: gains stored per-radian (converted from per-degree).
        self.assertAlmostEqual(float(limit_ke[d_ang]), 200.0 / deg2rad, places=2)
        self.assertAlmostEqual(float(limit_kd[d_ang]), 5.0 / deg2rad, places=3)
        self.assertAlmostEqual(float(damping[d_ang]), 2.0 / deg2rad, places=3)
        self.assertAlmostEqual(float(velocity_limit[d_ang]), 100.0 * deg2rad, places=5)

        # Armature and friction broadcast uniformly to both DOFs.
        for d in (d_lin, d_ang):
            self.assertAlmostEqual(float(armature[d]), 0.5, places=5)
            self.assertAlmostEqual(float(friction[d]), 0.1, places=5)

    @unittest.skipUnless(USD_AVAILABLE, "Requires usd-core")
    def test_newton_joint_api_velocity_limit_unlimited(self):
        """newton:velocityLimit=inf falls back to the builder default rather than storing inf."""
        from pxr import Usd

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

        def Sphere "Collision1" (
            prepend apiSchemas = ["PhysicsCollisionAPI"]
        )
        {
            double radius = 0.1
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
        }
    }

    def PhysicsRevoluteJoint "Joint1"
    {
        rel physics:body0 = </Articulation/Body1>
        rel physics:body1 = </Articulation/Body2>
        token physics:axis = "Z"
        float physics:lowerLimit = -45
        float physics:upperLimit = 45
        float newton:velocityLimit = inf
    }
}
"""
        stage = Usd.Stage.CreateInMemory()
        stage.GetRootLayer().ImportFromString(usd_content)

        builder = newton.ModelBuilder()
        builder.add_usd(stage)
        model = builder.finalize()

        d = int(model.joint_qd_start.numpy()[model.joint_label.index("/Articulation/Joint1")])
        velocity_limit = float(model.joint_velocity_limit.numpy()[d])

        self.assertNotEqual(velocity_limit, float("inf"))
        self.assertAlmostEqual(velocity_limit, builder.default_joint_cfg.velocity_limit, places=5)

    @unittest.skipUnless(USD_AVAILABLE, "Requires usd-core")
    def test_newton_limit_sentinel_precedence_over_mjc(self):
        """Authored newton:limitStiffness=-inf must select the builder default,
        not fall through to a lower-priority MuJoCo per-DOF gain."""
        from pxr import Sdf, Usd

        from newton._src.usd.schemas import SchemaResolverMjc, SchemaResolverNewton  # noqa: PLC0415

        # Prismatic joint with MjcJointAPI authoring mjc:solreflimit = [0.04, 2]
        # AND Newton authoring limitStiffness = -inf, limitDamping = -inf.
        # Expected: builder defaults win (Newton sentinel overrides MuJoCo).
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
        def Sphere "Collision1" (
            prepend apiSchemas = ["PhysicsCollisionAPI"]
        )
        {
            double radius = 0.1
        }
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

    def PhysicsPrismaticJoint "Joint" (
        prepend apiSchemas = ["MjcJointAPI"]
    )
    {
        rel physics:body0 = </Articulation/Body1>
        rel physics:body1 = </Articulation/Body2>
        token physics:axis = "X"
        float physics:lowerLimit = -1
        float physics:upperLimit = 1
        uniform double[] mjc:solreflimit = [0.04, 2]
    }
}
"""
        stage = Usd.Stage.CreateInMemory()
        stage.GetRootLayer().ImportFromString(usd_content)
        # Author Newton sentinels on the joint prim.
        joint_prim = stage.GetPrimAtPath("/Articulation/Joint")
        joint_prim.CreateAttribute("newton:limitStiffness", Sdf.ValueTypeNames.Float, custom=True).Set(float("-inf"))
        joint_prim.CreateAttribute("newton:limitDamping", Sdf.ValueTypeNames.Float, custom=True).Set(float("-inf"))

        builder = newton.ModelBuilder()
        SolverMuJoCo.register_custom_attributes(builder)
        builder.default_joint_cfg.limit_ke = 999.0
        builder.default_joint_cfg.limit_kd = 88.0
        builder.add_usd(stage, schema_resolvers=[SchemaResolverNewton(), SchemaResolverMjc()])
        model = builder.finalize()

        dof = int(model.joint_qd_start.numpy()[model.joint_label.index("/Articulation/Joint")])
        # Authored -inf must select builder defaults (999.0 / 88.0), NOT the MuJoCo
        # solreflimit-derived values.
        self.assertAlmostEqual(float(model.joint_limit_ke.numpy()[dof]), 999.0, places=2)
        self.assertAlmostEqual(float(model.joint_limit_kd.numpy()[dof]), 88.0, places=2)

    @unittest.skipUnless(USD_AVAILABLE, "Requires usd-core")
    def test_newton_limit_unset_falls_through_to_mjc(self):
        """When newton:limitStiffness is NOT authored, MuJoCo per-DOF gains
        from mjc:solreflimit must flow through as the fallback."""
        from pxr import Usd

        from newton._src.usd.schemas import SchemaResolverMjc, SchemaResolverNewton  # noqa: PLC0415

        # Prismatic joint with MjcJointAPI authoring solreflimit but NO Newton
        # limitStiffness / limitDamping authored.
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
        def Sphere "Collision1" (
            prepend apiSchemas = ["PhysicsCollisionAPI"]
        )
        {
            double radius = 0.1
        }
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

    def PhysicsPrismaticJoint "Joint" (
        prepend apiSchemas = ["MjcJointAPI"]
    )
    {
        rel physics:body0 = </Articulation/Body1>
        rel physics:body1 = </Articulation/Body2>
        token physics:axis = "X"
        float physics:lowerLimit = -1
        float physics:upperLimit = 1
        uniform double[] mjc:solreflimit = [0.04, 2]
    }
}
"""
        stage = Usd.Stage.CreateInMemory()
        stage.GetRootLayer().ImportFromString(usd_content)

        builder = newton.ModelBuilder()
        SolverMuJoCo.register_custom_attributes(builder)
        builder.default_joint_cfg.limit_ke = 999.0
        builder.default_joint_cfg.limit_kd = 88.0
        builder.add_usd(stage, schema_resolvers=[SchemaResolverNewton(), SchemaResolverMjc()])
        model = builder.finalize()

        dof = int(model.joint_qd_start.numpy()[model.joint_label.index("/Articulation/Joint")])
        # No Newton limitStiffness authored -> MuJoCo solreflimit-derived gains must flow.
        # solreflimit = [0.04, 2] -> ke = 1/(d*d) = 1/0.0016 = 625, kd = 2/(d) = 50
        # (exact values depend on the MuJoCo gain conversion; just verify NOT builder default)
        limit_ke = float(model.joint_limit_ke.numpy()[dof])
        limit_kd = float(model.joint_limit_kd.numpy()[dof])
        self.assertNotAlmostEqual(limit_ke, 999.0, places=0)
        self.assertNotAlmostEqual(limit_kd, 88.0, places=0)

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
            self.assertTrue(builder_dfs.joint_label[i + 1].endswith(expected[i]))

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
            self.assertTrue(builder_bfs.joint_label[i + 1].endswith(expected[i]))

    @unittest.skipUnless(USD_AVAILABLE, "Requires usd-core")
    def test_reversed_joints_in_articulation_raise(self):
        """Ensure reversed joints are reported when encountered in articulations."""
        from pxr import Gf, Usd, UsdGeom, UsdPhysics

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
        from pxr import Gf, Usd, UsdGeom, UsdPhysics

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

        fixed_idx = builder.joint_label.index("/World/Articulation/RootToWorld")
        root_idx = builder.body_label.index("/World/Articulation/Root")
        self.assertEqual(builder.joint_parent[fixed_idx], -1)
        self.assertEqual(builder.joint_child[fixed_idx], root_idx)

    @unittest.skipUnless(USD_AVAILABLE, "Requires usd-core")
    def test_floating_override_replaces_authored_root_joint(self):
        """Explicit floating overrides must not leave a duplicate USD root joint."""
        from pxr import Gf, Usd, UsdGeom, UsdPhysics

        def create_stage():
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
            link = define_body("/World/Articulation/Link")

            root_joint = UsdPhysics.FixedJoint.Define(stage, "/World/Articulation/RootToWorld")
            root_joint.CreateBody1Rel().SetTargets([root.GetPath()])
            root_joint.CreateLocalPos0Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
            root_joint.CreateLocalPos1Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
            root_joint.CreateLocalRot0Attr().Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
            root_joint.CreateLocalRot1Attr().Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))

            child_joint = UsdPhysics.RevoluteJoint.Define(stage, "/World/Articulation/RootToLink")
            child_joint.CreateBody0Rel().SetTargets([root.GetPath()])
            child_joint.CreateBody1Rel().SetTargets([link.GetPath()])
            child_joint.CreateLocalPos0Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
            child_joint.CreateLocalPos1Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
            child_joint.CreateLocalRot0Attr().Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
            child_joint.CreateLocalRot1Attr().Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
            child_joint.CreateAxisAttr().Set("Z")

            return stage

        for floating, expected_type in ((False, newton.JointType.FIXED), (True, newton.JointType.FREE)):
            with self.subTest(floating=floating):
                builder = newton.ModelBuilder()
                builder.add_usd(create_stage(), floating=floating)

                root_idx = builder.body_label.index("/World/Articulation/Root")
                root_joints = [
                    joint_idx for joint_idx, child_idx in enumerate(builder.joint_child) if child_idx == root_idx
                ]

                self.assertEqual(len(root_joints), 1)
                root_joint_idx = root_joints[0]
                self.assertEqual(builder.joint_parent[root_joint_idx], -1)
                self.assertEqual(builder.joint_type[root_joint_idx], expected_type)
                self.assertNotIn("/World/Articulation/RootToWorld", builder.joint_label)

    @unittest.skipUnless(USD_AVAILABLE, "Requires usd-core")
    def test_reversed_joint_unsupported_d6_raises(self):
        """Reversing a D6 joint should raise an error."""
        from pxr import Gf, Usd, UsdGeom, UsdPhysics

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
        from pxr import Gf, Usd, UsdGeom, UsdPhysics

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
                builder.body_label,
                expected_body_keys,
                f"Expected {expected_body_keys} bodies after filtering ({msg}; {bodies_follow_joint_ordering!s}), got {builder.body_label}",
            )
            self.assertEqual(
                builder.joint_label,
                expected_joint_keys,
                f"Expected {expected_joint_keys} joints after filtering ({msg}; {bodies_follow_joint_ordering!s}), got {builder.joint_label}",
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
        from pxr import Usd

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
        self.assertEqual(builder.body_label, ["/Articulation/Body1", "/Articulation/Body2"])
        self.assertEqual(
            builder.joint_label, ["/Articulation/Joint1", "/Articulation/Joint2", "/Articulation/LoopJoint"]
        )
        self.assertEqual(builder.joint_articulation, [0, 0, -1])

    @unittest.skipUnless(USD_AVAILABLE, "Requires usd-core")
    def test_solimp_friction_parsing(self):
        """Test that solimp_friction attribute is parsed correctly from USD."""
        from pxr import Usd

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


class TestImportUsdPhysics(unittest.TestCase):
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
    def test_mass_fallback_instanced_colliders(self):
        """Regression test: bodies with PhysicsMassAPI but no authored mass properties
        and instanceable collision shapes must get positive mass from shape accumulation.

        When collision shapes live inside instanceable prims, USD's
        ComputeMassProperties cannot traverse into them and returns invalid results
        (mass < 0). The importer must fall back to the mass properties already
        accumulated by the builder during add_shape_*() calls, and respect the
        body-level authored density instead of the builder's default shape density.
        """
        from pxr import Usd, UsdGeom, UsdPhysics

        stage = Usd.Stage.CreateInMemory()
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
        UsdPhysics.Scene.Define(stage, "/physicsScene")

        # Create a prototype with a collision sphere (outside the body hierarchy).
        stage.OverridePrim("/Prototype_Collisions")
        radius = 0.5
        sphere = UsdGeom.Sphere.Define(stage, "/Prototype_Collisions/sphere")
        sphere.CreateRadiusAttr().Set(radius)
        UsdPhysics.CollisionAPI.Apply(sphere.GetPrim())

        # Create a rigid body with MassAPI applied and only density authored.
        body_density = 5.0
        body_xform = UsdGeom.Xform.Define(stage, "/World/Body")
        body_prim = body_xform.GetPrim()
        UsdPhysics.RigidBodyAPI.Apply(body_prim)
        mass_api = UsdPhysics.MassAPI.Apply(body_prim)
        mass_api.CreateDensityAttr().Set(body_density)

        # Reference the collision prototype as an instanceable prim.
        collisions = stage.DefinePrim("/World/Body/collisions")
        collisions.GetReferences().AddInternalReference("/Prototype_Collisions")
        collisions.SetInstanceable(True)

        builder = newton.ModelBuilder()
        builder.add_usd(stage)

        self.assertEqual(builder.body_count, 1)
        # Expected mass = body_density * sphere_volume = 5 * (4/3 * pi * 0.5^3).
        expected_mass = body_density * (4.0 / 3.0 * np.pi * radius**3)
        np.testing.assert_allclose(builder.body_mass[0], expected_mass, rtol=1e-5)
        # Verify inertia is also positive (not garbage).
        inertia = np.array(builder.body_inertia[0]).reshape(3, 3)
        self.assertGreater(np.trace(inertia), 0.0, "Body inertia trace must be positive")

    @unittest.skipUnless(USD_AVAILABLE, "Requires usd-core")
    def test_kinematic_enabled_flag(self):
        """USD bodies with physics:kinematicEnabled=true get BodyFlags.KINEMATIC."""
        from pxr import Usd, UsdGeom, UsdPhysics

        stage = Usd.Stage.CreateInMemory()
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
        UsdPhysics.Scene.Define(stage, "/physicsScene")

        # Kinematic root body
        kin_xform = UsdGeom.Xform.Define(stage, "/World/Kinematic")
        kin_prim = kin_xform.GetPrim()
        rb_api = UsdPhysics.RigidBodyAPI.Apply(kin_prim)
        rb_api.CreateKinematicEnabledAttr().Set(True)
        mass_api = UsdPhysics.MassAPI.Apply(kin_prim)
        mass_api.CreateMassAttr().Set(1.0)
        sphere = UsdGeom.Sphere.Define(stage, "/World/Kinematic/Sphere")
        UsdPhysics.CollisionAPI.Apply(sphere.GetPrim())

        # Dynamic body
        dyn_xform = UsdGeom.Xform.Define(stage, "/World/Dynamic")
        dyn_prim = dyn_xform.GetPrim()
        UsdPhysics.RigidBodyAPI.Apply(dyn_prim)
        mass_api2 = UsdPhysics.MassAPI.Apply(dyn_prim)
        mass_api2.CreateMassAttr().Set(1.0)
        sphere2 = UsdGeom.Sphere.Define(stage, "/World/Dynamic/Sphere")
        UsdPhysics.CollisionAPI.Apply(sphere2.GetPrim())

        builder = newton.ModelBuilder()
        builder.add_usd(stage)

        kin_idx = builder.body_label.index("/World/Kinematic")
        dyn_idx = builder.body_label.index("/World/Dynamic")
        self.assertTrue(builder.body_flags[kin_idx] & int(BodyFlags.KINEMATIC))
        self.assertEqual(builder.body_flags[dyn_idx], int(BodyFlags.DYNAMIC))

        model = builder.finalize()
        flags = model.body_flags.numpy()
        self.assertTrue(flags[kin_idx] & int(BodyFlags.KINEMATIC))
        self.assertEqual(flags[dyn_idx], int(BodyFlags.DYNAMIC))

    @unittest.skipUnless(USD_AVAILABLE, "Requires usd-core")
    def test_kinematic_enabled_articulation(self):
        """Kinematic flag is parsed for articulation root bodies from USD."""
        builder = newton.ModelBuilder()
        builder.add_usd(os.path.join(os.path.dirname(__file__), "assets", "actuator_test.usda"))

        base_idx = builder.body_label.index("/World/Robot/Base")
        self.assertTrue(builder.body_flags[base_idx] & int(BodyFlags.KINEMATIC))

        # Non-root links should be dynamic
        link1_idx = builder.body_label.index("/World/Robot/Link1")
        link2_idx = builder.body_label.index("/World/Robot/Link2")
        self.assertEqual(builder.body_flags[link1_idx], int(BodyFlags.DYNAMIC))
        self.assertEqual(builder.body_flags[link2_idx], int(BodyFlags.DYNAMIC))

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
        from pxr import Gf, Usd, UsdGeom, UsdPhysics

        def box_mesh(scale=(1.0, 1.0, 1.0), transform: wp.transform | None = None):
            mesh = newton.Mesh.create_box(
                scale[0],
                scale[1],
                scale[2],
                duplicate_vertices=False,
                compute_normals=False,
                compute_uvs=False,
                compute_inertia=False,
            )
            vertices, indices = mesh.vertices, mesh.indices
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
    def test_mesh_approximation_cfg(self):
        from pxr import Gf, Usd, UsdGeom, UsdPhysics

        def create_collision_mesh(name, approximation_method):
            box = newton.Mesh.create_box(
                1.0,
                1.0,
                1.0,
                duplicate_vertices=False,
                compute_normals=False,
                compute_uvs=False,
                compute_inertia=False,
            )
            mesh = UsdGeom.Mesh.Define(stage, name)
            UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())
            mesh.CreateFaceVertexCountsAttr().Set([3] * (len(box.indices) // 3))
            mesh.CreateFaceVertexIndicesAttr().Set(box.indices.tolist())
            mesh.CreatePointsAttr().Set([Gf.Vec3f(*p) for p in box.vertices.tolist()])
            meshColAPI = UsdPhysics.MeshCollisionAPI.Apply(mesh.GetPrim())
            meshColAPI.GetApproximationAttr().Set(approximation_method)

        stage = Usd.Stage.CreateInMemory()
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
        UsdGeom.SetStageMetersPerUnit(stage, 1.0)
        UsdPhysics.Scene.Define(stage, "/physicsScene")
        create_collision_mesh("/meshDecomposition", UsdPhysics.Tokens.convexDecomposition)
        create_collision_mesh("/meshConvexHull", UsdPhysics.Tokens.convexHull)

        self.assertEqual(newton.ModelBuilder().default_mesh_approximation_cfg.coacd_threshold, 0.05)

        captured = {}
        fake_coacd = types.ModuleType("coacd")
        fake_coacd.Mesh = lambda vertices, indices: (vertices, indices)

        def run_coacd(cmesh, **kwargs):
            captured.update(kwargs)
            return [cmesh]

        fake_coacd.run_coacd = run_coacd

        with patch_sys_module("coacd", fake_coacd):
            builder = newton.ModelBuilder()
            builder.add_usd(stage)
            self.assertEqual(captured["threshold"], 0.05)

            captured.clear()
            builder = newton.ModelBuilder()
            builder.default_mesh_approximation_cfg.coacd_threshold = 0.5
            builder.add_usd(stage)
            self.assertEqual(captured["threshold"], 0.5)

    @unittest.skipUnless(USD_AVAILABLE, "Requires usd-core")
    def test_visual_match_collision_shapes(self):
        builder = newton.ModelBuilder()
        builder.add_usd(newton.examples.get_asset("humanoid.usda"))
        self.assertEqual(builder.shape_count, 38)
        self.assertEqual(builder.body_count, 16)
        visual_shape_keys = [k for k in builder.shape_label if "visuals" in k]
        collision_shape_keys = [k for k in builder.shape_label if "collisions" in k]
        self.assertEqual(len(visual_shape_keys), 19)
        self.assertEqual(len(collision_shape_keys), 19)
        visual_shapes = [i for i, k in enumerate(builder.shape_label) if "visuals" in k]
        # corresponding collision shapes
        collision_shapes = [builder.shape_label.index(k.replace("visuals", "collisions")) for k in visual_shape_keys]
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
        from pxr import Gf, Usd, UsdGeom, UsdPhysics

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
        self.assertEqual(builder.body_label[0], "/World/Box")
        self.assertEqual(builder.shape_label[0], "/World/Box")

        # Ensure the body has a free joint assigned and is in an articulation.
        self.assertEqual(builder.joint_count, 1)
        self.assertEqual(builder.joint_type[0], newton.JointType.FREE)
        self.assertEqual(builder.joint_parent[0], -1)
        self.assertEqual(builder.joint_child[0], 0)
        self.assertEqual(builder.articulation_count, 1)
        self.assertEqual(builder.articulation_label[0], "/World/Box")
        self.assertEqual(builder.joint_articulation, [0])

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
        from pxr import Usd, UsdGeom, UsdPhysics

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
        joint_idx = model.joint_label.index("/joint_AB")
        self.assertEqual(model.joint_type.numpy()[joint_idx], newton.JointType.REVOLUTE)
        joint_dof_idx = model.joint_qd_start.numpy()[joint_idx]
        self.assertEqual(model.joint_effort_limit.numpy()[joint_dof_idx], 24.0)

        # Test prismatic joint (A-C)
        joint_idx_AC = model.joint_label.index("/joint_AC")
        self.assertEqual(model.joint_type.numpy()[joint_idx_AC], newton.JointType.PRISMATIC)
        joint_dof_idx_AC = model.joint_qd_start.numpy()[joint_idx_AC]
        self.assertEqual(model.joint_effort_limit.numpy()[joint_dof_idx_AC], 15.0)

        # Test D6 joint (A-D) - check transX DOF
        joint_idx_AD = model.joint_label.index("/joint_AD")
        self.assertEqual(model.joint_type.numpy()[joint_idx_AD], newton.JointType.D6)
        joint_dof_idx_AD = model.joint_qd_start.numpy()[joint_idx_AD]
        self.assertEqual(model.joint_effort_limit.numpy()[joint_dof_idx_AD], 30.0)

    @unittest.skipUnless(USD_AVAILABLE, "Requires usd-core")
    def test_solimplimit_parsing(self):
        """Test that solimplimit attribute is parsed correctly from USD."""
        from pxr import Usd

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

    @unittest.skipUnless(USD_AVAILABLE, "Requires usd-core")
    def test_solreflimit_parsing(self):
        """Joint mjc:solreflimit on MjcJointAPI must populate joint_limit_ke / joint_limit_kd.

        Uses prismatic joints so the authored solreflimit values flow straight through to
        joint_limit_ke / joint_limit_kd without the revolute degree->radian rescaling that
        import_usd.py applies to angular limits.
        """
        from pxr import Usd

        from newton._src.usd.schemas import SchemaResolverMjc  # noqa: PLC0415

        # Joint1 authors mjc:solreflimit = [0.08, 1]. Joint2 applies MjcJointAPI but omits
        # solreflimit, so it should use MuJoCo's schema default [0.02, 1]. Joint3 has no
        # MjcJointAPI and should preserve the customized ModelBuilder defaults. Joint4
        # authors [0, 0], which is invalid for gain conversion but must remain raw.
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
        def Cube "Collision1" (
            prepend apiSchemas = ["PhysicsCollisionAPI"]
        )
        {
            double size = 0.2
        }
    }

    def PhysicsPrismaticJoint "Joint1" (
        prepend apiSchemas = ["MjcJointAPI"]
    )
    {
        rel physics:body0 = </Articulation/Body1>
        token physics:axis = "X"
        float physics:lowerLimit = -1
        float physics:upperLimit = 1

        uniform double[] mjc:solreflimit = [0.08, 1]
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

    def PhysicsPrismaticJoint "Joint2" (
        prepend apiSchemas = ["MjcJointAPI"]
    )
    {
        rel physics:body0 = </Articulation/Body1>
        rel physics:body1 = </Articulation/Body2>
        token physics:axis = "Z"
        float physics:lowerLimit = -1
        float physics:upperLimit = 1
    }

    def Xform "Body3" (
        prepend apiSchemas = ["PhysicsRigidBodyAPI"]
    )
    {
        double3 xformOp:translate = (2, 0, 0)
        uniform token[] xformOpOrder = ["xformOp:translate"]

        def Sphere "Collision3" (
            prepend apiSchemas = ["PhysicsCollisionAPI"]
        )
        {
            double radius = 0.1
        }
    }

    def PhysicsPrismaticJoint "Joint3"
    {
        rel physics:body0 = </Articulation/Body2>
        rel physics:body1 = </Articulation/Body3>
        token physics:axis = "Y"
        float physics:lowerLimit = -1
        float physics:upperLimit = 1
    }

    def Xform "Body4" (
        prepend apiSchemas = ["PhysicsRigidBodyAPI"]
    )
    {
        double3 xformOp:translate = (3, 0, 0)
        uniform token[] xformOpOrder = ["xformOp:translate"]

        def Sphere "Collision4" (
            prepend apiSchemas = ["PhysicsCollisionAPI"]
        )
        {
            double radius = 0.1
        }
    }

    def PhysicsPrismaticJoint "Joint4" (
        prepend apiSchemas = ["MjcJointAPI"]
    )
    {
        rel physics:body0 = </Articulation/Body3>
        rel physics:body1 = </Articulation/Body4>
        token physics:axis = "X"
        float physics:lowerLimit = -1
        float physics:upperLimit = 1

        uniform double[] mjc:solreflimit = [0, 0]
    }
}
"""
        stage = Usd.Stage.CreateInMemory()
        stage.GetRootLayer().ImportFromString(usd_content)

        builder = newton.ModelBuilder()
        builder.default_joint_cfg.limit_ke = 4321.0
        builder.default_joint_cfg.limit_kd = 43.0
        SolverMuJoCo.register_custom_attributes(builder)
        builder.add_usd(stage, schema_resolvers=[SchemaResolverMjc()])
        model = builder.finalize()

        joint1_idx = model.joint_label.index("/Articulation/Joint1")
        joint2_idx = model.joint_label.index("/Articulation/Joint2")
        joint3_idx = model.joint_label.index("/Articulation/Joint3")
        joint4_idx = model.joint_label.index("/Articulation/Joint4")
        joint_qd_start = model.joint_qd_start.numpy()
        limit_ke = model.joint_limit_ke.numpy()
        limit_kd = model.joint_limit_kd.numpy()
        raw_solreflimit = model.mujoco.solreflimit.numpy()
        solreflimit_mode = model.mujoco.solreflimit_mode.numpy()

        # Joint1: solreflimit=[0.08, 1] -> ke=1/(0.08^2)=156.25, kd=2/0.08=25.0
        dof1 = joint_qd_start[joint1_idx]
        self.assertAlmostEqual(float(limit_ke[dof1]), 156.25, places=4)
        self.assertAlmostEqual(float(limit_kd[dof1]), 25.0, places=4)
        self.assertEqual(int(solreflimit_mode[dof1]), SOLREF_MODE_RAW)

        # Joint2: no solreflimit authored -> MuJoCo default [0.02, 1]
        dof2 = joint_qd_start[joint2_idx]
        self.assertAlmostEqual(float(limit_ke[dof2]), 2500.0, places=4)
        self.assertAlmostEqual(float(limit_kd[dof2]), 100.0, places=4)
        self.assertEqual(int(solreflimit_mode[dof2]), SOLREF_MODE_MJCF_DEFAULT)

        # Joint3: no MjcJointAPI -> customized ModelBuilder defaults
        dof3 = joint_qd_start[joint3_idx]
        self.assertAlmostEqual(float(limit_ke[dof3]), builder.default_joint_cfg.limit_ke, places=4)
        self.assertAlmostEqual(float(limit_kd[dof3]), builder.default_joint_cfg.limit_kd, places=4)
        self.assertEqual(int(solreflimit_mode[dof3]), SOLREF_MODE_FORCE_SPACE)

        # Joint4: authored raw [0, 0] remains raw even though it cannot be converted to gains.
        dof4 = joint_qd_start[joint4_idx]
        np.testing.assert_array_equal(raw_solreflimit[dof4], [0.0, 0.0])
        self.assertEqual(int(solreflimit_mode[dof4]), SOLREF_MODE_RAW)

    @unittest.skipUnless(USD_AVAILABLE, "Requires usd-core")
    def test_solreflimit_mode_respects_resolver_priority(self):
        """Higher-priority authored gains must not be treated as MuJoCo's implicit default."""
        from pxr import Sdf, Usd

        from newton._src.usd.schemas import SchemaResolverMjc, SchemaResolverNewton  # noqa: PLC0415

        usd_content = """#usda 1.0
(
    upAxis = "Z"
)

def Xform "Articulation" (
    prepend apiSchemas = ["PhysicsArticulationRootAPI"]
)
{
    def Xform "Body1" (
        prepend apiSchemas = ["PhysicsRigidBodyAPI"]
    )
    {
        def Sphere "Collision1" (
            prepend apiSchemas = ["PhysicsCollisionAPI"]
        )
        {
            double radius = 0.1
        }
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

    def PhysicsPrismaticJoint "Joint" (
        prepend apiSchemas = ["MjcJointAPI"]
    )
    {
        rel physics:body0 = </Articulation/Body1>
        rel physics:body1 = </Articulation/Body2>
        token physics:axis = "X"
        float physics:lowerLimit = -1
        float physics:upperLimit = 1
    }
}
"""
        stage = Usd.Stage.CreateInMemory()
        stage.GetRootLayer().ImportFromString(usd_content)
        joint_prim = stage.GetPrimAtPath("/Articulation/Joint")
        joint_prim.CreateAttribute("newton:limitStiffness", Sdf.ValueTypeNames.Float, custom=True).Set(2500.0)
        joint_prim.CreateAttribute("newton:limitDamping", Sdf.ValueTypeNames.Float, custom=True).Set(100.0)

        builder = newton.ModelBuilder()
        SolverMuJoCo.register_custom_attributes(builder)
        builder.add_usd(stage, schema_resolvers=[SchemaResolverNewton(), SchemaResolverMjc()])
        model = builder.finalize()

        joint_idx = model.joint_label.index("/Articulation/Joint")
        dof = model.joint_qd_start.numpy()[joint_idx]
        self.assertAlmostEqual(float(model.joint_limit_ke.numpy()[dof]), 2500.0, places=4)
        self.assertAlmostEqual(float(model.joint_limit_kd.numpy()[dof]), 100.0, places=4)
        self.assertEqual(int(model.mujoco.solreflimit_mode.numpy()[dof]), SOLREF_MODE_FORCE_SPACE)

    @unittest.skipUnless(USD_AVAILABLE, "Requires usd-core")
    def test_solreflimit_mode_declared_on_physics_scene(self):
        """A PhysicsScene declaration must be available when joint modes are emitted."""
        from pxr import Usd

        from newton._src.usd.schemas import SchemaResolverMjc  # noqa: PLC0415

        usd_content = """#usda 1.0
(
    upAxis = "Z"
)

def PhysicsScene "physicsScene"
{
    custom int newton:mujoco:solreflimit_mode = 0 (
        customData = {
            string assignment = "model"
            string frequency = "joint_dof"
        }
    )
}

def Xform "Articulation" (
    prepend apiSchemas = ["PhysicsArticulationRootAPI"]
)
{
    def Xform "Body1" (
        prepend apiSchemas = ["PhysicsRigidBodyAPI"]
    )
    {
        def Sphere "Collision1" (
            prepend apiSchemas = ["PhysicsCollisionAPI"]
        )
        {
            double radius = 0.1
        }
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

    def PhysicsPrismaticJoint "Joint" (
        prepend apiSchemas = ["MjcJointAPI"]
    )
    {
        rel physics:body0 = </Articulation/Body1>
        rel physics:body1 = </Articulation/Body2>
        token physics:axis = "X"
        float physics:lowerLimit = -1
        float physics:upperLimit = 1
    }
}
"""
        stage = Usd.Stage.CreateInMemory()
        stage.GetRootLayer().ImportFromString(usd_content)

        builder = newton.ModelBuilder()
        builder.add_usd(stage, schema_resolvers=[SchemaResolverMjc()])
        self.assertIn("mujoco:solreflimit_mode", builder.custom_attributes)
        model = builder.finalize()

        joint_idx = model.joint_label.index("/Articulation/Joint")
        dof = model.joint_qd_start.numpy()[joint_idx]
        self.assertAlmostEqual(float(model.joint_limit_ke.numpy()[dof]), 2500.0, places=4)
        self.assertAlmostEqual(float(model.joint_limit_kd.numpy()[dof]), 100.0, places=4)
        self.assertEqual(int(model.mujoco.solreflimit_mode.numpy()[dof]), SOLREF_MODE_MJCF_DEFAULT)

    @unittest.skipUnless(USD_AVAILABLE, "Requires usd-core")
    def test_solreflimit_parsing_revolute(self):
        """Joint mjc:solreflimit on a revolute joint must produce per-radian limit_ke/_kd.

        mjModel always stores stiffness per-radian for hinge joints regardless of
        ``mjc:compiler:angle``. The USD importer divides revolute and D6-angular
        ``limit_ke``/``limit_kd`` by ``DegreesToRadian`` on the assumption that
        UsdPhysics-authored gains are per-degree. The MJC angular schema entries
        compensate by pre-multiplying so the per-radian value survives. Regression
        for #2536.
        """
        from pxr import Usd

        from newton._src.usd.schemas import SchemaResolverMjc  # noqa: PLC0415

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
        double3 xformOp:translate = (1, 0, 0)
        uniform token[] xformOpOrder = ["xformOp:translate"]

        def Sphere "Collision2" (
            prepend apiSchemas = ["PhysicsCollisionAPI"]
        )
        {
            double radius = 0.1
        }
    }

    def PhysicsRevoluteJoint "Joint1" (
        prepend apiSchemas = ["MjcJointAPI"]
    )
    {
        rel physics:body0 = </Articulation/Body1>
        rel physics:body1 = </Articulation/Body2>
        token physics:axis = "X"
        float physics:lowerLimit = -45
        float physics:upperLimit = 45

        uniform double[] mjc:solreflimit = [0.08, 1]
    }

    def Xform "Body3" (
        prepend apiSchemas = ["PhysicsRigidBodyAPI"]
    )
    {
        double3 xformOp:translate = (2, 0, 0)
        uniform token[] xformOpOrder = ["xformOp:translate"]

        def Sphere "Collision3" (
            prepend apiSchemas = ["PhysicsCollisionAPI"]
        )
        {
            double radius = 0.1
        }
    }

    def PhysicsRevoluteJoint "Joint2" (
        prepend apiSchemas = ["MjcJointAPI"]
    )
    {
        rel physics:body0 = </Articulation/Body2>
        rel physics:body1 = </Articulation/Body3>
        token physics:axis = "Y"
        float physics:lowerLimit = -45
        float physics:upperLimit = 45
    }
}
"""
        stage = Usd.Stage.CreateInMemory()
        stage.GetRootLayer().ImportFromString(usd_content)

        builder = newton.ModelBuilder()
        SolverMuJoCo.register_custom_attributes(builder)
        builder.add_usd(stage, schema_resolvers=[SchemaResolverMjc()])
        model = builder.finalize()

        joint1_idx = model.joint_label.index("/Articulation/Joint1")
        joint2_idx = model.joint_label.index("/Articulation/Joint2")
        joint_qd_start = model.joint_qd_start.numpy()
        dof1 = joint_qd_start[joint1_idx]
        dof2 = joint_qd_start[joint2_idx]
        solreflimit_mode = model.mujoco.solreflimit_mode.numpy()

        # solreflimit=[0.08, 1] -> per-radian ke = 1/0.08^2 = 156.25, kd = 2/0.08 = 25.0.
        # Without the MJC angular compensation, the importer would over-scale by
        # 1/(pi/180) ~= 57.3x giving ke ~= 8952 and kd ~= 1432.
        self.assertAlmostEqual(float(model.joint_limit_ke.numpy()[dof1]), 156.25, places=3)
        self.assertAlmostEqual(float(model.joint_limit_kd.numpy()[dof1]), 25.0, places=3)

        # Missing solreflimit uses MuJoCo's [0.02, 1] default in per-radian units.
        self.assertAlmostEqual(float(model.joint_limit_ke.numpy()[dof2]), 2500.0, places=3)
        self.assertAlmostEqual(float(model.joint_limit_kd.numpy()[dof2]), 100.0, places=3)
        self.assertEqual(int(solreflimit_mode[dof2]), SOLREF_MODE_MJCF_DEFAULT)

    def test_limit_margin_parsing(self):
        """Test importing limit_margin from USD with mjc:margin on joint."""
        from pxr import Sdf, Usd, UsdGeom, UsdPhysics

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
        joint1_prim.CreateAttribute("mjc:margin", Sdf.ValueTypeNames.Double).Set(0.01)

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
        joint2_prim.CreateAttribute("mjc:margin", Sdf.ValueTypeNames.Double).Set(0.02)

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
        from pxr import Usd

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
        from pxr import Usd

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
        from pxr import Usd

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
    def test_shape_gap_from_usd(self):
        """Test that mjc:gap attribute is parsed into shape_gap from USD."""
        from pxr import Usd

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

        from newton._src.usd.schemas import SchemaResolverMjc  # noqa: PLC0415

        builder = newton.ModelBuilder()
        SolverMuJoCo.register_custom_attributes(builder)
        builder.add_usd(stage, schema_resolvers=[SchemaResolverMjc()])
        model = builder.finalize()

        shape_gap = model.shape_gap.numpy()

        def floats_match(arr, expected, tol=1e-4):
            return abs(arr - expected) < tol

        # Check that we have shapes with expected values
        expected_explicit_1 = 0.8
        expected_explicit_2 = 0.7

        # Find shapes matching each expected value
        found_explicit_1 = any(floats_match(shape_gap[i], expected_explicit_1) for i in range(model.shape_count))
        found_explicit_2 = any(floats_match(shape_gap[i], expected_explicit_2) for i in range(model.shape_count))

        self.assertTrue(found_explicit_1, f"Expected gap {expected_explicit_1} not found in model")
        self.assertTrue(found_explicit_2, f"Expected gap {expected_explicit_2} not found in model")

    @unittest.skipUnless(USD_AVAILABLE, "Requires usd-core")
    def test_margin_gap_combined_conversion(self):
        """Test legacy MuJoCo->Newton conversion via legacy_margin_gap=True.

        Verifies that newton_margin = mjc_margin - mjc_gap when legacy_margin_gap
        is enabled.  Also tests the case where only mjc:margin is authored
        (gap defaults to 0, so no subtraction effect).
        """
        from pxr import Sdf, Usd, UsdPhysics

        stage = Usd.Stage.CreateInMemory()
        UsdPhysics.Scene.Define(stage, "/physicsScene")

        # Body 1: both mjc:margin and mjc:gap authored
        prim1 = stage.DefinePrim("/Body1", "Xform")
        UsdPhysics.RigidBodyAPI.Apply(prim1)
        UsdPhysics.ArticulationRootAPI.Apply(prim1)
        col1 = stage.DefinePrim("/Body1/Collision1", "Cube")
        UsdPhysics.CollisionAPI.Apply(col1)
        col1.GetAttribute("size").Set(0.2)
        col1.CreateAttribute("mjc:margin", Sdf.ValueTypeNames.Double).Set(0.5)
        col1.CreateAttribute("mjc:gap", Sdf.ValueTypeNames.Double).Set(0.2)
        col1.CreateAttribute("newton:contactMargin", Sdf.ValueTypeNames.Double).Set(0.7)

        # Body 2: only mjc:margin authored (gap defaults to 0)
        prim2 = stage.DefinePrim("/Body2", "Xform")
        UsdPhysics.RigidBodyAPI.Apply(prim2)
        col2 = stage.DefinePrim("/Body2/Collision2", "Sphere")
        UsdPhysics.CollisionAPI.Apply(col2)
        col2.GetAttribute("radius").Set(0.1)
        col2.CreateAttribute("mjc:margin", Sdf.ValueTypeNames.Double).Set(0.4)

        # Joint connecting them
        joint = UsdPhysics.RevoluteJoint.Define(stage, "/Joint1")
        joint.GetBody0Rel().SetTargets(["/Body1"])
        joint.GetBody1Rel().SetTargets(["/Body2"])
        joint.GetAxisAttr().Set("Z")

        from newton._src.usd.schemas import SchemaResolverMjc, SchemaResolverNewton  # noqa: PLC0415

        builder = newton.ModelBuilder()
        SolverMuJoCo.register_custom_attributes(builder)
        builder.add_usd(
            stage,
            schema_resolvers=[SchemaResolverMjc(), SchemaResolverNewton()],
            legacy_margin_gap=True,
        )
        model = builder.finalize()

        shape_margin = model.shape_margin.numpy()
        shape_gap = model.shape_gap.numpy()

        # Body 1: mjc_margin=0.5, mjc_gap=0.2 -> newton_margin = 0.5 - 0.2 = 0.3
        found_combined = any(
            abs(float(shape_margin[i]) - 0.3) < 1e-4 and abs(float(shape_gap[i]) - 0.2) < 1e-4
            for i in range(model.shape_count)
        )
        self.assertTrue(found_combined, "Expected margin=0.3, gap=0.2 from combined legacy conversion")

        # Body 2: mjc_margin=0.4, mjc_gap not authored -> gap defaults to 0.0
        # from SchemaResolverMjc, so newton_margin = 0.4 - 0 = 0.4, gap = 0.0
        found_margin_only = any(
            abs(float(shape_margin[i]) - 0.4) < 1e-4 and abs(float(shape_gap[i])) < 1e-4
            for i in range(model.shape_count)
        )
        self.assertTrue(found_margin_only, "Expected margin=0.4 with gap=0.0 when only margin authored")

        builder = newton.ModelBuilder()
        SolverMuJoCo.register_custom_attributes(builder)
        builder.add_usd(
            stage,
            schema_resolvers=[SchemaResolverNewton(), SchemaResolverMjc()],
            legacy_margin_gap=True,
        )
        shape_idx = builder.shape_label.index("/Body1/Collision1")
        self.assertAlmostEqual(builder.shape_margin[shape_idx], 0.7)

    @unittest.skipUnless(USD_AVAILABLE, "Requires usd-core")
    def test_usd_margin_gap_identity_import(self):
        """USD import of mjc:margin and mjc:gap is identity under MuJoCo 3.9
        semantics (margin/gap mean the same as Newton's shape_margin/shape_gap)."""
        from pxr import Sdf, Usd, UsdGeom, UsdPhysics

        from newton._src.usd.schemas import SchemaResolverMjc  # noqa: PLC0415

        stage = Usd.Stage.CreateInMemory()
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
        UsdPhysics.Scene.Define(stage, "/physicsScene")

        body = stage.DefinePrim("/Body", "Xform")
        UsdPhysics.RigidBodyAPI.Apply(body)
        UsdPhysics.ArticulationRootAPI.Apply(body)
        col = stage.DefinePrim("/Body/Collision", "Cube")
        UsdPhysics.CollisionAPI.Apply(col)
        col.GetAttribute("size").Set(0.2)
        col.CreateAttribute("mjc:margin", Sdf.ValueTypeNames.Float).Set(0.5)
        col.CreateAttribute("mjc:gap", Sdf.ValueTypeNames.Float).Set(0.2)

        builder = newton.ModelBuilder()
        SolverMuJoCo.register_custom_attributes(builder)
        builder.add_usd(stage, schema_resolvers=[SchemaResolverMjc()])
        model = builder.finalize()

        shape_margin = model.shape_margin.numpy()
        shape_gap = model.shape_gap.numpy()
        found = any(
            abs(float(shape_margin[i]) - 0.5) < 1e-5 and abs(float(shape_gap[i]) - 0.2) < 1e-5
            for i in range(model.shape_count)
        )
        self.assertTrue(found, "Expected identity: margin=0.5, gap=0.2 (MuJoCo 3.9 default semantics)")

    @unittest.skipUnless(USD_AVAILABLE, "Requires usd-core")
    def test_actuator_mode_inference_from_drive(self):
        """Test that JointTargetMode is correctly inferred from USD joint drives."""
        from pxr import Usd

        from newton._src.sim.enums import JointTargetMode  # noqa: PLC0415

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
            joint_idx = b.joint_label.index(joint_name)
            return sum(b.joint_dof_dim[i][0] + b.joint_dof_dim[i][1] for i in range(joint_idx))

        self.assertEqual(
            builder.joint_target_mode[get_qd_start(builder, "/Root/joint_effort")],
            int(JointTargetMode.EFFORT),
        )
        self.assertEqual(
            builder.joint_target_mode[get_qd_start(builder, "/Root/joint_passive")],
            int(JointTargetMode.NONE),
        )
        self.assertEqual(
            builder.joint_target_mode[get_qd_start(builder, "/Root/joint_position")],
            int(JointTargetMode.POSITION),
        )
        self.assertEqual(
            builder.joint_target_mode[get_qd_start(builder, "/Root/joint_velocity")],
            int(JointTargetMode.VELOCITY),
        )
        self.assertEqual(
            builder.joint_target_mode[get_qd_start(builder, "/Root/joint_both_gains")],
            int(JointTargetMode.POSITION),
        )

        stage2 = Usd.Stage.CreateInMemory()
        stage2.GetRootLayer().ImportFromString(usd_content)

        builder2 = newton.ModelBuilder()
        builder2.add_usd(stage2, force_position_velocity_actuation=True)

        self.assertEqual(
            builder2.joint_target_mode[get_qd_start(builder2, "/Root/joint_both_gains")],
            int(JointTargetMode.POSITION_VELOCITY),
        )
        self.assertEqual(
            builder2.joint_target_mode[get_qd_start(builder2, "/Root/joint_position")],
            int(JointTargetMode.POSITION),
        )
        self.assertEqual(
            builder2.joint_target_mode[get_qd_start(builder2, "/Root/joint_velocity")],
            int(JointTargetMode.VELOCITY),
        )

    def test_add_base_joint_default(self):
        """Test add_base_joint with default parameters creates a free joint."""
        builder = newton.ModelBuilder()
        body0 = builder.add_link(xform=wp.transform((1.0, 2.0, 3.0), wp.quat_identity()))
        builder.body_mass[body0] = 1.0  # Set mass

        joint_id = builder._add_base_joint(body0)

        self.assertEqual(builder.joint_count, 1)
        self.assertEqual(builder.joint_type[joint_id], newton.JointType.FREE)
        self.assertEqual(builder.joint_child[joint_id], body0)
        self.assertEqual(builder.joint_parent[joint_id], -1)

    def test_add_base_joint_fixed(self):
        """Test add_base_joint with floating=False creates a fixed joint."""
        builder = newton.ModelBuilder()
        body0 = builder.add_link(xform=wp.transform((1.0, 2.0, 3.0), wp.quat_identity()))
        builder.body_mass[body0] = 1.0

        joint_id = builder._add_base_joint(body0, floating=False)

        self.assertEqual(builder.joint_count, 1)
        self.assertEqual(builder.joint_type[joint_id], newton.JointType.FIXED)
        self.assertEqual(builder.joint_child[joint_id], body0)
        self.assertEqual(builder.joint_parent[joint_id], -1)

    def test_add_base_joint_dict(self):
        """Test _add_base_joint with base_joint dict creates a D6 joint."""
        builder = newton.ModelBuilder()
        body0 = builder.add_link(xform=wp.transform((1.0, 2.0, 3.0), wp.quat_identity()))
        builder.body_mass[body0] = 1.0

        joint_id = builder._add_base_joint(
            body0,
            base_joint={
                "joint_type": newton.JointType.D6,
                "linear_axes": [
                    newton.ModelBuilder.JointDofConfig(axis=[1.0, 0.0, 0.0]),
                    newton.ModelBuilder.JointDofConfig(axis=[0.0, 1.0, 0.0]),
                ],
                "angular_axes": [newton.ModelBuilder.JointDofConfig(axis=[0.0, 0.0, 1.0])],
            },
        )

        self.assertEqual(builder.joint_count, 1)
        self.assertEqual(builder.joint_type[joint_id], newton.JointType.D6)
        self.assertEqual(builder.joint_child[joint_id], body0)
        self.assertEqual(builder.joint_parent[joint_id], -1)

    def test_add_base_joint_dict_revolute(self):
        """Test _add_base_joint with base_joint dict creates a revolute joint with custom axis."""
        builder = newton.ModelBuilder()
        body0 = builder.add_link(xform=wp.transform((1.0, 2.0, 3.0), wp.quat_identity()))
        builder.body_mass[body0] = 1.0

        joint_id = builder._add_base_joint(
            body0,
            base_joint={
                "joint_type": newton.JointType.REVOLUTE,
                "angular_axes": [newton.ModelBuilder.JointDofConfig(axis=(0, 0, 1))],
            },
        )

        self.assertEqual(builder.joint_count, 1)
        self.assertEqual(builder.joint_type[joint_id], newton.JointType.REVOLUTE)
        self.assertEqual(builder.joint_child[joint_id], body0)
        self.assertEqual(builder.joint_parent[joint_id], -1)

    def test_add_base_joint_custom_label(self):
        """Test add_base_joint with custom label."""
        builder = newton.ModelBuilder()
        body0 = builder.add_link(xform=wp.transform((1.0, 2.0, 3.0), wp.quat_identity()))
        builder.body_mass[body0] = 1.0

        joint_id = builder._add_base_joint(body0, label="my_custom_joint")

        self.assertEqual(builder.joint_count, 1)
        self.assertEqual(builder.joint_label[joint_id], "my_custom_joint")


def verify_usdphysics_parser(test, file, model, compare_min_max_coords, floating):
    """Verify model based on the UsdPhysics Parsing Utils"""
    # [1] https://openusd.org/release/api/usd_physics_page_front.html
    from pxr import Sdf, Usd, UsdPhysics

    stage = Usd.Stage.Open(file)
    parsed = UsdPhysics.LoadUsdPhysicsFromRange(stage, ["/"])
    # since the key is generated from USD paths we can assume that keys are unique
    body_key_to_idx = dict(zip(model.body_label, range(model.body_count), strict=False))
    shape_key_to_idx = dict(zip(model.shape_label, range(model.shape_count), strict=False))

    parsed_bodies = list(zip(*parsed.get(UsdPhysics.ObjectType.RigidBody, ()), strict=False))

    # body presence
    for body_path, _ in parsed_bodies:
        assert body_key_to_idx.get(str(body_path), None) is not None
    test.assertEqual(len(parsed_bodies), model.body_count)

    # body colliders
    # TODO: exclude or handle bodies that have child shapes
    for body_path, body_desc in parsed_bodies:
        body_idx = body_key_to_idx.get(str(body_path), None)

        model_collisions = {model.shape_label[sk] for sk in model.body_shapes[body_idx]}
        parsed_collisions = {str(collider) for collider in body_desc.collisions}
        test.assertEqual(parsed_collisions, model_collisions)

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
                test.assertAlmostEqual(body_mass[body_idx], mass, places=5)
            if mass_api.GetDiagonalInertiaAttr().HasAuthoredValue():
                diag_inertia = mass_api.GetDiagonalInertiaAttr().Get()
                principal_axes = mass_api.GetPrincipalAxesAttr().Get().Normalize()
                p = np.array(wp.quat_to_matrix(wp.quat(*principal_axes.imaginary, principal_axes.real))).reshape((3, 3))
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

    joint_key_to_idx = dict(zip(model.joint_label, range(model.joint_count), strict=False))
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
    test.assertEqual(model.joint_count, expected_model_joints)

    body_q_array = model.body_q.numpy()
    joint_dof_dim_array = model.joint_dof_dim.numpy()
    body_positions = [body_q_array[i, 0:3].tolist() for i in range(body_q_array.shape[0])]
    body_quaternions = [body_q_array[i, 3:7].tolist() for i in range(body_q_array.shape[0])]

    total_dofs = 0
    for j in range(model.joint_count):
        lin = int(joint_dof_dim_array[j][0])
        ang = int(joint_dof_dim_array[j][1])
        total_dofs += lin + ang
        jt = int(model_joint_type[j])

        if jt == JointType.REVOLUTE:
            test.assertEqual((lin, ang), (0, 1), f"{model.joint_label[j]} DOF dim mismatch")
        elif jt == JointType.FIXED:
            test.assertEqual((lin, ang), (0, 0), f"{model.joint_label[j]} DOF dim mismatch")
        elif jt == JointType.FREE:
            test.assertGreater(lin + ang, 0, f"{model.joint_label[j]} expected nonzero DOFs for free joint")
        elif jt == JointType.PRISMATIC:
            test.assertEqual((lin, ang), (1, 0), f"{model.joint_label[j]} DOF dim mismatch")
        elif jt == JointType.BALL:
            test.assertEqual((lin, ang), (0, 3), f"{model.joint_label[j]} DOF dim mismatch")

    test.assertEqual(int(total_dofs), int(model.joint_axis.numpy().shape[0]))
    joint_enabled = model.joint_enabled.numpy()
    test.assertTrue(all(joint_enabled))

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

    for j, key in enumerate(model.joint_label):
        prim = stage.GetPrimAtPath(key)
        if not prim:
            continue

        dof_index = 0 if j <= 0 else sum(int(joint_dof_dim_array[i][0] + joint_dof_dim_array[i][1]) for i in range(j))

        p_rel = prim.GetRelationship("physics:body0")
        c_rel = prim.GetRelationship("physics:body1")
        p_targets = p_rel.GetTargets() if p_rel and p_rel.HasAuthoredTargets() else []
        c_targets = c_rel.GetTargets() if c_rel and c_rel.HasAuthoredTargets() else []

        if len(p_targets) == 1 and len(c_targets) == 1:
            p_path = str(p_targets[0])
            c_path = str(c_targets[0])
            if p_path in body_key_to_idx and c_path in body_key_to_idx:
                test.assertEqual(int(model.joint_parent.numpy()[j]), body_key_to_idx[p_path])
                test.assertEqual(int(model.joint_child.numpy()[j]), body_key_to_idx[c_path])

        if prim.IsA(UsdPhysics.RevoluteJoint) or prim.IsA(UsdPhysics.PrismaticJoint):
            axis_attr = prim.GetAttribute("physics:axis")
            axis_tok = axis_attr.Get() if axis_attr and axis_attr.HasAuthoredValue() else None
            if axis_tok:
                expected_axis = axis_vectors[str(axis_tok)]
                actual_axis = model.joint_axis.numpy()[dof_index].tolist()

                test.assertTrue(
                    all(abs(actual_axis[i] - expected_axis[i]) < 1e-6 for i in range(3))
                    or all(abs(actual_axis[i] - (-expected_axis[i])) < 1e-6 for i in range(3))
                )

            lower_attr = prim.GetAttribute("physics:lowerLimit")
            upper_attr = prim.GetAttribute("physics:upperLimit")
            lower = lower_attr.Get() if lower_attr and lower_attr.HasAuthoredValue() else None
            upper = upper_attr.Get() if upper_attr and upper_attr.HasAuthoredValue() else None

            if prim.IsA(UsdPhysics.RevoluteJoint):
                if lower is not None:
                    test.assertAlmostEqual(
                        float(model.joint_limit_lower.numpy()[dof_index]), math.radians(lower), places=5
                    )
                if upper is not None:
                    test.assertAlmostEqual(
                        float(model.joint_limit_upper.numpy()[dof_index]), math.radians(upper), places=5
                    )
            else:
                if lower is not None:
                    test.assertAlmostEqual(float(model.joint_limit_lower.numpy()[dof_index]), float(lower), places=5)
                if upper is not None:
                    test.assertAlmostEqual(float(model.joint_limit_upper.numpy()[dof_index]), float(upper), places=5)

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
                test.assertAlmostEqual(
                    float(model.joint_target_ke.numpy()[dof_index]), ke * math.degrees(drive_gain_scale), places=2
                )

        if kd_attr:
            kd_val = kd_attr.Get() if kd_attr.HasAuthoredValue() else None
            if kd_val is not None:
                kd = float(kd_val)
                test.assertAlmostEqual(
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

            test.assertTrue(
                all(abs(lhs_p[i] - rhs_p[i]) < 1e-6 for i in range(3)),
                f"Joint {j} ({model.joint_label[j]}) position mismatch: expected={rhs_p}, Newton={lhs_p}",
            )

            q_diff = lhs_q * wp.quat_inverse(rhs_q)
            angle_diff = 2.0 * math.acos(min(1.0, abs(q_diff[3])))
            test.assertLessEqual(
                angle_diff,
                3e-3,
                f"Joint {j} ({model.joint_label[j]}) rotation mismatch: expected={rhs_q}, Newton={lhs_q}, angle_diff={math.degrees(angle_diff)}°",
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
                test.assertEqual(
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
        test.assertEqual(
            collision_enabled_newton,
            collision_enabled_usd,
            f"Shape {sid} collision mismatch: USD={collision_enabled_usd}, Newton={collision_enabled_newton}",
        )

        usd_quat = usd.value_to_warp(shape_spec.localRot)
        newton_pos = newton_transform[:3]
        newton_quat = wp.quat(*newton_transform[3:7])

        for i, (n_pos, u_pos) in enumerate(zip(newton_pos, shape_spec.localPos, strict=False)):
            test.assertAlmostEqual(
                n_pos, u_pos, places=5, msg=f"Shape {sid} position[{i}]: USD={u_pos}, Newton={n_pos}"
            )

        if newton_type in {newton.GeoType.CAPSULE, newton.GeoType.CYLINDER, newton.GeoType.CONE}:
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
            test.fail(
                f"Shape {sid} rotation mismatch: expected={expected_quat}, Newton={newton_quat}, angle_diff={math.degrees(angle_diff)}°"
            )

        if newton_type == newton.GeoType.CAPSULE:
            test.assertAlmostEqual(newton_scale[0], shape_spec.radius, places=5)
            test.assertAlmostEqual(newton_scale[1], shape_spec.halfHeight, places=5)
        elif newton_type == newton.GeoType.BOX:
            for i, (n_scale, u_extent) in enumerate(zip(newton_scale, shape_spec.halfExtents, strict=False)):
                test.assertAlmostEqual(
                    n_scale, u_extent, places=5, msg=f"Box {sid} extent[{i}]: USD={u_extent}, Newton={n_scale}"
                )
        elif newton_type == newton.GeoType.SPHERE:
            test.assertAlmostEqual(newton_scale[0], shape_spec.radius, places=5)
        elif newton_type == newton.GeoType.CYLINDER:
            test.assertAlmostEqual(newton_scale[0], shape_spec.radius, places=5)
            test.assertAlmostEqual(newton_scale[1], shape_spec.halfHeight, places=5)


class TestImportSampleAssetsBasic(unittest.TestCase):
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
        verify_usdphysics_parser(self, asset_path, model, compare_min_max_coords=True, floating=True)

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
        verify_usdphysics_parser(self, stage_path, model, True, floating=True)

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
        verify_usdphysics_parser(self, asset_path, model, compare_min_max_coords=True, floating=False)

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
        verify_usdphysics_parser(self, asset_path, model, compare_min_max_coords=False, floating=True)

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
        verify_usdphysics_parser(self, asset_path, model, compare_min_max_coords=True, floating=True)

    @unittest.skipUnless(USD_AVAILABLE, "Requires usd-core")
    def test_granular_loading_flags(self):
        """Test the granular control over sites and visual shapes loading."""
        from pxr import Usd

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
        from pxr import Usd

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
    def test_granular_loading_with_newton_sites(self):
        """Verify that prims with NewtonSiteAPI are recognized as sites, in parity with MjcSiteAPI."""
        from pxr import Usd

        # Same shape mix as test_granular_loading_with_sites, but the two Site* prims
        # carry NewtonSiteAPI (from newton-usd-schemas) instead of MjcSiteAPI.
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
        prepend apiSchemas = ["NewtonSiteAPI"]
    )
    {
        double radius = 0.1
        double3 xformOp:translate = (0, 1, 0)
        uniform token[] xformOpOrder = ["xformOp:translate"]
    }

    def Cube "Site2" (
        prepend apiSchemas = ["NewtonSiteAPI"]
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

        # load_sites=True, load_visual_shapes=False -> collision + sites only
        builder_sites = newton.ModelBuilder()
        builder_sites.add_usd(stage, load_sites=True, load_visual_shapes=False)
        site_flag = int(newton.ShapeFlags.SITE)
        sites_in_result = sum(1 for i in range(builder_sites.shape_count) if builder_sites.shape_flags[i] & site_flag)
        self.assertEqual(sites_in_result, 2, "NewtonSiteAPI prims should be loaded as sites")
        self.assertEqual(
            builder_sites.shape_count,
            3,
            "Should load 1 collision + 2 NewtonSiteAPI sites with load_visual_shapes=False",
        )

        # load_sites=False -> NewtonSiteAPI prims must be skipped entirely (not loaded as plain visual shapes)
        builder_no_sites = newton.ModelBuilder()
        builder_no_sites.add_usd(stage, load_sites=False)
        sites_in_no_sites = sum(
            1 for i in range(builder_no_sites.shape_count) if builder_no_sites.shape_flags[i] & site_flag
        )
        self.assertEqual(sites_in_no_sites, 0, "load_sites=False should skip NewtonSiteAPI prims")
        self.assertEqual(
            builder_no_sites.shape_count,
            2,
            "load_sites=False should leave 1 collision + 1 visual shape, with NewtonSiteAPI prims excluded entirely",
        )

    @unittest.skipUnless(USD_AVAILABLE, "Requires usd-core")
    def test_newton_mass_api_parsing(self):
        """Exhaustive test of NewtonMassAPI mass/inertia combinations.

        Axes tested:
          Mass source:    explicit physics:mass  vs  density-derived
          Inertia source: newton:inertia  vs  physics:diagonalInertia  vs  density-derived
          Shape mode:     solid  vs  shell+thickness  vs  shell+margin-fallback
        """
        from pxr import Usd

        R = 0.5
        density = 1000.0
        shell_t = 0.05
        margin_t = 0.03
        authored_mass = 10.0

        solid_mass = 4.0 / 3.0 * np.pi * R**3 * density
        solid_I = 2.0 / 5.0 * solid_mass * R**2

        def _shell_mass(t):
            return 4.0 / 3.0 * np.pi * (R**3 - (R - t) ** 3) * density

        def _shell_I(t):
            m_outer = solid_mass
            m_inner = 4.0 / 3.0 * np.pi * (R - t) ** 3 * density
            return 2.0 / 5.0 * m_outer * R**2 - 2.0 / 5.0 * m_inner * (R - t) ** 2

        def _scaled_I(shape_I, shape_mass, target_mass):
            return shape_I * target_mass / shape_mass

        usd_content = """#usda 1.0
(
    upAxis = "Z"
)

def PhysicsScene "physicsScene"
{
}

# 1) Shell + thickness + authored mass, no inertia → shell-derived inertia scaled to mass
def Xform "ShellThicknessMass" (
    prepend apiSchemas = ["PhysicsRigidBodyAPI", "PhysicsMassAPI"]
)
{
    double3 xformOp:translate = (0, 0, 1)
    uniform token[] xformOpOrder = ["xformOp:translate"]

    float physics:mass = 10.0

    def Sphere "Collider" (
        prepend apiSchemas = ["PhysicsCollisionAPI", "NewtonMassAPI"]
    )
    {
        double radius = 0.5
        uniform token newton:massModel = "shell"
        float newton:shellThickness = 0.05
    }
}

# 2) Shell + margin fallback + authored mass, no inertia → margin used as thickness
def Xform "ShellMarginMass" (
    prepend apiSchemas = ["PhysicsRigidBodyAPI", "PhysicsMassAPI"]
)
{
    double3 xformOp:translate = (2, 0, 1)
    uniform token[] xformOpOrder = ["xformOp:translate"]

    float physics:mass = 10.0

    def Sphere "Collider" (
        prepend apiSchemas = ["PhysicsCollisionAPI", "NewtonCollisionAPI", "NewtonMassAPI"]
    )
    {
        double radius = 0.5
        uniform token newton:massModel = "shell"
        float newton:contactMargin = 0.03
    }
}

# 3) Solid + authored mass, no inertia → solid inertia scaled to mass
def Xform "SolidMass" (
    prepend apiSchemas = ["PhysicsRigidBodyAPI", "PhysicsMassAPI"]
)
{
    double3 xformOp:translate = (4, 0, 1)
    uniform token[] xformOpOrder = ["xformOp:translate"]

    float physics:mass = 10.0

    def Sphere "Collider" (
        prepend apiSchemas = ["PhysicsCollisionAPI"]
    )
    {
        double radius = 0.5
    }
}

# 4) Explicit mass + newton:inertia tensor + shell collider
def Xform "ExplicitTensor" (
    prepend apiSchemas = ["PhysicsRigidBodyAPI", "NewtonMassAPI"]
)
{
    double3 xformOp:translate = (6, 0, 1)
    uniform token[] xformOpOrder = ["xformOp:translate"]

    float physics:mass = 5.0
    double[] newton:inertia = [1.0, 2.0, 3.0, 0.1, 0.2, 0.3]
    float3 physics:diagonalInertia = (9.0, 9.0, 9.0)

    def Sphere "Collider" (
        prepend apiSchemas = ["PhysicsCollisionAPI", "NewtonMassAPI"]
    )
    {
        double radius = 0.5
        uniform token newton:massModel = "shell"
        float newton:shellThickness = 0.01
    }
}

# 5) Explicit mass + diagonalInertia (no newton:inertia)
def Xform "ExplicitDiag" (
    prepend apiSchemas = ["PhysicsRigidBodyAPI", "PhysicsMassAPI"]
)
{
    double3 xformOp:translate = (8, 0, 1)
    uniform token[] xformOpOrder = ["xformOp:translate"]

    float physics:mass = 3.0
    float3 physics:diagonalInertia = (0.5, 1.0, 1.5)

    def Sphere "Collider" (
        prepend apiSchemas = ["PhysicsCollisionAPI"]
    )
    {
        double radius = 0.5
    }
}

# 6) Solid, no authored mass or inertia (all density-derived via mass computer)
def Xform "SolidDensity" (
    prepend apiSchemas = ["PhysicsRigidBodyAPI", "PhysicsMassAPI"]
)
{
    double3 xformOp:translate = (10, 0, 1)
    uniform token[] xformOpOrder = ["xformOp:translate"]

    def Sphere "Collider" (
        prepend apiSchemas = ["PhysicsCollisionAPI"]
    )
    {
        double radius = 0.5
    }
}

# 7) Shell, no authored mass or inertia (all density-derived via mass computer)
def Xform "ShellDensity" (
    prepend apiSchemas = ["PhysicsRigidBodyAPI", "PhysicsMassAPI"]
)
{
    double3 xformOp:translate = (12, 0, 1)
    uniform token[] xformOpOrder = ["xformOp:translate"]

    def Sphere "Collider" (
        prepend apiSchemas = ["PhysicsCollisionAPI", "NewtonMassAPI"]
    )
    {
        double radius = 0.5
        uniform token newton:massModel = "shell"
        float newton:shellThickness = 0.05
    }
}

# 8) Shell with negative thickness → warning, falls back to margin
def Xform "NegativeThickness" (
    prepend apiSchemas = ["PhysicsRigidBodyAPI", "PhysicsMassAPI"]
)
{
    double3 xformOp:translate = (14, 0, 1)
    uniform token[] xformOpOrder = ["xformOp:translate"]

    float physics:mass = 10.0

    def Sphere "Collider" (
        prepend apiSchemas = ["PhysicsCollisionAPI", "NewtonMassAPI", "NewtonCollisionAPI"]
    )
    {
        double radius = 0.5
        uniform token newton:massModel = "shell"
        float newton:shellThickness = -0.5
        float newton:contactMargin = 0.03
    }
}

# 9) Singular PSD inertia tensor (valid but non-invertible)
def Xform "SingularTensor" (
    prepend apiSchemas = ["PhysicsRigidBodyAPI", "NewtonMassAPI"]
)
{
    double3 xformOp:translate = (16, 0, 1)
    uniform token[] xformOpOrder = ["xformOp:translate"]

    float physics:mass = 2.0
    double[] newton:inertia = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    def Sphere "Collider" (
        prepend apiSchemas = ["PhysicsCollisionAPI"]
    )
    {
        double radius = 0.5
    }
}

# 10) newton:inertia without physics:diagonalInertia
def Xform "TensorOnly" (
    prepend apiSchemas = ["PhysicsRigidBodyAPI", "NewtonMassAPI", "PhysicsMassAPI"]
)
{
    double3 xformOp:translate = (18, 0, 1)
    uniform token[] xformOpOrder = ["xformOp:translate"]

    float physics:mass = 4.0
    double[] newton:inertia = [1.0, 2.0, 3.0, 0.1, 0.2, 0.3]

    def Sphere "Collider" (
        prepend apiSchemas = ["PhysicsCollisionAPI"]
    )
    {
        double radius = 0.5
    }
}
"""
        stage = Usd.Stage.CreateInMemory()
        stage.GetRootLayer().ImportFromString(usd_content)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            builder = newton.ModelBuilder()
            builder.add_usd(stage)

        self.assertEqual(builder.body_count, 10)
        self.assertEqual(builder.shape_count, 10)

        # --- 1) Shell + thickness + authored mass: inertia from shell geometry, scaled ---
        body_id = builder.body_label.index("/ShellThicknessMass")
        shape_idx = builder.shape_label.index("/ShellThicknessMass/Collider")
        self.assertFalse(builder.shape_is_solid[shape_idx])
        self.assertAlmostEqual(builder.body_mass[body_id], authored_mass, places=5)
        inertia = np.array(builder.body_inertia[body_id]).reshape(3, 3)
        expected_shell_I = _scaled_I(_shell_I(shell_t), _shell_mass(shell_t), authored_mass)
        np.testing.assert_allclose(np.diag(inertia), [expected_shell_I] * 3, rtol=1e-4)

        # --- 2) Shell + margin fallback: different thickness → different inertia ---
        body_id = builder.body_label.index("/ShellMarginMass")
        shape_idx = builder.shape_label.index("/ShellMarginMass/Collider")
        self.assertFalse(builder.shape_is_solid[shape_idx])
        self.assertAlmostEqual(builder.body_mass[body_id], authored_mass, places=5)
        inertia2 = np.array(builder.body_inertia[body_id]).reshape(3, 3)
        expected_margin_I = _scaled_I(_shell_I(margin_t), _shell_mass(margin_t), authored_mass)
        np.testing.assert_allclose(np.diag(inertia2), [expected_margin_I] * 3, rtol=1e-4)
        # Thinner shell → higher I/m ratio → different inertia than body 1
        self.assertFalse(
            np.allclose(np.diag(inertia), np.diag(inertia2), atol=1e-3),
            "Shell thickness vs margin fallback should produce different inertia",
        )

        # --- 3) Solid + authored mass: solid inertia, scaled ---
        body_id = builder.body_label.index("/SolidMass")
        shape_idx = builder.shape_label.index("/SolidMass/Collider")
        self.assertTrue(builder.shape_is_solid[shape_idx])
        self.assertAlmostEqual(builder.body_mass[body_id], authored_mass, places=5)
        inertia3 = np.array(builder.body_inertia[body_id]).reshape(3, 3)
        expected_solid_I = _scaled_I(solid_I, solid_mass, authored_mass)
        np.testing.assert_allclose(np.diag(inertia3), [expected_solid_I] * 3, rtol=1e-4)
        # Shell inertia/mass ratio > solid inertia/mass ratio at same authored mass
        self.assertGreater(np.diag(inertia)[0], np.diag(inertia3)[0])

        # --- 4) Explicit mass + newton:inertia tensor + shell collider ---
        body_id = builder.body_label.index("/ExplicitTensor")
        shape_idx = builder.shape_label.index("/ExplicitTensor/Collider")
        self.assertFalse(builder.shape_is_solid[shape_idx])
        self.assertAlmostEqual(builder.body_mass[body_id], 5.0, places=5)
        inertia = np.array(builder.body_inertia[body_id]).reshape(3, 3)
        expected = np.array([[1.0, 0.1, 0.2], [0.1, 2.0, 0.3], [0.2, 0.3, 3.0]])
        np.testing.assert_allclose(inertia, expected, atol=1e-5)

        # --- 5) Explicit mass + diagonalInertia ---
        body_id = builder.body_label.index("/ExplicitDiag")
        self.assertAlmostEqual(builder.body_mass[body_id], 3.0, places=5)
        inertia = np.array(builder.body_inertia[body_id]).reshape(3, 3)
        np.testing.assert_allclose(np.diag(inertia), [0.5, 1.0, 1.5], atol=1e-5)
        np.testing.assert_allclose(inertia - np.diag(np.diag(inertia)), np.zeros((3, 3)), atol=1e-7)

        # --- 6) Solid, density-derived mass & inertia (no authored values) ---
        body_id = builder.body_label.index("/SolidDensity")
        shape_idx = builder.shape_label.index("/SolidDensity/Collider")
        self.assertTrue(builder.shape_is_solid[shape_idx])
        self.assertGreater(builder.body_mass[body_id], 0.0)
        inertia = np.array(builder.body_inertia[body_id]).reshape(3, 3)
        self.assertGreater(np.trace(inertia), 0.0)

        # --- 7) Shell, density-derived mass & inertia (no authored values) ---
        body_id = builder.body_label.index("/ShellDensity")
        shape_idx = builder.shape_label.index("/ShellDensity/Collider")
        self.assertFalse(builder.shape_is_solid[shape_idx])
        solid_density_id = builder.body_label.index("/SolidDensity")
        self.assertLess(builder.body_mass[body_id], builder.body_mass[solid_density_id])
        inertia = np.array(builder.body_inertia[body_id]).reshape(3, 3)
        self.assertGreater(np.trace(inertia), 0.0)

        # --- 8) Negative shell thickness: warning, falls back to margin, inertia matches margin path ---
        body_id = builder.body_label.index("/NegativeThickness")
        shape_idx = builder.shape_label.index("/NegativeThickness/Collider")
        self.assertFalse(builder.shape_is_solid[shape_idx])
        self.assertAlmostEqual(builder.body_mass[body_id], 10.0, places=5)
        inertia_neg = np.array(builder.body_inertia[body_id]).reshape(3, 3)
        expected_neg_I = _scaled_I(_shell_I(margin_t), _shell_mass(margin_t), 10.0)
        np.testing.assert_allclose(np.diag(inertia_neg), [expected_neg_I] * 3, rtol=1e-4)
        warning_messages = [str(w.message) for w in caught]
        self.assertTrue(any("negative shell thickness" in m and "NegativeThickness" in m for m in warning_messages))

        # --- 9) Singular PSD tensor: valid but non-invertible, inv_inertia set to zero ---
        body_id = builder.body_label.index("/SingularTensor")
        self.assertAlmostEqual(builder.body_mass[body_id], 2.0, places=5)
        inertia = np.array(builder.body_inertia[body_id]).reshape(3, 3)
        expected = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
        np.testing.assert_allclose(inertia, expected, atol=1e-5)
        inv_inertia = np.array(builder.body_inv_inertia[body_id]).reshape(3, 3)
        np.testing.assert_allclose(inv_inertia, np.zeros((3, 3)), atol=1e-7)

        # --- 10) newton:inertia without physics:diagonalInertia ---
        body_id = builder.body_label.index("/TensorOnly")
        self.assertAlmostEqual(builder.body_mass[body_id], 4.0, places=5)
        inertia = np.array(builder.body_inertia[body_id]).reshape(3, 3)
        expected = np.array([[1.0, 0.1, 0.2], [0.1, 2.0, 0.3], [0.2, 0.3, 3.0]])
        np.testing.assert_allclose(inertia, expected, atol=1e-5)

    @unittest.skipUnless(USD_AVAILABLE, "Requires usd-core")
    def test_newton_inertia_tensor_validation(self):
        """Malformed newton:inertia tensors emit warnings and fall back to shape-derived values."""
        from pxr import Usd

        usd_content = """#usda 1.0
(
    upAxis = "Z"
)

def PhysicsScene "physicsScene"
{
}

def Xform "NonFinite" (
    prepend apiSchemas = ["PhysicsRigidBodyAPI", "NewtonMassAPI"]
)
{
    double3 xformOp:translate = (0, 0, 1)
    uniform token[] xformOpOrder = ["xformOp:translate"]

    double[] newton:inertia = [1.0, 2.0, inf, 0.0, 0.0, 0.0]

    def Sphere "Collider" (
        prepend apiSchemas = ["PhysicsCollisionAPI"]
    )
    {
        double radius = 0.5
    }
}

def Xform "NegativeDiag" (
    prepend apiSchemas = ["PhysicsRigidBodyAPI", "NewtonMassAPI"]
)
{
    double3 xformOp:translate = (2, 0, 1)
    uniform token[] xformOpOrder = ["xformOp:translate"]

    double[] newton:inertia = [-1.0, 2.0, 3.0, 0.0, 0.0, 0.0]

    def Sphere "Collider" (
        prepend apiSchemas = ["PhysicsCollisionAPI"]
    )
    {
        double radius = 0.5
    }
}

def Xform "WrongLength" (
    prepend apiSchemas = ["PhysicsRigidBodyAPI", "NewtonMassAPI"]
)
{
    double3 xformOp:translate = (4, 0, 1)
    uniform token[] xformOpOrder = ["xformOp:translate"]

    double[] newton:inertia = [1.0, 2.0, 3.0]

    def Sphere "Collider" (
        prepend apiSchemas = ["PhysicsCollisionAPI"]
    )
    {
        double radius = 0.5
    }
}

def Xform "NotPSD" (
    prepend apiSchemas = ["PhysicsRigidBodyAPI", "NewtonMassAPI"]
)
{
    double3 xformOp:translate = (6, 0, 1)
    uniform token[] xformOpOrder = ["xformOp:translate"]

    double[] newton:inertia = [1.0, 1.0, 1.0, 5.0, 5.0, 5.0]

    def Sphere "Collider" (
        prepend apiSchemas = ["PhysicsCollisionAPI"]
    )
    {
        double radius = 0.5
    }
}
"""
        stage = Usd.Stage.CreateInMemory()
        stage.GetRootLayer().ImportFromString(usd_content)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            builder = newton.ModelBuilder()
            builder.add_usd(stage)

        self.assertEqual(builder.body_count, 4)

        body_id = builder.body_label.index("/NonFinite")
        self.assertGreater(builder.body_mass[body_id], 0.0)

        body_id = builder.body_label.index("/NegativeDiag")
        self.assertGreater(builder.body_mass[body_id], 0.0)

        body_id = builder.body_label.index("/WrongLength")
        self.assertGreater(builder.body_mass[body_id], 0.0)

        body_id = builder.body_label.index("/NotPSD")
        self.assertGreater(builder.body_mass[body_id], 0.0)

        warning_messages = [str(w.message) for w in caught]
        self.assertTrue(any("non-finite" in m and "NonFinite" in m for m in warning_messages))
        self.assertTrue(any("negative diagonal" in m and "NegativeDiag" in m for m in warning_messages))
        self.assertTrue(any("expected 6" in m and "WrongLength" in m for m in warning_messages))
        self.assertTrue(any("not positive semidefinite" in m and "NotPSD" in m for m in warning_messages))

    @unittest.skipUnless(USD_AVAILABLE, "Requires usd-core")
    def test_import_usd_gravcomp(self):
        """Test parsing of gravcomp from USD"""
        from pxr import Sdf, Usd, UsdPhysics

        stage = Usd.Stage.CreateInMemory()
        UsdPhysics.Scene.Define(stage, "/physicsScene")

        # Body 1 with gravcomp
        body1_path = "/Body1"
        prim1 = stage.DefinePrim(body1_path, "Xform")
        UsdPhysics.RigidBodyAPI.Apply(prim1)
        attr1 = prim1.CreateAttribute("mjc:gravcomp", Sdf.ValueTypeNames.Float)
        attr1.Set(0.5)

        # Body 2 without gravcomp
        body2_path = "/Body2"
        prim2 = stage.DefinePrim(body2_path, "Xform")
        UsdPhysics.RigidBodyAPI.Apply(prim2)

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
        from pxr import Usd

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

        joint_names = model.joint_label
        joint_qd_start = model.joint_qd_start.numpy()
        joint_stiffness = model.mujoco.dof_passive_stiffness.numpy()
        joint_damping = model.joint_damping.numpy()
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
        from pxr import Usd

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
    def test_geom_group_parsing_and_conversion(self):
        """Test USD geom groups are imported and converted to MuJoCo."""
        from pxr import Usd

        stage = Usd.Stage.CreateInMemory()
        stage.GetRootLayer().ImportFromString(
            """#usda 1.0
(
    upAxis = "Z"
)

def Xform "Body" (
    prepend apiSchemas = ["PhysicsRigidBodyAPI"]
)
{
    def Sphere "Collision" (
        prepend apiSchemas = ["PhysicsCollisionAPI"]
    )
    {
        double radius = 0.1
        int mjc:group = 3
    }
}
"""
        )

        builder = newton.ModelBuilder()
        SolverMuJoCo.register_custom_attributes(builder)
        builder.add_usd(stage)
        model = builder.finalize(device="cpu")
        solver = SolverMuJoCo(model, iterations=1, disable_contacts=True)

        np.testing.assert_array_equal(model.mujoco.geom_group.numpy(), [3])
        np.testing.assert_array_equal(solver.mj_model.geom_group, [3])
        np.testing.assert_array_equal(solver.mjw_model.geom_group.numpy(), [3])


class TestImportSampleAssetsParsing(unittest.TestCase):
    @unittest.skipUnless(USD_AVAILABLE, "Requires usd-core")
    def test_add_usd_mjc_schemas_without_mujoco(self):
        asset_path = os.path.join(os.path.dirname(__file__), "assets", "mjc_schema_import.usda")
        original_import = builtins.__import__
        optional_runtime_imports = []

        def track_optional_runtime_imports(name, *args, **kwargs):
            if name.partition(".")[0] in {"mujoco", "mujoco_warp"}:
                optional_runtime_imports.append(name)
            return original_import(name, *args, **kwargs)

        for register_mujoco, convert_equalities in (
            (False, False),
            (False, True),
            (True, False),
            (True, True),
        ):
            with self.subTest(register_mujoco=register_mujoco, convert_equalities=convert_equalities):
                builder = newton.ModelBuilder()
                if register_mujoco:
                    SolverMuJoCo.register_custom_attributes(builder)

                optional_runtime_imports.clear()
                with mock.patch.object(builtins, "__import__", side_effect=track_optional_runtime_imports):
                    builder.add_usd(
                        asset_path,
                        convert_mjc_equality_constraints=convert_equalities,
                        schema_resolvers=[usd.SchemaResolverMjc()],
                    )
                self.assertEqual(optional_runtime_imports, [])

                model = builder.finalize()
                self.assertEqual(model.mujoco.equality_constraint_count, 3)
                self.assertEqual(model.constraint_mimic_count, int(convert_equalities))
                self.assertEqual(model.custom_frequency_counts.get("mujoco:actuator", 0), int(register_mujoco))

                if register_mujoco:
                    np.testing.assert_array_equal(model.mujoco.actuator_dyntype.numpy(), [0])
                    np.testing.assert_array_equal(model.mujoco.actuator_gaintype.numpy(), [0])
                    np.testing.assert_array_equal(model.mujoco.actuator_biastype.numpy(), [1])
                    self.assertIn(int(newton.JointTargetMode.POSITION), builder.joint_target_mode)
                    self.assertEqual(
                        set(model.mujoco.solreflimit_mode.numpy().tolist()),
                        {SOLREF_MODE_FORCE_SPACE, SOLREF_MODE_RAW, SOLREF_MODE_MJCF_DEFAULT},
                    )

    @unittest.skipUnless(USD_AVAILABLE, "Requires usd-core")
    def test_jnt_actgravcomp_parsing(self):
        """Test that jnt_actgravcomp attribute is parsed correctly from USD."""
        from pxr import Usd

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
        from pxr import Usd

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
        from pxr import Usd

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
        from pxr import Usd

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
        from pxr import Usd

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
        from pxr import Usd

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

    def Xform "Articulation"
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
        from pxr import Usd

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

        revolute_joint_idx = model.joint_label.index("/Articulation/revolute_joint")
        self.assertAlmostEqual(dof_ref[qd_start[revolute_joint_idx]], 90.0, places=4)

    @unittest.skipUnless(USD_AVAILABLE, "Requires usd-core")
    def test_springref_attribute_parsing(self):
        """Test that 'mjc:springref' attribute is parsed for revolute and prismatic joints."""
        from pxr import Usd

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

        revolute_joint_idx = model.joint_label.index("/Articulation/revolute_joint")
        self.assertAlmostEqual(springref[qd_start[revolute_joint_idx]], 30.0, places=4)

        prismatic_joint_idx = model.joint_label.index("/Articulation/prismatic_joint")
        self.assertAlmostEqual(springref[qd_start[prismatic_joint_idx]], 0.25, places=4)

    @unittest.skipUnless(USD_AVAILABLE, "Requires usd-core")
    def test_material_parsing(self):
        """Test that material attributes are parsed correctly from USD."""
        from pxr import Usd, UsdGeom, UsdPhysics, UsdShade

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

        # Create a free-floating body with a collider (no joints, so no articulation)
        UsdGeom.Xform.Define(stage, "/Articulation")

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
        torsional = model.shape_material_mu_torsional.numpy()[shape_idx]
        self.assertAlmostEqual(torsional, 0.15, places=4)

        # Check rolling friction
        rolling = model.shape_material_mu_rolling.numpy()[shape_idx]
        self.assertAlmostEqual(rolling, 0.08, places=4)

    @unittest.skipUnless(USD_AVAILABLE, "Requires usd-core")
    def test_visual_mesh_material_subsets_create_separate_visual_shapes(self):
        """Test that visual mesh material subsets import as separate colored shapes."""
        from pxr import Sdf, Usd, UsdGeom, UsdPhysics, UsdShade, Vt

        stage = Usd.Stage.CreateInMemory()
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
        UsdGeom.SetStageMetersPerUnit(stage, 1.0)
        UsdPhysics.Scene.Define(stage, "/physicsScene")

        body = UsdGeom.Xform.Define(stage, "/Body")
        UsdPhysics.RigidBodyAPI.Apply(body.GetPrim())

        mesh = UsdGeom.Mesh.Define(stage, "/Body/VisualMesh")
        mesh.CreatePointsAttr().Set(
            [
                (-0.5, -0.5, 0.0),
                (0.5, -0.5, 0.0),
                (0.5, 0.5, 0.0),
                (-0.5, 0.5, 0.0),
            ]
        )
        mesh.CreateFaceVertexCountsAttr().Set([3, 3])
        mesh.CreateFaceVertexIndicesAttr().Set([0, 1, 2, 0, 2, 3])
        st = UsdGeom.PrimvarsAPI(mesh).CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.vertex)
        st.Set([(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)])

        grey_material = UsdShade.Material.Define(stage, "/Materials/Grey")
        grey_shader = UsdShade.Shader.Define(stage, "/Materials/Grey/PreviewSurface")
        grey_shader.CreateIdAttr("UsdPreviewSurface")
        grey_shader.CreateInput("baseColor", Sdf.ValueTypeNames.Color3f).Set((0.5, 0.5, 0.5))
        grey_material.CreateSurfaceOutput().ConnectToSource(grey_shader.ConnectableAPI(), "surface")
        UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(grey_material)

        red_material = UsdShade.Material.Define(stage, "/Materials/Red")
        red_shader = UsdShade.Shader.Define(stage, "/Materials/Red/PreviewSurface")
        red_shader.CreateIdAttr("UsdPreviewSurface")
        red_shader.CreateInput("baseColor", Sdf.ValueTypeNames.Color3f).Set((1.0, 0.0, 0.0))
        red_material.CreateSurfaceOutput().ConnectToSource(red_shader.ConnectableAPI(), "surface")

        blue_material = UsdShade.Material.Define(stage, "/Materials/Blue")
        blue_shader = UsdShade.Shader.Define(stage, "/Materials/Blue/PreviewSurface")
        blue_shader.CreateIdAttr("UsdPreviewSurface")
        blue_texture = UsdShade.Shader.Define(stage, "/Materials/Blue/DiffuseTexture")
        blue_texture.CreateIdAttr("UsdUVTexture")
        blue_texture.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath("blue.png"))
        blue_texture.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)
        blue_shader.CreateInput("baseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(
            blue_texture.ConnectableAPI(), "rgb"
        )
        blue_material.CreateSurfaceOutput().ConnectToSource(blue_shader.ConnectableAPI(), "surface")

        red_subset = UsdGeom.Subset.Define(stage, "/Body/VisualMesh/red")
        red_subset.CreateElementTypeAttr().Set(UsdGeom.Tokens.face)
        red_subset.CreateFamilyNameAttr().Set("materialBind")
        red_subset.CreateIndicesAttr().Set(Vt.IntArray([0]))
        UsdShade.MaterialBindingAPI.Apply(red_subset.GetPrim()).Bind(red_material)

        blue_subset = UsdGeom.Subset.Define(stage, "/Body/VisualMesh/blue")
        blue_subset.CreateElementTypeAttr().Set(UsdGeom.Tokens.face)
        blue_subset.CreateFamilyNameAttr().Set("materialBind")
        blue_subset.CreateIndicesAttr().Set(Vt.IntArray([1]))
        UsdShade.MaterialBindingAPI.Apply(blue_subset.GetPrim()).Bind(blue_material)

        builder = newton.ModelBuilder()
        result = builder.add_usd(stage)

        self.assertIn("/Body/VisualMesh/red", result["path_shape_map"])
        self.assertIn("/Body/VisualMesh/blue", result["path_shape_map"])

        red_shape = result["path_shape_map"]["/Body/VisualMesh/red"]
        blue_shape = result["path_shape_map"]["/Body/VisualMesh/blue"]

        self.assertEqual(builder.shape_count, 2)
        self.assertEqual(builder.shape_label[red_shape], "/Body/VisualMesh/red")
        self.assertEqual(builder.shape_label[blue_shape], "/Body/VisualMesh/blue")

        red_mesh = builder.shape_source[red_shape]
        blue_mesh = builder.shape_source[blue_shape]
        self.assertEqual(len(red_mesh.indices), 3)
        self.assertEqual(len(blue_mesh.indices), 3)
        np.testing.assert_allclose(np.array(red_mesh.color), np.array([1.0, 0.0, 0.0]), atol=1e-6, rtol=1e-6)
        self.assertIsNotNone(blue_mesh.uvs)
        self.assertEqual(blue_mesh.texture, "blue.png")
        np.testing.assert_allclose(np.array(blue_mesh.color), np.array([1.0, 1.0, 1.0]), atol=1e-6, rtol=1e-6)

    @unittest.skipUnless(USD_AVAILABLE, "Requires usd-core")
    def test_uv_length_mismatch_uses_info_logging(self):
        """Dropped-UV/texture diagnostics are render-only and surface via `logger.info`, not `warnings.warn`."""
        import logging as _logging  # noqa: PLC0415
        import warnings as _warnings  # noqa: PLC0415

        from pxr import Sdf, Usd, UsdGeom, UsdPhysics, UsdShade

        stage = Usd.Stage.CreateInMemory()
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
        UsdGeom.SetStageMetersPerUnit(stage, 1.0)
        UsdPhysics.Scene.Define(stage, "/physicsScene")

        body = UsdGeom.Xform.Define(stage, "/Body")
        UsdPhysics.RigidBodyAPI.Apply(body.GetPrim())

        mesh = UsdGeom.Mesh.Define(stage, "/Body/VisualMesh")
        mesh.CreatePointsAttr().Set(
            [
                (-0.5, -0.5, 0.0),
                (0.5, -0.5, 0.0),
                (0.5, 0.5, 0.0),
                (-0.5, 0.5, 0.0),
            ]
        )
        mesh.CreateFaceVertexCountsAttr().Set([3, 3])
        mesh.CreateFaceVertexIndicesAttr().Set([0, 1, 2, 0, 2, 3])
        # Author a single face-varying `st` primvar whose length does not match the mesh's
        # face-corner count, so the importer must drop UVs and (downstream) the bound texture.
        UsdGeom.PrimvarsAPI(mesh).CreatePrimvar(
            "st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.faceVarying
        ).Set([(0.0, 0.0)])

        material = UsdShade.Material.Define(stage, "/Materials/Tex")
        shader = UsdShade.Shader.Define(stage, "/Materials/Tex/PreviewSurface")
        shader.CreateIdAttr("UsdPreviewSurface")
        tex = UsdShade.Shader.Define(stage, "/Materials/Tex/DiffuseTexture")
        tex.CreateIdAttr("UsdUVTexture")
        tex.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath("ignored.png"))
        tex.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)
        shader.CreateInput("baseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(tex.ConnectableAPI(), "rgb")
        material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
        UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(material)

        builder = newton.ModelBuilder()
        with _warnings.catch_warnings(record=True) as caught, self.assertLogs("newton", level=_logging.INFO) as log_ctx:
            _warnings.simplefilter("always")
            builder.add_usd(stage)
        uv_warnings = [
            w for w in caught if "UV primvar length" in str(w.message) or "has a texture but no UVs" in str(w.message)
        ]
        self.assertEqual(uv_warnings, [], f"unexpected UV warnings: {[str(w.message) for w in uv_warnings]}")

        joined = "\n".join(log_ctx.output)
        self.assertIn("UV primvar length", joined)
        self.assertIn("dropping texture because UVs could not be recovered", joined)

    @unittest.skipUnless(USD_AVAILABLE, "Requires usd-core")
    def test_material_density_used_by_mass_properties(self):
        """Test that physics material density contributes to imported body mass/inertia."""
        from pxr import Usd, UsdGeom, UsdPhysics, UsdShade

        stage = Usd.Stage.CreateInMemory()
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
        UsdGeom.SetStageMetersPerUnit(stage, 1.0)
        UsdPhysics.Scene.Define(stage, "/physicsScene")

        body = UsdGeom.Xform.Define(stage, "/World/Body")
        body_prim = body.GetPrim()
        UsdPhysics.RigidBodyAPI.Apply(body_prim)
        # Ensure parse_usd enters the MassAPI override path.
        UsdPhysics.MassAPI.Apply(body_prim)

        collider = UsdGeom.Cube.Define(stage, "/World/Body/Collider")
        collider.CreateSizeAttr().Set(2.0)  # side length = 2.0 -> volume = 8.0
        collider_prim = collider.GetPrim()
        UsdPhysics.CollisionAPI.Apply(collider_prim)

        density = 250.0
        material = UsdShade.Material.Define(stage, "/World/Materials/Dense")
        material_prim = material.GetPrim()
        UsdPhysics.MaterialAPI.Apply(material_prim).CreateDensityAttr().Set(density)
        UsdShade.MaterialBindingAPI.Apply(collider_prim).Bind(material, "physics")

        builder = newton.ModelBuilder()
        result = builder.add_usd(stage)

        body_idx = result["path_body_map"]["/World/Body"]
        expected_mass = density * 8.0
        self.assertAlmostEqual(builder.body_mass[body_idx], expected_mass, places=4)
        body_com = np.array(builder.body_com[body_idx], dtype=np.float32)
        np.testing.assert_allclose(body_com, np.zeros(3, dtype=np.float32), atol=1e-6, rtol=1e-6)

        # For a solid cube with side length a: I = (1/6) * m * a^2 on each axis.
        expected_diag = (1.0 / 6.0) * expected_mass * (2.0**2)
        inertia = np.array(builder.body_inertia[body_idx]).reshape(3, 3)
        np.testing.assert_allclose(np.diag(inertia), np.array([expected_diag, expected_diag, expected_diag]), rtol=1e-4)
        np.testing.assert_allclose(
            inertia - np.diag(np.diag(inertia)),
            np.zeros((3, 3), dtype=np.float32),
            atol=1e-6,
        )

    @unittest.skipUnless(USD_AVAILABLE, "Requires usd-core")
    def test_material_density_mass_properties_with_stage_linear_scale(self):
        """Test mass/inertia parsing when stage metersPerUnit is not 1.0."""
        from pxr import Usd, UsdGeom, UsdPhysics, UsdShade

        stage = Usd.Stage.CreateInMemory()
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
        UsdGeom.SetStageMetersPerUnit(stage, 0.01)
        UsdPhysics.Scene.Define(stage, "/physicsScene")

        body = UsdGeom.Xform.Define(stage, "/World/Body")
        body_prim = body.GetPrim()
        UsdPhysics.RigidBodyAPI.Apply(body_prim)
        UsdPhysics.MassAPI.Apply(body_prim)

        collider = UsdGeom.Cube.Define(stage, "/World/Body/Collider")
        collider.CreateSizeAttr().Set(2.0)  # side length in stage units
        collider_prim = collider.GetPrim()
        UsdPhysics.CollisionAPI.Apply(collider_prim)

        density = 250.0
        material = UsdShade.Material.Define(stage, "/World/Materials/Dense")
        UsdPhysics.MaterialAPI.Apply(material.GetPrim()).CreateDensityAttr().Set(density)
        UsdShade.MaterialBindingAPI.Apply(collider_prim).Bind(material, "physics")

        builder = newton.ModelBuilder()
        with self.assertWarnsRegex(UserWarning, "non-unit linear units are not supported"):
            result = builder.add_usd(stage)

        self.assertAlmostEqual(result["linear_unit"], 0.01, places=7)

        body_idx = result["path_body_map"]["/World/Body"]
        expected_mass = density * 8.0  # 2^3 stage units
        self.assertAlmostEqual(builder.body_mass[body_idx], expected_mass, places=4)

        # For a solid cube: I = (1/6) * m * a^2 on each axis.
        expected_diag = (1.0 / 6.0) * expected_mass * (2.0**2)
        inertia = np.array(builder.body_inertia[body_idx]).reshape(3, 3)
        np.testing.assert_allclose(np.diag(inertia), np.array([expected_diag, expected_diag, expected_diag]), rtol=1e-4)
        np.testing.assert_allclose(
            inertia - np.diag(np.diag(inertia)),
            np.zeros((3, 3), dtype=np.float32),
            atol=1e-6,
        )

    @unittest.skipUnless(USD_AVAILABLE, "Requires usd-core")
    def test_massapi_authored_mass_with_non_unit_mass_unit_warns(self):
        """Test unsupported kilogramsPerUnit warning and unscaled authored mass."""
        from pxr import Gf, Usd, UsdGeom, UsdPhysics

        stage = Usd.Stage.CreateInMemory()
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
        UsdGeom.SetStageMetersPerUnit(stage, 1.0)
        UsdPhysics.SetStageKilogramsPerUnit(stage, 0.001)
        UsdPhysics.Scene.Define(stage, "/physicsScene")

        body = UsdGeom.Xform.Define(stage, "/World/Body")
        body_prim = body.GetPrim()
        UsdPhysics.RigidBodyAPI.Apply(body_prim)
        body_mass_api = UsdPhysics.MassAPI.Apply(body_prim)
        body_mass_api.CreateMassAttr().Set(3.0)
        body_mass_api.CreateDiagonalInertiaAttr().Set(Gf.Vec3f(0.1, 0.2, 0.3))

        collider = UsdGeom.Cube.Define(stage, "/World/Body/Collider")
        collider_prim = collider.GetPrim()
        UsdPhysics.CollisionAPI.Apply(collider_prim)

        builder = newton.ModelBuilder()
        with self.assertWarnsRegex(UserWarning, "non-unit mass units are not supported"):
            result = builder.add_usd(stage)

        self.assertAlmostEqual(result["mass_unit"], 0.001, places=7)
        body_idx = result["path_body_map"]["/World/Body"]
        self.assertAlmostEqual(builder.body_mass[body_idx], 3.0, places=6)
        inertia = np.array(builder.body_inertia[body_idx]).reshape(3, 3)
        np.testing.assert_allclose(np.diag(inertia), np.array([0.1, 0.2, 0.3]), atol=1e-6, rtol=1e-6)

    @unittest.skipUnless(USD_AVAILABLE, "Requires usd-core")
    def test_collider_massapi_density_used_by_mass_properties(self):
        """Test that collider MassAPI density contributes in ComputeMassProperties fallback."""
        from pxr import Usd, UsdGeom, UsdPhysics

        stage = Usd.Stage.CreateInMemory()
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
        UsdGeom.SetStageMetersPerUnit(stage, 1.0)
        UsdPhysics.Scene.Define(stage, "/physicsScene")

        body = UsdGeom.Xform.Define(stage, "/World/Body")
        body_prim = body.GetPrim()
        UsdPhysics.RigidBodyAPI.Apply(body_prim)
        # Partial body MassAPI -> triggers ComputeMassProperties callback path.
        UsdPhysics.MassAPI.Apply(body_prim)

        collider = UsdGeom.Cube.Define(stage, "/World/Body/Collider")
        collider.CreateSizeAttr().Set(2.0)  # side length = 2.0 -> volume = 8.0
        collider_prim = collider.GetPrim()
        UsdPhysics.CollisionAPI.Apply(collider_prim)

        density = 250.0
        UsdPhysics.MassAPI.Apply(collider_prim).CreateDensityAttr().Set(density)

        builder = newton.ModelBuilder()
        result = builder.add_usd(stage)

        body_idx = result["path_body_map"]["/World/Body"]
        expected_mass = density * 8.0
        self.assertAlmostEqual(builder.body_mass[body_idx], expected_mass, places=4)

        expected_diag = (1.0 / 6.0) * expected_mass * (2.0**2)
        inertia = np.array(builder.body_inertia[body_idx]).reshape(3, 3)
        np.testing.assert_allclose(np.diag(inertia), np.array([expected_diag, expected_diag, expected_diag]), rtol=1e-4)
        np.testing.assert_allclose(
            inertia - np.diag(np.diag(inertia)),
            np.zeros((3, 3), dtype=np.float32),
            atol=1e-6,
        )

    @unittest.skipUnless(USD_AVAILABLE, "Requires usd-core")
    def test_material_density_without_massapi_uses_shape_material(self):
        """Test that non-MassAPI bodies use collider material density for mass accumulation."""
        from pxr import Usd, UsdGeom, UsdPhysics, UsdShade

        stage = Usd.Stage.CreateInMemory()
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
        UsdGeom.SetStageMetersPerUnit(stage, 1.0)
        UsdPhysics.Scene.Define(stage, "/physicsScene")

        body = UsdGeom.Xform.Define(stage, "/World/Body")
        body_prim = body.GetPrim()
        UsdPhysics.RigidBodyAPI.Apply(body_prim)
        # Intentionally do NOT apply MassAPI here.

        collider = UsdGeom.Cube.Define(stage, "/World/Body/Collider")
        collider.CreateSizeAttr().Set(2.0)  # side length = 2.0 -> volume = 8.0
        collider_prim = collider.GetPrim()
        UsdPhysics.CollisionAPI.Apply(collider_prim)

        density = 250.0
        material = UsdShade.Material.Define(stage, "/World/Materials/Dense")
        UsdPhysics.MaterialAPI.Apply(material.GetPrim()).CreateDensityAttr().Set(density)
        UsdShade.MaterialBindingAPI.Apply(collider_prim).Bind(material, "physics")

        builder = newton.ModelBuilder()
        result = builder.add_usd(stage)

        body_idx = result["path_body_map"]["/World/Body"]
        expected_mass = density * 8.0
        self.assertAlmostEqual(builder.body_mass[body_idx], expected_mass, places=4)

        # For a solid cube with side length a: I = (1/6) * m * a^2 on each axis.
        expected_diag = (1.0 / 6.0) * expected_mass * (2.0**2)
        inertia = np.array(builder.body_inertia[body_idx]).reshape(3, 3)
        np.testing.assert_allclose(np.diag(inertia), np.array([expected_diag, expected_diag, expected_diag]), rtol=1e-4)
        np.testing.assert_allclose(
            inertia - np.diag(np.diag(inertia)),
            np.zeros((3, 3), dtype=np.float32),
            atol=1e-6,
        )

    @unittest.skipUnless(USD_AVAILABLE, "Requires usd-core")
    def test_material_without_density_uses_default_shape_density(self):
        """Test that bound materials without authored density fall back to default shape density."""
        from pxr import Usd, UsdGeom, UsdPhysics, UsdShade

        stage = Usd.Stage.CreateInMemory()
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
        UsdGeom.SetStageMetersPerUnit(stage, 1.0)
        UsdPhysics.Scene.Define(stage, "/physicsScene")

        body = UsdGeom.Xform.Define(stage, "/World/Body")
        body_prim = body.GetPrim()
        UsdPhysics.RigidBodyAPI.Apply(body_prim)
        # Intentionally do NOT apply MassAPI here.

        collider = UsdGeom.Cube.Define(stage, "/World/Body/Collider")
        collider.CreateSizeAttr().Set(2.0)  # side length = 2.0 -> volume = 8.0
        collider_prim = collider.GetPrim()
        UsdPhysics.CollisionAPI.Apply(collider_prim)

        # Bind a physics material but do not author density.
        material = UsdShade.Material.Define(stage, "/World/Materials/NoDensity")
        UsdPhysics.MaterialAPI.Apply(material.GetPrim())
        UsdShade.MaterialBindingAPI.Apply(collider_prim).Bind(material, "physics")

        builder = newton.ModelBuilder()
        builder.default_shape_cfg.density = 123.0
        result = builder.add_usd(stage)

        body_idx = result["path_body_map"]["/World/Body"]
        expected_mass = builder.default_shape_cfg.density * 8.0
        self.assertAlmostEqual(builder.body_mass[body_idx], expected_mass, places=4)

        # For a solid cube with side length a: I = (1/6) * m * a^2 on each axis.
        expected_diag = (1.0 / 6.0) * expected_mass * (2.0**2)
        inertia = np.array(builder.body_inertia[body_idx]).reshape(3, 3)
        np.testing.assert_allclose(np.diag(inertia), np.array([expected_diag, expected_diag, expected_diag]), rtol=1e-4)
        np.testing.assert_allclose(
            inertia - np.diag(np.diag(inertia)),
            np.zeros((3, 3), dtype=np.float32),
            atol=1e-6,
        )

    @unittest.skipUnless(USD_AVAILABLE, "Requires usd-core")
    def test_massapi_authored_com_matches_scaled_collider_frame(self):
        """Authored body COM uses the same scaled frame as collider offsets."""
        from pxr import Gf, Usd, UsdGeom, UsdPhysics

        stage = Usd.Stage.CreateInMemory()
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
        UsdGeom.SetStageMetersPerUnit(stage, 1.0)
        UsdPhysics.Scene.Define(stage, "/physicsScene")

        cases = (
            ("Nonuniform", (2.0, 3.0, 4.0), 45.0, (0.3, 0.0, 0.0), True),
            ("Negative", (-2.0, 3.0, 4.0), 30.0, (0.3, 0.2, 0.1), False),
        )
        mass_cases = (("Complete", True), ("Partial", False))

        for case_name, scale, angle, com, include_partial in cases:
            parent = UsdGeom.Xform.Define(stage, f"/World/{case_name}")
            parent.AddScaleOp().Set(Gf.Vec3d(*scale))
            for mass_name, author_all_mass_properties in mass_cases if include_partial else mass_cases[:1]:
                body_path = f"/World/{case_name}/{mass_name}"
                body = UsdGeom.Xform.Define(stage, body_path)
                body.AddRotateZOp().Set(angle)
                body_prim = body.GetPrim()
                UsdPhysics.RigidBodyAPI.Apply(body_prim)
                mass_api = UsdPhysics.MassAPI.Apply(body_prim)
                mass_api.CreateCenterOfMassAttr().Set(Gf.Vec3f(*com))
                if author_all_mass_properties:
                    mass_api.CreateMassAttr().Set(1.0)
                    mass_api.CreateDiagonalInertiaAttr().Set(Gf.Vec3f(0.01))

                collider = UsdGeom.Cube.Define(stage, f"{body_path}/Collider")
                collider.AddTranslateOp().Set(Gf.Vec3d(*com))
                UsdPhysics.CollisionAPI.Apply(collider.GetPrim())

        builder = newton.ModelBuilder()
        result = builder.add_usd(stage)

        for case_name, _scale, _angle, _com, include_partial in cases:
            for mass_name, _author_all_mass_properties in mass_cases if include_partial else mass_cases[:1]:
                with self.subTest(case=case_name, mass=mass_name):
                    body_path = f"/World/{case_name}/{mass_name}"
                    body_idx = result["path_body_map"][body_path]
                    shape_idx = result["path_shape_map"][f"{body_path}/Collider"]
                    shape_pos = builder.shape_transform[shape_idx].p
                    np.testing.assert_allclose(builder.body_com[body_idx], shape_pos, atol=1e-6, rtol=1e-6)

    @unittest.skipUnless(USD_AVAILABLE, "Requires usd-core")
    def test_massapi_authored_com_survives_failed_mass_computation(self):
        """Authored body COM keeps descriptor scale when mass properties cannot be computed."""
        from pxr import Gf, Usd, UsdGeom, UsdPhysics

        stage = Usd.Stage.CreateInMemory()
        parent = UsdGeom.Xform.Define(stage, "/World/Scaled")
        parent.AddScaleOp().Set(Gf.Vec3d(2.0))

        body = UsdGeom.Xform.Define(stage, "/World/Scaled/Body")
        body_prim = body.GetPrim()
        UsdPhysics.RigidBodyAPI.Apply(body_prim)
        mass_api = UsdPhysics.MassAPI.Apply(body_prim)
        mass_api.CreateCenterOfMassAttr().Set(Gf.Vec3f(0.3, 0.0, 0.0))

        builder = newton.ModelBuilder()
        with self.assertWarnsRegex(UserWarning, "zero mass and zero inertia"):
            result = builder.add_usd(stage)

        body_idx = result["path_body_map"]["/World/Scaled/Body"]
        np.testing.assert_allclose(builder.body_com[body_idx], [0.6, 0.0, 0.0], atol=1e-6, rtol=1e-6)

    @unittest.skipUnless(USD_AVAILABLE, "Requires usd-core")
    def test_massapi_authored_mass_and_inertia_short_circuits_compute(self):
        """If body has authored mass+diagonalInertia, use them directly without compute fallback."""
        from pxr import Gf, Usd, UsdGeom, UsdPhysics

        stage = Usd.Stage.CreateInMemory()
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
        UsdGeom.SetStageMetersPerUnit(stage, 1.0)
        UsdPhysics.Scene.Define(stage, "/physicsScene")

        body = UsdGeom.Xform.Define(stage, "/World/Body")
        body_prim = body.GetPrim()
        UsdPhysics.RigidBodyAPI.Apply(body_prim)
        body_mass_api = UsdPhysics.MassAPI.Apply(body_prim)
        body_mass_api.CreateMassAttr().Set(3.0)
        body_mass_api.CreateDiagonalInertiaAttr().Set(Gf.Vec3f(0.1, 0.2, 0.3))

        # Add collider with conflicting authored mass props that would affect computed inertia.
        collider = UsdGeom.Cube.Define(stage, "/World/Body/Collider")
        collider.CreateSizeAttr().Set(2.0)
        collider_prim = collider.GetPrim()
        UsdPhysics.CollisionAPI.Apply(collider_prim)
        collider_mass_api = UsdPhysics.MassAPI.Apply(collider_prim)
        collider_mass_api.CreateMassAttr().Set(20.0)
        collider_mass_api.CreateDiagonalInertiaAttr().Set(Gf.Vec3f(13.333334, 13.333334, 13.333334))

        builder = newton.ModelBuilder()
        result = builder.add_usd(stage)
        body_idx = result["path_body_map"]["/World/Body"]

        self.assertAlmostEqual(builder.body_mass[body_idx], 3.0, places=6)
        inertia = np.array(builder.body_inertia[body_idx]).reshape(3, 3)
        np.testing.assert_allclose(np.diag(inertia), np.array([0.1, 0.2, 0.3]), atol=1e-6, rtol=1e-6)
        np.testing.assert_allclose(inertia - np.diag(np.diag(inertia)), np.zeros((3, 3), dtype=np.float32), atol=1e-7)

    @unittest.skipUnless(USD_AVAILABLE, "Requires usd-core")
    def test_massapi_partial_body_falls_back_to_compute(self):
        """If body MassAPI is partial (missing inertia), compute fallback should provide inertia."""
        from pxr import Gf, Usd, UsdGeom, UsdPhysics

        stage = Usd.Stage.CreateInMemory()
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
        UsdGeom.SetStageMetersPerUnit(stage, 1.0)
        UsdPhysics.Scene.Define(stage, "/physicsScene")

        body = UsdGeom.Xform.Define(stage, "/World/Body")
        body_prim = body.GetPrim()
        UsdPhysics.RigidBodyAPI.Apply(body_prim)
        body_mass_api = UsdPhysics.MassAPI.Apply(body_prim)
        body_mass_api.CreateMassAttr().Set(1.0)  # inertia intentionally omitted

        collider = UsdGeom.Cube.Define(stage, "/World/Body/Collider")
        collider.CreateSizeAttr().Set(2.0)
        collider_prim = collider.GetPrim()
        UsdPhysics.CollisionAPI.Apply(collider_prim)
        collider_mass_api = UsdPhysics.MassAPI.Apply(collider_prim)
        collider_mass_api.CreateMassAttr().Set(2.0)
        # For side length 2 and mass 2: I_diag = (1/6) * m * a^2 = 4/3.
        collider_mass_api.CreateDiagonalInertiaAttr().Set(Gf.Vec3f(1.3333334, 1.3333334, 1.3333334))

        builder = newton.ModelBuilder()
        result = builder.add_usd(stage)
        body_idx = result["path_body_map"]["/World/Body"]

        # Body mass is authored and should still be honored.
        self.assertAlmostEqual(builder.body_mass[body_idx], 1.0, places=6)
        # Fallback computation should use collider information to derive inertia.
        expected_diag = (1.0 / 6.0) * 1.0 * (2.0**2)  # => 2/3
        inertia = np.array(builder.body_inertia[body_idx]).reshape(3, 3)
        np.testing.assert_allclose(
            np.diag(inertia), np.array([expected_diag, expected_diag, expected_diag]), atol=1e-5, rtol=1e-5
        )

    @unittest.skipUnless(USD_AVAILABLE, "Requires usd-core")
    def test_massapi_authored_mass_without_inertia_scales_to_uniform_density(self):
        """Authored mass without inertia should produce inertia consistent with a uniform-density body.

        Two identical 0.1 [m] cube bodies that should both end up with 8 [kg]
        mass and inertia I_diag = (1/6) * m * s^2 [kg*m^2]:
          A - density 8000 [kg/m^3] on the collider shape
          B - mass 8 [kg] on the body only, inertia via scaling
        """
        from pxr import Usd, UsdGeom, UsdPhysics

        stage = Usd.Stage.CreateInMemory()
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
        UsdGeom.SetStageMetersPerUnit(stage, 1.0)
        UsdPhysics.Scene.Define(stage, "/physicsScene")

        density = 8000.0
        size = 0.1
        mass = density * size**3
        expected_i = (1.0 / 6.0) * mass * size**2

        def create_body(name):
            body = UsdGeom.Xform.Define(stage, f"/World/{name}")
            UsdPhysics.RigidBodyAPI.Apply(body.GetPrim())
            collider = UsdGeom.Cube.Define(stage, f"/World/{name}/Collider")
            collider.CreateSizeAttr().Set(size)
            UsdPhysics.CollisionAPI.Apply(collider.GetPrim())
            return body.GetPrim(), collider.GetPrim()

        # A: density on the collider shape derives mass and inertia.
        body_prim, collider_prim = create_body("A")
        UsdPhysics.MassAPI.Apply(body_prim)
        UsdPhysics.MassAPI.Apply(collider_prim).CreateDensityAttr().Set(density)

        # B: only mass authored on body, inertia scaled from shape accumulation.
        body_prim, _ = create_body("B")
        UsdPhysics.MassAPI.Apply(body_prim).CreateMassAttr().Set(mass)

        builder = newton.ModelBuilder()
        result = builder.add_usd(stage)
        idx_a = result["path_body_map"]["/World/A"]
        idx_b = result["path_body_map"]["/World/B"]

        for idx, name in ((idx_a, "A"), (idx_b, "B")):
            self.assertAlmostEqual(builder.body_mass[idx], mass, places=5, msg=f"Body {name} mass")
            inertia = np.array(builder.body_inertia[idx]).reshape(3, 3)
            np.testing.assert_allclose(
                np.diag(inertia), [expected_i] * 3, atol=1e-5, rtol=1e-5, err_msg=f"Body {name} inertia"
            )

    @unittest.skipUnless(USD_AVAILABLE, "Requires usd-core")
    def test_massapi_partial_body_applies_axis_rotation_in_compute_callback(self):
        """Compute fallback must rotate cone/capsule/cylinder mass frame for non-Z axes."""
        from pxr import Usd, UsdGeom, UsdPhysics

        stage = Usd.Stage.CreateInMemory()
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
        UsdGeom.SetStageMetersPerUnit(stage, 1.0)
        UsdPhysics.Scene.Define(stage, "/physicsScene")

        body = UsdGeom.Xform.Define(stage, "/World/Body")
        body_prim = body.GetPrim()
        UsdPhysics.RigidBodyAPI.Apply(body_prim)
        # Partial body MassAPI -> triggers ComputeMassProperties callback path.
        UsdPhysics.MassAPI.Apply(body_prim).CreateMassAttr().Set(1.0)

        # Cone inertia/computation is defined in the local +Z frame; use +X axis to require
        # axis correction in the callback mass_info.localRot.
        cone = UsdGeom.Cone.Define(stage, "/World/Body/Collider")
        cone.CreateRadiusAttr().Set(0.5)
        cone.CreateHeightAttr().Set(2.0)
        cone.CreateAxisAttr().Set(UsdGeom.Tokens.x)
        collider_prim = cone.GetPrim()
        UsdPhysics.CollisionAPI.Apply(collider_prim)

        builder = newton.ModelBuilder()
        result = builder.add_usd(stage)
        body_idx = result["path_body_map"]["/World/Body"]

        # For cone mass m=1, radius r=0.5, height h=2.0:
        # Ia = Iyy = Izz = 3/20*m*r^2 + 3/80*m*h^2 = 0.1875 (about transverse axes)
        # Ib = Ixx = 3/10*m*r^2 = 0.075 (about symmetry axis along +X)
        inertia = np.array(builder.body_inertia[body_idx]).reshape(3, 3)
        expected_diag = np.array([0.075, 0.1875, 0.1875], dtype=np.float32)
        np.testing.assert_allclose(np.diag(inertia), expected_diag, atol=1e-5, rtol=1e-5)
        np.testing.assert_allclose(
            inertia - np.diag(np.diag(inertia)),
            np.zeros((3, 3), dtype=np.float32),
            atol=1e-6,
        )

        # Cone COM should also rotate from local -Z to world -X.
        body_com = np.array(builder.body_com[body_idx], dtype=np.float32)
        np.testing.assert_allclose(body_com, np.array([-0.5, 0.0, 0.0], dtype=np.float32), atol=1e-5, rtol=1e-5)

    @unittest.skipUnless(USD_AVAILABLE, "Requires usd-core")
    def test_massapi_partial_body_mesh_uses_cached_mesh_loading(self):
        """Mesh collider mass fallback should not reload the same USD mesh multiple times."""
        from pxr import Usd, UsdGeom, UsdPhysics

        stage = Usd.Stage.CreateInMemory()
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
        UsdGeom.SetStageMetersPerUnit(stage, 1.0)
        UsdPhysics.Scene.Define(stage, "/physicsScene")

        body = UsdGeom.Xform.Define(stage, "/World/Body")
        body_prim = body.GetPrim()
        UsdPhysics.RigidBodyAPI.Apply(body_prim)
        # Partial body MassAPI -> triggers ComputeMassProperties callback path.
        UsdPhysics.MassAPI.Apply(body_prim).CreateMassAttr().Set(1.0)

        mesh = UsdGeom.Mesh.Define(stage, "/World/Body/Collider")
        mesh_prim = mesh.GetPrim()
        UsdPhysics.CollisionAPI.Apply(mesh_prim)

        # Closed tetrahedron mesh so inertia/mass can be derived.
        mesh.CreatePointsAttr().Set(
            [
                (-1.0, -1.0, -1.0),
                (1.0, -1.0, 1.0),
                (-1.0, 1.0, 1.0),
                (1.0, 1.0, -1.0),
            ]
        )
        mesh.CreateFaceVertexCountsAttr().Set([3, 3, 3, 3])
        mesh.CreateFaceVertexIndicesAttr().Set(
            [
                0,
                2,
                1,
                0,
                1,
                3,
                0,
                3,
                2,
                1,
                2,
                3,
            ]
        )

        import newton._src.utils.import_usd as import_usd_module  # noqa: PLC0415

        original_get_mesh = import_usd_module.usd.get_mesh
        get_mesh_call_count = 0

        def _counting_get_mesh(*args, **kwargs):
            nonlocal get_mesh_call_count
            get_mesh_call_count += 1
            return original_get_mesh(*args, **kwargs)

        with mock.patch(
            "newton._src.utils.import_usd.usd.get_mesh",
            side_effect=_counting_get_mesh,
        ):
            builder = newton.ModelBuilder()
            builder.add_usd(stage)

        self.assertEqual(get_mesh_call_count, 1)

    @unittest.skipUnless(USD_AVAILABLE, "Requires usd-core")
    def test_massapi_partial_body_warns_and_skips_noncontributing_collider(self):
        """Fallback compute warns and skips colliders that cannot provide positive mass info."""
        from pxr import Usd, UsdGeom, UsdPhysics

        stage = Usd.Stage.CreateInMemory()
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
        UsdGeom.SetStageMetersPerUnit(stage, 1.0)
        UsdPhysics.Scene.Define(stage, "/physicsScene")

        body = UsdGeom.Xform.Define(stage, "/World/Body")
        body_prim = body.GetPrim()
        UsdPhysics.RigidBodyAPI.Apply(body_prim)
        # Partial body MassAPI -> triggers compute fallback.
        UsdPhysics.MassAPI.Apply(body_prim).CreateMassAttr().Set(1.0)

        collider = UsdGeom.Cube.Define(stage, "/World/Body/Collider")
        collider.CreateSizeAttr().Set(0.0)
        UsdPhysics.CollisionAPI.Apply(collider.GetPrim())
        # Intentionally no MassAPI and zero geometric size -> non-contributing collider.

        builder = newton.ModelBuilder()
        with self.assertWarnsRegex(UserWarning, r"Skipping collider .* mass aggregation"):
            builder.add_usd(stage)

    @unittest.skipUnless(USD_AVAILABLE, "Requires usd-core")
    def test_contact_margin_parsing(self):
        """Test that newton:contactMargin is parsed into shape margin [m]."""
        from pxr import Usd, UsdGeom, UsdPhysics

        stage = Usd.Stage.CreateInMemory()
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
        UsdGeom.SetStageMetersPerUnit(stage, 1.0)
        UsdPhysics.Scene.Define(stage, "/physicsScene")

        UsdGeom.Xform.Define(stage, "/Articulation")
        body = UsdGeom.Xform.Define(stage, "/Articulation/Body")
        UsdPhysics.RigidBodyAPI.Apply(body.GetPrim())

        collider1 = UsdGeom.Cube.Define(stage, "/Articulation/Body/Collider1")
        collider1_prim = collider1.GetPrim()
        collider1_prim.ApplyAPI("NewtonCollisionAPI")
        UsdPhysics.CollisionAPI.Apply(collider1_prim)
        collider1_prim.GetAttribute("newton:contactMargin").Set(0.05)

        collider2 = UsdGeom.Sphere.Define(stage, "/Articulation/Body/Collider2")
        collider2_prim = collider2.GetPrim()
        UsdPhysics.CollisionAPI.Apply(collider2_prim)

        builder = newton.ModelBuilder()
        builder.default_shape_cfg.margin = 1e-5
        builder.default_shape_cfg.gap = 0.01
        builder.rigid_gap = 0.01
        result = builder.add_usd(stage)
        model = builder.finalize()

        shape1_idx = result["path_shape_map"]["/Articulation/Body/Collider1"]
        shape2_idx = result["path_shape_map"]["/Articulation/Body/Collider2"]
        self.assertAlmostEqual(model.shape_margin.numpy()[shape1_idx], 0.05, places=4)
        self.assertAlmostEqual(model.shape_margin.numpy()[shape2_idx], 1e-5, places=6)

    @unittest.skipUnless(USD_AVAILABLE, "Requires usd-core")
    def test_contact_gap_parsing(self):
        """Test that newton:contactGap is parsed into shape gap [m]."""
        from pxr import Usd, UsdGeom, UsdPhysics

        stage = Usd.Stage.CreateInMemory()
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
        UsdGeom.SetStageMetersPerUnit(stage, 1.0)
        UsdPhysics.Scene.Define(stage, "/physicsScene")

        UsdGeom.Xform.Define(stage, "/Articulation")
        body = UsdGeom.Xform.Define(stage, "/Articulation/Body")
        UsdPhysics.RigidBodyAPI.Apply(body.GetPrim())

        collider1 = UsdGeom.Cube.Define(stage, "/Articulation/Body/Collider1")
        collider1_prim = collider1.GetPrim()
        collider1_prim.ApplyAPI("NewtonCollisionAPI")
        UsdPhysics.CollisionAPI.Apply(collider1_prim)
        collider1_prim.GetAttribute("newton:contactGap").Set(0.02)

        collider2 = UsdGeom.Sphere.Define(stage, "/Articulation/Body/Collider2")
        collider2_prim = collider2.GetPrim()
        UsdPhysics.CollisionAPI.Apply(collider2_prim)

        builder = newton.ModelBuilder()
        builder.default_shape_cfg.margin = 0.0
        builder.default_shape_cfg.gap = 0.01
        builder.rigid_gap = 0.01
        result = builder.add_usd(stage)
        model = builder.finalize()

        shape1_idx = result["path_shape_map"]["/Articulation/Body/Collider1"]
        shape2_idx = result["path_shape_map"]["/Articulation/Body/Collider2"]
        self.assertAlmostEqual(model.shape_gap.numpy()[shape1_idx], 0.02, places=4)
        self.assertAlmostEqual(model.shape_gap.numpy()[shape2_idx], 0.01, places=4)

    @unittest.skipUnless(USD_AVAILABLE, "Requires usd-core")
    def test_contact_response_parsing(self):
        """Test ke/kd/kf/ka parsed from NewtonMaterialAPI on bound material."""
        from pxr import Usd, UsdGeom, UsdPhysics, UsdShade

        stage = Usd.Stage.CreateInMemory()
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
        UsdGeom.SetStageMetersPerUnit(stage, 1.0)
        UsdPhysics.Scene.Define(stage, "/physicsScene")

        # Material with all contact response attrs authored
        mat_all = UsdShade.Material.Define(stage, "/Materials/AllAuthored")
        mat_all_prim = mat_all.GetPrim()
        mat_all_prim.ApplyAPI("NewtonMaterialAPI")
        UsdPhysics.MaterialAPI.Apply(mat_all_prim)
        mat_all_prim.GetAttribute("newton:contactStiffness").Set(5000.0)
        mat_all_prim.GetAttribute("newton:contactDamping").Set(200.0)
        mat_all_prim.GetAttribute("newton:contactFrictionGain").Set(800.0)
        mat_all_prim.GetAttribute("newton:contactAdhesion").Set(0.01)

        # Material with only ke/kd authored (kf/ka use builder defaults)
        mat_partial = UsdShade.Material.Define(stage, "/Materials/PartialAuthored")
        mat_partial_prim = mat_partial.GetPrim()
        mat_partial_prim.ApplyAPI("NewtonMaterialAPI")
        UsdPhysics.MaterialAPI.Apply(mat_partial_prim)
        mat_partial_prim.GetAttribute("newton:contactStiffness").Set(3000.0)
        mat_partial_prim.GetAttribute("newton:contactDamping").Set(150.0)

        UsdGeom.Xform.Define(stage, "/Articulation")
        body = UsdGeom.Xform.Define(stage, "/Articulation/Body")
        UsdPhysics.RigidBodyAPI.Apply(body.GetPrim())

        # Shape bound to full material
        col1 = UsdGeom.Cube.Define(stage, "/Articulation/Body/ColAll")
        col1_prim = col1.GetPrim()
        UsdPhysics.CollisionAPI.Apply(col1_prim)
        UsdShade.MaterialBindingAPI.Apply(col1_prim).Bind(mat_all, "physics")

        # Shape bound to partial material
        col2 = UsdGeom.Cube.Define(stage, "/Articulation/Body/ColPartial")
        col2_prim = col2.GetPrim()
        UsdPhysics.CollisionAPI.Apply(col2_prim)
        UsdShade.MaterialBindingAPI.Apply(col2_prim).Bind(mat_partial, "physics")

        # Shape with no material binding
        col3 = UsdGeom.Cube.Define(stage, "/Articulation/Body/ColNone")
        col3_prim = col3.GetPrim()
        UsdPhysics.CollisionAPI.Apply(col3_prim)

        builder = newton.ModelBuilder()
        result = builder.add_usd(stage)
        model = builder.finalize()

        idx_all = result["path_shape_map"]["/Articulation/Body/ColAll"]
        idx_partial = result["path_shape_map"]["/Articulation/Body/ColPartial"]
        idx_none = result["path_shape_map"]["/Articulation/Body/ColNone"]

        # Full material: all four attrs from material
        self.assertAlmostEqual(model.shape_material_ke.numpy()[idx_all], 5000.0, places=1)
        self.assertAlmostEqual(model.shape_material_kd.numpy()[idx_all], 200.0, places=1)
        self.assertAlmostEqual(model.shape_material_kf.numpy()[idx_all], 800.0, places=1)
        self.assertAlmostEqual(model.shape_material_ka.numpy()[idx_all], 0.01, places=4)

        # Partial material: ke/kd from material, kf/ka from builder defaults
        self.assertAlmostEqual(model.shape_material_ke.numpy()[idx_partial], 3000.0, places=1)
        self.assertAlmostEqual(model.shape_material_kd.numpy()[idx_partial], 150.0, places=1)
        self.assertAlmostEqual(model.shape_material_kf.numpy()[idx_partial], builder.default_shape_cfg.kf, places=1)
        self.assertAlmostEqual(model.shape_material_ka.numpy()[idx_partial], builder.default_shape_cfg.ka, places=4)

        # No material: all from builder defaults
        self.assertAlmostEqual(model.shape_material_ke.numpy()[idx_none], builder.default_shape_cfg.ke, places=1)
        self.assertAlmostEqual(model.shape_material_kd.numpy()[idx_none], builder.default_shape_cfg.kd, places=1)
        self.assertAlmostEqual(model.shape_material_kf.numpy()[idx_none], builder.default_shape_cfg.kf, places=1)
        self.assertAlmostEqual(model.shape_material_ka.numpy()[idx_none], builder.default_shape_cfg.ka, places=4)

    @unittest.skipUnless(USD_AVAILABLE, "Requires usd-core")
    def test_contact_response_inf_sentinel(self):
        """Test that -inf authored on material attrs yields builder defaults."""
        from pxr import Usd, UsdGeom, UsdPhysics, UsdShade

        stage = Usd.Stage.CreateInMemory()
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
        UsdGeom.SetStageMetersPerUnit(stage, 1.0)
        UsdPhysics.Scene.Define(stage, "/physicsScene")

        mat = UsdShade.Material.Define(stage, "/Materials/InfMat")
        mat_prim = mat.GetPrim()
        mat_prim.ApplyAPI("NewtonMaterialAPI")
        UsdPhysics.MaterialAPI.Apply(mat_prim)
        mat_prim.GetAttribute("newton:contactStiffness").Set(float("-inf"))
        mat_prim.GetAttribute("newton:contactDamping").Set(float("-inf"))
        mat_prim.GetAttribute("newton:contactFrictionGain").Set(float("-inf"))
        mat_prim.GetAttribute("newton:contactAdhesion").Set(float("-inf"))

        UsdGeom.Xform.Define(stage, "/Articulation")
        body = UsdGeom.Xform.Define(stage, "/Articulation/Body")
        UsdPhysics.RigidBodyAPI.Apply(body.GetPrim())

        col = UsdGeom.Cube.Define(stage, "/Articulation/Body/Col")
        col_prim = col.GetPrim()
        UsdPhysics.CollisionAPI.Apply(col_prim)
        UsdShade.MaterialBindingAPI.Apply(col_prim).Bind(mat, "physics")

        builder = newton.ModelBuilder()
        result = builder.add_usd(stage)
        model = builder.finalize()

        idx = result["path_shape_map"]["/Articulation/Body/Col"]
        self.assertAlmostEqual(model.shape_material_ke.numpy()[idx], builder.default_shape_cfg.ke, places=1)
        self.assertAlmostEqual(model.shape_material_kd.numpy()[idx], builder.default_shape_cfg.kd, places=1)
        self.assertAlmostEqual(model.shape_material_kf.numpy()[idx], builder.default_shape_cfg.kf, places=1)
        self.assertAlmostEqual(model.shape_material_ka.numpy()[idx], builder.default_shape_cfg.ka, places=4)

    @unittest.skipUnless(USD_AVAILABLE, "Requires usd-core")
    def test_contact_response_legacy_shape_fallback(self):
        """Test deprecated newton:contact_ke/kd/kf/ka on shape prim with exact warnings."""
        from pxr import Sdf, Usd, UsdGeom, UsdPhysics, UsdShade

        stage = Usd.Stage.CreateInMemory()
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
        UsdGeom.SetStageMetersPerUnit(stage, 1.0)
        UsdPhysics.Scene.Define(stage, "/physicsScene")

        # Material with NO contact response attrs
        mat_plain = UsdShade.Material.Define(stage, "/Materials/PlainMat")
        mat_plain_prim = mat_plain.GetPrim()
        mat_plain_prim.ApplyAPI("NewtonMaterialAPI")
        UsdPhysics.MaterialAPI.Apply(mat_plain_prim)

        # Material WITH all contact attrs authored
        mat_authored = UsdShade.Material.Define(stage, "/Materials/AuthoredMat")
        mat_authored_prim = mat_authored.GetPrim()
        mat_authored_prim.ApplyAPI("NewtonMaterialAPI")
        UsdPhysics.MaterialAPI.Apply(mat_authored_prim)
        mat_authored_prim.GetAttribute("newton:contactStiffness").Set(4000.0)
        mat_authored_prim.GetAttribute("newton:contactDamping").Set(100.0)
        mat_authored_prim.GetAttribute("newton:contactFrictionGain").Set(600.0)
        mat_authored_prim.GetAttribute("newton:contactAdhesion").Set(0.02)

        UsdGeom.Xform.Define(stage, "/Articulation")
        body = UsdGeom.Xform.Define(stage, "/Articulation/Body")
        UsdPhysics.RigidBodyAPI.Apply(body.GetPrim())

        # Legacy ke/kd/kf/ka on shape, plain material -> legacy used as fallback
        col_legacy = UsdGeom.Cube.Define(stage, "/Articulation/Body/ColLegacy")
        col_legacy_prim = col_legacy.GetPrim()
        UsdPhysics.CollisionAPI.Apply(col_legacy_prim)
        UsdShade.MaterialBindingAPI.Apply(col_legacy_prim).Bind(mat_plain, "physics")
        col_legacy_prim.CreateAttribute("newton:contact_ke", Sdf.ValueTypeNames.Float).Set(9999.0)
        col_legacy_prim.CreateAttribute("newton:contact_kd", Sdf.ValueTypeNames.Float).Set(777.0)
        col_legacy_prim.CreateAttribute("newton:contact_kf", Sdf.ValueTypeNames.Float).Set(500.0)
        col_legacy_prim.CreateAttribute("newton:contact_ka", Sdf.ValueTypeNames.Float).Set(0.05)

        # Legacy on shape AND material authored -> material wins over legacy
        col_both = UsdGeom.Cube.Define(stage, "/Articulation/Body/ColBoth")
        col_both_prim = col_both.GetPrim()
        UsdPhysics.CollisionAPI.Apply(col_both_prim)
        UsdShade.MaterialBindingAPI.Apply(col_both_prim).Bind(mat_authored, "physics")
        col_both_prim.CreateAttribute("newton:contact_ke", Sdf.ValueTypeNames.Float).Set(1111.0)
        col_both_prim.CreateAttribute("newton:contact_kd", Sdf.ValueTypeNames.Float).Set(222.0)
        col_both_prim.CreateAttribute("newton:contact_kf", Sdf.ValueTypeNames.Float).Set(333.0)
        col_both_prim.CreateAttribute("newton:contact_ka", Sdf.ValueTypeNames.Float).Set(0.09)

        builder = newton.ModelBuilder()
        with warnings.catch_warnings(record=True) as w:
            # Record only the deprecation warnings this test asserts on; leave every
            # other warning subject to the ambient policy so --strict-warnings still
            # fails on an unexpected newton.* warning here.
            warnings.filterwarnings("always", category=DeprecationWarning)
            result = builder.add_usd(stage)
            dep_warnings = [x for x in w if issubclass(x.category, DeprecationWarning)]
            dep_msgs = [str(x.message) for x in dep_warnings]

        model = builder.finalize()

        idx_legacy = result["path_shape_map"]["/Articulation/Body/ColLegacy"]
        idx_both = result["path_shape_map"]["/Articulation/Body/ColBoth"]

        # Legacy fallback used when material has no contact attrs
        self.assertAlmostEqual(model.shape_material_ke.numpy()[idx_legacy], 9999.0, places=1)
        self.assertAlmostEqual(model.shape_material_kd.numpy()[idx_legacy], 777.0, places=1)
        self.assertAlmostEqual(model.shape_material_kf.numpy()[idx_legacy], 500.0, places=1)
        self.assertAlmostEqual(model.shape_material_ka.numpy()[idx_legacy], 0.05, places=4)

        # Material value wins over legacy attr
        self.assertAlmostEqual(model.shape_material_ke.numpy()[idx_both], 4000.0, places=1)
        self.assertAlmostEqual(model.shape_material_kd.numpy()[idx_both], 100.0, places=1)
        self.assertAlmostEqual(model.shape_material_kf.numpy()[idx_both], 600.0, places=1)
        self.assertAlmostEqual(model.shape_material_ka.numpy()[idx_both], 0.02, places=4)

        # Each legacy attr emits an exact migration message naming its replacement;
        # both shapes author all four, so each message must appear exactly twice.
        legacy_to_material = {
            "newton:contact_ke": "newton:contactStiffness",
            "newton:contact_kd": "newton:contactDamping",
            "newton:contact_kf": "newton:contactFrictionGain",
            "newton:contact_ka": "newton:contactAdhesion",
        }
        for legacy_attr, material_attr in legacy_to_material.items():
            expected_msg = (
                f"'{legacy_attr}' on shape prim is deprecated; "
                f"author '{material_attr}' on the bound NewtonMaterialAPI material instead."
            )
            self.assertEqual(
                dep_msgs.count(expected_msg), 2, f"expected exactly two {legacy_attr!r} deprecation warnings"
            )

    @unittest.skipUnless(USD_AVAILABLE, "Requires usd-core")
    def test_contact_response_solref_over_material(self):
        """Test MuJoCo per-geom solref wins over material when MuJoCo resolver has priority."""
        from pxr import Sdf, Usd, UsdGeom, UsdPhysics, UsdShade

        from newton._src.usd.schemas import SchemaResolverMjc, SchemaResolverNewton  # noqa: PLC0415

        stage = Usd.Stage.CreateInMemory()
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
        UsdGeom.SetStageMetersPerUnit(stage, 1.0)
        UsdPhysics.Scene.Define(stage, "/physicsScene")

        mat = UsdShade.Material.Define(stage, "/Materials/Mat")
        mat_prim = mat.GetPrim()
        mat_prim.ApplyAPI("NewtonMaterialAPI")
        UsdPhysics.MaterialAPI.Apply(mat_prim)
        mat_prim.GetAttribute("newton:contactStiffness").Set(4000.0)
        mat_prim.GetAttribute("newton:contactDamping").Set(100.0)

        UsdGeom.Xform.Define(stage, "/Articulation")
        body = UsdGeom.Xform.Define(stage, "/Articulation/Body")
        UsdPhysics.RigidBodyAPI.Apply(body.GetPrim())

        col = UsdGeom.Cube.Define(stage, "/Articulation/Body/Col")
        col_prim = col.GetPrim()
        UsdPhysics.CollisionAPI.Apply(col_prim)
        UsdShade.MaterialBindingAPI.Apply(col_prim).Bind(mat, "physics")
        col_prim.CreateAttribute("mjc:solref", Sdf.ValueTypeNames.DoubleArray).Set([0.01, 0.5])

        # MuJoCo resolver first -> solref wins over material ke/kd
        builder = newton.ModelBuilder()
        SolverMuJoCo.register_custom_attributes(builder)
        result = builder.add_usd(stage, schema_resolvers=[SchemaResolverMjc(), SchemaResolverNewton()])
        model = builder.finalize()
        idx = result["path_shape_map"]["/Articulation/Body/Col"]

        expected_ke = 1.0 / (0.01**2 * 0.5**2)
        expected_kd = 2.0 / 0.01
        self.assertAlmostEqual(model.shape_material_ke.numpy()[idx], expected_ke, places=1)
        self.assertAlmostEqual(model.shape_material_kd.numpy()[idx], expected_kd, places=1)
        # kf/ka fall through to material (no MuJoCo per-shape kf/ka)
        self.assertAlmostEqual(model.shape_material_kf.numpy()[idx], builder.default_shape_cfg.kf, places=1)
        self.assertAlmostEqual(model.shape_material_ka.numpy()[idx], builder.default_shape_cfg.ka, places=4)

        # Newton resolver first -> material wins over solref
        builder2 = newton.ModelBuilder()
        SolverMuJoCo.register_custom_attributes(builder2)
        result2 = builder2.add_usd(stage, schema_resolvers=[SchemaResolverNewton(), SchemaResolverMjc()])
        model2 = builder2.finalize()
        idx2 = result2["path_shape_map"]["/Articulation/Body/Col"]

        self.assertAlmostEqual(model2.shape_material_ke.numpy()[idx2], 4000.0, places=1)
        self.assertAlmostEqual(model2.shape_material_kd.numpy()[idx2], 100.0, places=1)

    @unittest.skipUnless(USD_AVAILABLE, "Requires usd-core")
    def test_mimic_constraint_parsing(self):
        """Test that NewtonMimicAPI on a joint is parsed into a mimic constraint."""
        from pxr import Gf, Usd, UsdGeom, UsdPhysics

        stage = Usd.Stage.CreateInMemory()
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
        UsdGeom.SetStageMetersPerUnit(stage, 1.0)
        UsdPhysics.Scene.Define(stage, "/physicsScene")

        articulation = UsdGeom.Xform.Define(stage, "/World/Articulation")
        UsdPhysics.ArticulationRootAPI.Apply(articulation.GetPrim())

        root = UsdGeom.Xform.Define(stage, "/World/Articulation/Root")
        UsdPhysics.RigidBodyAPI.Apply(root.GetPrim())
        link1 = UsdGeom.Xform.Define(stage, "/World/Articulation/Link1")
        UsdPhysics.RigidBodyAPI.Apply(link1.GetPrim())
        link2 = UsdGeom.Xform.Define(stage, "/World/Articulation/Link2")
        UsdPhysics.RigidBodyAPI.Apply(link2.GetPrim())

        fixed = UsdPhysics.FixedJoint.Define(stage, "/World/Articulation/RootToWorld")
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
        joint2_prim = joint2.GetPrim()
        joint2_prim.ApplyAPI("NewtonMimicAPI")
        joint2_prim.GetRelationship("newton:mimicJoint").SetTargets([joint1.GetPrim().GetPath()])
        joint2_prim.GetAttribute("newton:mimicCoef0").Set(0.5)
        joint2_prim.GetAttribute("newton:mimicCoef1").Set(2.0)

        builder = newton.ModelBuilder()
        result = builder.add_usd(stage)
        model = builder.finalize()

        self.assertEqual(model.constraint_mimic_count, 1)
        path_joint_map = result["path_joint_map"]
        joint1_idx = path_joint_map["/World/Articulation/Joint1"]
        joint2_idx = path_joint_map["/World/Articulation/Joint2"]
        self.assertEqual(model.constraint_mimic_joint0.numpy()[0], joint2_idx)
        self.assertEqual(model.constraint_mimic_joint1.numpy()[0], joint1_idx)
        self.assertAlmostEqual(model.constraint_mimic_coef0.numpy()[0], 0.5, places=5)
        self.assertAlmostEqual(model.constraint_mimic_coef1.numpy()[0], 2.0, places=5)

    @unittest.skipUnless(USD_AVAILABLE, "Requires usd-core")
    def test_mjc_equality_joint_parsing(self):
        """Test that MjcEqualityJointAPI on a joint is parsed into an equality constraint."""
        from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics

        stage = Usd.Stage.CreateInMemory()
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
        UsdGeom.SetStageMetersPerUnit(stage, 1.0)
        UsdPhysics.Scene.Define(stage, "/physicsScene")

        articulation = UsdGeom.Xform.Define(stage, "/World/Articulation")
        UsdPhysics.ArticulationRootAPI.Apply(articulation.GetPrim())

        root = UsdGeom.Xform.Define(stage, "/World/Articulation/Root")
        UsdPhysics.RigidBodyAPI.Apply(root.GetPrim())
        link1 = UsdGeom.Xform.Define(stage, "/World/Articulation/Link1")
        UsdPhysics.RigidBodyAPI.Apply(link1.GetPrim())
        link2 = UsdGeom.Xform.Define(stage, "/World/Articulation/Link2")
        UsdPhysics.RigidBodyAPI.Apply(link2.GetPrim())

        fixed = UsdPhysics.FixedJoint.Define(stage, "/World/Articulation/RootToWorld")
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

        joint1_prim = joint1.GetPrim()
        joint1_prim.SetMetadata("apiSchemas", Sdf.TokenListOp.Create(prependedItems=["MjcEqualityJointAPI"]))
        joint1_prim.CreateRelationship("mjc:target").SetTargets([joint2.GetPrim().GetPath()])

        joint2_prim = joint2.GetPrim()
        joint2_prim.SetMetadata("apiSchemas", Sdf.TokenListOp.Create(prependedItems=["MjcEqualityJointAPI"]))
        joint2_prim.CreateRelationship("mjc:target").SetTargets([joint1.GetPrim().GetPath()])
        joint2_prim.CreateAttribute("mjc:coef0", Sdf.ValueTypeNames.Double).Set(0.5)
        joint2_prim.CreateAttribute("mjc:coef1", Sdf.ValueTypeNames.Double).Set(1.5)
        joint2_prim.CreateAttribute("mjc:coef2", Sdf.ValueTypeNames.Double).Set(0.1)
        joint2_prim.CreateAttribute("mjc:coef3", Sdf.ValueTypeNames.Double).Set(0.05)
        joint2_prim.CreateAttribute("mjc:coef4", Sdf.ValueTypeNames.Double).Set(0.02)
        joint2_prim.CreateAttribute("mjc:solref", Sdf.ValueTypeNames.DoubleArray).Set([0.03, 0.8])
        joint2_prim.CreateAttribute("mjc:solimp", Sdf.ValueTypeNames.DoubleArray).Set([0.8, 0.9, 0.002, 0.6, 3.0])

        builder = newton.ModelBuilder()
        SolverMuJoCo.register_custom_attributes(builder)
        result = builder.add_usd(stage, convert_mjc_equality_constraints=False)
        model = builder.finalize()

        self.assertEqual(model.mujoco.equality_constraint_count, 2)
        eq_by_label = {label: i for i, label in enumerate(model.mujoco.equality_constraint_label)}
        joint1_eq = eq_by_label["/World/Articulation/Joint1"]
        joint2_eq = eq_by_label["/World/Articulation/Joint2"]
        joint1_idx = result["path_joint_map"]["/World/Articulation/Joint1"]
        joint2_idx = result["path_joint_map"]["/World/Articulation/Joint2"]
        self.assertEqual(model.mujoco.equality_constraint_joint1.numpy()[joint1_eq], joint1_idx)
        self.assertEqual(model.mujoco.equality_constraint_joint2.numpy()[joint1_eq], joint2_idx)
        self.assertEqual(model.mujoco.equality_constraint_joint1.numpy()[joint2_eq], joint2_idx)
        self.assertEqual(model.mujoco.equality_constraint_joint2.numpy()[joint2_eq], joint1_idx)
        np.testing.assert_allclose(
            model.mujoco.equality_constraint_polycoef.numpy()[joint1_eq],
            np.array([0.0, 1.0, 0.0, 0.0, 0.0], dtype=np.float32),
            rtol=1e-6,
            atol=1e-6,
        )
        np.testing.assert_allclose(
            model.mujoco.equality_constraint_polycoef.numpy()[joint2_eq],
            np.array([0.5, 1.5, 0.1, 0.05, 0.02], dtype=np.float32),
            rtol=1e-6,
            atol=1e-6,
        )
        np.testing.assert_allclose(model.mujoco.eq_solref.numpy()[joint2_eq], np.array([0.03, 0.8], dtype=np.float32))
        np.testing.assert_allclose(
            model.mujoco.eq_solimp.numpy()[joint2_eq],
            np.array([0.8, 0.9, 0.002, 0.6, 3.0], dtype=np.float32),
        )

    @unittest.skipUnless(USD_AVAILABLE, "Requires usd-core")
    def test_mjc_equality_connect_site_parsing(self):
        """Test that MjcEqualityConnectAPI on a spherical joint is parsed as a connect equality constraint."""
        from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics

        stage = Usd.Stage.CreateInMemory()
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
        UsdGeom.SetStageMetersPerUnit(stage, 1.0)
        UsdPhysics.Scene.Define(stage, "/physicsScene")

        world = UsdGeom.Xform.Define(stage, "/World")
        stage.SetDefaultPrim(world.GetPrim())

        body0 = UsdGeom.Cube.Define(stage, "/World/Body0")
        body0.CreateSizeAttr(0.2)
        body0_prim = body0.GetPrim()
        UsdPhysics.RigidBodyAPI.Apply(body0_prim)
        UsdPhysics.CollisionAPI.Apply(body0_prim)

        body1 = UsdGeom.Cube.Define(stage, "/World/Body1")
        body1.CreateSizeAttr(0.2)
        body1.AddTranslateOp().Set(Gf.Vec3f(1.0, 0.0, 0.0))
        body1_prim = body1.GetPrim()
        UsdPhysics.RigidBodyAPI.Apply(body1_prim)
        UsdPhysics.CollisionAPI.Apply(body1_prim)

        site0 = UsdGeom.Xform.Define(stage, "/World/Body0/Site0")
        site0.AddTranslateOp().Set(Gf.Vec3f(0.1, 0.0, 0.0))
        site0.GetPrim().SetMetadata("apiSchemas", Sdf.TokenListOp.Create(prependedItems=["MjcSiteAPI"]))
        site1 = UsdGeom.Xform.Define(stage, "/World/Body1/Site1")
        site1.AddTranslateOp().Set(Gf.Vec3f(-0.2, 0.0, 0.0))
        site1.GetPrim().SetMetadata("apiSchemas", Sdf.TokenListOp.Create(prependedItems=["MjcSiteAPI"]))

        connect = UsdPhysics.SphericalJoint.Define(stage, "/World/EqualityConnect")
        connect.CreateBody0Rel().SetTargets([site0.GetPath()])
        connect.CreateBody1Rel().SetTargets([site1.GetPath()])
        connect.CreateLocalPos0Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
        connect.CreateLocalPos1Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
        connect.CreateLocalRot0Attr().Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
        connect.CreateLocalRot1Attr().Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
        connect.CreateExcludeFromArticulationAttr().Set(True)
        connect_prim = connect.GetPrim()
        connect_prim.SetMetadata("apiSchemas", Sdf.TokenListOp.Create(prependedItems=["MjcEqualityConnectAPI"]))
        connect_prim.CreateAttribute("mjc:solref", Sdf.ValueTypeNames.DoubleArray).Set([0.04, 0.7])

        connect_world = UsdPhysics.SphericalJoint.Define(stage, "/World/EqualityConnectBodyToWorld")
        connect_world.CreateBody0Rel().SetTargets([body0.GetPath()])
        # The MJCF-to-USD converter represents world with the non-rigid default
        # prim instead of leaving the relationship target empty.
        connect_world.CreateBody1Rel().SetTargets([world.GetPath()])
        connect_world.CreateLocalPos0Attr().Set(Gf.Vec3f(0.25, -0.1, 0.3))
        connect_world.CreateLocalPos1Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
        connect_world.CreateLocalRot0Attr().Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
        connect_world.CreateLocalRot1Attr().Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
        connect_world.CreateExcludeFromArticulationAttr().Set(True)
        connect_world_prim = connect_world.GetPrim()
        connect_world_prim.SetMetadata("apiSchemas", Sdf.TokenListOp.Create(prependedItems=["MjcEqualityConnectAPI"]))

        builder = newton.ModelBuilder()
        SolverMuJoCo.register_custom_attributes(builder)
        result = builder.add_usd(
            stage,
            load_sites=False,
            schema_resolvers=[usd.SchemaResolverMjc()],
            convert_mjc_equality_constraints=False,
        )
        self.assertEqual(builder.body_count, 2)
        self.assertEqual(builder.joint_count, 2)
        self.assertEqual(builder.joint_type.count(newton.JointType.FREE), 2)
        self.assertEqual(builder.joint_dof_count, 12)
        self.assertEqual(builder.joint_coord_count, 14)
        model = builder.finalize()

        self.assertNotIn("/World/EqualityConnect", result["path_joint_map"])
        self.assertNotIn("/World/EqualityConnectBodyToWorld", result["path_joint_map"])
        self.assertIn("/World/EqualityConnect", result["schema_attrs"]["mjc"])
        np.testing.assert_allclose(
            result["schema_attrs"]["mjc"]["/World/EqualityConnect"]["mjc:solref"],
            np.array([0.04, 0.7]),
        )
        self.assertEqual(model.joint_count, 2)
        self.assertEqual(model.joint_dof_count, 12)
        self.assertEqual(model.joint_coord_count, 14)
        self.assertEqual(model.mujoco.equality_constraint_count, 2)
        eq_by_label = {label: i for i, label in enumerate(model.mujoco.equality_constraint_label)}
        site_eq = eq_by_label["/World/EqualityConnect"]
        world_eq = eq_by_label["/World/EqualityConnectBodyToWorld"]
        body0_idx = result["path_body_map"]["/World/Body0"]
        body1_idx = result["path_body_map"]["/World/Body1"]
        self.assertEqual(model.mujoco.equality_constraint_body1.numpy()[site_eq], body0_idx)
        self.assertEqual(model.mujoco.equality_constraint_body2.numpy()[site_eq], body1_idx)
        np.testing.assert_allclose(model.mujoco.equality_constraint_anchor.numpy()[site_eq], np.array([0.1, 0.0, 0.0]))
        self.assertEqual(model.mujoco.equality_constraint_body1.numpy()[world_eq], body0_idx)
        self.assertEqual(model.mujoco.equality_constraint_body2.numpy()[world_eq], -1)
        np.testing.assert_allclose(
            model.mujoco.equality_constraint_anchor.numpy()[world_eq],
            np.array([0.25, -0.1, 0.3], dtype=np.float32),
            rtol=1e-6,
            atol=1e-6,
        )
        np.testing.assert_allclose(model.mujoco.eq_solref.numpy()[site_eq], np.array([0.04, 0.7], dtype=np.float32))
        np.testing.assert_allclose(
            model.mujoco.eq_solimp.numpy()[site_eq],
            np.array([0.9, 0.95, 0.001, 0.5, 2.0], dtype=np.float32),
        )

    @unittest.skipUnless(USD_AVAILABLE, "Requires usd-core")
    def test_mjc_equality_disabled_connect_filtering(self):
        """Test that disabled MjcEqualityConnectAPI prims honor only_load_enabled_joints."""
        from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics

        stage = Usd.Stage.CreateInMemory()
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
        UsdGeom.SetStageMetersPerUnit(stage, 1.0)
        UsdPhysics.Scene.Define(stage, "/physicsScene")

        body0 = UsdGeom.Cube.Define(stage, "/World/Body0")
        body0.CreateSizeAttr(0.2)
        UsdPhysics.RigidBodyAPI.Apply(body0.GetPrim())
        UsdPhysics.CollisionAPI.Apply(body0.GetPrim())

        body1 = UsdGeom.Cube.Define(stage, "/World/Body1")
        body1.CreateSizeAttr(0.2)
        UsdPhysics.RigidBodyAPI.Apply(body1.GetPrim())
        UsdPhysics.CollisionAPI.Apply(body1.GetPrim())

        connect = UsdPhysics.SphericalJoint.Define(stage, "/World/DisabledEqualityConnect")
        connect.CreateBody0Rel().SetTargets([body0.GetPath()])
        connect.CreateBody1Rel().SetTargets([body1.GetPath()])
        connect.CreateLocalPos0Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
        connect.CreateLocalPos1Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
        connect.CreateLocalRot0Attr().Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
        connect.CreateLocalRot1Attr().Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
        connect.CreateJointEnabledAttr().Set(False)
        connect_prim = connect.GetPrim()
        connect_prim.SetMetadata("apiSchemas", Sdf.TokenListOp.Create(prependedItems=["MjcEqualityConnectAPI"]))

        builder = newton.ModelBuilder()
        SolverMuJoCo.register_custom_attributes(builder)
        builder.add_usd(stage, convert_mjc_equality_constraints=False)
        model = builder.finalize()

        self.assertEqual(model.mujoco.equality_constraint_count, 0)

        builder = newton.ModelBuilder()
        SolverMuJoCo.register_custom_attributes(builder)
        builder.add_usd(stage, only_load_enabled_joints=False, convert_mjc_equality_constraints=False)
        model = builder.finalize()

        self.assertEqual(model.mujoco.equality_constraint_count, 1)
        self.assertFalse(bool(model.mujoco.equality_constraint_enabled.numpy()[0]))

    @unittest.skipUnless(USD_AVAILABLE, "Requires usd-core")
    def test_mjc_equality_weld_parsing(self):
        """Test that MjcEqualityWeldAPI on a fixed joint is parsed as a weld equality constraint."""
        from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics

        stage = Usd.Stage.CreateInMemory()
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
        UsdGeom.SetStageMetersPerUnit(stage, 1.0)
        UsdPhysics.Scene.Define(stage, "/physicsScene")

        world = UsdGeom.Xform.Define(stage, "/World")
        stage.SetDefaultPrim(world.GetPrim())

        body0 = UsdGeom.Cube.Define(stage, "/World/Body0")
        body0.CreateSizeAttr(0.2)
        body0_prim = body0.GetPrim()
        UsdPhysics.RigidBodyAPI.Apply(body0_prim)
        UsdPhysics.CollisionAPI.Apply(body0_prim)

        body1 = UsdGeom.Cube.Define(stage, "/World/Body1")
        body1.CreateSizeAttr(0.2)
        body1.AddTranslateOp().Set(Gf.Vec3f(1.0, 0.0, 0.0))
        body1_prim = body1.GetPrim()
        UsdPhysics.RigidBodyAPI.Apply(body1_prim)
        UsdPhysics.CollisionAPI.Apply(body1_prim)

        site1 = UsdGeom.Xform.Define(stage, "/World/Body1/Site1")
        site1.AddTranslateOp().Set(Gf.Vec3f(0.2, -0.1, 0.3))
        site1.GetPrim().SetMetadata("apiSchemas", Sdf.TokenListOp.Create(prependedItems=["MjcSiteAPI"]))

        sqrt_half = math.sqrt(0.5)
        weld = UsdPhysics.FixedJoint.Define(stage, "/World/EqualityWeld")
        weld.CreateBody0Rel().SetTargets([body0.GetPath()])
        weld.CreateBody1Rel().SetTargets([site1.GetPath()])
        weld.CreateLocalPos0Attr().Set(Gf.Vec3f(0.25, -0.2, 0.1))
        weld.CreateLocalPos1Attr().Set(Gf.Vec3f(0.1, 0.3, -0.2))
        weld.CreateLocalRot0Attr().Set(Gf.Quatf(sqrt_half, 0.0, 0.0, sqrt_half))
        weld.CreateLocalRot1Attr().Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
        weld.CreateExcludeFromArticulationAttr().Set(True)
        weld_prim = weld.GetPrim()
        weld_prim.SetMetadata("apiSchemas", Sdf.TokenListOp.Create(prependedItems=["MjcEqualityWeldAPI"]))
        weld_prim.CreateAttribute("mjc:torqueScale", Sdf.ValueTypeNames.Float).Set(2.5)

        weld_world = UsdPhysics.FixedJoint.Define(stage, "/World/EqualityWeldBodyToWorld")
        weld_world.CreateBody0Rel().SetTargets([body0.GetPath()])
        weld_world.CreateBody1Rel().SetTargets([world.GetPath()])
        weld_world.CreateLocalPos0Attr().Set(Gf.Vec3f(0.4, 0.5, 0.6))
        weld_world.CreateLocalPos1Attr().Set(Gf.Vec3f(0.1, 0.2, 0.3))
        weld_world.CreateLocalRot0Attr().Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
        weld_world.CreateLocalRot1Attr().Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
        weld_world.CreateExcludeFromArticulationAttr().Set(True)
        weld_world.GetPrim().SetMetadata("apiSchemas", Sdf.TokenListOp.Create(prependedItems=["MjcEqualityWeldAPI"]))

        builder = newton.ModelBuilder()
        SolverMuJoCo.register_custom_attributes(builder)
        result = builder.add_usd(
            stage,
            load_sites=False,
            schema_resolvers=[usd.SchemaResolverMjc()],
            convert_mjc_equality_constraints=False,
        )
        self.assertEqual(builder.body_count, 2)
        self.assertEqual(builder.joint_count, 2)
        self.assertEqual(builder.joint_type.count(newton.JointType.FREE), 2)
        self.assertEqual(builder.joint_dof_count, 12)
        self.assertEqual(builder.joint_coord_count, 14)
        model = builder.finalize()

        self.assertNotIn("/World/EqualityWeld", result["path_joint_map"])
        self.assertNotIn("/World/EqualityWeldBodyToWorld", result["path_joint_map"])
        self.assertEqual(model.joint_count, 2)
        self.assertEqual(model.joint_dof_count, 12)
        self.assertEqual(model.joint_coord_count, 14)
        self.assertEqual(model.mujoco.equality_constraint_count, 2)
        weld_eq = model.mujoco.equality_constraint_label.index("/World/EqualityWeld")
        world_eq = model.mujoco.equality_constraint_label.index("/World/EqualityWeldBodyToWorld")
        body0_idx = result["path_body_map"]["/World/Body0"]
        body1_idx = result["path_body_map"]["/World/Body1"]
        self.assertEqual(model.mujoco.equality_constraint_body1.numpy()[weld_eq], body0_idx)
        self.assertEqual(model.mujoco.equality_constraint_body2.numpy()[weld_eq], body1_idx)
        np.testing.assert_allclose(
            model.mujoco.equality_constraint_anchor.numpy()[weld_eq],
            np.array([0.2, -0.1, 0.3], dtype=np.float32),
            rtol=1e-6,
            atol=1e-6,
        )
        np.testing.assert_allclose(
            model.mujoco.equality_constraint_relpose.numpy()[weld_eq],
            np.array([0.45, -0.5, 0.0, 0.0, 0.0, sqrt_half, sqrt_half], dtype=np.float32),
            rtol=1e-6,
            atol=1e-6,
        )
        np.testing.assert_allclose(
            model.mujoco.equality_constraint_torquescale.numpy()[weld_eq], np.array(2.5), rtol=1e-6, atol=1e-6
        )
        np.testing.assert_allclose(model.mujoco.eq_solref.numpy()[weld_eq], np.array([0.02, 1.0], dtype=np.float32))
        np.testing.assert_allclose(
            model.mujoco.eq_solimp.numpy()[weld_eq],
            np.array([0.9, 0.95, 0.001, 0.5, 2.0], dtype=np.float32),
        )
        self.assertEqual(model.mujoco.equality_constraint_body1.numpy()[world_eq], body0_idx)
        self.assertEqual(model.mujoco.equality_constraint_body2.numpy()[world_eq], -1)
        np.testing.assert_allclose(
            model.mujoco.equality_constraint_anchor.numpy()[world_eq],
            np.array([0.1, 0.2, 0.3], dtype=np.float32),
            rtol=1e-6,
            atol=1e-6,
        )
        np.testing.assert_allclose(
            model.mujoco.equality_constraint_relpose.numpy()[world_eq],
            np.array([0.3, 0.3, 0.3, 0.0, 0.0, 0.0, 1.0], dtype=np.float32),
            rtol=1e-6,
            atol=1e-6,
        )

    @unittest.skipUnless(USD_AVAILABLE, "Requires usd-core")
    def test_mjc_equality_conversion_roundtrips_to_mujoco(self):
        """Converted USD MJC equalities recreate the same MuJoCo equality constraints."""
        from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics

        def build_stage():
            stage = Usd.Stage.CreateInMemory()
            UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
            UsdGeom.SetStageMetersPerUnit(stage, 1.0)
            UsdPhysics.Scene.Define(stage, "/physicsScene")

            articulation = UsdGeom.Xform.Define(stage, "/World/Articulation")
            UsdPhysics.ArticulationRootAPI.Apply(articulation.GetPrim())

            root = UsdGeom.Cube.Define(stage, "/World/Articulation/Root")
            root.CreateSizeAttr(0.2)
            UsdPhysics.RigidBodyAPI.Apply(root.GetPrim())
            UsdPhysics.CollisionAPI.Apply(root.GetPrim())

            link1 = UsdGeom.Cube.Define(stage, "/World/Articulation/Link1")
            link1.CreateSizeAttr(0.2)
            UsdPhysics.RigidBodyAPI.Apply(link1.GetPrim())
            UsdPhysics.CollisionAPI.Apply(link1.GetPrim())

            link2 = UsdGeom.Cube.Define(stage, "/World/Articulation/Link2")
            link2.CreateSizeAttr(0.2)
            UsdPhysics.RigidBodyAPI.Apply(link2.GetPrim())
            UsdPhysics.CollisionAPI.Apply(link2.GetPrim())

            fixed = UsdPhysics.FixedJoint.Define(stage, "/World/Articulation/RootToWorld")
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
            joint2_prim = joint2.GetPrim()
            joint2_prim.SetMetadata("apiSchemas", Sdf.TokenListOp.Create(prependedItems=["MjcEqualityJointAPI"]))
            joint2_prim.CreateRelationship("mjc:target").SetTargets([joint1.GetPrim().GetPath()])
            joint2_prim.CreateAttribute("mjc:coef0", Sdf.ValueTypeNames.Double).Set(0.5)
            joint2_prim.CreateAttribute("mjc:coef1", Sdf.ValueTypeNames.Double).Set(1.5)
            joint2_prim.CreateAttribute("mjc:coef2", Sdf.ValueTypeNames.Double).Set(0.1)
            joint2_prim.CreateAttribute("mjc:coef3", Sdf.ValueTypeNames.Double).Set(0.05)
            joint2_prim.CreateAttribute("mjc:coef4", Sdf.ValueTypeNames.Double).Set(0.02)
            joint2_prim.CreateAttribute("mjc:solref", Sdf.ValueTypeNames.DoubleArray).Set([0.03, 0.8])
            joint2_prim.CreateAttribute("mjc:solimp", Sdf.ValueTypeNames.DoubleArray).Set([0.6, 0.7, 0.004, 0.5, 1.5])

            connect = UsdPhysics.SphericalJoint.Define(stage, "/World/Articulation/EqualityConnect")
            connect.CreateBody0Rel().SetTargets([link1.GetPath()])
            connect.CreateBody1Rel().SetTargets([link2.GetPath()])
            connect.CreateLocalPos0Attr().Set(Gf.Vec3f(0.1, 0.2, 0.3))
            connect.CreateLocalPos1Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
            connect.CreateLocalRot0Attr().Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
            connect.CreateLocalRot1Attr().Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
            connect.CreateExcludeFromArticulationAttr().Set(True)
            connect_prim = connect.GetPrim()
            connect_prim.SetMetadata("apiSchemas", Sdf.TokenListOp.Create(prependedItems=["MjcEqualityConnectAPI"]))
            connect_prim.CreateAttribute("mjc:solref", Sdf.ValueTypeNames.DoubleArray).Set([0.04, 0.7])
            connect_prim.CreateAttribute("mjc:solimp", Sdf.ValueTypeNames.DoubleArray).Set([0.8, 0.9, 0.002, 0.6, 3.0])

            sqrt_half = math.sqrt(0.5)
            weld = UsdPhysics.FixedJoint.Define(stage, "/World/Articulation/EqualityWeld")
            weld.CreateBody0Rel().SetTargets([link2.GetPath()])
            weld.CreateLocalPos0Attr().Set(Gf.Vec3f(0.2, 0.3, 0.4))
            weld.CreateLocalPos1Attr().Set(Gf.Vec3f(0.05, -0.1, 0.2))
            weld.CreateLocalRot0Attr().Set(Gf.Quatf(sqrt_half, 0.0, 0.0, sqrt_half))
            weld.CreateLocalRot1Attr().Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
            weld.CreateExcludeFromArticulationAttr().Set(True)
            weld.CreateJointEnabledAttr().Set(False)
            weld_prim = weld.GetPrim()
            weld_prim.SetMetadata("apiSchemas", Sdf.TokenListOp.Create(prependedItems=["MjcEqualityWeldAPI"]))
            weld_prim.CreateAttribute("mjc:torqueScale", Sdf.ValueTypeNames.Float).Set(2.5)
            weld_prim.CreateAttribute("mjc:solref", Sdf.ValueTypeNames.DoubleArray).Set([0.05, 1.2])
            weld_prim.CreateAttribute("mjc:solimp", Sdf.ValueTypeNames.DoubleArray).Set([0.7, 0.8, 0.003, 0.4, 2.0])
            return stage

        legacy_builder = newton.ModelBuilder()
        SolverMuJoCo.register_custom_attributes(legacy_builder)
        legacy_builder.add_usd(
            build_stage(),
            only_load_enabled_joints=False,
            convert_mjc_equality_constraints=False,
        )
        legacy_model = legacy_builder.finalize()
        legacy_solver = SolverMuJoCo(legacy_model)

        converted_builder = newton.ModelBuilder()
        SolverMuJoCo.register_custom_attributes(converted_builder)
        with self.assertWarnsRegex(UserWarning, "higher-order polycoef"):
            converted_builder.add_usd(build_stage(), only_load_enabled_joints=False)
        converted_model = converted_builder.finalize()
        converted_solver = SolverMuJoCo(converted_model)

        self.assertEqual(converted_model.mujoco.equality_constraint_count, 3)
        self.assertEqual(converted_model.constraint_mimic_count, 1)
        eq_types = converted_model.mujoco.equality_constraint_type.numpy()
        target_kinds = converted_model.mujoco.equality_constraint_target_kind.numpy()
        self.assertEqual(eq_types.tolist().count(int(newton.solvers.SolverMuJoCo.EqType.CONNECT)), 1)
        self.assertEqual(eq_types.tolist().count(int(newton.solvers.SolverMuJoCo.EqType.WELD)), 1)
        self.assertEqual(eq_types.tolist().count(int(newton.solvers.SolverMuJoCo.EqType.JOINT)), 1)
        self.assertEqual(target_kinds.tolist().count(int(MjcEqualityTargetKind.JOINT)), 2)
        self.assertEqual(target_kinds.tolist().count(int(MjcEqualityTargetKind.MIMIC)), 1)
        joint_eq = int(np.flatnonzero(eq_types == int(newton.solvers.SolverMuJoCo.EqType.JOINT))[0])
        self.assertEqual(target_kinds[joint_eq], int(MjcEqualityTargetKind.MIMIC))
        np.testing.assert_allclose(
            converted_model.mujoco.equality_constraint_polycoef.numpy()[joint_eq],
            np.array([0.5, 1.5, 0.1, 0.05, 0.02], dtype=np.float32),
        )

        self.assertEqual(converted_solver.mj_model.neq, legacy_solver.mj_model.neq)

        def equality_rows(solver):
            rows = []
            for i in range(solver.mj_model.neq):
                rows.append(
                    (
                        int(solver.mj_model.eq_type[i]),
                        bool(solver.mj_model.eq_active0[i]),
                        int(solver.mj_model.eq_obj1id[i]),
                        int(solver.mj_model.eq_obj2id[i]),
                        np.array(solver.mj_model.eq_data[i], dtype=np.float32),
                        np.array(solver.mj_model.eq_solref[i], dtype=np.float32),
                        np.array(solver.mj_model.eq_solimp[i], dtype=np.float32),
                    )
                )
            return sorted(rows, key=lambda row: (row[0], row[1], row[2], row[3], tuple(np.round(row[4], 8))))

        for converted_row, legacy_row in zip(
            equality_rows(converted_solver),
            equality_rows(legacy_solver),
            strict=True,
        ):
            self.assertEqual(converted_row[:4], legacy_row[:4])
            np.testing.assert_allclose(converted_row[4], legacy_row[4], atol=1e-6)
            np.testing.assert_allclose(converted_row[5], legacy_row[5], atol=1e-6)
            np.testing.assert_allclose(converted_row[6], legacy_row[6], atol=1e-6)

    @unittest.skipUnless(USD_AVAILABLE, "Requires usd-core")
    def test_joint_ordering_cycle_raises(self):
        """Topological sort errors (cycle/multi-root) must propagate instead of silently falling back."""
        from pxr import Gf, Usd, UsdGeom, UsdPhysics

        stage = Usd.Stage.CreateInMemory()
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
        UsdPhysics.Scene.Define(stage, "/physicsScene")

        articulation = UsdGeom.Xform.Define(stage, "/World/Articulation")
        UsdPhysics.ArticulationRootAPI.Apply(articulation.GetPrim())

        base = UsdGeom.Cube.Define(stage, "/World/Articulation/Base")
        base.CreateSizeAttr(0.2)
        UsdPhysics.RigidBodyAPI.Apply(base.GetPrim())
        UsdPhysics.CollisionAPI.Apply(base.GetPrim())

        child = UsdGeom.Cube.Define(stage, "/World/Articulation/Child")
        child.CreateSizeAttr(0.2)
        UsdPhysics.RigidBodyAPI.Apply(child.GetPrim())
        UsdPhysics.CollisionAPI.Apply(child.GetPrim())

        joint_x = UsdPhysics.PrismaticJoint.Define(stage, "/World/Articulation/JointX")
        joint_x.CreateBody0Rel().SetTargets([base.GetPath()])
        joint_x.CreateBody1Rel().SetTargets([child.GetPath()])
        joint_x.CreateAxisAttr().Set("X")
        joint_x.CreateLocalPos0Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
        joint_x.CreateLocalPos1Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
        joint_x.CreateLocalRot0Attr().Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
        joint_x.CreateLocalRot1Attr().Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))

        with mock.patch(
            "newton._src.utils.topology.topological_sort_undirected",
            side_effect=ValueError("Joint graph contains a cycle at body 0"),
        ):
            builder = newton.ModelBuilder()
            with self.assertRaises(ValueError):
                builder.add_usd(stage, joint_ordering="dfs", load_visual_shapes=False, load_sites=False)

    @unittest.skipUnless(USD_AVAILABLE, "Requires usd-core")
    def test_scene_gravity_enabled_parsing(self):
        """Test that gravity_enabled is parsed correctly from USD scene."""
        from pxr import Usd, UsdGeom, UsdPhysics

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
        self.assertGreater(np.linalg.norm(np.asarray(builder.gravity)), 0.0)

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
        np.testing.assert_allclose(builder2.gravity, (0.0, 0.0, 0.0))

    @unittest.skipUnless(USD_AVAILABLE, "Requires usd-core")
    def test_scene_gravity_non_unit_linear_unit(self):
        """Test non-unit linear unit warning and unscaled PhysicsScene gravity."""
        from pxr import Usd, UsdGeom, UsdPhysics

        stage = Usd.Stage.CreateInMemory()
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
        UsdGeom.SetStageMetersPerUnit(stage, 0.01)
        scene = UsdPhysics.Scene.Define(stage, "/physicsScene")
        scene.CreateGravityMagnitudeAttr().Set(12.34)

        body = UsdGeom.Cube.Define(stage, "/Body")
        body_prim = body.GetPrim()
        UsdPhysics.RigidBodyAPI.Apply(body_prim)
        UsdPhysics.CollisionAPI.Apply(body_prim)

        builder = newton.ModelBuilder()
        with self.assertWarnsRegex(UserWarning, "non-unit linear units are not supported"):
            builder.add_usd(stage)

        np.testing.assert_allclose(builder.gravity, (0.0, 0.0, -12.34), atol=1.0e-6)

    @unittest.skipUnless(USD_AVAILABLE, "Requires usd-core")
    def test_scene_gravity_direction(self):
        from pxr import Gf, Usd, UsdGeom, UsdPhysics

        stage = Usd.Stage.CreateInMemory()
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
        scene = UsdPhysics.Scene.Define(stage, "/physicsScene")
        scene.CreateGravityDirectionAttr().Set(Gf.Vec3f(3.0, 4.0, 0.0))
        scene.CreateGravityMagnitudeAttr().Set(10.0)

        body = UsdGeom.Cube.Define(stage, "/Body")
        UsdPhysics.RigidBodyAPI.Apply(body.GetPrim())
        UsdPhysics.CollisionAPI.Apply(body.GetPrim())

        builder = newton.ModelBuilder()
        builder.add_usd(stage)
        np.testing.assert_allclose(builder.gravity, (6.0, 8.0, 0.0), atol=1.0e-6)

        builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
        builder.begin_world()
        builder.add_usd(
            stage,
            xform=wp.transform(wp.vec3(), wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), 0.5 * wp.pi)),
        )
        np.testing.assert_allclose(builder.world_gravity[0], (-8.0, 6.0, 0.0), atol=1.0e-5)

        builder = newton.ModelBuilder()
        builder.add_usd(
            stage,
            xform=wp.transform(wp.vec3(), wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), 0.5 * wp.pi)),
            override_root_xform=True,
        )
        np.testing.assert_allclose(builder.gravity, (6.0, 8.0, 0.0), atol=1.0e-5)

        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.y)
        scene.GetGravityDirectionAttr().Set(Gf.Vec3f(0.0, -1.0, 0.0))
        builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
        builder.add_usd(stage)
        np.testing.assert_allclose(builder.gravity, (0.0, 0.0, -10.0), atol=1.0e-5)

        # Unauthored direction/magnitude sentinels resolve to standard gravity along -up_axis
        stage2 = Usd.Stage.CreateInMemory()
        UsdGeom.SetStageUpAxis(stage2, UsdGeom.Tokens.z)
        UsdGeom.SetStageMetersPerUnit(stage2, 1.0)
        UsdPhysics.Scene.Define(stage2, "/physicsScene")
        body2 = UsdGeom.Cube.Define(stage2, "/Body")
        UsdPhysics.RigidBodyAPI.Apply(body2.GetPrim())
        UsdPhysics.CollisionAPI.Apply(body2.GetPrim())
        builder = newton.ModelBuilder()
        builder.add_usd(stage2)
        np.testing.assert_allclose(builder.gravity, (0.0, 0.0, -9.81), atol=1.0e-5)

    @unittest.skipUnless(USD_AVAILABLE, "Requires usd-core")
    def test_scene_time_steps_per_second_parsing(self):
        """Test that time_steps_per_second is parsed correctly from USD scene."""
        from pxr import Usd, UsdGeom, UsdPhysics

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
        from pxr import Usd, UsdGeom, UsdPhysics

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
        from pxr import Gf, Usd, UsdGeom, UsdPhysics

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


class TestImportSampleAssetsComposition(unittest.TestCase):
    @unittest.skipUnless(USD_AVAILABLE, "Requires usd-core")
    def test_custom_frequency_usd_defaults_when_no_authored_attrs(self):
        """Test that custom frequency counts increment for prims with no authored custom attributes.

        Regression test: when a usd_prim_filter returns True for prims that have no authored custom attributes,
        the frequency count should still increment for each prim, and default values should be applied.
        """
        from pxr import Usd, UsdGeom, UsdPhysics

        # Create a minimal USD stage with physics scene and two custom prims
        # under the imported root, plus a matching sibling outside it.
        stage = Usd.Stage.CreateInMemory()
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
        UsdPhysics.Scene.Define(stage, "/physicsScene")

        # Define two Xform prims that will be matched by our custom filter
        # These prims have NO authored custom attributes
        UsdGeom.Xform.Define(stage, "/World/RobotA/CustomItem0")
        UsdGeom.Xform.Define(stage, "/World/RobotA/CustomItem1")
        UsdGeom.Xform.Define(stage, "/World/RobotB/CustomItem0")

        # Define a prim filter that matches these custom items
        def is_custom_item(prim, context):
            return prim.GetName().startswith("CustomItem")

        builder = newton.ModelBuilder()

        # Register custom frequency with the prim filter
        builder.add_custom_frequency(
            newton.ModelBuilder.CustomFrequency(
                name="item",
                namespace="test",
                usd_prim_filter=is_custom_item,
            )
        )

        # Add a custom attribute with a non-zero default value
        default_value = 42.0
        builder.add_custom_attribute(
            newton.ModelBuilder.CustomAttribute(
                name="item_value",
                frequency="test:item",
                dtype=wp.float32,
                default=default_value,
                namespace="test",
            )
        )

        # Parse one subtree - this should find the 2 prims under RobotA and increment count
        builder.add_usd(stage, root_path="/World/RobotA")

        # Finalize and verify
        model = builder.finalize()

        # Verify the custom frequency count equals the number of prims found
        self.assertEqual(model.get_custom_frequency_count("test:item"), 2)

        # Verify the attribute array has the correct length and default values
        self.assertTrue(hasattr(model, "test"), "Model should have 'test' namespace")
        self.assertTrue(hasattr(model.test, "item_value"), "Model should have 'item_value' attribute")

        item_values = model.test.item_value.numpy()
        self.assertEqual(len(item_values), 2)
        self.assertAlmostEqual(item_values[0], default_value, places=5)
        self.assertAlmostEqual(item_values[1], default_value, places=5)

    @unittest.skipUnless(USD_AVAILABLE, "Requires usd-core")
    def test_custom_frequency_honors_ignore_paths(self):
        """Test that custom frequency parsing skips prims matching ignore_paths.

        Regression test: every other traversal in parse_usd honors ignore_paths,
        but the custom-frequency traversal visited ignored subtrees and registered
        spurious rows for prims that were excluded from the import.
        """
        from pxr import Usd, UsdGeom, UsdPhysics

        stage = Usd.Stage.CreateInMemory()
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
        UsdPhysics.Scene.Define(stage, "/physicsScene")

        # Two matching prims in the kept subtree, two more under an ignored subtree.
        UsdGeom.Xform.Define(stage, "/World/RobotA/CustomItem0")
        UsdGeom.Xform.Define(stage, "/World/RobotA/CustomItem1")
        UsdGeom.Xform.Define(stage, "/World/envs/env_0/CustomItem0")
        UsdGeom.Xform.Define(stage, "/World/envs/env_1/CustomItem0")

        def is_custom_item(prim, context):
            return prim.GetName().startswith("CustomItem")

        builder = newton.ModelBuilder()
        builder.add_custom_frequency(
            newton.ModelBuilder.CustomFrequency(
                name="item",
                namespace="test",
                usd_prim_filter=is_custom_item,
            )
        )
        builder.add_custom_attribute(
            newton.ModelBuilder.CustomAttribute(
                name="item_value",
                frequency="test:item",
                dtype=wp.float32,
                default=42.0,
                namespace="test",
            )
        )

        builder.add_usd(stage, ignore_paths=["/World/envs"])

        model = builder.finalize()

        # Only the two prims outside the ignored subtree may contribute rows.
        self.assertEqual(model.get_custom_frequency_count("test:item"), 2)
        self.assertEqual(len(model.test.item_value.numpy()), 2)

    @unittest.skipUnless(USD_AVAILABLE, "Requires usd-core")
    def test_custom_frequency_instance_proxy_traversal(self):
        """Test that custom frequency parsing traverses instance proxy prims.

        Regression test: prims under instanceable prims should be visited during
        custom frequency USD parsing via TraverseInstanceProxies predicate.
        """
        from pxr import Sdf, Usd, UsdGeom, UsdPhysics

        stage = Usd.Stage.CreateInMemory()
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
        UsdPhysics.Scene.Define(stage, "/physicsScene")

        # Create a prototype prim with a child that will be matched by our filter
        _proto_root = UsdGeom.Xform.Define(stage, "/Prototypes/MyProto")
        proto_child = UsdGeom.Xform.Define(stage, "/Prototypes/MyProto/CustomChild")
        proto_child_prim = proto_child.GetPrim()
        # Author a custom attribute on the prototype child
        # The USD attribute name defaults to "newton:<namespace>:<name>" = "newton:test:child_value"
        proto_child_prim.CreateAttribute("newton:test:child_value", Sdf.ValueTypeNames.Float).Set(99.0)

        # Create two instanceable prims that reference the prototype
        for i in range(2):
            instance = UsdGeom.Xform.Define(stage, f"/World/Instance{i}")
            instance_prim = instance.GetPrim()
            instance_prim.GetReferences().AddInternalReference("/Prototypes/MyProto")
            instance_prim.SetInstanceable(True)

        # Define a filter that matches prims named "CustomChild" (excluding the prototype)
        def is_custom_child(prim, context):
            path = prim.GetPath().pathString
            return prim.GetName() == "CustomChild" and not path.startswith("/Prototypes")

        builder = newton.ModelBuilder()

        # Register custom frequency with the prim filter
        builder.add_custom_frequency(
            newton.ModelBuilder.CustomFrequency(
                name="child",
                namespace="test",
                usd_prim_filter=is_custom_child,
            )
        )

        # Add a custom attribute
        builder.add_custom_attribute(
            newton.ModelBuilder.CustomAttribute(
                name="child_value",
                frequency="test:child",
                dtype=wp.float32,
                default=0.0,
                namespace="test",
            )
        )

        # Parse the USD stage - should find CustomChild under each instance proxy
        builder.add_usd(stage)

        # Finalize and verify
        model = builder.finalize()

        # Should have 2 entries (one per instance proxy)
        self.assertEqual(model.get_custom_frequency_count("test:child"), 2)

        child_values = model.test.child_value.numpy()
        self.assertEqual(len(child_values), 2)
        # Both should have the authored value from the prototype
        self.assertAlmostEqual(child_values[0], 99.0, places=5)
        self.assertAlmostEqual(child_values[1], 99.0, places=5)

    @unittest.skipUnless(USD_AVAILABLE, "Requires usd-core")
    def test_floating_true_creates_free_joint(self):
        """Test that floating=True creates a free joint for the root body."""
        from pxr import Usd, UsdGeom, UsdPhysics

        stage = Usd.Stage.CreateInMemory()
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
        UsdPhysics.Scene.Define(stage, "/physicsScene")

        body = UsdGeom.Cube.Define(stage, "/Body")
        body_prim = body.GetPrim()
        UsdPhysics.RigidBodyAPI.Apply(body_prim)
        UsdPhysics.CollisionAPI.Apply(body_prim)

        builder = newton.ModelBuilder()
        builder.add_usd(stage, floating=True)
        model = builder.finalize()

        self.assertEqual(model.joint_count, 1)
        self.assertEqual(model.joint_type.numpy()[0], newton.JointType.FREE)
        self.assertEqual(model.articulation_count, 1)
        self.assertEqual(model.joint_articulation.numpy().tolist(), [0])

    @unittest.skipUnless(USD_AVAILABLE, "Requires usd-core")
    def test_custom_frequency_wildcard_usd_attribute(self):
        """Test that usd_attribute_name='*' transforms every matching prim."""
        from pxr import Sdf, Usd, UsdGeom, UsdPhysics

        stage = Usd.Stage.CreateInMemory()
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
        UsdPhysics.Scene.Define(stage, "/physicsScene")

        sensor_positions = [(1.0, 2.0, 3.0), (4.0, 5.0, 6.0), (7.0, 8.0, 9.0)]
        for i, pos in enumerate(sensor_positions):
            xform = UsdGeom.Xform.Define(stage, f"/World/Sensor{i}")
            prim = xform.GetPrim()
            # Store the position as a custom (non-newton) attribute on the prim
            attr = prim.CreateAttribute("sensor:position", Sdf.ValueTypeNames.Float3)
            attr.Set(pos)

        # Filter that matches our sensor prims
        def is_sensor_prim(prim, context):
            return prim.GetName().startswith("Sensor")

        builder = newton.ModelBuilder()

        # Register the custom frequency
        builder.add_custom_frequency(
            newton.ModelBuilder.CustomFrequency(
                name="sensor",
                namespace="test",
                usd_prim_filter=is_sensor_prim,
            )
        )

        # Transformer that reads the prim's "sensor:position" attribute and computes
        # the Euclidean distance from the origin
        def compute_distance_from_origin(value, context):
            prim = context["prim"]
            pos = prim.GetAttribute("sensor:position").Get()
            return wp.float32(float(np.sqrt(pos[0] ** 2 + pos[1] ** 2 + pos[2] ** 2)))

        # Register a wildcard custom attribute: no specific USD attribute name,
        # the transformer is called for every prim matching this frequency
        builder.add_custom_attribute(
            newton.ModelBuilder.CustomAttribute(
                name="distance",
                frequency="test:sensor",
                dtype=wp.float32,
                default=0.0,
                namespace="test",
                usd_attribute_name="*",
                usd_value_transformer=compute_distance_from_origin,
            )
        )

        # Also add a second wildcard attribute that extracts the raw position
        def extract_position(value, context):
            prim = context["prim"]
            pos = prim.GetAttribute("sensor:position").Get()
            return wp.vec3(float(pos[0]), float(pos[1]), float(pos[2]))

        builder.add_custom_attribute(
            newton.ModelBuilder.CustomAttribute(
                name="position",
                frequency="test:sensor",
                dtype=wp.vec3,
                default=wp.vec3(0.0, 0.0, 0.0),
                namespace="test",
                usd_attribute_name="*",
                usd_value_transformer=extract_position,
            )
        )

        # Parse the USD stage
        builder.add_usd(stage)

        # Finalize and verify
        model = builder.finalize()

        # Should have found all 3 sensor prims
        self.assertEqual(model.get_custom_frequency_count("test:sensor"), 3)

        # Verify the distance attribute
        distances = model.test.distance.numpy()
        self.assertEqual(len(distances), 3)
        for i, pos in enumerate(sensor_positions):
            expected = np.sqrt(pos[0] ** 2 + pos[1] ** 2 + pos[2] ** 2)
            self.assertAlmostEqual(float(distances[i]), expected, places=4)

        # Verify the position attribute
        positions = model.test.position.numpy()
        self.assertEqual(len(positions), 3)
        for i, pos in enumerate(sensor_positions):
            assert_np_equal(positions[i], np.array(pos, dtype=np.float32), tol=1e-5)

    def test_custom_frequency_wildcard_without_transformer_raises(self):
        """Test that usd_attribute_name='*' without a usd_value_transformer raises ValueError."""
        builder = newton.ModelBuilder()
        builder.add_custom_frequency(
            newton.ModelBuilder.CustomFrequency(
                name="sensor",
                namespace="test",
            )
        )

        with self.assertRaises(ValueError) as ctx:
            builder.add_custom_attribute(
                newton.ModelBuilder.CustomAttribute(
                    name="bad_attr",
                    frequency="test:sensor",
                    dtype=wp.float32,
                    default=0.0,
                    namespace="test",
                    usd_attribute_name="*",
                    # No usd_value_transformer provided
                )
            )
        self.assertIn("usd_attribute_name='*'", str(ctx.exception))
        self.assertIn("usd_value_transformer", str(ctx.exception))

    @unittest.skipUnless(USD_AVAILABLE, "Requires usd-core")
    def test_floating_false_creates_fixed_joint(self):
        """Test that floating=False creates a fixed joint for the root body."""
        from pxr import Usd, UsdGeom, UsdPhysics

        stage = Usd.Stage.CreateInMemory()
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
        UsdPhysics.Scene.Define(stage, "/physicsScene")

        body = UsdGeom.Cube.Define(stage, "/Body")
        body_prim = body.GetPrim()
        UsdPhysics.RigidBodyAPI.Apply(body_prim)
        UsdPhysics.CollisionAPI.Apply(body_prim)

        builder = newton.ModelBuilder()
        builder.add_usd(stage, floating=False)
        model = builder.finalize()

        self.assertEqual(model.joint_count, 1)
        self.assertEqual(model.joint_type.numpy()[0], newton.JointType.FIXED)
        self.assertEqual(model.articulation_count, 1)
        self.assertEqual(model.joint_articulation.numpy().tolist(), [0])

    @unittest.skipUnless(USD_AVAILABLE, "Requires usd-core")
    def test_base_joint_dict_creates_d6_joint(self):
        """Test that base_joint dict with linear and angular axes creates a D6 joint."""
        from pxr import Usd, UsdGeom, UsdPhysics

        stage = Usd.Stage.CreateInMemory()
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
        UsdPhysics.Scene.Define(stage, "/physicsScene")

        body = UsdGeom.Cube.Define(stage, "/Body")
        body_prim = body.GetPrim()
        UsdPhysics.RigidBodyAPI.Apply(body_prim)
        UsdPhysics.CollisionAPI.Apply(body_prim)

        builder = newton.ModelBuilder()
        builder.add_usd(
            stage,
            base_joint={
                "joint_type": newton.JointType.D6,
                "linear_axes": [
                    newton.ModelBuilder.JointDofConfig(axis=[1.0, 0.0, 0.0]),
                    newton.ModelBuilder.JointDofConfig(axis=[0.0, 1.0, 0.0]),
                ],
                "angular_axes": [newton.ModelBuilder.JointDofConfig(axis=[0.0, 0.0, 1.0])],
            },
        )
        model = builder.finalize()

        self.assertEqual(model.joint_count, 1)
        self.assertEqual(model.joint_type.numpy()[0], newton.JointType.D6)
        self.assertEqual(model.articulation_count, 1)
        self.assertEqual(model.joint_articulation.numpy().tolist(), [0])

    @unittest.skipUnless(USD_AVAILABLE, "Requires usd-core")
    def test_base_joint_dict_creates_custom_joint(self):
        """Test that base_joint dict with JointType.REVOLUTE creates a revolute joint with custom axis."""
        from pxr import Usd, UsdGeom, UsdPhysics

        stage = Usd.Stage.CreateInMemory()
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
        UsdPhysics.Scene.Define(stage, "/physicsScene")

        body = UsdGeom.Cube.Define(stage, "/Body")
        body_prim = body.GetPrim()
        UsdPhysics.RigidBodyAPI.Apply(body_prim)
        UsdPhysics.CollisionAPI.Apply(body_prim)

        builder = newton.ModelBuilder()
        builder.add_usd(
            stage,
            base_joint={
                "joint_type": newton.JointType.REVOLUTE,
                "angular_axes": [newton.ModelBuilder.JointDofConfig(axis=(0, 0, 1))],
            },
        )
        model = builder.finalize()

        self.assertEqual(model.joint_count, 1)
        self.assertEqual(model.joint_type.numpy()[0], newton.JointType.REVOLUTE)
        self.assertEqual(model.articulation_count, 1)
        self.assertEqual(model.joint_articulation.numpy().tolist(), [0])

    @unittest.skipUnless(USD_AVAILABLE, "Requires usd-core")
    def test_floating_and_base_joint_mutually_exclusive(self):
        """Test that specifying both floating and base_joint raises an error."""
        from pxr import Usd, UsdGeom, UsdPhysics

        stage = Usd.Stage.CreateInMemory()
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
        UsdPhysics.Scene.Define(stage, "/physicsScene")

        body = UsdGeom.Cube.Define(stage, "/Body")
        body_prim = body.GetPrim()
        UsdPhysics.RigidBodyAPI.Apply(body_prim)
        UsdPhysics.CollisionAPI.Apply(body_prim)

        # Specifying both floating and base_joint should raise an error
        builder = newton.ModelBuilder()
        with self.assertRaises(ValueError) as ctx:
            builder.add_usd(
                stage,
                floating=True,
                base_joint={
                    "joint_type": newton.JointType.D6,
                    "linear_axes": [
                        newton.ModelBuilder.JointDofConfig(axis=[1.0, 0.0, 0.0]),
                        newton.ModelBuilder.JointDofConfig(axis=[0.0, 1.0, 0.0]),
                    ],
                },
            )
        self.assertIn("Cannot specify both", str(ctx.exception))

    @unittest.skipUnless(USD_AVAILABLE, "Requires usd-core")
    def test_custom_frequency_usd_entry_expander_multiple_rows(self):
        """Test that usd_entry_expander can emit multiple rows per matched prim."""
        from pxr import Sdf, Usd, UsdGeom, UsdPhysics

        stage = Usd.Stage.CreateInMemory()
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
        UsdPhysics.Scene.Define(stage, "/physicsScene")

        item_counts = [2, 1]
        for i, count in enumerate(item_counts):
            prim = UsdGeom.Xform.Define(stage, f"/World/Emitter{i}").GetPrim()
            prim.CreateAttribute("test:count", Sdf.ValueTypeNames.Int).Set(count)

        def is_emitter(prim, context):
            return prim.GetPath().pathString.startswith("/World/Emitter")

        def expand_rows(prim, context):
            count = int(prim.GetAttribute("test:count").Get())
            return [{"test:item_value": float(i + 1)} for i in range(count)]

        builder = newton.ModelBuilder()
        builder.add_custom_frequency(
            newton.ModelBuilder.CustomFrequency(
                name="item",
                namespace="test",
                usd_prim_filter=is_emitter,
                usd_entry_expander=expand_rows,
            )
        )
        builder.add_custom_attribute(
            newton.ModelBuilder.CustomAttribute(
                name="item_value",
                frequency="test:item",
                dtype=wp.float32,
                default=0.0,
                namespace="test",
            )
        )

        builder.add_usd(stage)
        model = builder.finalize()

        self.assertEqual(model.get_custom_frequency_count("test:item"), sum(item_counts))
        values = model.test.item_value.numpy()
        self.assertEqual(len(values), 3)
        assert_np_equal(values, np.array([1.0, 2.0, 1.0], dtype=np.float32), tol=1e-6)

    @unittest.skipUnless(USD_AVAILABLE, "Requires usd-core")
    def test_base_joint_respects_import_xform(self):
        """Test that base joints with parent == -1 use the import xform."""
        from pxr import Gf, Usd, UsdGeom, UsdPhysics

        stage = Usd.Stage.CreateInMemory()
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
        UsdPhysics.Scene.Define(stage, "/physicsScene")

        # Create body at position (1, 0, 0)
        body_xform = UsdGeom.Xform.Define(stage, "/FloatingBody")
        body_xform.AddTranslateOp().Set(Gf.Vec3d(1.0, 0.0, 0.0))
        body_prim = body_xform.GetPrim()
        UsdPhysics.RigidBodyAPI.Apply(body_prim)

        # Add collision shape
        cube = UsdGeom.Cube.Define(stage, "/FloatingBody/Collision")
        cube.GetSizeAttr().Set(0.2)
        UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
        UsdPhysics.MassAPI.Apply(cube.GetPrim()).GetMassAttr().Set(1.0)

        # Create import xform: translate + 90° Z rotation
        import_pos = wp.vec3(10.0, 20.0, 30.0)
        import_quat = wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), np.pi / 2)  # 90° Z
        import_xform = wp.transform(import_pos, import_quat)

        # Use base_joint to create a D6 joint
        builder = newton.ModelBuilder()
        builder.add_usd(
            stage,
            xform=import_xform,
            base_joint={
                "joint_type": newton.JointType.D6,
                "linear_axes": [
                    newton.ModelBuilder.JointDofConfig(axis=[1.0, 0.0, 0.0]),
                    newton.ModelBuilder.JointDofConfig(axis=[0.0, 1.0, 0.0]),
                    newton.ModelBuilder.JointDofConfig(axis=[0.0, 0.0, 1.0]),
                ],
            },
        )
        model = builder.finalize()

        # Verify body transform after forward kinematics
        state = model.state()
        newton.eval_fk(model, model.joint_q, model.joint_qd, state)

        body_idx = next((i for i, name in enumerate(model.body_label) if "FloatingBody" in name), None)
        self.assertIsNotNone(body_idx, "Expected a body with 'FloatingBody' in its label")
        body_q = state.body_q.numpy()[body_idx]

        # Expected position: import_pos + rotate_90z(body_pos)
        # = (10, 20, 30) + rotate_90z(1, 0, 0) = (10, 20, 30) + (0, 1, 0) = (10, 21, 30)
        np.testing.assert_allclose(
            body_q[:3],
            [10.0, 21.0, 30.0],
            atol=1e-5,
            err_msg="Body position should include import xform",
        )

        # Expected orientation: 90° Z rotation
        # In xyzw format: [0, 0, sin(45°), cos(45°)] = [0, 0, 0.7071, 0.7071]
        expected_quat = np.array([0, 0, 0.7071068, 0.7071068])
        actual_quat = body_q[3:7]
        quat_match = np.allclose(actual_quat, expected_quat, atol=1e-5) or np.allclose(
            actual_quat, -expected_quat, atol=1e-5
        )
        self.assertTrue(quat_match, f"Body orientation should include import xform. Got {actual_quat}")

    @unittest.skipUnless(USD_AVAILABLE, "Requires usd-core")
    @_expect_jointless_articulation_warning
    def test_parent_body_attaches_to_existing_body(self):
        """Test that parent_body attaches the USD root to an existing body."""
        from pxr import Usd, UsdGeom, UsdPhysics

        # Create first stage: a simple robot arm
        robot_stage = Usd.Stage.CreateInMemory()
        UsdGeom.SetStageUpAxis(robot_stage, UsdGeom.Tokens.z)
        UsdPhysics.Scene.Define(robot_stage, "/physicsScene")

        # Create articulation
        articulation = UsdGeom.Xform.Define(robot_stage, "/Articulation")
        UsdPhysics.ArticulationRootAPI.Apply(articulation.GetPrim())

        # Base link (fixed to world)
        base_link = UsdGeom.Cube.Define(robot_stage, "/Articulation/BaseLink")
        base_link.GetSizeAttr().Set(0.2)
        UsdPhysics.RigidBodyAPI.Apply(base_link.GetPrim())
        UsdPhysics.CollisionAPI.Apply(base_link.GetPrim())

        # End effector
        ee_link = UsdGeom.Cube.Define(robot_stage, "/Articulation/EndEffector")
        ee_link.GetSizeAttr().Set(0.1)
        ee_link.AddTranslateOp().Set((1.0, 0.0, 0.0))
        UsdPhysics.RigidBodyAPI.Apply(ee_link.GetPrim())
        UsdPhysics.CollisionAPI.Apply(ee_link.GetPrim())

        # Revolute joint between base and end effector
        joint = UsdPhysics.RevoluteJoint.Define(robot_stage, "/Articulation/ArmJoint")
        joint.CreateBody0Rel().SetTargets(["/Articulation/BaseLink"])
        joint.CreateBody1Rel().SetTargets(["/Articulation/EndEffector"])
        joint.CreateLocalPos0Attr().Set((0.5, 0.0, 0.0))
        joint.CreateLocalPos1Attr().Set((-0.5, 0.0, 0.0))
        joint.CreateAxisAttr().Set("Z")

        # Create second stage: a gripper
        gripper_stage = Usd.Stage.CreateInMemory()
        UsdGeom.SetStageUpAxis(gripper_stage, UsdGeom.Tokens.z)
        UsdPhysics.Scene.Define(gripper_stage, "/physicsScene")

        gripper_art = UsdGeom.Xform.Define(gripper_stage, "/Gripper")
        UsdPhysics.ArticulationRootAPI.Apply(gripper_art.GetPrim())

        gripper_body = UsdGeom.Cube.Define(gripper_stage, "/Gripper/GripperBase")
        gripper_body.GetSizeAttr().Set(0.05)
        UsdPhysics.RigidBodyAPI.Apply(gripper_body.GetPrim())
        UsdPhysics.CollisionAPI.Apply(gripper_body.GetPrim())

        # First, load the robot
        builder = newton.ModelBuilder()
        usd_result = builder.add_usd(robot_stage, floating=False)

        # Get the end effector body index
        ee_body_idx = usd_result["path_body_map"]["/Articulation/EndEffector"]

        # Remember counts before adding gripper
        robot_body_count = builder.body_count
        robot_joint_count = builder.joint_count

        # Now load the gripper attached to the end effector
        builder.add_usd(gripper_stage, parent_body=ee_body_idx)

        model = builder.finalize()

        # Verify body counts
        self.assertEqual(model.body_count, robot_body_count + 1)  # Robot + gripper

        # Verify the gripper's base joint has the end effector as parent
        gripper_joint_idx = robot_joint_count  # First joint after robot
        self.assertEqual(model.joint_parent.numpy()[gripper_joint_idx], ee_body_idx)

        # Verify all joints belong to the same articulation
        joint_articulations = model.joint_articulation.numpy()
        robot_articulation = joint_articulations[0]
        gripper_articulation = joint_articulations[gripper_joint_idx]
        self.assertEqual(robot_articulation, gripper_articulation)

    @unittest.skipUnless(USD_AVAILABLE, "Requires usd-core")
    @_expect_jointless_articulation_warning
    def test_parent_body_with_base_joint_creates_d6(self):
        """Test that parent_body with base_joint creates a D6 joint to parent."""
        from pxr import Usd, UsdGeom, UsdPhysics

        # Create robot stage
        robot_stage = Usd.Stage.CreateInMemory()
        UsdGeom.SetStageUpAxis(robot_stage, UsdGeom.Tokens.z)
        UsdPhysics.Scene.Define(robot_stage, "/physicsScene")

        robot_art = UsdGeom.Xform.Define(robot_stage, "/Robot")
        UsdPhysics.ArticulationRootAPI.Apply(robot_art.GetPrim())

        robot_body = UsdGeom.Cube.Define(robot_stage, "/Robot/Base")
        robot_body.GetSizeAttr().Set(0.2)
        UsdPhysics.RigidBodyAPI.Apply(robot_body.GetPrim())
        UsdPhysics.CollisionAPI.Apply(robot_body.GetPrim())

        # Create gripper stage
        gripper_stage = Usd.Stage.CreateInMemory()
        UsdGeom.SetStageUpAxis(gripper_stage, UsdGeom.Tokens.z)
        UsdPhysics.Scene.Define(gripper_stage, "/physicsScene")

        gripper_art = UsdGeom.Xform.Define(gripper_stage, "/Gripper")
        UsdPhysics.ArticulationRootAPI.Apply(gripper_art.GetPrim())

        gripper_body = UsdGeom.Cube.Define(gripper_stage, "/Gripper/GripperBase")
        gripper_body.GetSizeAttr().Set(0.05)
        UsdPhysics.RigidBodyAPI.Apply(gripper_body.GetPrim())
        UsdPhysics.CollisionAPI.Apply(gripper_body.GetPrim())

        builder = newton.ModelBuilder()
        builder.add_usd(robot_stage, floating=False)
        robot_body_idx = 0

        # Attach gripper with a D6 joint (rotation around Z)
        builder.add_usd(
            gripper_stage,
            parent_body=robot_body_idx,
            base_joint={
                "joint_type": newton.JointType.D6,
                "angular_axes": [newton.ModelBuilder.JointDofConfig(axis=[0.0, 0.0, 1.0])],
            },
        )

        model = builder.finalize()

        # The second joint should be a D6 connecting to the robot body
        self.assertEqual(model.joint_count, 2)  # Fixed base + D6
        self.assertEqual(model.joint_type.numpy()[1], newton.JointType.D6)
        self.assertEqual(model.joint_parent.numpy()[1], robot_body_idx)

    @unittest.skipUnless(USD_AVAILABLE, "Requires usd-core")
    @_expect_jointless_articulation_warning
    def test_parent_body_creates_joint_to_parent(self):
        """Test that parent_body creates a joint connecting to the parent body."""
        from pxr import Usd, UsdGeom, UsdPhysics

        robot_stage = Usd.Stage.CreateInMemory()
        UsdGeom.SetStageUpAxis(robot_stage, UsdGeom.Tokens.z)
        UsdPhysics.Scene.Define(robot_stage, "/physicsScene")

        robot_art = UsdGeom.Xform.Define(robot_stage, "/Robot")
        UsdPhysics.ArticulationRootAPI.Apply(robot_art.GetPrim())

        base_body = UsdGeom.Cube.Define(robot_stage, "/Robot/Base")
        base_body.GetSizeAttr().Set(0.2)
        UsdPhysics.RigidBodyAPI.Apply(base_body.GetPrim())
        UsdPhysics.CollisionAPI.Apply(base_body.GetPrim())
        UsdPhysics.MassAPI.Apply(base_body.GetPrim()).GetMassAttr().Set(1.0)

        gripper_stage = Usd.Stage.CreateInMemory()
        UsdGeom.SetStageUpAxis(gripper_stage, UsdGeom.Tokens.z)
        UsdPhysics.Scene.Define(gripper_stage, "/physicsScene")

        gripper_art = UsdGeom.Xform.Define(gripper_stage, "/Gripper")
        UsdPhysics.ArticulationRootAPI.Apply(gripper_art.GetPrim())

        gripper_body = UsdGeom.Cube.Define(gripper_stage, "/Gripper/GripperBase")
        gripper_body.GetSizeAttr().Set(0.05)
        UsdPhysics.RigidBodyAPI.Apply(gripper_body.GetPrim())
        UsdPhysics.CollisionAPI.Apply(gripper_body.GetPrim())
        UsdPhysics.MassAPI.Apply(gripper_body.GetPrim()).GetMassAttr().Set(0.2)

        builder = newton.ModelBuilder()
        builder.add_usd(robot_stage, floating=False)

        base_body_idx = 0
        initial_joint_count = builder.joint_count

        builder.add_usd(gripper_stage, parent_body=base_body_idx)

        self.assertEqual(builder.joint_count, initial_joint_count + 1)
        self.assertEqual(builder.joint_parent[initial_joint_count], base_body_idx)

        model = builder.finalize()
        joint_articulation = model.joint_articulation.numpy()
        self.assertEqual(joint_articulation[0], joint_articulation[initial_joint_count])

    @unittest.skipUnless(USD_AVAILABLE, "Requires usd-core")
    @_expect_jointless_articulation_warning
    def test_floating_true_with_parent_body_raises_error(self):
        """Test that floating=True with parent_body raises an error."""
        from pxr import Usd, UsdGeom, UsdPhysics

        # Create robot stage
        robot_stage = Usd.Stage.CreateInMemory()
        UsdGeom.SetStageUpAxis(robot_stage, UsdGeom.Tokens.z)
        UsdPhysics.Scene.Define(robot_stage, "/physicsScene")

        robot_art = UsdGeom.Xform.Define(robot_stage, "/Robot")
        UsdPhysics.ArticulationRootAPI.Apply(robot_art.GetPrim())

        base_body = UsdGeom.Cube.Define(robot_stage, "/Robot/Base")
        base_body.GetSizeAttr().Set(0.2)
        UsdPhysics.RigidBodyAPI.Apply(base_body.GetPrim())
        UsdPhysics.CollisionAPI.Apply(base_body.GetPrim())
        UsdPhysics.MassAPI.Apply(base_body.GetPrim()).GetMassAttr().Set(1.0)

        # Create gripper stage
        gripper_stage = Usd.Stage.CreateInMemory()
        UsdGeom.SetStageUpAxis(gripper_stage, UsdGeom.Tokens.z)
        UsdPhysics.Scene.Define(gripper_stage, "/physicsScene")

        gripper_body = UsdGeom.Cube.Define(gripper_stage, "/GripperBase")
        gripper_body.GetSizeAttr().Set(0.05)
        UsdPhysics.RigidBodyAPI.Apply(gripper_body.GetPrim())
        UsdPhysics.CollisionAPI.Apply(gripper_body.GetPrim())
        UsdPhysics.MassAPI.Apply(gripper_body.GetPrim()).GetMassAttr().Set(0.2)

        builder = newton.ModelBuilder()
        builder.add_usd(robot_stage, floating=False)
        base_body_idx = 0

        # Attempting to use floating=True with parent_body should raise ValueError
        with self.assertRaises(ValueError) as cm:
            builder.add_usd(gripper_stage, parent_body=base_body_idx, floating=True)
        self.assertIn("FREE joint", str(cm.exception))
        self.assertIn("parent_body", str(cm.exception))

    @unittest.skipUnless(USD_AVAILABLE, "Requires usd-core")
    @_expect_jointless_articulation_warning
    def test_floating_false_with_parent_body_succeeds(self):
        """Test that floating=False with parent_body is explicitly allowed."""
        from pxr import Usd, UsdGeom, UsdPhysics

        # Create robot stage
        robot_stage = Usd.Stage.CreateInMemory()
        UsdGeom.SetStageUpAxis(robot_stage, UsdGeom.Tokens.z)
        UsdPhysics.Scene.Define(robot_stage, "/physicsScene")

        robot_art = UsdGeom.Xform.Define(robot_stage, "/Robot")
        UsdPhysics.ArticulationRootAPI.Apply(robot_art.GetPrim())

        base_body = UsdGeom.Cube.Define(robot_stage, "/Robot/Base")
        base_body.GetSizeAttr().Set(0.2)
        UsdPhysics.RigidBodyAPI.Apply(base_body.GetPrim())
        UsdPhysics.CollisionAPI.Apply(base_body.GetPrim())
        UsdPhysics.MassAPI.Apply(base_body.GetPrim()).GetMassAttr().Set(1.0)

        # Create gripper stage
        gripper_stage = Usd.Stage.CreateInMemory()
        UsdGeom.SetStageUpAxis(gripper_stage, UsdGeom.Tokens.z)
        UsdPhysics.Scene.Define(gripper_stage, "/physicsScene")

        gripper_body = UsdGeom.Cube.Define(gripper_stage, "/GripperBase")
        gripper_body.GetSizeAttr().Set(0.05)
        UsdPhysics.RigidBodyAPI.Apply(gripper_body.GetPrim())
        UsdPhysics.CollisionAPI.Apply(gripper_body.GetPrim())
        UsdPhysics.MassAPI.Apply(gripper_body.GetPrim()).GetMassAttr().Set(0.2)

        builder = newton.ModelBuilder()
        builder.add_usd(robot_stage, floating=False)
        base_body_idx = 0

        # Explicitly using floating=False with parent_body should succeed
        builder.add_usd(gripper_stage, parent_body=base_body_idx, floating=False)
        model = builder.finalize()

        self.assertTrue(any("GripperBase" in key for key in builder.body_label))
        self.assertEqual(model.articulation_count, 1)

    @unittest.skipUnless(USD_AVAILABLE, "Requires usd-core")
    @_expect_jointless_articulation_warning
    def test_non_sequential_articulation_attachment(self):
        """Test that attaching to a non-sequential articulation raises an error."""
        from pxr import Usd, UsdGeom, UsdPhysics

        def create_robot_stage():
            stage = Usd.Stage.CreateInMemory()
            UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
            UsdPhysics.Scene.Define(stage, "/physicsScene")
            art = UsdGeom.Xform.Define(stage, "/Robot")
            UsdPhysics.ArticulationRootAPI.Apply(art.GetPrim())
            body = UsdGeom.Cube.Define(stage, "/Robot/Base")
            body.GetSizeAttr().Set(0.2)
            UsdPhysics.RigidBodyAPI.Apply(body.GetPrim())
            UsdPhysics.CollisionAPI.Apply(body.GetPrim())
            UsdPhysics.MassAPI.Apply(body.GetPrim()).GetMassAttr().Set(1.0)
            return stage

        builder = newton.ModelBuilder()
        builder.add_usd(create_robot_stage(), floating=False)
        robot1_body_idx = 0

        # Add more robots to make robot1_body_idx not part of the most recent articulation
        builder.add_usd(create_robot_stage(), floating=False)
        builder.add_usd(create_robot_stage(), floating=False)

        # Attempting to attach to a non-sequential articulation should raise ValueError
        gripper_stage = create_robot_stage()
        with self.assertRaises(ValueError) as cm:
            builder.add_usd(gripper_stage, parent_body=robot1_body_idx)
        self.assertIn("most recent", str(cm.exception))

    @unittest.skipUnless(USD_AVAILABLE, "Requires usd-core")
    def test_parent_body_not_in_articulation_raises_error(self):
        """Test that attaching to a body not in any articulation raises an error."""
        from pxr import Usd, UsdGeom, UsdPhysics

        builder = newton.ModelBuilder()

        # Create a standalone body (not in any articulation)
        standalone_body = builder.add_link(mass=1.0, inertia=wp.mat33(np.eye(3)))
        builder.add_shape_sphere(
            body=standalone_body,
            radius=0.1,
        )

        # Create a simple USD stage with a floating body
        stage = Usd.Stage.CreateInMemory()
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
        UsdPhysics.Scene.Define(stage, "/physicsScene")

        body = UsdGeom.Cube.Define(stage, "/Robot")
        UsdPhysics.RigidBodyAPI.Apply(body.GetPrim())
        UsdPhysics.CollisionAPI.Apply(body.GetPrim())
        UsdPhysics.MassAPI.Apply(body.GetPrim()).GetMassAttr().Set(1.0)

        # Attempting to attach to standalone body should raise ValueError
        with self.assertRaises(ValueError) as cm:
            builder.add_usd(stage, parent_body=standalone_body, floating=False)

        self.assertIn("not part of any articulation", str(cm.exception))

    @unittest.skipUnless(USD_AVAILABLE, "Requires usd-core")
    @_expect_jointless_articulation_warning
    def test_three_level_hierarchical_composition(self):
        """Test attaching multiple levels: arm → gripper → sensor."""
        from pxr import Gf, Usd, UsdGeom, UsdPhysics

        def create_simple_articulation(name, num_links):
            """Helper to create a simple chain articulation."""
            stage = Usd.Stage.CreateInMemory()
            UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
            UsdPhysics.Scene.Define(stage, "/physicsScene")

            # Create articulation root
            root = UsdGeom.Xform.Define(stage, f"/{name}")
            UsdPhysics.ArticulationRootAPI.Apply(root.GetPrim())

            # Create chain of bodies
            for i in range(num_links):
                body = UsdGeom.Xform.Define(stage, f"/{name}/Link{i}")
                UsdPhysics.RigidBodyAPI.Apply(body.GetPrim())
                UsdPhysics.MassAPI.Apply(body.GetPrim()).GetMassAttr().Set(1.0)

                if i > 0:
                    # Create joint connecting to previous link
                    joint = UsdPhysics.RevoluteJoint.Define(stage, f"/{name}/Joint{i}")
                    joint.CreateBody0Rel().SetTargets([f"/{name}/Link{i - 1}"])
                    joint.CreateBody1Rel().SetTargets([f"/{name}/Link{i}"])
                    joint.CreateLocalPos0Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
                    joint.CreateLocalPos1Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
                    joint.CreateLocalRot0Attr().Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
                    joint.CreateLocalRot1Attr().Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
                    joint.CreateAxisAttr().Set("Z")

            return stage

        builder = newton.ModelBuilder()

        # Level 1: Add arm (3 links)
        arm_stage = create_simple_articulation("Arm", 3)
        builder.add_usd(arm_stage, floating=False)
        ee_idx = next((i for i, name in enumerate(builder.body_label) if "Link2" in name), None)
        self.assertIsNotNone(ee_idx, "Expected a body with 'Link2' in its label")

        # Level 2: Attach gripper to end effector (2 links)
        gripper_stage = create_simple_articulation("Gripper", 2)
        builder.add_usd(gripper_stage, parent_body=ee_idx, floating=False)
        finger_idx = next(
            (i for i, name in enumerate(builder.body_label) if "Gripper" in name and "Link1" in name), None
        )
        self.assertIsNotNone(finger_idx, "Expected a Gripper body with 'Link1' in its label")

        # Level 3: Attach sensor to gripper finger (1 link)
        sensor_stage = create_simple_articulation("Sensor", 1)
        builder.add_usd(sensor_stage, parent_body=finger_idx, floating=False)

        model = builder.finalize()

        self.assertEqual(model.articulation_count, 1)
        self.assertEqual(model.joint_count, 6)

    @unittest.skipUnless(USD_AVAILABLE, "Requires usd-core")
    def test_xform_relative_to_parent_body(self):
        """Test that xform is interpreted relative to parent_body when attaching."""
        from pxr import Usd, UsdGeom, UsdPhysics

        def create_simple_body_stage(name):
            """Create a stage with a single rigid body."""
            stage = Usd.Stage.CreateInMemory()
            UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
            UsdPhysics.Scene.Define(stage, "/physicsScene")

            body = UsdGeom.Cube.Define(stage, f"/{name}")
            body.CreateSizeAttr().Set(0.1)
            UsdPhysics.RigidBodyAPI.Apply(body.GetPrim())
            UsdPhysics.MassAPI.Apply(body.GetPrim()).GetMassAttr().Set(1.0)

            return stage

        builder = newton.ModelBuilder()
        parent_stage = create_simple_body_stage("parent")
        builder.add_usd(parent_stage, xform=wp.transform((0.0, 0.0, 2.0), wp.quat_identity()), floating=False)

        parent_body_idx = builder.body_label.index("/parent")

        child_stage = create_simple_body_stage("child")
        builder.add_usd(
            child_stage, parent_body=parent_body_idx, xform=wp.transform((0.0, 0.0, 0.5), wp.quat_identity())
        )

        child_body_idx = builder.body_label.index("/child")

        model = builder.finalize()
        state = model.state()
        newton.eval_fk(model, model.joint_q, model.joint_qd, state)

        body_q = state.body_q.numpy()
        parent_world_pos = body_q[parent_body_idx, :3]
        child_world_pos = body_q[child_body_idx, :3]

        np.testing.assert_allclose(parent_world_pos, [0.0, 0.0, 2.0], atol=1e-5)
        np.testing.assert_allclose(child_world_pos, parent_world_pos + np.array([0.0, 0.0, 0.5]), atol=1e-5)

    @unittest.skipUnless(USD_AVAILABLE, "Requires usd-core")
    def test_many_independent_articulations(self):
        """Test creating many (5) independent articulations and verifying indexing."""
        from pxr import Gf, Usd, UsdGeom, UsdPhysics

        def create_robot_stage():
            """Helper to create a simple 2-link robot."""
            stage = Usd.Stage.CreateInMemory()
            UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
            UsdPhysics.Scene.Define(stage, "/physicsScene")

            root = UsdGeom.Xform.Define(stage, "/Robot")
            UsdPhysics.ArticulationRootAPI.Apply(root.GetPrim())

            base = UsdGeom.Xform.Define(stage, "/Robot/Base")
            UsdPhysics.RigidBodyAPI.Apply(base.GetPrim())
            UsdPhysics.MassAPI.Apply(base.GetPrim()).GetMassAttr().Set(1.0)

            link = UsdGeom.Xform.Define(stage, "/Robot/Link")
            UsdPhysics.RigidBodyAPI.Apply(link.GetPrim())
            UsdPhysics.MassAPI.Apply(link.GetPrim()).GetMassAttr().Set(0.5)

            joint = UsdPhysics.RevoluteJoint.Define(stage, "/Robot/Joint")
            joint.CreateBody0Rel().SetTargets(["/Robot/Base"])
            joint.CreateBody1Rel().SetTargets(["/Robot/Link"])
            joint.CreateLocalPos0Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
            joint.CreateLocalPos1Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
            joint.CreateLocalRot0Attr().Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
            joint.CreateLocalRot1Attr().Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
            joint.CreateAxisAttr().Set("Z")

            return stage

        builder = newton.ModelBuilder()

        # Add 5 independent robots
        for i in range(5):
            builder.add_usd(
                create_robot_stage(),
                xform=wp.transform(wp.vec3(float(i * 2), 0.0, 0.0), wp.quat_identity()),
                floating=False,
            )

        model = builder.finalize()

        self.assertEqual(model.articulation_count, 5)
        self.assertEqual(model.joint_count, 10)

    @unittest.skipUnless(USD_AVAILABLE, "Requires usd-core")
    def test_custom_frequency_usd_filter_and_expander_context_unified(self):
        """Test that usd_prim_filter and usd_entry_expander receive the same context contract."""
        from pxr import Sdf, Usd, UsdGeom, UsdPhysics

        stage = Usd.Stage.CreateInMemory()
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
        UsdPhysics.Scene.Define(stage, "/physicsScene")
        prim = UsdGeom.Xform.Define(stage, "/World/Emitter0").GetPrim()
        prim.CreateAttribute("test:count", Sdf.ValueTypeNames.Int).Set(1)

        captured_filter_contexts = []
        captured_expander_contexts = []

        def is_emitter(prim, context):
            if not prim.GetPath().pathString.startswith("/World/Emitter"):
                return False
            captured_filter_contexts.append(context)
            return True

        def expand_rows(prim, context):
            captured_expander_contexts.append(context)
            count = int(prim.GetAttribute("test:count").Get())
            return [{"test:item_value": float(i + 1)} for i in range(count)]

        builder = newton.ModelBuilder()
        builder.add_custom_frequency(
            newton.ModelBuilder.CustomFrequency(
                name="item",
                namespace="test",
                usd_prim_filter=is_emitter,
                usd_entry_expander=expand_rows,
            )
        )
        builder.add_custom_attribute(
            newton.ModelBuilder.CustomAttribute(
                name="item_value",
                frequency="test:item",
                dtype=wp.float32,
                default=0.0,
                namespace="test",
            )
        )

        import_result = builder.add_usd(stage)
        model = builder.finalize()

        self.assertEqual(model.get_custom_frequency_count("test:item"), 1)
        self.assertEqual(len(captured_filter_contexts), 1)
        self.assertEqual(len(captured_expander_contexts), 1)

        filter_ctx = captured_filter_contexts[0]
        expander_ctx = captured_expander_contexts[0]

        self.assertIs(filter_ctx["builder"], builder)
        self.assertIs(expander_ctx["builder"], builder)
        self.assertIs(filter_ctx["result"], import_result)
        self.assertIs(expander_ctx["result"], import_result)
        self.assertEqual(filter_ctx["prim"].GetPath(), prim.GetPath())
        self.assertEqual(expander_ctx["prim"].GetPath(), prim.GetPath())
        self.assertEqual(set(filter_ctx.keys()), {"prim", "builder", "result"})
        self.assertEqual(set(expander_ctx.keys()), {"prim", "builder", "result"})

    @unittest.skipUnless(USD_AVAILABLE, "Requires usd-core")
    def test_custom_frequency_usd_ordering_producer_before_consumer(self):
        """Test deterministic custom-frequency ordering for producer/consumer dependencies."""
        from pxr import Usd, UsdGeom, UsdPhysics

        stage = Usd.Stage.CreateInMemory()
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
        UsdPhysics.Scene.Define(stage, "/physicsScene")
        UsdGeom.Xform.Define(stage, "/World/Item0")
        UsdGeom.Xform.Define(stage, "/World/Item1")
        UsdGeom.Xform.Define(stage, "/World/Item2")

        def is_item(prim, context):
            return prim.GetPath().pathString.startswith("/World/Item")

        def expand_producer_rows(prim, context):
            return [{"test:producer_value": 1.0}]

        def read_producer_count(_value, context):
            builder = context["builder"]
            producer_attr = builder.custom_attributes["test:producer_value"]
            if not isinstance(producer_attr.values, list):
                return 0
            return int(len(producer_attr.values))

        builder = newton.ModelBuilder()
        builder.add_custom_frequency(
            newton.ModelBuilder.CustomFrequency(
                name="producer",
                namespace="test",
                usd_prim_filter=is_item,
                usd_entry_expander=expand_producer_rows,
            )
        )
        builder.add_custom_frequency(
            newton.ModelBuilder.CustomFrequency(
                name="consumer",
                namespace="test",
                usd_prim_filter=is_item,
            )
        )

        builder.add_custom_attribute(
            newton.ModelBuilder.CustomAttribute(
                name="producer_value",
                frequency="test:producer",
                dtype=wp.float32,
                default=0.0,
                namespace="test",
            )
        )
        builder.add_custom_attribute(
            newton.ModelBuilder.CustomAttribute(
                name="consumer_seen_producer_count",
                frequency="test:consumer",
                dtype=wp.int32,
                default=0,
                namespace="test",
                usd_attribute_name="*",
                usd_value_transformer=read_producer_count,
            )
        )

        builder.add_usd(stage)
        model = builder.finalize()

        self.assertEqual(model.get_custom_frequency_count("test:producer"), 3)
        self.assertEqual(model.get_custom_frequency_count("test:consumer"), 3)
        seen_counts = model.test.consumer_seen_producer_count.numpy()
        assert_np_equal(seen_counts, np.array([1, 2, 3], dtype=np.int32), tol=0)

    @unittest.skipUnless(USD_AVAILABLE, "Requires usd-core")
    def test_base_joint_dict_conflicting_keys_fails(self):
        """Test that base_joint dict with conflicting keys raises ValueError."""
        from pxr import Usd, UsdGeom, UsdPhysics

        stage = Usd.Stage.CreateInMemory()
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
        UsdPhysics.Scene.Define(stage, "/physicsScene")

        body = UsdGeom.Cube.Define(stage, "/Body")
        body_prim = body.GetPrim()
        UsdPhysics.RigidBodyAPI.Apply(body_prim)
        UsdPhysics.CollisionAPI.Apply(body_prim)
        UsdPhysics.MassAPI.Apply(body_prim).GetMassAttr().Set(1.0)

        builder = newton.ModelBuilder()

        with self.assertRaises(ValueError) as ctx:
            builder.add_usd(stage, base_joint={"joint_type": newton.JointType.REVOLUTE, "parent": 5})
        self.assertIn("cannot specify", str(ctx.exception))
        self.assertIn("parent", str(ctx.exception))

        with self.assertRaises(ValueError) as ctx:
            builder.add_usd(stage, base_joint={"joint_type": newton.JointType.REVOLUTE, "child": 3})
        self.assertIn("cannot specify", str(ctx.exception))
        self.assertIn("child", str(ctx.exception))

        with self.assertRaises(ValueError) as ctx:
            builder.add_usd(
                stage,
                base_joint={"joint_type": newton.JointType.REVOLUTE, "parent_xform": wp.transform_identity()},
            )
        self.assertIn("cannot specify", str(ctx.exception))
        self.assertIn("parent_xform", str(ctx.exception))

    @unittest.skipUnless(USD_AVAILABLE, "Requires usd-core")
    def test_base_joint_valid_dict_variations(self):
        """Test that various valid base_joint dict formats work correctly."""
        from pxr import Usd, UsdGeom, UsdPhysics

        def create_stage():
            stage = Usd.Stage.CreateInMemory()
            UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
            UsdPhysics.Scene.Define(stage, "/physicsScene")
            body = UsdGeom.Cube.Define(stage, "/Body")
            body_prim = body.GetPrim()
            UsdPhysics.RigidBodyAPI.Apply(body_prim)
            UsdPhysics.CollisionAPI.Apply(body_prim)
            UsdPhysics.MassAPI.Apply(body_prim).GetMassAttr().Set(1.0)
            return stage

        # Test linear with 'l' prefix
        builder = newton.ModelBuilder()
        builder.add_usd(
            create_stage(),
            base_joint={
                "joint_type": newton.JointType.D6,
                "linear_axes": [
                    newton.ModelBuilder.JointDofConfig(axis=[1.0, 0.0, 0.0]),
                    newton.ModelBuilder.JointDofConfig(axis=[0.0, 1.0, 0.0]),
                    newton.ModelBuilder.JointDofConfig(axis=[0.0, 0.0, 1.0]),
                ],
            },
        )
        model = builder.finalize()
        self.assertEqual(model.joint_type.numpy()[0], newton.JointType.D6)
        self.assertEqual(model.joint_dof_count, 3)  # 3 linear axes

        # Test positional with 'p' prefix
        builder = newton.ModelBuilder()
        builder.add_usd(
            create_stage(),
            base_joint={
                "joint_type": newton.JointType.D6,
                "linear_axes": [
                    newton.ModelBuilder.JointDofConfig(axis=[1.0, 0.0, 0.0]),
                    newton.ModelBuilder.JointDofConfig(axis=[0.0, 1.0, 0.0]),
                    newton.ModelBuilder.JointDofConfig(axis=[0.0, 0.0, 1.0]),
                ],
            },
        )
        model = builder.finalize()
        self.assertEqual(model.joint_type.numpy()[0], newton.JointType.D6)
        self.assertEqual(model.joint_dof_count, 3)  # 3 positional axes

        # Test angular with 'a' prefix
        builder = newton.ModelBuilder()
        builder.add_usd(
            create_stage(),
            base_joint={
                "joint_type": newton.JointType.D6,
                "angular_axes": [
                    newton.ModelBuilder.JointDofConfig(axis=[1.0, 0.0, 0.0]),
                    newton.ModelBuilder.JointDofConfig(axis=[0.0, 1.0, 0.0]),
                    newton.ModelBuilder.JointDofConfig(axis=[0.0, 0.0, 1.0]),
                ],
            },
        )
        model = builder.finalize()
        self.assertEqual(model.joint_type.numpy()[0], newton.JointType.D6)
        self.assertEqual(model.joint_dof_count, 3)  # 3 angular axes

        # Test rotational with 'r' prefix
        builder = newton.ModelBuilder()
        builder.add_usd(
            create_stage(),
            base_joint={
                "joint_type": newton.JointType.D6,
                "angular_axes": [
                    newton.ModelBuilder.JointDofConfig(axis=[1.0, 0.0, 0.0]),
                    newton.ModelBuilder.JointDofConfig(axis=[0.0, 1.0, 0.0]),
                    newton.ModelBuilder.JointDofConfig(axis=[0.0, 0.0, 1.0]),
                ],
            },
        )
        model = builder.finalize()
        self.assertEqual(model.joint_type.numpy()[0], newton.JointType.D6)
        self.assertEqual(model.joint_dof_count, 3)  # 3 rotational axes

    @unittest.skipUnless(USD_AVAILABLE, "Requires usd-core")
    def test_collision_shape_visibility_flags(self):
        """Collision shapes on bodies with visual shapes should not have the
        VISIBLE flag so they are toggleable via the viewer's 'Show Collision'."""
        from pxr import Usd

        usd_content = """#usda 1.0
(
    upAxis = "Z"
)

def PhysicsScene "physicsScene"
{
}

def Xform "BodyWithVisuals" (
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
    }
}

def Xform "BodyWithoutVisuals" (
    prepend apiSchemas = ["PhysicsRigidBodyAPI"]
)
{
    double3 xformOp:translate = (2, 0, 1)
    uniform token[] xformOpOrder = ["xformOp:translate"]

    def Sphere "CollisionSphere" (
        prepend apiSchemas = ["PhysicsCollisionAPI"]
    )
    {
        double radius = 0.5
    }
}
"""
        stage = Usd.Stage.CreateInMemory()
        stage.GetRootLayer().ImportFromString(usd_content)

        # Default: collision shapes on bodies WITH visuals should NOT have VISIBLE
        builder = newton.ModelBuilder()
        result = builder.add_usd(stage)
        path_shape_map = result["path_shape_map"]

        collision_with_visual = path_shape_map["/BodyWithVisuals/CollisionBox"]
        flags_with_visual = builder.shape_flags[collision_with_visual]
        self.assertTrue(flags_with_visual & ShapeFlags.COLLIDE_SHAPES)
        self.assertFalse(flags_with_visual & ShapeFlags.VISIBLE)

        # Collision shapes on bodies WITHOUT visuals should remain hidden by default
        collision_no_visual = path_shape_map["/BodyWithoutVisuals/CollisionSphere"]
        flags_no_visual = builder.shape_flags[collision_no_visual]
        self.assertTrue(flags_no_visual & ShapeFlags.COLLIDE_SHAPES)
        self.assertFalse(flags_no_visual & ShapeFlags.VISIBLE)

        # force_show_colliders=True: collision shapes always get VISIBLE
        builder2 = newton.ModelBuilder()
        result2 = builder2.add_usd(stage, force_show_colliders=True)
        path_shape_map2 = result2["path_shape_map"]

        collision_with_visual2 = path_shape_map2["/BodyWithVisuals/CollisionBox"]
        flags_forced = builder2.shape_flags[collision_with_visual2]
        self.assertTrue(flags_forced & ShapeFlags.COLLIDE_SHAPES)
        self.assertTrue(flags_forced & ShapeFlags.VISIBLE)

        # hide_collision_shapes=True keeps colliders hidden by default.
        builder3 = newton.ModelBuilder()
        result3 = builder3.add_usd(stage, hide_collision_shapes=True)
        path_shape_map3 = result3["path_shape_map"]

        flags_hidden_with_visual = builder3.shape_flags[path_shape_map3["/BodyWithVisuals/CollisionBox"]]
        self.assertTrue(flags_hidden_with_visual & ShapeFlags.COLLIDE_SHAPES)
        self.assertFalse(flags_hidden_with_visual & ShapeFlags.VISIBLE)

        flags_fallback_no_visual = builder3.shape_flags[path_shape_map3["/BodyWithoutVisuals/CollisionSphere"]]
        self.assertTrue(flags_fallback_no_visual & ShapeFlags.COLLIDE_SHAPES)
        self.assertFalse(flags_fallback_no_visual & ShapeFlags.VISIBLE)

        # load_visual_shapes=False: collision shapes remain visible because no
        # visual geometry is loaded for this import.
        builder4 = newton.ModelBuilder()
        result4 = builder4.add_usd(stage, load_visual_shapes=False)
        path_shape_map4 = result4["path_shape_map"]

        collision_no_load = path_shape_map4["/BodyWithVisuals/CollisionBox"]
        flags_no_load = builder4.shape_flags[collision_no_load]
        self.assertTrue(flags_no_load & ShapeFlags.COLLIDE_SHAPES)
        self.assertTrue(flags_no_load & ShapeFlags.VISIBLE)

    @unittest.skipUnless(USD_AVAILABLE, "Requires usd-core")
    def test_collision_only_asset_keeps_colliders_visible(self):
        """Collision-only USD assets must remain visible by default."""
        from pxr import Usd

        usd_content = """#usda 1.0

def PhysicsScene "physicsScene"
{
}

def Xform "Body" (
    prepend apiSchemas = ["PhysicsRigidBodyAPI"]
)
{
    def Sphere "CollisionSphere" (
        prepend apiSchemas = ["PhysicsCollisionAPI"]
    )
    {
        double radius = 0.5
    }
}
"""
        stage = Usd.Stage.CreateInMemory()
        stage.GetRootLayer().ImportFromString(usd_content)

        builder = newton.ModelBuilder()
        result = builder.add_usd(stage)
        collision_flags = builder.shape_flags[result["path_shape_map"]["/Body/CollisionSphere"]]
        self.assertTrue(collision_flags & ShapeFlags.COLLIDE_SHAPES)
        self.assertTrue(collision_flags & ShapeFlags.VISIBLE)

    @staticmethod
    def _create_stage_with_pbr_collision_mesh(color, roughness, metallic, *, add_visual_sphere=False):
        """Create a stage with a rigid body containing a collision mesh with PBR material."""
        from pxr import Sdf, Usd, UsdGeom, UsdPhysics, UsdShade

        stage = Usd.Stage.CreateInMemory()
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
        UsdGeom.SetStageMetersPerUnit(stage, 1.0)
        UsdPhysics.Scene.Define(stage, "/physicsScene")

        body = UsdGeom.Xform.Define(stage, "/Body")
        UsdPhysics.RigidBodyAPI.Apply(body.GetPrim())

        if add_visual_sphere:
            visual_sphere = UsdGeom.Sphere.Define(stage, "/Body/VisualSphere")
            visual_sphere.CreateRadiusAttr().Set(0.1)

        collision_mesh = UsdGeom.Mesh.Define(stage, "/Body/CollisionMesh")
        collision_mesh_prim = collision_mesh.GetPrim()
        UsdPhysics.CollisionAPI.Apply(collision_mesh_prim)
        collision_mesh.CreatePointsAttr().Set(
            [
                (-0.5, 0.0, 0.0),
                (0.5, 0.0, 0.0),
                (0.0, 0.5, 0.0),
                (0.0, 0.0, 0.5),
            ]
        )
        collision_mesh.CreateFaceVertexCountsAttr().Set([3, 3, 3, 3])
        collision_mesh.CreateFaceVertexIndicesAttr().Set([0, 2, 1, 0, 1, 3, 0, 3, 2, 1, 2, 3])

        material = UsdShade.Material.Define(stage, "/Materials/PBR")
        shader = UsdShade.Shader.Define(stage, "/Materials/PBR/PreviewSurface")
        shader.CreateIdAttr("UsdPreviewSurface")
        shader.CreateInput("baseColor", Sdf.ValueTypeNames.Color3f).Set(color)
        shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(roughness)
        shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(metallic)
        material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
        UsdShade.MaterialBindingAPI.Apply(collision_mesh_prim).Bind(material)

        return stage

    @unittest.skipUnless(USD_AVAILABLE, "Requires usd-core")
    def test_visible_collision_mesh_inherits_visual_material_properties(self):
        """Visible fallback collider meshes should carry resolved visual material data."""
        stage = self._create_stage_with_pbr_collision_mesh(color=(0.2, 0.4, 0.6), roughness=0.35, metallic=0.75)

        builder = newton.ModelBuilder()
        result = builder.add_usd(stage, hide_collision_shapes=True)
        collision_shape = result["path_shape_map"]["/Body/CollisionMesh"]

        flags = builder.shape_flags[collision_shape]
        self.assertTrue(flags & ShapeFlags.COLLIDE_SHAPES)
        self.assertTrue(flags & ShapeFlags.VISIBLE)

        mesh = builder.shape_source[collision_shape]
        self.assertIsNotNone(mesh)
        np.testing.assert_allclose(
            np.array(mesh.color),
            np.array(newton.utils.color_linear_to_srgb((0.2, 0.4, 0.6))),
            atol=1e-6,
            rtol=1e-6,
        )
        self.assertAlmostEqual(mesh.roughness, 0.35, places=6)
        self.assertAlmostEqual(mesh.metallic, 0.75, places=6)

    @unittest.skipUnless(USD_AVAILABLE, "Requires usd-core")
    def test_visible_collision_mesh_texture_does_not_change_body_mass(self):
        """Render-only UV loading must not perturb collider mass or inertia."""
        stage = self._create_stage_with_pbr_collision_mesh(color=(0.2, 0.4, 0.6), roughness=0.35, metallic=0.75)

        base_vertices = np.array(
            [
                (-0.5, 0.0, 0.0),
                (0.5, 0.0, 0.0),
                (0.0, 0.5, 0.0),
                (0.0, 0.0, 0.5),
            ],
            dtype=np.float32,
        )
        indices = np.array([0, 2, 1, 0, 1, 3, 0, 3, 2, 1, 2, 3], dtype=np.int32)
        physics_mesh = newton.Mesh(base_vertices, indices)
        render_mesh = newton.Mesh(base_vertices * 4.0, indices)
        render_mesh._uvs = np.zeros((render_mesh.vertices.shape[0], 2), dtype=np.float32)

        def _mock_get_mesh(_prim, *, load_uvs=False, load_normals=False):
            del load_normals
            return render_mesh if load_uvs else physics_mesh

        with (
            mock.patch(
                "newton._src.utils.import_usd.usd.resolve_material_properties_for_prim",
                return_value={
                    "color": None,
                    "roughness": 0.35,
                    "metallic": 0.75,
                    "texture": "dummy.png",
                },
            ),
            mock.patch(
                "newton._src.utils.import_usd.usd.get_mesh",
                side_effect=_mock_get_mesh,
            ),
        ):
            builder = newton.ModelBuilder()
            result = builder.add_usd(stage, hide_collision_shapes=True)

        body_idx = result["path_body_map"]["/Body"]
        collision_shape = result["path_shape_map"]["/Body/CollisionMesh"]
        expected_density = builder.default_shape_cfg.density

        self.assertAlmostEqual(builder.body_mass[body_idx], physics_mesh.mass * expected_density, places=6)
        self.assertNotAlmostEqual(builder.body_mass[body_idx], render_mesh.mass * expected_density, places=3)

        mesh = builder.shape_source[collision_shape]
        self.assertIsNotNone(mesh)
        self.assertEqual(mesh.texture, "dummy.png")
        self.assertIsNotNone(mesh.uvs)
        np.testing.assert_allclose(mesh.vertices, render_mesh.vertices, atol=1e-6, rtol=1e-6)
        self.assertAlmostEqual(mesh.mass, physics_mesh.mass, places=6)

    @unittest.skipUnless(USD_AVAILABLE, "Requires usd-core")
    def test_hide_collision_shapes_overrides_visual_material(self):
        """hide_collision_shapes=True hides colliders even when they have visual material data."""
        stage = self._create_stage_with_pbr_collision_mesh(
            color=(0.9, 0.1, 0.2), roughness=0.55, metallic=0.25, add_visual_sphere=True
        )

        builder = newton.ModelBuilder()
        result = builder.add_usd(stage, hide_collision_shapes=True)
        path_shape_map = result["path_shape_map"]

        self.assertIn("/Body/VisualSphere", path_shape_map)
        visual_shape = path_shape_map["/Body/VisualSphere"]
        self.assertFalse(builder.shape_flags[visual_shape] & ShapeFlags.COLLIDE_SHAPES)

        collision_shape = path_shape_map["/Body/CollisionMesh"]
        flags = builder.shape_flags[collision_shape]
        self.assertTrue(flags & ShapeFlags.COLLIDE_SHAPES)
        self.assertFalse(flags & ShapeFlags.VISIBLE)

    @unittest.skipUnless(USD_AVAILABLE, "Requires usd-core")
    def test_hide_collision_shapes_fallback_with_material(self):
        """Colliders with material stay visible when the body has no other visual shapes."""
        stage = self._create_stage_with_pbr_collision_mesh(
            color=(0.2, 0.4, 0.6), roughness=0.35, metallic=0.75, add_visual_sphere=False
        )

        builder = newton.ModelBuilder()
        result = builder.add_usd(stage, hide_collision_shapes=True)
        collision_shape = result["path_shape_map"]["/Body/CollisionMesh"]

        flags = builder.shape_flags[collision_shape]
        self.assertTrue(flags & ShapeFlags.COLLIDE_SHAPES)
        self.assertTrue(flags & ShapeFlags.VISIBLE)

    @unittest.skipUnless(USD_AVAILABLE, "Requires usd-core")
    def test_invisible_collision_shape_is_hidden(self):
        """Effective USD invisibility clears VISIBLE on colliders while preserving collision."""
        from pxr import Usd

        usd_content = """#usda 1.0

def PhysicsScene "physicsScene"
{
}

def Xform "Body" (
    prepend apiSchemas = ["PhysicsRigidBodyAPI"]
)
{
    double3 xformOp:translate = (0, 0, 1)
    uniform token[] xformOpOrder = ["xformOp:translate"]

    def Cube "CollisionBox" (
        prepend apiSchemas = ["PhysicsCollisionAPI"]
    )
    {
        token visibility = "invisible"
        double size = 1.0
    }
}
"""
        stage = Usd.Stage.CreateInMemory()
        stage.GetRootLayer().ImportFromString(usd_content)

        builder = newton.ModelBuilder()
        result = builder.add_usd(stage)
        collision_shape = result["path_shape_map"]["/Body/CollisionBox"]

        flags = builder.shape_flags[collision_shape]
        self.assertTrue(flags & ShapeFlags.COLLIDE_SHAPES)
        self.assertFalse(flags & ShapeFlags.VISIBLE)

    @unittest.skipUnless(USD_AVAILABLE, "Requires usd-core")
    def test_inherited_invisible_body_hides_collider_and_visual(self):
        """Parent invisibility propagates to child colliders and visual-only geometry."""
        from pxr import Usd

        usd_content = """#usda 1.0

def PhysicsScene "physicsScene"
{
}

def Xform "Body" (
    prepend apiSchemas = ["PhysicsRigidBodyAPI"]
)
{
    token visibility = "invisible"
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
    }
}
"""
        stage = Usd.Stage.CreateInMemory()
        stage.GetRootLayer().ImportFromString(usd_content)

        builder = newton.ModelBuilder()
        result = builder.add_usd(stage)
        path_shape_map = result["path_shape_map"]

        collision_shape = path_shape_map["/Body/CollisionBox"]
        flags = builder.shape_flags[collision_shape]
        self.assertTrue(flags & ShapeFlags.COLLIDE_SHAPES)
        self.assertFalse(flags & ShapeFlags.VISIBLE)
        self.assertIn("/Body/VisualSphere", path_shape_map)
        visual_shape = path_shape_map["/Body/VisualSphere"]
        self.assertFalse(builder.shape_flags[visual_shape] & ShapeFlags.COLLIDE_SHAPES)
        self.assertFalse(builder.shape_flags[visual_shape] & ShapeFlags.VISIBLE)

    @unittest.skipUnless(USD_AVAILABLE, "Requires usd-core")
    def test_invisible_visual_sibling_does_not_suppress_collider_visibility(self):
        """An invisible visual shape must not suppress fallback-visible colliders."""
        from pxr import Usd

        usd_content = """#usda 1.0

def PhysicsScene "physicsScene"
{
}

def Xform "Body" (
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

    def Sphere "InvisibleVisual"
    {
        token visibility = "invisible"
        double radius = 0.3
    }
}
"""
        stage = Usd.Stage.CreateInMemory()
        stage.GetRootLayer().ImportFromString(usd_content)

        builder = newton.ModelBuilder()
        result = builder.add_usd(stage)
        path_shape_map = result["path_shape_map"]

        # The invisible visual should still be imported (as hidden).
        self.assertIn("/Body/InvisibleVisual", path_shape_map)
        vis_shape = path_shape_map["/Body/InvisibleVisual"]
        self.assertFalse(builder.shape_flags[vis_shape] & ShapeFlags.VISIBLE)

        # The collider remains visible because the import has no effectively
        # visible visual shapes.
        collision_shape = path_shape_map["/Body/CollisionBox"]
        flags = builder.shape_flags[collision_shape]
        self.assertTrue(flags & ShapeFlags.COLLIDE_SHAPES)
        self.assertTrue(flags & ShapeFlags.VISIBLE)

    @unittest.skipUnless(USD_AVAILABLE, "Requires usd-core")
    def test_primitive_collider_with_roughness_only_material_stays_hidden(self):
        """Primitive (non-mesh) colliders must not become visible from roughness-only materials.

        When a body already has visual shapes, ``show_collider_by_policy`` is
        ``False``. Only ``collider_has_visual_material`` can promote a collider
        to visible, and that promotion is restricted to mesh colliders only.
        """
        from pxr import Sdf, Usd, UsdGeom, UsdPhysics, UsdShade

        usd_content = """#usda 1.0

def PhysicsScene "physicsScene"
{
}

def Xform "Body" (
    prepend apiSchemas = ["PhysicsRigidBodyAPI"]
)
{
    double3 xformOp:translate = (0, 0, 1)
    uniform token[] xformOpOrder = ["xformOp:translate"]

    def Sphere "VisualSphere"
    {
        double radius = 0.3
    }
}
"""
        stage = Usd.Stage.CreateInMemory()
        stage.GetRootLayer().ImportFromString(usd_content)

        # Add a collision box with a roughness-only PBR material.
        box_prim = UsdGeom.Cube.Define(stage, "/Body/CollisionBox").GetPrim()
        UsdPhysics.CollisionAPI.Apply(box_prim)
        material = UsdShade.Material.Define(stage, "/Materials/RoughnessOnly")
        shader = UsdShade.Shader.Define(stage, "/Materials/RoughnessOnly/PreviewSurface")
        shader.CreateIdAttr("UsdPreviewSurface")
        shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.8)
        material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
        UsdShade.MaterialBindingAPI.Apply(box_prim).Bind(material)

        builder = newton.ModelBuilder()
        # Default hide_collision_shapes=False so the MeshShape guard
        # is the deciding factor, not the unconditional hide override.
        result = builder.add_usd(stage)
        path_shape_map = result["path_shape_map"]

        collision_shape = path_shape_map["/Body/CollisionBox"]
        flags = builder.shape_flags[collision_shape]
        self.assertTrue(flags & ShapeFlags.COLLIDE_SHAPES)
        # Primitive colliders should NOT be promoted to visible just because
        # they have roughness metadata — only mesh colliders qualify.
        self.assertFalse(flags & ShapeFlags.VISIBLE)


class TestImportUsdMimicJoint(unittest.TestCase):
    """Tests for PhysxMimicJointAPI parsing during USD import."""

    @unittest.skipUnless(USD_AVAILABLE, "Requires usd-core")
    def test_physx_mimic_joint_basic(self):
        """PhysxMimicJointAPI on a revolute joint creates a mimic constraint."""
        from pxr import Gf, Usd, UsdGeom, UsdPhysics

        stage = Usd.Stage.CreateInMemory()
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
        UsdPhysics.SetStageKilogramsPerUnit(stage, 1.0)
        UsdGeom.SetStageMetersPerUnit(stage, 1.0)

        root = stage.DefinePrim("/Root", "Xform")
        stage.SetDefaultPrim(root)

        art = stage.DefinePrim("/Root/Robot", "Xform")
        UsdPhysics.ArticulationRootAPI.Apply(art)

        # base body
        base = stage.DefinePrim("/Root/Robot/base", "Cube")
        UsdPhysics.RigidBodyAPI.Apply(base)
        UsdPhysics.MassAPI.Apply(base).CreateMassAttr(1.0)

        # link1
        link1 = stage.DefinePrim("/Root/Robot/link1", "Cube")
        UsdPhysics.RigidBodyAPI.Apply(link1)
        UsdPhysics.MassAPI.Apply(link1).CreateMassAttr(1.0)

        # link2
        link2 = stage.DefinePrim("/Root/Robot/link2", "Cube")
        UsdPhysics.RigidBodyAPI.Apply(link2)
        UsdPhysics.MassAPI.Apply(link2).CreateMassAttr(1.0)

        # leader joint: base -> link1
        leader = UsdPhysics.RevoluteJoint.Define(stage, "/Root/Robot/Joints/leader")
        leader.CreateAxisAttr("Z")
        leader.CreateBody0Rel().SetTargets(["/Root/Robot/base"])
        leader.CreateBody1Rel().SetTargets(["/Root/Robot/link1"])
        leader.CreateLocalPos0Attr().Set(Gf.Vec3f(0, 0, 0.5))
        leader.CreateLocalPos1Attr().Set(Gf.Vec3f(0, 0, -0.5))
        leader.CreateLocalRot0Attr().Set(Gf.Quatf(1, 0, 0, 0))
        leader.CreateLocalRot1Attr().Set(Gf.Quatf(1, 0, 0, 0))

        # follower joint: base -> link2
        follower = UsdPhysics.RevoluteJoint.Define(stage, "/Root/Robot/Joints/follower")
        follower.CreateAxisAttr("Z")
        follower.CreateBody0Rel().SetTargets(["/Root/Robot/base"])
        follower.CreateBody1Rel().SetTargets(["/Root/Robot/link2"])
        follower.CreateLocalPos0Attr().Set(Gf.Vec3f(0.5, 0, 0.5))
        follower.CreateLocalPos1Attr().Set(Gf.Vec3f(0, 0, -0.5))
        follower.CreateLocalRot0Attr().Set(Gf.Quatf(1, 0, 0, 0))
        follower.CreateLocalRot1Attr().Set(Gf.Quatf(1, 0, 0, 0))

        # Apply PhysxMimicJointAPI:rotZ to follower via metadata
        # (PhysxMimicJointAPI is not in usd-core, so use raw metadata)
        follower_prim = follower.GetPrim()
        from pxr import Sdf

        follower_prim.SetMetadata("apiSchemas", Sdf.TokenListOp.Create(prependedItems=["PhysxMimicJointAPI:rotZ"]))
        follower_prim.CreateRelationship("physxMimicJoint:rotZ:referenceJoint").SetTargets(
            ["/Root/Robot/Joints/leader"]
        )
        follower_prim.CreateAttribute("physxMimicJoint:rotZ:gearing", Sdf.ValueTypeNames.Float).Set(-2.0)

        builder = newton.ModelBuilder()
        builder.add_usd(stage)

        self.assertEqual(len(builder.constraint_mimic_joint0), 1)

        model = builder.finalize()
        self.assertEqual(model.constraint_mimic_count, 1)

        joint0 = model.constraint_mimic_joint0.numpy()[0]
        joint1 = model.constraint_mimic_joint1.numpy()[0]
        coef0 = model.constraint_mimic_coef0.numpy()[0]
        coef1 = model.constraint_mimic_coef1.numpy()[0]

        follower_idx = model.joint_label.index("/Root/Robot/Joints/follower")
        leader_idx = model.joint_label.index("/Root/Robot/Joints/leader")

        self.assertEqual(joint0, follower_idx)
        self.assertEqual(joint1, leader_idx)
        # PhysX: jointPos + gearing * refPos + offset = 0
        # Newton: joint0 = coef0 + coef1 * joint1
        # So coef1 = -gearing = -(-2.0) = 2.0, coef0 = -offset = 0.0
        self.assertAlmostEqual(coef0, 0.0, places=5)
        self.assertAlmostEqual(coef1, 2.0, places=5)

    @unittest.skipUnless(USD_AVAILABLE, "Requires usd-core")
    def test_physx_mimic_joint_no_api_no_constraint(self):
        """Joints without PhysxMimicJointAPI produce no mimic constraints."""
        from pxr import Gf, Usd, UsdGeom, UsdPhysics

        stage = Usd.Stage.CreateInMemory()
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
        UsdPhysics.SetStageKilogramsPerUnit(stage, 1.0)
        UsdGeom.SetStageMetersPerUnit(stage, 1.0)

        root = stage.DefinePrim("/Root", "Xform")
        stage.SetDefaultPrim(root)

        art = stage.DefinePrim("/Root/Robot", "Xform")
        UsdPhysics.ArticulationRootAPI.Apply(art)

        base = stage.DefinePrim("/Root/Robot/base", "Cube")
        UsdPhysics.RigidBodyAPI.Apply(base)
        UsdPhysics.MassAPI.Apply(base).CreateMassAttr(1.0)

        link1 = stage.DefinePrim("/Root/Robot/link1", "Cube")
        UsdPhysics.RigidBodyAPI.Apply(link1)
        UsdPhysics.MassAPI.Apply(link1).CreateMassAttr(1.0)

        joint = UsdPhysics.RevoluteJoint.Define(stage, "/Root/Robot/Joints/joint1")
        joint.CreateAxisAttr("Z")
        joint.CreateBody0Rel().SetTargets(["/Root/Robot/base"])
        joint.CreateBody1Rel().SetTargets(["/Root/Robot/link1"])
        joint.CreateLocalPos0Attr().Set(Gf.Vec3f(0, 0, 0.5))
        joint.CreateLocalPos1Attr().Set(Gf.Vec3f(0, 0, -0.5))
        joint.CreateLocalRot0Attr().Set(Gf.Quatf(1, 0, 0, 0))
        joint.CreateLocalRot1Attr().Set(Gf.Quatf(1, 0, 0, 0))

        builder = newton.ModelBuilder()
        builder.add_usd(stage)

        self.assertEqual(len(builder.constraint_mimic_joint0), 0)


class TestHasAppliedApiSchema(unittest.TestCase):
    """Test the has_applied_api_schema helper in newton.usd.utils."""

    @unittest.skipUnless(USD_AVAILABLE, "Requires usd-core")
    def test_unregistered_schema_via_metadata(self):
        from pxr import Usd

        stage = Usd.Stage.CreateInMemory()
        stage.GetRootLayer().ImportFromString(
            """#usda 1.0
def Sphere "WithSiteAPI" (
    prepend apiSchemas = ["MjcSiteAPI"]
)
{
    double radius = 0.1
}

def Sphere "WithoutSiteAPI"
{
    double radius = 0.1
}
"""
        )

        prim_with = stage.GetPrimAtPath("/WithSiteAPI")
        prim_without = stage.GetPrimAtPath("/WithoutSiteAPI")

        self.assertTrue(usd.has_applied_api_schema(prim_with, "MjcSiteAPI"))
        self.assertFalse(usd.has_applied_api_schema(prim_without, "MjcSiteAPI"))
        self.assertFalse(usd.has_applied_api_schema(prim_with, "NonExistentAPI"))

    @unittest.skipUnless(USD_AVAILABLE, "Requires usd-core")
    def test_registered_schema_via_has_api(self):
        from pxr import Usd, UsdPhysics

        stage = Usd.Stage.CreateInMemory()
        prim = stage.DefinePrim("/Body", "Xform")
        UsdPhysics.RigidBodyAPI.Apply(prim)

        self.assertTrue(usd.has_applied_api_schema(prim, "PhysicsRigidBodyAPI"))
        self.assertFalse(usd.has_applied_api_schema(prim, "PhysicsMassAPI"))

    @unittest.skipUnless(USD_AVAILABLE, "Requires usd-core")
    def test_appended_and_explicit_items(self):
        from pxr import Usd

        stage = Usd.Stage.CreateInMemory()
        stage.GetRootLayer().ImportFromString(
            """#usda 1.0
def Sphere "AppendedSchema" (
    append apiSchemas = ["MjcSiteAPI"]
)
{
    double radius = 0.1
}
"""
        )

        prim = stage.GetPrimAtPath("/AppendedSchema")
        self.assertTrue(usd.has_applied_api_schema(prim, "MjcSiteAPI"))


class TestOverrideRootXform(unittest.TestCase):
    """Tests for override_root_xform parameter in the USD importer."""

    @staticmethod
    def _make_stage_with_root_offset():
        """Create a USD stage with an articulation under a translated ancestor."""
        from pxr import Gf, Usd, UsdGeom, UsdPhysics

        stage = Usd.Stage.CreateInMemory()
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
        UsdPhysics.Scene.Define(stage, "/physicsScene")

        env = UsdGeom.Xform.Define(stage, "/World/env")
        env.AddTranslateOp().Set(Gf.Vec3d(100.0, 200.0, 0.0))

        root = UsdGeom.Xform.Define(stage, "/World/env/Robot")
        UsdPhysics.ArticulationRootAPI.Apply(root.GetPrim())

        base = UsdGeom.Xform.Define(stage, "/World/env/Robot/Base")
        UsdPhysics.RigidBodyAPI.Apply(base.GetPrim())
        UsdPhysics.MassAPI.Apply(base.GetPrim()).GetMassAttr().Set(1.0)

        link = UsdGeom.Xform.Define(stage, "/World/env/Robot/Link")
        link.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, 1.0))
        UsdPhysics.RigidBodyAPI.Apply(link.GetPrim())
        UsdPhysics.MassAPI.Apply(link.GetPrim()).GetMassAttr().Set(0.5)

        joint = UsdPhysics.RevoluteJoint.Define(stage, "/World/env/Robot/Joint")
        joint.CreateBody0Rel().SetTargets(["/World/env/Robot/Base"])
        joint.CreateBody1Rel().SetTargets(["/World/env/Robot/Link"])
        joint.CreateLocalPos0Attr().Set(Gf.Vec3f(0.0, 0.0, 1.0))
        joint.CreateLocalPos1Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
        joint.CreateLocalRot0Attr().Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
        joint.CreateLocalRot1Attr().Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
        joint.CreateAxisAttr().Set("Z")

        return stage

    @unittest.skipUnless(USD_AVAILABLE, "Requires usd-core")
    def test_default_xform_is_relative(self):
        """With override_root_xform=False (default), xform composes with ancestor transforms."""
        stage = self._make_stage_with_root_offset()

        builder = newton.ModelBuilder()
        builder.add_usd(stage, xform=wp.transform((5.0, 0.0, 0.0), wp.quat_identity()), floating=False)

        model = builder.finalize()
        state = model.state()
        newton.eval_fk(model, model.joint_q, model.joint_qd, state)

        body_q = state.body_q.numpy()
        base_idx = builder.body_label.index("/World/env/Robot/Base")
        # xform (5,0,0) composed with ancestor (100,200,0) => (105, 200, 0)
        np.testing.assert_allclose(body_q[base_idx, :3], [105.0, 200.0, 0.0], atol=1e-4)

    @unittest.skipUnless(USD_AVAILABLE, "Requires usd-core")
    def test_override_places_at_xform(self):
        """With override_root_xform=True, root body is placed at exactly xform."""
        stage = self._make_stage_with_root_offset()

        builder = newton.ModelBuilder()
        builder.add_usd(
            stage,
            xform=wp.transform((5.0, 0.0, 0.0), wp.quat_identity()),
            floating=False,
            override_root_xform=True,
        )

        model = builder.finalize()
        state = model.state()
        newton.eval_fk(model, model.joint_q, model.joint_qd, state)

        body_q = state.body_q.numpy()
        base_idx = builder.body_label.index("/World/env/Robot/Base")
        np.testing.assert_allclose(body_q[base_idx, :3], [5.0, 0.0, 0.0], atol=1e-4)

    @unittest.skipUnless(USD_AVAILABLE, "Requires usd-core")
    def test_override_preserves_child_offset(self):
        """With override_root_xform=True, child body keeps its relative offset from root."""
        stage = self._make_stage_with_root_offset()

        builder = newton.ModelBuilder()
        builder.add_usd(
            stage,
            xform=wp.transform((5.0, 0.0, 0.0), wp.quat_identity()),
            floating=False,
            override_root_xform=True,
        )

        model = builder.finalize()
        state = model.state()
        newton.eval_fk(model, model.joint_q, model.joint_qd, state)

        body_q = state.body_q.numpy()
        link_idx = builder.body_label.index("/World/env/Robot/Link")
        np.testing.assert_allclose(body_q[link_idx, :3], [5.0, 0.0, 1.0], atol=1e-4)

    @unittest.skipUnless(USD_AVAILABLE, "Requires usd-core")
    def test_override_with_rotation(self):
        """override_root_xform=True with a non-identity rotation correctly rotates the articulation."""
        stage = self._make_stage_with_root_offset()
        angle = np.pi / 2
        quat = wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), angle)

        builder = newton.ModelBuilder()
        builder.add_usd(
            stage,
            xform=wp.transform((5.0, 0.0, 0.0), quat),
            floating=False,
            override_root_xform=True,
        )

        model = builder.finalize()
        state = model.state()
        newton.eval_fk(model, model.joint_q, model.joint_qd, state)

        body_q = state.body_q.numpy()
        base_idx = builder.body_label.index("/World/env/Robot/Base")
        link_idx = builder.body_label.index("/World/env/Robot/Link")

        np.testing.assert_allclose(body_q[base_idx, :3], [5.0, 0.0, 0.0], atol=1e-4)
        np.testing.assert_allclose(body_q[base_idx, 3:], [*quat], atol=1e-4)
        # Link is at (0,0,1) relative to root; Z-rotation doesn't affect Z offset
        np.testing.assert_allclose(body_q[link_idx, :3], [5.0, 0.0, 1.0], atol=1e-4)

    @unittest.skipUnless(USD_AVAILABLE, "Requires usd-core")
    def test_override_cloning(self):
        """Cloning the same articulation at multiple positions with override_root_xform=True."""
        stage = self._make_stage_with_root_offset()

        builder = newton.ModelBuilder()
        clone_positions = [(0.0, 0.0, 0.0), (2.0, 0.0, 0.0), (4.0, 0.0, 0.0)]
        for pos in clone_positions:
            builder.add_usd(
                stage, xform=wp.transform(pos, wp.quat_identity()), floating=False, override_root_xform=True
            )

        model = builder.finalize()
        state = model.state()
        newton.eval_fk(model, model.joint_q, model.joint_qd, state)

        body_q = state.body_q.numpy()
        base_indices = [j for j, lbl in enumerate(builder.body_label) if lbl.endswith("/Robot/Base")]
        for i, expected_pos in enumerate(clone_positions):
            np.testing.assert_allclose(
                body_q[base_indices[i], :3],
                list(expected_pos),
                atol=1e-4,
                err_msg=f"Clone {i} not at expected position",
            )

    @staticmethod
    def _make_two_articulation_stage():
        """Create a USD stage with two articulations at different positions."""
        from pxr import Gf, Usd, UsdGeom, UsdPhysics

        stage = Usd.Stage.CreateInMemory()
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
        UsdPhysics.Scene.Define(stage, "/physicsScene")

        for name, offset in [("RobotA", (10.0, 0.0, 0.0)), ("RobotB", (0.0, 20.0, 0.0))]:
            root_path = f"/World/{name}"
            root = UsdGeom.Xform.Define(stage, root_path)
            root.AddTranslateOp().Set(Gf.Vec3d(*offset))
            UsdPhysics.ArticulationRootAPI.Apply(root.GetPrim())

            base = UsdGeom.Xform.Define(stage, f"{root_path}/Base")
            UsdPhysics.RigidBodyAPI.Apply(base.GetPrim())
            UsdPhysics.MassAPI.Apply(base.GetPrim()).GetMassAttr().Set(1.0)

            link = UsdGeom.Xform.Define(stage, f"{root_path}/Link")
            link.AddTranslateOp().Set(Gf.Vec3d(offset[0], offset[1], offset[2] + 1.0))
            UsdPhysics.RigidBodyAPI.Apply(link.GetPrim())
            UsdPhysics.MassAPI.Apply(link.GetPrim()).GetMassAttr().Set(0.5)

            joint = UsdPhysics.RevoluteJoint.Define(stage, f"{root_path}/Joint")
            joint.CreateBody0Rel().SetTargets([f"{root_path}/Base"])
            joint.CreateBody1Rel().SetTargets([f"{root_path}/Link"])
            joint.CreateLocalPos0Attr().Set(Gf.Vec3f(0.0, 0.0, 1.0))
            joint.CreateLocalPos1Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
            joint.CreateLocalRot0Attr().Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
            joint.CreateLocalRot1Attr().Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
            joint.CreateAxisAttr().Set("Z")

        return stage

    @unittest.skipUnless(USD_AVAILABLE, "Requires usd-core")
    def test_multiple_articulations_default_keeps_relative(self):
        """Without override, multiple articulations keep their relative positions shifted by xform."""
        stage = self._make_two_articulation_stage()

        shift = (1.0, 2.0, 3.0)
        builder = newton.ModelBuilder()
        builder.add_usd(stage, xform=wp.transform(shift, wp.quat_identity()), floating=False)

        model = builder.finalize()
        state = model.state()
        newton.eval_fk(model, model.joint_q, model.joint_qd, state)

        body_q = state.body_q.numpy()
        offsets = {"RobotA": (10.0, 0.0, 0.0), "RobotB": (0.0, 20.0, 0.0)}
        for name, offset in offsets.items():
            idx = next(j for j, lbl in enumerate(builder.body_label) if f"{name}/Base" in lbl)
            expected = [shift[k] + offset[k] for k in range(3)]
            np.testing.assert_allclose(
                body_q[idx, :3],
                expected,
                atol=1e-4,
                err_msg=f"{name} should be at xform + original offset",
            )

    @unittest.skipUnless(USD_AVAILABLE, "Requires usd-core")
    def test_override_without_xform_raises(self):
        """override_root_xform=True without providing xform should raise a ValueError."""
        stage = self._make_stage_with_root_offset()
        builder = newton.ModelBuilder()
        with self.assertRaises(ValueError):
            builder.add_usd(stage, floating=False, override_root_xform=True)

    @staticmethod
    def _make_stage_with_visual():
        """Create a USD stage with a visual-only cube under a rigid body beneath a translated ancestor."""
        from pxr import Gf, Usd, UsdGeom, UsdPhysics

        stage = Usd.Stage.CreateInMemory()
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
        UsdPhysics.Scene.Define(stage, "/physicsScene")

        env = UsdGeom.Xform.Define(stage, "/World/env")
        env.AddTranslateOp().Set(Gf.Vec3d(100.0, 200.0, 0.0))

        root = UsdGeom.Xform.Define(stage, "/World/env/Robot")
        UsdPhysics.ArticulationRootAPI.Apply(root.GetPrim())

        base = UsdGeom.Xform.Define(stage, "/World/env/Robot/Base")
        UsdPhysics.RigidBodyAPI.Apply(base.GetPrim())
        UsdPhysics.MassAPI.Apply(base.GetPrim()).GetMassAttr().Set(1.0)

        UsdGeom.Cube.Define(stage, "/World/env/Robot/Base/Visual").GetSizeAttr().Set(0.1)

        link = UsdGeom.Xform.Define(stage, "/World/env/Robot/Link")
        link.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, 1.0))
        UsdPhysics.RigidBodyAPI.Apply(link.GetPrim())
        UsdPhysics.MassAPI.Apply(link.GetPrim()).GetMassAttr().Set(0.5)

        joint = UsdPhysics.RevoluteJoint.Define(stage, "/World/env/Robot/Joint")
        joint.CreateBody0Rel().SetTargets(["/World/env/Robot/Base"])
        joint.CreateBody1Rel().SetTargets(["/World/env/Robot/Link"])
        joint.CreateLocalPos0Attr().Set(Gf.Vec3f(0.0, 0.0, 1.0))
        joint.CreateLocalPos1Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
        joint.CreateLocalRot0Attr().Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
        joint.CreateLocalRot1Attr().Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
        joint.CreateAxisAttr().Set("Z")

        return stage

    @unittest.skipUnless(USD_AVAILABLE, "Requires usd-core")
    def test_visual_shape_aligned_with_body(self):
        """Visual shapes stay aligned with their rigid body with default override_root_xform=False."""
        stage = self._make_stage_with_visual()

        builder = newton.ModelBuilder()
        builder.add_usd(
            stage,
            xform=wp.transform((5.0, 0.0, 0.0), wp.quat_identity()),
            floating=False,
        )

        base_idx = builder.body_label.index("/World/env/Robot/Base")
        visual_shapes = [i for i, b in enumerate(builder.shape_body) if b == base_idx]
        self.assertGreater(len(visual_shapes), 0, "Expected at least one visual shape on Base")

        for sid in visual_shapes:
            shape_tf = builder.shape_transform[sid]
            np.testing.assert_allclose(
                shape_tf.p, [0.0, 0.0, 0.0], atol=1e-4, err_msg="Visual shape position should be at body origin"
            )
            np.testing.assert_allclose(
                shape_tf.q, [0.0, 0.0, 0.0, 1.0], atol=1e-4, err_msg="Visual shape rotation should be identity"
            )

    @unittest.skipUnless(USD_AVAILABLE, "Requires usd-core")
    def test_override_visual_shape_aligned_with_body(self):
        """Visual shapes stay aligned with their rigid body when override_root_xform=True
        strips a non-identity ancestor transform."""
        stage = self._make_stage_with_visual()

        builder = newton.ModelBuilder()
        builder.add_usd(
            stage,
            xform=wp.transform((5.0, 0.0, 0.0), wp.quat_identity()),
            floating=False,
            override_root_xform=True,
        )

        base_idx = builder.body_label.index("/World/env/Robot/Base")
        visual_shapes = [i for i, b in enumerate(builder.shape_body) if b == base_idx]
        self.assertGreater(len(visual_shapes), 0, "Expected at least one visual shape on Base")

        for sid in visual_shapes:
            shape_tf = builder.shape_transform[sid]
            np.testing.assert_allclose(
                shape_tf.p, [0.0, 0.0, 0.0], atol=1e-4, err_msg="Visual shape position should be at body origin"
            )
            np.testing.assert_allclose(
                shape_tf.q, [0.0, 0.0, 0.0, 1.0], atol=1e-4, err_msg="Visual shape rotation should be identity"
            )

    @staticmethod
    def _make_stage_with_loop_joint():
        """Create a USD stage with a 3-body chain and an excludeFromArticulation loop joint."""
        from pxr import Gf, Usd, UsdGeom, UsdPhysics

        stage = Usd.Stage.CreateInMemory()
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
        UsdPhysics.Scene.Define(stage, "/physicsScene")

        env = UsdGeom.Xform.Define(stage, "/World/env")
        env.AddTranslateOp().Set(Gf.Vec3d(100.0, 200.0, 0.0))

        root = UsdGeom.Xform.Define(stage, "/World/env/Robot")
        UsdPhysics.ArticulationRootAPI.Apply(root.GetPrim())

        for name, pos in [("Base", (0, 0, 0)), ("Mid", (0, 0, 1)), ("Tip", (0, 0, 2))]:
            body = UsdGeom.Xform.Define(stage, f"/World/env/Robot/{name}")
            body.AddTranslateOp().Set(Gf.Vec3d(*pos))
            UsdPhysics.RigidBodyAPI.Apply(body.GetPrim())
            UsdPhysics.MassAPI.Apply(body.GetPrim()).GetMassAttr().Set(1.0)

        for jname, b0, b1 in [("J1", "Base", "Mid"), ("J2", "Mid", "Tip")]:
            j = UsdPhysics.RevoluteJoint.Define(stage, f"/World/env/Robot/{jname}")
            j.CreateBody0Rel().SetTargets([f"/World/env/Robot/{b0}"])
            j.CreateBody1Rel().SetTargets([f"/World/env/Robot/{b1}"])
            j.CreateLocalPos0Attr().Set(Gf.Vec3f(0, 0, 1))
            j.CreateLocalPos1Attr().Set(Gf.Vec3f(0, 0, 0))
            j.CreateLocalRot0Attr().Set(Gf.Quatf(1, 0, 0, 0))
            j.CreateLocalRot1Attr().Set(Gf.Quatf(1, 0, 0, 0))
            j.CreateAxisAttr().Set("Z")

        loop = UsdPhysics.FixedJoint.Define(stage, "/World/env/Robot/LoopJoint")
        loop.CreateBody0Rel().SetTargets(["/World/env/Robot/Base"])
        loop.CreateBody1Rel().SetTargets(["/World/env/Robot/Tip"])
        loop.CreateLocalPos0Attr().Set(Gf.Vec3f(0, 0, 2))
        loop.CreateLocalPos1Attr().Set(Gf.Vec3f(0, 0, 0))
        loop.CreateLocalRot0Attr().Set(Gf.Quatf(1, 0, 0, 0))
        loop.CreateLocalRot1Attr().Set(Gf.Quatf(1, 0, 0, 0))
        from pxr import Sdf

        loop.GetPrim().CreateAttribute("physics:excludeFromArticulation", Sdf.ValueTypeNames.Bool).Set(True)

        return stage

    @unittest.skipUnless(USD_AVAILABLE, "Requires usd-core")
    def test_loop_joint_default(self):
        """Loop joint body positions are correct with default xform (relative)."""
        stage = self._make_stage_with_loop_joint()
        shift = (5.0, 0.0, 0.0)

        builder = newton.ModelBuilder()
        builder.add_usd(stage, xform=wp.transform(shift, wp.quat_identity()), floating=False)

        model = builder.finalize()
        state = model.state()
        newton.eval_fk(model, model.joint_q, model.joint_qd, state)

        body_q = state.body_q.numpy()
        # Default: xform composes with ancestor (100,200,0)
        base_idx = builder.body_label.index("/World/env/Robot/Base")
        tip_idx = builder.body_label.index("/World/env/Robot/Tip")
        np.testing.assert_allclose(body_q[base_idx, :3], [105.0, 200.0, 0.0], atol=1e-4)
        np.testing.assert_allclose(body_q[tip_idx, :3], [105.0, 200.0, 2.0], atol=1e-4)

        # Loop joint should exist and not be part of the articulation
        loop_idx = builder.joint_label.index("/World/env/Robot/LoopJoint")
        self.assertEqual(builder.joint_articulation[loop_idx], -1)

    @unittest.skipUnless(USD_AVAILABLE, "Requires usd-core")
    def test_loop_joint_override(self):
        """Loop joint body positions are correct with override_root_xform=True."""
        stage = self._make_stage_with_loop_joint()
        shift = (5.0, 0.0, 0.0)

        builder = newton.ModelBuilder()
        builder.add_usd(stage, xform=wp.transform(shift, wp.quat_identity()), floating=False, override_root_xform=True)

        model = builder.finalize()
        state = model.state()
        newton.eval_fk(model, model.joint_q, model.joint_qd, state)

        body_q = state.body_q.numpy()
        # Override: ancestor stripped, bodies at xform + internal offsets
        base_idx = builder.body_label.index("/World/env/Robot/Base")
        tip_idx = builder.body_label.index("/World/env/Robot/Tip")
        np.testing.assert_allclose(body_q[base_idx, :3], [5.0, 0.0, 0.0], atol=1e-4)
        np.testing.assert_allclose(body_q[tip_idx, :3], [5.0, 0.0, 2.0], atol=1e-4)

        loop_idx = builder.joint_label.index("/World/env/Robot/LoopJoint")
        self.assertEqual(builder.joint_articulation[loop_idx], -1)

    @staticmethod
    def _make_stage_with_world_joint():
        """Create a USD stage where a fixed joint connects a non-body prim to the root body.

        This exercises the root-joint code path (first_joint_parent == -1) where
        ``world_body_xform`` is derived from the non-body side of the joint.
        """
        from pxr import Gf, Usd, UsdGeom, UsdPhysics

        stage = Usd.Stage.CreateInMemory()
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
        UsdPhysics.Scene.Define(stage, "/physicsScene")

        env = UsdGeom.Xform.Define(stage, "/World/env")
        env.AddTranslateOp().Set(Gf.Vec3d(100.0, 200.0, 0.0))

        root = UsdGeom.Xform.Define(stage, "/World/env/Robot")
        UsdPhysics.ArticulationRootAPI.Apply(root.GetPrim())

        ground = UsdGeom.Xform.Define(stage, "/World/env/Ground")
        ground.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, 0.5))

        base = UsdGeom.Xform.Define(stage, "/World/env/Robot/Base")
        UsdPhysics.RigidBodyAPI.Apply(base.GetPrim())
        UsdPhysics.MassAPI.Apply(base.GetPrim()).GetMassAttr().Set(1.0)

        fixed = UsdPhysics.FixedJoint.Define(stage, "/World/env/Robot/WorldJoint")
        fixed.CreateBody0Rel().SetTargets(["/World/env/Ground"])
        fixed.CreateBody1Rel().SetTargets(["/World/env/Robot/Base"])
        fixed.CreateLocalPos0Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
        fixed.CreateLocalPos1Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
        fixed.CreateLocalRot0Attr().Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
        fixed.CreateLocalRot1Attr().Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))

        link = UsdGeom.Xform.Define(stage, "/World/env/Robot/Link")
        link.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, 1.0))
        UsdPhysics.RigidBodyAPI.Apply(link.GetPrim())
        UsdPhysics.MassAPI.Apply(link.GetPrim()).GetMassAttr().Set(0.5)

        rev = UsdPhysics.RevoluteJoint.Define(stage, "/World/env/Robot/RevJoint")
        rev.CreateBody0Rel().SetTargets(["/World/env/Robot/Base"])
        rev.CreateBody1Rel().SetTargets(["/World/env/Robot/Link"])
        rev.CreateLocalPos0Attr().Set(Gf.Vec3f(0.0, 0.0, 1.0))
        rev.CreateLocalPos1Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
        rev.CreateLocalRot0Attr().Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
        rev.CreateLocalRot1Attr().Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
        rev.CreateAxisAttr().Set("Z")

        return stage

    @staticmethod
    def _make_stage_with_empty_world_body0_joint():
        """Create a USD stage where a root fixed joint leaves physics:body0 empty."""
        from pxr import Gf, Usd, UsdGeom, UsdPhysics

        stage = Usd.Stage.CreateInMemory()
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
        UsdPhysics.Scene.Define(stage, "/physicsScene")

        env = UsdGeom.Xform.Define(stage, "/World/env")
        env.AddTranslateOp().Set(Gf.Vec3d(100.0, 200.0, 0.0))

        root = UsdGeom.Xform.Define(stage, "/World/env/Robot")
        UsdPhysics.ArticulationRootAPI.Apply(root.GetPrim())

        base = UsdGeom.Xform.Define(stage, "/World/env/Robot/Base")
        base.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, 0.5))
        base.AddOrientOp().Set(
            Gf.Quatf(0.70710677, 0.18898223, 0.37796447, 0.5669467)
        )  # 90-deg rotation around normalized axis (1,2,3)
        UsdPhysics.RigidBodyAPI.Apply(base.GetPrim())
        UsdPhysics.MassAPI.Apply(base.GetPrim()).GetMassAttr().Set(1.0)

        fixed = UsdPhysics.FixedJoint.Define(stage, "/World/env/Robot/WorldJointEmpty")
        fixed.CreateBody1Rel().SetTargets(["/World/env/Robot/Base"])
        fixed.CreateLocalPos0Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
        fixed.CreateLocalPos1Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
        fixed.CreateLocalRot0Attr().Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
        fixed.CreateLocalRot1Attr().Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))

        link = UsdGeom.Xform.Define(stage, "/World/env/Robot/Link")
        link.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, 1.0))
        UsdPhysics.RigidBodyAPI.Apply(link.GetPrim())
        UsdPhysics.MassAPI.Apply(link.GetPrim()).GetMassAttr().Set(0.5)

        rev = UsdPhysics.RevoluteJoint.Define(stage, "/World/env/Robot/RevJoint")
        rev.CreateBody0Rel().SetTargets(["/World/env/Robot/Base"])
        rev.CreateBody1Rel().SetTargets(["/World/env/Robot/Link"])
        rev.CreateLocalPos0Attr().Set(Gf.Vec3f(0.0, 0.0, 1.0))
        rev.CreateLocalPos1Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
        rev.CreateLocalRot0Attr().Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
        rev.CreateLocalRot1Attr().Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
        rev.CreateAxisAttr().Set("Z")

        return stage

    @unittest.skipUnless(USD_AVAILABLE, "Requires usd-core")
    def test_world_joint_default_xform(self):
        """Root joint from non-body prim: default xform composes with ancestor transforms."""
        stage = self._make_stage_with_world_joint()

        builder = newton.ModelBuilder()
        builder.add_usd(stage, xform=wp.transform((5.0, 0.0, 0.0), wp.quat_identity()))

        self.assertIn("/World/env/Robot/WorldJoint", builder.joint_label)

        model = builder.finalize()
        state = model.state()
        newton.eval_fk(model, model.joint_q, model.joint_qd, state)

        body_q = state.body_q.numpy()
        base_idx = builder.body_label.index("/World/env/Robot/Base")
        # xform (5,0,0) composed with Ground world xform (100,200,0.5) => (105, 200, 0.5)
        np.testing.assert_allclose(body_q[base_idx, :3], [105.0, 200.0, 0.5], atol=1e-4)

    @unittest.skipUnless(USD_AVAILABLE, "Requires usd-core")
    def test_world_joint_empty_body0_uses_child_pose(self):
        """Root fixed joint with empty body0 keeps joint_X_p aligned with imported root pose."""
        stage = self._make_stage_with_empty_world_body0_joint()

        builder = newton.ModelBuilder()
        builder.add_usd(stage, xform=wp.transform((5.0, 0.0, 0.0), wp.quat_identity()))

        self.assertIn("/World/env/Robot/WorldJointEmpty", builder.joint_label)
        root_joint_idx = builder.joint_label.index("/World/env/Robot/WorldJointEmpty")
        base_idx = builder.body_label.index("/World/env/Robot/Base")
        assert_np_equal(
            np.array(builder.joint_X_p[root_joint_idx].p),
            np.array(builder.body_q[base_idx].p),
            tol=1e-4,
        )

        model = builder.finalize()
        state = model.state()
        newton.eval_fk(model, model.joint_q, model.joint_qd, state)
        body_q = state.body_q.numpy()
        np.testing.assert_allclose(body_q[base_idx, :3], [105.0, 200.0, 0.5], atol=1e-4)
        # Verify rotation is preserved (sign-invariant: q and -q are equivalent)
        # Gf.Quatf stores (w, x, y, z); body_q uses xyzw.
        expected_quat = np.array([0.18898223, 0.37796447, 0.5669467, 0.70710677])
        actual_quat = body_q[base_idx, 3:]
        if np.dot(actual_quat, expected_quat) < 0:
            actual_quat = -actual_quat
        np.testing.assert_allclose(
            actual_quat,
            expected_quat,
            atol=1e-4,
            err_msg="Root body rotation must match USD prim orientation",
        )

    @unittest.skipUnless(USD_AVAILABLE, "Requires usd-core")
    def test_world_joint_override_root_xform(self):
        """Root joint from non-body prim: override_root_xform places root at xform."""
        stage = self._make_stage_with_world_joint()

        builder = newton.ModelBuilder()
        builder.add_usd(
            stage,
            xform=wp.transform((5.0, 0.0, 0.0), wp.quat_identity()),
            override_root_xform=True,
        )

        self.assertIn("/World/env/Robot/WorldJoint", builder.joint_label)

        model = builder.finalize()
        state = model.state()
        newton.eval_fk(model, model.joint_q, model.joint_qd, state)

        body_q = state.body_q.numpy()
        base_idx = builder.body_label.index("/World/env/Robot/Base")
        # override rebases at xform; Ground z=0.5 offset from articulation root is preserved
        np.testing.assert_allclose(body_q[base_idx, :3], [5.0, 0.0, 0.5], atol=1e-4)

    @unittest.skipUnless(USD_AVAILABLE, "Requires usd-core")
    def test_world_joint_override_with_rotation(self):
        """Root joint from non-body prim: override with rotation rebases correctly."""
        stage = self._make_stage_with_world_joint()
        angle = np.pi / 2
        quat = wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), angle)

        builder = newton.ModelBuilder()
        builder.add_usd(
            stage,
            xform=wp.transform((5.0, 0.0, 0.0), quat),
            override_root_xform=True,
        )

        self.assertIn("/World/env/Robot/WorldJoint", builder.joint_label)

        model = builder.finalize()
        state = model.state()
        newton.eval_fk(model, model.joint_q, model.joint_qd, state)

        body_q = state.body_q.numpy()
        base_idx = builder.body_label.index("/World/env/Robot/Base")
        # override rebases at xform; Ground z=0.5 offset preserved
        np.testing.assert_allclose(body_q[base_idx, :3], [5.0, 0.0, 0.5], atol=1e-4)

        link_idx = builder.body_label.index("/World/env/Robot/Link")
        # Link at Z+1 from Base; 90° Z-rotation doesn't affect Z offset
        np.testing.assert_allclose(body_q[link_idx, :3], [5.0, 0.0, 1.5], atol=1e-4)


class TestImportUsdMeshNormals(unittest.TestCase):
    """Tests for loading mesh normals from USD files."""

    # A simple quad (two triangles) with faceVarying normals that should be
    # smoothed across shared positions after vertex splitting.
    QUAD_WITH_FACEVARYING_NORMALS = """#usda 1.0
(
    upAxis = "Y"
)

def Mesh "quad"
{
    int[] faceVertexCounts = [3, 3]
    int[] faceVertexIndices = [0, 1, 2, 2, 1, 3]
    point3f[] points = [(0, 0, 0), (1, 0, 0), (0, 1, 0), (1, 1, 0)]
    normal3f[] primvars:normals = [(0, 0, 1), (0, 0, 1), (0, 0, 1), (0, 0, 1), (0, 0, 1), (0, 0, 1)] (
        interpolation = "faceVarying"
    )
}
"""

    # A cube with faceVarying normals — each face has its own flat normal,
    # but vertices at the same position should be clustered by the vertex
    # splitting algorithm where normals are within the angle threshold.
    CUBE_WITH_FACEVARYING_NORMALS = """#usda 1.0
(
    upAxis = "Y"
)

def Mesh "cube"
{
    int[] faceVertexCounts = [4, 4, 4, 4, 4, 4]
    int[] faceVertexIndices = [0,1,3,2, 4,6,7,5, 0,4,5,1, 2,3,7,6, 0,2,6,4, 1,5,7,3]
    point3f[] points = [
        (-0.5, -0.5, -0.5), (0.5, -0.5, -0.5),
        (-0.5, 0.5, -0.5), (0.5, 0.5, -0.5),
        (-0.5, -0.5, 0.5), (0.5, -0.5, 0.5),
        (-0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
    ]
    normal3f[] primvars:normals = [
        (0,0,-1),(0,0,-1),(0,0,-1),(0,0,-1),
        (0,0,1),(0,0,1),(0,0,1),(0,0,1),
        (0,-1,0),(0,-1,0),(0,-1,0),(0,-1,0),
        (0,1,0),(0,1,0),(0,1,0),(0,1,0),
        (-1,0,0),(-1,0,0),(-1,0,0),(-1,0,0),
        (1,0,0),(1,0,0),(1,0,0),(1,0,0)
    ] (
        interpolation = "faceVarying"
    )
}
"""

    @staticmethod
    def _create_stage_with_texture(texture_asset: str, source_color_space: str | None = None):
        from pxr import Sdf, Usd, UsdGeom, UsdShade

        stage = Usd.Stage.CreateInMemory()
        mesh = UsdGeom.Mesh.Define(stage, "/TexturedMesh")
        mesh.CreatePointsAttr().Set([(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)])
        mesh.CreateFaceVertexCountsAttr().Set([3])
        mesh.CreateFaceVertexIndicesAttr().Set([0, 1, 2])

        material = UsdShade.Material.Define(stage, "/Materials/PBR")
        preview = UsdShade.Shader.Define(stage, "/Materials/PBR/PreviewSurface")
        preview.CreateIdAttr("UsdPreviewSurface")
        texture = UsdShade.Shader.Define(stage, "/Materials/PBR/Albedo")
        texture.CreateIdAttr("UsdUVTexture")
        texture.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath(texture_asset))
        if source_color_space is not None:
            texture.CreateInput("sourceColorSpace", Sdf.ValueTypeNames.Token).Set(source_color_space)
        preview.CreateInput("baseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(texture.ConnectableAPI(), "rgb")
        material.CreateSurfaceOutput().ConnectToSource(preview.ConnectableAPI(), "surface")
        UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(material)
        return stage, mesh.GetPrim()

    @unittest.skipUnless(USD_AVAILABLE, "Requires usd-core")
    def test_get_mesh_converts_linear_texture_to_display_space(self):
        from PIL import Image

        source_rgba = np.array([[[64, 128, 255, 200]]], dtype=np.uint8)
        with tempfile.TemporaryDirectory() as tmpdir:
            texture_path = os.path.join(tmpdir, "linear.png")
            Image.fromarray(source_rgba).save(texture_path)

            _stage, prim = self._create_stage_with_texture(texture_path, source_color_space="raw")
            mesh = usd.get_mesh(prim)

        self.assertIsInstance(mesh.texture, np.ndarray)
        texture = np.asarray(mesh.texture)
        linear_rgb = source_rgba[0, 0, :3].astype(np.float32) / 255.0
        expected_rgb = np.where(
            linear_rgb <= 0.0031308,
            linear_rgb * 12.92,
            1.055 * np.power(linear_rgb, 1.0 / 2.4) - 0.055,
        )
        expected_rgb = np.clip(np.round(expected_rgb * 255.0), 0.0, 255.0).astype(np.uint8)
        np.testing.assert_array_equal(texture[0, 0, :3], expected_rgb)
        self.assertEqual(texture[0, 0, 3], source_rgba[0, 0, 3])

    @unittest.skipUnless(USD_AVAILABLE, "Requires usd-core")
    def test_get_mesh_leaves_display_texture_paths_lazy(self):
        _stage, prim = self._create_stage_with_texture("display.png", source_color_space="sRGB")

        mesh = usd.get_mesh(prim)

        self.assertEqual(mesh.texture, "display.png")

    @unittest.skipUnless(USD_AVAILABLE, "Requires usd-core")
    def test_get_mesh_loads_normals_when_requested(self):
        """get_mesh with load_normals=True produces a Mesh with non-None normals."""
        from pxr import Usd

        stage = Usd.Stage.CreateInMemory()
        stage.GetRootLayer().ImportFromString(self.QUAD_WITH_FACEVARYING_NORMALS)
        prim = stage.GetPrimAtPath("/quad")

        mesh_with = usd.get_mesh(prim, load_normals=True)
        self.assertIsNotNone(mesh_with.normals, "Normals should be loaded when load_normals=True")

        mesh_without = usd.get_mesh(prim, load_normals=False)
        self.assertIsNone(mesh_without.normals, "Normals should be None when load_normals=False")

    @unittest.skipUnless(USD_AVAILABLE, "Requires usd-core")
    def test_facevarying_normals_produce_correct_directions(self):
        """faceVarying normals on a flat quad should all point in +Z."""
        from pxr import Usd

        stage = Usd.Stage.CreateInMemory()
        stage.GetRootLayer().ImportFromString(self.QUAD_WITH_FACEVARYING_NORMALS)
        prim = stage.GetPrimAtPath("/quad")

        mesh = usd.get_mesh(prim, load_normals=True)
        normals = np.asarray(mesh.normals)
        expected_z = np.array([0.0, 0.0, 1.0])
        for i, n in enumerate(normals):
            np.testing.assert_allclose(
                n,
                expected_z,
                atol=1e-5,
                err_msg=f"Normal {i} should point in +Z for a flat quad",
            )

    @unittest.skipUnless(USD_AVAILABLE, "Requires usd-core")
    def test_cube_facevarying_normals_vertex_splitting(self):
        """Cube with 90-degree face angles should be split (hard edges)."""
        from pxr import Usd

        stage = Usd.Stage.CreateInMemory()
        stage.GetRootLayer().ImportFromString(self.CUBE_WITH_FACEVARYING_NORMALS)
        prim = stage.GetPrimAtPath("/cube")

        mesh = usd.get_mesh(prim, load_normals=True)
        # The default 25-degree threshold should split all cube edges (90 degrees).
        # Each of the 6 faces has 4 corners, triangulated to 6 indices.
        # With all edges split, we expect 6*4=24 unique vertices.
        self.assertEqual(len(mesh.vertices), 24)
        # Each normal should be unit-length and axis-aligned
        normals = np.asarray(mesh.normals)
        lengths = np.linalg.norm(normals, axis=1)
        np.testing.assert_allclose(lengths, 1.0, atol=1e-5)


class TestTetMesh(unittest.TestCase):
    def test_tetmesh_basic(self):
        """Test TetMesh construction from raw arrays."""
        vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1]], dtype=np.float32)
        tet_indices = np.array([0, 1, 2, 3, 1, 2, 3, 4], dtype=np.int32)
        tm = newton.TetMesh(vertices, tet_indices)

        self.assertEqual(tm.vertex_count, 5)
        self.assertEqual(tm.tet_count, 2)
        self.assertEqual(tm.vertices.shape, (5, 3))
        self.assertEqual(len(tm.tet_indices), 8)
        self.assertIsNone(tm.k_mu)
        self.assertIsNone(tm.density)

    def test_tetmesh_surface_triangles(self):
        """Test that surface triangles are correctly extracted from a single tet."""
        vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32)
        tet_indices = np.array([0, 1, 2, 3], dtype=np.int32)
        tm = newton.TetMesh(vertices, tet_indices)

        # A single tet has 4 boundary faces = 4 surface triangles
        self.assertEqual(len(tm.surface_tri_indices), 4 * 3)

    def test_tetmesh_surface_triangles_shared_face(self):
        """Test that shared faces between adjacent tets are eliminated."""
        vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1]], dtype=np.float32)
        tet_indices = np.array([0, 1, 2, 3, 1, 2, 3, 4], dtype=np.int32)
        tm = newton.TetMesh(vertices, tet_indices)

        # 2 tets * 4 faces = 8 total, minus 2 shared (face 1-2-3 appears in both) = 6 boundary
        self.assertEqual(len(tm.surface_tri_indices), 6 * 3)

        # Verify original winding is preserved (not lexicographically sorted)
        tris = tm.surface_tri_indices.reshape(-1, 3)
        sorted_tris = np.sort(tris, axis=1)
        has_unsorted = np.any(tris != sorted_tris)
        self.assertTrue(has_unsorted, "Surface triangles should preserve winding, not be sorted")

    def test_tetmesh_material_scalar_broadcast(self):
        """Test that scalar material values are broadcast to per-element arrays."""
        vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1]], dtype=np.float32)
        tet_indices = np.array([0, 1, 2, 3, 1, 2, 3, 4], dtype=np.int32)
        tm = newton.TetMesh(vertices, tet_indices, k_mu=1000.0, k_lambda=2000.0, k_damp=5.0, density=1.0)

        self.assertEqual(tm.k_mu.shape, (2,))
        assert_np_equal(tm.k_mu, np.array([1000.0, 1000.0], dtype=np.float32))
        assert_np_equal(tm.k_lambda, np.array([2000.0, 2000.0], dtype=np.float32))
        assert_np_equal(tm.k_damp, np.array([5.0, 5.0], dtype=np.float32))
        self.assertEqual(tm.density, 1.0)

    def test_tetmesh_material_per_element(self):
        """Test per-element material arrays."""
        vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1]], dtype=np.float32)
        tet_indices = np.array([0, 1, 2, 3, 1, 2, 3, 4], dtype=np.int32)
        k_mu = np.array([1000.0, 5000.0], dtype=np.float32)
        tm = newton.TetMesh(vertices, tet_indices, k_mu=k_mu)

        assert_np_equal(tm.k_mu, k_mu)

    def test_tetmesh_invalid_tet_indices_length(self):
        """Test that non-multiple-of-4 tet_indices raises ValueError."""
        vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32)
        with self.assertRaises(ValueError):
            newton.TetMesh(vertices, np.array([0, 1, 2], dtype=np.int32))

    @unittest.skipUnless(USD_AVAILABLE, "Requires usd-core")
    def test_get_tetmesh(self):
        from pxr import Usd

        stage = Usd.Stage.Open(os.path.join(os.path.dirname(__file__), "assets", "tetmesh_simple.usda"))
        prim = stage.GetPrimAtPath("/SimpleTetMesh")
        tm = usd.get_tetmesh(prim)

        self.assertEqual(tm.vertex_count, 5)
        self.assertEqual(tm.tet_count, 2)
        self.assertEqual(tm.vertices.dtype, np.float32)
        self.assertEqual(tm.tet_indices.dtype, np.int32)

        # Check vertices
        assert_np_equal(tm.vertices[0], np.array([0.0, 0.0, 0.0], dtype=np.float32))
        assert_np_equal(tm.vertices[4], np.array([1.0, 1.0, 1.0], dtype=np.float32))

        # Check tet indices (flattened)
        assert_np_equal(tm.tet_indices[:4], np.array([0, 1, 2, 3], dtype=np.int32))
        assert_np_equal(tm.tet_indices[4:], np.array([1, 2, 3, 4], dtype=np.int32))

    @unittest.skipUnless(USD_AVAILABLE, "Requires usd-core")
    def test_get_tetmesh_left_handed(self):
        """Test that left-handed TetMesh orientation flips winding order."""
        from pxr import Usd

        stage = Usd.Stage.CreateInMemory()
        stage.GetRootLayer().ImportFromString(
            """#usda 1.0
def TetMesh "LeftHandedTet" ()
{
    uniform token orientation = "leftHanded"
    point3f[] points = [(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 1)]
    int4[] tetVertexIndices = [(0, 1, 2, 3), (1, 2, 3, 4)]
}
"""
        )
        prim = stage.GetPrimAtPath("/LeftHandedTet")
        tm = usd.get_tetmesh(prim)

        # Indices 1 and 2 of each tet should be swapped compared to the original
        assert_np_equal(tm.tet_indices[:4], np.array([0, 2, 1, 3], dtype=np.int32))
        assert_np_equal(tm.tet_indices[4:], np.array([1, 3, 2, 4], dtype=np.int32))

    @unittest.skipUnless(USD_AVAILABLE, "Requires usd-core")
    def test_tetmesh_create_from_usd(self):
        """Test TetMesh.create_from_usd() static factory method."""
        from pxr import Usd

        stage = Usd.Stage.Open(os.path.join(os.path.dirname(__file__), "assets", "tetmesh_simple.usda"))
        prim = stage.GetPrimAtPath("/SimpleTetMesh")
        tm = newton.TetMesh.create_from_usd(prim)

        self.assertIsInstance(tm, newton.TetMesh)
        self.assertEqual(tm.tet_count, 2)
        self.assertEqual(tm.vertex_count, 5)

    @unittest.skipUnless(USD_AVAILABLE, "Requires usd-core")
    def test_get_tetmesh_missing_points(self):
        from pxr import Usd

        stage = Usd.Stage.CreateInMemory()
        stage.GetRootLayer().ImportFromString(
            """#usda 1.0
def TetMesh "Empty" ()
{
    int4[] tetVertexIndices = [(0, 1, 2, 3)]
}
"""
        )
        prim = stage.GetPrimAtPath("/Empty")
        with self.assertRaises(ValueError):
            usd.get_tetmesh(prim)

    @unittest.skipUnless(USD_AVAILABLE, "Requires usd-core")
    def test_get_tetmesh_missing_tet_indices(self):
        from pxr import Usd

        stage = Usd.Stage.CreateInMemory()
        stage.GetRootLayer().ImportFromString(
            """#usda 1.0
def TetMesh "NoTets" ()
{
    point3f[] points = [(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)]
}
"""
        )
        prim = stage.GetPrimAtPath("/NoTets")
        with self.assertRaises(ValueError):
            usd.get_tetmesh(prim)

    @unittest.skipUnless(USD_AVAILABLE, "Requires usd-core")
    def test_find_tetmesh_prims(self):
        from pxr import Usd

        stage = Usd.Stage.Open(os.path.join(os.path.dirname(__file__), "assets", "tetmesh_multi.usda"))
        prims = usd.find_tetmesh_prims(stage)

        # Should find TetA and TetB, but not NotATetMesh
        self.assertEqual(len(prims), 2)
        paths = sorted(str(p.GetPath()) for p in prims)
        self.assertEqual(paths, ["/Root/TetA", "/Root/TetB"])

    @unittest.skipUnless(USD_AVAILABLE, "Requires usd-core")
    def test_find_tetmesh_prims_load_all(self):
        """Test loading all TetMesh prims from a multi-mesh stage."""
        from pxr import Usd

        stage = Usd.Stage.Open(os.path.join(os.path.dirname(__file__), "assets", "tetmesh_multi.usda"))
        prims = usd.find_tetmesh_prims(stage)
        tetmeshes = [usd.get_tetmesh(p) for p in prims]

        self.assertEqual(len(tetmeshes), 2)
        # TetA: 4 verts, 1 tet; TetB: 5 verts, 2 tets
        counts = sorted((tm.vertex_count, tm.tet_count) for tm in tetmeshes)
        self.assertEqual(counts, [(4, 1), (5, 2)])

    @unittest.skipUnless(USD_AVAILABLE, "Requires usd-core")
    def test_find_tetmesh_prims_empty_stage(self):
        from pxr import Usd

        stage = Usd.Stage.CreateInMemory()
        stage.GetRootLayer().ImportFromString(
            """#usda 1.0
def Mesh "JustAMesh" ()
{
    point3f[] points = [(0, 0, 0), (1, 0, 0), (0, 1, 0)]
    int[] faceVertexCounts = [3]
    int[] faceVertexIndices = [0, 1, 2]
}
"""
        )
        prims = usd.find_tetmesh_prims(stage)
        self.assertEqual(len(prims), 0)

    @unittest.skipUnless(USD_AVAILABLE, "Requires usd-core")
    def test_get_tetmesh_with_material(self):
        """Test that physics material properties are read from USD."""
        from pxr import Usd

        stage = Usd.Stage.Open(os.path.join(os.path.dirname(__file__), "assets", "tetmesh_with_material.usda"))
        prim = stage.GetPrimAtPath("/World/SoftBody")
        tm = usd.get_tetmesh(prim, compat_namespaces=())

        # E = 300000, nu = 0.3
        # k_mu = E / (2 * (1 + nu)) = 300000 / 2.6 = 115384.615...
        # k_lambda = E * nu / ((1 + nu) * (1 - 2*nu)) = 90000 / (1.3 * 0.4) = 173076.923...
        self.assertIsNotNone(tm.k_mu)
        self.assertIsNotNone(tm.k_lambda)
        self.assertAlmostEqual(tm.k_mu[0], 300000.0 / (2.0 * 1.3), places=0)
        self.assertAlmostEqual(tm.k_lambda[0], 300000.0 * 0.3 / (1.3 * 0.4), places=0)
        self.assertAlmostEqual(tm.density, 40.0)

    @unittest.skipUnless(USD_AVAILABLE, "Requires usd-core")
    def test_get_tetmesh_vendor_namespace_legacy_recovery(self):
        """Vendor-namespaced material reads follow the ``compat_namespaces`` opt-in.

        Canonical-only (``compat_namespaces=()``) reads moduli only from a ``physics:`` material
        that applies ``PhysicsVolumeDeformableMaterialAPI``. The deprecated default (``None``) reads
        vendor namespaces off any bound material and emits a ``DeprecationWarning``;
        ``DEFORMABLE_LEGACY_NAMESPACES`` keeps that behavior explicitly (without the warning).
        """
        from pxr import Usd

        # Legacy-style asset: a vendor-namespaced material with no PhysicsVolumeDeformableMaterialAPI.
        usda = """#usda 1.0
(
    defaultPrim = "World"
    metersPerUnit = 1
    upAxis = "Y"
)
def Xform "World" ()
{
    def TetMesh "SoftBody" (
        prepend apiSchemas = ["MaterialBindingAPI"]
    )
    {
        point3f[] points = [(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)]
        int4[] tetVertexIndices = [(0, 1, 2, 3)]
        rel material:binding:physics = </World/PhysicsMaterial>
    }
    def Material "PhysicsMaterial"
    {
        float omniphysics:density = 40
        float omniphysics:youngsModulus = 300000
        float omniphysics:poissonsRatio = 0.3
    }
    def TetMesh "CanonicalBody" (
        prepend apiSchemas = ["MaterialBindingAPI"]
    )
    {
        point3f[] points = [(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)]
        int4[] tetVertexIndices = [(0, 1, 2, 3)]
        rel material:binding:physics = </World/CanonicalMaterial>
    }
    def Material "CanonicalMaterial" (
        prepend apiSchemas = ["PhysicsVolumeDeformableMaterialAPI"]
    )
    {
        custom float physics:density = 40
        custom float physics:youngsModulus = 300000
        custom float physics:poissonsRatio = 0.3
    }
    def TetMesh "UnscopedBody" (
        prepend apiSchemas = ["MaterialBindingAPI"]
    )
    {
        point3f[] points = [(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)]
        int4[] tetVertexIndices = [(0, 1, 2, 3)]
        rel material:binding:physics = </World/UnscopedMaterial>
    }
    def Material "UnscopedMaterial"
    {
        custom float physics:density = 40
        custom float physics:youngsModulus = 300000
        custom float physics:poissonsRatio = 0.3
    }
}
"""
        stage = Usd.Stage.CreateInMemory()
        stage.GetRootLayer().ImportFromString(usda)
        prim = stage.GetPrimAtPath("/World/SoftBody")

        # Canonical-only: vendor moduli are ignored.
        tm_canonical = usd.get_tetmesh(prim, compat_namespaces=())
        self.assertIsNone(tm_canonical.k_mu)
        self.assertIsNone(tm_canonical.density)

        # Deprecated default: reads the vendor namespaces off the bound material and warns.
        with self.assertWarns(DeprecationWarning):
            tm_default = usd.get_tetmesh(prim)
        self.assertIsNotNone(tm_default.k_mu)
        self.assertAlmostEqual(tm_default.density, 40.0)

        # Explicit legacy namespaces: same reads, no deprecation warning.
        tm_legacy = usd.get_tetmesh(prim, compat_namespaces=usd.DEFORMABLE_LEGACY_NAMESPACES)
        self.assertIsNotNone(tm_legacy.k_mu)
        self.assertAlmostEqual(tm_legacy.k_mu[0], 300000.0 / (2.0 * 1.3), places=0)
        self.assertAlmostEqual(tm_legacy.density, 40.0)

        # A canonical material under the deprecated default reads identically and must NOT
        # warn: the default change alters nothing for it (the gate matches add_usd's).
        with warnings.catch_warnings():
            warnings.simplefilter("error", DeprecationWarning)
            tm_canonical_default = usd.get_tetmesh(stage.GetPrimAtPath("/World/CanonicalBody"))
        self.assertAlmostEqual(tm_canonical_default.density, 40.0)
        self.assertIsNotNone(tm_canonical_default.k_mu)

        # Canonical physics: moduli on a material WITHOUT the deformable material API: the
        # deprecated default reads them off any bound material, but canonical-only scopes
        # moduli to API-applied materials and drops them. The default is load-bearing, so
        # it must warn.
        unscoped = stage.GetPrimAtPath("/World/UnscopedBody")
        with self.assertWarns(DeprecationWarning):
            tm_unscoped_default = usd.get_tetmesh(unscoped)
        self.assertIsNotNone(tm_unscoped_default.k_mu)
        self.assertAlmostEqual(tm_unscoped_default.density, 40.0)
        tm_unscoped_canonical = usd.get_tetmesh(unscoped, compat_namespaces=())
        self.assertIsNone(tm_unscoped_canonical.k_mu)
        self.assertIsNone(tm_unscoped_canonical.density)

    @unittest.skipUnless(USD_AVAILABLE, "Requires usd-core")
    def test_get_tetmesh_no_material(self):
        """Test that TetMesh without material binding has None material properties."""
        from pxr import Usd

        stage = Usd.Stage.Open(os.path.join(os.path.dirname(__file__), "assets", "tetmesh_simple.usda"))
        prim = stage.GetPrimAtPath("/SimpleTetMesh")
        tm = usd.get_tetmesh(prim)

        self.assertIsNone(tm.k_mu)
        self.assertIsNone(tm.k_lambda)
        self.assertIsNone(tm.density)

    @unittest.skipUnless(USD_AVAILABLE, "Requires usd-core")
    def test_get_tetmesh_warns_on_geometry_authored_moduli(self):
        """A material modulus authored on the TetMesh geometry instead of the bound material warns,
        pointing to the deformable material API, instead of being dropped silently."""
        from pxr import Usd

        stage = Usd.Stage.CreateInMemory()
        stage.GetRootLayer().ImportFromString(
            """#usda 1.0
def TetMesh "Soft" ()
{
    point3f[] points = [(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)]
    int4[] tetVertexIndices = [(0, 1, 2, 3)]
    custom float physics:youngsModulus = 300000
}
"""
        )
        prim = stage.GetPrimAtPath("/Soft")
        with self.assertWarnsRegex(UserWarning, "authored on the geometry"):
            usd.get_tetmesh(prim, compat_namespaces=())

    @unittest.skipUnless(USD_AVAILABLE, "Requires usd-core")
    def test_add_usd_imports_tetmesh(self):
        """Test that add_usd imports TetMesh prims as soft meshes."""
        asset_path = os.path.join(os.path.dirname(__file__), "assets", "tetmesh_with_material.usda")

        builder = newton.ModelBuilder()
        builder.add_usd(asset_path)

        self.assertEqual(len(builder.particle_q), 4)
        self.assertEqual(len(builder.tet_indices), 1)
        self.assertEqual(len(builder.tri_indices), 4)
        self.assertAlmostEqual(builder.tet_materials[0][0], 300000.0 / (2.0 * 1.3), places=0)
        self.assertAlmostEqual(builder.tet_materials[0][1], 300000.0 * 0.3 / (1.3 * 0.4), places=0)
        self.assertAlmostEqual(sum(builder.particle_mass), 40.0 / 6.0, places=5)

    @unittest.skipUnless(USD_AVAILABLE, "Requires usd-core")
    def test_add_usd_imports_instanced_tetmesh_once_per_instance(self):
        """Test that instance proxies import one TetMesh per instance."""
        from pxr import Sdf, Usd, UsdGeom

        stage = Usd.Stage.CreateInMemory()
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)

        world = UsdGeom.Xform.Define(stage, "/World")
        stage.SetDefaultPrim(world.GetPrim())

        # Author the template as a class prim (abstract, excluded by the default
        # traversal predicate) so only the per-instance proxies are imported.
        stage.CreateClassPrim("/TetProto")
        tetmesh = stage.DefinePrim("/TetProto/SoftBody", "TetMesh")
        tetmesh.CreateAttribute("points", Sdf.ValueTypeNames.Point3fArray).Set(
            [(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)]
        )
        tetmesh.CreateAttribute("tetVertexIndices", Sdf.ValueTypeNames.Int4Array).Set([(0, 1, 2, 3)])

        for i in range(2):
            instance = UsdGeom.Xform.Define(stage, f"/World/Instance{i}")
            instance_prim = instance.GetPrim()
            instance_prim.GetReferences().AddInternalReference("/TetProto")
            instance_prim.SetInstanceable(True)

        builder = newton.ModelBuilder()
        builder.add_usd(stage)

        self.assertEqual(len(builder.particle_q), 8)
        self.assertEqual(len(builder.tet_indices), 2)
        self.assertEqual(len(builder.tri_indices), 8)

    @unittest.skipUnless(USD_AVAILABLE, "Requires usd-core")
    def test_add_usd_imports_tetmesh_with_transforms(self):
        """Test that add_usd applies TetMesh rotation and non-uniform scale transforms."""
        from pxr import Usd

        stage = Usd.Stage.CreateInMemory()
        stage.GetRootLayer().ImportFromString(
            """#usda 1.0
(
    defaultPrim = "World"
    upAxis = "Z"
)

def Xform "World"
{
    double3 xformOp:translate = (1, 2, 3)
    float xformOp:rotateZ = 90
    float3 xformOp:scale = (2, 3, 4)
    uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:rotateZ", "xformOp:scale"]

    def Xform "Offset"
    {
        double3 xformOp:translate = (0.5, -1, 2)
        uniform token[] xformOpOrder = ["xformOp:translate"]

        def TetMesh "SoftBody" ()
        {
            point3f[] points = [(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)]
            int4[] tetVertexIndices = [(0, 1, 2, 3)]
        }
    }
}
"""
        )

        import_quat = wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), np.pi)
        builder = newton.ModelBuilder()
        builder.add_usd(stage, xform=wp.transform((4.0, 5.0, 6.0), import_quat))

        positions = np.array(builder.particle_q, dtype=np.float32)
        expected = np.array(
            [
                [0.0, 2.0, 17.0],
                [0.0, 0.0, 17.0],
                [3.0, 2.0, 17.0],
                [0.0, 2.0, 21.0],
            ],
            dtype=np.float32,
        )
        np.testing.assert_allclose(positions, expected, atol=1e-6)

    def test_tetmesh_save_load_npz(self):
        """Test TetMesh round-trip save/load via .npz."""

        vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32)
        tet_indices = np.array([0, 1, 2, 3], dtype=np.int32)
        tm = newton.TetMesh(vertices, tet_indices, k_mu=1000.0, k_lambda=2000.0, density=40.0)

        with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
            path = f.name

        try:
            tm.save(path)
            tm2 = newton.TetMesh.create_from_file(path)

            assert_np_equal(tm2.vertices, tm.vertices)
            assert_np_equal(tm2.tet_indices, tm.tet_indices)
            assert_np_equal(tm2.k_mu, tm.k_mu)
            assert_np_equal(tm2.k_lambda, tm.k_lambda)
            self.assertAlmostEqual(tm2.density, 40.0)
        finally:
            os.unlink(path)

    def test_tetmesh_save_load_vtk(self):
        """Test TetMesh round-trip save/load via .vtk (meshio)."""

        try:
            import meshio  # noqa: F401
        except ImportError:
            self.skipTest("meshio not installed")

        vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1]], dtype=np.float32)
        tet_indices = np.array([0, 1, 2, 3, 1, 2, 3, 4], dtype=np.int32)
        per_tet_region = np.array([10, 20], dtype=np.int32)
        per_vertex_temp = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
        tm = newton.TetMesh(
            vertices,
            tet_indices,
            k_mu=1000.0,
            k_lambda=2000.0,
            density=40.0,
            custom_attributes={"regionId": per_tet_region, "temperature": per_vertex_temp},
        )

        with tempfile.NamedTemporaryFile(suffix=".vtk", delete=False) as f:
            path = f.name

        try:
            tm.save(path)
            tm2 = newton.TetMesh.create_from_file(path)

            self.assertEqual(tm2.vertex_count, 5)
            self.assertEqual(tm2.tet_count, 2)
            assert_np_equal(tm2.tet_indices[:4], np.array([0, 1, 2, 3], dtype=np.int32))
            assert_np_equal(tm2.tet_indices[4:], np.array([1, 2, 3, 4], dtype=np.int32))

            # Material arrays round-trip
            self.assertIsNotNone(tm2.k_mu)
            assert_np_equal(tm2.k_mu, np.array([1000.0, 1000.0], dtype=np.float32))
            assert_np_equal(tm2.k_lambda, np.array([2000.0, 2000.0], dtype=np.float32))
            self.assertAlmostEqual(tm2.density, 40.0)

            # Custom attributes round-trip (check values, not just keys)
            self.assertIn("regionId", tm2.custom_attributes)
            self.assertIn("temperature", tm2.custom_attributes)
            region_arr, _region_freq = tm2.custom_attributes["regionId"]
            temp_arr, _temp_freq = tm2.custom_attributes["temperature"]
            assert_np_equal(region_arr.flatten(), per_tet_region)
            assert_np_equal(temp_arr.flatten(), per_vertex_temp)
        finally:
            os.unlink(path)

    def test_tetmesh_custom_attributes_reserved_name(self):
        """Test that reserved custom attribute names are rejected."""
        vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32)
        tet_indices = np.array([0, 1, 2, 3], dtype=np.int32)

        for reserved in ("vertices", "tet_indices", "k_mu", "k_lambda", "k_damp", "density"):
            with self.assertRaisesRegex(ValueError, "reserved", msg=f"Should reject reserved name '{reserved}'"):
                newton.TetMesh(vertices, tet_indices, custom_attributes={reserved: np.array([1.0])})

    def test_tetmesh_custom_attributes_constructor(self):
        """Test TetMesh stores custom attributes passed at construction."""
        vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32)
        tet_indices = np.array([0, 1, 2, 3], dtype=np.int32)
        temperature = np.array([100.0, 200.0, 300.0, 400.0], dtype=np.float32)
        region_id = np.array([7], dtype=np.int32)

        # Single tet: vertex_count == tri_count == 4, so temperature needs explicit frequency.
        # regionId also needs explicit frequency because tet_count == 1 is ambiguous with ONCE.
        tm = newton.TetMesh(
            vertices,
            tet_indices,
            custom_attributes={
                "temperature": (temperature, newton.Model.AttributeFrequency.PARTICLE),
                "regionId": (region_id, newton.Model.AttributeFrequency.TETRAHEDRON),
            },
        )

        self.assertIn("temperature", tm.custom_attributes)
        self.assertIn("regionId", tm.custom_attributes)
        arr, freq = tm.custom_attributes["temperature"]
        assert_np_equal(arr, temperature)
        self.assertEqual(freq, newton.Model.AttributeFrequency.PARTICLE)
        arr, freq = tm.custom_attributes["regionId"]
        assert_np_equal(arr, region_id)
        self.assertEqual(freq, newton.Model.AttributeFrequency.TETRAHEDRON)

    def test_tetmesh_custom_attributes_infer_once(self):
        """Test that length-1 arrays are inferred as ONCE when unambiguous."""
        vertices = np.array(
            [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [0.5, 0.5, 0.5]],
            dtype=np.float32,
        )
        tet_indices = np.array([0, 1, 2, 3, 0, 1, 2, 4], dtype=np.int32)
        constant = np.array([42.0], dtype=np.float32)

        tm = newton.TetMesh(
            vertices,
            tet_indices,
            custom_attributes={"constant": constant},
        )

        arr, freq = tm.custom_attributes["constant"]
        assert_np_equal(arr, constant)
        self.assertEqual(freq, newton.Model.AttributeFrequency.ONCE)

    def test_tetmesh_custom_attributes_ambiguous_once(self):
        """Test that length-1 arrays raise when tet_count is also 1."""
        vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32)
        tet_indices = np.array([0, 1, 2, 3], dtype=np.int32)

        with self.assertRaisesRegex(ValueError, "ONCE"):
            newton.TetMesh(
                vertices,
                tet_indices,
                custom_attributes={"ambig": np.array([1.0], dtype=np.float32)},
            )

    def test_tetmesh_custom_attributes_empty_by_default(self):
        """Test TetMesh has empty custom_attributes when none are provided."""
        vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32)
        tet_indices = np.array([0, 1, 2, 3], dtype=np.int32)
        tm = newton.TetMesh(vertices, tet_indices)
        self.assertEqual(len(tm.custom_attributes), 0)

    @unittest.skipUnless(USD_AVAILABLE, "Requires usd-core")
    def test_tetmesh_custom_attributes_from_usd(self):
        """Test that custom primvars are parsed from USD into custom_attributes."""
        from pxr import Usd

        assets_dir = os.path.join(os.path.dirname(__file__), "assets")
        stage = Usd.Stage.Open(os.path.join(assets_dir, "tetmesh_custom_attrs.usda"))
        prim = stage.GetPrimAtPath("/TetMeshWithAttrs")
        tm = newton.TetMesh.create_from_usd(prim)

        self.assertEqual(tm.vertex_count, 5)
        self.assertEqual(tm.tet_count, 2)

        # Per-vertex temperature primvar
        self.assertIn("temperature", tm.custom_attributes)
        arr, freq = tm.custom_attributes["temperature"]
        assert_np_equal(arr, np.array([100, 200, 300, 400, 500], dtype=np.float32))
        self.assertEqual(freq, newton.Model.AttributeFrequency.PARTICLE)

        # Per-tet regionId primvar
        self.assertIn("regionId", tm.custom_attributes)
        arr, freq = tm.custom_attributes["regionId"]
        assert_np_equal(arr, np.array([0, 1], dtype=np.int32))
        self.assertEqual(freq, newton.Model.AttributeFrequency.TETRAHEDRON)

        # Per-vertex vector primvar
        self.assertIn("velocityField", tm.custom_attributes)
        arr, freq = tm.custom_attributes["velocityField"]
        self.assertEqual(arr.shape, (5, 3))
        self.assertEqual(freq, newton.Model.AttributeFrequency.PARTICLE)

    def test_tetmesh_custom_attributes_npz_roundtrip(self):
        """Test custom attributes survive save/load via .npz."""
        vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32)
        tet_indices = np.array([0, 1, 2, 3], dtype=np.int32)
        temperature = np.array([10.0, 20.0, 30.0, 40.0], dtype=np.float32)
        region_id = np.array([3], dtype=np.int32)

        # Single tet: vertex_count == tri_count == 4, so temperature needs explicit frequency.
        # regionId also needs explicit frequency because tet_count == 1 is ambiguous with ONCE.
        tm = newton.TetMesh(
            vertices,
            tet_indices,
            custom_attributes={
                "temperature": (temperature, newton.Model.AttributeFrequency.PARTICLE),
                "regionId": (region_id, newton.Model.AttributeFrequency.TETRAHEDRON),
            },
        )

        with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
            path = f.name

        try:
            tm.save(path)
            tm2 = newton.TetMesh.create_from_file(path)

            self.assertIn("temperature", tm2.custom_attributes)
            arr, freq = tm2.custom_attributes["temperature"]
            assert_np_equal(arr, temperature)
            self.assertEqual(freq, newton.Model.AttributeFrequency.PARTICLE)
            self.assertIn("regionId", tm2.custom_attributes)
            arr, freq = tm2.custom_attributes["regionId"]
            assert_np_equal(arr, region_id)
            self.assertEqual(freq, newton.Model.AttributeFrequency.TETRAHEDRON)
        finally:
            os.unlink(path)

    def test_tetmesh_custom_attributes_to_model(self):
        """Test custom attributes flow from TetMesh through add_soft_mesh into the finalized Model."""
        vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1]], dtype=np.float32)
        tet_indices = np.array([0, 1, 2, 3, 1, 2, 3, 4], dtype=np.int32)

        # Per-vertex attribute (5 vertices)
        temperature = np.array([100.0, 200.0, 300.0, 400.0, 500.0], dtype=np.float32)
        # Per-tet attribute (2 tets)
        region_id = np.array([0, 1], dtype=np.int32)

        tm = newton.TetMesh(
            vertices,
            tet_indices,
            custom_attributes={
                "temperature": temperature,
                "regionId": region_id,
            },
        )

        builder = newton.ModelBuilder()

        # Register custom attributes before calling add_soft_mesh
        builder.add_custom_attribute(
            newton.ModelBuilder.CustomAttribute(
                name="temperature",
                dtype=wp.float32,
                frequency=newton.Model.AttributeFrequency.PARTICLE,
            )
        )
        builder.add_custom_attribute(
            newton.ModelBuilder.CustomAttribute(
                name="regionId",
                dtype=wp.int32,
                frequency=newton.Model.AttributeFrequency.TETRAHEDRON,
            )
        )

        builder.add_soft_mesh(
            mesh=tm,
            pos=(0.0, 0.0, 0.0),
            rot=wp.quat_identity(),
            scale=1.0,
            vel=(0.0, 0.0, 0.0),
        )

        model = builder.finalize()

        # Verify per-vertex attribute (PARTICLE frequency)
        self.assertTrue(hasattr(model, "temperature"))
        temp_arr = model.temperature.numpy()
        self.assertEqual(len(temp_arr), model.particle_count)
        np.testing.assert_allclose(temp_arr, temperature)

        # Verify per-tet attribute (TETRAHEDRON frequency)
        self.assertTrue(hasattr(model, "regionId"))
        region_arr = model.regionId.numpy()
        self.assertEqual(len(region_arr), model.tet_count)
        np.testing.assert_array_equal(region_arr, region_id)

    @unittest.skipUnless(USD_AVAILABLE, "Requires usd-core")
    def test_add_usd_does_not_mutate_loaded_tetmesh_custom_attributes(self):
        """Test that add_usd filters TetMesh custom attributes without mutating the loaded mesh."""
        from pxr import Usd

        stage = Usd.Stage.CreateInMemory()
        stage.GetRootLayer().ImportFromString(
            """#usda 1.0
(
    defaultPrim = "World"
)

def Xform "World"
{
    def TetMesh "SoftBody" ()
    {
        point3f[] points = [(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)]
        int4[] tetVertexIndices = [(0, 1, 2, 3)]
    }
}
"""
        )

        source_tetmesh = newton.TetMesh(
            vertices=np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32),
            tet_indices=np.array([0, 1, 2, 3], dtype=np.int32),
            custom_attributes={
                "temperature": (
                    np.array([100.0, 200.0, 300.0, 400.0], dtype=np.float32),
                    newton.Model.AttributeFrequency.PARTICLE,
                ),
                "regionId": (
                    np.array([7], dtype=np.int32),
                    newton.Model.AttributeFrequency.TETRAHEDRON,
                ),
            },
        )

        builder = newton.ModelBuilder()
        builder.add_custom_attribute(
            newton.ModelBuilder.CustomAttribute(
                name="temperature",
                dtype=wp.float32,
                frequency=newton.Model.AttributeFrequency.PARTICLE,
            )
        )

        with mock.patch("newton._src.utils.import_usd.usd.get_tetmesh", return_value=source_tetmesh):
            builder.add_usd(stage)

        self.assertEqual(set(source_tetmesh.custom_attributes), {"temperature", "regionId"})
        model = builder.finalize()
        self.assertTrue(hasattr(model, "temperature"))
        self.assertFalse(hasattr(model, "regionId"))

    def test_mesh_create_from_file_obj(self):
        """Test Mesh.create_from_file with an OBJ file."""

        # Write a minimal OBJ file (single triangle)
        obj_content = "v 0.0 0.0 0.0\nv 1.0 0.0 0.0\nv 0.0 1.0 0.0\nf 1 2 3\n"

        with tempfile.NamedTemporaryFile(suffix=".obj", delete=False, mode="w") as f:
            f.write(obj_content)
            path = f.name

        try:
            mesh = newton.Mesh.create_from_file(path)

            self.assertIsInstance(mesh, newton.Mesh)
            self.assertEqual(len(mesh.vertices), 3)
            self.assertEqual(len(mesh.indices), 3)
        finally:
            os.unlink(path)

    def test_mesh_create_from_file_not_found(self):
        """Test Mesh.create_from_file raises on missing file."""
        with self.assertRaises(FileNotFoundError):
            newton.Mesh.create_from_file("nonexistent_file.obj")

    def test_tetmesh_create_from_file_not_found(self):
        """Test TetMesh.create_from_file raises on missing file."""
        with self.assertRaises(FileNotFoundError):
            newton.TetMesh.create_from_file("nonexistent_file.vtk")

    # ------------------------------------------------------------------
    # add_soft_mesh(mesh=TetMesh) builder integration
    # ------------------------------------------------------------------

    def _make_two_tet_mesh(self, **kwargs):
        """Helper: 5 vertices, 2 tets sharing face (1,2,3)."""
        vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1]], dtype=np.float32)
        tet_indices = np.array([0, 1, 2, 3, 1, 2, 3, 4], dtype=np.int32)
        return newton.TetMesh(vertices, tet_indices, **kwargs)

    def test_add_soft_mesh_with_tetmesh(self):
        """Test add_soft_mesh accepts a TetMesh and populates the builder."""
        tm = self._make_two_tet_mesh()
        builder = newton.ModelBuilder()
        builder.add_soft_mesh(
            pos=(0.0, 0.0, 0.0),
            rot=wp.quat_identity(),
            scale=1.0,
            vel=(0.0, 0.0, 0.0),
            mesh=tm,
        )
        self.assertEqual(len(builder.particle_q), 5)
        self.assertEqual(len(builder.tet_indices), 2)
        # 6 boundary triangles (2 tets * 4 faces - 2 shared)
        self.assertEqual(len(builder.tri_indices), 6)

    def test_add_soft_mesh_tetmesh_density_override(self):
        """Test that explicit density overrides TetMesh density."""
        tm = self._make_two_tet_mesh(density=10.0)

        # Build with TetMesh density (10.0)
        builder_base = newton.ModelBuilder()
        builder_base.add_soft_mesh(
            pos=(0.0, 0.0, 0.0),
            rot=wp.quat_identity(),
            scale=1.0,
            vel=(0.0, 0.0, 0.0),
            mesh=tm,
        )
        mass_base = sum(builder_base.particle_mass)

        # Build with overridden density (99.0)
        builder_override = newton.ModelBuilder()
        builder_override.add_soft_mesh(
            pos=(0.0, 0.0, 0.0),
            rot=wp.quat_identity(),
            scale=1.0,
            vel=(0.0, 0.0, 0.0),
            mesh=tm,
            density=99.0,
        )
        mass_override = sum(builder_override.particle_mass)

        # Mass should scale with density ratio
        self.assertGreater(mass_base, 0.0)
        self.assertAlmostEqual(mass_override / mass_base, 99.0 / 10.0, places=4)

    def test_add_soft_mesh_tetmesh_per_element_materials(self):
        """Test per-element material arrays flow through to the builder."""
        tm = self._make_two_tet_mesh(
            k_mu=np.array([100.0, 200.0], dtype=np.float32),
            k_lambda=np.array([300.0, 400.0], dtype=np.float32),
            k_damp=np.array([0.1, 0.2], dtype=np.float32),
        )
        builder = newton.ModelBuilder()
        builder.add_soft_mesh(
            pos=(0.0, 0.0, 0.0),
            rot=wp.quat_identity(),
            scale=1.0,
            vel=(0.0, 0.0, 0.0),
            mesh=tm,
        )
        # Verify per-element values are stored
        self.assertAlmostEqual(builder.tet_materials[0][0], 100.0)
        self.assertAlmostEqual(builder.tet_materials[1][0], 200.0)
        self.assertAlmostEqual(builder.tet_materials[0][1], 300.0)
        self.assertAlmostEqual(builder.tet_materials[1][1], 400.0)

    def test_add_soft_mesh_backward_compat(self):
        """Test raw vertices/indices still work (backward compatibility)."""
        vertices = [(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)]
        indices = [0, 1, 2, 3]
        builder = newton.ModelBuilder()
        builder.add_soft_mesh(
            pos=(0.0, 0.0, 0.0),
            rot=wp.quat_identity(),
            scale=1.0,
            vel=(0.0, 0.0, 0.0),
            vertices=vertices,
            indices=indices,
            density=1.0,
            k_mu=1000.0,
            k_lambda=1000.0,
            k_damp=0.0,
        )
        self.assertEqual(len(builder.particle_q), 4)
        self.assertEqual(len(builder.tet_indices), 1)
        self.assertEqual(len(builder.tri_indices), 4)

    def test_add_soft_mesh_no_input_raises(self):
        """Test ValueError when neither mesh nor vertices/indices provided."""
        builder = newton.ModelBuilder()
        with self.assertRaises(ValueError):
            builder.add_soft_mesh(
                pos=(0.0, 0.0, 0.0),
                rot=wp.quat_identity(),
                scale=1.0,
                vel=(0.0, 0.0, 0.0),
            )

    def test_add_soft_mesh_invalid_mesh_type(self):
        """Test TypeError when mesh is not a TetMesh."""
        builder = newton.ModelBuilder()
        with self.assertRaises(TypeError):
            builder.add_soft_mesh(
                pos=(0.0, 0.0, 0.0),
                rot=wp.quat_identity(),
                scale=1.0,
                vel=(0.0, 0.0, 0.0),
                mesh="not_a_tetmesh",
            )

    def test_add_soft_mesh_instancing(self):
        """Test adding the same TetMesh twice creates independent instances."""
        tm = self._make_two_tet_mesh(k_mu=500.0, k_lambda=500.0, density=1.0)
        builder = newton.ModelBuilder()
        builder.add_soft_mesh(
            pos=(0.0, 0.0, 0.0),
            rot=wp.quat_identity(),
            scale=1.0,
            vel=(0.0, 0.0, 0.0),
            mesh=tm,
        )
        builder.add_soft_mesh(
            pos=(2.0, 0.0, 0.0),
            rot=wp.quat_identity(),
            scale=1.0,
            vel=(0.0, 0.0, 0.0),
            mesh=tm,
        )
        self.assertEqual(len(builder.particle_q), 10)
        self.assertEqual(len(builder.tet_indices), 4)
        self.assertEqual(len(builder.tri_indices), 12)


class TestResolveUsdFromUrl(unittest.TestCase):
    """Tests for recursive USD reference resolution in :func:`resolve_usd_from_url`."""

    @staticmethod
    def _cache_path_for_absolute_reference(url: str) -> str:
        """Return the expected safe cache-relative path for an absolute URL."""
        parsed = urlparse(url)
        basename = posixpath.basename(parsed.path) or "reference.usd"
        digest = hashlib.sha256(url.encode("utf-8")).hexdigest()[:16]
        return posixpath.join("_external_usd", digest, basename)

    def _run_resolve(self, url_to_layer, base_url="https://example.com/assets/scene.usd"):
        """Run resolve_usd_from_url with mocked network and USD stage I/O.

        Args:
            url_to_layer: mapping from URL to USDA layer string content.
            base_url: the top-level URL passed to resolve_usd_from_url.

        Returns:
            Tuple of (result_path, target_dir, downloaded_urls).
        """
        downloaded_urls = []

        def fake_get(url, **_kwargs):
            downloaded_urls.append(url)
            resp = mock.MagicMock()
            resp.url = url
            resp.headers = {}
            layer = url_to_layer.get(url)
            if isinstance(layer, tuple) and layer[0] == "redirect":
                resp.status_code = 302
                resp.headers = {"Location": layer[1]}
                resp.content = b""
                return resp
            if layer is None:
                resp.status_code = 404
                return resp
            resp.status_code = 200
            resp.content = layer.encode("utf-8")
            return resp

        # Map cache-relative path -> layer string so the mock stage can return it.
        file_to_layer = {}
        tmpdir = tempfile.mkdtemp()

        # Precompute exact local-key -> layer mapping from URLs.
        base_url_dir = base_url.rsplit("/", 1)[0]
        local_key_to_layer = {}

        for url, layer in url_to_layer.items():
            if isinstance(layer, str) and url.startswith(base_url_dir + "/"):
                local_key_to_layer[url[len(base_url_dir) + 1 :]] = layer
            if isinstance(layer, str) and url.startswith("https://"):
                local_key_to_layer[self._cache_path_for_absolute_reference(url)] = layer

        def _local_key(path):
            return os.path.relpath(path, tmpdir).replace(os.sep, "/")

        def fake_stage_open(path, _load_policy):
            layer_str = file_to_layer.get(_local_key(path), "")
            stage = mock.MagicMock()
            stage.GetRootLayer().ExportToString.return_value = layer_str
            return stage

        # Track writes so we can populate file_to_layer when the function
        # writes a downloaded file to disk.
        real_open = open

        def tracking_open(path, mode="r", **kwargs):
            fh = real_open(path, mode, **kwargs)
            if "w" in mode or "b" in mode:
                key = _local_key(path)
                if key in local_key_to_layer:
                    file_to_layer[key] = local_key_to_layer[key]
            return fh

        mock_requests = mock.MagicMock()
        mock_requests.get = fake_get

        mock_usd = mock.MagicMock()
        mock_usd.Stage.Open = fake_stage_open
        mock_usd.Stage.LoadNone = None

        mock_pxr = mock.MagicMock()
        mock_pxr.Usd = mock_usd

        from newton._src.utils.import_usd import resolve_usd_from_url  # noqa: PLC0415

        with (
            mock.patch.dict(
                "sys.modules",
                {"requests": mock_requests, "pxr": mock_pxr, "pxr.Usd": mock_usd},
            ),
            mock.patch("builtins.open", tracking_open),
        ):
            result = resolve_usd_from_url(base_url, target_folder_name=tmpdir)

        return result, tmpdir, downloaded_urls

    def test_single_level_references(self):
        """References in the root stage are downloaded."""
        url_to_layer = {
            "https://example.com/assets/scene.usd": "references = @./child_a.usd@\nreferences = @./child_b.usd@",
            "https://example.com/assets/child_a.usd": "",
            "https://example.com/assets/child_b.usd": "",
        }
        _result, tmpdir, downloaded_urls = self._run_resolve(url_to_layer)
        self.assertIn("https://example.com/assets/child_a.usd", downloaded_urls)
        self.assertIn("https://example.com/assets/child_b.usd", downloaded_urls)
        self.assertTrue(os.path.exists(os.path.join(tmpdir, "child_a.usd")))
        self.assertTrue(os.path.exists(os.path.join(tmpdir, "child_b.usd")))

    def test_recursive_references(self):
        """References in child stages are resolved recursively."""
        url_to_layer = {
            "https://example.com/assets/scene.usd": "references = @./robot.usd@",
            "https://example.com/assets/robot.usd": "references = @./collisions.usd@",
            "https://example.com/assets/collisions.usd": "",
        }
        _result, tmpdir, downloaded_urls = self._run_resolve(url_to_layer)
        self.assertIn("https://example.com/assets/robot.usd", downloaded_urls)
        self.assertIn("https://example.com/assets/collisions.usd", downloaded_urls)
        self.assertTrue(os.path.exists(os.path.join(tmpdir, "collisions.usd")))

    def test_deep_recursive_references(self):
        """Three levels of nesting are resolved."""
        url_to_layer = {
            "https://example.com/assets/scene.usd": "references = @./level1.usd@",
            "https://example.com/assets/level1.usd": "references = @./level2.usd@",
            "https://example.com/assets/level2.usd": "references = @./level3.usd@",
            "https://example.com/assets/level3.usd": "",
        }
        _result, tmpdir, downloaded_urls = self._run_resolve(url_to_layer)
        self.assertIn("https://example.com/assets/level3.usd", downloaded_urls)
        self.assertTrue(os.path.exists(os.path.join(tmpdir, "level3.usd")))

    def test_no_duplicate_downloads(self):
        """The same reference appearing in multiple stages is downloaded only once."""
        url_to_layer = {
            "https://example.com/assets/scene.usd": "references = @./a.usd@\nreferences = @./b.usd@",
            "https://example.com/assets/a.usd": "references = @./shared.usd@",
            "https://example.com/assets/b.usd": "references = @./shared.usd@",
            "https://example.com/assets/shared.usd": "",
        }
        _result, _tmpdir, downloaded_urls = self._run_resolve(url_to_layer)
        shared_downloads = [u for u in downloaded_urls if u.endswith("shared.usd")]
        self.assertEqual(len(shared_downloads), 1)

    def test_cyclic_references(self):
        """Cyclic references (including back to root) do not cause infinite recursion."""
        url_to_layer = {
            "https://example.com/assets/scene.usd": "references = @./a.usd@",
            "https://example.com/assets/a.usd": "references = @./b.usd@",
            "https://example.com/assets/b.usd": "references = @./scene.usd@",
        }
        _result, _tmpdir, downloaded_urls = self._run_resolve(url_to_layer)
        # Root URL is fetched once at the top level; recursive refs back to it must not re-download.
        self.assertEqual(downloaded_urls.count("https://example.com/assets/scene.usd"), 1)
        self.assertEqual(downloaded_urls.count("https://example.com/assets/a.usd"), 1)
        self.assertEqual(downloaded_urls.count("https://example.com/assets/b.usd"), 1)

    def test_nested_subdirectory_references(self):
        """References in subdirectories preserve correct local paths."""
        url_to_layer = {
            "https://example.com/assets/scene.usd": "references = @robots/robot.usd@",
            "https://example.com/assets/robots/robot.usd": "references = @./collisions.usd@",
            "https://example.com/assets/robots/collisions.usd": "",
        }
        _result, tmpdir, downloaded_urls = self._run_resolve(url_to_layer)
        self.assertIn("https://example.com/assets/robots/collisions.usd", downloaded_urls)
        # collisions.usd must be inside robots/, not at cache root
        self.assertTrue(os.path.exists(os.path.join(tmpdir, "robots", "collisions.usd")))
        self.assertFalse(os.path.exists(os.path.join(tmpdir, "collisions.usd")))

    def test_path_traversal_rejected(self):
        """References with .. that escape the target folder are skipped."""
        url_to_layer = {
            "https://example.com/assets/scene.usd": "references = @../secret.usd@",
        }
        _result, tmpdir, downloaded_urls = self._run_resolve(url_to_layer)
        # Escaped reference must not be fetched or written.
        escaped_urls = [u for u in downloaded_urls if "secret.usd" in u]
        self.assertEqual(len(escaped_urls), 0)
        self.assertFalse(os.path.exists(os.path.join(tmpdir, "..", "secret.usd")))

    def test_cleartext_top_level_url_rejected(self):
        """Top-level USD downloads must use HTTPS."""
        with self.assertRaisesRegex(ValueError, "USD URL downloads require HTTPS"):
            self._run_resolve({}, base_url="http://example.com/assets/scene.usd")

    def test_cleartext_reference_url_rejected(self):
        """Absolute HTTP references are rejected before download."""
        url_to_layer = {
            "https://example.com/assets/scene.usd": "references = @http://example.com/assets/child.usd@",
        }
        with self.assertRaisesRegex(ValueError, "USD URL downloads require HTTPS"):
            self._run_resolve(url_to_layer)

    def test_absolute_https_reference_cached_safely(self):
        """Absolute HTTPS references are cached under a relative path."""
        child_url = "https://cdn.example.com/assets/child.usd"
        url_to_layer = {
            "https://example.com/assets/scene.usd": f"references = @{child_url}@",
            child_url: "",
        }
        result, tmpdir, downloaded_urls = self._run_resolve(url_to_layer)
        local_ref = self._cache_path_for_absolute_reference(child_url)

        self.assertIn(child_url, downloaded_urls)
        self.assertTrue(os.path.exists(os.path.join(tmpdir, local_ref)))
        with open(result) as f:
            rewritten_layer = f.read()
        self.assertIn(f"@{local_ref}@", rewritten_layer)
        self.assertNotIn(child_url, rewritten_layer)

    def test_cleartext_redirect_url_rejected(self):
        """Redirects to HTTP targets are rejected before following them."""
        url_to_layer = {
            "https://example.com/assets/scene.usd": ("redirect", "http://example.com/assets/scene.usd"),
        }
        with self.assertRaisesRegex(ValueError, "USD URL downloads require HTTPS"):
            self._run_resolve(url_to_layer)

    def test_https_redirect_url_followed(self):
        """Redirects to HTTPS targets are followed."""
        url_to_layer = {
            "https://example.com/assets/scene.usd": ("redirect", "https://cdn.example.com/assets/scene.usd"),
            "https://cdn.example.com/assets/scene.usd": "",
        }
        _result, _tmpdir, downloaded_urls = self._run_resolve(url_to_layer)
        self.assertEqual(
            downloaded_urls,
            ["https://example.com/assets/scene.usd", "https://cdn.example.com/assets/scene.usd"],
        )


class TestUsdMaterialColorSpaces(unittest.TestCase):
    def test_texture_color_space_auto_uses_file_attribute_fallback(self):
        from newton._src.usd.utils import _get_texture_source_color_space  # noqa: PLC0415

        shader = mock.Mock()
        source_color_space_input = mock.Mock()
        source_color_space_input.Get.return_value = "auto"
        shader.GetInput.return_value = source_color_space_input

        file_attr = mock.Mock()
        file_attr.GetColorSpace.return_value = "raw"

        self.assertEqual(_get_texture_source_color_space(shader, file_attr), "raw")

    @unittest.skipUnless(USD_AVAILABLE, "Requires usd-core")
    def test_preview_surface_color_is_converted_to_display_space(self):
        from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade

        stage = Usd.Stage.CreateInMemory()
        mesh = UsdGeom.Mesh.Define(stage, "/World/Mesh")
        mesh.GetPointsAttr().Set(
            [
                Gf.Vec3f(0.0, 0.0, 0.0),
                Gf.Vec3f(1.0, 0.0, 0.0),
                Gf.Vec3f(0.0, 1.0, 0.0),
            ]
        )
        mesh.GetFaceVertexCountsAttr().Set([3])
        mesh.GetFaceVertexIndicesAttr().Set([0, 1, 2])

        material = UsdShade.Material.Define(stage, "/World/Looks/Material")
        shader = UsdShade.Shader.Define(stage, "/World/Looks/Material/PreviewSurface")
        shader.CreateIdAttr("UsdPreviewSurface")
        linear_color = (0.25, 0.5, 0.75)
        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*linear_color))
        material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
        UsdShade.MaterialBindingAPI(mesh).Bind(material)

        from newton._src.usd.utils import resolve_material_properties_for_prim  # noqa: PLC0415

        material_props = resolve_material_properties_for_prim(mesh.GetPrim())

        np.testing.assert_allclose(
            material_props["color"],
            newton.utils.color_linear_to_srgb(linear_color),
            atol=1e-6,
        )

    @unittest.skipUnless(USD_AVAILABLE, "Requires usd-core")
    def test_ancestor_binding_overrides_unapplied_mesh_binding(self):
        """An ancestor bind with strongerThanDescendants wins over a mesh's own binding.

        Many assets author ``material:binding`` on meshes without applying MaterialBindingAPI;
        resolution must still honor a stronger ancestor override (e.g. domain-randomization
        material rebinding on the asset root).
        """
        from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade

        stage = Usd.Stage.CreateInMemory()
        root = UsdGeom.Xform.Define(stage, "/World/Visuals")
        mesh = UsdGeom.Mesh.Define(stage, "/World/Visuals/Mesh")
        mesh.GetPointsAttr().Set([Gf.Vec3f(0, 0, 0), Gf.Vec3f(1, 0, 0), Gf.Vec3f(0, 1, 0)])
        mesh.GetFaceVertexCountsAttr().Set([3])
        mesh.GetFaceVertexIndicesAttr().Set([0, 1, 2])

        def define_material(path: str, color: tuple[float, float, float]) -> UsdShade.Material:
            material = UsdShade.Material.Define(stage, path)
            shader = UsdShade.Shader.Define(stage, f"{path}/PreviewSurface")
            shader.CreateIdAttr("UsdPreviewSurface")
            shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
            material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
            return material

        original = define_material("/World/Looks/Original", (1.0, 0.0, 0.0))
        override = define_material("/World/Looks/Override", (0.0, 1.0, 0.0))

        # mesh's own binding: a bare relationship, MaterialBindingAPI deliberately NOT applied
        mesh.GetPrim().CreateRelationship("material:binding").SetTargets([original.GetPrim().GetPath()])
        # ancestor override with descendant-winning strength
        ancestor_binding = UsdShade.MaterialBindingAPI.Apply(root.GetPrim())
        ancestor_binding.Bind(override, bindingStrength=UsdShade.Tokens.strongerThanDescendants)

        from newton._src.usd.utils import resolve_material_properties_for_prim  # noqa: PLC0415

        material_props = resolve_material_properties_for_prim(mesh.GetPrim())

        np.testing.assert_allclose(
            material_props["color"],
            newton.utils.color_linear_to_srgb((0.0, 1.0, 0.0)),
            atol=1e-6,
        )

    @unittest.skipUnless(USD_AVAILABLE, "Requires usd-core")
    def test_ancestor_binding_overrides_applied_mesh_binding_across_depth(self):
        """A grandparent strongerThanDescendants bind wins over a mesh's properly applied binding."""
        from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade

        stage = Usd.Stage.CreateInMemory()
        grandparent = UsdGeom.Xform.Define(stage, "/World/Robot")
        UsdGeom.Xform.Define(stage, "/World/Robot/link")
        mesh = UsdGeom.Mesh.Define(stage, "/World/Robot/link/Mesh")
        mesh.GetPointsAttr().Set([Gf.Vec3f(0, 0, 0), Gf.Vec3f(1, 0, 0), Gf.Vec3f(0, 1, 0)])
        mesh.GetFaceVertexCountsAttr().Set([3])
        mesh.GetFaceVertexIndicesAttr().Set([0, 1, 2])

        def define_material(path: str, color: tuple[float, float, float]) -> UsdShade.Material:
            material = UsdShade.Material.Define(stage, path)
            shader = UsdShade.Shader.Define(stage, f"{path}/PreviewSurface")
            shader.CreateIdAttr("UsdPreviewSurface")
            shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
            material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
            return material

        original = define_material("/World/Looks/Original", (1.0, 0.0, 0.0))
        override = define_material("/World/Looks/Override", (0.0, 1.0, 0.0))

        # mesh's own binding: MaterialBindingAPI properly applied this time
        UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(original)
        # override two levels up, exercising resolution beyond the immediate parent
        UsdShade.MaterialBindingAPI.Apply(grandparent.GetPrim()).Bind(
            override, bindingStrength=UsdShade.Tokens.strongerThanDescendants
        )

        from newton._src.usd.utils import resolve_material_properties_for_prim  # noqa: PLC0415

        material_props = resolve_material_properties_for_prim(mesh.GetPrim())

        np.testing.assert_allclose(
            material_props["color"],
            newton.utils.color_linear_to_srgb((0.0, 1.0, 0.0)),
            atol=1e-6,
        )

    @unittest.skipUnless(USD_AVAILABLE, "Requires usd-core")
    def test_collection_based_ancestor_binding_resolves(self):
        """Collection-based ancestor rebinds resolve through canonical ComputeBoundMaterial."""
        from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade

        stage = Usd.Stage.CreateInMemory()
        root = UsdGeom.Xform.Define(stage, "/World/Robot")
        mesh = UsdGeom.Mesh.Define(stage, "/World/Robot/Mesh")
        mesh.GetPointsAttr().Set([Gf.Vec3f(0, 0, 0), Gf.Vec3f(1, 0, 0), Gf.Vec3f(0, 1, 0)])
        mesh.GetFaceVertexCountsAttr().Set([3])
        mesh.GetFaceVertexIndicesAttr().Set([0, 1, 2])

        material = UsdShade.Material.Define(stage, "/World/Looks/CollectionBound")
        shader = UsdShade.Shader.Define(stage, "/World/Looks/CollectionBound/PreviewSurface")
        shader.CreateIdAttr("UsdPreviewSurface")
        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0.0, 0.0, 1.0))
        material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")

        collection = Usd.CollectionAPI.Apply(root.GetPrim(), "blueParts")
        collection.CreateIncludesRel().AddTarget(mesh.GetPrim().GetPath())
        UsdShade.MaterialBindingAPI.Apply(root.GetPrim()).Bind(
            collection, material, "blueParts", bindingStrength=UsdShade.Tokens.strongerThanDescendants
        )

        from newton._src.usd.utils import resolve_material_properties_for_prim  # noqa: PLC0415

        material_props = resolve_material_properties_for_prim(mesh.GetPrim())

        np.testing.assert_allclose(
            material_props["color"],
            newton.utils.color_linear_to_srgb((0.0, 0.0, 1.0)),
            atol=1e-6,
        )

    @unittest.skipUnless(USD_AVAILABLE, "Requires usd-core")
    def test_preview_surface_color_space_api_display_color_is_not_converted(self):
        from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade

        stage = Usd.Stage.CreateInMemory()
        mesh = UsdGeom.Mesh.Define(stage, "/World/Mesh")
        mesh.GetPointsAttr().Set(
            [
                Gf.Vec3f(0.0, 0.0, 0.0),
                Gf.Vec3f(1.0, 0.0, 0.0),
                Gf.Vec3f(0.0, 1.0, 0.0),
            ]
        )
        mesh.GetFaceVertexCountsAttr().Set([3])
        mesh.GetFaceVertexIndicesAttr().Set([0, 1, 2])

        material = UsdShade.Material.Define(stage, "/World/Looks/Material")
        shader = UsdShade.Shader.Define(stage, "/World/Looks/Material/PreviewSurface")
        shader.CreateIdAttr("UsdPreviewSurface")
        display_color = (0.25, 0.5, 0.75)
        color_input = shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f)
        color_input.Set(Gf.Vec3f(*display_color))
        Usd.ColorSpaceAPI.Apply(shader.GetPrim()).CreateColorSpaceNameAttr().Set("srgb_rec709_scene")
        material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
        UsdShade.MaterialBindingAPI(mesh).Bind(material)

        from newton._src.usd.utils import resolve_material_properties_for_prim  # noqa: PLC0415

        self.assertEqual(
            Usd.ColorSpaceAPI.ComputeColorSpaceName(color_input.GetAttr(), None),
            "srgb_rec709_scene",
        )

        material_props = resolve_material_properties_for_prim(mesh.GetPrim())

        np.testing.assert_allclose(
            material_props["color"],
            display_color,
            atol=1e-6,
        )


if __name__ == "__main__":
    unittest.main(verbosity=2, failfast=False)
