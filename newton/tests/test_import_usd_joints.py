# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import math
import os
import unittest

import numpy as np

import newton
from newton._src.geometry.flags import ShapeFlags
from newton._src.solvers.mujoco.constants import (
    SOLREF_MODE_FORCE_SPACE,
    SOLREF_MODE_MJCF_DEFAULT,
    SOLREF_MODE_RAW,
)
from newton.solvers import SolverMuJoCo
from newton.tests.unittest_utils import USD_AVAILABLE


class TestImportUsdJoints(unittest.TestCase):
    @unittest.skipUnless(USD_AVAILABLE, "Requires usd-core")
    def test_scene_drive_scale_and_per_import_joint_defaults(self):
        """Use scene drive scaling and keep joint defaults scoped to each import."""
        from pxr import Sdf, Usd, UsdGeom, UsdPhysics

        stage = Usd.Stage.CreateInMemory()
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
        UsdGeom.SetStageMetersPerUnit(stage, 1.0)
        scene = UsdPhysics.Scene.Define(stage, "/physicsScene")
        scene.GetPrim().CreateAttribute("newton:joint_drive_gains_scaling", Sdf.ValueTypeNames.Float).Set(2.5)
        root = UsdGeom.Xform.Define(stage, "/Articulation")
        UsdPhysics.ArticulationRootAPI.Apply(root.GetPrim())
        for name in ("Parent", "Child"):
            body = UsdGeom.Cube.Define(stage, f"/Articulation/{name}")
            UsdPhysics.RigidBodyAPI.Apply(body.GetPrim())
            UsdPhysics.CollisionAPI.Apply(body.GetPrim())
        joint = UsdPhysics.RevoluteJoint.Define(stage, "/Articulation/Joint")
        joint.CreateBody0Rel().SetTargets(["/Articulation/Parent"])
        joint.CreateBody1Rel().SetTargets(["/Articulation/Child"])
        drive = UsdPhysics.DriveAPI.Apply(joint.GetPrim(), "angular")
        drive.CreateStiffnessAttr(4.0)
        drive.CreateDampingAttr(0.5)

        for default in (1.0, 3.0):
            with self.subTest(default=default):
                builder = newton.ModelBuilder()
                builder.default_joint_cfg.armature = default
                builder.default_joint_cfg.friction = default / 10.0
                builder.default_joint_cfg.damping = default * 2.0
                builder.default_joint_cfg.limit_ke = default * 100.0
                builder.default_joint_cfg.limit_kd = default * 5.0
                builder.add_usd(stage, joint_drive_gains_scaling=9.0)
                dof = builder.joint_qd_start[builder.joint_label.index("/Articulation/Joint")]
                self.assertAlmostEqual(builder.joint_target_ke[dof], 4.0 * math.degrees(2.5))
                self.assertAlmostEqual(builder.joint_target_kd[dof], 0.5 * math.degrees(2.5))
                self.assertAlmostEqual(builder.joint_armature[dof], default)
                self.assertAlmostEqual(builder.joint_friction[dof], default / 10.0)
                self.assertAlmostEqual(builder.joint_damping[dof], default * 2.0)
                self.assertAlmostEqual(builder.joint_limit_ke[dof], default * 100.0)
                self.assertAlmostEqual(builder.joint_limit_kd[dof], default * 5.0)

    @unittest.skipUnless(USD_AVAILABLE, "Requires usd-core")
    def test_distance_joint(self):
        """Import independently enabled distance-joint limits."""
        from pxr import Usd, UsdGeom, UsdPhysics

        def import_limits(min_distance=None, max_distance=None):
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
            if min_distance is not None:
                joint.CreateMinDistanceAttr(min_distance)
            if max_distance is not None:
                joint.CreateMaxDistanceAttr(max_distance)

            builder = newton.ModelBuilder()
            builder.add_usd(stage)

            joint_index = builder.joint_label.index("/World/DistanceJoint")
            dof_index = builder.joint_qd_start[joint_index]
            return builder.joint_limit_lower[dof_index], builder.joint_limit_upper[dof_index]

        cases = (
            ("no limits", None, None, -1.0, -1.0),
            ("minimum only", 0.25, None, 0.25, -1.0),
            ("maximum only", None, 1.5, -1.0, 1.5),
            ("both limits", 0.25, 1.5, 0.25, 1.5),
            ("disabled minimum", -0.25, None, -1.0, -1.0),
            ("disabled maximum", None, -0.25, -1.0, -1.0),
        )
        for name, min_distance, max_distance, expected_min, expected_max in cases:
            with self.subTest(name=name):
                self.assertEqual(import_limits(min_distance, max_distance), (expected_min, expected_max))

    @unittest.skipUnless(USD_AVAILABLE, "Requires usd-core")
    def test_joint_collision_enabled(self):
        """Respect joint collision flags and the self-collision override."""
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
    def test_collision_filter_pairs_reference_only_colliding_shapes(self):
        """Generate USD collision filters only for colliding shapes."""
        from pxr import Usd, UsdGeom, UsdPhysics

        stage = Usd.Stage.CreateInMemory()
        articulation = UsdGeom.Xform.Define(stage, "/World")
        UsdPhysics.ArticulationRootAPI.Apply(articulation.GetPrim())

        bodies = []
        for body_name in ("Body0", "Body1"):
            body = UsdGeom.Xform.Define(stage, f"/World/{body_name}")
            UsdPhysics.RigidBodyAPI.Apply(body.GetPrim())
            bodies.append(body)
            for shape_name, collision_enabled in (("Collider", True), ("Visual", False)):
                shape = UsdGeom.Cube.Define(stage, f"/World/{body_name}/{shape_name}")
                collision = UsdPhysics.CollisionAPI.Apply(shape.GetPrim())
                collision.GetCollisionEnabledAttr().Set(collision_enabled)

        joint = UsdPhysics.RevoluteJoint.Define(stage, "/World/Joint")
        joint.CreateBody0Rel().SetTargets([bodies[0].GetPath()])
        joint.CreateBody1Rel().SetTargets([bodies[1].GetPath()])

        builder = newton.ModelBuilder()
        builder.add_usd(
            stage,
            enable_self_collisions=False,
            load_visual_shapes=False,
        )

        expected_colliding = {
            builder.shape_label.index("/World/Body0/Collider"),
            builder.shape_label.index("/World/Body1/Collider"),
        }
        colliding = {
            shape for shape in range(builder.shape_count) if builder.shape_flags[shape] & ShapeFlags.COLLIDE_SHAPES
        }
        self.assertEqual(colliding, expected_colliding)
        particle_colliding = {
            shape for shape in range(builder.shape_count) if builder.shape_flags[shape] & ShapeFlags.COLLIDE_PARTICLES
        }
        self.assertEqual(particle_colliding, expected_colliding)
        filter_pairs = set(builder.shape_collision_filter_pairs)
        self.assertEqual(filter_pairs, {tuple(sorted(colliding))})
        for pair in filter_pairs:
            self.assertLessEqual(set(pair), colliding)

    @unittest.skipUnless(USD_AVAILABLE, "Requires usd-core")
    def test_world_joint_does_not_filter_collisions(self):
        """Keep collisions between world-attached bodies and static colliders."""
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
        """Parse NewtonJointAPI broadcast attributes onto a revolute joint, including sentinels."""
        from pxr import Usd

        from newton._src.usd._resolution_policy import _HARD_LIMIT_KE  # noqa: PLC0415

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
        """Parse NewtonJointAPI attributes onto a prismatic joint without per-degree conversion."""
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
        """Use builder.default_joint_cfg.damping unchanged when newton:damping is not authored.

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
        """Use builder default damping unchanged when merging joints into a D6 joint.

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
    def test_merged_joint_gain_edit_before_solver_construction(self):
        """Verify that pre-solver gain edits promote merged USD joints to force space."""
        from pxr import Usd

        from newton._src.usd.schemas import SchemaResolverMjc  # noqa: PLC0415

        stage = Usd.Stage.CreateInMemory()
        stage.GetRootLayer().ImportFromString(
            """#usda 1.0
(
    upAxis = "Z"
)

def PhysicsScene "physicsScene"
{
}

def Xform "World" (
    prepend apiSchemas = ["PhysicsArticulationRootAPI"]
)
{
    def Cube "Body0" (
        prepend apiSchemas = ["PhysicsCollisionAPI", "PhysicsRigidBodyAPI"]
    )
    {
        double size = 0.2
    }

    def Cube "Body1" (
        prepend apiSchemas = ["PhysicsCollisionAPI", "PhysicsRigidBodyAPI"]
    )
    {
        double size = 0.2
        double3 xformOp:translate = (1, 0, 0)
        uniform token[] xformOpOrder = ["xformOp:translate"]
    }

    def PhysicsPrismaticJoint "slide" (
        prepend apiSchemas = ["MjcJointAPI"]
    )
    {
        rel physics:body0 = </World/Body0>
        rel physics:body1 = </World/Body1>
        token physics:axis = "X"
        float physics:lowerLimit = -1
        float physics:upperLimit = 1
    }

    def PhysicsRevoluteJoint "hinge" (
        prepend apiSchemas = ["MjcJointAPI"]
    )
    {
        rel physics:body0 = </World/Body0>
        rel physics:body1 = </World/Body1>
        token physics:axis = "Z"
        float physics:lowerLimit = -45
        float physics:upperLimit = 45
    }
}
"""
        )

        builder = newton.ModelBuilder()
        SolverMuJoCo.register_custom_attributes(builder)
        result = builder.add_usd(stage, schema_resolvers=[SchemaResolverMjc()], load_visual_shapes=False)
        model = builder.finalize(device="cpu")

        merged_joint = result["path_joint_map"]["/World/hinge"]
        self.assertEqual(result["path_joint_map"]["/World/slide"], merged_joint)
        self.assertEqual(model.joint_type.numpy()[merged_joint], newton.JointType.D6)
        dof_start = int(model.joint_qd_start.numpy()[merged_joint])
        dof_slice = slice(dof_start, dof_start + 2)
        np.testing.assert_array_equal(
            model.mujoco.solreflimit_mode.numpy()[dof_slice],
            [SOLREF_MODE_MJCF_DEFAULT, SOLREF_MODE_MJCF_DEFAULT],
        )
        np.testing.assert_allclose(
            model.mujoco.solreflimit_gain_baseline.numpy()[dof_slice],
            [[builder.default_joint_cfg.limit_ke, builder.default_joint_cfg.limit_kd]] * 2,
            rtol=0.0,
            atol=0.0,
        )

        limit_ke = model.joint_limit_ke.numpy()
        limit_kd = model.joint_limit_kd.numpy()
        limit_ke[dof_slice] = [5000.0, 6000.0]
        limit_kd[dof_slice] = [50.0, 60.0]
        model.joint_limit_ke.assign(limit_ke)
        model.joint_limit_kd.assign(limit_kd)
        SolverMuJoCo(model, iterations=1, disable_contacts=True, use_mujoco_cpu=True)

        np.testing.assert_array_equal(
            model.mujoco.solreflimit_mode.numpy()[dof_slice],
            [SOLREF_MODE_FORCE_SPACE, SOLREF_MODE_FORCE_SPACE],
        )

    @unittest.skipUnless(USD_AVAILABLE, "Requires usd-core")
    def test_newton_joint_api_d6(self):
        """Broadcast NewtonJointAPI attributes across a D6 joint's linear and angular DOFs."""
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
        """Use the builder default for newton:velocityLimit=inf rather than storing inf."""
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
        """Select the builder default for authored newton:limitStiffness=-inf.

        Do not fall through to a lower-priority MuJoCo per-DOF gain.
        """
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
    def test_mjc_solreflimit_does_not_replace_newton_gains(self):
        """Keep MuJoCo solref separate from generic Newton limit gains."""
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
        # Native MuJoCo solref remains available to SolverMuJoCo without being
        # converted into the generic gains used by other Newton solvers.
        limit_ke = float(model.joint_limit_ke.numpy()[dof])
        limit_kd = float(model.joint_limit_kd.numpy()[dof])
        self.assertAlmostEqual(limit_ke, 999.0, places=2)
        self.assertAlmostEqual(limit_kd, 88.0, places=2)
        np.testing.assert_allclose(model.mujoco.solreflimit.numpy()[dof], [0.04, 2.0], rtol=1.0e-6, atol=0.0)
        self.assertEqual(int(model.mujoco.solreflimit_mode.numpy()[dof]), SOLREF_MODE_RAW)

    @unittest.skipUnless(USD_AVAILABLE, "Requires usd-core")
    def test_joint_ordering(self):
        """Order ant articulation joints using DFS or BFS traversal."""
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

        for bodies_follow_joint_ordering in (True, False):
            with self.subTest(bodies_follow_joint_ordering=bodies_follow_joint_ordering):
                builder = newton.ModelBuilder()
                with self.assertRaises(ValueError) as exc_info:
                    builder.add_usd(stage, bodies_follow_joint_ordering=bodies_follow_joint_ordering)
                self.assertIn("/World/Articulation/Joint1", str(exc_info.exception))
                # Graph validation happens before deferred bodies, but after eagerly added bodies.
                self.assertEqual(builder.body_count, 0 if bodies_follow_joint_ordering else 3)
                self.assertEqual(builder.joint_count, 0)

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
        """Replace the authored root joint when an explicit floating override is provided."""
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
        """Reject a reversed D6 joint."""
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
        """Reject a reversed spherical joint."""
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
        """Filter ignored joints and bodies and rebuild the affected articulations."""

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
        """Import an articulation with a loop joint marked excludeFromArticulation."""
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
        """Parse solimp_friction attributes from USD."""
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


if __name__ == "__main__":
    unittest.main()
