# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Regression tests for MuJoCo tendon paths imported from USD."""

import unittest

import numpy as np

import newton
from newton.selection import ArticulationView
from newton.solvers import SolverMuJoCo
from newton.tests.unittest_utils import USD_AVAILABLE, assert_np_equal


@unittest.skipUnless(USD_AVAILABLE, "Requires usd-core")
class TestUSDTendonImport(unittest.TestCase):
    SCENE = """#usda 1.0
(
    upAxis = "Z"
    metersPerUnit = 1
)
def Xform "Robot" (prepend apiSchemas = ["PhysicsArticulationRootAPI"])
{
    def Xform "Base" (prepend apiSchemas = ["PhysicsRigidBodyAPI", "PhysicsMassAPI"])
    {
        float physics:mass = 1
        float3 physics:diagonalInertia = (0.1, 0.1, 0.1)
        def Sphere "s0" (prepend apiSchemas = ["MjcSiteAPI"])
        {
            double radius = 0.01
            double3 xformOp:translate = (0, 0, 0.1)
            uniform token[] xformOpOrder = ["xformOp:translate"]
        }
        def Sphere "wrap" (prepend apiSchemas = ["PhysicsCollisionAPI"])
        {
            double radius = 0.05
            double3 xformOp:translate = (0.25, 0, 0.1)
            uniform token[] xformOpOrder = ["xformOp:translate"]
        }
        def Sphere "side" (prepend apiSchemas = ["MjcSiteAPI"])
        {
            double radius = 0.01
            double3 xformOp:translate = (0.25, 0.1, 0.1)
            uniform token[] xformOpOrder = ["xformOp:translate"]
        }
    }
    def Xform "Link" (prepend apiSchemas = ["PhysicsRigidBodyAPI", "PhysicsMassAPI"])
    {
        float physics:mass = 1
        float3 physics:diagonalInertia = (0.1, 0.1, 0.1)
        double3 xformOp:translate = (0.5, 0, 0)
        uniform token[] xformOpOrder = ["xformOp:translate"]
        def Sphere "s1" (prepend apiSchemas = ["MjcSiteAPI"])
        {
            double radius = 0.01
            double3 xformOp:translate = (0, 0, 0.1)
            uniform token[] xformOpOrder = ["xformOp:translate"]
        }
    }
    def PhysicsFixedJoint "Root"
    {
        rel physics:body1 = </Robot/Base>
    }
    def PhysicsRevoluteJoint "Hinge"
    {
        uniform token physics:axis = "Y"
        rel physics:body0 = </Robot/Base>
        rel physics:body1 = </Robot/Link>
        point3f physics:localPos0 = (0.5, 0, 0)
    }
}
"""

    def _create_stage(self):
        from pxr import Usd

        stage = Usd.Stage.CreateInMemory()
        stage.GetRootLayer().ImportFromString(self.SCENE)
        return stage

    def _add_tendon(self, stage, *, name="Tendon", tendon_type="spatial", targets=None, **arrays):
        from pxr import Sdf

        prim = stage.DefinePrim(f"/Robot/{name}", "MjcTendon")
        if tendon_type is not None:
            prim.CreateAttribute("mjc:type", Sdf.ValueTypeNames.Token).Set(tendon_type)
        if targets is None:
            targets = ["/Robot/Base/s0", "/Robot/Link/s1"]
        prim.CreateRelationship("mjc:path").SetTargets(targets)
        for name, value in arrays.items():
            attr_type = Sdf.ValueTypeNames.DoubleArray if name in ("divisors", "coef") else Sdf.ValueTypeNames.IntArray
            prim.CreateAttribute(f"mjc:path:{name}", attr_type).Set(value)
        prim.CreateAttribute("mjc:stiffness", Sdf.ValueTypeNames.Double).Set(10.0)
        prim.CreateAttribute("mjc:damping", Sdf.ValueTypeNames.Double).Set(0.5)
        prim.CreateAttribute("mjc:springlength", Sdf.ValueTypeNames.DoubleArray).Set([0.25, 0.25])
        return prim

    def _import(self, stage, **kwargs):
        builder = newton.ModelBuilder()
        SolverMuJoCo.register_custom_attributes(builder)
        builder.add_usd(stage, **kwargs)
        return builder

    def _add_wrapped_tendon(self, stage):
        from pxr import Sdf

        prim = self._add_tendon(
            stage,
            targets=["/Robot/Link/s1", "/Robot/Base/wrap", "/Robot/Base/s0"],
            indices=[2, 1, 0, 2, 0],
            segments=[0, 0, 0, 1, 1],
            divisors=[1, 2],
        )
        prim.CreateRelationship("mjc:sideSites").SetTargets(["/Robot/Base/side"])
        prim.CreateAttribute("mjc:sideSites:indices", Sdf.ValueTypeNames.IntArray).Set([-1, 0, -1, -1, -1])
        return prim

    def test_fixed_tendon_attributes_and_joint_paths(self):
        """Import fixed tendon attributes and preserve indexed joint paths across tendons."""
        from pxr import Sdf, UsdPhysics, Vt

        stage = self._create_stage()
        layer = stage.GetRootLayer()
        for suffix in ("2", "3"):
            Sdf.CopySpec(layer, "/Robot/Link", layer, f"/Robot/Link{suffix}")
            Sdf.CopySpec(layer, "/Robot/Hinge", layer, f"/Robot/Hinge{suffix}")
            UsdPhysics.RevoluteJoint(stage.GetPrimAtPath(f"/Robot/Hinge{suffix}")).GetBody1Rel().SetTargets(
                [f"/Robot/Link{suffix}"]
            )

        tendon_prim = self._add_tendon(
            stage,
            tendon_type="fixed",
            targets=["/Robot/Hinge", "/Robot/Hinge2"],
            indices=[1, 0],
            coef=[0.25, 0.75],
        )
        tendon_prim.CreateAttribute("mjc:stiffness", Sdf.ValueTypeNames.Double, True).Set(11.0)
        tendon_prim.CreateAttribute("mjc:damping", Sdf.ValueTypeNames.Double, True).Set(0.33)
        tendon_prim.CreateAttribute("mjc:frictionloss", Sdf.ValueTypeNames.Double, True).Set(0.07)
        tendon_prim.CreateAttribute("mjc:limited", Sdf.ValueTypeNames.Token, True).Set("true")
        tendon_prim.CreateAttribute("mjc:range:min", Sdf.ValueTypeNames.Double, True).Set(-0.2)
        tendon_prim.CreateAttribute("mjc:range:max", Sdf.ValueTypeNames.Double, True).Set(0.8)
        tendon_prim.CreateAttribute("mjc:margin", Sdf.ValueTypeNames.Double, True).Set(0.01)
        tendon_prim.CreateAttribute("mjc:solreflimit", Sdf.ValueTypeNames.DoubleArray, True).Set(
            Vt.DoubleArray([0.1, 0.5])
        )
        tendon_prim.CreateAttribute("mjc:solimplimit", Sdf.ValueTypeNames.DoubleArray, True).Set(
            Vt.DoubleArray([0.91, 0.92, 0.003, 0.6, 2.3])
        )
        tendon_prim.CreateAttribute("mjc:solreffriction", Sdf.ValueTypeNames.DoubleArray, True).Set(
            Vt.DoubleArray([0.11, 0.55])
        )
        tendon_prim.CreateAttribute("mjc:solimpfriction", Sdf.ValueTypeNames.DoubleArray, True).Set(
            Vt.DoubleArray([0.81, 0.82, 0.004, 0.7, 2.4])
        )
        tendon_prim.CreateAttribute("mjc:armature", Sdf.ValueTypeNames.Double, True).Set(0.012)
        tendon_prim.CreateAttribute("mjc:springlength", Sdf.ValueTypeNames.DoubleArray, True).Set(
            Vt.DoubleArray([0.13, 0.23])
        )
        tendon_prim.CreateAttribute("mjc:actuatorfrcrange:min", Sdf.ValueTypeNames.Double, True).Set(-4.0)
        tendon_prim.CreateAttribute("mjc:actuatorfrcrange:max", Sdf.ValueTypeNames.Double, True).Set(6.0)
        tendon_prim.CreateAttribute("mjc:actuatorfrclimited", Sdf.ValueTypeNames.Token, True).Set("false")

        self._add_tendon(
            stage,
            name="Tendon2",
            tendon_type="fixed",
            targets=["/Robot/Hinge", "/Robot/Hinge2", "/Robot/Hinge3"],
            indices=[2, 0, 1],
            coef=[0.3, 0.4, 0.5],
        )
        model = self._import(stage).finalize(device="cpu")

        self.assertEqual(model.custom_frequency_counts["mujoco:tendon"], 2)
        self.assertEqual(model.custom_frequency_counts["mujoco:tendon_joint"], 5)
        np.testing.assert_array_equal(model.mujoco.tendon_type.numpy(), [0, 0])
        np.testing.assert_array_equal(model.mujoco.tendon_joint_adr.numpy(), [0, 2])
        np.testing.assert_array_equal(model.mujoco.tendon_joint_num.numpy(), [2, 3])
        joint1, joint2, joint3 = [
            model.joint_label.index(path) for path in ("/Robot/Hinge", "/Robot/Hinge2", "/Robot/Hinge3")
        ]
        np.testing.assert_array_equal(model.mujoco.tendon_joint.numpy(), [joint2, joint1, joint3, joint1, joint2])
        np.testing.assert_allclose(model.mujoco.tendon_coef.numpy(), [0.25, 0.75, 0.3, 0.4, 0.5])
        self.assertAlmostEqual(float(model.mujoco.tendon_stiffness.numpy()[0]), 11.0, places=6)
        self.assertAlmostEqual(float(model.mujoco.tendon_damping.numpy()[0]), 0.33, places=6)
        self.assertAlmostEqual(float(model.mujoco.tendon_frictionloss.numpy()[0]), 0.07, places=6)
        self.assertEqual(int(model.mujoco.tendon_limited.numpy()[0]), 1)
        assert_np_equal(model.mujoco.tendon_range.numpy()[0], np.array([-0.2, 0.8], dtype=np.float32), tol=1e-6)
        self.assertAlmostEqual(float(model.mujoco.tendon_margin.numpy()[0]), 0.01, places=6)
        assert_np_equal(model.mujoco.tendon_solref_limit.numpy()[0], np.array([0.1, 0.5], dtype=np.float32), tol=1e-6)
        assert_np_equal(
            model.mujoco.tendon_solimp_limit.numpy()[0],
            np.array([0.91, 0.92, 0.003, 0.6, 2.3], dtype=np.float32),
            tol=1e-6,
        )
        assert_np_equal(
            model.mujoco.tendon_solref_friction.numpy()[0], np.array([0.11, 0.55], dtype=np.float32), tol=1e-6
        )
        assert_np_equal(
            model.mujoco.tendon_solimp_friction.numpy()[0],
            np.array([0.81, 0.82, 0.004, 0.7, 2.4], dtype=np.float32),
            tol=1e-6,
        )
        self.assertAlmostEqual(float(model.mujoco.tendon_armature.numpy()[0]), 0.012, places=6)
        assert_np_equal(model.mujoco.tendon_springlength.numpy()[0], np.array([0.13, 0.23], dtype=np.float32), tol=1e-6)
        assert_np_equal(
            model.mujoco.tendon_actuator_force_range.numpy()[0], np.array([-4.0, 6.0], dtype=np.float32), tol=1e-6
        )
        self.assertEqual(int(model.mujoco.tendon_actuator_force_limited.numpy()[0]), 0)

    def test_spatial_tendon_defaults_and_export(self):
        """Import explicit and default spatial types and preserve spring forces in MuJoCo."""
        import mujoco

        for tendon_type in ("spatial", None):
            for arrays in ({}, {"indices": [0, 1], "segments": [0, 0], "divisors": [1]}):
                with self.subTest(tendon_type=tendon_type, arrays=arrays):
                    stage = self._create_stage()
                    self._add_tendon(stage, tendon_type=tendon_type, **arrays)
                    model = self._import(stage).finalize(device="cpu")
                    np.testing.assert_array_equal(model.mujoco.tendon_type.numpy(), [1])
                    np.testing.assert_array_equal(model.mujoco.tendon_joint_num.numpy(), [0])
                    np.testing.assert_array_equal(model.mujoco.tendon_wrap_num.numpy(), [2])
                    np.testing.assert_array_equal(model.mujoco.tendon_wrap_type.numpy(), [0, 0])
                    solver = SolverMuJoCo(model, use_mujoco_cpu=True)
                    self.assertEqual(solver.mj_model.ntendon, 1)
                    np.testing.assert_allclose(solver.mj_model.tendon_stiffness, [10])
                    np.testing.assert_allclose(solver.mj_model.tendon_damping, [0.5])
                    data = mujoco.MjData(solver.mj_model)
                    mujoco.mj_forward(solver.mj_model, data)
                    np.testing.assert_allclose(data.ten_length, [0.5])
                    self.assertGreater(np.linalg.norm(data.qfrc_passive), 0.0)

    def test_indexed_wraps_side_sites_and_pulleys(self):
        """Preserve repeated path targets, side sites, and segment pulley divisors."""
        import mujoco

        stage = self._create_stage()
        self._add_wrapped_tendon(stage)
        model = self._import(stage).finalize(device="cpu")
        attrs = model.mujoco
        np.testing.assert_array_equal(attrs.tendon_type.numpy(), [1])
        np.testing.assert_array_equal(attrs.tendon_wrap_num.numpy(), [6])
        np.testing.assert_array_equal(attrs.tendon_wrap_type.numpy(), [0, 1, 0, 2, 0, 0])
        s0, geom, s1, side = [
            model.shape_label.index(path)
            for path in ("/Robot/Base/s0", "/Robot/Base/wrap", "/Robot/Link/s1", "/Robot/Base/side")
        ]
        np.testing.assert_array_equal(attrs.tendon_wrap_shape.numpy(), [s0, geom, s1, -1, s0, s1])
        np.testing.assert_array_equal(attrs.tendon_wrap_sidesite.numpy(), [-1, side, -1, -1, -1, -1])
        np.testing.assert_allclose(attrs.tendon_wrap_prm.numpy(), [0, 0, 0, 2, 0, 0])
        np.testing.assert_array_equal(attrs.tendon_wrap_articulation.numpy(), [0] * 6)
        view = ArticulationView(model, "*")
        self.assertEqual(view.custom_frequency_counts["mujoco:tendon_wrap"], 6)
        solver = SolverMuJoCo(model, use_mujoco_cpu=True)
        self.assertEqual(solver.mj_model.ntendon, 1)
        np.testing.assert_array_equal(
            solver.mj_model.wrap_type,
            [
                mujoco.mjtWrap.mjWRAP_SITE,
                mujoco.mjtWrap.mjWRAP_SPHERE,
                mujoco.mjtWrap.mjWRAP_SITE,
                mujoco.mjtWrap.mjWRAP_PULLEY,
                mujoco.mjtWrap.mjWRAP_SITE,
                mujoco.mjtWrap.mjWRAP_SITE,
            ],
        )
        side_site = solver.mj_model.site(int(solver.mj_model.wrap_prm[1]))
        self.assertTrue(side_site.name.startswith("/Robot/Base/side_"))
        self.assertEqual(solver.mj_model.wrap_prm[3], 2.0)

    def test_cylinder_wrap_and_leading_pulley(self):
        """Import a cylinder without a side site and a pulley before the first path target."""
        import mujoco
        from pxr import Sdf

        stage = self._create_stage()
        geom = stage.GetPrimAtPath("/Robot/Base/wrap")
        geom.SetTypeName("Cylinder")
        geom.CreateAttribute("height", Sdf.ValueTypeNames.Double).Set(0.1)
        self._add_tendon(
            stage,
            targets=["/Robot/Base/s0", "/Robot/Base/wrap", "/Robot/Link/s1"],
            segments=[1, 1, 1],
            divisors=[1, 3],
        )
        model = self._import(stage).finalize(device="cpu")
        np.testing.assert_array_equal(model.mujoco.tendon_wrap_type.numpy(), [2, 0, 1, 0])
        np.testing.assert_array_equal(model.mujoco.tendon_wrap_sidesite.numpy(), [-1] * 4)
        solver = SolverMuJoCo(model, use_mujoco_cpu=True)
        self.assertEqual(solver.mj_model.ntendon, 1)
        self.assertEqual(solver.mj_model.wrap_type[2], mujoco.mjtWrap.mjWRAP_CYLINDER)
        self.assertEqual(solver.mj_model.wrap_prm[0], 3.0)
        self.assertEqual(solver.mj_model.wrap_prm[2], -1.0)

    def test_tendon_actuator_resolution_when_actuator_comes_first(self):
        """Resolve fixed and spatial tendon actuators within the requested USD subtree."""
        import mujoco
        from pxr import Sdf

        for tendon_type in ("fixed", "spatial"):
            for author_trntype in (False, True):
                with self.subTest(tendon_type=tendon_type, author_trntype=author_trntype):
                    stage = self._create_stage()
                    # Author the actuator first to exercise deferred target resolution.
                    actuator = stage.DefinePrim("/Robot/Actuator", "MjcActuator")
                    actuator.CreateRelationship("mjc:target").SetTargets(["/Robot/Tendon"])
                    if author_trntype:
                        actuator.CreateAttribute("mjc:trntype", Sdf.ValueTypeNames.Token).Set("tendon")
                    targets = ["/Robot/Hinge"] if tendon_type == "fixed" else None
                    self._add_tendon(stage, tendon_type=tendon_type, targets=targets)
                    layer = stage.GetRootLayer()
                    Sdf.CopySpec(layer, "/Robot", layer, "/IgnoredRobot")
                    model = self._import(stage, root_path="/Robot").finalize(device="cpu")

                    self.assertEqual(model.custom_frequency_counts["mujoco:tendon"], 1)
                    self.assertEqual(model.mujoco.actuator_target_label[0], "/Robot/Tendon")
                    np.testing.assert_array_equal(model.custom_frequency_articulation["mujoco:tendon"].numpy(), [0])
                    np.testing.assert_array_equal(model.custom_frequency_articulation["mujoco:actuator"].numpy(), [0])
                    frequency = "mujoco:tendon_joint" if tendon_type == "fixed" else "mujoco:tendon_wrap"
                    owners = [0] if tendon_type == "fixed" else [0, 0]
                    np.testing.assert_array_equal(model.custom_frequency_articulation[frequency].numpy(), owners)
                    solver = SolverMuJoCo(model, use_mujoco_cpu=True)
                    self.assertEqual(solver.mj_model.ntendon, 1)
                    self.assertEqual(solver.mj_model.nu, 1)
                    self.assertEqual(solver.mj_model.actuator_trntype[0], mujoco.mjtTrn.mjTRN_TENDON)
                    self.assertEqual(solver.mj_model.actuator_trnid[0, 0], 0)
                    self.assertEqual(solver.mj_model.tendon(0).name, "/Robot/Tendon")

    def test_mixed_tendons_and_repeated_imports(self):
        """Keep fixed and spatial row addresses valid across imports and world replication."""
        stage = self._create_stage()
        self._add_wrapped_tendon(stage)
        self._add_tendon(stage, name="Fixed", tendon_type="fixed", targets=["/Robot/Hinge"], coef=[0.75])
        self._add_tendon(stage, name="Spatial2")
        builder = self._import(stage)
        builder.add_usd(stage)
        model = builder.finalize(device="cpu")
        np.testing.assert_array_equal(model.mujoco.tendon_type.numpy(), [1, 0, 1, 1, 0, 1])
        np.testing.assert_array_equal(model.mujoco.tendon_wrap_adr.numpy(), [0, 6, 6, 8, 14, 14])
        np.testing.assert_array_equal(model.mujoco.tendon_wrap_num.numpy(), [6, 0, 2, 6, 0, 2])
        np.testing.assert_array_equal(model.mujoco.tendon_joint_adr.numpy(), [0, 0, 1, 1, 1, 2])
        np.testing.assert_array_equal(model.mujoco.tendon_joint_num.numpy(), [0, 1, 0, 0, 1, 0])
        shapes = model.mujoco.tendon_wrap_shape.numpy()
        np.testing.assert_array_equal(shapes[8:], np.where(shapes[:8] < 0, -1, shapes[:8] + model.shape_count // 2))
        worlds = newton.ModelBuilder()
        worlds.add_world(builder)
        worlds.add_world(builder)
        replicated = worlds.finalize(device="cpu")
        np.testing.assert_array_equal(replicated.mujoco.tendon_wrap_adr.numpy()[6:], [16, 22, 22, 24, 30, 30])

    def test_invalid_spatial_paths_warn(self):
        """Warn and discard complete paths when spatial tendon metadata or targets are invalid."""
        invalid = [
            {"targets": []},
            {"indices": [-1, 0]},
            {"indices": [0, 2]},
            {"segments": [0]},
            {"segments": [-1, -1], "divisors": [1]},
            {"segments": [1, 0], "divisors": [1, 2]},
            {"segments": [0, 1]},
            {"segments": [0, 1], "divisors": [1]},
            {"segments": [1, 1], "divisors": [1, 0]},
            {"segments": [1, 1], "divisors": [1, float("nan")]},
            {"targets": ["/Robot/Base/s0", "/Robot/Missing"]},
            {"targets": ["/Robot/Base/s0", "/Robot/Hinge"]},
        ]
        for arrays in invalid:
            with self.subTest(arrays=arrays):
                stage = self._create_stage()
                self._add_tendon(stage, **arrays)
                with self.assertWarnsRegex(UserWarning, "Skipping spatial MjcTendon /Robot/Tendon"):
                    builder = self._import(stage)
                model = builder.finalize(device="cpu")
                np.testing.assert_array_equal(model.mujoco.tendon_wrap_num.numpy(), [0])
                self.assertEqual(model.custom_frequency_counts.get("mujoco:tendon_wrap", 0), 0)

    def test_filtered_shapes_warn(self):
        """Warn when load options exclude a shape needed by a tendon."""
        for kwargs in ({"load_sites": False}, {"ignore_paths": ["/Robot/Base/wrap"]}):
            with self.subTest(kwargs=kwargs):
                stage = self._create_stage()
                self._add_wrapped_tendon(stage)
                with self.assertWarnsRegex(UserWarning, "Skipping spatial MjcTendon /Robot/Tendon"):
                    builder = self._import(stage, **kwargs)
                self.assertEqual(builder._custom_frequency_counts.get("mujoco:tendon_wrap", 0), 0)

    def test_invalid_side_sites_warn(self):
        """Reject malformed side-site indices and unresolved or non-site side targets."""
        invalid = [
            (["/Robot/Base/side"], [-1, 0]),
            (["/Robot/Base/side"], [-1, 1, -1, -1, -1]),
            (["/Robot/Base/side"], [-1, -2, -1, -1, -1]),
            ([], [-1, 0, -1, -1, -1]),
            (["/Robot/Missing"], [-1, 0, -1, -1, -1]),
            (["/Robot/Base/wrap"], [-1, 0, -1, -1, -1]),
        ]
        for targets, indices in invalid:
            with self.subTest(targets=targets, indices=indices):
                stage = self._create_stage()
                prim = self._add_wrapped_tendon(stage)
                prim.GetRelationship("mjc:sideSites").SetTargets(targets)
                prim.GetAttribute("mjc:sideSites:indices").Set(indices)
                with self.assertWarnsRegex(UserWarning, "Skipping spatial MjcTendon /Robot/Tendon"):
                    builder = self._import(stage)
                self.assertEqual(builder._custom_frequency_counts.get("mujoco:tendon_wrap", 0), 0)

    def test_empty_fixed_tendon_export_warns(self):
        """Emit a user-visible warning when MuJoCo export drops an empty fixed tendon."""
        stage = self._create_stage()
        self._add_tendon(stage, tendon_type="fixed", targets=[])
        model = self._import(stage).finalize(device="cpu")
        with self.assertWarnsRegex(UserWarning, "Skipping fixed tendon '/Robot/Tendon'.*no joint wraps"):
            solver = SolverMuJoCo(model, use_mujoco_cpu=True)
        self.assertEqual(solver.mj_model.ntendon, 0)


if __name__ == "__main__":
    unittest.main()
