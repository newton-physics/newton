# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Tests for USD curve-deformable (cable) import: topology, materials, normals, instancing,
replication, graph welding, and curve-to-curve junctions."""

import math
import tempfile
import unittest
from pathlib import Path

import numpy as np
import warp as wp

import newton
from newton.tests._usd_deformable_test_utils import (
    _add_cable_curve,
    _add_physics_attachment,
    _apply_deformable_body_api,
    _bind_deformable_material,
    deformable_maps,
)
from newton.tests.unittest_utils import USD_AVAILABLE, add_function_test, get_selected_cuda_test_devices
from newton.usd import SchemaResolverPhysx


@unittest.skipUnless(USD_AVAILABLE, "Requires usd-core")
class TestUSDDeformableCable(unittest.TestCase):
    """Curve-deformable (cable) parsing into VBD rod bodies + cable joints."""

    def test_basis_curves_imports_as_cable(self):
        """A BasisCurves with PhysicsCurvesDeformableSimAPI imports as a rod (capsule bodies + cable joints)."""
        from pxr import Usd, UsdPhysics

        with tempfile.TemporaryDirectory() as tmpdir:
            usd_path = Path(tmpdir) / "cable.usda"
            stage = Usd.Stage.CreateNew(str(usd_path))
            UsdPhysics.Scene.Define(stage, "/PhysicsScene")
            # 4 points -> 3 segments -> 3 capsule bodies, 2 cable joints (open chain).
            pts = [(0.0, 0.0, 1.0), (0.1, 0.0, 1.0), (0.2, 0.0, 1.0), (0.3, 0.0, 1.0)]
            _add_cable_curve(stage, "/World/Cable", pts)
            stage.Save()

            builder = newton.ModelBuilder()
            builder.add_usd(str(usd_path))
            cable_map, _, _ = deformable_maps(builder)

            self.assertIn("/World/Cable", cable_map)
            bodies, joints = cable_map["/World/Cable"]
            self.assertEqual(len(bodies), 3, "expected one capsule body per segment")
            self.assertEqual(len(joints), 2, "expected num_segments - 1 cable joints for an open chain")

    def test_cable_addressable_by_path_after_finalize(self):
        """After finalize(), a cable resolves by prim path to its body/joint ranges on the Model."""
        from pxr import Usd, UsdPhysics

        with tempfile.TemporaryDirectory() as tmpdir:
            usd_path = Path(tmpdir) / "cable.usda"
            stage = Usd.Stage.CreateNew(str(usd_path))
            UsdPhysics.Scene.Define(stage, "/PhysicsScene")
            pts = [(0.0, 0.0, 1.0), (0.1, 0.0, 1.0), (0.2, 0.0, 1.0), (0.3, 0.0, 1.0)]
            _add_cable_curve(stage, "/World/Cable", pts)
            stage.Save()

            builder = newton.ModelBuilder()
            builder.add_usd(str(usd_path))
            cable_map, _, _ = deformable_maps(builder)
            bodies, joints = cable_map["/World/Cable"]
            model = builder.finalize()

            self.assertEqual(model.cable_count, 1)
            self.assertEqual(model.cable_label, ["/World/Cable"])
            index = model.cable_index("/World/Cable")
            self.assertEqual(model.cable_body_range(index), (bodies[0], bodies[-1] + 1))
            self.assertEqual(model.cable_joint_range(index), (joints[0], joints[-1] + 1))
            self.assertEqual(int(model.cable_world.numpy()[index]), -1)  # no begin_world -> global
            # The resolved range addresses real model bodies.
            b0, b1 = model.cable_body_range(index)
            self.assertLessEqual(b1, model.body_count)
            self.assertEqual(b1 - b0, 3)  # one capsule body per segment
            with self.assertRaises(KeyError):
                model.cable_index("/World/DoesNotExist")

    @staticmethod
    def _author_attached_cable_pair(tmpdir, *, gap, stiffness=None, damping=None):
        """Two 4-point cables separated by ``gap`` in y with a point->point attachment
        (P0 of B onto P0 of A); returns the stage path."""
        from pxr import Usd, UsdGeom, UsdPhysics

        usd_path = Path(tmpdir) / "attached_pair.usda"
        stage = Usd.Stage.CreateNew(str(usd_path))
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
        UsdPhysics.Scene.Define(stage, "/PhysicsScene")
        pts_a = [(0.0, 0.0, 1.0), (0.1, 0.0, 1.0), (0.2, 0.0, 1.0), (0.3, 0.0, 1.0)]
        pts_b = [(0.0, gap, 1.0), (0.1, gap, 1.0), (0.2, gap, 1.0), (0.3, gap, 1.0)]
        _add_cable_curve(stage, "/World/CableA", pts_a)
        _add_cable_curve(stage, "/World/CableB", pts_b)
        _add_physics_attachment(
            stage,
            "/World/Junction",
            src0="/World/CableA",
            src1="/World/CableB",
            type0="point",
            type1="point",
            indices0=[0],
            indices1=[0],
            stiffness=stiffness,
            damping=damping,
        )
        stage.Save()
        return usd_path

    def test_zero_stiffness_attachment_is_not_welded(self):
        """A stiffness=0 curve-to-curve attachment must not weld the cables or move geometry."""

        with tempfile.TemporaryDirectory() as tmpdir:
            usd_path = self._author_attached_cable_pair(tmpdir, gap=10.0, stiffness=0.0)

            builder = newton.ModelBuilder()
            with self.assertWarnsRegex(UserWarning, "finite stiffness/damping; \\S*\\s*not welded"):
                result = builder.add_usd(str(usd_path))
            cable_map, _, _ = deformable_maps(builder)
            # Two independent cables in their own articulations; CableB stays at y=10.
            self.assertEqual(len(cable_map), 2)
            self.assertEqual(len(builder.articulation_label), 2)
            bodies_b, _ = cable_map["/World/CableB"]
            self.assertAlmostEqual(float(builder.body_q[bodies_b[0]][1]), 10.0, places=4)
            # The attachment is preserved as unsupported, not silently consumed.
            attrs = result["path_attachment_attrs"]["/World/Junction"]
            self.assertEqual(attrs["stiffness"], 0.0)
            self.assertIn("unsupported_reason", attrs)

    def test_finite_stiffness_attachment_is_not_welded(self):
        """A finite-stiffness curve-to-curve attachment is preserved, not welded."""

        with tempfile.TemporaryDirectory() as tmpdir:
            usd_path = self._author_attached_cable_pair(tmpdir, gap=0.0, stiffness=1.0e4)

            builder = newton.ModelBuilder()
            with self.assertWarnsRegex(UserWarning, "finite stiffness/damping"):
                result = builder.add_usd(str(usd_path))
            cable_map, _, _ = deformable_maps(builder)
            self.assertEqual(len(cable_map), 2)
            self.assertEqual(len(builder.articulation_label), 2)
            self.assertIn("unsupported_reason", result["path_attachment_attrs"]["/World/Junction"])

    def test_non_coincident_hard_attachment_is_not_welded(self):
        """A hard attachment whose sites are apart must not snap the cables together."""

        with tempfile.TemporaryDirectory() as tmpdir:
            usd_path = self._author_attached_cable_pair(tmpdir, gap=10.0)

            builder = newton.ModelBuilder()
            with self.assertWarnsRegex(UserWarning, "not coincident"):
                result = builder.add_usd(str(usd_path))
            cable_map, _, _ = deformable_maps(builder)
            self.assertEqual(len(cable_map), 2)
            bodies_b, _ = cable_map["/World/CableB"]
            self.assertAlmostEqual(float(builder.body_q[bodies_b[0]][1]), 10.0, places=4)
            self.assertIn("unsupported_reason", result["path_attachment_attrs"]["/World/Junction"])

    def test_coincident_hard_attachment_still_welds(self):
        """A hard, coincident junction keeps welding into one rod graph."""

        with tempfile.TemporaryDirectory() as tmpdir:
            usd_path = self._author_attached_cable_pair(tmpdir, gap=0.0)

            builder = newton.ModelBuilder()
            result = builder.add_usd(str(usd_path))
            cable_map, _, _ = deformable_maps(builder)
            # One welded component: both curves report the same graph_component and share
            # one articulation; the junction attachment is consumed as topology.
            self.assertEqual(len(builder.articulation_label), 1)
            self.assertEqual(
                result["path_cable_attrs"]["/World/CableA"]["graph_component"],
                result["path_cable_attrs"]["/World/CableB"]["graph_component"],
            )
            self.assertNotIn("/World/Junction", result["path_attachment_attrs"])
            self.assertEqual(len(cable_map), 2)

    def test_disabled_cable_body_is_skipped(self):
        """physics:bodyEnabled=false skips the cable instead of importing it dynamically."""
        from pxr import Sdf, Usd, UsdPhysics

        with tempfile.TemporaryDirectory() as tmpdir:
            usd_path = Path(tmpdir) / "disabled_cable.usda"
            stage = Usd.Stage.CreateNew(str(usd_path))
            UsdPhysics.Scene.Define(stage, "/PhysicsScene")
            pts = [(0.0, 0.0, 1.0), (0.1, 0.0, 1.0), (0.2, 0.0, 1.0), (0.3, 0.0, 1.0)]
            curves = _add_cable_curve(stage, "/World/Cable", pts)
            curves.GetPrim().CreateAttribute("physics:bodyEnabled", Sdf.ValueTypeNames.Bool).Set(False)
            stage.Save()

            builder = newton.ModelBuilder()
            with self.assertWarnsRegex(UserWarning, "bodyEnabled is false"):
                builder.add_usd(str(usd_path))
            self.assertEqual(builder.cable_label, [])
            self.assertEqual(builder.body_count, 0)

    def test_plain_curve_without_api_is_not_a_cable(self):
        """A BasisCurves without the curve-deformable API must not produce a cable."""
        from pxr import Usd, UsdGeom, UsdPhysics

        with tempfile.TemporaryDirectory() as tmpdir:
            usd_path = Path(tmpdir) / "plain_curve.usda"
            stage = Usd.Stage.CreateNew(str(usd_path))
            UsdPhysics.Scene.Define(stage, "/PhysicsScene")
            curves = UsdGeom.BasisCurves.Define(stage, "/World/Curve")
            curves.CreateTypeAttr().Set(UsdGeom.Tokens.linear)
            curves.CreatePointsAttr([(0.0, 0.0, 1.0), (0.1, 0.0, 1.0), (0.2, 0.0, 1.0)])
            curves.CreateCurveVertexCountsAttr([3])
            stage.Save()

            builder = newton.ModelBuilder()
            builder.add_usd(str(usd_path))
            cable_map, _, _ = deformable_maps(builder)

            self.assertEqual(cable_map, {})
            self.assertEqual(builder.body_count, 0)

    def test_cable_material_maps_to_rod_stiffness(self):
        """Bound curve-deformable material -> radius + per-joint stretch/bend stiffness."""
        from pxr import Usd, UsdPhysics

        with tempfile.TemporaryDirectory() as tmpdir:
            usd_path = Path(tmpdir) / "cable_mat.usda"
            stage = Usd.Stage.CreateNew(str(usd_path))
            UsdPhysics.Scene.Define(stage, "/PhysicsScene")
            # 3 segments of length 0.1 along x.
            pts = [(0.0, 0.0, 1.0), (0.1, 0.0, 1.0), (0.2, 0.0, 1.0), (0.3, 0.0, 1.0)]
            curves = _add_cable_curve(stage, "/World/Cable", pts)
            thickness, stretch_mod, bend_mod = 0.02, 2.0e6, 3.0e5
            _bind_deformable_material(
                stage,
                curves.GetPrim(),
                "/World/CableMat",
                thickness=thickness,
                stretchStiffness=stretch_mod,
                bendStiffness=bend_mod,
            )
            stage.Save()

            builder = newton.ModelBuilder()
            builder.add_usd(str(usd_path))
            cable_map, _, _ = deformable_maps(builder)
            bodies, joints = cable_map["/World/Cable"]
            self.assertEqual(len(bodies), 3)

            # radius = thickness / 2; stretch/bend converted with A/L, I/L.
            r = 0.5 * thickness
            seg_len = 0.3 / 3
            area = math.pi * r * r
            inertia = 0.25 * math.pi * r**4
            expected_stretch = stretch_mod * area / seg_len
            expected_bend = bend_mod * inertia / seg_len

            # Cable joints store stretch in the linear DOF target_ke, bend in the angular.
            j0 = joints[0]
            dof0 = builder.joint_qd_start[j0]
            ke = builder.joint_target_ke
            self.assertAlmostEqual(ke[dof0], expected_stretch, delta=expected_stretch * 1e-3)
            self.assertAlmostEqual(ke[dof0 + 1], expected_bend, delta=expected_bend * 1e-3)

    def test_cable_rest_length_from_rest_shape_points(self):
        """Per-joint stiffness uses the rest centerline (restShapePoints), not the possibly-deformed
        points, so an authored rest shape sets the rest length L in E*A/L (proposal: rest segment
        lengths derive from restShapePoints)."""
        from pxr import Sdf, Usd, UsdPhysics

        with tempfile.TemporaryDirectory() as tmpdir:
            usd_path = Path(tmpdir) / "cable_rest.usda"
            stage = Usd.Stage.CreateNew(str(usd_path))
            UsdPhysics.Scene.Define(stage, "/PhysicsScene")
            # Current points: 0.2-long segments (a stretched state).
            pts = [(0.0, 0.0, 1.0), (0.2, 0.0, 1.0), (0.4, 0.0, 1.0), (0.6, 0.0, 1.0)]
            curves = _add_cable_curve(stage, "/World/Cable", pts)
            # Rest centerline: 0.1-long segments (half the deformed length).
            rest = [(0.0, 0.0, 1.0), (0.1, 0.0, 1.0), (0.2, 0.0, 1.0), (0.3, 0.0, 1.0)]
            curves.GetPrim().CreateAttribute("physics:restShapePoints", Sdf.ValueTypeNames.Point3fArray).Set(rest)
            thickness, stretch_mod = 0.02, 2.0e6
            _bind_deformable_material(
                stage, curves.GetPrim(), "/World/CableMat", thickness=thickness, stretchStiffness=stretch_mod
            )
            stage.Save()

            builder = newton.ModelBuilder()
            # restShapePoints only normalizes stiffness (it does not set an initial strain state), so it warns.
            with self.assertWarnsRegex(UserWarning, "restShapePoints only sets the rest length"):
                builder.add_usd(str(usd_path))
            cable_map, _, _ = deformable_maps(builder)
            _, joints = cable_map["/World/Cable"]
            r = 0.5 * thickness
            area = math.pi * r * r
            expected = stretch_mod * area / 0.1  # rest length 0.1, not the 0.2 deformed segments
            dof0 = builder.joint_qd_start[joints[0]]
            self.assertAlmostEqual(builder.joint_target_ke[dof0], expected, delta=expected * 1e-3)

    def test_cable_zero_stiffness_is_preserved(self):
        """Authored zero stiffness (range [0, inf)) is kept, not replaced by add_rod's default."""
        from pxr import Usd, UsdPhysics

        with tempfile.TemporaryDirectory() as tmpdir:
            usd_path = Path(tmpdir) / "cable_zero.usda"
            stage = Usd.Stage.CreateNew(str(usd_path))
            UsdPhysics.Scene.Define(stage, "/PhysicsScene")
            pts = [(0.0, 0.0, 1.0), (0.1, 0.0, 1.0), (0.2, 0.0, 1.0), (0.3, 0.0, 1.0)]
            curves = _add_cable_curve(stage, "/World/Cable", pts)
            _bind_deformable_material(
                stage,
                curves.GetPrim(),
                "/World/CableMat",
                thickness=0.02,
                stretchStiffness=0.0,
                bendStiffness=3.0e5,
            )
            stage.Save()

            builder = newton.ModelBuilder()
            result = builder.add_usd(str(usd_path))
            cable_map, _, _ = deformable_maps(builder)
            _bodies, joints = cable_map["/World/Cable"]
            # Stretch DOF target_ke is the authored 0.0, not add_rod's 1.0e5 default.
            dof0 = builder.joint_qd_start[joints[0]]
            self.assertEqual(builder.joint_target_ke[dof0], 0.0)
            self.assertEqual(result["path_cable_attrs"]["/World/Cable"]["material"]["stretchStiffness"], 0.0)

    def test_non_linear_curve_is_skipped(self):
        """A non-linear (cubic) curve-deformable warns and is skipped (cable import is linear-only)."""
        from pxr import Usd, UsdGeom, UsdPhysics

        with tempfile.TemporaryDirectory() as tmpdir:
            usd_path = Path(tmpdir) / "cubic.usda"
            stage = Usd.Stage.CreateNew(str(usd_path))
            UsdPhysics.Scene.Define(stage, "/PhysicsScene")
            curves = UsdGeom.BasisCurves.Define(stage, "/World/Cubic")
            curves.CreateTypeAttr().Set(UsdGeom.Tokens.cubic)
            curves.CreatePointsAttr([(0.0, 0.0, 1.0), (0.1, 0.0, 1.0), (0.2, 0.0, 1.0), (0.3, 0.0, 1.0)])
            curves.CreateCurveVertexCountsAttr([4])
            curves.GetPrim().AddAppliedSchema("PhysicsCurvesDeformableSimAPI")
            stage.Save()

            builder = newton.ModelBuilder()
            with self.assertWarnsRegex(UserWarning, "non-linear"):
                builder.add_usd(str(usd_path))
            cable_map, _, _ = deformable_maps(builder)
            self.assertEqual(cable_map, {})
            self.assertEqual(builder.body_count, 0)

    def test_cable_material_without_family_api_is_ignored(self):
        """A physics-bound material lacking PhysicsCurvesDeformableMaterialAPI is not read as a cable material."""
        from pxr import Sdf, Usd, UsdPhysics, UsdShade

        with tempfile.TemporaryDirectory() as tmpdir:
            usd_path = Path(tmpdir) / "cable_no_api.usda"
            stage = Usd.Stage.CreateNew(str(usd_path))
            UsdPhysics.Scene.Define(stage, "/PhysicsScene")
            pts = [(0.0, 0.0, 1.0), (0.1, 0.0, 1.0), (0.2, 0.0, 1.0), (0.3, 0.0, 1.0)]
            curves = _add_cable_curve(stage, "/World/Cable", pts, thickness=None)
            # Material carries cable-shaped attributes but does NOT declare the family API.
            mat = UsdShade.Material.Define(stage, "/World/Mat")
            mat.GetPrim().CreateAttribute("physics:stretchStiffness", Sdf.ValueTypeNames.Float).Set(2.0e6)
            mat.GetPrim().CreateAttribute("physics:thickness", Sdf.ValueTypeNames.Float).Set(0.02)
            UsdShade.MaterialBindingAPI.Apply(curves.GetPrim()).Bind(mat, materialPurpose="physics")
            stage.Save()

            builder = newton.ModelBuilder()
            # The family-less material is ignored, so the cable falls back to the default radius and warns.
            with self.assertWarnsRegex(UserWarning, "no cable thickness"):
                result = builder.add_usd(str(usd_path))
            cable_map, _, _ = deformable_maps(builder)
            # Without the family API the material is ignored: no attrs, default rod stiffness.
            self.assertEqual(result["path_cable_attrs"]["/World/Cable"]["material"], {})
            _bodies, joints = cable_map["/World/Cable"]
            dof0 = builder.joint_qd_start[joints[0]]
            self.assertEqual(builder.joint_target_ke[dof0], 1.0e5)  # add_rod default stretch stiffness

    def test_material_attr_authored_on_geometry_warns(self):
        """Deformable material moduli authored on the geometry (not the material) warn and are ignored."""
        from pxr import Sdf, Usd, UsdPhysics

        with tempfile.TemporaryDirectory() as tmpdir:
            usd_path = Path(tmpdir) / "cable_misplaced.usda"
            stage = Usd.Stage.CreateNew(str(usd_path))
            UsdPhysics.Scene.Define(stage, "/PhysicsScene")
            pts = [(0.0, 0.0, 1.0), (0.1, 0.0, 1.0), (0.2, 0.0, 1.0), (0.3, 0.0, 1.0)]
            curves = _add_cable_curve(stage, "/World/Cable", pts)
            # stretchStiffness belongs on the bound material, not the curve geometry.
            curves.GetPrim().CreateAttribute("physics:stretchStiffness", Sdf.ValueTypeNames.Float).Set(5.0e5)
            stage.Save()

            builder = newton.ModelBuilder()
            with self.assertWarnsRegex(UserWarning, "authored on the geometry"):
                builder.add_usd(str(usd_path))

    def test_cable_resolved_density_reports_default_when_unauthored(self):
        """resolved_density reports the density actually used (the builder default), not None."""
        from pxr import Usd, UsdPhysics

        with tempfile.TemporaryDirectory() as tmpdir:
            usd_path = Path(tmpdir) / "cable_nodensity.usda"
            stage = Usd.Stage.CreateNew(str(usd_path))
            UsdPhysics.Scene.Define(stage, "/PhysicsScene")
            pts = [(0.0, 0.0, 1.0), (0.1, 0.0, 1.0), (0.2, 0.0, 1.0), (0.3, 0.0, 1.0)]
            curves = _add_cable_curve(stage, "/World/Cable", pts)
            _bind_deformable_material(stage, curves.GetPrim(), "/World/CableMat", thickness=0.02)  # no density
            stage.Save()

            builder = newton.ModelBuilder()
            result = builder.add_usd(str(usd_path))
            attrs = result["path_cable_attrs"]["/World/Cable"]
            self.assertEqual(attrs["resolved_density"], builder.default_shape_cfg.density)

    @staticmethod
    def _author_two_curve_prim_with_masses(tmpdir, vertex_counts, masses):
        """Author a two-curve BasisCurves prim (one 2-point curve the importer skips, one
        4-point curve) with a physics:masses array; returns the stage path."""
        from pxr import Sdf, Usd, UsdGeom, UsdPhysics

        usd_path = Path(tmpdir) / "multi_curve.usda"
        stage = Usd.Stage.CreateNew(str(usd_path))
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
        UsdPhysics.Scene.Define(stage, "/PhysicsScene")
        curves = UsdGeom.BasisCurves.Define(stage, "/World/Cable")
        curves.CreateTypeAttr().Set(UsdGeom.Tokens.linear)
        short = [(0.0, 1.0, 1.0), (0.1, 1.0, 1.0)]
        long = [(0.0, 0.0, 1.0), (0.1, 0.0, 1.0), (0.2, 0.0, 1.0), (0.3, 0.0, 1.0)]
        pts = long + short if vertex_counts == [4, 2] else short + long
        curves.CreatePointsAttr(pts)
        curves.CreateCurveVertexCountsAttr(vertex_counts)
        curves.GetPrim().AddAppliedSchema("PhysicsCurvesDeformableSimAPI")
        curves.GetPrim().CreateAttribute("physics:masses", Sdf.ValueTypeNames.FloatArray).Set(masses)
        stage.Save()
        return usd_path

    def test_skipped_curve_masses_validated_against_authored_points(self):
        """physics:masses is per authored point: a full-length array is applied to the
        imported curve even when another curve in the prim is skipped."""

        with tempfile.TemporaryDirectory() as tmpdir:
            usd_path = self._author_two_curve_prim_with_masses(tmpdir, [4, 2], [1.0] * 6)

            builder = newton.ModelBuilder()
            with self.assertWarnsRegex(UserWarning, "skipping that curve"):
                builder.add_usd(str(usd_path))
            cable_map, _, _ = deformable_maps(builder)
            bodies, _ = cable_map["/World/Cable"]
            # Unit point masses lump onto the 3 segments as endpoint 1+1/2, interior 1/2+1/2.
            masses = [builder.body_mass[b] for b in bodies]
            np.testing.assert_allclose(masses, [1.5, 1.0, 1.5], atol=1e-6)

    def test_skipped_first_curve_masses_use_absolute_offsets(self):
        """A skipped FIRST curve must not shift the imported curve's slice of physics:masses."""

        with tempfile.TemporaryDirectory() as tmpdir:
            # The imported curve's points are authored at offsets 2..5.
            usd_path = self._author_two_curve_prim_with_masses(tmpdir, [2, 4], [9.0, 9.0, 1.0, 2.0, 2.0, 1.0])

            builder = newton.ModelBuilder()
            with self.assertWarnsRegex(UserWarning, "skipping that curve"):
                builder.add_usd(str(usd_path))
            cable_map, _, _ = deformable_maps(builder)
            bodies, _ = cable_map["/World/Cable"]
            # pm = [1, 2, 2, 1] -> segments [1 + 2/2, 2/2 + 2/2, 2/2 + 1] = [2, 2, 2];
            # the 9.0 entries belong to the skipped curve and must not leak in.
            masses = [builder.body_mass[b] for b in bodies]
            np.testing.assert_allclose(masses, [2.0, 2.0, 2.0], atol=1e-6)

    def test_short_curve_masses_array_warns_without_crash(self):
        """A masses array matching only the imported point count is rejected with a warning
        (it cannot be indexed by authored offset), not an IndexError."""

        with tempfile.TemporaryDirectory() as tmpdir:
            usd_path = self._author_two_curve_prim_with_masses(tmpdir, [2, 4], [1.0] * 4)

            builder = newton.ModelBuilder()
            with self.assertWarnsRegex(UserWarning, r"!= 6 authored curve points"):
                builder.add_usd(str(usd_path))
            cable_map, _, _ = deformable_maps(builder)
            self.assertEqual(len(cable_map["/World/Cable"][0]), 3)

    def test_duplicate_consecutive_points_skips_curve(self):
        """A curve with a zero-length segment is warned and skipped, not aborting the import."""
        from pxr import Usd, UsdGeom, UsdPhysics

        with tempfile.TemporaryDirectory() as tmpdir:
            usd_path = Path(tmpdir) / "dup.usda"
            stage = Usd.Stage.CreateNew(str(usd_path))
            UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
            UsdPhysics.Scene.Define(stage, "/PhysicsScene")
            curves = UsdGeom.BasisCurves.Define(stage, "/World/Cable")
            curves.CreateTypeAttr().Set(UsdGeom.Tokens.linear)
            # A valid 4-point curve, then a 3-point curve with a duplicate consecutive point.
            curves.CreatePointsAttr(
                [
                    (0.0, 0.0, 1.0),
                    (0.1, 0.0, 1.0),
                    (0.2, 0.0, 1.0),
                    (0.3, 0.0, 1.0),
                    (0.0, 1.0, 1.0),
                    (0.0, 1.0, 1.0),
                    (0.2, 1.0, 1.0),
                ]
            )
            curves.CreateCurveVertexCountsAttr([4, 3])
            curves.GetPrim().AddAppliedSchema("PhysicsCurvesDeformableSimAPI")
            stage.Save()

            builder = newton.ModelBuilder()
            with self.assertWarnsRegex(UserWarning, "duplicate consecutive points"):
                builder.add_usd(str(usd_path))
            cable_map, _, _ = deformable_maps(builder)
            # The valid curve still imports (4 points -> 3 bodies); the degenerate one is skipped.
            bodies, _ = cable_map["/World/Cable"]
            self.assertEqual(len(bodies), 3)

    def test_vendor_namespace_material_needs_resolver(self):
        """Vendor-namespaced (omniphysics:) material is read only with a compat resolver.

        The base parser targets the canonical ``physics:`` schema as written; the
        omniphysics fallback is opt-in via a schema resolver that declares it
        (mirroring how rigid-body vendor namespaces are remapped).
        """
        from pxr import Usd, UsdPhysics

        with tempfile.TemporaryDirectory() as tmpdir:
            usd_path = Path(tmpdir) / "cable_omni.usda"
            stage = Usd.Stage.CreateNew(str(usd_path))
            UsdPhysics.Scene.Define(stage, "/PhysicsScene")
            pts = [(0.0, 0.0, 1.0), (0.1, 0.0, 1.0), (0.2, 0.0, 1.0), (0.3, 0.0, 1.0)]
            curves = _add_cable_curve(stage, "/World/Cable", pts, thickness=None)
            _bind_deformable_material(
                stage, curves.GetPrim(), "/World/CableMat", namespace="omniphysics", thickness=0.02, density=1234.0
            )
            stage.Save()

            def cable_radius(builder):
                return builder.shape_scale[builder.body_shapes[0][0]][0]  # capsule radius

            # Default resolvers: omniphysics:thickness is ignored, so the radius is the
            # builder default, not the authored thickness / 2 (and the importer warns).
            builder_default = newton.ModelBuilder()
            with self.assertWarnsRegex(UserWarning, "no cable thickness"):
                builder_default.add_usd(str(usd_path))
            default_radius = cable_radius(builder_default)

            # With the PhysX resolver active, omniphysics:thickness is honored (radius = thickness / 2).
            builder_compat = newton.ModelBuilder()
            builder_compat.add_usd(str(usd_path), schema_resolvers=[SchemaResolverPhysx()])
            self.assertAlmostEqual(cable_radius(builder_compat), 0.5 * 0.02, places=5)
            self.assertNotAlmostEqual(default_radius, 0.5 * 0.02, places=5)

    def test_deformable_ignores_generic_physx_namespaces(self):
        """Deformable material reads only deformable vendor namespaces, not generic PhysX ones."""
        from pxr import Usd, UsdPhysics

        def cable_radius(namespace):
            with tempfile.TemporaryDirectory() as tmpdir:
                usd_path = Path(tmpdir) / "cable.usda"
                stage = Usd.Stage.CreateNew(str(usd_path))
                UsdPhysics.Scene.Define(stage, "/PhysicsScene")
                pts = [(0.0, 0.0, 1.0), (0.1, 0.0, 1.0), (0.2, 0.0, 1.0), (0.3, 0.0, 1.0)]
                curves = _add_cable_curve(stage, "/World/Cable", pts, thickness=None)
                _bind_deformable_material(stage, curves.GetPrim(), "/World/Mat", namespace=namespace, thickness=0.02)
                stage.Save()
                builder = newton.ModelBuilder()
                builder.add_usd(str(usd_path), schema_resolvers=[SchemaResolverPhysx()])
                return builder.shape_scale[builder.body_shapes[0][0]][0]

        # omniphysics is a deformable vendor namespace -> thickness honored (no fallback warning).
        self.assertAlmostEqual(cable_radius("omniphysics"), 0.5 * 0.02, places=5)
        # physxScene is a generic resolver namespace -> NOT read as deformable material, so the
        # cable falls back to the default radius and warns.
        with self.assertWarnsRegex(UserWarning, "no cable thickness"):
            physx_radius = cable_radius("physxScene")
        self.assertNotAlmostEqual(physx_radius, 0.5 * 0.02, places=5)

    def test_two_cables_have_disjoint_body_ranges(self):
        """Two cables in one stage map to disjoint body/joint index ranges (addressability)."""
        from pxr import Usd, UsdPhysics

        with tempfile.TemporaryDirectory() as tmpdir:
            usd_path = Path(tmpdir) / "two_cables.usda"
            stage = Usd.Stage.CreateNew(str(usd_path))
            UsdPhysics.Scene.Define(stage, "/PhysicsScene")
            a = [(0.0, 0.0, 1.0), (0.1, 0.0, 1.0), (0.2, 0.0, 1.0), (0.3, 0.0, 1.0)]  # 3 seg
            b = [(0.0, 1.0, 1.0), (0.1, 1.0, 1.0), (0.2, 1.0, 1.0)]  # 2 seg
            _add_cable_curve(stage, "/World/CableA", a)
            _add_cable_curve(stage, "/World/CableB", b)
            stage.Save()

            builder = newton.ModelBuilder()
            builder.add_usd(str(usd_path))
            cmap, _, _ = deformable_maps(builder)

            bodies_a, joints_a = cmap["/World/CableA"]
            bodies_b, joints_b = cmap["/World/CableB"]
            self.assertEqual((len(bodies_a), len(joints_a)), (3, 2))
            self.assertEqual((len(bodies_b), len(joints_b)), (2, 1))
            # Disjoint, covering all created bodies.
            self.assertEqual(set(bodies_a) & set(bodies_b), set())
            self.assertEqual(sorted(bodies_a + bodies_b), list(range(builder.body_count)))

    def test_cable_body_range_matches_curve(self):
        """Each cable body origin matches its authored segment midpoint (map points at the right bodies)."""
        from pxr import Usd, UsdGeom, UsdPhysics

        with tempfile.TemporaryDirectory() as tmpdir:
            usd_path = Path(tmpdir) / "cable_pos.usda"
            stage = Usd.Stage.CreateNew(str(usd_path))
            # Z-up stage so authored points are not axis-converted on import.
            UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
            UsdPhysics.Scene.Define(stage, "/PhysicsScene")
            pts = [(0.0, 0.0, 1.0), (0.1, 0.0, 1.0), (0.2, 0.0, 1.0), (0.3, 0.0, 1.0)]
            _add_cable_curve(stage, "/World/Cable", pts)
            stage.Save()

            builder = newton.ModelBuilder()
            builder.add_usd(str(usd_path))
            cable_map, _, _ = deformable_maps(builder)
            bodies, _ = cable_map["/World/Cable"]

            # The importer builds rods with body_frame_origin="com", so body i origin sits
            # at the midpoint of segment i (between authored points[i] and points[i + 1]).
            for i, body in enumerate(bodies):
                origin = np.array(builder.body_q[body][:3], dtype=np.float32)
                midpoint = 0.5 * (np.array(pts[i]) + np.array(pts[i + 1]))
                np.testing.assert_allclose(origin, midpoint.astype(np.float32), atol=1e-5)

    def test_cable_attrs_surface_authored_material_including_dropped_moduli(self):
        """path_cable_attrs exposes the as-authored material - including the shear /
        twist moduli the VBD rod ignores - so a non-VBD solver can consume them."""
        from pxr import Usd, UsdGeom, UsdPhysics

        with tempfile.TemporaryDirectory() as tmpdir:
            usd_path = Path(tmpdir) / "cable_attrs.usda"
            stage = Usd.Stage.CreateNew(str(usd_path))
            UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
            UsdPhysics.Scene.Define(stage, "/PhysicsScene")
            pts = [(0.0, 0.0, 1.0), (0.1, 0.0, 1.0), (0.2, 0.0, 1.0), (0.3, 0.0, 1.0)]
            curve = _add_cable_curve(stage, "/World/Cable", pts, thickness=None)
            _bind_deformable_material(
                stage,
                curve.GetPrim(),
                "/World/Mat",
                thickness=0.02,
                density=1000.0,
                bendStiffness=10.0,
                shearStiffness=3.0,
                twistStiffness=4.0,
            )
            stage.Save()

            builder = newton.ModelBuilder()
            # shear / twist are preserved in the attrs but not mapped into the VBD rod, so the importer warns.
            with self.assertWarnsRegex(UserWarning, "not yet mapped"):
                result = builder.add_usd(str(usd_path))

            attrs = result["path_cable_attrs"]["/World/Cable"]
            mat = attrs["material"]
            # shear / twist are not mapped into the VBD rod but are preserved here.
            self.assertAlmostEqual(mat["shearStiffness"], 3.0, places=5)
            self.assertAlmostEqual(mat["twistStiffness"], 4.0, places=5)
            self.assertAlmostEqual(mat["bendStiffness"], 10.0, places=5)
            self.assertFalse(attrs["closed"])
            self.assertIsNotNone(attrs["resolved_density"])

    def test_cable_density_scales_segment_mass(self):
        """Material density maps to segment mass: doubling density doubles segment mass."""
        from pxr import Usd, UsdGeom, UsdPhysics

        with tempfile.TemporaryDirectory() as tmpdir:
            usd_path = Path(tmpdir) / "cable_density.usda"
            stage = Usd.Stage.CreateNew(str(usd_path))
            UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
            UsdPhysics.Scene.Define(stage, "/PhysicsScene")
            pts_a = [(0.0, 0.0, 1.0), (0.1, 0.0, 1.0), (0.2, 0.0, 1.0), (0.3, 0.0, 1.0)]
            pts_b = [(0.0, 1.0, 1.0), (0.1, 1.0, 1.0), (0.2, 1.0, 1.0), (0.3, 1.0, 1.0)]
            ca = _add_cable_curve(stage, "/World/CableA", pts_a)
            cb = _add_cable_curve(stage, "/World/CableB", pts_b)
            _bind_deformable_material(stage, ca.GetPrim(), "/World/MatA", thickness=0.02, density=1000.0)
            _bind_deformable_material(stage, cb.GetPrim(), "/World/MatB", thickness=0.02, density=2000.0)
            stage.Save()

            builder = newton.ModelBuilder()
            builder.add_usd(str(usd_path))
            cable_map, _, _ = deformable_maps(builder)
            bodies_a, _ = cable_map["/World/CableA"]
            bodies_b, _ = cable_map["/World/CableB"]

            mass_a = builder.body_mass[bodies_a[0]]
            mass_b = builder.body_mass[bodies_b[0]]
            self.assertGreater(mass_a, 0.0)
            self.assertAlmostEqual(mass_b, 2.0 * mass_a, delta=mass_a * 1e-3)

    def test_cable_density_segment_mass_is_cylinder_not_capsule(self):
        """A density-derived cable segment gets the cylinder mass rho*pi*r^2*L, not add_rod's capsule
        mass (cylinder plus two hemispherical caps), which overestimates short / thick segments."""
        from pxr import Usd, UsdGeom, UsdPhysics

        with tempfile.TemporaryDirectory() as tmpdir:
            usd_path = Path(tmpdir) / "cable_cyl.usda"
            stage = Usd.Stage.CreateNew(str(usd_path))
            UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
            UsdPhysics.Scene.Define(stage, "/PhysicsScene")
            # Short, thick segments make the spherical-cap bias large (4r/3L = 0.667 -> +67%).
            r, seg_len, rho = 0.05, 0.1, 1000.0
            pts = [(i * seg_len, 0.0, 1.0) for i in range(4)]  # 3 segments of length seg_len
            _add_cable_curve(stage, "/World/Cable", pts, thickness=2.0 * r, density=rho)
            stage.Save()

            builder = newton.ModelBuilder()
            builder.add_usd(str(usd_path))
            cable_map, _, _ = deformable_maps(builder)
            bodies, _ = cable_map["/World/Cable"]
            cylinder = rho * math.pi * r * r * seg_len
            capsule = cylinder + rho * (4.0 / 3.0) * math.pi * r**3
            for b in bodies:
                self.assertAlmostEqual(builder.body_mass[b], cylinder, delta=cylinder * 1e-3)
                self.assertNotAlmostEqual(builder.body_mass[b], capsule, delta=cylinder * 1e-2)

    def test_cable_density_segment_mass_scales_with_length(self):
        """Density-derived segment masses follow segment length (cylinder volume): a 2x-longer segment
        has ~2x the mass, not the cap-biased ratio add_rod's constant hemispherical ends would give."""
        from pxr import Usd, UsdGeom, UsdPhysics

        with tempfile.TemporaryDirectory() as tmpdir:
            usd_path = Path(tmpdir) / "cable_lenweight.usda"
            stage = Usd.Stage.CreateNew(str(usd_path))
            UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
            UsdPhysics.Scene.Define(stage, "/PhysicsScene")
            # First segment length 0.1, second length 0.2 -> cylinder mass ratio exactly 2.
            pts = [(0.0, 0.0, 1.0), (0.1, 0.0, 1.0), (0.3, 0.0, 1.0)]
            _add_cable_curve(stage, "/World/Cable", pts, thickness=0.1, density=1000.0)
            stage.Save()

            builder = newton.ModelBuilder()
            builder.add_usd(str(usd_path))
            cable_map, _, _ = deformable_maps(builder)
            bodies, _ = cable_map["/World/Cable"]
            self.assertEqual(len(bodies), 2)
            self.assertAlmostEqual(builder.body_mass[bodies[1]] / builder.body_mass[bodies[0]], 2.0, places=3)

    def test_cable_normals_orient_segments(self):
        """Authored normals set each segment's cross-section frame: +Z -> tangent, +Y -> normal."""
        from pxr import Usd, UsdGeom, UsdPhysics

        with tempfile.TemporaryDirectory() as tmpdir:
            usd_path = Path(tmpdir) / "cable_normals.usda"
            stage = Usd.Stage.CreateNew(str(usd_path))
            UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
            UsdPhysics.Scene.Define(stage, "/PhysicsScene")
            pts = [(0.0, 0.0, 1.0), (0.1, 0.0, 1.0), (0.2, 0.0, 1.0), (0.3, 0.0, 1.0)]  # tangent +X
            curves = _add_cable_curve(stage, "/World/Cable", pts)
            normals = curves.GetNormalsAttr()
            if not normals:
                normals = curves.CreateNormalsAttr()
            normals.Set([(0.0, 1.0, 0.0)] * len(pts))  # cross-section frame: +Y
            curves.SetNormalsInterpolation(UsdGeom.Tokens.vertex)
            stage.Save()

            builder = newton.ModelBuilder()
            builder.add_usd(str(usd_path))
            cable_map, _, _ = deformable_maps(builder)
            bodies, _ = cable_map["/World/Cable"]

            for body in bodies:
                t = builder.body_q[body]
                q = wp.quat(float(t[3]), float(t[4]), float(t[5]), float(t[6]))
                z_world = np.array(wp.quat_rotate(q, wp.vec3(0.0, 0.0, 1.0)), dtype=np.float32)
                y_world = np.array(wp.quat_rotate(q, wp.vec3(0.0, 1.0, 0.0)), dtype=np.float32)
                np.testing.assert_allclose(z_world, [1.0, 0.0, 0.0], atol=1e-5)  # +Z -> tangent +X
                np.testing.assert_allclose(y_world, [0.0, 1.0, 0.0], atol=1e-5)  # +Y -> normal

    def test_cable_normals_under_reflection_use_inverse_transpose(self):
        """Authored normals transform by the inverse-transpose of the world map, so a reflective scale
        flips the material frame correctly. A rot/scale decomposition drops the reflection parity and
        would leave the normal unflipped."""
        from pxr import Gf, Usd, UsdGeom, UsdPhysics

        with tempfile.TemporaryDirectory() as tmpdir:
            usd_path = Path(tmpdir) / "cable_reflect_normals.usda"
            stage = Usd.Stage.CreateNew(str(usd_path))
            UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
            UsdPhysics.Scene.Define(stage, "/PhysicsScene")
            pts = [(0.0, 0.0, 1.0), (0.1, 0.0, 1.0), (0.2, 0.0, 1.0), (0.3, 0.0, 1.0)]  # tangent +X
            curves = _add_cable_curve(stage, "/World/Cable", pts)
            curves.CreateNormalsAttr([(0.0, 1.0, 0.0)] * len(pts))  # local cross-section normal +Y
            curves.SetNormalsInterpolation(UsdGeom.Tokens.vertex)
            UsdGeom.Xformable(curves).AddScaleOp().Set(Gf.Vec3d(1.0, -1.0, 1.0))  # reflect across Y
            stage.Save()

            builder = newton.ModelBuilder()
            builder.add_usd(str(usd_path))
            cable_map, _, _ = deformable_maps(builder)
            bodies, _ = cable_map["/World/Cable"]
            for body in bodies:
                t = builder.body_q[body]
                q = wp.quat(float(t[3]), float(t[4]), float(t[5]), float(t[6]))
                y_world = np.array(wp.quat_rotate(q, wp.vec3(0.0, 1.0, 0.0)), dtype=np.float32)
                # inverse-transpose of diag(1, -1, 1) maps +Y -> -Y; a decomposition would keep +Y.
                np.testing.assert_allclose(y_world, [0.0, -1.0, 0.0], atol=1e-5)

    def test_cable_rest_length_uses_full_linear_under_shear(self):
        """Rest segment lengths (for stiffness) transform by the full linear block, so a shear that
        lengthens a segment lowers stretch stiffness E*A/L accordingly. A decomposed scale cannot
        represent the shear and would measure the rest length incorrectly."""
        from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics

        with tempfile.TemporaryDirectory() as tmpdir:
            usd_path = Path(tmpdir) / "cable_shear_rest.usda"
            stage = Usd.Stage.CreateNew(str(usd_path))
            UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
            UsdPhysics.Scene.Define(stage, "/PhysicsScene")
            seg_len, thickness, E, k = 0.1, 0.02, 2.0e6, 0.75  # shear maps a +X segment to length L*sqrt(1+k^2)
            pts = [(i * seg_len, 0.0, 1.0) for i in range(4)]  # along +X
            curves = _add_cable_curve(stage, "/World/Cable", pts, thickness=None)
            _bind_deformable_material(stage, curves.GetPrim(), "/World/Mat", thickness=thickness, stretchStiffness=E)
            # Rest centerline == authored points, so rest length equals the (sheared) segment length.
            curves.GetPrim().CreateAttribute("physics:restShapePoints", Sdf.ValueTypeNames.Point3fArray).Set(
                [tuple(p) for p in pts]
            )
            # Shear (row-major Gf matrix): world z += k * x, not expressible as a per-axis scale.
            m = Gf.Matrix4d(1.0, 0.0, k, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0)
            UsdGeom.Xformable(curves).AddTransformOp().Set(m)
            stage.Save()

            builder = newton.ModelBuilder()
            with self.assertWarnsRegex(UserWarning, "restShapePoints only sets the rest length"):
                builder.add_usd(str(usd_path))
            cable_map, _, _ = deformable_maps(builder)
            _bodies, joints = cable_map["/World/Cable"]
            r = 0.5 * thickness
            rest_len = seg_len * math.sqrt(1.0 + k * k)
            expected_ke = E * math.pi * r * r / rest_len
            dof0 = builder.joint_qd_start[joints[0]]
            self.assertAlmostEqual(builder.joint_target_ke[dof0], expected_ke, delta=expected_ke * 1e-3)

    def test_cable_velocities_warn(self):
        """Authored cable velocities are dropped with a warning rather than silently."""
        from pxr import Usd, UsdGeom, UsdPhysics

        with tempfile.TemporaryDirectory() as tmpdir:
            usd_path = Path(tmpdir) / "cable_vel.usda"
            stage = Usd.Stage.CreateNew(str(usd_path))
            UsdPhysics.Scene.Define(stage, "/PhysicsScene")
            pts = [(0.0, 0.0, 1.0), (0.1, 0.0, 1.0), (0.2, 0.0, 1.0), (0.3, 0.0, 1.0)]
            curves = _add_cable_curve(stage, "/World/Cable", pts)
            UsdGeom.PointBased(curves.GetPrim()).CreateVelocitiesAttr([(1.0, 2.0, 3.0)] * len(pts))
            stage.Save()

            builder = newton.ModelBuilder()
            with self.assertWarnsRegex(UserWarning, "velocities are not imported"):
                builder.add_usd(str(usd_path))

    def test_cable_non_per_point_normals_ignored(self):
        """Cable normals with non-per-point interpolation are warned and ignored, not misapplied."""
        from pxr import Usd, UsdGeom, UsdPhysics

        with tempfile.TemporaryDirectory() as tmpdir:
            usd_path = Path(tmpdir) / "cable_const_normals.usda"
            stage = Usd.Stage.CreateNew(str(usd_path))
            UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
            UsdPhysics.Scene.Define(stage, "/PhysicsScene")
            pts = [(0.0, 0.0, 1.0), (0.1, 0.0, 1.0), (0.2, 0.0, 1.0), (0.3, 0.0, 1.0)]
            curves = _add_cable_curve(stage, "/World/Cable", pts)
            curves.CreateNormalsAttr([(0.0, 1.0, 0.0)])
            curves.SetNormalsInterpolation(UsdGeom.Tokens.constant)  # not per-point
            stage.Save()

            builder = newton.ModelBuilder()
            with self.assertWarnsRegex(UserWarning, "not per-point"):
                builder.add_usd(str(usd_path))
            cable_map, _, _ = deformable_maps(builder)
            # The cable still imports (normals ignored, default segment orientation used).
            bodies, _ = cable_map["/World/Cable"]
            self.assertEqual(len(bodies), 3)

    def test_cable_primvars_normals_take_precedence(self):
        """Indexed primvars:normals take precedence over the schema normals attribute."""
        from pxr import Sdf, Usd, UsdGeom, UsdPhysics, Vt

        with tempfile.TemporaryDirectory() as tmpdir:
            usd_path = Path(tmpdir) / "cable_pvnormals.usda"
            stage = Usd.Stage.CreateNew(str(usd_path))
            UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
            UsdPhysics.Scene.Define(stage, "/PhysicsScene")
            pts = [(0.0, 0.0, 1.0), (0.1, 0.0, 1.0), (0.2, 0.0, 1.0), (0.3, 0.0, 1.0)]  # tangent +X
            curves = _add_cable_curve(stage, "/World/Cable", pts)
            # Schema normals say +Z; the indexed primvars:normals (+Y) must win.
            curves.CreateNormalsAttr([(0.0, 0.0, 1.0)] * len(pts))
            curves.SetNormalsInterpolation(UsdGeom.Tokens.vertex)
            pv = UsdGeom.PrimvarsAPI(curves.GetPrim()).CreatePrimvar(
                "normals", Sdf.ValueTypeNames.Normal3fArray, UsdGeom.Tokens.vertex
            )
            pv.Set([(0.0, 1.0, 0.0)])  # one unique value...
            pv.SetIndices(Vt.IntArray([0, 0, 0, 0]))  # ...indexed to all 4 points
            stage.Save()

            builder = newton.ModelBuilder()
            builder.add_usd(str(usd_path))
            cable_map, _, _ = deformable_maps(builder)
            bodies, _ = cable_map["/World/Cable"]
            for body in bodies:
                t = builder.body_q[body]
                q = wp.quat(float(t[3]), float(t[4]), float(t[5]), float(t[6]))
                y_world = np.array(wp.quat_rotate(q, wp.vec3(0.0, 1.0, 0.0)), dtype=np.float32)
                # +Y comes from primvars:normals; if the schema +Z had won it would be ~[0,0,1].
                np.testing.assert_allclose(y_world, [0.0, 1.0, 0.0], atol=1e-5)

    def test_instanced_cable_imports_proxies_not_prototype(self):
        """Instanced cables import once per instance proxy; the prototype master is skipped."""
        from pxr import Usd, UsdGeom, UsdPhysics

        with tempfile.TemporaryDirectory() as tmpdir:
            usd_path = Path(tmpdir) / "instanced.usda"
            stage = Usd.Stage.CreateNew(str(usd_path))
            UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
            UsdPhysics.Scene.Define(stage, "/PhysicsScene")
            pts = [(0.0, 0.0, 1.0), (0.1, 0.0, 1.0), (0.2, 0.0, 1.0), (0.3, 0.0, 1.0)]
            stage.CreateClassPrim("/Proto")  # template in a class, not the rendered scene
            _add_cable_curve(stage, "/Proto/Cable", pts)
            for name in ("A", "B"):
                inst = UsdGeom.Xform.Define(stage, f"/World/{name}")
                inst.GetPrim().GetReferences().AddInternalReference("/Proto")
                inst.GetPrim().SetInstanceable(True)
            stage.Save()

            builder = newton.ModelBuilder()
            builder.add_usd(str(usd_path))
            cable_map, _, _ = deformable_maps(builder)
            # Two instance proxies import; the prototype master (/__Prototype_*) is not.
            self.assertEqual(set(cable_map), {"/World/A/Cable", "/World/B/Cable"})

    def test_cable_replication_independent_per_world(self):
        """Replicating a cable across worlds yields independent, contiguous per-world segments (T5)."""
        from pxr import Usd, UsdGeom, UsdPhysics

        with tempfile.TemporaryDirectory() as tmpdir:
            usd_path = Path(tmpdir) / "cable.usda"
            stage = Usd.Stage.CreateNew(str(usd_path))
            UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
            UsdPhysics.Scene.Define(stage, "/PhysicsScene")
            pts = [(0.0, 0.0, 1.0), (0.1, 0.0, 1.0), (0.2, 0.0, 1.0), (0.3, 0.0, 1.0)]  # 3 segments
            _add_cable_curve(stage, "/World/Cable", pts)
            stage.Save()

            # Parse the cable into a prototype builder, then replicate across worlds.
            proto = newton.ModelBuilder()
            proto.add_usd(str(usd_path))
            cable_map, _, _ = deformable_maps(proto)
            base_bodies, _ = cable_map["/World/Cable"]
            self.assertEqual(base_bodies, list(range(proto.body_count)))  # cable is the whole prototype
            # The importer wraps the cable into its own "<path>_articulation", which replicate() copies per world.
            self.assertIn("/World/Cable_articulation", proto.articulation_label)

            num_envs = 3
            scene = newton.ModelBuilder()
            scene.replicate(proto, world_count=num_envs)

            nb = proto.body_count
            # One independent copy per world: counts scale, articulation repeats per env.
            self.assertEqual(scene.body_count, num_envs * nb)
            self.assertEqual(scene.articulation_label.count("/World/Cable_articulation"), num_envs)
            # Env e's cable segments are the contiguous block [e*nb : (e+1)*nb] - disjoint,
            # so state can be sliced as (num_envs, num_segments, ...).
            ranges = [list(range(e * nb, (e + 1) * nb)) for e in range(num_envs)]
            self.assertEqual(sorted(i for r in ranges for i in r), list(range(scene.body_count)))

    def test_imported_cable_is_wrapped_in_an_articulation(self):
        """add_usd wraps each cable into its own "<path>_articulation", so the model is finalize-ready
        without the caller wrapping the joints."""
        from pxr import Usd, UsdPhysics

        with tempfile.TemporaryDirectory() as tmpdir:
            usd_path = Path(tmpdir) / "cable_art.usda"
            stage = Usd.Stage.CreateNew(str(usd_path))
            UsdPhysics.Scene.Define(stage, "/PhysicsScene")
            pts = [(0.0, 0.0, 1.0), (0.1, 0.0, 1.0), (0.2, 0.0, 1.0), (0.3, 0.0, 1.0)]
            _add_cable_curve(stage, "/World/Cable", pts)
            stage.Save()

            builder = newton.ModelBuilder()
            builder.add_usd(str(usd_path))
            self.assertIn("/World/Cable_articulation", builder.articulation_label)
            builder.finalize()  # would raise on orphan cable joints if the cable were left unwrapped

    def test_periodic_cable_imports_closing_segment(self):
        """A periodic curve builds a body for the closing v[-1] -> v[0] segment."""
        from pxr import Usd, UsdPhysics

        with tempfile.TemporaryDirectory() as tmpdir:
            usd_path = Path(tmpdir) / "loop.usda"
            stage = Usd.Stage.CreateNew(str(usd_path))
            UsdPhysics.Scene.Define(stage, "/PhysicsScene")
            # 4 vertices -> 4 segments for a closed loop (incl. the wrap segment).
            pts = [(0.0, 0.0, 1.0), (1.0, 0.0, 1.0), (1.0, 1.0, 1.0), (0.0, 1.0, 1.0)]
            _add_cable_curve(stage, "/World/Cable", pts, periodic=True)
            stage.Save()

            builder = newton.ModelBuilder()
            builder.add_usd(str(usd_path))
            cable_map, _, _ = deformable_maps(builder)
            bodies, joints = cable_map["/World/Cable"]
            self.assertEqual(len(bodies), 4, "expected one body per segment, incl. the closing segment")
            self.assertEqual(len(joints), 4, "expected 3 chain joints + 1 loop joint")
            # The importer wraps the closed cable; add_rod keeps the loop-closing joint out of the tree.
            self.assertIn("/World/Cable_articulation", builder.articulation_label)

    def test_cable_group_remapped_after_collapse(self):
        """Cable group body indices still point at cable bodies after fixed-joint collapse."""
        from pxr import Usd, UsdGeom, UsdPhysics

        with tempfile.TemporaryDirectory() as tmpdir:
            usd_path = Path(tmpdir) / "mixed.usda"
            stage = Usd.Stage.CreateNew(str(usd_path))
            UsdPhysics.Scene.Define(stage, "/PhysicsScene")
            # Two rigid bodies joined by a fixed joint -> collapsed, reindexing all
            # bodies; these parse before the cable so the cable indices would shift.
            for name in ("A", "B"):
                body = UsdGeom.Xform.Define(stage, f"/World/{name}")
                UsdPhysics.RigidBodyAPI.Apply(body.GetPrim())
                UsdPhysics.CollisionAPI.Apply(body.GetPrim())
            fixed = UsdPhysics.FixedJoint.Define(stage, "/World/Fix")
            fixed.CreateBody0Rel().SetTargets(["/World/A"])
            fixed.CreateBody1Rel().SetTargets(["/World/B"])
            pts = [(0.0, 0.0, 1.0), (0.1, 0.0, 1.0), (0.2, 0.0, 1.0), (0.3, 0.0, 1.0)]
            _add_cable_curve(stage, "/World/Cable", pts)
            stage.Save()

            builder = newton.ModelBuilder()
            # The two rigid bodies' fixed joint has no articulation root, so the importer warns.
            with self.assertWarnsRegex(UserWarning, "No articulation was found"):
                builder.add_usd(str(usd_path), collapse_fixed_joints=True)
            cable_map, _, _ = deformable_maps(builder)
            bodies, _ = cable_map["/World/Cable"]
            self.assertTrue(all(0 <= b < builder.body_count for b in bodies), "cable body index out of range")
            self.assertTrue(
                all("/World/Cable" in builder.body_label[b] for b in bodies),
                "remapped cable indices point at non-cable bodies",
            )
            # The Model-level cable group range rides the same collapse remap.
            model = builder.finalize()
            b0, b1 = model.cable_body_range(model.cable_index("/World/Cable"))
            self.assertEqual((b0, b1), (min(bodies), max(bodies) + 1))
            self.assertTrue(all("/World/Cable" in model.body_label[b] for b in range(b0, b1)))

    def test_cable_negative_scale_mirrors_positions(self):
        """A reflective xformOp:scale mirrors cable body positions (parity preserved); a
        rotation+scale decomposition would drop the reflection and leave them un-mirrored."""
        from pxr import Gf, Usd, UsdGeom, UsdPhysics

        with tempfile.TemporaryDirectory() as tmpdir:
            usd_path = Path(tmpdir) / "cable_reflected.usda"
            stage = Usd.Stage.CreateNew(str(usd_path))
            UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
            UsdPhysics.Scene.Define(stage, "/PhysicsScene")
            pts = [(0.0, 0.0, 1.0), (0.1, 0.0, 1.0), (0.2, 0.0, 1.0), (0.3, 0.0, 1.0)]
            curves = _add_cable_curve(stage, "/World/Cable", pts)
            UsdGeom.Xformable(curves).AddScaleOp().Set(Gf.Vec3d(-1.0, 1.0, 1.0))
            stage.Save()

            builder = newton.ModelBuilder()
            builder.add_usd(str(usd_path))
            cable_map, _, _ = deformable_maps(builder)
            bodies, _ = cable_map["/World/Cable"]
            xs = [float(np.array(builder.body_q[b][:3])[0]) for b in bodies]
            # Body i origin sits at segment i's midpoint; midpoints 0.05/0.15/0.25 mirror to negative X.
            np.testing.assert_allclose(xs, [-0.05, -0.15, -0.25], atol=1e-5)

    def test_curve_to_curve_attachment_builds_rod_graph(self):
        """A curve->curve point attachment welds two curve deformables into one rod graph.

        The junction is topology, not a runtime constraint: it is consumed by the graph build
        (not surfaced as an attachment joint), the two curves share the welded node, and the whole
        component comes back pre-wrapped in a single articulation that finalizes.
        """
        from pxr import Usd, UsdGeom, UsdPhysics

        with tempfile.TemporaryDirectory() as tmpdir:
            usd_path = Path(tmpdir) / "rod_graph.usda"
            stage = Usd.Stage.CreateNew(str(usd_path))
            UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
            UsdPhysics.Scene.Define(stage, "/PhysicsScene")
            # Trunk along x (3 segments); branch goes +y from the trunk's interior point 1.
            trunk_pts = [(0.0, 0.0, 1.0), (0.1, 0.0, 1.0), (0.2, 0.0, 1.0), (0.3, 0.0, 1.0)]
            branch_pts = [(0.1, 0.0, 1.0), (0.1, 0.1, 1.0), (0.1, 0.2, 1.0)]
            _add_cable_curve(stage, "/World/Trunk", trunk_pts)
            _add_cable_curve(stage, "/World/Branch", branch_pts)
            # Weld branch point 0 to trunk point 1 (curve-to-curve junction).
            _add_physics_attachment(
                stage,
                "/World/Junction",
                src0="/World/Branch",
                src1="/World/Trunk",
                type0="point",
                type1="point",
                indices0=[0],
                indices1=[1],
            )
            stage.Save()

            builder = newton.ModelBuilder()
            result = builder.add_usd(str(usd_path))
            cable_map, _, _ = deformable_maps(builder)

            # Both curves import as one welded component; the junction is consumed as topology.
            self.assertIn("/World/Trunk", cable_map)
            self.assertIn("/World/Branch", cable_map)
            self.assertNotIn("/World/Junction", result["path_attachment_map"])
            self.assertNotIn("/World/Junction", result["path_attachment_attrs"])

            trunk_bodies, trunk_joints = cable_map["/World/Trunk"]
            branch_bodies, _ = cable_map["/World/Branch"]
            self.assertEqual(len(trunk_bodies), 3, "trunk has 3 segments")
            self.assertEqual(len(branch_bodies), 2, "branch has 2 segments")
            # Graph cables are returned pre-wrapped, so the caller does no articulation work.
            self.assertEqual(trunk_joints, [], "graph cable joints are pre-wrapped (empty)")
            self.assertEqual(builder.articulation_count, 1, "the welded component is one articulation")

            model = builder.finalize()
            self.assertEqual(model.body_count, 5)

    def test_welded_graph_degenerate_segment_skips_component(self):
        """A welded curve with a zero-length segment is rejected with a warning instead of aborting
        the whole import; the component's curves fall back to the per-curve pass."""
        from pxr import Usd, UsdGeom, UsdPhysics

        with tempfile.TemporaryDirectory() as tmpdir:
            usd_path = Path(tmpdir) / "graph_degenerate.usda"
            stage = Usd.Stage.CreateNew(str(usd_path))
            UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
            UsdPhysics.Scene.Define(stage, "/PhysicsScene")
            trunk_pts = [(0.0, 0.0, 1.0), (0.1, 0.0, 1.0), (0.2, 0.0, 1.0), (0.3, 0.0, 1.0)]
            # The branch has a duplicate consecutive point -> a zero-length segment.
            branch_pts = [(0.1, 0.0, 1.0), (0.1, 0.1, 1.0), (0.1, 0.1, 1.0)]
            _add_cable_curve(stage, "/World/Trunk", trunk_pts)
            _add_cable_curve(stage, "/World/Branch", branch_pts)
            _add_physics_attachment(
                stage,
                "/World/Junction",
                src0="/World/Branch",
                src1="/World/Trunk",
                type0="point",
                type1="point",
                indices0=[0],
                indices1=[1],
            )
            stage.Save()

            builder = newton.ModelBuilder()
            with self.assertWarnsRegex(UserWarning, "zero-length segment"):
                result = builder.add_usd(str(usd_path))
            cable_map, _, _ = deformable_maps(builder)
            # The import did not abort; the valid trunk imported as a single (unwrapped) cable.
            self.assertIn("/World/Trunk", cable_map)
            _bodies, joints = cable_map["/World/Trunk"]
            self.assertNotEqual(joints, [], "the skipped component leaves the trunk as a single cable")
            self.assertNotIn("graph_component", result["path_cable_attrs"]["/World/Trunk"])

    def test_welded_graph_collapse_with_masses_falls_back(self):
        """Welding two adjacent points of one curve onto the same node collapses a segment. With
        authored physics:masses the surviving body count no longer matches the per-point lumping; the
        importer warns and falls back instead of raising and aborting."""
        from pxr import Sdf, Usd, UsdGeom, UsdPhysics

        with tempfile.TemporaryDirectory() as tmpdir:
            usd_path = Path(tmpdir) / "graph_collapse.usda"
            stage = Usd.Stage.CreateNew(str(usd_path))
            UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
            UsdPhysics.Scene.Define(stage, "/PhysicsScene")
            trunk_pts = [(0.0, 0.0, 1.0), (0.1, 0.0, 1.0), (0.2, 0.0, 1.0), (0.3, 0.0, 1.0)]
            # Branch points 0 and 1 both sit within the weld coincidence tolerance of trunk
            # point 1, so both weld onto that node and the branch edge (0, 1) collapses.
            branch_pts = [(0.1, 0.0, 1.0), (0.1, 0.0005, 1.0), (0.1, 0.1, 1.0), (0.1, 0.15, 1.0)]
            _add_cable_curve(stage, "/World/Trunk", trunk_pts)
            branch = _add_cable_curve(stage, "/World/Branch", branch_pts)
            branch.GetPrim().CreateAttribute("physics:masses", Sdf.ValueTypeNames.FloatArray).Set([1.0, 1.0, 1.0, 1.0])
            # Weld branch points 0 and 1 both onto trunk point 1 -> collapses branch edge (0, 1).
            _add_physics_attachment(
                stage,
                "/World/Junction",
                src0="/World/Branch",
                src1="/World/Trunk",
                type0="point",
                type1="point",
                indices0=[0, 1],
                indices1=[1, 1],
            )
            stage.Save()

            builder = newton.ModelBuilder()
            with self.assertWarnsRegex(UserWarning, "collapsed a segment"):
                result = builder.add_usd(str(usd_path))
            # The welded graph still built (no abort); both curves are present in the graph component.
            self.assertIn("graph_component", result["path_cable_attrs"]["/World/Trunk"])
            self.assertIn("graph_component", result["path_cable_attrs"]["/World/Branch"])
            self.assertEqual(builder.finalize().body_count, builder.body_count)

    def test_welded_graph_drops_rest_shape_warns(self):
        """A welded curve's authored restShapePoints cannot be honored by add_rod_graph's scalar
        stiffness, so the importer warns rather than silently using the current segment lengths."""
        from pxr import Sdf, Usd, UsdGeom, UsdPhysics

        with tempfile.TemporaryDirectory() as tmpdir:
            usd_path = Path(tmpdir) / "graph_rest.usda"
            stage = Usd.Stage.CreateNew(str(usd_path))
            UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
            UsdPhysics.Scene.Define(stage, "/PhysicsScene")
            trunk_pts = [(0.0, 0.0, 1.0), (0.1, 0.0, 1.0), (0.2, 0.0, 1.0), (0.3, 0.0, 1.0)]
            branch_pts = [(0.1, 0.0, 1.0), (0.1, 0.1, 1.0), (0.1, 0.2, 1.0)]
            _add_cable_curve(stage, "/World/Trunk", trunk_pts)
            branch = _add_cable_curve(stage, "/World/Branch", branch_pts)
            branch.GetPrim().CreateAttribute("physics:restShapePoints", Sdf.ValueTypeNames.Point3fArray).Set(branch_pts)
            _add_physics_attachment(
                stage,
                "/World/Junction",
                src0="/World/Branch",
                src1="/World/Trunk",
                type0="point",
                type1="point",
                indices0=[0],
                indices1=[1],
            )
            stage.Save()

            builder = newton.ModelBuilder()
            with self.assertWarnsRegex(UserWarning, "restShapePoints is dropped"):
                result = builder.add_usd(str(usd_path))
            self.assertIn("graph_component", result["path_cable_attrs"]["/World/Branch"])

    def test_ignored_curve_to_curve_junction_does_not_weld(self):
        """An ``ignore_paths`` junction must not alter topology: the curves stay independent.

        Without honoring ``ignore_paths`` in the graph pre-pass, an ignored junction would
        still weld its curves into a pre-wrapped rod graph (and silently vanish from the
        attachment maps).
        """
        from pxr import Usd, UsdGeom, UsdPhysics

        with tempfile.TemporaryDirectory() as tmpdir:
            usd_path = Path(tmpdir) / "ignored_junction.usda"
            stage = Usd.Stage.CreateNew(str(usd_path))
            UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
            UsdPhysics.Scene.Define(stage, "/PhysicsScene")
            trunk_pts = [(0.0, 0.0, 1.0), (0.1, 0.0, 1.0), (0.2, 0.0, 1.0), (0.3, 0.0, 1.0)]
            branch_pts = [(0.1, 0.0, 1.0), (0.1, 0.1, 1.0), (0.1, 0.2, 1.0)]
            _add_cable_curve(stage, "/World/Trunk", trunk_pts)
            _add_cable_curve(stage, "/World/Branch", branch_pts)
            _add_physics_attachment(
                stage,
                "/World/Junction",
                src0="/World/Branch",
                src1="/World/Trunk",
                type0="point",
                type1="point",
                indices0=[0],
                indices1=[1],
            )
            stage.Save()

            builder = newton.ModelBuilder()
            result = builder.add_usd(str(usd_path), ignore_paths=["/World/Junction"])
            cable_map, _, _ = deformable_maps(builder)

            # Both curves still import, but as independent single cables (not a welded graph):
            # single cables expose their cable joints for the caller to wrap, so joints are non-empty.
            trunk_bodies, trunk_joints = cable_map["/World/Trunk"]
            _branch_bodies, branch_joints = cable_map["/World/Branch"]
            self.assertEqual(len(trunk_bodies), 3)
            self.assertNotEqual(trunk_joints, [], "an ignored junction must leave the cable unwelded")
            self.assertNotEqual(branch_joints, [], "an ignored junction must leave the cable unwelded")
            self.assertNotIn("graph_component", result["path_cable_attrs"]["/World/Trunk"])
            # The ignored junction is consumed by nothing: it is absent from the attachment maps.
            self.assertNotIn("/World/Junction", result["path_attachment_map"])
            self.assertNotIn("/World/Junction", result["path_attachment_attrs"])

    def test_curve_to_curve_junction_out_of_range_index_warns_and_skips(self):
        """An out-of-range junction index warns and skips the weld instead of raising IndexError."""
        from pxr import Usd, UsdGeom, UsdPhysics

        with tempfile.TemporaryDirectory() as tmpdir:
            usd_path = Path(tmpdir) / "junction_oob.usda"
            stage = Usd.Stage.CreateNew(str(usd_path))
            UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
            UsdPhysics.Scene.Define(stage, "/PhysicsScene")
            trunk_pts = [(0.0, 0.0, 1.0), (0.1, 0.0, 1.0), (0.2, 0.0, 1.0), (0.3, 0.0, 1.0)]
            branch_pts = [(0.1, 0.0, 1.0), (0.1, 0.1, 1.0), (0.1, 0.2, 1.0)]
            _add_cable_curve(stage, "/World/Trunk", trunk_pts)
            _add_cable_curve(stage, "/World/Branch", branch_pts)
            _add_physics_attachment(
                stage,
                "/World/Junction",
                src0="/World/Branch",
                src1="/World/Trunk",
                type0="point",
                type1="point",
                indices0=[99],  # out of range for the 3-point branch
                indices1=[1],
            )
            stage.Save()

            builder = newton.ModelBuilder()
            with self.assertWarnsRegex(UserWarning, "out of range"):
                result = builder.add_usd(str(usd_path))
            cable_map, _, _ = deformable_maps(builder)

            # The malformed junction does not weld: both curves import independently.
            _trunk_bodies, trunk_joints = cable_map["/World/Trunk"]
            self.assertIn("/World/Branch", cable_map)
            self.assertNotEqual(trunk_joints, [])
            self.assertNotIn("graph_component", result["path_cable_attrs"]["/World/Trunk"])

    def test_curve_to_curve_junction_mismatched_indices_warns_and_skips(self):
        """Mismatched-length junction index arrays warn and skip rather than reusing indices1[0]."""
        from pxr import Usd, UsdGeom, UsdPhysics

        with tempfile.TemporaryDirectory() as tmpdir:
            usd_path = Path(tmpdir) / "junction_mismatch.usda"
            stage = Usd.Stage.CreateNew(str(usd_path))
            UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
            UsdPhysics.Scene.Define(stage, "/PhysicsScene")
            trunk_pts = [(0.0, 0.0, 1.0), (0.1, 0.0, 1.0), (0.2, 0.0, 1.0), (0.3, 0.0, 1.0)]
            branch_pts = [(0.1, 0.0, 1.0), (0.1, 0.1, 1.0), (0.1, 0.2, 1.0)]
            _add_cable_curve(stage, "/World/Trunk", trunk_pts)
            _add_cable_curve(stage, "/World/Branch", branch_pts)
            _add_physics_attachment(
                stage,
                "/World/Junction",
                src0="/World/Branch",
                src1="/World/Trunk",
                type0="point",
                type1="point",
                indices0=[0, 1],  # two sites
                indices1=[1],  # but only one partner
            )
            stage.Save()

            builder = newton.ModelBuilder()
            with self.assertWarnsRegex(UserWarning, "differ in length"):
                result = builder.add_usd(str(usd_path))
            cable_map, _, _ = deformable_maps(builder)

            _trunk_bodies, trunk_joints = cable_map["/World/Trunk"]
            self.assertNotEqual(trunk_joints, [])
            self.assertNotIn("graph_component", result["path_cable_attrs"]["/World/Trunk"])

    def test_heterogeneous_welded_cable_materials_warn(self):
        """Welding curves with different materials warns that one representative is used.

        add_rod_graph applies one scalar radius/density/stiffness per component, so the graph
        flattens to the first curve's material; the disagreement must be surfaced, not silent.
        """
        from pxr import Usd, UsdGeom, UsdPhysics

        with tempfile.TemporaryDirectory() as tmpdir:
            usd_path = Path(tmpdir) / "heterogeneous_weld.usda"
            stage = Usd.Stage.CreateNew(str(usd_path))
            UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
            UsdPhysics.Scene.Define(stage, "/PhysicsScene")
            trunk_pts = [(0.0, 0.0, 1.0), (0.1, 0.0, 1.0), (0.2, 0.0, 1.0), (0.3, 0.0, 1.0)]
            branch_pts = [(0.1, 0.0, 1.0), (0.1, 0.1, 1.0), (0.1, 0.2, 1.0)]
            trunk = _add_cable_curve(stage, "/World/Trunk", trunk_pts)
            branch = _add_cable_curve(stage, "/World/Branch", branch_pts)
            _bind_deformable_material(stage, trunk.GetPrim(), "/World/TrunkMat", thickness=0.02)
            _bind_deformable_material(stage, branch.GetPrim(), "/World/BranchMat", thickness=0.06)
            _add_physics_attachment(
                stage,
                "/World/Junction",
                src0="/World/Branch",
                src1="/World/Trunk",
                type0="point",
                type1="point",
                indices0=[0],
                indices1=[1],
            )
            stage.Save()

            builder = newton.ModelBuilder()
            with self.assertWarnsRegex(UserWarning, "differing radius/density/stiffness"):
                result = builder.add_usd(str(usd_path))

            # Still welded into one graph; each curve keeps its own authored material in the attrs.
            self.assertIn("graph_component", result["path_cable_attrs"]["/World/Trunk"])
            self.assertIn("graph_component", result["path_cable_attrs"]["/World/Branch"])
            self.assertAlmostEqual(result["path_cable_attrs"]["/World/Trunk"]["material"]["thickness"], 0.02, places=5)
            self.assertAlmostEqual(result["path_cable_attrs"]["/World/Branch"]["material"]["thickness"], 0.06, places=5)

    def test_cable_body_mass_rescales_total(self):
        """PhysicsDeformableBodyAPI.mass rescales the rigid cable's segment masses."""
        from pxr import Usd, UsdPhysics

        with tempfile.TemporaryDirectory() as tmpdir:
            usd_path = Path(tmpdir) / "cable_mass.usda"
            stage = Usd.Stage.CreateNew(str(usd_path))
            UsdPhysics.Scene.Define(stage, "/PhysicsScene")
            pts = [(0.0, 0.0, 1.0), (0.1, 0.0, 1.0), (0.2, 0.0, 1.0), (0.3, 0.0, 1.0)]
            curves = _add_cable_curve(stage, "/World/Cable", pts)
            _apply_deformable_body_api(curves.GetPrim(), mass=2.5)
            stage.Save()

            builder = newton.ModelBuilder()
            builder.add_usd(str(usd_path))
            cable_map, _, _ = deformable_maps(builder)
            bodies, _ = cable_map["/World/Cable"]
            self.assertAlmostEqual(sum(builder.body_mass[b] for b in bodies), 2.5, places=4)

    def test_cable_default_radius_scales_with_stage_units(self):
        """With no authored thickness the importer assumes a default radius derived from the stage's
        linear unit, so it is the same physical size (~0.05 m) on a centimeter stage as on a meter
        stage, and it warns that a default was assumed (rather than a meters-flavored literal)."""
        from pxr import Usd, UsdGeom, UsdPhysics

        def capsule_radius(meters_per_unit):
            with tempfile.TemporaryDirectory() as tmpdir:
                usd_path = Path(tmpdir) / "cable.usda"
                stage = Usd.Stage.CreateNew(str(usd_path))
                UsdGeom.SetStageMetersPerUnit(stage, meters_per_unit)
                UsdPhysics.Scene.Define(stage, "/PhysicsScene")
                pts = [(0.0, 0.0, 1.0), (0.1, 0.0, 1.0), (0.2, 0.0, 1.0), (0.3, 0.0, 1.0)]
                _add_cable_curve(stage, "/World/Cable", pts, thickness=None)  # no bound material -> no thickness
                stage.Save()
                builder = newton.ModelBuilder()
                with self.assertWarnsRegex(UserWarning, "no cable thickness"):
                    builder.add_usd(str(usd_path))
                return float(builder.shape_scale[0][0])  # capsule radius is stored as scale.x

        # ~0.05 m on a meter stage; 0.05 / 0.01 = 5 stage units on a cm stage (still ~0.05 m physical).
        self.assertAlmostEqual(capsule_radius(1.0), 0.05, places=4)
        self.assertAlmostEqual(capsule_radius(0.01), 5.0, places=3)

        with tempfile.TemporaryDirectory() as tmpdir:
            usd_path = Path(tmpdir) / "cable.usda"
            stage = Usd.Stage.CreateNew(str(usd_path))
            UsdPhysics.Scene.Define(stage, "/PhysicsScene")
            pts = [(0.0, 0.0, 1.0), (0.1, 0.0, 1.0), (0.2, 0.0, 1.0), (0.3, 0.0, 1.0)]
            _add_cable_curve(stage, "/World/Cable", pts, thickness=None)
            stage.Save()
            builder = newton.ModelBuilder()
            with self.assertWarnsRegex(UserWarning, "no cable thickness"):
                builder.add_usd(str(usd_path))

    def test_cable_per_point_masses_lump_onto_segments(self):
        """Per-point physics:masses are lumped onto the segments they border, so a front-heavy mass
        array yields a front-heavy cable (not a uniform one) while preserving the total. Each point's
        mass splits between its adjacent segments; the two endpoints border a single segment each."""
        from pxr import Sdf, Usd, UsdPhysics

        with tempfile.TemporaryDirectory() as tmpdir:
            usd_path = Path(tmpdir) / "cable_asym.usda"
            stage = Usd.Stage.CreateNew(str(usd_path))
            UsdPhysics.Scene.Define(stage, "/PhysicsScene")
            pts = [(0.0, 0.0, 1.0), (0.1, 0.0, 1.0), (0.2, 0.0, 1.0), (0.3, 0.0, 1.0)]  # 4 points -> 3 segments
            curves = _add_cable_curve(stage, "/World/Cable", pts)
            masses = [10.0, 1.0, 1.0, 1.0]  # front-heavy, length == points
            curves.GetPrim().CreateAttribute("physics:masses", Sdf.ValueTypeNames.FloatArray).Set(masses)
            stage.Save()

            builder = newton.ModelBuilder()
            builder.add_usd(str(usd_path))
            cable_map, _, _ = deformable_maps(builder)
            bodies, _ = cable_map["/World/Cable"]
            seg_masses = [builder.body_mass[b] for b in bodies]
            # Lumping (endpoints contribute their full mass to their one segment, interior points
            # split): seg0 = m0 + m1/2, seg1 = m1/2 + m2/2, seg2 = m2/2 + m3.
            self.assertEqual(len(seg_masses), 3)
            self.assertAlmostEqual(seg_masses[0], 10.0 + 0.5, places=4)
            self.assertAlmostEqual(seg_masses[1], 0.5 + 0.5, places=4)
            self.assertAlmostEqual(seg_masses[2], 0.5 + 1.0, places=4)
            # Total is preserved and the front-heavy profile survives (not flattened).
            self.assertAlmostEqual(sum(seg_masses), sum(masses), places=4)
            self.assertGreater(seg_masses[0], seg_masses[2])

    def test_cable_masses_length_mismatch_is_ignored(self):
        """A physics:masses array whose length != curve points warns and is ignored."""
        from pxr import Sdf, Usd, UsdPhysics

        def total_cable_mass(masses=None):
            with tempfile.TemporaryDirectory() as tmpdir:
                usd_path = Path(tmpdir) / "cable.usda"
                stage = Usd.Stage.CreateNew(str(usd_path))
                UsdPhysics.Scene.Define(stage, "/PhysicsScene")
                pts = [(0.0, 0.0, 1.0), (0.1, 0.0, 1.0), (0.2, 0.0, 1.0), (0.3, 0.0, 1.0)]  # 4 points
                curves = _add_cable_curve(stage, "/World/Cable", pts)
                # Bind a thickness so the only warning under test is the mass-length mismatch.
                _bind_deformable_material(stage, curves.GetPrim(), "/World/CableMat", thickness=0.02)
                if masses is not None:
                    curves.GetPrim().CreateAttribute("physics:masses", Sdf.ValueTypeNames.FloatArray).Set(masses)
                stage.Save()
                builder = newton.ModelBuilder()
                builder.add_usd(str(usd_path))
                cable_map, _, _ = deformable_maps(builder)
                bodies, _ = cable_map["/World/Cable"]
                return sum(builder.body_mass[b] for b in bodies)

        baseline = total_cable_mass()
        with self.assertWarnsRegex(UserWarning, r"!= 4 authored curve points"):
            mismatched = total_cable_mass(masses=[1.0, 2.0, 3.0])  # length 3 != 4 points
        self.assertAlmostEqual(mismatched, baseline, places=6)


_CABLE_ASSET_USDA = """#usda 1.0
(
    defaultPrim = "World"
    metersPerUnit = 1.0
    upAxis = "Z"
)

def Xform "World"
{
    def PhysicsScene "PhysicsScene"
    {
    }

    def Material "CableMat" (
        prepend apiSchemas = ["PhysicsCurvesDeformableMaterialAPI"]
    )
    {
        float physics:thickness = 0.02
        float physics:stretchStiffness = 2000000
        float physics:bendStiffness = 50000
        float physics:density = 1000
    }

    def BasisCurves "Cable" (
        prepend apiSchemas = ["PhysicsCurvesDeformableSimAPI", "MaterialBindingAPI"]
    )
    {
        uniform token type = "linear"
        uniform token wrap = "nonperiodic"
        int[] curveVertexCounts = [6]
        point3f[] points = [(0, 0, 1), (0.1, 0, 1), (0.2, 0, 1), (0.3, 0, 1), (0.4, 0, 1), (0.5, 0, 1)]
        float[] widths = [0.02, 0.02, 0.02, 0.02, 0.02, 0.02]
        rel material:binding:physics = </World/CableMat>
    }
}
"""


def _stage_from_usda(usda: str):
    """Open an in-memory USD stage from an inline ``.usda`` string."""
    from pxr import Usd

    stage = Usd.Stage.CreateInMemory()
    stage.GetRootLayer().ImportFromString(usda)
    return stage


@unittest.skipUnless(USD_AVAILABLE, "Requires usd-core")
class TestUSDDeformableCableAsset(unittest.TestCase):
    """Round-trip an authored .usda cable asset through the importer and the VBD solver (T7)."""

    def test_asset_parses_to_cable(self):
        """The authored .usda cable asset imports into rod bodies + cable joints (device-free)."""
        builder = newton.ModelBuilder()
        builder.add_usd(_stage_from_usda(_CABLE_ASSET_USDA))
        cable_map, _, _ = deformable_maps(builder)
        bodies, joints = cable_map["/World/Cable"]
        # 6 points -> 5 segments -> 5 bodies, 4 cable joints (open chain).
        self.assertEqual(len(bodies), 5)
        self.assertEqual(len(joints), 4)

    def test_asset_cable_simulates(self, device=None):
        """After parsing, the asset cable runs through SolverVBD and stays finite (mirrors example test_final)."""
        if device is None or not wp.get_device(device).is_cuda:
            self.skipTest("VBD cable simulation requires a CUDA device")

        with wp.ScopedDevice(device):
            builder = newton.ModelBuilder()
            builder.add_usd(_stage_from_usda(_CABLE_ASSET_USDA))  # cables are wrapped by the importer
            builder.color()  # SolverVBD requires a coloring before finalize.
            model = builder.finalize()

            solver = newton.solvers.SolverVBD(model, iterations=10, rigid_body_contact_buffer_size=64)
            state_0 = model.state()
            state_1 = model.state()
            control = model.control()
            pipeline = newton.CollisionPipeline(model, contact_matching="latest")
            contacts = model.contacts(collision_pipeline=pipeline)

            dt = 1.0 / 240.0
            for _ in range(20):
                state_0.clear_forces()
                model.collide(state_0, contacts)
                solver.step(state_0, state_1, control, contacts, dt)
                state_0, state_1 = state_1, state_0

            body_q = state_0.body_q.numpy()
            body_qd = state_0.body_qd.numpy()
            self.assertTrue(np.isfinite(body_q).all(), "non-finite cable body positions after stepping")
            self.assertTrue(np.isfinite(body_qd).all(), "non-finite cable body velocities after stepping")
            self.assertTrue((np.abs(body_qd) < 5.0e2).all(), "cable body velocities diverged")


devices = get_selected_cuda_test_devices()
add_function_test(
    TestUSDDeformableCableAsset,
    "test_asset_cable_simulates",
    TestUSDDeformableCableAsset.test_asset_cable_simulates,
    devices=devices,
)


if __name__ == "__main__":
    unittest.main(verbosity=2)
