# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Tests for USD curve-deformable (cable) import: topology, materials, masses, normals,
transforms, instancing, graph welding, and curve-to-curve junctions."""

import math
import unittest

import numpy as np
import warp as wp

import newton
from newton.tests._usd_deformable_test_utils import (
    _add_cable_curve,
    _add_physics_attachment,
    _apply_deformable_body_api,
    _bind_deformable_material,
    _deformable_stage,
    group_labels,
    group_range,
)
from newton.tests.unittest_utils import USD_AVAILABLE
from newton.usd import SchemaResolverPhysx


@unittest.skipUnless(USD_AVAILABLE, "Requires usd-core")
class TestUSDDeformableCable(unittest.TestCase):
    """Curve-deformable (cable) parsing into rods of capsule bodies + cable joints."""

    @staticmethod
    def _author_attached_cable_pair(*, gap, stiffness=None, damping=None):
        """Two 4-point cables separated by ``gap`` in y with a point->point attachment
        (P0 of B onto P0 of A); returns the stage."""
        stage = _deformable_stage()
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
        return stage

    def test_attachment_weld_policy_rejects_compliant_or_apart(self):
        """A curve-to-curve attachment welds only when hard AND coincident: a compliant
        (zero or finite stiffness) or spatially-apart junction leaves two independent
        cables, moves no geometry, and preserves the attachment as unsupported."""
        cases = (
            # (name, fixture kwargs, warning regex, CableB stays at the authored gap)
            ("zero_stiffness", {"gap": 10.0, "stiffness": 0.0}, r"finite stiffness/damping; \S*\s*not welded", True),
            ("finite_stiffness", {"gap": 0.0, "stiffness": 1.0e4}, "finite stiffness/damping", False),
            ("hard_apart", {"gap": 10.0}, "not coincident", True),
        )
        for name, kwargs, regex, check_gap in cases:
            with self.subTest(case=name):
                stage = self._author_attached_cable_pair(**kwargs)

                builder = newton.ModelBuilder()
                with self.assertWarnsRegex(UserWarning, regex):
                    result = builder.add_usd(stage, deformable_results=True)
                # Two independent cables in their own articulations.
                self.assertEqual(len(group_labels(builder, "cable")), 2)
                self.assertEqual(len(builder.articulation_label), 2)
                if check_gap:
                    # The rejected junction must not snap the cables together: CableB stays at y=10.
                    bb0, _ = group_range(builder, "cable", "/World/CableB", "body")
                    self.assertAlmostEqual(float(builder.body_q[bb0][1]), 10.0, places=4)
                # The attachment is preserved as unsupported, not silently consumed.
                attrs = result["path_attachment_attrs"]["/World/Junction"]
                self.assertIn("unsupported_reason", attrs)
                if name == "zero_stiffness":
                    self.assertEqual(attrs["stiffness"], 0.0)

    def test_hard_coincident_junction_welds_rod_graph(self):
        """A hard, coincident curve-to-curve attachment welds the curves into one rod graph.

        The junction is topology, not a runtime constraint: it is consumed by the graph build
        (absent from the attachment maps), the curves share one graph_component, and the welded
        component comes back pre-wrapped in a single articulation that finalizes.
        """
        with self.subTest(weld="end_to_end"):
            stage = self._author_attached_cable_pair(gap=0.0)

            builder = newton.ModelBuilder()
            result = builder.add_usd(stage, deformable_results=True)
            self.assertEqual(len(builder.articulation_label), 1)
            self.assertEqual(
                result["path_cable_attrs"]["/World/CableA"]["graph_component"],
                result["path_cable_attrs"]["/World/CableB"]["graph_component"],
            )
            self.assertNotIn("/World/Junction", result["path_attachment_attrs"])
            self.assertEqual(len(group_labels(builder, "cable")), 2)

        with self.subTest(weld="branch_onto_interior_point"):
            stage = _deformable_stage()
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

            builder = newton.ModelBuilder()
            result = builder.add_usd(stage, deformable_results=True)

            # Both curves import as one welded component; the junction is consumed as topology.
            self.assertIn("/World/Trunk", group_labels(builder, "cable"))
            self.assertIn("/World/Branch", group_labels(builder, "cable"))
            self.assertNotIn("/World/Junction", result["path_attachment_map"])
            self.assertNotIn("/World/Junction", result["path_attachment_attrs"])

            tb0, tb1 = group_range(builder, "cable", "/World/Trunk", "body")
            tj0, tj1 = group_range(builder, "cable", "/World/Trunk", "joint")
            bb0, bb1 = group_range(builder, "cable", "/World/Branch", "body")
            self.assertEqual(tb1 - tb0, 3, "trunk has 3 segments")
            self.assertEqual(bb1 - bb0, 2, "branch has 2 segments")
            # Graph cables are returned pre-wrapped, so the caller does no articulation work.
            self.assertEqual(tj1 - tj0, 0, "graph cable joints are pre-wrapped (empty)")
            self.assertEqual(builder.articulation_count, 1, "the welded component is one articulation")

            model = builder.finalize()
            self.assertEqual(model.body_count, 5)

    def test_cable_material_maps_to_rod_stiffness(self):
        """Bound curve-deformable material -> radius + per-joint stretch/bend stiffness.

        Authored zero stiffness (range [0, inf)) is preserved, not replaced by add_rod's
        default, and the shear/twist moduli the rod cannot express warn and surface as-authored
        in path_cable_attrs for solvers with richer cable models.
        """
        # 3 segments of length 0.1 along x.
        pts = [(0.0, 0.0, 1.0), (0.1, 0.0, 1.0), (0.2, 0.0, 1.0), (0.3, 0.0, 1.0)]

        with self.subTest(material="full_moduli"):
            stage = _deformable_stage(up_axis="y")
            curves = _add_cable_curve(stage, "/World/Cable", pts, thickness=None)
            thickness, stretch_mod, bend_mod = 0.02, 2.0e6, 3.0e5
            _bind_deformable_material(
                stage,
                curves.GetPrim(),
                "/World/CableMat",
                thickness=thickness,
                density=1000.0,
                stretchStiffness=stretch_mod,
                bendStiffness=bend_mod,
                shearStiffness=3.0,
                twistStiffness=4.0,
            )

            builder = newton.ModelBuilder()
            # shear / twist are preserved in the attrs but cannot be expressed by the rod, so the importer warns.
            with self.assertWarnsRegex(UserWarning, "cannot be expressed"):
                result = builder.add_usd(stage, deformable_results=True)
            b0, b1 = group_range(builder, "cable", "/World/Cable", "body")
            j0, _ = group_range(builder, "cable", "/World/Cable", "joint")
            self.assertEqual(b1 - b0, 3)

            # radius = thickness / 2; stretch/bend converted with A/L, I/L.
            r = 0.5 * thickness
            seg_len = 0.3 / 3
            area = math.pi * r * r
            inertia = 0.25 * math.pi * r**4
            expected_stretch = stretch_mod * area / seg_len
            expected_bend = bend_mod * inertia / seg_len

            # Cable joints store stretch in the linear DOF target_ke, bend in the angular.
            dof0 = builder.joint_qd_start[j0]
            ke = builder.joint_target_ke
            self.assertAlmostEqual(ke[dof0], expected_stretch, delta=expected_stretch * 1e-3)
            self.assertAlmostEqual(ke[dof0 + 1], expected_bend, delta=expected_bend * 1e-3)

            # The as-authored material - including the dropped shear/twist moduli - is preserved.
            attrs = result["path_cable_attrs"]["/World/Cable"]
            mat = attrs["material"]
            self.assertAlmostEqual(mat["shearStiffness"], 3.0, places=5)
            self.assertAlmostEqual(mat["twistStiffness"], 4.0, places=5)
            self.assertAlmostEqual(mat["bendStiffness"], bend_mod, places=2)
            self.assertFalse(attrs["closed"])
            self.assertIsNotNone(attrs["resolved_density"])

        with self.subTest(material="authored_zero_stiffness"):
            stage = _deformable_stage(up_axis="y")
            curves = _add_cable_curve(stage, "/World/Cable", pts, thickness=None)
            _bind_deformable_material(
                stage,
                curves.GetPrim(),
                "/World/CableMat",
                thickness=0.02,
                stretchStiffness=0.0,
                bendStiffness=3.0e5,
            )

            builder = newton.ModelBuilder()
            result = builder.add_usd(stage, deformable_results=True)
            j0, _ = group_range(builder, "cable", "/World/Cable", "joint")
            # Stretch DOF target_ke is the authored 0.0, not add_rod's 1.0e5 default.
            dof0 = builder.joint_qd_start[j0]
            self.assertEqual(builder.joint_target_ke[dof0], 0.0)
            self.assertEqual(result["path_cable_attrs"]["/World/Cable"]["material"]["stretchStiffness"], 0.0)

    def test_cable_rest_length_from_rest_shape_points(self):
        """Per-joint stiffness uses the rest centerline (restShapePoints), not the possibly-deformed
        points, so an authored rest shape sets the rest length L in E*A/L (proposal: rest segment
        lengths derive from restShapePoints)."""
        from pxr import Sdf

        stage = _deformable_stage(up_axis="y")
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

        builder = newton.ModelBuilder()
        # restShapePoints only normalizes stiffness (it does not set an initial strain state), so it warns.
        with self.assertWarnsRegex(UserWarning, "restShapePoints only sets the rest length"):
            builder.add_usd(stage)
        j0, _ = group_range(builder, "cable", "/World/Cable", "joint")
        r = 0.5 * thickness
        area = math.pi * r * r
        expected = stretch_mod * area / 0.1  # rest length 0.1, not the 0.2 deformed segments
        dof0 = builder.joint_qd_start[j0]
        self.assertAlmostEqual(builder.joint_target_ke[dof0], expected, delta=expected * 1e-3)

    def test_non_linear_curve_is_skipped(self):
        """A non-linear (cubic) curve-deformable warns and is skipped (cable import is linear-only)."""
        from pxr import UsdGeom

        stage = _deformable_stage(up_axis="y")
        curves = UsdGeom.BasisCurves.Define(stage, "/World/Cubic")
        curves.CreateTypeAttr().Set(UsdGeom.Tokens.cubic)
        curves.CreatePointsAttr([(0.0, 0.0, 1.0), (0.1, 0.0, 1.0), (0.2, 0.0, 1.0), (0.3, 0.0, 1.0)])
        curves.CreateCurveVertexCountsAttr([4])
        curves.GetPrim().AddAppliedSchema("PhysicsCurvesDeformableSimAPI")

        builder = newton.ModelBuilder()
        with self.assertWarnsRegex(UserWarning, "non-linear"):
            builder.add_usd(stage)
        self.assertEqual(group_labels(builder, "cable"), [])
        self.assertEqual(builder.body_count, 0)

    def test_cable_material_without_family_api_is_ignored(self):
        """A physics-bound material lacking PhysicsCurvesDeformableMaterialAPI is not read as a cable material."""
        from pxr import Sdf, UsdShade

        stage = _deformable_stage(up_axis="y")
        pts = [(0.0, 0.0, 1.0), (0.1, 0.0, 1.0), (0.2, 0.0, 1.0), (0.3, 0.0, 1.0)]
        curves = _add_cable_curve(stage, "/World/Cable", pts, thickness=None)
        # Material carries cable-shaped attributes but does NOT declare the family API.
        mat = UsdShade.Material.Define(stage, "/World/Mat")
        mat.GetPrim().CreateAttribute("physics:stretchStiffness", Sdf.ValueTypeNames.Float).Set(2.0e6)
        mat.GetPrim().CreateAttribute("physics:thickness", Sdf.ValueTypeNames.Float).Set(0.02)
        UsdShade.MaterialBindingAPI.Apply(curves.GetPrim()).Bind(mat, materialPurpose="physics")

        builder = newton.ModelBuilder()
        # The family-less material is ignored, so the cable falls back to the default radius and warns.
        with self.assertWarnsRegex(UserWarning, "no cable thickness"):
            result = builder.add_usd(stage, deformable_results=True)
        # Without the family API the material is ignored: no attrs, default rod stiffness.
        self.assertEqual(result["path_cable_attrs"]["/World/Cable"]["material"], {})
        j0, _ = group_range(builder, "cable", "/World/Cable", "joint")
        dof0 = builder.joint_qd_start[j0]
        self.assertEqual(builder.joint_target_ke[dof0], 1.0e5)  # add_rod default stretch stiffness

    def test_material_attr_authored_on_geometry_warns(self):
        """Deformable material moduli authored on the geometry (not the material) warn and are ignored."""
        from pxr import Sdf

        stage = _deformable_stage(up_axis="y")
        pts = [(0.0, 0.0, 1.0), (0.1, 0.0, 1.0), (0.2, 0.0, 1.0), (0.3, 0.0, 1.0)]
        curves = _add_cable_curve(stage, "/World/Cable", pts)
        # stretchStiffness belongs on the bound material, not the curve geometry.
        curves.GetPrim().CreateAttribute("physics:stretchStiffness", Sdf.ValueTypeNames.Float).Set(5.0e5)

        builder = newton.ModelBuilder()
        with self.assertWarnsRegex(UserWarning, "authored on the geometry"):
            builder.add_usd(stage)

    def test_cable_resolved_density_reports_default_when_unauthored(self):
        """resolved_density reports the density actually used (the builder default), not None."""
        stage = _deformable_stage(up_axis="y")
        pts = [(0.0, 0.0, 1.0), (0.1, 0.0, 1.0), (0.2, 0.0, 1.0), (0.3, 0.0, 1.0)]
        curves = _add_cable_curve(stage, "/World/Cable", pts)
        _bind_deformable_material(stage, curves.GetPrim(), "/World/CableMat", thickness=0.02)  # no density

        builder = newton.ModelBuilder()
        result = builder.add_usd(stage, deformable_results=True)
        attrs = result["path_cable_attrs"]["/World/Cable"]
        self.assertEqual(attrs["resolved_density"], builder.default_shape_cfg.density)

    @staticmethod
    def _author_two_curve_prim_with_masses(vertex_counts, masses):
        """Author a two-curve BasisCurves prim (one 2-point curve the importer skips, one
        4-point curve) with a physics:masses array; returns the stage."""
        from pxr import Sdf, UsdGeom

        stage = _deformable_stage()
        curves = UsdGeom.BasisCurves.Define(stage, "/World/Cable")
        curves.CreateTypeAttr().Set(UsdGeom.Tokens.linear)
        short = [(0.0, 1.0, 1.0), (0.1, 1.0, 1.0)]
        long = [(0.0, 0.0, 1.0), (0.1, 0.0, 1.0), (0.2, 0.0, 1.0), (0.3, 0.0, 1.0)]
        pts = long + short if vertex_counts == [4, 2] else short + long
        curves.CreatePointsAttr(pts)
        curves.CreateCurveVertexCountsAttr(vertex_counts)
        curves.GetPrim().AddAppliedSchema("PhysicsCurvesDeformableSimAPI")
        curves.GetPrim().CreateAttribute("physics:masses", Sdf.ValueTypeNames.FloatArray).Set(masses)
        return stage

    def test_skipped_first_curve_masses_use_absolute_offsets(self):
        """physics:masses is per authored point: a full-length array is accepted and applied
        to the imported curve by absolute authored offset, so a skipped FIRST curve must not
        shift the imported curve's slice of the array."""

        # The imported curve's points are authored at offsets 2..5.
        stage = self._author_two_curve_prim_with_masses([2, 4], [9.0, 9.0, 1.0, 2.0, 2.0, 1.0])

        builder = newton.ModelBuilder()
        with self.assertWarnsRegex(UserWarning, "skipping that curve"):
            builder.add_usd(stage)
        b0, b1 = group_range(builder, "cable", "/World/Cable", "body")
        # The full-length array is applied: pm = [1, 2, 2, 1] -> segments
        # [1 + 2/2, 2/2 + 2/2, 2/2 + 1] = [2, 2, 2]; the 9.0 entries belong to the
        # skipped curve and must not leak in.
        masses = [builder.body_mass[b] for b in range(b0, b1)]
        np.testing.assert_allclose(masses, [2.0, 2.0, 2.0], atol=1e-6)

    def test_cable_masses_malformed_length_warns_and_ignored(self):
        """A physics:masses array whose length != the authored curve points warns and is
        ignored (density-derived masses are used), never an IndexError."""
        from pxr import Sdf

        with self.subTest(malformed="matches_imported_count_only"):
            # Length 4 matches the imported curve's points but not the 6 authored points, so
            # it cannot be indexed by authored offset and is rejected with a warning.
            stage = self._author_two_curve_prim_with_masses([2, 4], [1.0] * 4)

            builder = newton.ModelBuilder()
            with self.assertWarnsRegex(UserWarning, r"!= 6 authored curve points"):
                builder.add_usd(stage)
            b0, b1 = group_range(builder, "cable", "/World/Cable", "body")
            self.assertEqual(b1 - b0, 3)

        with self.subTest(malformed="simple_length_mismatch"):

            def total_cable_mass(masses=None):
                stage = _deformable_stage(up_axis="y")
                pts = [(0.0, 0.0, 1.0), (0.1, 0.0, 1.0), (0.2, 0.0, 1.0), (0.3, 0.0, 1.0)]  # 4 points
                curves = _add_cable_curve(stage, "/World/Cable", pts)
                if masses is not None:
                    curves.GetPrim().CreateAttribute("physics:masses", Sdf.ValueTypeNames.FloatArray).Set(masses)
                builder = newton.ModelBuilder()
                builder.add_usd(stage)
                b0, b1 = group_range(builder, "cable", "/World/Cable", "body")
                return sum(builder.body_mass[b] for b in range(b0, b1))

            baseline = total_cable_mass()
            with self.assertWarnsRegex(UserWarning, r"!= 4 authored curve points"):
                mismatched = total_cable_mass(masses=[1.0, 2.0, 3.0])  # length 3 != 4 points
            self.assertAlmostEqual(mismatched, baseline, places=6)

    def test_cable_per_point_masses_lump_onto_segments(self):
        """Per-point physics:masses are lumped onto the segments they border, so a front-heavy mass
        array yields a front-heavy cable (not a uniform one) while preserving the total. Each point's
        mass splits between its adjacent segments; the two endpoints border a single segment each."""
        from pxr import Sdf

        stage = _deformable_stage(up_axis="y")
        pts = [(0.0, 0.0, 1.0), (0.1, 0.0, 1.0), (0.2, 0.0, 1.0), (0.3, 0.0, 1.0)]  # 4 points -> 3 segments
        curves = _add_cable_curve(stage, "/World/Cable", pts)
        masses = [10.0, 1.0, 1.0, 1.0]  # front-heavy, length == points
        curves.GetPrim().CreateAttribute("physics:masses", Sdf.ValueTypeNames.FloatArray).Set(masses)

        builder = newton.ModelBuilder()
        builder.add_usd(stage)
        b0, b1 = group_range(builder, "cable", "/World/Cable", "body")
        seg_masses = [builder.body_mass[b] for b in range(b0, b1)]
        # Lumping (endpoints contribute their full mass to their one segment, interior points
        # split): seg0 = m0 + m1/2, seg1 = m1/2 + m2/2, seg2 = m2/2 + m3.
        self.assertEqual(len(seg_masses), 3)
        self.assertAlmostEqual(seg_masses[0], 10.0 + 0.5, places=4)
        self.assertAlmostEqual(seg_masses[1], 0.5 + 0.5, places=4)
        self.assertAlmostEqual(seg_masses[2], 0.5 + 1.0, places=4)
        # Total is preserved and the front-heavy profile survives (not flattened).
        self.assertAlmostEqual(sum(seg_masses), sum(masses), places=4)
        self.assertGreater(seg_masses[0], seg_masses[2])

    def test_cable_body_mass_rescales_total(self):
        """PhysicsDeformableBodyAPI.mass rescales the rigid cable's segment masses."""
        stage = _deformable_stage(up_axis="y")
        pts = [(0.0, 0.0, 1.0), (0.1, 0.0, 1.0), (0.2, 0.0, 1.0), (0.3, 0.0, 1.0)]
        curves = _add_cable_curve(stage, "/World/Cable", pts)
        _apply_deformable_body_api(curves.GetPrim(), mass=2.5)

        builder = newton.ModelBuilder()
        builder.add_usd(stage)
        b0, b1 = group_range(builder, "cable", "/World/Cable", "body")
        self.assertAlmostEqual(sum(builder.body_mass[b] for b in range(b0, b1)), 2.5, places=4)

    def test_cable_density_segment_mass_is_cylinder_not_capsule(self):
        """A density-derived cable segment gets the cylinder mass m = rho*pi*r^2*L per segment
        (so mass scales with density and segment length), not add_rod's capsule mass whose
        constant hemispherical caps overestimate short / thick segments."""
        r = 0.05
        for rho in (1000.0, 2000.0):
            with self.subTest(density=rho):
                stage = _deformable_stage()
                # Nonuniform segment lengths 0.1 and 0.2; short, thick segments make the
                # spherical-cap bias large (4r/3L = 0.667 -> +67% on the first segment).
                pts = [(0.0, 0.0, 1.0), (0.1, 0.0, 1.0), (0.3, 0.0, 1.0)]
                _add_cable_curve(stage, "/World/Cable", pts, thickness=2.0 * r, density=rho)

                builder = newton.ModelBuilder()
                builder.add_usd(stage)
                b0, b1 = group_range(builder, "cable", "/World/Cable", "body")
                self.assertEqual(b1 - b0, 2)
                for b, seg_len in zip(range(b0, b1), (0.1, 0.2), strict=True):
                    cylinder = rho * math.pi * r * r * seg_len
                    capsule = cylinder + rho * (4.0 / 3.0) * math.pi * r**3
                    self.assertAlmostEqual(builder.body_mass[b], cylinder, delta=cylinder * 1e-3)
                    self.assertNotAlmostEqual(builder.body_mass[b], capsule, delta=cylinder * 1e-2)
                # A 2x-longer segment has exactly 2x the mass (no constant-cap bias in the ratio).
                self.assertAlmostEqual(builder.body_mass[b0 + 1] / builder.body_mass[b0], 2.0, places=3)

    def test_cable_default_radius_scales_with_stage_units(self):
        """With no authored thickness the importer assumes a default radius derived from the stage's
        linear unit, so it is the same physical size (~0.05 m) on a centimeter stage as on a meter
        stage, and it warns that a default was assumed (rather than a meters-flavored literal)."""
        from pxr import UsdGeom

        def capsule_radius(meters_per_unit):
            stage = _deformable_stage(up_axis="y")
            UsdGeom.SetStageMetersPerUnit(stage, meters_per_unit)
            pts = [(0.0, 0.0, 1.0), (0.1, 0.0, 1.0), (0.2, 0.0, 1.0), (0.3, 0.0, 1.0)]
            _add_cable_curve(stage, "/World/Cable", pts, thickness=None)  # no bound material -> no thickness
            builder = newton.ModelBuilder()
            with self.assertWarnsRegex(UserWarning, "no cable thickness"):
                builder.add_usd(stage)
            return float(builder.shape_scale[0][0])  # capsule radius is stored as scale.x

        # ~0.05 m on a meter stage; 0.05 / 0.01 = 5 stage units on a cm stage (still ~0.05 m physical).
        self.assertAlmostEqual(capsule_radius(1.0), 0.05, places=4)
        self.assertAlmostEqual(capsule_radius(0.01), 5.0, places=3)

        stage = _deformable_stage(up_axis="y")
        pts = [(0.0, 0.0, 1.0), (0.1, 0.0, 1.0), (0.2, 0.0, 1.0), (0.3, 0.0, 1.0)]
        _add_cable_curve(stage, "/World/Cable", pts, thickness=None)
        builder = newton.ModelBuilder()
        with self.assertWarnsRegex(UserWarning, "no cable thickness"):
            builder.add_usd(stage)

    def test_duplicate_consecutive_points_skips_curve(self):
        """A curve with a zero-length segment is warned and skipped, not aborting the import."""
        from pxr import UsdGeom

        stage = _deformable_stage()
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

        builder = newton.ModelBuilder()
        with self.assertWarnsRegex(UserWarning, "duplicate consecutive points"):
            builder.add_usd(stage)
        # The valid curve still imports (4 points -> 3 bodies); the degenerate one is skipped.
        b0, b1 = group_range(builder, "cable", "/World/Cable", "body")
        self.assertEqual(b1 - b0, 3)

    def test_vendor_namespace_material_needs_resolver(self):
        """Vendor-namespaced (omniphysics:) material is read only with a compat resolver.

        The base parser targets the canonical ``physics:`` schema as written; the
        omniphysics fallback is opt-in via a schema resolver that declares it
        (mirroring how rigid-body vendor namespaces are remapped).
        """
        stage = _deformable_stage(up_axis="y")
        pts = [(0.0, 0.0, 1.0), (0.1, 0.0, 1.0), (0.2, 0.0, 1.0), (0.3, 0.0, 1.0)]
        curves = _add_cable_curve(stage, "/World/Cable", pts, thickness=None)
        _bind_deformable_material(
            stage, curves.GetPrim(), "/World/CableMat", namespace="omniphysics", thickness=0.02, density=1234.0
        )

        def cable_radius(builder):
            return builder.shape_scale[builder.body_shapes[0][0]][0]  # capsule radius

        # Default resolvers: omniphysics:thickness is ignored, so the radius is the
        # builder default, not the authored thickness / 2 (and the importer warns).
        builder_default = newton.ModelBuilder()
        with self.assertWarnsRegex(UserWarning, "no cable thickness"):
            builder_default.add_usd(stage)
        default_radius = cable_radius(builder_default)

        # With the PhysX resolver active, omniphysics:thickness is honored (radius = thickness / 2).
        builder_compat = newton.ModelBuilder()
        builder_compat.add_usd(stage, schema_resolvers=[SchemaResolverPhysx()])
        self.assertAlmostEqual(cable_radius(builder_compat), 0.5 * 0.02, places=5)
        self.assertNotAlmostEqual(default_radius, 0.5 * 0.02, places=5)

    def test_deformable_ignores_generic_physx_namespaces(self):
        """Deformable material reads only deformable vendor namespaces, not generic PhysX ones."""

        def cable_radius(namespace):
            stage = _deformable_stage(up_axis="y")
            pts = [(0.0, 0.0, 1.0), (0.1, 0.0, 1.0), (0.2, 0.0, 1.0), (0.3, 0.0, 1.0)]
            curves = _add_cable_curve(stage, "/World/Cable", pts, thickness=None)
            _bind_deformable_material(stage, curves.GetPrim(), "/World/Mat", namespace=namespace, thickness=0.02)
            builder = newton.ModelBuilder()
            builder.add_usd(stage, schema_resolvers=[SchemaResolverPhysx()])
            return builder.shape_scale[builder.body_shapes[0][0]][0]

        # omniphysics is a deformable vendor namespace -> thickness honored (no fallback warning).
        self.assertAlmostEqual(cable_radius("omniphysics"), 0.5 * 0.02, places=5)
        # physxScene is a generic resolver namespace -> NOT read as deformable material, so the
        # cable falls back to the default radius and warns.
        with self.assertWarnsRegex(UserWarning, "no cable thickness"):
            physx_radius = cable_radius("physxScene")
        self.assertNotAlmostEqual(physx_radius, 0.5 * 0.02, places=5)

    def test_cable_normals_orient_segments(self):
        """Authored normals set each segment's cross-section frame: +Z -> tangent, +Y -> normal."""
        from pxr import UsdGeom

        stage = _deformable_stage()
        pts = [(0.0, 0.0, 1.0), (0.1, 0.0, 1.0), (0.2, 0.0, 1.0), (0.3, 0.0, 1.0)]  # tangent +X
        curves = _add_cable_curve(stage, "/World/Cable", pts)
        normals = curves.GetNormalsAttr()
        if not normals:
            normals = curves.CreateNormalsAttr()
        normals.Set([(0.0, 1.0, 0.0)] * len(pts))  # cross-section frame: +Y
        curves.SetNormalsInterpolation(UsdGeom.Tokens.vertex)

        builder = newton.ModelBuilder()
        builder.add_usd(stage)
        b0, b1 = group_range(builder, "cable", "/World/Cable", "body")

        for body in range(b0, b1):
            t = builder.body_q[body]
            q = wp.quat(float(t[3]), float(t[4]), float(t[5]), float(t[6]))
            z_world = np.array(wp.quat_rotate(q, wp.vec3(0.0, 0.0, 1.0)), dtype=np.float32)
            y_world = np.array(wp.quat_rotate(q, wp.vec3(0.0, 1.0, 0.0)), dtype=np.float32)
            np.testing.assert_allclose(z_world, [1.0, 0.0, 0.0], atol=1e-5)  # +Z -> tangent +X
            np.testing.assert_allclose(y_world, [0.0, 1.0, 0.0], atol=1e-5)  # +Y -> normal

    def test_cable_normals_source_selection(self):
        """Indexed primvars:normals take precedence over the schema normals attribute, and
        normals with non-per-point interpolation are warned and ignored, not misapplied."""
        from pxr import Sdf, UsdGeom, Vt

        pts = [(0.0, 0.0, 1.0), (0.1, 0.0, 1.0), (0.2, 0.0, 1.0), (0.3, 0.0, 1.0)]  # tangent +X

        with self.subTest(normals="primvar_precedence"):
            stage = _deformable_stage()
            curves = _add_cable_curve(stage, "/World/Cable", pts)
            # Schema normals say +Z; the indexed primvars:normals (+Y) must win.
            curves.CreateNormalsAttr([(0.0, 0.0, 1.0)] * len(pts))
            curves.SetNormalsInterpolation(UsdGeom.Tokens.vertex)
            pv = UsdGeom.PrimvarsAPI(curves.GetPrim()).CreatePrimvar(
                "normals", Sdf.ValueTypeNames.Normal3fArray, UsdGeom.Tokens.vertex
            )
            pv.Set([(0.0, 1.0, 0.0)])  # one unique value...
            pv.SetIndices(Vt.IntArray([0, 0, 0, 0]))  # ...indexed to all 4 points

            builder = newton.ModelBuilder()
            builder.add_usd(stage)
            b0, b1 = group_range(builder, "cable", "/World/Cable", "body")
            for body in range(b0, b1):
                t = builder.body_q[body]
                q = wp.quat(float(t[3]), float(t[4]), float(t[5]), float(t[6]))
                y_world = np.array(wp.quat_rotate(q, wp.vec3(0.0, 1.0, 0.0)), dtype=np.float32)
                # +Y comes from primvars:normals; if the schema +Z had won it would be ~[0,0,1].
                np.testing.assert_allclose(y_world, [0.0, 1.0, 0.0], atol=1e-5)

        with self.subTest(normals="non_per_point_ignored"):
            stage = _deformable_stage()
            curves = _add_cable_curve(stage, "/World/Cable", pts)
            curves.CreateNormalsAttr([(0.0, 1.0, 0.0)])
            curves.SetNormalsInterpolation(UsdGeom.Tokens.constant)  # not per-point

            builder = newton.ModelBuilder()
            with self.assertWarnsRegex(UserWarning, "not per-point"):
                builder.add_usd(stage)
            # The cable still imports (normals ignored, default segment orientation used).
            b0, b1 = group_range(builder, "cable", "/World/Cable", "body")
            self.assertEqual(b1 - b0, 3)

    def test_cable_full_affine_xform_is_exact(self):
        """Cable import honors the full affine world transform. Under a reflected + sheared
        xform: body positions mirror (reflection parity preserved), authored normals orient
        by the inverse-transpose of the linear block, and rest segment lengths (for E*A/L)
        measure the full linear map. A rotation + per-axis-scale decomposition would drop
        both the reflection and the shear."""
        from pxr import Gf, Sdf, UsdGeom

        stage = _deformable_stage()
        seg_len, thickness, E, k = 0.1, 0.02, 2.0e6, 0.75  # shear maps a +X segment to length L*sqrt(1+k^2)
        # Curve along +X at y=0.5 so the Y reflection visibly mirrors positions.
        pts = [(i * seg_len, 0.5, 1.0) for i in range(4)]
        curves = _add_cable_curve(stage, "/World/Cable", pts, thickness=None)
        curves.CreateNormalsAttr([(0.0, 1.0, 0.0)] * len(pts))  # local cross-section normal +Y
        curves.SetNormalsInterpolation(UsdGeom.Tokens.vertex)
        _bind_deformable_material(stage, curves.GetPrim(), "/World/Mat", thickness=thickness, stretchStiffness=E)
        # Rest centerline == authored points, so rest length equals the transformed segment length.
        curves.GetPrim().CreateAttribute("physics:restShapePoints", Sdf.ValueTypeNames.Point3fArray).Set(
            [tuple(p) for p in pts]
        )
        # Row-major Gf matrix: reflect across Y (y -> -y) and shear world z += k * x --
        # neither is expressible as a rotation + per-axis scale.
        m = Gf.Matrix4d(1.0, 0.0, k, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0)
        UsdGeom.Xformable(curves).AddTransformOp().Set(m)

        builder = newton.ModelBuilder()
        with self.assertWarnsRegex(UserWarning, "restShapePoints only sets the rest length"):
            builder.add_usd(stage)
        b0, b1 = group_range(builder, "cable", "/World/Cable", "body")
        j0, _ = group_range(builder, "cable", "/World/Cable", "joint")
        self.assertEqual(b1 - b0, 3)

        tangent = np.array([1.0, 0.0, k]) / math.sqrt(1.0 + k * k)
        for i, body in enumerate(range(b0, b1)):
            t = builder.body_q[body]
            # Positions: segment midpoints under the full map, with y mirrored to -0.5.
            mid_x = (i + 0.5) * seg_len
            expected_origin = [mid_x, -0.5, 1.0 + k * mid_x]
            np.testing.assert_allclose(np.array(t[:3], dtype=np.float32), expected_origin, atol=1e-5)
            q = wp.quat(float(t[3]), float(t[4]), float(t[5]), float(t[6]))
            # Normals: the inverse-transpose of the linear block maps +Y -> -Y (reflection
            # parity); a rot/scale decomposition would drop the reflection and keep +Y.
            y_world = np.array(wp.quat_rotate(q, wp.vec3(0.0, 1.0, 0.0)), dtype=np.float32)
            np.testing.assert_allclose(y_world, [0.0, -1.0, 0.0], atol=1e-5)
            # The segment frame's +Z tracks the sheared tangent.
            z_world = np.array(wp.quat_rotate(q, wp.vec3(0.0, 0.0, 1.0)), dtype=np.float32)
            np.testing.assert_allclose(z_world, tangent, atol=1e-5)

        # Rest lengths: the shear stretches each segment to L * sqrt(1 + k^2), lowering
        # stretch stiffness E*A/L accordingly; a decomposed scale cannot represent this.
        r = 0.5 * thickness
        rest_len = seg_len * math.sqrt(1.0 + k * k)
        expected_ke = E * math.pi * r * r / rest_len
        dof0 = builder.joint_qd_start[j0]
        self.assertAlmostEqual(builder.joint_target_ke[dof0], expected_ke, delta=expected_ke * 1e-3)

    def test_instanced_cable_imports_proxies_not_prototype(self):
        """Instanced cables import once per instance proxy; the prototype master is skipped."""
        from pxr import UsdGeom

        stage = _deformable_stage()
        pts = [(0.0, 0.0, 1.0), (0.1, 0.0, 1.0), (0.2, 0.0, 1.0), (0.3, 0.0, 1.0)]
        stage.CreateClassPrim("/Proto")  # template in a class, not the rendered scene
        _add_cable_curve(stage, "/Proto/Cable", pts)
        for name in ("A", "B"):
            inst = UsdGeom.Xform.Define(stage, f"/World/{name}")
            inst.GetPrim().GetReferences().AddInternalReference("/Proto")
            inst.GetPrim().SetInstanceable(True)

        builder = newton.ModelBuilder()
        builder.add_usd(stage)
        # Two instance proxies import; the prototype master (/__Prototype_*) is not.
        self.assertEqual(set(group_labels(builder, "cable")), {"/World/A/Cable", "/World/B/Cable"})

    def test_periodic_cable_imports_closing_segment(self):
        """A periodic curve builds a body for the closing v[-1] -> v[0] segment."""
        stage = _deformable_stage(up_axis="y")
        # 4 vertices -> 4 segments for a closed loop (incl. the wrap segment).
        pts = [(0.0, 0.0, 1.0), (1.0, 0.0, 1.0), (1.0, 1.0, 1.0), (0.0, 1.0, 1.0)]
        _add_cable_curve(stage, "/World/Cable", pts, periodic=True)

        builder = newton.ModelBuilder()
        builder.add_usd(stage)
        b0, b1 = group_range(builder, "cable", "/World/Cable", "body")
        j0, j1 = group_range(builder, "cable", "/World/Cable", "joint")
        self.assertEqual(b1 - b0, 4, "expected one body per segment, incl. the closing segment")
        self.assertEqual(j1 - j0, 4, "expected 3 chain joints + 1 loop joint")
        # The importer wraps the closed cable; add_rod keeps the loop-closing joint out of the tree.
        self.assertIn("/World/Cable_articulation", builder.articulation_label)

    def test_welded_graph_degenerate_segment_skips_component(self):
        """A welded curve with a zero-length segment is rejected with a warning instead of aborting
        the whole import; the component's curves fall back to the per-curve pass."""
        stage = _deformable_stage()
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

        builder = newton.ModelBuilder()
        with self.assertWarnsRegex(UserWarning, "zero-length segment"):
            result = builder.add_usd(stage, deformable_results=True)
        # The import did not abort; the valid trunk imported as a single (unwrapped) cable.
        self.assertIn("/World/Trunk", group_labels(builder, "cable"))
        j0, j1 = group_range(builder, "cable", "/World/Trunk", "joint")
        self.assertNotEqual(j1 - j0, 0, "the skipped component leaves the trunk as a single cable")
        self.assertNotIn("graph_component", result["path_cable_attrs"]["/World/Trunk"])
        # The junction must not be silently consumed by the failed weld: it reaches the
        # attachment pass, which preserves the authored constraint as unsupported.
        self.assertNotIn("/World/Junction", result["path_attachment_map"])
        junction = result["path_attachment_attrs"]["/World/Junction"]
        self.assertIn("unsupported_reason", junction)

    def test_welded_graph_collapse_with_masses_falls_back(self):
        """Welding two adjacent points of one curve onto the same node collapses a segment. With
        authored physics:masses the surviving body count no longer matches the per-point lumping; the
        importer warns and falls back instead of raising and aborting."""
        from pxr import Sdf

        stage = _deformable_stage()
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

        builder = newton.ModelBuilder()
        with self.assertWarnsRegex(UserWarning, "collapsed a segment"):
            result = builder.add_usd(stage, deformable_results=True)
        # The welded graph still built (no abort); both curves are present in the graph component.
        self.assertIn("graph_component", result["path_cable_attrs"]["/World/Trunk"])
        self.assertIn("graph_component", result["path_cable_attrs"]["/World/Branch"])
        self.assertEqual(builder.finalize().body_count, builder.body_count)

    def test_welded_graph_drops_rest_shape_warns(self):
        """A welded curve's authored restShapePoints cannot be honored by add_rod_graph's scalar
        stiffness, so the importer warns rather than silently using the current segment lengths."""
        from pxr import Sdf

        stage = _deformable_stage()
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

        builder = newton.ModelBuilder()
        with self.assertWarnsRegex(UserWarning, "restShapePoints is dropped"):
            result = builder.add_usd(stage, deformable_results=True)
        self.assertIn("graph_component", result["path_cable_attrs"]["/World/Branch"])

    def test_ignored_curve_to_curve_junction_does_not_weld(self):
        """An ``ignore_paths`` junction must not alter topology: the curves stay independent.

        Without honoring ``ignore_paths`` in the graph pre-pass, an ignored junction would
        still weld its curves into a pre-wrapped rod graph (and silently vanish from the
        attachment maps).
        """
        stage = _deformable_stage()
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

        builder = newton.ModelBuilder()
        result = builder.add_usd(stage, ignore_paths=["/World/Junction"], deformable_results=True)

        # Both curves still import, but as independent single cables (not a welded graph):
        # single cables expose their cable joints for the caller to wrap, so joints are non-empty.
        tb0, tb1 = group_range(builder, "cable", "/World/Trunk", "body")
        tj0, tj1 = group_range(builder, "cable", "/World/Trunk", "joint")
        bj0, bj1 = group_range(builder, "cable", "/World/Branch", "joint")
        self.assertEqual(tb1 - tb0, 3)
        self.assertNotEqual(tj1 - tj0, 0, "an ignored junction must leave the cable unwelded")
        self.assertNotEqual(bj1 - bj0, 0, "an ignored junction must leave the cable unwelded")
        self.assertNotIn("graph_component", result["path_cable_attrs"]["/World/Trunk"])
        # The ignored junction is consumed by nothing: it is absent from the attachment maps.
        self.assertNotIn("/World/Junction", result["path_attachment_map"])
        self.assertNotIn("/World/Junction", result["path_attachment_attrs"])

    def test_malformed_junction_warns_and_skips_weld(self):
        """A malformed curve-to-curve junction (out-of-range index, or index arrays that differ
        in length) warns and skips the weld -- both curves import independently -- instead of
        raising an IndexError or silently reusing indices1[0]."""
        trunk_pts = [(0.0, 0.0, 1.0), (0.1, 0.0, 1.0), (0.2, 0.0, 1.0), (0.3, 0.0, 1.0)]
        branch_pts = [(0.1, 0.0, 1.0), (0.1, 0.1, 1.0), (0.1, 0.2, 1.0)]
        cases = (
            # 99 is out of range for the 3-point branch.
            ("out_of_range_index", [99], [1], "out of range"),
            # Two sites on the branch but only one partner index on the trunk.
            ("mismatched_index_lengths", [0, 1], [1], "differ in length"),
        )
        for name, indices0, indices1, regex in cases:
            with self.subTest(junction=name):
                stage = _deformable_stage()
                _add_cable_curve(stage, "/World/Trunk", trunk_pts)
                _add_cable_curve(stage, "/World/Branch", branch_pts)
                _add_physics_attachment(
                    stage,
                    "/World/Junction",
                    src0="/World/Branch",
                    src1="/World/Trunk",
                    type0="point",
                    type1="point",
                    indices0=indices0,
                    indices1=indices1,
                )

                builder = newton.ModelBuilder()
                with self.assertWarnsRegex(UserWarning, regex):
                    result = builder.add_usd(stage, deformable_results=True)

                # The malformed junction does not weld: both curves import independently.
                tj0, tj1 = group_range(builder, "cable", "/World/Trunk", "joint")
                self.assertIn("/World/Branch", group_labels(builder, "cable"))
                self.assertNotEqual(tj1 - tj0, 0)
                self.assertNotIn("graph_component", result["path_cable_attrs"]["/World/Trunk"])

    def test_heterogeneous_welded_cable_materials_warn(self):
        """Welding curves with different materials warns that one representative is used.

        add_rod_graph applies one scalar radius/density/stiffness per component, so the graph
        flattens to the first curve's material; the disagreement must be surfaced, not silent.
        """
        stage = _deformable_stage()
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

        builder = newton.ModelBuilder()
        with self.assertWarnsRegex(UserWarning, "differing radius/density/stiffness"):
            result = builder.add_usd(stage, deformable_results=True)

        # Still welded into one graph; each curve keeps its own authored material in the attrs.
        self.assertIn("graph_component", result["path_cable_attrs"]["/World/Trunk"])
        self.assertIn("graph_component", result["path_cable_attrs"]["/World/Branch"])
        self.assertAlmostEqual(result["path_cable_attrs"]["/World/Trunk"]["material"]["thickness"], 0.02, places=5)
        self.assertAlmostEqual(result["path_cable_attrs"]["/World/Branch"]["material"]["thickness"], 0.06, places=5)


if __name__ == "__main__":
    unittest.main(verbosity=2)
