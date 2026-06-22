# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
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

"""Tests for USD deformable (cable / curve) parsing."""

import math
import tempfile
import unittest
from pathlib import Path

import numpy as np
import warp as wp

import newton
from newton.tests.unittest_utils import add_function_test, get_selected_cuda_test_devices
from newton.usd import SchemaResolverPhysx


def _add_cable_curve(stage, path, points, *, periodic=False):
    """Author a GeomBasisCurves marked as a curve deformable (cable)."""
    from pxr import UsdGeom

    curves = UsdGeom.BasisCurves.Define(stage, path)
    curves.CreateTypeAttr().Set(UsdGeom.Tokens.linear)
    curves.CreatePointsAttr([tuple(p) for p in points])
    curves.CreateCurveVertexCountsAttr([len(points)])
    curves.CreateWrapAttr().Set(UsdGeom.Tokens.periodic if periodic else UsdGeom.Tokens.nonperiodic)
    # Metadata-based discovery: apply the curve-deformable sim API by token so it
    # is found even when the deformable schema is not registered with USD.
    curves.GetPrim().AddAppliedSchema("PhysicsCurvesDeformableSimAPI")
    return curves


def _bind_cable_material(stage, prim, mat_path, *, namespace="physics", **attrs):
    """Author a deformable material and bind it to a prim.

    Authors under the canonical ``physics:`` namespace by default; pass
    ``namespace`` to author under a vendor namespace (e.g. ``omniphysics``) to
    exercise the schema-resolver compatibility path.
    """
    from pxr import Sdf, UsdGeom, UsdShade

    mat = UsdShade.Material.Define(stage, mat_path)
    # Declare the per-family deformable material API the importer's readers gate on.
    if prim.IsA(UsdGeom.BasisCurves):
        mat.GetPrim().AddAppliedSchema("PhysicsCurvesDeformableMaterialAPI")
    elif prim.IsA(UsdGeom.TetMesh):
        mat.GetPrim().AddAppliedSchema("PhysicsVolumeDeformableMaterialAPI")
    elif prim.IsA(UsdGeom.Mesh):
        mat.GetPrim().AddAppliedSchema("PhysicsSurfaceDeformableMaterialAPI")
    for name, value in attrs.items():
        mat.GetPrim().CreateAttribute(f"{namespace}:{name}", Sdf.ValueTypeNames.Float).Set(value)
    binding = UsdShade.MaterialBindingAPI.Apply(prim)
    binding.Bind(mat, materialPurpose="physics")
    return mat


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
            result = builder.add_usd(str(usd_path))

            self.assertIn("/World/Cable", result["path_cable_map"])
            bodies, joints = result["path_cable_map"]["/World/Cable"]
            self.assertEqual(len(bodies), 3, "expected one capsule body per segment")
            self.assertEqual(len(joints), 2, "expected num_segments - 1 cable joints for an open chain")

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
            result = builder.add_usd(str(usd_path))

            self.assertEqual(result["path_cable_map"], {})
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
            _bind_cable_material(
                stage,
                curves.GetPrim(),
                "/World/CableMat",
                thickness=thickness,
                stretchStiffness=stretch_mod,
                bendStiffness=bend_mod,
            )
            stage.Save()

            builder = newton.ModelBuilder()
            result = builder.add_usd(str(usd_path))
            bodies, joints = result["path_cable_map"]["/World/Cable"]
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

    def test_cable_zero_stiffness_is_preserved(self):
        """Authored zero stiffness (range [0, inf)) is kept, not replaced by add_rod's default."""
        from pxr import Usd, UsdPhysics

        with tempfile.TemporaryDirectory() as tmpdir:
            usd_path = Path(tmpdir) / "cable_zero.usda"
            stage = Usd.Stage.CreateNew(str(usd_path))
            UsdPhysics.Scene.Define(stage, "/PhysicsScene")
            pts = [(0.0, 0.0, 1.0), (0.1, 0.0, 1.0), (0.2, 0.0, 1.0), (0.3, 0.0, 1.0)]
            curves = _add_cable_curve(stage, "/World/Cable", pts)
            _bind_cable_material(
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
            _bodies, joints = result["path_cable_map"]["/World/Cable"]
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
                result = builder.add_usd(str(usd_path))
            self.assertEqual(result["path_cable_map"], {})
            self.assertEqual(builder.body_count, 0)

    def test_cable_material_without_family_api_is_ignored(self):
        """A physics-bound material lacking PhysicsCurvesDeformableMaterialAPI is not read as a cable material."""
        from pxr import Sdf, Usd, UsdPhysics, UsdShade

        with tempfile.TemporaryDirectory() as tmpdir:
            usd_path = Path(tmpdir) / "cable_no_api.usda"
            stage = Usd.Stage.CreateNew(str(usd_path))
            UsdPhysics.Scene.Define(stage, "/PhysicsScene")
            pts = [(0.0, 0.0, 1.0), (0.1, 0.0, 1.0), (0.2, 0.0, 1.0), (0.3, 0.0, 1.0)]
            curves = _add_cable_curve(stage, "/World/Cable", pts)
            # Material carries cable-shaped attributes but does NOT declare the family API.
            mat = UsdShade.Material.Define(stage, "/World/Mat")
            mat.GetPrim().CreateAttribute("physics:stretchStiffness", Sdf.ValueTypeNames.Float).Set(2.0e6)
            mat.GetPrim().CreateAttribute("physics:thickness", Sdf.ValueTypeNames.Float).Set(0.02)
            UsdShade.MaterialBindingAPI.Apply(curves.GetPrim()).Bind(mat, materialPurpose="physics")
            stage.Save()

            builder = newton.ModelBuilder()
            result = builder.add_usd(str(usd_path))
            # Without the family API the material is ignored: no attrs, default rod stiffness.
            self.assertEqual(result["path_cable_attrs"]["/World/Cable"]["material"], {})
            _bodies, joints = result["path_cable_map"]["/World/Cable"]
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
            _bind_cable_material(stage, curves.GetPrim(), "/World/CableMat", thickness=0.02)  # no density
            stage.Save()

            builder = newton.ModelBuilder()
            result = builder.add_usd(str(usd_path))
            attrs = result["path_cable_attrs"]["/World/Cable"]
            self.assertEqual(attrs["resolved_density"], builder.default_shape_cfg.density)

    def test_skipped_curve_excluded_from_cable_mass_count(self):
        """Points from a skipped (too-short) curve are excluded from the per-point mass-count check."""
        from pxr import Sdf, Usd, UsdGeom, UsdPhysics

        with tempfile.TemporaryDirectory() as tmpdir:
            usd_path = Path(tmpdir) / "multi_curve.usda"
            stage = Usd.Stage.CreateNew(str(usd_path))
            UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
            UsdPhysics.Scene.Define(stage, "/PhysicsScene")
            curves = UsdGeom.BasisCurves.Define(stage, "/World/Cable")
            curves.CreateTypeAttr().Set(UsdGeom.Tokens.linear)
            # A valid 4-point curve plus a 2-point curve that the importer skips.
            curves.CreatePointsAttr(
                [(0.0, 0.0, 1.0), (0.1, 0.0, 1.0), (0.2, 0.0, 1.0), (0.3, 0.0, 1.0), (0.0, 1.0, 1.0), (0.1, 1.0, 1.0)]
            )
            curves.CreateCurveVertexCountsAttr([4, 2])
            curves.GetPrim().AddAppliedSchema("PhysicsCurvesDeformableSimAPI")
            # Masses authored for all 6 points; only 4 are imported, so the count check
            # validates against 4 (not 6) and rejects the array instead of applying
            # mass from the skipped curve's points.
            curves.GetPrim().CreateAttribute("physics:masses", Sdf.ValueTypeNames.FloatArray).Set([1.0] * 6)
            stage.Save()

            builder = newton.ModelBuilder()
            with self.assertWarnsRegex(UserWarning, r"!= 4 curve points"):
                builder.add_usd(str(usd_path))

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
                result = builder.add_usd(str(usd_path))
            # The valid curve still imports (4 points -> 3 bodies); the degenerate one is skipped.
            bodies, _ = result["path_cable_map"]["/World/Cable"]
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
            curves = _add_cable_curve(stage, "/World/Cable", pts)
            _bind_cable_material(
                stage, curves.GetPrim(), "/World/CableMat", namespace="omniphysics", thickness=0.02, density=1234.0
            )
            stage.Save()

            def cable_radius(builder):
                return builder.shape_scale[builder.body_shapes[0][0]][0]  # capsule radius

            # Default resolvers: omniphysics:thickness is ignored, so the radius is the
            # builder default, not the authored thickness / 2.
            builder_default = newton.ModelBuilder()
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
                curves = _add_cable_curve(stage, "/World/Cable", pts)
                _bind_cable_material(stage, curves.GetPrim(), "/World/Mat", namespace=namespace, thickness=0.02)
                stage.Save()
                builder = newton.ModelBuilder()
                builder.add_usd(str(usd_path), schema_resolvers=[SchemaResolverPhysx()])
                return builder.shape_scale[builder.body_shapes[0][0]][0]

        # omniphysics is a deformable vendor namespace -> thickness honored.
        self.assertAlmostEqual(cable_radius("omniphysics"), 0.5 * 0.02, places=5)
        # physxScene is a generic resolver namespace -> NOT read as deformable material.
        self.assertNotAlmostEqual(cable_radius("physxScene"), 0.5 * 0.02, places=5)

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
            result = builder.add_usd(str(usd_path))
            cmap = result["path_cable_map"]

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
            result = builder.add_usd(str(usd_path))
            bodies, _ = result["path_cable_map"]["/World/Cable"]

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
            curve = _add_cable_curve(stage, "/World/Cable", pts)
            _bind_cable_material(
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
        """Material density maps to capsule mass: doubling density doubles segment mass."""
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
            _bind_cable_material(stage, ca.GetPrim(), "/World/MatA", thickness=0.02, density=1000.0)
            _bind_cable_material(stage, cb.GetPrim(), "/World/MatB", thickness=0.02, density=2000.0)
            stage.Save()

            builder = newton.ModelBuilder()
            result = builder.add_usd(str(usd_path))
            bodies_a, _ = result["path_cable_map"]["/World/CableA"]
            bodies_b, _ = result["path_cable_map"]["/World/CableB"]

            mass_a = builder.body_mass[bodies_a[0]]
            mass_b = builder.body_mass[bodies_b[0]]
            self.assertGreater(mass_a, 0.0)
            self.assertAlmostEqual(mass_b, 2.0 * mass_a, delta=mass_a * 1e-3)

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
            result = builder.add_usd(str(usd_path))
            bodies, _ = result["path_cable_map"]["/World/Cable"]

            for body in bodies:
                t = builder.body_q[body]
                q = wp.quat(float(t[3]), float(t[4]), float(t[5]), float(t[6]))
                z_world = np.array(wp.quat_rotate(q, wp.vec3(0.0, 0.0, 1.0)), dtype=np.float32)
                y_world = np.array(wp.quat_rotate(q, wp.vec3(0.0, 1.0, 0.0)), dtype=np.float32)
                np.testing.assert_allclose(z_world, [1.0, 0.0, 0.0], atol=1e-5)  # +Z -> tangent +X
                np.testing.assert_allclose(y_world, [0.0, 1.0, 0.0], atol=1e-5)  # +Y -> normal

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
                result = builder.add_usd(str(usd_path))
            # The cable still imports (normals ignored, default segment orientation used).
            bodies, _ = result["path_cable_map"]["/World/Cable"]
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
            result = builder.add_usd(str(usd_path))
            bodies, _ = result["path_cable_map"]["/World/Cable"]
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
            result = builder.add_usd(str(usd_path))
            # Two instance proxies import; the prototype master (/__Prototype_*) is not.
            self.assertEqual(set(result["path_cable_map"]), {"/World/A/Cable", "/World/B/Cable"})

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
            result = proto.add_usd(str(usd_path))
            base_bodies, base_joints = result["path_cable_map"]["/World/Cable"]
            self.assertEqual(base_bodies, list(range(proto.body_count)))  # cable is the whole prototype
            # Imported cables are unwrapped; the caller wraps them into an articulation,
            # which replicate() then copies per world.
            proto.add_articulation(base_joints, label="/World/Cable_articulation")

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

    def test_imported_cable_is_unwrapped_until_caller_wraps(self):
        """add_usd imports cable joints unwrapped; wrapping into an articulation is the caller's job."""
        from pxr import Usd, UsdPhysics

        with tempfile.TemporaryDirectory() as tmpdir:
            usd_path = Path(tmpdir) / "cable_art.usda"
            stage = Usd.Stage.CreateNew(str(usd_path))
            UsdPhysics.Scene.Define(stage, "/PhysicsScene")
            pts = [(0.0, 0.0, 1.0), (0.1, 0.0, 1.0), (0.2, 0.0, 1.0), (0.3, 0.0, 1.0)]
            _add_cable_curve(stage, "/World/Cable", pts)
            stage.Save()

            builder = newton.ModelBuilder()
            _bodies, joints = builder.add_usd(str(usd_path))["path_cable_map"]["/World/Cable"]
            # The importer does not impose an articulation; the cable joints are unwrapped.
            self.assertEqual(len(builder.articulation_label), 0)

            # The caller wraps the returned joints before finalize() (an unwrapped cable
            # would otherwise fail finalize with orphan joints).
            builder.add_articulation(joints, label="/World/Cable_articulation")
            self.assertIn("/World/Cable_articulation", builder.articulation_label)

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
            result = builder.add_usd(str(usd_path))
            bodies, joints = result["path_cable_map"]["/World/Cable"]
            self.assertEqual(len(bodies), 4, "expected one body per segment, incl. the closing segment")
            self.assertEqual(len(joints), 4, "expected 3 chain joints + 1 loop joint")

    def test_path_cable_map_remapped_after_collapse(self):
        """path_cable_map indices still point at cable bodies after fixed-joint collapse."""
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
            result = builder.add_usd(str(usd_path), collapse_fixed_joints=True)
            bodies, _ = result["path_cable_map"]["/World/Cable"]
            self.assertTrue(all(0 <= b < builder.body_count for b in bodies), "cable body index out of range")
            self.assertTrue(
                all("/World/Cable" in builder.body_label[b] for b in bodies),
                "remapped cable indices point at non-cable bodies",
            )


# Authored .usda cable asset (T7): loading it should replace the programmatic
# cable construction used by the cable examples.
_CABLE_ASSET = Path(__file__).parent / "assets" / "cable_curve_deformable.usda"


class TestUSDDeformableCableAsset(unittest.TestCase):
    """Round-trip a committed .usda cable asset through the importer and the VBD solver (T7)."""

    def test_asset_parses_to_cable(self):
        """The committed .usda cable asset imports into rod bodies + cable joints (device-free)."""
        builder = newton.ModelBuilder()
        result = builder.add_usd(str(_CABLE_ASSET))
        bodies, joints = result["path_cable_map"]["/World/Cable"]
        # 6 points -> 5 segments -> 5 bodies, 4 cable joints (open chain).
        self.assertEqual(len(bodies), 5)
        self.assertEqual(len(joints), 4)

    def test_asset_cable_simulates(self, device=None):
        """After parsing, the asset cable runs through SolverVBD and stays finite (mirrors example test_final)."""
        if device is None or not wp.get_device(device).is_cuda:
            self.skipTest("VBD cable simulation requires a CUDA device")

        with wp.ScopedDevice(device):
            builder = newton.ModelBuilder()
            _bodies, joints = builder.add_usd(str(_CABLE_ASSET))["path_cable_map"]["/World/Cable"]
            builder.add_articulation(joints)  # imported cables are unwrapped; the caller wraps them.
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


def _add_cloth_mesh(stage, path):
    """Author a two-triangle quad GeomMesh marked as a surface deformable (cloth)."""
    from pxr import UsdGeom

    mesh = UsdGeom.Mesh.Define(stage, path)
    mesh.CreatePointsAttr([(0.0, 0.0, 1.0), (1.0, 0.0, 1.0), (1.0, 1.0, 1.0), (0.0, 1.0, 1.0)])
    mesh.CreateFaceVertexCountsAttr([3, 3])
    mesh.CreateFaceVertexIndicesAttr([0, 1, 2, 0, 2, 3])
    mesh.GetPrim().AddAppliedSchema("PhysicsSurfaceDeformableSimAPI")
    return mesh


class TestUSDDeformableCloth(unittest.TestCase):
    """Surface-deformable (cloth) parsing into particles + FEM triangles + bending edges."""

    def test_triangle_mesh_imports_as_cloth(self):
        """A triangle Mesh with PhysicsSurfaceDeformableSimAPI imports as cloth with per-cloth ranges."""
        from pxr import Usd, UsdPhysics

        with tempfile.TemporaryDirectory() as tmpdir:
            usd_path = Path(tmpdir) / "cloth.usda"
            stage = Usd.Stage.CreateNew(str(usd_path))
            UsdPhysics.Scene.Define(stage, "/PhysicsScene")
            _add_cloth_mesh(stage, "/World/Cloth")
            stage.Save()

            builder = newton.ModelBuilder()
            result = builder.add_usd(str(usd_path))

            ranges = result["path_cloth_map"]["/World/Cloth"]
            self.assertEqual(ranges["particle"], (0, 4))  # 4 quad vertices
            self.assertEqual(ranges["tri"], (0, 2))  # 2 triangles
            # Bending edges cover the cloth's full edge range starting at 0.
            self.assertEqual(ranges["edge"][0], 0)
            self.assertEqual(ranges["edge"][1], builder.edge_count)
            self.assertGreater(builder.edge_count, 0)
            self.assertEqual(builder.particle_count, 4)

    def test_cloth_quad_mesh_is_triangulated(self):
        """A quad-faced cloth mesh is fan-triangulated on import (n-gons are supported)."""
        from pxr import Usd, UsdGeom, UsdPhysics

        with tempfile.TemporaryDirectory() as tmpdir:
            usd_path = Path(tmpdir) / "quad_cloth.usda"
            stage = Usd.Stage.CreateNew(str(usd_path))
            UsdPhysics.Scene.Define(stage, "/PhysicsScene")
            mesh = UsdGeom.Mesh.Define(stage, "/World/Cloth")
            mesh.CreatePointsAttr([(0.0, 0.0, 1.0), (1.0, 0.0, 1.0), (1.0, 1.0, 1.0), (0.0, 1.0, 1.0)])
            mesh.CreateFaceVertexCountsAttr([4])  # single quad face
            mesh.CreateFaceVertexIndicesAttr([0, 1, 2, 3])
            mesh.GetPrim().AddAppliedSchema("PhysicsSurfaceDeformableSimAPI")
            stage.Save()

            builder = newton.ModelBuilder()
            result = builder.add_usd(str(usd_path))
            ranges = result["path_cloth_map"]["/World/Cloth"]
            self.assertEqual(ranges["particle"], (0, 4))  # 4 quad vertices stay 1:1 with particles
            self.assertEqual(ranges["tri"], (0, 2))  # quad fan-triangulates to 2 triangles
            self.assertEqual(builder.particle_count, 4)

    def test_cloth_left_handed_orientation_flips_winding(self):
        """A left-handed cloth mesh flips triangle winding, matching the rigid mesh path."""
        from pxr import Usd, UsdGeom, UsdPhysics

        with tempfile.TemporaryDirectory() as tmpdir:
            usd_path = Path(tmpdir) / "cloth_lh.usda"
            stage = Usd.Stage.CreateNew(str(usd_path))
            UsdPhysics.Scene.Define(stage, "/PhysicsScene")
            mesh = UsdGeom.Mesh.Define(stage, "/World/Cloth")
            mesh.CreatePointsAttr([(0.0, 0.0, 1.0), (1.0, 0.0, 1.0), (1.0, 1.0, 1.0), (0.0, 1.0, 1.0)])
            mesh.CreateFaceVertexCountsAttr([4])
            mesh.CreateFaceVertexIndicesAttr([0, 1, 2, 3])
            mesh.CreateOrientationAttr(UsdGeom.Tokens.leftHanded)
            mesh.GetPrim().AddAppliedSchema("PhysicsSurfaceDeformableSimAPI")
            stage.Save()

            builder = newton.ModelBuilder()
            builder.add_usd(str(usd_path))
            # The right-handed fan would give the first triangle (0, 1, 2); left-handed reverses it.
            self.assertEqual(list(builder.tri_indices[0]), [2, 1, 0])

    def test_plain_mesh_without_surface_api_is_not_cloth(self):
        """A triangle Mesh without the surface-deformable API must not produce cloth."""
        from pxr import Usd, UsdGeom, UsdPhysics

        with tempfile.TemporaryDirectory() as tmpdir:
            usd_path = Path(tmpdir) / "plain_mesh.usda"
            stage = Usd.Stage.CreateNew(str(usd_path))
            UsdPhysics.Scene.Define(stage, "/PhysicsScene")
            mesh = UsdGeom.Mesh.Define(stage, "/World/Mesh")
            mesh.CreatePointsAttr([(0.0, 0.0, 1.0), (1.0, 0.0, 1.0), (1.0, 1.0, 1.0)])
            mesh.CreateFaceVertexCountsAttr([3])
            mesh.CreateFaceVertexIndicesAttr([0, 1, 2])
            stage.Save()

            builder = newton.ModelBuilder()
            result = builder.add_usd(str(usd_path))
            self.assertEqual(result["path_cloth_map"], {})
            self.assertEqual(builder.particle_count, 0)

    def test_cloth_material_maps_to_triangle_and_edge_stiffness(self):
        """Bound surface material -> tri_ke (stretch), tri_ka (shear), edge bending (bend)."""
        from pxr import Usd, UsdPhysics

        with tempfile.TemporaryDirectory() as tmpdir:
            usd_path = Path(tmpdir) / "cloth_mat.usda"
            stage = Usd.Stage.CreateNew(str(usd_path))
            UsdPhysics.Scene.Define(stage, "/PhysicsScene")
            mesh = _add_cloth_mesh(stage, "/World/Cloth")
            stretch, shear, bend = 1.0e3, 5.0e2, 2.0e1
            _bind_cable_material(
                stage,
                mesh.GetPrim(),
                "/World/ClothMat",
                stretchStiffness=stretch,
                shearStiffness=shear,
                bendStiffness=bend,
            )
            stage.Save()

            builder = newton.ModelBuilder()
            result = builder.add_usd(str(usd_path))
            t0 = result["path_cloth_map"]["/World/Cloth"]["tri"][0]
            e0 = result["path_cloth_map"]["/World/Cloth"]["edge"][0]
            self.assertAlmostEqual(builder.tri_materials[t0][0], stretch, delta=stretch * 1e-3)  # tri_ke
            self.assertAlmostEqual(builder.tri_materials[t0][1], shear, delta=shear * 1e-3)  # tri_ka
            self.assertAlmostEqual(builder.edge_bending_properties[e0][0], bend, delta=bend * 1e-3)

    def test_cloth_zero_stiffness_is_preserved(self):
        """Authored zero stretch stiffness (range [0, inf)) maps to tri_ke = 0, not a default."""
        from pxr import Usd, UsdPhysics

        with tempfile.TemporaryDirectory() as tmpdir:
            usd_path = Path(tmpdir) / "cloth_zero.usda"
            stage = Usd.Stage.CreateNew(str(usd_path))
            UsdPhysics.Scene.Define(stage, "/PhysicsScene")
            mesh = _add_cloth_mesh(stage, "/World/Cloth")
            _bind_cable_material(
                stage,
                mesh.GetPrim(),
                "/World/ClothMat",
                stretchStiffness=0.0,
                bendStiffness=2.0e1,
            )
            stage.Save()

            builder = newton.ModelBuilder()
            result = builder.add_usd(str(usd_path))
            t0 = result["path_cloth_map"]["/World/Cloth"]["tri"][0]
            self.assertEqual(builder.tri_materials[t0][0], 0.0)  # tri_ke (stretch)
            self.assertEqual(result["path_cloth_attrs"]["/World/Cloth"]["material"]["stretchStiffness"], 0.0)

    def test_two_cloths_have_disjoint_ranges(self):
        """Two surface deformables map to disjoint, covering particle / triangle ranges."""
        from pxr import Usd, UsdPhysics

        with tempfile.TemporaryDirectory() as tmpdir:
            usd_path = Path(tmpdir) / "cloths.usda"
            stage = Usd.Stage.CreateNew(str(usd_path))
            UsdPhysics.Scene.Define(stage, "/PhysicsScene")
            _add_cloth_mesh(stage, "/World/ClothA")
            _add_cloth_mesh(stage, "/World/ClothB")
            stage.Save()

            builder = newton.ModelBuilder()
            result = builder.add_usd(str(usd_path))
            a = result["path_cloth_map"]["/World/ClothA"]
            b = result["path_cloth_map"]["/World/ClothB"]
            self.assertEqual(a["particle"], (0, 4))
            self.assertEqual(b["particle"], (4, 8))
            self.assertEqual(a["tri"], (0, 2))
            self.assertEqual(b["tri"], (2, 4))
            self.assertEqual(builder.particle_count, 8)

    def test_cloth_per_point_masses_take_precedence(self):
        """physics:masses authored on the cloth Mesh sets the particle masses directly."""
        from pxr import Sdf, Usd, UsdPhysics

        with tempfile.TemporaryDirectory() as tmpdir:
            usd_path = Path(tmpdir) / "cloth_masses.usda"
            stage = Usd.Stage.CreateNew(str(usd_path))
            UsdPhysics.Scene.Define(stage, "/PhysicsScene")
            mesh = _add_cloth_mesh(stage, "/World/Cloth")
            _bind_cable_material(stage, mesh.GetPrim(), "/World/ClothMat", density=1000.0)
            mesh.GetPrim().CreateAttribute("physics:masses", Sdf.ValueTypeNames.FloatArray).Set([1.0, 2.0, 3.0, 4.0])
            stage.Save()

            builder = newton.ModelBuilder()
            builder.add_usd(str(usd_path))
            self.assertEqual([builder.particle_mass[i] for i in range(4)], [1.0, 2.0, 3.0, 4.0])

    def test_cloth_thickness_scales_areal_density(self):
        """Surface thickness converts the volumetric material density to an areal density."""
        from pxr import Usd, UsdPhysics

        def total_mass(thickness=None):
            with tempfile.TemporaryDirectory() as tmpdir:
                usd_path = Path(tmpdir) / "cloth.usda"
                stage = Usd.Stage.CreateNew(str(usd_path))
                UsdPhysics.Scene.Define(stage, "/PhysicsScene")
                mesh = _add_cloth_mesh(stage, "/World/Cloth")
                attrs = {"density": 1000.0}
                if thickness is not None:
                    attrs["thickness"] = thickness
                _bind_cable_material(stage, mesh.GetPrim(), "/World/ClothMat", **attrs)
                stage.Save()
                builder = newton.ModelBuilder()
                builder.add_usd(str(usd_path))
                return sum(builder.particle_mass[:4])

        # Without thickness the density is used as areal; with thickness it scales by thickness.
        m_no_t = total_mass()
        m_with_t = total_mass(thickness=0.01)
        self.assertGreater(m_no_t, 0.0)
        self.assertAlmostEqual(m_with_t / m_no_t, 0.01, places=4)

    def test_cloth_resolved_density_is_volumetric_not_areal(self):
        """path_cloth_attrs.resolved_density is the solver-neutral volumetric density, not the areal value."""
        from pxr import Usd, UsdPhysics

        with tempfile.TemporaryDirectory() as tmpdir:
            usd_path = Path(tmpdir) / "cloth_density.usda"
            stage = Usd.Stage.CreateNew(str(usd_path))
            UsdPhysics.Scene.Define(stage, "/PhysicsScene")
            mesh = _add_cloth_mesh(stage, "/World/Cloth")
            _bind_cable_material(stage, mesh.GetPrim(), "/World/ClothMat", density=1000.0, thickness=0.01)
            stage.Save()

            builder = newton.ModelBuilder()
            result = builder.add_usd(str(usd_path))
            # Volumetric density (1000), not the areal 1000 * 0.01 passed to add_cloth_mesh.
            self.assertEqual(result["path_cloth_attrs"]["/World/Cloth"]["resolved_density"], 1000.0)

    def test_cloth_rest_bend_angles_warn(self):
        """Authored surface rest dihedral angles warn (import not yet supported), like rest shape."""
        from pxr import Sdf, Usd, UsdPhysics

        with tempfile.TemporaryDirectory() as tmpdir:
            usd_path = Path(tmpdir) / "cloth_rest.usda"
            stage = Usd.Stage.CreateNew(str(usd_path))
            UsdPhysics.Scene.Define(stage, "/PhysicsScene")
            mesh = _add_cloth_mesh(stage, "/World/Cloth")
            mesh.GetPrim().CreateAttribute("physics:restBendAngles", Sdf.ValueTypeNames.FloatArray).Set([0.1, 0.2])
            stage.Save()

            builder = newton.ModelBuilder()
            with self.assertWarnsRegex(UserWarning, "restBendAngles.*not yet supported"):
                builder.add_usd(str(usd_path))

    def test_cloth_non_uniform_scale_bakes_into_vertices(self):
        """A non-uniform xformOp:scale on a cloth mesh is baked into the particle positions."""
        from pxr import Gf, Usd, UsdGeom, UsdPhysics

        with tempfile.TemporaryDirectory() as tmpdir:
            usd_path = Path(tmpdir) / "cloth_scaled.usda"
            stage = Usd.Stage.CreateNew(str(usd_path))
            UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)  # avoid Y->Z axis conversion
            UsdPhysics.Scene.Define(stage, "/PhysicsScene")
            mesh = _add_cloth_mesh(stage, "/World/Cloth")  # points (0,0,1)(1,0,1)(1,1,1)(0,1,1)
            UsdGeom.Xformable(mesh).AddScaleOp().Set(Gf.Vec3d(2.0, 3.0, 4.0))
            stage.Save()

            builder = newton.ModelBuilder()
            builder.add_usd(str(usd_path))
            pq = np.array([list(builder.particle_q[i]) for i in range(builder.particle_count)])
            expected = np.array([(0.0, 0.0, 4.0), (2.0, 0.0, 4.0), (2.0, 3.0, 4.0), (0.0, 3.0, 4.0)])
            np.testing.assert_allclose(pq, expected, atol=1e-4)

    def test_cloth_simulates(self, device=None):
        """After parsing, a cloth runs through SolverVBD and stays finite."""
        from pxr import Usd, UsdPhysics

        if device is None or not wp.get_device(device).is_cuda:
            self.skipTest("VBD cloth simulation requires a CUDA device")

        with wp.ScopedDevice(device), tempfile.TemporaryDirectory() as tmpdir:
            usd_path = Path(tmpdir) / "cloth.usda"
            stage = Usd.Stage.CreateNew(str(usd_path))
            UsdPhysics.Scene.Define(stage, "/PhysicsScene")
            mesh = _add_cloth_mesh(stage, "/World/Cloth")
            _bind_cable_material(
                stage,
                mesh.GetPrim(),
                "/World/ClothMat",
                stretchStiffness=1.0e3,
                shearStiffness=1.0e3,
                bendStiffness=1.0e1,
                density=1.0,
            )
            stage.Save()

            builder = newton.ModelBuilder()
            builder.add_usd(str(usd_path))
            builder.add_ground_plane()
            builder.color()
            model = builder.finalize()

            solver = newton.solvers.SolverVBD(model, iterations=10)
            state_0, state_1, control = model.state(), model.state(), model.control()
            contacts = model.contacts()
            dt = 1.0 / 240.0
            for _ in range(20):
                state_0.clear_forces()
                model.collide(state_0, contacts)
                solver.step(state_0, state_1, control, contacts, dt)
                state_0, state_1 = state_1, state_0

            pq = state_0.particle_q.numpy()
            self.assertTrue(np.isfinite(pq).all(), "non-finite cloth particle positions after stepping")


class TestUSDDeformableVolume(unittest.TestCase):
    """Volume (TetMesh) soft-body addressability (REQ #3038)."""

    def test_tetmesh_imports_with_soft_range(self):
        """A UsdGeom.TetMesh imports as a soft body with a recoverable particle / tet range."""
        from pxr import Usd, UsdGeom, UsdPhysics

        with tempfile.TemporaryDirectory() as tmpdir:
            usd_path = Path(tmpdir) / "tet.usda"
            stage = Usd.Stage.CreateNew(str(usd_path))
            UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
            UsdPhysics.Scene.Define(stage, "/PhysicsScene")
            tet = UsdGeom.TetMesh.Define(stage, "/World/Soft")
            tet.CreatePointsAttr([(0.0, 0.0, 1.0), (1.0, 0.0, 1.0), (0.0, 1.0, 1.0), (0.0, 0.0, 2.0)])
            tet.CreateTetVertexIndicesAttr([(0, 1, 2, 3)])
            stage.Save()

            builder = newton.ModelBuilder()
            result = builder.add_usd(str(usd_path))
            ranges = result["path_soft_map"]["/World/Soft"]
            self.assertEqual(ranges["particle"], (0, 4))  # 4 tet vertices
            self.assertEqual(ranges["tet"], (0, 1))  # 1 tetrahedron
            self.assertEqual(builder.particle_count, 4)

    def test_rest_shape_points_warns(self):
        """An authored but unsupported physics:restShapePoints warns instead of being silently dropped."""
        from pxr import Sdf, Usd, UsdGeom, UsdPhysics

        with tempfile.TemporaryDirectory() as tmpdir:
            usd_path = Path(tmpdir) / "rest.usda"
            stage = Usd.Stage.CreateNew(str(usd_path))
            UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
            UsdPhysics.Scene.Define(stage, "/PhysicsScene")
            tet = _author_tet_cube(stage, "/World/Soft")
            tet.GetPrim().CreateAttribute("physics:restShapePoints", Sdf.ValueTypeNames.Point3fArray).Set(
                [(0.0, 0.0, 0.0)] * 8
            )
            stage.Save()

            builder = newton.ModelBuilder()
            with self.assertWarnsRegex(UserWarning, "restShapePoints.*not yet supported"):
                builder.add_usd(str(usd_path))

    def test_two_tetmeshes_have_disjoint_soft_ranges(self):
        """Two TetMesh soft bodies map to disjoint, covering particle ranges."""
        from pxr import Usd, UsdGeom, UsdPhysics

        with tempfile.TemporaryDirectory() as tmpdir:
            usd_path = Path(tmpdir) / "tets.usda"
            stage = Usd.Stage.CreateNew(str(usd_path))
            UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
            UsdPhysics.Scene.Define(stage, "/PhysicsScene")
            for name, dz in (("A", 0.0), ("B", 2.0)):
                tet = UsdGeom.TetMesh.Define(stage, f"/World/Soft{name}")
                tet.CreatePointsAttr(
                    [(0.0, 0.0, 1.0 + dz), (1.0, 0.0, 1.0 + dz), (0.0, 1.0, 1.0 + dz), (0.0, 0.0, 2.0 + dz)]
                )
                tet.CreateTetVertexIndicesAttr([(0, 1, 2, 3)])
            stage.Save()

            builder = newton.ModelBuilder()
            result = builder.add_usd(str(usd_path))
            ra = result["path_soft_map"]["/World/SoftA"]["particle"]
            rb = result["path_soft_map"]["/World/SoftB"]["particle"]
            self.assertEqual(ra, (0, 4))
            self.assertEqual(rb, (4, 8))
            self.assertEqual(builder.particle_count, 8)

    def test_soft_simulates(self, device=None):
        """After parsing, a tet soft body runs through SolverVBD and stays finite."""
        from pxr import Usd, UsdGeom, UsdPhysics

        if device is None or not wp.get_device(device).is_cuda:
            self.skipTest("VBD soft-body simulation requires a CUDA device")

        with wp.ScopedDevice(device), tempfile.TemporaryDirectory() as tmpdir:
            usd_path = Path(tmpdir) / "soft.usda"
            stage = Usd.Stage.CreateNew(str(usd_path))
            UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
            UsdPhysics.Scene.Define(stage, "/PhysicsScene")
            tet = _author_tet_cube(stage, "/World/Soft", z0=1.0)
            _bind_cable_material(
                stage, tet.GetPrim(), "/World/SoftMat", youngsModulus=1.0e5, poissonsRatio=0.3, density=1000.0
            )
            stage.Save()

            builder = newton.ModelBuilder()
            builder.add_usd(str(usd_path))
            builder.add_ground_plane()
            builder.color()
            model = builder.finalize()

            solver = newton.solvers.SolverVBD(model, iterations=10)
            state_0, state_1, control = model.state(), model.state(), model.control()
            contacts = model.contacts()
            dt = 1.0 / 240.0
            for _ in range(20):
                state_0.clear_forces()
                model.collide(state_0, contacts)
                solver.step(state_0, state_1, control, contacts, dt)
                state_0, state_1 = state_1, state_0

            pq = state_0.particle_q.numpy()
            self.assertTrue(np.isfinite(pq).all(), "non-finite soft-body particle positions after stepping")


def _author_tet_cube(stage, path, z0=0.0):
    """Author a unit-cube TetMesh (8 vertices, 5 tetrahedra) with its base at ``z0``."""
    from pxr import UsdGeom

    c = [
        (0.0, 0.0, z0),
        (1.0, 0.0, z0),
        (1.0, 1.0, z0),
        (0.0, 1.0, z0),
        (0.0, 0.0, z0 + 1.0),
        (1.0, 0.0, z0 + 1.0),
        (1.0, 1.0, z0 + 1.0),
        (0.0, 1.0, z0 + 1.0),
    ]
    tets = [(0, 1, 3, 4), (1, 2, 3, 6), (1, 3, 4, 6), (1, 4, 5, 6), (3, 4, 6, 7)]
    tet = UsdGeom.TetMesh.Define(stage, path)
    tet.CreatePointsAttr(c)
    tet.CreateTetVertexIndicesAttr(tets)
    tet.GetPrim().AddAppliedSchema("PhysicsVolumeDeformableSimAPI")
    return tet


def _author_unit_tet(stage, path):
    """Author a single-tetrahedron TetMesh (volume 1/6) at the given path."""
    from pxr import UsdGeom

    tet = UsdGeom.TetMesh.Define(stage, path)
    tet.CreatePointsAttr([(0.0, 0.0, 1.0), (1.0, 0.0, 1.0), (0.0, 1.0, 1.0), (0.0, 0.0, 2.0)])
    tet.CreateTetVertexIndicesAttr([(0, 1, 2, 3)])
    return tet


def _apply_deformable_body_api(prim, *, mass=None, density=None):
    """Apply PhysicsDeformableBodyAPI with optional mass / density overrides."""
    from pxr import Sdf

    prim.AddAppliedSchema("PhysicsDeformableBodyAPI")
    if mass is not None:
        prim.CreateAttribute("physics:mass", Sdf.ValueTypeNames.Float).Set(mass)
    if density is not None:
        prim.CreateAttribute("physics:density", Sdf.ValueTypeNames.Float).Set(density)


class TestUSDDeformableMass(unittest.TestCase):
    """Mass-distribution precedence (proposal "Mass Distribution")."""

    def _build_soft(self, author_fn):
        from pxr import Usd, UsdGeom, UsdPhysics

        with tempfile.TemporaryDirectory() as tmpdir:
            usd_path = Path(tmpdir) / "soft.usda"
            stage = Usd.Stage.CreateNew(str(usd_path))
            UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
            UsdPhysics.Scene.Define(stage, "/PhysicsScene")
            author_fn(stage)
            stage.Save()
            builder = newton.ModelBuilder()
            result = builder.add_usd(str(usd_path))
            return builder, result

    def test_per_point_masses_take_precedence(self):
        """physics:masses on the simulation geometry overrides body/material mass."""
        from pxr import Sdf

        def author(stage):
            tet = _author_unit_tet(stage, "/World/Soft")
            # Body mass + material density are present but per-point masses win.
            _apply_deformable_body_api(tet.GetPrim(), mass=99.0)
            tet.GetPrim().CreateAttribute("physics:masses", Sdf.ValueTypeNames.FloatArray).Set([1.0, 2.0, 3.0, 4.0])

        builder, _ = self._build_soft(author)
        self.assertEqual([builder.particle_mass[i] for i in range(4)], [1.0, 2.0, 3.0, 4.0])

    def test_body_mass_sets_total(self):
        """PhysicsDeformableBodyAPI.mass rescales the distribution to that total."""

        def author(stage):
            tet = _author_unit_tet(stage, "/World/Soft")
            _apply_deformable_body_api(tet.GetPrim(), mass=10.0)

        builder, _ = self._build_soft(author)
        self.assertAlmostEqual(sum(builder.particle_mass[:4]), 10.0, places=4)

    def test_body_density_overrides_material_density(self):
        """PhysicsDeformableBodyAPI.density takes precedence over the bound material."""

        def author_material_only(stage):
            tet = _author_unit_tet(stage, "/World/Soft")
            _bind_cable_material(stage, tet.GetPrim(), "/World/Mat", density=100.0)

        def author_with_override(stage):
            tet = _author_unit_tet(stage, "/World/Soft")
            _bind_cable_material(stage, tet.GetPrim(), "/World/Mat", density=100.0)
            _apply_deformable_body_api(tet.GetPrim(), density=500.0)

        builder_mat, _ = self._build_soft(author_material_only)
        builder_ovr, _ = self._build_soft(author_with_override)
        total_mat = sum(builder_mat.particle_mass[:4])
        total_ovr = sum(builder_ovr.particle_mass[:4])
        self.assertGreater(total_mat, 0.0)
        self.assertAlmostEqual(total_ovr / total_mat, 5.0, places=4)

    def test_body_api_on_ancestor_is_found(self):
        """PhysicsDeformableBodyAPI on an ancestor Xform governs a child sim geometry."""
        from pxr import UsdGeom

        def author(stage):
            UsdGeom.Xform.Define(stage, "/World/Body")
            _author_unit_tet(stage, "/World/Body/Soft")
            _apply_deformable_body_api(stage.GetPrimAtPath("/World/Body"), mass=7.0)

        builder, _ = self._build_soft(author)
        self.assertAlmostEqual(sum(builder.particle_mass[:4]), 7.0, places=4)

    def test_volume_sim_api_enables_per_point_masses(self):
        """A TetMesh marked PhysicsVolumeDeformableSimAPI honors physics:masses."""
        from pxr import Sdf

        def author(stage):
            tet = _author_unit_tet(stage, "/World/Soft")
            tet.GetPrim().AddAppliedSchema("PhysicsVolumeDeformableSimAPI")
            tet.GetPrim().CreateAttribute("physics:masses", Sdf.ValueTypeNames.FloatArray).Set([2.0, 4.0, 6.0, 8.0])

        builder, _ = self._build_soft(author)
        self.assertEqual([builder.particle_mass[i] for i in range(4)], [2.0, 4.0, 6.0, 8.0])

    def test_bare_tetmesh_ignores_per_point_masses(self):
        """A bare TetMesh (no deformable markers) keeps the legacy import; masses ignored."""
        from pxr import Sdf

        def author(stage):
            tet = _author_unit_tet(stage, "/World/Soft")
            tet.GetPrim().CreateAttribute("physics:masses", Sdf.ValueTypeNames.FloatArray).Set([2.0, 4.0, 6.0, 8.0])

        builder, _ = self._build_soft(author)
        # Legacy mass distribution (density-derived), not the authored per-point values.
        self.assertNotEqual([builder.particle_mass[i] for i in range(4)], [2.0, 4.0, 6.0, 8.0])

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
            result = builder.add_usd(str(usd_path))
            bodies, _ = result["path_cable_map"]["/World/Cable"]
            self.assertAlmostEqual(sum(builder.body_mass[b] for b in bodies), 2.5, places=4)

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
                if masses is not None:
                    curves.GetPrim().CreateAttribute("physics:masses", Sdf.ValueTypeNames.FloatArray).Set(masses)
                stage.Save()
                builder = newton.ModelBuilder()
                bodies, _ = builder.add_usd(str(usd_path))["path_cable_map"]["/World/Cable"]
                return sum(builder.body_mass[b] for b in bodies)

        baseline = total_cable_mass()
        with self.assertWarns(UserWarning):
            mismatched = total_cable_mass(masses=[1.0, 2.0, 3.0])  # length 3 != 4 points
        self.assertAlmostEqual(mismatched, baseline, places=6)


devices = get_selected_cuda_test_devices()
add_function_test(
    TestUSDDeformableCableAsset,
    "test_asset_cable_simulates",
    TestUSDDeformableCableAsset.test_asset_cable_simulates,
    devices=devices,
)
add_function_test(
    TestUSDDeformableCloth,
    "test_cloth_simulates",
    TestUSDDeformableCloth.test_cloth_simulates,
    devices=devices,
)
add_function_test(
    TestUSDDeformableVolume,
    "test_soft_simulates",
    TestUSDDeformableVolume.test_soft_simulates,
    devices=devices,
)


if __name__ == "__main__":
    unittest.main(verbosity=2)
