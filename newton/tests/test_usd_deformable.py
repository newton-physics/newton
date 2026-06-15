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


def _bind_cable_material(stage, prim, mat_path, **attrs):
    """Author a curve-deformable material under omniphysics: and bind it to a prim."""
    from pxr import Sdf, UsdShade

    mat = UsdShade.Material.Define(stage, mat_path)
    for name, value in attrs.items():
        mat.GetPrim().CreateAttribute(f"omniphysics:{name}", Sdf.ValueTypeNames.Float).Set(value)
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
        """Each cable body origin matches the authored segment start point (map points at the right bodies)."""
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

            # Body i origin sits at segment start points[i] (origin == start on main).
            for i, body in enumerate(bodies):
                origin = np.array(builder.body_q[body][:3], dtype=np.float32)
                np.testing.assert_allclose(origin, np.array(pts[i], dtype=np.float32), atol=1e-5)

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

    def test_cable_articulation_label_survives_finalize(self):
        """The cable's articulation is labeled by prim path - the replication-durable handle."""
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


if __name__ == "__main__":
    unittest.main(verbosity=2)
