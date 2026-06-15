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

import tempfile
import unittest
from pathlib import Path

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
        """Bound curve-deformable material → radius + per-joint stretch/bend stiffness."""
        import math

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


if __name__ == "__main__":
    unittest.main(verbosity=2)
