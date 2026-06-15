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


if __name__ == "__main__":
    unittest.main(verbosity=2)
