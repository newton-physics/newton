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

"""Tests for USD volume-deformable (TetMesh) import: soft-body ranges and tet reorientation."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import warp as wp

import newton
from newton.tests._usd_deformable_test_utils import _apply_deformable_body_api, _bind_deformable_material
from newton.tests.unittest_utils import USD_AVAILABLE, add_function_test, get_selected_cuda_test_devices


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


def _author_two_tet_wedge(stage, path):
    """Author a TetMesh of two tets that share a base triangle but have very different
    volumes, so density-based per-point masses must be non-uniform. Both tets are wound
    for positive signed volume.

    Vertices 0,1,2 form the shared base; vertex 3 is the apex of the large tet (V = 4/6)
    and vertex 4 the apex of the small tet (V = 1/6)."""
    from pxr import UsdGeom

    tet = UsdGeom.TetMesh.Define(stage, path)
    tet.CreatePointsAttr([(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 4.0), (0.0, 0.0, -1.0)])
    tet.CreateTetVertexIndicesAttr([(0, 1, 2, 3), (0, 2, 1, 4)])
    tet.GetPrim().AddAppliedSchema("PhysicsVolumeDeformableSimAPI")
    return tet


@unittest.skipUnless(USD_AVAILABLE, "Requires usd-core")
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

    def test_volume_negative_scale_mirrors_and_reorients_tets(self):
        """A reflective xformOp:scale mirrors the soft-body particles and reorients each tet to keep a
        positive rest volume; a rotation+scale decomposition would drop the reflection."""
        from pxr import Gf, Usd, UsdGeom, UsdPhysics

        with tempfile.TemporaryDirectory() as tmpdir:
            usd_path = Path(tmpdir) / "tet_reflected.usda"
            stage = Usd.Stage.CreateNew(str(usd_path))
            UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
            UsdPhysics.Scene.Define(stage, "/PhysicsScene")
            tet = UsdGeom.TetMesh.Define(stage, "/World/Soft")
            tet.CreatePointsAttr([(0.0, 0.0, 1.0), (1.0, 0.0, 1.0), (0.0, 1.0, 1.0), (0.0, 0.0, 2.0)])
            tet.CreateTetVertexIndicesAttr([(0, 1, 2, 3)])
            UsdGeom.Xformable(tet).AddScaleOp().Set(Gf.Vec3d(-1.0, 1.0, 1.0))
            stage.Save()

            builder = newton.ModelBuilder()
            result = builder.add_usd(str(usd_path))
            p0, p1 = result["path_soft_map"]["/World/Soft"]["particle"]
            pq = np.array([list(builder.particle_q[i]) for i in range(p0, p1)])
            # Original X {0, 1, 0, 0} mirrors to {0, -1, 0, 0}.
            np.testing.assert_allclose(sorted(pq[:, 0]), [-1.0, 0.0, 0.0, 0.0], atol=1e-4)

            # The imported tet keeps a positive signed rest volume (winding repaired for the reflection).
            t0, _t1 = result["path_soft_map"]["/World/Soft"]["tet"]
            i, j, k, m = builder.tet_indices[t0]

            def pos(n):
                return np.array(list(builder.particle_q[n]))

            signed_vol = np.dot(pos(j) - pos(i), np.cross(pos(k) - pos(i), pos(m) - pos(i))) / 6.0
            self.assertGreater(signed_vol, 0.0, "reflected tet must keep a positive rest volume")

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
            _bind_deformable_material(
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

    def test_body_mass_override_preserves_volume_weighting(self):
        """A body-mass override must rescale the per-point masses *proportionally*, preserving the
        volume weighting (proposal: m_p = sum_{e in tau(p)} V_e / T). The importer's rescale is
        ``particle_mass[i] *= body_mass / current``; a uniform ``body_mass / n`` would also hit the
        total but flatten the distribution, so assert the per-point ratios, not just the sum."""
        body_mass = 10.0
        v_large, v_small = 4.0 / 6.0, 1.0 / 6.0  # the two authored tet volumes
        total_vol = v_large + v_small

        def author(stage):
            tet = _author_two_tet_wedge(stage, "/World/Soft")
            _apply_deformable_body_api(tet.GetPrim(), mass=body_mass)

        builder, _ = self._build_soft(author)
        m = [builder.particle_mass[i] for i in range(5)]
        # The override sets the total ...
        self.assertAlmostEqual(sum(m), body_mass, places=4)
        # ... but the distribution still follows adjacent-element volume. Apexes sit on one tet
        # each (V_e / 4); shared base vertices sum both tets ((V_large + V_small) / 4).
        self.assertAlmostEqual(m[3], body_mass * (v_large / 4.0) / total_vol, places=4)  # large apex
        self.assertAlmostEqual(m[4], body_mass * (v_small / 4.0) / total_vol, places=4)  # small apex
        self.assertAlmostEqual(m[3] / m[4], v_large / v_small, places=4)  # = 4, weighting preserved
        for i in range(3):
            self.assertAlmostEqual(m[i], body_mass / 4.0, places=4)  # shared = (V_large+V_small)/4 scaled
        self.assertGreater(max(m) - min(m), 1.0e-6)  # genuinely non-uniform, not flattened

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
            _bind_deformable_material(stage, tet.GetPrim(), "/World/Mat", density=100.0)

        def author_with_override(stage):
            tet = _author_unit_tet(stage, "/World/Soft")
            _bind_deformable_material(stage, tet.GetPrim(), "/World/Mat", density=100.0)
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


devices = get_selected_cuda_test_devices()
add_function_test(
    TestUSDDeformableVolume,
    "test_soft_simulates",
    TestUSDDeformableVolume.test_soft_simulates,
    devices=devices,
)


if __name__ == "__main__":
    unittest.main(verbosity=2)
