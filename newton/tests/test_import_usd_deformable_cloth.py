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

"""Tests for USD surface-deformable (cloth) import: triangulation, materials, masses, scaling."""

import math
import tempfile
import unittest
from pathlib import Path

import numpy as np
import warp as wp

import newton
from newton.tests._usd_deformable_test_utils import _add_cloth_mesh, _bind_deformable_material
from newton.tests.unittest_utils import USD_AVAILABLE, add_function_test, get_selected_cuda_test_devices


@unittest.skipUnless(USD_AVAILABLE, "Requires usd-core")
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

    def test_cloth_material_maps_to_isotropic_membrane(self):
        """Surface material -> isotropic membrane: stretchStiffness -> tri_ke, bendStiffness -> edge
        bending. tri_ka (area-preservation/Poisson) is left at the solver default since the proposal
        has no such attribute. shearStiffness can't be represented independently: it warns but is
        preserved in path_cloth_attrs."""
        from pxr import Usd, UsdPhysics

        with tempfile.TemporaryDirectory() as tmpdir:
            usd_path = Path(tmpdir) / "cloth_mat.usda"
            stage = Usd.Stage.CreateNew(str(usd_path))
            UsdPhysics.Scene.Define(stage, "/PhysicsScene")
            mesh = _add_cloth_mesh(stage, "/World/Cloth")
            stretch, shear, bend = 1.0e3, 5.0e2, 2.0e1  # distinct stretch != shear
            _bind_deformable_material(
                stage,
                mesh.GetPrim(),
                "/World/ClothMat",
                stretchStiffness=stretch,
                shearStiffness=shear,
                bendStiffness=bend,
            )
            stage.Save()

            builder = newton.ModelBuilder()
            with self.assertWarnsRegex(UserWarning, "shearStiffness is not applied"):
                result = builder.add_usd(str(usd_path))
            t0 = result["path_cloth_map"]["/World/Cloth"]["tri"][0]
            e0 = result["path_cloth_map"]["/World/Cloth"]["edge"][0]
            # stretchStiffness -> tri_ke (mu); tri_ka left at the solver default (not fabricated).
            self.assertAlmostEqual(builder.tri_materials[t0][0], stretch, delta=stretch * 1e-3)  # tri_ke (mu)
            self.assertEqual(builder.tri_materials[t0][1], builder.default_tri_ka)  # tri_ka (lambda) = default
            self.assertAlmostEqual(builder.edge_bending_properties[e0][0], bend, delta=bend * 1e-3)
            # The unmapped shearStiffness survives for anisotropic solvers.
            self.assertAlmostEqual(result["path_cloth_attrs"]["/World/Cloth"]["material"]["shearStiffness"], shear)

    def test_cloth_zero_stiffness_is_preserved(self):
        """Authored zero stretch stiffness (range [0, inf)) maps to tri_ke = 0, not a default."""
        from pxr import Usd, UsdPhysics

        with tempfile.TemporaryDirectory() as tmpdir:
            usd_path = Path(tmpdir) / "cloth_zero.usda"
            stage = Usd.Stage.CreateNew(str(usd_path))
            UsdPhysics.Scene.Define(stage, "/PhysicsScene")
            mesh = _add_cloth_mesh(stage, "/World/Cloth")
            _bind_deformable_material(
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
            _bind_deformable_material(stage, mesh.GetPrim(), "/World/ClothMat", density=1000.0)
            mesh.GetPrim().CreateAttribute("physics:masses", Sdf.ValueTypeNames.FloatArray).Set([1.0, 2.0, 3.0, 4.0])
            stage.Save()

            builder = newton.ModelBuilder()
            builder.add_usd(str(usd_path))
            self.assertEqual([builder.particle_mass[i] for i in range(4)], [1.0, 2.0, 3.0, 4.0])

    def test_invalid_per_point_masses_warn_and_fall_back(self):
        """Negative / inf / nan physics:masses warn and are ignored in favor of density mass.

        Per-point masses have the highest precedence, so a poisoned array would otherwise be
        written straight into the particles.
        """
        from pxr import Sdf, Usd, UsdPhysics

        for label, bad in (
            ("negative", [1.0, -2.0, 3.0, 4.0]),
            ("inf", [1.0, float("inf"), 3.0, 4.0]),
            ("nan", [1.0, float("nan"), 3.0, 4.0]),
        ):
            with self.subTest(kind=label), tempfile.TemporaryDirectory() as tmpdir:
                usd_path = Path(tmpdir) / f"cloth_bad_masses_{label}.usda"
                stage = Usd.Stage.CreateNew(str(usd_path))
                UsdPhysics.Scene.Define(stage, "/PhysicsScene")
                mesh = _add_cloth_mesh(stage, "/World/Cloth")
                _bind_deformable_material(stage, mesh.GetPrim(), "/World/ClothMat", density=1000.0)
                mesh.GetPrim().CreateAttribute("physics:masses", Sdf.ValueTypeNames.FloatArray).Set(bad)
                stage.Save()

                builder = newton.ModelBuilder()
                with self.assertWarnsRegex(UserWarning, "non-finite or negative"):
                    builder.add_usd(str(usd_path))

                masses = [builder.particle_mass[i] for i in range(4)]
                # Fell back to density-derived masses: all finite and strictly positive.
                for m in masses:
                    self.assertTrue(math.isfinite(m) and m > 0.0)
                self.assertNotEqual(masses, bad)

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
                _bind_deformable_material(stage, mesh.GetPrim(), "/World/ClothMat", **attrs)
                stage.Save()
                builder = newton.ModelBuilder()
                builder.add_usd(str(usd_path))
                return sum(builder.particle_mass[:4])

        # Without thickness the density is used as areal; with thickness it scales by thickness.
        m_no_t = total_mass()
        m_with_t = total_mass(thickness=0.01)
        self.assertGreater(m_no_t, 0.0)
        self.assertAlmostEqual(m_with_t / m_no_t, 0.01, places=4)

    def test_cloth_thickness_falls_back_to_shell_mass_model(self):
        """When the material omits thickness, a NewtonMassAPI shell thickness is used to areal-scale density."""
        from pxr import Sdf, Usd, UsdPhysics

        def total_mass(apply_shell):
            with tempfile.TemporaryDirectory() as tmpdir:
                usd_path = Path(tmpdir) / "cloth_shell.usda"
                stage = Usd.Stage.CreateNew(str(usd_path))
                UsdPhysics.Scene.Define(stage, "/PhysicsScene")
                mesh = _add_cloth_mesh(stage, "/World/Cloth")
                # Material density only -- thickness is left to the shell mass model below.
                _bind_deformable_material(stage, mesh.GetPrim(), "/World/ClothMat", density=1000.0)
                if apply_shell:
                    mesh.GetPrim().AddAppliedSchema("NewtonMassAPI")
                    mesh.GetPrim().CreateAttribute("newton:massModel", Sdf.ValueTypeNames.Token).Set("shell")
                    mesh.GetPrim().CreateAttribute("newton:shellThickness", Sdf.ValueTypeNames.Float).Set(0.01)
                stage.Save()
                builder = newton.ModelBuilder()
                builder.add_usd(str(usd_path))
                return sum(builder.particle_mass[:4])

        m_no_shell = total_mass(apply_shell=False)
        m_shell = total_mass(apply_shell=True)
        self.assertGreater(m_no_shell, 0.0)
        # Shell thickness 0.01 scales the areal density relative to the unscaled fallback.
        self.assertAlmostEqual(m_shell / m_no_shell, 0.01, places=4)

    def test_cloth_resolved_density_is_volumetric_not_areal(self):
        """path_cloth_attrs.resolved_density is the solver-neutral volumetric density, not the areal value."""
        from pxr import Usd, UsdPhysics

        with tempfile.TemporaryDirectory() as tmpdir:
            usd_path = Path(tmpdir) / "cloth_density.usda"
            stage = Usd.Stage.CreateNew(str(usd_path))
            UsdPhysics.Scene.Define(stage, "/PhysicsScene")
            mesh = _add_cloth_mesh(stage, "/World/Cloth")
            _bind_deformable_material(stage, mesh.GetPrim(), "/World/ClothMat", density=1000.0, thickness=0.01)
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

    def test_cloth_negative_scale_mirrors_and_flips_winding(self):
        """A reflective (negative) xformOp:scale mirrors the particles (parity preserved) and flips
        triangle winding, which a rotation+scale decomposition would silently drop."""
        from pxr import Gf, Usd, UsdGeom, UsdPhysics

        def import_cloth(scale):
            with tempfile.TemporaryDirectory() as tmpdir:
                usd_path = Path(tmpdir) / "cloth.usda"
                stage = Usd.Stage.CreateNew(str(usd_path))
                UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
                UsdPhysics.Scene.Define(stage, "/PhysicsScene")
                mesh = _add_cloth_mesh(stage, "/World/Cloth")  # points (0,0,1)(1,0,1)(1,1,1)(0,1,1)
                UsdGeom.Xformable(mesh).AddScaleOp().Set(Gf.Vec3d(*scale))
                stage.Save()
                builder = newton.ModelBuilder()
                builder.add_usd(str(usd_path))
                pq = np.array([list(builder.particle_q[i]) for i in range(builder.particle_count)])
                t0 = builder.tri_indices[0]
                return pq, list(t0)

        pq, tri0_reflected = import_cloth((-1.0, 1.0, 1.0))
        # The full affine mirrors X; a decomposition would yield positive X (0, 1, 1, 0).
        expected_x = np.array([0.0, -1.0, -1.0, 0.0])
        np.testing.assert_allclose(pq[:, 0], expected_x, atol=1e-4)

        _pq_pos, tri0_positive = import_cloth((1.0, 1.0, 1.0))
        # The reflection reverses the first triangle's winding relative to the non-reflected import.
        self.assertEqual(tri0_reflected, tri0_positive[::-1], "reflective scale must flip triangle winding")

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
            _bind_deformable_material(
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
            # VBD uses an isotropic membrane, so the authored shearStiffness is preserved but not applied.
            with self.assertWarnsRegex(UserWarning, "shearStiffness is not applied"):
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


devices = get_selected_cuda_test_devices()
add_function_test(
    TestUSDDeformableCloth,
    "test_cloth_simulates",
    TestUSDDeformableCloth.test_cloth_simulates,
    devices=devices,
)


if __name__ == "__main__":
    unittest.main(verbosity=2)
