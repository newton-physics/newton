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

"""Tests for SDF USD attribute parsing."""

import tempfile
import unittest
from pathlib import Path

import warp as wp

import newton
from newton._src.utils.import_usd import parse_usd
from newton.tests.unittest_utils import add_function_test, get_selected_cuda_test_devices

CUBE_POINTS = [
    (-0.5, -0.5, -0.5),
    (0.5, -0.5, -0.5),
    (0.5, 0.5, -0.5),
    (-0.5, 0.5, -0.5),
    (-0.5, -0.5, 0.5),
    (0.5, -0.5, 0.5),
    (0.5, 0.5, 0.5),
    (-0.5, 0.5, 0.5),
]

CUBE_FACE_VERTEX_COUNTS = [4, 4, 4, 4, 4, 4]

CUBE_FACE_VERTEX_INDICES = [
    0,
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    0,
    1,
    5,
    4,
    2,
    3,
    7,
    6,
    0,
    3,
    7,
    4,
    1,
    2,
    6,
    5,
]


def _add_rigid_body(stage, path):
    from pxr import UsdPhysics

    prim = stage.DefinePrim(path, "Xform")
    UsdPhysics.RigidBodyAPI.Apply(prim)
    return prim


def _add_collision_mesh(stage, path):
    from pxr import UsdGeom, UsdPhysics

    mesh = UsdGeom.Mesh.Define(stage, path)
    UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())
    mesh.CreatePointsAttr(CUBE_POINTS)
    mesh.CreateFaceVertexCountsAttr(CUBE_FACE_VERTEX_COUNTS)
    mesh.CreateFaceVertexIndicesAttr(CUBE_FACE_VERTEX_INDICES)
    return mesh


class TestSDFUSDParsing(unittest.TestCase):
    """Tests for SDF attribute parsing from USD."""

    def test_usd_sdf_mesh_attributes(self, device=None):
        """USD newton:sdf* attributes cause SDF to be built during finalize()."""
        if device is None or not wp.get_device(device).is_cuda:
            self.skipTest("SDF tests require CUDA device")

        from pxr import Sdf, Usd, UsdPhysics

        with tempfile.TemporaryDirectory() as tmpdir:
            usd_path = Path(tmpdir) / "test_sdf.usda"
            stage = Usd.Stage.CreateNew(str(usd_path))
            UsdPhysics.Scene.Define(stage, "/PhysicsScene")

            # Body with SDF-configured mesh
            _add_rigid_body(stage, "/World/Body1")
            m1 = _add_collision_mesh(stage, "/World/Body1/CollisionMesh")
            p1 = m1.GetPrim()
            p1.CreateAttribute("newton:sdfMaxResolution", Sdf.ValueTypeNames.Int, custom=True).Set(128)
            p1.CreateAttribute("newton:sdfNarrowBandInner", Sdf.ValueTypeNames.Float, custom=True).Set(-0.02)
            p1.CreateAttribute("newton:sdfNarrowBandOuter", Sdf.ValueTypeNames.Float, custom=True).Set(0.02)

            # Body without SDF attributes (should use defaults)
            _add_rigid_body(stage, "/World/Body2")
            _add_collision_mesh(stage, "/World/Body2/CollisionMesh")

            stage.Save()

            builder = newton.ModelBuilder()
            result = parse_usd(builder, str(usd_path))
            psm = result["path_shape_map"]

            s1 = psm["/World/Body1/CollisionMesh"]
            s2 = psm["/World/Body2/CollisionMesh"]

            # SDF params stored on builder but not yet built (deferred to finalize)
            self.assertEqual(builder.shape_sdf_max_resolution[s1], 128)
            self.assertIsNone(builder.shape_sdf_max_resolution[s2])

            # After finalize, SDF is built on the mesh
            builder.finalize(device=device)
            mesh1 = builder.shape_source[s1]
            self.assertIsNotNone(mesh1.sdf, "Expected mesh.sdf built during finalize")

    def test_usd_sdf_defaults(self, device=None):
        """Shapes without SDF attributes should use builder defaults (None)."""
        if device is None or not wp.get_device(device).is_cuda:
            self.skipTest("SDF tests require CUDA device")

        from pxr import Usd, UsdPhysics

        with tempfile.TemporaryDirectory() as tmpdir:
            usd_path = Path(tmpdir) / "test_no_sdf.usda"
            stage = Usd.Stage.CreateNew(str(usd_path))
            UsdPhysics.Scene.Define(stage, "/PhysicsScene")

            _add_rigid_body(stage, "/World/Body1")
            _add_collision_mesh(stage, "/World/Body1/CollisionMesh")
            stage.Save()

            builder = newton.ModelBuilder()
            # Verify default_shape_cfg has no SDF enabled
            self.assertIsNone(builder.default_shape_cfg.sdf_max_resolution)
            self.assertIsNone(builder.default_shape_cfg.sdf_target_voxel_size)

            result = parse_usd(builder, str(usd_path))
            s1 = result["path_shape_map"]["/World/Body1/CollisionMesh"]

            # Mesh should not have SDF built
            mesh1 = builder.shape_source[s1]
            self.assertIsNone(mesh1.sdf)

    def test_usd_sdf_with_default_shape_cfg(self, device=None):
        """builder.default_shape_cfg.sdf_max_resolution applies to all shapes."""
        if device is None or not wp.get_device(device).is_cuda:
            self.skipTest("SDF tests require CUDA device")

        from pxr import Usd, UsdPhysics

        with tempfile.TemporaryDirectory() as tmpdir:
            usd_path = Path(tmpdir) / "test_default_sdf.usda"
            stage = Usd.Stage.CreateNew(str(usd_path))
            UsdPhysics.Scene.Define(stage, "/PhysicsScene")

            _add_rigid_body(stage, "/World/Body1")
            _add_collision_mesh(stage, "/World/Body1/CollisionMesh")
            stage.Save()

            builder = newton.ModelBuilder()
            builder.default_shape_cfg.sdf_max_resolution = 64

            result = parse_usd(builder, str(usd_path))
            s1 = result["path_shape_map"]["/World/Body1/CollisionMesh"]

            # SDF params stored, deferred to finalize
            self.assertEqual(builder.shape_sdf_max_resolution[s1], 64)

            builder.finalize(device=device)
            mesh1 = builder.shape_source[s1]
            self.assertIsNotNone(mesh1.sdf, "Expected SDF built from default_shape_cfg during finalize")

    def test_usd_hydroelastic_attributes(self, device=None):
        """Presence of newton:kh signals hydroelastic opt-in (NewtonHydroelasticCollisionAPI)."""
        if device is None or not wp.get_device(device).is_cuda:
            self.skipTest("SDF tests require CUDA device")

        from pxr import Sdf, Usd, UsdPhysics

        with tempfile.TemporaryDirectory() as tmpdir:
            usd_path = Path(tmpdir) / "test_hydro.usda"
            stage = Usd.Stage.CreateNew(str(usd_path))
            UsdPhysics.Scene.Define(stage, "/PhysicsScene")

            _add_rigid_body(stage, "/World/Body1")
            m1 = _add_collision_mesh(stage, "/World/Body1/CollisionMesh")
            p1 = m1.GetPrim()
            # Set SDF + hydroelastic (kh presence = opt-in, no isHydroelastic bool)
            p1.CreateAttribute("newton:sdfMaxResolution", Sdf.ValueTypeNames.Int, custom=True).Set(128)
            p1.CreateAttribute("newton:kh", Sdf.ValueTypeNames.Float, custom=True).Set(1e7)

            # Body2: no hydroelastic
            _add_rigid_body(stage, "/World/Body2")
            _add_collision_mesh(stage, "/World/Body2/CollisionMesh")

            stage.Save()

            builder = newton.ModelBuilder()
            result = parse_usd(builder, str(usd_path))
            psm = result["path_shape_map"]

            s1 = psm["/World/Body1/CollisionMesh"]
            s2 = psm["/World/Body2/CollisionMesh"]

            # Body1: hydroelastic enabled
            self.assertTrue(builder.shape_flags[s1] & newton.ShapeFlags.HYDROELASTIC)
            self.assertAlmostEqual(builder.shape_material_kh[s1], 1e7)

            # Body2: hydroelastic disabled (default)
            self.assertFalse(builder.shape_flags[s2] & newton.ShapeFlags.HYDROELASTIC)

    def test_usd_sdf_margin(self, device=None):
        """USD newton:sdfMargin is passed to mesh.build_sdf(margin=...)."""
        if device is None or not wp.get_device(device).is_cuda:
            self.skipTest("SDF tests require CUDA device")

        from pxr import Sdf, Usd, UsdPhysics

        with tempfile.TemporaryDirectory() as tmpdir:
            usd_path = Path(tmpdir) / "test_sdf_margin.usda"
            stage = Usd.Stage.CreateNew(str(usd_path))
            UsdPhysics.Scene.Define(stage, "/PhysicsScene")

            _add_rigid_body(stage, "/World/Body1")
            m1 = _add_collision_mesh(stage, "/World/Body1/CollisionMesh")
            p1 = m1.GetPrim()
            p1.CreateAttribute("newton:sdfMaxResolution", Sdf.ValueTypeNames.Int, custom=True).Set(64)
            p1.CreateAttribute("newton:sdfMargin", Sdf.ValueTypeNames.Float, custom=True).Set(0.05)

            stage.Save()

            builder = newton.ModelBuilder()
            result = parse_usd(builder, str(usd_path))
            s1 = result["path_shape_map"]["/World/Body1/CollisionMesh"]

            # SDF deferred to finalize
            builder.finalize(device=device)
            mesh1 = builder.shape_source[s1]
            self.assertIsNotNone(mesh1.sdf, "Expected SDF built with sdfMargin during finalize")

    def test_usd_sdf_enabled_false(self, device=None):
        """newton:sdfEnabled=false suppresses SDF building even with params authored."""
        if device is None or not wp.get_device(device).is_cuda:
            self.skipTest("SDF tests require CUDA device")

        from pxr import Sdf, Usd, UsdPhysics

        with tempfile.TemporaryDirectory() as tmpdir:
            usd_path = Path(tmpdir) / "test_sdf_disabled.usda"
            stage = Usd.Stage.CreateNew(str(usd_path))
            UsdPhysics.Scene.Define(stage, "/PhysicsScene")

            _add_rigid_body(stage, "/World/Body1")
            m1 = _add_collision_mesh(stage, "/World/Body1/CollisionMesh")
            p1 = m1.GetPrim()
            p1.CreateAttribute("newton:sdfEnabled", Sdf.ValueTypeNames.Bool, custom=True).Set(False)
            p1.CreateAttribute("newton:sdfMaxResolution", Sdf.ValueTypeNames.Int, custom=True).Set(128)

            stage.Save()

            builder = newton.ModelBuilder()
            result = parse_usd(builder, str(usd_path))
            s1 = result["path_shape_map"]["/World/Body1/CollisionMesh"]

            # SDF params should not be stored when sdfEnabled=false
            self.assertIsNone(builder.shape_sdf_max_resolution[s1])

            builder.finalize(device=device)
            mesh1 = builder.shape_source[s1]
            self.assertIsNone(mesh1.sdf, "SDF should not be built when sdfEnabled=false")

    def test_usd_hydroelastic_enabled_false(self, device=None):
        """newton:hydroelasticEnabled=false suppresses hydroelastic even with kh authored."""
        if device is None or not wp.get_device(device).is_cuda:
            self.skipTest("SDF tests require CUDA device")

        from pxr import Sdf, Usd, UsdPhysics

        with tempfile.TemporaryDirectory() as tmpdir:
            usd_path = Path(tmpdir) / "test_hydro_disabled.usda"
            stage = Usd.Stage.CreateNew(str(usd_path))
            UsdPhysics.Scene.Define(stage, "/PhysicsScene")

            _add_rigid_body(stage, "/World/Body1")
            m1 = _add_collision_mesh(stage, "/World/Body1/CollisionMesh")
            p1 = m1.GetPrim()
            p1.CreateAttribute("newton:sdfMaxResolution", Sdf.ValueTypeNames.Int, custom=True).Set(128)
            p1.CreateAttribute("newton:kh", Sdf.ValueTypeNames.Float, custom=True).Set(1e7)
            p1.CreateAttribute("newton:hydroelasticEnabled", Sdf.ValueTypeNames.Bool, custom=True).Set(False)

            stage.Save()

            builder = newton.ModelBuilder()
            result = parse_usd(builder, str(usd_path))
            s1 = result["path_shape_map"]["/World/Body1/CollisionMesh"]

            # SDF should still be built (sdfEnabled not false), but hydroelastic should be off
            self.assertEqual(builder.shape_sdf_max_resolution[s1], 128)
            self.assertFalse(builder.shape_flags[s1] & newton.ShapeFlags.HYDROELASTIC)

    def test_usd_sdf_fractional_narrow_band(self, device=None):
        """Fractional narrow band overrides absolute, scaled by local bbox diagonal."""
        if device is None or not wp.get_device(device).is_cuda:
            self.skipTest("SDF tests require CUDA device")

        from pxr import Sdf, Usd, UsdPhysics

        with tempfile.TemporaryDirectory() as tmpdir:
            usd_path = Path(tmpdir) / "test_sdf_frac_nb.usda"
            stage = Usd.Stage.CreateNew(str(usd_path))
            UsdPhysics.Scene.Define(stage, "/PhysicsScene")

            _add_rigid_body(stage, "/World/Body1")
            m1 = _add_collision_mesh(stage, "/World/Body1/CollisionMesh")
            p1 = m1.GetPrim()
            p1.CreateAttribute("newton:sdfMaxResolution", Sdf.ValueTypeNames.Int, custom=True).Set(64)
            # CUBE_POINTS is a unit cube centered at origin: bbox diagonal = sqrt(3)
            p1.CreateAttribute("newton:sdfNarrowBandInnerFraction", Sdf.ValueTypeNames.Float, custom=True).Set(-0.01)
            p1.CreateAttribute("newton:sdfNarrowBandOuterFraction", Sdf.ValueTypeNames.Float, custom=True).Set(0.01)

            stage.Save()

            builder = newton.ModelBuilder()
            result = parse_usd(builder, str(usd_path))
            s1 = result["path_shape_map"]["/World/Body1/CollisionMesh"]

            import math

            expected_diag = math.sqrt(3)  # unit cube [-0.5, 0.5]^3
            inner, outer = builder.shape_sdf_narrow_band_range[s1]
            self.assertAlmostEqual(inner, -0.01 * expected_diag, places=5)
            self.assertAlmostEqual(outer, 0.01 * expected_diag, places=5)

    def test_usd_sdf_fractional_overrides_absolute(self, device=None):
        """When both fraction and absolute are authored, fraction wins."""
        if device is None or not wp.get_device(device).is_cuda:
            self.skipTest("SDF tests require CUDA device")

        from pxr import Sdf, Usd, UsdPhysics

        with tempfile.TemporaryDirectory() as tmpdir:
            usd_path = Path(tmpdir) / "test_sdf_frac_wins.usda"
            stage = Usd.Stage.CreateNew(str(usd_path))
            UsdPhysics.Scene.Define(stage, "/PhysicsScene")

            _add_rigid_body(stage, "/World/Body1")
            m1 = _add_collision_mesh(stage, "/World/Body1/CollisionMesh")
            p1 = m1.GetPrim()
            p1.CreateAttribute("newton:sdfMaxResolution", Sdf.ValueTypeNames.Int, custom=True).Set(64)
            # Both authored — fraction should win
            p1.CreateAttribute("newton:sdfNarrowBandOuter", Sdf.ValueTypeNames.Float, custom=True).Set(0.5)
            p1.CreateAttribute("newton:sdfNarrowBandOuterFraction", Sdf.ValueTypeNames.Float, custom=True).Set(0.02)

            stage.Save()

            builder = newton.ModelBuilder()
            result = parse_usd(builder, str(usd_path))
            s1 = result["path_shape_map"]["/World/Body1/CollisionMesh"]

            import math

            expected_diag = math.sqrt(3)
            _inner, outer = builder.shape_sdf_narrow_band_range[s1]
            # Fraction (0.02 * sqrt(3) ≈ 0.0346) should win over absolute (0.5)
            self.assertAlmostEqual(outer, 0.02 * expected_diag, places=5)

    def test_usd_sdf_fractional_margin(self, device=None):
        """Fractional margin overrides absolute, scaled by local bbox diagonal."""
        if device is None or not wp.get_device(device).is_cuda:
            self.skipTest("SDF tests require CUDA device")

        from pxr import Sdf, Usd, UsdPhysics

        with tempfile.TemporaryDirectory() as tmpdir:
            usd_path = Path(tmpdir) / "test_sdf_frac_margin.usda"
            stage = Usd.Stage.CreateNew(str(usd_path))
            UsdPhysics.Scene.Define(stage, "/PhysicsScene")

            _add_rigid_body(stage, "/World/Body1")
            m1 = _add_collision_mesh(stage, "/World/Body1/CollisionMesh")
            p1 = m1.GetPrim()
            p1.CreateAttribute("newton:sdfMaxResolution", Sdf.ValueTypeNames.Int, custom=True).Set(64)
            p1.CreateAttribute("newton:sdfMarginFraction", Sdf.ValueTypeNames.Float, custom=True).Set(0.05)

            stage.Save()

            builder = newton.ModelBuilder()
            result = parse_usd(builder, str(usd_path))
            s1 = result["path_shape_map"]["/World/Body1/CollisionMesh"]

            import math

            # Unit cube bbox diagonal = sqrt(3). For meshes, sdfMargin is stored in shape_gap.
            expected_diag = math.sqrt(3)
            self.assertAlmostEqual(builder.shape_gap[s1], 0.05 * expected_diag, places=5)


devices = get_selected_cuda_test_devices()
add_function_test(
    TestSDFUSDParsing, "test_usd_sdf_mesh_attributes", TestSDFUSDParsing.test_usd_sdf_mesh_attributes, devices=devices
)
add_function_test(TestSDFUSDParsing, "test_usd_sdf_defaults", TestSDFUSDParsing.test_usd_sdf_defaults, devices=devices)
add_function_test(
    TestSDFUSDParsing,
    "test_usd_sdf_with_default_shape_cfg",
    TestSDFUSDParsing.test_usd_sdf_with_default_shape_cfg,
    devices=devices,
)
add_function_test(
    TestSDFUSDParsing,
    "test_usd_hydroelastic_attributes",
    TestSDFUSDParsing.test_usd_hydroelastic_attributes,
    devices=devices,
)
add_function_test(
    TestSDFUSDParsing,
    "test_usd_sdf_margin",
    TestSDFUSDParsing.test_usd_sdf_margin,
    devices=devices,
)
add_function_test(
    TestSDFUSDParsing,
    "test_usd_sdf_enabled_false",
    TestSDFUSDParsing.test_usd_sdf_enabled_false,
    devices=devices,
)
add_function_test(
    TestSDFUSDParsing,
    "test_usd_hydroelastic_enabled_false",
    TestSDFUSDParsing.test_usd_hydroelastic_enabled_false,
    devices=devices,
)
add_function_test(
    TestSDFUSDParsing,
    "test_usd_sdf_fractional_narrow_band",
    TestSDFUSDParsing.test_usd_sdf_fractional_narrow_band,
    devices=devices,
)
add_function_test(
    TestSDFUSDParsing,
    "test_usd_sdf_fractional_overrides_absolute",
    TestSDFUSDParsing.test_usd_sdf_fractional_overrides_absolute,
    devices=devices,
)
add_function_test(
    TestSDFUSDParsing,
    "test_usd_sdf_fractional_margin",
    TestSDFUSDParsing.test_usd_sdf_fractional_margin,
    devices=devices,
)


if __name__ == "__main__":
    wp.clear_kernel_cache()
    unittest.main(verbosity=2)
