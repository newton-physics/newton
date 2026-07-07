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
            p1.CreateAttribute("newton:sdfTextureFormat", Sdf.ValueTypeNames.Token, custom=True).Set("float32")

            # Body without SDF attributes (should use defaults)
            _add_rigid_body(stage, "/World/Body2")
            _add_collision_mesh(stage, "/World/Body2/CollisionMesh")

            stage.Save()

            builder = newton.ModelBuilder()
            result = builder.add_usd(str(usd_path))
            psm = result["path_shape_map"]

            s1 = psm["/World/Body1/CollisionMesh"]
            s2 = psm["/World/Body2/CollisionMesh"]

            # SDF params stored on builder but not yet built (deferred to finalize)
            self.assertEqual(builder.shape_sdf_max_resolution[s1], 128)
            self.assertIsNone(builder.shape_sdf_max_resolution[s2])
            self.assertAlmostEqual(builder.shape_sdf_narrow_band_range[s1][0], -0.02, places=4)
            self.assertAlmostEqual(builder.shape_sdf_narrow_band_range[s1][1], 0.02, places=4)
            self.assertEqual(builder.shape_sdf_texture_format[s1], "float32")
            self.assertEqual(builder.shape_sdf_texture_format[s2], "uint16")

            # After finalize, the deferred SDF lands in the model (not on the
            # shared Mesh — finalize() must not mutate user-owned geometry).
            model = builder.finalize(device=device)
            self.assertGreaterEqual(
                int(model._shape_sdf_index.numpy()[s1]),
                0,
                "Expected an SDF entry for shape s1 in the finalized model.",
            )

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

            result = builder.add_usd(str(usd_path))
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

            result = builder.add_usd(str(usd_path))
            s1 = result["path_shape_map"]["/World/Body1/CollisionMesh"]

            # SDF params stored, deferred to finalize
            self.assertEqual(builder.shape_sdf_max_resolution[s1], 64)

            model = builder.finalize(device=device)
            self.assertGreaterEqual(
                int(model._shape_sdf_index.numpy()[s1]),
                0,
                "Expected SDF built from default_shape_cfg during finalize.",
            )

    def test_usd_hydroelastic_attributes(self, device=None):
        """Authoring newton:hydroelasticEnabled=true with kh on NewtonSDFCollisionAPI opts into hydroelastic."""
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
            # Apply the SDF API and opt into hydroelastic via the enable bool.
            p1.AddAppliedSchema("NewtonSDFCollisionAPI")
            p1.CreateAttribute("newton:sdfMaxResolution", Sdf.ValueTypeNames.Int, custom=True).Set(128)
            p1.CreateAttribute("newton:hydroelasticEnabled", Sdf.ValueTypeNames.Bool, custom=True).Set(True)
            p1.CreateAttribute("newton:hydroelasticStiffness", Sdf.ValueTypeNames.Float, custom=True).Set(1e7)

            # Body2: no hydroelastic
            _add_rigid_body(stage, "/World/Body2")
            _add_collision_mesh(stage, "/World/Body2/CollisionMesh")

            stage.Save()

            builder = newton.ModelBuilder()
            result = builder.add_usd(str(usd_path))
            psm = result["path_shape_map"]

            s1 = psm["/World/Body1/CollisionMesh"]
            s2 = psm["/World/Body2/CollisionMesh"]

            # Body1: hydroelastic enabled
            self.assertTrue(builder.shape_flags[s1] & newton.ShapeFlags.HYDROELASTIC)
            self.assertAlmostEqual(builder.shape_material_kh[s1], 1e7)

            # Body2: hydroelastic disabled (default)
            self.assertFalse(builder.shape_flags[s2] & newton.ShapeFlags.HYDROELASTIC)

    def test_usd_sdf_padding(self, device=None):
        """USD newton:sdfPadding is passed to mesh.build_sdf(margin=...)."""
        if device is None or not wp.get_device(device).is_cuda:
            self.skipTest("SDF tests require CUDA device")

        from pxr import Sdf, Usd, UsdPhysics

        with tempfile.TemporaryDirectory() as tmpdir:
            usd_path = Path(tmpdir) / "test_sdf_padding.usda"
            stage = Usd.Stage.CreateNew(str(usd_path))
            UsdPhysics.Scene.Define(stage, "/PhysicsScene")

            _add_rigid_body(stage, "/World/Body1")
            m1 = _add_collision_mesh(stage, "/World/Body1/CollisionMesh")
            p1 = m1.GetPrim()
            p1.CreateAttribute("newton:sdfMaxResolution", Sdf.ValueTypeNames.Int, custom=True).Set(64)
            p1.CreateAttribute("newton:sdfPadding", Sdf.ValueTypeNames.Float, custom=True).Set(0.05)

            stage.Save()

            builder = newton.ModelBuilder()
            result = builder.add_usd(str(usd_path))
            s1 = result["path_shape_map"]["/World/Body1/CollisionMesh"]

            # SDF deferred to finalize; result lands in the model, not on the
            # shared Mesh.
            model = builder.finalize(device=device)
            self.assertGreaterEqual(
                int(model._shape_sdf_index.numpy()[s1]),
                0,
                "Expected SDF built with sdfPadding during finalize.",
            )

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
            p1.CreateAttribute("newton:hydroelasticStiffness", Sdf.ValueTypeNames.Float, custom=True).Set(1e7)
            p1.CreateAttribute("newton:hydroelasticEnabled", Sdf.ValueTypeNames.Bool, custom=True).Set(False)

            stage.Save()

            builder = newton.ModelBuilder()
            result = builder.add_usd(str(usd_path))
            s1 = result["path_shape_map"]["/World/Body1/CollisionMesh"]

            # SDF should still be built (sdfEnabled not false), but hydroelastic should be off
            self.assertEqual(builder.shape_sdf_max_resolution[s1], 128)
            self.assertFalse(builder.shape_flags[s1] & newton.ShapeFlags.HYDROELASTIC)

    def test_usd_sdf_padding_hydroelastic_primitive(self, device=None):
        """newton:sdfPadding on a hydroelastic primitive (Sphere) is routed to shape_sdf_padding."""
        if device is None or not wp.get_device(device).is_cuda:
            self.skipTest("SDF tests require CUDA device")

        from pxr import Sdf, Usd, UsdGeom, UsdPhysics

        with tempfile.TemporaryDirectory() as tmpdir:
            usd_path = Path(tmpdir) / "test_sdf_padding_sphere.usda"
            stage = Usd.Stage.CreateNew(str(usd_path))
            UsdPhysics.Scene.Define(stage, "/PhysicsScene")

            _add_rigid_body(stage, "/World/Body1")
            sphere = UsdGeom.Sphere.Define(stage, "/World/Body1/CollisionSphere")
            sphere.CreateRadiusAttr(0.2)
            UsdPhysics.CollisionAPI.Apply(sphere.GetPrim())
            p1 = sphere.GetPrim()
            p1.AddAppliedSchema("NewtonSDFCollisionAPI")
            p1.CreateAttribute("newton:sdfMaxResolution", Sdf.ValueTypeNames.Int, custom=True).Set(32)
            p1.CreateAttribute("newton:hydroelasticEnabled", Sdf.ValueTypeNames.Bool, custom=True).Set(True)
            p1.CreateAttribute("newton:hydroelasticStiffness", Sdf.ValueTypeNames.Float, custom=True).Set(1e7)
            p1.CreateAttribute("newton:sdfPadding", Sdf.ValueTypeNames.Float, custom=True).Set(0.03)

            stage.Save()

            builder = newton.ModelBuilder()
            result = builder.add_usd(str(usd_path))
            s1 = result["path_shape_map"]["/World/Body1/CollisionSphere"]

            self.assertTrue(builder.shape_flags[s1] & newton.ShapeFlags.HYDROELASTIC)
            # sdfPadding lives in its own per-shape list, not the collision gap.
            self.assertAlmostEqual(builder.shape_sdf_padding[s1], 0.03, places=5)

    def test_usd_sdf_padding_does_not_affect_shape_gap(self, device=None):
        """newton:sdfPadding and newton:contactGap populate distinct per-shape lists."""
        if device is None or not wp.get_device(device).is_cuda:
            self.skipTest("SDF tests require CUDA device")

        from pxr import Sdf, Usd, UsdGeom, UsdPhysics

        with tempfile.TemporaryDirectory() as tmpdir:
            usd_path = Path(tmpdir) / "test_sdf_padding_vs_gap.usda"
            stage = Usd.Stage.CreateNew(str(usd_path))
            UsdPhysics.Scene.Define(stage, "/PhysicsScene")

            _add_rigid_body(stage, "/World/Body1")
            sphere = UsdGeom.Sphere.Define(stage, "/World/Body1/CollisionSphere")
            sphere.CreateRadiusAttr(0.2)
            UsdPhysics.CollisionAPI.Apply(sphere.GetPrim())
            p1 = sphere.GetPrim()
            p1.AddAppliedSchema("NewtonSDFCollisionAPI")
            p1.CreateAttribute("newton:sdfMaxResolution", Sdf.ValueTypeNames.Int, custom=True).Set(32)
            p1.CreateAttribute("newton:hydroelasticEnabled", Sdf.ValueTypeNames.Bool, custom=True).Set(True)
            p1.CreateAttribute("newton:hydroelasticStiffness", Sdf.ValueTypeNames.Float, custom=True).Set(1e7)
            p1.CreateAttribute("newton:sdfPadding", Sdf.ValueTypeNames.Float, custom=True).Set(0.03)
            p1.CreateAttribute("newton:contactGap", Sdf.ValueTypeNames.Float, custom=True).Set(0.07)

            stage.Save()

            builder = newton.ModelBuilder()
            result = builder.add_usd(str(usd_path))
            s1 = result["path_shape_map"]["/World/Body1/CollisionSphere"]

            self.assertAlmostEqual(builder.shape_sdf_padding[s1], 0.03, places=5)
            self.assertAlmostEqual(builder.shape_gap[s1], 0.07, places=5)

    def test_usd_sdf_api_applied_no_authored_attrs(self, device=None):
        """Applying NewtonSDFCollisionAPI enables SDF generation with schema defaults."""
        if device is None or not wp.get_device(device).is_cuda:
            self.skipTest("SDF tests require CUDA device")

        from pxr import Usd, UsdPhysics

        with tempfile.TemporaryDirectory() as tmpdir:
            usd_path = Path(tmpdir) / "test_sdf_api_only.usda"
            stage = Usd.Stage.CreateNew(str(usd_path))
            UsdPhysics.Scene.Define(stage, "/PhysicsScene")

            _add_rigid_body(stage, "/World/Body1")
            m1 = _add_collision_mesh(stage, "/World/Body1/CollisionMesh")
            p1 = m1.GetPrim()
            # Apply the API without authoring any attributes. The importer
            # must enable SDF with schema defaults (sdfMaxResolution=64).
            p1.AddAppliedSchema("NewtonSDFCollisionAPI")

            stage.Save()

            builder = newton.ModelBuilder()
            result = builder.add_usd(str(usd_path))
            s1 = result["path_shape_map"]["/World/Body1/CollisionMesh"]

            self.assertEqual(builder.shape_sdf_max_resolution[s1], 64)

            model = builder.finalize(device=device)
            self.assertGreaterEqual(
                int(model._shape_sdf_index.numpy()[s1]),
                0,
                "Applied SDF API should land an SDF entry on the finalized model.",
            )

    def test_usd_mesh_invalid_sdf_max_resolution_warns_and_clears(self):
        """An sdfMaxResolution not divisible by 8 must warn and fall back to default rather than aborting the import."""
        from pxr import Sdf, Usd, UsdPhysics

        with tempfile.TemporaryDirectory() as tmpdir:
            usd_path = Path(tmpdir) / "test_mesh_invalid_sdf_res.usda"
            stage = Usd.Stage.CreateNew(str(usd_path))
            UsdPhysics.Scene.Define(stage, "/PhysicsScene")

            _add_rigid_body(stage, "/World/Body1")
            m1 = _add_collision_mesh(stage, "/World/Body1/CollisionMesh")
            p1 = m1.GetPrim()
            p1.CreateAttribute("newton:sdfMaxResolution", Sdf.ValueTypeNames.Int, custom=True).Set(63)

            stage.Save()

            builder = newton.ModelBuilder()
            with self.assertWarnsRegex(UserWarning, "must be divisible by 8"):
                result = builder.add_usd(str(usd_path))
            s1 = result["path_shape_map"]["/World/Body1/CollisionMesh"]
            # Invalid resolution should be dropped — builder default (None) wins.
            self.assertIsNone(builder.shape_sdf_max_resolution[s1])

    def test_usd_mesh_invalid_sdf_texture_format_warns_and_clears(self):
        """An unknown sdfTextureFormat must warn and fall back to default rather than aborting the import."""
        from pxr import Sdf, Usd, UsdPhysics

        with tempfile.TemporaryDirectory() as tmpdir:
            usd_path = Path(tmpdir) / "test_mesh_invalid_sdf_tex.usda"
            stage = Usd.Stage.CreateNew(str(usd_path))
            UsdPhysics.Scene.Define(stage, "/PhysicsScene")

            _add_rigid_body(stage, "/World/Body1")
            m1 = _add_collision_mesh(stage, "/World/Body1/CollisionMesh")
            p1 = m1.GetPrim()
            p1.CreateAttribute("newton:sdfMaxResolution", Sdf.ValueTypeNames.Int, custom=True).Set(64)
            p1.CreateAttribute("newton:sdfTextureFormat", Sdf.ValueTypeNames.Token, custom=True).Set("bogus")

            stage.Save()

            builder = newton.ModelBuilder()
            with self.assertWarnsRegex(UserWarning, "newton:sdfTextureFormat.*invalid"):
                result = builder.add_usd(str(usd_path))
            s1 = result["path_shape_map"]["/World/Body1/CollisionMesh"]
            # Bad texture format dropped — builder default ("uint16") wins.
            self.assertEqual(builder.shape_sdf_texture_format[s1], "uint16")

    def test_usd_mesh_both_sdf_resolution_and_voxel_size_warns(self):
        """Authoring both sdfMaxResolution and sdfTargetVoxelSize must warn; target voxel size takes precedence."""
        from pxr import Sdf, Usd, UsdPhysics

        with tempfile.TemporaryDirectory() as tmpdir:
            usd_path = Path(tmpdir) / "test_mesh_both_sdf_knobs.usda"
            stage = Usd.Stage.CreateNew(str(usd_path))
            UsdPhysics.Scene.Define(stage, "/PhysicsScene")

            _add_rigid_body(stage, "/World/Body1")
            m1 = _add_collision_mesh(stage, "/World/Body1/CollisionMesh")
            p1 = m1.GetPrim()
            p1.CreateAttribute("newton:sdfMaxResolution", Sdf.ValueTypeNames.Int, custom=True).Set(64)
            p1.CreateAttribute("newton:sdfTargetVoxelSize", Sdf.ValueTypeNames.Float, custom=True).Set(0.01)

            stage.Save()

            builder = newton.ModelBuilder()
            with self.assertWarnsRegex(UserWarning, "both.*sdfTargetVoxelSize.*sdfMaxResolution"):
                result = builder.add_usd(str(usd_path))
            s1 = result["path_shape_map"]["/World/Body1/CollisionMesh"]
            # Target voxel size wins; max_resolution is cleared.
            self.assertIsNone(builder.shape_sdf_max_resolution[s1])
            self.assertAlmostEqual(builder.shape_sdf_target_voxel_size[s1], 0.01, places=4)

    def test_deferred_sdf_distinguishes_shape_scales(self, device=None):
        """Two shapes sharing the same Mesh at different scales must produce distinct SDF entries."""
        if device is None or not wp.get_device(device).is_cuda:
            self.skipTest("SDF tests require CUDA device")

        mesh = newton.Mesh(
            vertices=[(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)],
            indices=[0, 1, 2, 0, 1, 3, 0, 2, 3, 1, 2, 3],
        )

        builder = newton.ModelBuilder()
        b0 = builder.add_body()
        s0 = builder.add_shape_mesh(b0, mesh=mesh, scale=(1.0, 1.0, 1.0))
        b1 = builder.add_body()
        s1 = builder.add_shape_mesh(b1, mesh=mesh, scale=(2.0, 1.0, 1.0))

        builder.shape_sdf_max_resolution[s0] = 32
        builder.shape_sdf_max_resolution[s1] = 32

        model = builder.finalize(device=device)
        idx0 = int(model._shape_sdf_index.numpy()[s0])
        idx1 = int(model._shape_sdf_index.numpy()[s1])

        self.assertGreaterEqual(idx0, 0)
        self.assertGreaterEqual(idx1, 0)
        self.assertNotEqual(idx0, idx1)

    def test_add_shape_convex_hull_rejects_hydroelastic_without_sdf(self):
        """add_shape_convex_hull must raise for hydroelastic convex meshes without mesh.sdf."""
        builder = newton.ModelBuilder()
        body = builder.add_body()
        # 4-vertex tetrahedron — minimum valid input for a convex hull
        mesh = newton.Mesh(
            vertices=[(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)],
            indices=[0, 1, 2, 0, 1, 3, 0, 2, 3, 1, 2, 3],
        )
        cfg = newton.ModelBuilder.ShapeConfig(is_hydroelastic=True)
        with self.assertRaisesRegex(ValueError, "Hydroelastic mesh-backed shapes require mesh.sdf"):
            builder.add_shape_convex_hull(body, mesh=mesh, cfg=cfg)

    def test_approximate_meshes_rejects_sdf_state(self):
        """approximate_meshes must raise on mesh-replacing methods when a shape carries deferred SDF or hydroelastic state."""
        builder = newton.ModelBuilder()
        body = builder.add_body()
        mesh = newton.Mesh(
            vertices=[(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)],
            indices=[0, 1, 2, 0, 1, 3, 0, 2, 3, 1, 2, 3],
        )
        shape_id = builder.add_shape_mesh(body, mesh=mesh)
        # Simulate the deferred-SDF state the USD importer writes directly to the builder.
        builder.shape_sdf_max_resolution[shape_id] = 64
        with self.assertRaisesRegex(ValueError, "SDF / hydroelastic configuration cannot be preserved"):
            builder.approximate_meshes(method="convex_hull", shape_indices=[shape_id])

    def test_usd_sdf_mesh_uses_simplified_collision_edges(self, device=None):
        """Deferred-built SDF on a USD mesh must surface its simplified collision edges to finalize."""
        if device is None or not wp.get_device(device).is_cuda:
            self.skipTest("SDF tests require CUDA device")

        from pxr import Sdf, Usd, UsdPhysics

        with tempfile.TemporaryDirectory() as tmpdir:
            usd_path = Path(tmpdir) / "test_sdf_edges.usda"
            stage = Usd.Stage.CreateNew(str(usd_path))
            UsdPhysics.Scene.Define(stage, "/PhysicsScene")
            _add_rigid_body(stage, "/World/Body1")
            m1 = _add_collision_mesh(stage, "/World/Body1/CollisionMesh")
            m1.GetPrim().CreateAttribute("newton:sdfMaxResolution", Sdf.ValueTypeNames.Int, custom=True).Set(64)
            stage.Save()

            builder = newton.ModelBuilder()
            result = builder.add_usd(str(usd_path))
            s1 = result["path_shape_map"]["/World/Body1/CollisionMesh"]
            # shape_source still has no collision edges before finalize (the
            # deferred build runs inside finalize on a Mesh clone).
            self.assertIsNone(builder.shape_source[s1]._collision_edges)
            full_edge_count = len(builder.shape_source[s1].edges)

            model = builder.finalize(device=device)
            # The simplified edges land in model.shape_edge_range / model.mesh_edge_indices
            # (rigid collision edges), not model.edge_count (cloth bending edges).
            _, simplified_edge_count = model.shape_edge_range.numpy()[s1]
            simplified_edge_count = int(simplified_edge_count)
            # The simplified set is strictly smaller than the full edge list on
            # a typical cube (default 0.1° threshold drops co-planar edges).
            self.assertLess(
                simplified_edge_count,
                full_edge_count,
                f"Expected simplified edges < full edges, got {simplified_edge_count} >= {full_edge_count}",
            )

    def test_usd_sdf_with_physics_approximation_warns_and_ignores(self):
        """physics:approximation on an SDF prim is ignored at parse time with a warning; SDF configuration survives."""
        from pxr import Sdf, Usd, UsdPhysics

        with tempfile.TemporaryDirectory() as tmpdir:
            usd_path = Path(tmpdir) / "test_sdf_with_approximation.usda"
            stage = Usd.Stage.CreateNew(str(usd_path))
            UsdPhysics.Scene.Define(stage, "/PhysicsScene")

            _add_rigid_body(stage, "/World/Body1")
            m1 = _add_collision_mesh(stage, "/World/Body1/CollisionMesh")
            p1 = m1.GetPrim()
            p1.AddAppliedSchema("NewtonSDFCollisionAPI")
            p1.CreateAttribute("newton:sdfMaxResolution", Sdf.ValueTypeNames.Int, custom=True).Set(64)
            p1.CreateAttribute("newton:hydroelasticEnabled", Sdf.ValueTypeNames.Bool, custom=True).Set(True)
            p1.CreateAttribute("physics:approximation", Sdf.ValueTypeNames.Token, custom=True).Set("convexHull")

            stage.Save()

            builder = newton.ModelBuilder()
            with self.assertWarnsRegex(UserWarning, "physics:approximation.*ignored"):
                result = builder.add_usd(str(usd_path))
            s1 = result["path_shape_map"]["/World/Body1/CollisionMesh"]
            # SDF configuration must survive the ignored approximation.
            self.assertEqual(builder.shape_sdf_max_resolution[s1], 64)
            self.assertTrue(builder.shape_flags[s1] & newton.ShapeFlags.HYDROELASTIC)

    def test_usd_sibling_collision_apis_warn_and_sdf_wins(self):
        """Co-applying NewtonSDFCollisionAPI and NewtonMeshCollisionAPI emits a warning and uses SDF configuration."""
        from pxr import Sdf, Usd, UsdPhysics

        with tempfile.TemporaryDirectory() as tmpdir:
            usd_path = Path(tmpdir) / "test_sibling_apis.usda"
            stage = Usd.Stage.CreateNew(str(usd_path))
            UsdPhysics.Scene.Define(stage, "/PhysicsScene")

            _add_rigid_body(stage, "/World/Body1")
            m1 = _add_collision_mesh(stage, "/World/Body1/CollisionMesh")
            p1 = m1.GetPrim()
            p1.AddAppliedSchema("NewtonSDFCollisionAPI")
            p1.AddAppliedSchema("NewtonMeshCollisionAPI")
            p1.CreateAttribute("newton:sdfMaxResolution", Sdf.ValueTypeNames.Int, custom=True).Set(64)

            stage.Save()

            builder = newton.ModelBuilder()
            with self.assertWarnsRegex(UserWarning, "independent collision representations"):
                result = builder.add_usd(str(usd_path))
            s1 = result["path_shape_map"]["/World/Body1/CollisionMesh"]
            self.assertEqual(builder.shape_sdf_max_resolution[s1], 64)

    def test_usd_sdf_api_applied_hydroelastic_schema_default_wins(self, device=None):
        """When NewtonSDFCollisionAPI is applied and hydroelasticEnabled is unauthored, the schema default (False) wins over a True builder default."""
        if device is None or not wp.get_device(device).is_cuda:
            self.skipTest("SDF tests require CUDA device")

        from pxr import Usd, UsdGeom, UsdPhysics

        with tempfile.TemporaryDirectory() as tmpdir:
            usd_path = Path(tmpdir) / "test_hydro_schema_default.usda"
            stage = Usd.Stage.CreateNew(str(usd_path))
            UsdPhysics.Scene.Define(stage, "/PhysicsScene")

            _add_rigid_body(stage, "/World/Body1")
            sphere = UsdGeom.Sphere.Define(stage, "/World/Body1/CollisionSphere")
            sphere.CreateRadiusAttr(0.2)
            UsdPhysics.CollisionAPI.Apply(sphere.GetPrim())
            p1 = sphere.GetPrim()
            p1.AddAppliedSchema("NewtonSDFCollisionAPI")

            stage.Save()

            builder = newton.ModelBuilder()
            builder.default_shape_cfg.is_hydroelastic = True
            result = builder.add_usd(str(usd_path))
            s1 = result["path_shape_map"]["/World/Body1/CollisionSphere"]

            self.assertFalse(builder.shape_flags[s1] & newton.ShapeFlags.HYDROELASTIC)

    def test_usd_sdf_api_applied_target_voxel_size_only(self):
        """Authoring only newton:sdfTargetVoxelSize must not also inject the API default for sdfMaxResolution."""
        from pxr import Sdf, Usd, UsdGeom, UsdPhysics

        with tempfile.TemporaryDirectory() as tmpdir:
            usd_path = Path(tmpdir) / "test_target_voxel_only.usda"
            stage = Usd.Stage.CreateNew(str(usd_path))
            UsdPhysics.Scene.Define(stage, "/PhysicsScene")

            _add_rigid_body(stage, "/World/Body1")
            sphere = UsdGeom.Sphere.Define(stage, "/World/Body1/CollisionSphere")
            sphere.CreateRadiusAttr(0.2)
            UsdPhysics.CollisionAPI.Apply(sphere.GetPrim())
            p1 = sphere.GetPrim()
            p1.AddAppliedSchema("NewtonSDFCollisionAPI")
            p1.CreateAttribute("newton:sdfTargetVoxelSize", Sdf.ValueTypeNames.Float, custom=True).Set(0.01)

            stage.Save()

            builder = newton.ModelBuilder()
            # Must not raise: target_voxel_size and max_resolution are mutually exclusive.
            result = builder.add_usd(str(usd_path))
            s1 = result["path_shape_map"]["/World/Body1/CollisionSphere"]

            self.assertIsNone(builder.shape_sdf_max_resolution[s1])
            self.assertAlmostEqual(builder.shape_sdf_target_voxel_size[s1], 0.01, places=5)

    def test_usd_sdf_api_applied_no_hydroelastic_by_default(self, device=None):
        """Applying NewtonSDFCollisionAPI without authoring hydroelasticEnabled leaves hydro OFF."""
        if device is None or not wp.get_device(device).is_cuda:
            self.skipTest("SDF tests require CUDA device")

        from pxr import Usd, UsdGeom, UsdPhysics

        with tempfile.TemporaryDirectory() as tmpdir:
            usd_path = Path(tmpdir) / "test_sdf_api_no_hydro.usda"
            stage = Usd.Stage.CreateNew(str(usd_path))
            UsdPhysics.Scene.Define(stage, "/PhysicsScene")

            _add_rigid_body(stage, "/World/Body1")
            sphere = UsdGeom.Sphere.Define(stage, "/World/Body1/CollisionSphere")
            sphere.CreateRadiusAttr(0.2)
            UsdPhysics.CollisionAPI.Apply(sphere.GetPrim())
            p1 = sphere.GetPrim()
            # SDF API applied, but hydroelasticEnabled is not authored.
            # Schema default is false, so hydroelastic stays off.
            p1.AddAppliedSchema("NewtonSDFCollisionAPI")

            stage.Save()

            builder = newton.ModelBuilder()
            result = builder.add_usd(str(usd_path))
            s1 = result["path_shape_map"]["/World/Body1/CollisionSphere"]

            self.assertFalse(builder.shape_flags[s1] & newton.ShapeFlags.HYDROELASTIC)

    def test_usd_kh_alone_does_not_enable_hydroelastic(self, device=None):
        """Authoring newton:hydroelasticStiffness without hydroelasticEnabled=true must not turn hydro on."""
        if device is None or not wp.get_device(device).is_cuda:
            self.skipTest("SDF tests require CUDA device")

        from pxr import Sdf, Usd, UsdGeom, UsdPhysics

        with tempfile.TemporaryDirectory() as tmpdir:
            usd_path = Path(tmpdir) / "test_kh_alone.usda"
            stage = Usd.Stage.CreateNew(str(usd_path))
            UsdPhysics.Scene.Define(stage, "/PhysicsScene")

            _add_rigid_body(stage, "/World/Body1")
            sphere = UsdGeom.Sphere.Define(stage, "/World/Body1/CollisionSphere")
            sphere.CreateRadiusAttr(0.2)
            UsdPhysics.CollisionAPI.Apply(sphere.GetPrim())
            p1 = sphere.GetPrim()
            # kh is just a material parameter. Without hydroelasticEnabled=true,
            # the importer must not flip hydro on.
            p1.CreateAttribute("newton:hydroelasticStiffness", Sdf.ValueTypeNames.Float, custom=True).Set(1e7)

            stage.Save()

            builder = newton.ModelBuilder()
            result = builder.add_usd(str(usd_path))
            s1 = result["path_shape_map"]["/World/Body1/CollisionSphere"]

            self.assertFalse(builder.shape_flags[s1] & newton.ShapeFlags.HYDROELASTIC)

    def test_usd_hydroelastic_mesh_without_sdf_config_warns_and_disables(self, device=None):
        """Hydroelastic mesh without SDF resolution or voxel size warns and falls back to plain mesh collider."""
        del device  # validation is host-side and independent of device

        from pxr import Sdf, Usd, UsdPhysics

        with tempfile.TemporaryDirectory() as tmpdir:
            usd_path = Path(tmpdir) / "test_hydro_mesh_invalid.usda"
            stage = Usd.Stage.CreateNew(str(usd_path))
            UsdPhysics.Scene.Define(stage, "/PhysicsScene")

            _add_rigid_body(stage, "/World/Body1")
            m1 = _add_collision_mesh(stage, "/World/Body1/CollisionMesh")
            p1 = m1.GetPrim()
            # hydroelasticEnabled=true opts into hydro explicitly, but the API is
            # not applied so the importer does not fill in a default
            # sdfMaxResolution. Importer should warn and disable hydro on this shape.
            p1.CreateAttribute("newton:hydroelasticEnabled", Sdf.ValueTypeNames.Bool, custom=True).Set(True)

            stage.Save()

            builder = newton.ModelBuilder()
            with self.assertWarnsRegex(UserWarning, "hydroelastic mesh requires"):
                result = builder.add_usd(str(usd_path))
            s1 = result["path_shape_map"]["/World/Body1/CollisionMesh"]
            self.assertFalse(builder.shape_flags[s1] & newton.ShapeFlags.HYDROELASTIC)

    def test_usd_hydroelastic_mesh_with_kh_without_sdf_config_warns_and_disables(self, device=None):
        """Authoring newton:hydroelasticStiffness must not bypass the hydroelastic-mesh SDF-source check."""
        del device  # validation is host-side and independent of device

        from pxr import Sdf, Usd, UsdPhysics

        with tempfile.TemporaryDirectory() as tmpdir:
            usd_path = Path(tmpdir) / "test_hydro_mesh_kh_invalid.usda"
            stage = Usd.Stage.CreateNew(str(usd_path))
            UsdPhysics.Scene.Define(stage, "/PhysicsScene")

            _add_rigid_body(stage, "/World/Body1")
            m1 = _add_collision_mesh(stage, "/World/Body1/CollisionMesh")
            p1 = m1.GetPrim()
            p1.CreateAttribute("newton:hydroelasticEnabled", Sdf.ValueTypeNames.Bool, custom=True).Set(True)
            p1.CreateAttribute("newton:hydroelasticStiffness", Sdf.ValueTypeNames.Float, custom=True).Set(1e7)

            stage.Save()

            builder = newton.ModelBuilder()
            with self.assertWarnsRegex(UserWarning, "hydroelastic mesh requires"):
                result = builder.add_usd(str(usd_path))
            s1 = result["path_shape_map"]["/World/Body1/CollisionMesh"]
            self.assertFalse(builder.shape_flags[s1] & newton.ShapeFlags.HYDROELASTIC)


devices = get_selected_cuda_test_devices()
add_function_test(
    TestSDFUSDParsing, "test_usd_sdf_mesh_attributes", TestSDFUSDParsing.test_usd_sdf_mesh_attributes, devices=devices
)
add_function_test(
    TestSDFUSDParsing,
    "test_usd_sdf_mesh_uses_simplified_collision_edges",
    TestSDFUSDParsing.test_usd_sdf_mesh_uses_simplified_collision_edges,
    devices=devices,
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
    "test_usd_sdf_padding",
    TestSDFUSDParsing.test_usd_sdf_padding,
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
    "test_usd_sdf_padding_hydroelastic_primitive",
    TestSDFUSDParsing.test_usd_sdf_padding_hydroelastic_primitive,
    devices=devices,
)
add_function_test(
    TestSDFUSDParsing,
    "test_usd_sdf_padding_does_not_affect_shape_gap",
    TestSDFUSDParsing.test_usd_sdf_padding_does_not_affect_shape_gap,
    devices=devices,
)
add_function_test(
    TestSDFUSDParsing,
    "test_usd_sdf_api_applied_no_authored_attrs",
    TestSDFUSDParsing.test_usd_sdf_api_applied_no_authored_attrs,
    devices=devices,
)
add_function_test(
    TestSDFUSDParsing,
    "test_usd_sdf_api_applied_no_hydroelastic_by_default",
    TestSDFUSDParsing.test_usd_sdf_api_applied_no_hydroelastic_by_default,
    devices=devices,
)
add_function_test(
    TestSDFUSDParsing,
    "test_usd_sdf_api_applied_hydroelastic_schema_default_wins",
    TestSDFUSDParsing.test_usd_sdf_api_applied_hydroelastic_schema_default_wins,
    devices=devices,
)
add_function_test(
    TestSDFUSDParsing,
    "test_usd_kh_alone_does_not_enable_hydroelastic",
    TestSDFUSDParsing.test_usd_kh_alone_does_not_enable_hydroelastic,
    devices=devices,
)
add_function_test(
    TestSDFUSDParsing,
    "test_deferred_sdf_distinguishes_shape_scales",
    TestSDFUSDParsing.test_deferred_sdf_distinguishes_shape_scales,
    devices=devices,
)
# The hydroelastic-mesh warn-and-degrade tests run host-side (no wp.launch / CUDA),
# so they're registered as plain unittest methods.


if __name__ == "__main__":
    wp.clear_kernel_cache()
    unittest.main(verbosity=2)
