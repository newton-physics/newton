# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Tests for USD mesh extraction from stage, path, URL, and prim sources."""

import tempfile
import unittest
from pathlib import Path

import numpy as np

import newton
import newton.usd
from newton.tests.unittest_utils import USD_AVAILABLE, assert_np_equal


def _create_referenced_mesh_stage(tmpdir: str) -> Path:
    """Create a USD stage with a referenced translated triangle mesh."""
    from pxr import Gf, Usd, UsdGeom

    asset_path = Path(tmpdir) / "asset.usda"
    asset_stage = Usd.Stage.CreateNew(str(asset_path))
    mesh = UsdGeom.Mesh.Define(asset_stage, "/Asset/Triangle")
    UsdGeom.Xformable(mesh.GetPrim()).AddTranslateOp().Set(Gf.Vec3d(1.0, 2.0, 3.0))
    mesh.CreatePointsAttr(
        [
            Gf.Vec3f(0.0, 0.0, 0.0),
            Gf.Vec3f(1.0, 0.0, 0.0),
            Gf.Vec3f(0.0, 1.0, 0.0),
        ]
    )
    mesh.CreateFaceVertexCountsAttr([3])
    mesh.CreateFaceVertexIndicesAttr([0, 1, 2])
    asset_stage.GetRootLayer().Save()

    stage_path = Path(tmpdir) / "marker.usda"
    stage = Usd.Stage.CreateNew(str(stage_path))
    marker = UsdGeom.Xform.Define(stage, "/Marker")
    marker.GetPrim().GetReferences().AddReference("./asset.usda", "/Asset")
    stage.GetRootLayer().Save()
    return stage_path


def _define_triangle_mesh(stage, path="/Triangle"):
    """Define a simple triangle mesh prim on a USD stage."""
    from pxr import Gf, UsdGeom

    mesh = UsdGeom.Mesh.Define(stage, path)
    mesh.CreatePointsAttr(
        [
            Gf.Vec3f(0.0, 0.0, 0.0),
            Gf.Vec3f(1.0, 0.0, 0.0),
            Gf.Vec3f(0.0, 1.0, 0.0),
        ]
    )
    mesh.CreateFaceVertexCountsAttr([3])
    mesh.CreateFaceVertexIndicesAttr([0, 1, 2])
    return mesh


@unittest.skipUnless(USD_AVAILABLE, "Requires usd-core")
class TestUsdMeshHelpers(unittest.TestCase):
    """Tests for loading Newton meshes from USD source variants."""

    def test_get_mesh_accepts_usd_file_with_reference(self):
        """Load a mesh from a USD file containing a referenced asset."""
        with tempfile.TemporaryDirectory() as tmpdir:
            stage_path = _create_referenced_mesh_stage(tmpdir)

            mesh = newton.usd.get_mesh(stage_path, root_path="/Marker", compute_inertia=False)

        self.assertIsInstance(mesh, newton.Mesh)
        assert_np_equal(
            mesh.vertices,
            np.array(
                [
                    [1.0, 2.0, 3.0],
                    [2.0, 2.0, 3.0],
                    [1.0, 3.0, 3.0],
                ],
                dtype=np.float32,
            ),
        )
        assert_np_equal(mesh.indices, np.array([0, 1, 2], dtype=np.int32))

    def test_get_mesh_accepts_usd_stage_handle(self):
        """Load a mesh from an already-open USD stage handle."""
        from pxr import Usd

        with tempfile.TemporaryDirectory() as tmpdir:
            stage_path = _create_referenced_mesh_stage(tmpdir)
            stage = Usd.Stage.Open(str(stage_path), Usd.Stage.LoadAll)

            mesh = newton.usd.get_mesh(stage, root_path="/Marker", compute_inertia=False)

        self.assertIsInstance(mesh, newton.Mesh)
        self.assertEqual(len(mesh.vertices), 3)
        self.assertEqual(len(mesh.indices), 3)

    def test_get_mesh_rejects_http_urls(self):
        """Reject cleartext HTTP USD asset URLs."""
        with self.assertRaisesRegex(ValueError, "HTTP USD URLs are not supported"):
            newton.usd.get_mesh("http://example.com/marker.usda", compute_inertia=False)

    def test_get_mesh_prim_source_keeps_authored_units(self):
        """Keep authored coordinates when loading a single mesh prim."""
        from pxr import Gf, Usd, UsdGeom

        stage = Usd.Stage.CreateInMemory()
        UsdGeom.SetStageMetersPerUnit(stage, 0.01)
        mesh = UsdGeom.Mesh.Define(stage, "/Triangle")
        mesh.CreatePointsAttr(
            [
                Gf.Vec3f(0.0, 0.0, 0.0),
                Gf.Vec3f(100.0, 0.0, 0.0),
                Gf.Vec3f(0.0, 100.0, 0.0),
            ]
        )
        mesh.CreateFaceVertexCountsAttr([3])
        mesh.CreateFaceVertexIndicesAttr([0, 1, 2])

        result = newton.usd.get_mesh(mesh.GetPrim(), compute_inertia=False)

        assert_np_equal(result.vertices[1], np.array([100.0, 0.0, 0.0], dtype=np.float32))

    def test_get_mesh_accepts_legacy_prim_keyword(self):
        """Keep ``prim=`` working for compatibility with the existing API."""
        from pxr import Usd

        stage = Usd.Stage.CreateInMemory()
        mesh_prim = _define_triangle_mesh(stage).GetPrim()

        mesh = newton.usd.get_mesh(prim=mesh_prim, compute_inertia=False)

        self.assertIsInstance(mesh, newton.Mesh)
        assert_np_equal(mesh.indices, np.array([0, 1, 2], dtype=np.int32))

    def test_mesh_create_from_usd_accepts_legacy_prim_keyword(self):
        """Keep ``Mesh.create_from_usd(prim=...)`` working."""
        from pxr import Usd

        stage = Usd.Stage.CreateInMemory()
        mesh_prim = _define_triangle_mesh(stage).GetPrim()

        mesh = newton.Mesh.create_from_usd(prim=mesh_prim, compute_inertia=False)

        self.assertIsInstance(mesh, newton.Mesh)
        assert_np_equal(mesh.vertices[1], np.array([1.0, 0.0, 0.0], dtype=np.float32))

    def test_get_mesh_rejects_source_and_legacy_prim_keyword(self):
        """Reject ambiguous calls that provide both source names."""
        from pxr import Usd

        stage = Usd.Stage.CreateInMemory()
        mesh_prim = _define_triangle_mesh(stage).GetPrim()

        with self.assertRaisesRegex(TypeError, "received both 'source' and legacy 'prim'"):
            newton.usd.get_mesh(mesh_prim, prim=mesh_prim, compute_inertia=False)

    def test_mesh_create_from_usd_rejects_source_and_legacy_prim_keyword(self):
        """Reject ambiguous factory calls that provide both source names."""
        from pxr import Usd

        stage = Usd.Stage.CreateInMemory()
        mesh_prim = _define_triangle_mesh(stage).GetPrim()

        with self.assertRaisesRegex(TypeError, "received both 'source' and legacy 'prim'"):
            newton.Mesh.create_from_usd(mesh_prim, prim=mesh_prim, compute_inertia=False)

    def test_get_mesh_merges_multiple_mesh_prims(self):
        """Merge multiple mesh prims under a selected root."""
        from pxr import Gf, Usd, UsdGeom

        with tempfile.TemporaryDirectory() as tmpdir:
            stage_path = Path(tmpdir) / "multi.usda"
            stage = Usd.Stage.CreateNew(str(stage_path))
            UsdGeom.Xform.Define(stage, "/Root")
            for name, tx in (("A", 0.0), ("B", 2.0)):
                mesh = UsdGeom.Mesh.Define(stage, f"/Root/{name}")
                UsdGeom.Xformable(mesh.GetPrim()).AddTranslateOp().Set(Gf.Vec3d(tx, 0.0, 0.0))
                mesh.CreatePointsAttr(
                    [
                        Gf.Vec3f(0.0, 0.0, 0.0),
                        Gf.Vec3f(1.0, 0.0, 0.0),
                        Gf.Vec3f(0.0, 1.0, 0.0),
                    ]
                )
                mesh.CreateFaceVertexCountsAttr([3])
                mesh.CreateFaceVertexIndicesAttr([0, 1, 2])
            stage.GetRootLayer().Save()

            mesh = newton.usd.get_mesh(stage_path, root_path="/Root", compute_inertia=False)

        self.assertEqual(len(mesh.vertices), 6)
        assert_np_equal(mesh.indices, np.array([0, 1, 2, 3, 4, 5], dtype=np.int32))
        assert_np_equal(mesh.vertices[3:], np.array([[2.0, 0.0, 0.0], [3.0, 0.0, 0.0], [2.0, 1.0, 0.0]]))

    def test_get_mesh_rejects_preserved_facevarying_uvs_for_merged_sources(self):
        """Reject merged-source loads that request face-varying UV preservation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            stage_path = _create_referenced_mesh_stage(tmpdir)

            with self.assertRaisesRegex(ValueError, "preserve_facevarying_uvs is not supported"):
                newton.usd.get_mesh(
                    stage_path,
                    root_path="/Marker",
                    preserve_facevarying_uvs=True,
                    compute_inertia=False,
                )

    def test_get_mesh_applies_root_relative_transform_and_stage_units(self):
        """Apply root-relative transforms and authored stage units."""
        from pxr import Gf, Usd, UsdGeom

        with tempfile.TemporaryDirectory() as tmpdir:
            stage_path = Path(tmpdir) / "units.usda"
            stage = Usd.Stage.CreateNew(str(stage_path))
            UsdGeom.SetStageMetersPerUnit(stage, 0.01)
            root = UsdGeom.Xform.Define(stage, "/Root")
            root.AddTranslateOp().Set(Gf.Vec3d(1000.0, 0.0, 0.0))
            mesh = UsdGeom.Mesh.Define(stage, "/Root/Triangle")
            UsdGeom.Xformable(mesh.GetPrim()).AddTranslateOp().Set(Gf.Vec3d(100.0, 0.0, 0.0))
            mesh.CreatePointsAttr(
                [
                    Gf.Vec3f(0.0, 0.0, 0.0),
                    Gf.Vec3f(100.0, 0.0, 0.0),
                    Gf.Vec3f(0.0, 100.0, 0.0),
                ]
            )
            mesh.CreateFaceVertexCountsAttr([3])
            mesh.CreateFaceVertexIndicesAttr([0, 1, 2])
            stage.GetRootLayer().Save()

            mesh = newton.usd.get_mesh(stage_path, root_path="/Root", compute_inertia=False)

        assert_np_equal(
            mesh.vertices,
            np.array([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [1.0, 1.0, 0.0]], dtype=np.float32),
        )

    def test_get_mesh_flips_winding_for_negative_scale(self):
        """Flip triangle winding when a merged transform mirrors handedness."""
        from pxr import Gf, Usd, UsdGeom

        with tempfile.TemporaryDirectory() as tmpdir:
            stage_path = Path(tmpdir) / "mirror.usda"
            stage = Usd.Stage.CreateNew(str(stage_path))
            UsdGeom.Xform.Define(stage, "/Root")
            mesh = UsdGeom.Mesh.Define(stage, "/Root/Triangle")
            UsdGeom.Xformable(mesh.GetPrim()).AddScaleOp().Set(Gf.Vec3d(-1.0, 1.0, 1.0))
            mesh.CreatePointsAttr(
                [
                    Gf.Vec3f(0.0, 0.0, 0.0),
                    Gf.Vec3f(1.0, 0.0, 0.0),
                    Gf.Vec3f(0.0, 1.0, 0.0),
                ]
            )
            mesh.CreateFaceVertexCountsAttr([3])
            mesh.CreateFaceVertexIndicesAttr([0, 1, 2])
            stage.GetRootLayer().Save()

            mesh = newton.usd.get_mesh(stage_path, root_path="/Root", compute_inertia=False)

        assert_np_equal(mesh.indices, np.array([0, 2, 1], dtype=np.int32))
        assert_np_equal(mesh.vertices[1], np.array([-1.0, 0.0, 0.0], dtype=np.float32))

    def test_get_mesh_transforms_normals_with_rotation(self):
        """Transform authored normals with the same row-vector convention as points."""
        from pxr import Gf, Usd, UsdGeom

        with tempfile.TemporaryDirectory() as tmpdir:
            stage_path = Path(tmpdir) / "normals.usda"
            stage = Usd.Stage.CreateNew(str(stage_path))
            UsdGeom.Xform.Define(stage, "/Root")
            mesh = UsdGeom.Mesh.Define(stage, "/Root/Triangle")
            UsdGeom.Xformable(mesh.GetPrim()).AddRotateZOp().Set(90.0)
            mesh.CreatePointsAttr(
                [
                    Gf.Vec3f(0.0, 0.0, 0.0),
                    Gf.Vec3f(1.0, 0.0, 0.0),
                    Gf.Vec3f(0.0, 1.0, 0.0),
                ]
            )
            mesh.CreateFaceVertexCountsAttr([3])
            mesh.CreateFaceVertexIndicesAttr([0, 1, 2])
            mesh.CreateNormalsAttr([Gf.Vec3f(1.0, 0.0, 0.0)] * 3)
            mesh.SetNormalsInterpolation(UsdGeom.Tokens.vertex)
            stage.GetRootLayer().Save()

            mesh = newton.usd.get_mesh(stage_path, root_path="/Root", load_normals=True, compute_inertia=False)

        self.assertIsNotNone(mesh.normals)
        np.testing.assert_allclose(mesh.normals[0], np.array([0.0, 1.0, 0.0], dtype=np.float32), atol=1e-6)


if __name__ == "__main__":
    unittest.main(verbosity=2)
