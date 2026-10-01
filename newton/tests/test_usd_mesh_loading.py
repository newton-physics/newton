# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Tests for USD mesh extraction from stage, path, URL, and prim sources."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import warp as wp

import newton
import newton.usd
from newton.sensors import SensorTiledCamera
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


def _define_folded_mesh(stage, path="/Fold", scheme=None):
    """Define two perpendicular triangles sharing an edge."""
    from pxr import UsdGeom

    mesh = UsdGeom.Mesh.Define(stage, path)
    mesh.CreatePointsAttr([(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)])
    mesh.CreateFaceVertexCountsAttr([3, 3])
    mesh.CreateFaceVertexIndicesAttr([0, 1, 2, 0, 3, 1])
    if scheme is not None:
        mesh.CreateSubdivisionSchemeAttr(scheme)
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

    def test_get_mesh_canonicalizes_shading(self):
        """Represent sharp and smooth USD shading with ordinary vertex normals."""
        from pxr import Usd, UsdGeom

        for scheme in (None, "none", "bilinear", "catmullClark", "loop"):
            with self.subTest(scheme=scheme):
                stage = Usd.Stage.CreateInMemory()
                source = _define_folded_mesh(stage, scheme=scheme)
                mesh = newton.usd.get_mesh(source.GetPrim(), load_normals=True, compute_inertia=False)
                self.assertIsNotNone(mesh.normals)
                corners = mesh.normals[mesh.indices].reshape(2, 3, 3)
                if scheme in (UsdGeom.Tokens.none, UsdGeom.Tokens.bilinear):
                    np.testing.assert_allclose(corners, np.repeat([[(0, 0, 1)], [(0, 1, 0)]], 3, axis=1))
                else:
                    np.testing.assert_allclose(corners[:, 0], np.tile([0, 2**-0.5, 2**-0.5], (2, 1)), atol=1e-6)
                np.testing.assert_array_equal(mesh.copy().normals, mesh.normals)

                geometry = newton.usd.get_mesh(source.GetPrim(), load_normals=False, compute_inertia=False)
                self.assertEqual(len(geometry.vertices), 4)
                self.assertIsNone(geometry.normals)

    def test_get_mesh_preserves_mixed_shading_when_merged(self):
        """Keep sharp and smooth components intact when merging USD meshes."""
        from pxr import Usd

        stage = Usd.Stage.CreateInMemory()
        sources = [
            _define_folded_mesh(stage, f"/mesh_{scheme}", scheme) for scheme in ("none", "bilinear", "catmullClark")
        ]
        sources[0].CreateNormalsAttr([(1, 0, 0)] * 4)
        sources[0].SetNormalsInterpolation("vertex")
        meshes = [newton.usd.get_mesh(source.GetPrim(), load_normals=True, compute_inertia=False) for source in sources]
        merged = newton.usd.get_mesh(stage, load_normals=True, compute_inertia=False)
        self.assertIsNotNone(merged.normals)
        np.testing.assert_allclose(
            merged.normals[merged.indices], np.concatenate([mesh.normals[mesh.indices] for mesh in meshes])
        )
        np.testing.assert_allclose(
            merged.vertices[merged.indices], np.concatenate([mesh.vertices[mesh.indices] for mesh in meshes])
        )

    def test_uniform_normals_honor_conversion_policy(self):
        """Honor both averaging policies and the requested splitting threshold."""
        from pxr import Usd

        stage = Usd.Stage.CreateInMemory()
        source = _define_folded_mesh(stage, scheme="none")
        source.CreateNormalsAttr([(0, 0, 1), (0, 1, 0)])
        source.SetNormalsInterpolation("uniform")
        for policy, threshold, vertices in (
            ("vertex_averaging", 0, 4),
            ("angle_weighted", 0, 4),
            ("vertex_splitting", 0, 6),
            ("vertex_splitting", 100, 4),
        ):
            with self.subTest(policy=policy, threshold=threshold):
                mesh = newton.usd.get_mesh(
                    source.GetPrim(),
                    load_normals=True,
                    compute_inertia=False,
                    face_varying_normal_conversion=policy,
                    vertex_splitting_angle_threshold_deg=threshold,
                )
                self.assertEqual(len(mesh.vertices), vertices)

    def test_sensor_renders_resolved_shading(self):
        """Render sharp and smooth imported normals through the camera sensor."""
        from pxr import Usd

        stage = Usd.Stage.CreateInMemory()
        source = _define_folded_mesh(stage)
        for scheme in ("none", "catmullClark"):
            with self.subTest(scheme=scheme), wp.ScopedDevice("cpu"):
                source.CreateSubdivisionSchemeAttr(scheme)
                mesh = newton.usd.get_mesh(source.GetPrim(), load_normals=True, compute_inertia=False)
                builder = newton.ModelBuilder()
                builder.add_shape_mesh(-1, mesh=mesh)
                model = builder.finalize(device="cpu")
                sensor = SensorTiledCamera(model=model)
                transforms = wp.array(
                    [[wp.transform(wp.vec3(0.25, 0.25, 2.0), wp.quat_identity())]], dtype=wp.transform
                )
                rays = sensor.utils.compute_camera_rays_pinhole(1, 1, camera_fovs=0.5)
                image = sensor.utils.create_normal_image_output(1, 1)
                sensor.update(model.state(), transforms, rays, normal_image=image)
                normal = image.numpy()[0, 0, 0, 0]
                if scheme == "none":
                    np.testing.assert_allclose(normal, [0, 0, 1], atol=1e-6)
                else:
                    expected = 0.75 * np.array([0, 2**-0.5, 2**-0.5]) + 0.25 * np.array([0, 0, 1])
                    expected /= np.linalg.norm(expected)
                    np.testing.assert_allclose(normal, expected, atol=1e-6)

    def test_generated_normals_preserve_uv_seams(self):
        """Generate shading before UV expansion and keep corner UVs aligned."""
        from pxr import Sdf, Usd, UsdGeom

        stage = Usd.Stage.CreateInMemory()
        source = _define_folded_mesh(stage)
        corner_uvs = np.array([(0, 0), (1, 0), (0, 1), (0.2, 0), (0.2, 1), (1, 0)], dtype=np.float32)
        UsdGeom.PrimvarsAPI(source).CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray, "faceVarying").Set(
            corner_uvs
        )
        for scheme in ("none", "catmullClark"):
            source.CreateSubdivisionSchemeAttr(scheme)
            for preserve in (False, True):
                with self.subTest(scheme=scheme, preserve=preserve):
                    mesh, uv_indices = newton.usd.get_mesh(
                        source.GetPrim(),
                        load_normals=True,
                        load_uvs=True,
                        preserve_facevarying_uvs=preserve,
                        return_uv_indices=True,
                        compute_inertia=False,
                    )
                    np.testing.assert_allclose(mesh.uvs[uv_indices], corner_uvs)
                    if scheme == "catmullClark":
                        np.testing.assert_allclose(
                            mesh.normals[mesh.indices[[0, 3]]], np.tile([0, 2**-0.5, 2**-0.5], (2, 1)), atol=1e-6
                        )

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
