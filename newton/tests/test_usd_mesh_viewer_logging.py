# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import warp as wp

import newton
import newton.usd
from newton.tests.unittest_utils import USD_AVAILABLE, assert_np_equal
from newton.viewer import ViewerNull


def _create_referenced_mesh_stage(tmpdir: str) -> Path:
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


@unittest.skipUnless(USD_AVAILABLE, "Requires usd-core")
class TestUsdMeshHelpers(unittest.TestCase):
    def test_get_mesh_accepts_usd_file_with_reference(self):
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
        from pxr import Usd

        with tempfile.TemporaryDirectory() as tmpdir:
            stage_path = _create_referenced_mesh_stage(tmpdir)
            stage = Usd.Stage.Open(str(stage_path), Usd.Stage.LoadAll)

            mesh = newton.usd.get_mesh(stage, root_path="/Marker", compute_inertia=False)

        self.assertIsInstance(mesh, newton.Mesh)
        self.assertEqual(len(mesh.vertices), 3)
        self.assertEqual(len(mesh.indices), 3)

    def test_get_mesh_rejects_http_urls(self):
        with self.assertRaisesRegex(ValueError, "HTTP USD URLs are not supported"):
            newton.usd.get_mesh("http://example.com/marker.usda", compute_inertia=False)

    def test_get_mesh_prim_source_keeps_authored_units(self):
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

    def test_get_mesh_merges_multiple_mesh_prims(self):
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


class _MeshLoggingProbe(ViewerNull):
    def __init__(self):
        super().__init__(num_frames=1)
        self.mesh_calls = []
        self.instance_calls = []

    def log_mesh(
        self,
        name,
        points,
        indices,
        normals=None,
        uvs=None,
        texture=None,
        hidden=False,
        backface_culling=True,
        color=None,
        roughness=None,
        metallic=None,
    ):
        self.mesh_calls.append(
            {
                "name": name,
                "points": points,
                "indices": indices,
                "hidden": hidden,
                "color": color,
                "roughness": roughness,
                "metallic": metallic,
            }
        )

    def log_instances(self, name, mesh, xforms, scales, colors, materials, hidden=False):
        self.instance_calls.append(
            {
                "name": name,
                "mesh": mesh,
                "xforms": xforms,
                "scales": scales,
                "colors": colors,
                "materials": materials,
                "hidden": hidden,
            }
        )


class TestViewerMeshLogging(unittest.TestCase):
    def test_log_mesh_instances_accepts_newton_mesh(self):
        viewer = _MeshLoggingProbe()
        mesh = newton.Mesh(
            vertices=np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32),
            indices=np.array([0, 1, 2], dtype=np.int32),
            compute_inertia=False,
            color=(0.2, 0.4, 0.6),
            roughness=0.7,
            metallic=0.1,
        )

        viewer.log_mesh_instances("/debug/marker", mesh)

        self.assertEqual(len(viewer.mesh_calls), 1)
        self.assertEqual(len(viewer.instance_calls), 1)
        self.assertTrue(viewer.mesh_calls[0]["hidden"])
        self.assertEqual(viewer.mesh_calls[0]["color"], (0.2, 0.4, 0.6))
        self.assertEqual(viewer.mesh_calls[0]["roughness"], 0.7)
        self.assertEqual(viewer.mesh_calls[0]["metallic"], 0.1)
        self.assertEqual(viewer.instance_calls[0]["name"], "/debug/marker")
        self.assertEqual(len(viewer.instance_calls[0]["xforms"]), 1)
        self.assertEqual(len(viewer.instance_calls[0]["scales"]), 1)

    def test_log_mesh_instances_qualifies_active_layer_name(self):
        viewer = _MeshLoggingProbe()
        viewer.activate("markers")
        mesh = newton.Mesh(
            vertices=np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32),
            indices=np.array([0, 1, 2], dtype=np.int32),
            compute_inertia=False,
        )

        viewer.log_mesh_instances("/debug/marker", mesh)

        self.assertEqual(viewer.instance_calls[0]["name"], "/layers/markers/debug/marker")

    def test_log_mesh_instances_rejects_mismatched_instance_arrays(self):
        viewer = _MeshLoggingProbe()
        mesh = newton.Mesh(
            vertices=np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32),
            indices=np.array([0, 1, 2], dtype=np.int32),
            compute_inertia=False,
        )
        xforms = wp.array([wp.transform_identity(), wp.transform_identity()], dtype=wp.transform)

        with self.assertRaisesRegex(ValueError, "Expected 1 or 2 vec3 values"):
            viewer.log_mesh_instances("/debug/bad_scales", mesh, xforms=xforms, scales=np.ones((3, 3)))

        with self.assertRaisesRegex(ValueError, "Expected 1 or 2 vec4 values"):
            viewer.log_mesh_instances("/debug/bad_materials", mesh, xforms=xforms, materials=np.ones((3, 4)))

    def test_log_mesh_instances_rejects_negative_determinant_scale(self):
        viewer = _MeshLoggingProbe()
        mesh = newton.Mesh(
            vertices=np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32),
            indices=np.array([0, 1, 2], dtype=np.int32),
            compute_inertia=False,
        )

        with self.assertRaisesRegex(ValueError, "negative-determinant scales"):
            viewer.log_mesh_instances("/debug/mirrored", mesh, scales=(-1.0, 1.0, 1.0))

    def test_log_usd_caches_loaded_mesh_by_default(self):
        viewer = _MeshLoggingProbe()
        mesh = newton.Mesh(
            vertices=np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32),
            indices=np.array([0, 1, 2], dtype=np.int32),
            compute_inertia=False,
        )

        with mock.patch("newton.usd.get_mesh", return_value=mesh) as get_mesh:
            first = viewer.log_usd("/debug/marker_a", "marker.usda")
            second = viewer.log_usd("/debug/marker_b", "marker.usda")
            uncached = viewer.log_usd("/debug/marker_c", "marker.usda", cache_mesh=False)

        self.assertIs(first, second)
        self.assertIs(first, uncached)
        self.assertEqual(get_mesh.call_count, 2)
        self.assertEqual(len(viewer.instance_calls), 3)

    def test_log_usd_does_not_cache_handle_sources(self):
        viewer = _MeshLoggingProbe()
        source = object()
        mesh_a = newton.Mesh(
            vertices=np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32),
            indices=np.array([0, 1, 2], dtype=np.int32),
            compute_inertia=False,
        )
        mesh_b = newton.Mesh(
            vertices=np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 2.0, 0.0]], dtype=np.float32),
            indices=np.array([0, 1, 2], dtype=np.int32),
            compute_inertia=False,
        )

        with mock.patch("newton.usd.get_mesh", side_effect=[mesh_a, mesh_b]) as get_mesh:
            first = viewer.log_usd("/debug/marker_a", source)
            second = viewer.log_usd("/debug/marker_b", source)

        self.assertIs(first, mesh_a)
        self.assertIs(second, mesh_b)
        self.assertEqual(get_mesh.call_count, 2)
        self.assertEqual(viewer._usd_mesh_cache, {})

    def test_log_mesh_instances_cache_distinguishes_mesh_colors(self):
        viewer = _MeshLoggingProbe()
        vertices = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
        indices = np.array([0, 1, 2], dtype=np.int32)
        red = newton.Mesh(vertices=vertices, indices=indices, compute_inertia=False, color=(1.0, 0.0, 0.0))
        blue = newton.Mesh(vertices=vertices, indices=indices, compute_inertia=False, color=(0.0, 0.0, 1.0))

        viewer.log_mesh_instances("/debug/red", red)
        viewer.log_mesh_instances("/debug/blue", blue)

        self.assertEqual(len(viewer.mesh_calls), 2)
        self.assertEqual(viewer.mesh_calls[0]["color"], (1.0, 0.0, 0.0))
        self.assertEqual(viewer.mesh_calls[1]["color"], (0.0, 0.0, 1.0))

    def test_winding_flipped_mesh_preserves_material_fields(self):
        viewer = _MeshLoggingProbe()
        mesh = newton.Mesh(
            vertices=np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32),
            indices=np.array([0, 1, 2], dtype=np.int32),
            compute_inertia=False,
            color=(0.2, 0.4, 0.6),
            roughness=0.7,
            metallic=0.1,
        )

        viewer._populate_geometry(
            int(newton.GeoType.MESH),
            (1.0, 1.0, 1.0),
            0.0,
            True,
            geo_src=mesh,
            mirror=True,
        )

        self.assertEqual(len(viewer.mesh_calls), 1)
        assert_np_equal(viewer.mesh_calls[0]["indices"].numpy(), np.array([0, 2, 1], dtype=np.int32))
        self.assertEqual(viewer.mesh_calls[0]["color"], (0.2, 0.4, 0.6))
        self.assertEqual(viewer.mesh_calls[0]["roughness"], 0.7)
        self.assertEqual(viewer.mesh_calls[0]["metallic"], 0.1)

    @unittest.skipUnless(USD_AVAILABLE, "Requires usd-core")
    def test_log_usd_loads_and_logs_mesh(self):
        viewer = _MeshLoggingProbe()
        with tempfile.TemporaryDirectory() as tmpdir:
            stage_path = _create_referenced_mesh_stage(tmpdir)

            mesh = viewer.log_usd("/debug/usd_marker", stage_path, root_path="/Marker")

        self.assertIsInstance(mesh, newton.Mesh)
        self.assertEqual(len(viewer.mesh_calls), 1)
        self.assertEqual(len(viewer.instance_calls), 1)
        self.assertEqual(viewer.instance_calls[0]["name"], "/debug/usd_marker")


if __name__ == "__main__":
    unittest.main(verbosity=2)
