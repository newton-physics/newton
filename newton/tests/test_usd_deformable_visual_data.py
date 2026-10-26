# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Read deformable visual geometry and bind poses without building a simulation."""

import unittest

import numpy as np

import newton.usd
from newton.tests.unittest_utils import USD_AVAILABLE


def _author_pose(prim, name, points, *, purposes=("bindPose",)):
    """Author proposal attributes without requiring the proposal schema plugin."""
    from pxr import Sdf

    prim.AddAppliedSchema(f"PhysicsDeformablePoseAPI:{name}")
    prefix = f"physics:deformablePose:{name}"
    prim.CreateAttribute(f"{prefix}:purposes", Sdf.ValueTypeNames.TokenArray).Set(purposes)
    if points is not None:
        prim.CreateAttribute(f"{prefix}:points", Sdf.ValueTypeNames.Point3fArray).Set(points)


# The tire example of the AOUSD deformables proposal (commit 63c74d79, section "Geometry
# Embeddings") with concrete points. It authors the pose attributes independently of the
# reader, so a name mismatch would fail here rather than in a fixture built by the reader's own
# vocabulary.
_PROPOSAL_TIRE_EXAMPLE = """#usda 1.0
def Xform "tire" (
    prepend apiSchemas = ["PhysicsDeformableBodyAPI"]
)
{
    def TetMesh "simulationMesh" (
        prepend apiSchemas = [
            "PhysicsVolumeDeformableSimAPI",
            "PhysicsDeformablePoseAPI:custom"
        ]
    ) {
        point3f[] points = [(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)]
        int4[] tetVertexIndices = [(0, 1, 2, 3)]
        token[] physics:deformablePose:custom:purposes = ["bindPose"]
        point3f[] physics:deformablePose:custom:points = [(0, 0, 0), (2, 0, 0), (0, 2, 0), (0, 0, 2)]
    }

    def Mesh "collisionMesh" (
        prepend apiSchemas = [
            "PhysicsCollisionAPI",
            "PhysicsDeformablePoseAPI:custom"
        ]
    ) {
        point3f[] points = [(0, 0, 0), (1, 0, 0), (0, 1, 0)]
        int[] faceVertexCounts = [3]
        int[] faceVertexIndices = [0, 1, 2]
        token[] physics:deformablePose:custom:purposes = ["bindPose"]
        point3f[] physics:deformablePose:custom:points = [(0, 0, 0), (2, 0, 0), (0, 2, 0)]
    }

    def Mesh "graphicsMesh" (
        prepend apiSchemas = [ "PhysicsDeformablePoseAPI:custom" ]
    ) {
        point3f[] points = [(0, 0, 0), (1, 0, 0), (0, 1, 0)]
        int[] faceVertexCounts = [3]
        int[] faceVertexIndices = [0, 1, 2]
        token[] physics:deformablePose:custom:purposes = ["bindPose"]
        point3f[] physics:deformablePose:custom:points = [(0, 0, 0), (3, 0, 0), (0, 3, 0)]
    }
}
"""


@unittest.skipUnless(USD_AVAILABLE, "Requires usd-core")
class TestUsdDeformableVisualData(unittest.TestCase):
    def test_proposal_example_reads_visual_bind_pose(self):
        """Read the proposal's tire example with the attribute names it authors."""
        from pxr import Usd

        stage = Usd.Stage.CreateInMemory()
        self.assertTrue(stage.GetRootLayer().ImportFromString(_PROPOSAL_TIRE_EXAMPLE))
        tire = stage.GetPrimAtPath("/tire")

        visuals = newton.usd.find_deformable_visual_prims(tire)
        self.assertEqual([str(p.GetPath()) for p in visuals], ["/tire/graphicsMesh"])
        np.testing.assert_array_equal(
            newton.usd.get_deformable_bind_pose(visuals[0]), [(0, 0, 0), (3, 0, 0), (0, 3, 0)]
        )
        np.testing.assert_array_equal(
            newton.usd.get_deformable_bind_pose(stage.GetPrimAtPath("/tire/simulationMesh")),
            [(0, 0, 0), (2, 0, 0), (0, 2, 0), (0, 0, 2)],
        )

    def test_bind_pose_follows_loaded_mesh_vertices(self):
        """Map authored bind points to render vertices split at UV seams."""
        from pxr import Sdf, Usd, UsdGeom

        stage = Usd.Stage.CreateInMemory()
        source = UsdGeom.Mesh.Define(stage, "/Visual")
        # Coincident points can have different bind positions: preserve IDs,
        # rather than matching render vertices by their initial coordinates.
        source.CreatePointsAttr([(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 0), (-1, 0, 0)])
        source.CreateFaceVertexCountsAttr([3, 3])
        source.CreateFaceVertexIndicesAttr([0, 1, 2, 3, 2, 4])
        UsdGeom.PrimvarsAPI(source).CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray, "faceVarying").Set(
            [(0, 0), (1, 0), (0, 1), (0.5, 0), (0.5, 1), (1, 0)]
        )
        _author_pose(source.GetPrim(), "reference", [(0, 0, 0), (2, 0, 0), (0, 2, 0), (0, 0, 1), (-2, 0, 0)])
        before = stage.GetRootLayer().ExportToString()

        mesh, point_indices = newton.usd.get_mesh(
            source.GetPrim(), load_uvs=True, return_source_point_indices=True, compute_inertia=False
        )
        bind_points = newton.usd.get_deformable_bind_pose(source.GetPrim())[point_indices]

        self.assertEqual(len(mesh.vertices), 6)
        np.testing.assert_array_equal(point_indices[mesh.indices], [0, 1, 2, 3, 2, 4])
        np.testing.assert_array_equal(
            bind_points[mesh.indices], [(0, 0, 0), (2, 0, 0), (0, 2, 0), (0, 0, 1), (0, 2, 0), (-2, 0, 0)]
        )
        self.assertEqual(stage.GetRootLayer().ExportToString(), before)

    def test_bind_pose_uses_authored_local_points(self):
        """Read explicit or default bind points without applying transforms or rest shapes."""
        from pxr import Sdf, Usd, UsdGeom

        stage = Usd.Stage.CreateInMemory()
        UsdGeom.SetStageMetersPerUnit(stage, 0.01)
        points = [(0.0, 0.0, 0.0), (1.0, 2.0, 3.0)]
        bind_points = [(4.0, 5.0, 6.0), (7.0, 8.0, 9.0)]
        for schema in (UsdGeom.Mesh, UsdGeom.BasisCurves, UsdGeom.TetMesh, UsdGeom.Points):
            with self.subTest(schema=schema.__name__):
                geometry = schema.Define(stage, f"/World/{schema.__name__}")
                prim = geometry.GetPrim()
                geometry.CreatePointsAttr(points)
                geometry.GetPointsAttr().Set(bind_points, Usd.TimeCode(1.0))
                UsdGeom.Xformable(prim).AddTranslateOp().Set((100.0, 0.0, 0.0))
                prim.CreateAttribute("physics:restShapePoints", Sdf.ValueTypeNames.Point3fArray).Set(bind_points)
                _author_pose(prim, "other", bind_points, purposes=("collisionFilter",))
                np.testing.assert_array_equal(newton.usd.get_deformable_bind_pose(prim), points)

                _author_pose(prim, "reference", bind_points, purposes=("collisionFilter", "bindPose"))
                before = stage.GetRootLayer().ExportToString()
                result = newton.usd.get_deformable_bind_pose(prim)
                np.testing.assert_array_equal(result, bind_points)
                result[0] = 0.0
                np.testing.assert_array_equal(newton.usd.get_deformable_bind_pose(prim), bind_points)
                self.assertEqual(stage.GetRootLayer().ExportToString(), before)

        empty = UsdGeom.Mesh.Define(stage, "/Empty")
        empty.CreatePointsAttr([])
        self.assertEqual(newton.usd.get_deformable_bind_pose(empty.GetPrim()).shape, (0, 3))

    def test_bind_pose_rejects_invalid_data(self):
        """Reject malformed or ambiguous bind poses instead of using a guessed pose."""
        from pxr import Usd, UsdGeom

        stage = Usd.Stage.CreateInMemory()
        for name, points, message in (
            ("missing", None, "default-time points"),
            ("count", [(0.0, 0.0, 0.0)], "point count"),
            ("nonfinite", [(float("nan"), 0.0, 0.0)] * 2, "finite"),
        ):
            with self.subTest(name=name):
                mesh = UsdGeom.Mesh.Define(stage, f"/{name}")
                mesh.CreatePointsAttr([(0.0, 0.0, 0.0)] * 2)
                _author_pose(mesh.GetPrim(), "reference", points)
                with self.assertRaisesRegex(ValueError, message):
                    newton.usd.get_deformable_bind_pose(mesh.GetPrim())

        mesh = UsdGeom.Mesh.Define(stage, "/Multiple")
        mesh.CreatePointsAttr([(0.0, 0.0, 0.0)])
        for name in ("first", "second"):
            _author_pose(mesh.GetPrim(), name, [(0.0, 0.0, 0.0)])
        with self.assertRaisesRegex(ValueError, "Multiple.*bindPose"):
            newton.usd.get_deformable_bind_pose(mesh.GetPrim())

        sampled = UsdGeom.Mesh.Define(stage, "/Sampled")
        sampled.CreatePointsAttr().Set([(1.0, 2.0, 3.0)], Usd.TimeCode(1.0))
        sampled.CreateFaceVertexCountsAttr([])
        sampled.CreateFaceVertexIndicesAttr([])
        with self.assertRaisesRegex(ValueError, "default-time points"):
            newton.usd.get_deformable_bind_pose(sampled.GetPrim())
        # The render mesh is read at the default time code, so an explicit bind pose on
        # animated-only points validates but has no loadable mesh.
        with self.assertRaisesRegex(ValueError, "default-time points"):
            newton.usd.get_mesh(sampled.GetPrim(), compute_inertia=False)
        _author_pose(sampled.GetPrim(), "reference", [(4.0, 5.0, 6.0)])
        np.testing.assert_array_equal(newton.usd.get_deformable_bind_pose(sampled.GetPrim()), [(4.0, 5.0, 6.0)])
        with self.assertRaisesRegex(ValueError, "PointBased"):
            newton.usd.get_deformable_bind_pose(UsdGeom.Xform.Define(stage, "/Xform").GetPrim())

    def test_visual_discovery_respects_body_ownership(self):
        """Find visual geometry without claiming simulation, collision, or nested-body geometry.

        A TetMesh or Points prim without a simulation API is visual geometry; prims
        applying an AOUSD simulation API or Newton's particle API are not, even when
        the body prim is the simulation geometry itself. Filtering by path is left to
        the caller.
        """
        from pxr import Usd, UsdGeom, UsdPhysics

        stage = Usd.Stage.CreateInMemory()
        body = UsdGeom.Xform.Define(stage, "/Body").GetPrim()
        body.AddAppliedSchema("PhysicsDeformableBodyAPI")
        mesh = UsdGeom.Mesh.Define(stage, "/Body/Graphics/Mesh").GetPrim()
        curves = UsdGeom.BasisCurves.Define(stage, "/Body/Graphics/Curves").GetPrim()
        points = UsdGeom.Points.Define(stage, "/Body/Graphics/Points").GetPrim()
        for name, schema, api in (
            ("SimCurve", UsdGeom.BasisCurves, "PhysicsCurvesDeformableSimAPI"),
            ("SimSurface", UsdGeom.Mesh, "PhysicsSurfaceDeformableSimAPI"),
            ("SimVolume", UsdGeom.TetMesh, "PhysicsVolumeDeformableSimAPI"),
            ("SimParticles", UsdGeom.Points, "NewtonPointsDeformableSimAPI"),
        ):
            schema.Define(stage, f"/Body/{name}").GetPrim().AddAppliedSchema(api)
        bare_tet = UsdGeom.TetMesh.Define(stage, "/Body/BareTet").GetPrim()
        collider = UsdGeom.Mesh.Define(stage, "/Body/Collider").GetPrim()
        UsdPhysics.CollisionAPI.Apply(collider).CreateCollisionEnabledAttr(False)
        nested = UsdGeom.Xform.Define(stage, "/Body/Nested").GetPrim()
        nested.AddAppliedSchema("PhysicsDeformableBodyAPI")
        nested_mesh = UsdGeom.Mesh.Define(stage, "/Body/Nested/Mesh").GetPrim()
        rigid = UsdGeom.Xform.Define(stage, "/Body/Rigid").GetPrim()
        UsdPhysics.RigidBodyAPI.Apply(rigid)
        UsdGeom.Mesh.Define(stage, "/Body/Rigid/Mesh")
        articulation = UsdGeom.Xform.Define(stage, "/Body/Articulation").GetPrim()
        UsdPhysics.ArticulationRootAPI.Apply(articulation)
        UsdGeom.Mesh.Define(stage, "/Body/Articulation/Mesh")

        particles = UsdGeom.Points.Define(stage, "/Particles").GetPrim()
        for api in ("PhysicsDeformableBodyAPI", "NewtonPointsDeformableSimAPI"):
            particles.AddAppliedSchema(api)

        self.assertEqual(newton.usd.find_deformable_visual_prims(body), [mesh, curves, points, bare_tet])
        self.assertEqual(newton.usd.find_deformable_visual_prims(nested), [nested_mesh])
        self.assertEqual(newton.usd.find_deformable_visual_prims(particles), [])
        with self.assertRaisesRegex(ValueError, "PhysicsDeformableBodyAPI"):
            newton.usd.find_deformable_visual_prims(mesh)

    def test_referenced_instance_visuals_and_pose(self):
        """Read visual geometry in each instance namespace and honor composed pose schemas."""
        from pxr import Sdf, Usd, UsdGeom, UsdShade

        asset = Usd.Stage.CreateInMemory()
        body = UsdGeom.Xform.Define(asset, "/Asset").GetPrim()
        body.AddAppliedSchema("PhysicsDeformableBodyAPI")
        mesh = UsdGeom.Mesh.Define(asset, "/Asset/Mesh")
        mesh.CreatePointsAttr([(0.0, 0.0, 0.0)])
        UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim())
        _author_pose(mesh.GetPrim(), "old", [(1.0, 0.0, 0.0)])
        _author_pose(mesh.GetPrim(), "current", [(2.0, 0.0, 0.0)])

        stage = Usd.Stage.CreateInMemory()
        for name in ("A", "B"):
            instance = stage.DefinePrim(f"/{name}")
            instance.GetReferences().AddReference(asset.GetRootLayer().identifier, "/Asset")
            # The stronger layer removes an obsolete pose without deleting its attributes.
            override = stage.OverridePrim(f"/{name}/Mesh")
            override.SetMetadata("apiSchemas", Sdf.TokenListOp.Create(deletedItems=["PhysicsDeformablePoseAPI:old"]))
            visuals = newton.usd.find_deformable_visual_prims(instance)
            self.assertEqual([str(p.GetPath()) for p in visuals], [f"/{name}/Mesh"])
            np.testing.assert_array_equal(newton.usd.get_deformable_bind_pose(visuals[0]), [(2.0, 0.0, 0.0)])

        mesh.GetPrim().RemoveAppliedSchema("PhysicsDeformablePoseAPI:old")
        stage.GetRootLayer().Clear()
        for name in ("A", "B"):
            instance = stage.DefinePrim(f"/{name}")
            instance.GetReferences().AddReference(asset.GetRootLayer().identifier, "/Asset")
            instance.SetInstanceable(True)
            visuals = newton.usd.find_deformable_visual_prims(instance)
            self.assertEqual([str(p.GetPath()) for p in visuals], [f"/{name}/Mesh"])
            self.assertTrue(visuals[0].IsInstanceProxy())
            np.testing.assert_array_equal(newton.usd.get_deformable_bind_pose(visuals[0]), [(2.0, 0.0, 0.0)])


if __name__ == "__main__":
    unittest.main()
