# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Discover deformable visual geometry without building a simulation."""

import unittest

import newton.usd
from newton.tests.unittest_utils import USD_AVAILABLE


@unittest.skipUnless(USD_AVAILABLE, "Requires usd-core")
class TestUsdDeformableVisualDiscovery(unittest.TestCase):
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

    def test_referenced_instance_visuals(self):
        """Find visual prims in each instance namespace without editing the stage."""
        from pxr import Usd, UsdGeom

        asset = Usd.Stage.CreateInMemory()
        body = UsdGeom.Xform.Define(asset, "/Asset").GetPrim()
        body.AddAppliedSchema("PhysicsDeformableBodyAPI")
        UsdGeom.Mesh.Define(asset, "/Asset/Mesh")
        stage = Usd.Stage.CreateInMemory()
        for name in ("A", "B"):
            instance = stage.DefinePrim(f"/{name}")
            instance.GetReferences().AddReference(asset.GetRootLayer().identifier, "/Asset")
            instance.SetInstanceable(True)
            before = stage.GetRootLayer().ExportToString()
            visuals = newton.usd.find_deformable_visual_prims(instance)
            self.assertEqual([str(p.GetPath()) for p in visuals], [f"/{name}/Mesh"])
            self.assertTrue(visuals[0].IsInstanceProxy())
            self.assertEqual(stage.GetRootLayer().ExportToString(), before)


if __name__ == "__main__":
    unittest.main()
