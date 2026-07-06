# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""End-to-end import of the bundled mixed-deformable scene (cables + cloth + volumes).

The fixture mirrors the shape of Isaac Lab generated assets -- each deformable is a
``PhysicsDeformableBodyAPI`` Xform with a simulation-geometry child and a bound family
material -- without external references. This module owns the happy-path import contract
for all three families and the single VBD simulation smoke test; per-family modules cover
family-specific lowering, and ``test_import_usd_deformable_groups`` covers the builder
group registries across the model lifecycle.
"""

import os
import unittest
import warnings

import numpy as np
import warp as wp

import newton
from newton.tests._usd_deformable_test_utils import (
    _add_cable_curve,
    _add_cloth_mesh,
    _deformable_stage,
    group_labels,
    group_range,
)
from newton.tests.unittest_utils import USD_AVAILABLE, add_function_test, get_selected_cuda_test_devices

_ASSET = os.path.join(os.path.dirname(__file__), "assets", "deformables_mixed.usda")

_CABLE_PTS = [(0.0, 0.0, 1.0), (0.1, 0.0, 1.0), (0.2, 0.0, 1.0), (0.3, 0.0, 1.0)]


def _author_unit_tet(stage, path, *, sim_api=True, y=0.0):
    """Author a single-tetrahedron TetMesh, optionally marked as a proposal simulation mesh."""
    from pxr import UsdGeom

    tet = UsdGeom.TetMesh.Define(stage, path)
    tet.CreatePointsAttr([(0.0, y, 1.0), (1.0, y, 1.0), (0.0, y + 1.0, 1.0), (0.0, y, 2.0)])
    tet.CreateTetVertexIndicesAttr([(0, 1, 2, 3)])
    if sim_api:
        tet.GetPrim().AddAppliedSchema("PhysicsVolumeDeformableSimAPI")
    return tet


@unittest.skipUnless(USD_AVAILABLE, "Requires usd-core")
class TestUSDDeformableMixed(unittest.TestCase):
    """Mixed cable/cloth/volume scene imports, groups, and finalizes in one pass."""

    def test_physics_material_api_density_supplies_mass(self):
        """The proposal reuses UsdPhysicsMaterialAPI for deformables: a bound material
        applying only the base API supplies density to all three families, while
        family-specific properties keep their normal fallbacks."""
        from pxr import Sdf, UsdPhysics, UsdShade

        def bind_density_material(stage, prim, density):
            mat = UsdShade.Material.Define(stage, "/World/BaseMat")
            UsdPhysics.MaterialAPI.Apply(mat.GetPrim())
            mat.GetPrim().CreateAttribute("physics:density", Sdf.ValueTypeNames.Float).Set(density)
            UsdShade.MaterialBindingAPI.Apply(prim).Bind(mat, materialPurpose="physics")

        with self.subTest(family="soft"):
            stage = _deformable_stage()
            tet = _author_unit_tet(stage, "/World/Soft", sim_api=True)
            bind_density_material(stage, tet.GetPrim(), 600.0)
            builder = newton.ModelBuilder()
            builder.default_tet_density = 1.0
            with warnings.catch_warnings(record=True):
                warnings.simplefilter("always")
                result = builder.add_usd(stage, deformable_results=True)
            self.assertEqual(result["path_soft_attrs"]["/World/Soft"]["resolved_density"], 600.0)
            # density 600 over the unit tet (V = 1/6) -> total particle mass = 100.
            self.assertAlmostEqual(sum(builder.particle_mass), 100.0, delta=1e-3)

        with self.subTest(family="cloth"):
            stage = _deformable_stage()
            mesh = _add_cloth_mesh(stage, "/World/Cloth")
            bind_density_material(stage, mesh.GetPrim(), 600.0)
            builder = newton.ModelBuilder()
            with warnings.catch_warnings(record=True):
                warnings.simplefilter("always")
                result = builder.add_usd(stage, deformable_results=True)
            self.assertEqual(result["path_cloth_attrs"]["/World/Cloth"]["resolved_density"], 600.0)

        with self.subTest(family="cable"):
            stage = _deformable_stage()
            curve = _add_cable_curve(
                stage, "/World/Cable", [(0.0, 0.0, 1.0), (0.1, 0.0, 1.0), (0.2, 0.0, 1.0)], thickness=None
            )
            bind_density_material(stage, curve.GetPrim(), 600.0)
            builder = newton.ModelBuilder()
            with warnings.catch_warnings(record=True):
                warnings.simplefilter("always")
                result = builder.add_usd(stage, deformable_results=True)
            self.assertEqual(result["path_cable_attrs"]["/World/Cable"]["resolved_density"], 600.0)

    def test_mixed_scene_imports_and_finalizes(self):
        """The mixed scene builds every family with correct counts, labels, disjoint
        ranges, materials, and per-cable articulations, and finalizes in one pass."""
        builder = newton.ModelBuilder()
        builder.add_usd(_ASSET)

        self.assertEqual(group_labels(builder, "cable"), ["/World/CableA/sim", "/World/CableB/sim"])
        self.assertEqual(group_labels(builder, "cloth"), ["/World/Cloth/sim"])
        self.assertEqual(group_labels(builder, "soft"), ["/World/SoftA/sim", "/World/SoftB/sim"])

        # Each cable: 3 segments -> 3 capsule bodies wrapped in its own articulation, with
        # the material thickness as capsule radius.
        cable_body_ranges = []
        for path in ("/World/CableA/sim", "/World/CableB/sim"):
            b0, b1 = group_range(builder, "cable", path, "body")
            self.assertEqual(b1 - b0, 3)
            cable_body_ranges.append((b0, b1))
            self.assertIn(f"{path}_articulation", builder.articulation_label)
            radius = builder.shape_scale[builder.body_shapes[b0][0]][0]
            self.assertAlmostEqual(float(radius), 0.01, places=6)  # thickness 0.02 -> radius
        self.assertNotEqual(cable_body_ranges[0], cable_body_ranges[1])
        self.assertEqual(builder.body_count, 6)
        self.assertEqual(len(builder.articulation_label), 2)

        # Cloth: 4 particles / 2 triangles with the material's stretch modulus scaled by
        # thickness (tri_ke = stretchStiffness * thickness) and no fabricated area term.
        p0, p1 = group_range(builder, "cloth", "/World/Cloth/sim", "particle")
        t0, t1 = group_range(builder, "cloth", "/World/Cloth/sim", "tri")
        self.assertEqual((p1 - p0, t1 - t0), (4, 2))
        self.assertAlmostEqual(builder.tri_materials[t0][0], 1.0e5 * 0.001, delta=1.0)  # tri_ke
        self.assertEqual(builder.tri_materials[t0][1], 0.0)  # tri_ka

        # Volumes: one tet of 4 particles each, disjoint back-to-back particle ranges.
        soft_ranges = [group_range(builder, "soft", f"/World/Soft{s}/sim", "particle") for s in ("A", "B")]
        self.assertEqual([end - start for start, end in soft_ranges], [4, 4])
        self.assertNotEqual(soft_ranges[0], soft_ranges[1])
        self.assertEqual(builder.particle_count, 12)
        self.assertEqual(builder.tet_count, 2)

        model = builder.finalize()
        self.assertEqual(model.particle_count, 12)
        self.assertEqual(model.body_count, 6)

    def test_disabled_and_kinematic_deformables_are_skipped(self):
        """physics:bodyEnabled=false / kinematicEnabled=true skip a deformable of any family
        (Newton has no disabled/kinematic deformable representation); enabled ones import."""
        from pxr import Sdf, UsdGeom

        stage = _deformable_stage()
        _add_cable_curve(stage, "/World/Cable", _CABLE_PTS)
        _add_cloth_mesh(stage, "/World/Cloth")
        _author_unit_tet(stage, "/World/Soft")
        disabled_cable = _add_cable_curve(stage, "/World/CableOff", [(x, 1.0, z) for x, _, z in _CABLE_PTS])
        disabled_cable.GetPrim().CreateAttribute("physics:bodyEnabled", Sdf.ValueTypeNames.Bool).Set(False)
        kinematic_cloth = _add_cloth_mesh(stage, "/World/ClothKin")
        kinematic_cloth.GetPrim().CreateAttribute("physics:kinematicEnabled", Sdf.ValueTypeNames.Bool).Set(True)
        # Governance variant: the flag sits on the deformable BODY prim, not the sim mesh.
        body = UsdGeom.Xform.Define(stage, "/World/SoftOff")
        body.GetPrim().AddAppliedSchema("PhysicsDeformableBodyAPI")
        body.GetPrim().CreateAttribute("physics:bodyEnabled", Sdf.ValueTypeNames.Bool).Set(False)
        _author_unit_tet(stage, "/World/SoftOff/sim", y=2.0)

        builder = newton.ModelBuilder()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            builder.add_usd(stage)
        messages = [str(w.message) for w in caught]

        self.assertEqual(group_labels(builder, "cable"), ["/World/Cable"])
        self.assertEqual(group_labels(builder, "cloth"), ["/World/Cloth"])
        self.assertEqual(group_labels(builder, "soft"), ["/World/Soft"])
        self.assertTrue(any("/World/CableOff" in m and "bodyEnabled is false" in m for m in messages))
        self.assertTrue(any("/World/ClothKin" in m and "kinematic deformables" in m for m in messages))
        self.assertTrue(any("/World/SoftOff/sim" in m and "bodyEnabled is false" in m for m in messages))

    def test_unsupported_rest_and_velocity_fields_warn(self):
        """Authored rest state and velocities warn per prim but do not block the import."""
        from pxr import Sdf, UsdGeom

        stage = _deformable_stage()
        cable = _add_cable_curve(stage, "/World/Cable", _CABLE_PTS)
        cable.GetPrim().CreateAttribute("physics:restNormals", Sdf.ValueTypeNames.Vector3fArray).Set(
            [(0.0, 0.0, 1.0)] * 4
        )
        UsdGeom.PointBased(cable.GetPrim()).CreateVelocitiesAttr([(1.0, 0.0, 0.0)] * 4)
        cloth = _add_cloth_mesh(stage, "/World/Cloth")
        cloth.GetPrim().CreateAttribute("physics:restBendAngles", Sdf.ValueTypeNames.FloatArray).Set([0.1])
        UsdGeom.PointBased(cloth.GetPrim()).CreateVelocitiesAttr([(1.0, 0.0, 0.0)] * 4)
        tet = _author_unit_tet(stage, "/World/Soft")
        tet.GetPrim().CreateAttribute("physics:restShapePoints", Sdf.ValueTypeNames.Point3fArray).Set(
            [(0.0, 0.0, 0.0)] * 4
        )

        builder = newton.ModelBuilder()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            builder.add_usd(stage)
        messages = [str(w.message) for w in caught]

        for field, path in (
            ("restNormals", "/World/Cable"),
            ("restBendAngles", "/World/Cloth"),
            ("restShapePoints", "/World/Soft"),
        ):
            self.assertTrue(any(path in m and field in m and "not yet supported" in m for m in messages), field)
        for path in ("/World/Cable", "/World/Cloth"):
            self.assertTrue(any(path in m and "velocities are not imported" in m for m in messages), path)
        # The unsupported fields are dropped, not the deformables.
        self.assertEqual(group_labels(builder, "cable"), ["/World/Cable"])
        self.assertEqual(group_labels(builder, "cloth"), ["/World/Cloth"])
        self.assertEqual(group_labels(builder, "soft"), ["/World/Soft"])

    def test_plain_geometry_is_not_reinterpreted(self):
        """Geometry without the family sim APIs must not become a deformable. A bare TetMesh
        is the documented exception: it keeps the legacy soft-body import."""
        from pxr import UsdGeom

        stage = _deformable_stage()
        curves = UsdGeom.BasisCurves.Define(stage, "/World/Curve")
        curves.CreateTypeAttr().Set(UsdGeom.Tokens.linear)
        curves.CreatePointsAttr(_CABLE_PTS)
        curves.CreateCurveVertexCountsAttr([len(_CABLE_PTS)])
        mesh = UsdGeom.Mesh.Define(stage, "/World/Mesh")
        mesh.CreatePointsAttr([(0.0, 1.0, 1.0), (1.0, 1.0, 1.0), (1.0, 2.0, 1.0)])
        mesh.CreateFaceVertexCountsAttr([3])
        mesh.CreateFaceVertexIndicesAttr([0, 1, 2])
        _author_unit_tet(stage, "/World/BareTet", sim_api=False, y=3.0)

        builder = newton.ModelBuilder()
        builder.add_usd(stage)

        self.assertEqual(group_labels(builder, "cable"), [])
        self.assertEqual(group_labels(builder, "cloth"), [])
        self.assertEqual(builder.body_count, 0)
        self.assertEqual(group_labels(builder, "soft"), ["/World/BareTet"])  # legacy import
        self.assertEqual(builder.particle_count, 4)

    def test_one_simulation_geometry_per_body_across_families(self):
        """A deformable body governs exactly one simulation geometry across ALL families:
        a body-level mass must not be applied once per family. The first candidate in
        traversal order wins; the others warn and are skipped."""
        from pxr import UsdGeom

        stage = _deformable_stage()
        body = UsdGeom.Xform.Define(stage, "/World/Body")
        body.GetPrim().AddAppliedSchema("PhysicsDeformableBodyAPI")
        from pxr import Sdf

        body.GetPrim().CreateAttribute("physics:mass", Sdf.ValueTypeNames.Float).Set(10.0)
        _add_cloth_mesh(stage, "/World/Body/Cloth")  # first in traversal order -> owner
        _add_cable_curve(stage, "/World/Body/Cable", _CABLE_PTS)

        builder = newton.ModelBuilder()
        with self.assertWarnsRegex(UserWarning, "already has simulation geometry"):
            builder.add_usd(stage)

        self.assertEqual(group_labels(builder, "cloth"), ["/World/Body/Cloth"])
        self.assertEqual(group_labels(builder, "cable"), [], "the second family must be skipped")
        self.assertAlmostEqual(sum(builder.particle_mass), 10.0, places=4)
        self.assertEqual(builder.body_count, 0, "no cable bodies: mass is counted once")

    def test_deformable_results_are_opt_in(self):
        """The default add_usd return carries no deformable entries; deformable_results=True
        adds exactly the documented map and attrs keys."""
        keys = (
            "path_cable_map",
            "path_cloth_map",
            "path_soft_map",
            "path_attachment_map",
            "path_cable_attrs",
            "path_cloth_attrs",
            "path_soft_attrs",
            "path_attachment_attrs",
        )
        builder = newton.ModelBuilder()
        result = builder.add_usd(_ASSET)
        for key in keys:
            self.assertNotIn(key, result)

        opt_in = newton.ModelBuilder()
        result = opt_in.add_usd(_ASSET, deformable_results=True)
        for key in keys:
            self.assertIn(key, result)
        self.assertIn("/World/CableA/sim", result["path_cable_map"])
        self.assertEqual(result["path_cloth_map"]["/World/Cloth/sim"]["particle"], (0, 4))

    def test_rigid_only_stage_skips_deformable_passes(self):
        """A stage with only rigid bodies imports its rigid content unchanged and registers no
        deformable results (the deformable passes are skipped because no candidate prims exist)."""
        from pxr import UsdGeom, UsdPhysics

        stage = _deformable_stage()
        body = UsdGeom.Xform.Define(stage, "/World/Body")
        UsdPhysics.RigidBodyAPI.Apply(body.GetPrim())
        cube = UsdGeom.Cube.Define(stage, "/World/Body/Col")
        cube.CreateSizeAttr(0.1)
        UsdPhysics.CollisionAPI.Apply(cube.GetPrim())

        builder = newton.ModelBuilder()
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            builder.add_usd(stage)

        self.assertEqual(builder.body_count, 1)
        self.assertEqual(builder.shape_count, 1)
        self.assertEqual(builder.particle_count, 0)
        for family in ("cable", "cloth", "soft"):
            self.assertEqual(group_labels(builder, family), [])
        model = builder.finalize()
        self.assertEqual(model.body_count, 1)

    def test_mixed_scene_simulates(self, device=None):
        """All three imported families coexist in one SolverVBD model and stay finite."""
        if device is None or not wp.get_device(device).is_cuda:
            self.skipTest("VBD deformable simulation requires a CUDA device")

        with wp.ScopedDevice(device):
            builder = newton.ModelBuilder()
            builder.add_usd(_ASSET)
            cable_ranges = [group_range(builder, "cable", p, "body") for p in group_labels(builder, "cable")]
            particle_ranges = [
                group_range(builder, family, path, "particle")
                for family in ("cloth", "soft")
                for path in group_labels(builder, family)
            ]
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

            body_q = state_0.body_q.numpy()
            for b0, b1 in cable_ranges:
                self.assertTrue(np.isfinite(body_q[b0:b1]).all(), "non-finite cable body state")
            particle_q = state_0.particle_q.numpy()
            for p0, p1 in particle_ranges:
                self.assertTrue(np.isfinite(particle_q[p0:p1]).all(), "non-finite particle state")


devices = get_selected_cuda_test_devices()
add_function_test(
    TestUSDDeformableMixed,
    "test_mixed_scene_simulates",
    TestUSDDeformableMixed.test_mixed_scene_simulates,
    devices=devices,
)


if __name__ == "__main__":
    unittest.main(verbosity=2)
