# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Tests for USD surface-deformable (cloth) import: triangulation, materials, masses, scaling.

Cross-family happy-path, skip-policy, and lifecycle contracts live in
``test_import_usd_deformable_mixed`` and ``test_import_usd_deformable_groups``; this module
owns the cloth-specific lowering (topology, membrane material, mass model, transforms).
"""

import math
import unittest
import warnings

import numpy as np
import warp as wp

import newton
from newton import ShapeFlags
from newton.exceptions import NewtonWarning
from newton.tests._usd_deformable_test_utils import (
    _add_cable_curve,
    _add_cloth_mesh,
    _add_mesh_prim,
    _apply_deformable_body_api,
    _author_deformable_element_array,
    _bind_deformable_material,
    _deformable_stage,
    group_labels,
    group_range,
)
from newton.tests.unittest_utils import USD_AVAILABLE
from newton.usd import SchemaResolverPhysx


@unittest.skipUnless(USD_AVAILABLE, "Requires usd-core")
class TestUSDDeformableCloth(unittest.TestCase):
    """Surface-deformable (cloth) parsing into particles, triangle elements, and bending edges."""

    def test_cloth_quad_mesh_is_triangulated(self):
        """Verify that import fan-triangulates quad faces to support n-gons."""
        from pxr import UsdGeom

        stage = _deformable_stage(up_axis="y")
        mesh = UsdGeom.Mesh.Define(stage, "/World/Cloth")
        mesh.CreatePointsAttr([(0.0, 0.0, 1.0), (1.0, 0.0, 1.0), (1.0, 1.0, 1.0), (0.0, 1.0, 1.0)])
        mesh.CreateFaceVertexCountsAttr([4])  # single quad face
        mesh.CreateFaceVertexIndicesAttr([0, 1, 2, 3])
        mesh.GetPrim().AddAppliedSchema("PhysicsSurfaceDeformableSimAPI")
        mesh.GetPrim().AddAppliedSchema("PhysicsCollisionAPI")
        _author_deformable_element_array(mesh.GetPrim(), "thicknesses", [0.001], "constant")
        _author_deformable_element_array(mesh.GetPrim(), "masses", [8.0], "face")

        builder = newton.ModelBuilder()
        builder.add_usd(stage)
        # 4 quad vertices stay 1:1 with particles.
        self.assertEqual(group_range(builder, "cloth", "/World/Cloth", "particle"), (0, 4))
        # The quad fan-triangulates to 2 triangles.
        self.assertEqual(group_range(builder, "cloth", "/World/Cloth", "tri"), (0, 2))
        self.assertEqual(builder.particle_count, 4)
        self.assertAlmostEqual(sum(builder.particle_mass), 8.0, places=6)

    def test_cloth_display_appearance_imports_to_triangles(self):
        """Import cloth display color and opacity onto every generated triangle."""
        from pxr import Sdf, UsdGeom

        stage = _deformable_stage()
        mesh = _add_cloth_mesh(stage, "/World/Cloth")
        _author_deformable_element_array(mesh.GetPrim(), "thicknesses", [0.001], "constant")
        primvars = UsdGeom.PrimvarsAPI(mesh)
        primvars.CreatePrimvar("displayColor", Sdf.ValueTypeNames.Color3fArray, UsdGeom.Tokens.constant, 1).Set(
            [(0.2, 0.4, 0.6)]
        )
        primvars.CreatePrimvar("displayOpacity", Sdf.ValueTypeNames.FloatArray, UsdGeom.Tokens.constant, 1).Set([0.44])

        expected_color = newton.utils.color_linear_to_srgb((0.2, 0.4, 0.6))
        builder = newton.ModelBuilder()
        builder.add_usd(stage)

        self.assertEqual(builder.tri_count, 2)
        np.testing.assert_allclose(builder.tri_color, np.tile(expected_color, (2, 1)), atol=1e-6, rtol=1e-6)
        np.testing.assert_allclose(builder.tri_opacity, np.full(2, 0.44), atol=1e-6, rtol=1e-6)

    def test_cloth_material_appearance_imports_to_triangles(self):
        """Import bound material color and opacity onto every generated triangle."""
        from pxr import Sdf, UsdShade

        stage = _deformable_stage()
        mesh = _add_cloth_mesh(stage, "/World/Cloth")
        _author_deformable_element_array(mesh.GetPrim(), "thicknesses", [0.001], "constant")
        material = UsdShade.Material.Define(stage, "/World/ClothLook")
        shader = UsdShade.Shader.Define(stage, "/World/ClothLook/PreviewSurface")
        shader.CreateIdAttr("UsdPreviewSurface")
        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set((0.8, 0.2, 0.1))
        shader.CreateInput("opacity", Sdf.ValueTypeNames.Float).Set(0.37)
        material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
        UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(material)

        builder = newton.ModelBuilder()
        builder.add_usd(stage)

        self.assertEqual(builder.tri_count, 2)
        expected_color = newton.utils.color_linear_to_srgb((0.8, 0.2, 0.1))
        np.testing.assert_allclose(builder.tri_color, np.tile(expected_color, (2, 1)), atol=1e-6, rtol=1e-6)
        np.testing.assert_allclose(builder.tri_opacity, np.full(2, 0.37), atol=1e-6, rtol=1e-6)

    def test_cloth_out_of_range_display_color_is_clamped(self):
        """Clamp an HDR cloth display color instead of aborting the import."""
        from pxr import Sdf, UsdGeom

        stage = _deformable_stage()
        mesh = _add_cloth_mesh(stage, "/World/Cloth")
        _author_deformable_element_array(mesh.GetPrim(), "thicknesses", [0.001], "constant")
        UsdGeom.PrimvarsAPI(mesh).CreatePrimvar(
            "displayColor", Sdf.ValueTypeNames.Color3fArray, UsdGeom.Tokens.constant, 1
        ).Set([(1.5, 0.5, 0.5)])

        builder = newton.ModelBuilder()
        with self.assertLogs("newton", level="WARNING") as caught:
            builder.add_usd(stage)
        self.assertTrue(any("Clamping imported color" in m for m in caught.output), caught.output)

        self.assertEqual(builder.tri_count, 2)
        expected_color = newton.utils.color_linear_to_srgb((1.0, 0.5, 0.5))
        np.testing.assert_allclose(builder.tri_color, np.tile(expected_color, (2, 1)), atol=1e-6, rtol=1e-6)

    def test_cloth_rest_shape_drives_material_state(self):
        """Import disjoint rest poses, thickness-weighted masses, lengths, and bend angles."""
        from pxr import Sdf, UsdGeom

        stage = _deformable_stage()
        mesh = UsdGeom.Mesh.Define(stage, "/World/Cloth")
        live_points = [(0.0, 0.0, 0.0), (2.0, 0.0, 0.0), (2.0, 2.0, 0.0), (0.0, 1.0, 0.0)]
        mesh.CreatePointsAttr(live_points)
        mesh.CreateFaceVertexCountsAttr([3, 3])
        mesh.CreateFaceVertexIndicesAttr([0, 1, 2, 0, 2, 3])
        mesh.GetPrim().AddAppliedSchema("PhysicsSurfaceDeformableSimAPI")
        mesh.GetPrim().AddAppliedSchema("PhysicsCollisionAPI")
        mesh.GetPrim().CreateAttribute("physics:restShapePoints", Sdf.ValueTypeNames.Point3fArray).Set(
            [
                (0.0, 0.0, 0.0),
                (1.0, 0.0, 0.0),
                (1.0, 1.0, 0.0),
                (0.0, 0.0, 0.0),
                (1.0, 1.0, 0.0),
                (0.0, 1.0, 0.0),
            ]
        )
        mesh.GetPrim().CreateAttribute("physics:restTriVertexIndices", Sdf.ValueTypeNames.Int3Array).Set(
            [(0, 1, 2), (3, 4, 5)]
        )
        mesh.GetPrim().CreateAttribute("physics:restBendAnglesDefault", Sdf.ValueTypeNames.Token).Set("flat")
        mesh.GetPrim().CreateAttribute("physics:restAdjTriPairs", Sdf.ValueTypeNames.Int2Array).Set([(0, 1)])
        mesh.GetPrim().CreateAttribute("physics:restBendAngles", Sdf.ValueTypeNames.FloatArray).Set([75.0])
        _bind_deformable_material(stage, mesh.GetPrim(), "/World/Mat", density=1.0)
        _author_deformable_element_array(mesh.GetPrim(), "thicknesses", [1.0, 2.0], "face")

        builder = newton.ModelBuilder()
        builder.add_usd(stage)

        np.testing.assert_allclose(np.asarray(builder.particle_q), live_points, atol=1.0e-6)
        self.assertAlmostEqual(sum(builder.tri_areas), 1.0, places=6)
        np.testing.assert_allclose(builder.particle_mass, [0.5, 1.0 / 6.0, 0.5, 1.0 / 3.0])
        np.testing.assert_allclose(builder.tri_poses[0], [[1.0, -1.0], [0.0, 1.0]])
        interior_edge = next(i for i, edge in enumerate(builder.edge_indices) if edge[1] != -1)
        self.assertAlmostEqual(builder.edge_rest_angle[interior_edge], math.radians(75.0), places=6)
        self.assertAlmostEqual(builder.edge_rest_length[interior_edge], math.sqrt(2.0), places=6)

        # Typed totals override body mass and use rest areas with per-face thickness.
        _author_deformable_element_array(mesh.GetPrim(), "masses", [9.0], "constant")
        _apply_deformable_body_api(mesh.GetPrim(), mass=99.0)
        typed_builder = newton.ModelBuilder()
        typed_builder.add_usd(stage)
        np.testing.assert_allclose(typed_builder.particle_mass, [3.0, 1.0, 3.0, 2.0])

    def test_disjoint_rest_topology_keeps_conflicting_edge_lengths(self):
        """Keep triangle rest poses and masses when shared rest-edge lengths disagree."""
        from pxr import Sdf, UsdGeom

        stage = _deformable_stage()
        mesh = UsdGeom.Mesh.Define(stage, "/World/Cloth")
        mesh.CreatePointsAttr([(0.0, 0.0, 0.0), (2.0, 0.0, 0.0), (2.0, 2.0, 0.0), (0.0, 2.0, 0.0)])
        mesh.CreateFaceVertexCountsAttr([3, 3])
        mesh.CreateFaceVertexIndicesAttr([0, 1, 2, 0, 2, 3])
        mesh.GetPrim().AddAppliedSchema("PhysicsSurfaceDeformableSimAPI")
        mesh.GetPrim().CreateAttribute("physics:restShapePoints", Sdf.ValueTypeNames.Point3fArray).Set(
            [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 1.0, 0.0), (0.0, 0.0, 0.0), (2.0, 2.0, 0.0), (0.0, 1.0, 0.0)]
        )
        mesh.GetPrim().CreateAttribute("physics:restTriVertexIndices", Sdf.ValueTypeNames.Int3Array).Set(
            [(0, 1, 2), (3, 4, 5)]
        )

        mesh.GetPrim().AddAppliedSchema("PhysicsCollisionAPI")
        builder = newton.ModelBuilder()
        _bind_deformable_material(stage, mesh.GetPrim(), "/World/Mat", density=1.0)
        _author_deformable_element_array(mesh.GetPrim(), "thicknesses", [1.0], "constant")
        with self.assertLogs("newton", level="WARNING") as caught:
            builder.add_usd(stage)
        self.assertTrue(any("inconsistent lengths" in m and "using the last" in m for m in caught.output))

        interior_edge = next(i for i, edge in enumerate(builder.edge_indices) if edge[1] != -1)
        self.assertAlmostEqual(builder.edge_rest_length[interior_edge], math.sqrt(8.0), places=6)
        np.testing.assert_allclose(builder.tri_areas, [0.5, 1.0])
        self.assertAlmostEqual(sum(builder.particle_mass), 1.5)
        np.testing.assert_allclose(builder.tri_poses[0], [[1.0, -1.0], [0.0, 1.0]])

    def test_disjoint_rest_panels_at_offsets_keep_quiet(self):
        """Accept float32 rest copies of one edge placed in different panels without a length warning."""
        from pxr import Sdf, UsdGeom

        stage = _deformable_stage()
        mesh = UsdGeom.Mesh.Define(stage, "/World/Cloth")
        mesh.CreatePointsAttr([(0.0, 0.0, 0.0), (0.02, 0.0, 0.0), (0.02, 0.02, 0.0), (0.0, 0.02, 0.0)])
        mesh.CreateFaceVertexCountsAttr([3, 3])
        mesh.CreateFaceVertexIndicesAttr([0, 1, 2, 0, 2, 3])
        mesh.GetPrim().AddAppliedSchema("PhysicsSurfaceDeformableSimAPI")
        mesh.GetPrim().AddAppliedSchema("PhysicsCollisionAPI")
        _author_deformable_element_array(mesh.GetPrim(), "thicknesses", [0.001], "constant")
        # Both rest copies of the shared edge are 0.01 * sqrt(2) long but round differently in float32.
        mesh.GetPrim().CreateAttribute("physics:restShapePoints", Sdf.ValueTypeNames.Point3fArray).Set(
            [
                (0.37, 0.61, 0.0),
                (0.38, 0.61, 0.0),
                (0.38, 0.62, 0.0),
                (0.83, 0.29, 0.0),
                (0.84, 0.30, 0.0),
                (0.83, 0.30, 0.0),
            ]
        )
        mesh.GetPrim().CreateAttribute("physics:restTriVertexIndices", Sdf.ValueTypeNames.Int3Array).Set(
            [(0, 1, 2), (3, 4, 5)]
        )

        builder = newton.ModelBuilder()
        with warnings.catch_warnings(), self.assertNoLogs("newton", level="WARNING"):
            warnings.simplefilter("error")
            builder.add_usd(stage)

        interior_edge = next(i for i, edge in enumerate(builder.edge_indices) if edge[1] != -1)
        self.assertAlmostEqual(builder.edge_rest_length[interior_edge], 0.01 * math.sqrt(2.0), places=6)
        np.testing.assert_allclose(builder.tri_areas, [0.5e-4, 0.5e-4], rtol=1.0e-5)

    def test_cloth_rest_bend_overrides_keep_valid_entries(self):
        """Log bad bend entries and duplicate pairs without losing valid angles."""
        from pxr import Sdf

        stage = _deformable_stage()
        mesh = _add_cloth_mesh(stage, "/World/Cloth")
        _author_deformable_element_array(mesh.GetPrim(), "thicknesses", [0.001], "constant")
        mesh.GetPrim().CreateAttribute("physics:restAdjTriPairs", Sdf.ValueTypeNames.Int2Array).Set(
            [(0, 1), (7, 9), (1, 0), (0, 1)]
        )
        mesh.GetPrim().CreateAttribute("physics:restBendAngles", Sdf.ValueTypeNames.FloatArray).Set(
            [30.0, 10.0, -75.0, float("nan")]
        )
        builder = newton.ModelBuilder()
        with self.assertLogs("newton", level="WARNING") as caught:
            builder.add_usd(stage)
        messages = caught.output
        for reason in ("non-adjacent", "non-finite", "duplicate"):
            self.assertTrue(any("/World/Cloth" in m and reason in m for m in messages), reason)
        interior_edge = next(i for i, edge in enumerate(builder.edge_indices) if edge[1] != -1)
        self.assertAlmostEqual(builder.edge_rest_angle[interior_edge], math.radians(-75.0), places=6)

        # A malformed pair must not silently select another edge through integer truncation.
        prim = mesh.GetPrim()
        prim.RemoveProperty("physics:restAdjTriPairs")
        prim.CreateAttribute("physics:restAdjTriPairs", Sdf.ValueTypeNames.FloatArray).Set([0.9, 1.9])
        prim.GetAttribute("physics:restBendAngles").Set([75.0])
        builder = newton.ModelBuilder()
        with warnings.catch_warnings(), self.assertLogs("newton", level="WARNING") as caught:
            warnings.simplefilter("error")
            builder.add_usd(stage)
        self.assertTrue(any("invalid authored rest bend angles" in m for m in caught.output))
        np.testing.assert_array_equal(builder.edge_rest_angle, 0.0)

    def test_cloth_rest_shape_bend_uses_authored_pose(self):
        """Keep default-time rest corners aligned through handedness and reflection changes."""
        from pxr import Gf, Sdf, Usd, UsdGeom

        rest_points = np.asarray([(0.0, 0.0, 0.0), (2.0, 0.0, 0.0), (2.0, 1.0, 0.0), (0.0, 1.0, 1.0)])
        live_points = rest_points * 2.0
        live_points[:, 2] = 0.0
        triangles = np.asarray([(0, 1, 2), (0, 2, 3)])
        cases = [
            (False, False, False),
            *((True, left, reflected) for left in (False, True) for reflected in (False, True)),
        ]
        for disjoint, left_handed, reflected in cases:
            with self.subTest(disjoint=disjoint, left_handed=left_handed, reflected=reflected):
                stage = _deformable_stage()
                mesh = _add_cloth_mesh(stage, "/World/Cloth")
                mesh.GetPointsAttr().Set(live_points)
                scale = [-1.0 if reflected else 1.0, 1.0, 1.0]
                mesh.AddScaleOp().Set(Gf.Vec3d(*scale))
                if left_handed:
                    mesh.CreateOrientationAttr(UsdGeom.Tokens.leftHanded)
                prim = mesh.GetPrim()
                rest = prim.CreateAttribute("physics:restShapePoints", Sdf.ValueTypeNames.Point3fArray)
                rest.Set(rest_points[triangles].reshape(-1, 3) if disjoint else rest_points)
                rest.Set(live_points[triangles].reshape(-1, 3) if disjoint else live_points, Usd.TimeCode(1.0))
                if disjoint:
                    prim.CreateAttribute("physics:restTriVertexIndices", Sdf.ValueTypeNames.Int3Array).Set(
                        [(0, 1, 2), (3, 4, 5)]
                    )
                prim.CreateAttribute("physics:restBendAnglesDefault", Sdf.ValueTypeNames.Token).Set("restShape")
                _author_deformable_element_array(prim, "thicknesses", [0.001], "constant")

                # The builder's welded rest mesh supplies independent pose, length, and signed-angle expectations.
                reference = newton.ModelBuilder()
                reference.add_cloth_mesh(
                    pos=wp.vec3(0.0, 0.0, 0.0),
                    rot=wp.quat_identity(),
                    scale=1.0,
                    vel=wp.vec3(0.0, 0.0, 0.0),
                    vertices=(rest_points * scale).tolist(),
                    indices=(triangles[:, ::-1] if left_handed != reflected else triangles).reshape(-1).tolist(),
                    density=1.0,
                )
                interior = next(i for i, edge in enumerate(reference.edge_indices) if edge[1] != -1)
                self.assertGreater(abs(reference.edge_rest_angle[interior]), 0.1)

                # Seed existing geometry to check that rest assignments respect builder offsets.
                builder = newton.ModelBuilder()
                builder.add_builder(reference)
                builder.add_usd(stage)
                np.testing.assert_allclose(builder.particle_q[:4], rest_points * scale)
                np.testing.assert_allclose(builder.particle_q[4:], live_points * scale)
                for field in ("tri_poses", "tri_areas", "edge_rest_angle", "edge_rest_length"):
                    expected = np.asarray(getattr(reference, field))
                    np.testing.assert_allclose(
                        getattr(builder, field), np.concatenate((expected, expected)), atol=1.0e-7
                    )

    def test_coincident_cloth_faces_keep_rest_bending(self):
        """Preserve both sides of coincident faces for rest-shape dihedrals and bend overrides."""
        from pxr import Sdf

        points = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)]
        for second_face in ([0, 1, 2], [0, 2, 1]):
            stage = _deformable_stage()
            mesh = _add_cloth_mesh(stage, "/World/Cloth")
            mesh.GetPointsAttr().Set(points)
            mesh.GetFaceVertexIndicesAttr().Set([0, 1, 2, *second_face])
            prim = mesh.GetPrim()
            _author_deformable_element_array(prim, "thicknesses", [0.001], "constant")
            prim.CreateAttribute("physics:restBendAnglesDefault", Sdf.ValueTypeNames.Token).Set("restShape")
            reference = newton.ModelBuilder()
            reference.add_usd(stage)
            np.testing.assert_allclose(reference.edge_rest_angle, [math.pi] * 3)

            for mode in ("rest_shape", "override"):
                with self.subTest(second_face=second_face, mode=mode):
                    if mode == "rest_shape":
                        prim.CreateAttribute("physics:restShapePoints", Sdf.ValueTypeNames.Point3fArray).Set(points)
                        expected = reference.edge_rest_angle
                    else:
                        prim.RemoveProperty("physics:restShapePoints")
                        prim.CreateAttribute("physics:restAdjTriPairs", Sdf.ValueTypeNames.Int2Array).Set([(0, 1)])
                        prim.CreateAttribute("physics:restBendAngles", Sdf.ValueTypeNames.FloatArray).Set([75.0])
                        expected = [math.radians(75.0)] * 3
                    builder = newton.ModelBuilder()
                    with warnings.catch_warnings():
                        warnings.simplefilter("error")
                        builder.add_usd(stage)
                    np.testing.assert_allclose(builder.edge_rest_angle, expected)
                    np.testing.assert_allclose(builder.edge_rest_length, reference.edge_rest_length)
                    np.testing.assert_allclose(builder.particle_q, points)

    def test_invalid_live_cloth_rest_basis_skips_without_partial_state(self):
        """Contain a live triangle that passes the area check but fails rest-basis validation."""
        from pxr import UsdGeom

        stage = _deformable_stage()
        bad = UsdGeom.Mesh.Define(stage, "/World/Bad")
        bad.CreatePointsAttr([(0.0, 0.0, 0.0), (1.0e-13, 0.0, 0.0), (0.0, 1.0e5, 0.0)])
        bad.CreateFaceVertexCountsAttr([3])
        bad.CreateFaceVertexIndicesAttr([0, 1, 2])
        bad.GetPrim().AddAppliedSchema("PhysicsSurfaceDeformableSimAPI")
        good = _add_cloth_mesh(stage, "/World/Good")
        _author_deformable_element_array(good.GetPrim(), "thicknesses", [0.001], "constant")
        builder = newton.ModelBuilder()
        with self.assertLogs("newton", level="WARNING") as caught:
            builder.add_usd(stage)
        self.assertTrue(any("/World/Bad" in m and "skipping" in m for m in caught.output))
        self.assertEqual(group_labels(builder, "cloth"), ["/World/Good"])
        self.assertEqual(builder.particle_count, 4)
        self.assertEqual(builder.tri_count, 2)

    def test_cloth_left_handed_orientation_flips_winding(self):
        """Verify that left-handed cloth flips winding like the rigid-mesh path."""
        from pxr import UsdGeom

        stage = _deformable_stage(up_axis="y")
        mesh = UsdGeom.Mesh.Define(stage, "/World/Cloth")
        mesh.CreatePointsAttr([(0.0, 0.0, 1.0), (1.0, 0.0, 1.0), (1.0, 1.0, 1.0), (0.0, 1.0, 1.0)])
        mesh.CreateFaceVertexCountsAttr([4])
        mesh.CreateFaceVertexIndicesAttr([0, 1, 2, 3])
        mesh.CreateOrientationAttr(UsdGeom.Tokens.leftHanded)
        mesh.GetPrim().AddAppliedSchema("PhysicsSurfaceDeformableSimAPI")
        mesh.GetPrim().AddAppliedSchema("PhysicsCollisionAPI")
        _author_deformable_element_array(mesh.GetPrim(), "thicknesses", [0.001], "constant")

        builder = newton.ModelBuilder()
        builder.add_usd(stage)
        # The right-handed fan would give the first triangle (0, 1, 2); left-handed reverses it.
        self.assertEqual(list(builder.tri_indices[0]), [2, 1, 0])

    def test_malformed_cloth_topology_logs_and_skips(self):
        """Log malformed cloth topology and import a valid sibling without partial state."""
        from pxr import UsdGeom

        points = [(0.0, 0.0, 1.0), (1.0, 0.0, 1.0), (1.0, 1.0, 1.0), (0.0, 1.0, 1.0)]
        cases = {
            "missing_topology": ([], [], "missing points / topology"),
            "short_face": ([2], [0, 1], "fewer than 3 vertices"),
            "count_index_mismatch": ([3, 3], [0, 1, 2, 0, 2], "!= faceVertexIndices length"),
            "index_out_of_range": ([3, 3], [0, 1, 2, 0, 2, 9], "outside the 4-point array"),
        }
        for name, (face_counts, face_indices, message) in cases.items():
            with self.subTest(name):
                stage = _deformable_stage()
                mesh = UsdGeom.Mesh.Define(stage, "/World/Cloth")
                mesh.CreatePointsAttr(points)
                mesh.CreateFaceVertexCountsAttr(face_counts)
                mesh.CreateFaceVertexIndicesAttr(face_indices)
                mesh.GetPrim().AddAppliedSchema("PhysicsSurfaceDeformableSimAPI")
                mesh.GetPrim().AddAppliedSchema("PhysicsCollisionAPI")

                good = _add_cloth_mesh(stage, "/World/Good")
                _author_deformable_element_array(good.GetPrim(), "thicknesses", [0.001], "constant")
                builder = newton.ModelBuilder()
                with warnings.catch_warnings(), self.assertLogs("newton", level="WARNING") as caught:
                    warnings.simplefilter("error", NewtonWarning)
                    result = builder.add_usd(stage, return_deformable_results=True)
                self.assertEqual(len(caught.output), 1)
                self.assertIn("/World/Cloth", caught.output[0])
                self.assertIn(message, caught.output[0])
                self.assertEqual(group_labels(builder, "cloth"), ["/World/Good"])
                self.assertNotIn("/World/Cloth", result["path_cloth_map"])
                self.assertEqual(builder.particle_count, 4)
                self.assertEqual(builder.tri_count, 2)

    def test_current_surface_structural_stiffness_is_thickness_independent(self):
        """Preserve current surface stiffnesses when the authored thickness changes."""
        stage = _deformable_stage(up_axis="y")
        stretch, shear, bend = 1234.0, 456.0, 37.0
        for name, thickness in (("Thin", 0.001), ("Thick", 0.01)):
            cloth = _add_cloth_mesh(stage, f"/World/{name}")
            _bind_deformable_material(
                stage,
                cloth.GetPrim(),
                f"/World/{name}Mat",
                surfaceStretchStiffness=stretch,
                surfaceShearStiffness=shear,
                surfaceBendStiffness=bend,
                density=900.0,
            )
            _author_deformable_element_array(cloth.GetPrim(), "thicknesses", [thickness], "constant")

        builder = newton.ModelBuilder()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = builder.add_usd(stage, return_deformable_results=True)
        messages = [str(w.message) for w in caught]

        for name in ("Thin", "Thick"):
            path = f"/World/{name}"
            tri_start, _ = group_range(builder, "cloth", path, "tri")
            edge_start, _ = group_range(builder, "cloth", path, "edge")
            self.assertAlmostEqual(builder.tri_materials[tri_start][0], stretch)
            self.assertEqual(builder.tri_materials[tri_start][1], 0.0)
            self.assertAlmostEqual(builder.edge_bending_properties[edge_start][0], bend)
            self.assertEqual(
                result["path_cloth_attrs"][path]["material"]["surfaceShearStiffness"],
                shear,
            )
            self.assertTrue(
                any(path in message and "surfaceShearStiffness is not applied" in message for message in messages)
            )

    def test_surface_missing_structural_modes_use_isotropic_fallback(self):
        """Derive each missing surface mode from the isotropic material independently."""
        stage = _deformable_stage(up_axis="y")
        youngs, poissons, thickness = 2.0e6, 0.25, 0.02
        derived = _add_cloth_mesh(stage, "/World/Derived")
        _bind_deformable_material(
            stage,
            derived.GetPrim(),
            "/World/DerivedMat",
            youngsModulus=youngs,
            poissonsRatio=poissons,
        )
        _author_deformable_element_array(derived.GetPrim(), "thicknesses", [thickness], "constant")
        overridden = _add_cloth_mesh(stage, "/World/Overridden")
        _bind_deformable_material(
            stage,
            overridden.GetPrim(),
            "/World/OverriddenMat",
            youngsModulus=youngs,
            poissonsRatio=poissons,
            surfaceStretchStiffness=77.0,
        )
        _author_deformable_element_array(overridden.GetPrim(), "thicknesses", [thickness], "constant")

        builder = newton.ModelBuilder()
        builder.add_usd(stage)

        derived_tri, _ = group_range(builder, "cloth", "/World/Derived", "tri")
        derived_edge, _ = group_range(builder, "cloth", "/World/Derived", "edge")
        self.assertAlmostEqual(builder.tri_materials[derived_tri][0], 42666.666666666664, delta=0.05)
        self.assertAlmostEqual(builder.edge_bending_properties[derived_edge][0], 1.4222222222222223, delta=2.0e-6)

        overridden_tri, _ = group_range(builder, "cloth", "/World/Overridden", "tri")
        overridden_edge, _ = group_range(builder, "cloth", "/World/Overridden", "edge")
        self.assertEqual(builder.tri_materials[overridden_tri][0], 77.0)
        self.assertAlmostEqual(builder.edge_bending_properties[overridden_edge][0], 1.4222222222222223, delta=2.0e-6)

    def test_current_surface_material_uses_proposal_fallbacks(self):
        """Use the proposal thickness and isotropic fallbacks for a current surface material."""
        stage = _deformable_stage(up_axis="y")
        cloth = _add_cloth_mesh(stage, "/World/Cloth")
        _bind_deformable_material(stage, cloth.GetPrim(), "/World/Mat", density=1000.0)

        builder = newton.ModelBuilder()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            builder.add_usd(stage)

        messages = [str(w.message) for w in caught]
        fallback_notices = [message for message in messages if "previous 2 mm default" in message]
        self.assertEqual(len(fallback_notices), 1)
        self.assertIn("0.001 stage units", fallback_notices[0])
        self.assertIn("thicknesses = [0.002]", fallback_notices[0])
        self.assertIn("thicknesses:elementType", fallback_notices[0])
        particle_start, particle_end = group_range(builder, "cloth", "/World/Cloth", "particle")
        tri_start, _ = group_range(builder, "cloth", "/World/Cloth", "tri")
        edge_start, _ = group_range(builder, "cloth", "/World/Cloth", "edge")
        self.assertAlmostEqual(sum(builder.particle_mass[particle_start:particle_end]), 1.0, places=5)
        self.assertAlmostEqual(builder.particle_radius[particle_start], 0.0005, places=7)
        self.assertAlmostEqual(builder.tri_materials[tri_start][0], 1098.901098901099, delta=2.0e-4)
        self.assertAlmostEqual(builder.edge_bending_properties[edge_start][0], 9.15750915750916e-5, delta=2.0e-10)

    def test_surface_material_validation_uses_independent_fallbacks(self):
        """Drop malformed surface values and derive their modes from valid fallbacks."""
        stage = _deformable_stage(up_axis="y")
        cloth = _add_cloth_mesh(stage, "/World/Cloth")
        _bind_deformable_material(
            stage,
            cloth.GetPrim(),
            "/World/Mat",
            youngsModulus=float("-inf"),
            poissonsRatio=1.0,
            surfaceStretchStiffness=-1.0,
        )

        builder = newton.ModelBuilder()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            builder.add_usd(stage)
        messages = [str(w.message) for w in caught]

        self.assertEqual(sum("invalid physics:poissonsRatio" in message for message in messages), 1)
        self.assertEqual(sum("invalid physics:surfaceStretchStiffness" in message for message in messages), 1)
        tri_start, _ = group_range(builder, "cloth", "/World/Cloth", "tri")
        self.assertAlmostEqual(builder.tri_materials[tri_start][0], 1098.901098901099, delta=2.0e-4)

    def test_surface_material_clamps_legacy_high_poissons_ratio(self):
        """Use the same warned compatibility approximation as volume materials."""
        stage = _deformable_stage(up_axis="y")
        cloth = _add_cloth_mesh(stage, "/World/Cloth")
        youngs = 300000.0
        thickness = 0.02
        _bind_deformable_material(
            stage,
            cloth.GetPrim(),
            "/World/Mat",
            youngsModulus=youngs,
            poissonsRatio=0.6,
        )
        _author_deformable_element_array(cloth.GetPrim(), "thicknesses", [thickness], "constant")

        builder = newton.ModelBuilder()
        with self.assertWarnsRegex(UserWarning, "outside the proposal range.*0.499.*compatibility"):
            result = builder.add_usd(stage, return_deformable_results=True)

        tri_start, _ = group_range(builder, "cloth", "/World/Cloth", "tri")
        expected_stretch = youngs * thickness / (1.0 - 0.499**2)
        self.assertAlmostEqual(builder.tri_materials[tri_start][0], expected_stretch, delta=0.01)
        self.assertEqual(result["path_cloth_attrs"]["/World/Cloth"]["material"]["poissonsRatio"], 0.499)

    def test_surface_vendor_namespace_material_needs_resolver(self):
        """Read vendor-namespaced surface attributes only through a compatibility resolver."""
        stage = _deformable_stage(up_axis="y")
        cloth = _add_cloth_mesh(stage, "/World/Cloth")
        _bind_deformable_material(
            stage,
            cloth.GetPrim(),
            "/World/Mat",
            namespace="omniphysics",
            surfaceStretchStiffness=77.0,
        )
        _author_deformable_element_array(cloth.GetPrim(), "thicknesses", [0.02], "constant")

        builder_default = newton.ModelBuilder()
        builder_default.add_usd(stage)
        default_tri, _ = group_range(builder_default, "cloth", "/World/Cloth", "tri")
        self.assertNotAlmostEqual(builder_default.tri_materials[default_tri][0], 77.0)
        self.assertAlmostEqual(builder_default.particle_radius[0], 0.01, places=7)

        builder_compat = newton.ModelBuilder()
        builder_compat.add_usd(stage, schema_resolvers=[SchemaResolverPhysx()])
        compat_tri, _ = group_range(builder_compat, "cloth", "/World/Cloth", "tri")
        self.assertAlmostEqual(builder_compat.tri_materials[compat_tri][0], 77.0)
        self.assertAlmostEqual(builder_compat.particle_radius[0], 0.01, places=7)

    def test_surface_material_attr_authored_on_geometry_warns(self):
        """Warn when a current surface material property is authored on geometry."""
        from pxr import Sdf

        stage = _deformable_stage(up_axis="y")
        cloth = _add_cloth_mesh(stage, "/World/Cloth")
        cloth.GetPrim().CreateAttribute("physics:surfaceStretchStiffness", Sdf.ValueTypeNames.Float).Set(77.0)

        builder = newton.ModelBuilder()
        with self.assertWarnsRegex(UserWarning, "surfaceStretchStiffness.*authored on the geometry"):
            builder.add_usd(stage)

    def test_legacy_geometry_thickness_warning_names_array_replacement(self):
        """Direct legacy geometry thickness users to the typed thicknesses array."""
        from pxr import Sdf

        stage = _deformable_stage(up_axis="y")
        cloth = _add_cloth_mesh(stage, "/World/Cloth")
        cloth.GetPrim().CreateAttribute("physics:surfaceThickness", Sdf.ValueTypeNames.Float).Set(0.02)

        builder = newton.ModelBuilder()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            builder.add_usd(stage)
        messages = [str(warning.message) for warning in caught]
        self.assertTrue(
            any(
                warning.category is newton.exceptions.NewtonDeprecationWarning
                and "thicknesses" in str(warning.message)
                and "thicknesses:elementType" in str(warning.message)
                for warning in caught
            )
        )
        fallback_notices = [message for message in messages if "no valid surface thickness" in message]
        self.assertEqual(len(fallback_notices), 1)
        self.assertIn("/World/Cloth:", fallback_notices[0])
        self.assertAlmostEqual(builder.particle_radius[0], 0.0005, places=7)

    def test_legacy_surface_material_retains_old_units_during_deprecation(self):
        """Preserve the old surface modulus conversion while warning users to migrate."""
        stage = _deformable_stage(up_axis="y")
        cloth = _add_cloth_mesh(stage, "/World/Cloth")
        _bind_deformable_material(
            stage,
            cloth.GetPrim(),
            "/World/Mat",
            thickness=0.02,
            stretchStiffness=2000.0,
            bendStiffness=300.0,
            density=1000.0,
        )

        builder = newton.ModelBuilder()
        with self.assertWarnsRegex(DeprecationWarning, "unprefixed surface material attributes"):
            builder.add_usd(stage)

        tri_start, _ = group_range(builder, "cloth", "/World/Cloth", "tri")
        edge_start, _ = group_range(builder, "cloth", "/World/Cloth", "edge")
        self.assertAlmostEqual(builder.tri_materials[tri_start][0], 40.0, delta=1.0e-5)
        self.assertAlmostEqual(builder.edge_bending_properties[edge_start][0], 0.0024, delta=1.0e-9)

    def test_cloth_material_maps_to_isotropic_membrane(self):
        """Map deprecated surface moduli to Newton's isotropic membrane during migration."""
        stage = _deformable_stage(up_axis="y")
        mesh = _add_cloth_mesh(stage, "/World/ClothA")
        stretch, shear, bend = 1.0e3, 5.0e2, 2.0e1  # distinct stretch != shear
        # thickness=1 makes the modulus -> membrane conversion (E*h, E*h^3) the identity,
        # so the assertions below pin the mapping itself, not the thickness scaling.
        _bind_deformable_material(
            stage,
            mesh.GetPrim(),
            "/World/MatA",
            stretchStiffness=stretch,
            shearStiffness=shear,
            bendStiffness=bend,
            thickness=1.0,
        )
        zero = _add_cloth_mesh(stage, "/World/ClothZero")
        _bind_deformable_material(
            stage,
            zero.GetPrim(),
            "/World/MatZero",
            stretchStiffness=0.0,
            bendStiffness=bend,
            thickness=1.0,
        )

        builder = newton.ModelBuilder()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = builder.add_usd(stage, return_deformable_results=True)
        messages = [str(w.message) for w in caught]
        # Only the material that authors shearStiffness warns, attributed to its prim path.
        self.assertTrue(any("/World/ClothA" in m and "shearStiffness is not applied" in m for m in messages))
        self.assertFalse(any("/World/ClothZero" in m and "shearStiffness" in m for m in messages))

        t0, _ = group_range(builder, "cloth", "/World/ClothA", "tri")
        e0, _ = group_range(builder, "cloth", "/World/ClothA", "edge")
        # stretchStiffness -> tri_ke (mu); tri_ka (lambda) = 0, not the builder default.
        self.assertAlmostEqual(builder.tri_materials[t0][0], stretch, delta=stretch * 1e-3)  # tri_ke (mu)
        self.assertEqual(builder.tri_materials[t0][1], 0.0)  # tri_ka (lambda): no proposal attribute
        self.assertAlmostEqual(builder.edge_bending_properties[e0][0], bend, delta=bend * 1e-3)
        # The unmapped shearStiffness survives for anisotropic solvers.
        self.assertAlmostEqual(result["path_cloth_attrs"]["/World/ClothA"]["material"]["shearStiffness"], shear)

        # Authored zero stretch stiffness maps to tri_ke = 0, not a default.
        tz, _ = group_range(builder, "cloth", "/World/ClothZero", "tri")
        self.assertEqual(builder.tri_materials[tz][0], 0.0)  # tri_ke (stretch)
        self.assertEqual(builder.tri_materials[tz][1], 0.0)  # tri_ka (area): no default leaks in
        self.assertEqual(result["path_cloth_attrs"]["/World/ClothZero"]["material"]["stretchStiffness"], 0.0)

    def test_cloth_default_thickness_converts_authored_density(self):
        """Convert volumetric density with the proposal's 1 mm thickness fallback."""
        stage = _deformable_stage()
        cloth = _add_cloth_mesh(stage, "/World/Cloth")
        _bind_deformable_material(stage, cloth.GetPrim(), "/World/Mat", density=1000.0)

        builder = newton.ModelBuilder()
        with self.assertWarnsRegex(UserWarning, "/World/Cloth:.*1 mm physical fallback"):
            builder.add_usd(stage)

        p0, p1 = group_range(builder, "cloth", "/World/Cloth", "particle")
        # Unit quad (area 1): mass = density * default thickness * area = 1000 * 0.001 = 1 kg.
        self.assertAlmostEqual(sum(builder.particle_mass[p0:p1]), 1.0, places=5)
        # The collision radius describes the same assumed shell: half the default thickness.
        self.assertAlmostEqual(builder.particle_radius[p0], 0.0005, places=7)

    def test_cloth_default_thickness_converts_body_and_base_material_density(self):
        """Verify default-thickness conversion for body and base-material density.

        Density from either the deformable body API or a base physics material is
        volumetric. Neither source can author a surface thickness, so without the
        default that value would be passed to add_cloth_mesh() as areal density
        (~500x too heavy at 1000 kg/m^3).
        """
        from pxr import Sdf, UsdShade

        stage = _deformable_stage()
        body_cloth = _add_cloth_mesh(stage, "/World/ClothBodyDensity")
        _apply_deformable_body_api(body_cloth.GetPrim(), density=1000.0)
        base_cloth = _add_cloth_mesh(stage, "/World/ClothBaseMat")
        # A plain rigid-style physics material: base PhysicsMaterialAPI density, no
        # surface-deformable material API.
        mat = UsdShade.Material.Define(stage, "/World/BaseMat")
        mat.GetPrim().AddAppliedSchema("PhysicsMaterialAPI")
        mat.GetPrim().CreateAttribute("physics:density", Sdf.ValueTypeNames.Float).Set(1000.0)
        UsdShade.MaterialBindingAPI.Apply(base_cloth.GetPrim()).Bind(mat, materialPurpose="physics")

        builder = newton.ModelBuilder()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            builder.add_usd(stage)
        fallback_notices = [
            str(warning.message) for warning in caught if "no valid surface thickness" in str(warning.message)
        ]
        self.assertEqual(len(fallback_notices), 2)
        for path in ("/World/ClothBodyDensity", "/World/ClothBaseMat"):
            self.assertTrue(any(message.startswith(f"{path}:") for message in fallback_notices))

        for path in ("/World/ClothBodyDensity", "/World/ClothBaseMat"):
            p0, p1 = group_range(builder, "cloth", path, "particle")
            # Unit quad (area 1): mass = density * default thickness * area = 1000 * 0.001 = 1 kg.
            self.assertAlmostEqual(sum(builder.particle_mass[p0:p1]), 1.0, places=5)
            # The collision radius describes the same assumed shell: half the default thickness.
            self.assertAlmostEqual(builder.particle_radius[p0], 0.0005, places=7)

    def test_cloth_thickness_density_and_radius(self):
        """Verify that resolved thickness controls cloth mass and collision radius.

        Use simulation-geometry thickness, the NewtonMassAPI shell fallback, or the
        AOUSD default to convert volumetric material density to areal density. Set the
        particle collision radius to half the thickness instead of the generic builder
        default, while keeping path_cloth_attrs.resolved_density solver-neutral and
        volumetric.
        """
        from pxr import Sdf

        thickness = 0.01
        stage = _deformable_stage(up_axis="y")
        thick = _add_cloth_mesh(stage, "/World/ClothThick")
        _bind_deformable_material(stage, thick.GetPrim(), "/World/MatThick", density=1000.0)
        _author_deformable_element_array(thick.GetPrim(), "thicknesses", [thickness], "constant")
        shell = _add_cloth_mesh(stage, "/World/ClothShell")
        # Material density only -- thickness is left to the shell mass model.
        _bind_deformable_material(stage, shell.GetPrim(), "/World/MatShell", density=1000.0)
        shell.GetPrim().AddAppliedSchema("NewtonMassAPI")
        shell.GetPrim().CreateAttribute("newton:massModel", Sdf.ValueTypeNames.Token).Set("shell")
        shell.GetPrim().CreateAttribute("newton:shellThickness", Sdf.ValueTypeNames.Float).Set(thickness)
        bare = _add_cloth_mesh(stage, "/World/ClothBare")
        _bind_deformable_material(stage, bare.GetPrim(), "/World/MatBare", density=1000.0)

        builder = newton.ModelBuilder()
        # The current surface material on the bare cloth uses AOUSD's 1 mm fallback.
        with self.assertWarnsRegex(UserWarning, "/World/ClothBare:.*1 mm physical fallback"):
            result = builder.add_usd(stage, return_deformable_results=True)

        def total_mass(path):
            p0, p1 = group_range(builder, "cloth", path, "particle")
            return sum(builder.particle_mass[p0:p1])

        # Mass scales with the resolved thickness: the bare cloth uses the 0.001 default,
        # so the authored 0.01 comes out exactly 10x heavier.
        m_bare = total_mass("/World/ClothBare")
        self.assertGreater(m_bare, 0.0)
        self.assertAlmostEqual(total_mass("/World/ClothThick") / m_bare, thickness / 0.001, places=4)
        # The NewtonMassAPI shell thickness areal-scales exactly like the material attribute.
        self.assertAlmostEqual(total_mass("/World/ClothShell") / m_bare, thickness / 0.001, places=4)

        # Volumetric density (1000), not the areal 1000 * thickness passed to add_cloth_mesh.
        self.assertEqual(result["path_cloth_attrs"]["/World/ClothThick"]["resolved_density"], 1000.0)

        # Collision radius is the shell's physical half-thickness (the proposal's physical
        # thickness), rather than the builder's generic default particle radius.
        p0, p1 = group_range(builder, "cloth", "/World/ClothThick", "particle")
        for i in range(p0, p1):
            self.assertAlmostEqual(builder.particle_radius[i], 0.5 * thickness, places=6)
        self.assertNotAlmostEqual(builder.particle_radius[p0], builder.default_particle_radius, places=6)

    def test_cloth_constant_geometry_thickness_controls_physics(self):
        """Apply constant simulation thickness to cloth mass, radius, and derived stiffness."""
        from pxr import Sdf, UsdGeom

        stage = _deformable_stage(up_axis="y")
        UsdGeom.SetStageMetersPerUnit(stage, 1.0)
        cloth = _add_cloth_mesh(stage, "/World/Cloth")
        _bind_deformable_material(
            stage,
            cloth.GetPrim(),
            "/World/Mat",
            density=1000.0,
            youngsModulus=1.0e6,
            poissonsRatio=0.3,
        )
        cloth.GetPrim().CreateAttribute("physics:thicknesses", Sdf.ValueTypeNames.FloatArray).Set([0.02])
        cloth.GetPrim().CreateAttribute("physics:thicknesses:elementType", Sdf.ValueTypeNames.Token).Set("constant")

        builder = newton.ModelBuilder()
        builder.add_usd(stage)

        self.assertAlmostEqual(sum(builder.particle_mass), 20.0, places=5)
        for radius in builder.particle_radius:
            self.assertAlmostEqual(radius, 0.01, places=7)
        expected_stretch = 1.0e6 * 0.02 / (1.0 - 0.3**2)
        expected_bend = 1.0e6 * 0.02**3 / (12.0 * (1.0 - 0.3**2))
        for material in builder.tri_materials:
            self.assertAlmostEqual(material[0], expected_stretch, delta=1.0e-3)
        for properties in builder.edge_bending_properties:
            self.assertAlmostEqual(properties[0], expected_bend, places=6)

    def test_cloth_face_thickness_and_constant_mass_lower_independently(self):
        """Lower face thicknesses and a constant total mass through their own element types."""
        from pxr import Sdf, UsdGeom

        stage = _deformable_stage(up_axis="y")
        UsdGeom.SetStageMetersPerUnit(stage, 1.0)
        cloth = _add_cloth_mesh(stage, "/World/Cloth")
        _bind_deformable_material(
            stage,
            cloth.GetPrim(),
            "/World/Mat",
            density=1000.0,
            youngsModulus=1.0e6,
            poissonsRatio=0.3,
        )
        cloth.GetPrim().CreateAttribute("physics:thicknesses", Sdf.ValueTypeNames.FloatArray).Set([0.01, 0.03])
        cloth.GetPrim().CreateAttribute("physics:thicknesses:elementType", Sdf.ValueTypeNames.Token).Set("face")
        cloth.GetPrim().CreateAttribute("physics:masses", Sdf.ValueTypeNames.FloatArray).Set([12.0])
        cloth.GetPrim().CreateAttribute("physics:masses:elementType", Sdf.ValueTypeNames.Token).Set("constant")

        builder = newton.ModelBuilder()
        builder.add_usd(stage)

        np.testing.assert_allclose(builder.particle_mass, [4.0, 1.0, 4.0, 3.0], atol=1.0e-6)
        np.testing.assert_allclose(builder.particle_radius, [0.01, 0.005, 0.01, 0.015], atol=1.0e-7)
        expected_stretch = [
            1.0e6 * 0.01 / (1.0 - 0.3**2),
            1.0e6 * 0.03 / (1.0 - 0.3**2),
        ]
        np.testing.assert_allclose([material[0] for material in builder.tri_materials], expected_stretch, rtol=1.0e-7)

        shared_edge = next(
            index
            for index, (_opposite_a, _opposite_b, edge_a, edge_b) in enumerate(builder.edge_indices)
            if {int(edge_a), int(edge_b)} == {0, 2}
        )
        expected_bend = 1.0e6 * 0.02**3 / (12.0 * (1.0 - 0.3**2))
        self.assertAlmostEqual(builder.edge_bending_properties[shared_edge][0], expected_bend, places=6)

    def test_cloth_point_thickness_samples_bend_on_edge_endpoints(self):
        """Average point thickness at edge endpoints when deriving bend stiffness."""
        stage = _deformable_stage(up_axis="y")
        cloth = _add_cloth_mesh(stage, "/World/Cloth")
        _bind_deformable_material(
            stage,
            cloth.GetPrim(),
            "/World/Mat",
            youngsModulus=1.0e6,
            poissonsRatio=0.3,
        )
        _author_deformable_element_array(cloth.GetPrim(), "thicknesses", [0.02, 0.10, 0.02, 0.20], "point")

        builder = newton.ModelBuilder()
        builder.add_usd(stage)

        shared_edge = next(
            index
            for index, (_opposite_a, _opposite_b, edge_a, edge_b) in enumerate(builder.edge_indices)
            if {int(edge_a), int(edge_b)} == {0, 2}
        )
        expected_bend = 1.0e6 * 0.02**3 / (12.0 * (1.0 - 0.3**2))
        self.assertAlmostEqual(builder.edge_bending_properties[shared_edge][0], expected_bend, places=6)

    def test_cloth_material_thickness_is_a_deprecated_fallback(self):
        """Prefer geometry thickness while warning for the removed material attribute."""
        from pxr import Sdf

        stage = _deformable_stage(up_axis="y")
        cloth = _add_cloth_mesh(stage, "/World/Cloth")
        _bind_deformable_material(
            stage,
            cloth.GetPrim(),
            "/World/Mat",
            density=1000.0,
            surfaceThickness=0.1,
        )
        cloth.GetPrim().CreateAttribute("physics:thicknesses", Sdf.ValueTypeNames.FloatArray).Set([0.02])
        cloth.GetPrim().CreateAttribute("physics:thicknesses:elementType", Sdf.ValueTypeNames.Token).Set("constant")

        builder = newton.ModelBuilder()
        with self.assertWarnsRegex(DeprecationWarning, "surfaceThickness.*thicknesses"):
            builder.add_usd(stage)

        self.assertAlmostEqual(sum(builder.particle_mass), 20.0, places=5)
        self.assertAlmostEqual(builder.particle_radius[0], 0.01, places=7)

    def test_cloth_thickness_requires_a_valid_element_type(self):
        """Ignore malformed thickness arrays and require a valid matching element type."""
        from pxr import Sdf

        cases = (
            ("scalar", None, 0.02, "must be an array of numeric values"),
            ("missing", None, [0.02], "requires physics:thicknesses:elementType"),
            ("unsupported", "vertex", [0.02], "invalid physics:thicknesses:elementType"),
            ("wrong_count", "face", [0.02], "element type 'face' count 2"),
            ("empty_with_type", "face", [], "is empty but physics:thicknesses:elementType"),
            ("type_without_array", "constant", None, "thicknesses:elementType.*thicknesses is not authored"),
            ("outside_float_range", "constant", [1.0e308], "outside the finite USD float range"),
        )
        for label, element_type, thicknesses, warning in cases:
            with self.subTest(kind=label):
                stage = _deformable_stage(up_axis="y")
                cloth = _add_cloth_mesh(stage, "/World/Cloth")
                if thicknesses is not None:
                    if label == "scalar":
                        value_type = Sdf.ValueTypeNames.Float
                    elif label == "outside_float_range":
                        value_type = Sdf.ValueTypeNames.DoubleArray
                    else:
                        value_type = Sdf.ValueTypeNames.FloatArray
                    cloth.GetPrim().CreateAttribute("physics:thicknesses", value_type).Set(thicknesses)
                if element_type is not None:
                    cloth.GetPrim().CreateAttribute("physics:thicknesses:elementType", Sdf.ValueTypeNames.Token).Set(
                        element_type
                    )

                builder = newton.ModelBuilder()
                with self.assertWarnsRegex(UserWarning, warning):
                    builder.add_usd(stage)
                for radius in builder.particle_radius:
                    self.assertAlmostEqual(radius, 0.0005, places=7)

    def test_cloth_rest_bend_default_is_flat_unless_rest_shape_is_requested(self):
        """Set missing cloth bend angles flat unless restShape derives them from the rest pose."""
        from pxr import Sdf, Usd

        points = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 1.0, 0.0), (0.0, 1.0, 1.0)]
        rest_points = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 1.0, 0.0), (0.0, 1.0, -0.5)]
        windings = {"consistent": [0, 1, 2, 0, 2, 3], "flipped": [0, 1, 2, 0, 3, 2]}
        # Invalid rest data must retain the migration notice; valid zero-angle overrides suppress it.
        rest_cases = {
            "rest_points": rest_points,
            "wrong_count": rest_points[:3],
            "nonfinite_rest": [*rest_points[:3], (0.0, 1.0, float("nan"))],
            "degenerate_rest": [(0.0, 0.0, 0.0)] * 4,
        }
        bend_cases = {
            "empty_pairs": ([], []),
            "mismatched_bends": ([(0, 1)], []),
            "invalid_pair": ([(7, 9)], [30.0]),
            "nonfinite_bend": ([(0, 1)], [float("nan")]),
            "zero_bend": ([(0, 1)], [0.0]),
            "mixed_bends": ([(0, 1), (7, 9)], [0.0, 30.0]),
        }
        log_reasons = {
            "wrong_count": "invalid cloth rest shape",
            "nonfinite_rest": "invalid cloth rest shape",
            "degenerate_rest": "invalid cloth rest shape",
            "mismatched_bends": "invalid authored rest bend angles",
            "invalid_pair": "non-adjacent face pair",
            "nonfinite_bend": "non-finite value",
            "mixed_bends": "non-adjacent face pair",
        }
        cases = [
            (None, "consistent", None),
            *((None, "consistent", authored) for authored in (*rest_cases, *bend_cases)),
            ("flat", "consistent", None),
            ("flat", "consistent", "wrong_count"),
            ("restShape", "consistent", "wrong_count"),
            ("invalid", "consistent", None),
            ("restShape", "consistent", None),
            ("restShape", "consistent", "rest_points"),
            ("restShape", "flipped", "rest_points"),
        ]
        for token, winding, authored in cases:
            with self.subTest(rest_bend_angles_default=token, winding=winding, authored=authored):
                face_indices = windings[winding]
                stage = _deformable_stage()
                cloth = _add_cloth_mesh(stage, "/World/Cloth")
                cloth.GetFaceVertexIndicesAttr().Set(face_indices)
                cloth.GetPointsAttr().Set(points)
                # Import must use these default-time points, not the planar animation sample.
                cloth.GetPointsAttr().Set(
                    [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 1.0, 0.0), (0.0, 1.0, 0.0)],
                    Usd.TimeCode(1.0),
                )
                cloth.GetPrim().CreateAttribute("physics:thicknesses", Sdf.ValueTypeNames.FloatArray).Set([0.01])
                cloth.GetPrim().CreateAttribute("physics:thicknesses:elementType", Sdf.ValueTypeNames.Token).Set(
                    "constant"
                )
                if token is not None:
                    cloth.GetPrim().CreateAttribute("physics:restBendAnglesDefault", Sdf.ValueTypeNames.Token).Set(
                        token
                    )
                if authored in rest_cases:
                    cloth.GetPrim().CreateAttribute("physics:restShapePoints", Sdf.ValueTypeNames.Point3fArray).Set(
                        rest_cases[authored]
                    )
                elif authored in bend_cases:
                    pairs, angles = bend_cases[authored]
                    cloth.GetPrim().CreateAttribute("physics:restAdjTriPairs", Sdf.ValueTypeNames.Int2Array).Set(pairs)
                    cloth.GetPrim().CreateAttribute("physics:restBendAngles", Sdf.ValueTypeNames.FloatArray).Set(angles)

                builder = newton.ModelBuilder()
                log_reason = (
                    "invalid physics:restBendAnglesDefault" if token == "invalid" else log_reasons.get(authored)
                )
                log_context = (
                    self.assertLogs("newton", level="WARNING")
                    if log_reason
                    else self.assertNoLogs("newton", level="WARNING")
                )
                with warnings.catch_warnings(record=True) as caught, log_context as logs:
                    warnings.simplefilter("always" if token is None else "error", NewtonWarning)
                    builder.add_usd(stage)
                if log_reason:
                    self.assertEqual(len(logs.output), 1)
                    self.assertIn("/World/Cloth", logs.output[0])
                    self.assertIn(log_reason, logs.output[0])
                    if token == "invalid":
                        self.assertIn("using 'flat'", logs.output[0])
                migration_notices = [
                    str(w.message) for w in caught if "replaces non-planar imported dihedral" in str(w.message)
                ]
                expect_notice = token is None and authored not in ("rest_points", "zero_bend", "mixed_bends")
                self.assertEqual(len(migration_notices), int(expect_notice))
                self.assertEqual(len(caught), int(expect_notice))
                if expect_notice:
                    self.assertIn("restShape", migration_notices[0])
                e0, e1 = group_range(builder, "cloth", "/World/Cloth", "edge")
                interior_edges = [
                    edge
                    for edge in range(e0, e1)
                    if builder.edge_indices[edge][0] != -1 and builder.edge_indices[edge][1] != -1
                ]
                self.assertEqual(len(interior_edges), 1)
                (edge,) = interior_edges
                if token != "restShape":
                    self.assertEqual(builder.edge_rest_angle[edge], 0.0)
                    continue
                # restShape derives the dihedral the builder would compute on the rest pose (the live
                # pose without one), with the same sign regardless of how the second face is wound.
                reference = newton.ModelBuilder()
                reference.add_cloth_mesh(
                    pos=wp.vec3(0.0, 0.0, 0.0),
                    rot=wp.quat_identity(),
                    scale=1.0,
                    vel=wp.vec3(0.0, 0.0, 0.0),
                    vertices=rest_points if authored == "rest_points" else points,
                    indices=face_indices,
                    density=1.0,
                )
                expected = reference.edge_rest_angle[edge - e0]
                self.assertGreater(abs(expected), 0.1)
                self.assertAlmostEqual(builder.edge_rest_angle[edge], expected, places=6)

    def test_cloth_rest_bend_pairs_index_polygon_faces(self):
        """Resolve restAdjTriPairs over authored faces on a fan-triangulated polygon mesh."""
        from pxr import Sdf, UsdGeom

        cases = {
            # Two quads: the shared face edge takes the angle; the fan diagonals stay flat.
            "two_quads": ([4, 4], [0, 1, 4, 3, 1, 2, 5, 4], {(1, 4)}),
            # A quad and a triangle sharing two edges: both shared edges take the angle.
            "two_shared_edges": ([4, 3], [0, 1, 4, 3, 3, 4, 1], {(1, 4), (3, 4)}),
        }
        for case, (face_counts, face_indices, shared_edges) in cases.items():
            with self.subTest(case=case):
                stage = _deformable_stage()
                mesh = UsdGeom.Mesh.Define(stage, "/World/Cloth")
                mesh.CreatePointsAttr(
                    [
                        (0.0, 0.0, 0.0),
                        (1.0, 0.0, 0.0),
                        (2.0, 0.0, 0.0),
                        (0.0, 1.0, 0.0),
                        (1.0, 1.0, 0.0),
                        (2.0, 1.0, 0.0),
                    ]
                )
                mesh.CreateFaceVertexCountsAttr(face_counts)
                mesh.CreateFaceVertexIndicesAttr(face_indices)
                mesh.GetPrim().AddAppliedSchema("PhysicsSurfaceDeformableSimAPI")
                mesh.GetPrim().AddAppliedSchema("PhysicsCollisionAPI")
                _author_deformable_element_array(mesh.GetPrim(), "thicknesses", [0.001], "constant")
                mesh.GetPrim().CreateAttribute("physics:restAdjTriPairs", Sdf.ValueTypeNames.Int2Array).Set(
                    [(1, 0), (0, 0)]
                )
                mesh.GetPrim().CreateAttribute("physics:restBendAngles", Sdf.ValueTypeNames.FloatArray).Set(
                    [45.0, 10.0]
                )

                builder = newton.ModelBuilder()
                with self.assertLogs("newton", level="WARNING") as caught:
                    builder.add_usd(stage)
                self.assertTrue(any("entry 1 contains non-adjacent" in m for m in caught.output))

                angles = {
                    tuple(sorted((int(edge[2]), int(edge[3])))): builder.edge_rest_angle[index]
                    for index, edge in enumerate(builder.edge_indices)
                    if edge[1] != -1
                }
                for shared_edge in shared_edges:
                    self.assertAlmostEqual(angles.pop(shared_edge), math.radians(45.0), places=6)
                self.assertEqual(set(angles.values()), {0.0})

    def test_planar_cloth_default_rest_bend_needs_no_migration_warning(self):
        """Avoid a migration notice when the flat fallback does not change the imported rest state."""
        stage = _deformable_stage()
        _add_cloth_mesh(stage, "/World/Cloth")

        builder = newton.ModelBuilder()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            builder.add_usd(stage)

        messages = [str(w.message) for w in caught]
        self.assertFalse(any("replaces non-planar imported dihedral" in message for message in messages))

    def test_degenerate_cloth_triangle_logs_and_skips(self):
        """Log and skip a zero-area cloth before builder mutation while importing valid siblings."""
        from pxr import UsdGeom

        stage = _deformable_stage()
        bad = UsdGeom.Mesh.Define(stage, "/World/Bad")
        bad.CreatePointsAttr([(0.0, 0.0, 1.0), (0.5, 0.0, 1.0), (1.0, 0.0, 1.0)])  # collinear
        bad.CreateFaceVertexCountsAttr([3])
        bad.CreateFaceVertexIndicesAttr([0, 1, 2])
        bad.GetPrim().AddAppliedSchema("PhysicsSurfaceDeformableSimAPI")
        bad.GetPrim().AddAppliedSchema("PhysicsCollisionAPI")
        good = _add_cloth_mesh(stage, "/World/Good")
        _author_deformable_element_array(good.GetPrim(), "thicknesses", [0.001], "constant")

        builder = newton.ModelBuilder()
        with self.assertLogs("newton", level="WARNING") as caught:
            result = builder.add_usd(stage, return_deformable_results=True)
        self.assertTrue(any("/World/Bad" in m and "degenerate" in m for m in caught.output))
        self.assertNotIn("/World/Bad", result["path_cloth_map"])
        self.assertIn("/World/Good", result["path_cloth_map"])
        self.assertEqual(builder.particle_count, 4)  # the good quad only
        self.assertEqual(builder.tri_count, 2)
        builder.finalize()

    def test_nonfinite_cloth_vertex_logs_and_skips(self):
        """Skip a cloth with a non-finite point without aborting sibling imports."""
        stage = _deformable_stage()
        bad = _add_cloth_mesh(stage, "/World/Bad")
        bad.GetPointsAttr().Set([(0.0, 0.0, 1.0), (1.0, 0.0, 1.0), (1.0, 1.0, 1.0), (0.0, 1.0, float("nan"))])
        good = _add_cloth_mesh(stage, "/World/Good")
        _author_deformable_element_array(good.GetPrim(), "thicknesses", [0.001], "constant")

        builder = newton.ModelBuilder()
        with self.assertLogs("newton", level="WARNING") as caught:
            result = builder.add_usd(stage, return_deformable_results=True)
        self.assertTrue(any("/World/Bad" in m and "non-finite" in m for m in caught.output))

        self.assertNotIn("/World/Bad", result["path_cloth_map"])
        self.assertNotIn("/World/Bad", result["path_cloth_attrs"])
        self.assertIn("/World/Good", result["path_cloth_map"])
        self.assertEqual(builder.particle_count, 4)
        self.assertEqual(builder.tri_count, 2)
        self.assertEqual(builder.edge_count, 5)
        self.assertEqual(group_labels(builder, "cloth"), ["/World/Good"])
        builder.finalize()

    def test_cloth_collision_limitation(self):
        """Newton cannot disable particle collision: a cloth without an enabled
        PhysicsCollisionAPI warns and imports colliding; an enabled one is silent."""
        from pxr import Sdf

        for case, expect_warning in (("none", True), ("enabled", False), ("disabled", True)):
            with self.subTest(case=case):
                stage = _deformable_stage()
                mesh = _add_cloth_mesh(stage, "/World/Cloth", collision=False)
                if case != "none":
                    mesh.GetPrim().AddAppliedSchema("PhysicsCollisionAPI")
                    if case == "disabled":
                        mesh.GetPrim().CreateAttribute("physics:collisionEnabled", Sdf.ValueTypeNames.Bool).Set(False)
                builder = newton.ModelBuilder()
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    builder.add_usd(stage)
                messages = [str(w.message) for w in caught]
                warned = any("cannot disable deformable particle collision" in m for m in messages)
                self.assertEqual(warned, expect_warning)
                self.assertEqual(builder.particle_count, 4)

    def test_nested_rigid_body_keeps_its_collider(self):
        """A rigid body nested under a deformable body is native content: its collider
        imports as a rigid shape and is neither excluded from native parsing nor
        claimed as a dedicated deformable collider."""
        from pxr import UsdGeom, UsdPhysics

        stage = _deformable_stage()
        body = UsdGeom.Xform.Define(stage, "/World/Body").GetPrim()
        _apply_deformable_body_api(body)
        _add_cloth_mesh(stage, "/World/Body/Sim", collision=False)
        gizmo = UsdGeom.Xform.Define(stage, "/World/Body/Gizmo").GetPrim()
        UsdPhysics.RigidBodyAPI.Apply(gizmo)
        cube = UsdGeom.Cube.Define(stage, "/World/Body/Gizmo/Col").GetPrim()
        UsdPhysics.CollisionAPI.Apply(cube)

        builder = newton.ModelBuilder()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = builder.add_usd(stage, return_deformable_results=True)
        messages = [str(w.message) for w in caught]
        self.assertFalse(any("approximated" in m for m in messages))
        # The cloth authors no collider, so the limitation warning names it instead.
        self.assertTrue(any("cannot disable deformable particle collision" in m for m in messages))
        self.assertEqual(builder.particle_count, 4)
        self.assertEqual(builder.shape_count, 1)
        self.assertIn("/World/Body/Gizmo/Col", result["path_shape_map"])

    def test_ignored_dedicated_collider_is_absent_everywhere(self):
        """A dedicated collider matched by ignore_paths is as-if-absent: it creates no
        shape, does not gate deformable collision on, and emits no approximation warning."""
        from pxr import UsdGeom

        stage = _deformable_stage()
        body = UsdGeom.Xform.Define(stage, "/World/Body").GetPrim()
        _apply_deformable_body_api(body)
        _add_cable_curve(stage, "/World/Body/Sim", [(0.0, 0.0, 1.0), (0.1, 0.0, 1.0), (0.2, 0.0, 1.0)], collision=False)
        _add_mesh_prim(stage, "/World/Body/Collider", collision=True)

        builder = newton.ModelBuilder()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = builder.add_usd(stage, ignore_paths=[".*Collider"], return_deformable_results=True)
        messages = [str(w.message) for w in caught]
        self.assertFalse(any("approximated" in m for m in messages))
        self.assertEqual(builder.body_count, 2)  # the cable imported
        self.assertNotIn("/World/Body/Collider", result["path_shape_map"])
        collide = int(ShapeFlags.COLLIDE_SHAPES | ShapeFlags.COLLIDE_PARTICLES)
        for i in range(builder.shape_count):
            self.assertFalse(int(builder.shape_flags[i]) & collide, f"shape {i} collides")

    def test_collision_api_on_non_pointbased_prim_is_not_a_deformable_collider(self):
        """The proposal limits deformable colliders to UsdGeomPointBased prims: a
        CollisionAPI on a plain Xform inside the body neither gates collision on nor
        poisons native parsing of the subtree under it."""
        from pxr import UsdGeom, UsdPhysics

        stage = _deformable_stage()
        body = UsdGeom.Xform.Define(stage, "/World/Body").GetPrim()
        _apply_deformable_body_api(body)
        _add_cloth_mesh(stage, "/World/Body/Sim", collision=False)
        frame = UsdGeom.Xform.Define(stage, "/World/Body/Frame").GetPrim()
        UsdPhysics.CollisionAPI.Apply(frame)

        builder = newton.ModelBuilder()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            builder.add_usd(stage)
        messages = [str(w.message) for w in caught]
        self.assertFalse(any("approximated" in m for m in messages))
        self.assertTrue(any("cannot disable deformable particle collision" in m for m in messages))

    def test_deformable_body_with_rigid_body_api_imports_rigid(self):
        """The proposal forbids PhysicsDeformableBodyAPI on a prim that has RigidBodyAPI.
        The rigid interpretation wins (an invalid API application must not steal a working
        rigid body): the importer warns, skips the deformable, and the native loader keeps
        the rigid body and its collision shape."""
        from pxr import UsdPhysics

        stage = _deformable_stage()
        cloth = _add_cloth_mesh(stage, "/World/Cloth")  # sim API + enabled CollisionAPI
        _apply_deformable_body_api(cloth.GetPrim())
        UsdPhysics.RigidBodyAPI.Apply(cloth.GetPrim())

        builder = newton.ModelBuilder()
        with self.assertWarnsRegex(UserWarning, "/World/Cloth.*RigidBodyAPI"):
            result = builder.add_usd(stage, return_deformable_results=True)

        self.assertEqual(builder.particle_count, 0)
        self.assertNotIn("/World/Cloth", result["path_cloth_map"])
        self.assertIn("/World/Cloth", result["path_body_map"])
        self.assertEqual(builder.body_count, 1)
        self.assertGreaterEqual(builder.shape_count, 1)
        builder.finalize()

    def test_subset_physics_material_binding_warns(self):
        """Verify warnings for physics material bindings on UsdGeomSubsets.

        Per-element density and the proposal's per-edge bendStiffness are not
        supported: the importer resolves one material for the whole simulation geometry,
        so a subset physics binding warns instead of being dropped silently. Render-only
        subset bindings and the simulation prim's inherited binding stay silent.
        """
        from pxr import Sdf, UsdGeom, UsdShade

        stage = _deformable_stage()
        cloth = _add_cloth_mesh(stage, "/World/Cloth")
        _bind_deformable_material(stage, cloth.GetPrim(), "/World/Mat", density=1000.0)
        _author_deformable_element_array(cloth.GetPrim(), "thicknesses", [0.01], "constant")
        subset = UsdGeom.Subset.Define(stage, "/World/Cloth/Patch")
        subset.CreateElementTypeAttr().Set(UsdGeom.Tokens.face)
        subset.CreateIndicesAttr().Set([0])

        with self.subTest(binding="render_only"):
            render_mat = UsdShade.Material.Define(stage, "/World/RenderMat")
            UsdShade.MaterialBindingAPI.Apply(subset.GetPrim()).Bind(render_mat)
            builder = newton.ModelBuilder()
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                builder.add_usd(stage)
            self.assertFalse(any("GeomSubset" in str(w.message) for w in caught))

        with self.subTest(binding="physics"):
            patch_mat = UsdShade.Material.Define(stage, "/World/PatchMat")
            patch_mat.GetPrim().AddAppliedSchema("PhysicsSurfaceDeformableMaterialAPI")
            patch_mat.GetPrim().CreateAttribute("physics:density", Sdf.ValueTypeNames.Float).Set(2000.0)
            UsdShade.MaterialBindingAPI.Apply(subset.GetPrim()).Bind(patch_mat, materialPurpose="physics")
            builder = newton.ModelBuilder()
            with self.assertWarnsRegex(UserWarning, "/World/Cloth/Patch.*physics material"):
                builder.add_usd(stage)

    def test_every_dropped_dedicated_collider_warns(self):
        """Every enabled CollisionAPI on a non-sim prim of a deformable body is dropped
        (approximated by the simulation geometry), so every one must warn: the 2nd+
        dedicated collider, and dedicated colliders whose sim geometry carries its own
        CollisionAPI, were previously silent."""
        from pxr import UsdGeom

        with self.subTest(case="two_dedicated_colliders"):
            stage = _deformable_stage()
            body = UsdGeom.Xform.Define(stage, "/World/Body").GetPrim()
            _apply_deformable_body_api(body)
            _add_cloth_mesh(stage, "/World/Body/Sim", collision=False)
            _add_mesh_prim(stage, "/World/Body/ColA", collision=True)
            _add_mesh_prim(stage, "/World/Body/ColB", collision=True)

            builder = newton.ModelBuilder()
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                builder.add_usd(stage)
            approximations = [str(w.message) for w in caught if "approximated by the simulation" in str(w.message)]
            self.assertEqual(len(approximations), 2)
            self.assertTrue(any("/World/Body/ColA" in m for m in approximations))
            self.assertTrue(any("/World/Body/ColB" in m for m in approximations))

        with self.subTest(case="sim_geometry_has_own_collision"):
            stage = _deformable_stage()
            body = UsdGeom.Xform.Define(stage, "/World/Body").GetPrim()
            _apply_deformable_body_api(body)
            _add_cloth_mesh(stage, "/World/Body/Sim", collision=True)
            _add_mesh_prim(stage, "/World/Body/Col", collision=True)

            builder = newton.ModelBuilder()
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                builder.add_usd(stage)
            approximations = [str(w.message) for w in caught if "approximated by the simulation" in str(w.message)]
            self.assertEqual(len(approximations), 1)
            self.assertIn("/World/Body/Col", approximations[0])

    def test_dedicated_mesh_collider_owned_by_deformable_pass(self):
        """A dedicated UsdGeom.Mesh collider under a deformable body belongs to the
        deformable contract: it enables collision on the simulation geometry with the
        approximation warning, and must not also become a native rigid shape."""
        from pxr import UsdGeom

        stage = _deformable_stage()
        body = UsdGeom.Xform.Define(stage, "/World/Body").GetPrim()
        _apply_deformable_body_api(body)
        _add_cloth_mesh(stage, "/World/Body/Sim", collision=False)
        _add_mesh_prim(stage, "/World/Body/Collider", collision=True)

        builder = newton.ModelBuilder()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = builder.add_usd(stage, return_deformable_results=True)
        messages = [str(w.message) for w in caught]
        approximations = [m for m in messages if "approximated by the simulation geometry" in m]
        self.assertEqual(len(approximations), 1)
        self.assertIn("/World/Body/Collider", approximations[0])
        self.assertIn("/World/Body/Sim", approximations[0])

        self.assertEqual(builder.particle_count, 4)
        self.assertEqual(builder.shape_count, 0)
        self.assertNotIn("/World/Body/Collider", result["path_shape_map"])
        builder.finalize()

    def test_cloth_per_point_mass_policy(self):
        """Convert point masses through faces and retain the deprecated direct interpretation.

        Invalid arrays warn and fall back to density-derived masses.
        """
        from pxr import Sdf

        def import_cloth(masses, *, element_type="point"):
            stage = _deformable_stage(up_axis="y")
            mesh = _add_cloth_mesh(stage, "/World/Cloth")
            # Geometry thickness keeps the volumetric density convertible (no unrelated warning under
            # --strict-warnings); per-point masses take precedence over it either way.
            _bind_deformable_material(stage, mesh.GetPrim(), "/World/ClothMat", density=1000.0)
            _author_deformable_element_array(mesh.GetPrim(), "thicknesses", [0.1], "constant")
            mesh.GetPrim().CreateAttribute("physics:masses", Sdf.ValueTypeNames.FloatArray).Set(masses)
            if element_type is not None:
                mesh.GetPrim().CreateAttribute("physics:masses:elementType", Sdf.ValueTypeNames.Token).Set(element_type)
            builder = newton.ModelBuilder()
            builder.add_usd(stage)
            return builder

        with self.subTest(kind="valid"):
            builder = import_cloth([1.0, 2.0, 3.0, 4.0])
            np.testing.assert_allclose(
                [builder.particle_mass[i] for i in range(4)],
                [10.0 / 3.0, 4.0 / 3.0, 10.0 / 3.0, 2.0],
                atol=1.0e-6,
            )

        with self.subTest(kind="legacy_implicit_point"):
            with self.assertWarnsRegex(DeprecationWarning, "masses:elementType"):
                builder = import_cloth([1.0, 2.0, 3.0, 4.0], element_type=None)
            self.assertEqual([builder.particle_mass[i] for i in range(4)], [1.0, 2.0, 3.0, 4.0])

        for label, bad in (
            ("zero", [1.0, 0.0, 3.0, 4.0]),
            ("negative", [1.0, -2.0, 3.0, 4.0]),
            ("inf", [1.0, float("inf"), 3.0, 4.0]),
            ("nan", [1.0, float("nan"), 3.0, 4.0]),
        ):
            with self.subTest(kind=label):
                with self.assertWarnsRegex(
                    UserWarning,
                    r"invalid values.*ignoring the entire array.*lower-precedence mass sources",
                ):
                    builder = import_cloth(bad)
                masses = [builder.particle_mass[i] for i in range(4)]
                # Fell back to density-derived masses: all finite and strictly positive.
                for m in masses:
                    self.assertTrue(math.isfinite(m) and m > 0.0)
                self.assertNotEqual(masses, bad)

    def test_cloth_point_masses_ignore_unreferenced_points(self):
        """Import masses on referenced cloth points while ignoring orphan-point values."""
        stage = _deformable_stage()
        cloth = _add_cloth_mesh(stage, "/World/Cloth")
        points = list(cloth.GetPointsAttr().Get())
        cloth.GetPointsAttr().Set([*points, (2.0, 2.0, 1.0)])
        _author_deformable_element_array(cloth.GetPrim(), "masses", [2.0] * 5, "point")

        builder = newton.ModelBuilder()
        with self.assertWarnsRegex(UserWarning, "unreferenced point"):
            builder.add_usd(stage)

        p0, p1 = group_range(builder, "cloth", "/World/Cloth", "particle")
        masses = builder.particle_mass[p0:p1]
        self.assertEqual(len(masses), 5)
        self.assertAlmostEqual(sum(masses), 8.0)
        self.assertEqual(masses[-1], 0.0)

    def test_cloth_scale_bakes_and_reflection_flips_winding(self):
        """Verify full-affine scale baking and reflection-aware cloth winding.

        Bake xformOp:scale into particle positions. A non-uniform positive scale changes
        vertices without changing winding, while a reflective scale mirrors particles and
        flips triangle winding while preserving parity. A rotation-and-scale decomposition
        would silently drop that reflection.
        """
        from pxr import Gf, UsdGeom

        def import_cloth(scale):
            stage = _deformable_stage()  # Z up: avoid Y->Z axis conversion
            mesh = _add_cloth_mesh(stage, "/World/Cloth")  # points (0,0,1)(1,0,1)(1,1,1)(0,1,1)
            _author_deformable_element_array(mesh.GetPrim(), "thicknesses", [0.001], "constant")
            UsdGeom.Xformable(mesh).AddScaleOp().Set(Gf.Vec3d(*scale))
            builder = newton.ModelBuilder()
            builder.add_usd(stage)
            pq = np.array([list(builder.particle_q[i]) for i in range(builder.particle_count)])
            return pq, list(builder.tri_indices[0])

        # A non-uniform positive scale is baked into the vertices and keeps the winding.
        pq_pos, tri0_positive = import_cloth((2.0, 3.0, 4.0))
        expected = np.array([(0.0, 0.0, 4.0), (2.0, 0.0, 4.0), (2.0, 3.0, 4.0), (0.0, 3.0, 4.0)])
        np.testing.assert_allclose(pq_pos, expected, atol=1e-4)

        pq_neg, tri0_reflected = import_cloth((-1.0, 1.0, 1.0))
        # The full affine mirrors X; a decomposition would yield positive X (0, 1, 1, 0).
        np.testing.assert_allclose(pq_neg[:, 0], np.array([0.0, -1.0, -1.0, 0.0]), atol=1e-4)
        # The reflection reverses the first triangle's winding relative to the positive-scale import.
        self.assertEqual(tri0_reflected, tri0_positive[::-1], "reflective scale must flip triangle winding")


if __name__ == "__main__":
    unittest.main(verbosity=2)
