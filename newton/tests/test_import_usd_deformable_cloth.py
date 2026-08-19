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

import newton
from newton import ShapeFlags
from newton.tests._usd_deformable_test_utils import (
    _add_cable_curve,
    _add_cloth_mesh,
    _apply_deformable_body_api,
    _bind_deformable_material,
    _deformable_stage,
    group_labels,
    group_range,
)
from newton.tests.unittest_utils import USD_AVAILABLE


@unittest.skipUnless(USD_AVAILABLE, "Requires usd-core")
class TestUSDDeformableCloth(unittest.TestCase):
    """Surface-deformable (cloth) parsing into particles + FEM triangles + bending edges."""

    def test_cloth_quad_mesh_is_triangulated(self):
        """A quad-faced cloth mesh is fan-triangulated on import (n-gons are supported)."""
        from pxr import UsdGeom

        stage = _deformable_stage(up_axis="y")
        mesh = UsdGeom.Mesh.Define(stage, "/World/Cloth")
        mesh.CreatePointsAttr([(0.0, 0.0, 1.0), (1.0, 0.0, 1.0), (1.0, 1.0, 1.0), (0.0, 1.0, 1.0)])
        mesh.CreateFaceVertexCountsAttr([4])  # single quad face
        mesh.CreateFaceVertexIndicesAttr([0, 1, 2, 3])
        mesh.GetPrim().AddAppliedSchema("PhysicsSurfaceDeformableSimAPI")
        mesh.GetPrim().AddAppliedSchema("PhysicsCollisionAPI")

        builder = newton.ModelBuilder()
        builder.add_usd(stage)
        # 4 quad vertices stay 1:1 with particles.
        self.assertEqual(group_range(builder, "cloth", "/World/Cloth", "particle"), (0, 4))
        # The quad fan-triangulates to 2 triangles.
        self.assertEqual(group_range(builder, "cloth", "/World/Cloth", "tri"), (0, 2))
        self.assertEqual(builder.particle_count, 4)

    def test_cloth_left_handed_orientation_flips_winding(self):
        """A left-handed cloth mesh flips triangle winding, matching the rigid mesh path."""
        from pxr import UsdGeom

        stage = _deformable_stage(up_axis="y")
        mesh = UsdGeom.Mesh.Define(stage, "/World/Cloth")
        mesh.CreatePointsAttr([(0.0, 0.0, 1.0), (1.0, 0.0, 1.0), (1.0, 1.0, 1.0), (0.0, 1.0, 1.0)])
        mesh.CreateFaceVertexCountsAttr([4])
        mesh.CreateFaceVertexIndicesAttr([0, 1, 2, 3])
        mesh.CreateOrientationAttr(UsdGeom.Tokens.leftHanded)
        mesh.GetPrim().AddAppliedSchema("PhysicsSurfaceDeformableSimAPI")
        mesh.GetPrim().AddAppliedSchema("PhysicsCollisionAPI")

        builder = newton.ModelBuilder()
        builder.add_usd(stage)
        # The right-handed fan would give the first triangle (0, 1, 2); left-handed reverses it.
        self.assertEqual(list(builder.tri_indices[0]), [2, 1, 0])

    def test_malformed_cloth_topology_warns_and_skips(self):
        """Malformed cloth topology (short faces, count/index mismatch, out-of-range index)
        warns and skips the cloth before any builder mutation instead of crashing."""
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

                builder = newton.ModelBuilder()
                with self.assertWarnsRegex(UserWarning, message):
                    builder.add_usd(stage)
                self.assertEqual(group_labels(builder, "cloth"), [])
                self.assertEqual(builder.particle_count, 0)
                self.assertEqual(builder.tri_count, 0)

    def test_cloth_material_maps_to_isotropic_membrane(self):
        """Surface material -> isotropic membrane: stretchStiffness -> tri_ke (authored zero
        included -- the range is [0, inf)), bendStiffness -> edge bending. tri_ka
        (area-preservation/Poisson) is 0 since the proposal has no such attribute (the builder
        default would fabricate an unauthored area response). shearStiffness can't be
        represented independently: it warns but is preserved in path_cloth_attrs."""
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
        """A surface material that authors a volumetric density but no thickness gets the
        importer's default thickness for the areal conversion, and the assumed default is
        warned."""
        stage = _deformable_stage()
        cloth = _add_cloth_mesh(stage, "/World/Cloth")
        _bind_deformable_material(stage, cloth.GetPrim(), "/World/Mat", density=1000.0)

        builder = newton.ModelBuilder()
        with self.assertWarnsRegex(UserWarning, "assuming the default thickness"):
            builder.add_usd(stage)

        p0, p1 = group_range(builder, "cloth", "/World/Cloth", "particle")
        # Unit quad (area 1): mass = density * default thickness * area = 1000 * 0.002 = 2 kg.
        self.assertAlmostEqual(sum(builder.particle_mass[p0:p1]), 2.0, places=5)
        # The collision radius describes the same assumed shell: half the default thickness.
        self.assertAlmostEqual(builder.particle_radius[p0], 0.001, places=6)

    def test_cloth_default_thickness_converts_body_and_base_material_density(self):
        """A volumetric density resolved from the deformable body API or a base physics
        material still gets the default-thickness areal conversion. Neither source can
        author a surface thickness, so without the default the volumetric value would be
        passed to add_cloth_mesh() as areal density (~500x too heavy at 1000 kg/m^3)."""
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
        messages = [str(w.message) for w in caught]

        for path in ("/World/ClothBodyDensity", "/World/ClothBaseMat"):
            self.assertTrue(
                any(path in m and "assuming the default thickness" in m for m in messages),
                f"{path}: expected a default-thickness warning",
            )
            p0, p1 = group_range(builder, "cloth", path, "particle")
            # Unit quad (area 1): mass = density * default thickness * area = 1000 * 0.002 = 2 kg.
            self.assertAlmostEqual(sum(builder.particle_mass[p0:p1]), 2.0, places=5)
            # The collision radius describes the same assumed shell: half the default thickness.
            self.assertAlmostEqual(builder.particle_radius[p0], 0.001, places=6)

    def test_cloth_thickness_density_and_radius(self):
        """Surface thickness (material attribute, or NewtonMassAPI shell fallback when the
        material omits it) converts the volumetric material density to an areal density and
        sets the particle collision radius to half the thickness, while
        path_cloth_attrs.resolved_density stays the solver-neutral volumetric value."""
        from pxr import Sdf

        thickness = 0.01
        stage = _deformable_stage(up_axis="y")
        thick = _add_cloth_mesh(stage, "/World/ClothThick")
        _bind_deformable_material(stage, thick.GetPrim(), "/World/MatThick", density=1000.0, thickness=thickness)
        shell = _add_cloth_mesh(stage, "/World/ClothShell")
        # Material density only -- thickness is left to the shell mass model.
        _bind_deformable_material(stage, shell.GetPrim(), "/World/MatShell", density=1000.0)
        shell.GetPrim().AddAppliedSchema("NewtonMassAPI")
        shell.GetPrim().CreateAttribute("newton:massModel", Sdf.ValueTypeNames.Token).Set("shell")
        shell.GetPrim().CreateAttribute("newton:shellThickness", Sdf.ValueTypeNames.Float).Set(thickness)
        bare = _add_cloth_mesh(stage, "/World/ClothBare")
        _bind_deformable_material(stage, bare.GetPrim(), "/World/MatBare", density=1000.0)

        builder = newton.ModelBuilder()
        # The bare cloth resolves no thickness, so the importer assumes its default (2 mm)
        # for the surface conversion and must say so instead of assuming silently.
        with self.assertWarnsRegex(UserWarning, "/World/ClothBare.*assuming the default thickness"):
            result = builder.add_usd(stage, return_deformable_results=True)

        def total_mass(path):
            p0, p1 = group_range(builder, "cloth", path, "particle")
            return sum(builder.particle_mass[p0:p1])

        # Mass scales with the resolved thickness: the bare cloth uses the 0.002 default,
        # so the authored 0.01 comes out exactly 5x heavier.
        m_bare = total_mass("/World/ClothBare")
        self.assertGreater(m_bare, 0.0)
        self.assertAlmostEqual(total_mass("/World/ClothThick") / m_bare, thickness / 0.002, places=4)
        # The NewtonMassAPI shell thickness areal-scales exactly like the material attribute.
        self.assertAlmostEqual(total_mass("/World/ClothShell") / m_bare, thickness / 0.002, places=4)

        # Volumetric density (1000), not the areal 1000 * thickness passed to add_cloth_mesh.
        self.assertEqual(result["path_cloth_attrs"]["/World/ClothThick"]["resolved_density"], 1000.0)

        # Collision radius is the shell's physical half-thickness (the proposal's physical
        # thickness), rather than the builder's generic default particle radius.
        p0, p1 = group_range(builder, "cloth", "/World/ClothThick", "particle")
        for i in range(p0, p1):
            self.assertAlmostEqual(builder.particle_radius[i], 0.5 * thickness, places=6)
        self.assertNotAlmostEqual(builder.particle_radius[p0], builder.default_particle_radius, places=6)

    def test_degenerate_cloth_triangle_warns_and_skips(self):
        """A zero-area (collinear) triangle cannot form an FEM element: the cloth is
        skipped whole with a warning instead of importing particles without their
        triangle, and a valid cloth in the same stage still imports."""
        from pxr import UsdGeom

        stage = _deformable_stage()
        bad = UsdGeom.Mesh.Define(stage, "/World/Bad")
        bad.CreatePointsAttr([(0.0, 0.0, 1.0), (0.5, 0.0, 1.0), (1.0, 0.0, 1.0)])  # collinear
        bad.CreateFaceVertexCountsAttr([3])
        bad.CreateFaceVertexIndicesAttr([0, 1, 2])
        bad.GetPrim().AddAppliedSchema("PhysicsSurfaceDeformableSimAPI")
        bad.GetPrim().AddAppliedSchema("PhysicsCollisionAPI")
        _add_cloth_mesh(stage, "/World/Good")

        builder = newton.ModelBuilder()
        with self.assertWarnsRegex(UserWarning, "/World/Bad.*degenerate"):
            result = builder.add_usd(stage, return_deformable_results=True)
        self.assertNotIn("/World/Bad", result["path_cloth_map"])
        self.assertIn("/World/Good", result["path_cloth_map"])
        self.assertEqual(builder.particle_count, 4)  # the good quad only
        self.assertEqual(builder.tri_count, 2)
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
        collider = UsdGeom.Mesh.Define(stage, "/World/Body/Collider")
        collider.CreatePointsAttr([(0.0, 0.0, 1.0), (1.0, 0.0, 1.0), (0.0, 1.0, 1.0)])
        collider.CreateFaceVertexCountsAttr([3])
        collider.CreateFaceVertexIndicesAttr([0, 1, 2])
        collider.GetPrim().AddAppliedSchema("PhysicsCollisionAPI")

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

    @staticmethod
    def _add_triangle_mesh(stage, path, *, collision=False):
        """Author a minimal one-triangle GeomMesh, optionally as an enabled collider."""
        from pxr import UsdGeom

        mesh = UsdGeom.Mesh.Define(stage, path)
        mesh.CreatePointsAttr([(0.0, 0.0, 1.0), (1.0, 0.0, 1.0), (0.0, 1.0, 1.0)])
        mesh.CreateFaceVertexCountsAttr([3])
        mesh.CreateFaceVertexIndicesAttr([0, 1, 2])
        if collision:
            mesh.GetPrim().AddAppliedSchema("PhysicsCollisionAPI")
        return mesh

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
        """Per-UsdGeomSubset physics material bindings (per-element density, per-edge
        bendStiffness in the proposal) are not supported: the importer resolves one
        material for the whole simulation geometry, so a subset binding a physics
        material warns instead of being dropped silently. Render-only subset bindings
        and the sim prim's own inherited binding stay silent."""
        from pxr import Sdf, UsdGeom, UsdShade

        stage = _deformable_stage()
        cloth = _add_cloth_mesh(stage, "/World/Cloth")
        _bind_deformable_material(stage, cloth.GetPrim(), "/World/Mat", density=1000.0, thickness=0.01)
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
            self._add_triangle_mesh(stage, "/World/Body/ColA", collision=True)
            self._add_triangle_mesh(stage, "/World/Body/ColB", collision=True)

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
            self._add_triangle_mesh(stage, "/World/Body/Col", collision=True)

            builder = newton.ModelBuilder()
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                builder.add_usd(stage)
            approximations = [str(w.message) for w in caught if "approximated by the simulation" in str(w.message)]
            self.assertEqual(len(approximations), 1)
            self.assertIn("/World/Body/Col", approximations[0])

    def test_unembedded_graphics_geometry_warns_and_skips(self):
        """A PointBased prim under a deformable body that is neither the simulation
        geometry nor a collider should deform with the simulation geometry per the
        proposal; embedding is not implemented, so it is skipped with a warning.
        Importing it as a static shape would leave a frozen copy behind while the
        deformable moves away."""
        from pxr import UsdGeom

        stage = _deformable_stage()
        body = UsdGeom.Xform.Define(stage, "/World/Body").GetPrim()
        _apply_deformable_body_api(body)
        _add_cloth_mesh(stage, "/World/Body/Sim", collision=True)
        self._add_triangle_mesh(stage, "/World/Body/Graphics", collision=False)

        builder = newton.ModelBuilder()
        with self.assertWarnsRegex(UserWarning, "/World/Body/Graphics.*cannot deform"):
            result = builder.add_usd(stage, return_deformable_results=True)

        # The graphics mesh is excluded from the native loader: no shape imports for it.
        self.assertEqual(builder.shape_count, 0)
        self.assertNotIn("/World/Body/Graphics", result["path_shape_map"])

    def test_dedicated_mesh_collider_owned_by_deformable_pass(self):
        """A dedicated UsdGeom.Mesh collider under a deformable body belongs to the
        deformable contract: it enables collision on the simulation geometry with the
        approximation warning, and must not also become a native rigid shape."""
        from pxr import UsdGeom

        stage = _deformable_stage()
        body = UsdGeom.Xform.Define(stage, "/World/Body").GetPrim()
        _apply_deformable_body_api(body)
        _add_cloth_mesh(stage, "/World/Body/Sim", collision=False)
        collider = UsdGeom.Mesh.Define(stage, "/World/Body/Collider")
        collider.CreatePointsAttr([(0.0, 0.0, 1.0), (1.0, 0.0, 1.0), (0.0, 1.0, 1.0)])
        collider.CreateFaceVertexCountsAttr([3])
        collider.CreateFaceVertexIndicesAttr([0, 1, 2])
        collider.GetPrim().AddAppliedSchema("PhysicsCollisionAPI")

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
        """Valid physics:masses on the cloth Mesh set the particle masses directly; negative /
        inf / nan arrays warn and fall back to density-derived masses.

        Per-point masses have the highest precedence, so a poisoned array would otherwise be
        written straight into the particles.
        """
        from pxr import Sdf

        def import_cloth(masses):
            stage = _deformable_stage(up_axis="y")
            mesh = _add_cloth_mesh(stage, "/World/Cloth")
            # thickness keeps the volumetric density convertible (no unrelated warning under
            # --strict-warnings); per-point masses take precedence over it either way.
            _bind_deformable_material(stage, mesh.GetPrim(), "/World/ClothMat", density=1000.0, thickness=0.1)
            mesh.GetPrim().CreateAttribute("physics:masses", Sdf.ValueTypeNames.FloatArray).Set(masses)
            builder = newton.ModelBuilder()
            builder.add_usd(stage)
            return builder

        with self.subTest(kind="valid"):
            builder = import_cloth([1.0, 2.0, 3.0, 4.0])
            self.assertEqual([builder.particle_mass[i] for i in range(4)], [1.0, 2.0, 3.0, 4.0])

        for label, bad in (
            ("negative", [1.0, -2.0, 3.0, 4.0]),
            ("inf", [1.0, float("inf"), 3.0, 4.0]),
            ("nan", [1.0, float("nan"), 3.0, 4.0]),
        ):
            with self.subTest(kind=label):
                with self.assertWarnsRegex(UserWarning, "non-finite or negative"):
                    builder = import_cloth(bad)
                masses = [builder.particle_mass[i] for i in range(4)]
                # Fell back to density-derived masses: all finite and strictly positive.
                for m in masses:
                    self.assertTrue(math.isfinite(m) and m > 0.0)
                self.assertNotEqual(masses, bad)

    def test_cloth_scale_bakes_and_reflection_flips_winding(self):
        """xformOp:scale bakes into the particle positions as the full affine: a non-uniform
        positive scale scales the vertices without touching winding, and a reflective
        (negative) scale mirrors the particles (parity preserved) and flips triangle winding,
        which a rotation+scale decomposition would silently drop."""
        from pxr import Gf, UsdGeom

        def import_cloth(scale):
            stage = _deformable_stage()  # Z up: avoid Y->Z axis conversion
            mesh = _add_cloth_mesh(stage, "/World/Cloth")  # points (0,0,1)(1,0,1)(1,1,1)(0,1,1)
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
