# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Tests for USD volume-deformable (TetMesh) import: mass policy, body hierarchy, tet reorientation.

Cross-family happy-path, skip-policy, and lifecycle contracts live in
``test_import_usd_deformable_mixed`` and ``test_import_usd_deformable_groups``; this module
owns the volume-specific lowering (mass precedence, deformable-body hierarchy, transforms).
"""

import math
import os
import unittest
import warnings

import numpy as np
import warp as wp

import newton
from newton.tests._usd_deformable_test_utils import (
    _apply_deformable_body_api,
    _author_deformable_element_array,
    _bind_deformable_material,
    _deformable_stage,
    group_labels,
    group_range,
)
from newton.tests.unittest_utils import USD_AVAILABLE
from newton.usd import SchemaResolverPhysx


def _author_tet_cube(stage, path, z0=0.0):
    """Author a unit-cube TetMesh (8 vertices, 5 tetrahedra) with its base at ``z0``."""
    from pxr import UsdGeom

    c = [
        (0.0, 0.0, z0),
        (1.0, 0.0, z0),
        (1.0, 1.0, z0),
        (0.0, 1.0, z0),
        (0.0, 0.0, z0 + 1.0),
        (1.0, 0.0, z0 + 1.0),
        (1.0, 1.0, z0 + 1.0),
        (0.0, 1.0, z0 + 1.0),
    ]
    tets = [(0, 1, 3, 4), (1, 2, 3, 6), (1, 3, 4, 6), (1, 4, 5, 6), (3, 4, 6, 7)]
    tet = UsdGeom.TetMesh.Define(stage, path)
    tet.CreatePointsAttr(c)
    tet.CreateTetVertexIndicesAttr(tets)
    tet.GetPrim().AddAppliedSchema("PhysicsVolumeDeformableSimAPI")
    tet.GetPrim().AddAppliedSchema("PhysicsCollisionAPI")
    return tet


def _author_unit_tet(stage, path, *, sim_api=False, collision=None):
    """Author a single-tetrahedron TetMesh (volume 1/6), optionally marked as a simulation mesh.

    ``collision`` authors an enabled ``PhysicsCollisionAPI``; defaults to ``sim_api`` so
    deformable-marked fixtures represent the common colliding case, while bare TetMeshes
    keep the legacy import untouched.
    """
    from pxr import UsdGeom

    tet = UsdGeom.TetMesh.Define(stage, path)
    tet.CreatePointsAttr([(0.0, 0.0, 1.0), (1.0, 0.0, 1.0), (0.0, 1.0, 1.0), (0.0, 0.0, 2.0)])
    tet.CreateTetVertexIndicesAttr([(0, 1, 2, 3)])
    if sim_api:
        tet.GetPrim().AddAppliedSchema("PhysicsVolumeDeformableSimAPI")
    if sim_api if collision is None else collision:
        tet.GetPrim().AddAppliedSchema("PhysicsCollisionAPI")
    return tet


def _author_two_tet_wedge(stage, path):
    """Author a TetMesh of two tets that share a base triangle but have very different
    volumes, so density-based per-point masses must be non-uniform. Both tets are wound
    for positive signed volume.

    Vertices 0,1,2 form the shared base; vertex 3 is the apex of the large tet (V = 4/6)
    and vertex 4 the apex of the small tet (V = 1/6)."""
    from pxr import UsdGeom

    tet = UsdGeom.TetMesh.Define(stage, path)
    tet.CreatePointsAttr([(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 4.0), (0.0, 0.0, -1.0)])
    tet.CreateTetVertexIndicesAttr([(0, 1, 2, 3), (0, 2, 1, 4)])
    tet.GetPrim().AddAppliedSchema("PhysicsVolumeDeformableSimAPI")
    tet.GetPrim().AddAppliedSchema("PhysicsCollisionAPI")
    return tet


@unittest.skipUnless(USD_AVAILABLE, "Requires usd-core")
class TestUSDDeformableVolume(unittest.TestCase):
    """Volume (TetMesh) soft-body mass policy, body hierarchy, and transform baking."""

    def _build_soft(self, author_fn):
        stage = _deformable_stage()
        author_fn(stage)
        builder = newton.ModelBuilder()
        result = builder.add_usd(stage, return_deformable_results=True)
        return builder, result

    def test_volume_mass_precedence(self):
        """Verify proposal precedence for per-prim volume mass sources.

        Give physics:masses on the simulation geometry precedence over a body-mass
        override. Rescale the density-derived distribution proportionally for a body-mass
        override, preserving volume weighting (m_e = m_tot V_e / V_tot, followed by
        m_p = sum_{e in tau(p)} m_e / T), and give PhysicsDeformableBodyAPI.density
        precedence over bound-material density.
        """
        from pxr import Sdf

        body_mass = 10.0
        v_large, v_small = 4.0 / 6.0, 1.0 / 6.0  # the two authored wedge tet volumes
        total_vol = v_large + v_small

        stage = _deformable_stage()
        # A body mass is present but per-point masses win (99 is never distributed).
        masses_tet = _author_unit_tet(stage, "/World/SoftMasses", sim_api=True)
        _apply_deformable_body_api(masses_tet.GetPrim(), mass=99.0)
        masses_tet.GetPrim().CreateAttribute("physics:masses", Sdf.ValueTypeNames.FloatArray).Set([1.0, 2.0, 3.0, 4.0])
        # Body-mass override on the non-uniform wedge.
        wedge = _author_two_tet_wedge(stage, "/World/SoftWedge")
        _apply_deformable_body_api(wedge.GetPrim(), mass=body_mass)
        # Material-density baseline vs. a 5x body-density override.
        mat_tet = _author_unit_tet(stage, "/World/SoftMatOnly", sim_api=True)
        _bind_deformable_material(stage, mat_tet.GetPrim(), "/World/MatA", density=100.0)
        ovr_tet = _author_unit_tet(stage, "/World/SoftDensity", sim_api=True)
        _bind_deformable_material(stage, ovr_tet.GetPrim(), "/World/MatB", density=100.0)
        _apply_deformable_body_api(ovr_tet.GetPrim(), density=500.0)

        builder = newton.ModelBuilder()
        with self.assertWarnsRegex(DeprecationWarning, "masses:elementType"):
            builder.add_usd(stage)

        def masses(path):
            p0, p1 = group_range(builder, "soft", path, "particle")
            return [builder.particle_mass[i] for i in range(p0, p1)]

        # physics:masses on the simulation geometry overrides body/material mass.
        self.assertEqual(masses("/World/SoftMasses"), [1.0, 2.0, 3.0, 4.0])

        # A body-mass override must rescale the per-point masses *proportionally*. The importer's
        # rescale is ``particle_mass[i] *= body_mass / current``; a uniform ``body_mass / n`` would
        # also hit the total but flatten the distribution, so assert the per-point ratios, not just
        # the sum.
        m = masses("/World/SoftWedge")
        self.assertAlmostEqual(sum(m), body_mass, places=4)  # the override sets the total ...
        # ... but the distribution still follows adjacent-element volume. Apexes sit on one tet
        # each (V_e / 4); shared base vertices sum both tets ((V_large + V_small) / 4).
        self.assertAlmostEqual(m[3], body_mass * (v_large / 4.0) / total_vol, places=4)  # large apex
        self.assertAlmostEqual(m[4], body_mass * (v_small / 4.0) / total_vol, places=4)  # small apex
        self.assertAlmostEqual(m[3] / m[4], v_large / v_small, places=4)  # = 4, weighting preserved
        for i in range(3):
            self.assertAlmostEqual(m[i], body_mass / 4.0, places=4)  # shared = (V_large+V_small)/4 scaled
        self.assertGreater(max(m) - min(m), 1.0e-6)  # genuinely non-uniform, not flattened

        # PhysicsDeformableBodyAPI.density takes precedence over the bound material (5x density -> 5x mass).
        total_mat = sum(masses("/World/SoftMatOnly"))
        total_ovr = sum(masses("/World/SoftDensity"))
        self.assertGreater(total_mat, 0.0)
        self.assertAlmostEqual(total_ovr / total_mat, 5.0, places=4)

    def test_volume_constant_mass_array_distributes_total(self):
        """Distribute a constant simulation-geometry mass over the volume elements."""
        from pxr import Sdf

        stage = _deformable_stage()
        tet = _author_unit_tet(stage, "/World/Soft", sim_api=True)
        tet.GetPrim().CreateAttribute("physics:masses", Sdf.ValueTypeNames.FloatArray).Set([12.0])
        tet.GetPrim().CreateAttribute("physics:masses:elementType", Sdf.ValueTypeNames.Token).Set("constant")

        builder = newton.ModelBuilder()
        builder.add_usd(stage)

        self.assertEqual([builder.particle_mass[i] for i in range(4)], [3.0, 3.0, 3.0, 3.0])

    def test_volume_element_mass_types_convert_to_particles(self):
        """Convert tetrahedron and point mass arrays through element mass distribution."""
        from pxr import Sdf

        cases = (
            ("tetrahedron", _author_two_tet_wedge, [8.0, 2.0], [2.5, 2.5, 2.5, 2.0, 0.5]),
            ("point", lambda stage, path: _author_unit_tet(stage, path, sim_api=True), [1.0, 2.0, 3.0, 4.0], [2.5] * 4),
        )
        for element_type, author, authored, expected in cases:
            with self.subTest(element_type=element_type):
                stage = _deformable_stage()
                tet = author(stage, "/World/Soft")
                tet.GetPrim().CreateAttribute("physics:masses", Sdf.ValueTypeNames.FloatArray).Set(authored)
                tet.GetPrim().CreateAttribute("physics:masses:elementType", Sdf.ValueTypeNames.Token).Set(element_type)

                builder = newton.ModelBuilder()
                builder.add_usd(stage)

                np.testing.assert_allclose(builder.particle_mass, expected, atol=1.0e-6)

    def test_volume_point_masses_ignore_unreferenced_points(self):
        """Import masses on referenced TetMesh points while ignoring orphan-point values."""
        stage = _deformable_stage()
        tet = _author_unit_tet(stage, "/World/Soft", sim_api=True)
        points = list(tet.GetPointsAttr().Get())
        tet.GetPointsAttr().Set([*points, (2.0, 2.0, 2.0)])
        _author_deformable_element_array(tet.GetPrim(), "masses", [2.0] * 5, "point")

        builder = newton.ModelBuilder()
        with self.assertWarnsRegex(UserWarning, "unreferenced point"):
            builder.add_usd(stage)

        p0, p1 = group_range(builder, "soft", "/World/Soft", "particle")
        masses = builder.particle_mass[p0:p1]
        self.assertEqual(len(masses), 5)
        self.assertAlmostEqual(sum(masses), 8.0)
        self.assertEqual(masses[-1], 0.0)

    def test_body_hierarchy_selects_single_sim_mesh(self):
        """A PhysicsDeformableBodyAPI ancestor governs exactly one simulation mesh: its
        authored mass applies to the child sim geometry, while a non-sim (graphics/collision)
        TetMesh and a second sim mesh under the same body root are warned about and skipped,
        so the body's authored total mass (12 kg) is not exceeded."""
        from pxr import UsdGeom

        stage = _deformable_stage()
        UsdGeom.Xform.Define(stage, "/World/Body")
        _apply_deformable_body_api(stage.GetPrimAtPath("/World/Body"), mass=12.0)
        _author_tet_cube(stage, "/World/Body/Sim")  # carries PhysicsVolumeDeformableSimAPI
        _author_unit_tet(stage, "/World/Body/Graphics")  # no sim API -> graphics/collision
        _author_tet_cube(stage, "/World/Body/SimB", z0=2.0)  # malformed second sim mesh

        builder = newton.ModelBuilder()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            builder.add_usd(stage)
        messages = [str(w.message) for w in caught]
        self.assertTrue(any("/World/Body/Graphics" in m and "graphics/collision geometry" in m for m in messages))
        self.assertTrue(
            any("/World/Body/SimB" in m and "skipping additional simulation geometry" in m for m in messages)
        )

        # Only the first simulation mesh is imported; the extras are skipped, not simulated.
        self.assertEqual(group_labels(builder, "soft"), ["/World/Body/Sim"])
        # The whole simulated system is exactly the ancestor body's authored 12 kg
        # (mass found on the ancestor Xform; no extra graphics or second-sim mass).
        self.assertAlmostEqual(sum(builder.particle_mass), 12.0, places=4)

    def test_nested_graphics_tetmesh_is_not_simulated(self):
        """A bare TetMesh anywhere in a deformable body's subtree is graphics/collision
        geometry, not a second soft body, including below an intermediate prim. A nested
        rigid body bounds the subtree: its bare TetMesh is native content and keeps the
        legacy soft-body import."""
        from pxr import UsdGeom, UsdPhysics

        with self.subTest(case="deep_graphics_tetmesh_skipped"):
            stage = _deformable_stage()
            body = UsdGeom.Xform.Define(stage, "/World/Body").GetPrim()
            _apply_deformable_body_api(body)
            _author_tet_cube(stage, "/World/Body/Sim")
            UsdGeom.Xform.Define(stage, "/World/Body/Group")
            _author_unit_tet(stage, "/World/Body/Group/Graphics")  # bare: no sim API

            builder = newton.ModelBuilder()
            with self.assertWarnsRegex(UserWarning, "/World/Body/Group/Graphics.*graphics/collision"):
                result = builder.add_usd(stage, return_deformable_results=True)
            self.assertEqual(group_labels(builder, "soft"), ["/World/Body/Sim"])
            self.assertNotIn("/World/Body/Group/Graphics", result["path_soft_map"])

        with self.subTest(case="rigid_boundary_keeps_legacy_import"):
            stage = _deformable_stage()
            body = UsdGeom.Xform.Define(stage, "/World/Body").GetPrim()
            _apply_deformable_body_api(body)
            _author_tet_cube(stage, "/World/Body/Sim")
            rigid = UsdGeom.Xform.Define(stage, "/World/Body/Rig").GetPrim()
            UsdPhysics.RigidBodyAPI.Apply(rigid)
            _author_unit_tet(stage, "/World/Body/Rig/Tet")  # native side of the boundary

            builder = newton.ModelBuilder()
            result = builder.add_usd(stage, return_deformable_results=True)
            self.assertIn("/World/Body/Rig/Tet", result["path_soft_map"])

    def test_body_api_on_distant_ancestor_is_not_used(self):
        """The proposal allows PhysicsDeformableBodyAPI on the simulation geometry itself or
        on its direct parent; an API on a deeper ancestor does not govern the mesh. The
        importer warns instead of silently applying the distant overrides, and the mesh
        imports with its own (density-derived) mass."""
        from pxr import UsdGeom

        stage = _deformable_stage()
        UsdGeom.Xform.Define(stage, "/World/Root")
        UsdGeom.Xform.Define(stage, "/World/Root/Group")
        _apply_deformable_body_api(stage.GetPrimAtPath("/World/Root"), mass=12.0)
        _author_tet_cube(stage, "/World/Root/Group/Sim")

        builder = newton.ModelBuilder()
        with self.assertWarnsRegex(UserWarning, "direct parent"):
            builder.add_usd(stage)

        # The mesh still imports, but the distant ancestor's 12 kg override is not applied.
        self.assertEqual(group_labels(builder, "soft"), ["/World/Root/Group/Sim"])
        self.assertGreater(sum(builder.particle_mass), 0.0)
        self.assertNotAlmostEqual(sum(builder.particle_mass), 12.0, places=4)

    def test_legacy_vendor_material_keeps_deprecation_window(self):
        """A TetMesh material authoring only vendor-namespaced (omniphysics:) moduli still
        imports its stiffness/density through add_usd() during the deprecation window, with a
        DeprecationWarning naming the recovery, instead of silently falling back to defaults."""
        asset = os.path.join(os.path.dirname(__file__), "assets", "tetmesh_with_legacy_material.usda")
        builder = newton.ModelBuilder()
        with self.assertWarnsRegex(DeprecationWarning, "vendor-namespaced deformable material"):
            builder.add_usd(asset)

        # E = 3e5, nu = 0.3 -> k_mu = E / (2 (1 + nu)), k_lambda = E nu / ((1 + nu)(1 - 2 nu)).
        k_mu, k_lambda, _k_damp = builder.tet_materials[0]
        self.assertAlmostEqual(k_mu, 3.0e5 / 2.6, delta=1.0)
        self.assertAlmostEqual(k_lambda, 3.0e5 * 0.3 / (1.3 * 0.4), delta=1.0)
        # density 40 over the unit tet (V = 1/6) -> total particle mass = 40 / 6.
        self.assertAlmostEqual(sum(builder.particle_mass), 40.0 / 6.0, delta=1e-3)

    def test_bare_tetmesh_ignores_per_point_masses(self):
        """A bare TetMesh (no deformable markers) keeps the legacy import; masses ignored."""
        from pxr import Sdf

        def author(stage):
            tet = _author_unit_tet(stage, "/World/Soft")
            tet.GetPrim().CreateAttribute("physics:masses", Sdf.ValueTypeNames.FloatArray).Set([2.0, 4.0, 6.0, 8.0])

        builder, _ = self._build_soft(author)
        # Legacy mass distribution (density-derived), not the authored per-point values.
        self.assertNotEqual([builder.particle_mass[i] for i in range(4)], [2.0, 4.0, 6.0, 8.0])

    def test_volume_material_poisson_ratio_schema_fallback(self):
        """A material authoring youngsModulus without poissonsRatio uses the proposal's
        declared poissonsRatio fallback of 0.3 instead of silently discarding the authored
        modulus. Canonical and vendor namespaces behave the same."""
        youngs = 300000.0
        nu = 0.3
        expected_mu = youngs / (2.0 * (1.0 + nu))
        expected_lambda = youngs * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))

        for namespace in ("physics", "omniphysics"):
            with self.subTest(namespace=namespace):
                stage = _deformable_stage()
                tet = _author_unit_tet(stage, "/World/Soft", sim_api=True)
                _bind_deformable_material(
                    stage, tet.GetPrim(), "/World/Mat", namespace=namespace, youngsModulus=youngs, density=10.0
                )
                builder = newton.ModelBuilder()
                if namespace == "physics":
                    builder.add_usd(stage)
                else:
                    builder.add_usd(stage, schema_resolvers=[SchemaResolverPhysx()])
                k_mu, k_lambda, _k_damp = builder.tet_materials[0]
                self.assertAlmostEqual(k_mu, expected_mu, places=1)
                self.assertAlmostEqual(k_lambda, expected_lambda, places=1)

    def test_volume_material_uses_proposal_elasticity_fallbacks(self):
        """Resolve missing volume elasticity fields independently from proposal defaults."""
        cases = (
            ("both_default", {}, 384615.3846153846, 576923.0769230769),
            ("authored_negative_poissons", {"poissonsRatio": -0.25}, 666666.6666666666, -222222.22222222222),
        )
        for name, material_attrs, expected_mu, expected_lambda in cases:
            with self.subTest(name=name):
                stage = _deformable_stage()
                tet = _author_unit_tet(stage, "/World/Soft", sim_api=True)
                _bind_deformable_material(stage, tet.GetPrim(), "/World/Mat", **material_attrs)
                builder = newton.ModelBuilder()
                builder.default_tet_k_mu = 13.0
                builder.default_tet_k_lambda = 17.0
                builder.add_usd(stage)

                k_mu, k_lambda, _k_damp = builder.tet_materials[0]
                self.assertAlmostEqual(k_mu, expected_mu, delta=1.0)
                self.assertAlmostEqual(k_lambda, expected_lambda, delta=1.0)

    def test_volume_material_rejects_value_outside_usd_float_range(self):
        """Reject a raw material scalar that cannot use the proposal's USD float type."""
        from pxr import Sdf

        stage = _deformable_stage()
        tet = _author_unit_tet(stage, "/World/Soft", sim_api=True)
        material = _bind_deformable_material(stage, tet.GetPrim(), "/World/Mat")
        material.GetPrim().CreateAttribute("physics:youngsModulus", Sdf.ValueTypeNames.Double).Set(1.0e39)

        with self.assertWarnsRegex(UserWarning, "youngsModulus.*outside the finite USD float range"):
            tetmesh = newton.usd.get_tetmesh(tet.GetPrim(), compat_namespaces=())
        self.assertAlmostEqual(tetmesh.k_mu[0], 384615.3846153846, delta=1.0)
        self.assertAlmostEqual(tetmesh.k_lambda[0], 576923.0769230769, delta=1.0)

        builder = newton.ModelBuilder()
        with self.assertWarnsRegex(UserWarning, "youngsModulus.*outside the finite USD float range"):
            builder.add_usd(stage)

        k_mu, k_lambda, _k_damp = builder.tet_materials[0]
        self.assertAlmostEqual(k_mu, 384615.3846153846, delta=1.0)
        self.assertAlmostEqual(k_lambda, 576923.0769230769, delta=1.0)

    def test_get_tetmesh_rejects_non_numeric_material_scalars(self):
        """Warn and use proposal fallbacks for non-numeric material scalars."""
        from pxr import Sdf, UsdShade

        expected_mu = 384615.3846153846
        expected_lambda = 576923.0769230769
        for attr_name in ("youngsModulus", "poissonsRatio", "density"):
            with self.subTest(attribute=attr_name):
                stage = _deformable_stage()
                tet = _author_unit_tet(stage, "/World/Soft", sim_api=True)
                material = UsdShade.Material.Define(stage, "/World/Mat")
                material.GetPrim().AddAppliedSchema("PhysicsVolumeDeformableMaterialAPI")
                material.GetPrim().CreateAttribute(f"physics:{attr_name}", Sdf.ValueTypeNames.Token).Set("bad")
                UsdShade.MaterialBindingAPI.Apply(tet.GetPrim()).Bind(material, materialPurpose="physics")

                with self.assertWarnsRegex(UserWarning, rf"/World/Mat.*physics:{attr_name}.*numeric"):
                    tetmesh = newton.usd.get_tetmesh(tet.GetPrim(), compat_namespaces=())

                self.assertAlmostEqual(tetmesh.k_mu[0], expected_mu, delta=1.0)
                self.assertAlmostEqual(tetmesh.k_lambda[0], expected_lambda, delta=1.0)
                self.assertIsNone(tetmesh.density)

    def test_unbound_volume_material_uses_builder_elasticity_defaults(self):
        """Preserve builder elasticity defaults when no volume material is bound."""
        stage = _deformable_stage()
        _author_unit_tet(stage, "/World/Soft", sim_api=True)
        builder = newton.ModelBuilder()
        builder.default_tet_k_mu = 13.0
        builder.default_tet_k_lambda = 17.0

        builder.add_usd(stage)

        k_mu, k_lambda, _k_damp = builder.tet_materials[0]
        self.assertEqual(k_mu, 13.0)
        self.assertEqual(k_lambda, 17.0)

    def test_get_tetmesh_clamps_legacy_high_poissons_ratio(self):
        """Preserve Newton's legacy high-ratio behavior while warning about the proposal range."""
        stage = _deformable_stage()
        tet = _author_unit_tet(stage, "/World/Soft", sim_api=True)
        _bind_deformable_material(
            stage,
            tet.GetPrim(),
            "/World/Mat",
            youngsModulus=300000.0,
            poissonsRatio=0.6,
        )

        with self.assertWarnsRegex(UserWarning, "outside the proposal range.*0.499.*compatibility"):
            tetmesh = newton.usd.get_tetmesh(tet.GetPrim(), compat_namespaces=())

        expected_mu = 300000.0 / (2.0 * (1.0 + 0.499))
        expected_lambda = 300000.0 * 0.499 / ((1.0 + 0.499) * (1.0 - 2.0 * 0.499))
        self.assertAlmostEqual(tetmesh.k_mu[0], expected_mu, delta=1.0)
        self.assertAlmostEqual(tetmesh.k_lambda[0], expected_lambda, delta=1.0)

    def test_get_tetmesh_rejects_invalid_poissons_ratio_boundaries(self):
        """Use the proposal fallback for values outside Newton's compatibility interval."""
        for value in (-1.0, 1.0, float("nan"), float("inf"), float("-inf")):
            with self.subTest(value=value):
                stage = _deformable_stage()
                tet = _author_unit_tet(stage, "/World/Soft", sim_api=True)
                _bind_deformable_material(
                    stage,
                    tet.GetPrim(),
                    "/World/Mat",
                    youngsModulus=300000.0,
                    poissonsRatio=value,
                )

                with self.assertWarnsRegex(UserWarning, "invalid physics:poissonsRatio"):
                    tetmesh = newton.usd.get_tetmesh(tet.GetPrim(), compat_namespaces=())

                self.assertAlmostEqual(tetmesh.k_mu[0], 115384.61538461538, delta=1.0)
                self.assertAlmostEqual(tetmesh.k_lambda[0], 173076.92307692306, delta=1.0)

    def test_get_tetmesh_uses_current_volume_elasticity_fallbacks(self):
        """Resolve missing current volume elasticity fields from proposal defaults."""
        from pxr import UsdGeom

        cases = (
            ("both_default", {}, None, 384615.3846153846, 576923.0769230769),
            ("youngs_default", {"poissonsRatio": -0.25}, None, 666666.6666666666, -222222.22222222222),
            ("centimeter_stage", {}, 0.01, 3846.153846153846, 5769.230769230769),
        )
        for name, material_attrs, meters_per_unit, expected_mu, expected_lambda in cases:
            with self.subTest(name=name):
                stage = _deformable_stage()
                if meters_per_unit is not None:
                    UsdGeom.SetStageMetersPerUnit(stage, meters_per_unit)
                tet = _author_unit_tet(stage, "/World/Soft", sim_api=True)
                _bind_deformable_material(stage, tet.GetPrim(), "/World/Mat", **material_attrs)

                tetmesh = newton.usd.get_tetmesh(tet.GetPrim(), compat_namespaces=())

                self.assertAlmostEqual(tetmesh.k_mu[0], expected_mu, delta=1.0)
                self.assertAlmostEqual(tetmesh.k_lambda[0], expected_lambda, delta=1.0)

    def test_volume_incompressible_poissons_ratio_warns_once(self):
        """Approximate an incompressible volume material once with finite Lamé parameters."""
        stage = _deformable_stage()
        tet = _author_unit_tet(stage, "/World/Soft", sim_api=True)
        _bind_deformable_material(
            stage,
            tet.GetPrim(),
            "/World/Mat",
            youngsModulus=300000.0,
            poissonsRatio=0.5,
        )

        builder = newton.ModelBuilder()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            builder.add_usd(stage)
        incompressible = [w for w in caught if "incompressible" in str(w.message)]

        self.assertEqual(len(incompressible), 1)
        k_mu, k_lambda, _k_damp = builder.tet_materials[0]
        self.assertTrue(math.isfinite(k_mu) and math.isfinite(k_lambda))

    def test_malformed_tetmesh_warns_and_spares_the_stage(self):
        """A TetMesh whose indices exceed its point count warns and is skipped; the rest of
        the stage (a valid soft body) still imports instead of the whole add_usd aborting."""
        from pxr import UsdGeom

        stage = _deformable_stage()
        bad = UsdGeom.TetMesh.Define(stage, "/World/Bad")
        bad.CreatePointsAttr([(0.0, 0.0, 1.0), (1.0, 0.0, 1.0), (0.0, 1.0, 1.0), (0.0, 0.0, 2.0)])
        bad.CreateTetVertexIndicesAttr([(0, 1, 2, 99)])
        bad.GetPrim().AddAppliedSchema("PhysicsVolumeDeformableSimAPI")
        _author_unit_tet(stage, "/World/Good", sim_api=True)

        builder = newton.ModelBuilder()
        with self.assertWarnsRegex(UserWarning, "/World/Bad.*exceeds vertex count"):
            builder.add_usd(stage)

        self.assertEqual(group_labels(builder, "soft"), ["/World/Good"])
        self.assertEqual(builder.particle_count, 4)

    def test_unregistered_custom_attribute_is_ignored_and_keeps_tetmesh(self):
        """An irrelevant USD array is ignored without warning or dropping the soft body."""
        from pxr import Sdf

        cases = (
            ("physxVolumeDeformableSim:simMeshHexCrc", list(range(16))),
            ("deformablePose:default:omniphysics:purposes", [1]),
        )
        for attr_name, values in cases:
            with self.subTest(attr_name=attr_name):
                stage = _deformable_stage()
                tet = _author_unit_tet(stage, "/World/Soft", sim_api=True)
                tet.GetPrim().CreateAttribute(attr_name, Sdf.ValueTypeNames.IntArray).Set(values)

                builder = newton.ModelBuilder()
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    result = builder.add_usd(stage, return_deformable_results=True)

                self.assertFalse(any(attr_name in str(w.message) for w in caught))
                self.assertIn("/World/Soft", result["path_soft_map"])

    def test_get_tetmesh_warns_for_uninferable_custom_attribute(self):
        """Direct TetMesh loading diagnoses and omits an uninferable custom array."""
        from pxr import Sdf

        stage = _deformable_stage()
        tet = _author_unit_tet(stage, "/World/Soft")
        attr_name = "physxVolumeDeformableSim:simMeshHexCrc"
        tet.GetPrim().CreateAttribute(attr_name, Sdf.ValueTypeNames.IntArray).Set(list(range(16)))

        with self.assertWarnsRegex(UserWarning, rf"{attr_name}.*length 16.*skipping the attribute"):
            tetmesh = newton.usd.get_tetmesh(tet.GetPrim(), compat_namespaces=())

        self.assertNotIn(attr_name, tetmesh.custom_attributes)

    def test_get_tetmesh_infers_once_for_issue_pose_attribute(self):
        """The issue's one-value pose metadata resolves to ONCE on a multi-tet mesh."""
        from pxr import Sdf

        stage = _deformable_stage()
        tet = _author_tet_cube(stage, "/World/Soft")
        attr_name = "deformablePose:default:omniphysics:purposes"
        tet.GetPrim().CreateAttribute(attr_name, Sdf.ValueTypeNames.IntArray).Set([1])

        tetmesh = newton.usd.get_tetmesh(tet.GetPrim(), compat_namespaces=())

        values, frequency = tetmesh.custom_attributes[attr_name]
        np.testing.assert_array_equal(values, [1])
        self.assertEqual(frequency, newton.Model.AttributeFrequency.ONCE)

    def test_registered_custom_attribute_uses_declared_frequency(self):
        """A builder declaration resolves a one-value per-tet attribute on a one-tet mesh."""
        from pxr import Sdf

        stage = _deformable_stage()
        tet = _author_unit_tet(stage, "/World/Soft", sim_api=True)
        tet.GetPrim().CreateAttribute("custom:regionId", Sdf.ValueTypeNames.IntArray).Set([7])

        builder = newton.ModelBuilder()
        builder.add_custom_attribute(
            newton.ModelBuilder.CustomAttribute(
                name="regionId",
                dtype=wp.int32,
                frequency=newton.Model.AttributeFrequency.TETRAHEDRON,
                usd_attribute_name="custom:regionId",
            )
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            builder.add_usd(stage)
        model = builder.finalize()

        self.assertFalse(any("custom:regionId" in str(w.message) for w in caught))
        np.testing.assert_array_equal(model.regionId.numpy(), [7])

    def test_registered_custom_attribute_wrong_length_warns_and_keeps_tetmesh(self):
        """A declared frequency does not let a malformed array abort soft-body import."""
        from pxr import Sdf

        stage = _deformable_stage()
        tet = _author_unit_tet(stage, "/World/Soft", sim_api=True)
        tet.GetPrim().CreateAttribute("custom:temperature", Sdf.ValueTypeNames.FloatArray).Set([10.0, 20.0])

        builder = newton.ModelBuilder()
        builder.add_custom_attribute(
            newton.ModelBuilder.CustomAttribute(
                name="temperature",
                dtype=wp.float32,
                frequency=newton.Model.AttributeFrequency.PARTICLE,
                usd_attribute_name="custom:temperature",
            )
        )
        with self.assertWarnsRegex(UserWarning, "/World/Soft.*custom:temperature.*length 2.*particle count 4"):
            result = builder.add_usd(stage, return_deformable_results=True)

        self.assertIn("/World/Soft", result["path_soft_map"])

    def test_constant_primvar_resolves_single_tet_ambiguity(self):
        """Explicit USD interpolation preserves a constant array on a single-tetrahedron mesh."""
        from pxr import Sdf, UsdGeom

        stage = _deformable_stage()
        tet = _author_unit_tet(stage, "/World/Soft")
        primvar = UsdGeom.PrimvarsAPI(tet.GetPrim()).CreatePrimvar(
            "constantValue", Sdf.ValueTypeNames.FloatArray, UsdGeom.Tokens.constant
        )
        primvar.Set([42.0])

        tetmesh = newton.usd.get_tetmesh(tet.GetPrim(), compat_namespaces=())

        _, frequency = tetmesh.custom_attributes["constantValue"]
        self.assertEqual(frequency, newton.Model.AttributeFrequency.ONCE)

    def test_indexed_primvar_is_flattened(self):
        """Indexed vertex primvars load in expanded particle order."""
        from pxr import Sdf, UsdGeom

        stage = _deformable_stage()
        tet = _author_unit_tet(stage, "/World/Soft")
        primvar = UsdGeom.PrimvarsAPI(tet.GetPrim()).CreatePrimvar(
            "temperature", Sdf.ValueTypeNames.FloatArray, UsdGeom.Tokens.vertex
        )
        primvar.Set([10.0, 20.0])
        primvar.SetIndices([0, 1, 0, 1])

        tetmesh = newton.usd.get_tetmesh(tet.GetPrim(), compat_namespaces=())

        values, frequency = tetmesh.custom_attributes["temperature"]
        np.testing.assert_array_equal(values, [10.0, 20.0, 10.0, 20.0])
        self.assertEqual(frequency, newton.Model.AttributeFrequency.PARTICLE)

    def test_registered_indexed_primvar_imports_flattened_values(self):
        """Builder-declared indexed primvars flow to particles in expanded order."""
        from pxr import Sdf, UsdGeom

        stage = _deformable_stage()
        tet = _author_unit_tet(stage, "/World/Soft", sim_api=True)
        primvar = UsdGeom.PrimvarsAPI(tet.GetPrim()).CreatePrimvar(
            "temperature", Sdf.ValueTypeNames.FloatArray, UsdGeom.Tokens.vertex
        )
        primvar.Set([10.0, 20.0])
        primvar.SetIndices([0, 1, 0, 1])

        builder = newton.ModelBuilder()
        builder.add_custom_attribute(
            newton.ModelBuilder.CustomAttribute(
                name="temperature",
                dtype=wp.float32,
                frequency=newton.Model.AttributeFrequency.PARTICLE,
                usd_attribute_name="primvars:temperature",
            )
        )
        builder.add_usd(stage)
        model = builder.finalize()

        np.testing.assert_array_equal(model.temperature.numpy(), [10.0, 20.0, 10.0, 20.0])

    def test_registered_once_attribute_warns_and_keeps_tetmesh(self):
        """Volume import diagnoses registered ONCE data that cannot attach per soft body."""
        from pxr import Sdf

        stage = _deformable_stage()
        tet = _author_unit_tet(stage, "/World/Soft", sim_api=True)
        tet.GetPrim().CreateAttribute("custom:bodyTag", Sdf.ValueTypeNames.IntArray).Set([9])

        builder = newton.ModelBuilder()
        builder.add_custom_attribute(
            newton.ModelBuilder.CustomAttribute(
                name="bodyTag",
                dtype=wp.int32,
                frequency=newton.Model.AttributeFrequency.ONCE,
                usd_attribute_name="custom:bodyTag",
            )
        )
        with self.assertWarnsRegex(UserWarning, "/World/Soft.*custom:bodyTag.*ONCE.*not imported"):
            result = builder.add_usd(stage, return_deformable_results=True)

        self.assertIn("/World/Soft", result["path_soft_map"])

    def test_resolved_density_reports_builder_default_fallback(self):
        """With no authored or material density, add_soft_mesh falls back to the builder's
        default_tet_density; path_soft_attrs must report that actually-used value, not None."""
        stage = _deformable_stage()
        _author_unit_tet(stage, "/World/Soft")

        builder = newton.ModelBuilder()
        builder.default_tet_density = 123.5
        result = builder.add_usd(stage, return_deformable_results=True)

        self.assertEqual(result["path_soft_attrs"]["/World/Soft"]["resolved_density"], 123.5)

    def test_volume_collision_limitation(self):
        """Newton cannot disable particle collision: a volume deformable without
        an enabled PhysicsCollisionAPI warns and imports colliding."""
        from pxr import Sdf

        for case, expect_warning in (("none", True), ("enabled", False), ("disabled", True)):
            with self.subTest(case=case):
                stage = _deformable_stage()
                tet = _author_unit_tet(stage, "/World/Soft", sim_api=True, collision=False)
                if case != "none":
                    tet.GetPrim().AddAppliedSchema("PhysicsCollisionAPI")
                    if case == "disabled":
                        tet.GetPrim().CreateAttribute("physics:collisionEnabled", Sdf.ValueTypeNames.Bool).Set(False)
                builder = newton.ModelBuilder()
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    builder.add_usd(stage)
                messages = [str(w.message) for w in caught]
                warned = any("cannot disable deformable particle collision" in m for m in messages)
                self.assertEqual(warned, expect_warning)
                self.assertEqual(builder.particle_count, 4)

    def test_dedicated_collider_enables_collision_with_approximation_warning(self):
        """An enabled collider elsewhere in the deformable body hierarchy turns
        collision on, approximated by the simulation geometry, with a warning
        naming the collider prim."""
        from pxr import UsdGeom

        stage = _deformable_stage()
        body = UsdGeom.Xform.Define(stage, "/World/Body").GetPrim()
        _apply_deformable_body_api(body)
        _author_unit_tet(stage, "/World/Body/Sim", sim_api=True, collision=False)
        # A dedicated TetMesh collider: not a simulation mesh, so the volume pass
        # leaves it alone, and it is not eligible for the rigid collider path.
        collider = UsdGeom.TetMesh.Define(stage, "/World/Body/Collider")
        collider.CreatePointsAttr([(0.0, 0.0, 1.0), (0.5, 0.0, 1.0), (0.0, 0.5, 1.0), (0.0, 0.0, 1.5)])
        collider.CreateTetVertexIndicesAttr([(0, 1, 2, 3)])
        collider.GetPrim().AddAppliedSchema("PhysicsCollisionAPI")

        builder = newton.ModelBuilder()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            builder.add_usd(stage)
        messages = [str(w.message) for w in caught]
        self.assertTrue(any("approximated by the simulation geometry" in m for m in messages))
        self.assertFalse(any("cannot disable deformable particle collision" in m for m in messages))
        self.assertEqual(builder.particle_count, 4)
        self.assertEqual(builder.shape_count, 0)

    def test_volume_material_density_validation(self):
        """Reject invalid densities and continue with finite proposal fallback masses."""
        for density in (-10.0, float("nan"), float("inf"), float("-inf"), 0.0):
            with self.subTest(density=density):
                stage = _deformable_stage()
                tet = _author_unit_tet(stage, "/World/Soft", sim_api=True)
                _bind_deformable_material(stage, tet.GetPrim(), "/World/Mat", density=density)
                builder = newton.ModelBuilder()
                builder.default_tet_density = 123.5
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    result = builder.add_usd(stage, return_deformable_results=True)
                invalid_warnings = [w for w in caught if "invalid physics:density" in str(w.message)]
                if density == 0.0:
                    self.assertEqual(invalid_warnings, [], "zero is the schema fallback, not an invalid value")
                else:
                    self.assertEqual(len(invalid_warnings), 1)
                    self.assertIn("/World/Mat", str(invalid_warnings[0].message))
                # Proposal-marked deformables do not inherit Newton's legacy builder default.
                self.assertEqual(result["path_soft_attrs"]["/World/Soft"]["resolved_density"], 1000.0)
                for i in range(4):
                    m = builder.particle_mass[i]
                    self.assertTrue(math.isfinite(m) and m > 0.0, f"particle mass {m}")
                model = builder.finalize()
                inv_mass = model.particle_inv_mass.numpy()
                self.assertTrue(np.all(np.isfinite(inv_mass)) and np.all(inv_mass >= 0.0))

    def test_volume_velocities_warn_and_do_not_crash(self):
        """Authored velocities are dropped with a warning (not silently), and must not crash the
        custom-attribute frequency inference on a single-tet mesh (vertex_count == tri_count)."""
        from pxr import UsdGeom

        def author(stage):
            tet = _author_unit_tet(stage, "/World/Soft", sim_api=True)
            UsdGeom.PointBased(tet.GetPrim()).CreateVelocitiesAttr([(1.0, 2.0, 3.0)] * 4)

        with self.assertWarnsRegex(UserWarning, "velocities are not imported"):
            builder, _result = self._build_soft(author)
        # Imported at rest (velocities dropped), no crash.
        p0, p1 = group_range(builder, "soft", "/World/Soft", "particle")
        for i in range(p0, p1):
            np.testing.assert_allclose(np.array(builder.particle_qd[i]), [0.0, 0.0, 0.0], atol=1e-6)

    def test_volume_negative_scale_mirrors_and_reorients_tets(self):
        """A reflective xformOp:scale mirrors the soft-body particles and reorients each tet to keep a
        positive rest volume; a rotation+scale decomposition would drop the reflection."""
        from pxr import Gf, UsdGeom

        stage = _deformable_stage()
        tet = UsdGeom.TetMesh.Define(stage, "/World/Soft")
        tet.CreatePointsAttr([(0.0, 0.0, 1.0), (1.0, 0.0, 1.0), (0.0, 1.0, 1.0), (0.0, 0.0, 2.0)])
        tet.CreateTetVertexIndicesAttr([(0, 1, 2, 3)])
        UsdGeom.Xformable(tet).AddScaleOp().Set(Gf.Vec3d(-1.0, 1.0, 1.0))

        builder = newton.ModelBuilder()
        builder.add_usd(stage)
        p0, p1 = group_range(builder, "soft", "/World/Soft", "particle")
        pq = np.array([list(builder.particle_q[i]) for i in range(p0, p1)])
        # Original X {0, 1, 0, 0} mirrors to {0, -1, 0, 0}.
        np.testing.assert_allclose(sorted(pq[:, 0]), [-1.0, 0.0, 0.0, 0.0], atol=1e-4)

        # The imported tet keeps a positive signed rest volume (winding repaired for the reflection).
        t0, _t1 = group_range(builder, "soft", "/World/Soft", "tet")
        i, j, k, m = builder.tet_indices[t0]

        def pos(n):
            return np.array(list(builder.particle_q[n]))

        signed_vol = np.dot(pos(j) - pos(i), np.cross(pos(k) - pos(i), pos(m) - pos(i))) / 6.0
        self.assertGreater(signed_vol, 0.0, "reflected tet must keep a positive rest volume")


if __name__ == "__main__":
    unittest.main(verbosity=2)
