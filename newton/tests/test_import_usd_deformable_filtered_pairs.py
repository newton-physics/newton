# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Tests for standard USD ``physics:filteredPairs`` with deformable participants.

The proposal's ``PhysicsElementCollisionFilter`` has its own suite; this module covers the
standard ``UsdPhysicsFilteredPairsAPI`` relationship: rigid/cable pairs expand to Newton
shape pairs, unsupported particle deformables warn, and unresolved targets never crash.
"""

import unittest
import warnings

import newton
from newton.tests._usd_deformable_test_utils import (
    _add_cable_curve,
    _add_cloth_mesh,
    _add_element_collision_filter,
    _deformable_stage,
    group_range,
)
from newton.tests.unittest_utils import USD_AVAILABLE

_CABLE_PTS_2SEG = [(0.0, 0.0, 1.0), (0.1, 0.0, 1.0), (0.2, 0.0, 1.0)]
_CABLE_PTS_3SEG = [(0.0, 1.0, 1.0), (0.1, 1.0, 1.0), (0.2, 1.0, 1.0), (0.3, 1.0, 1.0)]


def _add_filtered_pair(prim, target):
    prim.CreateRelationship("physics:filteredPairs").AddTarget(target)


def _add_box_collider(stage, path):
    from pxr import UsdGeom, UsdPhysics

    box = UsdGeom.Cube.Define(stage, path)
    UsdPhysics.CollisionAPI.Apply(box.GetPrim())
    return box.GetPrim()


def _cable_shape_ids(builder, path):
    b0, b1 = group_range(builder, "cable", path, "body")
    ids = []
    for b in range(b0, b1):
        ids.extend(builder.body_shapes[b])
    return ids


@unittest.skipUnless(USD_AVAILABLE, "Requires usd-core")
class TestUSDDeformableFilteredPairs(unittest.TestCase):
    """Standard physics:filteredPairs lowering for shape-backed deformables."""

    def test_rigid_to_cable_filtered_pair(self):
        """A filtered pair authored on a rigid collider targeting a cable expands to one
        pair per cable segment shape instead of raising KeyError."""
        stage = _deformable_stage()
        box = _add_box_collider(stage, "/World/Box")
        _add_filtered_pair(box, "/World/Cable")
        _add_cable_curve(stage, "/World/Cable", _CABLE_PTS_2SEG)

        builder = newton.ModelBuilder()
        builder.add_usd(stage)

        # The box is the only non-capsule shape; find it via the cable's shape ids.
        cable_shapes = set(_cable_shape_ids(builder, "/World/Cable"))
        self.assertEqual(len(cable_shapes), 2)
        (box_shape,) = set(range(builder.shape_count)) - cable_shapes
        expected = {(min(box_shape, s), max(box_shape, s)) for s in cable_shapes}
        self.assertTrue(expected.issubset(set(builder.shape_collision_filter_pairs)))

    def test_cable_source_and_symmetric_authoring_deduplicate(self):
        """The relationship authored on the cable (deformable source) works, and authoring
        both directions produces no duplicate pairs."""
        stage = _deformable_stage()
        box = _add_box_collider(stage, "/World/Box")
        cable = _add_cable_curve(stage, "/World/Cable", _CABLE_PTS_2SEG)
        _add_filtered_pair(cable.GetPrim(), "/World/Box")
        _add_filtered_pair(box, "/World/Cable")

        builder = newton.ModelBuilder()
        builder.add_usd(stage)

        cable_shapes = set(_cable_shape_ids(builder, "/World/Cable"))
        (box_shape,) = set(range(builder.shape_count)) - cable_shapes
        expected = {(min(box_shape, s), max(box_shape, s)) for s in cable_shapes}
        pairs = builder.shape_collision_filter_pairs
        self.assertTrue(expected.issubset(set(pairs)))
        self.assertEqual(len(pairs), len(set(pairs)), "symmetric authoring must not duplicate pairs")

    def test_cable_to_cable_cross_product(self):
        """A cable-to-cable pair filters every segment shape of one against the other."""
        stage = _deformable_stage()
        cable_a = _add_cable_curve(stage, "/World/CableA", _CABLE_PTS_2SEG)
        _add_cable_curve(stage, "/World/CableB", _CABLE_PTS_3SEG)
        _add_filtered_pair(cable_a.GetPrim(), "/World/CableB")

        builder = newton.ModelBuilder()
        builder.add_usd(stage)

        shapes_a = _cable_shape_ids(builder, "/World/CableA")
        shapes_b = _cable_shape_ids(builder, "/World/CableB")
        expected = {(min(a, b), max(a, b)) for a in shapes_a for b in shapes_b}
        self.assertEqual(len(expected), 6)
        self.assertTrue(expected.issubset(set(builder.shape_collision_filter_pairs)))

    def test_rigid_body_and_deformable_body_endpoints(self):
        """Body-level endpoints resolve to their shapes: a rigid body Xform expands to all
        its colliders, and a deformable body Xform resolves through its simulation cable."""
        from pxr import UsdGeom, UsdPhysics

        stage = _deformable_stage()
        body = UsdGeom.Xform.Define(stage, "/World/Body").GetPrim()
        UsdPhysics.RigidBodyAPI.Apply(body)
        _add_box_collider(stage, "/World/Body/ColA")
        _add_box_collider(stage, "/World/Body/ColB")

        soft_body = UsdGeom.Xform.Define(stage, "/World/Deformable").GetPrim()
        soft_body.AddAppliedSchema("PhysicsDeformableBodyAPI")
        _add_cable_curve(stage, "/World/Deformable/Cable", _CABLE_PTS_2SEG)
        _add_filtered_pair(soft_body, "/World/Body")

        builder = newton.ModelBuilder()
        builder.add_usd(stage)

        cable_shapes = _cable_shape_ids(builder, "/World/Deformable/Cable")
        rigid_shapes = set(range(builder.shape_count)) - set(cable_shapes)
        self.assertEqual(len(rigid_shapes), 2)
        expected = {(min(a, b), max(a, b)) for a in rigid_shapes for b in cable_shapes}
        self.assertTrue(expected.issubset(set(builder.shape_collision_filter_pairs)))

    def test_rigid_body_source_to_cable_filtered_pair(self):
        """A relationship authored ON a rigid-body prim (not its colliders) is collected and
        expands to the complete rigid-shape x cable-shape cross-product."""
        from pxr import UsdGeom, UsdPhysics

        stage = _deformable_stage()
        body = UsdGeom.Xform.Define(stage, "/World/Body").GetPrim()
        UsdPhysics.RigidBodyAPI.Apply(body)
        _add_box_collider(stage, "/World/Body/ColA")
        _add_box_collider(stage, "/World/Body/ColB")
        _add_filtered_pair(body, "/World/Cable")
        _add_cable_curve(stage, "/World/Cable", _CABLE_PTS_2SEG)

        builder = newton.ModelBuilder()
        builder.add_usd(stage)

        cable_shapes = set(_cable_shape_ids(builder, "/World/Cable"))
        rigid_shapes = set(range(builder.shape_count)) - cable_shapes
        self.assertEqual(len(rigid_shapes), 2)
        self.assertEqual(len(cable_shapes), 2)
        expected = {(min(a, b), max(a, b)) for a in rigid_shapes for b in cable_shapes}
        self.assertEqual(len(expected), 4)
        self.assertTrue(expected.issubset(set(builder.shape_collision_filter_pairs)))

    def test_rigid_only_pair_unchanged(self):
        """A plain collider-to-collider pair still produces exactly one filter pair after
        the deferred application."""
        stage = _deformable_stage()
        box_a = _add_box_collider(stage, "/World/BoxA")
        _add_box_collider(stage, "/World/BoxB")
        _add_filtered_pair(box_a, "/World/BoxB")

        builder = newton.ModelBuilder()
        builder.add_usd(stage)
        self.assertEqual(len(builder.shape_collision_filter_pairs), 1)

    def test_unsupported_particle_deformable_pairs_warn(self):
        """Cloth and volume targets warn with both paths and import everything else."""
        from pxr import UsdGeom

        def _author_tet(stage):
            tet = UsdGeom.TetMesh.Define(stage, "/World/Soft")
            tet.CreatePointsAttr([(0.0, 0.0, 1.0), (1.0, 0.0, 1.0), (0.0, 1.0, 1.0), (0.0, 0.0, 2.0)])
            tet.CreateTetVertexIndicesAttr([(0, 1, 2, 3)])
            tet.GetPrim().AddAppliedSchema("PhysicsVolumeDeformableSimAPI")
            tet.GetPrim().AddAppliedSchema("PhysicsCollisionAPI")
            return tet

        for family, author in (
            ("cloth", lambda stage: _add_cloth_mesh(stage, "/World/Soft")),
            ("volume", _author_tet),
        ):
            with self.subTest(family=family):
                stage = _deformable_stage()
                box = _add_box_collider(stage, "/World/Box")
                author(stage)
                _add_filtered_pair(box, "/World/Soft")

                builder = newton.ModelBuilder()
                with self.assertWarnsRegex(UserWarning, r"/World/Box <-> /World/Soft.*filteredPairs was not imported"):
                    builder.add_usd(stage)
                self.assertGreater(builder.particle_count, 0, "the deformable must still import")
                self.assertEqual(len(builder.shape_collision_filter_pairs), 0)

    def test_missing_and_ignored_targets_warn_and_continue(self):
        """A missing target warns 'does not exist'; an ignored target warns that it produced
        no collision participant. Neither aborts the import."""
        stage = _deformable_stage()
        box = _add_box_collider(stage, "/World/Box")
        _add_filtered_pair(box, "/World/Nope")
        _add_cable_curve(stage, "/World/Ignored", _CABLE_PTS_2SEG)
        _add_filtered_pair(box, "/World/Ignored")

        builder = newton.ModelBuilder()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            builder.add_usd(stage, ignore_paths=["/World/Ignored"])
        messages = [str(w.message) for w in caught]
        self.assertTrue(any("/World/Nope" in m and "does not exist" in m for m in messages))
        self.assertTrue(any("/World/Ignored" in m and "no collision participant" in m for m in messages))
        self.assertEqual(builder.shape_count, 1, "the box still imports")
        self.assertEqual(len(builder.shape_collision_filter_pairs), 0)

    def test_self_target_is_ignored(self):
        """A self-referencing filtered pair is dropped at collection; no self-pair, no crash."""
        stage = _deformable_stage()
        box = _add_box_collider(stage, "/World/Box")
        _add_filtered_pair(box, "/World/Box")

        builder = newton.ModelBuilder()
        builder.add_usd(stage)
        self.assertEqual(len(builder.shape_collision_filter_pairs), 0)

    def test_element_filter_pairs_are_not_duplicated(self):
        """Pairs already added by PhysicsElementCollisionFilter are not re-added by an
        equivalent standard filtered pair in the same stage."""
        stage = _deformable_stage()
        _add_box_collider(stage, "/World/Box")
        cable = _add_cable_curve(stage, "/World/Cable", _CABLE_PTS_2SEG)
        _add_filtered_pair(cable.GetPrim(), "/World/Box")
        # Empty indices select all elements (authoring indices without counts would warn).
        _add_element_collision_filter(
            stage,
            "/World/Filter",
            src0="/World/Cable",
            src1="/World/Box",
            indices0=[],
            indices1=[],
        )

        builder = newton.ModelBuilder()
        builder.add_usd(stage)
        pairs = builder.shape_collision_filter_pairs
        self.assertEqual(len(pairs), len(set(pairs)), "element + standard filters must deduplicate")


if __name__ == "__main__":
    unittest.main()
