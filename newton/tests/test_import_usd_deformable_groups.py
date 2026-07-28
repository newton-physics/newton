# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Lifecycle tests for the builder's deformable group registries.

The importer records each deformable as a prim-path-labelled, world-tagged index range on
:class:`ModelBuilder`. These tests cover how those registries behave across the model
lifecycle: replication, heterogeneous worlds, and fixed-joint collapse.
"""

import os
import unittest

import newton
from newton.tests._usd_deformable_test_utils import (
    _add_cable_curve,
    _add_cloth_mesh,
    _add_physics_attachment,
    _deformable_stage,
    group_labels,
    group_range,
)
from newton.tests.unittest_utils import USD_AVAILABLE

_MIXED_ASSET = os.path.join(os.path.dirname(__file__), "assets", "deformables_mixed.usda")

_CABLE_PTS = [(0.0, 0.0, 1.0), (0.1, 0.0, 1.0), (0.2, 0.0, 1.0), (0.3, 0.0, 1.0)]


@unittest.skipUnless(USD_AVAILABLE, "Requires usd-core")
class TestUSDDeformableGroups(unittest.TestCase):
    """Prim-path group registries across lifecycle transformations."""

    def test_mixed_scene_groups_survive_finalize(self):
        """The group registries describe the mixed scene and the model finalizes intact."""
        builder = newton.ModelBuilder()
        builder.add_usd(_MIXED_ASSET)

        b0, b1 = group_range(builder, "cable", "/World/CableA/sim", "body")
        self.assertEqual(b1 - b0, 3)
        j0, j1 = group_range(builder, "cable", "/World/CableA/sim", "joint")
        self.assertEqual(j1 - j0, 2)  # open 3-segment chain
        p0, p1 = group_range(builder, "cloth", "/World/Cloth/sim", "particle")
        self.assertEqual(p1 - p0, 4)
        t0, t1 = group_range(builder, "soft", "/World/SoftA/sim", "tet")
        self.assertEqual(t1 - t0, 1)
        # No begin_world -> global groups.
        self.assertEqual(builder._cable_world, [-1, -1])

        model = builder.finalize()
        self.assertEqual((model.particle_count, model.body_count), (12, 6))
        with self.assertRaises(LookupError):
            group_range(builder, "cable", "/World/DoesNotExist", "body")

    def test_replicated_groups_offset_ranges_per_world(self):
        """replicate() repeats each group per world with offset ranges and world tags, so
        a duplicated label resolves only with an explicit world."""
        stage = _deformable_stage()
        _add_cloth_mesh(stage, "/World/Cloth")
        sub = newton.ModelBuilder()
        sub.add_usd(stage)
        scene = newton.ModelBuilder()
        scene.replicate(sub, 3)

        self.assertEqual(group_labels(scene, "cloth"), ["/World/Cloth"] * 3)
        self.assertEqual(scene._cloth_world, [0, 1, 2])
        for w in range(3):
            self.assertEqual(group_range(scene, "cloth", "/World/Cloth", "particle", world=w), (4 * w, 4 * w + 4))
        with self.assertRaises(LookupError):
            group_range(scene, "cloth", "/World/Cloth", "particle")  # ambiguous without world
        with self.assertRaises(LookupError):
            group_range(scene, "cloth", "/World/Cloth", "particle", world=7)
        scene.finalize()  # replicated groups do not break finalization

    def test_heterogeneous_worlds_keep_world_tags(self):
        """Worlds holding different deformables each keep their own group and world tag."""
        cloth_stage = _deformable_stage()
        _add_cloth_mesh(cloth_stage, "/World/Cloth")
        cable_stage = _deformable_stage()
        _add_cable_curve(cable_stage, "/World/Cable", _CABLE_PTS)

        cloth_sub = newton.ModelBuilder()
        cloth_sub.add_usd(cloth_stage)
        cable_sub = newton.ModelBuilder()
        cable_sub.add_usd(cable_stage)
        scene = newton.ModelBuilder()
        scene.add_world(cloth_sub)  # world 0: cloth only
        scene.add_world(cable_sub)  # world 1: cable only

        self.assertEqual(scene._cloth_world, [0])
        self.assertEqual(scene._cable_world, [1])
        self.assertEqual(group_range(scene, "cloth", "/World/Cloth", "particle", world=0), (0, 4))
        b0, b1 = group_range(scene, "cable", "/World/Cable", "body", world=1)
        self.assertEqual(b1 - b0, 3)
        scene.finalize()

    def test_cable_group_survives_fixed_joint_collapse(self):
        """Cable body ranges follow the renumbered bodies of collapse_fixed_joints."""
        from pxr import UsdGeom, UsdPhysics

        stage = _deformable_stage()
        # Two rigid bodies joined by a fixed joint -> collapsed, reindexing all bodies;
        # these parse before the cable so the cable indices shift.
        for name in ("A", "B"):
            body = UsdGeom.Xform.Define(stage, f"/World/{name}")
            UsdPhysics.RigidBodyAPI.Apply(body.GetPrim())
        fixed = UsdPhysics.FixedJoint.Define(stage, "/World/Fix")
        fixed.CreateBody0Rel().SetTargets(["/World/A"])
        fixed.CreateBody1Rel().SetTargets(["/World/B"])
        _add_cable_curve(stage, "/World/Cable", _CABLE_PTS)

        builder = newton.ModelBuilder()
        builder.add_usd(stage, collapse_fixed_joints=True)

        b0, b1 = group_range(builder, "cable", "/World/Cable", "body")
        self.assertEqual(b1 - b0, 3)
        self.assertTrue(all("/World/Cable" in builder.body_label[b] for b in range(b0, b1)))
        builder.finalize()

    def test_welded_graph_empty_joint_ranges_survive_collapse(self):
        """A welded-graph curve records an empty joint range at its insertion boundary; when
        an earlier fixed joint is collapsed away, that boundary must shift with the retained
        joints instead of pointing past the final joint array."""
        from pxr import UsdGeom, UsdPhysics

        stage = _deformable_stage()
        for name in ("A", "B"):
            body = UsdGeom.Xform.Define(stage, f"/World/{name}")
            UsdPhysics.RigidBodyAPI.Apply(body.GetPrim())
        fixed = UsdPhysics.FixedJoint.Define(stage, "/World/Fix")
        fixed.CreateBody0Rel().SetTargets(["/World/A"])
        fixed.CreateBody1Rel().SetTargets(["/World/B"])
        _add_cable_curve(stage, "/World/Trunk", _CABLE_PTS)
        _add_cable_curve(stage, "/World/Branch", [(0.1, 0.0, 1.0), (0.1, 0.1, 1.0), (0.1, 0.2, 1.0)])
        _add_physics_attachment(
            stage,
            "/World/Junction",
            src0="/World/Branch",
            src1="/World/Trunk",
            type0="point",
            type1="point",
            indices0=[0],
            indices1=[1],
        )

        builder = newton.ModelBuilder()
        builder.add_usd(stage, collapse_fixed_joints=True)

        for path in ("/World/Trunk", "/World/Branch"):
            j0, j1 = group_range(builder, "cable", path, "joint")
            self.assertEqual(j0, j1, "welded-graph curves own no tree joints")
            self.assertLessEqual(j1, builder.joint_count, f"{path}: empty range points past the joint array")
        builder.finalize()


if __name__ == "__main__":
    unittest.main(verbosity=2)
