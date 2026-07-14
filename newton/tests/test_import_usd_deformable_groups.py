# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Lifecycle tests for addressing deformable groups on the finalized model.

Groups are addressed through :class:`newton.selection.DeformableView` (the public
surface; the Model-side group table is private). Every other deformable test locates
groups through the builder-registry seam in ``_usd_deformable_test_utils``; this module
covers the post-``finalize()`` path across lifecycle transformations (replication,
heterogeneous worlds, fixed-joint collapse).
"""

import os
import unittest

import newton
from newton.selection import DeformableView
from newton.tests._usd_deformable_test_utils import (
    _add_cable_curve,
    _add_cloth_mesh,
    _add_physics_attachment,
    _deformable_stage,
    group_range,
)
from newton.tests.unittest_utils import USD_AVAILABLE

_MIXED_ASSET = os.path.join(os.path.dirname(__file__), "assets", "deformables_mixed.usda")

_CABLE_PTS = [(0.0, 0.0, 1.0), (0.1, 0.0, 1.0), (0.2, 0.0, 1.0), (0.3, 0.0, 1.0)]


@unittest.skipUnless(USD_AVAILABLE, "Requires usd-core")
class TestUSDDeformableGroups(unittest.TestCase):
    """Prim-path group lookup on the finalized Model across lifecycle transformations."""

    def test_mixed_scene_groups_resolve_after_finalize(self):
        """Every family of the mixed scene resolves by prim path on the finalized Model."""
        builder = newton.ModelBuilder()
        builder.add_usd(_MIXED_ASSET)
        model = builder.finalize()

        cable = DeformableView(model, "/World/CableA/sim", family="curve")
        self.assertEqual(cable.count, 1)
        self.assertEqual(cable.bodies_per_group, 3)
        self.assertEqual(cable.elements_per_group("joint"), 2)  # open 3-segment chain
        cloth = DeformableView(model, "/World/Cloth/sim", family="surface")
        self.assertEqual(cloth.particles_per_group, 4)
        self.assertEqual(cloth.ranges("triangle"), [(0, 2)])
        soft = DeformableView(model, "/World/Soft*/sim", family="volume")
        self.assertEqual(soft.count, 2)
        soft_ranges = soft.ranges("particle")
        self.assertNotEqual(soft_ranges[0], soft_ranges[1])
        self.assertEqual(soft.elements_per_group("tetrahedron"), 1)
        # No begin_world -> global groups.
        self.assertEqual(cable.worlds, [-1])
        with self.assertRaises(KeyError):
            DeformableView(model, "/World/DoesNotExist", family="curve")

    def test_replicated_groups_select_per_world(self):
        """replicate() duplicates labels across worlds: one group per world, ranges
        offset per world, and raw ranges come back in world order."""
        stage = _deformable_stage()
        _add_cloth_mesh(stage, "/World/Cloth")
        sub = newton.ModelBuilder()
        sub.add_usd(stage)
        scene = newton.ModelBuilder()
        scene.replicate(sub, 3)
        model = scene.finalize()

        view = DeformableView(model, "/World/Cloth", family="surface")
        self.assertEqual((view.count, view.world_count, view.count_per_world), (3, 3, 1))
        self.assertEqual(view.labels, ["/World/Cloth"] * 3)
        self.assertEqual(view.worlds, [0, 1, 2])
        self.assertEqual(view.ranges("particle"), [(4 * w, 4 * w + 4) for w in range(3)])
        self.assertEqual(list(view.starts("particle").numpy()), [0, 4, 8])

    def test_heterogeneous_worlds_resolve_with_world_tags(self):
        """Worlds holding different deformables each resolve with the right world tag."""
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
        model = scene.finalize()

        self.assertEqual(DeformableView(model, "/World/Cloth", family="surface").worlds, [0])
        self.assertEqual(DeformableView(model, "/World/Cable", family="curve").worlds, [1])

    def test_cable_group_survives_fixed_joint_collapse(self):
        """Cable body ranges ride the reindexing of collapse_fixed_joints onto the Model."""
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
        model = builder.finalize()

        view = DeformableView(model, "/World/Cable", family="curve")
        ((b0, b1),) = view.ranges("body")
        self.assertEqual(b1 - b0, 3)
        self.assertTrue(all("/World/Cable" in model.body_label[b] for b in range(b0, b1)))

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
        model = builder.finalize()
        for path in ("/World/Trunk", "/World/Branch"):
            view = DeformableView(model, path, family="curve")
            ((j0, j1),) = view.ranges("joint")
            self.assertEqual(j0, j1, "finalized welded-graph curves own no tree joints")
            self.assertLessEqual(j1, model.joint_count, f"{path}: finalized empty range points past the joint array")


if __name__ == "__main__":
    unittest.main(verbosity=2)
