# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Lifecycle tests for the finalized model's deformable group-lookup API.

This module is the only test surface that exercises the experimental ``Model`` group
members (``cable_*`` / ``cloth_*`` / ``soft_*`` fields and the ``*_index()`` /
``*_range()`` helpers) directly; every other deformable test locates groups through the
builder-registry seam in ``_usd_deformable_test_utils``. If that ``Model`` API is
removed or reshaped, this module changes with it and the rest of the suite stands.
"""

import os
import unittest

import newton
from newton.tests._usd_deformable_test_utils import _add_cable_curve, _add_cloth_mesh, _deformable_stage
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

        self.assertEqual((model.cable_count, model.cloth_count, model.soft_count), (2, 1, 2))
        b0, b1 = model.cable_body_range(model.cable_index("/World/CableA/sim"))
        self.assertEqual(b1 - b0, 3)
        j0, j1 = model.cable_joint_range(model.cable_index("/World/CableA/sim"))
        self.assertEqual(j1 - j0, 2)  # open 3-segment chain
        cloth = model.cloth_index("/World/Cloth/sim")
        p0, p1 = model.cloth_particle_range(cloth)
        self.assertEqual(p1 - p0, 4)
        self.assertEqual(model.cloth_tri_range(cloth), (0, 2))
        soft_ranges = [model.soft_particle_range(model.soft_index(f"/World/Soft{s}/sim")) for s in ("A", "B")]
        self.assertNotEqual(soft_ranges[0], soft_ranges[1])
        t0, t1 = model.soft_tet_range(model.soft_index("/World/SoftA/sim"))
        self.assertEqual(t1 - t0, 1)
        # No begin_world -> global groups.
        self.assertEqual(int(model.cable_world.numpy()[0]), -1)
        with self.assertRaises(KeyError):
            model.cable_index("/World/DoesNotExist")

    def test_replicated_groups_need_explicit_world(self):
        """replicate() duplicates labels across worlds: ranges offset per world, lookup
        without a world raises, and (label, world) resolves exactly."""
        stage = _deformable_stage()
        _add_cloth_mesh(stage, "/World/Cloth")
        sub = newton.ModelBuilder()
        sub.add_usd(stage)
        scene = newton.ModelBuilder()
        scene.replicate(sub, 3)
        model = scene.finalize()

        self.assertEqual(model.cloth_count, 3)
        self.assertEqual(model.cloth_label, ["/World/Cloth"] * 3)
        self.assertEqual(list(model.cloth_world.numpy()), [0, 1, 2])
        for w in range(3):
            self.assertEqual(model.cloth_particle_range(w), (4 * w, 4 * w + 4))
        with self.assertRaisesRegex(ValueError, "pass world="):
            model.cloth_index("/World/Cloth")
        self.assertEqual(model.cloth_index("/World/Cloth", world=1), 1)
        self.assertEqual(model.cloth_particle_range(model.cloth_index("/World/Cloth", world=2)), (8, 12))
        with self.assertRaises(KeyError):
            model.cloth_index("/World/Cloth", world=7)

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

        self.assertEqual((model.cloth_count, model.cable_count), (1, 1))
        self.assertEqual(int(model.cloth_world.numpy()[model.cloth_index("/World/Cloth")]), 0)
        self.assertEqual(int(model.cable_world.numpy()[model.cable_index("/World/Cable")]), 1)

    def test_cable_group_survives_fixed_joint_collapse(self):
        """Cable body ranges ride the reindexing of collapse_fixed_joints onto the Model."""
        from pxr import UsdGeom, UsdPhysics

        stage = _deformable_stage()
        # Two rigid bodies joined by a fixed joint -> collapsed, reindexing all bodies;
        # these parse before the cable so the cable indices shift.
        for name in ("A", "B"):
            body = UsdGeom.Xform.Define(stage, f"/World/{name}")
            UsdPhysics.RigidBodyAPI.Apply(body.GetPrim())
            UsdPhysics.CollisionAPI.Apply(body.GetPrim())
        fixed = UsdPhysics.FixedJoint.Define(stage, "/World/Fix")
        fixed.CreateBody0Rel().SetTargets(["/World/A"])
        fixed.CreateBody1Rel().SetTargets(["/World/B"])
        _add_cable_curve(stage, "/World/Cable", _CABLE_PTS)

        builder = newton.ModelBuilder()
        # The two rigid bodies' fixed joint has no articulation root, so the importer warns.
        with self.assertWarnsRegex(UserWarning, "No articulation was found"):
            builder.add_usd(stage, collapse_fixed_joints=True)
        model = builder.finalize()

        b0, b1 = model.cable_body_range(model.cable_index("/World/Cable"))
        self.assertEqual(b1 - b0, 3)
        self.assertTrue(all("/World/Cable" in model.body_label[b] for b in range(b0, b1)))


if __name__ == "__main__":
    unittest.main(verbosity=2)
