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
    _author_deformable_element_array,
    _bind_deformable_material,
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

    def test_mixed_scene_groups_and_model_counts(self):
        """Record the mixed scene on the builder and finalize its simulation elements intact."""
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
        self.assertEqual(builder.curve_world, [-1, -1])
        self.assertEqual((len(builder.curve_label), len(builder.surface_label), len(builder.volume_label)), (2, 1, 2))

        model = builder.finalize()
        self.assertEqual((model.particle_count, model.body_count), (12, 6))
        with self.assertRaises(LookupError):
            group_range(builder, "cable", "/World/DoesNotExist", "body")

    def test_replicated_groups_offset_ranges_per_world(self):
        """Verify that replicate() offsets every deformable group per world.

        Preserve world tags while repeating groups and require an explicit world to
        resolve labels duplicated by replication.
        """
        stage = _deformable_stage()
        cloth = _add_cloth_mesh(stage, "/World/Cloth")
        _author_deformable_element_array(cloth.GetPrim(), "thicknesses", [0.001], "constant")
        _bind_deformable_material(stage, cloth.GetPrim(), "/World/ClothMat")
        sub = newton.ModelBuilder()
        sub.add_usd(stage)
        scene = newton.ModelBuilder()
        scene.replicate(sub, 3)

        self.assertEqual(group_labels(scene, "cloth"), ["/World/Cloth"] * 3)
        self.assertEqual(scene.surface_world, [0, 1, 2])
        for w in range(3):
            self.assertEqual(group_range(scene, "cloth", "/World/Cloth", "particle", world=w), (4 * w, 4 * w + 4))
        with self.assertRaises(LookupError):
            group_range(scene, "cloth", "/World/Cloth", "particle")  # ambiguous without world
        with self.assertRaises(LookupError):
            group_range(scene, "cloth", "/World/Cloth", "particle", world=7)
        model = scene.finalize()
        self.assertEqual((model.particle_count, model.world_count), (12, 3))

    def test_heterogeneous_worlds_keep_world_tags(self):
        """Verify that heterogeneous worlds preserve their group labels and world tags."""
        cloth_stage = _deformable_stage()
        cloth = _add_cloth_mesh(cloth_stage, "/World/Cloth")
        _author_deformable_element_array(cloth.GetPrim(), "thicknesses", [0.001], "constant")
        _bind_deformable_material(cloth_stage, cloth.GetPrim(), "/World/ClothMat")
        cable_stage = _deformable_stage()
        _add_cable_curve(cable_stage, "/World/Cable", _CABLE_PTS)

        cloth_sub = newton.ModelBuilder()
        cloth_sub.add_usd(cloth_stage)
        cable_sub = newton.ModelBuilder()
        cable_sub.add_usd(cable_stage)
        scene = newton.ModelBuilder()
        scene.add_world(cloth_sub)  # world 0: cloth only
        scene.add_world(cable_sub)  # world 1: cable only

        self.assertEqual(scene.surface_world, [0])
        self.assertEqual(scene.curve_world, [1])
        self.assertEqual(group_range(scene, "cloth", "/World/Cloth", "particle", world=0), (0, 4))
        b0, b1 = group_range(scene, "cable", "/World/Cable", "body", world=1)
        self.assertEqual(b1 - b0, 3)
        model = scene.finalize()
        self.assertEqual((model.particle_count, model.body_count, model.world_count), (4, 3, 2))

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
        model = builder.finalize()
        self.assertEqual(model.body_count, 4)

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
        self.assertCountEqual(builder.curve_label, ["/World/Trunk", "/World/Branch"])
        for path in builder.curve_label:
            j0, j1 = group_range(builder, "cable", path, "joint")
            self.assertEqual(j0, j1)
            self.assertLessEqual(j1, model.joint_count)

    def test_cable_prim_with_multiple_curves_records_once(self):
        """Record one USD prim rather than one record per native rod construction call."""
        stage = _deformable_stage()
        points = [*_CABLE_PTS, *((x, 1.0, z) for x, _, z in _CABLE_PTS)]
        curve = _add_cable_curve(stage, "/World/Cables", points)
        curve.CreateCurveVertexCountsAttr([4, 4])
        builder = newton.ModelBuilder()
        result = builder.add_usd(stage, return_deformable_results=True)
        self.assertEqual(builder.curve_label, ["/World/Cables"])
        self.assertEqual(len(result["path_cable_map"]["/World/Cables"][0]), 6)
        model = builder.finalize(device="cpu")
        self.assertEqual(model.body_count, 6)
        self.assertEqual(group_range(builder, "cable", "/World/Cables", "body"), (0, 6))

    def test_cable_records_replicate_with_free_and_attached_roots(self):
        """Preserve a cable's recorded ranges when either endpoint is attached to a rigid body."""
        from pxr import UsdGeom, UsdPhysics

        for attached_point in (None, 0, len(_CABLE_PTS) - 1):
            with self.subTest(attached_point=attached_point):
                stage = _deformable_stage()
                _add_cable_curve(stage, "/World/Cable", _CABLE_PTS)
                if attached_point is not None:
                    plug = UsdGeom.Cube.Define(stage, "/World/Plug")
                    plug.CreateSizeAttr(0.1)
                    UsdPhysics.RigidBodyAPI.Apply(plug.GetPrim())
                    UsdPhysics.CollisionAPI.Apply(plug.GetPrim())
                    _add_physics_attachment(
                        stage,
                        "/World/Attachment",
                        src0="/World/Cable",
                        src1="/World/Plug",
                        type0="point",
                        indices0=[attached_point],
                        coords1=[_CABLE_PTS[attached_point]],
                    )
                source = newton.ModelBuilder()
                result = source.add_usd(stage, return_deformable_results=True)
                bodies, joints = result["path_cable_map"]["/World/Cable"]
                self.assertEqual(source.curve_label, ["/World/Cable"])
                scene = newton.ModelBuilder()
                scene.replicate(source, 2)
                model = scene.finalize(device="cpu")
                self.assertEqual((model.body_count, model.joint_count), (2 * source.body_count, 2 * source.joint_count))
                self.assertEqual(scene.curve_world, [0, 1])
                self.assertEqual(
                    [group_range(scene, "cable", "/World/Cable", "body", world=w) for w in range(2)],
                    [(bodies[0] + w * source.body_count, bodies[-1] + 1 + w * source.body_count) for w in range(2)],
                )
                self.assertEqual(
                    [group_range(scene, "cable", "/World/Cable", "joint", world=w) for w in range(2)],
                    [(joints[0] + w * source.joint_count, joints[-1] + 1 + w * source.joint_count) for w in range(2)],
                )


if __name__ == "__main__":
    unittest.main(verbosity=2)
