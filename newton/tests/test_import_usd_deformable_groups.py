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
        self.assertEqual(j1 - j0, 3)  # free root and two rod joints
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
        j0, j1 = group_range(builder, "cable", "/World/Cable", "joint")
        self.assertEqual(builder.joint_type[j0:j1], [newton.JointType.FREE, newton.JointType.ROD, newton.JointType.ROD])
        self.assertEqual(builder.joint_child[j0:j1], list(range(b0, b1)))
        model = builder.finalize()
        self.assertEqual(model.body_count, 4)

    def test_welded_graph_ranges_survive_collapse_and_replication(self):
        """Preserve whole-graph ranges through collapse and both world-cloning paths."""
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
        result = builder.add_usd(stage, collapse_fixed_joints=True, return_deformable_results=True)

        graph_label = result["path_cable_attrs"]["/World/Branch"]["graph_component"]
        self.assertEqual(builder.curve_label, [graph_label])
        self.assertEqual(
            result["path_cable_map"],
            {"/World/Branch": ([1, 2], []), "/World/Trunk": ([3, 4, 5], [])},
        )
        self.assertEqual(group_range(builder, "cable", graph_label, "body"), (1, 6))
        self.assertEqual(group_range(builder, "cable", graph_label, "joint"), (0, 5))
        self.assertEqual(builder.joint_type, [newton.JointType.FREE, *[newton.JointType.ROD] * 4])

        for replicate in (False, True):
            with self.subTest(replicate=replicate):
                scene = newton.ModelBuilder()
                prefixes = ["env_0", "env_1"]
                if replicate:
                    scene.replicate(builder, 2, label_prefixes=prefixes)
                else:
                    for prefix in prefixes:
                        scene.add_world(builder, label_prefix=prefix)
                self.assertEqual(scene.curve_world, [0, 1])
                self.assertEqual(scene.curve_label, [f"{prefix}/{graph_label}" for prefix in prefixes])
                for world, label in enumerate(scene.curve_label):
                    self.assertEqual(
                        group_range(scene, "cable", label, "body", world=world), (6 * world + 1, 6 * world + 6)
                    )
                    self.assertEqual(
                        group_range(scene, "cable", label, "joint", world=world), (5 * world, 5 * world + 5)
                    )
                model = scene.finalize(device="cpu")
                self.assertEqual((model.body_count, model.joint_count, model.world_count), (12, 10, 2))

    def test_native_and_usd_cables_record_generated_roots(self):
        """Record every component's root while keeping returned rod joints unchanged."""
        for curve_count in (1, 2):
            with self.subTest(curve_count=curve_count):
                points = [(x, float(curve), z) for curve in range(curve_count) for x, _, z in _CABLE_PTS]
                edges = [(4 * curve + i, 4 * curve + i + 1) for curve in range(curve_count) for i in range(3)]
                native = newton.ModelBuilder()
                native_result = native.add_rod(
                    rod=newton.Rod(points, edges=edges, radius=0.02),
                    label="/World/Cables",
                    body_frame_origin="com",
                )

                stage = _deformable_stage()
                curve = _add_cable_curve(stage, "/World/Cables", points)
                curve.CreateCurveVertexCountsAttr([4] * curve_count)
                builder = newton.ModelBuilder()
                result = builder.add_usd(stage, return_deformable_results=True)
                expected_joints = [1, 2] if curve_count == 1 else [1, 2, 4, 5]
                self.assertEqual(result["path_cable_map"]["/World/Cables"], native_result)
                self.assertEqual(native_result[1], expected_joints)
                self.assertEqual(builder.joint_type, native.joint_type)
                self.assertEqual(builder.joint_parent, native.joint_parent)
                self.assertEqual(builder.joint_child, native.joint_child)
                self.assertEqual(builder.curve_label, ["/World/Cables"])
                for source in (native, builder):
                    for kind in ("body", "joint"):
                        self.assertEqual(group_range(source, "cable", "/World/Cables", kind), (0, 3 * curve_count))
                model = builder.finalize(device="cpu")
                self.assertEqual((model.body_count, model.joint_count), (3 * curve_count, 3 * curve_count))

    def test_welded_graph_records_one_object_like_native_rod(self):
        """Record the complete welded graph while preserving each source curve's import map."""
        stage = _deformable_stage()
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
        result = builder.add_usd(stage, return_deformable_results=True)
        graph_label = result["path_cable_attrs"]["/World/Branch"]["graph_component"]
        self.assertEqual(result["path_cable_attrs"]["/World/Trunk"]["graph_component"], graph_label)
        self.assertEqual(builder.curve_label, [graph_label])
        self.assertEqual(builder.curve_world, [-1])
        self.assertEqual(
            result["path_cable_map"],
            {"/World/Branch": ([0, 1], []), "/World/Trunk": ([2, 3, 4], [])},
        )

        # Match the importer's branch-first edge order so both build the same joint tree.
        native = newton.ModelBuilder()
        bodies, joints = native.add_rod(
            rod=newton.Rod(
                [(0.1, 0.0, 1.0), (0.1, 0.1, 1.0), (0.1, 0.2, 1.0), _CABLE_PTS[0], *_CABLE_PTS[2:]],
                edges=[(0, 1), (1, 2), (3, 0), (0, 4), (4, 5)],
                radius=0.02,
            ),
            label=graph_label,
            body_frame_origin="com",
        )
        self.assertEqual((bodies, joints), ([0, 1, 2, 3, 4], [1, 2, 3, 4]))
        for source in (native, builder):
            self.assertEqual(source.curve_label, [graph_label])
            for kind in ("body", "joint"):
                self.assertEqual(group_range(source, "cable", graph_label, kind), (0, 5))
        for field in ("joint_type", "joint_parent", "joint_child"):
            self.assertEqual(getattr(builder, field), getattr(native, field))
        self.assertEqual(builder.joint_type[0], newton.JointType.FREE)
        model = builder.finalize(device="cpu")
        self.assertEqual((model.body_count, model.joint_count), (5, 5))

    def test_skipped_curves_do_not_add_joint_records(self):
        """Skip invalid curves without losing the valid curve's root or adding empty objects."""
        for include_valid in (False, True):
            with self.subTest(include_valid=include_valid):
                stage = _deformable_stage()
                points = _CABLE_PTS[:2] + (_CABLE_PTS if include_valid else [])
                curve = _add_cable_curve(stage, "/World/Cables", points)
                curve.CreateCurveVertexCountsAttr([2, 4] if include_valid else [2])
                builder = newton.ModelBuilder()
                with self.assertWarnsRegex(UserWarning, "need >= 3"):
                    result = builder.add_usd(stage, return_deformable_results=True)
                if include_valid:
                    self.assertEqual(builder.curve_label, ["/World/Cables"])
                    self.assertEqual(group_range(builder, "cable", "/World/Cables", "joint"), (0, 3))
                    self.assertEqual(result["path_cable_map"]["/World/Cables"], ([0, 1, 2], [1, 2]))
                else:
                    self.assertEqual(builder.curve_label, [])
                    self.assertEqual(builder.joint_count, 0)
                    self.assertNotIn("/World/Cables", result["path_cable_map"])

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
                root = 0 if attached_point is None else 1
                self.assertEqual(joints, [root + 1, root + 2])
                self.assertEqual(group_range(source, "cable", "/World/Cable", "joint"), (root, root + 3))
                self.assertEqual(
                    source.joint_type[root], newton.JointType.FREE if attached_point is None else newton.JointType.BALL
                )
                if attached_point is not None:
                    self.assertEqual(result["path_attachment_map"]["/World/Attachment"], [root])
                    self.assertEqual(source.joint_type[0], newton.JointType.FREE)
                    self.assertEqual(source.joint_child[0], result["path_body_map"]["/World/Plug"])
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
                    [(root + w * source.joint_count, root + 3 + w * source.joint_count) for w in range(2)],
                )


if __name__ == "__main__":
    unittest.main(verbosity=2)
