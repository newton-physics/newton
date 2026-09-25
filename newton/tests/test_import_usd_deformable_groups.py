# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Lifecycle tests for addressing deformable objects on the finalized model.

Deformable objects are addressed through the public family-specific selection views
(the Model-side deformable object table is private). Every other deformable test locates
deformable objects through the builder-registry seam in ``_usd_deformable_test_utils``; this module
covers the post-``finalize()`` path across lifecycle transformations (replication,
heterogeneous worlds, fixed-joint collapse).
"""

import os
import unittest

import newton
from newton.selection import DeformableCurveView, DeformableSurfaceView, DeformableVolumeView
from newton.tests._usd_deformable_test_utils import (
    _add_cable_curve,
    _add_cloth_mesh,
    _add_physics_attachment,
    _author_deformable_element_array,
    _bind_deformable_material,
    _deformable_stage,
    group_range,
)
from newton.tests.unittest_utils import USD_AVAILABLE

_MIXED_ASSET = os.path.join(os.path.dirname(__file__), "assets", "deformables_mixed.usda")

_CABLE_PTS = [(0.0, 0.0, 1.0), (0.1, 0.0, 1.0), (0.2, 0.0, 1.0), (0.3, 0.0, 1.0)]


@unittest.skipUnless(USD_AVAILABLE, "Requires usd-core")
class TestUSDDeformableObjects(unittest.TestCase):
    """Prim-path deformable object lookup on the finalized Model across lifecycle transformations."""

    def test_mixed_scene_objects_resolve_after_finalize(self):
        """Every family of the mixed scene resolves by prim path on the finalized Model."""
        builder = newton.ModelBuilder()
        builder.add_usd(_MIXED_ASSET)
        model = builder.finalize()

        cable = DeformableCurveView(model, "/World/CableA/sim")
        self.assertEqual(cable.count, 1)
        self.assertEqual(cable.bodies_per_deformable_object, 3)
        self.assertEqual(cable.elements_per_deformable_object("joint"), 2)  # open 3-segment chain
        cloth = DeformableSurfaceView(model, "/World/Cloth/sim")
        self.assertEqual(cloth.particles_per_deformable_object, 4)
        self.assertEqual(cloth.ranges("triangle"), [(0, 2)])
        soft = DeformableVolumeView(model, "/World/Soft*/sim")
        self.assertEqual(soft.count, 2)
        soft_ranges = soft.ranges("particle")
        self.assertNotEqual(soft_ranges[0], soft_ranges[1])
        self.assertEqual(soft.elements_per_deformable_object("tetrahedron"), 1)
        # No begin_world -> global deformable objects.
        self.assertEqual(cable.worlds, [-1])
        with self.assertRaises(KeyError):
            DeformableCurveView(model, "/World/DoesNotExist")

    def test_replicated_objects_select_per_world(self):
        """replicate() duplicates labels across worlds: one deformable object per world, ranges
        offset per world, and raw ranges come back in world order."""
        stage = _deformable_stage()
        cloth = _add_cloth_mesh(stage, "/World/Cloth")
        _author_deformable_element_array(cloth.GetPrim(), "thicknesses", [0.001], "constant")
        _bind_deformable_material(stage, cloth.GetPrim(), "/World/ClothMat")
        sub = newton.ModelBuilder()
        sub.add_usd(stage)
        scene = newton.ModelBuilder()
        scene.replicate(sub, 3)
        model = scene.finalize()

        view = DeformableSurfaceView(model, "/World/Cloth")
        self.assertEqual((view.count, view.world_count, view.count_per_world), (3, 3, 1))
        self.assertEqual(view.labels, ["/World/Cloth"] * 3)
        self.assertEqual(view.worlds, [0, 1, 2])
        self.assertEqual(view.ranges("particle"), [(4 * w, 4 * w + 4) for w in range(3)])
        self.assertEqual(list(view.starts("particle").numpy()), [0, 4, 8])

    def test_cable_objects_replicate_with_free_and_attached_roots(self):
        """Keep one selectable cable per world with either free or attached roots."""
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
                self.assertEqual(len(bodies), 3)
                self.assertEqual(len(joints), 2)
                if attached_point is not None:
                    (attachment,) = result["path_attachment_map"]["/World/Attachment"]
                    plug_body = result["path_body_map"]["/World/Plug"]
                    plug_joint = next(j for j, child in enumerate(source.joint_child) if child == plug_body)
                    self.assertEqual(source.joint_parent[attachment], plug_body)
                    self.assertEqual(source.joint_child[attachment], bodies[0 if attached_point == 0 else -1])
                    self.assertEqual(source.joint_articulation[attachment], source.joint_articulation[plug_joint])

                scene = newton.ModelBuilder()
                scene.replicate(source, 2)
                view = DeformableCurveView(scene.finalize(device="cpu"), "/World/Cable")
                self.assertEqual((view.count, view.worlds), (2, [0, 1]))
                self.assertEqual(
                    view.ranges("body"),
                    [(bodies[0] + w * source.body_count, bodies[-1] + 1 + w * source.body_count) for w in range(2)],
                )
                self.assertEqual(
                    view.ranges("joint"),
                    [(joints[0] + w * source.joint_count, joints[-1] + 1 + w * source.joint_count) for w in range(2)],
                )

    def test_heterogeneous_worlds_resolve_with_world_tags(self):
        """Worlds holding different deformables each resolve with the right world tag."""
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
        model = scene.finalize()

        self.assertEqual(DeformableSurfaceView(model, "/World/Cloth").worlds, [0])
        self.assertEqual(DeformableCurveView(model, "/World/Cable").worlds, [1])

    def test_cable_object_survives_fixed_joint_collapse(self):
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

        view = DeformableCurveView(model, "/World/Cable")
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
            view = DeformableCurveView(model, path)
            ((j0, j1),) = view.ranges("joint")
            self.assertEqual(j0, j1, "finalized welded-graph curves own no tree joints")
            self.assertLessEqual(j1, model.joint_count, f"{path}: finalized empty range points past the joint array")


if __name__ == "__main__":
    unittest.main(verbosity=2)
