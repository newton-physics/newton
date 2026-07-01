# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Tests for USD deformable attachments and element collision filters on cable bodies."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import warp as wp

import newton
from newton.tests._usd_deformable_test_utils import (
    _add_cable_curve,
    _add_cloth_mesh,
    _add_element_collision_filter,
    _add_physics_attachment,
    _bind_deformable_material,
)
from newton.tests.unittest_utils import USD_AVAILABLE


@unittest.skipUnless(USD_AVAILABLE, "Requires usd-core")
class TestUSDDeformableAttachments(unittest.TestCase):
    """Proposal PhysicsAttachment + element-collision-filter import onto cable bodies."""

    def test_physics_attachment_segment_to_world_imports_ball_joint(self):
        """A proposal PhysicsAttachment from a cable segment to world imports as a point joint."""
        from pxr import Usd, UsdGeom, UsdPhysics

        with tempfile.TemporaryDirectory() as tmpdir:
            usd_path = Path(tmpdir) / "cable_attachment_world.usda"
            stage = Usd.Stage.CreateNew(str(usd_path))
            UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
            UsdPhysics.Scene.Define(stage, "/PhysicsScene")
            pts = [(0.0, 0.0, 1.0), (0.1, 0.0, 1.0), (0.2, 0.0, 1.0), (0.3, 0.0, 1.0)]
            curves = _add_cable_curve(stage, "/World/Cable", pts)
            _bind_deformable_material(stage, curves.GetPrim(), "/World/CableMat", thickness=0.02)
            _add_physics_attachment(
                stage,
                "/World/AttachMid",
                src0="/World/Cable",
                type0="segment",
                indices0=[1],
                coords0=[(0.5, 0.0, 0.0)],
                coords1=[(0.15, 0.0, 1.0)],
            )
            stage.Save()

            builder = newton.ModelBuilder()
            result = builder.add_usd(str(usd_path))

            bodies, _ = result["path_cable_map"]["/World/Cable"]
            joints = result["path_attachment_map"]["/World/AttachMid"]
            self.assertEqual(len(joints), 1)
            j = joints[0]
            self.assertEqual(builder.joint_type[j], newton.JointType.BALL)
            self.assertEqual(builder.joint_parent[j], -1)
            self.assertEqual(builder.joint_child[j], bodies[1])
            np.testing.assert_allclose(np.array(builder.joint_X_p[j].p), [0.15, 0.0, 1.0], atol=1e-6)
            np.testing.assert_allclose(np.array(builder.joint_X_c[j].p), [0.0, 0.0, 0.0], atol=1e-6)

    def test_physics_attachment_world_anchor_follows_import_xform(self):
        """A world-target anchor rides the import ``xform`` along with the cable geometry.

        Without transforming the world ``coords1``, the cable bodies move under ``xform`` but
        the world ball-joint anchor stays in original USD coordinates, pulling the cable off.
        """
        from pxr import Usd, UsdGeom, UsdPhysics

        with tempfile.TemporaryDirectory() as tmpdir:
            usd_path = Path(tmpdir) / "cable_attachment_world_xform.usda"
            stage = Usd.Stage.CreateNew(str(usd_path))
            UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
            UsdPhysics.Scene.Define(stage, "/PhysicsScene")
            pts = [(0.0, 0.0, 1.0), (0.1, 0.0, 1.0), (0.2, 0.0, 1.0), (0.3, 0.0, 1.0)]
            curves = _add_cable_curve(stage, "/World/Cable", pts)
            _bind_deformable_material(stage, curves.GetPrim(), "/World/CableMat", thickness=0.02)
            _add_physics_attachment(
                stage,
                "/World/AttachMid",
                src0="/World/Cable",
                type0="segment",
                indices0=[1],
                coords0=[(0.5, 0.0, 0.0)],
                coords1=[(0.15, 0.0, 1.0)],
            )
            stage.Save()

            builder = newton.ModelBuilder()
            result = builder.add_usd(str(usd_path), xform=wp.transform(wp.vec3(10.0, 0.0, 0.0), wp.quat_identity()))

            bodies, _ = result["path_cable_map"]["/World/Cable"]
            j = result["path_attachment_map"]["/World/AttachMid"][0]
            self.assertEqual(builder.joint_parent[j], -1)
            # The cable body and its world anchor both translate by xform's +10 in x.
            np.testing.assert_allclose(np.array(builder.body_q[bodies[1]].p)[0], 10.15, atol=1e-5)
            np.testing.assert_allclose(np.array(builder.joint_X_p[j].p), [10.15, 0.0, 1.0], atol=1e-5)

    def test_physics_attachment_interior_point_to_rigid_imports_incident_segment_joints(self):
        """A cable point attachment maps an interior point to both incident segment bodies."""
        from pxr import Usd, UsdGeom, UsdPhysics

        with tempfile.TemporaryDirectory() as tmpdir:
            usd_path = Path(tmpdir) / "cable_attachment_rigid.usda"
            stage = Usd.Stage.CreateNew(str(usd_path))
            UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
            UsdPhysics.Scene.Define(stage, "/PhysicsScene")
            rigid = UsdGeom.Xform.Define(stage, "/World/Rigid")
            UsdPhysics.RigidBodyAPI.Apply(rigid.GetPrim())
            pts = [(0.0, 0.0, 1.0), (0.1, 0.0, 1.0), (0.2, 0.0, 1.0), (0.3, 0.0, 1.0)]
            curves = _add_cable_curve(stage, "/World/Cable", pts)
            _bind_deformable_material(stage, curves.GetPrim(), "/World/CableMat", thickness=0.02)
            _add_physics_attachment(
                stage,
                "/World/AttachPoint",
                src0="/World/Cable",
                src1="/World/Rigid",
                type0="point",
                indices0=[1],
                coords1=[(0.1, 0.0, 1.0)],
            )
            stage.Save()

            builder = newton.ModelBuilder()
            result = builder.add_usd(str(usd_path))

            rigid_body = result["path_body_map"]["/World/Rigid"]
            cable_bodies, _ = result["path_cable_map"]["/World/Cable"]
            joints = result["path_attachment_map"]["/World/AttachPoint"]
            self.assertEqual(len(joints), 2)
            self.assertEqual({builder.joint_child[j] for j in joints}, {cable_bodies[0], cable_bodies[1]})
            self.assertTrue(all(builder.joint_parent[j] == rigid_body for j in joints))
            for j in joints:
                self.assertEqual(builder.joint_type[j], newton.JointType.BALL)
                np.testing.assert_allclose(np.array(builder.joint_X_p[j].p), [0.1, 0.0, 1.0], atol=1e-6)

    def test_physics_attachment_to_kinematic_body_finalizes(self):
        """A cable attached to a jointless kinematic body must finalize().

        The importer gives a jointless kinematic/floating rigid body its own base-joint
        articulation, then wraps the cable in its own. Both passes must emit joints in
        increasing order so articulation_start stays monotonic; otherwise finalize() rejects
        it. Regression for the StaticMeshAttach case where the attachment targets a kinematic
        anchor that carries no USD joint.
        """
        from pxr import Usd, UsdGeom, UsdPhysics

        with tempfile.TemporaryDirectory() as tmpdir:
            usd_path = Path(tmpdir) / "cable_attachment_kinematic.usda"
            stage = Usd.Stage.CreateNew(str(usd_path))
            UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
            UsdPhysics.Scene.Define(stage, "/PhysicsScene")
            # Kinematic anchor with a collider (so it gets a computed mass > 0) but no USD joint:
            # the importer gives it a base-joint articulation, which must be created before the
            # cable's own articulation so articulation_start stays monotonic. A massless anchor
            # would be skipped by the floating-body pass and would not reproduce the conflict.
            anchor = UsdGeom.Cube.Define(stage, "/World/Anchor")
            anchor.CreateSizeAttr(0.1)
            rigid_api = UsdPhysics.RigidBodyAPI.Apply(anchor.GetPrim())
            rigid_api.CreateKinematicEnabledAttr(True)
            UsdPhysics.CollisionAPI.Apply(anchor.GetPrim())
            pts = [(0.0, 0.0, 1.0), (0.1, 0.0, 1.0), (0.2, 0.0, 1.0), (0.3, 0.0, 1.0)]
            curves = _add_cable_curve(stage, "/World/Cable", pts)
            _bind_deformable_material(stage, curves.GetPrim(), "/World/CableMat", thickness=0.02)
            _add_physics_attachment(
                stage,
                "/World/AttachKinematic",
                src0="/World/Cable",
                src1="/World/Anchor",
                type0="point",
                indices0=[0],
                coords1=[(0.0, 0.0, 1.0)],
            )
            stage.Save()

            builder = newton.ModelBuilder()
            result = builder.add_usd(str(usd_path))
            self.assertIn("/World/Cable_articulation", builder.articulation_label)
            self.assertIn("/World/AttachKinematic", result["path_attachment_map"])

            # The regression: a non-monotonic articulation_start raised here before the fix.
            model = builder.finalize()
            self.assertGreater(model.body_count, 0)

    def test_element_collision_filter_filters_cable_segments_against_collider(self):
        """A PhysicsElementCollisionFilter suppresses collision between the named cable segments
        and a collider; unlisted segments stay collidable."""
        from pxr import Usd, UsdGeom, UsdPhysics

        with tempfile.TemporaryDirectory() as tmpdir:
            usd_path = Path(tmpdir) / "elem_filter.usda"
            stage = Usd.Stage.CreateNew(str(usd_path))
            UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
            UsdPhysics.Scene.Define(stage, "/PhysicsScene")
            box = UsdGeom.Cube.Define(stage, "/World/Box")
            box.CreateSizeAttr(0.1)
            UsdPhysics.RigidBodyAPI.Apply(box.GetPrim()).CreateKinematicEnabledAttr(True)
            UsdPhysics.CollisionAPI.Apply(box.GetPrim())
            pts = [(0.0, 0.0, 1.0), (0.1, 0.0, 1.0), (0.2, 0.0, 1.0), (0.3, 0.0, 1.0)]  # 3 segments
            curves = _add_cable_curve(stage, "/World/Cable", pts)
            _bind_deformable_material(stage, curves.GetPrim(), "/World/CableMat", thickness=0.02)
            # Filter the cable's first two segments (0, 1) against all of the box; empty indices1 = all.
            _add_element_collision_filter(
                stage, "/World/Filter", src0="/World/Cable", src1="/World/Box", indices0=[0, 1], indices1=[]
            )
            stage.Save()

            builder = newton.ModelBuilder()
            result = builder.add_usd(str(usd_path))
            seg_bodies, _ = result["path_cable_map"]["/World/Cable"]
            seg_shapes = [builder.body_shapes[b][0] for b in seg_bodies]
            box_shape = builder.body_shapes[result["path_body_map"]["/World/Box"]][0]
            pairs = {tuple(sorted(p)) for p in builder.shape_collision_filter_pairs}
            self.assertIn(tuple(sorted((seg_shapes[0], box_shape))), pairs)
            self.assertIn(tuple(sorted((seg_shapes[1], box_shape))), pairs)
            self.assertNotIn(tuple(sorted((seg_shapes[2], box_shape))), pairs, "segment 2 was not listed")

    def _two_cable_filter_stage(self, tmpdir, **filter_kwargs):
        """Two 3-segment cables plus a PhysicsElementCollisionFilter; returns (builder, result, pairs)."""
        from pxr import Usd, UsdGeom, UsdPhysics

        usd_path = Path(tmpdir) / "two_cable_filter.usda"
        stage = Usd.Stage.CreateNew(str(usd_path))
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
        UsdPhysics.Scene.Define(stage, "/PhysicsScene")
        a = [(0.0, 0.0, 1.0), (0.1, 0.0, 1.0), (0.2, 0.0, 1.0), (0.3, 0.0, 1.0)]  # 3 segments
        b = [(0.0, 1.0, 1.0), (0.1, 1.0, 1.0), (0.2, 1.0, 1.0), (0.3, 1.0, 1.0)]  # 3 segments
        _add_cable_curve(stage, "/World/CableA", a)
        _add_cable_curve(stage, "/World/CableB", b)
        _add_element_collision_filter(
            stage, "/World/Filter", src0="/World/CableA", src1="/World/CableB", **filter_kwargs
        )
        stage.Save()

        builder = newton.ModelBuilder()
        result = builder.add_usd(str(usd_path))
        pairs = {tuple(sorted(p)) for p in builder.shape_collision_filter_pairs}
        return builder, result, pairs

    @staticmethod
    def _cable_seg_shapes(builder, result, path):
        bodies, _ = result["path_cable_map"][path]
        return [builder.body_shapes[b][0] for b in bodies]

    def test_element_collision_filter_paired_groups(self):
        """groupElemCounts pair indices element-wise: only the paired (i, j) elements filter,
        not the full Cartesian product of the two index arrays."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # counts [1, 1] / [1, 1] pairs (A0 with B2) and (A1 with B0) only.
            builder, result, pairs = self._two_cable_filter_stage(
                tmpdir, indices0=[0, 1], counts0=[1, 1], indices1=[2, 0], counts1=[1, 1]
            )
            a = self._cable_seg_shapes(builder, result, "/World/CableA")
            b = self._cable_seg_shapes(builder, result, "/World/CableB")
            self.assertIn(tuple(sorted((a[0], b[2]))), pairs)
            self.assertIn(tuple(sorted((a[1], b[0]))), pairs)
            # The cross-product pairs that a non-paired reading would add must be absent.
            self.assertNotIn(tuple(sorted((a[0], b[0]))), pairs, "cross-product pair must not be filtered")
            self.assertNotIn(tuple(sorted((a[1], b[2]))), pairs, "cross-product pair must not be filtered")

    def test_element_collision_filter_zero_count_means_all(self):
        """A groupElemCount of 0 selects all elements of that source for the paired group."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # src0 group is count-0 (all of CableA) paired with CableB segments {0, 1}.
            builder, result, pairs = self._two_cable_filter_stage(
                tmpdir, indices0=[], counts0=[0], indices1=[0, 1], counts1=[2]
            )
            a = self._cable_seg_shapes(builder, result, "/World/CableA")
            b = self._cable_seg_shapes(builder, result, "/World/CableB")
            for sa in a:  # all of CableA filtered against B0 and B1
                self.assertIn(tuple(sorted((sa, b[0]))), pairs)
                self.assertIn(tuple(sorted((sa, b[1]))), pairs)
                self.assertNotIn(tuple(sorted((sa, b[2]))), pairs, "B segment 2 was not in the group")

    def test_element_collision_filter_single_group_broadcasts(self):
        """A single group on one side (e.g. empty counts) broadcasts against every group of the other."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # src0 has no counts -> one implicit all-elements group; src1 has two single-element groups.
            builder, result, pairs = self._two_cable_filter_stage(tmpdir, indices0=[], indices1=[0, 1], counts1=[1, 1])
            a = self._cable_seg_shapes(builder, result, "/World/CableA")
            b = self._cable_seg_shapes(builder, result, "/World/CableB")
            for sa in a:
                self.assertIn(tuple(sorted((sa, b[0]))), pairs)
                self.assertIn(tuple(sorted((sa, b[1]))), pairs)
                self.assertNotIn(tuple(sorted((sa, b[2]))), pairs, "B segment 2 was not in any group")

    def test_element_collision_filter_malformed_counts_warns_and_skips(self):
        """groupElemCounts whose sum exceeds the index array warns and applies no filter pairs."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with self.assertWarnsRegex(UserWarning, "sum exceeds"):
                builder, result, pairs = self._two_cable_filter_stage(
                    tmpdir, indices0=[0], counts0=[2], indices1=[0], counts1=[1]
                )
            # No cross-source pair is added (intra-cable adjacency filters from add_rod still exist).
            a = self._cable_seg_shapes(builder, result, "/World/CableA")
            b = self._cable_seg_shapes(builder, result, "/World/CableB")
            cross = {tuple(sorted((sa, sb))) for sa in a for sb in b}
            self.assertTrue(cross.isdisjoint(pairs), "a malformed counts array must add no cross-source filter pairs")

    def test_element_collision_filter_resolves_child_collider(self):
        """A filter source that is a child collider under a rigid Xform resolves to that shape."""
        from pxr import Usd, UsdGeom, UsdPhysics

        with tempfile.TemporaryDirectory() as tmpdir:
            usd_path = Path(tmpdir) / "child_collider.usda"
            stage = Usd.Stage.CreateNew(str(usd_path))
            UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
            UsdPhysics.Scene.Define(stage, "/PhysicsScene")
            # Rigid body Xform with the collider on a *child* geom (not the body prim itself).
            rigid = UsdGeom.Xform.Define(stage, "/World/Rigid")
            UsdPhysics.RigidBodyAPI.Apply(rigid.GetPrim()).CreateKinematicEnabledAttr(True)
            collider = UsdGeom.Cube.Define(stage, "/World/Rigid/Collider")
            collider.CreateSizeAttr(0.1)
            UsdPhysics.CollisionAPI.Apply(collider.GetPrim())
            pts = [(0.0, 0.0, 1.0), (0.1, 0.0, 1.0), (0.2, 0.0, 1.0), (0.3, 0.0, 1.0)]
            _add_cable_curve(stage, "/World/Cable", pts)
            _add_element_collision_filter(
                stage, "/World/Filter", src0="/World/Cable", src1="/World/Rigid/Collider", indices0=[0], indices1=[]
            )
            stage.Save()

            builder = newton.ModelBuilder()
            result = builder.add_usd(str(usd_path))
            seg0 = self._cable_seg_shapes(builder, result, "/World/Cable")[0]
            collider_shape = result["path_shape_map"]["/World/Rigid/Collider"]
            pairs = {tuple(sorted(p)) for p in builder.shape_collision_filter_pairs}
            self.assertIn(tuple(sorted((seg0, collider_shape))), pairs)

    def test_element_collision_filter_resolves_bodyless_static_collider(self):
        """A filter source that is a bodyless static collider resolves to its shape."""
        from pxr import Usd, UsdGeom, UsdPhysics

        with tempfile.TemporaryDirectory() as tmpdir:
            usd_path = Path(tmpdir) / "static_collider.usda"
            stage = Usd.Stage.CreateNew(str(usd_path))
            UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
            UsdPhysics.Scene.Define(stage, "/PhysicsScene")
            # Static collider: CollisionAPI but no RigidBodyAPI, so it has no body in path_body_map.
            ground = UsdGeom.Cube.Define(stage, "/World/Ground")
            ground.CreateSizeAttr(0.1)
            UsdPhysics.CollisionAPI.Apply(ground.GetPrim())
            pts = [(0.0, 0.0, 1.0), (0.1, 0.0, 1.0), (0.2, 0.0, 1.0), (0.3, 0.0, 1.0)]
            _add_cable_curve(stage, "/World/Cable", pts)
            _add_element_collision_filter(
                stage, "/World/Filter", src0="/World/Cable", src1="/World/Ground", indices0=[0], indices1=[]
            )
            stage.Save()

            builder = newton.ModelBuilder()
            result = builder.add_usd(str(usd_path))
            seg0 = self._cable_seg_shapes(builder, result, "/World/Cable")[0]
            ground_shape = result["path_shape_map"]["/World/Ground"]
            pairs = {tuple(sorted(p)) for p in builder.shape_collision_filter_pairs}
            self.assertIn(tuple(sorted((seg0, ground_shape))), pairs)

    def test_disabled_physics_attachment_is_recorded_but_not_imported(self):
        """attachmentEnabled=false preserves attrs but creates no joint."""
        from pxr import Usd, UsdGeom, UsdPhysics

        with tempfile.TemporaryDirectory() as tmpdir:
            usd_path = Path(tmpdir) / "disabled_attachment.usda"
            stage = Usd.Stage.CreateNew(str(usd_path))
            UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
            UsdPhysics.Scene.Define(stage, "/PhysicsScene")
            pts = [(0.0, 0.0, 1.0), (0.1, 0.0, 1.0), (0.2, 0.0, 1.0), (0.3, 0.0, 1.0)]
            curves = _add_cable_curve(stage, "/World/Cable", pts)
            _bind_deformable_material(stage, curves.GetPrim(), "/World/CableMat", thickness=0.02)
            _add_physics_attachment(
                stage,
                "/World/AttachDisabled",
                src0="/World/Cable",
                type0="segment",
                indices0=[0],
                coords0=[(0.5, 0.0, 0.0)],
                coords1=[(0.05, 0.0, 1.0)],
                enabled=False,
            )
            stage.Save()

            builder = newton.ModelBuilder()
            result = builder.add_usd(str(usd_path))

            self.assertNotIn("/World/AttachDisabled", result["path_attachment_map"])
            self.assertFalse(result["path_attachment_attrs"]["/World/AttachDisabled"]["enabled"])

    def test_physics_attachment_cloth_source_warns_and_preserves_attrs(self):
        """Cloth/volume attachments are surfaced but not lowered to fake constraints."""
        from pxr import Usd, UsdGeom, UsdPhysics

        with tempfile.TemporaryDirectory() as tmpdir:
            usd_path = Path(tmpdir) / "cloth_attachment.usda"
            stage = Usd.Stage.CreateNew(str(usd_path))
            UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
            UsdPhysics.Scene.Define(stage, "/PhysicsScene")
            _add_cloth_mesh(stage, "/World/Cloth")
            _add_physics_attachment(
                stage,
                "/World/AttachCloth",
                src0="/World/Cloth",
                type0="point",
                indices0=[0],
                coords1=[(0.0, 0.0, 1.0)],
            )
            stage.Save()

            builder = newton.ModelBuilder()
            with self.assertWarnsRegex(UserWarning, "cloth/volume"):
                result = builder.add_usd(str(usd_path))

            self.assertNotIn("/World/AttachCloth", result["path_attachment_map"])
            attrs = result["path_attachment_attrs"]["/World/AttachCloth"]
            self.assertEqual(attrs["src0"], "/World/Cloth")
            self.assertIn("unsupported_reason", attrs)


if __name__ == "__main__":
    unittest.main(verbosity=2)
