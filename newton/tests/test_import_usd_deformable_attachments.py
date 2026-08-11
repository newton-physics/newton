# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Tests for USD deformable attachments and element collision filters on cable bodies."""

import math
import unittest

import numpy as np
import warp as wp

import newton
from newton.tests._usd_deformable_test_utils import (
    _add_cable_curve,
    _add_cloth_mesh,
    _add_element_collision_filter,
    _add_physics_attachment,
    _deformable_stage,
    group_range,
)
from newton.tests.unittest_utils import USD_AVAILABLE


@unittest.skipUnless(USD_AVAILABLE, "Requires usd-core")
class TestUSDDeformableAttachments(unittest.TestCase):
    """Proposal PhysicsAttachment + element-collision-filter import onto cable bodies."""

    def test_result_maps_stay_valid_after_collapse(self):
        """With collapse_fixed_joints=True, the returned cable and attachment indices
        are remapped to valid, correctly-labelled entries (the documented contract),
        and joint_indices in the attachment attrs match the attachment map."""
        from pxr import UsdGeom, UsdPhysics

        stage = _deformable_stage()
        # A rigid fixed pair collapses, shifting every body/joint index after it.
        for name in ("A", "B"):
            UsdPhysics.RigidBodyAPI.Apply(UsdGeom.Xform.Define(stage, f"/World/{name}").GetPrim())
        fixed = UsdPhysics.FixedJoint.Define(stage, "/World/Fix")
        fixed.CreateBody0Rel().SetTargets(["/World/A"])
        fixed.CreateBody1Rel().SetTargets(["/World/B"])
        pts = [(0.0, 0.0, 1.0), (0.1, 0.0, 1.0), (0.2, 0.0, 1.0), (0.3, 0.0, 1.0)]
        _add_cable_curve(stage, "/World/Cable", pts)
        _add_physics_attachment(
            stage,
            "/World/Anchor",
            src0="/World/Cable",
            type0="point",
            indices0=[0],
            coords1=[(0.0, 0.0, 1.0)],
        )

        builder = newton.ModelBuilder()
        result = builder.add_usd(stage, collapse_fixed_joints=True, return_deformable_results=True)

        bodies, joints = result["path_cable_map"]["/World/Cable"]
        self.assertTrue(all("/World/Cable" in builder.body_label[b] for b in bodies))
        self.assertTrue(all("/World/Cable" in builder.joint_label[j] for j in joints))
        anchor_joints = result["path_attachment_map"]["/World/Anchor"]
        self.assertEqual(len(anchor_joints), 1)
        self.assertTrue(all(0 <= j < builder.joint_count for j in anchor_joints))
        self.assertEqual(result["path_attachment_attrs"]["/World/Anchor"]["joint_indices"], list(anchor_joints))
        builder.finalize()

    def test_element_filter_disabled_and_unsupported_sources_skip(self):
        """filterEnabled=false skips the filter; a cloth element source warns and skips."""
        with self.subTest(case="disabled"):

            def build(enabled):
                stage = _deformable_stage()
                pts = [(0.0, 0.0, 1.0), (0.1, 0.0, 1.0), (0.2, 0.0, 1.0)]
                _add_cable_curve(stage, "/World/CableA", pts)
                _add_cable_curve(stage, "/World/CableB", [(0.0, 0.1, 1.0), (0.1, 0.1, 1.0), (0.2, 0.1, 1.0)])
                if enabled is not None:
                    _add_element_collision_filter(
                        stage, "/World/Filter", src0="/World/CableA", src1="/World/CableB", enabled=enabled
                    )
                builder = newton.ModelBuilder()
                builder.add_usd(stage)
                return len(builder.shape_collision_filter_pairs)

            baseline = build(None)  # add_rod's own adjacent-segment filters
            self.assertGreater(build(True), baseline)
            self.assertEqual(build(False), baseline)

        with self.subTest(case="cloth_source"):
            stage = _deformable_stage()
            _add_cloth_mesh(stage, "/World/Cloth")
            _add_cable_curve(stage, "/World/Cable", [(0.0, 0.0, 1.0), (0.1, 0.0, 1.0), (0.2, 0.0, 1.0)])
            _add_element_collision_filter(stage, "/World/Filter", src0="/World/Cloth", src1="/World/Cable")
            builder = newton.ModelBuilder()
            with self.assertWarnsRegex(UserWarning, "/World/Filter"):
                builder.add_usd(stage)

    def test_invalid_attachment_stiffness_is_preserved_not_hardened(self):
        """Only +inf selects the hard path (the proposal's attachment sentinel with
        range [0, inf]): NaN, -inf, and negative stiffness or damping warn and are
        preserved as metadata instead of silently becoming hard joints."""
        for label, kwargs in (
            ("nan_stiffness", {"stiffness": float("nan")}),
            ("neg_inf_stiffness", {"stiffness": float("-inf")}),
            ("negative_stiffness", {"stiffness": -5.0}),
            ("negative_damping", {"damping": -1.0}),
        ):
            with self.subTest(kind=label):
                stage = _deformable_stage()
                pts = [(0.0, 0.0, 1.0), (0.1, 0.0, 1.0), (0.2, 0.0, 1.0)]
                _add_cable_curve(stage, "/World/Cable", pts)
                _add_physics_attachment(
                    stage,
                    "/World/Att",
                    src0="/World/Cable",
                    type0="point",
                    indices0=[0],
                    coords1=[(0.0, 0.0, 1.0)],
                    **kwargs,
                )
                builder = newton.ModelBuilder()
                with self.assertWarnsRegex(UserWarning, "invalid PhysicsAttachment"):
                    result = builder.add_usd(stage, return_deformable_results=True)
                self.assertNotIn("/World/Att", result["path_attachment_map"])
                self.assertIn("unsupported_reason", result["path_attachment_attrs"]["/World/Att"])
                builder.finalize()

    def test_compliant_attachment_is_preserved_not_hardened(self):
        """A finite-stiffness (compliant) attachment is preserved as metadata instead of
        being silently lowered into a hard joint: authored physics is not changed, the
        attrs keep the authored stiffness/damping, and no joint is created."""
        stage = _deformable_stage()
        pts = [(0.0, 0.0, 1.0), (0.1, 0.0, 1.0), (0.2, 0.0, 1.0), (0.3, 0.0, 1.0)]
        _add_cable_curve(stage, "/World/Cable", pts)
        _add_physics_attachment(
            stage,
            "/World/SoftAnchor",
            src0="/World/Cable",
            type0="point",
            indices0=[0],
            coords1=[(0.0, 0.0, 1.0)],
            stiffness=500.0,
            damping=2.0,
        )

        builder = newton.ModelBuilder()
        with self.assertWarnsRegex(UserWarning, "stiffness"):
            result = builder.add_usd(stage, return_deformable_results=True)

        # Only the cable's own joints exist; the compliant attachment created none.
        j0, j1 = group_range(builder, "cable", "/World/Cable", "joint")
        self.assertEqual(builder.joint_count, j1 - j0)
        self.assertNotIn("/World/SoftAnchor", result["path_attachment_map"])
        attrs = result["path_attachment_attrs"]["/World/SoftAnchor"]
        self.assertEqual(attrs["stiffness"], 500.0)
        self.assertEqual(attrs["damping"], 2.0)
        self.assertIn("unsupported_reason", attrs)
        builder.finalize()

    def test_damped_hard_attachment_imports_joint(self):
        """A +inf-stiffness attachment with nonzero damping is hard per the proposal
        (damping only applies when the constraint is not hard) and imports as a ball
        joint instead of being preserved as unsupported metadata."""
        stage = _deformable_stage()
        pts = [(0.0, 0.0, 1.0), (0.1, 0.0, 1.0), (0.2, 0.0, 1.0), (0.3, 0.0, 1.0)]
        _add_cable_curve(stage, "/World/Cable", pts)
        _add_physics_attachment(
            stage,
            "/World/HardDamped",
            src0="/World/Cable",
            type0="point",
            indices0=[0],
            coords1=[(0.0, 0.0, 1.0)],
            stiffness=math.inf,
            damping=5.0,
        )

        builder = newton.ModelBuilder()
        result = builder.add_usd(stage, return_deformable_results=True)

        joints = result["path_attachment_map"]["/World/HardDamped"]
        self.assertEqual(len(joints), 1)
        self.assertEqual(builder.joint_type[joints[0]], newton.JointType.BALL)
        self.assertNotIn("unsupported_reason", result["path_attachment_attrs"]["/World/HardDamped"])
        builder.finalize()

    def test_physics_attachment_segment_to_world_imports_ball_joint(self):
        """A segment-to-world PhysicsAttachment imports as a world ball joint whose anchor
        rides the import ``xform`` along with the cable geometry.

        Without transforming the world ``coords1``, the cable bodies move under ``xform`` but
        the world ball-joint anchor stays in original USD coordinates, pulling the cable off.
        The asymmetric u = 0.25 also pins the proposal's segment-coordinate convention:
        p = u*x0 + (1-u)*x1, so u weights the segment START vertex (u = 1 selects the start,
        u = 0 the end), and with the start at body-local -L/2 the site sits at (0.5-u)*L.
        """
        stage = _deformable_stage()
        pts = [(0.0, 0.0, 1.0), (0.1, 0.0, 1.0), (0.2, 0.0, 1.0), (0.3, 0.0, 1.0)]
        _add_cable_curve(stage, "/World/Cable", pts)
        # Segment 1 runs x0 = (0.1, 0, 1) -> x1 = (0.2, 0, 1), L = 0.1.
        # u = 0.25 -> p = 0.25*x0 + 0.75*x1 = (0.175, 0, 1), authored as the world target too.
        _add_physics_attachment(
            stage,
            "/World/AttachMid",
            src0="/World/Cable",
            type0="segment",
            indices0=[1],
            coords0=[(0.25, 0.0, 0.0)],
            coords1=[(0.175, 0.0, 1.0)],
        )

        builder = newton.ModelBuilder()
        result = builder.add_usd(
            stage, xform=wp.transform(wp.vec3(10.0, 0.0, 0.0), wp.quat_identity()), return_deformable_results=True
        )

        b0, _ = group_range(builder, "cable", "/World/Cable", "body")
        joints = result["path_attachment_map"]["/World/AttachMid"]
        self.assertEqual(len(joints), 1)
        j = joints[0]
        self.assertEqual(builder.joint_type[j], newton.JointType.BALL)
        self.assertEqual(builder.joint_parent[j], -1)
        self.assertEqual(builder.joint_child[j], b0 + 1)
        # The cable body and its world anchor both translate by xform's +10 in x.
        np.testing.assert_allclose(np.array(builder.body_q[b0 + 1].p)[0], 10.15, atol=1e-5)
        np.testing.assert_allclose(np.array(builder.joint_X_p[j].p), [10.175, 0.0, 1.0], atol=1e-5)
        # Child-local anchor: z = (0.5 - u) * L = 0.025, invariant under xform.
        np.testing.assert_allclose(np.array(builder.joint_X_c[j].p), [0.0, 0.0, 0.025], atol=1e-6)
        # Both joint frames name the same world point (a flipped u sign puts the child
        # anchor at world x = 10.125 and this fails).
        child_anchor_world = wp.transform_point(builder.body_q[builder.joint_child[j]], builder.joint_X_c[j].p)
        np.testing.assert_allclose(np.array(child_anchor_world), np.array(builder.joint_X_p[j].p), atol=1e-5)

    def test_physics_attachment_interior_point_imports_single_joint(self):
        """A point attachment site is a single point-point constraint per the proposal, so
        an interior cable point (which borders two segment bodies) creates exactly one ball
        joint, anchored to one flanking body at the shared vertex, not one joint per
        incident segment."""
        from pxr import UsdGeom, UsdPhysics

        stage = _deformable_stage()
        rigid = UsdGeom.Xform.Define(stage, "/World/Rigid")
        UsdPhysics.RigidBodyAPI.Apply(rigid.GetPrim())
        pts = [(0.0, 0.0, 1.0), (0.1, 0.0, 1.0), (0.2, 0.0, 1.0), (0.3, 0.0, 1.0)]
        _add_cable_curve(stage, "/World/Cable", pts)
        _add_physics_attachment(
            stage,
            "/World/AttachPoint",
            src0="/World/Cable",
            src1="/World/Rigid",
            type0="point",
            indices0=[1],
            coords1=[(0.1, 0.0, 1.0)],
        )

        builder = newton.ModelBuilder()
        result = builder.add_usd(stage, return_deformable_results=True)

        rigid_body = result["path_body_map"]["/World/Rigid"]
        b0, b1 = group_range(builder, "cable", "/World/Cable", "body")
        joints = result["path_attachment_map"]["/World/AttachPoint"]
        self.assertEqual(len(joints), 1)
        j = joints[0]
        self.assertEqual(builder.joint_type[j], newton.JointType.BALL)
        self.assertEqual(builder.joint_parent[j], rigid_body)
        self.assertIn(builder.joint_child[j], range(b0, b1))
        # Both frames name the authored vertex: the parent anchor directly, and the child
        # anchor through its body transform (so the single joint pins the shared point).
        np.testing.assert_allclose(np.array(builder.joint_X_p[j].p), [0.1, 0.0, 1.0], atol=1e-6)
        child_anchor_world = wp.transform_point(builder.body_q[builder.joint_child[j]], builder.joint_X_c[j].p)
        np.testing.assert_allclose(np.array(child_anchor_world), [0.1, 0.0, 1.0], atol=1e-6)

    def test_physics_attachment_to_kinematic_body_finalizes(self):
        """A cable attached to a jointless kinematic body must finalize().

        The importer gives a jointless kinematic/floating rigid body its own base-joint
        articulation, then wraps the cable in its own. Both passes must emit joints in
        increasing order so articulation_start stays monotonic; otherwise finalize() rejects
        it. Regression for the StaticMeshAttach case where the attachment targets a kinematic
        anchor that carries no USD joint.
        """
        from pxr import UsdGeom, UsdPhysics

        stage = _deformable_stage()
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
        _add_cable_curve(stage, "/World/Cable", pts)
        _add_physics_attachment(
            stage,
            "/World/AttachKinematic",
            src0="/World/Cable",
            src1="/World/Anchor",
            type0="point",
            indices0=[0],
            coords1=[(0.0, 0.0, 1.0)],
        )

        builder = newton.ModelBuilder()
        result = builder.add_usd(stage, return_deformable_results=True)
        self.assertIn("/World/Cable_articulation", builder.articulation_label)
        self.assertIn("/World/AttachKinematic", result["path_attachment_map"])

        # The regression: a non-monotonic articulation_start raised here before the fix.
        model = builder.finalize()
        self.assertGreater(model.body_count, 0)

    def test_physics_attachment_disabled_or_unsupported_is_recorded_not_imported(self):
        """Disabled and cloth/volume-source attachments create no joints; both preserve their
        authored attrs (enabled flag / unsupported_reason), and unsupported sources warn."""
        stage = _deformable_stage()
        pts = [(0.0, 0.0, 1.0), (0.1, 0.0, 1.0), (0.2, 0.0, 1.0), (0.3, 0.0, 1.0)]
        _add_cable_curve(stage, "/World/Cable", pts)
        _add_cloth_mesh(stage, "/World/Cloth")
        # attachmentEnabled=false on a supported cable-segment source.
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
        # Cloth/volume sources are surfaced but not lowered to fake constraints.
        _add_physics_attachment(
            stage,
            "/World/AttachCloth",
            src0="/World/Cloth",
            type0="point",
            indices0=[0],
            coords1=[(0.0, 0.0, 1.0)],
        )

        builder = newton.ModelBuilder()
        with self.assertWarnsRegex(UserWarning, "cloth/volume"):
            result = builder.add_usd(stage, return_deformable_results=True)

        # Neither policy case lowers to a joint.
        self.assertNotIn("/World/AttachDisabled", result["path_attachment_map"])
        self.assertNotIn("/World/AttachCloth", result["path_attachment_map"])
        self.assertEqual(len(result["path_attachment_map"]), 0)
        self.assertFalse(result["path_attachment_attrs"]["/World/AttachDisabled"]["enabled"])
        attrs = result["path_attachment_attrs"]["/World/AttachCloth"]
        self.assertEqual(attrs["src0"], "/World/Cloth")
        self.assertIn("unsupported_reason", attrs)

    def _two_cable_filter_stage(self, **filter_kwargs):
        """Two 3-segment cables plus a PhysicsElementCollisionFilter; returns (builder, result, pairs)."""
        stage = _deformable_stage()
        a = [(0.0, 0.0, 1.0), (0.1, 0.0, 1.0), (0.2, 0.0, 1.0), (0.3, 0.0, 1.0)]  # 3 segments
        b = [(0.0, 1.0, 1.0), (0.1, 1.0, 1.0), (0.2, 1.0, 1.0), (0.3, 1.0, 1.0)]  # 3 segments
        _add_cable_curve(stage, "/World/CableA", a)
        _add_cable_curve(stage, "/World/CableB", b)
        _add_element_collision_filter(
            stage, "/World/Filter", src0="/World/CableA", src1="/World/CableB", **filter_kwargs
        )

        builder = newton.ModelBuilder()
        result = builder.add_usd(stage, return_deformable_results=True)
        pairs = {tuple(sorted(p)) for p in builder.shape_collision_filter_pairs}
        return builder, result, pairs

    @staticmethod
    def _cable_seg_shapes(builder, path):
        b0, b1 = group_range(builder, "cable", path, "body")
        return [builder.body_shapes[b][0] for b in range(b0, b1)]

    def test_element_collision_filter_paired_groups(self):
        """groupElemCounts pair indices element-wise: only the paired (i, j) elements filter,
        not the full Cartesian product of the two index arrays."""
        # counts [1, 1] / [1, 1] pairs (A0 with B2) and (A1 with B0) only.
        builder, _result, pairs = self._two_cable_filter_stage(
            indices0=[0, 1], counts0=[1, 1], indices1=[2, 0], counts1=[1, 1]
        )
        a = self._cable_seg_shapes(builder, "/World/CableA")
        b = self._cable_seg_shapes(builder, "/World/CableB")
        self.assertIn(tuple(sorted((a[0], b[2]))), pairs)
        self.assertIn(tuple(sorted((a[1], b[0]))), pairs)
        # The cross-product pairs that a non-paired reading would add must be absent.
        self.assertNotIn(tuple(sorted((a[0], b[0]))), pairs, "cross-product pair must not be filtered")
        self.assertNotIn(tuple(sorted((a[1], b[2]))), pairs, "cross-product pair must not be filtered")

    def test_element_collision_filter_all_elements_group_broadcasts(self):
        """An all-elements src0 group — an explicit groupElemCount of 0 or an absent counts
        array (one implicit group) — is paired against every listed src1 group."""
        cases = {
            # src0 group is count-0 (all of CableA) paired with CableB segments {0, 1}.
            "zero count": {"indices0": [], "counts0": [0], "indices1": [0, 1], "counts1": [2]},
            # src0 has no counts -> one implicit all-elements group broadcast against two
            # single-element src1 groups.
            "empty counts": {"indices0": [], "indices1": [0, 1], "counts1": [1, 1]},
        }
        for name, kwargs in cases.items():
            with self.subTest(name):
                builder, _result, pairs = self._two_cable_filter_stage(**kwargs)
                a = self._cable_seg_shapes(builder, "/World/CableA")
                b = self._cable_seg_shapes(builder, "/World/CableB")
                for sa in a:  # all of CableA filtered against B0 and B1
                    self.assertIn(tuple(sorted((sa, b[0]))), pairs)
                    self.assertIn(tuple(sorted((sa, b[1]))), pairs)
                    self.assertNotIn(tuple(sorted((sa, b[2]))), pairs, "B segment 2 was not in any group")

    def test_element_collision_filter_explicit_singleton_does_not_broadcast(self):
        """An explicit single group (counts=[n]) must pair one-to-one, not broadcast: against
        two groups on the other side it is a group-count mismatch that warns and skips. Only
        the empty-counts form (no groupElemCounts authored) pairs against all groups."""
        with self.assertWarnsRegex(UserWarning, "pair one-to-one"):
            builder, _result, pairs = self._two_cable_filter_stage(
                indices0=[0], counts0=[1], indices1=[0, 1], counts1=[1, 1]
            )
        a = self._cable_seg_shapes(builder, "/World/CableA")
        b = self._cable_seg_shapes(builder, "/World/CableB")
        cross = {tuple(sorted((sa, sb))) for sa in a for sb in b}
        self.assertTrue(cross.isdisjoint(pairs), "an explicit singleton group must not broadcast")

    def test_element_collision_filter_empty_counts_select_all_and_broadcast(self):
        """Empty groupElemCounts means ALL elements of that source, paired against every group
        of the other side (the proposal defines group boundaries only through the counts
        array, so stray indices without counts define no subset and are ignored with a
        warning)."""
        with self.subTest(case="stray_indices_ignored"):
            with self.assertWarnsRegex(UserWarning, "indices are ignored"):
                builder, _result, pairs = self._two_cable_filter_stage(indices0=[1], indices1=[0, 2], counts1=[1, 1])
            a = self._cable_seg_shapes(builder, "/World/CableA")
            b = self._cable_seg_shapes(builder, "/World/CableB")
            for sa in a:  # ALL of CableA, including segments not in the stray indices
                self.assertIn(tuple(sorted((sa, b[0]))), pairs)
                self.assertIn(tuple(sorted((sa, b[2]))), pairs)
                self.assertNotIn(tuple(sorted((sa, b[1]))), pairs, "B segment 1 was not in any group")

        with self.subTest(case="both_sides_empty"):
            builder, _result, pairs = self._two_cable_filter_stage()
            a = self._cable_seg_shapes(builder, "/World/CableA")
            b = self._cable_seg_shapes(builder, "/World/CableB")
            for sa in a:  # all elements vs all elements
                for sb in b:
                    self.assertIn(tuple(sorted((sa, sb))), pairs)

    def test_element_collision_filter_deduplicates_pairs(self):
        """Overlapping groups and a self-filter's mirrored (sa, sb)/(sb, sa) orderings repeat
        the same shape combination; the filter adds each normalized pair once. The pair uses
        non-adjacent segments because add_rod files its own adjacent-segment filters."""
        stage = _deformable_stage()
        pts = [(0.0, 0.0, 1.0), (0.1, 0.0, 1.0), (0.2, 0.0, 1.0), (0.3, 0.0, 1.0)]
        _add_cable_curve(stage, "/World/CableA", pts)
        # Three group pairings that all normalize to (A0, A2): forward, mirrored, repeated.
        _add_element_collision_filter(
            stage,
            "/World/Filter",
            src0="/World/CableA",
            src1="/World/CableA",
            indices0=[0, 2, 0],
            counts0=[1, 1, 1],
            indices1=[2, 0, 2],
            counts1=[1, 1, 1],
        )

        builder = newton.ModelBuilder()
        builder.add_usd(stage)

        a = self._cable_seg_shapes(builder, "/World/CableA")
        raw = [tuple(sorted(p)) for p in builder.shape_collision_filter_pairs]
        self.assertEqual(raw.count(tuple(sorted((a[0], a[2])))), 1, "pair must be stored exactly once")

    def test_element_collision_filter_malformed_counts_warns_and_skips(self):
        """groupElemCounts whose sum exceeds the index array warns and applies no filter pairs."""
        with self.assertWarnsRegex(UserWarning, "sum exceeds"):
            builder, _result, pairs = self._two_cable_filter_stage(indices0=[0], counts0=[2], indices1=[0], counts1=[1])
        # No cross-source pair is added (intra-cable adjacency filters from add_rod still exist).
        a = self._cable_seg_shapes(builder, "/World/CableA")
        b = self._cable_seg_shapes(builder, "/World/CableB")
        cross = {tuple(sorted((sa, sb))) for sa in a for sb in b}
        self.assertTrue(cross.isdisjoint(pairs), "a malformed counts array must add no cross-source filter pairs")

    def test_element_collision_filter_resolves_collider_sources(self):
        """The rigid-side filter source resolves to its shape whether the collider sits on the
        rigid body prim itself, on a child geom under a rigid Xform, or on a bodyless static
        prim; only the listed cable segments are filtered, unlisted segments stay collidable."""
        from pxr import UsdGeom, UsdPhysics

        stage = _deformable_stage()
        # Collider on the rigid body prim itself.
        box = UsdGeom.Cube.Define(stage, "/World/Box")
        box.CreateSizeAttr(0.1)
        UsdPhysics.RigidBodyAPI.Apply(box.GetPrim()).CreateKinematicEnabledAttr(True)
        UsdPhysics.CollisionAPI.Apply(box.GetPrim())
        # Rigid body Xform with the collider on a *child* geom (not the body prim itself).
        rigid = UsdGeom.Xform.Define(stage, "/World/Rigid")
        UsdPhysics.RigidBodyAPI.Apply(rigid.GetPrim()).CreateKinematicEnabledAttr(True)
        collider = UsdGeom.Cube.Define(stage, "/World/Rigid/Collider")
        collider.CreateSizeAttr(0.1)
        UsdPhysics.CollisionAPI.Apply(collider.GetPrim())
        # Static collider: CollisionAPI but no RigidBodyAPI, so it has no body in path_body_map.
        ground = UsdGeom.Cube.Define(stage, "/World/Ground")
        ground.CreateSizeAttr(0.1)
        UsdPhysics.CollisionAPI.Apply(ground.GetPrim())

        pts = [(0.0, 0.0, 1.0), (0.1, 0.0, 1.0), (0.2, 0.0, 1.0), (0.3, 0.0, 1.0)]  # 3 segments
        _add_cable_curve(stage, "/World/Cable", pts)
        # Filter the cable's first two segments (0, 1) against all of the box (explicit counts
        # select the subset; empty indices1 with no counts = all elements of the box).
        _add_element_collision_filter(
            stage, "/World/FilterBox", src0="/World/Cable", src1="/World/Box", indices0=[0, 1], counts0=[2], indices1=[]
        )
        _add_element_collision_filter(
            stage,
            "/World/FilterChild",
            src0="/World/Cable",
            src1="/World/Rigid/Collider",
            indices0=[0],
            counts0=[1],
            indices1=[],
        )
        _add_element_collision_filter(
            stage,
            "/World/FilterGround",
            src0="/World/Cable",
            src1="/World/Ground",
            indices0=[0],
            counts0=[1],
            indices1=[],
        )

        builder = newton.ModelBuilder()
        result = builder.add_usd(stage, return_deformable_results=True)
        seg_shapes = self._cable_seg_shapes(builder, "/World/Cable")
        pairs = {tuple(sorted(p)) for p in builder.shape_collision_filter_pairs}

        box_shape = builder.body_shapes[result["path_body_map"]["/World/Box"]][0]
        self.assertIn(tuple(sorted((seg_shapes[0], box_shape))), pairs)
        self.assertIn(tuple(sorted((seg_shapes[1], box_shape))), pairs)
        self.assertNotIn(tuple(sorted((seg_shapes[2], box_shape))), pairs, "segment 2 was not listed")

        collider_shape = result["path_shape_map"]["/World/Rigid/Collider"]
        self.assertIn(tuple(sorted((seg_shapes[0], collider_shape))), pairs)

        ground_shape = result["path_shape_map"]["/World/Ground"]
        self.assertIn(tuple(sorted((seg_shapes[0], ground_shape))), pairs)


if __name__ == "__main__":
    unittest.main(verbosity=2)
