# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Tests for select_joints."""

import re
import unittest

import numpy as np
import warp as wp

import newton
from newton.controllers import ControllerJointImpedance, select_joints

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _gains(total_controlled_dofs, value, device):
    """Return a compact (total_controlled_dofs,) float32 gain array filled with value."""
    return wp.full(total_controlled_dofs, value, dtype=wp.float32, device=device)


def _flat(data, device):
    """Return a flat float32 Warp array from any array-like."""
    return wp.array(np.array(data, dtype=np.float32).flatten(), dtype=wp.float32, device=device)


def _build_single_prismatic():
    """Build a one-robot, one-DOF prismatic-joint ModelBuilder."""
    builder = newton.ModelBuilder()
    link = builder.add_link()
    j = builder.add_joint_prismatic(
        parent=-1,
        child=link,
        axis=wp.vec3(1.0, 0.0, 0.0),
        parent_xform=wp.transform_identity(),
        child_xform=wp.transform_identity(),
    )
    builder.add_articulation([j], label="robot")
    return builder


def _build_two_robot_mixed():
    """Build a ModelBuilder with robot 0 (2 revolute DOFs) and robot 1 (1 prismatic DOF)."""
    builder = newton.ModelBuilder()
    # Robot 0: 2-DOF revolute chain
    l0a = builder.add_link()
    l0b = builder.add_link()
    j0a = builder.add_joint_revolute(
        parent=-1,
        child=l0a,
        axis=wp.vec3(0.0, 0.0, 1.0),
        parent_xform=wp.transform_identity(),
        child_xform=wp.transform_identity(),
    )
    j0b = builder.add_joint_revolute(
        parent=l0a,
        child=l0b,
        axis=wp.vec3(0.0, 0.0, 1.0),
        parent_xform=wp.transform(p=wp.vec3(1.0, 0.0, 0.0)),
        child_xform=wp.transform_identity(),
    )
    builder.add_articulation([j0a, j0b], label="robot0")
    # Robot 1: 1-DOF prismatic
    l1 = builder.add_link()
    j1 = builder.add_joint_prismatic(
        parent=-1,
        child=l1,
        axis=wp.vec3(1.0, 0.0, 0.0),
        parent_xform=wp.transform_identity(),
        child_xform=wp.transform_identity(),
    )
    builder.add_articulation([j1], label="robot1")
    return builder


def _build_ball_then_revolute():
    """Build a one-robot model whose base is an uncontrollable 3-DOF ball joint.

    The ball joint spans four coordinates but three DOFs, so every joint after
    it has a different coordinate index than DOF index.
    """
    builder = newton.ModelBuilder()
    base = builder.add_link()
    arm = builder.add_link()
    j_ball = builder.add_joint_ball(
        parent=-1,
        child=base,
        parent_xform=wp.transform_identity(),
        child_xform=wp.transform_identity(),
    )
    j_rev = builder.add_joint_revolute(
        parent=base,
        child=arm,
        axis=wp.vec3(0.0, 0.0, 1.0),
        parent_xform=wp.transform_identity(),
        child_xform=wp.transform_identity(),
    )
    builder.add_articulation([j_ball, j_rev], label="robot")
    return builder, j_ball, j_rev


def _build_floating_base_fleet():
    """Build robot 0 (free base + 2 revolute) and robot 1 (1 revolute).

    The free joint spans seven coordinates but six DOFs, so coordinate and DOF
    indices diverge and the two robots have different controlled-DOF counts.
    """
    builder = newton.ModelBuilder()
    base = builder.add_link(mass=1.0)
    a1 = builder.add_link(mass=1.0)
    a2 = builder.add_link(mass=1.0)
    jf = builder.add_joint_free(child=base)
    j1 = builder.add_joint_revolute(parent=base, child=a1, axis=wp.vec3(0.0, 0.0, 1.0))
    j2 = builder.add_joint_revolute(parent=a1, child=a2, axis=wp.vec3(0.0, 0.0, 1.0))
    builder.add_articulation([jf, j1, j2], label="robot0")
    link = builder.add_link(mass=1.0)
    j3 = builder.add_joint_revolute(parent=-1, child=link, axis=wp.vec3(0.0, 0.0, 1.0))
    builder.add_articulation([j3], label="robot1")
    return builder


class TestSelectJoints(unittest.TestCase):
    def test_single_robot_all_scalar_joints(self):
        """Verify select_joints defaults to every single-coordinate, single-DOF joint of the model."""
        device = wp.get_device()
        model = _build_single_prismatic().finalize(device=device)
        selection = select_joints(model)
        np.testing.assert_array_equal(selection.q_idx.numpy(), [0])
        np.testing.assert_array_equal(selection.qd_idx.numpy(), [0])

    def test_returns_int32_arrays(self):
        """Verify both index arrays are int32 so they work as Warp indexed-view subscripts."""
        device = wp.get_device()
        model = _build_single_prismatic().finalize(device=device)
        selection = select_joints(model)
        self.assertEqual(selection.q_idx.dtype, wp.int32)
        self.assertEqual(selection.qd_idx.dtype, wp.int32)
        # Directly usable as a subscript — this is the documented binding idiom.
        sim = wp.zeros(model.joint_dof_count, dtype=wp.float32, device=device)
        self.assertEqual(sim[selection.qd_idx].size, 1)

    def test_coordinate_and_dof_indices_differ_with_ball_joint(self):
        """Verify the two index spaces diverge once a multi-coordinate joint is present."""
        device = wp.get_device()
        builder, _j_ball, _j_rev = _build_ball_then_revolute()
        model = builder.finalize(device=device)
        selection = select_joints(model)
        np.testing.assert_array_equal(selection.q_idx.numpy(), [4])  # after the 4-coordinate quaternion
        np.testing.assert_array_equal(selection.qd_idx.numpy(), [3])  # after the 3-DOF angular velocity

    def test_heterogeneous_two_robots(self):
        """Verify select_joints concatenates controlled DOFs per articulation for a mixed fleet."""
        device = wp.get_device()
        model = _build_two_robot_mixed().finalize(device=device)
        selection = select_joints(model)
        self.assertEqual(selection.q_idx.numpy().size, 3)
        self.assertEqual(selection.qd_idx.numpy().size, 3)

    def test_uncontrolled_joint_excluded_by_default(self):
        """Verify select_joints skips a non-scalar joint rather than raising."""
        device = wp.get_device()
        builder, _j_ball, _j_rev = _build_ball_then_revolute()
        model = builder.finalize(device=device)
        selection = select_joints(model)
        self.assertEqual(selection.q_idx.numpy().size, 1)

    def test_explicit_non_scalar_joint_passed_through(self):
        """Verify select_joints does not itself validate joint type, deferring to the controller."""
        device = wp.get_device()
        builder = newton.ModelBuilder()
        link = builder.add_link()
        j_ball = builder.add_joint_ball(
            parent=-1,
            child=link,
            parent_xform=wp.transform_identity(),
            child_xform=wp.transform_identity(),
        )
        builder.add_articulation([j_ball], label="robot")
        model = builder.finalize(device=device)
        selection = select_joints(model, joints=[j_ball])
        np.testing.assert_array_equal(selection.q_idx.numpy(), [0])

    def test_articulation_glob_pattern_selects_every_match(self):
        """Verify a glob pattern selects every articulation whose label matches."""
        device = wp.get_device()
        model = _build_two_robot_mixed().finalize(device=device)
        matched = select_joints(model, articulations="robot*").qd_idx.numpy()
        np.testing.assert_array_equal(matched, select_joints(model).qd_idx.numpy())

    def test_articulation_selected_by_label(self):
        """Verify select_joints resolves an articulation label to its indices."""
        device = wp.get_device()
        model = _build_two_robot_mixed().finalize(device=device)
        selection = select_joints(model, articulations=["robot1"])
        self.assertEqual(selection.q_idx.numpy().size, 1)

    def test_duplicate_articulations_deduplicated(self):
        """Verify naming one articulation twice does not select its joints twice.

        An index and a label can resolve to the same articulation, which would
        otherwise duplicate every one of its DOFs in the output.
        """
        device = wp.get_device()
        model = _build_two_robot_mixed().finalize(device=device)
        by_index_twice = select_joints(model, articulations=[0, 0])
        by_index_and_label = select_joints(model, articulations=[0, "robot0"])
        expected = select_joints(model, articulations=[0]).q_idx.numpy()
        np.testing.assert_array_equal(by_index_twice.q_idx.numpy(), expected)
        np.testing.assert_array_equal(by_index_and_label.q_idx.numpy(), expected)

    def test_joint_selected_by_label(self):
        """Verify select_joints resolves a joint label to its index within the selected articulation."""
        device = wp.get_device()
        builder = newton.ModelBuilder()
        link = builder.add_link()
        arm = builder.add_link()
        j_shoulder = builder.add_joint_revolute(
            parent=-1,
            child=link,
            axis=wp.vec3(0.0, 0.0, 1.0),
            parent_xform=wp.transform_identity(),
            child_xform=wp.transform_identity(),
            label="shoulder",
        )
        j_elbow = builder.add_joint_revolute(
            parent=link,
            child=arm,
            axis=wp.vec3(0.0, 0.0, 1.0),
            parent_xform=wp.transform_identity(),
            child_xform=wp.transform_identity(),
            label="elbow",
        )
        builder.add_articulation([j_shoulder, j_elbow], label="robot")
        model = builder.finalize(device=device)
        selection = select_joints(model, joints=["shoulder"])
        np.testing.assert_array_equal(selection.q_idx.numpy(), [0])

    def test_joint_label_selects_every_match(self):
        """Verify a joint label matching two joints in one articulation selects both."""
        device = wp.get_device()
        builder = newton.ModelBuilder()
        link = builder.add_link()
        arm = builder.add_link()
        j0 = builder.add_joint_revolute(
            parent=-1,
            child=link,
            axis=wp.vec3(0.0, 0.0, 1.0),
            parent_xform=wp.transform_identity(),
            child_xform=wp.transform_identity(),
            label="finger",
        )
        j1 = builder.add_joint_revolute(
            parent=link,
            child=arm,
            axis=wp.vec3(0.0, 0.0, 1.0),
            parent_xform=wp.transform_identity(),
            child_xform=wp.transform_identity(),
            label="finger",
        )
        builder.add_articulation([j0, j1], label="robot")
        model = builder.finalize(device=device)
        selection = select_joints(model, joints=["finger"])
        self.assertEqual(selection.q_idx.numpy().size, 2)

    def test_joint_glob_pattern_matches_leaf_name_across_prefixed_fleet(self):
        """Verify a glob joint pattern matches by leaf name despite an add_builder label prefix.

        A pattern is expected to serve a whole fleet, e.g. ``"should*"``
        selecting the "shoulder" joint on every robot regardless of how each
        robot's joints were relabeled by ``add_builder(..., label_prefix=...)``.
        """
        device = wp.get_device()
        sub = newton.ModelBuilder()
        link = sub.add_link()
        j = sub.add_joint_revolute(
            parent=-1,
            child=link,
            axis=wp.vec3(0.0, 0.0, 1.0),
            parent_xform=wp.transform_identity(),
            child_xform=wp.transform_identity(),
            label="shoulder",
        )
        sub.add_articulation([j], label="arm")
        builder = newton.ModelBuilder()
        builder.add_builder(sub, label_prefix="robot_0")
        builder.add_builder(sub, label_prefix="robot_1")
        model = builder.finalize(device=device)
        selection = select_joints(model, joints=["should*"])
        self.assertEqual(selection.q_idx.numpy().size, 2)

    def test_joint_regex_pattern_matches_leaf_name_across_prefixed_fleet(self):
        """Verify a compiled regex joint pattern matches by leaf name despite an add_builder label prefix.

        A pattern is expected to serve a whole fleet, e.g. ``re.compile("should.r")``
        selecting the "shoulder" joint on every robot regardless of how each
        robot's joints were relabeled by ``add_builder(..., label_prefix=...)``.
        """
        device = wp.get_device()
        sub = newton.ModelBuilder()
        link = sub.add_link()
        j = sub.add_joint_revolute(
            parent=-1,
            child=link,
            axis=wp.vec3(0.0, 0.0, 1.0),
            parent_xform=wp.transform_identity(),
            child_xform=wp.transform_identity(),
            label="shoulder",
        )
        sub.add_articulation([j], label="arm")
        builder = newton.ModelBuilder()
        builder.add_builder(sub, label_prefix="robot_0")
        builder.add_builder(sub, label_prefix="robot_1")
        model = builder.finalize(device=device)
        selection = select_joints(model, joints=[re.compile("should.r")])
        self.assertEqual(selection.q_idx.numpy().size, 2)

    def test_articulation_label_matches_nothing_raises(self):
        """Verify select_joints raises when an articulation label matches nothing."""
        device = wp.get_device()
        model = _build_single_prismatic().finalize(device=device)
        with self.assertRaises(ValueError):
            select_joints(model, articulations=["nonexistent"])

    def test_joint_label_matches_nothing_raises(self):
        """Verify select_joints raises when a joint label matches nothing in the selected articulations."""
        device = wp.get_device()
        model = _build_single_prismatic().finalize(device=device)
        with self.assertRaises(ValueError):
            select_joints(model, joints=["nonexistent"])

    def test_joint_index_matches_nothing_raises(self):
        """Verify select_joints raises when an explicit joint index is outside every selected articulation."""
        device = wp.get_device()
        model = _build_single_prismatic().finalize(device=device)
        with self.assertRaises(ValueError):
            select_joints(model, joints=[99])

    def test_selection_drives_controller_end_to_end(self):
        """Verify a select_joints result wires into a controller and reaches the simulation.

        Covers the full documented path — resolve joints, construct, bind an
        indexed view as the torque output — on a model whose coordinate and DOF
        spaces differ.
        """
        device = wp.get_device()
        model = _build_floating_base_fleet().finalize(device=device)
        selection = select_joints(model)

        ctrl = ControllerJointImpedance(
            model,
            joint_selection=selection,
            stiffness=_gains(3, 1.0, device),
            damping=_gains(3, 0.0, device),
            use_gravity_compensation=False,
            use_coriolis_compensation=False,
            use_inertia_decoupling=False,
            device=device,
        )
        ins, outs = ctrl.input(), ctrl.output()
        sim_f = wp.zeros(model.joint_dof_count, dtype=wp.float32, device=device)
        outs.joint_f = sim_f[selection.qd_idx]
        ins.joint_q_des = _flat([1.0, 1.0, 1.0], device)
        ctrl.step(inputs=ins, outputs=outs, dt=0.01)

        written = sim_f.numpy()
        np.testing.assert_allclose(written[selection.qd_idx.numpy()], [1.0, 1.0, 1.0], atol=1e-4)
        # The free base's six DOFs must be untouched.
        untouched = np.setdiff1d(np.arange(model.joint_dof_count), selection.qd_idx.numpy())
        np.testing.assert_allclose(written[untouched], 0.0, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
