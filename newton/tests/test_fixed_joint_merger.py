# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Tests for solver-level fixed-joint merging.

Covers:
- compute_fixed_joint_merge analysis (CPU-only, no device loop)
- SolverXPBD and SolverSemiImplicit integration (device loop via add_function_test)
"""

import unittest

import numpy as np
import warp as wp

import newton
from newton import ModelBuilder
from newton._src.solvers._fixed_joint_merger import compute_fixed_joint_merge
from newton.solvers import SolverNotifyFlags, SolverSemiImplicit, SolverXPBD
from newton.tests.unittest_utils import add_function_test, get_test_devices

_DEFAULT_OFFSET = wp.vec3(0, 1, 0)


def _make_two_body_fixed(device, mass0=1.0, mass1=2.0, offset=_DEFAULT_OFFSET):
    """Return a model with body0-(FREE)->world and body0-(FIXED)->body1."""
    b = ModelBuilder(gravity=0.0)
    b0 = b.add_body(mass=mass0)
    b1 = b.add_body(mass=mass1, xform=wp.transform(offset, wp.quat_identity()))
    b.add_joint_free(b0)
    b.add_joint_fixed(b0, b1, parent_xform=wp.transform(offset, wp.quat_identity()))
    return b.finalize(device=device), b0, b1


# ---------------------------------------------------------------------------
# CPU-only analysis tests
# ---------------------------------------------------------------------------


class TestComputeFixedJointMerge(unittest.TestCase):
    def test_no_fixed_joints_returns_none(self):
        b = ModelBuilder(gravity=0.0)
        b0 = b.add_body(mass=1.0)
        b1 = b.add_body(mass=1.0)
        b.add_joint_free(b0)
        b.add_joint_revolute(b0, b1, axis=newton.Axis.Y)
        model = b.finalize(device="cpu")
        result = compute_fixed_joint_merge(model)
        self.assertIsNone(result)

    def test_no_bodies_returns_none(self):
        b = ModelBuilder(gravity=0.0)
        model = b.finalize(device="cpu")
        result = compute_fixed_joint_merge(model)
        self.assertIsNone(result)

    def test_single_fixed_joint_merge(self):
        model, b0, b1 = _make_two_body_fixed("cpu")
        info = compute_fixed_joint_merge(model)
        self.assertIsNotNone(info)
        self.assertTrue(info.has_merges)
        # b1 merges into b0
        self.assertEqual(info.survivor_of[b0], b0)
        self.assertEqual(info.survivor_of[b1], b0)
        # b0 gets combined mass
        combined = 1.0 + 2.0
        self.assertAlmostEqual(info.merged_body_mass[b0], combined, places=5)
        self.assertAlmostEqual(info.merged_body_inv_mass[b0], 1.0 / combined, places=5)
        # b1 is zeroed
        self.assertAlmostEqual(info.merged_body_inv_mass[b1], 0.0, places=10)

    def test_chained_fixed_joints(self):
        """A-(FREE)->world, A-(FIXED)->B, B-(FIXED)->C: B and C both merge into A."""
        b = ModelBuilder(gravity=0.0)
        bA = b.add_body(mass=1.0)
        bB = b.add_body(mass=1.0, xform=wp.transform(wp.vec3(0, 1, 0), wp.quat_identity()))
        bC = b.add_body(mass=1.0, xform=wp.transform(wp.vec3(0, 2, 0), wp.quat_identity()))
        b.add_joint_free(bA)
        b.add_joint_fixed(bA, bB, parent_xform=wp.transform(wp.vec3(0, 1, 0), wp.quat_identity()))
        b.add_joint_fixed(bB, bC, parent_xform=wp.transform(wp.vec3(0, 1, 0), wp.quat_identity()))
        model = b.finalize(device="cpu")

        info = compute_fixed_joint_merge(model)
        self.assertIsNotNone(info)
        self.assertEqual(info.survivor_of[bA], bA)
        self.assertEqual(info.survivor_of[bB], bA)
        self.assertEqual(info.survivor_of[bC], bA)
        self.assertAlmostEqual(info.merged_body_mass[bA], 3.0, places=5)
        self.assertAlmostEqual(info.merged_body_inv_mass[bB], 0.0, places=10)
        self.assertAlmostEqual(info.merged_body_inv_mass[bC], 0.0, places=10)

    def test_joints_to_keep_exempts_joint(self):
        """joints_to_keep preserves the named FIXED joint."""
        b = ModelBuilder(gravity=0.0)
        b0 = b.add_body(mass=1.0)
        b1 = b.add_body(mass=2.0, xform=wp.transform(wp.vec3(0, 1, 0), wp.quat_identity()))
        b.add_joint_free(b0)
        b.add_joint_fixed(b0, b1, parent_xform=wp.transform(wp.vec3(0, 1, 0), wp.quat_identity()), label="keep_me")
        model = b.finalize(device="cpu")

        info = compute_fixed_joint_merge(model, joints_to_keep=["keep_me"])
        # No merges should happen since the only fixed joint is in keep list.
        self.assertIsNone(info)

    def test_accumulated_inv_mass_correctness(self):
        """Two 1-kg bodies: merged inv_mass = 0.5."""
        model, b0, _b1 = _make_two_body_fixed("cpu", mass0=1.0, mass1=1.0)
        info = compute_fixed_joint_merge(model)
        self.assertIsNotNone(info)
        self.assertAlmostEqual(info.merged_body_inv_mass[b0], 0.5, places=5)

    def test_gpu_arrays_allocated_on_correct_device(self):
        """GPU arrays in FixedJointMergeInfo land on model.device."""
        model, _, _ = _make_two_body_fixed("cpu")
        info = compute_fixed_joint_merge(model)
        self.assertIsNotNone(info)
        for arr in [
            info.survivor_indices_gpu,
            info.relative_xforms_gpu,
            info.merged_body_inv_mass_gpu,
            info.merged_body_inv_inertia_gpu,
        ]:
            self.assertEqual(arr.device, model.device)

    def test_collapse_false_disables_merge_in_xpbd(self):
        """collapse_fixed_joints=False leaves _merge_info as None."""
        model, _, _ = _make_two_body_fixed("cpu")
        solver = SolverXPBD(model, collapse_fixed_joints=False)
        self.assertIsNone(solver._merge_info)
        # joint_enabled_effective should be the model array itself
        self.assertIs(solver.joint_enabled_effective, model.joint_enabled)

    def test_collapse_false_disables_merge_in_semi_implicit(self):
        """collapse_fixed_joints=False leaves _merge_info as None in SemiImplicit."""
        model, _, _ = _make_two_body_fixed("cpu")
        solver = SolverSemiImplicit(model, collapse_fixed_joints=False)
        self.assertIsNone(solver._merge_info)
        self.assertIs(solver.joint_enabled_effective, model.joint_enabled)

    def test_notify_merges_to_no_merges_clears_merge_info_xpbd(self):
        """Topology change from merges to no merges clears stale _merge_info."""
        b = ModelBuilder(gravity=0.0)
        b0 = b.add_body(mass=1.0)
        b1 = b.add_body(mass=2.0, xform=wp.transform(wp.vec3(0, 1, 0), wp.quat_identity()))
        b.add_joint_free(b0)
        b.add_joint_fixed(b0, b1, parent_xform=wp.transform(wp.vec3(0, 1, 0), wp.quat_identity()), label="the_joint")
        model = b.finalize(device="cpu")

        solver = SolverXPBD(model)
        self.assertIsNotNone(solver._merge_info)

        # Now exempt the only fixed joint — should collapse to no merges.
        solver._joints_to_keep = ["the_joint"]
        solver.notify_model_changed(SolverNotifyFlags.BODY_INERTIAL_PROPERTIES)

        self.assertIsNone(solver._merge_info)
        self.assertIs(solver.joint_enabled_effective, model.joint_enabled)

    def test_notify_merges_to_no_merges_clears_merge_info_semi_implicit(self):
        """Topology change from merges to no merges clears stale _merge_info in SemiImplicit."""
        b = ModelBuilder(gravity=0.0)
        b0 = b.add_body(mass=1.0)
        b1 = b.add_body(mass=2.0, xform=wp.transform(wp.vec3(0, 1, 0), wp.quat_identity()))
        b.add_joint_free(b0)
        b.add_joint_fixed(b0, b1, parent_xform=wp.transform(wp.vec3(0, 1, 0), wp.quat_identity()), label="the_joint")
        model = b.finalize(device="cpu")

        solver = SolverSemiImplicit(model)
        self.assertIsNotNone(solver._merge_info)

        solver._joints_to_keep = ["the_joint"]
        solver.notify_model_changed(SolverNotifyFlags.BODY_INERTIAL_PROPERTIES)

        self.assertIsNone(solver._merge_info)
        self.assertIs(solver.joint_enabled_effective, model.joint_enabled)


# ---------------------------------------------------------------------------
# Device tests for XPBD
# ---------------------------------------------------------------------------


class TestSolverXPBDFixedJointCollapse(unittest.TestCase):
    pass


def test_xpbd_merged_child_pose_follows_survivor(test, device):
    """After N steps body_q[child] equals body_q[survivor] * relative_xform."""
    offset = wp.vec3(0.0, 1.5, 0.0)
    model, b0, b1 = _make_two_body_fixed(device, mass0=1.0, mass1=1.0, offset=offset)
    solver = SolverXPBD(model)
    mi = solver._merge_info
    test.assertIsNotNone(mi)

    state0, state1 = model.state(), model.state()
    newton.eval_fk(model, model.joint_q, model.joint_qd, state0)
    contacts = model.contacts()

    for _ in range(5):
        model.collide(state0, contacts)
        solver.step(state0, state1, None, contacts, 1.0 / 60.0)
        state0, state1 = state1, state0

    body_q_np = state0.body_q.numpy()
    rel_xform = mi.relative_xform_of[b1]
    survivor_q = wp.transform(*body_q_np[b0])
    expected = survivor_q * rel_xform
    actual = wp.transform(*body_q_np[b1])
    for i in range(7):
        test.assertAlmostEqual(float(actual[i]), float(expected[i]), places=4)


def test_xpbd_no_nans_in_state(test, device):
    """All body_q/body_qd entries are finite after stepping."""
    model, _, _ = _make_two_body_fixed(device)
    solver = SolverXPBD(model)
    state0, state1 = model.state(), model.state()
    newton.eval_fk(model, model.joint_q, model.joint_qd, state0)
    contacts = model.contacts()

    for _ in range(10):
        model.collide(state0, contacts)
        solver.step(state0, state1, None, contacts, 1.0 / 60.0)
        state0, state1 = state1, state0

    test.assertTrue(np.all(np.isfinite(state0.body_q.numpy())), "body_q contains non-finite values")
    test.assertTrue(np.all(np.isfinite(state0.body_qd.numpy())), "body_qd contains non-finite values")


def test_xpbd_notify_model_changed_refreshes_merge(test, device):
    """Updating body mass then calling notify_model_changed updates merged inv_mass."""
    model, b0, b1 = _make_two_body_fixed(device, mass0=1.0, mass1=1.0)
    solver = SolverXPBD(model)
    test.assertIsNotNone(solver._merge_info)

    # Combined mass was 2.0 → inv_mass = 0.5
    old_inv = solver._merge_info.merged_body_inv_mass_gpu.numpy()[b0]
    test.assertAlmostEqual(float(old_inv), 0.5, places=4)

    # Update model mass for b1 to 3.0
    new_mass = model.body_mass.numpy()
    new_mass[b1] = 3.0
    model.body_mass.assign(new_mass)
    new_inv_mass = model.body_inv_mass.numpy()
    new_inv_mass[b1] = 1.0 / 3.0
    model.body_inv_mass.assign(new_inv_mass)

    solver.notify_model_changed(SolverNotifyFlags.BODY_INERTIAL_PROPERTIES)

    # Combined mass is now 1.0 + 3.0 = 4.0 → inv_mass = 0.25
    new_inv = solver._merge_info.merged_body_inv_mass_gpu.numpy()[b0]
    test.assertAlmostEqual(float(new_inv), 0.25, places=4)


add_function_test(
    TestSolverXPBDFixedJointCollapse,
    "test_merged_child_pose_follows_survivor",
    test_xpbd_merged_child_pose_follows_survivor,
    devices=get_test_devices(),
)
add_function_test(
    TestSolverXPBDFixedJointCollapse, "test_no_nans_in_state", test_xpbd_no_nans_in_state, devices=get_test_devices()
)
add_function_test(
    TestSolverXPBDFixedJointCollapse,
    "test_notify_model_changed_refreshes_merge",
    test_xpbd_notify_model_changed_refreshes_merge,
    devices=get_test_devices(),
)


# ---------------------------------------------------------------------------
# Device tests for SemiImplicit
# ---------------------------------------------------------------------------


class TestSolverSemiImplicitFixedJointCollapse(unittest.TestCase):
    pass


def test_semi_merged_child_pose_follows_survivor(test, device):
    """After N steps body_q[child] equals body_q[survivor] * relative_xform."""
    offset = wp.vec3(0.0, 1.5, 0.0)
    model, b0, b1 = _make_two_body_fixed(device, mass0=1.0, mass1=1.0, offset=offset)
    solver = SolverSemiImplicit(model)
    mi = solver._merge_info
    test.assertIsNotNone(mi)

    state0, state1 = model.state(), model.state()
    newton.eval_fk(model, model.joint_q, model.joint_qd, state0)

    for _ in range(5):
        solver.step(state0, state1, None, None, 1.0 / 60.0)
        state0, state1 = state1, state0

    body_q_np = state0.body_q.numpy()
    rel_xform = mi.relative_xform_of[b1]
    survivor_q = wp.transform(*body_q_np[b0])
    expected = survivor_q * rel_xform
    actual = wp.transform(*body_q_np[b1])
    for i in range(7):
        test.assertAlmostEqual(float(actual[i]), float(expected[i]), places=4)


def test_semi_no_nans_in_state(test, device):
    """All body_q/body_qd entries are finite after stepping."""
    model, _, _ = _make_two_body_fixed(device)
    solver = SolverSemiImplicit(model)
    state0, state1 = model.state(), model.state()
    newton.eval_fk(model, model.joint_q, model.joint_qd, state0)

    for _ in range(10):
        solver.step(state0, state1, None, None, 1.0 / 60.0)
        state0, state1 = state1, state0

    test.assertTrue(np.all(np.isfinite(state0.body_q.numpy())), "body_q contains non-finite values")
    test.assertTrue(np.all(np.isfinite(state0.body_qd.numpy())), "body_qd contains non-finite values")


add_function_test(
    TestSolverSemiImplicitFixedJointCollapse,
    "test_merged_child_pose_follows_survivor",
    test_semi_merged_child_pose_follows_survivor,
    devices=get_test_devices(),
)
add_function_test(
    TestSolverSemiImplicitFixedJointCollapse,
    "test_no_nans_in_state",
    test_semi_no_nans_in_state,
    devices=get_test_devices(),
)


if __name__ == "__main__":
    unittest.main(verbosity=2)
