# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Tests for solver-level fixed-joint merging.

Covers:
- compute_fixed_joint_merge analysis (CPU-only, no device loop)
- SolverXPBD and SolverSemiImplicit integration (device loop via add_function_test)
- effective-index layer (shape_body, joint_parent/child, joint_X_p/X_c)
- body_f scatter from merged children to survivor
- disabled-fixed-joint handling
"""

import unittest

import numpy as np
import warp as wp

import newton
from newton import ModelBuilder
from newton._src.solvers._fixed_joint_merger import (
    _propagate_merged_body_velocities,
    compute_fixed_joint_merge,
)
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
        merged_mass = info.merged_body_mass_gpu.numpy()
        merged_inv_mass = info.merged_body_inv_mass_gpu.numpy()
        self.assertAlmostEqual(float(merged_mass[b0]), combined, places=5)
        self.assertAlmostEqual(float(merged_inv_mass[b0]), 1.0 / combined, places=5)
        # b1 is zeroed
        self.assertAlmostEqual(float(merged_inv_mass[b1]), 0.0, places=10)

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
        merged_mass = info.merged_body_mass_gpu.numpy()
        merged_inv_mass = info.merged_body_inv_mass_gpu.numpy()
        self.assertAlmostEqual(float(merged_mass[bA]), 3.0, places=5)
        self.assertAlmostEqual(float(merged_inv_mass[bB]), 0.0, places=10)
        self.assertAlmostEqual(float(merged_inv_mass[bC]), 0.0, places=10)

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

    def test_disabled_fixed_joint_not_collapsed(self):
        """A FIXED joint with joint_enabled=False must not collapse its child."""
        b = ModelBuilder(gravity=0.0)
        b0 = b.add_body(mass=1.0)
        b1 = b.add_body(mass=2.0, xform=wp.transform(wp.vec3(0, 1, 0), wp.quat_identity()))
        b.add_joint_free(b0)
        j = b.add_joint_fixed(b0, b1, parent_xform=wp.transform(wp.vec3(0, 1, 0), wp.quat_identity()))
        model = b.finalize(device="cpu")
        # Disable the FIXED joint after finalize.
        je = model.joint_enabled.numpy()
        je[j] = False
        model.joint_enabled.assign(je)

        info = compute_fixed_joint_merge(model)
        # Disabled fixed joint should leave the topology unchanged → no merges.
        self.assertIsNone(info)

    def test_accumulated_inv_mass_correctness(self):
        """Two 1-kg bodies: merged inv_mass = 0.5."""
        model, b0, _b1 = _make_two_body_fixed("cpu", mass0=1.0, mass1=1.0)
        info = compute_fixed_joint_merge(model)
        self.assertIsNotNone(info)
        merged_inv_mass = info.merged_body_inv_mass_gpu.numpy()
        self.assertAlmostEqual(float(merged_inv_mass[b0]), 0.5, places=5)

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
            info.joint_enabled_effective_gpu,
            info.joint_parent_effective_gpu,
            info.joint_child_effective_gpu,
            info.joint_X_p_effective_gpu,
            info.joint_X_c_effective_gpu,
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

    def test_constructor_positional_args_unchanged_xpbd(self):
        """Old positional callers like SolverXPBD(model, 4) still bind iterations correctly."""
        model, _, _ = _make_two_body_fixed("cpu")
        solver = SolverXPBD(model, 4)
        self.assertEqual(solver.iterations, 4)

    def test_constructor_positional_args_unchanged_semi_implicit(self):
        """Old positional callers like SolverSemiImplicit(model, 0.1) bind angular_damping correctly."""
        model, _, _ = _make_two_body_fixed("cpu")
        solver = SolverSemiImplicit(model, 0.1)
        self.assertAlmostEqual(solver.angular_damping, 0.1)

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

    def test_effective_shape_body_remaps_to_survivor(self):
        """Shapes attached to merged children must resolve to the survivor's body slot."""
        b = ModelBuilder(gravity=0.0)
        b0 = b.add_body(mass=1.0)
        b1 = b.add_body(mass=1.0, xform=wp.transform(wp.vec3(0, 1, 0), wp.quat_identity()))
        b.add_joint_free(b0)
        b.add_joint_fixed(b0, b1, parent_xform=wp.transform(wp.vec3(0, 1, 0), wp.quat_identity()))
        # One shape on each body.
        b.add_shape_sphere(b0, radius=0.1)
        s_on_child = b.add_shape_sphere(b1, radius=0.1)
        model = b.finalize(device="cpu")

        info = compute_fixed_joint_merge(model)
        self.assertIsNotNone(info)
        shape_body_eff = info.shape_body_effective_gpu.numpy()
        # Shape on the merged child b1 should now resolve to b0 (the survivor).
        self.assertEqual(int(shape_body_eff[s_on_child]), b0)

    def test_effective_downstream_joint_anchors_remapped(self):
        """A joint downstream of a merged subtree gets parent remapped to the survivor."""
        # Topology: A-(FREE)->world, A-(FIXED)->B, B-(REVOLUTE)->C
        # B merges into A; the revolute joint's parent should point at A and its
        # parent anchor should be expressed in A's frame.
        b = ModelBuilder(gravity=0.0)
        bA = b.add_body(mass=1.0)
        bB = b.add_body(mass=1.0, xform=wp.transform(wp.vec3(0, 1, 0), wp.quat_identity()))
        bC = b.add_body(mass=1.0, xform=wp.transform(wp.vec3(1, 1, 0), wp.quat_identity()))
        b.add_joint_free(bA)
        b.add_joint_fixed(bA, bB, parent_xform=wp.transform(wp.vec3(0, 1, 0), wp.quat_identity()))
        # Anchor on B's frame: simple identity-translated anchor.
        rev = b.add_joint_revolute(
            bB,
            bC,
            axis=newton.Axis.Z,
            parent_xform=wp.transform(wp.vec3(1, 0, 0), wp.quat_identity()),
        )
        model = b.finalize(device="cpu")

        info = compute_fixed_joint_merge(model)
        self.assertIsNotNone(info)
        parent_eff = info.joint_parent_effective_gpu.numpy()
        # Revolute parent must have been remapped from B → A.
        self.assertEqual(int(parent_eff[rev]), bA)
        # And the anchor for the revolute should now be expressed relative to A.
        # We can verify by reconstructing: anchor_in_A = relative_xform_of[B] * joint_X_p[rev]
        rel_B = info.relative_xform_of[bB]
        Xp_orig = wp.transform(*model.joint_X_p.numpy()[rev])
        expected = rel_B * Xp_orig
        Xp_eff = info.joint_X_p_effective_gpu.numpy()[rev]
        actual = wp.transform(*Xp_eff)
        for i in range(7):
            self.assertAlmostEqual(float(actual[i]), float(expected[i]), places=5)

    def test_fixed_joint_to_kinematic_parent_not_collapsed(self):
        """A dynamic body fixed to a kinematic parent must not be merged."""
        b = ModelBuilder(gravity=0.0)
        # Kinematic root (no inbound joint to world).
        kp = b.add_body(mass=1.0)
        b.body_flags[kp] = int(newton.BodyFlags.KINEMATIC)
        # Dynamic child rigidly attached.
        dc = b.add_body(mass=2.0, xform=wp.transform(wp.vec3(0, 1, 0), wp.quat_identity()))
        b.add_joint_fixed(kp, dc, parent_xform=wp.transform(wp.vec3(0, 1, 0), wp.quat_identity()))
        model = b.finalize(device="cpu")

        info = compute_fixed_joint_merge(model)
        # No merges should happen — the dynamic child must stay as its own body.
        self.assertIsNone(info)

    def test_merged_inertia_is_about_combined_com(self):
        """Merged inertia must be expressed about the new combined COM (parallel-axis identity)."""
        L = 2.0
        m = 1.0
        ident = wp.mat33(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
        b = ModelBuilder(gravity=0.0)
        b0 = b.add_body(mass=m, inertia=ident)
        b1 = b.add_body(mass=m, xform=wp.transform(wp.vec3(0, L, 0), wp.quat_identity()), inertia=ident)
        b.add_joint_free(b0)
        b.add_joint_fixed(b0, b1, parent_xform=wp.transform(wp.vec3(0, L, 0), wp.quat_identity()))
        model = b.finalize(device="cpu")

        info = compute_fixed_joint_merge(model)
        self.assertIsNotNone(info)

        merged_com = info.merged_body_com_gpu.numpy()[b0]
        merged_I = info.merged_body_inertia_gpu.numpy()[b0]
        self.assertAlmostEqual(float(merged_com[1]), L / 2.0, places=5)

        # Diagonal cross-term must be m*L^2/2, not m*L^2.
        cross_xx = float(merged_I[0, 0]) - float(merged_I[1, 1])
        cross_zz = float(merged_I[2, 2]) - float(merged_I[1, 1])
        self.assertAlmostEqual(cross_xx, m * L * L / 2.0, places=4)
        self.assertAlmostEqual(cross_zz, m * L * L / 2.0, places=4)

    def test_constraint_body_nested_under_dynamic_not_merged(self):
        """A body in an equality constraint must not be merged, even when nested under a dynamic ancestor."""
        b = ModelBuilder(gravity=0.0)
        bA = b.add_body(mass=1.0)
        bB = b.add_body(mass=1.0, xform=wp.transform(wp.vec3(0, 1, 0), wp.quat_identity()))
        bC = b.add_body(mass=1.0, xform=wp.transform(wp.vec3(0, 2, 0), wp.quat_identity()))
        bD = b.add_body(mass=1.0, xform=wp.transform(wp.vec3(2, 0, 0), wp.quat_identity()))
        b.add_joint_free(bA)
        b.add_joint_free(bD)
        # FIXED chain: A -(FIXED)-> B -(FIXED)-> C, with C in an equality.
        b.add_joint_fixed(bA, bB, parent_xform=wp.transform(wp.vec3(0, 1, 0), wp.quat_identity()))
        b.add_joint_fixed(bB, bC, parent_xform=wp.transform(wp.vec3(0, 1, 0), wp.quat_identity()))
        b.add_equality_constraint_connect(body1=bC, body2=bD, anchor=wp.vec3(0.0))
        model = b.finalize(device="cpu")

        info = compute_fixed_joint_merge(model)
        # bC participates in an equality constraint and must stay its own body.
        if info is not None:
            self.assertEqual(info.survivor_of[bC], bC)

    def test_disabled_fixed_joint_keeps_joint_enabled_effective_false(self):
        """A model-level disabled FIXED joint must remain disabled in joint_enabled_effective."""
        # Use chain A-FIXED-B-FIXED-C, disable the first FIXED joint.  The DFS
        # would then bail out of merging at the disabled joint, but the effective
        # enabled flag must still mirror the model's disabled flag so the
        # downstream solver doesn't try to enforce it as a constraint either.
        b = ModelBuilder(gravity=0.0)
        bA = b.add_body(mass=1.0)
        bB = b.add_body(mass=1.0, xform=wp.transform(wp.vec3(0, 1, 0), wp.quat_identity()))
        bC = b.add_body(mass=1.0, xform=wp.transform(wp.vec3(0, 2, 0), wp.quat_identity()))
        b.add_joint_free(bA)
        j_ab = b.add_joint_fixed(bA, bB, parent_xform=wp.transform(wp.vec3(0, 1, 0), wp.quat_identity()))
        b.add_joint_fixed(bB, bC, parent_xform=wp.transform(wp.vec3(0, 1, 0), wp.quat_identity()))
        model = b.finalize(device="cpu")
        je = model.joint_enabled.numpy()
        je[j_ab] = False
        model.joint_enabled.assign(je)

        info = compute_fixed_joint_merge(model)
        self.assertIsNotNone(info)
        eff = info.joint_enabled_effective_gpu.numpy()
        # Disabled joint stays disabled in effective layer.
        self.assertFalse(bool(eff[j_ab]))

    def test_velocity_propagation_uses_com_offset(self):
        """Velocity propagation must use the COM-to-COM offset, not the body-origin offset."""
        device = "cpu"
        L = 2.0
        sv_mass = 1.0
        child_mass = 4.0
        b = ModelBuilder(gravity=0.0)
        b0 = b.add_body(mass=sv_mass)
        b1 = b.add_body(mass=child_mass, xform=wp.transform(wp.vec3(0, L, 0), wp.quat_identity()))
        b.add_joint_free(b0)
        b.add_joint_fixed(b0, b1, parent_xform=wp.transform(wp.vec3(0, L, 0), wp.quat_identity()))
        model = b.finalize(device=device)

        info = compute_fixed_joint_merge(model)
        self.assertIsNotNone(info)

        # Survivor at world identity, pure angular velocity about Z.
        omega_z = 0.5
        body_q = wp.array(
            [wp.transform_identity(), wp.transform_identity()],
            dtype=wp.transform,
            device=device,
        )
        body_qd = wp.array(
            [
                wp.spatial_vector(0.0, 0.0, 0.0, 0.0, 0.0, omega_z),
                wp.spatial_vector(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
            ],
            dtype=wp.spatial_vector,
            device=device,
        )

        wp.launch(
            kernel=_propagate_merged_body_velocities,
            dim=model.body_count,
            inputs=[
                info.survivor_indices_gpu,
                body_q,
                info.relative_xforms_gpu,
                model.body_com,
                info.merged_body_com_gpu,
                body_qd,
            ],
            device=device,
        )

        merged_com = info.merged_body_com_gpu.numpy()[b0]
        # Survivor identity transform → world COMs equal local COMs.
        com_s_world = np.array(merged_com)
        com_c_world = np.array([0.0, L, 0.0])  # child body_com is zero, body origin at +L Y
        r_world = com_c_world - com_s_world
        omega_vec = np.array([0.0, 0.0, omega_z])
        expected_v = np.cross(omega_vec, r_world)

        actual_qd = body_qd.numpy()[b1]
        for i in range(3):
            self.assertAlmostEqual(float(actual_qd[i]), float(expected_v[i]), places=5)
        # Angular velocity must equal survivor's exactly.
        for i in range(3):
            self.assertAlmostEqual(float(actual_qd[3 + i]), float(omega_vec[i]), places=6)

    def test_collapse_false_survives_notify(self):
        """A solver built with collapse_fixed_joints=False stays opted out across notifies."""
        model, _, _ = _make_two_body_fixed("cpu")
        solver = SolverXPBD(model, collapse_fixed_joints=False)
        self.assertIsNone(solver._merge_info)
        solver.notify_model_changed(SolverNotifyFlags.JOINT_PROPERTIES)
        # Must not silently re-enable merging.
        self.assertIsNone(solver._merge_info)
        self.assertIs(solver.joint_enabled_effective, model.joint_enabled)

    def test_notify_joint_properties_recomputes_merge(self):
        """notify_model_changed(JOINT_PROPERTIES) must refresh the cached merge."""
        b = ModelBuilder(gravity=0.0)
        b0 = b.add_body(mass=1.0)
        b1 = b.add_body(mass=1.0, xform=wp.transform(wp.vec3(0, 1, 0), wp.quat_identity()))
        b.add_joint_free(b0)
        j = b.add_joint_fixed(b0, b1, parent_xform=wp.transform(wp.vec3(0, 1, 0), wp.quat_identity()))
        model = b.finalize(device="cpu")

        solver = SolverXPBD(model)
        self.assertIsNotNone(solver._merge_info)

        # Disable the only FIXED joint and notify with JOINT_PROPERTIES.
        je = model.joint_enabled.numpy()
        je[j] = False
        model.joint_enabled.assign(je)
        solver.notify_model_changed(SolverNotifyFlags.JOINT_PROPERTIES)

        # Topology now has no collapsible FIXED joint → merge cleared.
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


def test_xpbd_external_force_on_merged_child_moves_survivor(test, device):
    """A force written to state.body_f[merged_child] must move the survivor.

    Without the body_f scatter, the force would be silently dropped because
    the merged child has zero effective inverse mass.
    """
    offset = wp.vec3(0.0, 1.0, 0.0)
    model, b0, b1 = _make_two_body_fixed(device, mass0=1.0, mass1=1.0, offset=offset)
    solver = SolverXPBD(model)
    test.assertIsNotNone(solver._merge_info)

    state0, state1 = model.state(), model.state()
    newton.eval_fk(model, model.joint_q, model.joint_qd, state0)

    # Apply a +X force on the merged child; survivor should accelerate in +X.
    bf = state0.body_f.numpy()
    bf[b1] = [10.0, 0.0, 0.0, 0.0, 0.0, 0.0]  # [linear, angular]
    state0.body_f.assign(bf)

    initial_pos_x = float(state0.body_q.numpy()[b0][0])

    # No contacts; just integrate.
    contacts = model.contacts()
    solver.step(state0, state1, None, contacts, 1.0 / 60.0)

    final_pos_x = float(state1.body_q.numpy()[b0][0])
    test.assertGreater(
        final_pos_x - initial_pos_x,
        0.0,
        f"survivor body did not move under force applied to merged child (Δx = {final_pos_x - initial_pos_x})",
    )


def test_xpbd_downstream_revolute_joint_works(test, device):
    """A revolute joint downstream of a merged FIXED chain must still rotate.

    Topology: A-(FREE)->world, A-(FIXED)->B, B-(REVOLUTE)->C.
    B merges into A; the revolute's effective parent points at A.  Verifies
    the simulation runs and produces finite state for several steps under
    gravity, exercising the joint with a remapped parent + remapped anchor.
    """
    b = ModelBuilder(gravity=-9.81)
    bA = b.add_body(mass=1.0)
    bB = b.add_body(mass=1.0, xform=wp.transform(wp.vec3(0, 1, 0), wp.quat_identity()))
    bC = b.add_body(mass=1.0, xform=wp.transform(wp.vec3(1, 1, 0), wp.quat_identity()))
    b.add_joint_free(bA)
    b.add_joint_fixed(bA, bB, parent_xform=wp.transform(wp.vec3(0, 1, 0), wp.quat_identity()))
    b.add_joint_revolute(
        bB,
        bC,
        axis=newton.Axis.Z,
        parent_xform=wp.transform(wp.vec3(1, 0, 0), wp.quat_identity()),
    )
    model = b.finalize(device=device)

    solver = SolverXPBD(model)
    test.assertIsNotNone(solver._merge_info)
    state0, state1 = model.state(), model.state()
    newton.eval_fk(model, model.joint_q, model.joint_qd, state0)
    contacts = model.contacts()

    for _ in range(10):
        model.collide(state0, contacts)
        solver.step(state0, state1, None, contacts, 1.0 / 60.0)
        state0, state1 = state1, state0

    test.assertTrue(np.all(np.isfinite(state0.body_q.numpy())))
    test.assertTrue(np.all(np.isfinite(state0.body_qd.numpy())))


def test_xpbd_com_shifted_cluster_rigid_attachment(test, device):
    """End-to-end rigid attachment for a cluster whose merged COM is far from sv body origin.

    Uses a 4 times heavier child so the combined COM shifts strongly off the
    survivor's body origin. Exercises (1) inertia recentered to combined COM,
    (2) velocity propagation via COM-to-COM offset, (3) ``_apply_body_deltas``
    using merged ``body_com``/``body_inertia``. Asserts the merged child stays
    rigidly bonded to the survivor through gravity and contact-free integration.
    """
    offset = wp.vec3(0.0, 1.0, 0.0)
    b = ModelBuilder(gravity=-9.81, up_axis=newton.Axis.Z)
    sv = b.add_body(mass=1.0)
    ch = b.add_body(mass=4.0, xform=wp.transform(offset, wp.quat_identity()))
    b.add_joint_free(sv)
    b.add_joint_fixed(sv, ch, parent_xform=wp.transform(offset, wp.quat_identity()))
    model = b.finalize(device=device)

    solver = SolverXPBD(model)
    mi = solver._merge_info
    test.assertIsNotNone(mi)

    state0, state1 = model.state(), model.state()
    newton.eval_fk(model, model.joint_q, model.joint_qd, state0)
    contacts = model.contacts()

    for _ in range(20):
        model.collide(state0, contacts)
        solver.step(state0, state1, None, contacts, 1.0 / 120.0)
        state0, state1 = state1, state0

    body_q_np = state0.body_q.numpy()
    body_qd_np = state0.body_qd.numpy()
    test.assertTrue(np.all(np.isfinite(body_q_np)))
    test.assertTrue(np.all(np.isfinite(body_qd_np)))

    # Rigid bond: child pose must equal survivor pose composed with relative xform.
    rel = mi.relative_xform_of[ch]
    expected = wp.transform(*body_q_np[sv]) * rel
    actual = wp.transform(*body_q_np[ch])
    for i in range(7):
        test.assertAlmostEqual(float(actual[i]), float(expected[i]), places=4)


add_function_test(
    TestSolverXPBDFixedJointCollapse,
    "test_merged_child_pose_follows_survivor",
    test_xpbd_merged_child_pose_follows_survivor,
    devices=get_test_devices(),
)
add_function_test(
    TestSolverXPBDFixedJointCollapse,
    "test_com_shifted_cluster_rigid_attachment",
    test_xpbd_com_shifted_cluster_rigid_attachment,
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
add_function_test(
    TestSolverXPBDFixedJointCollapse,
    "test_external_force_on_merged_child_moves_survivor",
    test_xpbd_external_force_on_merged_child_moves_survivor,
    devices=get_test_devices(),
)
add_function_test(
    TestSolverXPBDFixedJointCollapse,
    "test_downstream_revolute_joint_works",
    test_xpbd_downstream_revolute_joint_works,
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


def test_semi_external_force_on_merged_child_moves_survivor(test, device):
    """Same scatter contract as XPBD: force on merged child must drive the survivor."""
    offset = wp.vec3(0.0, 1.0, 0.0)
    model, b0, b1 = _make_two_body_fixed(device, mass0=1.0, mass1=1.0, offset=offset)
    solver = SolverSemiImplicit(model)
    test.assertIsNotNone(solver._merge_info)

    state0, state1 = model.state(), model.state()
    newton.eval_fk(model, model.joint_q, model.joint_qd, state0)

    bf = state0.body_f.numpy()
    bf[b1] = [10.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    state0.body_f.assign(bf)

    initial_pos_x = float(state0.body_q.numpy()[b0][0])
    solver.step(state0, state1, None, None, 1.0 / 60.0)
    final_pos_x = float(state1.body_q.numpy()[b0][0])
    test.assertGreater(
        final_pos_x - initial_pos_x,
        0.0,
        f"survivor body did not move under force applied to merged child (Δx = {final_pos_x - initial_pos_x})",
    )


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
add_function_test(
    TestSolverSemiImplicitFixedJointCollapse,
    "test_external_force_on_merged_child_moves_survivor",
    test_semi_external_force_on_merged_child_moves_survivor,
    devices=get_test_devices(),
)


if __name__ == "__main__":
    unittest.main(verbosity=2)
