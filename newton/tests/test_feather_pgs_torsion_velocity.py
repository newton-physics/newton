# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Qualify load-bounded spin friction during the unbiased velocity post-pass."""

import unittest

import numpy as np
import warp as wp

from newton._src.solvers.feather_pgs import contact_torsion
from newton.tests.test_feather_pgs_contact_torsion import PATCH_OPTIONS, fixture


class TestTorsionVelocityAdmission(unittest.TestCase):
    def test_end_gap_retires_spin_but_preserves_target_and_carried_impulse(self):
        """Retire separated spin groups without deleting their required velocity correction."""
        for initial_gap, end_gap in ((-0.001, -0.001), (-0.001, 0.0001), (0.0001, -0.0001), (0.0001, 0.0002)):
            with self.subTest(initial_gap=initial_gap, end_gap=end_gap):
                count = wp.array([2], dtype=int, device="cpu")
                types = wp.array([[0, 7]], dtype=int, device="cpu")
                groups = wp.array([[1, -1]], dtype=int, device="cpu")
                phi = wp.array([[initial_gap, 0.0]], dtype=float, device="cpu")
                target = wp.array([[0.0, 0.25]], dtype=float, device="cpu")
                velocity = wp.array([(end_gap - initial_gap) / 0.01], dtype=float, device="cpu")
                dofs = wp.array([1], dtype=int, device="cpu")
                indices = wp.array([[0]], dtype=int, device="cpu")
                jacobian = wp.array(np.array([[[1.0], [0.0]]], np.float32), device="cpu")
                rhs = wp.array([[3.0, 9.0]], dtype=float, device="cpu")
                wp.launch(
                    contact_torsion._prepare_velocity_torsion_rows,
                    dim=2,
                    inputs=[count, 2, types, groups, phi, target, velocity, dofs, indices, jacobian, 0.01],
                    outputs=[rhs],
                    device="cpu",
                )
                retired = initial_gap > 1e-6 and end_gap > 1e-6
                self.assertEqual(int(groups.numpy()[0, 1]), contact_torsion._TORSION_SPIN_RETIRED if retired else -1)
                np.testing.assert_array_equal(rhs.numpy(), [[3.0, -0.25]])
                # Re-admission must clear a previous retirement, including when
                # these same buffers are reused by a captured graph replay.
                phi.assign(np.array([[-0.001, 0.0]], dtype=np.float32))
                wp.launch(
                    contact_torsion._prepare_velocity_torsion_rows,
                    dim=2,
                    inputs=[count, 2, types, groups, phi, target, velocity, dofs, indices, jacobian, 0.01],
                    outputs=[rhs],
                    device="cpu",
                )
                np.testing.assert_array_equal(groups.numpy(), [[1, -1]])


@unittest.skipUnless(wp.is_cuda_available(), "Requires CUDA")
class TestTorsionVelocitySolve(unittest.TestCase):
    def test_rebound_preserves_loaded_sliding_when_spin_retires(self):
        """Preserve ordinary sliding friction on an admitted rebounding contact."""
        for separation in (0.0, 5.0e-6):
            for restitution in (0.0, 0.5):
                with self.subTest(separation=separation, restitution=restitution):
                    options = dict(
                        separation=separation,
                        restitution=restitution,
                        closing=1.0,
                        sliding=1.0,
                        spin=0.0,
                        sat=True,
                        center_only=True,
                        pgs_iterations=64,
                        pgs_velocity_iterations=16,
                        enable_bilateral_preelimination=False,
                        **PATCH_OPTIONS,
                    )
                    baseline, *_ = fixture(0.0, **options)
                    actual, solver, *_ = fixture(0.01, **options)
                    normals = actual["row_type"] == 0
                    tangent = actual["row_type"] == 2
                    np.testing.assert_allclose(actual["impulses"][normals], baseline["impulses"][normals], atol=2e-6)
                    self.assertGreater(float(np.linalg.norm(baseline["impulses"][tangent])), 0.01)
                    np.testing.assert_allclose(actual["impulses"][tangent], baseline["impulses"][tangent], atol=2e-6)
                    np.testing.assert_allclose(actual["body_qd"], baseline["body_qd"], atol=3e-5)
                    self._assert_budget(actual, solver)

    def _assert_budget(self, result, solver):
        """Check the shared final-load budget and nonnegative normal impulses."""
        for group in solver._torsion_stats["groups"]:
            impulse = result["impulses"][group["world"]]
            normal = sum(max(float(impulse[n]), 0.0) for n in group["normal_rows"])
            sliding = sum(float(np.linalg.norm(impulse[n + 1 : n + 3])) for n in group["anchor_rows"])
            spin = abs(float(impulse[group["row"]])) / group["effective_radius_m"]
            self.assertLessEqual(sliding + spin, group["mu"] * normal + 3e-6)

    def test_spin_opposition_and_final_budget(self):
        """Dissipate angular slip and respect final normal load in each velocity count."""
        for velocity in (1, 4, 16):
            for spin, sliding in ((1.0, 0.0), (100.0, 0.0), (10.0, 1.0)):
                with self.subTest(velocity=velocity, spin=spin, sliding=sliding):
                    result, solver, *_ = fixture(
                        0.01,
                        pgs_iterations=16,
                        pgs_velocity_iterations=velocity,
                        spin=spin,
                        sliding=sliding,
                        center_only=sliding == 0,
                        enable_bilateral_preelimination=False,
                        **PATCH_OPTIONS,
                    )
                    self._assert_budget(result, solver)
                    self.assertTrue(np.isfinite(result["body_qd"]).all())
                    if sliding == 0:
                        self.assertLess(np.max(np.abs(result["body_qd"][:, 5])), spin)

    def test_postpass_accumulated_impulse_response_and_position_split(self):
        """Apply delta impulses once while retaining the biased position trajectory."""
        original = fixture(
            0.01,
            spin=100.0,
            closing=0.1,
            center_only=True,
            pgs_iterations=16,
            enable_bilateral_preelimination=False,
            **PATCH_OPTIONS,
        )
        result, solver, *_ = fixture(
            0.01,
            spin=100.0,
            closing=0.1,
            center_only=True,
            pgs_iterations=16,
            pgs_velocity_iterations=4,
            enable_bilateral_preelimination=False,
            **PATCH_OPTIONS,
        )
        baseline = original[0]
        np.testing.assert_array_equal(result["body_q"], baseline["body_q"])
        count = int(result["count"][0])
        delta = result["impulses"][0, :count] - baseline["impulses"][0, :count]
        np.testing.assert_allclose(
            result["v_out"] - baseline["v_out"], result["Y_world"][0, :count].T @ delta, atol=2e-5
        )
        self._assert_budget(result, solver)

    def test_zero_load_and_speculation(self):
        """Remove torque when normal load or current touching support disappears."""
        for options in ({"closing": 0.0}, {"closing": -0.1}, {"separation": 0.004}, {"mu": 0.0}):
            with self.subTest(options=options):
                result, solver, *_ = fixture(
                    0.01, pgs_velocity_iterations=4, enable_bilateral_preelimination=False, **PATCH_OPTIONS, **options
                )
                active = result["row_type"] == 7
                self.assertLess(np.abs(result["impulses"][active]).max(initial=0), 2e-7)
                self._assert_budget(result, solver)

    def test_carried_spin_is_refunded_when_postpass_retires_group(self):
        """Undo carried spin through its response column when final support is separated."""
        original, solver, model, state, contacts = fixture(
            0.01, spin=100.0, center_only=True, pgs_velocity_iterations=4, **PATCH_OPTIONS
        )
        group = solver._torsion_stats["groups"][0]
        count = int(solver.constraint_count.numpy()[0])
        before_lambda = solver.impulses.numpy()[0, :count].copy()
        self.assertGreater(abs(float(before_lambda[group["row"]])), 1e-8)
        before_velocity = solver.v_out.numpy().copy()
        phi = solver.phi.numpy()
        phi[0, group["normal_rows"]] = 0.01
        solver.phi.assign(phi)
        solver.v_out_snap.zero_()
        solver._conclude_matrix_free_position_problem(0.0025)
        self.assertEqual(float(solver.row_mu.numpy()[0, group["row"]]), group["mu"])
        self.assertEqual(
            int(solver._contact_torsion_group.numpy()[0, group["row"]]), contact_torsion._TORSION_SPIN_RETIRED
        )
        # Eligibility changes the admissible set, not the already-applied load.
        np.testing.assert_array_equal(solver.impulses.numpy()[0, :count], before_lambda)
        solver._run_matrix_free_velocity_post_solve()
        after_lambda = solver.impulses.numpy()[0, :count]
        self.assertLess(abs(float(after_lambda[group["row"]])), 1e-9)
        np.testing.assert_allclose(
            solver.v_out.numpy() - before_velocity,
            solver.Y_world.numpy()[0, :count].T @ (after_lambda - before_lambda),
            atol=2e-5,
        )
        # A new position solve must restore admission rather than inherit the
        # prior velocity-pass retirement in a reused membership buffer.
        output = model.state()
        solver.step(state, output, model.control(), contacts, 0.0025)
        group = solver._torsion_stats["groups"][0]
        self.assertEqual(int(solver._contact_torsion_group.numpy()[0, group["row"]]), -1)
        self.assertGreater(abs(float(solver.impulses.numpy()[0, group["row"]])), 1e-8)
        np.testing.assert_allclose(output.body_qd.numpy(), original["body_qd"], atol=2e-5)

    def test_internal_timestep_export_uses_final_linear_impulse(self):
        """Export final normal impulses in newtons at each internal substep duration."""
        for dt in (0.00125, 0.0025, 0.005):
            with self.subTest(dt=dt):
                result, solver, model, initial, contacts = fixture(
                    0.01, spin=100.0, center_only=True, dt=dt, pgs_velocity_iterations=4, **PATCH_OPTIONS
                )
                # The base fixture does not request the optional spatial export.
                contacts.force = wp.zeros(contacts.rigid_contact_max, dtype=wp.spatial_vector, device=model.device)
                for _ in range(4):
                    output = model.state()
                    solver.step(initial, output, model.control(), contacts, dt)
                    solver.update_contacts(contacts)
                    row = int(solver.contact_slot.numpy()[0])
                    normal = contacts.rigid_contact_normal.numpy()[0]
                    force = contacts.force.numpy()[0]
                    expected = max(float(solver.impulses.numpy()[0, row]), 0.0) / dt
                    self.assertAlmostEqual(abs(float(np.dot(force[:3], normal))), expected, delta=1e-4)
                    # Torque slots are deliberately placeholders, not solved spin wrench.
                    np.testing.assert_array_equal(force[3:], 0.0)
                    np.testing.assert_allclose(output.body_qd.numpy(), result["body_qd"], atol=2e-5)

    def test_selected_out_torsion_matches_zero_radius(self):
        """Keep the torsion-zero velocity path identical when no shapes are selected."""
        for velocity in (0, 4):
            expected, *_ = fixture(0.0, pgs_velocity_iterations=velocity, **PATCH_OPTIONS)
            actual, *_ = fixture(
                0.01, pgs_velocity_iterations=velocity, contact_torsion_shape_indices=(), **PATCH_OPTIONS
            )
            for key in expected:
                np.testing.assert_array_equal(actual[key], expected[key], err_msg=key)

    def test_runtime_incompatible_mutation_rejected(self):
        """Reject unsupported mutation before stepping an allocated torsion solver."""
        _, solver, model, state, contacts = fixture(0.01, pgs_velocity_iterations=4)
        solver.pgs_warmstart = True
        with self.assertRaisesRegex(ValueError, "warmstart"):
            solver.step(state, model.state(), model.control(), contacts, 0.0025)


if __name__ == "__main__":
    unittest.main()
