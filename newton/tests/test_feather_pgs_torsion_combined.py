# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Qualify regularized positions and unregularized torsional velocity cleanup."""

import unittest

import numpy as np
import warp as wp

from newton.tests import test_feather_pgs_torsion_velocity as velocity_tests
from newton.tests.test_feather_pgs_contact_torsion import PATCH_OPTIONS, fixture


@unittest.skipUnless(wp.is_cuda_available(), "Requires CUDA")
class TestTorsionCombined(unittest.TestCase):
    def check_pair(self, regularization, velocity, **options):
        """Check split position integration and the accumulated velocity correction."""
        common = dict(
            pgs_iterations=64,
            pgs_contact_regularization=regularization,
            enable_bilateral_preelimination=False,
            **options,
        )
        baseline, *_ = fixture(0.01, pgs_velocity_iterations=0, **common)
        result, solver, model, state, contacts = fixture(0.01, pgs_velocity_iterations=velocity, **common)
        np.testing.assert_array_equal(result["body_q"], baseline["body_q"])
        count = int(result["count"][0])
        delta = result["impulses"][0, :count] - baseline["impulses"][0, :count]
        np.testing.assert_allclose(
            result["v_out"],
            baseline["v_out"].astype(np.float64) + result["Y_world"][0, :count].astype(np.float64).T @ delta,
            atol=2e-5,
            rtol=8 * np.finfo(np.float32).eps,
        )
        weights = solver.row_w.numpy()[0, :count]
        types = result["row_type"][0, :count]
        np.testing.assert_allclose(weights[types == 0], 1 / (1 + regularization), atol=1e-7)
        velocity_tests.TestTorsionVelocitySolve._assert_budget(self, result, solver)
        self.assertTrue(np.isfinite(result["body_qd"]).all())
        return baseline, result, solver, model, state, contacts

    def test_analytic_normal_load_position_soft_velocity_rigid(self):
        """Undo normal softness only in velocity cleanup and budget spin from final load."""
        for regularization in (0.01, 0.02, 0.5):
            for velocity in (1, 4, 16):
                with self.subTest(reg=regularization, velocity=velocity):
                    baseline, result, solver, *_ = self.check_pair(
                        regularization, velocity, center_only=True, spin=100.0
                    )
                    group = solver._torsion_stats["groups"][0]
                    normals = group["normal_rows"]
                    self.assertAlmostEqual(
                        float(baseline["impulses"][0, normals].sum()), 0.03 / (1 + regularization), delta=2e-7
                    )
                    normal = float(result["impulses"][0, normals].sum())
                    self.assertAlmostEqual(normal, 0.03, delta=2e-7)
                    self.assertAlmostEqual(
                        abs(float(result["impulses"][0, group["row"]])), 0.01 * 0.5 * normal, delta=2e-8
                    )
                    self.assertLess(abs(float(result["body_qd"][0, 5])), 100.0)

    def test_distributed_budget_and_export(self):
        """Keep point and patch budgets coherent across internal timesteps."""
        for patch in ({}, PATCH_OPTIONS):
            for dt in (0.00125, 0.0025, 0.005):
                with self.subTest(patch=bool(patch), dt=dt):
                    _, result, solver, _, _, contacts = self.check_pair(
                        0.01, 16, spin=10.0, sliding=1.0, dt=dt, **patch
                    )
                    solver.update_contacts(contacts)
                    count = int(contacts.rigid_contact_count.numpy()[0])
                    forces = contacts.rigid_contact_force.numpy()[:count]
                    normals = contacts.rigid_contact_normal.numpy()[:count]
                    for index, slot in enumerate(solver.contact_slot.numpy()[:count]):
                        if slot >= 0:
                            self.assertAlmostEqual(
                                abs(float(forces[index] @ normals[index])) * dt,
                                float(result["impulses"][0, slot]),
                                delta=2e-7,
                            )

    def test_no_load_near_positive_gap_and_release(self):
        """Refund sliding and spin even for the narrow positive initial-gap tolerance."""
        for gap in (0.5e-6, 0.01):
            _, _, solver, *_ = self.check_pair(0.01, 16, center_only=True, spin=100.0, **PATCH_OPTIONS)
            group = solver._torsion_stats["groups"][0]
            n = int(solver.constraint_count.numpy()[0])
            before = solver.impulses.numpy()[0, :n].copy()
            phi = solver.phi.numpy()
            phi[0, group["normal_rows"]] = gap
            solver.phi.assign(phi)
            # Force a separating end-gap through the normal row's response vector.
            row = group["normal_rows"][0]
            separating = solver.Y_world.numpy()[0, row].copy()
            solver.v_out_snap.assign(separating)
            solver.v_hat.assign(separating)
            solver.v_out.assign(separating + solver.Y_world.numpy()[0, :n].T @ before)
            velocity_before = solver.v_out.numpy().copy()
            solver._conclude_matrix_free_position_problem(0.0025)
            solver._run_matrix_free_velocity_post_solve()
            after = solver.impulses.numpy()[0, :n]
            self.assertLess(np.abs(after[group["normal_rows"]]).max(initial=0), 2e-7)
            self.assertLess(abs(float(after[group["row"]])), 2e-7)
            for anchor in group["anchor_rows"]:
                self.assertLess(np.linalg.norm(after[anchor + 1 : anchor + 3]), 2e-7)
            np.testing.assert_allclose(
                solver.v_out.numpy(), velocity_before + solver.Y_world.numpy()[0, :n].T @ (after - before), atol=2e-5
            )


if __name__ == "__main__":
    unittest.main()
