# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Regress final normal-load reductions after persistent anchor updates."""

import unittest

import numpy as np
import warp as wp

from newton.tests.test_feather_pgs_contact_torsion import fixture


@unittest.skipUnless(wp.is_cuda_available(), "Requires CUDA")
class TestFinalPatchBudget(unittest.TestCase):
    def test_final_budget_and_velocity_response(self):
        """Bound final traction plus spin and apply every correction to velocity."""
        for kernel in ("loop", "tiled_row"):
            for iterations in (1, 2, 4, 8, 64):
                for spin, sliding in ((0.0, 1.0), (10.0, 1.0), (100.0, 0.0)):
                    with self.subTest(kernel=kernel, iterations=iterations, spin=spin, sliding=sliding):
                        result, solver, *_ = fixture(
                            0.0075,
                            friction_anchor_beta=0.2,
                            pgs_kernel=kernel,
                            pgs_iterations=iterations,
                            closing=0.1,
                            spin=spin,
                            sliding=sliding,
                            contact_shared_anchor=False,
                            contact_friction_shared_anchor=False,
                        )
                        count = int(result["count"][0])
                        response = result["Y_world"][0, :count].T @ result["impulses"][0, :count]
                        np.testing.assert_allclose(result["v_out"] - result["v_hat"], response, atol=2e-5)
                        groups = solver._torsion_stats["groups"]
                        self.assertTrue(groups, "The regression must exercise coupled patch/torsion rows")
                        for group in groups:
                            impulse = result["impulses"][group["world"]]
                            normal = sum(max(float(impulse[n]), 0.0) for n in group["normal_rows"])
                            traction = sum(float(np.linalg.norm(impulse[n + 1 : n + 3])) for n in group["anchor_rows"])
                            spin_used = abs(float(impulse[group["row"]])) / group["effective_radius_m"]
                            self.assertLessEqual(traction + spin_used, group["mu"] * normal + 2e-6)


if __name__ == "__main__":
    unittest.main()
