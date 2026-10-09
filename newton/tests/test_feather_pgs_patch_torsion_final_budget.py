# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Regress final normal-load reductions after persistent anchor updates."""

import unittest

import numpy as np

from newton.tests.test_feather_pgs_contact_torsion import fixture
from newton.tests.unittest_utils import add_function_test, get_cuda_test_devices


def test_final_budget_and_velocity_response(test, device):
    """Bound final traction plus spin and apply every correction to velocity."""
    for iterations in (1, 2, 4, 8, 64):
        for spin, sliding in ((0.0, 1.0), (10.0, 1.0), (100.0, 0.0)):
            with test.subTest(iterations=iterations, spin=spin, sliding=sliding):
                result, solver, *_ = fixture(
                    0.0075,
                    device=device,
                    friction_anchor_beta=0.2,
                    pgs_iterations=iterations,
                    closing=0.1,
                    spin=spin,
                    sliding=sliding,
                )
                count = int(result["count"][0])
                response = result["Y_world"][0, :count].T @ result["impulses"][0, :count]
                np.testing.assert_allclose(result["v_out"] - result["v_hat"], response, atol=2e-5)
                groups = solver._torsion_stats["groups"]
                test.assertTrue(groups, "The regression must exercise coupled patch/torsion rows")
                for group in groups:
                    impulse = result["impulses"][group["world"]]
                    normal = sum(max(float(impulse[n]), 0.0) for n in group["normal_rows"])
                    traction = sum(float(np.linalg.norm(impulse[n + 1 : n + 3])) for n in group["anchor_rows"])
                    spin_used = abs(float(impulse[group["row"]])) / group["effective_radius_m"]
                    test.assertLessEqual(traction + spin_used, group["mu"] * normal + 2e-6)


class TestFinalPatchBudget(unittest.TestCase):
    pass


add_function_test(
    TestFinalPatchBudget,
    "test_final_budget_and_velocity_response",
    test_final_budget_and_velocity_response,
    devices=get_cuda_test_devices(),
)


if __name__ == "__main__":
    unittest.main(verbosity=2)
