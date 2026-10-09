# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Validate spin sharing against accumulated regularized normal impulses."""

import unittest

import numpy as np

from newton._src.solvers.feather_pgs.kernels import PGS_CONSTRAINT_TYPE_TORSION
from newton.tests.test_feather_pgs_contact_torsion import PATCH_OPTIONS, fixture
from newton.tests.unittest_utils import add_function_test, get_cuda_test_devices


def _check_budget(test, result, solver):
    """Bound final shared impulses and account for their entire velocity response."""
    count = int(result["count"][0])
    impulses = result["impulses"][0, :count]
    # Compare full velocities in float64 rather than subtracting two large
    # float32 spins. Repeated in-place updates at 100rad/s accumulate several
    # velocity ULPs even though the impulse budget and linear response agree.
    response = result["Y_world"][0, :count].astype(np.float64).T @ impulses.astype(np.float64)
    np.testing.assert_allclose(
        result["v_out"],
        result["v_hat"].astype(np.float64) + response,
        rtol=8 * np.finfo(np.float32).eps,
        atol=2e-5,
    )
    test.assertTrue(solver._torsion_stats["groups"])
    for group in solver._torsion_stats["groups"]:
        load = sum(max(float(impulses[n]), 0.0) for n in group["normal_rows"])
        sliding = sum(float(np.linalg.norm(impulses[n + 1 : n + 3])) for n in group["anchor_rows"])
        spin = abs(float(impulses[group["row"]])) / group["effective_radius_m"]
        test.assertLessEqual(sliding + spin, group["mu"] * load + 2e-6)
    test.assertEqual(bool(np.any(solver.constraint_overflow.numpy())), False)


def test_analytic_regularized_load_and_saturated_spin(test, device):
    """Use the solved normal impulse once, without double-regularizing spin capacity."""
    for regularization in (0.01, 0.02, 0.5):
        with test.subTest(regularization=regularization):
            result, solver, *_ = fixture(
                0.01,
                device=device,
                center_only=True,
                spin=100.0,
                pgs_iterations=16,
                pgs_contact_regularization=regularization,
            )
            group = solver._torsion_stats["groups"][0]
            impulse = result["impulses"][0]
            # Two equal 0.3kg bodies close at 0.1m/s each: rigid lambda=.03Ns.
            normal = sum(float(impulse[n]) for n in group["normal_rows"])
            test.assertAlmostEqual(normal, 0.03 / (1 + regularization), delta=2e-7)
            test.assertAlmostEqual(abs(float(impulse[group["row"]])), 0.01 * 0.5 * normal, delta=2e-8)
            before = result["v_hat"].reshape(2, 6)
            after = result["v_out"].reshape(2, 6)
            test.assertLess(abs(after[0, 5] - after[1, 5]), abs(before[0, 5] - before[1, 5]))
            inertia = np.array([0.3, 0.3, 0.3, 0.00012, 0.00012, 0.00008])
            test.assertLessEqual(float(np.sum(after**2 * inertia)), float(np.sum(before**2 * inertia)) + 2e-6)
            _check_budget(test, result, solver)


def test_final_load_budget_across_iterations(test, device):
    """Correct stale sliding and spin when distributed normal loads shrink."""
    for regularization in (0.01, 0.02, 0.5):
        for iterations in (1, 2, 4, 16, 64):
            for spin, sliding in ((0.0, 1.0), (10.0, 1.0), (100.0, 0.0)):
                with test.subTest(reg=regularization, iterations=iterations, spin=spin):
                    result, solver, *_ = fixture(
                        0.0075,
                        device=device,
                        pgs_iterations=iterations,
                        pgs_contact_regularization=regularization,
                        spin=spin,
                        sliding=sliding,
                        **PATCH_OPTIONS,
                    )
                    _check_budget(test, result, solver)
                    count = int(result["count"][0])
                    types = result["row_type"][0, :count]
                    weights = solver.row_w.numpy()[0, :count]
                    normal_rows = types == 0
                    test.assertTrue(np.any(normal_rows))
                    np.testing.assert_allclose(weights[normal_rows], 1 / (1 + regularization), atol=1e-7)
                    # Torsion itself is a dry-friction row, not another soft normal.
                    np.testing.assert_array_equal(weights[types == PGS_CONSTRAINT_TYPE_TORSION], 1.0)


def test_zero_controls_and_no_load(test, device):
    """Preserve excluded/radius-zero controls and forbid spin without compressive load."""
    for regularization in (0.0, 0.01, 0.02):
        base, *_ = fixture(0.0, device=device, pgs_contact_regularization=regularization)
        excluded, *_ = fixture(
            0.01, device=device, pgs_contact_regularization=regularization, contact_torsion_shape_indices=()
        )
        for name in base:
            np.testing.assert_array_equal(base[name], excluded[name], err_msg=name)
        for options in ({"closing": 0.0}, {"mu": 0.0}):
            result, *_ = fixture(
                0.01, device=device, pgs_contact_regularization=regularization, center_only=True, **options
            )
            test.assertLess(
                np.abs(result["impulses"][result["row_type"] == PGS_CONSTRAINT_TYPE_TORSION]).max(initial=0), 1e-9
            )


def test_substep_force_export_and_release(test, device):
    """Export the actual solved normal load over dt and forget torque on release."""
    for substeps in (1, 2, 4):
        dt = 0.005 / substeps
        result, solver, model, initial, contacts = fixture(
            0.01,
            device=device,
            center_only=True,
            spin=100.0,
            dt=dt,
            pgs_contact_regularization=0.02,
            **PATCH_OPTIONS,
        )
        solver.update_contacts(contacts)
        count = int(contacts.rigid_contact_count.numpy()[0])
        forces = contacts.rigid_contact_force.numpy()[:count]
        normals = contacts.rigid_contact_normal.numpy()[:count]
        slots = solver.contact_slot.numpy()[:count]
        for contact, slot in enumerate(slots):
            if slot >= 0:
                test.assertAlmostEqual(
                    abs(float(np.dot(forces[contact], normals[contact]))) * dt,
                    float(result["impulses"][0, slot]),
                    delta=2e-7,
                )
        contacts.rigid_contact_count.zero_()
        solver.step(initial, model.state(), model.control(), contacts, dt)
        test.assertEqual(solver._torsion_stats["rows"], 0)


class TestTorsionRegularization(unittest.TestCase):
    pass


for _fn in (
    test_analytic_regularized_load_and_saturated_spin,
    test_final_load_budget_across_iterations,
    test_zero_controls_and_no_load,
    test_substep_force_export_and_release,
):
    add_function_test(TestTorsionRegularization, _fn.__name__, _fn, devices=get_cuda_test_devices())


if __name__ == "__main__":
    unittest.main(verbosity=2)
