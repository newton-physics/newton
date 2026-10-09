# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Sweep variants of ``articulated_contact_response="propagation-colored"``.

The lane-per-unit sweep prefetches each unit's rows, solves standard units straight-line and,
for a few worlds, reads the rows from a packed per-step record, optionally staged in shared
memory. Every variant must match the reference sweep bitwise.
"""

import unittest

import numpy as np

import newton
from newton._src.solvers.feather_pgs.kernels import PROPAGATION_UNIT_META, PROPAGATION_UNIT_RING
from newton._src.solvers.feather_pgs.solver_feather_pgs import _colored_staging_supported
from newton.solvers import SolverFeatherPGS
from newton.tests.test_feather_pgs_colored_scheduling import CONTACT_MAX, _heap
from newton.tests.unittest_utils import add_function_test, get_cuda_test_devices, get_test_devices

DT = 1.0 / 240.0


def _solver(model, regularization, **overrides):
    """Build the colored solver with the given sweep-variant overrides."""
    SolverFeatherPGS._kernel_overrides = overrides
    try:
        return SolverFeatherPGS(
            model,
            articulated_contact_response="propagation-colored",
            pgs_iterations=8,
            mf_max_constraints=CONTACT_MAX,
            dense_max_constraints=64,
            pgs_contact_regularization=regularization,
        )
    finally:
        SolverFeatherPGS._kernel_overrides = {}


def _settled_heap(device, regularization):
    """A heap settled into resting contact with patch history, and its contacts."""
    model = _heap(device)
    pipeline = newton.CollisionPipeline(model, rigid_contact_max=CONTACT_MAX, deterministic=True)
    contacts = pipeline.contacts()
    control = model.control()
    state_0, state_1 = model.state(), model.state()
    warm = _solver(model, regularization)
    for _ in range(8):
        pipeline.collide(state_0, contacts)
        warm.step(state_0, state_1, control, contacts, DT)
        state_0, state_1 = state_1, state_0
    pipeline.collide(state_0, contacts)
    start = {name: getattr(state_0, name).numpy().copy() for name in ("body_q", "body_qd", "joint_q", "joint_qd")}
    return model, contacts, control, start


def _run(solver, model, contacts, control, start, steps):
    state_0, state_1 = model.state(), model.state()
    for name, value in start.items():
        getattr(state_0, name).assign(value)
    for _ in range(steps):
        solver.step(state_0, state_1, control, contacts, DT)
        state_0, state_1 = state_1, state_0
    return state_0.body_q.numpy(), state_0.body_qd.numpy(), solver.propagation_impulses.numpy()


def test_staging_requires_cp_async_and_few_worlds(test, device):
    """Stage only with cp.async (compute capability 8.0), a few worlds and fitting shared memory."""
    test.assertFalse(_colored_staging_supported(75, 1, 60))
    test.assertTrue(_colored_staging_supported(80, 1, 60))
    test.assertTrue(_colored_staging_supported(120, 16, 60))
    test.assertFalse(_colored_staging_supported(120, 17, 60))
    test.assertFalse(_colored_staging_supported(120, 1, 400))


def test_sweep_variants_match_the_reference_bitwise(test, device):
    """Reproduce the reference sweep with the prefetched, packed and staged sweeps.

    Without regularization the row-weight buffer is a placeholder the sweeps must not read.
    """
    for regularization in (0.0, 0.01):
        model, contacts, control, start = _settled_heap(device, regularization)
        test.assertGreater(int(contacts.rigid_contact_count.numpy()[0]), 400)
        staged = _colored_staging_supported(model.device.arch, 1, _solver(model, regularization).max_propagation_bodies)
        variants = {
            "reference": ({"colored_prefetch": False}, "_pf0_st0"),
            "prefetch": ({"colored_packed": False}, "_pf1_st0_pk0"),
            "packed": ({}, "_pf1_st0_pk1"),
            "staged": ({"colored_staged": True}, "_pf1_st1" if staged else "_pf1_st0_pk1"),
        }
        results = {}
        for name, (overrides, tag) in variants.items():
            solver = _solver(model, regularization, **overrides)
            test.assertIn(tag, solver._pgs_solve_propagation_colored_warp_kernel.key)
            results[name] = _run(solver, model, contacts, control, start, 5)
        reference = results["reference"]
        test.assertTrue(all(np.isfinite(x).all() for x in reference))
        test.assertGreater(np.abs(reference[2]).max(), 0.0)
        for name in ("prefetch", "packed", "staged"):
            for label, got, want in zip(("body_q", "body_qd", "impulses"), results[name], reference, strict=True):
                with test.subTest(regularization=regularization, variant=name, field=label):
                    np.testing.assert_array_equal(got, want)


def test_straight_line_units_match_the_row_loop(test, device):
    """Agree with the row loop to rounding when standard units are solved straight-line.

    One step is compared: friction-patch decisions taken from the impulses may relayout later rows.
    """
    for regularization in (0.0, 0.01):
        model, contacts, control, start = _settled_heap(device, regularization)
        results = {}
        for name, overrides in (("row loop", {"colored_straight_line": False}), ("straight line", {})):
            solver = _solver(model, regularization, **overrides)
            results[name] = _run(solver, model, contacts, control, start, 1)
            if name == "straight line":
                n = int(solver.color_world_offsets.numpy()[-1])
                records = solver.color_unit_meta.numpy().reshape(-1, PROPAGATION_UNIT_META)[:n]
                test.assertGreater(int(np.sum(records[:, 6 + PROPAGATION_UNIT_RING] == 1)), n // 2)
        for label, got, want in zip(
            ("body_q", "body_qd", "impulses"), results["straight line"], results["row loop"], strict=True
        ):
            with test.subTest(regularization=regularization, field=label):
                np.testing.assert_allclose(got, want, rtol=1.0e-4, atol=1.0e-6)


class TestFeatherPGSColoredPipeline(unittest.TestCase):
    pass


add_function_test(
    TestFeatherPGSColoredPipeline,
    "test_staging_requires_cp_async_and_few_worlds",
    test_staging_requires_cp_async_and_few_worlds,
    devices=get_test_devices(),
)
for _fn in (test_sweep_variants_match_the_reference_bitwise, test_straight_line_units_match_the_row_loop):
    add_function_test(TestFeatherPGSColoredPipeline, _fn.__name__, _fn, devices=get_cuda_test_devices())


if __name__ == "__main__":
    unittest.main()
