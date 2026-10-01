# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Tests for the pipelined colored solve behind ``articulated_contact_response='propagation-colored'``.

The lane-per-unit colored kernel prefetches each unit's rows, keeps the unit's body
twists in registers, solves standard units on a straight-line path, and, for a few
worlds, reads each unit's rows from a packed per-step record (optionally staged in shared
memory). None of that may change the result: every variant must match the reference unit
body bitwise on the same state and contacts.
"""

import os
import unittest
from unittest import mock

import numpy as np
import warp as wp

import newton
from newton._src.solvers.feather_pgs.kernels import PROPAGATION_UNIT_META, PROPAGATION_UNIT_RING
from newton._src.solvers.feather_pgs.solver_feather_pgs import _colored_staging_supported

from .test_feather_pgs_colored_scheduling import _heap

_VARIANT_ENV = (
    "FEATHER_PGS_COLORED_PREFETCH",
    "FEATHER_PGS_COLORED_STAGED",
    "FEATHER_PGS_COLORED_PACKED",
    "FEATHER_PGS_COLORED_STRAIGHT_LINE",
)


def _solver(model, env, regularization=0.0):
    """Build the colored solver with only ``env`` set among the kernel-variant switches."""
    with mock.patch.dict(os.environ, env):
        for key in _VARIANT_ENV:
            if key not in env:
                os.environ.pop(key, None)
        return newton.solvers.SolverFeatherPGS(
            model,
            pgs_mode="matrix_free",
            articulated_contact_response="propagation-colored",
            pgs_iterations=8,
            mf_max_constraints=8192,
            dense_max_constraints=64,
            pgs_contact_regularization=regularization,
        )


class TestFeatherPGSColoredStagingSelection(unittest.TestCase):
    def test_staging_requires_cp_async_and_few_worlds(self):
        """Staging emits cp.async (compute capability 8.0+) and is for a few worlds only."""
        self.assertFalse(_colored_staging_supported(75, 1, 1, 60), "SM75 cannot compile cp.async")
        self.assertTrue(_colored_staging_supported(80, 1, 1, 60))
        self.assertTrue(_colored_staging_supported(120, 1, 16, 60))
        self.assertFalse(_colored_staging_supported(120, 1, 17, 60), "many worlds keep the packed kernel")
        self.assertFalse(_colored_staging_supported(120, 2, 1, 60), "staging needs the lane-per-unit body")
        self.assertFalse(_colored_staging_supported(120, 1, 1, 400), "staging must fit static shared memory")


@unittest.skipUnless(wp.get_device().is_cuda, "propagation-colored requires CUDA")
class TestFeatherPGSColoredPipeline(unittest.TestCase):
    def test_pipelined_variants_match_the_reference_bitwise(self):
        """Prefetched and (where supported) staged sweeps match the reference unit body."""
        for regularization in (0.0, 0.01):
            with self.subTest(regularization=regularization):
                self._check_variants(regularization)

    def test_straight_line_units_match_the_row_loop(self):
        """Standard units solved straight-line agree with the generic row loop.

        Every variant shares the straight-line code, so the bitwise check above cannot see
        a mistake in it; this compares it with the row loop it replaces. The two differ
        only in summation order (pairwise dot products), so one step (eight sweeps) agrees
        to rounding, impulses included. Later steps are not compared: friction-patch
        decisions taken from those impulses may relayout the rows.
        """
        for regularization in (0.0, 0.01):
            with self.subTest(regularization=regularization):
                model, contacts, control, start = self._settled_heap(regularization)
                results = {}
                for name, env in (("row loop", {"FEATHER_PGS_COLORED_STRAIGHT_LINE": "0"}), ("straight line", {})):
                    solver = _solver(model, env, regularization)
                    results[name] = self._run(solver, model, contacts, control, start, 1)
                    if name == "straight line":
                        n = int(solver.color_world_offsets.numpy()[-1])
                        kinds = solver.color_unit_meta.numpy().reshape(-1, PROPAGATION_UNIT_META)[:n]
                        kinds = kinds[:, 6 + PROPAGATION_UNIT_RING]
                        self.assertGreater(int(np.sum(kinds == 1)), n // 2, "few units took the straight-line path")
                for label, got, want in zip(
                    ("body_q", "body_qd", "impulses"), results["straight line"], results["row loop"], strict=True
                ):
                    np.testing.assert_allclose(got, want, rtol=1e-4, atol=1e-6, err_msg=label)

    def _settled_heap(self, regularization):
        model, _ = _heap(nx=6, ny=6, nz=3)
        model.rigid_contact_max = 8192
        pipeline = newton.CollisionPipeline(model, rigid_contact_max=8192)
        contacts = pipeline.contacts()
        control = model.control()
        state_0, state_1 = model.state(), model.state()
        warm = _solver(model, {}, regularization)
        for _ in range(8):  # settle into resting contact with patch history
            pipeline.collide(state_0, contacts)
            warm.step(state_0, state_1, control, contacts, 1.0 / 240.0)
            state_0, state_1 = state_1, state_0
        pipeline.collide(state_0, contacts)
        start = {name: getattr(state_0, name).numpy().copy() for name in ("body_q", "body_qd", "joint_q", "joint_qd")}
        return model, contacts, control, start

    @staticmethod
    def _run(solver, model, contacts, control, start, steps):
        a, b = model.state(), model.state()
        for field, value in start.items():
            getattr(a, field).assign(value)
        solver.reset(a, flags=0)
        for _ in range(steps):
            solver.step(a, b, control, contacts, 1.0 / 240.0)
            a, b = b, a
        return a.body_q.numpy(), a.body_qd.numpy(), solver.propagation_impulses.numpy()

    def _check_variants(self, regularization):
        """Prefetched and shared-memory-staged sweeps reproduce the reference unit body.

        A settling heap of boxes with friction patches (the default) gives colors with
        many units, contact rows with friction pairs, and patch rings. Contacts are
        generated once so every solver sees identical input; the variants then run the
        same steps and must agree bit for bit on body state and row impulses. Without
        regularization the row-weight buffer is a one-element placeholder the kernels
        must not read; with it the weights are real rows.
        """
        model, _ = _heap(nx=6, ny=6, nz=3)
        model.rigid_contact_max = 8192
        pipeline = newton.CollisionPipeline(model, rigid_contact_max=8192)
        contacts = pipeline.contacts()
        control = model.control()
        state_0, state_1 = model.state(), model.state()
        warm = _solver(model, {}, regularization)
        for _ in range(8):  # settle into resting contact with patch history
            pipeline.collide(state_0, contacts)
            warm.step(state_0, state_1, control, contacts, 1.0 / 240.0)
            state_0, state_1 = state_1, state_0
        pipeline.collide(state_0, contacts)
        self.assertGreater(int(contacts.rigid_contact_count.numpy()[0]), 400)
        start = {name: getattr(state_0, name).numpy().copy() for name in ("body_q", "body_qd", "joint_q", "joint_qd")}

        variants = {
            "reference": ({"FEATHER_PGS_COLORED_PREFETCH": "0"}, "_pf0_st0"),
            "prefetch": ({"FEATHER_PGS_COLORED_PACKED": "0"}, "_pf1_st0_pk0"),
            "default": ({}, "_pf1_st0_pk1"),
            "staged": ({"FEATHER_PGS_COLORED_STAGED": "1"}, None),
        }
        results = {}
        for name, (env, expected) in variants.items():
            solver = _solver(model, env, regularization)
            kernel = solver._pgs_solve_propagation_colored_warp_kernel
            self.assertIsNotNone(kernel)
            tag = expected
            if tag is None:
                # Staging needs cp.async (SM80+); older devices take the unstaged packed
                # kernel, which must match as well.
                staged = _colored_staging_supported(
                    int(model.device.arch), 1, solver.world_count, int(solver.max_propagation_bodies)
                )
                tag = "_pf1_st1" if staged else "_pf1_st0_pk1"
            self.assertIn(tag, kernel.key, f"{name} did not build its kernel variant")
            a, b = model.state(), model.state()
            for field, value in start.items():
                getattr(a, field).assign(value)
            solver.reset(a, flags=0)
            for _ in range(5):
                solver.step(a, b, control, contacts, 1.0 / 240.0)
                a, b = b, a
            results[name] = (a.body_q.numpy(), a.body_qd.numpy(), solver.propagation_impulses.numpy())

        ref = results["reference"]
        self.assertTrue(all(np.isfinite(x).all() for x in ref))
        self.assertGreater(np.abs(ref[2]).max(), 0.0, "no contact impulses were solved")
        for name in ("prefetch", "default", "staged"):
            for label, got, want in zip(("body_q", "body_qd", "impulses"), results[name], ref, strict=True):
                np.testing.assert_array_equal(got, want, err_msg=f"{name} {label} differs from the reference")


if __name__ == "__main__":
    unittest.main()
