# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Tests for bilateral (mimic + connect) pre-elimination in SolverFeatherPGS."""

import inspect
import unittest
import warnings

import numpy as np
import warp as wp

import newton
from newton._src.solvers.feather_pgs.kernels import (
    PGS_CONSTRAINT_TYPE_CONNECT,
    PGS_CONSTRAINT_TYPE_CONTACT,
    PGS_CONSTRAINT_TYPE_MIMIC,
)
from newton.tests.test_feather_pgs_connect import _build_four_bar, _loop_anchor_gap
from newton.tests.test_feather_pgs_mimic import _build_two_revolute_chain


class TestPreeliminationSignature(unittest.TestCase):
    def test_selective_option_is_keyword_only(self):
        """Keep the new selective option out of the legacy positional API."""
        parameters = inspect.signature(newton.solvers.SolverFeatherPGS).parameters
        option = parameters["bilateral_preelimination_include_mimics"]
        self.assertEqual(option.kind, inspect.Parameter.KEYWORD_ONLY)
        self.assertIs(option.default, True)

    def test_legacy_joint_gap_position_is_preserved(self):
        """Bind a legacy positional gap without silently coercing it to a boolean."""
        signature = inspect.signature(newton.solvers.SolverFeatherPGS)
        positional = [p for p in signature.parameters.values() if p.kind == inspect.Parameter.POSITIONAL_OR_KEYWORD]
        self.assertEqual(len(positional), 55)
        self.assertEqual(positional[12].name, "joint_limit_activation_gap")
        arguments = [object()] + [p.default for p in positional[1:12]] + [0.123]
        bound = signature.bind_partial(*arguments, bilateral_preelimination_include_mimics=False)
        self.assertEqual(bound.arguments["joint_limit_activation_gap"], 0.123)
        self.assertIs(bound.arguments["bilateral_preelimination_include_mimics"], False)


def _run_four_bar(steps: int = 720, crank_target: float = 0.6, pgs_iterations: int = 2, **solver_kwargs):
    """Drive the connect-test four-bar and track the worst loop-anchor gap.

    Deliberately runs at a LOW iteration count: with iterative closure rows the
    anchor gap is convergence-limited (measured ~0.5 mm at 2 iterations), while
    pre-elimination enforces the closure independently of the sweep budget.
    """
    model = _build_four_bar().finalize()
    solver = newton.solvers.SolverFeatherPGS(
        model, pgs_mode="matrix_free", pgs_iterations=pgs_iterations, pgs_beta=0.1, **solver_kwargs
    )
    state_0, state_1 = model.state(), model.state()
    control = model.control()
    targets = model.joint_target_q.numpy().copy()
    targets[0] = crank_target
    control.joint_target_q.assign(targets)
    gap_max = 0.0
    for _ in range(steps):
        state_0.clear_forces()
        solver.step(state_0, state_1, control, None, 1.0 / 240.0)
        state_0, state_1 = state_1, state_0
        gap_max = max(gap_max, _loop_anchor_gap(model, state_0))
    return solver, model, state_0, gap_max


def _build_overcapacity_mixed_bilateral_model():
    """Build one articulation with three connect rows and six mimic rows."""
    builder = _build_four_bar()
    # Six rows on one follower need the deprecated sparse API; joint-owned mimics allow one per joint.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        for index in range(6):
            builder.add_constraint_mimic(joint0=1, joint1=0, coef0=0.0, coef1=1.0, label=f"mimic_{index}")
    return builder.finalize()


@unittest.skipUnless(wp.get_device().is_cuda, "SolverFeatherPGS matrix-free mode requires CUDA")
class TestFeatherPGSPreelimination(unittest.TestCase):
    def test_default_preserves_all_bilateral_trajectory(self):
        """Match the implicit default to explicit all-bilateral elimination."""
        builder = _build_four_bar()
        builder.set_joint_mimic(1, 0)
        model = builder.finalize()
        solvers = [
            newton.solvers.SolverFeatherPGS(
                model, pgs_mode="matrix_free", pgs_iterations=8, enable_bilateral_preelimination=True, **options
            )
            for options in ({}, {"bilateral_preelimination_include_mimics": True})
        ]
        states = [(model.state(), model.state()) for _ in solvers]
        control = model.control()
        targets = model.joint_target_q.numpy().copy()
        targets[0] = 0.6
        control.joint_target_q.assign(targets)
        for _ in range(32):
            for index, solver in enumerate(solvers):
                state_in, state_out = states[index]
                state_in.clear_forces()
                solver.step(state_in, state_out, control, None, 1.0 / 240.0)
                states[index] = (state_out, state_in)
            for attribute in ("joint_q", "joint_qd", "body_q", "body_qd"):
                a, b = [getattr(pair[0], attribute).numpy() for pair in states]
                self.assertTrue(np.isfinite(a).all())
                np.testing.assert_array_equal(a, b)
        for solver in solvers:
            self.assertTrue(solver.bilateral_preelimination_include_mimics)
            self.assertTrue(solver._preelim_active)
            count = int(solver._preelim_nB.numpy()[0])
            slots = solver._preelim_slots.numpy()[:count]
            selected_types = solver.row_type.numpy()[0, slots]
            self.assertEqual(int(np.count_nonzero(selected_types == PGS_CONSTRAINT_TYPE_MIMIC)), 1)
            self.assertGreater(int(np.count_nonzero(selected_types == PGS_CONSTRAINT_TYPE_CONNECT)), 0)

    def test_mimics_can_remain_iterative_when_total_rows_exceed_capacity(self):
        """Pre-eliminate connect rows while leaving excess mimic rows iterative."""
        model = _build_overcapacity_mixed_bilateral_model()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            solver_all = newton.solvers.SolverFeatherPGS(
                model,
                pgs_mode="matrix_free",
                enable_bilateral_preelimination=True,
            )
        self.assertFalse(solver_all._preelim_active)
        self.assertTrue(any("9 bilateral rows" in str(w.message) for w in caught))

        solver = newton.solvers.SolverFeatherPGS(
            model,
            pgs_mode="matrix_free",
            pgs_iterations=8,
            enable_bilateral_preelimination=True,
            bilateral_preelimination_include_mimics=False,
        )
        self.assertTrue(solver._preelim_active)
        self.assertEqual(solver._preelim_count, 1)

        state_0, state_1 = model.state(), model.state()
        control = model.control()
        for _ in range(16):
            state_0.clear_forces()
            solver.step(state_0, state_1, control, None, 1.0 / 240.0)
            state_0, state_1 = state_1, state_0

        self.assertTrue(np.isfinite(state_0.body_q.numpy()).all())
        self.assertGreater(int(solver._preelim_nB.numpy()[0]), 0)
        self.assertLessEqual(int(solver._preelim_nB.numpy()[0]), 3)
        selected = solver._preelim_slots.numpy()[: int(solver._preelim_nB.numpy()[0])]
        selected_types = solver.row_type.numpy()[0, selected]
        np.testing.assert_array_equal(selected_types, np.full(len(selected), PGS_CONSTRAINT_TYPE_CONNECT))
        counts = solver.constraint_count.numpy()
        rows = solver.row_type.numpy()
        seen = {int(row) for world in range(rows.shape[0]) for row in rows[world, : counts[world]]}
        self.assertIn(int(PGS_CONSTRAINT_TYPE_CONNECT), seen)
        self.assertIn(int(PGS_CONSTRAINT_TYPE_MIMIC), seen)

    def test_closure_exact_at_low_iterations(self):
        """Pre-elimination makes the loop closure iteration-independent.

        At 2 PGS iterations the iterative CONNECT rows cannot converge (the
        anchor gap plateaus near half a millimetre); the pre-eliminated solve
        must hold the closure 100x tighter through the same driven stroke.
        This is the regression-first differential: without the feature the
        ratio is 1 and the assertions below fail.
        """
        solver_off, _, _state_off, gap_off = _run_four_bar()
        self.assertFalse(solver_off._preelim_active)
        solver_on, _, state_on, gap_on = _run_four_bar(enable_bilateral_preelimination=True)
        self.assertTrue(solver_on._preelim_active)
        self.assertEqual(solver_on._preelim_count, 1)

        self.assertTrue(np.isfinite(state_on.body_q.numpy()).all())
        self.assertGreater(gap_off, 1.0e-4, "iterative baseline unexpectedly tight; differential is meaningless")
        self.assertLess(gap_on, 2.0e-5, f"pre-eliminated closure gap {gap_on:.2e} m not exact")
        self.assertLess(gap_on, 0.01 * gap_off, f"expected >100x tightening, got {gap_off / max(gap_on, 1e-12):.1f}x")

    @unittest.skip(
        "Obsolete drive-only fixture: dense non-contact rows deliberately cold-start; "
        "test_noncontact_dense_cache_is_cold_initialized covers that contract. "
        "A contact-bearing fixture is required to exercise warm-start projection ordering."
    )
    def test_dense_warmstart_preserves_projected_closure(self):
        """Project loaded contact history and an unsaturated drive kick at zero sweeps."""
        builder = _build_four_bar()
        # An unlimited drive stays in the augmented predictor, so reversing it
        # still changes velocity in the final step with no PGS drive sweeps.
        builder.joint_effort_limit[0] = float("inf")
        builder.add_shape_box(
            -1,
            xform=wp.transform(wp.vec3(-0.04, 0.0, 0.5), wp.quat_identity()),
            hx=0.02,
            hy=0.1,
            hz=0.1,
        )
        model = builder.finalize()
        solver = newton.solvers.SolverFeatherPGS(
            model,
            pgs_mode="matrix_free",
            pgs_iterations=2,
            pgs_beta=0.1,
            dense_max_constraints=128,
            enable_bilateral_preelimination=True,
            pgs_warmstart=True,
        )
        pipeline = newton.CollisionPipeline(model, contact_matching="latest")
        contacts = pipeline.contacts()
        state_0, state_1 = model.state(), model.state()
        control = model.control()
        targets = model.joint_target_q.numpy().copy()
        targets[0] = 0.6
        control.joint_target_q.assign(targets)
        for _ in range(120):
            pipeline.collide(state_0, contacts)
            state_0.clear_forces()
            solver.step(state_0, state_1, control, contacts, 1.0 / 240.0)
            state_0, state_1 = state_1, state_0

        count = int(solver.constraint_count.numpy()[0])
        contact_rows = solver.row_type.numpy()[0, :count] == PGS_CONSTRAINT_TYPE_CONTACT
        self.assertTrue(contact_rows.any())
        cache_peak = np.max(solver.impulses.numpy()[0, :count][contact_rows])
        self.assertGreater(cache_peak, 1.0e-4, "loaded contact cache stayed empty")
        previous_predictor = solver.v_hat.numpy().copy()
        targets[0] = -0.6
        control.joint_target_q.assign(targets)
        solver.pgs_iterations = 0
        pipeline.collide(state_0, contacts)
        state_0.clear_forces()
        solver.step(state_0, state_1, control, contacts, 1.0 / 240.0)
        gap = _loop_anchor_gap(model, state_1)
        self.assertTrue(solver._preelim_active)
        self.assertTrue(np.isfinite(state_1.body_q.numpy()).all())
        self.assertGreater(np.max(solver.impulses.numpy()), 1.0e-4)
        self.assertGreater(np.linalg.norm(solver.v_hat.numpy() - previous_predictor), 1.0e-3)
        self.assertLess(gap, 2.0e-5, f"warm-started projection left a {gap:.2e} m closure gap")

    def test_propagation_uses_iterative_bilateral_fallback(self):
        """Keep bilateral rows active when propagation lacks a projected response."""
        builder, _, _ = _build_two_revolute_chain(0.0, 1.0)
        model = builder.finalize()
        with self.assertWarnsRegex(UserWarning, "propagation"):
            solver = newton.solvers.SolverFeatherPGS(
                model,
                pgs_mode="matrix_free",
                articulated_contact_response="propagation",
                enable_bilateral_preelimination=True,
            )
        self.assertFalse(solver._preelim_active)
        self.assertGreater(solver._mimic_count, 0)

    def test_mechanism_behavior_preserved(self):
        """The rocker still tracks the crank through the closed loop (parallel four-bar)."""
        _, _, state, _ = _run_four_bar(enable_bilateral_preelimination=True, pgs_iterations=16)
        q = state.joint_q.numpy()
        self.assertAlmostEqual(q[2], q[0], delta=0.08)
        self.assertGreater(abs(q[2]), 0.3, "rocker did not move — loop not transmitting")

    def test_rows_remain_allocated(self):
        """B rows keep their dense slots (layout/warm-start stability); they are neutralized, not removed."""
        solver, _, _, _ = _run_four_bar(steps=60, enable_bilateral_preelimination=True)
        counts = solver.constraint_count.numpy()
        rows = solver.row_type.numpy()
        seen = set()
        for w in range(rows.shape[0]):
            seen.update(rows[w, : counts[w]].tolist())
        self.assertIn(int(PGS_CONSTRAINT_TYPE_CONNECT), seen)

    def test_unsupported_mode_warns_and_falls_back(self):
        """Non-matrix-free modes warn and keep the iterative rows."""
        model = _build_four_bar().finalize()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            solver = newton.solvers.SolverFeatherPGS(
                model, pgs_mode="split", pgs_iterations=16, enable_bilateral_preelimination=True
            )
        self.assertFalse(solver._preelim_active)
        self.assertTrue(any("matrix_free" in str(w.message) for w in caught))

    def test_velocity_iterations_warn_and_fall_back(self):
        """The TGS velocity pass is unsupported in v1: warn and fall back."""
        model = _build_four_bar().finalize()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            solver = newton.solvers.SolverFeatherPGS(
                model,
                pgs_mode="matrix_free",
                pgs_iterations=16,
                pgs_velocity_iterations=4,
                enable_bilateral_preelimination=True,
            )
        self.assertFalse(solver._preelim_active)
        self.assertTrue(any("velocity_iterations" in str(w.message) for w in caught))


if __name__ == "__main__":
    unittest.main()
