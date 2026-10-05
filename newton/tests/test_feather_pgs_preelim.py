# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Bilateral (mimic and connect) pre-elimination of SolverFeatherPGS."""

import inspect
import unittest
import warnings

import numpy as np
import warp as wp

import newton
from newton._src.solvers.feather_pgs import kernels as feather_pgs_kernels
from newton._src.solvers.feather_pgs.kernels import (
    PGS_CONSTRAINT_TYPE_CONNECT,
    PGS_CONSTRAINT_TYPE_CONTACT,
    PGS_CONSTRAINT_TYPE_MIMIC,
)
from newton.solvers import SolverFeatherPGS
from newton.tests.test_feather_pgs_connect import _build_carried_load, _build_four_bar, _loop_anchor_gap
from newton.tests.test_feather_pgs_mimic import _build_two_revolute_chain
from newton.tests.unittest_utils import add_function_test, get_cuda_test_devices, get_test_devices

DT = 1.0 / 240.0


class TestPreeliminationSignature(unittest.TestCase):
    def test_options_are_keyword_only(self):
        """The pre-elimination options are keyword-only with an off-by-default switch."""
        parameters = inspect.signature(SolverFeatherPGS).parameters
        for name, default in (
            ("enable_bilateral_preelimination", False),
            ("bilateral_preelimination_include_mimics", True),
        ):
            with self.subTest(option=name):
                self.assertEqual(parameters[name].kind, inspect.Parameter.KEYWORD_ONLY)
                self.assertIs(parameters[name].default, default)


class TestPreeliminationKernels(unittest.TestCase):
    """Kernel-level contracts of the connect rows and the regularized elimination, on CPU."""

    def test_connect_rows_are_normalized(self):
        """Scale each connect row to a unit Jacobian so a short lever arm does not weaken it."""
        device = "cpu"

        def arr(values, dtype=int):
            return wp.array(values, dtype=dtype, device=device)

        J = wp.zeros((1, 3, 1), dtype=float, device=device)
        phi = wp.zeros((1, 3), dtype=float, device=device)
        target = wp.zeros((1, 3), dtype=float, device=device)
        # A world anchor and one rotational child DOF with a 2 mm lever arm.
        wp.launch(
            feather_pgs_kernels.populate_connect_J_for_size,
            dim=1,
            inputs=[
                arr([0]),
                arr([0]),
                arr([0]),
                1,
                arr([0]),
                arr([0]),
                arr([-1]),
                arr([0]),
                arr([[0.002, 0.001, 0.0]], wp.vec3),
                arr([[0.002, 0.0, 0.0]], wp.vec3),
                arr([1]),
                arr([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]], wp.transform),
                wp.zeros(1, dtype=wp.spatial_vector, device=device),
                wp.zeros(1, dtype=wp.vec3, device=device),
                arr([0]),
                arr([-1]),
                arr([0, 1]),
                arr([[0.0, 0.0, 0.0, 0.0, 0.0, 1.0]], wp.spatial_vector),
                wp.zeros(1, dtype=wp.vec3, device=device),
            ],
            outputs=[
                J,
                wp.zeros((1, 3), dtype=int, device=device),
                wp.zeros((1, 3), dtype=int, device=device),
                wp.zeros((1, 3), dtype=float, device=device),
                phi,
                target,
            ],
            device=device,
        )
        # Unnormalized, the row would be J = -0.002 with phi = 0.001.
        self.assertAlmostEqual(float(J.numpy()[0, 1, 0]), -1.0, places=5)
        self.assertAlmostEqual(float(phi.numpy()[0, 1]), 0.5, places=5)

    def test_regularized_projection_leaves_documented_residual(self):
        """Leave the residual ``R (S + R)^-1 (J_B v + b_B)`` of the regularized block factor.

        With ``H = I``, a bilateral row ``J_B = [1, -1]`` (``S = 2``) and a second row
        ``[1, 0]``, the projection of ``v = [1, 0]`` and the corrected response of the second
        row both keep the factor ``R / (S + R)`` of their bilateral violation, with
        ``R = 1e-3 S + 1e-7``. The elimination is not exact.
        """
        device = "cpu"

        def arr(values, dtype=int):
            return wp.array(values, dtype=dtype, device=device)

        J = arr([[[1.0, -1.0], [1.0, 0.0]]], float)
        Y = arr(J.numpy(), float)
        slots = wp.full(8, -1, dtype=int, device=device)
        nB = wp.zeros(1, dtype=int, device=device)
        S = wp.zeros(64, dtype=float, device=device)
        reg = wp.zeros(8, dtype=float, device=device)
        LS = wp.zeros(64, dtype=float, device=device)
        reg_rel, reg_floor = 1.0e-3, 1.0e-7
        wp.launch(
            feather_pgs_kernels.preelim_setup_for_size,
            dim=1,
            inputs=[
                arr([0]),
                arr([0]),
                arr([0]),
                arr([0, 1]),
                arr([0]),
                1,
                arr([-1]),
                arr([-1]),
                0,
                J,
                Y,
                2,
                reg_rel,
                reg_floor,
            ],
            outputs=[slots, nB, S, reg, LS],
            device=device,
        )
        v = arr([1.0, 0.0], float)
        wp.launch(
            feather_pgs_kernels.preelim_project_velocity_for_size,
            dim=1,
            inputs=[
                arr([0]),
                arr([0]),
                arr([0]),
                arr([0]),
                slots,
                nB,
                LS,
                J,
                Y,
                wp.zeros((1, 2), dtype=float, device=device),
                2,
            ],
            outputs=[v],
            device=device,
        )
        wp.launch(
            feather_pgs_kernels.preelim_correct_Y_for_size,
            dim=2,
            inputs=[arr([0]), arr([0]), arr([0]), arr([2]), slots, nB, LS, J, 2, 2, 1],
            outputs=[Y],
            device=device,
        )
        s_block = 2.0
        regularizer = reg_rel * s_block + reg_floor
        self.assertAlmostEqual(float(reg.numpy()[0]), regularizer, places=7)
        expected = regularizer / (s_block + regularizer)
        residual = float(v.numpy()[0] - v.numpy()[1])
        leak = float(J.numpy()[0, 0] @ Y.numpy()[0, 1])
        np.testing.assert_allclose([residual, leak], [expected, expected], rtol=1.0e-3)
        self.assertGreater(residual, 0.0)


def _run_four_bar(device, steps: int = 720, crank_target: float = 0.6, pgs_iterations: int = 2, **solver_kwargs):
    """Drive the four-bar and track the worst loop-anchor gap.

    Runs at a low iteration count on purpose: with iterative connect rows the anchor gap
    is limited by convergence (about 0.5 mm at 2 iterations), while pre-elimination
    keeps it small at the same budget.
    """
    model = _build_four_bar().finalize(device=device)
    solver = SolverFeatherPGS(
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
        solver.step(state_0, state_1, control, None, DT)
        state_0, state_1 = state_1, state_0
        gap_max = max(gap_max, _loop_anchor_gap(model, state_0))
    return solver, model, state_0, gap_max


def _build_overcapacity_mixed_bilateral_model(device):
    """Build one articulation with three connect rows and six mimic rows."""
    builder = _build_four_bar()
    # Six rows on one follower need the deprecated API; joint-owned mimics allow one per joint.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        for index in range(6):
            builder.add_constraint_mimic(joint0=1, joint1=0, coef0=0.0, coef1=1.0, label=f"mimic_{index}")
    return builder.finalize(device=device)


def test_default_preserves_all_bilateral_trajectory(test, device):
    """The implicit default matches explicitly eliminating every bilateral row."""
    builder = _build_four_bar()
    builder.set_joint_mimic(1, 0)
    model = builder.finalize(device=device)
    solvers = [
        SolverFeatherPGS(
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
            solver.step(state_in, state_out, control, None, DT)
            states[index] = (state_out, state_in)
        for attribute in ("joint_q", "joint_qd", "body_q", "body_qd"):
            a, b = [getattr(pair[0], attribute).numpy() for pair in states]
            test.assertTrue(np.isfinite(a).all())
            np.testing.assert_array_equal(a, b)
    for solver in solvers:
        test.assertTrue(solver.bilateral_preelimination_include_mimics)
        test.assertTrue(solver._preelim_active)
        count = int(solver._preelim_nB.numpy()[0])
        slots = solver._preelim_slots.numpy()[:count]
        selected_types = solver.row_type.numpy()[0, slots]
        test.assertEqual(int(np.count_nonzero(selected_types == PGS_CONSTRAINT_TYPE_MIMIC)), 1)
        test.assertGreater(int(np.count_nonzero(selected_types == PGS_CONSTRAINT_TYPE_CONNECT)), 0)


def test_mimics_can_remain_iterative_when_total_rows_exceed_capacity(test, device):
    """Eliminate the connect rows while excess mimic rows stay iterative."""
    model = _build_overcapacity_mixed_bilateral_model(device)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        solver_all = SolverFeatherPGS(model, pgs_mode="matrix_free", enable_bilateral_preelimination=True)
    test.assertFalse(solver_all._preelim_active)
    test.assertTrue(any("9 bilateral rows" in str(w.message) for w in caught))

    solver = SolverFeatherPGS(
        model,
        pgs_mode="matrix_free",
        pgs_iterations=8,
        enable_bilateral_preelimination=True,
        bilateral_preelimination_include_mimics=False,
    )
    test.assertTrue(solver._preelim_active)
    test.assertEqual(solver._preelim_count, 1)

    state_0, state_1 = model.state(), model.state()
    control = model.control()
    for _ in range(16):
        state_0.clear_forces()
        solver.step(state_0, state_1, control, None, DT)
        state_0, state_1 = state_1, state_0

    test.assertTrue(np.isfinite(state_0.body_q.numpy()).all())
    count = int(solver._preelim_nB.numpy()[0])
    test.assertGreater(count, 0)
    test.assertLessEqual(count, 3)
    selected_types = solver.row_type.numpy()[0, solver._preelim_slots.numpy()[:count]]
    np.testing.assert_array_equal(selected_types, np.full(count, PGS_CONSTRAINT_TYPE_CONNECT))
    counts = solver.constraint_count.numpy()
    rows = solver.row_type.numpy()
    seen = {int(row) for world in range(rows.shape[0]) for row in rows[world, : counts[world]]}
    test.assertIn(int(PGS_CONSTRAINT_TYPE_CONNECT), seen)
    test.assertIn(int(PGS_CONSTRAINT_TYPE_MIMIC), seen)


def test_closure_held_at_low_iterations(test, device):
    """Pre-elimination keeps the loop closure tight at a low iteration count.

    At 2 iterations the iterative connect rows cannot converge (the anchor gap stays
    near half a millimetre); the regularized elimination holds the closure 100x tighter
    through the same driven stroke.
    """
    solver_off, _, _, gap_off = _run_four_bar(device)
    test.assertFalse(solver_off._preelim_active)
    solver_on, _, state_on, gap_on = _run_four_bar(device, enable_bilateral_preelimination=True)
    test.assertTrue(solver_on._preelim_active)
    test.assertEqual(solver_on._preelim_count, 1)

    test.assertTrue(np.isfinite(state_on.body_q.numpy()).all())
    test.assertGreater(gap_off, 1.0e-4, "iterative baseline unexpectedly tight; the comparison is meaningless")
    test.assertLess(gap_on, 2.0e-5, f"pre-eliminated closure gap {gap_on:.2e} m too loose")
    test.assertLess(gap_on, 0.01 * gap_off, f"expected >100x tightening, got {gap_off / max(gap_on, 1e-12):.1f}x")


def test_mimic_held_at_low_iterations(test, device):
    """An eliminated mimic row coupled to a closure holds at one iteration.

    In the parallel four-bar the coupler joint angle is minus the crank angle; a mimic row
    stating the same relationship competes with the connect rows in an iterative sweep.
    """
    errors = []
    for enabled in (False, True):
        builder = _build_four_bar()
        builder.set_joint_mimic(1, 0, coeffs=(0.0, -1.0))
        model = builder.finalize(device=device)
        solver = SolverFeatherPGS(
            model, pgs_mode="matrix_free", pgs_iterations=1, pgs_beta=0.1, enable_bilateral_preelimination=enabled
        )
        test.assertEqual(solver._preelim_active, enabled)
        state_0, state_1 = model.state(), model.state()
        control = model.control()
        targets = model.joint_target_q.numpy().copy()
        targets[0] = 0.6
        control.joint_target_q.assign(targets)
        worst = 0.0
        for _ in range(480):
            solver.step(state_0, state_1, control, None, DT)
            state_0, state_1 = state_1, state_0
            q = state_0.joint_q.numpy()
            worst = max(worst, abs(float(q[1] + q[0])))
        test.assertGreater(abs(float(q[0])), 0.5)
        errors.append(worst)
    test.assertGreater(errors[0], 1.0e-5, "iterative baseline unexpectedly tight; the comparison is meaningless")
    test.assertLess(errors[1], 1.0e-6)
    test.assertLess(errors[1], 0.01 * errors[0], f"iterative {errors[0]:.2e} vs eliminated {errors[1]:.2e}")


def test_preelimination_with_free_body_in_world(test, device):
    """A free body elsewhere in the world does not change an eliminated, contacting four-bar.

    The four-bar's coupler rests on the ground, so its contact rows are corrected by the
    elimination. Adding a separate free box gives the world two size groups and the
    gathered (not aliased) world response; the correction must reach it.
    """
    trajectories = []
    for with_box in (False, True):
        builder = _build_four_bar()
        # The coupler's underside is at 0.38 m and starts in contact.
        builder.add_ground_plane(height=0.385)
        if with_box:
            box = builder.add_body(xform=wp.transform(wp.vec3(0.8, 0.0, 0.585), wp.quat_rpy(0.2, 0.1, 0.3)))
            builder.add_shape_box(box, hx=0.08, hy=0.08, hz=0.06)
        model = builder.finalize(device=device)
        solver = SolverFeatherPGS(
            model,
            pgs_mode="matrix_free",
            pgs_iterations=2,
            pgs_beta=0.1,
            dense_max_constraints=64,
            enable_bilateral_preelimination=True,
        )
        test.assertTrue(solver._preelim_active)
        test.assertEqual(solver._jy_world_aliased, not with_box)
        pipeline = newton.CollisionPipeline(model)
        contacts = pipeline.contacts()
        state_0, state_1 = model.state(), model.state()
        control = model.control()
        targets = model.joint_target_q.numpy().copy()
        targets[0] = 0.6
        control.joint_target_q.assign(targets)
        trajectory = []
        for _ in range(120):
            pipeline.collide(state_0, contacts)
            solver.step(state_0, state_1, control, contacts, DT)
            state_0, state_1 = state_1, state_0
            trajectory.append(state_0.joint_q.numpy()[:3].copy())
        test.assertTrue(np.isfinite(state_0.body_q.numpy()).all())
        test.assertFalse(bool(solver.constraint_overflow.numpy().any()))
        trajectories.append(np.asarray(trajectory))
    np.testing.assert_allclose(trajectories[1], trajectories[0], atol=1.0e-4)


def test_mechanism_behavior_preserved(test, device):
    """The rocker still tracks the crank through the closed loop (parallel four-bar)."""
    _, _, state, _ = _run_four_bar(device, enable_bilateral_preelimination=True, pgs_iterations=16)
    q = state.joint_q.numpy()
    test.assertAlmostEqual(q[2], q[0], delta=0.08)
    test.assertGreater(abs(q[2]), 0.3, "rocker did not move: loop not transmitting")


def test_rows_remain_allocated(test, device):
    """Eliminated rows keep their dense slots and stay in the sweep."""
    solver, _, _, _ = _run_four_bar(device, steps=60, enable_bilateral_preelimination=True)
    counts = solver.constraint_count.numpy()
    rows = solver.row_type.numpy()
    seen = set()
    for w in range(rows.shape[0]):
        seen.update(rows[w, : counts[w]].tolist())
    test.assertIn(int(PGS_CONSTRAINT_TYPE_CONNECT), seen)


def test_prescribed_parent_closure_warns_and_falls_back(test, device):
    """A closure with a kinematic or world parent disables elimination for the whole solver."""
    b, _, _, _ = _build_carried_load()
    with test.assertWarnsRegex(UserWarning, "kinematic or world parent.*whole solver"):
        solver = SolverFeatherPGS(
            b.finalize(device=device), pgs_mode="matrix_free", enable_bilateral_preelimination=True
        )
    test.assertFalse(solver._preelim_active)
    test.assertEqual(solver._connect_count, 3)

    # An eligible four-bar is eliminated alone, but not next to a world-parent closure.
    eligible = SolverFeatherPGS(
        _build_four_bar().finalize(device=device), pgs_mode="matrix_free", enable_bilateral_preelimination=True
    )
    test.assertTrue(eligible._preelim_active)
    b = _build_four_bar()
    load = b.add_body(xform=wp.transform(wp.vec3(1.0, 0.0, 0.5), wp.quat_identity()))
    b.add_shape_box(load, hx=0.04, hy=0.04, hz=0.04)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=r".*another joint already connects these bodies")
        b.add_joint_ball(-1, load, parent_xform=wp.transform(wp.vec3(1.0, 0.0, 0.5), wp.quat_identity()))
    with test.assertWarnsRegex(UserWarning, "whole solver"):
        mixed = SolverFeatherPGS(
            b.finalize(device=device), pgs_mode="matrix_free", enable_bilateral_preelimination=True
        )
    test.assertFalse(mixed._preelim_active)
    test.assertEqual(mixed._preelim_count, 0)
    test.assertEqual(mixed._connect_count, 2)


def test_preelimination_capture_matches_eager(test, device):
    """The pre-eliminated four-bar replays its eager trajectory from a CUDA graph."""
    model = _build_four_bar().finalize(device=device)
    control = model.control()
    targets = model.joint_target_q.numpy().copy()
    targets[0] = 0.6
    control.joint_target_q.assign(targets)
    options = {"pgs_iterations": 2, "pgs_beta": 0.1, "enable_bilateral_preelimination": True}
    eager, captured = (
        SolverFeatherPGS(model, pgs_mode="matrix_free", **options),
        SolverFeatherPGS(model, pgs_mode="matrix_free", **options),
    )
    e0, e1 = model.state(), model.state()
    c0, c1 = model.state(), model.state()

    def step_pair(solver, s0, s1):
        solver.step(s0, s1, control, None, DT)
        solver.step(s1, s0, control, None, DT)

    step_pair(captured, c0, c1)
    step_pair(eager, e0, e1)
    # Capturing records the launches without running them.
    with wp.ScopedCapture(device=device) as capture:
        step_pair(captured, c0, c1)
    for _ in range(60):
        wp.capture_launch(capture.graph)
        step_pair(eager, e0, e1)
    np.testing.assert_allclose(c0.joint_q.numpy(), e0.joint_q.numpy(), atol=2.0e-5)
    test.assertLess(_loop_anchor_gap(model, c0), 2.0e-5)


def _run_loaded_four_bar(device, pgs_warmstart):
    """Rest a 1 kg box on an eliminated four-bar's coupler; return the solver, worst gap and carried impulse."""
    builder = _build_four_bar()
    box = builder.add_body(xform=wp.transform(wp.vec3(0.2, 0.0, 0.4705), wp.quat_identity()))
    builder.add_shape_box(box, hx=0.05, hy=0.05, hz=0.05)
    model = builder.finalize(device=device)
    solver = SolverFeatherPGS(
        model,
        pgs_mode="matrix_free",
        pgs_iterations=1,
        pgs_beta=0.1,
        dense_max_constraints=64,
        friction_anchor_beta=0.0,
        enable_bilateral_preelimination=True,
        pgs_warmstart=pgs_warmstart,
    )
    pipeline = newton.CollisionPipeline(model, contact_matching="latest")
    contacts = pipeline.contacts()
    state_0, state_1 = model.state(), model.state()
    control = model.control()
    gap_max = carried = 0.0
    for step in range(240):
        pipeline.collide(state_0, contacts)
        solver.step(state_0, state_1, control, contacts, DT)
        state_0, state_1 = state_1, state_0
        if step >= 20:
            gap_max = max(gap_max, _loop_anchor_gap(model, state_0))
        if pgs_warmstart:
            history = solver._ws_prev_impulses.numpy()[0]
            contact_rows = solver._ws_prev_row_type.numpy()[0] == PGS_CONSTRAINT_TYPE_CONTACT
            carried = max(carried, float(np.abs(history[contact_rows]).max(initial=0.0)))
    return solver, model, state_0, box, gap_max, carried


def test_warmstart_with_preelimination_holds_a_loaded_closure(test, device):
    """Warm start a contact that loads an eliminated closure; the closure stays tight and the box rests.

    The carried contact impulse is installed into the velocity before the bilateral projection,
    so it cannot reopen the closure.
    """
    solver, model, state, box, gap_warm, carried = _run_loaded_four_bar(device, pgs_warmstart=True)
    _, _, _, _, gap_cold, _ = _run_loaded_four_bar(device, pgs_warmstart=False)
    test.assertTrue(solver._preelim_active)
    # The carried contact history is the box's weight impulse, m g dt.
    weight_impulse = float(model.body_mass.numpy()[box]) * 9.81 * DT
    test.assertAlmostEqual(carried, weight_impulse, delta=0.2 * weight_impulse)
    test.assertLess(gap_warm, 1.0e-6)
    test.assertLessEqual(gap_warm, gap_cold)
    test.assertAlmostEqual(float(state.body_q.numpy()[box, 2]), 0.47, delta=1.0e-3)


# -- Other solve paths --


def test_split_rejects_bilateral_rows(test, device):
    """Reject loop closures in the split solve, with or without pre-elimination."""
    model = _build_four_bar().finalize(device=device)
    for enabled in (False, True):
        with test.subTest(enable_bilateral_preelimination=enabled):
            with test.assertRaisesRegex(NotImplementedError, "require pgs_mode='matrix_free'"):
                SolverFeatherPGS(model, pgs_mode="split", enable_bilateral_preelimination=enabled)


def test_propagation_keeps_iterative_mimic_rows(test, device):
    """Keep iterative mimic rows with the propagation responses; pre-elimination falls back with a warning."""
    builder, _, _ = _build_two_revolute_chain(0.0, 1.0)
    model = builder.finalize(device=device)
    for response in ("propagation", "propagation-fused"):
        with test.subTest(response=response):
            with test.assertWarnsRegex(UserWarning, "propagation"):
                solver = SolverFeatherPGS(
                    model, articulated_contact_response=response, enable_bilateral_preelimination=True
                )
            test.assertFalse(solver._preelim_active)
            test.assertGreater(solver._mimic_count, 0)


def test_propagation_rejects_loop_joints(test, device):
    """Reject loop-closing joints with the propagation responses."""
    model = _build_four_bar().finalize(device=device)
    for response in ("propagation", "propagation-fused"):
        with test.subTest(response=response):
            with test.assertRaisesRegex(NotImplementedError, "Loop-closing joints.*articulated_contact_response"):
                SolverFeatherPGS(model, articulated_contact_response=response)


class TestFeatherPGSPreelimination(unittest.TestCase):
    pass


class TestFeatherPGSPreeliminationSplit(unittest.TestCase):
    pass


class TestFeatherPGSPreeliminationPropagation(unittest.TestCase):
    pass


cuda_devices = get_cuda_test_devices()
for _name in (
    "test_default_preserves_all_bilateral_trajectory",
    "test_mimics_can_remain_iterative_when_total_rows_exceed_capacity",
    "test_closure_held_at_low_iterations",
    "test_mimic_held_at_low_iterations",
    "test_preelimination_with_free_body_in_world",
    "test_mechanism_behavior_preserved",
    "test_rows_remain_allocated",
    "test_prescribed_parent_closure_warns_and_falls_back",
    "test_preelimination_capture_matches_eager",
    "test_warmstart_with_preelimination_holds_a_loaded_closure",
):
    add_function_test(TestFeatherPGSPreelimination, _name, globals()[_name], devices=cuda_devices)
add_function_test(
    TestFeatherPGSPreeliminationSplit,
    "test_split_rejects_bilateral_rows",
    test_split_rejects_bilateral_rows,
    devices=get_test_devices(),
)
add_function_test(
    TestFeatherPGSPreeliminationPropagation,
    "test_propagation_keeps_iterative_mimic_rows",
    test_propagation_keeps_iterative_mimic_rows,
    devices=cuda_devices,
)
add_function_test(
    TestFeatherPGSPreeliminationPropagation,
    "test_propagation_rejects_loop_joints",
    test_propagation_rejects_loop_joints,
    devices=cuda_devices,
)


if __name__ == "__main__":
    unittest.main()
