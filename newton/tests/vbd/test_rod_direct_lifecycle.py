# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""SolverVBD ``rigid_rod_direct_solve``: model changes, graph capture and failure reporting."""

from __future__ import annotations

import math
import unittest
import warnings

import numpy as np
import warp as wp

import newton
from newton._src.utils import is_graph_capture_allocation_enabled
from newton.exceptions import NewtonWarning
from newton.tests.unittest_utils import StdOutCapture, add_function_test, get_test_devices
from newton.tests.vbd.rod_direct_utils import GRAVITY, SEG, add_rod, clamped_chain, clamped_tree, make_rod, run_solver


def test_disabled_rod_joints_release_the_rod(test, device):
    """Release every body once all rod joints are disabled and the solver is notified."""

    def disable_joints(solver):
        model.joint_enabled.assign(np.zeros(model.joint_count, dtype=model.joint_enabled.numpy().dtype))
        solver.notify_model_changed(newton.ModelFlags.JOINT_PROPERTIES)

    results = []
    for direct in (False, True):
        builder, bodies, _ = clamped_chain(device, segments=6)
        builder.color()
        model = builder.finalize(device=device)
        _, state = run_solver(model, direct, steps=30, dt=1.0 / 300.0, prepare=disable_joints)
        results.append(state.body_q.numpy())
    np.testing.assert_array_equal(results[1], results[0])  # both run the per-body path once no rod remains
    fall = 1.0 - float(np.max(results[1][bodies[1:], 2]))
    test.assertGreater(fall, 0.9 * 0.5 * GRAVITY * (30 / 300.0) ** 2)


def test_body_made_kinematic_is_not_moved(test, device):
    """Leave a rod body in place once it is flagged kinematic and the solver is notified."""
    builder, bodies, _ = clamped_chain(device, segments=6)
    builder.color()
    model = builder.finalize(device=device)
    tip = bodies[-1]

    def make_tip_kinematic(solver):
        flags = model.body_flags.numpy()
        flags[tip] |= int(newton.BodyFlags.KINEMATIC)
        model.body_flags.assign(flags)
        solver.notify_model_changed(newton.ModelFlags.BODY_PROPERTIES)

    q0 = model.body_q.numpy()[tip].copy()
    solver, state = run_solver(model, True, steps=30, dt=1.0 / 300.0, prepare=make_tip_kinematic)
    np.testing.assert_array_equal(state.body_q.numpy()[tip], q0)
    solver.check_rod_direct_status()


def test_refused_steps_keep_the_pose_and_raise(test, device):
    """Refuse a non-finite step and a singular block: the rod keeps its pose, warns, and the status check raises
    until the solver is reset."""
    for what, value in (("non-finite material", np.nan), ("negative stiffness", -1.0e9)):
        with test.subTest(case=what):
            builder, bodies, joints = clamped_chain(device, segments=4, gravity=0.0)  # at rest: the pose is fixed
            builder.color()
            model = builder.finalize(device=device)
            solver = newton.solvers.SolverVBD(model, iterations=2, rigid_rod_direct_solve=True)
            ke = model.joint_target_ke.numpy()
            ke[model.joint_qd_start.numpy()[joints[-1]] + 2] = value  # the last joint's stretch row
            model.joint_target_ke.assign(ke)
            s0, s1, control = model.state(), model.state(), model.control()
            q0 = s0.body_q.numpy().copy()
            capture = StdOutCapture()
            capture.begin()
            try:
                solver.step(s0, s1, control, None, 1.0 / 300.0)
                wp.synchronize_device(device)  # flush the kernel's printf into the capture
            finally:
                out = capture.end()
            test.assertIn("rigid_rod_direct_solve", out)
            # kept at the iteration's start pose, which is the at-rest initial pose up to float32 re-rounding
            np.testing.assert_allclose(s1.body_q.numpy()[bodies], q0[bodies], rtol=0.0, atol=1.0e-6)
            with test.assertRaises(RuntimeError):
                solver.check_rod_direct_status()
            solver.reset(s1, flags=0)  # clears the status of the reset worlds
            solver.check_rod_direct_status()


def test_stalled_line_search_is_reported(test, device):
    """Report a step whose line search finds no decrease, and accept the same step once backtracking is allowed.

    A joint folded to 150 degrees and spun fast overshoots with a full Newton step; with no backtracking the rod
    keeps its pose and is flagged, with the default backtracking the step is accepted.
    """
    builder, bodies, _ = clamped_chain(device, segments=3, gravity=0.0)
    builder.color()
    model = builder.finalize(device=device)
    q = model.body_q.numpy()
    hinge = 0.5 * (q[bodies[1], :3] + q[bodies[2], :3])
    fold = wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), math.radians(150.0))
    q[bodies[2], :3] = hinge + np.array(wp.quat_rotate(fold, wp.vec3(*(q[bodies[2], :3] - hinge))))
    q[bodies[2], 3:7] = np.array(wp.mul(fold, wp.quat(*q[bodies[2], 3:7])))
    model.body_q.assign(q)
    qd = model.body_qd.numpy()
    qd[bodies[2], 4] = 400.0  # rad/s about world y, folding further
    model.body_qd.assign(qd)
    for halvings, stalled in ((0, True), (None, False)):
        with test.subTest(halvings=halvings):

            def limit_backtracking(solver, halvings=halvings):
                if halvings is not None:
                    solver._rod_direct_max_halvings = halvings

            solver, _ = run_solver(model, True, steps=1, dt=1.0 / 60.0, iterations=1, prepare=limit_backtracking)
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                solver.check_rod_direct_status()
            test.assertEqual(any(issubclass(w.category, NewtonWarning) for w in caught), stalled)


def test_captured_step_follows_model_changes(test, device):
    """Replay a step captured before ``notify_model_changed``: it follows the rewritten rod layout and status array.

    Disabling the rod joints, making the tip kinematic, or doubling or zeroing its mass after capture gives the eager
    result; a refused step replayed after a notification still raises; a layout rewrite during capture raises, while an
    inertial-property notification is captured and refuses the rod steps once an inverse mass crosses zero (reported
    until reset).
    """
    if not is_graph_capture_allocation_enabled(device):
        test.skipTest("graph capture needs the CUDA memory pool")
    dt = 1.0 / 300.0

    def disable_joints(model):
        model.joint_enabled.assign(np.zeros(model.joint_count, dtype=model.joint_enabled.numpy().dtype))
        return newton.ModelFlags.JOINT_PROPERTIES

    def make_tip_kinematic(model):
        flags = model.body_flags.numpy()
        flags[-1] |= int(newton.BodyFlags.KINEMATIC)
        model.body_flags.assign(flags)
        return newton.ModelFlags.BODY_PROPERTIES

    def scale_tip_mass(model, scale):
        inverse = 1.0 / scale if scale else 0.0
        for name, factor in (
            ("body_mass", scale),
            ("body_inertia", scale),
            ("body_inv_mass", inverse),
            ("body_inv_inertia", inverse),
        ):
            values = getattr(model, name).numpy()
            values[-1] *= factor
            getattr(model, name).assign(values)
        return newton.ModelFlags.BODY_INERTIAL_PROPERTIES

    def double_tip_mass(model):  # no membership change
        return scale_tip_mass(model, 2.0)

    def zero_tip_inverse_mass(model):
        return scale_tip_mass(model, 0.0)

    for change in (disable_joints, make_tip_kinematic, double_tip_mass, zero_tip_inverse_mass):
        with test.subTest(change=change.__name__):
            q = {}
            for captured in (False, True):
                builder, _, _ = clamped_chain(device, segments=6)
                builder.color()
                model = builder.finalize(device=device)
                solver = newton.solvers.SolverVBD(model, iterations=10, rigid_rod_direct_solve=True)
                s0, s1, control = model.state(), model.state(), model.control()

                def step(solver=solver, s0=s0, s1=s1, control=control):  # two steps: the buffers end where they began
                    for a, b in ((s0, s1), (s1, s0)):
                        a.clear_forces()
                        solver.step(a, b, control, None, dt)

                graph = None
                if captured:
                    with wp.ScopedCapture(device) as capture:
                        step()
                    graph = capture.graph
                solver.notify_model_changed(change(model))
                for _ in range(15):
                    if graph is None:
                        step()
                    else:
                        wp.capture_launch(graph)
                q[captured] = s0.body_q.numpy()
                solver.check_rod_direct_status()
            np.testing.assert_allclose(q[True], q[False], rtol=0.0, atol=1.0e-6)
            if change in (make_tip_kinematic, zero_tip_inverse_mass):
                np.testing.assert_array_equal(q[True][-1], model.body_q.numpy()[-1])
            elif change is disable_joints:
                test.assertGreater(1.0 - float(np.max(q[True][1:, 2])), 0.9 * 0.5 * GRAVITY * (30 * dt) ** 2)

    builder, _, joints = clamped_chain(device, segments=4, gravity=0.0)
    builder.color()
    model = builder.finalize(device=device)
    solver = newton.solvers.SolverVBD(model, iterations=2, rigid_rod_direct_solve=True)
    s0, s1, control = model.state(), model.state(), model.control()
    with wp.ScopedCapture(device) as capture:
        solver.step(s0, s1, control, None, dt)
    solver.notify_model_changed(newton.ModelFlags.BODY_PROPERTIES)  # same layout, rewritten in place
    ke = model.joint_target_ke.numpy()
    ke[model.joint_qd_start.numpy()[joints[-1]] + 2] = np.nan
    model.joint_target_ke.assign(ke)
    out = StdOutCapture()
    out.begin()
    try:
        wp.capture_launch(capture.graph)
        wp.synchronize_device(device)  # flush the kernel's printf into the capture
    finally:
        out.end()
    with test.assertRaises(RuntimeError):
        solver.check_rod_direct_status()
    wp.capture_begin(device)
    try:
        with test.assertRaisesRegex(RuntimeError, "graph capture"):
            solver.notify_model_changed(newton.ModelFlags.JOINT_PROPERTIES)
    finally:
        wp.capture_end(device)

    builder, bodies, _ = clamped_chain(device, segments=4)
    builder.color()
    model = builder.finalize(device=device)
    solver = newton.solvers.SolverVBD(model, iterations=2, rigid_rod_direct_solve=True)
    s0, s1, control = model.state(), model.state(), model.control()
    with wp.ScopedCapture(device) as capture:  # coupled solvers notify inertial changes on every step
        solver.notify_model_changed(newton.ModelFlags.BODY_INERTIAL_PROPERTIES)
        solver.step(s0, s1, control, None, dt)
    wp.capture_launch(capture.graph)
    solver.check_rod_direct_status()
    inv_mass = model.body_inv_mass.numpy()
    inv_mass[bodies[-1]] = 0.0
    model.body_inv_mass.assign(inv_mass)
    out = StdOutCapture()
    out.begin()
    try:
        wp.capture_launch(capture.graph)
        wp.synchronize_device(device)
    finally:
        out.end()
    with test.assertRaisesRegex(RuntimeError, "crossed zero"):
        solver.check_rod_direct_status()
    solver.notify_model_changed(newton.ModelFlags.BODY_INERTIAL_PROPERTIES)  # outside capture: rewrites the layout
    with test.assertRaisesRegex(RuntimeError, "reset"):  # the refused steps stay reported
        solver.check_rod_direct_status()
    solver.reset(s0, flags=0)
    solver.check_rod_direct_status()


def test_unsupported_options_raise(test, device):
    """Refuse options the direct rod solve does not support, and keep the flag-off solver free of direct state."""
    model, _ = clamped_tree(device)
    with test.assertRaises(NotImplementedError):
        newton.solvers.SolverVBD(
            model,
            rigid_rod_direct_solve=True,
            rigid_soft_enable_dat=True,
            collision_pipeline=newton.CollisionPipeline(model),
        )
    solver = newton.solvers.SolverVBD(model)
    test.assertIsNone(solver._rod_direct)
    solver.check_rod_direct_status()  # no-op without direct rod solves

    points = [(i * SEG, 0.0, 1.0) for i in range(4)]
    builder = newton.ModelBuilder()
    newton.solvers.SolverVBD.register_custom_attributes(builder)
    add_rod(builder, make_rod(points))
    builder.color()
    model = builder.finalize(device=device)
    model.vbd.dahl_eps_max.fill_(0.2)
    model.vbd.dahl_tau.fill_(0.2)
    with test.assertRaisesRegex(NotImplementedError, "Dahl"):
        newton.solvers.SolverVBD(model, rigid_rod_direct_solve=True)

    builder = newton.ModelBuilder()
    bodies, _ = add_rod(builder, make_rod(points))
    builder.add_tendon([newton.TendonGuide(body=bodies[0]), newton.TendonGuide(body=bodies[-1])])
    builder.color()
    model = builder.finalize(device=device)
    with test.assertRaisesRegex(NotImplementedError, "tendons"):
        newton.solvers.SolverVBD(model, rigid_rod_direct_solve=True)


devices = get_test_devices()


class TestRodDirectLifecycle(unittest.TestCase):
    """Model changes eager and under graph capture, refused and stalled steps, unsupported options."""


for _fn in (
    test_disabled_rod_joints_release_the_rod,
    test_body_made_kinematic_is_not_moved,
    test_stalled_line_search_is_reported,
    test_unsupported_options_raise,
):
    add_function_test(TestRodDirectLifecycle, _fn.__name__, _fn, devices=devices)
# capture their own output: the kernel's warning is the expected output
for _fn in (test_refused_steps_keep_the_pose_and_raise, test_captured_step_follows_model_changes):
    add_function_test(TestRodDirectLifecycle, _fn.__name__, _fn, devices=devices, check_output=False)


if __name__ == "__main__":
    unittest.main(verbosity=2)
