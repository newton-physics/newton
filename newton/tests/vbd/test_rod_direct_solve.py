# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""SolverVBD ``rigid_rod_direct_solve``: rod topology, model changes and failure reporting.

The physics of the slender-rod regime is covered by ``test_rod_direct_regime.py``. These tests use a stretch/bend
stiffness ratio ``EA * l**2 / (4 * EI)`` of 1, where per-body VBD converges in a few iterations, so the coloured
per-body solve is a cheap reference for trees and loops.
"""

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

SEG = 0.05  # m
EI = 0.01  # N m^2
EA = 4.0 * EI / SEG**2  # stretch/bend stiffness ratio 1
RADIUS = 0.01  # m
DENSITY = 1000.0  # kg/m^3
GRAVITY = 9.81


def _add_rod(builder: newton.ModelBuilder, rod: newton.Rod, wrap: bool = True) -> tuple[list[int], list[int]]:
    """Add a rod with explicit per-joint stiffness (a branched graph has no automatic section rigidity)."""
    k_stretch, k_bend = EA / SEG, EI / SEG
    bodies, joints = builder.add_rod(
        rod=rod,
        cfg=newton.ModelBuilder.ShapeConfig(density=DENSITY),
        stretch_stiffness=k_stretch,
        stretch_damping=0.05 * k_stretch,
        shear_stiffness=k_stretch,
        shear_damping=0.05 * k_stretch,
        bend_stiffness=k_bend,
        bend_damping=0.05 * k_bend,
        twist_stiffness=k_bend,
        twist_damping=0.05 * k_bend,
        wrap_in_articulation=wrap,
    )
    return [int(b) for b in bodies], [int(j) for j in joints]


def _rod(points, **kwargs) -> newton.Rod:
    return newton.Rod([wp.vec3(*p) for p in points], radius=RADIUS, **kwargs)


def _clamp(builder: newton.ModelBuilder, body: int) -> None:
    builder.body_mass[body] = 0.0
    builder.body_inv_mass[body] = 0.0
    builder.body_inertia[body] = wp.mat33(0.0)
    builder.body_inv_inertia[body] = wp.mat33(0.0)
    builder.body_flags[body] = int(newton.BodyFlags.KINEMATIC)


def _clamped_chain(device, segments: int, gravity: float = GRAVITY):
    """A straight rod along +X whose first segment is a kinematic clamp."""
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, -gravity))
    bodies, joints = _add_rod(builder, _rod([((i - 1) * SEG, 0.0, 1.0) for i in range(segments + 1)]))
    _clamp(builder, bodies[0])
    return builder, bodies, joints


def _clamped_tree(device) -> tuple[newton.Model, list[int]]:
    """A horizontal trunk clamped at its first edge, with a branch leaving its middle node sideways."""
    nodes = [(i * SEG, 0.0, 1.0) for i in range(6)] + [(3 * SEG, SEG, 1.0), (3 * SEG, 2 * SEG, 1.0)]
    edges = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (3, 6), (6, 7)]
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, -GRAVITY))
    bodies, _ = _add_rod(builder, _rod(nodes, edges=edges))
    _clamp(builder, bodies[0])
    builder.color()
    return builder.finalize(device=device), bodies


def _hanging_ring(test, device, segments: int = 12) -> tuple[newton.Model, list[int]]:
    """A closed vertical ring hung from the world by a ball joint at its top segment (no kinematic rod body)."""
    r = SEG / (2.0 * math.sin(math.pi / segments))
    theta = np.linspace(0.5 * math.pi, 2.5 * math.pi, segments + 1)
    points = [(r * math.cos(t), 0.0, 1.0 + r * math.sin(t)) for t in theta]
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, -GRAVITY))
    with test.assertWarnsRegex(NewtonWarning, "wrap_in_articulation=False"):
        bodies, joints = _add_rod(builder, _rod(points, closed=True), wrap=False)
    builder.add_articulation(joints[:-1])  # the open chain; the loop-closing rod joint stays outside it
    builder.add_joint_ball(  # the ring's support, a world joint outside the articulation
        parent=-1,
        child=bodies[0],
        parent_xform=wp.transform(wp.vec3(*points[0]), wp.quat_identity()),
        child_xform=wp.transform(wp.vec3(0.0, 0.0, -0.5 * SEG), wp.quat_identity()),
    )
    builder.color()
    return builder.finalize(device=device), bodies


def _run(model: newton.Model, direct: bool, steps: int, dt: float, iterations: int = 10, prepare=None):
    """Step from the model's initial state; ``prepare(solver)`` runs before the first step."""
    solver = newton.solvers.SolverVBD(model, iterations=iterations, rigid_rod_direct_solve=direct)
    if prepare is not None:
        prepare(solver)
    s0, s1, control = model.state(), model.state(), model.control()
    for _ in range(steps):
        s0.clear_forces()
        solver.step(s0, s1, control, None, dt)
        s0, s1 = s1, s0
    return solver, s0


def _assert_same_rest(test, model, bodies, what: str, dt: float, seconds: float, reference_iterations: int):
    """The direct solve settles where the converged per-body solve does, to 1 % of the droop."""
    steps = round(seconds / dt)
    _, ref = _run(model, False, steps, dt, iterations=reference_iterations)
    solver, out = _run(model, True, steps, dt)
    solver.check_rod_direct_status()
    q_ref, q = ref.body_q.numpy(), out.body_q.numpy()
    test.assertTrue(np.all(np.isfinite(q)))
    drop = 1.0 - float(np.min(q_ref[bodies, 2]))
    test.assertGreater(drop, 0.5 * SEG, msg=f"{what}: the reference barely moved ({drop:.4g} m)")
    err = float(np.max(np.linalg.norm(q[bodies, :3] - q_ref[bodies, :3], axis=1)))
    test.assertLess(err, 0.01 * drop, msg=f"{what}: direct vs per-body rest positions differ by {err:.4g} m")
    test.assertLess(float(np.max(np.abs(out.body_qd.numpy()[bodies]))), 0.1, msg=f"{what}: not settled")


def test_tree_settles_like_the_per_body_solve(test, device):
    """Settle a branched rod (one junction) to the converged per-body equilibrium."""
    model, bodies = _clamped_tree(device)
    _assert_same_rest(test, model, bodies, "clamped tree", dt=1.0 / 200.0, seconds=2.0, reference_iterations=30)


def test_loop_with_a_ball_joint_settles_like_the_per_body_solve(test, device):
    """Settle a closed ring hung by a ball joint (a loop closure and a non-rod joint) to the per-body equilibrium."""
    model, bodies = _hanging_ring(test, device)
    _assert_same_rest(test, model, bodies, "hanging ring", dt=1.0 / 300.0, seconds=1.0, reference_iterations=20)


def test_crossed_rods_settle_on_the_ground(test, device):
    """Settle three crossed layers of rods dropped on the ground without gaining energy, and keep them at rest.

    The rods touch each other and are solved in parallel, so their contacts couple simultaneous block solves; more
    iterations must converge, not amplify the contact response (with contact Hessian weight 1 the stack reaches 4 m).
    The resting bound (median speed below 2.5 mm/s) guards only against gross instability: on this scene the per-body
    solve rests at 1.6 mm/s (CPU) and the direct solve at 0.24 (CPU) / 1.55 (CUDA) mm/s.
    """
    radius, seg, segments = 0.012, 0.05, 6
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, -GRAVITY))
    builder.rigid_gap = 0.0
    builder.default_shape_cfg.ke, builder.default_shape_cfg.kd, builder.default_shape_cfg.mu = 1.0e5, 0.0, 1.0
    builder.add_ground_plane()
    length = segments * seg
    for layer in range(3):
        along = np.array([1.0, 0.0, 0.0]) if layer % 2 == 0 else np.array([0.0, 1.0, 0.0])
        across = np.array([0.0, 1.0, 0.0]) if layer % 2 == 0 else np.array([1.0, 0.0, 0.0])
        for lane in (-0.03, 0.03):
            origin = lane * across + np.array([0.0, 0.0, 0.05 + 3.0 * radius * layer])
            points = [wp.vec3(*(origin + (i * seg - 0.5 * length) * along)) for i in range(segments + 1)]
            builder.add_rod(
                rod=newton.Rod(points, radius=radius),
                stretch_stiffness=5.0e5,
                bend_stiffness=1.0e2,
                cfg=builder.default_shape_cfg,
            )
    builder.color()
    model = builder.finalize(device=device)
    solver = newton.solvers.SolverVBD(model, iterations=20, rigid_rod_direct_solve=True)
    pipeline = newton.CollisionPipeline(model)
    contacts = pipeline.contacts()
    s0, s1, control = model.state(), model.state(), model.control()
    dt = 1.0 / 600.0
    for _ in range(360):
        pipeline.collide(s0, contacts)
        s0.clear_forces()
        solver.step(s0, s1, control, contacts, dt)
        s0, s1 = s1, s0
    solver.check_rod_direct_status()
    q, qd = s0.body_q.numpy(), s0.body_qd.numpy()
    test.assertTrue(np.all(np.isfinite(q)))
    test.assertLess(float(np.max(q[:, 2])), 6.0 * radius, msg="the stack rose above its three layers")
    test.assertLess(float(np.max(np.abs(qd[:, :3]))), 0.05, msg="the stack has not settled")
    speed = float(np.median(np.linalg.norm(qd[:, :3], axis=1)))
    test.assertLess(speed, 2.5e-3, msg=f"the resting stack keeps moving (median speed {speed:.3g} m/s)")


def test_one_segment_rod_bends_about_its_loaded_axis(test, device):
    """Solve a rod of one dynamic segment directly, with per-axis bend stiffness, to its static droop.

    The segment hangs from a kinematic clamp by one rod joint; the droop of its center is the bend rotation's lever
    plus the joint's shear compliance.
    """
    for scale in (1.0, 10.0):
        with test.subTest(loaded_axis_bend_scale=scale):
            builder, bodies, joints = _clamped_chain(device, segments=2)
            dof = builder.joint_qd_start[joints[0]]
            builder.joint_target_ke[dof + 4] *= scale  # bend about the segment's y axis, which gravity loads
            builder.color()
            model = builder.finalize(device=device)
            z0 = float(model.body_q.numpy()[bodies[1], 2])
            solver, state = _run(model, True, steps=200, dt=1.0 / 200.0)
            solver.check_rod_direct_status()
            droop = z0 - float(state.body_q.numpy()[bodies[1], 2])
            ke = model.joint_target_ke.numpy()[dof : dof + 6]
            weight = float(model.body_mass.numpy()[bodies[1]]) * GRAVITY
            theta = 0.0  # k_bend * 2 tan(t/2) / cos^2(t/2) = weight * (l/2) * cos(t), by Newton iteration
            for _ in range(30):
                c2 = math.cos(0.5 * theta) ** 2
                f = ke[4] * 2.0 * math.tan(0.5 * theta) / c2 - weight * 0.5 * SEG * math.cos(theta)
                df = ke[4] * (1.0 + 2.0 * math.sin(0.5 * theta) ** 2) / c2**2 + weight * 0.5 * SEG * math.sin(theta)
                theta -= f / df
            expected = 0.5 * SEG * math.sin(theta) + weight / ke[0]
            test.assertAlmostEqual(droop / expected, 1.0, delta=0.01, msg=f"droop {droop:.6g} m vs {expected:.6g} m")


def test_disabled_rod_joints_release_the_rod(test, device):
    """Release every body once all rod joints are disabled and the solver is notified."""

    def disable_joints(solver):
        model.joint_enabled.assign(np.zeros(model.joint_count, dtype=model.joint_enabled.numpy().dtype))
        solver.notify_model_changed(newton.ModelFlags.JOINT_PROPERTIES)

    results = []
    for direct in (False, True):
        builder, bodies, _ = _clamped_chain(device, segments=6)
        builder.color()
        model = builder.finalize(device=device)
        _, state = _run(model, direct, steps=30, dt=1.0 / 300.0, prepare=disable_joints)
        results.append(state.body_q.numpy())
    np.testing.assert_array_equal(results[1], results[0])  # both run the per-body path once no rod remains
    fall = 1.0 - float(np.max(results[1][bodies[1:], 2]))
    test.assertGreater(fall, 0.9 * 0.5 * GRAVITY * (30 / 300.0) ** 2)


def test_body_made_kinematic_is_not_moved(test, device):
    """Leave a rod body in place once it is flagged kinematic and the solver is notified."""
    builder, bodies, _ = _clamped_chain(device, segments=6)
    builder.color()
    model = builder.finalize(device=device)
    tip = bodies[-1]

    def make_tip_kinematic(solver):
        flags = model.body_flags.numpy()
        flags[tip] |= int(newton.BodyFlags.KINEMATIC)
        model.body_flags.assign(flags)
        solver.notify_model_changed(newton.ModelFlags.BODY_PROPERTIES)

    q0 = model.body_q.numpy()[tip].copy()
    solver, state = _run(model, True, steps=30, dt=1.0 / 300.0, prepare=make_tip_kinematic)
    np.testing.assert_array_equal(state.body_q.numpy()[tip], q0)
    solver.check_rod_direct_status()


def test_refused_steps_keep_the_pose_and_raise(test, device):
    """Refuse a non-finite step and a singular block: the rod keeps its pose, warns, and the status check raises."""
    for what, value in (("non-finite material", np.nan), ("negative stiffness", -1.0e9)):
        with test.subTest(case=what):
            builder, bodies, joints = _clamped_chain(device, segments=4, gravity=0.0)  # at rest: the pose is fixed
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


def test_stalled_line_search_is_reported(test, device):
    """Report a step whose line search finds no decrease, and accept the same step once backtracking is allowed.

    A joint folded to 150 degrees and spun fast overshoots with a full Newton step; with no backtracking the rod
    keeps its pose and is flagged, with the default backtracking the step is accepted.
    """
    builder, bodies, _ = _clamped_chain(device, segments=3, gravity=0.0)
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

            solver, _ = _run(model, True, steps=1, dt=1.0 / 60.0, iterations=1, prepare=limit_backtracking)
            test.assertEqual(bool(solver.rod_direct_status.numpy()[0] & 2), stalled)
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                solver.check_rod_direct_status()
            test.assertEqual(any(issubclass(w.category, NewtonWarning) for w in caught), stalled)


def test_captured_step_follows_model_changes(test, device):
    """Replay a step captured before ``notify_model_changed``: it follows the rewritten rod layout and status array.

    Disabling the rod joints or making the tip kinematic after capture gives the eager result; a refused step replayed
    after a notification still raises; notifying during capture raises.
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

    for change in (disable_joints, make_tip_kinematic):
        with test.subTest(change=change.__name__):
            q = {}
            for captured in (False, True):
                builder, _, _ = _clamped_chain(device, segments=6)
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
            if change is make_tip_kinematic:
                np.testing.assert_array_equal(q[True][-1], model.body_q.numpy()[-1])
            else:
                test.assertGreater(1.0 - float(np.max(q[True][1:, 2])), 0.9 * 0.5 * GRAVITY * (30 * dt) ** 2)

    builder, _, joints = _clamped_chain(device, segments=4, gravity=0.0)
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


def test_unsupported_options_raise(test, device):
    """Refuse options the direct rod solve does not support, and keep the flag-off solver free of direct state."""
    model, _ = _clamped_tree(device)
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


devices = get_test_devices()


class TestRodDirectTopology(unittest.TestCase):
    """Trees, loops and contacting rods (the expensive scenarios)."""


class TestRodDirectSolve(unittest.TestCase):
    """Small scenarios: one-segment rods, model changes and failure reporting."""


for _fn in (
    test_tree_settles_like_the_per_body_solve,
    test_loop_with_a_ball_joint_settles_like_the_per_body_solve,
    test_crossed_rods_settle_on_the_ground,
):
    add_function_test(TestRodDirectTopology, _fn.__name__, _fn, devices=devices)
for _fn in (
    test_one_segment_rod_bends_about_its_loaded_axis,
    test_disabled_rod_joints_release_the_rod,
    test_body_made_kinematic_is_not_moved,
    test_stalled_line_search_is_reported,
    test_unsupported_options_raise,
):
    add_function_test(TestRodDirectSolve, _fn.__name__, _fn, devices=devices)
# capture their own output: the kernel's warning is the expected output
for _fn in (test_refused_steps_keep_the_pose_and_raise, test_captured_step_follows_model_changes):
    add_function_test(TestRodDirectSolve, _fn.__name__, _fn, devices=devices, check_output=False)


if __name__ == "__main__":
    unittest.main(verbosity=2)
