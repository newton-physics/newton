# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""SolverVBD ``rigid_rod_direct_solve``: rod topology, contact and per-axis bending.

The physics of the slender-rod regime is covered by ``test_rod_direct_regime.py``, model changes, graph capture and
failure reporting by ``test_rod_direct_lifecycle.py``. These tests use a stretch/bend stiffness ratio
``EA * l**2 / (4 * EI)`` of 1, where per-body VBD converges in a few iterations, so the coloured per-body solve is a
cheap reference for trees and loops.
"""

from __future__ import annotations

import math
import unittest

import numpy as np
import warp as wp

import newton
from newton.exceptions import NewtonWarning
from newton.tests.unittest_utils import add_function_test, get_test_devices
from newton.tests.vbd.rod_direct_utils import (
    DENSITY,
    GRAVITY,
    SEG,
    add_rod,
    clamped_chain,
    clamped_tree,
    make_rod,
    run_solver,
)


def _hanging_ring(test, device, segments: int = 12) -> tuple[newton.Model, list[int]]:
    """A closed vertical ring hung from the world by a ball joint at its top segment (no kinematic rod body)."""
    r = SEG / (2.0 * math.sin(math.pi / segments))
    theta = np.linspace(0.5 * math.pi, 2.5 * math.pi, segments + 1)
    points = [(r * math.cos(t), 0.0, 1.0 + r * math.sin(t)) for t in theta]
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, -GRAVITY))
    with test.assertWarnsRegex(NewtonWarning, "wrap_in_articulation=False"):
        bodies, joints = add_rod(builder, make_rod(points, closed=True), wrap=False)
    builder.add_articulation(joints[:-1])  # the open chain; the loop-closing rod joint stays outside it
    builder.add_joint_ball(  # the ring's support, a world joint outside the articulation
        parent=-1,
        child=bodies[0],
        parent_xform=wp.transform(wp.vec3(*points[0]), wp.quat_identity()),
        child_xform=wp.transform(wp.vec3(0.0, 0.0, -0.5 * SEG), wp.quat_identity()),
    )
    builder.color()
    return builder.finalize(device=device), bodies


def _assert_same_rest(test, model, bodies, what: str, dt: float, seconds: float, reference_iterations: int):
    """The direct solve settles where the converged per-body solve does, to 1 % of the droop."""
    steps = round(seconds / dt)
    _, ref = run_solver(model, False, steps, dt, iterations=reference_iterations)
    solver, out = run_solver(model, True, steps, dt)
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
    model, bodies = clamped_tree(device)
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
            builder, bodies, joints = clamped_chain(device, segments=2)
            dof = builder.joint_qd_start[joints[0]]
            builder.joint_target_ke[dof + 4] *= scale  # bend about the segment's y axis, which gravity loads
            builder.color()
            model = builder.finalize(device=device)
            z0 = float(model.body_q.numpy()[bodies[1], 2])
            solver, state = run_solver(model, True, steps=200, dt=1.0 / 200.0)
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


devices = get_test_devices()


class TestRodDirectTopology(unittest.TestCase):
    """Trees, loops and contacting rods (the expensive scenarios)."""


class TestRodDirectSolve(unittest.TestCase):
    """A one-segment rod with per-axis bending against analytic statics."""


for _fn in (
    test_tree_settles_like_the_per_body_solve,
    test_loop_with_a_ball_joint_settles_like_the_per_body_solve,
    test_crossed_rods_settle_on_the_ground,
):
    add_function_test(TestRodDirectTopology, _fn.__name__, _fn, devices=devices)
add_function_test(
    TestRodDirectSolve,
    test_one_segment_rod_bends_about_its_loaded_axis.__name__,
    test_one_segment_rod_bends_about_its_loaded_axis,
    devices=devices,
)


if __name__ == "__main__":
    unittest.main(verbosity=2)
