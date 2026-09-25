# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""SolverXPBD joint tests: hinge angles beyond +-pi, joint relaxation."""

import unittest

import numpy as np
import warp as wp

import newton
from newton.tests.unittest_utils import add_function_test, get_test_devices


def _hinge_link(device, limit_lower, limit_upper, q0=0.0, qd0=0.0, gravity=0.0, ke=0.0, kd=0.0, target=0.0):
    """One box link on a revolute joint (axis y) to the world, COM 0.15 m from the pivot."""
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, gravity))
    link = builder.add_link(xform=wp.transform((0.0, 0.0, 1.0), wp.quat_identity()), mass=1.0)
    builder.add_shape_box(link, xform=wp.transform((0.15, 0.0, 0.0), wp.quat_identity()), hx=0.15, hy=0.03, hz=0.03)
    joint = builder.add_joint_revolute(
        -1,
        link,
        parent_xform=wp.transform((0.0, 0.0, 1.0), wp.quat_identity()),
        axis=(0.0, 1.0, 0.0),
        limit_lower=limit_lower,
        limit_upper=limit_upper,
        target_ke=ke,
        target_kd=kd,
        target_pos=target,
    )
    builder.add_articulation([joint])
    builder.joint_q = [q0]
    builder.joint_qd = [qd0]
    return builder.finalize(device=device)


def _hinge_inertia(model):
    return (
        float(model.body_inertia.numpy()[0][1, 1])
        + float(model.body_mass.numpy()[0]) * float(model.body_com.numpy()[0][0]) ** 2
    )


def test_revolute_limit_beyond_pi_does_not_gain_energy(test, device):
    """A hinge spun into an upper limit beyond pi stops there instead of wrapping to -pi and exploding."""
    dt, w0 = 1.25e-3, 3.0
    for upper in (3.1, 3.4):
        model = _hinge_link(device, -0.5, upper, qd0=w0)
        solver = newton.solvers.SolverXPBD(
            model, iterations=4, joint_linear_relaxation=0.4, joint_angular_relaxation=0.4
        )
        s0, s1, control = model.state(), model.state(), model.control()
        newton.eval_fk(model, model.joint_q, model.joint_qd, s0)
        q = wp.zeros(1, dtype=float, device=device)
        qd = wp.zeros(1, dtype=float, device=device)
        peak = 0.0
        for _ in range(1200):
            solver.step(s0, s1, control, None, dt)
            s0, s1 = s1, s0
            newton.eval_ik(model, s0, q, qd)
            peak = max(peak, abs(float(qd.numpy()[0])))
        test.assertLessEqual(peak, w0 * 1.01)
        # comes to rest at the upper limit (the hinge is limit-bound; a small inelastic rebound is allowed)
        test.assertGreater(float(q.numpy()[0]), upper - 0.6)
        test.assertLessEqual(float(q.numpy()[0]), upper + 0.01)


def test_revolute_eval_ik_angle_beyond_pi(test, device):
    """eval_ik returns an angle beyond pi within the limit range, whatever the quaternion sign of the child body."""
    model = _hinge_link(device, -0.227, 3.421, q0=3.3)
    state = model.state()
    newton.eval_fk(model, model.joint_q, model.joint_qd, state)
    q = wp.zeros(1, dtype=float, device=device)
    qd = wp.zeros(1, dtype=float, device=device)
    newton.eval_ik(model, state, q, qd)
    test.assertAlmostEqual(float(q.numpy()[0]), 3.3, places=4)
    body_q = state.body_q.numpy()
    body_q[0, 3:] *= -1.0  # same orientation, other quaternion branch
    state.body_q.assign(body_q)
    newton.eval_ik(model, state, q, qd)
    test.assertAlmostEqual(float(q.numpy()[0]), 3.3, places=4)


def test_revolute_hold_beyond_pi(test, device):
    """A hinge resting beyond pi (inside its range) stays there; the solver must not read it as a limit violation."""
    model = _hinge_link(device, -0.227, 3.421, q0=3.3)
    solver = newton.solvers.SolverXPBD(model, iterations=4, joint_linear_relaxation=0.4, joint_angular_relaxation=0.4)
    s0, s1, control = model.state(), model.state(), model.control()
    newton.eval_fk(model, model.joint_q, model.joint_qd, s0)
    for _ in range(200):
        solver.step(s0, s1, control, None, 1.25e-3)
        s0, s1 = s1, s0
    q = wp.zeros(1, dtype=float, device=device)
    qd = wp.zeros(1, dtype=float, device=device)
    newton.eval_ik(model, s0, q, qd)
    test.assertAlmostEqual(float(q.numpy()[0]), 3.3, places=3)
    test.assertLess(abs(float(qd.numpy()[0])), 1e-3)


def test_revolute_unlimited_drive_takes_short_way(test, device):
    """An unlimited hinge driven to a target measures its error within pi of the target (no 2 pi detour)."""
    model = _hinge_link(device, -newton.MAXVAL, newton.MAXVAL, q0=3.0, ke=100.0, kd=5.0, target=-3.0)
    solver = newton.solvers.SolverXPBD(model, iterations=4, joint_linear_relaxation=0.4, joint_angular_relaxation=0.4)
    s0, s1, control = model.state(), model.state(), model.control()
    newton.eval_fk(model, model.joint_q, model.joint_qd, s0)
    for _ in range(800):
        solver.step(s0, s1, control, None, 1.25e-3)
        s0, s1 = s1, s0
    rel = s0.body_q.numpy()[0, 3:]
    angle = 2.0 * np.arctan2(rel[1], rel[3])  # rotation about y
    err = (angle - (-3.0) + np.pi) % (2.0 * np.pi) - np.pi
    test.assertLess(abs(err), 0.05)


def _pendulum_response(device, gravity, torque, **solver_kw):
    """Angular acceleration of a 6 kg hinged box (COM 0.25 m from the pivot) over 20 steps of 2.5 ms from rest,
    divided by the analytic value; N semi-implicit steps from rest give q_N = a dt^2 N (N + 1) / 2 exactly."""
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, gravity))
    link = builder.add_link(xform=wp.transform((0.0, 0.0, 1.0), wp.quat_identity()), mass=1.0)
    builder.add_shape_box(link, xform=wp.transform((0.25, 0.0, 0.0), wp.quat_identity()), hx=0.25, hy=0.05, hz=0.05)
    joint = builder.add_joint_revolute(
        -1, link, parent_xform=wp.transform((0.0, 0.0, 1.0), wp.quat_identity()), axis=(0.0, 1.0, 0.0)
    )
    builder.add_articulation([joint])
    model = builder.finalize(device=device)
    solver = newton.solvers.SolverXPBD(model, **solver_kw)
    s0, s1, control = model.state(), model.state(), model.control()
    control.joint_f.assign(np.array([torque], dtype=np.float32))
    dt, n = 2.5e-3, 20
    for _ in range(n):
        s0.clear_forces()
        solver.step(s0, s1, control, None, dt)
        s0, s1 = s1, s0
    q = wp.zeros(1, dtype=float, device=device)
    qd = wp.zeros(1, dtype=float, device=device)
    newton.eval_ik(model, s0, q, qd)
    accel = 2.0 * float(q.numpy()[0]) / (dt * dt * n * (n + 1))
    inertia = _hinge_inertia(model)
    mass, r = float(model.body_mass.numpy()[0]), float(model.body_com.numpy()[0][0])
    analytic = torque / inertia if torque != 0.0 else mass * 9.81 * r / inertia
    return accel / analytic


def test_joint_relaxation_transmits_torque_and_gravity(test, device):
    """A pendulum responds to a joint torque and to gravity as the analytic hinge, at the default and at unequal
    relaxation factors (each row applies one consistent impulse)."""
    for kw in ({}, {"joint_linear_relaxation": 0.7, "joint_angular_relaxation": 0.4, "iterations": 8}):
        test.assertAlmostEqual(_pendulum_response(device, 0.0, 1.0, **kw), 1.0, delta=0.01)
        test.assertAlmostEqual(_pendulum_response(device, -9.81, 0.0, **kw), 1.0, delta=0.01)


def test_joint_legacy_relaxation_switch(test, device):
    """joint_legacy_relaxation restores the former scaling (moment of a positional impulse by the angular factor)."""
    kw = {"joint_linear_relaxation": 0.7, "joint_angular_relaxation": 0.4, "joint_legacy_relaxation": True}
    test.assertGreater(_pendulum_response(device, 0.0, 1.0, **kw), 1.3)
    test.assertLess(_pendulum_response(device, -9.81, 0.0, **kw), 0.85)


devices = get_test_devices()


class TestSolverXPBDJoints(unittest.TestCase):
    pass


add_function_test(
    TestSolverXPBDJoints,
    "test_revolute_limit_beyond_pi_does_not_gain_energy",
    test_revolute_limit_beyond_pi_does_not_gain_energy,
    devices=devices,
    check_output=False,
)
add_function_test(
    TestSolverXPBDJoints,
    "test_revolute_eval_ik_angle_beyond_pi",
    test_revolute_eval_ik_angle_beyond_pi,
    devices=devices,
    check_output=False,
)
add_function_test(
    TestSolverXPBDJoints,
    "test_revolute_hold_beyond_pi",
    test_revolute_hold_beyond_pi,
    devices=devices,
    check_output=False,
)
add_function_test(
    TestSolverXPBDJoints,
    "test_revolute_unlimited_drive_takes_short_way",
    test_revolute_unlimited_drive_takes_short_way,
    devices=devices,
    check_output=False,
)

add_function_test(
    TestSolverXPBDJoints,
    "test_joint_relaxation_transmits_torque_and_gravity",
    test_joint_relaxation_transmits_torque_and_gravity,
    devices=devices,
    check_output=False,
)
add_function_test(
    TestSolverXPBDJoints,
    "test_joint_legacy_relaxation_switch",
    test_joint_legacy_relaxation_switch,
    devices=devices,
    check_output=False,
)

if __name__ == "__main__":
    unittest.main(verbosity=2, failfast=True)
