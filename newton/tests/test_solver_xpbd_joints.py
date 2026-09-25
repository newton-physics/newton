# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""SolverXPBD joint tests: joint relaxation."""

import unittest

import numpy as np
import warp as wp

import newton
from newton.tests.unittest_utils import add_function_test, get_test_devices


def _hinge_inertia(model):
    return (
        float(model.body_inertia.numpy()[0][1, 1])
        + float(model.body_mass.numpy()[0]) * float(model.body_com.numpy()[0][0]) ** 2
    )


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
