# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Articulated-dynamics accuracy tests for SolverXPBD joints: hinge angles beyond +-pi, joint relaxation, drives,
effort limits and armature."""

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


def _pendulum_model(device, gravity=-9.81, ke=0.0, kd=0.0, target=0.0, effort=1e6, armature=0.0):
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, gravity))
    link = builder.add_link(xform=wp.transform((0.0, 0.0, 1.0), wp.quat_identity()), mass=1.0)
    builder.add_shape_box(link, xform=wp.transform((0.25, 0.0, 0.0), wp.quat_identity()), hx=0.25, hy=0.05, hz=0.05)
    joint = builder.add_joint_revolute(
        -1,
        link,
        parent_xform=wp.transform((0.0, 0.0, 1.0), wp.quat_identity()),
        axis=(0.0, 1.0, 0.0),
        target_ke=ke,
        target_kd=kd,
        target_pos=target,
        effort_limit=effort,
        armature=armature,
    )
    builder.add_articulation([joint])
    return builder.finalize(device=device)


def _run(model, solver, steps, dt, torque=0.0):
    s0, s1, control = model.state(), model.state(), model.control()
    newton.eval_fk(model, model.joint_q, model.joint_qd, s0)
    control.joint_f.assign(np.array([torque], dtype=np.float32))
    for _ in range(steps):
        s0.clear_forces()
        solver.step(s0, s1, control, None, dt)
        s0, s1 = s1, s0
    q = wp.zeros(1, dtype=float, device=model.device)
    qd = wp.zeros(1, dtype=float, device=model.device)
    newton.eval_ik(model, s0, q, qd)
    return float(q.numpy()[0]), float(qd.numpy()[0])


def test_joint_drive_stiffness_is_ke_at_any_iteration_count(test, device):
    """A pendulum held by a position drive settles at the sag tau_gravity / ke for 1, 4 and 16 iterations."""
    for mode, tol in (("pd", 0.01), ("implicit", 0.03)):
        for iterations in (1, 4, 16):
            if mode == "implicit" and iterations == 1:
                continue  # the implicit rows converge with the iterations
            model = _pendulum_model(device, ke=200.0, kd=5.0, target=0.5)
            solver = newton.solvers.SolverXPBD(model, iterations=iterations, joint_drive_mode=mode)
            q, _ = _run(model, solver, 800, 2.5e-3)
            tau_g = float(model.body_mass.numpy()[0]) * 9.81 * float(model.body_com.numpy()[0][0]) * np.cos(q)
            test.assertAlmostEqual(tau_g / (q - 0.5) / 200.0, 1.0, delta=tol, msg=f"{mode}, {iterations} iterations")


def test_joint_drive_compliance_mode_is_legacy(test, device):
    """joint_drive_mode="compliance" keeps the former drive, whose stiffness grows with the iteration count."""
    stiffness = []
    for iterations in (2, 8):
        model = _pendulum_model(device, ke=200.0, kd=5.0, target=0.5)
        solver = newton.solvers.SolverXPBD(
            model,
            iterations=iterations,
            joint_drive_mode="compliance",
            joint_linear_relaxation=0.4,
            joint_angular_relaxation=0.4,
        )
        q, _ = _run(model, solver, 800, 2.5e-3)
        tau_g = float(model.body_mass.numpy()[0]) * 9.81 * float(model.body_com.numpy()[0][0]) * np.cos(q)
        stiffness.append(tau_g / (q - 0.5))
    test.assertGreater(stiffness[1], 3.0 * stiffness[0])


def test_joint_drive_effort_limit(test, device):
    """The drive force is clamped at joint_effort_limit (zero gravity, target far away: q = f dt^2 N (N + 1) / 2 / I)."""
    dt, n = 2.5e-3, 20
    for mode in ("pd", "implicit"):
        model = _pendulum_model(device, gravity=0.0, ke=1000.0, target=1.0, effort=2.0)
        solver = newton.solvers.SolverXPBD(model, iterations=4, joint_drive_mode=mode)
        q, _ = _run(model, solver, n, dt)
        accel = 2.0 * q / (dt * dt * n * (n + 1))
        test.assertAlmostEqual(accel * _hinge_inertia(model) / 2.0, 1.0, delta=0.03, msg=mode)


def test_joint_armature_inertia(test, device):
    """joint_armature adds rotor inertia about the axis with "isotropic" or "axis"; "none" (default) ignores it."""
    dt, n, arm = 2.5e-3, 20, 0.2
    for mode, extra in (("isotropic", arm), ("axis", arm), ("none", 0.0)):  # default "none"
        model = _pendulum_model(device, gravity=0.0, armature=arm)
        solver = newton.solvers.SolverXPBD(model, iterations=4, joint_armature_inertia=mode)
        q, _ = _run(model, solver, n, dt, torque=1.0)
        accel = 2.0 * q / (dt * dt * n * (n + 1))
        test.assertAlmostEqual(accel * (_hinge_inertia(model) + extra), 1.0, delta=0.01, msg=mode)


def test_joint_drive_light_damped_chain(test, device):
    """Overdamped light chain (3 links of 8 g, ke 40, kd 10, armature 0.001) stepped by 0.3 rad in zero gravity:
    each joint creeps first-order, q(t) = 0.3 (1 - exp(-t ke / kd)) (0.0544 rad at 50 ms). A damper evaluated on the
    joint's own inertia would let such links move several times faster."""
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    parent, joints = -1, []
    for i in range(3):
        link = builder.add_link(xform=wp.transform((0.03 * i, 0.0, 1.0), wp.quat_identity()), mass=0.0)
        builder.add_shape_box(
            link, xform=wp.transform((0.015, 0.0, 0.0), wp.quat_identity()), hx=0.015, hy=0.008, hz=0.008
        )
        joints.append(
            builder.add_joint_revolute(
                parent,
                link,
                parent_xform=wp.transform((0.03 if i else 0.0, 0.0, 0.0 if i else 1.0), wp.quat_identity()),
                axis=(0.0, 1.0, 0.0),
                target_ke=40.0,
                target_kd=10.0,
                target_pos=0.3,
                armature=0.001,
            )
        )
        parent = link
    builder.add_articulation(joints)
    model = builder.finalize(device=device)
    # the chain's three drive rows couple through the shared links (Jacobi within a pass): 8 iterations and a drive
    # relaxation of 0.7 converge them to within 10 % (4 iterations: within 35 %)
    solver = newton.solvers.SolverXPBD(
        model, iterations=8, joint_drive_mode="pd", joint_armature_inertia="isotropic", joint_drive_relaxation=0.7
    )
    s0, s1, control = model.state(), model.state(), model.control()
    newton.eval_fk(model, model.joint_q, model.joint_qd, s0)
    for _ in range(40):
        solver.step(s0, s1, control, None, 1.25e-3)
        s0, s1 = s1, s0
    q = wp.zeros(3, dtype=float, device=device)
    qd = wp.zeros(3, dtype=float, device=device)
    newton.eval_ik(model, s0, q, qd)
    np.testing.assert_allclose(q.numpy(), 0.3 * (1.0 - np.exp(-0.05 * 40.0 / 10.0)), rtol=0.1)


def test_joint_drive_gains_set_after_construction(test, device):
    """Drive gains written to the model after the solver is created take effect; the drive torque is reported."""
    model = _pendulum_model(device, target=0.5)
    solver = newton.solvers.SolverXPBD(model, iterations=4, joint_drive_mode="pd")
    model.joint_target_ke.fill_(200.0)
    model.joint_target_kd.fill_(5.0)
    q, _ = _run(model, solver, 800, 2.5e-3)
    tau_g = float(model.body_mass.numpy()[0]) * 9.81 * float(model.body_com.numpy()[0][0]) * np.cos(q)
    test.assertAlmostEqual(tau_g / (q - 0.5) / 200.0, 1.0, delta=0.01)
    # the reported drive torque balances gravity at rest
    test.assertAlmostEqual(float(solver.joint_drive_force.numpy()[0]) / -tau_g, 1.0, delta=0.01)


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

add_function_test(
    TestSolverXPBDJoints,
    "test_joint_drive_stiffness_is_ke_at_any_iteration_count",
    test_joint_drive_stiffness_is_ke_at_any_iteration_count,
    devices=devices,
    check_output=False,
)
add_function_test(
    TestSolverXPBDJoints,
    "test_joint_drive_compliance_mode_is_legacy",
    test_joint_drive_compliance_mode_is_legacy,
    devices=devices,
    check_output=False,
)
add_function_test(
    TestSolverXPBDJoints,
    "test_joint_drive_effort_limit",
    test_joint_drive_effort_limit,
    devices=devices,
    check_output=False,
)
add_function_test(
    TestSolverXPBDJoints,
    "test_joint_armature_inertia",
    test_joint_armature_inertia,
    devices=devices,
    check_output=False,
)
add_function_test(
    TestSolverXPBDJoints,
    "test_joint_drive_light_damped_chain",
    test_joint_drive_light_damped_chain,
    devices=devices,
    check_output=False,
)
add_function_test(
    TestSolverXPBDJoints,
    "test_joint_drive_gains_set_after_construction",
    test_joint_drive_gains_set_after_construction,
    devices=devices,
    check_output=False,
)


if __name__ == "__main__":
    unittest.main(verbosity=2, failfast=True)
