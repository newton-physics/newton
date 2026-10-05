# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Free rigid bodies must not gain energy from their gyroscopic bias."""

import unittest

import numpy as np
import warp as wp

import newton
from newton.tests.unittest_utils import add_function_test, get_test_devices


def _body_omega(state, body=0):
    q = wp.quat(*state.body_q.numpy()[body, 3:])
    omega = wp.vec3(*state.body_qd.numpy()[body, 3:])
    return np.asarray(wp.quat_rotate_inv(q, omega), dtype=np.float64)


def _model(device, inertia, omega, *, armature=0.0, mass=1.0):
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    builder.add_body(mass=mass, inertia=wp.mat33(inertia), com=wp.vec3(0.03, -0.02, 0.01))
    model = builder.finalize(device=device)
    model.joint_armature.fill_(armature)
    state, output = model.state(), model.state()
    state.joint_qd.assign(np.array([0.1, 0.2, 0.3, *omega], dtype=np.float32))
    newton.eval_fk(model, state.joint_q, state.joint_qd, state)
    return model, state, output


def test_free_spin_energy(test, device):
    inertia = np.diag([0.001, 0.02, 0.0205])
    for mode in ("split", "matrix_free"):
        if mode == "matrix_free" and not wp.get_device(device).is_cuda:
            continue
        model, state, output = _model(device, inertia, (100.0, 20.0, 50.0))
        solver = newton.solvers.SolverFeatherPGS(model, pgs_mode=mode, angular_damping=0.0)
        control = model.control()
        omega = _body_omega(state)
        energy = 0.5 * omega @ inertia @ omega
        momentum = np.linalg.norm(inertia @ omega)
        with test.subTest(mode=mode):
            for step in range(1000):
                solver.step(state, output, control, None, 1.0 / 240.0)
                state, output = output, state
                omega = _body_omega(state)
                observed = 0.5 * omega @ inertia @ omega
                test.assertTrue(np.isfinite(observed), f"nonfinite energy at step {step}")
                test.assertAlmostEqual(observed / energy, 1.0, delta=1.0e-3, msg=f"step {step}")
            test.assertAlmostEqual(np.linalg.norm(inertia @ omega) / momentum, 1.0, delta=1.0e-3)
            np.testing.assert_allclose(state.body_qd.numpy()[0, :3], [0.1, 0.2, 0.3], atol=2.0e-4)


def test_principal_axis_torque(test, device):
    inertia = np.diag([0.04, 0.02, 0.05])
    for armature in (0.0, 0.03):
        model, state, output = _model(device, inertia, (0.0, 0.0, 2.0), armature=armature)
        solver = newton.solvers.SolverFeatherPGS(model, angular_damping=0.0)
        control = model.control()
        torque = 0.1
        dt = 1.0 / 240.0
        for _ in range(120):
            state.body_f.assign(np.array([[0.0, 0.0, 0.0, 0.0, 0.0, torque]], dtype=np.float32))
            solver.step(state, output, control, None, dt)
            state, output = output, state
        expected = 2.0 + 120 * dt * torque / (inertia[2, 2] + armature)
        test.assertAlmostEqual(float(state.body_qd.numpy()[0, 5]), expected, delta=1.0e-4)


def test_rotated_inertia_and_armature(test, device):
    rotation = wp.quat_from_axis_angle(wp.normalize(wp.vec3(1.0, 2.0, 3.0)), 0.7)
    basis = np.asarray(wp.quat_to_matrix(rotation), dtype=np.float64).reshape(3, 3)
    local_basis = np.asarray(wp.quat_to_matrix(wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), 0.4))).reshape(3, 3)
    for scale, armature in ((1.0e-5, 0.0), (1.0, 0.003), (1.0e5, 0.0)):
        inertia = scale * local_basis @ np.diag([0.001, 0.02, 0.0205]) @ local_basis.T
        omega = basis @ local_basis @ np.array([20.0, 4.0, 10.0])
        model, state, output = _model(device, inertia, omega, mass=scale)
        rotor = armature * np.array([1.0, 2.0, 3.0])
        model.joint_armature.assign(np.array([0.0, 0.0, 0.0, *rotor], dtype=np.float32))
        q = state.joint_q.numpy()
        q[3:7] = np.asarray(rotation)
        state.joint_q.assign(q)
        newton.eval_fk(model, state.joint_q, state.joint_qd, state)
        solver = newton.solvers.SolverFeatherPGS(model, angular_damping=0.0)
        control = model.control()
        w = _body_omega(state)
        initial = w @ inertia @ w + np.dot(rotor, omega * omega)
        for _ in range(200):
            solver.step(state, output, control, None, 1.0 / 240.0)
            state, output = output, state
        w = _body_omega(state)
        omega = state.body_qd.numpy()[0, 3:].astype(np.float64)
        final = w @ inertia @ w + np.dot(rotor, omega * omega)
        test.assertAlmostEqual(final / initial, 1.0, delta=1.0e-3, msg=f"scale={scale}, armature={armature}")


def test_gyro_precession_converges(test, device):
    inertia = np.diag([0.001, 0.02, 0.0205])
    inverse = np.linalg.inv(inertia)
    initial = np.array([20.0, 4.0, 10.0])
    duration = 0.2

    def derivative(omega):
        return inverse @ np.cross(inertia @ omega, omega)

    # Independent double-precision Euler-equation reference in the body frame.
    reference = initial.copy()
    h = duration / 4096
    for _ in range(4096):
        k1 = derivative(reference)
        k2 = derivative(reference + 0.5 * h * k1)
        k3 = derivative(reference + 0.5 * h * k2)
        k4 = derivative(reference + h * k3)
        reference += h * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0

    errors = []
    for steps in (48, 96):
        model, state, output = _model(device, inertia, initial)
        solver = newton.solvers.SolverFeatherPGS(model, angular_damping=0.0)
        control = model.control()
        for _ in range(steps):
            solver.step(state, output, control, None, duration / steps)
            state, output = output, state
        errors.append(np.linalg.norm(_body_omega(state) - reference))
    test.assertLess(errors[0], 0.1)
    test.assertLess(errors[1], 0.35 * errors[0])


def _mixed_scene(device, *, arm, totes):
    """Build two worlds with free bodies on both sides of a 19-DOF articulation."""
    template = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    if totes:
        template.add_body(
            xform=wp.transform(wp.vec3(-0.5, 0.0, 0.0), wp.quat_identity()),
            mass=1.0,
            inertia=wp.mat33(np.diag([0.001, 0.02, 0.0205])),
            com=wp.vec3(0.03, -0.02, 0.01),
        )
    if arm:
        joints = []
        parent = -1
        for _ in range(19):
            body = template.add_link(mass=0.5, inertia=wp.mat33(np.eye(3) * 0.02))
            joints.append(
                template.add_joint_revolute(
                    parent,
                    body,
                    parent_xform=wp.transform(wp.vec3(0.1, 0.0, 0.0), wp.quat_identity()),
                    axis=newton.Axis.Z,
                    target_ke=40.0,
                    target_kd=4.0,
                    armature=0.01,
                )
            )
            parent = body
        template.add_articulation(joints)
    if totes:
        template.add_body(
            xform=wp.transform(wp.vec3(0.5, 0.0, 0.0), wp.quat_identity()),
            mass=2.0,
            inertia=wp.mat33(np.diag([0.04, 0.002, 0.041])),
            com=wp.vec3(-0.02, 0.01, 0.04),
        )
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    builder.replicate(template, 2, spacing=(3.0, 0.0, 0.0))
    model = builder.finalize(device=device)
    state, output = model.state(), model.state()
    velocity = state.joint_qd.numpy()
    free_joints = np.flatnonzero(model.joint_type.numpy() == newton.JointType.FREE)
    for i, joint in enumerate(free_joints):
        start = int(model.joint_qd_start.numpy()[joint])
        velocity[start + 3 : start + 6] = (100.0, 20.0, 50.0) if i % 2 == 0 else (20.0, 100.0, 50.0)
    state.joint_qd.assign(velocity)
    newton.eval_fk(model, state.joint_q, state.joint_qd, state)
    control = model.control()
    control.joint_target_q.fill_(0.1)
    solver = newton.solvers.SolverFeatherPGS(
        model, pgs_mode="matrix_free" if wp.get_device(device).is_cuda else "split", angular_damping=0.0
    )
    return model, solver, control, [state, output]


def test_mixed_arm_and_spinning_bodies(test, device):
    """Preserve isolated arm and rotor dynamics with mixed response sizes and graph replay."""
    executions = ("eager", "graph") if wp.get_device(device).is_cuda else ("eager",)
    for execution in executions:
        with test.subTest(execution=execution):
            scenes = [
                _mixed_scene(device, arm=True, totes=True),
                _mixed_scene(device, arm=True, totes=False),
                _mixed_scene(device, arm=False, totes=True),
            ]
            model, solver, _, states = scenes[0]
            free_bodies = (0, 20, 21, 41)
            inertias = model.body_inertia.numpy().astype(np.float64)

            def energies(state, free_bodies=free_bodies, inertias=inertias):
                """Measure each free body's rotational energy in its inertia frame."""
                values = []
                for body in free_bodies:
                    omega = _body_omega(state, body)
                    values.append(omega @ inertias[body] @ omega)
                return np.asarray(values)

            initial_energy = energies(states[0])
            test.assertIn(6, solver.L_by_size)
            test.assertIn(19, solver.L_by_size)

            def step(scenes=scenes):
                """Advance the mixed scene and its independent controls together."""
                for _, scene_solver, control, pair in scenes:
                    scene_solver.step(pair[0], pair[1], control, None, 1.0 / 240.0)
                    pair[0], pair[1] = pair[1], pair[0]

            # Compile and initialize the same layouts before capture.
            step()
            graph = None
            if execution == "graph":
                with wp.ScopedCapture(device=device) as capture:
                    for _, scene_solver, _, _ in scenes:
                        scene_solver.seed_double_buffer_events()
                    step()
                    step()
                graph = capture.graph
            for _ in range(100):
                if graph is None:
                    step()
                    step()
                else:
                    wp.capture_launch(graph)
                current = scenes[0][3][0]
                np.testing.assert_allclose(energies(current) / initial_energy, 1.0, atol=1.0e-3, rtol=0.0)
                for field in ("body_q", "body_qd"):
                    width = 7 if field == "body_q" else 6
                    mixed = getattr(current, field).numpy().reshape(2, 21, width)
                    arm = getattr(scenes[1][3][0], field).numpy().reshape(2, 19, width)
                    totes = getattr(scenes[2][3][0], field).numpy().reshape(2, 2, width)
                    np.testing.assert_allclose(mixed[:, 1:20], arm, atol=3.0e-5, rtol=1.0e-6)
                    np.testing.assert_allclose(mixed[:, [0, 20]], totes, atol=3.0e-5, rtol=1.0e-6)
            test.assertGreater(float(np.abs(scenes[1][3][0].joint_q.numpy()).max()), 0.01)


class TestFeatherPGSGyroscopic(unittest.TestCase):
    pass


for _device in get_test_devices():
    add_function_test(
        TestFeatherPGSGyroscopic,
        "test_mixed_arm_and_spinning_bodies",
        test_mixed_arm_and_spinning_bodies,
        devices=[_device],
    )
    add_function_test(TestFeatherPGSGyroscopic, "test_free_spin_energy", test_free_spin_energy, devices=[_device])
    add_function_test(
        TestFeatherPGSGyroscopic, "test_principal_axis_torque", test_principal_axis_torque, devices=[_device]
    )
    add_function_test(
        TestFeatherPGSGyroscopic,
        "test_rotated_inertia_and_armature",
        test_rotated_inertia_and_armature,
        devices=[_device],
    )
    add_function_test(
        TestFeatherPGSGyroscopic, "test_gyro_precession_converges", test_gyro_precession_converges, devices=[_device]
    )


if __name__ == "__main__":
    unittest.main()
