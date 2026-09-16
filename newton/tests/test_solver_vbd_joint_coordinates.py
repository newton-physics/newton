# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import math
import unittest

import numpy as np
import warp as wp

import newton
from newton._src.solvers.vbd.joint_coordinates import JointCoordinateData, JointCoordinates, evaluate_coordinate
from newton.tests.unittest_utils import add_function_test, get_test_devices


@wp.kernel
def _sample_coordinates(
    data: JointCoordinateData,
    poses: wp.array[wp.transform],
    coordinates: wp.array[float],
    gradients: wp.array[wp.vec3],
):
    component = wp.tid()
    q, _g_p, g_c = evaluate_coordinate(data, 0, component, poses)
    coordinates[component] = q
    gradients[component] = wp.spatial_bottom(g_c)


@wp.kernel
def _advance(q: wp.array[float], axis: int, increment: float):
    q[axis] += increment


def _model(device, *, axes=1, d6=False, initial=None, kinematic=False):
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    body = builder.add_link(mass=1.0, inertia=wp.mat33(np.eye(3)), is_kinematic=kinematic)
    config = newton.ModelBuilder.JointDofConfig
    angular = [
        config(axis=axis, target_ke=0.0, target_kd=0.0, limit_ke=0.0, limit_kd=0.0)
        for axis in (newton.Axis.X, newton.Axis.Y, newton.Axis.Z)[:axes]
    ]
    kind = newton.JointType.D6 if d6 else newton.JointType.REVOLUTE
    joint = builder.add_joint(kind, -1, body, angular_axes=angular)
    builder.add_articulation([joint])
    if initial is not None:
        builder.joint_q[:] = initial
    builder.color()
    model = builder.finalize(device=device)
    newton.eval_fk(model, model.joint_q, model.joint_qd, model)
    return model


def test_vbd_continuous_joint_output(test, device):
    """Publish continuous non-mimic coordinates on every revolute and D6 angular axis."""
    for d6, axes in ((False, 1), (True, 1), (True, 2), (True, 3)):
        for axis in range(axes):
            with test.subTest(d6=d6, axes=axes, axis=axis):
                initial = np.full(axes, 0.2)
                initial[axis] += 4.0 * math.pi
                model = _model(device, axes=axes, d6=d6, initial=initial, kinematic=True)
                state, next_state = model.state(), model.state()
                solver = newton.solvers.SolverVBD(model, iterations=1, rigid_compliant_alm=True)
                commanded = wp.clone(model.joint_q)
                for direction in (1.0, -1.0):
                    for step in range(140):
                        wp.launch(_advance, dim=1, inputs=[commanded, axis, direction * 0.1], device=device)
                        newton.eval_fk(model, commanded, model.joint_qd, state)
                        solver.step(state, next_state, None, None, 1.0 / 240.0)
                        state, next_state = next_state, state
                        np.testing.assert_allclose(state.joint_q.numpy(), commanded.numpy(), atol=1.0e-4)
                        # Euler velocity reconstruction is ill-conditioned near gimbal lock.
                        well_conditioned = axes < 3 or abs(math.cos(float(commanded.numpy()[1]))) > 0.1
                        if step > 0 and well_conditioned:
                            expected_rate = np.zeros(axes)
                            expected_rate[axis] = direction * 24.0
                            np.testing.assert_allclose(state.joint_qd.numpy(), expected_rate, atol=0.02)


def test_vbd_multiturn_drive_limit(test, device):
    """Drive ordinary joints over multiple turns and enforce limits beyond a revolution."""
    for compliant in (True, False):
        for d6 in (False, True):
            for limited in (False, True):
                with test.subTest(compliant=compliant, d6=d6, limited=limited):
                    model = _model(device, d6=d6)
                    model.joint_target_ke.fill_(100.0)
                    model.joint_target_kd.fill_(20.0)
                    target, upper = 5.0 * math.pi, 2.5 * math.pi
                    if limited:
                        model.joint_limit_lower.fill_(-100.0)
                        model.joint_limit_upper.fill_(upper)
                        model.joint_limit_ke.fill_(10000.0)
                        model.joint_limit_kd.fill_(100.0)
                    control = model.control()
                    control.joint_target_q.fill_(target)
                    state, next_state = model.state(), model.state()
                    solver = newton.solvers.SolverVBD(model, iterations=16, rigid_compliant_alm=compliant)
                    for _ in range(480):
                        solver.step(state, next_state, control, None, 1.0 / 120.0)
                        state, next_state = next_state, state
                    # VBD clamps position-drive targets to the authored joint limits.
                    expected = upper if limited else target
                    test.assertAlmostEqual(float(state.joint_q.numpy()[0]), expected, delta=0.02)


def test_vbd_joint_coordinate_graph_reset(test, device):
    """Track a non-mimic joint through graph replay, disabling, and a captured reset."""
    model = _model(device, initial=[4.0 * math.pi + 0.2], kinematic=True)
    state, next_state = model.state(), model.state()
    solver = newton.solvers.SolverVBD(model, iterations=1, rigid_compliant_alm=True)
    commanded = wp.clone(model.joint_q)

    def step():
        nonlocal state, next_state
        wp.launch(_advance, dim=1, inputs=[commanded, 0, 0.1], device=device)
        newton.eval_fk(model, commanded, model.joint_qd, state)
        solver.step(state, next_state, None, None, 1.0 / 240.0)
        state, next_state = next_state, state

    def reset():
        solver.reset(state)
        wp.copy(commanded, model.joint_q)

    step()
    reset()
    step_graph, reset_graph = None, None
    if device.is_cuda:
        with wp.ScopedCapture(device=device) as capture:
            step()
            step()
        step_graph = capture.graph
        with wp.ScopedCapture(device=device) as capture:
            reset()
        reset_graph = capture.graph
    for enabled in (False, True):
        model.joint_enabled.fill_(enabled)
        # The coordinate history is not rebuilt by joint-property notifications.
        solver.notify_model_changed(newton.ModelFlags.JOINT_PROPERTIES)
        for _ in range(100):
            if step_graph is None:
                step()
                step()
            else:
                wp.capture_launch(step_graph)
        np.testing.assert_allclose(state.joint_q.numpy(), commanded.numpy(), atol=1.0e-4)
        if reset_graph is None:
            reset()
            step()
            step()
        else:
            wp.capture_launch(reset_graph)
            wp.capture_launch(step_graph)
        np.testing.assert_allclose(state.joint_q.numpy(), commanded.numpy(), atol=1.0e-4)


def test_vbd_multiturn_passive_limits(test, device):
    """Stop force-driven rotation at upper and lower bounds beyond a full turn."""
    for direction in (-1.0, 1.0):
        model = _model(device)
        bound = 2.5 * math.pi
        model.joint_qd.fill_(direction * 10.0)
        newton.eval_fk(model, model.joint_q, model.joint_qd, model)
        model.joint_limit_lower.fill_(-bound if direction < 0.0 else -100.0)
        model.joint_limit_upper.fill_(bound if direction > 0.0 else 100.0)
        model.joint_limit_ke.fill_(10000.0)
        model.joint_limit_kd.fill_(100.0)
        state, next_state = model.state(), model.state()
        control = model.control()
        control.joint_f.fill_(direction)
        solver = newton.solvers.SolverVBD(model, iterations=12, rigid_compliant_alm=True)
        for _ in range(360):
            solver.step(state, next_state, control, None, 1.0 / 120.0)
            state, next_state = next_state, state
        test.assertAlmostEqual(float(state.joint_q.numpy()[0]), direction * (bound + 0.0001), delta=0.005)


def test_vbd_joint_coordinate_reset_flags(test, device):
    """Reset scalar-coordinate outputs and tolerate omitted coordinate arrays."""
    model = _model(device, initial=[4.0 * math.pi + 0.2], kinematic=True)
    state, next_state = model.state(), model.state()
    solver = newton.solvers.SolverVBD(model, iterations=1, rigid_compliant_alm=True)
    state.joint_q.fill_(1.0)
    state.joint_qd.fill_(2.0)
    solver.reset(state, flags=newton.StateFlags.JOINT_Q | newton.StateFlags.JOINT_QD)
    np.testing.assert_array_equal(state.joint_q.numpy(), model.joint_q.numpy())
    np.testing.assert_array_equal(state.joint_qd.numpy(), model.joint_qd.numpy())
    state.joint_q = state.joint_qd = None
    next_state.joint_q = next_state.joint_qd = None
    solver.reset(state)
    solver.step(state, next_state, None, None, 1.0 / 240.0)
    test.assertIsNone(next_state.joint_q)


def test_vbd_multiaxis_d6_drive(test, device):
    """Drive each axis of two- and three-axis D6 joints beyond two revolutions."""
    for axes in (2, 3):
        for axis in range(axes):
            with test.subTest(axes=axes, axis=axis):
                model = _model(device, axes=axes, d6=True)
                model.joint_target_ke.fill_(100.0)
                model.joint_target_kd.fill_(20.0)
                target = np.zeros(axes)
                target[axis] = 4.0 * math.pi + 0.3
                control = model.control()
                control.joint_target_q.assign(target)
                state, next_state = model.state(), model.state()
                solver = newton.solvers.SolverVBD(model, iterations=16, rigid_compliant_alm=True)

                def advance(solver=solver, control=control):
                    nonlocal state, next_state
                    for _ in range(2):
                        solver.step(state, next_state, control, None, 1.0 / 120.0)
                        state, next_state = next_state, state

                advance()
                graph = None
                if device.is_cuda:
                    with wp.ScopedCapture(device=device) as capture:
                        advance()
                    graph = capture.graph
                for _ in range(180):
                    if graph is None:
                        advance()
                    else:
                        wp.capture_launch(graph)
                np.testing.assert_allclose(state.joint_q.numpy(), target, atol=0.005)


def test_vbd_two_axis_coordinate_gradients(test, device):
    """Keep two-axis D6 coordinates regular through a right angle and match their gradients."""
    for pitch in (0.5 * math.pi, 2.0):
        model = _model(device, axes=2, d6=True, initial=[4.0 * math.pi + 0.4, pitch])
        data = JointCoordinates(model).data
        state = model.state()
        pose = state.body_q.numpy()[0]
        rotation = wp.quat(*pose[3:])
        coordinates = wp.empty(2, dtype=float, device=device)
        gradients = wp.empty(2, dtype=wp.vec3, device=device)

        def sample(data=data, state=state, coordinates=coordinates, gradients=gradients):
            wp.launch(
                _sample_coordinates, dim=2, inputs=[data, state.body_q], outputs=[coordinates, gradients], device=device
            )
            return coordinates.numpy().copy(), gradients.numpy().copy()

        q, analytic = sample()
        np.testing.assert_allclose(q, model.joint_q.numpy(), atol=2.0e-6)
        epsilon = 0.001
        for axis in range(3):
            direction = wp.vec3()
            direction[axis] = 1.0
            samples = []
            for sign in (-1.0, 1.0):
                perturbed = wp.quat_from_axis_angle(direction, sign * epsilon) * rotation
                state.body_q.assign([wp.transform(wp.vec3(), perturbed)])
                samples.append(sample()[0])
            np.testing.assert_allclose((samples[1] - samples[0]) / (2.0 * epsilon), analytic[:, axis], atol=0.0006)


class TestSolverVBDJointCoordinates(unittest.TestCase):
    pass


for test_function in (
    test_vbd_continuous_joint_output,
    test_vbd_multiturn_drive_limit,
    test_vbd_joint_coordinate_graph_reset,
    test_vbd_multiturn_passive_limits,
    test_vbd_joint_coordinate_reset_flags,
    test_vbd_multiaxis_d6_drive,
    test_vbd_two_axis_coordinate_gradients,
):
    add_function_test(TestSolverVBDJointCoordinates, test_function.__name__, test_function, devices=get_test_devices())


if __name__ == "__main__":
    unittest.main(verbosity=2)
