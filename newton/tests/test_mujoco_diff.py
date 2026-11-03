# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Exercise Newton's state, control, and parameter gradients through MuJoCo Warp."""

import unittest

import numpy as np
import warp as wp

import newton
from newton.solvers import SolverMuJoCo
from newton.tests.unittest_utils import add_function_test, get_test_devices


@wp.kernel
def trajectory_loss(q: wp.array[float], qd: wp.array[float], weight: float, loss: wp.array[float]):
    i = wp.tid()
    error = q[i] - 0.7
    wp.atomic_add(loss, 0, weight * (error * error + 0.1 * qd[i] * qd[i]))


def make_slider(device, *, requires_grad=True, world_count=2):
    robot = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    body = robot.add_link(mass=2.0, inertia=wp.mat33(0.2, 0.0, 0.0, 0.0, 0.3, 0.0, 0.0, 0.0, 0.4))
    joint = robot.add_joint_prismatic(
        -1,
        body,
        target_ke=12.0,
        target_kd=1.0,
        damping=0.4,
        armature=0.3,
        actuator_mode=newton.JointTargetMode.POSITION,
    )
    robot.add_articulation([joint])
    robot.joint_q[0] = 0.15
    robot.joint_qd[0] = 0.2
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    builder.replicate(robot, world_count)
    return builder.finalize(device=device, requires_grad=requires_grad)


def test_rollout_gradients(test, device):
    """Compare initial-state, per-step control, and system-ID gradients with finite differences."""
    model = make_slider(device)
    solver = SolverMuJoCo(model, kinematic_switchable_roots=False, njmax=4, nconmax=1)
    states = [model.state() for _ in range(5)]
    controls = [model.control() for _ in range(4)]
    for i, control in enumerate(controls):
        control.joint_target_q.assign(np.array([0.4 + 0.1 * i, -0.2 + 0.05 * i], dtype=np.float32))
        control.joint_f.assign(np.array([0.2, -0.1], dtype=np.float32))
    loss = wp.zeros(1, device=device, requires_grad=True)

    def forward():
        loss.zero_()
        for i, control in enumerate(controls):
            solver.step(states[i], states[i + 1], control, None, 0.02 + 0.002 * i)
            wp.launch(
                trajectory_loss,
                model.joint_coord_count,
                inputs=[states[i + 1].joint_q, states[i + 1].joint_qd, float(i + 1), loss],
                device=device,
            )

    parameters = {
        "initial position": states[0].joint_q,
        "initial velocity": states[0].joint_qd,
        "early target": controls[0].joint_target_q,
        "late target": controls[-1].joint_target_q,
        "joint force": controls[1].joint_f,
        "mass": model.body_mass,
        "armature": model.joint_armature,
        "damping": model.joint_damping,
    }
    tape = wp.Tape()
    with tape:
        forward()
    tape.backward(loss)
    analytic = {name: param.grad.numpy().copy() for name, param in parameters.items()}
    reference_loss = loss.numpy().copy()
    tape.zero()

    for name, param in parameters.items():
        original = param.numpy().copy()
        for index in range(param.size):
            with test.subTest(parameter=name, world=index):
                eps = 2.0e-3
                plus = original.copy()
                plus[index] += eps
                param.assign(plus)
                forward()
                high = float(loss.numpy()[0])
                minus = original.copy()
                minus[index] -= eps
                param.assign(minus)
                forward()
                low = float(loss.numpy()[0])
                param.assign(original)
                numeric = (high - low) / (2.0 * eps)
                test.assertGreater(abs(numeric), 1.0e-5)
                np.testing.assert_allclose(analytic[name][index], numeric, atol=2.5e-4, rtol=1.0e-2)

    # Reuse the same states and controls after Tape.zero(), as an optimizer does.
    tape = wp.Tape()
    with tape:
        forward()
    tape.backward(loss)
    np.testing.assert_allclose(loss.numpy(), reference_loss, atol=1.0e-6)
    for name, param in parameters.items():
        np.testing.assert_allclose(param.grad.numpy(), analytic[name], atol=1.0e-6)

    if device.is_cuda:
        tape.zero()
        with wp.ScopedCapture(device=device) as capture:
            graph_tape = wp.Tape()
            with graph_tape:
                forward()
            graph_tape.backward(loss)
        for _ in range(2):
            graph_tape.zero()
            wp.capture_launch(capture.graph)
            np.testing.assert_allclose(loss.numpy(), reference_loss, atol=1.0e-6)
            for name, param in parameters.items():
                np.testing.assert_allclose(param.grad.numpy(), analytic[name], atol=1.0e-6)


@wp.kernel
def body_loss(q: wp.array[wp.transform], qd: wp.array[wp.spatial_vector], loss: wp.array[float]):
    p = wp.transform_get_translation(q[0])
    v = qd[0]
    loss[0] = wp.dot(p, wp.vec3(0.3, 0.2, 1.0)) + wp.dot(v, wp.spatial_vector(0.1, 0.2, 0.3, 0.2, 0.1, 0.4))


def test_contact_gradients(test, device):
    """Differentiate free-joint coordinates and applied wrenches through native contact and Newton FK."""
    builder = newton.ModelBuilder()
    body = builder.add_link(xform=wp.transform(wp.vec3(0.0, 0.0, 0.095)), com=wp.vec3(0.02, 0.01, 0.0))
    builder.add_shape_sphere(body, radius=0.1)
    joint = builder.add_joint_free(body)
    builder.add_articulation([joint])
    builder.add_ground_plane()
    builder.joint_qd[:] = [0.1, 0.2, -0.4, 0.2, 0.1, 0.3]
    model = builder.finalize(device=device, requires_grad=True)
    solver = SolverMuJoCo(model, integrator="euler", njmax=16, nconmax=4, iterations=30)
    states = [model.state() for _ in range(4)]
    control = model.control()
    loss = wp.zeros(1, device=device, requires_grad=True)

    def forward():
        for i in range(3):
            solver.step(states[i], states[i + 1], control, None, 0.004)
        wp.launch(body_loss, 1, inputs=[states[-1].body_q, states[-1].body_qd, loss], device=device)

    parameters = [(states[0].joint_qd, (0, 2, 4)), (states[1].body_f, (0, 5)), (control.joint_f, (2, 4))]
    tape = wp.Tape()
    with tape:
        forward()
    tape.backward(loss)
    test.assertGreater(int(solver.mjw_data.nacon.numpy()[0]), 0)
    for param, indices in parameters:
        analytic = param.grad.numpy().ravel().copy()
        original = param.numpy().copy()
        for index in indices:
            with test.subTest(shape=param.shape, index=index):
                eps = 1.0e-3
                values = original.copy()
                values.flat[index] += eps
                param.assign(values)
                forward()
                high = float(loss.numpy()[0])
                values.flat[index] -= 2.0 * eps
                param.assign(values)
                forward()
                low = float(loss.numpy()[0])
                param.assign(original)
                np.testing.assert_allclose(analytic[index], (high - low) / (2.0 * eps), atol=5.0e-4, rtol=2.0e-2)


def test_configuration(test, device):
    """Reject unsupported differentiation modes before silently returning incomplete gradients."""
    model = make_slider(device, world_count=1)
    for options, message in (
        ({"use_mujoco_cpu": True}, "native MuJoCo contacts"),
        ({"use_mujoco_contacts": False}, "native MuJoCo contacts"),
        ({"solver": "cg"}, "solver='newton'"),
        ({"integrator": "rk4"}, "Euler or implicitfast"),
    ):
        with test.subTest(options=options), test.assertRaisesRegex(ValueError, message):
            SolverMuJoCo(model, **options)
    solver = SolverMuJoCo(model)
    state = model.state()
    control = model.control()
    with test.assertRaisesRegex(ValueError, "distinct input and output"):
        solver.step(state, state, control, None, 0.01)
    with test.assertRaisesRegex(ValueError, "requires_grad=True"):
        solver.step(state, model.state(requires_grad=False), control, None, 0.01)
    observables = solver.observables({solver.ObservableKind.OVERFLOW})
    with test.assertRaisesRegex(ValueError, "state outputs only"):
        solver.step(state, model.state(), control, None, 0.01, observables=observables)


class TestMuJoCoDifferentiability(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Require the actual optional MuJoCo Warp adjoint implementation."""
        try:
            import mujoco_warp
        except ImportError as error:
            raise unittest.SkipTest("MuJoCo Warp is not installed") from error
        if not hasattr(mujoco_warp, "enable_grad"):
            raise unittest.SkipTest("MuJoCo Warp with the adjoint API is required")


add_function_test(
    TestMuJoCoDifferentiability, "test_rollout_gradients", test_rollout_gradients, devices=get_test_devices()
)
add_function_test(
    TestMuJoCoDifferentiability, "test_contact_gradients", test_contact_gradients, devices=get_test_devices()
)
add_function_test(TestMuJoCoDifferentiability, "test_configuration", test_configuration, devices=get_test_devices())


if __name__ == "__main__":
    unittest.main()
