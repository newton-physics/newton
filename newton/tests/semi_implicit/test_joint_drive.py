# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import unittest

import warp as wp

import newton
from newton.solvers import SolverSemiImplicit
from newton.tests.unittest_utils import add_function_test, get_test_devices


def test_semi_implicit_revolute_winding(test, device):
    """Preserve turns across steps and state resets without wrapping drive errors."""
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    body = builder.add_link(mass=1.0, inertia=wp.mat33(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0))
    joint = builder.add_joint_revolute(-1, body, axis=newton.Axis.Z, target_ke=2.0, armature=0.0)
    builder.add_articulation([joint])
    model = builder.finalize(device=device)
    solver = SolverSemiImplicit(model, angular_damping=0.0)
    control = model.control()
    for initial in (7.0, -7.0):
        with test.subTest(initial=initial):
            state, output = model.state(), model.state()
            state.joint_q.fill_(initial)
            newton.eval_fk(model, state.joint_q, state.joint_qd, state)
            control.joint_target_q.fill_(initial)
            for _ in range(3):
                state.clear_forces()
                solver.step(state, output, control, None, 0.001)
                state, output = output, state
                test.assertAlmostEqual(float(state.joint_q.numpy()[0]), initial, delta=1.0e-5)
                test.assertAlmostEqual(float(state.joint_qd.numpy()[0]), 0.0, delta=1.0e-5)
            control.joint_target_q.zero_()
            state.clear_forces()
            solver.step(state, output, control, None, 0.001)
            test.assertAlmostEqual(float(output.joint_qd.numpy()[0]), -2.0 * initial * 0.001, delta=1.0e-5)

    model.joint_target_ke.zero_()
    for sign in (-1.0, 1.0):
        state, output = model.state(), model.state()
        initial = sign * 6.27
        state.joint_q.fill_(initial)
        state.joint_qd.fill_(sign * 2.0)
        newton.eval_fk(model, state.joint_q, state.joint_qd, state)
        for step in range(20):
            state.clear_forces()
            solver.step(state, output, control, None, 0.01)
            state, output = output, state
            test.assertAlmostEqual(float(state.joint_q.numpy()[0]), initial + sign * 0.02 * (step + 1), delta=3.0e-5)
            test.assertAlmostEqual(float(state.joint_qd.numpy()[0]), sign * 2.0, delta=1.0e-5)

    # Capture must replay the state transition, including its winding update.
    if device.is_cuda:
        state, output = model.state(), model.state()
        state.joint_q.fill_(6.27)
        state.joint_qd.fill_(2.0)
        newton.eval_fk(model, state.joint_q, state.joint_qd, state)
        with wp.ScopedCapture(device=device) as capture:
            state.clear_forces()
            solver.step(state, output, control, None, 0.01)
            output.clear_forces()
            solver.step(output, state, control, None, 0.01)
        for _ in range(10):
            wp.capture_launch(capture.graph)
        test.assertAlmostEqual(float(state.joint_q.numpy()[0]), 6.67, delta=3.0e-5)

    # A finite limit beyond one turn must use the same continuous coordinate.
    model.joint_limit_upper.fill_(6.5)
    model.joint_limit_ke.fill_(2.0)
    model.joint_limit_kd.zero_()
    state, output = model.state(), model.state()
    state.joint_q.fill_(7.0)
    newton.eval_fk(model, state.joint_q, state.joint_qd, state)
    state.clear_forces()
    solver.step(state, output, control, None, 0.001)
    test.assertAlmostEqual(float(output.joint_qd.numpy()[0]), -0.001, delta=1.0e-5)

    # The winding choice is discrete, but torque derivatives within a turn remain continuous.
    builder.joint_q[0] = 7.0
    differentiable = builder.finalize(device=device, requires_grad=True)
    state, output = differentiable.state(), differentiable.state()
    differentiable_solver = SolverSemiImplicit(differentiable, angular_damping=0.0)
    with wp.Tape() as tape:
        newton.eval_fk(differentiable, state.joint_q, state.joint_qd, state)
        differentiable_solver.step(state, output, differentiable.control(), None, 0.001)
    tape.backward(grads={output.joint_qd: wp.ones_like(output.joint_qd)})
    test.assertAlmostEqual(float(state.joint_q.grad.numpy()[0]), -0.002, delta=1.0e-5)
    tape.zero()
    tape.backward(grads={output.joint_q: wp.ones_like(output.joint_q)})
    test.assertAlmostEqual(float(state.joint_q.grad.numpy()[0]), 1.0 - 2.0e-6, delta=1.0e-7)

    # Reset an already-used state from maximal coordinates, explicitly discarding winding.
    model.joint_limit_upper.fill_(1.0e6)
    model.joint_target_ke.fill_(2.0)
    state, output = model.state(), model.state()
    state.joint_q.fill_(20.0)
    newton.eval_fk(model, state.joint_q, state.joint_qd, state)
    control.joint_target_q.fill_(20.0)
    solver.step(state, output, control, None, 0.001)
    output.body_q.assign(model.body_q)
    output.body_qd.zero_()
    newton.eval_ik(model, output, output.joint_q, output.joint_qd)
    control.joint_target_q.zero_()
    output.clear_forces()
    solver.step(output, state, control, None, 0.001)
    test.assertAlmostEqual(float(state.joint_q.numpy()[0]), 0.0, delta=1.0e-5)
    test.assertAlmostEqual(float(state.joint_qd.numpy()[0]), 0.0, delta=1.0e-5)


class TestSemiImplicitJointDrive(unittest.TestCase):
    """Test SemiImplicit joint drives independently of optional backends."""


add_function_test(
    TestSemiImplicitJointDrive,
    "test_semi_implicit_revolute_winding",
    test_semi_implicit_revolute_winding,
    devices=get_test_devices(),
)


if __name__ == "__main__":
    unittest.main(verbosity=2)
