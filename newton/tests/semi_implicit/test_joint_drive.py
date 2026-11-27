# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import math
import unittest

import numpy as np
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


def _build_hinge_model(device, *, revolute: bool, **dof_kwargs):
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    body = builder.add_link(mass=1.0, inertia=wp.mat33(np.eye(3)))
    if revolute:
        joint = builder.add_joint_revolute(-1, body, axis=newton.Axis.Z, **dof_kwargs)
    else:
        config = newton.ModelBuilder.JointDofConfig
        joint = builder.add_joint_d6(
            -1,
            body,
            linear_axes=[config.create_unlimited(newton.Axis.X)],
            angular_axes=[config(axis=newton.Axis.Z, **dof_kwargs)],
        )
    builder.add_articulation([joint])
    return builder.finalize(device=device)


def test_semi_implicit_passive_spring_winding(test, device):
    """An unwrapped passive-spring sweep must not run away past the pi boundary."""
    model = _build_hinge_model(device, revolute=True, stiffness=1.0, rest_q=3.0)
    finals = {}
    for solver in (
        newton.solvers.SolverSemiImplicit(model),
        newton.solvers.SolverFeatherstone(model),
    ):
        state_in, state_out = model.state(), model.state()
        state_in.joint_q.fill_(3.0)
        state_in.joint_qd.fill_(3.0)
        newton.eval_fk(model, state_in.joint_q, state_in.joint_qd, state_in)
        control = model.control()
        for _ in range(120):
            state_in.clear_forces()
            solver.step(state_in, state_out, control, None, 0.01)
            state_in, state_out = state_out, state_in
        finals[type(solver)] = float(state_in.joint_q.numpy()[0])

    semi = finals[newton.solvers.SolverSemiImplicit]
    featherstone = finals[newton.solvers.SolverFeatherstone]
    test.assertGreater(semi, math.pi)
    test.assertLess(semi, 6.5)
    test.assertAlmostEqual(semi, featherstone, delta=0.2)


def test_semi_implicit_d6_hinge_winding(test, device):
    """One-axis D6 hinges keep winding in joint_q like REVOLUTE joints."""
    model = _build_hinge_model(device, revolute=False, target_ke=100.0, target_kd=20.0)
    solver = newton.solvers.SolverSemiImplicit(model)
    state_in, state_out = model.state(), model.state()
    control = model.control()
    target_index = int(model.joint_target_q_start.numpy()[0]) + 1
    targets = control.joint_target_q.numpy().copy()
    angles = [0.0, 2.0 * math.pi]
    for frame in range(80):
        targets[target_index] = angles[1] * min((frame + 1) / 40.0, 1.0)
        control.joint_target_q.assign(targets)
        state_in.clear_forces()
        solver.step(state_in, state_out, control, None, 0.01)
        state_in, state_out = state_out, state_in
    test.assertGreater(float(state_in.joint_q.numpy()[1]), 2.0 * math.pi - 0.3)


def test_semi_implicit_reset_reanchors_hinge(test, device):
    """SolverBase.reset re-anchors winding from body poses for selected worlds."""
    model = _build_hinge_model(device, revolute=True, target_ke=0.0, target_kd=0.0)
    solver = newton.solvers.SolverSemiImplicit(model)
    state, output = model.state(), model.state()
    state.joint_q.fill_(7.0)
    newton.eval_fk(model, state.joint_q, state.joint_qd, state)
    solver.reset(state)
    # A pose alone cannot restore the independently intended 7 rad coordinate;
    # reset re-anchors to the equivalent angle nearest the authored rest (0).
    test.assertAlmostEqual(float(state.joint_q.numpy()[0]), 7.0 - 2.0 * math.pi, delta=1.0e-5)


class TestSemiImplicitJointDrive(unittest.TestCase):
    """Test SemiImplicit joint drives independently of optional backends."""


add_function_test(
    TestSemiImplicitJointDrive,
    "test_semi_implicit_revolute_winding",
    test_semi_implicit_revolute_winding,
    devices=get_test_devices(),
)


add_function_test(
    TestSemiImplicitJointDrive,
    "test_semi_implicit_passive_spring_winding",
    test_semi_implicit_passive_spring_winding,
    devices=get_test_devices(),
)


add_function_test(
    TestSemiImplicitJointDrive,
    "test_semi_implicit_d6_hinge_winding",
    test_semi_implicit_d6_hinge_winding,
    devices=get_test_devices(),
)


add_function_test(
    TestSemiImplicitJointDrive,
    "test_semi_implicit_reset_reanchors_hinge",
    test_semi_implicit_reset_reanchors_hinge,
    devices=get_test_devices(),
)


if __name__ == "__main__":
    unittest.main(verbosity=2)
