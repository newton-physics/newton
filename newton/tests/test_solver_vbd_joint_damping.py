# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import unittest

import numpy as np
import warp as wp

import newton
from newton.tests.unittest_utils import add_function_test, get_test_devices

_DT = 1.0 / 240.0


def _build_model(device, joint_type, damping, *, friction=None, serial=False, mimic_ratio=None):
    """Build unit-mass scalar joints, optionally in a serial or mimic chain."""
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    joints = []
    parent = -1
    for index, coefficient in enumerate(damping):
        body = builder.add_link(mass=1.0, inertia=wp.mat33(np.eye(3)), lock_inertia=True)
        cfg = newton.ModelBuilder.JointDofConfig(
            axis=newton.Axis.Z,
            damping=coefficient,
            friction=0.0 if friction is None else friction[index],
            target_ke=0.0,
            target_kd=0.0,
            limit_ke=0.0,
            limit_kd=0.0,
            armature=0.0,
        )
        axes = {"linear_axes": [cfg]} if joint_type == newton.JointType.PRISMATIC else {"angular_axes": [cfg]}
        joints.append(builder.add_joint(joint_type, parent, body, **axes))
        if serial:
            parent = body
    builder.add_articulation(joints)
    if mimic_ratio is not None:
        builder.set_joint_mimic(joints[1], joints[0], (0.0, mimic_ratio))
    builder.color()
    return builder.finalize(device=device)


def _simulate(model, *, steps=1, force=None, target_velocity=None, iterations=12, dt=_DT, capture=False):
    """Advance the model and return reconstructed joint velocities."""
    solver = newton.solvers.SolverVBD(model, iterations=iterations, rigid_compliant_alm=True)
    state_in, state_out = model.state(), model.state()
    control = model.control()
    if force is not None:
        control.joint_f.assign(force)
    if target_velocity is not None:
        control.joint_target_qd.assign(target_velocity)
    newton.eval_fk(model, model.joint_q, model.joint_qd, state_in)

    def advance():
        nonlocal state_in, state_out
        for _ in range(steps):
            solver.step(state_in, state_out, control, None, dt)
            state_in, state_out = state_out, state_in

    if capture and model.device.is_cuda:
        solver.step(state_in, state_out, control, None, dt)
        solver.reset(state_in)
        newton.eval_fk(model, model.joint_q, model.joint_qd, state_in)
        with wp.ScopedCapture(device=model.device) as captured:
            advance()
        wp.capture_launch(captured.graph)
    else:
        advance()
    newton.eval_ik(model, state_in, state_in.joint_q, state_in.joint_qd)
    return state_in.joint_qd.numpy()


def test_vbd_joint_damping_implicit_decay(test, device):
    """Match implicit viscous decay without drives, including stiff damping."""
    for joint_type in (newton.JointType.PRISMATIC, newton.JointType.REVOLUTE, newton.JointType.D6):
        for damping in (0.0, 4.0, 10000.0):
            for initial in (-2.0, 2.0):
                with test.subTest(joint_type=joint_type, damping=damping, initial=initial):
                    model = _build_model(device, joint_type, [damping])
                    model.joint_qd.assign([initial])
                    velocity = _simulate(model, iterations=1)[0]
                    test.assertAlmostEqual(float(velocity), initial / (1.0 + damping * _DT), delta=5.0e-5)


def test_vbd_joint_damping_coast_down(test, device):
    """Accumulate viscous decay consistently over timesteps and graph replay."""
    for dt in (_DT, 1.0 / 60.0):
        model = _build_model(device, newton.JointType.PRISMATIC, [4.0])
        model.joint_qd.assign([2.0])
        steps = round(0.5 / dt)
        velocity = _simulate(model, steps=steps, dt=dt, capture=True)[0]
        test.assertAlmostEqual(float(velocity), 2.0 / (1.0 + 4.0 * dt) ** steps, delta=2.0e-4)


def test_vbd_d6_joint_damping(test, device):
    """Apply each D6 axis's authored damping without coupling unrelated axes."""
    for angular_count in (1, 2, 3):
        builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
        body = builder.add_link(mass=1.0, inertia=wp.mat33(np.eye(3)), lock_inertia=True)
        config = newton.ModelBuilder.JointDofConfig
        linear = [config(axis=axis, damping=d) for axis, d in zip(newton.Axis, (1.0, 2.0, 3.0), strict=True)]
        angular = [config(axis=axis, damping=d) for axis, d in zip(newton.Axis, (4.0, 5.0, 6.0), strict=True)]
        joint = builder.add_joint_d6(-1, body, linear_axes=linear, angular_axes=angular[:angular_count])
        builder.add_articulation([joint])
        builder.color()
        model = builder.finalize(device=device)
        damping = np.arange(1.0, 4.0 + angular_count)
        for component in range(len(damping)):
            initial = np.zeros(len(damping), dtype=np.float32)
            initial[component] = 0.5
            model.joint_qd.assign(initial)
            velocity = _simulate(model)
            np.testing.assert_allclose(velocity, initial / (1.0 + damping * _DT), atol=2.0e-5)


def test_vbd_joint_damping_with_drive(test, device):
    """Keep passive damping independent of a nonzero drive velocity target."""
    model = _build_model(device, newton.JointType.PRISMATIC, [4.0])
    model.joint_target_kd.fill_(6.0)
    model.joint_qd.assign([1.0])
    velocity = _simulate(model, target_velocity=[3.0])[0]
    expected = (1.0 + _DT * 6.0 * 3.0) / (1.0 + _DT * (4.0 + 6.0))
    test.assertAlmostEqual(float(velocity), expected, delta=2.0e-5)


def test_vbd_mimic_damping_and_friction(test, device):
    """Reflect both damping and Coulomb friction through the mimic ratio."""
    for joint_type in (newton.JointType.PRISMATIC, newton.JointType.REVOLUTE, newton.JointType.D6):
        for ratio in (1.0, -1.0, -0.5, 2.0):
            with test.subTest(joint_type=joint_type, ratio=ratio):
                model = _build_model(device, joint_type, [4.0, 16.0], friction=[0.4, 0.8], mimic_ratio=ratio)
                velocity = _simulate(model, force=[10.0, 0.0], iterations=24, capture=True)
                np.testing.assert_allclose(velocity[1], ratio * velocity[0], atol=1.0e-5)
                momentum = (1.0 + ratio * ratio) * float(velocity[0]) / _DT
                net_force = 10.0 - (4.0 + ratio * ratio * 16.0) * float(velocity[0])
                net_force -= 0.4 * np.tanh(velocity[0] / 0.01) + ratio * 0.8 * np.tanh(velocity[1] / 0.01)
                test.assertAlmostEqual(momentum, net_force, delta=0.02)


def test_vbd_serial_mimic_passive_damping(test, device):
    """Balance a serial mimic pair and a passive downstream frictional damper."""
    model = _build_model(
        device, newton.JointType.REVOLUTE, [4.0, 8.0, 16.0], friction=[1.0, 2.0, 3.0], serial=True, mimic_ratio=1.0
    )
    velocity = _simulate(model, force=[4.0, 0.0, 0.0], iterations=64, capture=True)
    np.testing.assert_allclose(velocity[0], velocity[1], atol=1.0e-5)
    # Body speeds are (v, 2v, 2v + w); include the downstream body's inertia.
    momentum = np.array([[9.0, 2.0], [2.0, 1.0]]) @ velocity[[0, 2]] / _DT
    net_force = [
        4.0 - 12.0 * velocity[0] - 3.0 * np.tanh(velocity[0] / 0.01),
        -16.0 * velocity[2] - 3.0 * np.tanh(velocity[2] / 0.01),
    ]
    np.testing.assert_allclose(momentum, net_force, atol=0.02)


def test_vbd_joint_damping_live_updates(test, device):
    """Read damping changes during graph replay and respect disabled joints."""
    model = _build_model(device, newton.JointType.PRISMATIC, [0.0])
    model.joint_qd.assign([2.0])
    solver = newton.solvers.SolverVBD(model, iterations=6, rigid_compliant_alm=True)
    state_in, state_out = model.state(), model.state()
    control = model.control()

    def reset():
        solver.reset(state_in)
        newton.eval_fk(model, model.joint_q, model.joint_qd, state_in)

    reset()
    solver.step(state_in, state_out, control, None, _DT)
    graph = None
    if model.device.is_cuda:
        with wp.ScopedCapture(device=model.device) as captured:
            solver.step(state_in, state_out, control, None, _DT)
        graph = captured.graph
    previously_enabled = True
    for damping, enabled in ((4.0, True), (100.0, True), (100.0, False), (0.0, True)):
        model.joint_damping.fill_(damping)
        model.joint_enabled.fill_(enabled)
        if enabled != previously_enabled:
            solver.notify_model_changed(newton.ModelFlags.JOINT_PROPERTIES)
        previously_enabled = enabled
        reset()
        if graph is None:
            solver.step(state_in, state_out, control, None, _DT)
        else:
            wp.capture_launch(graph)
        newton.eval_ik(model, state_out, state_out.joint_q, state_out.joint_qd)
        expected = 2.0 / (1.0 + (damping if enabled else 0.0) * _DT)
        test.assertAlmostEqual(float(state_out.joint_qd.numpy()[0]), expected, delta=2.0e-5)


class TestSolverVBDJointDamping(unittest.TestCase):
    pass


for test_function in (
    test_vbd_joint_damping_implicit_decay,
    test_vbd_joint_damping_coast_down,
    test_vbd_d6_joint_damping,
    test_vbd_joint_damping_with_drive,
    test_vbd_mimic_damping_and_friction,
    test_vbd_serial_mimic_passive_damping,
    test_vbd_joint_damping_live_updates,
):
    add_function_test(TestSolverVBDJointDamping, test_function.__name__, test_function, devices=get_test_devices())


if __name__ == "__main__":
    unittest.main(verbosity=2)
