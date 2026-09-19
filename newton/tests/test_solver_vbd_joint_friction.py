# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import unittest

import numpy as np
import warp as wp

import newton
from newton._src.sim.joint_coordinates import eval_joint_coordinate
from newton._src.solvers.vbd.joint_coordinates import JointCoordinateData, JointCoordinates
from newton.tests.unittest_utils import add_function_test, get_test_devices

_DT = 1.0 / 240.0


@wp.kernel
def _sample_coordinates(
    data: JointCoordinateData,
    joint: int,
    poses: wp.array[wp.transform],
    q: wp.array[float],
    gradients: wp.array2d[wp.spatial_vector],
):
    component = wp.tid()
    coordinate, parent, child = eval_joint_coordinate(
        joint,
        component,
        poses,
        data.body_com,
        data.joint_type,
        data.parent,
        data.child,
        data.X_p,
        data.X_c,
        data.qd_start,
        data.dof_dim,
        data.axis,
    )
    q[component] = coordinate
    gradients[component, 0] = parent
    gradients[component, 1] = child


def test_vbd_friction_coordinate_gradients(test, device):
    """Check the coordinate covectors used for friction and mimic virtual work."""
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    parent = builder.add_link(mass=1.0, inertia=wp.mat33(np.eye(3)), com=wp.vec3(0.1, 0.2, 0.3))
    child = builder.add_link(mass=1.0, inertia=wp.mat33(np.eye(3)), com=wp.vec3(-0.2, 0.1, 0.0))
    free = builder.add_joint_free(child=parent)
    config = newton.ModelBuilder.JointDofConfig
    joint = builder.add_joint_d6(
        parent=parent,
        child=child,
        linear_axes=[config(axis=axis) for axis in (newton.Axis.X, newton.Axis.Y, newton.Axis.Z)],
        angular_axes=[config(axis=axis) for axis in (newton.Axis.X, newton.Axis.Y, newton.Axis.Z)],
    )
    builder.add_articulation([free, joint])
    builder.joint_q[-6:] = [0.6, 0.4, -0.2, 0.3, 0.5, -0.4]
    model = builder.finalize(device=device)
    data = JointCoordinates(model).data
    state = model.state()
    newton.eval_fk(model, model.joint_q, model.joint_qd, state)
    original = state.body_q.numpy().copy()
    q = wp.empty(6, dtype=float, device=device)
    gradients = wp.empty((6, 2), dtype=wp.spatial_vector, device=device)

    def sample(poses):
        state.body_q.assign(poses)
        wp.launch(_sample_coordinates, dim=6, inputs=[data, joint, state.body_q], outputs=[q, gradients], device=device)
        return q.numpy().copy(), gradients.numpy().copy()

    _, analytic = sample(original)
    epsilon = 1.0e-3
    com = model.body_com.numpy()
    for body in range(2):
        for axis in range(6):
            samples = []
            for sign in (-1.0, 1.0):
                poses = original.copy()
                if axis < 3:
                    poses[body, axis] += sign * epsilon
                else:
                    rotation = wp.quat(*original[body, 3:])
                    direction = wp.vec3()
                    direction[axis - 3] = 1.0
                    perturbed = wp.quat_from_axis_angle(direction, sign * epsilon) * rotation
                    poses[body, :3] += np.asarray(wp.quat_rotate(rotation, wp.vec3(*com[body])))
                    poses[body, :3] -= np.asarray(wp.quat_rotate(perturbed, wp.vec3(*com[body])))
                    poses[body, 3:] = np.asarray(perturbed)
                samples.append(sample(poses)[0])
            numeric = (samples[1] - samples[0]) / (2.0 * epsilon)
            np.testing.assert_allclose(analytic[:, body, axis], numeric, atol=2.0e-4)


def _simulate(
    model,
    *,
    steps=120,
    dt=_DT,
    joint_force=None,
    body_force=None,
    iterations=6,
    capture=False,
):
    """Return joint positions, velocities, and friction reactions after simulation."""
    state_in = model.state()
    state_out = model.state()
    control = model.control()
    if body_force is not None:
        state_in.body_f.assign(np.asarray(body_force, dtype=np.float32))
        state_out.body_f.assign(np.asarray(body_force, dtype=np.float32))
    if joint_force is not None:
        control.joint_f.assign(np.asarray(joint_force, dtype=np.float32))

    newton.eval_fk(model, model.joint_q, model.joint_qd, state_in)
    solver = newton.solvers.SolverVBD(model, iterations=iterations, rigid_compliant_alm=True)
    if capture and model.device.is_cuda:
        solver.step(state_in, state_out, control, None, dt)
        solver.reset(state_in)
        newton.eval_fk(model, model.joint_q, model.joint_qd, state_in)
        with wp.ScopedCapture(device=model.device) as graph:
            for _ in range(steps):
                solver.step(state_in, state_out, control, None, dt)
                state_in, state_out = state_out, state_in
        wp.capture_launch(graph.graph)
    else:
        for _ in range(steps):
            solver.step(state_in, state_out, control, None, dt)
            state_in, state_out = state_out, state_in

    joint_q = wp.empty_like(model.joint_q)
    joint_qd = wp.empty_like(model.joint_qd)
    newton.eval_ik(model, state_in, joint_q, joint_qd)
    return joint_q.numpy(), joint_qd.numpy(), solver.joint_friction_lambda.numpy()


def _build_single_dof_model(device, joint_type, friction):
    """Build one free scalar joint with unit mass and inertia."""
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    body = builder.add_link(
        mass=1.0,
        inertia=wp.mat33(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0),
        lock_inertia=True,
    )
    kwargs = {
        "parent": -1,
        "child": body,
        "axis": newton.Axis.Z,
        "target_ke": 0.0,
        "target_kd": 0.0,
        "limit_ke": 0.0,
        "limit_kd": 0.0,
        "friction": friction,
    }
    if joint_type == newton.JointType.REVOLUTE:
        joint = builder.add_joint_revolute(**kwargs)
    elif joint_type == newton.JointType.BALL:
        joint = builder.add_joint_ball(parent=-1, child=body, friction=friction)
    else:
        joint = builder.add_joint_prismatic(**kwargs)
    builder.joint_qd[0] = 2.0
    builder.add_articulation([joint])
    builder.color()
    return builder.finalize(device=device)


def _build_actuated_mimic_model(device, follower_friction):
    """Build an actuated reference and equally geared follower with dry friction."""
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    leader_body = builder.add_link(
        mass=1.0,
        inertia=wp.mat33(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0),
        lock_inertia=True,
    )
    follower_body = builder.add_link(
        mass=1.0,
        inertia=wp.mat33(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0),
        lock_inertia=True,
    )
    common = {
        "parent": -1,
        "axis": newton.Axis.Z,
        "target_ke": 0.0,
        "target_kd": 0.0,
        "limit_ke": 0.0,
        "limit_kd": 0.0,
    }
    leader = builder.add_joint_revolute(child=leader_body, friction=0.4, **common)
    follower = builder.add_joint_revolute(child=follower_body, friction=follower_friction, **common)
    builder.joint_qd[:] = [0.5, 0.5]
    builder.add_articulation([leader, follower])
    builder.set_joint_mimic(follower, leader)
    builder.color()
    return builder.finalize(device=device)


def _build_d6_model(device):
    """Build a two-axis translational D6 joint with distinct friction values."""
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    body = builder.add_link(
        mass=1.0,
        inertia=wp.mat33(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0),
        lock_inertia=True,
    )
    axes = [
        newton.ModelBuilder.JointDofConfig(
            axis=newton.Axis.X,
            limit_ke=0.0,
            limit_kd=0.0,
            friction=0.5,
        ),
        newton.ModelBuilder.JointDofConfig(
            axis=newton.Axis.Y,
            limit_ke=0.0,
            limit_kd=0.0,
            friction=1.0,
        ),
    ]
    joint = builder.add_joint_d6(parent=-1, child=body, linear_axes=axes)
    builder.joint_qd[:] = [2.0, 2.0]
    builder.add_articulation([joint])
    builder.color()
    return builder.finalize(device=device)


def test_vbd_joint_friction_coast_down(test, device):
    """Dissipate motion with revolute and prismatic Coulomb friction."""
    for joint_type in (newton.JointType.REVOLUTE, newton.JointType.PRISMATIC):
        with test.subTest(joint_type=joint_type):
            model = _build_single_dof_model(device, joint_type, friction=1.0)
            _, joint_qd, _ = _simulate(model)
            test.assertAlmostEqual(float(joint_qd[0]), 1.5, delta=0.1)


def test_vbd_d6_joint_friction(test, device):
    """Apply distinct Coulomb friction values to each free D6 axis."""
    model = _build_d6_model(device)
    _, joint_qd, _ = _simulate(model)
    np.testing.assert_allclose(joint_qd, [1.75, 1.5], atol=0.1)


def test_vbd_mimic_follower_friction(test, device):
    """Transfer follower friction through an actuated mimic relationship."""
    leader_only_model = _build_actuated_mimic_model(device, follower_friction=0.0)
    both_joints_model = _build_actuated_mimic_model(device, follower_friction=0.8)

    _, leader_only_qd, _ = _simulate(leader_only_model, joint_force=[2.0, 0.0])
    _, both_joints_qd, _ = _simulate(both_joints_model, joint_force=[2.0, 0.0])

    np.testing.assert_allclose(both_joints_qd[1], both_joints_qd[0], atol=2.0e-3)
    test.assertLess(float(both_joints_qd[0]), float(leader_only_qd[0]) - 0.1)


def test_vbd_joint_friction_stop(test, device):
    """Friction must not reverse or amplify a nearly stopped joint's velocity."""
    for joint_type in (newton.JointType.REVOLUTE, newton.JointType.PRISMATIC):
        for iterations in (1, 6, 7, 20):
            with test.subTest(joint_type=joint_type, iterations=iterations):
                model = _build_single_dof_model(device, joint_type, friction=100.0)
                model.joint_qd.assign([0.05])
                velocity = float(_simulate(model, steps=1, iterations=iterations)[1][0])
                test.assertGreaterEqual(velocity, -1.0e-5)
                test.assertLessEqual(velocity, 0.05)


def test_vbd_mimic_friction_force_balance(test, device):
    """Both friction forces must enter the constrained pair's momentum balance."""
    model = _build_actuated_mimic_model(device, follower_friction=16.0)
    model.joint_friction.assign([4.0, 16.0])
    model.joint_qd.zero_()
    for ratio in (1.0, -1.0, -0.5, 2.0):
        for force in (-10.0, 10.0, 50.0):
            with test.subTest(ratio=ratio, force=force):
                model.joint_mimic_coeffs.assign([[0.0, 1.0], [0.0, ratio]])
                _, velocity, reaction = _simulate(model, steps=1, joint_force=[force, 0.0], iterations=64, capture=True)
                # Reflect follower inertia and friction into the leader coordinate.
                momentum_change = (1.0 + ratio * ratio) * float(velocity[0]) / _DT
                net_force = force - reaction[0] - ratio * reaction[1]
                expected = np.sign(force) * max(abs(force) - 4.0 - abs(ratio) * 16.0, 0.0)
                expected *= _DT / (1.0 + ratio * ratio)
                test.assertAlmostEqual(float(velocity[0]), expected, delta=1.0e-5)
                test.assertTrue(np.all(np.abs(reaction) <= [4.0, 16.0]))
                np.testing.assert_allclose(ratio * velocity[0], velocity[1], atol=1.0e-5)
                test.assertAlmostEqual(momentum_change, net_force, delta=0.05)


def test_vbd_multiple_mimic_followers_friction(test, device):
    """Shared leaders need sequential constraint updates and retained reactions."""
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    joints = []
    frictions = (1.0, 2.0, 3.0)
    for friction in frictions:
        body = builder.add_link(mass=1.0, inertia=wp.mat33(np.eye(3)), lock_inertia=True)
        joints.append(builder.add_joint_prismatic(-1, body, axis=newton.Axis.X, friction=friction))
    builder.add_articulation(joints)
    builder.set_joint_mimic(joints[1], joints[0], (0.0, -1.0))
    builder.set_joint_mimic(joints[2], joints[0], (0.0, 2.0))
    builder.color()
    model = builder.finalize(device=device)
    _, velocity, reaction = _simulate(model, steps=1, iterations=64, joint_force=[4.0, 0.0, 0.0], capture=True)
    ratios = np.array([1.0, -1.0, 2.0])
    np.testing.assert_allclose(velocity, ratios * velocity[0], atol=1.0e-5)
    net_force = 4.0 - np.dot(ratios, reaction)
    np.testing.assert_allclose(velocity, 0.0, atol=1.0e-5)
    test.assertTrue(np.all(np.abs(reaction) <= frictions))
    test.assertAlmostEqual(6.0 * float(velocity[0]) / _DT, net_force, delta=0.02)


def test_vbd_serial_mimic_and_passive_friction(test, device):
    """Balance friction on a serial mimic pair and a downstream passive hinge."""
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    joints = []
    parent = -1
    for friction in (1.0, 2.0, 3.0):
        body = builder.add_link(mass=1.0, inertia=wp.mat33(np.eye(3)), lock_inertia=True)
        joints.append(builder.add_joint_revolute(parent, body, axis=newton.Axis.Z, friction=friction))
        parent = body
    builder.add_articulation(joints)
    builder.set_joint_mimic(joints[1], joints[0])
    builder.color()
    model = builder.finalize(device=device)
    _, velocity, reaction = _simulate(model, steps=1, iterations=64, joint_force=[4.0, 0.0, 0.0], capture=True)
    np.testing.assert_allclose(velocity[0], velocity[1], atol=1.0e-5)
    # Body speeds are (v, 2v, 2v + w), so the reduced inertia is [[9,2],[2,1]].
    momentum = np.array([[9.0, 2.0], [2.0, 1.0]]) @ velocity[[0, 2]] / _DT
    net_force = [4.0 - reaction[0] - reaction[1], -reaction[2]]
    test.assertLessEqual(abs(float(reaction[2])), 3.0)
    test.assertAlmostEqual(float(velocity[2]), 0.0, delta=1.0e-5)
    np.testing.assert_allclose(momentum, net_force, atol=0.02)


def test_vbd_joint_friction_response(test, device):
    """Verify static thresholds and Coulomb deceleration in both directions."""
    cases = [(0.0, force, 4.0) for force in (-2.0, 2.0, -8.0, 8.0)]
    cases.extend((velocity, 0.0, 1.0) for velocity in (-2.0, 2.0))
    for joint_type in (newton.JointType.REVOLUTE, newton.JointType.PRISMATIC):
        for initial_velocity, force, friction in cases:
            with test.subTest(joint_type=joint_type, initial_velocity=initial_velocity, force=force):
                model = _build_single_dof_model(device, joint_type, friction=friction)
                model.joint_qd.assign([initial_velocity])
                _, velocity, _ = _simulate(
                    model,
                    steps=1,
                    joint_force=[force],
                    iterations=12,
                    capture=True,
                )
                free_velocity = initial_velocity + force * _DT
                expected = np.sign(free_velocity) * max(abs(free_velocity) - friction * _DT, 0.0)
                test.assertAlmostEqual(float(velocity[0]), expected, delta=2.0e-5)


def test_vbd_ball_joint_friction(test, device):
    """Apply independent BALL torque bounds in the parent-anchor frame, not Euler axes."""
    for sign in (-1.0, 1.0):
        with test.subTest(sign=sign):
            builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
            body = builder.add_link(mass=1.0, inertia=wp.mat33(np.eye(3)), lock_inertia=True)
            joint = builder.add_joint_ball(
                parent=-1,
                child=body,
                parent_xform=wp.transform(wp.vec3(), wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), 0.7)),
                friction=1.0,
            )
            # This orientation is singular for Euler coordinates, but not for BALL motion.
            builder.joint_q[:] = sign * np.asarray(wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), wp.pi / 2))
            builder.add_articulation([joint])
            builder.color()
            model = builder.finalize(device=device)
            model.joint_friction.assign([1.0, 2.0, 3.0])
            torque = wp.quat_rotate(wp.transform_get_rotation(builder.joint_X_p[joint]), wp.vec3(0.5, -4.0, 6.0))
            _, velocity, reaction = _simulate(
                model,
                steps=1,
                body_force=[[0.0, 0.0, 0.0, *torque]],
                iterations=20,
                capture=True,
            )
            np.testing.assert_allclose(velocity, [0.0, -2.0 * _DT, 3.0 * _DT], atol=5.0e-5)
            np.testing.assert_allclose(reaction, [0.5, -2.0, 3.0], atol=0.01)


def test_vbd_joint_friction_persists_and_rebalances(test, device):
    """Retain static reaction without creep, then remove and reverse it with the load."""
    model = _build_single_dof_model(device, newton.JointType.REVOLUTE, friction=4.0)
    model.joint_qd.zero_()
    state_in, state_out = model.state(), model.state()
    control = model.control()
    newton.eval_fk(model, model.joint_q, model.joint_qd, state_in)
    # A deliberately aggressive structural-history decay must not decay
    # the bounded dry-friction reaction.
    solver = newton.solvers.SolverVBD(
        model,
        iterations=32,
        rigid_compliant_alm=True,
        rigid_avbd_gamma=0.9,
    )
    joint_q = wp.empty_like(model.joint_q)
    joint_qd = wp.empty_like(model.joint_qd)

    def advance(force, steps):
        nonlocal state_in, state_out
        control.joint_f.assign([force])
        for _ in range(steps):
            solver.step(state_in, state_out, control, None, _DT)
            state_in, state_out = state_out, state_in
        newton.eval_ik(model, state_in, joint_q, joint_qd)
        return (
            float(joint_q.numpy()[0]),
            float(joint_qd.numpy()[0]),
            float(solver.joint_friction_lambda.numpy()[0]),
        )

    loaded_q, loaded_qd, loaded_lambda = advance(2.0, 1000)
    test.assertAlmostEqual(loaded_qd, 0.0, delta=1.0e-6)
    test.assertAlmostEqual(loaded_lambda, 2.0, delta=1.0e-5)
    test.assertLess(abs(loaded_q), 1.0e-5)

    unloaded_q, unloaded_qd, unloaded_lambda = advance(0.0, 120)
    test.assertAlmostEqual(unloaded_qd, 0.0, delta=1.0e-6)
    test.assertAlmostEqual(unloaded_lambda, 0.0, delta=1.0e-5)
    test.assertLess(abs(unloaded_q - loaded_q), 1.5e-5)

    reversed_q, reversed_qd, reversed_lambda = advance(-2.0, 120)
    test.assertAlmostEqual(reversed_qd, 0.0, delta=1.0e-6)
    test.assertAlmostEqual(reversed_lambda, -2.0, delta=1.0e-5)
    test.assertLess(abs(reversed_q - unloaded_q), 1.0e-5)


def test_vbd_joint_friction_live_update_and_reset(test, device):
    """Read friction bounds dynamically under capture and clear their history on reset."""
    model = _build_single_dof_model(device, newton.JointType.REVOLUTE, friction=0.0)
    model.joint_qd.zero_()
    state_in = model.state()
    state_out = model.state()
    control = model.control()
    control.joint_f.assign([2.0])
    newton.eval_fk(model, model.joint_q, model.joint_qd, state_in)
    solver = newton.solvers.SolverVBD(model, iterations=12, rigid_compliant_alm=True)

    # Compile/lazily initialize before capture, then restore the initial state.
    solver.step(state_in, state_out, control, None, _DT)
    solver.reset(state_in)
    newton.eval_fk(model, model.joint_q, model.joint_qd, state_in)
    if model.device.is_cuda:
        with wp.ScopedCapture(device=model.device) as capture:
            solver.step(state_in, state_out, control, None, _DT)

        def launch():
            wp.capture_launch(capture.graph)

    else:

        def launch():
            solver.step(state_in, state_out, control, None, _DT)

    joint_q = wp.empty_like(model.joint_q)
    joint_qd = wp.empty_like(model.joint_qd)

    def velocity_for(friction):
        model.joint_friction.fill_(friction)
        state_in.body_qd.zero_()
        launch()
        newton.eval_ik(model, state_out, joint_q, joint_qd)
        return float(joint_qd.numpy()[0])

    test.assertAlmostEqual(velocity_for(0.0), 2.0 * _DT, delta=2.0e-5)
    test.assertAlmostEqual(velocity_for(4.0), 0.0, delta=2.0e-5)
    friction_lambda = float(solver.joint_friction_lambda.numpy()[0])
    test.assertGreater(friction_lambda, 0.0)
    test.assertLessEqual(friction_lambda, 4.0 + 1.0e-6)
    test.assertAlmostEqual(velocity_for(1.0), 1.0 * _DT, delta=2.0e-5)
    test.assertLessEqual(abs(float(solver.joint_friction_lambda.numpy()[0])), 1.0 + 1.0e-6)
    test.assertAlmostEqual(velocity_for(0.0), 2.0 * _DT, delta=2.0e-5)
    test.assertEqual(float(solver.joint_friction_lambda.numpy()[0]), 0.0)

    test.assertAlmostEqual(velocity_for(4.0), 0.0, delta=2.0e-5)
    test.assertGreater(abs(float(solver.joint_friction_lambda.numpy()[0])), 0.0)
    solver.reset(state_in, flags=0)
    test.assertAlmostEqual(float(solver.joint_friction_lambda.numpy()[0]), 0.0, delta=1.0e-7)
    test.assertEqual(float(solver.joint_friction_rho.numpy()[0]), 0.0)


def test_vbd_d6_static_friction(test, device):
    """Hold finite-angle D6 loads using the shared coordinate gradients."""
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    body = builder.add_link(mass=1.0, inertia=wp.mat33(np.eye(3)), lock_inertia=True)
    joint = builder.add_joint_d6(
        -1,
        body,
        angular_axes=[
            newton.ModelBuilder.JointDofConfig(axis=axis, friction=4.0)
            for axis in (newton.Axis.X, newton.Axis.Y, newton.Axis.Z)
        ],
    )
    builder.joint_q[:] = [0.3, 0.5, -0.4]
    builder.add_articulation([joint])
    builder.color()
    model = builder.finalize(device=device)
    state = model.state()
    newton.eval_fk(model, model.joint_q, model.joint_qd, state)
    q = wp.empty(3, dtype=float, device=device)
    gradients = wp.empty((3, 2), dtype=wp.spatial_vector, device=device)
    wp.launch(
        _sample_coordinates,
        dim=3,
        inputs=[JointCoordinates(model).data, joint, state.body_q],
        outputs=[q, gradients],
        device=device,
    )
    load = np.array([2.0, -2.0, 2.0])
    torque = load @ gradients.numpy()[:, 1, 3:]
    # Resolve reactions above the O(eps / dt**2) floor at a finite float32 pose.
    position, velocity, reaction = _simulate(
        model,
        steps=120,
        dt=1.0 / 60.0,
        iterations=32,
        body_force=[[0.0, 0.0, 0.0, *torque]],
    )
    np.testing.assert_allclose(position, model.joint_q.numpy(), atol=1.0e-5)
    # Finite float32 poses resolve velocity only to O(eps / dt).
    np.testing.assert_allclose(velocity, 0.0, atol=1.0e-4)
    np.testing.assert_allclose(reaction, load, atol=0.02)


def test_vbd_joint_friction_validation(test, device):
    """Validate bounds and D6 axes, and leave unsupported joint types inactive."""
    model = _build_single_dof_model(device, newton.JointType.REVOLUTE, friction=np.nan)
    with test.assertRaisesRegex(ValueError, "must contain finite values"):
        newton.solvers.SolverVBD(model, rigid_compliant_alm=True)

    def build_d6(axes, friction):
        builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
        body = builder.add_link(mass=1.0, inertia=wp.mat33(np.eye(3)), lock_inertia=True)
        config = newton.ModelBuilder.JointDofConfig
        joint = builder.add_joint_d6(
            -1,
            body,
            linear_axes=[config(axis=axis, friction=friction) for axis in axes],
            label="validated_d6",
        )
        builder.add_articulation([joint])
        builder.color()
        return builder.finalize(device=device)

    with test.assertRaisesRegex(ValueError, "at most three linear"):
        newton.solvers.SolverVBD(
            build_d6((newton.Axis.X, newton.Axis.Y, newton.Axis.Z, newton.Axis.X), 1.0),
            rigid_compliant_alm=True,
        )
    nonorthogonal = (newton.Axis.X, (1.0, 1.0, 0.0))
    with test.assertRaisesRegex(ValueError, "invalid linear axes"):
        newton.solvers.SolverVBD(build_d6(nonorthogonal, 1.0), rigid_compliant_alm=True)
    newton.solvers.SolverVBD(build_d6(nonorthogonal, 0.0), rigid_compliant_alm=True)

    for joint_type in (newton.JointType.FREE, newton.JointType.ROD):
        builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
        body = builder.add_link(mass=1.0, inertia=wp.mat33(np.eye(3)))
        if joint_type == newton.JointType.FREE:
            joint = builder.add_joint_free(child=body)
        else:
            joint = builder.add_joint_rod(parent=-1, child=body)
        builder.add_articulation([joint])
        builder.color()
        model = builder.finalize(device=device)
        model.joint_friction.fill_(1.0)
        solver = newton.solvers.SolverVBD(model, rigid_compliant_alm=True)
        solver.step(model.state(), model.state(), model.control(), None, _DT)
        np.testing.assert_array_equal(solver.joint_friction_rho.numpy(), 0.0)
        np.testing.assert_array_equal(solver.joint_friction_lambda.numpy(), 0.0)


class TestSolverVBDJointFriction(unittest.TestCase):
    pass


devices = get_test_devices(mode="basic")
add_function_test(
    TestSolverVBDJointFriction,
    "test_vbd_serial_mimic_and_passive_friction",
    test_vbd_serial_mimic_and_passive_friction,
    devices=devices,
)
add_function_test(
    TestSolverVBDJointFriction,
    "test_vbd_multiple_mimic_followers_friction",
    test_vbd_multiple_mimic_followers_friction,
    devices=devices,
)
add_function_test(
    TestSolverVBDJointFriction,
    "test_vbd_friction_coordinate_gradients",
    test_vbd_friction_coordinate_gradients,
    devices=devices,
)
add_function_test(
    TestSolverVBDJointFriction, "test_vbd_joint_friction_stop", test_vbd_joint_friction_stop, devices=devices
)
add_function_test(
    TestSolverVBDJointFriction,
    "test_vbd_mimic_friction_force_balance",
    test_vbd_mimic_friction_force_balance,
    devices=devices,
)
add_function_test(
    TestSolverVBDJointFriction,
    "test_vbd_joint_friction_coast_down",
    test_vbd_joint_friction_coast_down,
    devices=devices,
)
add_function_test(
    TestSolverVBDJointFriction,
    "test_vbd_mimic_follower_friction",
    test_vbd_mimic_follower_friction,
    devices=devices,
)
add_function_test(
    TestSolverVBDJointFriction,
    "test_vbd_d6_joint_friction",
    test_vbd_d6_joint_friction,
    devices=devices,
)


for test_function in (
    test_vbd_d6_static_friction,
    test_vbd_joint_friction_response,
    test_vbd_ball_joint_friction,
    test_vbd_joint_friction_persists_and_rebalances,
    test_vbd_joint_friction_live_update_and_reset,
    test_vbd_joint_friction_validation,
):
    add_function_test(TestSolverVBDJointFriction, test_function.__name__, test_function, devices=devices)


if __name__ == "__main__":
    unittest.main(verbosity=2, failfast=True)
