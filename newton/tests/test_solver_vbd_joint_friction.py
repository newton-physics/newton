# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import unittest

import numpy as np
import warp as wp

import newton
from newton._src.sim.joint_mimic import eval_joint_mimic_coordinate
from newton._src.solvers.vbd.joint_mimic import JointMimicSolver, _JointData
from newton.tests.unittest_utils import add_function_test, get_test_devices

_DT = 1.0 / 240.0


@wp.kernel
def _sample_coordinates(
    data: _JointData,
    joint: int,
    poses: wp.array[wp.transform],
    q: wp.array[float],
    gradients: wp.array2d[wp.spatial_vector],
):
    component = wp.tid()
    coordinate, parent, child = eval_joint_mimic_coordinate(
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
    data = JointMimicSolver(model).data
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


def _simulate(model, *, steps=120, joint_force=None, iterations=6, capture=False):
    """Simulate a VBD model and return its reconstructed joint velocities."""
    state_in = model.state()
    state_out = model.state()
    control = model.control()
    if joint_force is not None:
        control.joint_f.assign(np.asarray(joint_force, dtype=np.float32))

    newton.eval_fk(model, model.joint_q, model.joint_qd, state_in)
    solver = newton.solvers.SolverVBD(model, iterations=iterations, rigid_compliant_alm=True)
    if capture and model.device.is_cuda:
        solver.step(state_in, state_out, control, None, _DT)
        solver.reset(state_in)
        newton.eval_fk(model, model.joint_q, model.joint_qd, state_in)
        with wp.ScopedCapture(device=model.device) as graph:
            for _ in range(steps):
                solver.step(state_in, state_out, control, None, _DT)
                state_in, state_out = state_out, state_in
        wp.capture_launch(graph.graph)
    else:
        for _ in range(steps):
            solver.step(state_in, state_out, control, None, _DT)
            state_in, state_out = state_out, state_in

    joint_q = wp.empty_like(model.joint_q)
    joint_qd = wp.empty_like(model.joint_qd)
    newton.eval_ik(model, state_in, joint_q, joint_qd)
    return joint_qd.numpy()


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
    else:
        joint = builder.add_joint_prismatic(**kwargs)
    builder.joint_qd[0] = 2.0
    builder.add_articulation([joint])
    builder.color()
    return builder.finalize(device=device)


def _build_actuated_mimic_model(device, follower_friction):
    """Build an actuated leader and equally geared follower with dry friction."""
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
            joint_qd = _simulate(model)
            test.assertAlmostEqual(float(joint_qd[0]), 1.5, delta=0.1)


def test_vbd_d6_joint_friction(test, device):
    """Apply distinct Coulomb friction values to each free D6 axis."""
    model = _build_d6_model(device)
    joint_qd = _simulate(model)
    np.testing.assert_allclose(joint_qd, [1.75, 1.5], atol=0.1)


def test_vbd_mimic_follower_friction(test, device):
    """Transfer follower friction through an actuated mimic relationship."""
    leader_only_model = _build_actuated_mimic_model(device, follower_friction=0.0)
    both_joints_model = _build_actuated_mimic_model(device, follower_friction=0.8)

    leader_only_qd = _simulate(leader_only_model, joint_force=[2.0, 0.0])
    both_joints_qd = _simulate(both_joints_model, joint_force=[2.0, 0.0])

    np.testing.assert_allclose(both_joints_qd[1], both_joints_qd[0], atol=2.0e-3)
    test.assertLess(float(both_joints_qd[0]), float(leader_only_qd[0]) - 0.1)


def test_vbd_joint_friction_stop(test, device):
    """Friction must not reverse or amplify a nearly stopped joint's velocity."""
    for joint_type in (newton.JointType.REVOLUTE, newton.JointType.PRISMATIC):
        for iterations in (1, 6, 7, 20):
            with test.subTest(joint_type=joint_type, iterations=iterations):
                model = _build_single_dof_model(device, joint_type, friction=100.0)
                model.joint_qd.assign([0.05])
                velocity = float(_simulate(model, steps=1, iterations=iterations)[0])
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
                velocity = _simulate(model, steps=1, joint_force=[force, 0.0], iterations=24, capture=True)
                # Reflect follower inertia and friction into the leader coordinate.
                momentum_change = (1.0 + ratio * ratio) * float(velocity[0]) / _DT
                net_force = force - 4.0 * np.tanh(float(velocity[0]) / 0.01)
                net_force -= ratio * 16.0 * np.tanh(float(velocity[1]) / 0.01)
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
    velocity = _simulate(model, steps=1, iterations=24, joint_force=[4.0, 0.0, 0.0], capture=True)
    ratios = np.array([1.0, -1.0, 2.0])
    np.testing.assert_allclose(velocity, ratios * velocity[0], atol=1.0e-5)
    net_force = 4.0 - np.dot(ratios * frictions, np.tanh(velocity / 0.01))
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
    velocity = _simulate(model, steps=1, iterations=64, joint_force=[4.0, 0.0, 0.0], capture=True)
    np.testing.assert_allclose(velocity[0], velocity[1], atol=1.0e-5)
    # Body speeds are (v, 2v, 2v + w), so the reduced inertia is [[9,2],[2,1]].
    momentum = np.array([[9.0, 2.0], [2.0, 1.0]]) @ velocity[[0, 2]] / _DT
    net_force = [4.0 - 3.0 * np.tanh(velocity[0] / 0.01), -3.0 * np.tanh(velocity[2] / 0.01)]
    np.testing.assert_allclose(momentum, net_force, atol=0.02)


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


if __name__ == "__main__":
    unittest.main(verbosity=2, failfast=True)
