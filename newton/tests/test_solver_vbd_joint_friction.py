# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import unittest

import numpy as np
import warp as wp

import newton
from newton.tests.unittest_utils import add_function_test, get_test_devices


_DT = 1.0 / 240.0


def _simulate(model, *, steps=120, joint_force=None):
    """Simulate a VBD model and return its reconstructed joint velocities."""
    state_in = model.state()
    state_out = model.state()
    control = model.control()
    if joint_force is not None:
        control.joint_f.assign(np.asarray(joint_force, dtype=np.float32))

    newton.eval_fk(model, model.joint_q, model.joint_qd, state_in)
    solver = newton.solvers.SolverVBD(model, iterations=6, rigid_compliant_alm=True)
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
    kwargs = dict(
        parent=-1,
        child=body,
        axis=newton.Axis.Z,
        target_ke=0.0,
        target_kd=0.0,
        limit_ke=0.0,
        limit_kd=0.0,
        friction=friction,
    )
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
    common = dict(
        parent=-1,
        axis=newton.Axis.Z,
        target_ke=0.0,
        target_kd=0.0,
        limit_ke=0.0,
        limit_kd=0.0,
    )
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


class TestSolverVBDJointFriction(unittest.TestCase):
    pass


devices = get_test_devices(mode="basic")
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
