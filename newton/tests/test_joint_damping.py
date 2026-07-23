# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import unittest

import numpy as np
import warp as wp

import newton
from newton.tests.unittest_utils import add_function_test, get_cuda_test_devices, get_test_devices


class TestJointDamping(unittest.TestCase):
    pass


def _build_revolute_model(device, damping: float):
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0), up_axis=newton.Axis.Y)
    body = builder.add_link(
        mass=1.0,
        inertia=wp.mat33(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0),
        lock_inertia=True,
    )
    joint = builder.add_joint_revolute(
        parent=-1,
        child=body,
        axis=newton.Axis.Z,
        target_ke=0.0,
        target_kd=0.0,
        damping=damping,
        limit_lower=-1.0e6,
        limit_upper=1.0e6,
        limit_ke=0.0,
        limit_kd=0.0,
        armature=0.0,
        friction=0.0,
    )
    builder.add_articulation([joint])
    builder.joint_qd[0] = 1.0

    return builder.finalize(device=device)


def _build_ball_model(device, damping: float):
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0), up_axis=newton.Axis.Y)
    body = builder.add_link(
        mass=1.0,
        inertia=wp.mat33(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0),
        lock_inertia=True,
    )
    joint = builder.add_joint(
        newton.JointType.BALL,
        parent=-1,
        child=body,
        angular_axes=[
            newton.ModelBuilder.JointDofConfig(axis=newton.Axis.X, damping=damping, armature=0.0, friction=0.0),
            newton.ModelBuilder.JointDofConfig(axis=newton.Axis.Y, damping=damping, armature=0.0, friction=0.0),
            newton.ModelBuilder.JointDofConfig(axis=newton.Axis.Z, damping=damping, armature=0.0, friction=0.0),
        ],
    )
    builder.add_articulation([joint])
    builder.joint_qd[0:3] = [1.0, 0.0, 0.0]

    return builder.finalize(device=device)


def _simulate_joint_damping(device, solver_fn, damping: float, sync_joint_qd: bool) -> tuple[float, float]:
    model = _build_revolute_model(device, damping)
    solver = solver_fn(model)

    state_0, state_1 = model.state(), model.state()
    control = model.control()
    newton.eval_fk(model, model.joint_q, model.joint_qd, state_0)

    initial_qd = float(state_0.joint_qd.numpy()[0])
    for _ in range(8):
        solver.step(state_0, state_1, control, None, 0.01)
        state_0, state_1 = state_1, state_0

    if sync_joint_qd:
        newton.eval_ik(model, state_0, state_0.joint_q, state_0.joint_qd)

    return initial_qd, float(state_0.joint_qd.numpy()[0])


def _simulate_ball_joint_damping(device, solver_fn, damping: float) -> tuple[float, float]:
    model = _build_ball_model(device, damping)
    solver = solver_fn(model)

    state_0, state_1 = model.state(), model.state()
    control = model.control()
    newton.eval_fk(model, model.joint_q, model.joint_qd, state_0)

    initial_speed = float(np.linalg.norm(state_0.joint_qd.numpy()[0:3]))
    for _ in range(8):
        solver.step(state_0, state_1, control, None, 0.01)
        state_0, state_1 = state_1, state_0

    newton.eval_ik(model, state_0, state_0.joint_q, state_0.joint_qd)
    return initial_speed, float(np.linalg.norm(state_0.joint_qd.numpy()[0:3]))


def test_revolute_joint_damping_decays_velocity(test: TestJointDamping, device, solver_fn, sync_joint_qd):
    undamped_initial, undamped_final = _simulate_joint_damping(
        device, solver_fn, damping=0.0, sync_joint_qd=sync_joint_qd
    )
    damped_initial, damped_final = _simulate_joint_damping(device, solver_fn, damping=3.0, sync_joint_qd=sync_joint_qd)

    np.testing.assert_allclose(undamped_final, undamped_initial, atol=1.0e-5, rtol=1.0e-5)
    test.assertLess(abs(damped_final), abs(damped_initial) * 0.85)


def test_semi_implicit_ball_joint_damping_decays_velocity(test: TestJointDamping, device):
    def solver_fn(model):
        return newton.solvers.SolverSemiImplicit(model, angular_damping=0.0)

    undamped_initial, undamped_final = _simulate_ball_joint_damping(device, solver_fn, damping=0.0)
    damped_initial, damped_final = _simulate_ball_joint_damping(device, solver_fn, damping=3.0)

    np.testing.assert_allclose(undamped_final, undamped_initial, atol=1.0e-5, rtol=1.0e-5)
    test.assertLess(damped_final, damped_initial * 0.85)


def test_featherstone_ball_joint_damping_decays_velocity(test: TestJointDamping, device):
    def solver_fn(model):
        return newton.solvers.SolverFeatherstone(model, angular_damping=0.0)

    undamped_initial, undamped_final = _simulate_ball_joint_damping(device, solver_fn, damping=0.0)
    damped_initial, damped_final = _simulate_ball_joint_damping(device, solver_fn, damping=3.0)

    np.testing.assert_allclose(undamped_final, undamped_initial, atol=1.0e-5, rtol=1.0e-5)
    test.assertLess(damped_final, damped_initial * 0.85)


devices = get_test_devices()
solvers = {
    "featherstone": (lambda model: newton.solvers.SolverFeatherstone(model, angular_damping=0.0), False),
    "semi_implicit": (lambda model: newton.solvers.SolverSemiImplicit(model, angular_damping=0.0), True),
    "kamino": (newton.solvers.SolverKamino, False),
}

for device in devices:
    for solver_name, (solver_fn, sync_joint_qd) in solvers.items():
        add_function_test(
            TestJointDamping,
            f"test_revolute_joint_damping_decays_velocity_{solver_name}",
            test_revolute_joint_damping_decays_velocity,
            devices=[device],
            solver_fn=solver_fn,
            sync_joint_qd=sync_joint_qd,
        )

for device in devices:
    add_function_test(
        TestJointDamping,
        "test_semi_implicit_ball_joint_damping_decays_velocity",
        test_semi_implicit_ball_joint_damping_decays_velocity,
        devices=[device],
    )
    add_function_test(
        TestJointDamping,
        "test_featherstone_ball_joint_damping_decays_velocity",
        test_featherstone_ball_joint_damping_decays_velocity,
        devices=[device],
    )

for device in get_cuda_test_devices():
    add_function_test(
        TestJointDamping,
        "test_revolute_joint_damping_decays_velocity_mujoco_warp",
        test_revolute_joint_damping_decays_velocity,
        devices=[device],
        solver_fn=lambda model: newton.solvers.SolverMuJoCo(model, use_mujoco_cpu=False, disable_contacts=True),
        sync_joint_qd=False,
    )


if __name__ == "__main__":
    unittest.main(verbosity=2)
