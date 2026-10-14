# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import unittest

import numpy as np
import warp as wp

import newton
from newton.tests.unittest_utils import add_function_test, get_test_devices

# (X, Z, Y) is the non-XYZ hip ordering used by nv_humanoid.xml.
AXIS_SETS = {
    "xzy": (newton.Axis.X, newton.Axis.Z, newton.Axis.Y),
    "xz": (newton.Axis.X, newton.Axis.Z),
}


class TestJointD6(unittest.TestCase):
    pass


def _build_d6(device, axes, **dof_kwargs):
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    body = builder.add_link(mass=1.0, inertia=wp.mat33(np.diag([0.1, 0.2, 0.15])))
    joint = builder.add_joint_d6(
        parent=-1,
        child=body,
        angular_axes=[newton.ModelBuilder.JointDofConfig(axis=axis, armature=0.0, **dof_kwargs) for axis in axes],
    )
    builder.add_articulation([joint])
    return builder.finalize(device=device)


def _simulate(model, solver, state_0, control, dt, steps):
    state_1 = model.state()
    for _ in range(steps):
        state_0.clear_forces()
        solver.step(state_0, state_1, control, None, dt)
        state_0, state_1 = state_1, state_0
    newton.eval_ik(model, state_0, state_0.joint_q, state_0.joint_qd)
    return state_0


def test_d6_angular_target_matches_fk(test, device, solver_fn, axes):
    """Multi-angular D6 drives must reach the pose eval_fk assigns to the target coordinates."""
    model = _build_d6(device, axes, target_ke=2000.0, target_kd=100.0)
    dof_count = len(axes)
    target = np.array([0.5, -0.4, 0.7][:dof_count], dtype=np.float32)

    control = model.control()
    control.joint_target_q.assign(target)
    state = _simulate(model, solver_fn(model), model.state(), control, dt=1e-3, steps=2000)

    np.testing.assert_allclose(state.joint_q.numpy(), target, atol=2e-3)

    expected = model.state()
    expected.joint_q.assign(target)
    newton.eval_fk(model, expected.joint_q, expected.joint_qd, expected)
    quat_dot = abs(float(np.dot(state.body_q.numpy()[0, 3:], expected.body_q.numpy()[0, 3:])))
    test.assertGreater(quat_dot, 1.0 - 1e-5)


def test_d6_angular_joint_f_matches_featherstone(test, device, solver_fn, axes):
    """Generalized forces on a multi-angular D6 must act on their own coordinate, as in Featherstone."""
    dof_count = len(axes)
    q0 = np.array([0.4, -0.5, 0.3][:dof_count], dtype=np.float32)
    joint_f = np.zeros(dof_count, dtype=np.float32)
    joint_f[-1] = 0.5

    results = []
    for make_solver in (lambda m: newton.solvers.SolverFeatherstone(m, angular_damping=0.0), solver_fn):
        model = _build_d6(device, axes)
        state = model.state()
        state.joint_q.assign(q0)
        newton.eval_fk(model, state.joint_q, state.joint_qd, state)
        control = model.control()
        control.joint_f.assign(joint_f)
        state = _simulate(model, make_solver(model), state, control, dt=1e-4, steps=2000)
        results.append((state.joint_q.numpy(), state.joint_qd.numpy()))

    (q_ref, qd_ref), (q, qd) = results
    np.testing.assert_allclose(q, q_ref, atol=1e-3)
    np.testing.assert_allclose(qd, qd_ref, atol=2e-3)


solvers = {
    "xpbd": lambda model: newton.solvers.SolverXPBD(model, iterations=10, angular_damping=0.0),
    "semi_implicit": lambda model: newton.solvers.SolverSemiImplicit(
        model, angular_damping=0.0, joint_attach_ke=1.0e5, joint_attach_kd=1.0e2
    ),
}

devices = get_test_devices()
for solver_name, solver_fn in solvers.items():
    for axes_name, axes in AXIS_SETS.items():
        add_function_test(
            TestJointD6,
            f"test_d6_angular_target_matches_fk_{axes_name}_{solver_name}",
            test_d6_angular_target_matches_fk,
            devices=devices,
            solver_fn=solver_fn,
            axes=axes,
        )
        add_function_test(
            TestJointD6,
            f"test_d6_angular_joint_f_matches_featherstone_{axes_name}_{solver_name}",
            test_d6_angular_joint_f_matches_featherstone,
            devices=devices,
            solver_fn=solver_fn,
            axes=axes,
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
