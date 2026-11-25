# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import unittest

import numpy as np
import warp as wp

import newton
from newton.tests.unittest_utils import add_function_test, get_test_devices


def _build_model(device):
    """Build a two-link revolute pendulum whose dynamics depend on body mass and COM."""
    builder = newton.ModelBuilder()
    parent = -1
    joints = []
    for i in range(2):
        child = builder.add_link(mass=1.0, com=wp.vec3(0.1, 0.0, 0.0), inertia=wp.mat33(np.eye(3) * 0.01))
        xform = wp.transform((0.0, 0.0, 1.0) if parent == -1 else (0.2, 0.0, 0.0), wp.quat_identity())
        axis = newton.Axis.Y if i == 0 else newton.Axis.Z
        joints.append(builder.add_joint_revolute(parent, child, parent_xform=xform, axis=axis))
        parent = child
    builder.add_articulation(joints)
    model = builder.finalize(device=device)
    model.joint_qd.assign(np.array([0.5, -1.0], dtype=np.float32))
    return model


def _simulate(model, solver, state_0, state_1, steps):
    control = model.control()
    for _ in range(steps):
        solver.step(state_0, state_1, control, None, 1.0e-3)
        state_0, state_1 = state_1, state_0
    return state_0, state_1


def test_notify_body_inertial_properties(test, device):
    """Runtime mass and COM edits must reach the solver after notifying BODY_INERTIAL_PROPERTIES.

    The solvers cache the mass matrix across steps, so the notified solver only matches the fresh
    one if the notification both refreshes the inertia buffers and forces a mass-matrix rebuild.
    """
    model = _build_model(device)
    solver = newton.solvers.SolverFeatherstone(model, update_mass_matrix_interval=100)
    state_0, state_1 = _simulate(model, solver, model.state(), model.state(), 1)

    model.body_mass.assign(np.array([3.0, 0.5], dtype=np.float32))
    model.body_com.assign(np.array([[0.2, 0.0, 0.05], [0.05, 0.1, 0.0]], dtype=np.float32))
    solver.notify_model_changed(newton.ModelFlags.BODY_INERTIAL_PROPERTIES)

    ref_state_0, ref_state_1 = model.state(), model.state()
    ref_state_0.assign(state_0)
    ref_solver = newton.solvers.SolverFeatherstone(model, update_mass_matrix_interval=100)

    state, _ = _simulate(model, solver, state_0, state_1, 10)
    ref_state, _ = _simulate(model, ref_solver, ref_state_0, ref_state_1, 10)

    np.testing.assert_allclose(state.joint_q.numpy(), ref_state.joint_q.numpy(), rtol=1.0e-5, atol=1.0e-6)
    np.testing.assert_allclose(state.joint_qd.numpy(), ref_state.joint_qd.numpy(), rtol=1.0e-5, atol=1.0e-6)


class TestFeatherstoneNotifyModelChanged(unittest.TestCase):
    pass


add_function_test(
    TestFeatherstoneNotifyModelChanged,
    "test_notify_body_inertial_properties",
    test_notify_body_inertial_properties,
    devices=get_test_devices(),
)


if __name__ == "__main__":
    unittest.main(verbosity=2)
