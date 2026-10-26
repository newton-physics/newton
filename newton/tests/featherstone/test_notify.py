# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import unittest

import numpy as np
import warp as wp

import newton
from newton.tests.unittest_utils import add_function_test, get_test_devices


def _build_model(device):
    """Build one free 1 kg body without gravity."""
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    builder.add_body(mass=1.0, inertia=wp.mat33(np.eye(3) * 0.1))
    return builder.finalize(device=device)


def _push(model, solver):
    """Apply a unit force and torque for one step and return the body twist."""
    state_in, state_out = model.state(), model.state()
    state_in.body_f.assign(np.array([[1.0, 0.0, 0.0, 0.0, 0.0, 1.0]], dtype=np.float32))
    solver.step(state_in, state_out, None, None, 0.01)
    return state_out.body_qd.numpy()[0]


def test_inertial_update_reaches_dynamics(test, device):
    """Apply body mass and inertia changes notified after construction."""
    model = _build_model(device)
    solver = newton.solvers.SolverFeatherstone(model)
    model.body_mass.assign(model.body_mass.numpy() * 2.0)
    model.body_inv_mass.assign(model.body_inv_mass.numpy() * 0.5)
    model.body_inertia.assign(model.body_inertia.numpy() * 2.0)
    model.body_inv_inertia.assign(model.body_inv_inertia.numpy() * 0.5)
    solver.notify_model_changed(newton.ModelFlags.BODY_INERTIAL_PROPERTIES)
    twist = _push(model, solver)

    np.testing.assert_allclose(twist, [0.005, 0.0, 0.0, 0.0, 0.0, 0.05], rtol=1e-4, atol=1e-7)


class TestFeatherstoneNotify(unittest.TestCase):
    pass


add_function_test(
    TestFeatherstoneNotify,
    "test_inertial_update_reaches_dynamics",
    test_inertial_update_reaches_dynamics,
    devices=get_test_devices(),
)


if __name__ == "__main__":
    unittest.main(verbosity=2)
