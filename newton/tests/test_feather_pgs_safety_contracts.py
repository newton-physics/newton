# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Selected-world reset of mass factors under eager stepping and CUDA graph capture."""

import unittest

import numpy as np
import warp as wp

import newton
from newton.solvers import SolverFeatherPGS
from newton.tests.unittest_utils import add_function_test, get_cuda_test_devices


def _driven_model(device, worlds=2):
    """One driven slider per world plus one global (world -1) free body, without gravity."""
    template = newton.ModelBuilder(gravity=wp.vec3(0.0))
    body = template.add_link(mass=1.0, inertia=wp.mat33(np.eye(3)))
    joint = template.add_joint_prismatic(
        -1, body, axis=newton.Axis.X, target_ke=10000.0, target_kd=0.0, armature=0.0, damping=0.0
    )
    template.add_articulation([joint])
    builder = newton.ModelBuilder(gravity=wp.vec3(0.0))
    for _ in range(worlds):
        builder.add_world(template)
    builder.add_body(mass=1.0, inertia=wp.mat33(np.eye(3)))
    return builder.finalize(device=device)


def test_partial_reset_refreshes_only_selected_mass_factors(test, device):
    """Refresh only the selected world's (or the global slot's) factors, eagerly and under graph replay."""
    model = _driven_model(device)
    solver = SolverFeatherPGS(model, update_mass_matrix_interval=100)
    state, output = model.state(), model.state()
    control = model.control()
    solver.step(state, output, control, None, 0.01)

    mask = wp.array([True, False, False], dtype=wp.bool, device=device)
    solver.reset(state, mask)
    solver.step(state, output, control, None, 0.01)
    np.testing.assert_array_equal(solver.mass_update_mask.numpy(), [1, 0, 0])

    first_mask = wp.empty_like(solver.mass_update_mask)
    with wp.ScopedCapture(device=device) as capture:
        solver.reset(state, mask)
        solver.step(state, output, control, None, 0.01)
        wp.copy(first_mask, solver.mass_update_mask)
        solver.step(state, output, control, None, 0.01)

    for selected, expected in (([False, True, False], [0, 1, 0]), ([False, False, True], [0, 0, 1])):
        with test.subTest(mask=selected):
            mask.assign(np.array(selected))
            wp.capture_launch(capture.graph)
            np.testing.assert_array_equal(first_mask.numpy(), expected)
            # The request is consumed by the first step; the second reuses every factor.
            np.testing.assert_array_equal(solver.mass_update_mask.numpy(), [0, 0, 0])


class TestFeatherPGSSafety(unittest.TestCase):
    pass


add_function_test(
    TestFeatherPGSSafety,
    "test_partial_reset_refreshes_only_selected_mass_factors",
    test_partial_reset_refreshes_only_selected_mass_factors,
    devices=get_cuda_test_devices(),
)


if __name__ == "__main__":
    unittest.main()
