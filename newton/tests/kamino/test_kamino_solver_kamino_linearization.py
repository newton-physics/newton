# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Test the public Kamino body-velocity linearization API."""

import unittest

import numpy as np
import warp as wp

import newton
from newton.tests.kamino import setup_tests, test_context
from newton.tests.utils.basics import build_boxes_fourbar


class TestSolverKaminoBodyVelocityLinearization(unittest.TestCase):
    """Verify public multi-RHS body-velocity evaluation."""

    def setUp(self):
        """Initialize the shared Kamino test device."""
        if not test_context.setup_done:
            setup_tests(clear_cache=False)
        self.device = wp.get_device(test_context.device)

    def tearDown(self):
        """Release the test device reference."""
        self.device = None

    def test_linearization_preserves_rhs_linearity(self):
        """Map a summed input basis to the sum of its body-twist responses."""
        builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
        newton.solvers.SolverKamino.register_custom_attributes(builder)
        build_boxes_fourbar(
            builder=builder,
            floatingbase=True,
            ground=False,
            limits=False,
            actuator_ids=[1],
        )
        model = builder.finalize(device=self.device, skip_validation_joints=True)
        config = newton.solvers.SolverKamino.Config.from_model(model, dynamics_solver="dvi")
        config.use_fk_solver = True
        solver = newton.solvers.SolverKamino(model, config=config)
        state = model.state()
        solver.reset(state)

        actuator_u = wp.array([[0.0], [0.7], [0.7]], dtype=wp.float32, device=self.device)
        base_u = wp.array(
            [
                [[0.2, 0.0, -0.1, 0.0, 0.3, 0.0]],
                [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]],
                [[0.2, 0.0, -0.1, 0.0, 0.3, 0.0]],
            ],
            dtype=wp.spatial_vectorf,
            device=self.device,
        )
        body_u = wp.zeros((3, model.body_count), dtype=wp.spatial_vectorf, device=self.device)

        solver.eval_body_velocity_linearization(state, actuator_u, base_u, body_u)

        result = body_u.numpy()
        np.testing.assert_allclose(result[2], result[0] + result[1], rtol=2.0e-4, atol=2.0e-4)
        self.assertGreater(float(np.max(np.abs(result))), 1.0e-3)


if __name__ == "__main__":
    setup_tests()
    unittest.main(verbosity=2)
