# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Tests for LOX structural assembly and convergence validation."""

import unittest
from unittest import mock

import numpy as np
import warp as wp

from newton import StateFlags
from newton._src.solvers.kamino._src.core.model import ModelKamino
from newton._src.solvers.kamino._src.core.state import StateKamino
from newton._src.solvers.kamino._src.core.types import vec6f
from newton._src.solvers.kamino._src.solver_kamino_impl import SolverKaminoImpl
from newton._src.solvers.kamino._src.solvers.lox.system import BatchedPrimalBodySystem
from newton._src.solvers.kamino._src.solvers.lox.types import SplittingState
from newton._src.solvers.kamino.tests import setup_tests as setup_internal_tests
from newton.solvers import SolverKamino
from newton.tests.kamino import setup_tests, test_context
from newton.tests.utils.basics import build_box_on_plane


class TestLOXState(unittest.TestCase):
    def setUp(self):
        if not test_context.setup_done:
            setup_tests(clear_cache=False)
        self.device = wp.get_device(test_context.device)

    def test_internal_setup_preserves_backward_generation(self):
        """Preserve global backward generation when initializing internal Kamino tests."""
        with mock.patch.object(wp.config, "enable_backward", True):
            setup_internal_tests(device=self.device, clear_cache=False)
            self.assertTrue(wp.config.enable_backward)

    def test_consensus_impulse_follows_native_state(self):
        """Copy and warm-start consensus impulses through native Kamino states."""
        model = ModelKamino.from_newton(build_box_on_plane(ground=False).finalize(device=self.device))
        state_in, state_out = model.state(), model.state()
        state_in.lambda_w_i.fill_(3.0)
        state_out.copy_from(state_in)
        np.testing.assert_array_equal(state_out.lambda_w_i.numpy(), state_in.lambda_w_i.numpy())
        solver = SolverKaminoImpl(model, config=SolverKamino.Config(dynamics_solver="lox"))
        backend = solver.solver_fd
        solve = backend.solve

        def check_warmstart(**kwargs):
            np.testing.assert_array_equal(backend.splitting.splitting_dual_impulse.numpy(), state_in.lambda_w_i.numpy())
            solve(**kwargs)

        backend.splitting.splitting_dual_impulse.fill_(17.0)
        with mock.patch.object(backend, "solve", side_effect=check_warmstart):
            solver.step(state_in, state_out, model.control(), dt=0.01)
        np.testing.assert_array_equal(state_in.lambda_w_i.numpy(), np.full((1, 6), 3.0))
        np.testing.assert_array_equal(state_out.lambda_w_i.numpy(), np.zeros((1, 6)))

    def test_consensus_impulse_alias_and_selective_reset(self):
        """Alias Newton consensus history and preserve unselected reset fields and worlds."""
        builder = build_box_on_plane(ground=False)
        build_box_on_plane(builder=builder, ground=False)
        model = builder.finalize(device=self.device)
        solver = SolverKamino(model, SolverKamino.Config(dynamics_solver="lox"))
        state = model.state()
        adapted = StateKamino.from_newton(solver._model_kamino.size, model, state)
        self.assertIs(adapted.lambda_w_i, state.body_lox_dual_impulse)
        state.body_lox_dual_impulse.fill_(3.0)
        mask = wp.array([True, False, False], dtype=wp.bool, device=self.device)
        solver.reset(state, world_mask=mask, flags=StateFlags.BODY_Q)
        np.testing.assert_array_equal(state.body_lox_dual_impulse.numpy(), np.full((2, 6), 3.0))
        solver.reset(state, world_mask=mask, flags=StateFlags.BODY_QD)
        np.testing.assert_array_equal(state.body_lox_dual_impulse.numpy(), [[0.0] * 6, [3.0] * 6])

    def test_invalid_structural_world_clears_penalty(self):
        """Discard stale penalties and contributions from invalid structural worlds."""
        system = BatchedPrimalBodySystem([1], device=self.device)
        penalty = wp.full(3, 7.0, dtype=wp.float32, device=self.device)
        jacobian = wp.array([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0]] * 3, dtype=vec6f, device=self.device)
        system.add_structural_rows(
            row_world=wp.array([-1, 1, 0], dtype=wp.int32, device=self.device),
            body_first=wp.zeros(3, dtype=wp.int32, device=self.device),
            body_second=wp.full(3, -1, dtype=wp.int32, device=self.device),
            jacobian_first=jacobian,
            jacobian_second=wp.zeros(3, dtype=vec6f, device=self.device),
            residual=wp.ones(3, dtype=wp.float32, device=self.device),
            reaction=wp.zeros(3, dtype=wp.float32, device=self.device),
            effective_mass=wp.ones(3, dtype=wp.float32, device=self.device),
            linearization_twist=wp.zeros(1, dtype=vec6f, device=self.device),
            time_step=wp.ones(1, dtype=wp.float32, device=self.device),
            joint_penalty_scale=wp.full(1, 2.0, dtype=wp.float32, device=self.device),
            penalty=penalty,
        )
        np.testing.assert_array_equal(penalty.numpy(), [0.0, 0.0, 2.0])
        expected_matrix = np.zeros((6, 6), dtype=np.float32)
        expected_matrix[0, 0] = 2.0
        np.testing.assert_array_equal(system.smooth_matrix.numpy().reshape(6, 6), expected_matrix)
        np.testing.assert_array_equal(system.right_hand_side.numpy(), [-2.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    def test_invalid_convergence_tolerances_preserve_state(self):
        """Reject invalid tolerances before updating iteration state."""
        state = SplittingState([1], device=self.device)
        status = wp.zeros(1, dtype=wp.int32, device=self.device)
        time_step = wp.ones(1, dtype=wp.float32, device=self.device)
        for name in ("position_tolerance", "rotation_tolerance", "velocity_tolerance"):
            for value in (0.0, -1.0, float("nan"), float("inf")):
                with self.subTest(tolerance=name, value=value):
                    tolerances = {
                        "position_tolerance": 1.0e-5,
                        "rotation_tolerance": 1.0e-5,
                        "velocity_tolerance": 1.0e-5,
                    }
                    tolerances[name] = value
                    with self.assertRaisesRegex(ValueError, "Convergence tolerances"):
                        state.finish_iteration(status, time_step, **tolerances)
                    np.testing.assert_array_equal(state.iteration_count.numpy(), [0])
                    np.testing.assert_array_equal(state.world_active.numpy(), [True])


if __name__ == "__main__":
    unittest.main(verbosity=2)
