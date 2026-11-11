# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the decoupling of joint friction and effort rows from joint dynamics rows."""

import unittest

import numpy as np
import warp as wp

import newton
from newton._src.solvers.kamino._src.core.model import ModelKamino
from newton._src.solvers.kamino._src.dynamics.delassus import _set_decoupled_coupled_rows
from newton._src.solvers.kamino._src.dynamics.joint_decoupling import DynamicJointDecoupling
from newton._src.solvers.kamino.solver_kamino import SolverKamino
from newton.tests.kamino import setup_tests, test_context
from newton.tests.kamino.utils.make import make_containers


def _build_decoupled_model(num_joints: int = 3) -> ModelKamino:
    """Build one world with revolute joints that each have a joint dynamics, friction and effort row."""
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    SolverKamino.register_custom_attributes(builder)
    parent = -1
    joints = []
    for _ in range(num_joints):
        body = builder.add_link(mass=1.0, inertia=wp.mat33(np.eye(3)), lock_inertia=True)
        joints.append(
            builder.add_joint_revolute(
                parent,
                body,
                axis=newton.Axis.Z,
                armature=0.1,
                friction=0.5,
                target_ke=10.0,
                target_kd=1.0,
                effort_limit=2.0,
                actuator_mode=newton.JointTargetMode.POSITION,
            )
        )
        parent = body
    builder.add_articulation(joints)
    model = ModelKamino.from_newton(builder.finalize(device=test_context.device))
    # Constructs the constraint offsets of the model
    model, *_ = make_containers(model=model)
    return model


class TestDynamicJointDecoupling(unittest.TestCase):
    """The change of variables of :class:`DynamicJointDecoupling` and its uses."""

    def setUp(self):
        """Initialize the shared public Kamino test context."""
        if not test_context.setup_done:
            setup_tests(clear_cache=False)
        self.model = _build_decoupled_model()
        self.decoupling = DynamicJointDecoupling(self.model)
        self.table = self.decoupling.dofs.numpy()
        self.num_rows = self.model.size.sum_of_max_total_cts
        self.rng = np.random.default_rng(3)

    def _shares(self, low: float, high: float) -> np.ndarray:
        return self.rng.uniform(low, high, self.decoupling.num_dofs).astype(np.float32)

    def test_dof_table(self):
        """Each joint DoF pairs its joint dynamics row with its friction and effort rows."""
        self.assertEqual(self.decoupling.num_dofs, 3)
        rows = np.concatenate([self.table["dynamic_row"], self.table["friction_row"], self.table["effort_row"]])
        self.assertTrue(np.all(rows >= 0))
        self.assertEqual(np.unique(rows).size, rows.size)

    def test_rebase_solution_preserves_original_variables(self):
        """Rebasing a solution to a new body share leaves the original impulses and velocities unchanged."""
        device = self.model.device
        s_old = self._shares(0.1, 0.9)
        s_new = self._shares(0.1, 0.9)
        lambdas = self.rng.standard_normal(self.num_rows).astype(np.float32)
        v_plus = self.rng.standard_normal(self.num_rows).astype(np.float32)
        self.decoupling.solution_body_share.assign(s_old)
        self.decoupling.body_share.assign(s_new)
        lambdas_wp = wp.array(lambdas, dtype=wp.float32, device=device)
        v_plus_wp = wp.array(v_plus, dtype=wp.float32, device=device)

        self.decoupling.rebase_solution(lambdas_wp, v_plus_wp)

        rebased_lambdas = lambdas_wp.numpy()
        rebased_v_plus = v_plus_wp.numpy()
        q, f, e = self.table["dynamic_row"], self.table["friction_row"], self.table["effort_row"]
        c_old, c_new = s_old - 1.0, s_new - 1.0
        # Original impulses lambda_q = lambda_q' + c (lambda_f + lambda_e), and velocities v_f = v_f' - c v_q
        np.testing.assert_allclose(
            rebased_lambdas[q] + c_new * (rebased_lambdas[f] + rebased_lambdas[e]),
            lambdas[q] + c_old * (lambdas[f] + lambdas[e]),
            rtol=1e-5,
            atol=1e-6,
        )
        for row in (f, e):
            np.testing.assert_allclose(
                rebased_v_plus[row] - c_new * rebased_v_plus[q], v_plus[row] - c_old * v_plus[q], rtol=1e-5, atol=1e-6
            )
        # The friction and effort impulses, the joint dynamics velocities, and all other rows are unchanged
        unchanged = np.ones(self.num_rows, dtype=bool)
        unchanged[q] = False
        np.testing.assert_array_equal(rebased_lambdas[unchanged], lambdas[unchanged])
        unchanged = np.ones(self.num_rows, dtype=bool)
        unchanged[f] = False
        unchanged[e] = False
        np.testing.assert_array_equal(rebased_v_plus[unchanged], v_plus[unchanged])
        # The solution is now in the variables of the current body share
        np.testing.assert_array_equal(self.decoupling.solution_body_share.numpy(), s_new)

    def test_scale_rows_dense(self):
        """Scaling the decoupled rows and columns of a dense matrix gives ``S D S``."""
        device = self.model.device
        n = self.model.size.max_of_max_total_cts
        D = self.rng.standard_normal((n, n)).astype(np.float32)
        s = self._shares(0.1, 0.9)
        self.decoupling.body_share.assign(s)
        D_wp = wp.array(D.ravel(), dtype=wp.float32, device=device)
        dim = wp.array([n], dtype=wp.int32, device=device)
        offset = wp.zeros(1, dtype=wp.int32, device=device)

        self.decoupling.scale_rows_dense(dim, offset, offset, n, D_wp)

        scale = np.ones(n, dtype=np.float32)
        scale[self.table["friction_row"]] = s
        scale[self.table["effort_row"]] = s
        np.testing.assert_allclose(D_wp.numpy().reshape(n, n), scale[:, None] * D * scale[None, :], rtol=1e-6)

    def test_coupled_rows_with_zero_body_share(self):
        """A zero body share, from a zero joint compliance, gives finite zero couplings."""
        device = self.model.device
        self.decoupling.body_share.zero_()
        inv_m_j = wp.zeros(self.model.size.sum_of_num_dynamic_joint_cts, dtype=wp.float32, device=device)
        coupled_row = wp.full((self.num_rows, 2), -1, dtype=wp.int32, device=device)
        coupled_coef = wp.full((self.num_rows, 2), 1.0, dtype=wp.float32, device=device)
        wp.launch(
            _set_decoupled_coupled_rows,
            dim=self.decoupling.num_dofs,
            inputs=[
                self.decoupling.dofs,
                self.decoupling.body_share,
                inv_m_j,
                wp.zeros(1, dtype=wp.int32, device=device),
                None,
            ],
            outputs=[coupled_row, coupled_coef],
            device=device,
        )
        rows = np.concatenate([self.table["dynamic_row"], self.table["friction_row"], self.table["effort_row"]])
        np.testing.assert_array_equal(coupled_coef.numpy()[rows], 0.0)


if __name__ == "__main__":
    # Test setup
    setup_tests()

    # Run all tests
    unittest.main(verbosity=2)
