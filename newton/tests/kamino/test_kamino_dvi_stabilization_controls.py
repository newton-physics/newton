# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Tests for Kamino DVI joint and joint-limit stabilization controls."""

from __future__ import annotations

import unittest

import numpy as np
import warp as wp

import newton._src.solvers.kamino.config as kamino_config
from newton._src.solvers.kamino._src.core.model import ModelKamino
from newton._src.solvers.kamino._src.dynamics.dual import (
    DualProblem,
    DualProblemConfigStruct,
    _build_free_velocity_bias_joint_kinematics,
    _build_free_velocity_bias_limits,
)
from newton._src.solvers.kamino._src.linalg import LLTBlockedSolver
from newton._src.solvers.kamino.solver_kamino import SolverKamino
from newton.tests.kamino import setup_tests, test_context
from newton.tests.kamino.utils.extract import extract_problem_vector
from newton.tests.kamino.utils.make import make_containers, update_containers
from newton.tests.utils.testing import build_unary_revolute_joint_test


class TestDVIStabilizationControls(unittest.TestCase):
    """Exercise joint-family stabilization independently of the DVI scheduler."""

    def setUp(self):
        if not test_context.setup_done:
            setup_tests(clear_cache=False)
        self.device = wp.get_device(test_context.device)

    def _config(self, **kwargs: float | None) -> wp.array:
        constraints = kamino_config.ConstraintStabilizationConfig(**kwargs)
        return wp.array(
            [DualProblem.Config(constraints=constraints).to_struct()],
            dtype=DualProblemConfigStruct,
            device=self.device,
        )

    def _ints(self, values: list[int]) -> wp.array:
        return wp.array(values, dtype=wp.int32, device=self.device)

    def _floats(self, values: list[float]) -> wp.array:
        return wp.array(values, dtype=wp.float32, device=self.device)

    def _run_joint_bias(
        self,
        errors: list[float],
        *,
        dt: float = 0.1,
        alpha: float = 0.25,
        compliance: float = 0.0,
        stabilization_time: float | None = None,
        recovery_speed: float | None = None,
        row_offset: int = 0,
        output_size: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        if output_size is None:
            output_size = row_offset + len(errors)
        bias = self._floats([0.0] * output_size)
        compliance_diagonal = self._floats([0.0] * output_size)
        wp.launch(
            kernel=_build_free_velocity_bias_joint_kinematics,
            dim=1,
            inputs=[
                self._floats([1.0 / dt]),
                self._ints([0]),
                self._ints([0, len(errors)]),
                self._ints([row_offset]),
                self._floats(errors),
                self._config(
                    alpha=alpha,
                    joint_compliance=compliance,
                    joint_stabilization_time=stabilization_time,
                    joint_recovery_speed=recovery_speed,
                ),
                bias,
                compliance_diagonal,
            ],
            device=self.device,
        )
        return bias.numpy(), compliance_diagonal.numpy()

    def _run_limit_bias(
        self,
        errors: list[float],
        *,
        dt: float = 0.1,
        beta: float = 0.25,
        compliance: float = 0.0,
        stabilization_time: float | None = None,
        recovery_speed: float | None = None,
        active_count: int | None = None,
        row_offset: int = 0,
        output_size: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        max_limits = len(errors)
        if active_count is None:
            active_count = max_limits
        if output_size is None:
            output_size = row_offset + max_limits
        bias = self._floats([0.0] * output_size)
        compliance_diagonal = self._floats([0.0] * output_size)
        wp.launch(
            kernel=_build_free_velocity_bias_limits,
            dim=max_limits,
            inputs=[
                self._floats([1.0 / dt]),
                self._ints([row_offset]),
                max_limits,
                self._ints([active_count]),
                self._ints([0] * max_limits),
                self._ints(list(range(max_limits))),
                self._floats(errors),
                self._config(
                    beta=beta,
                    joint_limit_compliance=compliance,
                    joint_limit_stabilization_time=stabilization_time,
                    joint_limit_recovery_speed=recovery_speed,
                ),
                self._ints([0]),
                bias,
                compliance_diagonal,
            ],
            device=self.device,
        )
        return bias.numpy(), compliance_diagonal.numpy()

    def test_00_defaults_and_validation(self):
        """Keep legacy rules by default and reject invalid physical controls."""
        constraints = kamino_config.ConstraintStabilizationConfig()
        self.assertEqual(constraints.joint_compliance, 0.0)
        self.assertIsNone(constraints.joint_stabilization_time)
        self.assertIsNone(constraints.joint_recovery_speed)
        self.assertEqual(constraints.joint_limit_compliance, 0.0)
        self.assertIsNone(constraints.joint_limit_stabilization_time)
        self.assertIsNone(constraints.joint_limit_recovery_speed)

        invalid_settings = (
            {"joint_compliance": -1.0},
            {"joint_compliance": float("nan")},
            {"joint_compliance": float("inf")},
            {"joint_compliance": True},
            {"joint_stabilization_time": -1.0},
            {"joint_stabilization_time": float("nan")},
            {"joint_stabilization_time": float("inf")},
            {"joint_stabilization_time": True},
            {"joint_recovery_speed": 0.0},
            {"joint_recovery_speed": -1.0},
            {"joint_recovery_speed": float("nan")},
            {"joint_recovery_speed": float("inf")},
            {"joint_recovery_speed": True},
            {"joint_limit_compliance": -1.0},
            {"joint_limit_compliance": float("nan")},
            {"joint_limit_compliance": float("inf")},
            {"joint_limit_compliance": True},
            {"joint_limit_stabilization_time": -1.0},
            {"joint_limit_stabilization_time": float("nan")},
            {"joint_limit_stabilization_time": float("inf")},
            {"joint_limit_stabilization_time": True},
            {"joint_limit_recovery_speed": 0.0},
            {"joint_limit_recovery_speed": -1.0},
            {"joint_limit_recovery_speed": float("nan")},
            {"joint_limit_recovery_speed": float("inf")},
            {"joint_limit_recovery_speed": True},
        )
        for settings in invalid_settings:
            with self.subTest(settings=settings), self.assertRaises(ValueError):
                kamino_config.ConstraintStabilizationConfig(**settings)

        required_times = (
            ("joint_compliance", "joint_stabilization_time"),
            ("joint_limit_compliance", "joint_limit_stabilization_time"),
        )
        for compliance_name, time_name in required_times:
            with (
                self.subTest(compliance_name=compliance_name),
                self.assertRaisesRegex(ValueError, f"Nonzero {compliance_name} requires an explicit {time_name}"),
            ):
                kamino_config.ConstraintStabilizationConfig(**{compliance_name: 1.0e-4})
            compliant = kamino_config.ConstraintStabilizationConfig(**{compliance_name: 1.0e-4, time_name: 0.01})
            SolverKamino.Config(dynamics_solver="dvi", constraints=compliant)
            with self.assertRaisesRegex(ValueError, "supported only by the DVI solver"):
                SolverKamino.Config(dynamics_solver="padmm", constraints=compliant)

    def test_01_bilateral_joint_time_rule_and_symmetric_recovery_clamp(self):
        """Use error/(dt+time) and clamp signed bilateral drift symmetrically."""
        errors = [-0.06, 0.08]
        legacy, legacy_compliance = self._run_joint_bias(errors)
        time_based, _ = self._run_joint_bias(errors, stabilization_time=0.1)
        clamped, _ = self._run_joint_bias(errors, stabilization_time=0.1, recovery_speed=0.2)

        np.testing.assert_allclose(legacy, [-0.15, 0.2], rtol=0.0, atol=1.0e-7)
        np.testing.assert_array_equal(legacy_compliance, np.zeros(2, dtype=np.float32))
        np.testing.assert_allclose(time_based, [-0.3, 0.4], rtol=0.0, atol=1.0e-7)
        np.testing.assert_allclose(clamped, [-0.2, 0.2], rtol=0.0, atol=1.0e-7)

    def test_02_joint_limit_time_rule_and_one_sided_recovery_clamp(self):
        """Correct only active limit violation and clamp only negative recovery."""
        errors = [-0.06, 0.08]
        legacy, legacy_compliance = self._run_limit_bias(errors)
        time_based, _ = self._run_limit_bias(errors, stabilization_time=0.1)
        clamped, _ = self._run_limit_bias(errors, stabilization_time=0.1, recovery_speed=0.2)

        np.testing.assert_allclose(legacy, [-0.15, 0.0], rtol=0.0, atol=1.0e-7)
        np.testing.assert_array_equal(legacy_compliance, np.zeros(2, dtype=np.float32))
        np.testing.assert_allclose(time_based, [-0.3, 0.0], rtol=0.0, atol=1.0e-7)
        np.testing.assert_allclose(clamped, [-0.2, 0.0], rtol=0.0, atol=1.0e-7)

    def test_03_joint_compliance_formula_and_kinematic_row_scope(self):
        """Assemble physical compliance only on the kinematic bilateral rows."""
        _, compliance_diagonal = self._run_joint_bias(
            [-0.06, 0.08],
            dt=0.1,
            compliance=0.02,
            stabilization_time=0.2,
            row_offset=2,
            output_size=6,
        )

        expected = 0.02 / (0.1 * (0.1 + 0.2))
        np.testing.assert_allclose(
            compliance_diagonal,
            [0.0, 0.0, expected, expected, 0.0, 0.0],
            rtol=1.0e-6,
            atol=0.0,
        )

    def test_04_joint_limit_compliance_formula_and_active_row_scope(self):
        """Assemble physical compliance only on active joint-limit rows."""
        _, compliance_diagonal = self._run_limit_bias(
            [-0.06, 0.08, -0.02],
            dt=0.1,
            compliance=0.03,
            stabilization_time=0.05,
            active_count=2,
            row_offset=2,
            output_size=7,
        )

        expected = 0.03 / (0.1 * (0.1 + 0.05))
        np.testing.assert_allclose(
            compliance_diagonal,
            [0.0, 0.0, expected, expected, 0.0, 0.0, 0.0],
            rtol=1.0e-6,
            atol=0.0,
        )

    def test_05_real_model_joint_compliance_excludes_dynamic_actuator_rows(self):
        """Apply joint compliance to kinematic rows, not dynamic actuator rows."""
        dt = 0.01
        compliance = 2.0e-4
        stabilization_time = 0.02
        expected_compliance = compliance / (dt * (dt + stabilization_time))

        for sparse in (False, True):
            with self.subTest(sparse=sparse):
                builder = build_unary_revolute_joint_test(
                    dynamic=True,
                    implicit_pd=True,
                    effort_limit=None,
                    limits=False,
                    ground=False,
                )
                model = ModelKamino.from_newton(builder.finalize(device=self.device))
                model, data, state, limits, detector, jacobians = make_containers(
                    model=model,
                    max_world_contacts=0,
                    sparse=sparse,
                    dt=dt,
                )
                update_containers(
                    model=model,
                    data=data,
                    state=state,
                    limits=limits,
                    detector=None,
                    jacobians=jacobians,
                )

                problem = DualProblem(
                    model=model,
                    data=data,
                    limits=limits,
                    contacts=detector.contacts,
                    jacobians=jacobians,
                    solver=None if sparse else LLTBlockedSolver,
                    sparse=sparse,
                    config=DualProblem.Config(
                        constraints=kamino_config.ConstraintStabilizationConfig(
                            joint_compliance=compliance,
                            joint_stabilization_time=stabilization_time,
                        )
                    ),
                )
                problem.build(
                    model=model,
                    data=data,
                    limits=limits,
                    contacts=detector.contacts,
                    jacobians=jacobians,
                )

                np.testing.assert_array_equal(model.info.num_joint_dynamic_cts.numpy(), [1])
                np.testing.assert_array_equal(model.info.num_joint_kinematic_cts.numpy(), [5])
                np.testing.assert_array_equal(problem.data.njc.numpy(), [6])
                np.testing.assert_array_equal(problem.data.nbc.numpy(), [1])
                np.testing.assert_array_equal(problem.data.nl.numpy(), [0])
                np.testing.assert_array_equal(problem.data.nc.numpy(), [0])

                physical_compliance = extract_problem_vector(
                    problem.delassus,
                    problem.data.E.numpy(),
                    only_active_dims=True,
                )[0]
                np.testing.assert_allclose(
                    physical_compliance,
                    [0.0, *([expected_compliance] * 5), 0.0],
                    rtol=1.0e-6,
                    atol=0.0,
                )


if __name__ == "__main__":
    setup_tests()
    unittest.main(verbosity=2)
