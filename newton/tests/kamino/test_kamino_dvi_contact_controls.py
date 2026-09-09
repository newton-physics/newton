# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Tests for Kamino DVI contact compliance and stabilization controls."""

from __future__ import annotations

import unittest

import numpy as np
import warp as wp

import newton._src.solvers.kamino.config as kamino_config
from newton._src.solvers.kamino._src.core.model import ModelKamino
from newton._src.solvers.kamino._src.dynamics.dual import (
    DualProblem,
    DualProblemConfigStruct,
    _build_free_velocity_bias_contacts,
)
from newton._src.solvers.kamino._src.linalg import LLTBlockedSolver
from newton._src.solvers.kamino._src.solvers.common import WarmStartMode
from newton._src.solvers.kamino._src.solvers.dvi import DVISolver
from newton._src.solvers.kamino._src.solvers.metrics import SolutionMetrics
from newton._src.solvers.kamino.solver_kamino import SolverKamino
from newton.tests.kamino import setup_tests, test_context
from newton.tests.kamino.utils.make import make_containers, update_containers
from newton.tests.utils import basics


class TestDVIContactControls(unittest.TestCase):
    """Exercise contact-law controls independently of the DVI scheduler."""

    def setUp(self):
        if not test_context.setup_done:
            setup_tests(clear_cache=False)
        self.device = wp.get_device(test_context.device)

    def _run_contact_bias(
        self,
        distance: float,
        *,
        dt: float = 0.1,
        delta: float = 0.01,
        gamma: float = 0.25,
        stabilization_time: float | None = None,
        recovery_speed: float | None = None,
        compliance: float = 0.0,
        restitution: float = 0.0,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Build one contact's bias, impact, and compliance terms."""
        constraints = kamino_config.ConstraintStabilizationConfig(
            delta=delta,
            gamma=gamma,
            contact_compliance=compliance,
            contact_stabilization_time=stabilization_time,
            contact_recovery_speed=recovery_speed,
        )
        config = wp.array(
            [DualProblem.Config(constraints=constraints).to_struct()],
            dtype=DualProblemConfigStruct,
            device=self.device,
        )

        def ints(values: list[int]) -> wp.array:
            return wp.array(values, dtype=wp.int32, device=self.device)

        def floats(values: list[float]) -> wp.array:
            return wp.array(values, dtype=wp.float32, device=self.device)

        v_b = floats([0.0, 0.0, 0.0])
        v_i = floats([0.0, 0.0, 0.0])
        mu = floats([0.0])
        E = floats([0.0, 0.0, 0.0])
        wp.launch(
            kernel=_build_free_velocity_bias_contacts,
            dim=1,
            inputs=[
                floats([1.0 / dt]),
                ints([0]),
                ints([0]),
                1,
                ints([1]),
                ints([0]),
                ints([0]),
                wp.array([wp.vec4f(0.0, 0.0, 1.0, distance)], dtype=wp.vec4f, device=self.device),
                wp.array([wp.vec2f(0.7, restitution)], dtype=wp.vec2f, device=self.device),
                config,
                ints([0]),
                v_b,
                v_i,
                mu,
                E,
            ],
            device=self.device,
        )
        return v_b.numpy(), v_i.numpy(), E.numpy()

    def _solve_compliant_sphere(self, sparse: bool) -> tuple[ModelKamino, DualProblem, DVISolver]:
        """Build and solve a one-contact compliant sphere problem."""
        builder = basics.build_sphere_on_plane(friction=0.0, restitution=0.0)
        model = ModelKamino.from_newton(builder.finalize(device=self.device))
        model, data, state, limits, detector, jacobians = make_containers(
            model=model,
            max_world_contacts=2,
            sparse=sparse,
            dt=0.01,
        )
        update_containers(
            model=model,
            data=data,
            state=state,
            limits=limits,
            detector=detector,
            jacobians=jacobians,
        )
        self.assertEqual(int(detector.contacts.model_active_contacts.numpy()[0]), 1)

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
                    delta=0.0,
                    contact_compliance=2.0e-4,
                    contact_stabilization_time=0.02,
                ),
                dynamics=kamino_config.ConstrainedDynamicsConfig(preconditioning=False),
            ),
        )
        problem.build(
            model=model,
            data=data,
            limits=limits,
            contacts=detector.contacts,
            jacobians=jacobians,
        )

        solver = DVISolver(
            model=model,
            data=data,
            limits=limits,
            contacts=detector.contacts,
            jacobians=jacobians,
            config=kamino_config.DVISolverConfig(
                coupling_iterations=4,
                limit_pgs_sweeps=1,
                contact_pgs_sweeps=1,
                tolerance=1.0e-5,
                regularization=1.0e-8,
            ),
            warmstart=WarmStartMode.NONE,
        )
        solver.reset()
        solver.coldstart()
        solver.solve(problem)
        return model, problem, solver

    def test_00_config_defaults_validation_and_padmm_rejection(self):
        """Preserve rigid defaults and reject ambiguous or unsupported settings."""
        constraints = kamino_config.ConstraintStabilizationConfig()
        self.assertEqual(constraints.contact_compliance, 0.0)
        self.assertIsNone(constraints.contact_stabilization_time)
        self.assertIsNone(constraints.contact_recovery_speed)

        invalid_settings = (
            {"contact_compliance": -1.0},
            {"contact_compliance": float("nan")},
            {"contact_compliance": float("inf")},
            {"contact_compliance": True, "contact_stabilization_time": 0.01},
            {"contact_stabilization_time": -1.0},
            {"contact_stabilization_time": float("nan")},
            {"contact_stabilization_time": float("inf")},
            {"contact_stabilization_time": True},
            {"contact_recovery_speed": 0.0},
            {"contact_recovery_speed": -1.0},
            {"contact_recovery_speed": float("nan")},
            {"contact_recovery_speed": float("inf")},
            {"contact_recovery_speed": True},
            {"contact_compliance": 1.0e-4},
        )
        for settings in invalid_settings:
            with self.subTest(settings=settings), self.assertRaises(ValueError):
                kamino_config.ConstraintStabilizationConfig(**settings)

        compliant = kamino_config.ConstraintStabilizationConfig(
            contact_compliance=1.0e-4,
            contact_stabilization_time=0.01,
        )
        with self.assertRaisesRegex(ValueError, "supported only by the DVI solver"):
            SolverKamino.Config(dynamics_solver="padmm", constraints=compliant)
        SolverKamino.Config(dynamics_solver="dvi", constraints=compliant)

    def test_01_legacy_gamma_rule_is_unchanged(self):
        """Keep the existing asymmetric gamma rule when no time is selected."""
        penetrating, _, _ = self._run_contact_bias(-0.05)
        speculative, _, _ = self._run_contact_bias(0.05)
        np.testing.assert_allclose(penetrating, [0.0, 0.0, -0.1], rtol=0.0, atol=1.0e-7)
        np.testing.assert_allclose(speculative, [0.0, 0.0, 0.4], rtol=0.0, atol=1.0e-7)

    def test_02_time_stabilization_recovery_clamp_and_restitution_gate(self):
        """Apply alpha-time to signed distance and clamp only recovery velocity."""
        penetrating, _, _ = self._run_contact_bias(-0.05, stabilization_time=0.1)
        clamped, _, _ = self._run_contact_bias(-0.05, stabilization_time=0.1, recovery_speed=0.15)
        speculative, _, _ = self._run_contact_bias(0.05, stabilization_time=0.1, recovery_speed=0.15)
        restitutive, impact, _ = self._run_contact_bias(
            -0.05,
            stabilization_time=0.1,
            recovery_speed=0.15,
            restitution=1.2,
        )

        self.assertAlmostEqual(float(penetrating[2]), -0.2, places=6)
        self.assertAlmostEqual(float(clamped[2]), -0.15, places=6)
        self.assertAlmostEqual(float(speculative[2]), 0.2, places=6)
        self.assertEqual(float(restitutive[2]), 0.0)
        self.assertAlmostEqual(float(impact[2]), 1.2, places=6)

    def test_03_contact_compliance_formula_is_isotropic(self):
        """Assemble E=compliance/[dt*(dt+alpha_time)] on all contact rows."""
        _, _, E = self._run_contact_bias(
            -0.05,
            dt=0.1,
            stabilization_time=0.2,
            compliance=0.02,
        )
        expected = 0.02 / (0.1 * (0.1 + 0.2))
        np.testing.assert_allclose(E, expected, rtol=1.0e-6, atol=0.0)

    def test_04_dense_sparse_solve_uses_effective_but_exports_physical_velocity(self):
        """Solve with N+E while keeping exported v_plus equal to N lambda+v_f."""
        results: dict[bool, tuple[np.ndarray, np.ndarray]] = {}
        expected_E = 2.0e-4 / (0.01 * (0.01 + 0.02))

        for sparse in (False, True):
            with self.subTest(sparse=sparse):
                model, problem, solver = self._solve_compliant_sphere(sparse)
                dim = int(problem.data.dim.numpy()[0])
                vio = int(problem.data.vio.numpy()[0])
                ccgo = int(problem.data.ccgo.numpy()[0])
                contact_slice = slice(vio + ccgo, vio + ccgo + 3)
                lambdas = solver.data.solution.lambdas.numpy()
                v_plus = solver.data.solution.v_plus.numpy()
                v_effective = solver.data.state.v_aug.numpy()
                E = problem.data.E.numpy()
                E_hat = problem.data.E_hat.numpy()
                v_f = problem.data.v_f.numpy()

                np.testing.assert_allclose(E[contact_slice], expected_E, rtol=1.0e-6, atol=0.0)
                np.testing.assert_allclose(E_hat[contact_slice], E[contact_slice], rtol=0.0, atol=0.0)
                np.testing.assert_allclose(
                    v_effective[vio : vio + dim],
                    v_plus[vio : vio + dim] + E_hat[vio : vio + dim] * lambdas[vio : vio + dim],
                    rtol=1.0e-6,
                    atol=1.0e-7,
                )

                if sparse:
                    N_lambda = wp.empty_like(problem.data.v_f)
                    problem.delassus.matvec(
                        x=solver.data.solution.lambdas,
                        y=N_lambda,
                        world_mask=wp.ones(1, dtype=wp.bool, device=self.device),
                    )
                    physical_expected = N_lambda.numpy() + v_f
                else:
                    N = problem.data.D.numpy()[: dim * dim].reshape(dim, dim)
                    physical_expected = N @ lambdas[vio : vio + dim] + v_f[vio : vio + dim]
                np.testing.assert_allclose(
                    v_plus[vio : vio + dim],
                    physical_expected[vio : vio + dim] if sparse else physical_expected,
                    rtol=1.0e-6,
                    atol=1.0e-7,
                )

                normal = contact_slice.stop - 1
                self.assertGreater(float(lambdas[normal]), 0.0)
                self.assertLess(float(v_plus[normal]), -1.0e-3)
                self.assertAlmostEqual(float(v_effective[normal]), 0.0, delta=1.0e-6)
                self.assertEqual(int(solver.data.status.numpy()[0]["converged"]), 1)

                metrics = SolutionMetrics(model=model)
                metrics.reset()
                metrics._evaluate_dual_problem_perf(
                    solver.data.state.sigma,
                    solver.data.solution.lambdas,
                    solver.data.solution.v_plus,
                    problem,
                )
                self.assertLessEqual(float(metrics.data.r_v_plus.numpy()[0]), 1.0e-7)
                self.assertLessEqual(float(metrics.data.r_ncp_dual.numpy()[0]), 1.0e-6)
                self.assertLessEqual(float(metrics.data.r_ncp_compl.numpy()[0]), 1.0e-6)
                results[sparse] = (lambdas[vio : vio + dim], v_plus[vio : vio + dim])

        np.testing.assert_allclose(results[False][0], results[True][0], rtol=1.0e-6, atol=1.0e-7)
        np.testing.assert_allclose(results[False][1], results[True][1], rtol=1.0e-6, atol=1.0e-7)

    def test_05_preconditioned_compliance_has_one_contact_scale(self):
        """Represent contact compliance as P E P with one triplet preconditioner."""
        for sparse in (False, True):
            with self.subTest(sparse=sparse):
                builder = basics.build_sphere_on_plane(friction=0.0, restitution=0.0)
                model = ModelKamino.from_newton(builder.finalize(device=self.device))
                model, data, state, limits, detector, jacobians = make_containers(
                    model=model,
                    max_world_contacts=2,
                    sparse=sparse,
                    dt=0.01,
                )
                update_containers(model, data, state, limits, detector, jacobians)
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
                            contact_compliance=2.0e-4,
                            contact_stabilization_time=0.02,
                        ),
                        dynamics=kamino_config.ConstrainedDynamicsConfig(preconditioning=True),
                    ),
                )
                problem.build(model, data, jacobians, limits, detector.contacts)

                vio = int(problem.data.vio.numpy()[0])
                ccgo = int(problem.data.ccgo.numpy()[0])
                contact_slice = slice(vio + ccgo, vio + ccgo + 3)
                E = problem.data.E.numpy()[contact_slice]
                E_hat = problem.data.E_hat.numpy()[contact_slice]
                P = problem.data.P.numpy()[contact_slice]
                np.testing.assert_allclose(P, P[0], rtol=0.0, atol=0.0)
                np.testing.assert_allclose(E_hat, P * P * E, rtol=1.0e-6, atol=0.0)


if __name__ == "__main__":
    unittest.main()
