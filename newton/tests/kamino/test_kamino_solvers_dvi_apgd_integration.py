# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Integration tests for the APGD contact backend in Kamino DVI."""

from __future__ import annotations

import unittest

import numpy as np
import warp as wp

import newton._src.solvers.kamino.config as kamino_config
from newton._src.solvers.kamino._src.core.model import ModelKamino
from newton._src.solvers.kamino._src.dynamics.dual import DualProblem
from newton._src.solvers.kamino._src.linalg import LLTBlockedSolver
from newton._src.solvers.kamino._src.solvers.common import WarmStartMode
from newton._src.solvers.kamino._src.solvers.dvi import DVISolver
from newton._src.solvers.kamino._src.solvers.metrics import SolutionMetrics
from newton._src.solvers.kamino.solver_kamino import SolverKamino
from newton.tests.kamino import setup_tests, test_context
from newton.tests.kamino.utils.make import make_containers, update_containers
from newton.tests.utils import basics


class TestDVIAPGDIntegration(unittest.TestCase):
    """Exercise config, DVI dispatch, terminal status, and public stepping."""

    def setUp(self) -> None:
        if not test_context.setup_done:
            setup_tests(clear_cache=False)
        self.device = wp.get_device(test_context.device)

    @staticmethod
    def _apgd_config(**kwargs) -> kamino_config.DVISolverConfig:
        """Make a short but accurate associated-contact solve config."""
        max_alternating_iterations = kwargs.pop("max_alternating_iterations", 1)
        apgd = kamino_config.DVIAPGDConfig(
            max_iterations=100,
            max_backtrack_iterations=20,
            tolerance=1.0e-5,
            min_iterations=1,
            early_exit=True,
            use_graph_conditionals=False,
        )
        return kamino_config.DVISolverConfig(
            contact_solver="apgd",
            max_alternating_iterations=max_alternating_iterations,
            inequality_sweeps_per_iteration=1,
            tolerance=1.0e-4,
            apgd=apgd,
            **kwargs,
        )

    def test_00_config_selects_backend_native_laws_and_rejects_crossed_pairs(self) -> None:
        """Keep PGS/De-Saxce default and make APGD/associated selection explicit."""
        default = kamino_config.DVISolverConfig()
        self.assertEqual(default.contact_solver, "pgs")
        self.assertIsNone(default.contact_law)
        self.assertEqual(default.resolved_contact_law, "de_saxce")
        self.assertEqual(default.apgd.max_iterations, 20)
        self.assertEqual(default.apgd.tolerance, 1.0e-3)

        apgd = self._apgd_config()
        self.assertEqual(apgd.contact_solver, "apgd")
        self.assertIsNone(apgd.contact_law)
        self.assertEqual(apgd.resolved_contact_law, "associated_at")

        default.contact_solver = "apgd"
        default.validate()
        self.assertEqual(default.resolved_contact_law, "associated_at")

        with self.assertRaisesRegex(ValueError, "requires contact_law"):
            kamino_config.DVISolverConfig(contact_solver="apgd", contact_law="de_saxce")
        with self.assertRaisesRegex(ValueError, "requires contact_law"):
            kamino_config.DVISolverConfig(contact_solver="pgs", contact_law="associated_at")
        with self.assertRaises(ValueError):
            kamino_config.DVIAPGDConfig(max_iterations=0)
        with self.assertRaises(ValueError):
            kamino_config.DVIAPGDConfig(max_iterations=2, min_iterations=3)
        with self.assertRaises(ValueError):
            kamino_config.DVIAPGDConfig(max_backtrack_iterations=-1)
        for tolerance in (-1.0, float("nan"), float("inf"), True):
            with self.subTest(tolerance=tolerance), self.assertRaises(ValueError):
                kamino_config.DVIAPGDConfig(tolerance=tolerance)
        for flag_name in ("early_exit", "use_graph_conditionals"):
            with self.subTest(flag_name=flag_name), self.assertRaises(TypeError):
                kamino_config.DVIAPGDConfig(**{flag_name: 1})
        with self.assertRaises(TypeError):
            kamino_config.DVISolverConfig(post_stabilization_bilateral=1)

    def _assert_contact_phase(
        self,
        *,
        sparse: bool,
        solver_config: kamino_config.DVISolverConfig | None = None,
    ) -> None:
        """Check one integrated dense or sparse APGD contact phase."""
        model = ModelKamino.from_newton(
            basics.build_sphere_on_plane(friction=0.5, restitution=0.0).finalize(device=self.device)
        )
        model, data, state, limits, detector, jacobians = make_containers(
            model=model,
            max_world_contacts=2,
            sparse=sparse,
            dt=0.01,
        )
        update_containers(model, data, state, limits, detector, jacobians)
        self.assertEqual(int(detector.contacts.model_active_contacts.numpy()[0]), 1)
        problem_kwargs = {} if sparse else {"solver": LLTBlockedSolver}
        problem = DualProblem(
            model=model,
            data=data,
            limits=limits,
            contacts=detector.contacts,
            jacobians=jacobians,
            sparse=sparse,
            config=DualProblem.Config(
                constraints=kamino_config.ConstraintStabilizationConfig(
                    delta=0.0,
                    contact_compliance=2.0e-4,
                    contact_stabilization_time=0.02,
                ),
                dynamics=kamino_config.ConstrainedDynamicsConfig(preconditioning=False),
            ),
            **problem_kwargs,
        )
        problem.build(model, data, jacobians, limits, detector.contacts)
        solver_config = solver_config or self._apgd_config()
        solver = DVISolver(
            model=model,
            data=data,
            limits=limits,
            contacts=detector.contacts,
            jacobians=jacobians,
            problem=problem,
            config=solver_config,
            warmstart=WarmStartMode.NONE,
            collect_info=True,
        )
        solver.coldstart()
        solver.solve(problem)

        status = solver.data.status.numpy()[0]
        lambdas = solver.data.solution.lambdas.numpy()
        self.assertEqual(int(status["converged"]), 1, msg=str(status))
        self.assertEqual(int(status["iterations"]), 1)
        self.assertGreaterEqual(int(status["contact_iterations"]), 1)
        self.assertLessEqual(int(status["contact_iterations"]), solver.config[0].apgd.max_iterations)
        self.assertGreaterEqual(int(status["contact_backtracks"]), 0)
        self.assertGreaterEqual(int(status["contact_restarts"]), 0)
        self.assertLessEqual(float(status["r_natural"]), solver.config[0].tolerance)
        self.assertLessEqual(float(status["contact_solver_residual"]), solver.config[0].apgd.tolerance)
        self.assertTrue(np.all(np.isfinite(lambdas)))
        np.testing.assert_array_equal(solver.data.state.s.numpy(), np.zeros_like(solver.data.state.s.numpy()))
        np.testing.assert_array_equal(solver.data.info.status.numpy(), solver.data.status.numpy())

        vio = int(problem.data.vio.numpy()[0])
        ccgo = int(problem.data.ccgo.numpy()[0])
        contact_impulse = lambdas[vio + ccgo : vio + ccgo + 3]
        friction = float(problem.data.mu.numpy()[0])
        contact_slice = slice(vio + ccgo, vio + ccgo + 3)
        effective_velocity = solver.data.state.v_aug.numpy()[contact_slice]
        physical_velocity = solver.data.solution.v_plus.numpy()[contact_slice]
        compliance_velocity = problem.data.E_hat.numpy()[contact_slice] * contact_impulse
        self.assertGreaterEqual(float(contact_impulse[2]), 0.0)
        self.assertLessEqual(float(np.linalg.norm(contact_impulse[:2])), friction * float(contact_impulse[2]) + 1.0e-6)
        np.testing.assert_allclose(effective_velocity, physical_velocity + compliance_velocity, atol=1.0e-7)
        self.assertGreater(float(problem.data.E_hat.numpy()[vio + ccgo + 2]), 0.0)

        metrics = SolutionMetrics(model=model)
        metrics.reset()
        metrics._evaluate_dual_problem_perf(
            solver.data.state.sigma,
            solver.data.solution.lambdas,
            solver.data.solution.v_plus,
            problem,
            contact_law="associated_at",
        )
        self.assertLessEqual(float(metrics.data.r_vi_natmap.numpy()[0]), solver.config[0].tolerance)
        np.testing.assert_allclose(metrics.data.f_ncp.numpy(), metrics.data.f_ccp.numpy(), atol=1.0e-7)
        np.testing.assert_array_equal(metrics._buffer_s.numpy(), np.zeros_like(metrics._buffer_s.numpy()))

    def test_01_dense_contact_phase_reports_actual_apgd_work_and_associated_status(self) -> None:
        """Dispatch dense APGD through unified lambda storage with associated status."""
        self._assert_contact_phase(sparse=False)

    def test_02_sparse_contact_phase_reports_actual_apgd_work_and_associated_status(self) -> None:
        """Dispatch sparse APGD through unified lambda storage with associated status."""
        self._assert_contact_phase(sparse=True)

    def test_02a_contact_only_apgd_runs_one_phase_with_default_outer_budget(self) -> None:
        """Avoid redundant contact solves when no other family can change its RHS."""
        for sparse in (False, True):
            with self.subTest(sparse=sparse):
                self._assert_contact_phase(
                    sparse=sparse,
                    solver_config=self._apgd_config(max_alternating_iterations=24),
                )

    def _assert_public_rollout(self, *, sparse: bool) -> None:
        """Run one dense or sparse APGD rollout through the public solver."""
        model = basics.build_sphere_on_plane(friction=0.5, restitution=0.0).finalize(device=self.device)
        config = SolverKamino.Config(
            dynamics_solver="dvi",
            use_collision_detector=True,
            sparse_dynamics=sparse,
            sparse_jacobian=sparse,
            compute_solution_metrics=True,
            dvi=self._apgd_config(),
        )
        solver = SolverKamino(model, config=config)
        state_in = model.state()
        state_out = model.state()
        for _ in range(5):
            solver.step(state_in, state_out, control=None, contacts=None, dt=0.01)
            state_in, state_out = state_out, state_in

        self.assertTrue(np.all(np.isfinite(state_in.body_q.numpy())))
        self.assertTrue(np.all(np.isfinite(state_in.body_qd.numpy())))
        status = solver.status.numpy()[0]
        self.assertEqual(int(status["converged"]), 1, msg=str(status))
        self.assertGreaterEqual(int(status["contact_iterations"]), 1)
        metrics = solver._solver_kamino.metrics
        self.assertIsNotNone(metrics)
        np.testing.assert_allclose(metrics.data.f_ncp.numpy(), metrics.data.f_ccp.numpy(), atol=1.0e-6)

    def test_03_public_dense_solver_apgd_rollout_remains_finite(self) -> None:
        """Run dense APGD through SolverKamino's normal step interface."""
        self._assert_public_rollout(sparse=False)

    def test_04_public_sparse_solver_apgd_rollout_remains_finite(self) -> None:
        """Run sparse APGD through SolverKamino's normal step interface."""
        self._assert_public_rollout(sparse=True)

    def test_05_reusing_solver_rebinds_dense_and_sparse_contact_operators(self) -> None:
        """Rebind matrix, vector, compliance, and sparse regularization storage."""
        for sparse in (False, True):
            for lazy_binding in (False, True):
                with self.subTest(sparse=sparse, lazy_binding=lazy_binding):
                    model = ModelKamino.from_newton(
                        basics.build_sphere_on_plane(friction=0.5, restitution=0.0).finalize(device=self.device)
                    )
                    model, data, _state, limits, detector, jacobians = make_containers(
                        model=model,
                        max_world_contacts=2,
                        sparse=sparse,
                        dt=0.01,
                    )
                    update_containers(model, data, _state, limits, detector, jacobians)
                    problems = []
                    for normal_free_velocity in (1.0, -1.0):
                        problem_kwargs = {} if sparse else {"solver": LLTBlockedSolver}
                        problem = DualProblem(
                            model=model,
                            data=data,
                            limits=limits,
                            contacts=detector.contacts,
                            jacobians=jacobians,
                            sparse=sparse,
                            config=DualProblem.Config(
                                dynamics=kamino_config.ConstrainedDynamicsConfig(preconditioning=False)
                            ),
                            **problem_kwargs,
                        )
                        problem.build(model, data, jacobians, limits, detector.contacts)
                        normal_row = int(problem.data.vio.numpy()[0]) + int(problem.data.ccgo.numpy()[0]) + 2
                        free_velocity = problem.data.v_f.numpy()
                        free_velocity[normal_row] = normal_free_velocity
                        problem.data.v_f.assign(free_velocity)
                        if normal_free_velocity < 0.0:
                            compliance = problem.data.E.numpy()
                            represented_compliance = problem.data.E_hat.numpy()
                            compliance[normal_row] = 0.25
                            represented_compliance[normal_row] = 0.25
                            problem.data.E.assign(compliance)
                            problem.data.E_hat.assign(represented_compliance)
                        problems.append(problem)

                    solver = DVISolver(
                        model=model,
                        data=data,
                        limits=limits,
                        contacts=detector.contacts,
                        jacobians=jacobians,
                        problem=None if lazy_binding else problems[0],
                        config=self._apgd_config(),
                        warmstart=WarmStartMode.NONE,
                    )
                    if lazy_binding:
                        solver.coldstart()
                        solver.solve(problems[0])
                        self.assertAlmostEqual(float(solver.data.solution.lambdas.numpy()[normal_row]), 0.0)

                    solver.coldstart()
                    solver.solve(problems[1])
                    status = solver.data.status.numpy()[0]
                    self.assertEqual(int(status["converged"]), 1, msg=str(status))
                    self.assertGreater(float(solver.data.solution.lambdas.numpy()[normal_row]), 0.1)
                    self.assertLessEqual(float(status["r_d"]), solver.config[0].tolerance)
                    reused_solution = solver.data.solution.lambdas.numpy().copy()

                    fresh_solver = DVISolver(
                        model=model,
                        data=data,
                        limits=limits,
                        contacts=detector.contacts,
                        jacobians=jacobians,
                        problem=problems[1],
                        config=self._apgd_config(),
                        warmstart=WarmStartMode.NONE,
                    )
                    fresh_solver.coldstart()
                    fresh_solver.solve(problems[1])
                    np.testing.assert_allclose(
                        reused_solution,
                        fresh_solver.data.solution.lambdas.numpy(),
                        rtol=0.0,
                        atol=2.0e-6,
                    )

                    if sparse:
                        # Public Delassus setters may replace the array object while
                        # the DualProblem identity remains unchanged.
                        regularization = np.zeros(problems[1].delassus.regularization.shape, dtype=np.float32)
                        regularization[normal_row] = 0.75
                        problems[1].delassus.set_regularization(
                            wp.array(regularization, dtype=wp.float32, device=self.device)
                        )
                        solver.coldstart()
                        solver.solve(problems[1])
                        rebound_solution = solver.data.solution.lambdas.numpy().copy()

                        regularized_fresh_solver = DVISolver(
                            model=model,
                            data=data,
                            limits=limits,
                            contacts=detector.contacts,
                            jacobians=jacobians,
                            problem=problems[1],
                            config=self._apgd_config(),
                            warmstart=WarmStartMode.NONE,
                        )
                        regularized_fresh_solver.coldstart()
                        regularized_fresh_solver.solve(problems[1])
                        np.testing.assert_allclose(
                            rebound_solution,
                            regularized_fresh_solver.data.solution.lambdas.numpy(),
                            rtol=0.0,
                            atol=2.0e-6,
                        )
                        self.assertGreater(
                            abs(float(rebound_solution[normal_row] - reused_solution[normal_row])),
                            1.0e-3,
                        )

    def test_05a_default_pgs_retains_de_saxce_terminal_law(self) -> None:
        """Treat an omitted PGS law exactly like explicit De Saxce on sliding contact."""
        for sparse in (False, True):
            with self.subTest(sparse=sparse):
                model = ModelKamino.from_newton(
                    basics.build_sphere_on_plane(friction=0.5, restitution=0.0).finalize(device=self.device)
                )
                model, data, state, limits, detector, jacobians = make_containers(
                    model=model,
                    max_world_contacts=2,
                    sparse=sparse,
                    dt=0.01,
                )
                update_containers(model, data, state, limits, detector, jacobians)
                problem_kwargs = {} if sparse else {"solver": LLTBlockedSolver}
                problem = DualProblem(
                    model=model,
                    data=data,
                    limits=limits,
                    contacts=detector.contacts,
                    jacobians=jacobians,
                    sparse=sparse,
                    config=DualProblem.Config(dynamics=kamino_config.ConstrainedDynamicsConfig(preconditioning=False)),
                    **problem_kwargs,
                )
                problem.build(model, data, jacobians, limits, detector.contacts)
                contact_row = int(problem.data.vio.numpy()[0]) + int(problem.data.ccgo.numpy()[0])
                free_velocity = problem.data.v_f.numpy()
                # Keep enough tangential speed that both dense and sparse PGS
                # remain on the sliding branch after applying Coulomb friction.
                free_velocity[contact_row : contact_row + 3] = (10.0, 0.0, -1.0)
                problem.data.v_f.assign(free_velocity)

                outputs = []
                for contact_law in (None, "de_saxce"):
                    config = kamino_config.DVISolverConfig(
                        contact_solver="pgs",
                        contact_law=contact_law,
                        max_alternating_iterations=8,
                        inequality_sweeps_per_iteration=2,
                    )
                    solver = DVISolver(
                        model=model,
                        data=data,
                        limits=limits,
                        contacts=detector.contacts,
                        jacobians=jacobians,
                        problem=problem,
                        config=config,
                        warmstart=WarmStartMode.NONE,
                    )
                    solver.coldstart()
                    solver.solve(problem)
                    outputs.append(
                        (
                            solver.data.solution.lambdas.numpy().copy(),
                            solver.data.state.s.numpy().copy(),
                            solver.data.status.numpy().copy(),
                        )
                    )

                self.assertGreater(float(outputs[0][1][contact_row + 2]), 0.0)
                np.testing.assert_array_equal(outputs[0][0], outputs[1][0])
                np.testing.assert_array_equal(outputs[0][1], outputs[1][1])
                np.testing.assert_array_equal(outputs[0][2], outputs[1][2])

    def test_05b_apgd_rejects_invalid_contact_triplet_preconditioning(self) -> None:
        """Reject cone-changing or non-invertible scaling without exporting NaN/Inf."""
        for sparse in (False, True):
            with self.subTest(sparse=sparse):
                model = ModelKamino.from_newton(
                    basics.build_sphere_on_plane(friction=0.5, restitution=0.0).finalize(device=self.device)
                )
                model, data, state, limits, detector, jacobians = make_containers(
                    model=model,
                    max_world_contacts=2,
                    sparse=sparse,
                    dt=0.01,
                )
                update_containers(model, data, state, limits, detector, jacobians)
                problem_kwargs = {} if sparse else {"solver": LLTBlockedSolver}
                problem = DualProblem(
                    model=model,
                    data=data,
                    limits=limits,
                    contacts=detector.contacts,
                    jacobians=jacobians,
                    sparse=sparse,
                    config=DualProblem.Config(dynamics=kamino_config.ConstrainedDynamicsConfig(preconditioning=False)),
                    **problem_kwargs,
                )
                problem.build(model, data, jacobians, limits, detector.contacts)
                contact_row = int(problem.data.vio.numpy()[0]) + int(problem.data.ccgo.numpy()[0])
                solver = DVISolver(
                    model=model,
                    data=data,
                    limits=limits,
                    contacts=detector.contacts,
                    jacobians=jacobians,
                    problem=problem,
                    config=self._apgd_config(),
                    warmstart=WarmStartMode.NONE,
                )
                invalid_triplets = {
                    "unequal": (1.0, 2.0, 1.0),
                    "zero": (1.0, 0.0, 1.0),
                    "negative": (1.0, -1.0, 1.0),
                    "nan": (1.0, float("nan"), 1.0),
                }
                for invalid_kind, triplet in invalid_triplets.items():
                    with self.subTest(invalid_kind=invalid_kind):
                        preconditioner = np.ones_like(problem.data.P.numpy())
                        preconditioner[contact_row : contact_row + 3] = triplet
                        problem.data.P.assign(preconditioner)
                        solver.coldstart()
                        warmstart = np.zeros_like(solver.data.solution.lambdas.numpy())
                        warmstart[contact_row : contact_row + 3] = (0.2, 0.1, 0.5)
                        solver.data.solution.lambdas.assign(warmstart)
                        solver.solve(problem)
                        status = solver.data.status.numpy()[0]
                        self.assertEqual(int(status["invalid_contact_preconditioner"]), 1, msg=str(status))
                        self.assertEqual(int(status["converged"]), 0, msg=str(status))
                        self.assertEqual(int(status["contact_iterations"]), 0, msg=str(status))
                        self.assertTrue(np.isinf(float(status["contact_solver_residual"])))
                        self.assertTrue(np.isinf(float(status["r_natural"])))
                        self.assertTrue(np.all(np.isfinite(solver.data.solution.lambdas.numpy())))
                        self.assertTrue(np.all(np.isfinite(solver.data.solution.v_plus.numpy())))
                        self.assertTrue(np.all(np.isfinite(solver.data.state.v_aug.numpy())))
                        self.assertTrue(np.all(np.isfinite(solver.data.state.s.numpy())))
                        np.testing.assert_array_equal(solver.data.solution.lambdas.numpy(), 0.0)

    def test_06_public_graph_replay_handles_contact_activation(self) -> None:
        """Replay nested APGD conditionals as a falling sphere enters contact."""
        if not self.device.is_cuda or not wp.is_mempool_enabled(self.device):
            self.skipTest("Full solver graph replay requires CUDA memory-pool support.")
        if not wp.is_conditional_graph_supported():
            self.skipTest("Nested APGD graph conditionals are unavailable on this device.")

        model = basics.build_sphere_on_plane(
            z_offset=0.3,
            friction=0.5,
            restitution=0.2,
            use_custom_shape_cfg=True,
        ).finalize(device=self.device)
        results = {}
        for sparse in (False, True):
            with self.subTest(sparse=sparse):
                apgd = kamino_config.DVIAPGDConfig(
                    max_iterations=30,
                    max_backtrack_iterations=10,
                    tolerance=1.0e-5,
                    min_iterations=1,
                    early_exit=True,
                    use_graph_conditionals=True,
                )
                config = SolverKamino.Config(
                    dynamics_solver="dvi",
                    use_collision_detector=True,
                    sparse_dynamics=sparse,
                    sparse_jacobian=sparse,
                    dvi=kamino_config.DVISolverConfig(
                        contact_solver="apgd",
                        max_alternating_iterations=2,
                        inequality_sweeps_per_iteration=1,
                        tolerance=2.0e-4,
                        apgd=apgd,
                    ),
                )
                solver = SolverKamino(model, config=config)
                state_0 = model.state()
                state_1 = model.state()

                # Compile outside capture and enter it while the contact set is empty.
                solver.step(state_0, state_1, control=None, contacts=None, dt=0.01)
                state_0, state_1 = state_1, state_0
                self.assertEqual(int(solver._contacts_kamino.world_active_contacts.numpy()[0]), 0)
                with wp.ScopedCapture(self.device) as capture:
                    solver.step(state_0, state_1, control=None, contacts=None, dt=0.01)
                    solver.step(state_1, state_0, control=None, contacts=None, dt=0.01)

                observed_contact = False
                for _ in range(25):
                    wp.capture_launch(capture.graph)
                    observed_contact |= int(solver._contacts_kamino.world_active_contacts.numpy()[0]) > 0

                self.assertTrue(observed_contact)
                status = solver.status.numpy()[0]
                self.assertEqual(int(status["converged"]), 1, msg=str(status))
                self.assertTrue(np.all(np.isfinite(state_0.body_q.numpy())))
                self.assertTrue(np.all(np.isfinite(state_0.body_qd.numpy())))
                results[sparse] = (state_0.body_q.numpy(), state_0.body_qd.numpy())

        np.testing.assert_allclose(results[False][0], results[True][0], rtol=0.0, atol=2.0e-5)
        np.testing.assert_allclose(results[False][1], results[True][1], rtol=0.0, atol=2.0e-5)


if __name__ == "__main__":
    unittest.main()
