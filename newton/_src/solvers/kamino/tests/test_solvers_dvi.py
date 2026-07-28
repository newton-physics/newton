# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the Kamino DVI solver."""

from __future__ import annotations

import unittest
from types import SimpleNamespace

import numpy as np
import warp as wp

import newton
import newton._src.solvers.kamino.config as kamino_config
from newton._src.solvers.kamino._src.dynamics.dual import DualProblem
from newton._src.solvers.kamino._src.integrators.euler import integrate_euler_semi_implicit
from newton._src.solvers.kamino._src.kinematics.constraints import unpack_constraint_solutions, update_constraints_info
from newton._src.solvers.kamino._src.kinematics.jacobians import DenseSystemJacobians
from newton._src.solvers.kamino._src.linalg import LLTBlockedRCMSolver, LLTBlockedSolver
from newton._src.solvers.kamino._src.models.builders import basics, testing
from newton._src.solvers.kamino._src.models.builders import utils as builder_utils
from newton._src.solvers.kamino._src.solvers.common import WarmStartMode
from newton._src.solvers.kamino._src.solvers.dvi import DVISolver
from newton._src.solvers.kamino._src.solvers.dvi.kernels import _color_dvi_contacts, _initialize_dvi_status
from newton._src.solvers.kamino._src.solvers.dvi.sparse import (
    _SPARSE_DELASSUS_ROWS_JOINTS,
    _SPARSE_DELASSUS_ROWS_UNILATERAL,
    _sparse_delassus_matvec_rows,
)
from newton._src.solvers.kamino._src.solvers.metrics import SolutionMetrics
from newton._src.solvers.kamino.solver_kamino import SolverKamino
from newton._src.solvers.kamino.tests import setup_tests, test_context
from newton._src.solvers.kamino.tests.test_solvers_padmm import TestSetup
from newton._src.solvers.kamino.tests.utils.extract import extract_delassus, extract_problem_vector
from newton._src.solvers.kamino.tests.utils.make import make_containers, make_test_problem_fourbar, update_containers
from newton.tests.utils import basics as public_basics


def _reduce_solver_status(status: np.ndarray) -> dict[str, object]:
    """Reduce per-world status while requiring every world to converge."""
    return {
        name: bool(np.all(status[name])) if name == "converged" else np.max(status[name]).item()
        for name in status.dtype.names
    }


def _check_solution_matches_dual_problem(testcase: unittest.TestCase, problem: DualProblem, solver: DVISolver):
    """Check that final physical solution vectors match ``D lambda + v_f``."""
    D_np = extract_delassus(problem.delassus, only_active_dims=True)
    v_f_np = extract_problem_vector(problem.delassus, problem.data.v_f.numpy(), only_active_dims=True)
    P_np = extract_problem_vector(problem.delassus, problem.data.P.numpy(), only_active_dims=True)
    lambdas_np = extract_problem_vector(problem.delassus, solver.data.solution.lambdas.numpy(), only_active_dims=True)
    v_plus_np = extract_problem_vector(problem.delassus, solver.data.solution.v_plus.numpy(), only_active_dims=True)

    status = solver.data.status.numpy()
    for wid in range(problem.data.num_worlds):
        P_inv = np.diag(np.reciprocal(P_np[wid]))
        D_true = P_inv @ D_np[wid] @ P_inv
        v_f_true = P_inv @ v_f_np[wid]
        v_plus_true = D_true @ lambdas_np[wid] + v_f_true
        np.testing.assert_allclose(v_plus_np[wid], v_plus_true, rtol=1e-4, atol=1e-4)

        testcase.assertEqual(int(status[wid]["converged"]), 1)
        testcase.assertLessEqual(int(status[wid]["iterations"]), _status_iteration_budget(solver, wid))
        testcase.assertLessEqual(float(status[wid]["r_p"]), solver.config[wid].tolerance)
        testcase.assertLessEqual(float(status[wid]["r_d"]), solver.config[wid].tolerance)
        testcase.assertLessEqual(float(status[wid]["r_c"]), solver.config[wid].tolerance)
        testcase.assertLessEqual(float(status[wid]["r_b"]), solver.config[wid].tolerance)


def _make_dense_dual_problem(model, data, limits, contacts, jacobians) -> DualProblem:
    problem = DualProblem(
        model=model,
        data=data,
        limits=limits,
        contacts=contacts,
        jacobians=jacobians,
        solver=LLTBlockedSolver,
        sparse=False,
    )
    problem.build(model=model, data=data, limits=limits, contacts=contacts, jacobians=jacobians)
    return problem


def _make_sparse_dual_problem(model, data, limits, contacts, jacobians) -> DualProblem:
    problem = DualProblem(
        model=model,
        data=data,
        limits=limits,
        contacts=contacts,
        jacobians=jacobians,
        sparse=True,
    )
    problem.build(model=model, data=data, limits=limits, contacts=contacts, jacobians=jacobians)
    return problem


def _solve_dvi(
    model,
    problem,
    warmstart: WarmStartMode = WarmStartMode.NONE,
    config: kamino_config.DVISolverConfig | None = None,
    setup: TestSetup | None = None,
) -> DVISolver:
    solver = DVISolver(
        model=model,
        data=setup.data if setup is not None else None,
        limits=setup.limits if setup is not None else None,
        contacts=setup.contacts if setup is not None else None,
        jacobians=setup.jacobians if setup is not None else None,
        config=config or kamino_config.DVISolverConfig(max_iterations=300, tolerance=1e-4, regularization=1e-5),
        warmstart=warmstart,
    )
    solver.reset()
    solver.coldstart()
    solver.solve(problem)
    return solver


def _status_iteration_budget(solver: DVISolver, wid: int) -> int:
    config = solver.config[wid]
    if solver._bilateral_solver is not None and solver.data.bilateral_operator is not None:
        return max(config.max_iterations, config.block_iterations * config.contact_iterations)
    return config.max_iterations


def _assert_solver_status_converged(testcase: unittest.TestCase, solver: DVISolver):
    status = solver.data.status.numpy()
    for wid in range(solver.size.num_worlds):
        testcase.assertEqual(int(status[wid]["converged"]), 1)
        testcase.assertLessEqual(int(status[wid]["iterations"]), _status_iteration_budget(solver, wid))
        testcase.assertLessEqual(float(status[wid]["r_p"]), solver.config[wid].tolerance)
        testcase.assertLessEqual(float(status[wid]["r_d"]), solver.config[wid].tolerance)
        testcase.assertLessEqual(float(status[wid]["r_c"]), solver.config[wid].tolerance)
        testcase.assertLessEqual(float(status[wid]["r_b"]), solver.config[wid].tolerance)


def _assert_solution_finite(testcase: unittest.TestCase, solver: DVISolver):
    testcase.assertTrue(np.all(np.isfinite(solver.data.solution.lambdas.numpy())))
    testcase.assertTrue(np.all(np.isfinite(solver.data.solution.v_plus.numpy())))


def _evaluate_solution_metrics(test: TestSetup, solver: DVISolver) -> dict[str, float]:
    integrate_euler_semi_implicit(model=test.model, data=test.data)
    metrics = SolutionMetrics(model=test.model)
    metrics.evaluate(
        sigma=solver.data.state.sigma,
        lambdas=solver.data.solution.lambdas,
        v_plus=solver.data.solution.v_plus,
        model=test.model,
        data=test.data,
        state_p=test.state_p,
        problem=test.problem,
        jacobians=test.jacobians,
        limits=test.limits,
        contacts=test.contacts,
    )
    return {
        name: float(np.max(getattr(metrics.data, name).numpy()))
        for name in (
            "r_eom",
            "r_kinematics",
            "r_cts_joints",
            "r_cts_limits",
            "r_cts_contacts",
            "r_v_plus",
            "r_ncp_primal",
            "r_ncp_dual",
            "r_ncp_compl",
            "r_vi_natmap",
        )
    }


class TestDVISolver(unittest.TestCase):
    def setUp(self):
        if not test_context.setup_done:
            setup_tests(clear_cache=False)
        self.device = wp.get_device(test_context.device)

    def test_00_config_selection(self):
        default_config = SolverKamino.Config(dynamics_solver="dvi")
        self.assertFalse(default_config.sparse_dynamics)
        self.assertTrue(default_config.sparse_jacobian)
        self.assertEqual(default_config.integrator, "euler")
        self.assertEqual(default_config.dynamics.linear_solver_type, "LLTBRCM")
        self.assertEqual(default_config.dynamics.linear_solver_kwargs, {})
        self.assertEqual(default_config.dvi.omega, 1.0)
        self.assertEqual(default_config.dvi.block_iterations, 32)
        self.assertEqual(default_config.dvi.contact_iterations, 4)
        self.assertEqual(default_config.dvi.bilateral_solve_period, 1)
        self.assertEqual(default_config.dvi.bilateral_solver_type, "LLTB")
        self.assertEqual(default_config.dvi.bilateral_solver_kwargs, {})
        self.assertEqual(default_config.dvi.contact_jacobi_omega, 0.3)
        self.assertEqual(default_config.dvi.contact_jacobi_relaxation, 0.9)

        dense_config = SolverKamino.Config(
            dynamics_solver="dvi",
            sparse_dynamics=False,
            sparse_jacobian=False,
        )
        self.assertFalse(dense_config.sparse_dynamics)
        self.assertFalse(dense_config.sparse_jacobian)
        self.assertEqual(dense_config.integrator, "euler")
        self.assertEqual(dense_config.dynamics.linear_solver_type, "LLTBRCM")
        self.assertEqual(dense_config.dvi.block_iterations, 32)

        padmm_config = SolverKamino.Config()
        self.assertFalse(padmm_config.sparse_dynamics)
        self.assertFalse(padmm_config.sparse_jacobian)
        self.assertEqual(padmm_config.integrator, "euler")

        config = SolverKamino.Config(
            dynamics_solver="dvi",
            dvi=kamino_config.DVISolverConfig(max_iterations=32, tolerance=1e-4),
        )
        self.assertEqual(config.dynamics_solver, "dvi")
        self.assertEqual(config.dvi.max_iterations, 32)
        self.assertEqual(config.dvi.block_iterations, 32)
        self.assertEqual(config.dvi.contact_iterations, 4)
        self.assertEqual(config.dvi.bilateral_solve_period, 1)
        self.assertEqual(config.dvi.contact_jacobi_omega, 0.3)
        self.assertEqual(config.dvi.contact_jacobi_relaxation, 0.9)
        self.assertFalse(config.dvi.contact_block_preconditioner)
        self.assertEqual(config.dvi.contact_warmstart_method, "key_and_position_with_net_force_backup")
        self.assertFalse(config.dynamics.preconditioning)

        sparse_config = SolverKamino.Config(dynamics_solver="dvi", sparse_dynamics=True, sparse_jacobian=True)
        self.assertTrue(sparse_config.sparse_dynamics)
        self.assertTrue(sparse_config.sparse_jacobian)
        self.assertEqual(sparse_config.dynamics.linear_solver_type, "CR")
        self.assertEqual(sparse_config.dynamics.linear_solver_kwargs, {"maxiter": 9})
        with self.assertRaises(ValueError):
            SolverKamino.Config(
                dynamics_solver="dvi",
                dynamics=kamino_config.ConstrainedDynamicsConfig(preconditioning=True),
            )
        invalid_dvi_configs = (
            {"max_iterations": 0},
            {"tolerance": -1.0},
            {"regularization": 0.0},
            {"omega": 0.0},
            {"omega": 2.1},
            {"block_iterations": 0},
            {"contact_iterations": 0},
            {"bilateral_solve_period": 0},
            {"bilateral_solver_type": "invalid"},
            {"contact_jacobi_omega": 0.0},
            {"contact_jacobi_omega": 2.1},
            {"contact_jacobi_relaxation": 0.0},
            {"contact_jacobi_relaxation": 1.1},
            {"warmstart_mode": "invalid"},
        )
        for kwargs in invalid_dvi_configs:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                kamino_config.DVISolverConfig(**kwargs)
        for method in (
            "key_and_position",
            "geom_pair_net_force",
            "key_and_position_with_net_force_backup",
        ):
            self.assertEqual(
                kamino_config.DVISolverConfig(contact_warmstart_method=method).contact_warmstart_method, method
            )
        for method in ("reaction", "geom_pair_net_wrench"):
            with self.assertRaises(ValueError):
                kamino_config.DVISolverConfig(contact_warmstart_method=method)

        model_with_attrs = SimpleNamespace(
            kamino=SimpleNamespace(max_solver_iterations=wp.array([37], dtype=wp.int32, device=self.device))
        )
        self.assertEqual(kamino_config.DVISolverConfig.from_model(model_with_attrs).max_iterations, 37)

    def test_00b_bilateral_solver_selection(self):
        """Verify DVI constructs and validates the configured bilateral solver."""

        def make_model(dimensions):
            return SimpleNamespace(
                size=SimpleNamespace(sum_of_num_joint_cts=sum(dimensions)),
                info=SimpleNamespace(
                    num_joint_cts=wp.array(dimensions, dtype=wp.int32, device=self.device),
                    joint_cts_offset=wp.array(np.cumsum([0, *dimensions[:-1]]), dtype=wp.int32, device=self.device),
                ),
            )

        config = kamino_config.DVISolverConfig(
            bilateral_solver_type="LLTBRCM",
            bilateral_solver_kwargs={"block_size": 16, "reuse_permutation": True},
        )
        solver = DVISolver()
        solver._config = [config]
        solver._data = SimpleNamespace(bilateral_operator=None)
        solver._device = self.device
        solver._allocate_bilateral_solver(make_model([3]))

        self.assertIsInstance(solver._bilateral_solver, LLTBlockedRCMSolver)
        self.assertEqual(solver._bilateral_solver._block_size, 16)
        self.assertTrue(solver._bilateral_solver._reuse_permutation)

        solver._config = [
            kamino_config.DVISolverConfig(bilateral_solver_type="LLTB"),
            kamino_config.DVISolverConfig(bilateral_solver_type="LLTBRCM"),
        ]
        with self.assertRaisesRegex(ValueError, "All worlds must use the same"):
            solver._allocate_bilateral_solver(make_model([3, 3]))

    def test_00a_multiworld_status_reduction_requires_all_worlds_converged(self):
        status = np.array(
            [(True, 2, 1.0e-5), (False, 7, 2.0e-3)],
            dtype=[("converged", np.bool_), ("iterations", np.int32), ("r_d", np.float32)],
        )

        reduced = _reduce_solver_status(status)

        self.assertFalse(reduced["converged"])
        self.assertEqual(reduced["iterations"], 7)
        self.assertAlmostEqual(reduced["r_d"], 2.0e-3)

    def test_01_dvi_solve_dense_dual_problem(self):
        builder = basics.build_boxes_fourbar()
        model, data, state, limits, detector, jacobians = make_containers(
            builder=builder,
            device=self.device,
            max_world_contacts=0,
            sparse=False,
        )
        update_containers(
            model=model,
            data=data,
            state=state,
            limits=limits,
            detector=None,
            jacobians=jacobians,
        )

        dynamics_config = kamino_config.ConstrainedDynamicsConfig(preconditioning=True)
        problem = DualProblem(
            model=model,
            data=data,
            limits=limits,
            contacts=detector.contacts,
            jacobians=jacobians,
            config=DualProblem.Config(dynamics=dynamics_config),
            solver=LLTBlockedSolver,
            sparse=False,
        )
        problem.build(model=model, data=data, limits=limits, contacts=detector.contacts, jacobians=jacobians)

        solver = DVISolver(
            model=model,
            config=kamino_config.DVISolverConfig(max_iterations=1000, tolerance=1e-5, omega=1.0),
            warmstart=WarmStartMode.NONE,
            collect_info=True,
        )
        solver.reset()
        scratch = solver.data.state
        for array in (
            scratch.v_aug,
            scratch.s,
            scratch.scratch,
            scratch.bilateral_rhs,
            scratch.bilateral_solution,
            scratch.bilateral_preconditioner,
        ):
            array.fill_(float("nan"))
        scratch.bilateral_active_dim.fill_(-1)
        scratch.contact_colors.fill_(-1)
        scratch.contact_num_colors.fill_(-1)
        solver.coldstart()
        solver.solve(problem)
        _check_solution_matches_dual_problem(self, problem, solver)
        np.testing.assert_array_equal(solver.data.info.status.numpy(), solver.data.status.numpy())

    def test_02_public_solver_step_with_dvi(self):
        builder = newton.ModelBuilder()
        SolverKamino.register_custom_attributes(builder)
        builder.default_shape_cfg.margin = 0.0
        builder.default_shape_cfg.gap = 0.0
        builder.begin_world()
        body = builder.add_link(
            label="link",
            mass=1.0,
            xform=wp.transformf(wp.vec3f(0.0, 0.0, 1.0), wp.quat_identity(dtype=wp.float32)),
        )
        builder.add_shape_box(label="box", body=body, hx=0.1, hy=0.1, hz=0.1)
        joint = builder.add_joint_revolute(
            label="hinge",
            parent=-1,
            child=body,
            axis=newton.Axis.Y,
            parent_xform=wp.transformf(wp.vec3f(0.0, 0.0, 1.0), wp.quat_identity(dtype=wp.float32)),
            child_xform=wp.transformf(wp.vec3f(0.0, 0.0, 0.0), wp.quat_identity(dtype=wp.float32)),
        )
        builder.add_articulation([joint])
        builder.end_world()
        model = builder.finalize(device=self.device)

        config = SolverKamino.Config(
            dynamics_solver="dvi",
            dvi=kamino_config.DVISolverConfig(max_iterations=500, tolerance=1e-5),
            collect_solver_info=True,
        )
        solver = SolverKamino(model, config=config)
        state_in = model.state()
        state_out = model.state()
        solver.step(state_in, state_out, control=None, contacts=None, dt=1e-3)
        body_q = state_out.body_q.numpy()
        body_qd = state_out.body_qd.numpy()
        self.assertTrue(np.all(np.isfinite(body_q)))
        self.assertTrue(np.all(np.isfinite(body_qd)))
        self.assertIsInstance(solver._solver_kamino.solver_fd, DVISolver)
        self.assertFalse(solver._solver_kamino.config.dynamics.preconditioning)
        self.assertIsNotNone(solver._solver_kamino.solver_fd.data.info)

    def test_03_dvi_solve_single_contact(self):
        builder = basics.build_box_on_plane()
        model, data, state, limits, detector, jacobians = make_containers(
            builder=builder,
            device=self.device,
            max_world_contacts=1,
            sparse=False,
        )
        update_containers(
            model=model,
            data=data,
            state=state,
            limits=limits,
            detector=detector,
            jacobians=jacobians,
        )
        self.assertGreater(int(detector.contacts.model_active_contacts.numpy()[0]), 0)

        problem = DualProblem(
            model=model,
            data=data,
            limits=limits,
            contacts=detector.contacts,
            jacobians=jacobians,
            solver=LLTBlockedSolver,
            sparse=False,
        )
        problem.build(model=model, data=data, limits=limits, contacts=detector.contacts, jacobians=jacobians)

        solver = DVISolver(
            model=model,
            config=kamino_config.DVISolverConfig(max_iterations=200, tolerance=1e-4),
            warmstart=WarmStartMode.NONE,
        )
        solver.reset()
        solver.coldstart()
        solver.solve(problem)
        status = solver.data.status.numpy()[0]
        self.assertEqual(int(status["converged"]), 1)
        self.assertLessEqual(int(status["iterations"]), _status_iteration_budget(solver, 0))
        self.assertTrue(np.all(np.isfinite(solver.data.solution.lambdas.numpy())))
        self.assertTrue(np.all(np.isfinite(solver.data.solution.v_plus.numpy())))

    def test_03b_dvi_contact_block_preconditioner_smoke(self):
        builder = basics.build_boxes_hinged()
        model, data, state, limits, detector, jacobians = make_containers(
            builder=builder,
            device=self.device,
            max_world_contacts=8,
            sparse=False,
        )
        update_containers(
            model=model,
            data=data,
            state=state,
            limits=limits,
            detector=detector,
            jacobians=jacobians,
        )
        self.assertGreater(int(detector.contacts.model_active_contacts.numpy()[0]), 0)

        problem = _make_dense_dual_problem(model, data, limits, detector.contacts, jacobians)

        def solve_with_block_preconditioner(use_colored_contacts: bool) -> DVISolver:
            solver = DVISolver(
                model=model,
                config=kamino_config.DVISolverConfig(
                    max_iterations=200,
                    tolerance=1e-4,
                    contact_block_preconditioner=True,
                ),
                warmstart=WarmStartMode.NONE,
            )
            solver.reset()
            solver.coldstart()
            if use_colored_contacts:
                solver.set_contacts(detector.contacts)
            solver.solve(problem)
            return solver

        contact_paths = [False]
        if self.device.is_cuda:
            contact_paths.append(True)
        for use_colored_contacts in contact_paths:
            with self.subTest(use_colored_contacts=use_colored_contacts):
                solver = solve_with_block_preconditioner(use_colored_contacts)

                if use_colored_contacts:
                    self.assertGreater(int(solver.data.state.contact_num_colors.numpy()[0]), 0)
                else:
                    self.assertEqual(int(solver.data.state.contact_num_colors.numpy()[0]), 0)
                _assert_solution_finite(self, solver)
                _check_solution_matches_dual_problem(self, problem, solver)

    def test_03c_dvi_noncolored_contact_jacobi_uses_configured_omega(self):
        builder = basics.build_boxes_hinged()
        model, data, state, limits, detector, jacobians = make_containers(
            builder=builder,
            device=self.device,
            max_world_contacts=8,
            sparse=False,
        )
        update_containers(
            model=model,
            data=data,
            state=state,
            limits=limits,
            detector=detector,
            jacobians=jacobians,
        )
        self.assertGreater(int(detector.contacts.model_active_contacts.numpy()[0]), 0)

        problem = _make_dense_dual_problem(model, data, limits, detector.contacts, jacobians)

        def solve_normal_lambda(contact_jacobi_omega: float) -> float:
            solver = DVISolver(
                model=model,
                config=kamino_config.DVISolverConfig(
                    tolerance=0.0,
                    regularization=1e-5,
                    block_iterations=1,
                    contact_iterations=1,
                    contact_jacobi_omega=contact_jacobi_omega,
                    contact_jacobi_relaxation=1.0,
                ),
                warmstart=WarmStartMode.NONE,
            )
            solver.reset()
            solver.coldstart()
            solver.solve(problem)
            self.assertEqual(int(solver.data.state.contact_num_colors.numpy()[0]), 0)
            lambdas = extract_problem_vector(
                problem.delassus, solver.data.solution.lambdas.numpy(), only_active_dims=True
            )[0]
            contact_start = int(problem.data.ccgo.numpy()[0])
            contact_count = int(problem.data.nc.numpy()[0])
            normal_lambdas = lambdas[contact_start + 2 : contact_start + 3 * contact_count : 3]
            return float(np.sum(normal_lambdas))

        lambda_slow = solve_normal_lambda(0.1)
        lambda_fast = solve_normal_lambda(0.8)

        self.assertTrue(np.isfinite(lambda_slow))
        self.assertTrue(np.isfinite(lambda_fast))
        self.assertGreater(lambda_fast, lambda_slow)

    def test_03d_dvi_direct_block_honors_per_world_iteration_counts(self):
        builder = builder_utils.make_homogeneous_builder(
            num_worlds=3,
            build_fn=basics.build_boxes_hinged,
        )
        model, data, state, limits, detector, jacobians = make_containers(
            builder=builder,
            device=self.device,
            max_world_contacts=8,
            sparse=False,
        )
        update_containers(
            model=model,
            data=data,
            state=state,
            limits=limits,
            detector=detector,
            jacobians=jacobians,
        )
        self.assertTrue(np.all(detector.contacts.world_active_contacts.numpy() > 0))

        problem = _make_dense_dual_problem(model, data, limits, detector.contacts, jacobians)
        configs = [
            kamino_config.DVISolverConfig(
                tolerance=0.0,
                regularization=1e-5,
                block_iterations=1,
                contact_iterations=1,
                contact_jacobi_relaxation=1.0,
            ),
            kamino_config.DVISolverConfig(
                tolerance=0.0,
                regularization=1e-5,
                block_iterations=3,
                contact_iterations=1,
                contact_jacobi_relaxation=1.0,
            ),
            kamino_config.DVISolverConfig(
                tolerance=0.0,
                regularization=1e-5,
                block_iterations=1,
                contact_iterations=3,
                contact_jacobi_relaxation=1.0,
            ),
        ]

        def solve_normal_sums(use_colored_contacts: bool) -> list[float]:
            solver = DVISolver(model=model, config=configs, warmstart=WarmStartMode.NONE)
            solver.reset()
            solver.coldstart()
            if use_colored_contacts:
                solver.set_contacts(detector.contacts)
            solver.solve(problem)
            status = solver.data.status.numpy()
            self.assertEqual([int(status[wid]["iterations"]) for wid in range(3)], [1, 3, 3])
            if use_colored_contacts:
                self.assertTrue(np.all(solver.data.state.contact_num_colors.numpy() > 0))
            else:
                self.assertTrue(np.all(solver.data.state.contact_num_colors.numpy() == 0))
            np.testing.assert_array_equal(
                solver.data.state.bilateral_active_dim.numpy(),
                problem.data.njc.numpy(),
            )

            lambdas = extract_problem_vector(
                problem.delassus, solver.data.solution.lambdas.numpy(), only_active_dims=True
            )
            ccgo = problem.data.ccgo.numpy().astype(int)
            nc = problem.data.nc.numpy().astype(int)
            return [float(np.sum(lambdas[wid][ccgo[wid] + 2 : ccgo[wid] + 3 * nc[wid] : 3])) for wid in range(3)]

        contact_paths = [False]
        if self.device.is_cuda:
            contact_paths.append(True)
        for use_colored_contacts in contact_paths:
            with self.subTest(use_colored_contacts=use_colored_contacts):
                normal_sums = solve_normal_sums(use_colored_contacts)

                self.assertGreater(normal_sums[1], normal_sums[0])
                self.assertGreater(normal_sums[2], normal_sums[0])

    def test_03d2_dvi_direct_block_finishes_with_bilateral_solve(self):
        builder = basics.build_boxes_hinged()
        model, data, state, limits, detector, jacobians = make_containers(
            builder=builder,
            device=self.device,
            max_world_contacts=8,
            sparse=False,
        )
        update_containers(
            model=model,
            data=data,
            state=state,
            limits=limits,
            detector=detector,
            jacobians=jacobians,
        )
        self.assertGreater(int(detector.contacts.world_active_contacts.numpy()[0]), 0)

        problem = _make_dense_dual_problem(model, data, limits, detector.contacts, jacobians)
        solver = DVISolver(
            model=model,
            config=kamino_config.DVISolverConfig(
                tolerance=0.0,
                regularization=1e-5,
                block_iterations=1,
                contact_iterations=1,
                contact_jacobi_relaxation=1.0,
            ),
            warmstart=WarmStartMode.NONE,
        )
        solver.reset()
        solver.coldstart()
        solver.solve(problem)
        v_plus = extract_problem_vector(problem.delassus, solver.data.solution.v_plus.numpy(), only_active_dims=True)[0]
        njc = int(problem.data.njc.numpy()[0])
        status = solver.data.status.numpy()[0]

        self.assertGreater(njc, 0)
        self.assertLess(float(np.max(np.abs(v_plus[:njc]))), 1e-6)
        self.assertLess(float(status["r_b"]), 1e-6)

    def test_03e_dvi_direct_block_no_unilateral_rows_reports_single_iteration(self):
        builder = basics.build_box_pendulum(ground=False)
        model, data, state, limits, detector, jacobians = make_containers(
            builder=builder,
            device=self.device,
            max_world_contacts=4,
            sparse=False,
        )
        update_containers(
            model=model,
            data=data,
            state=state,
            limits=limits,
            detector=detector,
            jacobians=jacobians,
        )

        problem = _make_dense_dual_problem(model, data, limits, detector.contacts, jacobians)
        self.assertGreater(int(model.info.num_joint_cts.numpy()[0]), 0)
        self.assertEqual(int(problem.data.nl.numpy()[0]), 0)
        self.assertEqual(int(problem.data.nc.numpy()[0]), 0)

        solver = DVISolver(
            model=model,
            config=kamino_config.DVISolverConfig(
                tolerance=1e-4,
                regularization=1e-5,
                block_iterations=7,
                contact_iterations=3,
            ),
            warmstart=WarmStartMode.NONE,
        )
        solver.reset()
        solver.coldstart()
        solver.solve(problem)
        status = solver.data.status.numpy()[0]
        self.assertEqual(int(status["converged"]), 1)
        self.assertEqual(int(status["iterations"]), 1)
        self.assertEqual(int(solver.data.state.bilateral_active_dim.numpy()[0]), 0)
        _check_solution_matches_dual_problem(self, problem, solver)

    def test_03f_dvi_bilateral_only_solve_resets_stale_status(self):
        builder = basics.build_box_pendulum(ground=False)
        model, data, state, limits, detector, jacobians = make_containers(
            builder=builder,
            device=self.device,
            max_world_contacts=0,
            sparse=False,
        )
        update_containers(
            model=model,
            data=data,
            state=state,
            limits=limits,
            detector=None,
            jacobians=jacobians,
        )

        problem = _make_dense_dual_problem(model, data, limits, detector.contacts, jacobians)
        self.assertGreater(int(model.info.num_joint_cts.numpy()[0]), 0)
        self.assertEqual(int(problem.data.nl.numpy()[0]), 0)
        self.assertEqual(int(problem.data.nc.numpy()[0]), 0)

        solver = DVISolver(
            model=model,
            config=kamino_config.DVISolverConfig(
                tolerance=1e-4,
                regularization=1e-5,
                contact_iterations=5,
            ),
            warmstart=WarmStartMode.NONE,
        )
        solver.reset()
        solver.coldstart()
        wp.launch(
            kernel=_initialize_dvi_status,
            dim=solver.size.num_worlds,
            inputs=[
                solver.data.config,
                solver.data.status,
            ],
            device=self.device,
        )
        self.assertEqual(int(solver.data.status.numpy()[0]["iterations"]), 5)

        solver.solve(problem)
        status = solver.data.status.numpy()[0]
        self.assertEqual(int(status["converged"]), 1)
        self.assertEqual(int(status["iterations"]), 1)
        _check_solution_matches_dual_problem(self, problem, solver)

    def test_03g_dvi_contact_coloring_separates_dynamic_conflicts(self):
        problem_nc = wp.array([5], dtype=wp.int32, device=self.device)
        problem_cio = wp.array([0], dtype=wp.int32, device=self.device)
        contact_bid_ab = wp.array(
            [
                wp.vec2i(0, -1),
                wp.vec2i(0, 1),
                wp.vec2i(2, -1),
                wp.vec2i(-1, -1),
                wp.vec2i(1, -1),
            ],
            dtype=wp.vec2i,
            device=self.device,
        )
        contact_colors = wp.full(shape=5, value=-1, dtype=wp.int32, device=self.device)
        contact_num_colors = wp.zeros(shape=1, dtype=wp.int32, device=self.device)

        wp.launch(
            kernel=_color_dvi_contacts,
            dim=1,
            inputs=[
                problem_nc,
                problem_cio,
                contact_bid_ab,
                contact_colors,
                contact_num_colors,
            ],
            device=self.device,
        )
        colors = contact_colors.numpy()
        num_colors = int(contact_num_colors.numpy()[0])
        self.assertGreaterEqual(num_colors, 2)
        self.assertTrue(np.all(colors >= 0))
        self.assertNotEqual(colors[0], colors[1])
        self.assertNotEqual(colors[1], colors[4])
        self.assertLess(colors[3], num_colors)

    def test_03i_dvi_coldstart_is_repeatable(self):
        for sparse in (False, True):
            with self.subTest(sparse=sparse):
                test = TestSetup(
                    builder_fn=basics.build_boxes_hinged,
                    max_world_contacts=8,
                    gravity=True,
                    perturb=True,
                    device=self.device,
                    sparse=sparse,
                )
                test.build()
                config = SolverKamino.Config(
                    dynamics_solver="dvi",
                    sparse_dynamics=sparse,
                    sparse_jacobian=sparse,
                ).dvi
                solver = _solve_dvi(test.model, test.problem, config=config, setup=test)
                first_lambdas = solver.data.solution.lambdas.numpy().copy()
                first_v_plus = solver.data.solution.v_plus.numpy().copy()
                first_status = solver.data.status.numpy().copy()

                test.build()
                solver.reset()
                solver.coldstart()
                solver.solve(test.problem)

                np.testing.assert_allclose(solver.data.solution.lambdas.numpy(), first_lambdas, rtol=0.0, atol=1e-6)
                np.testing.assert_allclose(solver.data.solution.v_plus.numpy(), first_v_plus, rtol=0.0, atol=1e-6)
                status = solver.data.status.numpy()
                np.testing.assert_array_equal(status["converged"], first_status["converged"])
                np.testing.assert_array_equal(status["iterations"], first_status["iterations"])
                for residual in ("r_p", "r_d", "r_c", "r_b"):
                    np.testing.assert_allclose(status[residual], first_status[residual], rtol=1e-5, atol=1e-8)

    def test_04_dvi_solve_active_joint_limit(self):
        builder = testing.build_unary_revolute_joint_test(limits=True, ground=False)
        model, data, state, limits, detector, jacobians = make_containers(
            builder=builder,
            device=self.device,
            max_world_contacts=0,
            sparse=False,
        )
        update_containers(model=model, data=data, state=state, limits=limits, detector=None, jacobians=None)

        q_j = data.joints.q_j.numpy()
        q_j[:] = 1.0
        data.joints.q_j.assign(q_j)
        limits.detect(q_j=data.joints.q_j)
        update_constraints_info(model=model, data=data)
        jacobians.build(model=model, data=data, limits=limits.data, contacts=None)
        self.assertGreater(int(limits.model_active_limits.numpy()[0]), 0)

        problem = _make_dense_dual_problem(model, data, limits, detector.contacts, jacobians)
        solver = _solve_dvi(
            model,
            problem,
            config=kamino_config.DVISolverConfig(max_iterations=8, tolerance=1e-4, regularization=1e-5),
        )

        _assert_solver_status_converged(self, solver)
        iterations = int(solver.data.status.numpy()[0]["iterations"])
        self.assertEqual(iterations, solver.config[0].block_iterations * solver.config[0].contact_iterations)
        self.assertGreater(iterations, solver.config[0].max_iterations)
        _check_solution_matches_dual_problem(self, problem, solver)

    def test_07_dvi_singular_limit_rows_remain_finite(self):
        model, data, _state, limits, contacts = make_test_problem_fourbar(
            device=self.device,
            max_world_contacts=0,
            with_limits=True,
            with_contacts=False,
        )
        jacobians = DenseSystemJacobians(model=model, limits=limits, contacts=contacts)
        jacobians.build(model=model, data=data, limits=limits.data, contacts=None)
        self.assertGreater(int(limits.model_active_limits.numpy()[0]), 0)

        problem = _make_dense_dual_problem(model, data, limits, contacts, jacobians)
        solver = _solve_dvi(model, problem)

        status = solver.data.status.numpy()[0]
        self.assertEqual(int(status["converged"]), 0)
        _assert_solution_finite(self, solver)
        lambdas_np = extract_problem_vector(
            problem.delassus, solver.data.solution.lambdas.numpy(), only_active_dims=True
        )[0]
        limit_start = int(problem.data.lcgo.numpy()[0])
        limit_count = int(problem.data.nl.numpy()[0])
        limit_lambdas = lambdas_np[limit_start : limit_start + limit_count]
        self.assertLess(float(np.max(np.abs(limit_lambdas))), 1.0)

    def test_08_public_solver_short_rollout_with_dvi(self):
        builder = newton.ModelBuilder()
        SolverKamino.register_custom_attributes(builder)
        builder.default_shape_cfg.margin = 0.0
        builder.default_shape_cfg.gap = 0.0
        builder.begin_world()
        body = builder.add_link(
            label="link",
            mass=1.0,
            xform=wp.transformf(wp.vec3f(0.0, 0.0, 1.0), wp.quat_identity(dtype=wp.float32)),
        )
        builder.add_shape_box(label="box", body=body, hx=0.1, hy=0.1, hz=0.1)
        joint = builder.add_joint_revolute(
            label="hinge",
            parent=-1,
            child=body,
            axis=newton.Axis.Y,
            parent_xform=wp.transformf(wp.vec3f(0.0, 0.0, 1.0), wp.quat_identity(dtype=wp.float32)),
            child_xform=wp.transformf(wp.vec3f(0.0, 0.0, 0.0), wp.quat_identity(dtype=wp.float32)),
        )
        builder.add_articulation([joint])
        builder.end_world()
        model = builder.finalize(device=self.device)

        config = SolverKamino.Config(
            dynamics_solver="dvi",
            dvi=kamino_config.DVISolverConfig(max_iterations=300, tolerance=1e-4),
        )
        solver = SolverKamino(model, config=config)
        state_in = model.state()
        state_out = model.state()
        for _ in range(8):
            solver.step(state_in, state_out, control=None, contacts=None, dt=1e-3)
            state_in, state_out = state_out, state_in
        self.assertTrue(np.all(np.isfinite(state_in.body_q.numpy())))
        self.assertTrue(np.all(np.isfinite(state_in.body_qd.numpy())))
        self.assertIsInstance(solver._solver_kamino.solver_fd, DVISolver)

    def test_08a_public_solver_heterogeneous_contact_rollout_with_dvi(self):
        builder = newton.ModelBuilder()
        SolverKamino.register_custom_attributes(builder)
        public_basics.make_basics_heterogeneous_builder(builder=builder, ground=True)
        model = builder.finalize(device=self.device, skip_validation_joints=True)

        config = SolverKamino.Config(
            dynamics_solver="dvi",
            use_collision_detector=True,
            collision_detector=kamino_config.CollisionDetectorConfig(
                max_contacts=64 * model.world_count,
                max_contacts_per_world=64,
                max_contacts_per_pair=16,
            ),
        )
        solver = SolverKamino(model, config=config)
        state_in = model.state()
        state_out = model.state()
        control = model.control()

        contact_seen = False
        for _ in range(24):
            solver.step(state_in, state_out, control=control, contacts=None, dt=1.0e-3)
            state_in, state_out = state_out, state_in
            contact_seen = contact_seen or bool(np.any(solver._contacts_kamino.world_active_contacts.numpy() > 0))

        status = solver._solver_kamino.solver_fd.data.status.numpy()
        self.assertTrue(contact_seen)
        self.assertTrue(np.all(np.isfinite(state_in.body_q.numpy())))
        self.assertTrue(np.all(np.isfinite(state_in.body_qd.numpy())))
        self.assertTrue(np.all(np.isfinite(status["r_p"])))
        self.assertTrue(np.all(np.isfinite(status["r_d"])))
        self.assertLess(float(np.max(np.abs(state_in.body_qd.numpy()))), 100.0)
        self.assertIsInstance(solver._solver_kamino.solver_fd, DVISolver)
        self.assertFalse(config.sparse_dynamics)
        self.assertTrue(config.sparse_jacobian)
        self.assertEqual(config.dynamics.linear_solver_type, "LLTBRCM")

    def test_03a_sparse_dvi_filtered_matvec_matches_full_rows(self):
        builder = basics.build_box_on_plane()
        model, data, state, limits, detector, jacobians = make_containers(
            builder=builder,
            device=self.device,
            max_world_contacts=4,
            sparse=True,
        )
        update_containers(
            model=model,
            data=data,
            state=state,
            limits=limits,
            detector=detector,
            jacobians=jacobians,
        )
        self.assertGreater(int(detector.contacts.model_active_contacts.numpy()[0]), 0)

        problem = _make_sparse_dual_problem(model, data, limits, detector.contacts, jacobians)
        solver = DVISolver(
            model=model,
            data=data,
            limits=limits,
            contacts=detector.contacts,
            jacobians=jacobians,
            config=kamino_config.DVISolverConfig(
                tolerance=0.0,
                regularization=1e-5,
                block_iterations=1,
                contact_iterations=1,
            ),
            warmstart=WarmStartMode.NONE,
        )
        solver.reset()

        lambdas = np.linspace(-0.25, 0.5, problem.data.v_f.shape[0], dtype=np.float32)
        solver.data.solution.lambdas.assign(lambdas)

        full = wp.zeros_like(problem.data.v_f)
        problem.delassus.matvec(solver.data.solution.lambdas, full, solver.all_worlds_mask)
        full_np = full.numpy()

        _sparse_delassus_matvec_rows(solver, problem, _SPARSE_DELASSUS_ROWS_JOINTS)
        joint_np = solver.data.state.v_aug.numpy()
        _sparse_delassus_matvec_rows(solver, problem, _SPARSE_DELASSUS_ROWS_UNILATERAL)
        unilateral_np = solver.data.state.v_aug.numpy()

        dim = int(problem.data.dim.numpy()[0])
        njc = int(problem.data.njc.numpy()[0])
        np.testing.assert_allclose(joint_np[:njc], full_np[:njc], rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(unilateral_np[njc:dim], full_np[njc:dim], rtol=1e-5, atol=1e-5)

    def test_05_dvi_solve_multi_world_contacts(self):
        builder = builder_utils.make_homogeneous_builder(
            num_worlds=4,
            build_fn=basics.build_box_on_plane,
            ground=True,
        )
        model, data, state, limits, detector, jacobians = make_containers(
            builder=builder,
            device=self.device,
            max_world_contacts=4,
            sparse=False,
        )
        update_containers(
            model=model,
            data=data,
            state=state,
            limits=limits,
            detector=detector,
            jacobians=jacobians,
        )
        self.assertTrue(np.all(detector.contacts.world_active_contacts.numpy() > 0))

        problem = _make_dense_dual_problem(model, data, limits, detector.contacts, jacobians)
        solver = _solve_dvi(model, problem)

        _assert_solver_status_converged(self, solver)
        _check_solution_matches_dual_problem(self, problem, solver)

    def test_06_dvi_warmstart_modes(self):
        builder = basics.build_box_on_plane()
        model, data, state, limits, detector, jacobians = make_containers(
            builder=builder,
            device=self.device,
            max_world_contacts=4,
            sparse=False,
        )
        update_containers(
            model=model,
            data=data,
            state=state,
            limits=limits,
            detector=detector,
            jacobians=jacobians,
        )
        problem = _make_dense_dual_problem(model, data, limits, detector.contacts, jacobians)

        internal_solver = _solve_dvi(model, problem, warmstart=WarmStartMode.INTERNAL)
        cold_iterations = int(internal_solver.data.status.numpy()[0]["iterations"])
        _assert_solver_status_converged(self, internal_solver)

        problem.build(model=model, data=data, limits=limits, contacts=detector.contacts, jacobians=jacobians)
        internal_solver.warmstart(problem, model, data)
        internal_solver.solve(problem)
        _assert_solver_status_converged(self, internal_solver)
        self.assertLessEqual(int(internal_solver.data.status.numpy()[0]["iterations"]), cold_iterations)

        unpack_constraint_solutions(
            lambdas=internal_solver.data.solution.lambdas,
            v_plus=internal_solver.data.solution.v_plus,
            model=model,
            data=data,
            limits=limits,
            contacts=detector.contacts,
        )
        container_solver = DVISolver(
            model=model,
            config=kamino_config.DVISolverConfig(max_iterations=300, tolerance=1e-4, regularization=1e-5),
            warmstart=WarmStartMode.CONTAINERS,
        )
        problem.build(model=model, data=data, limits=limits, contacts=detector.contacts, jacobians=jacobians)
        container_solver.warmstart(problem, model, data, limits, detector.contacts)
        container_solver.solve(problem)
        _assert_solver_status_converged(self, container_solver)
        _check_solution_matches_dual_problem(self, problem, container_solver)

    def test_06a_dvi_masked_reset_preserves_unselected_worlds(self):
        builder = builder_utils.make_homogeneous_builder(
            num_worlds=3,
            build_fn=basics.build_box_on_plane,
            ground=True,
        )
        model, data, state, limits, detector, jacobians = make_containers(
            builder=builder,
            device=self.device,
            max_world_contacts=4,
            sparse=False,
        )
        update_containers(
            model=model,
            data=data,
            state=state,
            limits=limits,
            detector=detector,
            jacobians=jacobians,
        )
        problem = _make_dense_dual_problem(model, data, limits, detector.contacts, jacobians)
        solver = _solve_dvi(model, problem)
        lambdas_before = solver.data.solution.lambdas.numpy().copy()
        v_plus_before = solver.data.solution.v_plus.numpy().copy()

        world_mask = wp.array([False, True, False], dtype=wp.bool, device=self.device)
        solver.reset(problem=problem, world_mask=world_mask)

        lambdas_after = extract_problem_vector(
            problem.delassus, solver.data.solution.lambdas.numpy(), only_active_dims=False
        )
        v_plus_after = extract_problem_vector(
            problem.delassus, solver.data.solution.v_plus.numpy(), only_active_dims=False
        )
        lambdas_before = extract_problem_vector(problem.delassus, lambdas_before, only_active_dims=False)
        v_plus_before = extract_problem_vector(problem.delassus, v_plus_before, only_active_dims=False)
        np.testing.assert_array_equal(lambdas_after[0], lambdas_before[0])
        np.testing.assert_array_equal(lambdas_after[2], lambdas_before[2])
        np.testing.assert_array_equal(v_plus_after[0], v_plus_before[0])
        np.testing.assert_array_equal(v_plus_after[2], v_plus_before[2])
        np.testing.assert_array_equal(lambdas_after[1], np.zeros_like(lambdas_after[1]))
        np.testing.assert_array_equal(v_plus_after[1], np.zeros_like(v_plus_after[1]))

    def test_12_dvi_opening_contact_releases_warmstarted_force(self):
        if not self.device.is_cuda:
            self.skipTest("DVI colored contact release regression uses the CUDA graph-colored path")

        radius = 0.1
        separation = 0.005
        gap = 0.03
        z = radius + separation

        builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
        SolverKamino.register_custom_attributes(builder)
        shape_cfg = newton.ModelBuilder.ShapeConfig(gap=gap, margin=0.0)
        body = builder.add_link(
            xform=wp.transform(p=wp.vec3(0.0, 0.0, z), q=wp.quat_identity()),
            mass=1.0,
        )
        builder.add_shape_sphere(body=body, radius=radius, cfg=shape_cfg)
        joint = builder.add_joint_prismatic(
            parent=-1,
            child=body,
            axis=newton.Axis.Z,
            parent_xform=wp.transform(p=wp.vec3(0.0, 0.0, z), q=wp.quat_identity()),
            child_xform=wp.transform_identity(),
            limit_lower=-10.0,
            limit_upper=10.0,
        )
        builder.add_articulation([joint])
        builder.add_ground_plane(cfg=shape_cfg)
        model = builder.finalize(device=self.device)

        joint_qd = model.joint_qd.numpy()
        joint_qd[:] = 1.0
        model.joint_qd.assign(joint_qd)

        state_0 = model.state()
        state_1 = model.state()
        newton.eval_fk(model, model.joint_q, model.joint_qd, state_0)

        config = SolverKamino.Config(
            use_collision_detector=True,
            collision_detector=kamino_config.CollisionDetectorConfig(
                max_contacts_per_world=8,
                max_contacts_per_pair=8,
                default_gap=gap,
            ),
            dynamics_solver="dvi",
            dvi=kamino_config.DVISolverConfig(
                max_iterations=300,
                tolerance=1e-5,
                regularization=1e-5,
                block_iterations=32,
                contact_iterations=4,
            ),
        )
        solver = SolverKamino(model, config=config)

        solver.step(state_0, state_1, control=None, contacts=None, dt=1e-3)
        self.assertEqual(int(solver._contacts_kamino.model_active_contacts.numpy()[0]), 1)

        cache = solver._solver_kamino._ws_contacts.cache
        self.assertIsNotNone(cache)
        reaction = cache.reaction.numpy()
        reaction[:, :] = 0.0
        reaction[0, 2] = 10000.0
        cache.reaction.assign(reaction)
        velocity = cache.velocity.numpy()
        velocity[:, :] = 0.0
        cache.velocity.assign(velocity)

        solver.step(state_1, state_0, control=None, contacts=None, dt=1e-3)

        contact_count = int(solver._contacts_kamino.model_active_contacts.numpy()[0])
        gaps = solver._contacts_kamino.gapfunc.numpy()[:contact_count, 3]
        contact_velocity = solver._contacts_kamino.velocity.numpy()[:contact_count, 2]
        contact_reaction = solver._contacts_kamino.reaction.numpy()[:contact_count, 2]
        opening = (gaps > 0.0) & (contact_velocity > 0.0)

        self.assertTrue(np.any(opening))
        self.assertLess(float(np.max(np.abs(contact_reaction[opening]))), 1e-3)
        self.assertLess(float(abs(state_0.body_qd.numpy()[0, 2])), 2.0)
        self.assertEqual(int(solver._solver_kamino.solver_fd.data.status.numpy()[0]["converged"]), 1)

    def test_03h_dvi_canonical_contact_solution_metrics(self):
        for builder_fn, max_world_contacts in (
            (basics.build_box_on_plane, 4),
            (basics.build_boxes_hinged, 8),
        ):
            for sparse in (False, True):
                with self.subTest(builder=builder_fn.__name__, sparse=sparse):
                    test = TestSetup(
                        builder_fn=builder_fn,
                        max_world_contacts=max_world_contacts,
                        gravity=True,
                        perturb=True,
                        device=self.device,
                        sparse=sparse,
                    )
                    test.build()
                    config = SolverKamino.Config(
                        dynamics_solver="dvi",
                        sparse_dynamics=sparse,
                        sparse_jacobian=sparse,
                    ).dvi
                    solver = _solve_dvi(test.model, test.problem, config=config, setup=test)
                    solution_metrics = _evaluate_solution_metrics(test, solver)

                    _assert_solution_finite(self, solver)
                    for name, value in solution_metrics.items():
                        self.assertTrue(np.isfinite(value), msg=f"{name}={value}")

                    # DVI trades some contact accuracy for throughput, but its
                    # solution must still satisfy dynamics and cone feasibility.
                    self.assertLess(solution_metrics["r_eom"], 1.0e-4, msg=str(solution_metrics))
                    self.assertLess(solution_metrics["r_kinematics"], 1.0e-4, msg=str(solution_metrics))
                    self.assertLess(solution_metrics["r_cts_joints"], 1.0e-4, msg=str(solution_metrics))
                    self.assertLess(solution_metrics["r_v_plus"], 1.0e-4, msg=str(solution_metrics))
                    self.assertLess(solution_metrics["r_ncp_primal"], 1.0e-4, msg=str(solution_metrics))
                    self.assertLess(solution_metrics["r_ncp_dual"], 1.0e-2, msg=str(solution_metrics))
                    self.assertLess(solution_metrics["r_ncp_compl"], 1.0e-2, msg=str(solution_metrics))
                    self.assertLess(solution_metrics["r_vi_natmap"], 1.0e-2, msg=str(solution_metrics))

    def test_08b_dr_legs_contact_capacity_scales_with_world_count(self):
        if not self.device.is_cuda:
            self.skipTest("Dr Legs multi-world capacity regression uses the CUDA graph path")

        from types import SimpleNamespace  # noqa: PLC0415

        from newton.examples.kamino.example_kamino_robot_dr_legs import Example  # noqa: PLC0415
        from newton.viewer import ViewerNull  # noqa: PLC0415

        world_count = 3
        args = SimpleNamespace(
            world_count=world_count,
            use_kamino_contacts=False,
            dynamics_solver="dvi",
            dvi_contact_block_preconditioner=False,
            dvi_contact_jacobi_omega=0.45,
            dvi_contact_jacobi_relaxation=0.9,
        )
        example = Example(ViewerNull(num_frames=1), args)

        expected_capacity = 72 * world_count
        self.assertEqual(example.model.rigid_contact_max, expected_capacity)
        self.assertEqual(example.contacts.rigid_contact_max, expected_capacity)
        self.assertEqual(example.collision_pipeline.rigid_contact_max, expected_capacity)

    def test_09_dr_legs_dvi_first_contact_remains_finite(self):
        if not self.device.is_cuda:
            self.skipTest("Dr Legs DVI first-contact regression uses the CUDA graph path")

        from types import SimpleNamespace  # noqa: PLC0415

        from newton.examples.kamino.example_kamino_robot_dr_legs import Example  # noqa: PLC0415
        from newton.viewer import ViewerNull  # noqa: PLC0415

        args = SimpleNamespace(
            world_count=1,
            use_kamino_contacts=True,
            dynamics_solver="dvi",
            dvi_contact_block_preconditioner=False,
            dvi_contact_jacobi_omega=0.45,
            dvi_contact_jacobi_relaxation=0.9,
        )
        example = Example(ViewerNull(num_frames=1), args)

        contact_seen = False
        color_checked = False
        for _ in range(12):
            example.step()
            body_q = example.state_0.body_q.numpy()
            body_qd = example.state_0.body_qd.numpy()
            lambdas = example.solver._solver_kamino.solver_fd.data.solution.lambdas.numpy()
            kamino_contacts = example.solver._contacts_kamino
            contact_count = int(kamino_contacts.world_active_contacts.numpy()[0])
            contact_seen = contact_seen or contact_count > 0
            if contact_count > 0 and not example.config.sparse_dynamics:
                solver_fd = example.solver._solver_kamino.solver_fd
                color_count = int(solver_fd.data.state.contact_num_colors.numpy()[0])
                colors = solver_fd.data.state.contact_colors.numpy()
                bid_ab = kamino_contacts.bid_AB.numpy()
                self.assertGreater(color_count, 0)
                self.assertTrue(np.all(colors[:contact_count] >= 0))
                for ci in range(contact_count):
                    bodies_i = {int(bid_ab[ci][0]), int(bid_ab[ci][1])} - {-1}
                    for cj in range(ci):
                        if colors[ci] == colors[cj]:
                            bodies_j = {int(bid_ab[cj][0]), int(bid_ab[cj][1])} - {-1}
                            self.assertFalse(bodies_i & bodies_j)
                color_checked = True
            elif contact_count > 0:
                color_checked = True

            self.assertTrue(np.all(np.isfinite(body_q)))
            self.assertTrue(np.all(np.isfinite(body_qd)))
            self.assertTrue(np.all(np.isfinite(lambdas)))
            self.assertLess(float(np.max(np.abs(body_qd))), 100.0)
            self.assertLess(float(np.max(np.abs(lambdas))), 100.0)

        self.assertTrue(contact_seen)
        self.assertTrue(color_checked)

    def test_10_dr_legs_dvi_tipped_contact_does_not_creep(self):
        if not self.device.is_cuda:
            self.skipTest("Dr Legs DVI tipped-contact regression uses the CUDA graph path")

        from types import SimpleNamespace  # noqa: PLC0415

        from newton.examples.kamino.example_kamino_robot_dr_legs import Example  # noqa: PLC0415
        from newton.viewer import ViewerNull  # noqa: PLC0415

        args = SimpleNamespace(
            world_count=1,
            use_kamino_contacts=True,
            dynamics_solver="dvi",
            dvi_contact_block_preconditioner=False,
            dvi_contact_jacobi_omega=0.45,
            dvi_contact_jacobi_relaxation=0.9,
        )
        example = Example(ViewerNull(num_frames=1), args)

        q_tip = wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), float(np.pi * 0.5))
        example.base_q.assign([wp.transformf((0.0, 0.0, 0.25), q_tip)])
        reset_config = SolverKamino.ResetConfig(base_pose=SolverKamino.ResetConfig.FromBaseQ(example.base_q))
        example.solver.reset(state=example.state_0, config=reset_config)
        example.solver.reset(state=example.state_1, config=reset_config)
        example.capture()

        base_start = example.state_0.body_q.numpy()[0, :3].copy()
        contact_seen = False
        post_settle_penetration = []
        post_settle_xy = []
        for step_idx in range(400):
            example.step()
            contact_seen = contact_seen or int(example.contacts.rigid_contact_count.numpy()[0]) > 0
            contacts_kamino = example.solver._contacts_kamino
            contact_count = int(contacts_kamino.world_active_contacts.numpy()[0])
            if step_idx >= 40 and contact_count > 0:
                gaps = contacts_kamino.gapfunc.numpy()[:contact_count, 3]
                post_settle_penetration.append(float(max(0.0, -np.min(gaps))))
            if step_idx >= 200:
                post_settle_xy.append(example.state_0.body_q.numpy()[0, :2].copy())

        body_q = example.state_0.body_q.numpy()
        body_qd = example.state_0.body_qd.numpy()
        base_delta_xy = body_q[0, :2] - base_start[:2]

        self.assertTrue(contact_seen)
        self.assertTrue(np.all(np.isfinite(body_q)))
        self.assertTrue(np.all(np.isfinite(body_qd)))
        self.assertLess(float(np.linalg.norm(base_delta_xy)), 0.008)
        self.assertGreater(len(post_settle_penetration), 0)
        self.assertLess(float(np.percentile(post_settle_penetration, 95)), 0.0035)
        self.assertLess(float(np.linalg.norm(post_settle_xy[-1] - post_settle_xy[0])), 5.0e-4)

    def test_11_dr_legs_dvi_contact_force_balances_weight(self):
        if not self.device.is_cuda:
            self.skipTest("Dr Legs DVI contact-force regression uses the CUDA graph path")

        from types import SimpleNamespace  # noqa: PLC0415

        from newton._src.solvers.kamino._src.geometry.aggregation import ContactAggregation  # noqa: PLC0415
        from newton.examples.kamino.example_kamino_robot_dr_legs import Example  # noqa: PLC0415
        from newton.viewer import ViewerNull  # noqa: PLC0415

        args = SimpleNamespace(
            world_count=1,
            use_kamino_contacts=True,
            dynamics_solver="dvi",
            dvi_contact_block_preconditioner=False,
            dvi_contact_jacobi_omega=0.45,
            dvi_contact_jacobi_relaxation=0.9,
        )
        example = Example(ViewerNull(num_frames=1), args)

        base_z = []
        for _ in range(180):
            example.step()
            base_z.append(float(example.state_0.body_q.numpy()[0, 2]))

        contacts_kamino = example.solver._contacts_kamino
        aggregation = ContactAggregation(model=example.solver._model_kamino, contacts=contacts_kamino)
        aggregation.compute()

        contact_count = int(contacts_kamino.world_active_contacts.numpy()[0])
        total_contact_force = aggregation.body_net_force.numpy()[0].sum(axis=0)
        weight = float(example.model.body_mass.numpy().sum() * 9.81)
        force_ratio = float(total_contact_force[2] / weight)

        self.assertGreater(contact_count, 0)
        self.assertTrue(np.all(np.isfinite(total_contact_force)))
        self.assertGreater(force_ratio, 0.95)
        self.assertLess(force_ratio, 1.05)
        z = np.array(base_z[60:], dtype=np.float64)
        x = np.arange(z.size, dtype=np.float64)
        residual = z - np.polyval(np.polyfit(x, z, 1), x)
        self.assertLess(float(np.max(residual) - np.min(residual)), 0.001)


if __name__ == "__main__":
    unittest.main()
