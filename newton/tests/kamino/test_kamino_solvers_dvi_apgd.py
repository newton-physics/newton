# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Independent contact-law and integration regressions for unilateral APGD."""

import math
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import warp as wp

from newton._src.solvers.kamino._src.core.model import ModelKamino
from newton._src.solvers.kamino._src.dynamics.dual import DualProblem
from newton._src.solvers.kamino._src.geometry.aggregation import ContactAggregation
from newton._src.solvers.kamino._src.linalg import LLTBlockedSolver
from newton._src.solvers.kamino._src.solvers.common import WarmStartMode
from newton._src.solvers.kamino._src.solvers.dvi import DVISolver
from newton._src.solvers.kamino._src.solvers.dvi.apgd import UnilateralAPGD
from newton._src.solvers.kamino._src.solvers.dvi.types import DVIConfigStruct, DVIStatus, convert_config_to_struct
from newton._src.solvers.kamino.config import ConstrainedDynamicsConfig, DVIAPGDConfig, DVISolverConfig
from newton.solvers import SolverKamino
from newton.tests.kamino.test_kamino_solver_kamino_joint_friction import (
    _run_hold_and_breakaway_test,
    _run_spin_down_test,
)
from newton.tests.kamino.test_kamino_solvers_dvi import _build_five_box_stack
from newton.tests.kamino.utils.make import make_containers, update_containers
from newton.tests.kamino.utils.solver_configs import make_dvi_dense_config, make_dvi_sparse_config
from newton.tests.utils import basics
from newton.viewer import ViewerNull


def _devices():
    """Exercise CPU and the first CUDA device when available."""
    return ["cpu"] + (["cuda:0"] if wp.is_cuda_available() else [])


def _problem(matrices, biases, families, friction, device, *, options=None, lower=None, upper=None, configs=None):
    """Build independent small unilateral problems in the actual batched storage layout."""
    sizes = [len(b) for b in biases]
    offsets = np.cumsum([0, *[max(1, n) for n in sizes]])
    matrix_offsets = np.cumsum([0, *[max(1, n * n) for n in sizes]])
    bound_offsets = np.cumsum([0, *[f[0] for f in families]])
    contact_offsets = np.cumsum([0, *[f[2] for f in families]])

    def ints(values):
        """Allocate layout metadata on the test device."""
        return wp.array(values, dtype=wp.int32, device=device)

    def floats(values):
        """Allocate numerical inputs on the test device."""
        return wp.array(values, dtype=wp.float32, device=device)

    matrix = np.zeros(matrix_offsets[-1])
    velocity = np.zeros(offsets[-1])
    for wid, (a, b) in enumerate(zip(matrices, biases, strict=True)):
        matrix[matrix_offsets[wid] : matrix_offsets[wid] + sizes[wid] ** 2] = np.asarray(a).ravel()
        velocity[offsets[wid] : offsets[wid] + sizes[wid]] = b
    data = SimpleNamespace(
        dim=ints(sizes),
        vio=ints(offsets[:-1]),
        mio=ints(matrix_offsets[:-1]),
        njc=ints([0] * len(sizes)),
        nbc=ints([f[0] for f in families]),
        nl=ints([f[1] for f in families]),
        nc=ints([f[2] for f in families]),
        ccgo=ints([f[0] + f[1] for f in families]),
        bcio=ints(bound_offsets[:-1]),
        cio=ints(contact_offsets[:-1]),
        mu=floats(friction),
        D=floats(matrix),
        v_f=floats(velocity),
        bound_lower=floats(lower if lower is not None else []),
        bound_upper=floats(upper if upper is not None else []),
    )
    if configs is None:
        configs = [DVISolverConfig(unilateral_solver="apgd", apgd=options or DVIAPGDConfig()) for _ in sizes]
    size = SimpleNamespace(
        num_worlds=len(sizes),
        max_of_max_total_cts=max(1, *sizes),
        max_of_num_bilateral_joint_cts=0,
        sum_of_max_total_cts=int(offsets[-1]),
        sum_of_num_bilateral_joint_cts=0,
    )
    owner = SimpleNamespace(
        device=wp.get_device(device),
        size=size,
        config=configs,
        _use_schur_complement=False,
        _bilateral_solver=None,
        data=SimpleNamespace(
            status=wp.zeros(len(sizes), dtype=DVIStatus, device=device),
            config=wp.array([convert_config_to_struct(c) for c in configs], dtype=DVIConfigStruct, device=device),
            solution=SimpleNamespace(lambdas=wp.zeros(int(offsets[-1]), device=device)),
        ),
    )
    return UnilateralAPGD(owner), SimpleNamespace(data=data, sparse=False)


def _model_problem(builder, device, sparse, *, schur=False, max_contacts=64, iterations=24):
    """Assemble an actual Kamino model with rigid, unpreconditioned dynamics."""
    model = ModelKamino.from_newton(builder.finalize(device=device))
    model, data, state, limits, detector, jacobians = make_containers(
        model=model,
        max_world_contacts=max_contacts,
        sparse=sparse,
        dt=0.001,
    )
    update_containers(model, data, state, limits, detector, jacobians)
    problem = DualProblem(
        model=model,
        data=data,
        limits=limits,
        contacts=detector.contacts,
        jacobians=jacobians,
        sparse=sparse,
        solver=None if sparse else LLTBlockedSolver,
        config=DualProblem.Config(dynamics=ConstrainedDynamicsConfig(preconditioning=False)),
    )
    problem.build(model, data, jacobians, limits, detector.contacts)
    solver = DVISolver(
        model=model,
        data=data,
        limits=limits,
        contacts=detector.contacts,
        jacobians=jacobians,
        problem=problem,
        config=DVISolverConfig(
            unilateral_solver="apgd",
            use_schur_complement=schur,
            max_alternating_iterations=iterations,
            tolerance=1e-4,
            apgd=DVIAPGDConfig(max_iterations=128, max_nonlinear_corrections=30, tolerance=1e-6),
        ),
        warmstart=WarmStartMode.NONE,
    )
    return solver, problem


class TestDVIAPGD(unittest.TestCase):
    """Check Coulomb impulses rather than only the inner cone-QP residual."""

    def test_configuration_is_opt_in(self):
        """Retain PGS defaults and accept the APGD backend."""
        self.assertEqual(DVISolverConfig().unilateral_solver, "pgs")
        self.assertEqual(DVISolverConfig(unilateral_solver="apgd").unilateral_solver, "apgd")
        self.assertEqual(DVIAPGDConfig().max_nonlinear_corrections, 1)

    def test_configuration_rejects_invalid_controls(self):
        """Reject unusable budgets, damping, and tolerances before launching kernels."""
        for field in ("max_iterations", "max_backtracks", "max_nonlinear_corrections"):
            for value in (0, -1, True, 1.5):
                with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                    DVIAPGDConfig(**{field: value})
        for options in (
            {"tolerance": float("nan")},
            {"tolerance": -1.0},
            {"relaxation": 0.0},
            {"relaxation": 1.1},
            {"relaxation": float("inf")},
        ):
            with self.subTest(options=options), self.assertRaises(ValueError):
                DVIAPGDConfig(**options)
        with self.assertRaises(ValueError):
            DVISolverConfig(unilateral_solver="unknown")

    def test_de_saxce_sliding(self):
        """Recover Coulomb normal impulse instead of the associated cone-QP solution."""
        for device in _devices():
            with self.subTest(device=device):
                solver, problem = _problem(
                    [np.eye(3)],
                    [[10.0, 0.0, -1.0]],
                    [(0, 0, 1)],
                    [0.5],
                    device,
                    options=DVIAPGDConfig(max_nonlinear_corrections=8),
                )
                solver.solve(problem)
                impulses = solver.owner.data.solution.lambdas.numpy()
                np.testing.assert_allclose(impulses, [-0.5, 0.0, 1.0], atol=2.0e-5, rtol=0.0)
                velocity = impulses + np.array([10.0, 0.0, -1.0])
                self.assertAlmostEqual(float(velocity[2]), 0.0, places=4)
                self.assertLess(float(solver.owner.data.status.numpy()[0]["apgd_residual"]), 1.1e-5)
                self.assertGreater(int(solver.owner.data.status.numpy()[0]["apgd_corrections"]), 1)

    def test_operator_scale_and_replay(self):
        """Resolve small and large operators after their scale changes in the same graph."""
        scales = np.array([1e-8, 1e-4, 1.0, 1e4, 1e8], dtype=np.float32)
        for device in _devices():
            with self.subTest(device=device):
                solver, problem = _problem(
                    [np.array([[s]]) for s in scales],
                    [[-1.0]] * len(scales),
                    [(0, 1, 0)] * len(scales),
                    [],
                    device,
                )

                def solve(solver=solver, problem=problem):
                    """Reset impulses and status on every replay."""
                    solver.owner.data.solution.lambdas.zero_()
                    solver.owner.data.status.zero_()
                    solver.solve(problem)

                solve()
                graph = None
                if solver.device.is_cuda:
                    with wp.ScopedCapture(device=device) as capture:
                        solve()
                    graph = capture.graph
                for values in (scales, scales[::-1]):
                    problem.data.D.assign(values.copy())
                    if graph is None:
                        solve()
                    else:
                        wp.capture_launch(graph)
                    impulses = solver.owner.data.solution.lambdas.numpy().astype(np.float64)
                    np.testing.assert_allclose(values.astype(np.float64) * impulses, 1.0, atol=2e-6, rtol=0.0)
                    info = solver.owner.data.status.numpy()
                    np.testing.assert_array_equal(info["iterations"], np.ones(len(scales)))
                    np.testing.assert_array_equal(info["apgd_backtracks"], np.zeros(len(scales)))
                    np.testing.assert_array_equal(info["apgd_line_search_failed"], np.zeros(len(scales)))

    def test_cuda_requires_conditional_graphs(self):
        """Reject unsupported CUDA runtimes during allocation, while retaining CPU support."""
        if not wp.is_cuda_available():
            self.skipTest("CUDA allocation requires a CUDA device")
        with patch.object(wp, "is_conditional_graph_supported", return_value=False):
            with self.assertRaisesRegex(RuntimeError, "conditional.*12.4"):
                _problem([np.eye(1)], [[-1.0]], [(0, 1, 0)], [], "cuda:0")
            _problem([np.eye(1)], [[-1.0]], [(0, 1, 0)], [], "cpu")

    def test_null_scale_probe_and_empty_worlds(self):
        """Retry a null direction and handle zero operators and empty worlds on replay."""
        for device in _devices():
            with self.subTest(device=device):
                solver, problem = _problem(
                    [np.array([[1.0, -1.0], [-1.0, 1.0]]), np.zeros((1, 1)), np.empty((0, 0))],
                    [[-1.0, 1.0], [1.0], []],
                    [(0, 2, 0), (0, 1, 0), (0, 0, 0)],
                    [],
                    device,
                )
                solver.solve(problem)
                if solver.device.is_cuda:
                    with wp.ScopedCapture(device=device) as capture:
                        solver.owner.data.solution.lambdas.zero_()
                        solver.owner.data.status.zero_()
                        solver.solve(problem)
                    wp.capture_launch(capture.graph)
                np.testing.assert_allclose(solver.operator_scale.numpy(), [2.0, 1.0, 1.0], atol=1e-6)
                impulses = solver.owner.data.solution.lambdas.numpy()
                # The singular problem has a family of solutions with x0 - x1 = 1.
                self.assertAlmostEqual(float(impulses[0] - impulses[1]), 1.0, delta=2e-5)
                self.assertTrue(np.all(impulses >= 0.0))
                np.testing.assert_array_equal(impulses[2:], [0.0, 0.0])
                info = solver.owner.data.status.numpy()
                np.testing.assert_array_equal(info["apgd_line_search_failed"], [0, 0, 0])
                self.assertEqual(int(info[2]["iterations"]), 0)
                self.assertFalse(np.any(solver.estimating.numpy()))
                self.assertEqual(int(solver.scale_condition.numpy()[0]), 0)

    def test_scaled_coupled_operators(self):
        """Recover an independent solution across operator scales and eigenvalue directions."""
        base = np.array([[2.0, -0.5], [-0.5, 1.0]])
        scales = [1e-8, 1.0, 1e8]
        expected = np.array([1.0, 2.0])
        configs = [
            DVISolverConfig(
                unilateral_solver="apgd",
                apgd=DVIAPGDConfig(
                    max_iterations=128,
                    tolerance=1e-6 * min(1.0, 1.0 / scale),
                ),
            )
            for scale in scales
        ]
        for device in _devices():
            with self.subTest(device=device):
                solver, problem = _problem(
                    [scale * base for scale in scales],
                    [-base @ expected] * len(scales),
                    [(0, 2, 0)] * len(scales),
                    [],
                    device,
                    configs=configs,
                )
                if solver.device.is_cuda:
                    with wp.ScopedCapture(device=device) as capture:
                        solver.solve(problem)
                    wp.capture_launch(capture.graph)
                else:
                    solver.solve(problem)
                impulses = solver.owner.data.solution.lambdas.numpy().reshape(-1, 2).astype(np.float64)
                for scale, impulse in zip(scales, impulses, strict=True):
                    np.testing.assert_allclose(scale * impulse, expected, atol=2e-5, rtol=0.0)
                    np.testing.assert_allclose(base @ (scale * impulse - expected), 0.0, atol=2e-5)
                self.assertFalse(np.any(solver.owner.data.status.numpy()["apgd_line_search_failed"]))

    def test_scaled_frictional_impulses(self):
        """Satisfy Coulomb sliding across operator scales without false early exits."""
        scales = [1e-8, 1.0, 1e8]
        configs = [
            DVISolverConfig(
                unilateral_solver="apgd",
                apgd=DVIAPGDConfig(
                    max_nonlinear_corrections=20,
                    tolerance=1e-5 * min(1.0, 1.0 / scale),
                ),
            )
            for scale in scales
        ]
        for device in _devices():
            with self.subTest(device=device):
                solver, problem = _problem(
                    [scale * np.eye(3) for scale in scales],
                    [[10.0, 0.0, -1.0]] * len(scales),
                    [(0, 0, 1)] * len(scales),
                    [0.5] * len(scales),
                    device,
                    configs=configs,
                )
                if solver.device.is_cuda:
                    with wp.ScopedCapture(device=device) as capture:
                        solver.solve(problem)
                    wp.capture_launch(capture.graph)
                else:
                    solver.solve(problem)
                impulses = solver.owner.data.solution.lambdas.numpy().reshape(-1, 3).astype(np.float64)
                for scale, impulse in zip(scales, impulses, strict=True):
                    np.testing.assert_allclose(scale * impulse, [-0.5, 0.0, 1.0], atol=2e-5, rtol=0.0)
                    velocity = scale * impulse + [10.0, 0.0, -1.0]
                    self.assertAlmostEqual(float(velocity[2]), 0.0, delta=2e-5)
                    self.assertAlmostEqual(float(impulse[0] / impulse[2]), -0.5, delta=2e-6)
                self.assertFalse(np.any(solver.owner.data.status.numpy()["apgd_line_search_failed"]))

    def test_scale_reuse_between_alternating_blocks(self):
        """Retain learned curvature and reduce it only when a new QP restarts momentum."""
        for device in _devices():
            with self.subTest(device=device):
                solver, problem = _problem([[[4.0, -3.0], [-3.0, 4.0]]], [[-4.0, 3.0]], [(0, 2, 0)], [], device)
                previous_state = wp.empty_like(solver.state)
                previous_status = wp.empty_like(solver.owner.data.status)
                exact = wp.array([1.0, 0.0], dtype=wp.float32, device=device)

                def solve(
                    solver=solver,
                    problem=problem,
                    previous_state=previous_state,
                    previous_status=previous_status,
                    exact=exact,
                ):
                    """Capture both alternating phases and their intermediate diagnostics."""
                    solver.owner.data.solution.lambdas.zero_()
                    solver.owner.data.status.zero_()
                    solver.solve(problem, block_iteration=0)
                    wp.copy(previous_state, solver.state)
                    wp.copy(previous_status, solver.owner.data.status)
                    # Exact feasible warmstart avoids a roundoff-driven line search
                    # obscuring the retained estimate in the next phase.
                    wp.copy(solver.owner.data.solution.lambdas, exact)
                    solver.solve(problem, block_iteration=1)

                solve()
                if solver.device.is_cuda:
                    with wp.ScopedCapture(device=device) as capture:
                        solve()
                    wp.capture_launch(capture.graph)
                before, after = previous_state.numpy()[0], solver.state.numpy()[0]
                self.assertAlmostEqual(float(before["lipschitz"]), 4.0, delta=1e-5)
                self.assertAlmostEqual(float(after["lipschitz"]), 0.9 * float(before["lipschitz"]), delta=1e-6)
                self.assertEqual(int(previous_status.numpy()[0]["apgd_backtracks"]), 2)
                self.assertEqual(int(solver.owner.data.status.numpy()[0]["apgd_backtracks"]), 2)
                np.testing.assert_allclose(solver.owner.data.solution.lambdas.numpy(), [1.0, 0.0], atol=1e-6)

    def test_graph_size_is_independent_of_iteration_budgets(self):
        """Capture each loop body once, even when all three iteration budgets grow."""
        if not wp.is_cuda_available():
            self.skipTest("CUDA graph structure requires a CUDA device")
        counts = []
        for budget in (2, 128):
            solver, problem = _problem(
                [np.eye(1)],
                [[-1.0]],
                [(0, 1, 0)],
                [],
                "cuda:0",
                options=DVIAPGDConfig(max_iterations=budget, max_backtracks=budget, max_nonlinear_corrections=budget),
            )
            with wp.ScopedCapture(device="cuda:0") as capture:
                solver.solve(problem)
            wp.capture_launch(capture.graph)
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "apgd.dot"
                wp.capture_debug_dot_print(capture.graph, str(path))
                graph = path.read_text()
            self.assertEqual(graph.count("check_descent_"), 1)
            self.assertEqual(graph.count("Conditional Type| WHILE"), 3)
            counts.append(graph.count('label="{KERNEL'))
            self.assertEqual(int(solver.owner.data.status.numpy()[0]["iterations"]), 1)
        self.assertEqual(counts[0], counts[1])

    def test_contact_regimes_and_world_isolation(self):
        """Resolve sticking, sliding, separation, and frictionless contacts in one batch."""
        biases = [[0.1, -0.2, -1.0], [6.0, 8.0, -1.0], [1.0, 0.0, 2.0], [10.0, 2.0, -1.0], []]
        expected = [(-0.1, 0.2, 1.0), (-0.3, -0.4, 1.0), (0.0, 0.0, 0.0), (0.0, 0.0, 1.0)]
        for device in _devices():
            with self.subTest(device=device):
                solver, problem = _problem(
                    [np.eye(3)] * 4 + [np.empty((0, 0))],
                    biases,
                    [(0, 0, 1)] * 4 + [(0, 0, 0)],
                    [0.5, 0.5, 0.5, 0.0],
                    device,
                    options=DVIAPGDConfig(max_nonlinear_corrections=8),
                )
                solver.solve(problem)
                np.testing.assert_allclose(
                    solver.owner.data.solution.lambdas.numpy()[:12], np.ravel(expected), atol=2e-5
                )
                info = solver.owner.data.status.numpy()
                self.assertEqual(int(info[-1]["iterations"]), 0)
                self.assertTrue(np.all(info["apgd_line_search_failed"] == 0))

    def test_mixed_bounds_limits_and_contacts(self):
        """Project boxes and limits independently of the contact correction."""
        for device in _devices():
            with self.subTest(device=device):
                solver, problem = _problem(
                    [np.eye(5)],
                    [[-4.0, -2.0, 10.0, 0.0, -1.0]],
                    [(1, 1, 1)],
                    [0.5],
                    device,
                    lower=[-0.25],
                    upper=[0.25],
                    options=DVIAPGDConfig(max_nonlinear_corrections=8),
                )
                solver.solve(problem)
                np.testing.assert_allclose(
                    solver.owner.data.solution.lambdas.numpy(),
                    [0.25, 2.0, -0.5, 0.0, 1.0],
                    atol=2e-5,
                )

    def test_coupled_contact_oracle(self):
        """Recover a prescribed sliding solution with off-diagonal Delassus coupling."""
        rng = np.random.default_rng(7)
        basis = rng.normal(size=(6, 6))
        matrix = np.eye(6) + 0.04 * basis.T @ basis
        expected = np.array([-0.3, -0.4, 1.0, 0.6, -0.8, 2.0])
        velocity = np.array([3.0, 4.0, 0.0, -3.0, 4.0, 0.0])
        bias = velocity - matrix @ expected
        for device in _devices():
            with self.subTest(device=device):
                solver, problem = _problem(
                    [matrix],
                    [bias],
                    [(0, 0, 2)],
                    [0.5, 0.5],
                    device,
                    options=DVIAPGDConfig(max_iterations=100, max_nonlinear_corrections=40, tolerance=2e-6),
                )
                solver.solve(problem)
                np.testing.assert_allclose(solver.owner.data.solution.lambdas.numpy(), expected, atol=2e-5)

    def test_exhausted_line_search_retains_finite_iterate(self):
        """Reject an unverified step when the backtracking budget is exhausted."""
        for device in _devices():
            with self.subTest(device=device):
                solver, problem = _problem(
                    [np.diag([1.0, 1.0, 1000.0])],
                    [[0.0, 0.0, -1.0]],
                    [(0, 0, 1)],
                    [0.5],
                    device,
                    options=DVIAPGDConfig(max_backtracks=1),
                )
                solver.solve(problem)
                np.testing.assert_array_equal(solver.owner.data.solution.lambdas.numpy(), [0.0, 0.0, 0.0])
                status = solver.owner.data.status.numpy()[0]
                self.assertEqual(int(status["apgd_line_search_failed"]), 1)
                self.assertEqual(int(status["iterations"]), 0)

    def test_nonlinear_budget_does_not_hide_residual(self):
        """Distinguish a frozen-correction approximation from a consistent warmstart."""
        for device in _devices():
            with self.subTest(device=device):
                solver, problem = _problem(
                    [np.eye(3)],
                    [[10.0, 0.0, -1.0]],
                    [(0, 0, 1)],
                    [0.5],
                    device,
                )
                solver.solve(problem)
                # The cold-start shift is 5. Its cone QP gives n=0.8, while
                # Coulomb contact requires n=1 and zero normal velocity.
                np.testing.assert_allclose(
                    solver.owner.data.solution.lambdas.numpy(), [-0.4, 0.0, 0.8], atol=1e-6, rtol=0.0
                )
                info = solver.owner.data.status.numpy()[0]
                self.assertEqual(int(info["apgd_corrections"]), 1)
                self.assertAlmostEqual(float(info["apgd_residual"]), 0.16, delta=1e-6)

                # A consistent warmstart already supplies the final shift.
                solver.owner.data.solution.lambdas.assign(np.array([-0.5, 0.0, 1.0], dtype=np.float32))
                solver.owner.data.status.zero_()
                solver.solve(problem)
                np.testing.assert_allclose(
                    solver.owner.data.solution.lambdas.numpy(), [-0.5, 0.0, 1.0], atol=1e-6, rtol=0.0
                )
                info = solver.owner.data.status.numpy()[0]
                self.assertEqual(int(info["apgd_corrections"]), 1)
                self.assertLessEqual(float(info["apgd_residual"]), 1e-5)

    def test_nonfinite_backtracking_stops_all_loops(self):
        """Reject overflowing trial products without accepting a step or retrying corrections."""
        for device in _devices():
            with self.subTest(device=device):
                solver, problem = _problem(
                    [1e-20 * np.eye(3)],
                    [[0.0, 0.0, -1e20]],
                    [(0, 0, 1)],
                    [0.0],
                    device,
                    options=DVIAPGDConfig(
                        max_iterations=2,
                        max_backtracks=3,
                        max_nonlinear_corrections=2,
                    ),
                )

                def solve(solver=solver, problem=problem):
                    """Reset the initial impulse and status for each solve or replay."""
                    solver.owner.data.solution.lambdas.zero_()
                    solver.owner.data.status.zero_()
                    solver.solve(problem)

                solve()
                if solver.device.is_cuda:
                    with wp.ScopedCapture(device=device) as capture:
                        solve()
                    wp.capture_launch(capture.graph)
                info = solver.owner.data.status.numpy()[0]
                self.assertEqual(int(info["apgd_line_search_failed"]), 1)
                self.assertEqual(int(info["iterations"]), 0)
                self.assertEqual(int(info["apgd_corrections"]), 0)
                np.testing.assert_array_equal(solver.owner.data.solution.lambdas.numpy(), [0.0, 0.0, 0.0])
                self.assertFalse(np.any(solver.active.numpy()))
                self.assertFalse(np.any(solver.inner.numpy()))
                self.assertFalse(np.any(solver.searching.numpy()))

    def test_early_termination_at_every_level(self):
        """Exit all three loops before their budgets, including in a captured CUDA graph."""
        options = DVIAPGDConfig(max_iterations=12, max_backtracks=4, max_nonlinear_corrections=12)
        for device in _devices():
            with self.subTest(device=device), wp.ScopedDevice(device):
                if wp.get_device(device).is_cuda and not wp.is_conditional_graph_supported():
                    self.skipTest("CUDA conditional graphs are not supported")
                solver, problem = _problem(
                    [2.0 * np.eye(3)],
                    [[10.0, 0.0, -1.0]],
                    [(0, 0, 1)],
                    [0.5],
                    device,
                    options=options,
                )

                def solve(solver=solver, problem=problem):
                    """Reset impulses and diagnostics so each replay performs a cold solve."""
                    solver.owner.data.solution.lambdas.zero_()
                    solver.owner.data.status.zero_()
                    solver.solve(problem)

                solve()
                eager_status = solver.owner.data.status.numpy()
                graph = None
                if solver.device.is_cuda:
                    with wp.ScopedCapture(device=device) as capture:
                        solve()
                    graph = capture.graph

                for replay in range(2):
                    with self.subTest(replay=replay):
                        if graph is None:
                            solve()
                        else:
                            wp.capture_launch(graph)
                        status = solver.owner.data.status.numpy()
                        np.testing.assert_array_equal(status, eager_status)
                        info = status[0]
                        state = solver.state.numpy()[0]
                        corrections = int(info["apgd_corrections"])
                        backtracks = int(info["apgd_backtracks"])
                        self.assertEqual(int(info["apgd_line_search_failed"]), 0)
                        self.assertLessEqual(float(info["apgd_residual"]), options.tolerance)
                        self.assertGreater(corrections, 1)
                        self.assertLess(corrections, options.max_nonlinear_corrections)
                        # Each completed QP takes at least one accepted step, so
                        # equality proves every inner solve stopped after one.
                        self.assertEqual(int(info["iterations"]), corrections)
                        self.assertEqual(int(state["iterations"]), 1)
                        self.assertLess(int(state["iterations"]), options.max_iterations)
                        # The seeded curvature accepts the first trial in every search.
                        self.assertEqual(backtracks, 0)
                        self.assertLess(backtracks + 1, options.max_backtracks)
                        np.testing.assert_allclose(
                            solver.owner.data.solution.lambdas.numpy(), [-0.25, 0.0, 0.5], atol=1e-5, rtol=0.0
                        )
                        for mask in (solver.active, solver.inner, solver.searching):
                            self.assertFalse(np.any(mask.numpy()))
                        for condition in (solver.correction_condition, solver.inner_condition, solver.search_condition):
                            self.assertEqual(int(condition.numpy()[0]), 0)

    def test_batched_loop_termination_and_replay(self):
        """Stop each world at its own tolerance, iteration limit, or line-search failure."""
        matrices = [
            np.empty((0, 0)),
            [[2.0]],
            *([np.diag([1.0, 1.0, np.sqrt(10.0)])] * 3),
            np.eye(3),
            np.eye(3),
            [[4.0, -3.0], [-3.0, 4.0]],
            [[4.0, -3.0], [-3.0, 4.0]],
            np.diag([1.0, 1.0, np.sqrt(10.0)]),
        ]
        biases = [
            [],
            [-2.0],
            *([[-1.0, 0.0, 0.0]] * 3),
            [10.0, 0.0, -1.0],
            [10.0, 0.0, -1.0],
            [-4.0, 3.0],
            [-4.0, 3.0],
            [-1.0, 0.0, 0.0],
        ]
        families = [
            (0, 0, 0),
            (0, 1, 0),
            *([(0, 3, 0)] * 3),
            (0, 0, 1),
            (0, 0, 1),
            (0, 2, 0),
            (0, 2, 0),
            (0, 3, 0),
        ]
        overrides = [
            {},
            {},
            {},
            {"tolerance": 0.25},
            {"max_iterations": 2},
            {"max_nonlinear_corrections": 10, "tolerance": 1e-5},
            {"max_nonlinear_corrections": 2, "tolerance": 1e-5},
            {},
            {"max_backtracks": 2},
            {"max_iterations": 1, "max_nonlinear_corrections": 12},
        ]
        for device in _devices():
            with self.subTest(device=device):
                configs = [
                    DVISolverConfig(
                        unilateral_solver="apgd",
                        apgd=DVIAPGDConfig(
                            **{
                                "max_iterations": 12,
                                "max_backtracks": 3,
                                "tolerance": 1e-3,
                                **override,
                            }
                        ),
                    )
                    for override in overrides
                ]
                solver, problem = _problem(matrices, biases, families, [0.5, 0.5], device, configs=configs)

                def solve(solver=solver, problem=problem):
                    """Start each replay from the same cold impulse and clean counters."""
                    solver.owner.data.solution.lambdas.zero_()
                    solver.owner.data.status.zero_()
                    solver.solve(problem)

                solve()
                graph = None
                if solver.device.is_cuda:
                    with wp.ScopedCapture(device=device) as capture:
                        solve()
                    graph = capture.graph
                    wp.capture_launch(graph)
                impulses = solver.owner.data.solution.lambdas.numpy()
                info = solver.owner.data.status.numpy()
                np.testing.assert_array_equal(info["iterations"], [0, 1, 9, 2, 2, 8, 2, 1, 0, 10])
                np.testing.assert_array_equal(info["apgd_corrections"], [0, 1, 1, 1, 1, 8, 2, 1, 0, 10])
                np.testing.assert_array_equal(info["apgd_backtracks"], [0, 0, 0, 0, 0, 0, 0, 2, 2, 0])
                np.testing.assert_array_equal(info["apgd_line_search_failed"], [0] * 8 + [1, 0])
                for wid in (1, 2, 3, 5, 7, 9):
                    self.assertLessEqual(float(info[wid]["apgd_residual"]), configs[wid].apgd.tolerance)
                # A loose tolerance stops at equality; a tight tolerance with
                # the same two-step budget returns the same unfinished iterate.
                self.assertEqual(float(info[3]["apgd_residual"]), 0.25)
                self.assertEqual(float(info[4]["apgd_residual"]), 0.25)
                self.assertAlmostEqual(float(info[6]["apgd_residual"]), 0.032, delta=1e-6)
                self.assertFalse(np.any(solver.active.numpy()))
                self.assertFalse(np.any(solver.inner.numpy()))
                self.assertFalse(np.any(solver.searching.numpy()))
                for condition in (solver.correction_condition, solver.inner_condition, solver.search_condition):
                    self.assertEqual(int(condition.numpy()[0]), 0)

                # Reuse the same captured graph with no active constraints,
                # then restore them to check that masks and counters reset.
                dimensions = problem.data.dim.numpy()
                for active in (False, True):
                    problem.data.dim.assign(dimensions if active else np.zeros_like(dimensions))
                    if graph is None:
                        solve()
                    else:
                        wp.capture_launch(graph)
                    actual = solver.owner.data.status.numpy()
                    if active:
                        np.testing.assert_array_equal(actual, info)
                        np.testing.assert_array_equal(solver.owner.data.solution.lambdas.numpy(), impulses)
                    else:
                        np.testing.assert_array_equal(actual["iterations"], np.zeros(len(configs)))
                        np.testing.assert_array_equal(actual["apgd_corrections"], np.zeros(len(configs)))
                        np.testing.assert_array_equal(actual["apgd_line_search_failed"], np.zeros(len(configs)))

    def test_warmstart_and_phase_masks(self):
        """Project stale warmstarts and preserve worlds outside their alternating budget."""
        for device in _devices():
            configs = [
                DVISolverConfig(
                    unilateral_solver="apgd",
                    max_alternating_iterations=n,
                    apgd=DVIAPGDConfig(max_nonlinear_corrections=20),
                )
                for n in (1, 2)
            ]
            solver, problem = _problem(
                [np.eye(3)] * 2,
                [[10.0, 0.0, -1.0]] * 2,
                [(0, 0, 1)] * 2,
                [0.5, 0.5],
                device,
                configs=configs,
            )
            stale = np.array([20.0, -10.0, -2.0] * 2, dtype=np.float32)
            solver.owner.data.solution.lambdas.assign(stale)
            solver.solve(problem, block_iteration=1)
            result = solver.owner.data.solution.lambdas.numpy()
            np.testing.assert_array_equal(result[:3], stale[:3])
            np.testing.assert_allclose(result[3:], [-0.5, 0.0, 1.0], atol=2e-5)
            solver.solve(problem)
            np.testing.assert_allclose(solver.owner.data.solution.lambdas.numpy(), [-0.5, 0.0, 1.0] * 2, atol=2e-5)

    def test_alternation_matches_schur(self):
        """Converge existing bilateral alternation to the eliminated solution."""
        for device in _devices():
            for sparse in (False, True):
                outputs = []
                for schur in (False, True):
                    solver, problem = _model_problem(
                        basics.build_boxes_fourbar(limits=False, friction=0.0),
                        device,
                        sparse,
                        schur=schur,
                        iterations=64,
                    )
                    solver.coldstart()
                    solver.solve(problem)
                    n = int(problem.data.dim.numpy()[0])
                    outputs.append(solver.data.solution.v_plus.numpy()[:n])
                with self.subTest(device=device, sparse=sparse):
                    np.testing.assert_allclose(outputs[0], outputs[1], atol=3e-4)

    def test_dense_sparse_sphere(self):
        """Solve actual dense and sparse contact operators to the same Coulomb impulses."""
        for device in _devices():
            outputs = []
            for sparse in (False, True):
                solver, problem = _model_problem(
                    basics.build_sphere_on_plane(friction=0.5, use_custom_shape_cfg=True),
                    device,
                    sparse,
                )
                self.assertEqual(int(problem.data.nc.numpy()[0]), 1)
                bias = problem.data.v_f.numpy()
                row = int(problem.data.vio.numpy()[0] + problem.data.ccgo.numpy()[0])
                bias[row : row + 3] = [10.0, 0.0, -1.0]
                problem.data.v_f.assign(bias)
                solver.coldstart()
                solver.solve(problem)
                outputs.append(solver.data.solution.lambdas.numpy()[row : row + 3])
                status = solver.data.status.numpy()[0]
                self.assertEqual(int(status["converged"]), 1, str(status))
                np.testing.assert_allclose(outputs[-1], [-0.5, 0.0, 1.0], atol=2e-5)
            with self.subTest(device=device):
                np.testing.assert_allclose(outputs[0], outputs[1], atol=2e-5)

    def test_schur_operator_and_recovered_bilaterals(self):
        """Match dense and sparse Schur solves and independently eliminate bilateral rows."""
        for device in _devices():
            outputs = []
            for sparse in (False, True):
                solver, problem = _model_problem(
                    basics.build_boxes_fourbar(limits=False, friction=0.0),
                    device,
                    sparse,
                    schur=True,
                )
                solver.coldstart()
                solver.solve(problem)
                if solver.device.is_cuda:
                    eager_velocity = solver.data.solution.v_plus.numpy()
                    with wp.ScopedCapture(device=device) as capture:
                        solver.coldstart()
                        solver.solve(problem)
                    wp.capture_launch(capture.graph)
                    np.testing.assert_allclose(solver.data.solution.v_plus.numpy(), eager_velocity, atol=3e-4)
                data = problem.data
                n = int(data.dim.numpy()[0])
                nb = int(data.njc.numpy()[0])
                self.assertGreater(nb, 0)
                self.assertGreater(n, nb)
                outputs.append((solver.data.solution.lambdas.numpy()[:n], solver.data.solution.v_plus.numpy()[:n]))
                self.assertLess(float(np.max(np.abs(outputs[-1][1][:nb]))), 2e-4)
                if not sparse:
                    a = data.D.numpy()[: n * n].reshape(n, n).astype(np.float64)
                    # Existing bilateral factorization regularizes after symmetric scaling.
                    scale = solver.data.state.bilateral_preconditioner.numpy()[:nb].astype(np.float64)
                    regularized_b = a[:nb, :nb] + np.diag(7e-7 / scale**2)
                    schur_matrix = a[nb:, nb:] - a[nb:, :nb] @ np.linalg.solve(regularized_b, a[:nb, nb:])
                    probe = np.ones(n - nb)
                    expected_scale = np.linalg.norm(schur_matrix @ probe) / np.linalg.norm(probe)
                    self.assertAlmostEqual(float(solver._apgd.operator_scale.numpy()[0]), expected_scale, delta=1e-4)
                    x = np.zeros_like(solver._apgd.x.numpy())
                    x[nb:n] = np.linspace(0.1, 1.0, n - nb)
                    solver._apgd.x.assign(x)
                    solver._apgd.matvec(solver._apgd.x, solver._apgd.product, solver.all_worlds_mask)
                    actual = solver._apgd.product.numpy()[nb:n]
                    expected = schur_matrix @ x[nb:n]
                    np.testing.assert_allclose(actual, expected, atol=1e-4, rtol=1e-4)
            with self.subTest(device=device):
                np.testing.assert_allclose(outputs[0][1], outputs[1][1], atol=3e-4)

    def test_heterogeneous_schur_worlds(self):
        """Preserve padded bilateral offsets and inactive worlds in a mixed Schur batch."""
        for device in _devices():
            outputs = []
            for sparse in (False, True):
                builder = basics.build_sphere_on_plane()
                basics.build_boxes_fourbar(builder=builder, limits=False, ground=False, z_offset=5.0, actuator_ids=[])
                basics.build_boxes_fourbar(builder=builder, limits=False)
                solver, problem = _model_problem(builder, device, sparse, schur=True)
                self.assertEqual(int(problem.data.njc.numpy()[0]), 0)
                self.assertEqual(int(problem.data.nc.numpy()[1]), 0)
                self.assertEqual(int(problem.data.dim.numpy()[1]), int(problem.data.njc.numpy()[1]))
                solver.coldstart()
                solver.solve(problem)
                velocity = solver.data.solution.v_plus.numpy()
                dims, offsets = problem.data.dim.numpy(), problem.data.vio.numpy()
                outputs.append(np.concatenate([velocity[o : o + n] for n, o in zip(dims, offsets, strict=True)]))
                info = solver.data.status.numpy()
                self.assertTrue(np.all(info["apgd_line_search_failed"] == 0))
                self.assertEqual(int(info[1]["iterations"]), 0)
                self.assertLess(float(np.max(info["r_b"])), 3e-4)
            with self.subTest(device=device):
                np.testing.assert_allclose(outputs[0], outputs[1], atol=3e-4)

    def test_contact_stack_support(self):
        """Support a five-box stack using both operator representations."""
        for device in _devices():
            for sparse in (False, True):
                with self.subTest(device=device, sparse=sparse):
                    solver, problem = _model_problem(_build_five_box_stack(), device, sparse)
                    self.assertGreater(int(problem.data.nc.numpy()[0]), 4)
                    solver.coldstart()
                    solver.solve(problem)
                    info = solver.data.status.numpy()[0]
                    self.assertEqual(int(info["apgd_line_search_failed"]), 0, str(info))
                    self.assertLess(float(info["r_d"]), 1e-4, str(info))
                    self.assertLess(float(info["r_c"]), 1e-5, str(info))

    def test_graph_replay_with_active_contact_changes(self):
        """Replay a preallocated solve as a contact disappears and returns."""
        if not wp.is_cuda_available():
            self.skipTest("CUDA graph replay requires a CUDA device")
        solver, problem = _problem(
            [np.eye(3)],
            [[10.0, 0.0, -1.0]],
            [(0, 0, 1)],
            [0.5],
            "cuda:0",
            options=DVIAPGDConfig(max_nonlinear_corrections=8),
        )
        solver.solve(problem)
        with wp.ScopedCapture(device="cuda:0") as capture:
            solver.solve(problem)
        for active in (False, True, True):
            problem.data.dim.fill_(3 if active else 0)
            problem.data.nc.fill_(1 if active else 0)
            solver.owner.data.solution.lambdas.zero_()
            solver.owner.data.status.zero_()
            wp.capture_launch(capture.graph)
            expected = [-0.5, 0.0, 1.0] if active else [0.0, 0.0, 0.0]
            np.testing.assert_allclose(solver.owner.data.solution.lambdas.numpy(), expected, atol=2e-5)


class TestDVIAPGDDynamics(unittest.TestCase):
    """Exercise the actual integration, warmstart, and captured sparse solver paths."""

    def test_joint_friction_spin_down(self):
        """Reuse analytical joint-friction deceleration and stopped-pose checks for APGD."""
        for device in _devices():
            for factory in (make_dvi_dense_config, make_dvi_sparse_config):
                with self.subTest(device=device, config=factory.__name__), wp.ScopedDevice(device):
                    config = factory()
                    config.dvi.unilateral_solver = "apgd"
                    _run_spin_down_test(self, factory.__name__, config)

    def test_joint_friction_hold_and_breakaway(self):
        """Reuse static holding and saturated friction-torque checks for APGD."""
        for device in _devices():
            for factory in (make_dvi_dense_config, make_dvi_sparse_config):
                with self.subTest(device=device, config=factory.__name__), wp.ScopedDevice(device):
                    config = factory()
                    config.dvi.unilateral_solver = "apgd"
                    _run_hold_and_breakaway_test(self, factory.__name__, config)

    def test_contact_stack_rollout(self):
        """Keep a five-box stack supported through dense and sparse simulation steps."""
        for device in _devices():
            for sparse in (False, True):
                with self.subTest(device=device, sparse=sparse), wp.ScopedDevice(device):
                    model = _build_five_box_stack().finalize(device=device)
                    config = SolverKamino.Config(
                        dynamics_solver="dvi",
                        use_collision_detector=True,
                        sparse_jacobian=sparse,
                        sparse_dynamics=sparse,
                    )
                    config.dvi.unilateral_solver = "apgd"
                    config.dvi.apgd.tolerance = 1e-6
                    solver = SolverKamino(model, config=config)
                    state_0, state_1 = model.state(), model.state()
                    initial = state_0.body_q.numpy()[:, :3].copy()
                    dt = 1e-3

                    def step_pair(solver=solver, state_0=state_0, state_1=state_1, dt=dt):
                        """Advance both state buffers for a reusable capture."""
                        solver.step(state_0, state_1, control=None, contacts=None, dt=dt)
                        solver.step(state_1, state_0, control=None, contacts=None, dt=dt)

                    step_pair()
                    if wp.get_device(device).is_cuda:
                        with wp.ScopedCapture(device=device) as capture:
                            step_pair()
                        for _ in range(49):
                            wp.capture_launch(capture.graph)
                    else:
                        for _ in range(49):
                            step_pair()
                    final = state_0.body_q.numpy()[:, :3]
                    self.assertTrue(np.all(np.isfinite(final)))
                    self.assertLess(float(np.max(np.abs(final - initial))), 1e-3)
                    self.assertLess(float(np.max(np.abs(state_0.body_qd.numpy()))), 0.02)
                    info = solver._solver_kamino.solver_fd.data.status.numpy()[0]
                    self.assertEqual(int(info["apgd_line_search_failed"]), 0)
                    self.assertLess(float(info["r_d"]), 1e-4)
                    contacts = solver._contacts_kamino
                    aggregation = ContactAggregation(model=solver._model_kamino, contacts=contacts)
                    aggregation.compute()
                    force = aggregation.body_net_force.numpy()[0].sum(axis=0)
                    weight = float(model.body_mass.numpy().sum() * 9.81)
                    self.assertAlmostEqual(float(force[2] / weight), 1.0, delta=0.02)

    def test_dr_legs_support_and_reset(self):
        """Support DR Legs after impact and remain stable after a tipped-pose reset."""
        if not wp.is_cuda_available():
            self.skipTest("DR Legs acceptance exercises the CUDA graph path")
        from newton.examples.kamino.example_kamino_robot_dr_legs import Example  # noqa: PLC0415

        with wp.ScopedDevice("cuda:0"):
            args = SimpleNamespace(
                world_count=1,
                use_kamino_contacts=True,
                dynamics_solver="dvi",
                unilateral_solver="apgd",
                use_schur_complement=True,
                # Match the upstream support test; bounded rows have separate tests.
                joint_effort_limit=math.inf,
            )
            config_from_model = SolverKamino.Config.from_model

            def make_accuracy_config(*args, **kwargs):
                """Set the DR Legs residual budget before solver allocation and capture."""
                config = config_from_model(*args, **kwargs)
                config.dvi.apgd.max_nonlinear_corrections = 20
                return config

            with patch.object(SolverKamino.Config, "from_model", side_effect=make_accuracy_config):
                example = Example(ViewerNull(num_frames=1), args)
            base_z = []
            for _ in range(180):
                example.step()
                q = example.state_0.body_q.numpy()
                v = example.state_0.body_qd.numpy()
                self.assertTrue(np.all(np.isfinite(q)))
                self.assertTrue(np.all(np.isfinite(v)))
                self.assertLess(float(np.max(np.abs(v))), 100.0)
                base_z.append(float(q[0, 2]))
            contacts = example.solver._contacts_kamino
            aggregation = ContactAggregation(model=example.solver._model_kamino, contacts=contacts)
            aggregation.compute()
            force = aggregation.body_net_force.numpy()[0].sum(axis=0)
            weight = float(example.model.body_mass.numpy().sum() * 9.81)
            self.assertAlmostEqual(float(force[2] / weight), 1.0, delta=0.05)
            z = np.asarray(base_z[60:])
            t = np.arange(len(z))
            oscillation = z - np.polyval(np.polyfit(t, z, 1), t)
            self.assertLess(float(np.ptp(oscillation)), 0.001)
            info = example.solver._solver_kamino.solver_fd.data.status.numpy()[0]
            self.assertEqual(int(info["apgd_line_search_failed"]), 0)
            self.assertLess(float(info["apgd_residual"]), 1.1e-5)
            self.assertLess(float(info["r_b"]), 0.002)

            tip = wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), float(np.pi * 0.5))
            example.base_q.assign([wp.transformf((0.0, 0.0, 0.25), tip)])
            reset = SolverKamino.ResetConfig(base_pose=SolverKamino.ResetConfig.FromBaseQ(example.base_q))
            example.solver.reset(state=example.state_0, config=reset)
            example.solver.reset(state=example.state_1, config=reset)
            example.capture()
            start = example.state_0.body_q.numpy()[0, :2].copy()
            penetration = []
            settled_xy = []
            for step in range(400):
                example.step()
                count = int(contacts.world_active_contacts.numpy()[0])
                if step >= 40 and count:
                    penetration.append(float(max(0.0, -np.min(contacts.gapfunc.numpy()[:count, 3]))))
                if step >= 200:
                    settled_xy.append(example.state_0.body_q.numpy()[0, :2].copy())
            self.assertTrue(np.all(np.isfinite(example.state_0.body_qd.numpy())))
            self.assertGreater(len(penetration), 0)
            self.assertLess(float(np.percentile(penetration, 95)), 0.0035)
            self.assertLess(float(np.linalg.norm(settled_xy[-1] - start)), 0.008)
            self.assertLess(float(np.linalg.norm(settled_xy[-1] - settled_xy[0])), 2e-4)


if __name__ == "__main__":
    unittest.main()
