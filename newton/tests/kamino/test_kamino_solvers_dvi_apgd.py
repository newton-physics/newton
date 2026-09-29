# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Independent contact-law and integration regressions for unilateral APGD."""

import unittest
from types import SimpleNamespace

import numpy as np
import warp as wp

from newton._src.solvers.kamino._src.core.model import ModelKamino
from newton._src.solvers.kamino._src.dynamics.dual import DualProblem
from newton._src.solvers.kamino._src.linalg import LLTBlockedSolver
from newton._src.solvers.kamino._src.solvers.common import WarmStartMode
from newton._src.solvers.kamino._src.solvers.dvi import DVISolver
from newton._src.solvers.kamino._src.solvers.dvi.apgd import UnilateralAPGD
from newton._src.solvers.kamino._src.solvers.dvi.types import DVIConfigStruct, DVIStatus, convert_config_to_struct
from newton._src.solvers.kamino.config import ConstrainedDynamicsConfig, DVIAPGDConfig, DVISolverConfig
from newton.tests.kamino.utils.make import make_containers, update_containers
from newton.tests.utils import basics


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
            apgd=DVIAPGDConfig(max_iterations=128, max_corrections=30, tolerance=1e-6),
        ),
        warmstart=WarmStartMode.NONE,
    )
    return solver, problem


class TestDVIAPGD(unittest.TestCase):
    """Check Coulomb impulses rather than only the inner cone-QP residual."""

    def test_configuration_is_opt_in(self):
        """Retain PGS defaults and accept the experimental APGD backend."""
        self.assertEqual(DVISolverConfig().unilateral_solver, "pgs")
        self.assertEqual(DVISolverConfig(unilateral_solver="apgd").unilateral_solver, "apgd")

    def test_configuration_rejects_invalid_controls(self):
        """Reject unusable budgets, damping, and tolerances before launching kernels."""
        for field in ("max_iterations", "max_backtracks", "max_corrections"):
            for value in (0, -1, True, 1.5):
                with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                    DVIAPGDConfig(**{field: value})
        for options in (
            {"tolerance": float("nan")},
            {"tolerance": -1.0},
            {"relaxation": 0.0},
            {"relaxation": 1.1},
            {"relaxation": float("inf")},
            {"use_graph_conditionals": 1},
        ):
            with self.subTest(options=options), self.assertRaises(ValueError):
                DVIAPGDConfig(**options)
        with self.assertRaises(ValueError):
            DVISolverConfig(unilateral_solver="unknown")

    def test_de_saxce_sliding(self):
        """Recover Coulomb normal impulse instead of the associated cone-QP solution."""
        for device in _devices():
            with self.subTest(device=device):
                solver, problem = _problem([np.eye(3)], [[10.0, 0.0, -1.0]], [(0, 0, 1)], [0.5], device)
                solver.solve(problem)
                impulses = solver.owner.data.solution.lambdas.numpy()
                np.testing.assert_allclose(impulses, [-0.5, 0.0, 1.0], atol=2.0e-5, rtol=0.0)
                velocity = impulses + np.array([10.0, 0.0, -1.0])
                self.assertAlmostEqual(float(velocity[2]), 0.0, places=4)
                self.assertLess(float(solver.owner.data.status.numpy()[0]["apgd_residual"]), 1.1e-5)
                self.assertGreater(int(solver.owner.data.status.numpy()[0]["apgd_corrections"]), 1)

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
                    options=DVIAPGDConfig(max_iterations=100, max_corrections=40, tolerance=2e-6),
                )
                solver.solve(problem)
                np.testing.assert_allclose(solver.owner.data.solution.lambdas.numpy(), expected, atol=2e-5)

    def test_exhausted_line_search_retains_finite_iterate(self):
        """Reject an unverified step when the backtracking budget is exhausted."""
        for device in _devices():
            with self.subTest(device=device):
                solver, problem = _problem(
                    [1000.0 * np.eye(3)],
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
        """Expose an unconverged contact law when only one correction is allowed."""
        solver, problem = _problem(
            [np.eye(3)],
            [[10.0, 0.0, -1.0]],
            [(0, 0, 1)],
            [0.5],
            "cpu",
            options=DVIAPGDConfig(max_corrections=1),
        )
        solver.solve(problem)
        self.assertGreater(float(solver.owner.data.status.numpy()[0]["apgd_residual"]), 0.1)

    def test_warmstart_and_phase_masks(self):
        """Project stale warmstarts and preserve worlds outside their alternating budget."""
        for device in _devices():
            configs = [DVISolverConfig(unilateral_solver="apgd", max_alternating_iterations=n) for n in (1, 2)]
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

    def test_masked_fallback_matches_conditional_loops(self):
        """Run identical bounded iterations with and without conditional graphs."""
        for device in _devices():
            results = []
            for conditional in (False, True):
                options = DVIAPGDConfig(
                    max_iterations=3,
                    max_backtracks=2,
                    max_corrections=12,
                    use_graph_conditionals=conditional,
                )
                solver, problem = _problem(
                    [np.eye(3)], [[10.0, 0.0, -1.0]], [(0, 0, 1)], [0.5], device, options=options
                )
                solver.solve(problem)
                results.append((solver.owner.data.solution.lambdas.numpy(), solver.owner.data.status.numpy()))
            with self.subTest(device=device):
                np.testing.assert_array_equal(results[0][0], results[1][0])
                np.testing.assert_array_equal(results[0][1], results[1][1])

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
                    x = np.zeros_like(solver._apgd.x.numpy())
                    x[nb:n] = np.linspace(0.1, 1.0, n - nb)
                    solver._apgd.x.assign(x)
                    solver._apgd.matvec(solver._apgd.x, solver._apgd.product, solver.all_worlds_mask)
                    actual = solver._apgd.product.numpy()[nb:n]
                    expected = (a[nb:, nb:] - a[nb:, :nb] @ np.linalg.solve(regularized_b, a[:nb, nb:])) @ x[nb:n]
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
        from newton.tests.kamino.test_kamino_solvers_dvi import _build_five_box_stack  # noqa: PLC0415

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
        solver, problem = _problem([np.eye(3)], [[10.0, 0.0, -1.0]], [(0, 0, 1)], [0.5], "cuda:0")
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


if __name__ == "__main__":
    unittest.main()
