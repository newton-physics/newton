# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Numerical regressions for explicit Kamino DVI family coupling."""

from __future__ import annotations

import unittest
from unittest import mock

import numpy as np
import warp as wp

import newton
from newton._src.solvers.kamino._src.core.model import ModelKamino
from newton._src.solvers.kamino._src.dynamics.dual import DualProblem
from newton._src.solvers.kamino._src.linalg import LLTBlockedSolver
from newton._src.solvers.kamino._src.solvers.common import WarmStartMode
from newton._src.solvers.kamino._src.solvers.dvi import DVISolver
from newton._src.solvers.kamino._src.solvers.dvi.apgd_sparse import SparseContactOperator
from newton._src.solvers.kamino.config import (
    ConstrainedDynamicsConfig,
    ConstraintStabilizationConfig,
    DVIAPGDConfig,
    DVISolverConfig,
)
from newton.tests.kamino.utils.extract import extract_delassus, extract_problem_vector
from newton.tests.kamino.utils.make import make_containers, update_containers
from newton.tests.unittest_utils import add_function_test, get_test_devices


class TestDVIFamilyNumerics(unittest.TestCase):
    """Validate the numerical semantics of the explicit family schedule."""

    pass


def _build_coupled_problem(device: wp.DeviceLike, *, sparse: bool, preconditioning: bool = True):
    """Build one world with compliant bilateral/limit rows and rigid contacts."""
    builder = newton.ModelBuilder(gravity=wp.vec3f(0.0, 0.0, -9.81))
    builder.default_shape_cfg.margin = 0.0
    builder.default_shape_cfg.gap = 0.0
    builder.begin_world()
    body_xform = wp.transformf(wp.vec3f(0.4, 0.0, 0.45), wp.quat_identity(dtype=wp.float32))
    body = builder.add_link(mass=1.0, xform=body_xform)
    builder.add_shape_sphere(body=body, radius=0.5)
    joint = builder.add_joint_revolute(
        parent=-1,
        child=body,
        axis=newton.Axis.Y,
        parent_xform=body_xform,
        child_xform=wp.transform_identity(dtype=wp.float32),
        limit_lower=0.1,
        limit_upper=1.0,
    )
    builder.add_articulation([joint])
    builder.add_ground_plane()
    builder.end_world()

    model = ModelKamino.from_newton(builder.finalize(device=device))
    model, data, state, limits, detector, jacobians = make_containers(
        model=model,
        max_world_contacts=1,
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
    kwargs = {}
    if not sparse:
        kwargs["solver"] = LLTBlockedSolver
    problem = DualProblem(
        model=model,
        data=data,
        limits=limits,
        contacts=detector.contacts,
        jacobians=jacobians,
        sparse=sparse,
        config=DualProblem.Config(
            constraints=ConstraintStabilizationConfig(
                joint_compliance=2.0e-4,
                joint_stabilization_time=0.02,
                joint_limit_compliance=4.0e-4,
                joint_limit_stabilization_time=0.03,
            ),
            dynamics=ConstrainedDynamicsConfig(preconditioning=preconditioning),
        ),
        **kwargs,
    )
    problem.build(model=model, data=data, limits=limits, contacts=detector.contacts, jacobians=jacobians)

    np.testing.assert_array_equal(problem.data.njc.numpy(), np.array([5], dtype=np.int32))
    np.testing.assert_array_equal(problem.data.nbc.numpy(), np.array([0], dtype=np.int32))
    np.testing.assert_array_equal(problem.data.nl.numpy(), np.array([1], dtype=np.int32))
    np.testing.assert_array_equal(problem.data.nc.numpy(), np.array([1], dtype=np.int32))

    physical_compliance = extract_problem_vector(
        problem.delassus,
        problem.data.E.numpy(),
        only_active_dims=True,
    )[0]
    np.testing.assert_allclose(physical_compliance[:5], 2.0e-4 / (0.01 * (0.01 + 0.02)))
    np.testing.assert_allclose(physical_compliance[5], 4.0e-4 / (0.01 * (0.01 + 0.03)))
    np.testing.assert_array_equal(physical_compliance[6:], np.zeros(3, dtype=np.float32))
    return model, data, limits, detector.contacts, jacobians, problem


def _project_tangent(
    old: np.ndarray,
    velocity: np.ndarray,
    diagonal: np.ndarray,
    off_diagonal: float,
    regularization: float,
    omega: float,
    radius: float,
) -> np.ndarray:
    """Apply the DVI two-row tangential update independently in NumPy."""
    eps = np.finfo(np.float32).eps
    scalar_diagonal = float(np.max(diagonal))
    a00 = float(diagonal[0] + regularization)
    a11 = float(diagonal[1] + regularization)
    determinant = a00 * a11 - off_diagonal * off_diagonal
    updated = old.astype(np.float64, copy=True)
    if determinant > eps * a00 * a11:
        delta = np.array(
            [
                (a11 * velocity[0] - off_diagonal * velocity[1]) / determinant,
                (a00 * velocity[1] - off_diagonal * velocity[0]) / determinant,
            ]
        )
        updated -= omega * delta
    elif scalar_diagonal > eps:
        updated -= omega * velocity / (scalar_diagonal + regularization)

    tangent_norm = float(np.linalg.norm(updated))
    if tangent_norm > radius:
        updated = old.astype(np.float64, copy=True)
        if scalar_diagonal > eps:
            updated -= omega * velocity / (scalar_diagonal + regularization)
        tangent_norm = float(np.linalg.norm(updated))
    if tangent_norm > radius and tangent_norm > eps:
        updated *= radius / tangent_norm
    return updated


def _solve_numpy_family_sweep(
    matrix: np.ndarray,
    free_velocity: np.ndarray,
    represented_compliance: np.ndarray,
    friction: float,
    initial_lambdas: np.ndarray,
    *,
    num_bilateral: int,
    limit_row: int,
    contact_row: int,
    regularization: float,
    omega: float,
    post_stabilization_bilateral: bool,
) -> np.ndarray:
    """Evaluate one ``L -> B -> C`` sweep with optional post-stabilization bilateral solve."""
    eps = np.finfo(np.float32).eps
    lambdas = initial_lambdas.astype(np.float64, copy=True)

    velocity = matrix @ lambdas + free_velocity + represented_compliance * lambdas
    diagonal_limit = abs(float(matrix[limit_row, limit_row] + represented_compliance[limit_row]))
    if diagonal_limit > eps:
        lambdas[limit_row] = max(
            0.0,
            lambdas[limit_row] - omega * velocity[limit_row] / (diagonal_limit + regularization),
        )

    bilateral_operator = matrix[:num_bilateral, :num_bilateral] + np.diag(represented_compliance[:num_bilateral])
    bilateral_diagonal = np.diag(bilateral_operator)
    bilateral_preconditioner = np.sqrt(1.0 / (np.abs(bilateral_diagonal) + eps))
    bilateral_matrix = bilateral_preconditioner[:, None] * bilateral_operator * bilateral_preconditioner[
        None, :
    ] + 7.0e-7 * np.eye(num_bilateral)

    def solve_bilateral() -> None:
        rhs = bilateral_preconditioner * (
            -free_velocity[:num_bilateral] - matrix[:num_bilateral, num_bilateral:] @ lambdas[num_bilateral:]
        )
        lambdas[:num_bilateral] = bilateral_preconditioner * np.linalg.solve(bilateral_matrix, rhs)

    solve_bilateral()

    velocity = matrix @ lambdas + free_velocity + represented_compliance * lambdas
    normal_row = contact_row + 2
    diagonal_normal = abs(float(matrix[normal_row, normal_row] + represented_compliance[normal_row]))
    if diagonal_normal > eps:
        lambdas[normal_row] = max(
            0.0,
            lambdas[normal_row] - omega * velocity[normal_row] / (diagonal_normal + regularization),
        )

    velocity = matrix @ lambdas + free_velocity + represented_compliance * lambdas
    tangent_rows = slice(contact_row, contact_row + 2)
    tangent_diagonal = np.abs(
        np.diag(matrix)[contact_row : contact_row + 2] + represented_compliance[contact_row : contact_row + 2]
    )
    lambdas[tangent_rows] = _project_tangent(
        lambdas[tangent_rows],
        velocity[tangent_rows],
        tangent_diagonal,
        float(matrix[contact_row, contact_row + 1]),
        regularization,
        omega,
        friction * lambdas[normal_row],
    )

    if post_stabilization_bilateral:
        solve_bilateral()
    return lambdas


def _project_coulomb_cone(value: np.ndarray, friction: float) -> np.ndarray:
    """Project one ``[t0, t1, n]`` vector onto the Euclidean Coulomb cone."""
    tangent = value[:2]
    normal = float(value[2])
    tangent_norm = float(np.linalg.norm(tangent))
    if tangent_norm <= friction * normal:
        return value.copy()
    if friction * tangent_norm + normal <= 0.0 or friction == 0.0:
        return np.array([0.0, 0.0, max(0.0, normal)], dtype=np.float64)
    projected_normal = (friction * tangent_norm + normal) / (friction * friction + 1.0)
    projected_tangent = friction * projected_normal * tangent / tangent_norm
    return np.array([projected_tangent[0], projected_tangent[1], projected_normal])


def _solve_numpy_associated_contact(matrix: np.ndarray, rhs: np.ndarray, friction: float) -> np.ndarray:
    """Solve a three-row associated contact QP with reference projected gradient."""
    lipschitz = max(float(np.max(np.linalg.eigvalsh(matrix))), np.finfo(np.float64).eps)
    solution = np.zeros(3, dtype=np.float64)
    for _ in range(100_000):
        updated = _project_coulomb_cone(solution - (matrix @ solution - rhs) / lipschitz, friction)
        if np.linalg.norm(updated - solution) <= 1.0e-13 * max(1.0, np.linalg.norm(updated)):
            return updated
        solution = updated
    raise AssertionError("Reference associated contact solve did not converge.")


def _solve_numpy_apgd_family_sweep(
    matrix: np.ndarray,
    free_velocity: np.ndarray,
    represented_compliance: np.ndarray,
    friction: float,
    initial_lambdas: np.ndarray,
    *,
    num_bilateral: int,
    limit_row: int,
    contact_row: int,
    regularization: float,
    omega: float,
    post_stabilization_bilateral: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate ``L -> B -> associated C`` with optional post-stabilization bilateral solve."""
    eps = np.finfo(np.float32).eps
    lambdas = initial_lambdas.astype(np.float64, copy=True)

    velocity = matrix @ lambdas + free_velocity + represented_compliance * lambdas
    diagonal_limit = abs(float(matrix[limit_row, limit_row] + represented_compliance[limit_row]))
    if diagonal_limit > eps:
        lambdas[limit_row] = max(
            0.0,
            lambdas[limit_row] - omega * velocity[limit_row] / (diagonal_limit + regularization),
        )

    bilateral_operator = matrix[:num_bilateral, :num_bilateral] + np.diag(represented_compliance[:num_bilateral])
    bilateral_diagonal = np.diag(bilateral_operator)
    bilateral_preconditioner = np.sqrt(1.0 / (np.abs(bilateral_diagonal) + eps))
    bilateral_matrix = bilateral_preconditioner[:, None] * bilateral_operator * bilateral_preconditioner[
        None, :
    ] + 7.0e-7 * np.eye(num_bilateral)

    def solve_bilateral() -> None:
        rhs = bilateral_preconditioner * (
            -free_velocity[:num_bilateral] - matrix[:num_bilateral, num_bilateral:] @ lambdas[num_bilateral:]
        )
        lambdas[:num_bilateral] = bilateral_preconditioner * np.linalg.solve(bilateral_matrix, rhs)

    solve_bilateral()

    contact_rows = slice(contact_row, contact_row + 3)
    contact_rhs = -(free_velocity[contact_rows] + matrix[contact_rows, :contact_row] @ lambdas[:contact_row])
    contact_matrix = matrix[contact_rows, contact_rows] + np.diag(represented_compliance[contact_rows])
    lambdas[contact_rows] = _solve_numpy_associated_contact(contact_matrix, contact_rhs, friction)

    if post_stabilization_bilateral:
        solve_bilateral()
    return lambdas, contact_rhs


def _solve_family_fixture(
    device: wp.DeviceLike,
    *,
    sparse: bool,
    preconditioning: bool = True,
    post_stabilization_bilateral: bool,
    iterations: int,
):
    """Solve the coupled fixture from a fixed nonzero represented warm start."""
    model, data, limits, contacts, jacobians, problem = _build_coupled_problem(
        device,
        sparse=sparse,
        preconditioning=preconditioning,
    )
    config = DVISolverConfig(
        post_stabilization_bilateral=post_stabilization_bilateral,
        max_alternating_iterations=iterations,
        inequality_sweeps_per_iteration=1,
        tolerance=0.0,
        regularization=1.0e-6,
        omega=1.0,
    )
    solver = DVISolver(
        model=model,
        data=data,
        limits=limits,
        contacts=contacts,
        jacobians=jacobians,
        problem=problem if sparse else None,
        config=config,
        warmstart=WarmStartMode.NONE,
    )
    solver.coldstart()
    dimension = int(problem.data.dim.numpy()[0])
    initial = np.array([0.11, -0.07, 0.03, 0.09, -0.05, 0.4, 0.2, -0.15, 0.5], dtype=np.float32)
    if dimension != initial.size:
        raise AssertionError(f"Unexpected fixture dimension: {dimension}")
    packed_initial = np.zeros(solver.data.solution.lambdas.size, dtype=np.float32)
    packed_initial[:dimension] = initial
    solver.data.solution.lambdas.assign(packed_initial)
    solver.solve(problem)

    preconditioner = extract_problem_vector(problem.delassus, problem.data.P.numpy(), only_active_dims=True)[0]
    physical_lambdas = extract_problem_vector(
        problem.delassus,
        solver.data.solution.lambdas.numpy(),
        only_active_dims=True,
    )[0]
    represented_lambdas = physical_lambdas / preconditioner
    physical_velocity = extract_problem_vector(
        problem.delassus,
        solver.data.solution.v_plus.numpy(),
        only_active_dims=True,
    )[0]
    effective_velocity = extract_problem_vector(
        problem.delassus,
        solver.data.state.v_aug.numpy(),
        only_active_dims=True,
    )[0]
    return problem, config, initial, represented_lambdas, physical_lambdas, physical_velocity, effective_velocity


def _solve_apgd_family_fixture(
    device: wp.DeviceLike,
    *,
    sparse: bool,
    preconditioning: bool = True,
    post_stabilization_bilateral: bool,
    iterations: int = 1,
):
    """Solve the mixed family fixture with the associated APGD contact phase."""
    model, data, limits, contacts, jacobians, problem = _build_coupled_problem(
        device,
        sparse=sparse,
        preconditioning=preconditioning,
    )
    config = DVISolverConfig(
        contact_solver="apgd",
        post_stabilization_bilateral=post_stabilization_bilateral,
        max_alternating_iterations=iterations,
        inequality_sweeps_per_iteration=1,
        tolerance=0.0,
        regularization=1.0e-6,
        omega=1.0,
        apgd=DVIAPGDConfig(
            max_iterations=500,
            max_backtrack_iterations=20,
            tolerance=1.0e-8,
            min_iterations=1,
            early_exit=True,
            use_graph_conditionals=False,
        ),
    )
    solver = DVISolver(
        model=model,
        data=data,
        limits=limits,
        contacts=contacts,
        jacobians=jacobians,
        problem=problem,
        config=config,
        warmstart=WarmStartMode.NONE,
    )
    solver.coldstart()
    dimension = int(problem.data.dim.numpy()[0])
    initial = np.array([0.11, -0.07, 0.03, 0.09, -0.05, 0.4, 0.2, -0.15, 0.5], dtype=np.float32)
    if dimension != initial.size:
        raise AssertionError(f"Unexpected fixture dimension: {dimension}")
    packed_initial = np.zeros(solver.data.solution.lambdas.size, dtype=np.float32)
    packed_initial[:dimension] = initial
    solver.data.solution.lambdas.assign(packed_initial)
    solver.solve(problem)

    preconditioner = extract_problem_vector(problem.delassus, problem.data.P.numpy(), only_active_dims=True)[0]
    physical_lambdas = extract_problem_vector(
        problem.delassus,
        solver.data.solution.lambdas.numpy(),
        only_active_dims=True,
    )[0]
    represented_lambdas = physical_lambdas / preconditioner
    physical_velocity = extract_problem_vector(
        problem.delassus,
        solver.data.solution.v_plus.numpy(),
        only_active_dims=True,
    )[0]
    effective_velocity = extract_problem_vector(
        problem.delassus,
        solver.data.state.v_aug.numpy(),
        only_active_dims=True,
    )[0]
    return problem, config, initial, represented_lambdas, physical_lambdas, physical_velocity, effective_velocity


def _assert_compliant_solution_vectors(
    test: unittest.TestCase,
    problem: DualProblem,
    represented_lambdas: np.ndarray,
    physical_lambdas: np.ndarray,
    physical_velocity: np.ndarray,
    effective_velocity: np.ndarray,
) -> None:
    """Separate physical velocity from the constitutive velocity on B and L rows."""
    matrix = extract_delassus(problem.delassus, only_active_dims=True)[0].astype(np.float64)
    free_velocity = extract_problem_vector(
        problem.delassus,
        problem.data.v_f.numpy(),
        only_active_dims=True,
    )[0].astype(np.float64)
    preconditioner = extract_problem_vector(
        problem.delassus,
        problem.data.P.numpy(),
        only_active_dims=True,
    )[0].astype(np.float64)
    physical_compliance = extract_problem_vector(
        problem.delassus,
        problem.data.E.numpy(),
        only_active_dims=True,
    )[0].astype(np.float64)

    expected_physical_velocity = (matrix @ represented_lambdas + free_velocity) / preconditioner
    np.testing.assert_allclose(physical_velocity, expected_physical_velocity, rtol=3.0e-5, atol=3.0e-6)

    contact_row = int(problem.data.ccgo.numpy()[0])
    expected_effective_velocity = physical_velocity + physical_compliance * physical_lambdas
    np.testing.assert_allclose(
        effective_velocity[:contact_row],
        expected_effective_velocity[:contact_row],
        rtol=3.0e-5,
        atol=3.0e-6,
    )

    num_bilateral = int(problem.data.njc.numpy()[0])
    limit_row = int(problem.data.lcgo.numpy()[0])
    test.assertGreater(
        float(np.max(np.abs(physical_compliance[:num_bilateral] * physical_lambdas[:num_bilateral]))),
        1.0e-5,
    )
    test.assertGreater(abs(float(physical_compliance[limit_row] * physical_lambdas[limit_row])), 1.0e-5)


def test_family_sweep_matches_numpy_oracle(test: unittest.TestCase, device: wp.DeviceLike) -> None:
    """Match a warm-started ``L -> B -> C`` oracle with optional post-stabilization."""
    outputs = {}
    for sparse in (False, True):
        for preconditioning in (False, True):
            for post_stabilization_bilateral in (False, True):
                with test.subTest(
                    sparse=sparse,
                    preconditioning=preconditioning,
                    post_stabilization_bilateral=post_stabilization_bilateral,
                ):
                    problem, config, initial, actual, physical_lambdas, physical_velocity, effective_velocity = (
                        _solve_family_fixture(
                            device,
                            sparse=sparse,
                            preconditioning=preconditioning,
                            post_stabilization_bilateral=post_stabilization_bilateral,
                            iterations=1,
                        )
                    )
                    matrix = extract_delassus(problem.delassus, only_active_dims=True)[0].astype(np.float64)
                    free_velocity = extract_problem_vector(
                        problem.delassus,
                        problem.data.v_f.numpy(),
                        only_active_dims=True,
                    )[0].astype(np.float64)
                    compliance = extract_problem_vector(
                        problem.delassus,
                        problem.data.E_hat.numpy(),
                        only_active_dims=True,
                    )[0].astype(np.float64)
                    expected = _solve_numpy_family_sweep(
                        matrix,
                        free_velocity,
                        compliance,
                        float(problem.data.mu.numpy()[int(problem.data.cio.numpy()[0])]),
                        initial,
                        num_bilateral=int(problem.data.njc.numpy()[0]),
                        limit_row=int(problem.data.lcgo.numpy()[0]),
                        contact_row=int(problem.data.ccgo.numpy()[0]),
                        regularization=config.regularization,
                        omega=config.omega,
                        post_stabilization_bilateral=post_stabilization_bilateral,
                    )
                    np.testing.assert_allclose(actual, expected, rtol=2.0e-5, atol=2.0e-6)
                    _assert_compliant_solution_vectors(
                        test,
                        problem,
                        actual,
                        physical_lambdas,
                        physical_velocity,
                        effective_velocity,
                    )
                    outputs[sparse, preconditioning, post_stabilization_bilateral] = (
                        matrix,
                        free_velocity,
                        compliance,
                        actual,
                        int(problem.data.njc.numpy()[0]),
                    )

            matrix, free_velocity, compliance, without_post_stabilization, num_bilateral = outputs[
                sparse, preconditioning, False
            ]
            _, _, _, with_post_stabilization, _ = outputs[sparse, preconditioning, True]
            residual_without = np.max(
                np.abs(
                    matrix[:num_bilateral] @ without_post_stabilization
                    + free_velocity[:num_bilateral]
                    + compliance[:num_bilateral] * without_post_stabilization[:num_bilateral]
                )
            )
            residual_with = np.max(
                np.abs(
                    matrix[:num_bilateral] @ with_post_stabilization
                    + free_velocity[:num_bilateral]
                    + compliance[:num_bilateral] * with_post_stabilization[:num_bilateral]
                )
            )
            test.assertGreater(residual_without, 1.0e-3)
            test.assertLess(residual_with, 5.0e-6)
            test.assertLess(residual_with, 0.01 * residual_without)


def test_family_dense_sparse_parity_and_repeatability(test: unittest.TestCase, device: wp.DeviceLike) -> None:
    """Keep dense and sparse family solutions aligned and deterministic."""
    for post_stabilization_bilateral in (False, True):
        with test.subTest(post_stabilization_bilateral=post_stabilization_bilateral):
            results = {}
            for sparse in (False, True):
                first = _solve_family_fixture(
                    device,
                    sparse=sparse,
                    post_stabilization_bilateral=post_stabilization_bilateral,
                    iterations=4,
                )
                second = _solve_family_fixture(
                    device,
                    sparse=sparse,
                    post_stabilization_bilateral=post_stabilization_bilateral,
                    iterations=4,
                )
                np.testing.assert_array_equal(first[4], second[4])
                np.testing.assert_array_equal(first[5], second[5])
                np.testing.assert_array_equal(first[6], second[6])
                results[sparse] = first

            if False in results and True in results:
                np.testing.assert_allclose(results[False][4], results[True][4], rtol=2.0e-4, atol=2.0e-5)
                np.testing.assert_allclose(results[False][5], results[True][5], rtol=2.0e-4, atol=2.0e-5)
                np.testing.assert_allclose(results[False][6], results[True][6], rtol=2.0e-4, atol=2.0e-5)


def test_apgd_mixed_family_sweep_matches_associated_oracle(test: unittest.TestCase, device: wp.DeviceLike) -> None:
    """Couple APGD numerically to the latest limit and bilateral reactions."""
    outputs = {}
    for sparse in (False, True):
        for preconditioning in (False, True):
            for post_stabilization_bilateral in (False, True):
                with test.subTest(
                    sparse=sparse,
                    preconditioning=preconditioning,
                    post_stabilization_bilateral=post_stabilization_bilateral,
                ):
                    problem, config, initial, actual, physical_lambdas, physical_velocity, effective_velocity = (
                        _solve_apgd_family_fixture(
                            device,
                            sparse=sparse,
                            preconditioning=preconditioning,
                            post_stabilization_bilateral=post_stabilization_bilateral,
                        )
                    )
                    matrix = extract_delassus(problem.delassus, only_active_dims=True)[0].astype(np.float64)
                    free_velocity = extract_problem_vector(
                        problem.delassus,
                        problem.data.v_f.numpy(),
                        only_active_dims=True,
                    )[0].astype(np.float64)
                    compliance = extract_problem_vector(
                        problem.delassus,
                        problem.data.E_hat.numpy(),
                        only_active_dims=True,
                    )[0].astype(np.float64)
                    num_bilateral = int(problem.data.njc.numpy()[0])
                    limit_row = int(problem.data.lcgo.numpy()[0])
                    contact_row = int(problem.data.ccgo.numpy()[0])
                    expected, coupled_rhs = _solve_numpy_apgd_family_sweep(
                        matrix,
                        free_velocity,
                        compliance,
                        float(problem.data.mu.numpy()[int(problem.data.cio.numpy()[0])]),
                        initial,
                        num_bilateral=num_bilateral,
                        limit_row=limit_row,
                        contact_row=contact_row,
                        regularization=config.regularization,
                        omega=config.omega,
                        post_stabilization_bilateral=post_stabilization_bilateral,
                    )
                    uncoupled_rhs = -free_velocity[contact_row : contact_row + 3]
                    test.assertGreater(float(np.linalg.norm(coupled_rhs - uncoupled_rhs)), 1.0e-3)
                    np.testing.assert_allclose(actual, expected, rtol=3.0e-4, atol=3.0e-5)
                    _assert_compliant_solution_vectors(
                        test,
                        problem,
                        actual,
                        physical_lambdas,
                        physical_velocity,
                        effective_velocity,
                    )
                    outputs[sparse, preconditioning, post_stabilization_bilateral] = actual

    for preconditioning in (False, True):
        for post_stabilization_bilateral in (False, True):
            np.testing.assert_allclose(
                outputs[False, preconditioning, post_stabilization_bilateral],
                outputs[True, preconditioning, post_stabilization_bilateral],
                rtol=3.0e-4,
                atol=3.0e-5,
            )


def test_sparse_apgd_prepares_adjacency_once_per_solve(test: unittest.TestCase, device: wp.DeviceLike) -> None:
    """Build dynamic sparse adjacency once across all family iterations."""
    prepare_count = 0
    original_prepare = SparseContactOperator.prepare

    def counted_prepare(operator: SparseContactOperator) -> None:
        """Count and execute sparse topology preparation."""
        nonlocal prepare_count
        prepare_count += 1
        original_prepare(operator)

    with mock.patch.object(SparseContactOperator, "prepare", new=counted_prepare):
        _solve_apgd_family_fixture(
            device,
            sparse=True,
            post_stabilization_bilateral=False,
            iterations=3,
        )
    test.assertEqual(prepare_count, 1)


_DEVICES = get_test_devices(mode="basic")

add_function_test(
    TestDVIFamilyNumerics,
    "test_family_sweep_matches_numpy_oracle",
    test_family_sweep_matches_numpy_oracle,
    devices=_DEVICES,
)
add_function_test(
    TestDVIFamilyNumerics,
    "test_apgd_mixed_family_sweep_matches_associated_oracle",
    test_apgd_mixed_family_sweep_matches_associated_oracle,
    devices=_DEVICES,
)
add_function_test(
    TestDVIFamilyNumerics,
    "test_family_dense_sparse_parity_and_repeatability",
    test_family_dense_sparse_parity_and_repeatability,
    devices=_DEVICES,
)
add_function_test(
    TestDVIFamilyNumerics,
    "test_sparse_apgd_prepares_adjacency_once_per_solve",
    test_sparse_apgd_prepares_adjacency_once_per_solve,
    devices=_DEVICES,
)


if __name__ == "__main__":
    unittest.main()
