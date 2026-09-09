# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Focused tests for the Kamino contact APGD engine."""

from __future__ import annotations

import unittest

import numpy as np
import warp as wp

from newton._src.solvers.kamino._src.solvers.dvi.apgd import ContactAPGDOptions, ContactAPGDSolver
from newton._src.solvers.kamino._src.solvers.dvi.apgd_kernels import (
    _ACC_RESIDUAL_SQUARED,
    _GDIFF,
    _RESIDUAL_SQUARED,
)
from newton.tests.unittest_utils import add_function_test, get_test_devices


class TestDVIContactAPGD(unittest.TestCase):
    """Validate associated contact APGD independently of solver dispatch."""

    pass


@wp.kernel
def _increment_matvec_count(counter: wp.array[wp.int32]):
    """Count a represented-operator invocation on the device."""
    wp.atomic_add(counter, 0, 1)


@wp.kernel
def _identity_contact_matvec(
    contact_count: wp.array[wp.int32],
    contact_row_offset: wp.array[wp.int32],
    world_mask: wp.array[bool],
    x: wp.array[wp.float32],
    y: wp.array[wp.float32],
):
    """Apply an identity operator to active compact contact rows."""
    wid, local_row = wp.tid()
    if world_mask[wid] and local_row < 3 * contact_count[wid]:
        row = contact_row_offset[wid] + local_row
        y[row] = x[row]


def _project_coulomb(vector: np.ndarray, friction: float) -> np.ndarray:
    """Return an independent double-precision projection in ``[t0,t1,n]`` order."""
    tangent_norm = float(np.linalg.norm(vector[:2]))
    normal = float(vector[2])
    if friction * tangent_norm <= -normal:
        return np.zeros(3, dtype=np.float64)
    if tangent_norm <= friction * normal:
        return vector.astype(np.float64, copy=True)
    projected_normal = (friction * tangent_norm + normal) / (friction * friction + 1.0)
    projected_tangent = friction * projected_normal * vector[:2] / tangent_norm
    return np.array([projected_tangent[0], projected_tangent[1], projected_normal], dtype=np.float64)


def _project_product(vector: np.ndarray, friction: np.ndarray) -> np.ndarray:
    """Project a flat vector onto a product of associated Coulomb cones."""
    return np.concatenate(
        [_project_coulomb(vector[3 * cid : 3 * cid + 3], float(mu)) for cid, mu in enumerate(friction)]
    )


def _solve_numpy_contact_qp(matrix: np.ndarray, rhs: np.ndarray, friction: np.ndarray) -> np.ndarray:
    """Solve a small strongly-convex product-cone QP with plain projected gradient."""
    solution = np.zeros_like(rhs, dtype=np.float64)
    step = 1.0 / float(np.linalg.eigvalsh(matrix).max())
    for _ in range(100_000):
        solution_new = _project_product(solution - step * (matrix @ solution - rhs), friction)
        if np.linalg.norm(solution_new - solution) <= 1.0e-13:
            return solution_new
        solution = solution_new
    raise AssertionError("NumPy contact QP oracle failed to converge.")


def _make_dense_fixture(
    device: wp.DeviceLike,
    matrices: list[np.ndarray],
    free_velocities: list[np.ndarray],
    contact_counts: list[int],
    contact_group_offsets: list[int],
    friction: np.ndarray,
    *,
    contact_capacities: list[int] | None = None,
    represented_compliance: list[np.ndarray] | None = None,
    full_solution: list[np.ndarray] | None = None,
    config: ContactAPGDOptions | list[ContactAPGDOptions] | None = None,
):
    """Create an isolated packed dense problem and its APGD adapter."""
    if contact_capacities is None:
        contact_capacities = contact_counts
    if represented_compliance is None:
        represented_compliance = [np.zeros(matrix.shape[0], dtype=np.float32) for matrix in matrices]
    if full_solution is None:
        full_solution = [np.zeros(matrix.shape[0], dtype=np.float32) for matrix in matrices]

    dimensions = [matrix.shape[0] for matrix in matrices]
    matrix_offsets = np.cumsum([0, *(dimension * dimension for dimension in dimensions[:-1])])
    vector_offsets = np.cumsum([0, *dimensions[:-1]])
    contact_offsets = np.cumsum([0, *contact_capacities[:-1]])

    solver = ContactAPGDSolver(contact_capacities, config, device=device)

    def to_int(values) -> wp.array[wp.int32]:
        """Copy integer fixture data to the selected device."""
        return wp.array(np.asarray(values, dtype=np.int32), dtype=wp.int32, device=device)

    def to_float(values) -> wp.array[wp.float32]:
        """Copy floating-point fixture data to the selected device."""
        return wp.array(np.asarray(values, dtype=np.float32), dtype=wp.float32, device=device)

    operator = solver.make_dense_operator(
        problem_dim=to_int(dimensions),
        problem_mio=to_int(matrix_offsets),
        problem_vio=to_int(vector_offsets),
        problem_nc=to_int(contact_counts),
        problem_cio=to_int(contact_offsets),
        problem_ccgo=to_int(contact_group_offsets),
        matrix=to_float(np.concatenate([matrix.astype(np.float32).ravel() for matrix in matrices])),
        represented_compliance=to_float(np.concatenate(represented_compliance)),
        free_velocity=to_float(np.concatenate(free_velocities)),
    )
    return (
        solver,
        operator,
        to_float(friction),
        to_float(np.concatenate(full_solution)),
    )


def test_associated_analytic_contact_modes(test: unittest.TestCase, device: wp.DeviceLike) -> None:
    """Solve opening, frictionless, sticking, and sliding scalar-block contacts analytically."""
    alphas = np.array([1.0, 2.0, 2.0, 2.0], dtype=np.float64)
    right_hand_sides = np.array(
        [
            [0.2, -0.1, -1.0],
            [7.0, -9.0, 2.0],
            [0.2, -0.1, 2.0],
            [4.0, 0.0, 2.0],
        ],
        dtype=np.float64,
    )
    friction_np = np.array([0.5, 0.0, 0.5, 0.5], dtype=np.float32)
    config = ContactAPGDOptions(max_iterations=10, tolerance=2.0e-5, use_graph_conditionals=False)
    solver, operator, friction, solution = _make_dense_fixture(
        device,
        [alpha * np.eye(3) for alpha in alphas],
        [(-rhs).astype(np.float32) for rhs in right_hand_sides],
        [1, 1, 1, 1],
        [0, 0, 0, 0],
        friction_np,
        config=config,
    )

    solver.solve_dense(operator, friction, solution)
    expected = np.concatenate(
        [
            _project_coulomb(rhs / alpha, float(mu))
            for alpha, rhs, mu in zip(alphas, right_hand_sides, friction_np, strict=True)
        ]
    )
    np.testing.assert_allclose(solution.numpy(), expected, rtol=2.0e-5, atol=2.0e-5)
    status = solver.status.numpy()
    # A zero opening impulse is a genuine KKT solution: its natural-map Res4
    # vanishes, so it must stop just like the nonzero contact modes.
    np.testing.assert_array_equal(status["converged"], np.ones(4, dtype=np.int32))
    np.testing.assert_array_equal(status["iterations"], np.ones(4, dtype=np.int32))
    test.assertTrue(np.all(status["residual"] <= config.tolerance))


def test_dense_operator_offsets_rhs_and_compliance(test: unittest.TestCase, device: wp.DeviceLike) -> None:
    """Check heterogeneous compact offsets, cross-family RHS, and one compliance contribution."""
    rng = np.random.default_rng(712)
    factor_0 = rng.normal(size=(5, 5))
    factor_1 = rng.normal(size=(4, 4))
    matrices = [factor_0.T @ factor_0 + np.eye(5), factor_1.T @ factor_1 + np.eye(4)]
    free_velocities = [np.array([0.3, -0.2, 0.1, 0.4, -0.5]), np.array([-0.1, 0.2, -0.4, 0.7])]
    compliance = [np.array([0.0, 0.0, 0.2, 0.3, 0.4]), np.array([0.0, 0.5, 0.6, 0.7])]
    lambdas = [np.array([0.6, -0.3, 9.0, 8.0, 7.0]), np.array([-0.2, 6.0, 5.0, 4.0])]
    solver, operator, _, full_solution = _make_dense_fixture(
        device,
        matrices,
        free_velocities,
        [1, 1],
        [2, 1],
        np.array([0.4, np.nan, 0.7], dtype=np.float32),
        contact_capacities=[2, 1],
        represented_compliance=compliance,
        full_solution=lambdas,
        config=ContactAPGDOptions(max_iterations=2, use_graph_conditionals=False),
    )
    mask = wp.ones(2, dtype=wp.bool, device=device)
    compact_x_np = np.array([1.0, 2.0, 3.0, np.nan, np.nan, np.nan, 4.0, 5.0, 6.0], dtype=np.float32)
    compact_x = wp.array(compact_x_np, dtype=wp.float32, device=device)
    product = wp.full(9, -123.0, dtype=wp.float32, device=device)

    operator.matvec(compact_x, product, mask)
    operator.build_rhs(full_solution, solver.rhs, mask)
    gathered = wp.full(9, np.nan, dtype=wp.float32, device=device)
    operator.gather(full_solution, gathered, mask)

    product_np = product.numpy()
    rhs_np = solver.rhs.numpy()
    gathered_np = gathered.numpy()
    for matrix, velocity, diagonal, lambda_world, ccgo, compact_begin in zip(
        matrices, free_velocities, compliance, lambdas, [2, 1], [0, 6], strict=True
    ):
        contact_slice = slice(ccgo, ccgo + 3)
        compact_slice = slice(compact_begin, compact_begin + 3)
        x_contact = compact_x_np[compact_slice]
        expected_product = matrix[contact_slice, contact_slice] @ x_contact + diagonal[contact_slice] * x_contact
        lambda_noncontact = lambda_world.copy()
        lambda_noncontact[contact_slice] = 0.0
        expected_rhs = -(velocity[contact_slice] + matrix[contact_slice] @ lambda_noncontact)
        np.testing.assert_allclose(product_np[compact_slice], expected_product, rtol=1.0e-5, atol=1.0e-5)
        np.testing.assert_allclose(rhs_np[compact_slice], expected_rhs, rtol=1.0e-5, atol=1.0e-5)
        np.testing.assert_array_equal(gathered_np[compact_slice], lambda_world[contact_slice])

    np.testing.assert_array_equal(solver.contact_offset.numpy(), np.array([0, 2, 3], dtype=np.int32))
    np.testing.assert_array_equal(operator.problem_cio.numpy(), np.array([0, 2], dtype=np.int32))
    test.assertTrue(np.all(product_np[3:6] == -123.0))
    test.assertTrue(np.all(np.isnan(gathered_np[3:6])))


def test_coupled_qp_and_represented_preconditioning(test: unittest.TestCase, device: wp.DeviceLike) -> None:
    """Match a coupled NumPy oracle after cone-compatible represented-coordinate scaling."""
    rng = np.random.default_rng(3)
    factor = rng.normal(size=(6, 6))
    matrix = factor.T @ factor + 2.0 * np.eye(6)
    rhs = np.array([1.2, -0.5, 1.0, -0.2, 0.9, 1.5], dtype=np.float64)
    friction_np = np.array([0.4, 0.7], dtype=np.float64)
    expected = _solve_numpy_contact_qp(matrix, rhs, friction_np)

    preconditioner = np.diag([0.25, 0.25, 0.25, 2.0, 2.0, 2.0])
    represented_matrix = preconditioner @ matrix @ preconditioner
    represented_rhs = preconditioner @ rhs
    solver, operator, friction, solution = _make_dense_fixture(
        device,
        [represented_matrix],
        [(-represented_rhs).astype(np.float32)],
        [2],
        [0],
        friction_np.astype(np.float32),
        config=ContactAPGDOptions(max_iterations=300, tolerance=5.0e-5, use_graph_conditionals=False),
    )

    solver.solve_dense(operator, friction, solution)
    physical_solution = preconditioner @ solution.numpy()
    np.testing.assert_allclose(physical_solution, expected, rtol=2.0e-4, atol=3.0e-5)
    status = solver.status.numpy()[0]
    test.assertEqual(int(status["converged"]), 1)
    test.assertLessEqual(float(status["residual"]), 5.0e-5)


def test_warmstart_reprojects_opening_and_changed_friction(test: unittest.TestCase, device: wp.DeviceLike) -> None:
    """Ensure warm starts remain feasible when a contact opens or its friction drops."""
    config = ContactAPGDOptions(max_iterations=20, tolerance=2.0e-5, use_graph_conditionals=False)
    rhs = np.array([2.0, 0.0, 1.0], dtype=np.float32)
    solver, operator, friction, solution = _make_dense_fixture(
        device,
        [np.eye(3)],
        [-rhs],
        [1],
        [0],
        np.array([1.0], dtype=np.float32),
        config=config,
    )
    solver.solve_dense(operator, friction, solution)
    high_friction_solution = solution.numpy().copy()
    np.testing.assert_allclose(high_friction_solution, _project_coulomb(rhs, 1.0), atol=2.0e-5)

    friction.assign(np.array([0.2], dtype=np.float32))
    solver.solve_dense(operator, friction, solution)
    np.testing.assert_allclose(solution.numpy(), _project_coulomb(rhs, 0.2), atol=2.0e-5)
    test.assertLess(np.linalg.norm(solution.numpy()[:2]), np.linalg.norm(high_friction_solution[:2]))

    operator.free_velocity.assign(np.array([0.0, 0.0, 1.0], dtype=np.float32))
    solver.solve_dense(operator, friction, solution)
    np.testing.assert_allclose(solution.numpy(), np.zeros(3), atol=2.0e-5)
    opening_status = solver.status.numpy()[0]
    test.assertEqual(int(opening_status["converged"]), 1)
    test.assertEqual(int(opening_status["iterations"]), config.min_iterations)


def test_backtracking_caps_and_actual_counters(test: unittest.TestCase, device: wp.DeviceLike) -> None:
    """Count the initial descent check as the first backtracking pass."""
    stiff_direction = np.array([1.0, -1.0, 0.0]) / np.sqrt(2.0)
    matrix = np.eye(3) + 99.0 * np.outer(stiff_direction, stiff_direction)
    rhs = np.array([100.0, -100.0, 20.0], dtype=np.float32)
    configs = [
        ContactAPGDOptions(
            max_iterations=1,
            max_backtrack_iterations=backtrack_cap,
            tolerance=0.0,
            early_exit=False,
            use_graph_conditionals=False,
        )
        for backtrack_cap in range(4)
    ]
    solver, operator, friction, solution = _make_dense_fixture(
        device,
        [matrix] * 4,
        [-rhs] * 4,
        [1] * 4,
        [0] * 4,
        np.full(4, 10.0, dtype=np.float32),
        config=configs,
    )

    solver.solve_dense(operator, friction, solution)
    status = solver.status.numpy()
    np.testing.assert_array_equal(status["iterations"], np.ones(4, dtype=np.int32))
    np.testing.assert_array_equal(status["backtracks"], np.arange(4, dtype=np.int32))
    solutions = solution.numpy().reshape(4, 3)
    # Pass zero disables the check. Pass one may double L, but (as in final
    # newton-dvi) there is no second projection until pass two is permitted.
    np.testing.assert_array_equal(solutions[1], solutions[0])
    np.testing.assert_array_equal(solutions[2], 0.5 * solutions[1])
    np.testing.assert_array_equal(solutions[3], 0.5 * solutions[2])


def test_eager_conditionals_skip_predicated_work(test: unittest.TestCase, device: wp.DeviceLike) -> None:
    """Use device conditionals eagerly instead of launching every masked pass."""
    if device.is_cuda and not wp.is_conditional_graph_supported():
        test.skipTest("CUDA graph conditional nodes are not supported on this device.")

    max_iterations = 10
    max_backtracks = 5

    def solve_and_count(use_graph_conditionals: bool) -> tuple[int, np.void]:
        config = ContactAPGDOptions(
            max_iterations=max_iterations,
            max_backtrack_iterations=max_backtracks,
            tolerance=1.0e-6,
            use_graph_conditionals=use_graph_conditionals,
        )
        solver, operator, friction, full_solution = _make_dense_fixture(
            device,
            [np.eye(3)],
            [np.array([0.0, 0.0, -1.0], dtype=np.float32)],
            [1],
            [0],
            np.array([0.5], dtype=np.float32),
            config=config,
        )
        mask = wp.ones(1, dtype=wp.bool, device=device)
        operator.build_rhs(full_solution, solver.rhs, mask)
        operator.gather(full_solution, solver.solution, mask)
        counter = wp.zeros(1, dtype=wp.int32, device=device)

        def counted_matvec(x: wp.array, y: wp.array, world_mask: wp.array) -> None:
            operator.matvec(x, y, world_mask)
            wp.launch(_increment_matvec_count, dim=1, inputs=[counter], device=device)

        solver.solve(
            operator.problem_nc,
            friction,
            solver.rhs,
            solver.solution,
            counted_matvec,
            phase_mask=mask,
            contact_offset=operator.problem_cio,
        )
        return int(counter.numpy()[0]), solver.status.numpy()[0]

    conditional_calls, conditional_status = solve_and_count(True)
    fallback_calls, fallback_status = solve_and_count(False)
    test.assertEqual(int(conditional_status["iterations"]), 1)
    test.assertEqual(int(fallback_status["iterations"]), 1)
    test.assertEqual(conditional_calls, 3)  # Rayleigh, A*y, and A*gamma_new.
    test.assertEqual(fallback_calls, 1 + max_iterations * (max_backtracks + 1))


def test_fixed_order_float64_reduction(test: unittest.TestCase, device: wp.DeviceLike) -> None:
    """Preserve small contact partials accumulated after a large contribution."""
    contact_capacity = 512
    solver = ContactAPGDSolver(
        [contact_capacity],
        ContactAPGDOptions(max_iterations=1, use_graph_conditionals=False),
        device=device,
    )
    partials = np.zeros(solver._partials.shape, dtype=np.float32)
    partials[_ACC_RESIDUAL_SQUARED, :contact_capacity] = 1.0
    partials[_ACC_RESIDUAL_SQUARED, 0] = 1.0e8
    solver._partials.assign(partials)
    contact_count = wp.array([contact_capacity], dtype=wp.int32, device=device)
    mask = wp.ones(1, dtype=wp.bool, device=device)
    solver._reduce_partials(
        contact_count,
        mask,
        _ACC_RESIDUAL_SQUARED,
        -1,
        _RESIDUAL_SQUARED,
        -1,
    )

    actual = solver._scalars.numpy()[0, _RESIDUAL_SQUARED]
    expected = np.float32(np.sum(partials[_ACC_RESIDUAL_SQUARED], dtype=np.float64))
    test.assertEqual(actual, expected)
    test.assertGreater(actual, np.float32(1.0e8))


def test_res4_step_avoids_large_contact_count_overflow(test: unittest.TestCase, device: wp.DeviceLike) -> None:
    """Compute ``1 / (3*nc)^2`` without overflowing an int32 intermediate."""
    contact_count_value = 15_447
    row_count = 3 * contact_count_value
    solver = ContactAPGDSolver(
        [contact_count_value],
        ContactAPGDOptions(
            max_iterations=1,
            max_backtrack_iterations=0,
            tolerance=0.0,
            early_exit=False,
            use_graph_conditionals=False,
        ),
        device=device,
    )
    contact_count = wp.array([contact_count_value], dtype=wp.int32, device=device)
    friction = wp.zeros(contact_count_value, dtype=wp.float32, device=device)
    rhs = wp.zeros(row_count, dtype=wp.float32, device=device)
    solution = wp.zeros(row_count, dtype=wp.float32, device=device)

    def identity_matvec(x: wp.array, y: wp.array, world_mask: wp.array) -> None:
        wp.launch(
            _identity_contact_matvec,
            dim=(1, row_count),
            inputs=[contact_count, solver.contact_row_offset, world_mask, x, y],
            device=device,
        )

    solver.solve(contact_count, friction, rhs, solution, identity_matvec)
    actual = solver._scalars.numpy()[0, _GDIFF]
    expected = np.float32(1.0 / float(row_count * row_count))
    np.testing.assert_allclose(actual, expected, rtol=2.0e-7, atol=0.0)


def test_deterministic_reuse_and_best_res4(test: unittest.TestCase, device: wp.DeviceLike) -> None:
    """Repeat a backtracking solve bit-for-bit while returning its minimum-Res4 iterate."""
    matrix = np.diag([100.0, 1.0, 1.0])
    rhs = np.array([100.0, 0.0, 20.0], dtype=np.float32)
    config = ContactAPGDOptions(max_iterations=100, tolerance=1.0e-4, use_graph_conditionals=False)
    solver, operator, friction, solution = _make_dense_fixture(
        device,
        [matrix],
        [-rhs],
        [1],
        [0],
        np.array([10.0], dtype=np.float32),
        config=config,
    )

    solutions = []
    statuses = []
    for _ in range(5):
        solution.zero_()
        solver.solve_dense(operator, friction, solution)
        solutions.append(solution.numpy().copy())
        statuses.append(solver.status.numpy().copy())

    for repeated_solution, repeated_status in zip(solutions[1:], statuses[1:], strict=True):
        np.testing.assert_array_equal(repeated_solution, solutions[0])
        np.testing.assert_array_equal(repeated_status, statuses[0])
    np.testing.assert_allclose(solutions[0], np.array([1.0, 0.0, 20.0]), rtol=1.0e-4, atol=1.0e-4)
    terminal = statuses[0][0]
    test.assertEqual(int(terminal["converged"]), 1)
    test.assertLess(int(terminal["iterations"]), config.max_iterations)
    test.assertGreater(int(terminal["backtracks"]), 0)
    test.assertGreater(int(terminal["restarts"]), 0)
    test.assertLessEqual(float(terminal["residual"]), config.tolerance)


def test_nested_graph_conditionals_match_predicated_fallback(test: unittest.TestCase, device: wp.DeviceLike) -> None:
    """Match fixed loops with nested captured outer/backtracking loops and reusable replay."""
    if device.is_cuda and not wp.is_conditional_graph_supported():
        test.skipTest("CUDA graph conditional nodes are not supported on this device.")

    matrix = np.diag([100.0, 1.0, 1.0])
    rhs = np.array([100.0, 0.0, 20.0], dtype=np.float32)
    fallback_config = ContactAPGDOptions(max_iterations=100, tolerance=1.0e-4, use_graph_conditionals=False)
    conditional_config = ContactAPGDOptions(max_iterations=100, tolerance=1.0e-4, use_graph_conditionals=True)
    fallback, fallback_operator, fallback_friction, fallback_solution = _make_dense_fixture(
        device,
        [matrix],
        [-rhs],
        [1],
        [0],
        np.array([10.0], dtype=np.float32),
        config=fallback_config,
    )
    conditional, conditional_operator, conditional_friction, conditional_solution = _make_dense_fixture(
        device,
        [matrix],
        [-rhs],
        [1],
        [0],
        np.array([10.0], dtype=np.float32),
        config=conditional_config,
    )
    fallback.solve_dense(fallback_operator, fallback_friction, fallback_solution)
    wp.synchronize_device(device)

    conditional.solve_dense(conditional_operator, conditional_friction, conditional_solution)
    wp.synchronize_device(device)
    conditional_solution.zero_()
    with wp.ScopedCapture(device) as capture:
        conditional.solve_dense(conditional_operator, conditional_friction, conditional_solution)

    captured_solutions = []
    captured_statuses = []
    for _ in range(2):
        conditional_solution.zero_()
        wp.capture_launch(capture.graph)
        captured_solutions.append(conditional_solution.numpy().copy())
        captured_statuses.append(conditional.status.numpy().copy())

    np.testing.assert_array_equal(captured_solutions[0], fallback_solution.numpy())
    np.testing.assert_array_equal(captured_statuses[0], fallback.status.numpy())
    np.testing.assert_array_equal(captured_solutions[1], captured_solutions[0])
    np.testing.assert_array_equal(captured_statuses[1], captured_statuses[0])
    test.assertGreater(int(captured_statuses[0][0]["iterations"]), 1)
    test.assertGreater(int(captured_statuses[0][0]["backtracks"]), 0)


_DEVICES = get_test_devices(mode="basic")

add_function_test(
    TestDVIContactAPGD,
    "test_associated_analytic_contact_modes",
    test_associated_analytic_contact_modes,
    devices=_DEVICES,
)
add_function_test(
    TestDVIContactAPGD,
    "test_dense_operator_offsets_rhs_and_compliance",
    test_dense_operator_offsets_rhs_and_compliance,
    devices=_DEVICES,
)
add_function_test(
    TestDVIContactAPGD,
    "test_coupled_qp_and_represented_preconditioning",
    test_coupled_qp_and_represented_preconditioning,
    devices=_DEVICES,
)
add_function_test(
    TestDVIContactAPGD,
    "test_warmstart_reprojects_opening_and_changed_friction",
    test_warmstart_reprojects_opening_and_changed_friction,
    devices=_DEVICES,
)
add_function_test(
    TestDVIContactAPGD,
    "test_backtracking_caps_and_actual_counters",
    test_backtracking_caps_and_actual_counters,
    devices=_DEVICES,
)
add_function_test(
    TestDVIContactAPGD,
    "test_eager_conditionals_skip_predicated_work",
    test_eager_conditionals_skip_predicated_work,
    devices=_DEVICES,
)
add_function_test(
    TestDVIContactAPGD,
    "test_fixed_order_float64_reduction",
    test_fixed_order_float64_reduction,
    devices=_DEVICES,
)
add_function_test(
    TestDVIContactAPGD,
    "test_res4_step_avoids_large_contact_count_overflow",
    test_res4_step_avoids_large_contact_count_overflow,
    devices=_DEVICES,
)
add_function_test(
    TestDVIContactAPGD,
    "test_deterministic_reuse_and_best_res4",
    test_deterministic_reuse_and_best_res4,
    devices=_DEVICES,
)
add_function_test(
    TestDVIContactAPGD,
    "test_nested_graph_conditionals_match_predicated_fallback",
    test_nested_graph_conditionals_match_predicated_fallback,
    devices=_DEVICES,
)


if __name__ == "__main__":
    unittest.main()
