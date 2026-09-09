# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Focused tests for the deterministic sparse Kamino contact APGD operator."""

from __future__ import annotations

import unittest

import numpy as np
import warp as wp

from newton._src.solvers.kamino._src.core.types import vec6f
from newton._src.solvers.kamino._src.solvers.dvi.apgd import ContactAPGDOptions, ContactAPGDSolver
from newton._src.solvers.kamino._src.solvers.dvi.apgd_sparse import SparseContactOperator
from newton.tests.unittest_utils import add_function_test, get_test_devices


class TestDVIContactAPGDSparse(unittest.TestCase):
    """Validate the matrix-free sparse contact operator independently."""

    pass


def _project_coulomb(vector: np.ndarray, friction: float) -> np.ndarray:
    """Project one ``[t0, t1, n]`` vector onto an associated cone."""
    tangent_norm = float(np.linalg.norm(vector[:2]))
    normal = float(vector[2])
    if friction * tangent_norm <= -normal:
        return np.zeros(3, dtype=np.float64)
    if tangent_norm <= friction * normal:
        return vector.astype(np.float64, copy=True)
    projected_normal = (friction * tangent_norm + normal) / (friction * friction + 1.0)
    projected_tangent = friction * projected_normal * vector[:2] / tangent_norm
    return np.array([*projected_tangent, projected_normal], dtype=np.float64)


def _solve_numpy_contact_qp(matrix: np.ndarray, rhs: np.ndarray, friction: np.ndarray) -> np.ndarray:
    """Solve a small product-cone QP for an independent numerical oracle."""
    solution = np.zeros_like(rhs, dtype=np.float64)
    step = 1.0 / float(np.linalg.eigvalsh(matrix).max())
    for _ in range(100_000):
        candidate = solution - step * (matrix @ solution - rhs)
        solution_new = np.concatenate(
            [_project_coulomb(candidate[3 * cid : 3 * cid + 3], float(mu)) for cid, mu in enumerate(friction)]
        )
        if np.linalg.norm(solution_new - solution) <= 1.0e-13:
            return solution_new
        solution = solution_new
    raise AssertionError("NumPy contact QP oracle failed to converge.")


def _make_fixture(device: wp.DeviceLike, config: ContactAPGDOptions | None = None) -> dict[str, object]:
    """Build two heterogeneous represented dense and raw-sparse systems."""
    capacities = [3, 1]
    dimensions = np.array([8, 4], dtype=np.int32)
    max_dimensions = [11, 4]
    vector_offsets = np.array([0, 11], dtype=np.int32)
    matrix_offsets = np.array([0, 121], dtype=np.int32)
    contact_counts = np.array([2, 1], dtype=np.int32)
    contact_offsets = np.array([0, 3], dtype=np.int32)
    contact_group_offsets = np.array([2, 1], dtype=np.int32)
    body_offsets = np.array([0, 2, 3], dtype=np.int32)
    row_starts = vector_offsets.copy()
    column_starts = np.array([0, 12], dtype=np.int32)

    body_inv_mass = np.array([0.7, 1.3, 0.9], dtype=np.float32)
    body_inv_inertia = np.array(
        [
            np.diag([1.1, 0.8, 0.6]),
            np.diag([0.5, 1.4, 0.9]),
            np.diag([1.2, 0.7, 1.6]),
        ],
        dtype=np.float32,
    )

    # World zero stores contact cid=1 before cid=0 to exercise the dynamic
    # contact maps. Contact block layout remains [B rows, optional A rows].
    coords = np.array(
        [
            [0, 0],
            [0, 6],
            [1, 6],
            [5, 0],
            [6, 0],
            [7, 0],
            [2, 6],
            [3, 6],
            [4, 6],
            [2, 0],
            [3, 0],
            [4, 0],
            [0, 0],
            [1, 0],
            [2, 0],
            [3, 0],
        ],
        dtype=np.int32,
    )
    blocks = np.array(
        [
            [0.7, -0.2, 0.1, 0.3, 0.4, -0.1],
            [-0.4, 0.6, 0.2, -0.2, 0.1, 0.5],
            [0.2, 0.1, -0.8, 0.6, -0.3, 0.4],
            [0.9, 0.2, -0.1, 0.1, -0.4, 0.3],
            [-0.2, 0.8, 0.3, 0.5, 0.2, -0.3],
            [0.1, -0.3, 1.0, -0.2, 0.4, 0.1],
            [-0.7, 0.1, 0.2, 0.3, -0.5, 0.4],
            [0.2, -0.9, 0.1, -0.4, 0.2, 0.3],
            [0.1, 0.3, 0.8, 0.2, 0.1, -0.6],
            [0.6, -0.2, -0.1, -0.3, 0.4, -0.2],
            [-0.1, 0.7, -0.2, 0.5, -0.1, 0.4],
            [-0.2, -0.1, -0.9, 0.1, -0.3, 0.5],
            [0.4, -0.5, 0.2, 0.3, 0.1, -0.2],
            [0.8, 0.1, -0.2, -0.1, 0.5, 0.3],
            [0.2, -0.7, 0.1, 0.4, -0.3, 0.2],
            [-0.1, 0.2, 0.9, 0.2, 0.1, -0.4],
        ],
        dtype=np.float32,
    )
    num_blocks = np.array([12, 4], dtype=np.int32)
    block_starts = np.array([0, 12], dtype=np.int32)
    contact_indices = np.array([1, 0, -1, 2], dtype=np.int32)
    contact_block_offsets = np.array([3, 6, 13, -1], dtype=np.int32)

    preconditioners = [
        np.array([1.1, 0.8, 0.7, 0.7, 0.7, 1.3, 1.3, 1.3], dtype=np.float32),
        np.array([0.9, 1.2, 1.2, 1.2], dtype=np.float32),
    ]
    regularizations = [
        np.array([0.05, 0.03, 0.02, 0.02, 0.02, 0.01, 0.01, 0.01], dtype=np.float32),
        np.array([0.06, 0.04, 0.04, 0.04], dtype=np.float32),
    ]
    compliances = [
        np.array([0.0, 0.0, 0.06, 0.06, 0.06, 0.03, 0.03, 0.03], dtype=np.float32),
        np.array([0.0, 0.05, 0.05, 0.05], dtype=np.float32),
    ]
    free_velocities = [
        np.array([0.2, -0.1, -0.15, 0.08, -1.1, 0.05, -0.12, -0.8], dtype=np.float32),
        np.array([-0.2, 0.1, -0.05, -0.7], dtype=np.float32),
    ]
    solutions = [
        np.array([0.3, -0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32),
        np.array([-0.25, 0.0, 0.0, 0.0], dtype=np.float32),
    ]

    matrices: list[np.ndarray] = []
    for wid, (dimension, body_count) in enumerate(zip(dimensions, [2, 1], strict=True)):
        jacobian = np.zeros((dimension, 6 * body_count), dtype=np.float64)
        start = int(block_starts[wid])
        for local_block in range(int(num_blocks[wid])):
            row, column = coords[start + local_block]
            jacobian[row, column : column + 6] = blocks[start + local_block]
        inverse_mass = np.zeros((6 * body_count, 6 * body_count), dtype=np.float64)
        body_begin = int(body_offsets[wid])
        for local_body in range(body_count):
            global_body = body_begin + local_body
            body_dof = 6 * local_body
            inverse_mass[body_dof : body_dof + 3, body_dof : body_dof + 3] = body_inv_mass[global_body] * np.eye(3)
            inverse_mass[body_dof + 3 : body_dof + 6, body_dof + 3 : body_dof + 6] = body_inv_inertia[global_body]
        scaling = np.diag(preconditioners[wid])
        represented = scaling @ jacobian @ inverse_mass @ jacobian.T @ scaling
        represented += np.diag(regularizations[wid])
        matrices.append(represented)

    def to_int(values: np.ndarray | list[int]) -> wp.array[wp.int32]:
        """Copy an integer fixture array to the selected device."""
        return wp.array(np.asarray(values, dtype=np.int32), dtype=wp.int32, device=device)

    def to_float(values: np.ndarray | list[float]) -> wp.array[wp.float32]:
        """Copy a scalar fixture array to the selected device."""
        return wp.array(np.asarray(values, dtype=np.float32), dtype=wp.float32, device=device)

    problem_dim = to_int(dimensions)
    problem_vio = to_int(vector_offsets)
    problem_nc = to_int(contact_counts)
    problem_cio = to_int(contact_offsets)
    problem_ccgo = to_int(contact_group_offsets)
    problem_mio = to_int(matrix_offsets)
    full_preconditioner = np.zeros(sum(max_dimensions), dtype=np.float32)
    full_regularization = np.zeros_like(full_preconditioner)
    full_compliance = np.zeros_like(full_preconditioner)
    full_free_velocity = np.zeros_like(full_preconditioner)
    full_solution = np.zeros_like(full_preconditioner)
    for wid, offset in enumerate(vector_offsets):
        dimension = dimensions[wid]
        target = slice(offset, offset + dimension)
        full_preconditioner[target] = preconditioners[wid]
        full_regularization[target] = regularizations[wid]
        full_compliance[target] = compliances[wid]
        full_free_velocity[target] = free_velocities[wid]
        full_solution[target] = solutions[wid]

    flat_matrix = np.zeros(121 + 16, dtype=np.float32)
    flat_matrix[: matrices[0].size] = matrices[0].astype(np.float32).ravel()
    flat_matrix[121 : 121 + matrices[1].size] = matrices[1].astype(np.float32).ravel()

    solver = ContactAPGDSolver(
        capacities,
        config or ContactAPGDOptions(max_iterations=250, tolerance=1.0e-5, use_graph_conditionals=False),
        device=device,
    )
    dense = solver.make_dense_operator(
        problem_dim=problem_dim,
        problem_mio=problem_mio,
        problem_vio=problem_vio,
        problem_nc=problem_nc,
        problem_cio=problem_cio,
        problem_ccgo=problem_ccgo,
        matrix=to_float(flat_matrix),
        represented_compliance=to_float(full_compliance),
        free_velocity=to_float(full_free_velocity),
    )
    sparse = SparseContactOperator(
        problem_dim=problem_dim,
        problem_vio=problem_vio,
        problem_nc=problem_nc,
        problem_cio=problem_cio,
        problem_ccgo=problem_ccgo,
        contact_indices=to_int(contact_indices),
        contact_nzb_offsets=to_int(contact_block_offsets),
        jacobian_num_nzb=to_int(num_blocks),
        jacobian_nzb_start=to_int(block_starts),
        jacobian_nzb_coords=wp.array(coords, dtype=wp.int32, device=device),
        jacobian_nzb_values=wp.array(blocks, dtype=vec6f, device=device),
        jacobian_row_start=to_int(row_starts),
        jacobian_col_start=to_int(column_starts),
        body_offset=to_int(body_offsets),
        body_inv_mass=to_float(body_inv_mass),
        body_inv_inertia=wp.array(body_inv_inertia, dtype=wp.mat33f, device=device),
        preconditioner=to_float(full_preconditioner),
        regularization=to_float(full_regularization),
        represented_compliance=dense.represented_compliance,
        free_velocity=dense.free_velocity,
        contact_row_offset=solver.contact_row_offset,
        num_worlds=2,
        max_contacts_per_world=3,
        max_bodies_per_world=2,
        total_body_dofs=18,
        device=device,
    )
    return {
        "solver": solver,
        "dense": dense,
        "sparse": sparse,
        "matrices": matrices,
        "compliances": compliances,
        "free_velocities": free_velocities,
        "solutions": solutions,
        "full_solution": to_float(full_solution),
        "friction": to_float([0.45, 0.7, np.nan, 0.55]),
        "contact_slices": (slice(0, 6), slice(9, 12)),
        "full_contact_slices": (slice(2, 8), slice(12, 15)),
    }


def test_sparse_operator_matches_dense_and_numpy(test: unittest.TestCase, device: wp.DeviceLike) -> None:
    """Match dense and NumPy products, RHS vectors, and compact mappings."""
    fixture = _make_fixture(device)
    dense = fixture["dense"]
    sparse = fixture["sparse"]
    full_solution = fixture["full_solution"]
    mask = wp.ones(2, dtype=wp.bool, device=device)
    compact_x_np = np.array(
        [0.3, -0.4, 1.2, -0.2, 0.5, 0.9, np.nan, np.nan, np.nan, 0.1, -0.3, 0.8],
        dtype=np.float32,
    )
    compact_x = wp.array(compact_x_np, dtype=wp.float32, device=device)
    dense_product = wp.full(12, -123.0, dtype=wp.float32, device=device)
    sparse_product = wp.full(12, -123.0, dtype=wp.float32, device=device)
    dense_rhs = wp.full(12, -456.0, dtype=wp.float32, device=device)
    sparse_rhs = wp.full(12, -456.0, dtype=wp.float32, device=device)
    dense_gather = wp.full(12, np.nan, dtype=wp.float32, device=device)
    sparse_gather = wp.full(12, np.nan, dtype=wp.float32, device=device)

    sparse.prepare()
    dense.matvec(compact_x, dense_product, mask)
    sparse.matvec(compact_x, sparse_product, mask)
    dense.build_rhs(full_solution, dense_rhs, mask)
    sparse.build_rhs(full_solution, sparse_rhs, mask)
    dense.gather(full_solution, dense_gather, mask)
    sparse.gather(full_solution, sparse_gather, mask)

    dense_product_np = dense_product.numpy()
    sparse_product_np = sparse_product.numpy()
    dense_rhs_np = dense_rhs.numpy()
    sparse_rhs_np = sparse_rhs.numpy()
    np.testing.assert_allclose(sparse_product_np, dense_product_np, rtol=2.0e-6, atol=2.0e-6, equal_nan=True)
    np.testing.assert_allclose(sparse_rhs_np, dense_rhs_np, rtol=2.0e-6, atol=2.0e-6, equal_nan=True)
    np.testing.assert_array_equal(sparse_gather.numpy(), dense_gather.numpy())
    np.testing.assert_array_equal(sparse.body_contact_offsets.numpy(), np.array([0, 2, 3, 6, 7, 0]))
    np.testing.assert_array_equal(sparse.body_block_offsets.numpy(), np.array([0, 1, 3, 12, 13, 0]))
    np.testing.assert_array_equal(
        sparse.body_contact_slots.numpy()[[0, 1, 2, 6]],
        np.array([[0, 9], [1, 3], [0, 6], [0, 13]], dtype=np.int32),
    )
    block_slots = sparse.body_block_slots.numpy()
    block_rows = sparse.jacobian_nzb_coords.numpy()[:, 0]
    for begin, end in ((0, 1), (1, 3), (12, 13)):
        test.assertTrue(np.all(np.diff(block_rows[block_slots[begin:end]]) >= 0))

    for wid, (compact_slice, full_contact_slice) in enumerate(
        zip(fixture["contact_slices"], fixture["full_contact_slices"], strict=True)
    ):
        matrix = fixture["matrices"][wid]
        ccgo = [2, 1][wid]
        contact_count = [2, 1][wid]
        contact = slice(ccgo, ccgo + 3 * contact_count)
        x_world = compact_x_np[compact_slice]
        expected_product = matrix[contact, contact] @ x_world + fixture["compliances"][wid][contact] * x_world
        lambda_world = fixture["solutions"][wid].copy()
        lambda_world[contact] = 0.0
        expected_rhs = -(fixture["free_velocities"][wid][contact] + matrix[contact] @ lambda_world)
        np.testing.assert_allclose(sparse_product_np[compact_slice], expected_product, rtol=2.0e-6, atol=2.0e-6)
        np.testing.assert_allclose(sparse_rhs_np[compact_slice], expected_rhs, rtol=2.0e-6, atol=2.0e-6)
        np.testing.assert_array_equal(sparse_gather.numpy()[compact_slice], full_solution.numpy()[full_contact_slice])

    compact_scatter = wp.array(compact_x_np, dtype=wp.float32, device=device)
    dense_full = wp.full(15, 19.0, dtype=wp.float32, device=device)
    sparse_full = wp.full(15, 19.0, dtype=wp.float32, device=device)
    dense.scatter(compact_scatter, dense_full, mask)
    sparse.scatter(compact_scatter, sparse_full, mask)
    np.testing.assert_array_equal(sparse_full.numpy(), dense_full.numpy())

    repeated = []
    for _ in range(4):
        sparse.matvec(compact_x, sparse_product, mask)
        repeated.append(sparse_product.numpy().copy())
    for value in repeated[1:]:
        np.testing.assert_array_equal(value, repeated[0])
    test.assertTrue(np.all(sparse_product_np[6:9] == -123.0))


def test_sparse_apgd_matches_dense_and_numpy(test: unittest.TestCase, device: wp.DeviceLike) -> None:
    """Match dense APGD and an independent product-cone QP oracle."""
    config = ContactAPGDOptions(max_iterations=500, tolerance=2.0e-5, use_graph_conditionals=False)
    dense_fixture = _make_fixture(device, config)
    sparse_fixture = _make_fixture(device, config)
    dense_solver = dense_fixture["solver"]
    sparse_solver = sparse_fixture["solver"]
    dense_full = dense_fixture["full_solution"]
    sparse_full = sparse_fixture["full_solution"]
    dense_solver.solve_dense(dense_fixture["dense"], dense_fixture["friction"], dense_full)

    sparse_operator = sparse_fixture["sparse"]
    mask = wp.ones(2, dtype=wp.bool, device=device)
    sparse_operator.prepare()
    sparse_operator.build_rhs(sparse_full, sparse_solver.rhs, mask)
    sparse_operator.gather(sparse_full, sparse_solver.solution, mask)
    sparse_solver.solve(
        sparse_operator.problem_nc,
        sparse_fixture["friction"],
        sparse_solver.rhs,
        sparse_solver.solution,
        sparse_operator.matvec,
        phase_mask=mask,
        contact_offset=sparse_operator.problem_cio,
    )
    sparse_operator.scatter(sparse_solver.solution, sparse_full, mask)

    dense_result = dense_solver.solution.numpy()
    sparse_result = sparse_solver.solution.numpy()
    for wid, compact_slice in enumerate(sparse_fixture["contact_slices"]):
        np.testing.assert_allclose(sparse_result[compact_slice], dense_result[compact_slice], rtol=3.0e-4, atol=3.0e-5)
        matrix = sparse_fixture["matrices"][wid]
        ccgo = [2, 1][wid]
        count = [2, 1][wid]
        contact = slice(ccgo, ccgo + 3 * count)
        full_lambda = sparse_fixture["solutions"][wid].copy()
        full_lambda[contact] = 0.0
        rhs = -(sparse_fixture["free_velocities"][wid][contact] + matrix[contact] @ full_lambda)
        contact_matrix = matrix[contact, contact] + np.diag(sparse_fixture["compliances"][wid][contact])
        friction = [np.array([0.45, 0.7]), np.array([0.55])][wid]
        expected = _solve_numpy_contact_qp(contact_matrix, rhs, friction)
        np.testing.assert_allclose(sparse_result[compact_slice], expected, rtol=4.0e-4, atol=5.0e-5)

    np.testing.assert_array_equal(sparse_solver.status.numpy()["converged"], np.ones(2, dtype=np.int32))
    test.assertTrue(np.all(np.isfinite(sparse_result[[0, 1, 2, 3, 4, 5, 9, 10, 11]])))


def test_sparse_adjacency_scales_with_nonzeros(test: unittest.TestCase, device: wp.DeviceLike) -> None:
    """Keep per-body APGD traversals proportional to incident sparse blocks."""
    body_count = 128
    contact_count = 64
    contact_group_offset = body_count
    problem_dimension = contact_group_offset + 3 * contact_count
    block_count = body_count + 6 * contact_count
    coords = np.zeros((block_count, 2), dtype=np.int32)
    coords[:body_count, 0] = np.arange(body_count)
    coords[:body_count, 1] = 6 * np.arange(body_count)
    for cid in range(contact_count):
        row = contact_group_offset + 3 * cid
        for side in range(2):
            block_offset = body_count + 6 * cid + 3 * side
            body = 2 * cid + side
            coords[block_offset : block_offset + 3, 0] = np.arange(row, row + 3)
            coords[block_offset : block_offset + 3, 1] = 6 * body

    def ints(values: np.ndarray | list[int]) -> wp.array[wp.int32]:
        """Copy an integer array to the test device."""
        return wp.array(np.asarray(values, dtype=np.int32), dtype=wp.int32, device=device)

    scalar_data = wp.zeros(problem_dimension, dtype=wp.float32, device=device)
    operator = SparseContactOperator(
        problem_dim=ints([problem_dimension]),
        problem_vio=ints([0]),
        problem_nc=ints([contact_count]),
        problem_cio=ints([0]),
        problem_ccgo=ints([contact_group_offset]),
        contact_indices=ints(np.arange(contact_count)),
        contact_nzb_offsets=ints(body_count + 6 * np.arange(contact_count)),
        jacobian_num_nzb=ints([block_count]),
        jacobian_nzb_start=ints([0]),
        jacobian_nzb_coords=wp.array(coords, dtype=wp.int32, device=device),
        jacobian_nzb_values=wp.zeros(block_count, dtype=vec6f, device=device),
        jacobian_row_start=ints([0]),
        jacobian_col_start=ints([0]),
        body_offset=ints([0, body_count]),
        body_inv_mass=wp.ones(body_count, dtype=wp.float32, device=device),
        body_inv_inertia=wp.array(
            np.repeat(np.eye(3, dtype=np.float32)[None, :, :], body_count, axis=0),
            dtype=wp.mat33f,
            device=device,
        ),
        preconditioner=wp.ones(problem_dimension, dtype=wp.float32, device=device),
        regularization=scalar_data,
        represented_compliance=scalar_data,
        free_velocity=scalar_data,
        contact_row_offset=ints([0, 3 * contact_count]),
        num_worlds=1,
        max_contacts_per_world=contact_count,
        max_bodies_per_world=body_count,
        total_body_dofs=6 * body_count,
        device=device,
    )
    operator.prepare()

    contact_offsets = operator.body_contact_offsets.numpy()[: body_count + 1]
    block_offsets = operator.body_block_offsets.numpy()[: body_count + 1]
    np.testing.assert_array_equal(np.diff(contact_offsets), np.ones(body_count, dtype=np.int32))
    np.testing.assert_array_equal(np.diff(block_offsets), np.ones(body_count, dtype=np.int32))
    test.assertEqual(int(contact_offsets[-1]), 2 * contact_count)
    test.assertEqual(int(block_offsets[-1]), body_count)
    test.assertLess(int(contact_offsets[-1]), body_count * contact_count)


def test_sparse_operator_cuda_graph_replay(test: unittest.TestCase, device: wp.DeviceLike) -> None:
    """Replay every sparse adapter operation in a CUDA graph bit-for-bit."""
    if not device.is_cuda:
        test.skipTest("CUDA graph replay requires a CUDA device.")

    fixture = _make_fixture(device)
    sparse = fixture["sparse"]
    full_solution = fixture["full_solution"]
    mask = wp.ones(2, dtype=wp.bool, device=device)
    compact_x = wp.array(
        [0.3, -0.4, 1.2, -0.2, 0.5, 0.9, 0.0, 0.0, 0.0, 0.1, -0.3, 0.8],
        dtype=wp.float32,
        device=device,
    )
    product = wp.zeros(12, dtype=wp.float32, device=device)
    rhs = wp.zeros(12, dtype=wp.float32, device=device)
    gathered = wp.zeros(12, dtype=wp.float32, device=device)
    scattered = wp.zeros(15, dtype=wp.float32, device=device)

    sparse.prepare()
    sparse.matvec(compact_x, product, mask)
    sparse.build_rhs(full_solution, rhs, mask)
    sparse.gather(full_solution, gathered, mask)
    sparse.scatter(gathered, scattered, mask)
    wp.synchronize_device(device)

    with wp.ScopedCapture(device) as capture:
        sparse.prepare()
        sparse.matvec(compact_x, product, mask)
        sparse.build_rhs(full_solution, rhs, mask)
        sparse.gather(full_solution, gathered, mask)
        sparse.scatter(gathered, scattered, mask)

    outputs = []
    for _ in range(3):
        wp.capture_launch(capture.graph)
        outputs.append((product.numpy().copy(), rhs.numpy().copy(), gathered.numpy().copy(), scattered.numpy().copy()))
    for replay in outputs[1:]:
        for value, reference in zip(replay, outputs[0], strict=True):
            np.testing.assert_array_equal(value, reference)
    test.assertTrue(np.all(np.isfinite(outputs[0][0])))


_DEVICES = get_test_devices(mode="basic")

add_function_test(
    TestDVIContactAPGDSparse,
    "test_sparse_operator_matches_dense_and_numpy",
    test_sparse_operator_matches_dense_and_numpy,
    devices=_DEVICES,
)
add_function_test(
    TestDVIContactAPGDSparse,
    "test_sparse_apgd_matches_dense_and_numpy",
    test_sparse_apgd_matches_dense_and_numpy,
    devices=_DEVICES,
)
add_function_test(
    TestDVIContactAPGDSparse,
    "test_sparse_adjacency_scales_with_nonzeros",
    test_sparse_adjacency_scales_with_nonzeros,
    devices=_DEVICES,
)
add_function_test(
    TestDVIContactAPGDSparse,
    "test_sparse_operator_cuda_graph_replay",
    test_sparse_operator_cuda_graph_replay,
    devices=_DEVICES,
)


if __name__ == "__main__":
    unittest.main()
