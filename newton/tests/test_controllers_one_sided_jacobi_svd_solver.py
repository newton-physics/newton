# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Tests for ``_svd_one_sided_jacobi``/``_svd_one_sided_jacobi_kernel``."""

from __future__ import annotations

import unittest

import numpy as np
import warp as wp

from newton._src.controllers.impl.differential_ik._common import _svd_one_sided_jacobi_kernel
from newton.tests.unittest_utils import add_function_test, get_test_devices

devices = get_test_devices()


def _run_svd(a_np, n_columns_vals, device, max_sweeps=30, tol=1.0e-8):
    """Launch _svd_one_sided_jacobi_kernel on a batch of (m, n) matrices, returning numpy U, S, V."""
    batch, m, n = a_np.shape
    matrix = wp.array3d(a_np, dtype=wp.float32, device=device).view(wp.types.matrix(shape=(m, n), dtype=wp.float32))
    n_columns = wp.array(n_columns_vals, dtype=wp.int32, device=device)
    u = wp.zeros((batch, m, m), dtype=wp.float32, device=device).view(wp.types.matrix(shape=(m, m), dtype=wp.float32))
    s = wp.zeros((batch, n), dtype=wp.float32, device=device).view(wp.types.vector(length=n, dtype=wp.float32))
    v = wp.zeros((batch, n, n), dtype=wp.float32, device=device).view(wp.types.matrix(shape=(n, n), dtype=wp.float32))
    wp.launch(
        _svd_one_sided_jacobi_kernel,
        dim=batch,
        inputs=[matrix, n_columns, tol, max_sweeps],
        outputs=[u, s, v],
        device=device,
    )
    return u.view(wp.float32).numpy(), s.view(wp.float32).numpy(), v.view(wp.float32).numpy()


def test_svd_one_sided_jacobi_matches_numpy_for_diagonal_matrix(test: unittest.TestCase, device):
    """Verify a diagonal matrix's own entries come back as its singular values, exactly, with U = V = I."""
    a_np = np.diag([3.0, 1.0]).astype(np.float32)
    u, s, v = _run_svd(a_np[None], [2], device)
    np.testing.assert_allclose(s[0], [3.0, 1.0], atol=1e-5)
    np.testing.assert_allclose(u[0], np.eye(2), atol=1e-5)
    np.testing.assert_allclose(v[0], np.eye(2), atol=1e-5)


def test_svd_one_sided_jacobi_reconstructs_matrix_built_from_known_factors(test: unittest.TestCase, device):
    """Verify a matrix built from known orthogonal U/V and singular values S is recovered: U @ diag(S) @ Vᵀ == A.

    U and V are only defined up to a simultaneous sign flip per column, so
    this checks reconstruction and orthonormality rather than comparing the
    returned U/V directly against the factors A was built from.
    """
    rng = np.random.default_rng(7)
    u_true, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    v_true, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    s_true = np.array([5.0, 2.0, 0.3])
    a_np = ((u_true * s_true) @ v_true.T).astype(np.float32)

    u, s, v = _run_svd(a_np[None], [3], device)
    u_np, s_np, v_np = u[0], s[0], v[0]
    np.testing.assert_allclose(s_np, s_true, atol=1e-4)
    np.testing.assert_allclose(u_np @ np.diag(s_np) @ v_np.T, a_np, atol=1e-4)
    np.testing.assert_allclose(u_np.T @ u_np, np.eye(3), atol=1e-5)
    np.testing.assert_allclose(v_np.T @ v_np, np.eye(3), atol=1e-5)


def test_svd_one_sided_jacobi_ignores_zero_padding_beyond_n_columns(test: unittest.TestCase, device):
    """Verify a 2x2 problem zero-padded into a 3x3 buffer, with n_columns=2, matches the same problem solved directly."""
    a2x2 = np.array([[2.0, 1.0], [0.0, 1.0]], dtype=np.float32)
    padded = np.zeros((3, 3), dtype=np.float32)
    padded[:2, :2] = a2x2

    u, s, v = _run_svd(padded[None], [2], device)
    u_np, s_np, v_np = u[0], s[0], v[0]
    expected_s = np.linalg.svd(a2x2, compute_uv=False)
    np.testing.assert_allclose(s_np[:2], expected_s, atol=1e-4)
    np.testing.assert_allclose(u_np[:2, :2] @ np.diag(s_np[:2]) @ v_np[:2, :2].T, a2x2, atol=1e-4)


def test_svd_one_sided_jacobi_matches_numpy_for_non_square_matrix(test: unittest.TestCase, device):
    """Verify a genuinely non-square (3 rows, 2 columns) matrix is decomposed correctly.

    Regression test: an earlier version sized U from A's column count
    instead of its row count, which is only correct by coincidence for a
    square matrix and would fail to even compile for m != n.
    """
    rng = np.random.default_rng(3)
    a_np = rng.normal(size=(3, 2)).astype(np.float32)

    u, s, v = _run_svd(a_np[None], [2], device)
    u_np, s_np, v_np = u[0], s[0], v[0]
    np.testing.assert_allclose(s_np, np.linalg.svd(a_np, compute_uv=False), atol=1e-4)
    np.testing.assert_allclose(u_np[:, :2] @ np.diag(s_np) @ v_np.T, a_np, atol=1e-4)
    # Only the first n_columns=2 columns of U (3x3) are computed; those must be orthonormal.
    np.testing.assert_allclose(u_np[:, :2].T @ u_np[:, :2], np.eye(2), atol=1e-5)
    np.testing.assert_allclose(v_np.T @ v_np, np.eye(2), atol=1e-5)


def test_svd_one_sided_jacobi_recovers_ill_conditioned_singular_values(test: unittest.TestCase, device):
    """Verify singular values spanning [1, 1e-4] are recovered to good relative accuracy, including the smallest.

    The whole point of this algorithm over an eigendecomposition of AᵀA
    (``_truncated_pinv_matrix_kernel``'s current approach): that approach
    squares the condition number before any float32 arithmetic happens, so
    a genuinely retained small singular value comes back off by orders of
    magnitude (confirmed separately: ~9000x relative error on this same
    scale of input). This algorithm never forms AᵀA, so it should recover
    even the smallest singular value here to a few parts in 1e4 -- nowhere
    close to that kind of blowup.
    """
    rng = np.random.default_rng(11)
    u_true, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    v_true, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    s_true = np.array([1.0, 1.0e-2, 1.0e-4])
    a_np = ((u_true * s_true) @ v_true.T).astype(np.float32)

    u, s, v = _run_svd(a_np[None], [3], device)
    u_np, s_np, v_np = u[0], s[0], v[0]
    # Relative, not absolute: an absolute tolerance loose enough for the
    # largest singular value would trivially "pass" by ignoring the
    # smallest one entirely, which is exactly the failure this guards against.
    np.testing.assert_allclose(s_np, s_true, rtol=1.0e-2)
    np.testing.assert_allclose(u_np @ np.diag(s_np) @ v_np.T, a_np, atol=1e-4)


def test_svd_one_sided_jacobi_handles_more_columns_than_rows(test: unittest.TestCase, device):
    """Verify a 3x5 matrix (rank <= 3) recovers its 3 real singular values and drops the 2 excess directions to exactly 0.

    Regression test: an earlier version only ever examined the first
    min(n_columns, m) *positions* of the converged columns when picking the
    largest singular values, silently assuming the true directions land in
    that prefix. Convergence only drives off-diagonal correlations to
    zero -- it does not control which of the n_columns columns end up
    holding the genuine signal -- so that assumption was wrong whenever
    n_columns > m, and this is exactly that case.
    """
    rng = np.random.default_rng(21)
    a_np = rng.normal(size=(3, 5)).astype(np.float32)

    u, s, v = _run_svd(a_np[None], [5], device)
    u_np, s_np, v_np = u[0], s[0], v[0]
    expected_s = np.linalg.svd(a_np, compute_uv=False)  # length 3 = min(m, n)
    np.testing.assert_allclose(s_np[:3], expected_s, atol=1e-4)
    np.testing.assert_allclose(s_np[3:], [0.0, 0.0], atol=1e-6)
    np.testing.assert_allclose(u_np[:, :3] @ np.diag(s_np[:3]) @ v_np[:, :3].T, a_np, atol=1e-4)
    np.testing.assert_allclose(u_np.T @ u_np, np.eye(3), atol=1e-5)


def test_svd_one_sided_jacobi_batch_has_no_cross_talk_with_heterogeneous_n_columns(test: unittest.TestCase, device):
    """Verify two matrices in one batched launch, with different n_columns, match solving each independently.

    Robot 1's third column is zero-padding (n_columns=2), robot 0's is a
    genuine 3x3 problem (n_columns=3): mixing different active sizes in one
    launch must not let one batch element's data leak into another's.
    """
    rng = np.random.default_rng(31)
    a0 = rng.normal(size=(3, 3)).astype(np.float32)
    a1 = rng.normal(size=(3, 3)).astype(np.float32)
    a1[:, 2] = 0.0

    u_batch, s_batch, v_batch = _run_svd(np.stack([a0, a1]), [3, 2], device)
    u0, s0, v0 = _run_svd(a0[None], [3], device)
    u1, s1, v1 = _run_svd(a1[None], [2], device)

    np.testing.assert_allclose(s_batch[0], s0[0], atol=1e-5)
    np.testing.assert_allclose(u_batch[0], u0[0], atol=1e-5)
    np.testing.assert_allclose(v_batch[0], v0[0], atol=1e-5)
    np.testing.assert_allclose(s_batch[1], s1[0], atol=1e-5)
    np.testing.assert_allclose(u_batch[1], u1[0], atol=1e-5)
    np.testing.assert_allclose(v_batch[1], v1[0], atol=1e-5)


def test_svd_one_sided_jacobi_matches_numpy_for_medium_sized_matrix(test: unittest.TestCase, device):
    """Verify a 10x10 random matrix -- well beyond the 2x2/3x3 cases above -- still matches numpy's SVD closely."""
    rng = np.random.default_rng(32)
    a_np = rng.normal(size=(10, 10)).astype(np.float32)

    u, s, v = _run_svd(a_np[None], [10], device, max_sweeps=30)
    u_np, s_np, v_np = u[0], s[0], v[0]
    expected_s = np.linalg.svd(a_np, compute_uv=False)
    np.testing.assert_allclose(s_np, expected_s, rtol=1.0e-4)
    np.testing.assert_allclose(u_np @ np.diag(s_np) @ v_np.T, a_np, atol=1e-4)


def test_svd_one_sided_jacobi_default_sweep_budget_is_sufficient_for_medium_size(test: unittest.TestCase, device):
    """Verify 30 sweeps (the value used everywhere above) converges a 10x10 matrix; too few sweeps measurably does not.

    Reconstruction (U @ diag(S) @ Vᵀ == A) holds at *any* sweep count --
    every Jacobi rotation is an exact identity, applied consistently to
    both sides, so it stays algebraically exact whether or not the middle
    factor has actually converged to diagonal yet. So reconstruction error
    alone cannot show convergence; comparing S against numpy's true
    singular values can, and does here: 1-2 sweeps are off by multiples of
    the true values, 5+ sweeps already match to float32 precision, and 30
    (used as the default everywhere in this file) has comfortable margin.
    """
    rng = np.random.default_rng(32)
    a_np = rng.normal(size=(10, 10)).astype(np.float32)
    expected_s = np.linalg.svd(a_np, compute_uv=False)

    _, s_too_few, _ = _run_svd(a_np[None], [10], device, max_sweeps=2)
    test.assertGreater(np.max(np.abs(s_too_few[0] - expected_s) / expected_s), 1.0)

    _, s_enough, _ = _run_svd(a_np[None], [10], device, max_sweeps=30)
    np.testing.assert_allclose(s_enough[0], expected_s, rtol=1.0e-4)


def test_svd_one_sided_jacobi_handles_repeated_singular_values(test: unittest.TestCase, device):
    """Verify two equal singular values (a known-tricky case for the whole Jacobi family) still reconstruct correctly.

    U/V individually are not uniquely determined within the degenerate
    (equal-singular-value) subspace -- only their span is -- so this checks
    reconstruction and orthonormality rather than U/V matching a specific
    reference.
    """
    rng = np.random.default_rng(22)
    u_true, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    v_true, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    s_true = np.array([2.0, 2.0, 0.5])
    a_np = ((u_true * s_true) @ v_true.T).astype(np.float32)

    u, s, v = _run_svd(a_np[None], [3], device)
    u_np, s_np, v_np = u[0], s[0], v[0]
    np.testing.assert_allclose(np.sort(s_np)[::-1], s_true, atol=1e-4)
    np.testing.assert_allclose(u_np @ np.diag(s_np) @ v_np.T, a_np, atol=1e-4)
    np.testing.assert_allclose(u_np.T @ u_np, np.eye(3), atol=1e-5)
    np.testing.assert_allclose(v_np.T @ v_np, np.eye(3), atol=1e-5)


def test_svd_one_sided_jacobi_degrades_gracefully_near_float32_precision_floor(test: unittest.TestCase, device):
    """Verify singular values spanning [1, 1e-6] -- near float32's ~1e-7 relative precision -- stay finite and reasonable.

    Not a claim of high accuracy here (this is right at what float32 can
    represent at all), just that behavior degrades gracefully rather than
    catastrophically the way the JJᵀ-eigendecomposition approach does
    (which is already off by orders of magnitude at 1e-4, four orders of
    magnitude less extreme than this).
    """
    rng = np.random.default_rng(23)
    u_true, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    v_true, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    s_true = np.array([1.0, 1.0e-3, 1.0e-6])
    a_np = ((u_true * s_true) @ v_true.T).astype(np.float32)

    u, s, v = _run_svd(a_np[None], [3], device)
    u_np, s_np, v_np = u[0], s[0], v[0]
    test.assertTrue(np.all(np.isfinite(s_np)))
    np.testing.assert_allclose(s_np, s_true, rtol=0.1)
    np.testing.assert_allclose(u_np @ np.diag(s_np) @ v_np.T, a_np, atol=1e-4)


class TestOneSidedJacobiSvdSolver(unittest.TestCase):
    pass


add_function_test(
    TestOneSidedJacobiSvdSolver,
    "test_svd_one_sided_jacobi_matches_numpy_for_diagonal_matrix",
    test_svd_one_sided_jacobi_matches_numpy_for_diagonal_matrix,
    devices=devices,
)
add_function_test(
    TestOneSidedJacobiSvdSolver,
    "test_svd_one_sided_jacobi_reconstructs_matrix_built_from_known_factors",
    test_svd_one_sided_jacobi_reconstructs_matrix_built_from_known_factors,
    devices=devices,
)
add_function_test(
    TestOneSidedJacobiSvdSolver,
    "test_svd_one_sided_jacobi_ignores_zero_padding_beyond_n_columns",
    test_svd_one_sided_jacobi_ignores_zero_padding_beyond_n_columns,
    devices=devices,
)
add_function_test(
    TestOneSidedJacobiSvdSolver,
    "test_svd_one_sided_jacobi_matches_numpy_for_non_square_matrix",
    test_svd_one_sided_jacobi_matches_numpy_for_non_square_matrix,
    devices=devices,
)
add_function_test(
    TestOneSidedJacobiSvdSolver,
    "test_svd_one_sided_jacobi_recovers_ill_conditioned_singular_values",
    test_svd_one_sided_jacobi_recovers_ill_conditioned_singular_values,
    devices=devices,
)
add_function_test(
    TestOneSidedJacobiSvdSolver,
    "test_svd_one_sided_jacobi_handles_more_columns_than_rows",
    test_svd_one_sided_jacobi_handles_more_columns_than_rows,
    devices=devices,
)
add_function_test(
    TestOneSidedJacobiSvdSolver,
    "test_svd_one_sided_jacobi_batch_has_no_cross_talk_with_heterogeneous_n_columns",
    test_svd_one_sided_jacobi_batch_has_no_cross_talk_with_heterogeneous_n_columns,
    devices=devices,
)
add_function_test(
    TestOneSidedJacobiSvdSolver,
    "test_svd_one_sided_jacobi_matches_numpy_for_medium_sized_matrix",
    test_svd_one_sided_jacobi_matches_numpy_for_medium_sized_matrix,
    devices=devices,
)
add_function_test(
    TestOneSidedJacobiSvdSolver,
    "test_svd_one_sided_jacobi_default_sweep_budget_is_sufficient_for_medium_size",
    test_svd_one_sided_jacobi_default_sweep_budget_is_sufficient_for_medium_size,
    devices=devices,
)
add_function_test(
    TestOneSidedJacobiSvdSolver,
    "test_svd_one_sided_jacobi_handles_repeated_singular_values",
    test_svd_one_sided_jacobi_handles_repeated_singular_values,
    devices=devices,
)
add_function_test(
    TestOneSidedJacobiSvdSolver,
    "test_svd_one_sided_jacobi_degrades_gracefully_near_float32_precision_floor",
    test_svd_one_sided_jacobi_degrades_gracefully_near_float32_precision_floor,
    devices=devices,
)


if __name__ == "__main__":
    unittest.main()
