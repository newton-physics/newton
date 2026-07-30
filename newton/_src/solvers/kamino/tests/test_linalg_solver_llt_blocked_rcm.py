# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the LLTBlockedRCMSolver from linalg/linear.py"""

import unittest

import numpy as np
import warp as wp

from newton._src.solvers.kamino._src.linalg.core import DenseLinearOperatorData, DenseSquareMultiLinearInfo
from newton._src.solvers.kamino._src.linalg.factorize import rcm_batch
from newton._src.solvers.kamino._src.linalg.factorize.llt_blocked_rcm_solver import LLTBlockedRCMSolver
from newton._src.solvers.kamino._src.utils import logger as msg
from newton._src.solvers.kamino.tests import setup_tests, test_context
from newton._src.solvers.kamino.tests.utils.extract import get_matrix_block, get_vector_block
from newton._src.solvers.kamino.tests.utils.print import print_error_stats
from newton._src.solvers.kamino.tests.utils.rand import RandomProblemLLT

###
# Tests
###


class TestLinAlgLLTBlockedRCMSolver(unittest.TestCase):
    def setUp(self):
        # Configs
        if not test_context.setup_done:
            setup_tests(clear_cache=False)
        self.seed = 42
        self.default_device = wp.get_device(test_context.device)
        self.verbose = test_context.verbose  # Set to True for verbose output

        # Set debug-level logging to print verbose test output to console
        if self.verbose:
            print("\n")  # Add newline before test output for better readability
            msg.set_log_level(msg.LogLevel.DEBUG)
        else:
            msg.reset_log_level()

    def tearDown(self):
        self.default_device = None
        if self.verbose:
            msg.reset_log_level()

    def test_00_make_default_solver(self):
        """
        Test the default constructor of the LLTBlockedRCMSolver class.
        """
        llt = LLTBlockedRCMSolver(device=self.default_device)
        self.assertIsNone(llt._operator)
        self.assertEqual(llt.dtype, wp.float32)
        self.assertEqual(llt.device, self.default_device)
        self.assertTrue(llt._reuse_permutation)

    @staticmethod
    def _run_rcm(matrix: np.ndarray, device, use_cuda_graph: bool = False) -> np.ndarray:
        n = matrix.shape[0]
        matrix_wp = wp.array(matrix.reshape(-1), dtype=wp.float32, device=device)
        dims = wp.array([n], dtype=wp.int32, device=device)
        offsets = wp.array([0], dtype=wp.int32, device=device)
        permutation = wp.zeros(n, dtype=wp.int32, device=device)
        scratch = rcm_batch.allocate_rcm_batch_scratch(n, 1, device)
        reorder = rcm_batch.create_rcm_batch_launch(
            A_flat=matrix_wp,
            perm_flat=permutation,
            dims=dims,
            mio=offsets,
            vio=offsets,
            scratch=scratch,
            num_blocks=1,
            max_dim=n,
            use_cuda_graph=use_cuda_graph,
            device=device,
        )
        reorder()
        return permutation.numpy()

    @staticmethod
    def _path_bandwidth(permutation: np.ndarray, paths: tuple[np.ndarray, ...]) -> int:
        positions = np.empty(permutation.size, dtype=np.int64)
        positions[permutation] = np.arange(permutation.size)
        return max(int(np.max(np.abs(positions[path[:-1]] - positions[path[1:]]))) for path in paths)

    def test_complete_rcm_across_legacy_boundary(self):
        """Traverse long paths completely across the former 1024-row boundary."""
        if not self.default_device.is_cuda:
            self.skipTest("The legacy boundary applied only to the CUDA fast path")

        for n in (1024, 1025):
            with self.subTest(n=n):
                rng = np.random.default_rng(self.seed + n)
                path = rng.permutation(n)
                matrix = np.eye(n, dtype=np.float32) * 2.0
                matrix[path[:-1], path[1:]] = -0.25
                matrix[path[1:], path[:-1]] = -0.25

                permutation = self._run_rcm(matrix, self.default_device, use_cuda_graph=True)

                np.testing.assert_array_equal(np.sort(permutation), np.arange(n))
                self.assertEqual(self._path_bandwidth(permutation, (path,)), 1)

    def test_complete_rcm_on_disconnected_components(self):
        """Traverse every disconnected path component on each backend."""
        devices = [wp.get_device("cpu")]
        if self.default_device.is_cuda:
            devices.append(self.default_device)

        n = 96
        rng = np.random.default_rng(self.seed)
        labels = rng.permutation(n)
        paths = (labels[: n // 2], labels[n // 2 :])
        matrix = np.eye(n, dtype=np.float32) * 2.0
        for path in paths:
            matrix[path[:-1], path[1:]] = -0.25
            matrix[path[1:], path[:-1]] = -0.25

        for device in devices:
            with self.subTest(device=device):
                permutation = self._run_rcm(matrix, device, use_cuda_graph=device.is_cuda)

                np.testing.assert_array_equal(np.sort(permutation), np.arange(n))
                self.assertEqual(self._path_bandwidth(permutation, paths), 1)

    def test_cached_permutation_with_changed_sparsity(self):
        """Verify cached RCM remains correct when numeric sparsity changes."""
        n = 96
        rng = np.random.default_rng(self.seed)

        def make_banded_spd(width):
            matrix = np.zeros((n, n), dtype=np.float32)
            for offset in range(1, width + 1):
                values = rng.uniform(-0.2, 0.2, n - offset).astype(np.float32)
                rows = np.arange(n - offset)
                matrix[rows, rows + offset] = values
                matrix[rows + offset, rows] = values
            matrix[np.diag_indices(n)] = np.sum(np.abs(matrix), axis=1) + 1.0
            return matrix

        matrix_1 = make_banded_spd(2)
        matrix_2 = make_banded_spd(9)
        rhs_np = rng.standard_normal(n).astype(np.float32)

        info = DenseSquareMultiLinearInfo()
        info.finalize(dimensions=[n], dtype=wp.float32, device=self.default_device)
        matrix_wp = wp.array(matrix_1.reshape(-1), dtype=wp.float32, device=self.default_device)
        rhs_wp = wp.array(rhs_np, dtype=wp.float32, device=self.default_device)
        result_wp = wp.zeros(n, dtype=wp.float32, device=self.default_device)
        operator = DenseLinearOperatorData(info=info, mat=matrix_wp)
        solver = LLTBlockedRCMSolver(
            operator=operator,
            block_size=32,
            factorize_block_dim=256,
            reuse_permutation=True,
            parallel_factorization=True,
            device=self.default_device,
        )

        solver.compute(matrix_wp)
        permutation_1 = solver.P.numpy()
        matrix_wp.assign(matrix_2.reshape(-1))
        solver.compute(matrix_wp)
        solver.solve(rhs_wp, result_wp)

        np.testing.assert_array_equal(solver.P.numpy(), permutation_1)
        expected = np.linalg.solve(matrix_2, rhs_np)
        np.testing.assert_allclose(result_wp.numpy(), expected, rtol=1.0e-3, atol=1.0e-4)

    def test_parallel_factorization_with_partial_tile(self):
        """Factorize and solve a system whose final tile is partial."""
        n = 33
        matrix = np.eye(n, dtype=np.float32) * 2.0
        indices = np.arange(n - 1)
        matrix[indices, indices + 1] = -0.25
        matrix[indices + 1, indices] = -0.25
        rhs = np.ones(n, dtype=np.float32)

        info = DenseSquareMultiLinearInfo()
        info.finalize(dimensions=[n], dtype=wp.float32, device=self.default_device)
        matrix_wp = wp.array(matrix.reshape(-1), dtype=wp.float32, device=self.default_device)
        rhs_wp = wp.array(rhs, dtype=wp.float32, device=self.default_device)
        result_wp = wp.zeros(n, dtype=wp.float32, device=self.default_device)
        solver = LLTBlockedRCMSolver(
            operator=DenseLinearOperatorData(info=info, mat=matrix_wp),
            block_size=32,
            reuse_permutation=True,
            parallel_factorization=True,
            device=self.default_device,
        )

        solver.compute(matrix_wp)
        solver.solve(rhs_wp, result_wp)

        expected = np.linalg.solve(matrix, rhs)
        np.testing.assert_allclose(result_wp.numpy(), expected, rtol=1.0e-4, atol=1.0e-5)

    def test_cached_permutation_on_cpu_fallback(self):
        """Verify the CPU fallback reuses a cached permutation."""
        device = wp.get_device("cpu")
        n = 8

        def make_spd(edges):
            matrix = np.zeros((n, n), dtype=np.float32)
            for row, col in edges:
                matrix[row, col] = -0.2
                matrix[col, row] = -0.2
            matrix[np.diag_indices(n)] = np.sum(np.abs(matrix), axis=1) + 1.0
            return matrix

        path_matrix = make_spd([(i, i + 1) for i in range(n - 1)])
        star_matrix = make_spd([(0, i) for i in range(1, n)])
        matrix_wp = wp.array(path_matrix.reshape(-1), dtype=wp.float32, device=device)
        dims = wp.array([n], dtype=wp.int32, device=device)
        offsets = wp.array([0], dtype=wp.int32, device=device)
        permutation = wp.zeros(n, dtype=wp.int32, device=device)
        scratch = rcm_batch.allocate_rcm_batch_scratch(n, 1, device)
        reorder = rcm_batch.create_rcm_batch_launch(
            A_flat=matrix_wp,
            perm_flat=permutation,
            dims=dims,
            mio=offsets,
            vio=offsets,
            scratch=scratch,
            num_blocks=1,
            max_dim=n,
            use_cuda_graph=False,
            reuse_permutation=True,
            device=device,
        )

        reorder()
        cached = permutation.numpy()
        matrix_wp.assign(star_matrix.reshape(-1))
        reorder()
        np.testing.assert_array_equal(permutation.numpy(), cached)

        recomputed = wp.zeros_like(permutation)
        fresh_scratch = rcm_batch.allocate_rcm_batch_scratch(n, 1, device)
        recompute = rcm_batch.create_rcm_batch_launch(
            A_flat=matrix_wp,
            perm_flat=recomputed,
            dims=dims,
            mio=offsets,
            vio=offsets,
            scratch=fresh_scratch,
            num_blocks=1,
            max_dim=n,
            use_cuda_graph=False,
            device=device,
        )
        recompute()
        self.assertFalse(np.array_equal(recomputed.numpy(), cached))

    def test_solve_with_fewer_threads_than_tile_rows(self):
        """Verify the solve gathers every right-hand-side row."""
        n = 65
        rng = np.random.default_rng(self.seed)
        dense = rng.standard_normal((n, n)).astype(np.float32)
        matrix = dense @ dense.T + np.eye(n, dtype=np.float32)
        rhs = rng.standard_normal(n).astype(np.float32)

        info = DenseSquareMultiLinearInfo()
        info.finalize(dimensions=[n], dtype=wp.float32, device=self.default_device)
        matrix_wp = wp.array(matrix.reshape(-1), dtype=wp.float32, device=self.default_device)
        rhs_wp = wp.array(rhs, dtype=wp.float32, device=self.default_device)
        result_wp = wp.zeros(n, dtype=wp.float32, device=self.default_device)
        solver = LLTBlockedRCMSolver(
            operator=DenseLinearOperatorData(info=info, mat=matrix_wp),
            block_size=64,
            solve_block_dim=32,
            device=self.default_device,
        )

        solver.compute(matrix_wp)
        solver.solve(rhs_wp, result_wp)

        expected = np.linalg.solve(matrix, rhs)
        np.testing.assert_allclose(result_wp.numpy(), expected, rtol=1.0e-3, atol=1.0e-4)

    def test_01_single_problem_dims_all_active(self):
        """
        Test the blocked LLT solver on a single small problem.
        """
        # Constants
        # N = 12  # Use this for visual debugging with small matrices
        N = 2000  # Use this for performance testing with large matrices

        # Create a single-instance problem
        problem = RandomProblemLLT(
            dims=N,
            seed=self.seed,
            np_dtype=np.float32,
            wp_dtype=wp.float32,
            device=self.default_device,
        )

        # Optional verbose output
        msg.debug("Problem:\n%s\n", problem)
        msg.debug("b_np:\n%s\n", problem.b_np[0])
        msg.debug("A_np:\n%s\n", problem.A_np[0])
        msg.debug("X_np:\n%s\n", problem.X_np[0])
        msg.debug("y_np:\n%s\n", problem.y_np[0])
        msg.debug("x_np:\n%s\n", problem.x_np[0])
        msg.info("A_wp (%s):\n%s\n", problem.A_wp.shape, problem.A_wp.numpy().reshape((N, N)))
        msg.info("b_wp (%s):\n%s\n", problem.b_wp.shape, problem.b_wp.numpy().reshape((N,)))

        # Create the linear operator meta-data
        opinfo = DenseSquareMultiLinearInfo()
        opinfo.finalize(dimensions=problem.dims, dtype=problem.wp_dtype, device=self.default_device)
        msg.debug("opinfo:\n%s", opinfo)

        # Create the linear operator data structure
        operator = DenseLinearOperatorData(info=opinfo, mat=problem.A_wp)
        msg.debug("operator.info:\n%s\n", operator.info)
        msg.debug("operator.mat (%s):\n%s\n", operator.mat.shape, operator.mat.numpy().reshape((N, N)))

        # Create a SequentialCholeskyFactorizer instance
        llt = LLTBlockedRCMSolver(operator=operator, device=self.default_device)
        self.assertIsNotNone(llt._operator)
        self.assertEqual(llt.dtype, problem.wp_dtype)
        self.assertEqual(llt.device, self.default_device)
        self.assertIsNotNone(llt._L)
        self.assertIsNotNone(llt._y)
        self.assertEqual(llt.L.size, problem.A_wp.size)
        self.assertEqual(llt.y.size, problem.b_wp.size)

        ###
        # Matrix factorization
        ###

        # Factorize the target square-symmetric matrix
        llt.compute(A=problem.A_wp)
        msg.info("llt.L (%s):\n%s\n", llt.L.shape, llt.L.numpy().reshape((N, N)))

        # Extract L, P from warp to numpy
        P_wp_np = get_vector_block(0, llt.P.numpy(), problem.dims, problem.maxdims)
        msg.info("P_wp_np (%s):\n%s\n", P_wp_np.shape, P_wp_np)
        L_wp_np = get_matrix_block(0, llt.L.numpy(), problem.dims, problem.maxdims)
        msg.info("L_wp_np (%s):\n%s\n", L_wp_np.shape, L_wp_np)

        # Reconstruct the original matrix A from the factorization
        A_hat_wp_np = L_wp_np @ L_wp_np.T
        A_wp_np = A_hat_wp_np[P_wp_np][:, P_wp_np]
        msg.info("A_np (%s):\n%s\n", problem.A_np[0].shape, problem.A_np[0])
        msg.info("A_wp_np (%s):\n%s\n", A_wp_np.shape, A_wp_np)

        # Check matrix reconstruction against original matrix
        is_A_close = np.allclose(A_wp_np, problem.A_np[0], rtol=1e-3, atol=1e-4)
        if not is_A_close or self.verbose:
            print_error_stats("A", A_wp_np, problem.A_np[0], problem.dims[0])
        self.assertTrue(is_A_close)

        ###
        # Linear system solve
        ###

        # Prepare the solution vector x
        x_wp = wp.zeros_like(problem.b_wp, device=self.default_device)

        # Solve the linear system using the factorization
        llt.solve(b=problem.b_wp, x=x_wp)

        # Convert the warp array to numpy for verification
        x_wp_np = get_vector_block(0, x_wp.numpy(), problem.dims, problem.maxdims)
        msg.debug("x_np (%s):\n%s\n", problem.x_np[0].shape, problem.x_np[0])
        msg.debug("x_wp_np (%s):\n%s\n", x_wp_np.shape, x_wp_np)

        # Assert the result is as expected
        is_x_close = np.allclose(x_wp_np, problem.x_np[0], rtol=1e-3, atol=1e-4)
        if not is_x_close or self.verbose:
            print_error_stats("x", x_wp_np, problem.x_np[0], problem.dims[0])
        self.assertTrue(is_x_close)

        ###
        # Linear system solve in-place
        ###

        # Prepare the solution vector x
        x_wp = wp.zeros_like(problem.b_wp, device=self.default_device)
        wp.copy(x_wp, problem.b_wp)

        # Solve the linear system using the factorization
        llt.solve_inplace(x=x_wp)

        # Convert the warp array to numpy for verification
        x_wp_np = get_vector_block(0, x_wp.numpy(), problem.dims, problem.maxdims)
        msg.debug("x_wp_np (%s):\n%s\n", x_wp_np.shape, x_wp_np)

        # Assert the result is as expected
        is_x_close = np.allclose(x_wp_np, problem.x_np[0], rtol=1e-3, atol=1e-4)
        if not is_x_close or self.verbose:
            print_error_stats("x", x_wp_np, problem.x_np[0], problem.dims[0])
        self.assertTrue(is_x_close)

    def test_02_single_problem_dims_partially_active(self):
        """
        Test the blocked LLT solver on a single small problem.
        """
        # Constants
        # N_max = 16  # Use this for visual debugging with small matrices
        # N_act = 11
        N_max = 2000  # Use this for performance testing with large matrices
        N_act = 1537

        # Create a single-instance problem
        problem = RandomProblemLLT(
            dims=N_act,
            maxdims=N_max,
            seed=self.seed,
            np_dtype=np.float32,
            wp_dtype=wp.float32,
            device=self.default_device,
        )

        # Optional verbose output
        msg.debug("Problem:\n%s\n", problem)
        msg.debug("b_np:\n%s\n", problem.b_np[0])
        msg.debug("A_np:\n%s\n", problem.A_np[0])
        msg.debug("X_np:\n%s\n", problem.X_np[0])
        msg.debug("y_np:\n%s\n", problem.y_np[0])
        msg.debug("x_np:\n%s\n", problem.x_np[0])
        msg.info("A_wp (%s):\n%s\n", problem.A_wp.shape, problem.A_wp.numpy().reshape((N_max, N_max)))
        msg.info("b_wp (%s):\n%s\n", problem.b_wp.shape, problem.b_wp.numpy().reshape((N_max,)))

        # Create the linear operator meta-data
        opinfo = DenseSquareMultiLinearInfo()
        opinfo.finalize(dimensions=problem.maxdims, dtype=problem.wp_dtype, device=self.default_device)
        msg.debug("opinfo:\n%s", opinfo)

        # Create the linear operator data structure
        operator = DenseLinearOperatorData(info=opinfo, mat=problem.A_wp)
        msg.debug("operator.info:\n%s\n", operator.info)
        msg.debug("operator.mat (%s):\n%s\n", operator.mat.shape, operator.mat.numpy().reshape((N_max, N_max)))

        # Create a SequentialCholeskyFactorizer instance
        llt = LLTBlockedRCMSolver(operator=operator, device=self.default_device)
        self.assertIsNotNone(llt._operator)
        self.assertEqual(llt.dtype, problem.wp_dtype)
        self.assertEqual(llt.device, self.default_device)
        self.assertIsNotNone(llt._L)
        self.assertIsNotNone(llt._y)
        self.assertEqual(llt.L.size, problem.A_wp.size)
        self.assertEqual(llt.y.size, problem.b_wp.size)

        # IMPORTANT: Now we set the active dimensions in the operator info
        operator.info.dim.fill_(N_act)

        ###
        # Matrix factorization
        ###

        # Factorize the target square-symmetric matrix
        llt.compute(A=problem.A_wp)
        msg.info("llt.L (%s):\n%s\n", llt.L.shape, llt.L.numpy().reshape((N_max, N_max)))

        # Extract L, P from warp to numpy
        P_wp_np = get_vector_block(0, llt.P.numpy(), problem.dims, problem.maxdims)
        msg.info("P_wp_np (%s):\n%s\n", P_wp_np.shape, P_wp_np)
        L_wp_np = get_matrix_block(0, llt.L.numpy(), problem.dims, problem.maxdims)
        msg.info("L_wp_np (%s):\n%s\n", L_wp_np.shape, L_wp_np)

        # Reconstruct the original matrix A from the factorization
        A_hat_wp_np = L_wp_np @ L_wp_np.T
        A_wp_np = A_hat_wp_np[P_wp_np][:, P_wp_np]
        msg.info("A_np (%s):\n%s\n", problem.A_np[0].shape, problem.A_np[0])
        msg.info("A_wp_np (%s):\n%s\n", A_wp_np.shape, A_wp_np)

        # Check matrix reconstruction against original matrix
        is_A_close = np.allclose(A_wp_np, problem.A_np[0], rtol=1e-3, atol=1e-4)
        if not is_A_close or self.verbose:
            print_error_stats("A", A_wp_np, problem.A_np[0], problem.dims[0])
        self.assertTrue(is_A_close)

        ###
        # Linear system solve
        ###

        # Prepare the solution vector x
        x_wp = wp.zeros_like(problem.b_wp, device=self.default_device)

        # Solve the linear system using the factorization
        llt.solve(b=problem.b_wp, x=x_wp)

        # Convert the warp array to numpy for verification
        x_wp_np = get_vector_block(0, x_wp.numpy(), problem.dims, problem.maxdims)
        msg.debug("x_wp_np (%s):\n%s\n", x_wp_np.shape, x_wp_np)

        # Assert the result is as expected
        is_x_close = np.allclose(x_wp_np, problem.x_np[0], rtol=1e-3, atol=1e-4)
        if not is_x_close or self.verbose:
            print_error_stats("x", x_wp_np, problem.x_np[0], problem.dims[0])
        self.assertTrue(is_x_close)

        ###
        # Linear system solve in-place
        ###

        # Prepare the solution vector x
        x_wp = wp.zeros_like(problem.b_wp, device=self.default_device)
        wp.copy(x_wp, problem.b_wp)

        # Solve the linear system using the factorization
        llt.solve_inplace(x=x_wp)

        # Convert the warp array to numpy for verification
        x_wp_np = get_vector_block(0, x_wp.numpy(), problem.dims, problem.maxdims)
        msg.debug("x_wp_np (%s):\n%s\n", x_wp_np.shape, x_wp_np)

        # Assert the result is as expected
        is_x_close = np.allclose(x_wp_np, problem.x_np[0], rtol=1e-3, atol=1e-4)
        if not is_x_close or self.verbose:
            print_error_stats("x", x_wp_np, problem.x_np[0], problem.dims[0])
        self.assertTrue(is_x_close)

    def test_03_multiple_problems_dims_all_active(self):
        """
        Test the blocked LLT solver on multiple small problems.
        """
        # Constants
        N = [7, 8, 9, 10, 11]
        # N = [16, 64, 128, 512, 1024]

        # Create a single-instance problem
        problem = RandomProblemLLT(
            dims=N,
            seed=self.seed,
            np_dtype=np.float32,
            wp_dtype=wp.float32,
            device=self.default_device,
        )
        msg.debug("Problem:\n%s\n", problem)

        # Optional verbose output
        for i in range(problem.num_blocks):
            A_wp_np = get_matrix_block(i, problem.A_wp.numpy(), problem.dims, problem.maxdims)
            b_wp_np = get_vector_block(i, problem.b_wp.numpy(), problem.dims, problem.maxdims)
            msg.debug("[%d]: b_np:\n%s\n", i, problem.b_np[i])
            msg.debug("[%d]: A_np:\n%s\n", i, problem.A_np[i])
            msg.debug("[%d]: X_np:\n%s\n", i, problem.X_np[i])
            msg.debug("[%d]: y_np:\n%s\n", i, problem.y_np[i])
            msg.debug("[%d]: x_np:\n%s\n", i, problem.x_np[i])
            msg.info("[%d]: A_wp_np (%s):\n%s\n", i, A_wp_np.shape, A_wp_np)
            msg.info("[%d]: b_wp_np (%s):\n%s\n", i, b_wp_np.shape, b_wp_np)

        # Create the linear operator meta-data
        opinfo = DenseSquareMultiLinearInfo()
        opinfo.finalize(dimensions=problem.dims, dtype=problem.wp_dtype, device=self.default_device)
        msg.debug("opinfo:\n%s", opinfo)

        # Create the linear operator data structure
        operator = DenseLinearOperatorData(info=opinfo, mat=problem.A_wp)
        msg.debug("operator.info:\n%s\n", operator.info)
        msg.debug("operator.mat shape:\n%s\n", operator.mat.shape)

        # Create a SequentialCholeskyFactorizer instance
        llt = LLTBlockedRCMSolver(operator=operator, device=self.default_device)
        self.assertIsNotNone(llt._operator)
        self.assertEqual(llt.dtype, problem.wp_dtype)
        self.assertEqual(llt.device, self.default_device)
        self.assertIsNotNone(llt._L)
        self.assertIsNotNone(llt._y)
        self.assertEqual(llt.L.size, problem.A_wp.size)
        self.assertEqual(llt.y.size, problem.b_wp.size)

        ###
        # Matrix factorization
        ###

        # Factorize the target square-symmetric matrix
        llt.compute(A=problem.A_wp)

        # Iterate over all problems for verification
        for i in range(problem.num_blocks):
            # Extract L, P from warp to numpy
            P_wp_np = get_vector_block(i, llt.P.numpy(), problem.dims, problem.maxdims)
            msg.info("P_wp_np (%s):\n%s\n", P_wp_np.shape, P_wp_np)
            L_wp_np = get_matrix_block(i, llt.L.numpy(), problem.dims, problem.maxdims)
            msg.info("L_wp_np (%s):\n%s\n", L_wp_np.shape, L_wp_np)

            # Reconstruct the original matrix A from the factorization
            A_hat_wp_np = L_wp_np @ L_wp_np.T
            A_wp_np = A_hat_wp_np[P_wp_np][:, P_wp_np]
            msg.info("A_np (%s):\n%s\n", problem.A_np[i].shape, problem.A_np[i])
            msg.info("A_wp_np (%s):\n%s\n", A_wp_np.shape, A_wp_np)

            # Check matrix reconstruction against original matrix
            is_A_close = np.allclose(A_wp_np, problem.A_np[i], rtol=1e-3, atol=1e-4)
            if not is_A_close or self.verbose:
                print_error_stats("A", A_wp_np, problem.A_np[i], problem.dims[i])
            self.assertTrue(is_A_close)

        ###
        # Linear system solve
        ###

        # Prepare the solution vector x
        x_wp = wp.zeros_like(problem.b_wp, device=self.default_device)

        # Solve the linear system using the factorization
        llt.solve(b=problem.b_wp, x=x_wp)

        # Iterate over all problems for verification
        for i in range(problem.num_blocks):
            # Convert the warp array to numpy for verification
            x_wp_np = get_vector_block(i, x_wp.numpy(), problem.dims, problem.maxdims)
            msg.debug("x_wp_np (%s):\n%s\n", x_wp_np.shape, x_wp_np)
            msg.debug("x_np (%s):\n%s\n", problem.x_np[i].shape, problem.x_np[i])

            # Assert the result is as expected
            is_x_close = np.allclose(x_wp_np, problem.x_np[i], rtol=1e-3, atol=1e-4)
            if not is_x_close or self.verbose:
                print_error_stats("x", x_wp_np, problem.x_np[i], problem.dims[i])
            self.assertTrue(is_x_close)

        ###
        # Linear system solve in-place
        ###

        # Prepare the solution vector x
        x_wp = wp.zeros_like(problem.b_wp, device=self.default_device)
        wp.copy(x_wp, problem.b_wp)

        # Solve the linear system using the factorization
        llt.solve_inplace(x=x_wp)

        # Iterate over all problems for verification
        for i in range(problem.num_blocks):
            # Convert the warp array to numpy for verification
            x_wp_np = get_vector_block(i, x_wp.numpy(), problem.dims, problem.maxdims)
            msg.debug("x_wp_np (%s):\n%s\n", x_wp_np.shape, x_wp_np)
            msg.debug("x_np (%s):\n%s\n", problem.x_np[i].shape, problem.x_np[i])

            # Assert the result is as expected
            is_x_close = np.allclose(x_wp_np, problem.x_np[i], rtol=1e-3, atol=1e-4)
            if not is_x_close or self.verbose:
                print_error_stats("x", x_wp_np, problem.x_np[i], problem.dims[i])
            self.assertTrue(is_x_close)

    def test_04_multiple_problems_dims_partially_active(self):
        """
        Test the blocked LLT solver on multiple small problems.
        """
        # Constants
        # N_max = [7, 8, 9, 14, 21]  # Use this for visual debugging with small matrices
        # N_act = [5, 6, 4, 11, 17]
        N_max = [16, 64, 128, 512, 1024]  # Use this for performance testing with large matrices
        N_act = [11, 51, 101, 376, 999]

        # Create a single-instance problem
        problem = RandomProblemLLT(
            dims=N_act,
            maxdims=N_max,
            seed=self.seed,
            np_dtype=np.float32,
            wp_dtype=wp.float32,
            device=self.default_device,
        )
        msg.debug("Problem:\n%s\n", problem)

        # Optional verbose output
        for i in range(problem.num_blocks):
            A_wp_np = get_matrix_block(i, problem.A_wp.numpy(), problem.dims, problem.maxdims)
            b_wp_np = get_vector_block(i, problem.b_wp.numpy(), problem.dims, problem.maxdims)
            msg.debug("[%d]: b_np:\n%s\n", i, problem.b_np[i])
            msg.debug("[%d]: A_np:\n%s\n", i, problem.A_np[i])
            msg.debug("[%d]: X_np:\n%s\n", i, problem.X_np[i])
            msg.debug("[%d]: y_np:\n%s\n", i, problem.y_np[i])
            msg.debug("[%d]: x_np:\n%s\n", i, problem.x_np[i])
            msg.info("[%d]: A_wp_np (%s):\n%s\n", i, A_wp_np.shape, A_wp_np)
            msg.info("[%d]: b_wp_np (%s):\n%s\n", i, b_wp_np.shape, b_wp_np)

        # Create the linear operator meta-data
        opinfo = DenseSquareMultiLinearInfo()
        opinfo.finalize(dimensions=problem.maxdims, dtype=problem.wp_dtype, device=self.default_device)
        msg.debug("opinfo:\n%s", opinfo)

        # Create the linear operator data structure
        operator = DenseLinearOperatorData(info=opinfo, mat=problem.A_wp)
        msg.debug("operator.info:\n%s\n", operator.info)
        msg.debug("operator.mat shape:\n%s\n", operator.mat.shape)

        # Create a SequentialCholeskyFactorizer instance
        llt = LLTBlockedRCMSolver(operator=operator, device=self.default_device)
        self.assertIsNotNone(llt._operator)
        self.assertEqual(llt.dtype, problem.wp_dtype)
        self.assertEqual(llt.device, self.default_device)
        self.assertIsNotNone(llt._L)
        self.assertIsNotNone(llt._y)
        self.assertEqual(llt.L.size, problem.A_wp.size)
        self.assertEqual(llt.y.size, problem.b_wp.size)

        # IMPORTANT: Now we set the active dimensions in the operator info
        operator.info.dim.assign(N_act)

        ###
        # Matrix factorization
        ###

        # Factorize the target square-symmetric matrix
        llt.compute(A=problem.A_wp)

        # Iterate over all problems for verification
        for i in range(problem.num_blocks):
            # Extract L, P from warp to numpy
            P_wp_np = get_vector_block(i, llt.P.numpy(), problem.dims, problem.maxdims)
            msg.info("P_wp_np (%s):\n%s\n", P_wp_np.shape, P_wp_np)
            L_wp_np = get_matrix_block(i, llt.L.numpy(), problem.dims, problem.maxdims)
            msg.info("L_wp_np (%s):\n%s\n", L_wp_np.shape, L_wp_np)
            msg.info("X_np (%s):\n%s\n", problem.X_np[i].shape, problem.X_np[i])

            # Reconstruct the original matrix A from the factorization
            A_hat_wp_np = L_wp_np @ L_wp_np.T
            A_wp_np = A_hat_wp_np[P_wp_np][:, P_wp_np]
            msg.info("A_np (%s):\n%s\n", problem.A_np[i].shape, problem.A_np[i])
            msg.info("A_wp_np (%s):\n%s\n", A_wp_np.shape, A_wp_np)

            # Check matrix reconstruction against original matrix
            is_A_close = np.allclose(A_wp_np, problem.A_np[i], rtol=1e-3, atol=1e-4)
            if not is_A_close or self.verbose:
                print_error_stats("A", A_wp_np, problem.A_np[i], problem.dims[i])
            self.assertTrue(is_A_close)

        ###
        # Linear system solve
        ###

        # Prepare the solution vector x
        x_wp = wp.zeros_like(problem.b_wp, device=self.default_device)

        # Solve the linear system using the factorization
        llt.solve(b=problem.b_wp, x=x_wp)

        # Iterate over all problems for verification
        for i in range(problem.num_blocks):
            # Convert the warp array to numpy for verification
            x_wp_np = get_vector_block(i, x_wp.numpy(), problem.dims, problem.maxdims)
            msg.debug("x_wp_np (%s):\n%s\n", x_wp_np.shape, x_wp_np)
            msg.debug("x_np (%s):\n%s\n", problem.x_np[i].shape, problem.x_np[i])

            # Assert the result is as expected
            is_x_close = np.allclose(x_wp_np, problem.x_np[i], rtol=1e-3, atol=1e-4)
            if not is_x_close or self.verbose:
                print_error_stats("x", x_wp_np, problem.x_np[i], problem.dims[i])
            self.assertTrue(is_x_close)

        ###
        # Linear system solve in-place
        ###

        # Prepare the solution vector x
        x_wp = wp.zeros_like(problem.b_wp, device=self.default_device)
        wp.copy(x_wp, problem.b_wp)

        # Solve the linear system using the factorization
        llt.solve_inplace(x=x_wp)

        # Iterate over all problems for verification
        for i in range(problem.num_blocks):
            # Convert the warp array to numpy for verification
            x_wp_np = get_vector_block(i, x_wp.numpy(), problem.dims, problem.maxdims)
            msg.debug("x_wp_np (%s):\n%s\n", x_wp_np.shape, x_wp_np)
            msg.debug("x_np (%s):\n%s\n", problem.x_np[i].shape, problem.x_np[i])

            # Assert the result is as expected
            is_x_close = np.allclose(x_wp_np, problem.x_np[i], rtol=1e-3, atol=1e-4)
            if not is_x_close or self.verbose:
                print_error_stats("x", x_wp_np, problem.x_np[i], problem.dims[i])
            self.assertTrue(is_x_close)


###
# Test execution
###

if __name__ == "__main__":
    # Test setup
    setup_tests()

    # TODO: How can we get these to work?
    # Ensure the AOT module is compiled for the current device
    # wp.compile_aot_module(module=linear, device=wp.get_preferred_device())
    # wp.load_aot_module(module=linear.factorize.llt_sequential, device=wp.get_preferred_device())

    # Run all tests
    unittest.main(verbosity=2)
