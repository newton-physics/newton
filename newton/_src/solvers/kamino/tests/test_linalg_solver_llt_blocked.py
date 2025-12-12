# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Unit tests for the LLTBlockedSolver from linalg/linear.py"""

import unittest

import numpy as np
import warp as wp

from newton._src.solvers.kamino.core.types import float32
from newton._src.solvers.kamino.linalg.core import DenseLinearOperatorData, DenseSquareMultiLinearInfo
from newton._src.solvers.kamino.linalg.linear import LLTBlockedSolver
from newton._src.solvers.kamino.tests import setup_tests, test_context
from newton._src.solvers.kamino.tests.utils.extract import get_matrix_block, get_vector_block
from newton._src.solvers.kamino.tests.utils.print import print_error_stats
from newton._src.solvers.kamino.tests.utils.rand import RandomProblemLLT
from newton._src.solvers.kamino.utils import logger as msg

###
# Tests
###


class TestLinAlgLLTBlockedSolver(unittest.TestCase):
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
        Test the default constructor of the LLTBlockedSolver class.
        """
        llt = LLTBlockedSolver(device=self.default_device)
        self.assertIsNone(llt._operator)
        self.assertEqual(llt.dtype, float32)
        self.assertEqual(llt.device, self.default_device)

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
            wp_dtype=float32,
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
        llt = LLTBlockedSolver(operator=operator, device=self.default_device)
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

        # Convert the warp array to numpy for verification
        L_wp_np = get_matrix_block(0, llt.L.numpy(), problem.dims, problem.maxdims)
        msg.info("L_wp_np (%s):\n%s\n", L_wp_np.shape, L_wp_np)

        # Check matrix factorization against numpy
        is_L_close = np.allclose(L_wp_np, problem.X_np[0], rtol=1e-3, atol=1e-4)
        if not is_L_close or self.verbose:
            print_error_stats("L", L_wp_np, problem.X_np[0], problem.dims[0])
        self.assertTrue(is_L_close)

        # Reconstruct the original matrix A from the factorization
        A_wp_np = L_wp_np @ L_wp_np.T
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
        y_wp_np = get_vector_block(0, llt.y.numpy(), problem.dims, problem.maxdims)
        x_wp_np = get_vector_block(0, x_wp.numpy(), problem.dims, problem.maxdims)
        msg.debug("y_np (%s):\n%s\n", problem.y_np[0].shape, problem.y_np[0])
        msg.debug("y_wp_np (%s):\n%s\n", y_wp_np.shape, y_wp_np)
        msg.debug("x_np (%s):\n%s\n", problem.x_np[0].shape, problem.x_np[0])
        msg.debug("x_wp_np (%s):\n%s\n", x_wp_np.shape, x_wp_np)

        # Assert the result is as expected
        is_y_close = np.allclose(y_wp_np, problem.y_np[0], rtol=1e-3, atol=1e-4)
        if not is_y_close or self.verbose:
            print_error_stats("y", y_wp_np, problem.y_np[0], problem.dims[0])
        self.assertTrue(is_y_close)

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
        y_wp_np = get_vector_block(0, llt.y.numpy(), problem.dims, problem.maxdims)
        x_wp_np = get_vector_block(0, x_wp.numpy(), problem.dims, problem.maxdims)
        msg.debug("y_wp_np (%s):\n%s\n", y_wp_np.shape, y_wp_np)
        msg.debug("x_wp_np (%s):\n%s\n", x_wp_np.shape, x_wp_np)

        # Assert the result is as expected
        is_y_close = np.allclose(y_wp_np, problem.y_np[0], rtol=1e-3, atol=1e-4)
        if not is_y_close or self.verbose:
            print_error_stats("y", y_wp_np, problem.y_np[0], problem.dims[0])
        self.assertTrue(is_y_close)

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
            wp_dtype=float32,
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
        llt = LLTBlockedSolver(operator=operator, device=self.default_device)
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

        # Convert the warp array to numpy for verification
        L_wp_np = get_matrix_block(0, llt.L.numpy(), problem.dims, problem.maxdims)
        msg.info("L_wp_np (%s):\n%s\n", L_wp_np.shape, L_wp_np)

        # Check matrix factorization against numpy
        is_L_close = np.allclose(L_wp_np, problem.X_np[0], rtol=1e-3, atol=1e-4)
        if not is_L_close or self.verbose:
            print_error_stats("L", L_wp_np, problem.X_np[0], problem.dims[0])
        self.assertTrue(is_L_close)

        # Reconstruct the original matrix A from the factorization
        A_wp_np = L_wp_np @ L_wp_np.T
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
        y_wp_np = get_vector_block(0, llt.y.numpy(), problem.dims, problem.maxdims)
        x_wp_np = get_vector_block(0, x_wp.numpy(), problem.dims, problem.maxdims)
        msg.debug("y_wp_np (%s):\n%s\n", y_wp_np.shape, y_wp_np)
        msg.debug("x_wp_np (%s):\n%s\n", x_wp_np.shape, x_wp_np)

        # Assert the result is as expected
        is_y_close = np.allclose(y_wp_np, problem.y_np[0], rtol=1e-3, atol=1e-4)
        if not is_y_close or self.verbose:
            print_error_stats("y", y_wp_np, problem.y_np[0], problem.dims[0])
        self.assertTrue(is_y_close)

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
        y_wp_np = get_vector_block(0, llt.y.numpy(), problem.dims, problem.maxdims)
        x_wp_np = get_vector_block(0, x_wp.numpy(), problem.dims, problem.maxdims)
        msg.debug("y_wp_np (%s):\n%s\n", y_wp_np.shape, y_wp_np)
        msg.debug("x_wp_np (%s):\n%s\n", x_wp_np.shape, x_wp_np)

        # Assert the result is as expected
        is_y_close = np.allclose(y_wp_np, problem.y_np[0], rtol=1e-3, atol=1e-4)
        if not is_y_close or self.verbose:
            print_error_stats("y", y_wp_np, problem.y_np[0], problem.dims[0])
        self.assertTrue(is_y_close)

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
            wp_dtype=float32,
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
        llt = LLTBlockedSolver(operator=operator, device=self.default_device)
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
            # Convert the warp array to numpy for verification
            L_wp_np = get_matrix_block(i, llt.L.numpy(), problem.dims, problem.maxdims)
            msg.info("L_wp_np (%s):\n%s\n", L_wp_np.shape, L_wp_np)
            msg.info("X_np (%s):\n%s\n", problem.X_np[i].shape, problem.X_np[i])

            # Check matrix factorization against numpy
            is_L_close = np.allclose(L_wp_np, problem.X_np[i], rtol=1e-3, atol=1e-4)
            if not is_L_close or self.verbose:
                print_error_stats("L", L_wp_np, problem.X_np[i], problem.dims[i])
            self.assertTrue(is_L_close)

            # Reconstruct the original matrix A from the factorization
            A_wp_np = L_wp_np @ L_wp_np.T
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
            y_wp_np = get_vector_block(i, llt.y.numpy(), problem.dims, problem.maxdims)
            x_wp_np = get_vector_block(i, x_wp.numpy(), problem.dims, problem.maxdims)
            msg.debug("y_wp_np (%s):\n%s\n", y_wp_np.shape, y_wp_np)
            msg.debug("y_np (%s):\n%s\n", problem.y_np[i].shape, problem.y_np[i])
            msg.debug("x_wp_np (%s):\n%s\n", x_wp_np.shape, x_wp_np)
            msg.debug("x_np (%s):\n%s\n", problem.x_np[i].shape, problem.x_np[i])

            # Assert the result is as expected
            is_y_close = np.allclose(y_wp_np, problem.y_np[i], rtol=1e-3, atol=1e-4)
            if not is_y_close or self.verbose:
                print_error_stats("y", y_wp_np, problem.y_np[i], problem.dims[i])
            self.assertTrue(is_y_close)

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
            y_wp_np = get_vector_block(i, llt.y.numpy(), problem.dims, problem.maxdims)
            x_wp_np = get_vector_block(i, x_wp.numpy(), problem.dims, problem.maxdims)
            msg.debug("y_wp_np (%s):\n%s\n", y_wp_np.shape, y_wp_np)
            msg.debug("y_np (%s):\n%s\n", problem.y_np[i].shape, problem.y_np[i])
            msg.debug("x_wp_np (%s):\n%s\n", x_wp_np.shape, x_wp_np)
            msg.debug("x_np (%s):\n%s\n", problem.x_np[i].shape, problem.x_np[i])

            # Assert the result is as expected
            is_y_close = np.allclose(y_wp_np, problem.y_np[i], rtol=1e-3, atol=1e-4)
            if not is_y_close or self.verbose:
                print_error_stats("y", y_wp_np, problem.y_np[i], problem.dims[i])
            self.assertTrue(is_y_close)

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
            wp_dtype=float32,
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
        llt = LLTBlockedSolver(operator=operator, device=self.default_device)
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
            # Convert the warp array to numpy for verification
            L_wp_np = get_matrix_block(i, llt.L.numpy(), problem.dims, problem.maxdims)
            msg.info("L_wp_np (%s):\n%s\n", L_wp_np.shape, L_wp_np)
            msg.info("X_np (%s):\n%s\n", problem.X_np[i].shape, problem.X_np[i])

            # Check matrix factorization against numpy
            is_L_close = np.allclose(L_wp_np, problem.X_np[i], rtol=1e-3, atol=1e-4)
            if not is_L_close or self.verbose:
                print_error_stats("L", L_wp_np, problem.X_np[i], problem.dims[i])
            self.assertTrue(is_L_close)

            # Reconstruct the original matrix A from the factorization
            A_wp_np = L_wp_np @ L_wp_np.T
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
            y_wp_np = get_vector_block(i, llt.y.numpy(), problem.dims, problem.maxdims)
            x_wp_np = get_vector_block(i, x_wp.numpy(), problem.dims, problem.maxdims)
            msg.debug("y_wp_np (%s):\n%s\n", y_wp_np.shape, y_wp_np)
            msg.debug("y_np (%s):\n%s\n", problem.y_np[i].shape, problem.y_np[i])
            msg.debug("x_wp_np (%s):\n%s\n", x_wp_np.shape, x_wp_np)
            msg.debug("x_np (%s):\n%s\n", problem.x_np[i].shape, problem.x_np[i])

            # Assert the result is as expected
            is_y_close = np.allclose(y_wp_np, problem.y_np[i], rtol=1e-3, atol=1e-4)
            if not is_y_close or self.verbose:
                print_error_stats("y", y_wp_np, problem.y_np[i], problem.dims[i])
            self.assertTrue(is_y_close)

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
            y_wp_np = get_vector_block(i, llt.y.numpy(), problem.dims, problem.maxdims)
            x_wp_np = get_vector_block(i, x_wp.numpy(), problem.dims, problem.maxdims)
            msg.debug("y_wp_np (%s):\n%s\n", y_wp_np.shape, y_wp_np)
            msg.debug("y_np (%s):\n%s\n", problem.y_np[i].shape, problem.y_np[i])
            msg.debug("x_wp_np (%s):\n%s\n", x_wp_np.shape, x_wp_np)
            msg.debug("x_np (%s):\n%s\n", problem.x_np[i].shape, problem.x_np[i])

            # Assert the result is as expected
            is_y_close = np.allclose(y_wp_np, problem.y_np[i], rtol=1e-3, atol=1e-4)
            if not is_y_close or self.verbose:
                print_error_stats("y", y_wp_np, problem.y_np[i], problem.dims[i])
            self.assertTrue(is_y_close)

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
