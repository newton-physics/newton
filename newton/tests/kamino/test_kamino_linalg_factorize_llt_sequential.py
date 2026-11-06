# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the LLTSequentialSolver from linalg/linear.py"""

import logging
import unittest

import numpy as np
import warp as wp

from newton._src.solvers.kamino._src.linalg.core import DenseLinearOperatorData, DenseSquareMultiLinearInfo
from newton._src.solvers.kamino._src.linalg.factorize import (
    llt_sequential_factorize,
    llt_sequential_solve,
    llt_sequential_solve_inplace,
)
from newton._src.solvers.kamino._src.utils.logger import reset_log_level, set_log_level
from newton.tests.kamino import setup_tests, test_context
from newton.tests.kamino.utils.extract import get_matrix_block, get_vector_block
from newton.tests.kamino.utils.print import debug_print_error_stats
from newton.tests.kamino.utils.rand import RandomProblemLLT

logger = logging.getLogger(__name__)

###
# Tests
###


class TestLinAlgLLTSequential(unittest.TestCase):
    def setUp(self):
        # Configs
        if not test_context.setup_done:
            setup_tests(clear_cache=False)
        self.seed = 42
        self.default_device = wp.get_device(test_context.device)
        self.verbose = test_context.verbose  # Set to True for verbose output

        # Set debug-level logging to print verbose test output to console
        if self.verbose:
            set_log_level(logging.DEBUG)
            logger.debug("\n")  # Add newline before test output for better readability
        else:
            reset_log_level()

    def tearDown(self):
        self.default_device = None
        if self.verbose:
            reset_log_level()

    def test_01_single_problem_dims_all_active(self):
        """
        Test the sequential LLT solver on a single small problem.
        """
        # Constants
        N = 12  # Use this for unit testing with small matrices
        # N = 2000  # Use this for performance testing with large matrices

        # Create a single-instance problem
        problem = RandomProblemLLT(
            dims=N,
            seed=self.seed,
            np_dtype=np.float32,
            wp_dtype=wp.float32,
            device=self.default_device,
        )

        # Optional verbose output
        logger.debug("Problem:\n%s\n", problem)
        logger.debug("b_np:\n%s\n", problem.b_np[0])
        logger.debug("A_np:\n%s\n", problem.A_np[0])
        logger.debug("X_np:\n%s\n", problem.X_np[0])
        logger.debug("y_np:\n%s\n", problem.y_np[0])
        logger.debug("x_np:\n%s\n", problem.x_np[0])
        logger.debug("A_wp (%s):\n%s\n", problem.A_wp.shape, problem.A_wp.numpy().reshape((N, N)))
        logger.debug("b_wp (%s):\n%s\n", problem.b_wp.shape, problem.b_wp.numpy().reshape((N,)))

        # Create the linear operator meta-data
        opinfo = DenseSquareMultiLinearInfo()
        opinfo.finalize(dimensions=problem.dims, dtype=problem.wp_dtype, device=self.default_device)
        logger.debug("opinfo:\n%s", opinfo)

        # Create the linear operator data structure
        operator = DenseLinearOperatorData(info=opinfo, mat=problem.A_wp)
        logger.debug("operator.info:\n%s\n", operator.info)
        logger.debug("operator.mat (%s):\n%s\n", operator.mat.shape, operator.mat.numpy().reshape((N, N)))

        # Allocate LLT data arrays
        L_wp = wp.zeros_like(problem.A_wp, device=self.default_device)
        y_wp = wp.zeros_like(problem.b_wp, device=self.default_device)

        ###
        # Matrix factorization
        ###

        # Factorize the target square-symmetric matrix
        llt_sequential_factorize(
            num_blocks=problem.num_blocks,
            dim=operator.info.dim,
            mio=operator.info.mio,
            A=problem.A_wp,
            L=L_wp,
        )

        # Check matrix factorization against numpy
        L_wp_np = get_matrix_block(0, L_wp.numpy(), problem.dims, problem.maxdims)
        logger.debug("L_wp_np (%s):\n%s\n", L_wp_np.shape, L_wp_np)
        debug_print_error_stats("L", L_wp_np, problem.X_np[0], problem.dims[0])
        np.testing.assert_allclose(L_wp_np, problem.X_np[0], rtol=1e-3, atol=1e-4, err_msg="L")

        # Check matrix reconstruction against original matrix
        A_wp_np = L_wp_np @ L_wp_np.T
        logger.debug("A_np (%s):\n%s\n", problem.A_np[0].shape, problem.A_np[0])
        logger.debug("A_wp_np (%s):\n%s\n", A_wp_np.shape, A_wp_np)
        debug_print_error_stats("A", A_wp_np, problem.A_np[0], problem.dims[0])
        np.testing.assert_allclose(A_wp_np, problem.A_np[0], rtol=1e-3, atol=1e-4, err_msg="A")

        ###
        # Linear system solve
        ###

        # Prepare the solution vector x
        x_wp = wp.zeros_like(problem.b_wp, device=self.default_device)

        # Solve the linear system using the factorization
        llt_sequential_solve(
            num_blocks=problem.num_blocks,
            dim=operator.info.dim,
            mio=operator.info.mio,
            vio=operator.info.vio,
            L=L_wp,
            b=problem.b_wp,
            y=y_wp,
            x=x_wp,
        )

        # Convert the warp array to numpy for verification
        y_wp_np = get_vector_block(0, y_wp.numpy(), problem.dims, problem.maxdims)
        x_wp_np = get_vector_block(0, x_wp.numpy(), problem.dims, problem.maxdims)
        logger.debug("y_np (%s):\n%s\n", problem.y_np[0].shape, problem.y_np[0])
        logger.debug("y_wp_np (%s):\n%s\n", y_wp_np.shape, y_wp_np)
        logger.debug("x_np (%s):\n%s\n", problem.x_np[0].shape, problem.x_np[0])
        logger.debug("x_wp_np (%s):\n%s\n", x_wp_np.shape, x_wp_np)

        # Assert the result is as expected
        debug_print_error_stats("y", y_wp_np, problem.y_np[0], problem.dims[0])
        np.testing.assert_allclose(y_wp_np, problem.y_np[0], rtol=1e-3, atol=1e-4, err_msg="y")
        debug_print_error_stats("x", x_wp_np, problem.x_np[0], problem.dims[0])
        np.testing.assert_allclose(x_wp_np, problem.x_np[0], rtol=1e-3, atol=1e-4, err_msg="x")

        ###
        # Linear system solve in-place
        ###

        # Prepare the solution vector x
        x_wp = wp.zeros_like(problem.b_wp, device=self.default_device)
        wp.copy(x_wp, problem.b_wp)

        # Solve the linear system using the factorization
        llt_sequential_solve_inplace(
            num_blocks=problem.num_blocks,
            dim=operator.info.dim,
            mio=operator.info.mio,
            vio=operator.info.vio,
            L=L_wp,
            x=x_wp,
        )

        # Convert the warp array to numpy for verification
        y_wp_np = get_vector_block(0, y_wp.numpy(), problem.dims, problem.maxdims)
        x_wp_np = get_vector_block(0, x_wp.numpy(), problem.dims, problem.maxdims)
        logger.debug("y_wp_np (%s):\n%s\n", y_wp_np.shape, y_wp_np)
        logger.debug("x_wp_np (%s):\n%s\n", x_wp_np.shape, x_wp_np)

        # Assert the result is as expected
        debug_print_error_stats("y", y_wp_np, problem.y_np[0], problem.dims[0])
        np.testing.assert_allclose(y_wp_np, problem.y_np[0], rtol=1e-3, atol=1e-4, err_msg="y")
        debug_print_error_stats("x", x_wp_np, problem.x_np[0], problem.dims[0])
        np.testing.assert_allclose(x_wp_np, problem.x_np[0], rtol=1e-3, atol=1e-4, err_msg="x")

    def test_02_single_problem_dims_partially_active(self):
        """
        Test the sequential LLT solver on a single small problem.
        """
        # Constants
        N_max = 16  # Use this for unit testing with small matrices
        N_act = 11
        # N_max = 2000  # Use this for performance testing with large matrices
        # N_act = 1537

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
        logger.debug("Problem:\n%s\n", problem)
        logger.debug("b_np:\n%s\n", problem.b_np[0])
        logger.debug("A_np:\n%s\n", problem.A_np[0])
        logger.debug("X_np:\n%s\n", problem.X_np[0])
        logger.debug("y_np:\n%s\n", problem.y_np[0])
        logger.debug("x_np:\n%s\n", problem.x_np[0])
        logger.debug("A_wp (%s):\n%s\n", problem.A_wp.shape, problem.A_wp.numpy().reshape((N_max, N_max)))
        logger.debug("b_wp (%s):\n%s\n", problem.b_wp.shape, problem.b_wp.numpy().reshape((N_max,)))

        # Create the linear operator meta-data
        opinfo = DenseSquareMultiLinearInfo()
        opinfo.finalize(dimensions=problem.maxdims, dtype=problem.wp_dtype, device=self.default_device)
        logger.debug("opinfo:\n%s", opinfo)

        # Create the linear operator data structure
        operator = DenseLinearOperatorData(info=opinfo, mat=problem.A_wp)
        logger.debug("operator.info:\n%s\n", operator.info)
        logger.debug("operator.mat (%s):\n%s\n", operator.mat.shape, operator.mat.numpy().reshape((N_max, N_max)))

        # IMPORTANT: Now we set the active dimensions in the operator info
        operator.info.dim.fill_(N_act)

        # Allocate LLT data arrays
        L_wp = wp.zeros_like(problem.A_wp, device=self.default_device)
        y_wp = wp.zeros_like(problem.b_wp, device=self.default_device)

        ###
        # Matrix factorization
        ###

        # Factorize the target square-symmetric matrix
        llt_sequential_factorize(
            num_blocks=problem.num_blocks,
            dim=operator.info.dim,
            mio=operator.info.mio,
            A=problem.A_wp,
            L=L_wp,
        )

        # Check matrix factorization against numpy
        L_wp_np = get_matrix_block(0, L_wp.numpy(), problem.dims, problem.maxdims)
        logger.debug("L_wp_np (%s):\n%s\n", L_wp_np.shape, L_wp_np)
        debug_print_error_stats("L", L_wp_np, problem.X_np[0], problem.dims[0])
        np.testing.assert_allclose(L_wp_np, problem.X_np[0], rtol=1e-3, atol=1e-4, err_msg="L")

        # Check matrix reconstruction against original matrix
        A_wp_np = L_wp_np @ L_wp_np.T
        logger.debug("A_np (%s):\n%s\n", problem.A_np[0].shape, problem.A_np[0])
        logger.debug("A_wp_np (%s):\n%s\n", A_wp_np.shape, A_wp_np)
        debug_print_error_stats("A", A_wp_np, problem.A_np[0], problem.dims[0])
        np.testing.assert_allclose(A_wp_np, problem.A_np[0], rtol=1e-3, atol=1e-4, err_msg="A")

        ###
        # Linear system solve
        ###

        # Prepare the solution vector x
        x_wp = wp.zeros_like(problem.b_wp, device=self.default_device)

        # Solve the linear system using the factorization
        llt_sequential_solve(
            num_blocks=problem.num_blocks,
            dim=operator.info.dim,
            mio=operator.info.mio,
            vio=operator.info.vio,
            L=L_wp,
            b=problem.b_wp,
            y=y_wp,
            x=x_wp,
        )

        # Convert the warp array to numpy for verification
        y_wp_np = get_vector_block(0, y_wp.numpy(), problem.dims, problem.maxdims)
        x_wp_np = get_vector_block(0, x_wp.numpy(), problem.dims, problem.maxdims)
        logger.debug("y_wp_np (%s):\n%s\n", y_wp_np.shape, y_wp_np)
        logger.debug("x_wp_np (%s):\n%s\n", x_wp_np.shape, x_wp_np)

        # Assert the result is as expected
        debug_print_error_stats("y", y_wp_np, problem.y_np[0], problem.dims[0])
        np.testing.assert_allclose(y_wp_np, problem.y_np[0], rtol=1e-3, atol=1e-4, err_msg="y")
        debug_print_error_stats("x", x_wp_np, problem.x_np[0], problem.dims[0])
        np.testing.assert_allclose(x_wp_np, problem.x_np[0], rtol=1e-3, atol=1e-4, err_msg="x")

        ###
        # Linear system solve in-place
        ###

        # Prepare the solution vector x
        x_wp = wp.zeros_like(problem.b_wp, device=self.default_device)
        wp.copy(x_wp, problem.b_wp)

        # Solve the linear system using the factorization
        llt_sequential_solve_inplace(
            num_blocks=problem.num_blocks,
            dim=operator.info.dim,
            mio=operator.info.mio,
            vio=operator.info.vio,
            L=L_wp,
            x=x_wp,
        )

        # Convert the warp array to numpy for verification
        y_wp_np = get_vector_block(0, y_wp.numpy(), problem.dims, problem.maxdims)
        x_wp_np = get_vector_block(0, x_wp.numpy(), problem.dims, problem.maxdims)
        logger.debug("y_wp_np (%s):\n%s\n", y_wp_np.shape, y_wp_np)
        logger.debug("x_wp_np (%s):\n%s\n", x_wp_np.shape, x_wp_np)

        # Assert the result is as expected
        debug_print_error_stats("y", y_wp_np, problem.y_np[0], problem.dims[0])
        np.testing.assert_allclose(y_wp_np, problem.y_np[0], rtol=1e-3, atol=1e-4, err_msg="y")
        debug_print_error_stats("x", x_wp_np, problem.x_np[0], problem.dims[0])
        np.testing.assert_allclose(x_wp_np, problem.x_np[0], rtol=1e-3, atol=1e-4, err_msg="x")

    def test_03_multiple_problems_dims_all_active(self):
        """
        Test the sequential LLT solver on multiple small problems.
        """
        # Constants
        N = [7, 8, 9, 10, 11]  # Use this for unit testing with small matrices
        # N = [16, 64, 128, 512, 1024]  # Use this for performance testing with large matrices

        # Create a single-instance problem
        problem = RandomProblemLLT(
            dims=N,
            seed=self.seed,
            np_dtype=np.float32,
            wp_dtype=wp.float32,
            device=self.default_device,
        )
        logger.debug("Problem:\n%s\n", problem)

        # Optional verbose output
        for i in range(problem.num_blocks):
            A_wp_np = get_matrix_block(i, problem.A_wp.numpy(), problem.dims, problem.maxdims)
            b_wp_np = get_vector_block(i, problem.b_wp.numpy(), problem.dims, problem.maxdims)
            logger.debug("[%d]: b_np:\n%s\n", i, problem.b_np[i])
            logger.debug("[%d]: A_np:\n%s\n", i, problem.A_np[i])
            logger.debug("[%d]: X_np:\n%s\n", i, problem.X_np[i])
            logger.debug("[%d]: y_np:\n%s\n", i, problem.y_np[i])
            logger.debug("[%d]: x_np:\n%s\n", i, problem.x_np[i])
            logger.debug("[%d]: A_wp_np (%s):\n%s\n", i, A_wp_np.shape, A_wp_np)
            logger.debug("[%d]: b_wp_np (%s):\n%s\n", i, b_wp_np.shape, b_wp_np)

        # Create the linear operator meta-data
        opinfo = DenseSquareMultiLinearInfo()
        opinfo.finalize(dimensions=problem.dims, dtype=problem.wp_dtype, device=self.default_device)
        logger.debug("opinfo:\n%s", opinfo)

        # Create the linear operator data structure
        operator = DenseLinearOperatorData(info=opinfo, mat=problem.A_wp)
        logger.debug("operator.info:\n%s\n", operator.info)
        logger.debug("operator.mat shape:\n%s\n", operator.mat.shape)

        # Allocate LLT data arrays
        L_wp = wp.zeros_like(problem.A_wp, device=self.default_device)
        y_wp = wp.zeros_like(problem.b_wp, device=self.default_device)

        ###
        # Matrix factorization
        ###

        # Factorize the target square-symmetric matrix
        llt_sequential_factorize(
            num_blocks=problem.num_blocks,
            dim=operator.info.dim,
            mio=operator.info.mio,
            A=problem.A_wp,
            L=L_wp,
        )

        # Iterate over all problems for verification
        for i in range(problem.num_blocks):
            # Check matrix factorization against numpy
            L_wp_np = get_matrix_block(i, L_wp.numpy(), problem.dims, problem.maxdims)
            logger.debug("L_wp_np (%s):\n%s\n", L_wp_np.shape, L_wp_np)
            logger.debug("X_np (%s):\n%s\n", problem.X_np[i].shape, problem.X_np[i])
            debug_print_error_stats("L", L_wp_np, problem.X_np[i], problem.dims[i])
            np.testing.assert_allclose(L_wp_np, problem.X_np[i], rtol=1e-3, atol=1e-4, err_msg="L")

            # Check matrix reconstruction against original matrix
            A_wp_np = L_wp_np @ L_wp_np.T
            logger.debug("A_np (%s):\n%s\n", problem.A_np[i].shape, problem.A_np[i])
            logger.debug("A_wp_np (%s):\n%s\n", A_wp_np.shape, A_wp_np)
            debug_print_error_stats("A", A_wp_np, problem.A_np[i], problem.dims[i])
            np.testing.assert_allclose(A_wp_np, problem.A_np[i], rtol=1e-3, atol=1e-4, err_msg="A")

        ###
        # Linear system solve
        ###

        # Prepare the solution vector x
        x_wp = wp.zeros_like(problem.b_wp, device=self.default_device)

        # Solve the linear system using the factorization
        llt_sequential_solve(
            num_blocks=problem.num_blocks,
            dim=operator.info.dim,
            mio=operator.info.mio,
            vio=operator.info.vio,
            L=L_wp,
            b=problem.b_wp,
            y=y_wp,
            x=x_wp,
        )

        # Iterate over all problems for verification
        for i in range(problem.num_blocks):
            # Convert the warp array to numpy for verification
            y_wp_np = get_vector_block(i, y_wp.numpy(), problem.dims, problem.maxdims)
            x_wp_np = get_vector_block(i, x_wp.numpy(), problem.dims, problem.maxdims)
            logger.debug("y_wp_np (%s):\n%s\n", y_wp_np.shape, y_wp_np)
            logger.debug("y_np (%s):\n%s\n", problem.y_np[i].shape, problem.y_np[i])
            logger.debug("x_wp_np (%s):\n%s\n", x_wp_np.shape, x_wp_np)
            logger.debug("x_np (%s):\n%s\n", problem.x_np[i].shape, problem.x_np[i])

            # Assert the result is as expected
            debug_print_error_stats("y", y_wp_np, problem.y_np[i], problem.dims[i])
            np.testing.assert_allclose(y_wp_np, problem.y_np[i], rtol=1e-3, atol=1e-4, err_msg="y")
            debug_print_error_stats("x", x_wp_np, problem.x_np[i], problem.dims[i])
            np.testing.assert_allclose(x_wp_np, problem.x_np[i], rtol=1e-3, atol=1e-4, err_msg="x")

        ###
        # Linear system solve in-place
        ###

        # Prepare the solution vector x
        x_wp = wp.zeros_like(problem.b_wp, device=self.default_device)
        wp.copy(x_wp, problem.b_wp)

        # Solve the linear system using the factorization
        llt_sequential_solve_inplace(
            num_blocks=problem.num_blocks,
            dim=operator.info.dim,
            mio=operator.info.mio,
            vio=operator.info.vio,
            L=L_wp,
            x=x_wp,
        )

        # Iterate over all problems for verification
        for i in range(problem.num_blocks):
            # Convert the warp array to numpy for verification
            y_wp_np = get_vector_block(i, y_wp.numpy(), problem.dims, problem.maxdims)
            x_wp_np = get_vector_block(i, x_wp.numpy(), problem.dims, problem.maxdims)
            logger.debug("y_wp_np (%s):\n%s\n", y_wp_np.shape, y_wp_np)
            logger.debug("y_np (%s):\n%s\n", problem.y_np[i].shape, problem.y_np[i])
            logger.debug("x_wp_np (%s):\n%s\n", x_wp_np.shape, x_wp_np)
            logger.debug("x_np (%s):\n%s\n", problem.x_np[i].shape, problem.x_np[i])

            # Assert the result is as expected
            debug_print_error_stats("y", y_wp_np, problem.y_np[i], problem.dims[i])
            np.testing.assert_allclose(y_wp_np, problem.y_np[i], rtol=1e-3, atol=1e-4, err_msg="y")
            debug_print_error_stats("x", x_wp_np, problem.x_np[i], problem.dims[i])
            np.testing.assert_allclose(x_wp_np, problem.x_np[i], rtol=1e-3, atol=1e-4, err_msg="x")

    def test_04_multiple_problems_dims_partially_active(self):
        """
        Test the sequential LLT solver on multiple small problems.
        """
        # Constants
        N_max = [7, 8, 9, 14, 21]  # Use this for unit testing with small matrices
        N_act = [5, 6, 4, 11, 17]
        # N_max = [16, 64, 128, 512, 1024]  # Use this for performance testing with large matrices
        # N_act = [11, 51, 101, 376, 999]

        # Create a single-instance problem
        problem = RandomProblemLLT(
            dims=N_act,
            maxdims=N_max,
            seed=self.seed,
            np_dtype=np.float32,
            wp_dtype=wp.float32,
            device=self.default_device,
        )
        logger.debug("Problem:\n%s\n", problem)

        # Optional verbose output
        for i in range(problem.num_blocks):
            A_wp_np = get_matrix_block(i, problem.A_wp.numpy(), problem.dims, problem.maxdims)
            b_wp_np = get_vector_block(i, problem.b_wp.numpy(), problem.dims, problem.maxdims)
            logger.debug("[%d]: b_np:\n%s\n", i, problem.b_np[i])
            logger.debug("[%d]: A_np:\n%s\n", i, problem.A_np[i])
            logger.debug("[%d]: X_np:\n%s\n", i, problem.X_np[i])
            logger.debug("[%d]: y_np:\n%s\n", i, problem.y_np[i])
            logger.debug("[%d]: x_np:\n%s\n", i, problem.x_np[i])
            logger.debug("[%d]: A_wp_np (%s):\n%s\n", i, A_wp_np.shape, A_wp_np)
            logger.debug("[%d]: b_wp_np (%s):\n%s\n", i, b_wp_np.shape, b_wp_np)

        # Create the linear operator meta-data
        opinfo = DenseSquareMultiLinearInfo()
        opinfo.finalize(dimensions=problem.maxdims, dtype=problem.wp_dtype, device=self.default_device)
        logger.debug("opinfo:\n%s", opinfo)

        # Create the linear operator data structure
        operator = DenseLinearOperatorData(info=opinfo, mat=problem.A_wp)
        logger.debug("operator.info:\n%s\n", operator.info)
        logger.debug("operator.mat shape:\n%s\n", operator.mat.shape)

        # Allocate LLT data arrays
        L_wp = wp.zeros_like(problem.A_wp, device=self.default_device)
        y_wp = wp.zeros_like(problem.b_wp, device=self.default_device)

        # IMPORTANT: Now we set the active dimensions in the operator info
        operator.info.dim.assign(N_act)

        ###
        # Matrix factorization
        ###

        # Factorize the target square-symmetric matrix
        llt_sequential_factorize(
            num_blocks=problem.num_blocks,
            dim=operator.info.dim,
            mio=operator.info.mio,
            A=problem.A_wp,
            L=L_wp,
        )

        # Iterate over all problems for verification
        for i in range(problem.num_blocks):
            # Check matrix factorization against numpy
            L_wp_np = get_matrix_block(i, L_wp.numpy(), problem.dims, problem.maxdims)
            logger.debug("L_wp_np (%s):\n%s\n", L_wp_np.shape, L_wp_np)
            logger.debug("X_np (%s):\n%s\n", problem.X_np[i].shape, problem.X_np[i])
            debug_print_error_stats("L", L_wp_np, problem.X_np[i], problem.dims[i])
            np.testing.assert_allclose(L_wp_np, problem.X_np[i], rtol=1e-3, atol=1e-4, err_msg="L")

            # Check matrix reconstruction against original matrix
            A_wp_np = L_wp_np @ L_wp_np.T
            logger.debug("A_np (%s):\n%s\n", problem.A_np[i].shape, problem.A_np[i])
            logger.debug("A_wp_np (%s):\n%s\n", A_wp_np.shape, A_wp_np)
            debug_print_error_stats("A", A_wp_np, problem.A_np[i], problem.dims[i])
            np.testing.assert_allclose(A_wp_np, problem.A_np[i], rtol=1e-3, atol=1e-4, err_msg="A")

        ###
        # Linear system solve
        ###

        # Prepare the solution vector x
        x_wp = wp.zeros_like(problem.b_wp, device=self.default_device)

        # Solve the linear system using the factorization
        llt_sequential_solve(
            num_blocks=problem.num_blocks,
            dim=operator.info.dim,
            mio=operator.info.mio,
            vio=operator.info.vio,
            L=L_wp,
            b=problem.b_wp,
            y=y_wp,
            x=x_wp,
        )

        # Iterate over all problems for verification
        for i in range(problem.num_blocks):
            # Convert the warp array to numpy for verification
            y_wp_np = get_vector_block(i, y_wp.numpy(), problem.dims, problem.maxdims)
            x_wp_np = get_vector_block(i, x_wp.numpy(), problem.dims, problem.maxdims)
            logger.debug("y_wp_np (%s):\n%s\n", y_wp_np.shape, y_wp_np)
            logger.debug("y_np (%s):\n%s\n", problem.y_np[i].shape, problem.y_np[i])
            logger.debug("x_wp_np (%s):\n%s\n", x_wp_np.shape, x_wp_np)
            logger.debug("x_np (%s):\n%s\n", problem.x_np[i].shape, problem.x_np[i])

            # Assert the result is as expected
            debug_print_error_stats("y", y_wp_np, problem.y_np[i], problem.dims[i])
            np.testing.assert_allclose(y_wp_np, problem.y_np[i], rtol=1e-3, atol=1e-4, err_msg="y")
            debug_print_error_stats("x", x_wp_np, problem.x_np[i], problem.dims[i])
            np.testing.assert_allclose(x_wp_np, problem.x_np[i], rtol=1e-3, atol=1e-4, err_msg="x")

        ###
        # Linear system solve in-place
        ###

        # Prepare the solution vector x
        x_wp = wp.zeros_like(problem.b_wp, device=self.default_device)
        wp.copy(x_wp, problem.b_wp)

        # Solve the linear system using the factorization
        llt_sequential_solve_inplace(
            num_blocks=problem.num_blocks,
            dim=operator.info.dim,
            mio=operator.info.mio,
            vio=operator.info.vio,
            L=L_wp,
            x=x_wp,
        )

        # Iterate over all problems for verification
        for i in range(problem.num_blocks):
            # Convert the warp array to numpy for verification
            y_wp_np = get_vector_block(i, y_wp.numpy(), problem.dims, problem.maxdims)
            x_wp_np = get_vector_block(i, x_wp.numpy(), problem.dims, problem.maxdims)
            logger.debug("y_wp_np (%s):\n%s\n", y_wp_np.shape, y_wp_np)
            logger.debug("y_np (%s):\n%s\n", problem.y_np[i].shape, problem.y_np[i])
            logger.debug("x_wp_np (%s):\n%s\n", x_wp_np.shape, x_wp_np)
            logger.debug("x_np (%s):\n%s\n", problem.x_np[i].shape, problem.x_np[i])

            # Assert the result is as expected
            debug_print_error_stats("y", y_wp_np, problem.y_np[i], problem.dims[i])
            np.testing.assert_allclose(y_wp_np, problem.y_np[i], rtol=1e-3, atol=1e-4, err_msg="y")
            debug_print_error_stats("x", x_wp_np, problem.x_np[i], problem.dims[i])
            np.testing.assert_allclose(x_wp_np, problem.x_np[i], rtol=1e-3, atol=1e-4, err_msg="x")


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
