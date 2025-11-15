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

"""Unit tests for linalg/utils/rand.py"""

import unittest

import numpy as np
import warp as wp

from newton._src.solvers.kamino.linalg.utils.matrix import (
    SquareSymmetricMatrixProperties,
    is_square_matrix,
    is_symmetric_matrix,
)
from newton._src.solvers.kamino.linalg.utils.rand import (
    eigenvalues_from_distribution,
    random_rhs_for_matrix,
    random_spd_matrix,
    random_symmetric_matrix,
)
from newton._src.solvers.kamino.utils import logger as msg

###
# Tests
###


class TestLinAlgUtilsRandomMatrixSymmetric(unittest.TestCase):
    def setUp(self):
        # Configs
        self.seed = 42
        self.default_device = wp.get_device()
        self.verbose = False  # Set to True for verbose output

        # Set debug-level logging to print verbose test output to console
        if self.verbose:
            print("\n")  # Add newline before test output for better readability
            msg.set_log_level(msg.LogLevel.DEBUG)
        else:
            msg.set_log_level(msg.LogLevel.WARNING)

    def tearDown(self):
        self.default_device = None
        if self.verbose:
            msg.reset_log_level()

    def test_01_make_small_random_symmetric_matrix_in_fp64(self):
        # Generate a small random symmetric matrix in fp64
        A = random_symmetric_matrix(dim=10, dtype=np.float64, seed=self.seed)
        msg.debug("A:\n%s\n", A)

        # Check basic matrix properties
        self.assertEqual(A.shape, (10, 10))
        self.assertEqual(A.dtype, np.float64)
        self.assertTrue(is_square_matrix(A))
        self.assertTrue(is_symmetric_matrix(A))

    def test_02_make_medium_random_symmetric_matrix_in_fp64(self):
        # Generate a small random symmetric matrix in fp64
        A = random_symmetric_matrix(dim=100, dtype=np.float64, seed=self.seed)
        msg.debug("A:\n%s\n", A)

        # Check basic matrix properties
        self.assertEqual(A.shape, (100, 100))
        self.assertEqual(A.dtype, np.float64)
        self.assertTrue(is_square_matrix(A))
        self.assertTrue(is_symmetric_matrix(A))

    def test_03_make_large_random_symmetric_matrix_in_fp64(self):
        # Generate a small random symmetric matrix in fp64
        A = random_symmetric_matrix(dim=1000, dtype=np.float64, seed=self.seed)
        msg.debug("A:\n%s\n", A)

        # Check basic matrix properties
        self.assertEqual(A.shape, (1000, 1000))
        self.assertEqual(A.dtype, np.float64)
        self.assertTrue(is_square_matrix(A))
        self.assertTrue(is_symmetric_matrix(A))

    def test_04_make_small_random_symmetric_matrix_in_fp32(self):
        # Generate a small random symmetric matrix in fp32
        A = random_symmetric_matrix(dim=10, dtype=np.float32, seed=self.seed)
        msg.debug("A:\n%s\n", A)

        # Check basic matrix properties
        self.assertEqual(A.shape, (10, 10))
        self.assertEqual(A.dtype, np.float32)
        self.assertTrue(is_square_matrix(A))
        self.assertTrue(is_symmetric_matrix(A))

    def test_05_make_medium_random_symmetric_matrix_in_fp32(self):
        # Generate a small random symmetric matrix in fp32
        A = random_symmetric_matrix(dim=100, dtype=np.float32, seed=self.seed)
        msg.debug("A:\n%s\n", A)

        # Check basic matrix properties
        self.assertEqual(A.shape, (100, 100))
        self.assertEqual(A.dtype, np.float32)
        self.assertTrue(is_square_matrix(A))
        self.assertTrue(is_symmetric_matrix(A))

    def test_06_make_large_random_symmetric_matrix_in_fp32(self):
        # Generate a small random symmetric matrix in fp32
        A = random_symmetric_matrix(dim=1000, dtype=np.float32, seed=self.seed)
        msg.debug("A:\n%s\n", A)

        # Check basic matrix properties
        self.assertEqual(A.shape, (1000, 1000))
        self.assertEqual(A.dtype, np.float32)
        self.assertTrue(is_square_matrix(A))
        self.assertTrue(is_symmetric_matrix(A))

    def test_07_make_large_symmetric_matrix_in_fp64_with_eigenvalues_from_distribution(self):
        # Set matrix properties
        M = 1000
        dtype = np.float64

        # Generate a distribution of eigenvalues
        eigenvalues = eigenvalues_from_distribution(size=M, dtype=dtype, seed=self.seed)
        msg.debug("eigenvalues:\n%s\n", eigenvalues)

        # Generate a large random symmetric matrix in fp32
        A = random_symmetric_matrix(
            dim=M,
            dtype=dtype,
            seed=self.seed,
            eigenvalues=eigenvalues,
        )
        msg.debug("A:\n%s\n", A)

        # Check basic matrix properties
        self.assertEqual(A.shape, (M, M))
        self.assertEqual(A.dtype, dtype)
        self.assertTrue(is_square_matrix(A))
        self.assertTrue(is_symmetric_matrix(A))

        # Compute matrix properties
        props_A = SquareSymmetricMatrixProperties(A)
        msg.debug("Matrix properties of A:\n%s\n", props_A)

        # Check spectral matrix properties
        self.assertAlmostEqual(props_A.lambda_min, np.min(eigenvalues), places=6)
        self.assertAlmostEqual(props_A.lambda_max, np.max(eigenvalues), places=6)

    def test_08_make_large_symmetric_matrix_in_fp32_with_eigenvalues_from_distribution(self):
        # Set matrix properties
        M = 1000
        dtype = np.float32

        # Generate a distribution of eigenvalues
        eigenvalues = eigenvalues_from_distribution(size=M, dtype=dtype, seed=self.seed)
        msg.debug("eigenvalues:\n%s\n", eigenvalues)

        # Generate a large random symmetric matrix in fp32
        A = random_symmetric_matrix(
            dim=M,
            dtype=dtype,
            seed=self.seed,
            eigenvalues=eigenvalues,
        )
        msg.debug("A:\n%s\n", A)

        # Check basic matrix properties
        self.assertEqual(A.shape, (M, M))
        self.assertEqual(A.dtype, dtype)
        self.assertTrue(is_square_matrix(A))
        self.assertTrue(is_symmetric_matrix(A))

        # Compute matrix properties
        props_A = SquareSymmetricMatrixProperties(A)
        msg.debug("Matrix properties of A:\n%s\n", props_A)

        # Check spectral matrix properties
        self.assertAlmostEqual(props_A.lambda_min, np.min(eigenvalues), places=5)
        self.assertAlmostEqual(props_A.lambda_max, np.max(eigenvalues), places=5)

    def test_09_make_large_symmetric_matrix_in_fp64_with_rank(self):
        # Set matrix properties
        M = 1000
        rank = 513
        dtype = np.float64

        # Generate a large random symmetric matrix in fp32
        A = random_symmetric_matrix(
            dim=M,
            dtype=dtype,
            seed=self.seed,
            rank=rank,
        )
        msg.debug("A:\n%s\n", A)

        # Check basic matrix properties
        self.assertEqual(A.shape, (M, M))
        self.assertEqual(A.dtype, dtype)
        self.assertTrue(is_square_matrix(A))
        self.assertTrue(is_symmetric_matrix(A))

        # Compute matrix properties
        props_A = SquareSymmetricMatrixProperties(A)
        msg.debug("Matrix properties of A:\n%s\n", props_A)

        # Check spectral matrix properties
        self.assertAlmostEqual(props_A.rank, rank)

    def test_10_make_large_symmetric_matrix_in_fp32_with_rank(self):
        # Set matrix properties
        M = 1000
        rank = 513
        dtype = np.float32

        # Generate a large random symmetric matrix in fp32
        A = random_symmetric_matrix(
            dim=M,
            dtype=dtype,
            seed=self.seed,
            rank=rank,
        )
        msg.debug("A:\n%s\n", A)

        # Check basic matrix properties
        self.assertEqual(A.shape, (M, M))
        self.assertEqual(A.dtype, dtype)
        self.assertTrue(is_square_matrix(A))
        self.assertTrue(is_symmetric_matrix(A))

        # Compute matrix properties
        props_A = SquareSymmetricMatrixProperties(A)
        msg.debug("Matrix properties of A:\n%s\n", props_A)

        # Check spectral matrix properties
        self.assertAlmostEqual(props_A.rank, rank)


class TestLinAlgUtilsRandomMatrixSPD(unittest.TestCase):
    def setUp(self):
        # Configs
        self.seed = 42
        self.default_device = wp.get_device()
        self.verbose = False  # Set to True for verbose output

        # Set debug-level logging to print verbose test output to console
        if self.verbose:
            print("\n")  # Add newline before test output for better readability
            msg.set_log_level(msg.LogLevel.DEBUG)
        else:
            msg.set_log_level(msg.LogLevel.WARNING)

    def tearDown(self):
        self.default_device = None
        if self.verbose:
            msg.reset_log_level()

    def test_01_make_small_random_spd_matrix_in_fp64(self):
        # Generate a small random SPD matrix in fp64
        A = random_spd_matrix(dim=10, dtype=np.float64, seed=self.seed)
        msg.debug("A:\n%s\n", A)

        # Check basic matrix properties
        self.assertEqual(A.shape, (10, 10))
        self.assertEqual(A.dtype, np.float64)
        self.assertTrue(is_square_matrix(A))
        self.assertTrue(is_symmetric_matrix(A))

        # Compute matrix properties
        props_A = SquareSymmetricMatrixProperties(A)
        msg.debug("Matrix properties of A:\n%s\n", props_A)

        # Check spectral matrix properties
        self.assertGreater(props_A.lambda_min, 0.0)
        self.assertGreater(props_A.lambda_max, 0.0)

    def test_02_make_medium_random_spd_matrix_in_fp64(self):
        # Generate a small random SPD matrix in fp64
        A = random_spd_matrix(dim=100, dtype=np.float64, seed=self.seed)
        msg.debug("A:\n%s\n", A)

        # Check basic matrix properties
        self.assertEqual(A.shape, (100, 100))
        self.assertEqual(A.dtype, np.float64)
        self.assertTrue(is_square_matrix(A))
        self.assertTrue(is_symmetric_matrix(A))

        # Compute matrix properties
        props_A = SquareSymmetricMatrixProperties(A)
        msg.debug("Matrix properties of A:\n%s\n", props_A)

        # Check spectral matrix properties
        self.assertGreater(props_A.lambda_min, 0.0)
        self.assertGreater(props_A.lambda_max, 0.0)

    def test_03_make_large_random_spd_matrix_in_fp64(self):
        # Generate a small random SPD matrix in fp64
        A = random_spd_matrix(dim=1000, dtype=np.float64, seed=self.seed)
        msg.debug("A:\n%s\n", A)

        # Check basic matrix properties
        self.assertEqual(A.shape, (1000, 1000))
        self.assertEqual(A.dtype, np.float64)
        self.assertTrue(is_square_matrix(A))
        self.assertTrue(is_symmetric_matrix(A))

        # Compute matrix properties
        props_A = SquareSymmetricMatrixProperties(A)
        msg.debug("Matrix properties of A:\n%s\n", props_A)

        # Check spectral matrix properties
        self.assertGreater(props_A.lambda_min, 0.0)
        self.assertGreater(props_A.lambda_max, 0.0)

    def test_04_make_small_random_spd_matrix_in_fp32(self):
        # Generate a small random SPD matrix in fp32
        A = random_spd_matrix(dim=10, dtype=np.float32, seed=self.seed)
        msg.debug("A:\n%s\n", A)

        # Check basic matrix properties
        self.assertEqual(A.shape, (10, 10))
        self.assertEqual(A.dtype, np.float32)
        self.assertTrue(is_square_matrix(A))
        self.assertTrue(is_symmetric_matrix(A))

        # Compute matrix properties
        props_A = SquareSymmetricMatrixProperties(A)
        msg.debug("Matrix properties of A:\n%s\n", props_A)

        # Check spectral matrix properties
        self.assertGreater(props_A.lambda_min, 0.0)
        self.assertGreater(props_A.lambda_max, 0.0)

    def test_05_make_medium_random_spd_matrix_in_fp32(self):
        # Generate a small random SPD matrix in fp32
        A = random_spd_matrix(dim=100, dtype=np.float32, seed=self.seed)
        msg.debug("A:\n%s\n", A)

        # Check basic matrix properties
        self.assertEqual(A.shape, (100, 100))
        self.assertEqual(A.dtype, np.float32)
        self.assertTrue(is_square_matrix(A))
        self.assertTrue(is_symmetric_matrix(A))

        # Compute matrix properties
        props_A = SquareSymmetricMatrixProperties(A)
        msg.debug("Matrix properties of A:\n%s\n", props_A)

        # Check spectral matrix properties
        self.assertGreater(props_A.lambda_min, 0.0)
        self.assertGreater(props_A.lambda_max, 0.0)

    def test_06_make_large_random_spd_matrix_in_fp32(self):
        # Generate a small random SPD matrix in fp32
        A = random_spd_matrix(dim=1000, dtype=np.float32, seed=self.seed)
        msg.debug("A:\n%s\n", A)

        # Check basic matrix properties
        self.assertEqual(A.shape, (1000, 1000))
        self.assertEqual(A.dtype, np.float32)
        self.assertTrue(is_square_matrix(A))
        self.assertTrue(is_symmetric_matrix(A))

        # Compute matrix properties
        props_A = SquareSymmetricMatrixProperties(A)
        msg.debug("Matrix properties of A:\n%s\n", props_A)

        # Check spectral matrix properties
        self.assertGreater(props_A.lambda_min, 0.0)
        self.assertGreater(props_A.lambda_max, 0.0)


class TestLinAlgUtilsRandomRhsVectors(unittest.TestCase):
    def setUp(self):
        # Configs
        self.seed = 42
        self.default_device = wp.get_device()
        self.verbose = False  # Set to True for verbose output

        # Set debug-level logging to print verbose test output to console
        if self.verbose:
            print("\n")  # Add newline before test output for better readability
            msg.set_log_level(msg.LogLevel.DEBUG)
        else:
            msg.set_log_level(msg.LogLevel.WARNING)

    def tearDown(self):
        self.default_device = None
        if self.verbose:
            msg.reset_log_level()

    def test_01_make_rhs_for_small_random_spd_matrix_in_fp64(self):
        # Generate a small random SPD matrix in fp64
        A = random_spd_matrix(dim=10, dtype=np.float64, seed=self.seed)
        msg.debug("A:\n%s\n", A)

        # Generate a random rhs vector for A
        b, x = random_rhs_for_matrix(A, seed=self.seed, return_source=True)

        # Compute matrix properties
        props_A = SquareSymmetricMatrixProperties(A)
        msg.debug("Matrix properties of A:\n%s\n", props_A)

        # Check basic vector properties
        self.assertEqual(b.shape, (A.shape[0],))
        self.assertEqual(b.dtype, A.dtype)

        # Check norm properties
        norm_b = np.linalg.norm(b, ord=2)
        norm_x = np.linalg.norm(x, ord=2)
        msg.debug("||b||_2 = %f\n", norm_b)
        msg.debug("||x||_2 = %f\n", norm_x)
        msg.debug("||A||_2 = %f\n", props_A.lambda_max)
        self.assertGreater(norm_b, 0.0)
        self.assertGreater(norm_x, 0.0)
        self.assertLessEqual(norm_b, props_A.lambda_max * norm_x)

        # Check that A*x = b
        b_computed = A @ x
        error = np.linalg.norm(b - b_computed, ord=2)
        msg.debug("Error in A*x = b: ||b - A*x||_2 = %e\n", error)
        self.assertAlmostEqual(error, 0.0, places=12)

    def test_02_make_rhs_for_large_random_spd_matrix_in_fp64(self):
        # Generate a small random SPD matrix in fp64
        A = random_spd_matrix(dim=1000, dtype=np.float64, seed=self.seed)
        msg.debug("A:\n%s\n", A)

        # Generate a random rhs vector for A
        b, x = random_rhs_for_matrix(A, seed=self.seed, return_source=True)

        # Compute matrix properties
        props_A = SquareSymmetricMatrixProperties(A)
        msg.debug("Matrix properties of A:\n%s\n", props_A)

        # Check basic vector properties
        self.assertEqual(b.shape, (A.shape[0],))
        self.assertEqual(b.dtype, A.dtype)

        # Check norm properties
        norm_b = np.linalg.norm(b, ord=2)
        norm_x = np.linalg.norm(x, ord=2)
        msg.debug("||b||_2 = %f\n", norm_b)
        msg.debug("||x||_2 = %f\n", norm_x)
        msg.debug("||A||_2 = %f\n", props_A.lambda_max)
        self.assertGreater(norm_b, 0.0)
        self.assertGreater(norm_x, 0.0)
        self.assertLessEqual(norm_b, props_A.lambda_max * norm_x)

        # Check that A*x = b
        b_computed = A @ x
        error = np.linalg.norm(b - b_computed, ord=2)
        msg.debug("Error in A*x = b: ||b - A*x||_2 = %e\n", error)
        self.assertAlmostEqual(error, 0.0, places=12)

    def test_03_make_rhs_for_small_random_spd_matrix_in_fp32(self):
        # Generate a small random SPD matrix in fp32
        A = random_spd_matrix(dim=10, dtype=np.float32, seed=self.seed)
        msg.debug("A:\n%s\n", A)

        # Generate a random rhs vector for A
        b, x = random_rhs_for_matrix(A, seed=self.seed, return_source=True)

        # Compute matrix properties
        props_A = SquareSymmetricMatrixProperties(A)
        msg.debug("Matrix properties of A:\n%s\n", props_A)

        # Check basic vector properties
        self.assertEqual(b.shape, (A.shape[0],))
        self.assertEqual(b.dtype, A.dtype)

        # Check norm properties
        norm_b = np.linalg.norm(b, ord=2)
        norm_x = np.linalg.norm(x, ord=2)
        msg.debug("||b||_2 = %f\n", norm_b)
        msg.debug("||x||_2 = %f\n", norm_x)
        msg.debug("||A||_2 = %f\n", props_A.lambda_max)
        self.assertGreater(norm_b, 0.0)
        self.assertGreater(norm_x, 0.0)
        self.assertLessEqual(norm_b, props_A.lambda_max * norm_x)

        # Check that A*x = b
        b_computed = A @ x
        error = np.linalg.norm(b - b_computed, ord=2)
        msg.debug("Error in A*x = b: ||b - A*x||_2 = %e\n", error)
        self.assertAlmostEqual(error, 0.0, places=12)

    def test_02_make_rhs_for_large_random_spd_matrix_in_fp32(self):
        # Generate a small random SPD matrix in fp32
        A = random_spd_matrix(dim=1000, dtype=np.float32, seed=self.seed)
        msg.debug("A:\n%s\n", A)

        # Generate a random rhs vector for A
        b, x = random_rhs_for_matrix(A, seed=self.seed, return_source=True)

        # Compute matrix properties
        props_A = SquareSymmetricMatrixProperties(A)
        msg.debug("Matrix properties of A:\n%s\n", props_A)

        # Check basic vector properties
        self.assertEqual(b.shape, (A.shape[0],))
        self.assertEqual(b.dtype, A.dtype)

        # Check norm properties
        norm_b = np.linalg.norm(b, ord=2)
        norm_x = np.linalg.norm(x, ord=2)
        msg.debug("||b||_2 = %f\n", norm_b)
        msg.debug("||x||_2 = %f\n", norm_x)
        msg.debug("||A||_2 = %f\n", props_A.lambda_max)
        self.assertGreater(norm_b, 0.0)
        self.assertGreater(norm_x, 0.0)
        self.assertLessEqual(norm_b, props_A.lambda_max * norm_x)

        # Check that A*x = b
        b_computed = A @ x
        error = np.linalg.norm(b - b_computed, ord=2)
        msg.debug("Error in A*x = b: ||b - A*x||_2 = %e\n", error)
        self.assertAlmostEqual(error, 0.0, places=12)


###
# Test execution
###

if __name__ == "__main__":
    # Global numpy configurations
    np.set_printoptions(linewidth=500, precision=10, suppress=True)  # Suppress scientific notation

    # Initialize Warp
    wp.init()

    # Global warp configurations
    wp.config.enable_backward = False
    wp.config.verbose = False
    wp.clear_kernel_cache()
    wp.clear_lto_cache()

    # Run all tests
    unittest.main(verbosity=2)
