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

"""
KAMINO: UNIT TESTS: RANDOM DATA GENERATION
"""

import unittest

import numpy as np

# Module to be tested
import newton._src.solvers.kamino.tests.utils.random as rand

###
# Tests
###


class TestRandomSymmetricMatrix(unittest.TestCase):
    def setUp(self):
        self.verbose = False  # Set to True for verbose output

    def test_matrix_symmetry(self):
        dim = 5
        A = rand.random_symmetric_matrix(dim=dim)

        # Verify symmetry: A should equal its transpose
        np.testing.assert_array_equal(A, A.T, "Matrix is not symmetric.")

    def test_matrix_rank(self):
        dim = 5
        rank = 3
        A = rand.random_symmetric_matrix(dim=dim, rank=rank)
        # Verify the rank: The rank should be equal to or less than the specified rank
        actual_rank = np.linalg.matrix_rank(A)
        self.assertEqual(actual_rank, rank, f"Matrix rank is {actual_rank}, expected {rank}.")

    def test_matrix_eigenvalues(self):
        dim = 5
        eigenvalues = [1, 2, 3, 4, 5]  # Expected eigenvalues
        A = rand.random_symmetric_matrix(dim=dim, eigenvalues=eigenvalues)

        # Compute eigenvalues of the generated matrix
        actual_eigenvalues = np.linalg.eigvals(A)

        # Check if the eigenvalues are close to the expected ones
        np.testing.assert_allclose(
            sorted(actual_eigenvalues), sorted(eigenvalues), rtol=1e-5, err_msg="Eigenvalues do not match."
        )

    def test_invalid_eigenvalues(self):
        dim = 5
        eigenvalues = [1, 2, 3]  # Fewer eigenvalues than matrix dimension
        with self.assertRaises(ValueError):
            rand.random_symmetric_matrix(dim=dim, eigenvalues=eigenvalues)

    def test_invalid_rank(self):
        dim = 5
        rank = 6  # Rank is greater than the dimension
        with self.assertRaises(ValueError):
            rand.random_symmetric_matrix(dim=dim, rank=rank)


class TestRandomProblemCholesky(unittest.TestCase):
    def setUp(self):
        self.verbose = False  # Set to True for verbose output

    def test_generate_small_lower(self):
        dim = 10
        problem = rand.RandomProblemLLT(dims=[dim], seed=42, upper=False)
        A, b = problem.A_np[0], problem.b_np[0]

        # Verify the shapes of A and b
        self.assertEqual(A.shape, (dim, dim), "Matrix A has incorrect shape.")
        self.assertEqual(b.shape, (dim,), "Vector b has incorrect shape.")

    def test_generate_small_upper(self):
        dim = 10
        problem = rand.RandomProblemLLT(dims=[dim], seed=42, upper=True)
        A, b = problem.A_np[0], problem.b_np[0]

        # Verify the shapes of A and b
        self.assertEqual(A.shape, (dim, dim), "Matrix A has incorrect shape.")
        self.assertEqual(b.shape, (dim,), "Vector b has incorrect shape.")


class TestRandomProblemLDLT(unittest.TestCase):
    def setUp(self):
        self.verbose = False  # Set to True for verbose output

    def test_generate_small_lower(self):
        dim = 10
        problem = rand.RandomProblemLDLT(dims=[dim], seed=42, lower=True)
        A, b = problem.A_np[0], problem.b_np[0]

        # Verify the shapes of A and b
        self.assertEqual(A.shape, (dim, dim), "Matrix A has incorrect shape.")
        self.assertEqual(b.shape, (dim,), "Vector b has incorrect shape.")

    def test_generate_small_upper(self):
        dim = 10
        problem = rand.RandomProblemLDLT(dims=[dim], seed=42, lower=False)
        A, b = problem.A_np[0], problem.b_np[0]

        # Verify the shapes of A and b
        self.assertEqual(A.shape, (dim, dim), "Matrix A has incorrect shape.")
        self.assertEqual(b.shape, (dim,), "Vector b has incorrect shape.")


###
# Test execution
###

if __name__ == "__main__":
    # Global numpy configurations
    np.set_printoptions(linewidth=10000, precision=10, threshold=10000, suppress=True)  # Suppress scientific notation

    # Run all tests
    unittest.main(verbosity=2)
