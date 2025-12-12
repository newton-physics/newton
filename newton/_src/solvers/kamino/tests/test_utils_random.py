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
Unit tests for random matrix and problem generation utilities in `linalg/utils/random.py`.
"""

import unittest

import numpy as np

import newton._src.solvers.kamino.tests.utils.rand as rand

# Test utilities
from newton._src.solvers.kamino.tests import setup_tests, test_context

###
# Tests
###


class TestRandomSymmetricMatrix(unittest.TestCase):
    def setUp(self):
        if not test_context.setup_done:
            setup_tests(clear_cache=False)
        self.verbose = test_context.verbose  # Set to True for verbose output

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
        if not test_context.setup_done:
            setup_tests(clear_cache=False)
        self.verbose = test_context.verbose  # Set to True for verbose output

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
        if not test_context.setup_done:
            setup_tests(clear_cache=False)
        self.verbose = test_context.verbose  # Set to True for verbose output

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
    # Test setup
    setup_tests()

    # Run all tests
    unittest.main(verbosity=2)
