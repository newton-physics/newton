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
KAMINO: UNIT TESTS: DYNAMICS: DUAL PROBLEM
"""

import unittest

import numpy as np
import warp as wp

from newton._src.solvers.kamino.dynamics.dual import DualProblem
from newton._src.solvers.kamino.linalg import LLTSequentialSolver
from newton._src.solvers.kamino.models.builders.basics import make_basics_heterogeneous_builder
from newton._src.solvers.kamino.tests import setup_tests, test_context
from newton._src.solvers.kamino.tests.utils.extract import extract_problem_vector
from newton._src.solvers.kamino.tests.utils.make import make_containers, update_containers
from newton._src.solvers.kamino.tests.utils.print import print_model_info

###
# Tests
###


class TestDualProblem(unittest.TestCase):
    def setUp(self):
        if not test_context.setup_done:
            setup_tests(clear_cache=False)
        self.verbose = test_context.verbose  # Set to True for detailed output
        self.default_device = wp.get_device(test_context.device)

    def tearDown(self):
        self.default_device = None

    def test_01_allocate_dual_problem(self):
        """
        Tests the allocation of a DualProblem data members.
        """
        # Model constants
        max_world_contacts = 12

        # Construct the model description using model builders for different systems
        builder = make_basics_heterogeneous_builder()

        # Create the model and containers from the builder
        model, data, limits, detector, _jacobians = make_containers(
            builder=builder, max_world_contacts=max_world_contacts, device=self.default_device
        )

        # Create the Delassus operator
        problem = DualProblem(
            model=model,
            data=data,
            limits=limits,
            contacts=detector.contacts,
            solver=LLTSequentialSolver,
            device=self.default_device,
        )

        # Optional verbose output
        if self.verbose:
            print(f"problem.data.config: {problem.data.config}")
            print(f"problem.data.maxdim: {problem.data.maxdim}")
            print(f"problem.data.dim: {problem.data.dim}")
            print(f"problem.data.mio: {problem.data.mio}")
            print(f"problem.data.vio: {problem.data.vio}")
            print(f"problem.data.u_f (shape): {problem.data.u_f.shape}")
            print(f"problem.data.v_b (shape): {problem.data.v_b.shape}")
            print(f"problem.data.v_i (shape): {problem.data.v_i.shape}")
            print(f"problem.data.v_f (shape): {problem.data.v_f.shape}")
            print(f"problem.data.mu (shape): {problem.data.mu.shape}")
            print(f"problem.data.D (shape): {problem.data.D.shape}")

        # Extract expected allocation sizes
        nw = model.info.num_worlds
        nb = model.size.sum_of_num_bodies
        maxnl = limits.model_max_limits_host
        maxnc = detector.contacts.model_max_contacts_host
        maxdims = model.size.sum_of_num_joint_cts + maxnl + 3 * maxnc

        # Check allocations
        self.assertEqual(problem.data.config.size, nw)
        self.assertEqual(problem.data.maxdim.size, nw)
        self.assertEqual(problem.data.dim.size, nw)
        self.assertEqual(problem.data.mio.size, nw)
        self.assertEqual(problem.data.vio.size, nw)
        self.assertEqual(problem.data.u_f.size, nb)
        self.assertEqual(problem.data.v_b.size, maxdims)
        self.assertEqual(problem.data.v_i.size, maxdims)
        self.assertEqual(problem.data.v_f.size, maxdims)
        self.assertEqual(problem.data.mu.size, maxnc)
        maxdim_np = problem.data.maxdim.numpy()
        self.assertEqual(np.sum(maxdim_np), maxdims)
        dim_np = problem.data.dim.numpy()
        self.assertEqual(np.sum(dim_np), 0)

    def test_02_dual_problem_build(self):
        """
        Tests building the dual problem from time-varying data.
        """
        # Model constants
        max_world_contacts = 12

        # Construct the model description using model builders for different systems
        builder = make_basics_heterogeneous_builder()
        num_worlds = builder.num_worlds

        # Create the model and containers from the builder
        model, data, limits, detector, jacobians = make_containers(
            builder=builder, max_world_contacts=max_world_contacts, device=self.default_device
        )

        # Update the containers
        update_containers(model=model, data=data, limits=limits, detector=detector, jacobians=jacobians)
        if self.verbose:
            print_model_info(model)

        # Create the Delassus operator
        problem = DualProblem(
            model=model,
            data=data,
            limits=limits,
            contacts=detector.contacts,
            solver=LLTSequentialSolver,
            device=self.default_device,
        )

        # Build the dual problem
        problem.build(
            model=model, data=data, limits=limits.data, contacts=detector.contacts.data, jacobians=jacobians.data
        )

        # Extract numpy arrays from the problem data
        v_b_wp_np = problem.data.v_b.numpy()
        v_i_wp_np = problem.data.v_i.numpy()
        v_f_wp_np = problem.data.v_f.numpy()

        # Extract free-velocity and solution vectors lists of numpy arrays
        v_b_np = extract_problem_vector(problem.delassus, vector=v_b_wp_np, only_active_dims=True)
        v_i_np = extract_problem_vector(problem.delassus, vector=v_i_wp_np, only_active_dims=True)
        v_f_np = extract_problem_vector(problem.delassus, vector=v_f_wp_np, only_active_dims=True)

        # Optional verbose output
        if self.verbose:
            print("")  # Print a blank line for better readability
            print(f"problem.data.maxdim: {problem.data.maxdim}")
            print(f"problem.data.dim: {problem.data.dim}")
            print(f"problem.data.mio: {problem.data.mio}")
            print(f"problem.data.vio: {problem.data.vio}")
            print(f"problem.data.D: {problem.data.D.shape}")
            print(f"problem.data.u_f:\n{problem.data.u_f}")
            print(f"problem.data.v_b:\n{problem.data.v_b}")
            print(f"problem.data.v_i:\n{problem.data.v_i}")
            print(f"problem.data.v_f:\n{problem.data.v_f}")
            print(f"problem.data.mu:\n{problem.data.mu}")
            for w in range(num_worlds):
                print(f"problem.data.v_b[{w}]:\n{v_b_np[w]}")
            for w in range(num_worlds):
                print(f"problem.data.v_i[{w}]:\n{v_i_np[w]}")
            for w in range(num_worlds):
                print(f"problem.data.v_f[{w}]:\n{v_f_np[w]}")

        # Check the problem data
        # TODO


###
# Test execution
###

if __name__ == "__main__":
    # Test setup
    setup_tests()

    # Run all tests
    unittest.main(verbosity=2)
