# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""
KAMINO: UNIT TESTS: DYNAMICS: DUAL PROBLEM
"""

import logging
import unittest

import numpy as np
import warp as wp

from newton._src.solvers.kamino._src.core.model import ModelKamino
from newton._src.solvers.kamino._src.dynamics.dual import DualProblem
from newton._src.solvers.kamino._src.linalg import ConjugateGradientSolver
from newton._src.solvers.kamino._src.utils.logger import reset_log_level, set_log_level
from newton.tests.kamino import setup_tests, test_context
from newton.tests.kamino.utils.extract import extract_problem_vector
from newton.tests.kamino.utils.make import make_containers, update_containers
from newton.tests.kamino.utils.print import debug_print_model_info
from newton.tests.utils.basics import make_basics_heterogeneous_builder

logger = logging.getLogger(__name__)

###
# Tests
###


class TestDualProblem(unittest.TestCase):
    def setUp(self):
        if not test_context.setup_done:
            setup_tests(clear_cache=False)
        self.verbose = test_context.verbose  # Set to True for detailed output
        self.default_device = wp.get_device(test_context.device)

        # Set debug-level logging to print verbose test output to console
        if self.verbose:
            set_log_level(logging.DEBUG)
            logger.debug("\n")  # Add newline before test output for better readability
        else:
            reset_log_level()

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
        model = ModelKamino.from_newton(builder.finalize(device=self.default_device))

        # Create the model and containers from the builder
        model, data, _state, limits, detector, jacobians = make_containers(
            model=model, max_world_contacts=max_world_contacts
        )

        # Create the Delassus operator
        problem = DualProblem(
            model=model,
            data=data,
            limits=limits,
            contacts=detector.contacts,
            jacobians=jacobians,
            solver=ConjugateGradientSolver,
        )
        # Optional verbose output
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug("\n")  # Print a blank line for better readability
            logger.debug("problem.data.config: %s", problem.data.config)
            logger.debug("problem.data.maxdim: %s", problem.data.maxdim)
            logger.debug("problem.data.dim: %s", problem.data.dim)
            logger.debug("problem.data.mio: %s", problem.data.mio)
            logger.debug("problem.data.vio: %s", problem.data.vio)
            logger.debug("problem.data.u_f (shape): %s", problem.data.u_f.shape)
            logger.debug("problem.data.v_b (shape): %s", problem.data.v_b.shape)
            logger.debug("problem.data.v_i (shape): %s", problem.data.v_i.shape)
            logger.debug("problem.data.v_f (shape): %s", problem.data.v_f.shape)
            logger.debug("problem.data.mu (shape): %s", problem.data.mu.shape)
            logger.debug("problem.data.bound_lower (shape): %s", problem.data.bound_lower.shape)
            logger.debug("problem.data.bound_upper (shape): %s", problem.data.bound_upper.shape)
            logger.debug("problem.data.D: %s", problem.data.D)

        # Extract expected allocation sizes
        nw = model.info.num_worlds
        nb = model.size.sum_of_num_bodies
        maxnl = limits.model_max_limits_host
        maxnc = detector.contacts.model_max_contacts_host
        maxdims = (
            model.size.sum_of_num_bilateral_joint_cts + model.size.sum_of_num_bounded_joint_cts + maxnl + 3 * maxnc
        )

        # Check allocations
        self.assertEqual(problem.data.config.size, nw)
        self.assertEqual(problem.data.maxdim.size, nw)
        self.assertEqual(problem.data.dim.size, nw)
        if not problem.sparse:
            self.assertEqual(problem.data.mio.size, nw)
        self.assertEqual(problem.data.vio.size, nw)
        self.assertEqual(problem.data.u_f.size, nb)
        self.assertEqual(problem.data.v_b.size, maxdims)
        self.assertEqual(problem.data.v_i.size, maxdims)
        self.assertEqual(problem.data.v_f.size, maxdims)
        self.assertEqual(problem.data.mu.size, maxnc)
        self.assertEqual(problem.data.bound_lower.size, model.size.sum_of_num_bounded_joint_cts)
        self.assertEqual(problem.data.bound_upper.size, model.size.sum_of_num_bounded_joint_cts)
        maxdim_np = problem.data.maxdim.numpy()
        self.assertEqual(int(np.sum(maxdim_np)), maxdims)
        dim_np = problem.data.dim.numpy()
        self.assertEqual(int(np.sum(dim_np)), 46)  # Total number of active constraints in the default state

    def test_02_dual_problem_build(self):
        """
        Tests building the dual problem from time-varying data.
        """
        # Model constants
        max_world_contacts = 12

        # Construct the model description using model builders for different systems
        builder = make_basics_heterogeneous_builder()
        model = ModelKamino.from_newton(builder.finalize(device=self.default_device))
        num_worlds = model.info.num_worlds

        # Create the model and containers from the builder
        model, data, state, limits, detector, jacobians = make_containers(
            model=model, max_world_contacts=max_world_contacts
        )

        # Update the containers
        update_containers(model=model, data=data, state=state, limits=limits, detector=detector, jacobians=jacobians)
        debug_print_model_info(model)

        # Create the Delassus operator
        problem = DualProblem(
            model=model,
            data=data,
            limits=limits,
            contacts=detector.contacts,
            jacobians=jacobians,
            solver=ConjugateGradientSolver,
        )

        # Build the dual problem
        problem.build(model=model, data=data, limits=limits, contacts=detector.contacts, jacobians=jacobians)

        # Extract numpy arrays from the problem data
        v_b_wp_np = problem.data.v_b.numpy()
        v_i_wp_np = problem.data.v_i.numpy()
        v_f_wp_np = problem.data.v_f.numpy()

        # Extract free-velocity and solution vectors lists of numpy arrays
        v_b_np = extract_problem_vector(problem.delassus, vector=v_b_wp_np, only_active_dims=True)
        v_i_np = extract_problem_vector(problem.delassus, vector=v_i_wp_np, only_active_dims=True)
        v_f_np = extract_problem_vector(problem.delassus, vector=v_f_wp_np, only_active_dims=True)

        # Optional verbose output
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug("")  # Print a blank line for better readability
            logger.debug("problem.data.maxdim: %s", problem.data.maxdim)
            logger.debug("problem.data.dim: %s", problem.data.dim)
            logger.debug("problem.data.mio: %s", problem.data.mio)
            logger.debug("problem.data.vio: %s", problem.data.vio)
            logger.debug("problem.data.D: %s", problem.data.D)
            logger.debug("problem.data.u_f:\n%s", problem.data.u_f)
            logger.debug("problem.data.v_b:\n%s", problem.data.v_b)
            logger.debug("problem.data.v_i:\n%s", problem.data.v_i)
            logger.debug("problem.data.v_f:\n%s", problem.data.v_f)
            logger.debug("problem.data.mu:\n%s", problem.data.mu)
            logger.debug("problem.data.bound_lower:\n%s", problem.data.bound_lower)
            logger.debug("problem.data.bound_upper:\n%s", problem.data.bound_upper)
            for w in range(num_worlds):
                logger.debug("problem.data.v_b[%s]:\n%s", w, v_b_np[w])
            for w in range(num_worlds):
                logger.debug("problem.data.v_i[%s]:\n%s", w, v_i_np[w])
            for w in range(num_worlds):
                logger.debug("problem.data.v_f[%s]:\n%s", w, v_f_np[w])

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
