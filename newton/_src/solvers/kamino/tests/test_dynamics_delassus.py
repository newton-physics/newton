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

"""Unit tests for the DelassusOperator class"""

import unittest

import numpy as np
import warp as wp

from newton._src.solvers.kamino.core.model import Model
from newton._src.solvers.kamino.dynamics.delassus import DelassusOperator
from newton._src.solvers.kamino.geometry.contacts import Contacts
from newton._src.solvers.kamino.kinematics.constraints import max_constraints_per_world
from newton._src.solvers.kamino.kinematics.limits import Limits
from newton._src.solvers.kamino.linalg import LLTSequentialSolver
from newton._src.solvers.kamino.models.builders.basics import (
    build_boxes_fourbar,
    build_boxes_nunchaku,
    make_basics_heterogeneous_builder,
)
from newton._src.solvers.kamino.models.builders.utils import make_homogeneous_builder
from newton._src.solvers.kamino.tests import setup_tests, test_context
from newton._src.solvers.kamino.tests.utils.extract import (
    extract_active_constraint_dims,
    extract_cts_jacobians,
    extract_delassus,
    extract_problem_vector,
)
from newton._src.solvers.kamino.tests.utils.make import (
    make_containers,
    make_inverse_generalized_mass_matrices,
    update_containers,
)
from newton._src.solvers.kamino.tests.utils.print import print_error_stats
from newton._src.solvers.kamino.tests.utils.rand import random_rhs_for_matrix

###
# Helper functions
###


def check_delassus_allocations(
    fixture: unittest.TestCase,
    model: Model,
    limits: Limits,
    contacts: Contacts,
    delassus: DelassusOperator,
) -> None:
    # Compute expected and allocated dimensions and sizes
    expected_max_constraint_dims = max_constraints_per_world(model, limits, contacts)
    num_worlds = len(expected_max_constraint_dims)
    expected_D_sizes = [expected_max_constraint_dims[i] * expected_max_constraint_dims[i] for i in range(num_worlds)]
    delassus_maxdim_np = delassus.info.maxdim.numpy()
    fixture.assertEqual(
        len(delassus_maxdim_np), num_worlds, "Number of Delassus operator blocks does not match the number of worlds"
    )
    D_maxdims = [int(delassus_maxdim_np[i]) for i in range(num_worlds)]
    D_sizes = [D_maxdims[i] * D_maxdims[i] for i in range(num_worlds)]
    D_sizes_sum = sum(D_sizes)

    for i in range(num_worlds):
        fixture.assertEqual(
            D_maxdims[i],
            expected_max_constraint_dims[i],
            f"Delassus operator block {i} maxdim does not match expected maximum constraint dimension",
        )
        fixture.assertEqual(
            D_sizes[i], expected_D_sizes[i], f"Delassus operator block {i} max size does not match expected max size"
        )

    # Check Delassus operator data sizes
    fixture.assertEqual(delassus.info.maxdim.size, num_worlds)
    fixture.assertEqual(delassus.info.dim.size, num_worlds)
    fixture.assertEqual(delassus.info.mio.size, num_worlds)
    fixture.assertEqual(delassus.info.vio.size, num_worlds)
    fixture.assertEqual(delassus.D.size, D_sizes_sum)

    # Check if the factorizer info data to the same as the Delassus info data
    fixture.assertEqual(delassus.info.maxdim.ptr, delassus.solver.operator.info.maxdim.ptr)
    fixture.assertEqual(delassus.info.dim.ptr, delassus.solver.operator.info.dim.ptr)
    fixture.assertEqual(delassus.info.mio.ptr, delassus.solver.operator.info.mio.ptr)
    fixture.assertEqual(delassus.info.vio.ptr, delassus.solver.operator.info.vio.ptr)


def print_delassus_info(delassus: DelassusOperator) -> None:
    print(f"delassus.info.maxdim: {delassus.info.maxdim}")
    print(f"delassus.info.dim: {delassus.info.dim}")
    print(f"delassus.info.mio: {delassus.info.mio}")
    print(f"delassus.info.vio: {delassus.info.vio}")
    print(f"delassus.D: {delassus.D.shape}")


###
# Tests
###


class TestDelassusOperator(unittest.TestCase):
    def setUp(self):
        if not test_context.setup_done:
            setup_tests(clear_cache=False)
        self.verbose = test_context.verbose  # Set to True for detailed output
        self.default_device = wp.get_device(test_context.device)

    def tearDown(self):
        self.default_device = None

    def test_01_allocate_single_delassus_operator(self):
        # Model constants
        max_world_contacts = 12

        # Construct the model description using model builders for different systems
        builder = build_boxes_nunchaku()

        # Create the model and containers from the builder
        model, data, limits, detector, jacobians = make_containers(
            builder=builder, max_world_contacts=max_world_contacts, device=self.default_device
        )

        # Update the containers
        update_containers(model=model, data=data, limits=limits, detector=detector, jacobians=jacobians)

        # Create the Delassus operator
        delassus = DelassusOperator(
            model=model,
            data=data,
            limits=limits,
            contacts=detector.contacts,
            device=self.default_device,
            solver=LLTSequentialSolver,
        )

        # Compare expected to allocated dimensions and sizes
        check_delassus_allocations(self, model, limits, detector.contacts, delassus)

        # Optional verbose output
        if self.verbose:
            print("")  # Print a newline for better readability
            print_delassus_info(delassus)

    def test_02_allocate_homogeneous_delassus_operator(self):
        # Model constants
        num_worlds = 3
        max_world_contacts = 12

        # Construct a homogeneous model description using model builders
        builder = make_homogeneous_builder(num_worlds=num_worlds, build_fn=build_boxes_nunchaku)

        # Create the model and containers from the builder
        model, data, limits, detector, jacobians = make_containers(
            builder=builder, max_world_contacts=max_world_contacts, device=self.default_device
        )

        # Update the containers
        update_containers(model=model, data=data, limits=limits, detector=detector, jacobians=jacobians)

        # Create the Delassus operator
        delassus = DelassusOperator(
            model=model,
            data=data,
            limits=limits,
            contacts=detector.contacts,
            device=self.default_device,
            solver=LLTSequentialSolver,
        )

        # Compare expected to allocated dimensions and sizes
        check_delassus_allocations(self, model, limits, detector.contacts, delassus)

        # Optional verbose output
        if self.verbose:
            print("")  # Print a newline for better readability
            print_delassus_info(delassus)

    def test_03_allocate_heterogeneous_delassus_operator(self):
        # Model constants
        max_world_contacts = 12

        # Create a heterogeneous model description using model builders
        builder = make_basics_heterogeneous_builder()

        # Create the model and containers from the builder
        model, data, limits, detector, jacobians = make_containers(
            builder=builder, max_world_contacts=max_world_contacts, device=self.default_device
        )

        # Update the containers
        update_containers(model=model, data=data, limits=limits, detector=detector, jacobians=jacobians)

        # Create the Delassus operator
        delassus = DelassusOperator(
            model=model,
            data=data,
            limits=limits,
            contacts=detector.contacts,
            device=self.default_device,
            solver=LLTSequentialSolver,
        )

        # Compare expected to allocated dimensions and sizes
        check_delassus_allocations(self, model, limits, detector.contacts, delassus)

        # Optional verbose output
        if self.verbose:
            print("")  # Print a newline for better readability
            print_delassus_info(delassus)

    def test_04_build_delassus_operator(self):
        # Model constants
        max_world_contacts = 12

        # Construct the model description using model builders for different systems
        # builder = build_boxes_hinged(z_offset=0.0, ground=False)
        builder = build_boxes_fourbar(z_offset=0.0, ground=False)
        num_bodies = [builder.num_bodies]

        # Create the model and containers from the builder
        model, data, limits, detector, jacobians = make_containers(
            builder=builder, max_world_contacts=max_world_contacts, device=self.default_device
        )

        # Update the containers
        update_containers(model=model, data=data, limits=limits, detector=detector, jacobians=jacobians)

        # Create the Delassus operator
        delassus = DelassusOperator(
            model=model,
            data=data,
            limits=limits,
            contacts=detector.contacts,
            device=self.default_device,
            solver=LLTSequentialSolver,
        )

        # Build the Delassus operator from the current data
        delassus.build(model=model, data=data, jacobians=jacobians.data, reset_to_zero=True)

        # Extract the active constraint dimensions
        active_dims = extract_active_constraint_dims(delassus)
        active_size = [dims * dims for dims in active_dims]

        # Extract Jacobians as numpy arrays
        J_cts_np = extract_cts_jacobians(jacobians=jacobians, num_bodies=num_bodies, active_dims=active_dims)

        # Extract Delassus data as numpy arrays
        D_np = extract_delassus(delassus, only_active_dims=True)

        # Construct a list of generalized inverse mass matrices of each world
        invM_np = make_inverse_generalized_mass_matrices(model, data)

        # For each world, compute the Delassus matrix using numpy and
        # compare it with the one from the Delassus operator class
        for w in range(delassus.num_worlds):
            # Compute the Delassus matrix using the inverse mass matrix and the Jacobian
            D_w = J_cts_np[w] @ invM_np[w] @ J_cts_np[w].T

            # Compare the computed Delassus matrix with the one from the dual problem
            is_D_close = np.allclose(D_np[w], D_w, rtol=1e-3, atol=1e-4)
            if not is_D_close or self.verbose:
                print(f"[{w}]: D_w (shape={D_w.shape}):\n{D_w}")
                print(f"[{w}]: D_np (shape={D_np[w].shape}):\n{D_np[w]}")
                print_error_stats(f"D[{w}]", D_np[w], D_w, active_size[w], show_errors=True)
            self.assertTrue(is_D_close)

    def test_05_build_homogeneous_delassus_operator(self):
        # Model constants
        num_worlds = 3
        max_world_contacts = 12

        # Construct a homogeneous model description using model builders
        builder = make_homogeneous_builder(num_worlds=num_worlds, build_fn=build_boxes_nunchaku)

        # Create a list of number of bodies per world
        num_bodies = [builder.num_bodies // num_worlds for _ in range(num_worlds)]

        # Create the model and containers from the builder
        model, data, limits, detector, jacobians = make_containers(
            builder=builder, max_world_contacts=max_world_contacts, device=self.default_device
        )

        # Update the containers
        update_containers(model=model, data=data, limits=limits, detector=detector, jacobians=jacobians)

        # Create the Delassus operator
        delassus = DelassusOperator(
            model=model,
            data=data,
            limits=limits,
            contacts=detector.contacts,
            device=self.default_device,
            solver=LLTSequentialSolver,
        )

        # Build the Delassus operator from the current data
        delassus.build(model=model, data=data, jacobians=jacobians.data, reset_to_zero=True)

        # Extract the active constraint dimensions
        active_dims = extract_active_constraint_dims(delassus)

        # Extract Jacobians as numpy arrays
        J_cts_np = extract_cts_jacobians(jacobians=jacobians, num_bodies=num_bodies, active_dims=active_dims)

        # Extract Delassus data as numpy arrays
        D_np = extract_delassus(delassus, only_active_dims=True)

        # Construct a list of generalized inverse mass matrices of each world
        invM_np = make_inverse_generalized_mass_matrices(model, data)

        # Optional verbose output
        if self.verbose:
            print("")  # Print a newline for better readability
            for i in range(len(active_dims)):
                print(f"[{i}]: active_dims: {active_dims[i]}")
            for i in range(len(J_cts_np)):
                print(f"[{i}]: J_cts_np (shape={J_cts_np[i].shape}):\n{J_cts_np[i]}")
            for i in range(len(D_np)):
                print(f"[{i}]: D_np (shape={D_np[i].shape}):\n{D_np[i]}")
            for i in range(len(invM_np)):
                print(f"[{i}]: invM_np (shape={invM_np[i].shape}):\n{invM_np[i]}")
            print("")  # Add a newline for better readability
            print_delassus_info(delassus)
            print("")  # Add a newline for better readability

        # For each world, compute the Delassus matrix using numpy and
        # compare it with the one from the Delassus operator class
        for w in range(delassus.num_worlds):
            # Compute the Delassus matrix using the inverse mass matrix and the Jacobian
            D_w = (J_cts_np[w] @ invM_np[w]) @ J_cts_np[w].T

            # Compare the computed Delassus matrix with the one from the dual problem
            is_D_close = np.allclose(D_np[w], D_w, atol=1e-3, rtol=1e-4)
            if not is_D_close or self.verbose:
                print(f"[{w}]: D_w (shape={D_w.shape}):\n{D_w}")
                print(f"[{w}]: D_np (shape={D_np[w].shape}):\n{D_np[w]}")
                print_error_stats(f"D[{w}]", D_np[w], D_w, active_dims[w])
            self.assertTrue(is_D_close)

    def test_06_build_heterogeneous_delassus_operator(self):
        # Model constants
        max_world_contacts = 12

        # Create a heterogeneous model description using model builders
        builder = make_basics_heterogeneous_builder()
        num_bodies = [world.num_bodies for world in builder.worlds]

        # Create the model and containers from the builder
        model, data, limits, detector, jacobians = make_containers(
            builder=builder, max_world_contacts=max_world_contacts, device=self.default_device
        )

        # Update the containers
        update_containers(model=model, data=data, limits=limits, detector=detector, jacobians=jacobians)

        # Create the Delassus operator
        delassus = DelassusOperator(
            model=model,
            data=data,
            limits=limits,
            contacts=detector.contacts,
            device=self.default_device,
            solver=LLTSequentialSolver,
        )

        # Build the Delassus operator from the current data
        delassus.build(model=model, data=data, jacobians=jacobians.data, reset_to_zero=True)

        # Extract the active constraint dimensions
        active_dims = extract_active_constraint_dims(delassus)

        # Extract Jacobians as numpy arrays
        J_cts_np = extract_cts_jacobians(jacobians=jacobians, num_bodies=num_bodies, active_dims=active_dims)

        # Extract Delassus data as numpy arrays
        D_np = extract_delassus(delassus, only_active_dims=True)

        # Construct a list of generalized inverse mass matrices of each world
        invM_np = make_inverse_generalized_mass_matrices(model, data)

        # Optional verbose output
        if self.verbose:
            print("")  # Print a newline for better readability
            for i in range(len(active_dims)):
                print(f"[{i}]: active_dims: {active_dims[i]}")
            for i in range(len(J_cts_np)):
                print(f"[{i}]: J_cts_np (shape={J_cts_np[i].shape}):\n{J_cts_np[i]}")
            for i in range(len(D_np)):
                print(f"[{i}]: D_np (shape={D_np[i].shape}):\n{D_np[i]}")
            for i in range(len(invM_np)):
                print(f"[{i}]: invM_np (shape={invM_np[i].shape}):\n{invM_np[i]}")
            print("")  # Add a newline for better readability
            print_delassus_info(delassus)
            print("")  # Add a newline for better readability

        # For each world, compute the Delassus matrix using numpy and
        # compare it with the one from the Delassus operator class
        for w in range(delassus.num_worlds):
            # Compute the Delassus matrix using the inverse mass matrix and the Jacobian
            D_w = (J_cts_np[w] @ invM_np[w]) @ J_cts_np[w].T

            # Compare the computed Delassus matrix with the one from the dual problem
            is_D_close = np.allclose(D_np[w], D_w, atol=1e-3, rtol=1e-4)
            if not is_D_close or self.verbose:
                print(f"[{w}]: D_w (shape={D_w.shape}):\n{D_w}")
                print(f"[{w}]: D_np (shape={D_np[w].shape}):\n{D_np[w]}")
                print_error_stats(f"D[{w}]", D_np[w], D_w, active_dims[w])
            self.assertTrue(is_D_close)

    def test_07_regularize_delassus_operator(self):
        # Model constants
        max_world_contacts = 12

        # Create a heterogeneous model description using model builders
        builder = make_basics_heterogeneous_builder()

        # Create the model and containers from the builder
        model, data, limits, detector, jacobians = make_containers(
            builder=builder, max_world_contacts=max_world_contacts, device=self.default_device
        )

        # Update the containers
        update_containers(model=model, data=data, limits=limits, detector=detector, jacobians=jacobians)

        # Create the Delassus operator
        delassus = DelassusOperator(
            model=model,
            data=data,
            limits=limits,
            contacts=detector.contacts,
            device=self.default_device,
            solver=LLTSequentialSolver,
        )

        # Build the Delassus operator from the current data
        delassus.build(model=model, data=data, jacobians=jacobians.data, reset_to_zero=True)

        # Extract the active constraint dimensions
        active_dims = extract_active_constraint_dims(delassus)

        # Now we reset the Delassus operator to zero and use diagonal regularization to set the diagonal entries to 1.0
        eta_wp = wp.full(
            shape=(delassus._model_maxdims,), value=wp.float32(1.0), dtype=wp.float32, device=self.default_device
        )
        delassus.zero()
        delassus.regularize(eta_wp)

        # Extract Delassus data as numpy arrays
        D_np = extract_delassus(delassus, only_active_dims=True)

        # Optional verbose output
        if self.verbose:
            print("\n")
            for i in range(len(active_dims)):
                print(f"[{i}]: active_dims: {active_dims[i]}")
            for i in range(len(D_np)):
                print(f"[{i}]: D_np (shape={D_np[i].shape}):\n{D_np[i]}")
            print("")  # Add a newline for better readability
            print_delassus_info(delassus)
            print("")  # Add a newline for better readability

        # For each world, compute the Delassus matrix using numpy and
        # compare it with the one from the Delassus operator class
        num_worlds = delassus.num_worlds
        for w in range(num_worlds):
            # Create reference
            D_w = np.eye(active_dims[w], dtype=np.float32)

            # Compare the computed Delassus matrix with the one from the dual problem
            is_D_close = np.allclose(D_np[w], D_w, atol=1e-3, rtol=1e-4)
            if not is_D_close or self.verbose:
                print(f"[{w}]: D_w (shape={D_w.shape}):\n{D_w}")
                print(f"[{w}]: D_np (shape={D_np[w].shape}):\n{D_np[w]}")
                print_error_stats(f"D[{w}]", D_np[w], D_w, active_dims[w])
            self.assertTrue(is_D_close)

    def test_08_delassus_operator_factorize_and_solve_with_sequential_cholesky(self):
        """
        Tests the factorization of a Delassus matrix and solving linear
        systems with randomly generated right-hand-side vectors.
        """
        # Model constants
        max_world_contacts = 12

        # Create a heterogeneous model description using model builders
        builder = make_basics_heterogeneous_builder()

        # Create the model and containers from the builder
        model, data, limits, detector, jacobians = make_containers(
            builder=builder, max_world_contacts=max_world_contacts, device=self.default_device
        )

        # Update the containers
        update_containers(model=model, data=data, limits=limits, detector=detector, jacobians=jacobians)
        if self.verbose:
            print("")  # Print a newline for better readability
            print(f"model.info.num_joint_cts: {model.info.num_joint_cts}")
            print(f"limits.data.world_max_limits: {limits.data.world_max_limits}")
            print(f"limits.data.world_active_limits: {limits.data.world_active_limits}")
            print(f"contacts.data.world_max_contacts: {detector.contacts.data.world_max_contacts}")
            print(f"contacts.data.world_active_contacts: {detector.contacts.data.world_active_contacts}")
            print(f"data.info.num_total_cts: {data.info.num_total_cts}")
            print("")  # Print a newline for better readability

        # Create the Delassus operator
        delassus = DelassusOperator(
            model=model,
            data=data,
            limits=limits,
            contacts=detector.contacts,
            device=self.default_device,
            solver=LLTSequentialSolver,
        )

        # Build the Delassus operator from the current data
        delassus.build(model=model, data=data, jacobians=jacobians.data, reset_to_zero=True)

        # Extract the active constraint dimensions
        active_dims = extract_active_constraint_dims(delassus)

        # Add some regularization to the Delassus matrix to ensure it is positive definite
        eta = 10.0  # TODO: investigate why this has to be so large
        eta_wp = wp.full(
            shape=(delassus._model_maxdims,), value=wp.float32(eta), dtype=wp.float32, device=self.default_device
        )
        delassus.regularize(eta=eta_wp)

        # Factorize the Delassus matrix
        delassus.compute(reset_to_zero=True)

        # Extract Delassus data as numpy arrays
        D_np = extract_delassus(delassus, only_active_dims=True)

        # For each world, generate a random right-hand side vector
        num_worlds = delassus.num_worlds
        vio_np = delassus.info.vio.numpy()
        v_f_np = np.zeros(shape=(delassus._model_maxdims,), dtype=np.float32)
        for w in range(num_worlds):
            v_f_w = random_rhs_for_matrix(D_np[w])
            v_f_np[vio_np[w] : vio_np[w] + v_f_w.size] = v_f_w

        # Construct a warp array for the free-velocity and solution vectors
        v_f_wp = wp.array(v_f_np, dtype=wp.float32, device=self.default_device)
        x_wp = wp.zeros(shape=(delassus._model_maxdims,), dtype=wp.float32, device=self.default_device)

        # Solve the linear system using the Delassus operator
        delassus.solve(v=v_f_wp, x=x_wp)

        # Extract free-velocity and solution vectors lists of numpy arrays
        v_f_np = extract_problem_vector(delassus, vector=v_f_wp.numpy(), only_active_dims=True)
        x_wp_np = extract_problem_vector(delassus, vector=x_wp.numpy(), only_active_dims=True)

        # For each world, solve the linear system using numpy
        x_np: list[np.ndarray] = []
        for w in range(num_worlds):
            x_np.append(np.linalg.solve(D_np[w][: active_dims[w], : active_dims[w]], v_f_np[w]))

        # Optional verbose output
        if self.verbose:
            for i in range(len(active_dims)):
                print(f"[{i}]: active_dims: {active_dims[i]}")
            for i in range(len(D_np)):
                print(f"[{i}]: D_np (shape={D_np[i].shape}):\n{D_np[i]}")
            for w in range(num_worlds):
                print(f"[{w}]: v_f_np: {v_f_np[w]}")
            for w in range(num_worlds):
                print(f"[{w}]: x_np: {x_np[w]}")
                print(f"[{w}]: x_wp: {x_wp_np[w]}")
            print("")  # Add a newline for better readability
            print_delassus_info(delassus)
            print("")  # Add a newline for better readability

        # For each world, compare the numpy and DelassusOperator solutions
        for w in range(num_worlds):
            # Compare the reconstructed solution vector with the one computed using numpy
            is_x_close = np.allclose(x_wp_np[w], x_np[w], atol=1e-3, rtol=1e-4)
            if not is_x_close or self.verbose:
                print_error_stats(f"x[{w}]", x_wp_np[w], x_np[w], active_dims[w])
            self.assertTrue(is_x_close)


###
# Test execution
###

if __name__ == "__main__":
    # Test setup
    setup_tests()

    # Run all tests
    unittest.main(verbosity=2)
