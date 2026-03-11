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

from newton._src.solvers.kamino._src.core.data import DataKamino
from newton._src.solvers.kamino._src.core.model import ModelKamino
from newton._src.solvers.kamino._src.dynamics.delassus import BlockSparseMatrixFreeDelassusOperator, DelassusOperator
from newton._src.solvers.kamino._src.geometry.contacts import ContactsKamino
from newton._src.solvers.kamino._src.kinematics.constraints import get_max_constraints_per_world
from newton._src.solvers.kamino._src.kinematics.jacobians import SparseSystemJacobians
from newton._src.solvers.kamino._src.kinematics.limits import LimitsKamino
from newton._src.solvers.kamino._src.linalg import LLTSequentialSolver
from newton._src.solvers.kamino._src.models.builders.basics import (
    build_boxes_fourbar,
    build_boxes_nunchaku,
    make_basics_heterogeneous_builder,
)
from newton._src.solvers.kamino._src.models.builders.utils import make_homogeneous_builder
from newton._src.solvers.kamino._src.utils import logger as msg
from newton._src.solvers.kamino.tests import setup_tests, test_context
from newton._src.solvers.kamino.tests.utils.extract import (
    extract_active_constraint_dims,
    extract_cts_jacobians,
    extract_delassus,
    extract_delassus_sparse,
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
    model: ModelKamino,
    limits: LimitsKamino,
    contacts: ContactsKamino,
    delassus: DelassusOperator,
) -> None:
    # Compute expected and allocated dimensions and sizes
    expected_max_constraint_dims = get_max_constraints_per_world(model, limits, contacts)
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
        model, data, state, limits, detector, jacobians = make_containers(
            builder=builder, max_world_contacts=max_world_contacts, device=self.default_device
        )

        # Update the containers
        update_containers(model=model, data=data, state=state, limits=limits, detector=detector, jacobians=jacobians)

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
        model, data, state, limits, detector, jacobians = make_containers(
            builder=builder, max_world_contacts=max_world_contacts, device=self.default_device
        )

        # Update the containers
        update_containers(model=model, data=data, state=state, limits=limits, detector=detector, jacobians=jacobians)

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
        model, data, state, limits, detector, jacobians = make_containers(
            builder=builder, max_world_contacts=max_world_contacts, device=self.default_device
        )

        # Update the containers
        update_containers(model=model, data=data, state=state, limits=limits, detector=detector, jacobians=jacobians)

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
        builder = build_boxes_fourbar(z_offset=0.0, ground=False, dynamic_joints=True, implicit_pd=True)

        # Create the model and containers from the builder
        model, data, state, limits, detector, jacobians = make_containers(
            builder=builder,
            max_world_contacts=max_world_contacts,
            device=self.default_device,
            sparse=True,
        )

        # Update the containers
        update_containers(model=model, data=data, state=state, limits=limits, detector=detector, jacobians=jacobians)

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
        delassus.build(model=model, data=data, jacobians=jacobians, reset_to_zero=True)

        # Extract the active constraint dimensions
        active_dims = extract_active_constraint_dims(data)
        active_size = [dims * dims for dims in active_dims]

        # Extract Jacobians as numpy arrays
        J_cts_np = extract_cts_jacobians(model, limits, detector.contacts, jacobians, only_active_cts=True)

        # Extract Delassus data as numpy arrays
        D_np = extract_delassus(delassus, only_active_dims=True)

        # Construct a list of generalized inverse mass matrices of each world
        invM_np = make_inverse_generalized_mass_matrices(model, data)

        # Construct the joint armature regularization term for each world
        njdcts = model.info.num_joint_dynamic_cts.numpy()
        jdcts_start = model.info.joint_dynamic_cts_offset.numpy()
        inv_M_q_np = [np.zeros(shape=(dim,), dtype=np.float32) for dim in active_dims]
        if np.any(njdcts):
            inv_m_j_np = data.joints.inv_m_j.numpy()
            for w in range(delassus.num_worlds):
                inv_M_q_np[w][: njdcts[w]] = inv_m_j_np[jdcts_start[w] : jdcts_start[w] + njdcts[w]]
                inv_M_q_np[w] = np.diag(inv_M_q_np[w])
                msg.info(f"[{w}]: inv_M_q_np (shape={inv_M_q_np[w].shape}):\n{inv_M_q_np[w]}\n\n")

        # For each world, compute the Delassus matrix using numpy and
        # compare it with the one from the Delassus operator class
        for w in range(delassus.num_worlds):
            # Compute the Delassus matrix using the inverse mass matrix and the Jacobian
            D_w = (J_cts_np[w] @ invM_np[w]) @ J_cts_np[w].T + inv_M_q_np[w]

            # Compare the computed Delassus matrix with the one from the dual problem
            is_D_close = np.allclose(D_np[w], D_w, rtol=1e-3, atol=1e-4)
            if not is_D_close or self.verbose:
                msg.warning(f"[{w}]: D_w (shape={D_w.shape}):\n{D_w}")
                msg.warning(f"[{w}]: D_np (shape={D_np[w].shape}):\n{D_np[w]}")
                print_error_stats(f"D[{w}]", D_np[w], D_w, active_size[w], show_errors=True)
            self.assertTrue(is_D_close)

    def test_05_build_homogeneous_delassus_operator(self):
        # Model constants
        num_worlds = 3
        max_world_contacts = 12

        # Construct a homogeneous model description using model builders
        builder = make_homogeneous_builder(num_worlds=num_worlds, build_fn=build_boxes_nunchaku)

        # Create the model and containers from the builder
        model, data, state, limits, detector, jacobians = make_containers(
            builder=builder, max_world_contacts=max_world_contacts, device=self.default_device
        )

        # Update the containers
        update_containers(model=model, data=data, state=state, limits=limits, detector=detector, jacobians=jacobians)

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
        delassus.build(model=model, data=data, jacobians=jacobians, reset_to_zero=True)

        # Extract the active constraint dimensions
        active_dims = extract_active_constraint_dims(data)

        # Extract Jacobians as numpy arrays
        J_cts_np = extract_cts_jacobians(model, limits, detector.contacts, jacobians, only_active_cts=True)

        # Extract Delassus data as numpy arrays
        D_np = extract_delassus(delassus, only_active_dims=True)

        # Construct a list of generalized inverse mass matrices of each world
        invM_np = make_inverse_generalized_mass_matrices(model, data)

        # Construct the joint armature regularization term for each world
        njdcts = model.info.num_joint_dynamic_cts.numpy()
        jdcts_start = model.info.joint_dynamic_cts_offset.numpy()
        inv_M_q_np = [np.zeros(shape=(dim,), dtype=np.float32) for dim in active_dims]
        if np.any(njdcts):
            inv_m_j_np = data.joints.inv_m_j.numpy()
            for w in range(delassus.num_worlds):
                inv_M_q_np[w][: njdcts[w]] = inv_m_j_np[jdcts_start[w] : jdcts_start[w] + njdcts[w]]
                inv_M_q_np[w] = np.diag(inv_M_q_np[w])
                print(f"[{w}]: inv_M_q_np (shape={inv_M_q_np[w].shape}):\n{inv_M_q_np[w]}\n\n")

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
            D_w = (J_cts_np[w] @ invM_np[w]) @ J_cts_np[w].T + inv_M_q_np[w]

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

        # Create the model and containers from the builder
        model, data, state, limits, detector, jacobians = make_containers(
            builder=builder, max_world_contacts=max_world_contacts, device=self.default_device
        )

        # Update the containers
        update_containers(model=model, data=data, state=state, limits=limits, detector=detector, jacobians=jacobians)

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
        delassus.build(model=model, data=data, jacobians=jacobians, reset_to_zero=True)

        # Extract the active constraint dimensions
        active_dims = extract_active_constraint_dims(data)

        # Extract Jacobians as numpy arrays
        J_cts_np = extract_cts_jacobians(model, limits, detector.contacts, jacobians, only_active_cts=True)

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
        model, data, state, limits, detector, jacobians = make_containers(
            builder=builder, max_world_contacts=max_world_contacts, device=self.default_device
        )

        # Update the containers
        update_containers(model=model, data=data, state=state, limits=limits, detector=detector, jacobians=jacobians)

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
        delassus.build(model=model, data=data, jacobians=jacobians, reset_to_zero=True)

        # Extract the active constraint dimensions
        active_dims = extract_active_constraint_dims(data)

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
        model, data, state, limits, detector, jacobians = make_containers(
            builder=builder, max_world_contacts=max_world_contacts, device=self.default_device
        )

        # Update the containers
        update_containers(model=model, data=data, state=state, limits=limits, detector=detector, jacobians=jacobians)
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
        delassus.build(model=model, data=data, jacobians=jacobians, reset_to_zero=True)

        # Extract the active constraint dimensions
        active_dims = extract_active_constraint_dims(data)

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

    def test_09_compare_dense_sparse_delassus_operator_assembly(self):
        # Model constants
        max_world_contacts = 12

        # Construct the model description using model builders for different systems
        # builder = build_boxes_hinged(z_offset=0.0, ground=False)
        builder = build_boxes_fourbar(z_offset=0.0, ground=False)

        # Create the model and containers from the builder
        model, data, state, limits, detector, jacobians_dense = make_containers(
            builder=builder, max_world_contacts=max_world_contacts, device=self.default_device, sparse=False
        )
        jacobians_sparse = SparseSystemJacobians(
            model=model, limits=limits, contacts=detector.contacts, device=self.default_device
        )

        # Update the containers
        update_containers(model, data, state, limits, detector, jacobians_dense)
        update_containers(model, data, state, limits, detector, jacobians_sparse)

        # Create the Delassus operator
        delassus_dense = DelassusOperator(
            model=model,
            data=data,
            limits=limits,
            contacts=detector.contacts,
            device=self.default_device,
            solver=LLTSequentialSolver,
        )
        delassus_sparse = DelassusOperator(
            model=model,
            data=data,
            limits=limits,
            contacts=detector.contacts,
            device=self.default_device,
            solver=LLTSequentialSolver,
        )

        # Build the Delassus operator from the current data
        delassus_dense.build(model=model, data=data, jacobians=jacobians_dense, reset_to_zero=True)
        delassus_sparse.build(model=model, data=data, jacobians=jacobians_sparse, reset_to_zero=True)

        # Extract the active constraint dimensions
        active_dims = extract_active_constraint_dims(data)
        active_size = [dims * dims for dims in active_dims]

        # Extract Delassus data as numpy arrays
        D_dense_np = extract_delassus(delassus_dense, only_active_dims=True)
        D_sparse_np = extract_delassus(delassus_sparse, only_active_dims=True)

        # For each world, compare the Delassus matrix
        for w in range(delassus_dense.num_worlds):
            # Compare the computed Delassus matrix with the one from the dual problem
            is_D_close = np.allclose(D_dense_np[w], D_sparse_np[w], rtol=1e-3, atol=1e-4)
            if not is_D_close or self.verbose:
                print(f"[{w}]: D_dense_np (shape={D_dense_np[w].shape}):\n{D_dense_np[w]}")
                print(f"[{w}]: D_sparse_np (shape={D_sparse_np[w].shape}):\n{D_sparse_np[w]}")
                print_error_stats(f"D[{w}]", D_dense_np[w], D_sparse_np[w], active_size[w], show_errors=True)
            self.assertTrue(is_D_close)

    def test_10_compare_dense_sparse_homogeneous_delassus_operator_assembly(self):
        # Model constants
        num_worlds = 3
        max_world_contacts = 12

        # Construct a homogeneous model description using model builders
        builder = make_homogeneous_builder(num_worlds=num_worlds, build_fn=build_boxes_nunchaku)

        # Create the model and containers from the builder
        model, data, state, limits, detector, jacobians_dense = make_containers(
            builder=builder, max_world_contacts=max_world_contacts, device=self.default_device, sparse=False
        )
        jacobians_sparse = SparseSystemJacobians(
            model=model, limits=limits, contacts=detector.contacts, device=self.default_device
        )

        # Update the containers
        update_containers(model, data, state, limits, detector, jacobians_dense)
        update_containers(model, data, state, limits, detector, jacobians_sparse)

        # Create the Delassus operator
        delassus_dense = DelassusOperator(
            model=model,
            data=data,
            limits=limits,
            contacts=detector.contacts,
            device=self.default_device,
            solver=LLTSequentialSolver,
        )
        delassus_sparse = DelassusOperator(
            model=model,
            data=data,
            limits=limits,
            contacts=detector.contacts,
            device=self.default_device,
            solver=LLTSequentialSolver,
        )

        # Build the Delassus operator from the current data
        delassus_dense.build(model=model, data=data, jacobians=jacobians_dense, reset_to_zero=True)
        delassus_sparse.build(model=model, data=data, jacobians=jacobians_sparse, reset_to_zero=True)

        # Extract the active constraint dimensions
        active_dims = extract_active_constraint_dims(data)

        # Extract Delassus data as numpy arrays
        D_dense_np = extract_delassus(delassus_dense, only_active_dims=True)
        D_sparse_np = extract_delassus(delassus_sparse, only_active_dims=True)

        # For each world, compare the Delassus matrix
        for w in range(delassus_dense.num_worlds):
            # Compare the computed Delassus matrix with the one from the dual problem
            is_D_close = np.allclose(D_dense_np[w], D_sparse_np[w], rtol=1e-3, atol=1e-4)
            if not is_D_close or self.verbose:
                print(f"[{w}]: D_dense_np (shape={D_dense_np[w].shape}):\n{D_dense_np[w]}")
                print(f"[{w}]: D_sparse_np (shape={D_sparse_np[w].shape}):\n{D_sparse_np[w]}")
                print_error_stats(f"D[{w}]", D_dense_np[w], D_sparse_np[w], active_dims[w] * active_dims[w])
            self.assertTrue(is_D_close)

    def test_11_compare_dense_sparse_heterogeneous_delassus_operator_assembly(self):
        # Model constants
        max_world_contacts = 12

        # Create a heterogeneous model description using model builders
        builder = make_basics_heterogeneous_builder()

        # Create the model and containers from the builder
        model, data, state, limits, detector, jacobians_dense = make_containers(
            builder=builder, max_world_contacts=max_world_contacts, device=self.default_device, sparse=False
        )
        jacobians_sparse = SparseSystemJacobians(
            model=model, limits=limits, contacts=detector.contacts, device=self.default_device
        )

        # Update the containers
        update_containers(model, data, state, limits, detector, jacobians_dense)
        update_containers(model, data, state, limits, detector, jacobians_sparse)

        # Create the Delassus operator
        delassus_dense = DelassusOperator(
            model=model,
            data=data,
            limits=limits,
            contacts=detector.contacts,
            device=self.default_device,
            solver=LLTSequentialSolver,
        )
        delassus_sparse = DelassusOperator(
            model=model,
            data=data,
            limits=limits,
            contacts=detector.contacts,
            device=self.default_device,
            solver=LLTSequentialSolver,
        )

        # Build the Delassus operator from the current data
        delassus_dense.build(model=model, data=data, jacobians=jacobians_dense, reset_to_zero=True)
        delassus_sparse.build(model=model, data=data, jacobians=jacobians_sparse, reset_to_zero=True)

        # Extract the active constraint dimensions
        active_dims = extract_active_constraint_dims(data)

        # Extract Delassus data as numpy arrays
        D_dense_np = extract_delassus(delassus_dense, only_active_dims=True)
        D_sparse_np = extract_delassus(delassus_sparse, only_active_dims=True)

        # For each world, compare the Delassus matrix
        for w in range(delassus_dense.num_worlds):
            # Compare the computed Delassus matrix with the one from the dual problem
            is_D_close = np.allclose(D_dense_np[w], D_sparse_np[w], rtol=1e-3, atol=1e-4)
            if not is_D_close or self.verbose:
                print(f"[{w}]: D_dense_np (shape={D_dense_np[w].shape}):\n{D_dense_np[w]}")
                print(f"[{w}]: D_sparse_np (shape={D_sparse_np[w].shape}):\n{D_sparse_np[w]}")
                print_error_stats(f"D[{w}]", D_dense_np[w], D_sparse_np[w], active_dims[w] * active_dims[w])
            self.assertTrue(is_D_close)


class TestDelassusOperatorSparse(unittest.TestCase):
    def setUp(self):
        if not test_context.setup_done:
            setup_tests(clear_cache=False)
        self.verbose = test_context.verbose  # Set to True for detailed output
        self.default_device = wp.get_device(test_context.device)
        self.seed = 42
        self.dynamic_joints = True

    def tearDown(self):
        self.default_device = None

    ###
    # Helpers
    ###

    def _check_sparse_delassus_allocations(
        self,
        model: ModelKamino,
        delassus: BlockSparseMatrixFreeDelassusOperator,
    ):
        """Checks the allocation of a sparse Delassus operator."""
        self.assertEqual(delassus._model, model)
        self.assertIsNotNone(delassus._data)
        self.assertIsNone(delassus._preconditioner)
        self.assertIsNone(delassus._eta)

        # Check that body space temp vector is initialized
        self.assertEqual(delassus._vec_temp_body_space.shape, (model.size.sum_of_num_body_dofs,))

        rng = np.random.default_rng(seed=self.seed)
        regularization_np = rng.standard_normal((model.size.sum_of_max_total_cts,), dtype=np.float32)
        regularization = wp.from_numpy(regularization_np, dtype=wp.float32, device=self.default_device)
        delassus.set_regularization(regularization)

        # Check that setting regularization works
        self.assertEqual(delassus._eta, regularization)

        preconditioner_np = rng.standard_normal((model.size.sum_of_max_total_cts,), dtype=np.float32)
        preconditioner = wp.from_numpy(preconditioner_np, dtype=wp.float32, device=self.default_device)
        delassus.set_preconditioner(preconditioner)
        delassus.update()

        # Check that setting preconditioner works
        self.assertEqual(delassus._preconditioner, preconditioner)

    def _check_delassus_matrix(
        self,
        model: ModelKamino,
        data: DataKamino,
        delassus: BlockSparseMatrixFreeDelassusOperator,
        jacobians: SparseSystemJacobians,
    ):
        """Checks that a sparse Delassus operator represents the correct matrix."""
        rng = np.random.default_rng(seed=self.seed)

        def run_check(use_regularization: bool, use_preconditioner: bool):
            # Extract Jacobians as numpy arrays
            J_cts_np = jacobians._J_cts.bsm.numpy()

            # Add regularization
            if use_regularization:
                regularization_list: list[np.ndarray] = []
                regularization_np = np.zeros((model.size.sum_of_max_total_cts,), dtype=np.float32)
                delassus_sizes = data.info.num_total_cts.numpy()
                jac_row_start = jacobians._J_cts.bsm.row_start.numpy()
                for w in range(model.info.num_worlds):
                    d_size = delassus_sizes[w]
                    regularization_list.append(rng.standard_normal((d_size,), dtype=np.float32))
                    regularization_np[jac_row_start[w] : jac_row_start[w] + d_size] = regularization_list[-1]
                regularization = wp.from_numpy(regularization_np, dtype=wp.float32, device=self.default_device)
                delassus.set_regularization(regularization)
            else:
                delassus.set_regularization(None)

            # Add preconditioner
            if use_preconditioner:
                preconditioner_list: list[np.ndarray] = []
                preconditioner_np = np.zeros((model.size.sum_of_max_total_cts,), dtype=np.float32)
                delassus_sizes = data.info.num_total_cts.numpy()
                jac_row_start = jacobians._J_cts.bsm.row_start.numpy()
                for w in range(model.info.num_worlds):
                    d_size = delassus_sizes[w]
                    preconditioner_list.append(rng.standard_normal((d_size,), dtype=np.float32))
                    preconditioner_np[jac_row_start[w] : jac_row_start[w] + d_size] = preconditioner_list[-1]
                preconditioner = wp.from_numpy(preconditioner_np, dtype=wp.float32, device=self.default_device)
                delassus.set_preconditioner(preconditioner)
            else:
                delassus.set_preconditioner(None)

            delassus.update()

            # Extract Delassus matrices as numpy arrays
            D_np = extract_delassus_sparse(delassus, only_active_dims=True)

            # Construct a list of generalized inverse mass matrices of each world
            invM_np = make_inverse_generalized_mass_matrices(model, data)

            # Construct the joint armature regularization term for each world
            active_dims = data.info.num_total_cts.numpy()
            num_joint_dynamic_cts = model.info.num_joint_dynamic_cts.numpy()
            joint_dynamic_cts_offset = model.info.joint_dynamic_cts_offset.numpy()
            inv_M_q_np = [np.zeros(shape=(dim,), dtype=np.float32) for dim in active_dims]
            if np.any(num_joint_dynamic_cts):
                inv_m_j_np = data.joints.inv_m_j.numpy()
                for w in range(model.info.num_worlds):
                    inv_M_q_np[w][: num_joint_dynamic_cts[w]] = inv_m_j_np[
                        joint_dynamic_cts_offset[w] : joint_dynamic_cts_offset[w] + num_joint_dynamic_cts[w]
                    ]

            # Optional verbose output
            if self.verbose:
                print("")  # Print a newline for better readability
                for i in range(len(J_cts_np)):
                    print(f"[{i}]: J_cts_np (shape={J_cts_np[i].shape}):\n{J_cts_np[i]}")
                for i in range(len(invM_np)):
                    print(f"[{i}]: invM_np (shape={invM_np[i].shape}):\n{invM_np[i]}")
                for i in range(len(D_np)):
                    print(f"[{i}]: D_np (shape={D_np[i].shape}):\n{D_np[i]}")
                print("")  # Add a newline for better readability

            # For each world, compute the Delassus matrix using numpy and compare it with the matrix
            # represented by the Delassus operator.
            for w in range(model.info.num_worlds):
                # Compute the Delassus matrix using the inverse mass matrix and the Jacobian
                D_w = (J_cts_np[w] @ invM_np[w]) @ J_cts_np[w].T + np.diag(inv_M_q_np[w])
                if use_preconditioner:
                    D_w = np.diag(preconditioner_list[w]) @ D_w @ np.diag(preconditioner_list[w])
                if use_regularization:
                    D_w = D_w + np.diag(regularization_list[w])

                is_D_close = np.allclose(D_np[w], D_w, atol=1e-3, rtol=1e-4)
                if not is_D_close or self.verbose:
                    print(f"[{w}]: D_w (shape={D_w.shape}):\n{D_w}")
                    print(f"[{w}]: D_np (shape={D_np[w].shape}):\n{D_np[w]}")
                    print_error_stats(f"D[{w}]", D_np[w], D_w, D_w.shape[0])
                self.assertTrue(is_D_close)

        run_check(use_regularization=False, use_preconditioner=False)
        run_check(use_regularization=True, use_preconditioner=False)
        run_check(use_regularization=False, use_preconditioner=True)
        run_check(use_regularization=True, use_preconditioner=True)

    def _check_delassus_matrix_vector_product(
        self,
        model: ModelKamino,
        data: DataKamino,
        delassus: BlockSparseMatrixFreeDelassusOperator,
        jacobians: SparseSystemJacobians,
    ):
        """Checks the different matrix-vector products provided by the sparse Delassus operator."""
        rng = np.random.default_rng(seed=self.seed)

        def run_check(use_regularization: bool, use_preconditioner: bool, mask_worlds: bool):
            delassus_sizes = data.info.num_total_cts.numpy()
            jac_row_start = jacobians._J_cts.bsm.row_start.numpy()

            # Add regularization
            if use_regularization:
                regularization_list: list[np.ndarray] = []
                regularization_np = np.zeros((model.size.sum_of_max_total_cts,), dtype=np.float32)
                for w in range(model.info.num_worlds):
                    d_size = delassus_sizes[w]
                    regularization_list.append(rng.standard_normal((d_size,), dtype=np.float32))
                    regularization_np[jac_row_start[w] : jac_row_start[w] + d_size] = regularization_list[-1]
                regularization = wp.from_numpy(regularization_np, dtype=wp.float32, device=self.default_device)
                delassus.set_regularization(regularization)
            else:
                delassus.set_regularization(None)

            # Add preconditioner
            if use_preconditioner:
                preconditioner_list: list[np.ndarray] = []
                preconditioner_np = np.zeros((model.size.sum_of_max_total_cts,), dtype=np.float32)
                for w in range(model.info.num_worlds):
                    d_size = delassus_sizes[w]
                    preconditioner_list.append(rng.standard_normal((d_size,), dtype=np.float32))
                    preconditioner_np[jac_row_start[w] : jac_row_start[w] + d_size] = preconditioner_list[-1]
                preconditioner = wp.from_numpy(preconditioner_np, dtype=wp.float32, device=self.default_device)
                delassus.set_preconditioner(preconditioner)
            else:
                delassus.set_preconditioner(None)

            delassus.update()

            # Generate vectors for multiplication
            alpha = float(rng.standard_normal((1,))[0])
            beta = float(rng.standard_normal((1,))[0])
            input_vec_list: list[np.ndarray] = []
            offset_vec_list: list[np.ndarray] = []
            input_vec_np = np.zeros((model.size.sum_of_max_total_cts,), dtype=np.float32)
            offset_vec_np = np.zeros((model.size.sum_of_max_total_cts,), dtype=np.float32)
            for w in range(model.info.num_worlds):
                d_size = delassus_sizes[w]
                input_vec_list.append(rng.standard_normal((d_size,), dtype=np.float32))
                offset_vec_list.append(rng.standard_normal((d_size,), dtype=np.float32))
                input_vec_np[jac_row_start[w] : jac_row_start[w] + d_size] = input_vec_list[-1]
                offset_vec_np[jac_row_start[w] : jac_row_start[w] + d_size] = offset_vec_list[-1]
            input_vec = wp.from_numpy(input_vec_np, dtype=wp.float32, device=self.default_device)
            output_vec_matmul = wp.zeros_like(input_vec)
            output_vec_gemv = wp.from_numpy(offset_vec_np, dtype=wp.float32, device=self.default_device)
            output_vec_gemv_zero = wp.from_numpy(offset_vec_np, dtype=wp.float32, device=self.default_device)

            mask_np = np.ones((model.size.num_worlds,), dtype=np.int32)
            if mask_worlds:
                mask_np[::2] = 0
            world_mask = wp.from_numpy(mask_np, dtype=wp.int32, device=self.default_device)

            # Compute different products (simple matvec, gemv, and gemv with beta = 0.0)
            delassus.matvec(input_vec, output_vec_matmul, world_mask)
            delassus.gemv(input_vec, output_vec_gemv, world_mask, alpha, beta)
            delassus.gemv(input_vec, output_vec_gemv_zero, world_mask, alpha, 0.0)

            output_vec_matmul_np = output_vec_matmul.numpy()
            output_vec_gemv_np = output_vec_gemv.numpy()
            output_vec_gemv_zero_np = output_vec_gemv_zero.numpy()

            # Extract Jacobians as numpy arrays
            J_cts_np = jacobians._J_cts.bsm.numpy()

            # Construct a list of generalized inverse mass matrices of each world
            invM_np = make_inverse_generalized_mass_matrices(model, data)

            # Construct the joint armature regularization term for each world
            active_dims = data.info.num_total_cts.numpy()
            num_joint_dynamic_cts = model.info.num_joint_dynamic_cts.numpy()
            joint_dynamic_cts_offset = model.info.joint_dynamic_cts_offset.numpy()
            inv_M_q_np = [np.zeros(shape=(dim,), dtype=np.float32) for dim in active_dims]
            if np.any(num_joint_dynamic_cts):
                inv_m_j_np = data.joints.inv_m_j.numpy()
                for w in range(model.info.num_worlds):
                    inv_M_q_np[w][: num_joint_dynamic_cts[w]] = inv_m_j_np[
                        joint_dynamic_cts_offset[w] : joint_dynamic_cts_offset[w] + num_joint_dynamic_cts[w]
                    ]

            # For each world, compute the Delassus matrix-vector product using numpy and compare it
            # with the one from the Delassus operator class
            for w in range(model.info.num_worlds):
                vec_matmul_w = output_vec_matmul_np[jac_row_start[w] : jac_row_start[w] + delassus_sizes[w]]
                vec_gemv_w = output_vec_gemv_np[jac_row_start[w] : jac_row_start[w] + delassus_sizes[w]]
                vec_gemv_zero_w = output_vec_gemv_zero_np[jac_row_start[w] : jac_row_start[w] + delassus_sizes[w]]

                if mask_np[w] == 0:
                    self.assertEqual(np.max(np.abs(vec_matmul_w)), 0.0)
                    self.assertTrue((vec_gemv_w == offset_vec_list[w]).all())
                    self.assertTrue((vec_gemv_zero_w == offset_vec_list[w]).all())
                else:
                    # Compute the Delassus matrix using the inverse mass matrix and the Jacobian
                    D_w = (J_cts_np[w] @ invM_np[w]) @ J_cts_np[w].T + np.diag(inv_M_q_np[w])
                    if use_preconditioner:
                        D_w = np.diag(preconditioner_list[w]) @ D_w @ np.diag(preconditioner_list[w])
                    if use_regularization:
                        D_w = D_w + np.diag(regularization_list[w])

                    vec_matmul_ref = D_w @ input_vec_list[w]
                    vec_gemv_ref = alpha * (D_w @ input_vec_list[w]) + beta * offset_vec_list[w]
                    vec_gemv_zero_ref = alpha * (D_w @ input_vec_list[w])

                    # Compare the computed Delassus matrix with the one from the dual problem
                    is_matmul_close = np.allclose(vec_matmul_ref, vec_matmul_w, atol=1e-3, rtol=1e-4)
                    if not is_matmul_close or self.verbose:
                        print(f"[{w}]: vec_matmul_ref (shape={vec_matmul_ref.shape}):\n{vec_matmul_ref}")
                        print(f"[{w}]: vec_matmul_w (shape={vec_matmul_w.shape}):\n{vec_matmul_w}")
                        print_error_stats(
                            f"vec_gemv_zero[{w}]",
                            vec_matmul_ref,
                            vec_matmul_w,
                            vec_matmul_w.shape[0],
                        )
                    self.assertTrue(is_matmul_close)
                    is_gemv_close = np.allclose(vec_gemv_ref, vec_gemv_w, atol=1e-3, rtol=1e-4)
                    if not is_gemv_close or self.verbose:
                        print(f"[{w}]: vec_gemv_ref (shape={vec_gemv_ref.shape}):\n{vec_gemv_ref}")
                        print(f"[{w}]: vec_gemv_w (shape={vec_gemv_w.shape}):\n{vec_gemv_w}")
                        print_error_stats(
                            f"vec_gemv_zero[{w}]",
                            vec_gemv_ref,
                            vec_gemv_w,
                            vec_gemv_w.shape[0],
                        )
                    self.assertTrue(is_gemv_close)
                    is_gemv_zero_close = np.allclose(vec_gemv_zero_ref, vec_gemv_zero_w, atol=1e-3, rtol=1e-4)
                    if not is_gemv_zero_close or self.verbose:
                        print(f"[{w}]: vec_gemv_zero_ref (shape={vec_gemv_zero_ref.shape}):\n{vec_gemv_zero_ref}")
                        print(f"[{w}]: vec_gemv_zero_w (shape={vec_gemv_zero_w.shape}):\n{vec_gemv_zero_w}")
                        print_error_stats(
                            f"vec_gemv_zero[{w}]",
                            vec_gemv_zero_ref,
                            vec_gemv_zero_w,
                            vec_gemv_zero_w.shape[0],
                        )
                    self.assertTrue(is_gemv_zero_close)

        run_check(use_regularization=False, use_preconditioner=False, mask_worlds=False)
        run_check(use_regularization=True, use_preconditioner=False, mask_worlds=False)
        run_check(use_regularization=False, use_preconditioner=True, mask_worlds=False)
        run_check(use_regularization=True, use_preconditioner=True, mask_worlds=False)
        run_check(use_regularization=False, use_preconditioner=False, mask_worlds=True)
        run_check(use_regularization=True, use_preconditioner=False, mask_worlds=True)
        run_check(use_regularization=False, use_preconditioner=True, mask_worlds=True)
        run_check(use_regularization=True, use_preconditioner=True, mask_worlds=True)

    def _check_delassus_diagonal(
        self,
        model: ModelKamino,
        data: DataKamino,
        delassus: BlockSparseMatrixFreeDelassusOperator,
        jacobians: SparseSystemJacobians,
    ):
        """Check the diagonal extraction routine of the sparse Delassus operator."""
        # Extract Jacobians as numpy arrays
        J_cts_np = jacobians._J_cts.bsm.numpy()

        # Get diagonals from the sparse Delassus operator and split the single array into separate
        # numpy vectors
        D_diag = wp.zeros((model.size.sum_of_max_total_cts,), dtype=wp.float32, device=self.default_device)
        delassus.diagonal(D_diag)
        D_diag_np = D_diag.numpy()
        row_start_np = jacobians._J_cts.bsm.row_start.numpy()
        num_total_cts_np = data.info.num_total_cts.numpy()
        max_total_cts_np = model.info.max_total_cts.numpy()
        D_diag_list = []
        for w in range(model.info.num_worlds):
            D_diag_list.append(D_diag_np[row_start_np[w] : row_start_np[w] + num_total_cts_np[w]])
            if max_total_cts_np[w] > num_total_cts_np[w]:
                # Check that unused entries of the diagonal vector are zero
                diag_unused = D_diag_np[row_start_np[w] + num_total_cts_np[w] : row_start_np[w] + max_total_cts_np[w]]
                self.assertEqual(np.max(np.abs(diag_unused)), 0)

        # Construct a list of generalized inverse mass matrices of each world
        invM_np = make_inverse_generalized_mass_matrices(model, data)

        # Construct the joint armature regularization term for each world
        active_dims = data.info.num_total_cts.numpy()
        num_joint_dynamic_cts = model.info.num_joint_dynamic_cts.numpy()
        joint_dynamic_cts_offset = model.info.joint_dynamic_cts_offset.numpy()
        inv_M_q_np = [np.zeros(shape=(dim,), dtype=np.float32) for dim in active_dims]
        if np.any(num_joint_dynamic_cts):
            inv_m_j_np = data.joints.inv_m_j.numpy()
            for w in range(model.info.num_worlds):
                inv_M_q_np[w][: num_joint_dynamic_cts[w]] = inv_m_j_np[
                    joint_dynamic_cts_offset[w] : joint_dynamic_cts_offset[w] + num_joint_dynamic_cts[w]
                ]

        # Optional verbose output
        if self.verbose:
            print("")  # Print a newline for better readability
            for i in range(len(J_cts_np)):
                print(f"[{i}]: J_cts_np (shape={J_cts_np[i].shape}):\n{J_cts_np[i]}")
            for i in range(len(invM_np)):
                print(f"[{i}]: invM_np (shape={invM_np[i].shape}):\n{invM_np[i]}")
            for i in range(len(D_diag_list)):
                print(f"[{i}]: D_diag (shape={D_diag_list[i].shape}):\n{D_diag_list[i]}")
            print("")  # Add a newline for better readability

        # For each world, compute the Delassus matrix diagonal using numpy and compare it with the
        # one from the Delassus operator class
        for w in range(model.info.num_worlds):
            # Compute the Delassus matrix diagonal using the inverse mass matrix and the Jacobian
            D_w = (J_cts_np[w] @ invM_np[w]) @ J_cts_np[w].T + np.diag(inv_M_q_np[w])
            D_diag_ref = np.diag(D_w)

            # Compare the computed Delassus matrix diagonals
            is_D_diag_close = np.allclose(D_diag_list[w], D_diag_ref, atol=1e-3, rtol=1e-4)
            if not is_D_diag_close or self.verbose:
                print(f"[{w}]: D_diag_ref (shape={D_diag_ref.shape}):\n{D_diag_ref}")
                print(f"[{w}]: D_diag (shape={D_diag_list[w].shape}):\n{D_diag_list[w]}")
                print_error_stats(f"D_diag[{w}]", D_diag_list[w], D_diag_ref, D_diag_ref.shape[0])
            self.assertTrue(is_D_diag_close)

    ###
    # Allocation
    ###

    def test_01_allocate_single_delassus_operator(self):
        # Model constants
        max_world_contacts = 12

        # Construct the model description using model builders for different systems
        builder = build_boxes_nunchaku()

        # Create the model and containers from the builder
        model, data, state, limits, detector, jacobians = make_containers(
            builder=builder, max_world_contacts=max_world_contacts, device=self.default_device, sparse=True
        )

        # Update the containers
        update_containers(model=model, data=data, state=state, limits=limits, detector=detector, jacobians=jacobians)

        # Create the sparse Delassus operator
        delassus = BlockSparseMatrixFreeDelassusOperator(
            model=model,
            data=data,
            limits=limits,
            contacts=detector.contacts,
            device=self.default_device,
        )

        # Compare expected to allocated dimensions and sizes
        self._check_sparse_delassus_allocations(model, delassus)

    def test_02_allocate_homogeneous_delassus_operator(self):
        # Model constants
        num_worlds = 3
        max_world_contacts = 12

        # Construct a homogeneous model description using model builders
        builder = make_homogeneous_builder(num_worlds=num_worlds, build_fn=build_boxes_nunchaku)

        # Create the model and containers from the builder
        model, data, state, limits, detector, jacobians = make_containers(
            builder=builder, max_world_contacts=max_world_contacts, device=self.default_device, sparse=True
        )

        # Update the containers
        update_containers(model=model, data=data, state=state, limits=limits, detector=detector, jacobians=jacobians)

        # Create the Delassus operator
        delassus = BlockSparseMatrixFreeDelassusOperator(
            model=model,
            data=data,
            limits=limits,
            contacts=detector.contacts,
            device=self.default_device,
        )

        # Compare expected to allocated dimensions and sizes
        self._check_sparse_delassus_allocations(model, delassus)

    def test_03_allocate_heterogeneous_delassus_operator(self):
        # Model constants
        max_world_contacts = 12

        # Create a heterogeneous model description using model builders
        builder = make_basics_heterogeneous_builder()

        # Create the model and containers from the builder
        model, data, state, limits, detector, jacobians = make_containers(
            builder=builder, max_world_contacts=max_world_contacts, device=self.default_device, sparse=True
        )

        # Update the containers
        update_containers(model=model, data=data, state=state, limits=limits, detector=detector, jacobians=jacobians)

        # Create the Delassus operator
        delassus = BlockSparseMatrixFreeDelassusOperator(
            model=model,
            data=data,
            limits=limits,
            contacts=detector.contacts,
            device=self.default_device,
        )

        # Compare expected to allocated dimensions and sizes
        self._check_sparse_delassus_allocations(model, delassus)

    def test_04_build_delassus_operator(self):
        # Model constants
        max_world_contacts = 12

        # Construct the model description using model builders for different systems
        builder = build_boxes_fourbar(
            z_offset=0.0, ground=False, dynamic_joints=self.dynamic_joints, implicit_pd=self.dynamic_joints
        )

        # Create the model and containers from the builder
        model, data, state, limits, detector, jacobians = make_containers(
            builder=builder, max_world_contacts=max_world_contacts, device=self.default_device, sparse=True
        )

        # Update the containers
        update_containers(model=model, data=data, state=state, limits=limits, detector=detector, jacobians=jacobians)

        # Create the Delassus operator
        delassus = BlockSparseMatrixFreeDelassusOperator(
            model=model,
            data=data,
            limits=limits,
            contacts=detector.contacts,
            jacobians=jacobians,
            device=self.default_device,
        )

        # Check that the Delassus operator represents the actual Delassus matrix
        self._check_delassus_matrix(model, data, delassus, jacobians)

    def test_05_build_homogeneous_delassus_operator(self):
        # Model constants
        num_worlds = 3
        max_world_contacts = 12

        # Construct a homogeneous model description using model builders
        builder = make_homogeneous_builder(num_worlds=num_worlds, build_fn=build_boxes_nunchaku)

        # Create the model and containers from the builder
        model, data, state, limits, detector, jacobians = make_containers(
            builder=builder, max_world_contacts=max_world_contacts, device=self.default_device, sparse=True
        )

        # Update the containers
        update_containers(model=model, data=data, state=state, limits=limits, detector=detector, jacobians=jacobians)

        # Create the Delassus operator
        delassus = BlockSparseMatrixFreeDelassusOperator(
            model=model,
            data=data,
            limits=limits,
            contacts=detector.contacts,
            jacobians=jacobians,
            device=self.default_device,
        )

        # Check that the Delassus operator represents the actual Delassus matrix
        self._check_delassus_matrix(model, data, delassus, jacobians)

    def test_06_build_heterogeneous_delassus_operator(self):
        # Model constants
        max_world_contacts = 12

        # Create a heterogeneous model description using model builders
        builder = make_basics_heterogeneous_builder(dynamic_joints=self.dynamic_joints, implicit_pd=self.dynamic_joints)

        # Create the model and containers from the builder
        model, data, state, limits, detector, jacobians = make_containers(
            builder=builder, max_world_contacts=max_world_contacts, device=self.default_device, sparse=True
        )

        # Update the containers
        update_containers(model=model, data=data, state=state, limits=limits, detector=detector, jacobians=jacobians)

        # Create the Delassus operator
        delassus = BlockSparseMatrixFreeDelassusOperator(
            model=model,
            data=data,
            limits=limits,
            contacts=detector.contacts,
            jacobians=jacobians,
            device=self.default_device,
        )

        # Check that the Delassus operator represents the actual Delassus matrix
        self._check_delassus_matrix(model, data, delassus, jacobians)

    def test_07_extract_delassus_diagonal(self):
        # Model constants
        max_world_contacts = 12

        # Construct the model description using model builders for different systems
        builder = build_boxes_fourbar(
            z_offset=0.0, ground=False, dynamic_joints=self.dynamic_joints, implicit_pd=self.dynamic_joints
        )

        # Create the model and containers from the builder
        model, data, state, limits, detector, jacobians = make_containers(
            builder=builder, max_world_contacts=max_world_contacts, device=self.default_device, sparse=True
        )

        # Update the containers
        update_containers(model=model, data=data, state=state, limits=limits, detector=detector, jacobians=jacobians)

        # Create the Delassus operator
        delassus = BlockSparseMatrixFreeDelassusOperator(
            model=model,
            data=data,
            limits=limits,
            contacts=detector.contacts,
            jacobians=jacobians,
            device=self.default_device,
        )

        # Check that the Delassus operator represents the actual Delassus matrix
        self._check_delassus_diagonal(model, data, delassus, jacobians)

    def test_08_extract_delassus_diagonal_homogeneous(self):
        # Model constants
        num_worlds = 3
        max_world_contacts = 12

        # Construct a homogeneous model description using model builders
        builder = make_homogeneous_builder(num_worlds=num_worlds, build_fn=build_boxes_nunchaku)

        # Create the model and containers from the builder
        model, data, state, limits, detector, jacobians = make_containers(
            builder=builder, max_world_contacts=max_world_contacts, device=self.default_device, sparse=True
        )

        # Update the containers
        update_containers(model=model, data=data, state=state, limits=limits, detector=detector, jacobians=jacobians)

        # Create the Delassus operator
        delassus = BlockSparseMatrixFreeDelassusOperator(
            model=model,
            data=data,
            limits=limits,
            contacts=detector.contacts,
            jacobians=jacobians,
            device=self.default_device,
        )

        # Check that the Delassus operator represents the actual Delassus matrix
        self._check_delassus_diagonal(model, data, delassus, jacobians)

    def test_09_extract_delassus_diagonal_heterogeneous(self):
        # Model constants
        max_world_contacts = 12

        # Create a heterogeneous model description using model builders
        builder = make_basics_heterogeneous_builder(dynamic_joints=self.dynamic_joints, implicit_pd=self.dynamic_joints)

        # Create the model and containers from the builder
        model, data, state, limits, detector, jacobians = make_containers(
            builder=builder, max_world_contacts=max_world_contacts, device=self.default_device, sparse=True
        )

        # Update the containers
        update_containers(model=model, data=data, state=state, limits=limits, detector=detector, jacobians=jacobians)

        # Create the Delassus operator
        delassus = BlockSparseMatrixFreeDelassusOperator(
            model=model,
            data=data,
            limits=limits,
            contacts=detector.contacts,
            jacobians=jacobians,
            device=self.default_device,
        )

        # Check that the Delassus operator represents the actual Delassus matrix
        self._check_delassus_diagonal(model, data, delassus, jacobians)

    def test_10_delassus_operator_vector_product(self):
        # Model constants
        max_world_contacts = 12

        # Construct the model description using model builders for different systems
        builder = build_boxes_fourbar(
            z_offset=0.0, ground=False, dynamic_joints=self.dynamic_joints, implicit_pd=self.dynamic_joints
        )

        # Create the model and containers from the builder
        model, data, state, limits, detector, jacobians = make_containers(
            builder=builder, max_world_contacts=max_world_contacts, device=self.default_device, sparse=True
        )

        # Update the containers
        update_containers(model=model, data=data, state=state, limits=limits, detector=detector, jacobians=jacobians)

        # Create the Delassus operator
        delassus = BlockSparseMatrixFreeDelassusOperator(
            model=model,
            data=data,
            limits=limits,
            contacts=detector.contacts,
            jacobians=jacobians,
            device=self.default_device,
        )

        # Check that the Delassus operator represents the actual Delassus matrix
        self._check_delassus_matrix_vector_product(model, data, delassus, jacobians)

    def test_11_homogeneous_delassus_operator_vector_product(self):
        # Model constants
        num_worlds = 3
        max_world_contacts = 12

        # Construct a homogeneous model description using model builders
        builder = make_homogeneous_builder(num_worlds=num_worlds, build_fn=build_boxes_nunchaku)

        # Create the model and containers from the builder
        model, data, state, limits, detector, jacobians = make_containers(
            builder=builder, max_world_contacts=max_world_contacts, device=self.default_device, sparse=True
        )

        # Update the containers
        update_containers(model=model, data=data, state=state, limits=limits, detector=detector, jacobians=jacobians)

        # Create the Delassus operator
        delassus = BlockSparseMatrixFreeDelassusOperator(
            model=model,
            data=data,
            limits=limits,
            contacts=detector.contacts,
            jacobians=jacobians,
            device=self.default_device,
        )

        # Check that the Delassus operator represents the actual Delassus matrix
        self._check_delassus_matrix_vector_product(model, data, delassus, jacobians)

    def test_12_heterogeneous_delassus_operator_vector_product(self):
        # Model constants
        max_world_contacts = 12

        # Create a heterogeneous model description using model builders
        builder = make_basics_heterogeneous_builder(dynamic_joints=self.dynamic_joints, implicit_pd=self.dynamic_joints)

        # Create the model and containers from the builder
        model, data, state, limits, detector, jacobians = make_containers(
            builder=builder, max_world_contacts=max_world_contacts, device=self.default_device, sparse=True
        )

        # Update the containers
        update_containers(model=model, data=data, state=state, limits=limits, detector=detector, jacobians=jacobians)

        # Create the Delassus operator
        delassus = BlockSparseMatrixFreeDelassusOperator(
            model=model,
            data=data,
            limits=limits,
            contacts=detector.contacts,
            jacobians=jacobians,
            device=self.default_device,
        )

        # Check that the Delassus operator represents the actual Delassus matrix
        self._check_delassus_matrix_vector_product(model, data, delassus, jacobians)


###
# Test execution
###

if __name__ == "__main__":
    # Test setup
    setup_tests()

    # Run all tests
    unittest.main(verbosity=2)
