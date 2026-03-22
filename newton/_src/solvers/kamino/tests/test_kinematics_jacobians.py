# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""
Unit tests for `kinematics/jacobians.py`.
"""

import unittest

import numpy as np
import warp as wp

from newton._src.solvers.kamino._src.core.model import ModelKamino
from newton._src.solvers.kamino._src.geometry.contacts import ContactsKamino
from newton._src.solvers.kamino._src.kinematics.constraints import make_unilateral_constraints_info
from newton._src.solvers.kamino._src.kinematics.jacobians import (
    ColMajorSparseConstraintJacobians,
    DenseSystemJacobians,
    SparseSystemJacobians,
)
from newton._src.solvers.kamino._src.kinematics.limits import LimitsKamino
from newton._src.solvers.kamino._src.models.builders.basics import (
    build_boxes_fourbar,
    make_basics_heterogeneous_builder,
)
from newton._src.solvers.kamino._src.models.builders.utils import make_homogeneous_builder
from newton._src.solvers.kamino._src.utils import logger as msg
from newton._src.solvers.kamino.tests import setup_tests, test_context
from newton._src.solvers.kamino.tests.utils.extract import extract_cts_jacobians, extract_dofs_jacobians
from newton._src.solvers.kamino.tests.utils.make import make_test_problem_fourbar, make_test_problem_heterogeneous

###
# Module configs
###

wp.set_module_options({"enable_backward": False})


###
# Tests
###


class TestKinematicsDenseSystemJacobians(unittest.TestCase):
    def setUp(self):
        # Configs
        if not test_context.setup_done:
            setup_tests(clear_cache=False)
        self.default_device = wp.get_device(test_context.device)
        self.verbose = test_context.verbose  # Set to True for verbose output

        # Set debug-level logging to print verbose test output to console
        if self.verbose:
            print("\n")  # Add newline before test output for better readability
            msg.set_log_level(msg.LogLevel.INFO)
        else:
            msg.reset_log_level()

    def tearDown(self):
        self.default_device = None
        if self.verbose:
            msg.reset_log_level()

    def test_01_allocate_single_dense_system_jacobians_only_joints(self):
        # Construct the model description using the ModelBuilderKamino
        builder = build_boxes_fourbar()

        # Create the model from the builder
        model = builder.finalize(device=self.default_device)
        if self.verbose:
            print("")  # Add a newline for better readability
            print(f"model.size.sum_of_num_bodies: {model.size.sum_of_num_bodies}")
            print(f"model.size.sum_of_num_joints: {model.size.sum_of_num_joints}")
            print(f"model.size.sum_of_num_joint_cts: {model.size.sum_of_num_joint_cts}")
            print(f"model.size.sum_of_num_joint_dofs: {model.size.sum_of_num_joint_dofs}")

        # Create the Jacobians container
        jacobians = DenseSystemJacobians(model=model, device=self.default_device)
        if self.verbose:
            print(f"J_cts_offsets (shape={jacobians.data.J_cts_offsets.shape}): {jacobians.data.J_cts_offsets}")
            print(f"J_dofs_offsets (shape={jacobians.data.J_dofs_offsets.shape}): {jacobians.data.J_dofs_offsets}")
            print(f"J_cts_data: shape={jacobians.data.J_cts_data.shape}")
            print(f"J_dofs_data: shape={jacobians.data.J_dofs_data.shape}")

        # Check the allocations of Jacobians
        model_num_cts = model.size.sum_of_num_joint_cts
        self.assertEqual(jacobians.data.J_dofs_offsets.size, 1)
        self.assertEqual(jacobians.data.J_cts_offsets.size, 1)
        self.assertEqual(jacobians.data.J_dofs_offsets.numpy()[0], 0)
        self.assertEqual(jacobians.data.J_cts_offsets.numpy()[0], 0)
        self.assertEqual(
            jacobians.data.J_dofs_data.shape, (model.size.sum_of_num_joint_dofs * model.size.sum_of_num_body_dofs,)
        )
        self.assertEqual(jacobians.data.J_cts_data.shape, (model_num_cts * model.size.sum_of_num_body_dofs,))

    def test_02_allocate_single_dense_system_jacobians_with_limits(self):
        # Construct the model description using the ModelBuilderKamino
        builder = build_boxes_fourbar()

        # Create the model from the builder
        model = builder.finalize(device=self.default_device)
        if self.verbose:
            print("")  # Add a newline for better readability
            print(f"model.size.sum_of_num_bodies: {model.size.sum_of_num_bodies}")
            print(f"model.size.sum_of_num_joints: {model.size.sum_of_num_joints}")
            print(f"model.size.sum_of_num_joint_cts: {model.size.sum_of_num_joint_cts}")
            print(f"model.size.sum_of_num_joint_dofs: {model.size.sum_of_num_joint_dofs}")

        # Construct and allocate the limits container
        limits = LimitsKamino(model=model, device=self.default_device)
        if self.verbose:
            print("limits.model_max_limits_host: ", limits.model_max_limits_host)
            print("limits.world_max_limits_host: ", limits.world_max_limits_host)

        # Create the Jacobians container
        jacobians = DenseSystemJacobians(model=model, limits=limits, device=self.default_device)
        if self.verbose:
            print(f"J_dofs_offsets (shape={jacobians.data.J_dofs_offsets.shape}): {jacobians.data.J_dofs_offsets}")
            print(f"J_cts_offsets (shape={jacobians.data.J_cts_offsets.shape}): {jacobians.data.J_cts_offsets}")
            print(f"J_dofs_data: shape={jacobians.data.J_dofs_data.shape}")
            print(f"J_cts_data: shape={jacobians.data.J_cts_data.shape}")

        # Check the allocations of Jacobians
        model_num_cts = model.size.sum_of_num_joint_cts + limits.model_max_limits_host
        self.assertEqual(jacobians.data.J_dofs_offsets.size, 1)
        self.assertEqual(jacobians.data.J_cts_offsets.size, 1)
        self.assertEqual(jacobians.data.J_dofs_offsets.numpy()[0], 0)
        self.assertEqual(jacobians.data.J_cts_offsets.numpy()[0], 0)
        self.assertEqual(
            jacobians.data.J_dofs_data.shape, (model.size.sum_of_num_joint_dofs * model.size.sum_of_num_body_dofs,)
        )
        self.assertEqual(jacobians.data.J_cts_data.shape, (model_num_cts * model.size.sum_of_num_body_dofs,))

    def test_03_allocate_single_dense_system_jacobians_with_contacts(self):
        # Problem constants
        max_world_contacts = 12

        # Construct the model description using the ModelBuilderKamino
        builder = build_boxes_fourbar()

        # Create the model from the builder
        model = builder.finalize(device=self.default_device)
        if self.verbose:
            print("")  # Add a newline for better readability
            print(f"model.size.sum_of_num_bodies: {model.size.sum_of_num_bodies}")
            print(f"model.size.sum_of_num_joints: {model.size.sum_of_num_joints}")
            print(f"model.size.sum_of_num_joint_cts: {model.size.sum_of_num_joint_cts}")
            print(f"model.size.sum_of_num_joint_dofs: {model.size.sum_of_num_joint_dofs}")

        # Set the contact allocation capacities
        required_world_max_contacts = [max_world_contacts] * builder.num_worlds
        if self.verbose:
            print("required_world_max_contacts: ", required_world_max_contacts)

        # Construct and allocate the contacts container
        contacts = ContactsKamino(capacity=required_world_max_contacts, device=self.default_device)
        if self.verbose:
            print("contacts.default_max_world_contacts: ", contacts.default_max_world_contacts)
            print("contacts.model_max_contacts_host: ", contacts.model_max_contacts_host)
            print("contacts.world_max_contacts_host: ", contacts.world_max_contacts_host)

        # Create the Jacobians container
        jacobians = DenseSystemJacobians(model=model, contacts=contacts, device=self.default_device)
        if self.verbose:
            print(f"J_dofs_offsets (shape={jacobians.data.J_dofs_offsets.shape}): {jacobians.data.J_dofs_offsets}")
            print(f"J_cts_offsets (shape={jacobians.data.J_cts_offsets.shape}): {jacobians.data.J_cts_offsets}")
            print(f"J_dofs_data: shape={jacobians.data.J_dofs_data.shape}")
            print(f"J_cts_data: shape={jacobians.data.J_cts_data.shape}")

        # Check the allocations of Jacobians
        model_num_cts = model.size.sum_of_num_joint_cts + 3 * contacts.model_max_contacts_host
        self.assertEqual(jacobians.data.J_dofs_offsets.size, 1)
        self.assertEqual(jacobians.data.J_cts_offsets.size, 1)
        self.assertEqual(jacobians.data.J_dofs_offsets.numpy()[0], 0)
        self.assertEqual(jacobians.data.J_cts_offsets.numpy()[0], 0)
        self.assertEqual(
            jacobians.data.J_dofs_data.shape, (model.size.sum_of_num_joint_dofs * model.size.sum_of_num_body_dofs,)
        )
        self.assertEqual(jacobians.data.J_cts_data.shape, (model_num_cts * model.size.sum_of_num_body_dofs,))

    def test_04_allocate_single_dense_system_jacobians_with_limits_and_contacts(self):
        # Problem constants
        max_world_contacts = 12

        # Construct the model description using the ModelBuilderKamino
        builder = build_boxes_fourbar()

        # Create the model from the builder
        model = builder.finalize(device=self.default_device)
        if self.verbose:
            print("")  # Add a newline for better readability
            print(f"model.size.sum_of_num_bodies: {model.size.sum_of_num_bodies}")
            print(f"model.size.sum_of_num_joints: {model.size.sum_of_num_joints}")
            print(f"model.size.sum_of_num_joint_cts: {model.size.sum_of_num_joint_cts}")
            print(f"model.size.sum_of_num_joint_dofs: {model.size.sum_of_num_joint_dofs}")

        # Construct and allocate the limits container
        limits = LimitsKamino(model=model, device=self.default_device)
        if self.verbose:
            print("limits.model_max_limits_host: ", limits.model_max_limits_host)
            print("limits.world_max_limits_host: ", limits.world_max_limits_host)

        # Set the contact allocation capacities
        required_world_max_contacts = [max_world_contacts] * builder.num_worlds
        if self.verbose:
            print("required_world_max_contacts: ", required_world_max_contacts)

        # Construct and allocate the contacts container
        contacts = ContactsKamino(capacity=required_world_max_contacts, device=self.default_device)
        if self.verbose:
            print("contacts.default_max_world_contacts: ", contacts.default_max_world_contacts)
            print("contacts.model_max_contacts_host: ", contacts.model_max_contacts_host)
            print("contacts.world_max_contacts_host: ", contacts.world_max_contacts_host)

        # Create the Jacobians container
        jacobians = DenseSystemJacobians(model=model, limits=limits, contacts=contacts, device=self.default_device)
        if self.verbose:
            print(f"J_dofs_offsets (shape={jacobians.data.J_dofs_offsets.shape}): {jacobians.data.J_dofs_offsets}")
            print(f"J_cts_offsets (shape={jacobians.data.J_cts_offsets.shape}): {jacobians.data.J_cts_offsets}")
            print(f"J_dofs_data: shape={jacobians.data.J_dofs_data.shape}")
            print(f"J_cts_data: shape={jacobians.data.J_cts_data.shape}")

        # Check the allocations of Jacobians
        model_num_cts = (
            model.size.sum_of_num_joint_cts + limits.model_max_limits_host + 3 * contacts.model_max_contacts_host
        )
        self.assertEqual(jacobians.data.J_dofs_offsets.size, 1)
        self.assertEqual(jacobians.data.J_cts_offsets.size, 1)
        self.assertEqual(jacobians.data.J_dofs_offsets.numpy()[0], 0)
        self.assertEqual(jacobians.data.J_cts_offsets.numpy()[0], 0)
        self.assertEqual(
            jacobians.data.J_dofs_data.shape, (model.size.sum_of_num_joint_dofs * model.size.sum_of_num_body_dofs,)
        )
        self.assertEqual(jacobians.data.J_cts_data.shape, (model_num_cts * model.size.sum_of_num_body_dofs,))

    def test_05_allocate_homogeneous_dense_system_jacobians(self):
        # Problem constants
        num_worlds = 3
        max_world_contacts = 12

        # Construct the model description using the ModelBuilderKamino
        builder = make_homogeneous_builder(num_worlds=num_worlds, build_fn=build_boxes_fourbar)

        # Create the model from the builder
        model = builder.finalize(device=self.default_device)
        if self.verbose:
            print("")  # Add a newline for better readability
            print(f"model.size.sum_of_num_bodies: {model.size.sum_of_num_bodies}")
            print(f"model.size.sum_of_num_joints: {model.size.sum_of_num_joints}")
            print(f"model.size.sum_of_num_joint_cts: {model.size.sum_of_num_joint_cts}")
            print(f"model.size.sum_of_num_joint_dofs: {model.size.sum_of_num_joint_dofs}")

        # Construct and allocate the limits container
        limits = LimitsKamino(model=model, device=self.default_device)
        if self.verbose:
            print("limits.model_max_limits_host: ", limits.model_max_limits_host)
            print("limits.world_max_limits_host: ", limits.world_max_limits_host)

        # Set the contact allocation capacities
        required_world_max_contacts = [max_world_contacts] * builder.num_worlds
        if self.verbose:
            print("required_world_max_contacts: ", required_world_max_contacts)

        # Construct and allocate the contacts container
        contacts = ContactsKamino(capacity=required_world_max_contacts, device=self.default_device)
        if self.verbose:
            print("contacts.default_max_world_contacts: ", contacts.default_max_world_contacts)
            print("contacts.model_max_contacts_host: ", contacts.model_max_contacts_host)
            print("contacts.world_max_contacts_host: ", contacts.world_max_contacts_host)
            print("contacts.world_max_contacts_host: ", contacts.world_max_contacts_host)

        # Build model info
        make_unilateral_constraints_info(model, model.data(), limits, contacts)

        # Create the Jacobians container
        jacobians = DenseSystemJacobians(model=model, limits=limits, contacts=contacts, device=self.default_device)
        if self.verbose:
            print(f"J_dofs_offsets (shape={jacobians.data.J_dofs_offsets.shape}): {jacobians.data.J_dofs_offsets}")
            print(f"J_cts_offsets (shape={jacobians.data.J_cts_offsets.shape}): {jacobians.data.J_cts_offsets}")
            print(f"J_dofs_data: shape={jacobians.data.J_dofs_data.shape}")
            print(f"J_cts_data: shape={jacobians.data.J_cts_data.shape}")

        # Compute the total maximum number of constraints
        num_body_dofs = model.info.num_body_dofs.numpy().tolist()
        num_joint_dofs = model.info.num_joint_dofs.numpy().tolist()
        max_total_cts = model.info.max_total_cts.numpy().tolist()
        if self.verbose:
            print("num_body_dofs: ", num_body_dofs)
            print("max_total_cts: ", max_total_cts)
            print("num_joint_dofs: ", num_joint_dofs)

        # Compute Jacobian sizes
        J_dofs_size: list[int] = [0] * num_worlds
        J_cts_size: list[int] = [0] * num_worlds
        for w in range(num_worlds):
            J_dofs_size[w] = num_joint_dofs[w] * num_body_dofs[w]
            J_cts_size[w] = max_total_cts[w] * num_body_dofs[w]

        # Compute Jacobian offsets
        J_dofs_offsets: list[int] = [0] + [sum(J_dofs_size[:w]) for w in range(1, num_worlds)]
        J_cts_offsets: list[int] = [0] + [sum(J_cts_size[:w]) for w in range(1, num_worlds)]

        # Check the allocations of Jacobians
        self.assertEqual(jacobians.data.J_dofs_offsets.size, num_worlds)
        self.assertEqual(jacobians.data.J_cts_offsets.size, num_worlds)
        J_dofs_mio_np = jacobians.data.J_dofs_offsets.numpy()
        J_cts_mio_np = jacobians.data.J_cts_offsets.numpy()
        for w in range(num_worlds):
            self.assertEqual(J_dofs_mio_np[w], J_dofs_offsets[w])
            self.assertEqual(J_cts_mio_np[w], J_cts_offsets[w])
        self.assertEqual(jacobians.data.J_dofs_data.size, sum(J_dofs_size))
        self.assertEqual(jacobians.data.J_cts_data.size, sum(J_cts_size))

    def test_06_allocate_heterogeneous_dense_system_jacobians(self):
        # Problem constants
        max_world_contacts = 12

        # Construct the model description using the ModelBuilderKamino
        builder = make_basics_heterogeneous_builder()
        num_worlds = builder.num_worlds

        # Create the model from the builder
        model = builder.finalize(device=self.default_device)
        if self.verbose:
            print("")  # Add a newline for better readability
            print(f"model.size.sum_of_num_bodies: {model.size.sum_of_num_bodies}")
            print(f"model.size.sum_of_num_joints: {model.size.sum_of_num_joints}")
            print(f"model.size.sum_of_num_joint_cts: {model.size.sum_of_num_joint_cts}")
            print(f"model.size.sum_of_num_joint_dofs: {model.size.sum_of_num_joint_dofs}")

        # Construct and allocate the limits container
        limits = LimitsKamino(model=model, device=self.default_device)
        if self.verbose:
            print("limits.model_max_limits_host: ", limits.model_max_limits_host)
            print("limits.world_max_limits_host: ", limits.world_max_limits_host)

        # Set the contact allocation capacities
        required_world_max_contacts = [max_world_contacts] * builder.num_worlds
        if self.verbose:
            print("required_world_max_contacts: ", required_world_max_contacts)

        # Construct and allocate the contacts container
        contacts = ContactsKamino(capacity=required_world_max_contacts, device=self.default_device)
        if self.verbose:
            print("contacts.default_max_world_contacts: ", contacts.default_max_world_contacts)
            print("contacts.model_max_contacts_host: ", contacts.model_max_contacts_host)
            print("contacts.world_max_contacts_host: ", contacts.world_max_contacts_host)

        # Build model info
        make_unilateral_constraints_info(model, model.data(), limits, contacts)

        # Create the Jacobians container
        jacobians = DenseSystemJacobians(model=model, limits=limits, contacts=contacts)
        if self.verbose:
            print(f"J_dofs_offsets (shape={jacobians.data.J_dofs_offsets.shape}): {jacobians.data.J_dofs_offsets}")
            print(f"J_cts_offsets (shape={jacobians.data.J_cts_offsets.shape}): {jacobians.data.J_cts_offsets}")
            print(f"J_dofs_data: shape={jacobians.data.J_dofs_data.shape}")
            print(f"J_cts_data: shape={jacobians.data.J_cts_data.shape}")

        # Compute the total maximum number of constraints
        num_body_dofs = model.info.num_body_dofs.numpy().tolist()
        num_joint_dofs = model.info.num_joint_dofs.numpy().tolist()
        max_total_cts = model.info.max_total_cts.numpy().tolist()
        if self.verbose:
            print("num_body_dofs: ", num_body_dofs)
            print("max_total_cts: ", max_total_cts)
            print("num_joint_dofs: ", num_joint_dofs)

        # Compute Jacobian sizes
        J_dofs_size: list[int] = [0] * num_worlds
        J_cts_size: list[int] = [0] * num_worlds
        for w in range(num_worlds):
            J_dofs_size[w] = num_joint_dofs[w] * num_body_dofs[w]
            J_cts_size[w] = max_total_cts[w] * num_body_dofs[w]

        # Compute Jacobian offsets
        J_dofs_offsets: list[int] = [0] + [sum(J_dofs_size[:w]) for w in range(1, num_worlds)]
        J_cts_offsets: list[int] = [0] + [sum(J_cts_size[:w]) for w in range(1, num_worlds)]

        # Check the allocations of Jacobians
        self.assertEqual(jacobians.data.J_dofs_offsets.size, num_worlds)
        self.assertEqual(jacobians.data.J_cts_offsets.size, num_worlds)
        J_dofs_mio_np = jacobians.data.J_dofs_offsets.numpy()
        J_cts_mio_np = jacobians.data.J_cts_offsets.numpy()
        for w in range(num_worlds):
            self.assertEqual(J_dofs_mio_np[w], J_dofs_offsets[w])
            self.assertEqual(J_cts_mio_np[w], J_cts_offsets[w])
        self.assertEqual(jacobians.data.J_dofs_data.size, sum(J_dofs_size))
        self.assertEqual(jacobians.data.J_cts_data.size, sum(J_cts_size))

    def test_07_build_single_dense_system_jacobians(self):
        # Construct the test problem
        model, data, _state, limits, contacts = make_test_problem_fourbar(
            device=self.default_device,
            max_world_contacts=12,
            num_worlds=1,
            with_limits=True,
            with_contacts=True,
            verbose=self.verbose,
        )

        # Create the Jacobians container
        jacobians = DenseSystemJacobians(model=model, limits=limits, contacts=contacts)
        wp.synchronize()

        # Build the dense system Jacobians
        jacobians.build(model=model, data=data, limits=limits.data, contacts=contacts.data)
        wp.synchronize()

        # Reshape the flat actuation Jacobian as a matrix
        J_dofs_offsets = jacobians.data.J_dofs_offsets.numpy()
        J_dofs_flat = jacobians.data.J_dofs_data.numpy()
        njd = J_dofs_flat.size // model.size.sum_of_num_body_dofs
        J_dofs_mat = J_dofs_flat.reshape((njd, model.size.sum_of_num_body_dofs))

        # Reshape the flat constraintJacobian as a matrix
        J_cts_offsets = jacobians.data.J_cts_offsets.numpy()
        J_cts_flat = jacobians.data.J_cts_data.numpy()
        maxncts = J_cts_flat.size // model.size.sum_of_num_body_dofs
        J_cts_mat = J_cts_flat.reshape((maxncts, model.size.sum_of_num_body_dofs))

        # Check the shapes of the Jacobians
        self.assertEqual(J_dofs_offsets.size, 1)
        self.assertEqual(J_cts_offsets.size, 1)
        self.assertEqual(
            maxncts,
            model.size.sum_of_num_joint_cts + limits.model_max_limits_host + 3 * contacts.model_max_contacts_host,
        )
        self.assertEqual(njd, model.size.sum_of_num_joint_dofs)

        # Optional verbose output
        if self.verbose:
            print(f"J_cts_offsets (shape={jacobians.data.J_cts_offsets.shape}): {jacobians.data.J_cts_offsets}")
            print(f"J_cts_flat (shape={J_cts_flat.shape}):\n{J_cts_flat}")
            print(f"J_cts_mat (shape={J_cts_mat.shape}):\n{J_cts_mat}")
            print(f"J_dofs_offsets (shape={jacobians.data.J_dofs_offsets.shape}): {jacobians.data.J_dofs_offsets}")
            print(f"J_dofs_flat (shape={J_dofs_flat.shape}):\n{J_dofs_flat}")
            print(f"J_dofs_mat (shape={J_dofs_mat.shape}):\n{J_dofs_mat}")

    def test_08_build_homogeneous_dense_system_jacobians(self):
        # Construct the test problem
        model, data, _state, limits, contacts = make_test_problem_fourbar(
            device=self.default_device,
            max_world_contacts=12,
            num_worlds=3,
            with_limits=True,
            with_contacts=True,
            verbose=self.verbose,
        )

        # Create the Jacobians container
        jacobians = DenseSystemJacobians(model=model, limits=limits, contacts=contacts)
        wp.synchronize()

        # Build the dense system Jacobians
        jacobians.build(model=model, data=data, limits=limits.data, contacts=contacts.data)
        wp.synchronize()

        # Extract the Jacobian matrices
        J_cts = extract_cts_jacobians(model=model, limits=limits, contacts=contacts, jacobians=jacobians)
        J_dofs = extract_dofs_jacobians(model=model, jacobians=jacobians)
        for w in range(model.size.num_worlds):
            msg.info("[world='%d']: J_cts:\n%s", w, J_cts[w])
            msg.info("[world='%d']: J_dofs:\n%s", w, J_dofs[w])

    def test_09_build_heterogeneous_dense_system_jacobians(self):
        # Construct the test problem
        model, data, _state, limits, contacts = make_test_problem_heterogeneous(
            device=self.default_device,
            max_world_contacts=12,
            with_limits=True,
            with_contacts=True,
            with_implicit_joints=True,
            verbose=self.verbose,
        )

        # Create the Jacobians container
        jacobians = DenseSystemJacobians(model=model, limits=limits, contacts=contacts)
        wp.synchronize()

        # Build the dense system Jacobians
        jacobians.build(model=model, data=data, limits=limits.data, contacts=contacts.data)
        wp.synchronize()

        # Extract the Jacobian matrices
        J_cts = extract_cts_jacobians(model=model, limits=limits, contacts=contacts, jacobians=jacobians)
        J_dofs = extract_dofs_jacobians(model=model, jacobians=jacobians)
        for w in range(model.size.num_worlds):
            msg.info("[world='%d']: J_cts:\n%s", w, J_cts[w])
            msg.info("[world='%d']: J_dofs:\n%s", w, J_dofs[w])


class TestKinematicsSparseSystemJacobians(unittest.TestCase):
    def setUp(self):
        # Configs
        if not test_context.setup_done:
            setup_tests(clear_cache=False)
        self.default_device = wp.get_device(test_context.device)
        self.verbose = test_context.verbose  # Set to True for verbose output
        self.epsilon = 1e-6  # Threshold for sparse-dense comparison test

        # Set debug-level logging to print verbose test output to console
        if self.verbose:
            print("\n")  # Add newline before test output for better readability
            msg.set_log_level(msg.LogLevel.INFO)
        else:
            msg.reset_log_level()

    def tearDown(self):
        self.default_device = None
        if self.verbose:
            msg.reset_log_level()

    ###
    # Helpers
    ###

    def _compare_dense_sparse_jacobians(
        self,
        model: ModelKamino,
        limits: LimitsKamino | None,
        contacts: ContactsKamino | None,
        jacobians_dense: DenseSystemJacobians,
        jacobians_sparse: SparseSystemJacobians,
    ):
        # Reshape the dense Jacobian data as a matrices
        J_cts_dense = extract_cts_jacobians(
            model=model, limits=limits, contacts=contacts, jacobians=jacobians_dense, verbose=self.verbose
        )
        J_dofs_dense = extract_dofs_jacobians(model=model, jacobians=jacobians_dense, verbose=self.verbose)

        # Get the (dense) numpy version of the sparse Jacobians
        J_dofs_sparse = jacobians_sparse._J_dofs.bsm.numpy()
        J_cts_sparse = jacobians_sparse._J_cts.bsm.numpy()

        self.assertEqual(len(J_cts_dense), len(J_cts_sparse))
        self.assertEqual(len(J_dofs_dense), len(J_dofs_sparse))

        # Check that Jacobians match
        for mat_id in range(len(J_cts_dense)):
            if J_dofs_dense[mat_id].size > 0:
                diff_J_dofs = J_dofs_dense[mat_id] - J_dofs_sparse[mat_id]
                self.assertLess(np.max(np.abs(diff_J_dofs)), self.epsilon)

            diff_J_cts = J_cts_dense[mat_id][: J_cts_sparse[mat_id].shape[0], :] - J_cts_sparse[mat_id]
            self.assertLess(np.max(np.abs(diff_J_cts)), self.epsilon)

            # Extra entries in dense constraint Jacobian need to be zero
            if J_cts_dense[mat_id].shape[0] > J_cts_sparse[mat_id].shape[0]:
                self.assertEqual(np.max(np.abs(J_cts_dense[mat_id][J_cts_sparse[mat_id].shape[0] :, :])), 0)

    def _compare_row_col_major_jacobians(
        self,
        jacobians: SparseSystemJacobians,
        jacobians_col_major: ColMajorSparseConstraintJacobians,
    ):
        # Get the (dense) numpy version of the Jacobians
        J_cts_row_major = jacobians._J_cts.bsm.numpy()
        J_cts_col_major = jacobians_col_major.bsm.numpy()

        self.assertEqual(len(J_cts_row_major), len(J_cts_col_major))

        # Check that Jacobians match
        for mat_id in range(len(J_cts_row_major)):
            diff_J_cts = J_cts_row_major[mat_id] - J_cts_col_major[mat_id]
            max_diff = np.max(np.abs(diff_J_cts))
            if max_diff > self.epsilon and self.verbose:
                msg.warning(f"[{mat_id}] J_cts_row_major:\n{J_cts_row_major[mat_id]}")
                msg.warning(f"[{mat_id}] J_cts_col_major:\n{J_cts_col_major[mat_id]}")
            self.assertLess(max_diff, self.epsilon)

    ###
    # Construction
    ###

    def test_01_allocate_single_sparse_system_jacobians_only_joints(self):
        # Construct the test problem
        model, *_ = make_test_problem_fourbar(
            device=self.default_device,
            max_world_contacts=12,
            num_worlds=1,
            with_limits=False,
            with_contacts=False,
            verbose=self.verbose,
        )

        # Create the sparse Jacobians
        jacobians = SparseSystemJacobians(model=model)
        self.assertIs(jacobians._J_cts.bsm.device, model.device)
        self.assertIs(jacobians._J_cts.device, model.device)
        if self.verbose:
            print(f"J_cts max_dims (shape={jacobians._J_cts.bsm.max_dims.shape}): {jacobians._J_cts.bsm.max_dims}")
            print(f"J_cts dims (shape={jacobians._J_cts.bsm.dims.shape}): {jacobians._J_cts.bsm.dims}")
            print(f"J_cts max_nzb (shape={jacobians._J_cts.bsm.max_nzb.shape}): {jacobians._J_cts.bsm.max_nzb}")
            print(f"J_dofs max_dims (shape={jacobians._J_dofs.bsm.max_dims.shape}): {jacobians._J_dofs.bsm.max_dims}")
            print(f"J_dofs dims (shape={jacobians._J_dofs.bsm.dims.shape}): {jacobians._J_dofs.bsm.dims}")
            print(f"J_dofs max_nzb (shape={jacobians._J_dofs.bsm.max_nzb.shape}): {jacobians._J_dofs.bsm.max_nzb}")

        # Check the allocation of Jacobians
        model_num_cts = model.size.sum_of_num_joint_cts
        model_num_dofs = model.size.sum_of_num_joint_dofs
        model_num_bodies = model.size.sum_of_num_bodies
        self.assertEqual(jacobians._J_cts.bsm.num_matrices, 1)
        self.assertEqual(jacobians._J_dofs.bsm.num_matrices, 1)
        self.assertTrue((jacobians._J_cts.bsm.max_dims.numpy() == [[model_num_cts, 6 * model_num_bodies]]).all())
        self.assertTrue((jacobians._J_dofs.bsm.max_dims.numpy() == [[model_num_dofs, 6 * model_num_bodies]]).all())
        self.assertEqual(jacobians._J_cts.bsm.max_nzb.numpy()[0], 2 * model_num_cts)

    def test_02_allocate_single_sparse_system_jacobians_with_limits(self):
        # Construct the test problem
        model, _data, _state, limits, *_ = make_test_problem_fourbar(
            device=self.default_device,
            num_worlds=1,
            with_limits=True,
            with_contacts=False,
            verbose=self.verbose,
        )

        # Create the Jacobians container
        jacobians = SparseSystemJacobians(model=model, limits=limits)
        if self.verbose:
            print(f"J_cts max_dims (shape={jacobians._J_cts.bsm.max_dims.shape}): {jacobians._J_cts.bsm.max_dims}")
            print(f"J_cts dims (shape={jacobians._J_cts.bsm.dims.shape}): {jacobians._J_cts.bsm.dims}")
            print(f"J_cts max_nzb (shape={jacobians._J_cts.bsm.max_nzb.shape}): {jacobians._J_cts.bsm.max_nzb}")
            print(f"J_dofs max_dims (shape={jacobians._J_dofs.bsm.max_dims.shape}): {jacobians._J_dofs.bsm.max_dims}")
            print(f"J_dofs dims (shape={jacobians._J_dofs.bsm.dims.shape}): {jacobians._J_dofs.bsm.dims}")
            print(f"J_dofs max_nzb (shape={jacobians._J_dofs.bsm.max_nzb.shape}): {jacobians._J_dofs.bsm.max_nzb}")

        # Check the allocation of Jacobians
        model_num_cts = model.size.sum_of_num_joint_cts + limits.model_max_limits_host
        model_num_dofs = model.size.sum_of_num_joint_dofs
        model_num_bodies = model.size.sum_of_num_bodies
        self.assertEqual(jacobians._J_cts.bsm.num_matrices, 1)
        self.assertEqual(jacobians._J_dofs.bsm.num_matrices, 1)
        self.assertTrue((jacobians._J_cts.bsm.max_dims.numpy() == [[model_num_cts, 6 * model_num_bodies]]).all())
        self.assertTrue((jacobians._J_dofs.bsm.max_dims.numpy() == [[model_num_dofs, 6 * model_num_bodies]]).all())
        self.assertEqual(jacobians._J_cts.bsm.max_nzb.numpy()[0], 2 * model_num_cts)

    def test_03_allocate_single_sparse_system_jacobians_with_contacts(self):
        # Construct the test problem
        model, _data, _state, _limits, contacts = make_test_problem_fourbar(
            device=self.default_device,
            max_world_contacts=12,
            num_worlds=1,
            with_limits=False,
            with_contacts=True,
            verbose=self.verbose,
        )

        # Create the Jacobians container
        jacobians = SparseSystemJacobians(model=model, contacts=contacts)
        if self.verbose:
            print(f"J_cts max_dims (shape={jacobians._J_cts.bsm.max_dims.shape}): {jacobians._J_cts.bsm.max_dims}")
            print(f"J_cts dims (shape={jacobians._J_cts.bsm.dims.shape}): {jacobians._J_cts.bsm.dims}")
            print(f"J_cts max_nzb (shape={jacobians._J_cts.bsm.max_nzb.shape}): {jacobians._J_cts.bsm.max_nzb}")
            print(f"J_dofs max_dims (shape={jacobians._J_dofs.bsm.max_dims.shape}): {jacobians._J_dofs.bsm.max_dims}")
            print(f"J_dofs dims (shape={jacobians._J_dofs.bsm.dims.shape}): {jacobians._J_dofs.bsm.dims}")
            print(f"J_dofs max_nzb (shape={jacobians._J_dofs.bsm.max_nzb.shape}): {jacobians._J_dofs.bsm.max_nzb}")

        # Check the allocation of Jacobians
        model_num_cts = model.size.sum_of_num_joint_cts + 3 * contacts.model_max_contacts_host
        model_num_dofs = model.size.sum_of_num_joint_dofs
        model_num_bodies = model.size.sum_of_num_bodies
        self.assertEqual(jacobians._J_cts.bsm.num_matrices, 1)
        self.assertEqual(jacobians._J_dofs.bsm.num_matrices, 1)
        self.assertTrue((jacobians._J_cts.bsm.max_dims.numpy() == [[model_num_cts, 6 * model_num_bodies]]).all())
        self.assertTrue((jacobians._J_dofs.bsm.max_dims.numpy() == [[model_num_dofs, 6 * model_num_bodies]]).all())
        self.assertEqual(jacobians._J_cts.bsm.max_nzb.numpy()[0], 2 * model_num_cts)

    def test_04_allocate_single_sparse_system_jacobians_with_limits_and_contacts(self):
        # Construct the test problem
        model, _data, _state, limits, contacts = make_test_problem_fourbar(
            device=self.default_device,
            max_world_contacts=12,
            num_worlds=1,
            with_limits=True,
            with_contacts=True,
            verbose=self.verbose,
        )

        # Create the Jacobians container
        jacobians = SparseSystemJacobians(model=model, limits=limits, contacts=contacts)
        if self.verbose:
            print(f"J_cts max_dims (shape={jacobians._J_cts.bsm.max_dims.shape}): {jacobians._J_cts.bsm.max_dims}")
            print(f"J_cts dims (shape={jacobians._J_cts.bsm.dims.shape}): {jacobians._J_cts.bsm.dims}")
            print(f"J_cts max_nzb (shape={jacobians._J_cts.bsm.max_nzb.shape}): {jacobians._J_cts.bsm.max_nzb}")
            print(f"J_dofs max_dims (shape={jacobians._J_dofs.bsm.max_dims.shape}): {jacobians._J_dofs.bsm.max_dims}")
            print(f"J_dofs dims (shape={jacobians._J_dofs.bsm.dims.shape}): {jacobians._J_dofs.bsm.dims}")
            print(f"J_dofs max_nzb (shape={jacobians._J_dofs.bsm.max_nzb.shape}): {jacobians._J_dofs.bsm.max_nzb}")

        # Check the allocation of Jacobians
        model_num_cts = (
            model.size.sum_of_num_joint_cts + limits.model_max_limits_host + 3 * contacts.model_max_contacts_host
        )
        model_num_dofs = model.size.sum_of_num_joint_dofs
        model_num_bodies = model.size.sum_of_num_bodies
        self.assertEqual(jacobians._J_cts.bsm.num_matrices, 1)
        self.assertEqual(jacobians._J_dofs.bsm.num_matrices, 1)
        self.assertTrue((jacobians._J_cts.bsm.max_dims.numpy() == [[model_num_cts, 6 * model_num_bodies]]).all())
        self.assertTrue((jacobians._J_dofs.bsm.max_dims.numpy() == [[model_num_dofs, 6 * model_num_bodies]]).all())
        self.assertEqual(jacobians._J_cts.bsm.max_nzb.numpy()[0], 2 * model_num_cts)

    def test_05_allocate_homogeneous_sparse_system_jacobians(self):
        # Construct the test problem
        model, _data, _state, limits, contacts = make_test_problem_fourbar(
            device=self.default_device,
            max_world_contacts=12,
            num_worlds=3,
            with_limits=True,
            with_contacts=True,
            verbose=self.verbose,
        )

        # Create the Jacobians container
        jacobians = SparseSystemJacobians(model=model, limits=limits, contacts=contacts)
        if self.verbose:
            print(f"J_cts max_dims (shape={jacobians._J_cts.bsm.max_dims.shape}): {jacobians._J_cts.bsm.max_dims}")
            print(f"J_cts dims (shape={jacobians._J_cts.bsm.dims.shape}): {jacobians._J_cts.bsm.dims}")
            print(f"J_cts max_nzb (shape={jacobians._J_cts.bsm.max_nzb.shape}): {jacobians._J_cts.bsm.max_nzb}")
            print(f"J_dofs max_dims (shape={jacobians._J_dofs.bsm.max_dims.shape}): {jacobians._J_dofs.bsm.max_dims}")
            print(f"J_dofs dims (shape={jacobians._J_dofs.bsm.dims.shape}): {jacobians._J_dofs.bsm.dims}")
            print(f"J_dofs max_nzb (shape={jacobians._J_dofs.bsm.max_nzb.shape}): {jacobians._J_dofs.bsm.max_nzb}")

        # Check the allocation of Jacobians
        num_body_dofs = model.info.num_body_dofs.numpy().tolist()
        num_joint_dofs = model.info.num_joint_dofs.numpy().tolist()
        max_total_cts = model.info.max_total_cts.numpy().tolist()
        self.assertEqual(jacobians._J_cts.bsm.num_matrices, model.size.num_worlds)
        self.assertEqual(jacobians._J_dofs.bsm.num_matrices, model.size.num_worlds)
        self.assertTrue(
            (
                jacobians._J_cts.bsm.max_dims.numpy()
                == [[max_total_cts[w], num_body_dofs[w]] for w in range(model.size.num_worlds)]
            ).all()
        )
        self.assertTrue(
            (
                jacobians._J_dofs.bsm.max_dims.numpy()
                == [[num_joint_dofs[w], num_body_dofs[w]] for w in range(model.size.num_worlds)]
            ).all()
        )
        self.assertTrue(
            (jacobians._J_cts.bsm.max_nzb.numpy() == [2 * max_total_cts[w] for w in range(model.size.num_worlds)]).all()
        )

    def test_06_allocate_heterogeneous_sparse_system_jacobians(self):
        # Construct the test problem
        model, _data, _state, limits, contacts = make_test_problem_heterogeneous(
            device=self.default_device,
            max_world_contacts=12,
            with_limits=True,
            with_contacts=True,
            with_implicit_joints=True,
            verbose=self.verbose,
        )

        # Create the Jacobians container
        jacobians = SparseSystemJacobians(model=model, limits=limits, contacts=contacts, device=self.default_device)
        if self.verbose:
            print(f"J_cts max_dims (shape={jacobians._J_cts.bsm.max_dims.shape}): {jacobians._J_cts.bsm.max_dims}")
            print(f"J_cts dims (shape={jacobians._J_cts.bsm.dims.shape}): {jacobians._J_cts.bsm.dims}")
            print(f"J_cts max_nzb (shape={jacobians._J_cts.bsm.max_nzb.shape}): {jacobians._J_cts.bsm.max_nzb}")
            print(f"J_dofs max_dims (shape={jacobians._J_dofs.bsm.max_dims.shape}): {jacobians._J_dofs.bsm.max_dims}")
            print(f"J_dofs dims (shape={jacobians._J_dofs.bsm.dims.shape}): {jacobians._J_dofs.bsm.dims}")
            print(f"J_dofs max_nzb (shape={jacobians._J_dofs.bsm.max_nzb.shape}): {jacobians._J_dofs.bsm.max_nzb}")

        # Check the allocation of Jacobians
        num_body_dofs = model.info.num_body_dofs.numpy().tolist()
        num_joint_dofs = model.info.num_joint_dofs.numpy().tolist()
        max_total_cts = model.info.max_total_cts.numpy().tolist()
        self.assertEqual(jacobians._J_cts.bsm.num_matrices, model.size.num_worlds)
        self.assertEqual(jacobians._J_dofs.bsm.num_matrices, model.size.num_worlds)
        self.assertTrue(
            (
                jacobians._J_cts.bsm.max_dims.numpy()
                == [[max_total_cts[w], num_body_dofs[w]] for w in range(model.size.num_worlds)]
            ).all()
        )
        self.assertTrue(
            (
                jacobians._J_dofs.bsm.max_dims.numpy()
                == [[num_joint_dofs[w], num_body_dofs[w]] for w in range(model.size.num_worlds)]
            ).all()
        )

    def test_07_build_compare_single_system_jacobians(self):
        # Construct the test problem
        model, data, *_ = make_test_problem_fourbar(
            device=self.default_device,
            max_world_contacts=12,
            num_worlds=1,
            with_limits=False,
            with_contacts=False,
            verbose=self.verbose,
        )

        # Create the Jacobians container
        jacobians = SparseSystemJacobians(model=model)
        jacobians_dense = DenseSystemJacobians(model=model)
        wp.synchronize()

        # Build the system Jacobians
        jacobians.build(model=model, data=data)
        jacobians_dense.build(model=model, data=data)
        wp.synchronize()

        # Check that Jacobians match
        self._compare_dense_sparse_jacobians(model, None, None, jacobians_dense, jacobians)

    def test_08_build_compare_single_system_jacobians_with_limits(self):
        # Construct the test problem
        model, data, _state, limits, _contacts = make_test_problem_fourbar(
            device=self.default_device,
            num_worlds=1,
            with_limits=True,
            with_contacts=False,
            verbose=self.verbose,
        )

        # Create the Jacobians container
        jacobians = SparseSystemJacobians(model=model, limits=limits)
        jacobians_dense = DenseSystemJacobians(model=model, limits=limits)
        wp.synchronize()

        # Build the system Jacobians
        jacobians.build(model=model, data=data, limits=limits.data)
        jacobians_dense.build(model=model, data=data, limits=limits.data)
        wp.synchronize()

        # Check that Jacobians match
        self._compare_dense_sparse_jacobians(model, limits, None, jacobians_dense, jacobians)

    def test_09_build_compare_single_system_jacobians_with_contacts(self):
        # Construct the test problem
        model, data, _state, _limits, contacts = make_test_problem_fourbar(
            device=self.default_device,
            max_world_contacts=12,
            num_worlds=1,
            with_limits=False,
            with_contacts=True,
            verbose=self.verbose,
        )

        # Create the Jacobians container
        jacobians = SparseSystemJacobians(model=model, contacts=contacts)
        jacobians_dense = DenseSystemJacobians(model=model, contacts=contacts)
        wp.synchronize()

        # Build the system Jacobians
        jacobians.build(model=model, data=data, contacts=contacts.data)
        jacobians_dense.build(model=model, data=data, contacts=contacts.data)
        wp.synchronize()

        # Check that Jacobians match
        self._compare_dense_sparse_jacobians(model, None, contacts, jacobians_dense, jacobians)

    def test_10_build_compare_single_system_jacobians_with_limits_and_contacts(self):
        # Construct the test problem
        model, data, _state, limits, contacts = make_test_problem_fourbar(
            device=self.default_device,
            max_world_contacts=12,
            num_worlds=1,
            with_limits=True,
            with_contacts=True,
            verbose=self.verbose,
        )

        # Create the Jacobians container
        jacobians = SparseSystemJacobians(model=model, limits=limits, contacts=contacts)
        jacobians_dense = DenseSystemJacobians(model=model, limits=limits, contacts=contacts)
        wp.synchronize()

        # Build the system Jacobians
        jacobians.build(model=model, data=data, limits=limits.data, contacts=contacts.data)
        jacobians_dense.build(model=model, data=data, limits=limits.data, contacts=contacts.data)
        wp.synchronize()

        # Check that Jacobians match
        self._compare_dense_sparse_jacobians(model, limits, contacts, jacobians_dense, jacobians)

    def test_11_build_compare_homogeneous_system_jacobians(self):
        # Construct the test problem
        model, data, _state, limits, contacts = make_test_problem_fourbar(
            device=self.default_device,
            max_world_contacts=12,
            num_worlds=3,
            with_limits=True,
            with_contacts=True,
            verbose=self.verbose,
        )

        # Create the Jacobians container
        jacobians = SparseSystemJacobians(model=model, limits=limits, contacts=contacts)
        jacobians_dense = DenseSystemJacobians(model=model, limits=limits, contacts=contacts)
        wp.synchronize()

        # Build the system Jacobians
        jacobians.build(model=model, data=data, limits=limits.data, contacts=contacts.data)
        jacobians_dense.build(model=model, data=data, limits=limits.data, contacts=contacts.data)
        wp.synchronize()

        # Check that Jacobians match
        self._compare_dense_sparse_jacobians(model, limits, contacts, jacobians_dense, jacobians)

    def test_12_build_compare_heterogeneous_system_jacobians(self):
        # Construct the test problem
        model, data, _state, limits, contacts = make_test_problem_heterogeneous(
            device=self.default_device,
            max_world_contacts=12,
            with_limits=True,
            with_contacts=True,
            with_implicit_joints=True,
            verbose=self.verbose,
        )

        # Create the Jacobians container
        jacobians_sparse = SparseSystemJacobians(model=model, limits=limits, contacts=contacts)
        jacobians_dense = DenseSystemJacobians(model=model, limits=limits, contacts=contacts)
        wp.synchronize()

        # Build the system Jacobians
        jacobians_sparse.build(model=model, data=data, limits=limits.data, contacts=contacts.data)
        jacobians_dense.build(model=model, data=data, limits=limits.data, contacts=contacts.data)
        wp.synchronize()

        # Check that Jacobians match
        self._compare_dense_sparse_jacobians(model, limits, contacts, jacobians_dense, jacobians_sparse)

    def test_13_build_col_major_single_system_jacobians(self):
        # Construct the test problem
        model, data, *_ = make_test_problem_fourbar(
            device=self.default_device,
            max_world_contacts=12,
            num_worlds=1,
            with_limits=False,
            with_contacts=False,
            verbose=self.verbose,
        )

        # Create the Jacobians container
        jacobians = SparseSystemJacobians(model=model)
        wp.synchronize()

        # Build the system Jacobians
        jacobians.build(model=model, data=data)
        wp.synchronize()

        # Build column-major constraint Jacobian version
        jacobian_col_maj = ColMajorSparseConstraintJacobians(model=model, jacobians=jacobians)
        jacobian_col_maj.update(model=model, jacobians=jacobians)

        # Check that Jacobians match
        self._compare_row_col_major_jacobians(jacobians, jacobian_col_maj)

    def test_14_build_col_major_single_system_jacobians_with_limits(self):
        # Construct the test problem
        model, data, _state, limits, _contacts = make_test_problem_fourbar(
            device=self.default_device,
            max_world_contacts=12,
            num_worlds=1,
            with_limits=True,
            with_contacts=False,
            verbose=self.verbose,
        )

        # Create the Jacobians container
        jacobians = SparseSystemJacobians(model=model, limits=limits)
        wp.synchronize()

        # Build the system Jacobians
        jacobians.build(model=model, data=data, limits=limits.data)
        wp.synchronize()

        # Build column-major constraint Jacobian version
        jacobian_col_maj = ColMajorSparseConstraintJacobians(model=model, limits=limits, jacobians=jacobians)
        jacobian_col_maj.update(model=model, jacobians=jacobians, limits=limits)

        # Check that Jacobians match
        self._compare_row_col_major_jacobians(jacobians, jacobian_col_maj)

    def test_15_build_col_major_single_system_jacobians_with_contacts(self):
        # Construct the test problem
        model, data, _state, _limits, contacts = make_test_problem_fourbar(
            device=self.default_device,
            max_world_contacts=12,
            num_worlds=1,
            with_limits=False,
            with_contacts=True,
            verbose=self.verbose,
        )

        # Create the Jacobians container
        jacobians = SparseSystemJacobians(model=model, contacts=contacts, device=self.default_device)
        wp.synchronize()

        # Build the system Jacobians
        jacobians.build(model=model, data=data, contacts=contacts.data)
        wp.synchronize()

        # Build column-major constraint Jacobian version
        jacobian_col_maj = ColMajorSparseConstraintJacobians(model=model, contacts=contacts, jacobians=jacobians)
        jacobian_col_maj.update(model=model, jacobians=jacobians, contacts=contacts)

        # Check that Jacobians match
        self._compare_row_col_major_jacobians(jacobians, jacobian_col_maj)

    def test_16_build_col_major_single_system_jacobians_with_limits_and_contacts(self):
        # Construct the test problem
        model, data, _state, limits, contacts = make_test_problem_fourbar(
            device=self.default_device,
            max_world_contacts=12,
            num_worlds=1,
            with_limits=True,
            with_contacts=True,
            verbose=self.verbose,
        )

        # Create the Jacobians container
        jacobians = SparseSystemJacobians(model=model, limits=limits, contacts=contacts, device=self.default_device)
        wp.synchronize()

        # Build the system Jacobians
        jacobians.build(model=model, data=data, limits=limits.data, contacts=contacts.data)
        wp.synchronize()

        # Build column-major constraint Jacobian version
        jacobian_col_maj = ColMajorSparseConstraintJacobians(
            model=model, limits=limits, contacts=contacts, jacobians=jacobians
        )
        jacobian_col_maj.update(model=model, jacobians=jacobians, limits=limits, contacts=contacts)

        # Check that Jacobians match
        self._compare_row_col_major_jacobians(jacobians, jacobian_col_maj)

    def test_17_build_col_major_homogeneous_system_jacobians(self):
        # Construct the test problem
        model, data, _state, limits, contacts = make_test_problem_fourbar(
            device=self.default_device,
            max_world_contacts=12,
            num_worlds=3,
            with_limits=True,
            with_contacts=True,
            verbose=self.verbose,
        )

        # Create the Jacobians container
        jacobians = SparseSystemJacobians(model=model, limits=limits, contacts=contacts)
        wp.synchronize()

        # Build the system Jacobians
        jacobians.build(model=model, data=data, limits=limits.data, contacts=contacts.data)
        wp.synchronize()

        # Build column-major constraint Jacobian version
        jacobian_col_maj = ColMajorSparseConstraintJacobians(
            model=model, limits=limits, contacts=contacts, jacobians=jacobians
        )
        jacobian_col_maj.update(model=model, jacobians=jacobians, limits=limits, contacts=contacts)

        # Check that Jacobians match
        self._compare_row_col_major_jacobians(jacobians, jacobian_col_maj)

    def test_18_build_col_major_heterogeneous_system_jacobians(self):
        # Construct the test problem
        model, data, _state, limits, contacts = make_test_problem_heterogeneous(
            device=self.default_device,
            max_world_contacts=12,
            with_limits=True,
            with_contacts=True,
            with_implicit_joints=True,
            verbose=self.verbose,
        )

        # Create the Jacobians container
        jacobians = SparseSystemJacobians(model=model, limits=limits, contacts=contacts)
        wp.synchronize()

        # Build the system Jacobians
        jacobians.build(model=model, data=data, limits=limits.data, contacts=contacts.data)
        wp.synchronize()

        # Build column-major constraint Jacobian version
        jacobian_col_maj = ColMajorSparseConstraintJacobians(
            model=model, limits=limits, contacts=contacts, jacobians=jacobians
        )
        jacobian_col_maj.update(model=model, jacobians=jacobians, limits=limits, contacts=contacts)

        # Check that Jacobians match
        self._compare_row_col_major_jacobians(jacobians, jacobian_col_maj)


###
# Test execution
###

if __name__ == "__main__":
    # Test setup
    setup_tests()

    # Run all tests
    unittest.main(verbosity=2)
