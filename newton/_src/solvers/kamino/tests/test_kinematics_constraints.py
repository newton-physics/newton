# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""
KAMINO: UNIT TESTS: KINEMATICS: CONSTRAINTS
"""

import unittest

import warp as wp

from newton._src.solvers.kamino._src.core.model import ModelKamino
from newton._src.solvers.kamino._src.geometry.contacts import ContactsKamino
from newton._src.solvers.kamino._src.kinematics.constraints import make_unilateral_constraints_info
from newton._src.solvers.kamino._src.kinematics.limits import LimitsKamino
from newton._src.solvers.kamino._src.models.builders.basics import (
    build_boxes_fourbar,
    make_basics_heterogeneous_builder,
)
from newton._src.solvers.kamino._src.models.builders.utils import make_homogeneous_builder
from newton._src.solvers.kamino._src.utils import logger as msg
from newton._src.solvers.kamino.tests import setup_tests, test_context
from newton._src.solvers.kamino.tests.utils.print import print_data_info, print_model_constraint_info

###
# Module configs
###

wp.set_module_options({"enable_backward": False})


###
# Tests
###


class TestKinematicsConstraints(unittest.TestCase):
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
            msg.set_log_level(msg.LogLevel.INFO)
        else:
            msg.reset_log_level()

    def tearDown(self):
        self.default_device = None
        if self.verbose:
            msg.reset_log_level()

    def test_01_single_model_make_constraints(self):
        """
        Tests the population of model info with constraint sizes and offsets for a single-world model.
        """
        # Constants
        max_world_contacts = 20

        # Construct the model description using the ModelBuilderKamino
        builder = build_boxes_fourbar(dynamic_joints=True, implicit_pd=True)

        # Create the model from the builder
        model: ModelKamino = builder.finalize(device=self.default_device)
        msg.info(f"model.joints.cts_offset:\n{model.joints.cts_offset}")
        msg.info(f"model.joints.dynamic_cts_offset:\n{model.joints.dynamic_cts_offset}")
        msg.info(f"model.joints.kinematic_cts_offset:\n{model.joints.kinematic_cts_offset}")

        # Create a model data
        data = model.data(device=self.default_device)

        # Create a  limits container
        limits = LimitsKamino(model=model)
        if self.verbose:
            print("")
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

        # Create the constraints info
        make_unilateral_constraints_info(
            model=model,
            data=data,
            limits=limits,
            contacts=contacts,
        )
        if self.verbose:
            print(f"model.size:\n{model.size}\n\n")
            print_model_constraint_info(model)
            print_data_info(data)

    def test_02_homogeneous_model_make_constraints(self):
        """
        Tests the population of model info with constraint sizes and offsets for a homogeneous multi-world model.
        """
        # Constants
        num_worlds: int = 10
        max_world_contacts: int = 20

        # Construct the model description using the ModelBuilderKamino
        builder = make_homogeneous_builder(
            num_worlds=num_worlds,
            build_fn=build_boxes_fourbar,
            dynamic_joints=True,
            implicit_pd=True,
        )

        # Create the model from the builder
        model: ModelKamino = builder.finalize(device=self.default_device)
        msg.info(f"model.joints.cts_offset:\n{model.joints.cts_offset}")
        msg.info(f"model.joints.dynamic_cts_offset:\n{model.joints.dynamic_cts_offset}")
        msg.info(f"model.joints.kinematic_cts_offset:\n{model.joints.kinematic_cts_offset}")

        # Create a model data
        data = model.data(device=self.default_device)

        # Create a  limits container
        limits = LimitsKamino(model=model)
        if self.verbose:
            print("")
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

        # Create the constraints info
        make_unilateral_constraints_info(
            model=model,
            data=data,
            limits=limits,
            contacts=contacts,
        )
        if self.verbose:
            print_model_constraint_info(model)
            print_data_info(data)
            print("\n===============================================================")
            print("data.info.num_limits.ptr: ", data.info.num_limits.ptr)
            print("limits.world_active_limits.ptr: ", limits.world_active_limits.ptr)
            print("data.info.num_contacts.ptr: ", data.info.num_contacts.ptr)
            print("contacts.world_active_contacts.ptr: ", contacts.world_active_contacts.ptr)

        # Check if the data info entity counters point to the same arrays as the limits and contacts containers
        self.assertTrue(data.info.num_limits.ptr, limits.world_active_limits.ptr)
        self.assertTrue(data.info.num_contacts.ptr, contacts.world_active_contacts.ptr)

        # Extract numpy arrays from the model info
        model_max_limits = model.size.sum_of_max_limits
        model_max_contacts = model.size.sum_of_max_contacts
        max_limits = model.info.max_limits.numpy()
        max_contacts = model.info.max_contacts.numpy()
        max_limit_cts = model.info.max_limit_cts.numpy()
        max_contact_cts = model.info.max_contact_cts.numpy()
        max_total_cts = model.info.max_total_cts.numpy()
        limits_offset = model.info.limits_offset.numpy()
        contacts_offset = model.info.contacts_offset.numpy()
        unilaterals_offset = model.info.unilaterals_offset.numpy()
        total_cts_offset = model.info.total_cts_offset.numpy()

        # Check the model info entries
        nj = 0
        njc = 0
        nl = 0
        nlc = 0
        nc = 0
        ncc = 0
        for i in range(num_worlds):
            self.assertEqual(model_max_limits, 4 * num_worlds)
            self.assertEqual(model_max_contacts, max_world_contacts * num_worlds)
            self.assertEqual(max_limits[i], 4)
            self.assertEqual(max_contacts[i], max_world_contacts)
            self.assertEqual(max_limit_cts[i], 4)
            self.assertEqual(max_contact_cts[i], 3 * max_world_contacts)
            self.assertEqual(max_total_cts[i], 21 + 4 + 3 * max_world_contacts)
            self.assertEqual(limits_offset[i], nl)
            self.assertEqual(contacts_offset[i], nc)
            self.assertEqual(unilaterals_offset[i], nl + nc)
            self.assertEqual(total_cts_offset[i], njc + nlc + ncc)
            nj += 4
            njc += 21
            nl += 4
            nlc += 4
            nc += max_world_contacts
            ncc += 3 * max_world_contacts

    def test_03_heterogeneous_model_make_constraints(self):
        """
        Tests the population of model info with constraint sizes and offsets for a heterogeneous multi-world model.
        """
        # Constants
        max_world_contacts = 20

        # Construct the model description using the ModelBuilderKamino
        builder = make_basics_heterogeneous_builder()

        # Create the model from the builder
        model: ModelKamino = builder.finalize(device=self.default_device)

        # Create a model data
        data = model.data(device=self.default_device)

        # Create a  limits container
        limits = LimitsKamino(model=model)
        if self.verbose:
            print("")
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

        # Create the constraints info
        make_unilateral_constraints_info(
            model=model,
            data=data,
            limits=limits,
            contacts=contacts,
        )
        if self.verbose:
            print_model_constraint_info(model)
            print_data_info(data)
            print("\n===============================================================")
            print("data.info.num_limits.ptr: ", data.info.num_limits.ptr)
            print("limits.world_active_limits.ptr: ", limits.world_active_limits.ptr)
            print("data.info.num_contacts.ptr: ", data.info.num_contacts.ptr)
            print("contacts.world_active_contacts.ptr: ", contacts.world_active_contacts.ptr)

        # Check if the data info entity counters point to the same arrays as the limits and contacts containers
        self.assertTrue(data.info.num_limits.ptr, limits.world_active_limits.ptr)
        self.assertTrue(data.info.num_contacts.ptr, contacts.world_active_contacts.ptr)


###
# Test execution
###

if __name__ == "__main__":
    # Test setup
    setup_tests()

    # Run all tests
    unittest.main(verbosity=2)
