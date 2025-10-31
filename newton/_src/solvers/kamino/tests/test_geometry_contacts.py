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
KAMINO: UNIT TESTS
"""

import unittest

import numpy as np
import warp as wp

# Module to be tested
from newton._src.solvers.kamino.geometry.contacts import Contacts

###
# Tests
###


class TestGeometryContacts(unittest.TestCase):
    def setUp(self):
        self.default_device = wp.get_device()

    def tearDown(self):
        self.default_device = None

    def test_single_default_allocation(self):
        contacts = Contacts(capacity=0, device=self.default_device)
        self.assertEqual(contacts.num_model_max_contacts, contacts.default_max_world_contacts)
        self.assertEqual(contacts.num_world_max_contacts[0], contacts.default_max_world_contacts)
        self.assertEqual(len(contacts.model_max_contacts), 1)
        self.assertEqual(len(contacts.model_num_contacts), 1)
        self.assertEqual(len(contacts.world_max_contacts), 1)
        self.assertEqual(len(contacts.world_num_contacts), 1)
        self.assertEqual(contacts.model_max_contacts.numpy()[0], contacts.default_max_world_contacts)
        self.assertEqual(contacts.model_num_contacts.numpy()[0], 0)
        self.assertEqual(contacts.world_max_contacts.numpy()[0], contacts.default_max_world_contacts)
        self.assertEqual(contacts.world_num_contacts.numpy()[0], 0)
        self.assertEqual(len(contacts.wid), contacts.default_max_world_contacts)
        self.assertEqual(len(contacts.cid), contacts.default_max_world_contacts)
        self.assertEqual(len(contacts.body_A), contacts.default_max_world_contacts)
        self.assertEqual(len(contacts.body_B), contacts.default_max_world_contacts)
        self.assertEqual(len(contacts.gapfunc), contacts.default_max_world_contacts)
        self.assertEqual(len(contacts.material), contacts.default_max_world_contacts)

    def test_multiple_default_allocations(self):
        num_worlds = 10
        capacities = [0] * num_worlds
        contacts = Contacts(capacity=capacities, device=self.default_device)

        model_max_contacts = contacts.model_max_contacts.numpy()
        model_num_contacts = contacts.model_num_contacts.numpy()
        self.assertEqual(len(contacts.model_max_contacts), 1)
        self.assertEqual(len(contacts.model_num_contacts), 1)
        self.assertEqual(model_max_contacts[0], num_worlds * contacts.default_max_world_contacts)
        self.assertEqual(model_num_contacts[0], 0)

        world_max_contacts = contacts.world_max_contacts.numpy()
        world_num_contacts = contacts.world_num_contacts.numpy()
        self.assertEqual(len(contacts.world_max_contacts), num_worlds)
        self.assertEqual(len(contacts.world_num_contacts), num_worlds)
        for i in range(num_worlds):
            self.assertEqual(world_max_contacts[i], contacts.default_max_world_contacts)
            self.assertEqual(world_num_contacts[i], 0)

        self.assertEqual(len(contacts.wid), num_worlds * contacts.default_max_world_contacts)
        self.assertEqual(len(contacts.cid), num_worlds * contacts.default_max_world_contacts)
        self.assertEqual(len(contacts.body_A), num_worlds * contacts.default_max_world_contacts)
        self.assertEqual(len(contacts.body_B), num_worlds * contacts.default_max_world_contacts)
        self.assertEqual(len(contacts.gapfunc), num_worlds * contacts.default_max_world_contacts)
        self.assertEqual(len(contacts.material), num_worlds * contacts.default_max_world_contacts)

    def test_multiple_custom_allocations(self):
        capacities = [10, 20, 30, 40, 50, 60]
        contacts = Contacts(capacity=capacities, device=self.default_device)

        num_worlds = len(capacities)
        model_max_contacts = contacts.model_max_contacts.numpy()
        model_num_contacts = contacts.model_num_contacts.numpy()
        self.assertEqual(len(contacts.model_max_contacts), 1)
        self.assertEqual(len(contacts.model_num_contacts), 1)
        self.assertEqual(model_max_contacts[0], sum(capacities))
        self.assertEqual(model_num_contacts[0], 0)

        world_max_contacts = contacts.world_max_contacts.numpy()
        world_num_contacts = contacts.world_num_contacts.numpy()
        self.assertEqual(len(contacts.world_max_contacts), num_worlds)
        self.assertEqual(len(contacts.world_num_contacts), num_worlds)
        for i in range(num_worlds):
            self.assertEqual(world_max_contacts[i], capacities[i])
            self.assertEqual(world_num_contacts[i], 0)

        maxnc = sum(capacities)
        self.assertEqual(len(contacts.wid), maxnc)
        self.assertEqual(len(contacts.cid), maxnc)
        self.assertEqual(len(contacts.body_A), maxnc)
        self.assertEqual(len(contacts.body_B), maxnc)
        self.assertEqual(len(contacts.gapfunc), maxnc)
        self.assertEqual(len(contacts.material), maxnc)


###
# Test execution
###

if __name__ == "__main__":
    # Global numpy configurations
    np.set_printoptions(linewidth=200, precision=3, suppress=True)  # Suppress scientific notation

    # Global warp configurations
    wp.config.verbose = True
    wp.clear_kernel_cache()
    wp.clear_lto_cache()

    # Run all tests
    unittest.main(verbosity=2)
