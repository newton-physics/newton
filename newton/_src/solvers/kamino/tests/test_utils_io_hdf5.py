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
KAMINO: UNIT TESTS: Input/Output: HDF5
"""

import os
import unittest

import h5py
import numpy as np
import warp as wp

# Module to be tested
import newton._src.solvers.kamino.utils.io.hdf5 as hdf5
from newton._src.solvers.kamino.models.builders import (
    build_boxes_nunchaku,
)
from newton._src.solvers.kamino.models.utils import (
    make_single_builder,
)
from newton._src.solvers.kamino.simulation.simulator import Simulator

###
# Helper functions
###


def test_output_path():
    return os.path.dirname(os.path.realpath(__file__)) + "/output"


###
# Constants
###

DEFAULT_INPUT_DATASET_PATH = "../../data/hdf5/dual_problems_all.hdf5"
DEFAULT_OUTPUT_DATASET_PATH = test_output_path() + "/output.hdf5"


###
# Tests
###


class TestHDF5(unittest.TestCase):
    def setUp(self):
        self.dtype = np.float32
        self.default_device = wp.get_device()
        self.infilename = DEFAULT_INPUT_DATASET_PATH
        self.found_dataset = os.path.exists(self.infilename)
        self.datapath = "Primitive/boxes_nunchaku/DenseConstraints/0"
        # TODO: Add a helper function to create the dataset if it doesn't exist

    def tearDown(self):
        self.default_device = None

    def test_load_rigid_body(self):
        # Skip test if dataset doesn't exist on the system
        if not self.found_dataset:
            return

        # Load the data from the HDF5 file if it exists
        print(f"Loading HDF5 data from {self.infilename}")
        indata = h5py.File(self.infilename, "r")
        inframe = indata[self.datapath]
        print("indata: ", indata)
        print("inframe: ", inframe)

        # Load data for a single RigidBody
        datapath_rb0 = "RigidBodySystem/bodies/0"
        rbd0 = hdf5.RigidBodyData(inframe[datapath_rb0], self.dtype)
        print("rbd0: ", rbd0)

        # TODO: What to test here?

    def test_load_read_write_read_rigid_body(self):
        # Skip test if dataset doesn't exist on the system
        if not self.found_dataset:
            return

        # Load the data from the HDF5 file if it exists
        print(f"Loading HDF5 data from {self.infilename}")
        indata = h5py.File(self.infilename, "r")
        inframe = indata[self.datapath]
        print("indata: ", indata)
        print("inframe: ", inframe)

        # Load data for a single RigidBody
        datapath_rb0 = "RigidBodySystem/bodies/0"
        rbd0 = hdf5.RigidBodyData(inframe[datapath_rb0], self.dtype)
        print("rbd0: ", rbd0)

        # Test saving the data
        outdata = h5py.File(DEFAULT_OUTPUT_DATASET_PATH, "w")
        rbd0.store(dataset=outdata, namespace="RigidBodySystem/bodies/0")

        # Test loading the data again
        outframe = outdata["RigidBodySystem/bodies/0"]
        rbd0_loaded = hdf5.RigidBodyData(outframe, self.dtype)
        print("rbd0_loaded: ", rbd0_loaded)

        # Test if the loaded data matches the original data
        self.assertTrue(rbd0.name == rbd0_loaded.name)
        self.assertTrue(rbd0.uid == rbd0_loaded.uid)
        self.assertTrue(rbd0.m_i == rbd0_loaded.m_i)
        self.assertTrue(np.array_equal(rbd0.i_I_i, rbd0_loaded.i_I_i))
        self.assertTrue(np.array_equal(rbd0.s_i_0, rbd0_loaded.s_i_0))
        self.assertTrue(np.array_equal(rbd0.s_i, rbd0_loaded.s_i))
        self.assertTrue(np.array_equal(rbd0.w_i, rbd0_loaded.w_i))
        self.assertTrue(np.array_equal(rbd0.w_a_i, rbd0_loaded.w_a_i))
        self.assertTrue(np.array_equal(rbd0.w_j_i, rbd0_loaded.w_j_i))
        self.assertTrue(np.array_equal(rbd0.w_l_i, rbd0_loaded.w_l_i))
        self.assertTrue(np.array_equal(rbd0.w_c_i, rbd0_loaded.w_c_i))
        self.assertTrue(np.array_equal(rbd0.w_e_i, rbd0_loaded.w_e_i))

    def test_load_read_write_read_joint(self):
        # Skip test if dataset doesn't exist on the system
        if not self.found_dataset:
            return

        # Load the data from the HDF5 file if it exists
        print(f"Loading HDF5 data from {self.infilename}")
        indata = h5py.File(self.infilename, "r")
        inframe = indata[self.datapath]
        print("indata: ", indata)
        print("inframe: ", inframe)

        # Load data for a single Joint
        datapath_j0 = "RigidBodySystem/joints/0"
        j0 = hdf5.JointData(inframe[datapath_j0], self.dtype)
        print("j0: ", j0)

        # Test saving the data
        outdata = h5py.File(DEFAULT_OUTPUT_DATASET_PATH, "w")
        j0.store(dataset=outdata, namespace="RigidBodySystem/joints/0")

        # Test loading the data again
        outframe = outdata["RigidBodySystem/joints/0"]
        j0_loaded = hdf5.JointData(outframe, self.dtype)
        print("j0_loaded: ", j0_loaded)

        # Test if the loaded data matches the original data
        self.assertTrue(j0.name == j0_loaded.name)
        self.assertTrue(j0.uid == j0_loaded.uid)
        self.assertTrue(j0.dofs == j0_loaded.dofs)
        self.assertTrue(j0.type == j0_loaded.type)
        self.assertTrue(j0.base_id == j0_loaded.base_id)
        self.assertTrue(j0.follower_id == j0_loaded.follower_id)
        self.assertTrue(np.array_equal(j0.frame, j0_loaded.frame))

    def test_load_read_write_read_contact(self):
        # Skip test if dataset doesn't exist on the system
        if not self.found_dataset:
            return

        # Load the data from the HDF5 file if it exists
        print(f"Loading HDF5 data from {self.infilename}")
        indata = h5py.File(self.infilename, "r")
        inframe = indata[self.datapath]
        print("indata: ", indata)
        print("inframe: ", inframe)

        # Load data for a single Contact
        datapath_c0 = "Contacts/contacts/0"
        c0 = hdf5.ContactData(inframe[datapath_c0], self.dtype)
        print("c0: ", c0)

        # Test saving the data
        outdata = h5py.File(DEFAULT_OUTPUT_DATASET_PATH, "w")
        c0.store(dataset=outdata, namespace="Contacts/contacts/0")

        # Test loading the data again
        outframe = outdata["Contacts/contacts/0"]
        c0_loaded = hdf5.ContactData(outframe, self.dtype)
        print("c0_loaded: ", c0_loaded)

        # Test if the loaded data matches the original data
        self.assertTrue(c0.gid_A == c0_loaded.gid_A)
        self.assertTrue(c0.gid_B == c0_loaded.gid_B)
        self.assertTrue(c0.bid_A == c0_loaded.bid_A)
        self.assertTrue(np.array_equal(c0.position_A, c0_loaded.position_A))
        self.assertTrue(np.array_equal(c0.position_B, c0_loaded.position_B))
        self.assertTrue(np.array_equal(c0.frame, c0_loaded.frame))
        self.assertTrue(np.array_equal(c0.position, c0_loaded.position))
        self.assertTrue(np.array_equal(c0.normal, c0_loaded.normal))
        self.assertTrue(c0.penetration == c0_loaded.penetration)
        self.assertTrue(c0.friction == c0_loaded.friction)
        self.assertTrue(c0.restitutuon == c0_loaded.restitutuon)

    def test_update_from_simulator(self):
        # Skip test if dataset doesn't exist on the system
        if not self.found_dataset:
            return

        # Create a multi-instanced system
        builder, _, _ = make_single_builder(build_func=build_boxes_nunchaku)
        # sim = Simulator(builder=builder, device=self.default_device, shadow=True)  # TODO: Use shadow=True to test shadow data
        sim = Simulator(builder=builder, device=self.default_device)

        # Construct and configure the system data container
        sdata = hdf5.RigidBodySystemData()
        sdata.configure(simulator=sim)

        # Set data from the system
        sdata.update_from(simulator=sim)
        print("sdata.bodies: ", sdata.bodies)
        print("sdata.joints: ", sdata.joints)

        # Construct and configure the contact data container
        cdata = hdf5.ContactsData()
        cdata.update_from(simulator=sim)
        print("cdata.ncontacts: ", cdata.ncontacts)
        print("cdata.contacts: ", cdata.contacts)


###
# Test execution
###

if __name__ == "__main__":
    np.set_printoptions(linewidth=200, precision=3, suppress=True)
    wp.clear_kernel_cache()
    unittest.main(verbosity=2)
