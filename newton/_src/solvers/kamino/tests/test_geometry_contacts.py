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
Unit tests for `geometry/contacts.py`.

Tests all components of the Contacts data types and operations.
"""

import unittest

import numpy as np
import warp as wp

from newton._src.solvers.kamino.core.types import int32, mat33f, vec3f
from newton._src.solvers.kamino.geometry.contacts import (
    ContactMode,
    Contacts,
    make_contact_frame_xnorm,
    make_contact_frame_znorm,
)
from newton._src.solvers.kamino.utils import logger as msg

###
# Kernels
###


@wp.kernel
def _compute_contact_frame_znorm(
    # Inputs:
    normal: wp.array(dtype=vec3f),
    # Outputs:
    frame: wp.array(dtype=mat33f),
):
    tid = wp.tid()
    frame[tid] = make_contact_frame_znorm(normal[tid])


@wp.kernel
def _compute_contact_frame_xnorm(
    # Inputs:
    normal: wp.array(dtype=vec3f),
    # Outputs:
    frame: wp.array(dtype=mat33f),
):
    tid = wp.tid()
    frame[tid] = make_contact_frame_xnorm(normal[tid])


@wp.kernel
def _compute_contact_mode(
    # Inputs:
    velocity: wp.array(dtype=vec3f),
    # Outputs:
    mode: wp.array(dtype=int32),
):
    tid = wp.tid()
    mode[tid] = wp.static(ContactMode.make_compute_mode_func())(velocity[tid])


###
# Launchers
###


def compute_contact_frame_znorm(normal: wp.array, frame: wp.array, num_threads: int = 1):
    wp.launch(
        _compute_contact_frame_znorm,
        dim=num_threads,
        inputs=[normal],
        outputs=[frame],
    )


def compute_contact_frame_xnorm(normal: wp.array, frame: wp.array, num_threads: int = 1):
    wp.launch(
        _compute_contact_frame_xnorm,
        dim=num_threads,
        inputs=[normal],
        outputs=[frame],
    )


def compute_contact_mode(velocity: wp.array, mode: wp.array, num_threads: int = 1):
    wp.launch(
        _compute_contact_mode,
        dim=num_threads,
        inputs=[velocity],
        outputs=[mode],
    )


###
# Tests
###


class TestGeometryContactFrames(unittest.TestCase):
    def setUp(self):
        self.default_device = wp.get_device()
        self.verbose = False  # Set to True for detailed output

        # Set debug-level logging to print verbose test output to console
        if self.verbose:
            msg.info("\n")  # Add newline before test output for better readability
            msg.set_log_level(msg.LogLevel.DEBUG)
        else:
            msg.reset_log_level()

    def tearDown(self):
        self.default_device = None
        if self.verbose:
            msg.reset_log_level()

    def test_01_make_contact_frame_znorm(self):
        # Create a normal vectors
        test_normals: list[vec3f] = []

        # Add normals for which to test contact frame creation
        test_normals.append(vec3f(1.0, 0.0, 0.0))
        test_normals.append(vec3f(0.0, 1.0, 0.0))
        test_normals.append(vec3f(0.0, 0.0, 1.0))
        test_normals.append(vec3f(-1.0, 0.0, 0.0))
        test_normals.append(vec3f(0.0, -1.0, 0.0))
        test_normals.append(vec3f(0.0, 0.0, -1.0))

        # Create the input output arrays
        normals = wp.array(test_normals, dtype=vec3f)
        frames = wp.zeros(shape=(len(test_normals),), dtype=mat33f)

        # Compute the contact frames
        compute_contact_frame_znorm(normal=normals, frame=frames, num_threads=len(test_normals))
        if self.verbose:
            print(f"normals:\n{normals}\n")
            print(f"frames:\n{frames}\n")

        # Extract numpy arrays for comparison
        frames_np = frames.numpy()

        # Check determinants of each frame
        for i in range(len(test_normals)):
            det = np.linalg.det(frames_np[i])
            self.assertTrue(np.isclose(det, 1.0, atol=1e-6))

        # Check each primitive frame
        self.assertTrue(
            np.allclose(frames_np[0], np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]), atol=1e-6)
        )
        self.assertTrue(
            np.allclose(frames_np[1], np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]]), atol=1e-6)
        )
        self.assertTrue(
            np.allclose(frames_np[2], np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]), atol=1e-6)
        )
        self.assertTrue(
            np.allclose(frames_np[3], np.array([[0.0, 0.0, -1.0], [1.0, 0.0, 0.0], [0.0, -1.0, 0.0]]), atol=1e-6)
        )
        self.assertTrue(
            np.allclose(frames_np[4], np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]]), atol=1e-6)
        )
        self.assertTrue(
            np.allclose(frames_np[5], np.array([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]]), atol=1e-6)
        )

    def test_02_make_contact_frame_xnorm(self):
        # Create a normal vectors
        test_normals: list[vec3f] = []

        # Add normals for which to test contact frame creation
        test_normals.append(vec3f(1.0, 0.0, 0.0))
        test_normals.append(vec3f(0.0, 1.0, 0.0))
        test_normals.append(vec3f(0.0, 0.0, 1.0))
        test_normals.append(vec3f(-1.0, 0.0, 0.0))
        test_normals.append(vec3f(0.0, -1.0, 0.0))
        test_normals.append(vec3f(0.0, 0.0, -1.0))

        # Create the input output arrays
        normals = wp.array(test_normals, dtype=vec3f)
        frames = wp.zeros(shape=(len(test_normals),), dtype=mat33f)

        # Compute the contact frames
        compute_contact_frame_xnorm(normal=normals, frame=frames, num_threads=len(test_normals))
        if self.verbose:
            print(f"normals:\n{normals}\n")
            print(f"frames:\n{frames}\n")

        # Extract numpy arrays for comparison
        frames_np = frames.numpy()

        # Check determinants of each frame
        for i in range(len(test_normals)):
            det = np.linalg.det(frames_np[i])
            self.assertTrue(np.isclose(det, 1.0, atol=1e-6))


class TestGeometryContactMode(unittest.TestCase):
    def setUp(self):
        self.default_device = wp.get_device()
        self.verbose = False  # Set to True for detailed output

        # Set debug-level logging to print verbose test output to console
        if self.verbose:
            msg.info("\n")  # Add newline before test output for better readability
            msg.set_log_level(msg.LogLevel.DEBUG)
        else:
            msg.reset_log_level()

    def tearDown(self):
        self.default_device = None
        if self.verbose:
            msg.reset_log_level()

    def test_01_contact_mode_opening(self):
        v_input = wp.array([vec3f(0.0, 0.0, 0.01)], dtype=vec3f)
        mode_output = wp.zeros(shape=(1,), dtype=int32)
        compute_contact_mode(velocity=v_input, mode=mode_output, num_threads=1)
        mode_int32 = mode_output.numpy()[0]
        mode = ContactMode(int(mode_int32))
        msg.info(f"mode: {mode} (int: {int(mode_int32)})")
        self.assertEqual(mode, ContactMode.OPENING)

    def test_02_contact_mode_sticking(self):
        v_input = wp.array([vec3f(0.0, 0.0, 1e-7)], dtype=vec3f)
        mode_output = wp.zeros(shape=(1,), dtype=int32)
        compute_contact_mode(velocity=v_input, mode=mode_output, num_threads=1)
        mode_int32 = mode_output.numpy()[0]
        mode = ContactMode(int(mode_int32))
        msg.info(f"mode: {mode} (int: {int(mode_int32)})")
        self.assertEqual(mode, ContactMode.STICKING)

    def test_03_contact_mode_slipping(self):
        v_input = wp.array([vec3f(0.1, 0.0, 0.0)], dtype=vec3f)
        mode_output = wp.zeros(shape=(1,), dtype=int32)
        compute_contact_mode(velocity=v_input, mode=mode_output, num_threads=1)
        mode_int32 = mode_output.numpy()[0]
        mode = ContactMode(int(mode_int32))
        msg.info(f"mode: {mode} (int: {int(mode_int32)})")
        self.assertEqual(mode, ContactMode.SLIDING)


class TestGeometryContacts(unittest.TestCase):
    def setUp(self):
        self.default_device = wp.get_device()
        self.verbose = False  # Set to True for detailed output

        # Set debug-level logging to print verbose test output to console
        if self.verbose:
            msg.info("\n")  # Add newline before test output for better readability
            msg.set_log_level(msg.LogLevel.DEBUG)
        else:
            msg.reset_log_level()

    def tearDown(self):
        self.default_device = None
        if self.verbose:
            msg.reset_log_level()

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
        self.assertEqual(len(contacts.gid_AB), contacts.default_max_world_contacts)
        self.assertEqual(len(contacts.bid_AB), contacts.default_max_world_contacts)
        self.assertEqual(len(contacts.position_A), contacts.default_max_world_contacts)
        self.assertEqual(len(contacts.position_B), contacts.default_max_world_contacts)
        self.assertEqual(len(contacts.gapfunc), contacts.default_max_world_contacts)
        self.assertEqual(len(contacts.frame), contacts.default_max_world_contacts)
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
        self.assertEqual(len(contacts.gid_AB), num_worlds * contacts.default_max_world_contacts)
        self.assertEqual(len(contacts.bid_AB), num_worlds * contacts.default_max_world_contacts)
        self.assertEqual(len(contacts.position_A), num_worlds * contacts.default_max_world_contacts)
        self.assertEqual(len(contacts.position_B), num_worlds * contacts.default_max_world_contacts)
        self.assertEqual(len(contacts.gapfunc), num_worlds * contacts.default_max_world_contacts)
        self.assertEqual(len(contacts.frame), num_worlds * contacts.default_max_world_contacts)
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
        self.assertEqual(len(contacts.gid_AB), maxnc)
        self.assertEqual(len(contacts.bid_AB), maxnc)
        self.assertEqual(len(contacts.position_A), maxnc)
        self.assertEqual(len(contacts.position_B), maxnc)
        self.assertEqual(len(contacts.gapfunc), maxnc)
        self.assertEqual(len(contacts.frame), maxnc)
        self.assertEqual(len(contacts.material), maxnc)


###
# Test execution
###

if __name__ == "__main__":
    # Global numpy configurations
    np.set_printoptions(linewidth=10000, threshold=10000, precision=10, suppress=True)

    # Global warp configurations
    wp.config.verbose = False
    wp.clear_kernel_cache()
    wp.clear_lto_cache()

    # Run all tests
    unittest.main(verbosity=2)
