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

from newton._src.solvers.kamino.core.types import mat33f, vec3f

# Module to be tested
from newton._src.solvers.kamino.geometry.math import make_contact_frame_xnorm, make_contact_frame_znorm

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


###
# Launchers
###


def compute_contact_frame_znorm(
    normal: wp.array(dtype=vec3f),
    frame: wp.array(dtype=mat33f),
    num_threads: int = 1,
):
    wp.launch(
        _compute_contact_frame_znorm,
        dim=num_threads,
        inputs=[normal],
        outputs=[frame],
    )


def compute_contact_frame_xnorm(
    normal: wp.array(dtype=vec3f),
    frame: wp.array(dtype=mat33f),
    num_threads: int = 1,
):
    wp.launch(
        _compute_contact_frame_xnorm,
        dim=num_threads,
        inputs=[normal],
        outputs=[frame],
    )


###
# Tests
###


class TestGeometryMath(unittest.TestCase):
    def setUp(self):
        self.verbose = False  # Set to True for verbose output
        self.default_device = wp.get_device()

    def tearDown(self):
        self.default_device = None

    def test_make_contact_frame_znorm(self):
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

    def test_make_contact_frame_xnorm(self):
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


###
# Test execution
###

if __name__ == "__main__":
    wp.clear_kernel_cache()
    wp.clear_lto_cache()
    np.set_printoptions(linewidth=200, precision=3, suppress=True)  # Suppress scientific notation
    unittest.main(verbosity=2)
