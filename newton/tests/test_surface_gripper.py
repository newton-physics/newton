# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
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

"""Tests for the surface-gripper (suction cup) examples."""

import unittest

import warp as wp

from newton.examples.suctioncup.example_surface_gripper_kinematic_drive import Example
from newton.tests.unittest_utils import add_function_test, get_selected_cuda_test_devices
from newton.viewer import ViewerNull

# box A must ride with box B: their z gap may drift at most this much from its engagement value [m]
_MAX_Z_GAP_DRIFT = 0.01
_NUM_FRAMES = 120


def test_seal_z_separation(test, device):
    """Box A stays sealed to box B along z while box B is position-driven upward.

    Reuses the ``surface_gripper_kinematic_drive`` scene and steps it in process, asserting every frame
    that ``A_z - B_z`` stays within tolerance of the value at engagement -- the seal neither
    separates (gap grows) nor lets A sink into B (gap shrinks).
    """
    with wp.ScopedDevice(device):
        example = Example(ViewerNull(num_frames=_NUM_FRAMES), args=None)

        q = example.state_0.body_q.numpy()
        initial_gap = float(q[example.box_a][2] - q[example.box_b][2])

        for frame in range(_NUM_FRAMES):
            example.step()
            q = example.state_0.body_q.numpy()
            gap = float(q[example.box_a][2] - q[example.box_b][2])
            test.assertLess(
                abs(gap - initial_gap),
                _MAX_Z_GAP_DRIFT,
                msg=(
                    f"box A separated from box B along z at frame {frame}: "
                    f"gap={gap:.4f} m, engagement gap={initial_gap:.4f} m"
                ),
            )


# the example uses SolverMuJoCo, so restrict to CUDA devices
devices = get_selected_cuda_test_devices()


class TestSurfaceGripper(unittest.TestCase):
    pass


add_function_test(TestSurfaceGripper, "test_seal_z_separation", test_seal_z_separation, devices=devices)


if __name__ == "__main__":
    #wp.clear_kernel_cache()
    unittest.main(verbosity=2)
