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

import argparse
import functools
import unittest

import warp as wp

from newton.examples.suctioncup.example_surface_gripper_kinematic_drive import (
    DEFAULT_SIM_DT,
    FPS,
    Example,
    VelocityProfile,
    make_profile,
)
from newton.tests.unittest_utils import add_function_test, get_selected_cuda_test_devices
from newton.viewer import ViewerNull

# box B must ride with box A: their z gap may drift at most this much from its engagement value [m]
_MAX_Z_GAP_DRIFT = 0.01


def _seal_engaged_at(seal_times, seal_values, t):
    """Step lookup of the seal on/off series at time ``t`` (matches eval_seal_at_current_time)."""
    engaged = seal_values[0]
    for keyframe_time, value in zip(seal_times, seal_values, strict=True):
        if keyframe_time <= t:
            engaged = value
    return engaged


def _run_seal_z_separation(test, device, profile):
    """Box B stays sealed to box A along z while the seal is engaged, for a given profile.

    Builds the ``surface_gripper_kinematic_drive`` scene with ``profile``, steps the entire
    profile in process, and asserts -- while the seal is engaged -- that ``A_z - B_z`` stays
    within tolerance of the value at engagement (the seal neither separates nor lets B slip).
    Once the seal releases (if the profile drops B), the check is skipped.
    """
    with wp.ScopedDevice(device):
        vel_times, _, seal_times, seal_values = make_profile(profile)
        num_frames = round(vel_times[-1] * FPS)
        args = argparse.Namespace(profile=profile, sim_dt=DEFAULT_SIM_DT)
        example = Example(ViewerNull(num_frames=num_frames), args)

        q = example.state_0.body_q.numpy()
        initial_gap = float(q[example.box_a][2] - q[example.box_b][2])

        for frame in range(num_frames):
            example.step()
            if not _seal_engaged_at(seal_times, seal_values, example.sim_time):
                continue  # seal released; box B is meant to drop, so stop checking the gap
            q = example.state_0.body_q.numpy()
            gap = float(q[example.box_a][2] - q[example.box_b][2])
            test.assertLess(
                abs(gap - initial_gap),
                _MAX_Z_GAP_DRIFT,
                msg=(
                    f"[{profile.name}] box B separated from box A along z at frame {frame}: "
                    f"gap={gap:.4f} m, engagement gap={initial_gap:.4f} m"
                ),
            )


# the example uses SolverMuJoCo, so restrict to CUDA devices
devices = get_selected_cuda_test_devices()


class TestSurfaceGripper(unittest.TestCase):
    pass


# one test per velocity profile (x device); the seal must hold box B throughout each motion
for _profile in VelocityProfile:
    add_function_test(
        TestSurfaceGripper,
        f"test_seal_z_separation_{_profile.name.lower()}",
        functools.partial(_run_seal_z_separation, profile=_profile),
        devices=devices,
    )


if __name__ == "__main__":
    unittest.main(verbosity=2)
