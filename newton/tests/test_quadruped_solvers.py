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

import unittest
from pathlib import Path

import numpy as np
import warp as wp

import newton.examples.example_quadruped as quad
from newton.tests.unittest_utils import add_function_test, get_test_devices


def _run_and_compare(test, device, solver_name: str, num_frames: int = 10, num_envs: int = 8):
    """Run the quadruped simulation with *solver_name* and compare against golden outputs.

    If the golden snapshot is missing the test is skipped so CI remains green until
    new golden data is committed.
    """

    # Run the simulation head-less and pull final state as NumPy arrays.
    state = quad.run_quadruped(
        solver_name=solver_name,
        num_frames=num_frames,
        num_envs=num_envs,
        device=device,
        render=False,
        enable_timers=False,
    )

    # Path to golden snapshot (generated separately and checked-in via LFS or git).
    golden_file = Path(__file__).parent / "assets" / "golden_states" / f"quadruped_{solver_name}_{num_envs}envs_{num_frames}steps.npz"
    if not golden_file.exists():
        test.skipTest(f"Golden file {golden_file} not found – regenerate with the original solver and commit.")

    golden = np.load(golden_file)

    np.testing.assert_allclose(state["joint_q"], golden["joint_q"], rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(state["joint_qd"], golden["joint_qd"], rtol=1e-4, atol=1e-5)


# -----------------------------------------------------------------------------
# Additional helpers for MuJoCo vs. MuJoCo-native regression
# -----------------------------------------------------------------------------


def _run_and_compare_mujoco(
    test,
    device_mujoco,
    device_native,
    num_frames: int = 10,
    num_envs: int = 2,
):
    """Run the quadruped simulation with the Warp MuJoCo solver and compare against MuJoCo-native.

    Parameters
    ----------
    test : unittest.TestCase
        The active unittest instance – used for skipping if necessary.
    device_mujoco : wp.Device | str | None
        Device to run the Warp MuJoCo solver on.
    device_native : wp.Device | str | None
        Device to run MuJoCo-native on (must be CPU).
    num_frames : int, optional
        Number of simulation frames.
    num_envs : int, optional
        Number of parallel environments (MuJoCo-native supports max. 2).
    """

    # Run MuJoCo (Warp)
    state_mujoco = quad.run_quadruped(
        solver_name="mujoco",
        num_frames=num_frames,
        num_envs=num_envs,
        device=device_mujoco,
        render=False,
        enable_timers=False,
    )

    # Run MuJoCo-native reference (always on CPU)
    state_native = quad.run_quadruped(
        solver_name="mujoco-native",
        num_frames=num_frames,
        num_envs=num_envs,
        device=device_native,
        render=False,
        enable_timers=False,
    )

    # Compare final states – use tolerances similar to golden reference checks.
    np.testing.assert_allclose(state_mujoco["joint_q"], state_native["joint_q"], rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(state_mujoco["joint_qd"], state_native["joint_qd"], rtol=1e-4, atol=1e-5)

devices = get_test_devices()


class TestQuadrupedSolvers(unittest.TestCase):
    """Integration regression tests for the quadruped example using multiple solvers."""

    pass  # Tests are added dynamically below.

# -----------------------------------------------------------------------------
# Compare XPBD and Featherstone to golden reference states
# -----------------------------------------------------------------------------


for _solver in ("xpbd", "featherstone"):

    def _make_test(solver_name):
        return lambda test, device, _solver_name=solver_name: _run_and_compare(test, device, _solver_name, num_envs=8, num_frames=10)

    add_function_test(
        TestQuadrupedSolvers,
        f"test_quadruped_{_solver}",
        _make_test(_solver),
        devices=devices,
    )


# -----------------------------------------------------------------------------
# Compare mujoco to mujoco-native
# -----------------------------------------------------------------------------
cpu_device = wp.get_device("cpu")
gpu_devices = [d for d in devices if d.is_cuda]

def _make_mujoco_test():
    return lambda test, device, _native_device=cpu_device: _run_and_compare_mujoco(
        test, device_mujoco=device, device_native=_native_device, num_envs=2, num_frames=10
    )

add_function_test(
    TestQuadrupedSolvers,
    "test_quadruped_mujoco_native_vs_mujoco",
    _make_mujoco_test(),
    devices=gpu_devices, # warp mujoco is not usable on CPU
)

if __name__ == "__main__":
    wp.clear_kernel_cache()
    unittest.main(verbosity=2, failfast=True) 