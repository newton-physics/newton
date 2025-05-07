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
from typing import Any, Callable, Dict, Mapping
from unittest.mock import patch

import numpy as np
import warp as wp

import newton.examples.example_cartpole as example_cartpole
import newton.examples.example_cloth_self_contact as example_cloth_self_contact
import newton.examples.example_quadruped as example_quadruped
from newton.tests.unittest_utils import add_function_test, get_test_devices

# -----------------------------------------------------------------------------
# Example runner map
# -----------------------------------------------------------------------------

RUN_EXAMPLE: Dict[str, Callable[..., dict]] = {
    "quadruped": example_quadruped.run_quadruped,
    "cartpole": example_cartpole.run_cartpole,
    "cloth_self_contact": example_cloth_self_contact.run_cloth_self_contact,
}

# -----------------------------------------------------------------------------
# Default tolerances for regression testing
# -----------------------------------------------------------------------------

DEFAULT_TOLERANCES: dict[str, dict[str, float]] = {
    "q": {"rtol": 1e-5, "atol": 1e-6},
    "qd": {"rtol": 1e-4, "atol": 1e-5},
}

# -----------------------------------------------------------------------------
# Test cases
# -----------------------------------------------------------------------------

CASES: list[dict] = [
    # ────────── Cart-pole ──────────
    {
        "example": "cartpole",
        "solver": "euler",
        "policy": "sin",
        "num_frames": 10,
        "num_envs": 8,
    },
    {
        "example": "cartpole",
        "solver": "featherstone",
        "policy": "sin",
        "num_frames": 10,
        "num_envs": 8,
    },
    {
        "example": "cartpole",
        "solver": "xpbd",
        "policy": "sin",
        "num_frames": 10,
        "num_envs": 8,
    },
    # Warp-MuJoCo must run on GPU; we'll also compare against MuJoCo-native (CPU)
    {
        "example": "cartpole",
        "solver": "mujoco",
        "policy": "sin",
        "num_frames": 10,
        "num_envs": 2,                     # smaller batch for MuJoCo
        "device_filter": lambda d: d.is_cuda,
    },

    # ────────── Quadruped ──────────
    {
        "example": "quadruped",
        "solver": "featherstone",
        "policy": "sin",
        "num_frames": 10,
        "num_envs": 8,
    },
    {
        "example": "quadruped",
        "solver": "xpbd",
        "policy": "sin",
        "num_frames": 10,
        "num_envs": 8,
        # GPU shows higher numerical noise for XPBD → looser tolerances
        "tolerances": {
            "gpu": {
                "body_q":  {"rtol": 1e-4, "atol": 1e-4},
                "body_qd": {"rtol": 1e-4, "atol": 2e-3},
            }
        },
    },
    {
        "example": "quadruped",
        "solver": "mujoco",
        "policy": "sin",
        "num_frames": 10,
        "num_envs": 2,
        "device_filter": lambda d: d.is_cuda,
    },

    # ────────── Cloth self-contact ──────────
    {
        "example": "cloth_self_contact",
        "solver": "vbd",
        "policy": "none",
        "num_frames": 10,
        "num_envs": 1,
        "state_arrays": ("particle_q", "particle_qd"),
    },
]

# -----------------------------------------------------------------------------
# Helper functions
# -----------------------------------------------------------------------------

def golden_path(
    example_name: str,
    solver_name: str,
    policy: str,
    num_envs: int,
    num_frames: int,
    device_str: str,
) -> Path:
    """Return the path to the golden reference file for the given parameters."""
    return (
        Path(__file__).parent
        / "assets"
        / "golden_states"
        / f"{example_name}_{solver_name}_{policy}policy_{num_envs}envs_{num_frames}steps_{device_str}.npz"
    )


def resolve_tolerance(
    array_name: str,
    *,
    device: wp.context.Device,
    tolerances: dict[str, Any] | None = None,
) -> dict[str, float]:
    """Return tolerance dict for *array_name* considering overrides/device/defaults."""
    # Device-specific overrides take top priority.
    if tolerances is not None:
        dev_key = "gpu" if device.is_cuda else "cpu"
        if array_name in tolerances.get(dev_key, {}):
            return tolerances[dev_key][array_name]

        # Fallback to generic key ("q" or "qd") inside device-specific overrides.
        generic_key = "qd" if array_name.endswith("qd") else "q"
        if generic_key in tolerances.get(dev_key, {}):
            return tolerances[dev_key][generic_key]

    # Absolute fallback to global defaults.
    generic_key = "qd" if array_name.endswith("qd") else "q"
    return DEFAULT_TOLERANCES[generic_key]


def run_example_and_compare(
    test,
    device: wp.context.Device,
    *,
    case: dict,
) -> None:
    """Run example with given parameters and compare against golden reference or MuJoCo-native.
    
    Parameters
    ----------
    test : unittest.TestCase
        The test case instance.
    device : wp.context.Device
        Device to run the simulation on.
    case : dict
        Dictionary containing test parameters:
        - example: Name of the example (must exist in ``RUN_EXAMPLE``).
        - solver: Name of the solver to use.
        - policy: Control policy to apply ("none" or "sin").
        - num_frames: Number of simulation frames to advance.
        - num_envs: Number of parallel environments.
        - solver_kwargs: Additional keyword arguments to pass to the solver.
        - tolerances: Tolerance overrides for specific arrays or device types.
        - state_arrays: Which state arrays to compare against the golden reference.
        - device_filter: Optional filter function that returns True if the device is supported.
    """
    # Extract parameters from case
    example = case["example"]
    solver = case["solver"]
    policy = case.get("policy", "none")
    num_frames = case.get("num_frames", 100)
    num_envs = case.get("num_envs", 1)
    solver_kwargs = case.get("solver_kwargs")
    tolerances = case.get("tolerances")
    state_arrays = case.get("state_arrays", ("body_q", "body_qd"))
    device_filter = case.get("device_filter")

    # Skip if device filter is provided and returns False
    if device_filter is not None and not device_filter(device):
        test.skipTest(f"Solver {solver} not supported on device {device}")

    # Special case: MuJoCo vs MuJoCo-native comparison
    if solver == "mujoco":
        _run_and_compare_mujoco(
            test=test,
            example_name=example,
            device_mujoco=device,
            device_native=wp.get_device("cpu"),
            policy=policy,
            num_frames=num_frames,
            num_envs=num_envs,
            solver_kwargs=solver_kwargs,
            state_arrays=state_arrays,
        )
        return

    # Regular case: Compare against golden reference
    # Suppress warnings from cloth self contact example, which autofail test
    with patch("warp.utils.warn") as mock_warn:
        mock_warn.side_effect = lambda *args, **kwargs: None

        run_example = RUN_EXAMPLE[example]

        # Execute simulation
        state = run_example(
            solver_name=solver,
            policy=policy,
            num_frames=num_frames,
            num_envs=num_envs,
            solver_kwargs=solver_kwargs,
            device=device,
            stage_path=None,
            enable_timers=False,
        )

        device_str = "cpu" if device.is_cpu else "gpu"

        golden_file = golden_path(
            example_name=example,
            solver_name=solver,
            policy=policy,
            num_envs=num_envs,
            num_frames=num_frames,
            device_str=device_str,
        )
        golden = np.load(golden_file)

        # Compare all requested state arrays
        for arr in state_arrays:
            tol = resolve_tolerance(arr, device=device, tolerances=tolerances)
            np.testing.assert_allclose(state[arr], golden[arr], rtol=tol["rtol"], atol=tol["atol"])


def _run_and_compare_mujoco(
    test,
    example_name: str,
    device_mujoco: wp.context.Device,
    device_native: wp.context.Device,
    policy: str,
    num_frames: int,
    num_envs: int,
    solver_kwargs: dict[str, Any] | None = None,
    state_arrays: tuple[str, ...] = ("body_q", "body_qd"),
) -> None:
    """Run example with MuJoCo and compare against MuJoCo-native reference."""
    run_example = RUN_EXAMPLE[example_name]

    # Execute MuJoCo (Warp)
    state_mujoco = run_example(
        solver_name="mujoco",
        policy=policy,
        num_frames=num_frames,
        num_envs=num_envs,
        solver_kwargs=solver_kwargs,
        device=device_mujoco,
        stage_path=None,
        enable_timers=False,
    )

    # Execute MuJoCo-native reference (always CPU)
    state_native = run_example(
        solver_name="mujoco-native",
        policy=policy,
        num_frames=num_frames,
        num_envs=num_envs,
        solver_kwargs=solver_kwargs,
        device=device_native,
        stage_path=None,
        enable_timers=False,
    )

    # Compare the selected state arrays using a fixed tolerance suitable for
    # MuJoCo vs MuJoCo-native regression.
    for arr in state_arrays:
        np.testing.assert_allclose(state_mujoco[arr], state_native[arr], rtol=1e-5, atol=1e-5)


# -----------------------------------------------------------------------------
# Test class
# -----------------------------------------------------------------------------

class TestExampleSolvers(unittest.TestCase):
    """Integration regression tests for multiple examples / solvers."""
    pass  # Tests are added dynamically below


# -----------------------------------------------------------------------------
# Register tests
# -----------------------------------------------------------------------------

devices = get_test_devices()

for case in CASES:
    example = case["example"]
    solver = case["solver"]
    device_filter = case.get("device_filter")
    
    # Create test name
    test_name = f"test_{example}_{solver}"
    
    # For each device, create a test if the device filter passes
    filtered_devices = []
    for device in devices:
        if device_filter is None or device_filter(device):
            filtered_devices.append(device)
    
    # Register the test with the filtered devices
    if filtered_devices:
        add_function_test(
            TestExampleSolvers,
            test_name,
            run_example_and_compare,
            devices=filtered_devices,
            case=case,
        )


if __name__ == "__main__":
    wp.clear_kernel_cache()
    unittest.main(verbosity=2, failfast=True) 