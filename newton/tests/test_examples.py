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

"""Test examples in the newton.examples package.

Currently, this script mainly checks that the examples can run. There are no
correctness checks.

The test parameters are typically tuned so that each test can run in 10 seconds
or less, ignoring module compilation time. A notable exception is the robot
manipulating cloth example, which takes approximately 35 seconds to run on a
CUDA device.
"""

import os
import subprocess
import sys
import tempfile
import unittest
from typing import Any

import warp as wp

import newton.tests.unittest_utils
from newton.tests.unittest_utils import (
    USD_AVAILABLE,
    add_function_test,
    get_selected_cuda_test_devices,
    get_test_devices,
    sanitize_identifier,
)

wp.init()

supports_load_during_graph_capture = False

if wp.context.runtime.driver_version >= (12, 3):
    supports_load_during_graph_capture = True


def _build_command_line_options(test_options: dict[str, Any]) -> list:
    """Helper function to build command-line options from the test options dictionary."""
    additional_options = []

    for key, value in test_options.items():
        if isinstance(value, bool):
            # Default behavior expecting argparse.BooleanOptionalAction support
            additional_options.append(f"--{'no-' if not value else ''}{key.replace('_', '-')}")
        else:
            # Just add --key value
            additional_options.extend(["--" + key.replace("_", "-"), str(value)])

    return additional_options


def _merge_options(base_options: dict[str, Any], device_options: dict[str, Any]) -> dict[str, Any]:
    """Helper function to merge base test options with device-specific test options."""
    merged_options = base_options.copy()

    #  Update options with device-specific dictionary, overwriting existing keys with the more-specific values
    merged_options.update(device_options)
    return merged_options


def add_example_test(
    cls: type,
    name: str,
    devices: list | None = None,
    test_options: dict[str, Any] | None = None,
    test_options_cpu: dict[str, Any] | None = None,
    test_options_cuda: dict[str, Any] | None = None,
):
    """Registers a Newton example to run on ``devices`` as a TestCase."""

    if test_options is None:
        test_options = {}
    if test_options_cpu is None:
        test_options_cpu = {}
    if test_options_cuda is None:
        test_options_cuda = {}

    def run(test, device):
        if wp.get_device(device).is_cuda:
            options = _merge_options(test_options, test_options_cuda)
        else:
            options = _merge_options(test_options, test_options_cpu)

        # Mark the test as skipped if Torch is not installed but required
        torch_required = options.pop("torch_required", False)
        if torch_required:
            try:
                import torch  # noqa: PLC0415

                if wp.get_device(device).is_cuda and not torch.cuda.is_available():
                    # Ensure torch has CUDA support
                    test.skipTest("Torch not compiled with CUDA support")

            except Exception as e:
                test.skipTest(f"{e}")

        # Mark the test as skipped if USD is not installed but required
        usd_required = options.pop("usd_required", False)
        if usd_required and not USD_AVAILABLE:
            test.skipTest("Requires usd-core")

        # Find the current Warp cache
        warp_cache_path = wp.config.kernel_cache_dir

        env_vars = os.environ.copy()
        if warp_cache_path is not None:
            env_vars["WARP_CACHE_PATH"] = warp_cache_path

        if newton.tests.unittest_utils.coverage_enabled:
            # Generate a random coverage data file name - file is deleted along with containing directory
            with tempfile.NamedTemporaryFile(
                dir=newton.tests.unittest_utils.coverage_temp_dir, delete=False
            ) as coverage_file:
                pass

            command = ["coverage", "run", f"--data-file={coverage_file.name}"]

            if newton.tests.unittest_utils.coverage_branch:
                command.append("--branch")

        else:
            command = [sys.executable]

        # Append Warp commands
        command.extend(["-m", f"newton.examples.{name}", "--device", str(device)])

        stage_path = (
            options.pop(
                "stage_path",
                os.path.join(os.path.dirname(__file__), f"outputs/{name}_{sanitize_identifier(device)}.usd"),
            )
            if USD_AVAILABLE
            else "None"
        )

        if stage_path:
            command.extend(["--stage-path", stage_path])
            try:
                os.remove(stage_path)
            except OSError:
                pass

        command.extend(_build_command_line_options(options))

        # Set the test timeout in seconds
        test_timeout = options.pop("test_timeout", 600)

        # Can set active=True when tuning the test parameters
        with wp.ScopedTimer(f"{name}_{sanitize_identifier(device)}", active=False):
            # Run the script as a subprocess
            result = subprocess.run(
                command, capture_output=True, text=True, env=env_vars, timeout=test_timeout, check=False
            )

        # Check the return code (0 is standard for success)
        test.assertEqual(
            result.returncode,
            0,
            msg=f"Failed with return code {result.returncode}, command: {' '.join(command)}\n\nOutput:\n{result.stdout}\n{result.stderr}",
        )

        # If the test succeeded, try to clean up the output by default
        if stage_path and result.returncode == 0:
            try:
                os.remove(stage_path)
            except OSError:
                pass

    add_function_test(cls, f"test_{name}", run, devices=devices, check_output=False)


cuda_test_devices = get_selected_cuda_test_devices(mode="basic")  # Don't test on multiple GPUs to save time
test_devices = get_test_devices(mode="basic")


class TestClothExamples(unittest.TestCase):
    pass


add_example_test(
    TestClothExamples,
    name="example_cloth_bending",
    devices=test_devices,
    test_options={"usd_required": True, "stage_path": "None"},
    test_options_cpu={"num_frames": 100},
)
add_example_test(
    TestClothExamples,
    name="example_cloth_self_contact",
    devices=test_devices,
    test_options={"usd_required": True, "stage_path": "None"},
    test_options_cuda={"num_frames": 150},
    test_options_cpu={"num_frames": 5},
)
add_example_test(
    TestClothExamples,
    name="example_cloth_hanging",
    devices=test_devices,
    test_options={"stage_path": "None"},
    test_options_cpu={"width": 32, "height": 16, "num_frames": 10},
)
add_example_test(
    TestClothExamples,
    name="example_cloth_style3d",
    devices=test_devices,
    test_options={"usd_required": True, "stage_path": "None"},
    test_options_cuda={"num_frames": 500},
    test_options_cpu={"num_frames": 2},
)


class TestBasicRobotExamples(unittest.TestCase):
    pass


add_example_test(
    TestBasicRobotExamples,
    name="example_cartpole",
    devices=test_devices,
    test_options={
        "stage_path": "None",
        "num_frames": 100 if supports_load_during_graph_capture else 10,
        "use_cuda_graph": supports_load_during_graph_capture,
    },
    test_options_cpu={"num_frames": 10},
)
add_example_test(
    TestBasicRobotExamples,
    name="example_g1",
    devices=test_devices,
    test_options={
        "stage_path": "None",
        "num_frames": 500 if supports_load_during_graph_capture else 20,
        "use_cuda_graph": supports_load_during_graph_capture,
    },
    test_options_cpu={"num_frames": 10},
)
add_example_test(
    TestBasicRobotExamples,
    name="example_humanoid",
    devices=test_devices,
    test_options={
        "stage_path": "None",
        "num_frames": 500 if supports_load_during_graph_capture else 20,
        "use_cuda_graph": supports_load_during_graph_capture,
    },
    test_options_cpu={"num_frames": 10},
)
add_example_test(
    TestBasicRobotExamples,
    name="example_quadruped",
    devices=test_devices,
    test_options={"stage_path": "None", "num_frames": 1000},
    test_options_cpu={"num_envs": 10},
)


class TestAdvancedRobotExamples(unittest.TestCase):
    pass


add_example_test(
    TestAdvancedRobotExamples,
    name="example_robot_manipulating_cloth",
    devices=test_devices,
    test_options={"stage_path": "None", "num_frames": 300},
    test_options_cpu={"num_frames": 2},
)
add_example_test(
    TestAdvancedRobotExamples,
    name="example_anymal_c_walk",
    devices=cuda_test_devices,
    test_options={"stage_path": "None", "num_frames": 200, "headless": True, "torch_required": True},
)
add_example_test(
    TestAdvancedRobotExamples,
    name="example_anymal_c_walk_on_sand",
    devices=cuda_test_devices,
    test_options={"stage_path": "None", "num_frames": 100, "headless": True, "torch_required": True},
)


class TestSelectionAPIExamples(unittest.TestCase):
    pass


add_example_test(
    TestSelectionAPIExamples,
    name="example_selection_articulations",
    devices=test_devices,
    test_options={"stage_path": "None", "num_frames": 100, "use_cuda_graph": supports_load_during_graph_capture},
    test_options_cpu={"num_frames": 10},
)
add_example_test(
    TestSelectionAPIExamples,
    name="example_selection_cartpole",
    devices=test_devices,
    test_options={"stage_path": "None", "num_frames": 100, "use_cuda_graph": supports_load_during_graph_capture},
    test_options_cpu={"num_frames": 10},
)
add_example_test(
    TestSelectionAPIExamples,
    name="example_selection_materials",
    devices=test_devices,
    test_options={"stage_path": "None", "num_frames": 100, "use_cuda_graph": supports_load_during_graph_capture},
    test_options_cpu={"num_frames": 10},
)


class TestOtherExamples(unittest.TestCase):
    pass


add_example_test(
    TestOtherExamples,
    name="example_mpm_granular",
    devices=cuda_test_devices,
    test_options={"headless": True, "num_frames": 100},
)
add_example_test(
    TestOtherExamples,
    name="example_rigid_force",
    devices=test_devices,
    test_options={"headless": True},
)


if __name__ == "__main__":
    # force rebuild of all kernels
    wp.clear_kernel_cache()
    unittest.main(verbosity=2)
