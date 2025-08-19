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
import unittest
from typing import Any, Callable

import warp as wp

import newton.examples
import newton.tests.unittest_utils
from newton.tests.unittest_utils import (
    USD_AVAILABLE,
    add_function_test,
    get_selected_cuda_test_devices,
    get_test_devices,
    sanitize_identifier,
)

wp.init()


def add_example_test(
    cls: type,
    name: str,
    func: Callable,
    devices: list,
    cpu_args: dict[str, Any] | None = None,
    cuda_args: dict[str, Any] | None = None,
    num_frames: int = 100,
    requires_torch: bool = False,
    requires_usd: bool = False,
    timeout: int = 600,
):
    """Registers a Newton example to run on ``devices`` as a TestCase."""

    if cpu_args is None:
        cpu_args = {}
    if cuda_args is None:
        cuda_args = {}

    def run(test, device):
        # Mark the test as skipped if Torch is not installed but required
        if requires_torch:
            try:
                import torch  # noqa: PLC0415

                if wp.get_device(device).is_cuda and not torch.cuda.is_available():
                    # Ensure torch has CUDA support
                    test.skipTest("Torch not compiled with CUDA support")

            except Exception as e:
                test.skipTest(f"{e}")

        # Mark the test as skipped if USD is not installed but required
        if requires_usd and not USD_AVAILABLE:
            test.skipTest("Requires usd-core")

        # Find the current Warp cache
        warp_cache_path = wp.config.kernel_cache_dir

        env_vars = os.environ.copy()
        if warp_cache_path is not None:
            env_vars["WARP_CACHE_PATH"] = warp_cache_path

        if USD_AVAILABLE:
            stage_path = f"outputs/{name}_{sanitize_identifier(device)}.usd"
            viewer = newton.viewer.ViewerUSD(output_path=stage_path)
        else:
            stage_path = None
            viewer = newton.viewer.ViewerNull()

        # construct the example
        args = cpu_args if wp.get_device(device).is_cpu else cuda_args
        example = func(viewer, args)

        # run for N frames
        for _ in range(num_frames):
            example.step()
            example.render()
            example.test()

        viewer.close()

        # If the test succeeded, try to clean up the output by default
        if stage_path:
            try:
                os.remove(stage_path)
            except OSError:
                pass

    add_function_test(cls, f"test_{name}", run, devices=devices, check_output=False)


cuda_test_devices = get_selected_cuda_test_devices(mode="basic")  # Don't test on multiple GPUs to save time
test_devices = get_test_devices(mode="basic")


class TestBasicExamples(unittest.TestCase):
    pass


add_example_test(
    TestBasicExamples,
    "example_basic_pendulum",
    func=lambda viewer, args: newton.examples.basic_pendulum.Example(viewer, **args),
    devices=test_devices,
    num_frames=100,
)

add_example_test(
    TestBasicExamples,
    "example_basic_urdf",
    func=lambda viewer, args: newton.examples.basic_urdf.Example(viewer, **args),
    devices=test_devices,
    cpu_args={"num_envs": 16},
    cuda_args={"num_envs": 64},
    num_frames=100,
)

add_example_test(
    TestBasicExamples,
    "example_basic_viewer",
    func=lambda viewer, args: newton.examples.basic_viewer.Example(viewer, **args),
    devices=test_devices,
    num_frames=100,
)


class TestClothExamples(unittest.TestCase):
    pass


class TestDiffSimExamples(unittest.TestCase):
    pass


class TestIKExamples(unittest.TestCase):
    pass


class TestMPMExamples(unittest.TestCase):
    pass


class TestRobotExamples(unittest.TestCase):
    pass


class TestPolicyExamples(unittest.TestCase):
    pass


class TestSelectionExamples(unittest.TestCase):
    pass


class TestSensorExamples(unittest.TestCase):
    pass


if __name__ == "__main__":
    # force rebuild of all kernels
    # wp.clear_kernel_cache()
    unittest.main(verbosity=2)
