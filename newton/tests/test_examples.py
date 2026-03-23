# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Test examples in the newton.examples package.

Currently, this script mainly checks that the examples can run. There are no
correctness checks.

The test parameters are typically tuned so that each test can run in 10 seconds
or less, ignoring module compilation time. A notable exception is the robot
manipulating cloth example, which takes approximately 35 seconds to run on a
CUDA device.
"""

import os
import re
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

# Patterns that are always filtered from subprocess stderr — infrastructure
# noise that is never a test concern.
_ALWAYS_FILTER_STDERR = [
    r"PXR_WORK_THREAD_LIMIT",
    r"^#{10,}$",
    r"Warp CUDA warning: Could not find or load the NVIDIA CUDA driver",
]


def _check_and_filter_stderr(
    test: unittest.TestCase,
    stderr: str,
    *,
    expected_stderr: list[str] | None = None,
    expected_stderr_cpu: list[str] | None = None,
    allowed_stderr: list[str] | None = None,
    allowed_stderr_cpu: list[str] | None = None,
    is_cpu: bool,
) -> str:
    """Assert expected patterns are present in *stderr*, then filter matching lines.

    Returns the filtered stderr with expected/allowed lines removed.
    """
    expected_patterns = list(expected_stderr or [])
    if is_cpu:
        expected_patterns.extend(expected_stderr_cpu or [])
    for pattern in expected_patterns:
        test.assertRegex(stderr, pattern, f"Expected stderr pattern not found: {pattern}")

    filter_patterns = list(expected_patterns) + _ALWAYS_FILTER_STDERR
    filter_patterns.extend(allowed_stderr or [])
    if is_cpu:
        filter_patterns.extend(allowed_stderr_cpu or [])

    if stderr:
        filters = [re.compile(p) for p in filter_patterns]
        stderr = "\n".join(
            line
            for line in stderr.splitlines()
            if not any(f.search(line) for f in filters)
            and not re.match(r"^\s+self\.\w", line)  # warning source-code context
        )

    return stderr


def _build_command_line_options(test_options: dict[str, Any]) -> list:
    """Helper function to build command-line options from the test options dictionary."""
    additional_options = []

    for key, value in test_options.items():
        if isinstance(value, bool):
            # Default behavior expecting argparse.BooleanOptionalAction support
            additional_options.append(f"--{'no-' if not value else ''}{key.replace('_', '-')}")
        elif isinstance(value, list):
            additional_options.extend([f"--{key.replace('_', '-')}"] + [str(v) for v in value])
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
    use_viewer: bool = False,
    test_suffix: str | None = None,
    expected_stderr: list[str] | None = None,
    expected_stderr_cpu: list[str] | None = None,
    allowed_stderr: list[str] | None = None,
    allowed_stderr_cpu: list[str] | None = None,
):
    """Registers a Newton example to run on ``devices`` as a TestCase.

    Args:
        expected_stderr: Regex patterns expected in subprocess stderr on all
            devices.  Matching lines are filtered before output checking; the
            test fails if any pattern is absent.
        expected_stderr_cpu: Like *expected_stderr* but only asserted on CPU.
        allowed_stderr: Regex patterns that may appear in stderr on all
            devices.  Matching lines are filtered but their absence does
            **not** cause a failure.
        allowed_stderr_cpu: Like *allowed_stderr* but only filtered on CPU.
    """

    # verify the module exists (use package-relative path so this works from any CWD)
    _examples_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "examples")
    if not os.path.exists(os.path.join(_examples_dir, f"{name.replace('.', '/')}.py")):
        raise ValueError(f"Example {name} does not exist")

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
                import torch

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
        command.extend(["-m", f"newton.examples.{name}", "--device", str(device), "--test", "--quiet"])

        if not use_viewer:
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
        else:
            # new-style example, use null viewer for tests (no disk I/O needed)
            stage_path = "None"
            command.extend(["--viewer", "null"])
            # Remove viewer/stage_path from options so they can't override the null viewer
            options.pop("viewer", None)
            options.pop("stage_path", None)

        command.extend(_build_command_line_options(options))

        # Set the test timeout in seconds
        test_timeout = options.pop("test_timeout", 600)

        # Can set active=True when tuning the test parameters
        with wp.ScopedTimer(f"{name}_{sanitize_identifier(device)}", active=False):
            # Run the script as a subprocess
            result = subprocess.run(
                command, capture_output=True, text=True, env=env_vars, timeout=test_timeout, check=False
            )

        stderr = _check_and_filter_stderr(
            test,
            result.stderr,
            expected_stderr=expected_stderr,
            expected_stderr_cpu=expected_stderr_cpu,
            allowed_stderr=allowed_stderr,
            allowed_stderr_cpu=allowed_stderr_cpu,
            is_cpu=wp.get_device(device).is_cpu,
        )

        if stderr.strip():
            print(stderr)

        # Check the return code (0 is standard for success)
        test.assertEqual(
            result.returncode,
            0,
            msg=f"Failed with return code {result.returncode}, command: {' '.join(command)}\n\nOutput:\n{result.stdout}\n{result.stderr}",
        )

        # Clean up output file for old-style examples that may have created one
        if stage_path and stage_path != "None" and result.returncode == 0:
            try:
                os.remove(stage_path)
            except OSError:
                pass

    test_name = f"test_{name}_{test_suffix}" if test_suffix else f"test_{name}"
    add_function_test(cls, test_name, run, devices=devices)


cuda_test_devices = get_selected_cuda_test_devices(mode="basic")  # Don't test on multiple GPUs to save time
test_devices = get_test_devices(mode="basic")


class TestBasicExamples(unittest.TestCase):
    pass


add_example_test(TestBasicExamples, name="basic.example_basic_pendulum", devices=test_devices, use_viewer=True)

add_example_test(
    TestBasicExamples,
    name="basic.example_basic_urdf",
    devices=test_devices,
    test_options={"num-frames": 200},
    test_options_cpu={"world_count": 16},
    test_options_cuda={"world_count": 64},
    use_viewer=True,
    test_suffix="xpbd",
    expected_stderr=["Inertia validation corrected"],
)
add_example_test(
    TestBasicExamples,
    name="basic.example_basic_urdf",
    devices=test_devices,
    test_options={"num-frames": 200, "solver": "vbd"},
    test_options_cpu={"world_count": 16},
    test_options_cuda={"world_count": 64},
    use_viewer=True,
    test_suffix="vbd",
    expected_stderr=["Inertia validation corrected"],
)

add_example_test(TestBasicExamples, name="basic.example_basic_viewer", devices=test_devices, use_viewer=True)

add_example_test(TestBasicExamples, name="basic.example_basic_joints", devices=test_devices, use_viewer=True)

add_example_test(
    TestBasicExamples,
    name="basic.example_basic_shapes",
    devices=test_devices,
    use_viewer=True,
    test_options={"num-frames": 150},
    expected_stderr_cpu=[
        "mesh-mesh contacts will be skipped",
    ],
    allowed_stderr_cpu=[
        "Warp CUDA error 100: no CUDA-capable device is detected",
    ],
)


class TestCableExamples(unittest.TestCase):
    pass


add_example_test(
    TestCableExamples,
    name="cable.example_cable_twist",
    devices=test_devices,
    use_viewer=True,
    test_options={"num-frames": 20},
)
add_example_test(
    TestCableExamples,
    name="cable.example_cable_y_junction",
    devices=test_devices,
    use_viewer=True,
    test_options={"num-frames": 20},
    expected_stderr=["Inertia validation corrected"],
)
add_example_test(
    TestCableExamples,
    name="cable.example_cable_bundle_hysteresis",
    devices=test_devices,
    use_viewer=True,
    test_options={"num-frames": 20},
)
add_example_test(
    TestCableExamples,
    name="cable.example_cable_pile",
    devices=test_devices,
    use_viewer=True,
    test_options={"num-frames": 20},
)


class TestClothExamples(unittest.TestCase):
    pass


add_example_test(
    TestClothExamples,
    name="cloth.example_cloth_bending",
    devices=test_devices,
    test_options={"num-frames": 400},
    use_viewer=True,
)
add_example_test(
    TestClothExamples,
    name="cloth.example_cloth_hanging",
    devices=test_devices,
    test_options={},
    test_options_cpu={"width": 32, "height": 16, "num-frames": 10},
    use_viewer=True,
    test_suffix="vbd",
)
add_example_test(
    TestClothExamples,
    name="cloth.example_cloth_hanging",
    devices=test_devices,
    test_options={"solver": "style3d"},
    test_options_cpu={"width": 32, "height": 16, "num-frames": 10},
    use_viewer=True,
    test_suffix="style3d",
)
add_example_test(
    TestClothExamples,
    name="cloth.example_cloth_style3d",
    devices=cuda_test_devices,
    test_options={},
    test_options_cuda={"num-frames": 32},
    use_viewer=True,
    allowed_stderr=[
        "texture inputs are not yet supported",
        "_extract_preview_surface_properties",
    ],
)
add_example_test(
    TestClothExamples,
    name="cloth.example_cloth_h1",
    devices=cuda_test_devices,
    test_options={},
    test_options_cuda={"num-frames": 32},
    use_viewer=True,
    expected_stderr=["Inertia validation corrected"],
    allowed_stderr=[
        "texture inputs are not yet supported",
        "_extract_preview_surface_properties",
    ],
)
add_example_test(
    TestClothExamples,
    name="cloth.example_cloth_franka",
    devices=cuda_test_devices,
    test_options={"num-frames": 50},
    use_viewer=True,
)
add_example_test(
    TestClothExamples,
    name="cloth.example_cloth_twist",
    devices=cuda_test_devices,
    test_options={"num-frames": 100},
    use_viewer=True,
)
add_example_test(
    TestClothExamples,
    name="cloth.example_cloth_rollers",
    devices=cuda_test_devices,
    test_options={"num-frames": 200},
    use_viewer=True,
)


class TestRobotExamples(unittest.TestCase):
    pass


add_example_test(
    TestRobotExamples,
    name="robot.example_robot_cartpole",
    devices=test_devices,
    test_options={"usd_required": True, "num-frames": 100},
    test_options_cpu={"num-frames": 10},
    use_viewer=True,
)
add_example_test(
    TestRobotExamples,
    name="robot.example_robot_anymal_c_walk",
    devices=cuda_test_devices,
    test_options={"usd_required": True, "num-frames": 500, "torch_required": True},
    use_viewer=True,
    expected_stderr=["Inertia validation corrected"],
)
add_example_test(
    TestRobotExamples,
    name="robot.example_robot_anymal_d",
    devices=test_devices,
    test_options={"usd_required": True, "num-frames": 500},
    test_options_cpu={"num-frames": 10},
    use_viewer=True,
    expected_stderr=["Inertia validation corrected"],
    expected_stderr_cpu=[
        "mesh-mesh contacts will be skipped",
    ],
    allowed_stderr_cpu=[
        "Warp CUDA error 100: no CUDA-capable device is detected",
    ],
)
add_example_test(
    TestRobotExamples,
    name="robot.example_robot_g1",
    devices=cuda_test_devices,
    test_options={"usd_required": True, "num-frames": 500},
    use_viewer=True,
    expected_stderr=["Inertia validation corrected"],
)
add_example_test(
    TestRobotExamples,
    name="robot.example_robot_h1",
    devices=cuda_test_devices,
    test_options={"usd_required": True, "num-frames": 500},
    use_viewer=True,
    expected_stderr=["Inertia validation corrected"],
)
add_example_test(
    TestRobotExamples,
    name="robot.example_robot_ur10",
    devices=test_devices,
    test_options={"usd_required": True, "num-frames": 500},
    test_options_cpu={"num-frames": 10},
    use_viewer=True,
    expected_stderr=["possibly invalid inertia tensor", "Inertia validation corrected"],
)
add_example_test(
    TestRobotExamples,
    name="robot.example_robot_allegro_hand",
    devices=cuda_test_devices,
    test_options={"usd_required": True, "num-frames": 500},
    use_viewer=True,
    expected_stderr=[
        "possibly invalid inertia tensor",
        "authored mass and density without authored diagonalInertia",
        "return parse_usd",
    ],
    allowed_stderr=[
        "texture inputs are not yet supported",
        "_extract_preview_surface_properties",
    ],
)
add_example_test(
    TestRobotExamples,
    name="robot.example_robot_panda_hydro",
    devices=cuda_test_devices,
    test_options={"usd_required": True, "num-frames": 720},
    use_viewer=True,
    expected_stderr=["Inertia validation corrected"],
)


class TestRobotPolicyExamples(unittest.TestCase):
    pass


add_example_test(
    TestRobotPolicyExamples,
    name="robot.example_robot_policy",
    devices=cuda_test_devices,
    test_options={"num-frames": 500, "torch_required": True, "robot": "g1_29dof"},
    test_options_cpu={"num-frames": 10},
    use_viewer=True,
    test_suffix="G1_29dof",
    expected_stderr=["Inertia validation corrected"],
)
add_example_test(
    TestRobotPolicyExamples,
    name="robot.example_robot_policy",
    devices=cuda_test_devices,
    test_options={"num-frames": 500, "torch_required": True, "robot": "g1_23dof"},
    use_viewer=True,
    test_suffix="G1_23dof",
    expected_stderr=["Inertia validation corrected"],
)
add_example_test(
    TestRobotPolicyExamples,
    name="robot.example_robot_policy",
    devices=cuda_test_devices,
    test_options={"num-frames": 500, "torch_required": True, "robot": "g1_23dof", "physx": True},
    use_viewer=True,
    test_suffix="G1_23dof_Physx",
    expected_stderr=["Inertia validation corrected"],
)
add_example_test(
    TestRobotPolicyExamples,
    name="robot.example_robot_policy",
    devices=cuda_test_devices,
    test_options={"num-frames": 500, "torch_required": True, "robot": "anymal"},
    use_viewer=True,
    test_suffix="Anymal",
    expected_stderr=["Inertia validation corrected"],
)
add_example_test(
    TestRobotPolicyExamples,
    name="robot.example_robot_policy",
    devices=cuda_test_devices,
    test_options={"num-frames": 500, "torch_required": True, "robot": "anymal", "physx": True},
    use_viewer=True,
    test_suffix="Anymal_Physx",
    expected_stderr=["Inertia validation corrected"],
)
add_example_test(
    TestRobotPolicyExamples,
    name="robot.example_robot_policy",
    devices=cuda_test_devices,
    test_options={"torch_required": True},
    test_options_cuda={"num-frames": 500, "robot": "go2"},
    use_viewer=True,
    test_suffix="Go2",
    expected_stderr=["Inertia validation corrected"],
)
add_example_test(
    TestRobotPolicyExamples,
    name="robot.example_robot_policy",
    devices=cuda_test_devices,
    test_options={"torch_required": True},
    test_options_cuda={"num-frames": 500, "robot": "go2", "physx": True},
    use_viewer=True,
    test_suffix="Go2_Physx",
    expected_stderr=["Inertia validation corrected"],
)


class TestAdvancedRobotExamples(unittest.TestCase):
    pass


add_example_test(
    TestAdvancedRobotExamples,
    name="mpm.example_mpm_anymal",
    devices=cuda_test_devices,
    test_options={"num-frames": 100, "torch_required": True},
    use_viewer=True,
    expected_stderr=["Inertia validation corrected"],
)


class TestIKExamples(unittest.TestCase):
    pass


add_example_test(
    TestIKExamples,
    name="ik.example_ik_franka",
    devices=test_devices,
    use_viewer=True,
    expected_stderr=["Inertia validation corrected"],
)

add_example_test(
    TestIKExamples,
    name="ik.example_ik_h1",
    devices=test_devices,
    use_viewer=True,
    expected_stderr=["Inertia validation corrected"],
)

add_example_test(
    TestIKExamples,
    name="ik.example_ik_custom",
    devices=cuda_test_devices,
    use_viewer=True,
    expected_stderr=["Inertia validation corrected"],
)

add_example_test(
    TestIKExamples,
    name="ik.example_ik_cube_stacking",
    test_options_cuda={"world-count": 16, "num-frames": 2000},
    devices=cuda_test_devices,
    use_viewer=True,
    expected_stderr=["Inertia validation corrected"],
)


class TestSelectionAPIExamples(unittest.TestCase):
    pass


add_example_test(
    TestSelectionAPIExamples,
    name="selection.example_selection_articulations",
    devices=test_devices,
    test_options={"num-frames": 100},
    test_options_cpu={"num-frames": 10},
    use_viewer=True,
    expected_stderr=["Inertia validation corrected"],
)
add_example_test(
    TestSelectionAPIExamples,
    name="selection.example_selection_cartpole",
    devices=test_devices,
    test_options={"num-frames": 100},
    test_options_cpu={"num-frames": 10},
    use_viewer=True,
)
add_example_test(
    TestSelectionAPIExamples,
    name="selection.example_selection_materials",
    devices=test_devices,
    test_options={"num-frames": 100},
    test_options_cpu={"num-frames": 10},
    use_viewer=True,
    expected_stderr=["Inertia validation corrected"],
)
add_example_test(
    TestSelectionAPIExamples,
    name="selection.example_selection_multiple",
    devices=test_devices,
    test_options={"num-frames": 100},
    test_options_cpu={"num-frames": 10},
    use_viewer=True,
    expected_stderr=["Inertia validation corrected"],
)


class TestDiffSimExamples(unittest.TestCase):
    pass


add_example_test(
    TestDiffSimExamples,
    name="diffsim.example_diffsim_ball",
    devices=test_devices,
    test_options={"num-frames": 4 * 36},  # train_iters * sim_steps
    test_options_cpu={"num-frames": 2 * 36},
    use_viewer=True,
)

add_example_test(
    TestDiffSimExamples,
    name="diffsim.example_diffsim_cloth",
    devices=test_devices,
    test_options={"num-frames": 4 * 120},  # train_iters * sim_steps
    test_options_cpu={"num-frames": 2 * 120},
    use_viewer=True,
)

add_example_test(
    TestDiffSimExamples,
    name="diffsim.example_diffsim_drone",
    devices=test_devices,
    test_options={"num-frames": 180},  # sim_steps
    test_options_cpu={"num-frames": 10},
    use_viewer=True,
)

add_example_test(
    TestDiffSimExamples,
    name="diffsim.example_diffsim_spring_cage",
    devices=test_devices,
    test_options={"num-frames": 4 * 30},  # train_iters * sim_steps
    test_options_cpu={"num-frames": 2 * 30},
    use_viewer=True,
)

add_example_test(
    TestDiffSimExamples,
    name="diffsim.example_diffsim_soft_body",
    devices=test_devices,
    test_options={"num-frames": 4 * 60},  # train_iters * sim_steps
    test_options_cpu={"num-frames": 2 * 60},
    use_viewer=True,
)

add_example_test(
    TestDiffSimExamples,
    name="diffsim.example_diffsim_bear",
    devices=test_devices,
    test_options={"usd_required": True, "num-frames": 4 * 60},  # train_iters * sim_steps
    test_options_cpu={"num-frames": 2, "sim-steps": 10},
    use_viewer=True,
)


class TestSensorExamples(unittest.TestCase):
    pass


add_example_test(
    TestSensorExamples,
    name="sensors.example_sensor_contact",
    devices=test_devices,
    test_options={"num-frames": 160},  # required for ball to reach plate
    use_viewer=True,
    expected_stderr=[
        "possibly invalid inertia tensor",
        "zero mass and zero inertia",
        "return parse_usd",  # warning source-context line
    ],
)

add_example_test(
    TestSensorExamples,
    name="sensors.example_sensor_tiled_camera",
    devices=cuda_test_devices,
    test_options={"num-frames": 4 * 36},  # train_iters * sim_steps
    use_viewer=True,
    expected_stderr=["Inertia validation corrected"],
)

add_example_test(
    TestSensorExamples,
    name="sensors.example_sensor_imu",
    devices=test_devices,
    test_options={"num-frames": 200},  # allow cubes to settle
    use_viewer=True,
)


class TestMPMExamples(unittest.TestCase):
    pass


add_example_test(
    TestMPMExamples,
    name="mpm.example_mpm_granular",
    devices=cuda_test_devices,
    test_options={"num-frames": 100},
    use_viewer=True,
)

add_example_test(
    TestMPMExamples,
    name="mpm.example_mpm_multi_material",
    devices=cuda_test_devices,
    test_options={"num-frames": 10},
    use_viewer=True,
)

add_example_test(
    TestMPMExamples,
    name="mpm.example_mpm_grain_rendering",
    devices=cuda_test_devices,
    test_options={"num-frames": 10},
    use_viewer=True,
)

add_example_test(
    TestMPMExamples,
    name="mpm.example_mpm_twoway_coupling",
    devices=cuda_test_devices,
    test_options={"num-frames": 80},
    use_viewer=True,
)

add_example_test(
    TestMPMExamples,
    name="mpm.example_mpm_beam_twist",
    devices=cuda_test_devices,
    test_options={"num-frames": 100},
    use_viewer=True,
)

add_example_test(
    TestMPMExamples,
    name="mpm.example_mpm_snow_ball",
    devices=cuda_test_devices,
    test_options={"num-frames": 30, "voxel-size": 0.2},
    use_viewer=True,
)

add_example_test(
    TestMPMExamples,
    name="mpm.example_mpm_viscous",
    devices=cuda_test_devices,
    test_options={"num-frames": 30, "voxel-size": 0.01},
    use_viewer=True,
)


add_example_test(
    TestBasicExamples,
    name="basic.example_basic_plotting",
    devices=test_devices,
    test_options={"num-frames": 200},
    use_viewer=True,
    expected_stderr=["Inertia validation corrected"],
)


class TestContactsExamples(unittest.TestCase):
    pass


add_example_test(
    TestContactsExamples,
    name="contacts.example_nut_bolt_sdf",
    devices=cuda_test_devices,
    test_options={"num-frames": 120, "world-count": 1},
    use_viewer=True,
)
add_example_test(
    TestContactsExamples,
    name="contacts.example_nut_bolt_hydro",
    devices=cuda_test_devices,
    test_options={"num-frames": 120, "world-count": 1},
    use_viewer=True,
)
add_example_test(
    TestContactsExamples,
    name="contacts.example_brick_stacking",
    devices=cuda_test_devices,
    test_options={"num-frames": 1200},
    use_viewer=True,
    expected_stderr=["Inertia validation corrected"],
)
add_example_test(
    TestContactsExamples,
    name="contacts.example_pyramid",
    devices=cuda_test_devices,
    test_options={"num-frames": 120, "num-pyramids": 3, "pyramid-size": 5},
    use_viewer=True,
)


class TestMultiphysicsExamples(unittest.TestCase):
    pass


add_example_test(
    TestMultiphysicsExamples,
    name="multiphysics.example_softbody_gift",
    devices=cuda_test_devices,
    test_options={"num-frames": 200},
    use_viewer=True,
    expected_stderr=["Detected non-manifold edge"],
)
add_example_test(
    TestMultiphysicsExamples,
    name="cloth.example_cloth_poker_cards",
    devices=cuda_test_devices,
    test_options={"num-frames": 30},
    use_viewer=True,
)
add_example_test(
    TestMultiphysicsExamples,
    name="multiphysics.example_softbody_dropping_to_cloth",
    devices=cuda_test_devices,
    test_options={"num-frames": 200},
    use_viewer=True,
)


class TestSoftbodyExamples(unittest.TestCase):
    pass


add_example_test(
    TestSoftbodyExamples,
    name="softbody.example_softbody_hanging",
    devices=cuda_test_devices,
    test_options={"num-frames": 120},
    use_viewer=True,
)


class TestStderrFiltering(unittest.TestCase):
    """Tests for _check_and_filter_stderr used by add_example_test."""

    def test_expected_stderr_filters_matching_lines(self):
        stderr = "Warning: mesh-mesh contacts will be skipped\nreal output"
        result = _check_and_filter_stderr(
            self, stderr, expected_stderr=["mesh-mesh contacts will be skipped"], is_cpu=False
        )
        self.assertEqual(result, "real output")

    def test_expected_stderr_fails_when_absent(self):
        stderr = "some unrelated output"
        with self.assertRaises(AssertionError):
            _check_and_filter_stderr(self, stderr, expected_stderr=["missing pattern"], is_cpu=False)

    def test_expected_stderr_cpu_asserted_on_cpu(self):
        stderr = "Warning: mesh-mesh contacts will be skipped"
        result = _check_and_filter_stderr(self, stderr, expected_stderr_cpu=["mesh-mesh contacts"], is_cpu=True)
        self.assertEqual(result, "")

    def test_expected_stderr_cpu_ignored_on_cuda(self):
        stderr = "some output"
        # Should not assert — expected_stderr_cpu is skipped on CUDA
        result = _check_and_filter_stderr(self, stderr, expected_stderr_cpu=["pattern not present"], is_cpu=False)
        self.assertEqual(result, "some output")

    def test_allowed_stderr_cpu_filters_without_asserting(self):
        stderr = "Warp CUDA error 100: no CUDA-capable device is detected\nreal output"
        result = _check_and_filter_stderr(self, stderr, allowed_stderr_cpu=["Warp CUDA error 100"], is_cpu=True)
        self.assertEqual(result, "real output")

    def test_allowed_stderr_cpu_absent_does_not_fail(self):
        stderr = "real output"
        # Pattern absent — should NOT raise because it's allowed, not expected
        result = _check_and_filter_stderr(self, stderr, allowed_stderr_cpu=["Warp CUDA error 100"], is_cpu=True)
        self.assertEqual(result, "real output")

    def test_allowed_stderr_cpu_ignored_on_cuda(self):
        stderr = "Warp CUDA error 100: no CUDA-capable device is detected"
        # On CUDA, allowed_stderr_cpu should not filter
        result = _check_and_filter_stderr(self, stderr, allowed_stderr_cpu=["Warp CUDA error 100"], is_cpu=False)
        self.assertEqual(result, "Warp CUDA error 100: no CUDA-capable device is detected")

    def test_warning_source_context_lines_filtered(self):
        stderr = "Warning: something\n  self.some_call()"
        result = _check_and_filter_stderr(self, stderr, expected_stderr=["Warning: something"], is_cpu=False)
        self.assertEqual(result, "")

    def test_unmatched_lines_preserved(self):
        stderr = "expected warning\nunexpected output\nanother expected"
        result = _check_and_filter_stderr(
            self, stderr, expected_stderr=["expected warning", "another expected"], is_cpu=False
        )
        self.assertEqual(result, "unexpected output")

    def test_always_filter_pxr_banner(self):
        stderr = (
            "##################################################################\n"
            "#  PXR_WORK_THREAD_LIMIT is overridden to '1'.  Default is '0'.  #\n"
            "##################################################################\n"
            "real output"
        )
        result = _check_and_filter_stderr(self, stderr, is_cpu=False)
        self.assertEqual(result, "real output")

    def test_inertia_validation_not_always_filtered(self):
        stderr = (
            "/path/to/example.py:72: UserWarning: Inertia validation corrected 120 bodies."
            " Set validate_inertia_detailed=True for detailed per-body warnings."
        )
        # Without expected_stderr, the warning is NOT filtered
        result = _check_and_filter_stderr(self, stderr, is_cpu=False)
        self.assertEqual(result, stderr)


if __name__ == "__main__":
    unittest.main(verbosity=2)
