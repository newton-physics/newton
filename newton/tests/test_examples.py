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

import importlib
import os
import re
import unittest
import warnings
from typing import Any

import warp as wp

import newton.examples
import newton.viewer
from newton.tests.unittest_utils import (
    USD_AVAILABLE,
    CheckOutput,
    add_function_test,
    get_selected_cuda_test_devices,
    get_test_devices,
)


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
    test_suffix: str | None = None,
    expected_output: list[str] | None = None,
    expected_output_cpu: list[str] | None = None,
    allowed_output: list[str] | None = None,
):
    """Registers a Newton example to run on ``devices`` as a TestCase.

    Args:
        expected_output: Regex patterns expected in Python warnings or
            stdout on all devices.  Each pattern must match at least once
            in the combined output.  Any warning or stdout line not
            matched by at least one pattern fails the test.
        expected_output_cpu: Like *expected_output* but only asserted on
            CPU devices.
        allowed_output: Regex patterns for stdout lines that are allowed
            but not required.  These prevent unexpected-output failures
            without requiring a match.
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
                    test.skipTest("Torch not compiled with CUDA support")

            except ImportError as e:
                test.skipTest(f"torch not available: {e}")

        # Mark the test as skipped if USD is not installed but required
        usd_required = options.pop("usd_required", False)
        if usd_required and not USD_AVAILABLE:
            test.skipTest("Requires usd-core")

        # Import the example module and get its parser
        mod = importlib.import_module(f"newton.examples.{name}")
        parser = getattr(mod.Example, "create_parser", newton.examples.create_parser)()

        # Build CLI args and parse through the example's own parser.
        # Use parse_known_args so options not in the parser (e.g. --solver
        # for examples that only define it in __main__) don't cause errors.
        # Any unrecognized options are set directly on the namespace so they
        # are available via args.<name> in the Example constructor.
        num_frames = options.pop("num-frames", options.pop("num_frames", 100))
        cli_args = [
            "--device",
            str(device),
            "--test",
            "--quiet",
            "--viewer",
            "null",
            "--num-frames",
            str(num_frames),
        ]
        cli_args.extend(_build_command_line_options(options))
        args, remaining = parser.parse_known_args(cli_args)
        # Set unrecognized --key value pairs on args namespace
        i = 0
        while i < len(remaining):
            if remaining[i].startswith("--"):
                key = remaining[i].lstrip("-").replace("-", "_")
                if i + 1 < len(remaining) and not remaining[i + 1].startswith("--"):
                    setattr(args, key, remaining[i + 1])
                    i += 2
                else:
                    setattr(args, key, True)
                    i += 1
            else:
                i += 1

        # Build expected pattern list (used for both warning and stdout checks)
        is_cpu = wp.get_device(device).is_cpu
        expected_patterns = list(expected_output or [])
        if is_cpu:
            expected_patterns.extend(expected_output_cpu or [])

        # CheckOutput uses both expected and allowed patterns for filtering;
        # only expected_patterns are asserted to appear at least once.
        all_patterns = expected_patterns + list(allowed_output or [])
        compiled_patterns = [re.compile(p) for p in expected_patterns] if expected_patterns else []

        # Run example in-process.
        # CheckOutput captures stdout and fails on unexpected lines.
        # warnings.catch_warnings captures Python warnings for separate validation.
        viewer = newton.viewer.ViewerNull(num_frames=args.num_frames)
        factory = getattr(mod, "create_example", None) or mod.Example
        with CheckOutput(test, expected_patterns=all_patterns) as check:
            with wp.ScopedDevice(device), warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                example = factory(viewer, args)
                newton.examples.run(example, args)

        # Assert each expected pattern appears in warnings or stdout
        warning_messages = [str(w.message) for w in caught]
        all_text = "\n".join(warning_messages)
        if check.output.strip():
            all_text += "\n" + check.output
        for pattern in compiled_patterns:
            test.assertRegex(all_text, pattern, f"Expected output pattern not found: {pattern.pattern}")

        # Fail on unexpected warnings originating from newton code
        newton_root = os.path.dirname(os.path.dirname(newton.examples.__file__))
        for w in caught:
            if not str(w.filename).startswith(newton_root):
                continue
            msg = str(w.message)
            if not any(p.search(msg) for p in compiled_patterns):
                test.fail(f"Unexpected warning: {msg}")

    test_name = f"test_{name}_{test_suffix}" if test_suffix else f"test_{name}"
    add_function_test(cls, test_name, run, devices=devices, check_output=False)


cuda_test_devices = get_selected_cuda_test_devices(mode="basic")  # Don't test on multiple GPUs to save time
test_devices = get_test_devices(mode="basic")


class TestBasicExamples(unittest.TestCase):
    pass


add_example_test(TestBasicExamples, name="basic.example_basic_pendulum", devices=test_devices)

add_example_test(
    TestBasicExamples,
    name="basic.example_basic_urdf",
    devices=test_devices,
    test_options={"num-frames": 200},
    test_options_cpu={"world_count": 16},
    test_options_cuda={"world_count": 64},
    test_suffix="xpbd",
)
add_example_test(
    TestBasicExamples,
    name="basic.example_basic_urdf",
    devices=test_devices,
    test_options={"num-frames": 200, "solver": "vbd"},
    test_options_cpu={"world_count": 16},
    test_options_cuda={"world_count": 64},
    test_suffix="vbd",
)

add_example_test(TestBasicExamples, name="basic.example_basic_viewer", devices=test_devices)

add_example_test(TestBasicExamples, name="basic.example_basic_joints", devices=test_devices)

add_example_test(
    TestBasicExamples,
    name="basic.example_basic_shapes",
    devices=test_devices,
    test_options={"num-frames": 150},
    expected_output_cpu=[
        "mesh-mesh contacts will be skipped",
    ],
)


class TestCableExamples(unittest.TestCase):
    pass


add_example_test(
    TestCableExamples,
    name="cable.example_cable_twist",
    devices=test_devices,
    test_options={"num-frames": 20},
)
add_example_test(
    TestCableExamples,
    name="cable.example_cable_y_junction",
    devices=test_devices,
    test_options={"num-frames": 20},
)
add_example_test(
    TestCableExamples,
    name="cable.example_cable_bundle_hysteresis",
    devices=test_devices,
    test_options={"num-frames": 20},
)
add_example_test(
    TestCableExamples,
    name="cable.example_cable_pile",
    devices=test_devices,
    test_options={"num-frames": 20},
)


class TestClothExamples(unittest.TestCase):
    pass


add_example_test(
    TestClothExamples,
    name="cloth.example_cloth_bending",
    devices=test_devices,
    test_options={"num-frames": 400},
)
add_example_test(
    TestClothExamples,
    name="cloth.example_cloth_hanging",
    devices=test_devices,
    test_options={},
    test_options_cpu={"width": 32, "height": 16, "num-frames": 10},
    test_suffix="vbd",
)
add_example_test(
    TestClothExamples,
    name="cloth.example_cloth_hanging",
    devices=test_devices,
    test_options={"solver": "style3d"},
    test_options_cpu={"width": 32, "height": 16, "num-frames": 10},
    test_suffix="style3d",
)
add_example_test(
    TestClothExamples,
    name="cloth.example_cloth_style3d",
    devices=cuda_test_devices,
    test_options={},
    test_options_cuda={"num-frames": 32},
    expected_output=[
        "texture inputs are not yet supported",
        "2-dimensional vectors are deprecated",
        "SolverStyle3D::precompute",
    ],
)
add_example_test(
    TestClothExamples,
    name="cloth.example_cloth_h1",
    devices=cuda_test_devices,
    test_options={},
    test_options_cuda={"num-frames": 32},
    expected_output=[
        "texture inputs are not yet supported",
        "2-dimensional vectors are deprecated",
        "SolverStyle3D::precompute",
    ],
)
add_example_test(
    TestClothExamples,
    name="cloth.example_cloth_franka",
    devices=cuda_test_devices,
    test_options={"num-frames": 50},
)
add_example_test(
    TestClothExamples,
    name="cloth.example_cloth_twist",
    devices=cuda_test_devices,
    test_options={"num-frames": 100},
)
add_example_test(
    TestClothExamples,
    name="cloth.example_cloth_rollers",
    devices=cuda_test_devices,
    test_options={"num-frames": 200},
)


class TestRobotExamples(unittest.TestCase):
    pass


add_example_test(
    TestRobotExamples,
    name="robot.example_robot_cartpole",
    devices=test_devices,
    test_options={"usd_required": True, "num-frames": 100},
    test_options_cpu={"num-frames": 10},
)
add_example_test(
    TestRobotExamples,
    name="robot.example_robot_anymal_c_walk",
    devices=cuda_test_devices,
    test_options={"usd_required": True, "num-frames": 500, "torch_required": True},
)
add_example_test(
    TestRobotExamples,
    name="robot.example_robot_anymal_d",
    devices=test_devices,
    test_options={"usd_required": True, "num-frames": 500},
    test_options_cpu={"num-frames": 10},
    expected_output_cpu=[
        "mesh-mesh contacts will be skipped",
    ],
)
add_example_test(
    TestRobotExamples,
    name="robot.example_robot_g1",
    devices=cuda_test_devices,
    test_options={"usd_required": True, "num-frames": 500},
)
add_example_test(
    TestRobotExamples,
    name="robot.example_robot_h1",
    devices=cuda_test_devices,
    test_options={"usd_required": True, "num-frames": 500},
)
add_example_test(
    TestRobotExamples,
    name="robot.example_robot_ur10",
    devices=test_devices,
    test_options={"usd_required": True, "num-frames": 500},
    test_options_cpu={"num-frames": 10},
)
add_example_test(
    TestRobotExamples,
    name="robot.example_robot_allegro_hand",
    devices=cuda_test_devices,
    test_options={"usd_required": True, "num-frames": 500},
    expected_output=[
        "authored mass and density without authored diagonalInertia",
    ],
)
add_example_test(
    TestRobotExamples,
    name="robot.example_robot_panda_hydro",
    devices=cuda_test_devices,
    test_options={"usd_required": True, "num-frames": 720},
)


class TestRobotPolicyExamples(unittest.TestCase):
    pass


add_example_test(
    TestRobotPolicyExamples,
    name="robot.example_robot_policy",
    devices=cuda_test_devices,
    test_options={"num-frames": 500, "torch_required": True, "robot": "g1_29dof"},
    test_options_cpu={"num-frames": 10},
    test_suffix="G1_29dof",
    expected_output=["\\[INFO\\]"],
)
add_example_test(
    TestRobotPolicyExamples,
    name="robot.example_robot_policy",
    devices=cuda_test_devices,
    test_options={"num-frames": 500, "torch_required": True, "robot": "g1_23dof"},
    test_suffix="G1_23dof",
    expected_output=["\\[INFO\\]"],
)
add_example_test(
    TestRobotPolicyExamples,
    name="robot.example_robot_policy",
    devices=cuda_test_devices,
    test_options={"num-frames": 500, "torch_required": True, "robot": "g1_23dof", "physx": True},
    test_suffix="G1_23dof_Physx",
    expected_output=["\\[INFO\\]"],
)
add_example_test(
    TestRobotPolicyExamples,
    name="robot.example_robot_policy",
    devices=cuda_test_devices,
    test_options={"num-frames": 500, "torch_required": True, "robot": "anymal"},
    test_suffix="Anymal",
    expected_output=["\\[INFO\\]"],
)
add_example_test(
    TestRobotPolicyExamples,
    name="robot.example_robot_policy",
    devices=cuda_test_devices,
    test_options={"num-frames": 500, "torch_required": True, "robot": "anymal", "physx": True},
    test_suffix="Anymal_Physx",
    expected_output=["\\[INFO\\]"],
)
add_example_test(
    TestRobotPolicyExamples,
    name="robot.example_robot_policy",
    devices=cuda_test_devices,
    test_options={"torch_required": True},
    test_options_cuda={"num-frames": 500, "robot": "go2"},
    test_suffix="Go2",
    expected_output=["\\[INFO\\]"],
)
add_example_test(
    TestRobotPolicyExamples,
    name="robot.example_robot_policy",
    devices=cuda_test_devices,
    test_options={"torch_required": True},
    test_options_cuda={"num-frames": 500, "robot": "go2", "physx": True},
    test_suffix="Go2_Physx",
    expected_output=["\\[INFO\\]"],
)


class TestAdvancedRobotExamples(unittest.TestCase):
    pass


add_example_test(
    TestAdvancedRobotExamples,
    name="mpm.example_mpm_anymal",
    devices=cuda_test_devices,
    test_options={"num-frames": 100, "torch_required": True},
)


class TestIKExamples(unittest.TestCase):
    pass


add_example_test(
    TestIKExamples,
    name="ik.example_ik_franka",
    devices=test_devices,
)

add_example_test(
    TestIKExamples,
    name="ik.example_ik_h1",
    devices=test_devices,
)

add_example_test(
    TestIKExamples,
    name="ik.example_ik_custom",
    devices=cuda_test_devices,
)

add_example_test(
    TestIKExamples,
    name="ik.example_ik_cube_stacking",
    test_options_cuda={"world-count": 16, "num-frames": 2000},
    devices=cuda_test_devices,
    expected_output=["World success rate"],
)


class TestSelectionAPIExamples(unittest.TestCase):
    pass


add_example_test(
    TestSelectionAPIExamples,
    name="selection.example_selection_articulations",
    devices=test_devices,
    test_options={"num-frames": 100},
    test_options_cpu={"num-frames": 10},
    expected_output=["Articulation|Link|Joint|Shape|Fixed|Floating|DOF|\\["],
)
add_example_test(
    TestSelectionAPIExamples,
    name="selection.example_selection_cartpole",
    devices=test_devices,
    test_options={"num-frames": 100},
    test_options_cpu={"num-frames": 10},
    expected_output=["Articulation|Link|Joint|Shape|Fixed|Floating|DOF|\\["],
)
add_example_test(
    TestSelectionAPIExamples,
    name="selection.example_selection_materials",
    devices=test_devices,
    test_options={"num-frames": 100},
    test_options_cpu={"num-frames": 10},
    expected_output=["Articulation|Link|Joint|Shape|Fixed|Floating|DOF|\\["],
)
add_example_test(
    TestSelectionAPIExamples,
    name="selection.example_selection_multiple",
    devices=test_devices,
    test_options={"num-frames": 100},
    test_options_cpu={"num-frames": 10},
    expected_output=["Articulation|Link|Joint|Shape|Fixed|Floating|DOF|\\["],
)


class TestDiffSimExamples(unittest.TestCase):
    pass


add_example_test(
    TestDiffSimExamples,
    name="diffsim.example_diffsim_ball",
    devices=test_devices,
    test_options={"num-frames": 4 * 36},  # train_iters * sim_steps
    test_options_cpu={"num-frames": 2 * 36},
    expected_output=["numeric grad:", "analytic grad:"],
)

add_example_test(
    TestDiffSimExamples,
    name="diffsim.example_diffsim_cloth",
    devices=test_devices,
    test_options={"num-frames": 4 * 120},  # train_iters * sim_steps
    test_options_cpu={"num-frames": 2 * 120},
)

add_example_test(
    TestDiffSimExamples,
    name="diffsim.example_diffsim_drone",
    devices=test_devices,
    test_options={"num-frames": 180},  # sim_steps
    test_options_cpu={"num-frames": 10},
    expected_output=["loss=", "flight target"],
)

add_example_test(
    TestDiffSimExamples,
    name="diffsim.example_diffsim_spring_cage",
    devices=test_devices,
    test_options={"num-frames": 4 * 30},  # train_iters * sim_steps
    test_options_cpu={"num-frames": 2 * 30},
)

add_example_test(
    TestDiffSimExamples,
    name="diffsim.example_diffsim_soft_body",
    devices=test_devices,
    test_options={"num-frames": 4 * 60},  # train_iters * sim_steps
    test_options_cpu={"num-frames": 2 * 60},
)

add_example_test(
    TestDiffSimExamples,
    name="diffsim.example_diffsim_bear",
    devices=test_devices,
    test_options={"usd_required": True, "num-frames": 4 * 60},  # train_iters * sim_steps
    test_options_cpu={"num-frames": 2, "sim-steps": 10},
)


class TestSensorExamples(unittest.TestCase):
    pass


add_example_test(
    TestSensorExamples,
    name="sensors.example_sensor_contact",
    devices=test_devices,
    test_options={"num-frames": 160},  # required for ball to reach plate
    expected_output=[
        "zero mass and zero inertia",
        "SensorContact initialized",
        "Sensing objects|Counterpart|total_force|force_matrix",
        "Resetting",
    ],
)

add_example_test(
    TestSensorExamples,
    name="sensors.example_sensor_tiled_camera",
    devices=cuda_test_devices,
    test_options={"num-frames": 4 * 36},  # train_iters * sim_steps
)

add_example_test(
    TestSensorExamples,
    name="sensors.example_sensor_imu",
    devices=test_devices,
    test_options={"num-frames": 200},  # allow cubes to settle
)


class TestMPMExamples(unittest.TestCase):
    pass


add_example_test(
    TestMPMExamples,
    name="mpm.example_mpm_granular",
    devices=cuda_test_devices,
    test_options={"num-frames": 100},
)

add_example_test(
    TestMPMExamples,
    name="mpm.example_mpm_multi_material",
    devices=cuda_test_devices,
    test_options={"num-frames": 10},
)

add_example_test(
    TestMPMExamples,
    name="mpm.example_mpm_grain_rendering",
    devices=cuda_test_devices,
    test_options={"num-frames": 10},
    expected_output=["quadrature.*deprecated|Please use.*instead"],
)

add_example_test(
    TestMPMExamples,
    name="mpm.example_mpm_twoway_coupling",
    devices=cuda_test_devices,
    test_options={"num-frames": 80},
)

add_example_test(
    TestMPMExamples,
    name="mpm.example_mpm_beam_twist",
    devices=cuda_test_devices,
    test_options={"num-frames": 100},
)

add_example_test(
    TestMPMExamples,
    name="mpm.example_mpm_snow_ball",
    devices=cuda_test_devices,
    test_options={"num-frames": 30, "voxel-size": 0.2},
    expected_output=["Generating.*particles"],
)

add_example_test(
    TestMPMExamples,
    name="mpm.example_mpm_viscous",
    devices=cuda_test_devices,
    test_options={"num-frames": 30, "voxel-size": 0.01},
)


add_example_test(
    TestBasicExamples,
    name="basic.example_basic_plotting",
    devices=test_devices,
    test_options={"num-frames": 200},
    expected_output=["Diagnostics plot saved to|Simulation diagnostics summary"],
    allowed_output=[r"^\s+(Iterations|Kinetic E|Potential E|Constraints):"],
)


class TestContactsExamples(unittest.TestCase):
    pass


add_example_test(
    TestContactsExamples,
    name="contacts.example_nut_bolt_sdf",
    devices=cuda_test_devices,
    test_options={"num-frames": 120, "world-count": 1},
    expected_output=["Downloading nut/bolt assets", "Assets downloaded to"],
)
add_example_test(
    TestContactsExamples,
    name="contacts.example_nut_bolt_hydro",
    devices=cuda_test_devices,
    test_options={"num-frames": 120, "world-count": 1},
    expected_output=["Downloading nut/bolt assets", "Assets downloaded to"],
)
add_example_test(
    TestContactsExamples,
    name="contacts.example_brick_stacking",
    devices=cuda_test_devices,
    test_options={"num-frames": 1200},
)
add_example_test(
    TestContactsExamples,
    name="contacts.example_pyramid",
    devices=cuda_test_devices,
    test_options={"num-frames": 120, "num-pyramids": 3, "pyramid-size": 5},
    expected_output=["Built.*pyramids"],
)


class TestMultiphysicsExamples(unittest.TestCase):
    pass


add_example_test(
    TestMultiphysicsExamples,
    name="multiphysics.example_softbody_gift",
    devices=cuda_test_devices,
    test_options={"num-frames": 200},
    expected_output=["Detected non-manifold edge"],
)
add_example_test(
    TestMultiphysicsExamples,
    name="cloth.example_cloth_poker_cards",
    devices=cuda_test_devices,
    test_options={"num-frames": 30},
)
add_example_test(
    TestMultiphysicsExamples,
    name="multiphysics.example_softbody_dropping_to_cloth",
    devices=cuda_test_devices,
    test_options={"num-frames": 200},
)


class TestSoftbodyExamples(unittest.TestCase):
    pass


add_example_test(
    TestSoftbodyExamples,
    name="softbody.example_softbody_hanging",
    devices=cuda_test_devices,
    test_options={"num-frames": 120},
)


if __name__ == "__main__":
    unittest.main(verbosity=2)
