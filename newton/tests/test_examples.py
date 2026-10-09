# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Test examples in the newton.examples package.

Currently, this script mainly checks that the examples can run. When the test
runner is invoked with ``--strict-warnings`` (as CI does), example subprocesses
treat non-allowlisted deprecation warnings as failures so examples do not regress
onto deprecated APIs; otherwise deprecations are non-fatal. (The broader newton.*
escalation of ``--strict-warnings`` applies to the in-process tests, not example
subprocesses.)

The test parameters are typically tuned so that each test can run in 10 seconds
or less, ignoring module compilation time. A notable exception is the robot
manipulating cloth example, which takes approximately 35 seconds to run on a
CUDA device.
"""

import contextlib
import importlib.util
import io
import json
import os
import re
import shlex
import subprocess
import sys
import tempfile
import unittest
from types import SimpleNamespace
from typing import Any
from unittest import mock
from unittest.mock import call, create_autospec, patch

import numpy as np
import warp as wp

import newton.examples
import newton.tests.unittest_utils
from newton.examples.robot.example_robot_cartpole import Example as RobotCartpoleExample
from newton.tests.unittest_utils import (
    USD_AVAILABLE,
    NewtonTestCase,
    add_function_test,
    get_selected_cuda_test_devices,
    get_test_devices,
    sanitize_identifier,
)
from newton.viewer import ViewerNull

_HAS_ONNX_RUNTIME = importlib.util.find_spec("onnx") is not None and importlib.util.find_spec("warp_nn") is not None
_PXR_WORK_THREAD_LIMIT_OUTPUT_RE = (
    r"(?s)#+\n#  PXR_WORK_THREAD_LIMIT is overridden to '1'\.  Default is '0'\.  #\n#+\n?"
)
_WARP_CUDA_UNAVAILABLE_OUTPUT_RE = (
    r"(?:"
    r"Warp CUDA warning: Could not find or load the NVIDIA CUDA driver\. "
    r"GPU execution will not be available\."
    r"|"
    r"Warp CUDA error \d+(?:: [^\n]*)? "
    r"\(in function init_cuda_driver, [^\n]*cuda_util\.cpp:\d+\)"
    r")\n?"
)
_ASSET_CACHE_REFRESH_OUTPUT_RE = (
    r"(?:New version of [^\n]+ found "
    r"\(cached: [0-9a-f]{8}, latest: [0-9a-f]{8}\)\. Refreshing\.\.\.\n)?"
)
_NEWTON_ASSET_DOWNLOAD_OUTPUT_RE = (
    _ASSET_CACHE_REFRESH_OUTPUT_RE + r"Cloning https://github\.com/newton-physics/newton-assets\.git "
    r"\(ref: [0-9a-f]{40}\)\.\.\.\n"
    r"Successfully downloaded folder to: [^\n]+\n?"
)
_ISAACGYM_ASSET_DOWNLOAD_OUTPUT_RE = (
    _ASSET_CACHE_REFRESH_OUTPUT_RE + r"Cloning https://github\.com/isaac-sim/IsaacGymEnvs\.git "
    r"\(ref: main\)\.\.\.\n"
    r"Successfully downloaded folder to: [^\n]+\n?"
)
_NUT_BOLT_DOWNLOAD_START_OUTPUT_RE = r"Downloading nut/bolt assets\.\.\.\n?"
_NUT_BOLT_DOWNLOAD_DONE_OUTPUT_RE = r"Assets downloaded to: [^\n]+\n?"
_PYRAMID_BUILD_OUTPUT_RE = r"Built 3 pyramids x 5 rows = 45 boxes\n?"
_MATPLOTLIB_FONT_CACHE_OUTPUT_RE = r"Matplotlib is building the font cache; this may take a moment\.\n?"
_DIFFSIM_BALL_GRADIENT_OUTPUT_RE = r"(?:numeric grad: \[[^\n]+\]\nanalytic grad: \[[^\n]+\]\n?){2}"
_DIFFSIM_DRONE_LOSS_LINE_RE = r"\[\s*\d{1,3}/360\] loss=-?\d+\.\d{8}\n?"
_DIFFSIM_DRONE_LOSS_OUTPUT_RE = rf"(?:{_DIFFSIM_DRONE_LOSS_LINE_RE}){{10}}"
_BASIC_PLOTTING_OUTPUT_RE = (
    r"(?:"
    r"Diagnostics plot saved to solver_convergence\.png\n?"
    r"|"
    r"\n?Simulation diagnostics summary \(\d+ steps\):\n"
    r"  Iterations \(max\):   mean=[^\n]*\n"
    r"  Kinetic E \[J\]:    final=[^\n]*\n"
    r"  Potential E \[J\]:  final=[^\n]*\n"
    r"  Constraints:        mean=[^\n]*\n?"
    r")"
)
_WARP_SDF_CONSTANT_CONVERSION_WARNING_RE = (
    r"(?m)"
    r"(?:^.*wp_sdf_contact_write_contact_to_reducer_[^\n]*\.cpp:\d+:\d+: warning: "
    r"implicit conversion from 'long' to 'const wp::int32'.*\n"
    r"^.*\n"
    r"^.*\n"
    r")+"
    r"^\d+ warnings? generated\.\n?"
)
_KAMINO_NON_FLOATING_ROOT_WARNING_RE = (
    r"(?m)^.*Model has articulations whose root is not a free joint attached to the world, "
    r"disabling floating base resets for those worlds\..*\n?"
)
_ANYMAL_TEXTURE_WITHOUT_UVS_WARNING_RE = (
    r"^.*newton[/\\]_src[/\\]utils[/\\]import_urdf\.py:\d+: UserWarning: Warning: mesh "
    r"[^\n]*[/\\]base\.dae has a texture but no UVs; texture will be ignored\.\n"
    r"  parse_shapes\(link, visuals, density=0\.0, just_visual=True, visible=not hide_visuals\)\n?"
)
_EXAMPLE_ALLOW_OUTPUT_REGEXES = [
    (_PXR_WORK_THREAD_LIMIT_OUTPUT_RE, "stderr"),
    (_NEWTON_ASSET_DOWNLOAD_OUTPUT_RE, "stdout"),
]
_OutputRegexSpec = str | tuple[str, str]
_registered_examples: set[str] = set()


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


def _run_example_subprocess(module, argv, timeout, *, allow_deprecation_warnings=False):
    env = os.environ.copy()
    env.pop("PYTHONWARNINGS", None)
    if wp.config.kernel_cache_dir is not None:
        env["WARP_CACHE_PATH"] = os.path.dirname(wp.config.kernel_cache_dir)
    strict_warnings = newton.tests.unittest_utils.strict_warnings and not allow_deprecation_warnings
    warning_args = newton.tests.unittest_utils.get_strict_warning_args() if strict_warnings else []
    command = [sys.executable, *warning_args]
    if newton.tests.unittest_utils.coverage_enabled:
        with tempfile.NamedTemporaryFile(dir=newton.tests.unittest_utils.coverage_temp_dir, delete=False) as coverage:
            pass
        command.extend(["-m", "coverage", "run", f"--data-file={coverage.name}"])
        if newton.tests.unittest_utils.coverage_branch:
            command.append("--branch")
    command.extend(["-m", module, *argv])
    return subprocess.run(command, capture_output=True, text=True, env=env, timeout=timeout, check=False)


def add_example_test(
    cls: type,
    name: str,
    devices: list | None = None,
    test_options: dict[str, Any] | None = None,
    test_options_cpu: dict[str, Any] | None = None,
    test_options_cuda: dict[str, Any] | None = None,
    use_viewer: bool = False,
    test_suffix: str | None = None,
    expect_output_regexes: list[_OutputRegexSpec] | None = None,
    allow_output_regexes: list[_OutputRegexSpec] | None = None,
    *,
    variants: list[dict[str, Any]] | None = None,
):
    """Register an example, optionally sharing a subprocess across argument variants."""
    if variants is None:
        variants = [
            {
                "test_options": test_options or {},
                "test_options_cpu": test_options_cpu or {},
                "test_options_cuda": test_options_cuda or {},
                "expect_output_regexes": expect_output_regexes,
                "allow_output_regexes": allow_output_regexes,
                "test_suffix": test_suffix,
            }
        ]
    if not issubclass(cls, NewtonTestCase) and any(
        variant.get(key) is not None
        for variant in variants
        for key in ("expect_output_regexes", "allow_output_regexes")
    ):
        raise TypeError("Output regex expectations require a NewtonTestCase subclass")
    examples_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "examples")
    if not os.path.exists(os.path.join(examples_dir, f"{name.replace('.', '/')}.py")):
        raise ValueError(f"Example {name} does not exist")
    _registered_examples.add(name)

    def run(test, device):
        is_cuda = wp.get_device(device).is_cuda
        cases = []
        for variant in variants:
            with test.subTest(variant=variant.get("test_suffix")):
                options = _merge_options(
                    variant.get("test_options", {}),
                    variant.get("test_options_cuda" if is_cuda else "test_options_cpu", {}),
                )
                onnx_required = options.pop("onnx_required", False)
                torch_required = options.pop("torch_required", False)
                if (onnx_required or torch_required) and not _HAS_ONNX_RUNTIME:
                    test.skipTest("onnx or warp-nn not installed")
                if options.pop("usd_required", False) and not USD_AVAILABLE:
                    test.skipTest("Requires usd-core")
                timeout = options.pop("test_timeout", 600)
                allow_deprecations = options.pop("allow_deprecation_warnings", False)
                argv = ["--device", str(device), "--test", "--quiet"]
                for entry in newton.tests.unittest_utils.warp_config_overrides:
                    argv.extend(["--warp-config", entry])
                stage_path = None
                if use_viewer:
                    argv.extend(["--viewer", "null"])
                    options.pop("viewer", None)
                    options.pop("stage_path", None)
                else:
                    stage_path = (
                        options.pop(
                            "stage_path",
                            os.path.join(
                                os.path.dirname(__file__), f"outputs/{name}_{sanitize_identifier(device)}.usd"
                            ),
                        )
                        if USD_AVAILABLE
                        else "None"
                    )
                    if stage_path:
                        argv.extend(["--stage-path", stage_path])
                        with contextlib.suppress(OSError):
                            os.remove(stage_path)
                argv.extend(_build_command_line_options(options))
                cases.append(
                    {
                        "variant": variant,
                        "argv": argv,
                        "stage_path": stage_path,
                        "timeout": timeout,
                        "allow_deprecation_warnings": allow_deprecations,
                    }
                )
        if not cases:
            return

        module = f"newton.examples.{name}"
        if len(cases) == 1:
            case = cases[0]
            results = [
                _run_example_subprocess(
                    module,
                    case["argv"],
                    case["timeout"],
                    allow_deprecation_warnings=case["allow_deprecation_warnings"],
                )
            ]
        else:
            with tempfile.TemporaryDirectory() as directory:
                manifest = os.path.join(directory, "variants.json")
                output = os.path.join(directory, "results.json")
                with open(manifest, "w", encoding="utf-8") as stream:
                    json.dump(
                        {
                            "module": module,
                            "cases": [
                                {"argv": case["argv"], "allow_deprecation_warnings": case["allow_deprecation_warnings"]}
                                for case in cases
                            ],
                        },
                        stream,
                    )
                batch = _run_example_subprocess(
                    "newton.tests.example_batch", [manifest, output], sum(case["timeout"] for case in cases)
                )
                _check_example_result(test, batch, is_cuda=is_cuda)
                with open(output, encoding="utf-8") as stream:
                    results = [
                        subprocess.CompletedProcess(args=[sys.executable, "-m", module, *case["argv"]], **result)
                        for case, result in zip(cases, json.load(stream), strict=True)
                    ]

        for case, result in zip(cases, results, strict=True):
            with test.subTest(variant=case["variant"].get("test_suffix")):
                _check_example_result(test, result, is_cuda=is_cuda, variant=case["variant"])
                if case["stage_path"] and case["stage_path"] != "None":
                    with contextlib.suppress(OSError):
                        os.remove(case["stage_path"])

    test_name = f"test_{name}_{test_suffix}" if test_suffix else f"test_{name}"
    add_function_test(cls, test_name, run, devices=devices, check_output=False)


def _check_example_result(test, result, *, is_cuda, variant=None):
    test.assertEqual(
        result.returncode,
        0,
        f"Example failed. Reproduce with:\n{shlex.join(result.args)}\n\n{result.stdout}\n{result.stderr}",
    )
    if not isinstance(test, NewtonTestCase):
        if result.stderr:
            print(result.stderr)
        return
    variant = variant or {}
    output = newton.tests.unittest_utils._OutputCapture()
    _register_output_regexes(output, variant.get("expect_output_regexes"), required=True)
    _register_example_allow_output_regexes(output, is_cuda=is_cuda)
    _register_output_regexes(output, variant.get("allow_output_regexes"), required=False)
    output.record("stdout", result.stdout)
    output.record("stderr", result.stderr)
    failure = output._check_output()
    if failure:
        test.fail(failure)


def _register_output_regexes(output, regexes: list[_OutputRegexSpec] | None, *, required: bool):
    for regex_spec in regexes or ():
        regex, stream = regex_spec if isinstance(regex_spec, tuple) else (regex_spec, "any")
        output.add_pattern(regex, stream=stream, required=required)


def _register_example_allow_output_regexes(output, *, is_cuda: bool) -> None:
    _register_output_regexes(output, _EXAMPLE_ALLOW_OUTPUT_REGEXES, required=False)
    if not is_cuda:
        output.add_pattern(_WARP_CUDA_UNAVAILABLE_OUTPUT_RE, stream="stderr", required=False)


def add_example_batch(cls: type, name: str, variants: list[dict[str, Any]]):
    """Share a subprocess across an example's variants on each device."""
    devices = {str(device): device for variant in variants for device in variant["devices"]}
    if not devices:
        add_example_test(cls, name, devices=[], use_viewer=True, test_suffix="batch", variants=variants)
    for device in devices.values():
        selected = [variant for variant in variants if str(device) in map(str, variant["devices"])]
        suffix = selected[0].get("test_suffix") if len(selected) == 1 else "batch"
        add_example_test(cls, name, devices=[device], use_viewer=True, test_suffix=suffix, variants=selected)


class TestExampleOutputRegexes(unittest.TestCase):
    def _run_example_with_stderr(self, stderr: str, allowed_prefix: str) -> unittest.TestResult:
        process_result = subprocess.CompletedProcess(
            args=[sys.executable, "-m", "newton.examples.basic.example_basic_pendulum"],
            returncode=0,
            stdout="",
            stderr=stderr,
        )

        class ExampleWithAllowedDeprecation(NewtonTestCase):
            pass

        add_example_test(
            ExampleWithAllowedDeprecation,
            name="basic.example_basic_pendulum",
            use_viewer=True,
        )

        with (
            mock.patch.object(subprocess, "run", return_value=process_result),
            mock.patch.object(newton.tests.unittest_utils, "strict_warnings", True),
            mock.patch.object(
                newton.tests.unittest_utils,
                "allowed_deprecation_warnings",
                (allowed_prefix,),
            ),
        ):
            result = unittest.TestResult()
            unittest.defaultTestLoader.loadTestsFromTestCase(ExampleWithAllowedDeprecation).run(result)

        return result

    def test_allowlisted_deprecation_from_example_subprocess_is_allowed(self):
        """Allow an acknowledged deprecation emitted by an example subprocess."""
        allowed_prefix = "dependency.old_api is deprecated"
        allowed_message = f"{allowed_prefix}; use dependency.new_api instead"
        stderr = (
            f"{__file__}:1: DeprecationWarning: {allowed_message}\n"
            "  # SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers\n"
        )
        output = io.StringIO()

        with contextlib.redirect_stderr(output):
            result = self._run_example_with_stderr(stderr, allowed_prefix)

        self.assertTrue(result.wasSuccessful(), result.failures)
        self.assertEqual(output.getvalue(), stderr)

    def test_allowlisted_warp_deprecation_from_example_subprocess_is_allowed(self):
        """Allow an acknowledged deprecation emitted in Warp's log format."""
        allowed_prefix = "dependency.old_api is deprecated"
        stderr = f"Warp DeprecationWarning: {allowed_prefix}; use dependency.new_api instead\n"
        output = io.StringIO()

        with contextlib.redirect_stderr(output):
            result = self._run_example_with_stderr(stderr, allowed_prefix)

        self.assertTrue(result.wasSuccessful(), result.failures)
        self.assertEqual(output.getvalue(), stderr)

    def test_allowlisted_deprecation_does_not_hide_other_example_stderr(self):
        """Reject unrelated stderr following an acknowledged deprecation."""
        allowed_prefix = "dependency.old_api is deprecated"
        stderr = (
            f"{__file__}:1: DeprecationWarning: {allowed_prefix}; use dependency.new_api instead\n"
            "  # SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers\n"
            "unexpected stderr\n"
        )

        result = self._run_example_with_stderr(stderr, allowed_prefix)

        self.assertEqual(result.testsRun, 1)
        self.assertEqual(result.errors, [])
        self.assertEqual(len(result.failures), 1)
        self.assertIn("Unexpected stderr:\nunexpected stderr", result.failures[0][1])
        self.assertNotIn(allowed_prefix, result.failures[0][1])

    def test_unlisted_example_deprecations_remain_failures(self):
        """Reject a deprecation record whose message does not match the allowlist."""
        stderr = "<string>:1: DeprecationWarning: unexpected deprecation\n"
        result = self._run_example_with_stderr(stderr, "dependency.old_api is deprecated")

        self.assertEqual(result.errors, [])
        self.assertEqual(len(result.failures), 1)
        self.assertIn(f"Unexpected stderr:\n{stderr.rstrip()}", result.failures[0][1])

    def test_warp_cuda_unavailable_output_is_registered_only_for_cpu(self):
        """Register CUDA driver initialization diagnostics only for CPU examples."""
        cpu_test = create_autospec(newton.tests.unittest_utils._OutputCapture, instance=True)
        cuda_test = create_autospec(newton.tests.unittest_utils._OutputCapture, instance=True)

        _register_example_allow_output_regexes(cpu_test, is_cuda=False)
        _register_example_allow_output_regexes(cuda_test, is_cuda=True)

        warp_cuda_call = call(_WARP_CUDA_UNAVAILABLE_OUTPUT_RE, stream="stderr", required=False)
        self.assertIn(warp_cuda_call, cpu_test.add_pattern.call_args_list)
        self.assertNotIn(warp_cuda_call, cuda_test.add_pattern.call_args_list)

    def test_warp_cuda_unavailable_output_is_allowed(self):
        """Allow CUDA driver initialization diagnostics emitted on CPU-only systems."""
        outputs = (
            "Warp CUDA warning: Could not find or load the NVIDIA CUDA driver. GPU execution will not be available.\n",
            "Warp CUDA error 100: no CUDA-capable device is detected "
            "(in function init_cuda_driver, /builds/omniverse/warp/warp/native/cuda_util.cpp:319)\n",
            "Warp CUDA error 999: unknown error "
            "(in function init_cuda_driver, /builds/omniverse/warp/warp/native/cuda_util.cpp:333)\n",
        )

        for output in outputs:
            with self.subTest(output=output):
                unmatched_output = re.sub(_WARP_CUDA_UNAVAILABLE_OUTPUT_RE, "", output, flags=re.MULTILINE)
                self.assertEqual(unmatched_output, "")

    def test_warp_cuda_non_initialization_output_is_not_allowed(self):
        """Keep CUDA diagnostics outside driver initialization visible to tests."""
        output = (
            "Warp CUDA error 999: unknown error "
            "(in function wp_cuda_graphics_register_gl_buffer, /builds/omniverse/warp/warp/native/warp.cu:4419)\n"
        )

        unmatched_output = re.sub(_WARP_CUDA_UNAVAILABLE_OUTPUT_RE, "", output, flags=re.MULTILINE)

        self.assertEqual(unmatched_output, output)

    def test_basic_plotting_output_does_not_consume_trailing_output(self):
        unexpected_output = "unexpected output\n"
        output = (
            "Simulation diagnostics summary (3 steps):\n"
            "  Iterations (max):   mean=1.0, peak=2\n"
            "  Kinetic E [J]:    final=2.0\n"
            "  Potential E [J]:  final=3.0\n"
            "  Constraints:        mean=4.0, peak=5.0\n" + unexpected_output
        )

        unmatched_output = re.sub(_BASIC_PLOTTING_OUTPUT_RE, "", output, flags=re.MULTILINE)

        self.assertEqual(unmatched_output, unexpected_output)


cuda_test_devices = get_selected_cuda_test_devices(mode="basic")  # Don't test on multiple GPUs to save time
test_devices = get_test_devices(mode="basic")


class TestBasicExamples(NewtonTestCase):
    pass


def add_basic_example_test(**kwargs):
    add_example_test(TestBasicExamples, **kwargs)


add_basic_example_test(name="basic.example_basic_pendulum", devices=test_devices, use_viewer=True)
add_basic_example_test(
    name="basic.example_basic_pendulum",
    devices=cuda_test_devices,
    test_options={"solver": "kamino"},
    use_viewer=True,
    test_suffix="kamino",
    allow_output_regexes=[(_KAMINO_NON_FLOATING_ROOT_WARNING_RE, "stderr")],
)

add_basic_example_test(
    name="basic.example_recording",
    devices=test_devices,
    use_viewer=True,
    test_options={"num-frames": 120, "world-count": 8},
)

add_example_batch(
    TestBasicExamples,
    name="basic.example_basic_urdf",
    variants=[
        {
            "devices": test_devices,
            "test_options": {"num-frames": 200},
            "test_options_cpu": {"world_count": 16},
            "test_options_cuda": {"world_count": 64},
            "test_suffix": "xpbd",
        },
        {
            "devices": test_devices,
            "test_options": {"num-frames": 200, "solver": "vbd"},
            "test_options_cpu": {"world_count": 16},
            "test_options_cuda": {"world_count": 64},
            "test_suffix": "vbd",
        },
        {
            "devices": cuda_test_devices,
            "test_options": {"num-frames": 200, "solver": "kamino", "world-count": 4},
            "test_suffix": "kamino",
        },
    ],
)

add_basic_example_test(name="basic.example_basic_viewer", devices=test_devices, use_viewer=True)

add_example_batch(
    TestBasicExamples,
    name="basic.example_basic_joints",
    variants=[
        {"devices": test_devices, "test_suffix": "xpbd"},
        {"devices": test_devices, "test_options": {"solver": "vbd"}, "test_suffix": "vbd"},
        {
            "devices": cuda_test_devices,
            "test_options": {"solver": "kamino"},
            "test_suffix": "kamino",
            "allow_output_regexes": [(_KAMINO_NON_FLOATING_ROOT_WARNING_RE, "stderr")],
        },
    ],
)

add_example_batch(
    TestBasicExamples,
    name="basic.example_basic_mimic_joint",
    variants=[
        {
            "devices": test_devices,
            "test_options": {"num-frames": 120, "solver": solver},
            "test_suffix": solver,
        }
        for solver in ("featherstone", "semi_implicit", "xpbd", "mujoco", "vbd")
    ],
)

add_example_batch(
    TestBasicExamples,
    name="basic.example_basic_shapes",
    variants=[
        {
            "devices": test_devices,
            "test_options": {"num-frames": 150, "solver": "xpbd"},
            "test_suffix": "xpbd",
            "allow_output_regexes": [(_WARP_SDF_CONSTANT_CONVERSION_WARNING_RE, "stderr")],
        },
        {
            "devices": cuda_test_devices,
            "test_options": {"num-frames": 150, "solver": "kamino"},
            "test_suffix": "kamino",
            "allow_output_regexes": [(_WARP_SDF_CONSTANT_CONVERSION_WARNING_RE, "stderr")],
        },
        {
            "devices": test_devices,
            "test_options": {"num-frames": 150, "solver": "vbd"},
            "test_suffix": "vbd",
            "allow_output_regexes": [(_WARP_SDF_CONSTANT_CONVERSION_WARNING_RE, "stderr")],
        },
    ],
)

add_basic_example_test(
    name="basic.example_basic_heightfield",
    devices=cuda_test_devices,
    use_viewer=True,
    test_options={"num-frames": 120, "solver": "kamino"},
    test_suffix="kamino",
)

add_basic_example_test(
    name="basic.example_basic_conveyor",
    devices=test_devices,
    use_viewer=True,
    test_options={"num-frames": 100},
    allow_output_regexes=[(_WARP_SDF_CONSTANT_CONVERSION_WARNING_RE, "stderr")],
)
add_basic_example_test(
    name="basic.example_basic_conveyor",
    devices=cuda_test_devices,
    use_viewer=True,
    test_options={"num-frames": 100, "solver": "kamino"},
    test_suffix="kamino",
    allow_output_regexes=[(_WARP_SDF_CONSTANT_CONVERSION_WARNING_RE, "stderr")],
)
add_example_batch(
    TestBasicExamples,
    name="basic.example_basic_conveyor_forces",
    variants=[
        {
            "devices": test_devices,
            "test_options": {"num-frames": 100, "solver": "xpbd"},
            "test_suffix": "xpbd",
            "allow_output_regexes": [(_WARP_SDF_CONSTANT_CONVERSION_WARNING_RE, "stderr")],
        },
        {
            "devices": cuda_test_devices,
            "test_options": {"num-frames": 100, "solver": "vbd"},
            "test_suffix": "vbd",
            "allow_output_regexes": [(_WARP_SDF_CONSTANT_CONVERSION_WARNING_RE, "stderr")],
        },
        {
            "devices": cuda_test_devices,
            "test_options": {"num-frames": 100, "solver": "mujoco"},
            "test_suffix": "mujoco",
            "allow_output_regexes": [(_WARP_SDF_CONSTANT_CONVERSION_WARNING_RE, "stderr")],
        },
        {
            "devices": cuda_test_devices,
            "test_options": {"num-frames": 100, "solver": "kamino"},
            "test_suffix": "kamino",
            "allow_output_regexes": [(_WARP_SDF_CONSTANT_CONVERSION_WARNING_RE, "stderr")],
        },
    ],
)
add_example_batch(
    TestBasicExamples,
    name="basic.example_basic_dzhanibekov",
    variants=[
        {"devices": test_devices, "test_options": {"num-frames": 230, "solver": "vbd"}, "test_suffix": "vbd"},
        {"devices": test_devices, "test_options": {"num-frames": 230, "solver": "xpbd"}, "test_suffix": "xpbd"},
        {"devices": test_devices, "test_options": {"num-frames": 230, "solver": "mujoco"}, "test_suffix": "mujoco"},
        {
            "devices": cuda_test_devices,
            "test_options": {"num-frames": 230, "solver": "kamino"},
            "test_suffix": "kamino",
        },
    ],
)

add_basic_example_test(
    name="basic.example_basic_multi_solver_overlay",
    devices=test_devices,
    use_viewer=True,
    test_options={"num-frames": 50},
    allow_output_regexes=[(_KAMINO_NON_FLOATING_ROOT_WARNING_RE, "stderr")],
)


class TestCableExamples(NewtonTestCase):
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
)
add_example_batch(
    TestCableExamples,
    name="cable.example_cable_bundle_hysteresis",
    variants=[
        {"devices": test_devices, "test_options": {"num-frames": 20}},
        {
            "devices": test_devices,
            "test_options": {"num-frames": 150, "eps-max": 2.0, "tau": 0.1},
            "test_suffix": "dahl_retention",
        },
        {
            "devices": test_devices,
            "test_options": {"num-frames": 150, "no-dahl": True},
            "test_suffix": "no_dahl_recovery",
        },
    ],
)
add_example_test(
    TestCableExamples,
    name="cable.example_cable_cross_slide_table",
    devices=test_devices,
    use_viewer=True,
    test_options={"num-frames": 540},
)
add_example_test(
    TestCableExamples,
    name="cable.example_cable_pile",
    devices=test_devices,
    use_viewer=True,
    test_options={"num-frames": 20},
)
add_example_test(
    TestCableExamples,
    name="cable.example_cable_plectoneme",
    devices=cuda_test_devices,
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
add_example_batch(
    TestClothExamples,
    name="cloth.example_cloth_hanging",
    variants=[
        {
            "devices": test_devices,
            "test_options": {},
            "test_options_cpu": {"width": 32, "height": 16, "num-frames": 10},
            "test_suffix": "vbd",
        },
        {
            "devices": test_devices,
            "test_options": {"solver": "style3d"},
            "test_options_cpu": {"width": 32, "height": 16, "num-frames": 10},
            "test_suffix": "style3d",
        },
    ],
)
add_example_test(
    TestClothExamples,
    name="cloth.example_cloth_style3d",
    devices=cuda_test_devices,
    test_options={},
    test_options_cuda={"num-frames": 32},
    use_viewer=True,
)
add_example_test(
    TestClothExamples,
    name="cloth.example_cloth_h1",
    devices=cuda_test_devices,
    test_options={},
    test_options_cuda={"num-frames": 32},
    use_viewer=True,
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
add_example_test(
    TestClothExamples,
    name="vbd.example_cloth_stiff_material_hanging",
    devices=cuda_test_devices,
    test_options={"usd_required": True, "num-frames": 360},
    use_viewer=True,
)
add_example_test(
    TestClothExamples,
    name="vbd.example_cloth_stiff_material_stretch",
    devices=cuda_test_devices,
    test_options={"num-frames": 360},
    use_viewer=True,
)
add_example_test(
    TestClothExamples,
    name="vbd.example_vbd_gripper_soft_triangle",
    devices=cuda_test_devices,
    test_options={"num-frames": 360},
    use_viewer=True,
)
add_example_test(
    TestClothExamples,
    name="vbd.example_vbd_gripper_soft_grid",
    devices=cuda_test_devices,
    test_options={"num-frames": 360},
    use_viewer=True,
)


class TestRobotExamples(unittest.TestCase):
    pass


def _test_robot_cartpole_capture_matches_eager(test, device):
    if not USD_AVAILABLE:
        test.skipTest("Requires USD")

    args = SimpleNamespace(world_count=1, solver="kamino")
    with wp.ScopedDevice(device):
        with patch.object(RobotCartpoleExample, "capture", lambda example: setattr(example, "graph", None)):
            eager = RobotCartpoleExample(ViewerNull(num_frames=2), args)
        eager_frames = []
        for _ in range(2):
            eager.step()
            eager_frames.append((eager.state_0.body_q.numpy(), eager.state_0.body_qd.numpy()))

        captured = RobotCartpoleExample(ViewerNull(num_frames=2), args)
        for frame, (eager_q, eager_qd) in enumerate(eager_frames):
            captured.step()
            wp.synchronize_device(device)
            np.testing.assert_allclose(
                captured.state_0.body_q.numpy(),
                eager_q,
                rtol=1.0e-5,
                atol=1.0e-6,
                err_msg=f"body positions differ after frame {frame}",
            )
            np.testing.assert_allclose(
                captured.state_0.body_qd.numpy(),
                eager_qd,
                rtol=1.0e-5,
                atol=1.0e-6,
                err_msg=f"body velocities differ after frame {frame}",
            )


add_function_test(
    TestRobotExamples,
    "test_robot_cartpole_capture_matches_eager",
    _test_robot_cartpole_capture_matches_eager,
    devices=cuda_test_devices,
    check_output=False,
)


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
    test_options={"usd_required": True, "num-frames": 500, "onnx_required": True},
    use_viewer=True,
)
add_example_test(
    TestRobotExamples,
    name="robot.example_robot_anymal_d",
    devices=test_devices,
    test_options={"usd_required": True, "num-frames": 500},
    test_options_cpu={"num-frames": 10},
    use_viewer=True,
)
add_example_test(
    TestRobotExamples,
    name="robot.example_robot_anymal_d",
    devices=cuda_test_devices,
    test_options={"usd_required": True, "num-frames": 500, "world-count": 1, "solver": "kamino"},
    use_viewer=True,
    test_suffix="kamino",
)
add_example_test(
    TestRobotExamples,
    name="robot.example_robot_g1",
    devices=cuda_test_devices,
    test_options={"usd_required": True, "num-frames": 500},
    use_viewer=True,
)
add_example_test(
    TestRobotExamples,
    name="robot.example_robot_g1",
    devices=cuda_test_devices,
    test_options={"usd_required": True, "num-frames": 500, "world-count": 4, "solver": "kamino"},
    use_viewer=True,
    test_suffix="kamino",
)
add_example_test(
    TestRobotExamples,
    name="robot.example_robot_h1",
    devices=cuda_test_devices,
    test_options={"usd_required": True, "num-frames": 500},
    use_viewer=True,
)
add_example_test(
    TestRobotExamples,
    name="robot.example_robot_h1",
    devices=cuda_test_devices,
    test_options={"usd_required": True, "num-frames": 500, "world-count": 4, "solver": "kamino"},
    use_viewer=True,
    test_suffix="kamino",
)
add_example_test(
    TestRobotExamples,
    name="robot.example_robot_omniwheel",
    devices=cuda_test_devices,
    test_options={"num-frames": 500},
    use_viewer=True,
)
add_example_test(
    TestRobotExamples,
    name="robot.example_robot_asroballet",
    devices=cuda_test_devices,
    test_options={"num-frames": 500, "onnx_required": True},
    use_viewer=True,
)
add_example_test(
    TestRobotExamples,
    name="robot.example_robot_asroballet",
    devices=cuda_test_devices,
    test_options={"controller": "lqr", "num-frames": 500},
    use_viewer=True,
    test_suffix="LQR",
)
add_example_test(
    TestRobotExamples,
    name="robot.example_robot_ur10",
    devices=test_devices,
    test_options={"usd_required": True, "num-frames": 500},
    test_options_cpu={"num-frames": 10},
    use_viewer=True,
)
add_example_test(
    TestRobotExamples,
    name="robot.example_robot_ur10",
    devices=cuda_test_devices,
    test_options={"usd_required": True, "num-frames": 100, "world-count": 2, "solver": "kamino"},
    use_viewer=True,
    test_suffix="kamino",
)
add_example_test(
    TestRobotExamples,
    name="robot.example_robot_allegro_hand",
    devices=cuda_test_devices,
    test_options={"usd_required": True, "num-frames": 500},
    use_viewer=True,
)
add_example_test(
    TestRobotExamples,
    name="robot.example_robot_allegro_hand",
    devices=cuda_test_devices,
    test_options={"usd_required": True, "num-frames": 500, "world-count": 1, "solver": "kamino"},
    use_viewer=True,
    test_suffix="kamino",
)
add_example_test(
    TestRobotExamples,
    name="robot.example_robot_panda_hydro",
    devices=cuda_test_devices,
    test_options={"usd_required": True, "num-frames": 720},
    use_viewer=True,
)


class TestRobotPolicyExamples(unittest.TestCase):
    pass


add_example_batch(
    TestRobotPolicyExamples,
    name="robot.example_robot_policy",
    variants=[
        {
            "devices": cuda_test_devices,
            "test_options": {"num-frames": 500, "onnx_required": True, "robot": "g1_29dof"},
            "test_options_cpu": {"num-frames": 10},
            "test_suffix": "G1_29dof",
        },
        {
            "devices": cuda_test_devices,
            "test_options": {"num-frames": 500, "onnx_required": True, "robot": "g1_23dof"},
            "test_suffix": "G1_23dof",
        },
        {
            "devices": cuda_test_devices,
            "test_options": {"num-frames": 500, "onnx_required": True, "robot": "g1_23dof", "physx": True},
            "test_suffix": "G1_23dof_Physx",
        },
        {
            "devices": cuda_test_devices,
            "test_options": {"num-frames": 500, "onnx_required": True, "robot": "anymal"},
            "test_suffix": "Anymal",
        },
        {
            "devices": cuda_test_devices,
            "test_options": {"num-frames": 500, "onnx_required": True, "robot": "anymal", "physx": True},
            "test_suffix": "Anymal_Physx",
        },
        {
            "devices": cuda_test_devices,
            "test_options": {"onnx_required": True},
            "test_options_cuda": {"num-frames": 500, "robot": "go2"},
            "test_suffix": "Go2",
        },
        {
            "devices": cuda_test_devices,
            "test_options": {"onnx_required": True},
            "test_options_cuda": {"num-frames": 500, "robot": "go2", "physx": True},
            "test_suffix": "Go2_Physx",
        },
    ],
)


class TestAdvancedRobotExamples(NewtonTestCase):
    pass


add_example_test(
    TestAdvancedRobotExamples,
    name="mpm.example_mpm_anymal",
    devices=cuda_test_devices,
    test_options={"num-frames": 100, "onnx_required": True},
    allow_output_regexes=[(_ANYMAL_TEXTURE_WITHOUT_UVS_WARNING_RE, "stderr")],
    use_viewer=True,
)


class TestIKExamples(unittest.TestCase):
    pass


add_example_test(TestIKExamples, name="ik.example_ik_franka", devices=test_devices, use_viewer=True)

add_example_test(TestIKExamples, name="ik.example_ik_h1", devices=test_devices, use_viewer=True)

add_example_test(TestIKExamples, name="ik.example_ik_custom", devices=cuda_test_devices, use_viewer=True)

add_example_test(
    TestIKExamples,
    name="ik.example_ik_cube_stacking",
    test_options_cuda={"world-count": 16, "num-frames": 2000},
    devices=cuda_test_devices,
    use_viewer=True,
)


class TestMuJoCoExamples(unittest.TestCase):
    pass


add_example_test(
    TestMuJoCoExamples,
    name="mujoco.example_mujoco_sleeping",
    devices=cuda_test_devices,
    test_options={"stack-count": 2, "num-frames": 300},
    use_viewer=True,
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
)
add_example_test(
    TestSelectionAPIExamples,
    name="selection.example_selection_articulations",
    devices=cuda_test_devices,
    test_options={"num-frames": 100, "world-count": 2, "solver": "kamino"},
    use_viewer=True,
    test_suffix="kamino",
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
    TestRobotExamples,
    name="robot.example_robot_cartpole",
    devices=cuda_test_devices,
    test_options={"usd_required": True, "num-frames": 100, "world-count": 2, "solver": "kamino"},
    use_viewer=True,
    test_suffix="kamino",
)
add_example_test(
    TestSelectionAPIExamples,
    name="selection.example_selection_cartpole",
    devices=cuda_test_devices,
    test_options={"num-frames": 100, "world-count": 2, "solver": "kamino"},
    use_viewer=True,
    test_suffix="kamino",
)
add_example_test(
    TestSelectionAPIExamples,
    name="selection.example_selection_materials",
    devices=test_devices,
    test_options={"num-frames": 100},
    test_options_cpu={"num-frames": 10},
    use_viewer=True,
)
add_example_test(
    TestSelectionAPIExamples,
    name="selection.example_selection_materials",
    devices=cuda_test_devices,
    test_options={"num-frames": 100, "world-count": 2, "solver": "kamino"},
    use_viewer=True,
    test_suffix="kamino",
)
add_example_test(
    TestSelectionAPIExamples,
    name="selection.example_selection_multiple",
    devices=test_devices,
    test_options={"num-frames": 100},
    test_options_cpu={"num-frames": 10},
    use_viewer=True,
)
add_example_test(
    TestSelectionAPIExamples,
    name="selection.example_selection_multiple",
    devices=cuda_test_devices,
    test_options={"num-frames": 100, "world-count": 2, "solver": "kamino"},
    use_viewer=True,
    test_suffix="kamino",
)


class TestDiffSimExamples(NewtonTestCase):
    pass


def add_diffsim_example_test(**kwargs: Any) -> None:
    extra_allow_output_regexes = kwargs.pop("allow_output_regexes", None) or ()
    allow_output_regexes = [
        (_PXR_WORK_THREAD_LIMIT_OUTPUT_RE, "stderr"),
        *extra_allow_output_regexes,
    ]
    add_example_test(TestDiffSimExamples, allow_output_regexes=allow_output_regexes, **kwargs)


add_diffsim_example_test(
    name="diffsim.example_diffsim_ball",
    devices=test_devices,
    test_options={"num-frames": 4 * 36},  # train_iters * sim_steps
    test_options_cpu={"num-frames": 2 * 36},
    use_viewer=True,
    expect_output_regexes=[(_DIFFSIM_BALL_GRADIENT_OUTPUT_RE, "stdout")],
)

add_diffsim_example_test(
    name="diffsim.example_diffsim_cloth",
    devices=test_devices,
    test_options={"num-frames": 4 * 120},  # train_iters * sim_steps
    test_options_cpu={"num-frames": 2 * 120},
    use_viewer=True,
)

add_diffsim_example_test(
    name="diffsim.example_diffsim_drone",
    devices=test_devices,
    test_options={"num-frames": 180},  # sim_steps
    test_options_cpu={"num-frames": 10},
    use_viewer=True,
    expect_output_regexes=[(_DIFFSIM_DRONE_LOSS_OUTPUT_RE, "stdout")],
)

add_diffsim_example_test(
    name="diffsim.example_diffsim_spring_cage",
    devices=test_devices,
    test_options={"num-frames": 4 * 30},  # train_iters * sim_steps
    test_options_cpu={"num-frames": 2 * 30},
    use_viewer=True,
)

add_diffsim_example_test(
    name="diffsim.example_diffsim_soft_body",
    devices=test_devices,
    test_options={"num-frames": 4 * 60},  # train_iters * sim_steps
    test_options_cpu={"num-frames": 2 * 60},
    use_viewer=True,
)

add_diffsim_example_test(
    name="diffsim.example_diffsim_bear",
    devices=test_devices,
    test_options={"usd_required": True, "num-frames": 4 * 120, "sim-steps": 120},  # train_iters * sim_steps
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
)
add_example_test(
    TestSensorExamples,
    name="sensors.example_sensor_contact",
    devices=cuda_test_devices,
    test_options={"num-frames": 160, "solver": "kamino"},
    use_viewer=True,
    test_suffix="kamino",
)

add_example_test(
    TestSensorExamples,
    name="sensors.example_sensor_camera",
    devices=cuda_test_devices,
    test_options={"num-frames": 4 * 36},  # train_iters * sim_steps
    use_viewer=True,
)

add_example_test(
    TestSensorExamples,
    name="sensors.example_sensor_imu",
    devices=test_devices,
    test_options={"num-frames": 200},  # allow cubes to settle
    use_viewer=True,
)
add_example_test(
    TestSensorExamples,
    name="sensors.example_sensor_imu",
    devices=cuda_test_devices,
    test_options={"num-frames": 200, "solver": "kamino"},
    use_viewer=True,
    test_suffix="kamino",
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
    name="mpm.example_mpm_granular",
    devices=cuda_test_devices,
    test_options={"usd_required": True, "num-frames": 5, "from_usd": True},
    use_viewer=True,
    test_suffix="authored_usd",
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
    name="mpm.example_mpm_water_dam_break",
    devices=cuda_test_devices,
    test_options={
        "num-frames": 10,
        "voxel-size": 0.15,
        "surface-voxel-size": 0.075,
        "surface-max-grid-cells": 300_000,
        "particles-per-cell": 1,
        "world-count": 2,
    },
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


add_basic_example_test(
    name="basic.example_basic_plotting",
    devices=test_devices,
    test_options={"num-frames": 200},
    use_viewer=True,
    expect_output_regexes=[(_BASIC_PLOTTING_OUTPUT_RE, "stdout")],
    allow_output_regexes=[(_MATPLOTLIB_FONT_CACHE_OUTPUT_RE, "stderr")],
)


class TestContactsExamples(NewtonTestCase):
    pass


def test_pyramid_kamino_impact(test, device):
    """Check Kamino pyramid ground clearance through the wrecking-ball impact."""
    from newton.examples.contacts.example_pyramid import CUBE_HALF, Y_STACK, Example  # noqa: PLC0415

    with contextlib.redirect_stdout(io.StringIO()), wp.ScopedDevice(device):
        example = Example(
            ViewerNull(),
            SimpleNamespace(
                test=False,
                world_count=1,
                solver="kamino",
                num_pyramids=1,
                pyramid_size=20,
                broad_phase="sap",
            ),
        )
        for frame in range(451):
            poses = example.state_0.body_q.numpy()[: example.box_count]
            bottom = min(
                pose[2] - CUBE_HALF * np.abs(np.asarray(wp.quat_to_matrix(wp.quat(*pose[3:7]))).reshape(3, 3)[2]).sum()
                for pose in poses
            )
            test.assertGreater(bottom, -0.1, f"Frame {frame}: a cube penetrated the ground by {-bottom:.3f} m")
            if frame < 450:
                example.step()
        ball_y = example.state_0.body_q.numpy()[example.box_count, 1]
        test.assertLess(ball_y, Y_STACK - 5.0, "The wrecking ball did not pass through the pyramid")


add_function_test(
    TestContactsExamples,
    "test_pyramid_kamino_impact",
    test_pyramid_kamino_impact,
    devices=cuda_test_devices,
)


_CONTACT_EXAMPLE_ALLOW_OUTPUT_REGEXES = [
    (_PXR_WORK_THREAD_LIMIT_OUTPUT_RE, "stderr"),
]


def add_contact_example_test(**kwargs: Any) -> None:
    extra_allow_output_regexes = kwargs.pop("allow_output_regexes", None) or ()
    allow_output_regexes = [*_CONTACT_EXAMPLE_ALLOW_OUTPUT_REGEXES, *extra_allow_output_regexes]
    add_example_test(TestContactsExamples, allow_output_regexes=allow_output_regexes, **kwargs)


for example_name in (
    "contacts.example_balance_bird",
    "contacts.example_domino_spiral",
    "contacts.example_newton_cradle",
):
    for solver in ("xpbd", "vbd"):
        add_contact_example_test(
            name=example_name,
            devices=cuda_test_devices,
            test_options={"num-frames": 60, "solver": solver},
            use_viewer=True,
            test_suffix=solver,
        )


add_contact_example_test(
    name="contacts.example_nut_bolt_sdf",
    devices=cuda_test_devices,
    test_options={"num-frames": 120, "world-count": 1},
    use_viewer=True,
    expect_output_regexes=[
        (_NUT_BOLT_DOWNLOAD_START_OUTPUT_RE, "stdout"),
        (_NUT_BOLT_DOWNLOAD_DONE_OUTPUT_RE, "stdout"),
    ],
    allow_output_regexes=[(_ISAACGYM_ASSET_DOWNLOAD_OUTPUT_RE, "stdout")],
)
add_contact_example_test(
    name="contacts.example_nut_bolt_sdf",
    devices=cuda_test_devices,
    test_options={"num-frames": 120, "world-count": 1, "solver": "kamino"},
    use_viewer=True,
    test_suffix="kamino",
    expect_output_regexes=[
        (_NUT_BOLT_DOWNLOAD_START_OUTPUT_RE, "stdout"),
        (_NUT_BOLT_DOWNLOAD_DONE_OUTPUT_RE, "stdout"),
    ],
    allow_output_regexes=[(_ISAACGYM_ASSET_DOWNLOAD_OUTPUT_RE, "stdout")],
)
add_contact_example_test(
    name="contacts.example_nut_bolt_hydro",
    devices=cuda_test_devices,
    test_options={"num-frames": 120, "world-count": 1},
    use_viewer=True,
    expect_output_regexes=[
        (_NUT_BOLT_DOWNLOAD_START_OUTPUT_RE, "stdout"),
        (_NUT_BOLT_DOWNLOAD_DONE_OUTPUT_RE, "stdout"),
    ],
    allow_output_regexes=[(_ISAACGYM_ASSET_DOWNLOAD_OUTPUT_RE, "stdout")],
)
add_contact_example_test(
    name="contacts.example_nut_bolt_hydro",
    devices=cuda_test_devices,
    test_options={"num-frames": 120, "world-count": 1, "solver": "kamino"},
    use_viewer=True,
    test_suffix="kamino",
    expect_output_regexes=[
        (_NUT_BOLT_DOWNLOAD_START_OUTPUT_RE, "stdout"),
        (_NUT_BOLT_DOWNLOAD_DONE_OUTPUT_RE, "stdout"),
    ],
    allow_output_regexes=[(_ISAACGYM_ASSET_DOWNLOAD_OUTPUT_RE, "stdout")],
)
add_contact_example_test(
    name="contacts.example_brick_stacking",
    devices=cuda_test_devices,
    test_options={"num-frames": 1200},
    use_viewer=True,
    allow_output_regexes=[(_NEWTON_ASSET_DOWNLOAD_OUTPUT_RE, "stdout")],
)
add_contact_example_test(
    name="contacts.example_pyramid",
    devices=cuda_test_devices,
    test_options={"num-frames": 120, "num-pyramids": 3, "pyramid-size": 5},
    use_viewer=True,
    expect_output_regexes=[(_PYRAMID_BUILD_OUTPUT_RE, "stdout")],
)
add_contact_example_test(
    name="contacts.example_pyramid",
    devices=cuda_test_devices,
    test_options={"num-frames": 120, "num-pyramids": 3, "pyramid-size": 5, "solver": "kamino"},
    use_viewer=True,
    test_suffix="kamino",
    expect_output_regexes=[(_PYRAMID_BUILD_OUTPUT_RE, "stdout")],
)


class TestMultiphysicsExamples(NewtonTestCase):
    pass


add_example_test(
    TestMultiphysicsExamples,
    name="multiphysics.example_softbody_gift",
    devices=test_devices,
    test_options={"num-frames": 200},
    test_options_cpu={"num-frames": 2},
    use_viewer=True,
)
add_example_test(
    TestMultiphysicsExamples,
    name="cloth.example_cloth_poker_cards",
    devices=test_devices,
    test_options={"num-frames": 30},
    test_options_cpu={"num-frames": 2},
    use_viewer=True,
)
add_example_test(
    TestMultiphysicsExamples,
    name="multiphysics.example_softbody_dropping_to_cloth",
    devices=test_devices,
    test_options={"num-frames": 200},
    test_options_cpu={"num-frames": 2},
    use_viewer=True,
)
add_example_test(
    TestMultiphysicsExamples,
    name="multiphysics.example_softbody_dropping_to_cloth",
    devices=test_devices,
    test_options={"num-frames": 2, "solver": "coupled", "vbd-iterations": 2},
    use_viewer=True,
    test_suffix="coupled",
)
add_example_batch(
    TestMultiphysicsExamples,
    name="multiphysics.example_rigid_soft_contact",
    variants=[
        {
            "devices": test_devices,
            "test_options": {"num-frames": 180, "solver": "xpbd"},
            "test_options_cpu": {"num-frames": 2},
            "test_suffix": "xpbd",
        },
        {
            "devices": test_devices,
            "test_options": {"num-frames": 180, "solver": "semi_implicit"},
            "test_options_cpu": {"num-frames": 2},
            "test_suffix": "semi_implicit",
        },
        {
            "devices": test_devices,
            "test_options": {"num-frames": 180, "solver": "vbd"},
            "test_options_cpu": {"num-frames": 2},
            "test_suffix": "vbd",
        },
        {
            "devices": test_devices,
            "test_options": {"num-frames": 2, "solver": "coupled", "rigid-solver": "mjc", "vbd-iterations": 1},
            "test_suffix": "coupled_mjc",
        },
    ],
)
add_example_test(
    TestMultiphysicsExamples,
    name="multiphysics.example_mujoco_vbd_admm_solver",
    devices=test_devices,
    test_options={"num-frames": 30},
    use_viewer=True,
)
add_example_test(
    TestMultiphysicsExamples,
    name="multiphysics.example_admm_contact_solver",
    devices=test_devices,
    test_options={"num-frames": 120},
    use_viewer=True,
)
add_example_test(
    TestMultiphysicsExamples,
    name="multiphysics.example_kamino_mujoco_admm_solver",
    devices=["cpu"],
    test_options={"num-frames": 30, "world-count": 4},
    use_viewer=True,
)
add_example_test(
    TestMultiphysicsExamples,
    name="multiphysics.example_xpbd_vbd_coupled_solver",
    devices=test_devices,
    test_options={"num-frames": 5, "xpbd-iterations": 4, "vbd-iterations": 2},
    use_viewer=True,
)
add_example_test(
    TestMultiphysicsExamples,
    name="multiphysics.example_mujoco_franka_vbd_cable_admm_solver",
    devices=cuda_test_devices,
    test_options={
        "num-frames": 2,
        "world-count": 1,
        "substeps": 1,
        "admm-iterations": 1,
        "payload-segments": 3,
        "xpbd-iterations": 2,
        "graph-capture": False,
    },
    use_viewer=True,
    allow_output_regexes=[(_WARP_SDF_CONSTANT_CONVERSION_WARNING_RE, "stderr")],
)
add_example_test(
    TestMultiphysicsExamples,
    name="multiphysics.example_mujoco_mpm_coupled_solver",
    devices=cuda_test_devices,
    test_options={"num-frames": 2, "rigid-substeps": 1, "proxy-iterations": 1},
    use_viewer=True,
)
add_example_test(
    TestMultiphysicsExamples,
    name="multiphysics.example_mujoco_vbd_coupled_solver",
    devices=test_devices,
    test_options={"num-frames": 2, "proxy-iterations": 1},
    use_viewer=True,
)
add_example_test(
    TestMultiphysicsExamples,
    name="multiphysics.example_mujoco_xpbd_coupled_solver",
    devices=test_devices,
    test_options={"num-frames": 2, "proxy-iterations": 1},
    use_viewer=True,
)
add_example_test(
    TestMultiphysicsExamples,
    name="multiphysics.example_proxy_joint_gripper",
    devices=test_devices,
    test_options={"num-frames": 120},
    use_viewer=True,
)
add_example_test(
    TestMultiphysicsExamples,
    name="multiphysics.example_vbd_mpm_coupled_solver",
    devices=cuda_test_devices,
    test_options={"num-frames": 2, "proxy-iterations": 1, "vbd-iterations": 2, "mpm-iterations": 1},
    use_viewer=True,
)
add_example_test(
    TestMultiphysicsExamples,
    name="multiphysics.example_xpbd_mpm_coupled_solver",
    devices=cuda_test_devices,
    test_options={
        "num-frames": 2,
        "proxy-iterations": 1,
        "xpbd-iterations": 2,
        "xpbd-dim-x": 2,
        "xpbd-dim-y": 2,
        "xpbd-dim-z": 2,
        "mpm-iterations": 1,
        "grid-padding": 8,
        "substeps": 1,
    },
    use_viewer=True,
)
add_example_test(
    TestMultiphysicsExamples,
    name="multiphysics.example_vbd_dat_rigid_soft",
    devices=cuda_test_devices,
    test_options={"num-frames": 300},
    use_viewer=True,
)


class TestSoftbodyExamples(NewtonTestCase):
    pass


add_example_test(
    TestSoftbodyExamples,
    name="softbody.example_softbody_hanging",
    devices=test_devices,
    test_options={"num-frames": 120},
    test_options_cpu={"num-frames": 2},
    use_viewer=True,
)


class TestKaminoExamples(unittest.TestCase):
    pass


add_example_test(
    TestKaminoExamples,
    name="kamino.example_kamino_basic_fourbar",
    devices=cuda_test_devices,
    test_options={"num-frames": 120},
    use_viewer=True,
)
add_example_test(
    TestKaminoExamples,
    name="kamino.example_kamino_basic_heterogeneous",
    devices=cuda_test_devices,
    test_options={"num-frames": 120},
    use_viewer=True,
)
add_example_test(
    TestKaminoExamples,
    name="kamino.example_kamino_basic_dr_testmech",
    devices=cuda_test_devices,
    test_options={"num-frames": 120},
    use_viewer=True,
)
add_example_test(
    TestKaminoExamples,
    name="kamino.example_kamino_robot_dr_legs",
    devices=cuda_test_devices,
    test_options={"num-frames": 120},
    use_viewer=True,
)
add_example_test(
    TestKaminoExamples,
    name="kamino.example_kamino_robot_anymal_d",
    devices=cuda_test_devices,
    test_options={"num-frames": 500},
    use_viewer=True,
)


class TestControllersExamples(unittest.TestCase):
    pass


add_example_test(
    TestControllersExamples,
    name="controllers.example_controller_joint_impedance_heterogeneous",
    devices=cuda_test_devices,
    test_options={"num-frames": 120},
    use_viewer=True,
)
add_example_test(
    TestControllersExamples,
    name="controllers.example_controller_joint_impedance_heterogeneous",
    devices=cuda_test_devices,
    test_options={"num-frames": 360, "solver": "kamino"},
    use_viewer=True,
    test_suffix="kamino",
)
add_example_test(
    TestControllersExamples,
    name="controllers.example_controller_operational_space_hybrid_force_motion",
    devices=cuda_test_devices,
    test_options={"usd_required": True, "num-frames": 600},
    use_viewer=True,
)
add_example_test(
    TestControllersExamples,
    name="controllers.example_controller_differential_ik",
    devices=cuda_test_devices,
    test_options={"usd_required": True, "num-frames": 100},
    use_viewer=True,
)


class TestUSDDependentExamples(unittest.TestCase):
    pass


add_example_test(
    TestUSDDependentExamples,
    name="softbody.example_softbody_franka",
    devices=cuda_test_devices,
    test_options={"usd_required": True},
    use_viewer=True,
)
for example_name in (
    "contacts.example_contacts_rj45_plug",
    "vbd.example_vbd_rigid_rigid_contact",
    "vbd.example_vbd_soft_rigid_contact",
    "vbd.example_vbd_soft_rigid_mix_contact",
):
    add_example_test(
        TestUSDDependentExamples,
        name=example_name,
        devices=cuda_test_devices,
        test_options={"allow_deprecation_warnings": True, "usd_required": True},
        use_viewer=True,
    )


class TestAutoDiscoveredExamples(unittest.TestCase):
    pass


for example_module in newton.examples.get_examples().values():
    example_name = example_module.removeprefix("newton.examples.")
    if example_name not in _registered_examples:
        add_example_test(
            TestAutoDiscoveredExamples,
            name=example_name,
            devices=cuda_test_devices,
            use_viewer=True,
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
