# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import ctypes
import ctypes.util
import dataclasses
import importlib.util
import io
import os
import re
import shlex
import sys
import tempfile
import time
import unittest
import xml.etree.ElementTree as ET
from typing import Any

import numpy as np
import warp as wp

pxr = importlib.util.find_spec("pxr")
USD_AVAILABLE = pxr is not None

# default test mode (see get_test_devices())
#   "basic" - only run on CPU and first GPU device
#   "unique" - run on CPU and all unique GPU arches
#   "unique_or_2x" - run on CPU and all unique GPU arches. If there is a single GPU arch, add a second GPU if it exists.
#   "all" - run on all devices
test_mode = "unique_or_2x"

coverage_enabled = False
coverage_temp_dir = None
coverage_branch = None

# Set by the test runner from the --strict-warnings flag. When True, the example
# subprocesses spawned by test_examples.py escalate DeprecationWarnings to errors
# (the in-process tests additionally escalate any warning attributed to a newton.*
# module). Off by default so verifying an installation does not fail on warnings
# the user cannot act on.
strict_warnings = False

# Extra --warp-config KEY=VALUE entries forwarded to example subprocesses.
warp_config_overrides: list[str] = []

try:
    if sys.platform == "win32":
        LIBC = ctypes.CDLL("ucrtbase.dll")
    else:
        LIBC = ctypes.CDLL(ctypes.util.find_library("c"))
except OSError:
    print("Failed to load the standard C library")
    LIBC = None


def get_selected_cuda_test_devices(mode: str | None = None):
    """Returns a list of CUDA devices according the selected ``mode`` behavior.

    If ``mode`` is ``None``, the ``global test_mode`` value will be used and
    this list will be a subset of the devices returned from ``get_test_devices()``.

    Args:
        mode: ``"basic"``, returns a list containing up to a single CUDA device.
          ``"unique"``, returns a list containing no more than one device of
          every CUDA architecture on the system.
          ``"unique_or_2x"`` behaves like ``"unique"`` but adds up to one
          additional CUDA device if the system only devices of a single CUDA
          architecture.
    """

    if mode is None:
        mode = test_mode

    if mode == "basic":
        if wp.is_cuda_available():
            return [wp.get_device("cuda:0")]
        else:
            return []

    cuda_devices = wp.get_cuda_devices()
    first_cuda_devices = {}

    for d in cuda_devices:
        if d.arch not in first_cuda_devices:
            first_cuda_devices[d.arch] = d

    selected_cuda_devices = list(first_cuda_devices.values())

    if mode == "unique_or_2x" and len(selected_cuda_devices) == 1 and len(cuda_devices) > 1:
        for d in cuda_devices:
            if d not in selected_cuda_devices:
                selected_cuda_devices.append(d)
                break

    return selected_cuda_devices


def get_test_devices(mode: str | None = None):
    """Returns a list of devices based on the mode selected.

    Args:
        mode: The testing mode to specify which devices to include. If not provided or ``None``, the
          ``global test_mode`` value will be used.
          "basic": Returns the CPU and the first GPU device when available.
          "unique": Returns the CPU and all unique GPU architectures.
          "unique_or_2x" (default): Behaves like "unique" but adds up to one additional CUDA device
            if the system only devices of a single CUDA architecture.
          "all": Returns all available devices.
    """
    if mode is None:
        mode = test_mode

    devices = []

    if mode == "basic":
        # only run on CPU and first GPU device
        if wp.is_cpu_available():
            devices.append(wp.get_device("cpu"))
        if wp.is_cuda_available():
            devices.append(wp.get_device("cuda:0"))
    elif mode == "unique" or mode == "unique_or_2x":
        # run on CPU and a subset of GPUs
        if wp.is_cpu_available():
            devices.append(wp.get_device("cpu"))
        devices.extend(get_selected_cuda_test_devices(mode))
    elif mode == "all":
        # run on all devices
        devices = wp.get_devices()
    else:
        raise ValueError(f"Unknown test mode selected: {mode}")

    return devices


def get_cuda_test_devices(mode=None):
    devices = get_test_devices(mode=mode)
    return [d for d in devices if d.is_cuda]


def configure_sdf_for_collision_shapes(builder):
    """Force volume-SDF construction on every mesh/convex shape that collides with particles.

    Test helper for the full-surface rigid-soft path: sets ``force_sdf`` on the builder's mesh/convex
    ``COLLIDE_PARTICLES`` shapes (regardless of whether they used the default or an explicit config), so
    ``finalize()`` provisions their SDFs. Mirrors what a user would do with per-shape
    ``ShapeConfig.configure_sdf(force_sdf=True)``.
    """
    from newton import GeoType  # noqa: PLC0415  (deferred: keep unittest_utils import-light)
    from newton._src.geometry.flags import ShapeFlags  # noqa: PLC0415

    for i in range(len(builder.shape_type)):
        if int(builder.shape_type[i]) in (int(GeoType.MESH), int(GeoType.CONVEX_MESH)) and (
            builder.shape_flags[i] & int(ShapeFlags.COLLIDE_PARTICLES)
        ):
            builder.shape_force_sdf[i] = True


class StreamCapture:
    def __init__(self, stream_name):
        self.stream_name = stream_name  # 'stdout' or 'stderr'
        self.saved = None
        self.stream_fd = None
        self.target = None
        self.tempfile = None

    def begin(self):
        # Flush the stream buffers managed by libc.
        # This is needed at the moment due to Carbonite not flushing the logs
        # being printed out when extensions are starting up.
        if LIBC is not None:
            LIBC.fflush(None)

        # Get the stream object (sys.stdout or sys.stderr)
        self.saved = getattr(sys, self.stream_name)
        try:
            self.stream_fd = self.saved.fileno()
        except (AttributeError, io.UnsupportedOperation):
            self.stream_fd = getattr(sys, f"__{self.stream_name}__").fileno()
        self.target = os.dup(self.stream_fd)

        # Create temporary capture stream
        self.tempfile = io.TextIOWrapper(
            tempfile.TemporaryFile(buffering=0),
            encoding="utf-8",
            errors="replace",
            newline="",
            write_through=True,
        )

        # Redirect the stream
        os.dup2(self.tempfile.fileno(), self.stream_fd)
        setattr(sys, self.stream_name, self.tempfile)

    def end(self):
        # The following sleep doesn't seem to fix the test_print failure on Windows
        # if sys.platform == "win32":
        #    # Workaround for what seems to be a Windows-specific bug where
        #    # the output of CUDA's printf is not being immediately flushed
        #    # despite the context synchronization.
        #    time.sleep(0.01)
        if LIBC is not None:
            LIBC.fflush(None)

        # Restore the original stream
        os.dup2(self.target, self.stream_fd)
        os.close(self.target)

        # Read the captured output
        self.tempfile.seek(0)
        res = self.tempfile.buffer.read()
        self.tempfile.close()

        # Restore the stream object
        setattr(sys, self.stream_name, self.saved)

        return str(res.decode("utf-8"))


# Subclasses for specific streams
class StdErrCapture(StreamCapture):
    def __init__(self):
        super().__init__("stderr")


class StdOutCapture(StreamCapture):
    def __init__(self):
        super().__init__("stdout")


class CheckOutput:
    def __init__(self, test):
        self.test = test

    def __enter__(self):
        # wp.force_load()

        self.capture = StdOutCapture()
        self.capture.begin()

    def __exit__(self, exc_type, exc_value, traceback):
        # ensure any stdout output is flushed
        wp.synchronize()

        s = self.capture.end()
        if s != "":
            print(s.rstrip())

            # fail if test produces unexpected output (e.g.: from wp.expect_eq() builtins)
            # we allow strings starting of the form "Module xxx load on device xxx"
            # for lazy loaded modules
            filtered_s = "\n".join(
                [line for line in s.splitlines() if not (line.startswith("Module") and "load on device" in line)]
            )

            if filtered_s.strip():
                self.test.fail(f"Unexpected output:\n'{s.rstrip()}'")


@dataclasses.dataclass
class _OutputRegex:
    """A single output expectation for the strict output contract.

    Attributes:
        pattern: Regular expression matched against captured output.
        stream: Which stream the pattern applies to: ``"stdout"``,
            ``"stderr"``, or ``"any"``.
        required: Whether the pattern must match (expected output) or is
            merely permitted (allowed output).
    """

    pattern: str
    stream: str
    required: bool


class _OutputCapture:
    """Captures stdout/stderr during a test and checks it against patterns.

    Output is captured between :meth:`begin` and :meth:`finish`. Registered
    patterns are then matched against the captured streams: required patterns
    must appear, and any output left unmatched by every pattern is reported as
    unexpected. This enforces the strict output contract used by
    :class:`NewtonTestCase`.
    """

    def __init__(self):
        self.stdout_capture = StdOutCapture()
        self.stderr_capture = StdErrCapture()
        self.output = {"stdout": [], "stderr": []}
        self.patterns: list[_OutputRegex] = []
        self.active = False

    def begin(self):
        self.stdout_capture.begin()
        try:
            self.stderr_capture.begin()
        except BaseException:
            self.stdout_capture.end()
            raise
        self.active = True

    def add_pattern(self, pattern: str, *, stream: str, required: bool):
        if stream not in {"stdout", "stderr", "any"}:
            raise ValueError(f"Unknown stream {stream!r}; expected 'stdout', 'stderr', or 'any'")

        self.patterns.append(_OutputRegex(pattern=pattern, stream=stream, required=required))

    def record(self, stream: str, text: str | bytes | None):
        if text is None:
            return
        if isinstance(text, bytes):
            text = text.decode("utf-8", errors="replace")
        if text:
            self.output[stream].append(str(text))

    def finish(self) -> str | None:
        if not self.active:
            return None

        failure = None
        try:
            try:
                # Match CheckOutput: flush async Warp kernel output before reading captured fds.
                wp.synchronize()
            except BaseException as exc:
                failure = exc
            finally:
                for stream, capture in (("stderr", self.stderr_capture), ("stdout", self.stdout_capture)):
                    try:
                        self.record(stream, capture.end())
                    except BaseException as exc:
                        if failure is None:
                            failure = exc
        finally:
            self.active = False

        if failure is not None:
            raise failure

        return self._check_output()

    def _check_output(self) -> str | None:
        output_by_stream = {stream: "".join(chunks) for stream, chunks in self.output.items()}
        unmatched_by_stream = output_by_stream.copy()
        missing = []

        for pattern in self.patterns:
            streams = ("stdout", "stderr") if pattern.stream == "any" else (pattern.stream,)
            matched = any(
                re.search(pattern.pattern, output_by_stream[stream], flags=re.MULTILINE) for stream in streams
            )

            if pattern.required and not matched:
                missing.append(pattern)

            for stream in streams:
                unmatched_by_stream[stream] = re.sub(
                    pattern.pattern,
                    "",
                    unmatched_by_stream[stream],
                    flags=re.MULTILINE,
                )

        failures = []
        if missing:
            failures.append(
                "Missing expected output:\n"
                + "\n".join(f"- {pattern.stream}: /{pattern.pattern}/" for pattern in missing)
            )

        for stream, unmatched in unmatched_by_stream.items():
            if unmatched.strip():
                failures.append(f"Unexpected {stream}:\n{unmatched.rstrip()}")

        if failures:
            return "\n\n".join(failures)

        return None


class NewtonTestCase(unittest.TestCase):
    """TestCase with strict stdout/stderr output checking.

    Inheriting this class opts the test into a strict output contract:
    stdout and stderr must be empty unless a test explicitly expects or
    allows matching output.
    """

    _output_capture: _OutputCapture | None = None

    def _callSetUp(self):
        self._output_capture = _OutputCapture()
        self._output_capture.begin()
        self.addCleanup(self._finish_output_capture)
        super()._callSetUp()

    def expectOutputRegex(self, regex: str, *, stream: str = "any"):
        """Allow matching stdout/stderr output and fail if it does not appear."""

        self._require_output_capture().add_pattern(regex, stream=stream, required=True)

    def allowOutputRegex(self, regex: str, *, stream: str = "any"):
        """Allow matching stdout/stderr output without requiring it."""

        self._require_output_capture().add_pattern(regex, stream=stream, required=False)

    def assertSubprocessSuccess(self, result, *, command):
        """Assert a subprocess succeeded and include its output in this test's output contract."""

        output_capture = self._require_output_capture()
        stdout = getattr(result, "stdout", None)
        stderr = getattr(result, "stderr", None)
        output_capture.record("stdout", stdout)
        output_capture.record("stderr", stderr)

        if result.returncode != 0:
            command_text = _format_command(command)
            self.fail(
                f"Failed with return code {result.returncode}, command: {command_text}\n\nOutput:\n{stdout}\n{stderr}"
            )

    def _finish_output_capture(self):
        output_capture = self._output_capture
        self._output_capture = None
        if output_capture is None:
            return

        failure = output_capture.finish()
        if failure is not None and not self._has_recorded_failure_or_error():
            self.fail(failure)

    def _require_output_capture(self) -> _OutputCapture:
        if self._output_capture is None:
            raise RuntimeError("Output capture is not active for this test")

        return self._output_capture

    def _has_recorded_failure_or_error(self) -> bool:
        outcome = getattr(self, "_outcome", None)
        result = getattr(outcome, "result", None)
        if result is None:
            return False

        for issue_list in (result.failures, result.errors):
            if any(test is self for test, _ in issue_list):
                return True

        return False


def _format_command(command) -> str:
    if isinstance(command, str):
        return command

    return shlex.join(str(arg) for arg in command)


def assert_array_equal(result: wp.array, expect: wp.array):
    np.testing.assert_equal(result.numpy(), expect.numpy())


def assert_np_equal(result: np.ndarray, expect: np.ndarray, tol=0.0):
    if tol != 0.0:
        # TODO: Get all tests working without the .flatten()
        np.testing.assert_allclose(result.flatten(), expect.flatten(), atol=tol, equal_nan=True)
    else:
        # TODO: Get all tests working with strict=True
        np.testing.assert_array_equal(result, expect)


def most(x: np.ndarray, min_ratio: float = 0.8) -> bool:
    """Helper function to check if most elements of an array are greater than 0 (or True)."""
    if len(x) == 0:
        return True
    return bool(np.sum(x > 0) / len(x) >= min_ratio)


def find_nan_members(obj: Any | None) -> list[str]:
    """Helper function to find any Warp array members of an object that contain NaN values."""
    nan_members = []
    if obj is None:
        return nan_members
    for key, attr in obj.__dict__.items():
        if isinstance(attr, wp.array):
            arr = attr.numpy()
            # Skip structured arrays (e.g., arrays of warp structs) - np.isnan doesn't support them
            if arr.dtype.names is not None:
                continue
            if np.isnan(arr).any():
                nan_members.append(key)
    return nan_members


def find_nonfinite_members(obj: Any | None) -> list[str]:
    """Helper function to find any Warp array members of an object that contain non-finite values."""
    nonfinite_members = []
    if obj is None:
        return nonfinite_members
    for key, attr in obj.__dict__.items():
        if isinstance(attr, wp.array):
            arr = attr.numpy()
            # Skip structured arrays (e.g., arrays of warp structs) - np.isfinite doesn't support them
            if arr.dtype.names is not None:
                continue
            if not np.isfinite(arr).all():
                nonfinite_members.append(key)
    return nonfinite_members


# For legacy TestCase classes, check_output=True wraps the function in CheckOutput.
# NewtonTestCase subclasses use their own stdout/stderr output contract instead.
def create_test_func(func, device, check_output, **kwargs):
    # pass args to func
    def test_func(self):
        if check_output and not isinstance(self, NewtonTestCase):
            with CheckOutput(self):
                func(self, device, **kwargs)
        else:
            func(self, device, **kwargs)

    # Copy the __unittest_expecting_failure__ attribute from func to test_func
    if hasattr(func, "__unittest_expecting_failure__"):
        test_func.__unittest_expecting_failure__ = func.__unittest_expecting_failure__

    return test_func


def skip_test_func(self):
    # A function to use so we can tell unittest that the test was skipped.
    self.skipTest("No suitable devices to run the test.")


def sanitize_identifier(s):
    """replace all non-identifier characters with '_'"""

    s = str(s)
    if s.isidentifier():
        return s
    else:
        return re.sub(r"\W|^(?=\d)", "_", s)


def add_function_test(cls, name, func, devices=None, check_output=True, **kwargs):
    if devices is None:
        setattr(cls, name, create_test_func(func, None, check_output, **kwargs))
    elif isinstance(devices, list):
        if not devices:
            # No devices to run this test
            setattr(cls, name, skip_test_func)
        else:
            for device in devices:
                setattr(
                    cls,
                    name + "_" + sanitize_identifier(device),
                    create_test_func(func, device, check_output, **kwargs),
                )
    else:
        setattr(
            cls,
            name + "_" + sanitize_identifier(devices),
            create_test_func(func, devices, check_output, **kwargs),
        )


def add_kernel_test(cls, kernel, dim, name=None, expect=None, inputs=None, devices=None):
    def test_func(self, device):
        args = []
        if inputs:
            args.extend(inputs)

        if expect:
            # allocate outputs to match results
            result = wp.array(expect, dtype=int, device=device)
            output = wp.zeros_like(result)

            args.append(output)

        # force load so that we don't generate any log output during launch
        kernel.module.load(device)

        with CheckOutput(self):
            wp.launch(kernel, dim=dim, inputs=args, device=device)

        # check output values
        if expect:
            assert_array_equal(output, result)

    if name is None:
        name = kernel.key

    # device is required for kernel tests, so use all devices if none were given
    if devices is None:
        devices = get_test_devices()

    # register test func with class for the given devices
    for d in devices:
        # use a function to forward the device to the inner test function
        def test_func_wrapper(test, device=d):
            test_func(test, device)

        setattr(cls, name + "_" + sanitize_identifier(d), test_func_wrapper)


# helper that first calls the test function to generate all kernel permutations
# so that compilation is done in one-shot instead of per-test
def add_function_test_register_kernel(cls, name, func, devices=None, **kwargs):
    func(None, None, **kwargs, register_kernels=True)
    add_function_test(cls, name, func, devices=devices, **kwargs)


def write_junit_results(
    outfile: str,
    test_records: list,
    tests_run: int,
    tests_failed: int,
    tests_errored: int,
    tests_skipped: int,
    test_duration: float,
):
    """Write a JUnit XML from our report data

    The report file is needed for GitLab to add test reports in merge requests.
    """

    root = ET.Element(
        "testsuite",
        name="Warp Tests",
        failures=str(tests_failed),
        errors=str(tests_errored),
        skipped=str(tests_skipped),
        tests=str(tests_run),
        time=f"{test_duration:.3f}",
    )

    for test_data in test_records:
        test_classname = test_data[0]
        test_methodname = test_data[1]
        test_duration = test_data[2]
        test_status = test_data[3]

        test_case = ET.SubElement(
            root, "testcase", classname=test_classname, name=test_methodname, time=f"{test_duration:.3f}"
        )

        if test_status == "FAIL":
            failure = ET.SubElement(test_case, "failure", message=str(test_data[4]))
            failure.text = str(test_data[5])  # Stacktrace
        elif test_status == "ERROR":
            error = ET.SubElement(test_case, "error")
            error.text = str(test_data[5])  # Stacktrace
        elif test_status == "SKIP":
            skip = ET.SubElement(test_case, "skipped")
            # Set the skip reason
            skip.set("message", str(test_data[4]))

    tree = ET.ElementTree(root)

    if hasattr(ET, "indent"):
        ET.indent(root)  # Pretty-printed XML output, Python 3.9 required

    tree.write(outfile, encoding="utf-8", xml_declaration=True)


class ParallelJunitTestResult(unittest.TextTestResult):
    def __init__(self, stream, descriptions, verbosity):
        stream = type(stream)(sys.stderr)
        self.test_record = []
        super().__init__(stream, descriptions, verbosity)

    def startTest(self, test):
        if self.showAll:
            self.stream.writeln(f"{self.getDescription(test)} ...")
            self.stream.flush()
        elif self.dots:
            self.stream.writeln(f"{test} ...")
            self.stream.flush()
        self.start_time = time.perf_counter_ns()
        super(unittest.TextTestResult, self).startTest(test)

    def _add_helper(self, test, show_all_message):
        if self.showAll:
            self.stream.writeln(f"{self.getDescription(test)} ... {show_all_message}")
        elif self.dots:
            self.stream.writeln(f"{test} ... {show_all_message}")
        self.stream.flush()

    def _record_test(self, test, code, message=None, details=None):
        # For class-level skips (setUpClass raising SkipTest), unittest passes an
        # _ErrorHolder instead of a real test case, and startTest is never called.
        # Guard against missing start_time and _testMethodName.
        start = getattr(self, "start_time", None)
        duration = round((time.perf_counter_ns() - start) * 1e-9, 3) if start is not None else 0.0
        class_name = test.__class__.__name__
        method_name = getattr(test, "_testMethodName", str(test))
        self.test_record.append((class_name, method_name, duration, code, message, details))

    def addSuccess(self, test):
        super(unittest.TextTestResult, self).addSuccess(test)
        self._add_helper(test, "ok")
        self._record_test(test, "OK")

    def addError(self, test, err):
        super(unittest.TextTestResult, self).addError(test, err)
        self._add_helper(test, "ERROR")
        self._record_test(test, "ERROR", str(err[1]), self._exc_info_to_string(err, test))

    def addFailure(self, test, err):
        super(unittest.TextTestResult, self).addFailure(test, err)
        self._add_helper(test, "FAIL")
        self._record_test(test, "FAIL", str(err[1]), self._exc_info_to_string(err, test))

    def addSkip(self, test, reason):
        super(unittest.TextTestResult, self).addSkip(test, reason)
        self._add_helper(test, f"skipped {reason!r}")
        self._record_test(test, "SKIP", reason)

    def addExpectedFailure(self, test, err):
        super(unittest.TextTestResult, self).addExpectedFailure(test, err)
        self._add_helper(test, "expected failure")
        self._record_test(test, "OK", "expected failure")

    def addUnexpectedSuccess(self, test):
        super(unittest.TextTestResult, self).addUnexpectedSuccess(test)
        self._add_helper(test, "unexpected success")
        self._record_test(test, "FAIL", "unexpected success")

    def addSubTest(self, test, subtest, err):
        super(unittest.TextTestResult, self).addSubTest(test, subtest, err)
        if err is not None:
            self._add_helper(test, "ERROR")
            # err is (class, error, traceback)
            self._record_test(test, "FAIL", str(err[1]), self._exc_info_to_string(err, test))

    def stopTest(self, test):
        super().stopTest(test)
        # Force garbage collection of CPU-side allocations and release unused
        # CUDA mempool memory to reduce peak host RSS in parallel test runs
        # (see issue #1881).
        import gc  # noqa: PLC0415

        gc.collect()
        import warp as wp  # noqa: PLC0415

        for device_name in wp.get_cuda_devices():
            if wp.is_mempool_enabled(device_name):
                wp.set_mempool_release_threshold(device_name, 0)

    def printErrors(self):
        pass
