# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import contextlib
import io
import subprocess
import sys
import tempfile
import textwrap
import unittest
import warnings
from pathlib import Path
from unittest import mock

import newton.tests.unittest_utils
from newton.tests.thirdparty.unittest_parallel import (
    _enable_strict_warnings,
    _read_deprecation_allowlist,
)


class TestWarningPolicy(unittest.TestCase):
    def test_deprecation_allowlist_preserves_strictness(self):
        """Allow listed deprecations while rejecting unlisted ones."""
        allowed_prefix = "dependency.old_api is deprecated"
        allowed_message = f"{allowed_prefix}; use dependency.new_api instead"

        with warnings.catch_warnings(record=True) as emitted:
            _enable_strict_warnings((allowed_prefix,))
            warnings.warn(allowed_message, DeprecationWarning, stacklevel=2)
            with self.assertRaises(DeprecationWarning):
                warnings.warn("unexpected dependency deprecation", DeprecationWarning, stacklevel=2)

        self.assertEqual([str(item.message) for item in emitted], [allowed_message])

    @unittest.skipUnless(sys.version_info >= (3, 14), "Requires Python 3.14 ctypes layout warnings")
    @newton.tests.unittest_utils.requires("warp._src.apic.types")
    def test_warp_ctypes_layout_warnings(self):
        """Import real packed Warp structs while keeping other deprecations fatal."""
        with mock.patch.object(newton.tests.unittest_utils, "strict_warnings", True):
            warning_args = newton.tests.unittest_utils.get_strict_warning_args()
        for policy in ("runner", "subprocess"):
            with self.subTest(policy=policy):
                script = textwrap.dedent(
                    f"""
                    import ctypes
                    import ctypes._layout
                    import warnings
                    import sys
                    from types import SimpleNamespace

                    if {policy!r} == "runner":
                        from newton.tests.thirdparty.unittest_parallel import _enable_strict_warnings
                        _enable_strict_warnings()

                    # Exercise the real POSIX ctypes layout path on Windows too.
                    ctypes._layout.sys = SimpleNamespace(platform="linux", byteorder=sys.byteorder)
                    import warp._src.apic.types as apic

                    if hasattr(apic, "APICCudaCompileRecipe"):
                        assert ctypes.sizeof(apic.APICCudaCompileRecipe) == 16
                    assert ctypes.sizeof(apic.APICLaunchParamRecord) == 16
                    assert ctypes.sizeof(apic.APICLaunchPtrLocation) == 24
                    for message, module in (
                        ("unrelated deprecation", "warp._src.apic.types"),
                        ("Due to '_pack_', the 'APICCudaCompileRecipe' Structure will use "
                         "memory layout compatible with MSVC (Windows).", "newton.tests.other"),
                    ):
                        try:
                            warnings.warn_explicit(message, DeprecationWarning, "probe.py", 1, module=module)
                        except DeprecationWarning:
                            pass
                        else:
                            raise AssertionError("Unexpected deprecation was allowed")
                    """
                )
                result = subprocess.run(
                    [sys.executable, *(warning_args if policy == "subprocess" else []), "-c", script],
                    capture_output=True,
                    text=True,
                    check=False,
                )
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                self._check_dependency_warning_output(result.stderr)

    @unittest.skipUnless(sys.platform == "win32" and sys.version_info >= (3, 14), "Requires Windows Python 3.14")
    @newton.tests.unittest_utils.requires("pyglet")
    def test_pyglet_ctypes_pointer_warning(self):
        """Import real Windows COM types without permitting other pointer-cache deprecations."""
        with mock.patch.object(newton.tests.unittest_utils, "strict_warnings", True):
            warning_args = newton.tests.unittest_utils.get_strict_warning_args()
        for policy in ("runner", "subprocess"):
            with self.subTest(policy=policy):
                script = textwrap.dedent(
                    f"""
                    import ctypes
                    import warnings
                    if {policy!r} == "runner":
                        from newton.tests.thirdparty.unittest_parallel import _enable_strict_warnings
                        _enable_strict_warnings()
                    from pyglet.libs.win32 import com
                    assert ctypes.sizeof(com.GUID) == 16
                    try:
                        warnings.warn("'ctypes._pointer_type_cache' is deprecated", DeprecationWarning)
                    except DeprecationWarning:
                        pass
                    else:
                        raise AssertionError("Unrelated pointer-cache use was allowed")
                    """
                )
                result = subprocess.run(
                    [sys.executable, *(warning_args if policy == "subprocess" else []), "-c", script],
                    capture_output=True,
                    text=True,
                    check=False,
                )
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                self._check_dependency_warning_output(result.stderr)

    def _check_dependency_warning_output(self, stderr):
        output = newton.tests.unittest_utils._OutputCapture()
        output.record("stderr", stderr)
        replayed = io.StringIO()
        with (
            mock.patch.object(newton.tests.unittest_utils, "strict_warnings", True),
            contextlib.redirect_stderr(replayed),
        ):
            self.assertIsNone(output._check_output())
        self.assertCountEqual(replayed.getvalue().splitlines(), stderr.splitlines())

    def test_deprecation_allowlist_file(self):
        """Read message prefixes while rejecting Python -W separators."""
        with tempfile.TemporaryDirectory() as temp_dir:
            allowlist_path = Path(temp_dir) / "deprecations.txt"
            allowlist_path.write_text(
                "# Temporary migration\n# remove-when-minimum: dependency>=2\n\ndependency.old_api is deprecated\n",
                encoding="utf-8",
            )
            self.assertEqual(
                _read_deprecation_allowlist(allowlist_path),
                ("dependency.old_api is deprecated",),
            )

            allowlist_path.write_text("dependency: old_api is deprecated\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "unique message prefix ending before the colon"):
                _read_deprecation_allowlist(allowlist_path)

    def test_deprecation_allowlist_propagates_to_python_subprocesses(self):
        """Preserve strict deprecation filtering in Python subprocesses."""
        allowed_prefix = "dependency.old_api is deprecated"
        allowed_message = f"{allowed_prefix}; use new_api instead"
        unexpected_message = "unexpected dependency deprecation"
        with (
            mock.patch.object(newton.tests.unittest_utils, "strict_warnings", True),
            mock.patch.object(newton.tests.unittest_utils, "allowed_deprecation_warnings", (allowed_prefix,)),
        ):
            warning_args = newton.tests.unittest_utils.get_strict_warning_args()

        result = subprocess.run(
            [
                sys.executable,
                *warning_args,
                "-c",
                "import warnings; "
                f"warnings.warn({allowed_message!r}, DeprecationWarning); "
                f"warnings.warn({unexpected_message!r}, DeprecationWarning)",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertNotEqual(result.returncode, 0, msg=result.stdout + result.stderr)
        self.assertIn(allowed_message, result.stderr)
        self.assertIn(unexpected_message, result.stderr)


if __name__ == "__main__":
    unittest.main(verbosity=2)
