# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import subprocess
import sys
import tempfile
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

    def test_deprecation_allowlist_file(self):
        """Read message prefixes while rejecting Python -W separators."""
        with tempfile.TemporaryDirectory() as temp_dir:
            allowlist_path = Path(temp_dir) / "deprecations.txt"
            allowlist_path.write_text(
                "# Temporary migration\n\ndependency.old_api is deprecated\n",
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
