# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Assert that ``import newton`` does not trigger ``wp.init()``."""

import contextlib
import io
import os
import subprocess
import sys
import unittest
from unittest import mock

import newton.tests.unittest_utils


class TestLazyInit(unittest.TestCase):
    def test_import_newton_does_not_init_warp(self):
        """Import Newton without initializing the Warp runtime."""
        env = os.environ.copy()
        # Escalate import-time deprecations only when the runner opted in
        # (--strict-warnings); otherwise keep the import lenient so a dependency
        # deprecation does not fail a consumer's install check.
        env.pop("PYTHONWARNINGS", None)
        warning_args = newton.tests.unittest_utils.get_strict_warning_args()

        result = subprocess.run(
            [
                sys.executable,
                *warning_args,
                "-c",
                "import newton; import warp._src.context as wpc; import sys; sys.exit(0 if wpc.runtime is None else 1)",
            ],
            capture_output=True,
            env=env,
            text=True,
            check=False,
        )
        self.assertEqual(
            result.returncode,
            0,
            msg=f"import newton triggered wp.init().\nstderr:\n{result.stderr}",
        )
        sys.stderr.write(result.stderr)

    def test_allowlisted_deprecation_remains_visible(self):
        """Replay an allowlisted deprecation from the import subprocess."""
        allowed_prefix = "dependency.old_api is deprecated"
        warning_output = f"dependency.py:1: DeprecationWarning: {allowed_prefix}; use new_api instead\n"
        process_result = subprocess.CompletedProcess(
            args=[sys.executable, "-c", "import newton"],
            returncode=0,
            stdout="",
            stderr=warning_output,
        )
        stderr = io.StringIO()

        with (
            mock.patch.object(subprocess, "run", return_value=process_result),
            mock.patch.object(newton.tests.unittest_utils, "strict_warnings", True),
            mock.patch.object(newton.tests.unittest_utils, "allowed_deprecation_warnings", (allowed_prefix,)),
            contextlib.redirect_stderr(stderr),
        ):
            self.test_import_newton_does_not_init_warp()

        self.assertEqual(stderr.getvalue(), warning_output)


if __name__ == "__main__":
    unittest.main()
