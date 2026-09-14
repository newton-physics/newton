# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Assert that ``import newton`` does not trigger ``wp.init()``."""

import os
import subprocess
import sys
import unittest

import newton.tests.unittest_utils


class TestLazyInit(unittest.TestCase):
    def test_import_newton_does_not_init_warp(self):
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


if __name__ == "__main__":
    unittest.main()
