# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Assert that ``import newton`` does not trigger ``wp.init()``."""

import subprocess
import sys
import unittest


class TestLazyInit(unittest.TestCase):
    def test_import_newton_does_not_init_warp(self):
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                "import newton; import warp._src.context as wpc; import sys; sys.exit(0 if wpc.runtime is None else 1)",
            ],
            capture_output=True,
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
