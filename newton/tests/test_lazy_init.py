# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Guard against eager Warp initialization on ``import newton``.

Warp resolves its kernel cache directory inside ``Runtime.__init__`` from
``warp.config.kernel_cache_dir``.  Once Runtime exists, later assignments
to ``warp.config.kernel_cache_dir`` cannot trigger another resolution, so
test runners and tools that reconfigure the cache rely on being able to
do so before Warp initializes.

A module-scope ``wp.<builtin>(...)`` call (e.g. ``wp.sin``) inside Newton
forces ``wp.init()`` during import via Warp's builtin resolver.  This
test runs ``import newton`` in a fresh subprocess and fails if Warp's
Runtime has been created — pointing the regressing change toward
``math.<fn>`` substitutes that ``wp.static`` can fold to a literal.
"""

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
            msg=(
                "import newton triggered wp.init() (Warp runtime is non-None "
                "after import). This typically means a module-scope "
                "wp.<builtin>(...) call (e.g. wp.sin, wp.cos) is forcing "
                "eager initialization via Warp's builtin resolver. Replace "
                "it with the equivalent math.* function so wp.static folds "
                "to a literal.\n\n"
                f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
            ),
        )


if __name__ == "__main__":
    unittest.main()
