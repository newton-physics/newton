# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Import smoke test for every supported Python version.

The full suite runs only on the version in ``.python-version``; CI runs this
module on the others to catch import-time failures such as #3941.
"""

import unittest

import newton


class TestPythonCompatibility(unittest.TestCase):
    def test_stable_solver_exports_resolve(self):
        """Resolve every stable public solver export."""
        for name in newton.solvers.__all__:
            if name == "experimental":
                continue
            with self.subTest(name=name):
                _ = getattr(newton.solvers, name)


if __name__ == "__main__":
    unittest.main(verbosity=2)
