# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Compatibility checks that must hold on every supported Python version.

``pyproject.toml`` declares support down to Python 3.10, so the stable public
API has to resolve there. Resolving a solver export imports the module that
defines it, and annotations in that module are evaluated at import time unless
postponed -- which is how a Python 3.10 ``TypeError`` reached users through
``ModelBuilder()`` in #3941.
"""

import subprocess
import sys
import unittest

import newton


class TestPythonCompatibility(unittest.TestCase):
    def test_python_deprecation_warning(self):
        """Show the import warning once on Python 3.10 and never on newer Python."""
        result = subprocess.run(
            [sys.executable, "-I", "-c", "import newton; import newton"],
            capture_output=True,
            text=True,
            check=True,
        )
        message = "NewtonDeprecationWarning: Support for Python 3.10 is deprecated"
        self.assertEqual(result.stderr.count(message), int(sys.version_info < (3, 11)), result.stderr)
        if sys.version_info < (3, 11):
            self.assertIn("removed in Newton 1.9", result.stderr)
            self.assertIn("upgrade to Python 3.11 or newer", result.stderr)

    def test_stable_solver_exports_resolve(self):
        """Resolve every stable public solver export."""
        for name in newton.solvers.__all__:
            if name == "experimental":
                continue
            with self.subTest(name=name):
                _ = getattr(newton.solvers, name)


if __name__ == "__main__":
    unittest.main(verbosity=2)
