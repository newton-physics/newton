# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Compatibility checks that must hold on every supported Python version.

``pyproject.toml`` declares support down to Python 3.10, so the stable public
API has to resolve there. Resolving a solver export imports the module that
defines it, and annotations in that module are evaluated at import time unless
postponed -- which is how a Python 3.10 ``TypeError`` reached users through
``ModelBuilder()`` in #3941.
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
