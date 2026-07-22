# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import subprocess
import sys
import unittest

import newton
from newton._src import solvers as internal_solvers


class TestLazySolverImports(unittest.TestCase):
    def test_import_newton_does_not_import_solvers(self):
        """Verify that importing newton does not import any solver backend module."""
        backends = (
            "coupled",
            "featherstone",
            "implicit_mpm",
            "kamino",
            "mujoco",
            "semi_implicit",
            "style3d",
            "vbd",
            "xpbd",
        )
        code = (
            "import sys; import newton; "
            f"prefixes = tuple(f'newton._src.solvers.{{name}}' for name in {backends!r}); "
            "loaded = [m for m in sys.modules if m.startswith(prefixes)]; "
            "print(','.join(loaded))"
        )
        result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
        self.assertEqual(result.stdout.strip(), "", f"solver modules imported eagerly: {result.stdout.strip()}")

    def test_public_exports_match_internal_exports(self):
        """Expose the internal solver surface through the public module."""
        self.assertEqual(set(newton.solvers.__all__), set(internal_solvers.__all__) | {"experimental"})

    def test_lazy_attributes_resolve(self):
        """Verify that every public solver symbol resolves to the implementation object."""
        for name in newton.solvers.__all__:
            with self.subTest(name=name):
                self.assertTrue(hasattr(newton.solvers, name))
                self.assertIn(name, dir(newton.solvers))
        self.assertTrue(issubclass(newton.solvers.SolverSemiImplicit, newton.solvers.SolverBase))
        with self.assertRaises(AttributeError):
            _ = newton.solvers.SolverNonexistent

    def test_experimental_coupled_import(self):
        """Verify that the experimental coupled-solver package imports lazily in a fresh interpreter."""
        code = (
            "from newton.solvers.experimental.coupled import SolverCoupled, SolverCoupledProxy; "
            "import newton.solvers.experimental.coupled as coupled; "
            "assert coupled.SolverCoupled is SolverCoupled; "
            "print('ok')"
        )
        result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
        self.assertEqual(result.stdout.strip(), "ok")


if __name__ == "__main__":
    unittest.main(verbosity=2)
