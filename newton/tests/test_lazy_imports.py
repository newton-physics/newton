# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import os
import subprocess
import sys
import unittest
from unittest import mock

import newton
import newton.tests.unittest_utils
from newton._src import solvers as internal_solvers


def _run_in_fresh_interpreter(code: str) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env.pop("PYTHONWARNINGS", None)
    result = subprocess.run(
        [sys.executable, *newton.tests.unittest_utils.get_strict_warning_args(), "-c", code],
        capture_output=True,
        env=env,
        text=True,
        check=True,
    )
    sys.stderr.write(result.stderr)
    return result


class TestLazySolverImports(unittest.TestCase):
    def test_fresh_interpreter_propagates_strict_warning_policy(self):
        """Propagate allowed and unacknowledged deprecations to a fresh interpreter."""
        allowed_prefix = "dependency.old_api is deprecated"

        def warning_code(message: str) -> str:
            source = f"warnings.warn({message!r}, DeprecationWarning)"
            return (
                "import warnings; "
                f"exec(compile({source!r}, 'dependency.py', 'exec'), "
                "{'__name__': 'dependency', 'warnings': warnings})"
            )

        with (
            mock.patch.object(newton.tests.unittest_utils, "strict_warnings", True),
            mock.patch.object(
                newton.tests.unittest_utils,
                "allowed_deprecation_warnings",
                (allowed_prefix,),
            ),
        ):
            allowed_result = _run_in_fresh_interpreter(warning_code(f"{allowed_prefix}; use new_api instead"))
            with self.assertRaises(subprocess.CalledProcessError) as raised:
                _run_in_fresh_interpreter(warning_code("unexpected dependency deprecation"))

        self.assertEqual(allowed_result.returncode, 0)
        self.assertIn(allowed_prefix, allowed_result.stderr)
        self.assertIn("unexpected dependency deprecation", raised.exception.stderr)

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
        result = _run_in_fresh_interpreter(code)
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
        result = _run_in_fresh_interpreter(code)
        self.assertEqual(result.stdout.strip(), "ok")

    @newton.tests.unittest_utils.requires(*newton.tests.unittest_utils.MUJOCO_MODULES)
    def test_first_mujoco_import_does_not_pin_caller(self):
        """Verify that the first MuJoCo import releases the stack that triggered it."""
        code = (
            "import gc, os, sys, weakref\n"
            "import newton\n"
            "os.environ['MUJOCO_GL'] = 'glfw'\n"
            "sys.modules['glfw'] = None  # MuJoCo only stores the error when its GL backend fails to import\n"
            "class Local: pass\n"
            "def caller():\n"
            "    local = Local()\n"
            "    newton.solvers.SolverMuJoCo.import_mujoco()\n"
            "    return weakref.ref(local)\n"
            "ref = caller()\n"
            "gc.collect()\n"
            "print('released' if ref() is None else 'pinned')\n"
        )
        result = _run_in_fresh_interpreter(code)
        self.assertEqual(result.stdout.strip().splitlines()[-1], "released")


if __name__ == "__main__":
    unittest.main(verbosity=2)
