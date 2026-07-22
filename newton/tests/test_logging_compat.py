# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import ast
import contextlib
import io
import logging
import tempfile
import unittest
import warnings
from pathlib import Path
from types import SimpleNamespace

import warp as wp

import newton
import newton.examples as newton_examples
from newton._src.utils import diagnostics as newton_diagnostics
from newton.tests.thirdparty import unittest_parallel


class LoggingStateMixin:
    def setUp(self):
        self._root_logger = logging.getLogger()
        self._newton_logger = logging.getLogger("newton")
        self._saved_root_handlers = self._root_logger.handlers[:]
        self._saved_root_level = self._root_logger.level
        self._saved_newton_handlers = self._newton_logger.handlers[:]
        self._saved_newton_level = self._newton_logger.level
        self._saved_newton_propagate = self._newton_logger.propagate
        self._warp_logger = logging.getLogger("warp")
        self._saved_warp_handlers = self._warp_logger.handlers[:]
        self._saved_warp_level = self._warp_logger.level
        self._saved_warp_propagate = self._warp_logger.propagate
        self._saved_warp_logger = wp.get_logger()
        self._saved_legacy_verbose_stdout_warning_keys = newton_diagnostics._legacy_verbose_stdout_warning_keys.copy()
        newton_diagnostics._legacy_verbose_stdout_warning_keys.clear()
        self._saved_showwarning = warnings.showwarning
        self._saved_logging_showwarning = getattr(logging, "_warnings_showwarning", None)
        logging.captureWarnings(False)

    def tearDown(self):
        newton_diagnostics._legacy_verbose_stdout_warning_keys.clear()
        newton_diagnostics._legacy_verbose_stdout_warning_keys.update(self._saved_legacy_verbose_stdout_warning_keys)
        wp.set_logger(self._saved_warp_logger)
        logging.captureWarnings(False)
        warnings.showwarning = self._saved_showwarning
        logging._warnings_showwarning = self._saved_logging_showwarning
        self._root_logger.handlers[:] = self._saved_root_handlers
        self._root_logger.setLevel(self._saved_root_level)
        self._newton_logger.handlers[:] = self._saved_newton_handlers
        self._newton_logger.setLevel(self._saved_newton_level)
        self._newton_logger.propagate = self._saved_newton_propagate
        self._warp_logger.handlers[:] = self._saved_warp_handlers
        self._warp_logger.setLevel(self._saved_warp_level)
        self._warp_logger.propagate = self._saved_warp_propagate


class TestLibraryLoggingDefaults(LoggingStateMixin, unittest.TestCase):
    def test_newton_warning_logs_remain_visible_without_logging_config(self):
        """Verify WARNING-level Newton log records reach stderr without any logging config."""
        self._root_logger.handlers[:] = []
        self._newton_logger.handlers[:] = []
        self._root_logger.setLevel(logging.WARNING)
        self._newton_logger.setLevel(logging.NOTSET)
        self._newton_logger.propagate = True

        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr):
            logging.getLogger("newton.test").warning("visible Newton warning")

        self.assertIn("visible Newton warning", stderr.getvalue())

    def test_verbose_collapse_remains_visible_without_logging_config(self):
        """Verify verbose diagnostics fall back to stdout with a one-time deprecation warning."""
        self._root_logger.handlers[:] = []
        self._root_logger.setLevel(logging.WARNING)
        self._newton_logger.setLevel(logging.NOTSET)
        self._newton_logger.propagate = True

        builder = newton.ModelBuilder()
        root = builder.add_link(mass=1.0, label="root")
        free_joint = builder.add_joint_free(root, label="root_free")
        child = builder.add_link(mass=1.0, label="child")
        fixed_joint = builder.add_joint_fixed(parent=root, child=child, label="child_fixed")
        builder.add_articulation([free_joint, fixed_joint])

        stdout = io.StringIO()
        stderr = io.StringIO()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", newton.NewtonDeprecationWarning)
            with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                builder.collapse_fixed_joints(verbose=True)

        self.assertIn("Remove fixed joint child_fixed", stdout.getvalue())
        self.assertEqual(stderr.getvalue(), "")
        deprecations = [item for item in caught if issubclass(item.category, newton.NewtonDeprecationWarning)]
        self.assertEqual(len(deprecations), 1)
        self.assertIn("standard 'newton' logger", str(deprecations[0].message))


class TestNewtonWarnings(unittest.TestCase):
    def test_warning_categories_are_filterable(self):
        """Verify Newton warning categories subclass NewtonWarning for filtering."""
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            warnings.warn("geometry issue", newton.NewtonGeometryWarning, stacklevel=1)

        self.assertEqual(len(caught), 1)
        self.assertTrue(issubclass(caught[0].category, newton.NewtonGeometryWarning))
        self.assertTrue(issubclass(caught[0].category, newton.NewtonWarning))

    def test_degenerate_triangle_preserves_legacy_stdout_diagnostic(self):
        """Verify non-verbose legacy stdout diagnostics still print without emitting warnings."""
        builder = newton.ModelBuilder()
        builder.add_particle(wp.vec3(0.0, 0.0, 0.0), wp.vec3(0.0, 0.0, 0.0), 1.0)
        builder.add_particle(wp.vec3(1.0, 0.0, 0.0), wp.vec3(0.0, 0.0, 0.0), 1.0)
        builder.add_particle(wp.vec3(2.0, 0.0, 0.0), wp.vec3(0.0, 0.0, 0.0), 1.0)

        stdout = io.StringIO()
        stderr = io.StringIO()

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                area = builder.add_triangle(0, 1, 2)

        self.assertEqual(area, 0.0)
        self.assertEqual(stdout.getvalue(), "inverted or degenerate triangle element\n")
        self.assertEqual(stderr.getvalue(), "")
        self.assertEqual(len(caught), 0)


class TestEntryPointLogging(LoggingStateMixin, unittest.TestCase):
    def _build_fixed_joint_model(self):
        builder = newton.ModelBuilder()
        root = builder.add_link(mass=1.0, label="root")
        free_joint = builder.add_joint_free(root, label="root_free")
        child = builder.add_link(mass=1.0, label="child")
        fixed_joint = builder.add_joint_fixed(parent=root, child=child, label="child_fixed")
        builder.add_articulation([free_joint, fixed_joint])
        return builder

    def test_examples_configure_stdlib_logging_and_warning_capture(self):
        """Verify example entry points route info to stdout and warnings/errors to stderr."""
        self._root_logger.handlers[:] = []
        configure_logging = getattr(newton_examples, "_configure_logging", None)
        self.assertTrue(callable(configure_logging))
        args = newton_examples.create_parser().parse_args(["--viewer", "null"])

        stdout = io.StringIO()
        stderr = io.StringIO()
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            configure_logging(args)
            logging.getLogger("newton.test").info("visible example info")
            logging.getLogger("newton.test").warning("visible example warning")
            wp.get_logger().info("visible warp info")
            wp.get_logger().error("visible warp error")
            logging.getLogger("other.test").info("hidden third-party info")
            self._build_fixed_joint_model().collapse_fixed_joints(verbose=True)
            warnings.warn("captured example warning", UserWarning, stacklevel=1)

        self.assertTrue(self._root_logger.handlers)
        self.assertIn("Remove fixed joint child_fixed", stdout.getvalue())
        self.assertNotIn("hidden third-party info", stdout.getvalue())
        self.assertNotIn("hidden third-party info", stderr.getvalue())
        self.assertIn("visible example info", stdout.getvalue())
        self.assertNotIn("visible example info", stderr.getvalue())
        self.assertIn("visible warp info", stdout.getvalue())
        self.assertNotIn("visible warp info", stderr.getvalue())
        self.assertIn("visible example warning", stderr.getvalue())
        self.assertIn("Warp Error: visible warp error", stderr.getvalue())
        self.assertNotIn("visible warp error", stdout.getvalue())
        self.assertNotIn("Remove fixed joint child_fixed", stderr.getvalue())
        self.assertNotIn("visible example warning", stdout.getvalue())
        self.assertIn("captured example warning", stderr.getvalue())

    def test_examples_respect_preconfigured_warp_level(self):
        """Verify example logging setup honors a preconfigured warp logger level."""
        self._root_logger.handlers[:] = []
        self._warp_logger.handlers[:] = []
        self._warp_logger.setLevel(logging.WARNING)
        configure_logging = getattr(newton_examples, "_configure_logging", None)
        self.assertTrue(callable(configure_logging))
        args = newton_examples.create_parser().parse_args(["--viewer", "null"])

        stdout = io.StringIO()
        stderr = io.StringIO()
        with warnings.catch_warnings():
            warnings.filterwarnings("always", message="visible configured warp warning", category=UserWarning)
            with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                configure_logging(args)
                wp.get_logger().info("hidden configured warp info")
                wp.get_logger().warning("visible configured warp warning")

        self.assertNotIn("hidden configured warp info", stdout.getvalue())
        self.assertNotIn("hidden configured warp info", stderr.getvalue())
        self.assertIn("visible configured warp warning", stderr.getvalue())

    def test_examples_respect_preconfigured_newton_level(self):
        """Verify example logging setup honors a preconfigured newton logger level."""
        self._root_logger.handlers[:] = []
        self._newton_logger.handlers[:] = []
        self._newton_logger.setLevel(logging.WARNING)
        configure_logging = getattr(newton_examples, "_configure_logging", None)
        self.assertTrue(callable(configure_logging))
        args = newton_examples.create_parser().parse_args(["--viewer", "null"])

        stdout = io.StringIO()
        stderr = io.StringIO()
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            configure_logging(args)
            logging.getLogger("newton.test").info("hidden configured newton info")
            logging.getLogger("newton.test").warning("visible configured newton warning")

        self.assertNotIn("hidden configured newton info", stdout.getvalue())
        self.assertNotIn("hidden configured newton info", stderr.getvalue())
        self.assertIn("visible configured newton warning", stderr.getvalue())
        self.assertNotIn("visible configured newton warning", stdout.getvalue())

    def test_test_runner_configures_stdlib_logging_and_warning_capture(self):
        """Verify the parallel test runner installs stdout/stderr logging handlers."""
        self._root_logger.handlers[:] = []
        configure_logging = getattr(unittest_parallel, "_configure_logging", None)
        self.assertTrue(callable(configure_logging))

        stdout = io.StringIO()
        stderr = io.StringIO()
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            configure_logging(verbose=True)
            logging.getLogger("newton.test").info("visible test info")
            logging.getLogger("newton.test").warning("visible test warning")
            logging.getLogger("other.test").info("hidden third-party info")
            self._build_fixed_joint_model().collapse_fixed_joints(verbose=True)
            warnings.warn("captured test warning", UserWarning, stacklevel=1)

        self.assertTrue(self._root_logger.handlers)
        self.assertIn("visible test info", stdout.getvalue())
        self.assertIn("Remove fixed joint child_fixed", stdout.getvalue())
        self.assertNotIn("hidden third-party info", stdout.getvalue())
        self.assertNotIn("hidden third-party info", stderr.getvalue())
        self.assertNotIn("visible test info", stderr.getvalue())
        self.assertIn("visible test warning", stderr.getvalue())
        self.assertNotIn("Remove fixed joint child_fixed", stderr.getvalue())
        self.assertNotIn("visible test warning", stdout.getvalue())
        self.assertIn("captured test warning", stderr.getvalue())


class TestVerboseRouting(unittest.TestCase):
    def test_production_verbose_paths_do_not_print_directly(self):
        """Verify no verbose-gated production code calls print() instead of log_verbose()."""
        root = Path(newton.__file__).resolve().parent / "_src"
        offenders: list[str] = []

        for path in root.rglob("*.py"):
            if "/tests/" in path.as_posix():
                continue
            source = path.read_text(encoding="utf-8")
            tree = ast.parse(source, filename=str(path))
            parents = {}
            for parent in ast.walk(tree):
                for child in ast.iter_child_nodes(parent):
                    parents[child] = parent

            for node in ast.walk(tree):
                if not isinstance(node, ast.Call):
                    continue
                if not isinstance(node.func, ast.Name) or node.func.id != "print":
                    continue
                current = node
                while current in parents:
                    current = parents[current]
                    if isinstance(current, ast.If) and "verbose" in ast.unparse(current.test):
                        offenders.append(f"{path.relative_to(root.parent)}:{node.lineno}")
                        break

        self.assertEqual(offenders, [])


class TestFailfastProxyWarnings(LoggingStateMixin, unittest.TestCase):
    class _PassingTest(unittest.TestCase):
        def runTest(self):
            pass

    class _FailingTest(unittest.TestCase):
        def runTest(self):
            self.fail("trigger failfast")

    class _FailfastProxy:
        def __init__(self, *, fail_is_set: bool = False, fail_set: bool = False):
            self._fail_is_set = fail_is_set
            self._fail_set = fail_set

        def is_set(self):
            if self._fail_is_set:
                raise OSError("closed")
            return False

        def set(self):
            if self._fail_set:
                raise OSError("closed")

    def _manager(self, failfast):
        manager = unittest_parallel.ParallelTestManager.__new__(unittest_parallel.ParallelTestManager)
        manager.args = SimpleNamespace(
            buffer=False,
            coverage=False,
            coverage_branch=False,
            failfast=True,
            junit_report_xml=None,
            strict_warnings=False,
            verbose=0,
            warp_config=[],
        )
        manager.temp_dir = tempfile.gettempdir()
        manager.failfast = failfast
        return manager

    def test_failfast_is_set_proxy_error_still_runs_tests(self):
        """Verify a failing failfast is_set() proxy logs a warning and tests still run."""
        suite = unittest.TestSuite([self._PassingTest()])
        manager = self._manager(self._FailfastProxy(fail_is_set=True))

        with self.assertLogs("newton.tests.thirdparty.unittest_parallel", level="WARNING") as logs:
            result = manager.run_tests(suite)

        self.assertEqual(result[0], 1)
        self.assertEqual(result[1], [])
        self.assertEqual(result[2], [])
        self.assertTrue(any("failfast proxy is_set() failed (OSError)" in line for line in logs.output))

    def test_failfast_set_proxy_error_still_returns_failures(self):
        """Verify a failing failfast set() proxy logs a warning and failures are reported."""
        suite = unittest.TestSuite([self._FailingTest()])
        manager = self._manager(self._FailfastProxy(fail_set=True))

        with self.assertLogs("newton.tests.thirdparty.unittest_parallel", level="WARNING") as logs:
            result = manager.run_tests(suite)

        self.assertEqual(result[0], 1)
        self.assertEqual(result[1], [])
        self.assertEqual(len(result[2]), 1)
        self.assertTrue(any("failfast proxy set() failed (OSError)" in line for line in logs.output))


if __name__ == "__main__":
    unittest.main(verbosity=2)
