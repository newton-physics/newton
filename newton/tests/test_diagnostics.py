# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import ast
import contextlib
import io
import logging
import pathlib
import unittest
import warnings

import newton
from newton.exceptions import NewtonDeprecationWarning, NewtonWarning

_NEWTON_ROOT = pathlib.Path(newton.__file__).parent


def _library_sources():
    for path in sorted(_NEWTON_ROOT.rglob("*.py")):
        relative = path.relative_to(_NEWTON_ROOT)
        if relative.parts[0] in ("tests", "examples") or "tests" in relative.parts or "examples" in relative.parts:
            continue
        yield relative, ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


class TestWarningCategories(unittest.TestCase):
    def test_deprecation_category_matches_builtin_filters(self):
        """Keep Newton deprecations filterable as both Newton and Python deprecation warnings."""
        self.assertTrue(issubclass(NewtonWarning, UserWarning))
        self.assertTrue(issubclass(NewtonDeprecationWarning, NewtonWarning))
        self.assertTrue(issubclass(NewtonDeprecationWarning, DeprecationWarning))

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            warnings.filterwarnings("ignore", category=NewtonWarning)
            warnings.warn("ignored", NewtonDeprecationWarning, stacklevel=1)
        self.assertEqual(caught, [])

    def test_library_warnings_use_newton_categories(self):
        """Require an explicit, non-generic category for every warning Newton emits."""
        generic = {None, "UserWarning", "DeprecationWarning", "Warning"}
        offenders = []
        for relative, tree in _library_sources():
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call) or ast.unparse(node.func) not in ("warnings.warn", "warn"):
                    continue
                category = node.args[1] if len(node.args) > 1 else None
                category = next((k.value for k in node.keywords if k.arg == "category"), category)
                if (ast.unparse(category) if category is not None else None) in generic:
                    offenders.append(f"{relative}:{node.lineno}")
        self.assertEqual(offenders, [], "use a newton.exceptions warning category")


class _RecordingHandler(logging.Handler):
    def __init__(self):
        super().__init__()
        self.records = []

    def emit(self, record):
        self.records.append(record)


class TestLoggingDefaults(unittest.TestCase):
    def setUp(self):
        # Isolate from root handlers that the test runner or other tests may install.
        root = logging.getLogger()
        saved_handlers = root.handlers[:]
        root.handlers.clear()
        self.addCleanup(setattr, root, "handlers", saved_handlers)
        self.logger = logging.getLogger("newton.test_diagnostics")

    def _capture(self, *messages):
        stdout, stderr = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            for level, message in messages:
                self.logger.log(level, message)
        return stdout.getvalue(), stderr.getvalue()

    def test_unconfigured_logging_prints_like_before(self):
        """Print INFO records to stdout and prefixed warnings to stderr without logging configuration."""
        self.assertEqual(logging.getLogger("newton").level, logging.INFO)
        stdout, stderr = self._capture(
            (logging.DEBUG, "detail"), (logging.INFO, "loaded asset"), (logging.WARNING, "ignored option")
        )
        self.assertEqual(stdout, "loaded asset\n")
        self.assertEqual(stderr, "Warning: ignored option\n")

    def test_configured_handler_replaces_console_fallback(self):
        """Send records only to application handlers once the application configures logging."""
        handler = _RecordingHandler()
        logging.getLogger().addHandler(handler)

        stdout, stderr = self._capture((logging.INFO, "loaded asset"))

        self.assertEqual((stdout, stderr), ("", ""))
        self.assertEqual([record.getMessage() for record in handler.records], ["loaded asset"])

    def test_handler_on_child_logger_replaces_console_fallback(self):
        """Treat a handler on any logger between the record's logger and the root as configuration."""
        handler = _RecordingHandler()
        self.logger.addHandler(handler)
        self.addCleanup(self.logger.removeHandler, handler)

        stdout, stderr = self._capture((logging.WARNING, "ignored option"))

        self.assertEqual((stdout, stderr), ("", ""))
        self.assertEqual(len(handler.records), 1)

    def test_verbose_diagnostics_print_without_configuration(self):
        """Keep ``verbose=True`` importer output on stdout when the application has not configured logging."""
        mjcf = '<mujoco><worldbody><body name="b"><geom type="box" size="0.1 0.1 0.1"/></body></worldbody></mujoco>'
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            newton.ModelBuilder().add_mjcf(mjcf, verbose=True)
        self.assertIn("no class defined for geom", stdout.getvalue())


if __name__ == "__main__":
    unittest.main(verbosity=2)
