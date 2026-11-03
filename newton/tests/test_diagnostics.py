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
import newton.utils
from newton._src import diagnostics
from newton.exceptions import NewtonDeprecationWarning, NewtonWarning

_NEWTON_ROOT = pathlib.Path(newton.__file__).parent


def _library_sources():
    for path in sorted(_NEWTON_ROOT.rglob("*.py")):
        relative = path.relative_to(_NEWTON_ROOT)
        if relative.parts[0] in ("tests", "examples") or "tests" in relative.parts or "examples" in relative.parts:
            continue
        yield relative, ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


class TestWarningCategories(unittest.TestCase):
    def test_categories_match_the_same_builtin_filters(self):
        """Keep UserWarning filters from changing how Newton deprecations are reported."""
        self.assertTrue(issubclass(NewtonWarning, UserWarning))
        self.assertTrue(issubclass(NewtonDeprecationWarning, DeprecationWarning))

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            warnings.filterwarnings("error", category=UserWarning)
            warnings.warn("reported", NewtonDeprecationWarning, stacklevel=1)
        self.assertEqual([type(w.message) for w in caught], [NewtonDeprecationWarning])

    def test_library_warnings_use_newton_categories(self):
        """Require a newton.exceptions category for every warning Newton emits."""
        allowed = set(newton.exceptions.__all__)
        offenders = []
        for relative, tree in _library_sources():
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call) or ast.unparse(node.func) not in ("warnings.warn", "warn"):
                    continue
                category = node.args[1] if len(node.args) > 1 else None
                category = next((k.value for k in node.keywords if k.arg == "category"), category)
                if category is None or ast.unparse(category).rpartition(".")[2] not in allowed:
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

        # Start from the plain console format, which the test runner may have replaced.
        self.addCleanup(setattr, diagnostics, "_console_formatter", diagnostics._console_formatter)
        diagnostics._console_formatter = diagnostics._PlainFormatter()

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

    def test_kamino_messages_follow_newton_logger(self):
        """Route Kamino messages through the newton logger and its level without configuring logging."""
        from newton._src.solvers.kamino._src.utils.logger import reset_log_level  # noqa: PLC0415

        reset_log_level()  # undo a level an earlier test may have set
        newton_logger = logging.getLogger("newton")
        self.addCleanup(newton_logger.setLevel, newton_logger.level)

        logger = logging.getLogger("newton._src.solvers.kamino._src.core.model")
        stdout, stderr = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            logger.warning("kamino problem")
            logger.debug("kamino detail")
            logger.info("kamino status")
            newton_logger.setLevel(logging.DEBUG)
            logger.debug("kamino detail on request")

        self.assertEqual(logging.getLogger().handlers, [])
        self.assertEqual(stdout.getvalue(), "kamino status\nkamino detail on request\n")
        self.assertEqual(stderr.getvalue(), "Warning: kamino problem\n")

    def test_verbose_diagnostics_print_without_configuration(self):
        """Keep ``verbose=True`` importer output on stdout when the application has not configured logging."""
        mjcf = '<mujoco><worldbody><body name="b"><geom type="box" size="0.1 0.1 0.1"/></body></worldbody></mujoco>'
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            newton.ModelBuilder().add_mjcf(mjcf, verbose=True)
        self.assertIn("no class defined for geom", stdout.getvalue())

    def test_rich_console_prints_time_location_and_level(self):
        """Prefix each record with its time, source location, and level name once rich output is enabled."""
        newton.utils.enable_rich_console(color=False)
        stdout, stderr = self._capture((logging.INFO, "loaded asset"), (logging.WARNING, "ignored option"))

        prefix = r"^\[\d{4}-\d\d-\d\d \d\d:\d\d:\d\d,\d{3}\]\[test_diagnostics\.py:\d+\]"
        self.assertRegex(stdout, prefix + r"\[INFO\]: loaded asset\n$")
        self.assertRegex(stderr, prefix + r"\[WARNING\]: ignored option\n$")

    def test_rich_console_colors_by_level(self):
        """Wrap each line in the color of its level when coloring is requested."""
        newton.utils.enable_rich_console(color=True)
        stdout, stderr = self._capture((logging.INFO, "loaded asset"), (logging.ERROR, "failed"))

        self.assertTrue(stdout.startswith("\x1b[32;20m["))
        self.assertTrue(stderr.startswith("\x1b[31;20m["))
        self.assertTrue(stdout.endswith("\x1b[0m\n") and stderr.endswith("\x1b[0m\n"))

    def test_setup_console_logging_switches_back_to_plain_format(self):
        """Restore the plain console format when rich output is disabled."""
        newton.utils.setup_console_logging(color=False)
        newton.utils.setup_console_logging(rich=False)
        stdout, stderr = self._capture((logging.INFO, "loaded asset"), (logging.WARNING, "ignored option"))

        self.assertEqual(stdout, "loaded asset\n")
        self.assertEqual(stderr, "Warning: ignored option\n")

    def test_script_logging_prints_main_records(self):
        """Print INFO records of the ``__main__`` logger once script logging is enabled."""
        main_logger = logging.getLogger("__main__")
        self.addCleanup(setattr, main_logger, "handlers", main_logger.handlers[:])
        self.addCleanup(main_logger.setLevel, main_logger.level)
        main_logger.handlers = [
            h for h in main_logger.handlers if not isinstance(h, diagnostics._ConsoleFallbackHandler)
        ]
        main_logger.setLevel(logging.NOTSET)

        newton.utils.enable_script_logging()
        newton.utils.enable_script_logging()
        self.logger = main_logger
        stdout, _ = self._capture((logging.DEBUG, "detail"), (logging.INFO, "script status"))

        self.assertEqual(stdout, "script status\n")
        self.assertEqual(len(main_logger.handlers), 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
