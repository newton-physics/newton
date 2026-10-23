# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Kamino: Tests for logging utilities"""

import contextlib
import io
import logging
import unittest

from newton._src.solvers.kamino._src.utils.logger import (
    LOGGER_NAMES,
    ConsoleFormatter,
    enable_console_logging,
    reset_log_level,
    set_log_level,
)
from newton._src.solvers.kamino.tests import setup_tests, test_context

logger = logging.getLogger(__name__)

###
# Tests
###


class TestUtilsLogger(unittest.TestCase):
    def setUp(self):
        if not test_context.setup_done:
            setup_tests(clear_cache=False)

    def tearDown(self):
        enable_console_logging()
        reset_log_level()

    def test_00_console_levels(self):
        """Print one message per level with the colored console format."""
        print("")  # Print a newline for better readability in the output
        enable_console_logging(color=True)
        set_log_level(logging.DEBUG)
        logger.debug("This is a debug message.")
        logger.info("This is an info message.")
        logger.warning("This is a warning message.")
        logger.error("This is an error message.")

    def test_01_formatter(self):
        """Prefix records with the Kamino header and color them by level."""
        record = logging.LogRecord(logger.name, logging.WARNING, __file__, 1, "value: %s", (42,), None)
        plain = ConsoleFormatter(color=False).format(record)
        self.assertTrue(plain.startswith("[KAMINO]["))
        self.assertTrue(plain.endswith("[WARNING]: value: 42"))
        colored = ConsoleFormatter(color=True).format(record)
        self.assertIn(ConsoleFormatter.LEVEL_COLORS[logging.WARNING] + plain[len(ConsoleFormatter.HEADER) :], colored)

    def test_02_enable_is_idempotent(self):
        """Attach a single console handler per Kamino logger."""
        enable_console_logging(color=False)
        handlers = {name: list(logging.getLogger(name).handlers) for name in LOGGER_NAMES}
        enable_console_logging(color=False)
        for name in LOGGER_NAMES:
            self.assertEqual(logging.getLogger(name).handlers, handlers[name])
            self.assertEqual(len(handlers[name]), 1)

    def test_03_console_yields_to_application_handlers(self):
        """Leave records to handlers configured by the application, e.g. ``assertLogs``."""
        enable_console_logging(color=False)
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr), self.assertLogs("newton", level="WARNING") as logs:
            logger.warning("captured")
        self.assertEqual([record.getMessage() for record in logs.records], ["captured"])
        self.assertEqual(stderr.getvalue(), "")


###
# Test execution
###

if __name__ == "__main__":
    # Test setup
    setup_tests()

    # Run all tests
    unittest.main(verbosity=2)
