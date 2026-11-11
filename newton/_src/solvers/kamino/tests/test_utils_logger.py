# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Kamino: Tests for logging utilities"""

import logging
import unittest

from newton._src.solvers.kamino._src.utils.logger import LOGGER_NAMES, reset_log_level, set_log_level
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
        reset_log_level()

    def test_00_set_and_reset_log_level(self):
        """Set the level of all Kamino loggers, then make them inherit it again."""
        set_log_level(logging.DEBUG)
        for name in LOGGER_NAMES:
            self.assertEqual(logging.getLogger(name).level, logging.DEBUG)
        self.assertTrue(logger.isEnabledFor(logging.DEBUG))

        reset_log_level()
        for name in LOGGER_NAMES:
            self.assertEqual(logging.getLogger(name).level, logging.NOTSET)

    def test_01_records_reach_newton_logger(self):
        """Propagate Kamino records at every level to the ``newton`` logger."""
        set_log_level(logging.DEBUG)
        with self.assertLogs("newton", level=logging.DEBUG) as logs:
            logger.debug("This is a debug message.")
            logger.info("This is an info message.")
            logger.warning("This is a warning message.")
            logger.error("This is an error message.")
        self.assertEqual(
            [record.levelno for record in logs.records],
            [logging.DEBUG, logging.INFO, logging.WARNING, logging.ERROR],
        )


###
# Test execution
###

if __name__ == "__main__":
    # Test setup
    setup_tests()

    # Run all tests
    unittest.main(verbosity=2)
