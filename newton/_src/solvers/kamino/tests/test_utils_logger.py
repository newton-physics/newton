# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Kamino: Tests for logging utilities"""

import unittest

from newton._src.solvers.kamino._src.utils import logger as msg
from newton._src.solvers.kamino._src.utils.logger import Logger
from newton._src.solvers.kamino.tests import setup_tests, test_context

###
# Tests
###


class TestUtilsLogger(unittest.TestCase):
    def setUp(self):
        if not test_context.setup_done:
            setup_tests(clear_cache=False)

    def test_new_logger(self):
        """Test use of the custom logger."""
        print("")  # Print a newline for better readability in the output
        msg.set_log_level(msg.LogLevel.DEBUG)
        logger = Logger()
        log = logger.get()
        log.debug("This is a debug message.")
        log.info("This is an info message.")
        log.warning("This is a warning message.")
        log.error("This is an error message.")
        log.critical("This is a critical message.")
        msg.reset_log_level()

    def test_default_logger(self):
        """Test use of the custom logger."""
        print("")  # Print a newline for better readability in the output
        msg.set_log_level(msg.LogLevel.DEBUG)
        msg.debug("This is a debug message.")
        msg.info("This is an info message.")
        msg.notif("This is a notification message.")
        msg.warning("This is a warning message.")
        msg.error("This is an error message.")
        msg.critical("This is a critical message.")
        msg.reset_log_level()


###
# Test execution
###

if __name__ == "__main__":
    # Test setup
    setup_tests()

    # Run all tests
    unittest.main(verbosity=2)
