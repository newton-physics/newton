# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""
KAMINO: Utilities: Log Levels

Kamino modules log through ``logging.getLogger(__name__)`` like the rest of Newton.
The helpers below adjust the level of the Kamino loggers and are meant for
applications (examples, tests, benchmarks), not for library code.
"""

import logging

###
# Module interface
###

__all__ = [
    "LOGGER_NAMES",
    "reset_log_level",
    "set_log_level",
]


###
# Globals
###

LOGGER_NAMES = ("newton._src.solvers.kamino", "newton.tests.kamino")
"""Parent loggers of the Kamino solver and its tests."""

_SCRIPT_LOGGER_NAME = "__main__"
"""Logger of the module run as a script, such as a test file run directly."""


###
# Configurations
###


def set_log_level(level: int):
    """Set the level of the Kamino loggers and of the script logger."""
    for name in (*LOGGER_NAMES, _SCRIPT_LOGGER_NAME):
        logging.getLogger(name).setLevel(level)


def reset_log_level():
    """Make the Kamino loggers inherit their level from the ``newton`` logger again."""
    for name in LOGGER_NAMES:
        logging.getLogger(name).setLevel(logging.NOTSET)

    # The script logger sits outside the ``newton`` tree, so give it the ``newton`` level instead
    logging.getLogger(_SCRIPT_LOGGER_NAME).setLevel(logging.getLogger("newton").getEffectiveLevel())
