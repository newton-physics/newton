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


###
# Configurations
###


def set_log_level(level: int):
    """Set the level of the Kamino loggers."""
    for name in LOGGER_NAMES:
        logging.getLogger(name).setLevel(level)


def reset_log_level():
    """Make the Kamino loggers inherit their level from the ``newton`` logger again."""
    set_log_level(logging.NOTSET)
