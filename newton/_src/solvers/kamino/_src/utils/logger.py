# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
KAMINO: Utilities: Message Logging
"""

import logging
from enum import IntEnum
from typing import ClassVar


class LogLevel(IntEnum):
    """Enumeration for log levels."""

    DEBUG = logging.DEBUG
    INFO = logging.INFO
    NOTIF = logging.INFO + 5
    WARNING = logging.WARNING
    ERROR = logging.ERROR
    CRITICAL = logging.CRITICAL


class Logger(logging.Formatter):
    """Base logger with color highlighting for log levels."""

    HEADER = "[KAMINO]"
    HEADERCOL = "\x1b[38;5;13m"

    WHITE = "\x1b[37m"
    RED = "\x1b[31;20m"
    BOLD_RED = "\x1b[31;1m"
    BLUE = "\x1b[34;20m"
    BOLD_BLUE = "\x1b[34;1m"
    GREEN = "\x1b[32;20m"
    BOLD_GREEN = "\x1b[32;1m"
    YELLOW = "\x1b[33;20m"
    BOLD_YELLOW = "\x1b[33;1m"
    RESET = "\x1b[0m"

    LINE_FORMAT = "[%(asctime)s][%(filename)s:%(lineno)d][%(levelname)s]: %(message)s"
    """Line format for the log messages, including timestamp, filename, line number, log level, and message."""

    FORMATS: ClassVar[dict[int, str]] = {
        LogLevel.DEBUG: HEADERCOL + HEADER + RESET + BLUE + LINE_FORMAT + RESET,
        LogLevel.INFO: HEADERCOL + HEADER + RESET + WHITE + LINE_FORMAT + RESET,
        LogLevel.NOTIF: HEADERCOL + HEADER + RESET + GREEN + LINE_FORMAT + RESET,
        LogLevel.WARNING: HEADERCOL + HEADER + RESET + YELLOW + LINE_FORMAT + RESET,
        LogLevel.ERROR: HEADERCOL + HEADER + RESET + RED + LINE_FORMAT + RESET,
        LogLevel.CRITICAL: HEADERCOL + HEADER + RESET + BOLD_RED + LINE_FORMAT + RESET,
    }
    """Dictionary mapping log levels to their respective formats."""

    def __init__(self):
        """Initialize the Logger with a stream handler and set the default logging level."""
        super().__init__()

        # Create a stream handler with the custom logger format.
        self._streamhandler = logging.StreamHandler()
        self._streamhandler.setFormatter(self)

        # Add custom level NOTIF to the logging module
        logging.addLevelName(LogLevel.NOTIF, "NOTIF")

        # Set the default logging level to DEBUG
        logging.basicConfig(
            handlers=[self._streamhandler],
            level=LogLevel.NOTIF,  # Default level set to NOTIF
        )

    def format(self, record):
        """Format the log record with the appropriate color based on the log level."""
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)

    def get(self):
        """Get the global logger instance."""
        return logging.getLogger()

    def notif(self, msg, *args, **kwargs):
        """
        Log 'msg % args' with severity 'NOTIF'.

        To pass exception information, use the keyword argument exc_info with
        a true value, e.g.

        logger.notif("Houston, we have a %s", "thorny problem", exc_info=1)
        """
        self.get().log(LogLevel.NOTIF, msg, *args, **kwargs)


###
# Globals
###


LOGGER: Logger | None = None
"""Global logger instance for the application."""


###
# Configurations
###


def get_default_logger() -> logging.Logger:
    """Initialize the global logger instance."""
    global LOGGER  # noqa: PLW0603
    if LOGGER is None:
        LOGGER = Logger()
    return LOGGER.get()


def set_log_level(level: LogLevel):
    """Set the logging level for the default logger."""
    get_default_logger().setLevel(level)
    get_default_logger().debug(f"Log level set to: {logging.getLevelName(level)}")


def reset_log_level():
    """Reset the logging level for the default logger to WARNING."""
    get_default_logger().setLevel(LogLevel.NOTIF)
    get_default_logger().debug(f"Log level reset to: {logging.getLevelName(LogLevel.NOTIF)}")


def set_log_header(header: str):
    """Set the header for the logger."""
    Logger.HEADER = header


###
# Logging
###


def debug(msg: str, *args, **kwargs):
    """Log a debug message."""
    get_default_logger().debug(msg, *args, **kwargs, stacklevel=2)


def info(msg: str, *args, **kwargs):
    """Log an info message."""
    get_default_logger().info(msg, *args, **kwargs, stacklevel=2)


def notif(msg: str, *args, **kwargs):
    """Log a notification message."""
    get_default_logger().log(LogLevel.NOTIF, msg, *args, **kwargs, stacklevel=2)


def warning(msg: str, *args, **kwargs):
    """Log a warning message."""
    get_default_logger().warning(msg, *args, **kwargs, stacklevel=2)


def error(msg: str, *args, **kwargs):
    """Log an error message."""
    get_default_logger().error(msg, *args, **kwargs, stacklevel=2)


def critical(msg: str, *args, **kwargs):
    """Log a critical message."""
    get_default_logger().critical(msg, *args, **kwargs, stacklevel=2)
