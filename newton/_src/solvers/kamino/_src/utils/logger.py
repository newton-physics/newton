# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""
KAMINO: Utilities: Console Logging

Kamino modules log through ``logging.getLogger(__name__)`` like the rest of Newton.
The helpers below configure logging and are meant for applications (examples,
tests, benchmarks), not for library code.
"""

import logging
import os
import sys
from typing import ClassVar

from .....diagnostics import _ConsoleFallbackHandler

###
# Module interface
###

__all__ = [
    "LOGGER_NAMES",
    "ConsoleFormatter",
    "enable_console_logging",
    "reset_log_level",
    "set_log_level",
]


###
# Globals
###

LOGGER_NAMES = ("newton._src.solvers.kamino", "newton.tests.kamino")
"""Parent loggers of the Kamino solver and its tests."""


###
# Formatting
###


class ConsoleFormatter(logging.Formatter):
    """Formatter adding a ``[KAMINO]`` header and level-based colors."""

    HEADER = "[KAMINO]"
    HEADER_COLOR = "\x1b[38;5;13m"
    RESET = "\x1b[0m"

    LINE_FORMAT = "[%(asctime)s][%(filename)s:%(lineno)d][%(levelname)s]: %(message)s"
    """Line format for the log messages, including timestamp, filename, line number, log level, and message."""

    LEVEL_COLORS: ClassVar[dict[int, str]] = {
        logging.DEBUG: "\x1b[90m",
        logging.INFO: "\x1b[32;20m",
        logging.WARNING: "\x1b[33;20m",
        logging.ERROR: "\x1b[31;20m",
    }
    """Message colors, applied to all records at or above each level."""

    def __init__(self, color: bool = True):
        super().__init__(self.LINE_FORMAT)
        self.color = color

    def format(self, record: logging.LogRecord) -> str:
        line = super().format(record)
        if not self.color:
            return self.HEADER + line
        level = max((lvl for lvl in self.LEVEL_COLORS if lvl <= record.levelno), default=logging.DEBUG)
        return self.HEADER_COLOR + self.HEADER + self.RESET + self.LEVEL_COLORS[level] + line + self.RESET


class _ConsoleHandler(logging.Handler):
    """Print Kamino records until the application configures a handler of its own."""

    def emit(self, record: logging.LogRecord) -> None:
        if _application_handler_configured(record.name):
            return
        try:
            # Resolve the stream per record so redirections of stdout/stderr apply.
            stream = sys.stderr if record.levelno >= logging.WARNING else sys.stdout
            if stream is None:
                return
            stream.write(self.format(record) + "\n")
            stream.flush()
        except RecursionError:
            raise
        except Exception:
            self.handleError(record)


def _application_handler_configured(name: str) -> bool:
    # NOTE: Mirrors the ``newton`` console fallback, so test-time handlers such as ``assertLogs`` take over.
    logger = logging.Logger.manager.loggerDict.get(name)
    while isinstance(logger, logging.Logger):
        if any(not isinstance(handler, (_ConsoleHandler, _ConsoleFallbackHandler)) for handler in logger.handlers):
            return True
        if not logger.propagate:
            return False
        logger = logger.parent
    return False


def _supports_color() -> bool:
    if os.environ.get("NO_COLOR"):
        return False
    stream = sys.stdout
    return stream is not None and hasattr(stream, "isatty") and stream.isatty()


def _enable_windows_ansi_sequences():
    """Make legacy Windows consoles interpret ANSI escape sequences."""
    if sys.platform != "win32":
        return
    import ctypes  # noqa: PLC0415

    kernel32 = ctypes.windll.kernel32
    invalid_handle = ctypes.c_void_p(-1).value
    for std_handle in (-11, -12):  # STD_OUTPUT_HANDLE, STD_ERROR_HANDLE
        try:
            handle = kernel32.GetStdHandle(std_handle)
            if handle == 0 or ctypes.c_void_p(handle).value == invalid_handle:
                continue
            mode = ctypes.c_uint()
            if kernel32.GetConsoleMode(handle, ctypes.byref(mode)):
                kernel32.SetConsoleMode(handle, mode.value | 0x0004)  # ENABLE_VIRTUAL_TERMINAL_PROCESSING
        except Exception:
            # Getting/setting the console mode fails in some contexts (e.g., services, CI).
            pass


###
# Configurations
###


def enable_console_logging(color: bool | None = None, names: tuple[str, ...] = LOGGER_NAMES):
    """Print Kamino log records in the Kamino console format.

    Records still propagate to their parent loggers, and the console output
    yields to any handler the application configures. Calling this again only
    updates the formatting.

    Args:
        color: Whether to color the output. Defaults to coloring when stdout is
            a terminal and the ``NO_COLOR`` environment variable is not set.
        names: Loggers to attach the console output to.
    """
    if color is None:
        color = _supports_color()
    if color:
        _enable_windows_ansi_sequences()

    formatter = ConsoleFormatter(color=color)
    for name in names:
        logger = logging.getLogger(name)
        handler = next((h for h in logger.handlers if isinstance(h, _ConsoleHandler)), None)
        if handler is None:
            handler = _ConsoleHandler()
            logger.addHandler(handler)
        handler.setFormatter(formatter)


def set_log_level(level: int, names: tuple[str, ...] = LOGGER_NAMES):
    """Set the level of the given loggers, the Kamino loggers by default."""
    for name in names:
        logging.getLogger(name).setLevel(level)


def reset_log_level(names: tuple[str, ...] = LOGGER_NAMES):
    """Make the given loggers inherit their level from their parents again."""
    set_log_level(logging.NOTSET, names=names)
