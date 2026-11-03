# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import logging
import os
import sys
from typing import ClassVar

_LEVEL_PREFIXES = {logging.WARNING: "Warning: ", logging.ERROR: "Error: ", logging.CRITICAL: "Error: "}


class _PlainFormatter(logging.Formatter):
    """Print the message alone, prefixing warnings and errors with their level."""

    def format(self, record: logging.LogRecord) -> str:
        return _LEVEL_PREFIXES.get(record.levelno, "") + super().format(record)


def _console_stream(record: logging.LogRecord):
    """Return the stream a console record is printed to, resolved per record like ``print()``."""
    return sys.stderr if record.levelno >= logging.WARNING else sys.stdout


def _supports_color(stream) -> bool:
    if os.environ.get("NO_COLOR"):
        return False
    return stream is not None and hasattr(stream, "isatty") and stream.isatty()


class _RichFormatter(logging.Formatter):
    """Print the time, source location, and level of each message, optionally colored by level."""

    LINE_FORMAT = "[%(asctime)s][%(filename)s:%(lineno)d][%(levelname)s]: %(message)s"

    LEVEL_COLORS: ClassVar[dict[int, str]] = {
        logging.DEBUG: "\x1b[90m",
        logging.INFO: "\x1b[32m",
        logging.WARNING: "\x1b[33m",
        logging.ERROR: "\x1b[31m",
    }
    """Line colors, applied to all records at or above each level."""

    RESET = "\x1b[0m"

    def __init__(self, color: bool | None):
        super().__init__(self.LINE_FORMAT)
        self.color = color

    def format(self, record: logging.LogRecord) -> str:
        line = super().format(record)
        color = _supports_color(_console_stream(record)) if self.color is None else self.color
        if not color:
            return line
        level = max((lvl for lvl in self.LEVEL_COLORS if lvl <= record.levelno), default=logging.DEBUG)
        return self.LEVEL_COLORS[level] + line + self.RESET


_console_formatter: logging.Formatter = _PlainFormatter()
"""Formatter of every console fallback handler."""


class _ConsoleFallbackHandler(logging.Handler):
    """Print log records until the application configures a handler.

    Replaces :data:`logging.lastResort`, which drops ``INFO`` records, so that
    output Newton used to ``print()`` stays visible by default.
    """

    def format(self, record: logging.LogRecord) -> str:
        return (self.formatter or _console_formatter).format(record)

    def emit(self, record: logging.LogRecord) -> None:
        if self._other_handler_configured(record.name):
            return
        try:
            # Resolve the stream per record so contextlib.redirect_stdout() captures it like print().
            stream = _console_stream(record)
            if stream is None:  # pythonw and similar hosts have no console
                return
            stream.write(self.format(record) + "\n")
            stream.flush()
        except RecursionError:
            raise
        except Exception:
            self.handleError(record)

    def _other_handler_configured(self, name: str) -> bool:
        # logging.getLogger() takes the module lock, which must not be acquired under the handler lock.
        logger = logging.Logger.manager.loggerDict.get(name)
        while isinstance(logger, logging.Logger):
            if any(handler is not self for handler in logger.handlers):
                return True
            if not logger.propagate:
                return False
            logger = logger.parent
        return False


def _install_console_fallback(name: str) -> None:
    logger = logging.getLogger(name)
    if logger.level == logging.NOTSET:
        logger.setLevel(logging.INFO)
    if not any(isinstance(handler, _ConsoleFallbackHandler) for handler in logger.handlers):
        logger.addHandler(_ConsoleFallbackHandler())


def install_console_fallback() -> None:
    """Make ``newton`` INFO records visible without application logging configuration."""
    _install_console_fallback("newton")


def _enable_windows_ansi_sequences() -> bool:
    """Make legacy Windows consoles interpret ANSI escape sequences.

    Returns:
        Whether every console attached to stdout and stderr interprets them.
    """
    if sys.platform != "win32":
        return True
    import ctypes  # noqa: PLC0415

    kernel32 = ctypes.windll.kernel32
    invalid_handle = ctypes.c_void_p(-1).value
    supported = True
    for std_handle in (-11, -12):  # STD_OUTPUT_HANDLE, STD_ERROR_HANDLE
        try:
            handle = kernel32.GetStdHandle(std_handle)
            if handle == 0 or ctypes.c_void_p(handle).value == invalid_handle:
                continue
            mode = ctypes.c_uint()
            # Redirected streams are not consoles and need no console mode
            if kernel32.GetConsoleMode(handle, ctypes.byref(mode)):
                if not kernel32.SetConsoleMode(handle, mode.value | 0x0004):  # ENABLE_VIRTUAL_TERMINAL_PROCESSING
                    supported = False
        except Exception:
            # Getting/setting the console mode fails in some contexts (e.g., services, CI).
            supported = False
    return supported


def enable_rich_console(*, color: bool | None = None) -> None:
    """Print the time, source location, and level of each message in Newton's default console output.

    This only changes the output printed until the application configures
    logging; handlers configured by the application are unaffected.

    Args:
        color: Whether to color messages by level. Defaults to coloring the
            messages printed to a terminal, unless the ``NO_COLOR``
            environment variable is set or the terminal cannot interpret
            color sequences.
    """
    global _console_formatter  # noqa: PLW0603
    if color is not False and not _enable_windows_ansi_sequences() and color is None:
        color = False
    _console_formatter = _RichFormatter(color=color)


def enable_script_logging() -> None:
    """Print the records of the running script's module logger like ``newton`` records.

    A file run as a script logs through the ``__main__`` logger, which sits
    outside the ``newton`` logger tree, so its ``INFO`` records are otherwise
    dropped. Call this from the script's ``if __name__ == "__main__":`` block;
    the output yields to any handler the application configures.

    Sets the ``__main__`` logger to ``INFO`` unless it already has a level.
    Like any module logger, its level is independent of the ``newton``
    logger's level.
    """
    _install_console_fallback("__main__")


def setup_console_logging(*, rich: bool = True, color: bool | None = None) -> None:
    """Set up console logging for an entry point, such as an example or a test runner.

    Combines :func:`enable_script_logging` and :func:`enable_rich_console`.
    Can be called again to switch between the rich and plain formats.

    Args:
        rich: Whether to print the time, source location, and level of each
            message. If ``False``, print messages alone, prefixing warnings and
            errors with their level.
        color: Whether to color rich messages by level. Defaults to coloring
            the messages printed to a terminal, unless the ``NO_COLOR``
            environment variable is set.
    """
    global _console_formatter  # noqa: PLW0603
    enable_script_logging()
    if rich:
        enable_rich_console(color=color)
    else:
        _console_formatter = _PlainFormatter()
