# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Internal diagnostic routing helpers."""

from __future__ import annotations

import logging
import sys
import warnings

from newton.exceptions import NewtonDeprecationWarning

_legacy_verbose_stdout_warning_keys: set[str] = set()

_ENTRY_POINT_STDOUT_HANDLER = "_newton_entry_point_stdout_handler"


class _MaxLevelFilter(logging.Filter):
    def __init__(self, exclusive_maximum: int):
        super().__init__()
        self._exclusive_maximum = exclusive_maximum

    def filter(self, record: logging.LogRecord) -> bool:
        return record.levelno < self._exclusive_maximum


def _has_enabled_handler(logger: logging.Logger, level: int) -> bool:
    current: logging.Logger | None = logger
    while current is not None:
        if any(level >= handler.level for handler in current.handlers):
            return True
        if not current.propagate:
            return False
        current = current.parent
    return logging.lastResort is not None and level >= logging.lastResort.level


def _entry_point_stderr_handler() -> logging.Handler:
    formatter = logging.Formatter("%(message)s")
    stderr_handler = logging.StreamHandler()
    stderr_handler.setLevel(logging.WARNING)
    stderr_handler.setFormatter(formatter)
    return stderr_handler


def _remove_marked_handlers(logger: logging.Logger, marker: str = _ENTRY_POINT_STDOUT_HANDLER) -> None:
    logger.handlers[:] = [handler for handler in logger.handlers if not getattr(handler, marker, False)]


def _install_below_warning_stdout_handler(
    logger: logging.Logger,
    *,
    enabled_level: int = logging.INFO,
    handler_level: int = logging.INFO,
    logger_level: int | None = None,
    force_level: bool = False,
    marker: str = _ENTRY_POINT_STDOUT_HANDLER,
) -> logging.Handler | None:
    """Install a marked stdout handler for records below WARNING on entry-point loggers.

    Respects preconfigured logger levels and handlers unless ``force_level`` is set.
    """
    _remove_marked_handlers(logger, marker)

    target_logger_level = logger_level if logger_level is not None else enabled_level
    if force_level:
        logger.setLevel(target_logger_level)
    elif logger.level == logging.NOTSET and not logger.isEnabledFor(enabled_level):
        logger.setLevel(target_logger_level)

    if not logger.isEnabledFor(enabled_level) or _has_enabled_handler(logger, enabled_level):
        return None

    formatter = logging.Formatter("%(message)s")
    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setLevel(handler_level)
    stdout_handler.addFilter(_MaxLevelFilter(logging.WARNING))
    stdout_handler.setFormatter(formatter)
    setattr(stdout_handler, marker, True)
    logger.addHandler(stdout_handler)
    return stdout_handler


def log_verbose(logger: logging.Logger, *values: object, sep: str = " ", end: str = "\n", flush: bool = False) -> None:
    """Log a verbose diagnostic at INFO level, falling back to stdout.

    This preserves legacy ``print()`` visibility for explicit ``verbose=True``
    call sites until those diagnostics are fully stdlib-logging-only.
    """
    message = sep.join(str(value) for value in values)
    if end and end != "\n":
        message += end

    if logger.isEnabledFor(logging.INFO) and _has_enabled_handler(logger, logging.INFO):
        logger.info("%s", message)
    else:
        if "fallback" not in _legacy_verbose_stdout_warning_keys:
            warnings.warn(
                "Newton verbose diagnostics are currently falling back to stdout, but this legacy fallback is "
                "deprecated and these diagnostics will be emitted only through the standard 'newton' logger in a "
                "future release. Configure logging.getLogger('newton') at INFO level with a handler to route or "
                "suppress them.",
                NewtonDeprecationWarning,
                stacklevel=3,
            )
            _legacy_verbose_stdout_warning_keys.add("fallback")
        print(*values, sep=sep, end=end, flush=flush)
