# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Internal diagnostic routing helpers."""

from __future__ import annotations

import logging
import warnings

from newton.exceptions import NewtonDeprecationWarning

_legacy_verbose_stdout_warning_keys: set[str] = set()


def _has_enabled_handler(logger: logging.Logger, level: int) -> bool:
    current: logging.Logger | None = logger
    while current is not None:
        if any(level >= handler.level for handler in current.handlers):
            return True
        if not current.propagate:
            return False
        current = current.parent
    return logging.lastResort is not None and level >= logging.lastResort.level


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
