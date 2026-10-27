# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import logging
import sys

_LEVEL_PREFIXES = {logging.WARNING: "Warning: ", logging.ERROR: "Error: ", logging.CRITICAL: "Error: "}


class _ConsoleFallbackHandler(logging.Handler):
    """Print ``newton`` log records until the application configures a handler.

    Replaces :data:`logging.lastResort`, which drops ``INFO`` records, so that
    output Newton used to ``print()`` stays visible by default.
    """

    def emit(self, record: logging.LogRecord) -> None:
        if self._other_handler_configured(record.name):
            return
        try:
            # Resolve the stream per record so contextlib.redirect_stdout() captures it like print().
            stream = sys.stderr if record.levelno >= logging.WARNING else sys.stdout
            if stream is None:  # pythonw and similar hosts have no console
                return
            stream.write(_LEVEL_PREFIXES.get(record.levelno, "") + self.format(record) + "\n")
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


def install_console_fallback() -> None:
    """Make ``newton`` INFO records visible without application logging configuration."""
    logger = logging.getLogger("newton")
    if logger.level == logging.NOTSET:
        logger.setLevel(logging.INFO)
    if not any(isinstance(handler, _ConsoleFallbackHandler) for handler in logger.handlers):
        logger.addHandler(_ConsoleFallbackHandler())
