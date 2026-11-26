# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Internal warning utilities shared by builders and importers."""

import inspect
import os

_NEWTON_SRC_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), os.pardir)) + os.sep


def _external_warning_stacklevel() -> int:
    """Return a warning stack level pointing past all ``newton._src`` frames."""
    frame = inspect.currentframe()
    if frame is None:
        return 2
    frame = frame.f_back
    stacklevel = 1
    try:
        while frame is not None and os.path.normpath(frame.f_code.co_filename).startswith(_NEWTON_SRC_DIR):
            frame = frame.f_back
            stacklevel += 1
        return stacklevel
    finally:
        del frame
