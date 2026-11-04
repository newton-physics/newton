# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Experimental, lazy access to the exact external benchmark asset revisions.

Newton's downloaders honor ``NEWTON_CACHE_PATH`` and reuse existing downloads.
Importing this module does not download assets or initialize a device.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

__all__ = [
    "NEWTON_ASSETS_REVISION",
    "STYLE3D_REVISION",
    "asset_cache_directory",
    "newton_asset_path",
]

NEWTON_ASSETS_REVISION = "f8fb7abcbeba2318814a74f3eeb02780ad7925d6"
STYLE3D_REVISION = "26c7f5ccaadd9a163f4d6c8f867c789b3ac57d83"


def newton_asset_path(folder: str, *, revision: str = NEWTON_ASSETS_REVISION) -> Path:
    """Download or reuse a pinned Newton asset folder without changing its files."""
    import newton.utils  # noqa: PLC0415

    return Path(newton.utils.download_asset(folder, ref=revision))


def asset_cache_directory(cache_dir: str | Path | None = None) -> Path:
    """Locate the cache for a local-only inventory, without triggering downloads.

    Match Newton's documented environment override and platform user-cache
    convention. Runtime imports use the public downloaders, not cache filenames.
    """
    if cache_dir is not None:
        return Path(cache_dir)
    if "NEWTON_CACHE_PATH" in os.environ:
        return Path(os.environ["NEWTON_CACHE_PATH"])
    if sys.platform == "win32":
        return Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData/Local")) / "newton-physics/newton"
    if sys.platform == "darwin":
        return Path.home() / "Library/Caches/newton"
    return Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "newton"
