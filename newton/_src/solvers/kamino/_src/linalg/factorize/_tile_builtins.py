# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Availability checks for optional Warp tile builtins."""

import os


def _tile_transpose_update_enabled() -> bool:
    return os.environ.get("NEWTON_KAMINO_DISABLE_TILE_TRANSPOSE_UPDATE") not in {"1", "true", "True"}


def _has_warp_builtin(name: str) -> bool:
    if not _tile_transpose_update_enabled():
        return False

    try:
        from warp._src.context import builtin_functions  # noqa: PLC0415
    except Exception:
        return False
    return name in builtin_functions


HAS_TILE_MATMUL_TRANSPOSE_UPDATE = _has_warp_builtin("tile_matmul_transpose_update")
HAS_TILE_MATMUL_LEFT_TRANSPOSE_UPDATE = _has_warp_builtin("tile_matmul_left_transpose_update")
