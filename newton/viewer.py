# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

from typing import TYPE_CHECKING

from ._src import viewer as _viewer

if TYPE_CHECKING:
    from ._src.viewer import (
        Layer,
        ViewerBase,
        ViewerFile,
        ViewerGL,
        ViewerNull,
        ViewerRerun,
        ViewerRTX,
        ViewerUSD,
        ViewerViser,
    )

__all__ = [
    "Layer",
    "ViewerBase",
    "ViewerFile",
    "ViewerGL",
    "ViewerNull",
    "ViewerRTX",
    "ViewerRerun",
    "ViewerUSD",
    "ViewerViser",
]


def __getattr__(name: str):
    if name not in __all__:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None
    value = getattr(_viewer, name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
