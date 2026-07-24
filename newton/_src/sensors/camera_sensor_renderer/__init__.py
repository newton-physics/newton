# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

from typing import TYPE_CHECKING

from .types import ClearData, GaussianRenderMode, MeshData, RenderConfig, RenderLightType, RenderOrder, TextureData
from .utils import Utils

if TYPE_CHECKING:
    from .render_context import RenderContext

__all__ = [
    "ClearData",
    "GaussianRenderMode",
    "MeshData",
    "RenderConfig",
    "RenderContext",
    "RenderLightType",
    "RenderOrder",
    "TextureData",
    "Utils",
]


def __getattr__(name: str):
    if name == "RenderContext":
        from .render_context import RenderContext  # noqa: PLC0415

        return RenderContext

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
