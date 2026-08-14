# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

# =============================================================================
# FROZEN — do not modify.
#
# This package is the raytracer backing the deprecated ``SensorTiledCamera``.
# It is kept only to support that class through its deprecation window and is
# scheduled for removal alongside it. Do not change anything in this directory:
# it is intentionally not kept in sync with the active renderer, and a guard
# test (``newton/tests/test_frozen_tiled_renderer.py``) fails on any edit here.
#
# All new rendering work belongs in ``newton/_src/render`` (the renderer behind
# ``newton.sensors.SensorCamera`` / ``newton.RenderContext``).
# =============================================================================

from .render_context import RenderContext
from .types import (
    ClearData,
    GaussianRenderMode,
    MeshData,
    RenderConfig,
    RenderLightType,
    RenderOrder,
    TextureData,
    TextureProjectionMode,
)
from .utils import Utils

__all__ = [
    "ClearData",
    "GaussianRenderMode",
    "MeshData",
    "RenderConfig",
    "RenderContext",
    "RenderLightType",
    "RenderOrder",
    "TextureData",
    "TextureProjectionMode",
    "Utils",
]
