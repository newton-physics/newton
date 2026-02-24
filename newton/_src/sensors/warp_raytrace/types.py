# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import enum


class RenderLightType(enum.IntEnum):
    """Light types supported by the Warp raytracer."""

    SPOTLIGHT = 0
    """Spotlight."""

    DIRECTIONAL = 1
    """Directional Light."""


class RenderOrder(enum.IntEnum):
    """Render Order"""

    PIXEL_PRIORITY = 0
    """Render the same pixel of every view before continuing to the next one"""
    VIEW_PRIORITY = 1
    """Render all pixels of a whole view before continuing to the next one"""
    TILED = 2
    """Render pixels in tiles, defined by tile_width x tile_height"""
