# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
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

"""
Utilities for configuring SDF (Signed Distance Field) collision detection.

This module provides helper functions for SDF configuration. The main API is now
available as methods on ModelBuilder: :meth:`ModelBuilder.enable_sdf()` and
:meth:`ModelBuilder.disable_sdf()`.

This module is kept for backward compatibility and re-exports the builder methods.
"""

from __future__ import annotations

from ..sim.builder import ModelBuilder


def enable_sdf(
    builder: ModelBuilder,
    *,
    max_resolution: int = 64,
    narrow_band_range: tuple[float, float] = (-0.01, 0.01),
    contact_margin: float = 0.01,
) -> None:
    """Enable SDF-based collision detection for a ModelBuilder.

    .. deprecated::
        Use :meth:`ModelBuilder.enable_sdf()` instead.

    Args:
        builder: The ModelBuilder to configure
        max_resolution: Maximum SDF grid dimension (must be divisible by 8). Common: 32, 64, 128, 256
        narrow_band_range: (inner, outer) distance range for SDF computation in meters
        contact_margin: Contact detection margin in meters
    """
    builder.enable_sdf(
        max_resolution=max_resolution,
        narrow_band_range=narrow_band_range,
        contact_margin=contact_margin,
    )


def disable_sdf(builder: ModelBuilder) -> None:
    """Disable SDF-based collision detection for a ModelBuilder.

    .. deprecated::
        Use :meth:`ModelBuilder.disable_sdf()` instead.

    Args:
        builder: The ModelBuilder to configure
    """
    builder.disable_sdf()
