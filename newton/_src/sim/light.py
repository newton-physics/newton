# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Light declarations stored on :class:`~newton.Model`.

A model light describes what an asset authors about a light source: the frame it
is mounted in, the world it belongs to, its type, color, intensity, falloff and
cone. How a light is shaded (e.g. whether shadows are traced) belongs to the
renderer that draws the model (see :class:`~newton.sensors.SensorCamera`).
"""

from __future__ import annotations

from enum import IntEnum


class LightType(IntEnum):
    """Type of a model light.

    The values match MuJoCo's ``mjtLightType``. Lights emit along their local
    ``-Z`` axis, like ``UsdLux`` lights.
    """

    SPOT = 0
    """Point light whose emission is limited to a cone around its local ``-Z`` axis."""

    DIRECTIONAL = 1
    """Light at infinity whose parallel rays travel along its local ``-Z`` axis."""

    POINT = 2
    """Light that emits from its origin in every direction."""
