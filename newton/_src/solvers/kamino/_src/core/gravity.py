# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Gravity containers used by Kamino."""

from __future__ import annotations

from dataclasses import dataclass

import warp as wp

from .....sim.model import Model
from ....coupled.model_view import ModelView

__all__ = ["GRAVITY_DEFAULT", "GravityModel"]


GRAVITY_DEFAULT = -9.81
"""Default gravity along the world's up axis [m/s²]."""


@dataclass
class GravityModel:
    """Hold per-world gravity vectors."""

    vector: wp.array[wp.vec3] | None = None
    """Per-world gravity vector [m/s²]. Shape of ``(num_worlds,)``."""

    @staticmethod
    def from_newton(model: Model | ModelView) -> GravityModel:
        """Create a gravity model that aliases Newton's gravity array."""
        return GravityModel(vector=model.gravity)
