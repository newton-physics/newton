# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Gravity containers used by Kamino."""

from __future__ import annotations

from dataclasses import dataclass, field

import warp as wp

from .....core.types import Axis, override
from .....sim.model import Model
from ....coupled.model_view import ModelView
from .types import Descriptor

__all__ = ["GRAVITY_DEFAULT", "GravityDescriptor", "GravityModel"]


GRAVITY_DEFAULT = -9.81
"""Default gravity along the world's up axis [m/s²]."""


@dataclass
class GravityDescriptor(Descriptor):
    """Describe a world's gravity vector."""

    vector: wp.vec3f = field(default_factory=lambda: GravityDescriptor.default_for(Axis.Z).vector)
    """Gravity vector [m/s²]."""

    @staticmethod
    def default_for(up_axis: Axis, *, name: str = "gravity") -> GravityDescriptor:
        """Return Newton's default gravity along the negative up axis."""
        vector = wp.vec3f(*(component * GRAVITY_DEFAULT for component in up_axis.to_vector()))
        return GravityDescriptor(name=name, vector=vector)

    @override
    def __repr__(self) -> str:
        """Return a human-readable representation."""
        return f"GravityDescriptor(name={self.name!r}, uid={self.uid!r}, vector={self.vector})"


@dataclass
class GravityModel:
    """Hold per-world gravity vectors."""

    vector: wp.array[wp.vec3] | None = None
    """Per-world gravity vectors [m/s²], shape [world_count]."""

    @staticmethod
    def from_newton(model: Model | ModelView) -> GravityModel:
        """Create a gravity model that aliases Newton's gravity array."""
        return GravityModel(vector=model.gravity)
