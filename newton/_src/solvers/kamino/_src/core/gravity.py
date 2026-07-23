# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Gravity containers used by Kamino."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import warp as wp

from .....core.types import Axis, override
from .....sim.model import Model
from ....coupled.model_view import ModelView
from .types import ArrayLike, Descriptor

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

    @staticmethod
    def from_array(vector: ArrayLike, *, name: str = "gravity") -> GravityDescriptor:
        """Create a gravity descriptor from a three-component vector."""
        components = np.asarray(vector, dtype=np.float32)
        if components.shape != (3,):
            raise ValueError(f"Gravity vector must have shape (3,), got {components.shape}.")
        return GravityDescriptor(name=name, vector=wp.vec3f(*components))

    @staticmethod
    def from_usd(
        direction: ArrayLike, magnitude: float, up_axis: Axis, distance_unit: float, *, name: str = "gravity"
    ) -> GravityDescriptor:
        """Create a gravity descriptor from OpenUSD scene attributes."""
        direction_array = np.asarray(direction, dtype=np.float32)
        if direction_array.shape != (3,):
            raise ValueError(f"Gravity direction must have shape (3,), got {direction_array.shape}.")

        direction_length = np.linalg.norm(direction_array)
        if direction_length == 0.0:
            direction_array = -np.asarray(up_axis.to_vector(), dtype=np.float32)
        else:
            direction_array /= direction_length

        if magnitude == -float("inf"):
            magnitude = abs(GRAVITY_DEFAULT)
        return GravityDescriptor.from_array(direction_array * (distance_unit * magnitude), name=name)

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
