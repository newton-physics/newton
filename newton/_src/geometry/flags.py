# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

from enum import IntEnum


# Particle flags
class ParticleFlags(IntEnum):
    """
    Flags for particle properties.
    """

    ACTIVE = 1 << 0
    """Indicates that the particle is active."""


# Shape flags
class ShapeFlags(IntEnum):
    """
    Flags for shape properties.
    """

    VISIBLE = 1 << 0
    """Indicates that the shape is visible."""

    COLLIDE_SHAPES = 1 << 1
    """Indicates that the shape collides with other shapes."""

    COLLIDE_PARTICLES = 1 << 2
    """Indicates that the shape collides with particles."""

    SITE = 1 << 3
    """Indicates that the shape is a site (non-colliding reference point)."""

    HYDROELASTIC = 1 << 4
    """Legacy hydroelastic enable bit (defaults to compliant when mode bits are absent)."""

    HYDROELASTIC_RIGID = 1 << 5
    """Indicates rigid hydroelastic behavior for this shape."""

    HYDROELASTIC_COMPLIANT = 1 << 6
    """Indicates compliant hydroelastic behavior for this shape."""


class HydroelasticType(IntEnum):
    """Hydroelastic participation mode for a collision shape."""

    NONE = 0
    """Disable hydroelastic behavior for this shape."""

    RIGID = 1
    """Treat this shape as rigid in hydroelastic contacts."""

    COMPLIANT = 2
    """Treat this shape as compliant in hydroelastic contacts."""


def hydroelastic_type_from_flags(flags: int) -> HydroelasticType:
    """Decode hydroelastic mode from a shape flag bitmask.

    Args:
        flags: Integer bitmask using :class:`ShapeFlags`.

    Returns:
        Hydroelastic mode encoded by the bitmask.
    """
    if flags & int(ShapeFlags.HYDROELASTIC_RIGID):
        return HydroelasticType.RIGID
    if flags & int(ShapeFlags.HYDROELASTIC_COMPLIANT):
        return HydroelasticType.COMPLIANT
    if flags & int(ShapeFlags.HYDROELASTIC):
        # Backward compatibility with models that only set HYDROELASTIC.
        return HydroelasticType.COMPLIANT
    return HydroelasticType.NONE


__all__ = [
    "HydroelasticType",
    "ParticleFlags",
    "ShapeFlags",
    "hydroelastic_type_from_flags",
]
