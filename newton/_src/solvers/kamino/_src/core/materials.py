# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Material coefficient mixing for Kamino contact conversion."""

from __future__ import annotations

from collections.abc import Callable
from enum import IntEnum

import warp as wp

from .....core.types import override

DEFAULT_RESTITUTION = 0.0
"""
The global default restitution coefficient for material pairs.
Equals ``0.0``.
"""

DEFAULT_FRICTION = 0.7
"""
The global default friction coefficient for material pairs.
Equals ``0.7``.
"""


class MaterialMixMode(IntEnum):
    """
    An enumeration defining the heuristic modes for deriving
    pairwise material properties from individual materials.

    This is used when no specific material-pair properties
    are defined and the properties must be derived.
    """

    AVERAGE = 0
    """Pairwise property is the average of the two material properties."""

    MULTIPLY = 1
    """Pairwise property is the product of the two material properties."""

    MAX = 2
    """Pairwise property is the maximum of the two material properties."""

    MIN = 3
    """Pairwise property is the minimum of the two material properties."""

    GEOMETRIC_AVERAGE = 4
    """Pairwise property is the geometric average (square root of the product) of the two material properties."""

    @classmethod
    def from_string(cls, s: str) -> MaterialMixMode:
        """Converts a string to a MaterialMixMode enum value."""
        try:
            return cls[s.upper()]
        except KeyError as e:
            raise ValueError(f"Invalid MaterialMixMode: {s}. Valid options are: {[e.name for e in cls]}") from e

    @override
    def __str__(self):
        """Returns a string representation of the MaterialMixMode."""
        return f"MaterialMixMode.{self.name} ({self.value})"

    @override
    def __repr__(self):
        """Returns a string representation of the MaterialMixMode."""
        return self.__str__()


@wp.func
def material_average(
    value1: wp.float32,
    value2: wp.float32,
) -> wp.float32:
    """
    Computes the average of two material property values.

    Args:
        value1: The first material property value.
        value2: The second material property value.

    Returns:
        The average of the two material property values.
    """
    return 0.5 * (value1 + value2)


@wp.func
def material_multiply(
    value1: wp.float32,
    value2: wp.float32,
) -> wp.float32:
    """
    Computes the product of two material property values.

    Args:
        value1: The first material property value.
        value2: The second material property value.

    Returns:
        The product of the two material property values.
    """
    return value1 * value2


@wp.func
def material_max(
    value1: wp.float32,
    value2: wp.float32,
) -> wp.float32:
    """
    Computes the maximum of two material property values.

    Args:
        value1: The first material property value.
        value2: The second material property value.

    Returns:
        The maximum of the two material property values.
    """
    return wp.max(value1, value2)


@wp.func
def material_min(
    value1: wp.float32,
    value2: wp.float32,
) -> wp.float32:
    """
    Computes the minimum of two material property values.

    Args:
        value1: The first material property value.
        value2: The second material property value.

    Returns:
        The minimum of the two material property values.
    """
    return wp.min(value1, value2)


@wp.func
def material_geometric_average(
    value1: wp.float32,
    value2: wp.float32,
) -> wp.float32:
    """
    Computes the geometric average of two material property values.

    Args:
        value1: The first material property value.
        value2: The second material property value.

    Returns:
        The geometric average of the two material property values.
    """
    return wp.sqrt(value1 * value2)


def get_material_mixing_function(
    mixmode: MaterialMixMode = MaterialMixMode.AVERAGE,
) -> Callable[[wp.float32, wp.float32], wp.float32]:
    """
    Returns the appropriate material mix function based on the specified mixing mode.
    """
    # Select the appropriate mixing function based on the mixing mode
    match mixmode:
        case MaterialMixMode.AVERAGE:
            mix_func = material_average
        case MaterialMixMode.MULTIPLY:
            mix_func = material_multiply
        case MaterialMixMode.MAX:
            mix_func = material_max
        case MaterialMixMode.MIN:
            mix_func = material_min
        case MaterialMixMode.GEOMETRIC_AVERAGE:
            mix_func = material_geometric_average
        case _:
            raise ValueError(f"Unsupported material mixing mode: {mixmode}")

    # Return the selected mixing function
    return mix_func


def make_get_mixed_material_pair_property(mixmode: MaterialMixMode = MaterialMixMode.AVERAGE):
    """
    Generates a Warp function to retrieve material pair
    property based on the specified mixing mode.

    Args:
        mixmode: The mixing mode to use for material pair property.

    Returns:
        A Warp function that retrieves material pair property.
    """
    # Select the appropriate mixing function based on the mixing mode
    mix_func = get_material_mixing_function(mixmode)

    # Define the Warp function to retrieve material pair properties
    @wp.func
    def _get_mixed_material_pair_property(
        property_0: wp.float32,
        property_1: wp.float32,
    ) -> wp.float32:
        """
        Retrieves the mixed properties of a material pair given the properties of the two materials.

        Args:
            property_0: The property of the first material.
            property_1: The property of the second material.

        Returns:
            The mixed property of the two materials.
        """
        return mix_func(property_0, property_1)

    # Return the generated Warp function
    return _get_mixed_material_pair_property
