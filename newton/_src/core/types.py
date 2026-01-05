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

"""Common definitions for types and constants."""

from __future__ import annotations

from collections.abc import Sequence
from enum import IntEnum
from typing import Any, Literal

import numpy as np
import warp as wp
from warp.context import Devicelike

try:
    from typing import override
except ImportError:
    try:
        from typing_extensions import override
    except ImportError:
        # Fallback no-op decorator if override is not available
        def override(func):
            return func


warp_int_types = (wp.int8, wp.uint8, wp.int16, wp.uint16, wp.int32, wp.uint32, wp.int64, wp.uint64)


def flag_to_int(flag):
    """Converts a flag (Warp constant) to an integer."""
    if type(flag) in warp_int_types:
        return flag.value
    return int(flag)


Vec2 = list[float] | tuple[float, float] | wp.vec2
"""A 2D vector represented as a list or tuple of 2 floats."""
Vec3 = list[float] | tuple[float, float, float] | wp.vec3
"""A 3D vector represented as a list or tuple of 3 floats."""
Vec4 = list[float] | tuple[float, float, float, float] | wp.vec4
"""A 4D vector represented as a list or tuple of 4 floats."""
Quat = list[float] | tuple[float, float, float, float] | wp.quat
"""A quaternion represented as a list or tuple of 4 floats (in XYZW order)."""
Mat33 = list[float] | wp.mat33
"""A 3x3 matrix represented as a list of 9 floats or a ``warp.mat33``."""
Transform = tuple[Vec3, Quat] | wp.transform
"""A 3D transformation represented as a tuple of 3D translation and rotation quaternion (in XYZW order)."""

# type alias for numpy arrays
nparray = np.ndarray[Any, np.dtype[Any]]

# Warp vector types
vec5 = wp.types.vector(length=5, dtype=wp.float32)


class Axis(IntEnum):
    """Enumeration of axes in 3D space."""

    X = 0
    """X-axis."""
    Y = 1
    """Y-axis."""
    Z = 2
    """Z-axis."""

    @classmethod
    def from_string(cls, axis_str: str) -> Axis:
        """
        Convert a string representation of an axis ("x", "y", or "z") to the corresponding Axis enum member.

        Args:
            axis_str (str): The axis as a string. Should be "x", "y", or "z" (case-insensitive).

        Returns:
            Axis: The corresponding Axis enum member.

        Raises:
            ValueError: If the input string does not correspond to a valid axis.
        """
        axis_str = axis_str.lower()
        if axis_str == "x":
            return cls.X
        elif axis_str == "y":
            return cls.Y
        elif axis_str == "z":
            return cls.Z
        raise ValueError(f"Invalid axis string: {axis_str}")

    @classmethod
    def from_any(cls, value: AxisType) -> Axis:
        """
        Convert a value of various types to an Axis enum member.

        Args:
            value (AxisType): The value to convert. Can be an Axis, str, or int-like.

        Returns:
            Axis: The corresponding Axis enum member.

        Raises:
            TypeError: If the value cannot be converted to an Axis.
            ValueError: If the string or integer does not correspond to a valid Axis.
        """
        if isinstance(value, cls):
            return value
        if isinstance(value, str):
            return cls.from_string(value)
        if type(value) in {int, wp.int32, wp.int64, np.int32, np.int64}:
            return cls(value)
        raise TypeError(f"Cannot convert {type(value)} to Axis")

    @override
    def __str__(self):
        return self.name.capitalize()

    @override
    def __repr__(self):
        return f"Axis.{self.name.capitalize()}"

    @override
    def __eq__(self, other):
        if isinstance(other, str):
            return self.name.lower() == other.lower()
        if type(other) in {int, wp.int32, wp.int64, np.int32, np.int64}:
            return self.value == int(other)
        return NotImplemented

    @override
    def __hash__(self):
        return hash(self.name)

    def to_vector(self) -> tuple[float, float, float]:
        """
        Return the axis as a 3D unit vector.

        Returns:
            tuple[float, float, float]: The unit vector corresponding to the axis.
        """
        if self == Axis.X:
            return (1.0, 0.0, 0.0)
        elif self == Axis.Y:
            return (0.0, 1.0, 0.0)
        else:
            return (0.0, 0.0, 1.0)

    def to_vec3(self) -> wp.vec3:
        """
        Return the axis as a warp.vec3 unit vector.

        Returns:
            wp.vec3: The unit vector corresponding to the axis.
        """
        return wp.vec3(*self.to_vector())

    def quat_between_axes(self, other: Axis) -> wp.quat:
        """
        Return the quaternion between two axes.
        """
        return wp.quat_between_vectors(self.to_vec3(), other.to_vec3())


AxisType = Axis | Literal["X", "Y", "Z"] | Literal[0, 1, 2] | int | str
"""Type that can be used to represent an axis, including the enum, string, and integer representations."""


def axis_to_vec3(axis: AxisType | Vec3) -> wp.vec3:
    """Convert an axis representation to a 3D vector."""
    if isinstance(axis, list | tuple | np.ndarray):
        return wp.vec3(*axis)
    elif wp.types.type_is_vector(type(axis)):
        return wp.vec3(*axis)
    else:
        return Axis.from_any(axis).to_vec3()


__all__ = [
    "Axis",
    "AxisType",
    "Devicelike",
    "Mat33",
    "Quat",
    "Sequence",
    "Transform",
    "Vec2",
    "Vec3",
    "Vec4",
    "flag_to_int",
    "override",
    "vec5",
]
