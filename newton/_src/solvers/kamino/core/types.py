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

"""The core data types used throughout Kamino."""

from __future__ import annotations

import sys
import uuid
from enum import IntEnum
from typing import Literal

import numpy as np
import warp as wp

if sys.version_info >= (3, 12):
    from typing import override
else:
    try:
        from typing_extensions import override
    except ImportError:
        # Fallback no-op decorator if typing_extensions is not available
        def override(func):
            return func

###
# Module configs
###

wp.set_module_options({"enable_backward": False})


###
# Generics
###

Vec3 = list[float]
Vec4 = list[float]
Vec6 = list[float]
Quat = list[float]
Mat33 = list[float]
Transform = tuple[Vec3, Quat]


###
# Scalars
###

uint8 = wp.uint8
uint16 = wp.uint16
uint32 = wp.uint32
uint64 = wp.uint64

int8 = wp.int8
int16 = wp.int16
int32 = wp.int32
int64 = wp.int64

float16 = wp.float16
float32 = wp.float32
float64 = wp.float64


###
# Unions
###


FloatType = float16 | float32 | float64
IntType = int16 | int32 | int64
VecIntType = wp.vec2s | wp.vec2i | wp.vec2l


###
# Vectors
###


class vec1i(wp.types.vector(length=1, dtype=int32)):
    pass


class vec2i(wp.types.vector(length=2, dtype=int32)):
    pass


class vec2f(wp.types.vector(length=2, dtype=float32)):
    pass


class vec3i(wp.types.vector(length=3, dtype=int32)):
    pass


class vec3f(wp.types.vector(length=3, dtype=float32)):
    pass


class vec4i(wp.types.vector(length=4, dtype=int32)):
    pass


class vec4f(wp.types.vector(length=4, dtype=float32)):
    pass


class vec5i(wp.types.vector(length=5, dtype=int32)):
    pass


class vec5f(wp.types.vector(length=5, dtype=float32)):
    pass


class vec6i(wp.types.vector(length=6, dtype=int32)):
    pass


class vec6f(wp.types.vector(length=6, dtype=float32)):
    pass


class vec7f(wp.types.vector(length=7, dtype=float32)):
    pass


class vec8f(wp.types.vector(length=8, dtype=float32)):
    pass


class vec14f(wp.types.vector(length=14, dtype=float32)):
    pass


###
# Matrices
###


class mat22f(wp.types.matrix(shape=(2, 2), dtype=float32)):
    pass


class mat33f(wp.types.matrix(shape=(3, 3), dtype=float32)):
    pass


class mat44f(wp.types.matrix(shape=(4, 4), dtype=float32)):
    pass


class mat61f(wp.types.matrix(shape=(6, 1), dtype=float32)):
    pass


class mat16f(wp.types.matrix(shape=(1, 6), dtype=float32)):
    pass


class mat62f(wp.types.matrix(shape=(6, 2), dtype=float32)):
    pass


class mat26f(wp.types.matrix(shape=(2, 6), dtype=float32)):
    pass


class mat63f(wp.types.matrix(shape=(6, 3), dtype=float32)):
    pass


class mat36f(wp.types.matrix(shape=(3, 6), dtype=float32)):
    pass


class mat64f(wp.types.matrix(shape=(6, 4), dtype=float32)):
    pass


class mat46f(wp.types.matrix(shape=(4, 6), dtype=float32)):
    pass


class mat65f(wp.types.matrix(shape=(6, 5), dtype=float32)):
    pass


class mat56f(wp.types.matrix(shape=(5, 6), dtype=float32)):
    pass


class mat66f(wp.types.matrix(shape=(6, 6), dtype=float32)):
    pass


class mat34f(wp.types.matrix(shape=(3, 4), dtype=float32)):
    pass


class mat43f(wp.types.matrix(shape=(4, 3), dtype=float32)):
    pass


class mat38f(wp.types.matrix(shape=(3, 8), dtype=float32)):
    pass


class mat83f(wp.types.matrix(shape=(8, 3), dtype=float32)):
    pass


###
# Quaternions
###


class quatf(wp.types.quaternion(dtype=float32)):
    pass


###
# Transforms
###


class transformf(wp.types.transformation(dtype=float32)):
    pass


###
# Axis
###


class Axis(IntEnum):
    """Enum for representing the three axes in 3D space."""

    X = 0
    Y = 1
    Z = 2

    @classmethod
    def from_string(cls, axis_str: str) -> Axis:
        axis_str = axis_str.lower()
        if axis_str == "x" or axis_str == "X":
            return cls.X
        elif axis_str == "y" or axis_str == "Y":
            return cls.Y
        elif axis_str == "z" or axis_str == "Z":
            return cls.Z
        raise ValueError(f"Invalid axis string: {axis_str}")

    @classmethod
    def from_any(cls, value: AxisType) -> Axis:
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
        if self == Axis.X:
            return (1.0, 0.0, 0.0)
        elif self == Axis.Y:
            return (0.0, 1.0, 0.0)
        else:
            return (0.0, 0.0, 1.0)

    def to_matrix(self) -> Mat33:
        if self == Axis.X:
            return [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
        elif self == Axis.Y:
            return [0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
        else:
            return [0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0]

    def to_vec3(self) -> vec3f:
        return vec3f(*self.to_vector())

    def to_mat33(self) -> mat33f:
        return mat33f(*self.to_matrix())


AxisType = Axis | Literal["X", "Y", "Z"] | Literal[0, 1, 2] | int | str
"""Type that can be used to represent an axis, including the enum, string, and integer representations."""


def axis_to_vec3(axis: AxisType | Vec3) -> vec3f:
    """Convert an axis representation to a 3D vector."""
    if isinstance(axis, list | tuple | np.ndarray):
        return vec3f(*axis)
    elif wp.types.type_is_vector(type(axis)):
        return vec3f(*axis)
    else:
        return Axis.from_any(axis).to_vec3()


def axis_to_mat33(axis: AxisType | Vec3) -> mat33f:
    """Convert an axis representation to a 3x3 matrix."""
    if isinstance(axis, list | tuple | np.ndarray):
        return mat33f(*axis)
    elif wp.types.type_is_vector(type(axis)):
        return mat33f(*axis)
    else:
        return Axis.from_any(axis).to_mat33()


###
# Descriptor
###


class Descriptor:
    """
    Base class for entity descriptor objects.

    A descriptor object is one with a designated name and a unique identifier (UID).
    """

    @staticmethod
    def _assert_valid_uid(uid: str) -> str:
        """Check if a given UID string is valid."""
        try:
            val = uuid.UUID(uid, version=4)
        except ValueError as err:
            raise ValueError("Invalid UID string.") from err
        return str(val)

    def __init__(self, name: str, uid: str | None = None):
        # Instance name
        self._name: str = name
        # Instance UID
        if uid is None:
            # Generate a new UID if none is provided
            uid = str(uuid.uuid4())
        else:
            # Validate provided UID
            if not isinstance(uid, str):
                raise TypeError("UID must be a string.")
            uid = Descriptor._assert_valid_uid(uid)
        self._uid: str = Descriptor._assert_valid_uid(uid)

    @property
    def name(self) -> str:
        """The name of the entity descriptor."""
        return self._name

    @property
    def uid(self) -> str:
        """The unique identifier (UID) of the entity descriptor."""
        return self._uid

    def __repr__(self):
        return f"Descriptor(\nname={self._name},\nuid={self._uid}\n)"


###
# Utilities
###

ArrayLike = np.ndarray | list[int] | list[float] | list[list[int]] | list[list[float]]
"""An Array-like structure for aliasing various data types compatible with numpy."""


FloatArrayLike = np.ndarray | list[float] | list[list[float]]
"""An Array-like structure for aliasing various floating-point data types compatible with numpy."""


IntArrayLike = np.ndarray | list[int] | list[list[int]]
"""An Array-like structure for aliasing various integer data types compatible with numpy."""
