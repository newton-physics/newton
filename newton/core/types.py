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
from typing_extensions import override

from warp.context import Devicelike

# Particle flags
PARTICLE_FLAG_ACTIVE = wp.constant(wp.uint32(1 << 0))

# Shape flags
SHAPE_FLAG_VISIBLE = wp.constant(wp.uint32(1 << 0))
SHAPE_FLAG_COLLIDE_SHAPES = wp.constant(wp.uint32(1 << 1))
SHAPE_FLAG_COLLIDE_GROUND = wp.constant(wp.uint32(1 << 2))


# Types of joints linking rigid bodies
JOINT_PRISMATIC = wp.constant(0)
JOINT_REVOLUTE = wp.constant(1)
JOINT_BALL = wp.constant(2)
JOINT_FIXED = wp.constant(3)
JOINT_FREE = wp.constant(4)
JOINT_COMPOUND = wp.constant(5)
JOINT_UNIVERSAL = wp.constant(6)
JOINT_DISTANCE = wp.constant(7)
JOINT_D6 = wp.constant(8)


def get_joint_dof_count(joint_type: int, num_axes: int) -> tuple[int, int]:
    """Return the number of degrees of freedom in position and velocity for a given joint type."""
    dof_count = num_axes
    coord_count = num_axes
    if joint_type == JOINT_BALL:
        dof_count = 3
        coord_count = 4
    elif joint_type == JOINT_FREE or joint_type == JOINT_DISTANCE:
        dof_count = 6
        coord_count = 7
    elif joint_type == JOINT_FIXED:
        dof_count = 0
        coord_count = 0
    return dof_count, coord_count


# Joint axis control mode types
JOINT_MODE_FORCE = wp.constant(0)
JOINT_MODE_TARGET_POSITION = wp.constant(1)
JOINT_MODE_TARGET_VELOCITY = wp.constant(2)


def flag_to_int(flag):
    """Converts a flag to an integer."""
    if type(flag) in wp.types.int_types:
        return flag.value
    return int(flag)


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


class Axis(IntEnum):
    """Enum for representing the three axes in 3D space."""

    X = 0
    Y = 1
    Z = 2

    @classmethod
    def from_string(cls, axis_str: str) -> Axis:
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

    def to_vector(self) -> tuple[float, float, float]:
        if self == Axis.X:
            return (1.0, 0.0, 0.0)
        elif self == Axis.Y:
            return (0.0, 1.0, 0.0)
        else:
            return (0.0, 0.0, 1.0)

    def to_vec3(self) -> wp.vec3:
        return wp.vec3(*self.to_vector())


AxisType = Axis | Literal["X", "Y", "Z"] | Literal[0, 1, 2] | int | str
"""Type that can be used to represent an axis, including the enum, string, and integer representations."""


def axis_to_vec3(axis: AxisType | Vec3) -> wp.vec3:
    """Convert an axis representation to a 3D vector."""
    if isinstance(axis, (list, tuple, np.ndarray)):
        return wp.vec3(*axis)
    elif wp.types.type_is_vector(type(axis)):
        return wp.vec3(*axis)
    else:
        return Axis.from_any(axis).to_vec3()


@wp.struct
class ShapeMaterials:
    """
    Represents the contact material properties of a shape.
        ke: The contact elastic stiffness (only used by the Euler integrators)
        kd: The contact damping stiffness (only used by the Euler integrators)
        kf: The contact friction stiffness (only used by the Euler integrators)
        ka: The contact adhesion distance (values greater than 0 mean adhesive contact; only used by the Euler integrators)
        mu: The coefficient of friction
        restitution: The coefficient of restitution (only used by XPBD integrator)
    """

    ke: wp.array(dtype=float)
    kd: wp.array(dtype=float)
    kf: wp.array(dtype=float)
    ka: wp.array(dtype=float)
    mu: wp.array(dtype=float)
    restitution: wp.array(dtype=float)


@wp.struct
class ShapeGeometry:
    """
    Represents the geometry of a shape.
        type: The type of geometry (GEO_SPHERE, GEO_BOX, etc.)
        is_solid: Indicates whether the shape is solid or hollow
        thickness: The thickness of the shape (used for collision detection, and inertia computation of hollow shapes)
        source: Pointer to the source geometry (can be a mesh or SDF index, zero otherwise)
        scale: The 3D scale of the shape
        filter: The filter group of the shape
        transform: The transform of the shape in world space
    """

    type: wp.array(dtype=wp.int32)
    is_solid: wp.array(dtype=bool)
    thickness: wp.array(dtype=float)
    source: wp.array(dtype=wp.uint64)
    scale: wp.array(dtype=wp.vec3)
    filter: wp.array(dtype=int)


# model update flags - used for solver.notify_model_update()
NOTIFY_FLAG_JOINT_PROPERTIES = wp.constant(1 << 0)  # joint_q, joint_X_p, joint_X_c
NOTIFY_FLAG_JOINT_AXIS_PROPERTIES = wp.constant(
    1 << 1
)  # joint_target, joint_target_ke, joint_target_kd, joint_axis_mode, joint_limit_upper, joint_limit_lower, joint_limit_ke, joint_limit_kd
NOTIFY_FLAG_DOF_PROPERTIES = wp.constant(1 << 2)  # joint_qd, joint_f, joint_armature
NOTIFY_FLAG_BODY_PROPERTIES = wp.constant(1 << 3)  # body_q, body_qd
NOTIFY_FLAG_BODY_INERTIAL_PROPERTIES = wp.constant(
    1 << 4
)  # body_com, body_inertia, body_inv_inertia, body_mass, body_inv_mass
NOTIFY_FLAG_SHAPE_PROPERTIES = wp.constant(1 << 5)  # shape_transform, shape_geo

__all__ = [
    "JOINT_BALL",
    "JOINT_COMPOUND",
    "JOINT_D6",
    "JOINT_DISTANCE",
    "JOINT_FIXED",
    "JOINT_FREE",
    "JOINT_MODE_FORCE",
    "JOINT_MODE_TARGET_POSITION",
    "JOINT_MODE_TARGET_VELOCITY",
    "JOINT_PRISMATIC",
    "JOINT_REVOLUTE",
    "JOINT_UNIVERSAL",
    "NOTIFY_FLAG_BODY_INERTIAL_PROPERTIES",
    "NOTIFY_FLAG_BODY_PROPERTIES",
    "NOTIFY_FLAG_DOF_PROPERTIES",
    "NOTIFY_FLAG_JOINT_PROPERTIES",
    "NOTIFY_FLAG_SHAPE_PROPERTIES",
    "PARTICLE_FLAG_ACTIVE",
    "SHAPE_FLAG_COLLIDE_GROUND",
    "SHAPE_FLAG_COLLIDE_SHAPES",
    "SHAPE_FLAG_VISIBLE",
    "Axis",
    "AxisType",
    "Mat33",
    "Quat",
    "Sequence",
    "ShapeMaterials",
    "Transform",
    "Vec3",
    "Vec4",
    "flag_to_int",
    "get_joint_dof_count",
]
