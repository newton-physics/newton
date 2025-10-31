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

"""KAMINO: Shape Types & Containers"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable, Sequence
from enum import IntEnum

import warp as wp

from ....core.types import Mat33, Vec2, Vec3, nparray
from ....geometry.types import MESH_MAXHULLVERT, SDF, GeoType, Mesh
from .types import Descriptor, mat33f, override, vec3f, vec4f

###
# Module interface
###

__all__ = [
    "BoxShape",
    "CapsuleShape",
    "ConeShape",
    "CylinderShape",
    "EllipsoidShape",
    "EmptyShape",
    "MeshShape",
    "PlaneShape",
    "SDFShape",
    "ShapeDescriptor",
    "ShapeType",
    "SphereShape",
]


###
# Module configs
###

wp.set_module_options({"enable_backward": False})


###
# Containers
###


class ShapeType(IntEnum):
    """
    An enumeration of the different shape types.
    """

    EMPTY = 0
    """The empty shape type, which has no parameters and is used to represent the absence of a shape."""

    SPHERE = 1
    """The 1-parameter sphere shape type. Parameters: radius."""

    CYLINDER = 2
    """The 2-parameter cylinder shape type. Parameters: radius, height."""

    CONE = 3
    """The 2-parameter cone shape type. Parameters: radius, height."""

    CAPSULE = 4
    """The 2-parameter capsule shape type. Parameters: radius, height."""

    BOX = 5
    """The 3-parameter box shape type. Parameters: depth, width, height."""

    ELLIPSOID = 6
    """The 3-parameter ellipsoid shape type. Parameters: a, b, c."""

    PLANE = 7
    """The 4-parameter plane shape type. Parameters: normal_x, normal_y, normal_z, distance."""

    MESH = 8
    """The n-parameter mesh shape type. Parameters: vertices, normals, triangles, triangle_normals."""

    CONVEX = 9
    """The n-parameter height-field shape type. Parameters: height field data, etc."""

    HFIELD = 10
    """The n-parameter height-field shape type. Parameters: height field data, etc."""

    SDF = 11
    """The n-parameter signed-distance-field shape type. Parameters: sdf data, etc."""

    @override
    def __str__(self):
        """Returns a string representation of the shape type."""
        return f"ShapeType.{self.name} ({self.value})"

    @override
    def __repr__(self):
        """Returns a string representation of the shape type."""
        return self.__str__()

    @property
    def num_params(self) -> int:
        """
        The number of parameters that describe the shape type.
        """
        if self.value == self.EMPTY:
            return 0
        elif self.value == self.SPHERE:
            return 1
        elif self.value == self.CYLINDER:
            return 2
        elif self.value == self.CONE:
            return 2
        elif self.value == self.CAPSULE:
            return 2
        elif self.value == self.BOX:
            return 3
        elif self.value == self.ELLIPSOID:
            return 3
        elif self.value == self.PLANE:
            return 4
        elif self.value in {self.MESH, self.CONVEX, self.HFIELD, self.SDF}:
            return -1  # Indicates variable number of parameters
        else:
            raise ValueError(f"Unknown shape type value: {self.value}")

    def to_newton(self) -> GeoType:
        """
        Convert the shape type to the corresponding Newton shape type.
        """
        type = None
        match self:
            case ShapeType.EMPTY:
                type = GeoType.NONE
            case ShapeType.SPHERE:
                type = GeoType.SPHERE
            case ShapeType.CYLINDER:
                type = GeoType.CYLINDER
            case ShapeType.CONE:
                type = GeoType.CONE
            case ShapeType.CAPSULE:
                type = GeoType.CAPSULE
            case ShapeType.BOX:
                type = GeoType.BOX
            case ShapeType.ELLIPSOID:
                type = GeoType.ELLIPSOID
            case ShapeType.PLANE:
                type = GeoType.PLANE
            case ShapeType.CONVEX:
                type = GeoType.CONVEX_MESH
            case ShapeType.MESH:
                type = GeoType.MESH
            case ShapeType.HFIELD:
                type = GeoType.HFIELD
            case ShapeType.SDF:
                type = GeoType.SDF
            case _:
                raise ValueError(f"Unknown ShapeType value: {type}")
        return type


ShapeParamsLike = None | float | Iterable[float]
"""A type union that can represent any shape parameters, including None, single float, or iterable of floats."""

ShapeDataLike = None | Mesh | SDF
"""A type union that can represent any shape data, including None, Mesh, and SDF."""


class ShapeDescriptor(ABC, Descriptor):
    """Abstract base class for all shape descriptors."""

    def __init__(self, type: ShapeType, name: str = "", uid: str | None = None):
        super().__init__(name, uid)
        self._type: ShapeType = type

    def __repr__(self):
        return f"ShapeDescriptor(\ntype: {self.type},\nname: {self.name},\nuid: {self.uid},\n)"

    @property
    def type(self) -> ShapeType:
        return self._type

    @property
    def num_params(self) -> int:
        return self._type.num_params

    @property
    @abstractmethod
    def paramsvec(self) -> vec4f:
        return vec4f(0.0)

    @property
    @abstractmethod
    def params(self) -> ShapeParamsLike:
        return None

    @property
    @abstractmethod
    def data(self) -> ShapeDataLike:
        return None

    @property
    def is_solid(self) -> bool:
        return True


###
# Primitive Shapes
###


class EmptyShape(ShapeDescriptor):
    def __init__(self, name: str = "empty", uid: str | None = None):
        super().__init__(ShapeType.EMPTY, name, uid)

    @override
    def __repr__(self):
        return f"EmptyShape(\nname: {self.name},\nuid: {self.uid}\n)"

    @property
    @override
    def paramsvec(self) -> vec4f:
        return vec4f(0.0)

    @property
    @override
    def params(self) -> ShapeParamsLike:
        return None

    @property
    @override
    def data(self) -> None:
        return None


class SphereShape(ShapeDescriptor):
    def __init__(self, radius: float, name: str = "sphere", uid: str | None = None):
        super().__init__(ShapeType.SPHERE, name, uid)
        self.radius: float = radius

    @override
    def __repr__(self):
        return f"SphereShape(\nname: {self.name},\nuid: {self.uid},\nradius: {self.radius}\n)"

    @property
    @override
    def paramsvec(self) -> vec4f:
        return vec4f(self.radius, 0.0, 0.0, 0.0)

    @property
    @override
    def params(self) -> float:
        return self.radius

    @property
    @override
    def data(self) -> None:
        return None


class CylinderShape(ShapeDescriptor):
    def __init__(self, radius: float, height: float, name: str = "cylinder", uid: str | None = None):
        super().__init__(ShapeType.CYLINDER, name, uid)
        self.radius: float = radius
        self.height: float = height

    @override
    def __repr__(self):
        return f"CylinderShape(\nname: {self.name},\nuid: {self.uid},\nradius: {self.radius},\nheight: {self.height}\n)"

    @property
    @override
    def paramsvec(self) -> vec4f:
        return vec4f(self.radius, self.height, 0.0, 0.0)

    @property
    @override
    def params(self) -> tuple[float, float]:
        return (self.radius, self.height)

    @property
    @override
    def data(self) -> None:
        return None


class ConeShape(ShapeDescriptor):
    def __init__(self, radius: float, height: float, name: str = "cone", uid: str | None = None):
        super().__init__(ShapeType.CONE, name, uid)
        self.radius: float = radius
        self.height: float = height

    @override
    def __repr__(self):
        return f"ConeShape(\nname: {self.name},\nuid: {self.uid},\nradius: {self.radius},\nheight: {self.height}\n)"

    @property
    @override
    def paramsvec(self) -> vec4f:
        return vec4f(self.radius, self.height, 0.0, 0.0)

    @property
    @override
    def params(self) -> tuple[float, float]:
        return (self.radius, self.height)

    @property
    @override
    def data(self) -> None:
        return None


class CapsuleShape(ShapeDescriptor):
    def __init__(self, radius: float, height: float, name: str = "capsule", uid: str | None = None):
        super().__init__(ShapeType.CAPSULE, name, uid)
        self.radius: float = radius
        self.height: float = height

    @override
    def __repr__(self):
        return f"CapsuleShape(\nname: {self.name},\nuid: {self.uid},\nradius: {self.radius},\nheight: {self.height}\n)"

    @property
    @override
    def paramsvec(self) -> vec4f:
        return vec4f(self.radius, self.height, 0.0, 0.0)

    @property
    @override
    def params(self) -> tuple[float, float]:
        return (self.radius, self.height)

    @property
    @override
    def data(self) -> None:
        return None


class BoxShape(ShapeDescriptor):
    def __init__(self, depth: float, width: float, height: float, name: str = "box", uid: str | None = None):
        super().__init__(ShapeType.BOX, name, uid)
        self.depth: float = depth
        self.width: float = width
        self.height: float = height

    @override
    def __repr__(self):
        return (
            f"BoxShape(\n"
            f"name: {self.name},\n"
            f"uid: {self.uid},\n"
            f"depth: {self.depth},\n"
            f"width: {self.width},\n"
            f"height: {self.height}\n"
            f")"
        )

    @property
    @override
    def paramsvec(self) -> vec4f:
        return vec4f(self.depth, self.width, self.height, 0.0)

    @property
    @override
    def params(self) -> tuple[float, float, float]:
        return (self.depth, self.width, self.height)

    @property
    @override
    def data(self) -> None:
        return None


class EllipsoidShape(ShapeDescriptor):
    def __init__(self, a: float, b: float, c: float, name: str = "ellipsoid", uid: str | None = None):
        super().__init__(ShapeType.ELLIPSOID, name, uid)
        self.a: float = a
        self.b: float = b
        self.c: float = c

    @override
    def __repr__(self):
        return f"EllipsoidShape(\nname: {self.name},\nuid: {self.uid},\na: {self.a},\nb: {self.b},\nc: {self.c}\n)"

    @property
    @override
    def paramsvec(self) -> vec4f:
        return vec4f(self.a, self.b, self.c, 0.0)

    @property
    @override
    def params(self) -> tuple[float, float, float]:
        return (self.a, self.b, self.c)

    @property
    @override
    def data(self) -> None:
        return None


class PlaneShape(ShapeDescriptor):
    def __init__(self, normal: Vec3, distance: float, name: str = "plane", uid: str | None = None):
        super().__init__(ShapeType.PLANE, name, uid)
        self.normal: Vec3 = normal
        self.distance: float = distance

    @override
    def __repr__(self):
        return (
            f"PlaneShape(\nname: {self.name},\nuid: {self.uid},\nnormal: {self.normal},\ndistance: {self.distance}\n)"
        )

    @property
    @override
    def paramsvec(self) -> vec4f:
        return vec4f(self.normal[0], self.normal[1], self.normal[2], self.distance)

    @property
    @override
    def params(self) -> tuple[float, float, float, float]:
        return (self.normal[0], self.normal[1], self.normal[2], self.distance)

    @property
    @override
    def data(self) -> None:
        return None


###
# Explicit Shapes
###


class MeshShape(ShapeDescriptor):
    """
    A shape descriptor for mesh shapes.

    This class is a lightweight wrapper around the newton.Mesh geometry type,
    that provides the necessary interfacing to be used with the Kamino solver.
    """

    def __init__(
        self,
        vertices: Sequence[Vec3] | nparray,
        indices: Sequence[int] | nparray,
        normals: Sequence[Vec3] | nparray | None = None,
        uvs: Sequence[Vec2] | nparray | None = None,
        color: Vec3 | None = None,
        maxhullvert: int = MESH_MAXHULLVERT,
        compute_inertia: bool = True,
        is_solid: bool = True,
        is_convex: bool = False,
        name: str = "mesh",
        uid: str | None = None,
    ):
        # Determine the mesh shape type, and adapt default name if necessary
        if is_convex:
            shape_type = ShapeType.CONVEX
            name = "convex" if name == "mesh" else name
        else:
            shape_type = ShapeType.MESH

        # Initialize the base shape descriptor
        super().__init__(shape_type, name, uid)

        # Create the underlying mesh data container
        self._data: Mesh = Mesh(
            vertices=vertices,
            indices=indices,
            normals=normals,
            uvs=uvs,
            compute_inertia=compute_inertia,
            is_solid=is_solid,
            maxhullvert=maxhullvert,
            color=color,
        )

    @override
    def __repr__(self):
        return (
            "MeshShape(\n"
            if self.type == ShapeType.MESH
            else "ConvexShape(\n"
            f"name: {self.name},\n"
            f"uid: {self.uid},\n"
            f"vertices: {self._data.vertices.shape},\n"
            f"indices: {self._data.indices.shape},\n"
            f"normals: {self._data._normals.shape if self._data._normals is not None else None},\n"
            f")"
        )

    @property
    @override
    def paramsvec(self) -> vec4f:
        return vec4f(0.0)

    @property
    @override
    def params(self) -> float:
        return 1.0

    @property
    @override
    def data(self) -> Mesh:
        return self._data

    @property
    def vertices(self) -> nparray:
        return self._data.vertices

    @property
    def indices(self) -> nparray:
        return self._data.indices

    @property
    def normals(self) -> nparray | None:
        return self._data._normals

    @property
    def uvs(self) -> nparray | None:
        return self._data._uvs

    @property
    def color(self) -> Vec3 | None:
        return self._data._color


class HFieldShape(ShapeDescriptor):
    def __init__(self, name: str = "hfield", uid: str | None = None):
        super().__init__(ShapeType.HFIELD, name, uid)
        # TODO: Remove this when HFieldShape is implemented
        raise NotImplementedError("HFieldShape is not yet implemented.")

    @override
    def __repr__(self):
        return f"HFieldShape(\nname: {self.name},\nuid: {self.uid}\n)"

    @property
    @override
    def params(self) -> vec4f:
        return vec4f(0.0)


class SDFShape(ShapeDescriptor):
    """
    A shape descriptor for SDF shapes.

    This class is a lightweight wrapper around the newton.SDF geometry type,
    that provides the necessary interfacing to be used with the Kamino solver.
    """

    def __init__(
        self,
        volume: wp.Volume | None = None,
        mass: float = 1.0,
        com: Vec3 | None = None,
        inertia: Mat33 | None = None,
        name: str = "sdf",
        uid: str | None = None,
    ):
        super().__init__(ShapeType.SDF, name, uid)
        self._data: SDF = SDF(
            volume=volume,
            mass=mass,
            com=com,
            I=inertia,
        )

    @override
    def __repr__(self):
        return (
            f"SDFShape(\n"
            f"name: {self.name},\n"
            f"uid: {self.uid},\n"
            f"mass: {self._data.mass},\n"
            f"com: {self._data.com},\n"
            f"I:\n{self._data.I},\n"
            f")"
        )

    @property
    @override
    def paramsvec(self) -> vec4f:
        return vec4f(0.0)

    @property
    @override
    def params(self) -> None:
        return None

    @property
    @override
    def data(self) -> SDF:
        return self._data

    @property
    def volume(self) -> wp.Volume:
        return self._data.volume

    @property
    def mass(self) -> float:
        return self._data.mass

    @property
    def com(self) -> vec3f:
        return self._data.com

    @property
    def inertia(self) -> mat33f:
        return self._data.I


###
# Aliases
###


ShapeDescriptorType = (
    EmptyShape
    | SphereShape
    | CylinderShape
    | ConeShape
    | CapsuleShape
    | BoxShape
    | EllipsoidShape
    | PlaneShape
    | MeshShape
    | HFieldShape
    | SDFShape
)
"""A type union that can represent any shape descriptor, including primitive and explicit shapes."""
