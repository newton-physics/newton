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

from .....core.types import Vec2, Vec3, nparray
from .....geometry.types import GeoType, Heightfield, Mesh
from .types import Descriptor, override, vec3f, vec4f

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

    ###
    # Operations
    ###

    @override
    def __str__(self):
        """Returns a string representation of the shape type."""
        return f"ShapeType.{self.name} ({self.value})"

    @override
    def __repr__(self):
        """Returns a string representation of the shape type."""
        return self.__str__()

    @property
    def is_empty(self) -> bool:
        """
        Returns whether the shape type is the empty shape.
        """
        return self.value == self.EMPTY

    @property
    def is_primitive(self) -> bool:
        """
        Returns whether the shape type is a primitive shape.
        """
        return self.value in {
            self.SPHERE,
            self.CYLINDER,
            self.CONE,
            self.CAPSULE,
            self.BOX,
            self.ELLIPSOID,
            self.PLANE,
        }

    @property
    def is_explicit(self) -> bool:
        """
        Returns whether the shape type is an explicit shape.
        """
        return self.value in {
            self.MESH,
            self.CONVEX,
            self.HFIELD,
        }

    @property
    def num_params(self) -> int:
        """
        Returns the number of parameters that describe the shape type.
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
        elif self.value in {self.MESH, self.CONVEX, self.HFIELD}:
            return -1  # Indicates variable number of parameters
        else:
            raise ValueError(f"Unknown shape type value: {self.value}")

    @staticmethod
    def to_newton(shape_type: ShapeType, shape_params: ShapeParamsLike | None = None) -> tuple[GeoType, vec3f | None]:
        """
        Converts Kamino :class:`ShapeType` Newton :class:`GeoType`, and
        optionally converts shape parameters to Newton shape scale.

        Shape parameter formats:
        - BOX:
            - Newton: half-extents as `scale := (x, y, z)`
            - Kamino: dimensions as `params := (depth, width, height, _)`
        - SPHERE:
            - Newton: radius as `scale := (radius, _, _)`
            - Kamino: radius as `params := (radius, _, _, _)`
        - CAPSULE:
            - Newton: radius and half-height as `scale := (radius, half_height, _)`
            - Kamino: radius and height as `params := (radius, height, _, _)`
        - CYLINDER:
            - Newton: radius and half-height as `scale := (radius, half_height, _)`
            - Kamino: radius and height as `params := (radius, height, _, _)`
        - CONE:
            - Newton: radius and half-height as `scale := (radius, half_height, _)`
            - Kamino: radius and height as `params := (radius, height, _, _)`
        - ELLIPSOID:
            - Newton: semi-axes as `scale := (x, y, z)`
            - Kamino: radii as `params := (a, b, c, _)`
        - PLANE:
            - Newton: half-width in x, half-length in y
            - Kamino: normal and distance as `params := (normal_x, normal_y, normal_z, distance)`

        See :class:`GenericShapeData` in :file:`support_function.py` for further details.

        Args:
            shape_params(`ShapeParamsLike`, optional):
                Kamino shape parameters as an iterable of floats.\n
                Expected formats per shape type are described above.\n
                If not `None`, this argument must have the expected
                number of parameters for the given shape type.

        Returns:
            (`GeoType`, `vec3f | None`):
                A tuple containing the corresponding Newton :class:`GeoType` and the shape scale as a :class:`vec3f`.

        Raises:
            ValueError:
                If the shape type cannot be mapped to a Newton GeoType, or
                if the provided parameters are invalid for the shape type.
        """
        # First attempt to convert the current shape
        # type to the corresponding Newton GeoType
        _MAP_TO_NEWTON: dict[ShapeType, GeoType] = {
            ShapeType.EMPTY: GeoType.NONE,
            ShapeType.SPHERE: GeoType.SPHERE,
            ShapeType.CYLINDER: GeoType.CYLINDER,
            ShapeType.CONE: GeoType.CONE,
            ShapeType.CAPSULE: GeoType.CAPSULE,
            ShapeType.BOX: GeoType.BOX,
            ShapeType.ELLIPSOID: GeoType.ELLIPSOID,
            ShapeType.PLANE: GeoType.PLANE,
            ShapeType.MESH: GeoType.MESH,
            ShapeType.CONVEX: GeoType.CONVEX_MESH,
            ShapeType.HFIELD: GeoType.HFIELD,
        }
        geo_type = _MAP_TO_NEWTON.get(shape_type, None)
        if geo_type is None:
            raise ValueError(f"Unsupported mapping to `newton.GeoType` from shape type: {shape_type}")

        # Then, and if parameters are provided, attempt to convert the
        # geometry parameters to the corresponding Newton shape scale
        shape_scale = None
        if shape_params is not None:
            # Ensure params is either a single float or an iterable of floats
            # with the expected number of parameters for the shape type
            if isinstance(shape_params, float):
                shape_params = [shape_params]
            elif not isinstance(shape_params, Iterable):
                raise ValueError(f"Invalid parameters type: {type(shape_params)}")
            elif len(shape_params) != shape_type.num_params:
                raise ValueError(
                    f"Invalid number of parameters for shape type {shape_type}: "
                    f"expected {shape_type.num_params}, got {len(shape_params)}"
                )
            # Convert the parameters to the corresponding Newton shape scale based on the shape type
            match shape_type:
                case ShapeType.SPHERE:
                    # Kamino: (radius, 0, 0, 0) -> Newton: (radius, ?, ?)
                    shape_scale = vec3f(shape_params[0], 0.0, 0.0)
                case ShapeType.BOX:
                    # Kamino: (depth, width, height) full size -> Newton: half-extents
                    shape_scale = vec3f(shape_params[0] * 0.5, shape_params[1] * 0.5, shape_params[2] * 0.5)
                case ShapeType.CAPSULE:
                    # Kamino: (radius, height) full height -> Newton: (radius, half-height, ?)
                    shape_scale = vec3f(shape_params[0], shape_params[1] * 0.5, 0.0)
                case ShapeType.CYLINDER:
                    # Kamino: (radius, height) full height -> Newton: (radius, half-height, ?)
                    shape_scale = vec3f(shape_params[0], shape_params[1] * 0.5, 0.0)
                case ShapeType.CONE:
                    # Kamino: (radius, height) full height -> Newton: (radius, half-height, ?)
                    shape_scale = vec3f(shape_params[0], shape_params[1] * 0.5, 0.0)
                case ShapeType.ELLIPSOID:
                    # Kamino: (a, b, c) semi-axes -> Newton: same
                    shape_scale = vec3f(shape_params[0], shape_params[1], shape_params[2])
                case ShapeType.PLANE:
                    # NOTE: For an infinite plane, we use (0, 0, _) to signal an infinite extents
                    shape_scale = vec3f(0.0, 0.0, 0.0)  # Infinite plane
                case ShapeType.MESH | ShapeType.CONVEX | ShapeType.HFIELD:
                    shape_scale = vec3f(shape_params[0], shape_params[1], shape_params[2])
                case _:
                    raise ValueError(f"Unsupported `ShapeType` for parameter conversion: {shape_type}")

        # Return the mapped GeoType and the converted scale (if applicable)
        return geo_type, shape_scale

    @staticmethod
    def from_newton(geo_type: GeoType, shape_scale: vec3f | None = None) -> tuple[ShapeType, vec4f | None]:
        """
        Converts Newton :class:`GeoType` to Kamino :class:`ShapeType`, and
        optionally converts Newton shape scale to Kamino geometry parameters.

        Shape parameter formats:
        - BOX:
            - Newton: half-extents as `scale := (x, y, z)`
            - Kamino: dimensions as `params := (depth, width, height, _)`
        - SPHERE:
            - Newton: radius as `scale := (radius, _, _)`
            - Kamino: radius as `params := (radius, _, _, _)`
        - CAPSULE:
            - Newton: radius and half-height as `scale := (radius, half_height, _)`
            - Kamino: radius and height as `params := (radius, height, _, _)`
        - CYLINDER:
            - Newton: radius and half-height as `scale := (radius, half_height, _)`
            - Kamino: radius and height as `params := (radius, height, _, _)`
        - CONE:
            - Newton: radius and half-height as `scale := (radius, half_height, _)`
            - Kamino: radius and height as `params := (radius, height, _, _)`
        - ELLIPSOID:
            - Newton: semi-axes as `scale := (x, y, z)`
            - Kamino: radii as `params := (a, b, c, _)`
        - PLANE:
            - Newton: half-width in x, half-length in y
            - Kamino: normal and distance as `params := (normal_x, normal_y, normal_z, distance)`

        See :class:`GenericShapeData` in :file:`support_function.py` for further details.

        Args:
            geo_type (GeoType):
                The Newton GeoType as :class:`GeoType`, i.e. the shape geometry type.
            shape_scale (vec3f | None):
                Newton shape scale as an iterable of floats.\n
                Expected formats per shape type are described above.\n
                If not `None`, this argument must be of type :class:`vec3f`.

        Returns:
            (ShapeType, vec4f):
            A tuple containing the corresponding Kamino :class:`ShapeType` and parameters as :class:`vec4f`.
        """
        # First attempt to convert the newton.GeoType
        # to the corresponding Kamino ShapeType
        _MAP_TO_KAMINO: dict[GeoType, ShapeType] = {
            GeoType.NONE: ShapeType.EMPTY,
            GeoType.SPHERE: ShapeType.SPHERE,
            GeoType.CYLINDER: ShapeType.CYLINDER,
            GeoType.CONE: ShapeType.CONE,
            GeoType.CAPSULE: ShapeType.CAPSULE,
            GeoType.BOX: ShapeType.BOX,
            GeoType.ELLIPSOID: ShapeType.ELLIPSOID,
            GeoType.PLANE: ShapeType.PLANE,
            GeoType.MESH: ShapeType.MESH,
            GeoType.CONVEX_MESH: ShapeType.CONVEX,
            GeoType.HFIELD: ShapeType.HFIELD,
        }
        shape_type = _MAP_TO_KAMINO.get(geo_type, None)
        if shape_type is None:
            raise ValueError(f"Unsupported mapping to `ShapeType` from newton.GeoType: {geo_type}")

        # Then, and if parameters are provided, attempt to convert the
        # geometry parameters to the corresponding Newton shape scale
        shape_params = None
        if shape_scale is not None:
            # Ensure shape_scale is an iterable of floats with the expected number of parameters for the shape type
            if not isinstance(shape_scale, vec3f):
                raise ValueError(f"Invalid shape_scale type: {type(shape_scale)}")
            # Convert the Newton shape scale to the corresponding
            # Kamino geometry parameters based on the shape type
            match geo_type:
                case GeoType.SPHERE:
                    # Newton: (radius, ?, ?) -> Kamino: (radius, 0, 0, 0)
                    shape_params = vec4f(shape_scale[0], 0.0, 0.0, 0.0)
                case GeoType.BOX:
                    # Newton: half-extents -> Kamino: (depth, width, height) full size
                    shape_params = vec4f(shape_scale[0] * 2.0, shape_scale[1] * 2.0, shape_scale[2] * 2.0, 0.0)
                case GeoType.CAPSULE:
                    # Newton: (radius, half-height, ?) -> Kamino: (radius, height, _, _)
                    shape_params = vec4f(shape_scale[0], shape_scale[1] * 2.0, 0.0, 0.0)
                case GeoType.CYLINDER:
                    # Newton: (radius, half-height, ?) -> Kamino: (radius, height, _, _)
                    shape_params = vec4f(shape_scale[0], shape_scale[1] * 2.0, 0.0, 0.0)
                case GeoType.CONE:
                    # Newton: (radius, half-height, ?) -> Kamino: (radius, height, _, _)
                    shape_params = vec4f(shape_scale[0], shape_scale[1] * 2.0, 0.0, 0.0)
                case GeoType.ELLIPSOID:
                    # Newton: (a, b, c) semi-axes -> Kamino: (a, b, c, _)
                    shape_params = vec4f(shape_scale[0], shape_scale[1], shape_scale[2], 0.0)
                case GeoType.PLANE:
                    # NOTE: For an infinite plane, we use (0, 0, _) to signal an infinite extents
                    shape_params = vec4f(0.0, 0.0, 1.0, 0.0)  # Default normal and distance
                case GeoType.MESH | GeoType.CONVEX_MESH | GeoType.HFIELD:
                    # For mesh, convex mesh, and heightfield, parameters are not directly convertible
                    shape_params = vec4f(shape_scale[0], shape_scale[1], shape_scale[2], 0.0)
                case _:
                    raise ValueError(f"Unsupported `GeoType` for parameter conversion: {geo_type}")

        # Return the mapped ShapeType and the
        # converted parameters (if applicable)
        return shape_type, shape_params


ShapeParamsLike = None | float | Iterable[float]
"""A type union that can represent any shape parameters, including None, single float, or iterable of floats."""

ShapeDataLike = None | Mesh | Heightfield
"""A type union that can represent any shape data, including None, Mesh, and Heightfield."""


class ShapeDescriptor(ABC, Descriptor):
    """Abstract base class for all shape descriptors."""

    def __init__(self, type: ShapeType, name: str = "", uid: str | None = None):
        """
        Initialize the shape descriptor.

        Args:
            type (ShapeType): The type of the shape.
            name (str): The name of the shape descriptor.
            uid (str | None): Optional unique identifier of the shape descriptor.
        """
        super().__init__(name, uid)
        self._type: ShapeType = type

    @override
    def __hash__(self) -> int:
        """Returns a hash of the ShapeDescriptor based on its name, uid, type and params."""
        return hash((self.type, self.params))

    @override
    def __repr__(self):
        """Returns a human-readable string representation of the ShapeDescriptor."""
        return f"ShapeDescriptor(\ntype: {self.type},\nname: {self.name},\nuid: {self.uid},\n)"

    @property
    def type(self) -> ShapeType:
        """Returns the type of the shape."""
        return self._type

    @property
    def num_params(self) -> int:
        """Returns the number of parameters that describe the shape."""
        return self._type.num_params

    @property
    def is_solid(self) -> bool:
        """Returns whether the shape is solid (i.e., not empty)."""
        # TODO: Add support for other non-solid shapes if necessary
        return self._type != ShapeType.EMPTY

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


###
# Primitive Shapes
###


class EmptyShape(ShapeDescriptor):
    """
    A shape descriptor for the empty shape that can serve as a placeholder.
    """

    def __init__(self, name: str = "empty", uid: str | None = None):
        super().__init__(ShapeType.EMPTY, name, uid)

    @override
    def __repr__(self):
        """Returns a human-readable string representation of the EmptyShape."""
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
    """
    A shape descriptor for spheres.

    Attributes:
        radius (float): The radius of the sphere.
    """

    def __init__(self, radius: float, name: str = "sphere", uid: str | None = None):
        super().__init__(ShapeType.SPHERE, name, uid)
        self.radius: float = radius

    @override
    def __repr__(self):
        """Returns a human-readable string representation of the SphereShape."""
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
    """
    A shape descriptor for cylinders.

    Attributes:
        radius (float): The radius of the cylinder.
        height (float): The height of the cylinder.
    """

    def __init__(self, radius: float, height: float, name: str = "cylinder", uid: str | None = None):
        super().__init__(ShapeType.CYLINDER, name, uid)
        self.radius: float = radius
        self.height: float = height

    @override
    def __repr__(self):
        """Returns a human-readable string representation of the CylinderShape."""
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
    """
    A shape descriptor for cones.

    Attributes:
        radius (float): The radius of the cone.
        height (float): The height of the cone.
    """

    def __init__(self, radius: float, height: float, name: str = "cone", uid: str | None = None):
        super().__init__(ShapeType.CONE, name, uid)
        self.radius: float = radius
        self.height: float = height

    @override
    def __repr__(self):
        """Returns a human-readable string representation of the ConeShape."""
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
    """
    A shape descriptor for capsules.

    Attributes:
        radius (float): The radius of the capsule.
        height (float): The height of the capsule.
    """

    def __init__(self, radius: float, height: float, name: str = "capsule", uid: str | None = None):
        super().__init__(ShapeType.CAPSULE, name, uid)
        self.radius: float = radius
        self.height: float = height

    @override
    def __repr__(self):
        """Returns a human-readable string representation of the CapsuleShape."""
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
    """
    A shape descriptor for boxes.

    Attributes:
        depth (float): The depth of the box, defined along the local X-axis.
        width (float): The width of the box, defined along the local Y-axis.
        height (float): The height of the box, defined along the local Z-axis.
    """

    def __init__(self, depth: float, width: float, height: float, name: str = "box", uid: str | None = None):
        super().__init__(ShapeType.BOX, name, uid)
        self.depth: float = depth
        self.width: float = width
        self.height: float = height

    @override
    def __repr__(self):
        """Returns a human-readable string representation of the BoxShape."""
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
    """
    A shape descriptor for ellipsoids.

    Attributes:
        a (float): The semi-axis length along the X-axis.
        b (float): The semi-axis length along the Y-axis.
        c (float): The semi-axis length along the Z-axis.
    """

    def __init__(self, a: float, b: float, c: float, name: str = "ellipsoid", uid: str | None = None):
        super().__init__(ShapeType.ELLIPSOID, name, uid)
        self.a: float = a
        self.b: float = b
        self.c: float = c

    @override
    def __repr__(self):
        """Returns a human-readable string representation of the EllipsoidShape."""
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
    """
    A shape descriptor for planes.

    Attributes:
        normal (Vec3): The normal vector of the plane.
        distance (float): The distance from the origin to the plane along its normal.
    """

    def __init__(self, normal: Vec3, distance: float, name: str = "plane", uid: str | None = None):
        super().__init__(ShapeType.PLANE, name, uid)
        self.normal: Vec3 = normal
        self.distance: float = distance

    @override
    def __repr__(self):
        """Returns a human-readable string representation of the PlaneShape."""
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

    Attributes:
        vertices (nparray): The vertices of the mesh.
        indices (nparray): The triangle indices of the mesh.
        normals (nparray | None): The vertex normals of the mesh.
        uvs (nparray | None): The texture coordinates of the mesh.
        color (Vec3 | None): The color of the mesh.
        is_solid (bool): Whether the mesh is solid.
        is_convex (bool): Whether the mesh is convex.
    """

    MAX_HULL_VERTICES = Mesh.MAX_HULL_VERTICES
    """Utility attribute to expose this constant without needing to import the newton.Mesh class directly."""

    def __init__(
        self,
        vertices: Sequence[Vec3] | nparray,
        indices: Sequence[int] | nparray,
        normals: Sequence[Vec3] | nparray | None = None,
        uvs: Sequence[Vec2] | nparray | None = None,
        color: Vec3 | None = None,
        maxhullvert: int | None = None,
        compute_inertia: bool = True,
        is_solid: bool = True,
        is_convex: bool = False,
        name: str = "mesh",
        uid: str | None = None,
    ):
        """
        Initialize the mesh shape descriptor.

        Args:
            vertices (Sequence[Vec3] | nparray): The vertices of the mesh.
            indices (Sequence[int] | nparray): The triangle indices of the mesh.
            normals (Sequence[Vec3] | nparray | None): The vertex normals of the mesh.
            uvs (Sequence[Vec2] | nparray | None): The texture coordinates of the mesh.
            color (Vec3 | None): The color of the mesh.
            maxhullvert (int): The maximum number of hull vertices for convex shapes.
            compute_inertia (bool): Whether to compute inertia for the mesh.
            is_solid (bool): Whether the mesh is solid.
            is_convex (bool): Whether the mesh is convex.
            name (str): The name of the shape descriptor.
            uid (str | None): Optional unique identifier of the shape descriptor.
        """
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
    def __hash__(self) -> int:
        """Returns a hash computed using the underlying newton.Mesh hash implementation."""
        return self._data.__hash__()

    @override
    def __repr__(self):
        """Returns a human-readable string representation of the MeshShape."""
        label = "ConvexShape" if self.type == ShapeType.CONVEX else "MeshShape"
        normals_shape = self._data._normals.shape if self._data._normals is not None else None
        return (
            f"{label}(\n"
            f"name: {self.name},\n"
            f"uid: {self.uid},\n"
            f"vertices: {self._data.vertices.shape},\n"
            f"indices: {self._data.indices.shape},\n"
            f"normals: {normals_shape},\n"
            f")"
        )

    @property
    @override
    def paramsvec(self) -> vec4f:
        return vec4f(1.0, 1.0, 1.0, 0.0)

    @property
    @override
    def params(self) -> tuple[float, float, float]:
        """Returns the XYZ scaling of the mesh."""
        return 1.0, 1.0, 1.0

    @property
    @override
    def data(self) -> Mesh:
        return self._data

    @property
    def vertices(self) -> nparray:
        """Returns the vertices of the mesh."""
        return self._data.vertices

    @property
    def indices(self) -> nparray:
        """Returns the indices of the mesh."""
        return self._data.indices

    @property
    def normals(self) -> nparray | None:
        """Returns the normals of the mesh."""
        return self._data._normals

    @property
    def uvs(self) -> nparray | None:
        """Returns the UVs of the mesh."""
        return self._data._uvs

    @property
    def color(self) -> Vec3 | None:
        """Returns the color of the mesh."""
        return self._data._color


class HFieldShape(ShapeDescriptor):
    """
    A shape descriptor for height-field shapes.

    WARNING: This class is not yet implemented.
    """

    def __init__(self, name: str = "hfield", uid: str | None = None):
        super().__init__(ShapeType.HFIELD, name, uid)
        # TODO: Remove this when HFieldShape is implemented
        raise NotImplementedError("HFieldShape is not yet implemented.")

    @override
    def __repr__(self):
        return f"HFieldShape(\nname: {self.name},\nuid: {self.uid}\n)"

    @property
    @override
    def paramsvec(self) -> vec4f:
        return vec4f(1.0, 1.0, 1.0, 0.0)

    @property
    @override
    def params(self) -> tuple[float, float, float]:
        """Returns the XYZ scaling of the height-field."""
        return 1.0, 1.0, 1.0


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
)
"""A type union that can represent any shape descriptor, including primitive and explicit shapes."""


###
# Utilities
###


def max_contacts_for_shape_pair(type_a: int, type_b: int) -> tuple[int, int]:
    """
    Count the number of potential contact points for a collision pair in both
    directions of the collision pair (collisions from A to B and from B to A).

    Inputs must be canonicalized such that the type of shape A is less than or equal to the type of shape B.

    Args:
        type_a: First shape type
        type_b: Second shape type

    Returns:
        tuple[int, int]: Number of contact points for collisions between A->B and B->A.
    """
    # Ensure the shape types are ordered canonically
    if type_a > type_b:
        type_a, type_b = type_b, type_a

    if type_a == ShapeType.SPHERE:
        return 1, 0

    elif type_a == ShapeType.CYLINDER:
        if type_b == ShapeType.CYLINDER:
            return 4, 4
        elif type_b == ShapeType.CONE:
            return 4, 4
        elif type_b == ShapeType.CAPSULE:
            return 4, 4
        elif type_b == ShapeType.BOX:
            return 8, 8
        elif type_b == ShapeType.ELLIPSOID:
            return 4, 4
        elif type_b == ShapeType.PLANE:
            return 6, 6
        elif type_b == ShapeType.MESH or type_b == ShapeType.CONVEX:
            pass  # TODO: WHAT TO RETURN WHEN MESH SUPPORT IS ADDED?

    elif type_a == ShapeType.CONE:
        if type_b == ShapeType.CONE:
            return 4, 4
        elif type_b == ShapeType.CAPSULE:
            return 4, 4
        elif type_b == ShapeType.BOX:
            return 8, 8
        elif type_b == ShapeType.ELLIPSOID:
            return 8, 8
        elif type_b == ShapeType.PLANE:
            return 8, 8
        elif type_b == ShapeType.MESH or type_b == ShapeType.CONVEX:
            pass  # TODO: WHAT TO RETURN WHEN MESH SUPPORT IS ADDED?

    elif type_a == ShapeType.CAPSULE:
        if type_b == ShapeType.CAPSULE:
            return 2, 2
        elif type_b == ShapeType.BOX:
            return 8, 8
        elif type_b == ShapeType.ELLIPSOID:
            return 8, 8
        elif type_b == ShapeType.PLANE:
            return 8, 8
        elif type_b == ShapeType.MESH or type_b == ShapeType.CONVEX:
            pass  # TODO: WHAT TO RETURN WHEN MESH SUPPORT IS ADDED?

    elif type_a == ShapeType.BOX:
        if type_b == ShapeType.BOX:
            return 12, 12
        elif type_b == ShapeType.ELLIPSOID:
            return 8, 8
        elif type_b == ShapeType.PLANE:
            return 12, 12
        elif type_b == ShapeType.MESH or type_b == ShapeType.CONVEX:
            pass  # TODO: WHAT TO RETURN WHEN MESH SUPPORT IS ADDED?

    elif type_a == ShapeType.ELLIPSOID:
        if type_b == ShapeType.ELLIPSOID:
            return 4, 4
        elif type_b == ShapeType.PLANE:
            return 4, 4
        elif type_b == ShapeType.MESH or type_b == ShapeType.CONVEX:
            pass  # TODO: WHAT TO RETURN WHEN MESH SUPPORT IS ADDED?

    elif type_a == ShapeType.PLANE:
        if type_b == ShapeType.MESH or type_b == ShapeType.CONVEX:
            pass  # TODO: WHAT TO RETURN WHEN MESH SUPPORT IS ADDED?

    elif type_a == ShapeType.MESH or type_a == ShapeType.CONVEX:
        if type_a == ShapeType.HFIELD:
            pass  # TODO: WHAT TO RETURN WHEN MESH SUPPORT IS ADDED?
        else:
            pass  # TODO: WHAT TO RETURN WHEN MESH SUPPORT IS ADDED?

    # unsupported type combination
    return 0, 0
