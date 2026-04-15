# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""KAMINO: Shape Types & Containers"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence

import numpy as np
import warp as wp

from .....core.types import Vec2, Vec3
from .....geometry.types import GeoType, Heightfield, Mesh
from .types import Descriptor, override, vec4f

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
    "GeoType",
    "MeshShape",
    "PlaneShape",
    "ShapeDescriptor",
    "SphereShape",
]


###
# Module configs
###

wp.set_module_options({"enable_backward": False})


###
# Containers
###


ShapeDataLike = None | Mesh | Heightfield
"""A type union that can represent any shape data, including None, Mesh, and Heightfield."""


def is_primitive_geo_type(geo_type: GeoType) -> bool:
    """Return whether the geo type is a primitive shape.

    .. deprecated::
        Use :attr:`GeoType.is_primitive` instead.
    """
    return geo_type.is_primitive


def is_explicit_geo_type(geo_type: GeoType) -> bool:
    """Return whether the geo type is an explicit shape (mesh, convex, heightfield).

    .. deprecated::
        Use :attr:`GeoType.is_explicit` instead.
    """
    return geo_type.is_explicit


class ShapeDescriptor(ABC, Descriptor):
    """Abstract base class for all shape descriptors."""

    def __init__(self, geo_type: GeoType, name: str = "", uid: str | None = None):
        """
        Initialize the shape descriptor.

        Args:
            geo_type: The geometry type from Newton's :class:`GeoType`.
            name: The name of the shape descriptor.
            uid: Optional unique identifier of the shape descriptor.
        """
        super().__init__(name, uid)
        self._type: GeoType = geo_type

    @override
    def __hash__(self) -> int:
        """Returns a hash of the ShapeDescriptor based on its name, uid, type and params."""
        return hash((self.type, self.params))

    @override
    def __repr__(self):
        """Returns a human-readable string representation of the ShapeDescriptor."""
        return f"ShapeDescriptor(\ntype: {self.type},\nname: {self.name},\nuid: {self.uid},\n)"

    @property
    def type(self) -> GeoType:
        """Returns the geometry type of the shape."""
        return self._type

    @property
    def is_solid(self) -> bool:
        """Returns whether the shape is solid (i.e., not empty)."""
        return self._type != GeoType.NONE

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
        super().__init__(GeoType.NONE, name, uid)

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
        radius: The radius of the sphere [m].
    """

    def __init__(self, radius: float, name: str = "sphere", uid: str | None = None):
        super().__init__(GeoType.SPHERE, name, uid)
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
        radius: The radius of the cylinder [m].
        half_height: The half-height of the cylinder [m].
    """

    def __init__(self, radius: float, half_height: float, name: str = "cylinder", uid: str | None = None):
        super().__init__(GeoType.CYLINDER, name, uid)
        self.radius: float = radius
        self.half_height: float = half_height

    @override
    def __repr__(self):
        """Returns a human-readable string representation of the CylinderShape."""
        return f"CylinderShape(\nname: {self.name},\nuid: {self.uid},\nradius: {self.radius},\nhalf_height: {self.half_height}\n)"

    @property
    @override
    def paramsvec(self) -> vec4f:
        return vec4f(self.radius, self.half_height, 0.0, 0.0)

    @property
    @override
    def params(self) -> tuple[float, float]:
        return (self.radius, self.half_height)

    @property
    @override
    def data(self) -> None:
        return None


class ConeShape(ShapeDescriptor):
    """
    A shape descriptor for cones.

    Attributes:
        radius: The radius of the cone [m].
        half_height: The half-height of the cone [m].
    """

    def __init__(self, radius: float, half_height: float, name: str = "cone", uid: str | None = None):
        super().__init__(GeoType.CONE, name, uid)
        self.radius: float = radius
        self.half_height: float = half_height

    @override
    def __repr__(self):
        """Returns a human-readable string representation of the ConeShape."""
        return f"ConeShape(\nname: {self.name},\nuid: {self.uid},\nradius: {self.radius},\nhalf_height: {self.half_height}\n)"

    @property
    @override
    def paramsvec(self) -> vec4f:
        return vec4f(self.radius, self.half_height, 0.0, 0.0)

    @property
    @override
    def params(self) -> tuple[float, float]:
        return (self.radius, self.half_height)

    @property
    @override
    def data(self) -> None:
        return None


class CapsuleShape(ShapeDescriptor):
    """
    A shape descriptor for capsules.

    Attributes:
        radius: The radius of the capsule [m].
        half_height: The half-height of the capsule (cylindrical part) [m].
    """

    def __init__(self, radius: float, half_height: float, name: str = "capsule", uid: str | None = None):
        super().__init__(GeoType.CAPSULE, name, uid)
        self.radius: float = radius
        self.half_height: float = half_height

    @override
    def __repr__(self):
        """Returns a human-readable string representation of the CapsuleShape."""
        return f"CapsuleShape(\nname: {self.name},\nuid: {self.uid},\nradius: {self.radius},\nhalf_height: {self.half_height}\n)"

    @property
    @override
    def paramsvec(self) -> vec4f:
        return vec4f(self.radius, self.half_height, 0.0, 0.0)

    @property
    @override
    def params(self) -> tuple[float, float]:
        return (self.radius, self.half_height)

    @property
    @override
    def data(self) -> None:
        return None


class BoxShape(ShapeDescriptor):
    """
    A shape descriptor for boxes.

    Attributes:
        hx: The half-extent along the local X-axis [m].
        hy: The half-extent along the local Y-axis [m].
        hz: The half-extent along the local Z-axis [m].
    """

    def __init__(self, hx: float, hy: float, hz: float, name: str = "box", uid: str | None = None):
        super().__init__(GeoType.BOX, name, uid)
        self.hx: float = hx
        self.hy: float = hy
        self.hz: float = hz

    @override
    def __repr__(self):
        """Returns a human-readable string representation of the BoxShape."""
        return f"BoxShape(\nname: {self.name},\nuid: {self.uid},\nhx: {self.hx},\nhy: {self.hy},\nhz: {self.hz}\n)"

    @property
    @override
    def paramsvec(self) -> vec4f:
        return vec4f(self.hx, self.hy, self.hz, 0.0)

    @property
    @override
    def params(self) -> tuple[float, float, float]:
        return (self.hx, self.hy, self.hz)

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
        super().__init__(GeoType.ELLIPSOID, name, uid)
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
        super().__init__(GeoType.PLANE, name, uid)
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
        vertices (np.ndarray): The vertices of the mesh.
        indices (np.ndarray): The triangle indices of the mesh.
        normals (np.ndarray | None): The vertex normals of the mesh.
        uvs (np.ndarray | None): The texture coordinates of the mesh.
        color (Vec3 | None): The color of the mesh.
        is_solid (bool): Whether the mesh is solid.
        is_convex (bool): Whether the mesh is convex.
    """

    MAX_HULL_VERTICES = Mesh.MAX_HULL_VERTICES
    """Utility attribute to expose this constant without needing to import the newton.Mesh class directly."""

    def __init__(
        self,
        vertices: Sequence[Vec3] | np.ndarray,
        indices: Sequence[int] | np.ndarray,
        normals: Sequence[Vec3] | np.ndarray | None = None,
        uvs: Sequence[Vec2] | np.ndarray | None = None,
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
            vertices (Sequence[Vec3] | np.ndarray): The vertices of the mesh.
            indices (Sequence[int] | np.ndarray): The triangle indices of the mesh.
            normals (Sequence[Vec3] | np.ndarray | None): The vertex normals of the mesh.
            uvs (Sequence[Vec2] | np.ndarray | None): The texture coordinates of the mesh.
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
            geo_type = GeoType.CONVEX_MESH
            name = "convex" if name == "mesh" else name
        else:
            geo_type = GeoType.MESH

        # Initialize the base shape descriptor
        super().__init__(geo_type, name, uid)

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
        label = "ConvexShape" if self.type == GeoType.CONVEX_MESH else "MeshShape"
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
    def vertices(self) -> np.ndarray:
        """Returns the vertices of the mesh."""
        return self._data.vertices

    @property
    def indices(self) -> np.ndarray:
        """Returns the indices of the mesh."""
        return self._data.indices

    @property
    def normals(self) -> np.ndarray | None:
        """Returns the normals of the mesh."""
        return self._data._normals

    @property
    def uvs(self) -> np.ndarray | None:
        """Returns the UVs of the mesh."""
        return self._data._uvs

    @property
    def color(self) -> Vec3 | None:
        """Returns the color of the mesh."""
        return self._data._color


class HFieldShape(ShapeDescriptor):
    """A shape descriptor for height-field (terrain) shapes.

    Attributes:
        heightfield: The underlying :class:`Heightfield` data.
    """

    def __init__(self, heightfield: Heightfield, name: str = "hfield", uid: str | None = None):
        """Initialize the height-field shape descriptor.

        Args:
            heightfield: A :class:`Heightfield` instance containing elevation data.
            name: The name of the shape descriptor.
            uid: Optional unique identifier of the shape descriptor.
        """
        super().__init__(GeoType.HFIELD, name, uid)
        self._data: Heightfield = heightfield

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

    @property
    @override
    def data(self) -> Heightfield:
        return self._data


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
        type_a: First shape type as :class:`GeoType` integer value.
        type_b: Second shape type as :class:`GeoType` integer value.

    Returns:
        tuple[int, int]: Number of contact points for collisions between A->B and B->A.
    """
    # Ensure the shape types are ordered canonically
    if type_a > type_b:
        type_a, type_b = type_b, type_a

    # Contact counts for mesh/heightfield pairs are dynamic (bounded by the
    # pipeline's max_contacts_per_pair setting).  The values below are
    # conservative upper-bound estimates used for capacity allocation.
    _MESH_CONVEX_MAX = 32
    _MESH_MESH_MAX = 64

    if type_a == GeoType.SPHERE:
        return 1, 0

    elif type_a == GeoType.CAPSULE:
        if type_b == GeoType.CAPSULE:
            return 2, 2
        elif type_b == GeoType.ELLIPSOID:
            return 8, 8
        elif type_b == GeoType.CYLINDER:
            return 4, 4
        elif type_b == GeoType.BOX:
            return 8, 8
        elif type_b == GeoType.MESH or type_b == GeoType.CONVEX_MESH:
            return _MESH_CONVEX_MAX, 0
        elif type_b == GeoType.CONE:
            return 4, 4
        elif type_b == GeoType.PLANE:
            return 8, 8

    elif type_a == GeoType.ELLIPSOID:
        if type_b == GeoType.ELLIPSOID:
            return 4, 4
        elif type_b == GeoType.CYLINDER:
            return 4, 4
        elif type_b == GeoType.BOX:
            return 8, 8
        elif type_b == GeoType.MESH or type_b == GeoType.CONVEX_MESH:
            return _MESH_CONVEX_MAX, 0
        elif type_b == GeoType.CONE:
            return 8, 8
        elif type_b == GeoType.PLANE:
            return 4, 4

    elif type_a == GeoType.CYLINDER:
        if type_b == GeoType.CYLINDER:
            return 4, 4
        elif type_b == GeoType.BOX:
            return 8, 8
        elif type_b == GeoType.MESH or type_b == GeoType.CONVEX_MESH:
            return _MESH_CONVEX_MAX, 0
        elif type_b == GeoType.CONE:
            return 4, 4
        elif type_b == GeoType.PLANE:
            return 6, 6

    elif type_a == GeoType.BOX:
        if type_b == GeoType.BOX:
            return 12, 12
        elif type_b == GeoType.MESH or type_b == GeoType.CONVEX_MESH:
            return _MESH_CONVEX_MAX, 0
        elif type_b == GeoType.CONE:
            return 8, 8
        elif type_b == GeoType.PLANE:
            return 12, 12

    elif type_a == GeoType.MESH or type_a == GeoType.CONVEX_MESH:
        if type_b == GeoType.HFIELD:
            return _MESH_MESH_MAX, 0
        elif type_b == GeoType.CONE:
            return _MESH_CONVEX_MAX, 0
        elif type_b == GeoType.PLANE:
            return _MESH_CONVEX_MAX, 0
        else:
            return _MESH_MESH_MAX, 0

    elif type_a == GeoType.HFIELD:
        # Heightfield vs convex primitives
        return _MESH_CONVEX_MAX, 0

    elif type_a == GeoType.CONE:
        if type_b == GeoType.CONE:
            return 4, 4
        elif type_b == GeoType.PLANE:
            return 8, 8

    elif type_a == GeoType.PLANE:
        pass

    # unsupported type combination
    return 0, 0
