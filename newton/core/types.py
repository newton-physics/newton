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
SHAPE_FLAG_COLLIDE_PARTICLES = wp.constant(wp.uint32(1 << 3))

# Shape geometry types
GEO_SPHERE = wp.constant(0)
GEO_BOX = wp.constant(1)
GEO_CAPSULE = wp.constant(2)
GEO_CYLINDER = wp.constant(3)
GEO_CONE = wp.constant(4)
GEO_MESH = wp.constant(5)
GEO_SDF = wp.constant(6)
GEO_PLANE = wp.constant(7)
GEO_NONE = wp.constant(8)


def get_shape_radius(geo_type: int, scale: Vec3, src: Mesh | SDF | None) -> float:
    """
    Calculates the radius of a sphere that encloses the shape, used for broadphase collision detection.
    """
    if geo_type == GEO_SPHERE:
        return scale[0]
    elif geo_type == GEO_BOX:
        return np.linalg.norm(scale)
    elif geo_type == GEO_CAPSULE or geo_type == GEO_CYLINDER or geo_type == GEO_CONE:
        return scale[0] + scale[1]
    elif geo_type == GEO_MESH:
        vmax = np.max(np.abs(src.vertices), axis=0) * np.max(scale)
        return np.linalg.norm(vmax)
    elif geo_type == GEO_PLANE:
        if scale[0] > 0.0 and scale[1] > 0.0:
            # finite plane
            return np.linalg.norm(scale)
        else:
            return 1.0e6
    else:
        return 10.0


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
class ModelShapeMaterials:
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
class ModelShapeGeometry:
    """
    Represents the geometry of a shape.
        type: The type of geometry (GEO_SPHERE, GEO_BOX, etc.)
        is_solid: Indicates whether the shape is solid or hollow
        thickness: The thickness of the shape (used for collision detection, and inertia computation of hollow shapes)
        source: Pointer to the source geometry (can be a mesh or SDF index, zero otherwise)
        scale: The 3D scale of the shape
    """

    type: wp.array(dtype=wp.int32)
    is_solid: wp.array(dtype=bool)
    thickness: wp.array(dtype=float)
    source: wp.array(dtype=wp.uint64)
    scale: wp.array(dtype=wp.vec3)


class SDF:
    """Describes a signed distance field for simulation

    Attributes:

        volume (Volume): The volume defining the SDF
        I (Mat33): 3x3 inertia matrix of the SDF
        mass (float): The total mass of the SDF
        com (Vec3): The center of mass of the SDF
    """

    def __init__(self, volume: wp.Volume | None = None, I=None, mass=1.0, com=None):
        self.volume = volume
        self.I = I if I is not None else wp.mat33(np.eye(3))
        self.mass = mass
        self.com = com if com is not None else wp.vec3()

        # Need to specify these for now
        self.has_inertia = True
        self.is_solid = True

    def finalize(self) -> wp.uint64:
        return self.volume.id

    @override
    def __hash__(self) -> int:
        return hash(self.volume.id)


class Mesh:
    """Describes a triangle collision mesh for simulation

    Example mesh creation from a triangle OBJ mesh file:
    ====================================================

    See :func:`load_mesh` which is provided as a utility function.

    .. code-block:: python

        import numpy as np
        import warp as wp
        import newton
        import openmesh

        m = openmesh.read_trimesh("mesh.obj")
        mesh_points = np.array(m.points())
        mesh_indices = np.array(m.face_vertex_indices(), dtype=np.int32).flatten()
        mesh = newton.Mesh(mesh_points, mesh_indices)

    Attributes:

        vertices (List[Vec3]): Mesh 3D vertices points
        indices (List[int]): Mesh indices as a flattened list of vertex indices describing triangles
        I (Mat33): 3x3 inertia matrix of the mesh assuming density of 1.0 (around the center of mass)
        mass (float): The total mass of the body assuming density of 1.0
        com (Vec3): The center of mass of the body
    """

    def __init__(self, vertices: Sequence[Vec3], indices: Sequence[int], compute_inertia=True, is_solid=True):
        """Construct a Mesh object from a triangle mesh

        The mesh center of mass and inertia tensor will automatically be
        calculated using a density of 1.0. This computation is only valid
        if the mesh is closed (two-manifold).

        Args:
            vertices: List of vertices in the mesh
            indices: List of triangle indices, 3 per-element
            compute_inertia: If True, the mass, inertia tensor and center of mass will be computed assuming density of 1.0
            is_solid: If True, the mesh is assumed to be a solid during inertia computation, otherwise it is assumed to be a hollow surface
        """
        from .inertia import compute_mesh_inertia

        self.vertices = np.array(vertices).reshape(-1, 3)
        self.indices = np.array(indices, dtype=np.int32).flatten()
        self.is_solid = is_solid
        self.has_inertia = compute_inertia
        self.mesh = None

        if compute_inertia:
            self.mass, self.com, self.I, _ = compute_mesh_inertia(1.0, vertices, indices, is_solid=is_solid)
        else:
            self.I = wp.mat33(np.eye(3))
            self.mass = 1.0
            self.com = wp.vec3()

    # construct simulation ready buffers from points
    def finalize(self, device: Devicelike = None, requires_grad: bool = False) -> wp.uint64:
        """
        Constructs a simulation-ready :class:`Mesh` object from the mesh data and returns its ID.

        Args:
            device: The device on which to allocate the mesh buffers
            requires_grad: If True, the mesh points and velocity arrays will be allocated with gradient tracking enabled

        Returns:
            The ID of the simulation-ready :class:`Mesh`
        """
        with wp.ScopedDevice(device):
            pos = wp.array(self.vertices, requires_grad=requires_grad, dtype=wp.vec3)
            vel = wp.zeros_like(pos)
            indices = wp.array(self.indices, dtype=wp.int32)

            self.mesh = wp.Mesh(points=pos, velocities=vel, indices=indices)
            return self.mesh.id

    @override
    def __hash__(self) -> int:
        """
        Computes a hash of the mesh data for use in caching. The hash considers the mesh vertices, indices, and whether the mesh is solid or not.
        """
        return hash((tuple(np.array(self.vertices).flatten()), tuple(np.array(self.indices).flatten()), self.is_solid))


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
    "GEO_BOX",
    "GEO_CAPSULE",
    "GEO_CONE",
    "GEO_CYLINDER",
    "GEO_MESH",
    "GEO_NONE",
    "GEO_PLANE",
    "GEO_SDF",
    "GEO_SPHERE",
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
    "SDF",
    "SHAPE_FLAG_COLLIDE_GROUND",
    "SHAPE_FLAG_COLLIDE_SHAPES",
    "SHAPE_FLAG_VISIBLE",
    "Axis",
    "AxisType",
    "Mat33",
    "Mesh",
    "ModelShapeGeometry",
    "ModelShapeMaterials",
    "Quat",
    "Sequence",
    "Transform",
    "Vec3",
    "Vec4",
    "flag_to_int",
    "get_joint_dof_count",
]
