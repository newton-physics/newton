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

"""
KAMINO: Geometry Model Types & Containers
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field

import warp as wp

from .bv import aabb_geom, bs_geom
from .shapes import ShapeDescriptorType
from .types import Descriptor, float32, int32, mat83f, override, transformf, vec4f

###
# Module interface
###

__all__ = [
    "CollisionGeometriesData",
    "CollisionGeometriesModel",
    "CollisionGeometryDescriptor",
    "GeometriesData",
    "GeometriesModel",
    "GeometryDescriptor",
]


###
# Module configs
###

wp.set_module_options({"enable_backward": False})


###
# Base Geometry Containers
###


@dataclass
class GeometryDescriptor(Descriptor):
    """
    A container to describe a generic geometry entity.

    A geometry entity is an abstraction to represent the composition of a
    shape, with a pose w.r.t the world frame of a scene. Each geometry entity
    is assigned to a single body, and is organized into one of the distinct
    geometry sets called 'layers'. Each geometry descriptor bundles the unique
    object identifiers of the entity, indices to the associated body and layer,
    the offset pose w.r.t. the body, and a shape descriptor.

    Attributes:
        bid (int): Index of the body to which the geometry entity is attached.\n
            Defaults to `-1`, indicating that the geometry has not yet been assigned to a body.\n
            The value `-1` also indicates that the geometry, by default, is statically attached to the world.
        shape (ShapeDescriptorType | None): Definition of the shape of the geometry entity.
            See :class:`ShapeDescriptorType` for the list of supported shape types.
        offset (transformf): Offset pose transform of the geometry entity w.r.t. its corresponding body.\n
            Defaults to the identity transform with zero translation and identity rotation quaternion.
        wid (int): Index of the world to which the body belongs.\n
            Defaults to `-1`, indicating that the body has not yet been added to a world.
        gid (int): Index of the geometry w.r.t. its world.\n
            Defaults to `-1`, indicating that the geometry has not yet been added to a world.
        lid (int): Index of the geometry layer to which the geometry entity is assigned.\n
            Defaults to `-1`, indicating that the geometry has not yet been assigned to a layer.
    """

    ###
    # Attributes
    ###

    bid: int = -1
    """
    Index of the body to which the geometry entity is attached.\n
    Defaults to `-1`, indicating that the geometry has not yet been assigned to a body.\n
    The value `-1` also indicates that the geometry, by default, is statically attached to the world.
    """

    layer: str = "default"
    """
    Name of the geometry layer to which the geometry is assigned.\n
    Defaults to the `default` layer.
    """

    shape: ShapeDescriptorType | None = None
    """Definition of the shape of the geometry entity of type :class:`ShapeDescriptorType`."""

    offset: transformf = field(default_factory=transformf)
    """Offset pose of the geometry entity w.r.t. its corresponding body, of type :class:`transformf`."""

    ###
    # Metadata - to be set by the WorldDescriptor when added
    ###

    wid: int = -1
    """
    Index of the world to which the body belongs.\n
    Defaults to `-1`, indicating that the body has not yet been added to a world.
    """

    gid: int = -1
    """
    Index of the geometry w.r.t. its world.\n
    Defaults to `-1`, indicating that the geometry has not yet been added to a world.
    """

    lid: int = -1
    """
    Index of the geometry layer to which the geometry entity is assigned.\n
    Defaults to `-1`, indicating that the geometry has not yet been assigned to a layer.
    """

    @override
    def __hash__(self):
        """Returns a hash computed using the shape descriptor's hash implementation."""
        # NOTE: The name-uid-based hash implementation is called if no shape is defined
        if self.shape is None:
            return super().__hash__()
        # Otherwise, use the shape's hash implementation
        return self.shape.__hash__()

    @override
    def __repr__(self):
        """Returns a human-readable string representation of the GeometryDescriptor."""
        return (
            f"GeometryDescriptor(\n"
            f"name: {self.name},\n"
            f"uid: {self.uid},\n"
            f"lid: {self.lid},\n"
            f"bid: {self.bid},\n"
            f"shape: {self.shape}\n"
            f"offset: {self.offset},\n"
            f"wid: {self.wid},\n"
            f"gid: {self.gid},\n"
            f")"
        )


@dataclass
class GeometriesModel:
    """
    An SoA-based container to hold time-invariant model data of a set of generic geometry elements.

    Attributes:
        num_geoms (int): The total number of geometry elements in the model (host-side).
        wid (wp.array | None): World index each geometry element.\n
            Shape of ``(num_geoms,)`` and type :class:`int`.
        gid (wp.array | None): Geometry index of each geometry element w.r.t its world.\n
            Shape of ``(num_geoms,)`` and type :class:`int32`.
        lid (wp.array | None): Layer index of each geometry element w.r.t its body.\n
            Shape of ``(num_geoms,)`` and type :class:`int`.
        bid (wp.array | None): Body index of each geometry element.\n
            Shape of ``(num_geoms,)`` and type :class:`int`.
        sid (wp.array | None): Shape index of each geometry element.\n
            Shape of ``(num_geoms,)`` and type :class:`int`.
        ptr (wp.array | None): Pointer to the source data of the shape.\n
            For primitive shapes this is `0` indicating NULL, otherwise it points to
            the shape data, which can correspond to a mesh, heightfield, or SDF.\n
            Shape of ``(num_geoms,)`` and type :class:`uint64`.
        params (wp.array | None): Shape parameters of each geometry element if they are shape primitives.\n
            Shape of ``(num_geoms,)`` and type :class:`vec4f`.
        offset (wp.array | None): Offset poses of the geometry elements w.r.t. their corresponding bodies.\n
            Shape of ``(num_geoms,)`` and type :class:`transformf`.
    """

    num_geoms: int = 0
    """Total number of geometry elements in the model (host-side)."""

    wid: wp.array | None = None
    """
    World index each geometry element.\n
    Shape of ``(num_geoms,)`` and type :class:`int`.
    """

    gid: wp.array | None = None
    """
    Geometry index of each geometry element w.r.t its world.\n
    Shape of ``(num_geoms,)`` and type :class:`int32`.
    """

    lid: wp.array | None = None
    """
    Layer index of each geometry element w.r.t its body.\n
    Shape of ``(num_geoms,)`` and type :class:`int`.
    """

    bid: wp.array | None = None
    """
    Body index of each geometry element.\n
    Shape of ``(num_geoms,)`` and type :class:`int`.
    """

    sid: wp.array | None = None
    """
    Shape index of each geometry element.\n
    Shape of ``(num_geoms,)`` and type :class:`int`.
    """

    ptr: wp.array | None = None
    """
    Pointer to the source data of the shape.\n
    For primitive shapes this is `0` indicating NULL, otherwise it points to
    the shape data, which can correspond to a mesh, heightfield, or SDF.\n
    Shape of ``(num_geoms,)`` and type :class:`uint64`.
    """

    params: wp.array | None = None
    """
    Shape parameters of each geometry element if they are shape primitives.\n
    Shape of ``(num_geoms,)`` and type :class:`vec4f`.
    """

    offset: wp.array | None = None
    """
    Offset poses of the geometry elements w.r.t. their corresponding bodies.\n
    Shape of ``(num_geoms,)`` and type :class:`transformf`.
    """


@dataclass
class GeometriesData:
    """
    An SoA-based container to hold time-varying data of a set of generic geometry elements.

    Attributes:
        num_geoms (int32): The total number of geometry elements in the model (host-side).
        pose (wp.array | None): The poses of the geometry elements in world coordinates.\n
            Shape of ``(num_geoms,)`` and type :class:`transformf`.
    """

    num_geoms: int32 = 0
    """Total number of geometry elements in the model (host-side)."""

    pose: wp.array | None = None
    """
    The poses of the geometry elements in world coordinates.\n
    Shape of ``(num_geoms,)`` and type :class:`transformf`.
    """


###
# Collision Geometry Containers
###


class CollisionGeometryDescriptor(GeometryDescriptor):
    """
    A container to describe a collision geometry entity.

    Collision geometry are a specialization of the base geometry entity,
    that are extended to include additional properties relevant for collision detection.

    Attributes:
        lid (int): Index of the geometry layer to which the geometry entity belongs.\n
            Defaults to `-1`, indicating that the geometry has not yet been assigned to a layer.
        bid (int): Index of the body to which the geometry entity is attached.\n
            Defaults to `-1`, indicating that the geometry has not yet been assigned to a body.\n
            The value `-1` also indicates that the geometry, by default, is statically attached to the world.
        shape (ShapeDescriptorType | None): Definition of the shape of the geometry entity.
            See :class:`ShapeDescriptorType` for the list of supported shape types.
        offset (transformf): Offset pose transform of the geometry entity w.r.t. its corresponding body.\n
            Defaults to the identity transform with zero translation and identity rotation quaternion.
        wid (int): Index of the world to which the body belongs.\n
            Defaults to `-1`, indicating that the body has not yet been added to a world.
        gid (int): Index of the geometry w.r.t. its world.\n
            Defaults to `-1`, indicating that the geometry has not yet been added to a world.
        mid (int): The material index assigned to the collision geometry instance.\n
            Defaults to `0` indicating the default material.
        group (int): The collision group to which the collision geometry instance is assigned.\n
            Defaults to the default group with value `1`.
        collides (int): The collision group with which the collision geometry instance can collide.\n
            Defaults to enabling collisions with the default group with value `1`.
        max_contacts (int): The maximum number of contacts to generate for the collision geometry instance.\n
            Defaults to `0`, indicating no limit is imposed on the number of contacts generated for this geometry.
    """

    def __init__(
        self,
        base: GeometryDescriptor | None = None,
        mid: int = 0,
        group: int = 1,
        collides: int = 1,
        max_contacts: int = 0,
        **kwargs,
    ):
        """
        Initializes a CollisionGeometryDescriptor instance.

        Args:
            base (GeometryDescriptor | None): An optional base :class:`GeometryDescriptor` instance to encapsulate.\n
                If provided, the properties of the base descriptor will be copied to the new instance.
            mid (int): The material index assigned to the collision geometry instance.\n
                Defaults to `0` indicating the default material.
            group (int): The collision group to which the collision geometry instance is assigned.\n
                Defaults to the default group with value `1`.
            collides (int): The collision group with which the collision geometry instance can collide.\n
                Defaults to enabling collisions with the default group with value `1`.
            max_contacts (int): The maximum number of contacts to generate for the collision geometry instance.\n
                Defaults to `0`, indicating no limit is imposed on the number of contacts generated for this geometry.
            **kwargs: Additional keyword arguments to initialize the base :class:`GeometryDescriptor`.\n
                See :class:`GeometryDescriptor` for the list of supported properties.\n
                WARNING: Any properties provided via keyword arguments will override those of the given base descriptor.
        """

        # If no base descriptor is provided, create a new GeometryDescriptor with the given kwargs
        if base is None:
            _base = GeometryDescriptor(**kwargs)

        # NOTE: This will override any properties set from the base descriptor
        else:
            # NOTE: We use (shallow) copy to avoid modifying the original
            # base descriptor object just in case it is used elsewhere
            _base = copy.copy(base)
            for key, value in kwargs.items():
                setattr(_base, key, value)

        # Initialize the base GeometryDescriptor with any additional keyword arguments
        super().__init__(**_base.__dict__)

        # Declare and initialize collision geometry-specific properties
        self.mid: int = mid
        """
        The material index assigned to the collision geometry instance.\n
        Defaults to `0` indicating the default material.
        """

        self.group: int = group
        """
        The collision group to which the collision geometry instance is assigned.\n
        Defaults to the default group with value `1`.
        """

        self.collides: int = collides
        """
        The collision group with which the collision geometry instance can collide.\n
        Defaults to enabling collisions with the default group with value `1`.
        """

        self.max_contacts: int = max_contacts
        """
        The maximum number of contacts to generate for the collision geometry instance.\n
        Defaults to `0`, indicating no limit is imposed on the number of contacts generated for this geometry.
        """

    @override
    def __repr__(self):
        """Returns a human-readable string representation of the CollisionGeometryDescriptor."""
        return (
            f"CollisionGeometryDescriptor(\n"
            f"name: {self.name},\n"
            f"uid: {self.uid},\n"
            f"lid: {self.lid},\n"
            f"bid: {self.bid},\n"
            f"offset: {self.offset},\n"
            f"shape: {self.shape},\n"
            f"mid: {self.mid},\n"
            f"group: {self.group},\n"
            f"collides: {self.collides},\n"
            f"max_contacts: {self.max_contacts}\n"
            f"wid: {self.wid},\n"
            f"gid: {self.gid},\n"
            f")"
        )


@dataclass
class CollisionGeometriesModel(GeometriesModel):
    """
    An SoA-based container to hold time-invariant model data of a set of collision geometry elements.

    Attributes:
        mid (wp.array | None): Material indices assigned to each collision geometry instance.\n
            Shape of ``(num_geoms,)`` and type :class:`int`.
        group (wp.array | None): Collision groups to which each collision geometry instance is assigned.\n
            Shape of ``(num_geoms,)`` and type :class:`uint32`.
        collides (wp.array | None): Collision groups with which each collision geometry can collide.\n
            Shape of ``(num_geoms,)`` and type :class:`uint32`.
    """

    mid: wp.array | None = None
    """
    Material indices assigned to each collision geometry instance.\n
    Shape of ``(num_geoms,)`` and type :class:`int`.
    """

    group: wp.array | None = None
    """
    Collision groups to which each collision geometry instance is assigned.\n
    Shape of ``(num_geoms,)`` and type :class:`uint32`.
    """

    collides: wp.array | None = None
    """
    Collision groups with which each collision geometry can collide.\n
    Shape of ``(num_geoms,)`` and type :class:`uint32`.
    """


@dataclass
class CollisionGeometriesData(GeometriesData):
    """
    An SoA-based container to hold time-varying data of a set of collision geometry elements.

    Attributes:
        aabb (wp.array | None): The vertices of the Axis-Aligned Bounding Box (AABB) of each collision geometry element.\n
            Shape of ``(num_geoms,)`` and type :class:`mat83f`.
        radius (wp.array | None): The radius of the Bounding Sphere (BS) of each collision geometry element.\n
            Shape of ``(num_geoms,)`` and type :class:`float32`.
    """

    aabb: wp.array | None = None
    """
    The vertices of the Axis-Aligned Bounding Box (AABB) of each collision geometry element.\n
    Shape of ``(num_geoms,)`` and type :class:`mat83f`.
    """

    radius: wp.array | None = None
    """
    The radius of the Bounding Sphere (BS) of each collision geometry element.\n
    Shape of ``(num_geoms,)`` and type :class:`float32`.
    """


###
# Kernels
###


@wp.kernel
def _update_geometries_state(
    # Inputs:
    geom_bid: wp.array(dtype=int32),
    geom_offset: wp.array(dtype=transformf),
    body_pose: wp.array(dtype=transformf),
    # Outputs:
    geom_pose: wp.array(dtype=transformf),
):
    # Retrieve the thread ID
    tid = wp.tid()

    # Retrieve the geometry element's body index and pose
    bid = geom_bid[tid]

    # Retrieve the pose of the corresponding body
    # TODO: How to handle the case when bid is -1?
    T_b = wp.transform_identity(dtype=float32)
    if bid > -1:
        T_b = body_pose[bid]

    # Retrieve the geometry element's offset pose w.r.t. the body
    T_g_o = geom_offset[tid]

    # Compute the geometry element's pose in world coordinates
    T_g = wp.transform_multiply(T_b, T_g_o)

    # Store the geometry element's pose
    geom_pose[tid] = T_g


@wp.kernel
def _update_aabb(
    # Inputs:
    geom_sid: wp.array(dtype=int32),
    geom_params: wp.array(dtype=vec4f),
    geom_pose: wp.array(dtype=transformf),
    # Outputs:
    geom_aabb: wp.array(dtype=mat83f),
):
    # Retrieve the thread ID
    tid = wp.tid()

    # Store the geometry element's pose
    T_g = geom_pose[tid]

    # Compute the geometry element's AABB based on its shape parameters
    geom_aabb[tid] = aabb_geom(T_g, geom_params[tid], geom_sid[tid])


@wp.kernel
def _update_bs(
    # Inputs:
    geom_sid: wp.array(dtype=int32),
    geom_params: wp.array(dtype=vec4f),
    geom_pose: wp.array(dtype=transformf),
    # Outputs:
    geom_radius: wp.array(dtype=float32),
):
    # Retrieve the thread ID
    tid = wp.tid()

    # Store the geometry element's pose
    T_g = geom_pose[tid]

    # Compute the geometry element's radius based on its shape parameters
    geom_radius[tid] = bs_geom(T_g, geom_params[tid], geom_sid[tid])


@wp.kernel
def _update_collision_geometries_state(
    # Inputs:
    geom_bid: wp.array(dtype=int32),
    geom_sid: wp.array(dtype=int32),
    geom_params: wp.array(dtype=vec4f),
    geom_offset: wp.array(dtype=transformf),
    body_pose: wp.array(dtype=transformf),
    # Outputs:
    geom_pose: wp.array(dtype=transformf),
    geom_aabb: wp.array(dtype=mat83f),
):
    # Retrieve the thread ID
    tid = wp.tid()

    # Retrieve the geometry element's body index and pose
    bid = geom_bid[tid]

    # Retrieve the pose of the corresponding body
    # TODO: How to handle the case when bid is -1?
    T_b = wp.transform_identity(dtype=float32)
    if bid > -1:
        T_b = body_pose[bid]

    # Retrieve the geometry element's offset pose w.r.t. the body
    T_g_o = geom_offset[tid]

    # Compute the geometry element's pose in world coordinates
    T_g = wp.transform_multiply(T_b, T_g_o)

    # Store the geometry element's pose
    geom_pose[tid] = T_g

    # Compute the geometry element's AABB based on its shape parameters
    geom_aabb[tid] = aabb_geom(T_g, geom_params[tid], geom_sid[tid])


###
# Launchers
###


def update_geometries_state(
    body_poses: wp.array(dtype=transformf), geom_model: GeometriesModel, geom_data: GeometriesData
):
    # we need to figure out how to keep the overhead of this small - not launching anything
    # for pair types without collisions, as well as updating the launch dimensions.
    wp.launch(
        _update_geometries_state,
        dim=geom_model.num_geoms,
        inputs=[geom_model.bid, geom_model.offset, body_poses],
        outputs=[geom_data.pose],
    )


def update_aabb(geom_model: CollisionGeometriesModel, geom_data: CollisionGeometriesData):
    # we need to figure out how to keep the overhead of this small - not launching anything
    # for pair types without collisions, as well as updating the launch dimensions.
    wp.launch(
        _update_aabb,
        dim=geom_model.num_geoms,
        inputs=[geom_model.sid, geom_model.params, geom_data.pose],
        outputs=[geom_data.aabb],
    )


def update_bounding_spheres(geom_model: CollisionGeometriesModel, geom_data: CollisionGeometriesData):
    # we need to figure out how to keep the overhead of this small - not launching anything
    # for pair types without collisions, as well as updating the launch dimensions.
    wp.launch(
        _update_bs,
        dim=geom_model.num_geoms,
        inputs=[geom_model.sid, geom_model.params, geom_data.pose],
        outputs=[geom_data.radius],
    )


def update_collision_geometries_state(
    body_poses: wp.array(dtype=transformf), geom_model: CollisionGeometriesModel, geom_data: CollisionGeometriesData
):
    # we need to figure out how to keep the overhead of this small - not launching anything
    # for pair types without collisions, as well as updating the launch dimensions.
    wp.launch(
        _update_collision_geometries_state,
        dim=geom_model.num_geoms,
        inputs=[geom_model.bid, geom_model.sid, geom_model.params, geom_model.offset, body_poses],
        outputs=[geom_data.pose, geom_data.aabb],
    )
