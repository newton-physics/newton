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

import warp as wp

from .bv import aabb_geom, bs_geom
from .shapes import ShapeDescriptorType
from .types import float32, int32, mat83f, override, transformf, uint32, uint64, vec4f

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


class GeometryDescriptor:
    """
    A container to describe a generic geometry entity.

    A geometry entity is an abstraction to represent the composition of a
    shape, with a pose w.r.t the world frame of a scene. Each geometry entity
    is assigned to a single body, and is organized into one of the distinct
    geometry sets called 'layers'. Each geometry descriptor bundles the unique
    object identifiers of the entity, indices to the associated body and layer,
    the offset pose w.r.t. the body, and a shape descriptor.
    """

    def __init__(self):
        self.name: str | None = None
        """Name of the geometry element."""

        self.uid: str | None = None
        """Unique identifier of the geometry element."""

        self.wid: int = 0
        """Index of the world to which the geometry element belongs."""

        self.gid: int = -1
        """Geometry index of the geometry element."""

        self.lid: int = -1
        """Layer index of the geometry element."""

        self.bid: int = -1
        """Body index of the geometry element."""

        self.shape: ShapeDescriptorType | None = None
        """Definition of the shape of the geometry element of type :class:`ShapeDescriptorType`."""

        self.offset: transformf = transformf()
        """Offset pose of the geometry element w.r.t. its corresponding body, of type :class:`transformf`."""

    @override
    def __repr__(self):
        return (
            f"GeometryDescriptor(\n"
            f"name: {self.name},\n"
            f"uid: {self.uid},\n"
            f"wid: {self.wid},\n"
            f"gid: {self.gid},\n"
            f"lid: {self.lid},\n"
            f"bid: {self.bid},\n"
            f"shape: {self.shape}\n"
            f"offset: {self.offset},\n"
            f")"
        )


class GeometriesModel:
    """
    An SoA-based container to hold time-invariant model data of a set of generic geometry elements.
    """

    def __init__(self):
        self.num_geoms: int = 0
        """Total number of geometry elements in the model (host-side)."""

        self.wid: wp.array(dtype=int32) | None = None
        """
        World index each geometry element.\n
        Shape of ``(num_geoms,)`` and type :class:`int32`.
        """

        self.gid: wp.array(dtype=int32) | None = None
        """
        Geometry index of each geometry element w.r.t its world.\n
        Shape of ``(num_geoms,)`` and type :class:`int32`.
        """

        self.lid: wp.array(dtype=int32) | None = None
        """
        Layer index of each geometry element w.r.t its body.\n
        Shape of ``(num_geoms,)`` and type :class:`int32`.
        """

        self.bid: wp.array(dtype=int32) | None = None
        """
        Body index of each geometry element.\n
        Shape of ``(num_geoms,)`` and type :class:`int32`.
        """

        self.sid: wp.array(dtype=int32) | None = None
        """
        Shape index of each geometry element.\n
        Shape of ``(num_geoms,)`` and type :class:`int32`.
        """

        self.ptr: wp.array(dtype=uint64) | None = None
        """
        Pointer to the source data of the shape.\n
        For primitive shapes this is `0` indicating NULL, otherwise it points to
        the shape data, which can correspond to a mesh, heightfield, or SDF.\n
        Shape of ``(num_geoms,)`` and type :class:`uint64`.
        """

        self.params: wp.array(dtype=vec4f) | None = None
        """
        Shape parameters of each geometry element if they are shape primitives.\n
        Shape of ``(num_geoms,)`` and type :class:`vec4f`.
        """

        self.offset: wp.array(dtype=transformf) | None = None
        """
        Offset poses of the geometry elements w.r.t. their corresponding bodies.\n
        Shape of ``(num_geoms,)`` and type :class:`transformf`.
        """


class GeometriesData:
    """
    An SoA-based container to hold time-varying data of a set of generic geometry elements.
    """

    def __init__(self):
        self.num_geoms: int32 = 0
        """Total number of geometry elements in the model (host-side)."""

        self.pose: wp.array(dtype=transformf) | None = None
        """
        The poses of the geometry elements in world coordinates.\n
        Shape of ``(num_geoms,)`` and type :class:`transformf`.
        """


###
# Collision Geometry Containers
###


class CollisionGeometryDescriptor(GeometryDescriptor):
    """
    A container to describe a collision geometry element.

    Collision geometry elements are specializations of the base geometry elements,
    which are extended to include additional properties relevant for collision detection.
    """

    def __init__(self, base: GeometryDescriptor | None = None):
        super().__init__()

        # If a base descriptor is provided, copy its properties
        if base is not None:
            self.__dict__.update(base.__dict__)

        self.mid: int = 0
        """The material index assigned to the collision geometry instance (0 = the default material)."""

        self.group: int = 1
        """The collision group to which the collision geometry instance is assigned (1 = the default group)."""

        self.collides: int = 1
        """The collision group with which the collision geometry instance can collide (1 = the default group)."""

        self.max_contacts: int = 0
        """The maximum number of contacts to generate for the collision geometry instance (0 = unlimited, default)."""

    def copy_from(self, base: GeometryDescriptor):
        self.name = base.name
        self.uid = base.uid
        self.wid = base.wid
        self.gid = base.gid
        self.lid = base.lid
        self.bid = base.bid
        self.offset = base.offset
        self.shape = base.shape

    @override
    def __repr__(self):
        return (
            f"CollisionGeometryDescriptor(\n"
            f"name: {self.name},\n"
            f"uid: {self.uid},\n"
            f"wid: {self.wid},\n"
            f"gid: {self.gid},\n"
            f"lid: {self.lid},\n"
            f"bid: {self.bid},\n"
            f"offset: {self.offset},\n"
            f"shape: {self.shape},\n"
            f"mid: {self.mid},\n"
            f"group: {self.group},\n"
            f"collides: {self.collides},\n"
            f"max_contacts: {self.max_contacts}\n"
            f")"
        )


class CollisionGeometriesModel(GeometriesModel):
    """
    An SoA-based container to hold time-invariant model data of a set of collision geometry elements.
    """

    def __init__(self):
        super().__init__()

        self.mid: wp.array(dtype=int32) | None = None
        """
        Material indices assigned to each collision geometry instance.\n
        Shape of ``(num_geoms,)`` and type :class:`int32`.
        """

        self.group: wp.array(dtype=uint32) | None = None
        """
        Collision groups to which each collision geometry instance is assigned.\n
        Shape of ``(num_geoms,)`` and type :class:`uint32`.
        """

        self.collides: wp.array(dtype=uint32) | None = None
        """
        Collision groups with which each collision geometry can collide.\n
        Shape of ``(num_geoms,)`` and type :class:`uint32`.
        """


class CollisionGeometriesData(GeometriesData):
    """
    An SoA-based container to hold time-varying data of a set of collision geometry elements.
    """

    def __init__(self):
        super().__init__()

        self.aabb: wp.array(dtype=mat83f) | None = None
        """
        The vertices of the Axis-Aligned Bounding Box (AABB) of each collision geometry element.\n
        Shape of ``(num_geoms,)`` and type :class:`mat83f`.
        """

        self.radius: wp.array(dtype=float32) | None = None
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
