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
KAMINO: Constrained Rigid Multi-Body Model Builder
"""

from __future__ import annotations

import copy

import numpy as np
import warp as wp
from warp.context import Devicelike

from ..utils import logger as msg
from .bodies import RigidBodiesModel, RigidBodyDescriptor
from .geometry import (
    CollisionGeometriesModel,
    CollisionGeometryDescriptor,
    GeometriesModel,
    GeometryDescriptor,
)
from .gravity import GravityDescriptor, GravityModel
from .joints import (
    JointActuationType,
    JointDescriptor,
    JointDoFType,
    JointsModel,
)
from .materials import MaterialDescriptor, MaterialManager, MaterialPairProperties, MaterialPairsModel
from .math import FLOAT32_EPS
from .model import Model, ModelInfo
from .shapes import ShapeDescriptorType, ShapeType
from .time import TimeModel
from .types import Axis, float32, int32, mat33f, transformf, uint32, vec3f, vec4f, vec6f
from .world import WorldDescriptor

###
# Module interface
###

__all__ = [
    "ModelBuilder",
]


###
# Module configs
###

wp.set_module_options({"enable_backward": False})


###
# Containers
###


class ModelBuilder:
    """
    A class to facilitate construction of simulation models.
    """

    def __init__(self, default_world: bool = False):
        """
        Initializes a new empty model builder.

        Args:
            default_world (bool): Whether to create a default world upon initialization.
                If True, a default world will be created. Defaults to False.
        """
        # Meta-data
        self._num_worlds: int = 0
        self._device: Devicelike = None
        self._requires_grad: bool = False

        # Declare and initialize counters
        self._num_bodies: int = 0
        self._num_joints: int = 0
        self._num_cgeoms: int = 0
        self._num_pgeoms: int = 0
        self._num_materials: int = 0
        self._num_bdofs: int = 0
        self._num_jcoords: int = 0
        self._num_jdofs: int = 0
        self._num_jpcoords: int = 0
        self._num_jpdofs: int = 0
        self._num_jacoords: int = 0
        self._num_jadofs: int = 0
        self._num_jcts: int = 0

        # Declare per-world model descriptor sets
        self._up_axes: list[Axis] = []
        self._worlds: list[WorldDescriptor] = []
        self._gravity: list[GravityDescriptor] = []
        self._bodies: list[RigidBodyDescriptor] = []
        self._joints: list[JointDescriptor] = []
        self._pgeoms: list[GeometryDescriptor] = []
        self._cgeoms: list[CollisionGeometryDescriptor] = []

        # Declare a global material manager
        self._materials: MaterialManager = MaterialManager()
        self._num_materials = 1

        # Create a default world if requested
        if default_world:
            self.add_world()

    @property
    def num_worlds(self) -> int:
        """Returns the number of worlds represented in the model."""
        return self._num_worlds

    @property
    def num_bodies(self) -> int:
        """Returns the number of bodies contained in the model."""
        return self._num_bodies

    @property
    def num_joints(self) -> int:
        """Returns the number of joints contained in the model."""
        return self._num_joints

    @property
    def num_collision_geoms(self) -> int:
        """Returns the number of collision geometries contained in the model."""
        return self._num_cgeoms

    @property
    def num_physical_geoms(self) -> int:
        """Returns the number of physical geometries contained in the model."""
        return self._num_pgeoms

    @property
    def num_materials(self) -> int:
        """Returns the number of materials contained in the model."""
        return self._num_materials

    @property
    def num_body_dofs(self) -> int:
        """Returns the number of body degrees of freedom contained in the model."""
        return self._num_bdofs

    @property
    def num_joint_coords(self) -> int:
        """Returns the number of joint coordinates contained in the model."""
        return self._num_jcoords

    @property
    def num_joint_dofs(self) -> int:
        """Returns the number of joint degrees of freedom contained in the model."""
        return self._num_jdofs

    @property
    def num_passive_joint_coords(self) -> int:
        """Returns the number of passive joint coordinates contained in the model."""
        return self._num_jpcoords

    @property
    def num_passive_joint_dofs(self) -> int:
        """Returns the number of passive joint degrees of freedom contained in the model."""
        return self._num_jpdofs

    @property
    def num_actuated_joint_coords(self) -> int:
        """Returns the number of actuated joint coordinates contained in the model."""
        return self._num_jacoords

    @property
    def num_actuated_joint_dofs(self) -> int:
        """Returns the number of actuated joint degrees of freedom contained in the model."""
        return self._num_jadofs

    @property
    def num_joint_cts(self) -> int:
        """Returns the number of joint contact points contained in the model."""
        return self._num_jcts

    @property
    def worlds(self) -> list[WorldDescriptor]:
        """Returns the list of world descriptors contained in the model."""
        return self._worlds

    @property
    def up_axes(self) -> list[Axis]:
        """Returns the list of up axes for each world contained in the model."""
        return self._up_axes

    @property
    def gravity(self) -> list[GravityDescriptor]:
        """Returns the list of gravity descriptors for each world contained in the model."""
        return self._gravity

    @property
    def bodies(self) -> list[RigidBodyDescriptor]:
        """Returns the list of body descriptors contained in the model."""
        return self._bodies

    @property
    def joints(self) -> list[JointDescriptor]:
        """Returns the list of joint descriptors contained in the model."""
        return self._joints

    @property
    def collision_geoms(self) -> list[CollisionGeometryDescriptor]:
        """Returns the list of collision geometry descriptors contained in the model."""
        return self._cgeoms

    @property
    def physical_geoms(self) -> list[GeometryDescriptor]:
        """Returns the list of physical geometry descriptors contained in the model."""
        return self._pgeoms

    @property
    def materials(self) -> list[MaterialDescriptor]:
        """Returns the list of material descriptors contained in the model."""
        return self._materials.materials

    ###
    # Model Construction
    ###

    def add_world(
        self,
        name: str = "world",
        uid: str | None = None,
        up_axis: Axis | None = None,
        gravity: GravityDescriptor | None = None,
    ) -> int:
        """
        Add a new world to the model.

        Args:
            name (str): The name of the world.
            uid (str | None): The unique identifier of the world.\n
                If None, a UUID will be generated.
            up_axis (Axis | None): The up axis of the world.\n
                If None, Axis.Z will be used.
            gravity (GravityDescriptor | None): The gravity descriptor of the world.\n
                If None, a default gravity descriptor will be used.

        Returns:
            int: The index of the newly added world.
        """
        # Create a new world descriptor
        self._worlds.append(WorldDescriptor(name=name, uid=uid, wid=self._num_worlds))

        # Set up axis
        if up_axis is None:
            up_axis = Axis.Z
        self._up_axes.append(up_axis)

        # Set gravity
        if gravity is None:
            gravity = GravityDescriptor()
        self._gravity.append(gravity)

        # Register the default material in the new world
        self._worlds[-1].add_material(self._materials.default)

        # Update world counter
        self._num_worlds += 1

        # Return the new world index
        return self._worlds[-1].wid

    def add_rigid_body(
        self,
        m_i: float,
        i_I_i: mat33f,
        q_i_0: transformf,
        u_i_0: vec6f | None = None,
        name: str | None = None,
        uid: str | None = None,
        world_index: int = 0,
    ) -> int:
        """
        Add a rigid body to the model using explicit specifications.

        Args:
            m_i (float): The mass of the body.
            i_I_i (mat33f): The inertia tensor of the body.
            q_i_0 (transformf): The initial pose of the body.
            u_i_0 (vec6f): The initial velocity of the body.
            name (str | None): The name of the body.
            uid (str | None): The unique identifier of the body.
            world_index (int): The index of the world to which the body will be added.\n
                Defaults to the first world with index `0`.

        Returns:
            int: The index of the newly added body.
        """
        # Create a rigid body descriptor from the provided specifications
        # NOTE: Specifying a name is required by the base descriptor class,
        # but we allow it to be optional here for convenience. Thus, we
        # generate a default name if none is provided.
        body = RigidBodyDescriptor(
            name=name if name is not None else f"body_{self._num_bodies}",
            uid=uid,
            m_i=m_i,
            i_I_i=i_I_i,
            q_i_0=q_i_0,
            u_i_0=u_i_0 if u_i_0 is not None else vec6f(0.0),
        )

        # Add the body descriptor to the model
        return self.add_rigid_body_descriptor(body, world_index=world_index)

    def add_rigid_body_descriptor(self, body: RigidBodyDescriptor, world_index: int = 0) -> int:
        """
        Add a body to the model using a descriptor object.

        Args:
            body (RigidBodyDescriptor): The body descriptor to be added.
            world_index (int): The index of the world to which the body will be added.\n
                Defaults to the first world with index `0`.

        Returns:
            int: The body index of the newly added body w.r.t its world.
        """
        # Check if the descriptor is valid
        if not isinstance(body, RigidBodyDescriptor):
            raise TypeError(f"Invalid body descriptor type: {type(body)}. Must be `RigidBodyDescriptor`.")

        # Check if body properties are valid
        self._check_body_inertia(body.m_i, body.i_I_i)
        self._check_body_pose(body.q_i_0)

        # Check if the world index is valid
        world = self._check_world_index(world_index)

        # Append body model data
        world.add_body(body)
        self._insert_entity(self._bodies, body, world_index=world_index)

        # Update model-wide counters
        self._num_bodies += 1
        self._num_bdofs += 6

        # Return the new body index
        return body.bid

    def add_joint(
        self,
        act_type: JointActuationType,
        dof_type: JointDoFType,
        bid_B: int,
        bid_F: int,
        B_r_Bj: vec3f,
        F_r_Fj: vec3f,
        X_j: mat33f,
        q_j_min: list[float] | float | None = None,
        q_j_max: list[float] | float | None = None,
        dq_j_max: list[float] | float | None = None,
        tau_j_max: list[float] | float | None = None,
        name: str | None = None,
        uid: str | None = None,
        world_index: int = 0,
    ) -> int:
        """
        Add a joint to the model using explicit specifications.

        Args:
            act_type (JointActuationType): The actuation type of the joint.
            dof_type (JointDoFType): The degree of freedom type of the joint.
            bid_B (int): The index of the body on the "base" side of the joint.
            bid_F (int): The index of the body on the "follower" side of the joint.
            B_r_Bj (vec3f): The position of the joint in the base body frame.
            F_r_Fj (vec3f): The position of the joint in the follower body frame.
            X_j (mat33f): The orientation of the joint frame relative to the base body frame.
            q_j_min (list[float] | float | None): The minimum joint coordinate limits.
            q_j_max (list[float] | float | None): The maximum joint coordinate limits.
            dq_j_max (list[float] | float | None): The maximum joint velocity limits.
            tau_j_max (list[float] | float | None): The maximum joint effort limits.
            name (str | None): The name of the joint.
            uid (str | None): The unique identifier of the joint.
            world_index (int): The index of the world to which the joint will be added.\n
                Defaults to the first world with index `0`.

        Returns:
            int: The index of the newly added joint.
        """
        # Check if the actuation type is valid
        if not isinstance(act_type, JointActuationType):
            raise TypeError(f"Invalid actuation type: {act_type}. Must be `JointActuationType`.")

        # Check if the DoF type is valid
        if not isinstance(dof_type, JointDoFType):
            raise TypeError(f"Invalid DoF type: {dof_type}. Must be `JointDoFType`.")

        # Create a joint descriptor from the provided specifications
        # NOTE: Specifying a name is required by the base descriptor class,
        # but we allow it to be optional here for convenience. Thus, we
        # generate a default name if none is provided.
        joint = JointDescriptor(
            name=name if name is not None else f"joint_{self._num_joints}",
            uid=uid,
            act_type=act_type,
            dof_type=dof_type,
            bid_B=bid_B,
            bid_F=bid_F,
            B_r_Bj=B_r_Bj,
            F_r_Fj=F_r_Fj,
            X_j=X_j,
            q_j_min=q_j_min,
            q_j_max=q_j_max,
            dq_j_max=dq_j_max,
            tau_j_max=tau_j_max,
        )

        # Add the body descriptor to the model
        return self.add_joint_descriptor(joint, world_index=world_index)

    def add_joint_descriptor(self, joint: JointDescriptor, world_index: int = 0) -> int:
        """
        Add a joint to the model by descriptor.

        Args:
            joint (JointDescriptor): The joint descriptor to be added.
            world_index (int): The index of the world to which the joint will be added.\n
                Defaults to the first world with index `0`.

        Returns:
            int: The joint index of the newly added joint w.r.t its world.
        """
        # Check if the descriptor is valid
        if not isinstance(joint, JointDescriptor):
            raise TypeError(f"Invalid joint descriptor type: {type(joint)}. Must be `JointDescriptor`.")

        # Check if the world index is valid
        world = self._check_world_index(world_index)

        # Append joint model data
        world.add_joint(joint)
        self._insert_entity(self._joints, joint, world_index=world_index)

        # Update model-wide counters
        self._num_joints += 1
        self._num_jcoords += joint.num_coords
        self._num_jdofs += joint.num_dofs
        self._num_jpcoords += joint.num_coords if joint.is_passive else 0
        self._num_jpdofs += joint.num_dofs if joint.is_passive else 0
        self._num_jacoords += joint.num_coords if joint.is_actuated else 0
        self._num_jadofs += joint.num_dofs if joint.is_actuated else 0
        self._num_jcts += joint.num_cts

        # Return the new joint index
        return joint.jid

    def add_collision_layer(self, name: str, world_index: int = 0):
        """
        Add a new collision geometry layer to the model.

        Args:
            name (str): The name of the collision geometry layer to be added.
            world_index (int): The index of the world to which the layer will be added.\n
                Defaults to the first world with index `0`.
        """
        world = self._check_world_index(world_index)
        world.add_collision_layer(name)

    def add_physical_layer(self, name: str, world_index: int = 0):
        """
        Add a new physical geometry layer to the model.

        Args:
            name (str): The name of the physical geometry layer to be added.
            world_index (int): The index of the world to which the layer will be added.\n
                Defaults to the first world with index `0`.
        """
        world = self._check_world_index(world_index)
        world.add_physical_layer(name)

    def add_collision_geometry(
        self,
        body: int = -1,
        layer: str = "default",
        shape: ShapeDescriptorType | None = None,
        offset: transformf | None = None,
        material: str | int | None = None,
        max_contacts: int = 0,
        group: int = 1,
        collides: int = 1,
        name: str | None = None,
        uid: str | None = None,
        world_index: int = 0,
    ) -> int:
        """
        Add a collision geometry to the model using explicit specifications.

        Args:
            body (int): The index of the body to which the geometry will be attached.\n
                Defaults to -1 (world).
            layer (str): The name of the collision geometry layer.\n
                Defaults to "default".
            shape (ShapeDescriptorType | None): The shape descriptor of the geometry.
            offset (transformf | None): The local offset of the geometry relative to the body frame.
            material (str | int | None): The name or index of the material assigned to the geometry.
            max_contacts (int): The maximum number of contact points for the geometry.\n
                Defaults to 0 (unlimited).
            group (int): The collision group of the geometry.\n
                Defaults to 1.
            collides (int): The collision mask of the geometry.\n
                Defaults to 1.
            name (str | None): The name of the geometry.
            uid (str | None): The unique identifier of the geometry.
            world_index (int): The index of the world to which the geometry will be added.\n
                Defaults to the first world with index `0`.

        Returns:
            int: The index of the newly added collision geometry.
        """
        # Set the default material if not provided
        if material is None:
            material = self._materials.default.name
        # Otherwise, check if the material exists
        else:
            if not self._materials.has_material(material):
                raise ValueError(
                    f"Material '{material}' does not exist. "
                    "Please add the material using `add_material()` before assigning it to a geometry."
                )

        # If the shape is already provided, check if it's valid
        if shape is not None:
            if not isinstance(shape, ShapeDescriptorType):
                raise ValueError(
                    f"Shape '{shape}' must be a valid type.\n"
                    "See `ShapeDescriptorType` for the list of supported shapes."
                )

        # Create a joint descriptor from the provided specifications
        # NOTE: Specifying a name is required by the base descriptor class,
        # but we allow it to be optional here for convenience. Thus, we
        # generate a default name if none is provided.
        geom = CollisionGeometryDescriptor(
            name=name if name is not None else f"cgeom_{self._num_cgeoms}",
            uid=uid,
            bid=body,
            layer=layer,
            offset=offset if offset is not None else transformf(),
            shape=shape,
            material=self._materials[material],
            mid=self._materials.index(material),
            group=group,
            collides=collides,
            max_contacts=max_contacts,
        )

        # Add the body descriptor to the model
        return self.add_collision_geometry_descriptor(geom, world_index=world_index)

    def add_collision_geometry_descriptor(self, geom: CollisionGeometryDescriptor, world_index: int = 0) -> int:
        """
        Add a collision geometry to the model by descriptor.

        Args:
            geom (CollisionGeometryDescriptor): The collision geometry descriptor to be added.
            world_index (int): The index of the world to which the geometry will be added.\n
                Defaults to the first world with index `0`.

        Returns:
            int: The geometry index of the newly added collision geometry w.r.t its world.
        """
        # Check if the descriptor is valid
        if not isinstance(geom, CollisionGeometryDescriptor):
            raise TypeError(
                f"Invalid collision geometry descriptor type: {type(geom)}. Must be `CollisionGeometryDescriptor`."
            )

        # Check if the world index is valid
        world = self._check_world_index(world_index)

        # If the geom material is not assigned, set it to the global default
        if geom.mid is None:
            geom.mid = self._materials.default.mid

        # Append body model data
        world.add_collision_geom(geom)
        self._insert_entity(self._cgeoms, geom, world_index=world_index)

        # Update model-wide counters
        self._num_cgeoms += 1

        # Return the new geometry index
        return geom.gid

    def add_physical_geometry(
        self,
        shape: ShapeDescriptorType,
        body: int = -1,
        layer: str = "default",
        offset: transformf | None = None,
        name: str | None = None,
        uid: str | None = None,
        world_index: int = 0,
    ) -> int:
        """
        Add a physical geometry to the model using explicit specifications.

        Args:
            shape (ShapeDescriptorType): The shape descriptor of the geometry.
            body (int): The index of the body to which the geometry will be attached.\n
                Defaults to -1 (world).
            layer (str): The name of the physical geometry layer.\n
                Defaults to "default".
            offset (transformf | None): The local offset of the geometry relative to the body frame.
            name (str | None): The name of the geometry.
            uid (str | None): The unique identifier of the geometry.
            world_index (int): The index of the world to which the geometry will be added.\n
                Defaults to the first world with index `0`.

        Returns:
            int: The index of the newly added physical geometry.
        """
        # Check if shape is valid
        if not isinstance(shape, ShapeDescriptorType):
            raise ValueError(
                f"Shape '{shape}' must be a valid type.\nSee `ShapeDescriptorType` for the list of supported shapes."
            )

        # Create a joint descriptor from the provided specifications
        # NOTE: Specifying a name is required by the base descriptor class,
        # but we allow it to be optional here for convenience. Thus, we
        # generate a default name if none is provided.
        geom = GeometryDescriptor(
            name=name if name is not None else f"cgeom_{self._num_cgeoms}",
            uid=uid,
            bid=body,
            layer=layer,
            offset=offset if offset is not None else transformf(),
            shape=shape,
        )

        # Add the body descriptor to the model
        return self.add_physical_geometry_descriptor(geom, world_index=world_index)

    def add_physical_geometry_descriptor(self, geom: GeometryDescriptor, world_index: int = 0) -> int:
        """
        Add a physical geometry to the model by descriptor.

        Args:
            geom (GeometryDescriptor): The physical geometry descriptor to be added.
            world_index (int): The index of the world to which the geometry will be added.\n
                Defaults to the first world with index `0`.

        Returns:
            int: The geometry index of the newly added physical geometry w.r.t its world.
        """
        # Check if the descriptor is valid
        if not isinstance(geom, GeometryDescriptor):
            raise TypeError(f"Invalid physical geometry descriptor type: {type(geom)}. Must be `GeometryDescriptor`.")

        # Check if the world index is valid
        world = self._check_world_index(world_index)

        # Append body model data
        world.add_physical_geom(geom)
        self._insert_entity(self._pgeoms, geom, world_index=world_index)

        # Update model-wide counters
        self._num_pgeoms += 1

        # Return the new geometry index
        return geom.gid

    def add_material(self, material: MaterialDescriptor, world_index: int = 0) -> int:
        """
        Add a material to the model.

        Args:
            material (MaterialDescriptor): The material descriptor to be added.
            world_index (int): The index of the world to which the material will be added.\n
                Defaults to the first world with index `0`.
        """
        # Check if the world index is valid
        world = self._check_world_index(world_index)

        # Check if the material is valid
        if not isinstance(material, MaterialDescriptor):
            raise TypeError(f"Invalid material type: {type(material)}. Must be `MaterialDescriptor`.")

        # Register the material in the material manager
        world.add_material(material)

        # Update model-wide counter
        self._num_materials += 1

        return self._materials.register(material)

    def add_builder(self, other: ModelBuilder):
        """
        Extends the contents of the current ModelBuilder with those of another.

        Each builder represents a distinct world, and this method allows for the
        combination of multiple worlds into a single model. The method ensures that the
        indices of the elements in the other builder are adjusted to account for the
        existing elements in the current builder, preventing any index conflicts.

        Arguments:
            other (ModelBuilder): The other ModelBuilder whose contents are to be added to the current.

        Raises:
            ValueError: If the provided builder is not of type `ModelBuilder`.
        """
        # Check if the other builder is of valid type
        if not isinstance(other, ModelBuilder):
            raise ValueError(f"Invalid builder type: {type(other)}. Must be a ModelBuilder instance.")

        # Make a deep copy of the other builder to avoid modifying the original
        # TODO: How can we avoid this deep copy to improve performance
        # while avoiding copying expensive data like meshes?
        _other = copy.deepcopy(other)

        # Append the other per-world descriptors
        self._worlds.extend(_other._worlds)
        self._gravity.extend(_other._gravity)
        self._up_axes.extend(_other._up_axes)

        # Append the other per-entity descriptors
        self._bodies.extend(_other._bodies)
        self._joints.extend(_other._joints)
        self._cgeoms.extend(_other._cgeoms)
        self._pgeoms.extend(_other._pgeoms)

        # Append the other materials
        self._materials.merge(_other._materials)

        # Update the world index of the entities in the
        # other builder and update model-wide counters
        for w, world in enumerate(_other._worlds):
            # Offset world index of the other builder's world
            world.wid = self._num_worlds + w

            # Offset world indices of the other builders entities
            for body in self._bodies[self._num_bodies : self._num_bodies + world.num_bodies]:
                body.wid = self._num_worlds + w
            for joint in self._joints[self._num_joints : self._num_joints + world.num_joints]:
                joint.wid = self._num_worlds + w
            for cgeom in self._cgeoms[self._num_cgeoms : self._num_cgeoms + world.num_collision_geoms]:
                cgeom.wid = self._num_worlds + w
            for pgeom in self._pgeoms[self._num_pgeoms : self._num_pgeoms + world.num_physical_geoms]:
                pgeom.wid = self._num_worlds + w

            # Update model-wide counters
            self._num_bodies += world.num_bodies
            self._num_joints += world.num_joints
            self._num_cgeoms += world.num_collision_geoms
            self._num_pgeoms += world.num_physical_geoms
            self._num_bdofs += 6 * world.num_bodies
            self._num_jcoords += world.num_joint_coords
            self._num_jdofs += world.num_joint_dofs
            self._num_jpcoords += world.num_passive_joint_coords
            self._num_jpdofs += world.num_passive_joint_dofs
            self._num_jacoords += world.num_actuated_joint_coords
            self._num_jadofs += world.num_actuated_joint_dofs
            self._num_jcts += world.num_joint_cts

        # Update the number of worlds
        self._num_worlds += _other._num_worlds

    ###
    # Configurations
    ###

    def set_up_axis(self, axis: Axis, world_index: int = 0):
        """
        Set the up axis for a specific world.

        Args:
            axis (Axis): The new up axis to be set.
            world_index (int): The index of the world for which to set the up axis.\n
                Defaults to the first world with index `0`.

        Raises:
            TypeError: If the provided axis is not of type `Axis`.
        """
        # Check if the world index is valid
        self._check_world_index(world_index)

        # Check if the axis is valid
        if not isinstance(axis, Axis):
            raise TypeError(f"ModelBuilder: Invalid axis type: {type(axis)}. Must be `Axis`.")

        # Set the new up axis
        self._up_axes[world_index] = axis

    def set_gravity(self, gravity: GravityDescriptor, world_index: int = 0):
        """
        Set the gravity descriptor for a specific world.

        Args:
            gravity (GravityDescriptor): The new gravity descriptor to be set.
            world_index (int): The index of the world for which to set the gravity descriptor.\n
                Defaults to the first world with index `0`.

        Raises:
            TypeError: If the provided gravity descriptor is not of type `GravityDescriptor`.
        """
        # Check if the world index is valid
        self._check_world_index(world_index)

        # Check if the gravity descriptor is valid
        if not isinstance(gravity, GravityDescriptor):
            raise TypeError(f"Invalid gravity descriptor type: {type(gravity)}. Must be `GravityDescriptor`.")

        # Set the new gravity configurations
        self._gravity[world_index] = gravity

    def set_default_material(self, material: MaterialDescriptor, world_index: int = 0):
        """
        Sets the default material for the model.
        Raises an error if the material is not registered.
        """
        # Check if the world index is valid
        world = self._check_world_index(world_index)

        # Check if the material is valid
        if not isinstance(material, MaterialDescriptor):
            raise TypeError(f"Invalid material type: {type(material)}. Must be `MaterialDescriptor`.")

        # Reset the default material of the world
        world.set_material(material, 0)

        # Set the default material in the material manager
        self._materials.default = material

    def set_material_pair(
        self,
        first: int | str | MaterialDescriptor,
        second: int | str | MaterialDescriptor,
        material_pair: MaterialPairProperties,
        world_index: int = 0,
    ):
        """
        Sets the material pair properties for two materials.

        Args:
            first (int | str | MaterialDescriptor): The first material (by index, name, or descriptor).
            second (int | str | MaterialDescriptor): The second material (by index, name, or descriptor).
            material_pair (MaterialPairProperties): The material pair properties to be set.
            world_index (int): The index of the world for which to set the material pair properties.\n
                Defaults to the first world with index `0`.
        """
        # Check if the world index is valid
        self._check_world_index(world_index)

        # Extract the material names if arguments are descriptors
        first_id = first.name if isinstance(first, MaterialDescriptor) else first
        second_id = second.name if isinstance(second, MaterialDescriptor) else second

        # Register the material pair in the material manager
        self._materials.configure_pair(first=first_id, second=second_id, material_pair=material_pair)

    def set_base_body(self, body_key: int | str, world_index: int = 0):
        """
        Set the base body for a specific world specified either by name or by index.

        Args:
            body_key (int | str): Identifier of the body to be set as the base body.
                Can be either the body's index (within the world) or its name.
            world_index (int): The index of the world for which to set the base body.\n
                Defaults to the first world with index `0`.
        """
        # Check if the world index is valid
        world = self._check_world_index(world_index)

        # Find the body and set it as base in the world descriptor
        if isinstance(body_key, int):
            world.set_base_body(body_key)
            return
        elif isinstance(body_key, str):
            for body in self.bodies:
                if body.wid == world_index and body.name == body_key:
                    world.set_base_body(body.bid)
                    return
        raise ValueError(f"Failed to identify the base body in world `{world_index} given key {body_key}`.")

    def set_base_joint(self, joint_key: int | str, world_index: int = 0):
        """
        Set the base joint for a specific world specified either by name or by index.

        Args:
            joint_key (int | str): Identifier of the joint to be set as the base joint.
                Can be either the joint's index (within the world) or its name.
            world_index (int): The index of the world for which to set the base joint.\n
                Defaults to the first world with index `0`.
        """
        # Check if the world index is valid
        world = self._check_world_index(world_index)

        # Find the joint and set it as base in the world descriptor
        if isinstance(joint_key, int):
            world.set_base_joint(joint_key)
            return
        elif isinstance(joint_key, str):
            for joint in self.joints:
                if joint.wid == world_index and joint.name == joint_key:
                    world.set_base_joint(joint.jid)
                    return
        raise ValueError(f"Failed to identify the base joint in world `{world_index} given key {joint_key}`.")

    ###
    # Model Compilation
    ###

    def finalize(self, device: Devicelike = None, requires_grad: bool = False, base_auto: bool = True) -> Model:
        """
        Constructs a Model object from the current ModelBuilder.

        All description data contained in the builder is compiled into a Model
        object, allocating the necessary data structures on the target device.

        Args:
            device (Devicelike): The target device for the model data.\n
                If None, the default/preferred device will determined by Warp.
            requires_grad (bool): Whether the model data should support gradients.\n
                Defaults to False.
            base_auto (bool): Whether to automatically select a base body,
                and if possible, a base joint, if neither was set.

        Returns:
            Model: The constructed Model object containing the time-invariant simulation data.
        """
        # Number of model worlds
        num_worlds = len(self._worlds)
        if num_worlds == 0:
            raise ValueError("ModelBuilder: Cannot finalize an empty model with zero worlds.")
        if num_worlds != self._num_worlds:
            raise ValueError(
                f"ModelBuilder: Inconsistent number of worlds: expected {self._num_worlds}, but found {num_worlds}."
            )

        ###
        # Pre-processing
        ###

        # First compute per-world offsets before proceeding
        # NOTE: Computing world offsets only during the finalization step allows
        # users to add entities in any manner. For example, users can import a model
        # via USD, and then ad-hoc modify the model by adding bodies, joints, geoms, etc.
        self._compute_world_offsets()

        # Validate base body/joint data for each world, and fill in missing data if possible
        for w, world in enumerate(self._worlds):
            if world.has_base_joint:
                joint_idx = world.joints_idx_offset + world.base_joint_idx
                follower_idx = self._joints[joint_idx].bid_F  # Note: index among world bodies
                if world.has_base_body:  # Ensure base joint & body are compatible if both were set
                    if world.base_body_idx != follower_idx:
                        raise ValueError(
                            f"ModelBuilder: Inconsistent base body and base joint for world {world.name} ({w})"
                        )
                else:  # Set base body to be the follower of the base joint
                    world.set_base_body(follower_idx)
            elif not world.has_base_body and base_auto:
                world.set_base_body(0)  # Set the base body as the first body
                for jt_idx, joint in enumerate(
                    self._joints[world.joints_idx_offset : world.joints_idx_offset + world.num_joints]
                ):
                    if joint.wid == w and joint.is_unary and joint.is_connected_to_body(world.base_body_idx):
                        # If we find a unary joint connecting the base body to the world, we set this as the base joint
                        world.set_base_joint(jt_idx)
                        break

        ###
        # Model data collection
        ###

        # Initialize the info data collections
        info_nb = []
        info_nj = []
        info_njp = []
        info_nja = []
        info_ncg = []
        info_npg = []
        info_nbd = []
        info_njq = []
        info_njd = []
        info_njpq = []
        info_njpd = []
        info_njaq = []
        info_njad = []
        info_njc = []
        info_bio = []
        info_jio = []
        info_bdio = []
        info_jqio = []
        info_jdio = []
        info_jpqio = []
        info_jpdio = []
        info_jaqio = []
        info_jadio = []
        info_jcio = []
        info_base_bid = []
        info_base_jid = []
        info_mass_min = []
        info_mass_max = []
        info_mass_total = []
        info_inertia_total = []

        # Initialize the gravity data collections
        gravity_g_dir_acc = []
        gravity_vector = []

        # Initialize the body data collections
        bodies_wid = []
        bodies_bid = []
        bodies_m_i = []
        bodies_inv_m_i = []
        bodies_i_I_i = []
        bodies_inv_i_I_i = []
        bodies_q_i_0 = []
        bodies_u_i_0 = []

        # Initialize the joint data collections
        joints_wid = []
        joints_jid = []
        joints_dofid = []
        joints_actid = []
        joints_c_j = []
        joints_d_j = []
        joints_m_j = []
        joints_qio = []
        joints_dio = []
        joints_cio = []
        joints_pqio = []
        joints_pdio = []
        joints_aqio = []
        joints_adio = []
        joints_bid_B = []
        joints_bid_F = []
        joints_B_r_Bj = []
        joints_F_r_Fj = []
        joints_X_j = []
        joints_q_j_min = []
        joints_q_j_max = []
        joints_qd_j_max = []
        joints_tau_j_max = []
        joints_q_j_ref = []

        # Initialize the collision geometry data collections
        cgeoms_wid = []
        cgeoms_gid = []
        cgeoms_lid = []
        cgeoms_bid = []
        cgeoms_sid = []
        cgeoms_ptr = []
        cgeoms_params = []
        cgeoms_offset = []
        cgeoms_mid = []
        cgeoms_group = []
        cgeoms_collides = []
        cgeoms_margin = []

        # Initialize the physical geometry data collections
        pgeoms_wid = []
        pgeoms_gid = []
        pgeoms_lid = []
        pgeoms_bid = []
        pgeoms_sid = []
        pgeoms_ptr = []
        pgeoms_params = []
        pgeoms_offset = []

        # Initialize the material-pair data collections
        mpairs_rest = []
        mpairs_static_fric = []
        mpairs_dynamic_fric = []

        # A helper function to collect model info data
        def collect_model_info_data():
            for world in self._worlds:
                # First collect the immutable counts and
                # index offsets for bodies and joints
                info_nb.append(world.num_bodies)
                info_nj.append(world.num_joints)
                info_njp.append(world.num_passive_joints)
                info_nja.append(world.num_actuated_joints)
                info_ncg.append(world.num_collision_geoms)
                info_npg.append(world.num_physical_geoms)
                info_nbd.append(world.num_body_dofs)
                info_njq.append(world.num_joint_coords)
                info_njd.append(world.num_joint_dofs)
                info_njpq.append(world.num_passive_joint_coords)
                info_njpd.append(world.num_passive_joint_dofs)
                info_njaq.append(world.num_actuated_joint_coords)
                info_njad.append(world.num_actuated_joint_dofs)
                info_njc.append(world.num_joint_cts)
                info_bio.append(world.bodies_idx_offset)
                info_jio.append(world.joints_idx_offset)

                # Collect the model mass and inertia data
                info_mass_min.append(world.mass_min)
                info_mass_max.append(world.mass_max)
                info_mass_total.append(world.mass_total)
                info_inertia_total.append(world.inertia_total)

            # Collect the index offsets for bodies and joints
            for world in self._worlds:
                info_bdio.append(world.body_dofs_idx_offset)
                info_jqio.append(world.joint_coords_idx_offset)
                info_jdio.append(world.joint_dofs_idx_offset)
                info_jpqio.append(world.passive_joint_coords_idx_offset)
                info_jpdio.append(world.passive_joint_dofs_idx_offset)
                info_jaqio.append(world.actuated_joint_coords_idx_offset)
                info_jadio.append(world.actuated_joint_dofs_idx_offset)
                info_jcio.append(world.joint_cts_idx_offset)
                info_base_bid.append((world.base_body_idx + world.bodies_idx_offset) if world.has_base_body else -1)
                info_base_jid.append((world.base_joint_idx + world.joints_idx_offset) if world.has_base_joint else -1)

        # A helper function to collect model gravity data
        def collect_gravity_model_data():
            for w in range(num_worlds):
                gravity_g_dir_acc.append(self._gravity[w].dir_accel())
                gravity_vector.append(self._gravity[w].vector())

        # A helper function to collect model bodies data
        def collect_body_model_data():
            for body in self._bodies:
                bodies_wid.append(body.wid)
                bodies_bid.append(body.bid)
                bodies_m_i.append(body.m_i)
                bodies_inv_m_i.append(1.0 / body.m_i)
                bodies_i_I_i.append(body.i_I_i)
                bodies_inv_i_I_i.append(wp.inverse(body.i_I_i))
                bodies_q_i_0.append(body.q_i_0)
                bodies_u_i_0.append(body.u_i_0)

        # A helper function to collect model joints data
        def collect_joint_model_data():
            for joint in self._joints:
                world_bio = self._worlds[joint.wid].bodies_idx_offset
                joints_wid.append(joint.wid)
                joints_jid.append(joint.jid)
                joints_dofid.append(joint.dof_type.value)
                joints_actid.append(joint.act_type.value)
                joints_B_r_Bj.append(joint.B_r_Bj)
                joints_F_r_Fj.append(joint.F_r_Fj)
                joints_X_j.append(joint.X_j)
                joints_q_j_min.extend(joint.q_j_min)
                joints_q_j_max.extend(joint.q_j_max)
                joints_qd_j_max.extend(joint.dq_j_max)
                joints_tau_j_max.extend(joint.tau_j_max)
                joints_q_j_ref.extend(joint.dof_type.reference_coords)
                joints_c_j.append(joint.num_coords)
                joints_d_j.append(joint.num_dofs)
                joints_m_j.append(joint.num_cts)
                joints_qio.append(joint.coords_offset)
                joints_dio.append(joint.dofs_offset)
                joints_cio.append(joint.cts_offset)
                joints_pqio.append(joint.passive_coords_offset)
                joints_pdio.append(joint.passive_dofs_offset)
                joints_aqio.append(joint.actuated_coords_offset)
                joints_adio.append(joint.actuated_dofs_offset)
                joints_bid_B.append(joint.bid_B + world_bio if joint.bid_B >= 0 else -1)
                joints_bid_F.append(joint.bid_F + world_bio if joint.bid_F >= 0 else -1)

        # A helper function to create geometry pointers
        # NOTE: This also finalizes the mesh/SDF/HField data on the device
        def make_geometry_source_pointer(geom: GeometryDescriptor, mesh_geoms: dict, device) -> int:
            # Append to data pointers array of the shape has a Mesh, SDF or HField source
            if geom.shape.type in (ShapeType.MESH, ShapeType.CONVEX, ShapeType.HFIELD, ShapeType.SDF):
                geom_hash = hash(geom)  # avoid repeated hash computations
                # If the geometry has a Mesh, SDF or HField source,
                # finalize it and retrieve the mesh pointer/index
                if geom_hash not in mesh_geoms:
                    mesh_geoms[geom_hash] = geom.shape.data.finalize(device=device)
                # Return the mesh data pointer/index
                return mesh_geoms[geom_hash]
            # Otherwise, append a null (i.e. zero-valued) pointer
            else:
                return 0

        # A helper function to collect model collision geometries data
        def collect_collision_geometry_model_data():
            cgeom_meshes = {}
            for geom in self._cgeoms:
                cgeoms_wid.append(geom.wid)
                cgeoms_lid.append(geom.lid)
                cgeoms_gid.append(geom.gid)
                cgeoms_bid.append(geom.bid + self._worlds[geom.wid].bodies_idx_offset if geom.bid >= 0 else -1)
                cgeoms_sid.append(geom.shape.type.value)
                cgeoms_params.append(geom.shape.paramsvec)
                cgeoms_offset.append(geom.offset)
                cgeoms_mid.append(geom.mid)
                cgeoms_group.append(geom.group)
                cgeoms_collides.append(geom.collides)
                cgeoms_margin.append(geom.margin)
                cgeoms_ptr.append(make_geometry_source_pointer(geom, cgeom_meshes, device))

        # A helper function to collect model physical geometries data
        def collect_physical_geometry_model_data():
            pgeom_meshes = {}
            for geom in self._pgeoms:
                pgeoms_wid.append(geom.wid)
                pgeoms_lid.append(geom.lid)
                pgeoms_gid.append(geom.gid)
                pgeoms_bid.append(geom.bid + self._worlds[geom.wid].bodies_idx_offset if geom.bid >= 0 else -1)
                pgeoms_sid.append(geom.shape.type.value)
                pgeoms_params.append(geom.shape.paramsvec)
                pgeoms_offset.append(geom.offset)
                pgeoms_ptr.append(make_geometry_source_pointer(geom, pgeom_meshes, device))

        # A helper function to collect model material-pairs data
        def collect_material_pairs_model_data():
            mpairs_rest.append(self._materials.restitution_matrix())
            mpairs_static_fric.append(self._materials.static_friction_matrix())
            mpairs_dynamic_fric.append(self._materials.dynamic_friction_matrix())

        # Collect model data
        collect_model_info_data()
        collect_gravity_model_data()
        collect_body_model_data()
        collect_joint_model_data()
        collect_collision_geometry_model_data()
        collect_physical_geometry_model_data()
        collect_material_pairs_model_data()

        ###
        # Model construction
        ###

        # Create the model
        model = Model()

        # Configure model properties
        model.device = device
        model.requires_grad = requires_grad

        # Store the model builder info list as the model descriptors
        # NOTE This caches the info of each model on the host side
        model.worlds = self._worlds

        ###
        # Set the host-side model size
        ###

        # Compute the sum/max of model entities
        model.size.num_worlds = num_worlds
        model.size.sum_of_num_bodies = self._num_bodies
        model.size.max_of_num_bodies = max([world.num_bodies for world in self._worlds])
        model.size.sum_of_num_joints = self._num_joints
        model.size.max_of_num_joints = max([world.num_joints for world in self._worlds])
        model.size.sum_of_num_passive_joints = sum([world.num_passive_joints for world in self._worlds])
        model.size.max_of_num_passive_joints = max([world.num_passive_joints for world in self._worlds])
        model.size.sum_of_num_actuated_joints = sum([world.num_actuated_joints for world in self._worlds])
        model.size.max_of_num_actuated_joints = max([world.num_actuated_joints for world in self._worlds])
        model.size.sum_of_num_collision_geoms = self._num_cgeoms
        model.size.max_of_num_collision_geoms = max([world.num_collision_geoms for world in self._worlds])
        model.size.sum_of_num_physical_geoms = self._num_pgeoms
        model.size.max_of_num_physical_geoms = max([world.num_physical_geoms for world in self._worlds])
        model.size.sum_of_num_material_pairs = self._materials.num_material_pairs
        model.size.max_of_num_material_pairs = self._materials.num_material_pairs

        # Compute the sum/max of model DoFs and constraints
        model.size.sum_of_num_body_dofs = self._num_bdofs
        model.size.max_of_num_body_dofs = max([world.num_body_dofs for world in self._worlds])
        model.size.sum_of_num_joint_coords = self._num_jcoords
        model.size.max_of_num_joint_coords = max([world.num_joint_coords for world in self._worlds])
        model.size.sum_of_num_joint_dofs = self._num_jdofs
        model.size.max_of_num_joint_dofs = max([world.num_joint_dofs for world in self._worlds])
        model.size.sum_of_num_passive_joint_coords = self._num_jpcoords
        model.size.max_of_num_passive_joint_coords = max([world.num_passive_joint_coords for world in self._worlds])
        model.size.sum_of_num_passive_joint_dofs = self._num_jpdofs
        model.size.max_of_num_passive_joint_dofs = max([world.num_passive_joint_dofs for world in self._worlds])
        model.size.sum_of_num_actuated_joint_coords = self._num_jacoords
        model.size.max_of_num_actuated_joint_coords = max([world.num_actuated_joint_coords for world in self._worlds])
        model.size.sum_of_num_actuated_joint_dofs = self._num_jadofs
        model.size.max_of_num_actuated_joint_dofs = max([world.num_actuated_joint_dofs for world in self._worlds])
        model.size.sum_of_num_joint_cts = self._num_jcts
        model.size.max_of_num_joint_cts = max([world.num_joint_cts for world in self._worlds])

        # Initialize unilateral counts (limits, and contacts) to zero
        model.size.sum_of_max_limits = 0
        model.size.max_of_max_limits = 0
        model.size.sum_of_max_contacts = 0
        model.size.max_of_max_contacts = 0
        model.size.sum_of_max_unilaterals = 0
        model.size.max_of_max_unilaterals = 0

        # Initialize total constraint counts to the same as the joint constraint counts
        model.size.sum_of_max_total_cts = model.size.sum_of_num_joint_cts
        model.size.max_of_max_total_cts = model.size.max_of_num_joint_cts

        ###
        # On-device data allocation
        ###

        # Allocate the model data on the target device
        with wp.ScopedDevice(device):
            # Create the immutable model info arrays from the collected data
            model.info = ModelInfo(
                num_worlds=num_worlds,
                num_bodies=wp.array(info_nb, dtype=int32),
                num_joints=wp.array(info_nj, dtype=int32),
                num_passive_joints=wp.array(info_njp, dtype=int32),
                num_actuated_joints=wp.array(info_nja, dtype=int32),
                num_collision_geoms=wp.array(info_ncg, dtype=int32),
                num_physical_geoms=wp.array(info_npg, dtype=int32),
                num_body_dofs=wp.array(info_nbd, dtype=int32),
                num_joint_coords=wp.array(info_njq, dtype=int32),
                num_joint_dofs=wp.array(info_njd, dtype=int32),
                num_passive_joint_coords=wp.array(info_njpq, dtype=int32),
                num_passive_joint_dofs=wp.array(info_njpd, dtype=int32),
                num_actuated_joint_coords=wp.array(info_njaq, dtype=int32),
                num_actuated_joint_dofs=wp.array(info_njad, dtype=int32),
                num_joint_cts=wp.array(info_njc, dtype=int32),
                bodies_offset=wp.array(info_bio, dtype=int32),
                joints_offset=wp.array(info_jio, dtype=int32),
                body_dofs_offset=wp.array(info_bdio, dtype=int32),
                joint_coords_offset=wp.array(info_jqio, dtype=int32),
                joint_dofs_offset=wp.array(info_jdio, dtype=int32),
                joint_passive_coords_offset=wp.array(info_jpqio, dtype=int32),
                joint_passive_dofs_offset=wp.array(info_jpdio, dtype=int32),
                joint_actuated_coords_offset=wp.array(info_jaqio, dtype=int32),
                joint_actuated_dofs_offset=wp.array(info_jadio, dtype=int32),
                joint_cts_offset=wp.array(info_jcio, dtype=int32),
                base_body_index=wp.array(info_base_bid, dtype=int32),
                base_joint_index=wp.array(info_base_jid, dtype=int32),
                mass_min=wp.array(info_mass_min, dtype=float32),
                mass_max=wp.array(info_mass_max, dtype=float32),
                mass_total=wp.array(info_mass_total, dtype=float32),
                inertia_total=wp.array(info_inertia_total, dtype=float32),
            )

            # Create the model time data
            model.time = TimeModel(dt=wp.zeros(num_worlds, dtype=float32), inv_dt=wp.zeros(num_worlds, dtype=float32))

            # Construct model gravity data
            model.gravity = GravityModel(
                g_dir_acc=wp.array(gravity_g_dir_acc, dtype=vec4f),
                vector=wp.array(gravity_vector, dtype=vec4f, requires_grad=requires_grad),
            )

            # Create the bodies model
            model.bodies = RigidBodiesModel(
                num_bodies=model.size.sum_of_num_bodies,
                wid=wp.array(bodies_wid, dtype=int32),
                bid=wp.array(bodies_bid, dtype=int32),
                m_i=wp.array(bodies_m_i, dtype=float32, requires_grad=requires_grad),
                inv_m_i=wp.array(bodies_inv_m_i, dtype=float32, requires_grad=requires_grad),
                i_I_i=wp.array(bodies_i_I_i, dtype=mat33f, requires_grad=requires_grad),
                inv_i_I_i=wp.array(bodies_inv_i_I_i, dtype=mat33f, requires_grad=requires_grad),
                q_i_0=wp.array(bodies_q_i_0, dtype=transformf, requires_grad=requires_grad),
                u_i_0=wp.array(bodies_u_i_0, dtype=vec6f, requires_grad=requires_grad),
            )

            # Create the joints model
            model.joints = JointsModel(
                num_joints=model.size.sum_of_num_joints,
                wid=wp.array(joints_wid, dtype=int32),
                jid=wp.array(joints_jid, dtype=int32),
                dof_type=wp.array(joints_dofid, dtype=int32),
                act_type=wp.array(joints_actid, dtype=int32),
                num_coords=wp.array(joints_c_j, dtype=int32),
                num_dofs=wp.array(joints_d_j, dtype=int32),
                num_cts=wp.array(joints_m_j, dtype=int32),
                coords_offset=wp.array(joints_qio, dtype=int32),
                dofs_offset=wp.array(joints_dio, dtype=int32),
                passive_coords_offset=wp.array(joints_pqio, dtype=int32),
                passive_dofs_offset=wp.array(joints_pdio, dtype=int32),
                actuated_coords_offset=wp.array(joints_aqio, dtype=int32),
                actuated_dofs_offset=wp.array(joints_adio, dtype=int32),
                cts_offset=wp.array(joints_cio, dtype=int32),
                bid_B=wp.array(joints_bid_B, dtype=int32),
                bid_F=wp.array(joints_bid_F, dtype=int32),
                B_r_Bj=wp.array(joints_B_r_Bj, dtype=vec3f, requires_grad=requires_grad),
                F_r_Fj=wp.array(joints_F_r_Fj, dtype=vec3f, requires_grad=requires_grad),
                X_j=wp.array(joints_X_j, dtype=mat33f, requires_grad=requires_grad),
                q_j_min=wp.array(joints_q_j_min, dtype=float32, requires_grad=requires_grad),
                q_j_max=wp.array(joints_q_j_max, dtype=float32, requires_grad=requires_grad),
                dq_j_max=wp.array(joints_qd_j_max, dtype=float32, requires_grad=requires_grad),
                tau_j_max=wp.array(joints_tau_j_max, dtype=float32, requires_grad=requires_grad),
                q_j_ref=wp.array(joints_q_j_ref, dtype=float32, requires_grad=requires_grad),
            )

            # Create the collision geometries model
            model.cgeoms = CollisionGeometriesModel(
                num_geoms=model.size.sum_of_num_collision_geoms,
                wid=wp.array(cgeoms_wid, dtype=int32),
                gid=wp.array(cgeoms_gid, dtype=int32),
                lid=wp.array(cgeoms_lid, dtype=int32),
                bid=wp.array(cgeoms_bid, dtype=int32),
                sid=wp.array(cgeoms_sid, dtype=int32),
                ptr=wp.array(cgeoms_ptr, dtype=wp.uint64),
                params=wp.array(cgeoms_params, dtype=vec4f, requires_grad=requires_grad),
                offset=wp.array(cgeoms_offset, dtype=transformf, requires_grad=requires_grad),
                mid=wp.array(cgeoms_mid, dtype=int32),
                group=wp.array(cgeoms_group, dtype=uint32),
                collides=wp.array(cgeoms_collides, dtype=uint32),
                margin=wp.array(cgeoms_margin, dtype=float32),
            )

            # Create the physical geometries model
            model.pgeoms = GeometriesModel(
                num_geoms=model.size.sum_of_num_physical_geoms,
                wid=wp.array(pgeoms_wid, dtype=int32),
                gid=wp.array(pgeoms_gid, dtype=int32),
                lid=wp.array(pgeoms_lid, dtype=int32),
                bid=wp.array(pgeoms_bid, dtype=int32),
                sid=wp.array(pgeoms_sid, dtype=int32),
                ptr=wp.array(pgeoms_ptr, dtype=wp.uint64),
                params=wp.array(pgeoms_params, dtype=vec4f, requires_grad=requires_grad),
                offset=wp.array(pgeoms_offset, dtype=transformf, requires_grad=requires_grad),
            )

            # Create the material pairs model
            model.mpairs = MaterialPairsModel(
                num_pairs=model.size.sum_of_num_material_pairs,
                restitution=wp.array(mpairs_rest, dtype=float32),
                static_friction=wp.array(mpairs_static_fric, dtype=float32),
                dynamic_friction=wp.array(mpairs_dynamic_fric, dtype=float32),
            )

        # Return the constructed model data container
        return model

    def compute_required_contact_capacity(
        self,
        max_contacts_per_pair: int,
        max_contacts_per_world: int | None = None,
    ) -> tuple[int, list[int]]:
        # First check if there are any collision geometries
        if self._num_cgeoms == 0:
            return 0, [0] * self.num_worlds

        # Compute the maximum possible number of geom pairs per world
        world_max_contacts = [0] * self.num_worlds
        for w, world in enumerate(self._worlds):
            world_num_geom_pairs = (world.num_collision_geoms * (world.num_collision_geoms - 1)) // 2
            world_max_contacts[w] = world_num_geom_pairs * max_contacts_per_pair

        # Override the per-world maximum contacts if specified in the settings
        if max_contacts_per_world is not None:
            for w in range(self.num_worlds):
                world_max_contacts[w] = max_contacts_per_world

        # Return the per-world maximum contacts list
        return sum(world_max_contacts), world_max_contacts

    def make_collision_candidate_pairs(self, allow_neighbors: bool = False):
        """
        Construct the collision pair candidates for the given ModelBuilder instance.

        Filtering steps:
            1. filter out self-collisions
            2. filter out same-body collisions
            3. filter out collision between different worlds
            4. filter out collisions according to the collision groupings
            5. filter out neighbor collisions for fixed joints
            6. (optional) filter out neighbor collisions for joints w/ DoFs

        Args:
            builder (ModelBuilder): The model builder instance containing the worlds and geometries.
            allow_neighbors (bool, optional): If True, allows neighbor collisions for joints with DoF.

        Returns:
            tuple: A tuple containing:
                - world_num_geom_pairs (list[int]): Number of collision pairs per world.
                - model_geom_pairs (list[tuple[int, int]]): Geometry index pairs for each collision pair in the model.
                - model_pairid (list[int]): Pair IDs for each collision pair in the model.
                - model_wid (list[int]): World indices for each collision pair in the model.
        """
        # Retrieve the number of worlds
        nw = self.num_worlds

        # Extract the per-world info from the builder
        ncg = [self._worlds[i].num_collision_geoms for i in range(nw)]

        # Initialize the lists to store the collision candidate pairs and their properties of each world
        world_num_geom_pairs = []
        model_geom_pairs = []
        model_pairid = []
        model_wid = []

        # Iterate over each world and construct the collision geometry pairs info
        ncg_offset = 0
        for wid in range(nw):
            # Initialize the lists to store the collision candidate pairs and their properties
            world_geom_pair = []
            world_pairid = []
            world_wid = []

            # Iterate over each gid pair and filtering out pairs not viable for collision detection
            # NOTE: k=1 skips diagonal entries to exclude self-collisions
            for gid1_, gid2_ in zip(*np.triu_indices(ncg[wid], k=1), strict=False):
                # Convert the per-world local gids to model gid integers
                gid1 = int(gid1_) + ncg_offset
                gid2 = int(gid2_) + ncg_offset

                # Get references to the geometries
                geom1, geom2 = self.collision_geoms[gid1], self.collision_geoms[gid2]

                # Get body indices of each geom
                bid1, bid2 = geom1.bid, geom2.bid

                # Get world indices of each geom
                wid1, wid2 = geom1.wid, geom2.wid

                # 2. Check for same-body collision
                is_self_collision = bid1 == bid2

                # 3. Check for different-world collision
                in_same_world = wid1 == wid2

                # 4. Check for collision according to the collision groupings
                are_collidable = ((geom1.group & geom2.collides) != 0) and ((geom2.group & geom1.collides) != 0)

                # 5. Check for neighbor collision for fixed and DoF joints
                are_fixed_neighbors = False
                are_dof_neighbors = False
                for joint in self.joints:
                    if (joint.bid_B == bid1 and joint.bid_F == bid2) or (joint.bid_B == bid2 and joint.bid_F == bid1):
                        if joint.dof_type == JointDoFType.FIXED:
                            are_fixed_neighbors = True
                        elif joint.bid_B < 0:
                            pass
                        else:
                            are_dof_neighbors = True
                        break

                # Assign pairid based on filtering results
                if (not is_self_collision) and (in_same_world) and (are_collidable) and (not are_fixed_neighbors):
                    pairid = -1  # TODO: Compute as geom-pair key
                else:
                    continue  # Skip this pair if it does not pass the filtering

                # Apply final check for DoF neighbor collisions
                if (not allow_neighbors) and are_dof_neighbors:
                    continue  # Skip this pair if it does not pass the filtering

                # Append the geometry pair and pairid to the lists
                world_geom_pair.append((gid1, gid2))
                world_pairid.append(pairid)
                world_wid.append(wid)
                msg.debug("Adding broad-phase collision pair candidate: (gid1, gid2): (%d, %d)", gid1, gid2)

            # Append the world collision pairs to the model lists
            world_num_geom_pairs.append(len(world_geom_pair))
            model_geom_pairs.extend(world_geom_pair)
            model_pairid.extend(world_pairid)
            model_wid.extend(world_wid)

            # Update the geometry index offset for the next world
            ncg_offset += ncg[wid]

        # Return the model total collision pair candidates and their properties
        return world_num_geom_pairs, model_geom_pairs, model_pairid, model_wid

    ###
    # Internal Functions
    ###

    def _check_world_index(self, world_index: int) -> WorldDescriptor:
        """
        Checks if the provided world index is valid.

        Args:
            world_index (int): The index of the world to be checked.

        Raises:
            ValueError: If the world index is out of range.
        """
        if self._num_worlds == 0:
            raise ValueError(
                "Model does not contain any worlds. "
                "Please add at least one using `add_world()` before adding model entities."
            )
        if world_index < 0 or world_index >= self._num_worlds:
            raise ValueError(f"Invalid world index (wid): {world_index}. Must be between 0 and {self._num_worlds - 1}.")
        return self._worlds[world_index]

    def _compute_world_offsets(self):
        """
        Computes and sets the model offsets for each world in the model.
        """
        # Initialize the model offsets
        bodies_idx_offset: int = 0
        joints_idx_offset: int = 0
        collision_geoms_idx_offset: int = 0
        physical_geoms_idx_offset: int = 0
        body_dofs_idx_offset: int = 0
        joint_coords_idx_offset: int = 0
        joint_dofs_idx_offset: int = 0
        passive_joint_coords_idx_offset: int = 0
        passive_joint_dofs_idx_offset: int = 0
        actuated_joint_coords_idx_offset: int = 0
        actuated_joint_dofs_idx_offset: int = 0
        joint_cts_idx_offset: int = 0
        # Iterate over each world and set their model offsets
        for world in self._worlds:
            # Set the offsets in the world descriptor to the current values
            world.bodies_idx_offset = bodies_idx_offset
            world.joints_idx_offset = joints_idx_offset
            world.collision_geoms_idx_offset = collision_geoms_idx_offset
            world.physical_geoms_idx_offset = physical_geoms_idx_offset
            world.body_dofs_idx_offset = body_dofs_idx_offset
            world.joint_coords_idx_offset = joint_coords_idx_offset
            world.joint_dofs_idx_offset = joint_dofs_idx_offset
            world.passive_joint_coords_idx_offset = passive_joint_coords_idx_offset
            world.passive_joint_dofs_idx_offset = passive_joint_dofs_idx_offset
            world.actuated_joint_coords_idx_offset = actuated_joint_coords_idx_offset
            world.actuated_joint_dofs_idx_offset = actuated_joint_dofs_idx_offset
            world.joint_cts_idx_offset = joint_cts_idx_offset
            # Update the offsets for the next world
            bodies_idx_offset += world.num_bodies
            joints_idx_offset += world.num_joints
            collision_geoms_idx_offset += world.num_collision_geoms
            physical_geoms_idx_offset += world.num_physical_geoms
            body_dofs_idx_offset += 6 * world.num_bodies
            joint_coords_idx_offset += world.num_joint_coords
            joint_dofs_idx_offset += world.num_joint_dofs
            passive_joint_coords_idx_offset += world.num_passive_joint_coords
            passive_joint_dofs_idx_offset += world.num_passive_joint_dofs
            actuated_joint_coords_idx_offset += world.num_actuated_joint_coords
            actuated_joint_dofs_idx_offset += world.num_actuated_joint_dofs
            joint_cts_idx_offset += world.num_joint_cts

    def _collect_cgeom_max_contact_hints(self) -> tuple[int, list[int]]:
        """
        Collects the `max_contacts` hints from collision geometries.
        """
        model_max_contacts = 0
        world_max_contacts = [0] * self.num_worlds
        for w in range(len(self._worlds)):
            for cgeom_maxnc in self._worlds[w].collision_geometry_max_contacts:
                model_max_contacts += cgeom_maxnc
                world_max_contacts[w] += cgeom_maxnc
        return model_max_contacts, world_max_contacts

    EntityDescriptorType = RigidBodyDescriptor | JointDescriptor | GeometryDescriptor | CollisionGeometryDescriptor
    """A type alias for model entity descriptors."""

    @staticmethod
    def _insert_entity(entity_list: list[EntityDescriptorType], entity: EntityDescriptorType, world_index: int = 0):
        """
        Inserts an entity descriptor into the provided entity list at
        the end of the entities belonging to the specified world index.

        Insertion preserves the order of entities per world.

        Args:
            entity_list (list[EntityDescriptorType]): The list of entity descriptors.
            entity (EntityDescriptorType): The entity descriptor to be inserted.
            world_index (int): The world index to insert the entity into.
        """
        # NOTE: We initialize the last entity index to the length of the list
        # so that if no entities belong to the specified world, the new entity
        # is simply appended to the end of the list.
        last_entity_index = len(entity_list)
        for i, e in enumerate(entity_list):
            if e.wid == world_index:
                last_entity_index = i
        # NOTE: Insert the entity after the last entity of the specified
        # world so that the order of entities per world is preserved.
        entity_list.insert(last_entity_index + 1, entity)

    @staticmethod
    def _check_body_inertia(m_i: float, i_I_i: mat33f):
        """
        Checks if the body inertia is valid.

        Args:
            i_I_i (mat33f): The inertia matrix to be checked.

        Raises:
            ValueError: If the inertia matrix is not symmetric of positive definite.
        """
        # Convert to numpy array for easier checks
        i_I_i_np = np.ndarray(buffer=i_I_i, shape=(3, 3), dtype=np.float32)

        # Perform checks on the inertial properties
        if m_i <= 0.0:
            raise ValueError(f"Invalid body mass: {m_i}. Must be greater than 0.0")
        if not np.allclose(i_I_i_np, i_I_i_np.T, atol=float(FLOAT32_EPS)):
            raise ValueError(f"Invalid body inertia matrix:\n{i_I_i}\nMust be symmetric.")
        if not np.all(np.linalg.eigvals(i_I_i_np) > 0.0):
            raise ValueError(f"Invalid body inertia matrix:\n{i_I_i}\nMust be positive definite.")

    @staticmethod
    def _check_body_pose(q_i: transformf):
        """
        Checks if the body pose is valid.

        Args:
            q_i_0 (transformf): The pose of the body to be checked.

        Raises:
            ValueError: If the body pose is not a valid transformation.
        """
        if not isinstance(q_i, transformf):
            raise TypeError(f"Invalid body pose type: {type(q_i)}. Must be `transformf`.")

        # Extract the orientation quaternion
        if not np.isclose(wp.length(q_i.q), 1.0, atol=float(FLOAT32_EPS)):
            raise ValueError(f"Invalid body pose orientation quaternion: {q_i.q}. Must be a unit quaternion.")
