# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""
KAMINO: Constrained Rigid Multi-Body Model Builder
"""

from __future__ import annotations

import copy
from collections.abc import Iterable

import numpy as np
import warp as wp

from .....geometry import ShapeFlags
from .bodies import RigidBodiesModel, RigidBodyDescriptor
from .geometry import GeometriesModel, GeometryDescriptor
from .gravity import GravityDescriptor, GravityModel
from .joints import (
    JointActuationType,
    JointDescriptor,
    JointDoFType,
    JointsModel,
)
from .materials import MaterialDescriptor, MaterialManager, MaterialPairProperties, MaterialPairsModel, MaterialsModel
from .math import FLOAT32_EPS
from .model import ModelKamino, ModelKaminoInfo
from .shapes import ShapeDescriptorType, max_contacts_for_shape_pair
from .size import SizeKamino
from .time import TimeModel
from .types import Axis, float32, int32, mat33f, transformf, vec2i, vec3f, vec4f, vec6f
from .world import WorldDescriptor

###
# Module interface
###

__all__ = [
    "ModelBuilderKamino",
]


###
# Module configs
###

wp.set_module_options({"enable_backward": False})


###
# Containers
###


class ModelBuilderKamino:
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
        self._device: wp.DeviceLike = None
        self._requires_grad: bool = False

        # Declare and initialize counters
        self._num_bodies: int = 0
        self._num_joints: int = 0
        self._num_geoms: int = 0
        self._num_materials: int = 0
        self._num_bdofs: int = 0
        self._num_joint_coords: int = 0
        self._num_joint_dofs: int = 0
        self._num_joint_passive_coords: int = 0
        self._num_joint_passive_dofs: int = 0
        self._num_joint_actuated_coords: int = 0
        self._num_joint_actuated_dofs: int = 0
        self._num_joint_cts: int = 0
        self._num_joint_kinematic_cts: int = 0
        self._num_joint_dynamic_cts: int = 0

        # Contact capacity settings
        self._max_contacts_per_pair: int | None = None

        # Declare per-world model descriptor sets
        self._up_axes: list[Axis] = []
        self._worlds: list[WorldDescriptor] = []
        self._gravity: list[GravityDescriptor] = []
        self._bodies: list[list[RigidBodyDescriptor]] = []
        self._joints: list[list[JointDescriptor]] = []
        self._geoms: list[list[GeometryDescriptor]] = []
        self._shapes: dict[str, ShapeDescriptorType] = {}

        # Declare a global material manager
        self._materials: MaterialManager = MaterialManager()
        self._num_materials = 1

        # Create a default world if requested
        if default_world:
            self.add_world()

    @property
    def max_contacts_per_pair(self) -> int | None:
        """Maximum contacts per geometry pair override. When set, caps the per-pair contact count
        in `compute_required_contact_capacity()`, reducing the Delassus matrix size."""
        return self._max_contacts_per_pair

    @max_contacts_per_pair.setter
    def max_contacts_per_pair(self, value: int | None):
        self._max_contacts_per_pair = value

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
    def num_geoms(self) -> int:
        """Returns the number of geometries contained in the model."""
        return self._num_geoms

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
        return self._num_joint_coords

    @property
    def num_joint_dofs(self) -> int:
        """Returns the number of joint degrees of freedom contained in the model."""
        return self._num_joint_dofs

    @property
    def num_passive_joint_coords(self) -> int:
        """Returns the number of passive joint coordinates contained in the model."""
        return self._num_joint_passive_coords

    @property
    def num_passive_joint_dofs(self) -> int:
        """Returns the number of passive joint degrees of freedom contained in the model."""
        return self._num_joint_passive_dofs

    @property
    def num_actuated_joint_coords(self) -> int:
        """Returns the number of actuated joint coordinates contained in the model."""
        return self._num_joint_actuated_coords

    @property
    def num_actuated_joint_dofs(self) -> int:
        """Returns the number of actuated joint degrees of freedom contained in the model."""
        return self._num_joint_actuated_dofs

    @property
    def num_joint_cts(self) -> int:
        """Returns the total number of joint constraints contained in the model."""
        return self._num_joint_cts

    @property
    def num_dynamic_joint_cts(self) -> int:
        """Returns the number of dynamic joint constraints contained in the model."""
        return self._num_joint_dynamic_cts

    @property
    def num_kinematic_joint_cts(self) -> int:
        """Returns the number of kinematic joint constraints contained in the model."""
        return self._num_joint_kinematic_cts

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
    def bodies(self) -> list[list[RigidBodyDescriptor]]:
        """Returns the list of lists of body descriptors contained in the model,
        indexed first by world and then by body."""
        return self._bodies

    @property
    def all_bodies(self) -> Iterable[RigidBodyDescriptor]:
        """Returns the collection of all body descriptors contained in the model."""
        return (body for bodies in self._bodies for body in bodies)

    @property
    def joints(self) -> list[list[JointDescriptor]]:
        """Returns the list of joint descriptors contained in the model,
        indexed first by world and then by joint."""
        return self._joints

    @property
    def all_joints(self) -> Iterable[JointDescriptor]:
        """Returns the collection of all joint descriptors contained in the model."""
        return (joint for joints in self._joints for joint in joints)

    @property
    def geoms(self) -> list[list[GeometryDescriptor]]:
        """Returns the list of lists of geometry descriptors contained in the model,
        indexed first by world and then by geometry."""
        return self._geoms

    @property
    def all_geoms(self) -> Iterable[GeometryDescriptor]:
        """Returns the collection of all geometry descriptors contained in the model."""
        return (geom for geoms in self._geoms for geom in geoms)

    @property
    def shapes(self) -> dict[str, ShapeDescriptorType]:
        """Returns the dictionary of shape descriptors contained in the model, indexed by geom uid."""
        return self._shapes

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

        # Extend lists of entities
        self._bodies.append([])
        self._joints.append([])
        self._geoms.append([])

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
        Add a rigid body entity to the model using explicit specifications.

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
        Add a rigid body entity to the model using a descriptor object.

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
        self._bodies[world_index].append(body)

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
        a_j: list[float] | float | None = None,
        b_j: list[float] | float | None = None,
        k_p_j: list[float] | float | None = None,
        k_d_j: list[float] | float | None = None,
        name: str | None = None,
        uid: str | None = None,
        world_index: int = 0,
    ) -> int:
        """
        Add a joint entity to the model using explicit specifications.

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
            a_j (list[float] | float | None): The joint armature along each DoF.
            b_j (list[float] | float | None): The joint damping along each DoF.
            k_p_j (list[float] | float | None): The joint proportional gain along each DoF.
            k_d_j (list[float] | float | None): The joint derivative gain along each DoF.
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
            a_j=a_j,
            b_j=b_j,
            k_p_j=k_p_j,
            k_d_j=k_d_j,
        )

        # Add the body descriptor to the model
        return self.add_joint_descriptor(joint, world_index=world_index)

    def add_joint_descriptor(self, joint: JointDescriptor, world_index: int = 0) -> int:
        """
        Add a joint entity to the model by descriptor.

        Args:
            joint (JointDescriptor):
                The joint descriptor to be added.
            world_index (int):
                The index of the world to which the joint will be added.\n
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
        self._joints[world_index].append(joint)

        # Update model-wide counters
        self._num_joints += 1
        self._num_joint_coords += joint.num_coords
        self._num_joint_dofs += joint.num_dofs
        self._num_joint_passive_coords += joint.num_passive_coords
        self._num_joint_passive_dofs += joint.num_passive_dofs
        self._num_joint_actuated_coords += joint.num_actuated_coords
        self._num_joint_actuated_dofs += joint.num_actuated_dofs
        self._num_joint_cts += joint.num_cts
        self._num_joint_dynamic_cts += joint.num_dynamic_cts
        self._num_joint_kinematic_cts += joint.num_kinematic_cts

        # Return the new joint index
        return joint.jid

    def add_geometry(
        self,
        body: int = -1,
        shape: ShapeDescriptorType | None = None,
        offset: transformf | None = None,
        material: str | int | None = None,
        group: int = 1,
        collides: int = 1,
        max_contacts: int = 0,
        gap: float = 0.0,
        margin: float = 0.0,
        name: str | None = None,
        uid: str | None = None,
        world_index: int = 0,
    ) -> int:
        """
        Add a geometry entity to the model using explicit specifications.

        Args:
            body (int):
                The index of the body to which the geometry will be attached.\n
                Defaults to -1 (world).
            shape (ShapeDescriptorType | None):
                The shape descriptor of the geometry.
            offset (transformf | None):
                The local offset of the geometry relative to the body frame.
            material (str | int | None):
                The name or index of the material assigned to the geometry.
            max_contacts (int):
                The maximum number of contact points for the geometry.\n
                Defaults to 0 (unlimited).
            group (int):
                The collision group of the geometry.\n
                Defaults to 1.
            collides (int):
                The collision mask of the geometry.\n
                Defaults to 1.
            gap (float):
                The collision detection gap of the geometry.\n
                Defaults to 0.0.
            margin (float):
                The artificial surface margin of the geometry.\n
                Defaults to 0.0.
            name (str | None):
                The name of the geometry.\n
                If `None`, a default name will be generated based on the current number of geometries in the model.
            uid (str | None):
                The unique identifier of the geometry.\n
                If `None`, a UUID will be generated.
            world_index (int):
                The index of the world to which the geometry will be added.\n
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
        geom = GeometryDescriptor(
            name=name if name is not None else f"cgeom_{self._num_geoms}",
            uid=uid,
            body=body,
            offset=offset if offset is not None else transformf(),
            shape=shape,
            material=self._materials[material],
            mid=self._materials.index(material),
            group=group,
            collides=collides,
            max_contacts=max_contacts,
            gap=gap,
            margin=margin,
        )

        # Add the body descriptor to the model
        return self.add_geometry_descriptor(geom, world_index=world_index)

    def add_geometry_descriptor(self, geom: GeometryDescriptor, world_index: int = 0) -> int:
        """
        Add a geometry to the model by descriptor.

        Args:
            geom (GeometryDescriptor):
                The geometry descriptor to be added.
            world_index (int):
                The index of the world to which the geometry will be added.\n
                Defaults to the first world with index `0`.

        Returns:
            int: The geometry index of the newly added geometry w.r.t its world.
        """
        # Check if the descriptor is valid
        if not isinstance(geom, GeometryDescriptor):
            raise TypeError(f"Invalid geometry descriptor type: {type(geom)}. Must be `GeometryDescriptor`.")
        assert geom.shape is not None

        # Create a copy of the descriptor without the shape (stored separately)
        _geom = GeometryDescriptor.copy_without_shape(geom)

        # Check if the world index is valid
        world = self._check_world_index(world_index)

        # If the geom material is not assigned, set it to the global default
        if _geom.mid is None:
            _geom.mid = self._materials.default.mid

        # Append body model data
        world.add_geometry(_geom)
        self._geoms[world_index].append(_geom)

        # Update model-wide counters
        self._num_geoms += 1

        if _geom.uid not in self._shapes:
            self._shapes[_geom.uid] = geom.shape

        # Return the new geometry index
        return _geom.gid

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

    def add_builder(self, other: ModelBuilderKamino):
        """
        Extends the contents of the current ModelBuilderKamino with those of another.

        Each builder represents a distinct world, and this method allows for the
        combination of multiple worlds into a single model. The method ensures that the
        indices of the elements in the other builder are adjusted to account for the
        existing elements in the current builder, preventing any index conflicts.

        Arguments:
            other (ModelBuilderKamino): The other ModelBuilderKamino whose contents are to be added to the current.

        Raises:
            ValueError: If the provided builder is not of type `ModelBuilderKamino`.
        """
        # Check if the other builder is of valid type
        if not isinstance(other, ModelBuilderKamino):
            raise TypeError(f"Invalid builder type: {type(other)}. Must be a ModelBuilderKamino instance.")

        # Append the other per-world descriptors
        self._worlds.extend(copy.deepcopy(other._worlds))
        self._gravity.extend(copy.deepcopy(other._gravity))
        self._up_axes.extend(copy.deepcopy(other._up_axes))

        # Append the other per-entity descriptors
        self._bodies.extend(copy.deepcopy(other._bodies))
        self._joints.extend(copy.deepcopy(other._joints))
        self._geoms.extend(copy.deepcopy(other._geoms))

        # Append the other shapes as needed
        for uid, shape in other._shapes.items():
            if uid not in self._shapes:
                self._shapes[uid] = shape

        # Append the other materials
        self._materials.merge(copy.deepcopy(other._materials))

        # Update the world index of the entities in the
        # other builder and update model-wide counters
        for w in range(self._num_worlds, len(self._worlds)):
            # Update world index of the other builder's world
            world = self._worlds[w]
            world.wid = w

            # Update world indices of the other builders entities
            for body in self._bodies[w]:
                body.wid = w
            for joint in self._joints[w]:
                joint.wid = w
            for geom in self._geoms[w]:
                geom.wid = w

            # Update model-wide counters
            self._num_bodies += world.num_bodies
            self._num_joints += world.num_joints
            self._num_geoms += world.num_geoms
            self._num_bdofs += 6 * world.num_bodies
            self._num_joint_coords += world.num_joint_coords
            self._num_joint_dofs += world.num_joint_dofs
            self._num_joint_passive_coords += world.num_passive_joint_coords
            self._num_joint_passive_dofs += world.num_passive_joint_dofs
            self._num_joint_actuated_coords += world.num_actuated_joint_coords
            self._num_joint_actuated_dofs += world.num_actuated_joint_dofs
            self._num_joint_cts += world.num_joint_cts
            self._num_joint_dynamic_cts += world.num_dynamic_joint_cts
            self._num_joint_kinematic_cts += world.num_kinematic_joint_cts

        # Update the number of worlds
        self._num_worlds += other._num_worlds

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
            raise TypeError(f"ModelBuilderKamino: Invalid axis type: {type(axis)}. Must be `Axis`.")

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
            for body in self.bodies[world_index]:
                if body.name == body_key:
                    world.set_base_body(body.bid)
                    return
        raise ValueError(f"Failed to identify the base body in world `{world_index}` given key `{body_key}`.")

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
            for joint in self.joints[world_index]:
                if joint.name == joint_key:
                    world.set_base_joint(joint.jid)
                    return
        raise ValueError(f"Failed to identify the base joint in world `{world_index}` given key `{joint_key}`.")

    ###
    # Model Compilation
    ###

    def finalize(
        self, device: wp.DeviceLike = None, requires_grad: bool = False, base_auto: bool = True
    ) -> ModelKamino:
        """
        Constructs a ModelKamino object from the current ModelBuilderKamino.

        All description data contained in the builder is compiled into a ModelKamino
        object, allocating the necessary data structures on the target device.

        Args:
            device (wp.DeviceLike): The target device for the model data.\n
                If None, the default/preferred device will determined by Warp.
            requires_grad (bool): Whether the model data should support gradients.\n
                Defaults to False.
            base_auto (bool): Whether to automatically select a base body,
                and if possible, a base joint, if neither was set.

        Returns:
            ModelKamino: The constructed ModelKamino object containing the time-invariant simulation data.
        """
        # Number of model worlds
        num_worlds = len(self._worlds)
        if num_worlds == 0:
            raise ValueError("ModelBuilderKamino: Cannot finalize an empty model with zero worlds.")
        if num_worlds != self._num_worlds:
            raise ValueError(
                "ModelBuilderKamino: Inconsistent number of worlds: "
                f"expected {self._num_worlds}, but found {num_worlds}."
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
                follower_idx = self._joints[w][world.base_joint_idx].bid_F  # Note: index among world bodies
                if world.has_base_body:  # Ensure base joint & body are compatible if both were set
                    if world.base_body_idx != follower_idx:
                        raise ValueError(
                            f"ModelBuilderKamino: Inconsistent base body and base joint for world {world.name} ({w})"
                        )
                else:  # Set base body to be the follower of the base joint
                    world.set_base_body(follower_idx)
            elif not world.has_base_body and base_auto:
                world.set_base_body(0)  # Set the base body as the first body
                for jt_idx, joint in enumerate(self._joints[w]):
                    if joint.wid == w and joint.is_unary and joint.is_connected_to_body(world.base_body_idx):
                        # If we find a unary joint connecting the base body to the world, we set this as the base joint
                        world.set_base_joint(jt_idx)
                        break

        ###
        # ModelKamino data collection
        ###

        # Initialize the info data collections
        info_nb = []
        info_nj = []
        info_njp = []
        info_nja = []
        info_nji = []
        info_ng = []
        info_nbd = []
        info_njq = []
        info_njd = []
        info_njpq = []
        info_njpd = []
        info_njaq = []
        info_njad = []
        info_njc = []
        info_njdc = []
        info_njkc = []
        info_bio = []
        info_jio = []
        info_gio = []
        info_bdio = []
        info_jqio = []
        info_jdio = []
        info_jpqio = []
        info_jpdio = []
        info_jaqio = []
        info_jadio = []
        info_jcio = []
        info_jdcio = []
        info_jkcio = []
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
        bodies_label = []
        bodies_wid = []
        bodies_bid = []
        bodies_i_r_com_i = []
        bodies_m_i = []
        bodies_inv_m_i = []
        bodies_i_I_i = []
        bodies_inv_i_I_i = []
        bodies_q_i_0 = []
        bodies_u_i_0 = []

        # Initialize the joint data collections
        joints_label = []
        joints_wid = []
        joints_jid = []
        joints_dofid = []
        joints_actid = []
        joints_q_j_0 = []
        joints_dq_j_0 = []
        joints_bid_B = []
        joints_bid_F = []
        joints_B_r_Bj = []
        joints_F_r_Fj = []
        joints_X_j = []
        joints_q_j_min = []
        joints_q_j_max = []
        joints_qd_j_max = []
        joints_tau_j_max = []
        joints_a_j = []
        joints_b_j = []
        joints_k_p_j = []
        joints_k_d_j = []
        joints_ncoords_j = []
        joints_ndofs_j = []
        joints_ncts_j = []
        joints_nkincts_j = []
        joints_ndyncts_j = []
        joints_q_start = []
        joints_dq_start = []
        joints_pq_start = []
        joints_pdq_start = []
        joints_aq_start = []
        joints_adq_start = []
        joints_cts_start = []
        joints_dcts_start = []
        joints_kcts_start = []

        # Initialize the collision geometry data collections
        geoms_label = []
        geoms_wid = []
        geoms_gid = []
        geoms_bid = []
        geoms_type = []
        geoms_flags = []
        geoms_ptr = []
        geoms_params = []
        geoms_offset = []
        geoms_material = []
        geoms_group = []
        geoms_collides = []
        geoms_gap = []
        geoms_margin = []

        # Initialize the material data collections
        materials_rest = []
        materials_static_fric = []
        materials_dynamic_fric = []
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
                info_nji.append(world.num_dynamic_joints)
                info_ng.append(world.num_geoms)
                info_nbd.append(world.num_body_dofs)
                info_njq.append(world.num_joint_coords)
                info_njd.append(world.num_joint_dofs)
                info_njpq.append(world.num_passive_joint_coords)
                info_njpd.append(world.num_passive_joint_dofs)
                info_njaq.append(world.num_actuated_joint_coords)
                info_njad.append(world.num_actuated_joint_dofs)
                info_njc.append(world.num_joint_cts)
                info_njdc.append(world.num_dynamic_joint_cts)
                info_njkc.append(world.num_kinematic_joint_cts)
                info_bio.append(world.bodies_idx_offset)
                info_jio.append(world.joints_idx_offset)
                info_gio.append(world.geoms_idx_offset)

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
                info_jpqio.append(world.joint_passive_coords_idx_offset)
                info_jpdio.append(world.joint_passive_dofs_idx_offset)
                info_jaqio.append(world.joint_actuated_coords_idx_offset)
                info_jadio.append(world.joint_actuated_dofs_idx_offset)
                info_jcio.append(world.joint_cts_idx_offset)
                info_jdcio.append(world.joint_dynamic_cts_idx_offset)
                info_jkcio.append(world.joint_kinematic_cts_idx_offset)
                info_base_bid.append((world.base_body_idx + world.bodies_idx_offset) if world.has_base_body else -1)
                info_base_jid.append((world.base_joint_idx + world.joints_idx_offset) if world.has_base_joint else -1)

        # A helper function to collect model gravity data
        def collect_gravity_model_data():
            for w in range(num_worlds):
                gravity_g_dir_acc.append(self._gravity[w].dir_accel())
                gravity_vector.append(self._gravity[w].vector())

        # A helper function to collect model bodies data
        def collect_body_model_data():
            for body in self.all_bodies:
                bodies_label.append(body.name)
                bodies_wid.append(body.wid)
                bodies_bid.append(body.bid)
                bodies_i_r_com_i.append(body.i_r_com_i)
                bodies_m_i.append(body.m_i)
                bodies_inv_m_i.append(1.0 / body.m_i)
                bodies_i_I_i.append(body.i_I_i)
                bodies_inv_i_I_i.append(wp.inverse(body.i_I_i))
                bodies_q_i_0.append(body.q_i_0)
                bodies_u_i_0.append(body.u_i_0)

        # A helper function to collect model joints data
        def collect_joint_model_data():
            for joint in self.all_joints:
                world = self._worlds[joint.wid]
                world_bio = world.bodies_idx_offset
                joints_label.append(joint.name)
                joints_wid.append(joint.wid)
                joints_jid.append(joint.jid)
                joints_dofid.append(joint.dof_type.value)
                joints_actid.append(joint.act_type.value)
                joints_B_r_Bj.append(joint.B_r_Bj)
                joints_F_r_Fj.append(joint.F_r_Fj)
                joints_X_j.append(joint.X_j)
                joints_q_j_0.extend(joint.dof_type.reference_coords)
                joints_dq_j_0.extend(joint.dof_type.num_dofs * [0.0])
                joints_q_j_min.extend(joint.q_j_min)
                joints_q_j_max.extend(joint.q_j_max)
                joints_qd_j_max.extend(joint.dq_j_max)
                joints_tau_j_max.extend(joint.tau_j_max)
                joints_a_j.extend(joint.a_j)
                joints_b_j.extend(joint.b_j)
                joints_k_p_j.extend(joint.k_p_j)
                joints_k_d_j.extend(joint.k_d_j)
                joints_ncoords_j.append(joint.num_coords)
                joints_ndofs_j.append(joint.num_dofs)
                joints_ncts_j.append(joint.num_cts)
                joints_ndyncts_j.append(joint.num_dynamic_cts)
                joints_nkincts_j.append(joint.num_kinematic_cts)
                joints_q_start.append(joint.coords_offset + world.joint_coords_idx_offset)
                joints_dq_start.append(joint.dofs_offset + world.joint_dofs_idx_offset)
                joints_pq_start.append(joint.passive_coords_offset + world.joint_passive_coords_idx_offset)
                joints_pdq_start.append(joint.passive_dofs_offset + world.joint_passive_dofs_idx_offset)
                joints_aq_start.append(joint.actuated_coords_offset + world.joint_actuated_coords_idx_offset)
                joints_adq_start.append(joint.actuated_dofs_offset + world.joint_actuated_dofs_idx_offset)
                joints_cts_start.append(joint.cts_offset + world.joint_cts_idx_offset)
                joints_dcts_start.append(joint.dynamic_cts_offset + world.joint_dynamic_cts_idx_offset)
                joints_kcts_start.append(joint.kinematic_cts_offset + world.joint_kinematic_cts_idx_offset)
                joints_bid_B.append(joint.bid_B + world_bio if joint.bid_B >= 0 else -1)
                joints_bid_F.append(joint.bid_F + world_bio if joint.bid_F >= 0 else -1)

            # Append the N+1 entry (grand total) to each offset list
            joints_q_start.append(self._num_joint_coords)
            joints_dq_start.append(self._num_joint_dofs)
            joints_pq_start.append(self._num_joint_passive_coords)
            joints_pdq_start.append(self._num_joint_passive_dofs)
            joints_aq_start.append(self._num_joint_actuated_coords)
            joints_adq_start.append(self._num_joint_actuated_dofs)
            joints_cts_start.append(self._num_joint_cts)
            joints_dcts_start.append(self._num_joint_dynamic_cts)
            joints_kcts_start.append(self._num_joint_kinematic_cts)

        # A helper function to collect model collision geometries data
        def collect_geometry_model_data():
            shape_ptrs = {}
            for uid, shape in self._shapes.items():
                # If the geometry has a Mesh, SDF or HField source,
                # finalize it and retrieve the mesh pointer/index
                if shape.type.is_explicit:
                    shape_ptrs[uid] = shape.data.finalize(device=device)
                # Otherwise, append a null (i.e. zero-valued) pointer
                else:
                    shape_ptrs[uid] = 0
            for geom in self.all_geoms:
                shape = self._shapes[geom.uid]
                geoms_label.append(geom.name)
                geoms_wid.append(geom.wid)
                geoms_gid.append(geom.gid)
                geoms_bid.append(geom.body + self._worlds[geom.wid].bodies_idx_offset if geom.body >= 0 else -1)
                geoms_type.append(shape.type.value)
                geoms_flags.append(geom.flags)
                geoms_params.append(shape.paramsvec)
                geoms_offset.append(geom.offset)
                geoms_material.append(geom.mid)
                geoms_group.append(geom.group)
                geoms_collides.append(geom.collides)
                geoms_gap.append(geom.gap)
                geoms_margin.append(geom.margin)
                geoms_ptr.append(shape_ptrs[geom.uid])

        # A helper function to collect model material-pairs data
        def collect_material_pairs_model_data():
            materials_rest.append(self._materials.restitution_vector())
            materials_static_fric.append(self._materials.static_friction_vector())
            materials_dynamic_fric.append(self._materials.dynamic_friction_vector())
            mpairs_rest.append(self._materials.restitution_matrix())
            mpairs_static_fric.append(self._materials.static_friction_matrix())
            mpairs_dynamic_fric.append(self._materials.dynamic_friction_matrix())

        # Collect model data
        collect_model_info_data()
        collect_gravity_model_data()
        collect_body_model_data()
        collect_joint_model_data()
        collect_geometry_model_data()
        collect_material_pairs_model_data()

        ###
        # Host-side model size meta-data
        ###

        # Compute the sum/max of model entities
        model_size = SizeKamino(
            num_worlds=num_worlds,
            sum_of_num_bodies=self._num_bodies,
            max_of_num_bodies=max([world.num_bodies for world in self._worlds]),
            sum_of_num_joints=self._num_joints,
            max_of_num_joints=max([world.num_joints for world in self._worlds]),
            sum_of_num_passive_joints=sum([world.num_passive_joints for world in self._worlds]),
            max_of_num_passive_joints=max([world.num_passive_joints for world in self._worlds]),
            sum_of_num_actuated_joints=sum([world.num_actuated_joints for world in self._worlds]),
            max_of_num_actuated_joints=max([world.num_actuated_joints for world in self._worlds]),
            sum_of_num_dynamic_joints=sum([world.num_dynamic_joints for world in self._worlds]),
            max_of_num_dynamic_joints=max([world.num_dynamic_joints for world in self._worlds]),
            sum_of_num_geoms=self._num_geoms,
            max_of_num_geoms=max([world.num_geoms for world in self._worlds]),
            sum_of_num_materials=self._materials.num_materials,
            max_of_num_materials=self._materials.num_materials,
            sum_of_num_material_pairs=self._materials.num_material_pairs,
            max_of_num_material_pairs=self._materials.num_material_pairs,
            # Compute the sum/max of model coords, DoFs and constraints
            sum_of_num_body_dofs=self._num_bdofs,
            max_of_num_body_dofs=max([world.num_body_dofs for world in self._worlds]),
            sum_of_num_joint_coords=self._num_joint_coords,
            max_of_num_joint_coords=max([world.num_joint_coords for world in self._worlds]),
            sum_of_num_joint_dofs=self._num_joint_dofs,
            max_of_num_joint_dofs=max([world.num_joint_dofs for world in self._worlds]),
            sum_of_num_passive_joint_coords=self._num_joint_passive_coords,
            max_of_num_passive_joint_coords=max([world.num_passive_joint_coords for world in self._worlds]),
            sum_of_num_passive_joint_dofs=self._num_joint_passive_dofs,
            max_of_num_passive_joint_dofs=max([world.num_passive_joint_dofs for world in self._worlds]),
            sum_of_num_actuated_joint_coords=self._num_joint_actuated_coords,
            max_of_num_actuated_joint_coords=max([world.num_actuated_joint_coords for world in self._worlds]),
            sum_of_num_actuated_joint_dofs=self._num_joint_actuated_dofs,
            max_of_num_actuated_joint_dofs=max([world.num_actuated_joint_dofs for world in self._worlds]),
            sum_of_num_joint_cts=self._num_joint_cts,
            max_of_num_joint_cts=max([world.num_joint_cts for world in self._worlds]),
            sum_of_num_dynamic_joint_cts=self._num_joint_dynamic_cts,
            max_of_num_dynamic_joint_cts=max([world.num_dynamic_joint_cts for world in self._worlds]),
            sum_of_num_kinematic_joint_cts=self._num_joint_kinematic_cts,
            max_of_num_kinematic_joint_cts=max([world.num_kinematic_joint_cts for world in self._worlds]),
            # Initialize unilateral counts (limits, and contacts) to zero
            sum_of_max_limits=0,
            max_of_max_limits=0,
            sum_of_max_contacts=0,
            max_of_max_contacts=0,
            sum_of_max_unilaterals=0,
            max_of_max_unilaterals=0,
            # Initialize total constraint counts to the same as the joint constraint counts
            sum_of_max_total_cts=self._num_joint_cts,
            max_of_max_total_cts=max([world.num_joint_cts for world in self._worlds]),
        )

        # Append total number of bodies to body offsets
        info_bio.append(model_size.sum_of_num_bodies)

        ###
        # Collision detection and contact-allocation meta-data
        ###

        # Generate the lists of collidable and excluded geometry pairs for the entire model
        model_collidable_pairs, collidable_pairs_offset = self.make_collision_candidate_pairs()
        model_excluded_pairs, _ = self.make_collision_excluded_pairs()

        # Retrieve the number of collidable geoms for each world and
        # for the entire model based on the generated candidate pairs
        _, model_num_collidables = self.compute_num_collidable_geoms(
            collidable_geom_pairs=model_collidable_pairs, collidable_pairs_offset=collidable_pairs_offset
        )

        # Compute the maximum number of contacts required for the model and each world
        # NOTE: This is a conservative estimate based on the maximum per-world geom-pairs
        model_required_contacts, world_required_contacts = self.compute_required_contact_capacity(
            collidable_geom_pairs=model_collidable_pairs,
            collidable_pairs_offset=collidable_pairs_offset,
            max_contacts_per_pair=self._max_contacts_per_pair,
        )

        ###
        # On-device data allocation
        ###

        # Allocate the model data on the target device
        with wp.ScopedDevice(device):
            # Create the immutable model info arrays from the collected data
            model_info = ModelKaminoInfo(
                num_worlds=num_worlds,
                num_bodies=wp.array(info_nb, dtype=int32),
                num_joints=wp.array(info_nj, dtype=int32),
                num_passive_joints=wp.array(info_njp, dtype=int32),
                num_actuated_joints=wp.array(info_nja, dtype=int32),
                num_dynamic_joints=wp.array(info_nji, dtype=int32),
                num_geoms=wp.array(info_ng, dtype=int32),
                num_body_dofs=wp.array(info_nbd, dtype=int32),
                num_joint_coords=wp.array(info_njq, dtype=int32),
                num_joint_dofs=wp.array(info_njd, dtype=int32),
                num_passive_joint_coords=wp.array(info_njpq, dtype=int32),
                num_passive_joint_dofs=wp.array(info_njpd, dtype=int32),
                num_actuated_joint_coords=wp.array(info_njaq, dtype=int32),
                num_actuated_joint_dofs=wp.array(info_njad, dtype=int32),
                num_joint_cts=wp.array(info_njc, dtype=int32),
                num_joint_dynamic_cts=wp.array(info_njdc, dtype=int32),
                num_joint_kinematic_cts=wp.array(info_njkc, dtype=int32),
                bodies_offset=wp.array(info_bio, dtype=int32),
                joints_offset=wp.array(info_jio, dtype=int32),
                geoms_offset=wp.array(info_gio, dtype=int32),
                body_dofs_offset=wp.array(info_bdio, dtype=int32),
                joint_coords_offset=wp.array(info_jqio, dtype=int32),
                joint_dofs_offset=wp.array(info_jdio, dtype=int32),
                joint_passive_coords_offset=wp.array(info_jpqio, dtype=int32),
                joint_passive_dofs_offset=wp.array(info_jpdio, dtype=int32),
                joint_actuated_coords_offset=wp.array(info_jaqio, dtype=int32),
                joint_actuated_dofs_offset=wp.array(info_jadio, dtype=int32),
                joint_cts_offset=wp.array(info_jcio, dtype=int32),
                joint_dynamic_cts_offset=wp.array(info_jdcio, dtype=int32),
                joint_kinematic_cts_offset=wp.array(info_jkcio, dtype=int32),
                base_body_index=wp.array(info_base_bid, dtype=int32),
                base_joint_index=wp.array(info_base_jid, dtype=int32),
                mass_min=wp.array(info_mass_min, dtype=float32),
                mass_max=wp.array(info_mass_max, dtype=float32),
                mass_total=wp.array(info_mass_total, dtype=float32),
                inertia_total=wp.array(info_inertia_total, dtype=float32),
            )

            # Create the model time data
            model_time = TimeModel(dt=wp.zeros(num_worlds, dtype=float32), inv_dt=wp.zeros(num_worlds, dtype=float32))

            # Construct model gravity data
            model_gravity = GravityModel(
                g_dir_acc=wp.array(gravity_g_dir_acc, dtype=vec4f),
                vector=wp.array(gravity_vector, dtype=vec4f, requires_grad=requires_grad),
            )

            # Create the bodies model
            model_bodies = RigidBodiesModel(
                num_bodies=model_size.sum_of_num_bodies,
                label=bodies_label,
                wid=wp.array(bodies_wid, dtype=int32),
                bid=wp.array(bodies_bid, dtype=int32),
                i_r_com_i=wp.array(bodies_i_r_com_i, dtype=vec3f, requires_grad=requires_grad),
                m_i=wp.array(bodies_m_i, dtype=float32, requires_grad=requires_grad),
                inv_m_i=wp.array(bodies_inv_m_i, dtype=float32, requires_grad=requires_grad),
                i_I_i=wp.array(bodies_i_I_i, dtype=mat33f, requires_grad=requires_grad),
                inv_i_I_i=wp.array(bodies_inv_i_I_i, dtype=mat33f, requires_grad=requires_grad),
                q_i_0=wp.array(bodies_q_i_0, dtype=transformf, requires_grad=requires_grad),
                u_i_0=wp.array(bodies_u_i_0, dtype=vec6f, requires_grad=requires_grad),
            )

            # Create the joints model
            model_joints = JointsModel(
                num_joints=model_size.sum_of_num_joints,
                label=joints_label,
                wid=wp.array(joints_wid, dtype=int32),
                jid=wp.array(joints_jid, dtype=int32),
                dof_type=wp.array(joints_dofid, dtype=int32),
                act_type=wp.array(joints_actid, dtype=int32),
                bid_B=wp.array(joints_bid_B, dtype=int32),
                bid_F=wp.array(joints_bid_F, dtype=int32),
                B_r_Bj=wp.array(joints_B_r_Bj, dtype=vec3f, requires_grad=requires_grad),
                F_r_Fj=wp.array(joints_F_r_Fj, dtype=vec3f, requires_grad=requires_grad),
                X_j=wp.array(joints_X_j, dtype=mat33f, requires_grad=requires_grad),
                q_j_min=wp.array(joints_q_j_min, dtype=float32, requires_grad=requires_grad),
                q_j_max=wp.array(joints_q_j_max, dtype=float32, requires_grad=requires_grad),
                dq_j_max=wp.array(joints_qd_j_max, dtype=float32, requires_grad=requires_grad),
                tau_j_max=wp.array(joints_tau_j_max, dtype=float32, requires_grad=requires_grad),
                a_j=wp.array(joints_a_j, dtype=float32, requires_grad=requires_grad),
                b_j=wp.array(joints_b_j, dtype=float32, requires_grad=requires_grad),
                k_p_j=wp.array(joints_k_p_j, dtype=float32, requires_grad=requires_grad),
                k_d_j=wp.array(joints_k_d_j, dtype=float32, requires_grad=requires_grad),
                q_j_0=wp.array(joints_q_j_0, dtype=float32, requires_grad=requires_grad),
                dq_j_0=wp.array(joints_dq_j_0, dtype=float32, requires_grad=requires_grad),
                num_coords=wp.array(joints_ncoords_j, dtype=int32),
                num_dofs=wp.array(joints_ndofs_j, dtype=int32),
                num_cts=wp.array(joints_ncts_j, dtype=int32),
                num_dynamic_cts=wp.array(joints_ndyncts_j, dtype=int32),
                num_kinematic_cts=wp.array(joints_nkincts_j, dtype=int32),
                coords_offset=wp.array(joints_q_start, dtype=int32),
                dofs_offset=wp.array(joints_dq_start, dtype=int32),
                passive_coords_offset=wp.array(joints_pq_start, dtype=int32),
                passive_dofs_offset=wp.array(joints_pdq_start, dtype=int32),
                actuated_coords_offset=wp.array(joints_aq_start, dtype=int32),
                actuated_dofs_offset=wp.array(joints_adq_start, dtype=int32),
                cts_offset=wp.array(joints_cts_start, dtype=int32),
                dynamic_cts_offset=wp.array(joints_dcts_start, dtype=int32),
                kinematic_cts_offset=wp.array(joints_kcts_start, dtype=int32),
            )

            # Create the collision geometries model
            model_geoms = GeometriesModel(
                num_geoms=model_size.sum_of_num_geoms,
                num_collidable=model_num_collidables,
                num_collidable_pairs=len(model_collidable_pairs),
                num_excluded_pairs=len(model_excluded_pairs),
                model_minimum_contacts=model_required_contacts,
                world_minimum_contacts=world_required_contacts,
                label=geoms_label,
                wid=wp.array(geoms_wid, dtype=int32),
                gid=wp.array(geoms_gid, dtype=int32),
                bid=wp.array(geoms_bid, dtype=int32),
                type=wp.array(geoms_type, dtype=int32),
                flags=wp.array(geoms_flags, dtype=int32),
                ptr=wp.array(geoms_ptr, dtype=wp.uint64),
                params=wp.array(geoms_params, dtype=vec3f),
                offset=wp.array(geoms_offset, dtype=transformf),
                material=wp.array(geoms_material, dtype=int32),
                group=wp.array(geoms_group, dtype=int32),
                gap=wp.array(geoms_gap, dtype=float32),
                margin=wp.array(geoms_margin, dtype=float32),
                collidable_pairs=wp.array(np.array(model_collidable_pairs), dtype=vec2i),
                excluded_pairs=wp.array(np.array(model_excluded_pairs), dtype=vec2i),
            )

            # Create the material pairs model
            model_materials = MaterialsModel(
                num_materials=model_size.sum_of_num_materials,
                restitution=wp.array(materials_rest[0], dtype=float32),
                static_friction=wp.array(materials_static_fric[0], dtype=float32),
                dynamic_friction=wp.array(materials_dynamic_fric[0], dtype=float32),
            )

            # Create the material pairs model
            model_material_pairs = MaterialPairsModel(
                num_material_pairs=model_size.sum_of_num_material_pairs,
                restitution=wp.array(mpairs_rest[0], dtype=float32),
                static_friction=wp.array(mpairs_static_fric[0], dtype=float32),
                dynamic_friction=wp.array(mpairs_dynamic_fric[0], dtype=float32),
            )

        # Construct and return the complete model container
        return ModelKamino(
            _device=device,
            _requires_grad=requires_grad,
            size=model_size,
            info=model_info,
            time=model_time,
            gravity=model_gravity,
            bodies=model_bodies,
            joints=model_joints,
            geoms=model_geoms,
            materials=model_materials,
            material_pairs=model_material_pairs,
        )

    ###
    # Utilities
    ###

    def make_collision_candidate_pairs(self, allow_neighbors: bool = False) -> tuple[list[tuple[int, int]], list[int]]:
        """
        Constructs the collision pair candidates.

        Filtering steps:
            1. filter out self-collisions
            2. filter out same-body collisions
            3. filter out collision between different worlds
            4. filter out collisions according to the collision groupings
            5. filter out neighbor collisions for fixed joints
            6. (optional) filter out neighbor collisions for joints w/ DoFs

        Args:
            allow_neighbors (bool, optional):
                If True, includes geom-pairs with corresponding
                bodies that are neighbors via joints with DoF.

        Returns:
            model_collidable_pairs: A sorted list of geom index pairs (gid1, gid2) that are candidates for
                collision detection
            collidable_pairs_offset: A list of per-world offsets into model_collidable_pairs (with one
                extra entry giving the total length of model_collidable_pairs)
        """
        # Retrieve the number of worlds
        nw = self.num_worlds

        # Extract the per-world info from the builder
        ncg = [self._worlds[i].num_geoms for i in range(nw)]

        # Initialize the lists to store the collision candidate pairs and their properties of each world
        model_candidate_pairs = []
        candidate_pairs_offset = []

        # Iterate over each world and construct the collision geometry pairs info
        ncg_offset = 0
        for wid in range(nw):
            # Precompute body adjacency matrix for this world.
            # If we allow neighbor collisions, adjacent bodies are only bodies connected by a fixed joint.
            # Otherwise, they are all bodies connected by a non-free joint.
            # Note: for convenience we shift body indices by 1 to account for the -1 body of unary joints
            num_bodies = self.worlds[wid].num_bodies
            adjacent_bodies = np.zeros((num_bodies + 1, num_bodies + 1), dtype=np.int32)
            for joint in self.joints[wid]:
                if joint.dof_type == JointDoFType.FREE:
                    continue
                if not allow_neighbors or joint.dof_type == JointDoFType.FIXED:
                    adjacent_bodies[joint.bid_B + 1, joint.bid_F + 1] = 1
                    adjacent_bodies[joint.bid_F + 1, joint.bid_B + 1] = 1

            # Initialize the lists to store the collision candidate pairs and their properties
            world_candidate_pairs = []
            candidate_pairs_offset.append(len(model_candidate_pairs))

            # Iterate over each gid pair and filtering out pairs not viable for collision detection
            # NOTE: k=1 skips diagonal entries to exclude self-collisions
            for gid1_, gid2_ in zip(*np.triu_indices(ncg[wid], k=1), strict=False):
                # Convert the per-world local gids to model gid integers
                gid1 = int(gid1_) + ncg_offset
                gid2 = int(gid2_) + ncg_offset

                # Get references to the geometries
                geom1, geom2 = self.geoms[wid][gid1_], self.geoms[wid][gid2_]
                assert geom1.wid == wid
                assert geom2.wid == wid

                # Skip if either geometry is non-collidable
                if not geom1.is_collidable or not geom2.is_collidable:
                    continue

                # Get body indices of each geom
                bid1, bid2 = geom1.body, geom2.body

                # 2. Check for same-body collision
                is_self_collision = bid1 == bid2

                # 4. Check for collision according to the collision groupings
                are_collidable = ((geom1.group & geom2.collides) != 0) and ((geom2.group & geom1.collides) != 0)

                # Skip this pair if it does not pass the first round of filtering
                if is_self_collision or not are_collidable:
                    continue

                # 5. and 6. Check for neighbor collision for fixed and DoF joints
                if adjacent_bodies[bid1 + 1, bid2 + 1]:
                    continue

                # Append the geometry pair to the list of world collision candidates
                world_candidate_pairs.append((min(gid1, gid2), max(gid1, gid2)))

            # Sort the candidate pairs list for efficient lookup
            # on the device if there are any candidate pairs
            if len(world_candidate_pairs) > 0:
                world_candidate_pairs.sort()

            # Append the world collision pairs to the model lists
            model_candidate_pairs.extend(world_candidate_pairs)

            # Update the geometry index offset for the next world
            ncg_offset += ncg[wid]

        candidate_pairs_offset.append(len(model_candidate_pairs))

        # Return the model total candidate pairs
        return model_candidate_pairs, candidate_pairs_offset

    def make_collision_excluded_pairs(self, allow_neighbors: bool = False) -> tuple[list[tuple[int, int]], list[int]]:
        """
        Builds a sorted array of shape pairs that the NXN/SAP broadphase should exclude.

        Encodes the same filtering rules as
        :meth:`ModelBuilderKamino.make_collision_candidate_pairs` (same-body, group/collides
        bitmask, fixed-joint and DoF-joint neighbours) but returns the *complement*:
        pairs that should **not** collide.

        Args:
            allow_neighbors (bool, optional):
                If True, does not exclude geom-pairs with corresponding
                bodies that are neighbors via joints with DoF.

        Returns:
            model_excluded_pairs: A sorted list of geom index pairs (gid1, gid2) that should be
                excluded from collision detection.
            excluded_pairs_offset: A list of per-world offsets into model_excluded_pairs (with one
                extra entry giving the total length of model_excluded_pairs)
        """

        model_excluded_pairs: list[tuple[int, int]] = []
        excluded_pairs_offset = []
        ncg_offset = 0
        for wid in range(self.num_worlds):
            # Precompute body adjacency matrix for this world.
            # If we allow neighbor collisions, adjacent bodies are only bodies connected by a fixed joint.
            # Otherwise, they are all bodies connected by a non-free joint.
            # Note: for convenience we shift body indices by 1 to account for the -1 body of unary joints
            num_bodies = self.worlds[wid].num_bodies
            adjacent_bodies = np.zeros((num_bodies + 1, num_bodies + 1), dtype=np.int32)
            for joint in self.joints[wid]:
                if joint.dof_type == JointDoFType.FREE:
                    continue
                if not allow_neighbors or joint.dof_type == JointDoFType.FIXED:
                    adjacent_bodies[joint.bid_B + 1, joint.bid_F + 1] = 1
                    adjacent_bodies[joint.bid_F + 1, joint.bid_B + 1] = 1

            world_excluded_pairs = []
            excluded_pairs_offset.append(len(model_excluded_pairs))
            ncg = self.worlds[wid].num_geoms
            for idx1 in range(ncg):
                gid1 = idx1 + ncg_offset
                geom1 = self.geoms[wid][idx1]
                for idx2 in range(idx1 + 1, ncg):
                    gid2 = idx2 + ncg_offset
                    geom2 = self.geoms[wid][idx2]

                    # Skip if either geometry is non-collidable since they won't be considered in the broadphase anyway
                    if (geom1.flags & ShapeFlags.COLLIDE_SHAPES == 0) or (geom2.flags & ShapeFlags.COLLIDE_SHAPES == 0):
                        continue

                    # Form the candidate pair tuple with sorted geom index order
                    candidate_pair = (min(gid1, gid2), max(gid1, gid2))

                    # Same-body collision
                    if geom1.body == geom2.body:
                        world_excluded_pairs.append(candidate_pair)
                        continue

                    # Group/collides bitmask check
                    if not ((geom1.group & geom2.collides) != 0 and (geom2.group & geom1.collides) != 0):
                        world_excluded_pairs.append(candidate_pair)
                        continue

                    # Fixed-joint / DoF-joint neighbour check
                    if adjacent_bodies[geom1.body + 1, geom2.body + 1]:
                        world_excluded_pairs.append(candidate_pair)
                        continue

            # Sort the excluded pairs list for efficient lookup
            # on the device if there are any excluded pairs
            if len(world_excluded_pairs) > 0:
                world_excluded_pairs.sort()

            # Append the world excluded pairs to the model lists
            model_excluded_pairs.extend(world_excluded_pairs)

            ncg_offset += ncg

        excluded_pairs_offset.append(len(model_excluded_pairs))

        # Return the model total excluded pairs and their properties
        return model_excluded_pairs, excluded_pairs_offset

    def compute_num_collidable_geoms(
        self,
        collidable_geom_pairs: list[tuple[int, int]] | None = None,
        collidable_pairs_offset: list[int] | None = None,
    ) -> tuple[list[int], int]:
        """
        Computes the number of unique collidable geometries from the provided list of collidable geometry pairs.

        Args:
            collidable_geom_pairs (list[tuple[int, int]], optional):
                A list of geom-pair indices `(gid1, gid2)` (absolute w.r.t the model).\n
                If `None`, the number of collidable geometries will
                be extracted by exhaustively checking all geometries.
            collidable_pairs_offset: (list[int], optional):
                A list of per-world offsets into collidable_geom_pairs (with one
                extra entry giving the total length of collidable_geom_pairs).
                Cannot be `None` if collidable_geom_pairs is provided.


        Returns:
            (world_num_collidables, model_num_collidables):
                A tuple containing a list of unique collidable geometries per world and the total over the model.

        """
        # If an explicit list of collidable geometry pairs is provided,
        # compute the number of unique collidable geometries from the pairs
        if collidable_geom_pairs is not None:
            assert collidable_pairs_offset is not None
            world_num_collidables = [0] * self.num_worlds
            for wid in range(self.num_worlds):
                collidable_geoms: set[int] = set()
                for pair_id in range(collidable_pairs_offset[wid], collidable_pairs_offset[wid + 1]):
                    pair = collidable_geom_pairs[pair_id]
                    collidable_geoms.add(pair[0])
                    collidable_geoms.add(pair[1])
                world_num_collidables[wid] = len(collidable_geoms)
            return world_num_collidables, sum(world_num_collidables)

        # Otherwise, compute the number of collidable geometries by checking all geometries
        world_num_collidables = [0] * self.num_worlds
        for wid in range(self.num_worlds):
            for geom in self.geoms[wid]:
                if geom.is_collidable:
                    world_num_collidables[wid] += 1
        return world_num_collidables, sum(world_num_collidables)

    def compute_required_contact_capacity(
        self,
        collidable_geom_pairs: list[tuple[int, int]] | None = None,
        collidable_pairs_offset: list[int] | None = None,
        max_contacts_per_pair: int | None = None,
        max_contacts_per_world: int | None = None,
    ) -> tuple[int, list[int]]:
        # First check if there are any collision geometries
        if self._num_geoms == 0:
            return 0, [0] * self.num_worlds

        # Generate the collision candidate pairs if not provided
        if collidable_geom_pairs is None:
            collidable_geom_pairs, collidable_pairs_offset = self.make_collision_candidate_pairs()
        else:
            assert collidable_pairs_offset is not None

        # Generate the cumsum of geometries per world, to convert global to local geom indices
        num_geoms = np.array([self.worlds[i].num_geoms for i in range(self.num_worlds)])
        geom_offsets = np.concatenate(([0], num_geoms.cumsum()))

        # Compute the maximum possible number of geom pairs per world
        world_max_contacts = [0] * self.num_worlds
        for wid in range(self.num_worlds):
            offset = geom_offsets[wid]
            for pair_id in range(collidable_pairs_offset[wid], collidable_pairs_offset[wid + 1]):
                geom_pair = collidable_geom_pairs[pair_id]
                g1 = int(geom_pair[0]) - offset
                g2 = int(geom_pair[1]) - offset
                geom1 = self._geoms[wid][g1]
                geom2 = self._geoms[wid][g2]
                shape1 = self._shapes[geom1.uid]
                shape2 = self._shapes[geom2.uid]
                if shape1.type > shape2.type:
                    g1, g2 = g2, g1
                    geom1, geom2 = geom2, geom1
                    shape1, shape2 = shape2, shape1
                num_contacts_a, num_contacts_b = max_contacts_for_shape_pair(
                    type_a=int(shape1.type),
                    type_b=int(shape2.type),
                )
                num_contacts = num_contacts_a + num_contacts_b
                if max_contacts_per_pair is not None:
                    world_max_contacts[geom1.wid] += min(num_contacts, max_contacts_per_pair)
                else:
                    world_max_contacts[geom1.wid] += num_contacts

        # Override the per-world maximum contacts if specified in the settings
        if max_contacts_per_world is not None:
            for w in range(self.num_worlds):
                world_max_contacts[w] = min(world_max_contacts[w], max_contacts_per_world)

        # Return the per-world maximum contacts list
        return sum(world_max_contacts), world_max_contacts

    ###
    # Internals
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
        geoms_idx_offset: int = 0
        body_dofs_idx_offset: int = 0
        joint_coords_idx_offset: int = 0
        joint_dofs_idx_offset: int = 0
        joint_passive_coords_idx_offset: int = 0
        joint_passive_dofs_idx_offset: int = 0
        joint_actuated_coords_idx_offset: int = 0
        joint_actuated_dofs_idx_offset: int = 0
        joint_cts_idx_offset: int = 0
        joint_dynamic_cts_idx_offset: int = 0
        joint_kinematic_cts_idx_offset: int = 0
        # Iterate over each world and set their model offsets
        for world in self._worlds:
            # Set the offsets in the world descriptor to the current values
            world.bodies_idx_offset = int(bodies_idx_offset)
            world.joints_idx_offset = int(joints_idx_offset)
            world.geoms_idx_offset = int(geoms_idx_offset)
            world.body_dofs_idx_offset = int(body_dofs_idx_offset)
            world.joint_coords_idx_offset = int(joint_coords_idx_offset)
            world.joint_dofs_idx_offset = int(joint_dofs_idx_offset)
            world.joint_passive_coords_idx_offset = int(joint_passive_coords_idx_offset)
            world.joint_passive_dofs_idx_offset = int(joint_passive_dofs_idx_offset)
            world.joint_actuated_coords_idx_offset = int(joint_actuated_coords_idx_offset)
            world.joint_actuated_dofs_idx_offset = int(joint_actuated_dofs_idx_offset)
            world.joint_cts_idx_offset = int(joint_cts_idx_offset)
            world.joint_dynamic_cts_idx_offset = int(joint_dynamic_cts_idx_offset)
            world.joint_kinematic_cts_idx_offset = int(joint_kinematic_cts_idx_offset)
            # Update the offsets for the next world
            bodies_idx_offset += world.num_bodies
            joints_idx_offset += world.num_joints
            geoms_idx_offset += world.num_geoms
            body_dofs_idx_offset += 6 * world.num_bodies
            joint_coords_idx_offset += world.num_joint_coords
            joint_dofs_idx_offset += world.num_joint_dofs
            joint_passive_coords_idx_offset += world.num_passive_joint_coords
            joint_passive_dofs_idx_offset += world.num_passive_joint_dofs
            joint_actuated_coords_idx_offset += world.num_actuated_joint_coords
            joint_actuated_dofs_idx_offset += world.num_actuated_joint_dofs
            joint_cts_idx_offset += world.num_joint_cts
            joint_dynamic_cts_idx_offset += world.num_dynamic_joint_cts
            joint_kinematic_cts_idx_offset += world.num_kinematic_joint_cts

    def _collect_geom_max_contact_hints(self) -> tuple[int, list[int]]:
        """
        Collects the `max_contacts` hints from collision geometries.
        """
        model_max_contacts = 0
        world_max_contacts = [0] * self.num_worlds
        for w in range(len(self._worlds)):
            for geom_maxnc in self._worlds[w].geometry_max_contacts:
                model_max_contacts += geom_maxnc
                world_max_contacts[w] += geom_maxnc
        return model_max_contacts, world_max_contacts

    EntityDescriptorType = RigidBodyDescriptor | JointDescriptor | GeometryDescriptor
    """A type alias for model entity descriptors."""

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
