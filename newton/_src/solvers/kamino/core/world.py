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

"""Provides a host-side container to summarily describe simulation world."""

import math
from dataclasses import dataclass, field

import warp as wp

from .bodies import RigidBodyDescriptor
from .geometry import CollisionGeometryDescriptor, GeometryDescriptor
from .joints import JointActuationType, JointDescriptor, JointDoFType
from .materials import MaterialDescriptor
from .types import Descriptor

###
# Module interface
###

__all__ = [
    "WorldDescriptor",
]


###
# Module configs
###

wp.set_module_options({"enable_backward": False})


###
# Containers
###


@dataclass
class WorldDescriptor(Descriptor):
    """
    A container to describe the problem dimensions and elements of a single world.
    """

    wid: int = 0
    """
    Index of the world w.r.t. the entire model. Defaults to `0`.\n
    Used to identify the world in construction of multi-world models.
    """

    ###
    # Entity Counts
    ###

    num_bodies: int = 0
    """
    The number of rigid bodies defined in the world.
    """

    num_joints: int = 0
    """
    The number of joints defined in the world.
    """

    num_passive_joints: int = 0
    """
    The number of joints which are passive.\n
    This is less than or equal to `num_joints`.
    """

    num_actuated_joints: int = 0
    """
    The number of joints which are actuated.\n
    This is less than or equal to `num_joints`.
    """

    num_collision_geoms: int = 0
    """
    The number of collision geometries defined in the world.
    """

    num_physical_geoms: int = 0
    """
    The number of physical geometries defined in the world.
    """

    num_materials: int = 0
    """
    The number of materials defined in the world.
    """

    ###
    # Coordinates, DoFs & Constraints Counts
    ###

    num_body_coords: int = 0
    """
    The total number of body coordinates.\n
    This is always equal to `7 * num_bodies`.
    """

    num_body_dofs: int = 0
    """
    The total number of body DoFs.\n
    This is always equal to `6 * num_bodies`.
    """

    num_joint_coords: int = 0
    """
    The total number of joint coordinates.\n
    This is equal to the sum of the coordinates of all joints in the world.
    """

    num_joint_dofs: int = 0
    """
    The total number of joint DoFs.\n
    This is equal to the sum of the DoFs of all joints in the world.
    """

    num_passive_joint_coords: int = 0
    """
    The number of passive joint joint coordinates.\n
    This is equal to the sum of the coordinates of all passive joints defined
    in the world, and is always less than or equal to `num_joint_coords`.\n
    """

    num_passive_joint_dofs: int = 0
    """
    The number of passive joint joint DoFs.\n
    This is equal to the sum of the DoFs of all passive joints defined
    in the world, and is always less than or equal to `num_joint_dofs`.\n
    """

    num_actuated_joint_coords: int = 0
    """
    The number of actuated joint coordinates.\n
    This is equal to the sum of the coordinates of all actuated joints defined
    in the world, and is always less than or equal to `num_joint_coords`.\n
    """

    num_actuated_joint_dofs: int = 0
    """
    The number of actuated joint DoFs.\n
    This is equal to the sum of the DoFs of all actuated joints defined
    in the world, and is always less than or equal to `num_joint_dofs`.\n
    """

    num_joint_cts: int = 0
    """
    The total number of joint constraints.\n
    This is equal to the sum of the constraints of all joints defined in the world.
    """

    joint_coords: list[int] = field(default_factory=list[int])
    """
    The list of of all joint coordinates.\n
    This list is ordered according the joint indices in the world,
    and the sum of all elements is equal to `num_joint_coords`.\n
    """

    joint_dofs: list[int] = field(default_factory=list[int])
    """
    The list of of all joint DoFs.\n
    This list is ordered according the joint indices in the world,
    and the sum of all elements is equal to `num_joint_dofs`.\n
    """

    joint_passive_coords: list[int] = field(default_factory=list[int])
    """
    The list of of all passive joint coordinates.\n
    This list is ordered according the joint indices in the world,
    and the sum of all elements is equal to `num_passive_joint_coords`.\n
    """

    joint_passive_dofs: list[int] = field(default_factory=list[int])
    """
    The list of of all passive joint DoFs.\n
    This list is ordered according the joint indices in the world,
    and the sum of all elements is equal to `num_passive_joint_dofs`.\n
    """

    joint_actuated_coords: list[int] = field(default_factory=list[int])
    """
    The list of of all actuated joint coordinates.\n
    This list is ordered according the joint indices in the world,
    and the sum of all elements is equal to `num_actuated_joint_coords`.\n
    """

    joint_actuated_dofs: list[int] = field(default_factory=list[int])
    """
    The list of of all actuated joint DoFs.\n
    This list is ordered according the joint indices in the world,
    and the sum of all elements is equal to `num_actuated_joint_dofs`.\n
    """

    joint_cts: list[int] = field(default_factory=list[int])
    """
    The list of all joint constraints.\n
    This list is ordered according the joint indices in the world,
    and the sum of all elements is equal to `num_joint_cts`.\n
    """

    ###
    # Entity Offsets
    ###

    bodies_idx_offset: int = 0
    """Index offset of the world's bodies w.r.t the entire model."""

    joints_idx_offset: int = 0
    """Index offset of the world's joints w.r.t the entire model."""

    collision_geoms_idx_offset: int = 0
    """Index offset of the world's collision geometries w.r.t the entire model."""

    physical_geoms_idx_offset: int = 0
    """Index offset of the world's physical geometries w.r.t the entire model."""

    ###
    # Constraint & DoF Offsets
    ###

    body_dofs_idx_offset: int = 0
    """Index offset of the world's body DoFs w.r.t the entire model."""

    joint_coords_idx_offset: int = 0
    """Index offset of the world's joint coordinates w.r.t the entire model."""

    joint_dofs_idx_offset: int = 0
    """Index offset of the world's joint DoFs w.r.t the entire model."""

    passive_joint_coords_idx_offset: int = 0
    """Index offset of the world's passive joint coordinates w.r.t the entire model."""

    passive_joint_dofs_idx_offset: int = 0
    """Index offset of the world's passive joint DoFs w.r.t the entire model."""

    actuated_joint_coords_idx_offset: int = 0
    """Index offset of the world's actuated joint coordinates w.r.t the entire model."""

    actuated_joint_dofs_idx_offset: int = 0
    """Index offset of the world's actuated joint DoFs w.r.t the entire model."""

    joint_cts_idx_offset: int = 0
    """Index offset of the world's joint constraints w.r.t the entire model."""

    ###
    # Entity Identifiers
    ###

    body_names: list[str] = field(default_factory=list[str])
    """List of body names."""

    body_uids: list[str] = field(default_factory=list[str])
    """List of body unique identifiers (UIDs)."""

    joint_names: list[str] = field(default_factory=list[str])
    """List of joint names."""

    joint_uids: list[str] = field(default_factory=list[str])
    """List of joint unique identifiers (UIDs)."""

    collision_geom_names: list[str] = field(default_factory=list[str])
    """List of collision geometry names."""

    collision_geom_uids: list[str] = field(default_factory=list[str])
    """List of collision geometry unique identifiers (UIDs)."""

    physical_geom_names: list[str] = field(default_factory=list[str])
    """List of physical geometry names."""

    physical_geom_uids: list[str] = field(default_factory=list[str])
    """List of physical geometry unique identifiers (UIDs)."""

    material_names: list[str] = field(default_factory=list[str])
    """List of material names."""

    material_uids: list[str] = field(default_factory=list[str])
    """List of material unique identifiers (UIDs)."""

    unary_joint_names: list[str] = field(default_factory=list[str])
    """List of unary joint names."""

    fixed_joint_names: list[str] = field(default_factory=list[str])
    """List of fixed joint names."""

    passive_joint_names: list[str] = field(default_factory=list[str])
    """List of passive joint names."""

    actuated_joint_names: list[str] = field(default_factory=list[str])
    """List of actuated joint names."""

    physical_geometry_layers: list[str] = field(default_factory=list[str])
    """List of physical geometry layers."""

    collision_geometry_layers: list[str] = field(default_factory=list[str])
    """List of collision geometry layers."""

    collision_geometry_max_contacts: list[int] = field(default_factory=list[int])
    """List of maximum contacts prescribed for each collision geometry."""

    ###
    # Base Properties
    ###

    base_body_idx: int | None = None
    """
    Index of the `base body` w.r.t. the world, i.e., index of the central node of the
    body-joint connectivity graph.\n

    The `base body` is connected to the world through a `base joint`, which, if not specified
    is considered to be an implicit 6D free joint, indicating a floating-base system.
    Otherwise, the `base joint` must be a unary joint connecting the base body to the world.\n

    For articulated systems, the base body is the root body of the kinematic tree.\n

    For general mechanical assemblies, e.g. particle systems, rigid clusters or overconstrained
    multi-body systems, the base body serves only as a reference body for managing the system's
    pose in the world, and can thus be assigned arbitrarily to any body in the system.
    """

    base_joint_idx: int | None = None
    """
    Index of the base joint w.r.t. the world, i.e. the joint connecting the base body to the world.\n
    See `base_body_idx` for more details.
    """

    @property
    def has_base_body(self):
        """Whether the world has an assigned base body."""
        return self.base_body_idx is not None

    @property
    def base_body_name(self):
        """Name of the base body if set, otherwise empty string"""
        return self.body_names[self.base_body_idx] if self.base_body_idx is not None else ""

    @property
    def has_base_joint(self):
        """Whether the world has an assigned base joint."""
        return self.base_joint_idx is not None

    @property
    def base_joint_name(self):
        """Name of the base joint if set, otherwise empty string"""
        return self.joint_names[self.base_joint_idx] if self.base_joint_idx is not None else ""

    has_passive_dofs: bool = False
    """Whether the world has passive DoFs."""

    has_actuated_dofs: bool = False
    """Whether the world has actuated DoFs."""

    ###
    # Inertial Properties
    ###

    mass_min: float = math.inf
    """Smallest mass of any body in the world."""

    mass_max: float = 0.0
    """Largest mass of any body in the world."""

    mass_total: float = 0.0
    """Total mass of all bodies in the world."""

    inertia_total: float = 0.0
    """
    Total diagonal inertia over all bodies in the world.\n
    Equals the trace of the maximal-coordinate generalized mass matrix of the world.
    """

    ###
    # Operations
    ###

    def add_body(self, body: RigidBodyDescriptor):
        # Check if the body has already been added to a world
        if body.name in self.body_names:
            raise ValueError(f"Body name '{body.name}' already exists in world '{self.name}' ({self.wid}).")
        if body.uid in self.body_uids:
            raise ValueError(f"Body UID '{body.uid}' already exists in world '{self.name}' ({self.wid}).")

        # Assign body metadata based on the current contents of the world
        body.wid = self.wid
        body.bid = self.num_bodies

        # Append body info to world metadata
        self.body_names.append(body.name)
        self.body_uids.append(body.uid)

        # Update body entity counts
        self.num_bodies += 1
        self.num_body_coords += 7
        self.num_body_dofs += 6

        # Append body inertial properties to world totals
        self.mass_min = min(self.mass_min, body.m_i)
        self.mass_max = max(self.mass_max, body.m_i)
        self.mass_total += body.m_i
        self.inertia_total += 3.0 * body.m_i + float(body.i_I_i[0, 0] + body.i_I_i[1, 1] + body.i_I_i[2, 2])

    def add_joint(self, joint: JointDescriptor):
        # Check if the body has already been added to a world
        if joint.name in self.joint_names:
            raise ValueError(f"Joint name '{joint.name}' already exists in world '{self.name}' ({self.wid}).")
        if joint.uid in self.joint_uids:
            raise ValueError(f"Joint UID '{joint.uid}' already exists in world '{self.name}' ({self.wid}).")

        # Check if the specified Base-Follower body indices are valid
        if joint.bid_B < 0 and joint.bid_F < 0:
            raise ValueError(
                f"Invalid body indices: bid_B={joint.bid_B}, bid_F={joint.bid_F}:\n\
                - ==-1 indicates the world body, >=0 indicates finite rigid bodies\n\
                - Base BIDs must be in [-1, {self.num_bodies - 1}]\n\
                - Follower BIDs must be in [0, {self.num_bodies - 1}]"
            )
        if joint.bid_B >= self.num_bodies or joint.bid_F >= self.num_bodies:
            raise ValueError(
                f"Invalid body indices: bid_B={joint.bid_B}, bid_F={joint.bid_F}.\n\
                - ==-1 indicates the world body, >=0 indicates finite rigid bodies\n\
                - Base BIDs must be in [-1, {self.num_bodies - 1}]\n\
                - Follower BIDs must be in [0, {self.num_bodies - 1}]"
            )

        # Assign joint metadata based on the current contents of the world
        joint.wid = self.wid
        joint.jid = self.num_joints
        joint.coords_offset = self.num_joint_coords
        joint.dofs_offset = self.num_joint_dofs
        joint.cts_offset = self.num_joint_cts
        joint.passive_coords_offset = self.num_passive_joint_coords if joint.is_passive else -1
        joint.passive_dofs_offset = self.num_passive_joint_dofs if joint.is_passive else -1
        joint.actuated_coords_offset = self.num_actuated_joint_coords if joint.is_actuated else -1
        joint.actuated_dofs_offset = self.num_actuated_joint_dofs if joint.is_actuated else -1

        # Append joint info
        self.joint_coords.append(joint.num_coords)
        self.joint_dofs.append(joint.num_dofs)
        self.joint_cts.append(joint.num_cts)
        self.joint_names.append(joint.name)
        self.joint_uids.append(joint.uid)

        # Update joint entity counts
        self.num_joints += 1
        self.num_joint_coords += joint.num_coords
        self.num_joint_dofs += joint.num_dofs
        self.num_joint_cts += joint.num_cts

        # Append joint connection group info
        if joint.bid_B < 0:
            self.unary_joint_names.append(joint.name)

        # Append joint DoF group info
        if joint.dof_type == JointDoFType.FIXED:
            self.fixed_joint_names.append(joint.name)

        # Append joint control group info
        if joint.act_type == JointActuationType.PASSIVE:
            joint.passive_coords_offset = self.num_passive_joint_coords
            joint.passive_dofs_offset = self.num_passive_joint_dofs
            self.has_passive_dofs = True
            self.num_passive_joints += 1
            self.num_passive_joint_coords += joint.num_coords
            self.num_passive_joint_dofs += joint.num_dofs
            self.joint_passive_coords.append(joint.num_coords)
            self.joint_passive_dofs.append(joint.num_dofs)
            self.passive_joint_names.append(joint.name)
        else:
            joint.actuated_coords_offset = self.num_actuated_joint_coords
            joint.actuated_dofs_offset = self.num_actuated_joint_dofs
            self.has_actuated_dofs = True
            self.num_actuated_joints += 1
            self.num_actuated_joint_coords += joint.num_coords
            self.num_actuated_joint_dofs += joint.num_dofs
            self.joint_actuated_coords.append(joint.num_coords)
            self.joint_actuated_dofs.append(joint.num_dofs)
            self.actuated_joint_names.append(joint.name)

    def add_collision_layer(self, layer: str):
        if not isinstance(layer, str):
            raise TypeError(f"Collision layer must be a string, got `{type(layer)}` with value `{layer}` instead.")
        if layer not in self.collision_geometry_layers:
            self.collision_geometry_layers.append(layer)

    def add_physical_layer(self, layer: str):
        if not isinstance(layer, str):
            raise TypeError(f"Physical layer must be a string, got `{type(layer)}` with value `{layer}` instead.")
        if layer not in self.physical_geometry_layers:
            self.physical_geometry_layers.append(layer)

    def add_collision_geom(self, geom: CollisionGeometryDescriptor):
        # Check if the body has already been added to a world
        if geom.name in self.collision_geom_names:
            raise ValueError(f"Collision geom name '{geom.name}' already exists in world '{self.name}' ({self.wid}).")
        if geom.uid in self.collision_geom_uids:
            raise ValueError(f"Collision geom UID '{geom.uid}' already exists in world '{self.name}' ({self.wid}).")

        # Check if the layer is valid
        if geom.layer not in self.collision_geometry_layers:
            self.collision_geometry_layers.append(geom.layer)

        # Assign geometry metadata based on the current contents of the world
        geom.wid = self.wid
        geom.gid = self.num_collision_geoms
        geom.lid = self.collision_geometry_layers.index(geom.layer)

        # Update collision geometry entity counts
        self.num_collision_geoms += 1

        # Append geometry info
        self.collision_geom_names.append(geom.name)
        self.collision_geom_uids.append(geom.uid)
        self.collision_geometry_max_contacts.append(geom.max_contacts)

    def add_physical_geom(self, geom: GeometryDescriptor):
        # Check if the body has already been added to a world
        if geom.name in self.physical_geom_names:
            raise ValueError(f"Physical geom name '{geom.name}' already exists in world '{self.name}' ({self.wid}).")
        if geom.uid in self.physical_geom_uids:
            raise ValueError(f"Physical geom UID '{geom.uid}' already exists in world '{self.name}' ({self.wid}).")

        # Check if the layer is valid
        if geom.layer not in self.physical_geometry_layers:
            self.physical_geometry_layers.append(geom.layer)

        # Assign geometry metadata based on the current contents of the world
        geom.wid = self.wid
        geom.gid = self.num_physical_geoms
        geom.lid = self.physical_geometry_layers.index(geom.layer)

        # Update physical geometry entity counts
        self.num_physical_geoms += 1

        # Append geometry info
        self.physical_geom_names.append(geom.name)
        self.physical_geom_uids.append(geom.uid)

    def add_material(self, material: MaterialDescriptor):
        # Check if the material has already been added to a world
        if material.name in self.material_names:
            raise ValueError(f"Material name '{material.name}' already exists in world '{self.name}' ({self.wid}).")
        if material.uid in self.material_uids:
            raise ValueError(f"Material UID '{material.uid}' already exists in world '{self.name}' ({self.wid}).")

        # Assign material metadata based on the current contents of the world
        material.wid = self.wid
        material.mid = self.num_materials

        # Update material entity counts
        self.num_materials += 1

        # # nm=1 -> nmp=1: (0, 0)
        # # nm=2 -> nmp=3: (0, 0), (0, 1), (1, 1)
        # # nm=3 -> nmp=6: (0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2)
        # # nm=N -> nmp=N*(N+1)/2
        # self.num_material_pairs = self.num_materials * (self.num_materials + 1) // 2

        # Append material info
        self.material_names.append(material.name)
        self.material_uids.append(material.uid)

    def set_material(self, material: MaterialDescriptor, index: int):
        # Ensure index is valid
        if index < 0 or index >= self.num_materials:
            raise ValueError(f"Material index '{index}' out of range. Must be between 0 and {self.num_materials - 1}.")

        # Set material info
        self.material_names[index] = material.name
        self.material_uids[index] = material.uid

    def set_base_body(self, body_idx: int):
        # Ensure no different base body was already set
        if self.has_base_body and self.base_body_idx != body_idx:
            raise ValueError(
                f"World '{self.name}' ({self.wid}) already has a base body "
                f"assigned as '{self.body_names[self.base_body_idx]}' ({self.base_body_idx})."
            )

        # Ensure index is valid
        if body_idx < 0 or body_idx >= self.num_bodies:
            raise ValueError(f"Base body index '{body_idx}' out of range. Must be between 0 and {self.num_bodies - 1}.")

        # Set base body index
        self.base_body_idx = body_idx

    def set_base_joint(self, joint_idx: int):
        # Ensure no different base joint was already set
        if self.has_base_joint and self.base_joint_idx != joint_idx:
            raise ValueError(
                f"World '{self.name}' ({self.wid}) already has a base joint "
                f"assigned as '{self.joint_names[self.base_joint_idx]}' ({self.base_joint_idx})."
            )

        # Ensure index is valid
        if joint_idx < 0 or joint_idx >= self.num_joints:
            raise ValueError(
                f"Base joint index '{joint_idx}' out of range. Must be between 0 and {self.num_joints - 1}."
            )

        # Ensure joint is unary
        if self.joint_names[joint_idx] not in self.unary_joint_names:
            raise ValueError(
                f"Base joint name '{self.joint_names[joint_idx]}' not found in the registry of unary joints."
            )

        # Set base joint index
        self.base_joint_idx = joint_idx
