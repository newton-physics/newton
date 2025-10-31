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

"""Provides a host-side container to summarily describe a Kamino simulation world."""

from __future__ import annotations

import math

import warp as wp

from .bodies import RigidBodyDescriptor
from .geometry import CollisionGeometryDescriptor, GeometryDescriptor
from .joints import JointActuationType, JointDescriptor, JointDoFType
from .materials import MaterialDescriptor

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


class WorldDescriptor:
    """
    A container to describe the problem dimensions and elements of a single world.
    """

    def __init__(self):
        ###
        # Entity Counts
        ###

        self.num_bodies: int = 0
        """
        The number of rigid bodies defined in the world.
        """

        self.num_joints: int = 0
        """
        The number of joints defined in the world.
        """

        self.num_passive_joints: int = 0
        """
        The number of joints which are passive.\n
        This is less than or equal to `num_joints`.
        """

        self.num_actuated_joints: int = 0
        """
        The number of joints which are actuated.\n
        This is less than or equal to `num_joints`.
        """

        self.num_collision_geoms: int = 0
        """
        The number of collision geometries defined in the world.
        """

        self.num_physical_geoms: int = 0
        """
        The number of physical geometries defined in the world.
        """

        self.num_materials: int = 0
        """
        The number of materials defined in the world.
        """

        self.num_material_pairs: int = 0
        """
        The number of material pairs defined in the world.\n
        These are used to define surface interaction properties between geometries.\n
        The total number of material pairs defined in the world may be less than or
        equal to `2 ** num_materials`, since it is possible the model import may not
        specify interaction properties for all material pairs.\n
        """

        ###
        # Coordinates, DoFs & Constraints Counts
        ###

        self.num_body_dofs: int = 0
        """
        The total number of body DoFs.\n
        This is always equal to `6 * num_bodies`.
        """

        self.num_joint_coords: int = 0
        """
        The total number of joint coordinates.\n
        This is equal to the sum of the coordinates of all joints in the world.
        """

        self.num_joint_dofs: int = 0
        """
        The total number of joint DoFs.\n
        This is equal to the sum of the DoFs of all joints in the world.
        """

        self.num_passive_joint_coords: int = 0
        """
        The number of passive joint joint coordinates.\n
        This is equal to the sum of the coordinates of all passive joints defined
        in the world, and is always less than or equal to `num_joint_coords`.\n
        """

        self.num_passive_joint_dofs: int = 0
        """
        The number of passive joint joint DoFs.\n
        This is equal to the sum of the DoFs of all passive joints defined
        in the world, and is always less than or equal to `num_joint_dofs`.\n
        """

        self.num_actuated_joint_coords: int = 0
        """
        The number of actuated joint coordinates.\n
        This is equal to the sum of the coordinates of all actuated joints defined
        in the world, and is always less than or equal to `num_joint_coords`.\n
        """

        self.num_actuated_joint_dofs: int = 0
        """
        The number of actuated joint DoFs.\n
        This is equal to the sum of the DoFs of all actuated joints defined
        in the world, and is always less than or equal to `num_joint_dofs`.\n
        """

        self.num_joint_cts: int = 0
        """
        The total number of joint constraints.\n
        This is equal to the sum of the constraints of all joints defined in the world.
        """

        self.joint_coords: list[int] = []
        """
        The list of of all joint coordinates.\n
        This list is ordered according the joint indices in the world,
        and the sum of all elements is equal to `num_joint_coords`.\n
        """

        self.joint_dofs: list[int] = []
        """
        The list of of all joint DoFs.\n
        This list is ordered according the joint indices in the world,
        and the sum of all elements is equal to `num_joint_dofs`.\n
        """

        self.joint_passive_coords: list[int] = []
        """
        The list of of all passive joint coordinates.\n
        This list is ordered according the joint indices in the world,
        and the sum of all elements is equal to `num_passive_joint_coords`.\n
        """

        self.joint_passive_dofs: list[int] = []
        """
        The list of of all passive joint DoFs.\n
        This list is ordered according the joint indices in the world,
        and the sum of all elements is equal to `num_passive_joint_dofs`.\n
        """

        self.joint_actuated_coords: list[int] = []
        """
        The list of of all actuated joint coordinates.\n
        This list is ordered according the joint indices in the world,
        and the sum of all elements is equal to `num_actuated_joint_coords`.\n
        """

        self.joint_actuated_dofs: list[int] = []
        """
        The list of of all actuated joint DoFs.\n
        This list is ordered according the joint indices in the world,
        and the sum of all elements is equal to `num_actuated_joint_dofs`.\n
        """

        self.joint_cts: list[int] = []
        """
        The list of all joint constraints.\n
        This list is ordered according the joint indices in the world,
        and the sum of all elements is equal to `num_joint_cts`.\n
        """

        ###
        # Entity Offsets
        ###

        self.bodies_idx_offset: int = 0
        """Index offset of the world's bodies w.r.t the entire model."""

        self.joints_idx_offset: int = 0
        """Index offset of the world's joints w.r.t the entire model."""

        self.collision_geoms_idx_offset: int = 0
        """Index offset of the world's collision geometries w.r.t the entire model."""

        self.physical_geoms_idx_offset: int = 0
        """Index offset of the world's physical geometries w.r.t the entire model."""

        ###
        # Constraint & DoF Offsets
        ###

        self.body_dofs_idx_offset: int = 0
        """Index offset of the world's body DoFs w.r.t the entire model."""

        self.joint_coords_idx_offset: int = 0
        """Index offset of the world's joint coordinates w.r.t the entire model."""

        self.joint_dofs_idx_offset: int = 0
        """Index offset of the world's joint DoFs w.r.t the entire model."""

        self.passive_joint_coords_idx_offset: int = 0
        """Index offset of the world's passive joint coordinates w.r.t the entire model."""

        self.passive_joint_dofs_idx_offset: int = 0
        """Index offset of the world's passive joint DoFs w.r.t the entire model."""

        self.actuated_joint_coords_idx_offset: int = 0
        """Index offset of the world's actuated joint coordinates w.r.t the entire model."""

        self.actuated_joint_dofs_idx_offset: int = 0
        """Index offset of the world's actuated joint DoFs w.r.t the entire model."""

        self.joint_cts_idx_offset: int = 0
        """Index offset of the world's joint constraints w.r.t the entire model."""

        ###
        # Entity Identifiers
        ###

        self.body_names: list[str] = []
        """List of body names."""

        self.body_uids: list[str] = []
        """List of body unique identifiers (UIDs)."""

        self.joint_names: list[str] = []
        """List of joint names."""

        self.joint_uids: list[str] = []
        """List of joint unique identifiers (UIDs)."""

        self.collision_geom_names: list[str] = []
        """List of collision geometry names."""

        self.collision_geom_uids: list[str] = []
        """List of collision geometry unique identifiers (UIDs)."""

        self.physical_geom_names: list[str] = []
        """List of physical geometry names."""

        self.physical_geom_uids: list[str] = []
        """List of physical geometry unique identifiers (UIDs)."""

        self.material_names: list[str] = []
        """List of material names."""

        self.material_uids: list[str] = []
        """List of material unique identifiers (UIDs)."""

        self.unary_joint_names: list[str] = []
        """List of unary joint names."""

        self.fixed_joint_names: list[str] = []
        """List of fixed joint names."""

        self.passive_joint_names: list[str] = []
        """List of passive joint names."""

        self.actuated_joint_names: list[str] = []
        """List of actuated joint names."""

        self.physical_geometry_layers: list[str] = []
        """List of physical geometry layers."""

        self.collision_geometry_layers: list[str] = []
        """List of collision geometry layers."""

        self.collision_geometry_max_contacts: list[int] = []
        """List of maximum contacts prescribed for each collision geometry."""

        ###
        # Mass Properties
        ###

        self.mass_min: float = math.inf
        """Smallest mass of any body in the world."""

        self.mass_max: float = 0.0
        """Largest mass of any body in the world."""

        self.mass_total: float = 0.0
        """Total mass of all bodies in the world."""

        self.inertia_total: float = 0.0
        """Total inertia of all bodies in the world."""

        ###
        # Mass Properties
        ###

        self.base_name: str = ""
        """Name of the base body."""

        self.base_idx: int = -1
        """Index of the base body w.r.t. the world."""

        self.grounding_name: str = ""
        """Name of the grounding joint."""

        self.grounding_idx: int = -1
        """Index of the grounding joint w.r.t. the world."""

        self.has_base: bool = False
        """Whether the world has an assigned base body."""

        self.has_grounding: bool = False
        """Whether the world has an assigned grounding joint."""

        self.has_passive_dofs: bool = False
        """Whether the world has passive DoFs."""

        self.has_actuated_dofs: bool = False
        """Whether the world has actuated DoFs."""

    def add_body(self, body: RigidBodyDescriptor):
        # Append body info
        self.num_bodies += 1
        self.num_body_dofs += 6
        self.body_names.append(body.name)
        self.body_uids.append(body.uid)

        # Append body properties
        self.mass_min = min(self.mass_min, body.m_i)
        self.mass_max = max(self.mass_max, body.m_i)
        self.mass_total += body.m_i
        self.inertia_total += 3.0 * body.m_i + float(body.i_I_i[0, 0] + body.i_I_i[1, 1] + body.i_I_i[2, 2])

    def add_joint(self, joint: JointDescriptor):
        # Append joint info
        self.num_joints += 1
        self.num_joint_coords += joint.num_coords
        self.num_joint_dofs += joint.num_dofs
        self.num_joint_cts += joint.num_cts
        self.joint_coords.append(joint.num_coords)
        self.joint_dofs.append(joint.num_dofs)
        self.joint_cts.append(joint.num_cts)
        self.joint_names.append(joint.name)
        self.joint_uids.append(joint.uid)

        # Append joint connection group info
        if joint.bid_B < 0:
            self.unary_joint_names.append(joint.name)

        # Append joint DoF group info
        if joint.dof_type == JointDoFType.FIXED:
            self.fixed_joint_names.append(joint.name)

        # Append joint control group info
        if joint.act_type == JointActuationType.PASSIVE:
            self.has_passive_dofs = True
            self.num_passive_joints += 1
            self.num_passive_joint_coords += joint.num_coords
            self.num_passive_joint_dofs += joint.num_dofs
            self.joint_passive_coords.append(joint.num_coords)
            self.joint_passive_dofs.append(joint.num_dofs)
            self.passive_joint_names.append(joint.name)
        else:
            self.has_actuated_dofs = True
            self.num_actuated_joints += 1
            self.num_actuated_joint_coords += joint.num_coords
            self.num_actuated_joint_dofs += joint.num_dofs
            self.joint_actuated_coords.append(joint.num_coords)
            self.joint_actuated_dofs.append(joint.num_dofs)
            self.actuated_joint_names.append(joint.name)

    def add_cgeom(self, geom: CollisionGeometryDescriptor):
        # Append geometry info
        self.num_collision_geoms += 1
        self.collision_geom_names.append(geom.name)
        self.collision_geom_uids.append(geom.uid)
        self.collision_geometry_max_contacts.append(geom.max_contacts)

    def add_pgeom(self, geom: GeometryDescriptor):
        # Append geometry info
        self.num_physical_geoms += 1
        self.physical_geom_names.append(geom.name)
        self.physical_geom_uids.append(geom.uid)

    def add_material(self, material: MaterialDescriptor):
        # Append material info
        self.num_materials += 1
        self.num_material_pairs = 2**self.num_materials
        self.material_names.append(material.name)
        self.material_uids.append(material.uid)

    def set_material(self, material: MaterialDescriptor, index: int):
        # Ensure index is valid
        if index < 0 or index >= self.num_materials:
            raise ValueError(
                f"WorldDescriptor: Material index '{index}' out of range. "
                f"Must be between 0 and {self.num_materials - 1}."
            )
        # Set material info
        self.material_names[index] = material.name
        self.material_uids[index] = material.uid

    def set_base(self, body_name: str, body_idx: int):
        # Ensure name exists
        if body_name not in self.body_names:
            raise ValueError(f"WorldDescriptor: Base body name '{body_name}' not found in body names.")
        # Ensure index is valid
        if body_idx < 0 or body_idx >= self.num_bodies:
            raise ValueError(
                f"WorldDescriptor: Base body index '{body_idx}' out of range. Must be between 0 and {self.num_bodies - 1}."
            )
        # Set base body info
        self.base_name = body_name
        self.base_idx = body_idx
        self.has_base = True

    def set_grounding(self, joint_name: str, joint_idx: int):
        # Ensure name exists
        if joint_name not in self.joint_names:
            raise ValueError(f"WorldDescriptor: Grounding joint name '{joint_name}' not found in joint names.")
        # Ensure index is valid
        if joint_idx < 0 or joint_idx >= self.num_joints:
            raise ValueError(
                f"WorldDescriptor: Grounding joint index '{joint_idx}' out of range. Must be between 0 and {self.num_joints - 1}."
            )
        # Ensure joint is unary
        if joint_name not in self.unary_joint_names:
            raise ValueError(f"WorldDescriptor: Joint '{joint_name}' is not a unary joint.")
        # Set grounding joint info
        self.grounding_name = joint_name
        self.grounding_idx = joint_idx
        self.has_grounding = True
