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
KAMINO: Constrained Rigid Multi-Body Model Containers
"""

from __future__ import annotations

import warp as wp
from warp.context import Devicelike

from .bodies import RigidBodiesData, RigidBodiesModel
from .control import Control
from .geometry import (
    CollisionGeometriesData,
    CollisionGeometriesModel,
    GeometriesData,
    GeometriesModel,
)
from .gravity import GravityModel
from .joints import JointsData, JointsModel
from .materials import MaterialPairsModel
from .state import State
from .time import TimeData, TimeModel
from .types import float32, int32, mat33f, mat83f, transformf, vec6f
from .world import WorldDescriptor

###
# Module interface
###

__all__ = [
    "Model",
    "ModelData",
    "ModelDataInfo",
    "ModelInfo",
]


###
# Module configs
###

wp.set_module_options({"enable_backward": False})


###
# Containers
###


class ModelSize:
    """
    A container to hold the summary size of memory allocations and thread dimensions.

    Notes:
    - The sums are used for memory allocations.
    - The maximums are used to define 2D thread shapes: (num_worlds, max_of_max_XXX)
    - Where `XXX` is the maximum number of limits, contacts, unilaterals, or constraints in any world.
    """

    def __init__(self):
        self.num_worlds: int = 0
        """The number of worlds represented in the model."""

        self.sum_of_num_bodies: int = 0
        """The total number of bodies in the model across all worlds."""

        self.max_of_num_bodies: int = 0
        """The maximum number of bodies in any world."""

        self.sum_of_num_joints: int = 0
        """The total number of joints in the model across all worlds."""

        self.max_of_num_joints: int = 0
        """The maximum number of joints in any world."""

        self.sum_of_num_passive_joints: int = 0
        """The total number of passive joints in the model across all worlds."""

        self.max_of_num_passive_joints: int = 0
        """The maximum number of passive joints in any world."""

        self.sum_of_num_actuated_joints: int = 0
        """The total number of actuated joints in the model across all worlds."""

        self.max_of_num_actuated_joints: int = 0
        """The maximum number of actuated joints in any world."""

        self.sum_of_num_collision_geoms: int = 0
        """The total number of collision geometries in the model across all worlds."""

        self.max_of_num_collision_geoms: int = 0
        """The maximum number of collision geometries in any world."""

        self.sum_of_num_physical_geoms: int = 0
        """The total number of physical geometries in the model across all worlds."""

        self.max_of_num_physical_geoms: int = 0
        """The maximum number of physical geometries in any world."""

        self.sum_of_num_material_pairs: int = 0
        """The total number of material pairs in the model across all worlds."""

        self.max_of_num_material_pairs: int = 0
        """The maximum number of material pairs in any world."""

        self.sum_of_num_body_dofs: int = 0
        """The total number of body DoFs in the model across all worlds."""

        self.max_of_num_body_dofs: int = 0
        """The maximum number of body DoFs in any world."""

        self.sum_of_num_joint_coords: int = 0
        """The total number of joint coordinates in the model across all worlds."""

        self.max_of_num_joint_coords: int = 0
        """The maximum number of joint coordinates in any world."""

        self.sum_of_num_joint_dofs: int = 0
        """The total number of joint DoFs in the model across all worlds."""

        self.max_of_num_joint_dofs: int = 0
        """The maximum number of joint DoFs in any world."""

        self.sum_of_num_passive_joint_coords: int = 0
        """The total number of passive joint coordinates in the model across all worlds."""

        self.max_of_num_passive_joint_coords: int = 0
        """The maximum number of passive joint coordinates in any world."""

        self.sum_of_num_passive_joint_dofs: int = 0
        """The total number of passive joint DoFs in the model across all worlds."""

        self.max_of_num_passive_joint_dofs: int = 0
        """The maximum number of passive joint DoFs in any world."""

        self.sum_of_num_actuated_joint_coords: int = 0
        """The total number of actuated joint coordinates in the model across all worlds."""

        self.max_of_num_actuated_joint_coords: int = 0
        """The maximum number of actuated joint coordinates in any world."""

        self.sum_of_num_actuated_joint_dofs: int = 0
        """The total number of actuated joint DoFs in the model across all worlds."""

        self.max_of_num_actuated_joint_dofs: int = 0
        """The maximum number of actuated joint DoFs in any world."""

        self.sum_of_num_joint_cts: int = 0
        """The total number of joint constraints in the model across all worlds."""

        self.max_of_num_joint_cts: int = 0
        """The maximum number of joint constraints in any world."""

        self.sum_of_max_limits: int = 0
        """The total maximum number of limits allocated for the model across all worlds."""

        self.max_of_max_limits: int = 0
        """The maximum number of active limits of any world."""

        self.sum_of_max_contacts: int = 0
        """The total maximum number of contacts allocated for the model across all worlds."""

        self.max_of_max_contacts: int = 0
        """The maximum number of active contacts of any world."""

        self.sum_of_max_unilaterals: int = 0
        """The maximum number of active unilateral entities, i.e. joint-limits and contacts."""

        self.max_of_max_unilaterals: int = 0
        """The maximum number of active unilaterals of any world."""

        self.sum_of_max_total_cts: int = 0
        """The maximum number of active constraints."""

        self.max_of_max_total_cts: int = 0
        """The maximum number of active constraints of any world."""

    def __repr__(self):
        # List of (row title, sum attr, max attr)
        rows = [
            ("num_bodies", "sum_of_num_bodies", "max_of_num_bodies"),
            ("num_joints", "sum_of_num_joints", "max_of_num_joints"),
            ("num_passive_joints", "sum_of_num_passive_joints", "max_of_num_passive_joints"),
            ("num_actuated_joints", "sum_of_num_actuated_joints", "max_of_num_actuated_joints"),
            ("num_collision_geoms", "sum_of_num_collision_geoms", "max_of_num_collision_geoms"),
            ("num_physical_geoms", "sum_of_num_physical_geoms", "max_of_num_physical_geoms"),
            ("num_material_pairs", "sum_of_num_material_pairs", "max_of_num_material_pairs"),
            ("num_body_dofs", "sum_of_num_body_dofs", "max_of_num_body_dofs"),
            ("num_joint_coords", "sum_of_num_joint_coords", "max_of_num_joint_coords"),
            ("num_joint_dofs", "sum_of_num_joint_dofs", "max_of_num_joint_dofs"),
            ("num_passive_joint_coords", "sum_of_num_passive_joint_coords", "max_of_num_passive_joint_coords"),
            ("num_passive_joint_dofs", "sum_of_num_passive_joint_dofs", "max_of_num_passive_joint_dofs"),
            ("num_actuated_joint_coords", "sum_of_num_actuated_joint_coords", "max_of_num_actuated_joint_coords"),
            ("num_actuated_joint_dofs", "sum_of_num_actuated_joint_dofs", "max_of_num_actuated_joint_dofs"),
            ("num_joint_cts", "sum_of_num_joint_cts", "max_of_num_joint_cts"),
            ("max_limits", "sum_of_max_limits", "max_of_max_limits"),
            ("max_contacts", "sum_of_max_contacts", "max_of_max_contacts"),
            ("max_unilaterals", "sum_of_max_unilaterals", "max_of_max_unilaterals"),
            ("max_total_cts", "sum_of_max_total_cts", "max_of_max_total_cts"),
        ]

        # Compute column widths
        name_width = max(len("Name"), *(len(r[0]) for r in rows))
        sum_width = max(len("Sum"), *(len(str(getattr(self, r[1]))) for r in rows))
        max_width = max(len("Max"), *(len(str(getattr(self, r[2]))) for r in rows))

        # Write ModelSize members as a formatted table
        lines = []
        lines.append("-" * (name_width + 1 + sum_width + 1 + max_width))
        lines.append(f"{'Name':<{name_width}} {'Sum':>{sum_width}} {'Max':>{max_width}}")
        lines.append("-" * (name_width + 1 + sum_width + 1 + max_width))
        for name, sum_attr, max_attr in rows:
            sum_val = getattr(self, sum_attr)
            max_val = getattr(self, max_attr)
            line = f"{name:<{name_width}} {sum_val:>{sum_width}} {max_val:>{max_width}}"
            lines.append(line)
            lines.append("-" * (name_width + 1 + sum_width + 1 + max_width))

        # Join the lines into a single string
        return "\n".join(lines)


class ModelInfo:
    """
    A container to hold the time-invariant information and meta-data of a model.
    """

    def __init__(self):
        ###
        # Host-side Summary Counts
        ###

        self.num_worlds: int = 0
        """The number of worlds represented in the model."""

        ###
        # Entity Counts
        ###

        self.num_bodies: wp.array(dtype=int32) | None = None
        """
        The number of bodies in each world.\n
        Shape of ``(num_worlds,)`` and type :class:`int32`.
        """

        self.num_joints: wp.array(dtype=int32) | None = None
        """
        The number of joints in each world.\n
        Shape of ``(num_worlds,)`` and type :class:`int32`.
        """

        self.num_passive_joints: wp.array(dtype=int32) | None = None
        """
        The number of passive joints in each world.\n
        Shape of ``(num_worlds,)`` and type :class:`int32`.
        """

        self.num_actuated_joints: wp.array(dtype=int32) | None = None
        """
        The number of actuated joints in each world.\n
        Shape of ``(num_worlds,)`` and type :class:`int32`.
        """

        self.num_collision_geoms: wp.array(dtype=int32) | None = None
        """
        The number of collision geometries in each world.\n
        Shape of ``(num_worlds,)`` and type :class:`int32`.
        """

        self.num_physical_geoms: wp.array(dtype=int32) | None = None
        """
        The number of physical geometries in each world.\n
        Shape of ``(num_worlds,)`` and type :class:`int32`.
        """

        self.max_limits: wp.array(dtype=int32) | None = None
        """
        The maximum number of limits in each world.\n
        Shape of ``(num_worlds,)`` and type :class:`int32`.
        """

        self.max_contacts: wp.array(dtype=int32) | None = None
        """
        The maximum number of contacts in each world.\n
        Shape of ``(num_worlds,)`` and type :class:`int32`.
        """

        ###
        # DoF Counts
        ###

        self.num_body_dofs: wp.array(dtype=int32) | None = None
        """
        The number of body DoFs of each world.\n
        Shape of ``(num_worlds,)`` and type :class:`int32`.
        """

        self.num_joint_coords: wp.array(dtype=int32) | None = None
        """
        The number of joint coordinates of each world.\n
        Shape of ``(num_worlds,)`` and type :class:`int32`.
        """

        self.num_joint_dofs: wp.array(dtype=int32) | None = None
        """
        The number of joint DoFs of each world.\n
        Shape of ``(num_worlds,)`` and type :class:`int32`.
        """

        self.num_passive_joint_coords: wp.array(dtype=int32) | None = None
        """
        The number of passive joint coordinates of each world.\n
        Shape of ``(num_worlds,)`` and type :class:`int32`.
        """

        self.num_passive_joint_dofs: wp.array(dtype=int32) | None = None
        """
        The number of passive joint DoFs of each world.\n
        Shape of ``(num_worlds,)`` and type :class:`int32`.
        """

        self.num_actuated_joint_coords: wp.array(dtype=int32) | None = None
        """
        The number of actuated joint coordinates of each world.\n
        Shape of ``(num_worlds,)`` and type :class:`int32`.
        """

        self.num_actuated_joint_dofs: wp.array(dtype=int32) | None = None
        """
        The number of actuated joint DoFs of each world.\n
        Shape of ``(num_worlds,)`` and type :class:`int32`.
        """

        ###
        # Constraint Counts
        ###

        self.num_joint_cts: wp.array(dtype=int32) | None = None
        """
        The number of joint constraints of each world.\n
        Shape of ``(num_worlds,)`` and type :class:`int32`.
        """

        self.max_limit_cts: wp.array(dtype=int32) | None = None
        """
        The maximum number of active limit constraints of each world.\n
        Shape of ``(num_worlds,)`` and type :class:`int32`.
        """

        self.max_contact_cts: wp.array(dtype=int32) | None = None
        """
        The maximum number of active contact constraints of each world.\n
        Shape of ``(num_worlds,)`` and type :class:`int32`.
        """

        self.max_total_cts: wp.array(dtype=int32) | None = None
        """
        The maximum total number of active constraints of each world.\n
        Shape of ``(num_worlds,)`` and type :class:`int32`.
        """

        ###
        # Entity Offsets
        ###

        self.bodies_offset: wp.array(dtype=int32) | None = None
        """
        The body index offset of each world w.r.t the model.\n
        Shape of ``(num_worlds,)`` and type :class:`int32`.
        """

        self.joints_offset: wp.array(dtype=int32) | None = None
        """
        The joint index offset of each world w.r.t the model.\n
        Shape of ``(num_worlds,)`` and type :class:`int32`.
        """

        self.limits_offset: wp.array(dtype=int32) | None = None
        """
        The limit index offset of each world w.r.t the model.\n
        Shape of ``(num_worlds,)`` and type :class:`int32`.
        """

        self.contacts_offset: wp.array(dtype=int32) | None = None
        """
        The contact index offset of world w.r.t the model.\n
        Shape of ``(num_worlds,)`` and type :class:`int32`.
        """

        self.unilaterals_offset: wp.array(dtype=int32) | None = None
        """
        The index offset of the unilaterals (limits + contacts) block of each world.\n
        Shape of ``(num_worlds,)`` and type :class:`int32`.
        """

        ###
        # DoF Offsets
        ###

        self.body_dofs_offset: wp.array(dtype=int32) | None = None
        """
        The index offset of the body DoF block of each world.\n
        Shape of ``(num_worlds,)`` and type :class:`int32`.
        """

        self.joint_coords_offset: wp.array(dtype=int32) | None = None
        """
        The index offset of the joint coordinates block of each world.\n
        Shape of ``(num_worlds,)`` and type :class:`int32`.
        """

        self.joint_dofs_offset: wp.array(dtype=int32) | None = None
        """
        The index offset of the joint DoF block of each world.\n
        Shape of ``(num_worlds,)`` and type :class:`int32`.
        """

        self.joint_passive_coords_offset: wp.array(dtype=int32) | None = None
        """
        The index offset of the passive joint coordinates block of each world.\n
        Shape of ``(num_worlds,)`` and type :class:`int32`.
        """

        self.joint_passive_dofs_offset: wp.array(dtype=int32) | None = None
        """
        The index offset of the passive joint DoF block of each world.\n
        Shape of ``(num_worlds,)`` and type :class:`int32`.
        """

        self.joint_actuated_coords_offset: wp.array(dtype=int32) | None = None
        """
        The index offset of the actuated joint coordinates block of each world.\n
        Shape of ``(num_worlds,)`` and type :class:`int32`.
        """

        self.joint_actuated_dofs_offset: wp.array(dtype=int32) | None = None
        """
        The index offset of the actuated joint DoF block of each world.\n
        Shape of ``(num_worlds,)`` and type :class:`int32`.
        """

        ###
        # Constraint Offsets
        ###

        self.joint_cts_offset: wp.array(dtype=int32) | None = None
        """
        The index offset of the joint constraints block of each world.\n
        Shape of ``(num_worlds,)`` and type :class:`int32`.
        """

        self.limit_cts_offset: wp.array(dtype=int32) | None = None
        """
        The index offset of the limit constraints block of each world.\n
        Shape of ``(num_worlds,)`` and type :class:`int32`.
        """

        self.contact_cts_offset: wp.array(dtype=int32) | None = None
        """
        The index offset of the contact constraints block of each world.\n
        Shape of ``(num_worlds,)`` and type :class:`int32`.
        """

        self.unilateral_cts_offset: wp.array(dtype=int32) | None = None
        """
        The index offset of the unilateral constraints block of each world.\n
        Shape of ``(num_worlds,)`` and type :class:`int32`.
        """

        self.total_cts_offset: wp.array(dtype=int32) | None = None
        """
        The index offset of the total constraints block of each world.\n
        Shape of ``(num_worlds,)`` and type :class:`int32`.
        """

        ###
        # Inertial Properties
        ###

        self.mass_min: wp.array(dtype=float32) | None = None
        """
        Smallest mass amongst all bodies in each world.
        Shape of ``(num_worlds,)`` and type :class:`float32`.
        """

        self.mass_max: wp.array(dtype=float32) | None = None
        """
        Largest mass amongst all bodies in each world.
        Shape of ``(num_worlds,)`` and type :class:`float32`.
        """

        self.mass_total: wp.array(dtype=float32) | None = None
        """
        Total mass over all bodies in each world.
        Shape of ``(num_worlds,)`` and type :class:`float32`.
        """

        self.inertia_total: wp.array(dtype=float32) | None = None
        """
        Total inertia over all bodies in each world.
        Shape of ``(num_worlds,)`` and type :class:`float32`.
        """


class ModelDataInfo:
    """
    A container to hold the time-varying information and meta-data of a model-state.
    """

    def __init__(self):
        ###
        # Total Constraints
        ###

        self.num_total_cts: wp.array(dtype=int32) | None = None
        """
        The total number of active constraints.\n
        Shape of ``(num_worlds,)`` and type :class:`int32`.
        """

        ###
        # Limits
        ###

        self.num_limits: wp.array(dtype=int32) | None = None
        """
        The number of active limits in each world.\n
        Shape of ``(num_worlds,)`` and type :class:`int32`.
        """

        self.num_limit_cts: wp.array(dtype=int32) | None = None
        """
        The number of active limit constraints.\n
        Shape of ``(num_worlds,)`` and type :class:`int32`.
        """

        self.limit_cts_group_offset: wp.array(dtype=int32) | None = None
        """
        The index offset of the limit constraints group within the constraints block of each world.\n
        Shape of ``(num_worlds,)`` and type :class:`int32`.
        """

        ###
        # Contacts
        ###

        self.num_contacts: wp.array(dtype=int32) | None = None
        """
        The number of active contacts in each world.\n
        Shape of ``(num_worlds,)`` and type :class:`int32`.
        """

        self.num_contact_cts: wp.array(dtype=int32) | None = None
        """
        The number of active contact constraints.\n
        Shape of ``(num_worlds,)`` and type :class:`int32`.
        """

        self.contact_cts_group_offset: wp.array(dtype=int32) | None = None
        """
        The index offset of the contact constraints group within the constraints block of each world.\n
        Shape of ``(num_worlds,)`` and type :class:`int32`.
        """


class ModelData:
    """
    A container to hold the time-varying state of the model entities.
    """

    def __init__(self):
        self.info: ModelDataInfo | None = None
        """The info container holding the information and meta-data of the model data."""

        self.time: TimeData | None = None
        """Time state of the model, including the current simulation step and time."""

        self.bodies: RigidBodiesData | None = None
        """
        States of all rigid bodies in the model: poses, twists, wrenches,
        and moments of inertia computed in world coordinates.
        """

        self.joints: JointsData | None = None
        """
        States of joints in the model: joint frames computed in world coordinates,
        constraint residuals and reactions, and generalized (DoF) quantities.
        """

        self.cgeoms: CollisionGeometriesData | None = None
        """States of collision geometries in the model: poses, AABBs etc. computed in world coordinates."""

        self.pgeoms: GeometriesData | None = None
        """States of physical geometries in the model: poses computed in world coordinates."""


class Model:
    """
    A container to hold the time-invariant system model data.
    """

    def __init__(self):
        self.device: Devicelike = None
        """The device on which the model data is allocated."""

        self.requires_grad: bool = False
        """Whether the model requires gradients for its state. Defaults to `False`."""

        self.size: ModelSize = ModelSize()
        """
        Host-side cache of the model summary sizes.\n
        This is used for memory allocations and kernel thread dimensions.
        """

        self.worlds: list[WorldDescriptor] = []
        """
        Host-side cache of the world descriptors.\n
        This is used to construct the model and for memory allocations.
        """

        self.info: ModelInfo | None = None
        """The model info container holding the information and meta-data of the model."""

        self.time: TimeModel | None = None
        """The time model container holding time-step of each world."""

        self.gravity: GravityModel | None = None
        """The gravity model container holding the gravity configurations for each world."""

        self.bodies: RigidBodiesModel | None = None
        """The rigid bodies model container holdingall rigid body entities in the model."""

        self.joints: JointsModel | None = None
        """The joints model container holding all joint entities in the model."""

        self.cgeoms: CollisionGeometriesModel | None = None
        """The collision geometries model container holding all collision geometry entities in the model."""

        self.pgeoms: GeometriesModel | None = None
        """The physical geometries model container holding all physical geometry entities in the model."""

        self.mpairs: MaterialPairsModel | None = None
        """The material pairs model container holding all material pairs in the model."""

    def data(
        self,
        skip_body_dofs: bool = False,
        unilateral_cts: bool = False,
        requires_grad: bool = False,
        device: Devicelike = None,
    ) -> ModelData:
        """
        Create a model data container with the initial state of the model entities.

        Parameters
        ----------
        unilateral_cts : `bool`, optional
            Whether to include unilateral constraints (limits and contacts) in the model data. Defaults to `True`.
        requires_grad : `bool`
            Whether the model data should require gradients. Defaults to `False`.
        device : `Devicelike`, optional
            The device to create the model data on. If not specified, the model's device is used.
        """
        # If no device is specified, use the model's device
        if device is None:
            device = self.device

        # Construct the model data on the specified device
        with wp.ScopedDevice(device=device):
            # Retrieve entity counts
            nw = self.size.num_worlds
            nb = self.size.sum_of_num_bodies
            nj = self.size.sum_of_num_joints
            ncg = self.size.sum_of_num_collision_geoms
            npg = self.size.sum_of_num_physical_geoms

            # Retrieve the joint coordinate, DoF and constraint counts
            njq = self.size.sum_of_num_joint_coords
            njd = self.size.sum_of_num_joint_dofs
            njc = self.size.sum_of_num_joint_cts

            # Construct the model data container
            data = ModelData()

            # Construct the model data info
            # NOTE: By default, the model data info is initialized only with joint constraints.
            data.info = ModelDataInfo()
            data.info.num_total_cts = wp.zeros(shape=nw, dtype=int32)
            wp.copy(data.info.num_total_cts, self.info.num_joint_cts)

            # If unilateral constraints are enabled, initialize the additional state info
            if unilateral_cts:
                data.info.num_limits = wp.zeros(shape=nw, dtype=int32)
                data.info.num_contacts = wp.zeros(shape=nw, dtype=int32)
                data.info.num_limit_cts = wp.zeros(shape=nw, dtype=int32)
                data.info.num_contact_cts = wp.zeros(shape=nw, dtype=int32)
                data.info.limit_cts_group_offset = wp.zeros(shape=nw, dtype=int32)
                data.info.contact_cts_group_offset = wp.zeros(shape=nw, dtype=int32)

            # Construct the time state
            data.time = TimeData()
            data.time.steps = wp.zeros(shape=nw, dtype=int32, requires_grad=requires_grad)
            data.time.time = wp.zeros(shape=nw, dtype=float32, requires_grad=requires_grad)

            # Construct the rigid bodies state from the model's initial state
            data.bodies = RigidBodiesData()
            data.bodies.num_bodies = nb
            data.bodies.I_i = wp.zeros(shape=nb, dtype=mat33f, requires_grad=requires_grad)
            data.bodies.inv_I_i = wp.zeros(shape=nb, dtype=mat33f, requires_grad=requires_grad)
            if not skip_body_dofs:
                data.bodies.q_i = wp.clone(self.bodies.q_i_0, requires_grad=requires_grad)
                data.bodies.u_i = wp.clone(self.bodies.u_i_0, requires_grad=requires_grad)
            data.bodies.w_i = wp.zeros_like(self.bodies.u_i_0, requires_grad=requires_grad)
            data.bodies.w_a_i = wp.zeros_like(self.bodies.u_i_0, requires_grad=requires_grad)
            data.bodies.w_j_i = wp.zeros_like(self.bodies.u_i_0, requires_grad=requires_grad)
            data.bodies.w_l_i = wp.zeros_like(self.bodies.u_i_0, requires_grad=requires_grad)
            data.bodies.w_c_i = wp.zeros_like(self.bodies.u_i_0, requires_grad=requires_grad)
            data.bodies.w_e_i = wp.zeros_like(self.bodies.u_i_0, requires_grad=requires_grad)

            # Construct the joints state from the model's initial state
            data.joints = JointsData()
            data.joints.num_joints = nj
            data.joints.p_j = wp.zeros(shape=nj, dtype=transformf, requires_grad=requires_grad)
            data.joints.r_j = wp.zeros(shape=njc, dtype=float32, requires_grad=requires_grad)
            data.joints.dr_j = wp.zeros(shape=njc, dtype=float32, requires_grad=requires_grad)
            data.joints.lambda_j = wp.zeros(shape=njc, dtype=float32, requires_grad=requires_grad)
            data.joints.q_j = wp.zeros(shape=njq, dtype=float32, requires_grad=requires_grad)
            data.joints.dq_j = wp.zeros(shape=njd, dtype=float32, requires_grad=requires_grad)
            data.joints.tau_j = wp.zeros(shape=njd, dtype=float32, requires_grad=requires_grad)
            data.joints.j_w_j = wp.zeros(shape=nj, dtype=vec6f, requires_grad=requires_grad)
            data.joints.j_w_c_j = wp.zeros(shape=nj, dtype=vec6f, requires_grad=requires_grad)
            data.joints.j_w_a_j = wp.zeros(shape=nj, dtype=vec6f, requires_grad=requires_grad)
            data.joints.j_w_l_j = wp.zeros(shape=nj, dtype=vec6f, requires_grad=requires_grad)

            # Construct the collision geometries state from the model's initial state
            data.cgeoms = CollisionGeometriesData()
            data.cgeoms.num_geoms = ncg
            data.cgeoms.pose = wp.zeros(shape=ncg, dtype=transformf, requires_grad=requires_grad)
            data.cgeoms.aabb = wp.zeros(shape=ncg, dtype=mat83f, requires_grad=requires_grad)
            data.cgeoms.radius = wp.zeros(shape=ncg, dtype=float32, requires_grad=requires_grad)

            # Construct the physical geometries state from the model's initial state
            data.pgeoms = GeometriesData()
            data.pgeoms.num_geoms = npg
            data.pgeoms.pose = wp.zeros(shape=npg, dtype=transformf, requires_grad=requires_grad)

        # Return the constructed model data container
        return data

    def state(self, requires_grad: bool = False, device: Devicelike = None) -> State:
        """
        Creates a compact state container with the initial state of the model entities.

        Parameters
        ----------
        requires_grad : `bool`
            Whether the state should require gradients. Defaults to `False`.
        device : `Devicelike`, optional
            The device to create the state on. If not specified, the model's device is used.
        """
        # If no device is specified, use the model's device
        if device is None:
            device = self.device

        # Create a new state container with the initial state of the model entities on the specified device
        with wp.ScopedDevice(device=device):
            s = State(
                q_i=wp.clone(self.bodies.q_i_0, requires_grad=requires_grad),
                u_i=wp.clone(self.bodies.u_i_0, requires_grad=requires_grad),
                w_i=wp.zeros_like(self.bodies.u_i_0, requires_grad=requires_grad),
                q_j=wp.zeros(shape=self.size.sum_of_num_joint_coords, dtype=float32, requires_grad=requires_grad),
                dq_j=wp.zeros(shape=self.size.sum_of_num_joint_dofs, dtype=float32, requires_grad=requires_grad),
                lambda_j=wp.zeros(shape=self.size.sum_of_num_joint_cts, dtype=float32, requires_grad=requires_grad),
            )

        # Return the constructed state container
        return s

    def control(self, requires_grad: bool = False, device: Devicelike = None) -> Control:
        """
        Creates a compact control container with the initial state of the model entities.

        Parameters
        ----------
        requires_grad : `bool`
            Whether the control container should require gradients. Defaults to `False`.
        device : `Devicelike`, optional
            The device to create the control container on. If not specified, the model's device is used.
        """
        # If no device is specified, use the model's device
        if device is None:
            device = self.device

        # Create a new control container on the specified device
        with wp.ScopedDevice(device=device):
            c = Control(
                tau_j=wp.zeros(shape=self.size.sum_of_num_joint_dofs, dtype=float32, requires_grad=requires_grad)
            )

        # Return the constructed control container
        return c
