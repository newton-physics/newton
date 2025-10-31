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

"""KAMINO: Joint Model Types & Containers"""

from __future__ import annotations

from enum import IntEnum

import warp as wp

from .types import (
    float32,
    int32,
    mat33f,
    override,
    transformf,
    vec3f,
    vec6f,
)

###
# Module interface
###

__all__ = ["JointActuationType", "JointConnectionType", "JointDescriptor", "JointDoFType", "JointsData", "JointsModel"]


###
# Module configs
###

wp.set_module_options({"enable_backward": False})


###
# Enumerations
###


class JointActuationType(IntEnum):
    """
    An enumeration of the joint actuation types.
    """

    PASSIVE = 0
    """Passive joint type, i.e. not actuated."""

    FORCE = 1
    """Force-controlled joint type, i.e. actuated by set of forces and/or torques."""

    @override
    def __str__(self):
        """Returns a string representation of the joint actuation type."""
        return f"JointDoFType.{self.name} ({self.value})"

    @override
    def __repr__(self):
        """Returns a string representation of the joint actuation type."""
        return self.__str__()


class JointConnectionType(IntEnum):
    """
    An enumeration of the joint connection types.
    """

    UNARY = 0
    """Unary joint connection type, connecting a body to the world."""

    BINARY = 1
    """Binary joint connection type, connecting two bodies."""

    @override
    def __str__(self):
        """Returns a string representation of the joint connection type."""
        return f"JointDoFType.{self.name} ({self.value})"

    @override
    def __repr__(self):
        """Returns a string representation of the joint connection type."""
        return self.__str__()


class JointDoFType(IntEnum):
    """
    An enumeration of the joint degrees of freedom (DoF) types.
    """

    FREE = 0
    """6-DoF free-floating joint, with 6 rotational and translational DoFs, {R_x, R_y, R_z, T_x, T_y, T_z}."""

    REVOLUTE = 1
    """1-DoF revolute joint, with 1 rotational DoF, {R_x}."""

    PRISMATIC = 2
    """1-DoF prismatic joint, with 1 translational DoF, {T_x}."""

    CYLINDRICAL = 3
    """2-DoF cylindrical joint, with 1 rotational and 1 translational DoF, {R_x, T_x}."""

    UNIVERSAL = 4
    """2-DoF universal joint, with 2 rotational DoFs, {R_x, R_y}."""

    SPHERICAL = 5
    """3-DoF spherical joint, with 3 rotational DoFs, {R_x, R_y, R_z}."""

    GIMBAL = 6
    """3-DoF gimbal joint, with 3 rotational DoFs, {R_x, R_y, R_z}."""

    CARTESIAN = 7
    """3-DoF Cartesian joint, with 3 translational DoFs, {T_x, T_y, T_z}."""

    FIXED = 8
    """0-DoF fixed joint, with no relative motion between the bodies."""

    @override
    def __str__(self):
        """Returns a string representation of the joint DoF type."""
        return f"JointDoFType.{self.name} ({self.value})"

    @override
    def __repr__(self):
        """Returns a string representation of the joint DoF type."""
        return self.__str__()

    @property
    def num_coords(self) -> int:
        """
        The number of generalized coordinates defined by the joint DoF type.
        """
        if self.value == self.FREE:
            return 7
        elif self.value == self.REVOLUTE:
            return 1
        elif self.value == self.PRISMATIC:
            return 1
        elif self.value == self.CYLINDRICAL:
            return 2
        elif self.value == self.UNIVERSAL:
            return 2
        elif self.value == self.SPHERICAL:
            return 3  # TODO: 4
        elif self.value == self.GIMBAL:
            return 3
        elif self.value == self.CARTESIAN:
            return 3
        elif self.value == self.FIXED:
            return 0
        else:
            raise ValueError(f"Unknown joint DoF type: {self.value}")

    @property
    def num_dofs(self) -> int:
        """
        The number of DoFs defined by the joint DoF type.
        """
        if self.value == self.FREE:
            return 6
        elif self.value == self.REVOLUTE:
            return 1
        elif self.value == self.PRISMATIC:
            return 1
        elif self.value == self.CYLINDRICAL:
            return 2
        elif self.value == self.UNIVERSAL:
            return 2
        elif self.value == self.SPHERICAL:
            return 3
        elif self.value == self.GIMBAL:
            return 3
        elif self.value == self.CARTESIAN:
            return 3
        elif self.value == self.FIXED:
            return 0
        else:
            raise ValueError(f"Unknown joint DoF type: {self.value}")

    @property
    def num_cts(self) -> int:
        """
        The number of constraints defined by the joint DoF type.
        """
        if self.value == self.FREE:
            return 0
        elif self.value == self.REVOLUTE:
            return 5
        elif self.value == self.PRISMATIC:
            return 5
        elif self.value == self.CYLINDRICAL:
            return 4
        elif self.value == self.UNIVERSAL:
            return 4
        elif self.value == self.SPHERICAL:
            return 3
        elif self.value == self.GIMBAL:
            return 3
        elif self.value == self.CARTESIAN:
            return 3
        elif self.value == self.FIXED:
            return 6
        else:
            raise ValueError(f"Unknown joint DoF type: {self.value}")


###
# Containers
###


class JointDescriptor:
    """
    A container to describe a single joint in the model builder.
    """

    def __init__(self):
        self.name: str | None = None
        """Name of the joint."""

        self.uid: str | None = None
        """Unique identifier code (UID) of the joint."""

        self.wid: int = 0
        """Index of the world in which the joint is defined."""

        self.jid: int = -1
        """Index of the joint w.r.t. the world."""

        self.act_type: JointActuationType = JointActuationType.PASSIVE
        """Actuation type of the joint."""

        self.dof_type: JointDoFType = JointDoFType.FREE
        """DoF type of the joint."""

        self.num_coords: int = -1
        """Number of configuration coordinates of the joint."""

        self.num_dofs: int = -1
        """Number of DoFs of the joint."""

        self.num_cts: int = -1
        """Number of constraints of the joint."""

        self.coords_offset: int = -1
        """Index offset of this joint's coordinates among all joint coordinates in the world it belongs to."""

        self.dofs_offset: int = -1
        """Index offset of this joint's DoFs among all joint DoFs in the world it belongs to."""

        self.passive_coords_offset: int = -1
        """Index offset of this joint's passive coordinates among all joint coordinates in the world it belongs to."""

        self.passive_dofs_offset: int = -1
        """Index offset of this joint's passive DoFs among all joint DoFs in the world it belongs to."""

        self.actuated_coords_offset: int = -1
        """Index offset of this joint's actuated coordinates among all joint coordinates in the world it belongs to."""

        self.actuated_dofs_offset: int = -1
        """Index offset of this joint's actuated DoFs among all joint DoFs in the world it belongs to."""

        self.cts_offset: int = -1
        """Index offset of this joint's constraints among all joint constraints in the world it belongs to."""

        self.bid_B: int = -1
        """The Base body index of the joint (-1 for world, >=0 for bodies)."""

        self.bid_F: int = -1
        """The Follower body index of the joint (-1 for world, >=0 for bodies)."""

        # TODO: Change this to a transformf
        self.B_r_Bj: vec3f = vec3f()
        """The relative position of the joint in the base body coordinates."""

        # TODO: Change this to a transformf
        self.F_r_Fj: vec3f = vec3f()
        """The relative position of the joint in the follower body coordinates."""

        # TODO: Remove this when body offsets become transforms
        self.X_j: mat33f = mat33f()
        """The constant axes matrix of the joint."""

        self.q_j_min: list[float] | float | None = None
        """Minimum configuration limits of the joint."""

        self.q_j_max: list[float] | float | None = None
        """Maximum configuration limits of the joint."""

        self.dq_j_max: list[float] | float | None = None
        """Maximum velocity limits of the joint."""

        self.tau_j_max: list[float] | float | None = None
        """Maximum effort limits of the joint."""

    def __repr__(self):
        return (
            f"JointDescriptor(\n"
            f"name: {self.name},\n"
            f"uid: {self.uid},\n"
            f"wid: {self.wid},\n"
            f"jid: {self.jid},\n"
            f"act_type: {self.act_type},\n"
            f"dof_type: {self.dof_type},\n"
            f"num_coords: {self.num_coords},\n"
            f"num_dofs: {self.num_dofs},\n"
            f"num_cts: {self.num_cts},\n"
            f"coords_offset: {self.coords_offset},\n"
            f"dofs_offset: {self.dofs_offset},\n"
            f"cts_offset: {self.cts_offset},\n"
            f"passive_coords_offset: {self.passive_coords_offset},\n"
            f"passive_dofs_offset: {self.passive_dofs_offset},\n"
            f"actuated_coords_offset: {self.actuated_coords_offset},\n"
            f"actuated_dofs_offset: {self.actuated_dofs_offset},\n"
            f"bid_B: {self.bid_B},\n"
            f"bid_F: {self.bid_F},\n"
            f"B_r_Bj: {self.B_r_Bj},\n"
            f"F_r_Fj: {self.F_r_Fj},\n"
            f"X_j:\n{self.X_j},\n"
            f"q_j_min: {self.q_j_min},\n"
            f"q_j_max: {self.q_j_max},\n"
            f"dq_j_max: {self.dq_j_max},\n"
            f"tau_j_max: {self.tau_j_max}\n"
            f")"
        )


class JointsModel:
    """
    An SoA-based container to hold time-invariant model data of a joint system.
    """

    def __init__(self):
        self.num_joints: int = 0
        """Total number of joints in the model (host-side)."""

        self.wid: wp.array | None = None
        """
        Index each the world in which each joint is defined.\n
        Shape of ``(num_joints,)`` and type :class:`int`.
        """

        self.jid: wp.array | None = None
        """
        Index of each joint w.r.t the world.\n
        Shape of ``(num_joints,)`` and type :class:`int`.
        """

        self.dof_type: wp.array | None = None
        """
        Joint DoF type ID of each joint.\n
        Shape of ``(num_joints,)`` and type :class:`int`.
        """

        self.act_type: wp.array | None = None
        """
        Joint actuation type ID of each joint.\n
        Shape of ``(num_joints,)`` and type :class:`int`.
        """

        self.num_coords: wp.array | None = None
        """
        Number of configuration coordinates of each joint.\n
        Shape of ``(num_joints,)`` and type :class:`int`.
        """

        self.num_dofs: wp.array | None = None
        """
        Number of DoFs of each joint.\n
        Shape of ``(num_joints,)`` and type :class:`int`.
        """

        self.num_cts: wp.array | None = None
        """
        Number of constraints of each joint.\n
        Shape of ``(num_joints,)`` and type :class:`int`.
        """

        self.coords_offset: wp.array | None = None
        """
        Index offset of each joint's coordinates w.r.t the start
        index of joint coordinates of the corresponding world.\n
        Shape of ``(num_joints,)`` and type :class:`int`.
        """

        self.dofs_offset: wp.array | None = None
        """
        Index offset of each joint's DoFs w.r.t the start
        index of joint DoFs of the corresponding world.\n
        Shape of ``(num_joints,)`` and type :class:`int`.
        """

        self.passive_coords_offset: wp.array | None = None
        """
        Index offset of each joint's passive coordinates w.r.t the start
        index of passive joint coordinates of the corresponding world.\n
        Shape of ``(num_joints,)`` and type :class:`int`.
        """

        self.passive_dofs_offset: wp.array | None = None
        """
        Index offset of each joint's passive DoFs w.r.t the start
        index of passive joint DoFs of the corresponding world.\n
        Shape of ``(num_joints,)`` and type :class:`int`.
        """

        self.actuated_coords_offset: wp.array | None = None
        """
        Index offset of each joint's actuated coordinates w.r.t the start
        index of actuated joint coordinates of the corresponding world.\n
        Shape of ``(num_joints,)`` and type :class:`int`.
        """

        self.actuated_dofs_offset: wp.array | None = None
        """
        Index offset of each joint's actuated DoFs w.r.t the start
        index of actuated joint DoFs of the corresponding world.\n
        Shape of ``(num_joints,)`` and type :class:`int`.
        """

        self.cts_offset: wp.array | None = None
        """
        Index offset of each joint's constraints w.r.t the start
        index of joint constraints of the corresponding world.\n
        Shape of ``(num_joints,)`` and type :class:`int`.
        """

        self.bid_B: wp.array | None = None
        """
        Base body index of each joint (-1 for world, >=0 for bodies) w.r.t the model.\n
        Shape of ``(num_joints,)`` and type :class:`int`.
        """

        self.bid_F: wp.array | None = None
        """
        Follower body index of each joint (-1 for world, >=0 for bodies) w.r.t the model.\n
        Shape of ``(num_joints,)`` and type :class:`int`.
        """

        self.B_r_Bj: wp.array | None = None
        """
        Relative position of the joint, expressed in and w.r.t the base body coordinate frame.\n
        Shape of ``(num_joints, 3)`` and type :class:`vec3`.
        """

        self.F_r_Fj: wp.array | None = None
        """
        Relative position of the joint, expressed in and w.r.t the follower body coordinate frame.\n
        Shape of ``(num_joints, 3)`` and type :class:`vec3`.
        """

        self.X_j: wp.array | None = None
        """
        Joint axes matrix (local coordinates) of each joint, indicates the relative
        orientation of the the joint frame w.r.t the base body coordinate frame.\n
        Shape of ``(num_joints, 3, 3)`` and type :class:`mat33`.
        """

        self.q_j_min: wp.array | None = None
        """
        Minimum joint position limits of each joint (as flat array).\n
        Shape of ``(sum(c_j),)`` and type :class:`float`,\n
        where ``c_j`` is the number of coordinates of joint ``j``.
        """

        self.q_j_max: wp.array | None = None
        """
        Maximum joint position limits of each joint (as flat array).\n
        Shape of ``(sum(c_j),)`` and type :class:`float`,\n
        where ``c_j`` is the number of coordinates of joint ``j``.
        """

        self.dq_j_max: wp.array | None = None
        """
        Maximum joint velocity limits of each joint (as flat array).\n
        Shape of ``(sum(d_j),)`` and type :class:`float`,\n
        where ``d_j`` is the number of DoFs of joint ``j``.
        """

        self.tau_j_max: wp.array | None = None
        """
        Maximum joint torque limits of each joint (as flat array).\n
        Shape of ``(sum(d_j),)`` and type :class:`float`,\n
        where ``d_j`` is the number of DoFs of joint ``j``.
        """


class JointsData:
    """
    An SoA-based container to hold time-varying data of a joint system.
    """

    def __init__(self):
        self.num_joints: int = 0
        """Total number of joints in the model (host-side)."""

        self.p_j: wp.array | None = None
        """
        Array of joint frame pose transforms in world coordinates.\n
        Shape of ``(num_joints,)`` and type :class:`transform`.
        """

        self.r_j: wp.array | None = None
        """
        Flat array of joint constraint residuals.\n
        Shape of ``(sum(m_j),)`` and type :class:`float`,\n
        where ``m_j`` is the number of constraints of joint ``j``.
        """

        self.dr_j: wp.array | None = None
        """
        Flat array of joint constraint residual time-derivatives.\n
        Shape of ``(sum(m_j),)`` and type :class:`float`,\n
        where ``m_j`` is the number of constraints of joint ``j``.
        """

        self.lambda_j: wp.array | None = None
        """
        Flat array of joint constraint Lagrange multipliers.\n
        Shape of ``(sum(m_j),)`` and type :class:`float`,\n
        where ``m_j`` is the number of constraints of joint ``j``.
        """

        self.q_j: wp.array | None = None
        """
        Flat array of generalized coordinates of the joints.\n
        Shape of ``(sum(c_j),)`` and type :class:`float`,\n
        where ``c_j`` is the number of coordinates of joint ``j``.
        """

        self.dq_j: wp.array | None = None
        """
        Flat array of generalized velocities of the joints.\n
        Shape of ``(sum(d_j),)`` and type :class:`float`,\n
        where ``d_j`` is the number of DoFs of joint ``j``.
        """

        self.tau_j: wp.array | None = None
        """
        Flat array of generalized forces of the joints.\n
        Shape of ``(sum(d_j),)`` and type :class:`float`,\n
        where ``d_j`` is the number of DoFs of joint ``j``.
        """

        self.j_w_j: wp.array | None = None
        """
        Total wrench applied by each joint, expressed
        in and about the corresponding joint frame.\n
        Its direction follows the convention that
        joints act on the follower by the base body.\n
        Shape of ``(num_joints,)`` and type :class:`vec6`.
        """

        self.j_w_a_j: wp.array | None = None
        """
        Actuation wrench applied by each joint, expressed
        in and about the corresponding joint frame.\n
        Its direction is defined by the convention that positive wrenches
        in the joint frame are those inducing a positive change in the
        twist of the follower body relative to the base body.\n
        Shape of ``(num_joints,)`` and type :class:`vec6`.
        """

        self.j_w_c_j: wp.array | None = None
        """
        Constraint wrench applied by each joint, expressed
        in and about the corresponding joint frame.\n
        Its direction is defined by the convention that positive wrenches
        in the joint frame are those inducing a positive change in the
        twist of the follower body relative to the base body.\n
        Shape of ``(num_joints,)`` and type :class:`vec6`.
        """

        self.j_w_l_j: wp.array | None = None
        """
        Joint-limit wrench applied by each joint, expressed
        in and about the corresponding joint frame.\n
        Its direction is defined by the convention that positive wrenches
        in the joint frame are those inducing a positive change in the
        twist of the follower body relative to the base body.\n
        Shape of ``(num_joints,)`` and type :class:`vec6`.
        """

    def clear_residuals(self):
        """
        Reset all joint state variables to zero.
        """
        self.r_j.zero_()
        self.dr_j.zero_()

    def clear_state(self):
        """
        Reset all joint state variables to zero.
        """
        self.q_j.zero_()
        self.dq_j.zero_()

    def clear_constraint_reactions(self):
        """
        Reset all joint constraint reactions to zero.
        """
        self.lambda_j.zero_()

    def clear_actuation_forces(self):
        """
        Reset all joint actuation forces to zero.
        """
        self.tau_j.zero_()

    def clear_wrenches(self):
        """
        Reset all joint wrenches to zero.
        """
        self.j_w_j.zero_()
        self.j_w_c_j.zero_()
        self.j_w_a_j.zero_()
        self.j_w_l_j.zero_()


###
# Kernels
###


@wp.kernel
def _reset_joints_state(
    # Inputs:
    model_joint_num_coords: wp.array(dtype=int32),
    model_joint_num_dofs: wp.array(dtype=int32),
    model_joint_num_cts: wp.array(dtype=int32),
    model_joint_coords_offset: wp.array(dtype=int32),
    model_joint_dofs_offset: wp.array(dtype=int32),
    model_joint_cts_offset: wp.array(dtype=int32),
    model_joint_wid: wp.array(dtype=int32),
    model_joint_bid_B: wp.array(dtype=int32),
    model_joint_B_r_Bj: wp.array(dtype=vec3f),
    world_joint_coords_offsets: wp.array(dtype=int32),
    world_joint_dofs_offsets: wp.array(dtype=int32),
    world_joint_cts_offsets: wp.array(dtype=int32),
    state_body_q: wp.array(dtype=transformf),
    state_body_u: wp.array(dtype=vec6f),
    # Outputs:
    state_joint_p_j: wp.array(dtype=transformf),
    state_joint_r_j: wp.array(dtype=float32),
    state_joint_dr_j: wp.array(dtype=float32),
    state_joint_q_j: wp.array(dtype=float32),
    state_joint_dq_j: wp.array(dtype=float32),
    state_joint_tau_j: wp.array(dtype=float32),
):
    """
    Reset the current state to the initial state defined in the model.
    """
    # Retrieve the thread index
    jid = wp.tid()

    # Retrieve the joint model data
    wid = model_joint_wid[jid]
    c_j = model_joint_num_coords[jid]
    d_j = model_joint_num_dofs[jid]
    m_j = model_joint_num_cts[jid]
    cio_j = model_joint_coords_offset[jid]
    dio_j = model_joint_dofs_offset[jid]
    mio_j = model_joint_cts_offset[jid]
    bid_B_j = model_joint_bid_B[jid]
    B_r_Bj = model_joint_B_r_Bj[jid]

    # Retrieve the coordinate, DoF, and constraint index offsets of the world
    cio = world_joint_coords_offsets[wid]
    dio = world_joint_dofs_offsets[wid]
    mio = world_joint_cts_offsets[wid]

    # If the base body is the world (bid=-1), use the identity transform (frame of the world's origin)
    T_B_j = wp.transform_identity()
    if bid_B_j > -1:
        T_B_j = state_body_q[bid_B_j]

    # Extract the state of the base body
    r_B_j = wp.transform_get_translation(T_B_j)
    q_B_j = wp.transform_get_rotation(T_B_j)
    R_B_j = wp.quat_to_matrix(q_B_j)

    # Compute the initial pose of the joint frame
    r_j = r_B_j + R_B_j @ B_r_Bj
    T_j = wp.transformation(r_j, q_B_j, dtype=float32)
    state_joint_p_j[jid] = T_j

    # TODO: Compute initial joint velocities

    # Set the initial joint residuals and time-derivatives to zero
    mio_j += mio
    for j in range(m_j):
        state_joint_r_j[dio_j + j] = 0.0
        state_joint_dr_j[dio_j + j] = 0.0

    # Retrieve the initial generalized joint coordinates and velocities
    # NOTE: Currently we are always resetting these to zero
    # TODO: Copy entries from the model initial gencoords of joints
    q_j = vec6f(0.0)
    dq_j = vec6f(0.0)
    tau_j = vec6f(0.0)

    # Store the initial generalized joint coordinates
    cio_j += cio
    for j in range(c_j):
        state_joint_q_j[cio_j + j] = q_j[j]

    # Store the initial generalized joint velocities and forces
    dio_j += dio
    for j in range(d_j):
        state_joint_dq_j[dio_j + j] = dq_j[j]
        state_joint_tau_j[dio_j + j] = tau_j[j]


###
# Launchers
###


def reset_joints_state(
    world_joints_coords_offsets: wp.array(dtype=int32),
    world_joints_dofs_offsets: wp.array(dtype=int32),
    world_joints_cts_offsets: wp.array(dtype=int32),
    bodies_state_q_i: wp.array(dtype=transformf),
    bodies_state_u_i: wp.array(dtype=vec6f),
    joints_model: JointsModel,
    joints_data: JointsData,
):
    """
    Reset the current state to the initial state defined in the model.
    """
    wp.launch(
        _reset_joints_state,
        dim=joints_model.num_joints,
        inputs=[
            # Inputs:
            joints_model.num_coords,
            joints_model.num_dofs,
            joints_model.num_cts,
            joints_model.coords_offset,
            joints_model.dofs_offset,
            joints_model.cts_offset,
            joints_model.wid,
            joints_model.bid_B,
            joints_model.B_r_Bj,
            world_joints_coords_offsets,
            world_joints_dofs_offsets,
            world_joints_cts_offsets,
            bodies_state_q_i,
            bodies_state_u_i,
            # Outputs:
            joints_data.p_j,
            joints_data.r_j,
            joints_data.dr_j,
            joints_data.q_j,
            joints_data.dq_j,
            joints_data.tau_j,
        ],
    )
