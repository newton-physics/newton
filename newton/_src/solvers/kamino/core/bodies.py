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

"""Defines Types & Containers for Rigid Body Entities."""

from __future__ import annotations

from dataclasses import dataclass, field

import warp as wp

from .types import Descriptor, mat33f, override, transformf, vec6f

###
# Module interface
###

__all__ = [
    "RigidBodiesData",
    "RigidBodiesModel",
    "RigidBodyDescriptor",
    "update_body_inertias",
    "update_body_wrenches",
]


###
# Module configs
###

wp.set_module_options({"enable_backward": False})


###
# Rigid-Body Containers
###


@dataclass
class RigidBodyDescriptor(Descriptor):
    """
    A container to describe a single rigid body in the model builder.

    Attributes:
        name (str): The name of the body.
        uid (str): The unique identifier of the body.
        m_i (float): Mass of the body (in kg).
        i_I_i (mat33f): Moment of inertia matrix (in local coordinates) of the body (in kg*m^2).
        q_i_0 (transformf): Initial absolute pose of the body (in world coordinates).
        u_i_0 (vec6f): Initial absolute twist of the body (in world coordinates).
        wid (int): Index of the world to which the body belongs.
        bid (int): Index of the body w.r.t. its world.
    """

    ###
    # Attributes
    ###

    m_i: float = 0.0
    """Mass of the body."""

    i_I_i: mat33f = field(default_factory=mat33f)
    """Moment of inertia matrix (in local coordinates) of the body."""

    q_i_0: transformf = field(default_factory=transformf)
    """Initial absolute pose of the body (in world coordinates)."""

    u_i_0: vec6f = field(default_factory=vec6f)
    """Initial absolute twist of the body (in world coordinates)."""

    ###
    # Metadata - to be set by the WorldDescriptor when added
    ###

    wid: int = -1
    """
    Index of the world to which the body belongs.\n
    Defaults to `-1`, indicating that the body has not yet been added to a world.
    """

    bid: int = -1
    """
    Index of the body w.r.t. its world.\n
    Defaults to `-1`, indicating that the body has not yet been added to a world.
    """

    @override
    def __repr__(self) -> str:
        """Returns a human-readable string representation of the RigidBodyDescriptor."""
        return (
            f"RigidBodyDescriptor(\n"
            f"name: {self.name},\n"
            f"uid: {self.uid},\n"
            f"m_i: {self.m_i},\n"
            f"i_I_i:\n{self.i_I_i},\n"
            f"q_i_0: {self.q_i_0},\n"
            f"u_i_0: {self.u_i_0}\n"
            f"wid: {self.wid},\n"
            f"bid: {self.bid},\n"
            f")"
        )


@dataclass
class RigidBodiesModel:
    """
    An SoA-based container to hold time-invariant model data of a set of rigid body elements.

    Attributes:
        num_bodies (int): The total number of body elements in the model (host-side).
        wid (wp.array | None): World index each body.\n
            Shape of ``(num_bodies,)`` and type :class:`int`.
        bid (wp.array | None): Body index of each body w.r.t it's world.\n
            Shape of ``(num_bodies,)`` and type :class:`int`.
        m_i (wp.array | None): Mass of each body.\n
            Shape of ``(num_bodies,)`` and type :class:`float`.
        inv_m_i (wp.array | None): Inverse mass (1/m_i) of each body.\n
            Shape of ``(num_bodies,)`` and type :class:`float`.
        i_I_i (wp.array | None): Local moment of inertia of each body.\n
            Shape of ``(num_bodies,)`` and type :class:`mat33`.
        inv_i_I_i (wp.array | None): Inverse of the local moment of inertia of each body.\n
            Shape of ``(num_bodies,)`` and type :class:`mat33`.
        q_i_0 (wp.array | None): Initial pose of each body.\n
            Shape of ``(num_bodies,)`` and type :class:`transform`.
        u_i_0 (wp.array | None): Initial twist of each body.\n
            Shape of ``(num_bodies,)`` and type :class:`vec6`.
    """

    num_bodies: int = 0
    """Total number of body elements in the model (host-side)."""

    wid: wp.array | None = None
    """
    World index each body.\n
    Shape of ``(num_bodies,)`` and type :class:`int`.
    """

    bid: wp.array | None = None
    """
    Body index of each body w.r.t it's world.\n
    Shape of ``(num_bodies,)`` and type :class:`int`.
    """

    m_i: wp.array | None = None
    """
    Mass of each body.\n
    Shape of ``(num_bodies,)`` and type :class:`float`.
    """

    inv_m_i: wp.array | None = None
    """
    Inverse mass (1/m_i) of each body.\n
    Shape of ``(num_bodies,)`` and type :class:`float`.
    """

    i_I_i: wp.array | None = None
    """
    Local moment of inertia of each body.\n
    Shape of ``(num_bodies,)`` and type :class:`mat33`.
    """

    inv_i_I_i: wp.array | None = None
    """
    Inverse of the local moment of inertia of each body.\n
    Shape of ``(num_bodies,)`` and type :class:`mat33`.
    """

    q_i_0: wp.array | None = None
    """
    Initial pose of each body.\n
    Shape of ``(num_bodies,)`` and type :class:`transform`.
    """

    u_i_0: wp.array | None = None
    """
    Initial twist of each body.\n
    Shape of ``(num_bodies,)`` and type :class:`vec6`.
    """


@dataclass
class RigidBodiesData:
    """
    An SoA-based container to hold time-varying data of a set of rigid body entities.
    """

    num_bodies: int = 0
    """Total number of body entities in the model (host-side)."""

    q_i: wp.array | None = None
    """
    Absolute poses of each body (in world coordinates).\n
    Shape of ``(num_bodies,)`` and type :class:`transform`.
    """

    u_i: wp.array | None = None
    """
    Absolute twists of each body (in world coordinates).\n
    Shape of ``(num_bodies,)`` and type :class:`vec6`.
    """

    I_i: wp.array | None = None
    """
    Moment of inertia (in world coordinates) of each body.\n
    Shape of ``(num_bodies,)`` and type :class:`mat33`.
    """

    inv_I_i: wp.array | None = None
    """
    Inverse moment of inertia (in world coordinates) of each body.\n
    Shape of ``(num_bodies,)`` and type :class:`mat33`.
    """

    w_i: wp.array | None = None
    """
    Total wrench applied to each body (in world coordinates).\n
    Shape of ``(num_bodies,)`` and type :class:`vec6`.
    """

    w_a_i: wp.array | None = None
    """
    Joint actuation wrench applied to each body (in world coordinates).\n
    Shape of ``(num_bodies,)`` and type :class:`vec6`.
    """

    w_j_i: wp.array | None = None
    """
    Joint constraint wrench applied to each body (in world coordinates).\n
    Shape of ``(num_bodies,)`` and type :class:`vec6`.
    """

    w_l_i: wp.array | None = None
    """
    Joint limit wrench applied to each body (in world coordinates).\n
    Shape of ``(num_bodies,)`` and type :class:`vec6`.
    """

    w_c_i: wp.array | None = None
    """
    Contact wrench applied to each body (in world coordinates).\n
    Shape of ``(num_bodies,)`` and type :class:`vec6`.
    """

    w_e_i: wp.array | None = None
    """
    External wrench applied to each body (in world coordinates).\n
    Shape of ``(num_bodies,)`` and type :class:`vec6`.
    """

    def clear_all_wrenches(self):
        """
        Clears all body wrenches, total and components, setting them to zeros.
        """
        self.w_i.zero_()
        self.w_a_i.zero_()
        self.w_j_i.zero_()
        self.w_l_i.zero_()
        self.w_c_i.zero_()
        self.w_e_i.zero_()

    def clear_constraint_wrenches(self):
        """
        Clears all constraint wrenches, setting them to zeros.
        """
        self.w_j_i.zero_()
        self.w_l_i.zero_()
        self.w_c_i.zero_()

    def clear_actuation_wrenches(self):
        """
        Clears actuation wrenches, setting them to zeros.
        """
        self.w_a_i.zero_()

    def clear_external_wrenches(self):
        """
        Clears external wrenches, setting them to zeros.
        """
        self.w_e_i.zero_()


###
# Functions
###


# TODO: Use Warp generics to be applicable to other numeric types
@wp.func
def make_symmetric(A: mat33f) -> mat33f:
    """
    Makes a given matrix symmetric by averaging it with its transpose.

    Args:
        A (mat33f): The input matrix.

    Returns:
        mat33f: The symmetric matrix.
    """
    return 0.5 * (A + wp.transpose(A))


@wp.func
def transform_body_inertial_properties(
    p_i: transformf,
    i_I_i: mat33f,
    inv_i_I_i: mat33f,
) -> tuple[mat33f, mat33f]:
    """
    Transforms the inertial properties of a rigid body, specified in
    local coordinates, to world coordinates given its pose. The inertial
    properties include the moment of inertia matrix and its inverse.

    Args:
        p_i (transformf): The absolute pose of the body in world coordinates.
        i_I_i (mat33f): The local moment of inertia of the body.
        inv_i_I_i (mat33f): The inverse of the local moment of inertia of the body.

    Returns:
        tuple[mat33f, mat33f]: The moment of inertia and its inverse in world coordinates.
    """
    # Compute the moment of inertia matrices in world coordinates
    R_i = wp.quat_to_matrix(wp.transform_get_rotation(p_i))
    I_i = R_i @ i_I_i @ wp.transpose(R_i)
    inv_I_i = R_i @ inv_i_I_i @ wp.transpose(R_i)

    # TODO: Ensure symmetry of the inertia matrices (to avoid numerical issues)
    # I_i = make_symmetric(I_i)
    # inv_I_i = make_symmetric(inv_I_i)

    # Return the computed moment of inertia matrices in world coordinates
    return I_i, inv_I_i


###
# Kernels
###


@wp.kernel
def _update_body_inertias(
    # Inputs:
    model_bodies_i_I_i_in: wp.array(dtype=mat33f),
    model_bodies_inv_i_I_i_in: wp.array(dtype=mat33f),
    state_bodies_q_i_in: wp.array(dtype=transformf),
    # Outputs:
    state_bodies_I_i_out: wp.array(dtype=mat33f),
    state_bodies_inv_I_i_out: wp.array(dtype=mat33f),
):
    # Retrieve the thread index as the body index
    bid = wp.tid()

    # Retrieve the model data
    p_i = state_bodies_q_i_in[bid]
    i_I_i = model_bodies_i_I_i_in[bid]
    inv_i_I_i = model_bodies_inv_i_I_i_in[bid]

    # Compute the moment of inertia matrices in world coordinates
    I_i, inv_I_i = transform_body_inertial_properties(p_i, i_I_i, inv_i_I_i)

    # Store results in the output arrays
    state_bodies_I_i_out[bid] = I_i
    state_bodies_inv_I_i_out[bid] = inv_I_i


@wp.kernel
def _update_body_wrenches(
    # Inputs
    state_bodies_w_a_i_in: wp.array(dtype=vec6f),
    state_bodies_w_j_i_in: wp.array(dtype=vec6f),
    state_bodies_w_l_i_in: wp.array(dtype=vec6f),
    state_bodies_w_c_i_in: wp.array(dtype=vec6f),
    state_bodies_w_e_i_in: wp.array(dtype=vec6f),
    # Outputs
    state_bodies_w_i_out: wp.array(dtype=vec6f),
):
    # Retrieve the thread index as the body index
    bid = wp.tid()

    # Retrieve the model data
    w_a_i = state_bodies_w_a_i_in[bid]
    w_j_i = state_bodies_w_j_i_in[bid]
    w_l_i = state_bodies_w_l_i_in[bid]
    w_c_i = state_bodies_w_c_i_in[bid]
    w_e_i = state_bodies_w_e_i_in[bid]

    # Compute the total wrench applied to the body
    w_i = w_a_i + w_j_i + w_l_i + w_c_i + w_e_i

    # Store results in the output arrays
    state_bodies_w_i_out[bid] = w_i


###
# Launchers
###


def update_body_inertias(model: RigidBodiesModel, data: RigidBodiesData):
    wp.launch(
        _update_body_inertias,
        dim=model.num_bodies,
        inputs=[
            # Inputs:
            model.i_I_i,
            model.inv_i_I_i,
            data.q_i,
            # Outputs:
            data.I_i,
            data.inv_I_i,
        ],
    )


def update_body_wrenches(model: RigidBodiesModel, data: RigidBodiesData):
    wp.launch(
        _update_body_wrenches,
        dim=model.num_bodies,
        inputs=[
            # Inputs:
            data.w_a_i,
            data.w_j_i,
            data.w_l_i,
            data.w_c_i,
            data.w_e_i,
            # Outputs:
            data.w_i,
        ],
    )
