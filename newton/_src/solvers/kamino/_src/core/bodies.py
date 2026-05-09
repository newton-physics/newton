# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Defines Types & Containers for Rigid Body Entities."""

from __future__ import annotations

from dataclasses import dataclass, field

import warp as wp

from .types import Descriptor, int32, mat33f, override, transformf, vec3f, vec6f

###
# Module interface
###

__all__ = [
    "RigidBodiesData",
    "RigidBodiesModel",
    "RigidBodyDescriptor",
    "convert_base_origin_to_com",
    "convert_body_com_to_origin",
    "convert_body_origin_to_com",
    "convert_geom_offset_origin_to_com",
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
        i_r_com_i (vec3f): Translational offset of the body center of mass (in meters).
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

    i_r_com_i: vec3f = field(default_factory=vec3f)
    """Translational offset of the body center of mass w.r.t the reference frame expressed in local coordinates."""

    i_I_i: mat33f = field(default_factory=mat33f)
    """Moment of inertia matrix of the body expressed in local coordinates."""

    q_i_0: transformf = field(default_factory=transformf)
    """Initial absolute pose of the body expressed in world coordinates."""

    u_i_0: vec6f = field(default_factory=vec6f)
    """Initial absolute twist of the body expressed in world coordinates."""

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

    ###
    # Meta-Data
    ###

    num_bodies: int = 0
    """Total number of body elements in the model (host-side)."""

    label: list[str] | None = None
    """
    A list containing the label of each body.\n
    Length of ``num_bodies`` and type :class:`str`.
    """

    ###
    # Identifiers
    ###

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

    ###
    # Parameterization
    ###

    i_r_com_i: wp.array | None = None
    """
    Translational offset of the center of mass w.r.t the body's reference frame.\n
    Shape of ``(num_bodies,)`` and type :class:`vec3`.
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

    ###
    # Initial State
    ###

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

    # Ensure symmetry of the inertia matrices (to avoid numerical issues)
    I_i = make_symmetric(I_i)
    inv_I_i = make_symmetric(inv_I_i)

    # Return the computed moment of inertia matrices in world coordinates
    return I_i, inv_I_i


###
# Kernels
###


@wp.kernel
def _update_body_inertias(
    # Inputs:
    model_bodies_i_I_i_in: wp.array[mat33f],
    model_bodies_inv_i_I_i_in: wp.array[mat33f],
    state_bodies_q_i_in: wp.array[transformf],
    # Outputs:
    state_bodies_I_i_out: wp.array[mat33f],
    state_bodies_inv_I_i_out: wp.array[mat33f],
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
    state_bodies_w_a_i_in: wp.array[vec6f],
    state_bodies_w_j_i_in: wp.array[vec6f],
    state_bodies_w_l_i_in: wp.array[vec6f],
    state_bodies_w_c_i_in: wp.array[vec6f],
    state_bodies_w_e_i_in: wp.array[vec6f],
    # Outputs
    state_bodies_w_i_out: wp.array[vec6f],
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


@wp.kernel
def _convert_body_origin_to_com(
    # Inputs
    world_mask: wp.array[int32],
    body_wid: wp.array[int32],
    body_com: wp.array[vec3f],
    body_q: wp.array[transformf],
    # Outputs
    body_q_com: wp.array[transformf],
):
    bid = wp.tid()

    if world_mask:
        assert body_wid
        if not world_mask[body_wid[bid]]:
            return

    com = body_com[bid]
    q = body_q[bid]

    body_r = wp.transform_get_translation(q)
    body_rot = wp.transform_get_rotation(q)
    r_com = wp.quat_rotate(body_rot, com)

    body_q_com[bid] = transformf(body_r + r_com, body_rot)


@wp.kernel
def _convert_body_com_to_origin(
    # Inputs
    world_mask: wp.array[int32],
    body_wid: wp.array[int32],
    body_com: wp.array[vec3f],
    body_q_com: wp.array[transformf],
    # Outputs
    body_q: wp.array[transformf],
):
    bid = wp.tid()

    if world_mask:
        assert body_wid
        if not world_mask[body_wid[bid]]:
            return

    com = body_com[bid]
    q = body_q_com[bid]

    body_r_com = wp.transform_get_translation(q)
    body_rot = wp.transform_get_rotation(q)
    r_com = wp.quat_rotate(body_rot, com)

    body_q[bid] = transformf(body_r_com - r_com, body_rot)


@wp.kernel
def _convert_base_origin_to_com(
    # Inputs
    base_body_index: wp.array[int32],
    body_com: wp.array[vec3f],
    base_q: wp.array[transformf],
    # Outputs
    base_q_com: wp.array[transformf],
):
    wid = wp.tid()
    base_bid = base_body_index[wid]
    if base_bid < 0:
        return
    com = body_com[base_bid]
    q = base_q[wid]
    rot = wp.transform_get_rotation(q)
    r_com = wp.quat_rotate(rot, com)
    base_q_com[wid] = transformf(wp.transform_get_translation(q) + r_com, rot)


@wp.kernel
def _convert_geom_offset_origin_to_com(
    # Inputs
    body_com: wp.array[vec3f],
    geom_bid: wp.array[int32],
    geom_offset: wp.array[transformf],
    # Outputs
    geom_offset_com: wp.array[transformf],
):
    gid = wp.tid()
    bid = geom_bid[gid]
    if bid >= 0:
        com = body_com[bid]
        X = geom_offset[gid]
        pos = wp.transform_get_translation(X)
        rot = wp.transform_get_rotation(X)
        geom_offset_com[gid] = transformf(pos - com, rot)
    else:
        geom_offset_com[gid] = geom_offset[gid]


###
# Launchers
###


def convert_geom_offset_origin_to_com(
    body_com: wp.array,
    geom_bid: wp.array,
    geom_offset: wp.array,
    geom_offset_com: wp.array,
):
    wp.launch(
        _convert_geom_offset_origin_to_com,
        dim=geom_bid.shape[0],
        inputs=[body_com, geom_bid, geom_offset],
        outputs=[geom_offset_com],
        device=body_com.device,
    )


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
        device=model.i_I_i.device,
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
        device=data.w_i.device,
    )


def convert_body_origin_to_com(
    body_com: wp.array,
    body_q: wp.array,
    body_q_com: wp.array,
    body_wid: wp.array | None = None,
    world_mask: wp.array | None = None,
):
    wp.launch(
        _convert_body_origin_to_com,
        dim=body_com.shape[0],
        inputs=[world_mask, body_wid, body_com, body_q],
        outputs=[body_q_com],
        device=body_com.device,
    )


def convert_body_com_to_origin(
    body_com: wp.array,
    body_q_com: wp.array,
    body_q: wp.array,
    body_wid: wp.array | None = None,
    world_mask: wp.array | None = None,
):
    wp.launch(
        _convert_body_com_to_origin,
        dim=body_com.shape[0],
        inputs=[world_mask, body_wid, body_com, body_q_com],
        outputs=[body_q],
        device=body_com.device,
    )


def convert_base_origin_to_com(
    base_body_index: wp.array,
    body_com: wp.array,
    base_q: wp.array,
    base_q_com: wp.array,
):
    wp.launch(
        _convert_base_origin_to_com,
        dim=base_body_index.shape[0],
        inputs=[base_body_index, body_com, base_q],
        outputs=[base_q_com],
        device=base_q_com.device,
    )
