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

"""Provides data types, operations & interfaces for joint-limit detection."""

from dataclasses import dataclass, field

import warp as wp

from ..core.data import DataKamino
from ..core.joints import JOINT_QMAX, JOINT_QMIN, JointDoFType
from ..core.math import (
    quat_from_vec4,
    quat_log,
    screw,
)
from ..core.model import ModelKamino
from ..core.types import (
    float32,
    int32,
    uint32,
    uint64,
    vec1f,
    vec2f,
    vec2i,
    vec3f,
    vec4f,
    vec6f,
    vec7f,
)
from ..geometry.keying import build_pair_key2, make_bitmask
from ..utils import logger as msg

###
# Module interface
###

__all__ = ["LimitsKamino", "LimitsKaminoData"]


###
# Module configs
###

wp.set_module_options({"enable_backward": False})


###
# Containers
###


@dataclass
class LimitsKaminoData:
    """
    An SoA-based container to hold time-varying data of a set of active joint-limits.

    This container is intended as the final output of limit detectors and as input to solvers.
    """

    model_max_limits_host: int = 0
    """
    Host-side cache of the maximum number of limits allocated across all worlds.\n
    The number of allocated limits in the model is determined by the ModelBuilder when finalizing
    a ``ModelKamino``, and is equal to the sum over all finite-valued limits defined by each joint.\n
    The single entry is then less than or equal to the total ``num_joint_dofs`` of the entire model.\n
    This is cached on the host-side for managing data allocations and setting thread sizes in kernels.
    """

    world_max_limits_host: list[int] = field(default_factory=list)
    """
    Host-side cache of the maximum number of limits allocated per world.\n
    The number of allocated limits per world is determined by the ModelBuilder when finalizing a
    ``ModelKamino``, and is equal to the sum over all finite-valued limits defined by each joint of each world.\n
    Each entry is then less than or equal to the total ``num_joint_dofs`` of the corresponding world.\n
    This is cached on the host-side for managing data allocations and setting thread sizes in kernels.
    """

    model_max_limits: wp.array | None = None
    """
    The maximum number of limits allocated for the model across all worlds.\n
    The number of allocated limits in the model is determined by the ModelBuilder when finalizing
    a ``ModelKamino``, and is equal to the sum over all finite-valued limits defined by each joint.\n
    The single entry is then less than or equal to the total ``num_joint_dofs`` of the entire model.\n
    Shape of ``(1,)`` and type :class:`int32`.
    """

    model_active_limits: wp.array | None = None
    """
    The total number of active limits currently active in the model across all worlds.\n
    Shape of ``(1,)`` and type :class:`int32`.
    """

    world_max_limits: wp.array | None = None
    """
    The maximum number of limits allocated per world.\n
    The number of allocated limits per world is determined by the ModelBuilder when finalizing a
    ``ModelKamino``, and is equal to the sum over all finite-valued limits defined by each joint of each world.\n
    Each entry is then less than or equal to the total ``num_joint_dofs`` of the corresponding world.\n
    Shape of ``(num_worlds,)`` and type :class:`int32`.
    """

    world_active_limits: wp.array | None = None
    """
    The total number of active limits currently active per world.\n
    Shape of ``(num_worlds,)`` and type :class:`int32`.
    """

    wid: wp.array | None = None
    """
    The world index of each limit.\n
    Shape of ``(model_max_limits_host,)`` and type :class:`int32`.
    """

    lid: wp.array | None = None
    """
    The element index of each limit w.r.t its world.\n
    Shape of ``(model_max_limits_host,)`` and type :class:`int32`.
    """

    jid: wp.array | None = None
    """
    The element index of the corresponding joint w.r.t the model.\n
    Shape of ``(model_max_limits_host,)`` and type :class:`int32`.
    """

    bids: wp.array | None = None
    """
    The element indices of the interacting bodies w.r.t the model.\n
    Shape of ``(model_max_limits_host,)`` and type :class:`vec2i`.
    """

    dof: wp.array | None = None
    """
    The DoF indices along which limits are active w.r.t the world.\n
    Shape of ``(model_max_limits_host,)`` and type :class:`int32`.
    """

    side: wp.array | None = None
    """
    The direction (i.e. side) of the active limit.\n
    `1.0` for active min limits, `-1.0` for active max limits.\n
    Shape of ``(model_max_limits_host,)`` and type :class:`float32`.
    """

    r_q: wp.array | None = None
    """
    The amount of generalized coordinate violation per joint-limit.\n
    Shape of ``(model_max_limits_host,)`` and type :class:`float32`.
    """

    key: wp.array | None = None
    """
    Integer key uniquely identifying each limit.\n
    The per-limit key assignment is implementation-dependent, but is typically
    computed from the associated joint index as well as additional information such as:
    - limit index w.r.t the associated B/F body-pair\n
    Shape of ``(model_max_limits_host,)`` and type :class:`uint64`.
    """

    reaction: wp.array | None = None
    """
    The constraint reaction per joint-limit.\n
    This is to be set by solvers at each step, and also
    facilitates limit visualization and warm-starting.\n
    Shape of ``(model_max_limits_host,)`` and type :class:`float32`.
    """

    velocity: wp.array | None = None
    """
    The constraint velocity per joint-limit.\n
    This is to be set by solvers at each step, and also
    facilitates limit visualization and warm-starting.\n
    Shape of ``(model_max_limits_host,)`` and type :class:`float32`.
    """

    def clear(self):
        """
        Clears the count of active limits.
        """
        self.model_active_limits.zero_()
        self.world_active_limits.zero_()

    def reset(self):
        """
        Clears the count of active limits and resets limit data
        to sentinel values, indicating an empty set of limits.
        """
        self.clear()
        self.wid.fill_(-1)
        self.jid.fill_(-1)
        self.bids.fill_(vec2i(-1, -1))
        self.dof.fill_(-1)
        self.key.fill_(make_bitmask(63))
        self.reaction.zero_()
        self.velocity.zero_()


###
# Functions
###


@wp.func
def map_joint_coords_to_dofs_free(q_j: vec7f) -> vec6f:
    """Maps free joint quaternion to a local axes-aligned rotation vector."""
    v_j = quat_log(quat_from_vec4(q_j[3:7]))
    return screw(q_j[0:3], v_j)


@wp.func
def map_joint_coords_to_dofs_revolute(q_j: vec1f) -> vec1f:
    """No mapping needed for revolute joints."""
    return q_j


@wp.func
def map_joint_coords_to_dofs_prismatic(q_j: vec1f) -> vec1f:
    """No mapping needed for prismatic joints."""
    return q_j


@wp.func
def map_joint_coords_to_dofs_cylindrical(q_j: vec2f) -> vec2f:
    """No mapping needed for cylindrical joints."""
    return q_j


@wp.func
def map_joint_coords_to_dofs_universal(q_j: vec2f) -> vec2f:
    """No mapping needed for universal joints."""
    return q_j


@wp.func
def map_joint_coords_to_dofs_spherical(q_j: vec4f) -> vec3f:
    """Maps quaternion coordinates of a spherical
    joint to a local axes-aligned rotation vector."""
    return quat_log(quat_from_vec4(q_j))


@wp.func
def map_joint_coords_to_dofs_gimbal(q_j: vec3f) -> vec3f:
    """No mapping needed for gimbal joints."""
    return q_j


@wp.func
def map_joint_coords_to_dofs_cartesian(q_j: vec3f) -> vec3f:
    """No mapping needed for cartesian joints."""
    return q_j


def get_joint_coords_to_dofs_mapping_function(dof_type: JointDoFType):
    """
    Retrieves the function to map joint
    type-specific coordinates to DoF space.
    """
    if dof_type == JointDoFType.FREE:
        return map_joint_coords_to_dofs_free
    elif dof_type == JointDoFType.REVOLUTE:
        return map_joint_coords_to_dofs_revolute
    elif dof_type == JointDoFType.PRISMATIC:
        return map_joint_coords_to_dofs_prismatic
    elif dof_type == JointDoFType.CYLINDRICAL:
        return map_joint_coords_to_dofs_cylindrical
    elif dof_type == JointDoFType.UNIVERSAL:
        return map_joint_coords_to_dofs_universal
    elif dof_type == JointDoFType.SPHERICAL:
        return map_joint_coords_to_dofs_spherical
    elif dof_type == JointDoFType.GIMBAL:
        return map_joint_coords_to_dofs_gimbal
    elif dof_type == JointDoFType.CARTESIAN:
        return map_joint_coords_to_dofs_cartesian
    elif dof_type == JointDoFType.FIXED:
        return None
    else:
        raise ValueError(f"Unknown joint DoF type: {dof_type}")


def make_read_joint_coords_map_and_limits(dof_type: JointDoFType):
    """
    Generates a function to read the joint type-specific dof-count,
    limits, and coordinates, and map the latter to DoF space.
    """
    # Retrieve the number of constraints and dofs
    num_dofs = dof_type.num_dofs
    num_coords = dof_type.num_coords

    # Define a vector type for the joint coordinates
    coordsvec_type = dof_type.coords_storage_type

    # Generate a joint type-specific function to write the
    # computed joint state into the model data arrays
    @wp.func
    def _read_joint_coords_map_and_limits(
        # Inputs:
        dofs_offset: int32,  # Index offset of the joint DoFs
        coords_offset: int32,  # Index offset of the joint coordinates
        model_joint_q_j_min: wp.array(dtype=float32),
        model_joint_q_j_max: wp.array(dtype=float32),
        state_joints_q_j: wp.array(dtype=float32),
    ) -> tuple[int32, vec6f, vec6f, vec6f]:
        # Statically define the joint DoF counts
        d_j = wp.static(num_dofs)

        # Pre-allocate joint data for the largest-case (6 DoFs)
        q_j_min = vec6f(0.0)
        q_j_max = vec6f(0.0)
        q_j_map = vec6f(0.0)
        q_j = coordsvec_type(0.0)

        # Only write the DoF coordinates and velocities if the joint defines DoFs
        # NOTE: This will be disabled for fixed joints
        if wp.static(num_dofs > 0):
            # Read the joint DoF limits
            for j in range(num_dofs):
                q_j_min[j] = model_joint_q_j_min[dofs_offset + j]
                q_j_max[j] = model_joint_q_j_max[dofs_offset + j]

            # Read the joint coordinates
            for j in range(num_coords):
                q_j[j] = state_joints_q_j[coords_offset + j]

            # Map the joint coordinates to DoF space
            q_j_map[0:num_dofs] = wp.static(get_joint_coords_to_dofs_mapping_function(dof_type))(q_j)

        # Return the constructed joint DoF count, limits and mapped coordinates
        return d_j, q_j_min, q_j_max, q_j_map

    # Return the function
    return _read_joint_coords_map_and_limits


@wp.func
def read_joint_coords_map_and_limits(
    dof_type: int32,
    dofs_offset: int32,
    coords_offset: int32,
    model_joint_q_j_min: wp.array(dtype=float32),
    model_joint_q_j_max: wp.array(dtype=float32),
    state_joints_q_j: wp.array(dtype=float32),
) -> tuple[int32, vec6f, vec6f, vec6f]:
    if dof_type == JointDoFType.REVOLUTE:
        d_j, q_j_min, q_j_max, q_j_map = wp.static(make_read_joint_coords_map_and_limits(JointDoFType.REVOLUTE))(
            dofs_offset,
            coords_offset,
            model_joint_q_j_min,
            model_joint_q_j_max,
            state_joints_q_j,
        )

    elif dof_type == JointDoFType.PRISMATIC:
        d_j, q_j_min, q_j_max, q_j_map = wp.static(make_read_joint_coords_map_and_limits(JointDoFType.PRISMATIC))(
            dofs_offset,
            coords_offset,
            model_joint_q_j_min,
            model_joint_q_j_max,
            state_joints_q_j,
        )

    elif dof_type == JointDoFType.CYLINDRICAL:
        d_j, q_j_min, q_j_max, q_j_map = wp.static(make_read_joint_coords_map_and_limits(JointDoFType.CYLINDRICAL))(
            dofs_offset,
            coords_offset,
            model_joint_q_j_min,
            model_joint_q_j_max,
            state_joints_q_j,
        )

    elif dof_type == JointDoFType.UNIVERSAL:
        d_j, q_j_min, q_j_max, q_j_map = wp.static(make_read_joint_coords_map_and_limits(JointDoFType.UNIVERSAL))(
            dofs_offset,
            coords_offset,
            model_joint_q_j_min,
            model_joint_q_j_max,
            state_joints_q_j,
        )

    elif dof_type == JointDoFType.SPHERICAL:
        d_j, q_j_min, q_j_max, q_j_map = wp.static(make_read_joint_coords_map_and_limits(JointDoFType.SPHERICAL))(
            dofs_offset,
            coords_offset,
            model_joint_q_j_min,
            model_joint_q_j_max,
            state_joints_q_j,
        )

    elif dof_type == JointDoFType.GIMBAL:
        d_j, q_j_min, q_j_max, q_j_map = wp.static(make_read_joint_coords_map_and_limits(JointDoFType.GIMBAL))(
            dofs_offset,
            coords_offset,
            model_joint_q_j_min,
            model_joint_q_j_max,
            state_joints_q_j,
        )

    elif dof_type == JointDoFType.CARTESIAN:
        d_j, q_j_min, q_j_max, q_j_map = wp.static(make_read_joint_coords_map_and_limits(JointDoFType.CARTESIAN))(
            dofs_offset,
            coords_offset,
            model_joint_q_j_min,
            model_joint_q_j_max,
            state_joints_q_j,
        )

    elif dof_type == JointDoFType.FREE:
        d_j, q_j_min, q_j_max, q_j_map = wp.static(make_read_joint_coords_map_and_limits(JointDoFType.FREE))(
            dofs_offset,
            coords_offset,
            model_joint_q_j_min,
            model_joint_q_j_max,
            state_joints_q_j,
        )
    else:
        d_j = int32(0)
        q_j_min = vec6f(0.0)
        q_j_max = vec6f(0.0)
        q_j_map = vec6f(0.0)

    # Return the joint DoF count, limits and mapped coordinates
    return d_j, q_j_min, q_j_max, q_j_map


@wp.func
def detect_active_dof_limit(
    # Inputs:
    model_max_limits: int32,
    world_max_limits: int32,
    wid: int32,
    jid: int32,
    dof: int32,
    dofid: int32,
    bid_B: int32,
    bid_F: int32,
    q: float32,
    qmin: float32,
    qmax: float32,
    # Outputs:
    limits_model_num: wp.array(dtype=int32),
    limits_world_num: wp.array(dtype=int32),
    limits_wid: wp.array(dtype=int32),
    limits_lid: wp.array(dtype=int32),
    limits_jid: wp.array(dtype=int32),
    limits_bids: wp.array(dtype=vec2i),
    limits_dof: wp.array(dtype=int32),
    limits_side: wp.array(dtype=float32),
    limits_r_q: wp.array(dtype=float32),
    limits_key: wp.array(dtype=uint64),
):
    # Retrieve the state of the joint
    r_min = q - qmin
    r_max = qmax - q
    exceeds_min = r_min < 0.0
    exceeds_max = r_max < 0.0
    if exceeds_min or exceeds_max:
        mlid = wp.atomic_add(limits_model_num, 0, 1)
        wlid = wp.atomic_add(limits_world_num, wid, 1)
        if mlid < model_max_limits and wlid < world_max_limits:
            # Store the limit data
            limits_wid[mlid] = wid
            limits_lid[mlid] = wlid
            limits_jid[mlid] = jid
            limits_bids[mlid] = vec2i(bid_B, bid_F)
            limits_dof[mlid] = dofid
            limits_side[mlid] = 1.0 if exceeds_min else -1.0
            limits_r_q[mlid] = r_min if exceeds_min else r_max
            limits_key[mlid] = build_pair_key2(uint32(jid), uint32(dof))


###
# Kernels
###


@wp.kernel
def _detect_active_joint_configuration_limits(
    model_info_joint_dofs_offset: wp.array(dtype=int32),
    model_info_joint_coords_offset: wp.array(dtype=int32),
    model_joint_wid: wp.array(dtype=int32),
    model_joint_dof_type: wp.array(dtype=int32),
    model_joint_dofs_offset: wp.array(dtype=int32),
    model_joint_coords_offset: wp.array(dtype=int32),
    model_joint_bid_B: wp.array(dtype=int32),
    model_joint_bid_F: wp.array(dtype=int32),
    model_joint_q_j_min: wp.array(dtype=float32),
    model_joint_q_j_max: wp.array(dtype=float32),
    state_joints_q_j: wp.array(dtype=float32),
    limits_model_max: wp.array(dtype=int32),
    limits_world_max: wp.array(dtype=int32),
    # Outputs:
    limits_model_num: wp.array(dtype=int32),
    limits_world_num: wp.array(dtype=int32),
    limits_wid: wp.array(dtype=int32),
    limits_lid: wp.array(dtype=int32),
    limits_jid: wp.array(dtype=int32),
    limits_bids: wp.array(dtype=vec2i),
    limits_dof: wp.array(dtype=int32),
    limits_side: wp.array(dtype=float32),
    limits_r_q: wp.array(dtype=float32),
    limits_key: wp.array(dtype=uint64),
):
    # Retrieve the joint index for the current thread
    # This will be the index w.r.r the model
    jid = wp.tid()

    # Retrieve the joint-specific model data
    wid = model_joint_wid[jid]
    dof_type_j = model_joint_dof_type[jid]
    dofs_offset_j = model_joint_dofs_offset[jid]
    coords_offset_j = model_joint_coords_offset[jid]
    bid_B_j = model_joint_bid_B[jid]
    bid_F_j = model_joint_bid_F[jid]

    # Retrieve the max limits of the model and world
    model_max_limits = limits_model_max[0]
    world_max_limits = limits_world_max[wid]

    # Skip the joint limits check if:
    # - the DoF type is fixed
    # - if the world has not limits allocated
    # - if the model has not limits allocated
    if dof_type_j == JointDoFType.FIXED or world_max_limits == 0 or model_max_limits == 0:
        return

    # Extract the index offset of the world's joint DoFs w.r.t the model
    world_dofs_offset = model_info_joint_dofs_offset[wid]
    world_coords_offset = model_info_joint_coords_offset[wid]

    # Compute total index offset of the joint's DoFs w.r.t the model
    dofs_offset_total = dofs_offset_j + world_dofs_offset
    coords_offset_total = coords_offset_j + world_coords_offset

    # Read the joint DoF count, limits and coordinates mapped to DoF space
    # NOTE: We need to map to DoF space to compare against the limits when
    # the joint has non-minimal coordinates (e.g. spherical, free, etc.)
    d_j, q_j_min, q_j_max, q_j_map = read_joint_coords_map_and_limits(
        dof_type_j,
        dofs_offset_total,
        coords_offset_total,
        model_joint_q_j_min,
        model_joint_q_j_max,
        state_joints_q_j,
    )

    # Iterate over each DoF and check if a limit is active
    for dof in range(d_j):
        detect_active_dof_limit(
            # Inputs:
            model_max_limits,
            world_max_limits,
            wid,
            jid,
            dof,
            dofs_offset_j + dof,
            bid_B_j,
            bid_F_j,
            q_j_map[dof],
            q_j_min[dof],
            q_j_max[dof],
            # Outputs:
            limits_model_num,
            limits_world_num,
            limits_wid,
            limits_lid,
            limits_jid,
            limits_bids,
            limits_dof,
            limits_side,
            limits_r_q,
            limits_key,
        )


###
# Interfaces
###


class LimitsKamino:
    """
    A container to hold and manage time-varying joint-limits.
    """

    def __init__(
        self,
        model: ModelKamino | None = None,
        device: wp.DeviceLike = None,
    ):
        # The device on which to allocate the limits data
        self._device = device

        # Declare the joint-limits data container and initialize it to empty
        self._data: LimitsKaminoData = LimitsKaminoData()

        # Perform memory allocation if max_limits is specified
        if model is not None:
            self.finalize(model=model, device=device)

    ###
    # Properties
    ###

    @property
    def device(self) -> wp.DeviceLike:
        """
        Returns the device on which the limits data is allocated.
        """
        return self._device

    @property
    def data(self) -> LimitsKaminoData:
        """
        Returns the managed limits data container.
        """
        self._assert_has_data()
        return self._data

    @property
    def model_max_limits_host(self) -> int:
        """
        Returns the maximum number of limits allocated across all worlds.
        """
        self._assert_has_data()
        return self._data.model_max_limits_host

    @property
    def world_max_limits_host(self) -> list[int]:
        """
        Returns the maximum number of limits allocated per world.
        """
        self._assert_has_data()
        return self._data.world_max_limits_host

    @property
    def model_max_limits(self) -> wp.array:
        """
        Returns the total number of maximum limits for the model.\n
        Shape of ``(1,)`` and type :class:`int32`.
        """
        self._assert_has_data()
        return self._data.model_max_limits

    @property
    def model_active_limits(self) -> wp.array:
        """
        Returns the total number of active limits for the model.\n
        Shape of ``(1,)`` and type :class:`int32`.
        """
        self._assert_has_data()
        return self._data.model_active_limits

    @property
    def world_max_limits(self) -> wp.array:
        """
        Returns the total number of maximum limits per world.\n
        Shape of ``(num_worlds,)`` and type :class:`int32`.
        """
        self._assert_has_data()
        return self._data.world_max_limits

    @property
    def world_active_limits(self) -> wp.array:
        """
        Returns the total number of active limits per world.\n
        Shape of ``(num_worlds,)`` and type :class:`int32`.
        """
        self._assert_has_data()
        return self._data.world_active_limits

    @property
    def wid(self) -> wp.array:
        """
        Returns the world index of each limit.\n
        Shape of ``(model_max_limits_host,)`` and type :class:`int32`.
        """
        self._assert_has_data()
        return self._data.wid

    @property
    def lid(self) -> wp.array:
        """
        Returns the element index of each limit w.r.t its world.\n
        Shape of ``(model_max_limits_host,)`` and type :class:`int32`.
        """
        self._assert_has_data()
        return self._data.lid

    @property
    def jid(self) -> wp.array:
        """
        Returns the element index of the corresponding joint w.r.t the world.\n
        Shape of ``(model_max_limits_host,)`` and type :class:`int32`.
        """
        self._assert_has_data()
        return self._data.jid

    @property
    def bids(self) -> wp.array:
        """
        Returns the element indices of the interacting bodies w.r.t the model.\n
        Shape of ``(model_max_limits_host,)`` and type :class:`vec2i`.
        """
        self._assert_has_data()
        return self._data.bids

    @property
    def dof(self) -> wp.array:
        """
        Returns the DoF indices along which limits are active w.r.t the world.\n
        Shape of ``(model_max_limits_host,)`` and type :class:`int32`.
        """
        self._assert_has_data()
        return self._data.dof

    @property
    def side(self) -> wp.array:
        """
        Returns the direction (i.e. side) of the active limit.\n
        `1.0` for active min limits, `-1.0` for active max limits.\n
        Shape of ``(model_max_limits_host,)`` and type :class:`float32`.
        """
        self._assert_has_data()
        return self._data.side

    @property
    def r_q(self) -> wp.array:
        """
        Returns the amount of generalized coordinate violation per joint-limit.\n
        Shape of ``(model_max_limits_host,)`` and type :class:`float32`.
        """
        self._assert_has_data()
        return self._data.r_q

    @property
    def key(self) -> wp.array:
        """
        Returns the integer key uniquely identifying each limit.\n
        Shape of ``(model_max_limits_host,)`` and type :class:`uint64`.
        """
        self._assert_has_data()
        return self._data.key

    @property
    def reaction(self) -> wp.array:
        """
        Returns constraint velocity per joint-limit.\n
        Shape of ``(model_max_limits_host,)`` and type :class:`float32`.
        """
        self._assert_has_data()
        return self._data.reaction

    @property
    def velocity(self) -> wp.array:
        """
        Returns constraint velocity per joint-limit.\n
        Shape of ``(model_max_limits_host,)`` and type :class:`float32`.
        """
        self._assert_has_data()
        return self._data.velocity

    ###
    # Operations
    ###

    def finalize(self, model: ModelKamino, device: wp.DeviceLike = None):
        # Ensure the model is valid
        if model is None:
            raise ValueError("LimitsKamino: model must be specified for allocation (got None)")
        elif not isinstance(model, ModelKamino):
            raise TypeError("LimitsKamino: model must be an instance of ModelKamino")

        # Extract the joint limits allocation sizes from the model
        # The memory allocation requires the total number of limits (over multiple worlds)
        # as well as the limit capacities for each world. Corresponding sizes are defaulted to 0 (empty).
        model_max_limits = 0
        world_max_limits = [0] * model.size.num_worlds
        joint_wid = model.joints.wid.numpy()
        joint_num_dofs = model.joints.num_dofs.numpy()
        joint_q_j_min = model.joints.q_j_min.numpy()
        joint_q_j_max = model.joints.q_j_max.numpy()
        num_joints = len(joint_wid)
        dofs_start = 0
        for j in range(num_joints):
            for dof in range(joint_num_dofs[j]):
                if joint_q_j_min[dofs_start + dof] > JOINT_QMIN or joint_q_j_max[dofs_start + dof] < JOINT_QMAX:
                    model_max_limits += 1
                    world_max_limits[joint_wid[j]] += 1
            dofs_start += joint_num_dofs[j]

        # Skip allocation if there are no limits to allocate
        if model_max_limits == 0:
            msg.debug("LimitsKamino: Skipping joint-limit data allocations since total requested capacity was `0`.")
            return

        # Override the device if specified
        if device is not None:
            self._device = device

        # Allocate the limits data on the specified device
        with wp.ScopedDevice(self._device):
            self._data = LimitsKaminoData(
                model_max_limits_host=model_max_limits,
                world_max_limits_host=world_max_limits,
                model_max_limits=wp.array([model_max_limits], dtype=int32),
                model_active_limits=wp.zeros(shape=1, dtype=int32),
                world_max_limits=wp.array(world_max_limits, dtype=int32),
                world_active_limits=wp.zeros(shape=len(world_max_limits), dtype=int32),
                wid=wp.zeros(shape=model_max_limits, dtype=int32),
                lid=wp.zeros(shape=model_max_limits, dtype=int32),
                jid=wp.zeros(shape=model_max_limits, dtype=int32),
                bids=wp.zeros(shape=model_max_limits, dtype=vec2i),
                dof=wp.zeros(shape=model_max_limits, dtype=int32),
                side=wp.zeros(shape=model_max_limits, dtype=float32),
                r_q=wp.zeros(shape=model_max_limits, dtype=float32),
                key=wp.full(shape=model_max_limits, value=make_bitmask(63), dtype=uint64),
                reaction=wp.zeros(shape=model_max_limits, dtype=float32),
                velocity=wp.zeros(shape=model_max_limits, dtype=float32),
            )

    def clear(self):
        """
        Clears the active limits count.
        """
        if self._data is not None and self._data.model_max_limits_host > 0:
            self._data.clear()

    def reset(self):
        """
        Resets the limits data to sentinel values.
        """
        if self._data is not None and self._data.model_max_limits_host > 0:
            self._data.reset()

    def detect(
        self,
        model: ModelKamino,
        data: DataKamino,
    ):
        """
        Detects the active joint limits in the model and updates the limits data.

        Args:
            model (ModelKamino): The model to detect limits for.
            state (DataKamino): The current state of the model.
        """
        # Skip this operation if no contacts data has been allocated
        if self._data is None or self._data.model_max_limits_host <= 0:
            return

        # Ensure the model and state are valid
        if model is None:
            raise ValueError("LimitsKamino: model must be specified for detection (got None)")
        elif not isinstance(model, ModelKamino):
            raise TypeError("LimitsKamino: model must be an instance of ModelKamino")
        if data is None:
            raise ValueError("LimitsKamino: data must be specified for detection (got None)")
        elif not isinstance(data, DataKamino):
            raise TypeError("LimitsKamino: data must be an instance of DataKamino")

        # Ensure the limits data is allocated on the same device as the model
        if self._device is not None and self._device != model.device:
            raise ValueError(f"LimitsKamino: data device {self._device} does not match model device {model.device}")

        # Clear the current limits count
        self.clear()

        # Launch the detection kernel
        wp.launch(
            kernel=_detect_active_joint_configuration_limits,
            dim=model.size.sum_of_num_joints,
            inputs=[
                # Inputs:
                model.info.joint_dofs_offset,
                model.info.joint_coords_offset,
                model.joints.wid,
                model.joints.dof_type,
                model.joints.dofs_offset,
                model.joints.coords_offset,
                model.joints.bid_B,
                model.joints.bid_F,
                model.joints.q_j_min,
                model.joints.q_j_max,
                data.joints.q_j,
                self._data.model_max_limits,
                self._data.world_max_limits,
                # Outputs:
                self._data.model_active_limits,
                self._data.world_active_limits,
                self._data.wid,
                self._data.lid,
                self._data.jid,
                self._data.bids,
                self._data.dof,
                self._data.side,
                self._data.r_q,
                self._data.key,
            ],
        )

    ###
    # Internals
    ###

    def _assert_has_data(self):
        """
        Asserts that the limits data has been allocated.
        """
        if self._data is None:
            raise ValueError("LimitsKamino: data has not been allocated. Please call 'finalize()' first.")
