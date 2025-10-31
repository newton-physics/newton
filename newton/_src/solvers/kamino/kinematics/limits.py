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
KAMINO: Kinematics: Limits
"""

from __future__ import annotations

import warp as wp
from warp.context import Devicelike

from ..core.builder import ModelBuilder
from ..core.math import FLOAT32_MAX, FLOAT32_MIN
from ..core.model import Model, ModelData
from ..core.types import float32, int32, vec2i

###
# Module interface
###

__all__ = ["Limits", "LimitsData"]


###
# Module configs
###

wp.set_module_options({"enable_backward": False})


###
# Containers
###


class LimitsData:
    """
    An SoA-based container to hold time-varying data of a set of active joint-limits.

    This container is intended as the final output of limit detectors and as input to solvers.
    """

    def __init__(self):
        self.num_model_max_limits: int = 0
        """
        The maximum number of limits allocated across all worlds.\n
        This is cached on the host-side for managing data allocations and setting thread sizes in kernels.
        """

        self.num_world_max_limits: list[int] = [0]
        """
        The maximum number of limits allocated per world.\n
        This is cached on the host-side for managing data allocations and setting thread sizes in kernels.
        """

        self.model_max_limits: wp.array(dtype=int32) | None = None
        """
        The number of active limits per model.\n
        Shape of ``(1,)`` and type :class:`int32`.
        """

        self.model_num_limits: wp.array(dtype=int32) | None = None
        """
        The number of active limits per model.\n
        Shape of ``(1,)`` and type :class:`int32`.
        """

        self.world_max_limits: wp.array(dtype=int32) | None = None
        """
        The maximum number of limits per world.\n
        Shape of ``(num_worlds,)`` and type :class:`int32`.
        """

        self.world_num_limits: wp.array(dtype=int32) | None = None
        """
        The number of active limits per world.\n
        Shape of ``(num_worlds,)`` and type :class:`int32`.
        """

        self.wid: wp.array(dtype=int32) | None = None
        """
        The world index of each limit.\n
        Shape of ``(num_limits,)`` and type :class:`int32`.
        """

        self.lid: wp.array(dtype=int32) | None = None
        """
        The element index of each limit w.r.t its world.\n
        Shape of ``(num_limits,)`` and type :class:`int32`.
        """

        self.jid: wp.array(dtype=int32) | None = None
        """
        The element index of the corresponding joint w.r.t the world.\n
        Shape of ``(num_limits,)`` and type :class:`int32`.
        """

        self.bids: wp.array(dtype=vec2i) | None = None
        """
        The element indices of the interacting bodies w.r.t the model.\n
        Shape of ``(num_limits,)`` and type :class:`vec2i`.
        """

        self.dof: wp.array(dtype=int32) | None = None
        """
        The DoF indices along which limits are active w.r.t the world.\n
        Shape of ``(num_limits,)`` and type :class:`int32`.
        """

        self.side: wp.array(dtype=float32) | None = None
        """
        The direction (i.e. side) of the active limit.\n
        `1.0` for active min limits, `-1.0` for active max limits.\n
        Shape of ``(num_limits,)`` and type :class:`float32`.
        """

        self.r_q: wp.array(dtype=float32) | None = None
        """
        The amount of generalized coordinate violation per joint-limit.\n
        Shape of ``(num_limits,)`` and type :class:`float32`.
        """

        self.r_dq: wp.array(dtype=float32) | None = None
        """
        The amount of generalized velocity violation per joint-limit.\n
        Shape of ``(num_limits,)`` and type :class:`float32`.
        """

        self.r_tau: wp.array(dtype=float32) | None = None
        """
        The amount of generalized force violation per joint-limit.\n
        Shape of ``(num_limits,)`` and type :class:`float32`.
        """


###
# Kernels
###


@wp.kernel
def _detect_active_joint_configuration_limits(
    model_info_joint_dofs_offset: wp.array(dtype=int32),
    model_joints_wid: wp.array(dtype=int32),
    model_joints_num_dofs: wp.array(dtype=int32),
    model_joints_dofs_offset: wp.array(dtype=int32),
    model_joints_bid_B: wp.array(dtype=int32),
    model_joints_bid_F: wp.array(dtype=int32),
    model_joints_q_j_min: wp.array(dtype=float32),
    model_joints_q_j_max: wp.array(dtype=float32),
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
):
    # Retrieve the joint index for the current thread
    # This will be the index w.r.r the model
    jid = wp.tid()

    # Retrieve the world index of the joint
    wid = model_joints_wid[jid]

    # Retrieve the index offset of the joint's DoFs w.r.t the world
    dio = model_joints_dofs_offset[jid]

    # Retrieve the DoF size of the joint
    d_j = model_joints_num_dofs[jid]

    # Extract the index offset of the world's joint DoFs w.r.t the model
    jdio = model_info_joint_dofs_offset[wid]

    # Retrieve the max limits of the model and world
    model_max_limits = limits_model_max[0]
    world_max_limits = limits_world_max[wid]

    # Compute total index offset of the joint's DoFs w.r.t the model
    doi_j = jdio + dio

    # Iterate over each DoF and check if a limit is active
    for dof in range(d_j):
        # Compute the total index offset of the current DoF w.r.t the model
        dio_jd = doi_j + dof
        # Retrieve the state of the joint
        q = state_joints_q_j[dio_jd]
        qmin = model_joints_q_j_min[dio_jd]
        qmax = model_joints_q_j_max[dio_jd]
        r_min = q - qmin
        r_max = qmax - q
        exceeds_min = r_min < 0.0
        exceeds_max = r_max < 0.0
        if exceeds_min or exceeds_max:
            # TODO: This will cause problems if the number of limits exceeds the maximum as we are
            # incrementing the limits counters and do not decrement if we've exceeded the maximum
            mlid = wp.atomic_add(limits_model_num, 0, 1)
            wlid = wp.atomic_add(limits_world_num, wid, 1)
            if mlid < model_max_limits and wlid < world_max_limits:
                # Store the limit data
                limits_wid[mlid] = wid
                limits_lid[mlid] = wlid
                limits_jid[mlid] = jid
                limits_bids[mlid] = vec2i(model_joints_bid_B[jid], model_joints_bid_F[jid])
                limits_dof[mlid] = dio + dof
                limits_side[mlid] = 1.0 if exceeds_min else -1.0
                limits_r_q[mlid] = r_min if exceeds_min else r_max


###
# Interfaces
###


class Limits:
    """
    A container to hold and manage time-varying joint-limits.
    """

    def __init__(
        self,
        builder: ModelBuilder | None = None,
        device: Devicelike = None,
    ):
        # The device on which to allocate the limits data
        self._device = device

        # Declare and initialize the time-varying data container
        self._data: LimitsData = LimitsData()

        # Perofrm memory allocation if max_limits is specified
        if builder is not None:
            self.allocate(builder=builder, device=device)

    @property
    def num_model_max_limits(self) -> int:
        """
        The maximum number of limits allocated across all worlds.
        """
        return self._data.num_model_max_limits

    @property
    def num_world_max_limits(self) -> list[int]:
        """
        The maximum number of limits allocated per world.
        """
        return self._data.num_world_max_limits

    @property
    def data(self) -> LimitsData:
        """
        The time-varying limits data container.
        """
        return self._data

    @property
    def model_max_limits(self) -> wp.array:
        """
        The total number of maximum limits for the model.\n
        Shape of ``(1,)`` and type :class:`int32`.
        """
        return self._data.model_max_limits

    @property
    def model_num_limits(self) -> wp.array:
        """
        The total number of active limits for the model.\n
        Shape of ``(1,)`` and type :class:`int32`.
        """
        return self._data.model_num_limits

    @property
    def world_max_limits(self) -> wp.array:
        """
        The total number of maximum limits for the model.\n
        Shape of ``(num_worlds,)`` and type :class:`int32`.
        """
        return self._data.world_max_limits

    @property
    def world_num_limits(self) -> wp.array:
        """
        The total number of active limits for the model.\n
        Shape of ``(num_worlds,)`` and type :class:`int32`.
        """
        return self._data.world_num_limits

    @property
    def wid(self) -> wp.array:
        """
        The world index of each limit.
        """
        return self._data.wid

    @property
    def lid(self) -> wp.array:
        """
        The element index of each limit w.r.t its world.
        """
        return self._data.lid

    @property
    def jid(self) -> wp.array:
        """
        The element index of the corresponding joint w.r.t the world.
        """
        return self._data.jid

    @property
    def bids(self) -> wp.array:
        """
        The element indices of the interacting bodies w.r.t the model.
        """
        return self._data.bids

    @property
    def dof(self) -> wp.array:
        """
        The DoF indices along which limits are active w.r.t the world.
        """
        return self._data.dof

    @property
    def side(self) -> wp.array:
        """
        The direction (i.e. side) of the active limit.\n
        `1.0` for active min limits, `-1.0` for active max limits.
        """
        return self._data.side

    @property
    def r_q(self) -> wp.array:
        """
        The the amount of generalized coordinate violation per active joint-limit.
        """
        return self._data.r_q

    @property
    def r_dq(self) -> wp.array:
        """
        The the amount of generalized velocity violation per active joint-limit.
        """
        return self._data.r_dq

    @property
    def r_tau(self) -> wp.array:
        """
        The the amount of generalized force violation per active joint-limit.
        """
        return self._data.r_tau

    def allocate(self, builder: ModelBuilder, device: Devicelike = None):
        # Ensure the builder is valid
        if builder is None:
            raise ValueError("Limits: builder must be specified for allocation (got None)")
        elif not isinstance(builder, ModelBuilder):
            raise TypeError("Limits: builder must be an instance of ModelBuilder")

        # Extract the joint limits allocation sizes from the builder
        # The memory allocation requires the total number of limits (over multiple worlds)
        # as well as the limit capacities for each world. Corresponding sizes are defaulted to 0 (empty).
        model_max_limits = 0
        world_max_limits = [0] * builder.num_worlds
        for _j, joint in enumerate(builder.joints):
            for dof in range(joint.num_dofs):
                # Check if the joint has finite generalized coordinate limits
                if joint.q_j_min[dof] > float(FLOAT32_MIN) or joint.q_j_max[dof] < float(FLOAT32_MAX):
                    model_max_limits += 1
                    world_max_limits[joint.wid] += 1
                # TODO: handle generalized velocity and force limits (?)

        # Override the device if specified
        if device is not None:
            self._device = device

        # Allocate the limits data on the specified device
        with wp.ScopedDevice(self._device):
            self._data.num_model_max_limits = model_max_limits
            self._data.num_world_max_limits = world_max_limits
            self._data.model_max_limits = wp.array([model_max_limits], dtype=int32)
            self._data.model_num_limits = wp.zeros(shape=1, dtype=int32)
            self._data.world_max_limits = wp.array(world_max_limits, dtype=int32)
            self._data.world_num_limits = wp.zeros(shape=len(world_max_limits), dtype=int32)
            self._data.wid = wp.zeros(shape=self.num_model_max_limits, dtype=int32)
            self._data.lid = wp.zeros(shape=self.num_model_max_limits, dtype=int32)
            self._data.jid = wp.zeros(shape=self.num_model_max_limits, dtype=int32)
            self._data.bids = wp.zeros(shape=self.num_model_max_limits, dtype=vec2i)
            self._data.dof = wp.zeros(shape=self.num_model_max_limits, dtype=int32)
            self._data.side = wp.zeros(shape=self.num_model_max_limits, dtype=float32)
            self._data.r_q = wp.zeros(shape=self.num_model_max_limits, dtype=float32)
            self._data.r_dq = wp.zeros(shape=self.num_model_max_limits, dtype=float32)
            self._data.r_tau = wp.zeros(shape=self.num_model_max_limits, dtype=float32)

    def clear(self):
        """
        Clears the active limits count.
        """
        self._data.model_num_limits.zero_()
        self._data.world_num_limits.zero_()

    def zero(self):
        """
        Resets the limits data to zero.
        """
        self._data.model_num_limits.zero_()
        self._data.world_num_limits.zero_()
        self._data.wid.zero_()
        self._data.jid.zero_()
        self._data.lid.zero_()
        self._data.bids.zero_()
        self._data.dof.zero_()
        self._data.side.zero_()
        self._data.r_q.zero_()
        self._data.r_dq.zero_()
        self._data.r_tau.zero_()

    def detect(
        self,
        model: Model,
        data: ModelData,
    ):
        """
        Detects the active joint limits in the model and updates the limits data.

        Args:
            model (Model): The model to detect limits for.
            state (ModelData): The current state of the model.
        """
        # Ensure the model and state are valid
        if model is None:
            raise ValueError("Limits: model must be specified for detection (got None)")
        elif not isinstance(model, Model):
            raise TypeError("Limits: model must be an instance of Model")
        if data is None:
            raise ValueError("Limits: data must be specified for detection (got None)")
        elif not isinstance(data, ModelData):
            raise TypeError("Limits: data must be an instance of ModelData")

        # Ensure the limits data is allocated
        if self._data.model_max_limits is None:
            raise ValueError("Limits: data must be allocated before detection (got None)")

        # Ensure the limits data is allocated on the same device as the model
        if self._device is not None and self._device != model.device:
            raise ValueError(f"Limits: data device {self._device} does not match model device {model.device}")

        # Clear the limits data
        self.clear()

        # Launch the detection kernel
        wp.launch(
            kernel=_detect_active_joint_configuration_limits,
            dim=model.size.sum_of_num_joints,
            inputs=[
                # Inputs:
                model.info.joint_dofs_offset,
                model.joints.wid,
                model.joints.num_dofs,
                model.joints.dofs_offset,
                model.joints.bid_B,
                model.joints.bid_F,
                model.joints.q_j_min,
                model.joints.q_j_max,
                data.joints.q_j,
                self._data.model_max_limits,
                self._data.world_max_limits,
                # Outputs:
                self._data.model_num_limits,
                self._data.world_num_limits,
                self._data.wid,
                self._data.lid,
                self._data.jid,
                self._data.bids,
                self._data.dof,
                self._data.side,
                self._data.r_q,
            ],
        )
