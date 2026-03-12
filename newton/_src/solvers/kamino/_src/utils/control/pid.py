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

"""PID Controller Interfaces"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import warp as wp

from ...core.control import ControlKamino
from ...core.joints import JointActuationType
from ...core.model import ModelKamino
from ...core.state import StateKamino
from ...core.time import TimeData
from ...core.types import FloatArrayLike, IntArrayLike, float32, int32

###
# Module interface
###


__all__ = [
    "JointSpacePIDController",
    "PIDControllerData",
    "compute_jointspace_pid_control",
    "reset_jointspace_pid_references",
]


###
# Module configs
###

wp.set_module_options({"enable_backward": False})


###
# Types
###


@dataclass
class PIDControllerData:
    """A data container for joint-space PID controller parameters and state."""

    q_j_ref: wp.array | None = None
    """The reference actuator joint positions."""
    dq_j_ref: wp.array | None = None
    """The reference actuator joint velocities."""
    tau_j_ref: wp.array | None = None
    """The feedforward actuator joint torques."""
    K_p: wp.array | None = None
    """The proportional gains."""
    K_i: wp.array | None = None
    """The integral gains."""
    K_d: wp.array | None = None
    """The derivative gains."""
    integrator: wp.array | None = None
    """Integrator of joint-space position tracking error."""
    decimation: wp.array | None = None
    """The control decimation for each world expressed as a multiple of simulation steps."""


###
# Kernels
###


@wp.kernel
def _reset_jointspace_pid_references(
    # Inputs
    model_info_joint_dofs_offset: wp.array(dtype=int32),
    model_info_joint_actuated_dofs_offset: wp.array(dtype=int32),
    model_joints_wid: wp.array(dtype=int32),
    model_joints_act_type: wp.array(dtype=int32),
    model_joints_num_dofs: wp.array(dtype=int32),
    model_joints_dofs_offset: wp.array(dtype=int32),
    model_joints_actuated_dofs_offset: wp.array(dtype=int32),
    state_joints_q_j: wp.array(dtype=float32),
    state_joints_dq_j: wp.array(dtype=float32),
    # Outputs
    controller_q_j_ref: wp.array(dtype=float32),
    controller_dq_j_ref: wp.array(dtype=float32),
):
    """
    A kernel to reset motion references of the joint-space controller.
    """
    # Retrieve the the joint index from the thread indices
    jid = wp.tid()

    # Retrieve the joint actuation type
    act_type = model_joints_act_type[jid]

    # Retrieve the world index from the thread indices
    wid = model_joints_wid[jid]

    # Only proceed for force actuated joints and at
    # simulation steps matching the control decimation
    if act_type != JointActuationType.FORCE:
        return

    # Retrieve the offset of the world's joints in the global DoF vector
    world_dof_offset = model_info_joint_dofs_offset[wid]
    world_actuated_dof_offset = model_info_joint_actuated_dofs_offset[wid]

    # Retrieve the number of DoFs and offset of the joint
    num_dofs = model_joints_num_dofs[jid]
    dofs_offset = model_joints_dofs_offset[jid]
    actuated_dofs_offset = model_joints_actuated_dofs_offset[jid]

    # Compute the global DoF offset of the joint
    dofs_offset += world_dof_offset
    actuated_dofs_offset += world_actuated_dof_offset

    # Iterate over the DoFs of the joint
    for dof in range(num_dofs):
        # Compute the DoF index in the global DoF vector
        dof_index = dofs_offset + dof

        # Compute the actuator index in the controller vectors
        actuator_dof_index = actuated_dofs_offset + dof

        # Retrieve the current joint state
        q_j = state_joints_q_j[dof_index]
        dq_j = state_joints_dq_j[dof_index]

        # Retrieve the current controller references
        controller_q_j_ref[actuator_dof_index] = q_j
        controller_dq_j_ref[actuator_dof_index] = dq_j


@wp.kernel
def _compute_jointspace_pid_control(
    # Inputs
    model_info_joint_dofs_offset: wp.array(dtype=int32),
    model_info_joint_actuated_dofs_offset: wp.array(dtype=int32),
    model_joints_wid: wp.array(dtype=int32),
    model_joints_act_type: wp.array(dtype=int32),
    model_joints_num_dofs: wp.array(dtype=int32),
    model_joints_dofs_offset: wp.array(dtype=int32),
    model_joints_actuated_dofs_offset: wp.array(dtype=int32),
    model_joints_tau_j_max: wp.array(dtype=float32),
    model_time_dt: wp.array(dtype=float32),
    state_time_steps: wp.array(dtype=int32),
    state_joints_q_j: wp.array(dtype=float32),
    state_joints_dq_j: wp.array(dtype=float32),
    controller_q_j_ref: wp.array(dtype=float32),
    controller_dq_j_ref: wp.array(dtype=float32),
    controller_tau_j_ref: wp.array(dtype=float32),
    controller_K_p: wp.array(dtype=float32),
    controller_K_i: wp.array(dtype=float32),
    controller_K_d: wp.array(dtype=float32),
    controller_integrator: wp.array(dtype=float32),
    controller_decimation: wp.array(dtype=int32),
    # Outputs
    control_tau_j: wp.array(dtype=float32),
):
    """
    A kernel to compute joint-space PID control outputs for force-actuated joints.
    """
    # Retrieve the the joint index from the thread indices
    jid = wp.tid()

    # Retrieve the joint actuation type
    act_type = model_joints_act_type[jid]

    # Retrieve the world index from the thread indices
    wid = model_joints_wid[jid]

    # Retrieve the current simulation step
    step = state_time_steps[wid]

    # Retrieve the control decimation for the world
    decimation = controller_decimation[wid]

    # Only proceed for force actuated joints and at
    # simulation steps matching the control decimation
    if act_type != JointActuationType.FORCE or step % decimation != 0:
        return

    # Retrieve the time step and current time
    dt = model_time_dt[wid]

    # Decimate the simulation time-step by the control
    # decimation to get the effective control time-step
    dt *= float32(decimation)

    # Retrieve the offset of the world's joints in the global DoF vector
    world_dof_offset = model_info_joint_dofs_offset[wid]
    world_actuated_dof_offset = model_info_joint_actuated_dofs_offset[wid]

    # Retrieve the number of DoFs and offset of the joint
    num_dofs = model_joints_num_dofs[jid]
    dofs_offset = model_joints_dofs_offset[jid]
    actuated_dofs_offset = model_joints_actuated_dofs_offset[jid]

    # Compute the global DoF offset of the joint
    dofs_offset += world_dof_offset
    actuated_dofs_offset += world_actuated_dof_offset

    # Iterate over the DoFs of the joint
    for dof in range(num_dofs):
        # Compute the DoF index in the global DoF vector
        joint_dof_index = dofs_offset + dof

        # Compute the actuator index in the controller vectors
        actuator_dof_index = actuated_dofs_offset + dof

        # Retrieve the maximum limit of the generalized actuator forces
        tau_j_max = model_joints_tau_j_max[joint_dof_index]

        # Retrieve the current joint state
        q_j = state_joints_q_j[joint_dof_index]
        dq_j = state_joints_dq_j[joint_dof_index]

        # Retrieve the current controller references
        q_j_ref = controller_q_j_ref[actuator_dof_index]
        dq_j_ref = controller_dq_j_ref[actuator_dof_index]
        tau_j_ref = controller_tau_j_ref[actuator_dof_index]

        # Retrieve the controller gains and integrator state
        K_p = controller_K_p[actuator_dof_index]
        K_i = controller_K_i[actuator_dof_index]
        K_d = controller_K_d[actuator_dof_index]
        integrator = controller_integrator[actuator_dof_index]

        # Compute integral limit
        integrator_max = 0.0
        if K_i > 0.0:
            integrator_max = tau_j_max / K_i  # Overflow is not a concern. wp.clamp will work as expected with inf

        # Compute tracking errors
        q_j_err = q_j_ref - q_j
        dq_j_err = dq_j_ref - dq_j

        # Update the integrator state with anti-windup clamping
        integrator += q_j_err * dt
        integrator = wp.clamp(integrator, -integrator_max, integrator_max)

        # Compute the Feed-Forward + PID control generalized forces
        # NOTE: We also clamp the final control forces to avoid exceeding actuator limits
        tau_j_c = tau_j_ref + K_p * q_j_err + K_d * dq_j_err + K_i * integrator
        tau_j_c = wp.clamp(tau_j_c, -tau_j_max, tau_j_max)

        # Store the updated integrator state and actuator control forces
        controller_integrator[actuator_dof_index] = integrator
        control_tau_j[joint_dof_index] = tau_j_c


###
# Launchers
###


def reset_jointspace_pid_references(
    # Inputs:
    model: ModelKamino,
    state: StateKamino,
    # Outputs:
    controller: PIDControllerData,
) -> None:
    """
    A kernel launcher to reset joint-space PID controller motion references.
    """
    wp.launch(
        _reset_jointspace_pid_references,
        dim=model.size.sum_of_num_joints,
        inputs=[
            # Inputs
            model.info.joint_dofs_offset,
            model.info.joint_actuated_dofs_offset,
            model.joints.wid,
            model.joints.act_type,
            model.joints.num_dofs,
            model.joints.dofs_offset,
            model.joints.actuated_dofs_offset,
            state.q_j,
            state.dq_j,
            # Outputs
            controller.q_j_ref,
            controller.dq_j_ref,
        ],
    )


def compute_jointspace_pid_control(
    # Inputs:
    model: ModelKamino,
    state: StateKamino,
    time: TimeData,
    controller: PIDControllerData,
    # Outputs:
    control: ControlKamino,
) -> None:
    """
    A kernel launcher to compute joint-space PID control outputs for force-actuated joints.
    """
    wp.launch(
        _compute_jointspace_pid_control,
        dim=model.size.sum_of_num_joints,
        inputs=[
            # Inputs
            model.info.joint_dofs_offset,
            model.info.joint_actuated_dofs_offset,
            model.joints.wid,
            model.joints.act_type,
            model.joints.num_dofs,
            model.joints.dofs_offset,
            model.joints.actuated_dofs_offset,
            model.joints.tau_j_max,
            model.time.dt,
            time.steps,
            state.q_j,
            state.dq_j,
            controller.q_j_ref,
            controller.dq_j_ref,
            controller.tau_j_ref,
            controller.K_p,
            controller.K_i,
            controller.K_d,
            controller.integrator,
            controller.decimation,
            # Outputs
            control.tau_j,
        ],
    )


###
# Interfaces
###


class JointSpacePIDController:
    """
    A simple PID controller in joint space.

    This controller currently only supports single-DoF force-actuated joints.
    """

    def __init__(
        self,
        model: ModelKamino | None = None,
        K_p: FloatArrayLike | None = None,
        K_i: FloatArrayLike | None = None,
        K_d: FloatArrayLike | None = None,
        decimation: IntArrayLike | None = None,
        device: wp.DeviceLike = None,
    ):
        """
        A simple PID controller in joint space.

        Args:
            model (ModelKamino | None): The model container describing the system to be simulated.
                If None, call ``finalize()`` later.
            K_p (FloatArrayLike | None): Proportional gains per actuated joint DoF.
            K_i (FloatArrayLike | None): Integral gains per actuated joint DoF.
            K_d (FloatArrayLike | None): Derivative gains per actuated joint DoF.
            decimation (IntArrayLike | None): Control decimation for each world
                expressed as a multiple of simulation steps.
            device (wp.DeviceLike | None): Device to use for allocations and execution.
        """

        # Cache the device
        self._device: wp.DeviceLike = device

        # Declare the internal controller data
        self._data: PIDControllerData | None = None

        # If a model is provided, allocate the controller data
        if model is not None:
            self.finalize(model, K_p, K_i, K_d, decimation, device)

    ###
    # Properties
    ###

    @property
    def data(self) -> PIDControllerData:
        """The internal controller data."""
        if self._data is None:
            raise RuntimeError("Controller data is not allocated. Call finalize() first.")
        return self._data

    @property
    def device(self) -> wp.DeviceLike:
        """The device used for allocations and execution."""
        return self._device

    ###
    # Operations
    ###

    def finalize(
        self,
        model: ModelKamino,
        K_p: FloatArrayLike,
        K_i: FloatArrayLike,
        K_d: FloatArrayLike,
        decimation: IntArrayLike | None = None,
        device: wp.DeviceLike = None,
    ) -> None:
        """
        Allocates all internal data arrays of the controller.

        Args:
            model (ModelKamino): The model container describing the system to be simulated.
            K_p (FloatArrayLike): Proportional gains per actuated joint DoF.
            K_i (FloatArrayLike): Integral gains per actuated joint DoF.
            K_d (FloatArrayLike): Derivative gains per actuated joint DoF.
            decimation (IntArrayLike | None): Control decimation for each world expressed
                as a multiple of simulation steps. Defaults to 1 for all worlds if None.
            device (wp.DeviceLike | None): Device to use for allocations and execution.

        Raises:
            ValueError: If the model has no actuated DoFs.
            ValueError: If the model has multi-DoF actuated joints.
            ValueError: If the length of any gain array does not match the number of actuated DoFs.
            ValueError: If the length of the decimation array does not match the number of worlds.
        """

        # Get the number of actuated coordinates and DoFs
        num_actuated_coords = model.size.sum_of_num_actuated_joint_coords
        num_actuated_dofs = model.size.sum_of_num_actuated_joint_dofs

        # Check if there are any actuated DoFs
        if num_actuated_dofs == 0:
            raise ValueError("Model has no actuated DoFs.")

        # Ensure the model has only 1-DoF actuated joints
        if num_actuated_coords != num_actuated_dofs:
            raise ValueError(
                f"Model has {num_actuated_coords} actuated coordinates but {num_actuated_dofs} actuated DoFs. "
                "Joint-space PID control is currently incompatible with multi-DoF actuated joints."
            )

        # Check length of gain arrays
        if K_p is not None and len(K_p) != num_actuated_dofs:
            raise ValueError(f"K_p must have length {num_actuated_dofs}, but has length {len(K_p)}")
        if K_i is not None and len(K_i) != num_actuated_dofs:
            raise ValueError(f"K_i must have length {num_actuated_dofs}, but has length {len(K_i)}")
        if K_d is not None and len(K_d) != num_actuated_dofs:
            raise ValueError(f"K_d must have length {num_actuated_dofs}, but has length {len(K_d)}")
        if decimation is not None and len(decimation) != model.size.num_worlds:
            raise ValueError(f"decimation must have length {model.size.num_worlds}, but has length {len(decimation)}")

        # Override the device if provided
        if device is not None:
            self._device = device

        # Set default decimation if not provided
        if decimation is None:
            decimation = np.ones(model.size.num_worlds, dtype=np.int32)

        # Allocate the controller data
        with wp.ScopedDevice(self._device):
            self._data = PIDControllerData(
                q_j_ref=wp.zeros(num_actuated_dofs, dtype=float32),
                dq_j_ref=wp.zeros(num_actuated_dofs, dtype=float32),
                tau_j_ref=wp.zeros(num_actuated_dofs, dtype=float32),
                K_p=wp.array(K_p if K_p is not None else np.zeros(num_actuated_dofs), dtype=float32),
                K_i=wp.array(K_i if K_i is not None else np.zeros(num_actuated_dofs), dtype=float32),
                K_d=wp.array(K_d if K_d is not None else np.zeros(num_actuated_dofs), dtype=float32),
                integrator=wp.zeros(num_actuated_dofs, dtype=float32),
                decimation=wp.array(decimation, dtype=int32),
            )

    def reset(self, model: ModelKamino, state: StateKamino) -> None:
        """
        Reset the internal state of the controller.

        The motion references are reset to the current generalized
        joint states `q_j` and `dq_j`, while feedforward generalized
        forces `tau_j` and the integrator are set to zeros.

        Args:
            model (ModelKamino): The model container holding the time-invariant parameters of the simulation.
            state (StateKamino): The current state of the system to which the references will be reset.
        """

        # First reset the references to the current state
        reset_jointspace_pid_references(
            model=model,
            state=state,
            controller=self._data,
        )

        # Then zero the integrator and feedforward torques
        self._data.tau_j_ref.zero_()
        self._data.integrator.zero_()

    def set_references(
        self, q_j_ref: FloatArrayLike, dq_j_ref: FloatArrayLike | None = None, tau_j_ref: FloatArrayLike | None = None
    ) -> None:
        """
        Set the controller reference trajectories.

        Args:
            q_j_ref (FloatArrayLike): The reference generalized actuator positions.
            dq_j_ref (FloatArrayLike | None): The reference generalized actuator velocities.
            tau_j_ref (FloatArrayLike | None): The feedforward generalized actuator forces.
        """
        if len(q_j_ref) != len(self._data.q_j_ref):
            raise ValueError(f"q_j_ref must have length {len(self._data.q_j_ref)}, but has length {len(q_j_ref)}")
        self._data.q_j_ref.assign(q_j_ref)

        if dq_j_ref is not None:
            if len(dq_j_ref) != len(self._data.dq_j_ref):
                raise ValueError(
                    f"dq_j_ref must have length {len(self._data.dq_j_ref)}, but has length {len(dq_j_ref)}"
                )
            self._data.dq_j_ref.assign(dq_j_ref)

        if tau_j_ref is not None:
            if len(tau_j_ref) != len(self._data.tau_j_ref):
                raise ValueError(
                    f"tau_j_ref must have length {len(self._data.tau_j_ref)}, but has length {len(tau_j_ref)}"
                )
            self._data.tau_j_ref.assign(tau_j_ref)

    def compute(
        self,
        model: ModelKamino,
        state: StateKamino,
        time: TimeData,
        control: ControlKamino,
    ) -> None:
        """
        Compute the control torques.

        Args:
            model (ModelKamino): The input model container holding the time-invariant parameters of the simulation.
            state (StateKamino): The input state container holding the current state of the simulation.
            time (TimeData): The input time data container holding the current simulation time and steps.
            control (ControlKamino): The output control container where the computed control torques will be stored.
        """
        compute_jointspace_pid_control(
            model=model,
            state=state,
            time=time,
            controller=self._data,
            control=control,
        )
