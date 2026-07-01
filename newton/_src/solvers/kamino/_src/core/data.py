# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Defines the Kamino-specific data containers to hold time-varying simulation data."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import warp as wp

from .bodies import RigidBodiesData
from .control import ControlKamino
from .geometry import GeometriesData
from .joints import JointsData
from .state import StateKamino
from .time import TimeData

if TYPE_CHECKING:
    from .model import ModelKamino

###
# Module interface
###

__all__ = [
    "DataKamino",
    "DataKaminoInfo",
]


###
# Types
###


@dataclass
class DataKaminoInfo:
    """
    A container to hold the time-varying information about the set of active constraints.
    """

    ###
    # Total Constraints
    ###

    num_total_cts: wp.array[wp.int32] | None = None
    """
    The total number of active constraints.
    Shape of ``(num_worlds,)``.
    """

    ###
    # Limits
    ###

    num_limits: wp.array[wp.int32] | None = None
    """
    The number of active limits in each world.
    Shape of ``(num_worlds,)``.
    """

    num_limit_cts: wp.array[wp.int32] | None = None
    """
    The number of active limit constraints.
    Shape of ``(num_worlds,)``.
    """

    limit_cts_group_offset: wp.array[wp.int32] | None = None
    """
    The index offset of the limit constraints group within the constraints block of each world.
    Shape of ``(num_worlds,)``.
    """

    ###
    # Contacts
    ###

    num_contacts: wp.array[wp.int32] | None = None
    """
    The number of active contacts in each world.
    Shape of ``(num_worlds,)``.
    """

    num_contact_cts: wp.array[wp.int32] | None = None
    """
    The number of active contact constraints.
    Shape of ``(num_worlds,)``.
    """

    contact_cts_group_offset: wp.array[wp.int32] | None = None
    """
    The index offset of the contact constraints group within the constraints block of each world.
    Shape of ``(num_worlds,)``.
    """

    ###
    # Properties
    ###

    @property
    def device(self) -> wp.DeviceLike:
        """The device on which data is allocated."""
        if self.num_total_cts is None:
            raise RuntimeError("DataKaminoInfo has not been finalized.")
        return self.num_total_cts.device


@dataclass
class DataKamino:
    """
    A container to hold the time-varying data of the model entities.

    It includes all model-specific intermediate quantities used throughout the simulation, as needed
    to update the state of rigid bodies, joints, geometries, active constraints and time-keeping.
    """

    info: DataKaminoInfo | None = None
    """The info container holding information about the set of active constraints."""

    time: TimeData | None = None
    """Time-varying time-keeping data, including the current simulation step and time."""

    bodies: RigidBodiesData | None = None
    """
    Time-varying data of all rigid bodies in the model: poses, twists,
    wrenches, and moments of inertia computed in world coordinates.
    """

    joints: JointsData | None = None
    """
    Time-varying data of joints in the model: joint frames computed in world coordinates,
    constraint residuals and reactions, and generalized (DoF) quantities.
    """

    geoms: GeometriesData | None = None
    """Time-varying data of geometries in the model: poses computed in world coordinates."""

    ###
    # Properties
    ###

    @property
    def device(self) -> wp.DeviceLike:
        """The device on which data is allocated."""
        if self.info is None:
            raise RuntimeError("DataKamino has not been finalized.")
        return self.info.device

    ###
    # Operations
    ###

    def copy_body_state_from(self, state: StateKamino) -> None:
        """
        Copies the rigid bodies data from the given :class:`StateKamino`.

        This operation copies:
        - Body poses
        - Body twists

        Args:
            state: The state container holding time-varying state of the simulation.
        """
        # Ensure bodies data has been allocated
        if self.bodies is None:
            raise RuntimeError("DataKamino.bodies is not finalized.")

        # Copy rigid bodies data from the source state container
        wp.copy(self.bodies.q_i, state.q_i)
        wp.copy(self.bodies.u_i, state.u_i)

    def copy_body_state_to(self, state: StateKamino) -> None:
        """
        Copies the rigid bodies data to the given :class:`StateKamino`.

        This operation copies:
        - Body poses
        - Body twists
        - Body wrenches

        Args:
            state: The state container holding time-varying state of the simulation.
        """
        # Ensure bodies data has been allocated
        if self.bodies is None:
            raise RuntimeError("DataKamino.bodies is not finalized.")

        # Copy rigid bodies data to the target state container
        wp.copy(state.q_i, self.bodies.q_i)
        wp.copy(state.u_i, self.bodies.u_i)
        wp.copy(state.w_i, self.bodies.w_i)

    def copy_joint_state_from(self, state: StateKamino) -> None:
        """
        Copies the joint state data from the given :class:`StateKamino`.

        This operation copies:
        - Joint coordinates
        - Joint velocities

        Args:
            state: The state container holding time-varying state of the simulation.
        """
        # Ensure joints data has been allocated
        if self.joints is None:
            raise RuntimeError("DataKamino.joints is not finalized.")

        # Copy joint data from the source state container
        wp.copy(self.joints.q_j, state.q_j)
        wp.copy(self.joints.q_j_p, state.q_j_p)
        wp.copy(self.joints.dq_j, state.dq_j)

    def copy_joint_state_to(self, state: StateKamino) -> None:
        """
        Copies the joint state data to the given :class:`StateKamino`.

        This operation copies:
        - Joint coordinates
        - Joint velocities
        - Joint constraint reactions

        Args:
            state: The state container holding time-varying state of the simulation.
        """
        # Ensure joints data has been allocated
        if self.joints is None:
            raise RuntimeError("DataKamino.joints is not finalized.")

        # Copy joint data to the target state container
        wp.copy(state.q_j, self.joints.q_j)
        wp.copy(state.q_j_p, self.joints.q_j_p)
        wp.copy(state.dq_j, self.joints.dq_j)
        wp.copy(state.lambda_j, self.joints.lambda_j)

    def copy_joint_control_from(self, control: ControlKamino, model: ModelKamino | None = None) -> None:
        """
        Copies the joint control inputs from the given :class:`ControlKamino`.

        This operation copies:
        - Joint direct efforts
        - Joint position targets
        - Joint velocity targets
        - Joint feedforward efforts

        Any missing control inputs will be set to zero, or its initial value if
        the :class:`ModelKamino` is provided.

        Args:
            control: The control container holding the joint control inputs.
            model: The model providing default values for any missing control
                inputs.

        """
        # Ensure joints data has been allocated
        if self.joints is None:
            raise RuntimeError("DataKamino.joints is not finalized.")

        # Copy joint control inputs from the source control container, with
        # fallback options of copying the defaults from the model or zeroing them
        if control.tau_j is not None:
            wp.copy(self.joints.tau_j, control.tau_j)
        else:
            self.joints.tau_j.zero_()
        if control.q_j_ref is not None:
            wp.copy(self.joints.q_j_ref, control.q_j_ref)
        elif model is not None:
            wp.copy(self.joints.q_j_ref, model.joints.q_j_0)
        else:
            self.joints.q_j_ref.zero_()
        if control.dq_j_ref is not None:
            wp.copy(self.joints.dq_j_ref, control.dq_j_ref)
        elif model is not None:
            wp.copy(self.joints.dq_j_ref, model.joints.dq_j_0)
        else:
            self.joints.dq_j_ref.zero_()
        if control.tau_j_ref is not None:
            wp.copy(self.joints.tau_j_ref, control.tau_j_ref)
        else:
            self.joints.tau_j_ref.zero()
