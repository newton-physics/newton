# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Defines the Kamino-specific data containers to hold time-varying simulation data."""

from dataclasses import dataclass

import warp as wp

from .bodies import RigidBodiesData
from .control import ControlKamino
from .geometry import GeometriesData
from .joints import JointsData
from .state import StateKamino
from .time import TimeData

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

    num_total_cts: wp.array | None = None
    """
    The total number of active constraints.\n
    Shape of ``(num_worlds,)`` and type :class:`int32`.
    """

    ###
    # Limits
    ###

    num_limits: wp.array | None = None
    """
    The number of active limits in each world.\n
    Shape of ``(num_worlds,)`` and type :class:`int32`.
    """

    num_limit_cts: wp.array | None = None
    """
    The number of active limit constraints.\n
    Shape of ``(num_worlds,)`` and type :class:`int32`.
    """

    limit_cts_group_offset: wp.array | None = None
    """
    The index offset of the limit constraints group within the constraints block of each world.\n
    Shape of ``(num_worlds,)`` and type :class:`int32`.
    """

    ###
    # Contacts
    ###

    num_contacts: wp.array | None = None
    """
    The number of active contacts in each world.\n
    Shape of ``(num_worlds,)`` and type :class:`int32`.
    """

    num_contact_cts: wp.array | None = None
    """
    The number of active contact constraints.\n
    Shape of ``(num_worlds,)`` and type :class:`int32`.
    """

    contact_cts_group_offset: wp.array | None = None
    """
    The index offset of the contact constraints group within the constraints block of each world.\n
    Shape of ``(num_worlds,)`` and type :class:`int32`.
    """


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
    # Operations
    ###

    def copy_body_state_from(self, state: StateKamino) -> None:
        """
        Copies the rigid bodies data from the given :class:`StateKamino`.

        This operation copies:
        - Body poses
        - Body twists

        Args:
            state (StateKamino):
                The state container holding time-varying state of the simulation.
        """
        # Ensure bodies data has been allocated
        if self.bodies is None:
            raise RuntimeError("DataKamino.bodies is not finalized.")

        # Update rigid bodies data from the model state
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
            state (StateKamino):
                The state container holding time-varying state of the simulation.
        """
        # Ensure bodies data has been allocated
        if self.bodies is None:
            raise RuntimeError("DataKamino.bodies is not finalized.")

        # Update rigid bodies data from the model state
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
            state (StateKamino):
                The state container holding time-varying state of the simulation.
        """
        # Ensure joints data has been allocated
        if self.joints is None:
            raise RuntimeError("DataKamino.joints is not finalized.")

        # Update joint data from the model state
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
            state (StateKamino):
                The state container holding time-varying state of the simulation.
        """
        # Ensure joints data has been allocated
        if self.joints is None:
            raise RuntimeError("DataKamino.joints is not finalized.")

        # Update joint data from the model state
        wp.copy(state.q_j, self.joints.q_j)
        wp.copy(state.q_j_p, self.joints.q_j_p)
        wp.copy(state.dq_j, self.joints.dq_j)
        wp.copy(state.lambda_j, self.joints.lambda_j)

    def copy_joint_control_from(self, control: ControlKamino) -> None:
        """
        Copies the joint control inputs from the given :class:`ControlKamino`.

        This operation copies:
        - Joint direct efforts
        - Joint position targets
        - Joint velocity targets
        - Joint feedforward efforts

        Args:
            control (ControlKamino):
                The control container holding the joint control inputs.
        """
        # Ensure joints data has been allocated
        if self.joints is None:
            raise RuntimeError("DataKamino.joints is not finalized.")

        # Update joint data from the control inputs
        wp.copy(self.joints.tau_j, control.tau_j)
        wp.copy(self.joints.q_j_ref, control.q_j_ref)
        wp.copy(self.joints.dq_j_ref, control.dq_j_ref)
        wp.copy(self.joints.tau_j_ref, control.tau_j_ref)
