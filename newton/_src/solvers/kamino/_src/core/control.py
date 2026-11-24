# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Defines the control container of Kamino."""

from __future__ import annotations

from dataclasses import dataclass

import warp as wp

from .....sim.control import Control

###
# Types
###


@dataclass
class ControlKamino:
    """
    Time-varying control data for a :class:`ModelKamino`.

    Time-varying control data currently consists of generalized joint actuation forces, with
    the intention that external actuator models or controllers will populate these attributes.

    The exact attributes depend on the contents of the model. ControlKamino objects
    should generally be created using the :func:`kamino.ModelKamino.control()` function.

    We adopt the following notational conventions for the control attributes:
    - Generalized joint actuation forces are denoted by ``tau``
    - Subscripts ``_j`` denote joint-indexed quantities, e.g. :attr:`tau_j`.
    """

    ###
    # Attributes
    ###

    tau_j: wp.array[wp.float32] | None = None
    """
    Array of generalized joint actuation forces.
    Shape of ``(sum_of_num_joint_dofs,)``.
    """

    q_j_ref: wp.array[wp.float32] | None = None
    """
    Array of reference generalized joint coordinates for implicit PD control.
    Shape of ``(sum_of_num_joint_coords,)``.
    """

    dq_j_ref: wp.array[wp.float32] | None = None
    """
    Array of reference generalized joint velocities for implicit PD control.
    Shape of ``(sum_of_num_joint_dofs,)``.
    """

    tau_j_ref: wp.array[wp.float32] | None = None
    """
    Array of reference feed-forward generalized joint forces for implicit PD control.
    Shape of ``(sum_of_num_joint_dofs,)``.
    """

    ###
    # Properties
    ###

    @property
    def device(self) -> wp.DeviceLike:
        """The device used for allocations and execution."""
        if self.tau_j is None:
            raise RuntimeError("ControlKamino data is not allocated.")
        return self.tau_j.device

    ###
    # Operations
    ###

    def copy_to(self, other: ControlKamino) -> None:
        """
        Copies the ControlKamino data to another ControlKamino object.

        Args:
            other: The target ControlKamino object to copy data into.
        """
        if self.tau_j is None or other.tau_j is None:
            raise ValueError("Error copying from/to uninitialized ControlKamino")
        wp.copy(other.tau_j, self.tau_j)

    def copy_from(self, other: ControlKamino) -> None:
        """
        Copies the ControlKamino data from another ControlKamino object.

        Args:
            other: The source ControlKamino object to copy data from.
        """
        if self.tau_j is None or other.tau_j is None:
            raise ValueError("Error copying from/to uninitialized ControlKamino")
        wp.copy(self.tau_j, other.tau_j)

    def from_newton(self, control: Control) -> None:
        """Alias arrays from a :class:`newton.Control`.

        Args:
            control: Source :class:`newton.Control` to read from.
        """
        self.tau_j = control.joint_f
        self.dq_j_ref = control.joint_target_qd
        self.q_j_ref = control.joint_target_q

    def to_newton(self, control: Control) -> None:
        """Alias arrays into a :class:`newton.Control`.

        Args:
            control: Destination :class:`newton.Control` to write into.
        """
        control.joint_f = self.tau_j
        control.joint_target_qd = self.dq_j_ref
        control.joint_target_q = self.q_j_ref
