# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Defines the control container of Kamino."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import warp as wp

from .....sim.control import Control
from .conversions import convert_target_coords_to_target_dofs, convert_target_dofs_to_target_coords
from .types import float32

if TYPE_CHECKING:
    from .model import ModelKamino

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

    tau_j: wp.array | None = None
    """
    Array of generalized joint actuation forces.\n
    Shape is ``(sum(d_j),)`` and dtype is :class:`float32`,\n
    where ``d_j`` is the number of DoFs of each joint ``j``.
    """

    q_j_ref: wp.array | None = None
    """
    Array of reference generalized joint coordinates for implicit PD control.\n
    Shape of ``(sum(c_j),)`` and type :class:`float`,
    where ``c_j`` is the number of coordinates of joint ``j``.
    """

    dq_j_ref: wp.array | None = None
    """
    Array of reference generalized joint velocities for implicit PD control.\n
    Shape of ``(sum(d_j),)`` and type :class:`float`,
    where ``d_j`` is the number of DoFs of joint ``j``.
    """

    tau_j_ref: wp.array | None = None
    """
    Array of reference feed-forward generalized joint forces for implicit PD control.\n
    Shape of ``(sum(d_j),)`` and type :class:`float`,
    where ``d_j`` is the number of DoFs of joint ``j``.
    """

    ###
    # Internal state
    ###

    _needs_coord_conversion: bool = False
    """Whether dofs-to-coords conversion is required for this model."""

    _q_j_ref_coords_space: wp.array | None = None
    """Owned coords-space reference buffer used when ``dofs != coords``."""

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

    def finalize(self, model: ModelKamino, device: wp.DeviceLike = None) -> None:
        """
        Allocates any required internal buffer to interface with a :class:`newton.Control`.

        More specifically, an internal buffer is allocated for models for which joint coordinates
        and DoFs differ (i.e. models with spherical or free joints); otherwise, no allocation
        is performed.

        Args:
            model: The Kamino model describing the system.
            device: Optional device to create the state on. If not specified, the model's device is used.
        """
        if device is None:
            device = model.device

        if model.size.sum_of_num_joint_dofs != model.size.sum_of_num_joint_coords:
            self._needs_coord_conversion = True
            self._q_j_ref_coords_space = wp.zeros(
                shape=model.size.sum_of_num_joint_coords,
                dtype=float32,
                device=device,
            )
        else:
            self._needs_coord_conversion = False
            self._q_j_ref_coords_space = None

    def from_newton(self, control: Control, model: ModelKamino) -> None:
        """
        Reads a source :class:`newton.Control` object into this :class:`ControlKamino` instance.
        Arrays are simply aliased whenever possible, but in some cases a necessary conversion from joint
        target DoFs to coords is performed.

        Args:
            control: The source :class:`newton.Control` object to be interfaced.
            model: The source Kamino model holding the time-invariant description of the system.
        """
        self.tau_j = control.joint_f
        self.tau_j_ref = control.joint_act
        self.dq_j_ref = control.joint_target_vel
        if self._needs_coord_conversion:
            self.q_j_ref = self._q_j_ref_coords_space
            convert_target_dofs_to_target_coords(
                joint_target_dofs=control.joint_target_pos,
                joint_target_coords=self.q_j_ref,
                model=model,
            )
        else:
            self.q_j_ref = control.joint_target_pos

    def to_newton(self, control: Control, model: ModelKamino) -> None:
        """
        Writes this :class:`ControlKamino` instance into a destination :class:`newton.Control` object.
        Arrays are simply aliased whenever possible, but in some cases a necessary conversion from joint
        target coords to DoFs is performed.

        Args:
            control: The destination :class:`newton.Control` object to be interfaced.
            model: The source Kamino model holding the time-invariant description of the system.
        """
        control.joint_f = self.tau_j
        control.joint_act = self.tau_j_ref
        control.joint_target_vel = self.dq_j_ref
        if self._needs_coord_conversion:
            convert_target_coords_to_target_dofs(
                joint_target_coords=self.q_j_ref,
                joint_target_dofs=control.joint_target_pos,
                model=model,
            )
        else:
            control.joint_target_pos = self.q_j_ref
