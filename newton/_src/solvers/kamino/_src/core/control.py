# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Defines the control container of Kamino."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import warp as wp

from .....sim.control import Control
from .conversions import convert_target_coords_to_target_dofs, convert_target_dofs_to_target_coords

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
    # Internal state
    ###

    _needs_coord_conversion: bool = False
    """Whether dofs-to-coords conversion is required for this model."""

    _q_j_ref_coords_space: wp.array[wp.float32] | None = None
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

    def finalize(self, model: ModelKamino, device: wp.DeviceLike | None = None) -> None:
        """Allocate the coord-space side buffer used to interface with a
        :class:`newton.Control`.

        The buffer is allocated only when the wrapped Newton model was built
        under :data:`newton.use_coord_layout_targets` ``False`` *and* contains
        spherical or free joints — i.e. when ``Control.joint_target_q`` is
        DOF-shaped and needs Euler→quat conversion. Otherwise no allocation
        happens. The layout is read from
        :attr:`ModelKamino.use_coord_layout_targets` (snapshot) so toggling
        the global flag after ``finalize`` can't desynchronize.

        Args:
            model: The Kamino model describing the system.
            device: Optional allocation device. Defaults to the model's device.
        """
        if device is None:
            device = model.device

        self._needs_coord_conversion = (
            not model.use_coord_layout_targets
            and model.size.sum_of_num_joint_dofs != model.size.sum_of_num_joint_coords
        )
        self._q_j_ref_coords_space = (
            wp.zeros(shape=model.size.sum_of_num_joint_coords, dtype=wp.float32, device=device)
            if self._needs_coord_conversion
            else None
        )

    def from_newton(self, control: Control, model: ModelKamino) -> None:
        """Adopt arrays from a :class:`newton.Control`. Aliases directly when
        possible; runs Euler→quat conversion on ``joint_target_q`` only if the
        wrapped Newton model is DOF-layout (flag=False) and has spherical or
        free joints.

        Args:
            control: Source :class:`newton.Control` to read from.
            model: The Kamino model holding the system description.
        """
        self.tau_j = control.joint_f
        self.dq_j_ref = control.joint_target_qd
        if self._needs_coord_conversion:
            self.q_j_ref = self._q_j_ref_coords_space
            convert_target_dofs_to_target_coords(
                joint_target_dofs=control.joint_target_q,
                joint_target_coords=self.q_j_ref,
                model=model,
            )
        else:
            self.q_j_ref = control.joint_target_q

    def to_newton(self, control: Control, model: ModelKamino) -> None:
        """Write back into a :class:`newton.Control`. Aliases directly when
        possible; runs quat→Euler conversion only if the wrapped Newton model
        is DOF-layout (flag=False) and has spherical or free joints.

        Args:
            control: Destination :class:`newton.Control` to write into.
            model: The Kamino model holding the system description.
        """
        control.joint_f = self.tau_j
        control.joint_target_qd = self.dq_j_ref
        if self._needs_coord_conversion:
            convert_target_coords_to_target_dofs(
                joint_target_coords=self.q_j_ref,
                joint_target_dofs=control.joint_target_q,
                model=model,
            )
        else:
            control.joint_target_q = self.q_j_ref
