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

    @staticmethod
    def from_newton(control: Control) -> ControlKamino:
        """
        Constructs a :class:`kamino.ControlKamino` object from a :class:`newton.Control` object.

        This operation serves only as an adaptor-like constructor to interface a
        :class:`newton.Control`, effectively creating an alias without copying data.

        Args:
            control: The source :class:`newton.Control` object to be adapted.
        """
        return ControlKamino(
            tau_j=control.joint_f,
            tau_j_ref=control.joint_act,
            q_j_ref=control.joint_target_pos,
            dq_j_ref=control.joint_target_vel,
        )

    @staticmethod
    def to_newton(control: ControlKamino) -> Control:
        """
        Constructs a :class:`newton.Control` object from a :class:`kamino.ControlKamino` object.

        This operation serves only as an adaptor-like constructor to interface a
        :class:`kamino.ControlKamino`, effectively creating an alias without copying data.

        Args:
            control: The source :class:`kamino.ControlKamino` object to be adapted.
        """
        control_newton = Control()
        control_newton.joint_f = control.tau_j
        control_newton.joint_act = control.tau_j_ref
        control_newton.joint_target_pos = control.q_j_ref
        control_newton.joint_target_vel = control.dq_j_ref
        return control_newton
