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

from __future__ import annotations

import warp as wp


class Control:
    """Time-varying control data for a :class:`Model`.

    Time-varying control data includes joint torques, control inputs, muscle activations,
    and activation forces for triangle and tetrahedral elements.

    The exact attributes depend on the contents of the model. Control objects
    should generally be created using the :func:`newton.Model.control()` function.
    """

    def __init__(self):
        self.joint_f: wp.array | None = None
        """
        Array of generalized joint forces with shape ``(joint_dof_count,)`` and type ``float``.

        The degrees of freedom for free joints are included in this array and have the same
        convention as the :attr:`newton.State.body_f` array where the 6D wrench is defined as
        ``(f_x, f_y, f_z, t_x, t_y, t_z)``, where ``f_x``, ``f_y``, and ``f_z`` are the components
        of the force vector (linear) and ``t_x``, ``t_y``, and ``t_z`` are the
        components of the torque vector (angular). Both linear forces and angular torques applied to free joints are
        applied in world frame (same as :attr:`newton.State.body_f`).
        """
        self.joint_target_pos: wp.array | None = None
        """Per-DOF position targets, shape ``(joint_dof_count,)``, type ``float`` (optional)."""

        self.joint_target_vel: wp.array | None = None
        """Per-DOF velocity targets, shape ``(joint_dof_count,)``, type ``float`` (optional)."""

        self.tri_activations: wp.array | None = None
        """Array of triangle element activations with shape ``(tri_count,)`` and type ``float``."""

        self.tet_activations: wp.array | None = None
        """Array of tetrahedral element activations with shape with shape ``(tet_count,) and type ``float``."""

        self.muscle_activations: wp.array | None = None
        """
        Array of muscle activations with shape ``(muscle_count,)`` and type ``float``.

        .. note::
            Support for muscle dynamics is not yet implemented.
        """

    def clear(self) -> None:
        """Reset the control inputs to zero."""

        if self.joint_f is not None:
            self.joint_f.zero_()
        if self.tri_activations is not None:
            self.tri_activations.zero_()
        if self.tet_activations is not None:
            self.tet_activations.zero_()
        if self.muscle_activations is not None:
            self.muscle_activations.zero_()
        if self.joint_target_pos is not None:
            self.joint_target_pos.zero_()
        if self.joint_target_vel is not None:
            self.joint_target_vel.zero_()
        self._clear_namespaced_arrays()

    def _clear_namespaced_arrays(self) -> None:
        """Clear all wp.array attributes in namespaced containers (e.g., control.mujoco.ctrl)."""
        for attr in self.__dict__.values():
            if hasattr(attr, "_name"):
                for value in attr.__dict__.values():
                    if isinstance(value, wp.array):
                        value.zero_()
