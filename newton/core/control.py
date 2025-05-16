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

import warp as wp


class Control:
    """Time-varying control data for a :class:`Model`.

    Time-varying control data includes joint control inputs, muscle activations,
    and activation forces for triangle and tetrahedral elements.

    The exact attributes depend on the contents of the model. Control objects
    should generally be created using the :func:`Model.control()` function.
    """

    def __init__(self):
        self.joint_f: wp.array | None = None
        """Array of generalized joint forces with shape ``(joint_dof_count,)`` and type ``float``."""

        self.joint_target: wp.array | None = None
        """Array of joint targets with shape ``(joint_axis_count,)`` and type ``float``."""

        self.tri_activations: wp.array | None = None
        """Array of triangle element activations with shape ``(tri_count,)`` and type ``float``."""

        self.tet_activations: wp.array | None = None
        """Array of tetrahedral element activations with shape with shape ``(tet_count,) and type ``float``."""

        self.muscle_activations: wp.array | None = None
        """Array of muscle activations with shape ``(muscle_count,)`` and type ``float``."""

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
