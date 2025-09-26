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

"""Solver flags."""

from enum import IntEnum


# model update flags - used for solver.notify_model_update()
class SolverNotifyFlags(IntEnum):
    """
    Flags indicating which parts of the model have been updated and require the solver to be notified.

    These flags are used with `solver.notify_model_update()` to specify which properties have changed,
    allowing the solver to efficiently update only the necessary components.
    """

    JOINT_PROPERTIES = 1 << 0
    """Indicates joint property updates: joint_q, joint_X_p, joint_X_c."""

    JOINT_DOF_PROPERTIES = 1 << 1
    """Indicates joint DOF property updates: joint_target, joint_target_ke, joint_target_kd, joint_dof_mode, joint_limit_upper, joint_limit_lower, joint_limit_ke, joint_limit_kd, joint_qd, joint_f, joint_armature."""

    BODY_PROPERTIES = 1 << 2
    """Indicates body property updates: body_q, body_qd."""

    BODY_INERTIAL_PROPERTIES = 1 << 3
    """Indicates body inertial property updates: body_com, body_inertia, body_inv_inertia, body_mass, body_inv_mass."""

    SHAPE_PROPERTIES = 1 << 4
    """Indicates shape property updates: shape_transform, shape geometry and material properties"""

    MODEL_PROPERTIES = 1 << 5
    """Indicates model property updates: gravity and other global parameters."""

    ALL = (
        JOINT_PROPERTIES
        | JOINT_DOF_PROPERTIES
        | BODY_PROPERTIES
        | BODY_INERTIAL_PROPERTIES
        | SHAPE_PROPERTIES
        | MODEL_PROPERTIES
    )
    """Indicates all property updates."""


__all__ = [
    "SolverNotifyFlags",
]
