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

from enum import IntEnum


class CtrlSource(IntEnum):
    """Control source for MuJoCo actuators.

    Determines where an actuator gets its control input from:
    - JOINT_TARGET: Maps from Newton's joint_target_pos/vel arrays
    - CTRL_DIRECT: Uses control.mujoco.ctrl directly (for MuJoCo-native control)
    """

    JOINT_TARGET = 0
    CTRL_DIRECT = 1


class CtrlType(IntEnum):
    """Control type for MuJoCo actuators.

    For JOINT_TARGET mode, determines which target array to read from:
    - POSITION: Maps from joint_target_pos, syncs gains from joint_target_ke
    - VELOCITY: Maps from joint_target_vel, syncs gains from joint_target_kd
    - GENERAL: Used with CTRL_DIRECT mode for motor/general actuators
    """

    POSITION = 0
    VELOCITY = 1
    GENERAL = 2


from .solver_mujoco import SolverMuJoCo  # noqa: E402

__all__ = [
    "CtrlSource",
    "CtrlType",
    "SolverMuJoCo",
]
