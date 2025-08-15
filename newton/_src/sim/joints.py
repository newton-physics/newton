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


# Types of joints linking rigid bodies
class JointType(IntEnum):
    PRISMATIC = 0
    REVOLUTE = 1
    BALL = 2
    FIXED = 3
    FREE = 4
    DISTANCE = 5
    D6 = 6


def get_joint_dof_count(joint_type: int, num_axes: int) -> tuple[int, int]:
    """Return the number of degrees of freedom in position and velocity for a given joint type."""
    dof_count = num_axes
    coord_count = num_axes
    if joint_type == JointType.BALL:
        dof_count = 3
        coord_count = 4
    elif joint_type == JointType.FREE or joint_type == JointType.DISTANCE:
        dof_count = 6
        coord_count = 7
    elif joint_type == JointType.FIXED:
        dof_count = 0
        coord_count = 0
    return dof_count, coord_count


class JointMode(IntEnum):
    NONE = 0
    TARGET_POSITION = 1
    TARGET_VELOCITY = 2


# (temporary) equality constraint types
class EqType(IntEnum):
    CONNECT = 0
    WELD = 1
    JOINT = 2


__all__ = [
    "EqType",
    "JointMode",
    "JointType",
    "get_joint_dof_count",
]
