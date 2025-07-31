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

# Types of joints linking rigid bodies
JOINT_PRISMATIC = wp.constant(0)
JOINT_REVOLUTE = wp.constant(1)
JOINT_BALL = wp.constant(2)
JOINT_FIXED = wp.constant(3)
JOINT_FREE = wp.constant(4)
JOINT_DISTANCE = wp.constant(5)
JOINT_D6 = wp.constant(6)


def get_joint_dof_count(joint_type: int, num_axes: int) -> tuple[int, int]:
    """Return the number of degrees of freedom in position and velocity for a given joint type."""
    dof_count = num_axes
    coord_count = num_axes
    if joint_type == JOINT_BALL:
        dof_count = 3
        coord_count = 4
    elif joint_type == JOINT_FREE or joint_type == JOINT_DISTANCE:
        dof_count = 6
        coord_count = 7
    elif joint_type == JOINT_FIXED:
        dof_count = 0
        coord_count = 0
    return dof_count, coord_count


# Joint axis control mode types
JOINT_MODE_NONE = wp.constant(0)
JOINT_MODE_TARGET_POSITION = wp.constant(1)
JOINT_MODE_TARGET_VELOCITY = wp.constant(2)

# (temporary) equality constraint types
EQ_CONNECT = wp.constant(0)
EQ_WELD = wp.constant(1)
EQ_JOINT = wp.constant(2)

__all__ = [
    "EQ_CONNECT",
    "EQ_JOINT",
    "EQ_WELD",
    "JOINT_BALL",
    "JOINT_D6",
    "JOINT_DISTANCE",
    "JOINT_FIXED",
    "JOINT_FREE",
    "JOINT_MODE_NONE",
    "JOINT_MODE_TARGET_POSITION",
    "JOINT_MODE_TARGET_VELOCITY",
    "JOINT_PRISMATIC",
    "JOINT_REVOLUTE",
    "get_joint_dof_count",
]
