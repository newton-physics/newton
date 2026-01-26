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
    """
    Enumeration of joint types supported in Newton.
    """

    PRISMATIC = 0
    """Prismatic joint: allows translation along a single axis (1 DoF)."""

    REVOLUTE = 1
    """Revolute joint: allows rotation about a single axis (1 DoF)."""

    BALL = 2
    """Ball joint: allows rotation about all three axes (3 DoF, quaternion parameterization)."""

    FIXED = 3
    """Fixed joint: locks all relative motion (0 DoF)."""

    FREE = 4
    """Free joint: allows full 6-DoF motion (translation and rotation, 7 coordinates)."""

    DISTANCE = 5
    """Distance joint: keeps two bodies at a distance within its joint limits (6 DoF, 7 coordinates)."""

    D6 = 6
    """6-DoF joint: Generic joint with up to 3 translational and 3 rotational degrees of freedom."""

    CABLE = 7
    """Cable joint: one linear (stretch) and one angular (isotropic bend/twist) DoF."""


def get_joint_dof_count(joint_type: int, num_axes: int) -> tuple[int, int]:
    """
    Returns the number of degrees of freedom (DoF) in velocity and the number of coordinates
    in position for a given joint type.

    Args:
        joint_type (int): The type of the joint (see :class:`JointType`).
        num_axes (int): The number of axes for the joint.

    Returns:
        tuple[int, int]: A tuple (dof_count, coord_count) where:
            - dof_count: Number of velocity degrees of freedom for the joint.
            - coord_count: Number of position coordinates for the joint.

    Notes:
        - For PRISMATIC and REVOLUTE joints, both values are 1 (single axis).
        - For BALL joints, dof_count is 3 (angular velocity), coord_count is 4 (quaternion).
        - For FREE and DISTANCE joints, dof_count is 6 (3 translation + 3 rotation), coord_count is 7 (3 position + 4 quaternion).
        - For FIXED joints, both values are 0.
    """
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


def get_joint_constraint_count(joint_type: int, num_axes: int) -> int:
    """
    Returns the number of velocity-level bilateral kinematic constraints for a given joint type.

    Args:
        joint_type (int): The type of the joint (see :class:`JointType`).
        num_axes (int): The number of DoF axes for the joint.

    Returns:
        int: The number of bilateral kinematic constraints for the joint.

    Notes:
        - For PRISMATIC and REVOLUTE joints, this equals 5 (single DoF axis).
        - For FREE and DISTANCE joints, `cts_count = 0` since it yields no constraints.
        - For FIXED joints, `cts_count = 6` since it fully constrains the associated bodies.
    """
    cts_count = 6 - num_axes
    if joint_type == JointType.BALL:
        cts_count = 3
    elif joint_type == JointType.FREE or joint_type == JointType.DISTANCE:
        cts_count = 0
    elif joint_type == JointType.FIXED:
        cts_count = 6
    return cts_count


# (temporary) equality constraint types
class EqType(IntEnum):
    """
    Enumeration of equality constraint types between bodies or joints.

    Note:
        This is a temporary solution and the interface may change in the future.
    """

    CONNECT = 0
    """Constrains two bodies at a point (like a ball joint)."""

    WELD = 1
    """Welds two bodies together (like a fixed joint)."""

    JOINT = 2
    """Constrains the position or angle of one joint to be a quartic polynomial of another joint (like a prismatic or revolute joint)."""


class ActuatorMode(IntEnum):
    """
    Enumeration of actuator modes for joint degrees of freedom.

    This enum manages UsdPhysics compliance by specifying whether joint_target_pos/vel
    inputs are active for a given DOF. It determines which actuators are installed when
    using solvers that require explicit actuator definitions (e.g., MuJoCo solver).

    Note:
        MuJoCo general actuators (motor, general, etc.) are handled separately via
        custom attributes with "mujoco:actuator" frequency and control.mujoco.ctrl,
        not through this enum.
    """

    NONE = 0
    """No actuators are installed for this DOF. The joint is passive/unactuated."""

    POSITION = 1
    """Only a position actuator is installed for this DOF. Tracks joint_target_pos."""

    VELOCITY = 2
    """Only a velocity actuator is installed for this DOF. Tracks joint_target_vel."""

    POSITION_VELOCITY = 3
    """Both position and velocity actuators are installed. Tracks both joint_target_pos and joint_target_vel."""

    EFFORT = 4
    """A drive is applied but no gains are configured. No MuJoCo actuator is created for this DOF.
    The user is expected to supply force via joint_f."""


__all__ = [
    "ActuatorMode",
    "EqType",
    "JointType",
    "get_joint_constraint_count",
    "get_joint_dof_count",
]
