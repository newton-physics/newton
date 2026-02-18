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

    def dof_count(self, num_axes: int) -> tuple[int, int]:
        """
        Returns the number of degrees of freedom (DoF) in velocity and the number of coordinates
        in position for this joint type.

        Args:
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
        if self == JointType.BALL:
            dof_count = 3
            coord_count = 4
        elif self == JointType.FREE or self == JointType.DISTANCE:
            dof_count = 6
            coord_count = 7
        elif self == JointType.FIXED:
            dof_count = 0
            coord_count = 0
        return dof_count, coord_count

    def constraint_count(self, num_axes: int) -> int:
        """
        Returns the number of velocity-level bilateral kinematic constraints for this joint type.

        Args:
            num_axes (int): The number of DoF axes for the joint.

        Returns:
            int: The number of bilateral kinematic constraints for the joint.

        Notes:
            - For PRISMATIC and REVOLUTE joints, this equals 5 (single DoF axis).
            - For FREE and DISTANCE joints, `cts_count = 0` since it yields no constraints.
            - For FIXED joints, `cts_count = 6` since it fully constrains the associated bodies.
        """
        cts_count = 6 - num_axes
        if self == JointType.BALL:
            cts_count = 3
        elif self == JointType.FREE or self == JointType.DISTANCE:
            cts_count = 0
        elif self == JointType.FIXED:
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

    @staticmethod
    def from_gains(
        target_ke: float,
        target_kd: float,
        force_position_velocity: bool = False,
        has_drive: bool = False,
    ) -> "ActuatorMode":
        """Infer actuator mode from position and velocity gains.

        Args:
            target_ke: Position gain (stiffness).
            target_kd: Velocity gain (damping).
            force_position_velocity: If True and both gains are non-zero,
                forces POSITION_VELOCITY mode instead of just POSITION.
            has_drive: If True, a drive/actuator is applied to the joint.
                When True but both gains are 0, returns EFFORT mode.
                When False, returns NONE regardless of gains.

        Returns:
            The inferred ActuatorMode based on which gains are non-zero:
            - NONE: No drive applied
            - EFFORT: Drive applied but both gains are 0 (direct torque control)
            - POSITION: Only position gain is non-zero
            - VELOCITY: Only velocity gain is non-zero
            - POSITION_VELOCITY: Both gains non-zero (or forced)
        """
        if not has_drive:
            return ActuatorMode.NONE

        if force_position_velocity and (target_ke != 0.0 and target_kd != 0.0):
            return ActuatorMode.POSITION_VELOCITY
        elif target_ke != 0.0:
            return ActuatorMode.POSITION
        elif target_kd != 0.0:
            return ActuatorMode.VELOCITY
        else:
            return ActuatorMode.EFFORT


class KinematicMode(IntEnum):
    """Kinematic mode for a joint.

    Kinematic joints have very high armature on all DOFs, making them effectively
    immovable by external forces while still participating in velocity-level contact
    resolution. The joint's motion is driven by ``control.kinematic_target``.

    The ``kinematic_target`` array on :class:`~newton.Control` is indexed by
    ``joint_q_start`` offsets for both modes:

    - ``VELOCITY``: target values are per-DOF velocities [m/s or rad/s] written at
      ``q_start`` (extra coordinate slots unused for free/ball joints).
    - ``POSITION``: target values are per-coordinate positions [m or rad] in the same
      layout as ``joint_q``. The solver computes the required velocity internally.

    Attributes:
        NONE: Normal dynamic joint. No kinematic control.
        VELOCITY: Velocity-controlled kinematic joint. Target is per-DOF velocity [m/s or rad/s].
        POSITION: Position-controlled kinematic joint. Target is per-coordinate position [m or rad],
            velocity is computed as ``(target - current) / dt``.
    """

    NONE = 0
    """Normal dynamic joint. No kinematic control."""

    VELOCITY = 1
    """Velocity-controlled kinematic joint. Target is per-DOF velocity [m/s or rad/s]."""

    POSITION = 2
    """Position-controlled kinematic joint. Target is per-coordinate position [m or rad],
    velocity is computed as ``(target - current) / dt``."""


__all__ = [
    "ActuatorMode",
    "EqType",
    "JointType",
    "KinematicMode",
]
