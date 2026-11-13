# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""MuJoCo solver enums."""

from enum import IntEnum
from functools import cache


class EqType(IntEnum):
    """MuJoCo equality constraint type."""

    CONNECT = 0
    """Constrains two bodies at a point (like a ball joint)."""

    WELD = 1
    """Welds two bodies together (like a fixed joint)."""

    JOINT = 2
    """Constrains one scalar joint coordinate to a quartic polynomial of another."""

    TENDON = 3
    """Constrains one tendon length to a quartic polynomial of another, or to a constant offset."""


# Mirrors of MuJoCo's mjtBias/mjtDyn/mjtGain, duplicated so MJCF and USD import do not
# need MuJoCo. Types Newton does not support are listed too, so Newton's ordinals track
# MuJoCo's. TestActuatorTypes pins these against mujoco.mjt*.
class _ActuatorBiasType(IntEnum):
    NONE = 0
    AFFINE = 1
    MUSCLE = 2
    DCMOTOR = 3
    SO3 = 4
    USER = 5


class _ActuatorDynamicsType(IntEnum):
    NONE = 0
    INTEGRATOR = 1
    FILTER = 2
    FILTER_EXACT = 3
    MUSCLE = 4
    DCMOTOR = 5
    PID = 6
    USER = 7


class _ActuatorGainType(IntEnum):
    FIXED = 0
    AFFINE = 1
    MUSCLE = 2
    DCMOTOR = 3
    SO3 = 4
    PID = 5
    USER = 6


class CtrlSource(IntEnum):
    """Control source for MuJoCo actuators.

    Determines where an actuator gets its control input from:

    - :attr:`JOINT_TARGET`: Maps from Newton's :attr:`~newton.Control.joint_target_q`/:attr:`~newton.Control.joint_target_qd` arrays.
    - :attr:`CTRL_DIRECT`: Uses ``control.mujoco.ctrl`` directly (for MuJoCo-native control)
    """

    JOINT_TARGET = 0
    CTRL_DIRECT = 1


class CtrlType(IntEnum):
    """Control type for MuJoCo actuators.

    For :attr:`~newton.solvers.SolverMuJoCo.CtrlSource.JOINT_TARGET` mode, determines which target array to read from:

    - :attr:`POSITION`: Maps from :attr:`~newton.Control.joint_target_q`, syncs gains from
      :attr:`~newton.Model.joint_target_ke`. For :attr:`~newton.JointTargetMode.POSITION`-only actuators,
      also syncs damping from :attr:`~newton.Model.joint_target_kd`. For
      :attr:`~newton.JointTargetMode.POSITION_VELOCITY` mode, kd is handled by the separate velocity actuator.
    - :attr:`VELOCITY`: Maps from :attr:`~newton.Control.joint_target_qd`, syncs gains from :attr:`~newton.Model.joint_target_kd`
    - :attr:`GENERAL`: Used with :attr:`~newton.solvers.SolverMuJoCo.CtrlSource.CTRL_DIRECT` mode for motor/general actuators
    - :attr:`DCMOTOR`: Set by the MJCF importer and :func:`~newton.solvers.mujoco.add_actuator_dcmotor`
      for natively resolved DC-motor parameters.
      Use the helper or importer to populate the associated parameters. Runtime edits to ``actuator_gainprm``, ``actuator_biasprm``,
      ``actuator_dynprm``, and ``actuator_forcerange`` are ignored for these rows.
      Compiled MJCF and USD DC motors instead use :attr:`GENERAL` with ``actuator_gaintype`` set to DC motor.
    """

    POSITION = 0
    VELOCITY = 1
    GENERAL = 2
    DCMOTOR = 3


class TrnType(IntEnum):
    """Transmission type values for MuJoCo actuators."""

    UNDEFINED = -1

    JOINT = 0
    JOINT_IN_PARENT = 1
    TENDON = 2
    SITE = 3
    BODY = 4
    SLIDERCRANK = 5


class _JointType(IntEnum):
    FREE = 0
    BALL = 1
    SLIDE = 2
    HINGE = 3


def _joint_dof_count(kind: int) -> int:
    return 6 if kind == _JointType.FREE else 3 if kind == _JointType.BALL else 1


@cache
def _transmission_types():
    import mujoco

    return {
        TrnType.JOINT: mujoco.mjtTrn.mjTRN_JOINT,
        TrnType.JOINT_IN_PARENT: mujoco.mjtTrn.mjTRN_JOINTINPARENT,
        TrnType.TENDON: mujoco.mjtTrn.mjTRN_TENDON,
        TrnType.SITE: mujoco.mjtTrn.mjTRN_SITE,
        TrnType.BODY: mujoco.mjtTrn.mjTRN_BODY,
        TrnType.SLIDERCRANK: mujoco.mjtTrn.mjTRN_SLIDERCRANK,
    }


__all__ = ["EqType"]
