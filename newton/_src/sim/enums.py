# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import warnings
from enum import EnumMeta, IntEnum


class ModelFlags(IntEnum):
    """Flags indicating which parts of the model have been updated.

    These flags are used with :meth:`~newton.solvers.SolverBase.notify_model_changed`
    to specify which properties have changed, allowing the solver to efficiently
    update only the necessary components.
    """

    JOINT_PROPERTIES = 1 << 0
    """Indicates joint property updates: joint_q, joint_X_p, joint_X_c."""

    JOINT_DOF_PROPERTIES = 1 << 1
    """Indicates joint DOF property updates: joint_target_ke, joint_target_kd, joint_damping, joint_effort_limit, joint_armature, joint_friction, joint_limit_ke, joint_limit_kd, joint_limit_lower, joint_limit_upper."""

    BODY_PROPERTIES = 1 << 2
    """Indicates body property updates: body_q, body_qd, body_flags."""

    BODY_INERTIAL_PROPERTIES = 1 << 3
    """Indicates body inertial property updates: body_com, body_inertia, body_inv_inertia, body_mass, body_inv_mass."""

    SHAPE_PROPERTIES = 1 << 4
    """Indicates shape property updates: shape_transform, shape_scale, shape_collision_radius, shape_material_mu, shape_material_ke, shape_material_kd, rigid_contact_mu_torsional, rigid_contact_mu_rolling."""

    MODEL_PROPERTIES = 1 << 5
    """Indicates model property updates: gravity and other global parameters."""

    CONSTRAINT_PROPERTIES = 1 << 6
    """Indicates constraint property updates: equality constraints (mujoco.equality_constraint_anchor, mujoco.equality_constraint_relpose, mujoco.equality_constraint_polycoef, mujoco.equality_constraint_torquescale, mujoco.equality_constraint_enabled, mujoco.eq_solref, mujoco.eq_solimp) and mimic constraints (constraint_mimic_coef0, constraint_mimic_coef1, constraint_mimic_enabled)."""

    TENDON_PROPERTIES = 1 << 7
    """Indicates tendon properties: eg tendon_stiffness."""

    ACTUATOR_PROPERTIES = 1 << 8
    """Indicates actuator property updates: gains, biases, limits, etc."""

    ALL = (
        JOINT_PROPERTIES
        | JOINT_DOF_PROPERTIES
        | BODY_PROPERTIES
        | BODY_INERTIAL_PROPERTIES
        | SHAPE_PROPERTIES
        | MODEL_PROPERTIES
        | CONSTRAINT_PROPERTIES
        | TENDON_PROPERTIES
        | ACTUATOR_PROPERTIES
    )
    """Indicates all property updates."""


class StateFlags(IntEnum):
    """Flags indicating which state attributes were updated or should be reset.

    These flags are used with :meth:`~newton.solvers.SolverBase.reset` to
    control which parts of the simulation state are reset, and with
    :meth:`~newton.solvers.experimental.coupled.CouplingInterface.coupling_notify_input_state_update`
    to describe which public state inputs a coupler updated.

    .. experimental::

        The interpretation of these flags by
        :class:`~newton.solvers.experimental.coupled.CouplingInterface` may
        change without prior notice.
    """

    NONE = 0
    """Indicates no state attributes were updated."""

    JOINT_Q = 1 << 0
    """Indicates reduced joint position coordinates: ``State.joint_q``."""

    JOINT_QD = 1 << 1
    """Indicates reduced joint velocity coordinates: ``State.joint_qd``."""

    BODY_Q = 1 << 2
    """Indicates maximal body position coordinates: ``State.body_q``."""

    BODY_QD = 1 << 3
    """Indicates maximal body velocity coordinates: ``State.body_qd``."""

    PARTICLE_Q = 1 << 4
    """Indicates particle positions: ``State.particle_q``."""

    PARTICLE_QD = 1 << 5
    """Indicates particle velocities: ``State.particle_qd``."""

    BODY_F = 1 << 6
    """Indicates rigid-body force inputs: ``State.body_f``."""

    PARTICLE_F = 1 << 7
    """Indicates particle force inputs: ``State.particle_f``."""

    JOINT_F = 1 << 8
    """Indicates joint force inputs: ``Control.joint_f`` or solver-local equivalents."""

    BODY = BODY_Q | BODY_QD
    """Indicates rigid-body pose and velocity inputs."""

    PARTICLE = PARTICLE_Q | PARTICLE_QD
    """Indicates particle position and velocity inputs."""

    JOINT = JOINT_Q | JOINT_QD
    """Indicates joint position and velocity inputs."""

    FORCE = BODY_F | PARTICLE_F | JOINT_F
    """Indicates force-input arrays."""

    ALL = BODY | PARTICLE | JOINT | FORCE
    """Indicates all public state and force-input attributes."""


# Body flags
class BodyFlags(IntEnum):
    """
    Per-body dynamic state flags.

    Each finalized model body must store exactly one runtime state flag:
    :attr:`DYNAMIC` or :attr:`KINEMATIC`. Coupled solver views may OR in
    :attr:`PROXY` on view-local ``body_flags`` overrides. :attr:`ALL` is a
    convenience filter mask for APIs such as :func:`newton.eval_fk` and is not
    a valid stored body state.

    .. experimental::

        :attr:`PROXY` and its inclusion in :attr:`ALL` are part of the
        experimental coupled-solver contract and may change without prior
        notice.
    """

    DYNAMIC = 1 << 0
    """Dynamic body that participates in simulation dynamics."""

    KINEMATIC = 1 << 1
    """User-prescribed body that does not respond to applied forces."""

    PROXY = 1 << 2
    """View-local proxy body marker for coupled simulations."""

    ALL = DYNAMIC | KINEMATIC | PROXY
    """Filter bitmask selecting all body types."""


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
    """Cable joint: two DOF slots for linear stretch and angular bend/twist."""

    def dof_count(self, num_axes: int) -> tuple[int, int]:
        """
        Returns the number of degrees of freedom (DoF) in velocity and the number of coordinates
        in position for this joint type.

        Args:
            num_axes: The number of axes for the joint.

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
            num_axes: The number of DoF axes for the joint.

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


class _DeprecatedEqTypeMeta(EnumMeta):
    def __getattribute__(cls, name: str):
        value = super().__getattribute__(name)
        if not name.startswith("_"):
            member_map = super().__getattribute__("_member_map_")
            if name in member_map:
                _warn_eq_type_deprecated()
        return value

    def __call__(cls, *args, **kwargs):
        _warn_eq_type_deprecated()
        return super().__call__(*args, **kwargs)


def _warn_eq_type_deprecated() -> None:
    warnings.warn(
        "newton.EqType is deprecated in Newton 1.4; use newton.solvers.SolverMuJoCo.EqType instead.",
        DeprecationWarning,
        stacklevel=3,
    )


class EqType(IntEnum, metaclass=_DeprecatedEqTypeMeta):
    """Deprecated alias for :class:`~newton.solvers.SolverMuJoCo.EqType`.

    .. deprecated:: 1.4
        Use :class:`~newton.solvers.SolverMuJoCo.EqType` instead.
    """

    CONNECT = 0
    WELD = 1
    JOINT = 2


class JointTargetMode(IntEnum):
    """
    Enumeration of actuator modes for joint degrees of freedom.

    This enum manages UsdPhysics compliance by specifying whether joint_target_q/qd
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
    """Only a position actuator is installed for this DOF. Tracks joint_target_q."""

    VELOCITY = 2
    """Only a velocity actuator is installed for this DOF. Tracks joint_target_qd."""

    POSITION_VELOCITY = 3
    """Both position and velocity actuators are installed. Tracks both joint_target_q and joint_target_qd."""

    EFFORT = 4
    """A drive is applied but no gains are configured. No MuJoCo actuator is created for this DOF.
    The user is expected to supply force via joint_f."""

    @staticmethod
    def from_gains(
        target_ke: float,
        target_kd: float,
        force_position_velocity: bool = False,
        has_drive: bool = False,
    ) -> "JointTargetMode":
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
            The inferred JointTargetMode based on which gains are non-zero:
            - NONE: No drive applied
            - EFFORT: Drive applied but both gains are 0 (direct torque control)
            - POSITION: Only position gain is non-zero
            - VELOCITY: Only velocity gain is non-zero
            - POSITION_VELOCITY: Both gains non-zero (or forced)
        """
        if not has_drive:
            return JointTargetMode.NONE

        if force_position_velocity and (target_ke != 0.0 and target_kd != 0.0):
            return JointTargetMode.POSITION_VELOCITY
        elif target_ke != 0.0:
            return JointTargetMode.POSITION
        elif target_kd != 0.0:
            return JointTargetMode.VELOCITY
        else:
            return JointTargetMode.EFFORT


__all__ = [
    "BodyFlags",
    "EqType",
    "JointTargetMode",
    "JointType",
    "ModelFlags",
    "StateFlags",
]
