# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Solver flags."""

import warnings
from enum import EnumType, IntFlag


class SolverModelFlags(IntFlag):
    """Flags indicating which parts of the model have been updated.

    These flags are used with :meth:`~newton.solvers.SolverBase.notify_model_changed`
    to specify which properties have changed, allowing the solver to efficiently
    update only the necessary components.
    """

    JOINT_PROPERTIES = 1 << 0
    """Indicates joint property updates: joint_q, joint_X_p, joint_X_c."""

    JOINT_DOF_PROPERTIES = 1 << 1
    """Indicates joint DOF property updates: joint_target_ke, joint_target_kd, joint_effort_limit, joint_armature, joint_friction, joint_limit_ke, joint_limit_kd, joint_limit_lower, joint_limit_upper."""

    BODY_PROPERTIES = 1 << 2
    """Indicates body property updates: body_q, body_qd, body_flags."""

    BODY_INERTIAL_PROPERTIES = 1 << 3
    """Indicates body inertial property updates: body_com, body_inertia, body_inv_inertia, body_mass, body_inv_mass."""

    SHAPE_PROPERTIES = 1 << 4
    """Indicates shape property updates: shape_transform, shape_scale, shape_collision_radius, shape_material_mu, shape_material_ke, shape_material_kd, rigid_contact_mu_torsional, rigid_contact_mu_rolling."""

    MODEL_PROPERTIES = 1 << 5
    """Indicates model property updates: gravity and other global parameters."""

    CONSTRAINT_PROPERTIES = 1 << 6
    """Indicates constraint property updates: equality constraints (equality_constraint_anchor, equality_constraint_relpose, equality_constraint_polycoef, equality_constraint_torquescale, equality_constraint_enabled, mujoco.eq_solref, mujoco.eq_solimp) and mimic constraints (constraint_mimic_coef0, constraint_mimic_coef1, constraint_mimic_enabled)."""

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


class SolverStateFlags(IntFlag):
    """Flags indicating which state attributes should be reset.

    These flags are used with :meth:`~newton.solvers.SolverBase.reset` to control
    which parts of the simulation state are reset, allowing the solver to
    efficiently update only the necessary components.
    """

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

    ALL = JOINT_Q | JOINT_QD | BODY_Q | BODY_QD | PARTICLE_Q | PARTICLE_QD
    """Indicates all state attributes should be reset."""


class _DeprecatedSolverNotifyFlagsMeta(EnumType):
    def __getattribute__(cls, name: str):
        value = super().__getattribute__(name)
        if not name.startswith("_"):
            member_map = super().__getattribute__("_member_map_")
            if name in member_map:
                _warn_solver_notify_flags_deprecated()
        return value

    def __call__(cls, *args, **kwargs):
        _warn_solver_notify_flags_deprecated()
        return super().__call__(*args, **kwargs)


def _warn_solver_notify_flags_deprecated() -> None:
    warnings.warn(
        "SolverNotifyFlags is deprecated, use SolverModelFlags instead.",
        DeprecationWarning,
        stacklevel=3,
    )


class SolverNotifyFlags(IntFlag, metaclass=_DeprecatedSolverNotifyFlagsMeta):
    """Deprecated alias for :class:`SolverModelFlags`.

    .. deprecated::
        Use :class:`SolverModelFlags` instead.
    """

    JOINT_PROPERTIES = SolverModelFlags.JOINT_PROPERTIES.value
    JOINT_DOF_PROPERTIES = SolverModelFlags.JOINT_DOF_PROPERTIES.value
    BODY_PROPERTIES = SolverModelFlags.BODY_PROPERTIES.value
    BODY_INERTIAL_PROPERTIES = SolverModelFlags.BODY_INERTIAL_PROPERTIES.value
    SHAPE_PROPERTIES = SolverModelFlags.SHAPE_PROPERTIES.value
    MODEL_PROPERTIES = SolverModelFlags.MODEL_PROPERTIES.value
    CONSTRAINT_PROPERTIES = SolverModelFlags.CONSTRAINT_PROPERTIES.value
    TENDON_PROPERTIES = SolverModelFlags.TENDON_PROPERTIES.value
    ACTUATOR_PROPERTIES = SolverModelFlags.ACTUATOR_PROPERTIES.value
    ALL = SolverModelFlags.ALL.value


__all__ = [
    "SolverModelFlags",
    "SolverNotifyFlags",
    "SolverStateFlags",
]
