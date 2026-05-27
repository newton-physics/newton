# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import warnings

import warp as wp

_JOINT_TARGET_POS_DEPRECATION_MSG = (
    "Control.joint_target_pos is deprecated; use Control.joint_target_q. The "
    "legacy DOF-shaped layout is misaligned with State.joint_q for free/ball "
    "joints. The attribute will be removed in a future release."
)
_JOINT_TARGET_VEL_DEPRECATION_MSG = (
    "Control.joint_target_vel is deprecated; use Control.joint_target_qd. The "
    "attribute will be removed in a future release."
)
_JOINT_TARGET_POS_UNAVAILABLE_MSG = (
    "Control.joint_target_pos is unavailable when newton.use_coord_layout_targets is True; use Control.joint_target_q."
)
_JOINT_TARGET_VEL_UNAVAILABLE_MSG = (
    "Control.joint_target_vel is unavailable when newton.use_coord_layout_targets is True; use Control.joint_target_qd."
)


class Control:
    """Time-varying control data for a :class:`Model`.

    Time-varying control data includes joint torques, control inputs, muscle activations,
    and activation forces for triangle and tetrahedral elements.

    The exact attributes depend on the contents of the model. Control objects
    should generally be created using the :func:`newton.Model.control()` function.

    :attr:`joint_target_q` and :attr:`joint_target_qd` carry the position and
    velocity targets, matching :attr:`~newton.State.joint_q` and
    :attr:`~newton.State.joint_qd`. Their shape depends on
    :attr:`newton.use_coord_layout_targets`: under ``True``,
    :attr:`joint_target_q` is shaped ``(joint_coord_count,)``; under ``False``
    it is shaped ``(joint_dof_count,)`` to preserve the legacy layout, and the
    deprecated :attr:`joint_target_pos` / :attr:`joint_target_vel` aliases are
    available (they raise :class:`AttributeError` when the flag is ``True``).
    """

    def __init__(self):
        self.joint_f: wp.array | None = None
        """
        Array of generalized joint forces [N or N·m, depending on joint type] with shape ``(joint_dof_count,)``
        and type ``float``.

        The degrees of freedom for FREE and DISTANCE joints are included in this array and have the same
        convention as the :attr:`newton.State.body_f` array where the 6D wrench is defined as
        ``(f_x, f_y, f_z, t_x, t_y, t_z)``, where ``f_x``, ``f_y``, and ``f_z`` are the components
        of the force vector (linear) [N] and ``t_x``, ``t_y``, and ``t_z`` are the
        components of the torque vector (angular) [N·m]. For FREE and DISTANCE joints, the wrench is applied in world
        frame with the child body's center of mass (COM) as reference point.
        """
        self.joint_target_q: wp.array | None = None
        """Joint position targets [m or rad, depending on joint type], shape ``(joint_coord_count,)``, type ``float``.

        Shape matches :attr:`~newton.State.joint_q` when
        :attr:`newton.use_coord_layout_targets` is ``True``; otherwise the array
        is shaped ``(joint_dof_count,)`` for backward compatibility with the
        deprecated :attr:`joint_target_pos` alias.
        """

        self.joint_target_qd: wp.array | None = None
        """Joint velocity targets [m/s or rad/s, depending on joint type], shape ``(joint_dof_count,)``, type ``float``.

        Matches the layout of :attr:`~newton.State.joint_qd`. Replaces the
        deprecated :attr:`joint_target_vel`.
        """

        self.joint_act: wp.array | None = None
        """Per-DOF feedforward actuation input, shape ``(joint_dof_count,)``, type ``float`` (optional).

        This is an additive feedforward term used by actuators (e.g. :class:`ActuatorPD`) in their control law
        before PD/PID correction is applied.
        """

        self.tri_activations: wp.array | None = None
        """Array of triangle element activations [dimensionless] with shape ``(tri_count,)`` and type ``float``."""

        self.tet_activations: wp.array | None = None
        """Array of tetrahedral element activations [dimensionless] with shape ``(tet_count,)`` and type ``float``."""

        self.muscle_activations: wp.array | None = None
        """
        Array of muscle activations [dimensionless, 0 to 1] with shape ``(muscle_count,)`` and type ``float``.

        .. note::
            Support for muscle dynamics is not yet implemented.
        """

    @property
    def joint_target_pos(self) -> wp.array | None:
        """Deprecated alias for :attr:`joint_target_q` (legacy DOF-shape only).

        Raises :class:`AttributeError` when
        :attr:`newton.use_coord_layout_targets` is ``True``.

        .. deprecated::
            Use :attr:`joint_target_q`.
        """
        import newton  # noqa: PLC0415

        if newton.use_coord_layout_targets:
            raise AttributeError(_JOINT_TARGET_POS_UNAVAILABLE_MSG)
        warnings.warn(_JOINT_TARGET_POS_DEPRECATION_MSG, DeprecationWarning, stacklevel=2)
        return self.joint_target_q

    @joint_target_pos.setter
    def joint_target_pos(self, value: wp.array | None) -> None:
        import newton  # noqa: PLC0415

        if newton.use_coord_layout_targets:
            raise AttributeError(_JOINT_TARGET_POS_UNAVAILABLE_MSG)
        warnings.warn(_JOINT_TARGET_POS_DEPRECATION_MSG, DeprecationWarning, stacklevel=2)
        self.joint_target_q = value

    @property
    def joint_target_vel(self) -> wp.array | None:
        """Deprecated alias for :attr:`joint_target_qd`.

        Raises :class:`AttributeError` when
        :attr:`newton.use_coord_layout_targets` is ``True``.

        .. deprecated::
            Use :attr:`joint_target_qd`.
        """
        import newton  # noqa: PLC0415

        if newton.use_coord_layout_targets:
            raise AttributeError(_JOINT_TARGET_VEL_UNAVAILABLE_MSG)
        warnings.warn(_JOINT_TARGET_VEL_DEPRECATION_MSG, DeprecationWarning, stacklevel=2)
        return self.joint_target_qd

    @joint_target_vel.setter
    def joint_target_vel(self, value: wp.array | None) -> None:
        import newton  # noqa: PLC0415

        if newton.use_coord_layout_targets:
            raise AttributeError(_JOINT_TARGET_VEL_UNAVAILABLE_MSG)
        warnings.warn(_JOINT_TARGET_VEL_DEPRECATION_MSG, DeprecationWarning, stacklevel=2)
        self.joint_target_qd = value

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
        if self.joint_target_q is not None:
            self.joint_target_q.zero_()
        if self.joint_target_qd is not None:
            self.joint_target_qd.zero_()
        if self.joint_act is not None:
            self.joint_act.zero_()
        self._clear_namespaced_arrays()

    def _clear_namespaced_arrays(self) -> None:
        """Clear all wp.array attributes in namespaced containers (e.g., control.mujoco.ctrl)."""
        from .model import Model  # noqa: PLC0415

        for attr in self.__dict__.values():
            if isinstance(attr, Model.AttributeNamespace):
                for value in attr.__dict__.values():
                    if isinstance(value, wp.array):
                        value.zero_()
