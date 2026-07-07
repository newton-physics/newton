# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import warp as wp

if TYPE_CHECKING:
    from .model import Model

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

    Carries joint torques, control inputs, muscle activations, and tri/tet
    activation forces. Create via :func:`newton.Model.control()`.

    Position and velocity targets live on :attr:`joint_target_q` and
    :attr:`joint_target_qd`. The shape of :attr:`joint_target_q` depends on
    :data:`newton.use_coord_layout_targets` — coord-shaped when ``True``,
    DOF-shaped otherwise. Legacy :attr:`joint_target_pos` /
    :attr:`joint_target_vel` aliases are available under ``False`` and raise
    under ``True``.
    """

    def __init__(self):
        import newton  # noqa: PLC0415

        self._use_coord_layout_targets: bool = newton.use_coord_layout_targets

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
        """Joint position targets [m or rad]. Shape is ``(joint_coord_count,)``
        when :data:`newton.use_coord_layout_targets` is ``True``, otherwise
        ``(joint_dof_count,)`` for legacy compat with :attr:`joint_target_pos`.
        """

        self.joint_target_qd: wp.array | None = None
        """Joint velocity targets [m/s or rad/s], shape ``(joint_dof_count,)``.
        Matches :attr:`~newton.State.joint_qd`; replaces :attr:`joint_target_vel`.
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
        """Deprecated alias for :attr:`joint_target_q` (DOF-shape only).
        Raises :class:`AttributeError` under
        :data:`newton.use_coord_layout_targets` ``True``.

        .. deprecated:: 1.3
            Use :attr:`joint_target_q` instead.
        """
        if self._use_coord_layout_targets:
            raise AttributeError(_JOINT_TARGET_POS_UNAVAILABLE_MSG)
        warnings.warn(_JOINT_TARGET_POS_DEPRECATION_MSG, DeprecationWarning, stacklevel=2)
        return self.joint_target_q

    @joint_target_pos.setter
    def joint_target_pos(self, value: wp.array | None) -> None:
        if self._use_coord_layout_targets:
            raise AttributeError(_JOINT_TARGET_POS_UNAVAILABLE_MSG)
        warnings.warn(_JOINT_TARGET_POS_DEPRECATION_MSG, DeprecationWarning, stacklevel=2)
        self.joint_target_q = value

    @property
    def joint_target_vel(self) -> wp.array | None:
        """Deprecated alias for :attr:`joint_target_qd`. Raises
        :class:`AttributeError` under
        :data:`newton.use_coord_layout_targets` ``True``.

        .. deprecated:: 1.3
            Use :attr:`joint_target_qd` instead.
        """
        if self._use_coord_layout_targets:
            raise AttributeError(_JOINT_TARGET_VEL_UNAVAILABLE_MSG)
        warnings.warn(_JOINT_TARGET_VEL_DEPRECATION_MSG, DeprecationWarning, stacklevel=2)
        return self.joint_target_qd

    @joint_target_vel.setter
    def joint_target_vel(self, value: wp.array | None) -> None:
        if self._use_coord_layout_targets:
            raise AttributeError(_JOINT_TARGET_VEL_UNAVAILABLE_MSG)
        warnings.warn(_JOINT_TARGET_VEL_DEPRECATION_MSG, DeprecationWarning, stacklevel=2)
        self.joint_target_qd = value

    def clear(self, model: Model | None = None) -> None:
        """Reset all control inputs to zero.

        ``joint_target_q`` is special: zeroing it under coord layout corrupts
        FREE/BALL/DISTANCE quaternion slots (``(0,0,0,0)`` is not a valid
        rotation). Pass ``model`` to restore it from ``model.joint_target_q``
        instead. Without ``model`` it falls back to the legacy zero-fill.

        Args:
            model: Optional source :class:`Model` whose ``joint_target_q``
                seeds this Control. Required for models with FREE/BALL/DISTANCE
                joints under coord layout.
        """

        if self.joint_f is not None:
            self.joint_f.zero_()
        if self.tri_activations is not None:
            self.tri_activations.zero_()
        if self.tet_activations is not None:
            self.tet_activations.zero_()
        if self.muscle_activations is not None:
            self.muscle_activations.zero_()
        if self.joint_target_q is not None:
            if model is not None and model.joint_target_q is not None:
                wp.copy(self.joint_target_q, model.joint_target_q)
            else:
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
