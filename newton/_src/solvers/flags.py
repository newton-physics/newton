# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Solver flags."""

import warnings
from enum import EnumMeta, IntEnum

from ..sim.enums import ModelFlags


class _DeprecatedSolverNotifyFlagsMeta(EnumMeta):
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
        "SolverNotifyFlags is deprecated, use ModelFlags instead.",
        DeprecationWarning,
        stacklevel=3,
    )


class SolverNotifyFlags(IntEnum, metaclass=_DeprecatedSolverNotifyFlagsMeta):
    """Deprecated alias for :class:`~newton.ModelFlags`.

    .. deprecated:: 1.3
        Use :class:`~newton.ModelFlags` instead.
    """

    JOINT_PROPERTIES = ModelFlags.JOINT_PROPERTIES.value
    JOINT_DOF_PROPERTIES = ModelFlags.JOINT_DOF_PROPERTIES.value
    BODY_PROPERTIES = ModelFlags.BODY_PROPERTIES.value
    BODY_INERTIAL_PROPERTIES = ModelFlags.BODY_INERTIAL_PROPERTIES.value
    SHAPE_PROPERTIES = ModelFlags.SHAPE_PROPERTIES.value
    MODEL_PROPERTIES = ModelFlags.MODEL_PROPERTIES.value
    CONSTRAINT_PROPERTIES = ModelFlags.CONSTRAINT_PROPERTIES.value
    TENDON_PROPERTIES = ModelFlags.TENDON_PROPERTIES.value
    ACTUATOR_PROPERTIES = ModelFlags.ACTUATOR_PROPERTIES.value
    ALL = ModelFlags.ALL.value


__all__ = [
    "SolverNotifyFlags",
]
