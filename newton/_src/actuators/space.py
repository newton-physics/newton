# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

from enum import Enum


class DriveOutputSpace(str, Enum):
    """Physical interpretation of a drive output or clamping input."""

    EFFORT = "effort"
    """Joint force [N] or torque [N·m], with no explicit motor model."""
    DUTY = "duty"
    """Dimensionless signed duty, converted to voltage by the actuator."""


def _validate_spaces(output_space, clamp_spaces, has_motor):
    """Reject mixed modeling paths before allocating or stepping components."""
    expected = DriveOutputSpace.DUTY if has_motor else DriveOutputSpace.EFFORT
    if DriveOutputSpace(output_space) is not expected:
        raise ValueError(
            "A motor requires a DUTY drive; an EFFORT drive cannot be combined with a motor"
            if has_motor
            else "A DUTY drive requires a motor"
        )
    for space in clamp_spaces:
        if DriveOutputSpace(space) is not expected:
            raise ValueError(
                "A motor accepts only DUTY clamps; EFFORT clamps cannot be combined with a motor"
                if has_motor
                else "A DUTY clamp requires a motor; effort drives accept only EFFORT clamps"
            )
