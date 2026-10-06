# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Physical cable setup, independent of the fit procedure."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import Any

from .model import ANGLE_PARAM_EXP
from .schema import SCHEMA_VERSION, check_fields, check_schema_version
from .schema import is_real as _is_real


@dataclass
class CableSetupSpec:
    """Geometry, discretization and attachment of a cable experiment."""

    num_elements: int
    """Number of rod elements (capsules). It also sets the number of rest-angle pairs."""

    segment_length: float
    """Length of each element [m]."""

    cable_radius: float
    """Rod radius [m]."""

    cable_mass: float
    """Total cable mass [kg], divided equally across the elements."""

    angle_parametrization: str
    """How a rest-angle pair becomes a joint rotation. Only ``"exp_map"`` is supported."""

    attachment_transform: list[list[float]]
    """Attachment pose ``[[x, y, z], [qx, qy, qz, qw]]``, translation [m].

    For driven evidence, the pose is relative to the TCP. For non-driven
    evidence, the translation offsets ``cable_start`` and the rotation is in the
    robot base frame.
    """

    clamp_position: float
    """Arc length [m] of the grasp from the node-0 end of the cable.

    0.0 is the end clamp. An interior value grasps the cable between its ends,
    so both sides hang. The value applies to every recording.
    """

    stretch_stiffness: float
    """Per-joint stretch stiffness of the rod [N/m]. It is held fixed, not searched."""

    cable_axis: list[float] | None = None
    """Cable direction ``[x, y, z]`` at the grasp, or ``None`` to use the rotation of :attr:`attachment_transform`.

    The direction is in the TCP frame for driven evidence and in the robot base
    frame for non-driven evidence. The value applies to every recording.
    """

    schema_version: int = SCHEMA_VERSION
    """Format version of the serialized setup."""

    def to_dict(self) -> dict[str, Any]:
        """Return the setup as a JSON-serializable mapping."""
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> CableSetupSpec:
        """Build and validate a setup from :meth:`to_dict` output.

        Raises:
            ValueError: If the schema version is not supported, ``d`` has a
                field this class does not define, or :meth:`validate` fails.
            TypeError: If ``d`` has no value for a required field.
        """
        d = dict(d)
        check_schema_version(d.get("schema_version"), "CableSetupSpec")
        check_fields(cls, d, "CableSetupSpec")
        result = cls(**d)
        result.validate()
        return result

    def validate(self) -> None:
        """Check the geometry, mass, stiffness and attachment.

        Raises:
            ValueError: If the schema version is not supported, if a value has
                the wrong type, is not finite, not positive or out of range, if
                the angle parametrization is not
                supported, if the attachment is not a finite position and unit
                quaternion, or if the cable axis is not three finite numbers that
                are not all zero.
        """
        check_schema_version(self.schema_version, "CableSetupSpec")
        if type(self.num_elements) is not int or self.num_elements <= 0:
            raise ValueError(f"num_elements must be an integer > 0, got {self.num_elements!r}.")
        for name in ("segment_length", "cable_radius", "cable_mass", "stretch_stiffness"):
            value = getattr(self, name)
            if not _is_real(value) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be a finite number > 0, got {value!r}.")
        if self.angle_parametrization != ANGLE_PARAM_EXP:
            raise ValueError(
                f"angle_parametrization {self.angle_parametrization!r} is not supported; use {ANGLE_PARAM_EXP!r}."
            )
        transform = self.attachment_transform
        if (
            not isinstance(transform, (list, tuple))
            or len(transform) != 2
            or not isinstance(transform[0], (list, tuple))
            or not isinstance(transform[1], (list, tuple))
            or len(transform[0]) != 3
            or len(transform[1]) != 4
        ):
            raise ValueError("attachment_transform must be [[x, y, z], [qx, qy, qz, qw]].")
        if not all(_is_real(v) and math.isfinite(v) for part in transform for v in part):
            raise ValueError("attachment_transform must contain finite numbers.")
        if not math.isclose(sum(v * v for v in transform[1]), 1.0, abs_tol=1e-6):
            raise ValueError("attachment_transform rotation must be a unit quaternion.")
        cable_length = self.num_elements * self.segment_length
        if not _is_real(self.clamp_position) or not 0.0 <= self.clamp_position <= cable_length:
            raise ValueError(
                f"clamp_position must be in [0, num_elements * segment_length] = [0, {cable_length}], "
                f"got {self.clamp_position!r}."
            )
        axis = self.cable_axis
        if axis is not None and (
            not isinstance(axis, (list, tuple))
            or len(axis) != 3
            or not all(_is_real(v) and math.isfinite(v) for v in axis)
            or not any(axis)
        ):
            raise ValueError(f"cable_axis must be None or three finite numbers, not all zero, got {axis!r}.")
