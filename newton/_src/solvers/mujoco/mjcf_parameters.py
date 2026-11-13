# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Declare retained MJCF fields, using native defaults only when importing MJCF."""

from __future__ import annotations

import numpy as np
import warp as wp

from ...core.types import vec5
from ...sim import ModelBuilder
from ...sim.model import Model

_NATIVE_JOINT_FIELDS = {
    "type": wp.int32,
    "springdamper": wp.vec2,
    "limited": wp.int32,
    "actfrcrange": wp.vec2,
    "actfrclimited": wp.int32,
    "dampingpoly": wp.vec2,
    "stiffnesspoly": wp.vec2,
}


_NATIVE_GEOM_FIELDS = {
    "fluid_ellipsoid": wp.float32,
    "fluid_coefs": vec5,
    "surfacevel": wp.spatial_vector,
    "adhesion": wp.float32,
}
_NATIVE_ACTUATOR_FIELDS = {
    "inheritrange": wp.float32,
    "group": wp.int32,
    "delay": wp.float32,
    "nsample": wp.int32,
    "interp": wp.int32,
}


_NATIVE_OPTION_FIELDS = {
    "timestep": wp.float32,
    "enableflags": wp.int32,
    "disableflags": wp.int32,
    "disableactuator": wp.int32,
    "o_margin": wp.float32,
    "o_solref": wp.vec2,
    "o_solimp": vec5,
    "o_friction": vec5,
}


_NATIVE_LENGTHRANGE_FIELDS = {
    "mode": wp.int32,
    "useexisting": wp.int32,
    "uselimit": wp.int32,
    "accel": wp.float32,
    "maxforce": wp.float32,
    "timeconst": wp.float32,
    "timestep": wp.float32,
    "inttotal": wp.float32,
    "interval": wp.float32,
    "tolrange": wp.float32,
}


_NATIVE_FIELDS = {
    "joint": (Model.AttributeFrequency.JOINT_DOF, _NATIVE_JOINT_FIELDS),
    "geom": (Model.AttributeFrequency.SHAPE, _NATIVE_GEOM_FIELDS),
    "actuator": ("mujoco:actuator", _NATIVE_ACTUATOR_FIELDS),
    "option": (Model.AttributeFrequency.ONCE, _NATIVE_OPTION_FIELDS),
    "lengthrange": (Model.AttributeFrequency.ONCE, _NATIVE_LENGTHRANGE_FIELDS),
}


def _read_native_parameters(kind: str, element) -> dict:
    return {f"native_{kind}_{name}": getattr(element, name) for name in _NATIVE_FIELDS[kind][1]}


def _register_native_attributes(builder: ModelBuilder) -> None:
    import mujoco

    spec = mujoco.MjSpec()
    elements = {
        "joint": spec.worldbody.add_body().add_joint(),
        "geom": spec.worldbody.add_geom(),
        "actuator": spec.add_actuator(),
        "option": spec.option,
        "lengthrange": spec.compiler.LRopt,
    }
    tendon = spec.add_tendon()
    for kind, (frequency, fields) in _NATIVE_FIELDS.items():
        for name, dtype in fields.items():
            if kind == "joint" and name == "type":
                default = -1  # No imported joint owns this row.
            elif kind == "joint" and name.endswith("poly"):
                default = getattr(elements[kind], name.removesuffix("poly"))[1:].tolist()
            else:
                default = np.asarray(getattr(elements[kind], name)).tolist()
            builder.add_custom_attribute(
                ModelBuilder.CustomAttribute(
                    name=f"native_{kind}_{name}",
                    namespace="mujoco",
                    frequency=frequency,
                    dtype=dtype,
                    default=default,
                )
            )

    for name, frequency, dtype, default, reference in (
        ("native_tendon_fixed", "mujoco:tendon", wp.bool, False, None),
        ("native_tendon_joint_dof", "mujoco:tendon_joint", wp.int32, -1, "joint_dof"),
        ("native_tendon_stiffnesspoly", "mujoco:tendon", wp.vec2, tendon.stiffness[1:].tolist(), None),
        ("native_tendon_dampingpoly", "mujoco:tendon", wp.vec2, tendon.damping[1:].tolist(), None),
        ("native_equality_auto_relpose", "mujoco:equality_constraint", wp.bool, False, None),
        ("native_equality_site1", "mujoco:equality_constraint", wp.int32, -1, "shape"),
        ("native_equality_site2", "mujoco:equality_constraint", wp.int32, -1, "shape"),
        ("native_actuator_target_kd", "mujoco:actuator", wp.float32, -1.0, None),
        ("native_actuator_inheritrange_bounds", "mujoco:actuator", wp.vec2, (0.0, 0.0), None),
        ("native_joint_range", Model.AttributeFrequency.JOINT_DOF, wp.vec2, (0.0, 0.0), None),
        ("native_joint_gain_baseline", Model.AttributeFrequency.JOINT_DOF, wp.vec2, (-1.0, -1.0), None),
        ("native_joint_effort_baseline", Model.AttributeFrequency.JOINT_DOF, wp.float32, -1.0, None),
        ("native_joint_effort_limited", Model.AttributeFrequency.JOINT_DOF, wp.bool, False, None),
        ("native_target_kd_override", Model.AttributeFrequency.JOINT_DOF, wp.bool, False, None),
    ):
        builder.add_custom_attribute(
            ModelBuilder.CustomAttribute(
                name=name,
                namespace="mujoco",
                frequency=frequency,
                dtype=dtype,
                default=default,
                references=reference,
            )
        )
