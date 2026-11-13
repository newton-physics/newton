# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Transfer retained MJCF parameters into a solver-owned MuJoCo spec."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from ...sim.model import Model
from .constants import _DISABLE_FLAGS, _ENABLE_FLAGS
from .mjcf_parameters import (
    _NATIVE_ACTUATOR_FIELDS,
    _NATIVE_GEOM_FIELDS,
    _NATIVE_JOINT_FIELDS,
    _NATIVE_LENGTHRANGE_FIELDS,
    _NATIVE_OPTION_FIELDS,
)

if TYPE_CHECKING:
    import mujoco


def _apply_native_element_parameters(model, spec, shapes, actuator_indices):
    import mujoco

    for name in _NATIVE_OPTION_FIELDS:
        value = getattr(model.mujoco, f"native_option_{name}").numpy()[0].tolist()
        if name == "enableflags":
            mask = int(mujoco.mjtEnableBit.mjENBL_SLEEP) | sum(
                int(getattr(mujoco.mjtEnableBit, f"mjENBL_{flag.upper()}")) for flag in _ENABLE_FLAGS
            )
            value = spec.option.enableflags | (value & ~mask)
        elif name == "disableflags":
            mask = sum(int(getattr(mujoco.mjtDisableBit, f"mjDSBL_{flag.upper()}")) for flag in _DISABLE_FLAGS)
            value = spec.option.disableflags | (value & ~mask)
        setattr(spec.option, name, value)
    for name in _NATIVE_LENGTHRANGE_FIELDS:
        setattr(spec.compiler.LRopt, name, getattr(model.mujoco, f"native_lengthrange_{name}").numpy()[0].item())
    for kind, fields, elements in (
        ("geom", _NATIVE_GEOM_FIELDS, ((shape, spec.geom(name)) for shape, name in shapes.items())),
        (
            "actuator",
            _NATIVE_ACTUATOR_FIELDS,
            ((index, spec.actuators[i]) for i, index in enumerate(actuator_indices) if index >= 0),
        ),
    ):
        if not hasattr(model.mujoco, f"native_{kind}_{next(iter(fields))}"):
            continue
        values = {name: getattr(model.mujoco, f"native_{kind}_{name}").numpy() for name in fields}
        for index, element in elements:
            for name in fields:
                value = values[name][index]
                setattr(element, name, value.tolist())


def _apply_native_joint_parameters(model: Model, spec: mujoco.MjSpec, dof_to_joint) -> None:
    import mujoco

    attrs = model.mujoco
    if not hasattr(attrs, "native_joint_type"):
        return
    fields = {name: getattr(attrs, f"native_joint_{name}").numpy() for name in _NATIVE_JOINT_FIELDS}
    ranges = attrs.native_joint_range.numpy()
    stiffness = model.joint_stiffness.numpy()
    effort = model.joint_effort_limit.numpy()
    effort_baseline = attrs.native_joint_effort_baseline.numpy()
    actgravcomp = attrs.jnt_actgravcomp.numpy()
    damping, armature, friction = (
        model.joint_damping.numpy(),
        model.joint_armature.numpy(),
        model.joint_friction.numpy(),
    )
    constraint_fields = {
        name: getattr(attrs, name).numpy()
        for name in ("solreffriction", "solimpfriction", "limit_margin", "solreflimit", "solimplimit")
    }
    for dof, index in enumerate(dof_to_joint):
        if index < 0 or fields["type"][dof] < 0:
            continue
        joint = spec.joints[int(index)]
        if int(joint.type) != int(fields["type"][dof]):
            continue
        joint.springdamper[:] = fields["springdamper"][dof]
        joint.stiffness[:] = [stiffness[dof], *fields["stiffnesspoly"][dof]]
        joint.damping[:] = [damping[dof], *fields["dampingpoly"][dof]]
        joint.armature, joint.frictionloss = float(armature[dof]), float(friction[dof])
        joint.actgravcomp = bool(actgravcomp[dof])
        joint.actfrclimited = int(fields["actfrclimited"][dof])
        joint.actfrcrange[:] = fields["actfrcrange"][dof]
        if (
            joint.type in (mujoco.mjtJoint.mjJNT_SLIDE, mujoco.mjtJoint.mjJNT_HINGE)
            and effort[dof] != effort_baseline[dof]
        ):
            # MuJoCo rejects an enabled zero range; initial publication enables its clamp.
            joint.actfrclimited = bool(effort[dof])
            joint.actfrcrange[:] = (-effort[dof], effort[dof])
        joint.solref_friction[:] = constraint_fields["solreffriction"][dof]
        joint.solimp_friction[:] = constraint_fields["solimpfriction"][dof]
        if joint.type in (mujoco.mjtJoint.mjJNT_BALL, mujoco.mjtJoint.mjJNT_FREE):
            joint.limited = int(fields["limited"][dof])
            joint.range[:] = (
                np.rad2deg(ranges[dof]) if spec.compiler.degree and fields["limited"][dof] != 0 else ranges[dof]
            )
            joint.margin = float(constraint_fields["limit_margin"][dof])
            joint.solref_limit[:] = constraint_fields["solreflimit"][dof]
            joint.solimp_limit[:] = constraint_fields["solimplimit"][dof]


def _apply_native_parameters(model, spec, dof_to_joint, shapes, actuator_indices):
    if not hasattr(model, "mujoco") or not hasattr(model.mujoco, "native_option_enableflags"):
        return False
    _apply_native_joint_parameters(model, spec, dof_to_joint)
    _apply_native_element_parameters(model, spec, shapes, actuator_indices)
    return True
