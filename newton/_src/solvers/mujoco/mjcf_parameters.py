# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Preserve MJCF parameters through Newton composition and MuJoCo compilation."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import warp as wp

from ...core.types import MAXVAL, vec5
from ...sim import ModelBuilder
from ...sim.model import Model

if TYPE_CHECKING:
    import mujoco


_NATIVE_JOINT_FIELDS = {
    "type": (wp.int32, -1),
    "springdamper": (wp.vec2, (0.0, 0.0)),
    "range": (wp.vec2, (0.0, 0.0)),
    "limited": (wp.int32, 2),
    "actfrcrange": (wp.vec2, (0.0, 0.0)),
    "actfrclimited": (wp.int32, 2),
    "dampingpoly": (wp.vec2, (0.0, 0.0)),
    "stiffnesspoly": (wp.vec2, (0.0, 0.0)),
}


_NATIVE_GEOM_FIELDS = {
    "fluid_ellipsoid": (wp.float32, 0.0),
    "fluid_coefs": (vec5, (0.5, 0.25, 1.5, 1.0, 1.0)),
    "surfacevel": (wp.spatial_vector, (0.0,) * 6),
    "adhesion": (wp.float32, 0.0),
}
_NATIVE_ACTUATOR_FIELDS = {
    "inheritrange": (wp.float32, 0.0),
    "group": (wp.int32, 0),
    "delay": (wp.float32, 0.0),
    "nsample": (wp.int32, 0),
    "interp": (wp.int32, 0),
}


_NATIVE_OPTION_FIELDS = {
    "enableflags": (wp.int32, 0),
    "disableflags": (wp.int32, 0),
    "disableactuator": (wp.int32, 0),
    "o_margin": (wp.float32, 0.0),
    "o_solref": (wp.vec2, (0.02, 1.0)),
    "o_solimp": (vec5, (0.9, 0.95, 0.001, 0.5, 2.0)),
    "o_friction": (vec5, (1.0, 1.0, 0.005, 0.0001, 0.0001)),
}


_NATIVE_LENGTHRANGE_FIELDS = {
    "mode": (wp.int32, 1),
    "useexisting": (wp.int32, 1),
    "uselimit": (wp.int32, 0),
    "accel": (wp.float32, 20.0),
    "maxforce": (wp.float32, 0.0),
    "timeconst": (wp.float32, 1.0),
    "timestep": (wp.float32, 0.01),
    "inttotal": (wp.float32, 10.0),
    "interval": (wp.float32, 2.0),
    "tolrange": (wp.float32, 0.05),
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


def _apply_native_element_parameters(model, spec, shapes, actuator_sources, actuator_indices):
    import mujoco

    from .solver_mujoco import SolverMuJoCo  # noqa: PLC0415

    for name in _NATIVE_OPTION_FIELDS:
        value = getattr(model.mujoco, f"native_option_{name}").numpy()[0].tolist()
        if name == "enableflags":
            mask = int(mujoco.mjtEnableBit.mjENBL_SLEEP) | sum(
                int(getattr(mujoco.mjtEnableBit, f"mjENBL_{flag.upper()}")) for flag in SolverMuJoCo._ENABLE_FLAGS
            )
            value = spec.option.enableflags | (value & ~mask)
        elif name == "disableflags":
            mask = sum(
                int(getattr(mujoco.mjtDisableBit, f"mjDSBL_{flag.upper()}")) for flag in SolverMuJoCo._DISABLE_FLAGS
            )
            value = spec.option.disableflags | (value & ~mask)
        setattr(spec.option, name, value)
    for name in _NATIVE_LENGTHRANGE_FIELDS:
        setattr(spec.compiler.LRopt, name, getattr(model.mujoco, f"native_lengthrange_{name}").numpy()[0].item())
    for kind, fields, elements in (
        ("geom", _NATIVE_GEOM_FIELDS, ((shape, spec.geom(name)) for shape, name in shapes.items())),
        (
            "actuator",
            _NATIVE_ACTUATOR_FIELDS,
            (
                (index, spec.actuators[i])
                for i, (source, index) in enumerate(zip(actuator_sources, actuator_indices, strict=True))
                if source == 1
            ),
        ),
    ):
        if not hasattr(model.mujoco, f"native_{kind}_{next(iter(fields))}"):
            continue
        values = {name: getattr(model.mujoco, f"native_{kind}_{name}").numpy() for name in fields}
        for index, element in elements:
            for name in fields:
                value = values[name][index]
                setattr(element, name, value.tolist())
            if kind == "actuator" and element.inheritrange > 0:
                # Model arrays hold resolved ranges after the first solver construction.
                if element.dyntype == mujoco.mjtDyn.mjDYN_INTEGRATOR:
                    element.actrange[:] = 0
                else:
                    element.ctrlrange[:] = 0


def _register_native_attributes(builder: ModelBuilder) -> None:
    for kind, (frequency, fields) in _NATIVE_FIELDS.items():
        for name, (dtype, default) in fields.items():
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
        ("native_tendon_joint_dof", "mujoco:tendon_joint", wp.int32, -1, "joint_dof"),
        ("native_tendon_stiffnesspoly", "mujoco:tendon", wp.vec2, (0.0, 0.0), None),
        ("native_tendon_dampingpoly", "mujoco:tendon", wp.vec2, (0.0, 0.0), None),
        ("native_equality_auto_relpose", "mujoco:equality_constraint", wp.bool, False, None),
        ("native_equality_site1", "mujoco:equality_constraint", wp.int32, -1, "shape"),
        ("native_equality_site2", "mujoco:equality_constraint", wp.int32, -1, "shape"),
        ("native_equality_dof1", "mujoco:equality_constraint", wp.int32, -1, "joint_dof"),
        ("native_equality_dof2", "mujoco:equality_constraint", wp.int32, -1, "joint_dof"),
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


def _apply_native_joint_parameters(model: Model, spec: mujoco.MjSpec, dof_to_joint) -> dict[int, int]:
    import mujoco

    attrs = model.mujoco
    if not hasattr(attrs, "native_joint_type"):
        return {}
    fields = {name: getattr(attrs, f"native_joint_{name}").numpy() for name in _NATIVE_JOINT_FIELDS}
    stiffness = attrs.dof_passive_stiffness.numpy()
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
    resolved = {}
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
        scalar_has_limits = int(joint.limited) == 1
        joint.limited = int(fields["limited"][dof])
        joint.actgravcomp = bool(actgravcomp[dof])
        joint.actfrclimited = int(fields["actfrclimited"][dof])
        joint.actfrcrange[:] = fields["actfrcrange"][dof]
        joint.solref_friction[:] = constraint_fields["solreffriction"][dof]
        joint.solimp_friction[:] = constraint_fields["solimpfriction"][dof]
        if joint.type in (mujoco.mjtJoint.mjJNT_BALL, mujoco.mjtJoint.mjJNT_FREE):
            joint.range[:] = (
                np.rad2deg(fields["range"][dof])
                if spec.compiler.degree and fields["limited"][dof] != 0
                else fields["range"][dof]
            )
            joint.margin = float(constraint_fields["limit_margin"][dof])
            joint.solref_limit[:] = constraint_fields["solreflimit"][dof]
            joint.solimp_limit[:] = constraint_fields["solimplimit"][dof]
        elif not scalar_has_limits:
            joint.range[:] = fields["range"][dof]
        if np.all(fields["springdamper"][dof] > 0):
            resolved[dof] = int(index)
    return resolved


def _sync_native_joint_parameters(model: Model, native: mujoco.MjModel, resolved: dict[int, int]) -> None:
    if not resolved:
        return
    stiffness, damping = model.mujoco.dof_passive_stiffness.numpy(), model.joint_damping.numpy()
    dofs_per_world = model.joint_dof_count // max(model.world_count, 1)
    for dof, j in resolved.items():
        width = 6 if native.jnt_type[j] == 0 else 3 if native.jnt_type[j] == 1 else 1
        for offset in range(0, model.joint_dof_count, dofs_per_world):
            stiffness[offset + dof : offset + dof + width] = native.jnt_stiffness[j]
            damping[offset + dof : offset + dof + width] = native.dof_damping[
                native.jnt_dofadr[j] : native.jnt_dofadr[j] + width
            ]
    model.mujoco.dof_passive_stiffness.assign(stiffness)
    model.joint_damping.assign(damping)


def _sync_native_actuator_ranges(model: Model) -> None:
    from .solver_mujoco import SolverMuJoCo  # noqa: PLC0415

    attrs = model.mujoco
    factors = getattr(attrs, "native_actuator_inheritrange", None)
    if factors is None:
        return
    factors = factors.numpy()
    if not np.any(factors > 0):
        return
    targets = attrs.actuator_trnid.numpy()[:, 0]
    transmissions = attrs.actuator_trntype.numpy()
    tendon_ranges = attrs.tendon_range.numpy() if np.any(transmissions == SolverMuJoCo.TrnType.TENDON) else None
    lower, upper = model.joint_limit_lower.numpy(), model.joint_limit_upper.numpy()
    refs, native_ranges = attrs.dof_ref.numpy(), attrs.native_joint_range.numpy()
    dynamics = attrs.actuator_dyntype.numpy()
    control, activation = attrs.actuator_ctrlrange.numpy(), attrs.actuator_actrange.numpy()
    for row in np.flatnonzero(factors > 0):
        dof = targets[row]
        # The compiler only builds the first world; resolve each world's own limits.
        if transmissions[row] == SolverMuJoCo.TrnType.JOINT:
            bounds = (
                native_ranges[dof]
                if lower[dof] <= -MAXVAL and upper[dof] >= MAXVAL
                else np.array([lower[dof], upper[dof]]) + refs[dof]
            )
        elif transmissions[row] == SolverMuJoCo.TrnType.TENDON:
            bounds = tendon_ranges[targets[row]]
        else:
            continue
        center, radius = np.mean(bounds), (bounds[1] - bounds[0]) * factors[row] / 2
        ranges = activation if dynamics[row] == 1 else control
        ranges[row] = [center - radius, center + radius]
    attrs.actuator_ctrlrange.assign(control)
    attrs.actuator_actrange.assign(activation)


def _sync_native_equality_parameters(model, native, equality_rows):
    automatic = getattr(model.mujoco, "native_equality_auto_relpose", None)
    if automatic is None:
        return
    automatic = automatic.numpy()
    poses = model.mujoco.equality_constraint_relpose.numpy()
    bodies = model.body_q.numpy()
    body1 = model.mujoco.equality_constraint_body1.numpy()
    body2 = model.mujoco.equality_constraint_body2.numpy()
    for row in np.flatnonzero(automatic):
        first = wp.transform(*bodies[body1[row]]) if body1[row] >= 0 else wp.transform_identity()
        second = wp.transform(*bodies[body2[row]]) if body2[row] >= 0 else wp.transform_identity()
        poses[row] = wp.transform_inverse(first) * second
    for equality, row in equality_rows.items():
        if automatic[row]:
            data = native.eq_data[equality]
            poses[row] = np.concatenate((data[3:6], data[7:10], data[6:7]))
    model.mujoco.equality_constraint_relpose.assign(poses)


def _apply_native_parameters(model, spec, dof_to_joint, shapes, actuator_sources, actuator_indices):
    if not hasattr(model, "mujoco") or not hasattr(model.mujoco, "native_option_enableflags"):
        return None
    resolved = _apply_native_joint_parameters(model, spec, dof_to_joint)
    _apply_native_element_parameters(model, spec, shapes, actuator_sources, actuator_indices)
    return resolved


def _sync_native_parameters(model, native, resolved_joints, equality_rows):
    if resolved_joints is None:
        return
    _sync_native_joint_parameters(model, native, resolved_joints)
    _sync_native_equality_parameters(model, native, equality_rows)
    _sync_native_actuator_ranges(model)
