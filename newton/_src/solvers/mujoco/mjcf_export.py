# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Transfer retained MJCF parameters into solver-owned MuJoCo models."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from ...core.types import MAXVAL
from ...sim.model import Model
from .constants import _DISABLE_FLAGS, _ENABLE_FLAGS
from .enums import TrnType, _JointType
from .mjcf_parameters import (
    _NATIVE_ACTUATOR_FIELDS,
    _NATIVE_GEOM_FIELDS,
    _NATIVE_JOINT_FIELDS,
    _NATIVE_LENGTHRANGE_FIELDS,
    _NATIVE_OPTION_FIELDS,
)

if TYPE_CHECKING:
    import mujoco


_NATIVE_WORLD_FIELDS = {
    "joint": {"native_joint_stiffnesspoly": "jnt_stiffnesspoly"},
    "dof": {"native_joint_dampingpoly": "dof_dampingpoly"},
    "geom": {"native_geom_surfacevel": "geom_surfacevel", "native_geom_adhesion": "geom_adhesion"},
    "pair": {"native_pair_adhesion": "pair_adhesion"},
    "tendon": {
        "native_tendon_stiffnesspoly": "tendon_stiffnesspoly",
        "native_tendon_dampingpoly": "tendon_dampingpoly",
    },
    "actuator": {"native_actuator_delay": "actuator_delay", "native_actuator_interp": "actuator_history"},
}
_NATIVE_BATCHED_MODEL_FIELDS = tuple(field for fields in _NATIVE_WORLD_FIELDS.values() for field in fields.values())


def _native_pair_rows(model, native, geom_shapes):
    rows = np.full((len(geom_shapes), native.npair), -1, dtype=np.int32)
    if not model.custom_frequency_counts.get("mujoco:pair", 0):
        return rows
    groups = {}
    for row, (world, first, second) in enumerate(
        zip(
            model.mujoco.pair_world.numpy(),
            model.mujoco.pair_geom1.numpy(),
            model.mujoco.pair_geom2.numpy(),
            strict=True,
        )
    ):
        groups.setdefault(world, {})[first, second] = row
    expected = {tuple(sorted(pair)) for pair in zip(native.pair_geom1, native.pair_geom2, strict=True)}
    for world, shapes in enumerate(geom_shapes):
        slots = {shape: geom for geom, shape in enumerate(shapes) if shape >= 0}
        pairs = {
            tuple(sorted((slots[first], slots[second]))): row
            for (first, second), row in (groups.get(-1, {}) | groups.get(world, {})).items()
            if first in slots and second in slots
        }
        if pairs.keys() != expected:
            raise ValueError(
                f"SolverMuJoCo requires matching explicit contact pairs across worlds; world {world} differs."
            )
        for pair, endpoints in enumerate(zip(native.pair_geom1, native.pair_geom2, strict=True)):
            rows[world, pair] = pairs[tuple(sorted(endpoints))]
    return rows


def _validate_native_tendon_paths(model, tendon_rows):
    if tendon_rows is None or not tendon_rows.size:
        return
    attrs = model.mujoco
    fixed = attrs.native_tendon_fixed.numpy()
    if not np.any(fixed):
        return
    starts, counts = attrs.tendon_joint_adr.numpy(), attrs.tendon_joint_num.numpy()
    joints, coefs = attrs.tendon_joint.numpy(), attrs.tendon_coef.numpy()
    dofs = attrs.native_tendon_joint_dof.numpy()
    dofs = np.where(dofs >= 0, dofs, model.joint_qd_start.numpy()[joints])
    per_world = model.joint_dof_count // len(tendon_rows)
    for tendon, rows in enumerate(tendon_rows.T):
        reference = None
        for world, row in enumerate(rows):
            if row < 0 or not fixed[row]:
                path = None
            else:
                entries = slice(starts[row], starts[row] + counts[row])
                path = tuple(zip((dofs[entries] - world * per_world).tolist(), coefs[entries].tolist(), strict=True))
            if world == 0:
                reference = path
            elif path != reference:
                raise ValueError(
                    f"SolverMuJoCo requires matching fixed tendon paths across worlds; tendon {tendon}, world {world} differs."
                )


def _publish_native_world_parameters(
    model, native, mjw_model, joint_dofs, geom_shapes, tendon_rows, actuator_rows, shape_world_mapping
):
    import mujoco

    nworld = len(joint_dofs)
    if nworld <= 1:
        return
    if actuator_rows is not None:
        if np.any(actuator_rows >= 0):
            count = len(model.mujoco.actuator_world) // nworld
            actuator_rows = np.where(actuator_rows >= 0, actuator_rows + np.arange(nworld)[:, None] * count, -1)
            attrs = model.mujoco
            valid = actuator_rows >= 0
            rows = np.maximum(actuator_rows, 0)
            for name in ("actuator_trntype", "ctrl_source", "ctrl_type"):
                values = getattr(attrs, name).numpy()[rows]
                if np.any(valid & (values != values[0])):
                    raise ValueError("SolverMuJoCo requires matching actuator declarations across worlds.")
            transmissions = attrs.actuator_trntype.numpy()[rows]
            targets = attrs.actuator_trnid.numpy()[rows].copy()
            for kinds, frequency, columns in (
                ((TrnType.JOINT, TrnType.JOINT_IN_PARENT), "joint_dof", (0,)),
                ((TrnType.BODY,), "body", (0,)),
                ((TrnType.SITE, TrnType.SLIDERCRANK), "shape", (0, 1)),
                ((TrnType.TENDON,), "tendon", (0,)),
            ):
                mask = valid & np.isin(transmissions, kinds)
                if not np.any(mask):
                    continue
                if frequency == "shape":
                    canonical = np.arange(model.shape_count)
                    for shape, world_shapes in shape_world_mapping.items():
                        canonical[world_shapes[world_shapes >= 0]] = shape
                    for column in columns:
                        selected = mask & (targets[..., column] >= 0)
                        indices = np.where(selected, targets[..., column], 0)
                        targets[..., column] = np.where(selected, canonical[indices], targets[..., column])
                    continue
                starts = (
                    np.arange(nworld) * (len(attrs.tendon_world) // nworld)
                    if frequency == "tendon"
                    else getattr(model, f"{frequency}_world_start").numpy()[:nworld]
                )
                for column in columns:
                    targets[..., column] -= np.where(mask & (targets[..., column] >= 0), starts[:, None], 0)
            if np.any(valid[..., None] & (targets != targets[0])):
                raise ValueError("SolverMuJoCo requires matching actuator targets and declaration order across worlds.")
        else:
            actuator_rows = None
    pair_rows = _native_pair_rows(model, native, geom_shapes)
    _validate_native_tendon_paths(model, tendon_rows)
    if tendon_rows is not None and np.any(tendon_rows >= 0):
        rows = np.maximum(tendon_rows, 0)
        flags = model.mujoco.tendon_limited.numpy()[rows]
        ranges = model.mujoco.tendon_range.numpy()[rows]
        limited = np.where(flags == mujoco.mjtLimited.mjLIMITED_AUTO, np.any(ranges != 0, axis=-1), flags != 0)
        if np.any((tendon_rows >= 0) & (limited != native.tendon_limited)):
            raise ValueError("SolverMuJoCo requires matching tendon limit flags across worlds.")
    if np.any(joint_dofs >= 0):
        lower = model.joint_limit_lower.numpy()[np.maximum(joint_dofs, 0)]
        upper = model.joint_limit_upper.numpy()[np.maximum(joint_dofs, 0)]
        limited = ~((lower <= -MAXVAL) & (upper >= MAXVAL))
        scalars = (joint_dofs >= 0) & np.isin(native.jnt_type, (_JointType.SLIDE, _JointType.HINGE))
        if np.any(scalars & (limited != native.jnt_limited)):
            raise ValueError("SolverMuJoCo requires matching scalar joint limit flags across worlds.")
        source_types = model.mujoco.native_joint_type.numpy()[np.maximum(joint_dofs, 0)]
        joint_dofs = np.where(source_types == native.jnt_type, joint_dofs, -1)
        flags = model.mujoco.native_joint_limited.numpy()[np.maximum(joint_dofs, 0)]
        ranges = model.mujoco.native_joint_range.numpy()[np.maximum(joint_dofs, 0)]
        limited = np.where(flags == mujoco.mjtLimited.mjLIMITED_AUTO, np.any(ranges != 0, axis=-1), flags != 0)
        balls = (joint_dofs >= 0) & (native.jnt_type == _JointType.BALL)
        if np.any(balls & (limited != native.jnt_limited)):
            raise ValueError("SolverMuJoCo requires matching ball joint limit flags across worlds.")
    mappings = {
        "joint": joint_dofs,
        "dof": joint_dofs[:, native.dof_jntid],
        "geom": geom_shapes,
        "pair": pair_rows,
        "tendon": tendon_rows,
        "actuator": actuator_rows,
    }
    for kind, names in (
        ("geom", ("native_geom_fluid_ellipsoid", "native_geom_fluid_coefs")),
        ("actuator", ("native_actuator_group", "native_actuator_nsample")),
    ):
        rows = mappings[kind]
        if rows is None or not np.any(rows >= 0):
            continue
        valid = rows >= 0
        for name in names:
            values = getattr(model.mujoco, name).numpy()[np.maximum(rows, 0)]
            reference = values[np.argmax(valid, axis=0), np.arange(rows.shape[1])]
            if np.any(values[valid] != np.broadcast_to(reference, values.shape)[valid]):
                raise ValueError(
                    "SolverMuJoCo with separate_worlds=True requires "
                    f"identical native template parameters (mujoco:{name}) across worlds."
                )

    for kind, fields in _NATIVE_WORLD_FIELDS.items():
        rows = mappings[kind]
        if rows is None or not np.any(rows >= 0):
            continue
        valid = rows >= 0
        for name, field in fields.items():
            attribute = getattr(model.mujoco, name)
            values = attribute.numpy()[np.maximum(rows, 0)]
            output = getattr(mjw_model, field)
            compiled = output.numpy().copy()
            target = compiled[..., 1] if field == "actuator_history" else compiled
            target[valid] = values[valid]
            output.assign(compiled)
            if field in ("geom_adhesion", "pair_adhesion"):
                mjw_model.flg_adhesion |= bool(np.any(compiled))
            elif field == "geom_surfacevel":
                mjw_model.flg_surfacevel |= bool(np.any(compiled))
            elif kind == "tendon":
                flag = "has_tendon_stiffness" if field == "tendon_stiffnesspoly" else "has_tendon_damping"
                setattr(mjw_model, flag, getattr(mjw_model, flag) or bool(np.any(compiled)))


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
