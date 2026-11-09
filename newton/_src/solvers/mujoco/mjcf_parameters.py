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
from .constants import MJ_MINVAL

if TYPE_CHECKING:
    import mujoco


_NATIVE_JOINT_FIELDS = {
    "type": (wp.int32, -1),
    "springdamper": (wp.vec2, (0.0, 0.0)),
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
    "timestep": (wp.float32, 0.002),
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


def _apply_native_element_parameters(model, spec, shapes, actuator_indices):
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
        ("native_actuator_target_kd", "mujoco:actuator", wp.float32, -1.0, None),
        ("native_joint_gain_baseline", Model.AttributeFrequency.JOINT_DOF, wp.vec2, (-1.0, -1.0), None),
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


def _native_inertia_exclusions(model: Model) -> set[int]:
    excluded = set()
    for kind in ("actuator", "tendon"):
        armature = getattr(model.mujoco, f"{kind}_armature", None)
        if armature is None:
            continue
        rows = np.flatnonzero(armature.numpy() != 0)
        if len(rows):
            owners = getattr(model.mujoco, f"{kind}_articulation").numpy()[rows]
            if np.any(owners < 0):
                return set(range(model.articulation_count))
            excluded.update(map(int, owners))
    return excluded


def _reference_inverse_weights(model: Model, dofs: set[int], free_joints: list[int]) -> np.ndarray:
    from ...sim.articulation import _iter_mass_matrix_blocks, eval_fk  # noqa: PLC0415
    from ...sim.enums import JointType  # noqa: PLC0415
    from ...sim.state import State  # noqa: PLC0415

    weights = np.full(model.joint_dof_count, np.nan)
    if not dofs or model._fk_articulation_level_start is None:
        return weights
    starts = model.joint_qd_start.numpy()
    owners = model.joint_articulation.numpy()
    excluded = _native_inertia_exclusions(model)
    pending = {}
    for dof in sorted(dofs):
        joint = int(np.searchsorted(starts, dof, side="right") - 1)
        articulation = int(owners[joint])
        if articulation >= 0 and articulation not in excluded:
            pending.setdefault(articulation, []).append(dof)
    if not pending:
        return weights
    q = model.joint_q.numpy().copy()
    q_starts = model.joint_q_start.numpy()
    for joint, kind in enumerate(model.joint_type.numpy()):
        if kind == JointType.BALL:
            q[q_starts[joint] : q_starts[joint + 1]] = [0, 0, 0, 1]
        elif kind in (JointType.PRISMATIC, JointType.REVOLUTE, JointType.D6):
            q[q_starts[joint] : q_starts[joint + 1]] = 0
    state = State()
    state.joint_q = wp.array(q, dtype=float, device=model.device)
    state.body_q = wp.empty(model.body_count, dtype=wp.transform, device=model.device)
    state.body_qd = wp.empty(model.body_count, dtype=wp.spatial_vector, device=model.device)
    eval_fk(model, state.joint_q, model.joint_qd, state)
    origin_rows = {}
    if free_joints:
        poses, coms = state.body_q.numpy(), model.body_com.numpy()
        parents, children = model.joint_parent.numpy(), model.joint_child.numpy()
        anchors = model.joint_X_p.numpy()
        for joint in free_joints:
            parent = wp.transform(*anchors[joint])
            if parents[joint] >= 0:
                parent = wp.transform(*poses[parents[joint]]) * parent
            offset = wp.transform_vector(wp.transform(*poses[children[joint]]), wp.vec3(coms[children[joint]]))
            x, y, z = wp.transform_vector(wp.transform_inverse(parent), offset)
            # MuJoCo's free-joint translation is at the body origin; Newton's is at its COM.
            cross = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
            for axis in range(3):
                origin_rows[int(starts[joint]) + axis] = (int(starts[joint]) + 3, cross[axis])
    articulation_starts = model.articulation_start.numpy()
    for indices, matrices in _iter_mass_matrix_blocks(model, state, list(pending)):
        for articulation, matrix in zip(indices, matrices, strict=True):
            requested = pending[articulation]
            offset = starts[articulation_starts[articulation]]
            columns = np.array(requested) - offset
            rhs = np.zeros((len(matrix), len(columns)))
            rhs[columns, np.arange(len(columns))] = 1
            for column, dof in enumerate(requested):
                if dof in origin_rows:
                    angular, cross = origin_rows[dof]
                    rhs[angular - offset : angular - offset + 3, column] += cross
            try:
                factor = np.linalg.cholesky(matrix.astype(np.float64))
                if np.min(np.diag(factor)) ** 2 <= 1e-6 * np.max(np.diag(matrix)):
                    continue
                solved = np.linalg.solve(factor, rhs)
            except np.linalg.LinAlgError:
                continue
            result = np.sum(solved * solved, axis=0)
            weights[requested] = np.where(np.isfinite(result) & (result > 0), result, np.nan)
    return weights


def _spring_gains(springdamper: np.ndarray, inverse_weights: np.ndarray) -> tuple[float, float]:
    import mujoco

    inertia = len(inverse_weights) / max(mujoco.mjMINVAL, np.sum(inverse_weights))
    timeconst, ratio = map(float, springdamper)
    return (
        inertia / max(mujoco.mjMINVAL, timeconst * timeconst * ratio * ratio),
        2 * inertia / max(mujoco.mjMINVAL, timeconst),
    )


def _finalize_native_core_parameters(builder: ModelBuilder, model: Model) -> None:
    from ...sim.enums import JointTargetMode, JointType  # noqa: PLC0415

    attrs = model.mujoco
    types, starts = model.joint_type.numpy(), model.joint_qd_start.numpy()
    native_types = attrs.native_joint_type.numpy()
    springs = attrs.native_joint_springdamper.numpy()
    baseline = attrs.native_joint_gain_baseline.numpy()
    stiffness, damping = model.joint_stiffness.numpy(), model.joint_damping.numpy()
    ke, kd = model.joint_target_ke.numpy(), model.joint_target_kd.numpy()
    modes = model.joint_target_mode.numpy()
    spring_rows = []
    target_rows = []
    requested = set()
    free_joints = []
    for dof in np.flatnonzero(np.all(springs > 0, axis=1)):
        width = 6 if native_types[dof] == 0 else 3 if native_types[dof] == 1 else 1
        span = slice(dof, dof + width)
        auto_ke = stiffness[span] == baseline[span, 0]
        auto_kd = damping[span] == baseline[span, 1]
        spring_rows.append((dof, width, auto_ke, auto_kd))
        if np.any(auto_ke) or np.any(auto_kd):
            requested.update(range(dof, dof + width))
            if native_types[dof] == 0:
                free_joints.append(int(np.searchsorted(starts, dof, side="right") - 1))
    bias = None
    overrides = attrs.native_target_kd_override.numpy()
    if model.custom_frequency_counts.get("mujoco:actuator", 0):
        target_baseline = attrs.native_actuator_target_kd.numpy()
        targets, sources = attrs.actuator_trnid.numpy()[:, 0], attrs.ctrl_source.numpy()
        bias = attrs.actuator_biasprm.numpy()
        for row in np.flatnonzero(target_baseline >= 0):
            dof = int(targets[row])
            if sources[row] != 0 or dof < 0 or bias[row, 2] <= 0:
                continue
            joint = int(np.searchsorted(starts, dof, side="right") - 1)
            width = 3 if types[joint] == JointType.BALL else 1
            axes = np.arange(dof, dof + width)
            if np.any(modes[axes] != JointTargetMode.POSITION):
                continue
            automatic = kd[axes] == target_baseline[row]
            overrides[axes] = ~automatic
            target_rows.append((row, axes, automatic))
            requested.update(map(int, axes[automatic]))
    weights = _reference_inverse_weights(model, requested, free_joints)
    for dof, width, auto_ke, auto_kd in spring_rows:
        axes = np.arange(dof, dof + width)
        if not np.any(auto_ke) and not np.any(auto_kd):
            springs[dof] = 0
        elif np.all(np.isfinite(weights[axes])):
            resolved_ke, resolved_kd = _spring_gains(springs[dof], weights[axes])
            stiffness[axes[auto_ke]] = resolved_ke
            damping[axes[auto_kd]] = resolved_kd
            springs[dof] = 0
    for row, axes, automatic in target_rows:
        pending = axes[automatic]
        valid = np.isfinite(weights[pending]) & np.isfinite(ke[pending]) & (ke[pending] >= 0)
        kd[pending] = 0
        resolved = pending[valid]
        inertia = np.divide(1.0, weights[resolved], out=np.zeros(len(resolved)), where=weights[resolved] > MJ_MINVAL)
        kd[resolved] = 2 * bias[row, 2] * np.sqrt(ke[resolved] * inertia)
        if np.all(valid):
            bias[row, 2] = -kd[axes[0]]
    model.joint_stiffness.assign(stiffness)
    model.joint_damping.assign(damping)
    model.joint_target_kd.assign(kd)
    attrs.native_joint_springdamper.assign(springs)
    attrs.native_target_kd_override.assign(overrides)
    if bias is not None:
        attrs.actuator_biasprm.assign(bias)


def _apply_native_joint_parameters(model: Model, spec: mujoco.MjSpec, dof_to_joint) -> None:
    import mujoco

    attrs = model.mujoco
    if not hasattr(attrs, "native_joint_type"):
        return
    fields = {name: getattr(attrs, f"native_joint_{name}").numpy() for name in _NATIVE_JOINT_FIELDS}
    ranges = attrs.dof_limit_range.numpy()
    stiffness = model.joint_stiffness.numpy()
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


def _resolve_native_joint_parameters(solver, spec: mujoco.MjSpec) -> None:
    from ...sim.enums import ModelFlags  # noqa: PLC0415

    model, native = solver.model, solver.mj_model
    spring_attr = getattr(getattr(model, "mujoco", None), "native_joint_springdamper", None)
    if spring_attr is None:
        return
    springs = spring_attr.numpy()
    if not np.any(np.all(springs > 0, axis=1)):
        return
    types = model.mujoco.native_joint_type.numpy()
    mapping = solver.mjc_jnt_to_newton_dof.numpy()
    rows = [
        (world, joint, int(dof))
        for world in range(1 if solver.use_mujoco_cpu else len(mapping))
        for joint, dof in enumerate(mapping[world])
        if dof >= 0 and types[dof] == native.jnt_type[joint] and np.all(springs[dof] > 0)
    ]
    if not rows:
        return

    # Kinematic locking changes constraint weights, but spring gains use physical inertia.
    locked = solver._kinematic_weld_eq is not None
    try:
        if locked:
            solver._set_const_0(lock_kinematic=False)
        weights = (
            native.dof_invweight0[None].copy() if solver.use_mujoco_cpu else solver.mjw_model.dof_invweight0.numpy()
        )
    finally:
        if locked:
            solver._set_const_0()

    stiffness, damping = model.joint_stiffness.numpy(), model.joint_damping.numpy()
    baseline = model.mujoco.native_joint_gain_baseline.numpy()
    for world, joint, dof in rows:
        width = 6 if native.jnt_type[joint] == 0 else 3 if native.jnt_type[joint] == 1 else 1
        adr = native.jnt_dofadr[joint]
        ke, kd = _spring_gains(springs[dof], weights[world, adr : adr + width])
        span = slice(dof, dof + width)
        stiffness[span] = np.where(stiffness[span] == baseline[span, 0], ke, stiffness[span])
        damping[span] = np.where(damping[span] == baseline[span, 1], kd, damping[span])
        springs[dof] = 0
        if world == 0:
            spec.joints[joint].springdamper[:] = 0
            spec.joints[joint].stiffness[0] = stiffness[dof]
            spec.joints[joint].damping[0] = damping[dof]
    model.joint_stiffness.assign(stiffness)
    model.joint_damping.assign(damping)
    spring_attr.assign(springs)
    solver.notify_model_changed(ModelFlags.JOINT_DOF_FORCE_PROPERTIES)
    native.jnt_stiffness[:] = solver.mjw_model.jnt_stiffness.numpy()[0]
    native.dof_damping[:] = solver.mjw_model.dof_damping.numpy()[0]


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
    refs, native_ranges = attrs.dof_ref.numpy(), attrs.dof_limit_range.numpy()
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


def _apply_native_parameters(model, spec, dof_to_joint, shapes, actuator_indices):
    if not hasattr(model, "mujoco") or not hasattr(model.mujoco, "native_option_enableflags"):
        return False
    _apply_native_joint_parameters(model, spec, dof_to_joint)
    _apply_native_element_parameters(model, spec, shapes, actuator_indices)
    return True


def _sync_native_parameters(model, native, equality_rows):
    _sync_native_equality_parameters(model, native, equality_rows)
    _sync_native_actuator_ranges(model)
