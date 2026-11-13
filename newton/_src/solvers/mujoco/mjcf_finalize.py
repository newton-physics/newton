# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Resolve imported MJCF parameters from the composed Newton model."""

from __future__ import annotations

import numpy as np
import warp as wp

from ...core.types import MAXVAL
from ...sim import ModelBuilder
from ...sim.model import Model
from .constants import FLOAT64_EPS, MJ_MINVAL
from .enums import CtrlSource, TrnType, _ActuatorDynamicsType, _joint_dof_count, _JointType
from .reference import build_ref_q, compute_body_poses_at_qref


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
    state = State()
    state.joint_q = build_ref_q(model, wp.clone(model.joint_q))
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
    inertia = len(inverse_weights) / max(MJ_MINVAL, np.sum(inverse_weights))
    timeconst, ratio = map(float, springdamper)
    return (
        inertia / max(MJ_MINVAL, timeconst * timeconst * ratio * ratio),
        2 * inertia / max(MJ_MINVAL, timeconst),
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
        width = _joint_dof_count(native_types[dof])
        span = slice(dof, dof + width)
        auto_ke = stiffness[span] == baseline[span, 0]
        auto_kd = damping[span] == baseline[span, 1]
        spring_rows.append((dof, width, auto_ke, auto_kd))
        if np.any(auto_ke) or np.any(auto_kd):
            requested.update(range(dof, dof + width))
            if native_types[dof] == _JointType.FREE:
                free_joints.append(int(np.searchsorted(starts, dof, side="right") - 1))
    bias = None
    overrides = attrs.native_target_kd_override.numpy()
    if model.custom_frequency_counts.get("mujoco:actuator", 0):
        target_baseline = attrs.native_actuator_target_kd.numpy()
        targets, sources = attrs.actuator_trnid.numpy()[:, 0], attrs.ctrl_source.numpy()
        bias = attrs.actuator_biasprm.numpy()
        for row in np.flatnonzero(target_baseline >= 0):
            dof = int(targets[row])
            if sources[row] != CtrlSource.JOINT_TARGET or dof < 0 or bias[row, 2] <= 0:
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
        inertia = np.divide(1.0, weights[resolved], out=np.zeros(len(resolved)), where=weights[resolved] > FLOAT64_EPS)
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


def _finalize_native_actuator_ranges(builder: ModelBuilder, model: Model) -> None:

    attrs = model.mujoco
    factors = getattr(attrs, "native_actuator_inheritrange", None)
    if factors is None:
        return
    factors = factors.numpy()
    if not np.any(factors > 0):
        return
    targets = attrs.actuator_trnid.numpy()[:, 0]
    transmissions = attrs.actuator_trntype.numpy()
    tendon_ranges = attrs.tendon_range.numpy() if np.any(transmissions == TrnType.TENDON) else None
    lower, upper = model.joint_limit_lower.numpy(), model.joint_limit_upper.numpy()
    refs, source_bounds = attrs.dof_ref.numpy(), attrs.native_actuator_inheritrange_bounds.numpy()
    dynamics = attrs.actuator_dyntype.numpy()
    control, activation = attrs.actuator_ctrlrange.numpy(), attrs.actuator_actrange.numpy()
    for row in np.flatnonzero(factors > 0):
        dof = targets[row]
        if transmissions[row] == TrnType.JOINT:
            bounds = (
                source_bounds[row]
                if lower[dof] <= -MAXVAL and upper[dof] >= MAXVAL
                else np.array([lower[dof], upper[dof]]) + refs[dof]
            )
        elif transmissions[row] == TrnType.TENDON:
            bounds = tendon_ranges[targets[row]]
        else:
            continue
        center, radius = np.mean(bounds), (bounds[1] - bounds[0]) * factors[row] / 2
        ranges = activation if dynamics[row] == _ActuatorDynamicsType.INTEGRATOR else control
        ranges[row] = [center - radius, center + radius]
        factors[row] = 0
    attrs.actuator_ctrlrange.assign(control)
    attrs.actuator_actrange.assign(activation)
    attrs.native_actuator_inheritrange.assign(factors)


def _finalize_native_tendon_spring_lengths(builder: ModelBuilder, model: Model) -> None:
    attrs = model.mujoco
    lengths = attrs.tendon_springlength.numpy()
    rows = np.flatnonzero(attrs.native_tendon_fixed.numpy() & np.all(lengths == -1, axis=1))
    if not len(rows):
        return
    starts, counts = attrs.tendon_joint_adr.numpy(), attrs.tendon_joint_num.numpy()
    joints, coefficients = attrs.tendon_joint.numpy(), attrs.tendon_coef.numpy()
    dofs = attrs.native_tendon_joint_dof.numpy()
    q_starts, qd_starts = model.joint_q_start.numpy(), model.joint_qd_start.numpy()
    rest, refs = model.joint_rest_q.numpy(), attrs.dof_ref.numpy()
    for row in rows:
        length = 0.0
        for entry in range(starts[row], starts[row] + counts[row]):
            joint = joints[entry]
            dof = dofs[entry] if dofs[entry] >= 0 else qd_starts[joint]
            q = q_starts[joint] + dof - qd_starts[joint]
            length += float(coefficients[entry]) * (float(rest[q]) + float(refs[dof]))
        lengths[row] = length
    attrs.tendon_springlength.assign(lengths)


def _finalize_native_weld_poses(builder: ModelBuilder, model: Model) -> None:
    from ...sim.enums import JointType  # noqa: PLC0415
    from .equality import MjcEqualityTargetKind  # noqa: PLC0415
    from .utils import mjc_loop_joint_xforms  # noqa: PLC0415

    automatic = getattr(model.mujoco, "native_equality_auto_relpose", None)
    if automatic is None:
        return
    automatic = automatic.numpy()
    if not np.any(automatic):
        return
    poses = model.mujoco.equality_constraint_relpose.numpy()
    qref = build_ref_q(model)
    q = qref.numpy()
    bodies = model.body_q.numpy()
    parent_frames, child_frames = model.joint_X_p.numpy(), model.joint_X_c.numpy()
    parents, children = model.joint_parent.numpy(), model.joint_child.numpy()
    starts = model.joint_q_start.numpy()
    # Free-body reference poses come from body_q, independently of the initial joint state.
    for joint in np.flatnonzero(model.joint_type.numpy() == JointType.FREE):
        parent = wp.transform(*parent_frames[joint])
        if parents[joint] >= 0:
            parent = wp.transform(*bodies[parents[joint]]) * parent
        q[starts[joint] : starts[joint] + 7] = (
            wp.transform_inverse(parent) * wp.transform(*bodies[children[joint]]) * wp.transform(*child_frames[joint])
        )
    qref.assign(q)
    bodies = compute_body_poses_at_qref(model, qref, ref_body_q=wp.clone(model.body_q)).numpy()
    body1 = model.mujoco.equality_constraint_body1.numpy()
    body2 = model.mujoco.equality_constraint_body2.numpy()
    anchors = model.mujoco.equality_constraint_anchor.numpy()
    kinds = model.mujoco.equality_constraint_target_kind.numpy()
    targets = model.mujoco.equality_constraint_target.numpy()
    for row in np.flatnonzero(automatic):
        first = wp.transform(*bodies[body1[row]]) if body1[row] >= 0 else wp.transform_identity()
        second = wp.transform(*bodies[body2[row]]) if body2[row] >= 0 else wp.transform_identity()
        anchor = wp.vec3(anchors[row])
        poses[row] = wp.transform_inverse(first) * second * wp.transform(anchor, wp.quat_identity())
        joint = targets[row]
        if kinds[row] == MjcEqualityTargetKind.JOINT and joint >= 0:
            _, _, parent_frames[joint], child_frames[joint] = mjc_loop_joint_xforms(
                builder, int(body1[row]), int(body2[row]), anchor, wp.transform(*poses[row]), weld=True
            )
        automatic[row] = False
    model.mujoco.equality_constraint_relpose.assign(poses)
    model.mujoco.native_equality_auto_relpose.assign(automatic)
    model.joint_X_p.assign(parent_frames)
    model.joint_X_c.assign(child_frames)
