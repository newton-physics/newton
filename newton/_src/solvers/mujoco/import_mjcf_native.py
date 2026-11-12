# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Convert native MuJoCo models directly into Newton builder data."""

from __future__ import annotations

import logging
import os
import re
import warnings
import xml.etree.ElementTree as ET
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any

import numpy as np
import warp as wp

from ...core import quat_between_axes
from ...core.types import Axis, AxisType, Transform
from ...exceptions import NewtonWarning
from ...geometry import GeoType, Mesh, ShapeFlags
from ...geometry.types import Heightfield
from ...sim import JointTargetMode, JointType, ModelBuilder
from ...sim.model import Model
from ...usd.schemas import solref_to_stiffness_damping
from ...utils.import_utils import (
    clamp_imported_opacity,
    collapse_massless_fixed_root_joints,
    sanitize_name,
    sanitize_xml_content,
)
from . import dcmotor
from .collision_masks import (
    add_mjcf_body_collision_filters,
    compile_collision_masks,
    set_mjcf_joint_collision_filter,
)
from .constants import SOLREF_MODE_FORCE_SPACE, SOLREF_MODE_MJCF_DEFAULT, SOLREF_MODE_RAW
from .enums import EqType
from .equality import _add_equality_constraint
from .mjcf_parameters import (
    _finalize_native_actuator_ranges,
    _finalize_native_core_parameters,
    _finalize_native_tendon_spring_lengths,
    _finalize_native_weld_poses,
    _read_native_parameters,
    _register_native_attributes,
)
from .solver_mujoco import SolverMuJoCo
from .utils import mjc_add_equality_loop_joint, mjc_add_equality_mimic, mjc_polycoef_has_higher_order

if TYPE_CHECKING:
    import mujoco

logger = logging.getLogger(__name__)


def _extract_custom_attributes(builder, spec, scale, parse_options):
    frequency = Model.AttributeFrequency
    sources = {
        frequency.BODY: spec.bodies[1:],
        frequency.SHAPE: [*spec.geoms, *spec.sites],
        frequency.JOINT: spec.joints,
        frequency.JOINT_DOF: spec.joints,
        frequency.WORLD: [spec] if parse_options else [],
        frequency.ONCE: [spec] if parse_options else [],
        "mujoco:actuator": spec.actuators,
        "mujoco:tendon": spec.tendons,
        "mujoco:pair": spec.pairs,
        "mujoco:equality_constraint": spec.equalities,
    }
    values = {}
    for attr in builder.custom_attributes.values():
        hook = attr.mjcf_native_value_transformer
        if hook is None:
            continue
        if attr.frequency not in sources:
            raise ValueError(f"Native MJCF extraction does not support frequency {attr.frequency!r} for {attr.key!r}.")
        for element in sources[attr.frequency]:
            value = hook(element, {"spec": spec, "attr": attr, "scale": scale})
            if value is not None:
                # Compilation and scaling mutate native arrays in place.
                if attr.dtype is not str:
                    value = np.array(value, copy=True)
                    if value.ndim == 0:
                        value = value.item()
                values.setdefault((attr.frequency, element), {})[attr.key] = value

    def lookup(frequency, element):
        return values.get((frequency, element), {})

    return lookup


def _transform(pos: np.ndarray, quat: np.ndarray) -> wp.transform:
    return wp.transform(pos, wp.quat(*quat[1:], quat[0]))


def _scale_spec(spec: mujoco.MjSpec, scale: float) -> None:
    import mujoco

    for body in spec.bodies:
        body.pos[:] *= scale
        body.ipos[:] *= scale
        body.inertia[:] *= scale**2
        body.fullinertia[:] *= scale**2
    for frame in spec.frames:
        frame.pos[:] *= scale
    for element in [*spec.geoms, *spec.sites]:
        element.pos[:] *= scale
        element.size[:] *= scale
        element.fromto[:] *= scale
    for geom in spec.geoms:
        geom.margin *= scale
        geom.gap *= scale
    for mesh in spec.meshes:
        mesh.scale[:] *= scale
    for hfield in spec.hfields:
        hfield.size[:] *= scale
    for joint in spec.joints:
        joint.pos[:] *= scale
        if joint.type == mujoco.mjtJoint.mjJNT_SLIDE:
            joint.range[:] *= scale
            joint.ref *= scale
            joint.springref *= scale
            joint.margin *= scale
    linear_tendons = set()
    for tendon in spec.tendons:
        if any(w.type != mujoco.mjtWrap.mjWRAP_JOINT for w in tendon.path) or all(
            w.target.type == mujoco.mjtJoint.mjJNT_SLIDE for w in tendon.path
        ):
            linear_tendons.add(tendon.name)
            tendon.range[:] *= scale
            tendon.margin *= scale
            tendon.springlength[tendon.springlength >= 0] *= scale
            tendon.width *= scale
    for actuator in spec.actuators:
        actuator.cranklength *= scale
        linear = False
        if actuator.trntype in (mujoco.mjtTrn.mjTRN_JOINT, mujoco.mjtTrn.mjTRN_JOINTINPARENT):
            linear = spec.joint(actuator.target).type == mujoco.mjtJoint.mjJNT_SLIDE
        elif actuator.trntype == mujoco.mjtTrn.mjTRN_TENDON:
            linear = actuator.target in linear_tendons
        elif actuator.trntype == mujoco.mjtTrn.mjTRN_SLIDERCRANK:
            linear = True
        elif actuator.trntype == mujoco.mjtTrn.mjTRN_SITE:
            linear = bool(actuator.refsite) and not np.any(actuator.gear[3:])
        if linear:
            if actuator.lengthrange[0] < actuator.lengthrange[1]:
                actuator.lengthrange[:] *= scale
            if (
                actuator.gaintype == mujoco.mjtGain.mjGAIN_FIXED
                and actuator.biastype == mujoco.mjtBias.mjBIAS_AFFINE
                and _scalar_servo_mode(actuator.gainprm, actuator.biasprm) != JointTargetMode.NONE
            ):
                actuator.ctrlrange[:] *= scale
                actuator.actrange[:] *= scale
            elif actuator.gaintype == mujoco.mjtGain.mjGAIN_DCMOTOR and actuator.ctrlspec in (1, 2):
                actuator.ctrlrange[:] *= scale
    for pair in spec.pairs:
        pair.margin *= scale
        pair.gap *= scale
    for equality in spec.equalities:
        if equality.type == mujoco.mjtEq.mjEQ_JOINT:
            first = spec.joint(equality.name1)
            second = spec.joint(equality.name2) if equality.name2 else None
            s1 = scale if first is not None and first.type == mujoco.mjtJoint.mjJNT_SLIDE else 1.0
            s2 = scale if second is not None and second.type == mujoco.mjtJoint.mjJNT_SLIDE else 1.0
            equality.data[:5] *= s1 / np.power(s2, np.arange(5))
        elif equality.type == mujoco.mjtEq.mjEQ_TENDON:
            s1 = scale if equality.name1 in linear_tendons else 1.0
            s2 = scale if equality.name2 in linear_tendons else 1.0
            equality.data[:5] *= s1 / np.power(s2, np.arange(5))
        if equality.type in (mujoco.mjtEq.mjEQ_CONNECT, mujoco.mjtEq.mjEQ_WELD):
            equality.data[:3] *= scale
        if equality.type == mujoco.mjtEq.mjEQ_WELD:
            equality.data[3:6] *= scale
            equality.data[10] *= scale


def _add_joints(builder, model, selected, parent, body, local, label, attributes, armature_scale, convert_ball):
    import mujoco

    if not selected:
        return builder.add_joint_fixed(parent, body, parent_xform=local, label=f"{label}/fixed"), {}
    first = selected[0]
    kind = int(model.jnt_type[first])
    name = "_".join(
        sanitize_name(model.joint(j).name or (f"{label.rsplit('/', 1)[-1]}_freejoint" if kind == 0 else f"joint_{j}"))
        for j in selected
    )
    # Match legacy's shared anchor before reordering scalar DOFs.
    anchor = wp.transform(model.jnt_pos[first], wp.quat_identity())
    if kind == int(mujoco.mjtJoint.mjJNT_FREE):
        joint = builder.add_joint_free(body, parent=parent, label=f"{label}/{name}")
        return joint, {first: builder.joint_qd_start[joint]}
    if kind == int(mujoco.mjtJoint.mjJNT_BALL):
        joint = builder.add_joint_ball(
            parent,
            body,
            parent_xform=local * anchor,
            child_xform=anchor,
            label=f"{label}/{name}",
            damping=float(model.dof_damping[model.jnt_dofadr[first]]),
            armature=float(model.dof_armature[model.jnt_dofadr[first]]) * armature_scale,
            friction=float(model.dof_frictionloss[model.jnt_dofadr[first]]),
            stiffness=float(model.jnt_stiffness[first]),
        )
        return joint, {first: builder.joint_qd_start[joint]}
    ordered = sorted(selected, key=lambda j: model.jnt_type[j] != int(mujoco.mjtJoint.mjJNT_SLIDE))
    linear, angular, custom = [], [], {}
    for offset, j in enumerate(ordered):
        d = int(model.jnt_dofadr[j])
        ref = float(model.qpos0[model.jnt_qposadr[j]])
        values = attributes(
            dof_ref=ref,
            limit_margin=model.jnt_margin[j],
            solimplimit=model.jnt_solimp[j],
            solreffriction=model.dof_solref[d],
            solimpfriction=model.dof_solimp[d],
            jnt_actgravcomp=bool(model.jnt_actgravcomp[j]),
        )
        for key, value in values.items():
            custom.setdefault(key, {})[offset] = value
        axis = ModelBuilder.JointDofConfig(
            axis=model.jnt_axis[j],
            damping=float(model.dof_damping[d]),
            stiffness=float(model.jnt_stiffness[j]),
            rest_q=float(model.qpos_spring[model.jnt_qposadr[j]]) - ref,
            armature=float(model.dof_armature[d]) * armature_scale,
            friction=float(model.dof_frictionloss[d]),
            actuator_mode=JointTargetMode.NONE,
            limit_lower=float(model.jnt_range[j, 0] - ref) if model.jnt_limited[j] else -1.0e10,
            limit_upper=float(model.jnt_range[j, 1] - ref) if model.jnt_limited[j] else 1.0e10,
            limit_ke=builder.default_joint_cfg.limit_ke,
            limit_kd=builder.default_joint_cfg.limit_kd,
            effort_limit=float(np.max(np.abs(model.jnt_actfrcrange[j]))) if model.jnt_actfrclimited[j] else 1.0e6,
        )
        (linear if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_SLIDE else angular).append(axis)
    joint_type = JointType.D6
    if len(ordered) == 1:
        joint_type = JointType.PRISMATIC if linear else JointType.REVOLUTE
    elif convert_ball and not linear and len(angular) == 3:
        joint_type = JointType.BALL
    joint = builder.add_joint(
        joint_type,
        parent,
        body,
        linear_axes=linear,
        angular_axes=angular,
        parent_xform=local * anchor,
        child_xform=anchor,
        label=f"{label}/{name}",
        custom_attributes=custom,
    )
    start = builder.joint_qd_start[joint]
    return joint, {j: start + offset for offset, j in enumerate(ordered)}


def _color_texture(model, material_id):
    import mujoco

    if material_id >= 0:
        for role in (mujoco.mjtTextureRole.mjTEXROLE_RGB, mujoco.mjtTextureRole.mjTEXROLE_RGBA):
            texture = int(model.mat_texid[material_id, role])
            if texture >= 0:
                return texture
    return -1


def _mesh(model, mesh_id, material_id, maxhullvert):
    face_start = int(model.mesh_faceadr[mesh_id])
    faces = model.mesh_face[face_start : face_start + model.mesh_facenum[mesh_id]]
    vertices = model.mesh_vert[model.mesh_vertadr[mesh_id] + faces.reshape(-1)].copy()
    normal_indices = model.mesh_facenormal[face_start : face_start + len(faces)].reshape(-1)
    normals = model.mesh_normal[model.mesh_normaladr[mesh_id] + normal_indices].copy()
    uvs = None
    if model.mesh_texcoordnum[mesh_id]:
        uv_indices = model.mesh_facetexcoord[face_start : face_start + len(faces)].reshape(-1)
        uvs = model.mesh_texcoord[model.mesh_texcoordadr[mesh_id] + uv_indices].copy()
    texture = None
    material = {}
    if material_id >= 0:
        for name in ("roughness", "metallic"):
            value = float(getattr(model, f"mat_{name}")[material_id])
            if value >= 0:
                material[name] = value
        t = _color_texture(model, material_id)
        if t >= 0:
            h, w, c = int(model.tex_height[t]), int(model.tex_width[t]), int(model.tex_nchannel[t])
            adr = int(model.tex_adr[t])
            texture = model.tex_data[adr : adr + h * w * c].reshape(h, w, c).copy()
        if uvs is not None:
            uvs *= model.mat_texrepeat[material_id]
    if uvs is not None:
        # Newton's viewers use the opposite image-row convention.
        uvs[:, 1] = 1.0 - uvs[:, 1]
    return Mesh(
        vertices=vertices,
        indices=np.arange(len(vertices)),
        normals=normals,
        uvs=uvs,
        texture=texture,
        compute_inertia=False,
        maxhullvert=maxhullvert,
        **material,
    )


def _import_options(builder, spec, attributes):
    import mujoco

    values = attributes(
        **_read_native_parameters("lengthrange", spec.compiler.LRopt),
        **_read_native_parameters("option", spec.option),
    )
    for key, value in values.items():
        builder.custom_attributes[key].values[0] = value
    builder.custom_attributes["mujoco:enable_sleeping"].values[0] = bool(
        spec.option.enableflags & int(mujoco.mjtEnableBit.mjENBL_SLEEP)
    )
    for name in SolverMuJoCo._ENABLE_FLAGS:
        builder.custom_attributes[f"mujoco:flag_{name}"].values[0] = bool(
            spec.option.enableflags & int(getattr(mujoco.mjtEnableBit, f"mjENBL_{name.upper()}"))
        )
    for name in SolverMuJoCo._DISABLE_FLAGS:
        builder.custom_attributes[f"mujoco:flag_{name}"].values[0] = not bool(
            spec.option.disableflags & int(getattr(mujoco.mjtDisableBit, f"mjDSBL_{name.upper()}"))
        )
    if spec.option.disableflags & int(mujoco.mjtDisableBit.mjDSBL_GRAVITY):
        if builder.current_world >= 0:
            builder.world_gravity[builder.current_world] = wp.vec3()
        else:
            builder.gravity = wp.vec3()

    for attr in builder.get_custom_attributes_by_frequency(
        [Model.AttributeFrequency.WORLD, Model.AttributeFrequency.ONCE]
    ):
        if attr.namespace != "mujoco":
            continue
        name = attr.mjcf_attribute_name
        if name and name != "*":
            source = spec.compiler if hasattr(spec.compiler, name) else spec.option
            if hasattr(source, name):
                value = getattr(source, name)
                # Enums in native structures already carry the solver's integer values.
                if isinstance(value, (str, bool, int, float, np.ndarray)) or hasattr(value, "__int__"):
                    if hasattr(value, "__int__") and not isinstance(value, (float, np.ndarray)):
                        value = int(value)
                    attr.values[builder._model_value_key(attr)] = attributes(**{attr.name: value})[attr.key]


def _scalar_servo_mode(gain, bias):
    if np.any(gain[1:]) or np.any(bias[3:]) or gain[0] <= 0 or bias[0] != 0:
        return JointTargetMode.NONE
    if bias[1] == -gain[0]:
        return JointTargetMode.POSITION
    if bias[1] == 0 and bias[2] == -gain[0]:
        return JointTargetMode.VELOCITY
    return JointTargetMode.NONE


def _joint_target(builder, model, i, dof, bias, values, pending_damping, *, dof_count=1):
    gain = model.actuator_gainprm[i]
    if (
        model.actuator_dyntype[i] != 0
        or model.actuator_gaintype[i] != 0
        or model.actuator_biastype[i] != 1
        or (dof_count == 1 and not np.array_equal(model.actuator_gear[i], [1, 0, 0, 0, 0, 0]))
    ):
        return
    mode = _scalar_servo_mode(gain, bias)
    if mode == JointTargetMode.NONE:
        return
    for axis in range(dof, dof + dof_count):
        previous = builder.joint_target_mode[axis]
        if previous != JointTargetMode.NONE:
            if {previous, mode} != {JointTargetMode.POSITION, JointTargetMode.VELOCITY}:
                return
            # Paired targets have no slot for additional position-servo damping.
            position_damping = -bias[2] if mode == JointTargetMode.POSITION else builder.joint_target_kd[axis]
            if position_damping != 0 or axis in pending_damping:
                return
    for axis in range(dof, dof + dof_count):
        previous = builder.joint_target_mode[axis]
        if mode == JointTargetMode.POSITION:
            builder.joint_target_ke[axis] = float(gain[0])
            if previous == JointTargetMode.NONE:
                builder.joint_target_kd[axis] = float(max(0.0, -bias[2]))
                if bias[2] > 0:
                    pending_damping.add(axis)
                    baseline = float(-model.actuator_biasprm[i, 2])
                    if dof_count == 3:
                        native_dof = model.jnt_dofadr[model.actuator_trnid[i, 0]]
                        baseline = float(2 * bias[2] * np.sqrt(gain[0] / model.dof_invweight0[native_dof]))
                    if np.isfinite(baseline) and baseline > 0:
                        builder.joint_target_kd[axis] = baseline
                        values["mujoco:native_actuator_target_kd"] = baseline
                        builder._add_model_finalizer(_finalize_native_core_parameters)
        else:
            builder.joint_target_kd[axis] = float(gain[0])
        builder.joint_target_mode[axis] = (
            mode if previous == JointTargetMode.NONE else JointTargetMode.POSITION_VELOCITY
        )
    values["mujoco:ctrl_source"] = int(SolverMuJoCo.CtrlSource.JOINT_TARGET)
    values["mujoco:ctrl_type"] = int(
        SolverMuJoCo.CtrlType.POSITION if mode == JointTargetMode.POSITION else SolverMuJoCo.CtrlType.VELOCITY
    )


def _tendons(builder, spec, model, joints, dofs, geoms, sites, attributes, custom):
    import mujoco

    mapping = {}
    for i in range(model.ntendon):
        adr, num = int(model.tendon_adr[i]), int(model.tendon_num[i])
        path = range(adr, adr + num)
        fixed = all(model.wrap_type[w] == int(mujoco.mjtWrap.mjWRAP_JOINT) for w in path)
        entries = []
        for w in path:
            kind, target = int(model.wrap_type[w]), int(model.wrap_objid[w])
            parameter = float(model.wrap_prm[w])
            if fixed:
                if target not in joints:
                    break
                entries.append(
                    attributes(
                        tendon_joint=joints[target],
                        tendon_coef=parameter,
                        native_tendon_joint_dof=dofs[target]
                        if builder.joint_type[joints[target]] == JointType.D6
                        else -1,
                    )
                )
            elif kind == int(mujoco.mjtWrap.mjWRAP_SITE):
                if target not in sites:
                    break
                entries.append(
                    attributes(
                        tendon_wrap_type=0,
                        tendon_wrap_shape=sites[target],
                        tendon_wrap_sidesite=-1,
                        tendon_wrap_prm=0.0,
                    )
                )
            elif kind == int(mujoco.mjtWrap.mjWRAP_PULLEY):
                entries.append(
                    attributes(
                        tendon_wrap_type=2, tendon_wrap_shape=-1, tendon_wrap_sidesite=-1, tendon_wrap_prm=parameter
                    )
                )
            else:
                if target not in geoms or (parameter >= 0 and int(parameter) not in sites):
                    break
                entries.append(
                    attributes(
                        tendon_wrap_type=1,
                        tendon_wrap_shape=geoms[target],
                        tendon_wrap_sidesite=sites.get(int(parameter), -1),
                        tendon_wrap_prm=0.0,
                    )
                )
        if len(entries) != num:
            warnings.warn(
                f"Tendon {model.tendon(i).name!r} references filtered elements; skipping.", NewtonWarning, stacklevel=2
            )
            continue
        if fixed:
            builder._add_model_finalizer(_finalize_native_tendon_spring_lengths)
        frequency = "mujoco:tendon_joint" if fixed else "mujoco:tendon_wrap"
        start = builder._custom_frequency_counts.get(frequency, 0)
        for entry in entries:
            builder.add_custom_values(**entry)
        values = attributes(
            tendon_label=model.tendon(i).name,
            tendon_world=builder.current_world,
            tendon_type=0 if fixed else 1,
            native_tendon_fixed=fixed,
            tendon_joint_adr=start if fixed else 0,
            tendon_joint_num=num if fixed else 0,
            tendon_wrap_adr=0 if fixed else start,
            tendon_wrap_num=0 if fixed else num,
            tendon_stiffness=model.tendon_stiffness[i],
            native_tendon_stiffnesspoly=spec.tendons[i].stiffness[1:],
            native_tendon_dampingpoly=spec.tendons[i].damping[1:],
            tendon_damping=model.tendon_damping[i],
            tendon_frictionloss=model.tendon_frictionloss[i],
            tendon_armature=model.tendon_armature[i],
            tendon_limited=int(spec.tendons[i].limited),
            tendon_range=model.tendon_range[i],
            tendon_margin=model.tendon_margin[i],
            tendon_springlength=spec.tendons[i].springlength,
            tendon_solref_limit=model.tendon_solref_lim[i],
            tendon_solimp_limit=model.tendon_solimp_lim[i],
            tendon_solref_friction=model.tendon_solref_fri[i],
            tendon_solimp_friction=model.tendon_solimp_fri[i],
            tendon_actuator_force_limited=int(spec.tendons[i].actfrclimited),
            tendon_actuator_force_range=model.tendon_actfrcrange[i],
        )
        values.update(custom("mujoco:tendon", spec.tendons[i]))
        mapping[i] = builder.add_custom_values(**values)["mujoco:tendon_world"]
    return mapping


def _equalities(builder, spec, model, bodies, joints, dofs, sites, tendons, attributes, convert, custom):
    import mujoco

    mujoco_only = []
    for i in range(model.neq):
        kind = EqType(int(model.eq_type[i]))
        first, second = int(model.eq_obj1id[i]), int(model.eq_obj2id[i])
        data = model.eq_data[i]
        attrs = attributes(
            eq_solref=model.eq_solref[i],
            eq_solimp=model.eq_solimp[i],
            native_equality_auto_relpose=False,
            native_equality_site1=-1,
            native_equality_site2=-1,
        )
        attrs.update(custom("mujoco:equality_constraint", spec.equalities[i]))
        label, active = model.equality(i).name, bool(model.eq_active0[i])
        if kind == EqType.TENDON:
            if first not in tendons or (second >= 0 and second not in tendons):
                continue
            attrs.update(
                attributes(
                    equality_constraint_objtype=int(mujoco.mjtObj.mjOBJ_TENDON),
                )
            )
            _add_equality_constraint(
                builder,
                kind,
                tendon1=tendons[first],
                tendon2=tendons.get(second, -1),
                polycoef=data[:5].tolist(),
                label=label,
                enabled=active,
                custom_attributes=attrs,
            )
            continue
        if kind == EqType.JOINT:
            if first not in joints or (second >= 0 and second not in joints):
                continue
            for side, target in enumerate((first, second), 1):
                if target >= 0 and builder.joint_type[joints[target]] == JointType.D6:
                    attrs.update(attributes(**{f"equality_constraint_joint{side}_dof": dofs[target]}))
            scalar = second >= 0 and all(sum(builder.joint_dof_dim[joints[j]]) == 1 for j in (first, second))
            if convert and scalar:
                if mjc_polycoef_has_higher_order(data[:5]):
                    warnings.warn(
                        f"Joint equality '{label}' uses higher-order polycoef terms. "
                        "They are preserved for SolverMuJoCo, but generic Newton mimic constraints use "
                        "only coef0/coef1.",
                        NewtonWarning,
                        stacklevel=2,
                    )
                row, _ = mjc_add_equality_mimic(
                    builder, joints[first], joints.get(second, -1), data[:5], label, active, attrs
                )
            else:
                if convert:
                    mujoco_only.append(label or f"joint1={model.joint(first).name}")
                row = _add_equality_constraint(
                    builder,
                    kind,
                    joint1=joints[first],
                    joint2=joints.get(second, -1),
                    polycoef=data[:5].tolist(),
                    label=label,
                    enabled=active,
                    custom_attributes=attrs,
                )
            builder._process_custom_attributes(row, attrs, "mujoco:equality_constraint")
            continue
        if kind not in (EqType.CONNECT, EqType.WELD):
            raise NotImplementedError(f"Native equality type {kind} is not represented in Newton.")
        if model.eq_objtype[i] == int(mujoco.mjtObj.mjOBJ_SITE):
            if first not in sites or second not in sites:
                continue
            shape1, shape2 = sites[first], sites[second]
            attrs.update(
                attributes(
                    native_equality_site1=shape1,
                    native_equality_site2=shape2,
                    equality_constraint_objtype=int(mujoco.mjtObj.mjOBJ_SITE),
                )
            )
            body1, body2 = builder.shape_body[shape1], builder.shape_body[shape2]
            anchor = builder.shape_transform[shape1].p
            relpose = builder.shape_transform[shape1] * wp.transform_inverse(builder.shape_transform[shape2])
        else:
            body1, body2 = bodies[first], bodies[second] if second >= 0 else -1
            anchor = wp.vec3(data[:3])
            relpose = _transform(data[3:6], data[6:10]) if kind == EqType.WELD else None
        if body1 == -2 or body2 == -2:
            continue
        if (
            kind == EqType.WELD
            and model.eq_objtype[i] == int(mujoco.mjtObj.mjOBJ_BODY)
            and not np.any(spec.equalities[i].data[6:10])
        ):
            attrs.update(attributes(native_equality_auto_relpose=True))
            builder._add_model_finalizer(_finalize_native_weld_poses)
            pose1 = builder.body_q[body1] if body1 >= 0 else wp.transform_identity()
            pose2 = builder.body_q[body2] if body2 >= 0 else wp.transform_identity()
            relpose = wp.transform_inverse(pose1) * pose2
        if convert:
            row, loop = mjc_add_equality_loop_joint(
                builder, kind, body1, body2, anchor, relpose, float(data[10]), label, active, attrs
            )
            if model.eq_objtype[i] == int(mujoco.mjtObj.mjOBJ_SITE):
                parent_site, child_site = (shape1, shape2) if builder.joint_parent[loop] == body1 else (shape2, shape1)
                builder.joint_X_p[loop] = builder.shape_transform[parent_site]
                builder.joint_X_c[loop] = builder.shape_transform[child_site]
        else:
            row = _add_equality_constraint(
                builder,
                kind,
                body1=body1,
                body2=body2,
                anchor=anchor,
                relpose=relpose,
                torquescale=float(data[10]),
                label=label,
                enabled=active,
                custom_attributes=attrs,
            )

        builder._process_custom_attributes(row, attrs, "mujoco:equality_constraint")

    if mujoco_only:
        warnings.warn(
            f"{len(mujoco_only)} joint equalities omit joint2 or target a multi-DoF joint, "
            f"so they stay MuJoCo equality constraints: {', '.join(mujoco_only)}.",
            NewtonWarning,
            stacklevel=3,
        )


def _contacts(builder, spec, model, bodies, geoms, attributes, start_shape, self_collisions, custom):
    pairs = {pair.id: pair for pair in spec.pairs}
    for i in range(model.npair):
        first, second = int(model.pair_geom1[i]), int(model.pair_geom2[i])
        if first not in geoms or second not in geoms:
            continue
        values = attributes(pair_world=builder.current_world, pair_geom1=geoms[first], pair_geom2=geoms[second])
        for name in ("dim", "solref", "solreffriction", "solimp", "margin", "gap", "friction"):
            key = "condim" if name == "dim" else name
            values.update(attributes(**{f"pair_{key}": getattr(model, f"pair_{name}")[i]}))
        values.update(custom("mujoco:pair", pairs[i]))
        builder.add_custom_values(**values)
    for signature in model.exclude_signature:
        first, second = int(signature) >> 16, int(signature) & 65535
        if bodies[first] == -2 or bodies[second] == -2:
            continue
        for a in builder.body_shapes.get(bodies[first], []):
            for b in builder.body_shapes.get(bodies[second], []):
                builder.add_shape_collision_filter_pair(a, b)
    selected = [
        i for i in range(start_shape, builder.shape_count) if builder.shape_flags[i] & ShapeFlags.COLLIDE_SHAPES
    ]
    if not self_collisions:
        for a, shape in enumerate(selected):
            for other in selected[a + 1 :]:
                builder.add_shape_collision_filter_pair(shape, other)
    if not selected:
        return
    local = {shape: i for i, shape in enumerate(selected)}
    masks = [builder.custom_attributes[f"mujoco:{name}"] for name in ("contype", "conaffinity")]
    compiled = compile_collision_masks(
        *[[int(attr.values.get(shape, attr.default)) for shape in selected] for attr in masks],
        prefiltered_pairs=[
            (local[a], local[b])
            for a, b in builder._shape_collision_filter_pairs.explicit_pairs
            if a in local and b in local
        ],
        force_nonzero=True,
        max_positive_groups=1,
    )
    negative_start = min([0, *builder.shape_collision_group[:start_shape]])
    for i, shape in enumerate(selected):
        group = int(compiled.groups[i])
        builder.shape_collision_group[shape] = (
            max(1, builder.default_shape_cfg.collision_group)
            if group > 0
            else group + negative_start
            if group < 0
            else 0
        )
    for a, b in compiled.excluded_pairs:
        builder.add_shape_collision_filter_pair(selected[a], selected[b])


def _validate(spec: mujoco.MjSpec) -> None:
    import mujoco

    for mesh in spec.meshes:
        if not np.all(np.isfinite(mesh.refpos)):
            raise ValueError(f"Mesh {mesh.name!r} refpos must contain only finite values.")
        norm = np.linalg.norm(mesh.refquat)
        if not np.isfinite(norm) or norm == 0:
            raise ValueError(f"Mesh {mesh.name!r} refquat must be finite and nonzero.")

    def reject(feature: str) -> None:
        raise NotImplementedError(f"Native MJCF import does not support {feature}.")

    for field in ("flexes", "plugins"):
        if len(getattr(spec, field)):
            reject(field)
    omitted = [
        field
        for field in ("sensors", "cameras", "lights", "keys", "skins", "numerics", "texts", "tuples")
        if len(getattr(spec, field))
    ]
    if omitted:
        warnings.warn(
            f"Native MJCF import omits {', '.join(omitted)}; Newton has no corresponding model fields.",
            NewtonWarning,
            stacklevel=3,
        )
    for field in ("sensors", "keys"):
        for element in list(getattr(spec, field)):
            spec.delete(element)
    for equality in spec.equalities:
        if equality.type not in (
            mujoco.mjtEq.mjEQ_CONNECT,
            mujoco.mjtEq.mjEQ_WELD,
            mujoco.mjtEq.mjEQ_JOINT,
            mujoco.mjtEq.mjEQ_TENDON,
        ):
            reject(f"equality type {equality.type}")
    for actuator in spec.actuators:
        if actuator.gaintype == mujoco.mjtGain.mjGAIN_USER or actuator.biastype == mujoco.mjtBias.mjBIAS_USER:
            reject("actuator callbacks")
        if int(actuator.dyntype) > int(mujoco.mjtDyn.mjDYN_DCMOTOR):
            reject("actuator dynamics type")
        if actuator.gaintype == mujoco.mjtGain.mjGAIN_DCMOTOR:
            dcmotor.parse_input(int(actuator.ctrlspec) or 8)


def _parse_spec(builder: ModelBuilder, source: str, scale: float) -> tuple[mujoco.MjSpec, list[bool], list[bool]]:
    import mujoco

    is_xml = source.lstrip().startswith("<")
    root = ET.fromstring(sanitize_xml_content(source)) if is_xml else ET.parse(source).getroot()
    # Let authored flags override builder settings, including flags from earlier imports.
    options = ET.Element("option")
    flags = ET.SubElement(options, "flag")
    for name, default in SolverMuJoCo._FLAG_DEFAULTS.items():
        attr = builder.custom_attributes.get(f"mujoco:flag_{name}")
        enabled = attr.values.get(0, attr.default) if attr is not None else default
        flags.set(name, "enable" if enabled else "disable")
    sleeping = builder.custom_attributes.get("mujoco:enable_sleeping")
    enabled = sleeping.values.get(0, sleeping.default) if sleeping is not None else False
    flags.set("sleep", "enable" if enabled else "disable")
    root.insert(0, options)
    defaults = ET.Element("default")
    joint_defaults = ET.SubElement(
        defaults,
        "joint",
        damping=str(builder.default_joint_cfg.damping),
        stiffness=str(builder.default_joint_cfg.stiffness),
        armature=str(builder.default_joint_cfg.armature),
    )
    gap = builder.default_shape_cfg.gap
    if gap is None:
        gap = builder.rigid_gap
    # Builder gaps are already in meters; only file-authored gaps should scale.
    geom_defaults = ET.SubElement(defaults, "geom", gap=str(gap / scale))
    root.insert(0, defaults)

    def parse():
        xml = ET.tostring(root, encoding="unicode")
        if is_xml:
            return mujoco.MjSpec.from_string(xml)
        # Override only the root XML in the VFS, preserving its directory for includes/assets.
        filename = os.path.abspath(source)
        return mujoco.MjSpec.from_file(filename, include={filename: xml.encode("utf-8")})

    spec = parse()
    # Changing fallbacks reveals which values were omitted, including inherited defaults.
    joint_defaults.set("solreflimit", "0.03 1")
    joint_defaults.set("ref", "1")
    joint_defaults.set("springref", "1")
    geom_defaults.set("friction", "2")
    probe = parse()
    default_limits = [
        not np.array_equal(joint.solref_limit, other.solref_limit, equal_nan=True)
        for joint, other in zip(spec.joints, probe.joints, strict=True)
    ]
    default_friction = [
        not np.array_equal(geom.friction[:1], other.friction[:1], equal_nan=True)
        for geom, other in zip(spec.geoms, probe.geoms, strict=True)
    ]
    for geom, omitted in zip(spec.geoms, default_friction, strict=True):
        if omitted:
            cfg = builder.default_shape_cfg
            geom.friction[:] = (cfg.mu, cfg.mu_torsional, cfg.mu_rolling)
    for joint, other in zip(spec.joints, probe.joints, strict=True):
        if (
            builder.default_joint_cfg.rest_q is not None
            and joint.ref != other.ref
            and joint.springref != other.springref
        ):
            if joint.type == mujoco.mjtJoint.mjJNT_SLIDE:
                joint.springref = builder.default_joint_cfg.rest_q / scale
            elif joint.type == mujoco.mjtJoint.mjJNT_HINGE:
                joint.springref = (
                    np.rad2deg(builder.default_joint_cfg.rest_q)
                    if spec.compiler.degree
                    else builder.default_joint_cfg.rest_q
                )
    return spec, default_limits, default_friction


def parse_mjcf(
    builder: ModelBuilder,
    source: str | os.PathLike,
    *,
    floating: bool | None = None,
    base_joint: dict | None = None,
    parent_body: int = -1,
    xform: Transform | None = None,
    armature_scale: float = 1.0,
    scale: float = 1.0,
    hide_visuals: bool = False,
    parse_visuals_as_colliders: bool = False,
    parse_meshes: bool = True,
    parse_sites: bool = True,
    parse_visuals: bool = True,
    parse_mujoco_options: bool = True,
    up_axis: AxisType = Axis.Z,
    ignore_names: Sequence[str] = (),
    ignore_classes: Sequence[str] = (),
    visual_classes: Sequence[str] = ("visual",),
    collider_classes: Sequence[str] = ("collision",),
    no_class_as_colliders: bool = True,
    force_show_colliders: bool = False,
    enable_self_collisions: bool = True,
    ignore_inertial_definitions: bool = False,
    collapse_fixed_joints: bool = False,
    collapse_massless_fixed_root: bool = False,
    verbose: bool = False,
    skip_equality_constraints: bool = False,
    convert_mjc_equality_constraints: bool = True,
    convert_3d_hinge_to_ball_joints: bool = False,
    mesh_maxhullvert: int | None = None,
    ctrl_direct: bool = True,
    path_resolver: Callable[[str | None, str], str] | None = None,
    override_root_xform: bool = False,
    legacy_margin_gap: bool = False,
) -> None:
    """Import compiled data, retaining authored inputs that depend on composition."""
    import mujoco

    if floating is not None and base_joint is not None:
        raise ValueError("floating and base_joint are mutually exclusive")
    if floating and parent_body != -1:
        raise ValueError("floating=True requires parent_body=-1")
    if override_root_xform and xform is None:
        raise ValueError("override_root_xform=True requires xform")
    if scale <= 0 or not np.isfinite(scale):
        raise ValueError("scale must be positive and finite")
    source = os.fspath(source)
    if path_resolver is not None:
        from ...utils.import_mjcf import _load_and_expand_mjcf  # noqa: PLC0415

        root, _ = _load_and_expand_mjcf(source, path_resolver)
        source = ET.tostring(root, encoding="unicode")
    try:
        spec, default_limits, default_friction = _parse_spec(builder, source, scale)
    except (ValueError, ET.ParseError) as exc:
        raise ValueError(f"Native MJCF parsing failed: {exc}") from exc
    custom = _extract_custom_attributes(builder, spec, scale, parse_mujoco_options)
    if skip_equality_constraints:
        for equality in list(spec.equalities):
            spec.delete(equality)
    _validate(spec)
    if ignore_inertial_definitions:
        spec.compiler.inertiafromgeom = mujoco.mjtInertiaFromGeom.mjINERTIAFROMGEOM_TRUE
    if scale != 1.0:
        _scale_spec(spec, scale)
    # Compilation resolves these using the isolated asset's effective inertia.
    actuator_bias = [np.array(a.biasprm) for a in spec.actuators]
    model = spec.compile()
    canonical_joint_cfg = ModelBuilder.JointDofConfig()
    gains = (builder.default_joint_cfg.limit_ke, builder.default_joint_cfg.limit_kd)
    default_limit_mode = (
        SOLREF_MODE_FORCE_SPACE
        if gains != (canonical_joint_cfg.limit_ke, canonical_joint_cfg.limit_kd)
        else SOLREF_MODE_MJCF_DEFAULT
    )
    geom_classes = {g.id: g.classname.name for g in spec.geoms}
    body_classes = {b.id: b.classname.name for b in spec.bodies}
    joint_classes = {j.id: j.classname.name for j in spec.joints}
    shape_types = {
        mujoco.mjtGeom.mjGEOM_SPHERE: GeoType.SPHERE,
        mujoco.mjtGeom.mjGEOM_CAPSULE: GeoType.CAPSULE,
        mujoco.mjtGeom.mjGEOM_ELLIPSOID: GeoType.ELLIPSOID,
        mujoco.mjtGeom.mjGEOM_CYLINDER: GeoType.CYLINDER,
        mujoco.mjtGeom.mjGEOM_BOX: GeoType.BOX,
        mujoco.mjtGeom.mjGEOM_PLANE: GeoType.PLANE,
        mujoco.mjtGeom.mjGEOM_MESH: GeoType.MESH,
        mujoco.mjtGeom.mjGEOM_HFIELD: GeoType.HFIELD,
    }
    shape_types = {int(key): value for key, value in shape_types.items()}
    if any(int(t) not in shape_types for t in model.geom_type):
        raise NotImplementedError("Native MJCF import does not support SDF geometry.")
    if np.any(model.actuator_ctrlnum != 1):
        raise NotImplementedError("Newton actuator controls require one scalar per actuator.")
    hooked = {attr.frequency for attr in builder.custom_attributes.values() if attr.mjcf_native_value_transformer}
    frequency = Model.AttributeFrequency
    user_frequencies = {
        "body": {frequency.BODY},
        "jnt": {frequency.JOINT, frequency.JOINT_DOF},
        "geom": {frequency.SHAPE},
        "site": {frequency.SHAPE},
        "tendon": {"mujoco:tendon"},
        "actuator": {"mujoco:actuator"},
        "cam": set(),
        "sensor": set(),
    }
    if any(
        getattr(model, f"nuser_{kind}") and not hooked.intersection(freqs) for kind, freqs in user_frequencies.items()
    ) or (model.nuserdata and not (parse_mujoco_options and hooked.intersection({frequency.WORLD, frequency.ONCE}))):
        warnings.warn("Native MJCF import omits MuJoCo user data.", NewtonWarning, stacklevel=2)
    for i in range(1, model.nbody):
        start, count = model.body_jntadr[i], model.body_jntnum[i]
        kinds = list(map(int, model.jnt_type[start : start + count]))
        if count > 1 and int(mujoco.mjtJoint.mjJNT_BALL) in kinds:
            raise NotImplementedError(
                "Native MJCF import does not support a ball joint combined with other joints "
                f"in body {model.body(i).name!r}."
            )
    if parent_body < -1 or parent_body >= builder.body_count:
        raise ValueError("parent_body must identify an existing body or be -1")
    articulation = builder._check_sequential_composition(parent_body)
    if parent_body != -1 and articulation is None:
        raise ValueError("parent_body must belong to an articulation")
    SolverMuJoCo.register_custom_attributes(builder)
    _register_native_attributes(builder)

    def attributes(**values: Any) -> dict[str, Any]:
        result = {}
        for name, value in values.items():
            key = f"mujoco:{name}"
            dtype = builder.custom_attributes[key].dtype
            result[key] = dtype(*value) if np.ndim(value) else value
        return result

    if parse_mujoco_options:
        _import_options(builder, spec, attributes)
        for freq, row in ((frequency.ONCE, 0), (frequency.WORLD, builder.current_world)):
            builder._process_custom_attributes(row, custom(freq, spec), freq)
    disableflags = builder.custom_attributes["mujoco:native_option_disableflags"]
    filter_parent = not bool(
        disableflags.values.get(0, disableflags.default) & mujoco.mjtDisableBit.mjDSBL_FILTERPARENT
    )

    def matches(name: str, patterns: Sequence[str]) -> bool:
        return any(re.match(pattern, name) for pattern in patterns)

    prefix = "" if spec.modelname == "MuJoCo Model" else spec.modelname
    start_shape = builder.shape_count
    domain_attr = builder.custom_attributes["mujoco:collision_mask_domain"]
    domain = max(domain_attr.values.values(), default=-1) + 1
    placement = wp.transform_identity() if xform is None else wp.transform(*xform)
    placement = placement * wp.transform(wp.vec3(), quat_between_axes(up_axis, builder.up_axis))
    bodies = [parent_body]
    joints = {}
    joint_ids = {}
    body_labels = [f"{prefix}/worldbody" if prefix else "worldbody"]
    root_joints = {}
    roots = {0: 0}
    for i in range(1, model.nbody):
        native_parent = int(model.body_parentid[i])
        name = model.body(i).name or f"body_{builder.body_count}"
        label = f"{body_labels[native_parent]}/{sanitize_name(name)}"
        body_labels.append(label)
        if bodies[native_parent] == -2 or matches(name, ignore_names) or matches(body_classes[i], ignore_classes):
            bodies.append(-2)
            continue
        parent = bodies[native_parent]
        local = _transform(model.body_pos[i], model.body_quat[i])
        if native_parent == 0:
            local = placement if override_root_xform else placement * local
        world = local if parent == -1 else builder.body_q[parent] * local
        rotation = np.asarray(wp.quat_to_matrix(_transform(np.zeros(3), model.body_iquat[i]).q)).reshape(3, 3)
        inertia = rotation @ np.diag(model.body_inertia[i]) @ rotation.T
        body = builder.add_link(
            xform=world,
            mass=float(model.body_mass[i]),
            com=model.body_ipos[i],
            inertia=inertia,
            lock_inertia=True,
            label=label,
            is_kinematic=bool(model.body_mocapid[i] >= 0),
            custom_attributes=attributes(gravcomp=model.body_gravcomp[i], sleep_policy=int(spec.bodies[i].sleep))
            | custom(frequency.BODY, spec.bodies[i]),
        )
        bodies.append(body)
        roots[i] = i if native_parent == 0 else roots[native_parent]
        root_joints.setdefault(roots[i], [])
        selected = [
            j
            for j in range(model.body_jntadr[i], model.body_jntadr[i] + model.body_jntnum[i])
            if not matches(model.joint(j).name, ignore_names) and not matches(joint_classes[j], ignore_classes)
        ]
        if native_parent == 0 and (floating is not None or base_joint is not None or parent_body != -1):
            parent_xform, child_xform = local, wp.transform_identity()
            if base_joint is not None and not override_root_xform:
                parent_xform = wp.transform(local.p, wp.quat_identity())
                child_xform = wp.transform(wp.vec3(), wp.quat_inverse(local.q))
            joint = builder._add_base_joint(
                child=body,
                parent=parent,
                floating=floating,
                base_joint=base_joint,
                parent_xform=parent_xform,
                child_xform=child_xform,
                label=f"{label}/base_joint",
            )
        else:
            joint, mapping = _add_joints(
                builder,
                model,
                selected,
                parent,
                body,
                local,
                label,
                attributes,
                armature_scale,
                convert_3d_hinge_to_ball_joints,
            )
            mapped_dofs = [
                dof + k
                for native_joint, dof in mapping.items()
                for k in range(
                    6 if model.jnt_type[native_joint] == 0 else 3 if model.jnt_type[native_joint] == 1 else 1
                )
            ]
            start = builder.joint_qd_start[joint]
            expected_dofs = range(start, start + sum(builder.joint_dof_dim[joint]))
            if set(mapping) != set(selected) or sorted(mapped_dofs) != list(expected_dofs):
                raise RuntimeError(f"Native MJCF conversion did not preserve all joints and DOFs of body {name!r}.")
            for native_joint, dof in mapping.items():
                joints[native_joint] = dof
                joint_ids[native_joint] = joint
                source_joint = spec.joints[native_joint]
                count = 6 if model.jnt_type[native_joint] == 0 else 3 if model.jnt_type[native_joint] == 1 else 1
                source_dof = int(model.jnt_dofadr[native_joint])
                for k in range(count):
                    builder.joint_damping[dof + k] = float(model.dof_damping[source_dof + k])
                    builder.joint_stiffness[dof + k] = float(model.jnt_stiffness[native_joint])
                    builder.joint_armature[dof + k] = float(model.dof_armature[source_dof + k]) * armature_scale
                    builder.joint_friction[dof + k] = float(model.dof_frictionloss[source_dof + k])
                    if np.all(source_joint.springdamper > 0):
                        builder.custom_attributes["mujoco:native_joint_gain_baseline"].values[dof + k] = (
                            builder.joint_stiffness[dof + k],
                            builder.joint_damping[dof + k],
                        )
                        builder._add_model_finalizer(_finalize_native_core_parameters)
                    if model.jnt_type[native_joint] != mujoco.mjtJoint.mjJNT_FREE:
                        builder.joint_limit_ke[dof + k] = gains[0]
                        builder.joint_limit_kd[dof + k] = gains[1]
                extra = attributes(
                    jnt_actgravcomp=bool(model.jnt_actgravcomp[native_joint]),
                    native_joint_type=int(model.jnt_type[native_joint]),
                    native_joint_springdamper=source_joint.springdamper,
                    native_joint_limited=int(source_joint.limited),
                    native_joint_actfrcrange=model.jnt_actfrcrange[native_joint],
                    native_joint_actfrclimited=int(source_joint.actfrclimited),
                    native_joint_dampingpoly=source_joint.damping[1:],
                    native_joint_stiffnesspoly=source_joint.stiffness[1:],
                    limit_margin=model.jnt_margin[native_joint],
                    solreflimit=(0.0, 0.0) if default_limits[native_joint] else model.jnt_solref[native_joint],
                    solimplimit=model.jnt_solimp[native_joint],
                    solreflimit_mode=default_limit_mode if default_limits[native_joint] else SOLREF_MODE_RAW,
                    solreflimit_gain_baseline=gains,
                    solreffriction=model.dof_solref[source_dof],
                    solimpfriction=model.dof_solimp[source_dof],
                )
                if count > 1:
                    extra.update(attributes(native_joint_range=model.jnt_range[native_joint]))
                for key, value in extra.items():
                    for k in range(1 if key.startswith("mujoco:native_joint_") else count):
                        builder.custom_attributes[key].values[dof + k] = value
                for k in range(count):
                    builder._process_custom_attributes(
                        dof + k, custom(frequency.JOINT_DOF, source_joint), frequency.JOINT_DOF
                    )
            if selected:
                builder._process_custom_attributes(
                    joint, custom(frequency.JOINT, spec.joints[selected[0]]), frequency.JOINT
                )
        root_joints[roots[i]].append(joint)
        set_mjcf_joint_collision_filter(builder, joint, filter_parent)

    required_geoms = set(map(int, np.concatenate((model.pair_geom1, model.pair_geom2))))
    meshes = {}
    geoms = {}
    sites = {}
    visual_shapes = []
    collider_shapes = []
    for i in range(model.ngeom):
        native_body = int(model.geom_bodyid[i])
        if (
            bodies[native_body] == -2
            or matches(geom_classes[i], ignore_classes)
            or matches(model.geom(i).name, ignore_names)
        ):
            continue
        kind = int(model.geom_type[i])
        if not parse_meshes and kind in (int(mujoco.mjtGeom.mjGEOM_MESH), int(mujoco.mjtGeom.mjGEOM_HFIELD)):
            continue
        visual = matches(geom_classes[i], visual_classes)
        collider = matches(geom_classes[i], collider_classes)
        if not visual and not collider:
            collider = no_class_as_colliders and bool(model.geom_contype[i] or model.geom_conaffinity[i])
            visual = not collider
        if parse_visuals_as_colliders:
            collider, visual = visual, False
        if i in required_geoms:
            collider = True
        if not collider and not (visual and parse_visuals):
            continue
        pose = _transform(model.geom_pos[i], model.geom_quat[i])
        if native_body == 0:
            pose = placement * pose
        rgba = model.geom_rgba[i].copy()
        material_id = int(model.geom_matid[i])
        use_material = material_id >= 0 and np.array_equal(rgba, np.array([0.5, 0.5, 0.5, 1], dtype=np.float32))
        if use_material:
            rgba = model.mat_rgba[material_id].copy()
        cfg = builder.default_shape_cfg.copy()
        cfg.density = 0.0
        ke, kd = solref_to_stiffness_damping(model.geom_solref[i])
        if ke is not None:
            cfg.ke = ke
        if kd is not None:
            cfg.kd = kd
        cfg.mu, cfg.mu_torsional, cfg.mu_rolling = map(float, model.geom_friction[i])
        if not default_friction[i]:
            cfg.mu_static = None
        cfg.margin, cfg.gap = float(model.geom_margin[i]), float(model.geom_gap[i])
        if legacy_margin_gap:
            cfg.margin -= cfg.gap
        cfg.has_shape_collision = collider
        cfg.has_particle_collision = cfg.has_shape_collision
        cfg.collision_group = 0 if not (model.geom_contype[i] or model.geom_conaffinity[i]) else cfg.collision_group
        cfg.is_visible = (
            (visual and not hide_visuals)
            or force_show_colliders
            or parse_visuals_as_colliders
            or (collider and kind == int(mujoco.mjtGeom.mjGEOM_PLANE) and 0 <= model.geom_group[i] <= 2)
        )
        size = model.geom_size[i].copy()
        src = None
        if model.geom_type[i] == mujoco.mjtGeom.mjGEOM_SPHERE:
            size[:] = size[0]
        elif model.geom_type[i] in (mujoco.mjtGeom.mjGEOM_CAPSULE, mujoco.mjtGeom.mjGEOM_CYLINDER):
            size[2] = size[0]
        elif kind == int(mujoco.mjtGeom.mjGEOM_PLANE):
            size[:] = 0
        elif kind == int(mujoco.mjtGeom.mjGEOM_MESH):
            mesh_id = int(model.geom_dataid[i])
            material_id = int(model.geom_matid[i])
            key = (mesh_id, material_id)
            if key not in meshes:
                native_maxhullvert = spec.meshes[mesh_id].maxhullvert
                maxhullvert = native_maxhullvert if native_maxhullvert >= 4 else mesh_maxhullvert
                meshes[key] = _mesh(model, mesh_id, material_id, maxhullvert)
            src = meshes[key]
            size[:] = 1
            cfg.sdf_max_resolution = None
            cfg.sdf_target_voxel_size = None
            cfg.sdf_narrow_band_range = (-0.1, 0.1)
        elif kind == int(mujoco.mjtGeom.mjGEOM_HFIELD):
            h = int(model.geom_dataid[i])
            rows, cols = int(model.hfield_nrow[h]), int(model.hfield_ncol[h])
            adr = int(model.hfield_adr[h])
            src = Heightfield(
                data=model.hfield_data[adr : adr + rows * cols].reshape(rows, cols).copy(),
                nrow=rows,
                ncol=cols,
                hx=model.hfield_size[h, 0],
                hy=model.hfield_size[h, 1],
                min_z=0.0,
                max_z=model.hfield_size[h, 2],
            )
            size[:] = 1
        if use_material and not (isinstance(src, Mesh) and src.texture is not None):
            texture_id = _color_texture(model, material_id)
            if texture_id >= 0:
                texture = spec.textures[texture_id]
                if texture.builtin != mujoco.mjtBuiltin.mjBUILTIN_NONE:
                    mean = (
                        texture.rgb1
                        if texture.builtin == mujoco.mjtBuiltin.mjBUILTIN_FLAT
                        else 0.5 * (texture.rgb1 + texture.rgb2)
                    )
                    rgba[:3] *= mean
        shape = builder.add_shape(
            body=bodies[native_body] if native_body else -1,
            type=shape_types[model.geom_type[i]],
            xform=pose,
            scale=size,
            src=src,
            cfg=cfg,
            color=rgba[:3],
            opacity=clamp_imported_opacity(rgba[3], "MJCF geom rgba"),
            label=f"{body_labels[native_body]}/{sanitize_name(model.geom(i).name or f'geom_{i}')}",
            custom_attributes=attributes(
                hfield_base=float(model.hfield_size[model.geom_dataid[i], 3])
                if kind == int(mujoco.mjtGeom.mjGEOM_HFIELD)
                else -1.0,
                contype=int(model.geom_contype[i]),
                conaffinity=int(model.geom_conaffinity[i]),
                collision_mask_domain=domain,
                condim=int(model.geom_condim[i]),
                geom_priority=int(model.geom_priority[i]),
                geom_group=int(model.geom_group[i]),
                **_read_native_parameters("geom", spec.geoms[i]),
                geom_solmix=model.geom_solmix[i],
                solref=model.geom_solref[i],
                geom_solimp=model.geom_solimp[i],
                solref_mode=SOLREF_MODE_RAW,
            )
            | custom(frequency.SHAPE, spec.geoms[i]),
        )
        geoms[i] = shape
        (collider_shapes if collider else visual_shapes).append(shape)

    if not visual_shapes:
        for shape in collider_shapes:
            builder.shape_flags[shape] |= ShapeFlags.VISIBLE
    if parse_sites:
        for i in range(model.nsite):
            native_body = int(model.site_bodyid[i])
            if bodies[native_body] == -2 or matches(model.site(i).name, ignore_names):
                continue
            pose = _transform(model.site_pos[i], model.site_quat[i])
            if native_body == 0:
                pose = placement * pose
            site_size = model.site_size[i].copy()
            if model.site_type[i] == mujoco.mjtGeom.mjGEOM_CYLINDER:
                site_size[2] = 0
            sites[i] = builder.add_site(
                body=bodies[native_body] if native_body else -1,
                xform=pose,
                type=shape_types[int(model.site_type[i])],
                scale=site_size,
                visible=False,
                label=f"{body_labels[native_body]}/{sanitize_name(model.site(i).name or f'site_{i}')}",
                custom_attributes=custom(frequency.SHAPE, spec.sites[i]),
            )

    tendons = _tendons(builder, spec, model, joint_ids, joints, geoms, sites, attributes, custom)

    actuator_start = builder._custom_frequency_counts.get("mujoco:actuator", 0)
    skipped_actuators = 0
    pending_damping: set[int] = set()
    for i in range(model.nu):
        transmission = {
            int(mujoco.mjtTrn.mjTRN_JOINT): int(SolverMuJoCo.TrnType.JOINT),
            int(mujoco.mjtTrn.mjTRN_JOINTINPARENT): int(SolverMuJoCo.TrnType.JOINT_IN_PARENT),
            int(mujoco.mjtTrn.mjTRN_SLIDERCRANK): int(SolverMuJoCo.TrnType.SLIDERCRANK),
            int(mujoco.mjtTrn.mjTRN_TENDON): int(SolverMuJoCo.TrnType.TENDON),
            int(mujoco.mjtTrn.mjTRN_SITE): int(SolverMuJoCo.TrnType.SITE),
            int(mujoco.mjtTrn.mjTRN_BODY): int(SolverMuJoCo.TrnType.BODY),
        }[int(model.actuator_trntype[i])]
        source_target, source_alt = map(int, model.actuator_trnid[i])
        mapping = {
            0: joints,
            1: joints,
            2: tendons,
            3: sites,
            4: {j: b for j, b in enumerate(bodies) if b >= 0},
            5: sites,
        }[transmission]
        if source_target not in mapping or (source_alt >= 0 and transmission in (3, 5) and source_alt not in sites):
            warnings.warn(
                f"Actuator {model.actuator(i).name!r} targets an excluded element; skipping.",
                NewtonWarning,
                stacklevel=2,
            )
            skipped_actuators += 1
            continue
        target = mapping[source_target]
        alternate = sites[source_alt] if source_alt >= 0 and transmission in (3, 5) else -1
        native_parameters = _read_native_parameters("actuator", spec.actuators[i])
        values = attributes(
            native_actuator_target_kd=-1.0,
            actuator_label=f"{prefix}/{model.actuator(i).name or f'actuator_{i}'}",
            ctrl_source=int(SolverMuJoCo.CtrlSource.CTRL_DIRECT),
            ctrl_type=int(
                SolverMuJoCo.CtrlType.DCMOTOR
                if model.actuator_gaintype[i] == mujoco.mjtGain.mjGAIN_DCMOTOR
                else SolverMuJoCo.CtrlType.GENERAL
            ),
            actuator_world=builder.current_world,
            actuator_trntype=transmission,
            actuator_trnid=(target, alternate),
            actuator_gainprm=model.actuator_gainprm[i],
            actuator_biasprm=actuator_bias[i],
            actuator_dynprm=model.actuator_dynprm[i],
            actuator_gear=model.actuator_gear[i],
            actuator_dyntype=int(model.actuator_dyntype[i]),
            actuator_gaintype=int(model.actuator_gaintype[i]),
            actuator_biastype=int(model.actuator_biastype[i]),
            actuator_actdim=int(spec.actuators[i].actdim),
            actuator_actearly=bool(model.actuator_actearly[i]),
            actuator_lengthrange=spec.actuators[i].lengthrange,
            actuator_cranklength=model.actuator_cranklength[i],
            actuator_ctrlspec=int(model.actuator_ctrlspec[i]),
            actuator_damping=model.actuator_damping[i],
            actuator_dampingpoly=model.actuator_dampingpoly[i],
            actuator_armature=model.actuator_armature[i],
            **native_parameters,
        )
        for name in ("ctrl", "force", "act"):
            values.update(
                attributes(
                    **{
                        f"actuator_{name}range": getattr(spec.actuators[i], f"{name}range"),
                        f"actuator_{name}limited": int(getattr(spec.actuators[i], f"{name}limited")),
                        f"actuator_has_{name}range": True,
                    }
                )
            )
        if (
            not ctrl_direct
            and transmission == 0
            and int(model.jnt_type[source_target])
            in (int(mujoco.mjtJoint.mjJNT_HINGE), int(mujoco.mjtJoint.mjJNT_SLIDE), int(mujoco.mjtJoint.mjJNT_BALL))
        ):
            _joint_target(
                builder,
                model,
                i,
                target,
                actuator_bias[i],
                values,
                pending_damping,
                dof_count=3 if model.jnt_type[source_target] == mujoco.mjtJoint.mjJNT_BALL else 1,
            )
        if transmission == 0 and native_parameters["native_actuator_inheritrange"] > 0:
            values.update(attributes(native_actuator_inheritrange_bounds=model.jnt_range[source_target]))
        values.update(custom("mujoco:actuator", spec.actuators[i]))
        if values.get("mujoco:native_actuator_inheritrange", 0) > 0:
            builder._add_model_finalizer(_finalize_native_actuator_ranges)
        builder.add_custom_values(**values)

    imported_actuators = builder._custom_frequency_counts.get("mujoco:actuator", 0) - actuator_start
    if imported_actuators + skipped_actuators != model.nu:
        raise RuntimeError("Native MJCF conversion did not preserve all retained actuators.")

    groups = (
        [[joint for group in root_joints.values() for joint in group]] if parent_body != -1 else root_joints.values()
    )
    for group in groups:
        label = prefix
        if parent_body == -1 and len(root_joints) > 1 and group:
            root_name = builder.body_label[builder.joint_child[group[0]]].rsplit("/", 1)[-1]
            label = f"{prefix}/{root_name}" if prefix else root_name
        builder._finalize_imported_articulation(joint_indices=group, parent_body=parent_body, articulation_label=label)
    if not skip_equality_constraints:
        _equalities(
            builder,
            spec,
            model,
            bodies,
            joint_ids,
            joints,
            sites,
            tendons,
            attributes,
            convert_mjc_equality_constraints,
            custom,
        )
    add_mjcf_body_collision_filters(builder, [j for group in root_joints.values() for j in group], start_shape)
    _contacts(builder, spec, model, bodies, geoms, attributes, start_shape, enable_self_collisions, custom)
    if collapse_fixed_joints:
        builder.collapse_fixed_joints()
    elif collapse_massless_fixed_root:
        collapse_massless_fixed_root_joints(builder, [j for group in root_joints.values() for j in group])
    if verbose:
        logger.info("Imported %s bodies and %s geoms using native MuJoCo.", sum(b >= 0 for b in bodies[1:]), len(geoms))
