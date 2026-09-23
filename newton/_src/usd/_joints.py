# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Parse USD joint descriptions and add their joints to a builder."""

from __future__ import annotations

import warnings
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import numpy as np
import warp as wp

from ..sim.builder import ModelBuilder
from ..sim.enums import JointTargetMode
from ..sim.model import Model
from . import utils as usd
from ._resolution_policy import (
    _resolve_newton_limit_kd,
    _resolve_newton_limit_ke,
    _shift_joint_limits_for_reference,
)
from .schema_resolver import PrimType

if TYPE_CHECKING:
    from pxr import Usd, UsdPhysics

    from ..core.types import Axis
    from ._resolution_policy import _UsdJointProperties
    from .schema_resolver import SchemaResolverManager

AttributeFrequency = Model.AttributeFrequency


def resolve_joint_parent_child(
    joint_desc: UsdPhysics.JointDesc,
    body_index_map: dict[str, int],
    get_transforms: bool = True,
    *,
    verbose: bool,
):
    """Resolve the parent and child of a joint and return their parent + child transforms if requested."""
    if get_transforms:
        parent_tf = wp.transform(joint_desc.localPose0Position, usd.value_to_warp(joint_desc.localPose0Orientation))
        child_tf = wp.transform(joint_desc.localPose1Position, usd.value_to_warp(joint_desc.localPose1Orientation))
    else:
        parent_tf = None
        child_tf = None

    parent_path = str(joint_desc.body0)
    child_path = str(joint_desc.body1)
    parent_id = body_index_map.get(parent_path, -1)
    child_id = body_index_map.get(child_path, -1)
    # If child_id is -1, swap parent and child
    if child_id == -1:
        if parent_id == -1:
            raise ValueError(f"Unable to parse joint {joint_desc.primPath}: both bodies unresolved")
        parent_id, child_id = child_id, parent_id
        if get_transforms:
            parent_tf, child_tf = child_tf, parent_tf
        if verbose:
            print(f"Joint {joint_desc.primPath} connects {parent_path} to world")
    if get_transforms:
        return parent_id, child_id, parent_tf, child_tf
    else:
        return parent_id, child_id


def parse_joint(
    joint_desc: UsdPhysics.JointDesc,
    incoming_xform: wp.transform | None = None,
    *,
    builder: ModelBuilder,
    stage: Usd.Stage,
    R: SchemaResolverManager,
    joint_properties: _UsdJointProperties,
    path_body_map: dict[str, int],
    path_joint_map: dict[str, int],
    builder_custom_attr_joint: list[ModelBuilder.CustomAttribute],
    physics_scene_prim: Usd.Prim | None,
    usd_axis_to_axis: dict[UsdPhysics.Axis, Axis],
    DegreesToRadian: float,
    default_joint_armature: float,
    default_joint_friction: float,
    default_joint_limit_ke: float,
    default_joint_limit_kd: float,
    default_joint_velocity_limit: float,
    joint_drive_gains_scaling: float,
    force_position_velocity_actuation: bool,
    only_load_enabled_joints: bool,
    collect_schema_attrs: bool,
    verbose: bool,
    solreflimit_mode_key: str,
    solreflimit_gain_baseline_key: str,
    _should_write_solreflimit_mode: Callable[[], bool],
    _should_write_solreflimit_gain_baseline: Callable[[], bool],
    resolve_joint_parent_child: Callable[..., Any],
) -> int | None:
    """Parse a joint description and add it to the builder. Returns the resulting joint index if successful, None otherwise."""
    from pxr import UsdPhysics

    if not joint_desc.jointEnabled and only_load_enabled_joints:
        return None
    key = joint_desc.type
    joint_path = str(joint_desc.primPath)
    joint_prim = stage.GetPrimAtPath(joint_desc.primPath)
    # collect engine-specific attributes on the joint prim if requested
    if collect_schema_attrs:
        R.collect_prim_attrs(joint_prim)
    parent_id, child_id, parent_tf, child_tf = resolve_joint_parent_child(  # pyright: ignore[reportAssignmentType]
        joint_desc, path_body_map, get_transforms=True
    )

    if incoming_xform is not None:
        parent_tf = incoming_xform * parent_tf

    # Extract custom attributes for this joint
    joint_custom_attrs = usd.get_custom_attribute_values(
        joint_prim,
        builder_custom_attr_joint,
        context={"builder": builder, "physics_scene_prim": physics_scene_prim},
    )
    joint_params = {
        "parent": parent_id,
        "child": child_id,
        "parent_xform": parent_tf,
        "child_xform": child_tf,
        "label": joint_path,
        "collision_filter_parent": parent_id != -1 and not joint_desc.collisionEnabled,
        "enabled": joint_desc.jointEnabled,
        "custom_attributes": joint_custom_attrs,
    }

    joint_index: int | None = None
    if key == UsdPhysics.ObjectType.FixedJoint:
        joint_index = builder.add_joint_fixed(**joint_params)
    elif key == UsdPhysics.ObjectType.RevoluteJoint or key == UsdPhysics.ObjectType.PrismaticJoint:
        is_revolute = key == UsdPhysics.ObjectType.RevoluteJoint
        dof = joint_properties.resolve_dof_params(
            joint_prim,
            joint_desc,
            is_revolute,
            joint_drive_gains_scaling=joint_drive_gains_scaling,
            force_position_velocity_actuation=force_position_velocity_actuation,
        )
        _shift_joint_limits_for_reference(dof, joint_custom_attrs)
        if _should_write_solreflimit_mode():
            joint_custom_attrs[solreflimit_mode_key] = dof.limit_solref_mode
        if _should_write_solreflimit_gain_baseline():
            joint_custom_attrs[solreflimit_gain_baseline_key] = wp.vec2(dof.limit_ke, dof.limit_kd)
        joint_params["axis"] = usd_axis_to_axis[joint_desc.axis]
        joint_params["limit_lower"] = dof.limit_lower
        joint_params["limit_upper"] = dof.limit_upper
        joint_params["limit_ke"] = dof.limit_ke
        joint_params["limit_kd"] = dof.limit_kd
        joint_params["armature"] = dof.armature
        joint_params["friction"] = dof.friction
        joint_params["damping"] = dof.damping
        joint_params["velocity_limit"] = dof.velocity_limit
        if dof.has_drive:
            joint_params["target_vel"] = dof.target_vel
            joint_params["target_pos"] = dof.target_pos
            joint_params["target_ke"] = dof.target_ke
            joint_params["target_kd"] = dof.target_kd
            joint_params["effort_limit"] = dof.effort_limit
        joint_params["actuator_mode"] = dof.actuator_mode

        # Initial joint state, applied after creation (already in Newton units)
        initial_position = dof.initial_position
        initial_velocity = dof.initial_velocity

        if is_revolute:
            joint_index = builder.add_joint_revolute(**joint_params)
        else:
            joint_index = builder.add_joint_prismatic(**joint_params)
    elif key == UsdPhysics.ObjectType.SphericalJoint:
        _, joint_damping = joint_properties.resolve_joint_damping(joint_prim)
        joint_params["damping"] = joint_damping
        joint_index = builder.add_joint_ball(**joint_params)
    elif key == UsdPhysics.ObjectType.D6Joint:
        unsupported_ref_keys = ("mujoco:dof_ref", "mujoco:dof_springref")
        unsupported_ref_attrs = [key for key in unsupported_ref_keys if key in joint_custom_attrs]
        if unsupported_ref_attrs:
            usd_attrs = ", ".join(
                "mjc:ref" if key == "mujoco:dof_ref" else "mjc:springref" for key in unsupported_ref_attrs
            )
            warnings.warn(
                f"Ignoring {usd_attrs} on native D6 joint {joint_path}: "
                "MuJoCo has no D6 joint or corresponding reference-coordinate semantics.",
                stacklevel=2,
            )
            for attr_key in unsupported_ref_attrs:
                del joint_custom_attrs[attr_key]
        joint_armature = R.get_value(
            joint_prim, prim_type=PrimType.JOINT, key="armature", default=default_joint_armature, verbose=verbose
        )
        joint_friction = R.get_value(
            joint_prim, prim_type=PrimType.JOINT, key="friction", default=default_joint_friction, verbose=verbose
        )
        joint_linear_damping, joint_angular_damping = joint_properties.resolve_joint_damping(joint_prim)
        joint_velocity_limit = R.get_value(
            joint_prim, prim_type=PrimType.JOINT, key="velocity_limit", default=None, verbose=verbose
        )
        # NewtonJointAPI uses +inf for "unlimited"; treat it as the builder default below.
        if joint_velocity_limit == float("inf"):
            joint_velocity_limit = None
        limit_ke = R.get_value(joint_prim, prim_type=PrimType.JOINT, key="limit_ke", default=None, verbose=verbose)
        limit_kd = R.get_value(joint_prim, prim_type=PrimType.JOINT, key="limit_kd", default=None, verbose=verbose)
        linear_axes = []
        angular_axes = []
        num_dofs = 0
        # Store initial state for D6 joints
        d6_initial_positions = {}
        d6_initial_velocities = {}
        # Track which axes were added as DOFs (in order)
        d6_dof_axes = []
        linear_solref_modes: list[int] = []
        angular_solref_modes: list[int] = []
        # print(joint_desc.jointLimits, joint_desc.jointDrives)
        # print(joint_desc.body0)
        # print(joint_desc.body1)
        # print(joint_desc.jointLimits)
        # print("Limits")
        # for limit in joint_desc.jointLimits:
        #     print("joint_path :", joint_path, limit.first, limit.second.lower, limit.second.upper)
        # print("Drives")
        # for drive in joint_desc.jointDrives:
        #     print("joint_path :", joint_path, drive.first, drive.second.targetPosition, drive.second.targetVelocity)

        for limit in joint_desc.jointLimits:
            dof = limit.first
            if limit.second.enabled:
                limit_lower = limit.second.lower
                limit_upper = limit.second.upper
            else:
                limit_lower = builder.default_joint_cfg.limit_lower
                limit_upper = builder.default_joint_cfg.limit_upper

            free_axis = limit_lower < limit_upper

            def define_joint_targets(dof, joint_desc):
                target_pos = (
                    0.0  # TODO: parse target from state:*:physics:appliedForce usd attribute when no drive is present
                )
                target_vel = 0.0
                target_ke = 0.0
                target_kd = 0.0
                effort_limit = np.inf
                has_drive = False
                for drive in joint_desc.jointDrives:
                    if drive.first != dof:
                        continue
                    if drive.second.enabled:
                        has_drive = True
                        target_vel = drive.second.targetVelocity
                        target_pos = drive.second.targetPosition
                        target_ke = drive.second.stiffness
                        target_kd = drive.second.damping
                        effort_limit = drive.second.forceLimit
                actuator_mode = JointTargetMode.from_gains(
                    target_ke, target_kd, force_position_velocity_actuation, has_drive=has_drive
                )
                return target_pos, target_vel, target_ke, target_kd, effort_limit, actuator_mode

            target_pos, target_vel, target_ke, target_kd, effort_limit, actuator_mode = define_joint_targets(
                dof, joint_desc
            )

            _trans_axes = {
                UsdPhysics.JointDOF.TransX: (1.0, 0.0, 0.0),
                UsdPhysics.JointDOF.TransY: (0.0, 1.0, 0.0),
                UsdPhysics.JointDOF.TransZ: (0.0, 0.0, 1.0),
            }
            _rot_axes = {
                UsdPhysics.JointDOF.RotX: (1.0, 0.0, 0.0),
                UsdPhysics.JointDOF.RotY: (0.0, 1.0, 0.0),
                UsdPhysics.JointDOF.RotZ: (0.0, 0.0, 1.0),
            }
            _rot_names = {
                UsdPhysics.JointDOF.RotX: "rotX",
                UsdPhysics.JointDOF.RotY: "rotY",
                UsdPhysics.JointDOF.RotZ: "rotZ",
            }
            if free_axis and dof in _trans_axes:
                # Per-axis translation names: transX/transY/transZ
                trans_name = {
                    UsdPhysics.JointDOF.TransX: "transX",
                    UsdPhysics.JointDOF.TransY: "transY",
                    UsdPhysics.JointDOF.TransZ: "transZ",
                }[dof]
                # Store initial state for this axis
                d6_initial_positions[trans_name] = R.get_value(
                    joint_prim,
                    PrimType.JOINT,
                    f"{trans_name}_position",
                    default=None,
                    verbose=verbose,
                )
                d6_initial_velocities[trans_name] = R.get_value(
                    joint_prim,
                    PrimType.JOINT,
                    f"{trans_name}_velocity",
                    default=None,
                    verbose=verbose,
                )
                fallback_limit_ke, limit_ke_source = joint_properties.resolve_joint_limit_gain(
                    joint_prim,
                    f"limit_{trans_name}_ke",
                    default_joint_limit_ke,
                )
                fallback_limit_kd, limit_kd_source = joint_properties.resolve_joint_limit_gain(
                    joint_prim,
                    f"limit_{trans_name}_kd",
                    default_joint_limit_kd,
                )
                current_joint_limit_ke, limit_ke_source = _resolve_newton_limit_ke(
                    limit_ke, fallback_limit_ke, limit_ke_source, default_joint_limit_ke
                )
                current_joint_limit_kd, limit_kd_source = _resolve_newton_limit_kd(
                    limit_ke, limit_kd, fallback_limit_kd, limit_kd_source, default_joint_limit_kd
                )
                linear_axes.append(
                    ModelBuilder.JointDofConfig(
                        axis=_trans_axes[dof],
                        limit_lower=limit_lower,
                        limit_upper=limit_upper,
                        limit_ke=current_joint_limit_ke,
                        limit_kd=current_joint_limit_kd,
                        target_pos=target_pos,
                        target_vel=target_vel,
                        target_ke=target_ke,
                        target_kd=target_kd,
                        damping=joint_linear_damping,
                        armature=joint_armature,
                        effort_limit=effort_limit,
                        velocity_limit=joint_velocity_limit
                        if joint_velocity_limit is not None
                        else default_joint_velocity_limit,
                        friction=joint_friction,
                        actuator_mode=actuator_mode,
                    )
                )
                linear_solref_modes.append(
                    joint_properties.joint_limit_solref_mode(joint_prim, limit_ke_source, limit_kd_source)
                )
                # Track that this axis was added as a DOF
                d6_dof_axes.append(trans_name)
            elif free_axis and dof in _rot_axes:
                # Resolve per-axis rotational gains
                rot_name = _rot_names[dof]
                # Store initial state for this axis
                d6_initial_positions[rot_name] = R.get_value(
                    joint_prim,
                    PrimType.JOINT,
                    f"{rot_name}_position",
                    default=None,
                    verbose=verbose,
                )
                d6_initial_velocities[rot_name] = R.get_value(
                    joint_prim,
                    PrimType.JOINT,
                    f"{rot_name}_velocity",
                    default=None,
                    verbose=verbose,
                )
                fallback_limit_ke, limit_ke_source = joint_properties.resolve_joint_limit_gain(
                    joint_prim,
                    f"limit_{rot_name}_ke",
                    default_joint_limit_ke * DegreesToRadian,
                )
                fallback_limit_kd, limit_kd_source = joint_properties.resolve_joint_limit_gain(
                    joint_prim,
                    f"limit_{rot_name}_kd",
                    default_joint_limit_kd * DegreesToRadian,
                )
                current_joint_limit_ke, limit_ke_source = _resolve_newton_limit_ke(
                    limit_ke,
                    fallback_limit_ke,
                    limit_ke_source,
                    default_joint_limit_ke * DegreesToRadian,
                )
                current_joint_limit_kd, limit_kd_source = _resolve_newton_limit_kd(
                    limit_ke,
                    limit_kd,
                    fallback_limit_kd,
                    limit_kd_source,
                    default_joint_limit_kd * DegreesToRadian,
                )

                angular_axes.append(
                    ModelBuilder.JointDofConfig(
                        axis=_rot_axes[dof],
                        limit_lower=limit_lower * DegreesToRadian,
                        limit_upper=limit_upper * DegreesToRadian,
                        limit_ke=current_joint_limit_ke / DegreesToRadian,
                        limit_kd=current_joint_limit_kd / DegreesToRadian,
                        target_pos=target_pos * DegreesToRadian,
                        target_vel=target_vel * DegreesToRadian,
                        target_ke=target_ke / DegreesToRadian / joint_drive_gains_scaling,
                        target_kd=target_kd / DegreesToRadian / joint_drive_gains_scaling,
                        damping=joint_angular_damping,
                        armature=joint_armature,
                        effort_limit=effort_limit,
                        velocity_limit=joint_velocity_limit * DegreesToRadian
                        if joint_velocity_limit is not None
                        else default_joint_velocity_limit,
                        friction=joint_friction,
                        actuator_mode=actuator_mode,
                    )
                )
                angular_solref_modes.append(
                    joint_properties.joint_limit_solref_mode(joint_prim, limit_ke_source, limit_kd_source)
                )
                # Track that this axis was added as a DOF
                d6_dof_axes.append(rot_name)
                num_dofs += 1

        if _should_write_solreflimit_mode():
            joint_custom_attrs[solreflimit_mode_key] = linear_solref_modes + angular_solref_modes
        if _should_write_solreflimit_gain_baseline():
            joint_custom_attrs[solreflimit_gain_baseline_key] = [
                wp.vec2(axis.limit_ke, axis.limit_kd) for axis in [*linear_axes, *angular_axes]
            ]

        joint_index = builder.add_joint_d6(**joint_params, linear_axes=linear_axes, angular_axes=angular_axes)
    elif key == UsdPhysics.ObjectType.DistanceJoint:
        joint_index = builder.add_joint_distance(
            **joint_params,
            min_distance=joint_desc.limit.lower if joint_desc.minEnabled else -1.0,
            max_distance=joint_desc.limit.upper if joint_desc.maxEnabled else -1.0,
        )
    else:
        raise NotImplementedError(f"Unsupported joint type {key}")

    if joint_index is None:
        raise ValueError(f"Failed to add joint {joint_path}")

    # map the joint path to the index at insertion time
    path_joint_map[joint_path] = joint_index

    # Apply saved initial joint state after joint creation
    if key in (UsdPhysics.ObjectType.RevoluteJoint, UsdPhysics.ObjectType.PrismaticJoint):
        joint_type_str = "revolute" if key == UsdPhysics.ObjectType.RevoluteJoint else "prismatic"
        if initial_position is not None:
            builder.joint_q[builder.joint_q_start[joint_index]] = initial_position
            if verbose:
                unit = "rad" if key == UsdPhysics.ObjectType.RevoluteJoint else "m"
                print(f"Set {joint_type_str} joint {joint_index} position to {initial_position} ({unit})")
        if initial_velocity is not None:
            builder.joint_qd[builder.joint_qd_start[joint_index]] = initial_velocity
            if verbose:
                unit = "rad/s" if key == UsdPhysics.ObjectType.RevoluteJoint else "m/s"
                print(f"Set {joint_type_str} joint {joint_index} velocity to {initial_velocity} {unit}")
    elif key == UsdPhysics.ObjectType.D6Joint:
        # Apply D6 joint initial state
        q_start = builder.joint_q_start[joint_index]
        qd_start = builder.joint_qd_start[joint_index]

        # Get joint coordinate and DOF ranges
        if joint_index + 1 < len(builder.joint_q_start):
            q_end = builder.joint_q_start[joint_index + 1]
            qd_end = builder.joint_qd_start[joint_index + 1]
        else:
            q_end = len(builder.joint_q)
            qd_end = len(builder.joint_qd)

        # Apply initial values for each axis that was actually added as a DOF
        for dof_idx, axis_name in enumerate(d6_dof_axes):
            if dof_idx >= (qd_end - qd_start):
                break

            is_rot = axis_name.startswith("rot")
            pos = d6_initial_positions.get(axis_name)
            vel = d6_initial_velocities.get(axis_name)

            if pos is not None and q_start + dof_idx < q_end:
                coord_val = pos * DegreesToRadian if is_rot else pos
                builder.joint_q[q_start + dof_idx] = coord_val
                if verbose:
                    print(f"Set D6 joint {joint_index} {axis_name} position to {pos} ({'deg' if is_rot else 'm'})")

            if vel is not None and qd_start + dof_idx < qd_end:
                vel_val = vel  # D6 velocities are already in correct units
                builder.joint_qd[qd_start + dof_idx] = vel_val
                if verbose:
                    print(f"Set D6 joint {joint_index} {axis_name} velocity to {vel} rad/s")

    return joint_index


def parse_merged_joints(
    joint_paths: list[str],
    incoming_xform: wp.transform | None = None,
    *,
    builder: ModelBuilder,
    stage: Usd.Stage,
    R: SchemaResolverManager,
    joint_properties: _UsdJointProperties,
    joint_descriptions: dict[str, UsdPhysics.JointDesc],
    path_body_map: dict[str, int],
    path_joint_map: dict[str, int],
    merged_dof_offset: dict[str, int],
    builder_custom_attr_joint: list[ModelBuilder.CustomAttribute],
    physics_scene_prim: Usd.Prim | None,
    usd_axis_to_axis: dict[UsdPhysics.Axis, Axis],
    default_joint_velocity_limit: float,
    joint_drive_gains_scaling: float,
    force_position_velocity_actuation: bool,
    only_load_enabled_joints: bool,
    collect_schema_attrs: bool,
    verbose: bool,
    solreflimit_mode_key: str,
    solreflimit_gain_baseline_key: str,
    _should_write_solreflimit_mode: Callable[[], bool],
    _should_write_solreflimit_gain_baseline: Callable[[], bool],
    resolve_joint_parent_child: Callable[..., Any],
) -> int | None:
    """Combine multiple single-DOF joints between the same two bodies into one D6 joint.

    This handles USD files where multi-DOF MuJoCo joints are represented as
    separate PhysicsRevoluteJoint / PhysicsPrismaticJoint prims connecting the
    same parent and child bodies.  The individual joints are merged into a
    single :func:`~newton.ModelBuilder.add_joint_d6` call, following the same
    pattern used by the MJCF importer.

    Args:
        joint_paths: Prim paths of the joints to merge (all must share the
            same body pair).
        incoming_xform: Optional world-space transform applied to the parent
            frame of the first joint.

    Returns:
        The builder joint index of the newly created D6 joint, or ``None`` if
        all joints in the group are disabled.
    """
    from pxr import UsdPhysics

    linear_axes: list[ModelBuilder.JointDofConfig] = []
    angular_axes: list[ModelBuilder.JointDofConfig] = []
    # Track prim paths and initial state separately for linear/angular DOFs
    # because add_joint_d6 orders linear DOFs first, then angular
    linear_prim_paths: list[str] = []
    angular_prim_paths: list[str] = []
    linear_initial_pos: list[float | None] = []
    linear_initial_vel: list[float | None] = []
    angular_initial_pos: list[float | None] = []
    angular_initial_vel: list[float | None] = []
    enabled_count = 0
    collision_filter_parent = False

    # Find the first enabled joint to use as representative for transforms and metadata
    first_desc = None
    first_prim = None
    for jp in joint_paths:
        jd = joint_descriptions[jp]
        if not jd.jointEnabled and only_load_enabled_joints:
            continue
        first_desc = jd
        first_prim = stage.GetPrimAtPath(jd.primPath)
        break
    if first_desc is None:
        return None  # all joints disabled

    parent_id, child_id, parent_tf, child_tf = resolve_joint_parent_child(  # pyright: ignore[reportAssignmentType]
        first_desc, path_body_map, get_transforms=True
    )
    if incoming_xform is not None:
        parent_tf = incoming_xform * parent_tf

    # Warn if any sibling joint has a different anchor position.
    # Different local rotations are expected (they encode different DOF axis directions)
    # and are handled by remapping axes into the representative frame.
    for jp in joint_paths:
        jd = joint_descriptions[jp]
        if jd is first_desc:
            continue
        _, _, other_parent_tf, other_child_tf = resolve_joint_parent_child(  # pyright: ignore[reportAssignmentType]
            jd, path_body_map, get_transforms=True
        )
        parent_pos_match = np.allclose(parent_tf.p, other_parent_tf.p, atol=1e-6)
        child_pos_match = np.allclose(child_tf.p, other_child_tf.p, atol=1e-6)
        if not (parent_pos_match and child_pos_match):
            warnings.warn(
                f"Merged joint {jp} has different anchor positions than representative "
                f"{first_desc.primPath}; using representative positions for the D6 joint.",
                stacklevel=2,
            )
            break

    # Split custom attributes into joint-level (one value per joint) and
    # per-DOF (one value per DOF).  Joint-level attrs come from the
    # representative prim; per-DOF attrs are collected from each sibling.
    joint_freq_attrs = [a for a in builder_custom_attr_joint if a.frequency == AttributeFrequency.JOINT]
    dof_freq_attrs = [
        a
        for a in builder_custom_attr_joint
        if a.frequency in (AttributeFrequency.JOINT_DOF, AttributeFrequency.JOINT_COORD)
    ]
    joint_custom_attrs = usd.get_custom_attribute_values(
        first_prim,
        joint_freq_attrs,
        context={"builder": builder, "physics_scene_prim": physics_scene_prim},
    )
    # Per-DOF custom attributes accumulated separately for linear / angular
    # so we can reorder to D6 DOF order (linear first, then angular).
    linear_dof_custom: list[dict[str, Any]] = []
    angular_dof_custom: list[dict[str, Any]] = []

    # Cache the representative parent-side rotation for axis remapping
    rep_parent_rot = np.array(parent_tf.q, dtype=float)

    for jp in joint_paths:
        jd = joint_descriptions[jp]
        if not jd.jointEnabled and only_load_enabled_joints:
            continue
        collision_filter_parent = collision_filter_parent or not jd.collisionEnabled
        jp_prim = stage.GetPrimAtPath(jd.primPath)
        if collect_schema_attrs:
            R.collect_prim_attrs(jp_prim)

        key = jd.type
        if key not in (UsdPhysics.ObjectType.RevoluteJoint, UsdPhysics.ObjectType.PrismaticJoint):
            raise ValueError(
                f"Cannot merge joint {jp} of type {key} into a D6 joint. "
                "Only RevoluteJoint and PrismaticJoint are supported for merging."
            )

        is_revolute = key == UsdPhysics.ObjectType.RevoluteJoint
        dof = joint_properties.resolve_dof_params(
            jp_prim,
            jd,
            is_revolute,
            joint_drive_gains_scaling=joint_drive_gains_scaling,
            force_position_velocity_actuation=force_position_velocity_actuation,
        )
        initial_position = dof.initial_position
        initial_velocity = dof.initial_velocity

        # Collect per-DOF custom attributes before constructing the D6
        # axis so MuJoCo reference offsets can be applied to its limits.
        sibling_dof_attrs = usd.get_custom_attribute_values(
            jp_prim,
            dof_freq_attrs,
            context={"builder": builder, "physics_scene_prim": physics_scene_prim},
        )
        _shift_joint_limits_for_reference(dof, sibling_dof_attrs)
        if _should_write_solreflimit_mode():
            sibling_dof_attrs[solreflimit_mode_key] = dof.limit_solref_mode
        if _should_write_solreflimit_gain_baseline():
            sibling_dof_attrs[solreflimit_gain_baseline_key] = wp.vec2(dof.limit_ke, dof.limit_kd)

        # Compute the DOF axis in the representative joint's frame.
        # Each USD joint may have a different localRot that orients its fixed axis
        # (X, Y, or Z) to the physical DOF direction.  We remap into the rep frame.
        _, _, jp_parent_tf, _ = resolve_joint_parent_child(  # pyright: ignore[reportAssignmentType]
            jd, path_body_map, get_transforms=True
        )
        jp_parent_rot = np.array(jp_parent_tf.q, dtype=float)
        # q and -q represent the same rotation
        if abs(np.dot(rep_parent_rot, jp_parent_rot)) > 1.0 - 1e-6:
            # Same rotation — use the original axis directly
            dof_axis = usd_axis_to_axis[jd.axis]
        else:
            # Different rotation — transform axis into rep frame
            rep_q_inv = wp.quat_inverse(wp.quat(*rep_parent_rot.tolist()))
            jp_q = wp.quat(*jp_parent_rot.tolist())
            relative_q = wp.mul(rep_q_inv, jp_q)
            # Axis enum value: 0=X, 1=Y, 2=Z → unit vector
            axis_idx = int(usd_axis_to_axis[jd.axis])
            axis_unit = [0.0, 0.0, 0.0]
            axis_unit[axis_idx] = 1.0
            rotated = wp.quat_rotate(relative_q, wp.vec3(axis_unit[0], axis_unit[1], axis_unit[2]))
            dof_axis = (float(rotated[0]), float(rotated[1]), float(rotated[2]))

        ax = ModelBuilder.JointDofConfig(
            axis=dof_axis,
            limit_lower=dof.limit_lower,
            limit_upper=dof.limit_upper,
            limit_ke=dof.limit_ke,
            limit_kd=dof.limit_kd,
            target_pos=dof.target_pos,
            target_vel=dof.target_vel,
            target_ke=dof.target_ke,
            target_kd=dof.target_kd,
            damping=dof.damping,
            armature=dof.armature,
            friction=dof.friction,
            effort_limit=dof.effort_limit,
            velocity_limit=dof.velocity_limit if dof.velocity_limit is not None else default_joint_velocity_limit,
            actuator_mode=dof.actuator_mode,
        )

        if is_revolute:
            angular_axes.append(ax)
            angular_prim_paths.append(jp)
            angular_initial_pos.append(initial_position)
            angular_initial_vel.append(initial_velocity)
            angular_dof_custom.append(sibling_dof_attrs)
        else:
            linear_axes.append(ax)
            linear_prim_paths.append(jp)
            linear_initial_pos.append(initial_position)
            linear_initial_vel.append(initial_velocity)
            linear_dof_custom.append(sibling_dof_attrs)

        enabled_count += 1

    if enabled_count == 0:
        return None

    # D6 DOF order: linear first, then angular
    dof_prim_paths = linear_prim_paths + angular_prim_paths
    dof_initial_pos = linear_initial_pos + angular_initial_pos
    dof_initial_vel = linear_initial_vel + angular_initial_vel
    ordered_dof_custom = linear_dof_custom + angular_dof_custom

    # Merge per-DOF custom attributes into DOF-indexed dicts for add_joint_d6.
    # Each entry in ordered_dof_custom is a dict of {attr_key: value} from one sibling prim.
    # We assemble {attr_key: {dof_index: value}} so _process_joint_custom_attributes
    # assigns each DOF its own value instead of broadcasting from the representative.
    for dof_idx, dof_attrs in enumerate(ordered_dof_custom):
        for attr_key, value in dof_attrs.items():
            if attr_key not in joint_custom_attrs:
                joint_custom_attrs[attr_key] = {}
            existing = joint_custom_attrs[attr_key]
            if not isinstance(existing, dict):
                # First per-DOF value for an attr that was already set as a scalar
                # from the representative — convert to a dict to allow per-DOF override.
                joint_custom_attrs[attr_key] = {dof_idx: value}
            else:
                existing[dof_idx] = value

    # Use the representative (first enabled) joint path as the D6 joint label
    label = str(first_desc.primPath)

    # Register original prim paths as DOF labels so MjcActuator targets resolve correctly
    if "mujoco:joint_dof_label" in builder.custom_attributes:
        joint_custom_attrs["mujoco:joint_dof_label"] = dof_prim_paths

    joint_index = builder.add_joint_d6(
        parent=parent_id,
        child=child_id,
        linear_axes=linear_axes if linear_axes else None,
        angular_axes=angular_axes if angular_axes else None,
        parent_xform=parent_tf,
        child_xform=child_tf,
        label=label,
        collision_filter_parent=parent_id != -1 and collision_filter_parent,
        enabled=first_desc.jointEnabled,
        custom_attributes=joint_custom_attrs,
    )

    # Register all original joint prim paths in path_joint_map and track per-path DOF offsets
    for jp in joint_paths:
        path_joint_map[jp] = joint_index
    for dof_idx, dof_path in enumerate(dof_prim_paths):
        merged_dof_offset[dof_path] = dof_idx

    # Apply initial positions/velocities
    q_start = builder.joint_q_start[joint_index]
    qd_start = builder.joint_qd_start[joint_index]
    for dof_idx, (pos, vel) in enumerate(zip(dof_initial_pos, dof_initial_vel, strict=True)):
        if pos is not None:
            builder.joint_q[q_start + dof_idx] = pos
        if vel is not None:
            builder.joint_qd[qd_start + dof_idx] = vel

    if verbose:
        print(
            f"Merged {len(joint_paths)} joints into D6 joint {joint_index}: "
            f"{len(linear_axes)} linear + {len(angular_axes)} angular DOFs"
        )

    return joint_index
