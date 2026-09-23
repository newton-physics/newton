# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Import authored USD articulations using the importer's body and joint helpers."""

from __future__ import annotations

import re
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import warp as wp

from . import utils as usd
from .schema_resolver import PrimType

if TYPE_CHECKING:
    from pxr import Sdf, Usd, UsdGeom, UsdPhysics

    from ..sim.builder import ModelBuilder
    from .schema_resolver import SchemaResolverManager


def _parse_articulations(
    builder: ModelBuilder,
    stage: Usd.Stage,
    articulation_entries: list[tuple[Sdf.Path, UsdPhysics.ArticulationDesc]],
    *,
    R: SchemaResolverManager,
    xform_cache: UsdGeom.XformCache,
    body_specs: dict[str, UsdPhysics.RigidBodyDesc],
    joint_descriptions: dict[str, UsdPhysics.JointDesc],
    ignored_body_paths: set[str],
    mjc_equality_connect_or_weld_paths: set[str],
    path_body_map: dict[str, int],
    processed_joints: set[str],
    excluded_articulation_joints: dict[str, wp.transform],
    articulation_has_self_collision: dict[int, bool],
    builder_custom_attr_articulation: list[ModelBuilder.CustomAttribute],
    incoming_world_xform: wp.transform,
    override_root_xform: bool,
    bodies_follow_joint_ordering: bool,
    joint_ordering: Literal["bfs", "dfs"] | None,
    parent_body: int,
    floating: bool | None,
    base_joint: dict[str, Any] | None,
    enable_self_collisions: bool,
    ignore_paths: list[str],
    collect_schema_attrs: bool,
    verbose: bool,
    warn_invalid_desc: Callable[..., bool],
    parse_body: Callable[..., int | dict[str, Any]],
    add_body: Callable[..., int],
    resolve_joint_parent_child: Callable[..., Any],
    parse_joint: Callable[..., int | None],
    parse_merged_joints: Callable[..., int | None],
    import_attached_cables: Callable[[list[str]], None],
    topological_sort_undirected: Callable[..., tuple[list[int], list[int]]],
) -> None:
    """Parse articulation descriptions in their existing order.

    The mutable maps and sets are shared with the remaining importer passes.
    Body and joint helpers retain their source reads and builder side effects;
    attached cables are imported immediately after articulation finalization.
    """
    from pxr import Sdf, UsdPhysics

    parent_prim = None
    body_data = {}
    for path, desc in articulation_entries:
        if warn_invalid_desc(path, desc):
            continue
        articulation_path = str(path)
        if any(re.match(p, articulation_path) for p in ignore_paths):
            continue
        articulation_prim = stage.GetPrimAtPath(path)
        articulation_root_xform = usd.get_transform(articulation_prim, local=False, xform_cache=xform_cache)
        root_joint_xform = (
            incoming_world_xform if override_root_xform else incoming_world_xform * articulation_root_xform
        )
        # Collect engine-specific attributes for the articulation root on first encounter
        if collect_schema_attrs:
            R.collect_prim_attrs(articulation_prim)
            # Also collect on the parent prim (e.g. Xform with PhysxArticulationAPI)
            try:
                parent_prim = articulation_prim.GetParent()
            except Exception:
                parent_prim = None
            if parent_prim is not None and parent_prim.IsValid():
                R.collect_prim_attrs(parent_prim)

        # Extract custom attributes for articulation frequency from the articulation root prim
        # (the one with PhysicsArticulationRootAPI, typically the articulation_prim itself or its parent)
        articulation_custom_attrs = {}
        # First check if articulation_prim itself has the PhysicsArticulationRootAPI
        if articulation_prim.HasAPI(UsdPhysics.ArticulationRootAPI):
            if verbose:
                print(f"Extracting articulation custom attributes from {articulation_prim.GetPath()}")
            articulation_custom_attrs = usd.get_custom_attribute_values(
                articulation_prim, builder_custom_attr_articulation
            )
        # If not, check the parent prim
        elif parent_prim is not None and parent_prim.IsValid() and parent_prim.HasAPI(UsdPhysics.ArticulationRootAPI):
            if verbose:
                print(f"Extracting articulation custom attributes from parent {parent_prim.GetPath()}")
            articulation_custom_attrs = usd.get_custom_attribute_values(parent_prim, builder_custom_attr_articulation)
        if verbose and articulation_custom_attrs:
            print(f"Extracted articulation custom attributes: {articulation_custom_attrs}")
        body_ids = {}
        body_labels = []
        current_body_id = 0
        art_bodies = []
        if verbose:
            print(f"Bodies under articulation {path!s}:")
        for p in desc.articulatedBodies:
            if verbose:
                print(f"\t{p!s}")
            if p == Sdf.Path.emptyPath:
                continue
            key = str(p)
            if key in ignored_body_paths:
                continue

            usd_prim = stage.GetPrimAtPath(p)
            if collect_schema_attrs:
                # Collect on each articulated body prim encountered
                R.collect_prim_attrs(usd_prim)

            if key in body_specs:
                body_desc = body_specs[key]
                desc_xform = wp.transform(body_desc.position, usd.value_to_warp(body_desc.rotation))
                body_world = usd.get_transform(usd_prim, local=False, xform_cache=xform_cache)
                if override_root_xform:
                    # Strip the articulation root's world-space pose and rebase at the user-specified xform.
                    body_in_root_frame = wp.transform_inverse(articulation_root_xform) * body_world
                    desired_world = incoming_world_xform * body_in_root_frame
                else:
                    desired_world = incoming_world_xform * body_world
                body_incoming_xform = desired_world * wp.transform_inverse(desc_xform)
                art_root_for_visuals = articulation_root_xform if override_root_xform else None
                if bodies_follow_joint_ordering:
                    # we just parse the body information without yet adding it to the builder
                    body_data[current_body_id] = parse_body(
                        body_desc,
                        stage.GetPrimAtPath(p),
                        incoming_xform=body_incoming_xform,
                        add_body_to_builder=False,
                        articulation_root_xform=art_root_for_visuals,
                    )
                else:
                    # look up description and add body to builder
                    bid: int = parse_body(  # pyright: ignore[reportAssignmentType]
                        body_desc,
                        stage.GetPrimAtPath(p),
                        incoming_xform=body_incoming_xform,
                        add_body_to_builder=True,
                        articulation_root_xform=art_root_for_visuals,
                    )
                    if bid >= 0:
                        art_bodies.append(bid)
                # remove body spec once we inserted it
                del body_specs[key]

            body_ids[key] = current_body_id
            body_labels.append(key)
            current_body_id += 1

        if len(body_ids) == 0:
            # no bodies under the articulation or we ignored all of them
            continue

        # determine the joint graph for this articulation
        joint_names: list[str] = []
        joint_edges: list[tuple[int, int]] = []
        # keys of joints that are excluded from the articulation (loop joints)
        joint_excluded: set[str] = set()
        # Groups of joints that share the same body pair (multi-DOF joints from MuJoCo USD).
        # Maps the representative joint path (first encountered) to all joint paths in the group.
        merged_joint_groups: dict[str, list[str]] = {}
        # Track which body pair maps to which representative joint path
        body_pair_to_representative: dict[tuple[int, int], str] = {}
        for p in desc.articulatedJoints:
            joint_path = str(p)
            joint_desc = joint_descriptions[joint_path]
            if joint_path in mjc_equality_connect_or_weld_paths:
                if verbose:
                    print(f"Skipping equality connect/weld joint '{joint_path}' from articulation joint graph")
                continue
            # it may be possible that a joint is filtered out in the middle of
            # a chain of joints, which results in a disconnected graph
            # we should raise an error in this case
            if any(re.match(p, joint_path) for p in ignore_paths):
                continue
            if str(joint_desc.body0) in ignored_body_paths:
                continue
            if str(joint_desc.body1) in ignored_body_paths:
                continue
            parent_id, child_id = resolve_joint_parent_child(joint_desc, body_ids, get_transforms=False)  # pyright: ignore[reportAssignmentType]
            if joint_desc.excludeFromArticulation:
                joint_excluded.add(joint_path)
            else:
                body_pair = (parent_id, child_id)
                if body_pair in body_pair_to_representative:
                    # Another joint between the same bodies — merge into existing group
                    rep = body_pair_to_representative[body_pair]
                    merged_joint_groups[rep].append(joint_path)
                else:
                    # First joint for this body pair
                    body_pair_to_representative[body_pair] = joint_path
                    merged_joint_groups[joint_path] = [joint_path]
                    joint_edges.append(body_pair)
                    joint_names.append(joint_path)

        articulation_joint_indices = []
        articulation_ids: set[int] = set()

        if len(joint_edges) == 0:
            # We have an articulation without joints, i.e. only free rigid bodies
            # Use add_base_joint to honor floating, base_joint, and parent_body parameters
            base_parent = parent_body
            if bodies_follow_joint_ordering:
                for i in body_ids.values():
                    child_body_id = add_body(**body_data[i])
                    # Compute parent_xform to preserve imported pose when attaching to parent_body
                    parent_xform = None
                    if base_parent != -1:
                        # When parent_body is specified, interpret xform parameter as parent-relative offset
                        # body_data[i]["xform"] = USD_local * incoming_world_xform
                        # We want parent_xform to position the child at this location relative to parent
                        # Use incoming_world_xform as the base parent-relative offset
                        parent_xform = incoming_world_xform
                        # If the USD body has a non-identity local transform, compose it with incoming_xform
                        # Note: incoming_world_xform already includes the child's USD local transform via body_incoming_xform
                        # So we can use body_data[i]["xform"] directly for the intended position
                        # But we need it relative to parent. Since parent's body_q may not reflect joint offsets,
                        # we interpret body_data[i]["xform"] as the intended parent-relative transform directly.
                        # For articulations without joints, incoming_world_xform IS the parent-relative offset.
                        parent_xform = incoming_world_xform
                    joint_id = builder._add_base_joint(
                        child_body_id,
                        floating=floating,
                        base_joint=base_joint,
                        parent=base_parent,
                        parent_xform=parent_xform,
                    )
                    # note the free joint's coordinates will be initialized by the body_q of the
                    # child body
                    builder._finalize_imported_articulation(
                        joint_indices=[joint_id],
                        parent_body=parent_body,
                        articulation_label=body_data[i]["label"],
                        custom_attributes=articulation_custom_attrs,
                    )
                    articulation_ids.add(builder.joint_articulation[joint_id])
                    import_attached_cables([body_data[i]["label"]])
            else:
                for i, child_body_id in enumerate(art_bodies):
                    # Compute parent_xform to preserve imported pose when attaching to parent_body
                    parent_xform = None
                    if base_parent != -1:
                        # When parent_body is specified, interpret xform parameter as parent-relative offset
                        parent_xform = incoming_world_xform
                    joint_id = builder._add_base_joint(
                        child_body_id,
                        floating=floating,
                        base_joint=base_joint,
                        parent=base_parent,
                        parent_xform=parent_xform,
                    )
                    # note the free joint's coordinates will be initialized by the body_q of the
                    # child body
                    builder._finalize_imported_articulation(
                        joint_indices=[joint_id],
                        parent_body=parent_body,
                        articulation_label=body_labels[i],
                        custom_attributes=articulation_custom_attrs,
                    )
                    articulation_ids.add(builder.joint_articulation[joint_id])
                    import_attached_cables([body_labels[i]])
            sorted_joints = []
        else:
            # we have an articulation with joints, we need to sort them topologically
            if joint_ordering is not None:
                if verbose:
                    print(f"Sorting joints using {joint_ordering} ordering...")
                sorted_joints, reversed_joint_list = topological_sort_undirected(
                    joint_edges, use_dfs=joint_ordering == "dfs", ensure_single_root=True
                )
                if reversed_joint_list:
                    reversed_joint_paths = [joint_names[joint_id] for joint_id in reversed_joint_list]
                    reversed_joint_names = ", ".join(reversed_joint_paths)
                    raise ValueError(
                        f"Reversed joints are not supported: {reversed_joint_names}. Ensure that the joint parent body is defined as physics:body0 and the child is defined as physics:body1 in the joint prim."
                    )
                if verbose:
                    print("Joint ordering:", sorted_joints)
            else:
                # we keep the original order of the joints
                sorted_joints = np.arange(len(joint_names))

        if len(sorted_joints) > 0:
            # insert the bodies in the order of the joints
            if bodies_follow_joint_ordering:
                inserted_bodies = set()
                for jid in sorted_joints:
                    parent, child = joint_edges[jid]
                    if parent >= 0 and parent not in inserted_bodies:
                        b = add_body(**body_data[parent])
                        inserted_bodies.add(parent)
                        art_bodies.append(b)
                        path_body_map[body_data[parent]["label"]] = b
                    if child >= 0 and child not in inserted_bodies:
                        b = add_body(**body_data[child])
                        inserted_bodies.add(child)
                        art_bodies.append(b)
                        path_body_map[body_data[child]["label"]] = b

            first_joint_parent = joint_edges[sorted_joints[0]][0]
            if first_joint_parent != -1:
                # the mechanism is floating since there is no joint connecting it to the world
                # we explicitly add a joint connecting the first body in the articulation to the world
                # (or to parent_body if specified) to make sure generalized-coordinate solvers can simulate it
                base_parent = parent_body
                if bodies_follow_joint_ordering:
                    child_body = body_data[first_joint_parent]
                    child_body_id = path_body_map[child_body["label"]]
                else:
                    child_body_id = art_bodies[first_joint_parent]
                # Compute parent_xform to preserve imported pose when attaching to parent_body
                parent_xform = None
                if base_parent != -1:
                    # When parent_body is specified, use incoming_world_xform as parent-relative offset
                    parent_xform = incoming_world_xform
                base_joint_id = builder._add_base_joint(
                    child_body_id,
                    floating=floating,
                    base_joint=base_joint,
                    parent=base_parent,
                    parent_xform=parent_xform,
                )
                articulation_joint_indices.append(base_joint_id)

            # insert the remaining joints in topological order
            for joint_id, i in enumerate(sorted_joints):
                if joint_id == 0 and first_joint_parent == -1:
                    # The root joint connects to the world (parent_id=-1).
                    # If base_joint or floating is specified, override the USD's root joint.
                    if base_joint is not None or floating is not None:
                        # Get the child body of the root joint
                        root_joint_child = joint_edges[sorted_joints[0]][1]
                        if bodies_follow_joint_ordering:
                            child_body = body_data[root_joint_child]
                            child_body_id = path_body_map[child_body["label"]]
                        else:
                            child_body_id = art_bodies[root_joint_child]
                        base_parent = parent_body
                        # Compute parent_xform to preserve imported pose
                        parent_xform = None
                        if base_parent != -1:
                            # When parent_body is specified, use incoming_world_xform as parent-relative offset
                            parent_xform = incoming_world_xform
                        else:
                            # body_q is already in world space, use it directly
                            parent_xform = builder.body_q[child_body_id]
                        base_joint_id = builder._add_base_joint(
                            child_body_id,
                            floating=floating,
                            base_joint=base_joint,
                            parent=base_parent,
                            parent_xform=parent_xform,
                        )
                        articulation_joint_indices.append(base_joint_id)
                        group = merged_joint_groups.get(joint_names[i])
                        if group is not None:
                            processed_joints.update(group)
                        else:
                            processed_joints.add(joint_names[i])
                        continue  # Skip parsing the USD's root joint
                    # When body0 maps to world the physics API may resolve
                    # localPose0 into world space (baking the non-body prim's
                    # transform). JointDesc.body0 returns "" for non-rigid
                    # targets, so we attempt to look up the prim directly.
                    root_joint_desc = joint_descriptions[joint_names[i]]
                    b0 = str(root_joint_desc.body0)
                    b1 = str(root_joint_desc.body1)
                    # Determine the world-facing side from this articulation's body set.
                    # path_body_map includes previously imported articulations, so using
                    # it here can misidentify the world-side path for the current root
                    # joint when b0 references an external rigid body.
                    if b0 not in body_ids:
                        world_body_path = b0
                    elif b1 not in body_ids:
                        world_body_path = b1
                    else:
                        # Defensive fallback; root joints should have exactly one side
                        # outside the articulation.
                        world_body_path = b0
                    world_body_prim = stage.GetPrimAtPath(world_body_path) if world_body_path else None
                    if world_body_prim is not None and world_body_prim.IsValid():
                        world_body_xform = usd.get_transform(world_body_prim, local=False, xform_cache=xform_cache)
                    else:
                        # body0/body1 can resolve to world with an empty path (""),
                        # leaving no world-side prim to query.
                        # If the authored world-side local pose is identity, recover
                        # the missing world-side frame from the resolved child body
                        # pose and local poses so root-joint FK stays consistent with
                        # imported body_q.
                        # If the world-side local pose is non-identity, keep the
                        # previous identity fallback: USD often bakes non-rigid world
                        # anchors directly into localPose0/localPose1 in this case.
                        _, child_local_id, parent_tf, child_tf = resolve_joint_parent_child(  # pyright: ignore[reportAssignmentType]
                            root_joint_desc,
                            body_ids,
                            get_transforms=True,
                        )
                        assert parent_tf is not None and child_tf is not None
                        identity_tf = wp.transform_identity()
                        parent_pos = np.array(parent_tf.p, dtype=float)
                        parent_quat = np.array(parent_tf.q, dtype=float)
                        identity_pos = np.array(identity_tf.p, dtype=float)
                        identity_quat = np.array(identity_tf.q, dtype=float)
                        parent_pos_is_identity = np.allclose(parent_pos, identity_pos, atol=1e-6)
                        # q and -q represent the same rotation
                        parent_rot_is_identity = abs(np.dot(parent_quat, identity_quat)) > 1.0 - 1e-6
                        if parent_pos_is_identity and parent_rot_is_identity and 0 <= child_local_id < len(body_labels):
                            child_path = body_labels[child_local_id]
                            child_prim = stage.GetPrimAtPath(child_path)
                        else:
                            child_prim = None
                        if child_prim is not None and child_prim.IsValid():
                            child_world_xform = usd.get_transform(child_prim, local=False, xform_cache=xform_cache)
                            world_body_xform = child_world_xform * child_tf * wp.transform_inverse(parent_tf)
                        else:
                            world_body_xform = wp.transform_identity()
                    root_frame_xform = (
                        wp.transform_inverse(articulation_root_xform)
                        if override_root_xform
                        else wp.transform_identity()
                    )
                    root_incoming_xform = incoming_world_xform * root_frame_xform * world_body_xform
                    group = merged_joint_groups.get(joint_names[i])
                    if group is not None and len(group) > 1:
                        joint = parse_merged_joints(group, incoming_xform=root_incoming_xform)
                    else:
                        joint = parse_joint(
                            joint_descriptions[joint_names[i]],
                            incoming_xform=root_incoming_xform,
                        )
                else:
                    group = merged_joint_groups.get(joint_names[i])
                    if group is not None and len(group) > 1:
                        joint = parse_merged_joints(group)
                    else:
                        joint = parse_joint(
                            joint_descriptions[joint_names[i]],
                        )
                if joint is not None:
                    articulation_joint_indices.append(joint)
                    processed_joints.add(joint_names[i])
                    # Mark all paths in the group as processed
                    group = merged_joint_groups.get(joint_names[i])
                    if group is not None:
                        for gp in group:
                            processed_joints.add(gp)

        # Create the articulation from all collected joints
        if articulation_joint_indices:
            builder._finalize_imported_articulation(
                joint_indices=articulation_joint_indices,
                parent_body=parent_body,
                articulation_label=articulation_path,
                custom_attributes=articulation_custom_attrs,
            )
            articulation_ids.add(builder.joint_articulation[articulation_joint_indices[0]])
            import_attached_cables(body_labels)

        # Defer external constraints until later bodies and cables have extended their
        # articulations. Reserve these paths so the orphan-joint pass does not emit them.
        for joint_path in sorted(joint_excluded):
            excluded_articulation_joints[joint_path] = root_joint_xform
        processed_joints.update(joint_excluded)

        self_collisions = bool(
            R.get_value(
                articulation_prim,
                prim_type=PrimType.ARTICULATION,
                key="self_collision_enabled",
                default=enable_self_collisions,
                verbose=verbose,
            )
        )
        for articulation in articulation_ids:
            articulation_has_self_collision[articulation] = self_collisions
