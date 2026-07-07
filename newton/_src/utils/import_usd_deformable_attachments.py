# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""USD attachment / element-collision-filter import passes.

Lowers ``PhysicsAttachment`` prims onto imported cables (as ball joints to xform targets) and
``PhysicsElementCollisionFilter`` prims to shape collision-filter pairs, plus the post-collapse
index remap. Driven by :func:`.import_usd.parse_usd` via a
:class:`.import_usd_deformable_utils._DeformableImportContext`.
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Mapping, Sequence
from typing import Any

import warp as wp

from .import_usd_deformable_utils import (
    _attachment_vec3_list,
    _attachment_vec3_tuples,
    _builder_body_xform,
    _cable_attachment_anchors,
    _DeformableImportContext,
    _is_ignored_path,
    _mark_attachment_unsupported,
)


def _deformable_import_attachments(ctx: _DeformableImportContext, consumed_junction_attachment_paths: set[str]) -> None:
    """Lower supported AOUSD ``PhysicsAttachment`` prims onto the imported cables.

    Cable ``point`` / ``segment`` sites with ``type1 = "xform"`` become hard ball joints to the
    target xform / rigid body / world frame (``path_attachment_map``); curve-to-curve junctions
    already consumed as rod-graph topology are skipped, and unsupported sites (cloth/volume source,
    non-xform target, ...) are warned and preserved in ``path_attachment_attrs``.
    """
    builder = ctx.builder
    stage = ctx.stage
    root_prim = ctx.root_prim
    ignore_paths = ctx.ignore_paths
    incoming_world_xform = ctx.incoming_world_xform
    verbose = ctx.verbose
    deformable_read = ctx.deformable_read
    get_prim_world_mat = ctx.get_prim_world_mat
    get_rigid_body_ancestor_path = ctx.get_rigid_body_ancestor_path
    get_first_target = ctx.get_first_target
    path_body_map = ctx.path_body_map
    path_cable_segments = ctx.path_cable_segments
    path_cable_point_anchors = ctx.path_cable_point_anchors
    path_cloth_map = ctx.path_cloth_map
    path_soft_map = ctx.path_soft_map
    path_attachment_map = ctx.path_attachment_map
    path_attachment_attrs = ctx.path_attachment_attrs

    def _attachment_world_point_from_xform(target_path: str, local_point: wp.vec3) -> tuple[int, wp.vec3] | None:
        if target_path in ("", "/"):
            # A world target's coords are authored in stage space, so they ride the same
            # import/up-axis transform applied to the cable geometry (otherwise the anchor
            # stays in original USD coordinates and yanks a translated cable off-position).
            return -1, wp.transform_point(incoming_world_xform, local_point)

        target_prim = stage.GetPrimAtPath(target_path)
        if not target_prim or not target_prim.IsValid():
            return None

        target_mat = get_prim_world_mat(target_prim, None, incoming_world_xform)
        # Apply the full affine (incl. non-uniform scale, shear, reflection) to the local anchor.
        world_point = wp.transform_point(target_mat, local_point)

        body_path = get_rigid_body_ancestor_path(target_prim)
        if body_path is None:
            return -1, world_point

        body_idx = path_body_map.get(body_path, -1)
        if body_idx < 0:
            return None
        local_body_point = wp.transform_point(wp.transform_inverse(_builder_body_xform(builder, body_idx)), world_point)
        return body_idx, local_body_point

    if not (root_prim and root_prim.IsValid()):
        return
    for prim in ctx.prims.attachments:
        path = str(prim.GetPath())
        if _is_ignored_path(path, ignore_paths):
            continue
        if path in consumed_junction_attachment_paths:
            continue  # already consumed as rod-graph topology (curve-to-curve junction)

        src0 = get_first_target(prim, "physics:src0")
        src1 = get_first_target(prim, "physics:src1")
        type0 = str(deformable_read(prim, "type0") or "")
        type1 = str(deformable_read(prim, "type1") or "")
        indices0 = [int(i) for i in (deformable_read(prim, "indices0") or [])]
        indices1 = [int(i) for i in (deformable_read(prim, "indices1") or [])]
        coords0 = _attachment_vec3_list(deformable_read(prim, "coords0"))
        coords1 = _attachment_vec3_list(deformable_read(prim, "coords1"))
        enabled_val = deformable_read(prim, "attachmentEnabled")
        enabled = True if enabled_val is None else bool(enabled_val)
        stiffness_val = deformable_read(prim, "stiffness")
        damping_val = deformable_read(prim, "damping")
        stiffness = math.inf if stiffness_val is None else float(stiffness_val)
        damping = 0.0 if damping_val is None else float(damping_val)

        attrs: dict[str, Any] = {
            "src0": src0,
            "src1": src1,
            "type0": type0,
            "type1": type1,
            "indices0": list(indices0),
            "indices1": list(indices1),
            "coords0": _attachment_vec3_tuples(coords0),
            "coords1": _attachment_vec3_tuples(coords1),
            "enabled": enabled,
            "stiffness": stiffness,
            "damping": damping,
        }
        path_attachment_attrs[path] = attrs

        if not enabled:
            continue
        # The proposal gives attachment stiffness the range [0, inf] with +inf meaning hard
        # (the -inf sentinel belongs to deformable materials) and damping the range [0, inf).
        # Nonconforming values must not silently select the hard path.
        if math.isnan(stiffness) or stiffness < 0.0 or not (math.isfinite(damping) and damping >= 0.0):
            _mark_attachment_unsupported(
                attrs,
                path,
                f"invalid PhysicsAttachment stiffness/damping (stiffness={stiffness}, damping={damping}); "
                f"expected stiffness in [0, inf] and finite damping >= 0; preserved as metadata.",
            )
            continue
        if src0 not in path_cable_segments:
            if src0 in path_cloth_map or src0 in path_soft_map:
                _mark_attachment_unsupported(
                    attrs,
                    path,
                    "PhysicsAttachment on cloth/volume deformables is parsed but not imported yet; "
                    "Newton needs a deformable-site attachment constraint for this source type.",
                )
            else:
                _mark_attachment_unsupported(
                    attrs,
                    path,
                    f"physics:src0 target '{src0}' is not an imported cable deformable; skipping.",
                )
            continue
        if type0 not in ("point", "segment"):
            _mark_attachment_unsupported(
                attrs,
                path,
                f"PhysicsAttachment type0='{type0}' is not supported for imported cables; "
                "supported cable site types are 'point' and 'segment'.",
            )
            continue
        if type1 != "xform":
            _mark_attachment_unsupported(
                attrs,
                path,
                f"PhysicsAttachment type1='{type1}' is parsed but not imported yet; only xform targets "
                "are currently supported for cable attachments.",
            )
            continue
        if indices1:
            _mark_attachment_unsupported(
                attrs,
                path,
                "PhysicsAttachment type1='xform' must not author indices1; skipping.",
            )
            continue
        if not indices0:
            _mark_attachment_unsupported(attrs, path, "PhysicsAttachment has no indices0 attachment sites; skipping.")
            continue
        if type0 == "segment" and len(coords0) != len(indices0):
            _mark_attachment_unsupported(
                attrs,
                path,
                f"PhysicsAttachment coords0 length {len(coords0)} does not match indices0 length "
                f"{len(indices0)} for segment sites; skipping.",
            )
            continue
        if coords1 and len(coords1) != len(indices0):
            _mark_attachment_unsupported(
                attrs,
                path,
                f"PhysicsAttachment coords1 length {len(coords1)} does not match indices0 length "
                f"{len(indices0)} for xform sites; skipping.",
            )
            continue
        if not coords1:
            coords1 = [wp.vec3(0.0, 0.0, 0.0) for _ in indices0]

        if math.isfinite(stiffness):
            # Silently hardening a compliant attachment would change the authored physics;
            # preserve it as metadata until a compliant lowering exists. Damping does not
            # affect hardness: the proposal applies it only when the constraint is not
            # hard, so a damped +inf-stiffness attachment still lowers to a hard joint.
            _mark_attachment_unsupported(
                attrs,
                path,
                "finite PhysicsAttachment stiffness cannot be represented by Newton's "
                "cable-to-xform lowering yet; preserved as metadata, no joint created.",
            )
            continue

        joints: list[int] = []
        for site_idx, src_index in enumerate(indices0):
            coord0 = coords0[site_idx] if type0 == "segment" else None
            cable_anchors = _cable_attachment_anchors(
                path, src0, type0, src_index, coord0, path_cable_segments, path_cable_point_anchors
            )
            if cable_anchors is None:
                _mark_attachment_unsupported(
                    attrs,
                    path,
                    f"PhysicsAttachment type0='{type0}' could not be resolved on cable '{src0}'.",
                )
                break
            target_info = _attachment_world_point_from_xform(src1, coords1[site_idx])
            if target_info is None:
                warnings.warn(
                    f"{path}: physics:src1 target '{src1}' could not be resolved as an xform; "
                    "skipping that attachment site.",
                    stacklevel=2,
                )
                continue
            parent_body, parent_local = target_info
            for anchor_idx, (child_body, child_local) in enumerate(cable_anchors):
                label = f"{path}_site{site_idx}"
                if len(cable_anchors) > 1:
                    label = f"{label}_anchor{anchor_idx}"
                joint_idx = builder.add_joint_ball(
                    parent=parent_body,
                    child=child_body,
                    parent_xform=wp.transform(parent_local, wp.quat_identity()),
                    child_xform=wp.transform(child_local, wp.quat_identity()),
                    label=label,
                    enabled=True,
                )
                joints.append(joint_idx)

        if joints:
            path_attachment_map[path] = joints
            attrs["joint_indices"] = list(joints)
            if verbose:
                print(f"Added PhysicsAttachment {path} with {len(joints)} joint(s).")


def _deformable_remap_collapsed(
    path_cable_map: dict,
    path_attachment_map: dict,
    path_attachment_attrs: dict,
    joint_remap: Mapping[int, int],
    body_remap: Mapping[int, int],
    body_merged_parent: Mapping[int, int],
) -> tuple[dict, dict]:
    """Remap the cable / attachment index maps after ``collapse_fixed_joints``.

    Cable bodies/joints and attachment joints are addressed by index (not prim path), so they must
    ride the collapse remaps to stay valid. Returns the rebuilt ``path_cable_map`` and
    ``path_attachment_map``; ``path_attachment_attrs`` joint indices are refreshed in place.
    """

    def remap_body(body_id: int) -> int:
        # Mirror the path_body_map handling: a reindexed body is in body_remap; a body merged
        # away is resolved via its merge parent.
        if body_id in body_remap:
            return body_remap[body_id]
        if body_id in body_merged_parent:
            parent = body_merged_parent[body_id]
            return body_remap.get(parent, parent)
        return body_id

    def remap_joints(path: str, joints) -> list[int]:
        # A joint missing from the remap was deleted by the collapse; passing its stale
        # index through would silently alias a different retained joint.
        remapped = []
        for j in joints:
            if j in joint_remap:
                remapped.append(joint_remap[j])
            else:
                warnings.warn(
                    f"{path}: joint {j} was removed by collapse_fixed_joints; dropping it from "
                    f"the returned index maps.",
                    stacklevel=3,
                )
        return remapped

    if path_cable_map:
        path_cable_map = {
            path: ([remap_body(b) for b in bodies], remap_joints(path, joints))
            for path, (bodies, joints) in path_cable_map.items()
        }

    if path_attachment_map:
        path_attachment_map = {path: remap_joints(path, joints) for path, joints in path_attachment_map.items()}
        for path, joints in path_attachment_map.items():
            if path in path_attachment_attrs:
                path_attachment_attrs[path]["joint_indices"] = list(joints)

    return path_cable_map, path_attachment_map


def _element_collision_filter_groups(
    counts: Sequence[int], indices: Sequence[int], which: str, filter_path: str
) -> tuple[list[list[int]], bool] | None:
    """Partition a source's flat ``groupElemIndices`` into per-group index lists by ``groupElemCounts``.

    Each count slices the next run of indices into one group; a count of ``0`` selects *all* elements
    of the source for that paired group (represented as an empty list, resolved downstream). With no
    counts authored, all elements of the source are selected and pair against every group of the
    other side; group boundaries exist only through the counts array, so stray indices define no
    subset and are ignored with a warning.
    Returns ``(groups, broadcast)``: ``broadcast`` is True only for the no-counts form, because the
    proposal reserves pairing a side against every group of the other side for that form -- an
    explicit single group must pair one-to-one like any other explicit list. Returns ``None`` (after
    warning) for malformed counts: negative, or a total that does not match the index-array length.
    """
    if not counts:
        if indices:
            warnings.warn(
                f"{filter_path}: PhysicsElementCollisionFilter authors groupElemIndices{which} without "
                f"groupElemCounts{which}; empty counts select all elements, so the indices are ignored.",
                stacklevel=2,
            )
        return [[]], True  # all elements, paired against every group of the other side
    groups: list[list[int]] = []
    offset = 0
    for count in counts:
        if count < 0:
            warnings.warn(
                f"{filter_path}: PhysicsElementCollisionFilter groupElemCounts{which} has a negative "
                f"count {count}; skipping.",
                stacklevel=2,
            )
            return None
        if count == 0:
            groups.append([])  # count 0 -> all elements of this source for the paired group
            continue
        if offset + count > len(indices):
            warnings.warn(
                f"{filter_path}: PhysicsElementCollisionFilter groupElemCounts{which} sum exceeds the "
                f"groupElemIndices{which} length ({len(indices)}); skipping.",
                stacklevel=2,
            )
            return None
        groups.append([int(i) for i in indices[offset : offset + count]])
        offset += count
    if offset != len(indices):
        warnings.warn(
            f"{filter_path}: PhysicsElementCollisionFilter groupElemIndices{which} has "
            f"{len(indices) - offset} trailing index(es) not covered by groupElemCounts{which}; skipping.",
            stacklevel=2,
        )
        return None
    return groups, False


def _deformable_import_element_collision_filters(ctx: _DeformableImportContext) -> None:
    """Lower AOUSD ``PhysicsElementCollisionFilter`` prims to shape collision filter pairs.

    Each prim suppresses collision between paired *element groups* of ``src0`` and ``src1``.
    ``groupElemCounts0`` / ``groupElemCounts1`` slice the flat ``groupElemIndices0`` /
    ``groupElemIndices1`` arrays into groups that pair up element-wise; collisions are filtered only
    within each paired group (not across the full Cartesian product). A count of ``0`` -- or an empty
    counts array -- means *all* elements of that source. Only a side that authors no
    ``groupElemCounts`` pairs against every group of the other side.

    Supported element sources are imported cables (indices select the cable's segments), rigid bodies
    (all of the body's collider shapes), and collider prims (the exact shape, e.g. a child collider
    under a rigid Xform or a bodyless static collider). Element indices are not meaningful for a rigid
    collider, so its whole shape set is filtered. Cloth/volume (triangle/tet) element sources have no
    per-element rigid shape in Newton's shape-filter model and are warned and skipped.
    """
    builder = ctx.builder
    root_prim = ctx.root_prim
    ignore_paths = ctx.ignore_paths
    deformable_read = ctx.deformable_read
    get_first_target = ctx.get_first_target
    verbose = ctx.verbose
    path_cable_segments = ctx.path_cable_segments
    path_body_map = ctx.path_body_map
    path_shape_map = ctx.path_shape_map
    path_cloth_map = ctx.path_cloth_map
    path_soft_map = ctx.path_soft_map

    if not (root_prim and root_prim.IsValid()):
        return

    def _src_shapes(src_path: str, indices: list[int], filter_path: str) -> list[int] | None:
        # Resolve a source prim + element indices to the builder shape ids to filter. Returns None for
        # an unsupported source (already warned), or a (possibly empty) shape list otherwise.
        if src_path in path_cable_segments:
            segs = path_cable_segments[src_path]  # flat segment index -> (body, length)
            if indices:
                bodies = []
                for idx in indices:
                    if idx not in segs:
                        warnings.warn(
                            f"{filter_path}: element index {idx} is not an imported segment of cable "
                            f"'{src_path}'; skipping that element.",
                            stacklevel=2,
                        )
                        continue
                    bodies.append(segs[idx][0])
            else:
                bodies = [body for body, _length in segs.values()]  # empty indices -> all segments
            shapes: list[int] = []
            for b in bodies:
                shapes.extend(builder.body_shapes.get(b, []))
            return shapes
        if src_path in path_body_map:
            # A rigid body: filter against all of its collider shapes (per-element indices not meaningful).
            return list(builder.body_shapes.get(path_body_map[src_path], []))
        if src_path in path_shape_map:
            # An exact collider prim: a child collider under a rigid Xform, or a bodyless static
            # collider. Filter just that shape (a single rigid collider has no per-element shapes).
            return [path_shape_map[src_path]]
        if src_path in path_cloth_map or src_path in path_soft_map:
            warnings.warn(
                f"{filter_path}: PhysicsElementCollisionFilter on cloth/volume source '{src_path}' is not "
                "supported (no per-element rigid shapes); skipping.",
                stacklevel=2,
            )
            return None
        warnings.warn(
            f"{filter_path}: PhysicsElementCollisionFilter source '{src_path}' is not an imported "
            "deformable or collider; skipping.",
            stacklevel=2,
        )
        return None

    for prim in ctx.prims.element_filters:
        path = str(prim.GetPath())
        if _is_ignored_path(path, ignore_paths):
            continue
        enabled = deformable_read(prim, "filterEnabled")
        if enabled is not None and not bool(enabled):
            continue
        src0 = get_first_target(prim, "physics:src0")
        src1 = get_first_target(prim, "physics:src1")
        idx0 = [int(i) for i in (deformable_read(prim, "groupElemIndices0") or [])]
        idx1 = [int(i) for i in (deformable_read(prim, "groupElemIndices1") or [])]
        counts0 = [int(c) for c in (deformable_read(prim, "groupElemCounts0") or [])]
        counts1 = [int(c) for c in (deformable_read(prim, "groupElemCounts1") or [])]
        parsed0 = _element_collision_filter_groups(counts0, idx0, "0", path)
        parsed1 = _element_collision_filter_groups(counts1, idx1, "1", path)
        if parsed0 is None or parsed1 is None:
            continue
        groups0, broadcast0 = parsed0
        groups1, broadcast1 = parsed1
        # Pair explicit groups element-wise. Only the no-counts form pairs its single group
        # against every group of the other side; an explicit single group (counts=[n]) must
        # pair one-to-one like any other explicit list, so a group-count mismatch is malformed.
        if broadcast0 and not broadcast1:
            pairs = [(groups0[0], g1) for g1 in groups1]
        elif broadcast1 and not broadcast0:
            pairs = [(g0, groups1[0]) for g0 in groups0]
        elif len(groups0) == len(groups1):
            pairs = list(zip(groups0, groups1, strict=True))
        else:
            warnings.warn(
                f"{path}: PhysicsElementCollisionFilter has {len(groups0)} src0 group(s) but "
                f"{len(groups1)} src1 group(s); groups must pair one-to-one (or a side must author "
                "no groupElemCounts to pair against all groups); skipping.",
                stacklevel=2,
            )
            continue
        # Overlapping groups and a self-filter's mirrored (sa, sb)/(sb, sa) orderings repeat
        # shape combinations; dedup locally so each normalized pair is added once (the
        # builder's filter list itself does not deduplicate).
        seen_pairs: set[tuple[int, int]] = set()
        skip = False
        for g0, g1 in pairs:
            shapes0 = _src_shapes(src0, g0, path)
            shapes1 = _src_shapes(src1, g1, path)
            if shapes0 is None or shapes1 is None:
                skip = True  # unsupported source (already warned); the same kind repeats per group
                break
            for sa in shapes0:
                for sb in shapes1:
                    if sa != sb and (min(sa, sb), max(sa, sb)) not in seen_pairs:
                        seen_pairs.add((min(sa, sb), max(sa, sb)))
                        builder.add_shape_collision_filter_pair(sa, sb)
        if skip:
            continue
        if verbose:
            print(f"Applied PhysicsElementCollisionFilter {path}: {len(seen_pairs)} shape pair(s).")
