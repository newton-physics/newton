# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Collect and apply USD collision filters at the importer's existing phases."""

from __future__ import annotations

import collections
import itertools
import warnings
from typing import TYPE_CHECKING

from ..geometry import ShapeFlags

if TYPE_CHECKING:
    from pxr import Usd

    from ..sim.builder import ModelBuilder


def _collect_filtered_pairs(prim: Usd.Prim, pairs: set[tuple[str, str]]) -> None:
    """Collect symmetric authored path pairs without resolving shape indices yet."""
    if not prim.HasRelationship("physics:filteredPairs"):
        return
    src = str(prim.GetPath())
    for target in prim.GetRelationship("physics:filteredPairs").GetTargets():
        dst = str(target)
        # The relationship may be authored on either or both endpoints and Newton's
        # filter pair is symmetric; canonicalizing dedups both. A self-pair is invalid.
        if src != dst:
            pairs.add((src, dst) if src < dst else (dst, src))


def _apply_collision_groups(
    builder: ModelBuilder,
    stage: Usd.Stage,
    imported_rigid_collider_groups: dict[str, tuple[str, ...]],
    path_shape_map: dict[str, int],
) -> None:
    """Apply USD collision groups once the imported rigid collider shapes exist."""
    from pxr import Sdf, UsdPhysics

    # Lower OpenUSD collision groups to explicit Newton filter pairs. Group colliders by their
    # complete membership signature so table queries scale with the number of distinct group
    # combinations, while materializing only the pairs that OpenUSD actually disables.
    if imported_rigid_collider_groups:
        collision_group_table = UsdPhysics.CollisionGroup.ComputeCollisionGroupTable(stage)
        colliders_by_groups: dict[tuple[str, ...], list[tuple[str, int]]] = collections.defaultdict(list)
        for collider_path, collision_groups in imported_rigid_collider_groups.items():
            shape_id = path_shape_map[collider_path]
            if builder.shape_flags[shape_id] & ShapeFlags.COLLIDE_SHAPES:
                colliders_by_groups[collision_groups].append((collider_path, shape_id))

        inverted_groups: set[str] = set()
        groups_by_merge_name: dict[str, set[str]] = collections.defaultdict(set)
        group_merge_names: dict[str, str] = {}
        for prim in stage.Traverse():
            if not prim.IsA(UsdPhysics.CollisionGroup):
                continue
            group = UsdPhysics.CollisionGroup(prim)
            group_path = str(prim.GetPath())
            if group.GetInvertFilteredGroupsAttr().Get():
                inverted_groups.add(group_path)
            merge_name = group.GetMergeGroupNameAttr().Get() or ""
            group_merge_names[group_path] = merge_name
            if merge_name:
                groups_by_merge_name[merge_name].add(group_path)

        def _groups_collide(groups_a: tuple[str, ...], groups_b: tuple[str, ...]) -> bool:
            if groups_a and groups_b:
                return all(
                    collision_group_table.IsCollisionEnabled(Sdf.Path(group_a), Sdf.Path(group_b))
                    for group_a in groups_a
                    for group_b in groups_b
                )
            groups = groups_a or groups_b
            for group_path in groups:
                merge_name = group_merge_names.get(group_path, "")
                effective_groups = groups_by_merge_name[merge_name] if merge_name else (group_path,)
                if any(effective_group in inverted_groups for effective_group in effective_groups):
                    return False
            return True

        existing_filter_pairs = set(builder._materialized_filter_template())
        group_classes = sorted(colliders_by_groups.items())
        for class_index_a, (groups_a, colliders_a) in enumerate(group_classes):
            for class_index_b in range(class_index_a, len(group_classes)):
                groups_b, colliders_b = group_classes[class_index_b]
                if class_index_a == class_index_b:
                    if len(colliders_a) < 2:
                        continue
                    collider_pairs = itertools.combinations(colliders_a, 2)
                else:
                    collider_pairs = itertools.product(colliders_a, colliders_b)

                if _groups_collide(groups_a, groups_b):
                    continue
                for (_, shape_a), (_, shape_b) in collider_pairs:
                    if shape_a == shape_b:
                        continue
                    pair = (shape_a, shape_b) if shape_a < shape_b else (shape_b, shape_a)
                    if pair not in existing_filter_pairs:
                        existing_filter_pairs.add(pair)
                        builder.add_shape_collision_filter_pair(*pair)


def _apply_filtered_pairs(
    builder: ModelBuilder,
    stage: Usd.Stage,
    authored_filtered_path_pairs: set[tuple[str, str]],
    *,
    path_shape_map: dict[str, int],
    path_body_map: dict[str, int],
    path_cable_map: dict[str, tuple[list[int], list[int]]],
    path_cloth_map: dict[str, dict[str, tuple[int, int]]],
    path_soft_map: dict[str, dict[str, tuple[int, int]]],
    body_owner: dict[str, str],
) -> None:
    """Apply authored pairs after rigid shapes, cables, and element filters exist."""

    def _resolve_collision_shape_ids(path: str) -> tuple[list[int], str | None]:
        """Resolve a filtered-pair endpoint to Newton shape indices, or an unsupported reason.

        Endpoint ownership comes only from the import maps (never path-prefix matching): a
        native collider is one shape, a rigid body or cable is all of its shapes, and a
        deformable body prim resolves through its simulation geometry. Cloth and volume
        deformables are particles, which Newton's shape filter pairs cannot express.
        """
        if path in path_shape_map:
            return [path_shape_map[path]], None
        if path in path_body_map:
            return sorted(set(builder.body_shapes.get(path_body_map[path], []))), None
        if path in path_cable_map:
            shape_ids: set[int] = set()
            for cable_body in path_cable_map[path][0]:
                shape_ids.update(builder.body_shapes.get(cable_body, []))
            return sorted(shape_ids), None
        owner_path = body_owner.get(path)
        if owner_path is not None and owner_path != path:
            return _resolve_collision_shape_ids(owner_path)
        if path in path_cloth_map:
            return [], "it is a cloth particle deformable, and standard particle collision filters are not supported"
        if path in path_soft_map:
            return [], "it is a volume particle deformable, and standard particle collision filters are not supported"
        target_prim = stage.GetPrimAtPath(path)
        if not target_prim or not target_prim.IsValid():
            return [], "the target path does not exist"
        return [], "it produced no collision participant (it may be disabled, ignored, malformed, or non-colliding)"

    # Apply the authored filtered pairs: every native shape and cable capsule exists now, and
    # the deformable maps allow precise unsupported diagnostics. Shape indices are stable from
    # here on (collapse_fixed_joints only remaps bodies). Seed the dedup set from the builder
    # so pairs the element-filter pass already added are not appended again.
    if authored_filtered_path_pairs:
        existing_filter_pairs = set(builder._materialized_filter_template())
        for filter_path1, filter_path2 in sorted(authored_filtered_path_pairs):
            shapes1, reason1 = _resolve_collision_shape_ids(filter_path1)
            shapes2, reason2 = _resolve_collision_shape_ids(filter_path2)
            if not shapes1 or not shapes2:
                bad_path, reason = (filter_path1, reason1) if not shapes1 else (filter_path2, reason2)
                warnings.warn(
                    f"{filter_path1} <-> {filter_path2}: physics:filteredPairs was not imported "
                    f"because {bad_path}: {reason}.",
                    stacklevel=3,
                )
                continue
            for shape1 in shapes1:
                if not builder.shape_flags[shape1] & ShapeFlags.COLLIDE_SHAPES:
                    continue
                for shape2 in shapes2:
                    if not builder.shape_flags[shape2] & ShapeFlags.COLLIDE_SHAPES:
                        continue
                    if shape1 == shape2:
                        continue
                    pair = (shape1, shape2) if shape1 < shape2 else (shape2, shape1)
                    if pair not in existing_filter_pairs:
                        existing_filter_pairs.add(pair)
                        builder.add_shape_collision_filter_pair(*pair)
