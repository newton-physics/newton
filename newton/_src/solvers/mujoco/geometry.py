# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import numpy as np

from ...geometry import GeoType, ShapeFlags
from ...sim import Model
from .collision_masks import (
    NEWTON_COLLISION_MASK_MAX_SHAPE_COUNT,
    CollisionGraphCompileResult,
    compile_newton_collision_graph,
)


def supports_missing_meshes() -> bool:
    """Detect the broadphase fix until Newton requires MJWarp #1689."""
    from mujoco_warp._src import collision_driver

    return hasattr(collision_driver, "_mesh_missing")


def build_shape_layout(
    model: Model,
    *,
    skip_visual_only_geoms: bool,
    include_sites: bool,
    required_shapes: set[int],
) -> tuple[np.ndarray, np.ndarray]:
    """Match shapes by body and shared geom properties, padding only mesh slots."""
    worlds = model.shape_world.numpy()
    bodies = model.shape_body.numpy()
    types = model.shape_type.numpy()
    flags = model.shape_flags.numpy()
    body_count = model.body_count // model.world_count
    attrs = model.mujoco
    condim = attrs.condim.numpy() if hasattr(attrs, "condim") else np.full(model.shape_count, 3)
    priority = (
        attrs.geom_priority.numpy() if hasattr(attrs, "geom_priority") else np.zeros(model.shape_count, dtype=int)
    )
    mesh_types = (GeoType.MESH, GeoType.CONVEX_MESH)
    counts = [{} for _ in range(model.world_count)]
    candidates = []
    required_keys = set()
    for shape in range(model.shape_count):
        site = bool(flags[shape] & ShapeFlags.SITE)
        world = int(worlds[shape])
        if world < 0:
            slot_key = ("global", shape)
        else:
            body = int(bodies[shape]) % body_count if bodies[shape] >= 0 else -1
            shape_type = GeoType.MESH if types[shape] in mesh_types and not site else int(types[shape])
            key = (
                body,
                shape_type,
                site,
                int(condim[shape]),
                int(priority[shape]),
                bool(flags[shape] & ShapeFlags.COLLIDE_SHAPES),
            )
            ordinal = counts[world].get(key, 0)
            counts[world][key] = ordinal + 1
            slot_key = (*key, ordinal)
        candidates.append((shape, world, slot_key))
        if shape in required_shapes:
            required_keys.add(slot_key)

    columns = {}
    representatives = []
    entries = []
    for shape, world, slot_key in candidates:
        site = bool(flags[shape] & ShapeFlags.SITE)
        if slot_key not in required_keys:
            if site and not include_sites:
                continue
            if not site and skip_visual_only_geoms and not (flags[shape] & ShapeFlags.COLLIDE_SHAPES):
                continue
        if slot_key not in columns:
            columns[slot_key] = len(columns)
            representatives.append(shape)
        entries.append((slice(None) if world < 0 else world, columns[slot_key], shape))
    mapping = np.full((model.world_count, len(columns)), -1, dtype=np.int32)
    for world, column, shape in entries:
        mapping[world, column] = shape
    representatives = np.asarray(representatives, dtype=np.int32)
    for column in np.flatnonzero(np.any(mapping < 0, axis=0)):
        shape = representatives[column]
        if types[shape] not in mesh_types or flags[shape] & ShapeFlags.SITE:
            missing_world = int(np.flatnonzero(mapping[:, column] < 0)[0])
            raise ValueError(
                f"SolverMuJoCo shape types mismatch at position {column}: "
                f"world {missing_world} has no matching shape for {model.shape_label[shape]!r}. "
                "Only mesh geoms support differing counts; body, type, condim and priority must match for other shapes."
            )
    return representatives, mapping


def compile_layout_collision_masks(model: Model, mapping: np.ndarray) -> CollisionGraphCompileResult:
    """Require one collision graph to describe every simultaneously present pair."""
    count = mapping.shape[1]
    a, b = np.triu_indices(count, 1)
    groups = model.shape_collision_group.numpy()
    flags = model.shape_flags.numpy()
    bodies = model.shape_body.numpy()
    representatives = mapping[np.argmax(mapping >= 0, axis=0), np.arange(count)]
    local_bodies = bodies[representatives]
    world_offsets = model.shape_world.numpy()[representatives] * (model.body_count // model.world_count)
    local_bodies = np.where(local_bodies >= 0, local_bodies - world_offsets, -1)
    # Same-body contacts are already excluded by MuJoCo, independent of masks.
    cross_body = local_bodies[a] != local_bodies[b]
    a, b = a[cross_body], b[cross_body]
    allowed = np.full(len(a), -1, dtype=np.int8)
    for world, shapes in enumerate(mapping):
        present = (shapes[a] >= 0) & (shapes[b] >= 0)
        pairs = np.column_stack((shapes[a[present]], shapes[b[present]]))
        ga, gb = groups[pairs[:, 0]], groups[pairs[:, 1]]
        values = (ga != 0) & (gb != 0)
        values &= ((ga > 0) & ((ga == gb) | (gb < 0))) | ((ga < 0) & (ga != gb))
        values &= (flags[pairs[:, 0]] & int(ShapeFlags.COLLIDE_SHAPES)) != 0
        values &= (flags[pairs[:, 1]] & int(ShapeFlags.COLLIDE_SHAPES)) != 0
        values &= ~model.shape_collision_filter_mask(pairs)
        previous = allowed[present]
        conflicts = (previous >= 0) & (previous != values)
        if np.any(conflicts):
            pair = pairs[np.flatnonzero(conflicts)[0]]
            raise ValueError(
                f"SolverMuJoCo collision filters differ in world {world} for shapes {pair.tolist()}; "
                "MJWarp requires shared filtering for corresponding geom slots."
            )
        allowed[present] = values
    excluded = np.column_stack((a[allowed == 0], b[allowed == 0]))
    result = compile_newton_collision_graph(
        np.ones(count, dtype=np.int32),
        excluded_pairs=excluded,
        max_shape_count=NEWTON_COLLISION_MASK_MAX_SHAPE_COUNT if len(excluded) else None,
        max_excluded_pair_count=None,
    )
    if not result.exact:
        raise ValueError(
            "The heterogeneous geom collision graph exceeds the MuJoCo mask compiler's capacity. "
            "Simplify the collision filters or use use_mujoco_contacts=False."
        )
    return result
