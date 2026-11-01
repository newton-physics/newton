# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ...geometry import GeoType, ShapeFlags
from ...sim import Model
from .collision_masks import (
    NEWTON_COLLISION_MASK_MAX_SHAPE_COUNT,
    CollisionGraphCompileResult,
    compile_newton_collision_graph,
)


@dataclass(frozen=True)
class ShapeLayout:
    """Corresponding Newton shapes in the shared MuJoCo topology."""

    representative_shapes: np.ndarray
    """One existing Newton shape per slot, used to compile the template."""

    world_shapes: np.ndarray
    """Newton shape indices [world, slot], with -1 for absent meshes."""

    body_indices: np.ndarray
    """Template Newton body index per slot, with -1 for static shapes."""

    colliders: np.ndarray
    """Whether the shape in any world collides, per slot."""

    @property
    def has_missing_shapes(self) -> bool:
        return bool(np.any(self.world_shapes < 0))

    def select(self, slots: np.ndarray) -> ShapeLayout:
        return ShapeLayout(
            self.representative_shapes[slots],
            self.world_shapes[:, slots],
            self.body_indices[slots],
            self.colliders[slots],
        )


def build_shape_layout(
    model: Model,
    *,
    skip_visual_only_geoms: bool,
    include_sites: bool,
    required_shapes: set[int],
    match_collision_flags: bool,
) -> ShapeLayout:
    """Match shapes by body and shared geom properties, padding only mesh slots.

    Collision flags only need to match when native MuJoCo contacts use them, since
    all worlds share each slot's contype and conaffinity.
    """
    worlds = model.shape_world.numpy()
    bodies = model.shape_body.numpy()
    types = model.shape_type.numpy()
    # A variant row keeps its role (collider, visual, or site) for every variant it can hold, but
    # may be empty when the solver is built.
    flags = model._shape_option_flags() if model._variant_row_max else model.shape_flags.numpy()
    body_count = model.body_count // model.world_count
    local_bodies = bodies.copy()
    local_bodies[bodies >= 0] %= body_count
    attrs = model.mujoco
    condim = attrs.condim.numpy() if hasattr(attrs, "condim") else np.full(model.shape_count, 3)
    priority = (
        attrs.geom_priority.numpy() if hasattr(attrs, "geom_priority") else np.zeros(model.shape_count, dtype=int)
    )
    mesh_types = (GeoType.MESH, GeoType.CONVEX_MESH)
    sites = (flags & int(ShapeFlags.SITE)) != 0
    colliders = (flags & int(ShapeFlags.COLLIDE_SHAPES)) != 0
    slot_types = np.where(np.isin(types, mesh_types) & ~sites, int(GeoType.MESH), types)
    columns = [local_bodies, slot_types, sites, condim, priority]
    if match_collision_flags:
        columns.append(colliders)
    properties = np.column_stack(columns)
    starts = model.shape_world_start.numpy()
    counts = np.diff(starts[:-1])
    matching_layout = np.all(counts == counts[0])
    if matching_layout:
        world_properties = properties[starts[0] : starts[-2]].reshape(model.world_count, counts[0], properties.shape[1])
        matching_layout = np.all(world_properties == world_properties[:1])
    if matching_layout:
        representatives = np.flatnonzero(worlds <= 0).astype(np.int32)
        mapping = np.tile(representatives, (model.world_count, 1))
        mapping[:, worlds[representatives] == 0] += (starts[:-2] - starts[0])[:, None]
    else:
        # Convert in bulk to avoid NumPy scalar operations for every shape.
        properties = properties.tolist()
        ordinals = [{} for _ in range(model.world_count)]
        slots = {}
        for shape, world in enumerate(worlds.tolist()):
            if world < 0:
                slot_key = ("global", shape)
            else:
                key = tuple(properties[shape])
                ordinal = ordinals[world].get(key, 0)
                ordinals[world][key] = ordinal + 1
                slot_key = (*key, ordinal)
            slots.setdefault(slot_key, []).append((world, shape))
        representatives = np.asarray([shapes[0][1] for shapes in slots.values()], dtype=np.int32)
        mapping = np.full((model.world_count, len(slots)), -1, dtype=np.int32)
        for column, shapes in enumerate(slots.values()):
            for world, shape in shapes:
                mapping[slice(None) if world < 0 else world, column] = shape

    slot_colliders = np.any(colliders[mapping] & (mapping >= 0), axis=0)
    keep = np.where(sites[representatives], include_sites, (not skip_visual_only_geoms) | slot_colliders)
    if required_shapes:
        keep |= np.isin(representatives, list(required_shapes))
    representatives, mapping, slot_colliders = representatives[keep], mapping[:, keep], slot_colliders[keep]
    for column in np.flatnonzero(np.any(mapping < 0, axis=0)):
        shape = representatives[column]
        if types[shape] not in mesh_types or sites[shape]:
            missing_world = int(np.flatnonzero(mapping[:, column] < 0)[0])
            if model._variant_row_max and model._shape_variant_row.numpy()[shape]:
                raise ValueError(
                    f"SolverMuJoCo found different body variants in world 0 and world {missing_world} for "
                    f"{model.shape_label[shape]!r}; construct it before selecting different variants per world."
                )
            raise ValueError(
                f"SolverMuJoCo shape types mismatch at position {column}: "
                f"world {missing_world} has no matching shape for {model.shape_label[shape]!r}. "
                "Only mesh geoms support differing counts; body, type, condim and priority must match for other shapes"
                + (", as must collision flags with native MuJoCo contacts." if match_collision_flags else ".")
            )
    return ShapeLayout(representatives, mapping, local_bodies[representatives], slot_colliders)


def compile_layout_collision_masks(model: Model, layout: ShapeLayout) -> CollisionGraphCompileResult:
    """Map shared collision groups and cross-body exclusions into geom slots."""
    count = len(layout.representative_shapes)
    groups = model.shape_collision_group.numpy()
    slot_groups = groups[layout.representative_shapes]
    present = layout.world_shapes >= 0
    if np.any(groups[layout.world_shapes[present]] != np.broadcast_to(slot_groups, present.shape)[present]):
        raise ValueError("Native MuJoCo collision groups must match across corresponding geom slots.")

    shape_to_slot = np.full(model.shape_count, -1, dtype=np.int32)
    shape_to_slot[layout.world_shapes[present]] = np.broadcast_to(np.arange(count), present.shape)[present]
    pairs = model.shape_collision_filter_pairs_array()
    worlds = model.shape_world.numpy()[pairs]
    pairs = pairs[(worlds[:, 0] == worlds[:, 1]) | np.any(worlds < 0, axis=1)]
    excluded = shape_to_slot[pairs]
    excluded = excluded[np.all(excluded >= 0, axis=1)]
    # MuJoCo already suppresses same-body contacts, regardless of hull count.
    excluded = excluded[layout.body_indices[excluded[:, 0]] != layout.body_indices[excluded[:, 1]]]
    excluded = np.unique(np.sort(excluded, axis=1), axis=0)
    if len(excluded):
        for world, shapes in enumerate(layout.world_shapes):
            pairs = shapes[excluded]
            pairs = pairs[np.all(pairs >= 0, axis=1)]
            if not np.all(model.shape_collision_filter_mask(pairs)):
                raise ValueError(
                    f"SolverMuJoCo collision filters differ in world {world}; "
                    "native MuJoCo requires shared exclusions for corresponding geom slots."
                )
    result = compile_newton_collision_graph(
        slot_groups,
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
