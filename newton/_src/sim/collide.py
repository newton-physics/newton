# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import warnings
from typing import Literal

import numpy as np
import warp as wp

from ..core.reset import normalize_reset_world_mask
from ..geometry.broad_phase_nxn import BroadPhaseAllPairs, BroadPhaseExplicit
from ..geometry.broad_phase_sap import BroadPhaseSAP
from ..geometry.collision_core import compute_tight_aabb_from_support
from ..geometry.contact_data import ContactData, make_contact_sort_key
from ..geometry.contact_match import ContactMatcher
from ..geometry.contact_sort import ContactSorter
from ..geometry.differentiable_contacts import launch_differentiable_contact_augment
from ..geometry.flags import ShapeFlags
from ..geometry.kernels import create_soft_contacts
from ..geometry.narrow_phase import NarrowPhase
from ..geometry.sdf_hydroelastic import HydroelasticSDF
from ..geometry.soft_contacts_bvh import build_rigid_soft_bvh_rigid_feature_tables, launch_rigid_soft_bvh_contacts
from ..geometry.soft_contacts_sdf import launch_soft_ef_contacts
from ..geometry.support_function import (
    GenericShapeData,
    SupportMapDataProvider,
    pack_mesh_ptr,
)
from ..geometry.tri_mesh_collision import TriMeshCollisionDetector
from ..geometry.types import GeoType
from ..sim.contacts import Contacts
from ..sim.model import Model
from ..sim.state import State


def _shape_collide_mask(model: Model, shape_count: int | None = None) -> np.ndarray:
    """Return a host mask for shapes participating in shape-shape collision."""
    shape_flags = getattr(model, "shape_flags", None)
    if shape_flags is None:
        count = model.shape_count if shape_count is None else shape_count
        return np.ones(count, dtype=bool)

    flags = shape_flags.numpy()
    if shape_count is not None and len(flags) != shape_count:
        raise ValueError("model.shape_flags and model.shape_type must have the same length")
    return (flags & int(ShapeFlags.COLLIDE_SHAPES)) != 0


@wp.struct
class ContactWriterData:
    """Contact writer data for collide write_contact function."""

    contact_max: int
    # Body information arrays (for transforming to body-local coordinates)
    body_q: wp.array[wp.transform]
    shape_body: wp.array[int]
    shape_gap: wp.array[float]
    # Output arrays
    contact_count: wp.array[int]
    out_shape0: wp.array[int]
    out_shape1: wp.array[int]
    out_point0: wp.array[wp.vec3]
    out_point1: wp.array[wp.vec3]
    out_offset0: wp.array[wp.vec3]
    out_offset1: wp.array[wp.vec3]
    out_normal: wp.array[wp.vec3]
    out_margin0: wp.array[float]
    out_margin1: wp.array[float]
    out_tids: wp.array[int]
    # Per-contact shape properties, empty arrays if not enabled.
    # Zero-values indicate that no per-contact shape properties are set for this contact
    out_stiffness: wp.array[float]
    out_damping: wp.array[float]
    out_friction: wp.array[float]
    out_sort_key: wp.array[wp.int64]


@wp.func
def write_contact(
    contact_data: ContactData,
    writer_data: ContactWriterData,
    output_index: int,
):
    """
    Write a contact to the output arrays using ContactData and ContactWriterData.

    Args:
        contact_data: ContactData struct containing contact information
        writer_data: ContactWriterData struct containing body info and output arrays
        output_index: If -1, use atomic_add to get the next available index if contact distance is less than margin. If >= 0, use this index directly and skip margin check.
    """
    total_separation_needed = (
        contact_data.radius_eff_a + contact_data.radius_eff_b + contact_data.margin_a + contact_data.margin_b
    )

    offset_mag_a = contact_data.radius_eff_a + contact_data.margin_a
    offset_mag_b = contact_data.radius_eff_b + contact_data.margin_b

    # Distance calculation matching box_plane_collision
    contact_normal_a_to_b = wp.normalize(contact_data.contact_normal_a_to_b)

    a_contact_world = contact_data.contact_point_center - contact_normal_a_to_b * (
        0.5 * contact_data.contact_distance + contact_data.radius_eff_a
    )
    b_contact_world = contact_data.contact_point_center + contact_normal_a_to_b * (
        0.5 * contact_data.contact_distance + contact_data.radius_eff_b
    )

    diff = b_contact_world - a_contact_world
    distance = wp.dot(diff, contact_normal_a_to_b)
    d = distance - total_separation_needed

    # Use per-shape contact gaps (sum of both shapes)
    gap_a = writer_data.shape_gap[contact_data.shape_a]
    gap_b = writer_data.shape_gap[contact_data.shape_b]
    contact_gap = gap_a + gap_b

    index = output_index

    if index < 0:
        # compute index using atomic counter
        if d > contact_gap:
            return
        index = wp.atomic_add(writer_data.contact_count, 0, 1)
    if index >= writer_data.contact_max:
        return

    writer_data.out_shape0[index] = contact_data.shape_a
    writer_data.out_shape1[index] = contact_data.shape_b

    # Get body indices for the shapes
    body0 = writer_data.shape_body[contact_data.shape_a]
    body1 = writer_data.shape_body[contact_data.shape_b]

    # Compute body inverse transforms
    X_bw_a = wp.transform_identity() if body0 == -1 else wp.transform_inverse(writer_data.body_q[body0])
    X_bw_b = wp.transform_identity() if body1 == -1 else wp.transform_inverse(writer_data.body_q[body1])

    # Contact points are stored in body frames
    writer_data.out_point0[index] = wp.transform_point(X_bw_a, a_contact_world)
    writer_data.out_point1[index] = wp.transform_point(X_bw_b, b_contact_world)

    contact_normal = contact_normal_a_to_b

    # Offsets in body frames (offset0 points toward B, offset1 points toward A)
    writer_data.out_offset0[index] = wp.transform_vector(X_bw_a, offset_mag_a * contact_normal)
    writer_data.out_offset1[index] = wp.transform_vector(X_bw_b, -offset_mag_b * contact_normal)

    writer_data.out_normal[index] = contact_normal
    writer_data.out_margin0[index] = offset_mag_a
    writer_data.out_margin1[index] = offset_mag_b
    writer_data.out_tids[index] = 0  # tid not available in this context

    # Write stiffness/damping/friction only if per-contact shape properties are enabled
    if writer_data.out_stiffness.shape[0] > 0:
        writer_data.out_stiffness[index] = contact_data.contact_stiffness
        writer_data.out_damping[index] = contact_data.contact_damping
        writer_data.out_friction[index] = contact_data.contact_friction_scale

    if writer_data.out_sort_key.shape[0] > 0:
        writer_data.out_sort_key[index] = make_contact_sort_key(
            contact_data.shape_a, contact_data.shape_b, contact_data.sort_sub_key
        )


@wp.kernel(enable_backward=False)
def compute_shape_aabbs(
    body_q: wp.array[wp.transform],
    shape_transform: wp.array[wp.transform],
    shape_body: wp.array[int],
    shape_type: wp.array[int],
    shape_scale: wp.array[wp.vec3],
    shape_collision_radius: wp.array[float],
    shape_source_ptr: wp.array[wp.uint64],
    shape_margin: wp.array[float],
    shape_gap: wp.array[float],
    shape_collision_aabb_lower: wp.array[wp.vec3],
    shape_collision_aabb_upper: wp.array[wp.vec3],
    # Fused counter arrays — zeroed by thread 0 to avoid separate kernel launches.
    contact_counters: wp.array[wp.int32],
    contact_generation: wp.array[wp.int32],
    broad_phase_pair_count: wp.array[wp.int32],
    num_contact_counters: int,
    # outputs
    aabb_lower: wp.array[wp.vec3],
    aabb_upper: wp.array[wp.vec3],
    geom_data: wp.array[wp.vec4],
    geom_xform: wp.array[wp.transform],
):
    """Compute AABBs, narrow-phase geometry data, and zero collision counters.

    Fuses AABB computation, narrow-phase data preparation, contact counter
    zeroing, and generation bumping into a single kernel launch.
    """
    shape_id = wp.tid()

    # Thread 0: zero contact counters, bump contact generation, and zero the
    # broad phase candidate-pair count in a single fused step.
    if shape_id == 0:
        for c in range(num_contact_counters):
            contact_counters[c] = 0
        g = contact_generation[0]
        if g == 2147483647:
            g = 0
        else:
            g = g + 1
        contact_generation[0] = g
        broad_phase_pair_count[0] = 0

    rigid_id = shape_body[shape_id]
    geo_type = shape_type[shape_id]

    # Compute world transform
    if rigid_id == -1:
        X_ws = shape_transform[shape_id]
    else:
        X_ws = wp.transform_multiply(body_q[rigid_id], shape_transform[shape_id])

    pos = wp.transform_get_translation(X_ws)
    orientation = wp.transform_get_rotation(X_ws)

    margin = shape_margin[shape_id]

    # Enlarge AABB by per-shape effective gap for contact detection
    effective_gap = margin + shape_gap[shape_id]
    margin_vec = wp.vec3(effective_gap, effective_gap, effective_gap)

    # Check if this is an infinite plane or a shape with a pre-computed local AABB
    scale = shape_scale[shape_id]
    is_infinite_plane = (geo_type == GeoType.PLANE) and (scale[0] == 0.0 and scale[1] == 0.0)
    has_local_aabb = geo_type == GeoType.MESH or geo_type == GeoType.HFIELD or geo_type == GeoType.CONVEX_MESH

    geom_scale = scale

    if is_infinite_plane:
        # Bounding sphere fallback for infinite planes
        radius = shape_collision_radius[shape_id]
        half_extents = wp.vec3(radius, radius, radius)
        aabb_lower[shape_id] = pos - half_extents - margin_vec
        aabb_upper[shape_id] = pos + half_extents + margin_vec
    elif has_local_aabb:
        # Pre-computed local AABB transformed to world space.
        # Scale is already baked into shape_collision_aabb by the builder,
        # so we only need to handle the rotation here.
        local_lo = shape_collision_aabb_lower[shape_id]
        local_hi = shape_collision_aabb_upper[shape_id]

        center = (local_lo + local_hi) * 0.5
        half = (local_hi - local_lo) * 0.5

        # Rotate center to world frame
        world_center = wp.quat_rotate(orientation, center) + pos

        # Rotated AABB half-extents via abs of rotation matrix columns
        r0 = wp.quat_rotate(orientation, wp.vec3(1.0, 0.0, 0.0))
        r1 = wp.quat_rotate(orientation, wp.vec3(0.0, 1.0, 0.0))
        r2 = wp.quat_rotate(orientation, wp.vec3(0.0, 0.0, 1.0))

        world_half = wp.vec3(
            wp.abs(r0[0]) * half[0] + wp.abs(r1[0]) * half[1] + wp.abs(r2[0]) * half[2],
            wp.abs(r0[1]) * half[0] + wp.abs(r1[1]) * half[1] + wp.abs(r2[1]) * half[2],
            wp.abs(r0[2]) * half[0] + wp.abs(r1[2]) * half[1] + wp.abs(r2[2]) * half[2],
        )

        aabb_lower[shape_id] = world_center - world_half - margin_vec
        aabb_upper[shape_id] = world_center + world_half + margin_vec
    else:
        # Use support function to compute tight AABB
        # Create generic shape data
        shape_data = GenericShapeData()
        shape_data.shape_type = geo_type
        if geo_type == GeoType.PLANE:
            geom_scale = wp.vec3(scale[0] * 0.5, scale[1] * 0.5, 0.0)
        shape_data.scale = geom_scale
        shape_data.auxiliary = wp.vec3(0.0, 0.0, 0.0)

        # For CONVEX_MESH, pack the mesh pointer
        if geo_type == GeoType.CONVEX_MESH:
            shape_data.auxiliary = pack_mesh_ptr(shape_source_ptr[shape_id])

        data_provider = SupportMapDataProvider()

        # Compute tight AABB using helper function
        aabb_min_world, aabb_max_world = compute_tight_aabb_from_support(shape_data, orientation, pos, data_provider)

        aabb_lower[shape_id] = aabb_min_world - margin_vec
        aabb_upper[shape_id] = aabb_max_world + margin_vec

    # Narrow-phase geometry data (reuses X_ws and scale already computed above)
    geom_data[shape_id] = wp.vec4(geom_scale[0], geom_scale[1], geom_scale[2], margin)
    geom_xform[shape_id] = X_ws


# Primitive pairs (GJK/MPR) produce up to 5 manifold contacts.
# Mesh-involved pairs (SDF + contact reduction) typically retain about 40.
_RIGID_CONTACTS_PER_PRIMITIVE_PAIR = 5
_RIGID_CONTACTS_PER_MESH_PAIR = 40
_RIGID_CONTACT_MAX_NEIGHBORS_PER_SHAPE = 20
_RIGID_CONTACT_MIN_CAPACITY = 1000


def _estimate_rigid_contact_max(model: Model) -> int:
    """
    Estimate the maximum number of rigid contacts for the collision pipeline.

    Uses a linear neighbor-budget estimate assuming each non-plane shape contacts
    at most ``MAX_NEIGHBORS_PER_SHAPE`` others (spatial locality).  The non-plane
    term is additive across independent worlds so a single-pool computation is
    correct.  The plane term (each plane vs all non-planes in its world) would be
    quadratic if computed globally, so it is evaluated per world when metadata is
    available.

    When precomputed contact pairs are available their count is used as an
    alternative tighter bound (``min`` of heuristic and pair-based estimate).

    Args:
        model: The simulation model.

    Returns:
        Estimated maximum number of rigid contacts.
    """
    if not hasattr(model, "shape_type") or model.shape_type is None:
        return 1000  # Fallback

    shape_types = model.shape_type.numpy()
    colliding_mask = _shape_collide_mask(model, len(shape_types))

    mesh_mask = colliding_mask & ((shape_types == int(GeoType.MESH)) | (shape_types == int(GeoType.HFIELD)))
    plane_mask = colliding_mask & (shape_types == int(GeoType.PLANE))
    non_plane_mask = colliding_mask & ~plane_mask
    num_meshes = int(np.count_nonzero(mesh_mask))
    num_non_planes = int(np.count_nonzero(non_plane_mask))
    num_primitives = num_non_planes - num_meshes
    num_planes = int(np.count_nonzero(plane_mask))

    # Weighted contacts from non-plane shape types.
    # Each shape's neighbor pairs are weighted by its type's contacts-per-pair.
    # Divide by 2 to avoid double-counting pairs.
    non_plane_contacts = (
        num_primitives * _RIGID_CONTACT_MAX_NEIGHBORS_PER_SHAPE * _RIGID_CONTACTS_PER_PRIMITIVE_PAIR
        + num_meshes * _RIGID_CONTACT_MAX_NEIGHBORS_PER_SHAPE * _RIGID_CONTACTS_PER_MESH_PAIR
    ) // 2

    # Weighted average contacts-per-pair based on the scene's shape mix.
    avg_cpp = (
        (num_primitives * _RIGID_CONTACTS_PER_PRIMITIVE_PAIR + num_meshes * _RIGID_CONTACTS_PER_MESH_PAIR)
        // max(num_non_planes, 1)
        if num_non_planes > 0
        else 0
    )

    # Plane contacts: each plane contacts all non-plane shapes *in its world*.
    # The naive global formula (num_planes * num_non_planes) is O(worlds²) when
    # both counts grow with the number of worlds.  Use per-world counts instead.
    plane_contacts = 0
    if num_planes > 0 and num_non_planes > 0:
        has_world_info = (
            hasattr(model, "shape_world")
            and model.shape_world is not None
            and hasattr(model, "world_count")
            and model.world_count > 0
        )
        shape_world = model.shape_world.numpy() if has_world_info else None

        if shape_world is not None and len(shape_world) == len(shape_types):
            global_mask = shape_world == -1
            local_mask = ~global_mask
            n_worlds = model.world_count

            global_planes = int(np.count_nonzero(global_mask & plane_mask))
            global_non_planes = int(np.count_nonzero(global_mask & non_plane_mask))

            local_plane_counts = np.bincount(shape_world[local_mask & plane_mask], minlength=n_worlds)[:n_worlds]
            local_non_plane_counts = np.bincount(shape_world[local_mask & non_plane_mask], minlength=n_worlds)[
                :n_worlds
            ]

            per_world_planes = local_plane_counts + global_planes
            per_world_non_planes = local_non_plane_counts + global_non_planes

            # Global-global pairs appear in every world slice; keep one copy.
            plane_pair_count = int(np.sum(per_world_planes * per_world_non_planes))
            if n_worlds > 1:
                plane_pair_count -= (n_worlds - 1) * global_planes * global_non_planes
            plane_contacts = plane_pair_count * avg_cpp
        else:
            # Fallback: exact type-weighted sum (correct for single-world models).
            plane_contacts = num_planes * (
                num_primitives * _RIGID_CONTACTS_PER_PRIMITIVE_PAIR + num_meshes * _RIGID_CONTACTS_PER_MESH_PAIR
            )

    total_contacts = non_plane_contacts + plane_contacts

    # When precomputed contact pairs are available, use as a tighter bound.
    if hasattr(model, "shape_contact_pair_count") and model.shape_contact_pair_count > 0:
        weighted_cpp = max(avg_cpp, _RIGID_CONTACTS_PER_PRIMITIVE_PAIR)
        pair_contacts = int(model.shape_contact_pair_count) * weighted_cpp
        total_contacts = min(total_contacts, pair_contacts)

    # Ensure minimum allocation
    return max(_RIGID_CONTACT_MIN_CAPACITY, total_contacts)


def _compute_per_world_shape_pairs_max(model: Model) -> int:
    """Compute the maximum number of candidate shape pairs using per-world counts.

    For multi-world scenes the global formula ``N*(N-1)/2`` is O(W^2 * S^2)
    where W is the number of worlds and S is shapes per world.  The correct
    upper bound is the sum of per-world lower-triangular counts which is
    O(W * S^2).

    The result mirrors the segment layout produced by
    :func:`precompute_world_map`: each regular world's segment contains the
    world's local shapes **plus** all global shapes (world == -1), and a
    dedicated final segment contains only the global shapes.  Each segment
    contributes ``n*(n-1)/2`` candidate pairs independently.
    """
    shape_world = getattr(model, "shape_world", None)
    shape_count = model.shape_count
    if shape_world is None or shape_count <= 1:
        return max(0, (shape_count * (shape_count - 1)) // 2)

    sw = shape_world.numpy()
    shape_flags = getattr(model, "shape_flags", None)
    if shape_flags is not None:
        sf = shape_flags.numpy()
        colliding = (sf & int(ShapeFlags.COLLIDE_SHAPES)) != 0
    else:
        colliding = np.ones(len(sw), dtype=bool)

    global_count = int(np.count_nonzero((sw == -1) & colliding))
    world_ids = np.unique(sw[(sw >= 0) & colliding])

    total = 0
    for wid in world_ids:
        n = int(np.count_nonzero((sw == wid) & colliding)) + global_count
        total += (n * (n - 1)) // 2

    # Dedicated global-vs-global segment (appended by precompute_world_map).
    total += (global_count * (global_count - 1)) // 2

    return max(0, total)


def _resolve_shape_pairs_max(model: Model, override: int | None) -> int:
    """Pick the broad-phase candidate-pair buffer capacity.

    ``override`` lets the caller cap the SAP/NXN pair buffer, which is
    otherwise sized to the worst-case ``N*(N-1)/2`` per-world bound.
    SAP and NXN scenes with thousands of bodies typically emit only a
    tiny fraction of that bound, so the default sizing is grossly
    wasteful (multi-GB on 10k+ shape scenes). ``None`` keeps the legacy
    behaviour; a positive integer overrides it. ``0`` is rejected --
    use ``None`` instead.  Values larger than the natural bound are
    accepted as-is: allocating beyond the bound never produces more
    pairs, but we honour the user's explicit capacity request rather
    than silently shrinking it.
    """
    if override is None:
        return _compute_per_world_shape_pairs_max(model)
    if override <= 0:
        raise ValueError(f"shape_pairs_max must be a positive integer or None, got {override}")
    return int(override)


BROAD_PHASE_MODES = ("nxn", "sap", "explicit")


def _normalize_broad_phase_mode(mode: str) -> str:
    mode_str = str(mode).lower()
    if mode_str not in BROAD_PHASE_MODES:
        raise ValueError(f"Unsupported broad phase mode: {mode!r}")
    return mode_str


def _infer_broad_phase_mode_from_instance(broad_phase: BroadPhaseAllPairs | BroadPhaseSAP | BroadPhaseExplicit) -> str:
    if isinstance(broad_phase, BroadPhaseAllPairs):
        return "nxn"
    if isinstance(broad_phase, BroadPhaseSAP):
        return "sap"
    if isinstance(broad_phase, BroadPhaseExplicit):
        return "explicit"
    raise TypeError(
        "broad_phase must be a BroadPhaseAllPairs, BroadPhaseSAP, or BroadPhaseExplicit instance "
        f"(got {type(broad_phase)!r})"
    )


def _world_compatible_pairs(
    feature_world: np.ndarray,
    shape_world: np.ndarray,
    world_count: int,
    device,
    shape_ok: np.ndarray | None = None,
) -> wp.array[wp.vec2i]:
    """Emit ``(feature, shape)`` index pairs whose worlds are compatible: same world, or either is
    global (``-1``). ``feature_world[i]`` / ``shape_world[s]`` give each entity's world (-1 == global).

    Worlds are immutable after :meth:`~newton.ModelBuilder.finalize`, so this filtering is safe to
    precompute; mutable per-entity flags (ACTIVE / COLLIDE_PARTICLES) are deliberately left to the
    per-thread kernel. The compatibility predicate splits into three disjoint groups, each a
    vectorized cross product (disjoint => no de-duplication; no Python loop over features or shapes).
    Reads host arrays, so it is not graph-capture-safe; call at pipeline construction.
    """
    n_features = len(feature_world)
    n_shapes = len(shape_world)

    def _pairs(f_idx: np.ndarray, s_idx: np.ndarray) -> wp.array[wp.vec2i]:
        # ``shape_ok`` (optional, indexed by shape) drops pairs whose shape cannot participate -- e.g.
        # full-surface edge/face excludes shapes without a usable SDF, which fall back to per-particle.
        if shape_ok is not None and len(s_idx):
            keep = shape_ok[s_idx.astype(np.intp)]
            f_idx, s_idx = f_idx[keep], s_idx[keep]
        stacked = np.column_stack((f_idx, s_idx)).astype(np.int32) if len(f_idx) else np.empty((0, 2), np.int32)
        return wp.array(stacked, dtype=wp.vec2i, device=device)

    if n_features == 0 or n_shapes == 0:
        return _pairs(np.empty(0), np.empty(0))

    features = np.arange(n_features)
    shapes = np.arange(n_shapes)
    f_local = (feature_world >= 0) & (feature_world < world_count)
    s_local = (shape_world >= 0) & (shape_world < world_count)

    f_cols: list[np.ndarray] = []
    s_cols: list[np.ndarray] = []

    # 1. Global features pair with every shape (any world).
    global_features = features[feature_world < 0]
    if len(global_features):
        f_cols.append(np.repeat(global_features, len(shapes)))
        s_cols.append(np.tile(shapes, len(global_features)))

    # 2. Local-world features additionally pair with every global shape.
    local_features = features[f_local]
    global_shapes = shapes[shape_world < 0]
    if len(local_features) and len(global_shapes):
        f_cols.append(np.repeat(local_features, len(global_shapes)))
        s_cols.append(np.tile(global_shapes, len(local_features)))

    # 3. Local-world features pair with the shapes that share their world. Group the local shapes by
    #    world so each world's shapes are contiguous, then for every feature slice out its world's block.
    local_feature_world = feature_world[f_local]
    shapes_per_world = np.bincount(shape_world[s_local], minlength=world_count)
    reps = shapes_per_world[local_feature_world] if len(local_feature_world) else np.zeros(0, np.intp)
    if reps.sum():
        shapes_by_world = shapes[s_local][np.argsort(shape_world[s_local], kind="stable")]
        world_start = np.cumsum(shapes_per_world) - shapes_per_world
        within = np.arange(reps.sum()) - np.repeat(np.cumsum(reps) - reps, reps)
        f_cols.append(np.repeat(local_features, reps))
        s_cols.append(shapes_by_world[np.repeat(world_start[local_feature_world], reps) + within])

    if not f_cols:
        return _pairs(np.empty(0), np.empty(0))
    return _pairs(np.concatenate(f_cols), np.concatenate(s_cols))


def _build_soft_particle_rigid_contact_pairs(
    model: Model, capable_shape_mask: np.ndarray | None = None
) -> wp.array[wp.vec2i]:
    """Build the soft-rigid (particle-shape) candidate pairs for ``model``.

    Emits every particle-shape pair whose worlds are compatible (see :func:`_world_compatible_pairs`).
    :attr:`~newton.ParticleFlags.ACTIVE` and :attr:`~newton.ShapeFlags.COLLIDE_PARTICLES` are applied
    per-thread in :func:`~newton._src.geometry.kernels.create_soft_contacts`, not here, so the
    candidate set stays valid when those flags change after the pipeline is constructed.
    """
    particle_count = int(getattr(model, "particle_count", 0) or 0)
    shape_count = int(getattr(model, "shape_count", 0) or 0)
    if particle_count == 0 or shape_count == 0:
        return wp.array(np.empty((0, 2), np.int32), dtype=wp.vec2i, device=model.device)
    world_count = int(getattr(model, "world_count", 0) or 0)
    return _world_compatible_pairs(
        model.particle_world.numpy(),
        model.shape_world.numpy(),
        world_count,
        model.device,
        shape_ok=capable_shape_mask,
    )


def _count_soft_particle_rigid_contact_pairs(model: Model) -> int:
    """Count exactly how many pairs :func:`_build_soft_particle_rigid_contact_pairs` emits for ``model``.

    Reads only the per-world start offsets, so solvers can pre-size soft-contact buffers without
    downloading per-entity world ids. This is not :attr:`CollisionPipeline.soft_contact_max`, which
    additionally reserves edge/face headroom when ``enable_rigid_soft_full_surface_contact`` is set.
    Reads host arrays, so it is not graph-capture-safe; call at solver construction.
    """
    particle_start = model.particle_world_start.numpy()
    shape_start = model.shape_world_start.numpy()
    global_particles = int(particle_start[-1] - particle_start[-2] + particle_start[0])
    global_shapes = int(shape_start[-1] - shape_start[-2] + shape_start[0])
    # Global particles pair with every shape; local particles additionally pair with global shapes.
    total = global_particles * model.shape_count
    total += (model.particle_count - global_particles) * global_shapes
    # Local particles pair with the shapes sharing their world.
    per_world = slice(0, model.world_count + 1)
    return total + int(
        np.dot(np.diff(particle_start[per_world]).astype(np.int64), np.diff(shape_start[per_world]).astype(np.int64))
    )


def _build_soft_face_rigid_contact_pairs(
    model: Model, capable_shape_mask: np.ndarray | None = None
) -> wp.array[wp.vec2i]:
    """World-compatible ``(soft triangle, shape)`` candidate pairs for the full-surface FACE pass,
    mirroring :func:`_build_soft_particle_rigid_contact_pairs`. A triangle's world is the world of
    its first vertex (all three share it). Empty when there are no triangles or no shapes.
    """
    device = model.device
    empty = wp.array(np.empty((0, 2), np.int32), dtype=wp.vec2i, device=device)
    shape_count = int(getattr(model, "shape_count", 0) or 0)
    n_tris = int(getattr(model, "tri_count", 0) or 0)
    if shape_count == 0 or n_tris == 0:
        return empty
    world_count = int(getattr(model, "world_count", 0) or 0)
    face_world = model.particle_world.numpy()[model.tri_indices.numpy()[:, 0]]
    return _world_compatible_pairs(
        face_world, model.shape_world.numpy(), world_count, device, shape_ok=capable_shape_mask
    )


def _build_soft_edge_rigid_contact_pairs(
    model: Model, capable_shape_mask: np.ndarray | None = None
) -> wp.array[wp.vec2i]:
    """World-compatible ``(soft edge, shape)`` candidate pairs for the full-surface EDGE pass,
    mirroring :func:`_build_soft_particle_rigid_contact_pairs`. An edge's world is that of one of its
    endpoints. Endpoints come straight from ``model.edge_indices`` (no mesh adjacency needed). Empty
    when there are no edges or no shapes.
    """
    device = model.device
    empty = wp.array(np.empty((0, 2), np.int32), dtype=wp.vec2i, device=device)
    shape_count = int(getattr(model, "shape_count", 0) or 0)
    n_edges = int(getattr(model, "edge_count", 0) or 0)
    if shape_count == 0 or n_edges == 0:
        return empty
    world_count = int(getattr(model, "world_count", 0) or 0)
    # edge_indices rows are [o0, o1, v0, v1]; col 2 (v0) is an endpoint, so its world is the edge's.
    edge_world = model.particle_world.numpy()[model.edge_indices.numpy()[:, 2]]
    return _world_compatible_pairs(
        edge_world, model.shape_world.numpy(), world_count, device, shape_ok=capable_shape_mask
    )


def _full_surface_capable_shape_mask(model: Model) -> np.ndarray:
    """Boolean mask over shapes: ``True`` where the shape can generate full-surface edge/face contacts.

    Capable: analytic primitives (sphere/box/capsule/cylinder/cone/ellipsoid), an *infinite* plane
    (width=length=0), and a mesh/convex with a real provisioned SDF (nonnegative ``_shape_sdf_index``
    pointing at a non-empty descriptor). Not capable -- the shape falls back to per-particle soft
    contact: heightfields (edge/face SDF optimization is unsupported), finite planes (the +Z normal is
    wrong off the quad), and mesh/convex shapes without a real SDF (a nonnegative index can still point
    at an empty BVH-fallback descriptor, whose coarse texture is ``None``).
    """
    stype = model.shape_type.numpy()
    scale = model.shape_scale.numpy()
    analytic = np.isin(
        stype,
        (
            int(GeoType.SPHERE),
            int(GeoType.BOX),
            int(GeoType.CAPSULE),
            int(GeoType.CYLINDER),
            int(GeoType.CONE),
            int(GeoType.ELLIPSOID),
        ),
    )
    infinite_plane = (stype == int(GeoType.PLANE)) & (scale[:, 0] == 0.0) & (scale[:, 1] == 0.0)
    is_mesh = np.isin(stype, (int(GeoType.MESH), int(GeoType.CONVEX_MESH)))
    has_real_sdf = np.zeros(len(stype), dtype=bool)
    if getattr(model, "_shape_sdf_index", None) is not None:
        sidx = model._shape_sdf_index.numpy()
        coarse = getattr(model, "_texture_sdf_coarse_textures", None)
        has_real_sdf = np.array(
            [s >= 0 and coarse is not None and s < len(coarse) and coarse[s] is not None for s in sidx],
            dtype=bool,
        )
    return analytic | infinite_plane | (is_mesh & has_real_sdf)


def _raise_on_unprovisioned_full_surface_meshes(model: Model, capable: np.ndarray) -> None:
    """A participating mesh/convex without a real SDF is a provisioning *mistake*, not an inherent
    limitation, so fail loudly (the edge/face passes would otherwise sample an empty descriptor and a
    soft body could pass straight through). Distinct from the unsupported shape *types*, which warn
    and fall back -- see :func:`_warn_full_surface_fallbacks`."""
    stype = model.shape_type.numpy()
    is_mesh = np.isin(stype, (int(GeoType.MESH), int(GeoType.CONVEX_MESH)))
    collide_particles = (model.shape_flags.numpy() & int(ShapeFlags.COLLIDE_PARTICLES)) != 0
    unprovisioned = np.where(is_mesh & collide_particles & ~capable)[0]
    if unprovisioned.size == 0:
        return
    labels = getattr(model, "shape_key", None)
    missing = [(labels[i] if labels is not None and i < len(labels) else f"shape {int(i)}") for i in unprovisioned]
    raise ValueError(
        f"enable_rigid_soft_full_surface_contact=True, but these participating rigid shapes have no "
        f"signed-distance field: {missing}. The edge and face contact passes sample each rigid "
        f"mesh/convex shape's SDF, so a shape without one is skipped and a soft body can pass straight "
        f"through it. Provision an SDF before ModelBuilder.finalize(), any one of these ways:\n"
        f"  - For shapes that use the builder's default config (including importer-added shapes): "
        f"set builder.default_shape_cfg.configure_sdf(force_sdf=True) before you add or import them.\n"
        f"  - For a shape you gave an explicit config: call configure_sdf() on that config, e.g. "
        f"cfg.configure_sdf(force_sdf=True) (optionally max_resolution=... or target_voxel_size=...).\n"
        f"  - Manually: build one with mesh.build_sdf() and attach it to the shape.\n"
        f"Or set enable_rigid_soft_full_surface_contact=False to use per-vertex (particle) contacts only."
    )


def _warn_full_surface_fallbacks(model: Model, capable: np.ndarray) -> None:
    """Warn about participating shapes whose *type* cannot do edge/face -- heightfields, finite planes,
    Gaussian splats, the NONE placeholder -- which fall back to per-particle soft contact. Mesh/convex
    without an SDF is handled separately (it raises; see
    :func:`_raise_on_unprovisioned_full_surface_meshes`), so it is excluded here."""
    stype = model.shape_type.numpy()
    is_mesh = np.isin(stype, (int(GeoType.MESH), int(GeoType.CONVEX_MESH)))
    collide_particles = (model.shape_flags.numpy() & int(ShapeFlags.COLLIDE_PARTICLES)) != 0
    fallback = np.where(collide_particles & ~capable & ~is_mesh)[0]
    if fallback.size == 0:
        return
    labels = getattr(model, "shape_key", None)

    def _label(i: int) -> str:
        return labels[i] if labels is not None and i < len(labels) else f"shape {int(i)}"

    heightfields, finite_planes, other = [], [], []
    for i in fallback:
        if stype[i] == int(GeoType.HFIELD):
            heightfields.append(_label(i))
        elif stype[i] == int(GeoType.PLANE):
            finite_planes.append(_label(i))
        else:
            other.append(_label(i))
    reasons = []
    if heightfields:
        reasons.append(f"heightfields {heightfields} (edge/face SDF optimization is not supported)")
    if finite_planes:
        reasons.append(f"finite planes {finite_planes} (only infinite planes are supported)")
    if other:
        reasons.append(f"shape types without an analytic signed-distance field {other}")
    warnings.warn(
        "enable_rigid_soft_full_surface_contact=True: these participating shapes cannot generate "
        "edge/face contacts and fall back to per-particle soft contact only -- "
        + "; ".join(reasons)
        + ". Full-surface contacts still apply to the rest of the scene.",
        stacklevel=3,
    )


class CollisionPipeline:
    """
    Full-featured collision pipeline with GJK/MPR narrow phase and pluggable broad phase.

    Key features:
        - GJK/MPR algorithms for convex-convex collision detection
        - Multiple broad phase options: NXN (all-pairs), SAP (sweep-and-prune), EXPLICIT (precomputed pairs)
        - Mesh-mesh collision via SDF with contact reduction
        - Optional hydroelastic contact model for compliant surfaces

    For most users, construct with ``CollisionPipeline(model, ...)``.

    .. experimental::

        Differentiable rigid contacts (the ``rigid_contact_diff_*`` arrays when
        ``requires_grad`` is enabled) may change without prior notice. The
        narrow phase stays frozen and gradients are a tangent approximation;
        validate accuracy and usefulness on your workflow before relying on
        them in optimization loops.
    """

    def __init__(
        self,
        model: Model,
        *,
        reduce_contacts: bool = True,
        rigid_contact_max: int | None = None,
        max_triangle_pairs: int = 1000000,
        shape_pairs_filtered: wp.array[wp.vec2i] | None = None,
        include_static_kinematic_pairs: bool = True,
        soft_contact_max: int | None = None,
        soft_contact_gap: float | None = None,
        soft_contact_margin: float | None = None,
        enable_rigid_soft_full_surface_contact: bool = False,
        full_surface_mesh_backend: Literal["bvh", "sdf"] = "bvh",
        rigid_soft_bvh_candidate_max: int | None = None,
        requires_grad: bool | None = None,
        broad_phase: Literal["nxn", "sap", "explicit"]
        | BroadPhaseAllPairs
        | BroadPhaseSAP
        | BroadPhaseExplicit
        | None = None,
        narrow_phase: NarrowPhase | None = None,
        sdf_hydroelastic_config: HydroelasticSDF.Config | None = None,
        shape_pairs_max: int | None = None,
        deterministic: bool = False,
        contact_matching: Literal["disabled", "latest", "sticky"] = "disabled",
        contact_matching_pos_threshold: float = 0.0005,
        contact_matching_normal_dot_threshold: float = 0.995,
        contact_report: bool = False,
        verify_buffers: bool = True,
        contact_reduction_hashtable_size_factor: float = 0.25,
    ):
        """
        Initialize the CollisionPipeline (expert API).

        Args:
            model: The simulation model.
            reduce_contacts: Whether to reduce contacts for mesh-mesh collisions. Defaults to True.
            rigid_contact_max: Maximum number of rigid contacts to allocate.
                Resolution order:
                - If provided, use this value.
                - Else if ``model.rigid_contact_max > 0``, use the model value.
                - Else estimate automatically from model shape and pair metadata.
            max_triangle_pairs:
                Maximum number of triangle pairs allocated by narrow phase
                for mesh and heightfield collisions.  Increase this when
                scenes with large/complex meshes or heightfields report
                triangle-pair overflow warnings.
            contact_reduction_hashtable_size_factor: Multiplier applied to
                ``max_triangle_pairs`` when allocating the global contact
                reduction hashtable. Increase this if hashtable fill/failure
                warnings appear. Defaults to ``0.25`` for memory compatibility.
            soft_contact_max: Maximum number of soft contacts to allocate.
                If None, defaults to ``soft_contact_pair_count``, the number
                of precomputed soft-rigid (particle-shape) pairs launched for soft
                contact generation, plus the full-surface edge/face headroom when
                ``enable_rigid_soft_full_surface_contact`` is set.
            soft_contact_gap: Detection-only distance [m] added to the
                per-particle radius for particle-shape (soft) contact queries.
                Defaults to 0.01.
            soft_contact_margin: Deprecated alias of ``soft_contact_gap`` (the
                value is detection-only slack on top of the particle radius,
                i.e. a gap under the margin/gap convention).
            enable_rigid_soft_full_surface_contact: Generate soft contacts over the full soft-mesh
                surface, in addition to the ordinary particle-shape contacts. Analytic rigid shapes
                use SDF minimization. Mesh and convex-mesh shapes use the backend selected by
                ``full_surface_mesh_backend``. This catches rigid features that pass between soft
                vertices (for example, a thin box edge through a coarse cloth cell).
            full_surface_mesh_backend: Backend for mesh and convex-mesh rigid shapes when
                ``enable_rigid_soft_full_surface_contact`` is enabled. ``"bvh"`` (default)
                enumerates full VF/FV/EE primitive pairs and does not require a volume SDF;
                ``"sdf"`` retains the SDF-minimization implementation and requires a provisioned SDF.
                BVH normals use the rigid mesh winding: closed solid meshes require coherent
                outward winding, while an open mesh is treated as a one-sided surface whose
                positive-winding side is outside. The BVH backend does not reconstruct a global
                inside/outside sign for open or inconsistently wound meshes.
                The SDF backend requires an SDF on every participating rigid mesh/convex shape
                (provision via :meth:`ModelBuilder.ShapeConfig.configure_sdf`, for example
                ``configure_sdf(force_sdf=True)`` on the builder's ``default_shape_cfg``).
                Full-surface records are consumed only by :class:`~newton.solvers.SolverVBD`;
                other solvers raise on such contacts. Both options are fixed at construction
                because they determine contact and intermediate-buffer capacities.
            rigid_soft_bvh_candidate_max: Capacity of the intermediate rigid-soft BVH
                primitive-pair buffer. Defaults to four candidates per BVH query seed. This is
                independent of ``soft_contact_max``, which sizes the shared final soft-contact
                output buffer. Ignored when the rigid-soft BVH path has no query seeds.
            requires_grad: Whether to enable gradient computation. If None, uses model.requires_grad.
            broad_phase:
                Either a broad phase mode string ("explicit", "nxn", "sap") or
                a prebuilt broad phase instance for expert usage.
            narrow_phase: Optional prebuilt narrow phase instance. Must be
                provided together with a broad phase instance for expert usage.
            shape_pairs_filtered: Precomputed shape pairs for EXPLICIT mode.
                When broad_phase is "explicit", uses model.shape_contact_pairs if not provided. For
                "nxn"/"sap" modes, ignored.
            include_static_kinematic_pairs: Whether to generate contacts for
                pairs where both shapes are immovable. Set to ``False`` to
                filter static-static, static-kinematic, and
                kinematic-kinematic pairs. Defaults to ``True`` for backward
                compatibility.
            sdf_hydroelastic_config: Configuration for hydroelastic collision
                handling. Defaults to None.
            shape_pairs_max: Override for the broad-phase candidate-pair
                buffer capacity used by the ``"nxn"`` and ``"sap"`` modes.
                Defaults to the worst-case ``N*(N-1)/2`` per-world bound,
                which is rarely hit by either ``"nxn"`` or ``"sap"`` in
                practice -- ``"nxn"`` still applies AABB overlap, group,
                and excluded-pair filtering inside ``BroadPhaseAllPairs``
                before writing, and ``"sap"`` is sparse by design -- so
                the default sizing is typically 10-100x larger than what
                gets emitted on real scenes. Set this to a tighter value
                (e.g. measured peak with ~25% headroom) to avoid multi-GB
                allocations on large scenes; a too-small value triggers
                a buffer overflow warning at runtime. Ignored for the
                ``"explicit"`` mode (which uses the filtered pair list
                length directly) and for expert paths that pass a
                pre-built ``narrow_phase``.
            deterministic: Sort contacts after the narrow phase so that results
                are independent of GPU thread scheduling. This also enables
                deterministic hydroelastic accumulation and contact allocation.
                Adds a radix sort + gather pass.
            contact_matching: Frame-to-frame contact matching mode.  One of
                ``"disabled"``, ``"latest"``, or ``"sticky"``.  Any
                non-disabled mode implies ``deterministic=True`` and
                populates :attr:`Contacts.rigid_contact_match_index`.
                Defaults to ``"disabled"``.

                .. experimental::

                    The ``"sticky"`` mode may change without prior notice.
            contact_matching_pos_threshold: World-space distance threshold [m]
                between the previous and current contact midpoints
                ``0.5 * (world(point0) + world(point1))``.  Contacts whose
                midpoint moves more than this are considered broken.  Defaults
                to ``0.0005``.
            contact_matching_normal_dot_threshold: Minimum dot product between
                old and new contact normals for a match.
            contact_report: Allocate ``rigid_contact_new_indices`` /
                ``rigid_contact_new_count`` / ``rigid_contact_broken_indices``
                / ``rigid_contact_broken_count`` on the :class:`Contacts`
                container, populated each frame.  Requires a non-disabled
                ``contact_matching`` mode.
            verify_buffers: Run a ``dim=[1]`` diagnostic kernel at the end of
                the narrow phase that prints warnings on any intermediate
                candidate-pair or final rigid contact buffer overflow; see
                :class:`NarrowPhase` for the full counter list.  Defaults to
                ``True``.  Overhead is one extra kernel launch per collision
                pass; disable in hot loops or CUDA graph capture once buffer
                sizes are known to be adequate.

        .. experimental::

            When ``requires_grad`` is true (explicitly or via
            ``model.requires_grad``), rigid-contact autodiff via
            ``rigid_contact_diff_*`` may change without prior notice; see
            :meth:`collide`.
        """
        if contact_matching not in ("disabled", "latest", "sticky"):
            raise ValueError(
                f"contact_matching must be one of 'disabled', 'latest', 'sticky', got {contact_matching!r}"
            )
        if full_surface_mesh_backend not in ("bvh", "sdf"):
            raise ValueError(
                f"full_surface_mesh_backend must be either 'bvh' or 'sdf', got {full_surface_mesh_backend!r}"
            )
        if contact_matching_pos_threshold < 0.0:
            raise ValueError(
                f"contact_matching_pos_threshold must be non-negative, got {contact_matching_pos_threshold}"
            )
        if not -1.0 <= contact_matching_normal_dot_threshold <= 1.0:
            raise ValueError(
                f"contact_matching_normal_dot_threshold must be in [-1, 1], got {contact_matching_normal_dot_threshold}"
            )
        matching_enabled = contact_matching != "disabled"
        matching_sticky = contact_matching == "sticky"
        if contact_report and not matching_enabled:
            raise ValueError('contact_report=True requires contact_matching != "disabled"')

        # Any non-disabled matching mode implies deterministic sorting.
        if matching_enabled:
            deterministic = True

        mode_from_broad_phase: str | None = None
        broad_phase_instance: BroadPhaseAllPairs | BroadPhaseSAP | BroadPhaseExplicit | None = None
        if broad_phase is not None:
            if isinstance(broad_phase, str):
                mode_from_broad_phase = _normalize_broad_phase_mode(broad_phase)
            else:
                broad_phase_instance = broad_phase

        shape_count = model.shape_count
        device = model.device
        using_expert_components = broad_phase_instance is not None or narrow_phase is not None

        # Resolve rigid contact capacity with explicit > model > estimated precedence.
        if rigid_contact_max is None:
            model_rigid_contact_max = int(getattr(model, "rigid_contact_max", 0) or 0)
            if model_rigid_contact_max > 0:
                rigid_contact_max = model_rigid_contact_max
            else:
                rigid_contact_max = _estimate_rigid_contact_max(model)
        self._rigid_contact_max = rigid_contact_max
        if soft_contact_margin is not None:
            if soft_contact_gap is not None:
                raise ValueError("soft_contact_margin is a deprecated alias of soft_contact_gap; pass only one")
            warnings.warn(
                "The soft_contact_margin parameter of CollisionPipeline is deprecated; "
                "use soft_contact_gap (same value: detection-only distance added to the particle radius).",
                DeprecationWarning,
                stacklevel=2,
            )
            soft_contact_gap = soft_contact_margin
        elif soft_contact_gap is None:
            soft_contact_gap = 0.01
        if soft_contact_gap < 0.0:
            raise ValueError(f"soft_contact_gap must be >= 0, got {soft_contact_gap}")

        if max_triangle_pairs <= 0:
            raise ValueError("max_triangle_pairs must be > 0")
        # Keep model-level default in sync with the resolved pipeline capacity.
        # This avoids divergence between model- and contacts-based users (e.g. VBD init).
        model.rigid_contact_max = rigid_contact_max
        if requires_grad is None:
            requires_grad = model.requires_grad

        shape_world = getattr(model, "shape_world", None)
        shape_flags = getattr(model, "shape_flags", None)
        with wp.ScopedDevice(device):
            shape_aabb_lower = wp.zeros(shape_count, dtype=wp.vec3, device=device)
            shape_aabb_upper = wp.zeros(shape_count, dtype=wp.vec3, device=device)

        self.model = model
        self.shape_count = shape_count
        self.device = device
        self.reduce_contacts = reduce_contacts
        self.requires_grad = requires_grad
        self.include_static_kinematic_pairs = include_static_kinematic_pairs

        if using_expert_components:
            if broad_phase_instance is None or narrow_phase is None:
                raise ValueError("Provide both broad_phase and narrow_phase for expert component construction")
            if sdf_hydroelastic_config is not None:
                raise ValueError("sdf_hydroelastic_config cannot be used when narrow_phase is provided")
            if contact_reduction_hashtable_size_factor != 0.25:
                raise ValueError(
                    "contact_reduction_hashtable_size_factor cannot be used when narrow_phase is provided; "
                    "construct the NarrowPhase with that value instead"
                )
            inferred_mode = _infer_broad_phase_mode_from_instance(broad_phase_instance)
            self.broad_phase_mode = inferred_mode
            self.broad_phase = broad_phase_instance

            if self.broad_phase_mode == "explicit":
                if shape_pairs_filtered is None:
                    shape_pairs_filtered = getattr(model, "shape_contact_pairs", None)
                if shape_pairs_filtered is None:
                    raise ValueError(
                        "shape_pairs_filtered must be provided for explicit broad phase "
                        "(or set model.shape_contact_pairs)"
                    )
                self.shape_pairs_filtered = shape_pairs_filtered
                self.shape_pairs_max = len(shape_pairs_filtered)
                self.shape_pairs_excluded = None
                self.shape_pairs_excluded_count = 0
            else:
                self.shape_pairs_filtered = None
                self.shape_pairs_max = _compute_per_world_shape_pairs_max(model)
                self.shape_pairs_excluded = self._build_excluded_pairs(model)
                self.shape_pairs_excluded_count = (
                    self.shape_pairs_excluded.shape[0] if self.shape_pairs_excluded is not None else 0
                )

            if deterministic and not narrow_phase.deterministic:
                raise ValueError(
                    "CollisionPipeline(deterministic=True) requires a deterministic "
                    "NarrowPhase. Either omit narrow_phase or construct it with "
                    "deterministic=True."
                )
            if narrow_phase.max_candidate_pairs < self.shape_pairs_max:
                raise ValueError(
                    "Provided narrow_phase.max_candidate_pairs is too small for this model and broad phase mode "
                    f"(required at least {self.shape_pairs_max}, got {narrow_phase.max_candidate_pairs})"
                )
            self.narrow_phase = narrow_phase
            self.hydroelastic_sdf = self.narrow_phase.hydroelastic_sdf
        else:
            self.broad_phase_mode = mode_from_broad_phase if mode_from_broad_phase is not None else "explicit"

            if self.broad_phase_mode == "explicit":
                if shape_pairs_filtered is None:
                    shape_pairs_filtered = getattr(model, "shape_contact_pairs", None)
                if shape_pairs_filtered is None:
                    raise ValueError(
                        "shape_pairs_filtered must be provided for broad_phase=EXPLICIT "
                        "(or set model.shape_contact_pairs)"
                    )
                self.broad_phase = BroadPhaseExplicit()
                self.shape_pairs_filtered = shape_pairs_filtered
                self.shape_pairs_max = len(shape_pairs_filtered)
                self.shape_pairs_excluded = None
                self.shape_pairs_excluded_count = 0
            elif self.broad_phase_mode == "nxn":
                if shape_world is None:
                    raise ValueError("model.shape_world is required for broad_phase=NXN")
                self.broad_phase = BroadPhaseAllPairs(shape_world, shape_flags=shape_flags, device=device)
                self.shape_pairs_filtered = None
                self.shape_pairs_max = _resolve_shape_pairs_max(model, shape_pairs_max)
                self.shape_pairs_excluded = self._build_excluded_pairs(model)
                self.shape_pairs_excluded_count = (
                    self.shape_pairs_excluded.shape[0] if self.shape_pairs_excluded is not None else 0
                )
            elif self.broad_phase_mode == "sap":
                if shape_world is None:
                    raise ValueError("model.shape_world is required for broad_phase=SAP")
                self.broad_phase = BroadPhaseSAP(shape_world, shape_flags=shape_flags, device=device)
                self.shape_pairs_filtered = None
                self.shape_pairs_max = _resolve_shape_pairs_max(model, shape_pairs_max)
                self.shape_pairs_excluded = self._build_excluded_pairs(model)
                self.shape_pairs_excluded_count = (
                    self.shape_pairs_excluded.shape[0] if self.shape_pairs_excluded is not None else 0
                )
            else:
                raise ValueError(f"Unsupported broad phase mode: {self.broad_phase_mode}")

            # Initialize SDF hydroelastic (returns None if no hydroelastic shape pairs in the model)
            hydroelastic_sdf = HydroelasticSDF._from_model(
                model,
                config=sdf_hydroelastic_config,
                writer_func=write_contact,
                deterministic=deterministic,
            )

            # Detect shape classes to optimize narrow-phase kernel launches.
            # Keep mesh and heightfield flags independent: heightfield-only scenes
            # should not trigger mesh-only kernel setup/launches.
            has_meshes = False
            use_lean_gjk_mpr = False
            mesh_sdf_texture_only = False
            mesh_sdf_identity_scale_only = False
            if hasattr(model, "shape_type") and model.shape_type is not None:
                shape_types = model.shape_type.numpy()
                colliding_mask = _shape_collide_mask(model, len(shape_types))
                colliding_shape_types = shape_types[colliding_mask]
                has_meshes = bool((colliding_shape_types == int(GeoType.MESH)).any())
                if (
                    hasattr(model, "_shape_sdf_index")
                    and model._shape_sdf_index is not None
                    and hasattr(model, "shape_edge_range")
                    and model.shape_edge_range is not None
                ):
                    shape_sdf_index = model._shape_sdf_index.numpy()
                    shape_edge_range = model.shape_edge_range.numpy()
                    has_planar_sdf_shapes = bool(
                        np.any(colliding_mask & (shape_sdf_index >= 0) & (shape_edge_range[:, 1] > 0))
                    )
                    has_meshes = has_meshes or has_planar_sdf_shapes
                    mesh_sdf_shapes = colliding_mask & (
                        (shape_types != int(GeoType.HFIELD))
                        & ((shape_types == int(GeoType.MESH)) | (shape_edge_range[:, 1] > 0))
                    )
                    coarse_textures = getattr(model, "_texture_sdf_coarse_textures", None)
                    has_texture_sdf = np.array(
                        [
                            sdf_idx >= 0
                            and coarse_textures is not None
                            and sdf_idx < len(coarse_textures)
                            and coarse_textures[sdf_idx] is not None
                            for sdf_idx in shape_sdf_index
                        ],
                        dtype=bool,
                    )
                    mesh_sdf_texture_only = bool(np.any(mesh_sdf_shapes) and np.all(has_texture_sdf[mesh_sdf_shapes]))
                    if mesh_sdf_texture_only:
                        texture_sdf_data = model._texture_sdf_data.numpy()
                        scale_baked = texture_sdf_data["scale_baked"]
                        shape_scale = model.shape_scale.numpy()
                        identity_shape_scale = np.all(shape_scale == np.float32(1.0), axis=1)
                        mesh_sdf_identity_scale_only = all(
                            bool(scale_baked[shape_sdf_index[shape_idx]]) or identity_shape_scale[shape_idx]
                            for shape_idx in np.flatnonzero(mesh_sdf_shapes)
                        )
                # Use lean GJK/MPR kernel when scene has no capsules, ellipsoids,
                # cylinders, or cones (which need full support function and axial
                # rolling post-processing)
                lean_unsupported = {
                    int(GeoType.CAPSULE),
                    int(GeoType.ELLIPSOID),
                    int(GeoType.CYLINDER),
                    int(GeoType.CONE),
                }
                use_lean_gjk_mpr = not bool(lean_unsupported & set(colliding_shape_types.tolist()))

            # Initialize narrow phase with pre-allocated buffers
            # max_triangle_pairs is a conservative estimate for mesh collision triangle pairs
            # Pass write_contact as custom writer to write directly to final Contacts format
            #
            # contact_max is passed explicitly so NarrowPhase sizes its internal
            # deterministic sort buffers to rigid_contact_max (the same capacity
            # the Contacts buffer uses) rather than falling back to the default
            # max_candidate_pairs.  On SAP/NXN scenes with thousands of shapes
            # the candidate-pair bound (N*(N-1)/2 per world) is orders of
            # magnitude larger than the neighbor-budget contact estimate and
            # allocating sorter scratch at that size burns multi-GB of VRAM.
            self.narrow_phase = NarrowPhase(
                max_candidate_pairs=self.shape_pairs_max,
                max_triangle_pairs=max_triangle_pairs,
                reduce_contacts=self.reduce_contacts,
                device=device,
                shape_aabb_lower=shape_aabb_lower,
                shape_aabb_upper=shape_aabb_upper,
                contact_writer_warp_func=write_contact,
                shape_voxel_resolution=model._shape_voxel_resolution,
                hydroelastic_sdf=hydroelastic_sdf,
                has_meshes=has_meshes,
                has_heightfields=model.heightfield_count > 0,
                use_lean_gjk_mpr=use_lean_gjk_mpr,
                mesh_sdf_identity_scale_only=mesh_sdf_identity_scale_only,
                mesh_sdf_texture_only=mesh_sdf_texture_only,
                deterministic=deterministic,
                contact_max=rigid_contact_max,
                verify_buffers=verify_buffers,
                contact_reduction_hashtable_size_factor=contact_reduction_hashtable_size_factor,
            )
            self.hydroelastic_sdf = self.narrow_phase.hydroelastic_sdf

        # Allocate buffers
        with wp.ScopedDevice(device):
            self.broad_phase_pair_count = wp.zeros(1, dtype=wp.int32, device=device)
            self.broad_phase_shape_pairs = wp.zeros(self.shape_pairs_max, dtype=wp.vec2i, device=device)
            self.geom_data = wp.zeros(shape_count, dtype=wp.vec4, device=device)
            self.geom_transform = wp.zeros(shape_count, dtype=wp.transform, device=device)

        if (
            getattr(self.narrow_phase, "shape_aabb_lower", None) is None
            or getattr(self.narrow_phase, "shape_aabb_upper", None) is None
        ):
            raise ValueError("narrow_phase must expose shape_aabb_lower and shape_aabb_upper arrays")
        if self.narrow_phase.shape_aabb_lower.shape[0] != shape_count:
            raise ValueError(
                "narrow_phase.shape_aabb_lower must have one entry per model shape "
                f"(expected {shape_count}, got {self.narrow_phase.shape_aabb_lower.shape[0]})"
            )
        if self.narrow_phase.shape_aabb_upper.shape[0] != shape_count:
            raise ValueError(
                "narrow_phase.shape_aabb_upper must have one entry per model shape "
                f"(expected {shape_count}, got {self.narrow_phase.shape_aabb_upper.shape[0]})"
            )

        self.enable_rigid_soft_full_surface_contact = enable_rigid_soft_full_surface_contact
        self.full_surface_mesh_backend = full_surface_mesh_backend
        shape_types = model.shape_type.numpy() if model.shape_count > 0 else np.empty(0, dtype=np.int32)
        is_mesh = np.isin(shape_types, (int(GeoType.MESH), int(GeoType.CONVEX_MESH)))
        has_mesh_source = (
            np.array([source is not None for source in model.shape_source], dtype=bool)
            if model.shape_count > 0
            else np.empty(0, dtype=bool)
        )
        self._rigid_soft_bvh_rigid_mesh_shape_mask = np.zeros(model.shape_count, dtype=bool)

        # Full-surface SDF feature pairs and BVH mesh features are disjoint. Analytic shapes always
        # retain the exact SDF edge/face path; mesh shapes use exactly the requested backend.
        if enable_rigid_soft_full_surface_contact:
            sdf_capable = _full_surface_capable_shape_mask(model) if model.shape_count > 0 else np.empty(0, dtype=bool)
            if full_surface_mesh_backend == "sdf":
                _raise_on_unprovisioned_full_surface_meshes(model, sdf_capable)
            else:
                missing_mesh = np.flatnonzero(is_mesh & ~has_mesh_source)
                if len(missing_mesh):
                    labels = getattr(model, "shape_label", None)
                    missing = [
                        f"{labels[i] if labels is not None and i < len(labels) else f'shape_{int(i)}'} (index {int(i)})"
                        for i in missing_mesh
                    ]
                    raise ValueError(
                        "full_surface_mesh_backend='bvh' requires every mesh and convex-mesh shape "
                        f"to have a Mesh source; missing sources: {missing}"
                    )
                # Include every mesh instance in the fixed owner tables. COLLIDE_PARTICLES is
                # mutable and is therefore tested in the kernels, rather than frozen here.
                self._rigid_soft_bvh_rigid_mesh_shape_mask = is_mesh & has_mesh_source
                # Meshes move to the BVH family; the remaining SDF feature path is analytic only.
                sdf_capable = sdf_capable & ~is_mesh
            _warn_full_surface_fallbacks(model, sdf_capable | self._rigid_soft_bvh_rigid_mesh_shape_mask)
            self.soft_edge_rigid_pairs = _build_soft_edge_rigid_contact_pairs(model, sdf_capable)
            self.soft_face_rigid_pairs = _build_soft_face_rigid_contact_pairs(model, sdf_capable)
        else:
            _empty_pairs = wp.array(np.empty((0, 2), np.int32), dtype=wp.vec2i, device=model.device)
            self.soft_edge_rigid_pairs, self.soft_face_rigid_pairs = _empty_pairs, _empty_pairs

        # BVH mesh shapes obtain their soft-vertex records from the VF query, so only the remaining
        # shapes need entries in the particle-shape candidate list. This does not exclude those shapes
        # from the separate SDF edge/face passes.
        non_rigid_soft_bvh_shape_mask = ~self._rigid_soft_bvh_rigid_mesh_shape_mask
        self.soft_rigid_contact_pairs = _build_soft_particle_rigid_contact_pairs(model, non_rigid_soft_bvh_shape_mask)
        self._soft_contact_pair_count = len(self.soft_rigid_contact_pairs)
        self.rigid_soft_bvh_soft_particle_rigid_shape_pairs = _build_soft_particle_rigid_contact_pairs(
            model, self._rigid_soft_bvh_rigid_mesh_shape_mask
        )
        # These feature tables contain geometry only from rigid mesh/convex-mesh shapes selected for
        # the BVH backend; analytic rigid shapes and soft-mesh features are not included.
        # Naming: ``rigid_soft_bvh`` scopes the subsystem; the following ``rigid`` identifies the
        # side whose vertices and edges are stored.
        (
            self.rigid_soft_bvh_rigid_vertex_table,
            self.rigid_soft_bvh_rigid_vertex_position,
            self.rigid_soft_bvh_rigid_vertex_normal,
            self.rigid_soft_bvh_rigid_edge_table,
            self.rigid_soft_bvh_rigid_edge_vertex_rows,
            self.rigid_soft_bvh_rigid_edge_outward,
        ) = build_rigid_soft_bvh_rigid_feature_tables(model, self._rigid_soft_bvh_rigid_mesh_shape_mask)
        self._rigid_soft_bvh_query_seed_count = (
            len(self.rigid_soft_bvh_soft_particle_rigid_shape_pairs)
            + (len(self.rigid_soft_bvh_rigid_vertex_table) if model.tri_count > 0 else 0)
            + (len(self.rigid_soft_bvh_rigid_edge_table) if model.edge_count > 0 else 0)
        )
        self._rigid_soft_bvh_requires_soft_feature_bvhs = (
            model.tri_count > 0 and len(self.rigid_soft_bvh_rigid_vertex_table) > 0
        ) or (model.edge_count > 0 and len(self.rigid_soft_bvh_rigid_edge_table) > 0)

        if rigid_soft_bvh_candidate_max is not None and rigid_soft_bvh_candidate_max < 0:
            raise ValueError(f"rigid_soft_bvh_candidate_max must be non-negative, got {rigid_soft_bvh_candidate_max}")
        if self._rigid_soft_bvh_query_seed_count > 0:
            # Candidate storage is private to the BVH traversal. Do not include particle/SDF
            # contact capacity here: those contacts append directly to the final output buffer.
            self._rigid_soft_bvh_candidate_max = (
                4 * self._rigid_soft_bvh_query_seed_count
                if rigid_soft_bvh_candidate_max is None
                else rigid_soft_bvh_candidate_max
            )
        else:
            self._rigid_soft_bvh_candidate_max = 0

        if soft_contact_max is None:
            soft_contact_max = self.soft_contact_pair_count
            soft_contact_max += len(self.soft_edge_rigid_pairs) + len(self.soft_face_rigid_pairs)
            # Every retained BVH candidate can emit at most one final contact row.
            soft_contact_max += self._rigid_soft_bvh_candidate_max
        if soft_contact_max < 0:
            raise ValueError(f"soft_contact_max must be non-negative, got {soft_contact_max}")
        self.soft_contact_gap = soft_contact_gap
        # Soft (cloth) self-contact tuning values, populated by
        # init_soft_self_contact(); consumed at detection time like
        # soft_contact_gap (detection query radius = margin + gap; pairs
        # closer than the exclusion radius in the rest shape are skipped).
        self.soft_self_contact_margin = 0.0
        self.soft_self_contact_gap = 0.0
        self.soft_self_contact_rest_shape_exclusion_radius = 0.0
        self._soft_contact_max = soft_contact_max
        with wp.ScopedDevice(device):
            self._rigid_soft_bvh_candidate_count = wp.zeros(1, dtype=wp.int32, device=device)
            self._rigid_soft_bvh_candidates = wp.empty(
                self._rigid_soft_bvh_candidate_max, dtype=wp.vec4i, device=device
            )

        self.requires_grad = requires_grad
        self.deterministic = deterministic
        per_contact_props = self.narrow_phase.hydroelastic_sdf is not None
        if deterministic:
            with wp.ScopedDevice(device):
                self._sort_key_array = wp.zeros(rigid_contact_max, dtype=wp.int64, device=device)
            self._contact_sorter = ContactSorter(
                rigid_contact_max, per_contact_shape_properties=per_contact_props, device=device
            )
        else:
            self._sort_key_array = wp.zeros(0, dtype=wp.int64, device=device)
            self._contact_sorter = None

        self.contact_matching = contact_matching
        self._matching_enabled = matching_enabled
        self._matching_sticky = matching_sticky
        self.contact_report = contact_report
        if matching_enabled:
            self._contact_matcher = ContactMatcher(
                rigid_contact_max,
                sorter=self._contact_sorter,
                shape_world=model.shape_world,
                world_count=model.world_count,
                pos_threshold=contact_matching_pos_threshold,
                normal_dot_threshold=contact_matching_normal_dot_threshold,
                contact_report=contact_report,
                sticky=matching_sticky,
                device=device,
            )
        else:
            self._contact_matcher = None

        # One detector owns the deformable surface BVHs. BVH rigid-soft contact needs only the
        # acceleration structures; self-contact result buffers remain an independent opt-in.
        self._soft_self_contact_enabled = False
        self._soft_surface_detector: TriMeshCollisionDetector | None = None
        if self._rigid_soft_bvh_requires_soft_feature_bvhs:
            self._soft_surface_detector = TriMeshCollisionDetector(model)
        # Private compatibility alias retained for callers/tests that inspected the detector
        # before its ownership was generalized from self-contact to all soft-surface BVH users.
        self._soft_self_contact_detector: TriMeshCollisionDetector | None = None

    @property
    def rigid_contact_max(self) -> int:
        """Maximum rigid contact buffer capacity used by this pipeline."""
        return self._rigid_contact_max

    @property
    def soft_contact_max(self) -> int:
        """Maximum soft contact buffer capacity used by this pipeline."""
        return self._soft_contact_max

    @property
    def rigid_soft_bvh_requires_soft_feature_bvhs(self) -> bool:
        """Whether rigid-soft detection requires BVHs over soft triangles or edges."""
        return self._rigid_soft_bvh_requires_soft_feature_bvhs

    @property
    def rigid_soft_bvh_candidate_count(self) -> wp.array[wp.int32]:
        """Attempted BVH primitive-pair count from the latest collision pass.

        A value greater than :attr:`rigid_soft_bvh_candidate_max` indicates truncation of the
        intermediate BVH candidate stream; final soft-contact overflow remains visible through
        :attr:`Contacts.soft_contact_count` versus :attr:`Contacts.soft_contact_max`. Candidate
        slots are claimed atomically, so the retained subset is not deterministic after overflow,
        even when rigid-contact sorting is deterministic.
        """
        return self._rigid_soft_bvh_candidate_count

    @property
    def rigid_soft_bvh_candidate_max(self) -> int:
        """Capacity of the intermediate BVH primitive-pair stream."""
        return self._rigid_soft_bvh_candidate_max

    @property
    def soft_contact_margin(self) -> float:
        """Deprecated alias of :attr:`soft_contact_gap`."""
        warnings.warn(
            "CollisionPipeline.soft_contact_margin is deprecated; use soft_contact_gap.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.soft_contact_gap

    @soft_contact_margin.setter
    def soft_contact_margin(self, value: float) -> None:
        warnings.warn(
            "CollisionPipeline.soft_contact_margin is deprecated; use soft_contact_gap.",
            DeprecationWarning,
            stacklevel=2,
        )
        self.soft_contact_gap = value

    @property
    def soft_contact_pair_count(self) -> int:
        """Number of precomputed (particle, shape) pairs launched for soft contacts.

        This is the base of the default ``soft_contact_max``, which additionally reserves
        edge/face headroom when ``enable_rigid_soft_full_surface_contact`` is set.
        """
        return self._soft_contact_pair_count

    @property
    def soft_rigid_contact_pair_count(self) -> int:
        """Deprecated alias of :attr:`soft_contact_pair_count`."""
        warnings.warn(
            "CollisionPipeline.soft_rigid_contact_pair_count is deprecated; use soft_contact_pair_count.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.soft_contact_pair_count

    def contacts(self) -> Contacts:
        """
        Allocate and return a new :class:`newton.Contacts` object for this pipeline.

        The returned buffer uses this pipeline's ``requires_grad`` flag (resolved at
        construction from the argument or ``model.requires_grad``).

        Returns:
            A newly allocated contacts buffer sized for this pipeline.

        .. experimental::

            If ``requires_grad`` is true, ``rigid_contact_diff_*`` arrays may be
            allocated; rigid-contact differentiability may change without prior
            notice (see :meth:`collide`).
        """
        detector = self._soft_surface_detector
        soft_self_contact = self._soft_self_contact_enabled
        contacts = Contacts(
            self.rigid_contact_max,
            self.soft_contact_max,
            # Self-contact buffer sizing mirrors the detector configured by
            # init_soft_self_contact(); Contacts ignores it when the flag is False.
            soft_self_contact=soft_self_contact,
            particle_count=self.model.particle_count,
            tri_count=self.model.tri_count,
            edge_count=self.model.edge_count,
            soft_self_contact_vertex_buffer_pre_alloc=(
                detector.vertex_collision_buffer_pre_alloc if soft_self_contact else 0
            ),
            soft_self_contact_edge_buffer_pre_alloc=(
                detector.edge_collision_buffer_pre_alloc if soft_self_contact else 0
            ),
            soft_self_contact_triangle_buffer_pre_alloc=(
                detector.triangle_collision_buffer_pre_alloc if soft_self_contact else 0
            ),
            soft_self_contact_record_triangle_vertices=(
                detector.record_triangle_contacting_vertices if soft_self_contact else False
            ),
            # The replay array spans every particle/SDF launch thread and fixed BVH emission thread,
            # independent of output capacity, so differentiable slots never alias.
            soft_contact_tids_size=(
                self._soft_contact_pair_count
                + len(self.soft_edge_rigid_pairs)
                + len(self.soft_face_rigid_pairs)
                + self._rigid_soft_bvh_candidate_max
            ),
            requires_grad=self.requires_grad,
            device=self.model.device,
            per_contact_shape_properties=self.narrow_phase.hydroelastic_sdf is not None,
            requested_attributes=self.model.get_requested_contact_attributes(),
            contact_matching=self._matching_enabled,
            contact_report=self.contact_report,
        )
        contacts._contact_matching_mode = self.contact_matching
        # Flag the buffer so solvers that only consume particle contacts can refuse it (see
        # Contacts._enable_rigid_soft_full_surface_contact); edge/face records appear only when this is set.
        contacts._enable_rigid_soft_full_surface_contact = self.enable_rigid_soft_full_surface_contact

        # attach custom attributes with assignment==CONTACT
        self.model._add_custom_attributes(contacts, Model.AttributeAssignment.CONTACT, requires_grad=self.requires_grad)
        return contacts

    def init_soft_self_contact(
        self,
        *,
        margin: float = 0.2,
        gap: float = 0.0,
        rest_shape_exclusion_radius: float = 0.0,
        vertex_buffer_pre_alloc: int = 32,
        edge_buffer_pre_alloc: int = 64,
        triangle_buffer_pre_alloc: int = 16,
        edge_edge_parallel_epsilon: float = 1e-5,
        record_triangle_contacting_vertices: bool = False,
        topological_filter_threshold: int = 2,
        external_vertex_filter_map: dict | None = None,
        external_edge_filter_map: dict | None = None,
    ) -> None:
        """Configure soft (cloth) self-contact detection on this pipeline.

        After configuration, :meth:`contacts` allocates the self-contact result
        buffers (:attr:`Contacts.soft_self_contact_data`) on every returned
        buffer, and :meth:`collide` runs vertex-triangle and edge-edge
        detection into them when called with ``soft_self_contact=True``.

        This is the configuration entry point for standalone pipeline use; a
        solver that owns the pipeline calls this internally, seeded from its
        own self-contact parameters.

        Args:
            margin: Self-contact interaction distance [m] (surface offset at
                which force terms begin to act), consumed by solver force terms.
            gap: Additional detection-only distance [m]; detection queries use
                ``margin + gap``, mirroring the ``ShapeConfig.margin`` /
                ``ShapeConfig.gap`` convention.
            rest_shape_exclusion_radius: Pairs closer than this distance [m]
                in the rest shape (``model.particle_q``) are excluded from
                detection — for meshes whose regions are close by design
                (layered cloth, seams). ``0`` disables the filter.
            vertex_buffer_pre_alloc: Per-vertex collision buffer capacity;
                pairs beyond it are silently dropped during detection.
            edge_buffer_pre_alloc: Per-edge collision buffer capacity;
                pairs beyond it are silently dropped during detection.
            triangle_buffer_pre_alloc: Per-triangle collision buffer capacity;
                used only when ``record_triangle_contacting_vertices=True``.
            edge_edge_parallel_epsilon: Near-parallel edge-pair threshold.
            record_triangle_contacting_vertices: Also record per-triangle
                contacting vertices.
            topological_filter_threshold: Ring distance under which candidate
                pairs are filtered out.
            external_vertex_filter_map: Extra vertex-triangle exclusions.
            external_edge_filter_map: Extra edge-edge exclusions.
        """
        if margin < 0.0:
            raise ValueError(f"soft self-contact margin must be >= 0, got {margin}")
        if gap < 0.0:
            raise ValueError(f"soft self-contact gap must be >= 0, got {gap}")
        if rest_shape_exclusion_radius < 0.0:
            raise ValueError(f"rest_shape_exclusion_radius must be >= 0, got {rest_shape_exclusion_radius}")
        if self.model.tri_count == 0:
            raise ValueError("init_soft_self_contact() requires a model with triangles (cloth/soft mesh).")
        self.soft_self_contact_margin = margin
        self.soft_self_contact_gap = gap
        self.soft_self_contact_rest_shape_exclusion_radius = rest_shape_exclusion_radius
        # The explicit opt-in is what creates the detector (its BVHs are built
        # from model.particle_q); the result struct stays unallocated until the
        # first Contacts buffer is bound. Re-configuring rebuilds the detector.
        self._soft_surface_detector = TriMeshCollisionDetector(
            self.model,
            record_triangle_contacting_vertices=record_triangle_contacting_vertices,
            vertex_collision_buffer_pre_alloc=vertex_buffer_pre_alloc,
            edge_collision_buffer_pre_alloc=edge_buffer_pre_alloc,
            triangle_collision_buffer_pre_alloc=triangle_buffer_pre_alloc,
            edge_edge_parallel_epsilon=edge_edge_parallel_epsilon,
            topological_contact_filter_threshold=topological_filter_threshold,
            external_vertex_triangle_filtering_map=external_vertex_filter_map,
            external_edge_edge_filtering_map=external_edge_filter_map,
        )
        self._soft_self_contact_detector = self._soft_surface_detector
        self._soft_self_contact_enabled = True

    def set_collision_detection_range(
        self,
        *,
        soft_contact_gap: float | None = None,
        soft_self_contact_margin: float | None = None,
        soft_self_contact_gap: float | None = None,
    ) -> None:
        """Update the detection ranges consumed by :meth:`collide`.

        Only the values provided are changed; ``None`` keeps the current
        setting, and changes take effect at the next :meth:`collide` call.
        Rigid (shape-shape) ranges are per-shape model data
        (:attr:`Model.shape_margin`, :attr:`Model.shape_gap`) and are not
        covered here. A solver that owns the pipeline drives self-contact
        detection from its own parameters, so this setter affects standalone
        :meth:`collide` use.

        Args:
            soft_contact_gap: Detection-only distance [m] added to the
                per-particle radius for particle-shape contact queries.
            soft_self_contact_margin: Self-contact interaction distance [m];
                requires :meth:`init_soft_self_contact` to have been called.
            soft_self_contact_gap: Additional detection-only self-contact
                distance [m] (queries use ``margin + gap``); requires
                :meth:`init_soft_self_contact` to have been called.
        """
        for name, value in (
            ("soft_contact_gap", soft_contact_gap),
            ("soft_self_contact_margin", soft_self_contact_margin),
            ("soft_self_contact_gap", soft_self_contact_gap),
        ):
            if value is not None and value < 0.0:
                raise ValueError(f"{name} must be >= 0, got {value}")
        if soft_self_contact_margin is not None or soft_self_contact_gap is not None:
            self._ensure_soft_self_contact_detector()
        if soft_contact_gap is not None:
            self.soft_contact_gap = soft_contact_gap
        if soft_self_contact_margin is not None:
            self.soft_self_contact_margin = soft_self_contact_margin
        if soft_self_contact_gap is not None:
            self.soft_self_contact_gap = soft_self_contact_gap

    def _ensure_soft_self_contact_detector(self) -> TriMeshCollisionDetector:
        """Return the shared detector created by :meth:`init_soft_self_contact`."""
        if not self._soft_self_contact_enabled or self._soft_surface_detector is None:
            raise ValueError("configure the pipeline with init_soft_self_contact() first.")
        return self._soft_surface_detector

    def _ensure_soft_surface_detector(self) -> TriMeshCollisionDetector:
        """Return the detector that owns the deformable surface BVHs."""
        if self._soft_surface_detector is None:
            raise ValueError(
                "This pipeline has no soft-surface BVH; enable BVH full-surface contact or "
                "configure soft self-contact first."
            )
        return self._soft_surface_detector

    def _get_soft_self_contact_detector(self, contacts: Contacts) -> TriMeshCollisionDetector:
        """Return the shared detector re-pointed at ``contacts.soft_self_contact_data``."""
        data = contacts.soft_self_contact_data
        if data is None:
            raise ValueError(
                "This Contacts buffer has no soft_self_contact_data; allocate it with "
                "CollisionPipeline.contacts() after init_soft_self_contact()."
            )
        detector = self._ensure_soft_self_contact_detector()
        if detector.collision_info is not data:
            detector._bind_external_buffers(data)
        return detector

    @staticmethod
    def _soft_surface_positions(state_or_positions: State | wp.array) -> wp.array:
        return state_or_positions.particle_q if isinstance(state_or_positions, State) else state_or_positions

    def refit_soft_surface_bvh(self, state_or_positions: State | wp.array[wp.vec3]) -> None:
        """Refit the shared soft triangle/edge BVHs to current particle positions.

        Keeping the BVHs up to date is the caller's responsibility:
        :meth:`collide` never updates them. Call this before a collision pass
        after particle positions change. An owning solver does this at each
        scheduled rigid-soft BVH detection point.

        Args:
            state_or_positions: A :class:`State` or its particle position array.
        """
        detector = self._ensure_soft_surface_detector()
        positions = self._soft_surface_positions(state_or_positions)
        # Warp's grouped CPU BVH refit is not currently reliable; rebuilding preserves the
        # public freshness contract on CPU. CUDA uses the intended cheaper refit path.
        if self.device.is_cpu:
            detector.rebuild(positions)
        else:
            detector.refit(positions)

    def rebuild_soft_surface_bvh(self, state_or_positions: State | wp.array[wp.vec3]) -> None:
        """Fully rebuild the shared soft triangle/edge BVHs from current positions."""
        self._ensure_soft_surface_detector().rebuild(self._soft_surface_positions(state_or_positions))

    def refit_soft_self_contact_bvh(self, new_pos: wp.array[wp.vec3], rebuild: bool = False) -> None:
        """Deprecated alias for the shared soft-surface BVH lifecycle methods."""
        warnings.warn(
            "refit_soft_self_contact_bvh() is deprecated; use refit_soft_surface_bvh() or rebuild_soft_surface_bvh().",
            DeprecationWarning,
            stacklevel=2,
        )
        if rebuild:
            self.rebuild_soft_surface_bvh(new_pos)
        else:
            self.refit_soft_surface_bvh(new_pos)

    def reset_contact_matching(self, world_mask: wp.array[wp.bool] | None = None) -> None:
        """Clear all or reset-selected previous-frame contact history.

        Masked selections accumulate until the next :meth:`collide` call
        consumes them.

        .. experimental::

        Args:
            world_mask: Optional one-dimensional Warp boolean mask on the
                model device with shape ``(model.world_count + 1,)``. The final
                entry selects global entities whose world index is ``-1``. If
                ``None``, clear all previous-frame contact history immediately.
        """
        world_mask = normalize_reset_world_mask(
            world_mask,
            world_count=int(self.model.world_count),
            device=self.model.device,
        )
        if self._contact_matcher is not None:
            self._contact_matcher.reset(world_mask)

    @staticmethod
    def _build_excluded_pairs(model: Model) -> wp.array[wp.vec2i] | None:
        sorted_pairs = model.shape_collision_filter_pairs_array()
        if sorted_pairs.shape[0] == 0:
            return None
        return wp.array(
            sorted_pairs,
            dtype=wp.vec2i,
            device=model.device,
        )

    def collide(
        self,
        state: State,
        contacts: Contacts,
        *,
        soft_contact_margin: float | None = None,
        soft_self_contact: bool = False,
    ):
        """Run the collision pipeline using NarrowPhase.

        Safe to call inside a :class:`wp.Tape` context.  The non-differentiable
        broad-phase and narrow-phase kernels are launched with tape recording
        hardcoded ``record_tape=False`` internally.  The differentiable kernels
        (soft-contact generation and rigid-contact augmentation) are recorded on
        the tape so that gradients flow through ``state.body_q`` and
        ``state.particle_q``.

        When ``requires_grad=True``, the differentiable rigid-contact arrays
        (``contacts.rigid_contact_diff_*``) are populated by a lightweight
        augmentation kernel that reconstructs world-space contact points from
        the frozen narrow-phase output through the body transforms.

        .. experimental::

            This rigid-contact gradient path may change without prior notice.
            Usefulness and numerical behaviour are still being assessed across
            real-world scenarios.

        Args:
            state: The current simulation state.
            contacts: The contacts buffer to populate (will be cleared first).
            soft_contact_margin: Deprecated; set ``soft_contact_gap`` on the
                :class:`CollisionPipeline` constructor instead. When not
                ``None``, the value is still honored for this call and a
                :class:`DeprecationWarning` is emitted.
            soft_self_contact: Also run soft (cloth) self-contact detection
                into ``contacts.soft_self_contact_data``. Requires
                :meth:`init_soft_self_contact` to have been called. The
                self-contact BVHs are **not** updated by this call — keep them
                current via :meth:`refit_soft_surface_bvh`.
        """
        # Keep the buffer's full-surface capability marker in sync with this pipeline on every call.
        # collide() may be handed a Contacts created elsewhere (or by a flag-off pipeline); the edge/
        # face passes below would otherwise populate records while the marker stayed False, so
        # particle-only solvers (XPBD, semi-implicit, Style3D) would not raise and would silently
        # ignore them. Mirrors the assignment in CollisionPipeline.contacts().
        contacts._enable_rigid_soft_full_surface_contact = self.enable_rigid_soft_full_surface_contact

        # Counter zeroing and generation bump are fused into compute_shape_aabbs.
        # Only call contacts.clear() if clear_buffers mode is enabled (debug path).
        # Skip the generation bump here since compute_shape_aabbs will bump it immediately
        # afterwards -- otherwise the generation would advance by 2 per collide() call.
        if contacts.clear_buffers:
            contacts.clear(bump_generation=False)

        model = self.model
        # update any additional parameters
        if soft_contact_margin is not None:
            warnings.warn(
                "The soft_contact_margin argument of CollisionPipeline.collide() is deprecated; "
                "set soft_contact_gap on the CollisionPipeline constructor instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            soft_contact_gap = soft_contact_margin
        else:
            soft_contact_gap = self.soft_contact_gap

        # Rigid contact detection -- broad phase + narrow phase.
        # These kernels hardcode record_tape=False internally so they are
        # never captured on an active wp.Tape.  The differentiable
        # augmentation and soft-contact kernels that follow are tape-safe
        # and recorded normally.

        # Compute AABBs for all shapes, zero counters, bump generation.
        # Fuses contacts.clear() + broad_phase_pair_count.zero_() + AABB update.
        wp.launch(
            kernel=compute_shape_aabbs,
            dim=model.shape_count,
            inputs=[
                state.body_q,
                model.shape_transform,
                model.shape_body,
                model.shape_type,
                model.shape_scale,
                model.shape_collision_radius,
                model.shape_source_ptr,
                model.shape_margin,
                model.shape_gap,
                model.shape_collision_aabb_lower,
                model.shape_collision_aabb_upper,
                contacts.contact_counters,
                contacts.contact_generation,
                self.broad_phase_pair_count,
                contacts.contact_counters.shape[0],
            ],
            outputs=[
                self.narrow_phase.shape_aabb_lower,
                self.narrow_phase.shape_aabb_upper,
                self.geom_data,
                self.geom_transform,
            ],
            device=self.device,
            record_tape=False,
        )

        # Run broad phase (AABBs are already expanded by effective gaps, so pass None)
        if isinstance(self.broad_phase, BroadPhaseAllPairs):
            self.broad_phase.launch(
                self.narrow_phase.shape_aabb_lower,
                self.narrow_phase.shape_aabb_upper,
                None,  # AABBs are pre-expanded, no additional margin needed
                model.shape_collision_group,
                model.shape_world,
                model.shape_count,
                self.broad_phase_shape_pairs,
                self.broad_phase_pair_count,
                shape_body=model.shape_body,
                body_flags=model.body_flags,
                include_static_kinematic_pairs=self.include_static_kinematic_pairs,
                device=self.device,
                filter_pairs=self.shape_pairs_excluded,
                num_filter_pairs=self.shape_pairs_excluded_count,
                skip_count_zero=True,  # Already zeroed by compute_shape_aabbs
            )
        elif isinstance(self.broad_phase, BroadPhaseSAP):
            self.broad_phase.launch(
                self.narrow_phase.shape_aabb_lower,
                self.narrow_phase.shape_aabb_upper,
                None,  # AABBs are pre-expanded, no additional margin needed
                model.shape_collision_group,
                model.shape_world,
                model.shape_count,
                self.broad_phase_shape_pairs,
                self.broad_phase_pair_count,
                shape_body=model.shape_body,
                body_flags=model.body_flags,
                include_static_kinematic_pairs=self.include_static_kinematic_pairs,
                device=self.device,
                filter_pairs=self.shape_pairs_excluded,
                num_filter_pairs=self.shape_pairs_excluded_count,
                skip_count_zero=True,  # Already zeroed by compute_shape_aabbs
            )
        else:  # BroadPhaseExplicit
            self.broad_phase.launch(
                self.narrow_phase.shape_aabb_lower,
                self.narrow_phase.shape_aabb_upper,
                None,  # AABBs are pre-expanded, no additional margin needed
                self.shape_pairs_filtered,
                len(self.shape_pairs_filtered),
                self.broad_phase_shape_pairs,
                self.broad_phase_pair_count,
                shape_body=model.shape_body,
                body_flags=model.body_flags,
                include_static_kinematic_pairs=self.include_static_kinematic_pairs,
                device=self.device,
                skip_count_zero=True,  # Already zeroed by compute_shape_aabbs
            )

        # Create ContactWriterData struct for custom contact writing
        writer_data = ContactWriterData()
        writer_data.contact_max = contacts.rigid_contact_max
        writer_data.body_q = state.body_q
        writer_data.shape_body = model.shape_body
        writer_data.shape_gap = model.shape_gap
        writer_data.contact_count = contacts.rigid_contact_count
        writer_data.out_shape0 = contacts.rigid_contact_shape0
        writer_data.out_shape1 = contacts.rigid_contact_shape1
        writer_data.out_point0 = contacts.rigid_contact_point0
        writer_data.out_point1 = contacts.rigid_contact_point1
        writer_data.out_offset0 = contacts.rigid_contact_offset0
        writer_data.out_offset1 = contacts.rigid_contact_offset1
        writer_data.out_normal = contacts.rigid_contact_normal
        writer_data.out_margin0 = contacts.rigid_contact_margin0
        writer_data.out_margin1 = contacts.rigid_contact_margin1
        writer_data.out_tids = contacts.rigid_contact_tids

        writer_data.out_stiffness = contacts.rigid_contact_stiffness
        writer_data.out_damping = contacts.rigid_contact_damping
        writer_data.out_friction = contacts.rigid_contact_friction
        if self.deterministic and contacts.rigid_contact_max != self._sort_key_array.shape[0]:
            raise ValueError(
                f"Contacts buffer capacity ({contacts.rigid_contact_max}) does not match the "
                f"deterministic sort buffer size ({self._sort_key_array.shape[0]}). "
                f"The sorter operates over fixed-capacity buffers for CUDA graph capture "
                f"compatibility, so the sizes must match exactly. Use CollisionPipeline.contacts() "
                f"or pass matching rigid_contact_max."
            )
        writer_data.out_sort_key = self._sort_key_array
        # Run narrow phase with custom contact writer (writes directly to Contacts format)
        self.narrow_phase.launch_custom_write(
            candidate_pair=self.broad_phase_shape_pairs,
            candidate_pair_count=self.broad_phase_pair_count,
            shape_types=model.shape_type,
            shape_data=self.geom_data,
            shape_transform=self.geom_transform,
            shape_source=model.shape_source_ptr,
            shape_mesh_properties=model._shape_mesh_properties,
            shape_sdf_index=model._shape_sdf_index,
            texture_sdf_data=model._texture_sdf_data,
            shape_gap=model.shape_gap,
            shape_collision_radius=model.shape_collision_radius,
            shape_flags=model.shape_flags,
            shape_collision_aabb_lower=model.shape_collision_aabb_lower,
            shape_collision_aabb_upper=model.shape_collision_aabb_upper,
            shape_voxel_resolution=self.narrow_phase.shape_voxel_resolution,
            shape_heightfield_index=model.shape_heightfield_index,
            heightfield_data=model.heightfield_data,
            heightfield_elevations=model.heightfield_elevations,
            mesh_edge_indices=model.mesh_edge_indices,
            mesh_edge_centers=model.mesh_edge_centers,
            mesh_edge_halves=model.mesh_edge_halves,
            shape_edge_range=model.shape_edge_range,
            writer_data=writer_data,
            device=self.device,
        )

        # Match contacts against previous frame before sorting.
        if self._contact_matcher is not None:
            if contacts.rigid_contact_match_index is None:
                raise ValueError(
                    "CollisionPipeline has contact_matching enabled but the "
                    "Contacts buffer was created without contact_matching. "
                    "Use pipeline.contacts() to create a compatible buffer."
                )
            self._contact_matcher.match(
                sort_keys=self._sort_key_array,
                contact_count=contacts.rigid_contact_count,
                point0=contacts.rigid_contact_point0,
                point1=contacts.rigid_contact_point1,
                shape0=contacts.rigid_contact_shape0,
                shape1=contacts.rigid_contact_shape1,
                normal=contacts.rigid_contact_normal,
                body_q=state.body_q,
                shape_body=model.shape_body,
                match_index_out=contacts.rigid_contact_match_index,
                device=self.device,
            )

        if self.deterministic and self._contact_sorter is not None:
            self._contact_sorter.sort_full(
                self._sort_key_array,
                contacts.rigid_contact_count,
                shape0=contacts.rigid_contact_shape0,
                shape1=contacts.rigid_contact_shape1,
                point0=contacts.rigid_contact_point0,
                point1=contacts.rigid_contact_point1,
                offset0=contacts.rigid_contact_offset0,
                offset1=contacts.rigid_contact_offset1,
                normal=contacts.rigid_contact_normal,
                margin0=contacts.rigid_contact_margin0,
                margin1=contacts.rigid_contact_margin1,
                tids=contacts.rigid_contact_tids,
                stiffness=contacts.rigid_contact_stiffness,
                damping=contacts.rigid_contact_damping,
                friction=contacts.rigid_contact_friction,
                match_index=contacts.rigid_contact_match_index,
                device=self.device,
            )

        # Sticky mode: overwrite matched rows with the saved previous-frame
        # contact geometry.  Must run after sort_full (so match_index points at
        # the sorted prev-frame layout *and* we target the final sorted rows)
        # and before save_sorted_state (we save the record we actually used
        # this frame, carrying the sticky history forward).
        if self._matching_sticky:
            self._contact_matcher.replay_matched(
                contact_count=contacts.rigid_contact_count,
                match_index=contacts.rigid_contact_match_index,
                point0=contacts.rigid_contact_point0,
                point1=contacts.rigid_contact_point1,
                offset0=contacts.rigid_contact_offset0,
                offset1=contacts.rigid_contact_offset1,
                normal=contacts.rigid_contact_normal,
                shape0=contacts.rigid_contact_shape0,
                shape1=contacts.rigid_contact_shape1,
                margin0=contacts.rigid_contact_margin0,
                margin1=contacts.rigid_contact_margin1,
                body_q=state.body_q,
                shape_body=writer_data.shape_body,
                device=self.device,
            )

        # Build the contact report before saving state, because save
        # overwrites _prev_count and the report needs the old value.
        if self._contact_matcher is not None:
            if self._contact_matcher.has_report:
                if contacts.rigid_contact_new_indices is None:
                    raise ValueError(
                        "CollisionPipeline has contact_report enabled but the Contacts "
                        "buffer was created without contact_report=True. "
                        "Use pipeline.contacts() to create a compatible buffer."
                    )
                self._contact_matcher.build_report(
                    contacts.rigid_contact_match_index,
                    contacts.rigid_contact_count,
                    contacts.rigid_contact_new_indices,
                    contacts.rigid_contact_new_count,
                    contacts.rigid_contact_broken_indices,
                    contacts.rigid_contact_broken_count,
                    device=self.device,
                )
            sticky_offsets: dict[str, wp.array] = (
                {
                    "sorted_offset0": contacts.rigid_contact_offset0,
                    "sorted_offset1": contacts.rigid_contact_offset1,
                }
                if self._matching_sticky
                else {}
            )
            self._contact_matcher.save_sorted_state(
                sorted_keys=self._contact_sorter.sorted_keys_view,
                contact_count=contacts.rigid_contact_count,
                sorted_point0=contacts.rigid_contact_point0,
                sorted_point1=contacts.rigid_contact_point1,
                sorted_shape0=contacts.rigid_contact_shape0,
                sorted_shape1=contacts.rigid_contact_shape1,
                sorted_normal=contacts.rigid_contact_normal,
                body_q=state.body_q,
                shape_body=model.shape_body,
                device=self.device,
                **sticky_offsets,
            )

        # Differentiable contact augmentation: reconstruct world-space contact
        # quantities through body_q so that gradients flow via wp.Tape.
        if self.requires_grad and contacts.rigid_contact_diff_distance is not None:
            launch_differentiable_contact_augment(
                contacts=contacts,
                body_q=state.body_q,
                shape_body=model.shape_body,
                device=self.device,
            )

        # Generate soft contacts for particles and shapes
        if state.particle_q and self.soft_contact_max > 0 and self.soft_contact_pair_count > 0:
            wp.launch(
                kernel=create_soft_contacts,
                dim=self.soft_contact_pair_count,
                inputs=[
                    self.soft_rigid_contact_pairs,
                    state.particle_q,
                    model.particle_radius,
                    model.particle_flags,
                    model.particle_world,
                    state.body_q,
                    model.shape_transform,
                    model.shape_body,
                    model.shape_type,
                    model.shape_scale,
                    model.shape_source_ptr,
                    model._shape_mesh_properties,
                    model.shape_world,
                    soft_contact_gap,
                    model.shape_margin,
                    self.soft_contact_max,
                    model.shape_flags,
                    model.shape_heightfield_index,
                    model.heightfield_data,
                    model.heightfield_elevations,
                ],
                outputs=[
                    contacts.soft_contact_count,
                    contacts.soft_contact_particle,
                    contacts.soft_contact_indices,
                    contacts.soft_contact_barycentric,
                    contacts.soft_contact_shape,
                    contacts.soft_contact_body_pos,
                    contacts.soft_contact_body_vel,
                    contacts.soft_contact_normal,
                    contacts.soft_contact_tids,
                ],
                device=self.device,
            )

        # Full-surface EDGE/FACE passes (opt-in, set at construction): add the soft edge/face contacts
        # the per-particle path cannot detect. Run after the particle-contact launch on the same stream;
        # the particle records therefore occupy [0, particle_count) and the edge/face records append.
        # The flag is fixed at construction because soft_contact_max headroom is sized there.
        if self.enable_rigid_soft_full_surface_contact and state.particle_q:
            launch_soft_ef_contacts(
                model=model,
                state=state,
                contacts=contacts,
                margin=soft_contact_gap,
                device=self.device,
                edge_pairs=self.soft_edge_rigid_pairs,
                face_pairs=self.soft_face_rigid_pairs,
                n_particle_pairs=self.soft_contact_pair_count,
            )
            if self._rigid_soft_bvh_query_seed_count > 0:
                launch_rigid_soft_bvh_contacts(
                    model=model,
                    state=state,
                    contacts=contacts,
                    detector=self._soft_surface_detector,
                    particle_shape_pairs=self.rigid_soft_bvh_soft_particle_rigid_shape_pairs,
                    rigid_vertex_table=self.rigid_soft_bvh_rigid_vertex_table,
                    rigid_vertex_position=self.rigid_soft_bvh_rigid_vertex_position,
                    rigid_vertex_normal=self.rigid_soft_bvh_rigid_vertex_normal,
                    rigid_edge_table=self.rigid_soft_bvh_rigid_edge_table,
                    rigid_edge_vertex_rows=self.rigid_soft_bvh_rigid_edge_vertex_rows,
                    rigid_edge_outward=self.rigid_soft_bvh_rigid_edge_outward,
                    candidate_count=self._rigid_soft_bvh_candidate_count,
                    candidates=self._rigid_soft_bvh_candidates,
                    candidate_max=self._rigid_soft_bvh_candidate_max,
                    gap=soft_contact_gap,
                    tid_base=(
                        self.soft_contact_pair_count + len(self.soft_edge_rigid_pairs) + len(self.soft_face_rigid_pairs)
                    ),
                    edge_edge_parallel_epsilon=(
                        self._soft_surface_detector.edge_edge_parallel_epsilon
                        if self._soft_surface_detector is not None
                        else 1.0e-5
                    ),
                    device=self.device,
                )

        # Preserve the previous provenance if validation or collision setup fails.
        contacts._contact_matching_mode = self.contact_matching

        # Soft (cloth) self-contact detection (opt-in per call; results land in
        # contacts.soft_self_contact_data).
        if soft_self_contact:
            detector = self._get_soft_self_contact_detector(contacts)
            query_radius = self.soft_self_contact_margin + self.soft_self_contact_gap
            # The BVHs (and the positions detection reads) are NOT updated here —
            # keeping them current via refit_soft_surface_bvh() is
            # the caller's responsibility. Rest-shape exclusion measures pair
            # distances in the model's initial (rest) positions.
            detector.vertex_triangle_collision_detection(
                query_radius,
                min_query_radius=self.soft_self_contact_rest_shape_exclusion_radius,
                min_distance_filtering_ref_pos=self.model.particle_q,
            )
            detector.edge_edge_collision_detection(
                query_radius,
                min_query_radius=self.soft_self_contact_rest_shape_exclusion_radius,
                min_distance_filtering_ref_pos=self.model.particle_q,
            )
