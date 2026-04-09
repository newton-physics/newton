# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Literal

import numpy as np
import warp as wp

from ..geometry.broad_phase_nxn import BroadPhaseAllPairs, BroadPhaseExplicit
from ..geometry.broad_phase_sap import BroadPhaseSAP
from ..geometry.collision_core import compute_tight_aabb_from_support
from ..geometry.contact_data import ContactData
from ..geometry.differentiable_contacts import launch_differentiable_contact_augment
from ..geometry.flags import ShapeFlags
from ..geometry.kernels import create_soft_contacts
from ..geometry.narrow_phase import NarrowPhase
from ..geometry.sdf_hydroelastic import HydroelasticSDF
from ..geometry.support_function import (
    GenericShapeData,
    SupportMapDataProvider,
    pack_mesh_ptr,
)
from ..geometry.types import GeoType
from ..sim.contacts import Contacts
from ..sim.model import Model
from ..sim.state import State


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
    # outputs
    aabb_lower: wp.array[wp.vec3],
    aabb_upper: wp.array[wp.vec3],
):
    """Compute axis-aligned bounding boxes for each shape in world space.

    Uses support function for most shapes. Meshes and heightfields use the pre-computed
    local AABB transformed to world frame. Infinite planes use bounding sphere fallback.
    AABBs are enlarged by per-shape effective gap for contact detection.
    Effective expansion is ``shape_margin + shape_gap``.
    """
    shape_id = wp.tid()

    rigid_id = shape_body[shape_id]
    geo_type = shape_type[shape_id]

    # Compute world transform
    if rigid_id == -1:
        X_ws = shape_transform[shape_id]
    else:
        X_ws = wp.transform_multiply(body_q[rigid_id], shape_transform[shape_id])

    pos = wp.transform_get_translation(X_ws)
    orientation = wp.transform_get_rotation(X_ws)

    # Enlarge AABB by per-shape effective gap for contact detection
    effective_gap = shape_margin[shape_id] + shape_gap[shape_id]
    margin_vec = wp.vec3(effective_gap, effective_gap, effective_gap)

    # Check if this is an infinite plane, mesh, or heightfield
    scale = shape_scale[shape_id]
    is_infinite_plane = (geo_type == GeoType.PLANE) and (scale[0] == 0.0 and scale[1] == 0.0)
    is_mesh = geo_type == GeoType.MESH
    is_hfield = geo_type == GeoType.HFIELD

    if is_infinite_plane:
        # Bounding sphere fallback for infinite planes
        radius = shape_collision_radius[shape_id]
        half_extents = wp.vec3(radius, radius, radius)
        aabb_lower[shape_id] = pos - half_extents - margin_vec
        aabb_upper[shape_id] = pos + half_extents + margin_vec
    elif is_mesh or is_hfield:
        # Tight local AABB transformed to world space.
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
        shape_data.scale = scale
        shape_data.auxiliary = wp.vec3(0.0, 0.0, 0.0)

        # For CONVEX_MESH, pack the mesh pointer
        if geo_type == GeoType.CONVEX_MESH:
            shape_data.auxiliary = pack_mesh_ptr(shape_source_ptr[shape_id])

        data_provider = SupportMapDataProvider()

        # Compute tight AABB using helper function
        aabb_min_world, aabb_max_world = compute_tight_aabb_from_support(shape_data, orientation, pos, data_provider)

        aabb_lower[shape_id] = aabb_min_world - margin_vec
        aabb_upper[shape_id] = aabb_max_world + margin_vec


@wp.kernel(enable_backward=False)
def prepare_geom_data_kernel(
    shape_transform: wp.array[wp.transform],
    shape_body: wp.array[int],
    shape_type: wp.array[int],
    shape_scale: wp.array[wp.vec3],
    shape_margin: wp.array[float],
    body_q: wp.array[wp.transform],
    # Outputs
    geom_data: wp.array[wp.vec4],  # scale xyz, margin w
    geom_transform: wp.array[wp.transform],  # world space transform
):
    """Prepare geometry data arrays for NarrowPhase API."""
    idx = wp.tid()

    # Pack scale and margin into geom_data
    scale = shape_scale[idx]
    margin = shape_margin[idx]
    geom_data[idx] = wp.vec4(scale[0], scale[1], scale[2], margin)

    # Compute world space transform
    body_idx = shape_body[idx]
    if body_idx >= 0:
        geom_transform[idx] = wp.transform_multiply(body_q[body_idx], shape_transform[idx])
    else:
        geom_transform[idx] = shape_transform[idx]


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

    # Primitive pairs (GJK/MPR) produce up to 5 manifold contacts.
    # Mesh-involved pairs (SDF + contact reduction) typically retain ~40.
    PRIMITIVE_CPP = 5
    MESH_CPP = 40
    MAX_NEIGHBORS_PER_SHAPE = 20

    mesh_mask = (shape_types == int(GeoType.MESH)) | (shape_types == int(GeoType.HFIELD))
    plane_mask = shape_types == int(GeoType.PLANE)
    non_plane_mask = ~plane_mask
    num_meshes = int(np.count_nonzero(mesh_mask))
    num_non_planes = int(np.count_nonzero(non_plane_mask))
    num_primitives = num_non_planes - num_meshes
    num_planes = int(np.count_nonzero(plane_mask))

    # Weighted contacts from non-plane shape types.
    # Each shape's neighbor pairs are weighted by its type's contacts-per-pair.
    # Divide by 2 to avoid double-counting pairs.
    non_plane_contacts = (
        num_primitives * MAX_NEIGHBORS_PER_SHAPE * PRIMITIVE_CPP + num_meshes * MAX_NEIGHBORS_PER_SHAPE * MESH_CPP
    ) // 2

    # Weighted average contacts-per-pair based on the scene's shape mix.
    avg_cpp = (
        (num_primitives * PRIMITIVE_CPP + num_meshes * MESH_CPP) // max(num_non_planes, 1) if num_non_planes > 0 else 0
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
            plane_contacts = num_planes * (num_primitives * PRIMITIVE_CPP + num_meshes * MESH_CPP)

    total_contacts = non_plane_contacts + plane_contacts

    # When precomputed contact pairs are available, use as a tighter bound.
    if hasattr(model, "shape_contact_pair_count") and model.shape_contact_pair_count > 0:
        weighted_cpp = max(avg_cpp, PRIMITIVE_CPP)
        pair_contacts = int(model.shape_contact_pair_count) * weighted_cpp
        total_contacts = min(total_contacts, pair_contacts)

    # Ensure minimum allocation
    return max(1000, total_contacts)


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


class CollisionPipeline:
    """
    Full-featured collision pipeline with GJK/MPR narrow phase and pluggable broad phase.

    Key features:
        - GJK/MPR algorithms for convex-convex collision detection
        - Multiple broad phase options: NXN (all-pairs), SAP (sweep-and-prune), EXPLICIT (precomputed pairs)
        - Mesh-mesh collision via SDF with contact reduction
        - Optional hydroelastic contact model for compliant surfaces

    For most users, construct with ``CollisionPipeline(model, ...)``.

    .. note::
        Differentiable rigid contacts (the ``rigid_contact_diff_*`` arrays when
        ``requires_grad`` is enabled) are **experimental**. The narrow phase stays
        frozen and gradients are a tangent approximation; validate accuracy and
        usefulness on your workflow before relying on them in optimization loops.
    """

    def __init__(
        self,
        model: Model,
        *,
        reduce_contacts: bool = True,
        rigid_contact_max: int | None = None,
        max_triangle_pairs: int = 1000000,
        shape_pairs_filtered: wp.array[wp.vec2i] | None = None,
        soft_contact_max: int | None = None,
        soft_contact_margin: float = 0.01,
        requires_grad: bool | None = None,
        broad_phase: Literal["nxn", "sap", "explicit"]
        | BroadPhaseAllPairs
        | BroadPhaseSAP
        | BroadPhaseExplicit
        | None = None,
        narrow_phase: NarrowPhase | None = None,
        sdf_hydroelastic_config: HydroelasticSDF.Config | None = None,
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
            soft_contact_max: Maximum number of soft contacts to allocate.
                If None, computed as shape_count * particle_count.
            soft_contact_margin: Margin for soft contact generation. Defaults to 0.01.
            requires_grad: Whether to enable gradient computation. If None, uses model.requires_grad.
            broad_phase:
                Either a broad phase mode string ("explicit", "nxn", "sap") or
                a prebuilt broad phase instance for expert usage.
            narrow_phase: Optional prebuilt narrow phase instance. Must be
                provided together with a broad phase instance for expert usage.
            shape_pairs_filtered: Precomputed shape pairs for EXPLICIT mode.
                When broad_phase is "explicit", uses model.shape_contact_pairs if not provided. For
                "nxn"/"sap" modes, ignored.
            sdf_hydroelastic_config: Configuration for
                hydroelastic collision handling. Defaults to None.

        .. note::
            When ``requires_grad`` is true (explicitly or via ``model.requires_grad``),
            rigid-contact autodiff via ``rigid_contact_diff_*`` is **experimental**;
            see :meth:`collide`.
        """
        mode_from_broad_phase: str | None = None
        broad_phase_instance: BroadPhaseAllPairs | BroadPhaseSAP | BroadPhaseExplicit | None = None
        if broad_phase is not None:
            if isinstance(broad_phase, str):
                mode_from_broad_phase = _normalize_broad_phase_mode(broad_phase)
            else:
                broad_phase_instance = broad_phase

        shape_count = model.shape_count
        particle_count = model.particle_count
        device = model.device

        # Resolve rigid contact capacity with explicit > model > estimated precedence.
        if rigid_contact_max is None:
            model_rigid_contact_max = int(getattr(model, "rigid_contact_max", 0) or 0)
            if model_rigid_contact_max > 0:
                rigid_contact_max = model_rigid_contact_max
            else:
                rigid_contact_max = _estimate_rigid_contact_max(model)
        self._rigid_contact_max = rigid_contact_max
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
        self.soft_contact_margin = soft_contact_margin

        using_expert_components = broad_phase_instance is not None or narrow_phase is not None
        if using_expert_components:
            if broad_phase_instance is None or narrow_phase is None:
                raise ValueError("Provide both broad_phase and narrow_phase for expert component construction")
            if sdf_hydroelastic_config is not None:
                raise ValueError("sdf_hydroelastic_config cannot be used when narrow_phase is provided")

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
                self.shape_pairs_max = _compute_per_world_shape_pairs_max(model)
                self.shape_pairs_excluded = self._build_excluded_pairs(model)
                self.shape_pairs_excluded_count = (
                    self.shape_pairs_excluded.shape[0] if self.shape_pairs_excluded is not None else 0
                )
            elif self.broad_phase_mode == "sap":
                if shape_world is None:
                    raise ValueError("model.shape_world is required for broad_phase=SAP")
                self.broad_phase = BroadPhaseSAP(shape_world, shape_flags=shape_flags, device=device)
                self.shape_pairs_filtered = None
                self.shape_pairs_max = _compute_per_world_shape_pairs_max(model)
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
            )

            # Detect shape classes to optimize narrow-phase kernel launches.
            # Keep mesh and heightfield flags independent: heightfield-only scenes
            # should not trigger mesh-only kernel setup/launches.
            has_meshes = False
            has_heightfields = False
            use_lean_gjk_mpr = False
            if hasattr(model, "shape_type") and model.shape_type is not None:
                shape_types = model.shape_type.numpy()
                has_heightfields = bool((shape_types == int(GeoType.HFIELD)).any())
                has_meshes = bool((shape_types == int(GeoType.MESH)).any())
                # Use lean GJK/MPR kernel when scene has no capsules, ellipsoids,
                # cylinders, or cones (which need full support function and axial
                # rolling post-processing)
                lean_unsupported = {
                    int(GeoType.CAPSULE),
                    int(GeoType.ELLIPSOID),
                    int(GeoType.CYLINDER),
                    int(GeoType.CONE),
                }
                use_lean_gjk_mpr = not bool(lean_unsupported & set(shape_types.tolist()))

            # Initialize narrow phase with pre-allocated buffers
            # max_triangle_pairs is a conservative estimate for mesh collision triangle pairs
            # Pass write_contact as custom writer to write directly to final Contacts format
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
                has_heightfields=has_heightfields,
                use_lean_gjk_mpr=use_lean_gjk_mpr,
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

        if soft_contact_max is None:
            soft_contact_max = shape_count * particle_count
        self.soft_contact_margin = soft_contact_margin
        self._soft_contact_max = soft_contact_max
        self.requires_grad = requires_grad

    @property
    def rigid_contact_max(self) -> int:
        """Maximum rigid contact buffer capacity used by this pipeline."""
        return self._rigid_contact_max

    @property
    def soft_contact_max(self) -> int:
        """Maximum soft contact buffer capacity used by this pipeline."""
        return self._soft_contact_max

    def contacts(self) -> Contacts:
        """
        Allocate and return a new :class:`newton.Contacts` object for this pipeline.

        The returned buffer uses this pipeline's ``requires_grad`` flag (resolved at
        construction from the argument or ``model.requires_grad``).

        Returns:
            A newly allocated contacts buffer sized for this pipeline.

        .. note::
            If ``requires_grad`` is true, ``rigid_contact_diff_*`` arrays may be
            allocated; rigid-contact differentiability is **experimental** (see
            :meth:`collide`).
        """
        contacts = Contacts(
            self.rigid_contact_max,
            self.soft_contact_max,
            requires_grad=self.requires_grad,
            device=self.model.device,
            per_contact_shape_properties=self.narrow_phase.hydroelastic_sdf is not None,
            requested_attributes=self.model.get_requested_contact_attributes(),
        )

        # attach custom attributes with assignment==CONTACT
        self.model._add_custom_attributes(contacts, Model.AttributeAssignment.CONTACT, requires_grad=self.requires_grad)
        return contacts

    @staticmethod
    def _build_excluded_pairs(model: Model) -> wp.array[wp.vec2i] | None:
        if not hasattr(model, "shape_collision_filter_pairs"):
            return None
        filters = model.shape_collision_filter_pairs
        if not filters:
            return None
        sorted_pairs = sorted(filters)  # lexicographic (already canonical min,max)
        return wp.array(
            np.array(sorted_pairs),
            dtype=wp.vec2i,
            device=model.device,
        )

    def collide(
        self,
        state: State,
        contacts: Contacts,
        *,
        soft_contact_margin: float | None = None,
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

        .. note::
            This rigid-contact gradient path is **experimental**: usefulness and
            numerical behaviour are still being assessed across real-world scenarios.

        Args:
            state: The current simulation state.
            contacts: The contacts buffer to populate (will be cleared first).
            soft_contact_margin: Margin for soft contact generation.
                If ``None``, uses the value from construction.
        """

        contacts.clear()
        # TODO: validate contacts dimensions & compatibility

        # Clear counters
        self.broad_phase_pair_count.zero_()

        model = self.model
        # update any additional parameters
        soft_contact_margin = soft_contact_margin if soft_contact_margin is not None else self.soft_contact_margin

        # Rigid contact detection -- broad phase + narrow phase.
        # These kernels hardcode record_tape=False internally so they are
        # never captured on an active wp.Tape.  The differentiable
        # augmentation and soft-contact kernels that follow are tape-safe
        # and recorded normally.

        # Compute AABBs for all shapes (already expanded by per-shape effective gaps)
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
            ],
            outputs=[
                self.narrow_phase.shape_aabb_lower,
                self.narrow_phase.shape_aabb_upper,
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
                device=self.device,
                filter_pairs=self.shape_pairs_excluded,
                num_filter_pairs=self.shape_pairs_excluded_count,
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
                device=self.device,
                filter_pairs=self.shape_pairs_excluded,
                num_filter_pairs=self.shape_pairs_excluded_count,
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
                device=self.device,
            )

        # Prepare geometry data arrays for NarrowPhase API
        wp.launch(
            kernel=prepare_geom_data_kernel,
            dim=model.shape_count,
            inputs=[
                model.shape_transform,
                model.shape_body,
                model.shape_type,
                model.shape_scale,
                model.shape_margin,
                state.body_q,
            ],
            outputs=[
                self.geom_data,
                self.geom_transform,
            ],
            device=self.device,
            record_tape=False,
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

        # Run narrow phase with custom contact writer (writes directly to Contacts format)
        self.narrow_phase.launch_custom_write(
            candidate_pair=self.broad_phase_shape_pairs,
            candidate_pair_count=self.broad_phase_pair_count,
            shape_types=model.shape_type,
            shape_data=self.geom_data,
            shape_transform=self.geom_transform,
            shape_source=model.shape_source_ptr,
            shape_sdf_index=model.shape_sdf_index,
            texture_sdf_data=model.texture_sdf_data,
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
            shape_edge_range=model.shape_edge_range,
            writer_data=writer_data,
            device=self.device,
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
        particle_count = len(state.particle_q) if state.particle_q else 0
        if state.particle_q and model.shape_count > 0:
            wp.launch(
                kernel=create_soft_contacts,
                dim=particle_count * model.shape_count,
                inputs=[
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
                    model.shape_world,
                    soft_contact_margin,
                    self.soft_contact_max,
                    model.shape_count,
                    model.shape_flags,
                    model.shape_heightfield_index,
                    model.heightfield_data,
                    model.heightfield_elevations,
                ],
                outputs=[
                    contacts.soft_contact_count,
                    contacts.soft_contact_particle,
                    contacts.soft_contact_shape,
                    contacts.soft_contact_body_pos,
                    contacts.soft_contact_body_vel,
                    contacts.soft_contact_normal,
                    contacts.soft_contact_tids,
                ],
                device=self.device,
            )
