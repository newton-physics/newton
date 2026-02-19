# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from __future__ import annotations

from typing import Literal

import numpy as np
import warp as wp

from ..geometry.broad_phase_nxn import BroadPhaseAllPairs, BroadPhaseExplicit
from ..geometry.broad_phase_sap import BroadPhaseSAP
from ..geometry.collision_core import compute_tight_aabb_from_support
from ..geometry.contact_data import ContactData
from ..geometry.kernels import create_soft_contacts
from ..geometry.narrow_phase import NarrowPhase
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
    body_q: wp.array(dtype=wp.transform)
    shape_body: wp.array(dtype=int)
    shape_contact_margin: wp.array(dtype=float)
    # Output arrays
    contact_count: wp.array(dtype=int)
    out_shape0: wp.array(dtype=int)
    out_shape1: wp.array(dtype=int)
    out_point0: wp.array(dtype=wp.vec3)
    out_point1: wp.array(dtype=wp.vec3)
    out_offset0: wp.array(dtype=wp.vec3)
    out_offset1: wp.array(dtype=wp.vec3)
    out_normal: wp.array(dtype=wp.vec3)
    out_thickness0: wp.array(dtype=float)
    out_thickness1: wp.array(dtype=float)
    out_tids: wp.array(dtype=int)
    # Per-contact shape properties, empty arrays if not enabled.
    # Zero-values indicate that no per-contact shape properties are set for this contact
    out_stiffness: wp.array(dtype=float)
    out_damping: wp.array(dtype=float)
    out_friction: wp.array(dtype=float)


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
        contact_data.radius_eff_a + contact_data.radius_eff_b + contact_data.thickness_a + contact_data.thickness_b
    )

    offset_mag_a = contact_data.radius_eff_a + contact_data.thickness_a
    offset_mag_b = contact_data.radius_eff_b + contact_data.thickness_b

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

    # Use per-shape contact margins (sum of both shapes, consistent with thickness)
    margin_a = writer_data.shape_contact_margin[contact_data.shape_a]
    margin_b = writer_data.shape_contact_margin[contact_data.shape_b]
    contact_margin = margin_a + margin_b

    index = output_index

    if index < 0:
        # compute index using atomic counter
        if d > contact_margin:
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

    # Match kernels.py convention
    contact_normal = -contact_normal_a_to_b

    # Offsets in body frames
    writer_data.out_offset0[index] = wp.transform_vector(X_bw_a, -offset_mag_a * contact_normal)
    writer_data.out_offset1[index] = wp.transform_vector(X_bw_b, offset_mag_b * contact_normal)

    writer_data.out_normal[index] = contact_normal
    writer_data.out_thickness0[index] = offset_mag_a
    writer_data.out_thickness1[index] = offset_mag_b
    writer_data.out_tids[index] = 0  # tid not available in this context

    # Write stiffness/damping/friction only if per-contact shape properties are enabled
    if writer_data.out_stiffness.shape[0] > 0:
        writer_data.out_stiffness[index] = contact_data.contact_stiffness
        writer_data.out_damping[index] = contact_data.contact_damping
        writer_data.out_friction[index] = contact_data.contact_friction_scale


@wp.kernel
def compute_shape_aabbs(
    body_q: wp.array(dtype=wp.transform),
    shape_transform: wp.array(dtype=wp.transform),
    shape_body: wp.array(dtype=int),
    shape_type: wp.array(dtype=int),
    shape_scale: wp.array(dtype=wp.vec3),
    shape_collision_radius: wp.array(dtype=float),
    shape_source_ptr: wp.array(dtype=wp.uint64),
    shape_contact_margin: wp.array(dtype=float),
    # outputs
    aabb_lower: wp.array(dtype=wp.vec3),
    aabb_upper: wp.array(dtype=wp.vec3),
):
    """Compute axis-aligned bounding boxes for each shape in world space.

    Uses support function for most shapes. Infinite planes and meshes use bounding sphere fallback.
    AABBs are enlarged by per-shape contact margin for contact detection.

    Note: Shape thickness is NOT included in AABB expansion - it is applied during narrow phase.
    Therefore, shape_contact_margin should be >= shape_thickness to ensure proper broad phase detection.
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

    # Enlarge AABB by per-shape contact margin for contact detection
    contact_margin = shape_contact_margin[shape_id]
    margin_vec = wp.vec3(contact_margin, contact_margin, contact_margin)

    # Check if this is an infinite plane, mesh, or heightfield - use bounding sphere fallback
    scale = shape_scale[shape_id]
    is_infinite_plane = (geo_type == GeoType.PLANE) and (scale[0] == 0.0 and scale[1] == 0.0)
    is_mesh = geo_type == GeoType.MESH
    is_hfield = geo_type == GeoType.HFIELD

    if is_infinite_plane or is_mesh or is_hfield:
        # Use conservative bounding sphere approach for infinite planes, meshes, and heightfields
        radius = shape_collision_radius[shape_id]
        half_extents = wp.vec3(radius, radius, radius)
        aabb_lower[shape_id] = pos - half_extents - margin_vec
        aabb_upper[shape_id] = pos + half_extents + margin_vec
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


@wp.kernel
def prepare_geom_data_kernel(
    shape_transform: wp.array(dtype=wp.transform),
    shape_body: wp.array(dtype=int),
    shape_type: wp.array(dtype=int),
    shape_scale: wp.array(dtype=wp.vec3),
    shape_thickness: wp.array(dtype=float),
    body_q: wp.array(dtype=wp.transform),
    # Outputs
    geom_data: wp.array(dtype=wp.vec4),  # scale xyz, thickness w
    geom_transform: wp.array(dtype=wp.transform),  # world space transform
):
    """Prepare geometry data arrays for NarrowPhase API."""
    idx = wp.tid()

    # Pack scale and thickness into geom_data
    scale = shape_scale[idx]
    thickness = shape_thickness[idx]
    geom_data[idx] = wp.vec4(scale[0], scale[1], scale[2], thickness)

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

    mesh_mask = shape_types == int(GeoType.MESH)
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


BROAD_PHASE_MODES = ("nxn", "sap", "explicit")
BroadPhaseMode = Literal["nxn", "sap", "explicit"]
BroadPhaseInstance = BroadPhaseAllPairs | BroadPhaseSAP | BroadPhaseExplicit


def _normalize_broad_phase_mode(mode: str) -> str:
    mode_str = str(mode).lower()
    if mode_str not in BROAD_PHASE_MODES:
        raise ValueError(f"Unsupported broad phase mode: {mode!r}")
    return mode_str


def _infer_broad_phase_mode_from_instance(broad_phase: BroadPhaseInstance) -> str:
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
    """

    def __init__(
        self,
        model: Model,
        *,
        reduce_contacts: bool = True,
        rigid_contact_max: int | None = None,
        shape_pairs_filtered: wp.array(dtype=wp.vec2i) | None = None,
        soft_contact_max: int | None = None,
        soft_contact_margin: float = 0.01,
        requires_grad: bool | None = None,
        broad_phase: BroadPhaseMode | BroadPhaseInstance | None = None,
        narrow_phase: NarrowPhase | None = None,
        sdf_hydroelastic_config: NarrowPhase.HydroelasticSDF.Config | None = None,
    ):
        """
        Initialize the CollisionPipeline (expert API).

        Args:
            model (Model): The simulation model.
            reduce_contacts (bool, optional): Whether to reduce contacts for mesh-mesh collisions. Defaults to True.
            rigid_contact_max (int | None, optional): Maximum number of rigid contacts to allocate.
                Resolution order:
                - If provided, use this value.
                - Else if ``model.rigid_contact_max > 0``, use the model value.
                - Else estimate automatically from model shape and pair metadata.
            soft_contact_max (int | None, optional): Maximum number of soft contacts to allocate.
                If None, computed as shape_count * particle_count.
            soft_contact_margin (float, optional): Margin for soft contact generation. Defaults to 0.01.
            requires_grad (bool | None, optional): Whether to enable gradient computation. If None, uses model.requires_grad.
            broad_phase (BroadPhaseMode | BroadPhaseAllPairs | BroadPhaseSAP | BroadPhaseExplicit | None, optional):
                Either a broad phase mode string ("explicit", "nxn", "sap") or
                a prebuilt broad phase instance for expert usage.
            narrow_phase (NarrowPhase | None, optional): Optional prebuilt narrow phase instance. Must be
                provided together with a broad phase instance for expert usage.
            shape_pairs_filtered (wp.array | None, optional): Precomputed shape pairs for EXPLICIT mode.
                When broad_phase is "explicit", uses model.shape_contact_pairs if not provided. For
                "nxn"/"sap" modes, ignored.
            sdf_hydroelastic_config (NarrowPhase.HydroelasticSDF.Config | None, optional): Configuration for
                hydroelastic collision handling. Defaults to None.
        """
        mode_from_broad_phase: str | None = None
        broad_phase_instance: BroadPhaseInstance | None = None
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
                self.shape_pairs_max = (shape_count * (shape_count - 1)) // 2
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
                self.shape_pairs_max = (shape_count * (shape_count - 1)) // 2
                self.shape_pairs_excluded = self._build_excluded_pairs(model)
                self.shape_pairs_excluded_count = (
                    self.shape_pairs_excluded.shape[0] if self.shape_pairs_excluded is not None else 0
                )
            elif self.broad_phase_mode == "sap":
                if shape_world is None:
                    raise ValueError("model.shape_world is required for broad_phase=SAP")
                self.broad_phase = BroadPhaseSAP(shape_world, shape_flags=shape_flags, device=device)
                self.shape_pairs_filtered = None
                self.shape_pairs_max = (shape_count * (shape_count - 1)) // 2
                self.shape_pairs_excluded = self._build_excluded_pairs(model)
                self.shape_pairs_excluded_count = (
                    self.shape_pairs_excluded.shape[0] if self.shape_pairs_excluded is not None else 0
                )
            else:
                raise ValueError(f"Unsupported broad phase mode: {self.broad_phase_mode}")

            # Initialize SDF hydroelastic (returns None if no hydroelastic shape pairs in the model)
            hydroelastic_sdf = NarrowPhase.HydroelasticSDF._from_model(
                model,
                config=sdf_hydroelastic_config,
                writer_func=write_contact,
            )

            # Detect if any mesh or heightfield shapes are present to optimize kernel launches
            has_meshes = False
            has_heightfields = False
            if hasattr(model, "shape_type") and model.shape_type is not None:
                shape_types = model.shape_type.numpy()
                has_meshes = bool((shape_types == int(GeoType.MESH)).any())
                has_heightfields = bool((shape_types == int(GeoType.HFIELD)).any())

            # Initialize narrow phase with pre-allocated buffers
            # max_triangle_pairs is a conservative estimate for mesh collision triangle pairs
            # Pass write_contact as custom writer to write directly to final Contacts format
            self.narrow_phase = NarrowPhase(
                max_candidate_pairs=self.shape_pairs_max,
                max_triangle_pairs=1000000,
                reduce_contacts=self.reduce_contacts,
                device=device,
                shape_aabb_lower=shape_aabb_lower,
                shape_aabb_upper=shape_aabb_upper,
                contact_writer_warp_func=write_contact,
                shape_voxel_resolution=model._shape_voxel_resolution,
                hydroelastic_sdf=hydroelastic_sdf,
                has_meshes=has_meshes,
                has_heightfields=has_heightfields,
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
        Allocate and return a new :class:`Contacts` object for this pipeline.

        Returns:
            Contacts: A newly allocated contacts buffer sized for this pipeline.
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
    def _build_excluded_pairs(model: Model) -> wp.array(dtype=wp.vec2i) | None:
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
        """
        Run the collision pipeline using NarrowPhase.

        Args:
            state: The current simulation state.
            contacts: The contacts buffer to populate (will be cleared first).
            soft_contact_margin: Margin for soft contact generation. If None, uses the value from construction.

        """

        contacts.clear()
        # TODO: validate contacts dimensions & compatibility

        # Clear counters
        self.broad_phase_pair_count.zero_()

        model = self.model
        # update any additional parameters
        soft_contact_margin = soft_contact_margin if soft_contact_margin is not None else self.soft_contact_margin

        # When requires_grad, skip rigid contact path so the tape does not record narrow phase
        # kernels (they have enable_backward=False). Only soft contacts are differentiable.
        if not self.requires_grad:
            # Compute AABBs for all shapes (already expanded by per-shape contact margins)
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
                    model.shape_contact_margin,
                ],
                outputs=[
                    self.narrow_phase.shape_aabb_lower,
                    self.narrow_phase.shape_aabb_upper,
                ],
                device=self.device,
            )

            # Run broad phase (AABBs are already expanded by contact margins, so pass None)
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
                    model.shape_thickness,
                    state.body_q,
                ],
                outputs=[
                    self.geom_data,
                    self.geom_transform,
                ],
                device=self.device,
            )

            # Create ContactWriterData struct for custom contact writing
            writer_data = ContactWriterData()
            writer_data.contact_max = contacts.rigid_contact_max
            writer_data.body_q = state.body_q
            writer_data.shape_body = model.shape_body
            writer_data.shape_contact_margin = model.shape_contact_margin
            writer_data.contact_count = contacts.rigid_contact_count
            writer_data.out_shape0 = contacts.rigid_contact_shape0
            writer_data.out_shape1 = contacts.rigid_contact_shape1
            writer_data.out_point0 = contacts.rigid_contact_point0
            writer_data.out_point1 = contacts.rigid_contact_point1
            writer_data.out_offset0 = contacts.rigid_contact_offset0
            writer_data.out_offset1 = contacts.rigid_contact_offset1
            writer_data.out_normal = contacts.rigid_contact_normal
            writer_data.out_thickness0 = contacts.rigid_contact_thickness0
            writer_data.out_thickness1 = contacts.rigid_contact_thickness1
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
                sdf_data=model.sdf_data,
                shape_sdf_index=model.shape_sdf_index,
                shape_contact_margin=model.shape_contact_margin,
                shape_collision_radius=model.shape_collision_radius,
                shape_flags=model.shape_flags,
                shape_collision_aabb_lower=model.shape_collision_aabb_lower,
                shape_collision_aabb_upper=model.shape_collision_aabb_upper,
                shape_voxel_resolution=self.narrow_phase.shape_voxel_resolution,
                shape_heightfield_data=model.shape_heightfield_data,
                heightfield_elevation_data=model.heightfield_elevation_data,
                writer_data=writer_data,
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

    def get_hydro_contact_surface(self):
        """Get hydroelastic contact surface data for visualization, if available.

        Returns:
            HydroelasticContactSurfaceData if hydroelastic_sdf is configured, None otherwise.
        """
        if self.hydroelastic_sdf is not None:
            return self.hydroelastic_sdf.get_hydro_contact_surface()
        return None

    def set_output_contact_surface(self, enabled: bool) -> None:
        """Enable or disable contact surface visualization.

        Note: When ``output_contact_surface=True`` in the config, the kernel always
        writes debug surface data. This method is provided for API compatibility but
        the actual display is controlled by the viewer's ``show_hydro_contact_surface`` flag.

        Args:
            enabled: If True, visualization is enabled (viewer will display the data).
                     If False, visualization is disabled (viewer will hide the data).
        """
        if self.hydroelastic_sdf is not None:
            self.hydroelastic_sdf.set_output_contact_surface(enabled)
