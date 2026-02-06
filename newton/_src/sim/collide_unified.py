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

from enum import IntEnum

import warp as wp

from ..geometry.broad_phase_nxn import BroadPhaseAllPairs, BroadPhaseExplicit
from ..geometry.broad_phase_sap import BroadPhaseSAP
from ..geometry.collision_core import compute_tight_aabb_from_support
from ..geometry.contact_data import ContactData
from ..geometry.kernels import create_soft_contacts
from ..geometry.narrow_phase import NarrowPhase
from ..geometry.sdf_hydroelastic import SDFHydroelastic, SDFHydroelasticConfig
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
class UnifiedContactWriterData:
    """Contact writer data for collide_unified write_contact function."""

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


class BroadPhaseMode(IntEnum):
    """Broad phase collision detection mode."""

    NXN = 0
    """All-pairs broad phase with AABB checks (simple, O(N²) but good for small scenes)"""

    SAP = 1
    """Sweep and Prune broad phase with AABB sorting (faster for larger scenes, O(N log N))"""

    EXPLICIT = 2
    """Use precomputed shape pairs (most efficient when pairs are known ahead of time)"""


@wp.func
def write_contact(
    contact_data: ContactData,
    writer_data: UnifiedContactWriterData,
    output_index: int,
):
    """
    Write a contact to the output arrays using ContactData and UnifiedContactWriterData.

    Args:
        contact_data: ContactData struct containing contact information
        writer_data: UnifiedContactWriterData struct containing body info and output arrays
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
            # Reached buffer limit
            wp.atomic_add(writer_data.contact_count, 0, -1)
            return

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

    # Check if this is an infinite plane, mesh, or SDF - use bounding sphere fallback
    scale = shape_scale[shape_id]
    is_infinite_plane = (geo_type == int(GeoType.PLANE)) and (scale[0] == 0.0 and scale[1] == 0.0)
    is_mesh = geo_type == int(GeoType.MESH)
    is_sdf = geo_type == int(GeoType.SDF)

    if is_infinite_plane or is_mesh or is_sdf:
        # Use conservative bounding sphere approach for infinite planes, meshes, and SDFs
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
        if geo_type == int(GeoType.CONVEX_MESH):
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


def _estimate_rigid_contact_max_unified(model: Model) -> int:
    """
    Estimate the maximum number of rigid contacts for the unified collision pipeline.

    Unlike the standard collision pipeline which uses per-vertex contact counting,
    the unified pipeline uses SDF-based collision with contact reduction for mesh-mesh
    pairs. This function provides a linear estimate assuming each shape contacts
    only a limited number of neighbors (due to spatial locality).

    Args:
        model: The simulation model.

    Returns:
        Estimated maximum number of rigid contacts.
    """
    # Get shape types
    if not hasattr(model, "shape_type") or model.shape_type is None:
        return 1000  # Fallback

    shape_types = model.shape_type.numpy()

    # Constants for contact estimation
    CONTACTS_PER_PAIR = 20
    # Assume each shape contacts at most this many other shapes (spatial locality)
    MAX_NEIGHBORS_PER_SHAPE = 20

    # Count shapes by type
    mesh_types = {int(GeoType.MESH), int(GeoType.CONVEX_MESH)}
    num_meshes = sum(1 for t in shape_types if t in mesh_types)
    num_planes = sum(1 for t in shape_types if t == int(GeoType.PLANE))
    num_other = len(shape_types) - num_meshes - num_planes

    # Linear estimate: each shape contacts up to MAX_NEIGHBORS_PER_SHAPE others
    # Divide by 2 to avoid double-counting pairs
    num_shapes = num_meshes + num_other
    estimated_pairs = (num_shapes * MAX_NEIGHBORS_PER_SHAPE) // 2

    # Add plane contacts (each plane can contact all shapes)
    plane_contacts = num_planes * (num_meshes + num_other) * CONTACTS_PER_PAIR

    total_contacts = estimated_pairs * CONTACTS_PER_PAIR + plane_contacts

    # Ensure minimum allocation
    return max(1000, total_contacts)


class CollisionPipelineUnified:
    """
    Full-featured collision pipeline with GJK/MPR narrow phase and pluggable broad phase.

    Key features:
        - GJK/MPR algorithms for convex-convex collision detection
        - Multiple broad phase modes: NXN (all-pairs), SAP (sweep-and-prune), EXPLICIT (precomputed pairs)
        - Mesh-mesh collision via SDF with contact reduction
        - Optional hydroelastic contact model for compliant surfaces
    """

    def __init__(
        self,
        model: Model,
        *,
        reduce_contacts: bool = True,
        rigid_contact_max: int | None = None,
        soft_contact_max: int | None = None,
        soft_contact_margin: float = 0.01,
        iterate_mesh_vertices: bool = True,
        requires_grad: bool | None = None,
        broad_phase_mode: BroadPhaseMode = BroadPhaseMode.NXN,
        shape_pairs_filtered: wp.array(dtype=wp.vec2i) | None = None,
        sap_sort_type=None,
        sdf_hydroelastic_config: SDFHydroelasticConfig | None = None,
    ):
        """
        Initialize the CollisionPipelineUnified.

        Args:
            model (Model): The simulation model.
            reduce_contacts (bool, optional): Whether to reduce contacts for mesh-mesh collisions. Defaults to True.
            rigid_contact_max (int | None, optional): Maximum number of rigid contacts to allocate.
                If None, estimated automatically.
            soft_contact_max (int | None, optional): Maximum number of soft contacts to allocate.
                If None, computed as shape_count * particle_count.
            soft_contact_margin (float, optional): Margin for soft contact generation. Defaults to 0.01.
            iterate_mesh_vertices (bool, optional): Whether to iterate mesh vertices for collision. Defaults to True.
            requires_grad (bool | None, optional): Whether to enable gradient computation. If None, uses model.requires_grad.
            broad_phase_mode (BroadPhaseMode, optional): Broad phase collision detection mode. Defaults to BroadPhaseMode.NXN.
            shape_pairs_filtered (wp.array | None, optional): Precomputed shape pairs for EXPLICIT mode.
                Required when broad_phase_mode is BroadPhaseMode.EXPLICIT. For NXN/SAP modes, can use model.shape_contact_pairs if available.
            sap_sort_type (SAPSortType | None, optional): Sorting algorithm for SAP broad phase.
                Only used when broad_phase_mode is BroadPhaseMode.SAP. If None, uses default (SEGMENTED).
            sdf_hydroelastic_config (SDFHydroelasticConfig | None, optional): Configuration for SDF hydroelastic collision handling. Defaults to None.
        """
        self.model = model
        shape_count = model.shape_count
        particle_count = model.particle_count
        device = model.device

        # Estimate rigid_contact_max for unified pipeline (accounts for contact reduction)
        if rigid_contact_max is None:
            rigid_contact_max = _estimate_rigid_contact_max_unified(model)
        if requires_grad is None:
            requires_grad = model.requires_grad

        # For EXPLICIT mode, use provided shape_pairs_filtered or fall back to model pairs
        if shape_pairs_filtered is None and broad_phase_mode == BroadPhaseMode.EXPLICIT:
            if hasattr(model, "shape_contact_pairs") and model.shape_contact_pairs is not None:
                shape_pairs_filtered = model.shape_contact_pairs

        # Initialize SDF hydroelastic (returns None if no hydroelastic shape pairs in the model)
        sdf_hydroelastic = SDFHydroelastic._from_model(model, config=sdf_hydroelastic_config, writer_func=write_contact)

        # Detect if any mesh shapes are present to optimize kernel launches
        has_meshes = False
        if hasattr(model, "shape_type") and model.shape_type is not None:
            shape_types = model.shape_type.numpy()
            has_meshes = bool((shape_types == int(GeoType.MESH)).any())

        shape_world = getattr(model, "shape_world", None)
        shape_flags = getattr(model, "shape_flags", None)

        self.shape_count = shape_count
        self.broad_phase_mode = broad_phase_mode
        self.device = device
        self.reduce_contacts = reduce_contacts
        self.shape_pairs_max = (shape_count * (shape_count - 1)) // 2

        # Initialize broad phase
        if self.broad_phase_mode == BroadPhaseMode.NXN:
            if shape_world is None:
                raise ValueError("shape_world must be provided when using BroadPhaseMode.NXN")
            self.nxn_broadphase = BroadPhaseAllPairs(shape_world, shape_flags=shape_flags, device=device)
            self.sap_broadphase = None
            self.explicit_broadphase = None
            self.shape_pairs_filtered = None
        elif self.broad_phase_mode == BroadPhaseMode.SAP:
            if shape_world is None:
                raise ValueError("shape_world must be provided when using BroadPhaseMode.SAP")
            self.sap_broadphase = BroadPhaseSAP(
                shape_world,
                shape_flags=shape_flags,
                sort_type=sap_sort_type,
                device=device,
            )
            self.nxn_broadphase = None
            self.explicit_broadphase = None
            self.shape_pairs_filtered = None
        else:  # BroadPhaseMode.EXPLICIT
            if shape_pairs_filtered is None:
                raise ValueError("shape_pairs_filtered must be provided when using EXPLICIT mode")
            self.explicit_broadphase = BroadPhaseExplicit()
            self.nxn_broadphase = None
            self.sap_broadphase = None
            self.shape_pairs_filtered = shape_pairs_filtered
            self.shape_pairs_max = len(shape_pairs_filtered)

        self.rigid_contact_max = rigid_contact_max

        # Allocate buffers
        with wp.ScopedDevice(device):
            self.broad_phase_pair_count = wp.zeros(1, dtype=wp.int32, device=device)
            self.broad_phase_shape_pairs = wp.zeros(self.shape_pairs_max, dtype=wp.vec2i, device=device)
            self.shape_aabb_lower = wp.zeros(shape_count, dtype=wp.vec3, device=device)
            self.shape_aabb_upper = wp.zeros(shape_count, dtype=wp.vec3, device=device)

        # Initialize narrow phase with pre-allocated buffers
        # Pass AABB arrays so narrow phase can use them instead of computing AABBs internally
        # max_triangle_pairs is a conservative estimate for mesh collision triangle pairs
        # Pass write_contact as custom writer to write directly to final Contacts format
        self.narrow_phase = NarrowPhase(
            max_candidate_pairs=self.shape_pairs_max,
            max_triangle_pairs=1000000,
            reduce_contacts=self.reduce_contacts,
            device=device,
            shape_aabb_lower=self.shape_aabb_lower,
            shape_aabb_upper=self.shape_aabb_upper,
            contact_writer_warp_func=write_contact,
            sdf_hydroelastic=sdf_hydroelastic,
            has_meshes=has_meshes,
        )
        self.sdf_hydroelastic = self.narrow_phase.sdf_hydroelastic

        with wp.ScopedDevice(device):
            # Narrow phase input arrays
            self.geom_data = wp.zeros(shape_count, dtype=wp.vec4, device=device)
            self.geom_transform = wp.zeros(shape_count, dtype=wp.transform, device=device)

        if soft_contact_max is None:
            soft_contact_max = shape_count * particle_count
        self.soft_contact_margin = soft_contact_margin
        self.soft_contact_max = soft_contact_max
        self.requires_grad = requires_grad

    def contacts(self) -> Contacts:
        """
        Allocate and return a new Contacts object for this pipeline.

        Returns:
            Contacts: A newly allocated contacts object for this pipeline.
        """
        return Contacts(
            self.rigid_contact_max,
            self.soft_contact_max,
            requires_grad=self.requires_grad,
            device=self.device,
            per_contact_shape_properties=self.narrow_phase.sdf_hydroelastic is not None,
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

        Note:
            Rigid contact margins are controlled per-shape via :attr:`Model.shape_contact_margin`.
        """

        contacts.clear()
        # TODO: validate contacts dimensions & compatibility

        model = self.model

        # update any additional parameters
        soft_contact_margin = soft_contact_margin if soft_contact_margin is not None else self.soft_contact_margin

        # Clear counters
        self.broad_phase_pair_count.zero_()

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
                self.shape_aabb_lower,
                self.shape_aabb_upper,
            ],
            device=self.device,
        )

        # Run broad phase (AABBs are already expanded by contact margins, so pass None)
        if self.broad_phase_mode == BroadPhaseMode.NXN:
            self.nxn_broadphase.launch(
                self.shape_aabb_lower,
                self.shape_aabb_upper,
                None,  # AABBs are pre-expanded, no additional margin needed
                model.shape_collision_group,
                model.shape_world,
                model.shape_count,
                self.broad_phase_shape_pairs,
                self.broad_phase_pair_count,
                device=self.device,
            )
        elif self.broad_phase_mode == BroadPhaseMode.SAP:
            self.sap_broadphase.launch(
                self.shape_aabb_lower,
                self.shape_aabb_upper,
                None,  # AABBs are pre-expanded, no additional margin needed
                model.shape_collision_group,
                model.shape_world,
                model.shape_count,
                self.broad_phase_shape_pairs,
                self.broad_phase_pair_count,
                device=self.device,
            )
        else:  # BroadPhaseMode.EXPLICIT
            self.explicit_broadphase.launch(
                self.shape_aabb_lower,
                self.shape_aabb_upper,
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

        # Create UnifiedContactWriterData struct for custom contact writing
        writer_data = UnifiedContactWriterData()
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
            num_candidate_pair=self.broad_phase_pair_count,
            shape_types=model.shape_type,
            shape_data=self.geom_data,
            shape_transform=self.geom_transform,
            shape_source=model.shape_source_ptr,
            shape_sdf_data=model.shape_sdf_data,
            shape_contact_margin=model.shape_contact_margin,
            shape_collision_radius=model.shape_collision_radius,
            shape_flags=model.shape_flags,
            shape_local_aabb_lower=model.shape_local_aabb_lower,
            shape_local_aabb_upper=model.shape_local_aabb_upper,
            shape_voxel_resolution=model.shape_voxel_resolution,
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

        return contacts

    def get_hydro_contact_surface(self):
        """Get hydroelastic contact surface data for visualization, if available.

        Returns:
            HydroelasticContactSurfaceData if sdf_hydroelastic is configured, None otherwise.
        """
        if self.sdf_hydroelastic is not None:
            return self.sdf_hydroelastic.get_hydro_contact_surface()
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
        if self.sdf_hydroelastic is not None:
            self.sdf_hydroelastic.set_output_contact_surface(enabled)
