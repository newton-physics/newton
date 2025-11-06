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

from ..core.types import Devicelike
from ..geometry.broad_phase_nxn import BroadPhaseAllPairs, BroadPhaseExplicit
from ..geometry.broad_phase_sap import BroadPhaseSAP
from ..geometry.collision_core import compute_tight_aabb_from_support
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
    contact_point_center: wp.vec3,
    contact_normal_a_to_b: wp.vec3,
    contact_distance: float,
    radius_eff_a: float,
    radius_eff_b: float,
    thickness_a: float,
    thickness_b: float,
    shape_a: int,
    shape_b: int,
    X_bw_a: wp.transform,
    X_bw_b: wp.transform,
    tid: int,
    rigid_contact_margin: float,
    contact_max: int,
    # outputs
    contact_count: wp.array(dtype=int),
    out_shape0: wp.array(dtype=int),
    out_shape1: wp.array(dtype=int),
    out_point0: wp.array(dtype=wp.vec3),
    out_point1: wp.array(dtype=wp.vec3),
    out_offset0: wp.array(dtype=wp.vec3),
    out_offset1: wp.array(dtype=wp.vec3),
    out_normal: wp.array(dtype=wp.vec3),
    out_thickness0: wp.array(dtype=float),
    out_thickness1: wp.array(dtype=float),
    out_tids: wp.array(dtype=int),
):
    """
    Write a contact to the output arrays.

    Args:
        contact_point_center: Center point of contact in world space
        contact_normal_a_to_b: Contact normal pointing from shape A to B
        contact_distance: Distance between contact points
        radius_eff_a: Effective radius of shape A (only use nonzero values for shapes that are a minkowski sum of a sphere and another object, eg sphere or capsule)
        radius_eff_b: Effective radius of shape B (only use nonzero values for shapes that are a minkowski sum of a sphere and another object, eg sphere or capsule)
        thickness_a: Contact thickness for shape A (similar to contact offset)
        thickness_b: Contact thickness for shape B (similar to contact offset)
        shape_a: Shape A index
        shape_b: Shape B index
        X_bw_a: Transform from world to body A
        X_bw_b: Transform from world to body B
        tid: Thread ID
        rigid_contact_margin: Contact margin for rigid bodies
        contact_max: Maximum number of contacts
        contact_count: Array to track contact count
        out_shape0: Output array for shape A indices
        out_shape1: Output array for shape B indices
        out_point0: Output array for contact points on shape A
        out_point1: Output array for contact points on shape B
        out_offset0: Output array for offsets on shape A
        out_offset1: Output array for offsets on shape B
        out_normal: Output array for contact normals
        out_thickness0: Output array for thickness values for shape A
        out_thickness1: Output array for thickness values for shape B
        out_tids: Output array for thread IDs
    """

    total_separation_needed = radius_eff_a + radius_eff_b + thickness_a + thickness_b

    offset_mag_a = radius_eff_a + thickness_a
    offset_mag_b = radius_eff_b + thickness_b

    # Distance calculation matching box_plane_collision
    contact_normal_a_to_b = wp.normalize(contact_normal_a_to_b)

    a_contact_world = contact_point_center - contact_normal_a_to_b * (0.5 * contact_distance + radius_eff_a)
    b_contact_world = contact_point_center + contact_normal_a_to_b * (0.5 * contact_distance + radius_eff_b)

    diff = b_contact_world - a_contact_world
    distance = wp.dot(diff, contact_normal_a_to_b)
    d = distance - total_separation_needed
    if d < rigid_contact_margin:
        index = wp.atomic_add(contact_count, 0, 1)
        if index >= contact_max:
            # Reached buffer limit
            return

        out_shape0[index] = shape_a
        out_shape1[index] = shape_b

        # Contact points are stored in body frames
        out_point0[index] = wp.transform_point(X_bw_a, a_contact_world)
        out_point1[index] = wp.transform_point(X_bw_b, b_contact_world)

        # Match kernels.py convention
        contact_normal = -contact_normal_a_to_b

        # Offsets in body frames
        out_offset0[index] = wp.transform_vector(X_bw_a, -offset_mag_a * contact_normal)
        out_offset1[index] = wp.transform_vector(X_bw_b, offset_mag_b * contact_normal)

        out_normal[index] = contact_normal
        out_thickness0[index] = offset_mag_a
        out_thickness1[index] = offset_mag_b
        out_tids[index] = tid


@wp.kernel
def compute_shape_aabbs(
    body_q: wp.array(dtype=wp.transform),
    shape_transform: wp.array(dtype=wp.transform),
    shape_body: wp.array(dtype=int),
    shape_type: wp.array(dtype=int),
    shape_scale: wp.array(dtype=wp.vec3),
    shape_collision_radius: wp.array(dtype=float),
    shape_source_ptr: wp.array(dtype=wp.uint64),
    rigid_contact_margin: float,
    # outputs
    aabb_lower: wp.array(dtype=wp.vec3),
    aabb_upper: wp.array(dtype=wp.vec3),
):
    """Compute axis-aligned bounding boxes for each shape in world space.

    Uses support function for most shapes. Infinite planes and meshes use bounding sphere fallback.
    AABBs are enlarged by rigid_contact_margin for contact detection margin.
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

    # Enlarge AABB by rigid_contact_margin for contact detection
    margin_vec = wp.vec3(rigid_contact_margin, rigid_contact_margin, rigid_contact_margin)

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
def convert_narrow_phase_to_contacts_kernel(
    contact_pair: wp.array(dtype=wp.vec2i),
    contact_position: wp.array(dtype=wp.vec3),
    contact_normal: wp.array(dtype=wp.vec3),
    contact_penetration: wp.array(dtype=float),
    narrow_contact_count: wp.array(dtype=int),
    geom_data: wp.array(dtype=wp.vec4),  # Contains thickness in w component
    shape_type: wp.array(dtype=int),
    body_q: wp.array(dtype=wp.transform),
    shape_body: wp.array(dtype=int),
    rigid_contact_margin: float,
    contact_max: int,
    # Outputs (Contacts format)
    out_count: wp.array(dtype=int),
    out_shape0: wp.array(dtype=int),
    out_shape1: wp.array(dtype=int),
    out_point0: wp.array(dtype=wp.vec3),
    out_point1: wp.array(dtype=wp.vec3),
    out_offset0: wp.array(dtype=wp.vec3),
    out_offset1: wp.array(dtype=wp.vec3),
    out_normal: wp.array(dtype=wp.vec3),
    out_thickness0: wp.array(dtype=float),
    out_thickness1: wp.array(dtype=float),
    out_tids: wp.array(dtype=int),
):
    """
    Convert NarrowPhase output format to Contacts format using write_contact.

    NarrowPhase outputs:
    - contact_position: center point of contact
    - contact_normal: NEGATED normal (pointing from shape1 to shape0)
    - contact_penetration: d = distance - total_separation, negative if penetrating

    write_contact expects:
    - contact_normal_a_to_b: normal pointing from shape0 to shape1
    - contact_distance: distance such that d = contact_distance - (thickness_a + thickness_b)

    This kernel handles the conversion between these two formats.
    """
    idx = wp.tid()
    num_contacts = narrow_contact_count[0]

    if idx >= num_contacts:
        return

    # Get contact pair
    pair = contact_pair[idx]
    shape0 = pair[0]
    shape1 = pair[1]

    # Extract thickness values
    thickness_a = geom_data[shape0][3]
    thickness_b = geom_data[shape1][3]

    # Extract effective radius for sphere and capsule shapes
    type_a = shape_type[shape0]
    type_b = shape_type[shape1]

    radius_eff_a = 0.0
    radius_eff_b = 0.0

    # For spheres and capsules, extract the radius from scale[0]
    if type_a == int(GeoType.SPHERE) or type_a == int(GeoType.CAPSULE):
        radius_eff_a = geom_data[shape0][0]

    if type_b == int(GeoType.SPHERE) or type_b == int(GeoType.CAPSULE):
        radius_eff_b = geom_data[shape1][0]

    # Get contact data from narrow phase
    contact_point_center = contact_position[idx]
    # Narrow phase outputs negated normal (pointing B to A), but write_contact expects A to B
    contact_normal_a_to_b = contact_normal[idx]  # Undo the negation from narrow_phase
    # Narrow phase outputs penetration (negative when overlapping)
    # write_contact expects contact_distance such that when recomputed gives same d
    # Since d = distance - (radii + thickness), and we have d directly, we need to add back thickness
    contact_distance = contact_penetration[idx] + thickness_a + thickness_b

    # Get body inverse transforms
    body0 = shape_body[shape0]
    body1 = shape_body[shape1]

    X_bw_a = wp.transform_identity() if body0 == -1 else wp.transform_inverse(body_q[body0])
    X_bw_b = wp.transform_identity() if body1 == -1 else wp.transform_inverse(body_q[body1])

    # Use write_contact to format the contact
    write_contact(
        contact_point_center,
        contact_normal_a_to_b,
        contact_distance,
        radius_eff_a,
        radius_eff_b,
        thickness_a,
        thickness_b,
        shape0,
        shape1,
        X_bw_a,
        X_bw_b,
        idx,
        rigid_contact_margin,
        contact_max,
        out_count,
        out_shape0,
        out_shape1,
        out_point0,
        out_point1,
        out_offset0,
        out_offset1,
        out_normal,
        out_thickness0,
        out_thickness1,
        out_tids,
    )


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


class CollisionPipelineUnified:
    """
    Collision pipeline using NarrowPhase class for narrow phase collision detection.

    This is similar to CollisionPipelineUnified but uses the NarrowPhase API,
    mainly for testing purposes.
    """

    def __init__(
        self,
        shape_count: int,
        particle_count: int,
        shape_pairs_filtered: wp.array(dtype=wp.vec2i) | None = None,
        rigid_contact_max: int | None = None,
        rigid_contact_max_per_pair: int = 10,
        rigid_contact_margin: float = 0.01,
        soft_contact_max: int | None = None,
        soft_contact_margin: float = 0.01,
        edge_sdf_iter: int = 10,
        iterate_mesh_vertices: bool = True,
        requires_grad: bool = False,
        device: Devicelike = None,
        broad_phase_mode: BroadPhaseMode = BroadPhaseMode.NXN,
        shape_collision_group: wp.array(dtype=int) | None = None,
        shape_world: wp.array(dtype=int) | None = None,
        shape_flags: wp.array(dtype=int) | None = None,
        sap_sort_type=None,
    ):
        """
        Initialize the CollisionPipelineUnified.

        Args:
            shape_count (int): Number of shapes in the simulation.
            particle_count (int): Number of particles in the simulation.
            shape_pairs_filtered (wp.array | None, optional): Precomputed shape pairs for EXPLICIT broad phase mode.
                Required when broad_phase_mode is BroadPhaseMode.EXPLICIT, ignored otherwise.
            rigid_contact_max (int | None, optional): Maximum number of rigid contacts to allocate.
                If None, computed as shape_pairs_max * rigid_contact_max_per_pair.
            rigid_contact_max_per_pair (int, optional): Maximum number of contact points per shape pair. Defaults to 10.
            rigid_contact_margin (float, optional): Margin for rigid contact generation. Defaults to 0.01.
            soft_contact_max (int | None, optional): Maximum number of soft contacts to allocate.
                If None, computed as shape_count * particle_count.
            soft_contact_margin (float, optional): Margin for soft contact generation. Defaults to 0.01.
            edge_sdf_iter (int, optional): Number of iterations for edge SDF collision. Defaults to 10.
            iterate_mesh_vertices (bool, optional): Whether to iterate mesh vertices for collision. Defaults to True.
            requires_grad (bool, optional): Whether to enable gradient computation. Defaults to False.
            device (Devicelike, optional): The device on which to allocate arrays and perform computation.
            broad_phase_mode (BroadPhaseMode, optional): Broad phase mode for collision detection.
                - BroadPhaseMode.NXN: Use all-pairs AABB broad phase (O(N²), good for small scenes)
                - BroadPhaseMode.SAP: Use sweep-and-prune AABB broad phase (O(N log N), better for larger scenes)
                - BroadPhaseMode.EXPLICIT: Use precomputed shape pairs (most efficient when pairs known)
                Defaults to BroadPhaseMode.NXN.
            shape_collision_group (wp.array | None, optional): Array of collision group IDs for each shape.
                Used during broad phase kernel execution to filter pairs based on collision group rules.
            shape_world (wp.array | None, optional): Array of world indices for each shape.
                Required by NXN and SAP broad phases to organize geometries by world. If None, will be set during collide().
            shape_flags (wp.array | None, optional): Array of shape flags (ShapeFlags) for each shape.
                Used by NXN and SAP broad phases to filter out non-colliding shapes (e.g., visual-only).
                If provided, only shapes with COLLIDE_SHAPES flag will participate in broad phase.
            sap_sort_type (SAPSortType | None, optional): Sorting algorithm for SAP broad phase.
                Only used when broad_phase_mode is BroadPhaseMode.SAP. Options: SEGMENTED or TILE.
                If None, uses default (SEGMENTED).
        """
        self.contacts = None
        self.shape_count = shape_count
        self.broad_phase_mode = broad_phase_mode
        self.device = device

        self.shape_pairs_max = (shape_count * (shape_count - 1)) // 2
        self.rigid_contact_margin = rigid_contact_margin

        if rigid_contact_max is not None:
            self.rigid_contact_max = rigid_contact_max
        else:
            self.rigid_contact_max = self.shape_pairs_max * rigid_contact_max_per_pair

        # Initialize broad phase
        if self.broad_phase_mode == BroadPhaseMode.NXN:
            if shape_world is None:
                raise ValueError("shape_world must be provided when using BroadPhaseMode.NXN")
            self.nxn_broadphase = BroadPhaseAllPairs(shape_world, geom_flags=shape_flags, device=device)
            self.sap_broadphase = None
            self.explicit_broadphase = None
            self.shape_pairs_filtered = None
        elif self.broad_phase_mode == BroadPhaseMode.SAP:
            if shape_world is None:
                raise ValueError("shape_world must be provided when using BroadPhaseMode.SAP")
            self.sap_broadphase = BroadPhaseSAP(
                shape_world,
                geom_flags=shape_flags,
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

        # Allocate buffers
        with wp.ScopedDevice(device):
            self.broad_phase_pair_count = wp.zeros(1, dtype=wp.int32, device=device)
            self.broad_phase_shape_pairs = wp.zeros(self.shape_pairs_max, dtype=wp.vec2i, device=device)
            self.shape_aabb_lower = wp.zeros(shape_count, dtype=wp.vec3, device=device)
            self.shape_aabb_upper = wp.zeros(shape_count, dtype=wp.vec3, device=device)

        # Initialize narrow phase with pre-allocated buffers
        # Pass AABB arrays so narrow phase can use them instead of computing AABBs internally
        # max_triangle_pairs is a conservative estimate for mesh collision triangle pairs
        self.narrow_phase = NarrowPhase(
            max_candidate_pairs=self.shape_pairs_max,
            max_triangle_pairs=1000000,
            device=device,
            geom_aabb_lower=self.shape_aabb_lower,
            geom_aabb_upper=self.shape_aabb_upper,
        )

        with wp.ScopedDevice(device):
            # Narrow phase input/output arrays
            self.geom_data = wp.zeros(shape_count, dtype=wp.vec4, device=device)
            self.geom_transform = wp.zeros(shape_count, dtype=wp.transform, device=device)
            self.geom_cutoff = wp.full(shape_count, rigid_contact_margin, dtype=wp.float32, device=device)

            # Narrow phase output arrays
            self.narrow_contact_pair = wp.zeros(self.rigid_contact_max, dtype=wp.vec2i, device=device)
            self.narrow_contact_position = wp.zeros(self.rigid_contact_max, dtype=wp.vec3, device=device)
            self.narrow_contact_normal = wp.zeros(self.rigid_contact_max, dtype=wp.vec3, device=device)
            self.narrow_contact_penetration = wp.zeros(self.rigid_contact_max, dtype=wp.float32, device=device)
            self.narrow_contact_count = wp.zeros(1, dtype=wp.int32, device=device)

        if soft_contact_max is None:
            soft_contact_max = shape_count * particle_count
        self.soft_contact_margin = soft_contact_margin
        self.soft_contact_max = soft_contact_max
        self.requires_grad = requires_grad
        self.edge_sdf_iter = edge_sdf_iter

    @classmethod
    def from_model(
        cls,
        model: Model,
        rigid_contact_max_per_pair: int | None = None,
        rigid_contact_margin: float = 0.01,
        soft_contact_max: int | None = None,
        soft_contact_margin: float = 0.01,
        edge_sdf_iter: int = 10,
        iterate_mesh_vertices: bool = True,
        requires_grad: bool | None = None,
        broad_phase_mode: BroadPhaseMode = BroadPhaseMode.NXN,
        shape_pairs_filtered: wp.array(dtype=wp.vec2i) | None = None,
        sap_sort_type=None,
    ) -> CollisionPipelineUnified:
        """
        Create a CollisionPipelineUnified instance from a Model.

        Args:
            model (Model): The simulation model.
            rigid_contact_max_per_pair (int | None, optional): Maximum number of contact points per shape pair.
                If None, uses model.rigid_contact_max and sets per-pair to 0.
            rigid_contact_margin (float, optional): Margin for rigid contact generation. Defaults to 0.01.
            soft_contact_max (int | None, optional): Maximum number of soft contacts to allocate.
            soft_contact_margin (float, optional): Margin for soft contact generation. Defaults to 0.01.
            edge_sdf_iter (int, optional): Number of iterations for edge SDF collision. Defaults to 10.
            iterate_mesh_vertices (bool, optional): Whether to iterate mesh vertices for collision. Defaults to True.
            requires_grad (bool | None, optional): Whether to enable gradient computation. If None, uses model.requires_grad.
            broad_phase_mode (BroadPhaseMode, optional): Broad phase collision detection mode. Defaults to BroadPhaseMode.NXN.
            shape_pairs_filtered (wp.array | None, optional): Precomputed shape pairs for EXPLICIT mode.
                Required when broad_phase_mode is BroadPhaseMode.EXPLICIT. For NXN/SAP modes, can use model.shape_contact_pairs if available.
            sap_sort_type (SAPSortType | None, optional): Sorting algorithm for SAP broad phase.
                Only used when broad_phase_mode is BroadPhaseMode.SAP. If None, uses default (SEGMENTED).

        Returns:
            CollisionPipeline: The constructed collision pipeline.
        """
        rigid_contact_max = None
        if rigid_contact_max_per_pair is None:
            rigid_contact_max = model.rigid_contact_max
            rigid_contact_max_per_pair = 0
        if requires_grad is None:
            requires_grad = model.requires_grad

        # For EXPLICIT mode, use provided shape_pairs_filtered or fall back to model pairs
        # For NXN/SAP modes, shape_pairs_filtered is not used (but can be provided for EXPLICIT)
        if shape_pairs_filtered is None and broad_phase_mode == BroadPhaseMode.EXPLICIT:
            # Try to use model.shape_contact_pairs if available
            if hasattr(model, "shape_contact_pairs") and model.shape_contact_pairs is not None:
                shape_pairs_filtered = model.shape_contact_pairs
            else:
                # Will raise error in __init__ if EXPLICIT mode requires it
                shape_pairs_filtered = None

        return CollisionPipelineUnified(
            model.shape_count,
            model.particle_count,
            shape_pairs_filtered,
            rigid_contact_max,
            rigid_contact_max_per_pair,
            rigid_contact_margin,
            soft_contact_max,
            soft_contact_margin,
            edge_sdf_iter,
            iterate_mesh_vertices,
            requires_grad,
            model.device,
            broad_phase_mode,
            shape_collision_group=model.shape_collision_group if hasattr(model, "shape_collision_group") else None,
            shape_world=model.shape_world if hasattr(model, "shape_world") else None,
            shape_flags=model.shape_flags if hasattr(model, "shape_flags") else None,
            sap_sort_type=sap_sort_type,
        )

    def collide(self, model: Model, state: State) -> Contacts:
        """
        Run the collision pipeline using NarrowPhase.

        Args:
            model: The simulation model
            state: The current simulation state

        Returns:
            Contacts: The generated contacts
        """
        # Allocate or clear contacts
        if self.contacts is None or self.requires_grad:
            self.contacts = Contacts(
                self.rigid_contact_max,
                self.soft_contact_max,
                requires_grad=self.requires_grad,
                device=self.device,
            )
        else:
            self.contacts.clear()

        contacts = self.contacts

        # Clear counters
        self.broad_phase_pair_count.zero_()
        self.narrow_contact_count.zero_()
        contacts.rigid_contact_count.zero_()  # Clear since write_contact uses atomic_add

        # Compute AABBs for all shapes
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
                self.rigid_contact_margin,
            ],
            outputs=[
                self.shape_aabb_lower,
                self.shape_aabb_upper,
            ],
            device=self.device,
        )

        # Run broad phase
        if self.broad_phase_mode == BroadPhaseMode.NXN:
            self.nxn_broadphase.launch(
                self.shape_aabb_lower,
                self.shape_aabb_upper,
                model.shape_thickness,
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
                model.shape_thickness,
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
                model.shape_thickness,  # Use thickness as cutoff
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

        # Run narrow phase
        self.narrow_phase.launch(
            candidate_pair=self.broad_phase_shape_pairs,
            num_candidate_pair=self.broad_phase_pair_count,
            geom_types=model.shape_type,
            geom_data=self.geom_data,
            geom_transform=self.geom_transform,
            geom_source=model.shape_source_ptr,
            geom_cutoff=self.geom_cutoff,
            geom_collision_radius=model.shape_collision_radius,
            contact_pair=self.narrow_contact_pair,
            contact_position=self.narrow_contact_position,
            contact_normal=self.narrow_contact_normal,
            contact_penetration=self.narrow_contact_penetration,
            contact_tangent=None,
            contact_count=self.narrow_contact_count,
            device=self.device,
        )

        # Convert NarrowPhase output to Contacts format using write_contact
        wp.launch(
            kernel=convert_narrow_phase_to_contacts_kernel,
            dim=self.rigid_contact_max,
            inputs=[
                self.narrow_contact_pair,
                self.narrow_contact_position,
                self.narrow_contact_normal,
                self.narrow_contact_penetration,
                self.narrow_contact_count,
                self.geom_data,
                model.shape_type,
                state.body_q,
                model.shape_body,
                self.rigid_contact_margin,
                contacts.rigid_contact_max,
            ],
            outputs=[
                contacts.rigid_contact_count,
                contacts.rigid_contact_shape0,
                contacts.rigid_contact_shape1,
                contacts.rigid_contact_point0,
                contacts.rigid_contact_point1,
                contacts.rigid_contact_offset0,
                contacts.rigid_contact_offset1,
                contacts.rigid_contact_normal,
                contacts.rigid_contact_thickness0,
                contacts.rigid_contact_thickness1,
                contacts.rigid_contact_tids,
            ],
            device=self.device,
        )

        return contacts
