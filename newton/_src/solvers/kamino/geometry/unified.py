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

"""
Provides a specialization of Newton's unified collision-detection pipeline for Kamino.

This module provides interfaces and data-conversion specializations for Kamino that wraps
the broad-phase and narrow-phase of Newton's CollisionPipelineUnified, writing generated
contacts data directly into Kamino's respective format.
"""

# Warp imports
import warp as wp
from warp.context import Devicelike

# Newton imports
from ....geometry.broad_phase_nxn import BroadPhaseAllPairs, BroadPhaseExplicit
from ....geometry.broad_phase_sap import BroadPhaseSAP
from ....geometry.collision_core import compute_tight_aabb_from_support
from ....geometry.contact_data import ContactData
from ....geometry.narrow_phase import NarrowPhase
from ....geometry.support_function import GenericShapeData, SupportMapDataProvider, pack_mesh_ptr
from ....geometry.types import GeoType
from ....sim.collide_unified import BroadPhaseMode

# Kamino imports
from ..core.builder import ModelBuilder
from ..core.materials import DEFAULT_FRICTION, DEFAULT_RESTITUTION
from ..core.model import Model, ModelData
from ..core.shapes import ShapeType
from ..core.types import float32, int32, quatf, transformf, uint32, uint64, vec2f, vec2i, vec3f, vec4f
from ..geometry.contacts import (
    DEFAULT_GEOM_PAIR_CONTACT_MARGIN,
    DEFAULT_GEOM_PAIR_MAX_CONTACTS,
    Contacts,
    make_contact_frame_znorm,
)
from ..geometry.keying import build_pair_key2

###
# Module configs
###

wp.set_module_options({"enable_backward": False})


###
# Types
###


@wp.struct
class ContactWriterDataKamino:
    """Contact writer data for writing contacts directly in Kamino format."""

    # Contact limits
    model_max_contacts: int32
    world_max_contacts: wp.array(dtype=int32)

    # Global settings
    default_margin: float32

    # Geometry information arrays
    geom_wid: wp.array(dtype=int32)  # World ID for each geometry
    geom_bid: wp.array(dtype=int32)  # Body ID for each geometry
    geom_mid: wp.array(dtype=int32)  # Material ID for each geometry
    geom_margin: wp.array(dtype=float32)  # Contact margin for each geometry

    # Material properties (indexed by material pair)
    material_friction: wp.array(dtype=float32)
    material_restitution: wp.array(dtype=float32)

    # Output arrays (Kamino Contacts format)
    contacts_model_num_active: wp.array(dtype=int32)
    contacts_world_num_active: wp.array(dtype=int32)
    contact_wid: wp.array(dtype=int32)
    contact_cid: wp.array(dtype=int32)
    contact_gid_AB: wp.array(dtype=vec2i)
    contact_bid_AB: wp.array(dtype=vec2i)
    contact_position_A: wp.array(dtype=vec3f)
    contact_position_B: wp.array(dtype=vec3f)
    contact_gapfunc: wp.array(dtype=vec4f)
    contact_frame: wp.array(dtype=quatf)
    contact_material: wp.array(dtype=vec2f)
    contact_key: wp.array(dtype=uint64)


###
# Functions
###


@wp.func
def convert_kamino_shape_to_newton_geo(sid: int32, params: vec4f) -> tuple[int32, vec3f]:
    """
    Converts Kamino :class:`ShapeType` and parameters to Newton :class:`GeoType` and scale.

    Shape parameter formats:
    - BOX:
        - Newton: half-extents as `scale := (x, y, z)`
        - Kamino: dimensions as `params := (depth, width, height, _)`

    - SPHERE:
        - Newton: radius as `scale := (radius, _, _)`
        - Kamino: radius as `params := (radius, _, _, _)`

    - CAPSULE:
        - Newton: radius and half-height as `scale := (radius, half_height, _)`
        - Kamino: radius and height as `params := (radius, height, _, _)`

    - CYLINDER:
        - Newton: radius and half-height as `scale := (radius, half_height, _)`
        - Kamino: radius and height as `params := (radius, height, _, _)`

    - CONE:
        - Newton: radius and half-height as `scale := (radius, half_height, _)`
        - Kamino: radius and height as `params := (radius, height, _, _)`

    - ELLIPSOID:
        - Newton: semi-axes as `scale := (x, y, z)`
        - Kamino: radii as `params := (a, b, c, _)`

    - PLANE:
        - Newton: half-width in x, half-length in y
        - Kamino: normal and distance as `params := (normal_x, normal_y, normal_z, distance)`

    See :class:`GenericShapeData` in :file:`support_function.py` for further details.

    Args:
        sid (int32):
            The Kamino ShapeType as :class:`int32`, i.e. the shape index.
        params(vec4f):
            Kamino shape parameters as :class:`vec4f`.

    Returns:
        (int32, vec3f):
        A tuple containing the corresponding Newton :class:`GeoType`
        as an :class:`int32`, and the shape scale as a :class:`vec3f`.
    """
    geo_type = int32(GeoType.NONE)
    scale = vec3f(0.0)

    if sid == ShapeType.SPHERE:
        # Kamino: (radius, 0, 0, 0) -> Newton: (radius, ?, ?)
        geo_type = GeoType.SPHERE
        scale = vec3f(params[0], 0.0, 0.0)

    elif sid == ShapeType.BOX:
        # Kamino: (depth, width, height) full size -> Newton: half-extents
        geo_type = GeoType.BOX
        scale = vec3f(params[0] * 0.5, params[1] * 0.5, params[2] * 0.5)

    elif sid == ShapeType.CAPSULE:
        # Kamino: (radius, height) full height -> Newton: (radius, half-height, ?)
        geo_type = GeoType.CAPSULE
        scale = vec3f(params[0], params[1] * 0.5, 0.0)

    elif sid == ShapeType.CYLINDER:
        # Kamino: (radius, height) full height -> Newton: (radius, half-height, ?)
        geo_type = GeoType.CYLINDER
        scale = vec3f(params[0], params[1] * 0.5, 0.0)

    elif sid == ShapeType.CONE:
        # Kamino: (radius, height) full height -> Newton: (radius, half-height, ?)
        geo_type = GeoType.CONE
        scale = vec3f(params[0], params[1] * 0.5, 0.0)

    elif sid == ShapeType.ELLIPSOID:
        # Kamino: (a, b, c) semi-axes -> Newton: same
        geo_type = GeoType.ELLIPSOID
        scale = vec3f(params[0], params[1], params[2])

    elif sid == ShapeType.PLANE:
        # NOTE: For an infinite plane, we use (0, 0, _) to signal an infinite extents
        geo_type = GeoType.PLANE
        scale = vec3f(0.0, 0.0, 0.0)  # Infinite plane

    # TODO: Implement MESH, CONVEX, HFIELD, SDF
    # elif sid == ShapeType.MESH:
    #     geo_type = GeoType.MESH
    #     scale = vec3f(0.0, 0.0, 0.0)
    # elif sid == ShapeType.CONVEX:
    #     geo_type = GeoType.CONVEX_MESH
    #     scale = vec3f(0.0, 0.0, 0.0)
    # elif sid == ShapeType.HFIELD:
    #     geo_type = GeoType.HFIELD
    #     scale = vec3f(0.0, 0.0, 0.0)
    # elif sid == ShapeType.SDF:
    #     geo_type = GeoType.SDF
    #     scale = vec3f(0.0, 0.0, 0.0)

    return geo_type, scale


@wp.func
def write_contact_unified_kamino(
    contact_data: ContactData,
    writer_data: ContactWriterDataKamino,
):
    """
    Write a contact to Kamino-compatible output arrays.

    This function is used as a custom contact writer for NarrowPhase.launch_custom_write().
    It converts ContactData from the narrow phase directly to Kamino's contact format.

    Args:
        contact_data: ContactData struct from narrow phase containing contact information
        writer_data: ContactWriterDataKamino struct containing output arrays
    """
    total_separation_needed = (
        contact_data.radius_eff_a + contact_data.radius_eff_b + contact_data.thickness_a + contact_data.thickness_b
    )

    # Normalize contact normal
    contact_normal_a_to_b = wp.normalize(contact_data.contact_normal_a_to_b)

    # Compute contact points on each shape
    a_contact_world = contact_data.contact_point_center - contact_normal_a_to_b * (
        0.5 * contact_data.contact_distance + contact_data.radius_eff_a
    )
    b_contact_world = contact_data.contact_point_center + contact_normal_a_to_b * (
        0.5 * contact_data.contact_distance + contact_data.radius_eff_b
    )

    # Calculate penetration distance
    diff = b_contact_world - a_contact_world
    distance = wp.dot(diff, contact_normal_a_to_b)
    d = distance - total_separation_needed

    # Use per-shape contact margin (max of both shapes)
    margin_a = writer_data.geom_margin[contact_data.shape_a]
    margin_b = writer_data.geom_margin[contact_data.shape_b]
    margin = wp.max(writer_data.default_margin, wp.max(margin_a, margin_b))

    # Only write contact if within margin
    if d < margin:
        # Retrieve the geom/body/material indices
        gid_a = contact_data.shape_a
        gid_b = contact_data.shape_b
        bid_a = writer_data.geom_bid[contact_data.shape_a]
        bid_b = writer_data.geom_bid[contact_data.shape_b]
        # TODO: mid_a = writer_data.geom_mid[contact_data.shape_a]
        # TODO: mid_b = writer_data.geom_mid[contact_data.shape_b]

        # TODO: Check this logic: Are we sure this is guaranteed by the broadphase?
        # Assume both geoms are in same world
        wid = writer_data.geom_wid[contact_data.shape_a]

        # Retrieve the max contacts of the corresponding world
        world_max_contacts = writer_data.world_max_contacts[wid]

        # Atomically increment contact counts
        mcid = wp.atomic_add(writer_data.contacts_model_num_active, 0, 1)
        wcid = wp.atomic_add(writer_data.contacts_world_num_active, wid, 1)

        # If within the max contact allocations, write the new contact
        if mcid < writer_data.model_max_contacts and wcid < world_max_contacts:
            # Perform A/B geom and body assignment,
            # ensuring static bodies is always body A
            # NOTE: We want the normal to always point from A to B,
            # and hence body B to be the "effected" body in the contact
            # so we have to ensure that bid_B is always non-negative
            if bid_b < 0:
                gid_AB = vec2i(gid_b, gid_a)
                bid_AB = vec2i(bid_b, bid_a)
                normal = -contact_normal_a_to_b
                pos_A = b_contact_world
                pos_B = a_contact_world
            else:
                gid_AB = vec2i(gid_a, gid_b)
                bid_AB = vec2i(bid_a, bid_b)
                normal = contact_normal_a_to_b
                pos_A = a_contact_world
                pos_B = b_contact_world

            # TODO: Change this to extract from material-pair data
            # Currently we only have a single material in the pipeline (i.e. the default)
            friction = writer_data.material_friction[0]
            restitution = writer_data.material_restitution[0]
            material = vec2f(friction, restitution)

            # Generate the gap-function (normal.x, normal.y, normal.z, distance),
            # contact frame (z-norm aligned with contact normal)
            gapfunc = vec4f(normal[0], normal[1], normal[2], d)
            q_frame = wp.quat_from_matrix(make_contact_frame_znorm(normal))
            key = build_pair_key2(uint32(gid_AB[0]), uint32(gid_AB[1]))

            # Store contact data in Kamino format
            writer_data.contact_wid[mcid] = wid
            writer_data.contact_cid[mcid] = wcid
            writer_data.contact_gid_AB[mcid] = gid_AB
            writer_data.contact_bid_AB[mcid] = bid_AB
            writer_data.contact_position_A[mcid] = pos_A
            writer_data.contact_position_B[mcid] = pos_B
            writer_data.contact_gapfunc[mcid] = gapfunc
            writer_data.contact_frame[mcid] = q_frame
            writer_data.contact_material[mcid] = material
            writer_data.contact_key[mcid] = key

        # TODO: Isnt it possible that this will create 'bubbles' of unused contacts?
        # TODO: We may need an flaging mechanism to indicate invalid contacts
        # Otherwise roll-back the atomic add if we exceeded limits
        else:
            wp.atomic_sub(writer_data.contacts_model_num_active, 0, 1)
            wp.atomic_sub(writer_data.contacts_world_num_active, wid, 1)


###
# Kernels
###


@wp.kernel
def _convert_geom_data_kamino_to_newton(
    # Inputs:
    default_margin: float32,
    geom_sid: wp.array(dtype=int32),
    geom_params: wp.array(dtype=vec4f),
    # Outputs:
    geom_margin: wp.array(dtype=float32),
    geom_type: wp.array(dtype=int32),
    geom_data: wp.array(dtype=vec4f),
):
    """
    Converts Kamino geometry :class:`ShapeType` and parameters to Newton :class:`GeoType` and scale.

    Inputs:
        geom_sid (wp.array): Array of Kamino shape indices corresponding to :class:`ShapeType` values.
        geom_params (wp.array): Array of Kamino shape parameters.

    Outputs:
        geom_type (wp.array): Array for Newton geometry indices corresponding to :class:`GeoType` values.
        geom_data (wp.array): Array for Newton geometry data (scale and thickness).
    """
    # Retrieve the geometry index from the thread grid
    gid = wp.tid()

    # Retrieve the geom-specific data
    sid = geom_sid[gid]
    params = geom_params[gid]
    margin = geom_margin[gid]

    # NOTE: Thickness is not currently used in Kamino; set to zero
    thickness = float32(0.0)

    # Convert Kamino ShapeType to Newton GeoType and transform params to Newton scale
    geo_type, scale = convert_kamino_shape_to_newton_geo(sid, params)

    # Store converted geometry data
    # NOTE: the per-geom margin is overridden because
    # the unified pipeline needs it during narrow-phase
    geom_type[gid] = geo_type
    geom_data[gid] = vec4f(scale[0], scale[1], scale[2], thickness)
    geom_margin[gid] = wp.max(default_margin, margin)


@wp.kernel
def _update_geom_poses_and_compute_aabbs(
    # Inputs:
    geom_type: wp.array(dtype=int32),
    geom_data: wp.array(dtype=vec4f),
    geom_bid: wp.array(dtype=int32),
    geom_ptr: wp.array(dtype=wp.uint64),
    geom_offset: wp.array(dtype=transformf),
    geom_margin: wp.array(dtype=float32),
    geom_collision_radius: wp.array(dtype=float32),
    body_pose: wp.array(dtype=transformf),
    # Outputs:
    geom_pose: wp.array(dtype=transformf),
    shape_aabb_lower: wp.array(dtype=vec3f),
    shape_aabb_upper: wp.array(dtype=vec3f),
):
    """
    Updates the pose of each Kamino geometry in world coordinates and computes its axis-aligned bounding box (AABB).

    Notes:
        Uses the support function for most shapes.
        Infinite planes and meshes use bounding sphere fallback.
        AABBs are enlarged by per-shape contact margin for contact detection.

    Inputs:
        geom_bid (wp.array): Array of body indices for each geometry.
        geom_type (wp.array): Array of geometry type indices corresponding to :class:`GeoType` values.
        geom_data (wp.array): Array of geometry data (scale and thickness).
        geom_ptr (wp.array): Array of geometry pointers (used for mesh shapes).
        geom_offset (wp.array): Array of geometry local pose offset transforms w.r.t the associated body.
        geom_contact_margin (wp.array): Array of per-geometry contact margins.
        geom_collision_radius (wp.array): Array of geometry collision bounding sphere radii.
        body_pose (wp.array): Array of body poses in world coordinates.

    Outputs:
        geom_pose (wp.array): Array of geometry poses in world coordinates.
        shape_aabb_lower (wp.array): Array of lower bounds of geometry axis-aligned bounding boxes.
        shape_aabb_upper (wp.array): Array of upper bounds of geometry axis-aligned bounding boxes.
    """
    # Retrieve the geometry index from the thread grid
    gid = wp.tid()

    # Retrieve the geom-specific data
    geo_type = geom_type[gid]
    geo_data = geom_data[gid]
    bid = geom_bid[gid]
    margin = geom_margin[gid]
    X_bg = geom_offset[gid]

    # Retrieve the pose of the corresponding body
    X_b = wp.transform_identity(dtype=float32)
    if bid > -1:
        X_b = body_pose[bid]

    # Compute the geometry pose in world coordinates
    X_g = wp.transform_multiply(X_b, X_bg)

    # Decompose geometry world transform
    r_g = wp.transform_get_translation(X_g)
    q_g = wp.transform_get_rotation(X_g)

    # Extract geometry scale from the geo_data
    # NOTE: Format is (vec3f scale, float32 thickness)
    scale = vec3f(geo_data[0], geo_data[1], geo_data[2])

    # Enlarge AABB by per-shape contact margin for contact detection
    margin_vec = wp.vec3(margin, margin, margin)

    # Check if this is an infinite plane or mesh - use bounding sphere fallback
    is_infinite_plane = (geo_type == GeoType.PLANE) and (scale[0] == 0.0 and scale[1] == 0.0)
    is_mesh = geo_type == GeoType.MESH
    is_sdf = geo_type == GeoType.SDF

    # Compute the geometry AABB in world coordinates
    aabb_lower = wp.vec3(0.0)
    aabb_upper = wp.vec3(0.0)
    if is_infinite_plane or is_mesh or is_sdf:
        # Use conservative bounding sphere approach
        radius = geom_collision_radius[gid]
        half_extents = wp.vec3(radius, radius, radius)
        aabb_lower = r_g - half_extents - margin_vec
        aabb_upper = r_g + half_extents + margin_vec
    else:
        # Use support function to compute tight AABB
        shape_data = GenericShapeData()
        shape_data.shape_type = geo_type
        shape_data.scale = scale
        shape_data.auxiliary = wp.vec3(0.0, 0.0, 0.0)

        # For CONVEX_MESH, pack the mesh pointer
        if geo_type == GeoType.CONVEX_MESH:
            shape_data.auxiliary = pack_mesh_ptr(geom_ptr[gid])

        # Compute tight AABB using helper function
        data_provider = SupportMapDataProvider()
        aabb_min_world, aabb_max_world = compute_tight_aabb_from_support(shape_data, q_g, r_g, data_provider)
        aabb_lower = aabb_min_world - margin_vec
        aabb_upper = aabb_max_world + margin_vec

    # Store the updated geometry pose in world coordinates and computed AABB
    geom_pose[gid] = X_g
    shape_aabb_lower[gid] = aabb_lower
    shape_aabb_upper[gid] = aabb_upper


###
# Interfaces
###


class CollisionPipelineUnifiedKamino:
    """
    A specialization of the Newton's unified collision detection pipeline for Kamino.

    This pipeline uses the same broad phase algorithms (NXN, SAP, EXPLICIT) and narrow phase
    (NarrowPhase with GJK/MPR) as Newton's CollisionPipelineUnified, but writes contacts
    directly in Kamino's format using a custom contact writer.
    """

    def __init__(
        self,
        model: Model,
        builder: ModelBuilder,
        broadphase: BroadPhaseMode = BroadPhaseMode.EXPLICIT,
        max_contacts: int | None = None,
        max_contacts_per_pair: int = DEFAULT_GEOM_PAIR_MAX_CONTACTS,
        max_triangle_pairs: int = 1_000_000,
        default_margin: float = DEFAULT_GEOM_PAIR_CONTACT_MARGIN,
        default_friction: float = DEFAULT_FRICTION,
        default_restitution: float = DEFAULT_RESTITUTION,
        device: Devicelike = None,
    ):
        """
        Initialize an instance of Kamino's wrapper of the unified collision detection pipeline.

        Args:
            builder (ModelBuilder): Kamino ModelBuilder (used to extract collision pair information)
            broadphase (BroadPhaseMode): Broad-phase back-end to use (NXN, SAP, or EXPLICIT)
            max_contacts (int | None): Maximum contacts for the entire model (overrides computed value)
            max_contacts_per_pair (int): Maximum contacts per colliding geometry pair
            max_triangle_pairs (int): Maximum triangle pairs for mesh/mesh and mesh/hfield collisions
            default_margin (float): Default contact margin for collision detection
            default_friction (float): Default contact friction coefficient
            default_restitution (float): Default impact restitution coefficient
            device (Devicelike): Warp device used to allocate memory and operate on
        """
        # Set the target Warp device for the pipeline
        # If not specified explicitly, use the device of the model
        self._device: Devicelike = None
        if device is not None:
            self._device = device
        else:
            self._device = model.device

        # Cache pipeline settings
        self._broadphase: BroadPhaseMode = broadphase
        self._default_margin: float = default_margin
        self._default_friction: float = default_friction
        self._default_restitution: float = default_restitution
        self._max_contacts_per_pair: int = max_contacts_per_pair
        self._max_triangle_pairs: int = max_triangle_pairs

        # Get geometry count from builder
        self._num_geoms: int = builder.num_collision_geoms

        # Compute the maximum possible number of geom pairs (worst-case, needed for NXN/SAP)
        self._max_shape_pairs: int = (self._num_geoms * (self._num_geoms - 1)) // 2
        self._max_contacts: int = self._max_shape_pairs * self._max_contacts_per_pair

        # Override max contacts if specified explicitly
        if max_contacts is not None:
            self._max_contacts = max_contacts

        # Build shape pairs for EXPLICIT mode
        self.shape_pairs_filtered: wp.array | None = None
        if broadphase == BroadPhaseMode.EXPLICIT:
            _, model_filtered_geom_pairs, _, _ = builder.make_collision_candidate_pairs()
            self.shape_pairs_filtered = wp.array(model_filtered_geom_pairs, dtype=vec2i, device=self._device)
            self._max_shape_pairs = len(model_filtered_geom_pairs)
            self._max_contacts = self._max_shape_pairs * self._max_contacts_per_pair

        # Build collision group array for NXN/SAP modes
        # Newton's broad phase uses a simpler collision group system than Kamino's bitmask approach
        # For NXN/SAP, we use a simple group ID (1 = collide with all, 0 = no collision)
        # The actual filtering based on Kamino's group/collides bitmasks is done in EXPLICIT mode
        # For NXN/SAP, we allow all pairs and let the narrow phase handle the actual collision
        geom_collision_group_list = [1] * self._num_geoms  # All geometries can collide

        # Capture a reference to per-geometry world indices already present in the model
        self.geom_wid: wp.array = model.cgeoms.wid

        # Allocate internal data needed by the pipeline that
        # the Kamino model and data do not yet provide
        with wp.ScopedDevice(self._device):
            self.geom_type = wp.zeros(self._num_geoms, dtype=int32)
            self.geom_data = wp.zeros(self._num_geoms, dtype=vec4f)
            self.geom_collision_group = wp.array(geom_collision_group_list, dtype=int32)
            self.geom_collision_radius = wp.zeros(self._num_geoms, dtype=float32)
            self.shape_aabb_lower = wp.zeros(self._num_geoms, dtype=wp.vec3)
            self.shape_aabb_upper = wp.zeros(self._num_geoms, dtype=wp.vec3)
            self.broad_phase_pair_count = wp.zeros(1, dtype=wp.int32)
            self.broad_phase_shape_pairs = wp.zeros(self._max_shape_pairs, dtype=wp.vec2i)
            self.material_friction = wp.full(1, self._default_friction, dtype=float32)
            self.material_restitution = wp.full(1, self._default_restitution, dtype=float32)

        # Initialize the broad-phase backend depending on the selected mode
        match self._broadphase:
            case BroadPhaseMode.NXN:
                self.nxn_broadphase = BroadPhaseAllPairs(self.geom_wid, shape_flags=None, device=self._device)
            case BroadPhaseMode.SAP:
                self.sap_broadphase = BroadPhaseSAP(self.geom_wid, shape_flags=None, device=self._device)
            case BroadPhaseMode.EXPLICIT:
                self.explicit_broadphase = BroadPhaseExplicit()
            case _:
                raise ValueError(f"Unsupported broad phase mode: {self._broadphase}")

        # Initialize narrow-phase backend with the contact writer customized for Kamino
        self.narrow_phase = NarrowPhase(
            max_candidate_pairs=self._max_shape_pairs,
            max_triangle_pairs=self._max_triangle_pairs,
            device=self._device,
            shape_aabb_lower=self.shape_aabb_lower,
            shape_aabb_upper=self.shape_aabb_upper,
            contact_writer_warp_func=write_contact_unified_kamino,
        )

        # Convert geometry data from Kamino to Newton format
        self._convert_geometry_data(model)

    ###
    # Properties
    ###

    @property
    def device(self) -> Devicelike:
        """Returns the Warp device the pipeline operates on."""
        return self._device

    ###
    # Operations
    ###

    def collide(self, model: Model, data: ModelData, contacts: Contacts):
        """
        Runs the unified collision detection pipeline to generate discrete contacts.

        Args:
            model (Model): The model container holding the time-invariant parameters of the simulation.
            data (ModelData): The data container holding the time-varying state of the simulation.
            contacts (Contacts): Output contacts container (will be cleared and populated)
        """
        # Check if contacts is allocated on the same device
        if contacts.device != self._device:
            raise ValueError(
                f"Contacts container device ({contacts.device}) does not match the pipeline device ({self._device})."
            )

        # Check if contacts can hold the maximum number of contacts
        if contacts.num_model_max_contacts < self._max_contacts:
            raise ValueError(
                f"Contacts container has insufficient capacity "
                f"({contacts.num_model_max_contacts} < {self._max_contacts}) "
                f"to hold all possible contacts generated by the pipeline."
            )

        # Clear contacts
        contacts.clear()

        # Update geometry poses from body states and compute respective AABBs
        self._update_geom_data(model, data)

        # Run broad-phase collision detection to get candidate shape pairs
        self._run_broadphase()

        # Run narrow-phase collision detection to generate contacts
        self._run_narrowphase(model, data, contacts)

    ###
    # Internals
    ###

    def _convert_geometry_data(self, model: Model):
        """
        Converts Kamino geometry data to the Newton format.

        This operation needs to be called only once during initialization.

        Args:
            model (Model): The model container holding the time-invariant parameters of the simulation.
        """
        wp.launch(
            kernel=_convert_geom_data_kamino_to_newton,
            dim=self._num_geoms,
            inputs=[
                self._default_margin,
                model.cgeoms.sid,
                model.cgeoms.params,
            ],
            outputs=[
                model.cgeoms.margin,
                self.geom_type,
                self.geom_data,
            ],
            device=self._device,
        )

    def _set_model_materials(self, model: Model):
        """
        Update material arrays from the model's material pairs.

        Args:
            model (Model): The Kamino model containing material-pair properties.
        """
        # TODO: Fix this to:
        # 1. handle default material if no mpairs exist
        # 2. copy the per material-pair properties from the model-builder
        if model.mpairs is not None and model.mpairs.num_pairs > 0:
            # Reallocate material arrays if needed
            if self.material_friction.shape[0] != model.mpairs.num_pairs:
                self.material_friction = wp.zeros(model.mpairs.num_pairs, dtype=float32, device=self._device)
                self.material_restitution = wp.zeros(model.mpairs.num_pairs, dtype=float32, device=self._device)

            # Copy material properties
            wp.copy(self.material_friction, model.mpairs.dynamic_friction)
            wp.copy(self.material_restitution, model.mpairs.restitution)

    def _update_geom_data(self, model: Model, data: ModelData):
        """
        Updates geometry poses from corresponding body states and computes respective AABBs.

        Args:
            model (Model): The model container holding the time-invariant parameters of the simulation.
            data (ModelData): The data container holding the time-varying state of the simulation.
        """
        wp.launch(
            kernel=_update_geom_poses_and_compute_aabbs,
            dim=self._num_geoms,
            inputs=[
                self.geom_type,
                self.geom_data,
                model.cgeoms.bid,
                model.cgeoms.ptr,
                model.cgeoms.offset,
                model.cgeoms.margin,
                self.geom_collision_radius,
                data.bodies.q_i,
            ],
            outputs=[
                data.cgeoms.pose,
                self.shape_aabb_lower,
                self.shape_aabb_upper,
            ],
            device=self._device,
        )

    def _run_broadphase(self):
        """
        Runs broad-phase collision detection to generate candidate geom/shape pairs.
        """
        # First clear broad phase counter
        self.broad_phase_pair_count.zero_()

        # Then launch the configured broad-phase collision detection
        match self._broadphase:
            case BroadPhaseMode.NXN:
                self.nxn_broadphase.launch(
                    self.shape_aabb_lower,
                    self.shape_aabb_upper,
                    None,  # AABBs are pre-expanded
                    self.geom_collision_group,  # Simple collision groups (all = 1)
                    self.geom_wid,
                    self._num_geoms,
                    self.broad_phase_shape_pairs,
                    self.broad_phase_pair_count,
                    device=self._device,
                )
            case BroadPhaseMode.SAP:
                self.sap_broadphase.launch(
                    self.shape_aabb_lower,
                    self.shape_aabb_upper,
                    None,  # AABBs are pre-expanded
                    self.geom_collision_group,  # Simple collision groups (all = 1)
                    self.geom_wid,
                    self._num_geoms,
                    self.broad_phase_shape_pairs,
                    self.broad_phase_pair_count,
                    device=self._device,
                )
            case BroadPhaseMode.EXPLICIT:
                self.explicit_broadphase.launch(
                    self.shape_aabb_lower,
                    self.shape_aabb_upper,
                    None,  # AABBs are pre-expanded
                    self.shape_pairs_filtered,
                    len(self.shape_pairs_filtered),
                    self.broad_phase_shape_pairs,
                    self.broad_phase_pair_count,
                    device=self._device,
                )
            case _:
                raise ValueError(f"Unsupported broad phase mode: {self._broadphase}")

    def _run_narrowphase(self, model: Model, data: ModelData, contacts: Contacts):
        """
        Runs narrow-phase collision detection to generate contacts.

        Args:
            model (Model): The model container holding the time-invariant parameters of the simulation.
            data (ModelData): The data container holding the time-varying state of the simulation.
            contacts (Contacts): Output contacts container (will be populated by this function)
        """
        # Create a writer data struct to bundle all necessary input/output
        # arrays into a single object for the narrow phase custom writer
        # NOTE: Unfortunately, we need to do this on every call in python,
        # but graph-capture ensures this actually happens only once
        writer_data = ContactWriterDataKamino()
        writer_data.model_max_contacts = int32(contacts.num_model_max_contacts)
        writer_data.world_max_contacts = contacts.world_max_contacts
        writer_data.default_margin = float32(self._default_margin)
        writer_data.geom_bid = model.cgeoms.bid
        writer_data.geom_wid = model.cgeoms.wid
        writer_data.geom_mid = model.cgeoms.mid
        writer_data.geom_margin = model.cgeoms.margin
        writer_data.material_friction = self.material_friction
        writer_data.material_restitution = self.material_restitution
        writer_data.contacts_model_num_active = contacts.model_num_contacts
        writer_data.contacts_world_num_active = contacts.world_num_contacts
        writer_data.contact_wid = contacts.wid
        writer_data.contact_cid = contacts.cid
        writer_data.contact_gid_AB = contacts.gid_AB
        writer_data.contact_bid_AB = contacts.bid_AB
        writer_data.contact_position_A = contacts.position_A
        writer_data.contact_position_B = contacts.position_B
        writer_data.contact_gapfunc = contacts.gapfunc
        writer_data.contact_frame = contacts.frame
        writer_data.contact_material = contacts.material
        writer_data.contact_key = contacts.key

        # Run narrow phase with the custom Kamino contact writer
        self.narrow_phase.launch_custom_write(
            candidate_pair=self.broad_phase_shape_pairs,
            num_candidate_pair=self.broad_phase_pair_count,
            shape_types=self.geom_type,
            shape_data=self.geom_data,
            shape_transform=data.cgeoms.pose,
            shape_source=model.cgeoms.ptr,
            shape_contact_margin=model.cgeoms.margin,
            shape_collision_radius=self.geom_collision_radius,
            writer_data=writer_data,
            device=self._device,
        )
