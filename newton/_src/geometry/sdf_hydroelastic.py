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

"""SDF-based hydroelastic contact generation.

This module implements hydroelastic contact modeling between shapes represented
by Signed Distance Fields (SDFs). Hydroelastic contacts model compliant surfaces
where contact force is distributed over a contact patch rather than point contacts.

**Pipeline Overview:**

1. **Broadphase**: OBB intersection tests between SDF shape pairs
2. **Octree Refinement**: Hierarchical subdivision (8x8x8 → 4x4x4 → 2x2x2 → voxels)
   to find iso-voxels where the zero-isosurface between SDFs exists
3. **Marching Cubes**: Extract contact surface triangles from iso-voxels
4. **Contact Generation**: Generate contacts at triangle centroids with force
   proportional to penetration depth and surface area
5. **Contact Reduction**: Reduce contacts via ``HydroelasticContactReduction``

**Usage:**

Configure shapes with ``ShapeConfig(hydroelastic_type="compliant", kh=1e9)`` and
pass :class:`HydroelasticSDF.Config` to the collision pipeline.

See Also:
    :class:`HydroelasticSDF.Config`: Configuration options for this module.
    :class:`HydroelasticContactReduction`: Contact reduction for hydroelastic contacts.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any

import warp as wp

from newton._src.core.types import MAXVAL, Devicelike

from ..sim.model import Model
from ..utils.heightfield import HeightfieldData, sample_sdf_heightfield
from . import hydroelastic_pressure_fields
from .collision_core import sat_box_intersection
from .contact_data import ContactData
from .contact_reduction import get_slot
from .contact_reduction_global import (
    GlobalContactReducerData,
    decode_oct,
    encode_oct,
    make_contact_key,
)
from .contact_reduction_hydroelastic import (
    HydroelasticContactReduction,
    HydroelasticReductionConfig,
    export_hydroelastic_contact_to_buffer,
)
from .flags import HydroelasticContactWorkflow, ShapeFlags
from .hashtable import hashtable_find_or_insert
from .hydroelastic_runtime_adapter import collect_hydroelastic_runtime_data
from .hydroelastic_workflow import resolve_pair_contact_workflow
from .sdf_mc import (
    MC_DEGENERATE_N_SQ_EPS,
    MC_EDGE_CLAMP_MAX,
    MC_EDGE_CLAMP_MIN,
    MC_EDGE_VAL_DIFF_EPS,
    get_mc_tables,
    get_triangle_fraction,
)
from .sdf_texture import TextureSDFData, texture_sample_sdf, texture_sample_sdf_at_voxel
from .sdf_contact import sample_sdf_extrapolated
from .kernels import (
    sdf_box,
    sdf_capsule,
    sdf_cone,
    sdf_cylinder,
    sdf_ellipsoid,
    sdf_mesh,
    sdf_sphere,
)
from .sdf_contact import sample_sdf_extrapolated
from .sdf_mc import get_mc_tables, mc_calc_face
from .sdf_utils import SDFData
from .types import GeoType
from .utils import scan_with_total

vec8f = wp.types.vector(length=8, dtype=wp.float32)
PRE_PRUNE_MAX_PENETRATING = 2

HYDROELASTIC_MODE_NONE = wp.int32(0)
HYDROELASTIC_MODE_RIGID = wp.int32(1)
HYDROELASTIC_MODE_COMPLIANT = wp.int32(2)
HYDROELASTIC_WORKFLOW_CLASSIC = wp.int32(int(HydroelasticContactWorkflow.CLASSIC))
HYDROELASTIC_WORKFLOW_PRESSURE = wp.int32(int(HydroelasticContactWorkflow.PRESSURE))

# Backward-compatible re-exports for internal tests and downstream callers.
PressureFieldData = hydroelastic_pressure_fields.PressureFieldData
create_empty_pressure_field_data = hydroelastic_pressure_fields.create_empty_pressure_field_data
sample_geometry_sdf_grid_kernel = hydroelastic_pressure_fields.sample_geometry_sdf_grid_kernel
sample_sdf_grid_kernel = hydroelastic_pressure_fields.sample_sdf_grid_kernel
_apply_pressure_axis_sine_modulation_to_grid = hydroelastic_pressure_fields._apply_pressure_axis_sine_modulation_to_grid
_compute_pressure_grid_spec = hydroelastic_pressure_fields._compute_pressure_grid_spec
_normalize_axis_triplet = hydroelastic_pressure_fields._normalize_axis_triplet
_solve_poisson_pressure_extent = hydroelastic_pressure_fields._solve_poisson_pressure_extent


@wp.kernel
def map_shape_sdf_data_kernel(
    sdf_data: wp.array(dtype=SDFData),
    shape_sdf_index: wp.array(dtype=wp.int32),
    out_shape_sdf_data: wp.array(dtype=SDFData),
):
    """Map compact SDF table entries to per-shape SDFData."""
    shape_idx = wp.tid()
    sdf_idx = shape_sdf_index[shape_idx]
    if sdf_idx < 0:
        out_shape_sdf_data[shape_idx].sparse_sdf_ptr = wp.uint64(0)
        out_shape_sdf_data[shape_idx].sparse_voxel_size = wp.vec3(0.0, 0.0, 0.0)
        out_shape_sdf_data[shape_idx].sparse_voxel_radius = 0.0
        out_shape_sdf_data[shape_idx].coarse_sdf_ptr = wp.uint64(0)
        out_shape_sdf_data[shape_idx].coarse_voxel_size = wp.vec3(0.0, 0.0, 0.0)
        out_shape_sdf_data[shape_idx].center = wp.vec3(0.0, 0.0, 0.0)
        out_shape_sdf_data[shape_idx].half_extents = wp.vec3(0.0, 0.0, 0.0)
        out_shape_sdf_data[shape_idx].background_value = MAXVAL
        out_shape_sdf_data[shape_idx].scale_baked = False
    else:
        out_shape_sdf_data[shape_idx] = sdf_data[sdf_idx]


@wp.kernel
def map_shape_pressure_data_kernel(
    pressure_data: wp.array(dtype=PressureFieldData),
    shape_pressure_index: wp.array(dtype=wp.int32),
    out_shape_pressure_data: wp.array(dtype=PressureFieldData),
):
    """Map compact pressure field table entries to per-shape PressureFieldData."""
    shape_idx = wp.tid()
    pressure_idx = shape_pressure_index[shape_idx]
    if pressure_idx < 0:
        out_shape_pressure_data[shape_idx].pressure_ptr = wp.uint64(0)
        out_shape_pressure_data[shape_idx].pressure_max = 0.0
    else:
        out_shape_pressure_data[shape_idx] = pressure_data[pressure_idx]
@wp.func
def int_to_vec3f(x: wp.int32, y: wp.int32, z: wp.int32):
    return wp.vec3f(float(x), float(y), float(z))


@wp.func
def get_effective_stiffness(k_a: wp.float32, k_b: wp.float32) -> wp.float32:
    """Compute effective stiffness for two materials in series."""
    denom = k_a + k_b
    if denom <= 0.0:
        return 0.0
    return (k_a * k_b) / denom


@wp.func
def mc_calc_face_texture(
    flat_edge_verts_table: wp.array[wp.vec2ub],
    corner_offsets_table: wp.array[wp.vec3ub],
    tri_range_start: wp.int32,
    corner_vals: vec8f,
    corner_sdf_vals: vec8f,
    sdf_a: TextureSDFData,
    x_id: wp.int32,
    y_id: wp.int32,
    z_id: wp.int32,
) -> tuple[float, wp.vec3, wp.vec3, float, wp.mat33f]:
    """Extract a triangle face from a marching cubes voxel using texture SDF.

    Vertex positions are returned in the SDF's local coordinate space.

    A tiny thickness (1e-4 x voxel_radius) biases the signed-distance depth
    just enough to classify touching-surface vertices as penetrating.  The
    resulting phantom force is negligible (< 0.1 % of typical contact forces)
    but prevents zero-area contacts at exactly-touching surfaces.
    """
    thickness = sdf_a.voxel_radius * 1.0e-4

    face_verts = wp.mat33f()
    vert_depths = wp.vec3f()
    num_inside = wp.int32(0)
    for vi in range(3):
        edge_verts = wp.vec2i(flat_edge_verts_table[tri_range_start + vi])
        v_idx_from = edge_verts[0]
        v_idx_to = edge_verts[1]
        val_0 = wp.float32(corner_vals[v_idx_from])
        val_1 = wp.float32(corner_vals[v_idx_to])

        p_0 = wp.vec3f(corner_offsets_table[v_idx_from])
        p_1 = wp.vec3f(corner_offsets_table[v_idx_to])
        val_diff = wp.float32(val_1 - val_0)
        if wp.abs(val_diff) < wp.static(MC_EDGE_VAL_DIFF_EPS):
            t = float(0.5)
        else:
            # Clamp t away from cube corners to prevent vertex collapse when
            # corner values are near zero (e.g. at SDF ridge boundaries where
            # both shapes share the same nearest face).  Without the clamp,
            # t close to 0 or 1 places multiple vertices at the same corner,
            # producing degenerate (zero-area) triangles.
            t = wp.clamp((0.0 - val_0) / val_diff, wp.static(MC_EDGE_CLAMP_MIN), wp.static(MC_EDGE_CLAMP_MAX))
        p = p_0 + t * (p_1 - p_0)
        vol_idx = p + int_to_vec3f(x_id, y_id, z_id)
        local_pos = sdf_a.sdf_box_lower + wp.cw_mul(vol_idx, sdf_a.voxel_size)
        face_verts[vi] = local_pos
        # Interpolate SDF depth from cached corner values (avoids texture lookup)
        sdf_from = wp.float32(corner_sdf_vals[v_idx_from])
        sdf_to = wp.float32(corner_sdf_vals[v_idx_to])
        depth = sdf_from + t * (sdf_to - sdf_from) - thickness
        vert_depths[vi] = depth
        if depth < 0.0:
            num_inside += 1

    n = wp.cross(face_verts[1] - face_verts[0], face_verts[2] - face_verts[0])
    n_sq = wp.dot(n, n)
    if n_sq < wp.static(MC_DEGENERATE_N_SQ_EPS):
        # Degenerate triangle — return zero area with a valid (non-NaN) normal.
        area = 0.0
        normal = wp.vec3(0.0, 0.0, 1.0)
    else:
        n_len = wp.sqrt(n_sq)
        normal = n / n_len
        area = n_len / 2.0
    center = (face_verts[0] + face_verts[1] + face_verts[2]) / 3.0
    pen_depth = (vert_depths[0] + vert_depths[1] + vert_depths[2]) / 3.0
    area *= get_triangle_fraction(vert_depths, num_inside)
    return area, normal, center, pen_depth, face_verts


def hydroelastic_mode_from_flags(flags: wp.int32) -> wp.int32:
    """Decode hydroelastic mode from shape flags inside Warp kernels."""
    if (flags & wp.int32(ShapeFlags.HYDROELASTIC_RIGID)) != 0:
        return HYDROELASTIC_MODE_RIGID
    if (flags & wp.int32(ShapeFlags.HYDROELASTIC_COMPLIANT)) != 0:
        return HYDROELASTIC_MODE_COMPLIANT
    if (flags & wp.int32(ShapeFlags.HYDROELASTIC)) != 0:
        # Backward compatibility: legacy HYDROELASTIC implies compliant.
        return HYDROELASTIC_MODE_COMPLIANT
    return HYDROELASTIC_MODE_NONE


@wp.func
def is_hydroelastic_compliant(mode: wp.int32) -> wp.bool:
    return mode == HYDROELASTIC_MODE_COMPLIANT


@wp.func
def get_pair_effective_stiffness(
    k_a: wp.float32,
    k_b: wp.float32,
    mode_a: wp.int32,
    mode_b: wp.int32,
) -> wp.float32:
    """Compute pair stiffness for hydroelastic contacts.

    Rules:
    - compliant-compliant: harmonic mean
    - rigid-compliant: compliant stiffness
    - rigid-rigid / unsupported: 0
    """
    compliant_a = is_hydroelastic_compliant(mode_a)
    compliant_b = is_hydroelastic_compliant(mode_b)
    if compliant_a and compliant_b:
        return get_effective_stiffness(k_a, k_b)
    if compliant_a:
        return k_a
    if compliant_b:
        return k_b
    return 0.0


@wp.func
def get_pair_damping(
    kd_a: wp.float32,
    kd_b: wp.float32,
    mode_a: wp.int32,
    mode_b: wp.int32,
) -> wp.float32:
    """Compute hydroelastic contact damping from per-shape damping settings."""
    compliant_a = is_hydroelastic_compliant(mode_a)
    compliant_b = is_hydroelastic_compliant(mode_b)
    if compliant_a and compliant_b:
        return 0.5 * (kd_a + kd_b)
    if compliant_a:
        return kd_a
    if compliant_b:
        return kd_b
    return 0.0


@wp.func
def get_pair_friction_scale(
    mu_a: wp.float32,
    mu_b: wp.float32,
    mode_a: wp.int32,
    mode_b: wp.int32,
) -> wp.float32:
    """Compute friction scale so hydroelastic contacts can target compliant-surface friction."""
    compliant_a = is_hydroelastic_compliant(mode_a)
    compliant_b = is_hydroelastic_compliant(mode_b)
    base_mu = 0.5 * (mu_a + mu_b)
    if base_mu <= wp.float32(1.0e-8):
        return 1.0

    desired_mu = base_mu
    if compliant_a and compliant_b:
        desired_mu = base_mu
    elif compliant_a:
        desired_mu = mu_a
    elif compliant_b:
        desired_mu = mu_b

    if desired_mu <= wp.float32(0.0):
        return 1.0
    return desired_mu / base_mu


class HydroelasticSDF:
    """Hydroelastic contact generation with SDF-based collision detection.

    This class implements hydroelastic contact modeling between shapes represented
    by Signed Distance Fields (SDFs). It uses an octree-based broadphase to identify
    potentially colliding regions, then applies marching cubes to extract the
    zero-isosurface where both SDFs intersect. Contact points are generated at
    triangle centroids on this isosurface, with contact forces proportional to
    penetration depth and represented area.

    The collision pipeline consists of:
        1. Broadphase: Identifies overlapping OBBs of SDF between shape pairs
        2. Octree refinement: Hierarchically subdivides blocks to find iso-voxels
        3. Marching cubes: Extracts contact surface triangles from iso-voxels
        4. Contact generation: Computes contact points, normals, depths, and areas
        5. Optional contact reduction: Bins and reduces contacts per shape pair

    Args:
        num_shape_pairs: Maximum number of hydroelastic shape pairs to process.
        total_num_tiles: Total number of SDF blocks across all hydroelastic shapes.
        max_num_blocks_per_shape: Maximum block count for any single shape.
        shape_sdf_block_coords: Block coordinates for each shape's SDF representation.
        shape_sdf_shape2blocks: Mapping from shape index to (start, end) block range.
        shape_material_kh: Hydroelastic stiffness coefficient for each shape.
        shape_contact_workflow: Contact workflow enum value for each shape.
        shape_pressure_index: Per-shape index into compact pressure field table.
        compact_pressure_field_data: Compact immutable pressure-field table
            aligned with compact SDF indices.
        n_shapes: Total number of shapes in the simulation.
        config: Configuration options controlling buffer sizes, contact reduction,
            and other behavior. Defaults to :class:`HydroelasticSDF.Config`.
        device: Warp device for GPU computation.
        writer_func: Callback for writing decoded contact data.

    Note:
        Use :meth:`_from_model` to construct from a simulation :class:`Model`,
        which automatically extracts the required SDF data and shape information.

        Contact IDs are packed into 32-bit integers using 9 bits per voxel axis coordinate.
        For SDF grids larger than 512 voxels per axis, contact ID collisions may occur,
        which can affect contact matching accuracy for warm-starting physics solvers.

    See Also:
        :class:`HydroelasticSDF.Config`: Configuration options for this class.
    """

    @dataclass
    class Config:
        """Controls properties of SDF hydroelastic collision handling."""

        reduce_contacts: bool = True
        """Whether to reduce contacts to a smaller representative set per shape pair.
        When False, all generated contacts are passed through without reduction."""
        pre_prune_contacts: bool = True
        """Whether to perform local-first face compaction during generation.
        This mode avoids global hashtable traffic in the hot generation loop and
        writes a smaller contact set to the buffer before the normal reduce pass.
        Only active when ``reduce_contacts`` is True."""
        buffer_fraction: float = 1.0
        """Fraction of worst-case hydroelastic buffer allocations. Range: (0, 1].

        This scales pre-allocated broadphase, iso-refinement, and face-contact
        buffers before applying stage multipliers. Lower values reduce memory
        usage and may cause overflows in dense scenes. Overflows are bounds-safe
        and emit warnings; increase this value when warnings appear.
        """
        buffer_mult_broad: int = 1
        """Multiplier for the preallocated broadphase buffer that stores overlapping
        block pairs. Increase only if a broadphase overflow warning is issued."""
        buffer_mult_iso: int = 1
        """Multiplier for preallocated iso-surface extraction buffers used during
        hierarchical octree refinement (subblocks and voxels). Increase only if an iso buffer overflow warning is issued."""
        buffer_mult_contact: int = 1
        """Multiplier for the preallocated face contact buffer that stores contact
        positions, normals, depths, and areas. Increase only if a face contact overflow warning is issued."""
        contact_buffer_fraction: float = 0.5
        """Fraction of the face contact buffer to allocate when ``reduce_contacts`` is True.
        The reduce kernel selects winners from whatever fits in the buffer, so a smaller
        buffer trades off coverage for memory savings.
        Range: (0, 1]. Only applied when ``reduce_contacts`` is enabled; ignored otherwise."""
        grid_size: int = 256 * 8 * 128
        """Grid size for contact handling. Can be tuned for performance."""
        output_contact_surface: bool = False
        """Whether to output hydroelastic contact surface vertices for visualization."""
        normal_matching: bool = True
        """Whether to rotate reduced contact normals so their weighted sum aligns with
        the aggregate force direction. Only active when reduce_contacts is True."""
        anchor_contact: bool = False
        """Whether to add an anchor contact at the center of pressure for each normal bin.
        The anchor contact helps preserve moment balance. Only active when reduce_contacts is True."""
        margin_contact_area: float = 1e-2
        """Contact area used for non-penetrating contacts at the margin."""
        pre_prune_accumulate_all_penetrating_aggregates: bool = False
        """When pre-pruning is enabled, also accumulate aggregate force terms from all
        penetrating faces before pruning writes to the contact buffer.

        This preserves aggregate stiffness/normal/anchor fidelity while keeping the
        fast local compaction path for contact storage. The default keeps the current
        fastest behavior (aggregates from retained contacts only).
        """

    @dataclass
    class ContactSurfaceData:
        """
        Data container for hydroelastic contact surface visualization.

        Contains the vertex arrays and metadata needed for rendering
        the contact surface triangles from hydroelastic collision detection.
        """

        contact_surface_point: wp.array(dtype=wp.vec3f)
        """World-space positions of contact surface triangle vertices (3 per face)."""
        contact_surface_depth: wp.array(dtype=wp.float32)
        """Penetration depth at each face centroid."""
        contact_surface_shape_pair: wp.array(dtype=wp.vec2i)
        """Shape pair indices (shape_a, shape_b) for each face."""
        face_contact_count: wp.array(dtype=wp.int32)
        """Array containing the number of face contacts."""
        max_num_face_contacts: int
        """Maximum number of face contacts (buffer size)."""

    def __init__(
        self,
        num_shape_pairs: int,
        total_num_tiles: int,
        max_num_blocks_per_shape: int,
        shape_sdf_block_coords: wp.array(dtype=wp.vec3us),
        shape_sdf_shape2blocks: wp.array(dtype=wp.vec2i),
        shape_material_kh: wp.array(dtype=wp.float32),
        shape_material_kd: wp.array(dtype=wp.float32),
        shape_material_mu: wp.array(dtype=wp.float32),
        shape_contact_workflow: wp.array(dtype=wp.int32),
        shape_pressure_index: wp.array(dtype=wp.int32),
        compact_pressure_field_data: wp.array(dtype=PressureFieldData),
        pressure_field_volume: list[wp.Volume],
        n_shapes: int,
        config: HydroelasticSDF.Config | None = None,
        device: Devicelike | None = None,
        writer_func: Any = None,
    ) -> None:
        if config is None:
            config = HydroelasticSDF.Config()

        self.config = config
        if device is None:
            device = wp.get_device()
        self.device = device

        # keep local references for model arrays
        self.shape_sdf_block_coords = shape_sdf_block_coords
        self.shape_sdf_shape2blocks = shape_sdf_shape2blocks
        self.shape_material_kh = shape_material_kh
        self.shape_material_kd = shape_material_kd
        self.shape_material_mu = shape_material_mu
        self.shape_contact_workflow = shape_contact_workflow
        self.shape_pressure_index = shape_pressure_index
        self.compact_pressure_field_data = compact_pressure_field_data
        # Keep pressure volumes alive for device-side pointer validity.
        self.pressure_field_volume = pressure_field_volume

        self.n_shapes = n_shapes
        self.max_num_shape_pairs = num_shape_pairs
        self.total_num_tiles = total_num_tiles
        self.max_num_blocks_per_shape = max_num_blocks_per_shape

        frac = float(self.config.buffer_fraction)
        if frac <= 0.0 or frac > 1.0:
            raise ValueError(f"HydroelasticSDF.Config.buffer_fraction must be in (0, 1], got {frac}")
        contact_frac = float(self.config.contact_buffer_fraction)
        if contact_frac <= 0.0 or contact_frac > 1.0:
            raise ValueError(f"HydroelasticSDF.Config.contact_buffer_fraction must be in (0, 1], got {contact_frac}")

        mult = max(int(self.config.buffer_mult_iso * self.total_num_tiles * frac), 64)
        self.max_num_blocks_broad = max(
            int(self.max_num_shape_pairs * self.max_num_blocks_per_shape * self.config.buffer_mult_broad * frac),
            64,
        )
        # Output buffer sizes for each octree level (subblocks 8x8x8 -> 4x4x4 -> 2x2x2 -> voxels)
        self.iso_max_dims = (int(2 * mult), int(2 * mult), int(16 * mult), int(32 * mult))
        self.max_num_iso_voxels = self.iso_max_dims[3]
        # Input buffer sizes for each octree level
        self.input_sizes = (self.max_num_blocks_broad, *self.iso_max_dims[:3])

        with wp.ScopedDevice(device):
            self.num_shape_pairs_array = wp.full((1,), self.max_num_shape_pairs, dtype=wp.int32)

            # Allocate buffers for octree traversal (broadphase + 4 refinement levels)
            self.iso_buffer_counts = [wp.zeros((1,), dtype=wp.int32) for _ in range(5)]
            # Scratch buffers are per-level to avoid scanning the worst-case
            # size at all refinement levels during graph-captured execution.
            self.iso_buffer_prefix_scratch = [wp.zeros(level_input, dtype=wp.int32) for level_input in self.input_sizes]
            self.iso_buffer_num_scratch = [wp.zeros(level_input, dtype=wp.int32) for level_input in self.input_sizes]
            self.iso_subblock_idx_scratch = [wp.zeros(level_input, dtype=wp.uint8) for level_input in self.input_sizes]
            self.iso_buffer_coords = [wp.empty((self.max_num_blocks_broad,), dtype=wp.vec3us)] + [
                wp.empty((self.iso_max_dims[i],), dtype=wp.vec3us) for i in range(4)
            ]
            self.iso_buffer_shape_pairs = [wp.empty((self.max_num_blocks_broad,), dtype=wp.vec2i)] + [
                wp.empty((self.iso_max_dims[i],), dtype=wp.vec2i) for i in range(4)
            ]

            # Aliases for commonly accessed final buffers
            self.block_broad_collide_count = self.iso_buffer_counts[0]
            self.iso_voxel_count = self.iso_buffer_counts[4]
            self.iso_voxel_coords = self.iso_buffer_coords[4]
            self.iso_voxel_shape_pair = self.iso_buffer_shape_pairs[4]

            # Broadphase buffers
            self.block_start_prefix = wp.zeros((self.max_num_shape_pairs,), dtype=wp.int32)
            self.num_blocks_per_pair = wp.zeros((self.max_num_shape_pairs,), dtype=wp.int32)
            self.block_broad_idx = wp.empty((self.max_num_blocks_broad,), dtype=wp.int32)
            self.block_broad_collide_coords = self.iso_buffer_coords[0]
            self.block_broad_collide_shape_pair = self.iso_buffer_shape_pairs[0]

            # Face contacts written directly to GlobalContactReducer (no intermediate buffers)
            # When pre-pruning is active, far fewer contacts reach the buffer so we
            # scale down by contact_buffer_fraction to save memory.
            face_contact_budget = config.buffer_mult_contact * self.max_num_iso_voxels
            if config.reduce_contacts and config.pre_prune_contacts:
                face_contact_budget = face_contact_budget * config.contact_buffer_fraction
            self.max_num_face_contacts = max(int(face_contact_budget), 64)

            if self.config.output_contact_surface:
                # stores the point and depth of the contact surface vertex
                self.iso_vertex_point = wp.empty((3 * self.max_num_face_contacts,), dtype=wp.vec3f)
                self.iso_vertex_depth = wp.empty((self.max_num_face_contacts,), dtype=wp.float32)
                self.iso_vertex_shape_pair = wp.empty((self.max_num_face_contacts,), dtype=wp.vec2i)
            else:
                self.iso_vertex_point = wp.empty((0,), dtype=wp.vec3f)
                self.iso_vertex_depth = wp.empty((0,), dtype=wp.float32)
                self.iso_vertex_shape_pair = wp.empty((0,), dtype=wp.vec2i)

            self.mc_tables = get_mc_tables(device)

            # Placeholder empty arrays for kernel parameters unused in no-prune mode
            self._empty_vec3 = wp.empty((0,), dtype=wp.vec3, device=device)
            self._empty_vec3i = wp.empty((0,), dtype=wp.vec3i, device=device)

            # Pre-allocate per-shape SDF data buffer used in launch() so that
            # no wp.empty() call occurs during CUDA graph capture (#1616).
            self._shape_sdf_data = wp.empty(n_shapes, dtype=SDFData, device=device)
            self._shape_pressure_data = wp.empty(n_shapes, dtype=PressureFieldData, device=device)

            self.generate_contacts_kernel = get_generate_contacts_kernel(
                output_vertices=self.config.output_contact_surface,
                pre_prune=self.config.reduce_contacts and self.config.pre_prune_contacts,
                accumulate_all_penetrating_aggregates=(
                    self.config.reduce_contacts
                    and self.config.pre_prune_contacts
                    and self.config.pre_prune_accumulate_all_penetrating_aggregates
                ),
            )

            if self.config.reduce_contacts:
                # Use HydroelasticContactReduction for efficient hashtable-based contact reduction
                # The reducer uses spatial extremes + max-depth per normal bin + voxel-based slots
                reduction_config = HydroelasticReductionConfig(
                    normal_matching=self.config.normal_matching,
                    anchor_contact=self.config.anchor_contact,
                    margin_contact_area=self.config.margin_contact_area,
                )
                self.contact_reduction = HydroelasticContactReduction(
                    capacity=self.max_num_face_contacts,
                    device=device,
                    writer_func=writer_func,
                    config=reduction_config,
                )
                self.decode_contacts_kernel = None
            else:
                # No reduction - create a simple reducer for buffer storage and decode kernel
                self.contact_reduction = HydroelasticContactReduction(
                    capacity=self.max_num_face_contacts,
                    device=device,
                    writer_func=writer_func,
                    config=HydroelasticReductionConfig(margin_contact_area=self.config.margin_contact_area),
                )
                self.decode_contacts_kernel = get_decode_contacts_kernel(
                    self.config.margin_contact_area,
                    writer_func,
                )

        self.grid_size = min(self.config.grid_size, self.max_num_face_contacts)
        self._host_warning_poll_interval = 120
        self._launch_counter = 0

    @classmethod
    def _from_model(
        cls, model: Model, config: HydroelasticSDF.Config | None = None, writer_func: Any = None
    ) -> HydroelasticSDF | None:
        """Create HydroelasticSDF from a model.

        Args:
            model: The simulation model.
            config: Optional configuration for hydroelastic collision handling.
            writer_func: Optional writer function for decoding contacts.

        Returns:
            HydroelasticSDF instance, or None if no hydroelastic shape pairs exist.
        """
        runtime_data = collect_hydroelastic_runtime_data(model)
        if runtime_data is None:
            return None

        return cls(
            num_shape_pairs=runtime_data.num_hydroelastic_pairs,
            total_num_tiles=runtime_data.total_num_tiles,
            max_num_blocks_per_shape=runtime_data.max_num_blocks_per_shape,
            shape_sdf_block_coords=model.sdf_block_coords,
            shape_sdf_shape2blocks=wp.array(runtime_data.shape_sdf_shape2blocks, dtype=wp.vec2i, device=model.device),
            shape_material_kh=model.shape_material_kh,
            shape_material_kd=model.shape_material_kd,
            shape_material_mu=model.shape_material_mu,
            shape_contact_workflow=wp.array(runtime_data.shape_contact_workflow, dtype=wp.int32, device=model.device),
            shape_pressure_index=wp.array(runtime_data.shape_pressure_index, dtype=wp.int32, device=model.device),
            compact_pressure_field_data=wp.array(runtime_data.compact_pressure_field_data, dtype=PressureFieldData, device=model.device),
            pressure_field_volume=runtime_data.pressure_field_volume,
            n_shapes=model.shape_count,
            config=config,
            device=model.device,
            writer_func=writer_func,
        )

    def get_contact_surface(self) -> ContactSurfaceData | None:
        """Get hydroelastic :class:`ContactSurfaceData` for visualization.

        Returns:
            A :class:`ContactSurfaceData` instance containing vertex arrays and metadata for rendering,
            or None if :attr:`config.output_contact_surface` is False.
        """
        if not self.config.output_contact_surface:
            return None
        return self.ContactSurfaceData(
            contact_surface_point=self.iso_vertex_point,
            contact_surface_depth=self.iso_vertex_depth,
            contact_surface_shape_pair=self.iso_vertex_shape_pair,
            face_contact_count=self.contact_reduction.contact_count,
            max_num_face_contacts=self.max_num_face_contacts,
        )

    def launch(
        self,
        sdf_data: wp.array(dtype=SDFData),
        shape_sdf_index: wp.array(dtype=wp.int32),
        shape_type: wp.array(dtype=wp.int32),
        shape_data: wp.array(dtype=wp.vec4),
        shape_transform: wp.array(dtype=wp.transform),
        shape_flags: wp.array(dtype=wp.int32),
        shape_gap: wp.array(dtype=wp.float32),
        shape_heightfield_data: wp.array(dtype=HeightfieldData),
        heightfield_elevation_data: wp.array(dtype=wp.float32),
        shape_collision_aabb_lower: wp.array(dtype=wp.vec3),
        shape_collision_aabb_upper: wp.array(dtype=wp.vec3),
        shape_voxel_resolution: wp.array(dtype=wp.vec3i),
        shape_pairs_sdf_sdf: wp.array(dtype=wp.vec2i),
        shape_pairs_sdf_sdf_count: wp.array(dtype=wp.int32),
        writer_data: Any,
    ) -> None:
        """Run the full hydroelastic collision pipeline.

        Args:
            sdf_data: Compact SDF table.
            shape_sdf_index: Per-shape SDF index into sdf_data.
            shape_type: Per-shape geometry type.
            shape_data: Per-shape scale/thickness data.
            shape_transform: World transforms for each shape.
            shape_flags: Per-shape flags with hydroelastic mode bits.
            shape_gap: Per-shape contact gap (detection threshold) for each shape.
            shape_heightfield_data: Per-shape heightfield metadata.
            heightfield_elevation_data: Concatenated heightfield elevation array.
            shape_collision_aabb_lower: Per-shape collision AABB lower bounds.
            shape_collision_aabb_upper: Per-shape collision AABB upper bounds.
            shape_voxel_resolution: Per-shape voxel grid resolution.
            shape_pairs_sdf_sdf: Pairs of shape indices to check for collision.
            shape_pairs_sdf_sdf_count: Number of valid shape pairs.
            writer_data: Contact data writer for output.
        """
        shape_sdf_data = self._shape_sdf_data
        shape_pressure_data = self._shape_pressure_data
        wp.launch(
            kernel=map_shape_sdf_data_kernel,
            dim=shape_sdf_index.shape[0],
            inputs=[sdf_data, shape_sdf_index],
            outputs=[shape_sdf_data],
            device=self.device,
        )
        wp.launch(
            kernel=map_shape_pressure_data_kernel,
            dim=shape_sdf_index.shape[0],
            inputs=[self.compact_pressure_field_data, self.shape_pressure_index],
            outputs=[shape_pressure_data],
            device=self.device,
        )

        self._broadphase_sdfs(
            shape_sdf_data,
            shape_transform,
            shape_flags,
            shape_type,
            shape_pairs_sdf_sdf,
            shape_pairs_sdf_sdf_count,
        )

        self._find_iso_voxels(
            shape_sdf_data,
            shape_pressure_data,
            shape_type,
            shape_data,
            shape_transform,
            shape_flags,
            shape_heightfield_data,
            heightfield_elevation_data,
            shape_gap,
        )

        if self.config.reduce_contacts:
            self._generate_contacts(
                shape_sdf_data,
                shape_pressure_data,
                shape_type,
                shape_data,
                shape_transform,
                shape_flags,
                shape_heightfield_data,
                heightfield_elevation_data,
                shape_gap,
            )
            self._reduce_decode_contacts(
                shape_transform,
                shape_flags,
                shape_collision_aabb_lower,
                shape_collision_aabb_upper,
                shape_voxel_resolution,
                shape_gap,
                writer_data,
            )
        else:
            self._generate_contacts(
                shape_sdf_data,
                shape_pressure_data,
                shape_type,
                shape_data,
                shape_transform,
                shape_flags,
                shape_heightfield_data,
                heightfield_elevation_data,
                shape_gap,
            )
            self._decode_contacts(
                shape_transform,
                shape_flags,
                shape_gap,
                writer_data,
            )

        wp.launch(
            kernel=verify_collision_step,
            dim=[1],
            inputs=[
                self.block_broad_collide_count,
                self.max_num_blocks_broad,
                self.iso_buffer_counts[1],
                self.iso_max_dims[0],
                self.iso_buffer_counts[2],
                self.iso_max_dims[1],
                self.iso_buffer_counts[3],
                self.iso_max_dims[2],
                self.iso_voxel_count,
                self.max_num_iso_voxels,
                self.contact_reduction.contact_count,
                self.max_num_face_contacts,
                writer_data.contact_count,
                writer_data.contact_max,
                self.contact_reduction.reducer.ht_insert_failures,
            ],
            device=self.device,
        )

        # Poll infrequently to avoid per-step host sync overhead while still surfacing
        # dropped-contact conditions outside stdout-captured environments.
        self._launch_counter += 1
        if self._launch_counter % self._host_warning_poll_interval == 0:
            hashtable_failures = int(self.contact_reduction.reducer.ht_insert_failures.numpy()[0])
            if hashtable_failures > 0:
                warnings.warn(
                    "Hydroelastic reduction dropped contacts due to hashtable insert "
                    f"failures ({hashtable_failures}). Increase rigid_contact_max "
                    "and/or HydroelasticSDF.Config.buffer_fraction.",
                    RuntimeWarning,
                    stacklevel=2,
                )

    def _broadphase_sdfs(
        self,
        shape_sdf_data: wp.array(dtype=SDFData),
        shape_transform: wp.array(dtype=wp.transform),
        shape_flags: wp.array(dtype=wp.int32),
        shape_type: wp.array(dtype=wp.int32),
        shape_pairs_sdf_sdf: wp.array(dtype=wp.vec2i),
        shape_pairs_sdf_sdf_count: wp.array(dtype=wp.int32),
    ) -> None:
        # Test collisions between OBB of SDFs
        self.num_blocks_per_pair.zero_()

        wp.launch(
            kernel=broadphase_collision_pairs_count,
            dim=[self.max_num_shape_pairs],
            inputs=[
                shape_transform,
                shape_sdf_data,
                shape_pairs_sdf_sdf,
                shape_pairs_sdf_sdf_count,
                shape_flags,
                shape_type,
                self.shape_sdf_shape2blocks,
            ],
            outputs=[
                self.num_blocks_per_pair,
            ],
            device=self.device,
        )

        scan_with_total(
            self.num_blocks_per_pair,
            self.block_start_prefix,
            self.num_shape_pairs_array,
            self.block_broad_collide_count,
        )

        wp.launch(
            kernel=broadphase_collision_pairs_scatter,
            dim=[self.max_num_shape_pairs],
            inputs=[
                self.num_blocks_per_pair,
                shape_sdf_data,
                self.block_start_prefix,
                shape_pairs_sdf_sdf,
                shape_pairs_sdf_sdf_count,
                shape_flags,
                self.shape_sdf_shape2blocks,
                self.max_num_blocks_broad,
            ],
            outputs=[
                self.block_broad_collide_shape_pair,
                self.block_broad_idx,
            ],
            device=self.device,
        )

        wp.launch(
            kernel=broadphase_get_block_coords,
            dim=[self.grid_size],
            inputs=[
                self.grid_size,
                self.block_broad_collide_count,
                self.block_broad_idx,
                self.shape_sdf_block_coords,
                self.max_num_blocks_broad,
            ],
            outputs=[
                self.block_broad_collide_coords,
            ],
            device=self.device,
        )

    def _find_iso_voxels(
        self,
        shape_sdf_data: wp.array(dtype=SDFData),
        shape_pressure_data: wp.array(dtype=PressureFieldData),
        shape_type: wp.array(dtype=wp.int32),
        shape_data: wp.array(dtype=wp.vec4),
        shape_transform: wp.array(dtype=wp.transform),
        shape_flags: wp.array(dtype=wp.int32),
        shape_heightfield_data: wp.array(dtype=HeightfieldData),
        heightfield_elevation_data: wp.array(dtype=wp.float32),
        shape_gap: wp.array(dtype=wp.float32),
    ) -> None:
        # Find voxels which contain the isosurface between the shapes using octree-like pruning.
        # We do this by computing the difference between sdfs at the voxel/subblock center and comparing it to the voxel/subblock radius.
        # The check is first performed for subblocks of size (8 x 8 x 8), then (4 x 4 x 4), then (2 x 2 x 2), and finally for each voxel.
        for i, (subblock_size, n_blocks) in enumerate([(8, 1), (4, 2), (2, 2), (1, 2)]):
            wp.launch(
                kernel=count_iso_voxels_block,
                dim=[self.grid_size],
                inputs=[
                    self.grid_size,
                    self.iso_buffer_counts[i],
                    shape_sdf_data,
                    shape_pressure_data,
                    shape_type,
                    shape_data,
                    shape_transform,
                    self.shape_material_kh,
                    shape_flags,
                    self.shape_contact_workflow,
                    shape_heightfield_data,
                    heightfield_elevation_data,
                    self.iso_buffer_coords[i],
                    self.iso_buffer_shape_pairs[i],
                    shape_gap,
                    subblock_size,
                    n_blocks,
                    self.input_sizes[i],
                ],
                outputs=[
                    self.iso_buffer_num_scratch[i],
                    self.iso_subblock_idx_scratch[i],
                ],
                device=self.device,
            )

            scan_with_total(
                self.iso_buffer_num_scratch[i],
                self.iso_buffer_prefix_scratch[i],
                self.iso_buffer_counts[i],
                self.iso_buffer_counts[i + 1],
            )

            wp.launch(
                kernel=scatter_iso_subblock,
                dim=[self.grid_size],
                inputs=[
                    self.grid_size,
                    self.iso_buffer_counts[i],
                    self.iso_buffer_prefix_scratch[i],
                    self.iso_subblock_idx_scratch[i],
                    self.iso_buffer_shape_pairs[i],
                    self.iso_buffer_coords[i],
                    subblock_size,
                    self.input_sizes[i],
                    self.iso_max_dims[i],
                ],
                outputs=[
                    self.iso_buffer_coords[i + 1],
                    self.iso_buffer_shape_pairs[i + 1],
                ],
                device=self.device,
            )

    def _generate_contacts(
        self,
        shape_sdf_data: wp.array(dtype=SDFData),
        shape_pressure_data: wp.array(dtype=PressureFieldData),
        shape_type: wp.array(dtype=wp.int32),
        shape_data: wp.array(dtype=wp.vec4),
        shape_transform: wp.array(dtype=wp.transform),
        shape_flags: wp.array(dtype=wp.int32),
        shape_heightfield_data: wp.array(dtype=HeightfieldData),
        heightfield_elevation_data: wp.array(dtype=wp.float32),
        shape_gap: wp.array(dtype=wp.float32),
        shape_local_aabb_lower: wp.array | None = None,
        shape_local_aabb_upper: wp.array | None = None,
        shape_voxel_resolution: wp.array | None = None,
    ) -> None:
        """Generate marching cubes contacts and write directly to the contact buffer.

        Single pass: compute cube state and immediately write faces to reducer buffer.
        When pre-pruning is active the extra AABB/voxel-resolution arrays must be
        provided so the kernel can populate the hashtable and gate buffer writes.
        """
        self.contact_reduction.clear()
        reducer_data = self.contact_reduction.get_data_struct()

        # Placeholder arrays for the pre-prune parameters when not used
        if shape_local_aabb_lower is None:
            shape_local_aabb_lower = self._empty_vec3
        if shape_local_aabb_upper is None:
            shape_local_aabb_upper = self._empty_vec3
        if shape_voxel_resolution is None:
            shape_voxel_resolution = self._empty_vec3i

        wp.launch(
            kernel=self.generate_contacts_kernel,
            dim=[self.grid_size],
            inputs=[
                self.grid_size,
                self.iso_voxel_count,
                shape_sdf_data,
                shape_pressure_data,
                shape_type,
                shape_data,
                shape_transform,
                self.shape_material_kh,
                shape_flags,
                self.shape_contact_workflow,
                shape_heightfield_data,
                heightfield_elevation_data,
                self.iso_voxel_coords,
                self.iso_voxel_shape_pair,
                self.mc_tables[0],
                self.mc_tables[4],
                self.mc_tables[3],
                shape_gap,
                self.max_num_iso_voxels,
                reducer_data,
                shape_local_aabb_lower,
                shape_local_aabb_upper,
                shape_voxel_resolution,
            ],
            outputs=[
                self.iso_vertex_point,
                self.iso_vertex_depth,
                self.iso_vertex_shape_pair,
            ],
            device=self.device,
        )

    def _decode_contacts(
        self,
        shape_transform: wp.array(dtype=wp.transform),
        shape_flags: wp.array(dtype=wp.int32),
        shape_gap: wp.array(dtype=wp.float32),
        writer_data: Any,
    ) -> None:
        """Decode hydroelastic contacts without reduction.

        Contacts are already in the buffer (written by _generate_contacts).
        This method exports all contacts directly without any reduction.
        """
        wp.launch(
            kernel=self.decode_contacts_kernel,
            dim=[self.grid_size],
            inputs=[
                self.grid_size,
                self.contact_reduction.contact_count,
                self.shape_material_kh,
                self.shape_material_kd,
                self.shape_material_mu,
                shape_flags,
                shape_transform,
                shape_gap,
                self.contact_reduction.reducer.position_depth,
                self.contact_reduction.reducer.normal,
                self.contact_reduction.reducer.shape_pairs,
                self.contact_reduction.reducer.contact_area,
                self.max_num_face_contacts,
            ],
            outputs=[writer_data],
            device=self.device,
        )

    def _reduce_decode_contacts(
        self,
        shape_transform: wp.array(dtype=wp.transform),
        shape_flags: wp.array(dtype=wp.int32),
        shape_collision_aabb_lower: wp.array(dtype=wp.vec3),
        shape_collision_aabb_upper: wp.array(dtype=wp.vec3),
        shape_voxel_resolution: wp.array(dtype=wp.vec3i),
        shape_gap: wp.array(dtype=wp.float32),
        writer_data: Any,
    ) -> None:
        """Reduce buffered contacts and export the winners.

        Runs the reduction kernel to populate the hashtable (spatial extremes,
        max-depth, voxel bins) and accumulate aggregates, then exports the
        winning contacts via the writer function.
        """
        self.contact_reduction.reduce(
            shape_material_k_hydro=self.shape_material_kh,
            shape_flags=shape_flags,
            shape_transform=shape_transform,
            shape_collision_aabb_lower=shape_collision_aabb_lower,
            shape_collision_aabb_upper=shape_collision_aabb_upper,
            shape_voxel_resolution=shape_voxel_resolution,
            grid_size=self.grid_size,
            skip_aggregates=(
                self.config.pre_prune_contacts and self.config.pre_prune_accumulate_all_penetrating_aggregates
            ),
        )
        self.contact_reduction.export(
            shape_material_kd=self.shape_material_kd,
            shape_material_mu=self.shape_material_mu,
            shape_flags=shape_flags,
            shape_gap=shape_gap,
            shape_transform=shape_transform,
            writer_data=writer_data,
            grid_size=self.grid_size,
        )


@wp.kernel(enable_backward=False)
def broadphase_collision_pairs_count(
    shape_transform: wp.array(dtype=wp.transform),
    shape_sdf_data: wp.array(dtype=SDFData),
    shape_pairs_sdf_sdf: wp.array(dtype=wp.vec2i),
    shape_pairs_sdf_sdf_count: wp.array(dtype=wp.int32),
    shape_flags: wp.array(dtype=wp.int32),
    shape_type: wp.array(dtype=wp.int32),
    shape2blocks: wp.array(dtype=wp.vec2i),
    # outputs
    thread_num_blocks: wp.array(dtype=wp.int32),
):
    tid = wp.tid()
    if tid >= shape_pairs_sdf_sdf_count[0]:
        return

    pair = shape_pairs_sdf_sdf[tid]
    shape_a = pair[0]
    shape_b = pair[1]
    mode_a = hydroelastic_mode_from_flags(shape_flags[shape_a])
    mode_b = hydroelastic_mode_from_flags(shape_flags[shape_b])
    compliant_a = mode_a == HYDROELASTIC_MODE_COMPLIANT
    compliant_b = mode_b == HYDROELASTIC_MODE_COMPLIANT

    type_a = shape_type[shape_a]
    type_b = shape_type[shape_b]

    analytic_rigid_a = (
        mode_a == HYDROELASTIC_MODE_RIGID
        and (type_a == wp.int32(GeoType.PLANE) or type_a == wp.int32(GeoType.HFIELD))
        and shape_sdf_data[shape_a].sparse_sdf_ptr == wp.uint64(0)
    )
    analytic_rigid_b = (
        mode_b == HYDROELASTIC_MODE_RIGID
        and (type_b == wp.int32(GeoType.PLANE) or type_b == wp.int32(GeoType.HFIELD))
        and shape_sdf_data[shape_b].sparse_sdf_ptr == wp.uint64(0)
    )
    has_analytic_rigid_terrain = (analytic_rigid_a and compliant_b) or (analytic_rigid_b and compliant_a)

    does_collide = wp.bool(False)
    if has_analytic_rigid_terrain:
        # Conservative admission for analytic rigid terrain. Downstream iso pruning
        # rejects non-overlapping blocks.
        does_collide = True
    else:
        half_extents_a = shape_sdf_data[shape_a].half_extents
        half_extents_b = shape_sdf_data[shape_b].half_extents
        center_offset_a = shape_sdf_data[shape_a].center
        center_offset_b = shape_sdf_data[shape_b].center
        world_transform_a = shape_transform[shape_a]
        world_transform_b = shape_transform[shape_b]
        centered_transform_a = wp.transform_multiply(
            world_transform_a, wp.transform(center_offset_a, wp.quat_identity())
        )
        centered_transform_b = wp.transform_multiply(
            world_transform_b, wp.transform(center_offset_b, wp.quat_identity())
        )
        does_collide = sat_box_intersection(centered_transform_a, half_extents_a, centered_transform_b, half_extents_b)

    # For rigid-compliant pairs keep the compliant body in slot B so downstream
    # depth sampling (mc_calc_face over sdf_b) uses the compliant field.
    if compliant_a and (not compliant_b):
        shape_a, shape_b = shape_b, shape_a
    elif compliant_a == compliant_b:
        # For same-mode pairs preserve the previous smaller-voxel-in-B rule.
        voxel_radius_a = shape_sdf_data[shape_a].sparse_voxel_radius
        voxel_radius_b = shape_sdf_data[shape_b].sparse_voxel_radius
        if voxel_radius_b > voxel_radius_a:
            shape_b, shape_a = shape_a, shape_b

    shape_b_idx = shape2blocks[shape_b]
    block_start, block_end = shape_b_idx[0], shape_b_idx[1]
    num_blocks = block_end - block_start

    if does_collide:
        thread_num_blocks[tid] = num_blocks
    else:
        thread_num_blocks[tid] = 0


@wp.kernel(enable_backward=False)
def broadphase_collision_pairs_scatter(
    thread_num_blocks: wp.array(dtype=wp.int32),
    shape_sdf_data: wp.array(dtype=SDFData),
    block_start_prefix: wp.array(dtype=wp.int32),
    shape_pairs_sdf_sdf: wp.array(dtype=wp.vec2i),
    shape_pairs_sdf_sdf_count: wp.array(dtype=wp.int32),
    shape_flags: wp.array(dtype=wp.int32),
    shape2blocks: wp.array(dtype=wp.vec2i),
    max_num_blocks_broad: int,
    # outputs
    block_broad_collide_shape_pair: wp.array(dtype=wp.vec2i),
    block_broad_idx: wp.array(dtype=wp.int32),
):
    tid = wp.tid()
    if tid >= shape_pairs_sdf_sdf_count[0]:
        return

    num_blocks = thread_num_blocks[tid]
    if num_blocks == 0:
        return

    pair = shape_pairs_sdf_sdf[tid]
    shape_a = pair[0]
    shape_b = pair[1]

    mode_a = hydroelastic_mode_from_flags(shape_flags[shape_a])
    mode_b = hydroelastic_mode_from_flags(shape_flags[shape_b])
    compliant_a = mode_a == HYDROELASTIC_MODE_COMPLIANT
    compliant_b = mode_b == HYDROELASTIC_MODE_COMPLIANT

    if compliant_a and (not compliant_b):
        shape_a, shape_b = shape_b, shape_a
    elif compliant_a == compliant_b:
        # Sort shapes such that the shape with the smaller voxel size is in second place.
        voxel_radius_a = shape_sdf_data[shape_a].sparse_voxel_radius
        voxel_radius_b = shape_sdf_data[shape_b].sparse_voxel_radius
        if voxel_radius_b > voxel_radius_a:
            shape_b, shape_a = shape_a, shape_b

    shape_b_idx = shape2blocks[shape_b]
    shape_b_block_start = shape_b_idx[0]

    block_start = block_start_prefix[tid]

    remaining = max_num_blocks_broad - block_start
    if remaining <= 0:
        return
    num_blocks = wp.min(num_blocks, remaining)

    pair = wp.vec2i(shape_a, shape_b)
    for i in range(num_blocks):
        block_broad_collide_shape_pair[block_start + i] = pair
        block_broad_idx[block_start + i] = shape_b_block_start + i


@wp.kernel(enable_backward=False)
def broadphase_get_block_coords(
    grid_size: int,
    block_count: wp.array(dtype=wp.int32),
    block_broad_idx: wp.array(dtype=wp.int32),
    block_coords: wp.array(dtype=wp.vec3us),
    max_num_blocks_broad: int,
    # outputs
    block_broad_collide_coords: wp.array(dtype=wp.vec3us),
):
    offset = wp.tid()
    num_blocks = wp.min(block_count[0], max_num_blocks_broad)
    for tid in range(offset, num_blocks, grid_size):
        block_idx = block_broad_idx[tid]
        block_broad_collide_coords[tid] = block_coords[block_idx]


@wp.func
def encode_coords_8(x: wp.int32, y: wp.int32, z: wp.int32) -> wp.uint8:
    # Encode 3D coordinates in range [0, 1] per axis into a single 8-bit integer
    return wp.uint8(1) << (wp.uint8(x) + wp.uint8(y) * wp.uint8(2) + wp.uint8(z) * wp.uint8(4))


@wp.func
def decode_coords_8(bit_pos: wp.uint8) -> wp.vec3ub:
    # Decode bit position back to 3D coordinates
    return wp.vec3ub(
        bit_pos & wp.uint8(1), (bit_pos >> wp.uint8(1)) & wp.uint8(1), (bit_pos >> wp.uint8(2)) & wp.uint8(1)
    )


@wp.func
def get_rel_stiffness(
    k_a: wp.float32,
    k_b: wp.float32,
    mode_a: wp.int32,
    mode_b: wp.int32,
) -> tuple[wp.float32, wp.float32]:
    """Compute relative stiffness weights used in SDF difference evaluation.

    Returns ``(w_a, w_b)`` for the affine field ``w_a*phi_a - w_b*phi_b``.
    """
    compliant_a = is_hydroelastic_compliant(mode_a)
    compliant_b = is_hydroelastic_compliant(mode_b)

    # Compliant-compliant: original normalized weighting from Elandt et al.
    if compliant_a and compliant_b:
        denom = wp.max(k_a * k_b, wp.float32(1.0e-12))
        k_m_inv = 1.0 / wp.sqrt(denom)
        return k_a * k_m_inv, k_b * k_m_inv

    # Rigid-compliant: interface is the rigid surface.
    if compliant_a and not compliant_b:
        return 0.0, 1.0
    if not compliant_a and compliant_b:
        return 1.0, 0.0

    # Rigid-rigid is unsupported in hydroelastic path, but keep a safe fallback.
    return 1.0, 1.0


@wp.func
def weighted_sdf_difference(
    val_a: wp.float32,
    val_b: wp.float32,
    w_a: wp.float32,
    w_b: wp.float32,
) -> wp.float32:
    """Evaluate the weighted SDF difference used for hydroelastic isosurfaces.

    Using one affine expression everywhere avoids discontinuities caused by
    switching formulas at ``phi=0`` boundaries.
    """
    return w_a * val_a - w_b * val_b


@wp.func
def sample_plane_signed_field_clipped(local_pos: wp.vec3, width: wp.float32, length: wp.float32) -> wp.float32:
    """Sample signed plane field with finite-footprint clipping.

    Infinite planes (``width <= 0`` or ``length <= 0``) use signed local-Z.
    Finite planes use signed local-Z inside the XY footprint and return a
    positive distance outside the footprint to suppress off-plane contacts.
    """
    if width <= 0.0 or length <= 0.0:
        return local_pos[2]

    ax = wp.abs(local_pos[0])
    ay = wp.abs(local_pos[1])
    if ax <= width and ay <= length:
        return local_pos[2]

    dx = wp.max(ax - width, 0.0)
    dy = wp.max(ay - length, 0.0)
    lateral = wp.sqrt(dx * dx + dy * dy)
    if local_pos[2] > 0.0:
        return wp.sqrt(lateral * lateral + local_pos[2] * local_pos[2])
    return lateral


@wp.func
def sample_signed_field_with_terrain_fallback(
    sdf_data: SDFData,
    shape_type: wp.int32,
    shape_scale: wp.vec3,
    shape_hfield_data: HeightfieldData,
    elevation_data: wp.array(dtype=wp.float32),
    local_pos: wp.vec3,
) -> tuple[wp.float32, wp.bool]:
    """Sample signed field from SDF when available, else terrain analytic path."""
    if sdf_data.sparse_sdf_ptr != wp.uint64(0):
        return sample_sdf_extrapolated(sdf_data, local_pos), True

    if shape_type == wp.int32(GeoType.PLANE):
        return sample_plane_signed_field_clipped(local_pos, shape_scale[0], shape_scale[1]), True

    if shape_type == wp.int32(GeoType.HFIELD):
        return sample_sdf_heightfield(shape_hfield_data, elevation_data, local_pos), True

    return wp.float32(MAXVAL), False


@wp.func
def sample_pressure_field(pressure_data: PressureFieldData, local_pos: wp.vec3) -> wp.float32:
    """Sample immutable pressure field at local position."""
    if pressure_data.pressure_ptr == wp.uint64(0):
        return 0.0

    idx = wp.volume_world_to_index(pressure_data.pressure_ptr, local_pos)
    p = wp.volume_sample_f(pressure_data.pressure_ptr, idx, wp.Volume.LINEAR)
    if wp.isnan(p):
        return 0.0
    return wp.max(p, 0.0)


@wp.func
def eval_signed_field(
    sdf_val: wp.float32,
    local_pos: wp.vec3,
    sdf_data: SDFData,
    mode: wp.int32,
    pressure_data: PressureFieldData,
    pair_workflow: wp.int32,
) -> tuple[wp.float32, wp.bool]:
    """Evaluate a robust signed field and validity flag for one shape.

    In classic workflow this returns the SDF value. In pressure workflow, compliant
    shapes may use immutable pressure potential in the interior (negative sign
    convention), decoupling contact from sparse/coarse SDF sign artifacts in deep
    interior samples.
    """
    invalid_sdf = sdf_val >= wp.static(MAXVAL * 0.99) or wp.isnan(sdf_val)

    if (
        pair_workflow == HYDROELASTIC_WORKFLOW_PRESSURE
        and mode == HYDROELASTIC_MODE_COMPLIANT
        and pressure_data.pressure_ptr != wp.uint64(0)
    ):
        pressure = sample_pressure_field(pressure_data, local_pos)
        if pressure > 0.0:
            if (not invalid_sdf) and sdf_val < 0.0:
                depth = -sdf_val
                boundary_shell = wp.max(
                    0.3 * wp.length(sdf_data.half_extents),
                    2.0 * sdf_data.sparse_voxel_radius,
                )
                blend = wp.clamp(depth / boundary_shell, 0.0, 1.0)
                return -((1.0 - blend) * depth + blend * pressure), True
            if invalid_sdf:
                return -pressure, True

    if invalid_sdf:
        return wp.float32(MAXVAL), False

    return sdf_val, True


@wp.func
def sdf_diff_sdf(
    sdfA_data: SDFData,
    sdfB_data: SDFData,
    pressureA_data: PressureFieldData,
    pressureB_data: PressureFieldData,
    mode_a: wp.int32,
    mode_b: wp.int32,
    workflow_a: wp.int32,
    workflow_b: wp.int32,
    transfA: wp.transform,
    transfB: wp.transform,
    k_eff_a: wp.float32,
    k_eff_b: wp.float32,
    shape_b_type: wp.int32,
    shape_b_scale: wp.vec3,
    shape_b_hfield_data: HeightfieldData,
    heightfield_elevation_data: wp.array(dtype=wp.float32),
    x_id: wp.int32,
    y_id: wp.int32,
    z_id: wp.int32,
) -> tuple[wp.float32, wp.float32, wp.float32, wp.bool]:
    """Compute hydroelastic field difference at a voxel position.

    SDF A is queried directly on the sparse grid (allocated voxel).
    SDF B is queried with extrapolation. For compliant shapes, deep interior
    evaluation uses immutable pressure potential when available.
    """
    sdfA = sdfA_data.sparse_sdf_ptr
    pointA = wp.volume_index_to_world(sdfA, int_to_vec3f(x_id, y_id, z_id))
    pointA_world = wp.transform_point(transfA, pointA)
    pointB = wp.transform_point(wp.transform_inverse(transfB), pointA_world)
    valA = wp.volume_lookup_f(sdfA, x_id, y_id, z_id)

    valB, sampled_b = sample_signed_field_with_terrain_fallback(
        sdfB_data,
        shape_b_type,
        shape_b_scale,
        shape_b_hfield_data,
        heightfield_elevation_data,
        pointB,
    )

    pair_workflow = resolve_pair_contact_workflow(mode_a, mode_b, workflow_a, workflow_b)
    fieldA, validA = eval_signed_field(valA, pointA, sdfA_data, mode_a, pressureA_data, pair_workflow)
    fieldB, validB = eval_signed_field(valB, pointB, sdfB_data, mode_b, pressureB_data, pair_workflow)
    is_valid = sampled_b and validA and validB

    diff = weighted_sdf_difference(fieldA, fieldB, k_eff_a, k_eff_b)
    return diff, fieldA, fieldB, is_valid


@wp.func
def sdf_diff_sdf(
    sdfA_data: SDFData,
    sdfB_data: SDFData,
    pressureA_data: PressureFieldData,
    pressureB_data: PressureFieldData,
    mode_a: wp.int32,
    mode_b: wp.int32,
    workflow_a: wp.int32,
    workflow_b: wp.int32,
    transfA: wp.transform,
    transfB: wp.transform,
    k_eff_a: wp.float32,
    k_eff_b: wp.float32,
    shape_b_type: wp.int32,
    shape_b_scale: wp.vec3,
    shape_b_hfield_data: HeightfieldData,
    heightfield_elevation_data: wp.array(dtype=wp.float32),
    pos_a_local: wp.vec3,
) -> tuple[wp.float32, wp.float32, wp.float32, wp.bool]:
    """Compute hydroelastic field difference at a local position.

    SDF A is queried directly on the sparse grid.
    SDF B is queried with extrapolation. For compliant shapes, deep interior
    evaluation uses immutable pressure potential when available.
    """
    sdfA = sdfA_data.sparse_sdf_ptr
    pointA = wp.volume_index_to_world(sdfA, pos_a_local)
    pointA_world = wp.transform_point(transfA, pointA)
    pointB = wp.transform_point(wp.transform_inverse(transfB), pointA_world)
    valA = wp.volume_sample_f(sdfA, pos_a_local, wp.Volume.LINEAR)

    valB, sampled_b = sample_signed_field_with_terrain_fallback(
        sdfB_data,
        shape_b_type,
        shape_b_scale,
        shape_b_hfield_data,
        heightfield_elevation_data,
        pointB,
    )

    pair_workflow = resolve_pair_contact_workflow(mode_a, mode_b, workflow_a, workflow_b)
    fieldA, validA = eval_signed_field(valA, pointA, sdfA_data, mode_a, pressureA_data, pair_workflow)
    fieldB, validB = eval_signed_field(valB, pointB, sdfB_data, mode_b, pressureB_data, pair_workflow)
    is_valid = sampled_b and validA and validB

    diff = weighted_sdf_difference(fieldA, fieldB, k_eff_a, k_eff_b)
    return diff, fieldA, fieldB, is_valid


@wp.kernel(enable_backward=False)
def count_iso_voxels_block(
    grid_size: int,
    in_buffer_collide_count: wp.array(dtype=int),
    shape_sdf_data: wp.array(dtype=SDFData),
    shape_pressure_data: wp.array(dtype=PressureFieldData),
    shape_type: wp.array(dtype=wp.int32),
    shape_data: wp.array(dtype=wp.vec4),
    shape_transform: wp.array(dtype=wp.transform),
    shape_material_kh: wp.array(dtype=float),
    shape_flags: wp.array(dtype=wp.int32),
    shape_contact_workflow: wp.array(dtype=wp.int32),
    shape_heightfield_data: wp.array(dtype=HeightfieldData),
    heightfield_elevation_data: wp.array(dtype=wp.float32),
    in_buffer_collide_coords: wp.array(dtype=wp.vec3us),
    in_buffer_collide_shape_pair: wp.array(dtype=wp.vec2i),
    shape_gap: wp.array(dtype=wp.float32),
    subblock_size: int,
    n_blocks: int,
    max_input_buffer_size: int,
    # outputs
    iso_subblock_counts: wp.array(dtype=wp.int32),
    iso_subblock_idx: wp.array(dtype=wp.uint8),
):
    # checks if the isosurface between shapes a and b lies inside the subblock (iterating over subblocks of b).
    # if so, write the subblock coordinates to the output.
    offset = wp.tid()
    num_items = wp.min(in_buffer_collide_count[0], max_input_buffer_size)
    for tid in range(offset, num_items, grid_size):
        pair = in_buffer_collide_shape_pair[tid]
        shape_a = pair[0]
        shape_b = pair[1]

        sdf_data_a = shape_sdf_data[shape_a]
        sdf_data_b = shape_sdf_data[shape_b]
        pressure_data_a = shape_pressure_data[shape_a]
        pressure_data_b = shape_pressure_data[shape_b]

        X_ws_a = shape_transform[shape_a]
        X_ws_b = shape_transform[shape_b]
        type_a = shape_type[shape_a]
        scale_data_a = shape_data[shape_a]
        scale_a = wp.vec3(scale_data_a[0], scale_data_a[1], scale_data_a[2])
        hfield_data_a = shape_heightfield_data[shape_a]

        margin_a = shape_gap[shape_a]
        margin_b = shape_gap[shape_b]

        voxel_radius = sdf_data_b.sparse_voxel_radius
        r = float(subblock_size) * voxel_radius

        k_a = shape_material_kh[shape_a]
        k_b = shape_material_kh[shape_b]
        mode_a = hydroelastic_mode_from_flags(shape_flags[shape_a])
        mode_b = hydroelastic_mode_from_flags(shape_flags[shape_b])
        workflow_a = shape_contact_workflow[shape_a]
        workflow_b = shape_contact_workflow[shape_b]
        k_eff_a, k_eff_b = get_rel_stiffness(k_a, k_b, mode_a, mode_b)
        r_eff = r * (k_eff_a + k_eff_b)

        # get global voxel coordinates
        bc = in_buffer_collide_coords[tid]

        num_iso_subblocks = wp.int32(0)
        subblock_idx = wp.uint8(0)
        for x_local in range(n_blocks):
            for y_local in range(n_blocks):
                for z_local in range(n_blocks):
                    x_global = wp.vec3i(bc) + wp.vec3i(x_local, y_local, z_local) * subblock_size

                    # lookup distances at subblock center
                    # for subblock_size = 1 this is equivalent to the voxel center
                    x_center = wp.vec3f(x_global) + wp.vec3f(0.5 * float(subblock_size))
                    diff_val, vb, va, is_valid = sdf_diff_sdf(
                        sdf_data_b,
                        sdf_data_a,
                        pressure_data_b,
                        pressure_data_a,
                        mode_b,
                        mode_a,
                        workflow_b,
                        workflow_a,
                        X_ws_b,
                        X_ws_a,
                        k_eff_b,
                        k_eff_a,
                        type_a,
                        scale_a,
                        hfield_data_a,
                        heightfield_elevation_data,
                        x_center,
                    )

                    # check if bounding sphere contains the isosurface and the distance is within contact margin
                    if wp.abs(diff_val) > r_eff or va > r + margin_a or vb > r + margin_b or not is_valid:
                        continue
                    num_iso_subblocks += 1
                    subblock_idx |= encode_coords_8(x_local, y_local, z_local)

        iso_subblock_counts[tid] = num_iso_subblocks
        iso_subblock_idx[tid] = subblock_idx


@wp.kernel(enable_backward=False)
def scatter_iso_subblock(
    grid_size: int,
    in_iso_subblock_count: wp.array(dtype=int),
    in_iso_subblock_prefix: wp.array(dtype=int),
    in_iso_subblock_idx: wp.array(dtype=wp.uint8),
    in_iso_subblock_shape_pair: wp.array(dtype=wp.vec2i),
    in_buffer_collide_coords: wp.array(dtype=wp.vec3us),
    subblock_size: int,
    max_input_buffer_size: int,
    max_num_iso_subblocks: int,
    # outputs
    out_iso_subblock_coords: wp.array(dtype=wp.vec3us),
    out_iso_subblock_shape_pair: wp.array(dtype=wp.vec2i),
):
    offset = wp.tid()
    num_items = wp.min(in_iso_subblock_count[0], max_input_buffer_size)
    for tid in range(offset, num_items, grid_size):
        write_idx = in_iso_subblock_prefix[tid]
        subblock_idx = in_iso_subblock_idx[tid]
        pair = in_iso_subblock_shape_pair[tid]
        bc = in_buffer_collide_coords[tid]
        if write_idx >= max_num_iso_subblocks:
            continue
        for i in range(8):
            bit_pos = wp.uint8(i)
            if (subblock_idx >> bit_pos) & wp.uint8(1) and not write_idx >= max_num_iso_subblocks:
                local_coords = wp.vec3us(decode_coords_8(bit_pos))
                global_coords = bc + local_coords * wp.uint16(subblock_size)
                out_iso_subblock_coords[write_idx] = global_coords
                out_iso_subblock_shape_pair[write_idx] = pair
                write_idx += 1


@wp.func
def mc_iterate_voxel_vertices(
    x_id: wp.int32,
    y_id: wp.int32,
    z_id: wp.int32,
    corner_offsets_table: wp.array(dtype=wp.vec3ub),
    sdf_data: SDFData,
    sdf_other_data: SDFData,
    pressure_data: PressureFieldData,
    pressure_other_data: PressureFieldData,
    mode: wp.int32,
    mode_other: wp.int32,
    workflow: wp.int32,
    workflow_other: wp.int32,
    X_ws: wp.transform,
    X_ws_other: wp.transform,
    k_eff: wp.float32,
    k_eff_other: wp.float32,
    other_shape_type: wp.int32,
    other_shape_scale: wp.vec3,
    other_shape_hfield_data: HeightfieldData,
    heightfield_elevation_data: wp.array(dtype=wp.float32),
    margin: wp.float32,
) -> tuple[wp.uint8, vec8f, bool, bool]:
    """Iterate over the vertices of a voxel and return the cube index, corner values, and whether any vertices are inside the shape."""
    cube_idx = wp.uint8(0)
    any_verts_inside_margin = False
    corner_vals = vec8f()

    for i in range(8):
        corner_offset = wp.vec3i(corner_offsets_table[i])
        x = x_id + corner_offset.x
        y = y_id + corner_offset.y
        z = z_id + corner_offset.z

        v_diff, v, _v_other, is_valid = sdf_diff_sdf(
            sdf_data,
            sdf_other_data,
            pressure_data,
            pressure_other_data,
            mode,
            mode_other,
            workflow,
            workflow_other,
            X_ws,
            X_ws_other,
            k_eff,
            k_eff_other,
            other_shape_type,
            other_shape_scale,
            other_shape_hfield_data,
            heightfield_elevation_data,
            x,
            y,
            z,
        )

        if not is_valid:
            return wp.uint8(0), corner_vals, False, False

        corner_vals[i] = v_diff

        if v_diff < 0.0:
            cube_idx |= wp.uint8(1) << wp.uint8(i)

        if v <= margin:
            any_verts_inside_margin = True

    return cube_idx, corner_vals, any_verts_inside_margin, True


# =============================================================================
# Contact decode kernel (no reduction)
# =============================================================================


def get_decode_contacts_kernel(margin_contact_area: float = 1e-4, writer_func: Any = None):
    """Create a kernel that decodes hydroelastic contacts without reduction.

    This kernel is used when reduce_contacts=False. It exports all generated
    contacts directly to the writer without any spatial reduction.

    Args:
        margin_contact_area: Contact area used for non-penetrating contacts at the margin.
        writer_func: Warp function for writing decoded contacts.

    Returns:
        A warp kernel that can be launched to decode all contacts.
    """

    @wp.kernel(enable_backward=False)
    def decode_contacts_kernel(
        grid_size: int,
        contact_count: wp.array(dtype=int),
        shape_material_kh: wp.array(dtype=wp.float32),
        shape_material_kd: wp.array(dtype=wp.float32),
        shape_material_mu: wp.array(dtype=wp.float32),
        shape_flags: wp.array(dtype=wp.int32),
        shape_transform: wp.array(dtype=wp.transform),
        shape_gap: wp.array(dtype=wp.float32),
        position_depth: wp.array(dtype=wp.vec4),
        normal: wp.array(dtype=wp.vec2),  # Octahedral-encoded
        shape_pairs: wp.array(dtype=wp.vec2i),
        contact_area: wp.array(dtype=wp.float32),
        max_num_face_contacts: int,
        # outputs
        writer_data: Any,
    ):
        """Decode all hydroelastic contacts without reduction.

        Uses grid stride loop to process all contacts in the buffer.
        """
        offset = wp.tid()
        num_contacts = wp.min(contact_count[0], max_num_face_contacts)

        # Calculate how many contacts this thread will process
        my_contact_count = 0
        if offset < num_contacts:
            my_contact_count = (num_contacts - 1 - offset) // grid_size + 1

        if my_contact_count == 0:
            return

        # Single atomic to reserve all slots for this thread (no rollback)
        my_base_index = wp.atomic_add(writer_data.contact_count, 0, my_contact_count)

        # Write contacts using reserved range
        local_idx = int(0)
        for tid in range(offset, num_contacts, grid_size):
            output_index = my_base_index + local_idx
            local_idx += 1

            if output_index >= writer_data.contact_max:
                continue

            pair = shape_pairs[tid]
            shape_a = pair[0]
            shape_b = pair[1]

            transform_b = shape_transform[shape_b]

            pd = position_depth[tid]
            pos = wp.vec3(pd[0], pd[1], pd[2])
            depth = pd[3]
            contact_normal = decode_oct(normal[tid])

            normal_world = wp.transform_vector(transform_b, contact_normal)
            pos_world = wp.transform_point(transform_b, pos)

            # Sum margins for consistency with thickness summing
            margin_a = shape_gap[shape_a]
            margin_b = shape_gap[shape_b]
            margin = margin_a + margin_b

            k_a = shape_material_kh[shape_a]
            k_b = shape_material_kh[shape_b]
            kd_a = shape_material_kd[shape_a]
            kd_b = shape_material_kd[shape_b]
            mu_a = shape_material_mu[shape_a]
            mu_b = shape_material_mu[shape_b]
            mode_a = hydroelastic_mode_from_flags(shape_flags[shape_a])
            mode_b = hydroelastic_mode_from_flags(shape_flags[shape_b])
            k_eff = get_pair_effective_stiffness(k_a, k_b, mode_a, mode_b)
            area = contact_area[tid]

            # Compute stiffness, use margin_contact_area for non-penetrating contacts
            # Standard convention: depth < 0 = penetrating
            if depth < 0.0:
                c_stiffness = area * k_eff
            else:
                c_stiffness = wp.static(margin_contact_area) * k_eff
            c_damping = get_pair_damping(kd_a, kd_b, mode_a, mode_b)
            c_friction_scale = get_pair_friction_scale(mu_a, mu_b, mode_a, mode_b)

            # Create ContactData for the writer function
            # contact_distance = 2 * depth (depth is negative for penetrating)
            contact_data = ContactData()
            contact_data.contact_point_center = pos_world
            contact_data.contact_normal_a_to_b = normal_world
            contact_data.contact_distance = 2.0 * depth
            contact_data.radius_eff_a = 0.0
            contact_data.radius_eff_b = 0.0
            contact_data.margin_a = 0.0
            contact_data.margin_b = 0.0
            contact_data.shape_a = shape_a
            contact_data.shape_b = shape_b
            contact_data.margin = margin
            contact_data.contact_stiffness = c_stiffness
            contact_data.contact_damping = c_damping
            contact_data.contact_friction_scale = c_friction_scale

            writer_func(contact_data, writer_data, output_index)

    return decode_contacts_kernel


# =============================================================================
# Contact generation kernels
# =============================================================================


def get_generate_contacts_kernel(
    output_vertices: bool,
    pre_prune: bool = False,
    accumulate_all_penetrating_aggregates: bool = False,
):
    """Create kernel for hydroelastic contact generation.

    This is a merged kernel that computes cube state and immediately writes
    faces to the reducer buffer in a single pass, eliminating intermediate
    storage for cube indices and corner values.

    A separate ``reduce_hydroelastic_contacts_kernel`` then runs on the
    buffer to populate the hashtable and select representative contacts.

    When ``pre_prune`` is enabled, this kernel applies a local-first compaction
    rule before writing contacts:
    - keep top-K penetrating faces by area*|depth| (K=2)
    - keep at most one non-penetrating fallback face (closest to penetration)

    When ``accumulate_all_penetrating_aggregates`` is enabled, all penetrating
    faces contribute to aggregate force terms (via hashtable entries) even if
    they are later pruned from buffer writes.

    Args:
        output_vertices: Whether to output contact surface vertices for visualization.

    Returns:
        generate_contacts_kernel: Warp kernel for contact generation.
    """

    @wp.kernel(enable_backward=False)
    def generate_contacts_kernel(
        grid_size: int,
        iso_voxel_count: wp.array(dtype=wp.int32),
        shape_sdf_data: wp.array(dtype=SDFData),
        shape_pressure_data: wp.array(dtype=PressureFieldData),
        shape_type: wp.array(dtype=wp.int32),
        shape_data: wp.array(dtype=wp.vec4),
        shape_transform: wp.array(dtype=wp.transform),
        shape_material_kh: wp.array(dtype=float),
        shape_flags: wp.array(dtype=wp.int32),
        shape_contact_workflow: wp.array(dtype=wp.int32),
        shape_heightfield_data: wp.array(dtype=HeightfieldData),
        heightfield_elevation_data: wp.array(dtype=wp.float32),
        iso_voxel_coords: wp.array(dtype=wp.vec3us),
        iso_voxel_shape_pair: wp.array(dtype=wp.vec2i),
        tri_range_table: wp.array(dtype=wp.int32),
        flat_edge_verts_table: wp.array(dtype=wp.vec2ub),
        corner_offsets_table: wp.array(dtype=wp.vec3ub),
        shape_gap: wp.array(dtype=wp.float32),
        max_num_iso_voxels: int,
        reducer_data: GlobalContactReducerData,
        # Unused — kept for signature compatibility with prior callers
        _shape_local_aabb_lower: wp.array(dtype=wp.vec3),
        _shape_local_aabb_upper: wp.array(dtype=wp.vec3),
        _shape_voxel_resolution: wp.array(dtype=wp.vec3i),
        # Outputs for visualization (optional)
        iso_vertex_point: wp.array(dtype=wp.vec3f),
        iso_vertex_depth: wp.array(dtype=wp.float32),
        iso_vertex_shape_pair: wp.array(dtype=wp.vec2i),
    ):
        """Generate marching cubes contacts and write to GlobalContactReducer."""
        offset = wp.tid()
        num_voxels = wp.min(iso_voxel_count[0], max_num_iso_voxels)
        for tid in range(offset, num_voxels, grid_size):
            pair = iso_voxel_shape_pair[tid]
            shape_a = pair[0]
            shape_b = pair[1]

            sdf_data_a = shape_sdf_data[shape_a]
            sdf_data_b = shape_sdf_data[shape_b]
            pressure_data_a = shape_pressure_data[shape_a]
            pressure_data_b = shape_pressure_data[shape_b]

            transform_a = shape_transform[shape_a]
            transform_b = shape_transform[shape_b]

            iso_coords = iso_voxel_coords[tid]

            margin_a = shape_gap[shape_a]
            margin_b = shape_gap[shape_b]
            margin = margin_a + margin_b

            k_a = shape_material_kh[shape_a]
            k_b = shape_material_kh[shape_b]
            mode_a = hydroelastic_mode_from_flags(shape_flags[shape_a])
            mode_b = hydroelastic_mode_from_flags(shape_flags[shape_b])
            workflow_a = shape_contact_workflow[shape_a]
            workflow_b = shape_contact_workflow[shape_b]
            k_eff_a, k_eff_b = get_rel_stiffness(k_a, k_b, mode_a, mode_b)
            type_a = shape_type[shape_a]
            scale_data_a = shape_data[shape_a]
            scale_a = wp.vec3(scale_data_a[0], scale_data_a[1], scale_data_a[2])
            hfield_data_a = shape_heightfield_data[shape_a]

            x_id = wp.int32(iso_coords.x)
            y_id = wp.int32(iso_coords.y)
            z_id = wp.int32(iso_coords.z)

            # Compute cube state (marching cubes lookup)
            cube_idx, corner_vals, any_verts_inside, all_verts_valid = mc_iterate_voxel_vertices(
                x_id,
                y_id,
                z_id,
                corner_offsets_table,
                sdf_data_b,
                sdf_data_a,
                pressure_data_b,
                pressure_data_a,
                mode_b,
                mode_a,
                workflow_b,
                workflow_a,
                transform_b,
                transform_a,
                k_eff_b,
                k_eff_a,
                type_a,
                scale_a,
                hfield_data_a,
                heightfield_elevation_data,
                margin,
            )

            range_idx = wp.int32(cube_idx)
            tri_range_start = tri_range_table[range_idx]
            tri_range_end = tri_range_table[range_idx + 1]
            num_verts = tri_range_end - tri_range_start

            num_faces = num_verts // 3

            if not any_verts_inside or not all_verts_valid:
                num_faces = 0

            if num_faces == 0:
                continue

            # Compute effective stiffness coefficient
            k_eff = get_pair_effective_stiffness(k_a, k_b, mode_a, mode_b)

            sdf_b = sdf_data_b.sparse_sdf_ptr
            X_ws_b = transform_b

            # Generate faces and locally compact candidates before writing to the
            # global contact buffer (reduces atomics and downstream reduction load).
            best_pen0_valid = int(0)
            best_pen0_score = float(-MAXVAL)
            best_pen0_depth = float(0.0)
            best_pen0_area = float(0.0)
            best_pen0_normal = wp.vec3(0.0, 0.0, 1.0)
            best_pen0_center = wp.vec3(0.0, 0.0, 0.0)
            best_pen0_v0 = wp.vec3(0.0, 0.0, 0.0)
            best_pen0_v1 = wp.vec3(0.0, 0.0, 0.0)
            best_pen0_v2 = wp.vec3(0.0, 0.0, 0.0)

            best_pen1_valid = int(0)
            best_pen1_score = float(-MAXVAL)
            best_pen1_depth = float(0.0)
            best_pen1_area = float(0.0)
            best_pen1_normal = wp.vec3(0.0, 0.0, 1.0)
            best_pen1_center = wp.vec3(0.0, 0.0, 0.0)
            best_pen1_v0 = wp.vec3(0.0, 0.0, 0.0)
            best_pen1_v1 = wp.vec3(0.0, 0.0, 0.0)
            best_pen1_v2 = wp.vec3(0.0, 0.0, 0.0)

            best_nonpen_valid = int(0)
            best_nonpen_depth = float(MAXVAL)
            best_nonpen_area = float(0.0)
            best_nonpen_normal = wp.vec3(0.0, 0.0, 1.0)
            best_nonpen_center = wp.vec3(0.0, 0.0, 0.0)
            best_nonpen_v0 = wp.vec3(0.0, 0.0, 0.0)
            best_nonpen_v1 = wp.vec3(0.0, 0.0, 0.0)
            best_nonpen_v2 = wp.vec3(0.0, 0.0, 0.0)
            for fi in range(num_faces):
                area, normal, face_center, pen_depth, face_verts = mc_calc_face(
                    flat_edge_verts_table,
                    corner_offsets_table,
                    tri_range_start + 3 * fi,
                    corner_vals,
                    sdf_b,
                    x_id,
                    y_id,
                    z_id,
                )
                if area <= 0.0:
                    continue
                # Accumulate stats per normal bin
                if pen_depth < 0.0:
                    bin_id = get_slot(normal)
                    key = make_contact_key(shape_a, shape_b, bin_id)
                    entry_idx = hashtable_find_or_insert(key, reducer_data.ht_keys, reducer_data.ht_active_slots)
                    if entry_idx >= 0:
                        force_weight = area * (-pen_depth)
                        wp.atomic_add(reducer_data.agg_force, entry_idx, force_weight * normal)
                        wp.atomic_add(reducer_data.weighted_pos_sum, entry_idx, force_weight * face_center)
                        wp.atomic_add(reducer_data.weight_sum, entry_idx, force_weight)
                        reducer_data.entry_k_eff[entry_idx] = k_eff
                    else:
                        wp.atomic_add(reducer_data.ht_insert_failures, 0, 1)
                if wp.static(not pre_prune):
                    contact_id = export_hydroelastic_contact_to_buffer(
                        shape_a,
                        shape_b,
                        face_center,
                        normal,
                        pen_depth,
                        area,
                        k_eff,
                        reducer_data,
                    )
                    if wp.static(output_vertices) and contact_id >= 0:
                        for vi in range(3):
                            iso_vertex_point[3 * contact_id + vi] = wp.transform_point(X_ws_b, face_verts[vi])
                        iso_vertex_depth[contact_id] = pen_depth
                        iso_vertex_shape_pair[contact_id] = pair
                    continue

                if wp.static(accumulate_all_penetrating_aggregates) and pen_depth < 0.0:
                    # Optional accurate aggregate mode: accumulate ALL penetrating
                    # faces before local write pruning. This preserves downstream
                    # aggregate stiffness/normal/anchor terms.
                    bin_id = get_slot(normal)
                    key = make_contact_key(shape_a, shape_b, bin_id)
                    entry_idx = hashtable_find_or_insert(key, reducer_data.ht_keys, reducer_data.ht_active_slots)
                    if entry_idx >= 0:
                        force_weight = area * (-pen_depth)
                        wp.atomic_add(reducer_data.agg_force, entry_idx, force_weight * normal)
                        wp.atomic_add(reducer_data.weighted_pos_sum, entry_idx, force_weight * face_center)
                        wp.atomic_add(reducer_data.weight_sum, entry_idx, force_weight)
                        reducer_data.entry_k_eff[entry_idx] = k_eff
                    else:
                        wp.atomic_add(reducer_data.ht_insert_failures, 0, 1)

                # Local-first compaction: keep top-K penetrating faces by score.
                if pen_depth < 0.0:
                    score = area * (-pen_depth)
                    if best_pen0_valid == 0 or score > best_pen0_score:
                        # Shift slot0 -> slot1
                        best_pen1_valid = best_pen0_valid
                        best_pen1_score = best_pen0_score
                        best_pen1_depth = best_pen0_depth
                        best_pen1_area = best_pen0_area
                        best_pen1_normal = best_pen0_normal
                        best_pen1_center = best_pen0_center
                        best_pen1_v0 = best_pen0_v0
                        best_pen1_v1 = best_pen0_v1
                        best_pen1_v2 = best_pen0_v2

                        best_pen0_valid = int(1)
                        best_pen0_score = score
                        best_pen0_depth = pen_depth
                        best_pen0_area = area
                        best_pen0_normal = normal
                        best_pen0_center = face_center
                        best_pen0_v0 = face_verts[0]
                        best_pen0_v1 = face_verts[1]
                        best_pen0_v2 = face_verts[2]
                    elif wp.static(PRE_PRUNE_MAX_PENETRATING > 1):
                        if best_pen1_valid == 0 or score > best_pen1_score:
                            best_pen1_valid = int(1)
                            best_pen1_score = score
                            best_pen1_depth = pen_depth
                            best_pen1_area = area
                            best_pen1_normal = normal
                            best_pen1_center = face_center
                            best_pen1_v0 = face_verts[0]
                            best_pen1_v1 = face_verts[1]
                            best_pen1_v2 = face_verts[2]
                else:
                    # Defer non-penetrating contact and keep only the closest one.
                    if pen_depth < best_nonpen_depth:
                        best_nonpen_valid = int(1)
                        best_nonpen_depth = pen_depth
                        best_nonpen_area = area
                        best_nonpen_normal = normal
                        best_nonpen_center = face_center
                        best_nonpen_v0 = face_verts[0]
                        best_nonpen_v1 = face_verts[1]
                        best_nonpen_v2 = face_verts[2]

            if wp.static(pre_prune):
                # Batched reservation: one atomic for all kept contacts.
                keep_count = int(0)
                if best_pen0_valid == 1:
                    keep_count = keep_count + 1
                if wp.static(PRE_PRUNE_MAX_PENETRATING > 1):
                    if best_pen1_valid == 1:
                        keep_count = keep_count + 1
                if best_nonpen_valid == 1:
                    keep_count = keep_count + 1

                if keep_count > 0:
                    base = wp.atomic_add(reducer_data.contact_count, 0, keep_count)
                    if base < reducer_data.capacity:
                        out_idx = base

                        if best_pen0_valid == 1 and out_idx < reducer_data.capacity:
                            reducer_data.position_depth[out_idx] = wp.vec4(
                                best_pen0_center[0], best_pen0_center[1], best_pen0_center[2], best_pen0_depth
                            )
                            reducer_data.normal[out_idx] = encode_oct(best_pen0_normal)
                            reducer_data.shape_pairs[out_idx] = wp.vec2i(shape_a, shape_b)
                            reducer_data.contact_area[out_idx] = best_pen0_area
                            if wp.static(output_vertices):
                                iso_vertex_point[3 * out_idx + 0] = wp.transform_point(X_ws_b, best_pen0_v0)
                                iso_vertex_point[3 * out_idx + 1] = wp.transform_point(X_ws_b, best_pen0_v1)
                                iso_vertex_point[3 * out_idx + 2] = wp.transform_point(X_ws_b, best_pen0_v2)
                                iso_vertex_depth[out_idx] = best_pen0_depth
                                iso_vertex_shape_pair[out_idx] = pair
                            out_idx = out_idx + 1

                        if wp.static(PRE_PRUNE_MAX_PENETRATING > 1):
                            if best_pen1_valid == 1 and out_idx < reducer_data.capacity:
                                reducer_data.position_depth[out_idx] = wp.vec4(
                                    best_pen1_center[0], best_pen1_center[1], best_pen1_center[2], best_pen1_depth
                                )
                                reducer_data.normal[out_idx] = encode_oct(best_pen1_normal)
                                reducer_data.shape_pairs[out_idx] = wp.vec2i(shape_a, shape_b)
                                reducer_data.contact_area[out_idx] = best_pen1_area
                                if wp.static(output_vertices):
                                    iso_vertex_point[3 * out_idx + 0] = wp.transform_point(X_ws_b, best_pen1_v0)
                                    iso_vertex_point[3 * out_idx + 1] = wp.transform_point(X_ws_b, best_pen1_v1)
                                    iso_vertex_point[3 * out_idx + 2] = wp.transform_point(X_ws_b, best_pen1_v2)
                                    iso_vertex_depth[out_idx] = best_pen1_depth
                                    iso_vertex_shape_pair[out_idx] = pair
                                out_idx = out_idx + 1

                        if best_nonpen_valid == 1 and out_idx < reducer_data.capacity:
                            reducer_data.position_depth[out_idx] = wp.vec4(
                                best_nonpen_center[0], best_nonpen_center[1], best_nonpen_center[2], best_nonpen_depth
                            )
                            reducer_data.normal[out_idx] = encode_oct(best_nonpen_normal)
                            reducer_data.shape_pairs[out_idx] = wp.vec2i(shape_a, shape_b)
                            reducer_data.contact_area[out_idx] = best_nonpen_area
                            if wp.static(output_vertices):
                                iso_vertex_point[3 * out_idx + 0] = wp.transform_point(X_ws_b, best_nonpen_v0)
                                iso_vertex_point[3 * out_idx + 1] = wp.transform_point(X_ws_b, best_nonpen_v1)
                                iso_vertex_point[3 * out_idx + 2] = wp.transform_point(X_ws_b, best_nonpen_v2)
                                iso_vertex_depth[out_idx] = best_nonpen_depth
                                iso_vertex_shape_pair[out_idx] = pair

    return generate_contacts_kernel


# =============================================================================
# Verification kernel
# =============================================================================


@wp.kernel(enable_backward=False)
def verify_collision_step(
    num_broad_collide: wp.array(dtype=int),
    max_num_broad_collide: int,
    num_iso_subblocks_0: wp.array(dtype=int),
    max_num_iso_subblocks_0: int,
    num_iso_subblocks_1: wp.array(dtype=int),
    max_num_iso_subblocks_1: int,
    num_iso_subblocks_2: wp.array(dtype=int),
    max_num_iso_subblocks_2: int,
    num_iso_voxels: wp.array(dtype=int),
    max_num_iso_voxels: int,
    face_contact_count: wp.array(dtype=int),
    max_face_contact_count: int,
    contact_count: wp.array(dtype=int),
    max_contact_count: int,
    ht_insert_failures: wp.array(dtype=int),
):
    # Checks if any buffer overflowed in any stage of the collision pipeline.
    has_overflow = False
    if num_broad_collide[0] > max_num_broad_collide:
        wp.printf(
            "  [hydroelastic] broad phase overflow: %d > %d. Increase buffer_fraction or buffer_mult_broad.\n",
            num_broad_collide[0],
            max_num_broad_collide,
        )
        has_overflow = True
    if num_iso_subblocks_0[0] > max_num_iso_subblocks_0:
        wp.printf(
            "  [hydroelastic] iso subblock L0 overflow: %d > %d. Increase buffer_fraction or buffer_mult_iso.\n",
            num_iso_subblocks_0[0],
            max_num_iso_subblocks_0,
        )
        has_overflow = True
    if num_iso_subblocks_1[0] > max_num_iso_subblocks_1:
        wp.printf(
            "  [hydroelastic] iso subblock L1 overflow: %d > %d. Increase buffer_fraction or buffer_mult_iso.\n",
            num_iso_subblocks_1[0],
            max_num_iso_subblocks_1,
        )
        has_overflow = True
    if num_iso_subblocks_2[0] > max_num_iso_subblocks_2:
        wp.printf(
            "  [hydroelastic] iso subblock L2 overflow: %d > %d. Increase buffer_fraction or buffer_mult_iso.\n",
            num_iso_subblocks_2[0],
            max_num_iso_subblocks_2,
        )
        has_overflow = True
    if num_iso_voxels[0] > max_num_iso_voxels:
        wp.printf(
            "  [hydroelastic] iso voxel overflow: %d > %d. Increase buffer_fraction or buffer_mult_iso.\n",
            num_iso_voxels[0],
            max_num_iso_voxels,
        )
        has_overflow = True
    if face_contact_count[0] > max_face_contact_count:
        wp.printf(
            "  [hydroelastic] face contact overflow: %d > %d. Increase buffer_fraction or buffer_mult_contact.\n",
            face_contact_count[0],
            max_face_contact_count,
        )
        has_overflow = True
    if contact_count[0] > max_contact_count:
        wp.printf(
            "  [hydroelastic] rigid contact output overflow: %d > %d. Increase rigid_contact_max.\n",
            contact_count[0],
            max_contact_count,
        )
        has_overflow = True
    if ht_insert_failures[0] > 0:
        wp.printf(
            "  [hydroelastic] reduction hashtable full: %d insert failures. "
            "Increase rigid_contact_max and/or buffer_fraction.\n",
            ht_insert_failures[0],
        )
        has_overflow = True

    if has_overflow:
        wp.printf(
            "Warning: Hydroelastic buffers overflowed; some contacts may be dropped. "
            "Increase HydroelasticSDF.Config.buffer_fraction and/or per-stage buffer multipliers.\n",
        )
