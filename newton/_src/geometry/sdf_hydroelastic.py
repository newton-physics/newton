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

Configure shapes with ``ShapeConfig(is_hydroelastic=True, k_hydro=1e9)`` and
pass ``SDFHydroelasticConfig`` to the collision pipeline.

See Also:
    :class:`SDFHydroelasticConfig`: Configuration options for this module.
    :class:`HydroelasticContactReduction`: Contact reduction for hydroelastic contacts.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import warp as wp

from newton._src.core.types import MAXVAL

from ..sim.builder import ShapeFlags
from ..sim.model import Model
from .collision_core import sat_box_intersection
from .contact_data import ContactData
from .contact_reduction_global import GlobalContactReducerData
from .contact_reduction_hydroelastic import (
    HydroelasticContactReduction,
    HydroelasticReductionConfig,
    export_hydroelastic_contact_to_buffer,
)
from .sdf_contact import sample_sdf_extrapolated
from .sdf_mc import get_mc_tables, mc_calc_face
from .sdf_utils import SDFData
from .utils import scan_with_total

vec8f = wp.types.vector(length=8, dtype=wp.float32)


@wp.func
def int_to_vec3f(x: wp.int32, y: wp.int32, z: wp.int32):
    return wp.vec3f(float(x), float(y), float(z))


@wp.func
def get_effective_stiffness(k_a: wp.float32, k_b: wp.float32) -> wp.float32:
    """Compute effective stiffness for two materials in series."""
    return (k_a * k_b) / (k_a + k_b)


@dataclass
class HydroelasticContactSurfaceData:
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


@dataclass
class SDFHydroelasticConfig:
    """
    Controls properties of SDF hydroelastic collision handling.
    """

    reduce_contacts: bool = True
    """Whether to reduce contacts to a smaller representative set per shape pair.
    When False, all generated contacts are passed through without reduction."""
    buffer_mult_broad: int = 1
    """Multiplier for the preallocated broadphase buffer that stores overlapping
    block pairs. Increase only if a broadphase overflow warning is issued."""
    buffer_mult_iso: int = 1
    """Multiplier for preallocated iso-surface extraction buffers used during
    hierarchical octree refinement (subblocks and voxels). Increase only if an iso buffer overflow warning is issued."""
    buffer_mult_contact: int = 1
    """Multiplier for the preallocated face contact buffer that stores contact
    positions, normals, depths, and areas. Increase only if a face contact overflow warning is issued."""
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
    moment_matching: bool = False
    """Whether to scale friction coefficients to match the aggregate moment from unreduced contacts.
    Requires anchor_contact=True to be effective. Only active when reduce_contacts is True."""
    margin_contact_area: float = 1e-2
    """Contact area used for non-penetrating contacts at the margin."""


class SDFHydroelastic:
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
        shape_material_k_hydro: Hydroelastic stiffness coefficient for each shape.
        n_shapes: Total number of shapes in the simulation.
        config: Configuration options controlling buffer sizes, contact reduction,
            and other behavior. Defaults to :class:`SDFHydroelasticConfig`.
        device: Warp device for GPU computation.
        writer_func: Callback for writing decoded contact data.

    Note:
        Use :meth:`_from_model` to construct from a simulation :class:`Model`,
        which automatically extracts the required SDF data and shape information.

        Contact IDs are packed into 32-bit integers using 9 bits per voxel axis coordinate.
        For SDF grids larger than 512 voxels per axis, contact ID collisions may occur,
        which can affect contact matching accuracy for warm-starting physics solvers.

    See Also:
        :class:`SDFHydroelasticConfig`: Configuration options for this class.
    """

    def __init__(
        self,
        num_shape_pairs: int,
        total_num_tiles: int,
        max_num_blocks_per_shape: int,
        shape_sdf_block_coords: wp.array(dtype=wp.vec3us),
        shape_sdf_shape2blocks: wp.array(dtype=wp.vec2i),
        shape_material_k_hydro: wp.array(dtype=wp.float32),
        n_shapes: int,
        config: SDFHydroelasticConfig = None,
        device: Any = None,
        writer_func: Any = None,
    ):
        if config is None:
            config = SDFHydroelasticConfig()

        self.config = config
        if device is None:
            device = wp.get_device()
        self.device = device

        # keep local references for model arrays
        self.shape_sdf_block_coords = shape_sdf_block_coords
        self.shape_sdf_shape2blocks = shape_sdf_shape2blocks
        self.shape_material_k_hydro = shape_material_k_hydro

        self.n_shapes = n_shapes
        self.max_num_shape_pairs = num_shape_pairs
        self.total_num_tiles = total_num_tiles
        self.max_num_blocks_per_shape = max_num_blocks_per_shape

        mult = self.config.buffer_mult_iso * self.total_num_tiles
        self.max_num_blocks_broad = int(
            self.max_num_shape_pairs * self.max_num_blocks_per_shape * self.config.buffer_mult_broad
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
            self.iso_buffer_prefix = [wp.zeros(self.input_sizes[i], dtype=wp.int32) for i in range(4)]
            self.iso_buffer_num = [wp.zeros(self.input_sizes[i], dtype=wp.int32) for i in range(4)]
            self.iso_subblock_idx = [wp.zeros(self.input_sizes[i], dtype=wp.uint8) for i in range(4)]
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
            self.max_num_face_contacts = int(config.buffer_mult_contact * self.max_num_iso_voxels)

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

            self.generate_contacts_kernel = get_generate_contacts_kernel(
                self.config.output_contact_surface,
            )

            if self.config.reduce_contacts:
                # Use HydroelasticContactReduction for efficient hashtable-based contact reduction
                # The reducer uses spatial extremes + max-depth per normal bin + voxel-based slots
                reduction_config = HydroelasticReductionConfig(
                    normal_matching=self.config.normal_matching,
                    anchor_contact=self.config.anchor_contact,
                    moment_matching=self.config.moment_matching,
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

    @classmethod
    def _from_model(
        cls, model: Model, config: SDFHydroelasticConfig = None, writer_func: Any = None
    ) -> SDFHydroelastic | None:
        """Create SDFHydroelastic from a model.

        Args:
            model: The simulation model.
            config: Optional configuration for hydroelastic collision handling.
            writer_func: Optional writer function for decoding contacts.

        Returns:
            SDFHydroelastic instance, or None if no hydroelastic shape pairs exist.
        """
        shape_flags = model.shape_flags.numpy()

        # Check if any shapes have hydroelastic flag
        has_hydroelastic = any((flags & ShapeFlags.HYDROELASTIC) for flags in shape_flags)
        if not has_hydroelastic:
            return None

        shape_pairs = model.shape_contact_pairs.numpy()
        num_hydroelastic_pairs = 0
        for shape_a, shape_b in shape_pairs:
            if (shape_flags[shape_a] & ShapeFlags.HYDROELASTIC) and (shape_flags[shape_b] & ShapeFlags.HYDROELASTIC):
                num_hydroelastic_pairs += 1

        if num_hydroelastic_pairs == 0:
            return None

        shape_sdf_shape2blocks = model.shape_sdf_shape2blocks.numpy()

        # Get indices of shapes that can collide and are hydroelastic
        hydroelastic_indices = [
            i
            for i in range(model.shape_count)
            if (shape_flags[i] & ShapeFlags.COLLIDE_SHAPES) and (shape_flags[i] & ShapeFlags.HYDROELASTIC)
        ]

        # Verify all hydroelastic shapes have scale baked into their SDF
        shape_sdf_data = model.shape_sdf_data.numpy()
        for idx in hydroelastic_indices:
            if not shape_sdf_data[idx]["scale_baked"]:
                raise ValueError(f"Hydroelastic shape {idx} does not have scale baked into its SDF.")

        # Count total tiles and max blocks per shape for hydroelastic shapes
        total_num_tiles = 0
        max_num_blocks_per_shape = 0
        for idx in hydroelastic_indices:
            start_block, end_block = shape_sdf_shape2blocks[idx]
            num_blocks = end_block - start_block
            total_num_tiles += num_blocks
            max_num_blocks_per_shape = max(max_num_blocks_per_shape, num_blocks)

        return cls(
            num_shape_pairs=num_hydroelastic_pairs,
            total_num_tiles=total_num_tiles,
            max_num_blocks_per_shape=max_num_blocks_per_shape,
            shape_sdf_block_coords=model.shape_sdf_block_coords,
            shape_sdf_shape2blocks=model.shape_sdf_shape2blocks,
            shape_material_k_hydro=model.shape_material_k_hydro,
            n_shapes=model.shape_count,
            config=config,
            device=model.device,
            writer_func=writer_func,
        )

    def get_hydro_contact_surface(self) -> HydroelasticContactSurfaceData | None:
        """Get the hydroelastic contact surface data for visualization.

        Returns:
            HydroelasticContactSurfaceData containing vertex arrays and metadata for rendering,
            or None if `output_contact_surface` is False in the config.
        """
        if not self.config.output_contact_surface:
            return None
        return HydroelasticContactSurfaceData(
            contact_surface_point=self.iso_vertex_point,
            contact_surface_depth=self.iso_vertex_depth,
            contact_surface_shape_pair=self.iso_vertex_shape_pair,
            face_contact_count=self.contact_reduction.contact_count,
            max_num_face_contacts=self.max_num_face_contacts,
        )

    def set_output_contact_surface(self, enabled: bool) -> None:
        """Toggle contact surface visualization at runtime.

        Note: This is a no-op. When `output_contact_surface=True` in the config,
        the kernel always writes surface data. Display is controlled by the
        viewer's `show_hydro_contact_surface` flag. This method exists for API
        compatibility with ``CollisionPipeline``.
        """
        pass

    def launch(
        self,
        shape_sdf_data: wp.array(dtype=SDFData),
        shape_transform: wp.array(dtype=wp.transform),
        shape_contact_margin: wp.array(dtype=wp.float32),
        shape_local_aabb_lower: wp.array(dtype=wp.vec3),
        shape_local_aabb_upper: wp.array(dtype=wp.vec3),
        shape_voxel_resolution: wp.array(dtype=wp.vec3i),
        shape_pairs_sdf_sdf: wp.array(dtype=wp.vec2i),
        shape_pairs_sdf_sdf_count: wp.array(dtype=wp.int32),
        writer_data: Any,
    ) -> None:
        """Run the full hydroelastic collision pipeline.

        Args:
            shape_sdf_data: SDF data for each shape.
            shape_transform: World transforms for each shape.
            shape_contact_margin: Contact margin for each shape.
            shape_local_aabb_lower: Per-shape local AABB lower bounds.
            shape_local_aabb_upper: Per-shape local AABB upper bounds.
            shape_voxel_resolution: Per-shape voxel grid resolution.
            shape_pairs_sdf_sdf: Pairs of shape indices to check for collision.
            shape_pairs_sdf_sdf_count: Number of valid shape pairs.
            writer_data: Contact data writer for output.
        """
        self._broadphase_sdfs(
            shape_sdf_data,
            shape_transform,
            shape_pairs_sdf_sdf,
            shape_pairs_sdf_sdf_count,
        )

        self._find_iso_voxels(shape_sdf_data, shape_transform, shape_contact_margin)

        self._generate_contacts(shape_sdf_data, shape_transform, shape_contact_margin)

        if self.config.reduce_contacts:
            self._reduce_decode_contacts(
                shape_transform,
                shape_local_aabb_lower,
                shape_local_aabb_upper,
                shape_voxel_resolution,
                shape_contact_margin,
                writer_data,
            )
        else:
            self._decode_contacts(
                shape_transform,
                shape_contact_margin,
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
            ],
            device=self.device,
        )

    def _broadphase_sdfs(
        self,
        shape_sdf_data: wp.array(dtype=SDFData),
        shape_transform: wp.array(dtype=wp.transform),
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
        shape_transform: wp.array(dtype=wp.transform),
        shape_contact_margin: wp.array(dtype=wp.float32),
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
                    shape_transform,
                    self.shape_material_k_hydro,
                    self.iso_buffer_coords[i],
                    self.iso_buffer_shape_pairs[i],
                    shape_contact_margin,
                    subblock_size,
                    n_blocks,
                    self.input_sizes[i],
                ],
                outputs=[
                    self.iso_buffer_num[i],
                    self.iso_subblock_idx[i],
                ],
                device=self.device,
            )

            scan_with_total(
                self.iso_buffer_num[i],
                self.iso_buffer_prefix[i],
                self.iso_buffer_counts[i],
                self.iso_buffer_counts[i + 1],
            )

            wp.launch(
                kernel=scatter_iso_subblock,
                dim=[self.grid_size],
                inputs=[
                    self.grid_size,
                    self.iso_buffer_counts[i],
                    self.iso_buffer_prefix[i],
                    self.iso_subblock_idx[i],
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
        shape_transform: wp.array(dtype=wp.transform),
        shape_contact_margin: wp.array(dtype=wp.float32),
    ) -> None:
        """Generate marching cubes contacts and write directly to the contact buffer.

        Single pass: compute cube state and immediately write faces to reducer buffer.
        """
        self.contact_reduction.clear()
        reducer_data = self.contact_reduction.get_data_struct()
        wp.launch(
            kernel=self.generate_contacts_kernel,
            dim=[self.grid_size],
            inputs=[
                self.grid_size,
                self.iso_voxel_count,
                shape_sdf_data,
                shape_transform,
                self.shape_material_k_hydro,
                self.iso_voxel_coords,
                self.iso_voxel_shape_pair,
                self.mc_tables[0],
                self.mc_tables[4],
                self.mc_tables[3],
                shape_contact_margin,
                self.max_num_iso_voxels,
                reducer_data,
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
        shape_contact_margin: wp.array(dtype=wp.float32),
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
                self.shape_material_k_hydro,
                shape_transform,
                shape_contact_margin,
                self.contact_reduction.reducer.position_depth,
                self.contact_reduction.reducer.normal,
                self.contact_reduction.reducer.shape_pairs,
                self.contact_reduction.reducer.contact_area,
                self.contact_reduction.reducer.contact_k_eff,
                self.max_num_face_contacts,
            ],
            outputs=[writer_data],
            device=self.device,
        )

    def _reduce_decode_contacts(
        self,
        shape_transform: wp.array(dtype=wp.transform),
        shape_local_aabb_lower: wp.array(dtype=wp.vec3),
        shape_local_aabb_upper: wp.array(dtype=wp.vec3),
        shape_voxel_resolution: wp.array(dtype=wp.vec3i),
        shape_contact_margin: wp.array(dtype=wp.float32),
        writer_data: Any,
    ) -> None:
        """Reduce and decode hydroelastic contacts.

        Contacts are already in the buffer (written by _generate_contacts).
        This method registers them in the hashtable and exports the reduced set.
        """
        self.contact_reduction.reduce_and_export(
            shape_transform=shape_transform,
            shape_local_aabb_lower=shape_local_aabb_lower,
            shape_local_aabb_upper=shape_local_aabb_upper,
            shape_voxel_resolution=shape_voxel_resolution,
            shape_contact_margin=shape_contact_margin,
            writer_data=writer_data,
            grid_size=self.grid_size,
        )


@wp.kernel(enable_backward=False)
def broadphase_collision_pairs_count(
    shape_transform: wp.array(dtype=wp.transform),
    shape_sdf_data: wp.array(dtype=SDFData),
    shape_pairs_sdf_sdf: wp.array(dtype=wp.vec2i),
    shape_pairs_sdf_sdf_count: wp.array(dtype=wp.int32),
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
    half_extents_a = shape_sdf_data[shape_a].half_extents
    half_extents_b = shape_sdf_data[shape_b].half_extents

    center_offset_a = shape_sdf_data[shape_a].center
    center_offset_b = shape_sdf_data[shape_b].center

    does_collide = wp.bool(False)

    world_transform_a = shape_transform[shape_a]
    world_transform_b = shape_transform[shape_b]

    # Apply center offset to transforms (since SAT assumes centered boxes)
    centered_transform_a = wp.transform_multiply(world_transform_a, wp.transform(center_offset_a, wp.quat_identity()))
    centered_transform_b = wp.transform_multiply(world_transform_b, wp.transform(center_offset_b, wp.quat_identity()))

    does_collide = sat_box_intersection(centered_transform_a, half_extents_a, centered_transform_b, half_extents_b)

    # Sort shapes so shape with smaller voxel size is shape_b (must match scatter kernel)
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

    # sort shapes such that the shape with the smaller voxel size is in second place
    # NOTE: Confirm that this is OK to do for downstream code
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
def get_rel_stiffness(k_a: wp.float32, k_b: wp.float32) -> tuple[wp.float32, wp.float32]:
    k_m_inv = 1.0 / wp.sqrt(k_a * k_b)
    return k_a * k_m_inv, k_b * k_m_inv


@wp.func
def sdf_diff_sdf(
    sdfA_data: SDFData,
    sdfB_data: SDFData,
    transfA: wp.transform,
    transfB: wp.transform,
    k_eff_a: wp.float32,
    k_eff_b: wp.float32,
    x_id: wp.int32,
    y_id: wp.int32,
    z_id: wp.int32,
) -> tuple[wp.float32, wp.float32, wp.float32, wp.bool]:
    """Compute signed distance difference between two SDFs at a voxel position.

    SDF A is queried directly on the sparse grid since we know the voxel is allocated.
    SDF B is queried using extrapolation to handle points outside the narrow band or extent.
    """
    sdfA = sdfA_data.sparse_sdf_ptr
    pointA = wp.volume_index_to_world(sdfA, int_to_vec3f(x_id, y_id, z_id))
    pointA_world = wp.transform_point(transfA, pointA)
    pointB = wp.transform_point(wp.transform_inverse(transfB), pointA_world)
    valA = wp.volume_lookup_f(sdfA, x_id, y_id, z_id)

    valB = sample_sdf_extrapolated(sdfB_data, pointB)

    is_valid = not (
        valA >= wp.static(MAXVAL * 0.99) or wp.isnan(valA) or valB >= wp.static(MAXVAL * 0.99) or wp.isnan(valB)
    )

    if valA < 0 and valB < 0:
        diff = k_eff_a * valA - k_eff_b * valB
    else:
        diff = valA - valB
    return diff, valA, valB, is_valid


@wp.func
def sdf_diff_sdf(
    sdfA_data: SDFData,
    sdfB_data: SDFData,
    transfA: wp.transform,
    transfB: wp.transform,
    k_eff_a: wp.float32,
    k_eff_b: wp.float32,
    pos_a_local: wp.vec3,
) -> tuple[wp.float32, wp.float32, wp.float32, wp.bool]:
    """Compute signed distance difference between two SDFs at a local position.

    SDF A is queried directly on the sparse grid since we know the voxel is allocated.
    SDF B is queried using extrapolation to handle points outside the narrow band or extent.
    """
    sdfA = sdfA_data.sparse_sdf_ptr
    pointA = wp.volume_index_to_world(sdfA, pos_a_local)
    pointA_world = wp.transform_point(transfA, pointA)
    pointB = wp.transform_point(wp.transform_inverse(transfB), pointA_world)
    valA = wp.volume_sample_f(sdfA, pos_a_local, wp.Volume.LINEAR)

    valB = sample_sdf_extrapolated(sdfB_data, pointB)

    is_valid = not (
        valA >= wp.static(MAXVAL * 0.99) or wp.isnan(valA) or valB >= wp.static(MAXVAL * 0.99) or wp.isnan(valB)
    )

    if valA < 0 and valB < 0:
        diff = k_eff_a * valA - k_eff_b * valB
    else:
        diff = valA - valB
    return diff, valA, valB, is_valid


@wp.kernel(enable_backward=False)
def count_iso_voxels_block(
    grid_size: int,
    in_buffer_collide_count: wp.array(dtype=int),
    shape_sdf_data: wp.array(dtype=SDFData),
    shape_transform: wp.array(dtype=wp.transform),
    shape_material_k_hydro: wp.array(dtype=float),
    in_buffer_collide_coords: wp.array(dtype=wp.vec3us),
    in_buffer_collide_shape_pair: wp.array(dtype=wp.vec2i),
    shape_contact_margin: wp.array(dtype=wp.float32),
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

        X_ws_a = shape_transform[shape_a]
        X_ws_b = shape_transform[shape_b]

        margin_a = shape_contact_margin[shape_a]
        margin_b = shape_contact_margin[shape_b]

        voxel_radius = sdf_data_b.sparse_voxel_radius
        r = float(subblock_size) * voxel_radius

        k_a = shape_material_k_hydro[shape_a]
        k_b = shape_material_k_hydro[shape_b]

        k_eff_a, k_eff_b = get_rel_stiffness(k_a, k_b)
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
                        sdf_data_b, sdf_data_a, X_ws_b, X_ws_a, k_eff_b, k_eff_a, x_center
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
    X_ws: wp.transform,
    X_ws_other: wp.transform,
    k_eff: wp.float32,
    k_eff_other: wp.float32,
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
            sdf_data, sdf_other_data, X_ws, X_ws_other, k_eff, k_eff_other, x, y, z
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
        shape_material_k_hydro: wp.array(dtype=wp.float32),
        shape_transform: wp.array(dtype=wp.transform),
        shape_contact_margin: wp.array(dtype=wp.float32),
        position_depth: wp.array(dtype=wp.vec4),
        normal: wp.array(dtype=wp.vec3),
        shape_pairs: wp.array(dtype=wp.vec2i),
        contact_area: wp.array(dtype=wp.float32),
        contact_k_eff: wp.array(dtype=wp.float32),
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
            contact_normal = normal[tid]

            normal_world = wp.transform_vector(transform_b, contact_normal)
            pos_world = wp.transform_point(transform_b, pos)

            # Sum margins for consistency with thickness summing
            margin_a = shape_contact_margin[shape_a]
            margin_b = shape_contact_margin[shape_b]
            margin = margin_a + margin_b

            k_eff = contact_k_eff[tid]
            area = contact_area[tid]

            # Compute stiffness, use margin_contact_area for non-penetrating contacts
            # Standard convention: depth < 0 = penetrating
            if depth < 0.0:
                c_stiffness = area * k_eff
            else:
                c_stiffness = wp.static(margin_contact_area) * k_eff

            # Create ContactData for the writer function
            # contact_distance = 2 * depth (depth is negative for penetrating)
            contact_data = ContactData()
            contact_data.contact_point_center = pos_world
            contact_data.contact_normal_a_to_b = normal_world
            contact_data.contact_distance = 2.0 * depth
            contact_data.radius_eff_a = 0.0
            contact_data.radius_eff_b = 0.0
            contact_data.thickness_a = 0.0
            contact_data.thickness_b = 0.0
            contact_data.shape_a = shape_a
            contact_data.shape_b = shape_b
            contact_data.margin = margin
            contact_data.contact_stiffness = c_stiffness

            writer_func(contact_data, writer_data, output_index)

    return decode_contacts_kernel


# =============================================================================
# Contact generation kernels
# =============================================================================


def get_generate_contacts_kernel(output_vertices: bool):
    """Create kernel for hydroelastic contact generation.

    This is a merged kernel that computes cube state and immediately writes
    faces to the reducer buffer in a single pass, eliminating intermediate
    storage for cube indices and corner values.

    Args:
        output_vertices: Whether to output contact surface vertices for visualization

    Returns:
        generate_contacts_kernel: Warp kernel for contact generation
    """

    @wp.kernel(enable_backward=False)
    def generate_contacts_kernel(
        grid_size: int,
        iso_voxel_count: wp.array(dtype=wp.int32),
        shape_sdf_data: wp.array(dtype=SDFData),
        shape_transform: wp.array(dtype=wp.transform),
        shape_material_k_hydro: wp.array(dtype=float),
        iso_voxel_coords: wp.array(dtype=wp.vec3us),
        iso_voxel_shape_pair: wp.array(dtype=wp.vec2i),
        tri_range_table: wp.array(dtype=wp.int32),
        flat_edge_verts_table: wp.array(dtype=wp.vec2ub),
        corner_offsets_table: wp.array(dtype=wp.vec3ub),
        shape_contact_margin: wp.array(dtype=wp.float32),
        max_num_iso_voxels: int,
        reducer_data: GlobalContactReducerData,
        # outputs for visualization (optional)
        iso_vertex_point: wp.array(dtype=wp.vec3f),
        iso_vertex_depth: wp.array(dtype=wp.float32),
        iso_vertex_shape_pair: wp.array(dtype=wp.vec2i),
    ):
        """Generate marching cubes contacts and write directly to GlobalContactReducer.

        Computes cube state and immediately writes faces to the reducer buffer
        in a single pass, using atomic allocation for buffer slots.
        """
        offset = wp.tid()
        num_voxels = wp.min(iso_voxel_count[0], max_num_iso_voxels)
        for tid in range(offset, num_voxels, grid_size):
            pair = iso_voxel_shape_pair[tid]
            shape_a = pair[0]
            shape_b = pair[1]

            sdf_data_a = shape_sdf_data[shape_a]
            sdf_data_b = shape_sdf_data[shape_b]

            transform_a = shape_transform[shape_a]
            transform_b = shape_transform[shape_b]

            iso_coords = iso_voxel_coords[tid]

            margin_a = shape_contact_margin[shape_a]
            margin_b = shape_contact_margin[shape_b]
            margin = margin_a + margin_b

            k_a = shape_material_k_hydro[shape_a]
            k_b = shape_material_k_hydro[shape_b]

            k_eff_a, k_eff_b = get_rel_stiffness(k_a, k_b)

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
                transform_b,
                transform_a,
                k_eff_b,
                k_eff_a,
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
            k_eff = get_effective_stiffness(k_a, k_b)

            sdf_b = sdf_data_b.sparse_sdf_ptr
            X_ws_b = transform_b

            # Generate faces and write directly to reducer buffer
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

                # Write directly to reducer buffer with atomic allocation
                contact_id = export_hydroelastic_contact_to_buffer(
                    shape_a,
                    shape_b,
                    face_center,  # Position in SDF local space
                    normal,
                    pen_depth,
                    area,
                    k_eff,
                    reducer_data,
                )

                # Write debug surface vertices if enabled (compile-time check only)
                # The viewer controls whether to display this data via show_hydro_contact_surface
                if wp.static(output_vertices) and contact_id >= 0:
                    for vi in range(3):
                        iso_vertex_point[3 * contact_id + vi] = wp.transform_point(X_ws_b, face_verts[vi])
                    iso_vertex_depth[contact_id] = pen_depth
                    iso_vertex_shape_pair[contact_id] = pair

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
):
    # Checks if any buffer overflowed in any stage of the collision pipeline and print a warning
    if num_broad_collide[0] > max_num_broad_collide:
        wp.printf(
            "Warning: Broad phase buffer overflowed %d > %d. Increase buffer_mult_broad.\n",
            num_broad_collide[0],
            max_num_broad_collide,
        )
    if num_iso_subblocks_0[0] > max_num_iso_subblocks_0:
        wp.printf(
            "Warning: Iso subblock 0 buffer overflowed %d > %d. Increase buffer_mult_iso.\n",
            num_iso_subblocks_0[0],
            max_num_iso_subblocks_0,
        )
    if num_iso_subblocks_1[0] > max_num_iso_subblocks_1:
        wp.printf(
            "Warning: Iso subblock 1 buffer overflowed %d > %d. Increase buffer_mult_iso.\n",
            num_iso_subblocks_1[0],
            max_num_iso_subblocks_1,
        )
    if num_iso_subblocks_2[0] > max_num_iso_subblocks_2:
        wp.printf(
            "Warning: Iso subblock 2 buffer overflowed %d > %d. Increase buffer_mult_iso.\n",
            num_iso_subblocks_2[0],
            max_num_iso_subblocks_2,
        )
    if num_iso_voxels[0] > max_num_iso_voxels:
        wp.printf(
            "Warning: Iso voxel buffer overflowed %d > %d. Increase buffer_mult_iso.\n",
            num_iso_voxels[0],
            max_num_iso_voxels,
        )
    if face_contact_count[0] > max_face_contact_count:
        wp.printf(
            "Warning: Face contact buffer overflowed %d > %d. Increase buffer_mult_contact.\n",
            face_contact_count[0],
            max_face_contact_count,
        )
    if contact_count[0] > max_contact_count:
        wp.printf(
            "Warning: Contact buffer overflowed %d > %d. Increase contact buffer size.\n",
            contact_count[0],
            max_contact_count,
        )
