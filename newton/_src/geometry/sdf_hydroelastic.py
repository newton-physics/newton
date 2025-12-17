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

from dataclasses import dataclass
from typing import Any

import warp as wp

from ..sim.model import Model
from .collision_core import build_pair_key2, sat_box_intersection
from .contact_data import ContactData
from .contact_reduction import (
    NUM_NORMAL_BINS,
    NUM_SPATIAL_DIRECTIONS,
    get_slot_flat,
    get_spatial_direction_2d,
    project_point_to_plane,
)
from .sdf_contact import sample_sdf_extrapolated
from .sdf_mc import get_mc_tables, mc_calc_face
from .sdf_utils import SDFData
from .utils import scan_with_total

vec8f = wp.types.vector(length=8, dtype=wp.float32)


@wp.func
def int_to_vec3f(x: wp.int32, y: wp.int32, z: wp.int32):
    return wp.vec3f(float(x), float(y), float(z))


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
    """Whether to reduce contacts."""
    buffer_mult_broad: int = 1
    """Multiplier for buffer size for broadphase."""
    buffer_mult_iso: int = 2
    """Multiplier for buffer size for iso voxel traversal."""
    buffer_mult_contact: int = 1
    """Multiplier for buffer size for contact handling."""
    grid_size: int = 256 * 8 * 128
    """Grid size for contact handling. Can be tuned for performance."""
    output_contact_surface: bool = False
    """Whether to output hydroelastic contact surface vertices for visualization."""
    betas: tuple[float, float] = (10.0, -0.5)
    """Penetration beta values."""
    sticky_contacts: float = 0.0
    """Stickiness factor for temporal contact persistence."""
    normal_matching: bool = True
    """Whether to match the aggregated force direction."""
    moment_matching: bool = False
    """Whether to match the reference torque from all contacts."""
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
            self.face_contact_count = wp.zeros((1,), dtype=wp.int32)

            # Broadphase buffers
            self.block_start_prefix = wp.zeros((self.max_num_shape_pairs,), dtype=wp.int32)
            self.num_blocks_per_pair = wp.zeros((self.max_num_shape_pairs,), dtype=wp.int32)
            self.block_broad_idx = wp.empty((self.max_num_blocks_broad,), dtype=wp.int32)
            self.block_broad_collide_coords = self.iso_buffer_coords[0]
            self.block_broad_collide_shape_pair = self.iso_buffer_shape_pairs[0]

            # Iso voxel buffers
            self.voxel_face_count = wp.zeros((self.max_num_iso_voxels,), dtype=wp.int32)
            self.voxel_face_prefix = wp.zeros((self.max_num_iso_voxels,), dtype=wp.int32)
            self.voxel_cube_indices = wp.zeros((self.max_num_iso_voxels,), dtype=wp.uint8)
            self.voxel_corner_vals = wp.zeros((self.max_num_iso_voxels,), dtype=vec8f)

            # Face contact buffers
            self.max_num_face_contacts = 2 * self.max_num_iso_voxels
            self.face_contact_pair = wp.empty((self.max_num_face_contacts,), dtype=wp.vec2i)
            self.face_contact_pos = wp.empty((self.max_num_face_contacts,), dtype=wp.vec3)
            self.face_contact_normal = wp.empty((self.max_num_face_contacts,), dtype=wp.vec3)
            self.face_contact_depth = wp.empty((self.max_num_face_contacts,), dtype=wp.float32)
            self.face_contact_id = wp.empty((self.max_num_face_contacts,), dtype=wp.int32)
            self.face_contact_area = wp.empty((self.max_num_face_contacts,), dtype=wp.float32)
            self.contact_normal_bin_idx = wp.empty((self.max_num_face_contacts,), dtype=wp.int32)

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

            self.count_faces_kernel, self.scatter_faces_kernel = get_generate_contacts_kernel(
                self.config.output_contact_surface,
            )

            if self.config.reduce_contacts:
                self.penetration_betas = wp.array(self.config.betas, dtype=wp.float32)
                self.num_betas = len(self.config.betas)
                self.bin_directions = NUM_SPATIAL_DIRECTIONS
                num_normal_bins = NUM_NORMAL_BINS

                self.max_num_bins = self.max_num_shape_pairs
                self.sparse_pair_size = self.n_shapes * self.n_shapes

                self.shape_pairs_mask = wp.zeros(self.sparse_pair_size, dtype=wp.int32)
                self.shape_pairs_to_bin = wp.zeros((self.sparse_pair_size,), dtype=wp.int32)
                self.shape_pairs_to_bin_prev = wp.clone(self.shape_pairs_to_bin)
                self.bin_to_shape_pair = wp.zeros((self.max_num_bins,), dtype=wp.int32)
                self.num_total_pairs = wp.array((self.max_num_shape_pairs,), dtype=wp.int32)
                self.num_active_pairs = wp.zeros((1,), dtype=wp.int32)

                n_slots = (
                    self.num_betas * self.bin_directions + 1
                )  # track the max dot product for each beta and direction + contact with deepest penetration depth
                self.binned_normals = wp.zeros((self.max_num_bins, num_normal_bins, n_slots), dtype=wp.vec3f)
                self.binned_pos = wp.zeros((self.max_num_bins, num_normal_bins, n_slots), dtype=wp.vec3f)
                self.binned_depth = wp.zeros((self.max_num_bins, num_normal_bins, n_slots), dtype=wp.float32)
                self.binned_dot_product = wp.zeros((self.max_num_bins, num_normal_bins, n_slots), dtype=wp.float32)
                self.binned_id = wp.full((self.max_num_bins, num_normal_bins, n_slots), dtype=wp.int32, value=-1)
                self.binned_id_prev = wp.clone(self.binned_id)

                self.bin_occupied = wp.zeros((self.max_num_bins, num_normal_bins), dtype=wp.bool)
                self.binned_agg_force = wp.zeros((self.max_num_bins, num_normal_bins), dtype=wp.vec3f)
                self.binned_weighted_pos_sum = wp.zeros((self.max_num_bins, num_normal_bins), dtype=wp.vec3f)
                self.binned_weight_sum = wp.zeros((self.max_num_bins, num_normal_bins), dtype=wp.float32)
                self.binned_agg_moment = wp.zeros((self.max_num_bins, num_normal_bins), dtype=wp.float32)

                self.compute_bin_scores, self.assign_contacts_to_bins, self.generate_contacts_from_bins = (
                    get_binning_kernels(
                        self.bin_directions,
                        num_normal_bins,
                        self.num_betas,
                        self.config.sticky_contacts,
                        self.config.normal_matching,
                        self.config.moment_matching,
                        self.config.margin_contact_area,
                        writer_func,
                    )
                )
            else:
                self.decode_contacts_kernel = get_decode_contacts_kernel(self.config.margin_contact_area, writer_func)

            self.max_depth = wp.zeros((1,), dtype=wp.float32)

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

        from ..sim.builder import ShapeFlags  # noqa: PLC0415

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
            face_contact_count=self.face_contact_count,
            max_num_face_contacts=self.max_num_face_contacts,
        )

    def launch(
        self,
        shape_sdf_data: wp.array(dtype=SDFData),
        shape_transform: wp.array(dtype=wp.transform),
        shape_contact_margin: wp.array(dtype=wp.float32),
        shape_pairs_sdf_sdf: wp.array(dtype=wp.vec2i),
        shape_pairs_sdf_sdf_count: wp.array(dtype=wp.int32),
        writer_data: Any,
    ) -> None:
        """Run the full hydroelastic collision pipeline.

        Args:
            shape_sdf_data: SDF data for each shape.
            shape_transform: World transforms for each shape.
            shape_contact_margin: Contact margin for each shape.
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
                self.face_contact_count,
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
                    self.iso_buffer_num[i],
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
        self.voxel_face_count.zero_()
        wp.launch(
            kernel=self.count_faces_kernel,
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
                self.mc_tables[3],
                shape_contact_margin,
                self.max_num_iso_voxels,
            ],
            outputs=[
                self.voxel_face_count,
                self.voxel_cube_indices,
                self.voxel_corner_vals,
            ],
            device=self.device,
        )

        scan_with_total(self.voxel_face_count, self.voxel_face_prefix, self.iso_voxel_count, self.face_contact_count)

        wp.launch(
            kernel=self.scatter_faces_kernel,
            dim=[self.grid_size],
            inputs=[
                self.grid_size,
                self.iso_voxel_count,
                shape_sdf_data,
                shape_transform,
                self.iso_voxel_coords,
                self.iso_voxel_shape_pair,
                self.mc_tables[0],
                self.mc_tables[4],
                self.mc_tables[3],
                self.max_num_face_contacts,
                self.max_num_iso_voxels,
                self.voxel_face_prefix,
                self.voxel_cube_indices,
                self.voxel_corner_vals,
                self.voxel_face_count,
            ],
            outputs=[
                self.face_contact_pair,
                self.face_contact_pos,
                self.face_contact_depth,
                self.face_contact_normal,
                self.face_contact_id,
                self.face_contact_area,
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
        wp.launch(
            kernel=self.decode_contacts_kernel,
            dim=[self.grid_size],
            inputs=[
                self.grid_size,
                self.face_contact_count,
                self.shape_material_k_hydro,
                shape_transform,
                shape_contact_margin,
                self.face_contact_pair,
                self.face_contact_pos,
                self.face_contact_depth,
                self.face_contact_normal,
                self.face_contact_area,
                self.max_num_face_contacts,
            ],
            outputs=[writer_data],
            device=self.device,
        )

    def _reduce_decode_contacts(
        self,
        shape_transform: wp.array(dtype=wp.transform),
        shape_contact_margin: wp.array(dtype=wp.float32),
        writer_data: Any,
    ) -> None:
        wp.copy(self.binned_id_prev, self.binned_id)
        wp.copy(self.shape_pairs_to_bin_prev, self.shape_pairs_to_bin)

        self.binned_dot_product.fill_(-1e10)
        self.binned_agg_force.zero_()
        self.binned_weighted_pos_sum.zero_()
        self.binned_weight_sum.zero_()
        self.binned_agg_moment.zero_()
        self.bin_occupied.zero_()
        self.shape_pairs_mask.zero_()
        self.bin_to_shape_pair.fill_(-1)
        self.num_active_pairs.zero_()

        # Pass 1: Mark active shape pairs from face_contact_pair (vec2i)
        wp.launch(
            mark_active_shape_pairs,
            dim=[self.max_num_face_contacts],
            inputs=[
                self.face_contact_pair,
                self.face_contact_count,
                self.n_shapes,
            ],
            outputs=[self.shape_pairs_mask],
            device=self.device,
        )

        # Pass 2: Prefix sum to get contiguous bin indices
        wp.utils.array_scan(self.shape_pairs_mask, self.shape_pairs_to_bin, inclusive=False)

        wp.launch(
            kernel=self.compute_bin_scores,
            dim=[self.grid_size],
            inputs=[
                self.grid_size,
                self.face_contact_count,
                self.face_contact_normal,
                self.face_contact_pos,
                self.face_contact_depth,
                self.face_contact_area,
                self.face_contact_id,
                self.face_contact_pair,
                self.n_shapes,
                self.shape_pairs_to_bin,
                self.penetration_betas,
                self.binned_id_prev,
                self.shape_pairs_to_bin_prev,
            ],
            outputs=[
                self.binned_dot_product,
                self.binned_agg_force,
                self.bin_occupied,
                self.contact_normal_bin_idx,
                self.bin_to_shape_pair,
                self.binned_weighted_pos_sum,
                self.binned_weight_sum,
            ],
            device=self.device,
        )

        wp.launch(
            kernel=self.assign_contacts_to_bins,
            dim=[self.grid_size],
            inputs=[
                self.grid_size,
                self.face_contact_count,
                self.face_contact_normal,
                self.contact_normal_bin_idx,
                self.face_contact_pos,
                self.face_contact_depth,
                self.face_contact_area,
                self.face_contact_pair,
                self.n_shapes,
                self.face_contact_id,
                self.shape_pairs_to_bin,
                self.penetration_betas,
                self.binned_dot_product,
                self.binned_id_prev,
                self.shape_pairs_to_bin_prev,
                self.binned_weighted_pos_sum,
                self.binned_weight_sum,
            ],
            outputs=[
                self.binned_normals,
                self.binned_pos,
                self.binned_depth,
                self.binned_id,
                self.binned_agg_moment,
            ],
            device=self.device,
        )

        wp.launch(
            kernel=self.generate_contacts_from_bins,
            dim=[self.binned_pos.shape[0], self.binned_pos.shape[1]],
            inputs=[
                shape_transform,
                shape_contact_margin,
                self.shape_material_k_hydro,
                self.n_shapes,
                self.bin_to_shape_pair,
                self.binned_normals,
                self.binned_pos,
                self.binned_depth,
                self.binned_id,
                self.binned_dot_product,
                self.bin_occupied,
                self.binned_agg_force,
                self.binned_agg_moment,
                self.binned_weighted_pos_sum,
                self.binned_weight_sum,
            ],
            outputs=[
                writer_data,
            ],
            device=self.device,
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
def get_effective_stiffness(k_a: wp.float32, k_b: wp.float32) -> wp.float32:
    return (k_a * k_b) / (k_a + k_b)


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

    is_valid = not (valA >= wp.inf or wp.isnan(valA) or valB >= wp.inf or wp.isnan(valB))

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

    is_valid = not (valA >= wp.inf or wp.isnan(valA) or valB >= wp.inf or wp.isnan(valB))

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
        the = r * (k_eff_a + k_eff_b)

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
                    if wp.abs(diff_val) > the or va > r + margin_a or vb > r + margin_b or not is_valid:
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
    in_iso_subblock_num: wp.array(dtype=int),
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
        num = in_iso_subblock_num[tid]
        bc = in_buffer_collide_coords[tid]
        if write_idx + num > max_num_iso_subblocks:
            continue
        for i in range(8):
            bit_pos = wp.uint8(i)
            if (subblock_idx >> bit_pos) & wp.uint8(1):
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


@wp.func
def get_face_id(x_id: wp.int32, y_id: wp.int32, z_id: wp.int32, fi: wp.int32) -> wp.int32:
    # generate a unique contact id based on the voxel coordinates and face index (assumes max 512 voxels per axis)
    return x_id & 0x1FF | (y_id & 0x1FF) << 9 | (z_id & 0x1FF) << 18 | (fi & 0x07) << 27


def get_generate_contacts_kernel(output_vertices: bool):
    @wp.kernel(enable_backward=False)
    def count_faces_kernel(
        grid_size: int,
        iso_voxel_count: wp.array(dtype=wp.int32),
        shape_sdf_data: wp.array(dtype=SDFData),
        shape_transform: wp.array(dtype=wp.transform),
        shape_material_k_hydro: wp.array(dtype=float),
        iso_voxel_coords: wp.array(dtype=wp.vec3us),
        iso_voxel_shape_pair: wp.array(dtype=wp.vec2i),
        tri_range_table: wp.array(dtype=wp.int32),
        corner_offsets_table: wp.array(dtype=wp.vec3ub),
        shape_contact_margin: wp.array(dtype=wp.float32),
        max_num_iso_voxels: int,
        # outputs
        voxel_face_count: wp.array(dtype=wp.int32),
        voxel_cube_indices: wp.array(dtype=wp.uint8),
        voxel_corner_vals: wp.array(dtype=vec8f),
    ):
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
            margin = wp.max(margin_a, margin_b)

            k_a = shape_material_k_hydro[shape_a]
            k_b = shape_material_k_hydro[shape_b]

            k_eff_a, k_eff_b = get_rel_stiffness(k_a, k_b)

            x_id = wp.int32(iso_coords.x)
            y_id = wp.int32(iso_coords.y)
            z_id = wp.int32(iso_coords.z)

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
            # look up the tri range for the cube index
            tri_range_start = tri_range_table[range_idx]
            tri_range_end = tri_range_table[range_idx + 1]
            num_verts = tri_range_end - tri_range_start  # number of intersected edges

            num_faces = num_verts // 3

            if not any_verts_inside or not all_verts_valid:
                num_faces = 0

            voxel_face_count[tid] = num_faces
            voxel_cube_indices[tid] = cube_idx
            voxel_corner_vals[tid] = corner_vals

    @wp.kernel(enable_backward=False)
    def scatter_faces_kernel(
        grid_size: int,
        iso_voxel_count: wp.array(dtype=int),
        shape_sdf_data: wp.array(dtype=SDFData),
        shape_transform: wp.array(dtype=wp.transform),
        iso_voxel_coords: wp.array(dtype=wp.vec3us),
        iso_voxel_shape_pair: wp.array(dtype=wp.vec2i),
        tri_range_table: wp.array(dtype=wp.int32),
        flat_edge_verts_table: wp.array(dtype=wp.vec2ub),
        corner_offsets_table: wp.array(dtype=wp.vec3ub),
        max_num_contacts: int,
        max_num_iso_voxels: int,
        face_contact_prefix: wp.array(dtype=wp.int32),
        voxel_cube_indices: wp.array(dtype=wp.uint8),
        voxel_corner_vals: wp.array(dtype=vec8f),
        voxel_face_count: wp.array(dtype=wp.int32),
        # outputs
        contact_pair: wp.array(dtype=wp.vec2i),
        contact_pos: wp.array(dtype=wp.vec3),
        contact_depth: wp.array(dtype=wp.float32),
        contact_normal: wp.array(dtype=wp.vec3),
        contact_id: wp.array(dtype=wp.int32),
        contact_area: wp.array(dtype=wp.float32),
        iso_vertex_point: wp.array(dtype=wp.vec3f),
        iso_vertex_depth: wp.array(dtype=wp.float32),
        iso_vertex_shape_pair: wp.array(dtype=wp.vec2i),
    ):
        offset = wp.tid()
        num_voxels = wp.min(iso_voxel_count[0], max_num_iso_voxels)
        for tid in range(offset, num_voxels, grid_size):
            num_faces = voxel_face_count[tid]
            idx_base = face_contact_prefix[tid]
            if num_faces == 0 or idx_base + num_faces > max_num_contacts:
                continue

            pair = iso_voxel_shape_pair[tid]
            shape_b = pair[1]

            sdf_b = shape_sdf_data[shape_b].sparse_sdf_ptr

            iso_coords = iso_voxel_coords[tid]
            X_ws_b = shape_transform[shape_b]

            x_id = wp.int32(iso_coords.x)
            y_id = wp.int32(iso_coords.y)
            z_id = wp.int32(iso_coords.z)

            cube_idx = voxel_cube_indices[tid]
            corner_vals = voxel_corner_vals[tid]

            tri_range_start = tri_range_table[wp.int32(cube_idx)]

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

                cid = get_face_id(x_id, y_id, z_id, fi)

                idx = idx_base + fi

                contact_pair[idx] = pair
                contact_pos[idx] = face_center
                contact_depth[idx] = pen_depth
                contact_normal[idx] = normal
                contact_id[idx] = cid
                contact_area[idx] = area

                if wp.static(output_vertices):
                    for vi in range(3):
                        iso_vertex_point[3 * idx + vi] = wp.transform_point(X_ws_b, face_verts[vi])
                    iso_vertex_depth[idx] = pen_depth
                    iso_vertex_shape_pair[idx] = pair

    return count_faces_kernel, scatter_faces_kernel


@wp.func
def compute_score(spatial_dot_product: wp.float32, pen_depth: wp.float32, beta: wp.float32) -> wp.float32:
    if beta < 0.0:
        if pen_depth < 0.0:
            return pen_depth
        else:
            return spatial_dot_product * wp.pow(pen_depth, -beta)
    else:
        return spatial_dot_product + pen_depth * beta


def get_decode_contacts_kernel(margin_contact_area: float = 1e-4, writer_func: Any = None):
    @wp.kernel(enable_backward=False)
    def decode_contacts_kernel(
        grid_size: int,
        contact_count: wp.array(dtype=int),
        shape_material_k_hydro: wp.array(dtype=wp.float32),
        shape_transform: wp.array(dtype=wp.transform),
        shape_contact_margin: wp.array(dtype=wp.float32),
        contact_pair: wp.array(dtype=wp.vec2i),
        contact_pos: wp.array(dtype=wp.vec3),
        contact_depth: wp.array(dtype=wp.float32),
        contact_normal: wp.array(dtype=wp.vec3),
        contact_area: wp.array(dtype=wp.float32),
        max_num_face_contacts: int,
        # outputs
        writer_data: Any,
    ):
        offset = wp.tid()

        num_contacts = wp.min(contact_count[0], max_num_face_contacts)

        # Calculate how many contacts this thread will process
        my_contact_count = 0
        if offset < num_contacts:
            my_contact_count = (num_contacts - 1 - offset) // grid_size + 1

        if my_contact_count == 0:
            return
        # Single atomic to reserve all slots for this thread
        my_base_index = wp.atomic_add(writer_data.contact_count, 0, my_contact_count)

        # Calculate how many slots we can actually write
        remaining_capacity = writer_data.contact_max - my_base_index
        if remaining_capacity <= 0:
            # Roll back the entire reservation since we can't write anything
            wp.atomic_sub(writer_data.contact_count, 0, my_contact_count)
            return

        writable = wp.min(my_contact_count, remaining_capacity)
        if writable < my_contact_count:
            # Roll back the excess reservation
            wp.atomic_sub(writer_data.contact_count, 0, my_contact_count - writable)

        # Write contacts using reserved range
        local_idx = int(0)
        for tid in range(offset, num_contacts, grid_size):
            if local_idx >= writable:
                return  # Already wrote all we can

            output_index = my_base_index + local_idx
            local_idx += 1

            pair = contact_pair[tid]
            shape_a = pair[0]
            shape_b = pair[1]

            transform_b = shape_transform[shape_b]

            depth = contact_depth[tid]
            normal = contact_normal[tid]
            pos = contact_pos[tid]

            normal_world = wp.transform_vector(transform_b, normal)
            pos_world = wp.transform_point(transform_b, pos)

            # Use per-shape contact margin (max of both shapes)
            margin_a = shape_contact_margin[shape_a]
            margin_b = shape_contact_margin[shape_b]
            margin = wp.max(margin_a, margin_b)

            k_a = shape_material_k_hydro[shape_a]
            k_b = shape_material_k_hydro[shape_b]

            k_eff = get_effective_stiffness(k_a, k_b)
            # Compute stiffness, use margin_contact_area for non-penetrating contacts
            if depth > 0.0:
                c_stiffness = contact_area[tid] * k_eff
            else:
                c_stiffness = wp.static(margin_contact_area) * k_eff

            # Create ContactData for the writer function
            contact_data = ContactData()
            contact_data.contact_point_center = pos_world
            contact_data.contact_normal_a_to_b = normal_world
            contact_data.contact_distance = -2.0 * depth  # depth is the distance to the isosurface
            contact_data.radius_eff_a = 0.0
            contact_data.radius_eff_b = 0.0
            contact_data.thickness_a = 0.0
            contact_data.thickness_b = 0.0
            contact_data.shape_a = shape_a
            contact_data.shape_b = shape_b
            contact_data.margin = margin
            contact_data.feature = wp.uint32(tid + 1)
            contact_data.feature_pair_key = build_pair_key2(wp.uint32(shape_a), wp.uint32(shape_b))
            contact_data.contact_stiffness = c_stiffness

            writer_func(contact_data, writer_data, output_index)

    return decode_contacts_kernel


@wp.kernel(enable_backward=False)
def iso_shape_pair_mask(
    iso_voxel_shape_pair: wp.array(dtype=wp.int32),
    iso_voxel_count: wp.array(dtype=int),
    shape_pairs_mask: wp.array(dtype=wp.int32),
):
    tid = wp.tid()
    if tid >= iso_voxel_count[0]:
        return
    shape_pair_idx = iso_voxel_shape_pair[tid]
    shape_pairs_mask[shape_pair_idx] = 1


@wp.kernel(enable_backward=False)
def mark_active_shape_pairs(
    face_contact_pair: wp.array(dtype=wp.vec2i),
    face_contact_count: wp.array(dtype=wp.int32),
    n_shapes: int,
    shape_pairs_mask: wp.array(dtype=wp.int32),
):
    tid = wp.tid()
    if tid >= face_contact_count[0]:
        return
    pair = face_contact_pair[tid]
    sparse_idx = pair[0] * n_shapes + pair[1]
    shape_pairs_mask[sparse_idx] = 1


def get_binning_kernels(
    n_bin_dirs: int,
    num_normal_bins: int,
    num_betas: int,
    sticky_contacts: float = 1e-6,
    normal_matching: bool = True,
    moment_matching: bool = True,
    margin_contact_area: float = 1e-4,
    writer_func: Any = None,
):
    """
    Factory method for creating binning kernels for hydroelastic contacts.

    Args:
        n_bin_dirs: Number of spatial bin directions.
        num_normal_bins: Number of normal bins.
        num_betas: Number of penetration beta values.
        sticky_contacts: Stickiness factor for temporal contact persistence.
        normal_matching: If True, rotate original normals so their weighted sum aligns with
            the aggregated force direction.
        moment_matching: If True, attempt to match the reference maximum moment from unreduced contacts.
        margin_contact_area: Contact area used for non-penetrating contacts at the margin.
        writer_func: Function to write contact data.
    """

    @wp.kernel(enable_backward=False)
    def compute_bin_scores(
        grid_size: int,
        contact_count: wp.array(dtype=int),
        contact_normals: wp.array(dtype=wp.vec3f),
        contact_pos: wp.array(dtype=wp.vec3f),
        contact_depth: wp.array(dtype=wp.float32),
        contact_area: wp.array(dtype=wp.float32),
        contact_id: wp.array(dtype=wp.int32),
        contact_pair: wp.array(dtype=wp.vec2i),
        n_shapes: int,
        shape_pairs_to_bin: wp.array(dtype=wp.int32),
        penetration_betas: wp.array(dtype=wp.float32),
        binned_id_prev: wp.array(dtype=wp.int32, ndim=3),
        shape_pairs_to_bin_prev: wp.array(dtype=wp.int32),
        # outputs
        binned_dot_product: wp.array(dtype=wp.float32, ndim=3),
        binned_agg_force: wp.array(dtype=wp.vec3f, ndim=2),
        bin_occupied: wp.array(dtype=wp.bool, ndim=2),
        contact_normal_bin_idx: wp.array(dtype=wp.int32),
        bin_to_shape_pair: wp.array(dtype=wp.int32),
        binned_weighted_pos_sum: wp.array(dtype=wp.vec3f, ndim=2),
        binned_weight_sum: wp.array(dtype=wp.float32, ndim=2),
    ):
        offset = wp.tid()
        for tid in range(offset, contact_count[0], grid_size):
            pair = contact_pair[tid]
            sparse_idx = pair[0] * n_shapes + pair[1]
            bin_idx_0 = shape_pairs_to_bin[sparse_idx]
            normal = contact_normals[tid]
            face_center = contact_pos[tid]
            pen_depth = contact_depth[tid]
            area = contact_area[tid]
            id = contact_id[tid]
            # find the normal bin which is closest to the face normal (in body frame of b)
            bin_normal_idx = get_slot_flat(normal)

            bin_to_shape_pair[bin_idx_0] = sparse_idx
            bin_occupied[bin_idx_0, bin_normal_idx] = True
            # aggregate force direction weighted by force magnitude (only penetrating contacts)
            contact_normal_bin_idx[tid] = bin_normal_idx
            force_weight = area * pen_depth

            # accumulate for penetrating contacts only
            if pen_depth > 0.0:
                wp.atomic_add(binned_agg_force, bin_idx_0, bin_normal_idx, force_weight * normal)
                wp.atomic_add(binned_weighted_pos_sum, bin_idx_0, bin_normal_idx, force_weight * face_center)
                wp.atomic_add(binned_weight_sum, bin_idx_0, bin_normal_idx, force_weight)

            bin_idx_prev = shape_pairs_to_bin_prev[sparse_idx]

            face_center_2d = project_point_to_plane(bin_normal_idx, face_center)

            # track the max dot product for the deepest penetration depth
            wp.atomic_max(binned_dot_product, bin_idx_0, bin_normal_idx, wp.static(num_betas * n_bin_dirs), pen_depth)

            # Loop over bin_directions, store the max dot product for each direction
            for dir_idx in range(wp.static(n_bin_dirs)):
                direction_2d = get_spatial_direction_2d(dir_idx)
                spatial_dot_product = wp.dot(face_center_2d, direction_2d)
                for i in range(wp.static(num_betas)):
                    offset_i = i * n_bin_dirs
                    idx_dir = dir_idx + offset_i
                    dp = compute_score(spatial_dot_product, pen_depth, penetration_betas[i])
                    if wp.static(sticky_contacts > 0.0):
                        bin_id_prev = binned_id_prev[bin_idx_prev, bin_normal_idx, idx_dir]
                        if bin_id_prev == id:
                            dp += wp.static(sticky_contacts)

                    wp.atomic_max(binned_dot_product, bin_idx_0, bin_normal_idx, idx_dir, dp)

    @wp.kernel(enable_backward=False)
    def assign_contacts_to_bins(
        grid_size: int,
        contact_count: wp.array(dtype=int),
        contact_normals: wp.array(dtype=wp.vec3f),
        contact_normal_bin_idx: wp.array(dtype=wp.int32),
        contact_pos: wp.array(dtype=wp.vec3f),
        contact_depth: wp.array(dtype=wp.float32),
        contact_area: wp.array(dtype=wp.float32),
        contact_pair: wp.array(dtype=wp.vec2i),
        n_shapes: int,
        contact_id: wp.array(dtype=wp.int32),
        shape_pairs_to_bin: wp.array(dtype=wp.int32),
        penetration_betas: wp.array(dtype=wp.float32),
        binned_dot_product: wp.array(dtype=wp.float32, ndim=3),
        binned_id_prev: wp.array(dtype=wp.int32, ndim=3),
        shape_pairs_to_bin_prev: wp.array(dtype=wp.int32),
        binned_weighted_pos_sum: wp.array(dtype=wp.vec3f, ndim=2),
        binned_weight_sum: wp.array(dtype=wp.float32, ndim=2),
        # outputs
        binned_normals: wp.array(dtype=wp.vec3f, ndim=3),
        binned_pos: wp.array(dtype=wp.vec3f, ndim=3),
        binned_depth: wp.array(dtype=wp.float32, ndim=3),
        binned_id: wp.array(dtype=wp.int32, ndim=3),
        binned_moment: wp.array(dtype=wp.float32, ndim=2),
    ):
        offset = wp.tid()
        for tid in range(offset, contact_count[0], grid_size):
            pair = contact_pair[tid]
            sparse_idx = pair[0] * n_shapes + pair[1]
            bin_idx_0 = shape_pairs_to_bin[sparse_idx]
            bin_normal_idx = contact_normal_bin_idx[tid]

            face_center = contact_pos[tid]
            area = contact_area[tid]
            pen_depth = contact_depth[tid]
            normal = contact_normals[tid]
            id = contact_id[tid]

            # compute mean position from accumulated weighted sum and weight sum
            weight_sum = binned_weight_sum[bin_idx_0, bin_normal_idx]
            if weight_sum > 0.0 and pen_depth > 0.0:
                anchor_pos = binned_weighted_pos_sum[bin_idx_0, bin_normal_idx] / weight_sum
                r_anchor_to_face = face_center - anchor_pos
                max_moment = wp.length(wp.cross(r_anchor_to_face, normal)) * area * pen_depth
                wp.atomic_add(binned_moment, bin_idx_0, bin_normal_idx, max_moment)

            bin_idx_prev = shape_pairs_to_bin_prev[sparse_idx]

            # track the contact with the deepest penetration depth
            max_depth_idx = wp.static(num_betas * n_bin_dirs)
            max_dp_depth = binned_dot_product[bin_idx_0, bin_normal_idx, max_depth_idx]
            if pen_depth >= max_dp_depth:
                binned_normals[bin_idx_0, bin_normal_idx, max_depth_idx] = normal
                binned_pos[bin_idx_0, bin_normal_idx, max_depth_idx] = face_center
                binned_depth[bin_idx_0, bin_normal_idx, max_depth_idx] = pen_depth
                binned_id[bin_idx_0, bin_normal_idx, max_depth_idx] = id

            face_center_2d = project_point_to_plane(bin_normal_idx, face_center)

            # track the max dot product for each beta and direction
            for dir_idx in range(wp.static(n_bin_dirs)):
                direction_2d = get_spatial_direction_2d(dir_idx)
                spatial_dot_product = wp.dot(face_center_2d, direction_2d)
                for i in range(wp.static(num_betas)):
                    offset_i = i * n_bin_dirs
                    idx_dir = dir_idx + offset_i
                    dp = compute_score(spatial_dot_product, pen_depth, penetration_betas[i])
                    if wp.static(sticky_contacts > 0.0):
                        bin_id_prev = binned_id_prev[bin_idx_prev, bin_normal_idx, idx_dir]
                        if bin_id_prev == id:
                            dp += wp.static(sticky_contacts)
                    max_dp = binned_dot_product[bin_idx_0, bin_normal_idx, idx_dir]
                    if dp >= max_dp:
                        binned_normals[bin_idx_0, bin_normal_idx, idx_dir] = normal
                        binned_pos[bin_idx_0, bin_normal_idx, idx_dir] = face_center
                        binned_depth[bin_idx_0, bin_normal_idx, idx_dir] = pen_depth
                        binned_id[bin_idx_0, bin_normal_idx, idx_dir] = id

    @wp.kernel(enable_backward=False)
    def generate_contacts_from_bins(
        shape_transform: wp.array(dtype=wp.transform),
        shape_contact_margin: wp.array(dtype=wp.float32),
        shape_material_k_hydro: wp.array(dtype=float),
        n_shapes: int,
        bin_to_shape_pair: wp.array(dtype=wp.int32),
        binned_normals: wp.array(dtype=wp.vec3f, ndim=3),
        binned_pos: wp.array(dtype=wp.vec3f, ndim=3),
        binned_depth: wp.array(dtype=wp.float32, ndim=3),
        binned_id: wp.array(dtype=wp.int32, ndim=3),
        binned_dot_product: wp.array(dtype=wp.float32, ndim=3),
        bin_occupied: wp.array(dtype=wp.bool, ndim=2),
        binned_agg_force: wp.array(dtype=wp.vec3f, ndim=2),
        binned_moment: wp.array(dtype=wp.float32, ndim=2),
        binned_weighted_pos_sum: wp.array(dtype=wp.vec3f, ndim=2),
        binned_weight_sum: wp.array(dtype=wp.float32, ndim=2),
        # outputs
        writer_data: Any,
    ):
        tid, normal_bin_idx = wp.tid()
        if not bin_occupied[tid, normal_bin_idx]:
            return
        sparse_idx = bin_to_shape_pair[tid]

        # Get the shape pair from sparse index
        shape_a = sparse_idx // n_shapes
        shape_b = sparse_idx % n_shapes

        k_a = shape_material_k_hydro[shape_a]
        k_b = shape_material_k_hydro[shape_b]
        k_eff = get_effective_stiffness(k_a, k_b)

        transform_b = shape_transform[shape_b]

        # Get contact margin
        margin_a = shape_contact_margin[shape_a]
        margin_b = shape_contact_margin[shape_b]
        margin = wp.max(margin_a, margin_b)

        # at this point, we know that each direction has a contact in it, but the same contact id can be present in multiple directions
        # here deduplicate contacts based on contact id
        n_bins = wp.static(num_betas * n_bin_dirs + 1)

        unique_indices = wp.zeros(shape=(n_bins,), dtype=wp.int32)

        # always choose the last contact (with the deepest penetration depth)
        unique_indices[0] = n_bins - 1
        num_unique_contacts = wp.int32(1)
        max_depth = binned_depth[tid, normal_bin_idx, -1]
        num_penetrating_contacts = wp.int32(max_depth > 0.0)
        agg_depth = wp.max(max_depth, 0.0)
        last_bin_id = binned_id[tid, normal_bin_idx, n_bins - 1]
        for i in range(n_bins - 1):
            found_duplicate = wp.bool(False)
            if binned_dot_product[tid, normal_bin_idx, i] <= -1e9:  # not active contact
                continue
            if binned_id[tid, normal_bin_idx, i] == last_bin_id:
                found_duplicate = True
            else:
                for j in range(i - 1, -1, -1):
                    if binned_id[tid, normal_bin_idx, j] == binned_id[tid, normal_bin_idx, i]:
                        found_duplicate = True
                        break
            if not found_duplicate:
                unique_indices[num_unique_contacts] = i
                num_unique_contacts += 1
                if binned_depth[tid, normal_bin_idx, i] > 0.0:
                    num_penetrating_contacts += 1
                    agg_depth += binned_depth[tid, normal_bin_idx, i]

        # Compute the aggregated force and its direction for this normal bin
        agg_force = binned_agg_force[tid, normal_bin_idx]
        agg_force_mag = wp.length(agg_force)

        # Determine if anchor contact will be added
        weight_sum = binned_weight_sum[tid, normal_bin_idx]
        anchor_pos = wp.vec3f(0.0, 0.0, 0.0)
        add_anchor_contact = wp.int32(0)
        if wp.static(moment_matching) and max_depth > 1e-6 and weight_sum > 1e-20:
            anchor_pos = binned_weighted_pos_sum[tid, normal_bin_idx] / weight_sum
            add_anchor_contact = 1

        # Total depth includes anchor contribution if anchor will be added
        total_depth = agg_depth + wp.float32(add_anchor_contact) * max_depth
        c_stiffness = k_eff * agg_force_mag / (total_depth + 1e-8)

        rotation_q = wp.quat_identity()
        if wp.static(normal_matching):
            # Rotate original normals so their weighted sum aligns with aggregated force
            selected_sum_normals = wp.vec3f(0.0, 0.0, 0.0)
            for i in range(num_unique_contacts):
                dir_idx = unique_indices[i]
                depth = binned_depth[tid, normal_bin_idx, dir_idx]
                if depth > 0.0:
                    selected_sum_normals += depth * binned_normals[tid, normal_bin_idx, dir_idx]

            # Add anchor's contribution to the weighted sum (anchor normal = agg_force direction)
            if add_anchor_contact == 1 and agg_force_mag > 1e-8:
                anchor_normal_contribution = max_depth * (agg_force / agg_force_mag)
                selected_sum_normals += anchor_normal_contribution

            # Compute rotation that aligns selected_sum with agg_force
            selected_mag = wp.length(selected_sum_normals)
            if selected_mag > 1e-8 and agg_force_mag > 1e-8:
                selected_dir = selected_sum_normals / selected_mag
                agg_dir = agg_force / agg_force_mag

                cross = wp.cross(selected_dir, agg_dir)
                cross_mag = wp.length(cross)
                dot_val = wp.dot(selected_dir, agg_dir)

                if cross_mag > 1e-8:
                    # Normal case: compute rotation around cross product axis
                    axis = cross / cross_mag
                    angle = wp.acos(wp.clamp(dot_val, -1.0, 1.0))
                    rotation_q = wp.quat_from_axis_angle(axis, angle)
                elif dot_val < 0.0:
                    # Vectors are anti-parallel: rotate 180 degrees around a perpendicular axis
                    perp = wp.vec3f(1.0, 0.0, 0.0)
                    if wp.abs(wp.dot(selected_dir, perp)) > 0.9:
                        perp = wp.vec3f(0.0, 1.0, 0.0)
                    axis = wp.normalize(wp.cross(selected_dir, perp))
                    rotation_q = wp.quat_from_axis_angle(axis, wp.pi)

        unique_friction = wp.float32(1.0)
        anchor_friction = wp.float32(1.0)

        if wp.static(moment_matching) and add_anchor_contact == 1:
            # Moment matching: scale friction to match the moment of the selected contacts to the reference moment
            ref_moment = binned_moment[tid, normal_bin_idx]
            selected_moment = wp.float32(0.0)
            for i in range(num_unique_contacts):
                dir_idx = unique_indices[i]
                depth = binned_depth[tid, normal_bin_idx, dir_idx]
                pos = binned_pos[tid, normal_bin_idx, dir_idx]
                normal = binned_normals[tid, normal_bin_idx, dir_idx]
                if depth > 0.0:
                    selected_moment += depth * wp.length(wp.cross(pos - anchor_pos, normal))

            # Scale unique friction to match moments:
            denom = wp.max(agg_force_mag * selected_moment, 1e-20)
            unique_friction = (ref_moment * total_depth) / denom

            # Compute anchor friction to preserve tangential friction invariant:
            # unique_friction * agg_depth + anchor_friction * max_depth = total_depth
            anchor_friction = (total_depth - unique_friction * agg_depth) / max_depth

            # Joint clamping: if one friction hits lower bound, adjust the other
            min_friction = wp.float32(1e-4)

            if anchor_friction < min_friction:
                anchor_friction = min_friction
                unique_friction = (total_depth - min_friction * max_depth) / wp.max(agg_depth, 1e-20)

            if unique_friction < min_friction:
                unique_friction = min_friction
                anchor_friction = (total_depth - min_friction * agg_depth) / max_depth
                anchor_friction = wp.max(anchor_friction, min_friction)

        # single atomic_add per bin; drop this bin if we can't fit all its contacts
        total_contacts = num_unique_contacts + add_anchor_contact
        if total_contacts == 0:
            return

        contact_idx = wp.atomic_add(writer_data.contact_count, 0, total_contacts)

        if contact_idx + total_contacts > writer_data.contact_max:
            # Roll back reservation and skip this bin to keep counts and writes consistent
            wp.atomic_sub(writer_data.contact_count, 0, total_contacts)
            return

        pair_key = build_pair_key2(wp.uint32(shape_a), wp.uint32(shape_b))

        # Store contacts
        for i in range(num_unique_contacts):
            dir_idx = unique_indices[i]

            original_normal = binned_normals[tid, normal_bin_idx, dir_idx]
            depth = binned_depth[tid, normal_bin_idx, dir_idx]

            if wp.static(normal_matching) and depth > 0.0:
                # Rotate original normal to align sum with aggregated force direction
                normal = wp.normalize(wp.quat_rotate(rotation_q, original_normal))
            else:
                normal = original_normal

            pos = binned_pos[tid, normal_bin_idx, dir_idx]

            # Transform to world space
            normal_world = wp.transform_vector(transform_b, normal)
            pos_world = wp.transform_point(transform_b, pos)

            c_idx = contact_idx + i

            # Create ContactData for the writer function
            contact_data = ContactData()
            contact_data.contact_point_center = pos_world
            contact_data.contact_normal_a_to_b = normal_world
            contact_data.contact_distance = -2.0 * depth
            contact_data.radius_eff_a = 0.0
            contact_data.radius_eff_b = 0.0
            contact_data.thickness_a = 0.0
            contact_data.thickness_b = 0.0
            contact_data.shape_a = shape_a
            contact_data.shape_b = shape_b
            contact_data.margin = margin
            contact_data.feature = wp.uint32(binned_id[tid, normal_bin_idx, dir_idx] + 1)
            contact_data.feature_pair_key = pair_key
            contact_data.contact_stiffness = c_stiffness
            contact_data.contact_friction_scale = unique_friction * wp.float32(depth > 0.0)

            writer_func(contact_data, writer_data, c_idx)

        # Store anchor contact
        if add_anchor_contact == 1:
            anchor_idx = contact_idx + num_unique_contacts

            anchor_normal = wp.normalize(agg_force)
            anchor_normal_world = wp.transform_vector(transform_b, anchor_normal)
            anchor_pos_world = wp.transform_point(transform_b, anchor_pos)

            contact_data = ContactData()
            contact_data.contact_point_center = anchor_pos_world
            contact_data.contact_normal_a_to_b = anchor_normal_world
            contact_data.contact_distance = -2.0 * max_depth
            contact_data.radius_eff_a = 0.0
            contact_data.radius_eff_b = 0.0
            contact_data.thickness_a = 0.0
            contact_data.thickness_b = 0.0
            contact_data.shape_a = shape_a
            contact_data.shape_b = shape_b
            contact_data.margin = margin
            contact_data.feature = wp.uint32(0)
            contact_data.feature_pair_key = pair_key
            contact_data.contact_stiffness = c_stiffness
            contact_data.contact_friction_scale = anchor_friction

            writer_func(contact_data, writer_data, anchor_idx)

    return compute_bin_scores, assign_contacts_to_bins, generate_contacts_from_bins


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
        wp.printf("Warning: Broad phase buffer overflowed %u > %u\n", num_broad_collide[0], max_num_broad_collide)
    if num_iso_subblocks_0[0] > max_num_iso_subblocks_0:
        wp.printf(
            "Warning: Iso subblock 0 buffer overflowed %u > %u\n", num_iso_subblocks_0[0], max_num_iso_subblocks_0
        )
    if num_iso_subblocks_1[0] > max_num_iso_subblocks_1:
        wp.printf(
            "Warning: Iso subblock 1 buffer overflowed %u > %u\n", num_iso_subblocks_1[0], max_num_iso_subblocks_1
        )
    if num_iso_subblocks_2[0] > max_num_iso_subblocks_2:
        wp.printf(
            "Warning: Iso subblock 2 buffer overflowed %u > %u\n", num_iso_subblocks_2[0], max_num_iso_subblocks_2
        )
    if num_iso_voxels[0] > max_num_iso_voxels:
        wp.printf("Warning: Iso voxel buffer overflowed %u > %u\n", num_iso_voxels[0], max_num_iso_voxels)
    if face_contact_count[0] > max_face_contact_count:
        wp.printf("Warning: Face contact buffer overflowed %u > %u\n", face_contact_count[0], max_face_contact_count)
    if contact_count[0] > max_contact_count:
        wp.printf("Warning: Contact buffer overflowed %u > %u\n", contact_count[0], max_contact_count)
