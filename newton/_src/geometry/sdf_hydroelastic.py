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

import numpy as np
import warp as wp

from ..sim.model import Model
from .types import SDFData
from .utils import scan_with_total
from .collision_core import sat_box_intersection
from .sdf_utils import SDFData


vec8f = wp.types.vector(length=8, dtype=wp.float32)

@wp.func
def int_to_vec3f(x: wp.int32, y: wp.int32, z: wp.int32):
    return wp.vec3f(float(x), float(y), float(z))

@dataclass
class SDFHydroelasticConfig:
    """
    Controls properties of SDF hydroelastic collision handling.
    """
    buffer_mult_broad: int = 1
    """Multiplier for buffer size for broadphase."""
    buffer_mult_iso: int = 2
    """Multiplier for buffer size for iso voxel traversal."""
    buffer_mult_contact: int = 1
    """Multiplier for buffer size for contact handling."""
    grid_size: int = 256 * 8 * 128
    """Grid size for contact handling. Can be tuned for performance."""
    output_iso_vertices: bool = False
    """Whether to output iso vertices of isosurfaces."""


class SDFHydroelastic:
    """
    Handles SDF hydroelastic collision handling.
    """
    def __init__(
        self,
        num_shape_pairs: int,
        total_num_tiles: int,
        total_num_voxels: int,
        max_num_blocks_per_shape: int,
        config: SDFHydroelasticConfig = None,
    ):
        if config is None:
            config = SDFHydroelasticConfig()

        self.config = config

        self.max_num_shape_pairs = num_shape_pairs
        self.total_num_tiles = total_num_tiles
        self.total_num_voxels = total_num_voxels
        self.max_num_blocks_per_shape = max_num_blocks_per_shape

        self.num_shape_pairs_array = wp.full((1,), self.max_num_shape_pairs, dtype=wp.int32)

        mult = self.config.buffer_mult_iso * self.total_num_tiles

        self.max_num_blocks_broad = int(
            self.max_num_shape_pairs
            * self.max_num_blocks_per_shape
            * self.config.buffer_mult_broad
        )
        # Output buffer sizes for each octree level (subblocks 8x8x8 -> 4x4x4 -> 2x2x2 -> voxels)
        self.iso_max_dims = (int(2 * mult), int(2 * mult), int(16 * mult), int(32 * mult))
        self.max_num_iso_voxels = self.iso_max_dims[3]
        # Input buffer sizes for each octree level
        input_sizes = (self.max_num_blocks_broad, *self.iso_max_dims[:3])

        # Allocate buffers for octree traversal (broadphase + 4 refinement levels)
        self.iso_buffer_counts = [wp.zeros((1,), dtype=wp.int32) for _ in range(5)]
        self.iso_buffer_prefix = [wp.zeros(input_sizes[i], dtype=wp.int32) for i in range(4)]
        self.iso_buffer_num = [wp.zeros(input_sizes[i], dtype=wp.int32) for i in range(4)]
        self.iso_subblock_idx = [wp.zeros(input_sizes[i], dtype=wp.uint8) for i in range(4)]
        self.iso_buffer_coords = [wp.empty((self.max_num_blocks_broad,), dtype=wp.vec3us)] + [
            wp.empty((self.iso_max_dims[i],), dtype=wp.vec3us) for i in range(4)
        ]
        self.iso_buffer_shape_pairs = [wp.empty((self.max_num_blocks_broad,), dtype=wp.int32)] + [
            wp.empty((self.iso_max_dims[i],), dtype=wp.int32) for i in range(4)
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

        # Contact buffers
        self.max_num_face_contacts = 2 * self.max_num_iso_voxels
        self.face_contact_pair_idx = wp.empty((self.max_num_face_contacts,), dtype=wp.int32)
        self.face_contact_pos = wp.empty((self.max_num_face_contacts,), dtype=wp.vec3)
        self.face_contact_normal = wp.empty((self.max_num_face_contacts,), dtype=wp.vec3)
        self.contact_normal_bin_idx = wp.empty((self.max_num_face_contacts,), dtype=wp.int32)
        self.face_contact_depth = wp.empty((self.max_num_face_contacts,), dtype=wp.float32)
        self.face_contact_id = wp.empty((self.max_num_face_contacts,), dtype=wp.int32)
        self.face_contact_area = wp.empty((self.max_num_face_contacts,), dtype=wp.float32)

        if self.config.output_iso_vertices:
            # stores the point and depth of the iso vertex
            self.iso_vertex_point = wp.empty((3 * self.max_num_face_contacts,), dtype=wp.vec3f)
            self.iso_vertex_depth = wp.empty((self.max_num_face_contacts,), dtype=wp.float32)
            self.iso_vertex_shape_pair = wp.empty((self.max_num_face_contacts,), dtype=wp.int32)
        else:
            self.iso_vertex_point = wp.empty((0,), dtype=wp.vec3f)
            self.iso_vertex_depth = wp.empty((0,), dtype=wp.float32)
            self.iso_vertex_shape_pair = wp.empty((0,), dtype=wp.int32)

        # self.count_faces_kernel, self.scatter_faces_kernel = get_generate_contacts_kernel()

        self.max_depth = wp.zeros((1,), dtype=wp.float32)
        self.grid_size = min(self.config.grid_size, self.max_num_face_contacts)

    @classmethod
    def _from_model(cls, model: Model, config: SDFHydroelasticConfig = None) -> "SDFHydroelastic | None":
        """Create SDFHydroelastic from a model.

        Args:
            model: The simulation model.
            config: Optional configuration for hydroelastic collision handling.

        Returns:
            SDFHydroelastic instance, or None if no hydroelastic shape pairs exist.
        """

        num_hydroelastic_pairs = 0
        shape_pairs = model.shape_contact_pairs.numpy()
        shape_is_hydroelastic = model.shape_is_hydroelastic.numpy()

        for shape_a, shape_b in shape_pairs:
            if shape_is_hydroelastic[shape_a] and shape_is_hydroelastic[shape_b]:
                num_hydroelastic_pairs += 1

        if num_hydroelastic_pairs == 0:
            return None

        shape_is_hydroelastic = model.shape_is_hydroelastic.numpy()
        shape_flags = model.shape_flags.numpy()
        shape_sdf_shape2blocks = model.shape_sdf_shape2blocks.numpy()

        from ..sim.builder import ShapeFlags  # noqa: PLC0415

        # Get indices of shapes that can collide and are hydroelastic
        hydroelastic_indices = [
            i for i in range(model.shape_count)
            if (shape_flags[i] & ShapeFlags.COLLIDE_SHAPES) and shape_is_hydroelastic[i]
        ]

        # Count total tiles and max blocks per shape for hydroelastic shapes
        total_num_tiles = 0
        max_num_blocks_per_shape = 0
        for idx in hydroelastic_indices:
            start_block, end_block = shape_sdf_shape2blocks[idx]
            num_blocks = end_block - start_block
            total_num_tiles += num_blocks
            max_num_blocks_per_shape = max(max_num_blocks_per_shape, num_blocks)

        total_num_voxels = total_num_tiles * 512  # 8*8*8 voxels per tile

        return cls(
            num_shape_pairs=num_hydroelastic_pairs,
            total_num_tiles=total_num_tiles,
            total_num_voxels=total_num_voxels,
            max_num_blocks_per_shape=max_num_blocks_per_shape,
            config=config,
        )

    
    def launch(
        self,
        shape_sdf_data: wp.array(dtype=SDFData),
        shape_transform: wp.array(dtype=wp.transform),
        shape_contact_margin: wp.array(dtype=wp.float32),
        shape_pairs_sdf_sdf: wp.array(dtype=wp.vec2i),
        shape_pairs_sdf_sdf_count: wp.array(dtype=wp.int32),
        shape_sdf_block_coords: wp.array(dtype=wp.vec3us),
        shape_sdf_shape2blocks: wp.array(dtype=wp.vec2i),
        writer_data: Any,
        device: Any = None,
    ) -> None:
        self._broadphase_collision_sdfs(
            shape_sdf_data,
            shape_transform,
            shape_contact_margin,
            shape_pairs_sdf_sdf,
            shape_pairs_sdf_sdf_count,
            shape_sdf_block_coords,
            shape_sdf_shape2blocks,
            device,
        )

    def _broadphase_collision_sdfs(
        self,
        shape_sdf_data: wp.array(dtype=SDFData),
        shape_transform: wp.array(dtype=wp.transform),
        shape_contact_margin: wp.array(dtype=wp.float32),
        shape_pairs_sdf_sdf: wp.array(dtype=wp.vec2i),
        shape_pairs_sdf_sdf_count: wp.array(dtype=wp.int32),
        shape_sdf_block_coords: wp.array(dtype=wp.vec3us),
        shape_sdf_shape2blocks: wp.array(dtype=wp.vec2i),
        device: Any = None,
    ) -> None:
        # broadphase stage 0: Test collisions between OBB shapes
        wp.launch(
            kernel=broadphase_collision_pairs_count,
            dim=[self.max_num_shape_pairs],
            inputs=[
                shape_transform,
                shape_sdf_data,
                shape_pairs_sdf_sdf,
                shape_pairs_sdf_sdf_count,
                shape_sdf_shape2blocks,
            ],
            outputs=[
                self.num_blocks_per_pair,
            ],
            device=device,
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
                self.block_start_prefix,
                shape_pairs_sdf_sdf,
                shape_pairs_sdf_sdf_count,
                shape_sdf_shape2blocks,
            ],
            outputs=[
                self.block_broad_collide_shape_pair,
                self.block_broad_idx,
            ],
            device=device,
        )

        wp.launch(
            kernel=broadphase_get_block_coords,
            dim=[self.grid_size],
            inputs=[
                self.grid_size,
                self.block_broad_collide_count,
                self.block_broad_idx,
                shape_sdf_block_coords,
            ],
            outputs=[
                self.block_broad_collide_coords,
            ],
            device=device,
        )


@wp.kernel
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
    centered_transform_a = wp.transform_multiply(
        world_transform_a, wp.transform(center_offset_a, wp.quat_identity())
    )
    centered_transform_b = wp.transform_multiply(
        world_transform_b, wp.transform(center_offset_b, wp.quat_identity())
    )

    does_collide = sat_box_intersection(centered_transform_a, half_extents_a, centered_transform_b, half_extents_b)

    shape_b_idx = shape2blocks[shape_b]
    block_start, block_end = shape_b_idx[0], shape_b_idx[1]
    num_blocks = block_end - block_start

    if does_collide:
        thread_num_blocks[tid] = num_blocks
    else:
        thread_num_blocks[tid] = 0


@wp.kernel
def broadphase_collision_pairs_scatter(
    thread_num_blocks: wp.array(dtype=wp.int32),
    block_start_prefix: wp.array(dtype=wp.int32),
    shape_pairs_sdf_sdf: wp.array(dtype=wp.vec2i),
    shape_pairs_sdf_sdf_count: wp.array(dtype=wp.int32),
    shape2blocks: wp.array(dtype=wp.vec2i),
    # outputs
    block_broad_collide_shape_pair: wp.array(dtype=wp.int32),
    block_broad_idx: wp.array(dtype=wp.int32),
):
    tid = wp.tid()
    if tid >= shape_pairs_sdf_sdf_count[0]:
        return

    num_blocks = thread_num_blocks[tid]
    if num_blocks == 0:
        return

    pair = shape_pairs_sdf_sdf[tid]
    shape_b = pair[1]
    shape_b_idx = shape2blocks[shape_b]
    shape_b_block_start = shape_b_idx[0]

    block_start = block_start_prefix[tid]
    for i in range(num_blocks):
        block_broad_collide_shape_pair[block_start + i] = tid
        block_broad_idx[block_start + i] = shape_b_block_start + i


@wp.kernel
def broadphase_get_block_coords(
    grid_size: int,
    block_count: wp.array(dtype=wp.int32),
    block_broad_idx: wp.array(dtype=wp.int32),
    block_coords: wp.array(dtype=wp.vec3us),
    # outputs
    block_broad_collide_coords: wp.array(dtype=wp.vec3us),
):
    offset = wp.tid()
    for tid in range(offset, block_count[0], grid_size):
        block_idx = block_broad_idx[tid]
        block_broad_collide_coords[tid] = block_coords[block_idx]