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

import warp as wp


@wp.func
def tid_to_coord_tiled(
    tid: wp.int32,
    camera_count: wp.int32,
    width: wp.int32,
    height: wp.int32,
    tile_width: wp.int32,
    tile_height: wp.int32,
):
    num_pixels_per_view = width * height
    num_pixels_per_tile = tile_width * tile_height
    num_tiles_per_row = width // tile_width

    pixel_idx = tid % num_pixels_per_view
    view_idx = tid // num_pixels_per_view

    world_index = view_idx // camera_count
    camera_index = view_idx % camera_count

    tile_idx = pixel_idx // num_pixels_per_tile
    tile_pixel_idx = pixel_idx % num_pixels_per_tile

    tile_y = tile_idx // num_tiles_per_row
    tile_x = tile_idx % num_tiles_per_row

    py = tile_y * tile_height + tile_pixel_idx // tile_width
    px = tile_x * tile_width + tile_pixel_idx % tile_width

    return world_index, camera_index, py, px


@wp.func
def tid_to_coord_pixel_priority(tid: wp.int32, world_count: wp.int32, camera_count: wp.int32, width: wp.int32):
    num_views_per_pixel = world_count * camera_count

    pixel_idx = tid // num_views_per_pixel
    view_idx = tid % num_views_per_pixel

    world_index = view_idx % world_count
    camera_index = view_idx // world_count

    py = pixel_idx // width
    px = pixel_idx % width

    return world_index, camera_index, py, px


@wp.func
def tid_to_coord_view_priority(tid: wp.int32, camera_count: wp.int32, width: wp.int32, height: wp.int32):
    num_pixels_per_view = width * height

    pixel_idx = tid % num_pixels_per_view
    view_idx = tid // num_pixels_per_view

    world_index = view_idx // camera_count
    camera_index = view_idx % camera_count

    py = pixel_idx // width
    px = pixel_idx % width

    return world_index, camera_index, py, px


@wp.func
def pack_rgba_to_uint32(rgb: wp.vec3f, alpha: wp.float32) -> wp.uint32:
    """Pack RGBA values into a single uint32 for efficient memory access."""
    return (
        (wp.clamp(wp.uint32(alpha * 255.0), wp.uint32(0), wp.uint32(255)) << wp.uint32(24))
        | (wp.clamp(wp.uint32(rgb[2] * 255.0), wp.uint32(0), wp.uint32(255)) << wp.uint32(16))
        | (wp.clamp(wp.uint32(rgb[1] * 255.0), wp.uint32(0), wp.uint32(255)) << wp.uint32(8))
        | wp.clamp(wp.uint32(rgb[0] * 255.0), wp.uint32(0), wp.uint32(255))
    )
