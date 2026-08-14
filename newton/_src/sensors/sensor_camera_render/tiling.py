# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import warp as wp


@wp.func
def tid_to_coord_tiled(
    tid: wp.int32,
    width: wp.int32,
    height: wp.int32,
    tile_width: wp.int32,
    tile_height: wp.int32,
):
    num_pixels_per_tile = tile_width * tile_height
    num_tiles_per_row = (width + tile_width - 1) // tile_width
    num_tiles_per_col = (height + tile_height - 1) // tile_height
    num_pixels_per_view = num_tiles_per_row * num_tiles_per_col * num_pixels_per_tile

    pixel_idx = tid % num_pixels_per_view
    world_index = tid // num_pixels_per_view

    tile_idx = pixel_idx // num_pixels_per_tile
    tile_pixel_idx = pixel_idx % num_pixels_per_tile

    tile_y = tile_idx // num_tiles_per_row
    tile_x = tile_idx % num_tiles_per_row

    py = tile_y * tile_height + tile_pixel_idx // tile_width
    px = tile_x * tile_width + tile_pixel_idx % tile_width

    return world_index, py, px


@wp.func
def tid_to_coord_pixel_priority(tid: wp.int32, view_count: wp.int32, width: wp.int32):
    pixel_idx = tid // view_count
    view_index = tid % view_count

    py = pixel_idx // width
    px = pixel_idx % width

    return view_index, py, px


@wp.func
def tid_to_coord_view_priority(tid: wp.int32, width: wp.int32, height: wp.int32):
    num_pixels_per_view = width * height

    pixel_idx = tid % num_pixels_per_view
    world_index = tid // num_pixels_per_view

    py = pixel_idx // width
    px = pixel_idx % width

    return world_index, py, px


@wp.func
def pack_rgba_to_uint32(rgb: wp.vec3f, alpha: wp.float32) -> wp.uint32:
    """Pack RGBA values into a single uint32 for efficient memory access."""
    # Clamp in floating point before the uint32 cast: casting a negative component
    # first wraps to a large unsigned value that clamp() then saturates to 255,
    # turning e.g. a black pixel white.
    r = wp.uint32(wp.clamp(rgb[0], 0.0, 1.0) * 255.0)
    g = wp.uint32(wp.clamp(rgb[1], 0.0, 1.0) * 255.0)
    b = wp.uint32(wp.clamp(rgb[2], 0.0, 1.0) * 255.0)
    a = wp.uint32(wp.clamp(alpha, 0.0, 1.0) * 255.0)
    return (a << wp.uint32(24)) | (b << wp.uint32(16)) | (g << wp.uint32(8)) | r
