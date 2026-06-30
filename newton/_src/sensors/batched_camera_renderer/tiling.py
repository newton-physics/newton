# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

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

    camera_index = view_idx % camera_count

    tile_idx = pixel_idx // num_pixels_per_tile
    tile_pixel_idx = pixel_idx % num_pixels_per_tile

    tile_y = tile_idx // num_tiles_per_row
    tile_x = tile_idx % num_tiles_per_row

    py = tile_y * tile_height + tile_pixel_idx // tile_width
    px = tile_x * tile_width + tile_pixel_idx % tile_width

    return camera_index, py, px


@wp.func
def tid_to_coord_pixel_priority(tid: wp.int32, camera_count: wp.int32, width: wp.int32):
    num_views_per_pixel = camera_count

    pixel_idx = tid // num_views_per_pixel
    view_idx = tid % num_views_per_pixel

    camera_index = view_idx % camera_count

    py = pixel_idx // width
    px = pixel_idx % width

    return camera_index, py, px


@wp.func
def tid_to_coord_view_priority(tid: wp.int32, camera_count: wp.int32, width: wp.int32, height: wp.int32):
    num_pixels_per_view = width * height

    pixel_idx = tid % num_pixels_per_view
    view_idx = tid // num_pixels_per_view

    camera_index = view_idx % camera_count

    py = pixel_idx // width
    px = pixel_idx % width

    return camera_index, py, px
