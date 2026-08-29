# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Warp kernels used only by :mod:`newton._src.viewer.viewer_viser`."""

import warp as wp


@wp.kernel
def detect_shape_color_changes(
    shape_colors: wp.array[wp.vec3],
    previous_shape_colors: wp.array[wp.vec3],
    changed: wp.array[wp.int32],
):
    """Set a device flag when any packed shape color changed."""
    tid = wp.tid()
    color = shape_colors[tid]
    previous = previous_shape_colors[tid]
    if color[0] != previous[0] or color[1] != previous[1] or color[2] != previous[2]:
        wp.atomic_add(changed, 0, 1)
