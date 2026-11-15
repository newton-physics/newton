# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Warp kernels used only by :mod:`newton._src.viewer.viewer_viser`."""

import warp as wp


@wp.kernel
def detect_shape_appearance_changes(
    colors: wp.array[wp.vec3],
    opacities: wp.array[wp.float32],
    slot_to_shape: wp.array[wp.int32],
    packed_colors: wp.array[wp.vec3],
    packed_opacities: wp.array[wp.float32],
    changed: wp.array[wp.int32],
):
    """Repack appearance and detect edits in one pass, using the existing batch buffers."""
    tid = wp.tid()
    shape = slot_to_shape[tid]
    color = colors[shape]
    previous = packed_colors[tid]
    opacity = wp.clamp(opacities[shape], 0.0, 1.0)
    if color[0] != previous[0] or color[1] != previous[1] or color[2] != previous[2]:
        wp.atomic_max(changed, 0, 1)
    if opacity != packed_opacities[tid]:
        wp.atomic_max(changed, 1, 1)
    packed_colors[tid] = color
    packed_opacities[tid] = opacity
