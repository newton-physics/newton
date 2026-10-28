# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Warp kernels used only by :mod:`newton._src.viewer.viewer_viser`."""

import warp as wp


@wp.kernel
def detect_shape_opacity_changes(
    shape_opacities: wp.array[wp.float32],
    previous_shape_opacities: wp.array[wp.float32],
    changed: wp.array[wp.int32],
):
    """Set a device flag when any packed shape opacity changed."""
    tid = wp.tid()
    if shape_opacities[tid] != previous_shape_opacities[tid]:
        wp.atomic_add(changed, 0, 1)
