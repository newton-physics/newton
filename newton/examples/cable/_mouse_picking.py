# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Mouse-picking helpers for cable examples."""

from __future__ import annotations

import warp as wp


@wp.kernel(enable_backward=False)
def zero_picked_body_torque_kernel(
    body_f: wp.array[wp.spatial_vector],
    pick_body: wp.array[int],
    linear_only_picking_body_mask: wp.array[wp.int32],
):
    """Keep viewer linear force but remove torque when the picked body is a cable body."""
    body_id = pick_body[0]
    if body_id < 0 or body_id >= linear_only_picking_body_mask.shape[0]:
        return
    if linear_only_picking_body_mask[body_id] == 0:
        return

    force = wp.spatial_top(body_f[body_id])
    body_f[body_id] = wp.spatial_vector(force, wp.vec3(0.0))


def make_linear_only_picking_body_mask(body_ids, body_count: int, device) -> wp.array | None:
    """Create a device mask for bodies whose mouse picking should be translation-only."""
    ids = sorted({int(body_id) for body_id in body_ids})
    if not ids:
        return None

    mask = [0] * int(body_count)
    for body_id in ids:
        if body_id < 0 or body_id >= body_count:
            raise ValueError(f"Body id {body_id} is outside the model body range [0, {body_count}).")
        mask[body_id] = 1
    return wp.array(mask, dtype=wp.int32, device=device)


def apply_viewer_forces_with_linear_only_picking(viewer, state, linear_only_picking_body_mask: wp.array | None) -> None:
    """Apply viewer forces, then remove mouse-picking torque from selected cable bodies."""
    viewer.apply_forces(state)
    if linear_only_picking_body_mask is None:
        return

    picking = getattr(viewer, "picking", None)
    pick_body = getattr(picking, "pick_body", None)
    if pick_body is None:
        return

    wp.launch(
        zero_picked_body_torque_kernel,
        dim=1,
        inputs=[state.body_f, pick_body, linear_only_picking_body_mask],
        device=state.body_f.device,
    )
