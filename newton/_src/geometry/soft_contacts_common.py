# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Back-end-neutral helpers shared by the full-surface rigid-soft contact passes.

Both the SDF back-end (:mod:`soft_contacts_sdf`) and the BVH back-end (:mod:`soft_contacts_bvh`)
resolve shape frames the same way and append records into the same unified soft-contact stream;
keeping the helpers here prevents the two implementations from drifting.
"""

import warp as wp


@wp.func
def _shape_frames(
    shape_body: wp.array[wp.int32],
    body_q: wp.array[wp.transform],
    shape_transform: wp.array[wp.transform],
    shape_index: wp.int32,
):
    """Return (X_bs, X_ws, X_sw): shape-local->body, shape-local->world, world->shape-local."""
    rigid_body = shape_body[shape_index]
    X_wb = wp.transform_identity()
    if rigid_body >= 0:
        X_wb = body_q[rigid_body]
    X_bs = shape_transform[shape_index]
    X_ws = wp.transform_multiply(X_wb, X_bs)
    X_sw = wp.transform_inverse(X_ws)
    return X_bs, X_ws, X_sw


@wp.func
def _write_soft_contact(
    idx: wp.int32,
    soft_contact_particle: wp.array[wp.int32],
    soft_contact_indices: wp.array[wp.vec3i],
    soft_contact_barycentric: wp.array[wp.vec3],
    soft_contact_shape: wp.array[wp.int32],
    soft_contact_rigid_indices: wp.array[wp.vec3i],
    soft_contact_body_pos: wp.array[wp.vec3],
    soft_contact_body_vel: wp.array[wp.vec3],
    soft_contact_normal: wp.array[wp.vec3],
    particle: wp.int32,
    corners: wp.vec3i,
    bary: wp.vec3,
    shape_index: wp.int32,
    rigid_indices: wp.vec3i,
    body_pos: wp.vec3,
    body_vel: wp.vec3,
    normal: wp.vec3,
):
    """Write one record of the unified soft-contact stream at ``idx`` (no-op when ``idx < 0``).

    ``idx`` must come from :func:`kernels.counter_increment` called **directly in the kernel
    body** -- never from inside a nested ``wp.func``: Warp's ``@wp.func_replay`` substitution does
    not apply through nested calls, so a nested counter silently re-runs the real atomic during
    backward replay, misroutes every adjoint, and double-increments the counter. (The
    one-emission-per-thread replay contract also still applies: a taped kernel may call
    ``counter_increment`` at most once per thread.)

    ``particle`` keeps the particle-only compatibility view alive for vertex records (``-1`` for
    edge/face records, which have no single particle id).
    """
    if idx >= 0:
        soft_contact_particle[idx] = particle
        soft_contact_indices[idx] = corners
        soft_contact_barycentric[idx] = bary
        soft_contact_shape[idx] = shape_index
        soft_contact_rigid_indices[idx] = rigid_indices
        soft_contact_body_pos[idx] = body_pos
        soft_contact_body_vel[idx] = body_vel
        soft_contact_normal[idx] = normal
