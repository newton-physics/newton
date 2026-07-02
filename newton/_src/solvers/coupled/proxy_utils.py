# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Reusable GPU kernels for proxy coupling between solvers.

These kernels implement the core operations needed for staggered two-way
coupling via proxy bodies or proxy particles:

1. **Sync** -- copy poses/velocities from the driving solver to proxy bodies.
2. **Feedback cancellation** -- remove previously applied coupling forces,
   external force inputs, and gravity from proxy dynamics to prevent
   double-counting.
3. **Harvest feedback** -- accumulate proxy feedback from destination solver
   momentum changes.
"""

from __future__ import annotations

from typing import Any

import warp as wp

# ------------------------------------------------------------------
# 1. Sync proxy states
# ------------------------------------------------------------------


@wp.kernel(enable_backward=False)
def sync_proxy_states_kernel(
    src_body_q: wp.array[wp.transform],
    src_body_qd: wp.array[wp.spatial_vector],
    source_local_to_proxy_local: wp.array[int],
    dst_body_q: wp.array[wp.transform],
    dst_body_qd: wp.array[wp.spatial_vector],
):
    """Copy body poses and velocities from a source solver to proxy bodies in a destination solver.

    Args:
        src_body_q: Source solver begin-of-step body transforms.
        src_body_qd: Source solver begin-of-step body velocities.
        source_local_to_proxy_local: Dense map from source-local body id to
            proxy-local body id. ``-1`` means no proxy exists for that source
            body.
        dst_body_q: Destination solver body transforms (written for proxies).
        dst_body_qd: Destination solver body velocities (written for proxies).
    """
    source_local_id = wp.tid()
    proxy_local_id = source_local_to_proxy_local[source_local_id]

    if proxy_local_id >= 0:
        dst_body_q[proxy_local_id] = src_body_q[source_local_id]
        dst_body_qd[proxy_local_id] = src_body_qd[source_local_id]


@wp.kernel(enable_backward=False)
def sync_proxy_particles_kernel(
    src_particle_q: wp.array[wp.vec3],
    src_particle_qd: wp.array[wp.vec3],
    source_local_to_proxy_local: wp.array[int],
    dst_particle_q: wp.array[wp.vec3],
    dst_particle_qd: wp.array[wp.vec3],
):
    """Copy particle positions and velocities from a source solver to proxy particles."""
    source_local_id = wp.tid()
    proxy_local_id = source_local_to_proxy_local[source_local_id]

    if proxy_local_id >= 0:
        dst_particle_q[proxy_local_id] = src_particle_q[source_local_id]
        dst_particle_qd[proxy_local_id] = src_particle_qd[source_local_id]


# ------------------------------------------------------------------
# 2. Rewind proxy velocities
# ------------------------------------------------------------------


@wp.kernel(enable_backward=False)
def subtract_proxy_body_forces_kernel(
    body_gravity_acceleration: wp.array[wp.vec3],
    dst_body_f: wp.array[wp.spatial_vector],
    coupling_forces: wp.array[wp.spatial_vector],
    body_local_to_proxy_global: wp.array[int],
    dst_body_mass: wp.array[float],
    dst_body_inv_mass: wp.array[float],
):
    """Subtract lagged proxy feedback and gravity from destination body force inputs.

    Args:
        body_gravity_acceleration: Per-body acceleration applied internally by
            the destination solver's gravity-like forces [m/s^2].
        dst_body_f: Destination body force inputs (written in-place).
        coupling_forces: Spatial forces previously applied to the driving solver,
            indexed by global proxy body id.
        body_local_to_proxy_global: Dense map from local body id to global
            proxy body id. ``-1`` entries are skipped.
        dst_body_mass: Destination body masses [kg].
        dst_body_inv_mass: Destination inverse masses.
    """
    local_id = wp.tid()
    global_id = body_local_to_proxy_global[local_id]
    if global_id < 0:
        return

    f = coupling_forces[global_id]

    inv_m = dst_body_inv_mass[local_id]
    g = body_gravity_acceleration[local_id]
    f_grav = wp.vec3(0.0, 0.0, 0.0)
    if inv_m > 0.0:
        f_grav = dst_body_mass[local_id] * g

    dst_body_f[local_id] = -f - wp.spatial_vector(f_grav, wp.vec3(0.0, 0.0, 0.0))


@wp.kernel(enable_backward=False)
def subtract_proxy_particle_forces_kernel(
    dt: float,
    particle_gravity_acceleration: wp.array[wp.vec3],
    dst_particle_f: wp.array[wp.vec3],
    coupling_forces: wp.array[wp.vec3],
    particle_local_to_proxy_global: wp.array[int],
    dst_particle_inv_mass: wp.array[float],
    dst_particle_qd: wp.array[wp.vec3],
):
    """Subtract default velocity-level feedback, particle force inputs, and gravity."""
    local_id = wp.tid()
    global_id = particle_local_to_proxy_global[local_id]
    if global_id < 0:
        return

    inv_m = dst_particle_inv_mass[local_id]
    delta_v = dt * inv_m * (coupling_forces[global_id] + dst_particle_f[local_id])

    g = particle_gravity_acceleration[local_id]
    delta_v_grav = wp.vec3(0.0, 0.0, 0.0)
    if inv_m > 0.0:
        delta_v_grav = dt * g

    dst_particle_qd[local_id] = dst_particle_qd[local_id] - (delta_v + delta_v_grav)


# ------------------------------------------------------------------
# 3. Harvest proxy feedback
# ------------------------------------------------------------------


@wp.kernel(enable_backward=False)
def harvest_proxy_momentum_forces_kernel(
    dt: float,
    body_local_to_proxy_global: wp.array[int],
    qd_before: wp.array[wp.spatial_vector],
    qd_after: wp.array[wp.spatial_vector],
    body_mass: wp.array[float],
    body_inertia: wp.array[wp.mat33],
    body_q: wp.array[wp.transform],
    out_coupling_forces: wp.array[wp.spatial_vector],
):
    """Estimate proxy feedback force from destination velocity change."""
    local_id = wp.tid()
    global_id = body_local_to_proxy_global[local_id]
    if global_id < 0:
        return

    dv = wp.spatial_top(qd_after[local_id]) - wp.spatial_top(qd_before[local_id])
    dw = wp.spatial_bottom(qd_after[local_id]) - wp.spatial_bottom(qd_before[local_id])

    m = body_mass[local_id]
    I_body = body_inertia[local_id]
    r = wp.transform_get_rotation(body_q[local_id])

    f = m * dv / dt
    tau = wp.quat_rotate(r, I_body * wp.quat_rotate_inv(r, dw)) / dt

    wp.atomic_add(out_coupling_forces, global_id, wp.spatial_vector(f, tau))


@wp.kernel(enable_backward=False)
def harvest_proxy_particle_momentum_forces_kernel(
    dt: float,
    particle_local_to_proxy_global: wp.array[int],
    qd_before: wp.array[wp.vec3],
    qd_after: wp.array[wp.vec3],
    particle_mass: wp.array[float],
    particle_flags: wp.array[wp.int32],
    active_flag: int,
    out_coupling_forces: wp.array[wp.vec3],
):
    """Estimate proxy particle feedback force from destination velocity change."""
    local_id = wp.tid()
    global_id = particle_local_to_proxy_global[local_id]
    if global_id < 0:
        return
    if (particle_flags[local_id] & active_flag) == 0:
        return

    dv = qd_after[local_id] - qd_before[local_id]
    m = particle_mass[local_id]

    f = m * dv / dt
    wp.atomic_add(out_coupling_forces, global_id, f)


@wp.kernel(enable_backward=False)
def stash_proxy_forces_kernel(
    proxy_ids_global: wp.array[int],
    coupling_forces: wp.array[Any],
    out_previous_coupling_forces: wp.array[Any],
):
    """Save the current proxy feedback for a later relaxation blend."""
    i = wp.tid()
    out_previous_coupling_forces[i] = coupling_forces[proxy_ids_global[i]]


@wp.kernel(enable_backward=False)
def blend_proxy_forces_kernel(
    proxy_relaxation: float,
    proxy_ids_global: wp.array[int],
    previous_coupling_forces: wp.array[Any],
    coupling_forces: wp.array[Any],
):
    """Blend harvested proxy feedback with the saved lagged value."""
    i = wp.tid()
    global_id = proxy_ids_global[i]
    coupling_forces[global_id] = (
        proxy_relaxation * coupling_forces[global_id] + (1.0 - proxy_relaxation) * previous_coupling_forces[i]
    )


@wp.kernel(enable_backward=False)
def filter_proxy_rigid_contacts_kernel(
    rigid_contact_count: wp.array[int],
    rigid_contact_shape0: wp.array[wp.int32],
    rigid_contact_shape1: wp.array[wp.int32],
    shape_body: wp.array[wp.int32],
    body_flags: wp.array[wp.int32],
    body_inv_mass: wp.array[float],
    proxy_flag: int,
):
    """Invalidate proxy-vs-static and proxy-vs-proxy rigid contacts."""
    contact_id = wp.tid()
    if contact_id >= rigid_contact_count[0]:
        return

    s0 = rigid_contact_shape0[contact_id]
    s1 = rigid_contact_shape1[contact_id]
    body0 = shape_body[s0] if s0 >= 0 and s0 < shape_body.shape[0] else -1
    body1 = shape_body[s1] if s1 >= 0 and s1 < shape_body.shape[0] else -1

    is_proxy0 = 0
    if body0 >= 0 and body0 < body_flags.shape[0]:
        if (body_flags[body0] & proxy_flag) != 0:
            is_proxy0 = 1
    is_proxy1 = 0
    if body1 >= 0 and body1 < body_flags.shape[0]:
        if (body_flags[body1] & proxy_flag) != 0:
            is_proxy1 = 1

    is_static0 = 0
    if body0 < 0:
        is_static0 = 1
    elif body0 < body_inv_mass.shape[0] and body_inv_mass[body0] == 0.0:
        is_static0 = 1

    is_static1 = 0
    if body1 < 0:
        is_static1 = 1
    elif body1 < body_inv_mass.shape[0] and body_inv_mass[body1] == 0.0:
        is_static1 = 1

    discard = 0
    if is_proxy0 == 1 and is_proxy1 == 1:
        discard = 1
    if is_proxy0 == 1 and is_static1 == 1:
        discard = 1
    if is_proxy1 == 1 and is_static0 == 1:
        discard = 1

    if discard == 1:
        if s0 >= 0:
            rigid_contact_shape0[contact_id] = -s0 - 2
        if s1 >= 0:
            rigid_contact_shape1[contact_id] = -s1 - 2


@wp.kernel(enable_backward=False)
def restore_filtered_proxy_rigid_contacts_kernel(
    rigid_contact_count: wp.array[int],
    rigid_contact_shape0: wp.array[wp.int32],
    rigid_contact_shape1: wp.array[wp.int32],
):
    """Restore contacts temporarily encoded by proxy contact filtering."""
    contact_id = wp.tid()
    if contact_id >= rigid_contact_count[0]:
        return

    s0 = rigid_contact_shape0[contact_id]
    s1 = rigid_contact_shape1[contact_id]
    if s0 < -1:
        rigid_contact_shape0[contact_id] = -s0 - 2
    if s1 < -1:
        rigid_contact_shape1[contact_id] = -s1 - 2
