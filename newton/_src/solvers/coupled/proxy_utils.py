# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Reusable GPU kernels for proxy coupling between solvers.

These kernels implement the core operations needed for staggered two-way
coupling via proxy bodies or proxy particles:

1. **Sync** -- copy poses/velocities from the driving solver to proxy bodies.
2. **Smooth teleportation** -- encode position jumps as velocity corrections
   to avoid discontinuities in penetration-free solvers.
3. **Velocity rewind** -- remove previously applied coupling forces, external
   force inputs, and gravity from proxy velocities to prevent double-counting.
4. **Harvest wrenches** -- extract forces from contact data and convert to
   spatial wrenches on the driving solver's bodies.
"""

from __future__ import annotations

import warp as wp

# ------------------------------------------------------------------
# Helper: angular velocity from two quaternions
# ------------------------------------------------------------------


@wp.func
def _quat_velocity(q_now: wp.quat, q_prev: wp.quat, dt: float) -> wp.vec3:
    """Angular velocity (world frame) from successive quaternions."""
    q1 = wp.normalize(q_now)
    q0 = wp.normalize(q_prev)
    if wp.dot(q1, q0) < 0.0:
        q0 = wp.quat(-q0[0], -q0[1], -q0[2], -q0[3])
    dq = wp.normalize(wp.mul(q1, wp.quat_inverse(q0)))
    axis, angle = wp.quat_to_axis_angle(dq)
    return axis * (angle / dt)


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
# 2. Smooth teleportation
# ------------------------------------------------------------------


@wp.kernel(enable_backward=False)
def smooth_proxy_teleportation_kernel(
    dt: float,
    proxy_body_ids_local: wp.array[int],
    dst_body_q: wp.array[wp.transform],
    dst_body_qd: wp.array[wp.spatial_vector],
    dst_body_q_prev: wp.array[wp.transform],
):
    """Encode a proxy teleportation as a velocity correction and undo the position jump.

    After :func:`sync_proxy_states_kernel` teleports proxy ``body_q`` to the
    driving solver's begin-of-step pose, this kernel computes the residual
    between the teleported pose and the destination solver's previous
    end-of-step pose (``body_q_prev``), folds it into ``body_qd`` as a smooth
    velocity correction, and resets ``body_q`` back to ``body_q_prev``.

    This avoids a position discontinuity that would contaminate
    finite-difference velocity estimates used for contact damping and friction
    in penetration-free solvers (e.g. VBD).

    Args:
        dt: Substep time step [s].
        proxy_body_ids_local: Compact list of proxy-local body ids.
        dst_body_q: Destination body transforms (read then overwritten).
        dst_body_qd: Destination body velocities (accumulated).
        dst_body_q_prev: Destination solver's previous end-of-step body transforms.
    """
    i = wp.tid()
    if i >= proxy_body_ids_local.shape[0]:
        return

    b = proxy_body_ids_local[i]
    q_teleported = dst_body_q[b]
    q_prev = dst_body_q_prev[b]

    # Translational correction
    p_teleported = wp.transform_get_translation(q_teleported)
    p_prev = wp.transform_get_translation(q_prev)
    dv = (p_teleported - p_prev) / dt

    # Rotational correction
    r_teleported = wp.transform_get_rotation(q_teleported)
    r_prev = wp.transform_get_rotation(q_prev)
    dw = _quat_velocity(r_teleported, r_prev, dt)

    # Add correction to the synced velocity
    qd = dst_body_qd[b]
    dst_body_qd[b] = qd + wp.spatial_vector(dv, dw)

    # Reset position to previous end-of-step (no discontinuity)
    dst_body_q[b] = q_prev


# ------------------------------------------------------------------
# 3. Rewind proxy velocities
# ------------------------------------------------------------------


@wp.kernel(enable_backward=False)
def subtract_proxy_forces_kernel(
    dt: float,
    gravity: wp.array[wp.vec3],
    body_world: wp.array[wp.int32],
    dst_body_q: wp.array[wp.transform],
    dst_body_f: wp.array[wp.spatial_vector],
    coupling_forces: wp.array[wp.spatial_vector],
    body_local_to_proxy_global: wp.array[int],
    dst_body_inv_mass: wp.array[float],
    dst_body_inv_inertia: wp.array[wp.mat33],
    dst_body_qd: wp.array[wp.spatial_vector],
):
    """Subtract already-integrated forces and gravity from proxy velocities.

    Coupling forces were already applied to the driving solver; undoing them
    here prevents double-counting. Destination external force inputs and
    gravity are also subtracted because the synced proxy velocity came from a
    driving solver that already accounted for those contributions.

    Args:
        dt: Substep time step [s].
        gravity: Per-world gravity vectors.
        body_world: Per-body world index in the destination model.
        dst_body_q: Destination body transforms (for rotating inertia).
        dst_body_f: Destination body force inputs.
        coupling_forces: Spatial forces previously applied to the driving solver,
            indexed by global proxy body id.
        body_local_to_proxy_global: Dense map from local body id to global
            proxy body id. ``-1`` entries are skipped.
        dst_body_inv_mass: Destination inverse masses.
        dst_body_inv_inertia: Destination inverse inertia tensors.
        dst_body_qd: Destination body velocities (modified in-place).
    """
    local_id = wp.tid()
    global_id = body_local_to_proxy_global[local_id]
    if global_id < 0:
        return

    f = coupling_forces[global_id] + dst_body_f[local_id]

    inv_m = dst_body_inv_mass[local_id]
    r = wp.transform_get_rotation(dst_body_q[local_id])
    inv_I = dst_body_inv_inertia[local_id]

    delta_v = dt * inv_m * wp.spatial_top(f)
    delta_w = dt * wp.quat_rotate(r, inv_I * wp.quat_rotate_inv(r, wp.spatial_bottom(f)))

    # Subtract gravity (driving solver already applied it)
    world_idx = body_world[local_id]
    g = gravity[wp.max(world_idx, 0)]
    delta_v_grav = wp.vec3(0.0, 0.0, 0.0)
    if inv_m > 0.0:
        delta_v_grav = dt * g

    dst_body_qd[local_id] = dst_body_qd[local_id] - wp.spatial_vector(delta_v + delta_v_grav, delta_w)


@wp.kernel(enable_backward=False)
def subtract_proxy_particle_forces_kernel(
    dt: float,
    gravity: wp.array[wp.vec3],
    particle_world: wp.array[wp.int32],
    dst_particle_f: wp.array[wp.vec3],
    coupling_forces: wp.array[wp.vec3],
    particle_local_to_proxy_global: wp.array[int],
    dst_particle_inv_mass: wp.array[float],
    dst_particle_qd: wp.array[wp.vec3],
):
    """Subtract already-integrated particle forces and gravity from proxy velocities."""
    local_id = wp.tid()
    global_id = particle_local_to_proxy_global[local_id]
    if global_id < 0:
        return

    inv_m = dst_particle_inv_mass[local_id]
    delta_v = dt * inv_m * (coupling_forces[global_id] + dst_particle_f[local_id])

    world_idx = particle_world[local_id]
    g = gravity[wp.max(world_idx, 0)]
    delta_v_grav = wp.vec3(0.0, 0.0, 0.0)
    if inv_m > 0.0:
        delta_v_grav = dt * g

    dst_particle_qd[local_id] = dst_particle_qd[local_id] - (delta_v + delta_v_grav)


# ------------------------------------------------------------------
# 4. Harvest proxy wrenches from contact forces
# ------------------------------------------------------------------


@wp.kernel(enable_backward=False)
def harvest_proxy_wrenches_kernel(
    rigid_contact_count: wp.array[int],
    contact_body0: wp.array[wp.int32],
    contact_body1: wp.array[wp.int32],
    contact_point0_world: wp.array[wp.vec3],
    contact_point1_world: wp.array[wp.vec3],
    contact_force_on_body1: wp.array[wp.vec3],
    dst_body_inv_mass: wp.array[float],
    dst_body_flags: wp.array[wp.int32],
    body_local_to_proxy_global: wp.array[int],
    proxy_flag: int,
    body_com: wp.array[wp.vec3],
    body_q: wp.array[wp.transform],
    out_proxy_body_f: wp.array[wp.spatial_vector],
):
    """Extract contact forces on proxy bodies and convert to spatial wrenches.

    Only contacts where exactly one body is a proxy and the other is dynamic
    are included.  Proxy-proxy and proxy-static contacts are excluded.

    The output wrench is accumulated in ``out_proxy_body_f``, indexed by the
    global proxy body id. The caller is responsible for mapping those
    proxy-indexed values back onto source ids if needed.

    Args:
        rigid_contact_count: Scalar array holding the number of active contacts.
        contact_body0: Body index of first contact body (destination solver).
        contact_body1: Body index of second contact body (destination solver).
        contact_point0_world: World-space contact point on body 0.
        contact_point1_world: World-space contact point on body 1.
        contact_force_on_body1: Force vector applied to body 1 [N].
        dst_body_inv_mass: Inverse masses in the destination solver.
        dst_body_flags: Body flags in the destination solver view.
        body_local_to_proxy_global: Dense map from local body id to global
            proxy body id. ``-1`` entries are skipped.
        proxy_flag: Integer value of :attr:`~newton.BodyFlags.PROXY`.
        body_com: Center-of-mass offsets in the destination solver.
        body_q: Body transforms in the destination solver.
        out_proxy_body_f: Output spatial wrenches on proxy bodies.
    """
    contact_id = wp.tid()
    if contact_id >= rigid_contact_count[0]:
        return

    body0 = contact_body0[contact_id]
    body1 = contact_body1[contact_id]
    if body0 < 0 or body1 < 0:
        return

    is_proxy0 = int(0)
    is_proxy1 = int(0)
    proxy_global0 = int(-1)
    proxy_global1 = int(-1)
    if body0 < dst_body_flags.shape[0] and (dst_body_flags[body0] & proxy_flag) != 0:
        proxy_global0 = body_local_to_proxy_global[body0]
        if proxy_global0 >= 0:
            is_proxy0 = 1
    if body1 < dst_body_flags.shape[0] and (dst_body_flags[body1] & proxy_flag) != 0:
        proxy_global1 = body_local_to_proxy_global[body1]
        if proxy_global1 >= 0:
            is_proxy1 = 1

    # Exactly one body must be a proxy
    if (is_proxy0 + is_proxy1) != 1:
        return

    # Non-proxy body must be dynamic
    other_id = body1 if is_proxy0 == 1 else body0
    if other_id < 0 or other_id >= dst_body_inv_mass.shape[0]:
        return
    if dst_body_inv_mass[other_id] <= 0.0:
        return

    # Determine force direction and proxy body.
    force_on_b1 = contact_force_on_body1[contact_id]
    if is_proxy1 == 1:
        proxy_local_id = body1
        proxy_global_id = proxy_global1
        contact_point = contact_point1_world[contact_id]
        force_on_proxy = force_on_b1
    else:
        proxy_local_id = body0
        proxy_global_id = proxy_global0
        contact_point = contact_point0_world[contact_id]
        force_on_proxy = -force_on_b1

    if proxy_global_id < 0 or proxy_global_id >= out_proxy_body_f.shape[0]:
        return

    # Wrench = force + torque about proxy body COM.
    com_world = wp.transform_point(body_q[proxy_local_id], body_com[proxy_local_id])
    torque = wp.cross(contact_point - com_world, force_on_proxy)
    wp.atomic_add(out_proxy_body_f, proxy_global_id, wp.spatial_vector(force_on_proxy, torque))
