# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Warp kernels supporting the ADMM coupling scheme.

See :class:`SolverCoupled` and ``docs/plans/2026-04-23-admm-coupling.tex``
for the overall algorithm. Kernels here cover proximal velocity shifts,
point-velocity computation, ``J^T W`` force splatting, and the per-constraint
``u`` / ``lambda`` updates for model-joint attachments and detected contacts.
"""

from __future__ import annotations

import warp as wp

from ...geometry import ParticleFlags
from ...math.spatial import velocity_at_point
from ...sim.contacts import contact_surface_point

_PARTICLE_FLAG_ACTIVE = wp.constant(int(ParticleFlags.ACTIVE))
_INACTIVE_U_MIN = wp.constant(-1.0e8)
_WEIGHT_RESCALE_EPS = wp.constant(1.0e-12)
_POINT_ANGULAR_LUMP_SCALE = wp.constant(2.0 / 3.0)


@wp.func
def _interface_weight(m_a: float, m_b: float) -> float:
    if m_a > 0.0 and m_b > 0.0:
        return wp.sqrt((m_a * m_b) / (m_a + m_b))
    if m_a > 0.0:
        return wp.sqrt(m_a)
    if m_b > 0.0:
        return wp.sqrt(m_b)
    return 1.0


@wp.kernel(enable_backward=False)
def compute_interface_weights_kernel(
    indices_a: wp.array[int],
    masses_a: wp.array[float],
    indices_b: wp.array[int],
    masses_b: wp.array[float],
    weights: wp.array[float],
):
    """Compute interface weights for indexed endpoint mass pairs."""
    i = wp.tid()
    weights[i] = _interface_weight(masses_a[indices_a[i]], masses_b[indices_b[i]])


@wp.func
def _rescale_lambda(lambda_value: wp.vec3, old_weight: float, new_weight: float) -> wp.vec3:
    if old_weight > _WEIGHT_RESCALE_EPS and new_weight > _WEIGHT_RESCALE_EPS:
        return lambda_value * (old_weight / new_weight)
    return wp.vec3(0.0, 0.0, 0.0)


@wp.func
def _contact_u_min_from_gap(gap: float, baumgarte: float, dt: float) -> float:
    if dt <= 0.0:
        return _INACTIVE_U_MIN
    if gap < 0.0:
        return -baumgarte * gap / dt
    return -gap / dt


@wp.func
def _point_angular_lump(arm: wp.vec3) -> float:
    # Isotropic lump of [r]x^T [r]x: trace(|r|^2 I - r r^T) / 3.
    return _POINT_ANGULAR_LUMP_SCALE * wp.dot(arm, arm)


@wp.kernel(enable_backward=False)
def scatter_effective_mass_kernel(
    global_id: wp.array[int],
    local_mass: wp.array[float],
    mass_out: wp.array[float],
):
    """Scatter local effective masses into a global effective-mass buffer."""
    i = wp.tid()
    mass_out[global_id[i]] = local_mass[i]


@wp.kernel(enable_backward=False)
def scatter_body_effective_mass_block_kernel(
    global_body: wp.array[int],
    local_mass: wp.array[float],
    local_inertia: wp.array[wp.mat33],
    mass_out: wp.array[float],
    inertia_scalar_out: wp.array[float],
):
    """Scatter local effective mass blocks into global scalar buffers."""
    i = wp.tid()
    global_id = global_body[i]
    mass_out[global_id] = local_mass[i]
    inertia_scalar_out[global_id] = wp.max(wp.trace(local_inertia[i]) / 3.0, 0.0)


@wp.kernel(enable_backward=False)
def velocity_proximal_shift_body_kernel(
    v_n: wp.array[wp.spatial_vector],
    v_k: wp.array[wp.spatial_vector],
    gamma: float,
    v_out: wp.array[wp.spatial_vector],
):
    """Write ``(v_n + gamma * v_k) / (1 + gamma)`` into ``v_out`` for each body.

    Paired with a ``(1 + gamma)`` rescaling of the body mass on the sub-solver's
    :class:`ModelView`, this produces the ADMM proximal term
    ``(gamma/2) ||v - v_k||^2_M`` in the sub-solver's per-step optimization.
    """
    i = wp.tid()
    inv_denom = 1.0 / (1.0 + gamma)
    v_out[i] = (v_n[i] + gamma * v_k[i]) * inv_denom


@wp.kernel(enable_backward=False)
def velocity_proximal_shift_body_masked_kernel(
    v_n: wp.array[wp.spatial_vector],
    v_k: wp.array[wp.spatial_vector],
    proximal_mask: wp.array[int],
    gamma: float,
    v_out: wp.array[wp.spatial_vector],
):
    """Masked body proximal shift; unmasked bodies keep the step input velocity."""
    i = wp.tid()
    if proximal_mask[i] == 0:
        v_out[i] = v_n[i]
        return
    inv_denom = 1.0 / (1.0 + gamma)
    v_out[i] = (v_n[i] + gamma * v_k[i]) * inv_denom


@wp.kernel(enable_backward=False)
def velocity_proximal_shift_particle_kernel(
    v_n: wp.array[wp.vec3],
    v_k: wp.array[wp.vec3],
    gamma: float,
    v_out: wp.array[wp.vec3],
):
    """Particle analogue of :func:`velocity_proximal_shift_body_kernel`."""
    i = wp.tid()
    inv_denom = 1.0 / (1.0 + gamma)
    v_out[i] = (v_n[i] + gamma * v_k[i]) * inv_denom


@wp.kernel(enable_backward=False)
def velocity_proximal_shift_particle_masked_kernel(
    v_n: wp.array[wp.vec3],
    v_k: wp.array[wp.vec3],
    proximal_mask: wp.array[int],
    gamma: float,
    v_out: wp.array[wp.vec3],
):
    """Masked particle proximal shift; unmasked particles keep the step input velocity."""
    i = wp.tid()
    if proximal_mask[i] == 0:
        v_out[i] = v_n[i]
        return
    inv_denom = 1.0 / (1.0 + gamma)
    v_out[i] = (v_n[i] + gamma * v_k[i]) * inv_denom


@wp.kernel(enable_backward=False)
def velocity_proximal_shift_joint_kernel(
    v_n: wp.array[float],
    v_k: wp.array[float],
    gamma: float,
    v_out: wp.array[float],
):
    """Joint-space analogue of :func:`velocity_proximal_shift_body_kernel`.

    Operates on the flat ``joint_qd`` array so the shift covers generalized
    DOFs for solvers whose authoritative velocity state is joint-space
    (e.g. :class:`~newton.solvers.SolverMuJoCo`).
    """
    i = wp.tid()
    inv_denom = 1.0 / (1.0 + gamma)
    v_out[i] = (v_n[i] + gamma * v_k[i]) * inv_denom


@wp.kernel(enable_backward=False)
def velocity_proximal_shift_joint_masked_kernel(
    v_n: wp.array[float],
    v_k: wp.array[float],
    proximal_mask: wp.array[int],
    gamma: float,
    v_out: wp.array[float],
):
    """Masked joint-space proximal shift."""
    i = wp.tid()
    if proximal_mask[i] == 0:
        v_out[i] = v_n[i]
        return
    inv_denom = 1.0 / (1.0 + gamma)
    v_out[i] = (v_n[i] + gamma * v_k[i]) * inv_denom


@wp.kernel(enable_backward=False)
def velocity_proximal_shift_body_lumped_kernel(
    v_n: wp.array[wp.spatial_vector],
    v_k: wp.array[wp.spatial_vector],
    proximal_mass: wp.array[float],
    proximal_inertia: wp.array[float],
    body_mass: wp.array[float],
    body_inertia: wp.array[wp.mat33],
    v_out: wp.array[wp.spatial_vector],
):
    """Shift body velocities using the lumped ADMM proximal metric."""
    i = wp.tid()

    v_n_i = v_n[i]
    v_k_i = v_k[i]
    linear = wp.spatial_top(v_n_i)
    angular = wp.spatial_bottom(v_n_i)

    mass_lump = proximal_mass[i]
    if mass_lump > 0.0:
        base_mass = body_mass[i] - mass_lump
        if base_mass > 0.0:
            inv_denom = 1.0 / (base_mass + mass_lump)
            linear = (base_mass * wp.spatial_top(v_n_i) + mass_lump * wp.spatial_top(v_k_i)) * inv_denom

    inertia_lump = proximal_inertia[i]
    if inertia_lump > 0.0:
        base_inertia = wp.max(wp.trace(body_inertia[i]) / 3.0 - inertia_lump, 0.0)
        if base_inertia > 0.0:
            inv_denom = 1.0 / (base_inertia + inertia_lump)
            angular = (base_inertia * wp.spatial_bottom(v_n_i) + inertia_lump * wp.spatial_bottom(v_k_i)) * inv_denom

    v_out[i] = wp.spatial_vector(linear, angular)


@wp.kernel(enable_backward=False)
def velocity_proximal_shift_particle_lumped_kernel(
    v_n: wp.array[wp.vec3],
    v_k: wp.array[wp.vec3],
    proximal_mass: wp.array[float],
    particle_mass: wp.array[float],
    v_out: wp.array[wp.vec3],
):
    """Shift particle velocities using the lumped ADMM proximal metric."""
    i = wp.tid()
    mass_lump = proximal_mass[i]
    if mass_lump <= 0.0:
        v_out[i] = v_n[i]
        return

    base_mass = particle_mass[i] - mass_lump
    if base_mass <= 0.0:
        v_out[i] = v_n[i]
        return

    inv_denom = 1.0 / (base_mass + mass_lump)
    v_out[i] = (base_mass * v_n[i] + mass_lump * v_k[i]) * inv_denom


@wp.kernel(enable_backward=False)
def velocity_proximal_shift_joint_lumped_kernel(
    v_n: wp.array[float],
    v_k: wp.array[float],
    proximal_factor: wp.array[float],
    v_out: wp.array[float],
):
    """Shift generalized velocities using a dimensionless lumped factor."""
    i = wp.tid()
    factor = proximal_factor[i]
    if factor <= 0.0:
        v_out[i] = v_n[i]
        return
    inv_denom = 1.0 / (1.0 + factor)
    v_out[i] = (v_n[i] + factor * v_k[i]) * inv_denom


@wp.kernel(enable_backward=False)
def body_gravity_compensation_kernel(
    gamma: float,
    body_mass: wp.array[float],
    body_inv_mass: wp.array[float],
    body_gravity_acceleration: wp.array[wp.vec3],
    body_f: wp.array[wp.spatial_vector],
):
    """Cancel gravity force introduced by ADMM proximal mass scaling.

    ``body_gravity_acceleration`` is evaluated by the sub-solver's coupling
    hook and therefore already includes solver-specific gravity scaling.
    """
    i = wp.tid()
    if body_inv_mass[i] <= 0.0:
        return

    mass = body_mass[i] / (1.0 + gamma)
    g = body_gravity_acceleration[i]
    wp.atomic_add(body_f, i, wp.spatial_vector(-gamma * mass * g, wp.vec3(0.0, 0.0, 0.0)))


@wp.kernel(enable_backward=False)
def body_gravity_compensation_masked_kernel(
    gamma: float,
    proximal_mask: wp.array[int],
    body_mass: wp.array[float],
    body_inv_mass: wp.array[float],
    body_gravity_acceleration: wp.array[wp.vec3],
    body_f: wp.array[wp.spatial_vector],
):
    """Cancel gravity introduced by masked ADMM proximal mass scaling."""
    i = wp.tid()
    if proximal_mask[i] == 0 or body_inv_mass[i] <= 0.0:
        return

    mass = body_mass[i] / (1.0 + gamma)
    g = body_gravity_acceleration[i]
    wp.atomic_add(body_f, i, wp.spatial_vector(-gamma * mass * g, wp.vec3(0.0, 0.0, 0.0)))


@wp.kernel(enable_backward=False)
def particle_gravity_compensation_kernel(
    gamma: float,
    particle_mass: wp.array[float],
    particle_inv_mass: wp.array[float],
    particle_flags: wp.array[wp.int32],
    particle_gravity_acceleration: wp.array[wp.vec3],
    particle_f: wp.array[wp.vec3],
):
    """Cancel gravity force introduced by ADMM proximal mass scaling."""
    i = wp.tid()
    if particle_inv_mass[i] <= 0.0 or (particle_flags[i] & _PARTICLE_FLAG_ACTIVE) == 0:
        return

    mass = particle_mass[i] / (1.0 + gamma)
    g = particle_gravity_acceleration[i]
    wp.atomic_add(particle_f, i, -gamma * mass * g)


@wp.kernel(enable_backward=False)
def particle_gravity_compensation_masked_kernel(
    gamma: float,
    proximal_mask: wp.array[int],
    particle_mass: wp.array[float],
    particle_inv_mass: wp.array[float],
    particle_flags: wp.array[wp.int32],
    particle_gravity_acceleration: wp.array[wp.vec3],
    particle_f: wp.array[wp.vec3],
):
    """Cancel gravity introduced by masked ADMM proximal mass scaling."""
    i = wp.tid()
    if proximal_mask[i] == 0 or particle_inv_mass[i] <= 0.0 or (particle_flags[i] & _PARTICLE_FLAG_ACTIVE) == 0:
        return

    mass = particle_mass[i] / (1.0 + gamma)
    g = particle_gravity_acceleration[i]
    wp.atomic_add(particle_f, i, -gamma * mass * g)


@wp.kernel(enable_backward=False)
def body_gravity_compensation_lumped_kernel(
    proximal_mass: wp.array[float],
    body_inv_mass: wp.array[float],
    body_gravity_acceleration: wp.array[wp.vec3],
    body_f: wp.array[wp.spatial_vector],
):
    """Cancel gravity introduced by lumped proximal mass increments."""
    i = wp.tid()
    mass_lump = proximal_mass[i]
    if mass_lump <= 0.0 or body_inv_mass[i] <= 0.0:
        return

    g = body_gravity_acceleration[i]
    wp.atomic_add(body_f, i, wp.spatial_vector(-mass_lump * g, wp.vec3(0.0, 0.0, 0.0)))


@wp.kernel(enable_backward=False)
def particle_gravity_compensation_lumped_kernel(
    proximal_mass: wp.array[float],
    particle_inv_mass: wp.array[float],
    particle_flags: wp.array[wp.int32],
    particle_gravity_acceleration: wp.array[wp.vec3],
    particle_f: wp.array[wp.vec3],
):
    """Cancel gravity introduced by lumped proximal mass increments."""
    i = wp.tid()
    mass_lump = proximal_mass[i]
    if mass_lump <= 0.0 or particle_inv_mass[i] <= 0.0 or (particle_flags[i] & _PARTICLE_FLAG_ACTIVE) == 0:
        return

    g = particle_gravity_acceleration[i]
    wp.atomic_add(particle_f, i, -mass_lump * g)


@wp.kernel(enable_backward=False)
def mark_indices_mask_kernel(indices: wp.array[int], mask: wp.array[int]):
    """Mark ``mask[indices[i]]`` for each valid index."""
    i = wp.tid()
    index = indices[i]
    if 0 <= index and index < mask.shape[0]:
        mask[index] = 1


@wp.kernel(enable_backward=False)
def mark_active_indices_mask_kernel(active_count: wp.array[int], indices: wp.array[int], mask: wp.array[int]):
    """Mark active compact contact indices in a proximal mask."""
    i = wp.tid()
    if i >= active_count[0]:
        return
    index = indices[i]
    if 0 <= index and index < mask.shape[0]:
        mask[index] = 1


@wp.kernel(enable_backward=False)
def mark_global_indices_mask_kernel(
    indices: wp.array[int],
    global_to_local: wp.array[int],
    local_mask: wp.array[int],
):
    """Mark local mask entries mapped from global ids."""
    i = wp.tid()
    global_id = indices[i]
    if 0 <= global_id and global_id < global_to_local.shape[0]:
        local_id = global_to_local[global_id]
        if 0 <= local_id and local_id < local_mask.shape[0]:
            local_mask[local_id] = 1


@wp.kernel(enable_backward=False)
def mark_active_global_indices_mask_kernel(
    active_count: wp.array[int],
    indices: wp.array[int],
    global_to_local: wp.array[int],
    local_mask: wp.array[int],
):
    """Mark active global ids after mapping them to local mask entries."""
    i = wp.tid()
    if i >= active_count[0]:
        return
    global_id = indices[i]
    if 0 <= global_id and global_id < global_to_local.shape[0]:
        local_id = global_to_local[global_id]
        if 0 <= local_id and local_id < local_mask.shape[0]:
            local_mask[local_id] = 1


@wp.kernel(enable_backward=False)
def mark_active_pair_indices_mask_kernel(
    active_count: wp.array[int],
    indices_a: wp.array[int],
    mask_a: wp.array[int],
    indices_b: wp.array[int],
    mask_b: wp.array[int],
):
    """Mark both endpoints of active compact contact rows."""
    i = wp.tid()
    if i >= active_count[0]:
        return
    index_a = indices_a[i]
    index_b = indices_b[i]
    if 0 <= index_a and index_a < mask_a.shape[0]:
        mask_a[index_a] = 1
    if 0 <= index_b and index_b < mask_b.shape[0]:
        mask_b[index_b] = 1


@wp.kernel(enable_backward=False)
def mark_local_indices_from_global_mask_kernel(
    local_to_global: wp.array[int],
    global_mask: wp.array[int],
    local_mask: wp.array[int],
):
    """Mark local entries whose global id is enabled in ``global_mask``."""
    local = wp.tid()
    global_id = local_to_global[local]
    if 0 <= global_id and global_id < global_mask.shape[0] and global_mask[global_id] != 0:
        local_mask[local] = 1


@wp.kernel(enable_backward=False)
def accumulate_body_point_proximal_lump_kernel(
    body_id: wp.array[int],
    point_local: wp.array[wp.vec3],
    body_q: wp.array[wp.transform],
    body_com: wp.array[wp.vec3],
    W: wp.array[float],
    gamma_rho: float,
    mass_lump: wp.array[float],
    inertia_lump: wp.array[float],
    mask: wp.array[int],
):
    """Accumulate point-velocity ``gamma rho W^2 J^T J`` lumps for bodies."""
    i = wp.tid()
    body = body_id[i]
    if body < 0 or body >= mass_lump.shape[0]:
        return

    weight = gamma_rho * W[i] * W[i]
    xform = body_q[body]
    point = wp.transform_point(xform, point_local[i])
    arm = point - wp.transform_point(xform, body_com[body])
    wp.atomic_add(mass_lump, body, weight)
    wp.atomic_add(inertia_lump, body, weight * _point_angular_lump(arm))
    mask[body] = 1


@wp.kernel(enable_backward=False)
def accumulate_active_body_point_proximal_lump_kernel(
    active_count: wp.array[int],
    body_id: wp.array[int],
    point_local: wp.array[wp.vec3],
    body_q: wp.array[wp.transform],
    body_com: wp.array[wp.vec3],
    W: wp.array[float],
    gamma_rho: float,
    mass_lump: wp.array[float],
    inertia_lump: wp.array[float],
    mask: wp.array[int],
):
    """Accumulate active point-velocity proximal lumps for bodies."""
    i = wp.tid()
    if i >= active_count[0]:
        return
    body = body_id[i]
    if body < 0 or body >= mass_lump.shape[0]:
        return

    weight = gamma_rho * W[i] * W[i]
    xform = body_q[body]
    point = wp.transform_point(xform, point_local[i])
    arm = point - wp.transform_point(xform, body_com[body])
    wp.atomic_add(mass_lump, body, weight)
    wp.atomic_add(inertia_lump, body, weight * _point_angular_lump(arm))
    mask[body] = 1


@wp.kernel(enable_backward=False)
def accumulate_active_body_contact_proximal_lump_kernel(
    active_count: wp.array[int],
    body_id: wp.array[int],
    point_local: wp.array[wp.vec3],
    point_offset_local: wp.array[wp.vec3],
    body_q: wp.array[wp.transform],
    body_com: wp.array[wp.vec3],
    W: wp.array[float],
    gamma_rho: float,
    mass_lump: wp.array[float],
    inertia_lump: wp.array[float],
    mask: wp.array[int],
):
    """Accumulate active contact point-velocity proximal lumps for bodies."""
    i = wp.tid()
    if i >= active_count[0]:
        return
    body = body_id[i]
    if body < 0 or body >= mass_lump.shape[0]:
        return

    weight = gamma_rho * W[i] * W[i]
    xform = body_q[body]
    point = contact_surface_point(xform, point_local[i], point_offset_local[i])
    arm = point - wp.transform_point(xform, body_com[body])
    wp.atomic_add(mass_lump, body, weight)
    wp.atomic_add(inertia_lump, body, weight * _point_angular_lump(arm))
    mask[body] = 1


@wp.kernel(enable_backward=False)
def accumulate_body_angular_proximal_lump_kernel(
    body_id: wp.array[int],
    W: wp.array[float],
    gamma_rho: float,
    component_lump: float,
    inertia_lump: wp.array[float],
    mask: wp.array[int],
):
    """Accumulate angular ``gamma rho W^2 J^T J`` lumps for bodies."""
    i = wp.tid()
    body = body_id[i]
    if body < 0 or body >= inertia_lump.shape[0]:
        return

    weight = gamma_rho * W[i] * W[i] * component_lump
    wp.atomic_add(inertia_lump, body, weight)
    mask[body] = 1


@wp.kernel(enable_backward=False)
def accumulate_indices_proximal_lump_kernel(
    indices: wp.array[int],
    W: wp.array[float],
    gamma_rho: float,
    lump: wp.array[float],
    mask: wp.array[int],
):
    """Accumulate scalar proximal lumps for local indices."""
    i = wp.tid()
    index = indices[i]
    if 0 <= index and index < lump.shape[0]:
        wp.atomic_add(lump, index, gamma_rho * W[i] * W[i])
        mask[index] = 1


@wp.kernel(enable_backward=False)
def accumulate_active_indices_proximal_lump_kernel(
    active_count: wp.array[int],
    indices: wp.array[int],
    W: wp.array[float],
    gamma_rho: float,
    lump: wp.array[float],
    mask: wp.array[int],
):
    """Accumulate scalar proximal lumps for active local indices."""
    i = wp.tid()
    if i >= active_count[0]:
        return
    index = indices[i]
    if 0 <= index and index < lump.shape[0]:
        wp.atomic_add(lump, index, gamma_rho * W[i] * W[i])
        mask[index] = 1


@wp.kernel(enable_backward=False)
def accumulate_global_indices_proximal_lump_kernel(
    indices: wp.array[int],
    global_to_local: wp.array[int],
    W: wp.array[float],
    gamma_rho: float,
    local_lump: wp.array[float],
    local_mask: wp.array[int],
):
    """Accumulate scalar proximal lumps for global ids mapped to local ids."""
    i = wp.tid()
    global_id = indices[i]
    if 0 <= global_id and global_id < global_to_local.shape[0]:
        local_id = global_to_local[global_id]
        if 0 <= local_id and local_id < local_lump.shape[0]:
            wp.atomic_add(local_lump, local_id, gamma_rho * W[i] * W[i])
            local_mask[local_id] = 1


@wp.kernel(enable_backward=False)
def accumulate_active_global_indices_proximal_lump_kernel(
    active_count: wp.array[int],
    indices: wp.array[int],
    global_to_local: wp.array[int],
    W: wp.array[float],
    gamma_rho: float,
    local_lump: wp.array[float],
    local_mask: wp.array[int],
):
    """Accumulate active scalar proximal lumps for global ids."""
    i = wp.tid()
    if i >= active_count[0]:
        return
    global_id = indices[i]
    if 0 <= global_id and global_id < global_to_local.shape[0]:
        local_id = global_to_local[global_id]
        if 0 <= local_id and local_id < local_lump.shape[0]:
            wp.atomic_add(local_lump, local_id, gamma_rho * W[i] * W[i])
            local_mask[local_id] = 1


@wp.kernel(enable_backward=False)
def accumulate_joint_qd_factor_from_body_proximal_lump_kernel(
    body_mass_lump: wp.array[float],
    body_inertia_lump: wp.array[float],
    body_local_to_global: wp.array[int],
    body_effective_mass: wp.array[float],
    body_effective_inertia_scalar: wp.array[float],
    body_joint_qd_start: wp.array[int],
    body_joint_qd_count: wp.array[int],
    body_joint_qd_indices: wp.array[int],
    joint_qd_factor: wp.array[float],
    joint_qd_mask: wp.array[int],
):
    """Propagate body proximal lumps to generalized-velocity DOFs."""
    body = wp.tid()
    global_body = body_local_to_global[body]
    factor = float(0.0)

    mass = body_effective_mass[global_body]
    if mass > 0.0 and body_mass_lump[body] > 0.0:
        factor = factor + body_mass_lump[body] / mass

    inertia = body_effective_inertia_scalar[global_body]
    if inertia > 0.0 and body_inertia_lump[body] > 0.0:
        factor = factor + body_inertia_lump[body] / inertia

    if factor <= 0.0:
        return

    start = body_joint_qd_start[body]
    count = body_joint_qd_count[body]
    for offset in range(count):
        dof = body_joint_qd_indices[start + offset]
        if 0 <= dof and dof < joint_qd_factor.shape[0]:
            wp.atomic_add(joint_qd_factor, dof, factor)
            joint_qd_mask[dof] = 1


# ----------------------------------------------------------------------
# Quadratic attachment local solve
# ----------------------------------------------------------------------
#
# The ADMM update rules for a quadratic coupling energy
# ``E_c(u) = (kappa/2) ||u - u_target||^2 + (damping/2) ||u||^2`` are:
#
#     u^{k+1}      = (rho W^2 Jv + kappa u_target - W lambda) / (kappa + damping + rho W^2)
#     lambda^{k+1} = lambda^k + rho W (u^{k+1} - Jv)
#
# With ``u_target = 0`` the coupling damps the relative velocity to zero;
# a non-zero ``u_target`` acts as a Baumgarte-style position stabiliser.


@wp.kernel(enable_backward=False)
def u_update_quadratic_kernel(
    kappa: wp.array[float],
    damping: wp.array[float],
    W: wp.array[float],
    rho: float,
    lambda_k: wp.array[wp.vec3],
    Jv: wp.array[wp.vec3],
    u_target: wp.array[wp.vec3],
    u_out: wp.array[wp.vec3],
):
    """Closed-form u-update for a quadratic coupling energy."""
    i = wp.tid()
    W_i = W[i]
    W2 = W_i * W_i
    denom = kappa[i] + damping[i] + rho * W2
    u_out[i] = (rho * W2 * Jv[i] + kappa[i] * u_target[i] - W_i * lambda_k[i]) / denom


@wp.kernel(enable_backward=False)
def lambda_update_kernel(
    rho: float,
    W: wp.array[float],
    u: wp.array[wp.vec3],
    Jv: wp.array[wp.vec3],
    lambda_inout: wp.array[wp.vec3],
):
    """Dual-variable update ``lambda += rho * W * (u - Jv)``."""
    i = wp.tid()
    lambda_inout[i] = lambda_inout[i] + rho * W[i] * (u[i] - Jv[i])


@wp.func
def _soft_threshold_box(value: float, threshold: float) -> float:
    threshold = wp.max(0.0, threshold)
    if value > threshold:
        return value - threshold
    if value < -threshold:
        return value + threshold
    return 0.0


@wp.kernel(enable_backward=False)
def joint_box_friction_u_update_kernel(
    friction: wp.array[wp.vec3],
    W: wp.array[float],
    rho: float,
    lambda_k: wp.array[wp.vec3],
    Jv: wp.array[wp.vec3],
    u_out: wp.array[wp.vec3],
):
    """Local ADMM u-update for per-axis box dry friction.

    ``friction`` stores positive physical force/torque limits. The proximal
    solve is the component-wise soft-threshold of the relative velocity in the
    joint frame, giving a maximum-dissipation box-friction law.
    """
    i = wp.tid()
    W_i = W[i]
    p = Jv[i]
    denom = rho * W_i
    if denom > 0.0:
        p = p - lambda_k[i] / denom

    threshold = wp.vec3(0.0, 0.0, 0.0)
    force_denom = rho * W_i * W_i
    if force_denom > 0.0:
        threshold = friction[i] / force_denom

    u_out[i] = wp.vec3(
        _soft_threshold_box(p[0], threshold[0]),
        _soft_threshold_box(p[1], threshold[1]),
        _soft_threshold_box(p[2], threshold[2]),
    )


@wp.func
def solve_coulomb_isotropic(mu: float, normal: wp.vec3, u: wp.vec3):
    """Solve the isotropic local Coulomb law in velocity space.

    This is the local maximum-dissipation solve used by Daviet's contact
    projection: separating contacts keep their relative velocity, sticking
    contacts return zero velocity, and sliding contacts keep zero normal
    velocity while reducing the tangential velocity so the force lies on the
    Coulomb boundary.
    """
    u_n = wp.dot(u, normal)
    if u_n < 0.0:
        u = u - u_n * normal
        tau = wp.length_sq(u)
        alpha = mu * u_n
        if tau <= alpha * alpha:
            u = wp.vec3(0.0, 0.0, 0.0)
        else:
            u = u * (1.0 + mu * u_n / wp.sqrt(tau))

    return u


@wp.kernel(enable_backward=False)
def contact_u_update_kernel(
    active_count: wp.array[int],
    u_min: wp.array[float],
    W: wp.array[float],
    rho: float,
    friction: wp.array[float],
    normal: wp.array[wp.vec3],
    lambda_k: wp.array[wp.vec3],
    Jv: wp.array[wp.vec3],
    u_out: wp.array[wp.vec3],
):
    """Local contact solve for compact active contact rows."""
    i = wp.tid()
    if i >= active_count[0]:
        return

    W_i = W[i]
    p = Jv[i]
    denom = rho * W_i
    if denom > 0.0:
        p = p - lambda_k[i] / denom

    u_min_i = u_min[i]
    if u_min_i <= _INACTIVE_U_MIN:
        u_out[i] = p
        return

    n = normal[i]
    mu = wp.max(0.0, friction[i])
    shifted = p - u_min_i * n
    u_out[i] = solve_coulomb_isotropic(mu, n, shifted) + u_min_i * n


@wp.kernel(enable_backward=False)
def contact_lambda_update_kernel(
    active_count: wp.array[int],
    rho: float,
    W: wp.array[float],
    u: wp.array[wp.vec3],
    Jv: wp.array[wp.vec3],
    lambda_inout: wp.array[wp.vec3],
):
    """Dual update for compact active contact rows."""
    i = wp.tid()
    if i >= active_count[0]:
        return
    lambda_inout[i] = lambda_inout[i] + rho * W[i] * (u[i] - Jv[i])


# ----------------------------------------------------------------------
# Rigid-particle attachment kernels
# ----------------------------------------------------------------------
#
# Custom model annotations can bind a rigid-body anchor to a deformable
# particle. The sign convention is
#
#     Jv = v_body_at_anchor - v_particle
#
# so side A receives ``+f`` and the particle receives ``-f``.


@wp.kernel(enable_backward=False)
def attach_rp_compute_Jv_kernel(
    body_a: wp.array[int],
    point_a_local: wp.array[wp.vec3],
    particle_b: wp.array[int],
    body_q: wp.array[wp.transform],
    body_com: wp.array[wp.vec3],
    body_qd: wp.array[wp.spatial_vector],
    particle_qd: wp.array[wp.vec3],
    Jv: wp.array[wp.vec3],
):
    """Compute ``Jv = v_body_at_anchor - v_particle`` per attachment."""
    i = wp.tid()
    ba = body_a[i]
    pb = particle_b[i]
    xform_a = body_q[ba]
    world_pt_a = wp.transform_point(xform_a, point_a_local[i])
    arm_a = world_pt_a - wp.transform_point(xform_a, body_com[ba])
    Jv[i] = velocity_at_point(body_qd[ba], arm_a) - particle_qd[pb]


@wp.kernel(enable_backward=False)
def attach_rp_compute_u_target_kernel(
    body_a: wp.array[int],
    point_a_local: wp.array[wp.vec3],
    particle_b: wp.array[int],
    body_q: wp.array[wp.transform],
    particle_q: wp.array[wp.vec3],
    baumgarte: float,
    dt: float,
    u_target: wp.array[wp.vec3],
):
    """Compute Baumgarte target velocity for a rigid-particle attachment."""
    i = wp.tid()
    ba = body_a[i]
    pb = particle_b[i]
    anchor = wp.transform_point(body_q[ba], point_a_local[i])
    gap = particle_q[pb] - anchor
    u_target[i] = (baumgarte / dt) * gap


@wp.kernel(enable_backward=False)
def attach_rp_accumulate_forces_kernel(
    body_a: wp.array[int],
    point_a_local: wp.array[wp.vec3],
    particle_b: wp.array[int],
    body_q: wp.array[wp.transform],
    body_com: wp.array[wp.vec3],
    rho: float,
    W: wp.array[float],
    lambda_k: wp.array[wp.vec3],
    u_k: wp.array[wp.vec3],
    Jv_k: wp.array[wp.vec3],
    body_f: wp.array[wp.spatial_vector],
    particle_f: wp.array[wp.vec3],
):
    """Splat rigid-particle attachment forces into body and particle buffers."""
    i = wp.tid()
    ba = body_a[i]
    pb = particle_b[i]
    W_i = W[i]
    force = W_i * (lambda_k[i] + rho * W_i * (u_k[i] - Jv_k[i]))

    xform_a = body_q[ba]
    world_pt_a = wp.transform_point(xform_a, point_a_local[i])
    arm_a = world_pt_a - wp.transform_point(xform_a, body_com[ba])
    wp.atomic_add(body_f, ba, wp.spatial_vector(force, wp.cross(arm_a, force)))
    wp.atomic_sub(particle_f, pb, force)


# ----------------------------------------------------------------------
# Rigid-rigid attachment kernels
# ----------------------------------------------------------------------
#
# Body-body model joints are converted to quadratic ADMM attachments. The
# translational row uses anchor point velocities:
#
#     Jv = v_a_at_anchor - v_b_at_anchor
#
# The fixed-joint angular row uses world angular velocities:
#
#     Jw = w_a - w_b


@wp.kernel(enable_backward=False)
def attach_rr_compute_u_target_kernel(
    body_a: wp.array[int],
    point_a_local: wp.array[wp.vec3],
    body_b: wp.array[int],
    point_b_local: wp.array[wp.vec3],
    body_q_a: wp.array[wp.transform],
    body_q_b: wp.array[wp.transform],
    baumgarte: float,
    dt: float,
    u_target: wp.array[wp.vec3],
):
    """Compute Baumgarte target velocity for a rigid-rigid anchor attachment."""
    i = wp.tid()
    ba = body_a[i]
    bb = body_b[i]
    point_a = wp.transform_point(body_q_a[ba], point_a_local[i])
    point_b = wp.transform_point(body_q_b[bb], point_b_local[i])
    gap = point_b - point_a
    u_target[i] = (baumgarte / dt) * gap


@wp.kernel(enable_backward=False)
def attach_rr_angular_compute_Jv_kernel(
    body_a: wp.array[int],
    body_b: wp.array[int],
    body_qd_a: wp.array[wp.spatial_vector],
    body_qd_b: wp.array[wp.spatial_vector],
    Jv: wp.array[wp.vec3],
):
    """Compute ``Jv = w_a - w_b`` per angular rigid-rigid attachment."""
    i = wp.tid()
    Jv[i] = wp.spatial_bottom(body_qd_a[body_a[i]]) - wp.spatial_bottom(body_qd_b[body_b[i]])


@wp.kernel(enable_backward=False)
def attach_rr_angular_local_compute_Jv_kernel(
    body_a: wp.array[int],
    frame_a: wp.array[wp.transform],
    body_b: wp.array[int],
    body_q_a: wp.array[wp.transform],
    body_qd_a: wp.array[wp.spatial_vector],
    body_qd_b: wp.array[wp.spatial_vector],
    Jv: wp.array[wp.vec3],
):
    """Compute angular relative velocity in body A's attachment frame."""
    i = wp.tid()
    ba = body_a[i]
    rel_w_world = wp.spatial_bottom(body_qd_a[ba]) - wp.spatial_bottom(body_qd_b[body_b[i]])
    frame_world = body_q_a[ba] * frame_a[i]
    Jv[i] = wp.quat_rotate_inv(wp.transform_get_rotation(frame_world), rel_w_world)


@wp.kernel(enable_backward=False)
def attach_rr_revolute_angular_local_compute_Jv_kernel(
    body_a: wp.array[int],
    frame_a: wp.array[wp.transform],
    body_b: wp.array[int],
    body_q_a: wp.array[wp.transform],
    body_qd_a: wp.array[wp.spatial_vector],
    body_qd_b: wp.array[wp.spatial_vector],
    Jv: wp.array[wp.vec3],
):
    """Compute the two constrained angular velocity components of a revolute joint."""
    i = wp.tid()
    ba = body_a[i]
    rel_w_world = wp.spatial_bottom(body_qd_a[ba]) - wp.spatial_bottom(body_qd_b[body_b[i]])
    frame_world = body_q_a[ba] * frame_a[i]
    rel_w_local = wp.quat_rotate_inv(wp.transform_get_rotation(frame_world), rel_w_world)
    Jv[i] = wp.vec3(0.0, rel_w_local[1], rel_w_local[2])


@wp.kernel(enable_backward=False)
def attach_rr_angular_compute_u_target_kernel(
    body_a: wp.array[int],
    frame_a: wp.array[wp.transform],
    body_b: wp.array[int],
    frame_b: wp.array[wp.transform],
    body_q_a: wp.array[wp.transform],
    body_q_b: wp.array[wp.transform],
    baumgarte: float,
    dt: float,
    u_target: wp.array[wp.vec3],
):
    """Compute Baumgarte target angular velocity for a fixed-joint row."""
    i = wp.tid()
    rot_a = wp.transform_get_rotation(body_q_a[body_a[i]] * frame_a[i])
    rot_b = wp.transform_get_rotation(body_q_b[body_b[i]] * frame_b[i])
    dq = wp.normalize(wp.mul(rot_b, wp.quat_inverse(rot_a)))
    axis, angle = wp.quat_to_axis_angle(dq)
    u_target[i] = axis * (baumgarte * angle / dt)


@wp.kernel(enable_backward=False)
def attach_rr_revolute_angular_local_compute_u_target_kernel(
    body_a: wp.array[int],
    frame_a: wp.array[wp.transform],
    body_b: wp.array[int],
    frame_b: wp.array[wp.transform],
    body_q_a: wp.array[wp.transform],
    body_q_b: wp.array[wp.transform],
    baumgarte: float,
    dt: float,
    u_target: wp.array[wp.vec3],
):
    """Compute Baumgarte target for the two constrained revolute angular axes."""
    i = wp.tid()
    rot_a = wp.transform_get_rotation(body_q_a[body_a[i]] * frame_a[i])
    rot_b = wp.transform_get_rotation(body_q_b[body_b[i]] * frame_b[i])
    dq = wp.normalize(wp.mul(rot_b, wp.quat_inverse(rot_a)))
    axis, angle = wp.quat_to_axis_angle(dq)
    target_world = axis * (baumgarte * angle / dt)
    target_local = wp.quat_rotate_inv(rot_a, target_world)
    u_target[i] = wp.vec3(0.0, target_local[1], target_local[2])


@wp.kernel(enable_backward=False)
def attach_rr_angular_accumulate_forces_kernel(
    body_a: wp.array[int],
    body_b: wp.array[int],
    rho: float,
    W: wp.array[float],
    lambda_k: wp.array[wp.vec3],
    u_k: wp.array[wp.vec3],
    Jv_k: wp.array[wp.vec3],
    body_f_a: wp.array[wp.spatial_vector],
    body_f_b: wp.array[wp.spatial_vector],
):
    """Splat angular attachment torques into both rigid-body force buffers."""
    i = wp.tid()
    W_i = W[i]
    torque_a = W_i * (lambda_k[i] + rho * W_i * (u_k[i] - Jv_k[i]))
    wp.atomic_add(body_f_a, body_a[i], wp.spatial_vector(wp.vec3(0.0, 0.0, 0.0), torque_a))
    wp.atomic_sub(body_f_b, body_b[i], wp.spatial_vector(wp.vec3(0.0, 0.0, 0.0), torque_a))


@wp.kernel(enable_backward=False)
def attach_rr_angular_local_accumulate_forces_kernel(
    body_a: wp.array[int],
    frame_a: wp.array[wp.transform],
    body_b: wp.array[int],
    body_q_a: wp.array[wp.transform],
    rho: float,
    W: wp.array[float],
    lambda_k: wp.array[wp.vec3],
    u_k: wp.array[wp.vec3],
    Jv_k: wp.array[wp.vec3],
    body_f_a: wp.array[wp.spatial_vector],
    body_f_b: wp.array[wp.spatial_vector],
):
    """Splat local angular attachment torques in world coordinates."""
    i = wp.tid()
    ba = body_a[i]
    W_i = W[i]
    torque_local = W_i * (lambda_k[i] + rho * W_i * (u_k[i] - Jv_k[i]))
    frame_world = body_q_a[ba] * frame_a[i]
    torque_world = wp.quat_rotate(wp.transform_get_rotation(frame_world), torque_local)
    wp.atomic_add(body_f_a, ba, wp.spatial_vector(wp.vec3(0.0, 0.0, 0.0), torque_world))
    wp.atomic_sub(body_f_b, body_b[i], wp.spatial_vector(wp.vec3(0.0, 0.0, 0.0), torque_world))


@wp.kernel(enable_backward=False)
def attach_rr_revolute_angular_local_accumulate_forces_kernel(
    body_a: wp.array[int],
    frame_a: wp.array[wp.transform],
    body_b: wp.array[int],
    body_q_a: wp.array[wp.transform],
    rho: float,
    W: wp.array[float],
    lambda_k: wp.array[wp.vec3],
    u_k: wp.array[wp.vec3],
    Jv_k: wp.array[wp.vec3],
    body_f_a: wp.array[wp.spatial_vector],
    body_f_b: wp.array[wp.spatial_vector],
):
    """Splat revolute angular constraint torques, leaving the hinge axis free."""
    i = wp.tid()
    ba = body_a[i]
    W_i = W[i]
    torque_local = W_i * (lambda_k[i] + rho * W_i * (u_k[i] - Jv_k[i]))
    torque_local = wp.vec3(0.0, torque_local[1], torque_local[2])
    frame_world = body_q_a[ba] * frame_a[i]
    torque_world = wp.quat_rotate(wp.transform_get_rotation(frame_world), torque_local)
    wp.atomic_add(body_f_a, ba, wp.spatial_vector(wp.vec3(0.0, 0.0, 0.0), torque_world))
    wp.atomic_sub(body_f_b, body_b[i], wp.spatial_vector(wp.vec3(0.0, 0.0, 0.0), torque_world))


# ----------------------------------------------------------------------
# Rigid-rigid contact kernels
# ----------------------------------------------------------------------
#
# Contact normals point from endpoint B toward endpoint A. A positive scalar
# contact force applies +normal to endpoint A and -normal to endpoint B.


@wp.kernel(enable_backward=False)
def attach_rr_compute_Jv_kernel(
    body_a: wp.array[int],
    point_a_local: wp.array[wp.vec3],
    body_b: wp.array[int],
    point_b_local: wp.array[wp.vec3],
    body_q_a: wp.array[wp.transform],
    body_com_a: wp.array[wp.vec3],
    body_qd_a: wp.array[wp.spatial_vector],
    body_q_b: wp.array[wp.transform],
    body_com_b: wp.array[wp.vec3],
    body_qd_b: wp.array[wp.spatial_vector],
    Jv: wp.array[wp.vec3],
):
    """Compute relative point velocity for a rigid-rigid point attachment."""
    i = wp.tid()
    ba = body_a[i]
    bb = body_b[i]

    xform_a = body_q_a[ba]
    point_a = wp.transform_point(xform_a, point_a_local[i])
    arm_a = point_a - wp.transform_point(xform_a, body_com_a[ba])
    vel_a = velocity_at_point(body_qd_a[ba], arm_a)

    xform_b = body_q_b[bb]
    point_b = wp.transform_point(xform_b, point_b_local[i])
    arm_b = point_b - wp.transform_point(xform_b, body_com_b[bb])
    vel_b = velocity_at_point(body_qd_b[bb], arm_b)

    Jv[i] = vel_a - vel_b


@wp.kernel(enable_backward=False)
def attach_rr_accumulate_forces_kernel(
    body_a: wp.array[int],
    point_a_local: wp.array[wp.vec3],
    body_b: wp.array[int],
    point_b_local: wp.array[wp.vec3],
    body_q_a: wp.array[wp.transform],
    body_com_a: wp.array[wp.vec3],
    body_q_b: wp.array[wp.transform],
    body_com_b: wp.array[wp.vec3],
    rho: float,
    W: wp.array[float],
    lambda_k: wp.array[wp.vec3],
    u_k: wp.array[wp.vec3],
    Jv_k: wp.array[wp.vec3],
    body_f_a: wp.array[wp.spatial_vector],
    body_f_b: wp.array[wp.spatial_vector],
):
    """Splat point-attachment forces for a rigid-rigid row."""
    i = wp.tid()
    ba = body_a[i]
    bb = body_b[i]
    W_i = W[i]
    force_a = W_i * (lambda_k[i] + rho * W_i * (u_k[i] - Jv_k[i]))

    xform_a = body_q_a[ba]
    point_a = wp.transform_point(xform_a, point_a_local[i])
    arm_a = point_a - wp.transform_point(xform_a, body_com_a[ba])
    wp.atomic_add(body_f_a, ba, wp.spatial_vector(force_a, wp.cross(arm_a, force_a)))

    force_b = -force_a
    xform_b = body_q_b[bb]
    point_b = wp.transform_point(xform_b, point_b_local[i])
    arm_b = point_b - wp.transform_point(xform_b, body_com_b[bb])
    wp.atomic_add(body_f_b, bb, wp.spatial_vector(force_b, wp.cross(arm_b, force_b)))


@wp.kernel(enable_backward=False)
def contact_rr_compute_Jv_kernel(
    active_count: wp.array[int],
    body_a: wp.array[int],
    point_a_local: wp.array[wp.vec3],
    point_a_offset_local: wp.array[wp.vec3],
    body_b: wp.array[int],
    point_b_local: wp.array[wp.vec3],
    point_b_offset_local: wp.array[wp.vec3],
    body_q_a: wp.array[wp.transform],
    body_com_a: wp.array[wp.vec3],
    body_qd_a: wp.array[wp.spatial_vector],
    body_q_b: wp.array[wp.transform],
    body_com_b: wp.array[wp.vec3],
    body_qd_b: wp.array[wp.spatial_vector],
    Jv: wp.array[wp.vec3],
):
    """Compute relative point velocity for compact active rigid contacts."""
    i = wp.tid()
    if i >= active_count[0]:
        return
    ba = body_a[i]
    bb = body_b[i]

    xform_a = body_q_a[ba]
    point_a = contact_surface_point(xform_a, point_a_local[i], point_a_offset_local[i])
    arm_a = point_a - wp.transform_point(xform_a, body_com_a[ba])
    vel_a = velocity_at_point(body_qd_a[ba], arm_a)

    xform_b = body_q_b[bb]
    point_b = contact_surface_point(xform_b, point_b_local[i], point_b_offset_local[i])
    arm_b = point_b - wp.transform_point(xform_b, body_com_b[bb])
    vel_b = velocity_at_point(body_qd_b[bb], arm_b)

    Jv[i] = vel_a - vel_b


@wp.kernel(enable_backward=False)
def contact_rr_compute_u_min_kernel(
    active_count: wp.array[int],
    body_a: wp.array[int],
    point_a_local: wp.array[wp.vec3],
    point_a_offset_local: wp.array[wp.vec3],
    body_b: wp.array[int],
    point_b_local: wp.array[wp.vec3],
    point_b_offset_local: wp.array[wp.vec3],
    normal: wp.array[wp.vec3],
    body_q_a: wp.array[wp.transform],
    body_q_b: wp.array[wp.transform],
    baumgarte: float,
    dt: float,
    u_min: wp.array[float],
):
    """Compute minimum normal velocity for compact active rigid contacts."""
    i = wp.tid()
    if i >= active_count[0]:
        return
    ba = body_a[i]
    bb = body_b[i]
    point_a = contact_surface_point(body_q_a[ba], point_a_local[i], point_a_offset_local[i])
    point_b = contact_surface_point(body_q_b[bb], point_b_local[i], point_b_offset_local[i])
    gap = wp.dot(normal[i], point_a - point_b)
    u_min[i] = _contact_u_min_from_gap(gap, baumgarte, dt)


@wp.kernel(enable_backward=False)
def contact_rr_accumulate_forces_kernel(
    active_count: wp.array[int],
    body_a: wp.array[int],
    point_a_local: wp.array[wp.vec3],
    point_a_offset_local: wp.array[wp.vec3],
    body_b: wp.array[int],
    point_b_local: wp.array[wp.vec3],
    point_b_offset_local: wp.array[wp.vec3],
    body_q_a: wp.array[wp.transform],
    body_com_a: wp.array[wp.vec3],
    body_q_b: wp.array[wp.transform],
    body_com_b: wp.array[wp.vec3],
    rho: float,
    W: wp.array[float],
    lambda_k: wp.array[wp.vec3],
    u_k: wp.array[wp.vec3],
    Jv_k: wp.array[wp.vec3],
    body_f_a: wp.array[wp.spatial_vector],
    body_f_b: wp.array[wp.spatial_vector],
):
    """Splat contact wrenches for compact active rigid contacts."""
    i = wp.tid()
    if i >= active_count[0]:
        return
    ba = body_a[i]
    bb = body_b[i]
    W_i = W[i]
    force_a = W_i * (lambda_k[i] + rho * W_i * (u_k[i] - Jv_k[i]))

    xform_a = body_q_a[ba]
    point_a = contact_surface_point(xform_a, point_a_local[i], point_a_offset_local[i])
    arm_a = point_a - wp.transform_point(xform_a, body_com_a[ba])
    wp.atomic_add(body_f_a, ba, wp.spatial_vector(force_a, wp.cross(arm_a, force_a)))

    force_b = -force_a
    xform_b = body_q_b[bb]
    point_b = contact_surface_point(xform_b, point_b_local[i], point_b_offset_local[i])
    arm_b = point_b - wp.transform_point(xform_b, body_com_b[bb])
    wp.atomic_add(body_f_b, bb, wp.spatial_vector(force_b, wp.cross(arm_b, force_b)))


@wp.kernel(enable_backward=False)
def contact_rr_clear_contact_snapshot_kernel(
    prev_contact_active: wp.array[int],
    prev_contact_lambda: wp.array[wp.vec3],
    prev_contact_W: wp.array[float],
):
    """Clear dynamic rigid-rigid ADMM dual state snapshots."""
    i = wp.tid()
    prev_contact_active[i] = 0
    prev_contact_lambda[i] = wp.vec3(0.0, 0.0, 0.0)
    prev_contact_W[i] = 0.0


@wp.kernel(enable_backward=False)
def contact_rr_snapshot_by_contact_kernel(
    active_count: wp.array[int],
    contact_id: wp.array[int],
    active: wp.array[int],
    W: wp.array[float],
    lambda_: wp.array[wp.vec3],
    prev_contact_active: wp.array[int],
    prev_contact_lambda: wp.array[wp.vec3],
    prev_contact_W: wp.array[float],
):
    """Snapshot dynamic rigid-rigid ADMM dual state by collision contact row."""
    i = wp.tid()
    if i >= active_count[0]:
        return

    cid = contact_id[i]
    if cid < 0 or cid >= prev_contact_active.shape[0] or active[i] == 0:
        return

    prev_contact_active[cid] = 1
    prev_contact_lambda[cid] = lambda_[i]
    prev_contact_W[cid] = W[i]


@wp.kernel(enable_backward=False)
def contact_rr_reset_kernel(
    active_count: wp.array[int],
    body_a: wp.array[int],
    point_a_local: wp.array[wp.vec3],
    point_a_offset_local: wp.array[wp.vec3],
    body_b: wp.array[int],
    point_b_local: wp.array[wp.vec3],
    point_b_offset_local: wp.array[wp.vec3],
    contact_id: wp.array[int],
    shape_a: wp.array[int],
    shape_b: wp.array[int],
    point_id: wp.array[int],
    active: wp.array[int],
    normal: wp.array[wp.vec3],
    W: wp.array[float],
    friction: wp.array[float],
    lambda_: wp.array[wp.vec3],
    Jv: wp.array[wp.vec3],
    u_min: wp.array[float],
):
    """Clear a fixed-capacity dynamic rigid-rigid contact group."""
    i = wp.tid()
    if i == 0:
        active_count[0] = 0

    body_a[i] = 0
    point_a_local[i] = wp.vec3(0.0, 0.0, 0.0)
    point_a_offset_local[i] = wp.vec3(0.0, 0.0, 0.0)
    body_b[i] = 0
    point_b_local[i] = wp.vec3(0.0, 0.0, 0.0)
    point_b_offset_local[i] = wp.vec3(0.0, 0.0, 0.0)
    contact_id[i] = -1
    shape_a[i] = -1
    shape_b[i] = -1
    point_id[i] = -1
    active[i] = 0
    normal[i] = wp.vec3(0.0, 0.0, 0.0)
    W[i] = 0.0
    friction[i] = 0.0
    lambda_[i] = wp.vec3(0.0, 0.0, 0.0)
    Jv[i] = wp.vec3(0.0, 0.0, 0.0)
    u_min[i] = _INACTIVE_U_MIN


@wp.kernel(enable_backward=False)
def contact_rr_fill_from_rigid_contacts_kernel(
    rigid_contact_count: wp.array[int],
    rigid_contact_shape0: wp.array[int],
    rigid_contact_shape1: wp.array[int],
    rigid_contact_point0: wp.array[wp.vec3],
    rigid_contact_point1: wp.array[wp.vec3],
    rigid_contact_offset0: wp.array[wp.vec3],
    rigid_contact_offset1: wp.array[wp.vec3],
    rigid_contact_normal: wp.array[wp.vec3],
    rigid_contact_point_id: wp.array[int],
    rigid_contact_match_index: wp.array[wp.int32],
    shape_body: wp.array[int],
    body_mask_a: wp.array[int],
    body_mask_b: wp.array[int],
    shape_mask_a: wp.array[int],
    shape_mask_b: wp.array[int],
    body_global_to_local_a: wp.array[int],
    body_global_to_local_b: wp.array[int],
    body_mass_a: wp.array[float],
    body_mass_b: wp.array[float],
    shape_material_mu: wp.array[float],
    use_contact_matching: int,
    contact_matching_force_scale: float,
    capacity: int,
    active_count: wp.array[int],
    active_count_max: wp.array[int],
    prev_contact_active: wp.array[int],
    prev_contact_lambda: wp.array[wp.vec3],
    prev_contact_W: wp.array[float],
    body_a: wp.array[int],
    point_a_local: wp.array[wp.vec3],
    point_a_offset_local: wp.array[wp.vec3],
    body_b: wp.array[int],
    point_b_local: wp.array[wp.vec3],
    point_b_offset_local: wp.array[wp.vec3],
    contact_id: wp.array[int],
    shape_a: wp.array[int],
    shape_b: wp.array[int],
    point_id: wp.array[int],
    active: wp.array[int],
    normal: wp.array[wp.vec3],
    W: wp.array[float],
    friction: wp.array[float],
    lambda_: wp.array[wp.vec3],
):
    """Convert detected rigid contacts into oriented ADMM rigid-rigid rows."""
    i = wp.tid()
    if i >= rigid_contact_count[0]:
        return

    s0 = rigid_contact_shape0[i]
    s1 = rigid_contact_shape1[i]
    if s0 < 0 or s1 < 0:
        return

    b0 = shape_body[s0]
    b1 = shape_body[s1]
    if b0 < 0 or b1 < 0:
        return

    ba = int(0)
    bb = int(0)
    sa = int(-1)
    sb = int(-1)
    pa = wp.vec3(0.0, 0.0, 0.0)
    pb = wp.vec3(0.0, 0.0, 0.0)
    oa = wp.vec3(0.0, 0.0, 0.0)
    ob = wp.vec3(0.0, 0.0, 0.0)
    n = wp.vec3(0.0, 0.0, 0.0)

    if body_mask_a[b1] != 0 and body_mask_b[b0] != 0 and shape_mask_a[s1] != 0 and shape_mask_b[s0] != 0:
        ba = b1
        bb = b0
        sa = s1
        sb = s0
        pa = rigid_contact_point1[i]
        pb = rigid_contact_point0[i]
        oa = rigid_contact_offset1[i]
        ob = rigid_contact_offset0[i]
        n = rigid_contact_normal[i]
    elif body_mask_a[b0] != 0 and body_mask_b[b1] != 0 and shape_mask_a[s0] != 0 and shape_mask_b[s1] != 0:
        ba = b0
        bb = b1
        sa = s0
        sb = s1
        pa = rigid_contact_point0[i]
        pb = rigid_contact_point1[i]
        oa = rigid_contact_offset0[i]
        ob = rigid_contact_offset1[i]
        n = -rigid_contact_normal[i]
    else:
        return

    ba_local = body_global_to_local_a[ba]
    bb_local = body_global_to_local_b[bb]
    if ba_local < 0 or bb_local < 0:
        return

    dst = wp.atomic_add(active_count, 0, 1)
    if dst >= capacity:
        wp.atomic_min(active_count, 0, capacity)
        wp.atomic_max(active_count_max, 0, capacity)
        return
    wp.atomic_max(active_count_max, 0, dst + 1)

    body_a[dst] = ba_local
    point_a_local[dst] = pa
    point_a_offset_local[dst] = oa
    body_b[dst] = bb_local
    point_b_local[dst] = pb
    point_b_offset_local[dst] = ob
    contact_id[dst] = i
    shape_a[dst] = sa
    shape_b[dst] = sb
    point_id[dst] = rigid_contact_point_id[i]
    active[dst] = 1
    normal[dst] = n

    ma = body_mass_a[ba]
    mb = body_mass_b[bb]
    weight = _interface_weight(ma, mb)
    W[dst] = weight
    # Geometric-mean combining of shape material friction (matches AVBD).
    friction[dst] = wp.sqrt(shape_material_mu[sa] * shape_material_mu[sb])

    lambda_out = wp.vec3(0.0, 0.0, 0.0)
    if use_contact_matching != 0:
        prev_id = rigid_contact_match_index[i]
        if prev_id >= 0 and prev_id < prev_contact_active.shape[0] and prev_contact_active[prev_id] != 0:
            lambda_out = contact_matching_force_scale * _rescale_lambda(
                prev_contact_lambda[prev_id], prev_contact_W[prev_id], weight
            )

    lambda_[dst] = lambda_out


# ----------------------------------------------------------------------
# Rigid-particle contact kernels
# ----------------------------------------------------------------------
#
# Contact normals point from endpoint B toward endpoint A. A positive scalar
# contact force applies +normal to endpoint A and -normal to endpoint B.


@wp.kernel(enable_backward=False)
def contact_rp_compute_Jv_kernel(
    active_count: wp.array[int],
    body_id: wp.array[int],
    point_body_local: wp.array[wp.vec3],
    particle_id: wp.array[int],
    body_sign: wp.array[int],
    body_q: wp.array[wp.transform],
    body_com: wp.array[wp.vec3],
    body_qd: wp.array[wp.spatial_vector],
    particle_qd: wp.array[wp.vec3],
    Jv: wp.array[wp.vec3],
):
    """Compute relative velocity for compact active rigid-particle contacts."""
    i = wp.tid()
    if i >= active_count[0]:
        return
    b = body_id[i]
    p = particle_id[i]
    xform = body_q[b]
    world_pt = wp.transform_point(xform, point_body_local[i])
    arm = world_pt - wp.transform_point(xform, body_com[b])
    body_v = velocity_at_point(body_qd[b], arm)
    particle_v = particle_qd[p]
    if body_sign[i] > 0:
        Jv[i] = body_v - particle_v
    else:
        Jv[i] = particle_v - body_v


@wp.kernel(enable_backward=False)
def contact_rp_compute_u_min_kernel(
    active_count: wp.array[int],
    body_id: wp.array[int],
    point_body_local: wp.array[wp.vec3],
    particle_id: wp.array[int],
    normal: wp.array[wp.vec3],
    body_sign: wp.array[int],
    body_q: wp.array[wp.transform],
    particle_q: wp.array[wp.vec3],
    particle_radius: wp.array[float],
    baumgarte: float,
    dt: float,
    u_min: wp.array[float],
):
    """Compute the minimum normal velocity for compact active rigid-particle contacts."""
    i = wp.tid()
    if i >= active_count[0]:
        return
    b = body_id[i]
    p = particle_id[i]
    body_pt = wp.transform_point(body_q[b], point_body_local[i])
    particle_pt = particle_q[p]
    n = normal[i]
    gap = float(0.0)
    if body_sign[i] > 0:
        gap = wp.dot(n, body_pt - particle_pt)
    else:
        gap = wp.dot(n, particle_pt - body_pt)

    surface_gap = gap - particle_radius[p]
    u_min[i] = _contact_u_min_from_gap(surface_gap, baumgarte, dt)


@wp.kernel(enable_backward=False)
def contact_rp_accumulate_forces_kernel(
    active_count: wp.array[int],
    body_id: wp.array[int],
    point_body_local: wp.array[wp.vec3],
    particle_id: wp.array[int],
    body_sign: wp.array[int],
    body_q: wp.array[wp.transform],
    body_com: wp.array[wp.vec3],
    rho: float,
    W: wp.array[float],
    lambda_k: wp.array[wp.vec3],
    u_k: wp.array[wp.vec3],
    Jv_k: wp.array[wp.vec3],
    body_f: wp.array[wp.spatial_vector],
    particle_f: wp.array[wp.vec3],
):
    """Splat Coulomb contact forces for compact active rigid-particle contacts."""
    i = wp.tid()
    if i >= active_count[0]:
        return
    b = body_id[i]
    p = particle_id[i]
    W_i = W[i]
    force = W_i * (lambda_k[i] + rho * W_i * (u_k[i] - Jv_k[i]))
    force_body = float(body_sign[i]) * force

    xform = body_q[b]
    world_pt = wp.transform_point(xform, point_body_local[i])
    arm = world_pt - wp.transform_point(xform, body_com[b])
    wp.atomic_add(body_f, b, wp.spatial_vector(force_body, wp.cross(arm, force_body)))
    wp.atomic_sub(particle_f, p, force_body)


@wp.kernel(enable_backward=False)
def contact_rp_snapshot_kernel(
    body_id: wp.array[int],
    particle_id: wp.array[int],
    shape_id: wp.array[int],
    active: wp.array[int],
    W: wp.array[float],
    lambda_: wp.array[wp.vec3],
    prev_body_id: wp.array[int],
    prev_particle_id: wp.array[int],
    prev_shape_id: wp.array[int],
    prev_active: wp.array[int],
    prev_W: wp.array[float],
    prev_lambda: wp.array[wp.vec3],
):
    """Snapshot dynamic rigid-particle contacts for key-based warm starting."""
    i = wp.tid()
    prev_body_id[i] = body_id[i]
    prev_particle_id[i] = particle_id[i]
    prev_shape_id[i] = shape_id[i]
    prev_active[i] = active[i]
    prev_W[i] = W[i]
    prev_lambda[i] = lambda_[i]


@wp.kernel(enable_backward=False)
def contact_rp_reset_kernel(
    active_count: wp.array[int],
    body_id: wp.array[int],
    point_body_local: wp.array[wp.vec3],
    particle_id: wp.array[int],
    shape_id: wp.array[int],
    active: wp.array[int],
    normal: wp.array[wp.vec3],
    body_sign: wp.array[int],
    W: wp.array[float],
    friction: wp.array[float],
    lambda_: wp.array[wp.vec3],
    Jv: wp.array[wp.vec3],
    u_min: wp.array[float],
):
    """Clear a fixed-capacity dynamic rigid-particle contact group."""
    i = wp.tid()
    if i == 0:
        active_count[0] = 0

    active[i] = 0
    body_id[i] = 0
    point_body_local[i] = wp.vec3(0.0, 0.0, 0.0)
    particle_id[i] = 0
    shape_id[i] = -1
    normal[i] = wp.vec3(0.0, 0.0, 0.0)
    body_sign[i] = -1
    W[i] = 0.0
    friction[i] = 0.0
    lambda_[i] = wp.vec3(0.0, 0.0, 0.0)
    Jv[i] = wp.vec3(0.0, 0.0, 0.0)
    u_min[i] = _INACTIVE_U_MIN


@wp.kernel(enable_backward=False)
def contact_rp_fill_from_soft_contacts_kernel(
    soft_contact_count: wp.array[int],
    soft_contact_particle: wp.array[int],
    soft_contact_shape: wp.array[int],
    soft_contact_body_pos: wp.array[wp.vec3],
    soft_contact_normal: wp.array[wp.vec3],
    shape_body: wp.array[int],
    particle_owner_mask: wp.array[int],
    body_owner_mask: wp.array[int],
    shape_filter_mask: wp.array[int],
    body_global_to_local: wp.array[int],
    body_mass: wp.array[float],
    particle_mass: wp.array[float],
    shape_material_mu: wp.array[float],
    particle_mu: float,
    capacity: int,
    active_count: wp.array[int],
    active_count_max: wp.array[int],
    prev_particle_id: wp.array[int],
    prev_shape_id: wp.array[int],
    prev_active: wp.array[int],
    prev_W: wp.array[float],
    prev_lambda: wp.array[wp.vec3],
    body_id: wp.array[int],
    point_body_local: wp.array[wp.vec3],
    particle_id: wp.array[int],
    shape_id: wp.array[int],
    active: wp.array[int],
    normal: wp.array[wp.vec3],
    body_sign: wp.array[int],
    W: wp.array[float],
    friction: wp.array[float],
    lambda_: wp.array[wp.vec3],
):
    """Populate a dynamic rigid-particle group from soft particle-shape contacts."""
    i = wp.tid()
    if i >= soft_contact_count[0]:
        return

    p = soft_contact_particle[i]
    s = soft_contact_shape[i]
    if p < 0 or s < 0:
        return
    if particle_owner_mask[p] == 0 or shape_filter_mask[s] == 0:
        return

    b = shape_body[s]
    if b < 0 or body_owner_mask[b] == 0:
        return

    b_local = body_global_to_local[b]
    if b_local < 0:
        return

    n = soft_contact_normal[i]
    n_len = wp.length(n)
    if n_len <= 0.0:
        return
    n = n / n_len

    dst = wp.atomic_add(active_count, 0, 1)
    if dst >= capacity:
        wp.atomic_min(active_count, 0, capacity)
        wp.atomic_max(active_count_max, 0, capacity)
        return
    wp.atomic_max(active_count_max, 0, dst + 1)

    m_a = body_mass[b]
    m_b = particle_mass[p]
    weight = _interface_weight(m_a, m_b)

    lambda0 = wp.vec3(0.0, 0.0, 0.0)
    for j in range(capacity):
        if prev_active[j] != 0 and prev_particle_id[j] == p and prev_shape_id[j] == s:
            lambda0 = _rescale_lambda(prev_lambda[j], prev_W[j], weight)
            break

    active[dst] = 1
    body_id[dst] = b_local
    point_body_local[dst] = soft_contact_body_pos[i]
    particle_id[dst] = p
    shape_id[dst] = s
    normal[dst] = n
    body_sign[dst] = -1
    W[dst] = weight
    # Geometric-mean combining of shape and particle friction.
    friction[dst] = wp.sqrt(shape_material_mu[s] * particle_mu)
    lambda_[dst] = lambda0


# ----------------------------------------------------------------------
# Particle-particle contact kernels
# ----------------------------------------------------------------------


@wp.kernel(enable_backward=False)
def contact_pp_compute_Jv_kernel(
    active_count: wp.array[int],
    particle_a: wp.array[int],
    particle_b: wp.array[int],
    particle_qd_a: wp.array[wp.vec3],
    particle_qd_b: wp.array[wp.vec3],
    Jv: wp.array[wp.vec3],
):
    """Compute relative velocity for compact active particle-particle contacts."""
    i = wp.tid()
    if i >= active_count[0]:
        return
    pa = particle_a[i]
    pb = particle_b[i]
    Jv[i] = particle_qd_a[pa] - particle_qd_b[pb]


@wp.kernel(enable_backward=False)
def contact_pp_compute_u_min_kernel(
    active_count: wp.array[int],
    particle_a: wp.array[int],
    particle_b: wp.array[int],
    normal: wp.array[wp.vec3],
    particle_q_a: wp.array[wp.vec3],
    particle_q_b: wp.array[wp.vec3],
    particle_radius: wp.array[float],
    baumgarte: float,
    dt: float,
    u_min: wp.array[float],
):
    """Compute the minimum normal velocity for compact active particle-particle contacts."""
    i = wp.tid()
    if i >= active_count[0]:
        return
    pa = particle_a[i]
    pb = particle_b[i]
    gap = wp.dot(normal[i], particle_q_a[pa] - particle_q_b[pb]) - (particle_radius[pa] + particle_radius[pb])
    u_min[i] = _contact_u_min_from_gap(gap, baumgarte, dt)


@wp.kernel(enable_backward=False)
def contact_pp_accumulate_forces_kernel(
    active_count: wp.array[int],
    particle_a: wp.array[int],
    particle_b: wp.array[int],
    rho: float,
    W: wp.array[float],
    lambda_k: wp.array[wp.vec3],
    u_k: wp.array[wp.vec3],
    Jv_k: wp.array[wp.vec3],
    particle_f_a: wp.array[wp.vec3],
    particle_f_b: wp.array[wp.vec3],
):
    """Splat Coulomb contact forces for compact active particle-particle contacts."""
    i = wp.tid()
    if i >= active_count[0]:
        return
    pa = particle_a[i]
    pb = particle_b[i]
    W_i = W[i]
    force = W_i * (lambda_k[i] + rho * W_i * (u_k[i] - Jv_k[i]))
    wp.atomic_add(particle_f_a, pa, force)
    wp.atomic_sub(particle_f_b, pb, force)


@wp.kernel(enable_backward=False)
def contact_pp_snapshot_kernel(
    particle_a: wp.array[int],
    particle_b: wp.array[int],
    active: wp.array[int],
    W: wp.array[float],
    lambda_: wp.array[wp.vec3],
    prev_particle_a: wp.array[int],
    prev_particle_b: wp.array[int],
    prev_active: wp.array[int],
    prev_W: wp.array[float],
    prev_lambda: wp.array[wp.vec3],
):
    """Snapshot dynamic particle-particle contacts for key-based warm starting."""
    i = wp.tid()
    prev_particle_a[i] = particle_a[i]
    prev_particle_b[i] = particle_b[i]
    prev_active[i] = active[i]
    prev_W[i] = W[i]
    prev_lambda[i] = lambda_[i]


@wp.kernel(enable_backward=False)
def contact_pp_reset_kernel(
    active_count: wp.array[int],
    particle_a: wp.array[int],
    particle_b: wp.array[int],
    active: wp.array[int],
    normal: wp.array[wp.vec3],
    W: wp.array[float],
    friction: wp.array[float],
    lambda_: wp.array[wp.vec3],
    Jv: wp.array[wp.vec3],
    u_min: wp.array[float],
):
    """Clear a fixed-capacity dynamic particle-particle contact group."""
    i = wp.tid()
    if i == 0:
        active_count[0] = 0

    particle_a[i] = 0
    particle_b[i] = 0
    active[i] = 0
    normal[i] = wp.vec3(0.0, 0.0, 0.0)
    W[i] = 0.0
    friction[i] = 0.0
    lambda_[i] = wp.vec3(0.0, 0.0, 0.0)
    Jv[i] = wp.vec3(0.0, 0.0, 0.0)
    u_min[i] = _INACTIVE_U_MIN


@wp.kernel(enable_backward=False)
def particle_contact_count_reset_kernel(particle_contact_count: wp.array[int]):
    """Reset a particle-particle contact stream count."""
    particle_contact_count[0] = 0


@wp.kernel(enable_backward=False)
def particle_particle_contacts_hashgrid_kernel(
    grid: wp.uint64,
    particle_q: wp.array[wp.vec3],
    particle_radius: wp.array[float],
    particle_flags: wp.array[wp.int32],
    particle_world: wp.array[int],
    particle_mask_a: wp.array[int],
    particle_mask_b: wp.array[int],
    query_radius: float,
    capacity: int,
    particle_contact_count: wp.array[int],
    particle_contact_count_max: wp.array[int],
    particle_contact_particle0: wp.array[int],
    particle_contact_particle1: wp.array[int],
    particle_contact_normal: wp.array[wp.vec3],
    particle_contact_tids: wp.array[int],
):
    """Detect particle-particle contacts into a contacts-like stream."""
    tid = wp.tid()
    pa = wp.hash_grid_point_id(grid, tid)
    if pa == -1:
        return
    if particle_mask_a[pa] == 0:
        return
    if (particle_flags[pa] & wp.int32(1)) == 0:
        return

    qa = particle_q[pa]
    world_a = particle_world[pa]
    query = wp.hash_grid_query(grid, qa, query_radius)
    pb = int(0)

    while wp.hash_grid_query_next(query, pb):
        if pb == pa:
            continue
        if particle_mask_b[pb] == 0:
            continue
        if (particle_flags[pb] & wp.int32(1)) == 0:
            continue

        world_b = particle_world[pb]
        if world_a != -1 and world_b != -1 and world_a != world_b:
            continue

        delta = qa - particle_q[pb]
        gap = wp.length(delta)
        if gap >= particle_radius[pa] + particle_radius[pb]:
            continue

        n = wp.vec3(1.0, 0.0, 0.0)
        if gap > 1.0e-8:
            n = delta / gap

        dst = wp.atomic_add(particle_contact_count, 0, 1)
        if dst >= capacity:
            wp.atomic_min(particle_contact_count, 0, capacity)
            wp.atomic_max(particle_contact_count_max, 0, capacity)
            continue
        wp.atomic_max(particle_contact_count_max, 0, dst + 1)

        particle_contact_particle0[dst] = pa
        particle_contact_particle1[dst] = pb
        particle_contact_normal[dst] = n
        particle_contact_tids[dst] = tid


@wp.kernel(enable_backward=False)
def contact_pp_fill_from_particle_contacts_kernel(
    particle_contact_count: wp.array[int],
    particle_contact_particle0: wp.array[int],
    particle_contact_particle1: wp.array[int],
    particle_contact_normal: wp.array[wp.vec3],
    particle_mass_a: wp.array[float],
    particle_mass_b: wp.array[float],
    particle_mu: float,
    capacity: int,
    active_count: wp.array[int],
    active_count_max: wp.array[int],
    prev_particle_a: wp.array[int],
    prev_particle_b: wp.array[int],
    prev_active: wp.array[int],
    prev_W: wp.array[float],
    prev_lambda: wp.array[wp.vec3],
    particle_a: wp.array[int],
    particle_b: wp.array[int],
    active: wp.array[int],
    normal: wp.array[wp.vec3],
    W: wp.array[float],
    friction: wp.array[float],
    lambda_: wp.array[wp.vec3],
):
    """Populate a dynamic particle-particle group from a contacts-like stream."""
    i = wp.tid()
    if i >= particle_contact_count[0]:
        return

    pa = particle_contact_particle0[i]
    pb = particle_contact_particle1[i]
    dst = wp.atomic_add(active_count, 0, 1)
    if dst >= capacity:
        wp.atomic_min(active_count, 0, capacity)
        wp.atomic_max(active_count_max, 0, capacity)
        return
    wp.atomic_max(active_count_max, 0, dst + 1)

    m_a = particle_mass_a[pa]
    m_b = particle_mass_b[pb]
    weight = _interface_weight(m_a, m_b)

    lambda0 = wp.vec3(0.0, 0.0, 0.0)
    for j in range(capacity):
        if prev_active[j] != 0 and prev_particle_a[j] == pa and prev_particle_b[j] == pb:
            lambda0 = _rescale_lambda(prev_lambda[j], prev_W[j], weight)
            break

    particle_a[dst] = pa
    particle_b[dst] = pb
    active[dst] = 1
    normal[dst] = particle_contact_normal[i]
    W[dst] = weight
    friction[dst] = particle_mu
    lambda_[dst] = lambda0
