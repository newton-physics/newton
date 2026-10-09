# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Per-iteration convergence diagnostics of the FeatherPGS position solve (``pgs_debug``)."""

import warp as wp

from .friction_patches import patch_normal_load
from .kernels import PGS_CONSTRAINT_TYPE_CONTACT, PGS_CONSTRAINT_TYPE_FRICTION


@wp.func
def world_response_velocity(
    v_out: wp.array[float], world_dof_indices: wp.array2d[int], world: int, local_dof: int
) -> float:
    """Read the velocity of a world-local DOF, zero for padding."""
    global_dof = world_dof_indices[world, local_dof]
    if global_dof >= 0:
        return v_out[global_dof]
    return 0.0


@wp.func
def _sibling_friction_row(parent_idx: int, i: int) -> int:
    """Return the other tangent row of the friction pair that contains row ``i``."""
    if i == parent_idx + 1:
        return parent_idx + 2
    return parent_idx + 1


@wp.kernel(enable_backward=False)
def pgs_convergence_diagnostic(
    constraint_count: wp.array[int],
    world_dof_indices: wp.array2d[int],
    rhs: wp.array2d[float],
    impulses: wp.array2d[float],
    prev_impulses: wp.array2d[float],
    row_type: wp.array2d[int],
    row_parent: wp.array2d[int],
    row_mu: wp.array2d[float],
    J_world: wp.array3d[float],
    max_world_dofs: int,
    mf_constraint_count: wp.array[int],
    mf_rhs: wp.array2d[float],
    mf_impulses: wp.array2d[float],
    prev_mf_impulses: wp.array2d[float],
    mf_row_type: wp.array2d[int],
    mf_row_parent: wp.array2d[int],
    mf_row_mu: wp.array2d[float],
    mf_J_a: wp.array3d[float],
    mf_J_b: wp.array3d[float],
    mf_dof_a: wp.array2d[int],
    mf_dof_b: wp.array2d[int],
    propagation_constraint_count: wp.array[int],
    propagation_rhs: wp.array2d[float],
    propagation_impulses: wp.array2d[float],
    prev_propagation_impulses: wp.array2d[float],
    propagation_row_type: wp.array2d[int],
    propagation_row_parent: wp.array2d[int],
    propagation_row_mu: wp.array2d[float],
    propagation_J_a: wp.array3d[float],
    propagation_J_b: wp.array3d[float],
    propagation_body_a: wp.array2d[int],
    propagation_body_b: wp.array2d[int],
    propagation_body_qd: wp.array2d[float],
    v_out: wp.array[float],
    metrics: wp.array2d[float],
):
    """Write one world's convergence metrics of the last sweep of the matrix-free solve.

    ``metrics[world]`` holds the largest impulse change of any row, the complementarity gap
    ``sum(lambda_n r_n)`` and the Fischer-Burmeister merit ``sum(FB(lambda_n, r_n)^2)`` of the
    contact rows, and the residual energy ``sum(r_t^2)`` of sticking friction rows, in the
    order ``[max_delta, complementarity, tangent_residual, fb_merit]``. ``r = J v + b`` is
    each row's velocity residual with the bias ``b`` of the right-hand side.
    """
    world = wp.tid()
    max_delta = float(0.0)
    complementarity = float(0.0)
    tangent_residual = float(0.0)
    fb_merit = float(0.0)

    for i in range(constraint_count[world]):
        lam = impulses[world, i]
        max_delta = wp.max(max_delta, wp.abs(lam - prev_impulses[world, i]))
        jv = float(0.0)
        for d in range(max_world_dofs):
            jv += J_world[world, i, d] * world_response_velocity(v_out, world_dof_indices, world, d)
        residual = jv + rhs[world, i]
        rt = row_type[world, i]
        if rt == PGS_CONSTRAINT_TYPE_CONTACT:
            complementarity += lam * residual
            fb = wp.sqrt(lam * lam + residual * residual) - lam - residual
            fb_merit += fb * fb
        elif rt == PGS_CONSTRAINT_TYPE_FRICTION:
            parent_idx = row_parent[world, i]
            radius = row_mu[world, i] * patch_normal_load(row_parent, impulses, world, parent_idx)
            if radius > 0.0:
                sib = _sibling_friction_row(parent_idx, i)
                if wp.length(wp.vec2(lam, impulses[world, sib])) < radius * 0.999:
                    tangent_residual += residual * residual

    for i in range(mf_constraint_count[world]):
        lam = mf_impulses[world, i]
        max_delta = wp.max(max_delta, wp.abs(lam - prev_mf_impulses[world, i]))
        jv = float(0.0)
        dof_a = mf_dof_a[world, i]
        dof_b = mf_dof_b[world, i]
        if dof_a >= 0:
            for k in range(6):
                jv += mf_J_a[world, i, k] * world_response_velocity(v_out, world_dof_indices, world, dof_a + k)
        if dof_b >= 0:
            for k in range(6):
                jv += mf_J_b[world, i, k] * world_response_velocity(v_out, world_dof_indices, world, dof_b + k)
        residual = jv + mf_rhs[world, i]
        rt = mf_row_type[world, i]
        if rt == PGS_CONSTRAINT_TYPE_CONTACT:
            complementarity += lam * residual
            fb = wp.sqrt(lam * lam + residual * residual) - lam - residual
            fb_merit += fb * fb
        elif rt == PGS_CONSTRAINT_TYPE_FRICTION:
            parent_idx = mf_row_parent[world, i]
            radius = mf_row_mu[world, i] * patch_normal_load(mf_row_parent, mf_impulses, world, parent_idx)
            if radius > 0.0:
                sib = _sibling_friction_row(parent_idx, i)
                if wp.length(wp.vec2(lam, mf_impulses[world, sib])) < radius * 0.999:
                    tangent_residual += residual * residual

    m_propagation = propagation_constraint_count[world]
    for i in range(m_propagation):
        lam = propagation_impulses[world, i]
        max_delta = wp.max(max_delta, wp.abs(lam - prev_propagation_impulses[world, i]))
        jv = float(0.0)
        ba = propagation_body_a[world, i]
        bb = propagation_body_b[world, i]
        if ba >= 0:
            for k in range(6):
                jv += propagation_J_a[world, i, k] * propagation_body_qd[ba, k]
        if bb >= 0:
            for k in range(6):
                jv += propagation_J_b[world, i, k] * propagation_body_qd[bb, k]
        residual = jv + propagation_rhs[world, i]
        rt = propagation_row_type[world, i]
        if rt == PGS_CONSTRAINT_TYPE_CONTACT:
            complementarity += lam * residual
            fb = wp.sqrt(lam * lam + residual * residual) - lam - residual
            fb_merit += fb * fb
        elif rt == PGS_CONSTRAINT_TYPE_FRICTION:
            parent_idx = propagation_row_parent[world, i]
            radius = propagation_row_mu[world, i] * patch_normal_load(
                propagation_row_parent, propagation_impulses, world, parent_idx
            )
            sib = _sibling_friction_row(parent_idx, i)
            if radius > 0.0 and sib < m_propagation:
                if wp.length(wp.vec2(lam, propagation_impulses[world, sib])) < radius * 0.999:
                    tangent_residual += residual * residual

    metrics[world, 0] = max_delta
    metrics[world, 1] = complementarity
    metrics[world, 2] = tangent_residual
    metrics[world, 3] = fb_merit


@wp.func
def contact_friction_residuals(
    normal_impulse: float,
    friction_load: float,
    mu: float,
    normal_velocity: float,
    tangent_velocity: wp.vec2,
    tangent_impulse: wp.vec2,
):
    """Return the cone violation, De Saxce complementarity, dual feasibility and MDP direction error.

    ``friction_load`` is the normal impulse that budgets the friction, the whole region's for a
    friction patch; normal complementarity uses the contact's own ``normal_impulse``.
    """
    speed = wp.length(tangent_velocity)
    radius = mu * friction_load
    cone = wp.max(wp.length(tangent_impulse) - radius, 0.0)
    complementarity = wp.abs(
        normal_impulse * normal_velocity + radius * speed + wp.dot(tangent_impulse, tangent_velocity)
    )
    dual = wp.max(-normal_velocity, 0.0)
    direction = float(0.0)
    if speed > 1.0e-8 and friction_load > 1.0e-8:
        direction = wp.length(tangent_impulse + radius * tangent_velocity / speed)
        if radius > 1.0e-8:
            direction /= radius
    return wp.vec4(cone, complementarity, dual, direction)


@wp.func
def _accumulate_contact_residuals(
    residuals: wp.spatial_vector,
    normal_impulse: float,
    friction_load: float,
    mu: float,
    normal_velocity: float,
    normal_bias: float,
    phi: float,
    tangent_velocity: wp.vec2,
    tangent_impulse: wp.vec2,
):
    """Fold one contact's residuals into the running per-world maxima."""
    biased = normal_velocity + normal_bias
    complementarity = wp.abs(biased)
    if normal_impulse < biased:
        complementarity = wp.abs(normal_impulse)
    friction = contact_friction_residuals(
        normal_impulse, friction_load, mu, normal_velocity, tangent_velocity, tangent_impulse
    )
    return wp.spatial_vector(
        wp.max(residuals[0], complementarity),
        wp.max(residuals[1], friction[0]),
        wp.max(residuals[2], -phi),
        wp.max(residuals[3], friction[1]),
        wp.max(residuals[4], friction[2]),
        wp.max(residuals[5], friction[3]),
    )


@wp.kernel(enable_backward=False)
def pgs_ncp_residuals_diagnostic(
    constraint_count: wp.array[int],
    world_dof_indices: wp.array2d[int],
    rhs: wp.array2d[float],
    impulses: wp.array2d[float],
    row_type: wp.array2d[int],
    row_parent: wp.array2d[int],
    row_mu: wp.array2d[float],
    row_phi: wp.array2d[float],
    J_world: wp.array3d[float],
    max_world_dofs: int,
    mf_constraint_count: wp.array[int],
    mf_rhs: wp.array2d[float],
    mf_impulses: wp.array2d[float],
    mf_row_type: wp.array2d[int],
    mf_row_parent: wp.array2d[int],
    mf_row_mu: wp.array2d[float],
    mf_row_phi: wp.array2d[float],
    mf_J_a: wp.array3d[float],
    mf_J_b: wp.array3d[float],
    mf_dof_a: wp.array2d[int],
    mf_dof_b: wp.array2d[int],
    propagation_constraint_count: wp.array[int],
    propagation_rhs: wp.array2d[float],
    propagation_impulses: wp.array2d[float],
    propagation_row_type: wp.array2d[int],
    propagation_row_parent: wp.array2d[int],
    propagation_row_mu: wp.array2d[float],
    propagation_row_phi: wp.array2d[float],
    propagation_J_a: wp.array3d[float],
    propagation_J_b: wp.array3d[float],
    propagation_body_a: wp.array2d[int],
    propagation_body_b: wp.array2d[int],
    propagation_body_qd: wp.array2d[float],
    v_out: wp.array[float],
    metrics: wp.array2d[float],
):
    """Write one world's contact NCP residuals, each the maximum over its contacts.

    ``metrics[world]`` holds ``[r_compl, r_cone, r_gap, r_ds_compl, r_ds_dual, r_mdp_dir]``:
    the normal complementarity ``|min(lambda_n, u_n + b_n)|``, the Coulomb cone violation
    ``max(|lambda_t| - mu lambda_n, 0)``, the penetration ``max(-phi, 0)``, the De Saxce
    complementarity ``|<lambda, c + (0, 0, mu |c_t|)>|``, the dual feasibility
    ``max(-u_n, 0)`` and, for sliding contacts, the maximum-dissipation direction error
    ``|lambda_t + mu lambda_n c_t / |c_t|| / (mu lambda_n)``. ``u`` and ``c`` are bias-free
    row velocities ``J v``. Each contact row is read with the friction pair that follows it;
    a friction patch budgets its cone and dissipation terms with the region's normal load.
    Other rows do not contribute.
    """
    world = wp.tid()
    residuals = wp.spatial_vector(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    m_dense = constraint_count[world]
    for i in range(m_dense):
        if row_type[world, i] != PGS_CONSTRAINT_TYPE_CONTACT:
            continue
        u = wp.vec3(0.0)
        lam_t = wp.vec2(0.0)
        mu = float(0.0)
        for r in range(3):
            row = i + r
            if r > 0:
                if row >= m_dense:
                    continue
                if row_type[world, row] != PGS_CONSTRAINT_TYPE_FRICTION or row_parent[world, row] != i:
                    continue
                lam_t[r - 1] = impulses[world, row]
                if mu == 0.0:
                    mu = row_mu[world, row]
            for d in range(max_world_dofs):
                u[r] += J_world[world, row, d] * world_response_velocity(v_out, world_dof_indices, world, d)
        lam_n = impulses[world, i]
        load = lam_n
        if mu > 0.0:
            load = patch_normal_load(row_parent, impulses, world, i)
        residuals = _accumulate_contact_residuals(
            residuals, lam_n, load, mu, u[0], rhs[world, i], row_phi[world, i], wp.vec2(u[1], u[2]), lam_t
        )

    m_mf = mf_constraint_count[world]
    for i in range(m_mf):
        if mf_row_type[world, i] != PGS_CONSTRAINT_TYPE_CONTACT:
            continue
        u = wp.vec3(0.0)
        lam_t = wp.vec2(0.0)
        mu = float(0.0)
        for r in range(3):
            row = i + r
            if r > 0:
                if row >= m_mf:
                    continue
                if mf_row_type[world, row] != PGS_CONSTRAINT_TYPE_FRICTION or mf_row_parent[world, row] != i:
                    continue
                lam_t[r - 1] = mf_impulses[world, row]
                if mu == 0.0:
                    mu = mf_row_mu[world, row]
            dof_a = mf_dof_a[world, row]
            dof_b = mf_dof_b[world, row]
            if dof_a >= 0:
                for k in range(6):
                    u[r] += mf_J_a[world, row, k] * world_response_velocity(v_out, world_dof_indices, world, dof_a + k)
            if dof_b >= 0:
                for k in range(6):
                    u[r] += mf_J_b[world, row, k] * world_response_velocity(v_out, world_dof_indices, world, dof_b + k)
        lam_n = mf_impulses[world, i]
        load = lam_n
        if mu > 0.0:
            load = patch_normal_load(mf_row_parent, mf_impulses, world, i)
        residuals = _accumulate_contact_residuals(
            residuals, lam_n, load, mu, u[0], mf_rhs[world, i], mf_row_phi[world, i], wp.vec2(u[1], u[2]), lam_t
        )

    m_propagation = propagation_constraint_count[world]
    for i in range(m_propagation):
        if propagation_row_type[world, i] != PGS_CONSTRAINT_TYPE_CONTACT:
            continue
        u = wp.vec3(0.0)
        lam_t = wp.vec2(0.0)
        mu = float(0.0)
        for r in range(3):
            row = i + r
            if r > 0:
                if row >= m_propagation:
                    continue
                if (
                    propagation_row_type[world, row] != PGS_CONSTRAINT_TYPE_FRICTION
                    or propagation_row_parent[world, row] != i
                ):
                    continue
                lam_t[r - 1] = propagation_impulses[world, row]
                if mu == 0.0:
                    mu = propagation_row_mu[world, row]
            ba = propagation_body_a[world, row]
            bb = propagation_body_b[world, row]
            if ba >= 0:
                for k in range(6):
                    u[r] += propagation_J_a[world, row, k] * propagation_body_qd[ba, k]
            if bb >= 0:
                for k in range(6):
                    u[r] += propagation_J_b[world, row, k] * propagation_body_qd[bb, k]
        lam_n = propagation_impulses[world, i]
        load = lam_n
        if mu > 0.0:
            load = patch_normal_load(propagation_row_parent, propagation_impulses, world, i)
        residuals = _accumulate_contact_residuals(
            residuals,
            lam_n,
            load,
            mu,
            u[0],
            propagation_rhs[world, i],
            propagation_row_phi[world, i],
            wp.vec2(u[1], u[2]),
            lam_t,
        )

    for k in range(6):
        metrics[world, k] = residuals[k]
