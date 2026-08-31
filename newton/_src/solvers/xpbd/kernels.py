# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import warp as wp

from ...geometry import ParticleFlags
from ...math import (
    vec_abs,
    vec_leaky_max,
    vec_leaky_min,
    vec_max,
    vec_min,
    velocity_at_point,
)
from ...sim import BodyFlags, JointType
from ...sim.contacts import contact_surface_point, contact_surface_separation


@wp.kernel
def copy_kinematic_body_state_kernel(
    body_flags: wp.array[wp.int32],
    body_q_in: wp.array[wp.transform],
    body_qd_in: wp.array[wp.spatial_vector],
    body_q_out: wp.array[wp.transform],
    body_qd_out: wp.array[wp.spatial_vector],
):
    """Copy prescribed maximal state through the solve for kinematic bodies."""
    tid = wp.tid()
    if (body_flags[tid] & int(BodyFlags.KINEMATIC)) == 0:
        return
    body_q_out[tid] = body_q_in[tid]
    body_qd_out[tid] = body_qd_in[tid]


@wp.kernel
def apply_particle_shape_restitution(
    particle_v_new: wp.array[wp.vec3],
    particle_x_old: wp.array[wp.vec3],
    particle_v_old: wp.array[wp.vec3],
    particle_radius: wp.array[float],
    particle_flags: wp.array[wp.int32],
    body_q: wp.array[wp.transform],
    body_q_pre_solve: wp.array[wp.transform],
    body_qd: wp.array[wp.spatial_vector],
    body_qd_pre_solve: wp.array[wp.spatial_vector],
    body_com: wp.array[wp.vec3],
    shape_body: wp.array[int],
    particle_ka: float,
    restitution: float,
    contact_count: wp.array[int],
    contact_particle: wp.array[int],
    contact_shape: wp.array[int],
    contact_body_pos: wp.array[wp.vec3],
    contact_body_vel: wp.array[wp.vec3],
    contact_normal: wp.array[wp.vec3],
    contact_max: int,
    particle_v_out: wp.array[wp.vec3],
):
    tid = wp.tid()

    count = min(contact_max, contact_count[0])
    if tid >= count:
        return

    shape_index = contact_shape[tid]
    body_index = shape_body[shape_index]
    particle_index = contact_particle[tid]

    if (particle_flags[particle_index] & ParticleFlags.ACTIVE) == 0:
        return

    v_new = particle_v_new[particle_index]
    px = particle_x_old[particle_index]
    v_old = particle_v_old[particle_index]

    X_wb = wp.transform_identity()
    X_wb_pre_solve = wp.transform_identity()
    X_com = wp.vec3()

    if body_index >= 0:
        X_wb = body_q[body_index]
        X_wb_pre_solve = body_q_pre_solve[body_index]
        X_com = body_com[body_index]

    # body position in world space
    bx = wp.transform_point(X_wb, contact_body_pos[tid])

    n = contact_normal[tid]
    c = wp.dot(n, px - bx) - particle_radius[particle_index]

    if c > particle_ka:
        return

    # Use the same pre-solve pose and velocity snapshot as rigid restitution.
    bx_pre_solve = wp.transform_point(X_wb_pre_solve, contact_body_pos[tid])
    r = bx_pre_solve - wp.transform_point(X_wb_pre_solve, X_com)

    # compute body velocity at the contact point
    bv_contact = wp.transform_vector(X_wb_pre_solve, contact_body_vel[tid])
    bv_old = bv_contact
    bv_new = bv_contact
    if body_index >= 0:
        bv_old = velocity_at_point(body_qd_pre_solve[body_index], r) + bv_contact
        bv_new = velocity_at_point(body_qd[body_index], r) + bv_contact

    rel_vel_old = wp.dot(n, v_old - bv_old)
    rel_vel_new = wp.dot(n, v_new - bv_new)

    if rel_vel_old < 0.0:
        dv = n * (-rel_vel_new + wp.max(-restitution * rel_vel_old, 0.0))

        wp.atomic_add(particle_v_out, particle_index, dv)


@wp.kernel
def solve_particle_shape_contacts(
    particle_x: wp.array[wp.vec3],
    particle_v: wp.array[wp.vec3],
    particle_invmass: wp.array[float],
    particle_radius: wp.array[float],
    particle_flags: wp.array[wp.int32],
    body_q: wp.array[wp.transform],
    body_qd: wp.array[wp.spatial_vector],
    body_com: wp.array[wp.vec3],
    body_m_inv: wp.array[float],
    body_I_inv: wp.array[wp.mat33],
    body_flags: wp.array[wp.int32],
    shape_body: wp.array[int],
    shape_material_mu: wp.array[float],
    particle_mu: float,
    particle_ka: float,
    contact_count: wp.array[int],
    contact_particle: wp.array[int],
    contact_shape: wp.array[int],
    contact_body_pos: wp.array[wp.vec3],
    contact_body_vel: wp.array[wp.vec3],
    contact_normal: wp.array[wp.vec3],
    contact_max: int,
    dt: float,
    relaxation: float,
    # outputs
    delta: wp.array[wp.vec3],
    body_delta: wp.array[wp.spatial_vector],
):
    tid = wp.tid()

    count = min(contact_max, contact_count[0])
    if tid >= count:
        return

    shape_index = contact_shape[tid]
    body_index = shape_body[shape_index]
    particle_index = contact_particle[tid]

    particle_flag = particle_flags[particle_index]
    if (particle_flag & ParticleFlags.ACTIVE) == 0:
        return
    if (particle_flag & ParticleFlags.PROXY) != 0:
        if body_index < 0:
            return
        if (body_flags[body_index] & int(BodyFlags.PROXY)) != 0:
            return
        if body_m_inv[body_index] == 0.0:
            return

    px = particle_x[particle_index]
    pv = particle_v[particle_index]

    X_wb = wp.transform_identity()
    X_com = wp.vec3()

    if body_index >= 0:
        X_wb = body_q[body_index]
        X_com = body_com[body_index]

    # body position in world space
    bx = wp.transform_point(X_wb, contact_body_pos[tid])
    r = bx - wp.transform_point(X_wb, X_com)

    n = contact_normal[tid]
    c = wp.dot(n, px - bx) - particle_radius[particle_index]

    if c > particle_ka:
        return

    # take average material properties of shape and particle parameters
    mu = 0.5 * (particle_mu + shape_material_mu[shape_index])

    # body velocity
    body_v_s = wp.spatial_vector()
    if body_index >= 0:
        body_v_s = body_qd[body_index]

    body_w = wp.spatial_bottom(body_v_s)
    body_v = wp.spatial_top(body_v_s)

    # compute the body velocity at the particle position
    bv = body_v + wp.cross(body_w, r) + wp.transform_vector(X_wb, contact_body_vel[tid])

    # relative velocity
    v = pv - bv

    # normal
    lambda_n = c
    delta_n = n * lambda_n

    # friction
    vn = wp.dot(n, v)
    vt = v - n * vn

    # compute inverse masses
    w1 = particle_invmass[particle_index]
    w2 = 0.0
    if body_index >= 0:
        angular = wp.cross(r, n)
        q = wp.transform_get_rotation(X_wb)
        rot_angular = wp.quat_rotate_inv(q, angular)
        I_inv = body_I_inv[body_index]
        w2 = body_m_inv[body_index] + wp.dot(rot_angular, I_inv * rot_angular)
    denom = w1 + w2
    if denom == 0.0:
        return

    lambda_f = wp.max(mu * lambda_n, -wp.length(vt) * dt)
    delta_f = wp.normalize(vt) * lambda_f
    delta_total = (delta_f - delta_n) / denom * relaxation

    wp.atomic_add(delta, particle_index, w1 * delta_total)

    if body_index >= 0:
        # apply_body_deltas() treats body_delta as a velocity-like correction:
        # it multiplies by inverse mass/inertia and dt to update the body pose.
        # delta_total is a positional contact correction, matching the particle
        # path above, so convert it to the body-delta convention here.
        delta_v = delta_total / dt
        delta_w = wp.cross(r, delta_v)
        wp.atomic_sub(body_delta, body_index, wp.spatial_vector(delta_v, delta_w))


@wp.kernel
def solve_particle_particle_contacts(
    grid: wp.uint64,
    particle_x: wp.array[wp.vec3],
    particle_v: wp.array[wp.vec3],
    particle_invmass: wp.array[float],
    particle_radius: wp.array[float],
    particle_flags: wp.array[wp.int32],
    k_mu: float,
    k_cohesion: float,
    max_radius: float,
    dt: float,
    relaxation: float,
    # outputs
    deltas: wp.array[wp.vec3],
):
    tid = wp.tid()

    # order threads by cell
    i = wp.hash_grid_point_id(grid, tid)
    if i == -1:
        # hash grid has not been built yet
        return
    particle_flag = particle_flags[i]
    if (particle_flag & ParticleFlags.ACTIVE) == 0:
        return
    is_proxy = particle_flag & ParticleFlags.PROXY

    x = particle_x[i]
    v = particle_v[i]
    radius = particle_radius[i]
    w1 = particle_invmass[i]

    # particle contact
    query = wp.hash_grid_query(grid, x, radius + max_radius + k_cohesion)
    index = int(0)

    delta = wp.vec3(0.0)

    while wp.hash_grid_query_next(query, index):
        neighbor_flag = particle_flags[index]
        if (
            (neighbor_flag & ParticleFlags.ACTIVE) != 0
            and (is_proxy == 0 or ((neighbor_flag & ParticleFlags.PROXY) == 0 and particle_invmass[index] > 0.0))
            and index != i
        ):
            # compute distance to point
            n = x - particle_x[index]
            d = wp.length(n)
            err = d - radius - particle_radius[index]

            # compute inverse masses
            w2 = particle_invmass[index]
            denom = w1 + w2

            if err <= k_cohesion and denom > 0.0 and d > 0.0:
                n = n / d
                vrel = v - particle_v[index]

                # normal
                lambda_n = err
                delta_n = n * lambda_n

                # friction
                vn = wp.dot(n, vrel)
                vt = vrel - n * vn

                lambda_f = wp.max(k_mu * lambda_n, -wp.length(vt) * dt)
                delta_f = wp.normalize(vt) * lambda_f
                delta += (delta_f - delta_n) / denom

    wp.atomic_add(deltas, i, delta * w1 * relaxation)


@wp.kernel
def solve_springs(
    x: wp.array[wp.vec3],
    v: wp.array[wp.vec3],
    invmass: wp.array[float],
    spring_indices: wp.array[int],
    spring_rest_lengths: wp.array[float],
    spring_stiffness: wp.array[float],
    spring_damping: wp.array[float],
    dt: float,
    lambdas: wp.array[float],
    delta: wp.array[wp.vec3],
):
    tid = wp.tid()

    i = spring_indices[tid * 2 + 0]
    j = spring_indices[tid * 2 + 1]

    ke = spring_stiffness[tid]
    kd = spring_damping[tid]
    rest = spring_rest_lengths[tid]

    xi = x[i]
    xj = x[j]

    vi = v[i]
    vj = v[j]

    xij = xi - xj
    vij = vi - vj

    l = wp.length(xij)

    if l == 0.0:
        return

    n = xij / l

    c = l - rest
    grad_c_xi = n
    grad_c_xj = -1.0 * n

    wi = invmass[i]
    wj = invmass[j]

    denom = wi + wj

    # Note strict inequality for damping -- 0 damping is ok
    if denom <= 0.0 or ke <= 0.0 or kd < 0.0:
        return

    alpha = 1.0 / (ke * dt * dt)
    gamma = kd / (ke * dt)

    grad_c_dot_v = dt * wp.dot(grad_c_xi, vij)  # Note: dt because from the paper we want x_i - x^n, not v...
    dlambda = -1.0 * (c + alpha * lambdas[tid] + gamma * grad_c_dot_v) / ((1.0 + gamma) * denom + alpha)

    dxi = wi * dlambda * grad_c_xi
    dxj = wj * dlambda * grad_c_xj

    lambdas[tid] = lambdas[tid] + dlambda

    wp.atomic_add(delta, i, dxi)
    wp.atomic_add(delta, j, dxj)


@wp.kernel
def bending_constraint(
    x: wp.array[wp.vec3],
    v: wp.array[wp.vec3],
    invmass: wp.array[float],
    indices: wp.array2d[int],
    rest: wp.array[float],
    bending_properties: wp.array2d[float],
    dt: float,
    lambdas: wp.array[float],
    delta: wp.array[wp.vec3],
):
    tid = wp.tid()
    eps = 1.0e-6

    ke = bending_properties[tid, 0]
    kd = bending_properties[tid, 1]

    i = indices[tid, 0]
    j = indices[tid, 1]
    k = indices[tid, 2]
    l = indices[tid, 3]

    if i == -1 or j == -1 or k == -1 or l == -1:
        return

    rest_angle = rest[tid]

    x1 = x[i]
    x2 = x[j]
    x3 = x[k]
    x4 = x[l]

    v1 = v[i]
    v2 = v[j]
    v3 = v[k]
    v4 = v[l]

    w1 = invmass[i]
    w2 = invmass[j]
    w3 = invmass[k]
    w4 = invmass[l]

    n1 = wp.cross(x3 - x1, x4 - x1)  # normal to face 1
    n2 = wp.cross(x4 - x2, x3 - x2)  # normal to face 2
    e = x4 - x3

    n1_length = wp.length(n1)
    n2_length = wp.length(n2)
    e_length = wp.length(e)

    # Check for degenerate cases
    if n1_length < eps or n2_length < eps or e_length < eps:
        return

    n1_hat = n1 / n1_length
    n2_hat = n2 / n2_length
    e_hat = e / e_length

    cos_theta = wp.dot(n1_hat, n2_hat)
    sin_theta = wp.dot(wp.cross(n1_hat, n2_hat), e_hat)
    theta = wp.atan2(sin_theta, cos_theta)

    c = theta - rest_angle

    grad_x1 = -n1_hat * e_length
    grad_x2 = -n2_hat * e_length
    grad_x3 = -n1_hat * wp.dot(x1 - x4, e_hat) - n2_hat * wp.dot(x2 - x4, e_hat)
    grad_x4 = -n1_hat * wp.dot(x3 - x1, e_hat) - n2_hat * wp.dot(x3 - x2, e_hat)

    denominator = (
        w1 * wp.length_sq(grad_x1)
        + w2 * wp.length_sq(grad_x2)
        + w3 * wp.length_sq(grad_x3)
        + w4 * wp.length_sq(grad_x4)
    )

    # Note strict inequality for damping -- 0 damping is ok
    if denominator <= 0.0 or ke <= 0.0 or kd < 0.0:
        return

    alpha = 1.0 / (ke * dt * dt)
    gamma = kd / (ke * dt)

    grad_dot_v = dt * (wp.dot(grad_x1, v1) + wp.dot(grad_x2, v2) + wp.dot(grad_x3, v3) + wp.dot(grad_x4, v4))

    dlambda = -1.0 * (c + alpha * lambdas[tid] + gamma * grad_dot_v) / ((1.0 + gamma) * denominator + alpha)

    delta0 = w1 * dlambda * grad_x1
    delta1 = w2 * dlambda * grad_x2
    delta2 = w3 * dlambda * grad_x3
    delta3 = w4 * dlambda * grad_x4

    lambdas[tid] = lambdas[tid] + dlambda

    wp.atomic_add(delta, i, delta0)
    wp.atomic_add(delta, j, delta1)
    wp.atomic_add(delta, k, delta2)
    wp.atomic_add(delta, l, delta3)


@wp.kernel
def solve_tetrahedra(
    x: wp.array[wp.vec3],
    v: wp.array[wp.vec3],
    inv_mass: wp.array[float],
    indices: wp.array2d[int],
    rest_matrix: wp.array[wp.mat33],
    activation: wp.array[float],
    materials: wp.array2d[float],
    dt: float,
    relaxation: float,
    delta: wp.array[wp.vec3],
):
    # Tetrahedral XPBD constraint solve.
    #
    # ModelBuilder stores rest_matrix as inv(Dm), where
    # Dm = [x1_0 - x0_0, x2_0 - x0_0, x3_0 - x0_0] in the rest pose.  Each
    # iteration rebuilds Ds from the current particle positions and computes the
    # deformation gradient
    #
    #     F = Ds * inv(Dm).
    #
    # The material is the same compressible Neo-Hookean-style split used by the
    # FEM path: a distortional term controlled by the first Lame parameter
    # k_mu, and a volume term controlled by the second Lame parameter k_lambda.
    # In XPBD form these are solved as two scalar constraints:
    #
    #     C_dev = trace(F^T F) - 3
    #     C_vol = det(F) - 1 + activation
    #
    # Their gradients are dC/dF = 2F for C_dev and cof(F) for C_vol.  The chain
    # rule dF/dx contributes inv(Dm)^T, giving the per-particle gradients below.
    #
    # A tetrahedron's energy scales with rest volume V0, so the XPBD compliance
    # for a material stiffness k is 1 / (V0 * k).  Since rest_matrix is inv(Dm),
    # det(rest_matrix) * 6 = 1 / V0.
    #
    # Damping uses XPBD's compliant Rayleigh term:
    #
    #     gamma = k_damp / (k * dt)
    #     dlambda = -(C + gamma * dt * grad(C).dot(v))
    #               / ((1 + gamma) * sum_i(w_i |grad_i C|^2) + alpha)
    #
    # The solver does not persist lambdas for this constraint, so each iteration
    # computes a local multiplier and accumulates relaxed position corrections.
    tid = wp.tid()

    i = indices[tid, 0]
    j = indices[tid, 1]
    k = indices[tid, 2]
    l = indices[tid, 3]

    act = activation[tid]

    k_mu = materials[tid, 0]
    k_lambda = materials[tid, 1]
    k_damp = materials[tid, 2]

    x0 = x[i]
    x1 = x[j]
    x2 = x[k]
    x3 = x[l]

    v0 = v[i]
    v1 = v[j]
    v2 = v[k]
    v3 = v[l]

    w0 = inv_mass[i]
    w1 = inv_mass[j]
    w2 = inv_mass[k]
    w3 = inv_mass[l]

    x10 = x1 - x0
    x20 = x2 - x0
    x30 = x3 - x0

    Ds = wp.matrix_from_cols(x10, x20, x30)
    Dm = rest_matrix[tid]
    inv_QT = wp.transpose(Dm)

    inv_rest_volume = wp.determinant(Dm) * 6.0
    if inv_rest_volume <= 0.0 or k_mu <= 0.0 or k_lambda <= 0.0:
        return

    # F = Xs*Xm^-1
    F = Ds * Dm

    f1 = wp.vec3(F[0, 0], F[1, 0], F[2, 0])
    f2 = wp.vec3(F[0, 1], F[1, 1], F[2, 1])
    f3 = wp.vec3(F[0, 2], F[1, 2], F[2, 2])

    tr = wp.dot(f1, f1) + wp.dot(f2, f2) + wp.dot(f3, f3)

    C = float(0.0)
    dC = wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    compliance = float(0.0)
    stiffness = float(0.0)

    num_terms = 2
    for term in range(0, num_terms):
        if term == 0:
            # deviatoric, stable
            C = tr - 3.0
            dC = F * 2.0
            compliance = inv_rest_volume / k_mu
            stiffness = k_mu
        elif term == 1:
            # volume conservation
            C = wp.determinant(F) - 1.0 + act
            dC = wp.matrix_from_cols(wp.cross(f2, f3), wp.cross(f3, f1), wp.cross(f1, f2))
            compliance = inv_rest_volume / k_lambda
            stiffness = k_lambda

        if C != 0.0:
            dP = dC * inv_QT
            grad1 = wp.vec3(dP[0][0], dP[1][0], dP[2][0])
            grad2 = wp.vec3(dP[0][1], dP[1][1], dP[2][1])
            grad3 = wp.vec3(dP[0][2], dP[1][2], dP[2][2])
            grad0 = -grad1 - grad2 - grad3

            w = (
                wp.dot(grad0, grad0) * w0
                + wp.dot(grad1, grad1) * w1
                + wp.dot(grad2, grad2) * w2
                + wp.dot(grad3, grad3) * w3
            )

            if w > 0.0:
                alpha = compliance / dt / dt
                gamma = float(0.0)
                grad_dot_v = float(0.0)
                if k_damp > 0.0 and stiffness > 0.0:
                    gamma = k_damp / (stiffness * dt)
                    grad_dot_v = dt * (wp.dot(grad0, v0) + wp.dot(grad1, v1) + wp.dot(grad2, v2) + wp.dot(grad3, v3))
                dlambda = -1.0 * (C + gamma * grad_dot_v) / ((1.0 + gamma) * w + alpha)

                wp.atomic_add(delta, i, w0 * dlambda * grad0 * relaxation)
                wp.atomic_add(delta, j, w1 * dlambda * grad1 * relaxation)
                wp.atomic_add(delta, k, w2 * dlambda * grad2 * relaxation)
                wp.atomic_add(delta, l, w3 * dlambda * grad3 * relaxation)
                # wp.atomic_add(particle.num_corr, id0, 1)
                # wp.atomic_add(particle.num_corr, id1, 1)
                # wp.atomic_add(particle.num_corr, id2, 1)
                # wp.atomic_add(particle.num_corr, id3, 1)

    # C_Spherical
    # r_s = wp.sqrt(wp.dot(f1, f1) + wp.dot(f2, f2) + wp.dot(f3, f3))
    # r_s_inv = 1.0/r_s
    # C = r_s - wp.sqrt(3.0)
    # dCdx = F*wp.transpose(Dm)*r_s_inv
    # alpha = 1.0

    # C_D
    # r_s = wp.sqrt(wp.dot(f1, f1) + wp.dot(f2, f2) + wp.dot(f3, f3))
    # C = r_s*r_s - 3.0
    # dCdx = F*wp.transpose(Dm)*2.0
    # alpha = 1.0

    # grad1 = wp.vec3(dCdx[0, 0], dCdx[1, 0], dCdx[2, 0])
    # grad2 = wp.vec3(dCdx[0, 1], dCdx[1, 1], dCdx[2, 1])
    # grad3 = wp.vec3(dCdx[0, 2], dCdx[1, 2], dCdx[2, 2])
    # grad0 = (grad1 + grad2 + grad3) * (0.0 - 1.0)

    # denom = (
    #     wp.dot(grad0, grad0) * w0 + wp.dot(grad1, grad1) * w1 + wp.dot(grad2, grad2) * w2 + wp.dot(grad3, grad3) * w3
    # )
    # multiplier = C / (denom + 1.0 / (k_mu * dt * dt * rest_volume))

    # delta0 = grad0 * multiplier
    # delta1 = grad1 * multiplier
    # delta2 = grad2 * multiplier
    # delta3 = grad3 * multiplier

    # # hydrostatic part
    # J = wp.determinant(F)

    # C_vol = J - alpha
    # # dCdx = wp.matrix_from_cols(wp.cross(f2, f3), wp.cross(f3, f1), wp.cross(f1, f2))*wp.transpose(Dm)

    # # grad1 = wp.vec3(dCdx[0,0], dCdx[1,0], dCdx[2,0])
    # # grad2 = wp.vec3(dCdx[0,1], dCdx[1,1], dCdx[2,1])
    # # grad3 = wp.vec3(dCdx[0,2], dCdx[1,2], dCdx[2,2])
    # # grad0 = (grad1 + grad2 + grad3)*(0.0 - 1.0)

    # s = inv_rest_volume / 6.0
    # grad1 = wp.cross(x20, x30) * s
    # grad2 = wp.cross(x30, x10) * s
    # grad3 = wp.cross(x10, x20) * s
    # grad0 = -(grad1 + grad2 + grad3)

    # denom = (
    #     wp.dot(grad0, grad0) * w0 + wp.dot(grad1, grad1) * w1 + wp.dot(grad2, grad2) * w2 + wp.dot(grad3, grad3) * w3
    # )
    # multiplier = C_vol / (denom + 1.0 / (k_lambda * dt * dt * rest_volume))

    # delta0 += grad0 * multiplier
    # delta1 += grad1 * multiplier
    # delta2 += grad2 * multiplier
    # delta3 += grad3 * multiplier

    # # # apply forces
    # # wp.atomic_sub(delta, i, delta0 * w0 * relaxation)
    # # wp.atomic_sub(delta, j, delta1 * w1 * relaxation)
    # # wp.atomic_sub(delta, k, delta2 * w2 * relaxation)
    # # wp.atomic_sub(delta, l, delta3 * w3 * relaxation)


@wp.kernel
def solve_tetrahedra2(
    x: wp.array[wp.vec3],
    v: wp.array[wp.vec3],
    inv_mass: wp.array[float],
    indices: wp.array2d[int],
    pose: wp.array[wp.mat33],
    activation: wp.array[float],
    materials: wp.array2d[float],
    dt: float,
    relaxation: float,
    delta: wp.array[wp.vec3],
):
    tid = wp.tid()

    i = indices[tid, 0]
    j = indices[tid, 1]
    k = indices[tid, 2]
    l = indices[tid, 3]

    # act = activation[tid]

    k_mu = materials[tid, 0]
    k_lambda = materials[tid, 1]
    # k_damp = materials[tid, 2]

    x0 = x[i]
    x1 = x[j]
    x2 = x[k]
    x3 = x[l]

    w0 = inv_mass[i]
    w1 = inv_mass[j]
    w2 = inv_mass[k]
    w3 = inv_mass[l]

    x10 = x1 - x0
    x20 = x2 - x0
    x30 = x3 - x0

    Ds = wp.matrix_from_cols(x10, x20, x30)
    Dm = pose[tid]

    inv_rest_volume = wp.determinant(Dm) * 6.0
    rest_volume = 1.0 / inv_rest_volume

    # F = Xs*Xm^-1
    F = Ds * Dm

    f1 = wp.vec3(F[0, 0], F[1, 0], F[2, 0])
    f2 = wp.vec3(F[0, 1], F[1, 1], F[2, 1])
    f3 = wp.vec3(F[0, 2], F[1, 2], F[2, 2])

    # C_sqrt
    # tr = wp.dot(f1, f1) + wp.dot(f2, f2) + wp.dot(f3, f3)
    # r_s = wp.sqrt(abs(tr - 3.0))
    # C = r_s

    # if (r_s == 0.0):
    #     return

    # if (tr < 3.0):
    #     r_s = 0.0 - r_s

    # dCdx = F*wp.transpose(Dm)*(1.0/r_s)
    # alpha = 1.0 + k_mu / k_lambda

    # C_Neo
    r_s = wp.sqrt(wp.dot(f1, f1) + wp.dot(f2, f2) + wp.dot(f3, f3))
    if r_s == 0.0:
        return
    # tr = wp.dot(f1, f1) + wp.dot(f2, f2) + wp.dot(f3, f3)
    # if (tr < 3.0):
    #     r_s = -r_s
    r_s_inv = 1.0 / r_s
    C = r_s
    dCdx = F * wp.transpose(Dm) * r_s_inv
    alpha = 1.0 + k_mu / k_lambda

    # C_Spherical
    # r_s = wp.sqrt(wp.dot(f1, f1) + wp.dot(f2, f2) + wp.dot(f3, f3))
    # r_s_inv = 1.0/r_s
    # C = r_s - wp.sqrt(3.0)
    # dCdx = F*wp.transpose(Dm)*r_s_inv
    # alpha = 1.0

    # C_D
    # r_s = wp.sqrt(wp.dot(f1, f1) + wp.dot(f2, f2) + wp.dot(f3, f3))
    # C = r_s*r_s - 3.0
    # dCdx = F*wp.transpose(Dm)*2.0
    # alpha = 1.0

    grad1 = wp.vec3(dCdx[0, 0], dCdx[1, 0], dCdx[2, 0])
    grad2 = wp.vec3(dCdx[0, 1], dCdx[1, 1], dCdx[2, 1])
    grad3 = wp.vec3(dCdx[0, 2], dCdx[1, 2], dCdx[2, 2])
    grad0 = (grad1 + grad2 + grad3) * (0.0 - 1.0)

    denom = (
        wp.dot(grad0, grad0) * w0 + wp.dot(grad1, grad1) * w1 + wp.dot(grad2, grad2) * w2 + wp.dot(grad3, grad3) * w3
    )
    multiplier = C / (denom + 1.0 / (k_mu * dt * dt * rest_volume))

    delta0 = grad0 * multiplier
    delta1 = grad1 * multiplier
    delta2 = grad2 * multiplier
    delta3 = grad3 * multiplier

    # hydrostatic part
    J = wp.determinant(F)

    C_vol = J - alpha
    # dCdx = wp.matrix_from_cols(wp.cross(f2, f3), wp.cross(f3, f1), wp.cross(f1, f2))*wp.transpose(Dm)

    # grad1 = wp.vec3(dCdx[0,0], dCdx[1,0], dCdx[2,0])
    # grad2 = wp.vec3(dCdx[0,1], dCdx[1,1], dCdx[2,1])
    # grad3 = wp.vec3(dCdx[0,2], dCdx[1,2], dCdx[2,2])
    # grad0 = (grad1 + grad2 + grad3)*(0.0 - 1.0)

    s = inv_rest_volume / 6.0
    grad1 = wp.cross(x20, x30) * s
    grad2 = wp.cross(x30, x10) * s
    grad3 = wp.cross(x10, x20) * s
    grad0 = -(grad1 + grad2 + grad3)

    denom = (
        wp.dot(grad0, grad0) * w0 + wp.dot(grad1, grad1) * w1 + wp.dot(grad2, grad2) * w2 + wp.dot(grad3, grad3) * w3
    )
    multiplier = C_vol / (denom + 1.0 / (k_lambda * dt * dt * rest_volume))

    delta0 += grad0 * multiplier
    delta1 += grad1 * multiplier
    delta2 += grad2 * multiplier
    delta3 += grad3 * multiplier

    # apply forces
    wp.atomic_sub(delta, i, delta0 * w0 * relaxation)
    wp.atomic_sub(delta, j, delta1 * w1 * relaxation)
    wp.atomic_sub(delta, k, delta2 * w2 * relaxation)
    wp.atomic_sub(delta, l, delta3 * w3 * relaxation)


@wp.kernel
def apply_particle_deltas(
    x_orig: wp.array[wp.vec3],
    x_pred: wp.array[wp.vec3],
    particle_flags: wp.array[wp.int32],
    delta: wp.array[wp.vec3],
    dt: float,
    v_max: float,
    x_out: wp.array[wp.vec3],
    v_out: wp.array[wp.vec3],
):
    tid = wp.tid()
    if (particle_flags[tid] & ParticleFlags.ACTIVE) == 0:
        return

    x0 = x_orig[tid]
    xp = x_pred[tid]

    # constraint deltas
    d = delta[tid]

    x_new = xp + d
    v_new = (x_new - x0) / dt

    # enforce velocity limit to prevent instability
    v_new_mag = wp.length(v_new)
    if v_new_mag > v_max:
        v_new *= v_max / v_new_mag
        x_new = x0 + v_new * dt

    x_out[tid] = x_new
    v_out[tid] = v_new


@wp.kernel
def apply_body_deltas(
    q_in: wp.array[wp.transform],
    qd_in: wp.array[wp.spatial_vector],
    body_com: wp.array[wp.vec3],
    body_I: wp.array[wp.mat33],
    body_inv_m: wp.array[float],
    body_inv_I: wp.array[wp.mat33],
    deltas: wp.array[wp.spatial_vector],
    constraint_inv_weights: wp.array[float],
    dt: float,
    # outputs
    q_out: wp.array[wp.transform],
    qd_out: wp.array[wp.spatial_vector],
):
    tid = wp.tid()
    inv_m = body_inv_m[tid]
    if inv_m == 0.0:
        q_out[tid] = q_in[tid]
        qd_out[tid] = qd_in[tid]
        return
    inv_I = body_inv_I[tid]

    tf = q_in[tid]
    delta = deltas[tid]

    v0 = wp.spatial_top(qd_in[tid])
    w0 = wp.spatial_bottom(qd_in[tid])

    p0 = wp.transform_get_translation(tf)
    q0 = wp.transform_get_rotation(tf)

    weight = 1.0
    if constraint_inv_weights:
        inv_weight = constraint_inv_weights[tid]
        if inv_weight > 0.0:
            weight = 1.0 / inv_weight

    dp = wp.spatial_top(delta) * (inv_m * weight)
    dq = wp.spatial_bottom(delta) * weight

    wb = wp.quat_rotate_inv(q0, w0)
    dwb = inv_I * wp.quat_rotate_inv(q0, dq)
    # coriolis forces delta from dwb = (wb + dwb) I (wb + dwb) - wb I wb
    tb = wp.cross(dwb, body_I[tid] * (wb + dwb)) + wp.cross(wb, body_I[tid] * dwb)
    dw1 = wp.quat_rotate(q0, dwb - dt * inv_I * tb)

    # update orientation
    q1 = q0 + 0.5 * wp.quat(dw1 * dt, 0.0) * q0
    q1 = wp.normalize(q1)

    # update position
    com = body_com[tid]
    x_com = p0 + wp.quat_rotate(q0, com)
    p1 = x_com + dp * dt
    p1 -= wp.quat_rotate(q1, com)

    q_out[tid] = wp.transform(p1, q1)

    # update linear and angular velocity
    v1 = v0 + dp
    w1 = w0 + dw1

    # XXX this improves gradient stability
    if wp.length(v1) < 1e-4:
        v1 = wp.vec3(0.0)
    if wp.length(w1) < 1e-4:
        w1 = wp.vec3(0.0)

    qd_out[tid] = wp.spatial_vector(v1, w1)


@wp.kernel
def apply_body_delta_velocities(
    deltas: wp.array[wp.spatial_vector],
    constraint_inv_weights: wp.array[float],
    qd_out: wp.array[wp.spatial_vector],
):
    tid = wp.tid()
    weight = 1.0
    if constraint_inv_weights:
        inv_weight = constraint_inv_weights[tid]
        if inv_weight > 0.0:
            weight = 1.0 / inv_weight
    wp.atomic_add(qd_out, tid, deltas[tid] * weight)


@wp.kernel
def apply_joint_forces(
    body_q: wp.array[wp.transform],
    body_com: wp.array[wp.vec3],
    joint_type: wp.array[int],
    joint_enabled: wp.array[bool],
    joint_parent: wp.array[int],
    joint_child: wp.array[int],
    joint_X_p: wp.array[wp.transform],
    joint_X_c: wp.array[wp.transform],
    joint_qd_start: wp.array[int],
    joint_dof_dim: wp.array2d[int],
    joint_axis: wp.array[wp.vec3],
    joint_f: wp.array[float],
    dt: float,
    body_f: wp.array[wp.spatial_vector],
    joint_impulse: wp.array[wp.spatial_vector],
):
    tid = wp.tid()
    type = joint_type[tid]
    if not joint_enabled[tid]:
        return
    if type == JointType.FIXED or type == JointType.ROD:
        return

    # rigid body indices of the child and parent
    id_c = joint_child[tid]
    id_p = joint_parent[tid]

    X_pj = joint_X_p[tid]
    X_cj = joint_X_c[tid]

    X_wp = X_pj
    pose_p = X_pj
    com_p = wp.vec3(0.0)
    # parent transform and moment arm
    if id_p >= 0:
        pose_p = body_q[id_p]
        X_wp = pose_p * X_wp
        com_p = body_com[id_p]
    r_p = wp.transform_get_translation(X_wp) - wp.transform_point(pose_p, com_p)

    # child transform and moment arm
    pose_c = body_q[id_c]
    X_wc = pose_c * X_cj
    com_c = body_com[id_c]
    r_c = wp.transform_get_translation(X_wc) - wp.transform_point(pose_c, com_c)

    # # local joint rotations
    # q_p = wp.transform_get_rotation(X_wp)
    # q_c = wp.transform_get_rotation(X_wc)

    # joint properties (for 1D joints)
    qd_start = joint_qd_start[tid]
    lin_axis_count = joint_dof_dim[tid, 0]
    ang_axis_count = joint_dof_dim[tid, 1]

    # total force/torque on the parent
    t_total = wp.vec3()
    f_total = wp.vec3()

    if type == JointType.FREE or type == JointType.DISTANCE:
        f_total = wp.vec3(joint_f[qd_start + 0], joint_f[qd_start + 1], joint_f[qd_start + 2])
        t_total = wp.vec3(joint_f[qd_start + 3], joint_f[qd_start + 4], joint_f[qd_start + 5])
        # Interpret free-joint forces as spatial wrench at the COM (same as body_f).
        # Avoid adding a moment arm that would introduce torque for pure forces.
        wp.atomic_add(body_f, id_c, wp.spatial_vector(f_total, t_total))
        if id_p >= 0:
            wp.atomic_sub(body_f, id_p, wp.spatial_vector(f_total, t_total))
        # Record the contribution to the inbound joint wrench (used to populate
        # ``State.body_parent_f``).  For FREE joints this is a diagnostic only;
        # for DISTANCE joints the constraint solver adds its own contribution.
        # Convention: positive = wrench transmitted parent->child at child COM.
        if joint_impulse:
            wp.atomic_add(joint_impulse, tid, wp.spatial_vector(f_total, t_total) * dt)
        return
    elif type == JointType.BALL:
        t_total = wp.vec3(joint_f[qd_start + 0], joint_f[qd_start + 1], joint_f[qd_start + 2])

    elif type == JointType.REVOLUTE or type == JointType.PRISMATIC or type == JointType.D6:
        # unroll for loop to ensure joint actions remain differentiable
        # (since differentiating through a dynamic for loop that updates a local variable is not supported)

        if lin_axis_count > 0:
            axis = joint_axis[qd_start + 0]
            f = joint_f[qd_start + 0]
            a_p = wp.transform_vector(X_wp, axis)
            f_total += f * a_p
        if lin_axis_count > 1:
            axis = joint_axis[qd_start + 1]
            f = joint_f[qd_start + 1]
            a_p = wp.transform_vector(X_wp, axis)
            f_total += f * a_p
        if lin_axis_count > 2:
            axis = joint_axis[qd_start + 2]
            f = joint_f[qd_start + 2]
            a_p = wp.transform_vector(X_wp, axis)
            f_total += f * a_p

        if ang_axis_count > 0:
            axis = joint_axis[qd_start + lin_axis_count + 0]
            f = joint_f[qd_start + lin_axis_count + 0]
            a_p = wp.transform_vector(X_wp, axis)
            t_total += f * a_p
        if ang_axis_count > 1:
            axis = joint_axis[qd_start + lin_axis_count + 1]
            f = joint_f[qd_start + lin_axis_count + 1]
            a_p = wp.transform_vector(X_wp, axis)
            t_total += f * a_p
        if ang_axis_count > 2:
            axis = joint_axis[qd_start + lin_axis_count + 2]
            f = joint_f[qd_start + lin_axis_count + 2]
            a_p = wp.transform_vector(X_wp, axis)
            t_total += f * a_p

    else:
        print("joint type not handled in apply_joint_forces")

    # write forces
    child_wrench_at_com = wp.spatial_vector(f_total, t_total + wp.cross(r_c, f_total))
    if id_p >= 0:
        wp.atomic_sub(body_f, id_p, wp.spatial_vector(f_total, t_total + wp.cross(r_p, f_total)))
    wp.atomic_add(body_f, id_c, child_wrench_at_com)

    # Record the joint-f contribution to the inbound joint wrench (used to
    # populate ``State.body_parent_f``).  We accumulate the child-side spatial
    # wrench (linear ``[N]``, torque ``[N·m]`` at the child COM, world frame)
    # multiplied by ``dt`` so that the same `impulse / dt` conversion applied
    # in :func:`convert_joint_impulse_to_parent_f` recovers the wrench.
    if joint_impulse:
        wp.atomic_add(joint_impulse, tid, child_wrench_at_com * dt)


@wp.func
def update_joint_axis_limits(axis: wp.vec3, limit_lower: float, limit_upper: float, input_limits: wp.spatial_vector):
    # update the 3D linear/angular limits (spatial_vector [lower, upper]) given the axis vector and limits
    lo_temp = axis * limit_lower
    up_temp = axis * limit_upper
    lo = vec_min(lo_temp, up_temp)
    up = vec_max(lo_temp, up_temp)
    input_lower = wp.spatial_top(input_limits)
    input_upper = wp.spatial_bottom(input_limits)
    lower = vec_min(input_lower, lo)
    upper = vec_max(input_upper, up)
    return wp.spatial_vector(lower, upper)


@wp.func
def update_joint_axis_weighted_target(
    axis: wp.vec3, target: float, weight: float, input_target_weight: wp.spatial_vector
):
    axis_targets = wp.spatial_top(input_target_weight)
    axis_weights = wp.spatial_bottom(input_target_weight)

    weighted_axis = axis * weight
    axis_targets += weighted_axis * target  # weighted target (to be normalized later by sum of weights)
    axis_weights += vec_abs(weighted_axis)

    return wp.spatial_vector(axis_targets, axis_weights)


@wp.func
def compute_linear_correction_3d(
    dx: wp.vec3,
    r1: wp.vec3,
    r2: wp.vec3,
    tf1: wp.transform,
    tf2: wp.transform,
    m_inv1: float,
    m_inv2: float,
    I_inv1: wp.mat33,
    I_inv2: wp.mat33,
    lambda_in: float,
    compliance: float,
    damping: float,
    dt: float,
) -> float:
    c = wp.length(dx)
    if c == 0.0:
        # print("c == 0.0 in positional correction")
        return 0.0

    n = wp.normalize(dx)

    q1 = wp.transform_get_rotation(tf1)
    q2 = wp.transform_get_rotation(tf2)

    # Eq. 2-3 (make sure to project into the frame of the body)
    r1xn = wp.quat_rotate_inv(q1, wp.cross(r1, n))
    r2xn = wp.quat_rotate_inv(q2, wp.cross(r2, n))

    w1 = m_inv1 + wp.dot(r1xn, I_inv1 * r1xn)
    w2 = m_inv2 + wp.dot(r2xn, I_inv2 * r2xn)
    w = w1 + w2
    if w == 0.0:
        return 0.0
    alpha = compliance
    gamma = compliance * damping

    # Eq. 4-5
    d_lambda = -c - alpha * lambda_in
    # TODO consider damping for velocity correction?
    # delta_lambda = -(err + alpha * lambda_in + gamma * derr)
    if w + alpha > 0.0:
        d_lambda /= w * (dt + gamma) + alpha / dt

    return d_lambda


@wp.func
def compute_angular_correction_3d(
    corr: wp.vec3,
    q1: wp.quat,
    q2: wp.quat,
    m_inv1: float,
    m_inv2: float,
    I_inv1: wp.mat33,
    I_inv2: wp.mat33,
    alpha_tilde: float,
    # lambda_prev: float,
    relaxation: float,
    dt: float,
):
    # compute and apply the correction impulse for an angular constraint
    theta = wp.length(corr)
    if theta == 0.0:
        return 0.0

    n = wp.normalize(corr)

    # project variables to body rest frame as they are in local matrix
    n1 = wp.quat_rotate_inv(q1, n)
    n2 = wp.quat_rotate_inv(q2, n)

    # Eq. 11-12
    w1 = wp.dot(n1, I_inv1 * n1)
    w2 = wp.dot(n2, I_inv2 * n2)
    w = w1 + w2
    if w == 0.0:
        return 0.0

    # Eq. 13-14
    lambda_prev = 0.0
    d_lambda = (-theta - alpha_tilde * lambda_prev) / (w * dt + alpha_tilde / dt)
    # TODO consider lambda_prev?
    # p = d_lambda * n * relaxation

    # Eq. 15-16
    return d_lambda


@wp.kernel
def solve_simple_body_joints(
    body_q: wp.array[wp.transform],
    body_qd: wp.array[wp.spatial_vector],
    body_com: wp.array[wp.vec3],
    body_inv_m: wp.array[float],
    body_inv_I: wp.array[wp.mat33],
    joint_type: wp.array[int],
    joint_enabled: wp.array[bool],
    joint_parent: wp.array[int],
    joint_child: wp.array[int],
    joint_X_p: wp.array[wp.transform],
    joint_X_c: wp.array[wp.transform],
    joint_limit_lower: wp.array[float],
    joint_limit_upper: wp.array[float],
    joint_qd_start: wp.array[int],
    joint_dof_dim: wp.array2d[int],
    joint_axis: wp.array[wp.vec3],
    joint_target: wp.array[float],
    joint_target_ke: wp.array[float],
    joint_target_kd: wp.array[float],
    joint_linear_compliance: float,
    joint_angular_compliance: float,
    angular_relaxation: float,
    linear_relaxation: float,
    dt: float,
    deltas: wp.array[wp.spatial_vector],
):
    tid = wp.tid()
    type = joint_type[tid]

    if not joint_enabled[tid]:
        return
    if type == JointType.FREE:
        return
    if type == JointType.DISTANCE:
        return
    if type == JointType.D6:
        return

    # rigid body indices of the child and parent
    id_c = joint_child[tid]
    id_p = joint_parent[tid]

    X_pj = joint_X_p[tid]
    X_cj = joint_X_c[tid]

    X_wp = X_pj
    m_inv_p = 0.0
    I_inv_p = wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    pose_p = X_pj
    com_p = wp.vec3(0.0)
    # parent transform and moment arm
    if id_p >= 0:
        pose_p = body_q[id_p]
        X_wp = pose_p * X_wp
        com_p = body_com[id_p]
        m_inv_p = body_inv_m[id_p]
        I_inv_p = body_inv_I[id_p]
    r_p = wp.transform_get_translation(X_wp) - wp.transform_point(pose_p, com_p)

    # child transform and moment arm
    pose_c = body_q[id_c]
    X_wc = pose_c * X_cj
    com_c = body_com[id_c]
    m_inv_c = body_inv_m[id_c]
    I_inv_c = body_inv_I[id_c]
    r_c = wp.transform_get_translation(X_wc) - wp.transform_point(pose_c, com_c)

    if m_inv_p == 0.0 and m_inv_c == 0.0:
        # connection between two immovable bodies
        return

    # accumulate constraint deltas
    lin_delta_p = wp.vec3(0.0)
    ang_delta_p = wp.vec3(0.0)
    lin_delta_c = wp.vec3(0.0)
    ang_delta_c = wp.vec3(0.0)

    # rel_pose = wp.transform_inverse(X_wp) * X_wc
    # rel_p = wp.transform_get_translation(rel_pose)

    # joint connection points
    # x_p = wp.transform_get_translation(X_wp)
    x_c = wp.transform_get_translation(X_wc)

    # linear_compliance = joint_linear_compliance
    angular_compliance = joint_angular_compliance
    damping = 0.0

    axis_start = joint_qd_start[tid]
    # mode = joint_dof_mode[axis_start]

    # local joint rotations
    q_p = wp.transform_get_rotation(X_wp)
    q_c = wp.transform_get_rotation(X_wc)
    inertial_q_p = wp.transform_get_rotation(pose_p)
    inertial_q_c = wp.transform_get_rotation(pose_c)

    # joint properties (for 1D joints)
    axis = joint_axis[axis_start]

    if type == JointType.FIXED:
        limit_lower = 0.0
        limit_upper = 0.0
    else:
        limit_lower = joint_limit_lower[axis_start]
        limit_upper = joint_limit_upper[axis_start]

    # linear_alpha_tilde = linear_compliance / dt / dt
    angular_alpha_tilde = angular_compliance / dt / dt

    # prevent division by zero
    # linear_alpha_tilde = wp.max(linear_alpha_tilde, 1e-6)
    # angular_alpha_tilde = wp.max(angular_alpha_tilde, 1e-6)

    # accumulate constraint deltas
    lin_delta_p = wp.vec3(0.0)
    ang_delta_p = wp.vec3(0.0)
    lin_delta_c = wp.vec3(0.0)
    ang_delta_c = wp.vec3(0.0)

    # handle angular constraints
    if type == JointType.REVOLUTE:
        # align joint axes
        a_p = wp.quat_rotate(q_p, axis)
        a_c = wp.quat_rotate(q_c, axis)
        # Eq. 20
        corr = wp.cross(a_p, a_c)
        ncorr = wp.normalize(corr)

        angular_relaxation = 0.2
        # angular_correction(
        #     corr, inertial_q_p, inertial_q_c, m_inv_p, m_inv_c, I_inv_p, I_inv_c,
        #     angular_alpha_tilde, angular_relaxation, deltas, id_p, id_c)
        lambda_n = compute_angular_correction_3d(
            corr, inertial_q_p, inertial_q_c, m_inv_p, m_inv_c, I_inv_p, I_inv_c, angular_alpha_tilde, damping, dt
        )
        lambda_n *= angular_relaxation
        ang_delta_p -= lambda_n * ncorr
        ang_delta_c += lambda_n * ncorr

        # limit joint angles (Alg. 3)
        pi = 3.14159265359
        two_pi = 2.0 * pi
        if limit_lower > -two_pi or limit_upper < two_pi:
            # find a perpendicular vector to joint axis
            a = axis
            # https://math.stackexchange.com/a/3582461
            g = wp.sign(a[2])
            h = a[2] + g
            b = wp.vec3(g - a[0] * a[0] / h, -a[0] * a[1] / h, -a[0])
            c = wp.normalize(wp.cross(a, b))
            # b = c  # TODO verify

            # joint axis
            n = wp.quat_rotate(q_p, a)
            # the axes n1 and n2 are aligned with the two bodies
            n1 = wp.quat_rotate(q_p, b)
            n2 = wp.quat_rotate(q_c, b)

            phi = wp.asin(wp.dot(wp.cross(n1, n2), n))
            # print("phi")
            # print(phi)
            if wp.dot(n1, n2) < 0.0:
                phi = pi - phi
            if phi > pi:
                phi -= two_pi
            if phi < -pi:
                phi += two_pi
            if phi < limit_lower or phi > limit_upper:
                phi = wp.clamp(phi, limit_lower, limit_upper)
                # print("clamped phi")
                # print(phi)
                # rot = wp.quat(phi, n[0], n[1], n[2])
                # rot = wp.quat(n, phi)
                rot = wp.quat_from_axis_angle(n, phi)
                n1 = wp.quat_rotate(rot, n1)
                corr = wp.cross(n1, n2)
                # print("corr")
                # print(corr)
                # TODO expose
                # angular_alpha_tilde = 0.0001 / dt / dt
                # angular_relaxation = 0.5
                # TODO fix this constraint
                # angular_correction(
                #     corr, inertial_q_p, inertial_q_c, m_inv_p, m_inv_c, I_inv_p, I_inv_c,
                #     angular_alpha_tilde, angular_relaxation, deltas, id_p, id_c)
                lambda_n = compute_angular_correction_3d(
                    corr,
                    inertial_q_p,
                    inertial_q_c,
                    m_inv_p,
                    m_inv_c,
                    I_inv_p,
                    I_inv_c,
                    angular_alpha_tilde,
                    damping,
                    dt,
                )
                lambda_n *= angular_relaxation
                ncorr = wp.normalize(corr)
                ang_delta_p -= lambda_n * ncorr
                ang_delta_c += lambda_n * ncorr

        # handle joint targets
        target_ke = joint_target_ke[axis_start]
        # target_kd = joint_target_kd[axis_start]
        target = joint_target[axis_start]
        if target_ke > 0.0:
            # find a perpendicular vector to joint axis
            a = axis
            # https://math.stackexchange.com/a/3582461
            g = wp.sign(a[2])
            h = a[2] + g
            b = wp.vec3(g - a[0] * a[0] / h, -a[0] * a[1] / h, -a[0])
            c = wp.normalize(wp.cross(a, b))
            b = c

            q = wp.quat_from_axis_angle(a_p, target)
            b_target = wp.quat_rotate(q, wp.quat_rotate(q_p, b))
            b2 = wp.quat_rotate(q_c, b)
            # Eq. 21
            d_target = wp.cross(b_target, b2)

            target_compliance = 1.0 / target_ke  # / dt / dt
            # angular_correction(
            #     d_target, inertial_q_p, inertial_q_c, m_inv_p, m_inv_c, I_inv_p, I_inv_c,
            #     target_compliance, angular_relaxation, deltas, id_p, id_c)
            lambda_n = compute_angular_correction_3d(
                d_target, inertial_q_p, inertial_q_c, m_inv_p, m_inv_c, I_inv_p, I_inv_c, target_compliance, damping, dt
            )
            lambda_n *= angular_relaxation
            ncorr = wp.normalize(d_target)
            # TODO fix
            ang_delta_p -= lambda_n * ncorr
            ang_delta_c += lambda_n * ncorr

    if (type == JointType.FIXED) or (type == JointType.PRISMATIC):
        # align the mutual orientations of the two bodies
        # Eq. 18-19
        q = q_p * wp.quat_inverse(q_c)
        corr = -2.0 * wp.vec3(q[0], q[1], q[2])
        # angular_correction(
        #     -corr, inertial_q_p, inertial_q_c, m_inv_p, m_inv_c, I_inv_p, I_inv_c,
        #     angular_alpha_tilde, angular_relaxation, deltas, id_p, id_c)
        lambda_n = compute_angular_correction_3d(
            corr, inertial_q_p, inertial_q_c, m_inv_p, m_inv_c, I_inv_p, I_inv_c, angular_alpha_tilde, damping, dt
        )
        lambda_n *= angular_relaxation
        ncorr = wp.normalize(corr)
        ang_delta_p -= lambda_n * ncorr
        ang_delta_c += lambda_n * ncorr

    # handle positional constraints

    # joint connection points
    x_p = wp.transform_get_translation(X_wp)
    x_c = wp.transform_get_translation(X_wc)

    # compute error between the joint attachment points on both bodies
    # delta x is the difference of point r_2 minus point r_1 (Fig. 3)
    dx = x_c - x_p

    # rotate the error vector into the joint frame
    q_dx = q_p
    # q_dx = q_c
    # q_dx = wp.transform_get_rotation(pose_p)
    dx = wp.quat_rotate_inv(q_dx, dx)

    lower_pos_limits = wp.vec3(0.0)
    upper_pos_limits = wp.vec3(0.0)
    if type == JointType.PRISMATIC:
        lower_pos_limits = axis * limit_lower
        upper_pos_limits = axis * limit_upper

    # compute linear constraint violations
    corr = wp.vec3(0.0)
    zero = wp.vec3(0.0)
    corr -= vec_leaky_min(zero, upper_pos_limits - dx)
    corr -= vec_leaky_max(zero, lower_pos_limits - dx)

    # if (type == JointType.PRISMATIC):
    #     if mode == JointMode.TARGET_POSITION:
    #         target = wp.clamp(target, limit_lower, limit_upper)
    #         if target_ke > 0.0:
    #             err = dx - target * axis
    #             compliance = 1.0 / target_ke
    #         damping = axis_damping[dim]
    #     elif mode == JointMode.TARGET_VELOCITY:
    #         if target_ke > 0.0:
    #             err = (derr - target) * dt
    #             compliance = 1.0 / target_ke
    #         damping = axis_damping[dim]

    # rotate correction vector into world frame
    corr = wp.quat_rotate(q_dx, corr)

    lambda_in = 0.0
    linear_alpha = joint_linear_compliance
    lambda_n = compute_linear_correction_3d(
        corr, r_p, r_c, pose_p, pose_c, m_inv_p, m_inv_c, I_inv_p, I_inv_c, lambda_in, linear_alpha, damping, dt
    )
    lambda_n *= linear_relaxation
    n = wp.normalize(corr)

    lin_delta_p -= n * lambda_n
    lin_delta_c += n * lambda_n
    ang_delta_p -= wp.cross(r_p, n) * lambda_n
    ang_delta_c += wp.cross(r_c, n) * lambda_n

    if id_p >= 0:
        wp.atomic_add(deltas, id_p, wp.spatial_vector(lin_delta_p, ang_delta_p))
    if id_c >= 0:
        wp.atomic_add(deltas, id_c, wp.spatial_vector(lin_delta_c, ang_delta_c))


@wp.kernel
def solve_body_joints(
    body_q: wp.array[wp.transform],
    body_qd: wp.array[wp.spatial_vector],
    body_com: wp.array[wp.vec3],
    body_inv_m: wp.array[float],
    body_inv_I: wp.array[wp.mat33],
    joint_type: wp.array[int],
    joint_enabled: wp.array[bool],
    joint_parent: wp.array[int],
    joint_child: wp.array[int],
    joint_X_p: wp.array[wp.transform],
    joint_X_c: wp.array[wp.transform],
    joint_limit_lower: wp.array[float],
    joint_limit_upper: wp.array[float],
    joint_qd_start: wp.array[int],
    joint_target_q_start: wp.array[int],
    joint_dof_dim: wp.array2d[int],
    joint_axis: wp.array[wp.vec3],
    joint_target_q: wp.array[float],
    joint_target_qd: wp.array[float],
    joint_target_ke: wp.array[float],
    joint_target_kd: wp.array[float],
    joint_linear_compliance: float,
    joint_angular_compliance: float,
    angular_relaxation: float,
    linear_relaxation: float,
    dt: float,
    deltas: wp.array[wp.spatial_vector],
    joint_impulse: wp.array[wp.spatial_vector],
):
    tid = wp.tid()
    type = joint_type[tid]

    if not joint_enabled[tid]:
        return
    if type == JointType.FREE:
        return
    # if type == JointType.FIXED:
    #     return
    # if type == JointType.REVOLUTE:
    #     return
    # if type == JointType.PRISMATIC:
    #     return
    # if type == JointType.BALL:
    #     return

    # rigid body indices of the child and parent
    id_c = joint_child[tid]
    id_p = joint_parent[tid]

    X_pj = joint_X_p[tid]
    X_cj = joint_X_c[tid]

    X_wp = X_pj
    m_inv_p = 0.0
    I_inv_p = wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    pose_p = X_pj
    com_p = wp.vec3(0.0)
    vel_p = wp.vec3(0.0)
    omega_p = wp.vec3(0.0)
    # parent transform and moment arm
    if id_p >= 0:
        pose_p = body_q[id_p]
        X_wp = pose_p * X_wp
        com_p = body_com[id_p]
        m_inv_p = body_inv_m[id_p]
        I_inv_p = body_inv_I[id_p]
        vel_p = wp.spatial_top(body_qd[id_p])
        omega_p = wp.spatial_bottom(body_qd[id_p])

    # child transform and moment arm
    pose_c = body_q[id_c]
    X_wc = pose_c * X_cj
    com_c = body_com[id_c]
    m_inv_c = body_inv_m[id_c]
    I_inv_c = body_inv_I[id_c]
    vel_c = wp.spatial_top(body_qd[id_c])
    omega_c = wp.spatial_bottom(body_qd[id_c])

    if m_inv_p == 0.0 and m_inv_c == 0.0:
        # connection between two immovable bodies
        return

    # accumulate constraint deltas
    lin_delta_p = wp.vec3(0.0)
    ang_delta_p = wp.vec3(0.0)
    lin_delta_c = wp.vec3(0.0)
    ang_delta_c = wp.vec3(0.0)

    rel_pose = wp.transform_inverse(X_wp) * X_wc
    rel_p = wp.transform_get_translation(rel_pose)

    # joint connection points
    x_p = wp.transform_get_translation(X_wp)
    x_c = wp.transform_get_translation(X_wc)

    linear_compliance = joint_linear_compliance
    angular_compliance = joint_angular_compliance

    axis_start = joint_qd_start[tid]
    target_axis_start = joint_target_q_start[tid]
    lin_axis_count = joint_dof_dim[tid, 0]
    ang_axis_count = joint_dof_dim[tid, 1]

    world_com_p = wp.transform_point(pose_p, com_p)
    world_com_c = wp.transform_point(pose_c, com_c)

    # handle positional constraints
    if type == JointType.DISTANCE:
        r_p = x_p - world_com_p
        r_c = x_c - world_com_c
        lower = joint_limit_lower[axis_start]
        upper = joint_limit_upper[axis_start]
        if lower < 0.0 and upper < 0.0:
            # no limits
            return
        anchor_delta = x_c - x_p
        d = wp.length(anchor_delta)
        err = 0.0
        if lower >= 0.0 and d < lower:
            err = d - lower
        elif upper >= 0.0 and d > upper:
            err = d - upper

        if wp.abs(err) > 1e-9:
            # compute gradients
            if d > 1e-9:
                linear_c = anchor_delta / d
            else:
                com_delta = world_com_c - world_com_p
                if wp.length_sq(com_delta) > 1e-18:
                    linear_c = wp.normalize(com_delta)
                else:
                    # The parent joint frame supplies a stable direction when the geometry cannot.
                    linear_c = wp.transform_vector(X_wp, wp.vec3(1.0, 0.0, 0.0))
            linear_p = -linear_c
            angular_p = -wp.cross(r_p, linear_c)
            angular_c = wp.cross(r_c, linear_c)
            # constraint time derivative
            derr = (
                wp.dot(linear_p, vel_p)
                + wp.dot(linear_c, vel_c)
                + wp.dot(angular_p, omega_p)
                + wp.dot(angular_c, omega_c)
            )
            lambda_in = 0.0
            compliance = linear_compliance
            ke = joint_target_ke[axis_start]
            if ke > 0.0:
                compliance = 1.0 / ke
            damping = joint_target_kd[axis_start]
            d_lambda = compute_positional_correction(
                err,
                derr,
                pose_p,
                pose_c,
                m_inv_p,
                m_inv_c,
                I_inv_p,
                I_inv_c,
                linear_p,
                linear_c,
                angular_p,
                angular_c,
                lambda_in,
                compliance,
                damping,
                dt,
            )

            lin_delta_p += linear_p * (d_lambda * linear_relaxation)
            ang_delta_p += angular_p * (d_lambda * angular_relaxation)
            lin_delta_c += linear_c * (d_lambda * linear_relaxation)
            ang_delta_c += angular_c * (d_lambda * angular_relaxation)

    else:
        # compute joint target, stiffness, damping
        axis_limits = wp.spatial_vector(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

        axis_target_pos_ke = wp.spatial_vector()
        axis_target_vel_kd = wp.spatial_vector()
        # avoid a for loop here since local variables would need to be modified which is not yet differentiable
        if lin_axis_count > 0:
            axis = joint_axis[axis_start]
            lo_temp = axis * joint_limit_lower[axis_start]
            up_temp = axis * joint_limit_upper[axis_start]
            axis_limits = wp.spatial_vector(vec_min(lo_temp, up_temp), vec_max(lo_temp, up_temp))
            ke = joint_target_ke[axis_start]
            kd = joint_target_kd[axis_start]
            target_pos = joint_target_q[target_axis_start]
            target_vel = joint_target_qd[axis_start]
            if ke > 0.0:  # has position control
                axis_target_pos_ke = update_joint_axis_weighted_target(axis, target_pos, ke, axis_target_pos_ke)
            if kd > 0.0:  # has velocity control
                axis_target_vel_kd = update_joint_axis_weighted_target(axis, target_vel, kd, axis_target_vel_kd)
        if lin_axis_count > 1:
            axis_idx = axis_start + 1
            target_axis_idx = target_axis_start + 1
            axis = joint_axis[axis_idx]
            lower = joint_limit_lower[axis_idx]
            upper = joint_limit_upper[axis_idx]
            axis_limits = update_joint_axis_limits(axis, lower, upper, axis_limits)
            ke = joint_target_ke[axis_idx]
            kd = joint_target_kd[axis_idx]
            target_pos = joint_target_q[target_axis_idx]
            target_vel = joint_target_qd[axis_idx]
            if ke > 0.0:  # has position control
                axis_target_pos_ke = update_joint_axis_weighted_target(axis, target_pos, ke, axis_target_pos_ke)
            if kd > 0.0:  # has velocity control
                axis_target_vel_kd = update_joint_axis_weighted_target(axis, target_vel, kd, axis_target_vel_kd)
        if lin_axis_count > 2:
            axis_idx = axis_start + 2
            target_axis_idx = target_axis_start + 2
            axis = joint_axis[axis_idx]
            lower = joint_limit_lower[axis_idx]
            upper = joint_limit_upper[axis_idx]
            axis_limits = update_joint_axis_limits(axis, lower, upper, axis_limits)
            ke = joint_target_ke[axis_idx]
            kd = joint_target_kd[axis_idx]
            target_pos = joint_target_q[target_axis_idx]
            target_vel = joint_target_qd[axis_idx]
            if ke > 0.0:  # has position control
                axis_target_pos_ke = update_joint_axis_weighted_target(axis, target_pos, ke, axis_target_pos_ke)
            if kd > 0.0:  # has velocity control
                axis_target_vel_kd = update_joint_axis_weighted_target(axis, target_vel, kd, axis_target_vel_kd)

        axis_target_pos = wp.spatial_top(axis_target_pos_ke)
        axis_stiffness = wp.spatial_bottom(axis_target_pos_ke)
        axis_target_vel = wp.spatial_top(axis_target_vel_kd)
        axis_damping = wp.spatial_bottom(axis_target_vel_kd)
        for i in range(3):
            if axis_stiffness[i] > 0.0:
                axis_target_pos[i] /= axis_stiffness[i]
        for i in range(3):
            if axis_damping[i] > 0.0:
                axis_target_vel[i] /= axis_damping[i]
        axis_limits_lower = wp.spatial_top(axis_limits)
        axis_limits_upper = wp.spatial_bottom(axis_limits)

        # A relative offset can contain both valid joint motion and error. For example,
        # a prismatic joint may be 0.5 m along its free axis and 1 m off it.
        # Start from the current offset so unconstrained coordinates keep their extension.
        projected_rel_p = rel_p
        for dim in range(3):
            lower = axis_limits_lower[dim]
            upper = axis_limits_upper[dim]
            # Limit violations project to the nearest admissible boundary.
            if rel_p[dim] < lower:
                projected_rel_p[dim] = lower
            elif rel_p[dim] > upper:
                projected_rel_p[dim] = upper
            # A position-driven coordinate projects to its target. Locked coordinates
            # have zero-width limits above and therefore already project to zero.
            elif axis_stiffness[dim] > 0.0:
                projected_rel_p[dim] = wp.clamp(axis_target_pos[dim], lower, upper)

        frame_p = wp.quat_to_matrix(wp.transform_get_rotation(X_wp))
        # Use the admissible point for the parent lever arm: the parent anchor would
        # discard valid extension, while the child anchor would include separation error.
        r_p = wp.transform_point(X_wp, projected_rel_p) - world_com_p
        r_c = x_c - world_com_c

        # for loop will be unrolled, so we can modify local variables
        for dim in range(3):
            e = rel_p[dim]

            # compute gradients
            linear_c = wp.vec3(frame_p[0, dim], frame_p[1, dim], frame_p[2, dim])
            linear_p = -linear_c
            angular_p = -wp.cross(r_p, linear_c)
            angular_c = wp.cross(r_c, linear_c)
            # constraint time derivative
            derr = (
                wp.dot(linear_p, vel_p)
                + wp.dot(linear_c, vel_c)
                + wp.dot(angular_p, omega_p)
                + wp.dot(angular_c, omega_c)
            )

            err = 0.0
            compliance = linear_compliance
            damping = 0.0

            target_vel = axis_target_vel[dim]
            derr_rel = derr - target_vel

            # consider joint limits irrespective of axis mode
            lower = axis_limits_lower[dim]
            upper = axis_limits_upper[dim]
            if e < lower:
                err = e - lower
            elif e > upper:
                err = e - upper
            else:
                target_pos = axis_target_pos[dim]
                target_pos = wp.clamp(target_pos, lower, upper)

                if axis_stiffness[dim] > 0.0:
                    err = e - target_pos
                    compliance = 1.0 / axis_stiffness[dim]
                    damping = axis_damping[dim]
                elif axis_damping[dim] > 0.0:
                    compliance = 1.0 / axis_damping[dim]
                    damping = axis_damping[dim]

            if wp.abs(err) > 1e-9 or wp.abs(derr_rel) > 1e-9:
                lambda_in = 0.0
                d_lambda = compute_positional_correction(
                    err,
                    derr_rel,
                    pose_p,
                    pose_c,
                    m_inv_p,
                    m_inv_c,
                    I_inv_p,
                    I_inv_c,
                    linear_p,
                    linear_c,
                    angular_p,
                    angular_c,
                    lambda_in,
                    compliance,
                    damping,
                    dt,
                )

                lin_delta_p += linear_p * (d_lambda * linear_relaxation)
                ang_delta_p += angular_p * (d_lambda * angular_relaxation)
                lin_delta_c += linear_c * (d_lambda * linear_relaxation)
                ang_delta_c += angular_c * (d_lambda * angular_relaxation)

    if type == JointType.FIXED or type == JointType.PRISMATIC or type == JointType.REVOLUTE or type == JointType.D6:
        # handle angular constraints

        # local joint rotations
        q_p = wp.transform_get_rotation(X_wp)
        q_c = wp.transform_get_rotation(X_wc)

        # make quats lie in same hemisphere
        if wp.dot(q_p, q_c) < 0.0:
            q_c *= -1.0

        rel_q = wp.quat_inverse(q_p) * q_c

        qtwist = wp.normalize(wp.quat(rel_q[0], 0.0, 0.0, rel_q[3]))
        qswing = rel_q * wp.quat_inverse(qtwist)

        # decompose to a compound rotation each axis
        s = wp.sqrt(rel_q[0] * rel_q[0] + rel_q[3] * rel_q[3])
        invs = 1.0 / s
        invscube = invs * invs * invs

        # handle axis-angle joints

        # rescale twist from quaternion space to angular
        err_0 = 2.0 * wp.asin(wp.clamp(qtwist[0], -1.0, 1.0))
        err_1 = qswing[1]
        err_2 = qswing[2]
        # analytic gradients of swing-twist decomposition
        grad_0 = wp.quat(invs - rel_q[0] * rel_q[0] * invscube, 0.0, 0.0, -(rel_q[3] * rel_q[0]) * invscube)
        grad_1 = wp.quat(
            -rel_q[3] * (rel_q[3] * rel_q[2] + rel_q[0] * rel_q[1]) * invscube,
            rel_q[3] * invs,
            -rel_q[0] * invs,
            rel_q[0] * (rel_q[3] * rel_q[2] + rel_q[0] * rel_q[1]) * invscube,
        )
        grad_2 = wp.quat(
            rel_q[3] * (rel_q[3] * rel_q[1] - rel_q[0] * rel_q[2]) * invscube,
            rel_q[0] * invs,
            rel_q[3] * invs,
            rel_q[0] * (rel_q[2] * rel_q[0] - rel_q[3] * rel_q[1]) * invscube,
        )
        grad_0 *= 2.0 / wp.abs(qtwist[3])
        # grad_0 *= 2.0 / wp.sqrt(1.0-qtwist[0]*qtwist[0])	# derivative of asin(x) = 1/sqrt(1-x^2)

        # rescale swing
        swing_sq = qswing[3] * qswing[3]
        # if swing axis magnitude close to zero vector, just treat in quaternion space
        angularEps = 1.0e-4
        if swing_sq + angularEps < 1.0:
            d = wp.sqrt(1.0 - qswing[3] * qswing[3])
            theta = 2.0 * wp.acos(wp.clamp(qswing[3], -1.0, 1.0))
            scale = theta / d

            err_1 *= scale
            err_2 *= scale

            grad_1 *= scale
            grad_2 *= scale

        errs = wp.vec3(err_0, err_1, err_2)
        grad_x = wp.vec3(grad_0[0], grad_1[0], grad_2[0])
        grad_y = wp.vec3(grad_0[1], grad_1[1], grad_2[1])
        grad_z = wp.vec3(grad_0[2], grad_1[2], grad_2[2])
        grad_w = wp.vec3(grad_0[3], grad_1[3], grad_2[3])

        # compute joint target, stiffness, damping
        axis_limits = wp.spatial_vector(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

        axis_target_pos_ke = wp.spatial_vector()  # [weighted_target_pos, ke_weights]
        axis_target_vel_kd = wp.spatial_vector()  # [weighted_target_vel, kd_weights]
        # avoid a for loop here since local variables would need to be modified which is not yet differentiable
        if ang_axis_count > 0:
            axis_idx = axis_start + lin_axis_count
            target_axis_idx = target_axis_start + lin_axis_count
            axis = joint_axis[axis_idx]
            lo_temp = axis * joint_limit_lower[axis_idx]
            up_temp = axis * joint_limit_upper[axis_idx]
            axis_limits = wp.spatial_vector(vec_min(lo_temp, up_temp), vec_max(lo_temp, up_temp))
            ke = joint_target_ke[axis_idx]
            kd = joint_target_kd[axis_idx]
            target_pos = joint_target_q[target_axis_idx]
            target_vel = joint_target_qd[axis_idx]
            if ke > 0.0:  # has position control
                axis_target_pos_ke = update_joint_axis_weighted_target(axis, target_pos, ke, axis_target_pos_ke)
            if kd > 0.0:  # has velocity control
                axis_target_vel_kd = update_joint_axis_weighted_target(axis, target_vel, kd, axis_target_vel_kd)
        if ang_axis_count > 1:
            axis_idx = axis_start + lin_axis_count + 1
            target_axis_idx = target_axis_start + lin_axis_count + 1
            axis = joint_axis[axis_idx]
            lower = joint_limit_lower[axis_idx]
            upper = joint_limit_upper[axis_idx]
            axis_limits = update_joint_axis_limits(axis, lower, upper, axis_limits)
            ke = joint_target_ke[axis_idx]
            kd = joint_target_kd[axis_idx]
            target_pos = joint_target_q[target_axis_idx]
            target_vel = joint_target_qd[axis_idx]
            if ke > 0.0:  # has position control
                axis_target_pos_ke = update_joint_axis_weighted_target(axis, target_pos, ke, axis_target_pos_ke)
            if kd > 0.0:  # has velocity control
                axis_target_vel_kd = update_joint_axis_weighted_target(axis, target_vel, kd, axis_target_vel_kd)
        if ang_axis_count > 2:
            axis_idx = axis_start + lin_axis_count + 2
            target_axis_idx = target_axis_start + lin_axis_count + 2
            axis = joint_axis[axis_idx]
            lower = joint_limit_lower[axis_idx]
            upper = joint_limit_upper[axis_idx]
            axis_limits = update_joint_axis_limits(axis, lower, upper, axis_limits)
            ke = joint_target_ke[axis_idx]
            kd = joint_target_kd[axis_idx]
            target_pos = joint_target_q[target_axis_idx]
            target_vel = joint_target_qd[axis_idx]
            if ke > 0.0:  # has position control
                axis_target_pos_ke = update_joint_axis_weighted_target(axis, target_pos, ke, axis_target_pos_ke)
            if kd > 0.0:  # has velocity control
                axis_target_vel_kd = update_joint_axis_weighted_target(axis, target_vel, kd, axis_target_vel_kd)

        axis_target_pos = wp.spatial_top(axis_target_pos_ke)
        axis_stiffness = wp.spatial_bottom(axis_target_pos_ke)
        axis_target_vel = wp.spatial_top(axis_target_vel_kd)
        axis_damping = wp.spatial_bottom(axis_target_vel_kd)
        for i in range(3):
            if axis_stiffness[i] > 0.0:
                axis_target_pos[i] /= axis_stiffness[i]
        for i in range(3):
            if axis_damping[i] > 0.0:
                axis_target_vel[i] /= axis_damping[i]
        axis_limits_lower = wp.spatial_top(axis_limits)
        axis_limits_upper = wp.spatial_bottom(axis_limits)

        # if type == JointType.D6:
        #     wp.printf("axis_target: %f %f %f\t axis_stiffness: %f %f %f\t axis_damping: %f %f %f\t axis_limits_lower: %f %f %f \t axis_limits_upper: %f %f %f\n",
        #               axis_target[0], axis_target[1], axis_target[2],
        #               axis_stiffness[0], axis_stiffness[1], axis_stiffness[2],
        #               axis_damping[0], axis_damping[1], axis_damping[2],
        #               axis_limits_lower[0], axis_limits_lower[1], axis_limits_lower[2],
        #               axis_limits_upper[0], axis_limits_upper[1], axis_limits_upper[2])
        #     # wp.printf("wp.sqrt(1.0-qtwist[0]*qtwist[0]) = %f\n", wp.sqrt(1.0-qtwist[0]*qtwist[0]))

        for dim in range(3):
            e = errs[dim]

            # analytic gradients of swing-twist decomposition
            grad = wp.quat(grad_x[dim], grad_y[dim], grad_z[dim], grad_w[dim])

            quat_c = 0.5 * q_p * grad * wp.quat_inverse(q_c)
            angular_c = wp.vec3(quat_c[0], quat_c[1], quat_c[2])
            angular_p = -angular_c
            # time derivative of the constraint
            derr = wp.dot(angular_p, omega_p) + wp.dot(angular_c, omega_c)

            err = 0.0
            compliance = angular_compliance
            damping = 0.0

            target_vel = axis_target_vel[dim]
            angular_c_len = wp.length(angular_c)
            derr_rel = derr - target_vel * angular_c_len

            # consider joint limits irrespective of mode
            lower = axis_limits_lower[dim]
            upper = axis_limits_upper[dim]
            if e < lower:
                err = e - lower
            elif e > upper:
                err = e - upper
            else:
                target_pos = axis_target_pos[dim]
                target_pos = wp.clamp(target_pos, lower, upper)

                if axis_stiffness[dim] > 0.0:
                    err = e - target_pos
                    compliance = 1.0 / axis_stiffness[dim]
                    damping = axis_damping[dim]
                elif axis_damping[dim] > 0.0:
                    damping = axis_damping[dim]
                    compliance = 1.0 / axis_damping[dim]

            d_lambda = (
                compute_angular_correction(
                    err, derr_rel, pose_p, pose_c, I_inv_p, I_inv_c, angular_p, angular_c, 0.0, compliance, damping, dt
                )
                * angular_relaxation
            )

            # update deltas
            ang_delta_p += angular_p * d_lambda
            ang_delta_c += angular_c * d_lambda

    if id_p >= 0:
        wp.atomic_add(deltas, id_p, wp.spatial_vector(lin_delta_p, ang_delta_p))
    if id_c >= 0:
        wp.atomic_add(deltas, id_c, wp.spatial_vector(lin_delta_c, ang_delta_c))

    # Optionally accumulate the child-side spatial impulse for this joint.
    # The convention matches `body_parent_f`: incoming joint wrench in world
    # frame, referenced to the child body's COM (see `r_c` above which is
    # measured from the child COM).
    if joint_impulse:
        wp.atomic_add(joint_impulse, tid, wp.spatial_vector(lin_delta_c, ang_delta_c))


@wp.func
def compute_contact_constraint_delta(
    err: float,
    tf_a: wp.transform,
    tf_b: wp.transform,
    m_inv_a: float,
    m_inv_b: float,
    I_inv_a: wp.mat33,
    I_inv_b: wp.mat33,
    linear_a: wp.vec3,
    linear_b: wp.vec3,
    angular_a: wp.vec3,
    angular_b: wp.vec3,
    relaxation: float,
    dt: float,
) -> float:
    denom = 0.0
    denom += wp.length_sq(linear_a) * m_inv_a
    denom += wp.length_sq(linear_b) * m_inv_b

    q1 = wp.transform_get_rotation(tf_a)
    q2 = wp.transform_get_rotation(tf_b)

    # Eq. 2-3 (make sure to project into the frame of the body)
    rot_angular_a = wp.quat_rotate_inv(q1, angular_a)
    rot_angular_b = wp.quat_rotate_inv(q2, angular_b)

    denom += wp.dot(rot_angular_a, I_inv_a * rot_angular_a)
    denom += wp.dot(rot_angular_b, I_inv_b * rot_angular_b)

    delta_lambda = -err
    if denom > 0.0:
        delta_lambda /= dt * denom

    return delta_lambda * relaxation


@wp.func
def compute_positional_correction(
    err: float,
    derr: float,
    tf_a: wp.transform,
    tf_b: wp.transform,
    m_inv_a: float,
    m_inv_b: float,
    I_inv_a: wp.mat33,
    I_inv_b: wp.mat33,
    linear_a: wp.vec3,
    linear_b: wp.vec3,
    angular_a: wp.vec3,
    angular_b: wp.vec3,
    lambda_in: float,
    compliance: float,
    damping: float,
    dt: float,
) -> float:
    denom = 0.0
    denom += wp.length_sq(linear_a) * m_inv_a
    denom += wp.length_sq(linear_b) * m_inv_b

    q1 = wp.transform_get_rotation(tf_a)
    q2 = wp.transform_get_rotation(tf_b)

    # Eq. 2-3 (make sure to project into the frame of the body)
    rot_angular_a = wp.quat_rotate_inv(q1, angular_a)
    rot_angular_b = wp.quat_rotate_inv(q2, angular_b)

    denom += wp.dot(rot_angular_a, I_inv_a * rot_angular_a)
    denom += wp.dot(rot_angular_b, I_inv_b * rot_angular_b)

    alpha = compliance
    gamma = compliance * damping

    delta_lambda = -(err + alpha * lambda_in + gamma * derr)
    if denom + alpha > 0.0:
        delta_lambda /= (dt + gamma) * denom + alpha / dt

    return delta_lambda


@wp.func
def compute_angular_correction(
    err: float,
    derr: float,
    tf_a: wp.transform,
    tf_b: wp.transform,
    I_inv_a: wp.mat33,
    I_inv_b: wp.mat33,
    angular_a: wp.vec3,
    angular_b: wp.vec3,
    lambda_in: float,
    compliance: float,
    damping: float,
    dt: float,
) -> float:
    denom = 0.0

    q1 = wp.transform_get_rotation(tf_a)
    q2 = wp.transform_get_rotation(tf_b)

    # Eq. 2-3 (make sure to project into the frame of the body)
    rot_angular_a = wp.quat_rotate_inv(q1, angular_a)
    rot_angular_b = wp.quat_rotate_inv(q2, angular_b)

    denom += wp.dot(rot_angular_a, I_inv_a * rot_angular_a)
    denom += wp.dot(rot_angular_b, I_inv_b * rot_angular_b)

    alpha = compliance
    gamma = compliance * damping

    delta_lambda = -(err + alpha * lambda_in + gamma * derr)
    if denom + alpha > 0.0:
        delta_lambda /= (dt + gamma) * denom + alpha / dt

    return delta_lambda


@wp.kernel
def solve_body_contact_positions(
    body_q: wp.array[wp.transform],
    body_qd: wp.array[wp.spatial_vector],
    body_flags: wp.array[wp.int32],
    body_com: wp.array[wp.vec3],
    body_m_inv: wp.array[float],
    body_I_inv: wp.array[wp.mat33],
    shape_body: wp.array[int],
    contact_count: wp.array[int],
    contact_point0: wp.array[wp.vec3],
    contact_point1: wp.array[wp.vec3],
    contact_offset0: wp.array[wp.vec3],
    contact_offset1: wp.array[wp.vec3],
    contact_normal: wp.array[wp.vec3],
    contact_thickness0: wp.array[float],
    contact_thickness1: wp.array[float],
    contact_shape0: wp.array[int],
    contact_shape1: wp.array[int],
    shape_material_mu: wp.array[float],
    shape_material_mu_torsional: wp.array[float],
    shape_material_mu_rolling: wp.array[float],
    relaxation: float,
    dt: float,
    # outputs
    deltas: wp.array[wp.spatial_vector],
    contact_inv_weight: wp.array[float],
    contact_impulse: wp.array[wp.spatial_vector],
    restitution_contact_active: wp.array[wp.int32],
):
    tid = wp.tid()

    count = contact_count[0]
    if tid >= count:
        return

    shape_a = contact_shape0[tid]
    shape_b = contact_shape1[tid]
    if shape_a == shape_b:
        return
    body_a = -1
    if shape_a >= 0:
        body_a = shape_body[shape_a]
    body_b = -1
    if shape_b >= 0:
        body_b = shape_body[shape_b]
    if body_a == body_b:
        return

    # find body to world transform
    X_wb_a = wp.transform_identity()
    X_wb_b = wp.transform_identity()
    if body_a >= 0:
        X_wb_a = body_q[body_a]
    if body_b >= 0:
        X_wb_b = body_q[body_b]

    # compute body position in world space
    bx_a = wp.transform_point(X_wb_a, contact_point0[tid])
    bx_b = wp.transform_point(X_wb_b, contact_point1[tid])

    n = contact_normal[tid]
    d = contact_surface_separation(bx_a, bx_b, n, contact_thickness0[tid], contact_thickness1[tid])

    if d >= 0.0:
        return

    if restitution_contact_active:
        restitution_contact_active[tid] = 1

    m_inv_a = 0.0
    m_inv_b = 0.0
    I_inv_a = wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    I_inv_b = wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    # center of mass in body frame
    com_a = wp.vec3(0.0)
    com_b = wp.vec3(0.0)
    # body to world transform
    X_wb_a = wp.transform_identity()
    X_wb_b = wp.transform_identity()
    # angular velocities
    omega_a = wp.vec3(0.0)
    omega_b = wp.vec3(0.0)
    # contact offset in body frame
    offset_a = contact_offset0[tid]
    offset_b = contact_offset1[tid]

    if body_a >= 0:
        X_wb_a = body_q[body_a]
        com_a = body_com[body_a]
        m_inv_a = body_m_inv[body_a]
        I_inv_a = body_I_inv[body_a]
        omega_a = wp.spatial_bottom(body_qd[body_a])

    if body_b >= 0:
        X_wb_b = body_q[body_b]
        com_b = body_com[body_b]
        m_inv_b = body_m_inv[body_b]
        I_inv_b = body_I_inv[body_b]
        omega_b = wp.spatial_bottom(body_qd[body_b])

    # use average contact material properties
    mat_nonzero = 0
    mu = 0.0
    mu_torsional = 0.0
    mu_rolling = 0.0
    if shape_a >= 0:
        mat_nonzero += 1
        mu += shape_material_mu[shape_a]
        mu_torsional += shape_material_mu_torsional[shape_a]
        mu_rolling += shape_material_mu_rolling[shape_a]
    if shape_b >= 0:
        mat_nonzero += 1
        mu += shape_material_mu[shape_b]
        mu_torsional += shape_material_mu_torsional[shape_b]
        mu_rolling += shape_material_mu_rolling[shape_b]
    if mat_nonzero > 0:
        mu /= float(mat_nonzero)
        mu_torsional /= float(mat_nonzero)
        mu_rolling /= float(mat_nonzero)

    r_a = bx_a - wp.transform_point(X_wb_a, com_a)
    r_b = bx_b - wp.transform_point(X_wb_b, com_b)

    angular_a = -wp.cross(r_a, n)
    angular_b = wp.cross(r_b, n)

    if contact_inv_weight:
        if body_a >= 0:
            wp.atomic_add(contact_inv_weight, body_a, 1.0)
        if body_b >= 0:
            wp.atomic_add(contact_inv_weight, body_b, 1.0)

    lambda_n = compute_contact_constraint_delta(
        d, X_wb_a, X_wb_b, m_inv_a, m_inv_b, I_inv_a, I_inv_b, -n, n, angular_a, angular_b, relaxation, dt
    )

    lin_delta_a = -n * lambda_n
    lin_delta_b = n * lambda_n
    ang_delta_a = angular_a * lambda_n
    ang_delta_b = angular_b * lambda_n

    # linear friction
    if mu > 0.0:
        # add on displacement from surface offsets, this ensures we include any rotational effects due to thickness from feature
        # need to use the current rotation to account for friction due to angular effects (e.g.: slipping contact)
        bx_a = contact_surface_point(X_wb_a, contact_point0[tid], offset_a)
        bx_b = contact_surface_point(X_wb_b, contact_point1[tid], offset_b)

        # update delta
        delta = bx_b - bx_a
        friction_delta = delta - wp.dot(n, delta) * n

        r_a = bx_a - wp.transform_point(X_wb_a, com_a)
        r_b = bx_b - wp.transform_point(X_wb_b, com_b)

        # Add only prescribed kinematic surface motion here.
        # Dynamic-body tangential motion is already reflected in the
        # positional slip `delta`; adding full relative velocity would
        # double-count ordinary ground friction and destabilize contacts.
        rel_v_kin_t = wp.vec3(0.0)
        if body_a >= 0 and (body_flags[body_a] & int(BodyFlags.KINEMATIC)) != 0:
            v_a = velocity_at_point(body_qd[body_a], r_a)
            rel_v_kin_t = rel_v_kin_t - (v_a - wp.dot(n, v_a) * n)
        if body_b >= 0 and (body_flags[body_b] & int(BodyFlags.KINEMATIC)) != 0:
            v_b = velocity_at_point(body_qd[body_b], r_b)
            rel_v_kin_t = rel_v_kin_t + (v_b - wp.dot(n, v_b) * n)
        friction_delta += rel_v_kin_t * dt

        perp = wp.normalize(friction_delta)

        angular_a = -wp.cross(r_a, perp)
        angular_b = wp.cross(r_b, perp)

        err = wp.length(friction_delta)

        if err > 0.0:
            lambda_fr = compute_contact_constraint_delta(
                err,
                X_wb_a,
                X_wb_b,
                m_inv_a,
                m_inv_b,
                I_inv_a,
                I_inv_b,
                -perp,
                perp,
                angular_a,
                angular_b,
                relaxation,
                dt,
            )

            # limit friction based on incremental normal force, good approximation to limiting on total force
            lambda_fr = wp.max(lambda_fr, -lambda_n * mu)

            lin_delta_a -= perp * lambda_fr
            lin_delta_b += perp * lambda_fr

            ang_delta_a += angular_a * lambda_fr
            ang_delta_b += angular_b * lambda_fr

    delta_omega = omega_b - omega_a

    if mu_torsional > 0.0:
        err = wp.dot(delta_omega, n) * dt

        if wp.abs(err) > 0.0:
            lin = wp.vec3(0.0)
            lambda_torsion = compute_contact_constraint_delta(
                err, X_wb_a, X_wb_b, m_inv_a, m_inv_b, I_inv_a, I_inv_b, lin, lin, -n, n, relaxation, dt
            )

            lambda_torsion = wp.clamp(lambda_torsion, -lambda_n * mu_torsional, lambda_n * mu_torsional)

            ang_delta_a -= n * lambda_torsion
            ang_delta_b += n * lambda_torsion

    if mu_rolling > 0.0:
        delta_omega -= wp.dot(n, delta_omega) * n
        err = wp.length(delta_omega) * dt
        if err > 0.0:
            lin = wp.vec3(0.0)
            roll_n = wp.normalize(delta_omega)
            lambda_roll = compute_contact_constraint_delta(
                err, X_wb_a, X_wb_b, m_inv_a, m_inv_b, I_inv_a, I_inv_b, lin, lin, -roll_n, roll_n, relaxation, dt
            )

            lambda_roll = wp.max(lambda_roll, -lambda_n * mu_rolling)

            ang_delta_a -= roll_n * lambda_roll
            ang_delta_b += roll_n * lambda_roll

    if body_a >= 0:
        wp.atomic_add(deltas, body_a, wp.spatial_vector(lin_delta_a, ang_delta_a))
    if body_b >= 0:
        wp.atomic_add(deltas, body_b, wp.spatial_vector(lin_delta_b, ang_delta_b))

    if contact_impulse:
        wp.atomic_add(contact_impulse, tid, wp.spatial_vector(lin_delta_a, ang_delta_a))


@wp.kernel
def accumulate_weighted_contact_impulse(
    contact_count: wp.array[int],
    contact_impulse_iter: wp.array[wp.spatial_vector],
    contact_shape0: wp.array[int],
    contact_shape1: wp.array[int],
    shape_body: wp.array[int],
    constraint_inv_weight: wp.array[float],
    # output (accumulated across iterations)
    contact_impulse: wp.array[wp.spatial_vector],
):
    """Scale per-contact impulse from one iteration by 1/N and accumulate.

    ``constraint_inv_weight[body]`` holds the number of active contacts on
    each body for the current iteration.  ``apply_body_deltas`` divides the
    positional correction by that count, so the raw impulse stored per contact
    is N times too large relative to what was actually applied.

    When only one body is dynamic (the other is kinematic / ground), the
    weight is simply ``1/N_dynamic``.  When both bodies are dynamic the
    solver applies ``1/N_a`` to body A and ``1/N_b`` to body B, so there is
    no single exact scalar.  We use the harmonic mean ``2/(N_a + N_b)`` which
    is symmetric with respect to body ordering and reduces to ``1/N`` when
    both counts are equal.
    """
    tid = wp.tid()
    count = contact_count[0]
    if tid >= count:
        return

    impulse = contact_impulse_iter[tid]

    weight = 1.0
    if constraint_inv_weight:
        n_a = 0.0
        n_b = 0.0
        shape_a = contact_shape0[tid]
        if shape_a >= 0:
            body_a = shape_body[shape_a]
            if body_a >= 0:
                n_a = constraint_inv_weight[body_a]
        shape_b = contact_shape1[tid]
        if shape_b >= 0:
            body_b = shape_body[shape_b]
            if body_b >= 0:
                n_b = constraint_inv_weight[body_b]
        n_sum = n_a + n_b
        if n_sum > 0.0:
            if n_a == 0.0:
                weight = 1.0 / n_b
            elif n_b == 0.0:
                weight = 1.0 / n_a
            else:
                weight = 2.0 / n_sum

    scaled = wp.spatial_vector(
        wp.spatial_top(impulse) * weight,
        wp.spatial_bottom(impulse) * weight,
    )
    wp.atomic_add(contact_impulse, tid, scaled)


@wp.kernel
def convert_contact_impulse_to_force(
    contact_count: wp.array[int],
    contact_impulse: wp.array[wp.spatial_vector],
    dt: float,
    # output
    contact_force: wp.array[wp.spatial_vector],
):
    """Convert accumulated per-contact spatial impulse to ``contacts.force`` spatial vectors.

    The XPBD lambda convention used in this solver already absorbs one power
    of ``dt`` (see ``compute_contact_constraint_delta``), so dividing the
    accumulated impulse by the substep ``dt`` yields force [N] and torque [N·m].
    The linear component includes normal and friction forces; the angular
    component includes torsional and rolling friction torques.

    The impulse is expected to already include the 1/N contact-weighting
    correction (applied by ``accumulate_weighted_contact_impulse`` each
    iteration).
    """
    tid = wp.tid()
    count = contact_count[0]
    if tid >= count:
        contact_force[tid] = wp.spatial_vector()
        return

    inv_dt = 1.0 / dt
    impulse = contact_impulse[tid]
    f = wp.spatial_top(impulse) * inv_dt
    tau = wp.spatial_bottom(impulse) * inv_dt
    contact_force[tid] = wp.spatial_vector(f, tau)


@wp.kernel
def convert_joint_impulse_to_parent_f(
    joint_impulse: wp.array[wp.spatial_vector],
    joint_enabled: wp.array[bool],
    joint_type: wp.array[int],
    joint_child: wp.array[int],
    dt: float,
    # output
    body_parent_f: wp.array[wp.spatial_vector],
):
    """Convert accumulated child-side joint impulse to ``state.body_parent_f``.

    The accumulated ``joint_impulse[joint_id]`` contains two contributions:

    * The XPBD constraint correction accumulated by ``solve_body_joints`` over
      every iteration.  The lambda convention used there already absorbs one
      power of ``dt`` (see ``compute_positional_correction`` /
      ``compute_angular_correction``), so dividing by the substep ``dt``
      yields the constraint reaction wrench.
    * The body-frame contribution from ``Control.joint_f`` recorded by
      ``apply_joint_forces``, pre-multiplied by ``dt`` for the same
      conversion to compose correctly.

    The result is the **total** wrench transmitted from the parent through the
    inbound joint to the child, expressed in world frame at the child body's
    COM (linear ``[N]``, torque ``[N·m]``).  This matches the convention used
    by :class:`SolverFeatherstone` and :class:`SolverMuJoCo`.

    Free joints and disabled joints contribute zero (their bodies inherit the
    zero-init from the caller).  Multiple joints sharing the same child body
    accumulate atomically, so loop-closure topologies remain race-free.
    """
    tid = wp.tid()

    if not joint_enabled[tid]:
        return
    if joint_type[tid] == JointType.FREE:
        return

    id_c = joint_child[tid]
    if id_c < 0:
        return

    inv_dt = 1.0 / dt
    impulse = joint_impulse[tid]
    f = wp.spatial_top(impulse) * inv_dt
    tau = wp.spatial_bottom(impulse) * inv_dt
    wp.atomic_add(body_parent_f, id_c, wp.spatial_vector(f, tau))


# Maximum number of contacts the restitution solve keeps per manifold.
# Manifolds larger than the cap are reduced to a bounded best-K subset by
# select_manifold_contacts (deepest anchor + greedy position/normal spread),
# which is lossless in practice because the velocity-level rigid solve has
# rank <= 6 per body pair. Single cap, no size bucketing.
RESTITUTION_MANIFOLD_MAX_CONTACTS = 12

_restitution_manifold_max = wp.constant(RESTITUTION_MANIFOLD_MAX_CONTACTS)
_restitution_vecNf = wp.types.vector(length=RESTITUTION_MANIFOLD_MAX_CONTACTS, dtype=wp.float32)
_restitution_vecNi = wp.types.vector(length=RESTITUTION_MANIFOLD_MAX_CONTACTS, dtype=wp.int32)


@wp.kernel
def build_restitution_manifolds(
    body_q_pre_solve: wp.array[wp.transform],
    body_qd_pre_solve: wp.array[wp.spatial_vector],
    body_com: wp.array[wp.vec3],
    body_m_inv: wp.array[float],
    body_I_inv: wp.array[wp.mat33],
    body_world: wp.array[wp.int32],
    shape_body: wp.array[int],
    contact_count: wp.array[int],
    restitution_contact_active: wp.array[wp.int32],
    contact_normal: wp.array[wp.vec3],
    contact_shape0: wp.array[int],
    contact_shape1: wp.array[int],
    shape_material_restitution: wp.array[float],
    contact_point0: wp.array[wp.vec3],
    contact_point1: wp.array[wp.vec3],
    contact_offset0: wp.array[wp.vec3],
    contact_offset1: wp.array[wp.vec3],
    gravity: wp.array[wp.vec3],
    dt: float,
    body_count: int,
    # outputs
    manifold_key: wp.array[wp.int64],
    manifold_head: wp.array[wp.int32],
    manifold_total: wp.array[wp.int32],
    contact_next: wp.array[wp.int32],
    contact_n_K: wp.array[wp.vec4],
    contact_axn_lo_target: wp.array[wp.vec4],
    contact_axn_hi_sigma: wp.array[wp.vec4],
    contact_pos_depth: wp.array[wp.vec4],
):
    """Group contacts that can fire restitution into per-body-pair manifolds.

    The collision pipeline interleaves contacts across pairs, so contacts are
    not pair-contiguous. Each active contact inserts its canonical body-pair
    key (``(lo+1)*(body_count+1) + hi+1 >= 1``; 0 means an empty slot) into an
    open-addressing hash table via ``atomic_cas`` and pushes its contact index
    onto the slot's uncapped linked chain
    (``manifold_head``/``contact_next``); :func:`select_manifold_contacts`
    later reduces each chain to a bounded best-K subset. Fixed-size table, no
    host sync: CUDA-graph-capture safe.

    Alongside, packed per-contact records consumed by
    :func:`solve_manifold_restitution` are cached:

    * ``contact_n_K``: contact normal and effective inverse mass
      ``K = m_inv + (r x n)^T I^-1 (r x n)`` summed over both bodies.
    * ``contact_axn_lo_target``: ``r_lo x n`` (world) and the restitution
      target ``-e * rel_vel_old`` for an impact, or the anchored velocity for
      an already-separating penetrating contact.
    * ``contact_axn_hi_sigma``: ``r_hi x n`` (world) and the normal sign
      ``sigma`` for the lower-indexed body (-1 when it is the shape0 side).
      A magnitude of 2 marks an already-separating contact, which only needs
      its positional correction removed on the first outer iteration.
    * ``contact_pos_depth``: world mid-surface contact point and the
      pre-solve penetration depth (larger = deeper), consumed by the
      best-K selection.

    Everything derives from pre-solve state, so the grouping and records are
    valid for all restitution iterations of the current substep.
    """
    tid = wp.tid()

    if tid >= contact_count[0]:
        return
    if restitution_contact_active[tid] == 0:
        return

    shape_a = contact_shape0[tid]
    shape_b = contact_shape1[tid]
    if shape_a == shape_b:
        return
    body_a = -1
    body_b = -1
    mat_nonzero = 0
    restitution = 0.0
    if shape_a >= 0:
        mat_nonzero += 1
        restitution += shape_material_restitution[shape_a]
        body_a = shape_body[shape_a]
    if shape_b >= 0:
        mat_nonzero += 1
        restitution += shape_material_restitution[shape_b]
        body_b = shape_body[shape_b]
    if mat_nonzero > 0:
        restitution /= float(mat_nonzero)
    if body_a == body_b or restitution < 0.0:
        return

    lo = wp.min(body_a, body_b)
    hi = wp.max(body_a, body_b)
    # sigma: the contact's normal sign for the lo body (normal points A -> B)
    sigma = 1.0
    if body_a == lo:
        sigma = -1.0

    X_lo = wp.transform_identity()
    X_hi = wp.transform_identity()
    m_inv_lo = 0.0
    m_inv_hi = 0.0
    I_inv_lo = wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    I_inv_hi = wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    com_lo = wp.vec3(0.0)
    com_hi = wp.vec3(0.0)
    qd_lo_pre = wp.spatial_vector()
    qd_hi_pre = wp.spatial_vector()
    gravity_magnitude = 0.0
    if lo >= 0:
        X_lo = body_q_pre_solve[lo]
        m_inv_lo = body_m_inv[lo]
        I_inv_lo = body_I_inv[lo]
        com_lo = body_com[lo]
        qd_lo_pre = body_qd_pre_solve[lo]
        gravity_magnitude = wp.max(gravity_magnitude, wp.length(gravity[body_world[lo]]))
    if hi >= 0:
        X_hi = body_q_pre_solve[hi]
        m_inv_hi = body_m_inv[hi]
        I_inv_hi = body_I_inv[hi]
        com_hi = body_com[hi]
        qd_hi_pre = body_qd_pre_solve[hi]
        gravity_magnitude = wp.max(gravity_magnitude, wp.length(gravity[body_world[hi]]))

    p_lo = contact_point1[tid]
    o_lo = contact_offset1[tid]
    p_hi = contact_point0[tid]
    o_hi = contact_offset0[tid]
    if body_a == lo:
        p_lo = contact_point0[tid]
        o_lo = contact_offset0[tid]
        p_hi = contact_point1[tid]
        o_hi = contact_offset1[tid]

    n = contact_normal[tid]
    r_lo = contact_surface_point(X_lo, p_lo, o_lo) - wp.transform_point(X_lo, com_lo)
    r_hi = contact_surface_point(X_hi, p_hi, o_hi) - wp.transform_point(X_hi, com_hi)
    axn_lo = wp.cross(r_lo, n)
    axn_hi = wp.cross(r_hi, n)

    inv_mass = 0.0
    rel_vel_old = 0.0
    if lo >= 0:
        q_lo = wp.transform_get_rotation(X_lo)
        rxn_lo = wp.quat_rotate_inv(q_lo, axn_lo)
        inv_mass += m_inv_lo + wp.dot(rxn_lo, I_inv_lo * rxn_lo)
        rel_vel_old += sigma * wp.dot(n, velocity_at_point(qd_lo_pre, r_lo))
    if hi >= 0:
        q_hi = wp.transform_get_rotation(X_hi)
        rxn_hi = wp.quat_rotate_inv(q_hi, axn_hi)
        inv_mass += m_inv_hi + wp.dot(rxn_hi, I_inv_hi * rxn_hi)
        rel_vel_old -= sigma * wp.dot(n, velocity_at_point(qd_hi_pre, r_hi))

    if inv_mass == 0.0:
        return
    impact_threshold = 2.0 * gravity_magnitude * dt
    # Use a wider dead band on the separating side. Small velocity reversals
    # across coupled resting manifolds otherwise feed positional jitter back
    # through the restitution pass and destabilize stacks.
    separating_threshold = 8.0 * impact_threshold
    if rel_vel_old >= -impact_threshold and rel_vel_old <= separating_threshold:
        return

    target = rel_vel_old
    sigma_record = 2.0 * sigma
    if rel_vel_old < -impact_threshold:
        target = -restitution * rel_vel_old
        sigma_record = sigma

    contact_n_K[tid] = wp.vec4(n[0], n[1], n[2], inv_mass)
    contact_axn_lo_target[tid] = wp.vec4(axn_lo[0], axn_lo[1], axn_lo[2], target)
    contact_axn_hi_sigma[tid] = wp.vec4(axn_hi[0], axn_hi[1], axn_hi[2], sigma_record)
    bx_lo = contact_surface_point(X_lo, p_lo, o_lo)
    bx_hi = contact_surface_point(X_hi, p_hi, o_hi)
    p_mid = 0.5 * (bx_lo + bx_hi)
    # pre-solve penetration depth along the A->B normal (larger = deeper)
    depth = sigma * wp.dot(n, bx_hi - bx_lo)
    contact_pos_depth[tid] = wp.vec4(p_mid[0], p_mid[1], p_mid[2], depth)

    key = wp.int64(lo + 1) * wp.int64(body_count + 1) + wp.int64(hi + 1)
    table_size = manifold_key.shape[0]
    slot = wp.int32(key % wp.int64(table_size))
    # linear probing; table capacity >= contact capacity so a slot always exists
    for _attempt in range(table_size):
        prev = wp.atomic_cas(manifold_key, slot, wp.int64(0), key)
        if prev == wp.int64(0) or prev == key:
            wp.atomic_add(manifold_total, slot, 1)
            # heads store index+1 so a zeroed table means empty (memset-cheap reset)
            contact_next[tid] = wp.atomic_exch(manifold_head, slot, tid + 1) - 1
            return
        slot += 1
        if slot == table_size:
            slot = 0


@wp.kernel
def select_manifold_contacts(
    manifold_key: wp.array[wp.int64],
    manifold_head: wp.array[wp.int32],
    manifold_total: wp.array[wp.int32],
    contact_next: wp.array[wp.int32],
    contact_pos_depth: wp.array[wp.vec4],
    contact_n_K: wp.array[wp.vec4],
    # outputs
    manifold_contact: wp.array[wp.int32],
    manifold_size: wp.array[wp.int32],
    contact_sel_score: wp.array[float],
):
    """Reduce each manifold chain to a bounded best-K contact subset.

    One thread per occupied hash slot. The deepest contact anchors the
    selection (ties break to the lower contact index); the remaining picks
    greedily maximize the minimum distance to the already-selected set in a
    combined metric ``|dp|^2 + w |dn|^2`` with ``w`` the squared patch radius
    about the anchor, approximating coverage of the contact Jacobian rows
    (position spread spans the torque rows, normal spread the force rows).
    Farthest-point selection with a per-contact running score
    (``contact_sel_score``; each contact belongs to exactly one manifold, so
    the scratch is race-free): O(K * N) chain walks. Every argmax breaks ties
    to the lower contact index, so the result is independent of chain
    (atomic append) order — deterministic. Manifolds with at most
    ``RESTITUTION_MANIFOLD_MAX_CONTACTS`` contacts keep every contact
    unchanged (fast path). Over-cap manifolds drop exact duplicates in
    position and normal — their max-min score is exactly zero, so they are
    never picked and the selected size can be below the cap. The
    velocity-level rigid solve is rank <= 6 per body pair, so a spread-K
    subset preserves the reachable impulse space.
    """
    tid = wp.tid()

    if manifold_key[tid] == wp.int64(0):
        return
    head = manifold_head[tid] - 1
    if head < 0:
        return

    # fast path: within-cap manifolds keep every contact; chain order is
    # irrelevant because the solve kernel sorts its indices canonically
    total = manifold_total[tid]
    if total <= _restitution_manifold_max:
        n_all = wp.int32(0)
        c = head
        while c >= 0:
            manifold_contact[tid * _restitution_manifold_max + n_all] = c
            n_all += 1
            c = contact_next[c]
        manifold_size[tid] = n_all
        return

    # pass 1: anchor = deepest contact (tie -> lower index)
    anchor = wp.int32(-1)
    best_depth = float(-1.0e30)
    c = head
    while c >= 0:
        depth = contact_pos_depth[c][3]
        if depth > best_depth or (depth == best_depth and (anchor < 0 or c < anchor)):
            best_depth = depth
            anchor = c
        c = contact_next[c]

    if anchor < 0:
        return

    # pass 2: squared patch radius about the anchor -> normal-term weight,
    # and initialize per-contact scores as distance-to-anchor
    d_a = contact_pos_depth[anchor]
    p_anchor = wp.vec3(d_a[0], d_a[1], d_a[2])
    d_an = contact_n_K[anchor]
    n_anchor = wp.vec3(d_an[0], d_an[1], d_an[2])
    r2 = float(0.0)
    c = head
    while c >= 0:
        d_c = contact_pos_depth[c]
        p_c = wp.vec3(d_c[0], d_c[1], d_c[2])
        r2 = wp.max(r2, wp.length_sq(p_c - p_anchor))
        c = contact_next[c]
    w = wp.max(r2, 1.0e-12)
    c = head
    while c >= 0:
        if c == anchor:
            contact_sel_score[c] = -1.0  # selected sentinel
        else:
            d_c = contact_pos_depth[c]
            p_c = wp.vec3(d_c[0], d_c[1], d_c[2])
            d_cn = contact_n_K[c]
            n_c = wp.vec3(d_cn[0], d_cn[1], d_cn[2])
            contact_sel_score[c] = wp.length_sq(p_c - p_anchor) + w * wp.length_sq(n_c - n_anchor)
        c = contact_next[c]

    sel = _restitution_vecNi()
    sel[0] = anchor
    n_sel = wp.int32(1)
    for _pick in range(_restitution_manifold_max - 1):
        # argmax of running min-distance score (tie -> lower index)
        best = wp.int32(-1)
        best_score = float(0.0)
        c = head
        while c >= 0:
            sc = contact_sel_score[c]
            if sc > best_score or (sc == best_score and sc > 0.0 and (best < 0 or c < best)):
                best_score = sc
                best = c
            c = contact_next[c]
        if best < 0:
            break  # nothing left that adds span (or chain exhausted)
        sel[n_sel] = best
        n_sel += 1
        contact_sel_score[best] = -1.0
        # relax remaining scores against the new pick
        d_b = contact_pos_depth[best]
        p_b = wp.vec3(d_b[0], d_b[1], d_b[2])
        d_bn = contact_n_K[best]
        n_b = wp.vec3(d_bn[0], d_bn[1], d_bn[2])
        c = head
        while c >= 0:
            if contact_sel_score[c] > 0.0:
                d_c = contact_pos_depth[c]
                p_c = wp.vec3(d_c[0], d_c[1], d_c[2])
                d_cn = contact_n_K[c]
                n_c = wp.vec3(d_cn[0], d_cn[1], d_cn[2])
                cand = wp.length_sq(p_c - p_b) + w * wp.length_sq(n_c - n_b)
                contact_sel_score[c] = wp.min(contact_sel_score[c], cand)
            c = contact_next[c]

    for j in range(n_sel):
        manifold_contact[tid * _restitution_manifold_max + j] = sel[j]
    manifold_size[tid] = n_sel


@wp.func
def _world_inv_inertia(q: wp.quat, I_inv: wp.mat33) -> wp.mat33:
    R = wp.quat_to_matrix(q)
    return R @ I_inv @ wp.transpose(R)


@wp.kernel
def solve_manifold_restitution(
    body_qd: wp.array[wp.spatial_vector],
    body_q_pre_solve: wp.array[wp.transform],
    body_m_inv: wp.array[float],
    body_I_inv: wp.array[wp.mat33],
    manifold_key: wp.array[wp.int64],
    manifold_size: wp.array[wp.int32],
    manifold_contact: wp.array[wp.int32],
    body_count: int,
    contact_n_K: wp.array[wp.vec4],
    contact_axn_lo_target: wp.array[wp.vec4],
    contact_axn_hi_sigma: wp.array[wp.vec4],
    inner_iterations: int,
    outer_iteration: int,
    # outputs
    deltas: wp.array[wp.spatial_vector],
    body_manifold_count: wp.array[float],
):
    """Per-manifold effective-mass restitution solve.

    One thread per occupied hash-table slot (canonical body pair) iterates the
    manifold's contacts Gauss-Seidel style against local working copies of the
    two bodies' velocities, using the per-contact records cached by
    :func:`build_restitution_manifolds` and an accumulated lower-bounded clamp
    on each contact impulse (PGS). On the first outer iteration only
    (``outer_iteration == 0``), a contact whose starting separating velocity
    already exceeds its restitution target gets a bounded negative allowance,
    ``(target - rel_vel_start) / k_eff``, attributing the over-separation to
    the contact's own positional impulse through its effective mass — so the
    velocity pass can reduce a positional over-separation down to the target
    and the authored coefficient is honored across the full ``[0, 1]`` range.
    Later outer iterations clamp at zero: after the first pass, above-target
    separation at a pair can be another manifold's restitution impulse, and a
    re-frozen bound would claw it back (adhesive; dissipates energy at
    ``e = 1``). Contacts starting at or below their target keep the plain
    non-negative clamp on every pass. The one-shot attribution ignores
    off-diagonal effective-mass coupling between contacts, so it bounds the
    allowance but does not strictly guarantee per-contact non-adhesion. The
    manifold's net velocity change is written to ``deltas`` with atomics and
    averaged per body by the caller over manifolds that fired
    (``body_manifold_count``). Restitution targets stay anchored to the
    pre-solve velocities throughout the substep.
    """
    tid = wp.tid()

    key = manifold_key[tid]
    if key == wp.int64(0):
        return

    num_contacts = wp.min(manifold_size[tid], _restitution_manifold_max)

    # gather + sort contact indices so Gauss-Seidel order (and float rounding)
    # is independent of the atomic append order
    con = _restitution_vecNi()
    for k in range(num_contacts):
        con[k] = manifold_contact[tid * _restitution_manifold_max + k]
    for i in range(1, num_contacts):
        v = con[i]
        j = i
        while j > 0:
            if con[j - 1] <= v:
                break
            con[j] = con[j - 1]
            j -= 1
        con[j] = v

    lo = wp.int32(key / wp.int64(body_count + 1)) - 1
    hi = wp.int32(key % wp.int64(body_count + 1)) - 1

    # per-manifold body data; zero response for a static side (lo == -1)
    m_inv_lo = 0.0
    m_inv_hi = 0.0
    I_inv_lo_world = wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    I_inv_hi_world = wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    v_lo0 = wp.vec3(0.0)
    w_lo0 = wp.vec3(0.0)
    v_hi0 = wp.vec3(0.0)
    w_hi0 = wp.vec3(0.0)
    if lo >= 0:
        m_inv_lo = body_m_inv[lo]
        I_inv_lo_world = _world_inv_inertia(wp.transform_get_rotation(body_q_pre_solve[lo]), body_I_inv[lo])
        qd = body_qd[lo]
        v_lo0 = wp.spatial_top(qd)
        w_lo0 = wp.spatial_bottom(qd)
    if hi >= 0:
        m_inv_hi = body_m_inv[hi]
        I_inv_hi_world = _world_inv_inertia(wp.transform_get_rotation(body_q_pre_solve[hi]), body_I_inv[hi])
        qd = body_qd[hi]
        v_hi0 = wp.spatial_top(qd)
        w_hi0 = wp.spatial_bottom(qd)
    v_lo = v_lo0
    w_lo = w_lo0
    v_hi = v_hi0
    w_hi = w_hi0

    # Per-contact impulse lower bound, frozen at the sweep-start working
    # velocities of the FIRST outer iteration only: the positional solve can
    # reconstruct a separating velocity above the restitution target, and the
    # velocity pass must be able to pull that overshoot back down (Eq. 34
    # drives to the target from either side). Bounding the accumulator at
    # (target - rel_vel_start) / k_eff instead of zero allows that reduction
    # and no more. Later outer iterations must NOT re-freeze the bound:
    # above-target separation there can be another manifold's restitution
    # impulse from an earlier pass, and clawing it back is adhesive (measured
    # 23% KE loss at e=1 for a body impacting two supports simultaneously).
    lambda_min = _restitution_vecNf()
    if outer_iteration == 0:
        for k in range(num_contacts):
            c = con[k]
            d_nK = contact_n_K[c]
            k_eff = d_nK[3]
            if k_eff == 0.0:
                continue
            n = wp.vec3(d_nK[0], d_nK[1], d_nK[2])
            d_lot = contact_axn_lo_target[c]
            axn_lo = wp.vec3(d_lot[0], d_lot[1], d_lot[2])
            d_his = contact_axn_hi_sigma[c]
            axn_hi = wp.vec3(d_his[0], d_his[1], d_his[2])
            sigma = wp.sign(d_his[3])
            rel_vel_start = sigma * (wp.dot(n, v_lo) + wp.dot(axn_lo, w_lo) - wp.dot(n, v_hi) - wp.dot(axn_hi, w_hi))
            lambda_min[k] = wp.min((d_lot[3] - rel_vel_start) / k_eff, 0.0)

    # Gauss-Seidel sweeps with accumulated clamp on local working copies
    lambda_acc = _restitution_vecNf()
    fired = wp.bool(False)
    for _it in range(inner_iterations):
        for k in range(num_contacts):
            c = con[k]
            d_nK = contact_n_K[c]
            k_eff = d_nK[3]
            if k_eff == 0.0:
                continue
            n = wp.vec3(d_nK[0], d_nK[1], d_nK[2])
            d_lot = contact_axn_lo_target[c]
            axn_lo = wp.vec3(d_lot[0], d_lot[1], d_lot[2])
            target = d_lot[3]
            d_his = contact_axn_hi_sigma[c]
            axn_hi = wp.vec3(d_his[0], d_his[1], d_his[2])
            separating_contact = wp.abs(d_his[3]) > 1.5
            if separating_contact and outer_iteration > 0:
                continue
            sigma = wp.sign(d_his[3])

            rel_vel_new = sigma * (wp.dot(n, v_lo) + wp.dot(axn_lo, w_lo) - wp.dot(n, v_hi) - wp.dot(axn_hi, w_hi))
            dv = (target - rel_vel_new) / k_eff
            new_acc = wp.max(lambda_acc[k] + dv, lambda_min[k])
            d_lambda = new_acc - lambda_acc[k]
            lambda_acc[k] = new_acc
            if d_lambda == 0.0:
                continue
            fired = True
            s = sigma * d_lambda
            v_lo += n * (m_inv_lo * s)
            w_lo += (I_inv_lo_world * axn_lo) * s
            v_hi -= n * (m_inv_hi * s)
            w_hi -= (I_inv_hi_world * axn_hi) * s

    if not fired:
        return

    # net manifold contribution; cross-manifold coupling is Jacobi (averaged
    # per body by the caller over manifolds that fired)
    if lo >= 0 and (m_inv_lo > 0.0 or wp.ddot(I_inv_lo_world, I_inv_lo_world) > 0.0):
        wp.atomic_add(deltas, lo, wp.spatial_vector(v_lo - v_lo0, w_lo - w_lo0))
        wp.atomic_add(body_manifold_count, lo, 1.0)
    if hi >= 0 and (m_inv_hi > 0.0 or wp.ddot(I_inv_hi_world, I_inv_hi_world) > 0.0):
        wp.atomic_add(deltas, hi, wp.spatial_vector(v_hi - v_hi0, w_hi - w_hi0))
        wp.atomic_add(body_manifold_count, hi, 1.0)


@wp.kernel
def apply_restitution_deltas(
    deltas: wp.array[wp.spatial_vector],
    body_manifold_count: wp.array[float],
    qd_out: wp.array[wp.spatial_vector],
):
    """Apply per-body restitution deltas averaged over fired manifolds.

    Consumes and clears both accumulators so the next restitution iteration
    starts from zero without extra zeroing launches (in-place path only; the
    ``requires_grad`` path uses fresh buffers instead).
    """
    tid = wp.tid()
    inv_weight = body_manifold_count[tid]
    if inv_weight <= 0.0:
        return
    qd_out[tid] = qd_out[tid] + deltas[tid] / inv_weight
    deltas[tid] = wp.spatial_vector()
    body_manifold_count[tid] = 0.0
