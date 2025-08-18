# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import warp as wp

from ...core import (
    quat_decompose,
    quat_twist,
)
from ...geometry import ParticleFlags
from ...geometry.kernels import triangle_closest_point_barycentric
from ...sim import (
    Contacts,
    Control,
    JointMode,
    JointType,
    Model,
    State,
)


@wp.kernel
def eval_springs(
    x: wp.array(dtype=wp.vec3),
    v: wp.array(dtype=wp.vec3),
    spring_indices: wp.array(dtype=int),
    spring_rest_lengths: wp.array(dtype=float),
    spring_stiffness: wp.array(dtype=float),
    spring_damping: wp.array(dtype=float),
    f: wp.array(dtype=wp.vec3),
):
    tid = wp.tid()

    i = spring_indices[tid * 2 + 0]
    j = spring_indices[tid * 2 + 1]

    if i == -1 or j == -1:
        return

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
    l_inv = 1.0 / l

    # normalized spring direction
    dir = xij * l_inv

    c = l - rest
    dcdt = wp.dot(dir, vij)

    # damping based on relative velocity
    fs = dir * (ke * c + kd * dcdt)

    wp.atomic_sub(f, i, fs)
    wp.atomic_add(f, j, fs)


@wp.kernel
def eval_triangles(
    x: wp.array(dtype=wp.vec3),
    v: wp.array(dtype=wp.vec3),
    indices: wp.array2d(dtype=int),
    pose: wp.array(dtype=wp.mat22),
    activation: wp.array(dtype=float),
    materials: wp.array2d(dtype=float),
    f: wp.array(dtype=wp.vec3),
):
    tid = wp.tid()

    k_mu = materials[tid, 0]
    k_lambda = materials[tid, 1]
    k_damp = materials[tid, 2]
    k_drag = materials[tid, 3]
    k_lift = materials[tid, 4]

    i = indices[tid, 0]
    j = indices[tid, 1]
    k = indices[tid, 2]

    x0 = x[i]  # point zero
    x1 = x[j]  # point one
    x2 = x[k]  # point two

    v0 = v[i]  # vel zero
    v1 = v[j]  # vel one
    v2 = v[k]  # vel two

    x10 = x1 - x0  # barycentric coordinates (centered at p)
    x20 = x2 - x0

    v10 = v1 - v0
    v20 = v2 - v0

    Dm = pose[tid]

    inv_rest_area = wp.determinant(Dm) * 2.0  # 1 / det(A) = det(A^-1)
    rest_area = 1.0 / inv_rest_area

    # scale stiffness coefficients to account for area
    k_mu = k_mu * rest_area
    k_lambda = k_lambda * rest_area
    k_damp = k_damp * rest_area

    # F = Xs*Xm^-1
    F1 = x10 * Dm[0, 0] + x20 * Dm[1, 0]
    F2 = x10 * Dm[0, 1] + x20 * Dm[1, 1]

    # dFdt = Vs*Xm^-1
    dFdt1 = v10 * Dm[0, 0] + v20 * Dm[1, 0]
    dFdt2 = v10 * Dm[0, 1] + v20 * Dm[1, 1]

    # deviatoric PK1 + damping term
    P1 = F1 * k_mu + dFdt1 * k_damp
    P2 = F2 * k_mu + dFdt2 * k_damp

    # -----------------------------
    # Neo-Hookean (with rest stability)

    # force = P*Dm'
    f1 = P1 * Dm[0, 0] + P2 * Dm[0, 1]
    f2 = P1 * Dm[1, 0] + P2 * Dm[1, 1]

    # -----------------------------
    # Area Preservation

    n = wp.cross(x10, x20)
    area = wp.length(n) * 0.5
    n = wp.normalize(n)

    # actuation
    act = activation[tid]

    # Apply area preservation only when k_lambda > 0
    if k_lambda > 0.0:
        alpha = 1.0 + k_mu / k_lambda

        # J-alpha
        c = area * inv_rest_area - alpha + act

        # dJdx
        dcdq = wp.cross(x20, n) * inv_rest_area * 0.5
        dcdr = wp.cross(n, x10) * inv_rest_area * 0.5

        f_area = k_lambda * c

        # -----------------------------
        # Area Damping

        dcdt = wp.dot(dcdq, v1) + wp.dot(dcdr, v2) - wp.dot(dcdq + dcdr, v0)
        f_damp = k_damp * dcdt

        f1 = f1 + dcdq * (f_area + f_damp)
        f2 = f2 + dcdr * (f_area + f_damp)

    f0 = f1 + f2

    # -----------------------------
    # Lift + Drag

    vmid = (v0 + v1 + v2) * 0.3333
    vdir = wp.normalize(vmid)

    f_drag = vmid * (k_drag * area * wp.abs(wp.dot(n, vmid)))
    f_lift = n * (k_lift * area * (wp.HALF_PI - wp.acos(wp.dot(n, vdir)))) * wp.dot(vmid, vmid)

    f0 = f0 - f_drag - f_lift
    f1 = f1 + f_drag + f_lift
    f2 = f2 + f_drag + f_lift

    # apply forces
    wp.atomic_add(f, i, f0)
    wp.atomic_sub(f, j, f1)
    wp.atomic_sub(f, k, f2)


@wp.kernel
def eval_triangles_contact(
    # idx : wp.array(dtype=int), # list of indices for colliding particles
    num_particles: int,  # size of particles
    x: wp.array(dtype=wp.vec3),
    v: wp.array(dtype=wp.vec3),
    indices: wp.array2d(dtype=int),
    materials: wp.array2d(dtype=float),
    particle_radius: wp.array(dtype=float),
    contact_stiffness: float,
    f: wp.array(dtype=wp.vec3),
):
    tid = wp.tid()
    face_no = tid // num_particles  # which face
    particle_no = tid % num_particles  # which particle

    # at the moment, just one particle
    pos = x[particle_no]

    i = indices[face_no, 0]
    j = indices[face_no, 1]
    k = indices[face_no, 2]

    if i == particle_no or j == particle_no or k == particle_no:
        return

    p = x[i]  # point zero
    q = x[j]  # point one
    r = x[k]  # point two

    bary = triangle_closest_point_barycentric(p, q, r, pos)
    closest = p * bary[0] + q * bary[1] + r * bary[2]

    diff = pos - closest
    dist = wp.length(diff)

    # early exit if no contact or degenerate case
    collision_radius = particle_radius[particle_no]
    if dist >= collision_radius or dist < 1e-6:
        return

    # contact normal (points from triangle to particle)
    n = diff / dist

    # penetration depth
    penetration_depth = collision_radius - dist

    # contact force
    fn = contact_stiffness * penetration_depth * n

    wp.atomic_add(f, particle_no, fn)
    wp.atomic_add(f, i, -fn * bary[0])
    wp.atomic_add(f, j, -fn * bary[1])
    wp.atomic_add(f, k, -fn * bary[2])


@wp.kernel
def eval_triangles_body_contacts(
    num_particles: int,  # number of particles (size of contact_point)
    x: wp.array(dtype=wp.vec3),  # position of particles
    v: wp.array(dtype=wp.vec3),
    indices: wp.array(dtype=int),  # triangle indices
    body_x: wp.array(dtype=wp.vec3),  # body body positions
    body_r: wp.array(dtype=wp.quat),
    body_v: wp.array(dtype=wp.vec3),
    body_w: wp.array(dtype=wp.vec3),
    contact_body: wp.array(dtype=int),
    contact_point: wp.array(dtype=wp.vec3),  # position of contact points relative to body
    contact_dist: wp.array(dtype=float),
    contact_mat: wp.array(dtype=int),
    materials: wp.array(dtype=float),
    #   body_f : wp.array(dtype=wp.vec3),
    #   body_t : wp.array(dtype=wp.vec3),
    tri_f: wp.array(dtype=wp.vec3),
):
    tid = wp.tid()

    face_no = tid // num_particles  # which face
    particle_no = tid % num_particles  # which particle

    # -----------------------
    # load body body point
    c_body = contact_body[particle_no]
    c_point = contact_point[particle_no]
    c_dist = contact_dist[particle_no]
    c_mat = contact_mat[particle_no]

    # hard coded surface parameter tensor layout (ke, kd, kf, mu)
    ke = materials[c_mat * 4 + 0]  # restitution coefficient
    kd = materials[c_mat * 4 + 1]  # damping coefficient
    kf = materials[c_mat * 4 + 2]  # friction coefficient
    mu = materials[c_mat * 4 + 3]  # coulomb friction

    x0 = body_x[c_body]  # position of colliding body
    r0 = body_r[c_body]  # orientation of colliding body

    v0 = body_v[c_body]
    w0 = body_w[c_body]

    # transform point to world space
    pos = x0 + wp.quat_rotate(r0, c_point)
    # use x0 as center, everything is offset from center of mass

    # moment arm
    r = pos - x0  # basically just c_point in the new coordinates
    rhat = wp.normalize(r)
    pos = pos + rhat * c_dist  # add on 'thickness' of shape, e.g.: radius of sphere/capsule

    # contact point velocity
    dpdt = v0 + wp.cross(w0, r)  # this is body velocity cross offset, so it's the velocity of the contact point.

    # -----------------------
    # load triangle
    i = indices[face_no * 3 + 0]
    j = indices[face_no * 3 + 1]
    k = indices[face_no * 3 + 2]

    p = x[i]  # point zero
    q = x[j]  # point one
    r = x[k]  # point two

    vp = v[i]  # vel zero
    vq = v[j]  # vel one
    vr = v[k]  # vel two

    bary = triangle_closest_point_barycentric(p, q, r, pos)
    closest = p * bary[0] + q * bary[1] + r * bary[2]

    diff = pos - closest  # vector from tri to point
    dist = wp.dot(diff, diff)  # squared distance
    n = wp.normalize(diff)  # points into the object
    c = wp.min(dist - 0.05, 0.0)  # 0 unless within 0.05 of surface
    # c = wp.leaky_min(wp.dot(n, x0)-0.01, 0.0, 0.0)
    # fn = n * c * 1e6    # points towards cloth (both n and c are negative)

    # wp.atomic_sub(tri_f, particle_no, fn)

    fn = c * ke  # normal force (restitution coefficient * how far inside for ground) (negative)

    vtri = vp * bary[0] + vq * bary[1] + vr * bary[2]  # bad approximation for centroid velocity
    vrel = vtri - dpdt

    vn = wp.dot(n, vrel)  # velocity component of body in negative normal direction
    vt = vrel - n * vn  # velocity component not in normal direction

    # contact damping
    fd = -wp.max(vn, 0.0) * kd * wp.step(c)  # again, negative, into the ground

    # # viscous friction
    # ft = vt*kf

    # Coulomb friction (box)
    lower = mu * (fn + fd)
    upper = -lower

    nx = wp.cross(n, wp.vec3(0.0, 0.0, 1.0))  # basis vectors for tangent
    nz = wp.cross(n, wp.vec3(1.0, 0.0, 0.0))

    vx = wp.clamp(wp.dot(nx * kf, vt), lower, upper)
    vz = wp.clamp(wp.dot(nz * kf, vt), lower, upper)

    ft = (nx * vx + nz * vz) * (-wp.step(c))  # wp.vec3(vx, 0.0, vz)*wp.step(c)

    # # Coulomb friction (smooth, but gradients are numerically unstable around |vt| = 0)
    # #ft = wp.normalize(vt)*wp.min(kf*wp.length(vt), -mu*c*ke)

    f_total = n * (fn + fd) + ft

    wp.atomic_add(tri_f, i, f_total * bary[0])
    wp.atomic_add(tri_f, j, f_total * bary[1])
    wp.atomic_add(tri_f, k, f_total * bary[2])


@wp.kernel
def eval_bending(
    x: wp.array(dtype=wp.vec3),
    v: wp.array(dtype=wp.vec3),
    indices: wp.array2d(dtype=int),
    rest: wp.array(dtype=float),
    bending_properties: wp.array2d(dtype=float),
    f: wp.array(dtype=wp.vec3),
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

    d1 = -n1_hat * e_length
    d2 = -n2_hat * e_length
    d3 = -n1_hat * wp.dot(x1 - x4, e_hat) - n2_hat * wp.dot(x2 - x4, e_hat)
    d4 = -n1_hat * wp.dot(x3 - x1, e_hat) - n2_hat * wp.dot(x3 - x2, e_hat)

    # elastic
    f_elastic = ke * (theta - rest_angle)

    # damping
    f_damp = kd * (wp.dot(d1, v1) + wp.dot(d2, v2) + wp.dot(d3, v3) + wp.dot(d4, v4))

    # total force, proportional to edge length
    f_total = -e_length * (f_elastic + f_damp)

    wp.atomic_add(f, i, d1 * f_total)
    wp.atomic_add(f, j, d2 * f_total)
    wp.atomic_add(f, k, d3 * f_total)
    wp.atomic_add(f, l, d4 * f_total)


@wp.kernel
def eval_tetrahedra(
    x: wp.array(dtype=wp.vec3),
    v: wp.array(dtype=wp.vec3),
    indices: wp.array2d(dtype=int),
    pose: wp.array(dtype=wp.mat33),
    activation: wp.array(dtype=float),
    materials: wp.array2d(dtype=float),
    f: wp.array(dtype=wp.vec3),
):
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

    x10 = x1 - x0
    x20 = x2 - x0
    x30 = x3 - x0

    v10 = v1 - v0
    v20 = v2 - v0
    v30 = v3 - v0

    Ds = wp.matrix_from_cols(x10, x20, x30)
    Dm = pose[tid]

    inv_rest_volume = wp.determinant(Dm) * 6.0
    rest_volume = 1.0 / inv_rest_volume

    alpha = 1.0 + k_mu / k_lambda - k_mu / (4.0 * k_lambda)

    # scale stiffness coefficients to account for area
    k_mu = k_mu * rest_volume
    k_lambda = k_lambda * rest_volume
    k_damp = k_damp * rest_volume

    # F = Xs*Xm^-1
    F = Ds * Dm
    dFdt = wp.matrix_from_cols(v10, v20, v30) * Dm

    col1 = wp.vec3(F[0, 0], F[1, 0], F[2, 0])
    col2 = wp.vec3(F[0, 1], F[1, 1], F[2, 1])
    col3 = wp.vec3(F[0, 2], F[1, 2], F[2, 2])

    # -----------------------------
    # Neo-Hookean (with rest stability [Smith et al 2018])

    Ic = wp.dot(col1, col1) + wp.dot(col2, col2) + wp.dot(col3, col3)

    # deviatoric part
    P = F * k_mu * (1.0 - 1.0 / (Ic + 1.0)) + dFdt * k_damp
    H = P * wp.transpose(Dm)

    f1 = wp.vec3(H[0, 0], H[1, 0], H[2, 0])
    f2 = wp.vec3(H[0, 1], H[1, 1], H[2, 1])
    f3 = wp.vec3(H[0, 2], H[1, 2], H[2, 2])

    # -----------------------------
    # C_sqrt

    # alpha = 1.0

    # r_s = wp.sqrt(wp.abs(dot(col1, col1) + dot(col2, col2) + dot(col3, col3) - 3.0))

    # f1 = wp.vec3()
    # f2 = wp.vec3()
    # f3 = wp.vec3()

    # if (r_s > 0.0):
    #     r_s_inv = 1.0/r_s

    #     C = r_s
    #     dCdx = F*wp.transpose(Dm)*r_s_inv*wp.sign(r_s)

    #     grad1 = vec3(dCdx[0,0], dCdx[1,0], dCdx[2,0])
    #     grad2 = vec3(dCdx[0,1], dCdx[1,1], dCdx[2,1])
    #     grad3 = vec3(dCdx[0,2], dCdx[1,2], dCdx[2,2])

    #     f1 = grad1*C*k_mu
    #     f2 = grad2*C*k_mu
    #     f3 = grad3*C*k_mu

    # -----------------------------
    # C_spherical

    # alpha = 1.0

    # r_s = wp.sqrt(dot(col1, col1) + dot(col2, col2) + dot(col3, col3))
    # r_s_inv = 1.0/r_s

    # C = r_s - wp.sqrt(3.0)
    # dCdx = F*wp.transpose(Dm)*r_s_inv

    # grad1 = vec3(dCdx[0,0], dCdx[1,0], dCdx[2,0])
    # grad2 = vec3(dCdx[0,1], dCdx[1,1], dCdx[2,1])
    # grad3 = vec3(dCdx[0,2], dCdx[1,2], dCdx[2,2])

    # f1 = grad1*C*k_mu
    # f2 = grad2*C*k_mu
    # f3 = grad3*C*k_mu

    # ----------------------------
    # C_D

    # alpha = 1.0

    # r_s = wp.sqrt(dot(col1, col1) + dot(col2, col2) + dot(col3, col3))

    # C = r_s*r_s - 3.0
    # dCdx = F*wp.transpose(Dm)*2.0

    # grad1 = vec3(dCdx[0,0], dCdx[1,0], dCdx[2,0])
    # grad2 = vec3(dCdx[0,1], dCdx[1,1], dCdx[2,1])
    # grad3 = vec3(dCdx[0,2], dCdx[1,2], dCdx[2,2])

    # f1 = grad1*C*k_mu
    # f2 = grad2*C*k_mu
    # f3 = grad3*C*k_mu

    # ----------------------------
    # Hookean

    # alpha = 1.0

    # I = wp.matrix_from_cols(wp.vec3(1.0, 0.0, 0.0),
    #                         wp.vec3(0.0, 1.0, 0.0),
    #                         wp.vec3(0.0, 0.0, 1.0))

    # P = (F + wp.transpose(F) + I*(0.0-2.0))*k_mu
    # H = P * wp.transpose(Dm)

    # f1 = wp.vec3(H[0, 0], H[1, 0], H[2, 0])
    # f2 = wp.vec3(H[0, 1], H[1, 1], H[2, 1])
    # f3 = wp.vec3(H[0, 2], H[1, 2], H[2, 2])

    # hydrostatic part
    J = wp.determinant(F)

    # print(J)
    s = inv_rest_volume / 6.0
    dJdx1 = wp.cross(x20, x30) * s
    dJdx2 = wp.cross(x30, x10) * s
    dJdx3 = wp.cross(x10, x20) * s

    f_volume = (J - alpha + act) * k_lambda
    f_damp = (wp.dot(dJdx1, v1) + wp.dot(dJdx2, v2) + wp.dot(dJdx3, v3)) * k_damp

    f_total = f_volume + f_damp

    f1 = f1 + dJdx1 * f_total
    f2 = f2 + dJdx2 * f_total
    f3 = f3 + dJdx3 * f_total
    f0 = -(f1 + f2 + f3)

    # apply forces
    wp.atomic_sub(f, i, f0)
    wp.atomic_sub(f, j, f1)
    wp.atomic_sub(f, k, f2)
    wp.atomic_sub(f, l, f3)


@wp.kernel
def eval_particle_ground_contacts(
    particle_x: wp.array(dtype=wp.vec3),
    particle_v: wp.array(dtype=wp.vec3),
    particle_radius: wp.array(dtype=float),
    particle_flags: wp.array(dtype=wp.int32),
    ke: float,
    kd: float,
    kf: float,
    mu: float,
    ground: wp.array(dtype=float),
    # outputs
    f: wp.array(dtype=wp.vec3),
):
    tid = wp.tid()
    if (particle_flags[tid] & ParticleFlags.ACTIVE) == 0:
        return

    x = particle_x[tid]
    v = particle_v[tid]
    radius = particle_radius[tid]

    n = wp.vec3(ground[0], ground[1], ground[2])
    c = wp.min(wp.dot(n, x) + ground[3] - radius, 0.0)

    vn = wp.dot(n, v)
    jn = c * ke

    if c >= 0.0:
        return

    jd = min(vn, 0.0) * kd

    # contact force
    fn = jn + jd

    # friction force
    vt = v - n * vn
    vs = wp.length(vt)

    if vs > 0.0:
        vt = vt / vs

    # Coulomb condition
    ft = wp.min(vs * kf, mu * wp.abs(fn))

    # total force
    f[tid] = f[tid] - n * fn - vt * ft


@wp.kernel
def eval_particle_contacts(
    particle_x: wp.array(dtype=wp.vec3),
    particle_v: wp.array(dtype=wp.vec3),
    body_q: wp.array(dtype=wp.transform),
    body_qd: wp.array(dtype=wp.spatial_vector),
    particle_radius: wp.array(dtype=float),
    particle_flags: wp.array(dtype=wp.int32),
    body_com: wp.array(dtype=wp.vec3),
    shape_body: wp.array(dtype=int),
    shape_material_ke: wp.array(dtype=float),
    shape_material_kd: wp.array(dtype=float),
    shape_material_kf: wp.array(dtype=float),
    shape_material_mu: wp.array(dtype=float),
    shape_material_ka: wp.array(dtype=float),
    particle_ke: float,
    particle_kd: float,
    particle_kf: float,
    particle_mu: float,
    particle_ka: float,
    contact_count: wp.array(dtype=int),
    contact_particle: wp.array(dtype=int),
    contact_shape: wp.array(dtype=int),
    contact_body_pos: wp.array(dtype=wp.vec3),
    contact_body_vel: wp.array(dtype=wp.vec3),
    contact_normal: wp.array(dtype=wp.vec3),
    contact_max: int,
    body_f_in_world_frame: bool,
    # outputs
    particle_f: wp.array(dtype=wp.vec3),
    body_f: wp.array(dtype=wp.spatial_vector),
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

    px = particle_x[particle_index]
    pv = particle_v[particle_index]

    X_wb = wp.transform_identity()
    X_com = wp.vec3()
    body_v_s = wp.spatial_vector()

    if body_index >= 0:
        X_wb = body_q[body_index]
        X_com = body_com[body_index]
        body_v_s = body_qd[body_index]

    # body position in world space
    bx = wp.transform_point(X_wb, contact_body_pos[tid])
    r = bx - wp.transform_point(X_wb, X_com)

    n = contact_normal[tid]
    c = wp.dot(n, px - bx) - particle_radius[particle_index]

    if c > particle_ka:
        return

    # take average material properties of shape and particle parameters
    ke = 0.5 * (particle_ke + shape_material_ke[shape_index])
    kd = 0.5 * (particle_kd + shape_material_kd[shape_index])
    kf = 0.5 * (particle_kf + shape_material_kf[shape_index])
    mu = 0.5 * (particle_mu + shape_material_mu[shape_index])

    body_w = wp.spatial_top(body_v_s)
    body_v = wp.spatial_bottom(body_v_s)

    # compute the body velocity at the particle position
    bv = body_v + wp.transform_vector(X_wb, contact_body_vel[tid])
    if body_f_in_world_frame:
        bv += wp.cross(body_w, bx)
    else:
        bv += wp.cross(body_w, r)

    # relative velocity
    v = pv - bv

    # decompose relative velocity
    vn = wp.dot(n, v)
    vt = v - n * vn

    # contact elastic
    fn = n * c * ke

    # contact damping
    fd = n * wp.min(vn, 0.0) * kd

    # viscous friction
    # ft = vt*kf

    # Coulomb friction (box)
    # lower = mu * c * ke
    # upper = -lower

    # vx = wp.clamp(wp.dot(wp.vec3(kf, 0.0, 0.0), vt), lower, upper)
    # vz = wp.clamp(wp.dot(wp.vec3(0.0, 0.0, kf), vt), lower, upper)

    # ft = wp.vec3(vx, 0.0, vz)

    # Coulomb friction (smooth, but gradients are numerically unstable around |vt| = 0)
    ft = wp.normalize(vt) * wp.min(kf * wp.length(vt), abs(mu * c * ke))

    f_total = fn + (fd + ft)

    wp.atomic_sub(particle_f, particle_index, f_total)

    if body_index >= 0:
        if body_f_in_world_frame:
            wp.atomic_sub(body_f, body_index, wp.spatial_vector(wp.cross(bx, f_total), f_total))
        else:
            wp.atomic_add(body_f, body_index, wp.spatial_vector(wp.cross(r, f_total), f_total))


@wp.kernel
def eval_rigid_contacts(
    body_q: wp.array(dtype=wp.transform),
    body_qd: wp.array(dtype=wp.spatial_vector),
    body_com: wp.array(dtype=wp.vec3),
    shape_material_ke: wp.array(dtype=float),
    shape_material_kd: wp.array(dtype=float),
    shape_material_kf: wp.array(dtype=float),
    shape_material_ka: wp.array(dtype=float),
    shape_material_mu: wp.array(dtype=float),
    shape_body: wp.array(dtype=int),
    contact_count: wp.array(dtype=int),
    contact_point0: wp.array(dtype=wp.vec3),
    contact_point1: wp.array(dtype=wp.vec3),
    contact_normal: wp.array(dtype=wp.vec3),
    contact_shape0: wp.array(dtype=int),
    contact_shape1: wp.array(dtype=int),
    contact_thickness0: wp.array(dtype=float),
    contact_thickness1: wp.array(dtype=float),
    force_in_world_frame: bool,
    friction_smoothing: float,
    # outputs
    body_f: wp.array(dtype=wp.spatial_vector),
):
    tid = wp.tid()

    count = contact_count[0]
    if tid >= count:
        return

    # retrieve contact thickness, compute average contact material properties
    ke = 0.0  # contact normal force stiffness
    kd = 0.0  # damping coefficient
    kf = 0.0  # friction force stiffness
    ka = 0.0  # adhesion distance
    mu = 0.0  # friction coefficient
    mat_nonzero = 0
    thickness_a = contact_thickness0[tid]
    thickness_b = contact_thickness1[tid]
    shape_a = contact_shape0[tid]
    shape_b = contact_shape1[tid]
    if shape_a == shape_b:
        return
    body_a = -1
    body_b = -1
    if shape_a >= 0:
        mat_nonzero += 1
        ke += shape_material_ke[shape_a]
        kd += shape_material_kd[shape_a]
        kf += shape_material_kf[shape_a]
        ka += shape_material_ka[shape_a]
        mu += shape_material_mu[shape_a]
        body_a = shape_body[shape_a]
    if shape_b >= 0:
        mat_nonzero += 1
        ke += shape_material_ke[shape_b]
        kd += shape_material_kd[shape_b]
        kf += shape_material_kf[shape_b]
        ka += shape_material_ka[shape_b]
        mu += shape_material_mu[shape_b]
        body_b = shape_body[shape_b]
    if mat_nonzero > 0:
        ke /= float(mat_nonzero)
        kd /= float(mat_nonzero)
        kf /= float(mat_nonzero)
        ka /= float(mat_nonzero)
        mu /= float(mat_nonzero)

    # contact normal in world space
    n = contact_normal[tid]
    bx_a = contact_point0[tid]
    bx_b = contact_point1[tid]
    r_a = wp.vec3(0.0)
    r_b = wp.vec3(0.0)
    if body_a >= 0:
        X_wb_a = body_q[body_a]
        X_com_a = body_com[body_a]
        bx_a = wp.transform_point(X_wb_a, bx_a) - thickness_a * n
        r_a = bx_a - wp.transform_point(X_wb_a, X_com_a)

    if body_b >= 0:
        X_wb_b = body_q[body_b]
        X_com_b = body_com[body_b]
        bx_b = wp.transform_point(X_wb_b, bx_b) + thickness_b * n
        r_b = bx_b - wp.transform_point(X_wb_b, X_com_b)

    d = wp.dot(n, bx_a - bx_b)

    if d >= ka:
        return

    # compute contact point velocity
    bv_a = wp.vec3(0.0)
    bv_b = wp.vec3(0.0)
    if body_a >= 0:
        body_v_s_a = body_qd[body_a]
        body_w_a = wp.spatial_top(body_v_s_a)
        body_v_a = wp.spatial_bottom(body_v_s_a)
        if force_in_world_frame:
            bv_a = body_v_a + wp.cross(body_w_a, bx_a)
        else:
            bv_a = body_v_a + wp.cross(body_w_a, r_a)

    if body_b >= 0:
        body_v_s_b = body_qd[body_b]
        body_w_b = wp.spatial_top(body_v_s_b)
        body_v_b = wp.spatial_bottom(body_v_s_b)
        if force_in_world_frame:
            bv_b = body_v_b + wp.cross(body_w_b, bx_b)
        else:
            bv_b = body_v_b + wp.cross(body_w_b, r_b)

    # relative velocity
    v = bv_a - bv_b

    # print(v)

    # decompose relative velocity
    vn = wp.dot(n, v)
    vt = v - n * vn

    # contact elastic
    fn = d * ke

    # contact damping
    fd = wp.min(vn, 0.0) * kd * wp.step(d)

    # viscous friction
    # ft = vt*kf

    # Coulomb friction (box)
    # lower = mu * d * ke
    # upper = -lower

    # vx = wp.clamp(wp.dot(wp.vec3(kf, 0.0, 0.0), vt), lower, upper)
    # vz = wp.clamp(wp.dot(wp.vec3(0.0, 0.0, kf), vt), lower, upper)

    # ft = wp.vec3(vx, 0.0, vz)

    # Coulomb friction (smooth, but gradients are numerically unstable around |vt| = 0)
    ft = wp.vec3(0.0)
    if d < 0.0:
        # use a smooth vector norm to avoid gradient instability at/around zero velocity
        vs = wp.norm_huber(vt, delta=friction_smoothing)
        if vs > 0.0:
            fr = vt / vs
            ft = fr * wp.min(kf * vs, -mu * (fn + fd))

    f_total = n * (fn + fd) + ft
    # f_total = n * (fn + fd)
    # f_total = n * fn

    if body_a >= 0:
        if force_in_world_frame:
            wp.atomic_add(body_f, body_a, wp.spatial_vector(wp.cross(bx_a, f_total), f_total))
        else:
            wp.atomic_sub(body_f, body_a, wp.spatial_vector(wp.cross(r_a, f_total), f_total))

    if body_b >= 0:
        if force_in_world_frame:
            wp.atomic_sub(body_f, body_b, wp.spatial_vector(wp.cross(bx_b, f_total), f_total))
        else:
            wp.atomic_add(body_f, body_b, wp.spatial_vector(wp.cross(r_b, f_total), f_total))


@wp.func
def eval_joint_force(
    q: float,
    qd: float,
    act: float,
    target_ke: float,
    target_kd: float,
    limit_lower: float,
    limit_upper: float,
    limit_ke: float,
    limit_kd: float,
    mode: wp.int32,
) -> float:
    """Joint force evaluation for a single degree of freedom."""

    limit_f = 0.0
    damping_f = 0.0
    target_f = 0.0

    if mode == JointMode.TARGET_POSITION:
        target_f = target_ke * (act - q) - target_kd * qd
    elif mode == JointMode.TARGET_VELOCITY:
        target_f = target_ke * (act - qd)

    # compute limit forces, damping only active when limit is violated
    if q < limit_lower:
        limit_f = limit_ke * (limit_lower - q)
        damping_f = -limit_kd * qd
        if mode == JointMode.TARGET_VELOCITY:
            target_f = 0.0  # override target force when limit is violated
    elif q > limit_upper:
        limit_f = limit_ke * (limit_upper - q)
        damping_f = -limit_kd * qd
        if mode == JointMode.TARGET_VELOCITY:
            target_f = 0.0  # override target force when limit is violated

    return limit_f + damping_f + target_f


@wp.kernel
def eval_body_joints(
    body_q: wp.array(dtype=wp.transform),
    body_qd: wp.array(dtype=wp.spatial_vector),
    body_com: wp.array(dtype=wp.vec3),
    joint_qd_start: wp.array(dtype=int),
    joint_type: wp.array(dtype=int),
    joint_enabled: wp.array(dtype=int),
    joint_child: wp.array(dtype=int),
    joint_parent: wp.array(dtype=int),
    joint_X_p: wp.array(dtype=wp.transform),
    joint_X_c: wp.array(dtype=wp.transform),
    joint_axis: wp.array(dtype=wp.vec3),
    joint_dof_dim: wp.array(dtype=int, ndim=2),
    joint_dof_mode: wp.array(dtype=int),
    joint_f: wp.array(dtype=float),
    joint_target: wp.array(dtype=float),
    joint_target_ke: wp.array(dtype=float),
    joint_target_kd: wp.array(dtype=float),
    joint_limit_lower: wp.array(dtype=float),
    joint_limit_upper: wp.array(dtype=float),
    joint_limit_ke: wp.array(dtype=float),
    joint_limit_kd: wp.array(dtype=float),
    joint_attach_ke: float,
    joint_attach_kd: float,
    body_f: wp.array(dtype=wp.spatial_vector),
):
    tid = wp.tid()
    type = joint_type[tid]

    c_child = joint_child[tid]
    c_parent = joint_parent[tid]

    if joint_enabled[tid] == 0:
        return

    qd_start = joint_qd_start[tid]
    if type == JointType.FREE or type == JointType.DISTANCE:
        wrench = wp.spatial_vector(
            joint_f[qd_start + 0],
            joint_f[qd_start + 1],
            joint_f[qd_start + 2],
            joint_f[qd_start + 3],
            joint_f[qd_start + 4],
            joint_f[qd_start + 5],
        )

        wp.atomic_add(body_f, c_child, wrench)
        return

    X_pj = joint_X_p[tid]
    X_cj = joint_X_c[tid]

    X_wp = X_pj
    r_p = wp.vec3()
    w_p = wp.vec3()
    v_p = wp.vec3()

    # parent transform and moment arm
    if c_parent >= 0:
        X_wp = body_q[c_parent] * X_wp
        r_p = wp.transform_get_translation(X_wp) - wp.transform_point(body_q[c_parent], body_com[c_parent])

        twist_p = body_qd[c_parent]

        w_p = wp.spatial_top(twist_p)
        v_p = wp.spatial_bottom(twist_p) + wp.cross(w_p, r_p)

    # child transform and moment arm
    X_wc = body_q[c_child] * X_cj
    r_c = wp.transform_get_translation(X_wc) - wp.transform_point(body_q[c_child], body_com[c_child])

    twist_c = body_qd[c_child]

    w_c = wp.spatial_top(twist_c)
    v_c = wp.spatial_bottom(twist_c) + wp.cross(w_c, r_c)

    lin_axis_count = joint_dof_dim[tid, 0]
    ang_axis_count = joint_dof_dim[tid, 1]

    x_p = wp.transform_get_translation(X_wp)
    x_c = wp.transform_get_translation(X_wc)

    q_p = wp.transform_get_rotation(X_wp)
    q_c = wp.transform_get_rotation(X_wc)

    # translational error
    x_err = x_c - x_p
    r_err = wp.quat_inverse(q_p) * q_c
    v_err = v_c - v_p
    w_err = w_c - w_p

    # total force/torque on the parent
    t_total = wp.vec3()
    f_total = wp.vec3()

    # reduce angular damping stiffness for stability
    angular_damping_scale = 0.01

    if type == JointType.FIXED:
        ang_err = wp.normalize(wp.vec3(r_err[0], r_err[1], r_err[2])) * wp.acos(r_err[3]) * 2.0

        f_total += x_err * joint_attach_ke + v_err * joint_attach_kd
        t_total += (
            wp.transform_vector(X_wp, ang_err) * joint_attach_ke + w_err * joint_attach_kd * angular_damping_scale
        )

    if type == JointType.PRISMATIC:
        axis = joint_axis[qd_start]

        # world space joint axis
        axis_p = wp.transform_vector(X_wp, axis)

        # evaluate joint coordinates
        q = wp.dot(x_err, axis_p)
        qd = wp.dot(v_err, axis_p)
        act = joint_target[qd_start]

        f_total = axis_p * (
            -joint_f[qd_start]
            - eval_joint_force(
                q,
                qd,
                act,
                joint_target_ke[qd_start],
                joint_target_kd[qd_start],
                joint_limit_lower[qd_start],
                joint_limit_upper[qd_start],
                joint_limit_ke[qd_start],
                joint_limit_kd[qd_start],
                joint_dof_mode[qd_start],
            )
        )

        # attachment dynamics
        ang_err = wp.normalize(wp.vec3(r_err[0], r_err[1], r_err[2])) * wp.acos(r_err[3]) * 2.0

        # project off any displacement along the joint axis
        f_total += (x_err - q * axis_p) * joint_attach_ke + (v_err - qd * axis_p) * joint_attach_kd
        t_total += (
            wp.transform_vector(X_wp, ang_err) * joint_attach_ke + w_err * joint_attach_kd * angular_damping_scale
        )

    if type == JointType.REVOLUTE:
        axis = joint_axis[qd_start]

        axis_p = wp.transform_vector(X_wp, axis)
        axis_c = wp.transform_vector(X_wc, axis)

        # swing twist decomposition
        twist = quat_twist(axis, r_err)

        q = wp.acos(twist[3]) * 2.0 * wp.sign(wp.dot(axis, wp.vec3(twist[0], twist[1], twist[2])))
        qd = wp.dot(w_err, axis_p)
        act = joint_target[qd_start]

        t_total = axis_p * (
            -joint_f[qd_start]
            - eval_joint_force(
                q,
                qd,
                act,
                joint_target_ke[qd_start],
                joint_target_kd[qd_start],
                joint_limit_lower[qd_start],
                joint_limit_upper[qd_start],
                joint_limit_ke[qd_start],
                joint_limit_kd[qd_start],
                joint_dof_mode[qd_start],
            )
        )

        # attachment dynamics
        swing_err = wp.cross(axis_p, axis_c)

        f_total += x_err * joint_attach_ke + v_err * joint_attach_kd
        t_total += swing_err * joint_attach_ke + (w_err - qd * axis_p) * joint_attach_kd * angular_damping_scale

    if type == JointType.BALL:
        ang_err = wp.normalize(wp.vec3(r_err[0], r_err[1], r_err[2])) * wp.acos(r_err[3]) * 2.0

        # TODO joint limits
        # TODO expose target_kd or target_ke for ball joints
        # t_total += target_kd * w_err + target_ke * wp.transform_vector(X_wp, ang_err)
        f_total += x_err * joint_attach_ke + v_err * joint_attach_kd
        t_total += wp.vec3(-joint_f[qd_start], -joint_f[qd_start + 1], -joint_f[qd_start + 2])

    if type == JointType.D6:
        pos = wp.vec3(0.0)
        vel = wp.vec3(0.0)
        if lin_axis_count >= 1:
            axis_0 = wp.transform_vector(X_wp, joint_axis[qd_start + 0])
            q0 = wp.dot(x_err, axis_0)
            qd0 = wp.dot(v_err, axis_0)

            f_total += axis_0 * (
                -joint_f[qd_start]
                - eval_joint_force(
                    q0,
                    qd0,
                    joint_target[qd_start + 0],
                    joint_target_ke[qd_start + 0],
                    joint_target_kd[qd_start + 0],
                    joint_limit_lower[qd_start + 0],
                    joint_limit_upper[qd_start + 0],
                    joint_limit_ke[qd_start + 0],
                    joint_limit_kd[qd_start + 0],
                    joint_dof_mode[qd_start + 0],
                )
            )

            pos += q0 * axis_0
            vel += qd0 * axis_0

        if lin_axis_count >= 2:
            axis_1 = wp.transform_vector(X_wp, joint_axis[qd_start + 1])
            q1 = wp.dot(x_err, axis_1)
            qd1 = wp.dot(v_err, axis_1)

            f_total += axis_1 * (
                -joint_f[qd_start + 1]
                - eval_joint_force(
                    q1,
                    qd1,
                    joint_target[qd_start + 1],
                    joint_target_ke[qd_start + 1],
                    joint_target_kd[qd_start + 1],
                    joint_limit_lower[qd_start + 1],
                    joint_limit_upper[qd_start + 1],
                    joint_limit_ke[qd_start + 1],
                    joint_limit_kd[qd_start + 1],
                    joint_dof_mode[qd_start + 1],
                )
            )

            pos += q1 * axis_1
            vel += qd1 * axis_1

        if lin_axis_count == 3:
            axis_2 = wp.transform_vector(X_wp, joint_axis[qd_start + 2])
            q2 = wp.dot(x_err, axis_2)
            qd2 = wp.dot(v_err, axis_2)

            f_total += axis_2 * (
                -joint_f[qd_start + 2]
                - eval_joint_force(
                    q2,
                    qd2,
                    joint_target[qd_start + 2],
                    joint_target_ke[qd_start + 2],
                    joint_target_kd[qd_start + 2],
                    joint_limit_lower[qd_start + 2],
                    joint_limit_upper[qd_start + 2],
                    joint_limit_ke[qd_start + 2],
                    joint_limit_kd[qd_start + 2],
                    joint_dof_mode[qd_start + 2],
                )
            )

            pos += q2 * axis_2
            vel += qd2 * axis_2

        f_total += (x_err - pos) * joint_attach_ke + (v_err - vel) * joint_attach_kd

        if ang_axis_count == 0:
            ang_err = wp.normalize(wp.vec3(r_err[0], r_err[1], r_err[2])) * wp.acos(r_err[3]) * 2.0
            t_total += (
                wp.transform_vector(X_wp, ang_err) * joint_attach_ke + w_err * joint_attach_kd * angular_damping_scale
            )

        i_0 = lin_axis_count + qd_start + 0
        i_1 = lin_axis_count + qd_start + 1
        i_2 = lin_axis_count + qd_start + 2
        qdi_start = qd_start + lin_axis_count

        if ang_axis_count == 1:
            axis = joint_axis[i_0]

            axis_p = wp.transform_vector(X_wp, axis)
            axis_c = wp.transform_vector(X_wc, axis)

            # swing twist decomposition
            twist = quat_twist(axis, r_err)

            q = wp.acos(twist[3]) * 2.0 * wp.sign(wp.dot(axis, wp.vec3(twist[0], twist[1], twist[2])))
            qd = wp.dot(w_err, axis_p)

            t_total = axis_p * (
                -joint_f[qdi_start]
                - eval_joint_force(
                    q,
                    qd,
                    joint_target[i_0],
                    joint_target_ke[i_0],
                    joint_target_kd[i_0],
                    joint_limit_lower[i_0],
                    joint_limit_upper[i_0],
                    joint_limit_ke[i_0],
                    joint_limit_kd[i_0],
                    joint_dof_mode[i_0],
                )
            )

            # attachment dynamics
            swing_err = wp.cross(axis_p, axis_c)

            t_total += swing_err * joint_attach_ke + (w_err - qd * axis_p) * joint_attach_kd * angular_damping_scale

        if ang_axis_count == 2:
            q_pc = wp.quat_inverse(q_p) * q_c

            # decompose to a compound rotation each axis
            angles = quat_decompose(q_pc)

            orig_axis_0 = joint_axis[i_0]
            orig_axis_1 = joint_axis[i_1]
            orig_axis_2 = wp.cross(orig_axis_0, orig_axis_1)

            # reconstruct rotation axes
            axis_0 = orig_axis_0
            q_0 = wp.quat_from_axis_angle(axis_0, angles[0])

            axis_1 = wp.quat_rotate(q_0, orig_axis_1)
            q_1 = wp.quat_from_axis_angle(axis_1, angles[1])

            axis_2 = wp.quat_rotate(q_1 * q_0, orig_axis_2)

            axis_0 = wp.transform_vector(X_wp, axis_0)
            axis_1 = wp.transform_vector(X_wp, axis_1)
            axis_2 = wp.transform_vector(X_wp, axis_2)

            # joint dynamics

            t_total += axis_0 * (
                -joint_f[qdi_start]
                - eval_joint_force(
                    angles[0],
                    wp.dot(axis_0, w_err),
                    joint_target[i_0],
                    joint_target_ke[i_0],
                    joint_target_kd[i_0],
                    joint_limit_lower[i_0],
                    joint_limit_upper[i_0],
                    joint_limit_ke[i_0],
                    joint_limit_kd[i_0],
                    joint_dof_mode[i_0],
                )
            )
            t_total += axis_1 * (
                -joint_f[qdi_start + 1]
                - eval_joint_force(
                    angles[1],
                    wp.dot(axis_1, w_err),
                    joint_target[i_1],
                    joint_target_ke[i_1],
                    joint_target_kd[i_1],
                    joint_limit_lower[i_1],
                    joint_limit_upper[i_1],
                    joint_limit_ke[i_1],
                    joint_limit_kd[i_1],
                    joint_dof_mode[i_1],
                )
            )

            # last axis (fixed)
            t_total += axis_2 * -eval_joint_force(
                angles[2],
                wp.dot(axis_2, w_err),
                0.0,
                joint_attach_ke,
                joint_attach_kd * angular_damping_scale,
                0.0,
                0.0,
                0.0,
                0.0,
                JointMode.NONE,
            )

        if ang_axis_count == 3:
            q_pc = wp.quat_inverse(q_p) * q_c

            # decompose to a compound rotation each axis
            angles = quat_decompose(q_pc)

            orig_axis_0 = joint_axis[i_0]
            orig_axis_1 = joint_axis[i_1]
            orig_axis_2 = joint_axis[i_2]

            # reconstruct rotation axes
            axis_0 = orig_axis_0
            q_0 = wp.quat_from_axis_angle(axis_0, angles[0])

            axis_1 = wp.quat_rotate(q_0, orig_axis_1)
            q_1 = wp.quat_from_axis_angle(axis_1, angles[1])

            axis_2 = wp.quat_rotate(q_1 * q_0, orig_axis_2)

            axis_0 = wp.transform_vector(X_wp, axis_0)
            axis_1 = wp.transform_vector(X_wp, axis_1)
            axis_2 = wp.transform_vector(X_wp, axis_2)

            t_total += axis_0 * (
                -joint_f[qdi_start]
                - eval_joint_force(
                    angles[0],
                    wp.dot(axis_0, w_err),
                    joint_target[i_0],
                    joint_target_ke[i_0],
                    joint_target_kd[i_0],
                    joint_limit_lower[i_0],
                    joint_limit_upper[i_0],
                    joint_limit_ke[i_0],
                    joint_limit_kd[i_0],
                    joint_dof_mode[i_0],
                )
            )
            t_total += axis_1 * (
                -joint_f[qdi_start + 1]
                - eval_joint_force(
                    angles[1],
                    wp.dot(axis_1, w_err),
                    joint_target[i_1],
                    joint_target_ke[i_1],
                    joint_target_kd[i_1],
                    joint_limit_lower[i_1],
                    joint_limit_upper[i_1],
                    joint_limit_ke[i_1],
                    joint_limit_kd[i_1],
                    joint_dof_mode[i_1],
                )
            )
            t_total += axis_2 * (
                -joint_f[qdi_start + 2]
                - eval_joint_force(
                    angles[2],
                    wp.dot(axis_2, w_err),
                    joint_target[i_2],
                    joint_target_ke[i_2],
                    joint_target_kd[i_2],
                    joint_limit_lower[i_2],
                    joint_limit_upper[i_2],
                    joint_limit_ke[i_2],
                    joint_limit_kd[i_2],
                    joint_dof_mode[i_2],
                )
            )

    # write forces
    if c_parent >= 0:
        wp.atomic_add(body_f, c_parent, wp.spatial_vector(t_total + wp.cross(r_p, f_total), f_total))

    wp.atomic_sub(body_f, c_child, wp.spatial_vector(t_total + wp.cross(r_c, f_total), f_total))


@wp.func
def compute_muscle_force(
    i: int,
    body_X_s: wp.array(dtype=wp.transform),
    body_v_s: wp.array(dtype=wp.spatial_vector),
    body_com: wp.array(dtype=wp.vec3),
    muscle_links: wp.array(dtype=int),
    muscle_points: wp.array(dtype=wp.vec3),
    muscle_activation: float,
    body_f_s: wp.array(dtype=wp.spatial_vector),
):
    link_0 = muscle_links[i]
    link_1 = muscle_links[i + 1]

    if link_0 == link_1:
        return 0

    r_0 = muscle_points[i]
    r_1 = muscle_points[i + 1]

    xform_0 = body_X_s[link_0]
    xform_1 = body_X_s[link_1]

    pos_0 = wp.transform_point(xform_0, r_0 - body_com[link_0])
    pos_1 = wp.transform_point(xform_1, r_1 - body_com[link_1])

    n = wp.normalize(pos_1 - pos_0)

    # todo: add passive elastic and viscosity terms
    f = n * muscle_activation

    wp.atomic_sub(body_f_s, link_0, wp.spatial_vector(f, wp.cross(pos_0, f)))
    wp.atomic_add(body_f_s, link_1, wp.spatial_vector(f, wp.cross(pos_1, f)))


@wp.kernel
def eval_muscles(
    body_X_s: wp.array(dtype=wp.transform),
    body_v_s: wp.array(dtype=wp.spatial_vector),
    body_com: wp.array(dtype=wp.vec3),
    muscle_start: wp.array(dtype=int),
    muscle_params: wp.array(dtype=float),
    muscle_links: wp.array(dtype=int),
    muscle_points: wp.array(dtype=wp.vec3),
    muscle_activation: wp.array(dtype=float),
    # output
    body_f_s: wp.array(dtype=wp.spatial_vector),
):
    tid = wp.tid()

    m_start = muscle_start[tid]
    m_end = muscle_start[tid + 1] - 1

    activation = muscle_activation[tid]

    for i in range(m_start, m_end):
        compute_muscle_force(i, body_X_s, body_v_s, body_com, muscle_links, muscle_points, activation, body_f_s)


def eval_spring_forces(model: Model, state: State, particle_f: wp.array):
    if model.spring_count:
        wp.launch(
            kernel=eval_springs,
            dim=model.spring_count,
            inputs=[
                state.particle_q,
                state.particle_qd,
                model.spring_indices,
                model.spring_rest_length,
                model.spring_stiffness,
                model.spring_damping,
            ],
            outputs=[particle_f],
            device=model.device,
        )


def eval_triangle_forces(model: Model, state: State, control: Control, particle_f: wp.array):
    if model.tri_count:
        wp.launch(
            kernel=eval_triangles,
            dim=model.tri_count,
            inputs=[
                state.particle_q,
                state.particle_qd,
                model.tri_indices,
                model.tri_poses,
                control.tri_activations,
                model.tri_materials,
            ],
            outputs=[particle_f],
            device=model.device,
        )


def eval_triangle_contact_forces(model: Model, state: State, particle_f: wp.array):
    if model.tri_count and model.particle_count:
        wp.launch(
            kernel=eval_triangles_contact,
            dim=model.tri_count * model.particle_count,
            inputs=[
                model.particle_count,
                state.particle_q,
                state.particle_qd,
                model.tri_indices,
                model.tri_materials,
                model.particle_radius,
                model.soft_contact_ke,
            ],
            outputs=[particle_f],
            device=model.device,
        )


def eval_bending_forces(model: Model, state: State, particle_f: wp.array):
    if model.edge_count:
        wp.launch(
            kernel=eval_bending,
            dim=model.edge_count,
            inputs=[
                state.particle_q,
                state.particle_qd,
                model.edge_indices,
                model.edge_rest_angle,
                model.edge_bending_properties,
            ],
            outputs=[particle_f],
            device=model.device,
        )


def eval_tetrahedral_forces(model: Model, state: State, control: Control, particle_f: wp.array):
    if model.tet_count:
        wp.launch(
            kernel=eval_tetrahedra,
            dim=model.tet_count,
            inputs=[
                state.particle_q,
                state.particle_qd,
                model.tet_indices,
                model.tet_poses,
                control.tet_activations,
                model.tet_materials,
            ],
            outputs=[particle_f],
            device=model.device,
        )


def eval_body_contact_forces(
    model: Model,
    state: State,
    contacts: Contacts | None,
    friction_smoothing: float = 1.0,
    force_in_world_frame: bool = False,
    body_f_out: wp.array | None = None,
):
    if contacts is not None and contacts.rigid_contact_max:
        if body_f_out is None:
            body_f_out = state.body_f
        wp.launch(
            kernel=eval_rigid_contacts,
            dim=contacts.rigid_contact_max,
            inputs=[
                state.body_q,
                state.body_qd,
                model.body_com,
                model.shape_material_ke,
                model.shape_material_kd,
                model.shape_material_kf,
                model.shape_material_ka,
                model.shape_material_mu,
                model.shape_body,
                contacts.rigid_contact_count,
                contacts.rigid_contact_point0,
                contacts.rigid_contact_point1,
                contacts.rigid_contact_normal,
                contacts.rigid_contact_shape0,
                contacts.rigid_contact_shape1,
                contacts.rigid_contact_thickness0,
                contacts.rigid_contact_thickness1,
                force_in_world_frame,
                friction_smoothing,
            ],
            outputs=[body_f_out],
            device=model.device,
        )


def eval_body_joint_forces(
    model: Model, state: State, control: Control, body_f: wp.array, joint_attach_ke: float, joint_attach_kd: float
):
    if model.joint_count:
        wp.launch(
            kernel=eval_body_joints,
            dim=model.joint_count,
            inputs=[
                state.body_q,
                state.body_qd,
                model.body_com,
                model.joint_qd_start,
                model.joint_type,
                model.joint_enabled,
                model.joint_child,
                model.joint_parent,
                model.joint_X_p,
                model.joint_X_c,
                model.joint_axis,
                model.joint_dof_dim,
                model.joint_dof_mode,
                control.joint_f,
                control.joint_target,
                model.joint_target_ke,
                model.joint_target_kd,
                model.joint_limit_lower,
                model.joint_limit_upper,
                model.joint_limit_ke,
                model.joint_limit_kd,
                joint_attach_ke,
                joint_attach_kd,
            ],
            outputs=[body_f],
            device=model.device,
        )


def eval_particle_body_contact_forces(
    model: Model,
    state: State,
    contacts: Contacts | None,
    particle_f: wp.array,
    body_f: wp.array,
    body_f_in_world_frame: bool = False,
):
    if contacts is not None and contacts.soft_contact_max:
        wp.launch(
            kernel=eval_particle_contacts,
            dim=contacts.soft_contact_max,
            inputs=[
                state.particle_q,
                state.particle_qd,
                state.body_q,
                state.body_qd,
                model.particle_radius,
                model.particle_flags,
                model.body_com,
                model.shape_body,
                model.shape_material_ke,
                model.shape_material_kd,
                model.shape_material_kf,
                model.shape_material_mu,
                model.shape_material_ka,
                model.soft_contact_ke,
                model.soft_contact_kd,
                model.soft_contact_kf,
                model.soft_contact_mu,
                model.particle_adhesion,
                contacts.soft_contact_count,
                contacts.soft_contact_particle,
                contacts.soft_contact_shape,
                contacts.soft_contact_body_pos,
                contacts.soft_contact_body_vel,
                contacts.soft_contact_normal,
                contacts.soft_contact_max,
                body_f_in_world_frame,
            ],
            # outputs
            outputs=[particle_f, body_f],
            device=model.device,
        )


def eval_muscle_forces(model: Model, state: State, control: Control, body_f: wp.array):
    if model.muscle_count:
        wp.launch(
            kernel=eval_muscles,
            dim=model.muscle_count,
            inputs=[
                state.body_q,
                state.body_qd,
                model.body_com,
                model.muscle_start,
                model.muscle_params,
                model.muscle_bodies,
                model.muscle_points,
                control.muscle_activations,
            ],
            outputs=[body_f],
            device=model.device,
        )
