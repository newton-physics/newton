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

import warp as wp

from ...geometry import ParticleFlags
from ...geometry.kernels import triangle_closest_point_barycentric
from ...sim import Contacts, Model, State


@wp.func
def particle_force(n: wp.vec3, v: wp.vec3, c: float, k_n: float, k_d: float, k_f: float, k_mu: float):
    # compute normal and tangential friction force for a single contact
    vn = wp.dot(n, v)
    jn = c * k_n
    jd = min(vn, 0.0) * k_d

    # contact force
    fn = jn + jd

    # friction force
    vt = v - n * vn
    vs = wp.length(vt)

    if vs > 0.0:
        vt = vt / vs

    # Coulomb condition
    ft = wp.min(vs * k_f, k_mu * wp.abs(fn))

    # total force
    return -n * fn - vt * ft


@wp.kernel
def eval_particle_contact(
    grid: wp.uint64,
    particle_x: wp.array(dtype=wp.vec3),
    particle_v: wp.array(dtype=wp.vec3),
    particle_radius: wp.array(dtype=float),
    particle_flags: wp.array(dtype=wp.int32),
    k_contact: float,
    k_damp: float,
    k_friction: float,
    k_mu: float,
    k_cohesion: float,
    max_radius: float,
    # outputs
    particle_f: wp.array(dtype=wp.vec3),
):
    tid = wp.tid()

    # order threads by cell
    i = wp.hash_grid_point_id(grid, tid)
    if i == -1:
        # hash grid has not been built yet
        return
    if (particle_flags[i] & ParticleFlags.ACTIVE) == 0:
        return

    x = particle_x[i]
    v = particle_v[i]
    radius = particle_radius[i]

    f = wp.vec3()

    # particle contact
    query = wp.hash_grid_query(grid, x, radius + max_radius + k_cohesion)
    index = int(0)

    count = int(0)

    while wp.hash_grid_query_next(query, index):
        if (particle_flags[index] & ParticleFlags.ACTIVE) != 0 and index != i:
            # compute distance to point
            n = x - particle_x[index]
            d = wp.length(n)
            err = d - radius - particle_radius[index]

            count += 1

            if err <= k_cohesion:
                n = n / d
                vrel = v - particle_v[index]

                f = f + particle_force(n, vrel, err, k_contact, k_damp, k_friction, k_mu)

    particle_f[i] = f


@wp.kernel
def eval_triangle_contact(
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
def eval_particle_body_contact(
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

    body_w = wp.spatial_bottom(body_v_s)
    body_v = wp.spatial_top(body_v_s)

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
            wp.atomic_sub(body_f, body_index, wp.spatial_vector(f_total, wp.cross(bx, f_total)))
        else:
            wp.atomic_add(body_f, body_index, wp.spatial_vector(f_total, wp.cross(r, f_total)))


@wp.kernel
def eval_triangles_body_contact(
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
def eval_body_contact(
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
    rigid_contact_stiffness: wp.array(dtype=float),
    rigid_contact_damping: wp.array(dtype=float),
    rigid_contact_friction: wp.array(dtype=float),
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

    # per-contact stiffness/damping/friction
    if rigid_contact_stiffness.shape[0] > 0:
        contact_ke = rigid_contact_stiffness[tid]
        ke = contact_ke if contact_ke > 0.0 else ke
    if rigid_contact_damping.shape[0] > 0:
        contact_kd = rigid_contact_damping[tid]
        kd = contact_kd if contact_kd > 0.0 else kd
    if rigid_contact_friction.shape[0] > 0:
        contact_mu = rigid_contact_friction[tid]
        mu = contact_mu if contact_mu > 0.0 else mu
    
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
        body_w_a = wp.spatial_bottom(body_v_s_a)
        body_v_a = wp.spatial_top(body_v_s_a)
        if force_in_world_frame:
            bv_a = body_v_a + wp.cross(body_w_a, bx_a)
        else:
            bv_a = body_v_a + wp.cross(body_w_a, r_a)

    if body_b >= 0:
        body_v_s_b = body_qd[body_b]
        body_w_b = wp.spatial_bottom(body_v_s_b)
        body_v_b = wp.spatial_top(body_v_s_b)
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
            wp.atomic_add(body_f, body_a, wp.spatial_vector(f_total, wp.cross(bx_a, f_total)))
        else:
            wp.atomic_sub(body_f, body_a, wp.spatial_vector(f_total, wp.cross(r_a, f_total)))

    if body_b >= 0:
        if force_in_world_frame:
            wp.atomic_sub(body_f, body_b, wp.spatial_vector(f_total, wp.cross(bx_b, f_total)))
        else:
            wp.atomic_add(body_f, body_b, wp.spatial_vector(f_total, wp.cross(r_b, f_total)))


def eval_particle_contact_forces(model: Model, state: State, particle_f: wp.array):
    if model.particle_count > 1 and model.particle_max_radius > 0.0:
        wp.launch(
            kernel=eval_particle_contact,
            dim=model.particle_count,
            inputs=[
                model.particle_grid.id,
                state.particle_q,
                state.particle_qd,
                model.particle_radius,
                model.particle_flags,
                model.particle_ke,
                model.particle_kd,
                model.particle_kf,
                model.particle_mu,
                model.particle_cohesion,
                model.particle_max_radius,
            ],
            outputs=[particle_f],
            device=model.device,
        )


def eval_triangle_contact_forces(model: Model, state: State, particle_f: wp.array):
    if model.tri_count and model.particle_count:
        wp.launch(
            kernel=eval_triangle_contact,
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
            kernel=eval_body_contact,
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
                contacts.rigid_contact_stiffness,
                contacts.rigid_contact_damping,
                contacts.rigid_contact_friction,
                force_in_world_frame,
                friction_smoothing,
            ],
            outputs=[body_f_out],
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
            kernel=eval_particle_body_contact,
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
