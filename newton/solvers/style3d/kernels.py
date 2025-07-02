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

from newton.geometry import PARTICLE_FLAG_ACTIVE
from newton.sim.model import ShapeMaterials

from ..vbd.solver_vbd import evaluate_body_particle_contact


@wp.func
def triangle_deformation_gradient(x0: wp.vec3, x1: wp.vec3, x2: wp.vec3, inv_dm: wp.mat22):
    x01, x02 = x1 - x0, x2 - x0
    Fu = x01 * inv_dm[0, 0] + x02 * inv_dm[1, 0]
    Fv = x01 * inv_dm[0, 1] + x02 * inv_dm[1, 1]
    return Fu, Fv


@wp.kernel
def eval_stretch_kernel(
    pos: wp.array(dtype=wp.vec3),
    face_areas: wp.array(dtype=float),
    inv_dms: wp.array(dtype=wp.mat22),
    faces: wp.array(dtype=wp.int32, ndim=2),
    aniso_ke: wp.array(dtype=wp.vec3),
    # outputs
    forces: wp.array(dtype=wp.vec3),
):
    """
    Ref. Large Steps in Cloth Simulation, Baraff & Witkin in 1998.
    """
    fid = wp.tid()

    inv_dm = inv_dms[fid]
    face_area = face_areas[fid]
    face = wp.vec3i(faces[fid, 0], faces[fid, 1], faces[fid, 2])

    Fu, Fv = triangle_deformation_gradient(pos[face[0]], pos[face[1]], pos[face[2]], inv_dm)

    len_Fu = wp.length(Fu)
    len_Fv = wp.length(Fv)

    Fu = wp.normalize(Fu) if (len_Fu > 1e-6) else wp.vec3(0.0)
    Fv = wp.normalize(Fv) if (len_Fv > 1e-6) else wp.vec3(0.0)

    dFu_dx = wp.vec3(-inv_dm[0, 0] - inv_dm[1, 0], inv_dm[0, 0], inv_dm[1, 0])
    dFv_dx = wp.vec3(-inv_dm[0, 1] - inv_dm[1, 1], inv_dm[0, 1], inv_dm[1, 1])

    ku = aniso_ke[fid][0]
    kv = aniso_ke[fid][1]
    ks = aniso_ke[fid][2]

    for i in range(3):
        force = -face_area * (
            ku * (len_Fu - 1.0) * dFu_dx[i] * Fu
            + kv * (len_Fv - 1.0) * dFv_dx[i] * Fv
            + ks * wp.dot(Fu, Fv) * (Fu * dFv_dx[i] + Fv * dFu_dx[i])
        )
        wp.atomic_add(forces, face[i], force)


@wp.kernel
def eval_bend_kernel(
    pos: wp.array(dtype=wp.vec3),
    edge_rest_area: wp.array(dtype=float),
    edge_bending_cot: wp.array(dtype=wp.vec4),
    edges: wp.array(dtype=wp.int32, ndim=2),
    edge_bending_properties: wp.array(dtype=float, ndim=2),
    # outputs
    forces: wp.array(dtype=wp.vec3),
):
    eid = wp.tid()
    if edges[eid][0] < 0 or edges[eid][1] < 0:
        return
    edge = edges[eid]
    edge_stiff = edge_bending_properties[eid][0] / edge_rest_area[eid]
    bend_weight = wp.vec4(0.0)
    bend_weight[2] = edge_bending_cot[eid][2] + edge_bending_cot[eid][3]
    bend_weight[3] = edge_bending_cot[eid][0] + edge_bending_cot[eid][1]
    bend_weight[0] = -edge_bending_cot[eid][0] - edge_bending_cot[eid][2]
    bend_weight[1] = -edge_bending_cot[eid][1] - edge_bending_cot[eid][3]
    bend_weight = bend_weight * edge_stiff
    for i in range(4):
        force = wp.vec3(0.0)
        for j in range(4):
            force = force - bend_weight[i] * bend_weight[j] * pos[edge[j]]
        wp.atomic_add(forces, edge[i], force)


@wp.kernel
def eval_drag_kernel(
    spring_stiff: float,
    face_index: wp.array(dtype=int),
    drag_pos: wp.array(dtype=wp.vec3),
    drag_bary_coord: wp.array(dtype=wp.vec3),
    faces: wp.array(dtype=wp.int32, ndim=2),
    vert_pos: wp.array(dtype=wp.vec3),
    # outputs
    forces: wp.array(dtype=wp.vec3),
    # pd_diags: wp.array(dtype=float),
):
    fid = face_index[0]
    if fid != -1:
        coord = drag_bary_coord[0]
        face = wp.vec3i(faces[fid, 0], faces[fid, 1], faces[fid, 2])
        x0 = vert_pos[face[0]]
        x1 = vert_pos[face[1]]
        x2 = vert_pos[face[2]]
        p = x0 * coord[0] + x1 * coord[1] + x2 * coord[2]
        dir = drag_pos[0] - p

        # spring_stiff = 1e2
        force = spring_stiff * dir
        wp.atomic_add(forces, face[0], force * coord[0])
        wp.atomic_add(forces, face[1], force * coord[1])
        wp.atomic_add(forces, face[2], force * coord[2])

        # pd_diags[face[0]] += spring_k * coord[0]
        # pd_diags[face[1]] += spring_k * coord[1]
        # pd_diags[face[2]] += spring_k * coord[2]


@wp.kernel
def eval_body_contact_kernel(
    # inputs
    dt: float,
    pos_prev: wp.array(dtype=wp.vec3),
    pos: wp.array(dtype=wp.vec3),
    # body-particle contact
    soft_contact_ke: float,
    soft_contact_kd: float,
    friction_mu: float,
    friction_epsilon: float,
    particle_radius: wp.array(dtype=float),
    soft_contact_particle: wp.array(dtype=int),
    contact_count: wp.array(dtype=int),
    contact_max: int,
    shape_materials: ShapeMaterials,
    shape_body: wp.array(dtype=int),
    body_q: wp.array(dtype=wp.transform),
    body_q_prev: wp.array(dtype=wp.transform),
    body_qd: wp.array(dtype=wp.spatial_vector),
    body_com: wp.array(dtype=wp.vec3),
    contact_shape: wp.array(dtype=int),
    contact_body_pos: wp.array(dtype=wp.vec3),
    contact_body_vel: wp.array(dtype=wp.vec3),
    contact_normal: wp.array(dtype=wp.vec3),
    # outputs: particle force and hessian
    forces: wp.array(dtype=wp.vec3),
    hessians: wp.array(dtype=wp.mat33),
):
    t_id = wp.tid()

    particle_body_contact_count = min(contact_max, contact_count[0])

    if t_id < particle_body_contact_count:
        particle_idx = soft_contact_particle[t_id]
        body_contact_force, body_contact_hessian = evaluate_body_particle_contact(
            particle_idx,
            pos[particle_idx],
            pos_prev[particle_idx],
            t_id,
            soft_contact_ke,
            soft_contact_kd,
            friction_mu,
            friction_epsilon,
            particle_radius,
            shape_materials,
            shape_body,
            body_q,
            body_q_prev,
            body_qd,
            body_com,
            contact_shape,
            contact_body_pos,
            contact_body_vel,
            contact_normal,
            dt,
        )
        wp.atomic_add(forces, particle_idx, body_contact_force)
        wp.atomic_add(hessians, particle_idx, body_contact_hessian)


@wp.kernel
def init_step_kernel(
    dt: float,
    gravity: wp.vec3,
    f_ext: wp.array(dtype=wp.vec3),
    v_curr: wp.array(dtype=wp.vec3),
    x_curr: wp.array(dtype=wp.vec3),
    x_prev: wp.array(dtype=wp.vec3),
    pd_diags: wp.array(dtype=float),
    particle_masses: wp.array(dtype=float),
    particle_flags: wp.array(dtype=wp.uint32),
    # outputs
    x_inertia: wp.array(dtype=wp.vec3),
    static_A_diags: wp.array(dtype=float),
    dx: wp.array(dtype=wp.vec3),
):
    tid = wp.tid()
    x_last = x_curr[tid]
    x_prev[tid] = x_last

    if not particle_flags[tid] & PARTICLE_FLAG_ACTIVE:
        x_inertia[tid] = x_prev[tid]
        static_A_diags[tid] = 0.0
        dx[tid] = wp.vec3(0.0)
    else:
        v_prev = v_curr[tid]
        mass = particle_masses[tid]
        static_A_diags[tid] = pd_diags[tid] + mass / (dt * dt)
        x_inertia[tid] = x_last + v_prev * dt + (gravity + f_ext[tid] / mass) * (dt * dt)
        dx[tid] = v_prev * dt

        # temp
        # x_curr[tid] = x_last + v_prev * dt


@wp.kernel
def init_rhs_kernel(
    dt: float,
    x_curr: wp.array(dtype=wp.vec3),
    x_inertia: wp.array(dtype=wp.vec3),
    particle_masses: wp.array(dtype=float),
    # outputs
    rhs: wp.array(dtype=wp.vec3),
):
    tid = wp.tid()
    rhs[tid] = (x_inertia[tid] - x_curr[tid]) * particle_masses[tid] / (dt * dt)


@wp.kernel
def prepare_jacobi_preconditioner_kernel(
    static_A_diags: wp.array(dtype=float),
    contact_hessian_diags: wp.array(dtype=wp.mat33),
    # outputs
    inv_A_diags: wp.array(dtype=wp.mat33),
    A_diags: wp.array(dtype=wp.mat33),
):
    tid = wp.tid()
    diag = wp.identity(3, float) * static_A_diags[tid]
    diag += contact_hessian_diags[tid]
    inv_A_diags[tid] = wp.inverse(diag)
    A_diags[tid] = diag


@wp.kernel
def PD_jacobi_step_kernel(
    rhs: wp.array(dtype=wp.vec3),
    x_in: wp.array(dtype=wp.vec3),
    inv_diags: wp.array(dtype=wp.mat33),
    # outputs
    x_out: wp.array(dtype=wp.vec3),
):
    tid = wp.tid()
    x_out[tid] = x_in[tid] + inv_diags[tid] * rhs[tid]


@wp.kernel
def nonlinear_step_kernel(
    x_in: wp.array(dtype=wp.vec3),
    # outputs
    x_out: wp.array(dtype=wp.vec3),
    dx: wp.array(dtype=wp.vec3),
):
    tid = wp.tid()
    x_out[tid] = x_in[tid] + dx[tid]
    dx[tid] = wp.vec3(0.0)


@wp.kernel
def apply_chebyshev_kernel(
    omega: float,
    prev_verts: wp.array(dtype=wp.vec3),
    next_verts: wp.array(dtype=wp.vec3),
):
    tid = wp.tid()
    next_verts[tid] = wp.lerp(prev_verts[tid], next_verts[tid], omega)


@wp.kernel
def update_velocity(
    dt: float,
    prev_pos: wp.array(dtype=wp.vec3),
    pos: wp.array(dtype=wp.vec3),
    vel: wp.array(dtype=wp.vec3),
):
    particle = wp.tid()
    vel[particle] = 0.998 * (pos[particle] - prev_pos[particle]) / dt
