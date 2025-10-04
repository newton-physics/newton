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

from ...sim import Control, Model, State


@wp.kernel
def eval_spring(
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
def eval_triangle(
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


def eval_spring_forces(model: Model, state: State, particle_f: wp.array):
    if model.spring_count:
        wp.launch(
            kernel=eval_spring,
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
            kernel=eval_triangle,
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


def eval_tetrahedra_forces(model: Model, state: State, control: Control, particle_f: wp.array):
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
