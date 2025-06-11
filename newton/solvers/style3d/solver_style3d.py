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

import numpy as np
import warp as wp

from newton.core import PARTICLE_FLAG_ACTIVE, Contact, Control, Model, State

from ..solver import SolverBase

########################################################################################################################
#################################################    Style3D Solver    #################################################
########################################################################################################################


class PDMatrixBuilder:
    def __init__(self, num_verts: int, max_neighbor: int = 32):
        self.num_verts = num_verts
        self.max_neighbors = max_neighbor
        self.counts = np.zeros(num_verts, dtype=np.int32)
        self.diags = np.zeros(num_verts, dtype=np.float32)
        self.values = np.zeros(shape=(num_verts, max_neighbor), dtype=np.float32)
        self.neighbors = np.zeros(shape=(num_verts, max_neighbor), dtype=np.int32)

    def add_connection(self, v0: int, v1: int) -> int:
        if v0 >= self.num_verts:
            raise ValueError(f"Vertex index{v0} out of range {self.num_verts}")
        if v1 >= self.num_verts:
            raise ValueError(f"Vertex index{v1} out of range {self.num_verts}")

        for slot in range(self.counts[v0]):
            if self.neighbors[v0, slot] == v1:
                return slot

        if self.counts[v0] >= self.max_neighbors:
            raise ValueError(f"Exceeds max neighbors limit {self.max_neighbors}")

        slot = self.counts[v0]
        self.neighbors[v0, slot] = v1
        self.counts[v0] += 1
        return slot

    def add_stretch_constraints(
        self,
        tri_indices: list[list[int]],
        tri_poses: list[list[list[float]]],
        tri_aniso_ke: list[list[int]],
        tri_areas: list[float],
    ):
        for fid in range(len(tri_indices)):
            area = tri_areas[fid]
            inv_dm = tri_poses[fid]
            ku, kv, ks = tri_aniso_ke[fid]
            face = wp.vec3i(tri_indices[fid])
            dFu_dx = wp.vec3(-inv_dm[0][0] - inv_dm[1][0], inv_dm[0][0], inv_dm[1][0])
            dFv_dx = wp.vec3(-inv_dm[0][1] - inv_dm[1][1], inv_dm[0][1], inv_dm[1][1])
            for i in range(3):
                for j in range(i, 3):
                    weight = area * ((ku + ks) * dFu_dx[i] * dFu_dx[j] + (kv + ks) * dFv_dx[i] * dFv_dx[j])
                    if i != j:
                        slot_ij = self.add_connection(face[i], face[j])
                        slot_ji = self.add_connection(face[j], face[i])
                        self.values[face[i], slot_ij] += weight
                        self.values[face[j], slot_ji] += weight
                    else:
                        self.diags[face[i]] += weight


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
def forward_step(
    dt: float,
    gravity: wp.vec3,
    vel: wp.array(dtype=wp.vec3),
    last_pos: wp.array(dtype=wp.vec3),
    external_forces: wp.array(dtype=wp.vec3),
    particle_flags: wp.array(dtype=wp.uint32),
    inv_mass: wp.array(dtype=float),
    inertia: wp.array(dtype=wp.vec3),
    pos: wp.array(dtype=wp.vec3),
):
    tid = wp.tid()
    last_x = pos[tid]
    last_pos[tid] = last_x

    if not particle_flags[tid] & PARTICLE_FLAG_ACTIVE:
        inertia[tid] = last_pos[tid]
    else:
        last_v = vel[tid]
        new_v = last_v + (gravity + external_forces[tid] * inv_mass[tid]) * dt
        inertia[tid] = last_x + new_v * dt
        pos[tid] = last_x + last_v * dt


@wp.kernel
def solve_pd(
    dt: float,
    pos: wp.array(dtype=wp.vec3),
    forces: wp.array(dtype=wp.vec3),
    diags: wp.array(dtype=float),
    mass: wp.array(dtype=float),
    inertia: wp.array(dtype=wp.vec3),
    particle_flags: wp.array(dtype=wp.uint32),
    pos_new: wp.array(dtype=wp.vec3),
):
    tid = wp.tid()

    particle_pos = pos[tid]
    if not particle_flags[tid] & PARTICLE_FLAG_ACTIVE:
        pos_new[tid] = particle_pos
        return

    dt_sqr_reciprocal = 1.0 / (dt * dt)
    # inertia force and hessian
    diag = mass[tid] * dt_sqr_reciprocal
    f = mass[tid] * (inertia[tid] - particle_pos) * (dt_sqr_reciprocal)

    f += forces[tid]
    diag += diags[tid]

    pos_new[tid] = particle_pos + f / diag


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


class Style3DSolver(SolverBase):
    """ """

    def __init__(
        self,
        model: Model,
        iterations=10,
        friction_epsilon=1e-2,
    ):
        super().__init__(model)
        self.nonlinear_iterations = iterations
        self.pd_matrix_builder = PDMatrixBuilder(model.particle_count)

        # add new attributes for VBD solve
        self.particle_q_prev = wp.zeros_like(model.particle_q, device=self.device)
        self.inertia = wp.zeros_like(model.particle_q, device=self.device)
        self.temp_verts0 = wp.zeros(model.particle_count, dtype=wp.vec3, device=self.device)
        self.temp_verts1 = wp.zeros(model.particle_count, dtype=wp.vec3, device=self.device)
        self.enable_chebyshev = True
        self.forces = wp.zeros(model.particle_count, dtype=wp.vec3, device=self.device)
        self.pd_diags = wp.zeros(model.particle_count, dtype=float, device=self.device)
        self.body_particle_contact_count = wp.zeros((model.particle_count,), dtype=wp.int32, device=self.device)
        self.collision_evaluation_kernel_launch_size = self.model.soft_contact_max

        # spaces for particle force and hessian
        self.particle_forces = wp.zeros(self.model.particle_count, dtype=wp.vec3, device=self.device)
        self.particle_hessians = wp.zeros(self.model.particle_count, dtype=wp.mat33, device=self.device)
        self.friction_epsilon = friction_epsilon

    @staticmethod
    def get_chebyshev_omega(omega: float, iter: int):
        rho = 0.997
        if iter <= 5:
            return 1.0
        elif iter == 6:
            return 2.0 / (2.0 - rho * rho)
        else:
            return 4.0 / (4.0 - omega * rho * rho)

    def step(self, model: Model, state_in: State, state_out: State, control: Control, contacts: Contact, dt: float):
        if model is not self.model:
            raise ValueError("model must be the one used to initialize VBDSolver")

        wp.launch(
            kernel=forward_step,
            dim=self.model.particle_count,
            inputs=[
                dt,
                model.gravity,
                state_in.particle_qd,
                self.particle_q_prev,
                state_in.particle_f,
                self.model.particle_flags,
                self.model.particle_inv_mass,
            ],
            outputs=[
                self.inertia,
                state_in.particle_q,
            ],
            device=self.device,
        )

        omega = 1.0
        self.temp_verts1.assign(state_in.particle_q)
        for _iter in range(self.nonlinear_iterations):
            self.forces.zero_()
            wp.launch(
                eval_stretch_kernel,
                dim=len(self.model.tri_areas),
                inputs=[
                    state_in.particle_q,
                    self.model.tri_areas,
                    self.model.tri_poses,
                    self.model.tri_indices,
                    self.model.tri_aniso_ke,
                ],
                outputs=[
                    self.forces,
                ],
                device=self.device,
            )
            wp.launch(
                solve_pd,
                dim=self.model.particle_count,
                inputs=[
                    dt,
                    state_in.particle_q,
                    self.forces,
                    self.pd_diags,
                    self.model.particle_mass,
                    self.inertia,
                    self.model.particle_flags,
                ],
                outputs=[
                    self.temp_verts0,
                ],
                device=self.device,
            )

            if self.enable_chebyshev:
                omega = self.get_chebyshev_omega(omega, _iter)
                if omega > 1.0:
                    wp.launch(
                        apply_chebyshev_kernel,
                        dim=self.model.particle_count,
                        inputs=[omega, self.temp_verts1],
                        outputs=[self.temp_verts0],
                        device=self.device,
                    )
                self.temp_verts1.assign(state_in.particle_q)

            state_out.particle_q.assign(self.temp_verts0)
            state_in.particle_q.assign(state_out.particle_q)

        wp.launch(
            kernel=update_velocity,
            dim=self.model.particle_count,
            inputs=[dt, self.particle_q_prev, state_out.particle_q],
            outputs=[state_out.particle_qd],
            device=self.device,
        )

    def precompute(
        self,
        tri_indices: list[list[int]],
        tri_poses: list[list[list[float]]],
        tri_aniso_ke: list[list[int]],
        tri_areas: list[float],
    ):
        self.pd_matrix_builder.add_stretch_constraints(tri_indices, tri_poses, tri_aniso_ke, tri_areas)
        self.pd_diags = wp.array(self.pd_matrix_builder.diags, dtype=float, device=self.device)
