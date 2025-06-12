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

from newton.core import PARTICLE_FLAG_ACTIVE, Contact, Control, Model, State

from ..solver import SolverBase
from .builder import PDMatrixBuilder
from .linear_solver import SparseMatrixELL

########################################################################################################################
#################################################    Style3D Solver    #################################################
########################################################################################################################


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
    inv_diags: wp.array(dtype=float),
    diags: wp.array(dtype=float),
    dx: wp.array(dtype=wp.vec3),
):
    tid = wp.tid()
    x_last = x_curr[tid]
    x_prev[tid] = x_last

    if not particle_flags[tid] & PARTICLE_FLAG_ACTIVE:
        x_inertia[tid] = x_prev[tid]
        dx[tid] = wp.vec3(0.0)
        inv_diags[tid] = 0.0
        diags[tid] = 0.0
    else:
        v_prev = v_curr[tid]
        mass = particle_masses[tid]
        diags[tid] = pd_diags[tid] + mass / (dt * dt)
        inv_diags[tid] = 1.0 / (pd_diags[tid] + mass / (dt * dt))
        x_inertia[tid] = x_last + v_prev * dt + (gravity + f_ext[tid] / mass) * (dt * dt)
        dx[tid] = v_prev * dt

        # temp
        x_curr[tid] = x_last + v_prev * dt


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
def PD_jacobi_step_kernel(
    rhs: wp.array(dtype=wp.vec3),
    x_in: wp.array(dtype=wp.vec3),
    inv_diags: wp.array(dtype=float),
    # outputs
    x_out: wp.array(dtype=wp.vec3),
):
    tid = wp.tid()
    x_out[tid] = x_in[tid] + rhs[tid] * inv_diags[tid]


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
    """Projective dynamic based cloth simulator.

    Ref[1]. Large Steps in Cloth Simulation, Baraff & Witkin.
    Ref[2]. Fast Simulation of Mass-Spring Systems, Tiantian Liu etc.

    Implicit-Euler method solves the following non-linear equation:

        (M / dt^2 + H(x)) * dx = (M / dt^2) * (x_prev + v_prev * dt - x) + f_ext(x) + f_int(x)
                               = (M / dt^2) * (x_prev + v_prev * dt + (dt^2 / M) * f_ext(x) - x) + f_int(x)
                                              ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                               = (M / dt^2) * (x_inertia - x) + f_int(x)

    Notations:
        M:  mass matrix
        x:  unsolved particle position
        H:  hessian matrix (function of x)
        P:  PD-approximated hessian matrix (constant)
        A:  M / dt^2 + H(x) or M / dt^2 + P
      rhs:  Right hand side of the equation: (M / dt^2) * (inertia_x - x) + f_int(x)
      res:  Residual: rhs - A * dx_init, or rhs if dx_init == 0
    """

    def __init__(
        self,
        model: Model,
        iterations=10,
    ):
        super().__init__(model)
        self.enable_chebyshev = True
        self.nonlinear_iterations = iterations
        self.pd_matrix_builder = PDMatrixBuilder(model.particle_count)

        self.A = SparseMatrixELL()
        self.dx = wp.zeros(model.particle_count, dtype=wp.vec3, device=self.device)
        self.rhs = wp.zeros(model.particle_count, dtype=wp.vec3, device=self.device)
        self.x_prev = wp.zeros(model.particle_count, dtype=wp.vec3, device=self.device)
        self.pd_diags = wp.zeros(model.particle_count, dtype=float, device=self.device)
        self.inv_diags = wp.zeros(model.particle_count, dtype=float, device=self.device)
        self.x_inertia = wp.zeros(model.particle_count, dtype=wp.vec3, device=self.device)

        # add new attributes for Style3D solve
        self.temp_verts0 = wp.zeros(model.particle_count, dtype=wp.vec3, device=self.device)
        self.temp_verts1 = wp.zeros(model.particle_count, dtype=wp.vec3, device=self.device)
        self.body_particle_contact_count = wp.zeros((model.particle_count,), dtype=wp.int32, device=self.device)
        self.collision_evaluation_kernel_launch_size = self.model.soft_contact_max

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
            raise ValueError("model must be the one used to initialize Style3DSolver")

        wp.launch(
            kernel=init_step_kernel,
            dim=self.model.particle_count,
            inputs=[
                dt,
                model.gravity,
                state_in.particle_f,
                state_in.particle_qd,
                state_in.particle_q,
                self.x_prev,
                self.pd_diags,
                self.model.particle_mass,
                self.model.particle_flags,
            ],
            outputs=[
                self.x_inertia,
                self.inv_diags,
                self.A.diag,
                self.dx,
            ],
            device=self.device,
        )

        omega = 1.0
        self.temp_verts1.assign(state_in.particle_q)
        for _iter in range(self.nonlinear_iterations):
            wp.launch(
                init_rhs_kernel,
                dim=self.model.particle_count,
                inputs=[
                    dt,
                    state_in.particle_q,
                    self.x_inertia,
                    self.model.particle_mass,
                ],
                outputs=[
                    self.rhs,
                ],
                device=self.device,
            )

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
                outputs=[self.rhs],
                device=self.device,
            )

            wp.launch(
                PD_jacobi_step_kernel,
                dim=self.model.particle_count,
                inputs=[
                    self.rhs,
                    state_in.particle_q,
                    self.inv_diags,
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
            inputs=[dt, self.x_prev, state_out.particle_q],
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
        self.pd_diags, self.A.num_nz, self.A.nz_ell = self.pd_matrix_builder.finialize(self.device)
        self.A.diag = wp.zeros_like(self.pd_diags)
