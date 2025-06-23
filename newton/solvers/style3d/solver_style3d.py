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

from newton.sim import Contacts, Control, State, Style3DModel, Style3DModelBuilder

from ..solver import SolverBase
from .builder import PDMatrixBuilder
from .kernels import (
    PD_jacobi_step_kernel,
    apply_chebyshev_kernel,
    eval_bend_kernel,
    eval_stretch_kernel,
    init_rhs_kernel,
    init_step_kernel,
    nonlinear_step_kernel,
    update_velocity,
)
from .linear_solver import PcgSolver, SparseMatrixELL

########################################################################################################################
#################################################    Style3D Solver    #################################################
########################################################################################################################


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
        model: Style3DModel,
        iterations=10,
    ):
        super().__init__(model)
        self.style3d_model = model
        self._enable_chebyshev = True
        self.nonlinear_iterations = iterations
        self.pd_matrix_builder = PDMatrixBuilder(model.particle_count)
        self.linear_solver = PcgSolver(model.particle_count, self.device)

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

    @staticmethod
    def get_chebyshev_omega(omega: float, iter: int):
        rho = 0.997
        if iter <= 5:
            return 1.0
        elif iter == 6:
            return 2.0 / (2.0 - rho * rho)
        else:
            return 4.0 / (4.0 - omega * rho * rho)

    def step(
        self, model: Style3DModel, state_in: State, state_out: State, control: Control, contacts: Contacts, dt: float
    ):
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
                    self.style3d_model.tri_aniso_ke,
                ],
                outputs=[self.rhs],
                device=self.device,
            )

            wp.launch(
                eval_bend_kernel,
                dim=len(self.style3d_model.edge_rest_area),
                inputs=[
                    state_in.particle_q,
                    self.style3d_model.edge_rest_area,
                    self.style3d_model.edge_bending_cot,
                    self.style3d_model.edge_indices,
                    self.style3d_model.edge_bending_properties,
                ],
                outputs=[self.rhs],
                device=self.device,
            )

            if self.linear_solver is None:  # for debug
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

                if self._enable_chebyshev:
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
            else:
                self.linear_solver.solve(self.A, self.dx, self.rhs, self.inv_diags, self.dx, 10)

                wp.launch(
                    nonlinear_step_kernel,
                    dim=self.model.particle_count,
                    inputs=[
                        state_in.particle_q,
                    ],
                    outputs=[state_out.particle_q, self.dx],
                    device=self.device,
                )

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
        builder: Style3DModelBuilder,
    ):
        with wp.ScopedTimer("Style3DSolver::precompute()"):
            self.pd_matrix_builder.add_stretch_constraints(
                builder.tri_indices, builder.tri_poses, builder.tri_aniso_ke, builder.tri_areas
            )
            self.pd_matrix_builder.add_bend_constraints(
                builder.edge_indices,
                builder.edge_rest_angle,
                builder.edge_rest_length,
                builder.edge_bending_properties,
                builder.edge_rest_area,
                builder.edge_bending_cot,
            )
            self.pd_diags, self.A.num_nz, self.A.nz_ell = self.pd_matrix_builder.finialize(self.device)
            self.A.diag = wp.zeros_like(self.pd_diags)
