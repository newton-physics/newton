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

from ...sim import Contacts, Control, State, Style3DModel, Style3DModelBuilder
from ..solver import SolverBase
from .builder import PDMatrixBuilder
from .collision import Collision
from .kernels import (
    accumulate_dragging_pd_diag_kernel,
    eval_bend_kernel,
    eval_drag_force_kernel,
    eval_stretch_kernel,
    init_rhs_kernel,
    init_step_kernel,
    nonlinear_step_kernel,
    prepare_jacobi_preconditioner_kernel,
    prepare_jacobi_preconditioner_no_contact_hessian_kernel,
    update_velocity,
)
from .linear_solver import PcgSolver, SparseMatrixELL

########################################################################################################################
#################################################    Style3D Solver    #################################################
########################################################################################################################


class SolverStyle3D(SolverBase):
    """Projective dynamic based cloth simulator.

    Ref[1]. Large Steps in Cloth Simulation, Baraff & Witkin.
    Ref[2]. Fast Simulation of Mass-Spring Systems, Tiantian Liu etc.

    Implicit-Euler method solves the following non-linear equation::

        (M / dt^2 + H(x)) * dx = (M / dt^2) * (x_prev + v_prev * dt - x) + f_ext(x) + f_int(x)
                               = (M / dt^2) * (x_prev + v_prev * dt + (dt^2 / M) * f_ext(x) - x) + f_int(x)
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
        linear_iterations=10,
        drag_spring_stiff: float = 1e2,
        enable_mouse_dragging: bool = False,
    ):
        """
        Args:
            model: The `Style3DModel` to integrate.
            iterations: Number of non-linear iterations per step.
            linear_iterations: Number of linear iterations (currently PCG iter) per non-linear iteration.
            drag_spring_stiff: The stiffness of spring connecting barycentric-weighted drag-point and target-point.
            enable_mouse_dragging: Enable/disable dragging kernel.
        """

        super().__init__(model)
        self.style3d_model = model
        self.collision = Collision(model)  # set None to disable
        self.linear_iterations = linear_iterations
        self.nonlinear_iterations = iterations
        self.drag_spring_stiff = drag_spring_stiff
        self.enable_mouse_dragging = enable_mouse_dragging
        self.pd_matrix_builder = PDMatrixBuilder(model.particle_count)
        self.linear_solver = PcgSolver(model.particle_count, self.device)

        # Fixed PD matrix
        self.pd_non_diags = SparseMatrixELL()
        self.pd_diags = wp.zeros(model.particle_count, dtype=float, device=self.device)

        # Non-linear equation variables
        self.dx = wp.zeros(model.particle_count, dtype=wp.vec3, device=self.device)
        self.rhs = wp.zeros(model.particle_count, dtype=wp.vec3, device=self.device)
        self.x_prev = wp.zeros(model.particle_count, dtype=wp.vec3, device=self.device)
        self.x_inertia = wp.zeros(model.particle_count, dtype=wp.vec3, device=self.device)

        # Static part of A_diag, full A_diag, and inverse of A_diag
        self.static_A_diags = wp.zeros(model.particle_count, dtype=float, device=self.device)
        self.inv_A_diags = wp.zeros(model.particle_count, dtype=wp.mat33, device=self.device)
        self.A_diags = wp.zeros(model.particle_count, dtype=wp.mat33, device=self.device)

        # Drag info
        self.drag_pos = wp.zeros(1, dtype=wp.vec3, device=self.device)
        self.drag_index = wp.array([-1], dtype=int, device=self.device)
        self.drag_bary_coord = wp.zeros(1, dtype=wp.vec3, device=self.device)

    def step(self, state_in: State, state_out: State, control: Control, contacts: Contacts, dt: float):
        if self.collision is not None:
            self.collision.frame_begin(state_in.particle_q, state_in.particle_qd, dt)

        wp.launch(
            kernel=init_step_kernel,
            dim=self.model.particle_count,
            inputs=[
                dt,
                self.model.gravity,
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
                self.static_A_diags,
                self.dx,
            ],
            device=self.device,
        )

        if self.enable_mouse_dragging:
            wp.launch(
                accumulate_dragging_pd_diag_kernel,
                dim=1,
                inputs=[
                    self.drag_spring_stiff,
                    self.drag_index,
                    self.drag_bary_coord,
                    self.model.tri_indices,
                    self.model.particle_flags,
                ],
                outputs=[self.static_A_diags],
                device=self.device,
            )

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
                outputs=[self.rhs],
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

            if self.enable_mouse_dragging:
                wp.launch(
                    eval_drag_force_kernel,
                    dim=1,
                    inputs=[
                        self.drag_spring_stiff,
                        self.drag_index,
                        self.drag_pos,
                        self.drag_bary_coord,
                        self.model.tri_indices,
                        state_in.particle_q,
                    ],
                    outputs=[self.rhs],
                    device=self.device,
                )

            if self.collision is not None:
                self.collision.accumulate_contact_force(
                    dt,
                    _iter,
                    state_in,
                    state_out,
                    contacts,
                    self.rhs,
                    self.x_prev,
                    self.static_A_diags,
                )
                wp.launch(
                    prepare_jacobi_preconditioner_kernel,
                    dim=self.model.particle_count,
                    inputs=[
                        self.static_A_diags,
                        self.collision.contact_hessian_diagonal(),
                        self.model.particle_flags,
                    ],
                    outputs=[self.inv_A_diags],
                    device=self.device,
                )
            else:
                wp.launch(
                    prepare_jacobi_preconditioner_no_contact_hessian_kernel,
                    dim=self.model.particle_count,
                    inputs=[self.static_A_diags],
                    outputs=[self.inv_A_diags],
                    device=self.device,
                )

            self.linear_solver.solve(
                self.pd_non_diags,
                self.static_A_diags,
                self.dx if _iter == 0 else None,
                self.rhs,
                self.inv_A_diags,
                self.dx,
                self.linear_iterations,
                None if self.collision is None else self.collision.hessian_multiply,
            )

            if self.collision is not None:
                self.collision.linear_iteration_end(self.dx)

            wp.launch(
                nonlinear_step_kernel,
                dim=self.model.particle_count,
                inputs=[state_in.particle_q],
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

        if self.collision is not None:
            self.collision.frame_end(state_out.particle_q, state_out.particle_qd, dt)

    def rebuild_bvh(self, state: State):
        if self.collision is not None:
            self.collision.rebuild_bvh(state.particle_q)

    def precompute(self, builder: Style3DModelBuilder):
        with wp.ScopedTimer("SolverStyle3D::precompute()"):
            self.pd_matrix_builder.add_stretch_constraints(
                builder.tri_indices, builder.tri_poses, builder.tri_aniso_ke, builder.tri_areas
            )
            self.pd_matrix_builder.add_bend_constraints(
                builder.edge_indices,
                builder.edge_bending_properties,
                builder.edge_rest_area,
                builder.edge_bending_cot,
            )
            self.pd_diags, self.pd_non_diags.num_nz, self.pd_non_diags.nz_ell = self.pd_matrix_builder.finalize(
                self.device
            )

    def update_drag_info(self, index: int, pos: wp.vec3, bary_coord: wp.vec3):
        """Should be invoked when state changed."""
        # print([index, pos, bary_coord])
        self.drag_bary_coord.fill_(bary_coord)
        self.drag_index.fill_(index)
        self.drag_pos.fill_(pos)
