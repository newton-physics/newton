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

from ...core.types import override
from ...sim import Contacts, Control, Model, ModelBuilder, State
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

AttributeAssignment = Model.AttributeAssignment
AttributeFrequency = Model.AttributeFrequency

########################################################################################################################
#################################################    Style3D Solver    #################################################
########################################################################################################################


class SolverStyle3D(SolverBase):
    r"""Projective dynamics based cloth solver.

    References:
        1. Baraff, D. & Witkin, A. "Large Steps in Cloth Simulation."
        2. Liu, T. et al. "Fast Simulation of Mass-Spring Systems."

    Implicit-Euler method solves the following non-linear equation:

    .. math::

        (M / dt^2 + H(x)) \cdot dx &= (M / dt^2) \cdot (x_{prev} + v_{prev} \cdot dt - x) + f_{ext}(x) + f_{int}(x) \\
                                   &= (M / dt^2) \cdot (x_{prev} + v_{prev} \cdot dt + (dt^2 / M) \cdot f_{ext}(x) - x) + f_{int}(x) \\
                                   &= (M / dt^2) \cdot (x_{inertia} - x) + f_{int}(x)

    Notations:
        - :math:`M`: mass matrix
        - :math:`x`: unsolved particle position
        - :math:`H`: Hessian matrix (function of x)
        - :math:`P`: PD-approximated Hessian matrix (constant)
        - :math:`A`: :math:`M / dt^2 + H(x)` or :math:`M / dt^2 + P`
        - :math:`rhs`: Right hand side of the equation: :math:`(M / dt^2) \cdot (x_{inertia} - x) + f_{int}(x)`
        - :math:`res`: Residual: :math:`rhs - A \cdot dx_{init}`, or rhs if :math:`dx_{init} = 0`

    See Also:
        :doc:`newton.solvers.style3d </api/newton_solvers_style3d>` exposes
        helper functions that populate Style3D cloth data on a
        :class:`~newton.ModelBuilder`.

    Example:
        Build a mesh-based cloth with
        :func:`newton.solvers.style3d.add_cloth_mesh`::

            from newton.solvers import style3d

            builder = newton.ModelBuilder()
            SolverStyle3D.register_custom_attributes(builder)
            style3d.add_cloth_mesh(
                builder,
                pos=wp.vec3(0.0, 0.0, 0.0),
                rot=wp.quat_identity(),
                vel=wp.vec3(0.0, 0.0, 0.0),
                vertices=mesh.vertices.tolist(),
                indices=mesh.indices.tolist(),
                density=0.3,
                tri_aniso_ke=wp.vec3(1.0e2, 1.0e2, 1.0e1),
                edge_aniso_ke=wp.vec3(2.0e-5, 1.0e-5, 5.0e-6),
            )

        Or build a grid with :func:`newton.solvers.style3d.add_cloth_grid`::

            style3d.add_cloth_grid(
                builder,
                pos=wp.vec3(-0.5, 0.0, 2.0),
                rot=wp.quat_identity(),
                dim_x=64,
                dim_y=32,
                cell_x=0.1,
                cell_y=0.1,
                vel=wp.vec3(0.0, 0.0, 0.0),
                mass=0.1,
                tri_aniso_ke=wp.vec3(1.0e2, 1.0e2, 1.0e1),
                edge_aniso_ke=wp.vec3(2.0e-4, 1.0e-4, 5.0e-5),
            )

    """

    def __init__(
        self,
        model: Model,
        iterations: int = 10,
        linear_iterations: int = 10,
        drag_spring_stiff: float = 1e2,
        enable_mouse_dragging: bool = False,
    ):
        """
        Args:
            model: The :class:`~newton.Model` containing Style3D attributes to integrate.
            iterations: Number of non-linear iterations per step.
            linear_iterations: Number of linear iterations (currently PCG iter) per non-linear iteration.
            drag_spring_stiff: The stiffness of spring connecting barycentric-weighted drag-point and target-point.
            enable_mouse_dragging: Enable/disable dragging kernel.
        """

        super().__init__(model)
        if not hasattr(model, "style3d"):
            raise AttributeError(
                "Style3D custom attributes are missing from the model. "
                "Call SolverStyle3D.register_custom_attributes() before building the model."
            )
        self.style3d = model.style3d
        self.collision: Collision | None = Collision(model)  # set None to disable
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

    @override
    def step(self, state_in: State, state_out: State, control: Control, contacts: Contacts, dt: float):
        """Advance the Style3D solver by one time step.

        The solver performs non-linear projective dynamics iterations with
        optional collision handling. During the solve, positions in
        ``state_in`` are updated in-place to the current iterate; the final
        positions and velocities are written to ``state_out``.

        Args:
            state_in: Input :class:`newton.State` (positions updated in-place).
            state_out: Output :class:`newton.State` with the final state.
            control: :class:`newton.Control` input (currently unused).
            contacts: :class:`newton.Contacts` used for collision response.
            dt: Time step in seconds.
        """
        if self.collision is not None:
            self.collision.frame_begin(state_in.particle_q, state_in.particle_qd, dt)

        wp.launch(
            kernel=init_step_kernel,
            dim=self.model.particle_count,
            inputs=[
                dt,
                self.model.gravity,
                self.model.particle_world,
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
                    self.style3d.tri_aniso_ke,
                ],
                outputs=[self.rhs],
                device=self.device,
            )

            wp.launch(
                eval_bend_kernel,
                dim=len(self.style3d.edge_rest_area),
                inputs=[
                    state_in.particle_q,
                    self.style3d.edge_rest_area,
                    self.style3d.edge_bending_cot,
                    self.model.edge_indices,
                    self.model.edge_bending_properties,
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

    @override
    @classmethod
    def register_custom_attributes(cls, builder: ModelBuilder) -> None:
        """Declare Style3D custom attributes under the ``style3d`` namespace.

        See Also:
            :ref:`custom_attributes` for the custom attribute system overview.
        """
        builder.add_custom_attribute(
            ModelBuilder.CustomAttribute(
                name="tri_aniso_ke",
                frequency=AttributeFrequency.TRIANGLE,
                assignment=AttributeAssignment.MODEL,
                dtype=wp.vec3,
                default=wp.vec3(0.0),
                namespace="style3d",
            )
        )
        builder.add_custom_attribute(
            ModelBuilder.CustomAttribute(
                name="edge_rest_area",
                frequency=AttributeFrequency.EDGE,
                assignment=AttributeAssignment.MODEL,
                dtype=wp.float32,
                default=0.0,
                namespace="style3d",
            )
        )
        builder.add_custom_attribute(
            ModelBuilder.CustomAttribute(
                name="edge_bending_cot",
                frequency=AttributeFrequency.EDGE,
                assignment=AttributeAssignment.MODEL,
                dtype=wp.vec4,
                default=wp.vec4(0.0, 0.0, 0.0, 0.0),
                namespace="style3d",
            )
        )
        builder.add_custom_attribute(
            ModelBuilder.CustomAttribute(
                name="aniso_ke",
                frequency=AttributeFrequency.EDGE,
                assignment=AttributeAssignment.MODEL,
                dtype=wp.vec3,
                default=wp.vec3(0.0),
                namespace="style3d",
            )
        )

    def precompute(self, builder: ModelBuilder):
        with wp.ScopedTimer("SolverStyle3D::precompute()"):
            tri_aniso_attr = builder.custom_attributes.get("style3d:tri_aniso_ke")
            edge_rest_area_attr = builder.custom_attributes.get("style3d:edge_rest_area")
            edge_bending_cot_attr = builder.custom_attributes.get("style3d:edge_bending_cot")

            if tri_aniso_attr is None or edge_rest_area_attr is None or edge_bending_cot_attr is None:
                raise AttributeError(
                    "Style3D custom attributes are missing from the builder. "
                    "Call SolverStyle3D.register_custom_attributes() before building the model."
                )

            tri_aniso_ke = tri_aniso_attr.build_array(len(builder.tri_indices), device="cpu").numpy().tolist()
            edge_rest_area = edge_rest_area_attr.build_array(len(builder.edge_indices), device="cpu").numpy().tolist()
            edge_bending_cot = (
                edge_bending_cot_attr.build_array(len(builder.edge_indices), device="cpu").numpy().tolist()
            )

            self.pd_matrix_builder.add_stretch_constraints(
                builder.tri_indices, builder.tri_poses, tri_aniso_ke, builder.tri_areas
            )
            self.pd_matrix_builder.add_bend_constraints(
                builder.edge_indices,
                builder.edge_bending_properties,
                edge_rest_area,
                edge_bending_cot,
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
