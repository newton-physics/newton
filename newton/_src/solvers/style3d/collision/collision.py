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

from newton import Contacts, Model, State
from newton._src.solvers.style3d.collision.bvh import BvhEdge, BvhTri
from newton._src.solvers.style3d.collision.kernels import (
    eval_body_contact_kernel,
    handle_edge_edge_contacts_kernel,
    handle_vertex_triangle_contacts_kernel,
    solve_untangling_kernel,
)

########################################################################################################################
###################################################    Collision    ####################################################
########################################################################################################################


class Collision:
    """
    Collision handler for cloth simulation.
    """

    def __init__(self, model: Model):
        """
        Initialize the collision handler, including BVHs and buffers.

        Args:
            model: The simulation model containing particle and geometry data.
        """
        self.model = model
        self.radius = 3e-3  # Contact radius
        self.stiff_vf = 0.5  # Stiffness coefficient for vertex-face (VF) collision constraints
        self.stiff_ee = 0.1  # Stiffness coefficient for edge-edge (EE) collision constraints
        self.stiff_ef = 1.0  # Stiffness coefficient for edge-face (EF) collision constraints
        self.friction_epsilon = 1e-2
        self.integrate_with_external_rigid_solver = True
        self.tri_bvh = BvhTri(model.tri_count, self.model.device)
        self.edge_bvh = BvhEdge(model.edge_count, self.model.device)
        self.body_contact_max = model.shape_count * model.particle_count
        self.broad_phase_ee = wp.array(shape=(32, model.edge_count), dtype=int, device=self.model.device)
        self.broad_phase_ef = wp.array(shape=(32, model.edge_count), dtype=int, device=self.model.device)
        self.broad_phase_vf = wp.array(shape=(32, model.particle_count), dtype=int, device=self.model.device)

        self.Hx = wp.zeros(model.particle_count, dtype=wp.vec3, device=self.model.device)
        self.contact_hessian_diags = wp.zeros(model.particle_count, dtype=wp.mat33, device=self.model.device)

        self.edge_bvh.build(model.particle_q, self.model.edge_indices, self.radius)
        self.tri_bvh.build(model.particle_q, self.model.tri_indices, self.radius)

    def rebuild_bvh(self, pos: wp.array(dtype=wp.vec3)):
        """
        Rebuild triangle and edge BVHs.

        Args:
            pos: Array of vertex positions.
        """
        self.tri_bvh.rebuild(pos, self.model.tri_indices, self.radius)
        self.edge_bvh.rebuild(pos, self.model.edge_indices, self.radius)

    def refit_bvh(self, pos: wp.array(dtype=wp.vec3)):
        """
        Refit (update) triangle and edge BVHs based on new positions without changing topology.

        Args:
            pos: Array of vertex positions.
        """
        self.tri_bvh.refit(pos, self.model.tri_indices, self.radius)
        self.edge_bvh.refit(pos, self.model.edge_indices, self.radius)

    def frame_begin(self, particle_q: wp.array(dtype=wp.vec3), particle_qd: wp.array(dtype=wp.vec3), dt: float):
        """
        Perform broad-phase collision detection using BVHs.

        Args:
            particle_q: Array of vertex positions.
            particle_qd: Array of vertex velocities.
            dt: simulation time step.
        """
        max_dist = self.radius * 3.0
        query_radius = self.radius

        self.refit_bvh(particle_q)

        # Vertex-face collision candidates
        if self.stiff_vf > 0.0:
            self.tri_bvh.triangle_vs_point(
                particle_q,
                particle_q,
                self.model.tri_indices,
                self.broad_phase_vf,
                True,
                max_dist,
                query_radius,
            )

        # Edge-edge collision candidates
        if self.stiff_ee > 0.0:
            self.edge_bvh.edge_vs_edge(
                particle_q,
                self.model.edge_indices,
                particle_q,
                self.model.edge_indices,
                self.broad_phase_ee,
                True,
                max_dist,
                query_radius,
            )

        # Face-edge collision candidates
        if self.stiff_ef > 0.0:
            self.tri_bvh.aabb_vs_aabb(
                self.edge_bvh.lower_bounds,
                self.edge_bvh.upper_bounds,
                self.broad_phase_ef,
                query_radius,
                False,
            )

    def accumulate_contact_force(
        self,
        dt: float,
        _iter: int,
        state_in: State,
        state_out: State,
        contacts: Contacts,
        particle_forces: wp.array(dtype=wp.vec3),
        particle_q_prev: wp.array(dtype=wp.vec3),
        particle_stiff: wp.array(dtype=wp.vec3) = None,
    ):
        """
        Evaluates contact forces and the diagonal of the Hessian for implicit time integration.

        This method launches kernels to compute contact forces and Hessian contributions
        based on broad-phase collision candidates computed in frame_begin().

        Args:
            dt (float): Time step.
            state_in (State): Current simulation state (input).
            state_out (State): Next simulation state (output).
            contacts (Contacts): Contact data structure containing contact information.
            particle_forces (wp.array): Output array for computed contact forces.
            particle_q_prev (wp.array): Previous positions (optional, for velocity-based damping).
            particle_stiff (wp.array): Optional stiffness array for particles.
        """
        thickness = 2.0 * self.radius
        self.contact_hessian_diags.zero_()

        if self.stiff_vf > 0:
            wp.launch(
                handle_vertex_triangle_contacts_kernel,
                dim=len(state_in.particle_q),
                inputs=[
                    thickness,
                    self.stiff_vf,
                    state_in.particle_q,
                    self.model.tri_indices,
                    self.broad_phase_vf,
                    particle_stiff,
                ],
                outputs=[particle_forces, self.contact_hessian_diags],
                device=self.model.device,
            )

        if self.stiff_ee > 0:
            wp.launch(
                handle_edge_edge_contacts_kernel,
                dim=self.model.edge_indices.shape[0],
                inputs=[
                    thickness,
                    self.stiff_ee,
                    state_in.particle_q,
                    self.model.edge_indices,
                    self.broad_phase_ee,
                    particle_stiff,
                ],
                outputs=[particle_forces, self.contact_hessian_diags],
                device=self.model.device,
            )

        if self.stiff_ef > 0:
            wp.launch(
                solve_untangling_kernel,
                dim=self.model.edge_indices.shape[0],
                inputs=[
                    thickness,
                    self.stiff_ef,
                    state_in.particle_q,
                    self.model.tri_indices,
                    self.model.edge_indices,
                    self.broad_phase_ef,
                    particle_stiff,
                ],
                outputs=[particle_forces, self.contact_hessian_diags],
                device=self.model.device,
            )

        wp.launch(
            kernel=eval_body_contact_kernel,
            dim=self.body_contact_max,
            inputs=[
                dt,
                particle_q_prev,
                state_in.particle_q,
                # body-particle contact
                self.model.soft_contact_ke,
                self.model.soft_contact_kd,
                self.model.soft_contact_mu,
                self.friction_epsilon,
                self.model.particle_radius,
                contacts.soft_contact_particle,
                contacts.soft_contact_count,
                contacts.soft_contact_max,
                self.model.shape_material_mu,
                self.model.shape_body,
                state_out.body_q if self.integrate_with_external_rigid_solver else state_in.body_q,
                state_in.body_q if self.integrate_with_external_rigid_solver else None,
                self.model.body_qd,
                self.model.body_com,
                contacts.soft_contact_shape,
                contacts.soft_contact_body_pos,
                contacts.soft_contact_body_vel,
                contacts.soft_contact_normal,
            ],
            outputs=[particle_forces, self.contact_hessian_diags],
            device=self.model.device,
        )

    def contact_hessian_diagonal(self):
        """Return diagonal of contact Hessian for preconditioning.
        Note:
            Should be called after `accumulate_contact_force()`.
        """
        return self.contact_hessian_diags

    def hessian_multiply(self, x: wp.array(dtype=wp.vec3)):
        """Computes the Hessian-vector product for implicit integration."""

        @wp.kernel
        def hessian_multiply_kernel(
            hessian_diags: wp.array(dtype=wp.mat33),
            x: wp.array(dtype=wp.vec3),
            # outputs
            Hx: wp.array(dtype=wp.vec3),
        ):
            tid = wp.tid()
            Hx[tid] = hessian_diags[tid] * x[tid]

        wp.launch(
            hessian_multiply_kernel,
            dim=self.model.particle_count,
            inputs=[self.contact_hessian_diags, x],
            outputs=[self.Hx],
            device=self.model.device,
        )
        return self.Hx

    def linear_iteration_end(self, dx: wp.array(dtype=wp.vec3)):
        """Displacement constraints"""
        pass

    def frame_end(self, pos: wp.array(dtype=wp.vec3), vel: wp.array(dtype=wp.vec3), dt: float):
        """Apply post-processing"""
        pass
