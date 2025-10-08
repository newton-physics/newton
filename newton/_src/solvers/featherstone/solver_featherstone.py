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

from ...core.types import override
from ...sim import Contacts, Control, Model, State, eval_fk
from ..semi_implicit.kernels_contact import (
    eval_body_contact,
    eval_particle_body_contact_forces,
    eval_particle_contact_forces,
)
from ..semi_implicit.kernels_muscle import (
    eval_muscle_forces,
)
from ..semi_implicit.kernels_particle import (
    eval_bending_forces,
    eval_spring_forces,
    eval_tetrahedra_forces,
    eval_triangle_forces,
)
from ..solver import SolverBase
from .kernels import (
    compute_com_transforms,
    compute_spatial_inertia,
    create_inertia_matrix_cholesky_kernel,
    create_inertia_matrix_kernel,
    eval_dense_cholesky_batched,
    eval_dense_gemm_batched,
    eval_dense_solve_batched,
    eval_rigid_fk,
    eval_rigid_id,
    eval_rigid_jacobian,
    eval_rigid_mass,
    eval_rigid_tau,
    integrate_generalized_joints,
)


class SolverFeatherstone(SolverBase):
    """A semi-implicit integrator using symplectic Euler that operates
    on reduced (also called generalized) coordinates to simulate articulated rigid body dynamics
    based on Featherstone's composite rigid body algorithm (CRBA).

    See: Featherstone, Roy. Rigid Body Dynamics Algorithms. Springer US, 2014.

    Instead of maximal coordinates :attr:`~newton.State.body_q` (rigid body positions) and :attr:`~newton.State.body_qd`
    (rigid body velocities) as is the case in :class:`~newton.solvers.SolverSemiImplicit` and :class:`~newton.solvers.SolverXPBD`,
    :class:`~newton.solvers.SolverFeatherstone` uses :attr:`~newton.State.joint_q` and :attr:`~newton.State.joint_qd` to represent
    the positions and velocities of joints without allowing any redundant degrees of freedom.

    After constructing :class:`~newton.Model` and :class:`~newton.State` objects this time-integrator
    may be used to advance the simulation state forward in time.

    Note:
        Unlike :class:`~newton.solvers.SolverSemiImplicit` and :class:`~newton.solvers.SolverXPBD`, :class:`~newton.solvers.SolverFeatherstone`
        does not simulate rigid bodies with nonzero mass as floating bodies if they are not connected through any joints.
        Floating-base systems require an explicit free joint with which the body is connected to the world,
        see :meth:`newton.ModelBuilder.add_joint_free`.

    Semi-implicit time integration is a variational integrator that
    preserves energy, however it not unconditionally stable, and requires a time-step
    small enough to support the required stiffness and damping forces.

    See: https://en.wikipedia.org/wiki/Semi-implicit_Euler_method

    This solver uses the routines from :class:`~newton.solvers.SolverSemiImplicit` to simulate particles, cloth, and soft bodies.

    Example
    -------

    .. code-block:: python

        solver = newton.solvers.SolverFeatherstone(model)

        # simulation loop
        for i in range(100):
            solver.step(state_in, state_out, control, contacts, dt)
            state_in, state_out = state_out, state_in

    """

    def __init__(
        self,
        model: Model,
        angular_damping: float = 0.05,
        update_mass_matrix_interval: int = 1,
        friction_smoothing: float = 1.0,
        use_tile_gemm: bool = False,
        fuse_cholesky: bool = True,
    ):
        """
        Args:
            model (Model): the model to be simulated.
            angular_damping (float, optional): Angular damping factor. Defaults to 0.05.
            update_mass_matrix_interval (int, optional): How often to update the mass matrix (every n-th time the :meth:`step` function gets called). Defaults to 1.
            friction_smoothing (float, optional): The delta value for the Huber norm (see :func:`warp.math.norm_huber`) used for the friction velocity normalization. Defaults to 1.0.
            use_tile_gemm (bool, optional): Whether to use operators from Warp's Tile API to solve for joint accelerations. Defaults to False.
            fuse_cholesky (bool, optional): Whether to fuse the Cholesky decomposition into the inertia matrix evaluation kernel when using the Tile API. Only used if `use_tile_gemm` is true. Defaults to True.
        """
        super().__init__(model)

        self.angular_damping = angular_damping
        self.update_mass_matrix_interval = update_mass_matrix_interval
        self.friction_smoothing = friction_smoothing
        self.use_tile_gemm = use_tile_gemm
        self.fuse_cholesky = fuse_cholesky

        self._step = 0

        self.compute_articulation_indices(model)
        self.allocate_model_aux_vars(model)

        if self.use_tile_gemm:
            # create a custom kernel to evaluate the system matrix for this type
            if self.fuse_cholesky:
                self.eval_inertia_matrix_cholesky_kernel = create_inertia_matrix_cholesky_kernel(
                    int(self.joint_count), int(self.dof_count)
                )
            else:
                self.eval_inertia_matrix_kernel = create_inertia_matrix_kernel(
                    int(self.joint_count), int(self.dof_count)
                )

            # ensure matrix is reloaded since otherwise an unload can happen during graph capture
            # todo: should not be necessary?
            wp.load_module(device=wp.get_device())

    def compute_articulation_indices(self, model):
        # calculate total size and offsets of Jacobian and mass matrices for entire system
        if model.joint_count:
            self.J_size = 0
            self.M_size = 0
            self.H_size = 0

            articulation_J_start = []
            articulation_M_start = []
            articulation_H_start = []

            articulation_M_rows = []
            articulation_H_rows = []
            articulation_J_rows = []
            articulation_J_cols = []

            articulation_dof_start = []
            articulation_coord_start = []

            articulation_start = model.articulation_start.numpy()
            joint_q_start = model.joint_q_start.numpy()
            joint_qd_start = model.joint_qd_start.numpy()

            for i in range(model.articulation_count):
                first_joint = articulation_start[i]
                last_joint = articulation_start[i + 1]

                first_coord = joint_q_start[first_joint]

                first_dof = joint_qd_start[first_joint]
                last_dof = joint_qd_start[last_joint]

                joint_count = last_joint - first_joint
                dof_count = last_dof - first_dof

                articulation_J_start.append(self.J_size)
                articulation_M_start.append(self.M_size)
                articulation_H_start.append(self.H_size)
                articulation_dof_start.append(first_dof)
                articulation_coord_start.append(first_coord)

                # bit of data duplication here, but will leave it as such for clarity
                articulation_M_rows.append(joint_count * 6)
                articulation_H_rows.append(dof_count)
                articulation_J_rows.append(joint_count * 6)
                articulation_J_cols.append(dof_count)

                if self.use_tile_gemm:
                    # store the joint and dof count assuming all
                    # articulations have the same structure
                    self.joint_count = joint_count
                    self.dof_count = dof_count

                self.J_size += 6 * joint_count * dof_count
                self.M_size += 6 * joint_count * 6 * joint_count
                self.H_size += dof_count * dof_count

            # matrix offsets for batched gemm
            self.articulation_J_start = wp.array(articulation_J_start, dtype=wp.int32, device=model.device)
            self.articulation_M_start = wp.array(articulation_M_start, dtype=wp.int32, device=model.device)
            self.articulation_H_start = wp.array(articulation_H_start, dtype=wp.int32, device=model.device)

            self.articulation_M_rows = wp.array(articulation_M_rows, dtype=wp.int32, device=model.device)
            self.articulation_H_rows = wp.array(articulation_H_rows, dtype=wp.int32, device=model.device)
            self.articulation_J_rows = wp.array(articulation_J_rows, dtype=wp.int32, device=model.device)
            self.articulation_J_cols = wp.array(articulation_J_cols, dtype=wp.int32, device=model.device)

            self.articulation_dof_start = wp.array(articulation_dof_start, dtype=wp.int32, device=model.device)
            self.articulation_coord_start = wp.array(articulation_coord_start, dtype=wp.int32, device=model.device)

    def allocate_model_aux_vars(self, model):
        # allocate mass, Jacobian matrices, and other auxiliary variables pertaining to the model
        if model.joint_count:
            # system matrices
            self.M = wp.zeros((self.M_size,), dtype=wp.float32, device=model.device, requires_grad=model.requires_grad)
            self.J = wp.zeros((self.J_size,), dtype=wp.float32, device=model.device, requires_grad=model.requires_grad)
            self.P = wp.empty_like(self.J, requires_grad=model.requires_grad)
            self.H = wp.empty((self.H_size,), dtype=wp.float32, device=model.device, requires_grad=model.requires_grad)

            # zero since only upper triangle is set which can trigger NaN detection
            self.L = wp.zeros_like(self.H)

        if model.body_count:
            self.body_I_m = wp.empty(
                (model.body_count,), dtype=wp.spatial_matrix, device=model.device, requires_grad=model.requires_grad
            )
            wp.launch(
                compute_spatial_inertia,
                model.body_count,
                inputs=[model.body_inertia, model.body_mass],
                outputs=[self.body_I_m],
                device=model.device,
            )
            self.body_X_com = wp.empty(
                (model.body_count,), dtype=wp.transform, device=model.device, requires_grad=model.requires_grad
            )
            wp.launch(
                compute_com_transforms,
                model.body_count,
                inputs=[model.body_com],
                outputs=[self.body_X_com],
                device=model.device,
            )

    def allocate_state_aux_vars(self, model, target, requires_grad):
        # allocate auxiliary variables that vary with state
        if model.body_count:
            # joints
            target.joint_qdd = wp.zeros_like(model.joint_qd, requires_grad=requires_grad)
            target.joint_tau = wp.empty_like(model.joint_qd, requires_grad=requires_grad)
            if requires_grad:
                # used in the custom grad implementation of eval_dense_solve_batched
                target.joint_solve_tmp = wp.zeros_like(model.joint_qd, requires_grad=True)
            else:
                target.joint_solve_tmp = None
            target.joint_S_s = wp.empty(
                (model.joint_dof_count,),
                dtype=wp.spatial_vector,
                device=model.device,
                requires_grad=requires_grad,
            )

            # derived rigid body data (maximal coordinates)
            target.body_q_com = wp.empty_like(model.body_q, requires_grad=requires_grad)
            target.body_I_s = wp.empty(
                (model.body_count,), dtype=wp.spatial_matrix, device=model.device, requires_grad=requires_grad
            )
            target.body_v_s = wp.empty(
                (model.body_count,), dtype=wp.spatial_vector, device=model.device, requires_grad=requires_grad
            )
            target.body_a_s = wp.empty(
                (model.body_count,), dtype=wp.spatial_vector, device=model.device, requires_grad=requires_grad
            )
            target.body_f_s = wp.zeros(
                (model.body_count,), dtype=wp.spatial_vector, device=model.device, requires_grad=requires_grad
            )
            target.body_ft_s = wp.zeros(
                (model.body_count,), dtype=wp.spatial_vector, device=model.device, requires_grad=requires_grad
            )

            target._featherstone_augmented = True

    @override
    def step(
        self,
        state_in: State,
        state_out: State,
        control: Control,
        contacts: Contacts,
        dt: float,
    ):
        requires_grad = state_in.requires_grad

        # optionally create dynamical auxiliary variables
        if requires_grad:
            state_aug = state_out
        else:
            state_aug = self

        model = self.model

        if not getattr(state_aug, "_featherstone_augmented", False):
            self.allocate_state_aux_vars(model, state_aug, requires_grad)
        if control is None:
            control = model.control(clone_variables=False)

        with wp.ScopedTimer("simulate", False):
            particle_f = None
            body_f = None

            if state_in.particle_count:
                particle_f = state_in.particle_f

            if state_in.body_count:
                body_f = state_in.body_f

            # damped springs
            eval_spring_forces(model, state_in, particle_f)

            # triangle elastic and lift/drag forces
            eval_triangle_forces(model, state_in, control, particle_f)

            # triangle bending
            eval_bending_forces(model, state_in, particle_f)

            # tetrahedral FEM
            eval_tetrahedra_forces(model, state_in, control, particle_f)

            # particle-particle interactions
            eval_particle_contact_forces(model, state_in, particle_f)

            # particle shape contact
            eval_particle_body_contact_forces(model, state_in, contacts, particle_f, body_f, body_f_in_world_frame=True)

            # muscles
            if False:
                eval_muscle_forces(model, state_in, control, body_f)

            # ----------------------------
            # articulations

            if model.joint_count:
                # evaluate body transforms
                wp.launch(
                    eval_rigid_fk,
                    dim=model.articulation_count,
                    inputs=[
                        model.articulation_start,
                        model.joint_type,
                        model.joint_parent,
                        model.joint_child,
                        model.joint_q_start,
                        model.joint_qd_start,
                        state_in.joint_q,
                        model.joint_X_p,
                        model.joint_X_c,
                        self.body_X_com,
                        model.joint_axis,
                        model.joint_dof_dim,
                    ],
                    outputs=[state_in.body_q, state_aug.body_q_com],
                    device=model.device,
                )

                # print("body_X_sc:")
                # print(state_in.body_q.numpy())

                # evaluate joint inertias, motion vectors, and forces
                state_aug.body_f_s.zero_()

                wp.launch(
                    eval_rigid_id,
                    dim=model.articulation_count,
                    inputs=[
                        model.articulation_start,
                        model.joint_type,
                        model.joint_parent,
                        model.joint_child,
                        model.joint_qd_start,
                        state_in.joint_qd,
                        model.joint_axis,
                        model.joint_dof_dim,
                        self.body_I_m,
                        state_in.body_q,
                        state_aug.body_q_com,
                        model.joint_X_p,
                        model.gravity,
                    ],
                    outputs=[
                        state_aug.joint_S_s,
                        state_aug.body_I_s,
                        state_aug.body_v_s,
                        state_aug.body_f_s,
                        state_aug.body_a_s,
                    ],
                    device=model.device,
                )

                if contacts is not None and contacts.rigid_contact_max:
                    wp.launch(
                        kernel=eval_body_contact,
                        dim=contacts.rigid_contact_max,
                        inputs=[
                            state_in.body_q,
                            state_aug.body_v_s,
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
                            True,
                            self.friction_smoothing,
                        ],
                        outputs=[body_f],
                        device=model.device,
                    )

                if model.articulation_count:
                    # evaluate joint torques
                    state_aug.body_ft_s.zero_()
                    wp.launch(
                        eval_rigid_tau,
                        dim=model.articulation_count,
                        inputs=[
                            model.articulation_start,
                            model.joint_type,
                            model.joint_parent,
                            model.joint_child,
                            model.joint_q_start,
                            model.joint_qd_start,
                            model.joint_dof_dim,
                            model.joint_dof_mode,
                            state_in.joint_q,
                            state_in.joint_qd,
                            control.joint_f,
                            control.joint_target,
                            model.joint_target_ke,
                            model.joint_target_kd,
                            model.joint_limit_lower,
                            model.joint_limit_upper,
                            model.joint_limit_ke,
                            model.joint_limit_kd,
                            state_aug.joint_S_s,
                            state_aug.body_f_s,
                            body_f,
                        ],
                        outputs=[
                            state_aug.body_ft_s,
                            state_aug.joint_tau,
                        ],
                        device=model.device,
                    )

                    # print("joint_tau:")
                    # print(state_aug.joint_tau.numpy())
                    # print("body_q:")
                    # print(state_in.body_q.numpy())
                    # print("body_qd:")
                    # print(state_in.body_qd.numpy())

                    if self._step % self.update_mass_matrix_interval == 0:
                        # build J
                        wp.launch(
                            eval_rigid_jacobian,
                            dim=model.articulation_count,
                            inputs=[
                                model.articulation_start,
                                self.articulation_J_start,
                                model.joint_ancestor,
                                model.joint_qd_start,
                                state_aug.joint_S_s,
                            ],
                            outputs=[self.J],
                            device=model.device,
                        )

                        # build M
                        wp.launch(
                            eval_rigid_mass,
                            dim=model.articulation_count,
                            inputs=[
                                model.articulation_start,
                                self.articulation_M_start,
                                state_aug.body_I_s,
                            ],
                            outputs=[self.M],
                            device=model.device,
                        )

                        if self.use_tile_gemm:
                            # reshape arrays
                            M_tiled = self.M.reshape((-1, 6 * self.joint_count, 6 * self.joint_count))
                            J_tiled = self.J.reshape((-1, 6 * self.joint_count, self.dof_count))
                            R_tiled = model.joint_armature.reshape((-1, self.dof_count))
                            H_tiled = self.H.reshape((-1, self.dof_count, self.dof_count))
                            L_tiled = self.L.reshape((-1, self.dof_count, self.dof_count))
                            assert H_tiled.shape == (model.articulation_count, 18, 18)
                            assert L_tiled.shape == (model.articulation_count, 18, 18)
                            assert R_tiled.shape == (model.articulation_count, 18)

                            if self.fuse_cholesky:
                                wp.launch_tiled(
                                    self.eval_inertia_matrix_cholesky_kernel,
                                    dim=model.articulation_count,
                                    inputs=[J_tiled, M_tiled, R_tiled],
                                    outputs=[H_tiled, L_tiled],
                                    device=model.device,
                                    block_dim=64,
                                )

                            else:
                                wp.launch_tiled(
                                    self.eval_inertia_matrix_kernel,
                                    dim=model.articulation_count,
                                    inputs=[J_tiled, M_tiled],
                                    outputs=[H_tiled],
                                    device=model.device,
                                    block_dim=256,
                                )

                                wp.launch(
                                    eval_dense_cholesky_batched,
                                    dim=model.articulation_count,
                                    inputs=[
                                        self.articulation_H_start,
                                        self.articulation_H_rows,
                                        self.H,
                                        model.joint_armature,
                                    ],
                                    outputs=[self.L],
                                    device=model.device,
                                )

                            # import numpy as np
                            # J = J_tiled.numpy()
                            # M = M_tiled.numpy()
                            # R = R_tiled.numpy()
                            # for i in range(model.articulation_count):
                            #     r = R[i,:,0]
                            #     H = J[i].T @ M[i] @ J[i]
                            #     L = np.linalg.cholesky(H + np.diag(r))
                            #     np.testing.assert_allclose(H, H_tiled.numpy()[i], rtol=1e-2, atol=1e-2)
                            #     np.testing.assert_allclose(L, L_tiled.numpy()[i], rtol=1e-1, atol=1e-1)

                        else:
                            # form P = M*J
                            wp.launch(
                                eval_dense_gemm_batched,
                                dim=model.articulation_count,
                                inputs=[
                                    self.articulation_M_rows,
                                    self.articulation_J_cols,
                                    self.articulation_J_rows,
                                    False,
                                    False,
                                    self.articulation_M_start,
                                    self.articulation_J_start,
                                    # P start is the same as J start since it has the same dims as J
                                    self.articulation_J_start,
                                    self.M,
                                    self.J,
                                ],
                                outputs=[self.P],
                                device=model.device,
                            )

                            # form H = J^T*P
                            wp.launch(
                                eval_dense_gemm_batched,
                                dim=model.articulation_count,
                                inputs=[
                                    self.articulation_J_cols,
                                    self.articulation_J_cols,
                                    # P rows is the same as J rows
                                    self.articulation_J_rows,
                                    True,
                                    False,
                                    self.articulation_J_start,
                                    # P start is the same as J start since it has the same dims as J
                                    self.articulation_J_start,
                                    self.articulation_H_start,
                                    self.J,
                                    self.P,
                                ],
                                outputs=[self.H],
                                device=model.device,
                            )

                            # compute decomposition
                            wp.launch(
                                eval_dense_cholesky_batched,
                                dim=model.articulation_count,
                                inputs=[
                                    self.articulation_H_start,
                                    self.articulation_H_rows,
                                    self.H,
                                    model.joint_armature,
                                ],
                                outputs=[self.L],
                                device=model.device,
                            )

                        # print("joint_target:")
                        # print(control.joint_target.numpy())
                        # print("joint_tau:")
                        # print(state_aug.joint_tau.numpy())
                        # print("H:")
                        # print(self.H.numpy())
                        # print("L:")
                        # print(self.L.numpy())

                    # solve for qdd
                    state_aug.joint_qdd.zero_()
                    wp.launch(
                        eval_dense_solve_batched,
                        dim=model.articulation_count,
                        inputs=[
                            self.articulation_H_start,
                            self.articulation_H_rows,
                            self.articulation_dof_start,
                            self.H,
                            self.L,
                            state_aug.joint_tau,
                        ],
                        outputs=[
                            state_aug.joint_qdd,
                            state_aug.joint_solve_tmp,
                        ],
                        device=model.device,
                    )
                    # print("joint_qdd:")
                    # print(state_aug.joint_qdd.numpy())
                    # print("\n\n")

            # -------------------------------------
            # integrate bodies

            if model.joint_count:
                wp.launch(
                    kernel=integrate_generalized_joints,
                    dim=model.joint_count,
                    inputs=[
                        model.joint_type,
                        model.joint_q_start,
                        model.joint_qd_start,
                        model.joint_dof_dim,
                        state_in.joint_q,
                        state_in.joint_qd,
                        state_aug.joint_qdd,
                        dt,
                    ],
                    outputs=[state_out.joint_q, state_out.joint_qd],
                    device=model.device,
                )

                # update maximal coordinates
                eval_fk(model, state_out.joint_q, state_out.joint_qd, state_out)

            self.integrate_particles(model, state_in, state_out, dt)

            self._step += 1

            return state_out
