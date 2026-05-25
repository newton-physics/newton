# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import warp as wp

from ...core.types import override
from ...sim import BodyFlags, Contacts, Control, JointType, Model, State
from ..flags import SolverNotifyFlags
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
    accumulate_free_distance_joint_f_to_body_force,
    compute_body_parent_f,
    compute_com_transforms,
    compute_spatial_inertia,
    convert_body_force_com_to_origin,
    convert_free_distance_joint_f_public_to_internal,
    convert_free_distance_joint_qd_internal_to_public,
    convert_free_distance_joint_qd_public_to_internal,
    copy_kinematic_joint_state,
    correct_free_distance_body_pose_from_world_twist,
    create_inertia_matrix_cholesky_kernel,
    create_inertia_matrix_kernel,
    eval_dense_cholesky_batched,
    eval_dense_gemm_batched,
    eval_dense_solve_batched,
    eval_fk_with_velocity_conversion,
    eval_fk_with_velocity_conversion_from_joint_starts,
    eval_rigid_fk,
    eval_rigid_id,
    eval_rigid_jacobian,
    eval_rigid_mass,
    eval_rigid_tau,
    integrate_generalized_joints,
    reconstruct_free_distance_joint_q_from_body_pose,
    zero_kinematic_body_forces,
    zero_kinematic_joint_qdd,
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

    Joint limitations:
        - Supported joint types: PRISMATIC, REVOLUTE, BALL, FIXED, FREE, DISTANCE (treated as FREE), D6.
          CABLE joints are not supported.
        - :attr:`~newton.Model.joint_armature`, :attr:`~newton.Model.joint_limit_ke`/:attr:`~newton.Model.joint_limit_kd`,
          :attr:`~newton.Model.joint_target_ke`/:attr:`~newton.Model.joint_target_kd`, and :attr:`~newton.Control.joint_f`
          are supported.
        - :attr:`~newton.Model.joint_friction`, :attr:`~newton.Model.joint_effort_limit`,
          :attr:`~newton.Model.joint_velocity_limit`, :attr:`~newton.Model.joint_enabled`,
          and :attr:`~newton.Model.joint_target_mode` are not supported.
        - Equality and mimic constraints are not supported.

        See :ref:`Joint feature support` for the full comparison across solvers.

    Extended state attributes:
        :attr:`~newton.State.body_parent_f` is populated when requested via
        :meth:`~newton.ModelBuilder.request_state_attributes`. The reported
        wrench is the per-body net spatial force from the RNEA backward pass
        translated to the body's COM (linear ``[N]`` first, torque ``[N·m]``
        in world frame at the COM), matching the wrench-transmitted-through-
        the-inbound-joint convention used by :class:`~newton.solvers.SolverMuJoCo`'s
        ``cfrc_int``. In equilibrium this reaction counters all forces acting
        on the body's subtree (gravity, contacts, ``State.body_f``, and the
        net effect of :attr:`~newton.Control.joint_f`) by Newton's third law.

        Free-floating roots (bodies whose only inbound joint is FREE) are
        not special-cased: the same RNEA sum is written, which for such
        bodies is the residual force balance from the recursion (e.g.
        contacts vs. gravity in equilibrium, or the gyroscopic
        ``v x* (I*v)`` term during tumbling) rather than a true joint
        reaction. Treat it as a diagnostic in that case.

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
            model: The model to be simulated.
            angular_damping: Angular damping factor. Defaults to 0.05.
            update_mass_matrix_interval: How often to update the mass matrix (every n-th time the :meth:`step` function gets called). Defaults to 1.
            friction_smoothing: The delta value for the Huber norm (see :func:`warp.norm_huber() <warp._src.lang.norm_huber>`) used for the friction velocity normalization. Defaults to 1.0.
            use_tile_gemm: Whether to use operators from Warp's Tile API to solve for joint accelerations. Defaults to False.
            fuse_cholesky: Whether to fuse the Cholesky decomposition into the inertia matrix evaluation kernel when using the Tile API. Only used if `use_tile_gemm` is true. Defaults to True.
        """
        super().__init__(model)

        self.angular_damping = angular_damping
        self.update_mass_matrix_interval = update_mass_matrix_interval
        self.friction_smoothing = friction_smoothing
        self.use_tile_gemm = use_tile_gemm
        self.fuse_cholesky = fuse_cholesky

        self._step = 0
        self._mass_matrix_dirty = False

        self._update_kinematic_state()

        self._compute_articulation_indices(model)
        self._allocate_model_aux_vars(model)

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

    def _update_kinematic_state(self):
        """Recompute cached solver flags and effective joint armature."""
        model = self.model
        self.has_kinematic_bodies = False
        self.has_kinematic_joints = False
        self.descendant_free_distance_joint_indices = None
        self.descendant_free_distance_articulation_indices = None
        self.descendant_free_distance_refresh_joint_starts = None
        self.joint_armature_effective = model.joint_armature

        if model.joint_count:
            joint_type = model.joint_type.numpy()
            joint_parent = model.joint_parent.numpy()
            descendant_free_distance_mask = (
                (joint_type == int(JointType.FREE)) | (joint_type == int(JointType.DISTANCE))
            ) & (joint_parent >= 0)
            if np.any(descendant_free_distance_mask):
                joint_indices = np.flatnonzero(descendant_free_distance_mask)
                self.descendant_free_distance_joint_indices = wp.array(
                    joint_indices,
                    dtype=wp.int32,
                    device=model.device,
                )
                articulation_start = model.articulation_start.numpy()
                articulation_indices = []
                refresh_joint_starts = []
                for articulation in range(model.articulation_count):
                    joint_start = articulation_start[articulation]
                    joint_end = articulation_start[articulation + 1]
                    descendant_joint_offsets = np.flatnonzero(descendant_free_distance_mask[joint_start:joint_end])
                    if descendant_joint_offsets.size == 0:
                        continue
                    articulation_indices.append(articulation)
                    refresh_joint_starts.append(joint_start + int(descendant_joint_offsets[0]))
                if articulation_indices:
                    self.descendant_free_distance_articulation_indices = wp.array(
                        articulation_indices,
                        dtype=wp.int32,
                        device=model.device,
                    )
                    self.descendant_free_distance_refresh_joint_starts = wp.array(
                        refresh_joint_starts,
                        dtype=wp.int32,
                        device=model.device,
                    )

        if model.body_count:
            body_flags = model.body_flags.numpy()
            kinematic_mask = (body_flags & int(BodyFlags.KINEMATIC)) != 0
            self.has_kinematic_bodies = bool(np.any(kinematic_mask))
            if model.joint_count and self.has_kinematic_bodies:
                joint_child = model.joint_child.numpy()
                joint_qd_start = model.joint_qd_start.numpy()
                joint_armature = model.joint_armature.numpy().copy()
                for joint_idx in range(model.joint_count):
                    if not kinematic_mask[joint_child[joint_idx]]:
                        continue
                    self.has_kinematic_joints = True
                    dof_start = int(joint_qd_start[joint_idx])
                    dof_end = int(joint_qd_start[joint_idx + 1])
                    joint_armature[dof_start:dof_end] = 1.0e10
                if self.has_kinematic_joints:
                    self.joint_armature_effective = wp.array(joint_armature, dtype=float, device=model.device)

    @override
    def notify_model_changed(self, flags: int) -> None:
        if flags & (SolverNotifyFlags.BODY_PROPERTIES | SolverNotifyFlags.JOINT_DOF_PROPERTIES):
            self._update_kinematic_state()
            self._mass_matrix_dirty = True

    def _compute_articulation_indices(self, model):
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

    def _allocate_model_aux_vars(self, model):
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

    def _allocate_state_aux_vars(self, model, target, requires_grad):
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
            target.joint_qd_internal_in = wp.empty_like(model.joint_qd, requires_grad=requires_grad)
            target.joint_qd_internal_out = wp.empty_like(model.joint_qd, requires_grad=requires_grad)
            target.joint_f_internal = wp.empty_like(model.joint_qd, requires_grad=requires_grad)
            target.joint_S_s = wp.empty(
                (model.joint_dof_count,),
                dtype=wp.spatial_vector,
                device=model.device,
                requires_grad=requires_grad,
            )

            # derived rigid body data (maximal coordinates)
            target.body_q_prev = wp.empty_like(model.body_q, requires_grad=requires_grad)
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
            target.body_f_ext = wp.zeros(
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
    ) -> None:
        requires_grad = state_in.requires_grad
        step_in_place = state_in is state_out

        # optionally create dynamical auxiliary variables
        if requires_grad:
            state_aug = state_out
        else:
            state_aug = self

        model = self.model
        descendant_body_q_prev = state_in.body_q

        if not getattr(state_aug, "_featherstone_augmented", False):
            self._allocate_state_aux_vars(model, state_aug, requires_grad)
        if control is None:
            control = model.control(clone_variables=False)

        with wp.ScopedTimer("simulate", False):
            if model.joint_count:
                # Keep articulated body poses current before any body/world-frame
                # force accumulation. Generalized-coordinate callers should not
                # need an explicit pre-step eval_fk() for FREE/DISTANCE wrenches.
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
                if step_in_place and self.descendant_free_distance_joint_indices is not None:
                    wp.copy(state_aug.body_q_prev, state_in.body_q)
                    descendant_body_q_prev = state_aug.body_q_prev

            particle_f = None
            body_f = None

            if state_in.particle_count:
                particle_f = state_in.particle_f

            if state_in.body_count:
                body_f = state_aug.body_f_ext
                wp.copy(body_f, state_in.body_f)
                wp.launch(
                    convert_body_force_com_to_origin,
                    dim=model.body_count,
                    inputs=[state_in.body_q, self.body_X_com],
                    outputs=[body_f],
                    device=model.device,
                )
                if model.joint_count:
                    wp.launch(
                        accumulate_free_distance_joint_f_to_body_force,
                        dim=model.joint_count,
                        inputs=[
                            model.joint_type,
                            model.joint_child,
                            model.joint_qd_start,
                            state_in.body_q,
                            self.body_X_com,
                            control.joint_f,
                        ],
                        outputs=[body_f],
                        device=model.device,
                    )

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
                wp.launch(
                    convert_free_distance_joint_qd_public_to_internal,
                    dim=model.joint_count,
                    inputs=[
                        model.joint_type,
                        model.joint_parent,
                        model.joint_child,
                        model.joint_qd_start,
                        model.joint_X_p,
                        state_in.body_q,
                        model.body_com,
                        state_in.joint_qd,
                    ],
                    outputs=[state_aug.joint_qd_internal_in],
                    device=model.device,
                )

                wp.launch(
                    convert_free_distance_joint_f_public_to_internal,
                    dim=model.joint_count,
                    inputs=[
                        model.joint_type,
                        model.joint_qd_start,
                        control.joint_f,
                    ],
                    outputs=[state_aug.joint_f_internal],
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
                        state_aug.joint_qd_internal_in,
                        model.joint_axis,
                        model.joint_dof_dim,
                        self.body_I_m,
                        state_in.body_q,
                        state_aug.body_q_com,
                        model.joint_X_p,
                        model.body_world,
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
                            contacts.rigid_contact_margin0,
                            contacts.rigid_contact_margin1,
                            contacts.rigid_contact_stiffness,
                            contacts.rigid_contact_damping,
                            contacts.rigid_contact_friction,
                            True,
                            self.friction_smoothing,
                        ],
                        outputs=[body_f],
                        device=model.device,
                    )

                if self.has_kinematic_bodies and body_f is not None:
                    wp.launch(
                        zero_kinematic_body_forces,
                        dim=model.body_count,
                        inputs=[model.body_flags],
                        outputs=[body_f],
                        device=model.device,
                    )

                # ``body_parent_f`` is populated below from the RNEA
                # backward-pass spatial forces. Zero it unconditionally so
                # bodies that are not the child of any joint (or models
                # without articulations) report a deterministic zero rather
                # than stale buffer contents.
                if state_out.body_parent_f is not None:
                    state_out.body_parent_f.zero_()

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
                            control.joint_target_pos,
                            control.joint_target_vel,
                            state_in.joint_q,
                            state_aug.joint_qd_internal_in,
                            state_aug.joint_f_internal,
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

                    # Optionally populate ``state_out.body_parent_f`` (incoming
                    # joint wrench per body in world frame at COM) from the
                    # RNEA backward-pass spatial forces. Only runs when the
                    # extended state attribute has been requested.
                    if state_out.body_parent_f is not None:
                        wp.launch(
                            compute_body_parent_f,
                            dim=model.body_count,
                            inputs=[
                                state_aug.body_q_com,
                                state_aug.body_f_s,
                                state_aug.body_ft_s,
                                body_f,
                            ],
                            outputs=[state_out.body_parent_f],
                            device=model.device,
                        )

                    # print("joint_tau:")
                    # print(state_aug.joint_tau.numpy())
                    # print("body_q:")
                    # print(state_in.body_q.numpy())
                    # print("body_qd:")
                    # print(state_in.body_qd.numpy())

                    if self._mass_matrix_dirty or self._step % self.update_mass_matrix_interval == 0:
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
                            R_tiled = self.joint_armature_effective.reshape((-1, self.dof_count))
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
                                        self.articulation_dof_start,
                                        self.H,
                                        self.joint_armature_effective,
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
                                    self.articulation_dof_start,
                                    self.H,
                                    self.joint_armature_effective,
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
                        self._mass_matrix_dirty = False

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

                    if self.has_kinematic_joints:
                        wp.launch(
                            zero_kinematic_joint_qdd,
                            dim=model.joint_count,
                            inputs=[model.joint_child, model.body_flags, model.joint_qd_start],
                            outputs=[state_aug.joint_qdd],
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
                        model.joint_parent,
                        model.joint_child,
                        model.joint_q_start,
                        model.joint_qd_start,
                        model.joint_dof_dim,
                        model.joint_X_c,
                        model.body_com,
                        state_in.joint_q,
                        state_aug.joint_qd_internal_in,
                        state_aug.joint_qdd,
                        dt,
                    ],
                    outputs=[state_out.joint_q, state_aug.joint_qd_internal_out],
                    device=model.device,
                )

                if self.has_kinematic_joints:
                    wp.launch(
                        copy_kinematic_joint_state,
                        dim=model.joint_count,
                        inputs=[
                            model.joint_child,
                            model.body_flags,
                            model.joint_q_start,
                            model.joint_qd_start,
                            state_in.joint_q,
                            state_aug.joint_qd_internal_in,
                        ],
                        outputs=[state_out.joint_q, state_aug.joint_qd_internal_out],
                        device=model.device,
                    )

                # Reconstruct public maximal coordinates once from the updated
                # generalized state while the solver still carries internal
                # FREE/DISTANCE speeds.
                eval_fk_with_velocity_conversion(
                    model,
                    state_out.joint_q,
                    state_aug.joint_qd_internal_out,
                    state_out,
                )

                if self.descendant_free_distance_joint_indices is not None:
                    wp.launch(
                        correct_free_distance_body_pose_from_world_twist,
                        dim=len(self.descendant_free_distance_joint_indices),
                        inputs=[
                            self.descendant_free_distance_joint_indices,
                            model.joint_child,
                            model.body_com,
                            descendant_body_q_prev,
                            state_out.body_qd,
                            state_out.body_q,
                            dt,
                        ],
                        device=model.device,
                    )

                    wp.launch(
                        reconstruct_free_distance_joint_q_from_body_pose,
                        dim=len(self.descendant_free_distance_joint_indices),
                        inputs=[
                            self.descendant_free_distance_joint_indices,
                            model.joint_parent,
                            model.joint_child,
                            model.joint_q_start,
                            model.joint_X_p,
                            model.joint_X_c,
                            state_out.body_q,
                        ],
                        outputs=[state_out.joint_q],
                        device=model.device,
                    )

                    eval_fk_with_velocity_conversion_from_joint_starts(
                        model,
                        self.descendant_free_distance_articulation_indices,
                        self.descendant_free_distance_refresh_joint_starts,
                        state_out.joint_q,
                        state_aug.joint_qd_internal_out,
                        state_out,
                    )

                wp.launch(
                    convert_free_distance_joint_qd_internal_to_public,
                    dim=model.joint_count,
                    inputs=[
                        model.joint_type,
                        model.joint_parent,
                        model.joint_child,
                        model.joint_qd_start,
                        model.joint_X_p,
                        state_out.body_q,
                        model.body_com,
                        state_aug.joint_qd_internal_out,
                    ],
                    outputs=[state_out.joint_qd],
                    device=model.device,
                )

            self.integrate_particles(model, state_in, state_out, dt)

            self._step += 1
