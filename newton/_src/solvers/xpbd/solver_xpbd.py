# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import warp as wp

from ...core.types import override
from ...sim import Contacts, Control, Model, State
from ..flags import SolverNotifyFlags
from ..solver import SolverBase
from .kernels import (
    accumulate_weighted_contact_impulse,
    apply_body_delta_velocities,
    apply_body_deltas,
    apply_joint_forces,
    apply_particle_deltas,
    apply_particle_shape_restitution,
    apply_rigid_restitution,
    bending_constraint,
    convert_contact_impulse_to_force,
    convert_joint_impulse_to_parent_f,
    copy_kinematic_body_state_kernel,
    solve_body_contact_positions,
    solve_body_joints,
    solve_particle_particle_contacts,
    solve_particle_shape_contacts,
    # solve_simple_body_joints,
    solve_springs,
    solve_tetrahedra,
    update_body_velocities,
)


class SolverXPBD(SolverBase):
    """An implicit integrator using eXtended Position-Based Dynamics (XPBD) for rigid and soft body simulation.

    References:
        - Miles Macklin, Matthias Müller, and Nuttapong Chentanez. 2016. XPBD: position-based simulation of compliant constrained dynamics. In Proceedings of the 9th International Conference on Motion in Games (MIG '16). Association for Computing Machinery, New York, NY, USA, 49-54. https://doi.org/10.1145/2994258.2994272
        - Matthias Müller, Miles Macklin, Nuttapong Chentanez, Stefan Jeschke, and Tae-Yong Kim. 2020. Detailed rigid body simulation with extended position based dynamics. In Proceedings of the ACM SIGGRAPH/Eurographics Symposium on Computer Animation (SCA '20). Eurographics Association, Goslar, DEU, Article 10, 1-12. https://doi.org/10.1111/cgf.14105

    After constructing :class:`Model`, :class:`State`, and :class:`Control` (optional) objects, this time-integrator
    may be used to advance the simulation state forward in time.

    Limitations:
        **Momentum conservation** -- When ``rigid_contact_con_weighting`` is
        enabled (the default), each body's positional correction is divided by
        its number of active contacts.  This improves convergence for stacking
        scenarios but means the solver does not conserve momentum at contacts.
        Reported per-contact forces (see :meth:`update_contacts`) are
        approximate: for contacts between two dynamic bodies the force is
        computed using the harmonic mean of the two bodies' contact counts,
        which is symmetric but not exact.

        **Reported parent-joint forces** (see :attr:`~newton.State.body_parent_f`,
        populated when the extended state attribute is requested) are also
        approximate.  XPBD applies relaxation factors
        (``joint_linear_relaxation``, ``joint_angular_relaxation``) to each
        joint constraint correction, and with a finite ``iterations`` count
        residual constraint error remains at end-of-step, so the reported
        wrench is the *applied* constraint reaction rather than the exact
        wrench needed to enforce the joint perfectly.  The convention matches
        :class:`~newton.solvers.SolverMuJoCo`: it is the spatial wrench
        transmitted from the parent through the inbound joint, in world frame
        at the child body's COM. In equilibrium this reaction counters all
        applied forces (gravity, contacts, ``State.body_f``, and the net
        effect of :attr:`~newton.Control.joint_f`) by Newton's third law.

    Joint limitations:
        - Supported joint types: PRISMATIC, REVOLUTE, BALL, FIXED, FREE, DISTANCE, D6.
          CABLE joints are not supported.
        - :attr:`~newton.Model.joint_enabled`,
          :attr:`~newton.Model.joint_target_ke`/:attr:`~newton.Model.joint_target_kd`, and
          :attr:`~newton.Control.joint_f` are supported.
          Joint limits are enforced as hard positional constraints (``joint_limit_ke``/``joint_limit_kd`` are not used).
        - :attr:`~newton.Model.joint_armature`, :attr:`~newton.Model.joint_friction`,
          :attr:`~newton.Model.joint_effort_limit`, :attr:`~newton.Model.joint_velocity_limit`,
          and :attr:`~newton.Model.joint_target_mode` are not supported.
        - Equality and mimic constraints are not supported.

        See :ref:`Joint feature support` for the full comparison across solvers.

    Example
    -------

    .. code-block:: python

        solver = newton.solvers.SolverXPBD(model)

        # simulation loop
        for i in range(100):
            solver.step(state_in, state_out, control, contacts, dt)
            state_in, state_out = state_out, state_in

    """

    def __init__(
        self,
        model: Model,
        iterations: int = 2,
        soft_body_relaxation: float = 0.9,
        soft_contact_relaxation: float = 0.9,
        joint_linear_relaxation: float = 0.7,
        joint_angular_relaxation: float = 0.4,
        joint_linear_compliance: float = 0.0,
        joint_angular_compliance: float = 0.0,
        rigid_contact_relaxation: float = 0.8,
        rigid_contact_con_weighting: bool = True,
        angular_damping: float = 0.0,
        enable_restitution: bool = False,
    ):
        super().__init__(model=model)
        self.iterations = iterations

        self.soft_body_relaxation = soft_body_relaxation
        self.soft_contact_relaxation = soft_contact_relaxation

        self.joint_linear_relaxation = joint_linear_relaxation
        self.joint_angular_relaxation = joint_angular_relaxation
        self.joint_linear_compliance = joint_linear_compliance
        self.joint_angular_compliance = joint_angular_compliance

        self.rigid_contact_relaxation = rigid_contact_relaxation
        self.rigid_contact_con_weighting = rigid_contact_con_weighting

        self.angular_damping = angular_damping

        self.enable_restitution = enable_restitution

        self.compute_body_velocity_from_position_delta = False

        self._init_kinematic_state()

        # helper variables to track constraint resolution vars
        self._particle_delta_counter = 0
        self._body_delta_counter = 0

        if model.particle_count > 1 and model.particle_grid is not None:
            # reserve space for the particle hash grid
            with wp.ScopedDevice(model.device):
                model.particle_grid.reserve(model.particle_count)

    @override
    def notify_model_changed(self, flags: int) -> None:
        if flags & (SolverNotifyFlags.BODY_PROPERTIES | SolverNotifyFlags.BODY_INERTIAL_PROPERTIES):
            self._refresh_kinematic_state()

    def copy_kinematic_body_state(self, model: Model, state_in: State, state_out: State):
        if model.body_count == 0:
            return
        wp.launch(
            kernel=copy_kinematic_body_state_kernel,
            dim=model.body_count,
            inputs=[model.body_flags, state_in.body_q, state_in.body_qd],
            outputs=[state_out.body_q, state_out.body_qd],
            device=model.device,
        )

    def _apply_particle_deltas(
        self,
        model: Model,
        state_in: State,
        state_out: State,
        particle_deltas: wp.array,
        dt: float,
    ):
        if state_in.requires_grad:
            particle_q = state_out.particle_q
            # allocate new particle arrays so gradients can be tracked correctly without overwriting
            new_particle_q = wp.empty_like(state_out.particle_q)
            new_particle_qd = wp.empty_like(state_out.particle_qd)
            self._particle_delta_counter += 1
        else:
            if self._particle_delta_counter == 0:
                particle_q = state_out.particle_q
                new_particle_q = state_in.particle_q
                new_particle_qd = state_in.particle_qd
            else:
                particle_q = state_in.particle_q
                new_particle_q = state_out.particle_q
                new_particle_qd = state_out.particle_qd
            self._particle_delta_counter = 1 - self._particle_delta_counter

        wp.launch(
            kernel=apply_particle_deltas,
            dim=model.particle_count,
            inputs=[
                self.particle_q_init,
                particle_q,
                model.particle_flags,
                particle_deltas,
                dt,
                model.particle_max_velocity,
            ],
            outputs=[new_particle_q, new_particle_qd],
            device=model.device,
        )

        if state_in.requires_grad:
            state_out.particle_q = new_particle_q
            state_out.particle_qd = new_particle_qd

        return new_particle_q, new_particle_qd

    def _apply_body_deltas(
        self,
        model: Model,
        state_in: State,
        state_out: State,
        body_deltas: wp.array,
        dt: float,
        rigid_contact_inv_weight: wp.array = None,
    ):
        with wp.ScopedTimer("apply_body_deltas", False):
            if state_in.requires_grad:
                body_q = state_out.body_q
                body_qd = state_out.body_qd
                new_body_q = wp.clone(body_q)
                new_body_qd = wp.clone(body_qd)
                self._body_delta_counter += 1
            else:
                if self._body_delta_counter == 0:
                    body_q = state_out.body_q
                    body_qd = state_out.body_qd
                    new_body_q = state_in.body_q
                    new_body_qd = state_in.body_qd
                else:
                    body_q = state_in.body_q
                    body_qd = state_in.body_qd
                    new_body_q = state_out.body_q
                    new_body_qd = state_out.body_qd
                self._body_delta_counter = 1 - self._body_delta_counter

            wp.launch(
                kernel=apply_body_deltas,
                dim=model.body_count,
                inputs=[
                    body_q,
                    body_qd,
                    model.body_com,
                    model.body_inertia,
                    self.body_inv_mass_effective,
                    self.body_inv_inertia_effective,
                    body_deltas,
                    rigid_contact_inv_weight,
                    dt,
                ],
                outputs=[
                    new_body_q,
                    new_body_qd,
                ],
                device=model.device,
            )

            if state_in.requires_grad:
                state_out.body_q = new_body_q
                state_out.body_qd = new_body_qd

        return new_body_q, new_body_qd

    @override
    def step(self, state_in: State, state_out: State, control: Control, contacts: Contacts, dt: float) -> None:
        requires_grad = state_in.requires_grad
        self._particle_delta_counter = 0
        self._body_delta_counter = 0

        model = self.model

        particle_q = None
        particle_qd = None
        particle_deltas = None

        body_q = None
        body_qd = None
        body_q_init = None
        body_qd_init = None
        body_deltas = None

        rigid_contact_inv_weight = None

        contact_impulse = None
        contact_impulse_iter = None

        if contacts:
            if self.rigid_contact_con_weighting:
                rigid_contact_inv_weight = wp.zeros(model.body_count, dtype=float, device=model.device)
            rigid_contact_inv_weight_init = None

            if contacts.force is not None:
                contact_impulse = wp.zeros(contacts.rigid_contact_max, dtype=wp.spatial_vector, device=model.device)
                contact_impulse_iter = wp.zeros(
                    contacts.rigid_contact_max, dtype=wp.spatial_vector, device=model.device
                )

        # Optional per-joint accumulated child-side spatial impulse, used to
        # populate ``state_out.body_parent_f`` after the iteration loop.
        joint_impulse = None
        if state_out.body_parent_f is not None and model.joint_count > 0:
            joint_impulse = wp.zeros(model.joint_count, dtype=wp.spatial_vector, device=model.device)

        if control is None:
            control = model.control(clone_variables=False)

        with wp.ScopedTimer("simulate", False):
            if model.particle_count:
                particle_q = state_out.particle_q
                particle_qd = state_out.particle_qd

                self.particle_q_init = wp.clone(state_in.particle_q)
                if self.enable_restitution:
                    self.particle_qd_init = wp.clone(state_in.particle_qd)
                particle_deltas = wp.empty_like(state_out.particle_qd)

                self.integrate_particles(model, state_in, state_out, dt)

                # Build/update the particle hash grid for particle-particle contact queries
                if model.particle_count > 1 and model.particle_grid is not None:
                    # Search radius must cover the maximum interaction distance used by the contact query
                    search_radius = model.particle_max_radius * 2.0 + model.particle_cohesion
                    with wp.ScopedDevice(model.device):
                        model.particle_grid.build(state_out.particle_q, radius=search_radius)

            if model.body_count:
                body_q = state_out.body_q
                body_qd = state_out.body_qd

                if self.compute_body_velocity_from_position_delta or self.enable_restitution:
                    body_q_init = wp.clone(state_in.body_q)
                    body_qd_init = wp.clone(state_in.body_qd)

                body_deltas = wp.empty_like(state_out.body_qd)

                body_f_tmp = state_in.body_f
                if model.joint_count:
                    # Avoid accumulating joint_f into the persistent state body_f buffer.
                    body_f_tmp = wp.clone(state_in.body_f)
                    wp.launch(
                        kernel=apply_joint_forces,
                        dim=model.joint_count,
                        inputs=[
                            state_in.body_q,
                            model.body_com,
                            model.joint_type,
                            model.joint_enabled,
                            model.joint_parent,
                            model.joint_child,
                            model.joint_X_p,
                            model.joint_X_c,
                            model.joint_qd_start,
                            model.joint_dof_dim,
                            model.joint_axis,
                            control.joint_f,
                        ],
                        outputs=[body_f_tmp],
                        device=model.device,
                    )

                if body_f_tmp is state_in.body_f:
                    self.integrate_bodies(model, state_in, state_out, dt, self.angular_damping)
                else:
                    body_f_prev = state_in.body_f
                    state_in.body_f = body_f_tmp
                    self.integrate_bodies(model, state_in, state_out, dt, self.angular_damping)
                    state_in.body_f = body_f_prev

            spring_constraint_lambdas = None
            if model.spring_count:
                spring_constraint_lambdas = wp.empty_like(model.spring_rest_length)
            edge_constraint_lambdas = None
            if model.edge_count:
                edge_constraint_lambdas = wp.empty_like(model.edge_rest_angle)

            for i in range(self.iterations):
                with wp.ScopedTimer(f"iteration_{i}", False):
                    if model.body_count:
                        if requires_grad and i > 0:
                            body_deltas = wp.zeros_like(body_deltas)
                        else:
                            body_deltas.zero_()

                    if model.particle_count:
                        if requires_grad and i > 0:
                            particle_deltas = wp.zeros_like(particle_deltas)
                        else:
                            particle_deltas.zero_()

                        # particle-rigid body contacts (besides ground plane)
                        if model.shape_count:
                            wp.launch(
                                kernel=solve_particle_shape_contacts,
                                dim=contacts.soft_contact_max,
                                inputs=[
                                    particle_q,
                                    particle_qd,
                                    model.particle_inv_mass,
                                    model.particle_radius,
                                    model.particle_flags,
                                    body_q,
                                    body_qd,
                                    model.body_com,
                                    self.body_inv_mass_effective,
                                    self.body_inv_inertia_effective,
                                    model.shape_body,
                                    model.shape_material_mu,
                                    model.soft_contact_mu,
                                    model.particle_adhesion,
                                    contacts.soft_contact_count,
                                    contacts.soft_contact_particle,
                                    contacts.soft_contact_shape,
                                    contacts.soft_contact_body_pos,
                                    contacts.soft_contact_body_vel,
                                    contacts.soft_contact_normal,
                                    contacts.soft_contact_max,
                                    dt,
                                    self.soft_contact_relaxation,
                                ],
                                # outputs
                                outputs=[particle_deltas, body_deltas],
                                device=model.device,
                            )

                        if model.particle_max_radius > 0.0 and model.particle_count > 1:
                            # assert model.particle_grid.reserved, "model.particle_grid must be built, see HashGrid.build()"
                            assert model.particle_grid is not None
                            wp.launch(
                                kernel=solve_particle_particle_contacts,
                                dim=model.particle_count,
                                inputs=[
                                    model.particle_grid.id,
                                    particle_q,
                                    particle_qd,
                                    model.particle_inv_mass,
                                    model.particle_radius,
                                    model.particle_flags,
                                    model.particle_mu,
                                    model.particle_cohesion,
                                    model.particle_max_radius,
                                    dt,
                                    self.soft_contact_relaxation,
                                ],
                                outputs=[particle_deltas],
                                device=model.device,
                            )

                        # distance constraints
                        if model.spring_count:
                            spring_constraint_lambdas.zero_()
                            wp.launch(
                                kernel=solve_springs,
                                dim=model.spring_count,
                                inputs=[
                                    particle_q,
                                    particle_qd,
                                    model.particle_inv_mass,
                                    model.spring_indices,
                                    model.spring_rest_length,
                                    model.spring_stiffness,
                                    model.spring_damping,
                                    dt,
                                    spring_constraint_lambdas,
                                ],
                                outputs=[particle_deltas],
                                device=model.device,
                            )

                        # bending constraints
                        if model.edge_count:
                            edge_constraint_lambdas.zero_()
                            wp.launch(
                                kernel=bending_constraint,
                                dim=model.edge_count,
                                inputs=[
                                    particle_q,
                                    particle_qd,
                                    model.particle_inv_mass,
                                    model.edge_indices,
                                    model.edge_rest_angle,
                                    model.edge_bending_properties,
                                    dt,
                                    edge_constraint_lambdas,
                                ],
                                outputs=[particle_deltas],
                                device=model.device,
                            )

                        # tetrahedral FEM
                        if model.tet_count:
                            wp.launch(
                                kernel=solve_tetrahedra,
                                dim=model.tet_count,
                                inputs=[
                                    particle_q,
                                    particle_qd,
                                    model.particle_inv_mass,
                                    model.tet_indices,
                                    model.tet_poses,
                                    model.tet_activations,
                                    model.tet_materials,
                                    dt,
                                    self.soft_body_relaxation,
                                ],
                                outputs=[particle_deltas],
                                device=model.device,
                            )

                        particle_q, particle_qd = self._apply_particle_deltas(
                            model, state_in, state_out, particle_deltas, dt
                        )

                    # handle rigid bodies
                    # ----------------------------

                    # Solve rigid contact constraints
                    if model.body_count and contacts is not None:
                        if self.rigid_contact_con_weighting:
                            rigid_contact_inv_weight.zero_()

                        if contact_impulse_iter is not None:
                            contact_impulse_iter.zero_()

                        wp.launch(
                            kernel=solve_body_contact_positions,
                            dim=contacts.rigid_contact_max,
                            inputs=[
                                body_q,
                                body_qd,
                                model.body_flags,
                                model.body_com,
                                self.body_inv_mass_effective,
                                self.body_inv_inertia_effective,
                                model.shape_body,
                                contacts.rigid_contact_count,
                                contacts.rigid_contact_point0,
                                contacts.rigid_contact_point1,
                                contacts.rigid_contact_offset0,
                                contacts.rigid_contact_offset1,
                                contacts.rigid_contact_normal,
                                contacts.rigid_contact_margin0,
                                contacts.rigid_contact_margin1,
                                contacts.rigid_contact_shape0,
                                contacts.rigid_contact_shape1,
                                model.shape_material_mu,
                                model.shape_material_mu_torsional,
                                model.shape_material_mu_rolling,
                                self.rigid_contact_relaxation,
                                dt,
                            ],
                            outputs=[
                                body_deltas,
                                rigid_contact_inv_weight,
                                contact_impulse_iter,
                            ],
                            device=model.device,
                        )

                        if contact_impulse_iter is not None:
                            wp.launch(
                                kernel=accumulate_weighted_contact_impulse,
                                dim=contacts.rigid_contact_max,
                                inputs=[
                                    contacts.rigid_contact_count,
                                    contact_impulse_iter,
                                    contacts.rigid_contact_shape0,
                                    contacts.rigid_contact_shape1,
                                    model.shape_body,
                                    rigid_contact_inv_weight,
                                ],
                                outputs=[contact_impulse],
                                device=model.device,
                            )

                        # if model.rigid_contact_count.numpy()[0] > 0:
                        #     print("rigid_contact_count:", model.rigid_contact_count.numpy().flatten())
                        #     # print("rigid_active_contact_distance:", rigid_active_contact_distance.numpy().flatten())
                        #     # print("rigid_active_contact_point0:", rigid_active_contact_point0.numpy().flatten())
                        #     # print("rigid_active_contact_point1:", rigid_active_contact_point1.numpy().flatten())
                        #     print("body_deltas:", body_deltas.numpy().flatten())

                        # print(rigid_active_contact_distance.numpy().flatten())

                        if self.enable_restitution and i == 0:
                            # remember contact constraint weighting from the first iteration
                            if self.rigid_contact_con_weighting:
                                rigid_contact_inv_weight_init = wp.clone(rigid_contact_inv_weight)
                            else:
                                rigid_contact_inv_weight_init = None

                        body_q, body_qd = self._apply_body_deltas(
                            model, state_in, state_out, body_deltas, dt, rigid_contact_inv_weight
                        )

                    if model.joint_count:
                        if requires_grad:
                            body_deltas = wp.zeros_like(body_deltas)
                        else:
                            body_deltas.zero_()

                        wp.launch(
                            kernel=solve_body_joints,
                            dim=model.joint_count,
                            inputs=[
                                body_q,
                                body_qd,
                                model.body_com,
                                self.body_inv_mass_effective,
                                self.body_inv_inertia_effective,
                                model.joint_type,
                                model.joint_enabled,
                                model.joint_parent,
                                model.joint_child,
                                model.joint_X_p,
                                model.joint_X_c,
                                model.joint_limit_lower,
                                model.joint_limit_upper,
                                model.joint_qd_start,
                                model.joint_dof_dim,
                                model.joint_axis,
                                control.joint_target_pos,
                                control.joint_target_vel,
                                model.joint_target_ke,
                                model.joint_target_kd,
                                self.joint_linear_compliance,
                                self.joint_angular_compliance,
                                self.joint_angular_relaxation,
                                self.joint_linear_relaxation,
                                dt,
                            ],
                            outputs=[body_deltas, joint_impulse],
                            device=model.device,
                        )

                        body_q, body_qd = self._apply_body_deltas(model, state_in, state_out, body_deltas, dt)

            self._contact_impulse = contact_impulse
            self._contact_impulse_capacity = contacts.rigid_contact_max if contacts is not None else 0
            self._last_dt = dt

            # Populate optional ``state_out.body_parent_f`` (incoming joint
            # wrench per body) from the per-joint accumulated child-side
            # impulse.  Bodies without an inbound joint (roots / free bodies)
            # remain zero-initialized, matching MuJoCo's behavior.
            if state_out.body_parent_f is not None:
                state_out.body_parent_f.zero_()
                if joint_impulse is not None:
                    wp.launch(
                        kernel=convert_joint_impulse_to_parent_f,
                        dim=model.joint_count,
                        inputs=[
                            joint_impulse,
                            model.joint_enabled,
                            model.joint_type,
                            model.joint_child,
                            dt,
                        ],
                        outputs=[state_out.body_parent_f],
                        device=model.device,
                    )

            if model.particle_count:
                if particle_q.ptr != state_out.particle_q.ptr:
                    state_out.particle_q.assign(particle_q)
                    state_out.particle_qd.assign(particle_qd)

            if model.body_count:
                if body_q.ptr != state_out.body_q.ptr:
                    state_out.body_q.assign(body_q)
                    state_out.body_qd.assign(body_qd)

            # update body velocities from position changes
            if self.compute_body_velocity_from_position_delta and model.body_count and not requires_grad:
                # causes gradient issues (probably due to numerical problems
                # when computing velocities from position changes)
                if requires_grad:
                    out_body_qd = wp.clone(state_out.body_qd)
                else:
                    out_body_qd = state_out.body_qd

                # update body velocities
                wp.launch(
                    kernel=update_body_velocities,
                    dim=model.body_count,
                    inputs=[state_out.body_q, body_q_init, model.body_com, dt],
                    outputs=[out_body_qd],
                    device=model.device,
                )

            if self.enable_restitution and contacts is not None:
                if model.particle_count:
                    wp.launch(
                        kernel=apply_particle_shape_restitution,
                        dim=contacts.soft_contact_max,
                        inputs=[
                            particle_qd,
                            self.particle_q_init,
                            self.particle_qd_init,
                            model.particle_radius,
                            model.particle_flags,
                            body_q,
                            body_q_init,
                            body_qd,
                            body_qd_init,
                            model.body_com,
                            model.shape_body,
                            model.particle_adhesion,
                            model.soft_contact_restitution,
                            contacts.soft_contact_count,
                            contacts.soft_contact_particle,
                            contacts.soft_contact_shape,
                            contacts.soft_contact_body_pos,
                            contacts.soft_contact_body_vel,
                            contacts.soft_contact_normal,
                            contacts.soft_contact_max,
                        ],
                        outputs=[state_out.particle_qd],
                        device=model.device,
                    )

                if model.body_count:
                    body_deltas.zero_()

                    wp.launch(
                        kernel=apply_rigid_restitution,
                        dim=contacts.rigid_contact_max,
                        inputs=[
                            state_out.body_q,
                            state_out.body_qd,
                            body_q_init,
                            body_qd_init,
                            model.body_com,
                            self.body_inv_mass_effective,
                            self.body_inv_inertia_effective,
                            model.body_world,
                            model.shape_body,
                            contacts.rigid_contact_count,
                            contacts.rigid_contact_normal,
                            contacts.rigid_contact_shape0,
                            contacts.rigid_contact_shape1,
                            model.shape_material_restitution,
                            contacts.rigid_contact_point0,
                            contacts.rigid_contact_point1,
                            contacts.rigid_contact_offset0,
                            contacts.rigid_contact_offset1,
                            contacts.rigid_contact_margin0,
                            contacts.rigid_contact_margin1,
                            rigid_contact_inv_weight_init,
                            model.gravity,
                            dt,
                        ],
                        outputs=[
                            body_deltas,
                        ],
                        device=model.device,
                    )

                    wp.launch(
                        kernel=apply_body_delta_velocities,
                        dim=model.body_count,
                        inputs=[
                            body_deltas,
                        ],
                        outputs=[state_out.body_qd],
                        device=model.device,
                    )

            if model.body_count:
                self.copy_kinematic_body_state(model, state_in, state_out)

    @override
    def update_contacts(self, contacts: Contacts, state: State | None = None) -> None:
        """Populate ``contacts.force`` from XPBD contact impulses accumulated during the last :meth:`step`.

        Both force [N] and torque [N·m] components are written.  The torque
        includes torsional and rolling friction contributions that cannot be
        reconstructed from the linear force alone.

        When ``rigid_contact_con_weighting`` is enabled, the raw per-contact
        impulse is scaled to reflect the ``1/N`` correction that
        ``apply_body_deltas`` applies.  For contacts between a dynamic and a
        kinematic body, ``N`` is the dynamic body's contact count.  For
        contacts between two dynamic bodies, the harmonic mean
        ``2/(N_a + N_b)`` is used so that the reported force is symmetric with
        respect to body ordering.  This is an approximation -- the solver
        applies ``1/N_a`` and ``1/N_b`` independently to each side, so no
        single scalar can exactly represent both.

        Args:
            contacts: :class:`Contacts` object whose :attr:`~Contacts.force` buffer will be written.
                Must have been created with ``"force"`` in its requested attributes and must
                match the :class:`Contacts` instance (same ``rigid_contact_max``) passed to
                the preceding :meth:`step`.
            state: Unused (accepted for API compatibility with :class:`SolverBase`).

        Raises:
            ValueError: If ``contacts.force`` is ``None`` (not requested), if no step has been run yet,
                or if the contacts capacity does not match the one used in the last :meth:`step`.
        """
        if contacts.force is None:
            raise ValueError(
                "contacts.force is not allocated. Call model.request_contact_attributes('force') "
                "before creating the Contacts object."
            )
        if not hasattr(self, "_contact_impulse") or self._contact_impulse is None:
            raise ValueError("No contact impulse data available. Call step() before update_contacts().")
        if contacts.rigid_contact_max != self._contact_impulse_capacity:
            raise ValueError(
                f"Contacts capacity mismatch: update_contacts() received rigid_contact_max="
                f"{contacts.rigid_contact_max}, but step() used {self._contact_impulse_capacity}. "
                f"Pass the same Contacts instance to both step() and update_contacts()."
            )

        contacts.force.zero_()

        wp.launch(
            kernel=convert_contact_impulse_to_force,
            dim=contacts.rigid_contact_max,
            inputs=[
                contacts.rigid_contact_count,
                self._contact_impulse,
                self._last_dt,
            ],
            outputs=[contacts.force],
            device=self.model.device,
        )
