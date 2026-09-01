# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import warnings

import warp as wp

from ...core.types import override
from ...sim import Contacts, Control, Model, ModelFlags, State
from ...utils.deprecation import deprecate_nonkeyword_arguments
from ..coupled.interface import CouplingInterface
from ..solver import SolverBase
from . import kernels, restitution_kernels
from .kernels import (
    accumulate_weighted_contact_impulse,
    apply_body_delta_velocities,
    apply_body_deltas,
    apply_joint_forces,
    apply_particle_deltas,
    apply_particle_shape_restitution,
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
from .restitution_kernels import (
    RESTITUTION_MANIFOLD_MAX_CONTACTS,
    apply_restitution_deltas,
    build_restitution_manifolds,
    mark_restitution_contacts,
    select_manifold_contacts,
    solve_manifold_restitution,
)

_COMPUTE_BODY_VELOCITY_DEPRECATION_MSG = (
    "SolverXPBD.compute_body_velocity_from_position_delta is deprecated in Newton 1.6 and will be removed in 1.7 "
    "or later. Leave it at False because XPBD now updates rigid-body velocities incrementally after every position "
    "correction."
)


class SolverXPBD(SolverBase, CouplingInterface):
    """An implicit integrator using eXtended Position-Based Dynamics (XPBD) for rigid and soft body simulation.

    References:
        - Miles Macklin, Matthias Müller, and Nuttapong Chentanez. 2016. XPBD: position-based simulation of compliant constrained dynamics. In Proceedings of the 9th International Conference on Motion in Games (MIG '16). Association for Computing Machinery, New York, NY, USA, 49-54. https://doi.org/10.1145/2994258.2994272
        - Matthias Müller, Miles Macklin, Nuttapong Chentanez, Stefan Jeschke, and Tae-Yong Kim. 2020. Detailed rigid body simulation with extended position based dynamics. In Proceedings of the ACM SIGGRAPH/Eurographics Symposium on Computer Animation (SCA '20). Eurographics Association, Goslar, DEU, Article 10, 1-12. https://doi.org/10.1111/cgf.14105

    After constructing :class:`Model`, :class:`State`, and :class:`Control` (optional) objects, this time-integrator
    may be used to advance the simulation state forward in time.

    Rigid-body velocities use Newton's public ``(v_com_world, omega_world)`` convention throughout integration and
    constraint projection. Enabling restitution adds velocity-level contact constraints without changing that
    integration path for other bodies.

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
        populated when the extended state attribute is requested) are
        approximate.  XPBD applies relaxation factors
        (``joint_linear_relaxation``, ``joint_angular_relaxation``) to each
        joint constraint correction, and with a finite ``iterations`` count
        residual constraint error remains at end-of-step, so the reported
        wrench is the *applied* constraint reaction rather than the exact
        wrench needed to enforce the joint perfectly.  The convention matches
        :class:`~newton.solvers.SolverFeatherstone` and
        :class:`~newton.solvers.SolverMuJoCo`: it is the spatial wrench
        transmitted from the parent through the inbound joint, in world frame
        at the child body's COM, **including** both the constraint reaction
        and the body-frame contribution of :attr:`~newton.Control.joint_f`.
        In equilibrium this wrench counters all applied forces (gravity,
        contacts, ``State.body_f``) by Newton's third law.

    Joint limitations:
        - Supported joint types: PRISMATIC, REVOLUTE, BALL, FIXED, FREE, DISTANCE, D6.
          ROD joints are not supported.
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

        solver = newton.solvers.SolverXPBD(model, enable_restitution=True)

        # simulation loop
        for i in range(100):
            solver.step(state_in, state_out, control, contacts, dt)
            state_in, state_out = state_out, state_in

    """

    @deprecate_nonkeyword_arguments
    def __init__(
        self,
        model: Model,
        *,
        iterations: int = 2,
        soft_body_relaxation: float = 0.9,
        soft_contact_relaxation: float = 0.9,
        joint_linear_relaxation: float = 0.7,
        joint_angular_relaxation: float = 0.4,
        joint_linear_compliance: float = 0.0,
        joint_angular_compliance: float = 0.0,
        rigid_contact_relaxation: float = 0.8,
        rigid_contact_restitution_iterations: int = 2,
        rigid_contact_con_weighting: bool = True,
        angular_damping: float = 0.0,
        enable_restitution: bool = False,
        deterministic: wp.DeterministicMode | None = None,
    ):
        """Initialize the XPBD solver.

        Args:
            model: Simulation model to integrate.
            iterations: Number of constraint-solver iterations per time step. Defaults to 2.
            soft_body_relaxation: Relaxation factor applied to tetrahedral constraint corrections
                [dimensionless]. Defaults to 0.9.
            soft_contact_relaxation: Relaxation factor applied to particle-particle and particle-shape contact
                corrections [dimensionless]. Defaults to 0.9.
            joint_linear_relaxation: Relaxation factor applied to linear joint constraint corrections
                [dimensionless]. Defaults to 0.7.
            joint_angular_relaxation: Relaxation factor applied to angular joint constraint corrections
                [dimensionless]. Defaults to 0.4.
            joint_linear_compliance: Compliance shared by linear joint constraints [m/N]. Defaults to 0.0.
            joint_angular_compliance: Compliance shared by angular joint constraints [rad/(N·m)]. Defaults to 0.0.
            rigid_contact_relaxation: Relaxation factor applied to rigid contact constraint corrections
                [dimensionless]. Defaults to 0.8.
            rigid_contact_restitution_iterations: Number of outer iterations of the rigid-body restitution
                velocity solve. Each outer iteration solves every contact manifold (body pair) independently
                with a fixed number of inner Gauss-Seidel sweeps, then couples manifolds by averaging the
                resulting velocity changes per body, so values above 1 primarily matter when a body
                participates in several manifolds (or when a large manifold leaves its inner sweeps
                under-converged). Defaults to 2.
            rigid_contact_con_weighting: Whether to divide each rigid body's contact correction by its number of
                active contacts. Defaults to ``True``.
            angular_damping: Rigid-body angular velocity damping coefficient [1/s]. Defaults to 0.0.
            enable_restitution: Whether to apply restitution to rigid and particle-shape contacts after the
                positional solve. Defaults to ``False``.
            deterministic: Opt-in determinism for this solver's atomic-emitting
                kernel module. Pass a :class:`warp.DeterministicMode`, or
                ``None`` (default) to inherit the current
                ``wp.config.deterministic`` mode.
        """
        super().__init__(model=model)
        effective_deterministic = deterministic if deterministic is not None else wp.config.deterministic
        module_options = {
            "deterministic": effective_deterministic,
            "deterministic_max_records": 0,
        }
        self._set_module_options(module_options, module=kernels)
        self._restitution_module_options = module_options

        self.iterations = iterations

        self.soft_body_relaxation = soft_body_relaxation
        self.soft_contact_relaxation = soft_contact_relaxation

        self.joint_linear_relaxation = joint_linear_relaxation
        self.joint_angular_relaxation = joint_angular_relaxation
        self.joint_linear_compliance = joint_linear_compliance
        self.joint_angular_compliance = joint_angular_compliance

        self.rigid_contact_relaxation = rigid_contact_relaxation
        if rigid_contact_restitution_iterations < 1:
            raise ValueError("rigid_contact_restitution_iterations must be at least 1")
        self.rigid_contact_restitution_iterations = rigid_contact_restitution_iterations
        # Eight local sweeps converge flat manifolds in one pass; outer iterations couple manifolds.
        self._restitution_manifold_inner_iterations = 8
        self.rigid_contact_con_weighting = rigid_contact_con_weighting

        self.angular_damping = angular_damping

        self.enable_restitution = enable_restitution
        self._rigid_restitution_enabled = False
        self._refresh_rigid_restitution_enabled()
        self._compute_body_velocity_from_position_delta = False

        self._init_kinematic_state()

        # helper variables to track constraint resolution vars
        self._particle_delta_counter = 0
        self._body_delta_counter = 0

        if model.particle_count > 1 and model.particle_grid is not None:
            # reserve space for the particle hash grid
            with wp.ScopedDevice(model.device):
                model.particle_grid.reserve(model.particle_count)

    @property
    def compute_body_velocity_from_position_delta(self) -> bool:
        """Whether to reconstruct rigid-body velocities after position projection.

        .. deprecated:: 1.6
            Leave this setting at ``False`` because XPBD now maintains rigid-body
            velocities incrementally. ``True`` temporarily retains the legacy
            full-step velocity reconstruction for compatibility.
        """
        warnings.warn(_COMPUTE_BODY_VELOCITY_DEPRECATION_MSG, DeprecationWarning, stacklevel=2)
        return self._compute_body_velocity_from_position_delta

    @compute_body_velocity_from_position_delta.setter
    def compute_body_velocity_from_position_delta(self, value: bool) -> None:
        warnings.warn(_COMPUTE_BODY_VELOCITY_DEPRECATION_MSG, DeprecationWarning, stacklevel=2)
        self._compute_body_velocity_from_position_delta = value

    @override
    def notify_model_changed(self, flags: ModelFlags | int) -> None:
        """Refresh cached body data after model properties change.

        Effective inverse masses and inertia tensors are refreshed for body-property changes. The cached restitution
        state is refreshed for shape-property changes. Other flags are ignored.

        Args:
            flags: Bitmask of :class:`~newton.ModelFlags` or custom ``int`` bits indicating which model properties
                changed.
        """
        self._ensure_restitution_module_options()
        self._apply_module_options()
        if flags & (ModelFlags.BODY_PROPERTIES | ModelFlags.BODY_INERTIAL_PROPERTIES):
            self._refresh_kinematic_state()
        if self.enable_restitution and flags & ModelFlags.SHAPE_PROPERTIES:
            self._refresh_rigid_restitution_enabled()

    def _refresh_rigid_restitution_enabled(self) -> None:
        restitution = self.model.shape_material_restitution
        self._rigid_restitution_enabled = restitution is not None and restitution.size > 0

    def _ensure_restitution_module_options(self) -> None:
        if self.enable_restitution and restitution_kernels not in self._module_options:
            self._set_module_options(self._restitution_module_options, module=restitution_kernels)
            # Registration may advance the shared revision while this solver's
            # core module options are stale, so force a complete reapplication.
            self._applied_module_options_revision = -1

    @override
    def coupling_supports_inertial_property_refresh(self) -> bool:
        """Return whether inertial properties can be refreshed during graph capture.

        Returns:
            ``True`` because :meth:`notify_model_changed` refreshes the derived inertial buffers with device work.
        """
        return True

    def copy_kinematic_body_state(self, model: Model, state_in: State, state_out: State):
        """Copy kinematic body poses and velocities from an input state to an output state.

        Args:
            model: Simulation model that owns the body data.
            state_in: State containing the source kinematic body poses and velocities.
            state_out: State that receives the kinematic body poses and velocities.
        """
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
    def step(
        self,
        state_in: State,
        state_out: State,
        control: Control | None,
        contacts: Contacts | None,
        dt: float,
    ) -> None:
        """Advance the simulation state by one time step using XPBD.

        Args:
            state_in: State at the beginning of the time step.
            state_out: State that receives the simulation result.
            control: Control inputs. If ``None``, the model's default control values are used.
            contacts: Contact data populated by :meth:`~newton.CollisionPipeline.collide` and allocated with
                :meth:`~newton.CollisionPipeline.contacts`. If ``None``, rigid and particle-shape contact handling
                is skipped; particle-particle contacts and model constraints are still solved.
            dt: Time step size [s].
        """
        self._ensure_restitution_module_options()
        self._apply_module_options()
        requires_grad = state_in.requires_grad
        self._particle_delta_counter = 0
        self._body_delta_counter = 0

        model = self.model

        particle_q = None
        particle_qd = None
        particle_deltas = None

        body_q = None
        body_qd = None
        body_q_step_start = None
        body_q_pre_solve = None
        body_qd_pre_solve = None
        body_deltas = None

        rigid_contact_inv_weight = None
        restitution_contact_active = None
        restitution_manifold_key = None
        restitution_manifold_size = None
        restitution_manifold_contact = None
        restitution_manifold_head = None
        restitution_manifold_total = None
        restitution_contact_next = None
        restitution_contact_pos_depth = None
        restitution_contact_sel_score = None
        restitution_body_manifold_count = None
        restitution_contact_n_K = None
        restitution_contact_axn_lo_target = None
        restitution_contact_axn_hi_sigma = None

        contact_impulse = None
        contact_impulse_iter = None

        if contacts:
            if self.rigid_contact_con_weighting:
                rigid_contact_inv_weight = wp.zeros(model.body_count, dtype=float, device=model.device)
            if self.enable_restitution and self._rigid_restitution_enabled and model.body_count:
                restitution_contact_active = wp.zeros(contacts.rigid_contact_max, dtype=wp.int32, device=model.device)
                # manifold hash table (one slot may host every contact, so
                # capacity == contact capacity guarantees insertion succeeds)
                restitution_manifold_key = wp.zeros(contacts.rigid_contact_max, dtype=wp.int64, device=model.device)
                restitution_manifold_size = wp.zeros(contacts.rigid_contact_max, dtype=wp.int32, device=model.device)
                restitution_manifold_contact = wp.empty(
                    contacts.rigid_contact_max * RESTITUTION_MANIFOLD_MAX_CONTACTS,
                    dtype=wp.int32,
                    device=model.device,
                )
                restitution_manifold_head = wp.zeros(contacts.rigid_contact_max, dtype=wp.int32, device=model.device)
                restitution_manifold_total = wp.zeros(contacts.rigid_contact_max, dtype=wp.int32, device=model.device)
                restitution_contact_next = wp.empty(contacts.rigid_contact_max, dtype=wp.int32, device=model.device)
                restitution_contact_pos_depth = wp.empty(contacts.rigid_contact_max, dtype=wp.vec4, device=model.device)
                restitution_contact_sel_score = wp.empty(contacts.rigid_contact_max, dtype=float, device=model.device)
                restitution_body_manifold_count = wp.zeros(model.body_count, dtype=float, device=model.device)
                # per-contact solve records cached by build_restitution_manifolds
                restitution_contact_n_K = wp.empty(contacts.rigid_contact_max, dtype=wp.vec4, device=model.device)
                restitution_contact_axn_lo_target = wp.empty(
                    contacts.rigid_contact_max, dtype=wp.vec4, device=model.device
                )
                restitution_contact_axn_hi_sigma = wp.empty(
                    contacts.rigid_contact_max, dtype=wp.vec4, device=model.device
                )

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
                particle_deltas = wp.empty_like(state_out.particle_qd)

                self.integrate_particles(model, state_in, state_out, dt)
                if self.enable_restitution:
                    self.particle_qd_init = wp.clone(state_out.particle_qd)

                # Build/update the particle hash grid for particle-particle contact queries
                if model.particle_count > 1 and model.particle_grid is not None:
                    # Search radius must cover the maximum interaction distance used by the contact query
                    search_radius = model.particle_max_radius * 2.0 + model.particle_cohesion
                    with wp.ScopedDevice(model.device):
                        model.particle_grid.build(state_out.particle_q, radius=search_radius)

            if model.body_count:
                body_q = state_out.body_q
                body_qd = state_out.body_qd

                if self._compute_body_velocity_from_position_delta and not requires_grad:
                    body_q_step_start = wp.clone(state_in.body_q)

                body_deltas = wp.empty_like(state_out.body_qd)

                body_f_tmp = state_in.body_f
                if model.joint_count:
                    # Avoid accumulating joint_f into the persistent state body_f buffer.
                    body_f_tmp = wp.clone(state_in.body_f)
                    # ``joint_impulse`` (may be ``None`` when ``body_parent_f``
                    # was not requested) accumulates both the joint_f wrench
                    # contribution recorded here and the constraint-correction
                    # contribution added by :func:`solve_body_joints` inside
                    # the iteration loop.  Together they recover the total
                    # wrench transmitted to the child body, matching the
                    # :attr:`State.body_parent_f` convention.
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
                            dt,
                        ],
                        outputs=[body_f_tmp, joint_impulse],
                        device=model.device,
                    )

                if body_f_tmp is state_in.body_f:
                    self.integrate_bodies(model, state_in, state_out, dt, self.angular_damping)
                else:
                    body_f_prev = state_in.body_f
                    state_in.body_f = body_f_tmp
                    self.integrate_bodies(model, state_in, state_out, dt, self.angular_damping)
                    state_in.body_f = body_f_prev

                if self.enable_restitution:
                    body_q_pre_solve = wp.clone(state_out.body_q)
                    body_qd_pre_solve = wp.clone(state_out.body_qd)

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
                        if model.shape_count and contacts is not None:
                            contacts._assert_particle_only_soft_contacts("SolverXPBD")
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
                                    model.body_flags,
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
                                    control.tet_activations,
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

                        if restitution_contact_active is not None:
                            wp.launch(
                                kernel=mark_restitution_contacts,
                                dim=contacts.rigid_contact_max,
                                inputs=[
                                    body_q,
                                    model.shape_body,
                                    contacts.rigid_contact_count,
                                    contacts.rigid_contact_point0,
                                    contacts.rigid_contact_point1,
                                    contacts.rigid_contact_normal,
                                    contacts.rigid_contact_margin0,
                                    contacts.rigid_contact_margin1,
                                    contacts.rigid_contact_shape0,
                                    contacts.rigid_contact_shape1,
                                ],
                                outputs=[restitution_contact_active],
                                device=model.device,
                            )

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
                                model.joint_target_q_start,
                                model.joint_dof_dim,
                                model.joint_axis,
                                control.joint_target_q,
                                control.joint_target_qd,
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

            if self._compute_body_velocity_from_position_delta and model.body_count and not requires_grad:
                assert body_q_step_start is not None
                wp.launch(
                    kernel=update_body_velocities,
                    dim=model.body_count,
                    inputs=[state_out.body_q, body_q_step_start, model.body_com, dt],
                    outputs=[state_out.body_qd],
                    device=model.device,
                )

            # Rigid integration and every positional correction update all
            # bodies' public COM-referenced velocities incrementally. Velocity
            # constraints consume that same convention without selecting a
            # different integration path when restitution is enabled.
            body_qd_for_restitution = state_out.body_qd

            if self.enable_restitution and contacts is not None:
                if model.particle_count:
                    # Grad-enabled steps write into a cloned buffer to avoid
                    # mutating a recorded array in place.
                    assert particle_qd is not None
                    particle_qd_with_restitution = wp.clone(particle_qd) if requires_grad else state_out.particle_qd
                    wp.launch(
                        kernel=apply_particle_shape_restitution,
                        dim=contacts.soft_contact_max,
                        inputs=[
                            particle_qd,
                            self.particle_q_init,
                            self.particle_qd_init,
                            model.particle_radius,
                            model.particle_flags,
                            model.particle_world,
                            body_q,
                            body_q_pre_solve,
                            body_qd_for_restitution,
                            body_qd_pre_solve,
                            model.body_com,
                            model.shape_body,
                            model.particle_adhesion,
                            model.soft_contact_restitution,
                            model.gravity,
                            dt,
                            contacts.soft_contact_count,
                            contacts.soft_contact_particle,
                            contacts.soft_contact_shape,
                            contacts.soft_contact_body_pos,
                            contacts.soft_contact_body_vel,
                            contacts.soft_contact_normal,
                            contacts.soft_contact_max,
                        ],
                        outputs=[particle_qd_with_restitution],
                        device=model.device,
                    )
                    if requires_grad:
                        state_out.particle_qd = particle_qd_with_restitution

                if model.body_count and self._rigid_restitution_enabled:
                    # Group contacts that can fire restitution into manifolds
                    # (canonical body pairs) and cache their solve records.
                    # The collision pipeline interleaves contacts across
                    # pairs, so contacts are not pair-contiguous; a fixed-size
                    # hash table built with atomics keeps this graph-capture
                    # safe.
                    wp.launch(
                        kernel=build_restitution_manifolds,
                        dim=contacts.rigid_contact_max,
                        inputs=[
                            body_q_pre_solve,
                            body_qd_pre_solve,
                            model.body_com,
                            self.body_inv_mass_effective,
                            self.body_inv_inertia_effective,
                            model.body_world,
                            model.shape_body,
                            contacts.rigid_contact_count,
                            restitution_contact_active,
                            contacts.rigid_contact_normal,
                            contacts.rigid_contact_shape0,
                            contacts.rigid_contact_shape1,
                            model.shape_material_restitution,
                            contacts.rigid_contact_point0,
                            contacts.rigid_contact_point1,
                            contacts.rigid_contact_offset0,
                            contacts.rigid_contact_offset1,
                            model.gravity,
                            dt,
                            model.body_count,
                        ],
                        outputs=[
                            restitution_manifold_key,
                            restitution_manifold_head,
                            restitution_manifold_total,
                            restitution_contact_next,
                            restitution_contact_n_K,
                            restitution_contact_axn_lo_target,
                            restitution_contact_axn_hi_sigma,
                            restitution_contact_pos_depth,
                        ],
                        device=model.device,
                    )

                    # Reduce each manifold chain to its bounded best-K subset
                    # (deterministic; see restitution_kernels.select_manifold_contacts).
                    wp.launch(
                        kernel=select_manifold_contacts,
                        dim=contacts.rigid_contact_max,
                        inputs=[
                            restitution_manifold_key,
                            restitution_manifold_head,
                            restitution_manifold_total,
                            restitution_contact_next,
                            restitution_contact_pos_depth,
                            restitution_contact_n_K,
                        ],
                        outputs=[
                            restitution_manifold_contact,
                            restitution_manifold_size,
                            restitution_contact_sel_score,
                        ],
                        device=model.device,
                    )

                    body_qd_with_restitution = body_qd_for_restitution
                    if not requires_grad:
                        # apply_restitution_deltas consumes and clears the
                        # accumulators, so they only need zeroing once
                        body_deltas.zero_()
                        restitution_body_manifold_count.zero_()
                    for outer_iteration in range(self.rigid_contact_restitution_iterations):
                        if requires_grad:
                            body_deltas = wp.zeros_like(body_deltas)
                            restitution_body_manifold_count = wp.zeros_like(restitution_body_manifold_count)

                        wp.launch(
                            kernel=solve_manifold_restitution,
                            dim=contacts.rigid_contact_max,
                            inputs=[
                                body_qd_with_restitution,
                                body_q_pre_solve,
                                self.body_inv_mass_effective,
                                self.body_inv_inertia_effective,
                                restitution_manifold_key,
                                restitution_manifold_size,
                                restitution_manifold_contact,
                                model.body_count,
                                restitution_contact_n_K,
                                restitution_contact_axn_lo_target,
                                restitution_contact_axn_hi_sigma,
                                self._restitution_manifold_inner_iterations,
                                outer_iteration,
                            ],
                            outputs=[body_deltas, restitution_body_manifold_count],
                            device=model.device,
                        )

                        if requires_grad:
                            next_body_qd = wp.clone(body_qd_with_restitution)
                            wp.launch(
                                kernel=apply_body_delta_velocities,
                                dim=model.body_count,
                                inputs=[body_deltas, restitution_body_manifold_count],
                                outputs=[next_body_qd],
                                device=model.device,
                            )
                            body_qd_with_restitution = next_body_qd
                        else:
                            wp.launch(
                                kernel=apply_restitution_deltas,
                                dim=model.body_count,
                                inputs=[body_deltas, restitution_body_manifold_count],
                                outputs=[body_qd_with_restitution],
                                device=model.device,
                            )
                    if requires_grad:
                        state_out.body_qd = body_qd_with_restitution

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
        self._apply_module_options()
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
