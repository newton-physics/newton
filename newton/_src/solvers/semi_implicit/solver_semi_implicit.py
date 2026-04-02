# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import warp as wp

from ...core.types import override
from ...sim import Contacts, Control, Model, State
from ..solver import SolverBase
from .kernels_body import (
    eval_body_joint_forces,
)
from .kernels_contact import (
    eval_body_contact_forces,
    eval_particle_body_contact_forces,
    eval_particle_contact_forces,
    eval_triangle_contact_forces,
)
from .kernels_muscle import (
    eval_muscle_forces,
)
from .kernels_particle import (
    eval_bending_forces,
    eval_spring_forces,
    eval_tetrahedra_forces,
    eval_triangle_forces,
)


class SolverSemiImplicit(SolverBase):
    """A semi-implicit integrator using symplectic Euler.

    After constructing `Model` and `State` objects this time-integrator
    may be used to advance the simulation state forward in time.

    Semi-implicit time integration is a variational integrator that
    preserves energy, however it not unconditionally stable, and requires a time-step
    small enough to support the required stiffness and damping forces.

    See: https://en.wikipedia.org/wiki/Semi-implicit_Euler_method

    Joint limitations:
        - Supported joint types: PRISMATIC, REVOLUTE, BALL, FIXED, FREE, DISTANCE (treated as FREE), D6.
          CABLE joints are not supported.
        - :attr:`~newton.Model.joint_enabled`, :attr:`~newton.Model.joint_limit_ke`/:attr:`~newton.Model.joint_limit_kd`,
          :attr:`~newton.Model.joint_target_ke`/:attr:`~newton.Model.joint_target_kd`, and :attr:`~newton.Control.joint_f`
          are supported.
        - Joint limits and targets are not enforced for BALL joints.
        - :attr:`~newton.Model.joint_armature`, :attr:`~newton.Model.joint_friction`,
          :attr:`~newton.Model.joint_effort_limit`, :attr:`~newton.Model.joint_velocity_limit`,
          and :attr:`~newton.Model.joint_target_mode` are not supported.
        - Equality and mimic constraints are not supported.

        See :ref:`Joint feature support` for the full comparison across solvers.

    Example
    -------

    .. code-block:: python

        solver = newton.solvers.SolverSemiImplicit(model)

        # simulation loop
        for i in range(100):
            solver.step(state_in, state_out, control, contacts, dt)
            state_in, state_out = state_out, state_in

    """

    def __init__(
        self,
        model: Model,
        angular_damping: float = 0.05,
        friction_smoothing: float = 1.0,
        joint_attach_ke: float = 1.0e4,
        joint_attach_kd: float = 1.0e2,
        enable_tri_contact: bool = True,
    ):
        """
        Args:
            model: The model to be simulated.
            angular_damping: Angular damping factor to be used in rigid body integration. Defaults to 0.05.
            friction_smoothing: Huber norm delta used for friction velocity normalization (see :func:`warp.norm_huber() <warp._src.lang.norm_huber>`). Defaults to 1.0.
            joint_attach_ke: Joint attachment spring stiffness. Defaults to 1.0e4.
            joint_attach_kd: Joint attachment spring damping. Defaults to 1.0e2.
            enable_tri_contact: Enable triangle contact. Defaults to True.
        """
        super().__init__(model=model)
        self.angular_damping = angular_damping
        self.friction_smoothing = friction_smoothing
        self.joint_attach_ke = joint_attach_ke
        self.joint_attach_kd = joint_attach_kd
        self.enable_tri_contact = enable_tri_contact

    @override
    def step(
        self,
        state_in: State,
        state_out: State,
        control: Control | None,
        contacts: Contacts | None,
        dt: float,
    ) -> None:
        """
        Simulate the model for a given time step using the given control input.

        Args:
            state_in: The input state.
            state_out: The output state.
            control: The control input.
                Defaults to `None` which means the control values from the
                :class:`Model` are used.
            contacts: The contact information.
                Defaults to `None` which means no contacts are used.
            dt: The time step (typically in seconds).

        .. warning::
            The ``eval_particle_contact`` kernel for particle-particle contact handling may corrupt the gradient computation
            for simulations involving particle collisions.
            To disable it, set :attr:`newton.Model.particle_grid` to `None` prior to calling :meth:`step`.
        """
        with wp.ScopedTimer("simulate", False):
            particle_f = None
            body_f = None

            if state_in.particle_count:
                particle_f = state_in.particle_f

            if state_in.body_count:
                body_f = state_in.body_f

            model = self.model

            if control is None:
                control = model.control(clone_variables=False)

            body_f_work = body_f
            if body_f is not None and model.joint_count and control.joint_f is not None:
                # Avoid accumulating joint_f into the persistent state body_f buffer.
                body_f_work = wp.clone(body_f)

            # damped springs
            eval_spring_forces(model, state_in, particle_f)

            # triangle elastic and lift/drag forces
            eval_triangle_forces(model, state_in, control, particle_f)

            # triangle bending
            eval_bending_forces(model, state_in, particle_f)

            # tetrahedral FEM
            eval_tetrahedra_forces(model, state_in, control, particle_f)

            # body joints
            eval_body_joint_forces(model, state_in, control, body_f_work, self.joint_attach_ke, self.joint_attach_kd)

            # muscles
            if False:
                eval_muscle_forces(model, state_in, control, body_f)

            # particle-particle interactions
            eval_particle_contact_forces(model, state_in, particle_f)

            # triangle/triangle contacts
            if self.enable_tri_contact:
                eval_triangle_contact_forces(model, state_in, particle_f)

            # body contacts
            eval_body_contact_forces(
                model, state_in, contacts, friction_smoothing=self.friction_smoothing, body_f_out=body_f_work
            )

            # particle shape contact
            eval_particle_body_contact_forces(
                model, state_in, contacts, particle_f, body_f_work, body_f_in_world_frame=False
            )

            self.integrate_particles(model, state_in, state_out, dt)

            if body_f_work is body_f:
                self.integrate_bodies(model, state_in, state_out, dt, self.angular_damping)
            else:
                body_f_prev = state_in.body_f
                state_in.body_f = body_f_work
                self.integrate_bodies(model, state_in, state_out, dt, self.angular_damping)
                state_in.body_f = body_f_prev
