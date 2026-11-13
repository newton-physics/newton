# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import warnings
from dataclasses import dataclass
from enum import IntFlag
from typing import Any

import warp as wp

from ...core.types import override
from ...exceptions import NewtonDeprecationWarning
from ...sim import Contacts, Control, Model, ModelBuilder, State
from ..solver import SolverBase
from .affine_body import AffineBody
from .builder import PDMatrixBuilder, _compute_max_neighbors
from .collision import Collision
from .contact import Contact
from .kernels import (
    accumulate_dragging_pd_diag_kernel,
    compute_solver_flags_kernel,
    eval_bend_kernel,
    eval_drag_force_kernel,
    eval_stretch_kernel,
    init_rhs_kernel,
    init_step_kernel,
    nonlinear_step_kernel,
    prepare_jacobi_preconditioner_kernel,
    update_velocity_kernel,
)
from .linear_solver import PcgSolver, SparseMatrixELL

AttributeAssignment = Model.AttributeAssignment
AttributeFrequency = Model.AttributeFrequency


class _LegacyUnset:
    def __repr__(self) -> str:
        return "_LEGACY_UNSET"


_LEGACY_UNSET: Any = _LegacyUnset()


def _warn_legacy_attribute(name: str, config_name: str) -> None:
    """Point deprecated attribute warnings at the caller's read or assignment."""
    warnings.warn(
        f"SolverStyle3D.{name} is deprecated as of Newton 1.8; use solver.config.{config_name} instead.",
        NewtonDeprecationWarning,
        stacklevel=3,
    )


########################################################################################################################
#################################################    Style3D Solver    #################################################
########################################################################################################################


class SolverStyle3D(SolverBase):
    r"""Implicit solver for cloth and free rigid bodies using affine body dynamics.

    Select :attr:`Dynamics.DEFORMABLE` (the default), :attr:`Dynamics.AFFINE_BODY`,
    or :attr:`Dynamics.ALL` to advance cloth, rigid bodies, or both together.
    Joint constraints are not supported. See the :doc:`Style3D solver guide
    </solvers/style3d>` for setup, rigidity, and contact parameters.

    With affine body dynamics enabled, rigid-rigid and particle-rigid contacts
    use a normal penalty, closing-only damping, and Coulomb-limited viscous friction.
    The contact tangent holds the normal load and friction gain fixed to remain symmetric
    positive semidefinite and resist slip reversal overshoot. Contact candidates
    and normals come from the supplied :class:`~newton.Contacts` and remain
    fixed during each step.

    :attr:`Dynamics.ALL` couples particles and affine bodies in one linear
    system. Particle-rigid contacts use :attr:`Model.soft_contact_ke`,
    :attr:`Model.soft_contact_kd`, and :attr:`Model.soft_contact_kf`; friction
    mixes :attr:`Model.soft_contact_mu` with the shape coefficient geometrically.
    Only particle soft contacts are supported, not edge or triangle contacts.
    Deformable-only solves retain the one-way response to prescribed bodies.

    References:
        1. Baraff, D. & Witkin, A. "Large Steps in Cloth Simulation."
        2. Liu, T. et al. "Fast Simulation of Mass-Spring Systems."
        3. Lei Lan, et al. "Affine Body Dynamics: Fast, Stable and Intersection-free Simulation of Stiff Materials."

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

    class Dynamics(IntFlag):
        """Select dynamics to advance, combining flags with bitwise OR (``|``)."""

        NONE = 0
        """Select no dynamics."""

        DEFORMABLE = 1 << 0
        """Advance deformable particles, including cloth."""

        AFFINE_BODY = 1 << 1
        """Advance rigid bodies using affine body dynamics."""

        ALL = DEFORMABLE | AFFINE_BODY
        """Enable all supported dynamics."""

    @dataclass(kw_only=True)
    class Config:
        """Runtime options for :class:`SolverStyle3D`.

        The solver keeps the supplied object as :attr:`SolverStyle3D.config`.
        Changes made before a subsequent :meth:`SolverStyle3D.step` take
        effect on that step. Captured CUDA graphs must be recaptured to use
        updated configuration values.

        Example:
            Configure a solver for an existing model::

                config = SolverStyle3D.Config(iterations=20, linear_iterations=10)
                solver = SolverStyle3D(model, config=config)
                solver.config.iterations = 30
        """

        iterations: int = 10
        """Number of non-linear iterations per step. Must be at least 1."""

        linear_iterations: int = 10
        """Number of PCG iterations per non-linear iteration, capped by PCG capacity (999 by default)."""

        affine_rigidity_ratio: float = 1e2
        """Dimensionless SO(3) stiffness ratio relative to mean directional inertia.

        Zero disables the penalty. See :doc:`/solvers/style3d` for the stiffness
        formula and its distinction from contact stiffness.
        """

        drag_spring_stiff: float = 1e2
        """Stiffness of the mouse-dragging spring [N/m]."""

        enable_mouse_dragging: bool = False
        """Whether to enable the mouse-dragging kernel."""

    config: Config
    """Live solver options, retained by reference when supplied at construction.

    Edit fields or replace this object with a :class:`Config` between calls
    to :meth:`step`. Captured CUDA graphs must be recaptured after changes.
    """

    @property
    def linear_iterations(self) -> int:
        """Legacy alias for :attr:`Config.linear_iterations`.

        .. deprecated:: 1.8
            Use ``solver.config.linear_iterations`` instead.
        """
        _warn_legacy_attribute("linear_iterations", "linear_iterations")
        return self.config.linear_iterations

    @linear_iterations.setter
    def linear_iterations(self, value: int) -> None:
        _warn_legacy_attribute("linear_iterations", "linear_iterations")
        self.config.linear_iterations = value

    @property
    def nonlinear_iterations(self) -> int:
        """Legacy alias for :attr:`Config.iterations`.

        .. deprecated:: 1.8
            Use ``solver.config.iterations`` instead.
        """
        _warn_legacy_attribute("nonlinear_iterations", "iterations")
        return self.config.iterations

    @nonlinear_iterations.setter
    def nonlinear_iterations(self, value: int) -> None:
        _warn_legacy_attribute("nonlinear_iterations", "iterations")
        self.config.iterations = value

    @property
    def drag_spring_stiff(self) -> float:
        """Legacy alias for :attr:`Config.drag_spring_stiff`.

        .. deprecated:: 1.8
            Use ``solver.config.drag_spring_stiff`` instead.
        """
        _warn_legacy_attribute("drag_spring_stiff", "drag_spring_stiff")
        return self.config.drag_spring_stiff

    @drag_spring_stiff.setter
    def drag_spring_stiff(self, value: float) -> None:
        _warn_legacy_attribute("drag_spring_stiff", "drag_spring_stiff")
        self.config.drag_spring_stiff = value

    @property
    def enable_mouse_dragging(self) -> bool:
        """Legacy alias for :attr:`Config.enable_mouse_dragging`.

        .. deprecated:: 1.8
            Use ``solver.config.enable_mouse_dragging`` instead.
        """
        _warn_legacy_attribute("enable_mouse_dragging", "enable_mouse_dragging")
        return self.config.enable_mouse_dragging

    @enable_mouse_dragging.setter
    def enable_mouse_dragging(self, value: bool) -> None:
        _warn_legacy_attribute("enable_mouse_dragging", "enable_mouse_dragging")
        self.config.enable_mouse_dragging = value

    def __init__(
        self,
        model: Model,
        *,
        config: Config | None = None,
        iterations: int = _LEGACY_UNSET,
        linear_iterations: int = _LEGACY_UNSET,
        drag_spring_stiff: float = _LEGACY_UNSET,
        enable_mouse_dragging: bool = _LEGACY_UNSET,
        dynamics: Dynamics = Dynamics.DEFORMABLE,
    ):
        """Initialize the Style3D solver.

        .. deprecated:: 1.8
            The ``iterations``, ``linear_iterations``, ``drag_spring_stiff``,
            and ``enable_mouse_dragging`` keyword arguments are deprecated.
            Pass their values through ``config=SolverStyle3D.Config(...)`` instead.

        Args:
            model: The :class:`~newton.Model` to integrate. Deformable dynamics require
                Style3D custom attributes; call :meth:`register_custom_attributes`
                before building the model.
            config: Runtime options. If ``None``, use :class:`Config` defaults with
                any supplied legacy keyword options. The solver keeps this object
                publicly as :attr:`config`; changes take effect on the next step.
                An explicit configuration cannot be combined with legacy options.
            iterations: Deprecated keyword option for :attr:`Config.iterations`.
            linear_iterations: Deprecated keyword option for :attr:`Config.linear_iterations`.
            drag_spring_stiff: Deprecated keyword option for :attr:`Config.drag_spring_stiff`.
            enable_mouse_dragging: Deprecated keyword option for :attr:`Config.enable_mouse_dragging`.
            dynamics: Dynamics to advance, selected with :class:`Dynamics` flags.
                Combine flags with bitwise OR (``|``). Defaults to
                :attr:`Dynamics.DEFORMABLE` for deformable particles, including cloth.
                Use :attr:`Dynamics.AFFINE_BODY` to advance rigid bodies using affine
                body dynamics, :attr:`Dynamics.ALL` to select all dynamics,
                or :attr:`Dynamics.NONE` to select none.

        Raises:
            TypeError: If ``config`` is not a :class:`Config` or ``None``, or if
                a configuration is combined with legacy keyword options.
            AttributeError: If ``dynamics`` is :attr:`Dynamics.DEFORMABLE` and the
                model is missing Style3D custom attributes.
            ValueError: If ``dynamics`` contains unsupported flags.
        """

        legacy_options = {
            name: value
            for name, value in (
                ("iterations", iterations),
                ("linear_iterations", linear_iterations),
                ("drag_spring_stiff", drag_spring_stiff),
                ("enable_mouse_dragging", enable_mouse_dragging),
            )
            if value is not _LEGACY_UNSET
        }
        if config is None:
            config = self.Config(**legacy_options)
        elif not isinstance(config, self.Config):
            raise TypeError(f"Expected config to be {self.Config.__qualname__}, got {type(config).__qualname__}.")
        elif legacy_options:
            raise TypeError("Specify either 'config' or legacy solver options, not both.")

        if legacy_options:
            warnings.warn(
                "SolverStyle3D keyword options "
                f"({', '.join(legacy_options)}) are deprecated as of Newton 1.8; "
                "use config=SolverStyle3D.Config(...) instead.",
                NewtonDeprecationWarning,
                stacklevel=2,
            )

        if dynamics == SolverStyle3D.Dynamics.DEFORMABLE and not hasattr(model, "style3d"):
            raise AttributeError(
                "Style3D custom attributes are missing from the model. "
                "Call SolverStyle3D.register_custom_attributes() before building the model."
            )
        elif int(dynamics) & ~int(SolverStyle3D.Dynamics.ALL):
            raise ValueError(f"Unsupported dynamics flags: {dynamics!r}.")

        enable_deformable = bool(dynamics & SolverStyle3D.Dynamics.DEFORMABLE)
        enable_affine_body = bool(dynamics & SolverStyle3D.Dynamics.AFFINE_BODY)

        super().__init__(model)
        self._style3d_attr = getattr(model, "style3d", None)
        self.collision = Collision(model) if enable_deformable else None  # set None to disable
        self.dynamics = dynamics
        self.config = config

        # Number of DOF blocks in the linear system, with three scalar DOFs per block.
        self._particle_count = model.particle_count if enable_deformable else 0
        self._affine_body_count = model.body_count if enable_affine_body else 0
        self._dof_block_count = self._particle_count + self._affine_body_count * 4
        self._affine_body = AffineBody(model) if enable_affine_body else None
        self._linear_solver = PcgSolver(self._dof_block_count, self.device)
        self._contact = Contact(model, self._particle_count) if enable_affine_body else None

        # ParticleFlags for every particle and affine block, separate from model flags.
        self._particle_flags = wp.zeros(self._dof_block_count, dtype=int, device=self.device)

        # Fixed PD matrix
        self._pd_off_diag = SparseMatrixELL()
        self._pd_diag = wp.zeros(self._dof_block_count, dtype=float, device=self.device)
        self._precompute(model)

        # Non-linear equation variables
        self._x = wp.zeros(self._dof_block_count, dtype=wp.vec3, device=self.device)
        self._v = wp.zeros(self._dof_block_count, dtype=wp.vec3, device=self.device)
        self._dx = wp.zeros(self._dof_block_count, dtype=wp.vec3, device=self.device)
        self._rhs = wp.zeros(self._dof_block_count, dtype=wp.vec3, device=self.device)
        self._x_prev = wp.zeros(self._dof_block_count, dtype=wp.vec3, device=self.device)
        self._x_inertia = wp.zeros(self._dof_block_count, dtype=wp.vec3, device=self.device)

        # Per-step scalar mass/PD diagonal, excluding nonlinear contributions.
        self._A_diag_base = wp.zeros(self._dof_block_count, dtype=float, device=self.device)
        # Assembled diagonal blocks, including shape contacts but excluding cloth self-contacts.
        self._A_diag = wp.zeros(self._dof_block_count, dtype=wp.mat33, device=self.device)
        # Per-body (S_01, S_02, S_12) / dt^2, refreshed alongside the mass diagonal.
        self._affine_mass_off_diag = wp.zeros(self._affine_body_count, dtype=wp.vec3, device=self.device)
        # Inverse full diagonal blocks for Jacobi preconditioning.
        self._inv_A_diag = wp.zeros(self._dof_block_count, dtype=wp.mat33, device=self.device)
        # Matrix-free contribution to A x beyond the assembled base matrix.
        self._matrix_free_Ax = wp.zeros(self._dof_block_count, dtype=wp.vec3, device=self.device)

        # Drag info
        self._drag_pos = wp.zeros(1, dtype=wp.vec3, device=self.device)
        self._drag_index = wp.array([-1], dtype=int, device=self.device)
        self._drag_bary_coord = wp.zeros(1, dtype=wp.vec3, device=self.device)

    @override
    def step(
        self,
        state_in: State,
        state_out: State,
        control: Control,
        contacts: Contacts,
        dt: float,
        *,
        observables: SolverBase.Observables | None = None,
    ) -> None:
        """Advance the Style3D solver by one time step.

        The solver performs non-linear projective dynamics iterations in internal
        buffers with optional collision handling. Final positions and velocities
        for the selected dynamics are written to ``state_out``. Kinematic and
        zero-mass body outputs remain owned by the caller.

        With affine body dynamics enabled, positive-mass kinematic bodies use
        rigid-body integration of ``state_in`` to predict their contact poses.
        The caller must supply velocities consistent with the supplied forces,
        model gravity and inertia, and ``dt``, so the predictor reproduces the
        externally prescribed end pose, including rotation. The external driver
        owns the corresponding ``state_out`` body poses and velocities. See
        :ref:`style3d-externally-driven-bodies` for the integration contract.

        Args:
            state_in: Input :class:`newton.State`.
            state_out: Output :class:`newton.State` with the final state.
            control: :class:`newton.Control` input (currently unused).
            contacts: :class:`newton.Contacts` used for collision response.
            dt: Time step in seconds.
            observables: Optional solver observable arrays allocated by :meth:`observables`.
                This solver declares no supported observables, so only an empty
                container is accepted.

        Raises:
            ValueError: If :attr:`Config.iterations` is less than 1. Neither
                input nor output state is modified in this case.
        """
        self.validate_observables(observables, contacts)
        self._require_unreduced_contacts(contacts)

        if self.dynamics == SolverStyle3D.Dynamics.NONE:
            return
        elif self.config.iterations < 1:
            raise ValueError(f"Config.iterations must be at least 1, got {self.config.iterations}.")

        self._to_internal_state(state_in, dt)
        particle_q = self._x[: self._particle_count]
        particle_qd = self._v[: self._particle_count]

        if self.collision is not None:
            self.collision.frame_begin(particle_q, particle_qd, dt)

        wp.launch(
            kernel=init_step_kernel,
            dim=self._particle_count + self._affine_body_count,
            inputs=[
                self._particle_count,
                dt,
                self.model.gravity,
                self.model.particle_world,
                state_in.particle_f if self._particle_count > 0 else self._x[:0],
                particle_qd,
                self._x,
                self._x_prev,
                self._pd_diag,
                self.model.particle_mass,
                self._particle_flags,
                self.model.body_mass,
                self.model.body_inertia,
                self.model.body_flags,
            ],
            outputs=[self._x_inertia, self._A_diag_base, self._dx, self._affine_mass_off_diag],
            device=self.device,
        )

        if self._particle_count > 0 and self.config.enable_mouse_dragging:
            wp.launch(
                accumulate_dragging_pd_diag_kernel,
                dim=1,
                inputs=[
                    self.config.drag_spring_stiff,
                    self._drag_index,
                    self._drag_bary_coord,
                    self.model.tri_indices,
                    self._particle_flags,
                ],
                outputs=[self._A_diag_base],
                device=self.device,
            )

        for _iter in range(self.config.iterations):
            wp.launch(
                init_rhs_kernel,
                dim=self._particle_count + self._affine_body_count,
                inputs=[
                    dt,
                    self._x,
                    self._x_inertia,
                    self._particle_count,
                    self.model.particle_mass,
                    self.model.body_mass,
                    self.model.body_inertia,
                    self.model.body_flags,
                ],
                outputs=[self._rhs],
                device=self.device,
            )

            if self._particle_count > 0:
                wp.launch(
                    eval_stretch_kernel,
                    dim=self.model.tri_count,
                    inputs=[
                        particle_q,
                        self.model.tri_areas,
                        self.model.tri_poses,
                        self.model.tri_indices,
                        self._style3d_attr.tri_aniso_ke,
                    ],
                    outputs=[self._rhs],
                    device=self.device,
                )

                wp.launch(
                    eval_bend_kernel,
                    dim=len(self._style3d_attr.edge_rest_area),
                    inputs=[
                        particle_q,
                        self._style3d_attr.edge_rest_area,
                        self._style3d_attr.edge_bending_cot,
                        self.model.edge_indices,
                        self.model.edge_bending_properties,
                    ],
                    outputs=[self._rhs],
                    device=self.device,
                )

                if self.config.enable_mouse_dragging:
                    wp.launch(
                        eval_drag_force_kernel,
                        dim=1,
                        inputs=[
                            self.config.drag_spring_stiff,
                            self._drag_index,
                            self._drag_pos,
                            self._drag_bary_coord,
                            self.model.tri_indices,
                            particle_q,
                        ],
                        outputs=[self._rhs],
                        device=self.device,
                    )

            if self.collision is not None:
                self.collision.accumulate_contact_force(
                    dt,
                    _iter,
                    contacts,
                    state_in.body_q,
                    state_out.body_q,
                    particle_q,
                    self._rhs[: self._particle_count],
                    self._x_prev[: self._particle_count],
                    self._A_diag_base[: self._particle_count],
                    include_body_contacts=self._contact is None,
                )

            self._A_diag[: self._particle_count].zero_()
            if self._affine_body is not None:
                self._affine_body.eval_orthogonality(
                    self._x[self._particle_count :],
                    self._rhs[self._particle_count :],
                    self._A_diag[self._particle_count :],
                    self.config.affine_rigidity_ratio,
                    dt,
                )

            if self._contact is not None:
                self._contact.eval_contacts(
                    contacts,
                    self._x,
                    self._x_prev,
                    self._x_inertia,
                    self._particle_flags,
                    self._rhs,
                    self._A_diag,
                    dt,
                )

            wp.launch(
                prepare_jacobi_preconditioner_kernel,
                dim=self._dof_block_count,
                inputs=[
                    self._particle_count,
                    self._A_diag_base,
                    self.collision.contact_hessian_diagonal() if self.collision is not None else self._A_diag[:0],
                    self._particle_flags,
                ],
                outputs=[self._A_diag, self._inv_A_diag],
                device=self.device,
            )

            self._linear_solver.solve(
                self._pd_off_diag,
                self._A_diag,
                self._dx if _iter == 0 else None,
                self._rhs,
                self._inv_A_diag,
                self._dx,
                self.config.linear_iterations,
                self._matrix_free_multiply,
            )

            if self.collision is not None:
                self.collision.linear_iteration_end(self._dx[: self._particle_count])

            wp.launch(
                nonlinear_step_kernel,
                dim=self._dof_block_count,
                inputs=[self._x],
                outputs=[self._x, self._dx],
                device=self.device,
            )

        wp.launch(
            kernel=update_velocity_kernel,
            dim=self._dof_block_count,
            inputs=[dt, self._particle_count, self._x_prev, self._x],
            outputs=[self._v],
            device=self.device,
        )

        if self.collision is not None:
            self.collision.frame_end(particle_q, particle_qd, dt)

        self._from_internal_state(state_out, dt)

    def _matrix_free_multiply(self, x: wp.array[wp.vec3]) -> wp.array[wp.vec3]:
        """Return the matrix-free contribution to A x, excluding the assembled base matrix."""
        if self.collision is not None:
            self.collision.hessian_multiply(x[: self._particle_count], self._matrix_free_Ax[: self._particle_count])
        else:
            self._matrix_free_Ax[: self._particle_count].zero_()
        if self._affine_body is not None:
            self._affine_body.off_diag_multiply(
                x[self._particle_count :], self._affine_mass_off_diag, self._matrix_free_Ax[self._particle_count :]
            )
        if self._contact is not None:
            self._contact.off_diag_multiply(x, self._matrix_free_Ax)
        return self._matrix_free_Ax

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

    def _precompute(self, model: Model):
        with wp.ScopedTimer("SolverStyle3D::precompute()"):
            if self._particle_count > 0 and (
                not hasattr(model, "style3d")
                or not hasattr(model.style3d, "tri_aniso_ke")
                or not hasattr(model.style3d, "edge_rest_area")
                or not hasattr(model.style3d, "edge_bending_cot")
            ):
                raise AttributeError(
                    "Style3D custom attributes are missing from the model. "
                    "Call SolverStyle3D.register_custom_attributes() before building the model."
                )

            if self._particle_count > 0:
                tri_indices = model.tri_indices.numpy()
                tri_poses = model.tri_poses.numpy().tolist()
                tri_areas = model.tri_areas.numpy().tolist()
                edge_indices = model.edge_indices.numpy()
                edge_bending_properties = model.edge_bending_properties.numpy().tolist()
                tri_aniso_ke = model.style3d.tri_aniso_ke.numpy().tolist()
                edge_rest_area = model.style3d.edge_rest_area.numpy().tolist()
                edge_bending_cot = model.style3d.edge_bending_cot.numpy().tolist()

            def assemble(max_neighbors):
                """Assemble from scratch so failed attempts never contribute partial entries."""
                matrix = PDMatrixBuilder(self._dof_block_count, max_neighbor=max_neighbors)
                if self._particle_count > 0:
                    matrix.add_stretch_constraints(tri_indices, tri_poses, tri_aniso_ke, tri_areas)
                    matrix.add_bend_constraints(edge_indices, edge_bending_properties, edge_rest_area, edge_bending_cot)
                return matrix

            try:
                pd_matrix_builder = assemble(32)
            except ValueError:
                # Only overflowing meshes pay for counting stretch and bending connections.
                if self._particle_count > 0:
                    max_neighbors = _compute_max_neighbors(self._dof_block_count, tri_indices, edge_indices)
                    pd_matrix_builder = assemble(max_neighbors)
            self._pd_diag, self._pd_off_diag.num_nz, self._pd_off_diag.nz_ell = pd_matrix_builder.finalize(self.device)

    def _to_internal_state(self, state: State, dt: float):
        """Populate internal solver state and refresh flags for all DOF blocks."""
        if self._particle_count > 0:
            wp.copy(self._x, state.particle_q, count=self._particle_count)
            wp.copy(self._v, state.particle_qd, count=self._particle_count)
        if self._dof_block_count > 0:
            wp.launch(
                compute_solver_flags_kernel,
                dim=self._particle_count + self._affine_body_count,
                inputs=[
                    self._particle_count,
                    self.model.particle_mass,
                    self.model.particle_flags,
                    self.model.body_mass,
                    self.model.body_flags,
                ],
                outputs=[self._particle_flags],
                device=self.device,
            )
        if self._affine_body_count > 0:
            self._affine_body.to_internal_state(
                state, self._x[self._particle_count :], self._x_inertia[self._particle_count :], dt
            )

    def _from_internal_state(self, state: State, dt: float):
        """Write solved positions and velocities to the output state."""
        if self._particle_count > 0:
            wp.copy(state.particle_q, self._x, count=self._particle_count)
            wp.copy(state.particle_qd, self._v, count=self._particle_count)
        if self._affine_body_count > 0:
            self._affine_body.from_internal_state(
                state, self._x[self._particle_count :], self._v[self._particle_count :], dt
            )

    def _update_drag_info(self, index: int, pos: wp.vec3, bary_coord: wp.vec3):
        """Should be invoked when state changed."""
        # print([index, pos, bary_coord])
        self._drag_bary_coord.fill_(bary_coord)
        self._drag_index.fill_(index)
        self._drag_pos.fill_(pos)
