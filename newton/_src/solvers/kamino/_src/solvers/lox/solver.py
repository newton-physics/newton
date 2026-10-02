# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Frozen-contact orchestration for the LOX rigid-body solve."""

from __future__ import annotations

from typing import TYPE_CHECKING

import warp as wp

from ......sim import ModelFlags
from ....config import LOXSolverConfig
from ...core.joints import JointCorrectionMode
from ...core.state import StateKamino
from ...core.types import vec6f
from .metrics import DualSolution
from .problem import LOXProblem
from .projection import ProjectionSchedule
from .projection_kernels import _compute_box_residuals, _compute_contact_residuals, _compute_spatial_contact_residuals
from .solver_kernels import (
    _finalize_residual_iteration,
    _initialize_bodies,
    _initialize_fixed_iteration,
    _initialize_iteration_residuals,
    _initialize_world_convergence,
    _prepare_projection,
    _reduce_body_residuals,
    _reduce_final_body_residuals,
    _reset_dual_wrench,
    _restore_dual_from_wrench,
    _store_dual_wrench,
    _update_splitting_dual,
    _write_integrator_body_inputs,
    _write_world_status,
)
from .types import LOXData, ProjectionAcceleration

if TYPE_CHECKING:
    from ....config import ConstraintStabilizationConfig
    from ...core.data import DataKamino
    from ...core.model import ModelKamino
    from ...dynamics.dual import DualProblem
    from ...geometry.contacts import ContactsKamino
    from ...kinematics.jacobians import SparseSystemJacobians
    from ...kinematics.limits import LimitsKamino

###
# Module interface
###

__all__ = [
    "LOXSolver",
]

###
# Module configs
###

wp.set_module_options({"enable_backward": False})


###
# Constants
###


_WORLD_THREAD_GRANULE = 32
"""Granule of the number of threads that loop over the active contacts of one world."""


_WORLD_THREADS_PER_SM = 256
"""Threads per SM that the per-world contact loops launch when there are few worlds."""


###
# Functions
###


def _world_thread_count(device: wp.DeviceLike, num_worlds: int, max_world_items: int) -> int:
    """Return the number of threads that loop over the active items of each world.

    Each world gets one warp when there are enough worlds to fill the device; fewer, larger
    worlds get more warps, up to one item per thread.

    Args:
        device: Device of the launch.
        num_worlds: Number of worlds.
        max_world_items: Largest number of items of one world.
    """
    device = wp.get_device(device)
    if not device.is_cuda or num_worlds == 0:
        return 1
    granules = (device.sm_count * _WORLD_THREADS_PER_SM) // (_WORLD_THREAD_GRANULE * num_worlds)
    max_granules = (max_world_items + _WORLD_THREAD_GRANULE - 1) // _WORLD_THREAD_GRANULE
    return _WORLD_THREAD_GRANULE * max(1, min(granules, max_granules))


###
# Interfaces
###


class LOXSolver:
    """Run a fixed LOX splitting solve on one frozen linearization.

    The caller owns collision detection, Jacobian construction, pose
    integration, and nonlinear relinearization. This class owns the smooth
    system update, weight construction, blocked dense LLT solve, unilateral
    projections, convergence freezing, and output conversion for one such
    linearization. A nonfailed world that reaches the iteration limit is
    accepted using its last projected iterate.
    """

    Config = LOXSolverConfig
    """Type alias of the LOX solver configuration container."""

    def __init__(
        self,
        model: ModelKamino | None = None,
        data: DataKamino | None = None,
        jacobians: SparseSystemJacobians | None = None,
        limits: LimitsKamino | None = None,
        contacts: ContactsKamino | None = None,
        config: LOXSolver.Config | None = None,
        constraints: ConstraintStabilizationConfig | None = None,
        rotation_correction: JointCorrectionMode = JointCorrectionMode.TWOPI,
        *,
        compute_solution_metrics: bool = False,
    ):
        """Creates a LOX solver.

        If a model is provided, all necessary memory allocations are performed on
        the model's device; otherwise :meth:`finalize` must be called before using
        the solver.

        Args:
            model: The model for which to allocate the solver data.
            data: The model data holding the state of the bodies and joints.
            jacobians: The sparse system Jacobians of the model.
            limits: The joint limits container of the model.
            contacts: The contacts container of the model.
            config: The solver config to use.
            constraints: The constraint stabilization config to use.
            rotation_correction: The joint rotation correction mode.
            compute_solution_metrics: Whether to allocate the dual solution of the solution metrics.
        """
        self._config: LOXSolver.Config | None = None
        self._constraints: ConstraintStabilizationConfig | None = None
        self._model: ModelKamino | None = None
        self._device: wp.DeviceLike = None
        self._problem: LOXProblem | None = None
        self._data: LOXData | None = None

        if model is not None:
            self.finalize(
                model,
                data,
                jacobians,
                limits,
                contacts,
                config,
                constraints,
                rotation_correction,
                compute_solution_metrics=compute_solution_metrics,
            )

    ###
    # Properties
    ###

    @property
    def data(self) -> LOXData:
        """Returns the solver state and terminal status."""
        if self._data is None:
            raise RuntimeError("Solver data has not been allocated yet. Call `finalize()` first.")
        return self._data

    @property
    def problem(self) -> LOXProblem:
        """Returns the constraint rows and the smooth body system."""
        return self._problem

    ###
    # Public API
    ###

    def finalize(
        self,
        model: ModelKamino,
        data: DataKamino,
        jacobians: SparseSystemJacobians,
        limits: LimitsKamino | None,
        contacts: ContactsKamino | None,
        config: LOXSolver.Config,
        constraints: ConstraintStabilizationConfig,
        rotation_correction: JointCorrectionMode = JointCorrectionMode.TWOPI,
        *,
        compute_solution_metrics: bool = False,
    ) -> None:
        """Allocates the solver data structures on the model's device.

        Args:
            model: The model for which to allocate the solver data.
            data: The model data holding the state of the bodies and joints.
            jacobians: The sparse system Jacobians of the model.
            limits: The joint limits container of the model.
            contacts: The contacts container of the model.
            config: The solver config to use.
            constraints: The constraint stabilization config to use.
            rotation_correction: The joint rotation correction mode.
            compute_solution_metrics: Whether to allocate the dual solution of the solution metrics.
        """
        if model.size.sum_of_num_bodies == 0:
            raise ValueError("LOX requires at least one rigid body.")
        self._config = config
        self._constraints = constraints
        self._model = model
        self._device = model.device
        self._problem = LOXProblem(
            model=model,
            data=data,
            jacobians=jacobians,
            limits=limits,
            contacts=contacts,
            eliminate_fixed_world_islands=config.eliminate_fixed_world_islands,
            selective_weights=config.selective_weights,
            rotation_correction=rotation_correction,
            joint_proximal_relaxation=config.joint_proximal_relaxation,
            contact_compliance=config.contact_compliance,
            contact_restitution=config.contact_restitution,
            contact_spatial_friction=config.contact_spatial_friction,
        )
        self._data = LOXData(
            model.info.num_bodies.numpy().tolist(),
            DualSolution(model, config) if compute_solution_metrics else None,
            config.joint_penalty_scale,
            self._device,
        )
        self._build_projection_schedule()
        self._iteration_condition = wp.zeros(1, dtype=wp.int32, device=self._device)
        self._initialize_splitting()

    def notify_model_changed(self, flags: ModelFlags | int) -> None:
        """Refresh or rebuild the LOX problem after model changes."""
        if flags & ModelFlags.BODY_PROPERTIES and self._problem.dynamic_body_topology_changed():
            self._rebuild_rigid_topology()

    def validate_model_changed(self) -> None:
        """Validate LOX-specific values derived from the Newton model."""
        # Host-side validation cannot synchronize aliased Newton arrays while a
        # CUDA graph is being captured. The same topology was validated when the
        # solver was built; captured property updates retain that topology.
        if self._device.is_cuda and self._device.is_capturing:
            return
        self._problem.validate_model_changed()

    def reset(
        self,
        problem: DualProblem | None = None,
        world_mask: wp.array[wp.bool] | None = None,
    ) -> None:
        """Clear the structural and splitting warm starts and the per-world diagnostics.

        Args:
            problem: Unused; accepted for interface compatibility with the dual solvers.
            world_mask: Worlds to reset, or ``None`` for all worlds.
        """
        del problem
        if world_mask is not None:
            if world_mask.shape != (self._model.info.num_worlds,) or world_mask.dtype != wp.bool:
                raise ValueError(f"world_mask must have shape ({self._model.info.num_worlds},) and dtype bool.")
            if world_mask.device != self._device:
                raise ValueError(f"world_mask must be allocated on {self._device}, found {world_mask.device}.")

        self._initialize_splitting(world_mask=world_mask)
        if world_mask is None:
            self._data.dual_wrench.zero_()
        else:
            wp.launch(
                _reset_dual_wrench,
                dim=self._data.state.num_bodies,
                inputs=[self._model.bodies.wid, world_mask],
                outputs=[self._data.dual_wrench],
                device=self._device,
            )
        self._problem.reset_structural_multipliers(world_mask=world_mask)
        self._problem.reset_effort_counters(world_mask=world_mask)
        self._problem.reset_box_reactions(world_mask=world_mask)
        self._problem.reset_angular_contact_reactions(world_mask)

    def joint_penalty_scale_seed(self, time_step: float) -> wp.array[wp.float32]:
        """Estimate and apply a timestep-aware structural ALM scale.

        The estimate is the reciprocal of the discrete second-percentile
        positive eigenvalue of the effective-mass-normalized structural
        Delassus. Both the percentile and the resulting scale are evaluated
        independently for each world. Systems with fewer than 50 positive
        modes therefore retain the minimum-eigenvalue estimate. The smooth
        operator excludes the body contact-consensus metric, which is a
        separate splitting concern.

        This operation prepares the LOX problem from the initial rigid state and
        performs a one-time dense structural assembly and host eigensolve, so
        it is intended for initialization, outside the captured simulation
        loop.

        Args:
            time_step: Uniform simulation time step [s].

        Returns:
            The applied dimensionless structural ALM penalty scale for each
            world, shape ``(num_worlds,)``. This is the solver's own array.
        """
        problem = self._problem
        model_time = problem.model.time
        # Seeding assembles the system with the given uniform time step, so restore the
        # time step of each world afterwards
        saved_time_step = wp.clone(model_time.dt)
        saved_inverse_time_step = wp.clone(model_time.inv_dt)
        try:
            problem.prepare_joint_penalty_scale_seed(time_step)
            if problem.data.structural_rows.count > 0:
                self.reset()
                self._begin_time_step()
                try:
                    problem.build_smooth_system(model_time.dt, joint_penalty_scale=None)
                    scales = problem.estimate_joint_penalty_scale(self._data.joint_penalty_scale.numpy())
                    self._data.joint_penalty_scale.assign(scales)
                finally:
                    self.reset()
        finally:
            wp.copy(model_time.dt, saved_time_step)
            wp.copy(model_time.inv_dt, saved_inverse_time_step)
        return self._data.joint_penalty_scale

    def solve_forward_dynamics(self, contacts: ContactsKamino | None = None) -> None:
        """Solve one LOX forward-dynamics step from prepared Kamino data.

        The splitting dual is warm-started from the previous solve, see :attr:`LOXData.dual_wrench`.

        Args:
            contacts: Unused; LOX reads the contacts container bound at construction.
        """
        del contacts
        self._begin_time_step()

        # Reset the per-world solver state from the inertial velocity guess and assemble the smooth system
        self._initialize_splitting(inertial_warmstart_fraction=self._config.inertial_warmstart_fraction)
        self._update_system()
        self._restore_dual()

        # Iterate until every world converges, fails, or reaches the iteration limit
        config = self._config
        use_conditional_loop = not config.fixed_iterations and config.use_graph_conditionals
        if use_conditional_loop:
            self._iteration_condition.fill_(1)
            wp.capture_while(self._iteration_condition, self._iterate, conditional=True)
        else:
            for _ in range(config.max_iterations):
                self._iterate(conditional=False)

        # Store the terminal status and the final residuals, which also fail the worlds with
        # non-finite iterates. Store the warm start for non-failed worlds only
        self._write_final_status_and_residuals()
        self._store_dual_wrench()
        self._write_accepted_dynamics()

    def build_dual_solution(
        self,
        model: ModelKamino,
        data: DataKamino,
        state_p: StateKamino,
        problem: DualProblem,
        jacobians: SparseSystemJacobians,
        limits: LimitsKamino | None = None,
        contacts: ContactsKamino | None = None,
    ) -> None:
        """Reconstruct the dual solution of the last solve into ``data.solution`` for the solution metrics.

        Evaluate the metrics from it with ``use_solution_velocity=True``. Contacts whose
        law departs from hard contact are excluded from the dual residuals.

        Args:
            model: The model containing the time-invariant simulation data.
            data: The model data holding the final body state and joint reactions.
            state_p: The state at the beginning of the time step.
            problem: The analysis dual problem built at the LOX linearization before the solve.
            jacobians: The sparse system Jacobians used by the LOX solve.
            limits: Active joint-limit constraints.
            contacts: Active contact constraints.
        """
        if self._data.solution is None:
            raise RuntimeError("The dual solution requires a solver built with `compute_solution_metrics=True`.")
        self._data.solution.build(model, data, state_p, problem, jacobians, limits, contacts)

    ###
    # Internals - Setup
    ###

    def _rebuild_rigid_topology(self) -> None:
        """Rebuild rigid allocations after dynamic-body classification changes."""
        self._problem.rebuild_dynamic_body_topology()
        self._build_projection_schedule()
        self.reset()

    def _build_projection_schedule(self) -> None:
        """(Re)create the projection schedule for the current problem layout."""
        self._projection = ProjectionSchedule(
            self._problem.data,
            self._config.gauss_seidel_max_colors,
            ProjectionAcceleration.from_string(self._config.projection_acceleration),
        )

    def _initialize_splitting(
        self,
        inertial_warmstart_fraction: float = 0.0,
        world_mask: wp.array[wp.bool] | None = None,
    ) -> None:
        """Initialize the splitting iterates of the masked worlds, or of all worlds.

        Args:
            inertial_warmstart_fraction: Fraction of the unconstrained velocity increment added to the
                begin-step body velocities to seed the body twists.
            world_mask: Worlds to initialize, or ``None`` for all worlds.
        """
        state = self._data.state
        residuals = self._data.residuals
        problem = self._problem
        wp.launch(
            _initialize_bodies,
            dim=state.num_bodies,
            inputs=[
                self._model.bodies.wid,
                world_mask,
                problem.body_velocity_begin,
                problem.system.data.body_block,
                problem.model.bodies.m_i,
                problem.model.bodies.inv_m_i,
                problem.kamino_data.bodies.I_i,
                problem.kamino_data.bodies.inv_I_i,
                problem.kamino_data.bodies.w_e_i.view(dtype=vec6f),
                problem.kamino_data.bodies.w_a_i.view(dtype=vec6f),
                problem.model.gravity.vector,
                self._model.time.dt,
                inertial_warmstart_fraction,
                self._data.dual_wrench,
            ],
            outputs=[
                state.projected_twist,
                state.projected_twist_previous,
                state.global_twist,
                state.global_twist_previous,
            ],
            device=self._device,
        )
        wp.launch(
            _initialize_world_convergence,
            dim=state.num_worlds,
            inputs=[world_mask],
            outputs=[
                state.world_active,
                state.world_converged,
                state.world_failed,
                state.iteration_count,
                residuals.r_change,
                residuals.r_split,
                residuals.r_cross_iterate,
                residuals.r_unilateral,
                residuals.r_lagged,
                residuals.r_primal,
                residuals.r_dual,
                self._problem.data.box_rows.world_residual_max,
                self._problem.data.contact_rows.world_residual_max,
                self._problem.data.structural_rows.world_residual,
                self._problem.data.effort_rows.world_residual_max if self._problem.has_bounded_effort else None,
                self._data.status,
            ],
            device=self._device,
        )

    ###
    # Internals - Time Step
    ###

    def _begin_time_step(self) -> None:
        """Load the constraint rows of the time step with the configured stabilization and warm start."""
        constraints = self._constraints
        self._problem.begin_time_step(
            self._model.time.dt,
            limit_stabilization_fraction=constraints.beta,
            contact_stabilization_fraction=constraints.gamma,
            contact_dead_zone=constraints.delta,
            structural_warmstart_factor=self._config.joint_warmstart_factor,
        )

    def _update_system(self) -> None:
        """Assemble and factorize the weighted smooth system, then prepare the projections."""
        problem = self._problem
        problem.build_smooth_system(
            self._model.time.dt,
            joint_penalty_scale=self._data.joint_penalty_scale,
        )
        problem.build_consensus_weights(sigma=self._config.weight_sigma, beta=self._config.weight_beta)
        system = problem.system
        system.factorize()
        if system.dynamic_bodies:
            self._projection.prepare(system.data.inverse_weight, self._data.state.world_failed)

    def _restore_dual(self) -> None:
        """Convert the warm-start wrench to the scaled splitting dual with the current body weight."""
        state = self._data.state
        wp.launch(
            _restore_dual_from_wrench,
            dim=state.num_bodies,
            inputs=[
                self._model.bodies.wid,
                self._problem.data.body_unilateral_count,
                self._model.time.dt,
                self._problem.system.data.inverse_weight,
                self._data.dual_wrench,
            ],
            outputs=[state.splitting_dual],
            device=self._device,
        )

    ###
    # Internals - Iterations
    ###

    def _iterate(self, conditional: bool) -> None:
        """Perform one splitting iteration."""
        # Solve the smooth system for the candidate twist
        self._update_candidate()

        # Project the candidate onto the unilateral constraints
        self._update_projection()

        # Update the splitting dual, the structural multipliers, and the effort counters
        self._update_multipliers()

        # Check convergence
        self._update_convergence_status(conditional)

    def _update_candidate(self) -> None:
        """Solve the smooth system for the candidate twist."""
        problem = self._problem
        system = self._problem.system
        splitting = self._data.state
        system.build_candidate_right_hand_side(
            splitting.projected_twist,
            splitting.splitting_dual,
            problem.data.dynamic_rows,
            problem.data.effort_rows,
            problem.data.structural_rows,
        )
        system.solve_candidate()
        wp.launch(
            _prepare_projection,
            dim=splitting.num_bodies,
            inputs=[
                self._model.bodies.wid,
                splitting.world_active,
                system.data.body_vector_index,
                system.data.packed_solution,
                problem.body_velocity_begin,
            ],
            outputs=[
                splitting.splitting_dual,
                splitting.global_twist_previous,
                splitting.global_twist,
                splitting.projected_twist_previous,
                splitting.projected_twist,
            ],
            device=self._device,
        )

    def _update_projection(self) -> None:
        """Project the candidate twist onto the unilateral constraints."""
        if not self._problem.system.dynamic_bodies:
            return
        self._projection.project(
            self._config.projection_iterations,
            self._config.projection_anderson_recycle,
            self._data.state.world_active,
            self._data.state.world_failed,
            self._model.bodies.wid,
            self._problem.system.data.inverse_weight,
            self._data.state.projected_twist,
        )

    def _update_multipliers(self) -> None:
        """Update the splitting dual, the structural multipliers, and the effort counters.

        With convergence checks, the body residual reduction advances the splitting dual.
        """
        config = self._config
        problem = self._problem
        splitting = self._data.state
        time_step = self._model.time.dt
        if config.fixed_iterations:
            wp.launch(
                _update_splitting_dual,
                dim=splitting.num_bodies,
                inputs=[
                    self._model.bodies.wid,
                    splitting.world_active,
                    splitting.world_failed,
                    splitting.global_twist,
                    splitting.projected_twist,
                ],
                outputs=[splitting.splitting_dual],
                device=self._device,
            )
        problem.update_structural_multipliers_from_twist(
            time_step,
            min(config.position_tolerance, config.rotation_tolerance),
            splitting.global_twist,
            splitting.projected_twist,
            splitting.world_active,
            splitting.world_failed,
            projected_fraction=config.joint_multiplier_projected_fraction,
        )
        if self._problem.has_bounded_effort:
            problem.update_effort_counters(
                time_step,
                config.velocity_tolerance,
                splitting.projected_twist,
                splitting.world_active,
            )

    def _update_convergence_status(self, conditional: bool) -> None:
        """Reduce the iteration residuals, then deactivate converged and failed worlds."""
        config = self._config
        problem = self._problem
        splitting = self._data.state
        residuals = self._data.residuals
        if config.fixed_iterations:
            wp.launch(
                _initialize_fixed_iteration,
                dim=splitting.num_worlds,
                inputs=[splitting.world_failed],
                outputs=[splitting.world_active, splitting.iteration_count],
                device=self._device,
            )
            return
        iteration_condition = self._iteration_condition if conditional else None
        wp.launch(
            _initialize_iteration_residuals,
            dim=splitting.num_worlds,
            inputs=[splitting.world_failed],
            outputs=[
                splitting.world_active,
                splitting.iteration_count,
                residuals.r_change,
                residuals.r_split,
                residuals.r_cross_iterate,
                residuals.r_unilateral,
                residuals.r_lagged,
                problem.data.box_rows.world_residual_max,
                problem.data.contact_rows.world_residual_max,
                iteration_condition,
            ],
            device=self._device,
        )
        self._update_residuals(splitting.world_active, residuals.r_unilateral, residuals.r_lagged)
        wp.launch(
            _reduce_body_residuals,
            dim=splitting.num_bodies,
            inputs=[
                self._model.time.dt,
                config.position_tolerance,
                config.rotation_tolerance,
                config.velocity_tolerance,
                self._model.bodies.wid,
                splitting.global_twist_previous,
                splitting.global_twist,
                splitting.projected_twist_previous,
                splitting.projected_twist,
                splitting.world_active,
            ],
            outputs=[
                splitting.splitting_dual,
                splitting.world_failed,
                residuals.r_change,
                residuals.r_split,
                residuals.r_cross_iterate,
            ],
            device=self._device,
        )
        wp.launch(
            _finalize_residual_iteration,
            dim=splitting.num_worlds,
            inputs=[
                residuals.r_change,
                residuals.r_split,
                residuals.r_cross_iterate,
                residuals.r_lagged,
                residuals.r_unilateral,
                splitting.iteration_count,
                splitting.world_failed,
                config.max_iterations,
            ],
            outputs=[
                problem.data.structural_rows.world_residual,
                problem.data.effort_rows.world_residual_max if self._problem.has_bounded_effort else None,
                splitting.world_active,
                splitting.world_converged,
                iteration_condition,
            ],
            device=self._device,
        )

    def _update_residuals(
        self,
        world_active: wp.array[wp.bool] | None = None,
        velocity_residual: wp.array[wp.float32] | None = None,
        lagged_residual: wp.array[wp.float32] | None = None,
    ) -> None:
        """Evaluate the unilateral velocities and natural-map residuals.

        Args:
            world_active: Worlds to evaluate; ``None`` evaluates the final residuals of
                the worlds that did not fail.
            velocity_residual: Per-world residual mapped back to velocity and relative to
                the velocity tolerance, reduced by the convergence check, or ``None``.
            lagged_residual: Per-world change of the unilateral row velocities over the
                last iteration, relative to the velocity tolerance, or ``None``.
        """
        splitting = self._data.state
        problem = self._problem
        if world_active is None:
            problem.data.box_rows.world_residual_max.zero_()
            problem.data.contact_rows.world_residual_max.zero_()
        inverse_velocity_tolerance = 1.0 / self._config.velocity_tolerance
        if problem.data.box_rows.capacity > 0:
            wp.launch(
                _compute_box_residuals,
                dim=problem.data.box_rows.capacity,
                inputs=[
                    world_active,
                    self._data.state.world_failed,
                    problem.data.box_rows.world,
                    problem.data.box_rows.local,
                    problem.data.box_rows.world_count,
                    problem.data.box_rows.body_a,
                    problem.data.box_rows.body_b,
                    problem.data.box_rows.jacobian_a,
                    problem.data.box_rows.jacobian_b,
                    problem.data.box_rows.bias,
                    problem.data.box_rows.lower,
                    problem.data.box_rows.upper,
                    problem.data.box_rows.reaction,
                    problem.data.box_rows.physical_delassus,
                    self._data.state.projected_twist,
                    splitting.global_twist,
                    splitting.projected_twist_previous,
                    inverse_velocity_tolerance,
                ],
                outputs=[
                    problem.data.box_rows.velocity,
                    problem.data.box_rows.world_residual_max,
                    velocity_residual,
                    lagged_residual,
                ],
                device=self._device,
            )
        rows = problem.data.contact_rows
        if rows.capacity > 0 and rows.frame is not None:
            thread_count = _world_thread_count(self._device, problem.data.num_worlds, rows.max_world_capacity)
            wp.launch(
                _compute_spatial_contact_residuals,
                dim=(problem.data.num_worlds, thread_count),
                inputs=[
                    thread_count,
                    world_active,
                    rows.world_offset,
                    rows.world_count,
                    rows.body_a,
                    rows.body_b,
                    rows.jacobian_a,
                    rows.jacobian_b,
                    rows.reaction,
                    rows.frame,
                    rows.friction,
                    rows.angular_friction,
                    rows.bias,
                    rows.normal_compliance,
                    rows.restitution,
                    rows.angular_reaction,
                    rows.physical_delassus,
                    self._data.state.projected_twist,
                    splitting.global_twist,
                    splitting.projected_twist_previous,
                    inverse_velocity_tolerance,
                ],
                outputs=[
                    self._data.state.world_failed,
                    rows.velocity,
                    rows.world_residual_max,
                    velocity_residual,
                    lagged_residual,
                ],
                device=self._device,
            )
        elif rows.capacity > 0:
            thread_count = _world_thread_count(self._device, problem.data.num_worlds, rows.max_world_capacity)
            wp.launch(
                _compute_contact_residuals,
                dim=(problem.data.num_worlds, thread_count),
                inputs=[
                    thread_count,
                    world_active,
                    self._data.state.world_failed,
                    problem.data.contact_rows.world_offset,
                    problem.data.contact_rows.world_count,
                    problem.data.contact_rows.body_a,
                    problem.data.contact_rows.body_b,
                    problem.data.contact_rows.jacobian_a,
                    problem.data.contact_rows.jacobian_b,
                    problem.data.contact_rows.bias,
                    problem.data.contact_rows.friction,
                    problem.data.contact_rows.normal_compliance,
                    problem.data.contact_rows.restitution,
                    problem.data.contact_rows.reaction,
                    problem.data.contact_rows.physical_delassus,
                    self._data.state.projected_twist,
                    splitting.global_twist,
                    splitting.projected_twist_previous,
                    inverse_velocity_tolerance,
                ],
                outputs=[
                    problem.data.contact_rows.velocity,
                    problem.data.contact_rows.world_residual_max,
                    velocity_residual,
                    lagged_residual,
                ],
                device=self._device,
            )

    ###
    # Internals - Outputs
    ###

    def _store_dual_wrench(self) -> None:
        """Store the scaled splitting dual as the warm-start wrench of the next time step."""
        state = self._data.state
        wp.launch(
            _store_dual_wrench,
            dim=state.num_bodies,
            inputs=[
                self._model.bodies.wid,
                self._problem.data.body_unilateral_count,
                state.world_failed,
                self._model.time.inv_dt,
                self._problem.system.data.weight,
                state.splitting_dual,
            ],
            outputs=[self._data.dual_wrench],
            device=self._device,
        )

    def _write_final_status_and_residuals(self) -> None:
        """Store the terminal world status and the final residuals."""
        state = self._data.state
        residuals = self._data.residuals
        wp.launch(
            _reduce_final_body_residuals,
            dim=state.num_bodies,
            inputs=[
                self._model.bodies.wid,
                self._problem.system.data.weight,
                state.global_twist_previous,
                state.global_twist,
                state.projected_twist,
                state.splitting_dual,
            ],
            outputs=[state.world_failed, residuals.r_primal, residuals.r_dual],
            device=self._device,
        )
        # The convergence check evaluates the unilateral residuals at the final twists of
        # each world; fixed iterations skip that check and evaluate them once here.
        if self._config.fixed_iterations:
            self._update_residuals()
        wp.launch(
            _write_world_status,
            dim=self._model.info.num_worlds,
            inputs=[
                self._data.state.world_converged,
                state.world_failed,
                self._data.state.iteration_count,
                residuals.r_primal,
                residuals.r_dual,
                self._problem.data.box_rows.world_residual_max,
                self._problem.data.contact_rows.world_residual_max,
            ],
            outputs=[self._data.status],
            device=self._device,
        )

    def _write_accepted_dynamics(self) -> None:
        """Write outputs and expose the accepted LOX velocity as equivalent inputs to the selected Kamino integrator."""
        accepted_twist = self._data.state.projected_twist
        self._problem.write_output_wrenches(self._model.time.inv_dt, accepted_twist, self._data.state.world_failed)
        model = self._model
        problem = self._problem
        wp.launch(
            _write_integrator_body_inputs,
            dim=model.size.sum_of_num_bodies,
            inputs=[
                self._problem.system.data.body_vector_index,
                model.bodies.wid,
                self._data.state.world_failed,
                model.time.dt,
                model.bodies.m_i,
                problem.kamino_data.bodies.I_i,
                model.bodies.inv_m_i,
                model.bodies.inv_i_I_i,
                model.gravity.vector,
                problem.body_velocity_begin,
                accepted_twist,
            ],
            outputs=[problem.kamino_data.bodies.w_i, problem.kamino_data.bodies.u_i],
            device=self._device,
        )
