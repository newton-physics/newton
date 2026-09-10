# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Frozen-contact orchestration for the LOX rigid-body solve."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np
import warp as wp

from ......sim import ModelFlags
from ...core.bodies import update_body_wrenches
from ...core.state import StateKamino
from ...core.types import vec6f
from .adapter import _write_integrator_body_inputs
from .colored_gauss_seidel import ColoredGaussSeidelProjection
from .jacobi import project_constraints_jacobi
from .joint_delassus import BatchedStructuralDelassus
from .problem import LOXProblem
from .projection import (
    PROJECTION_STATUS_VALID,
    compute_projection_residuals,
    prepare_jacobi_projection_data,
    prepare_physical_projection_data,
)

if TYPE_CHECKING:
    from ....config import ConstraintStabilizationConfig, LOXSolverConfig
    from ...core.data import DataKamino
    from ...core.joints import JointCorrectionMode
    from ...core.model import ModelKamino
    from ...dynamics.dual import DualProblem
    from ...geometry.contacts import ContactsKamino
    from ...kinematics.jacobians import SparseSystemJacobians
    from ...kinematics.limits import LimitsKamino
    from ..metrics import SolutionMetricsData

__all__ = [
    "LOX_STATUS_ACTIVE",
    "LOX_STATUS_CONVERGED",
    "LOX_STATUS_FAILED",
    "LOX_STATUS_ITERATION_LIMIT",
    "LOXSolver",
    "LOXStatus",
]

LOX_STATUS_ACTIVE = 0
"""The world is active in the current splitting solve."""

LOX_STATUS_CONVERGED = 1
"""The world met the configured splitting tolerances."""

LOX_STATUS_FAILED = 2
"""A unilateral projection failed for the world."""

LOX_STATUS_ITERATION_LIMIT = 3
"""The world reached the configured splitting iteration limit."""

wp.set_module_options({"enable_backward": False})

_JOINT_PENALTY_SEED_PERCENTILE = 0.02


@wp.struct
class LOXStatus:
    """Per-world LOX terminal status."""

    converged: wp.int32
    """Whether the world met every configured LOX stopping tolerance."""
    iterations: wp.int32
    """Number of splitting iterations performed for the world."""
    r_p: wp.float32
    """NCP primal feasibility residual, or NaN when solution metrics are disabled."""
    r_d: wp.float32
    """NCP dual feasibility residual, or NaN when solution metrics are disabled."""
    r_c: wp.float32
    """NCP complementarity residual, or NaN when solution metrics are disabled."""
    accepted: wp.int32
    """Whether LOX accepted the world's final projected iterate."""
    failed: wp.int32
    """Whether a projection failed for the world."""
    iteration_limit: wp.int32
    """Whether the world stopped at the configured iteration limit."""


def _low_mode_eigenvalue(eigenvalues: np.ndarray) -> float:
    """Select the discrete low-mode percentile used for ALM scale seeding."""
    index = int(_JOINT_PENALTY_SEED_PERCENTILE * eigenvalues.size)
    return float(eigenvalues[index])


@wp.kernel
def _finalize_world_status(
    world_converged: wp.array[wp.bool],
    world_failed: wp.array[wp.bool],
    world_iteration_limit: wp.array[wp.bool],
    iteration_count: wp.array[wp.int32],
    world_accepted: wp.array[wp.bool],
    world_status: wp.array[wp.int32],
    solver_status: wp.array[LOXStatus],
):
    world = wp.tid()
    converged = world_converged[world] and not world_failed[world]
    accepted = not world_failed[world]
    world_accepted[world] = accepted
    if world_failed[world]:
        world_status[world] = LOX_STATUS_FAILED
    elif converged:
        world_status[world] = LOX_STATUS_CONVERGED
    elif world_iteration_limit[world]:
        world_status[world] = LOX_STATUS_ITERATION_LIMIT
    else:
        world_status[world] = LOX_STATUS_ACTIVE

    status = LOXStatus()
    status.converged = wp.int32(converged)
    status.iterations = iteration_count[world]
    status.r_p = wp.nan
    status.r_d = wp.nan
    status.r_c = wp.nan
    status.accepted = wp.int32(accepted)
    status.failed = wp.int32(world_failed[world])
    status.iteration_limit = wp.int32(world_iteration_limit[world])
    solver_status[world] = status


@wp.kernel
def _reset_solver_status(solver_status: wp.array[LOXStatus]):
    world = wp.tid()
    status = LOXStatus()
    status.converged = wp.int32(0)
    status.iterations = wp.int32(0)
    status.r_p = wp.nan
    status.r_d = wp.nan
    status.r_c = wp.nan
    status.accepted = wp.int32(0)
    status.failed = wp.int32(0)
    status.iteration_limit = wp.int32(0)
    solver_status[world] = status


@wp.kernel
def _update_solver_status_metrics(
    r_p: wp.array[wp.float32],
    r_d: wp.array[wp.float32],
    r_c: wp.array[wp.float32],
    solver_status: wp.array[LOXStatus],
):
    world = wp.tid()
    status = solver_status[world]
    status.r_p = r_p[world]
    status.r_d = r_d[world]
    status.r_c = r_c[world]
    solver_status[world] = status


@wp.kernel
def _reset_solver_worlds_masked(
    world_mask: wp.array[wp.bool],
    world_active: wp.array[wp.bool],
    world_converged: wp.array[wp.bool],
    world_failed: wp.array[wp.bool],
    world_iteration_limit: wp.array[wp.bool],
    iteration_count: wp.array[wp.int32],
    world_accepted: wp.array[wp.bool],
    world_status: wp.array[wp.int32],
    solver_status: wp.array[LOXStatus],
    contact_residual_max: wp.array[wp.float32],
    limit_residual_max: wp.array[wp.float32],
    friction_residual_max: wp.array[wp.float32],
):
    world = wp.tid()
    if world_mask[world]:
        world_active[world] = True
        world_converged[world] = False
        world_failed[world] = False
        world_iteration_limit[world] = False
        iteration_count[world] = 0
        world_accepted[world] = False
        world_status[world] = LOX_STATUS_ACTIVE
        status = LOXStatus()
        status.converged = wp.int32(0)
        status.iterations = wp.int32(0)
        status.r_p = wp.nan
        status.r_d = wp.nan
        status.r_c = wp.nan
        status.accepted = wp.int32(0)
        status.failed = wp.int32(0)
        status.iteration_limit = wp.int32(0)
        solver_status[world] = status
        contact_residual_max[world] = 0.0
        limit_residual_max[world] = 0.0
        friction_residual_max[world] = 0.0


@wp.kernel
def _update_iteration_condition(
    max_iterations: wp.int32,
    world_active: wp.array[wp.bool],
    iteration_count: wp.array[wp.int32],
    condition: wp.array[wp.int32],
):
    world = wp.tid()
    if world_active[world] and iteration_count[world] < max_iterations:
        wp.atomic_max(condition, 0, 1)


@wp.kernel
def _mark_iteration_limit(
    world_active: wp.array[wp.bool],
    world_iteration_limit: wp.array[wp.bool],
):
    world = wp.tid()
    if world_active[world]:
        world_active[world] = False
        world_iteration_limit[world] = True


@wp.kernel
def _initialize_body_velocity_guess(
    body_block: wp.array[wp.int32],
    body_world: wp.array[wp.int32],
    inverse_mass: wp.array[wp.float32],
    inverse_inertia_world: wp.array[wp.mat33f],
    velocity_start: wp.array[vec6f],
    external_wrench: wp.array[wp.spatial_vectorf],
    gravity: wp.array[wp.vec3f],
    time_step: wp.array[wp.float32],
    fraction: float,
    velocity_guess: wp.array[vec6f],
):
    body = wp.tid()
    dt = time_step[body_world[body]]
    guess = velocity_start[body]
    if body_block[body] >= 0 and fraction > 0.0:
        wrench = external_wrench[body]
        inv_mass = inverse_mass[body]
        linear_acceleration = inv_mass * wp.vec3f(wrench[0], wrench[1], wrench[2])
        if inv_mass > 0.0:
            linear_acceleration += gravity[body_world[body]]
        angular_acceleration = inverse_inertia_world[body] @ wp.vec3f(wrench[3], wrench[4], wrench[5])
        for axis in range(3):
            guess[axis] += fraction * dt * linear_acceleration[axis]
            guess[axis + 3] += fraction * dt * angular_acceleration[axis]
    velocity_guess[body] = guess


class LOXSolver:
    """Run a fixed LOX splitting solve on one frozen linearization.

    The caller owns collision detection, Jacobian construction, pose
    integration, and nonlinear relinearization. This class owns the smooth
    system update, weight construction, blocked dense LLT solve, unilateral
    projections, convergence freezing, and output conversion for one such
    linearization. A nonfailed world that reaches the iteration limit is
    accepted using its last projected iterate.
    """

    def __init__(
        self,
        model: ModelKamino,
        data: DataKamino,
        jacobians: SparseSystemJacobians,
        limits: LimitsKamino | None,
        contacts: ContactsKamino | None,
        config: LOXSolverConfig,
        constraints: ConstraintStabilizationConfig,
        rotation_correction: JointCorrectionMode,
    ):
        problem = None
        if model.size.sum_of_num_bodies > 0:
            problem = LOXProblem(
                model=model,
                data=data,
                jacobians=jacobians,
                limits=limits,
                contacts=contacts,
                eliminate_fixed_world_islands=config.eliminate_fixed_world_islands,
                projection_method=config.projection_method,
                rotation_correction=rotation_correction,
                joint_proximal_relaxation=config.joint_proximal_relaxation,
            )
        if problem is None:
            raise ValueError("LOX requires at least one rigid body.")
        self._config = config
        self.model = model
        self._constraints = constraints
        self.problem = problem
        self.device = model.device
        self.num_worlds = model.info.num_worlds
        self.max_iterations = config.max_iterations
        self.use_graph_conditionals = config.use_graph_conditionals
        self.fixed_iterations = config.fixed_iterations
        self.projection_iterations = config.projection_iterations
        self.projection_method = config.projection_method
        self.gauss_seidel_max_colors = config.gauss_seidel_max_colors
        self._colored_gauss_seidel = None
        self.inertial_warmstart_fraction = config.inertial_warmstart_fraction
        self.position_tolerance = config.position_tolerance
        self.rotation_tolerance = config.rotation_tolerance
        self.velocity_tolerance = config.velocity_tolerance
        self.weight_sigma = config.weight_sigma
        self.weight_beta = config.weight_beta
        self.joint_penalty_scale = wp.full(
            self.num_worlds, config.joint_penalty_scale, dtype=wp.float32, device=self.device
        )
        self.joint_multiplier_projected_fraction = config.joint_multiplier_projected_fraction
        self.joint_warmstart_factor = config.joint_warmstart_factor
        self._bind_rigid_topology()
        self.world_accepted = wp.zeros(self.num_worlds, dtype=wp.bool, device=self.device)
        self.world_status = wp.zeros(self.num_worlds, dtype=wp.int32, device=self.device)
        self.status = wp.empty(self.num_worlds, dtype=LOXStatus, device=self.device)
        wp.launch(_reset_solver_status, dim=self.num_worlds, outputs=[self.status], device=self.device)
        self._iteration_condition = wp.zeros(1, dtype=wp.int32, device=self.device)
        self._projection_theta = (
            wp.ones(self.num_worlds, dtype=wp.float32, device=self.device)
            if config.projection_method == "apgd"
            else None
        )
        self._projection_beta = (
            wp.zeros(self.num_worlds, dtype=wp.float32, device=self.device)
            if config.projection_method == "apgd"
            else None
        )
        self._projection_restart_dot = (
            wp.zeros(self.num_worlds, dtype=wp.float32, device=self.device)
            if config.projection_method == "apgd"
            else None
        )
        self.has_bounded_effort = problem is not None and problem.has_bounded_effort
        self._time_step: wp.array[wp.float32] | None = None
        self._inverse_time_step: wp.array[wp.float32] | None = None
        self._time_step_prepared = False

    def _bind_rigid_topology(self) -> None:
        """Bind solver views and scratch arrays to the current rigid topology."""
        problem = self.problem
        self.system = problem.system if problem is not None else None
        self._has_dynamic_rigid_bodies = self.system is not None and bool(self.system.dynamic_bodies)
        if self.system is not None:
            self.system.selective_body_weights = self._config.selective_weights
        self.splitting = problem.splitting if problem is not None else None
        self.projected_twist = (
            self.splitting.projected_twist
            if self.splitting is not None
            else wp.empty(0, dtype=vec6f, device=self.device)
        )
        self.world_active = (
            self.splitting.world_active
            if self.splitting is not None
            else wp.ones(self.num_worlds, dtype=wp.bool, device=self.device)
        )
        self.world_converged = (
            self.splitting.world_converged
            if self.splitting is not None
            else wp.zeros(self.num_worlds, dtype=wp.bool, device=self.device)
        )
        self.world_failed = (
            self.splitting.world_failed
            if self.splitting is not None
            else wp.zeros(self.num_worlds, dtype=wp.bool, device=self.device)
        )
        self.world_iteration_limit = (
            self.splitting.world_iteration_limit
            if self.splitting is not None
            else wp.zeros(self.num_worlds, dtype=wp.bool, device=self.device)
        )
        self.iteration_count = (
            self.splitting.iteration_count
            if self.splitting is not None
            else wp.zeros(self.num_worlds, dtype=wp.int32, device=self.device)
        )
        self.contact_residual_max = (
            problem.world_contact_residual_max
            if problem is not None
            else wp.zeros(self.num_worlds, dtype=wp.float32, device=self.device)
        )
        self.limit_residual_max = (
            problem.world_limit_residual_max
            if problem is not None
            else wp.zeros(self.num_worlds, dtype=wp.float32, device=self.device)
        )
        self.friction_residual_max = (
            problem.world_friction_residual_max
            if problem is not None
            else wp.zeros(self.num_worlds, dtype=wp.float32, device=self.device)
        )
        rigid_body_count = problem.model.size.sum_of_num_bodies if problem is not None else 0
        self._initial_twist = wp.zeros(rigid_body_count, dtype=vec6f, device=self.device)

    def _make_structural_delassus(self) -> BatchedStructuralDelassus:
        problem = self.problem
        return BatchedStructuralDelassus(
            body_system=self.system,
            body_first_global=problem.structural_body_first_global,
            body_second_global=problem.structural_body_second_global,
            jacobian_first=problem.structural_jacobian_first,
            jacobian_second=problem.structural_jacobian_second,
        )

    def rebuild_rigid_topology(self) -> None:
        """Rebuild rigid allocations after dynamic-body classification changes."""
        problem = self.problem
        if problem is None:
            return
        problem.rebuild_dynamic_body_topology()
        self._colored_gauss_seidel = None
        self._bind_rigid_topology()
        self.has_bounded_effort = problem.has_bounded_effort
        self.reset()

    def joint_penalty_scale_seed(
        self,
        time_step: float,
    ) -> list[float]:
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
        it is intended for initialization rather than the captured simulation
        loop.

        Args:
            time_step: Uniform simulation time step [s].

        Returns:
            The estimated dimensionless structural ALM penalty scale for each
            world.
        """
        problem = self.problem
        if problem is None:
            raise ValueError("joint penalty scale seeding requires a LOX rigid-body system.")
        problem.prepare_joint_penalty_scale_seed(time_step)
        time_step_array = problem.model.time.dt
        inverse_time_step = problem.model.time.inv_dt
        if problem.structural_row_count == 0:
            return self.joint_penalty_scale.numpy().tolist()

        self.reset()
        problem.begin_time_step(
            time_step_array,
            inverse_time_step,
            limit_stabilization_fraction=self._constraints.beta,
            contact_stabilization_fraction=self._constraints.gamma,
            contact_dead_zone=self._constraints.delta,
            impact_velocity_threshold=self._config.impact_velocity_threshold,
            contact_recoverable_response=self._config.contact_recoverable_response,
        )
        try:
            problem.body_linearization_twist.zero_()
            problem.update(
                time_step_array,
                joint_penalty_scale=1.0,
                linearization_twist=problem.body_linearization_twist,
                assemble_structural_penalty=False,
            )
            wp.copy(self.system.weighted_matrix, self.system.smooth_matrix)
            self.system.factorize()

            structural_delassus = self._make_structural_delassus()
            structural_delassus.assemble()
            matrix_values = structural_delassus.matrix.numpy()
            matrix_offsets = structural_delassus.info.mio.numpy()
            vector_offsets = structural_delassus.info.vio.numpy()
            vector_rows = structural_delassus.vector_row.numpy()
            effective_mass = problem.structural_effective_mass.numpy()
            matrix_epsilon = np.finfo(matrix_values.dtype).eps

            seeds = self.joint_penalty_scale.numpy().astype(np.float64)
            positive_by_world: list[list[np.ndarray]] = [[] for _ in range(problem.num_worlds)]
            for component, row_count in enumerate(structural_delassus.component_row_counts):
                if row_count == 0:
                    continue
                matrix_offset = int(matrix_offsets[component])
                vector_offset = int(vector_offsets[component])
                rows = vector_rows[vector_offset : vector_offset + row_count]
                row_metric = effective_mass[rows].astype(np.float64)
                metric_sqrt = np.sqrt(row_metric)
                matrix = (
                    matrix_values[matrix_offset : matrix_offset + row_count * row_count]
                    .reshape(row_count, row_count)
                    .astype(np.float64)
                )
                normalized = metric_sqrt[:, None] * matrix * metric_sqrt[None, :]
                eigenvalues = np.linalg.eigvalsh(normalized)
                positive_threshold = matrix_epsilon * row_count * float(eigenvalues[-1])
                positive = eigenvalues[eigenvalues > positive_threshold]
                if positive.size == 0:
                    world = structural_delassus.component_world_host[component]
                    raise RuntimeError(
                        f"World {world} component {component} structural Delassus has no resolvable positive eigenvalue."
                    )
                positive_by_world[structural_delassus.component_world_host[component]].append(positive)

            for world, component_eigenvalues in enumerate(positive_by_world):
                if not component_eigenvalues:
                    continue
                positive = np.sort(np.concatenate(component_eigenvalues))
                seed = 1.0 / _low_mode_eigenvalue(positive)
                if not math.isfinite(seed) or seed > np.finfo(np.float32).max:
                    raise RuntimeError(f"World {world} structural ALM penalty seed is not representable in float32.")
                seeds[world] = seed

            applied_seeds = seeds.astype(np.float32)
            self.joint_penalty_scale.assign(applied_seeds)
            return applied_seeds.tolist()
        finally:
            self.reset()

    def solve_forward_dynamics(
        self,
        state_in: StateKamino,
        state_out: StateKamino,
        _contacts: object | None = None,
    ) -> None:
        """Solve one LOX forward-dynamics step from prepared Kamino data."""
        if self.splitting is not None:
            wp.copy(self.splitting.splitting_dual_impulse, state_in.lambda_w_i.view(dtype=vec6f))
        constraints = self._constraints
        time_step = self.model.time.dt
        inverse_time_step = self.model.time.inv_dt
        problem = self.problem
        self.begin_time_step(
            time_step,
            inverse_time_step,
            limit_stabilization_fraction=constraints.beta,
            contact_stabilization_fraction=constraints.gamma,
            contact_dead_zone=constraints.delta,
            impact_velocity_threshold=self._config.impact_velocity_threshold,
            contact_recoverable_response=self._config.contact_recoverable_response,
        )

        linearization_twist = None
        if problem is not None:
            problem.body_linearization_twist.zero_()
            linearization_twist = problem.body_linearization_twist
        self.solve(linearization_twist=linearization_twist, write_output=False)
        if self.splitting is not None:
            wp.copy(state_out.lambda_w_i.view(dtype=vec6f), self.splitting.splitting_dual_impulse)

        if problem is not None:
            problem.write_outputs(time_step, inverse_time_step, write_body_velocity=False)
            model = problem.model
            data = problem.data
            update_body_wrenches(model.bodies, data.bodies)
            # Expose the accepted LOX velocity as equivalent inputs to the
            # selected Kamino integrator.
            wp.launch(
                _write_integrator_body_inputs,
                dim=model.size.sum_of_num_bodies,
                inputs=[
                    problem.system.body_vector_index,
                    model.bodies.wid,
                    self.world_accepted,
                    time_step,
                    model.bodies.m_i,
                    data.bodies.I_i,
                    model.bodies.inv_m_i,
                    model.bodies.inv_i_I_i,
                    model.gravity.vector,
                    problem.body_velocity_begin,
                    self.projected_twist,
                ],
                outputs=[data.bodies.w_i, data.bodies.u_i],
                device=model.device,
            )

    def notify_model_changed(self, flags: ModelFlags | int) -> None:
        """Refresh or rebuild the LOX problem after model changes."""
        problem = self.problem
        if problem is None:
            return
        if flags & ModelFlags.BODY_PROPERTIES and problem.dynamic_body_topology_changed():
            self.rebuild_rigid_topology()

    def validate_model_changed(self) -> None:
        """Validate LOX-specific values derived from the Newton model."""
        # Host-side validation cannot synchronize aliased Newton arrays while a
        # CUDA graph is being captured. The same topology was validated when the
        # solver was built; captured property updates retain that topology.
        if self.device.is_cuda and self.device.is_capturing:
            return
        if self.problem is not None:
            self.problem.validate_model_changed()

    def reset(
        self,
        problem: DualProblem | None = None,
        world_mask: wp.array[wp.bool] | None = None,
    ) -> None:
        """Clear structural/splitting warm starts and per-world diagnostics."""
        del problem
        if world_mask is not None:
            if world_mask.shape != (self.num_worlds,) or world_mask.dtype != wp.bool:
                raise ValueError(f"world_mask must have shape ({self.num_worlds},) and dtype bool.")
            if world_mask.device != self.device:
                raise ValueError(f"world_mask must be allocated on {self.device}, found {world_mask.device}.")
        if self.splitting is not None:
            self.splitting.reset(world_mask=world_mask)
            self.problem.reset_structural_multipliers(world_mask=world_mask)
            if self.has_bounded_effort:
                self.problem.reset_effort_counters(world_mask=world_mask)
            self.problem.projection_status.zero_()
            self.problem.contact_residual.zero_()
            self.problem.limit_residual.zero_()
            self.problem.friction_residual.zero_()
            self.problem.reset_friction_reactions(world_mask=world_mask)
        elif world_mask is None:
            self.world_active.fill_(True)
            self.world_converged.zero_()
            self.world_failed.zero_()
            self.world_iteration_limit.zero_()
            self.iteration_count.zero_()
        if world_mask is None:
            self.world_accepted.zero_()
            self.world_status.fill_(LOX_STATUS_ACTIVE)
            wp.launch(_reset_solver_status, dim=self.num_worlds, outputs=[self.status], device=self.device)
            self.contact_residual_max.zero_()
            self.limit_residual_max.zero_()
            self.friction_residual_max.zero_()
        else:
            wp.launch(
                _reset_solver_worlds_masked,
                dim=self.num_worlds,
                inputs=[world_mask],
                outputs=[
                    self.world_active,
                    self.world_converged,
                    self.world_failed,
                    self.world_iteration_limit,
                    self.iteration_count,
                    self.world_accepted,
                    self.world_status,
                    self.status,
                    self.contact_residual_max,
                    self.limit_residual_max,
                    self.friction_residual_max,
                ],
                device=self.device,
            )
        self._initial_twist.zero_()
        self._time_step_prepared = False

    def update_status_metrics(self, metrics: SolutionMetricsData) -> None:
        """Populate terminal NCP residuals from the optional metrics evaluation."""
        wp.launch(
            _update_solver_status_metrics,
            dim=self.num_worlds,
            inputs=[metrics.r_ncp_primal, metrics.r_ncp_dual, metrics.r_ncp_compl],
            outputs=[self.status],
            device=self.device,
        )

    def begin_time_step(
        self,
        time_step: wp.array[wp.float32],
        inverse_time_step: wp.array[wp.float32],
        *,
        limit_stabilization_fraction: float,
        contact_stabilization_fraction: float,
        contact_dead_zone: float,
        impact_velocity_threshold: float,
        contact_recoverable_response: bool,
        reset_dual: bool = False,
    ) -> None:
        """Freeze inertial velocities and import damped constraint warm starts.

        The body consensus warm start is loaded from the Newton input state as
        the generalized impulse ``W u`` and converted back to the scaled dual
        after the solve builds its current body weight.
        """
        self._time_step = time_step
        self._inverse_time_step = inverse_time_step
        if self.problem is not None:
            self.problem.scale_structural_multipliers(self.joint_warmstart_factor)
            self.problem.begin_time_step(
                time_step,
                inverse_time_step,
                limit_stabilization_fraction=limit_stabilization_fraction,
                contact_stabilization_fraction=contact_stabilization_fraction,
                contact_dead_zone=contact_dead_zone,
                impact_velocity_threshold=impact_velocity_threshold,
                contact_recoverable_response=contact_recoverable_response,
            )
            wp.launch(
                _initialize_body_velocity_guess,
                dim=self.problem.model.size.sum_of_num_bodies,
                inputs=[
                    self.system.body_block,
                    self.problem.model.bodies.wid,
                    self.problem.model.bodies.inv_m_i,
                    self.problem.data.bodies.inv_I_i,
                    self.problem.body_velocity_begin,
                    self.problem.data.bodies.w_e_i,
                    self.problem.model.gravity.vector,
                    time_step,
                    self.inertial_warmstart_fraction,
                ],
                outputs=[self._initial_twist],
                device=self.device,
            )
            if reset_dual:
                self.splitting.splitting_dual.zero_()
                self.splitting.splitting_dual_impulse.zero_()
        self._time_step_prepared = True

    def _prepare_colored_gauss_seidel_projection(self) -> None:
        projection_problem = self.problem
        if self._colored_gauss_seidel is None or self._colored_gauss_seidel.problem is not projection_problem:
            self._colored_gauss_seidel = ColoredGaussSeidelProjection(
                projection_problem,
                self.gauss_seidel_max_colors,
            )
        prepared_status = self.problem.world_jacobi_projection_status
        inverse_weight = self.system.inverse_weight
        self._colored_gauss_seidel.prepare(inverse_weight, prepared_status)

    def _prepare_body_space_projection(self) -> None:
        problem = self.problem
        if not self._has_dynamic_rigid_bodies:
            problem.projection_status.fill_(PROJECTION_STATUS_VALID)
            return
        prepare_physical_projection_data(
            problem.friction_world,
            problem.friction_local,
            problem.world_friction_count,
            problem.friction_body_first,
            problem.friction_body_second,
            problem.friction_jacobian_first,
            problem.friction_jacobian_second,
            problem.contact_world,
            problem.contact_local,
            problem.world_contact_count,
            problem.contact_body_first,
            problem.contact_body_second,
            problem.contact_jacobian_first,
            problem.contact_jacobian_second,
            problem.contact_bias,
            problem.contact_friction,
            problem.limit_world,
            problem.limit_local,
            problem.world_limit_count,
            problem.limit_body_first,
            problem.limit_body_second,
            problem.limit_jacobian_first,
            problem.limit_jacobian_second,
            self.system.inverse_weight,
            problem.friction_physical_delassus,
            problem.contact_physical_delassus,
            problem.contact_prepared_delassus,
            problem.limit_physical_delassus,
            problem.world_physical_projection_status,
        )
        if self.projection_method in ("jacobi", "apgd") or (
            self.projection_method == "gauss_seidel" and self.gauss_seidel_max_colors == 1
        ):
            prepare_jacobi_projection_data(
                problem.friction_world,
                problem.friction_local,
                problem.world_friction_count,
                problem.friction_body_first,
                problem.friction_body_second,
                problem.friction_jacobian_first,
                problem.friction_jacobian_second,
                problem.contact_world,
                problem.contact_local,
                problem.world_contact_count,
                problem.contact_body_first,
                problem.contact_body_second,
                problem.contact_jacobian_first,
                problem.contact_jacobian_second,
                problem.contact_bias,
                problem.contact_friction,
                problem.limit_world,
                problem.limit_local,
                problem.world_limit_count,
                problem.limit_body_first,
                problem.limit_body_second,
                problem.limit_jacobian_first,
                problem.limit_jacobian_second,
                problem.body_constraint_count,
                problem.static_body_constraint_count,
                self.system.inverse_weight,
                problem.friction_projection_delassus,
                problem.contact_projection_delassus,
                problem.limit_projection_delassus,
                problem.world_jacobi_projection_status,
            )
        elif self.projection_method == "gauss_seidel" and self.gauss_seidel_max_colors > 1:
            self._prepare_colored_gauss_seidel_projection()

    def _project_body_space_constraints(self) -> None:
        if not self._has_dynamic_rigid_bodies:
            return
        problem = self.problem
        splitting = self.splitting
        if self.projection_method in ("jacobi", "apgd") or (
            self.projection_method == "gauss_seidel" and self.gauss_seidel_max_colors == 1
        ):
            project_constraints_jacobi(
                self.projection_iterations,
                splitting.world_active,
                splitting.body_world,
                problem.friction_world,
                problem.friction_local,
                problem.world_friction_count,
                problem.friction_body_first,
                problem.friction_body_second,
                problem.friction_jacobian_first,
                problem.friction_jacobian_second,
                problem.friction_impulse_bound,
                problem.friction_projection_delassus,
                problem.contact_world,
                problem.contact_local,
                problem.world_contact_count,
                problem.contact_body_first,
                problem.contact_body_second,
                problem.contact_jacobian_first,
                problem.contact_jacobian_second,
                problem.contact_bias,
                problem.contact_friction,
                problem.contact_projection_delassus,
                problem.limit_world,
                problem.limit_local,
                problem.world_limit_count,
                problem.limit_body_first,
                problem.limit_body_second,
                problem.limit_jacobian_first,
                problem.limit_jacobian_second,
                problem.limit_bias,
                problem.limit_projection_delassus,
                self.system.inverse_weight,
                splitting.projected_twist,
                problem.projection_twist_delta,
                problem.contact_reaction,
                problem.limit_reaction,
                problem.friction_reaction,
                problem.world_jacobi_projection_status,
                problem.projection_status,
                world_body_offset=problem.model.info.bodies_offset,
                world_body_count=problem.model.info.num_bodies,
                world_friction_offset=problem.world_friction_offset,
                world_contact_offset=problem.world_contact_offset,
                world_limit_offset=problem.world_limit_offset,
                accelerated=self.projection_method == "apgd",
                theta=self._projection_theta,
                beta=self._projection_beta,
                restart_dot=self._projection_restart_dot,
                friction_trial=problem.friction_acceleration_trial,
                friction_previous=problem.friction_acceleration_previous,
                contact_trial=problem.contact_acceleration_trial,
                contact_previous=problem.contact_acceleration_previous,
                limit_trial=problem.limit_acceleration_trial,
                limit_previous=problem.limit_acceleration_previous,
            )
        elif self.projection_method == "gauss_seidel" and self.gauss_seidel_max_colors > 1:
            self._colored_gauss_seidel.project(
                self.projection_iterations,
                splitting.world_active,
                splitting.body_world,
                self.system.inverse_weight,
                splitting.projected_twist,
                problem.projection_twist_delta,
                problem.world_jacobi_projection_status,
                problem.projection_status,
            )

    def _update_conditional_iteration(self) -> None:
        self._iteration_condition.zero_()
        wp.launch(
            _update_iteration_condition,
            dim=self.num_worlds,
            inputs=[
                self.max_iterations,
                self.world_active,
                self.iteration_count,
            ],
            outputs=[self._iteration_condition],
            device=self.device,
        )

    def _prepare_body_space_candidate(self) -> None:
        """Solve and prepare the rigid candidate without unilateral projection."""
        problem = self.problem
        system = self.system
        splitting = self.splitting
        if self.has_bounded_effort:
            problem.promote_effort_counters(splitting.world_active)
            system.build_candidate_right_hand_side_with_effort(
                splitting.projected_twist,
                splitting.splitting_dual,
                problem.body_effort_offset,
                problem.body_effort_index,
                problem.body_effort_side,
                problem.effort_dynamic_row_index,
                problem.dynamic_jacobian_first,
                problem.dynamic_jacobian_second,
                problem.effort_counter_applied,
            )
        else:
            system.build_candidate_right_hand_side(splitting.projected_twist, splitting.splitting_dual)
        system.solve_candidate(problem.body_velocity_begin)
        splitting.prepare_projection(system.body_solution)

    def _finish_body_space_iteration(
        self, time_step: wp.array[wp.float32], linearization_twist: wp.array[vec6f]
    ) -> None:
        """Update structural state and finish the rigid splitting iteration."""
        problem = self.problem
        splitting = self.splitting
        problem.update_structural_multipliers_from_twist(
            time_step,
            min(self.position_tolerance, self.rotation_tolerance),
            linearization_twist,
            splitting.global_twist,
            splitting.projected_twist,
            splitting.world_active,
            projected_fraction=self.joint_multiplier_projected_fraction,
        )
        if self.has_bounded_effort:
            problem.update_effort_counters(
                time_step,
                self.velocity_tolerance,
                splitting.projected_twist,
                splitting.world_active,
            )
        if not self.fixed_iterations:
            problem.evaluate_lagged_velocity_consistency(
                self.velocity_tolerance,
                splitting.global_twist,
                splitting.projected_twist_previous,
                splitting.world_active,
            )
            splitting.finish_iteration(
                problem.projection_status,
                time_step,
                self.position_tolerance,
                self.rotation_tolerance,
                self.velocity_tolerance,
                effort_residual=(problem.world_effort_residual_max if self.has_bounded_effort else None),
                structural_residual=problem.world_structural_residual,
                projected_structural_residual=problem.world_projected_structural_residual,
                lagged_velocity_residual=problem.world_lagged_velocity_residual,
                lagged_velocity_required=problem.world_lagged_velocity_required,
            )
        else:
            splitting.finish_fixed_iteration(
                problem.projection_status,
            )

    def _body_space_iteration(
        self,
        time_step: wp.array[wp.float32],
        linearization_twist: wp.array[vec6f],
        conditional: bool,
    ) -> None:
        self._prepare_body_space_candidate()
        self._project_body_space_constraints()
        self._finish_body_space_iteration(time_step, linearization_twist)
        if conditional:
            self._update_conditional_iteration()

    def solve(
        self,
        *,
        initial_twist: wp.array[vec6f] | None = None,
        linearization_twist: wp.array[vec6f] | None = None,
        write_output: bool = True,
    ) -> None:
        """Assemble and solve one frozen-contact smooth linearization.

        Args:
            initial_twist: Optional initial projected body twist.
            linearization_twist: Optional body twist used to linearize joint constraints.
            write_output: Whether to write the solution to the Kamino containers.
        """
        if not self._time_step_prepared:
            raise RuntimeError("begin_time_step() must be called before solving LOX.")
        time_step = self._time_step
        inverse_time_step = self._inverse_time_step
        if time_step is None or inverse_time_step is None:
            raise RuntimeError("LOX per-world timestep arrays are not prepared.")
        if self.splitting is not None:
            if initial_twist is None:
                initial_twist = self._initial_twist
            self.splitting.begin(initial_twist, reset_dual=False)
        self.world_accepted.zero_()
        self.world_status.fill_(LOX_STATUS_ACTIVE)
        if self.problem is not None:
            self.problem.contact_residual.zero_()
            self.problem.limit_residual.zero_()
            self.problem.friction_residual.zero_()
        self.contact_residual_max.zero_()
        self.limit_residual_max.zero_()
        self.friction_residual_max.zero_()
        problem = self.problem
        system = self.system
        splitting = self.splitting
        if problem is not None:
            problem.update(
                time_step,
                joint_penalty_scale=self.joint_penalty_scale,
                linearization_twist=linearization_twist,
                assemble_structural_penalty=True,
            )
            if linearization_twist is None:
                linearization_twist = problem.body_linearization_twist
            system.build_weighted_matrix(
                body_has_unilateral=problem.body_has_unilateral,
                sigma=self.weight_sigma,
                beta=self.weight_beta,
            )
            splitting.restore_dual_from_impulse(system.inverse_weight, problem.body_has_unilateral)
            system.factorize()
            self._prepare_body_space_projection()

        use_conditional_loop = (
            not self.fixed_iterations
            and self.use_graph_conditionals
            and (not self.device.is_cuda or not self.device.is_capturing or wp.is_conditional_graph_supported())
        )
        if use_conditional_loop:
            self._iteration_condition.fill_(1)
            wp.capture_while(
                self._iteration_condition,
                self._body_space_iteration,
                time_step=time_step,
                linearization_twist=linearization_twist,
                conditional=True,
            )
        else:
            for _iteration in range(self.max_iterations):
                self._body_space_iteration(time_step, linearization_twist, conditional=False)

        if splitting is not None:
            splitting.store_dual_impulse(system.weight, problem.body_has_unilateral)
            splitting.mark_iteration_limit()
        else:
            wp.launch(
                _mark_iteration_limit,
                dim=self.num_worlds,
                inputs=[],
                outputs=[self.world_active, self.world_iteration_limit],
                device=self.device,
            )
        wp.launch(
            _finalize_world_status,
            dim=self.num_worlds,
            inputs=[self.world_converged, self.world_failed, self.world_iteration_limit, self.iteration_count],
            outputs=[self.world_accepted, self.world_status, self.status],
            device=self.device,
        )
        if problem is not None:
            compute_projection_residuals(
                self.world_accepted,
                problem.projection_status,
                problem.friction_world,
                problem.friction_local,
                problem.world_friction_count,
                problem.friction_body_first,
                problem.friction_body_second,
                problem.friction_jacobian_first,
                problem.friction_jacobian_second,
                problem.friction_impulse_bound,
                problem.friction_reaction,
                problem.friction_physical_delassus,
                problem.contact_world,
                problem.contact_local,
                problem.world_contact_count,
                problem.contact_body_first,
                problem.contact_body_second,
                problem.contact_jacobian_first,
                problem.contact_jacobian_second,
                problem.contact_bias,
                problem.contact_friction,
                problem.contact_reaction,
                problem.contact_physical_delassus,
                problem.limit_world,
                problem.limit_local,
                problem.world_limit_count,
                problem.limit_body_first,
                problem.limit_body_second,
                problem.limit_jacobian_first,
                problem.limit_jacobian_second,
                problem.limit_bias,
                problem.limit_reaction,
                problem.limit_physical_delassus,
                splitting.projected_twist,
                problem.friction_velocity,
                problem.contact_velocity,
                problem.limit_velocity,
                problem.contact_residual,
                problem.limit_residual,
                problem.friction_residual,
                problem.world_contact_residual_max,
                problem.world_limit_residual_max,
                problem.world_friction_residual_max,
            )
            if write_output:
                problem.write_outputs(time_step, inverse_time_step, body_velocity=splitting.projected_twist)
